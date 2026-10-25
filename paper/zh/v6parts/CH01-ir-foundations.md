## 一、一段 IR 如何成为程序：Operation、SSA 与区域

MLIR 的程序由操作（Operation）及其嵌套区域组成。文本中的百分号、花括号与类型注解共同描述数据怎样产生、在哪里可见、由谁消费；读懂这些结构之后，方言名称与具体算法才有可靠的附着点。

本书的两套编号先固定：L0 是 `gdn.fwd/bwd` 的语义入口，L1 是可执行算法结构，L2 是调度，L3 是机器指令与流水，L4 是 NVVM、PTX 与 cubin；G0–G6 是证据与复现门，和 L 层不是同一套编号。

### 1. 一个 `gdn.fwd` 操作包含什么

下面的冒烟输入保留了一个前向算子的完整通用形式。

```mlir
func.func @smoke(%q: tensor<2x128x2x16xf32>,
                 %k: tensor<2x128x2x16xf32>,
                 %v: tensor<2x128x2x16xf32>,
                 %beta: tensor<2x128x2xf32>,
                 %g: tensor<2x128x2xf32>,
                 %h0: tensor<2x2x16x16xf32>)
    -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>) {
  %o, %ht = "gdn.fwd"(%q, %k, %v, %beta, %g, %h0)
      <{config = #gdn.config<64, true, false, true, true, true, false>}>
      : (tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>,
         tensor<2x128x2xf32>, tensor<2x128x2xf32>, tensor<2x2x16x16xf32>)
      -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
  return %o, %ht : tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>
}
```

[dialect/test/smoke_l0.mlir:2-14]

`"gdn.fwd"` 是操作名，圆括号中的六个 SSA 值是操作数（operands），`%o` 与 `%ht` 是结果（results）。尖括号内的属性字典携带编译期已知的配置（attributes）；冒号后的函数类型依次列出操作数类型与结果类型。`gdn.fwd` 自身没有区域（regions），它所在的花括号属于外层 `func.func` 的函数体区域。操作数、结果、属性与区域回答不同问题，不能用一串无类型的文本互相替代。

这段采用通用形式（generic form）：名字加引号，操作数、属性和类型完整显式给出。美化形式（pretty form）由操作自定义的解析器、打印器或 `assemblyFormat` 提供，可省略部分标点与类型。两种形式可以构造同样的 `Operation`；`gdn.fwd` 当前未声明专用操作格式，所以冒烟文件使用通用形式。同文件中的 `func.func`、`return` 以及后文的 `linalg.generic` 都采用 pretty 形式，两种格式可以在一份 IR 中混用；第二章还会看到 hatch 操作的自定义格式。

阅读打印结果时，还要把 SSA 值的身份与显示名字分开。`%o` 只是便于阅读的符号，重新编号后程序仍可等价；多结果操作在自动打印时常呈现为 `%0:2`，单个结果再以 `%0#0`、`%0#1` 引用。操作本身不是 SSA 值，只有它声明的 results 才进入 use-def 图。属性也没有结果编号，因此不能让 `config` 像 `%q` 一样流向另一个操作。

类型注解提供独立于名字的检查依据。这里的张量维数全部静态写出，其他程序也可能用 `?` 表示动态维；解析器保留这些类型事实，操作 verifier 再判断该操作能否接受它们。张量的元素类型、秩和形状属于类型，序列用哪组算法配置属于属性，实际张量内容则要在执行时才确定。

### 2. SSA 名称连接定义与使用

静态单赋值（SSA，Static Single Assignment）约束一个值只有一个定义点，却不限制它被使用多少次。`%q` 等六个名字来自函数入口块参数（block arguments），并非某个操作的 results；`%o`、`%ht` 各自由 `gdn.fwd` 定义一次，随后被 `return` 使用。沿“定义到使用”建立的边称为 use-def 链。编译器沿它寻找生产者、消费者与可替换的边，改写时无需猜测同名列的赋值历史。

块参数也承担控制流合流处的值传递。分支把实际值交给目标块，目标块以参数接收，因此 MLIR 无需把 phi 表示成一种特殊操作。值具有类型，SSA 还提供作用域与支配信息；名字唯一、类型相同和控制流上可用是三项独立条件。

### 3. 区域与块让操作成为程序树

`gdn.fwd` 的函数体已经展示一层区域。真实 L1 IR 中的 `linalg.generic` 再展示一层：操作可以在属性与操作数之外拥有自己的计算体。

```mlir
linalg.fill ins(%f0 : f32) outs(%kkt : memref<64x64xf32>)
linalg.generic {
      // indexing_maps 与 iterator_types 在此只作区域属性，详见第九章
    } ins(%k_c, %k_c) outs(%kkt) {
  ^bb0(%a: f32, %b2: f32, %acc: f32):
  %p = arith.mulf %a, %b2 : f32
  %s2 = arith.addf %acc, %p : f32
  linalg.yield %s2 : f32
}
```

[evidence/G1/l1_chunked_T128_C64.mlir:1925-1938；indexing_maps、memref 类型有省略]

这里先只读区域结构，不要求提前掌握 linalg 的映射或 memref 的布局。`linalg.generic` 的区域含入口块 `^bb0`，该块又含乘、加与 `linalg.yield`；`%a`、`%b2`、`%acc` 是由 `linalg.generic` 的迭代语义绑定的块参数，`%p` 与 `%s2` 则有各自的定义操作。`yield` 把区域末次迭代的值交给父操作，不能换成函数级 `return`。映射、迭代器和内存类型分别在第九章与第十一章展开。

区域中的值默认不能越过定义它的区域向外引用；向内捕获外层值则受层级支配以及父操作的隔离 Trait 约束。区域的具体执行含义由所属操作定义，花括号本身不提供循环或并行语义。

普通 SSACFG 区域的块以终结操作结束：函数体常见 `return`，结构化区域常见 `yield`，多块控制流还可使用分支。`linalg.yield` 与 `scf.for` 的关系也由父操作规定，不能把任意终结操作拼到任意区域末尾。遍历 IR 时，仅扫描顶层函数体会漏掉这些嵌套操作；递归 walk 才能看到区域内部的乘法和加法。

use-def 边与区域树可以交叉。一个外层 SSA 值可能被深层区域使用，一个区域结果则可由父操作暴露给外层；内层普通临时值不能直接逃逸。分析改写时要先确定值定义所在的层级，再检查使用点是否处于它的作用域和支配范围内。

{{FIG9}}

概念图应对应上面两段真实 IR：树的一侧展开模块、函数区域、操作、嵌套区域与块，边的另一侧连接 `%q` 到 `gdn.fwd`、`%o` 到 `return`，以及 `^bb0` 中的乘法到加法。它能辅助区分“包含关系”与“值依赖”，这些边不能仅凭文本缩进混为一谈。

### 4. 支配决定一个定义能否服务某个使用

在普通控制流区域中，定义支配（dominates）使用，要求从区域入口到该使用的每条控制流路径都经过定义。同一块内通常表现为定义先于使用；跨块与跨区域时还要考虑所有路径和层级作用域。

【通用补充】下面两个函数给出最小对照。它们是新写示例，不属于项目测试。

```mlir
func.func @ok(%c: i1, %a: i32) -> i32 {
  cf.cond_br %c, ^then, ^else
^then:
  %t = arith.addi %a, %a : i32
  cf.br ^join(%t : i32)
^else:
  cf.br ^join(%a : i32)
^join(%x: i32):
  return %x : i32
}

func.func @bad(%c: i1, %a: i32) -> i32 {
  cf.cond_br %c, ^then, ^else
^then:
  %t = arith.addi %a, %a : i32
  cf.br ^join
^else:
  cf.br ^join
^join:
  return %t : i32
}
```

`@ok` 的两条前驱分别给 `^join` 传入值，入口参数 `%x` 在合流块中有定义，且每条路径都为其提供一个来源。`@bad` 的 `%t` 只在 `^then` 定义；走 `^else` 时合流块没有经过该定义，所以 `return %t` 违反支配。把 `%t` 写成块参数并补齐两个前驱的实参，修复的是控制流上的定义可用性；仅修改名称无法修复支配。

### 5. SSA 值唯一仍允许内存原地更新

张量值与内存描述符需要分开理解。这里只借 L1 的状态说明 SSA 与内存是两件事；memref 的布局、subview 和 bufferize 在第十一章再展开。`memref` 类型的 SSA 值指向一块存储，该值可以只有一个定义，而存储内容能被后续操作修改。L1 的真实状态递推提供了直接证据。

```mlir
%S = memref.alloc() : memref<16x16xf32>
// 中间的 h0h 是 h0 的 subview；此处略去其长类型声明
linalg.copy ins(%h0h : memref<16x16xf32, strided<[16, 1], offset: ?>>) outs(%S : memref<16x16xf32>)
// 中间的 chunk 计算略去
linalg.generic {
      // indexing_maps 与 iterator_types 略去
    } ins(%S, %prod : memref<16x16xf32>, memref<16x16xf32>) outs(%S : memref<16x16xf32>) {
  ^bb0(%xs: f32, %xp: f32, %out8: f32):
  %ms = arith.mulf %xs, %dl : f32
  %rs = arith.addf %ms, %xp : f32
  linalg.yield %rs : f32
}
```

[evidence/G1/l1_chunked_T128_C64.mlir:1880、1897-1898、2115-2127；长属性有省略]

`%S` 只由 `memref.alloc` 定义一次，`linalg.copy` 已向它写入初态，后面的 `linalg.generic` 又同时把它列为输入与输出，实现原地状态更新。可见 SSA 约束的是值定义，内存写由具有副作用的操作表达。`subview` 还会产生新的 SSA 描述符并引用既有存储，因此不同 SSA 值也可能别名；是否可重排、可消除，必须查询内存效果与别名信息。

当前 L1 已是 memref 级、f32、原地状态的 CPU 标量 IR，`run_l1.sh` 中保留的 bufferize 对该输入没有实际 tensor 转换工作。反过来看高层的 `gdn.fwd`，它只给出签名、配置与语义入口，未在这段操作中包含已实现的 chunk 算法。当前 `gdn-to-chunk` 仅把它改为 L1 调用边界；可执行算法文本由 `tools/gdn_l1_gen.py` 生成，不能从一个高层操作名推断实现已经展开。
