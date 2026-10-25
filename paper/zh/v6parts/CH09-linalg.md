## 九、Linalg：把迭代空间和计算体留在 IR 里

线性代数方言（Linalg）的 `linalg.generic` 把一项计算拆成可独立检查的四部分：迭代空间（iteration space）、每个操作数的下标映射（indexing maps）、各维的迭代器类型（iterator types），以及 region 中的标量计算体。转置、广播和归约因此可以直接从 IR 读出，无需先展开成嵌套循环再反推数学结构。

阅读 generic 可以先不进入 region。映射左侧所有不同变量构成迭代坐标，操作数形状与映射共同约束每个坐标的范围；随后检查输出映射能否唯一确定每个写位置，最后才看标量体怎样组合这些元素。`linalg.matmul`、`linalg.fill` 等命名 op 将常用合同预封装，generic 则把映射与计算体完整展开。两种形式处在同一结构化抽象层，下降前都能参与 tiling、fusion 等依赖这些信息的变换。

`parallel` 与 `reduction` 是交给编译器的依赖类别，不会自行创建 CPU 线程或 GPU 并行单元。前端需要保证 region 的副作用与声明一致；若 region 捕获其他缓冲并产生跨迭代写冲突，仅写上 `parallel` 不能让程序获得正确语义。映射也只描述逻辑元素访问，memref 类型中的 strided 布局继续负责物理地址计算，两层信息不能互相替代。

当前静态形状还允许直接推出迭代范围：64×64 输出约束 `i、j` 各有 64 点，64×16 输入与 `(i,r)`、`(j,r)` 共同约束 `r` 有 16 点。遇到动态形状时，相应范围来自运行时维度及 op 的合法域，不能从 `?` 猜一个常数。下标映射、形状和布局三方一致后，后续手算或下降才有明确基础。

### 1. 用三条映射读出 KKT

`ins` 给出只读操作数，`outs` 给出目标操作数，两者按先输入后输出的顺序各自对应一条仿射映射（affine map）。映射左侧是迭代坐标，右侧是该操作数的下标。目标传递风格（destination-passing style）意味着计算写到调用者提供的目标中：memref 形式原地更新已有缓冲，tensor 形式则以 `outs` 的值作为目标初值并产生新的 tensor 结果。region 的块参数依次是当前迭代点读到的输入标量和目标标量，`linalg.yield` 给出要写回目标的标量。

`outs` 同时承担初值来源与写目标两种角色，但 region 是否读取初值要看输出块参数的实际使用。累加 op 使用它，完全覆盖的逐元素 op 可以忽略它。分析 generic 时还应核对映射数与操作数总数、块参数数与标量操作数总数是否一致；任一错位都会让后续数学解读失去依据。多个输入可以共享同一个 SSA 值，输出也可以与输入别名，合法性取决于每个迭代点的读写关系，不能只看操作数列表是否出现重名。

反向 L1 先清零 `%kkt`，再计算 chunk 内键向量的 Gram 矩阵：

```mlir
linalg.fill ins(%f0 : f32) outs(%kkt : memref<64x64xf32>)
linalg.generic {
  indexing_maps = [
    affine_map<(i, j, r) -> (i, r)>,
    affine_map<(i, j, r) -> (j, r)>,
    affine_map<(i, j, r) -> (i, j)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%k_c, %k_c :
      memref<64x16xf32, strided<[32, 1], offset: ?>>,
      memref<64x16xf32, strided<[32, 1], offset: ?>>)
  outs(%kkt : memref<64x64xf32>) {
^bb0(%a0: f32, %a1: f32, %o0: f32):
  %p = arith.mulf %a0, %a1 : f32
  %s2 = arith.addf %o0, %p : f32
  linalg.yield %s2 : f32
}
```

[evidence/G1/l1_bwd_chunked_T128_C64.mlir:2524-2537]

读代码时把三条映射逐一作用到坐标 `(i,j,r)`。第一条读取 `k_c[i,r]`，提供左因子第 `i` 行、归约维第 `r` 个分量。第二条读取 `k_c[j,r]`，等价于把第二个逻辑操作数看成 `k_cᵀ[r,j]`；存储中的矩阵没有转置，访问坐标已经表达了 `k·kᵀ`。第三条定位 `kkt[i,j]`，右侧没有 `r`，所以同一输出元素会接收全部 `r` 迭代点的更新。两个输入虽然都是 `%k_c`，不同映射仍使它们承担矩阵乘中的不同角色。

`parallel` 表明不同的 `(i,j)` 输出点可以独立计算，`reduction` 标明 `r` 是归约坐标。归约怎样组合仍由 region 决定：`%o0` 是该输出位置已有的累加值，`%p` 是当前乘积，`addf` 与 `yield` 明确指定加法归约。把 region 改为只 yield `%p`，同样的映射与迭代器就不会再表达求和。这里的初值来自目标缓冲，前面的 `linalg.fill` 因而属于数值定义；省去清零会把旧内容加入结果。整个 op 对应：

```text
KKT[i,j] = Σ(r=0..15) k_c[i,r] × k_c[j,r]
```

用更小的两行输入可验证坐标读法。设第 0 行为 `(1,2)`，第 1 行为 `(3,4)`，在 `i=0、j=1` 时，两条输入映射依次给出 `(1,3)` 与 `(2,4)`，目标映射始终指向 `KKT[0,1]`，region 因而累加 `1×3+2×4=11`。若把第二条误写成 `(r,j)`，同一坐标代入就会读到不同元素，说明第二个逻辑操作数的转置方向已经改变；在原来的 64×16 输入上还会与维度范围冲突。若把第三条也写成 `(i,r)`，输出位置会随归约坐标移动，无法表达固定 `(i,j)` 的内积。逐坐标代入比背下 matmul 模板更容易发现这类错误。

归约也不等于固定使用加法。相同的三条映射可以配合求最大值或带权累积的 region，迭代空间保持不变，数值语义却随 yield 前的计算改变。因此审查一个 generic 时，映射回答“在哪里取、写到哪里”，region 回答“取得标量后怎样组合”，两部分必须一起与公式对账。

{{FIG10}}

### 2. 广播与 A 矩阵的位置谓词

广播（broadcasting）也来自映射。若某个迭代坐标没有出现在一条映射右侧，沿该坐标移动时会重复读取同一元素。A 矩阵的 generic 同时使用矩阵、按行广播的向量、按列广播的向量和另一个行尺度：

```mlir
linalg.generic {
  indexing_maps = [
    affine_map<(i, j) -> (i, j)>,
    affine_map<(i, j) -> (i)>,
    affine_map<(i, j) -> (j)>,
    affine_map<(i, j) -> (i)>,
    affine_map<(i, j) -> (i, j)>
  ],
  iterator_types = ["parallel", "parallel"]
} ins(%kkt, %gc, %gc, %beta_c :
      memref<64x64xf32>, memref<64xf32>, memref<64xf32>,
      memref<64xf32, strided<[2], offset: ?>>)
  outs(%Abuf : memref<64x64xf32>) {
^bb0(%a0: f32, %a1: f32, %a2: f32, %a3: f32, %o0: f32):
  %d0 = arith.subf %a1, %a2 : f32
  %e0 = math.exp2 %d0 : f32
  %m0 = arith.mulf %a0, %a3 : f32
  %m1 = arith.mulf %m0, %e0 : f32
  %i0 = linalg.index 0 : index
  %j0 = linalg.index 1 : index
  %lt = arith.cmpi slt, %j0, %i0 : index
  %r0 = arith.select %lt, %m1, %f0 : f32
  linalg.yield %r0 : f32
}
```

[evidence/G1/l1_bwd_chunked_T128_C64.mlir:2538-2558]

五条映射依次读取 `kkt[i,j]`、`gc[i]`、`gc[j]`、`beta[i]`，并写入 `A[i,j]`。`gc[i]` 不含 `j`，所以同一行内复用一个值；`gc[j]` 不含 `i`，所以按列复用。两个 `parallel` 维与 region 中不读 `%o0` 的事实相互印证：每个输出元素只由当前坐标计算并完整覆盖，目标旧值不参与结果。

`linalg.index 0/1` 取出当前坐标，`cmpi slt` 形成 `j<i` 的严格下三角（strict lower triangular）谓词，`strict_tril` 就由这个谓词和随后的 `arith.select` 直接表达。于是 region 计算了 `beta_i × KKT[i,j] × exp2(gc_i-gc_j)`，并在对角线及上三角写零。这里的 `%gc` 是内部 log2 域的包含式前缀和；外部语义仍是 `g≤0`、衰减倍率 `exp(g)`，生成器先计算 `g_log2=g×log2(e)`，所以 IR 中的 `math.exp2` 与外部自然指数一致 [tools/gdn_l1_gen.py:16-24]。

### 3. A 用 slt，P 用 sle

后续重算中的 P 包含对角衰减项。它先把 `q·kᵀ` 写入 `%Abuf`，再在同一个缓冲上施加衰减和含对角掩蔽（mask）：

```mlir
linalg.generic {
  indexing_maps = [
    affine_map<(i, j) -> (i, j)>,
    affine_map<(i, j) -> (i)>,
    affine_map<(i, j) -> (j)>,
    affine_map<(i, j) -> (i, j)>
  ],
  iterator_types = ["parallel", "parallel"]
} ins(%Abuf, %gc, %gc : memref<64x64xf32>, memref<64xf32>, memref<64xf32>)
  outs(%Abuf : memref<64x64xf32>) {
^bb0(%a0: f32, %a1: f32, %a2: f32, %o0: f32):
  %d3 = arith.subf %a1, %a2 : f32
  %e4 = math.exp2 %d3 : f32
  %m4 = arith.mulf %a0, %e4 : f32
  %i3 = linalg.index 0 : index
  %j3 = linalg.index 1 : index
  %le = arith.cmpi sle, %j3, %i3 : index
  %r5 = arith.select %le, %m4, %f0 : f32
  linalg.yield %r5 : f32
}
```

[evidence/G1/l1_bwd_chunked_T128_C64.mlir:2835-2852；前置 `q·kᵀ` 在 2821-2834 行]

这里要读 `sle` 与前面的 `slt`：`j≤i` 保留对角，`j<i` 排除对角，谓词差异就是 A 与 P 的数学边界。`ins` 和 `outs` 使用同一 `%Abuf`，每个 `(i,j)` 只读写自己的元素，因此可以原地更新。条件直接进入比较和选择，没有分配、填充再读取一个 0/1 mask 矩阵。`arith.select` 只选择值；它之前的 `exp2` 与乘法仍可能对被掩蔽的位置执行，不能把省去 mask 物化理解成省去这些位置的算术。

对角线给出了最短的数值检查。A 在 `i=j` 时条件为假，输出必为零；P 在同一位置条件为真，且 `gc_i-gc_i=0`，衰减因子为 1，所以对角保留 `q_i·k_i`。对 `j>i`，两者都写零；对 `j<i`，两者都保留各自已有的矩阵值并施加衰减。用这四类位置核对输出，比只比较矩阵整体范数更容易定位谓词写反的错误。

原地形式还多出一项审查义务：确认后续仍需要旧 `%Abuf` 的读取者已经完成，并确认单次 generic 内不存在跨输出点的依赖。此处的逐点映射满足这个条件；换成同时读取邻近元素的 region 后，同样的 ins/outs 别名可能改变结果。

### 4. 【练习】一个小 tensor generic

下面的 tensor 形式计算 `y[i,j]=x[i,j]+b[j]`。`%init` 提供目标初值，`%0` 是新的 tensor 值，原来的 `%init` 仍可供其他使用读取。

```mlir
func.func @add_bias(%x: tensor<2x3xf32>, %b: tensor<3xf32>,
                    %init: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (j)>,
      affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%x, %b : tensor<2x3xf32>, tensor<3xf32>)
    outs(%init : tensor<2x3xf32>) {
  ^bb0(%xv: f32, %bv: f32, %ov: f32):
    %sum = arith.addf %xv, %bv : f32
    linalg.yield %sum : f32
  } -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
```

先指出哪条映射产生广播，以及每个 `b[j]` 被几个迭代点读取；再把 region 改为 `linalg.yield %ov`，判断 `%0` 与 `%init` 的关系。检查后解释为何 memref 形式的同一 generic 没有 `-> tensor` 结果。

`convert-linalg-to-loops` 把这些迭代坐标实体化后，KKT 的加法归约会显式进入循环。第十章的 `scf.for`、`iter_args` 与 `scf.yield` 将直接呈现这种跨迭代数据流。
