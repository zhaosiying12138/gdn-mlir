## 十四、把配置做成属性：GdnConfigAttr

一个算子家族通常包含许多合法变体：分块多大、门控走标量还是逐通道、要不要初始状态。这些选择有三种安放方式——做成不同的 op 名、做成额外操作数、或做成 op 的属性。gdn 方言选择第三种，定义了参数化属性 GdnConfigAttr：op 名与图结构保持稳定（永远只有 gdn.fwd 与 gdn.bwd 两个节点），变体选择作为编译期常量挂在 op 上，参与打印、比较与缓存键。

### 1. 直觉：配置为什么放在属性里

操作数是 SSA 值，参与数据流分析，运行时才能确定取值；属性是编译期常量，在 IR 构建时就固化为驻留对象。chunk_size 这类选择在进入编译器的那一刻已经确定，之后所有层都需要读取它来决定展开形态——这正是属性的定义域。放进操作数会污染数据流图（一个整数要伪装成张量流过图），拆成多个 op 名则让 pattern 匹配与打印都退化为枚举地狱。属性还有两个工程便利：MLIR 对属性做全局驻留与按值比较，同一份配置在 IR 任何位置都是同一对象，可直接充当哈希键；属性的打印由 assemblyFormat 声明式给出，逐字节确定。

### 2. AttrDef：参数化属性的定义

GdnConfigAttr 用 AttrDef 后端定义，这是现代 MLIR 推荐的写法（旧写法是 StructAttr 包一袋命名属性）：

```
def GdnConfigAttr : AttrDef<Gdn_Dialect, "GdnConfig"> {
  let mnemonic = "config";
  let summary = "GDN algorithm knobs";
  let description = [{
    chunk_size ∈ {16,32,64}; use_g (scalar log2 decay); use_gk (per-channel);
    normalize_q; use_initial_state; store_final_state; is_varlen.
  }];
  let parameters = (
    ins
      "int":$chunk_size,
      "bool":$use_g,
      "bool":$use_gk,
      "bool":$normalize_q,
      "bool":$use_initial_state,
      "bool":$store_final_state,
      "bool":$is_varlen
  );
  let assemblyFormat = "`<` $chunk_size `,` $use_g `,` $use_gk `,` $normalize_q `,` $use_initial_state `,` $store_final_state `,` $is_varlen `>`";
}
```
［dialect/include/Gdn/GdnDialect.td:50-68，节选］

语法要素逐个说明。AttrDef 的第一个模板参数是所属方言，第二个是 C++ 类名前缀，mnemonic 决定文本形式的前缀 #gdn.config。parameters 是一个 dag，形式与 op 的 arguments 相同：每项是 C++ 类型加参数名，生成的类为每个参数给出强类型访问器——getChunkSize() 返回整数、getUseG() 返回布尔，类型错误在编译期暴露。这一点相对 StructAttr 的优势是实质的：StructAttr 生成的是匿名字典式访问，字段名拼错要等运行期断言；AttrDef 参数表则把七个字段的拼写固化进 C++ 接口。assemblyFormat 用反引号包住字面标点、$名字 引用参数，声明了尖括号内逗号分隔的固定打印序——于是 IR 里出现 #gdn.config<64, true, false, true, true, true, false>，七个值的顺序永远与参数表一致，diff 两份配置只需对齐文本。

### 3. 七个旋钮各自改变什么数学形态

旋钮的权威记载在语义文档与方言声明里：

| 旋钮 | 改变的数学形态 | G1 对拍取值 |
|---|---|---|
| chunk_size | chunk 化的块长，限定 16、32、64 三档 | 32/64 已对拍 |
| use_g | 门控衰减取每头一个标量 | true |
| use_gk | 门控衰减取每通道各一份 | false |
| normalize_q | 算子内对 q 做 L2 归一 | false |
| use_initial_state | 初态 h0 参与递推 | true（含 h0 为零用例） |
| store_final_state | 终态 ht 作为输出物化 | true |
| is_varlen | 批内序列变长处理 | false |

［docs/L0_SEMANTICS.md:31-39 旋钮表；合法域 {16,32,64} 另见 Gdn.h:32 与第十三章 verifier］

逐项说清"数学形态"的含义。chunk_size 是 chunk 化递推的块长 C：一个 chunk 内 C 步递推联立成的三角线性系统是 C 阶的，跨 chunk 状态递推的步数随 T/C 变化——它改变的是 L1 算法结构的粒度，verifier 把它限定在三档合法值内。use_g 与 use_gk 是一对互斥选择：门控衰减既可取每头一个标量，也可取每通道各一份。衰减在 chunk 化数学中以两个显式因子进入——状态上的衰减折叠与键行上的行衰减（第三章的 S' 递推与 KKT 权重），旋钮决定这些因子的通道形状：标量门在通道维共享一份，逐通道门则每个通道独立衰减，KKT 权重与跨 chunk 折合相应从标量乘变成逐通道乘。normalize_q 决定 q 是否在算子内做 L2 归一：归一开启时按 K 的负二分之一次方缩放作用于归一后的 q（GdnOps.td 的 op 描述原文如此）；语义规约同时记载默认关闭、由模型侧负责归一。use_initial_state 与 store_final_state 是一对边界开关：前者决定初态 h0 是否作为递推起点参与，后者决定终态 ht 是否作为结果物化；反向算子中 dh0 的输出以 h0 槽的存在为条件（GdnOps.td 对 gdn.bwd 的描述：dh0 仅在给定 h0 时输出）。is_varlen 切换批内序列变长的处理方式，语义对拍阶段固定关闭。七个旋钮的共同点：任何一个取值的变化都会改变算子的输出数值，因此都属于语义层。

### 4. 两层旋钮的分离：属性与 transform 工件

编译器里还有另一族旋钮——调度旋钮：数据类型、mma 精度、tile 三维、warps 数、流水深度、warp 专化、TMA 共十维。它们的定义性区别在于：改变语义旋钮会改变计算结果，改变调度旋钮只改变性能与资源占用，数值结果在容差合同内不变。这决定了两者必须放在不同的载体、归属不同的层。

载体上，语义旋钮是 GdnConfigAttr，挂在 L0 op 上、随 IR 的值流一起被打印与改写；调度旋钮是 transform dialect 的参数化工件，L2 层每份内核对应一份工件文本加一组旋钮取值（第十三章总览的 transform.mlir 工件即其形态），旋钮值由求解器注入工件的参数槽。两层各自的验证合同不同：语义层的任何改动要回到 JAX fp64 规约做三方交叉；调度层的改动走 ptxas 静态资源复核与逐配置 fp64 参考对照。分层带来直接的工程收益——Z3 求解器在调度空间做全枚举（kkt 阶段 368 个合法配置 0.29 秒穷尽），枚举中语义冻结不动，无需每试一个 tile 就重跑语义对拍；反过来，语义旋钮变化的组合（chunk_size 三档乘以各布尔档位）只需要在 L1 层完成一次展开与对拍。

若把两层混进同一袋旋钮，代价是复合的：枚举空间把语义维也卷进排序问题，而排序对语义维没有意义（语义没有优劣，只有取值）；每个调度候选都要全量数值重验，验证成本乘上空间规模。分属两层之后，"哪些改动必须触发语义重验"有了机械判据——看改动落在属性还是工件。

### 5. 透传链：从 L0 op 到 L1 消费方

属性定义好后还要在层间搬运。第十五章将展开的 gdn-to-chunk pass 里，搬运是一行代码：

```
    // 语义属性原样透传（L1 消费方按此选择算法变体）
    call->setAttr("gdn.config", op.getConfig());
    rewriter.replaceOp(op, call.getResults());
```
［dialect/lib/GdnPasses.cpp:54-56；getConfig() 即 AttrDef 生成的访问器］

改写后的真实 IR 显示属性落在 call 节点上：

```
  func.func private @__gdn_l1_fwd_chunk(tensor<2x128x2x16xf32>, ..., tensor<2x2x16x16xf32>)
      -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
  ...
    %0:2 = call @__gdn_l1_fwd_chunk(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
        {gdn.config = #gdn.config<64, true, false, true, true, true, false>}
        : (...) -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
```
［build-dialect/bin/gdn-opt --gdn-to-chunk 的真实 stdout，类型串有删节标记；输入为 dialect/test/smoke_l0.mlir］

链条读法：L0 的 gdn.fwd 携带 config 声明一次；pass 把 op 替换为对 L1 私有入口的调用，并把同一个属性对象原样挂到 call 上；L1 的消费方（chunk 展开的选择逻辑）读取 call 上的属性决定算法变体。全程只有一个事实源，中间层不重述、不翻译配置。属性驻留机制保证了"原样"的强度——setAttr 搬动的是同一驻留对象，逐字节比较必然相等。这条链在确定性上闭合：属性取值连同全部调度旋钮环境变量进入缓存键，两次完整编译逐字节一致，全部旋钮关断时产物与默认配置逐字节等价（27/27 payload，第九章第四节）——配置既可组合，也可作为证据存档与 diff。

### 6. 小结

GdnConfigAttr 展示了属性参数化的完整做法：AttrDef 声明参数表与打印格式，生成强类型访问器；七个旋钮各自对应算子数学的一处形态变化，由 verifier 与语义文档双重把守；它与 transform 工件分属两层，使语义验证与调度枚举互不拖累；透传链保证配置从声明点到消费点单一事实源。下一章进入改写本身：第一个下降 pass 如何写成。
