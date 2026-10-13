## 一、总览：一个算子的编译之旅

全书只跟踪一个对象——方言算子 gdn.fwd——看它从一行语义声明出发，沿五层渐进式下降走到 sm_120a 架构的 cubin 文件。两个用词先约定：旋钮，指编译期可调的调度参数；尺子，指某一层裁决对错所用的验证工具。

GDN-MLIR 是 Gated DeltaNet（chunk_gated_delta_rule，线性注意力家族中带门控 delta rule 的算子）前向与反向算子的编译器：接受一个只声明语义的方言 op，经 L0（语义）、L1（算法结构）、L2（调度）、L3（张量核与流水）、L4（sm_120a 目标代码）五层下降，产出可直接装载的 cubin。实现全部建立在 stock MLIR（2026-10 LLVM 主干）之上，引入的自定义 NVVM op 为 1 个，预算上限 5 个。仓库的交付物是编译器本身：所有前向与反向内核都是同一条管线的产物；fla（Triton）与 FlashQLA（TileLang）两家手写实现的设计点，在编译器中被表达为调度层旋钮空间中的两个极端配置——复现一个参考实现的工作从重写内核变成调参。内核速度用于检验方法论的有效性。

### 1. 旅程的骨架：每层回答一个问题

渐进式下降的组织原则是：每层固定一类决策，把另一类决策留给下层作为旋钮，并由一套独立验证把关；下层改动破坏语义时，可回退上层口径重新对照。六站的问答与尺子总览如下。

| 站 | 回答的问题 | 尺子 |
|---|---|---|
| L0 语义 | 这个算子是什么 | 方言 verifier 合法域 |
| L1 算法结构 | 它等价于哪条程序 | JAX fp64 规约三方交叉 |
| L2 调度 | 这条程序在硬件上怎么铺 | ptxas 静态资源复核加排序质量门槛 |
| L3 机器 | 数据怎么流进张量核 | fp64 参考逐配置对照 |
| PTX 后处理 | stock 流水线遗留什么问题 | 位级等价与逐位复现 |
| L4 目标 | 最终代码如何定案 | 跨进程逐位复现 |

### 2. 第一站：L0，这个算子是什么

L0 只声明意图，展开选择全部留给下层。gdn.fwd 接收 q、k、v、β、g 与可选初始状态 h0，输出 o 与终态 ht；gdn.bwd 输出六个梯度 dq、dk、dv、dβ、dg、dh0。GdnConfigAttr 捆绑七个语义旋钮（chunk_size、use_g、use_gk、normalize_q、use_initial_state、store_final_state、is_varlen），它们改变算子的数学形态，与 L2 的调度旋钮分属两个层面。数值约定（衰减在先、β 作用于修正项、L2 归一化位置）固定于方言 verifier 与生成器——GDN 的数值约定密集，任何一条弄错都会产生形状正确而数值整体错误的静默偏移，因此语义只有一份。真实 IR（玩具形状）：

```
%o, %ht = "gdn.fwd"(%q, %k, %v, %beta, %g, %h0)
    <{config = #gdn.config<64, true, false, true, true, true, false>}>
    : (...) -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
```
［dialect/test/smoke_l0.mlir］

### 3. 第二站：L1，它等价于哪条程序

L1 把算子的算法结构写成真实的 linalg 加 scf MLIR：chunk-WY 分解（其数学内容在第三章展开）在 IR 里逐行可见，三角掩蔽是显式控制流，前代换是三重循环。这份 IR 可在 CPU 上经树内 pass 下降后由 mlir-runner 执行，因此它本身就是尺子的被测对象：与 JAX fp64 可执行规约、fla naive 参考三方交叉，前向 7/7 用例约 6e-7，反向六梯度最差 6.4e-6（验证门 2e-4），三方闭合。"数学即 IR"是这一层的验证基石——任何下层改动破坏语义，都能回退到这份 CPU 可执行的 IR 重新对照。

### 4. 第三站：L2，这条程序在硬件上怎么铺

L2 的核心机制是调度即 IR：每份内核是一个参数化 transform 工件加一组旋钮取值的实例化。旋钮空间十维（dtype、mma 精度、tile M/N/K、warps、流水深度、mma 开关、warp 专化、TMA），合法性由 Z3 枚举，排序由校准解析成本模型给出。一份真实工件的核心动作：

```
%allocs = transform.structured.match ops{["memref.alloc"]} in %arg0
transform.memref.multibuffer %allocs <factor = 2, skip_analysis>
%loops = transform.structured.match ops{["scf.for"]} in %arg0
transform.nvgpu.pipeline_shared_memory_copies failures(propagate)
    %loops depth = 2 peel_epilogue
```
［evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/transform.mlir:1-15，节选］

这份工件先对共享内存 tile 做双缓冲，再对循环做深度 2 的软件流水，数值 2 由求解器注入。调度因此是可 diff 的文本：复现一个配置等于复现一份工件。规模与质量：kkt 阶段 368 个合法配置 0.29 秒枚举完毕，o 阶段 465 个配置 0.33 秒；每新形状调度 0.8 秒纯 CPU；排序与实测的 Spearman 秩相关 0.8962（门槛 0.7）。机制展开在第四至十二章。

### 5. 第四站：L3，数据怎么流进张量核

L3 的机制按功能分五组：数值到指令的映射（split-bf16 与 mma 发射器族）、算法级改写（扫描的仿射携带形式）、数据通路（异步 staging、乒乓预取、warp 专化与 TMA）、存储几何（bank 冲突、垫行与 32 位宽存）、驻留资源（生存期别名与寄存器上限）。落到 PTX 后，异步搬运呈现为线程跨步认领的 16 字节块：

```
cp.async.ca.shared.global [%rd5],  [%rd549], 16;
cp.async.ca.shared.global [%rd14], [%rd549+16], 16;
cp.async.ca.shared.global [%rd15], [%rd549+32], 16;
```
［evidence/G4/gen_verify/kkt_bf16_mma_p0_t64x64x32_w4/kernel.ptx:606-610，节选］

尺子是 fp64 参考逐配置对照：20 个旋钮配置全部通过、双随机种子。机制细节在第三十至三十五章。

### 6. 最后两站：PTX 后处理与 L4 定案

MLIR 语义完整下译之后、ptxas 定案之前，stock 流水线遗留四个真实问题——静态共享内存、生存期别名的实现载体、寄存器上限、bank 几何。856 行文本改写器 gdn_smem_fix.py 承载全部四个变换；工程原则是每步改写要么逐位相同、要么回退到正确形态、失败时显式报告。L4 定案的命令链：gpu-lower-to-nvvm-pipeline 以 cubin-chip=sm_120a、cubin-features=+ptx87、cubin-format=assembly 产出 PTX，改写后由 ptxas-blackwell -v 编译为 cubin，SASS 由 nvdisasm 审计；全链的冒烟以树内 sm_80 transform-mma 集成测试完成——文件名沿用上游目录名，chip 重定向为 sm_120a 后在 5090 上编译运行（sm_120 的张量核即 Ampere 风格 mma.sync，同指令族故该测试有效，第二十四章展开）。这一段的尺子是位级等价与跨进程逐位复现——同一 cubin 同一输入跨进程 5 次全过方为通过。

### 7. 四个可检验主张

旅程的产出物接受四条检验，数字继承自证据章（第二章速览，第三十六至四十二章完整论证）：

1. 表达力：旋钮空间在 PTX 指令族与结构两层覆盖两家手写参考；mma 指令签名层对两家双向一致；为此只需 1 个自定义 NVVM op（预算 ≤5）。
2. 调度：编译期 Z3 求解加一次性硬件标定的解析成本模型，每新形状调度 0.8 秒纯 CPU，两次完整编译逐字节一致，排序与实测的 Spearman 秩相关 0.8962。
3. 性能：前向三个基准形状在两类负载窗口下全面超过 fla Triton（共载窗口 2.125 / 1.221 / 1.349 倍，窗口口径第二章定义）；反向小形状 1.72 倍。
4. 边界：反向大形状 0.41–0.49 倍，四类独立分析证明该值由梯度容差合同决定——在 fla 的容差下，理想执行的上限为 0.85–0.92 倍。

### 8. 与 XLA/TPU 四支柱对位

对标的哲学是 Google 的 XLA/TPU 栈：调度决策全部在编译期做出、编译产物不携带运行时搜索、每层 IR 独立可验证。把它移植到原生 MLIR、消费级 Blackwell 与单算子族上，四支柱各有对应物：

| XLA/TPU 支柱 | 本编译器对应物 |
|---|---|
| 语义单一事实源（HLO 加 autodiff） | gdn 方言 L0 op 加 JAX fp64 可执行规约（三方交叉验证闭合） |
| 编译期确定性调度 | Z3 合法性求解加校准解析成本模型（固定种子，两次完整编译逐字节一致） |
| 分层可验证 IR | 每层一套独立验证：L1 三方交叉、L2 静态资源复核、L3 fp64 参考对照、L4 统计裁决 |
| 布局系统 | 共享内存与寄存器布局旋钮族加生存期别名 |

### 9. 章节地图

- 第四至十二章（调度求解器与成本模型）：把 autotune 变成求解器的动机与设计、旋钮空间建模、约束工程、确定性枚举、一次性标定与排序质量——编译期调度的两个引擎，自顶向下先讲设计决策。
- 第十三至二十三章（MLIR 体系）：方言定义、L0 与 L1 的全部 IR 展品、下降 pass、IR 中层（linalg、scf、bufferize、transform dialect）、树内行为的边界与生成器工程——语义层到调度工件的实现。
- 第二十四至二十九章（下译与 PTX 层）：NVVM 下译旋钮、nvgpu op 地图、inline_asm 逃生舱、PTX 后处理的四个变换与 ptxas/SASS 审计——离硬件最近的一层放最后讲。
- 第三十至三十五章（kernel 工程）：前向与反向内核族的制胜机制、PTX 后处理的四个变换与前后对照——第四站与第五站的机器层细节。
- 第三十六至四十二章（验证与边界）：分层验证塔、测量口径、覆盖证明与反向精度上界的四类分析、工具链发现与相关工作——全部尺子的说明书，也是主张 1 与主张 4 的完整论证。
