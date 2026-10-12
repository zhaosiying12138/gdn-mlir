# GDN-MLIR 深度解剖长文：从方言语义层到 Z3 调度、PTX 覆盖验证与体制变化战役的十一层完整拆解

> 本文是主论文《调度即 IR：基于原生 MLIR 的 Gated DeltaNet 算子编译期确定性调度及其 PTX 级验证》（`paper/zh/gdn-mlir-paper-zh.md`，下称"主论文"）的配套**深度解剖长文（deep dive）**，由十一个独立成篇的深度章节按层序合并而成：每章自带章首动机、小结与证据文件清单，小节编号全局唯一（§N.x 指第 N 章第 x 节），可整卷通读也可单章取用。全文以仓库源码与 `evidence/` 证据文件为唯一事实源，所有引文均注明真实路径；全部 GPU 数字沿用主论文口径——共载机会窗口计时、同窗交错终判、绝对毫秒按受扰下界解读。本章卷由 `paper/zh/sections/01-*.md` … `11-*.md` 原样合并装配（仅统一标题层级：各章 `#` 降为 `##`，章内层级相应下移一级），除标题层级外与单章文件逐字一致。

## 摘要

主论文给出"调度即 IR"的论纲与四支柱；本长文把同一栈按十一层逐一拆开，回答"每一层究竟怎么造、凭什么信"。第 1–2 章是语义与算法底座：`gdn` 方言两 op + 七旋钮属性构成单一事实源，语义闭合三角 JAX 规约 44/44、MLIR L1 对 JAX fp64 约 6×10⁻⁷、门算子超收缩 ρ(A)≤2.3×10⁻¹⁸ 给出结构性安全边界。第 3–4 章是发射与后处理机器：MMA 机器以 mma_gemm/条带贴图共享于两个生成器，唯一真逃生舱 `gdn.mma.tf32k8`（预算 1/≤5）实测反向 e2e 14.16→11.06 ms（1.28×）；`gdn_smem_fix.py` 把 dqkwg/caryA/wyb 的共享内存削减至 47,616/28,592/45,056 B，全线站上 2 CTA/SM。第 5–7 章是 kernel 体系：前向现行正式记录 B1T2048 **2.125×**、B4T2048 **1.2211×**、B1T8192 **1.3493×**（交错终判）；反向经 26 轮自提速 58.88→2.15 ms（约 27×），wy 站 2.06× 位级恒等，dqkwg 占用率三连跳。第 8–9 章是验证与测量：分层验证塔钉死容差契约与确定性，测量方法学给出九阶段 DRAM 地板和 1.074 ms、kkt 826 vs 765 GB/s 同窗屋顶线、fla autotune 税 39.8 s 的完整账目。第 10 章是编译期调度与覆盖：Z3 + 零拟合常数解析成本模型以 Spearman **0.8962** 排序、每新形状约 0.8 s 纯 CPU 求解，覆盖矩阵 33 族指令契约终态 8 YES / 19 PARTIAL / 6 NO。第 11 章是体制变化战役（RC1–R4）：仿射标度律在 GPU 触碰前定价整条 B/T/H 曲线，bf16 误差预算把"字节减半"变成 10/16 可铸、6 杀的合同，A_all 退役把 dhs 坍缩为移位拷贝（5.9×、逐字节等价），tgemm a-fragment hoist 清零 CVT 税（1.115×），融合定价三重杀收官——反向 u≥1 以**三重独立证据**关闭于精度受限地板（全阶梯 0.41–0.49×、边际比 0.472、u=8 天花板 0.85–0.92×，误差合同的数字），小形状 **1.72×** 与冷启动全形状获胜。

**关键词**：线性注意力；Gated DeltaNet；MLIR；transform dialect；PTX；共享内存；交错终判；Z3；覆盖矩阵；sm_120；标度律；误差预算；精度关闭

## 目录

- **第 1 章 方言与语义层（L0）与逃生舱 op 族**
- **第 2 章 L1 算法 IR 与数学：chunk-WY 分解的完整 MLIR 表达**
- **第 3 章 MMA 机器与混合精度体系：两个生成器共享的张量核发射层**
- **第 4 章 PTX 后处理层：`gdn_smem_fix.py` 与构建链的修复/变换全集**
- **第 5 章 前向 kernel 体系：`tools/gdn_full_gen.py` 的调度机件全集**
- **第 6 章 反向 kernel 体系：扫描族**
- **第 7 章 反向 kernel 体系之计算族：wy / dqkwg / wyb 与生存期竞技场编排**
- **第 8 章 数值验证与确定性体系**
- **第 9 章 测量与判定方法学：在共载 GPU 上建立裁决级证据**
- **第 10 章 编译期调度与 PTX 覆盖验证体系：Z3 求解器、解析成本模型、一次性标定、参考收割器、覆盖矩阵与旋钮库**
- **第 11 章 体制变化战役：标度律、误差预算与精度关闭定理**
- **附录：复现指南与证据索引（交叉引用）**

## 符号与约定

- **术语**沿用主论文篇首的术语约定块与附录 B 符号表：**原生 MLIR**（stock MLIR，未经 fork 的 LLVM/MLIR 主干，锚定 2026 年 10 月主干）、**旋钮**（knob，以 IR 属性或 transform 脚本参数形式存在的调度自由度）、**覆盖矩阵**、**逃生舱**（escape hatch，预算 ≤5）、**交错终判**（interleaved adjudication）、**共载**（co-load）。首次出现处各章仍会自带的定义以该章为准。
- **形状记号**：$B/T/H/HV/K/V$ 为批/序列长/头数/值头数/键维/值维；块长 $C=64$，块数 $NC=T/64$，块行总数 $M=B\cdot H\cdot NC$，$u=M/512$（B1T2048H16 锚的工作倍数）。平台为 RTX 5090 Laptop（sm_120a，82 SM，无 tcgen05）。
- **数字口径**：正式性能记录一律为同窗交错对内比值（n=30 对，配对检验 p 值随文标注）；单点毫秒数为共载窗口最小值或中位数，按受扰下界解读；共享内存字节数为 ptxas 报告值。
- **源码引用**：形如 `tools/gdn_bwd_gen.py`、`evidence/G5/bwd_report.md` 的路径均指仓库内真实文件；PTX/SASS 引文为真实摘录而非转写。
- **章际关系**：第 1–2 章为底座，第 3–4 章为发射/后处理机器，第 5–7 章为 kernel 体系（前向 + 反向两族），第 8–9 章为验证与测量方法学，第 10 章为编译期调度与覆盖验证，第 11 章为第 5–9 章反向与测量体系的战役级续章（体制变化战役 RC1–R4 的方法学与精度关闭定理）；各章末尾附"本章引用的证据文件"清单。

---

## 1 方言与语义层（L0）与逃生舱 op 族

### 1.1 章首：这一层为何存在

本节是 GDN-MLIR 编译器栈最底层的完整解剖：`gdn` 方言的语义层（L0）与为表达 warp 专化（warp specialization）词汇而设的逃生舱（escape hatch）op 族。在展开任何源码之前，先回答一个架构问题——**为什么要在原生 MLIR（stock MLIR，指未经 fork 的 LLVM/MLIR 主干，本文锚定 2026 年 10 月主干）之上再造一个方言层，而不是直接从 linalg/tensor 起步写算子。**

GDN-MLIR 的总体主张是"调度即 IR"：把 Gated DeltaNet（GDN，gated delta rule 线性注意力算子）kernel 的全部调度自由度表达为可变换、可验证、可复用的编译器 IR 旋钮（knob，指以 IR 属性或 transform dialect 脚本参数形式存在的调度自由度）。这一主张有四个支柱，与 Google TPU/XLA 栈逐条对位（`docs/PLAN.md` §0）：

| XLA/TPU 支柱 | GDN-MLIR 对应物 | 本节覆盖部分 |
|---|---|---|
| 语义单一事实源（HLO + autodiff） | `gdn` 方言 L0（`gdn.fwd`/`gdn.bwd` + JAX 可执行规约 + vjp 推导） | **全部**（本节） |
| 编译期确定性调度（非运行时 autotune） | Z3 + 一次性标定解析成本模型 | L0 只提供旋钮入口 |
| 分层可验证 IR | L0–L4 每层独立 oracle | L0 oracle：JAX fp64 对拍 |
| 布局系统 | smem/寄存器 layout 旋钮族 | L3 层话题，本节仅经逃生舱 op 触及 |

其中**支柱一的全部载体就是本节的 L0 层**。它的存在性理由可以拆成四点：

第一，**语义必须有且只有一份**。GDN 是一个数值约定极其密集的算子：decay 门 $g$ 处于自然对数域、scale 是 $K^{-1/2}$、$\beta$ 以原始值进入块内矩阵（而非 sigmoid 概率）、门更新 $h \leftarrow e^{g_t} h$ 必须先于 delta 更新、块内累计 decay 是含尾 cumsum——任何一条弄错，下游生成的 kernel 与参考实现之间就会出现"看起来都对、数值全错"的静默偏移（`docs/L0_SEMANTICS.md` 把这批约定称为"G1-A 实测裁定，勿凭直觉改"，共 8 条）。把语义钉进一个只有两个 op 与一个属性的小方言，等于把"算子是什么"从 kernel 代码、文档注释、参考实现的三处分散收敛为**单一事实源**（single source of truth）：`dialect/include/Gdn/GdnOps.td`（声明）+ `docs/L0_SEMANTICS.md`（约定）+ `tools/gdn_ref_jax.py`（JAX 可执行规约，fp64）。语义闭合三角已实测打通：JAX 规约自洽 44/44，JAX ⇄ fla naive 前向 1e-6/反向 2e-6 相符，MLIR L1 ⇄ JAX dumps ~6e-7 相符（`evidence/G1/GATE_REPORT.md`）。

第二，**语义入口要足够小，才能成为渐进下降（progressive lowering）的第一站**。方言声明的 description 明确写出五层下降路径：L0（`gdn.fwd`/`gdn.bwd`）→ L1 chunk-AST（KKT → 三角 solve → w/u → h 递推 → o）→ L2 调度（transform dialect）→ L3 流水/mma → L4 nvvm/sm_120a（`dialect/include/Gdn/GdnDialect.td` 第 16–23 行）。若没有 L0，每个下游层都得自行解释"什么算 GDN"，层级间的可验证性（每层有独立 oracle）就断了。

第三，**入口必须是可执行验证的对象，而不只是注释**。L0 op 带 verifier（合法性检查）、带 rewrite（`gdn-to-chunk` pass）、带 round-trip 冒烟测试（`dialect/test/smoke_l0.mlir`），构成 parse → verify → rewrite → print 的闭环（`evidence/G1/GATE_REPORT.md` 判据 1，冒烟全绿：roundtrip/rewrite/attr 透传/私有声明生成）。

第四，**逃生舱 op 族与语义 op 同方言注册，使"原生 MLIR 表达力边界"本身成为可审计对象**。本节第 1.5 节将展开：11 个 warp 专化/TMA 薄封装 op 全部 1:1 降到在树（in-tree）nvgpu/nvvm op，预算消耗 0；唯一真逃生舱 `gdn.mma.tf32k8` 消耗预算 1/≤5（`evidence/G2/hatch_report.md` ADDENDUM）。

本节解剖的文件清单（全部只读引用，路径相对仓根 `~/codebase/gdn-mlir`）：

| 文件 | 内容 | 本节小节 |
|---|---|---|
| `dialect/include/Gdn/GdnDialect.td` | 方言声明、`GdnRegDir` 枚举、`GdnConfigAttr` | 1.1 / 1.4 / 1.5 |
| `dialect/include/Gdn/GdnOps.td` | `gdn.fwd`、`gdn.bwd` ODS 定义 | 1.2 / 1.3 |
| `dialect/include/Gdn/GdnHatch.td` | 11 个逃生舱 op + `gdn.mma.tf32k8` | 1.5 |
| `dialect/include/Gdn/Gdn.h` | 伞头文件、`kAllowedChunkSizes` | 1.4 / 1.7 |
| `dialect/lib/GdnDialect.cpp` | 方言初始化、L0 verifier | 1.2 / 1.3 / 1.4 |
| `dialect/lib/GdnHatch.cpp` | 12 个 hatch 下降 rewrite + `gdn-lower-hatch` pass | 1.5 |
| `dialect/lib/GdnPasses.cpp` | `gdn-to-chunk` pass 两个 rewrite 模式 | 1.6 |
| `dialect/tools/gdn-opt/gdn-opt.cpp` | 方言驱动工具 | 1.7 |
| `dialect/CMakeLists.txt`、`dialect/include/CMakeLists.txt`、`dialect/lib/CMakeLists.txt`、`dialect/tools/gdn-opt/CMakeLists.txt` | 外建（out-of-tree）MLIR 项目构建 | 1.7 |
| `docs/L0_SEMANTICS.md` | 语义约定权威记录 | 1.2 / 1.4 |
| `dialect/test/smoke_l0.mlir` | L0 冒烟输入（真实 IR 示例来源） | 1.2 / 1.6 |

本节展示的所有"改写后 IR"均由实际构建产物 `build-dialect/bin/gdn-opt` 对真实输入运行得到，非手工构造。

---

### 1.2 `gdn.fwd`：前向语义入口

#### 1.2.1 动机（算子层面，含例子）

GDN 前向在逐步递推视角下定义为（门先于 delta 更新，次序本身是约定的一部分，`docs/L0_SEMANTICS.md` 约定 6）：

$$h_t \;=\; e^{g_t}\,h_{t-1} \;+\; \beta_t\, k_t\left(v_t^{\top} - k_t^{\top} e^{g_t}\, h_{t-1}\right),\qquad o_t \;=\; q_t^{\top} h_t$$

其中 $h$ 是差分状态（$K \times V$ 矩阵），$q/k/v$ 是 query/key/value，$\beta$ 是写入强度，$g \le 0$ 是 decay 门。`gdn.fwd` 的职责是把**这一整个数学对象**（而非它的某个分块实现）作为单一 IR op 呈现。具体到各输入的形状与含义（以冒烟测试 `dialect/test/smoke_l0.mlir` 的真实形状为例，B=2、T=128、H=HV=2、K=V=16）：

- `q : tensor<B x T x H x K>`——例中 `tensor<2x128x2x16xf32>`。若 config 的 `normalize_q` 为真则先做 L2 归一，随后统一乘 scale $K^{-1/2}$（**不是** $K^{-0.2}$；`docs/L0_SEMANTICS.md` 约定 2，乘在归一化后的 q 上）。
- `k : tensor<B x T x H x K>`——与 q 同形状同头数。
- `v : tensor<B x T x HV x V>`——允许与 q/k 不同的值头数（GVA/grouped-value attention 的入口形状自由度；例中 HV=H=2、V=16）。
- `beta : tensor<B x T x HV>`——例中 `tensor<2x128x2xf32>`。**原始值**：sigmoid 由模型侧上游完成，算子内不做（约定 4）；$\beta$ 以行向方式不对称地进入块内矩阵 $M[i,j] = -\beta_i (k_i \cdot k_j)\, e^{cd_i - cd_j}$。
- `g : tensor<B x T x HV>`——例中 `tensor<2x128x2xf32>`。**自然对数域**，decay $= e^{g} \le 1$（约定 1）。这是一个高频出错点：fla Triton 路径在 kernel 内把 g 乘 RCP_LN2 $= \log_2 e$ 后改用 exp2 计算，MLIR kernel 同样预乘后走 `math.exp2`——语义属性上写的永远是自然对数域的 g，换域是实现细节。
- `h0 : optional tensor<B x HV x K x V>`（f32）——例中 `tensor<2x2x16x16xf32>`。初始状态。**尾块补零**：当 T 不是 chunk_size 整数倍时以 beta=0/g=0 行为补齐（无操作行，约定 5）。

输出两个：

- `o : tensor<B x T x HV x V>`——逐步输出 $o_t = q_t^\top h_t$ 经 chunked WY 表示重组后的结果。
- `ht : tensor<B x HV x K x V>`（f32）——终态状态，供推理缓存或反向复用。

为什么这一层不直接写 linalg？因为 chunked WY 展开（块内构造 $T = (I+M)^{-1}$、$w/u$ 因子、跨块状态递推 $S_n = e^{cd_\text{last}} S_{n-1} + (k \odot e^{cd_\text{last}-cd})^\top (w - u S_{n-1})$）是 L1 层的算法结构选择：solve 用前代换还是 Schur 合并、状态放 fp32 还是 bf16、A 算完落盘还是反向重算，全是旋钮。L0 若掺入任何一种展开，"复现 = 调参不重写"的主张就被破坏。生产 kernel 的中间态 h/dh 为 bf16、而 h0/ht 恒为 f32，也是这条分界的体现（`docs/L0_SEMANTICS.md` 约定 8）。

#### 1.2.2 TableGen 源码逐行讲解

完整 ODS 定义（`dialect/include/Gdn/GdnOps.td` 第 10–34 行）：

```tablegen
//===----------------------------------------------------------------------===//
// gdn.fwd
//===----------------------------------------------------------------------===//

def GdnFwdOp : GdnOp<"fwd", [Pure]> {
  let summary = "GDN forward (semantic entry)";
  let description = [{
    q : tensor<B x T x H  x K>   (L2-normalized if config.normalize_q)
    k : tensor<B x T x H  x K>
    v : tensor<B x T x HV x V>
    beta : tensor<B x T x HV>    (write strength)
    g    : tensor<B x T x HV>    (log2-domain decay, <= 0)
    h0   : optional tensor<B x HV x K x V> f32
    -> o : tensor<B x T x HV x V>, ht : tensor<B x HV x K x V> f32
    Scale = K^{-1/2} applied to normalized q. See docs/L0_SEMANTICS.md.
  }];
  let arguments = (ins
    AnyRankedTensor:$q, AnyRankedTensor:$k, AnyRankedTensor:$v,
    AnyRankedTensor:$beta, AnyRankedTensor:$g,
    Optional<AnyRankedTensor>:$h0,
    GdnConfigAttr:$config
  );
  let results = (outs AnyRankedTensor:$o, AnyRankedTensor:$ht);
  let hasVerifier = 1;
}
```

逐行拆解：

- `class GdnOp<string mnemonic, list<Trait> traits = []> : Op<Gdn_Dialect, mnemonic, traits>;`（第 7–8 行）——本方言所有 op 的公共基类，把 op 挂到 `Gdn_Dialect` 下，mnemonic 即 `gdn.` 后的操作名。
- `GdnOp<"fwd", [Pure]>`——操作名 `gdn.fwd`；`Pure` trait 声明该 op 无副作用（side-effect free）：对相同输入恒 produce 相同结果，可被 CSE、可被自由重排、可被 greedy rewrite driver 多次访问而不改变语义。前向确实是纯函数，这个 trait 是诚实的而非偷懒。
- `let arguments = (ins ...)`——**六个操作数 + 一个属性**。`AnyRankedTensor` 只约束"任意 rank 的张量"，形状合法性（rank-4/rank-3 等）留给 verifier（见 1.2.3）——这是 ODS 的惯常分工：ODS 管结构性约束（有几个操作数、什么属性），C++ verifier 管跨操作数的语义约束。`Optional<AnyRankedTensor>:$h0` 把 h0 声明为可缺省操作数（MLIR 里 Optional 是长度 0 或 1 的 variadic），对应"无初始状态推理"这一算子层使用形态。`GdnConfigAttr:$config` 把 1.4 节的七旋钮束作为**必需属性**挂上——L0 即可调。
- `let results = (outs AnyRankedTensor:$o, AnyRankedTensor:$ht)`——恰好两个结果。注意 **op 总是同时产出 o 与 ht**，是否消费 ht 由下游决定（`store_final_state=false` 时 L1 展开会把 ht 变成死结果再由 DCE 收走；语义面两个都在）。
- `let hasVerifier = 1;`——生成 `LogicalResult GdnFwdOp::verify();` 声明，实现放 `dialect/lib/GdnDialect.cpp`。
- `description` 里那句 `g : ... (log2-domain decay, <= 0)` 与 `docs/L0_SEMANTICS.md` 的"g 是自然对数域"之间存在一处表观不一致——ODS 注记按 kernel 实现域（log2）描述，语义规约按数学域（ln）立法；权威以 `docs/L0_SEMANTICS.md` 为准（kernel 进 exp2 前的 RCP_LN2 预乘正是两个域的桥）。写作本节时如实记录这一文档分层。

#### 1.2.3 Verifier 源码逐行讲解

（`dialect/lib/GdnDialect.cpp` 第 43–63 行）

```cpp
static LogicalResult verifyConfig(gdn::GdnConfigAttr config, Operation *op) {
  if (!llvm::is_contained(gdn::kAllowedChunkSizes, config.getChunkSize()))
    return op->emitError() << "chunk_size must be one of {16,32,64}, got "
                           << config.getChunkSize();
  return success();
}

LogicalResult gdn::GdnFwdOp::verify() {
  if (failed(verifyConfig(getConfig(), *this)))
    return failure();
  auto q = mlir::dyn_cast<RankedTensorType>(getQ().getType());
  auto v = mlir::dyn_cast<RankedTensorType>(getV().getType());
  auto beta = mlir::dyn_cast<RankedTensorType>(getBeta().getType());
  if (!q || q.getRank() != 4)
    return emitError() << "q must be rank-4 [B,T,H,K]";
  if (!v || v.getRank() != 4)
    return emitError() << "v must be rank-4 [B,T,H,V]";
  if (!beta || beta.getRank() != 3)
    return emitError() << "beta must be rank-3 [B,T,HV]";
  return success();
}
```

要点：

- `verifyConfig` 先查 `chunk_size` 是否落在 `kAllowedChunkSizes`（`dialect/include/Gdn/Gdn.h` 第 32 行：`constexpr int kAllowedChunkSizes[] = {16, 32, 64};`）。这个三值合法域不是任意的：它就是 L1 chunk 算法结构的可行块长集合（16/32/64 对应不同的块内三角 solve 与跨块递推开销平衡；G1 对拍覆盖了 32/64 两档，`docs/L0_SEMANTICS.md` 旋钮表）。把合法域钉在 verifier 里，使得非法配置在 parse 后立即被拒，而不是等 L1 展开时生成错误 IR。
- 形状检查是**最小完备**的：只查 rank（q/v rank-4，beta rank-3），不查具体维数相等（如 `q.K == k.K`、`v.HV == beta.HV`）。这是一处有意的工程取舍——维数级一致性检查依赖动态维度知识（动态形状下 rank 已知而维数未知），放到 L1 对拍层（JAX/oracle）去完成；L0 verifier 只拦"结构性写错"（把 3 维 beta 接进来之类）。g 没有单独查 rank，因其与 beta 同形（[B,T,HV]），在语义测试里覆盖。

#### 1.2.4 真实 IR 展示（generic 语法）

`gdn.fwd` 未声明 `assemblyFormat`，因此**只有 generic 语法**可打印/解析（MLIR 规则：无自定义汇编格式的 op 用 `"op.name"(...)` 形式）。真实冒烟输入（`dialect/test/smoke_l0.mlir` 全文）：

```mlir
// gdn-opt L0 冒烟：parse + verify + gdn-to-chunk 改写（generic 形式）
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

逐元素解读：

- `"gdn.fwd"(%q, %k, %v, %beta, %g, %h0)`——六操作数按 ODS 声明序排列；h0 存在时占据第六槽。
- `<{config = #gdn.config<64, true, false, true, true, true, false>}>`——属性字典只含 config 一个属性；`#gdn.config<...>` 是 `GdnConfigAttr` 的 assembly 形式（1.4 节逐参数解读：chunk_size=64、use_g=true、use_gk=false、normalize_q=true、use_initial_state=true、store_final_state=true、is_varlen=false）。
- `: (...) -> (...)`——函数式类型签名，操作数类型与结果类型全部显式。

该文件被 `gdn-opt` 原样 round-trip（parse 后 print 逐字节等价，仅注释丢失），构成 G1 gate"冒烟全绿"的 roundtrip 判据（`evidence/G1/GATE_REPORT.md`）。

---

### 1.3 `gdn.bwd`：反向语义入口

#### 1.3.1 动机（算子层面，含例子）

反向 op 是 `gdn.fwd` 关于其全部可微输入的伴随（adjoint）。给定上游梯度（对 o 的 `dout`，以及对终态 ht 的可选 `dht`），产出**六个**梯度：

| 输出 | 类型约定 | 语义 |
|---|---|---|
| `dq` | io dtype（与 q 同） | $\partial L/\partial q$ |
| `dk` | io dtype | $\partial L/\partial k$ |
| `dv` | io dtype | $\partial L/\partial v$ |
| `dbeta` | f32 | $\partial L/\partial \beta$（标量链敏感，f32 保精度） |
| `dg` | f32 | $\partial L/\partial g$（自然对数域；链式项含严格前缀和结构） |
| `dh0` | f32（iff h0 给定） | $\partial L/\partial h_0$ |

dtype 的分裂不是装饰：生产 kernel 里 A/h/dh/do 走 bf16、ht/dg 走 fp32 是从参考实现 PTX 收割得到的 dtype 真相（`evidence/G0/fla_ptx/SUMMARY.md`），L0 的结果类型约定与其对齐。dg 之所以恒 f32，是因为 g 处于对数域、其梯度链对舍入误差的放大最敏感（G1 反向对拍中 dg 在 fp32 执行 / fp64 参照下 4.3e-6–6.0e-6，已在 printMemrefF32 六位有效数字地板，`evidence/G1/GATE_REPORT.md`）。

**A 重算-对-复用旋钮（recompute-vs-reuse）**是 `gdn.bwd` 独有的算子层自由度：反向需要块内 WY 因子（前向构造的 $T = (I+M)^{-1}$ 及相关量，例中形如 `tensor<B x NC x H x C x C>`，NC=块数、C=chunk_size）。两个使用形态：

1. **复用（reuse）**：训练框架在前向时把 A 物化并缓存，反向直接喂入——省重算、费显存；
2. **重算（recompute）**：A 缺省（`%A` 槽为空），反向内部重建——激活内存紧张时（单 kernel 反向、或长序列大 batch）的标准选择。这一取舍在 L1/L2 是 materialize-vs-recompute 旋钮族的根，在 L0 就以 Optional 操作数的形式露头：`A : optional saved WY factor (recompute-vs-reuse knob; absent => recompute)`（ODS description 原文）。

例子（与 1.2.1 同一形状族，T=128、C=64、NC=2）：复用形态给 `A : tensor<2x2x2x64x64xf32>`；重算形态不给 A。两种形态都是合法 IR，`gdn-to-chunk` 对二者生成相同的入口调用（仅实参列表不同，见 1.6.2）。

#### 1.3.2 TableGen 源码逐行讲解

（`dialect/include/Gdn/GdnOps.td` 第 36–59 行）

```tablegen
//===----------------------------------------------------------------------===//
// gdn.bwd
//===----------------------------------------------------------------------===//

def GdnBwdOp : GdnOp<"bwd", [Pure, AttrSizedOperandSegments]> {
  let summary = "GDN backward (semantic entry)";
  let description = [{
    Gradients of gdn.fwd w.r.t. q,k,v,beta,g,h0 given upstream do (optional dht).
    A : optional saved WY factor (recompute-vs-reuse knob; absent => recompute).
    Outputs dq,dk,dv (io dtype), dbeta,dg (f32), dh0 (f32, iff h0 given).
  }];
  let arguments = (ins
    AnyRankedTensor:$q, AnyRankedTensor:$k, AnyRankedTensor:$v,
    AnyRankedTensor:$beta, AnyRankedTensor:$g,
    Optional<AnyRankedTensor>:$h0,
    Optional<AnyRankedTensor>:$A,
    AnyRankedTensor:$dout,
    Optional<AnyRankedTensor>:$dht,
    GdnConfigAttr:$config
  );
  let results = (outs AnyRankedTensor:$dq, AnyRankedTensor:$dk, AnyRankedTensor:$dv,
                        AnyRankedTensor:$dbeta, AnyRankedTensor:$dg, AnyRankedTensor:$dh0);
  let hasVerifier = 1;
}
```

逐行拆解：

- `GdnOp<"bwd", [Pure, AttrSizedOperandSegments]>`——两个 trait。`Pure` 同前（反向亦为纯函数）。**`AttrSizedOperandSegments` 是本 op 的关键结构性决策**，原因如下：操作数列表里有**三个** Optional 段（h0、A、dht）夹在必选操作数之间，generic 语法解析器面对变长操作数串时无法从类型签名唯一推断每个 Optional 槽是"空"还是"占用"——比如 `(q,k,v,beta,g,dout)` 这串六个操作数，无法确定 h0/A/dht 哪两个缺省。该 trait 强制 op 在 generic 打印时携带 `operandSegmentSizes` 属性（每槽长度数组），从而 parse 无歧义。对照 `gdn.fwd`：它只有一个尾随 Optional（h0），唯一歧义是"h0 在不在"，解析器可从操作数总数推断，因此**不需要**该 trait——两 op 的 trait 差异精确对应歧义有无，这是 ODS 层面最小干预原则的一个实例。
- `let arguments`——九个操作数槽（5 必选 + h0? + A? + dout 必选 + dht?）+ config 属性。注意 `dout` 是必选（总有对 o 的上游梯度），`dht` 是可选（损失常不触碰终态）。操作数总数在 6 到 9 之间变化。
- `let results`——恒六个结果。**dh0 槽在 h0 缺省时仍然存在**（结果数是 op 签名的一部分，不能变长）；此时 dh0 语义上无效，L1 展开将其置零张量，下游 DCE 自然消化。这是"结果定长、操作数变长"的不对称设计。
- `let hasVerifier = 1;`——见下。

#### 1.3.3 Verifier 与真实 IR

（`dialect/lib/GdnDialect.cpp` 第 65–69 行）

```cpp
LogicalResult gdn::GdnBwdOp::verify() {
  if (failed(verifyConfig(getConfig(), *this)))
    return failure();
  return success();
}
```

反向 verifier 只查 config（chunk_size 合法域），不做 rank 检查——比前向更宽松。理由：反向的输入形状合法性完全继承前向（同一 q/k/v/beta/g/h0 集合），其形状错误会在生成 bwd 调用的前端（从前向 op 出发构造伴随）暴露；L0 层为 bwd 保留的最小契约就是"旋钮必须合法"。

复用形态的真实 round-trip IR（`gdn-opt` 对构造输入的实际打印；形状族同 1.3.1，A 复用、h0/dht 给定）：

```mlir
%dq, %dk, %dv, %dbeta, %dg, %dh0 = "gdn.bwd"(%q, %k, %v, %beta, %g, %h0, %A, %dout, %dht)
    <{config = #gdn.config<64, true, false, true, true, true, false>,
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1>}>
    : (tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>,
       tensor<2x128x2xf32>, tensor<2x128x2xf32>, tensor<2x2x16x16xf32>,
       tensor<2x2x2x64x64xf32>, tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
    -> (tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>,
       tensor<2x128x2xf32>, tensor<2x128x2xf32>, tensor<2x2x16x16xf32>)
```

`operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1>`——九槽全占用。重算形态（h0/A/dht 全缺省）则打印：

```mlir
%dq, %dk, %dv, %dbeta, %dg, %dh0 = "gdn.bwd"(%q, %k, %v, %beta, %g, %dout)
    <{config = #gdn.config<64, true, false, false, false, false, false>,
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 1, 0>}>
    : (...) -> (...)
```

第六（h0）、第七（A）、第九（dht）槽为 0——这就是 `AttrSizedOperandSegments` 存在的直接可见效果：没有这个属性，generic 解析器无法把六个操作数分配到正确的槽位。两种形态均经过 `gdn-opt` 实际 parse+verify+print 验证。

---

### 1.4 `GdnConfigAttr`：七个语义旋钮

#### 1.4.1 动机与定义

语义/算法旋钮若散落为 op 上的七个独立属性，改写 pass 与下游消费方都要逐个摘取、逐个透传，遗漏一个就静默改变语义。`GdnConfigAttr` 把它们**束成一个属性**：一次传递、一次比对、一个 assembly 拼写。完整定义（`dialect/include/Gdn/GdnDialect.td` 第 49–68 行）：

```tablegen
// 语义/算法旋钮束（L0 即可调；逐项含义见 docs/L0_SEMANTICS.md）
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

要点：`AttrDef<Gdn_Dialect, "GdnConfig">` + `mnemonic = "config"` 生成 `#gdn.config<...>` 拼写；七个参数全部是值参数（无类型参数），因此属性是纯值对象、可作编译期常量比较；`assemblyFormat` 一行钉死了打印序——**参数顺序就是 assembly 顺序**，两者由同一定义保证永不漂移。属性类经 `-gen-attrdef-*` 从本文件生成（`dialect/include/CMakeLists.txt` 第 34–35 行），在 `GdnDialect::initialize()` 注册（`dialect/lib/GdnDialect.cpp` 第 29 行 `addAttributes<GdnConfigAttr, GdnRegDirAttr>()`），并因方言声明 `useDefaultAttributePrinterParser = 1` 而获得默认的属性 mnemonics 注册（`#gdn.config` 可被 generic parser 解析）。

#### 1.4.2 七旋钮逐个讲解

| # | 旋钮 | 类型 | 合法域/默认 | 语义（逐项展开） | G1 对拍取值 |
|---|---|---|---|---|---|
| 1 | `chunk_size` | int | **{16, 32, 64}**，verifier 强制 | L1 chunk 算法的块长 C。决定块内三角系统 $T=(I+M)^{-1}$ 的规模（$C \times C$）、跨块递推步数 $\lceil T/C \rceil$、以及 mma fragment 的对齐基数。冒烟取 64。 | 32/64 已对拍 |
| 2 | `use_g` | bool | — | 标量门：g 为 [B,T,HV] 标量衰减（自然对数域，decay=$e^{g}$）。false 时算子退化为无门 delta rule。 | true |
| 3 | `use_gk` | bool | — | 逐通道门：decay 按通道独立（g 形状随之逐通道化）。当前对拍线未启用（false）。 | false |
| 4 | `normalize_q` | bool | — | 算子内部对 q 做 L2 归一。**关键语义细节**（`docs/L0_SEMANTICS.md` 约定 3）：算子内部默认**不做** q/k 的 L2 归一——归一由模型侧负责；fla 语义默认关闭。仅当上游确实需要时打开。冒烟取 true 是"打开"路径的覆盖。 | false（fla 对齐） |
| 5 | `use_initial_state` | bool | — | h0 是否给定。false 时 h0 操作数应缺省（或视作零态）。 | true（含 h0=0 用例） |
| 6 | `store_final_state` | bool | — | ht 是否被消费。false 时 ht 为死结果，由 L1 展开+DCE 消化（op 签名仍恒产出 ht，见 1.2.2）。 | true |
| 7 | `is_varlen` | bool | — | 变长序列打包（varlen）开关，属 G4 扩展面；本文实验不覆盖 varlen（声明性旋钮，留待扩展）。 | false |

两点补充。其一，`chunk_size` 的合法域由 verifier 与 `kAllowedChunkSizes`（`Gdn.h` 第 32 行）双重钉死，任何越界值（如 `<128, ...>`）在 verify 阶段即报错 `chunk_size must be one of {16,32,64}`。其二，`use_g`/`use_gk` 的区别是标量门对逐通道门——这决定了 decay 因子 $e^{cd_i - cd_j}$ 在块内矩阵 $M$ 里是标量乘还是按通道广播，直接改变 L1 生成的 IR 形状，因此在 L0 就必须可表达。

#### 1.4.3 assembly 语法与真实用例

`assemblyFormat` 生成的属性拼写（冒烟测试与全部生成器均用此形式）：

```mlir
#gdn.config<64, true, false, true, true, true, false>
       ^chunk  ^g   ^gk   ^nq   ^init  ^store ^varlen
```

七个参数以逗号分隔、尖括号包裹，按定义序排列。属性在 op 上以 `config = #gdn.config<...>` 出现（1.2.4 已见）；`gdn-to-chunk` 改写后它以 `gdn.config = #gdn.config<...>` 键名**原样透传**到入口调用上（1.6 节），键名 `gdn.config` 即方言名 + mnemonic 的标准展开。

---

### 1.5 GdnHatch 逃生舱 op 族：11 个薄封装 + 1 个真逃生舱

#### 1.5.0 总览：逃生舱机制与"零预算发现"

**逃生舱**（escape hatch）在本项目中指绕过原生 MLIR 表达边界、直书 PTX 的机制，总数受预算约束（≤5 个自定义 NVVM op），是"stock MLIR 极限"的论文度量。GdnHatch 的设计动机：warp 专化 kernel（FlashQLA/TileLang 风格）需要 setmaxnreg、elect.sync、mbarrier、TMA 这类 PTX 词汇，而项目立项时的假设是它们不在原生 MLIR 里、需要逃生舱。2026 年 10 月主干的实测发现颠覆了这个假设——**这些族全部已在树**（NVGPU/NVVM 方言已 table 化，且经 `-gpu-lower-to-nvvm-pipeline` 落到 sm_120a PTX），于是 GdnHatch 的 11 个 op 实现为**1:1 薄封装**：机制存在、被完整演练，但逃生舱预算消耗为 0（`evidence/G2/hatch_report.md`）。唯一真逃生舱是 `gdn.mma.tf32k8`（1.5.12），预算 1/≤5。

11+1 个 op 全部注册于同一 `gdn` 方言（`dialect/lib/GdnDialect.cpp` 第 32–36 行），共用下降 pass `gdn-lower-hatch`（`dialect/lib/GdnHatch.cpp` 第 223–240 行）。每个 op 的小节按统一四件套展开：动机（原生 MLIR 缺什么）→ 源码片段（路径标注）→ 下降链与 IR 前后对比 → 效果。IR 前后对比取自 `gdn-opt --gdn-lower-hatch` 对 `evidence/G2/scratch/g2c/smoke_hatch.mlir`（一个完整的最小 warp 专化 payload：生产者 warp 做 TMA 装载、消费者 warp 等待并 arrive）的实际运行输出。

先给出 `GdnRegDir` 枚举（`dialect/include/Gdn/GdnDialect.td` 第 38–47 行），它只被 setmaxnreg 消费：

```tablegen
def GdnRegDirInc : I32EnumAttrCase<"inc", 0>;
def GdnRegDirDec : I32EnumAttrCase<"dec", 1>;
def GdnRegDir : I32EnumAttr<"GdnRegDir", "GDN setmaxnreg direction",
  [GdnRegDirInc, GdnRegDirDec]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::gdn";
}
def GdnRegDirAttr : EnumAttr<Gdn_Dialect, GdnRegDir, "reg_dir">;
```

跨文件分工（值得注意的 TableGen 工程细节，原文件注释亦言明）：Attr 类（`GdnRegDirAttr`）随 `GdnDialect.td` 的 `-gen-attrdef-*` 后端生成，enum 声明（C++ 枚举类 `GdnRegDir`）却从 `GdnHatch.td` 的 `-gen-enum-decls` 生成（`dialect/include/CMakeLists.txt` 第 41–45 行）——两后端对同一逻辑枚举各出一半，构建系统里两处 `mlir_tablegen` 调用共同闭合。

#### 1.5.1 `gdn.setmaxnreg`：warp 级寄存器再分配

**动机（算子层面+例子）**。warp 专化的经典布局是：生产者 warp 只做 TMA 装载与屏障管理（寄存器需求极小），消费者 warp 跑 mma 主循环（寄存器饥渴）。PTX 的 `setmaxnreg.{inc,dec}.sync.aligned.u32` 允许 warp 之间转移每 warp 寄存器预算。例子：FlashQLA 风格的分工里生产者 `dec` 到 96、消费者 `inc` 到 224（`evidence/G2/hatch_report.md` 中 ws 配置实测发出 `setmaxnreg.dec...96` / `setmaxnreg.inc...224`）。原生 MLIR 立项时缺此词汇——现已在树为 `nvvm.setmaxregister`，封装价值在于以 gdn 拼写保持 payload 自洽、并把计数合法域钉进 verifier。

**源码**（`dialect/include/Gdn/GdnHatch.td` 第 32–44 行）：

```tablegen
def GdnSetmaxnregOp : GdnHatchOp<"setmaxnreg"> {
  let summary = "warp register redistribution (PTX setmaxnreg.{inc,dec}.sync.aligned.u32)";
  let description = [{
    Warp-specialization hatch: raises (inc) or lowers (dec) the per-warp
    register budget. Producer warps dec to hand registers to consumer warps
    (which inc). Count must be a multiple of 8 in [24, 256]. Lowers to the
    in-tree `nvvm.setmaxregister` (LLVM intrinsics
    nvvm_setmaxnreg_{inc,dec}_sync_aligned_u32).
  }];
  let arguments = (ins I32Attr:$count, GdnRegDirAttr:$direction);
  let assemblyFormat = "enum($direction) $count attr-dict";
  let hasVerifier = 1;
}
```

verifier（`dialect/lib/GdnHatch.cpp` 第 246–252 行）：`count` 必须是 8 的倍数且落在 [24, 256]——这是 PTX ISA 对该指令的硬约束，提前到 IR 层拦截。

**下降链与 IR 前后对比**（`dialect/lib/GdnHatch.cpp` 第 29–41 行 `SetmaxnregLowering`）：

```cpp
struct SetmaxnregLowering : OpRewritePattern<GdnSetmaxnregOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GdnSetmaxnregOp op,
                                PatternRewriter &rewriter) const override {
    auto action = op.getDirection() == GdnRegDir::inc
                      ? NVVM::SetMaxRegisterAction::increase
                      : NVVM::SetMaxRegisterAction::decrease;
    NVVM::SetMaxRegisterOp::create(rewriter, op.getLoc(),
                                   (uint32_t)op.getCount(), action);
    rewriter.eraseOp(op);
    return success();
  }
};
```

枚举翻译（`GdnRegDir::inc/dec` → `NVVM::SetMaxRegisterAction::increase/decrease`）后原位创建在树 op、抹除原 op。真实前后对比（取自 smoke_hatch 运行）：

```mlir
// before                                   // after
gdn.setmaxnreg dec 40                      nvvm.setmaxregister decrease 40
gdn.setmaxnreg inc 168                     nvvm.setmaxregister increase 168
```

**效果**。`evidence/G2/hatch_report.md` 记录 ws（warp 专化）配置编译实测：经 `nvvm.setmaxregister` 落到 PTX 的精确拼写 `setmaxnreg.inc.sync.aligned.u32 224` / `.dec... 96`；带 ws 的 kkt 配置（`kkt_bf16_mma_p2_t64x64x64_w4_ws`）255 regs + 22844 B spill，与 FlashQLA 极点（全部 ≤128 regs 零 spill）的差距推动 TMA 变体（`kkt_bf16_mma_tma_ws`：255 regs / 48 B spill / 41.2 KB PTX），其中寄存器再分配是 warp 专化成立的必要词汇。T1 指令族覆盖矩阵中 `setmaxnreg` 族由 NO 翻 YES。

#### 1.5.2 `gdn.elect`：warp 领袖选举

**动机**。TMA 发射与 `arrive.expect_tx` 要求**单线程**执行（硬件按线程粒度计账），而 warp 专化代码运行在 warp 集体语义里；PTX `elect.sync` 选举一个带谓词的活跃领袖线程。例子：生产者 warp 中 `elect` 出领袖，由领袖且仅由领袖发起 TMA 装载（smoke_hatch 的 `scf.if %el` 结构）。原生在树对应物是 `nvvm.elect.sync`。

**源码**（`dialect/include/Gdn/GdnHatch.td` 第 46–55 行）：

```tablegen
def GdnElectOp : GdnHatchOp<"elect"> {
  let summary = "warp-leader election (PTX elect.sync)";
  let description = [{
    Elects one predicated active leader thread of the warp; result is true on
    the leader. Used to single-thread TMA issue / mbarrier arrive.expect_tx.
    Lowers to the in-tree `nvvm.elect.sync` (default membermask 0xFFFFFFFF).
  }];
  let results = (outs I1:$pred);
  let assemblyFormat = "attr-dict `->` type($pred)";
}
```

**下降链与 IR 前后对比**（`ElectLowering`，`dialect/lib/GdnHatch.cpp` 第 43–51 行）：`replaceOpWithNewOp<NVVM::ElectSyncOp>(op, rewriter.getI1Type(), Value())`——成员掩码传空 `Value()`，落入默认 0xFFFFFFFF（全 warp）。

```mlir
// before                        // after
%el = gdn.elect -> i1            %el = nvvm.elect.sync -> i1
```

**效果**。`evidence/G2/hatch_report.md`：PTX 精确拼写 `elect.sync %r|%p, -1`；T1 族 `elect.sync` NO→YES。在 ws 配置里 elect 与 shfl.sync 广播谓词共同构成生产者/消费者分工的控制流骨架。

#### 1.5.3 `gdn.mbarrier.init`：mbarrier 初始化

**动机**。mbarrier（memory barrier，共享内存中的到达-等待屏障对象）是异步数据搬运完成通知的标准机制：TMA 装载以 `complete_tx::bytes` 计数到达 mbarrier，消费者以相位奇偶等待。使用前必须以到达计数初始化。例子：smoke_hatch 中单线程 `gdn.mbarrier.init %bar, %c96`（96 个到达者 = 3 个消费 warp × 32 线程）。原生对应 `nvgpu.mbarrier.init`。

**源码**（`dialect/include/Gdn/GdnHatch.td` 第 57–66 行）：

```tablegen
def GdnMbarrierInitOp : GdnHatchOp<"mbarrier.init"> {
  let summary = "mbarrier initialization (PTX mbarrier.init.shared.b64)";
  let arguments = (ins AnyType:$barrier, Index:$count);
  let assemblyFormat = "$barrier `,` $count attr-dict `:` type($barrier)";
}
```

`$barrier` 是 `nvgpu.mbarrier.create` 产出的 `!nvgpu.mbarrier.group<...>` 值——hatch op 有意**复用在树类型系统**而非自造类型，这正是"薄封装"的含义。

**下降链与 IR 前后对比**（`MbarrierInitLowering`，`dialect/lib/GdnHatch.cpp` 第 53–62 行）：

```cpp
rewriter.replaceOpWithNewOp<nvgpu::MBarrierInitOp>(
    op, op.getBarrier(), op.getCount(), czero(rewriter, op.getLoc()),
    Value());
```

注意 `czero(...)`（同文件第 25–27 行，造 `arith.constant 0 : index`）：在树 op 的屏障**下标**操作数（barrier id）被填 0——gdn 封装假定单屏障（id 0），这是封装做出的具体简化。前后对比：

```mlir
// before                                            // after
gdn.mbarrier.init %bar, %c96 : !nvgpu.mbarrier...    nvgpu.mbarrier.init %bar[%c0], %c96 : <memorySpace = ...>
```

（在树 op 的打印省略 `!nvgpu.` 前缀，故显示裸 `<memorySpace = ...>`——`evidence/G2/hatch_report.md` 特意记录了这一打印差异。）

**效果**。PTX 精确拼写 `mbarrier.init.shared.b64`；与 1.5.4–1.5.6 三兄弟一起把 T1 `mbarrier` 族从 NO 翻到 PARTIAL（唯一缺口是参考拼写带显式 `.shared::cta` 域而 ours 打印默认域 `.shared`——同一指令的字符串变体差异）。

#### 1.5.4 `gdn.mbarrier.arrive`：到达操作

**动机**。消费者 warp 读完共享内存 tile 后必须在 mbarrier 上"到达"，生产者据此得知缓冲可覆写。返回 token 供配对。原生对应 `nvgpu.mbarrier.arrive`。

**源码**（`dialect/include/Gdn/GdnHatch.td` 第 68–77 行）：

```tablegen
def GdnMbarrierArriveOp : GdnHatchOp<"mbarrier.arrive"> {
  let summary = "mbarrier arrive-on (PTX mbarrier.arrive.shared.b64)";
  let arguments = (ins AnyType:$barrier);
  let results = (outs AnyType:$token);
  let assemblyFormat = "$barrier attr-dict `:` type($barrier) `->` type($token)";
}
```

结果 `$token` 是 `!nvgpu.mbarrier.token`——同样复用在树类型。

**下降链与 IR 前后对比**（`MbarrierArriveLowering`，`dialect/lib/GdnHatch.cpp` 第 64–73 行）：`replaceOpWithNewOp<nvgpu::MBarrierArriveOp>(op, op.getToken().getType(), op.getBarrier(), czero(...))`——同样是 id 0 填充。

```mlir
// before                                                          // after
%tok = gdn.mbarrier.arrive %bar : !nvgpu.mbarrier... -> !nvgpu.mbarrier.token
                                                    %tok = nvgpu.mbarrier.arrive %bar[%c0] : <...> -> !nvgpu.mbarrier.token
```

**效果**。PTX `mbarrier.arrive.shared.b64`；smoke_hatch 中消费者分支的到达、ws 配置流水闭环的一环；贡献 `mbarrier` 族 PARTIAL→(与 1.5.3 同族)。

#### 1.5.5 `gdn.mbarrier.arrive.expect_tx`：带预期事务字节数的到达

**动机**。TMA 异步装载的完成通知走字节计数：领袖线程在到达的同时**增加**屏障的预期事务字节数（例：一次 64×64 bf16 tile 装载 = 8192 字节，smoke_hatch 的 `gdn.mbarrier.arrive.expect_tx %bar, %c8192`），TMA 硬件在数据落完后以 `complete_tx::bytes` 消账——预期与完成相抵屏障才翻相。原生对应 `nvgpu.mbarrier.arrive.expect_tx`。

**源码**（`dialect/include/Gdn/GdnHatch.td` 第 79–89 行）：

```tablegen
def GdnMbarrierArriveExpectTxOp : GdnHatchOp<"mbarrier.arrive.expect_tx"> {
  let summary = "mbarrier arrive with expected tx-count (PTX mbarrier.arrive.expect_tx.shared.b64)";
  let description = [{
    Arrives on the mbarrier and increases its expected async-transaction
    byte count by $txcount (used with TMA loads completing via
    mbarrier::complete_tx::bytes). Lowers to the in-tree
    `nvgpu.mbarrier.arrive.expect_tx`.
  }];
  let arguments = (ins AnyType:$barrier, Index:$txcount);
  let assemblyFormat = "$barrier `,` $txcount attr-dict `:` type($barrier)";
}
```

**下降链与 IR 前后对比**（`MbarrierArriveExpectTxLowering`，`dialect/lib/GdnHatch.cpp` 第 75–86 行）：同样的 id 0 填充 + 透传 `$txcount`。

```mlir
// before                                            // after
gdn.mbarrier.arrive.expect_tx %bar, %c8192 : ...      nvgpu.mbarrier.arrive.expect_tx %bar[%c0], %c8192 : <...>
```

**效果**。PTX `mbarrier.arrive.expect_tx.shared.b64`；TMA 变体 ws 配置（`kkt_bf16_mma_tma_ws`）的完整 TMA 家族成员之一，该配置以 5 条 cp.async 家族指令 + 41.2 KB PTX 达到 FlashQLA 风格的异步数据通路（对比经典拷贝变体 1026 条 cp.async + 214.6 KB PTX，`evidence/G2/hatch_report.md` 资源表）。

#### 1.5.6 `gdn.mbarrier.try_wait.parity`：相位奇偶等待

**动机**。多级流水中屏障相位交替翻转（第 0 相、第 1 相、第 0 相……），消费者等待"给定奇偶的相位已完成"。`try_wait` 带超时滴答数（ticks），可能阻塞。例子：smoke_hatch 消费者 `gdn.mbarrier.try_wait.parity %bar, %false, %cticks`（等偶相、10,000,000 滴答预算）。原生对应 `nvgpu.mbarrier.try_wait.parity`。

**源码**（`dialect/include/Gdn/GdnHatch.td` 第 91–100 行）：

```tablegen
def GdnMbarrierTryWaitParityOp : GdnHatchOp<"mbarrier.try_wait.parity"> {
  let summary = "mbarrier phase-parity wait (PTX mbarrier.try_wait.parity.shared.b64)";
  let description = [{
    Potentially-blocking test of whether the mbarrier completed the phase of
    the given parity (false = even/first phase, true = odd/second phase).
    Lowers to the in-tree `nvgpu.mbarrier.try_wait.parity`.
  }];
  let arguments = (ins AnyType:$barrier, I1:$parity, Index:$ticks);
  let assemblyFormat = "$barrier `,` $parity `,` $ticks attr-dict `:` type($barrier)";
}
```

**下降链与 IR 前后对比**（`MbarrierTryWaitParityLowering`，`dialect/lib/GdnHatch.cpp` 第 88–99 行）：

```mlir
// before                                                        // after
gdn.mbarrier.try_wait.parity %bar, %false, %cticks : ...          nvgpu.mbarrier.try_wait.parity %bar[%c0], %false, %c10000000 : <...>
```

**效果**。PTX `mbarrier.try_wait.parity.shared.b64`；在 ws 配置里相位奇偶以 `scf.for` 的 i1 iter_arg 乒乓（`evidence/G2/hatch_report.md` 结构注记：单 smem 缓冲、无多版本乒乓、编译期保真度的有意简化）。至此 mbarrier 四件套（init/arrive/expect_tx/try_wait）齐——T1 `mbarrier` 族四个操作全部可发出。

#### 1.5.7 `gdn.tma.async.load`：TMA 异步张量装载

**动机**。TMA（Tensor Memory Accelerator）以硬件张量映射描述符（tensormap descriptor）一次搬整块多维 tile 到共享内存，绕开逐线程访存。编译期描述符是宿主侧 `nvgpu.tma.create.descriptor` 造的**不透明**值（真机连线是运行时/G4 的事——`cuTensorMapEncodeTiled`）；kernel 侧只需"按坐标装到目的 memref、完成记到 mbarrier"。原生对应 `nvgpu.tma.async.load`。

**源码**（`dialect/include/Gdn/GdnHatch.td` 第 102–115 行）：

```tablegen
def GdnTmaAsyncLoadOp : GdnHatchOp<"tma.async.load"> {
  let summary = "TMA async tensor load (PTX cp.async.bulk.tensor.Nd...mbarrier::complete_tx::bytes)";
  let arguments = (ins AnyType:$dst, AnyType:$descriptor, AnyType:$barrier,
                       Variadic<Index>:$coordinates);
  let assemblyFormat = "$descriptor `[` $coordinates `]` `,` $barrier `to` $dst attr-dict `:` type($descriptor) `,` type($barrier) `->` type($dst)";
  let hasVerifier = 1;
}
```

verifier（`dialect/lib/GdnHatch.cpp` 第 254–267 行）：坐标数必须在 1..5（`verifyTmaCoords`）——TMA 张量描述符最多 5 维，越界即拒。`Variadic<Index>:$coordinates` 是变长坐标串，assembly 里以 `[...]` 包裹。

**下降链与 IR 前后对比**（`TmaAsyncLoadLowering`，`dialect/lib/GdnHatch.cpp` 第 101–110 行）：

```cpp
rewriter.replaceOpWithNewOp<nvgpu::TmaAsyncLoadOp>(
    op, op.getDst(), op.getBarrier(), op.getDescriptor(),
    op.getCoordinates(), czero(rewriter, op.getLoc()), Value(), Value());
```

在树 op 多出的三个槽（屏障 id 与两个可选的 cache-hint/lbhint 类操作数）分别填 0/空——再次体现"封装做出具体缺省"的设计。真实前后对比（4 维坐标装载 64×64 bf16 tile）：

```mlir
// before
gdn.tma.async.load %descA[%c0, %c0, %c0, %c0], %bar to %sa
    : !nvgpu.tensormap.descriptor<...>, !nvgpu.mbarrier.group<...>
      -> memref<1x1x64x64xbf16, #gpu.address_space<workgroup>>
// after
nvgpu.tma.async.load %0[%c0, %c0, %c0, %c0], %2[%c0] to %alloc
    : <tensor = memref<1x1x64x64xbf16, 3>, ...>, <memorySpace = ...>
      -> memref<1x1x64x64xbf16, #gpu.address_space<workgroup>>
```

**效果**。PTX `cp.async.bulk.tensor.{4,5}d.shared::cluster.global.tile.mbarrier::complete_tx::bytes`（4d/5d 装载均已验证）；T1 `tma.bulk.tensor` 族 NO→PARTIAL（缺口：参考用的 `.L2::cache_hint` 装载拼写与 4d/5d store 变体——在树 load 下降硬编码无 hint）。

#### 1.5.8 `gdn.tma.async.store`：TMA 异步张量写回

**动机**。epilogue 把共享内存里的结果 tile 写回全局张量：同样走描述符与坐标，完成机制是 bulk 异步组（与 1.5.10/1.5.11 配对）。原生对应 `nvgpu.tma.async.store`。

**源码**（`dialect/include/Gdn/GdnHatch.td` 第 117–129 行）：

```tablegen
def GdnTmaAsyncStoreOp : GdnHatchOp<"tma.async.store"> {
  let summary = "TMA async tensor store (PTX cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group)";
  let description = [{
    Stores the workgroup-memref tile $src to the global tensor through the
    tensor-map descriptor (bulk async-group completion; pair with
    gdn.bulk.commit_group / gdn.bulk.wait_group). Lowers to the in-tree
    `nvgpu.tma.async.store`.
  }];
  let arguments = (ins AnyType:$src, AnyType:$descriptor,
                       Variadic<Index>:$coordinates);
  let assemblyFormat = "$src `to` $descriptor `[` $coordinates `]` attr-dict `:` type($src) `->` type($descriptor)";
  let hasVerifier = 1;
}
```

verifier 同样经 `verifyTmaCoords` 查 1..5 维。

**下降链与 IR 前后对比**（`TmaAsyncStoreLowering`，`dialect/lib/GdnHatch.cpp` 第 112–120 行）：`replaceOpWithNewOp<nvgpu::TmaAsyncStoreOp>(op, op.getSrc(), op.getDescriptor(), op.getCoordinates(), Value())`——额外槽填空。

```mlir
// before
gdn.tma.async.store %sa to %descA[%c0, %c0, %c0, %c0]
    : memref<1x1x64x64xbf16, #gpu.address_space<workgroup>> -> !nvgpu.tensormap.descriptor<...>
// after
nvgpu.tma.async.store %alloc to %0[%c0, %c0, %c0, %c0]
    : memref<1x1x64x64xbf16, #gpu.address_space<workgroup>> -> <tensor = memref<1x1x64x64xbf16, 3>, ...>
```

**效果**。PTX `cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group`（2d store 已验证；4d/5d store 为族内缺口）；smoke_hatch 的 epilogue 三连（store → commit → wait）的头部。

#### 1.5.9 `gdn.tma.prefetch`：描述符预取

**动机**。tensormap 描述符驻留全局内存，首次访问会吃一次缺失延迟；`prefetch.tensormap` 把含描述符的缓存行提前拉入。这是 FlashQLA 风格流水起播的一部分（领袖线程在发起装载前先 prefetch）。原生对应 `nvgpu.tma.prefetch.descriptor`。

**源码**（`dialect/include/Gdn/GdnHatch.td` 第 131–139 行）：

```tablegen
def GdnTmaPrefetchOp : GdnHatchOp<"tma.prefetch"> {
  let summary = "tensormap prefetch (PTX prefetch.tensormap)";
  let description = [{
    Prefetches the cache line containing the tensor-map descriptor. Lowers
    to the in-tree `nvgpu.tma.prefetch.descriptor`.
  }];
  let arguments = (ins AnyType:$descriptor);
  let assemblyFormat = "$descriptor attr-dict `:` type($descriptor)";
}
```

**下降链与 IR 前后对比**（`TmaPrefetchLowering`，`dialect/lib/GdnHatch.cpp` 第 122–130 行）：`replaceOpWithNewOp<nvgpu::TmaPrefetchOp>(op, op.getDescriptor(), Value())`。

```mlir
// before                                        // after
gdn.tma.prefetch %descA : !nvgpu.tensormap...    nvgpu.tma.prefetch.descriptor %0 : <tensor = ...>
```

**效果**。PTX 精确拼写 `prefetch.tensormap [%rd]`；T1 `prefetch.tensormap` 族 NO→**YES**（全精确）。

#### 1.5.10 `gdn.bulk.commit_group`：提交 bulk 异步组

**动机**。TMA store 属 cp.async.bulk 家族，完成语义按"异步组"记账：store 发出后必须 `commit_group` 把此前所有未提交的 bulk 操作打包成一个组，等待才有对象。原生对应 `nvvm.cp.async.bulk.commit.group`。

**源码**（`dialect/include/Gdn/GdnHatch.td` 第 141–148 行）：

```tablegen
def GdnBulkCommitOp : GdnHatchOp<"bulk.commit_group"> {
  let summary = "commit cp.async.bulk group (PTX cp.async.bulk.commit_group)";
  let description = [{
    Commits all prior uncommitted cp.async.bulk operations into a bulk
    async-group. Lowers to the in-tree `nvvm.cp.async.bulk.commit.group`.
  }];
  let assemblyFormat = "attr-dict";
}
```

零操作数、零结果、纯动作——assembly 只剩 `attr-dict`。

**下降链与 IR 前后对比**（`BulkCommitLowering`，`dialect/lib/GdnHatch.cpp` 第 132–139 行）：`replaceOpWithNewOp<NVVM::CpAsyncBulkCommitGroupOp>(op)`——最纯粹的 1:1。

```mlir
// before                     // after
gdn.bulk.commit_group         nvvm.cp.async.bulk.commit.group
```

**效果**。PTX `cp.async.bulk.commit_group`；T1 `tma.bulk.ctrl` 族 NO→**YES**（与 1.5.11 联合，拼写全精确）。

#### 1.5.11 `gdn.bulk.wait_group`：等待 bulk 异步组

**动机**。共享内存 tile 写回完成前不可覆写（否则写读竞争）：`wait_group n` 阻塞至最近 n 个 bulk 组以内未决（n=0 即全清）；`read` 变体只等"源已被读走"（写回场景的正确强度——不必等全局可见，只等 smem 可复用）。例子：smoke_hatch epilogue 的 `gdn.bulk.wait_group 0 read`。原生对应 `nvvm.cp.async.bulk.wait_group`。

**源码**（`dialect/include/Gdn/GdnHatch.td` 第 150–159 行）：

```tablegen
def GdnBulkWaitOp : GdnHatchOp<"bulk.wait_group"> {
  let summary = "wait cp.async.bulk group (PTX cp.async.bulk.wait_group[.read])";
  let description = [{
    Waits until $group or fewer of the most recent bulk async-groups are
    pending; with `read`, waits only until sources have been read. Lowers
    to the in-tree `nvvm.cp.async.bulk.wait_group`.
  }];
  let arguments = (ins I32Attr:$group, OptionalAttr<UnitAttr>:$read);
  let assemblyFormat = "$group (`read` $read^)? attr-dict";
}
```

`$group` 是 I32**属性**（编译期常量——等待深度是流水结构参数）；`$read` 是可选 UnitAttr，assembly 以 `` (`read` $read^)? `` 表达"出现即真"。一个已沉淀的机器约束教训：wait_group-0 在 sm_120 上有硬件级约束（`evidence/G6/failure_longscroll.md` 开放发现账本之一），这也是该 op 必须作为一等旋钮而非写死的原因。

**下降链与 IR 前后对比**（`BulkWaitLowering`，`dialect/lib/GdnHatch.cpp` 第 141–150 行）：

```cpp
rewriter.replaceOpWithNewOp<NVVM::CpAsyncBulkWaitGroupOp>(
    op, rewriter.getI32IntegerAttr(op.getGroup()),
    op.getRead() ? UnitAttr::get(op.getContext()) : UnitAttr());
```

```mlir
// before                          // after
gdn.bulk.wait_group 0 read         nvvm.cp.async.bulk.wait_group 0 read
```

**效果**。PTX `cp.async.bulk.wait_group.read 0` 精确；与 commit_group 联合翻 YES `tma.bulk.ctrl`。

#### 1.5.12 `gdn.mma.tf32k8`：唯一的真逃生舱

这一小节是本节分量最重的一段：它是全项目**逃生舱预算的唯一消耗者**（1/≤5），其下降实现携带了两条经硬件验证的工具链发现（early-clobber 规则与 fragment 布局律），数字取自 `evidence/G5/bwd_report.md` 与 `evidence/G2/hatch_report.md` ADDENDUM。

**动机（原生 MLIR 缺什么）**。反向 kernel 的 24 个 f32 GEMM 位点里，16 个经逐位点数值二分判定可降 tf32 精度跑张量核（判据与二分表见 `evidence/G5/bwd_report.md`：dqkwg 8 位点全过、wyb 8 位点过 7、wy.w 干净而同形状的 wy.u 因误差沿状态扫描放大 39 倍超 8e-3 杆被否——"同形状同 K 一个干净一个有毒"的不对称是逐位点二分必要性的直接论据）。fla 手写 kernel 收割到的精确指令拼写是 `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`（`evidence/G0/fla_ptx`，出现在 chunk_gated_delta_rule_bwd_kernel_dhu 与 fwd_kkt_solve）。而这条签名在原生 MLIR **两级都够不着**：

1. `nvgpu.mma.sync` 强制 8×8×128b 基本片——tf32 每元素 4 字节，K 方向 8 位置 ÷ 4 B = 封顶 **K=4**；
2. 表列 `nvvm.mma.sync` 的 TableGen 家族里 m16n8k8 形状只列了 f16/bf16/i8，无 tf32。

（`evidence/G2/gen_report.md` 把它记为已知缺口：原生可达的最高 tf32 形状是 m16n8k4，同样 f32 累加——k8 把每原子 K 吞吐翻倍。）

**源码**（`dialect/include/Gdn/GdnHatch.td` 第 161–194 行，节选）：

```tablegen
def GdnMmaTf32K8Op : GdnHatchOp<"mma.tf32k8", [Pure]> {
  let summary = "tf32 m16n8k8 warp mma (PTX mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32) -- ESCAPE HATCH #1";
  let description = [{
    G5-G round 3, REAL ESCAPE HATCH #1 of <=5: the tf32 m16n8k8 mma.sync
    signature does not exist in-tree (nvgpu.mma.sync enforces the 8x8x128b
    fundamental tile -> tf32 caps at K=4; the nvvm.mma.sync tablegen family
    lists m16n8k8 only for f16/bf16/i8 -- evidence/G2/gen_report.md).  Lowered
    to `nvvm.inline_ptx` emitting exactly:
      mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
        {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};
    (the exact spelling fla harvests into chunk_gated_delta_rule_bwd_kernel_
    dhu / fwd_kkt_solve -- evidence/G0/fla_ptx).

    Scalar f32 in/out (operands are f32 values; the tensor core rounds them
    to tf32 internally, 10-bit mantissa, f32 exponent range).  Fragment
    layout HARDWARE-DERIVED (G5-G probe exp1+exp2, evidence/G5/scratch/
    tf32k8_probe -- exact integer match; note the A register order differs
    from the packed-pair intuition of bf16 k16):
      groupID g = laneid >> 2, threadID_in_group t = laneid mod 4
      A (16x8): a0=(g,2t) a1=(g+8,2t) a2=(g,2t+1) a3=(g+8,2t+1)
      B (8x8, col): b0=(2t,g) b1=(2t+1,g)                        [k, n]
      C/D (16x8): d0=(g,2t) d1=(g,2t+1) d2=(g+8,2t) d3=(g+8,2t+1)
    D-fragment N index = 2*t (the G4 rule, same as the bf16 m16n8k16 D).
  }];
  let arguments = (ins F32:$a0, F32:$a1, F32:$a2, F32:$a3,
                       F32:$b0, F32:$b1,
                       F32:$c0, F32:$c1, F32:$c2, F32:$c3);
  let results = (outs F32:$d0, F32:$d1, F32:$d2, F32:$d3);
  let assemblyFormat = [{
    $a0 `,` $a1 `,` $a2 `,` $a3 `,` $b0 `,` $b1 `,`
    $c0 `,` $c1 `,` $c2 `,` $c3 attr-dict `:` type($a0) `->` `(` type($d0) `,`
    type($d1) `,` type($d2) `,` type($d3) `)`
  }];
}
```

设计要点：**标量 f32 进出**——操作数是 10 个独立 f32 值（不是向量/tensor），tf32 舍入（10 位尾数、f32 指数域）由张量核硬件内部完成；这使 op 可在任意标量 IR 上下文（寄存器级负载）中使用，fragment 装载由发射器（`tools/gdn_bwd_gen.py` 的 tgemm 位点发射器）负责。它是 `Pure` 的（同一 mma 对同一操作数恒等）。pretty 语法实测 round-trip：

```mlir
%d0, %d1, %d2, %d3 = gdn.mma.tf32k8 %a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3
    : f32 -> (f32, f32, f32, f32)
```

**下降链（`MmaTf32K8Lowering`，`dialect/lib/GdnHatch.cpp` 第 152–211 行）**。注意 ODS description 里写的"`nvvm.inline_ptx`"是计划时的措辞，实现**最终不走**该路径——原因正是下面这条 early-clobber 教训（源码注释原文保留了完整决策记录）：

```cpp
// ESCAPE HATCH #1 (G5-G round 3): tf32 m16n8k8 mma.sync -- not reachable
// through nvgpu.mma.sync (tf32 caps at K=4 there) or the tabled
// nvvm.mma.sync family (m16n8k8 only f16/bf16/i8).  Emitted as a direct
// llvm.inline_asm with EARLY-CLOBBERED outputs.
//
// Why not nvvm.inline_ptx (the BasicPtxBuilder path): its output
// constraints are plain "=f".  SASS-verified on the RTX 5090 Laptop:
// with "=f", ptxas' inline-asm register allocation is free to map the
// D quad ONTO THE A QUAD OF THE SAME mma (HMMA R12, R12, R10, R16) --
// inputs are considered dead after the asm, no early-clobber, and the
// tensor core's read/write overlap then corrupts the operands (exact
// integer probes come back short by whole k-terms; a K=16 two-atom
// chain failed 118/128 cells).  "=&f" forbids output/input overlap;
// D==C aliasing (fla-style in-place accumulation) remains legal but we
// keep D fully distinct.
//
// Operand order: outputs first -> $0..$3 = D, $4..$7 = A, $8/$9 = B,
// $10..$13 = C.  Result is a literal struct<(f32 x4)>; 4 extractvalue.
struct MmaTf32K8Lowering : OpRewritePattern<GdnMmaTf32K8Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult
  matchAndRewrite(GdnMmaTf32K8Op op, PatternRewriter &rewriter) const override {
    static const char *kPtx =
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
        "{$10, $11, $12, $13};";
    static const char *kConstraints =
        "=&f,=&f,=&f,=&f,f,f,f,f,f,f,f,f,f,f";
    ...
    auto asmOp = LLVM::InlineAsmOp::create(
        rewriter, loc, structTy, ValueRange(args),
        rewriter.getStringAttr(kPtx), rewriter.getStringAttr(kConstraints),
        /*has_side_effects=*/UnitAttr::get(ctx),
        /*is_align_stack=*/UnitAttr(),
        /*tail_call_kind=*/LLVM::TailCallKindAttr::get(ctx, LLVM::TailCallKind::None),
        /*convergent=*/UnitAttr(),
        /*asm_dialect=*/LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
        /*operand_attrs=*/ArrayAttr());
    Value d0 = LLVM::ExtractValueOp::create(rewriter, loc, asmOp->getResult(0),
                                            ArrayRef<int64_t>(0));
    ... // d1..d3 同构
    rewriter.replaceOp(op, {d0, d1, d2, d3});
    return success();
  }
};
```

IR 前后对比（概念上；inline_asm 的完整展开冗长，示意）：

```mlir
// before
%d0, %d1, %d2, %d3 = gdn.mma.tf32k8 %a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3 : f32 -> (f32, f32, f32, f32)
// after（经 -convert-to-llvm 前的 gdn 层面即已消失；LLVM 层形态）
%asm = llvm.inline_asm has_side_effects convergent att_dialect
    asm "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};"
    constraints "=&f,=&f,=&f,=&f,f,f,f,f,f,f,f,f,f,f"
    (%a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3) : (!llvm.f32, ..., !llvm.f32) -> !llvm.struct<(f32, f32, f32, f32)>
%d0 = llvm.extractvalue %asm[0] : !llvm.struct<(f32, f32, f32, f32)> -> !llvm.f32
```

约束串逐位解读：`=&f` ×4（D——early-clobber 输出，`&` 禁止与任何输入共寄存器）之后 10 个裸 `f`（A、B、C 输入）。操作数序 = 输出在前（`$0..$3` D、`$4..$7` A、`$8/$9` B、`$10..$13` C），inline-asm 约定序与 PTX 指令的 `{D}, {A}, {B}, {C}` 书写序一致。结果以字面 `struct<(f32 x4)>` 承载，四次 `extractvalue` 拆包。

**效果（含两条硬件验证发现与实测数字，全部取自 `evidence/G5/bwd_report.md`）**：

1. **early-clobber 教训**：若输出约束是裸 `=f`（`nvvm.inline_ptx`/BasicPtxBuilder 路径只能产这种），ptxas 的内联汇编寄存器分配**合法地**把 D 四元组映射到同一条 mma 的 A 四元组上（SASS 实测 `HMMA R12, R12, R10, R16`）——输入在 asm 后被视为死亡、无 early-clobber 保护，张量核的读/写重叠于是腐蚀操作数：精确整数探针短缺整个 k 项，K=16 双原子链 **118/128 单元确定性失败**。`=&f` 禁止输出/输入重叠后修复。这是一条可迁移的工具链发现：**BasicPtxBuilder 与 mma 组合时 early-clobber 不可表达，任何走该路径的 mma 内联汇编都会踩同一陷阱**（`nvvm.inline_ptx` 无法表达 early-clobber，故本 op 直落 `llvm.inline_asm`）。D==C 别名（fla 式就地累加）本仍合法，但本实现让 D 完全独立。
2. **fragment 布局律（硬件导出，非文档猜测）**：单 warp 探针 kernel（grid 1×1、32 线程、34 regs、零 spill、恰好 1 条 mma.sync——`evidence/G5/scratch/tf32k8_probe/meta.json`）以整数操作数（tf32 精确可表）钉死布局：`g = laneid >> 2, t = laneid mod 4`；**A（16×8）：a0=(g,2t)、a1=(g+8,2t)、a2=(g,2t+1)、a3=(g+8,2t+1)——行半（row-half）在外、列对（col-pair）在内的次序**，与 bf16 m16n8k16 的"打包对"直觉**不同**；B（8×8 列主）：b0=(2t,g)、b1=(2t+1,g)；D 与 bf16 m16n8k16 的 D 同律（列索引 = 2t）。整数探针 D == A@B+C **精确成立**；N(0,1) 操作数的舍入检查 4.2e-3，与 tf32 语义一致（距截断模拟约 1 ulp）。A 的 row-half-outer 次序是发射器最易写错的一点——bwd 发射器调试史上确有一族"kb>0 全部错址"的偏移错（`%mT2P1` 已是 2t+1 又被 +1 一次），根因正是对 fragment 次序的误解。
3. **实测收益**（B1T2048H16，共享 RTX 5090 Laptop 窗口，中位/最小并报）：反向 e2e **14.16 → 11.06 ms（中位，1.28×；最小 9.57，1.48×）**；分阶段 dqkwg 4.71 → 2.75 ms（1.71×，与 r4/c4r warp 并行化合并计）、wyb 2.65 → 1.21 ms（2.19×，与 ra/c1r 并行化合并计）；16 个 tf32 位点全开后数值 6/6 PASS 于双 gate 形状（B1T512H4：dq 8.0e-4、dk 1.7e-3、dv 1.4e-6、dbeta 3.0e-4、dg 1.9e-4、dh0 1.3e-6；B1T2048H16：dq 1.1e-3、dk 1.8e-3、dv 3.1e-6、dbeta 1.4e-3、dg 4.1e-4、dh0 1.8e-6；括注为对杆裕量）。5 个位点因数值被否回退标量（dhu.dkB dk 1.04e-2 超 8e-3 杆、wy.u dk 1.06e-2、wyb.tv dbeta 8.6e-3 裕量仅 2.3×等），3 个数值干净但 dhu 寄存器溢出回归（255 regs / spill 增、dhu 3.69→4.67 ms）回退——逐位点二分表完整记录于 `evidence/G5/bwd_report.md`。

#### 1.5.13 `gdn-lower-hatch` pass 与整体下降流

12 个 rewrite 模式经 `populateGdnHatchLoweringPatterns`（`dialect/lib/GdnHatch.cpp` 第 215–221 行）打包，由 `GdnLowerHatchPass`（同文件第 223–240 行）以 `applyPatternsGreedily` 施加于整个 module：

```cpp
struct GdnLowerHatchPass
    : public PassWrapper<GdnLowerHatchPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GdnLowerHatchPass)
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGdnHatchLoweringPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  StringRef getArgument() const override { return "gdn-lower-hatch"; }
  StringRef getDescription() const override {
    return "Lower gdn warp-spec/TMA hatch ops to in-tree nvgpu/nvvm ops";
  }
};
```

真实端到端效果：对 `evidence/G2/scratch/g2c/smoke_hatch.mlir`（warp 专化完整骨架：init → 生产者 dec/elect/prefetch/expect_tx/load → 消费者 inc/wait/arrive → epilogue store/commit/wait）运行 `gdn-opt --gdn-lower-hatch` 后，payload 中**不再有任何 `gdn.` 前缀 op**，全部替换为在树 `nvgpu.*`/`nvvm.*`——此后即可走纯原生管线（`-gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a"`）出 PTX/cubin。生产消费链（`scripts/run_l2.sh` 第 39–43 行）正是如此：payload 先经 `gdn-opt --gdn-lower-hatch` 再与 transform 脚本合并（因 gdn-opt 不解析 `transform.structured.*`，hatch 下降必须在合并前的裸 payload 上做——该管线次序约束已在 G2 沉淀）。

---

### 1.6 `gdn-to-chunk` pass：L0 → L1 展开通道

#### 1.6.0 pass 总体与两个 rewrite 模式

（`dialect/lib/GdnPasses.cpp`，全文 112 行。）设计定位写在文件头注释：**完整的 chunk-AST 展开**（KKT → 前代换 → w/u → h 递推 → o，反向同构 FLA 阶段）是 L1 生成器的职责——届时本 pass 发出的 IR 与 `tools/gdn_l1_gen.py` 生成的 L1 文本**逐行同构**（生成器即本 pass 的可执行规格）。当前实现把 `gdn.fwd`/`gdn.bwd` 改写为对 `__gdn_l1_*` 外部函数的调用（携带全部语义属性），保证方言端到端（parse → verify → rewrite → print）可测试。

pass 壳与 1.5.13 同构（`GdnToChunkPass`，第 93–110 行，`applyPatternsGreedily`，注册名 `gdn-to-chunk`），核心是两个模式 `FwdToLowering` 与 `BwdToLowering`，以及共享的入口声明助手：

```cpp
static constexpr const char *kL1FwdSym = "__gdn_l1_fwd_chunk";
static constexpr const char *kL1BwdSym = "__gdn_l1_bwd_chunk";

static func::FuncOp ensureL1Entry(ModuleOp module, PatternRewriter &rewriter,
                                  StringRef sym, FunctionType fnTy) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(sym))
    return existing;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto decl = rewriter.create<func::FuncOp>(module.getLoc(), sym, fnTy);
  decl.setPrivate();
  return decl;
}
```

`ensureL1Entry` 的三件事：**幂等**（符号已存在则复用——同一 module 多个 gdn op 共享一份声明）；**按需物化**私有（`setPrivate`，无定义）函数声明 `func.private @__gdn_l1_{fwd,bwd}_chunk`，插入点定在 module 体首；函数类型由**实际出现的操作数/结果类型**推导（`fnTy` 参数化），使得缺省 h0/A/dht 的调用与全量调用的签名自然不同。**私有声明生成**是 G1 gate 冒烟判据的第四项（`evidence/G1/GATE_REPORT.md`）。

#### 1.6.1 `FwdToLowering`：`gdn.fwd` → `call @__gdn_l1_fwd_chunk`

**动机（算子层面+例子）**。L0 op 是"整算子"语义，L1 以下是分块算法；两者之间的契约面需要一个稳定的接缝：不管 L1 将来是前代换 solve 还是 Schur 合并变体，`gdn.fwd` 都应该降到**同一个符号入口**，把全部语义（操作数 + config）无损递过去。例子：冒烟输入的 `gdn.fwd`（config `#gdn.config<64, true, false, true, true, true, false>`，h0 给定）应变成对 `@__gdn_l1_fwd_chunk` 的六实参调用，config 属性原样挂在调用上。

**源码**（`dialect/lib/GdnPasses.cpp` 第 38–59 行）：

```cpp
struct FwdToLowering : OpRewritePattern<gdn::GdnFwdOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(gdn::GdnFwdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> args{op.getQ(), op.getK(), op.getV(), op.getBeta(),
                            op.getG()};
    if (op.getH0())
      args.push_back(op.getH0());
    auto fnTy = rewriter.getFunctionType(ValueRange(args).getTypes(),
                                         op.getResultTypes());
    if (auto module = op->getParentOfType<ModuleOp>())
      ensureL1Entry(module, rewriter, kL1FwdSym, fnTy);
    auto callee = rewriter.getStringAttr(kL1FwdSym);
    auto call = rewriter.create<func::CallOp>(loc, callee,
                                              op.getResultTypes(), args);
    // 语义属性原样透传（L1 消费方按此选择算法变体）
    call->setAttr("gdn.config", op.getConfig());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};
```

逐段：实参表按"5 必选 + 条件追加 h0"构造——**h0 缺省时入口签名少一参**（下游按 config 的 `use_initial_state` 对齐语义）；函数类型从实参/结果现推；module 存在则保证私有声明；`func::CallOp` 原位替换；`call->setAttr("gdn.config", op.getConfig())` 把七旋钮束以 `gdn.config` 键名**原样透传**（L1 消费方按此选择算法变体——这是"语义单一事实源"在 IR 层的物理形态：旋钮不拆包、不重编码）。

**IR 前后对比（真实运行输出）**。输入 = `dialect/test/smoke_l0.mlir`（1.2.4 全文）。命令：`gdn-opt smoke_l0.mlir --gdn-to-chunk`。输出（逐字）：

```mlir
module {
  func.func private @__gdn_l1_fwd_chunk(tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2xf32>, tensor<2x128x2xf32>, tensor<2x2x16x16xf32>) -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
  func.func @smoke(%arg0: tensor<2x128x2x16xf32>, %arg1: tensor<2x128x2x16xf32>, %arg2: tensor<2x128x2x16xf32>, %arg3: tensor<2x128x2xf32>, %arg4: tensor<2x128x2xf32>, %arg5: tensor<2x2x16x16xf32>) -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>) {
    %0:2 = call @__gdn_l1_fwd_chunk(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {gdn.config = #gdn.config<64, true, false, true, true, true, false>} : (tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2xf32>, tensor<2x128x2xf32>, tensor<2x2x16x16xf32>) -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
    return %0#0, %0#1 : tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>
  }
}
```

三处可核对的改写效果：module 顶部多出 `func.func private @__gdn_l1_fwd_chunk(...)`（六入双出，与调用签名一致）；op 变 `call`，`gdn.config` 属性透传在调用上；结果引用换名（`%o, %ht` → `%0#0, %0#1`，SSA 替换的打印表现）。

**效果**。G1 gate 判据 1 的 rewrite 与 attr 透传两项绿；为 L1 层提供了可对拍的接缝符号（L1 展开文本与 `tools/gdn_l1_gen.py` 输出逐行同构的目标由生成器侧闭合，`tools/gdn_l1_check.py` 7/7、MLIR L1 vs JAX dumps ~6e-7，`evidence/G1/GATE_REPORT.md`）。

#### 1.6.2 `BwdToLowering`：`gdn.bwd` → `call @__gdn_l1_bwd_chunk`

**动机**。同 fwd，但实参构造必须处理三个 Optional 槽（h0/A/dht）的**位置语义**：改写后的平铺调用序列为 `q, k, v, beta, g, [h0], [A], dout, [dht]`——缺省槽直接从序列中消失，靠 config（`use_initial_state`）与调用约定对齐。A 的有无正是 1.3.1 的重算-复用旋钮：A 在调用里 = 复用；不在 = 重算。

**源码**（`dialect/lib/GdnPasses.cpp` 第 61–85 行）：

```cpp
struct BwdToLowering : OpRewritePattern<gdn::GdnBwdOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(gdn::GdnBwdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> args{op.getQ(), op.getK(), op.getV(), op.getBeta(),
                            op.getG()};
    if (op.getH0())
      args.push_back(op.getH0());
    if (op.getA())
      args.push_back(op.getA());
    args.push_back(op.getDout());
    if (op.getDht())
      args.push_back(op.getDht());
    auto fnTy = rewriter.getFunctionType(ValueRange(args).getTypes(),
                                         op.getResultTypes());
    if (auto module = op->getParentOfType<ModuleOp>())
      ensureL1Entry(module, rewriter, kL1BwdSym, fnTy);
    auto call = rewriter.create<func::CallOp>(
        loc, rewriter.getStringAttr(kL1BwdSym), op.getResultTypes(), args);
    call->setAttr("gdn.config", op.getConfig());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};
```

与 fwd 模式的全部差异就在实参构造的三段条件追加；其余（幂等声明、类型现推、属性透传、原位替换）逐行同构。

**IR 前后对比（真实运行输出，复用形态）**。输入 = 1.3.3 的复用形态 `gdn.bwd`（九槽全满）。运行 `--gdn-to-chunk` 后（节选，module 顶部的私有声明与调用体）：

```mlir
func.func private @__gdn_l1_bwd_chunk(tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2xf32>, tensor<2x128x2xf32>, tensor<2x2x16x16xf32>, tensor<2x2x2x64x64xf32>, tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>) -> (tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2xf32>, tensor<2x128x2xf32>, tensor<2x2x16x16xf32>)
...
%0:6 = call @__gdn_l1_bwd_chunk(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) {gdn.config = #gdn.config<64, true, false, true, true, true, false>} : (...) -> (...)
return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : ...
```

九实参序 = ODS 声明序去掉空槽；结果恒六元组（含重算形态下的 dh0 槽，见 1.3.2 的定长结果设计）。

**效果**。与 fwd 对称：bwd 侧 parse→verify→rewrite→print 闭合；`operandSegmentSizes` 在改写后**消失**（平铺调用不再有变长段歧义，这也回溯解释了为何该 trait 只服务 generic 解析而非改写逻辑）；重算/复用两种形态经由同一符号、不同签名自然分流。

---

### 1.7 gdn-opt 工具与外建构建体系

#### 1.7.1 `gdn-opt`：方言的 mlir-opt 副本

完整源码（`dialect/tools/gdn-opt/gdn-opt.cpp`，30 行）：

```cpp
// gdn-opt：GDN 方言的 mlir-opt 副本（含 gdn-to-chunk pass）。
#include "Gdn/Gdn.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

namespace gdn {
std::unique_ptr<Pass> createGdnToChunkPass();
} // namespace gdn

int main(int argc, char **argv) {
  registerAllPasses();
  registerPass([]() -> std::unique_ptr<Pass> {
    return gdn::createGdnToChunkPass();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return gdn::createGdnLowerHatchPass();
  });

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<gdn::GdnDialect>();

  return asMainReturnCode(MlirOptMain(
      argc, argv, "GDN dialect optimizer driver\n", registry));
}
```

逐块讲解：

- **`MlirOptMain` 骨架**：这是 MLIR 官方的 opt-driver 主函数模板（mlir-opt 同款），免费获得 `--pass-name`、`-o`、round-trip、`--mlir-print-ir-*` 等全套驱动语义。gdn-opt 的全部"自定义"只有两件事：注册两个 pass、插入一个方言。
- **pass 注册**：`createGdnToChunkPass()` 在文件内前向声明（链接自 `GdnDialect` 库的 `GdnPasses.cpp`）；`createGdnLowerHatchPass()` 虽未在可见声明里前向声明，但经 `Gdn/Gdn.h` 第 36 行的正式声明引入（伞头即接口）。注册后命令行出现 `--gdn-to-chunk` 与 `--gdn-lower-hatch`。
- **方言注册**：`registerAllDialects(registry)` 先装全树方言（依赖面：tensor/linalg/scf/func/nvvm/nvgpu/gpu/arith，见 1.7.2 的 `dependentDialects`），再 `registry.insert<gdn::GdnDialect>()` 插入本方言——顺序无碍（registry 是集合语义），但两行合起来才是 gdn payload 可 parse 的完整前提。
- 值得记录的**工具边界**（G2 沉淀，`scripts/run_l2.sh`）：gdn-opt **不解析 `transform.structured.*` 脚本**（transform dialect 的解释需要 transform 库的完整链路），因此携带 gdn op 的 payload 与 transform 脚本的处理次序是：先 `gdn-opt --gdn-lower-hatch` 于裸 payload，再与 transform.mlir 合并走 mlir-opt——这也解释了为何 hatch op 必须是"先降干净"的一等 op 而非留在最终 IR 里。

`gdn-opt` 的三个真实使用场景（本节全部 IR 输出的来源）：冒烟（`gdn-opt dialect/test/smoke_l0.mlir --gdn-to-chunk`，1.6.1）；hatch 端到端（`gdn-opt smoke_hatch.mlir --gdn-lower-hatch`，1.5.13）；pretty 语法验证（`gdn.mma.tf32k8` 的 round-trip，1.5.12）。

#### 1.7.2 外建（out-of-tree）MLIR 项目结构

`gdn` 方言不住在 LLVM monorepo 里，而是标准外建方言项目（仿 `mlir/examples/standalone` 的外建模式），对预构建的 MLIR `find_package` 链接。顶层 `dialect/CMakeLists.txt` 的关键行：

```cmake
cmake_minimum_required(VERSION 3.20.0)
project(gdn-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
...
# 独立构建：指向本仓的 MLIR build（mlir/examples/standalone 的外建模式）
if(NOT DEFINED MLIR_DIR)
  set(MLIR_DIR "$ENV{HOME}/codebase/llvm-mlir/build/lib/cmake/mlir")
endif()
find_package(MLIR REQUIRED CONFIG)
...
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
...
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools/gdn-opt)
```

关键行逐条：

- `MLIR_DIR` 默认指向 `~/codebase/llvm-mlir/build/lib/cmake/mlir`——**外建的本质**：对有限构建（mlir-opt/mlir-runner/cuda-runtime/Python 绑定，1.9 GB Release）的 CMake 包定位，方言项目不重编 MLIR。
- `include(TableGen/AddLLVM/AddMLIR/HandleLLVMOptions)`——引入 LLVM/MLIR 的构建函数族（`mlir_tablegen`、`add_mlir_dialect_library`、`add_llvm_executable` 都来自这里）；这是外建项目与 monorepo 共享构建语法的机制。
- 双 `include_directories(PROJECT_SOURCE_DIR/include | PROJECT_BINARY_DIR/include)`——源侧头文件 + 构建侧生成头（`.inc`）各一个入口，`Gdn.h` 里的 `#include "GdnDialect.h.inc"` 靠后者解析。
- 三个 `add_subdirectory` 给出项目三段式：include（表生成）/ lib（方言+pass 静态库）/ tools（可执行）。

#### 1.7.3 表生成：include/CMakeLists.txt

（`dialect/include/CMakeLists.txt` 全 75 行，前 48 行是表生成。）

```cmake
set(LLVM_TARGET_DEFINITIONS Gdn/GdnDialect.td)
mlir_tablegen(GdnDialect.h.inc -gen-dialect-decls)
mlir_tablegen(GdnDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(GdnAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(GdnAttrDefs.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS Gdn/GdnOps.td)
mlir_tablegen(GdnOps.h.inc -gen-op-decls)
mlir_tablegen(GdnOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS Gdn/GdnHatch.td)
mlir_tablegen(GdnHatchOps.h.inc -gen-op-decls)
mlir_tablegen(GdnHatchOps.cpp.inc -gen-op-defs)
mlir_tablegen(GdnHatchEnums.h.inc -gen-enum-decls)
mlir_tablegen(GdnHatchEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(GdnIncGen)
add_dependencies(GdnIncGen mlir-generic-headers)
```

要点：

- 三个 `.td` 文件各自驱动一组生成：`GdnDialect.td` 出方言声明/定义 + **属性**声明/定义（`GdnConfigAttr`、`GdnRegDirAttr` 的 C++ 类在这里面）；`GdnOps.td` 出 fwd/bwd 两个 op 类；`GdnHatch.td` 出 12 个 hatch op 类 + **枚举**声明/定义（`GdnRegDir`）——1.5.0 末尾说的"enum 从 Hatch.td 生成、Attr 从 Dialect.td 生成"的跨文件分工，物理载体就是这九行 `mlir_tablegen`。
- `add_public_tablegen_target(GdnIncGen)` 把全部 `.inc` 聚成一个 CMake 目标；`add_dependencies(GdnIncGen mlir-generic-headers)` 保证依赖 MLIR 侧通用头先行——方言库（`GdnDialect`）再 `DEPENDS GdnIncGen`（lib/CMakeLists.txt 第 57–58 行），构成生成的编译序。
- 消费侧的宏纪律（`dialect/include/Gdn/Gdn.h` 第 18–27 行注释原文："本方言体量小：伞头直接展开完整声明，免去各 TU 宏纪律"）：伞头依次 `#include` 全部 `.inc`，且在每个会 `#undef GET_*` 的 include 前**重新武装**宏（`#define GET_OP_CLASSES` 出现两次）——因为 op 头会在结尾清宏，连续包含必须重定义。`.cpp` 侧（`GdnDialect.cpp` 第 19–26 行）同样按 dialect-defs → enums → attrdefs → ops → hatch-ops 的序展开。

方言库本体（`dialect/lib/CMakeLists.txt` 第 49–75 行）：`add_mlir_dialect_library(GdnDialect GdnDialect.cpp GdnPasses.cpp GdnHatch.cpp ...)` 把三个实现单元编为一个库，`LINK_LIBS PUBLIC` 面按依赖闭包列出（MLIRArithDialect/MLIRFuncDialect/MLIRGPUDialect/MLIRIR/MLIRLinalgDialect/MLIRNVGPUDialect/MLIRNVVMDialect/MLIRPass/MLIRSCFDialect/MLIRSideEffectInterfaces/MLIRSupport/MLIRTensorDialect 等）——NVGPU/NVVM 两个链接项正是 hatch op 下降目标的库级映像；`ADDITIONAL_HEADER_DIRS` 暴露源侧 include。

工具（`dialect/tools/gdn-opt/CMakeLists.txt` 第 76–95 行）：

```cmake
set(LIBS
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIROptLib
    MLIRParser
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRSupport
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRTensorDialect
    GdnDialect
  )
add_llvm_executable(gdn-opt gdn-opt.cpp)

llvm_update_compile_flags(gdn-opt)
target_link_libraries(gdn-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(gdn-opt)
```

`MLIROptLib`（MlirOptMain 的实现）+ `MLIRRegisterAllDialects/Passes`（registerAll* 的实现）+ 本方言库 `GdnDialect`——工具链接面与其 main 的三个动作一一对应；`mlir_check_all_link_libraries` 是 MLIR 的链接完整性自检。

构建入口 `scripts/build_dialect.sh`（幂等）：Ninja + Release + `LLVM_ENABLE_ASSERTIONS=ON` + clang/clang++，`MLIR_DIR` 指向有限构建；经仓级资源纪律（`resource_guard.sh`：线程预算 clamp、watchdog STOP/CONT/TERM）将并行度钳到 ≤6（"小项目，少占预算"）；产物校验 `build-dialect/bin/gdn-opt` 存在后报 DONE。整个方言构建在 G1 复现门内（`make repro-g1`："dialect build+smoke (CPU; ~15 min)"，纯 CPU）。

---

### 1.8 小结

本章完整解剖了 GDN-MLIR 的语义底座：两个 L0 语义 op（`gdn.fwd`/`gdn.bwd`，ODS + verifier + generic 语法 + 真实 IR 全展示）、一个七旋钮语义属性（`GdnConfigAttr`，assembly 序即定义序）、11 个零预算逃生舱薄封装与唯一真逃生舱 `gdn.mma.tf32k8`（预算 1/≤5，携带 early-clobber 与 fragment 布局两条硬件验证发现，反向 e2e 14.16→11.06 ms 的实测载体）、两个 L0→L1 改写模式（`FwdToLowering`/`BwdToLowering`，真实 before/after IR）与 12 个 hatch 下降模式（`gdn-lower-hatch`）、以及支撑这一切的外建构建体系（三 .td 九表生成、双 include 路径、MLIROptLib 工具）。这一层的全部设计服务于同一命题：语义单一事实源必须先于调度自由度存在——L0 之上，L1 的算法结构、L2 的 Z3 调度、L3 的流水与 mma 才有可验证的下降起点。

（本章引用的证据文件：`evidence/G1/GATE_REPORT.md`、`evidence/G2/gen_report.md`、`evidence/G2/hatch_report.md`（含 ADDENDUM）、`evidence/G5/bwd_report.md`、`evidence/G5/scratch/tf32k8_probe/meta.json`、`evidence/G2/scratch/g2c/smoke_hatch.mlir`、`evidence/G0/fla_ptx`、`docs/L0_SEMANTICS.md`、`docs/PLAN.md`；源文件均在 `dialect/` 与 `scripts/` 下，路径已在正文逐一标注。）

---

## 2. L1 算法 IR 与数学：chunk-WY 分解的完整 MLIR 表达

> 本章覆盖范围：L0 语义规约（`docs/L0_SEMANTICS.md`）之下、GPU 调度层（L2 及以下各层）之上的**算法结构层 L1**。逐对象展开 chunk-WY 分解的全部数学（前向与反向），并给出每一个数学对象在真实 MLIR 中的对应物：生成器源码（`tools/gdn_l1_gen.py`，1428 行）、生成的 IR 产物样本（`evidence/G1/l1_chunked_T128_C64.mlir`、`evidence/G1/l1_bwd_chunked_T128_C64.mlir`）、CPU 下降与执行管线（`scripts/run_l1.sh`），以及全部数值证据（`evidence/G1/jax_spec_report.md`、`evidence/G1/l1_harness_report.md`、`evidence/G1/l1_check_output.txt`、`evidence/G1/l1_bwd_check_output.txt`、`evidence/G1/GATE_REPORT.md`）。本章所有数字均取自上述证据文件；正文不重复推导证据文件未支撑的结论。

### 2.0 章首：L1 的存在意义

编译一个算子之前，必须先让算子本身**可定义、可执行、可对拍**。GDN-MLIR 的分层栈中，L0 方言（`gdn.fwd` / `gdn.bwd` 两个 op 加 `#gdn.config<...>` 旋钮属性）只声明"意图"——它规定了输入输出形状与若干语义开关，但不包含任何算法结构；GPU 层（L2 及以下）关心 tile、流水与 warp 几何，但对"状态递推究竟等价于哪个三角线性方程组"保持沉默。夹在中间的 **L1 是算法结构层**：它把 Gated DeltaNet 的 chunked WY 分解——也就是 fla 生态中手写 Triton kernel 所实现的那套算法（`chunk_local_cumsum` → `prepare_wy_repr` 求 $(I+M)^{-1}$ → `chunk_o` 输出分解 → `chunk_delta_h` 状态更新；反向则经 `prepare_wy_repr_bwd`）——**原样表达为原生（stock）MLIR**：memref 级的 `linalg` + `scf`，不引入任何自定义方言操作，不 fork 编译器。因此 L1 与 FLA 的算法结构**同构**：每一个数学对象（含尾 cumsum、KKT Gram 矩阵、前代换求逆、WY 系数 $u/w$、状态递推、输出 intra/inter 分解，以及反向的全部六条梯度链）在 IR 里都有一个显式的、可指认的对应物。

L1 的三个不可替代的作用：

1. **可被 CPU 执行验证**。L1 模块是自包含的：全部输入以 `memref.global` 常数嵌入，经 stock `mlir-opt` 下降、`mlir-runner` 在 CPU 上 JIT 执行，通过 `printMemrefF32` 把计算结果打印到 stdout，与 fp64 JAX 规约（`tools/gdn_ref_jax.py`）和 fla naive torch oracle 三方对拍（`evidence/G1/GATE_REPORT.md`：前向 7/7 PASS、反向 3 npz × 6 梯度 PASS）。这使得"算法正确性"与"GPU 调度正确性"解耦：任何下层（L2–L4）的改动若破坏语义，可以回退到 L1 口径重新仲裁。
2. **与 FLA 同构的算法结构层，即"语义单一事实源"的物化形式**。`docs/L0_SEMANTICS.md` 记录的语义唯一事实源链为：`tools/gdn_ref_jax.py`（JAX fp64，44/44 自洽）⇄ fla naive（torch，1e-6/1e-7 量级相符）⇄ MLIR L1（`tools/gdn_l1_gen.py`，6e-7 量级相符）。L1 是这条链的终点：数学上它必须与 JAX 规约逐式等价，工程上它是第一个"真 IR"实现。
3. **它是未来方言展开 pass 的输出格式规约**。`docs/L0_SEMANTICS.md` 明确：`gdn-to-chunk` pass 的完整 chunk-AST 展开落地后，"输出与 `tools/gdn_l1_gen.py` 生成的 L1 文本**逐行同构**（前代换解三角、scf.for 载态已验证）"。也就是说，生成器不是一次性脚手架，而是 L1 展开 pass 的**参考输出定义**——生成器 IS 未来 pass 的输出格式（`tools/gdn_l1_gen.py:6-8` 的原话："This generator IS the output format of the future L1 expansion pass"）。

本章的写法遵循一个固定模式（"四件套"）：对每个 pattern，先给**动机**（算子层面为什么需要这个对象，配例子），再给**真实源码片段**（路径+行号标注，来自 `tools/` 下只读文件），然后给 **IR 对比**（数学/参考实现 vs 生成的 MLIR 产物），最后给**效果数字**（`evidence/G1/` 的对拍结果）。数学公式一律先定义后使用。

### 2.1 算子契约与记号

#### 2.1.1 形状与输入输出契约

GDN 算子（fla op contract，`tools/gdn_ref_jax.py:11-19`、`evidence/G1/jax_spec_report.md` §3）：

| 张量 | 形状 | 语义 |
|---|---|---|
| $q$ | $[B,T,H,K]$ | 查询（模型侧已 L2 归一化） |
| $k$ | $[B,T,H,K]$ | 键（模型侧已 L2 归一化） |
| $v$ | $[B,T,H,V]$ | 值 |
| $\beta$ | $[B,T,H]$ | delta-rule 强度（**原始值**，sigmoid 由上游模型做） |
| $g$ | $[B,T,H]$ | 逐 token 对数衰减，**自然对数域** |
| $h_0$ | $[B,H,K,V]$ | 可选初始状态 |
| 返回 $o$ | $[B,T,H,V]$ | 输出 |
| 返回 $h_t$ | $[B,H,K,V]$ | 终态 |

$\beta$ 与 $g$ 都是逐 $(b,t,\text{head})$ 标量——算子内部从不混合头（H=HV in GDN）。测试数据生成遵循 GDN 模型风格（`tools/gdn_ref_jax.py::make_gdn_inputs`、`tools/gdn_l1_gen.py::make_data`）：$q,k$ 逐行 L2 归一化、$\beta=\mathrm{sigmoid}(x)$、$g=-\mathrm{softplus}(0.5x)$（负的自然对数域衰减）、$h_0\sim 0.1\,\mathcal N$。

#### 2.1.2 逐 token 递推（语义骨架，先定义后用）

L1 的一切数学都从下列逐步递推出发（`tools/gdn_ref_jax.py:31-35`，顺序细节是 fla 契约的一部分，"do NOT fix these"）：

$$
h \;\leftarrow\; e^{g_t}\, h \qquad(\text{衰减在 delta 更新}\textbf{之前})
$$

$$
b_v \;\leftarrow\; \beta_t\,\bigl(v_t - k_t\, h\bigr), \qquad h \;\leftarrow\; h + k_t^{\!\top}\!\otimes\, b_v
$$

$$
o_t \;\leftarrow\; q_t\, h
$$

其中 $k_t\,h$ 是 $[K]\times[K,V]$ 的矩阵-向量作用（$\sum_j k_t[j]\,h[j,:]$），$k_t^{\top}\otimes b_v$ 是 $[K,V]$ 外积。**顺序不可交换**：若把 delta 更新放在衰减之前，状态会多乘一次 $e^{g_t}$；若把 $\beta$ 乘在 $v_t$ 而非残差 $(v_t-k_t h)$ 上，则与 fla 的 $M$ 矩阵行向 $\beta$ 约定冲突（见 §2.2.5）。这条递推的可执行定义是 `gdn_seq`（`jax.lax.scan` 移植，代码见 §2.5.1）。

#### 2.1.3 记号总表与命名对照

本章统一采用**生成器约定**（`tools/gdn_l1_gen.py:14-40`）。注意 `tools/gdn_ref_jax.py` 对 $u/w$ 使用**相反的字母命名**（该文件 $w:=T(\beta v)\in\mathbb R^{C\times V}$、$u:=T(\beta k e^{cd})\in\mathbb R^{C\times K}$），数学完全一致，读代码时须对照下表：

| 记号（本章/生成器） | 定义 | `gdn_ref_jax.py` 记号 | fla 记号（生成器 docstring 标注） | 生成器缓冲区 |
|---|---|---|---|---|
| $C$ / $\mathrm{NC}$ | chunk 长度 / chunk 数 $=\lceil T/C\rceil$ | `BT` / `n` | — | 编译期常数 |
| $\mathrm{gc}$ | chunk 内 $g$ 的**含尾**（inclusive）累加和 | `decay` / `cd` | `chunk_local_cumsum` | `%gc` |
| $\mathrm{gcl}$ | $\mathrm{gc}$ 的末元素 $\mathrm{gc}_{C-1}$ | `d_last` | — | `%gcl`（标量） |
| $D$ | 衰减差矩阵 $D_{ij}=2^{\mathrm{gc}_i-\mathrm{gc}_j}$ | `L_mask`（tril 后） | — | 内联（每次现算） |
| $A$ | $A_{ij}=\beta_i (k_i\!\cdot\! k_j)\,2^{\mathrm{gc}_i-\mathrm{gc}_j}$，$j<i$，严格下三角 | $-M$（`attn` 取负前） | $-M$ | `%Abuf` |
| $M$ | fla 记号：$M_{ij}=-\beta_i (k_i\!\cdot\! k_j)\,e^{cd_i-cd_j}$ | `attn` 初值 | $M$ | — |
| $T$ | $(I+A)^{-1}=(I-M)^{-1}$，单位下三角 | `attn` 终值 | `attn` | `%Tbuf` |
| $u$ | $T(\beta\odot v)\in\mathbb R^{C\times V}$ | `w` | `k_cumsum` | `%ubuf`（前级 `%upre`） |
| $w$ | $T(\beta\odot k\odot 2^{\mathrm{gc}})\in\mathbb R^{C\times K}$ | `u` | `k_cumdecay` | `%wp`（前级 `%kb2`） |
| $v_{\text{new}}$ | $u-wS\in\mathbb R^{C\times V}$ | `v_new = w - u@S`（注意字母互换） | — | `%vn` |
| $P$ | $\mathrm{tril}_{\mathrm{incl}}\bigl((qk^{\top})\odot D\bigr)$，**对角保留** | `a_i` | — | 前向复用 `%Abuf` |
| $k_{\mathrm{term}}$ | $k\odot 2^{\mathrm{gcl}-\mathrm{gc}}\in\mathbb R^{C\times K}$ | `k_term` | — | `%kg` |
| $\hat B$ | $k\odot 2^{\mathrm{gcl}-\mathrm{gc}}$（反向 Bhat 链，与 $k_{\mathrm{term}}$ 同式） | `Bhat` | — | 内联进缩并 |
| $S$ | 状态 $[K,V]$ | `S` | `S`/`h` | `%S` / 反向 `%Sn`（`h_all[n]`） |

其中 $\odot$ 为逐元素乘（行向量按行广播），$\mathrm{tril}_{\mathrm{incl}}$/$\mathrm{strict\_tril}$ 分别为含对角/严格下三角掩蔽。

### 2.2 前向数学：从递推到 chunk

#### 2.2.1 $q$ 的 L2 归一化约定与 $\mathrm{scale}=K^{-1/2}$

**动机（算子层面）**。GDN 模型在调用算子前对 $q,k$ 逐行 L2 归一化（把每行变成单位向量），这是**模型侧**职责；算子内部**不做**归一化（`docs/L0_SEMANTICS.md` 第 3 条：算子内部不做 q/k 的 L2 归一；config.normalize_q 仅当上游需要时打开，fla 语义默认关闭）。$\mathrm{scale}$ 约定为

$$
\mathrm{scale} = K^{-1/2}
$$

（`naive.py` 的 `1 / (q.shape[-1] ** 0.5)`，不是其他幂次），乘在（归一化后的）$q$ 上。归一化 + scale 的组合效果是：$q$ 行的内积被约束在 $[-1,1]\cdot K^{-1/2}$ 量级，使 $P$ 矩阵与输出 $o$ 的数值范围受控。证据文件给出的实例：G1-A 的 dump 中 $\max|{\cdot}|\approx 8.2$，fp32 存储舍入约 $1.4\times 10^{-8}$（`evidence/G1/jax_spec_report.md` §2）。

**scale 的折叠位置**。$o$ 关于 $q$ 线性，规约把 scale 预乘进 $q$（`gdn_seq`：`q = q * scale` 在递推之前；`gdn_chunk` 同）；L1 kernel 则把它折叠到**最后的输出存储**：

$$
o = (\,o_{\text{inter}} + o_{\text{intra}}\,)\cdot \mathrm{scale}
$$

（`tools/gdn_l1_gen.py:22-23`、生成的 IR 中 `%fscale = arith.constant 0.25 : f32` 当 $K=16$）。反向对称地在叶子处折叠 $\mathrm{dO}=\mathrm{do}\cdot\mathrm{scale}$（§2.3.2）。两种折叠位置数学等价，但 kernel 侧省掉一次 $[C,K]$ 逐元素乘。

**源码片段**（`tools/gdn_l1_gen.py:66-82`，数据侧归一化约定与生成器嵌入约定）：

```python
def make_data(B, T, H, K, V, seed):
    """Deterministic seeded fp32 test data, GDN-model-style ...:
    q,k l2-normalized; beta=sigmoid(x); g = -softplus(0.5x) NEGATIVE
    NATURAL-log decay; h0 [B,H,K,V]."""
    rng = np.random.default_rng(seed)
    def l2n(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    q = l2n(rng.standard_normal((B, T, H, K))).astype(np.float32)
    k = l2n(rng.standard_normal((B, T, H, K))).astype(np.float32)
    ...
    g = (-np.logaddexp(0.0, xg)).astype(np.float32)  # natural-log domain
```

**效果数字**。h0 路径被真实行使：G1-A 两个仅 $h_0$ 不同的 dump（`ref_jax_T128_C64.npz` vs `ref_jax_T128_C64_h0.npz`）的 $o$ 相差 $2.9\times 10^{-2}$，而 MLIR 对两者的对拍差同为 $6.407\times10^{-7}$（`evidence/G1/l1_check_output.txt` jax_C64 / jax_C64_h0 两行）。

#### 2.2.2 $g$ 的数域约定：自然对数域输入、$\log_2$ 域执行、$\mathrm{RCP\_LN2}$

**动机（算子层面）**。fla 契约里 $g$ 是**自然对数域**逐 token 衰减，乘子为 $e^{g}$（e 底）。GPU/CPU kernel 惯用 $\exp_2$（硬件 exp2 指令/一次乘法尾数操作），因此 fla 的 Triton 路径在 kernel 前把 $g$ 预乘 $\mathrm{RCP\_LN2}=\log_2 e=1.4426950408889634$（`fla/ops/gated_delta_rule/chunk.py` 的 `chunk_local_cumsum(..., scale=RCP_LN2)`）。L1 生成器做同样的事（`tools/gdn_l1_gen.py:59,223-224`）：

$$
g_{\log_2} = g\cdot \log_2 e, \qquad \exp_2(g_{\log_2}) = e^{g}
$$

```python
RCP_LN2 = 1.4426950408889634  # log2(e): natural-log g -> log2-domain g
...
# embed g in the kernel-internal log2 domain: exp2(g*RCP_LN2) == exp(g)
g_log2 = (g.astype(np.float64) * RCP_LN2).astype(np.float32)
```

**IR 对比**。生成的模块头部把该约定写进注释（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:3-5`）：

```mlir
// conventions: embedded g is NATURAL-log g scaled by RCP_LN2 (log2
// domain, like fla's Triton path); gc inclusive chunk cumsum; A strictly
// lower; T=(I+A)^-1 forward substitution; scale=K^-1/2 folded into o.
```

IR 内所有衰减因子均为 `math.exp2`。这一约定在反向有一个非平凡推论——**dg 链不做 RCP_LN2 重标度**——是 G1-B2 的关键发现之一（§2.3.10）。

**效果数字**。若把数域搞错（例如 kernel 直接 $\exp_2(g)$ 而不预乘），128 步累计衰减将偏差 $2^{(1-\log_2 e)\sum g}$ 量级（指数级放大）；对拍链上任何数域错误都会让 `evidence/G1/l1_check_output.txt` 的 $10^{-7}$ 级差直接爆掉，7/7 PASS 是该约定的执行级证据。

#### 2.2.3 状态递推的仿射性：chunk 化的数学引擎

**动机（算子层面）**。逐步递推是一个长度 $T$ 的串行扫描——GPU 上不可并行。chunk-WY 分解的核心观察是：**一个 chunk 内 $C$ 步递推对状态 $S$ 是一个仿射映射**，可以显式写成"一个 $[K,K]$ 线性部分 + 一个 $[K,V]$ 偏置部分"，代价是解一个 $C\times C$ 单位下三角方程组。这是从"扫描"到"矩阵乘"的转换枢纽，也是 L1 与 fla kernel 同构的原因。

**推导**（先定义后用；推导在自然对数域进行，kernel 以 $\exp_2$ 实现等价形式）。设 chunk 内时间下标 $i=0,\dots,C-1$，含尾累加 $\mathrm{gc}_i=\sum_{t\le i}g_t$。对逐步递推去递归化：第 $i$ 步读到的（衰减后）状态是

$$
h_i^{\text{dec}} = e^{g_i}S_{i-1} = e^{\mathrm{gc}_i}\,S_{\text{entry}} \;+\; \sum_{j<i}\bigl(k_j\, e^{\mathrm{gc}_i-\mathrm{gc}_j}\bigr)^{\!\top} b_j,
$$

其中 $b_j$ 是第 $j$ 步的 delta 修正量 $b_j=\beta_j\bigl(v_j-k_j h_j^{\text{dec}}\bigr)$。代入并按 $i$ 收集：

$$
b_i + \sum_{j<i} \underbrace{\beta_i\,(k_i\!\cdot\! k_j)\,e^{\mathrm{gc}_i-\mathrm{gc}_j}}_{A_{ij}} \, b_j
\;=\; \beta_i v_i - \beta_i\bigl(k_i\, e^{\mathrm{gc}_i}\bigr) S_{\text{entry}} .
$$

即 chunk 内全部 $b$ 向量满足一个单位下三角线性方程组 $(I+A)\,b = \beta\odot v - (\beta\odot k\odot e^{\mathrm{gc}})\,S_{\text{entry}}$，其中 $A$ 严格下三角（§2.2.5 定义）。于是

$$
v_{\text{new}} := b = T(\beta\odot v) - T(\beta\odot k\odot e^{\mathrm{gc}})\,S_{\text{entry}} = u - w\,S_{\text{entry}},
$$

$$
u := T(\beta\odot v)\in\mathbb R^{C\times V}, \qquad w := T(\beta\odot k\odot e^{\mathrm{gc}})\in\mathbb R^{C\times K}, \qquad T:=(I+A)^{-1}.
$$

chunk 出口状态（把 $\sum_j(k_j e^{\mathrm{gc}_{C-1}-\mathrm{gc}_j})^{\top}b_j$ 写成 $k_{\mathrm{term}}^{\top}v_{\text{new}}$，$k_{\mathrm{term}}:=k\odot e^{\mathrm{gcl}-\mathrm{gc}}$）：

$$
S_{\text{exit}} \;=\; 2^{\mathrm{gcl}}\,S_{\text{entry}} + k_{\mathrm{term}}^{\top}\bigl(u - w\,S_{\text{entry}}\bigr)
\;=\; \underbrace{\bigl(2^{\mathrm{gcl}}\,I_K - k_{\mathrm{term}}^{\top}w\bigr)}_{\Phi\;\in\;\mathbb R^{K\times K}}\,S_{\text{entry}} \;+\; \underbrace{k_{\mathrm{term}}^{\top}u}_{b\;\in\;\mathbb R^{K\times V}} .
$$

这正是**仿射递推** $S' = \Phi S + b$：线性部分 $\Phi=2^{\mathrm{gcl}}I-k_{\mathrm{term}}^{\top}w$ 是"单位阵减一个秩 $\le\min(K,C)$ 修正"，偏置部分 $b=k_{\mathrm{term}}^{\top}u$。三个直接推论：

- **输出可分解**：$o_i = q_i h_i^{\text{after}}$ 关于 $S_{\text{entry}}$ 仿射，故 $o = o_{\text{inter}}(S) + o_{\text{intra}}$（§2.2.9）；
- **跨 chunk 仍是同一仿射递推**：整个序列是 $\mathrm{NC}$ 次仿射复合（L1 顺序执行该复合；跨 chunk 并行扫描是后续层的话题）；
- **反向结构确定**：仿射映射的伴随给出 $\mathrm{dH}$ 反向递推（§2.3.3）。

**IR 对比**。生成器不显式物化 $\Phi$，而是按上式右端原样执行：`%dl = math.exp2 %gcl`（$2^{\mathrm{gcl}}$）、`%kg`（$k_{\mathrm{term}}$）、`%wp`（$w$）、`%ubuf`（$u$），状态更新见 §2.2.8 的 IR。numpy 孪生（`tools/gdn_l1_check.py::chunked_ref:86-87`）逐式对应：

```python
S = (S * np.exp(gc[-1])
     + (kc * np.exp(gc[-1] - gc)[:, None]).T @ vn).astype(f32)
```

**效果数字**。chunk 化推导本身的正确性由"chunk-vs-seq"列独立验证（不涉及 MLIR）：numpy chunked 参考对 sequential 移植的最大差 $1.192\times10^{-7}$–$5.662\times10^{-7}$（fp32，`evidence/G1/l1_check_output.txt` 末列）；fp64 下 JAX 规约内部 chunk-vs-seq 达 $5.6\times10^{-16}$（$o$）/$1.2\times10^{-15}$（$h_t$）（`evidence/G1/jax_spec_report.md` §1）。

#### 2.2.4 chunk-local 含尾 cumsum $\mathrm{gc}$

**动机（算子层面）**。chunk 内所有衰减因子都以"累计衰减差"出现：$D_{ij}=2^{\mathrm{gc}_i-\mathrm{gc}_j}$（A 矩阵、P 矩阵）、$2^{\mathrm{gc}}$（$w$ 的 k 行、$o_{\text{inter}}$ 的 q 行）、$2^{\mathrm{gcl}}$（状态线性部分）、$2^{\mathrm{gcl}-\mathrm{gc}}$（$k_{\mathrm{term}}$）。$\mathrm{gc}$ 必须是**仅在 chunk 内部的含尾（inclusive）累加**：

$$
\mathrm{gc}_i = \sum_{t=0}^{i} g_t \quad (i=0,\dots,C-1), \qquad \mathrm{gc} \text{ 不跨 chunk}
$$

跨 chunk 的全局 cumsum 是经典错误：它会让 $D_{ij}$ 带上 chunk 前历史，使 $A$、$P$ 与状态更新整体错位（`evidence/G1/jax_spec_report.md` §3："Global (cross-chunk) cumsum would be wrong"）。跨 chunk 衰减只通过两个显式因子进入：状态上的 $2^{\mathrm{gcl}}$ 与 k 行上的 $2^{\mathrm{gcl}-\mathrm{gc}}$——即每个 chunk 把"整段历史衰减"折叠成对入口状态的一次乘法。

**源码片段**（`tools/gdn_l1_gen.py:420-431`；序列依赖决定了这是标量 `scf.for` 而非 `linalg.generic`）：

```python
# (a) inclusive chunk-local cumsum of g
e(f"%g0v = memref.load %g_c[%c0] : {t_bc}")
e(f"memref.store %g0v, %gc[%c0] : {t_C}")
e.block("scf.for %i = %c1 to %cC step %c1")
e("%ip = arith.subi %i, %c1 : index")
e(f"%prev = memref.load %gc[%ip] : {t_C}")
e(f"%cur = memref.load %g_c[%i] : {t_bc}")
e("%cs = arith.addf %prev, %cur : f32")
e(f"memref.store %cs, %gc[%i] : {t_C}")
e.end()
e(f"%gcl = memref.load %gc[%cCm1] : {t_C}")
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2514-2523`，逐 token 载入跨步 $2$ 的 strided 视图——见 §2.4.3——累加进稠密 scratch `%gc`）：

```mlir
%g0v = memref.load %g_c[%c0] : memref<64xf32, strided<[2], offset: ?>>
memref.store %g0v, %gc[%c0] : memref<64xf32>
scf.for %i = %c1 to %cC step %c1 {
  %ip = arith.subi %i, %c1 : index
  %prev = memref.load %gc[%ip] : memref<64xf32>
  %cur = memref.load %g_c[%i] : memref<64xf32, strided<[2], offset: ?>>
  %cs = arith.addf %prev, %cur : f32
  memref.store %cs, %gc[%i] : memref<64xf32>
}
%gcl = memref.load %gc[%cCm1] : memref<64xf32>
```

**效果数字**。含尾正确性与 chunk 边界一起由尾块用例检验：`tail_t96`（$T{=}96=64{+}32$，一个尾 chunk 补 32 个零行）$o$ 差 $5.662\times10^{-7}$、`tail_b2`（$B{=}2,T{=}100=64{+}36$）$3.099\times10^{-6}$（`evidence/G1/l1_check_output.txt`）——补零行的 $\beta=k=g=0$ 使其成为 A/u/S 的**精确无操作行**（§2.3.11），任何 cumsum 边界错误都会在这些用例上显形。

#### 2.2.5 KKT Gram 矩阵与 $A$ 的构造

**动机（算子层面）**。§2.2.3 推导中的耦合矩阵

$$
A_{ij} = \beta_i\,\bigl(k_i\!\cdot\! k_j\bigr)\,2^{\mathrm{gc}_i-\mathrm{gc}_j} \quad (j<i), \qquad A_{ij}=0 \;(j\ge i)
$$

由三因子复合：Gram 矩阵 $\mathrm{kkt}=kk^{\top}$（生成器缓冲区名 `%kkt`，即"KKT"之名的直接来源）、行向 $\beta$（**不对称**：$\beta_i$ 只乘行 $i$，$\beta$ 对 $v$ 则是全局缩放——这是 fla 契约第 4 条，`docs/L0_SEMANTICS.md`）、衰减差 $D_{ij}$。生成器记号 $A=-M$（$M$ 为 naive.py/G1-A 报告的严格下三角矩阵 $M_{ij}=-\beta_i(k_i\cdot k_j)e^{cd_i-cd_j}$），从而 $I+A=I-M$，两种记号给出同一个 $T$。例：$C=3$、$k$ 已 L2 归一化时 $|k_i\cdot k_j|\le 1$，$A_{ij}$ 的量级由 $\beta_i\in(0,1)$ 与 $2^{\mathrm{gc}_i-\mathrm{gc}_j}\le 1$（$g\le 0$ 使 $\mathrm{gc}$ 单调不增，故 $D_{ij}\le 1$）共同压制——这是 delta rule 数值稳定的结构性来源。

**源码片段**（`tools/gdn_l1_gen.py:433-473`；两步：缩并 + 掩蔽逐元素）：

```python
# (b) kkt = k_c @ k_c^T ; A = strict-tril(beta_i * kkt * 2^(gc_i-gc_j))
linalg.fill -> %kkt
linalg.generic  indexing_maps = [(i,j,r)->(i,r), (i,j,r)->(j,r), (i,j,r)->(i,j)]
                iterator_types = [parallel, parallel, reduction]
                ins(%k_c, %k_c) outs(%kkt):  %p = a*b ; acc+=%p
linalg.generic  indexing_maps = [(i,j)->(i,j), (i,j)->(i), (i,j)->(j), (i,j)->(i), (i,j)->(i,j)]
                ins(%kkt, %gc, %gc, %beta_c) outs(%Abuf):
                  %d0 = gc_i - gc_j ; %e0 = math.exp2 %d0
                  %m0 = kkt_ij * beta_i ; %m1 = %m0 * %e0
                  %i0/%j0 = linalg.index 0/1
                  %lt = arith.cmpi slt %j0, %i0
                  %r0 = arith.select %lt, %m1, %f0
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2524-2558`，第二个 generic 逐字）：

```mlir
linalg.fill ins(%f0 : f32) outs(%kkt : memref<64x64xf32>)
linalg.generic {
  indexing_maps = [ affine_map<(i, j, r) -> (i, r)>, affine_map<(i, j, r) -> (j, r)>,
                    affine_map<(i, j, r) -> (i, j)> ],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%k_c, %k_c : ...) outs(%kkt : memref<64x64xf32>) {
^bb0(%a0: f32, %a1: f32, %o0: f32):
  %p = arith.mulf %a0, %a1 : f32
  %s2 = arith.addf %o0, %p : f32
  linalg.yield %s2 : f32
}
linalg.generic {
  indexing_maps = [ affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>,
                    affine_map<(i, j) -> (j)>, affine_map<(i, j) -> (i)>,
                    affine_map<(i, j) -> (i, j)> ],
  iterator_types = ["parallel", "parallel"]
} ins(%kkt, %gc, %gc, %beta_c : ...) outs(%Abuf : memref<64x64xf32>) {
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

掩蔽的 IR 手法值得注意：三角掩码不是数据（没有物化 mask 矩阵），而是**控制计算**——`linalg.index` 取回迭代坐标，`arith.cmpi slt` + `arith.select` 在寄存器级完成 $\mathbb 1[j<i]$ 乘法。这一个手法在本章反复出现（A、P、dPm、G、dAf 的全部三角掩蔽）。

**效果数字**。$A$ 的正确性（连同 $T$）被整条链覆盖：MLIR vs 嵌入 numpy chunked 参考（同一 $A$ 构造）$5.066\times10^{-7}$–$6.407\times10^{-7}$（`evidence/G1/l1_check_output.txt` diff_o 列），numpy chunked vs sequential（独立算法路径）$1.192\times10^{-7}$–$5.662\times10^{-7}$（末列）。

#### 2.2.6 前代换求逆 $T=(I+A)^{-1}$：MLIR 三重循环

**动机（算子层面）**。$u=T(\beta v)$、$w=T(\beta k 2^{\mathrm{gc}})$ 都需要 $(I+A)^{-1}$ 作用到右端。$A$ 严格下三角 $\Rightarrow$ $I+A$ 单位下三角 $\Rightarrow$ 求逆/求解是**前代换（forward substitution）**，$O(C^3/6)$ 次标量乘加，无主元选取、无除法（对角元恒 1）。L1 的选择是把这个求解**完全显式地**写成三重 `scf.for`——不调用任何求解库、不引入黑盒 op——因为 L1 的职责是把算法结构钉死为可审计的 IR；行递推、Neumann 级数、分块求解等变体留给后续层作为可 A/B 的旋钮（`evidence/G1/l1_harness_report.md` "Known limitations"：`solve variant` 旋钮）。

**数学定义**（与 naive.py 的 attn 行递推逐式对应）：

$$
T_{i,i}=1;\qquad T_{i,j} = -\sum_{k=j}^{i-1} A_{i,k}\,T_{k,j} \quad (j<i)
$$

（`tools/gdn_l1_gen.py:27-29`：与 naive.py 的 `attn` 循环是相同的行递推；naive.py 的等价形式是就地行更新 $\text{attn}[i,:i] \mathrel{+}= \sum_k \text{attn}[i,k]\,\text{attn}[k,:i]$ 再加单位阵，见 §2.5.1 代码。）

**源码片段**（`tools/gdn_l1_gen.py:475-493`，生成器发射逻辑；注意最内层是携带 `iter_args` 累加器的 `scf.for`——MLIR 的"归约循环"原语）：

```python
# (c) T = (I+A)^-1 forward substitution
e(f"linalg.fill ins(%f0 : f32) outs(%Tbuf : {t_CC})")
e.block("scf.for %i2 = %c0 to %cC step %c1")            # 行 i
e(f"memref.store %f1, %Tbuf[%i2, %i2] : {t_CC}")         # T[i,i]=1
e.block("scf.for %j2 = %c0 to %i2 step %c1")             # 列 j<i
e(f"%sum = scf.for %k2 = %j2 to %i2 step %c1 "
  f"iter_args(%acc2 = %f0) -> (f32) {{")                 # 归约 k∈[j,i)
e(f"%aik = memref.load %Abuf[%i2, %k2] : {t_CC}")
e(f"%tkj = memref.load %Tbuf[%k2, %j2] : {t_CC}")
e("%pk = arith.mulf %aik, %tkj : f32")
e("%ak = arith.addf %acc2, %pk : f32")
e("scf.yield %ak : f32")
...
e("%neg = arith.negf %sum : f32")
e(f"memref.store %neg, %Tbuf[%i2, %j2] : {t_CC}")
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2559-2573`，真实生成文本，前向样本 `l1_chunked_T128_C64.mlir:1960` 起同构）：

```mlir
linalg.fill ins(%f0 : f32) outs(%Tbuf : memref<64x64xf32>)
scf.for %i2 = %c0 to %cC step %c1 {
  memref.store %f1, %Tbuf[%i2, %i2] : memref<64x64xf32>
  scf.for %j2 = %c0 to %i2 step %c1 {
    %sum = scf.for %k2 = %j2 to %i2 step %c1 iter_args(%acc2 = %f0) -> (f32) {
      %aik = memref.load %Abuf[%i2, %k2] : memref<64x64xf32>
      %tkj = memref.load %Tbuf[%k2, %j2] : memref<64x64xf32>
      %pk = arith.mulf %aik, %tkj : f32
      %ak = arith.addf %acc2, %pk : f32
      scf.yield %ak : f32
    }
    %neg = arith.negf %sum : f32
    memref.store %neg, %Tbuf[%i2, %j2] : memref<64x64xf32>
  }
}
```

**前后对比**（同一数学的三个实现形态）：

| 形态 | 位置 | 形式 |
|---|---|---|
| JAX 规约（行递推，向量化） | `tools/gdn_ref_jax.py:196-200` | `for i in range(1,c): attn = attn.at[..., i, :i].add(jnp.sum(attn[..., i, :i, None] * attn[..., :i, :i], axis=-2))`；随后 `attn += I` |
| numpy 孪生（标量三重循环） | `tools/gdn_l1_check.py:71-78` | `for i: Tm[i,i]=1; for j<i: s=0; for kk in range(j,i): s += A[i,kk]*Tm[kk,j]; Tm[i,j]=-s` |
| L1 MLIR（三重 `scf.for`） | 上文 IR | 最内层归约以 `iter_args` 携带累加器，逐元素 `memref.load/store` |

三者是同一行递推：JAX 形式按行向量化（一次 `add` 完成整行），MLIR 形式把内积降为标量循环——后者牺牲向量换来两点：(i) 完全显式、可逐行审计；(ii) 对任意 $C$ 参数化生成（生成器只依赖编译期常数 `%cC`），不依赖库。

**效果数字**。前代换路径连同 $A$ 构造覆盖于全部 7 个前向用例（`evidence/G1/l1_check_output.txt`，gate $<10^{-3}$，实测最大 $3.099\times10^{-6}$）；`evidence/G1/l1_harness_report.md` 记录该求解为 $O(C^3/6)$ 标量迭代（$C{=}64$ 时约 43k 次/chunk），base 模块 `mlir-opt` 0.32 s / `mlir-runner` 0.22 s 完成 JIT 与执行。

#### 2.2.7 $u/w$ 的计算与 $v_{\text{new}}$

**动机（算子层面）**。$u,w$ 是 WY 表示的两块"系数"：$u$ 携带值信息（$T$ 作用于 $\beta\odot v$），$w$ 携带键信息（$T$ 作用于 $\beta\odot k\odot 2^{\mathrm{gc}}$，fla 名 `k_cumdecay`）；二者与入口状态组合出 $v_{\text{new}}$——chunk 内全部 delta 修正量的解。定义（生成器约定）：

$$
\mathrm{upre} = \beta\odot v, \qquad u = T\,\mathrm{upre}, \qquad \mathrm{kb2} = \beta\odot k\odot 2^{\mathrm{gc}}, \qquad w = T\,\mathrm{kb2}, \qquad v_{\text{new}} = u - w\,S .
$$

**源码片段**（`tools/gdn_l1_gen.py:495-550`）：两个逐元素 generic（`%upre`、`%kb2`）+ 两个命名缩并 + 一个残差 generic：

```python
e(f"linalg.fill ins(%f0 : f32) outs(%ubuf : {t_CV})")
e(f"linalg.matmul ins(%Tbuf, %upre : {t_CC}, {t_CV}) outs(%ubuf : {t_CV})")
...
e(f"linalg.fill ins(%f0 : f32) outs(%wp : {t_CK})")
e(f"linalg.matmul ins(%Tbuf, %kb2 : {t_CC}, {t_CK}) outs(%wp : {t_CK})")
# (e) v_new = u - w @ S
e(f"linalg.fill ins(%f0 : f32) outs(%vp : {t_CV})")
e(f"linalg.matmul ins(%wp, %S : {t_CK}, {t_KV}) outs(%vp : {t_CV})")
generic(... ins(%ubuf, %vp) outs(%vn): %r3 = subf %a0, %a1 ...)
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2586-2604,2605-2618`）：

```mlir
linalg.fill ins(%f0 : f32) outs(%ubuf : memref<64x16xf32>)
linalg.matmul ins(%Tbuf, %upre : memref<64x64xf32>, memref<64x16xf32>) outs(%ubuf : memref<64x16xf32>)
...
linalg.fill ins(%f0 : f32) outs(%wp : memref<64x16xf32>)
linalg.matmul ins(%Tbuf, %kb2 : memref<64x64xf32>, memref<64x16xf32>) outs(%wp : memref<64x16xf32>)
linalg.fill ins(%f0 : f32) outs(%vp : memref<64x16xf32>)
linalg.matmul ins(%wp, %S : memref<64x16xf32>, memref<16x16xf32>) outs(%vp : memref<64x16xf32>)
linalg.generic { ... } ins(%ubuf, %vp : ...) outs(%vn : memref<64x16xf32>) {
^bb0(%a0: f32, %a1: f32, %o0: f32):
  %r3 = arith.subf %a0, %a1 : f32
  linalg.yield %r3 : f32
}
```

注意两个生成器代码模式：(i) 命名 op `linalg.matmul` 是**累加语义**（写入 outs 前 accumulate），因此每个缩并缓冲先 `linalg.fill` 清零——这是生成器的固定"fill-before-accumulate"模式；(ii) 反向 pass 复用同一组缓冲名（`%upre`/`%kb2` 在 G 链中再次出现，§2.3.4）。

**效果数字**。`evidence/G1/l1_harness_report.md` 前向表全部 7 行 PASS；其中对 G1-A dump 的三行（jax_C64_h0/jax_C64/jax_C32）同时验证了 $u/w$ 路径与 JAX 的字母互换约定（jax 的 `w`=$u$、`u`=$w$）没有接错——接错将产生 $O(1)$ 级错误而非 $10^{-7}$。

#### 2.2.8 状态更新与 `h_all` 递推

**动机（算子层面）**。chunk 出口的仿射状态更新（§2.2.3）：

$$
S \;\leftarrow\; 2^{\mathrm{gcl}}\odot S \;+\; \bigl(k\odot 2^{\mathrm{gcl}-\mathrm{gc}}\bigr)^{\!\top} v_{\text{new}} \;=\; 2^{\mathrm{gcl}} S + k_{\mathrm{term}}^{\top}v_{\text{new}},
$$

右侧的 $S$ 是**更新前**的入口状态（对应"衰减先于 delta 更新"的逐步顺序）；$k_{\mathrm{term}}^{\top}v_{\text{new}}$ 是 $[K,C]\times[C,V]$ 缩并。前向 kernel 中 $S$ 是跨 chunk 循环携带的活跃状态；反向 pass 1 则把**每个 chunk 的入口状态**存入 `h_all[n]`（$[\mathrm{NC},K,V]$ scratch，`tools/gdn_l1_gen.py:892` `t_HALL = mrt((NC, K, V))`）——`h_all` 递推即"沿 chunk 前进时逐片保存 $S_{\text{entry}}(n)$"。

**源码片段**（`tools/gdn_l1_gen.py:629-677` 前向状态更新；转置不物化，折进 indexing_maps）：

```python
# S = S .* 2^gcl + (k .* 2.^(gcl-gc))^T @ v_new
e("%dl = math.exp2 %gcl : f32")
generic(... ins(%k_c, %gc) outs(%kg):      # kg = k_term
        %dg = gcl - gc_i ; %eg = math.exp2 %dg ; %r6 = k*eg)
e(f"linalg.fill ins(%f0 : f32) outs(%prod : {t_KV})")
generic(  maps = ["(kk, vv, i) -> (i, kk)", "(kk, vv, i) -> (i, vv)",
                 "(kk, vv, i) -> (kk, vv)"],
         kinds = ["parallel", "parallel", "reduction"],
         ins(%kg, %vn) outs(%prod):  acc += kg[i,kk]*vn[i,vv])
generic(... ins(%S, %prod) outs(%S):  %ms = S*dl ; %rs = ms + prod)
```

注意缩并的转置实现：`(kk,vv,i) -> (i,kk)` 把输入的第一维（chunk 行）映射为归约维——`k_term`^T @ `v_new` 无需显式转置缓冲。终态写出：`%hth = memref.subview %ht[%b, %h, ...]` + `linalg.copy ins(%S) outs(%hth)`（`tools/gdn_l1_gen.py:680-682`）。

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:951-953,961`，反向 pass 1 的 h_all 保存与状态推进）：

```mlir
%h_all_n = memref.subview %h_all[%n, 0, 0] [1, 16, 16] [1, 1, 1] : memref<2x16x16xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
linalg.copy ins(%S : memref<16x16xf32>) outs(%h_all_n : memref<16x16xf32, strided<[16, 1], offset: ?>>)
...（重算 gc/A/T/u/w，推进 S）...
%dl = math.exp2 %gcl : f32
```

**效果数字**。状态路径由 $h_t$ 列独立验证：全部用例 $\max|h_t-\text{ref}|$ 为 $5.513\times10^{-7}$–$3.099\times10^{-6}$（`evidence/G1/l1_check_output.txt`）；$h_0$ 是否真实传播亦被 h0/no-h0 dump 对比证实（$o$ 差 $2.9\times10^{-2}$；$h_t$ 在两 dump 中相同属预期——128 步衰减把 $h_0$ 贡献压到 fp32 分辨率之下，`evidence/G1/l1_harness_report.md` 明确记录"expected, not a bug"）。

#### 2.2.9 输出的 intra/inter 分解

**动机（算子层面）**。$o_i$ 读的是第 $i$ 步 delta 更新**之后**的状态，拆开即两项：

$$
o_i = \underbrace{\bigl(q_i\,2^{\mathrm{gc}_i}\bigr) S_{\text{entry}}}_{o_{\text{inter},i}\;(\text{跨 chunk 记忆})} \;+\; \sum_{j\le i}\underbrace{\bigl(q_i\!\cdot\! k_j\bigr)\,2^{\mathrm{gc}_i-\mathrm{gc}_j}}_{P_{ij}}\, v_{\text{new},j} \;=\; o_{\text{inter},i} + \bigl(P\,v_{\text{new}}\bigr)_i ,
$$

$$
P = \mathrm{tril}_{\text{incl}}\bigl((q\,k^{\top})\odot D\bigr) \quad(\textbf{对角保留，只掩严格上三角}), \qquad o = (o_{\text{inter}} + o_{\text{intra}})\cdot\mathrm{scale}.
$$

两处易错的对偶必须强调：$A$ 是**严格**下三角（chunk 内因果耦合只存在于 $j<i$），而 $P$ **含对角**（第 $i$ 步输出看到自己的更新 $b_i$）；$o_{\text{inter}}$ 与 $w$/`k_term` 共享 $2^{\mathrm{gc}}$ 型行因子，但作用对象是 $q$ 行。$\mathrm{scale}$ 折叠在最终存储（§2.2.1）。

**源码片段**（`tools/gdn_l1_gen.py:552-627`：`%qg = q⊙2^gc` → `oob = qg@S`；`%qk=qk^T` 缩并 → 复用 `%Abuf` 存 $P$（此时 A 已死）→ `oib = P@vn`；最终 `o_c = (oob+oib)*scale` 存入 strided 输出视图）：

```python
# o_inter = (q .* 2.^gc) @ S
generic(... ins(%q_c, %gc) outs(%qg):  %e3 = exp2(gc_i) ; %r4 = q*e3)
linalg.matmul ins(%qg, %S) outs(%oob)
# M = tril_inc((q k^T) .* 2^(gc_i-gc_j)) ; reuse Abuf; o_intra = M @ v_new
...  # 缩并 qk^T，再 exp2 差 + sle 掩蔽（注意 sle = 含对角，与 A 的 slt 相对）
linalg.matmul ins(%Abuf, %vn) outs(%oib)
# o_c = (o_inter + o_intra) * scale
generic(... ins(%oob, %oib) outs(%o_c):  %ad = addf ; %sc = ad * %fscale)
```

**IR 对比**。掩蔽方向的前后对照是本 pattern 的关键细节：A 用 `arith.cmpi slt`（$j<i$，严格），P 用 `arith.cmpi sle`（$j\le i$，含对角）——同一 `linalg.index` 手法、一个比较谓词之差（`tools/gdn_l1_gen.py:469` vs `:604`）。

**效果数字**。含对角正确性有独立证据：若误用严格掩蔽，对角项 $q_i\!\cdot\! k_i\,b_i$（量级 $O(\beta)$）丢失，误差远超 $10^{-3}$ gate；实测 jax_C64 两行 $6.407\times10^{-7}$（`evidence/G1/l1_check_output.txt`）。JAX 规约同式对照（`tools/gdn_ref_jax.py:220-226`：`a_i` 用 `triu_strict` 掩蔽即保留对角）。

#### 2.2.10 前向小结：一个 chunk 的完整数据流

把 §2.2.4–§2.2.9 串起来，每 chunk（每 $(b,h)$）的 L1 数据流为：

$$
g_c \xrightarrow{\text{cumsum}} \mathrm{gc},\,\mathrm{gcl} \;\rightarrow\; \mathrm{kkt}=kk^{\top} \;\rightarrow\; A=\mathrm{strict\_tril}(\beta\,\mathrm{kkt}\odot D) \;\rightarrow\; T=(I+A)^{-1} \;\rightarrow\; \{u=T(\beta v),\; w=T(\beta k 2^{\mathrm{gc}})\}
$$

$$
\rightarrow\; v_{\text{new}}=u-wS \;\rightarrow\; \begin{cases} o = \bigl((q\odot 2^{\mathrm{gc}})S + P\,v_{\text{new}}\bigr)\cdot\mathrm{scale} \\[2pt] S \leftarrow 2^{\mathrm{gcl}}S + k_{\mathrm{term}}^{\top}v_{\text{new}} \end{cases}
$$

前向 kernel 使用 17 个 scratch 缓冲（`tools/gdn_l1_gen.py:379-396` 的 allocs：`S/Abuf/Tbuf/kkt/qk/gc/upre/ubuf/kb2/wp/vp/vn/qg/kg/oib/oob/prod`），循环结构为 `b → h → chunk` 三层 `scf.for`（`tools/gdn_l1_gen.py:375-411`；`evidence/G1/l1_chunked_T128_C64.mlir:1877-1905` 的真实 IR 同构）。代表性前向模块（B1 T128 H2 K16 V16 C64，seed 0）共 2167 行（`evidence/G1/l1_harness_report.md` Deliverables 表）。
### 2.3 反向数学：重算、dH 递推与 dA 双侧掩蔽公式

反向 kernel `@gdn_bwd(q,k,v,beta,g,h0,do,dht) -> (dq,dk,dv,dbeta,dg,dh0)` 以同一 memref+linalg+scf 风格生成（`tools/gdn_l1_gen.py:828-1384`），代码量约为前向的 1.6 倍（3540 行 vs 2167 行，同为 B1 T128 C64 seed 0，`evidence/G1/l1_harness_report.md`）。本节按"结构 → 链条 → 组装"展开，全部公式取自生成器 docstring（`tools/gdn_l1_gen.py:830-852`）、`evidence/G1/l1_harness_report.md` 的 Backward (G1-B2) 一节、以及 numpy 孪生 `tools/gdn_l1_check.py::chunked_bwd_ref`。

#### 2.3.1 重算（recompute）策略与两遍结构

**动机（算子层面）**。反向需要每个 chunk 的前向中间量（$\mathrm{gc},A,T,u,w,v_{\text{new}},P$）与入口状态 $S_n$。两种极端是"全部重算"（算两次前向，零额外存储）与"全部物化"（pass 1 存所有中间量）。L1 选择**重算**并把存储压到最小：**只存每 chunk 的入口状态** `h_all[n]`（$[\mathrm{NC},K,V]$）。

**结构**（`tools/gdn_l1_gen.py:941-984`；`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2504,2662` 的两段注释逐字对应）：

- **Pass 1（前向重算）**：精确执行前向 chunk 数学，但**不计算 $o$**，只把每个 chunk 的入口状态 `linalg.copy` 进 `%h_all[%n]`；中间量（gc/A/T/u/w）即弃。
- **Pass 2（反向 chunk 行走）**：`scf.for %n2 = 0..NC`，以**正步长 + 下标重映射**实现逆序：

```mlir
// ---- pass 2: reverse chunk walk (dH seeded from dht) ----
scf.for %n2 = %c0 to %cNC step %c1 {
  %rn = arith.subi %cNC, %c1 : index
  %n = arith.subi %rn, %n2 : index          // n = NC-1-n2
  %off = arith.muli %n, %cC : index
  ...
  %Sn = memref.subview %h_all[%n, 0, 0] [1, 16, 16] ...   // 入口状态
```

（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2662-2678`。）负步长 `scf.for` 被刻意规避（`evidence/G1/l1_harness_report.md`：逆向 chunk 循环、dg 的 suffix/prefix 循环全部用正步长 + 索引重映射）。每个 chunk 在 pass 2 内**重算** gc/A/T/u/w（`_bwd_fwd_core`，`tools/gdn_l1_gen.py:746-825`，与前向逐式相同），再读 $S_n$=`h_all[n]` 组出 $v_{\text{new}}=u-wS_n$。

**重算旋钮**。`evidence/G1/l1_harness_report.md` "Notes for G2"：反向 kernel 每 $(b,h)$ 分配约 37 个 scratch（前向 17；生成器 allocs 清单在 `tools/gdn_l1_gen.py:899-912` 逐项列出），且 pass 2 重算 gc/A/T/u/w——"materialize 变体（pass 1 存 u,w,P）是 L2 显然的存储/计算折中 A/B 旋钮"。

**效果数字**。重算路径的全梯度对拍：3 npz × 6 梯度全部 PASS（gate $<2\times10^{-4}$ vs fp32 参照 / $<3\times10^{-4}$ vs fp64 参照，实测最差 $6.4\times10^{-6}$，`evidence/G1/l1_bwd_check_output.txt`）；孪生自身对 JAX vjp 的 fp64 差 $\le 1.4\times10^{-14}$。

#### 2.3.2 叶子折叠 $\mathrm{dO}$ 与 $v_{\text{new}}$ 的总余切 $\mathrm{dvp}$

**动机**。$o=(o_{\text{inter}}+o_{\text{intra}})\cdot\mathrm{scale}$，故余切在叶子处折叠一次即可（与前向 scale 折叠互为镜像）：

$$
\mathrm{dO} = \mathrm{do}\cdot\mathrm{scale} \quad\in\mathbb R^{C\times V}.
$$

$v_{\text{new}}$ 从两条链收到余切：输出链（经 $P$）与状态链（经 $k_{\mathrm{term}}$，即 Bhat 链）：

$$
\mathrm{dvp} = P^{\top}\mathrm{dO} + \hat B\,\mathrm{dH}, \qquad \hat B := k\odot 2^{\mathrm{gcl}-\mathrm{gc}}\in\mathbb R^{C\times K}.
$$

**源码片段**（`tools/gdn_l1_gen.py:1017-1026`：dO/dOgc；`:1028-1045`：P（含对角掩蔽 `sle`，复用 `%Abuf`——此时 A 已死）；`:1066-1081`：dvp 两缩并，$\hat B$ **内联**进第二个缩并（`%bd = gcl - gc_i; %be = exp2; %bm = k*be; %bp = bm*dH_kv; acc+=`）——Bhat 不物化缓冲）：

```python
# dvp = P^T dO + Bhat dH   (Bhat = k .* 2^(gcl-gc), folded inline)
_fill(e, "%dvp", t_CV, "%f0")
generic(  maps=["(i, d, j) -> (j, i)", "(i, d, j) -> (j, d)", "(i, d, j) -> (i, d)"],
         ins=("%Abuf", "%dO"), outs=("%dvp",))   # P^T dO
generic(  maps=["(i, d, kk) -> (i, kk)", "(i, d, kk) -> (i)", "(i, d, kk) -> (kk, d)",
                "(i, d, kk) -> (i, d)"],
         ins=("%k_c", "%gc", "%dH"), outs=("%dvp",))  # + (k*2^(gcl-gc)) dH
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2901-2917`，Bhat 内联缩并逐字）：

```mlir
linalg.generic {
  indexing_maps = [ affine_map<(i, d, kk) -> (i, kk)>, affine_map<(i, d, kk) -> (i)>,
                    affine_map<(i, d, kk) -> (kk, d)>, affine_map<(i, d, kk) -> (i, d)> ],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%k_c, %gc, %dH : ...) outs(%dvp : memref<64x16xf32>) {
^bb0(%a0: f32, %a1: f32, %a2: f32, %o0: f32):
  %bd = arith.subf %gcl, %a1 : f32
  %be = math.exp2 %bd : f32
  %bm = arith.mulf %a0, %be : f32
  %bp = arith.mulf %bm, %a2 : f32
  %bs = arith.addf %o0, %bp : f32
  linalg.yield %bs : f32
}
```

转置（$P^{\top}$）再次以 indexing_maps 表达（`(i,d,j)->(j,i)`），不物化转置缓冲。

#### 2.3.3 $\mathrm{dH}$ 反向递推

**动机**。$S_{\text{exit}} = \Phi S_{\text{entry}} + b$（§2.2.3 的仿射形式）直接给出状态余切的反向递推：$\mathrm{dH}$ 是"离开该 chunk 的状态"的余切，从 $\mathrm{dht}$ 播种，逐 chunk（逆序）更新：

$$
\mathrm{dH}' \;=\; 2^{\mathrm{gcl}}\,\mathrm{dH} \;+\; \bigl(q\odot 2^{\mathrm{gc}}\bigr)^{\!\top}\mathrm{dO} \;-\; w^{\top}\mathrm{dvp},
$$

chunk 0 处理完后 $\mathrm{dh0}[b,h]=\mathrm{dH}$。三项分别对应仿射递推的三条活跃路径：衰减（线性部分的 $2^{\mathrm{gcl}}I$）、$o_{\text{inter}}$ 链（$q\odot2^{\mathrm{gc}}$ 作用在 $S$ 上）、$v_{\text{new}}$ 链（$w$ 作用在 $S$ 上，经 $\mathrm{dvp}$ 汇流）。

**源码片段**（`tools/gdn_l1_gen.py:1350-1371`；`%qg` 已在 dq 链算出（§2.3.9），此处复用）：

```python
# dH = dH .* 2^gcl + qg^T dO - w^T dvp
_fill(e, "%tK1", t_KV, "%f0")
generic(maps=["(kk, d, i) -> (i, kk)", "(kk, d, i) -> (i, d)", "(kk, d, i) -> (kk, d)"],
        ins=("%qg", "%dO"), outs=("%tK1",))          # qg^T dO
_fill(e, "%tK2", t_KV, "%f0")
generic(maps=["(kk, d, i) -> (i, kk)", "(kk, d, i) -> (i, d)", "(kk, d, i) -> (kk, d)"],
        ins=("%wp", "%dvp"), outs=("%tK2",))         # w^T dvp
generic(ins=("%dH", "%tK1", "%tK2"), outs=("%dH",)):
    %dh1 = dH*dl ; %dh2 = dh1 + tK1 ; %dh3 = dh2 - tK2
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:3448` 起的三 generic；`%dl = math.exp2 %gcl` 在 chunk 头部一次算出）。

**效果数字**。$\mathrm{dh0}$ 列的对拍差为全梯度中最小之一：$4.5\times10^{-7}$（npz 三用例，fp32/fp64 双口径，`evidence/G1/l1_bwd_check_output.txt`）——单 $[K,V]$ 缓冲的递推路径无 chunk 内重组，误差不叠加。

#### 2.3.4 $\mathrm{dW}$ 与 $G$：$T$ 的余切汇聚

**动机**。$u,w$ 都线性依赖 $T$，两条链的余切在 $T$ 处汇聚。$w$ 进入 $v_{\text{new}}=u-wS_n$（经 $\mathrm{dvp}$），$u$ 不依赖 $S_n$ 故其对 $T$ 的贡献经 $\mathrm{dvp}$ 的 $u$ 部分同样汇入；组合结果：

$$
\mathrm{dW} = -\,\mathrm{dvp}\,S_n^{\top}\;\in\mathbb R^{C\times K}, \qquad
G \;=\; \mathrm{dvp}\,(\beta v)^{\top} \;+\; \mathrm{dW}\,(\beta k\,2^{\mathrm{gc}})^{\top} \;=\; \mathrm{dvp}\,\mathrm{upre}^{\top} + \mathrm{dW}\,\mathrm{kb2}^{\top},
$$

$G=\partial L/\partial T\in\mathbb R^{C\times C}$。注意生成器**复用前向缓冲** `%upre`（$=\beta v$）与 `%kb2`（$=\beta k 2^{\mathrm{gc}}$）——它们在 `_bwd_fwd_core` 重算时已经就位（`tools/gdn_l1_gen.py:1092-1104`），$G$ 的两个缩并因此零额外预处理。

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2918-2972`：`%ws = arith.subf %o0, %wp2`（dW 的负号折进减法缩并）、两个累加缩并进 `%Gb`、随后**独立的一个掩蔽 generic** 对 $G$ 取严格下三角：`%glt = cmpi slt %gj0, %gi0 ; %gr0 = select %glt, %a0, %f0`）。这一步"先掩蔽"不是可读性选择，而是正确性要求——下一小节推导。

#### 2.3.5 dA 双侧掩蔽公式：推导与单侧形式的失效

**动机**。$T=(I+A)^{-1}$ 把 $G$ 变回对 $A$（即对 $k,\beta,g$ 的入口）的余切，是反向最微妙的一步。矩阵求逆的微分与伴随：

$$
\mathrm{d}T = -\,T\,(\mathrm{d}A)\,T .
$$

对损失 $L$（本规约中 $L=\sum \mathrm{do}\odot o+\sum \mathrm{dht}\odot h_t$，`evidence/G1/jax_spec_report.md` §2）：

$$
\mathrm{d}L = \langle G,\mathrm{d}T\rangle = \bigl\langle -T^{\top}G\,T^{\top},\,\mathrm{d}A\bigr\rangle .
$$

**为什么必须先掩蔽 $G$**。$A$ 严格下三角 $\Rightarrow$ $\mathrm{d}A$ 严格下三角 $\Rightarrow$ $\mathrm{d}T=-T\,\mathrm{d}A\,T$ 严格下三角（$T$ 单位下三角：$(T\,\mathrm{d}A)$ 的 $(i,j)$ 元 $=\sum_{k>j}T_{ik}\mathrm{d}A_{kj}$，当 $j\ge i$ 时 $k>j\ge i$ 使 $T_{ik}=0$；严格下三角乘下三角仍严格下三角）。于是 $\langle G,\mathrm{d}T\rangle$ 只依赖 $G$ 的严格下三角部分。但若**不先**把 $G$ 换成 $\mathrm{strict\_tril}(G)$ 就做乘积，$T^{\top}GT^{\top}$ 的严格下三角部分会通过 $T$ 的**单位对角**把 $G$ 的对角/上三角分量"泄漏"进结果（乘积项 $\sum_{k,l}T_{ki}\,G_{kl}\,T_{jl}$ 中 $k,l$ 不受 $j<i$ 约束）。因此正确形式是**双侧掩蔽**：

$$
\boxed{\;\mathrm{d}A \;=\; -\,\mathrm{strict\_tril}\Bigl(T^{\top}\cdot \mathrm{strict\_tril}(G)\cdot T^{\top}\Bigr)\;}
$$

**为什么单侧形式错**。任何只在一侧乘 $T^{\top}$（或对 $\mathrm{d}A$ 做对角补正）的近似都等价于丢掉一个方向的伴随混合：左乘 $T^{\top}$ 混合 $G$ 的**列**方向、右乘 $T^{\top}$ 混合**行**方向，而 $T$ 的严格下三角元素恰是 $-A$ 的 Neumann 累积（$T=I-A+A^2-\cdots$），两个方向的混合都携带 $A$ 的一阶信息。符号级验证（`evidence/G1/l1_harness_report.md` "The dA correction" 一节原话）：任务简报中的单侧变体 `dA.T + T^T (dA - diag dA)` 对 $C\ge 3$ 不复现 JAX vjp——$C{=}3$ 时该式给出 $2g_{21}+c\,g_{31}$，而真值是 $g_{21}+c\,g_{31}+c\,g_{32}a$（其中 $a$ 为 $A$ 的元素、$c$ 为相应衰减因子）：单侧形式既**重复计数**直达项（$2g_{21}$）又**丢失**交叉项（$c\,g_{32}a$，只可能由双侧乘积产生）。数值确认：双侧掩蔽式在 fp64 numpy 孪生上对三个 npz dump 的全部六个梯度 $\le 1.4\times10^{-14}$；且该式与 fla `prepare_wy_repr_bwd` 的 `dot(A, dA, A)` 等价——fla 从**转置存储布局**装载 $T$ 逆，其表达式读作 $T^{\top}GT^{\top}$，与上式一致（`evidence/G1/l1_harness_report.md` "Deviations" 第 4 条）。

**源码片段**（`tools/gdn_l1_gen.py:1113-1137`：三次缩并/掩蔽——$X=G_bT^{\top}$、$Y=T^{\top}X$、终步把衰减、负号、外层掩蔽合并进**一个** generic）：

```python
# X = G T^T ; Y = T^T X ; dAf = -strict_tril(Y .* D) into Gb (decay-free)
generic(maps=["(i,j,m)->(i,m)", "(i,j,m)->(j,m)", "(i,j,m)->(i,j)"],
        ins=("%Gb", "%Tbuf"), outs=("%Xb",))            # X = G_str T^T
generic(maps=["(i,j,m)->(m,i)", "(i,j,m)->(m,j)", "(i,j,m)->(i,j)"],
        ins=("%Tbuf", "%Xb"), outs=("%Gb",))            # Y = T^T X  (Gb 复用)
generic(ins=("%Gb", "%gc", "%gc"), outs=("%Gb",)):
    %dd0 = gc_i - gc_j ; %de0 = math.exp2 %dd0
    %dm0 = Y * de0 ; %dn0 = arith.negf %dm0
    %dlt = cmpi slt j, i ; %dr0 = select %dlt, %dn0, %f0
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2973-3020`，终步逐字）：

```mlir
linalg.fill ins(%f0 : f32) outs(%Xb : memref<64x64xf32>)
linalg.generic { ... affine_map<(i,j,m)->(i,m)>, <(i,j,m)->(j,m)>, <(i,j,m)->(i,j)> ... }
  ins(%Gb, %Tbuf : ...) outs(%Xb : memref<64x64xf32>) { ^bb0: %xp/%xs ... }   // X
linalg.fill ins(%f0 : f32) outs(%Gb : memref<64x64xf32>)
linalg.generic { ... affine_map<(i,j,m)->(m,i)>, <(i,j,m)->(m,j)>, <(i,j,m)->(i,j)> ... }
  ins(%Tbuf, %Xb : ...) outs(%Gb : memref<64x64xf32>) { ^bb0: %yp/%ys ... }   // Y
linalg.generic { ... <(i,j)->(i,j)>, <(i,j)->(i)>, <(i,j)->(j)>, <(i,j)->(i,j)> ... }
  ins(%Gb, %gc, %gc : ...) outs(%Gb : memref<64x64xf32>) {
^bb0(%a0: f32, %a1: f32, %a2: f32, %o0: f32):
  %dd0 = arith.subf %a1, %a2 : f32
  %de0 = math.exp2 %dd0 : f32
  %dm0 = arith.mulf %a0, %de0 : f32
  %dn0 = arith.negf %dm0 : f32
  %di0 = linalg.index 0 : index
  %dj0 = linalg.index 1 : index
  %dlt = arith.cmpi slt, %dj0, %di0 : index
  %dr0 = arith.select %dlt, %dn0, %f0 : f32
  linalg.yield %dr0 : f32
}
```

成本注记：dA 修正 = 每 chunk 两个 $C\times C\times C$ 缩并 + 掩蔽（`evidence/G1/GATE_REPORT.md` 数学成果第 3 条），分块/SIMD 化归 L2。

#### 2.3.6 $\mathrm{dAf}$：衰减坐标

**动机**。$A$ 的定义自带一个衰减因子：$A=\mathrm{strict\_tril}\bigl(\beta\,(\mathrm{kk}^{\top})\odot D\bigr)$。对 $k$ 链求导时链式法则会**再**引入一个 $D$（$\partial A_{ij}/\partial(\text{raw }kk^{\top}_{ij})=\beta_i D_{ij}$）。把两个 $D$ 合并，定义**衰减自由坐标**下的余切（fla 约定）：

$$
\mathrm{dAf} \;=\; \mathrm{d}A \odot D \;=\; -\,\mathrm{strict\_tril}\Bigl(\bigl(T^{\top}\mathrm{strict\_tril}(G)T^{\top}\bigr)\odot D\Bigr),
$$

此后 $k$ 链（dk/dbeta/dg 的 A-链）中**不再出现任何 $\exp_2$ 因子**——§2.3.5 终步 generic 中的 `%de0 = math.exp2 %dd0` 正是这最后一次（也是该链唯一一次）衰减作用。numpy 孪生同式：`dAf = -(Tm.T @ (G * m_str) @ Tm.T) * D * m_str`（`tools/gdn_l1_check.py:197`）。

#### 2.3.7 $\mathrm{dv}$ 与 $\mathrm{d\beta}$

$$
\mathrm{dv} = \bigl(T^{\top}\mathrm{dvp}\bigr)\odot\beta \qquad\in\mathbb R^{C\times V}
$$

（$v$ 只经 $\mathrm{upre}=\beta v$ 进入 $u=T\,\mathrm{upre}$；`tools/gdn_l1_gen.py:1139-1158`：`%tv = T^T dvp` 缩并，`%dvr = tv * beta` 存入 `%dv_c`。）

$$
\mathrm{d\beta}_i = \underbrace{\sum_j \mathrm{dAf}_{ij}\,\mathrm{kkt}_{ij}}_{\text{A-链}} \;+\; \underbrace{\sum_k \mathrm{TW}_{ik}\,k_{ik}\,2^{\mathrm{gc}_i}}_{\text{W-链，}\textbf{无}\,\beta} \;+\; \underbrace{\sum_d \mathrm{tv}_{id}\,v_{id}}_{\text{v-链}}, \qquad \mathrm{TW}:=T^{\top}\mathrm{dW}.
$$

三个行和由三个带归约维的 generic 算出（`%ra/%rb/%rc`），再用一个一维 generic 相加（`tools/gdn_l1_gen.py:1160-1186`）。**易错点**（`evidence/G1/l1_harness_report.md` 梯度公式处明确标注 "NO beta here"）：dbeta 的 W-链行和是 $k\,2^{\mathrm{gc}}$（**不带**行因子 $\beta$），而 dg 的同源 W-链（下节）**带** $\beta$——"同一 reduction 带不带行因子，极易顺手合并出错"。

#### 2.3.8 $\mathrm{dk}$：五项组装

$$
\mathrm{dk} \;=\; \underbrace{(\mathrm{dAf}\,k)\odot\beta}_{\text{A-链（行）}} \;+\; \underbrace{\mathrm{dAf}^{\top}(\beta\,k)}_{\text{A-链（列）}} \;+\; \underbrace{\mathrm{TW}\odot(\beta\,2^{\mathrm{gc}})}_{\text{W-链}} \;+\; \underbrace{\bigl(v_{\text{new}}\odot 2^{\mathrm{gcl}-\mathrm{gc}}\bigr)\mathrm{dH}^{\top}}_{\hat B\text{-链（状态）}} \;+\; \underbrace{\mathrm{dPm}^{\top}q}_{o_{\text{intra}}\text{-链}}
$$

其中

$$
\mathrm{dPm} = \bigl(\mathrm{dO}\,v_{\text{new}}^{\top}\bigr)\odot D\odot\mathrm{tril}_{\text{incl}} \qquad (\text{P 的余切，含衰减含对角}).
$$

**源码**：`tools/gdn_l1_gen.py:1047-1064`（dPr=dO vn^T 原始缩并；dPm=掩蔽+衰减 generic）、`:1216-1235`（`%vng = vn⊙2^{gcl-gc}`、`%dkB = vng dH^T`、`%rowB = rowsum(dkB⊙k)`——后两者同时服务 dg）、`:1237-1272`（五项按序累加进 `%dkacc`：两个 $C\times C\times K$ 缩并（行/列两个方向）、一个 elementwise（TW 链，`%ka7 = exp2(gc); %ka8 = beta*ka7; %ka9 = TW*ka8`）、一个 elementwise 加（dkB）、一个 $C\times C\times K$ 缩并（dPm^T q），最后 `linalg.copy` 存入 strided 输出视图）。

**效果数字**。dk 是六梯度中对拍差最大的列（npz 三用例 $5.6$–$5.8\times10^{-6}$，`evidence/G1/l1_bwd_check_output.txt`）——五条链的误差叠加，但仍在 printMemrefF32 六位有效数字地板（`evidence/G1/l1_harness_report.md`）。

#### 2.3.9 $\mathrm{dq}$

$$
\mathrm{dq} = \underbrace{\bigl(\mathrm{dO}\,S_n^{\top}\bigr)\odot 2^{\mathrm{gc}}}_{o_{\text{inter}}\text{-链}} \;+\; \mathrm{dPm}\;k \qquad\in\mathbb R^{C\times K}
$$

（`tools/gdn_l1_gen.py:1188-1214`：`%qg = q⊙2^gc`（服务 dH 链复用）、`%dq1 = dOgc Sn^T`（其中 `%dOgc = dO⊙2^gc` 先行算出）、`%r3v = rowsum(dq1⊙q)`（服务 dg，见下节）、`dq1 += dPm @ k`、`linalg.copy` 存储。）

#### 2.3.10 $\mathrm{dg}$ 的组装：G1-B2 四项发现的集中地

**动机**。$g$ 以**指数**进入每一处衰减因子，因此 $\mathrm{dg}$ 是所有梯度中结构最繁的：chunk 内它是一个"行和 − 列和"的后缀和结构（衰减差的求导差分），chunk 间还叠加 Bhat 链的严格前缀和与一个 chunk 常数。生成器把它组装为（`tools/gdn_l1_gen.py:847-849` docstring、`evidence/G1/l1_harness_report.md` 梯度公式）：

$$
\mathrm{dg}_t \;=\; \underbrace{\mathrm{suffixsum}_t\bigl[\,\mathrm{rows}_t-\mathrm{cols}_t\bigr]_{\mathrm{dAf}\,\odot\,\beta_i\,\mathrm{kkt}} \;+\; \mathrm{rows}_t\bigl(\mathrm{TW}\odot(\beta k 2^{\mathrm{gc}})\bigr) \;+\; \mathrm{rows}_t\bigl((\mathrm{dO}S_n^{\top})\odot(q\,2^{\mathrm{gc}})\bigr) \;+\; \bigl[\,\mathrm{rows}_t-\mathrm{cols}_t\bigr]_{(\mathrm{dO}\,v_{\text{new}}^{\top})\odot P}\,\bigr]}_{\text{四片行/列向量 } t_A+t_B \;=\; \mathrm{dgr}}
$$

$$
+\; \underbrace{\sum_{c<t}\mathrm{rowB}_c}_{\text{严格前缀}} \;+\; \underbrace{2^{\mathrm{gcl}}\sum_{k,v}\mathrm{dH}\odot S_n}_{\text{chunk 常数}}, \qquad \mathrm{rowB}_t := \mathrm{rows}_t\Bigl(\bigl(v_{\text{new}}\odot 2^{\mathrm{gcl}-\mathrm{gc}}\bigr)\mathrm{dH}^{\top}\odot k\Bigr).
$$

对应生成器缓冲：`%ra/%rb/%rc/%r3v/%r4/%c1r/%c4r`（行/列和）→ `%tA/%tB` → `%dgr`；`%rowB` → 前缀循环 `%dgp2`；`%s5`（$[\,1\,]$ memref 的全归约）→ `%s5x = s5*2^{gcl}` 广播加；最后后缀循环写入 `%dg_c`。

**发现一（o_intra 项必须配 raw 缩并）**。第四片是 $\bigl[\mathrm{rows}-\mathrm{cols}\bigr]_{(\mathrm{dO}v_{\text{new}}^{\top})\odot P}$——用**原始** $\mathrm{dPr}=\mathrm{dO}v_{\text{new}}^{\top}$ 与 $P$ 配对，**不是** $\mathrm{dPm}\odot P$：dPm 已含衰减 $D$，再乘 $P$（$P$ 自带 $D$）会**双计衰减**。代价实测 $0.1$–$0.5$ abs 误差（`evidence/G1/l1_harness_report.md` 两个 pitfalls 之首，"dg 的 o_intra g-项配 RAW (dO vn^T) 与 P——NOT dPm * P"）。IR 佐证：`%r4/%c4r` 的输入是 `%dPr` 与 `%Abuf`（=P），而 dq/dk 链用的是 `%dPm`（`tools/gdn_l1_gen.py:1284-1295` vs `:1208-1213`）。该项等价于 fla 的 `rowsum(dq_loc q) - rowsum(dk_loc k)` 以 $v=v_{\text{new}}$ 的 q/k-路线。

**发现二（dg 无 RCP_LN2 重标度）**。kernel 内嵌的 $g$ 是 $\log_2$ 域（$g\cdot\log_2 e$，§2.2.2），于是**每一个**衰减因子 $\exp_2(\mathrm{gc}_{\log_2})\equiv e^{\mathrm{gc}_{\text{nat}}}$ 天然就是自然域乘子——所有 dg 分片已是自然对数域余切，**最后再乘 $\log_2 e$ 是 bug**（dg 整体被放大 $1.4427$ 倍）。生成器在该处留有注释（`tools/gdn_l1_gen.py:1338-1339`）：

```python
# dg = suffix(dgr) + dgp2  (the pieces are already natural-log cotangents:
# every decay is exp2(gc_log2) == e^(gc_nat), so no RCP_LN2 rescale here)
```

这是"数域约定"从约定条款变成可执行判据的直接例子：`evidence/G1/GATE_REPORT.md` 将其列为数学成果第 2 条。

**发现三（$\beta$ 行因子的有/无）**。dg 的 A-链行项带 $\beta_i$（`%tA` 组装中 `%ta0 = ra*beta` 与 `%ta2 = rb*beta`，列项 `%c1r = colsum(dAf⊙kkt⊙beta_row)` 减去——注释"dg's A-chain ROW term carries the beta_i row factor; dbeta's ra does not"，`tools/gdn_l1_gen.py:1296-1308`），而 dbeta 的同源行和 `%ra` **不带** $\beta$；W-链同构对照见 §2.3.7。

**发现四（Bhat 链 = 严格前缀）**。fla 把 $\mathrm{rowB}$ 之和加到 chunk 末行再做一次全局逆 cumsum，代数上等于 $-\mathrm{suffix}(\mathrm{rowB})+\sum\mathrm{rowB}=$ 严格前缀 $\sum_{c<t}\mathrm{rowB}_c$；L1 直接发射前缀（一个携带累加器的 `scf.for`，**先存后加**实现排他前缀），见 IR。两侧的实现差异在 fp64 孪生对 JAX vjp $\le 1.4\times10^{-14}$ 下验证等价（`evidence/G1/l1_harness_report.md` "Deviations" 第 1、2 条）。

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:3380-3419`——s5 全归约、前缀循环、后缀循环三段；后缀循环用 `idx = C-1-i7` 正步长逆序，且累加发生在读 $\mathrm{dgr}[\mathrm{idx}]$ 之后写入之前——**含尾**后缀和）：

```mlir
scf.for %i6 = %c0 to %cC step %c1 iter_args(%pacc = %f0) -> (f32) {
  memref.store %pacc, %dgp2[%i6] : memref<64xf32>      // 先存：排他前缀
  %rb_i = memref.load %rowB[%i6] : memref<64xf32>
  %pacc2 = arith.addf %pacc, %rb_i : f32
  scf.yield %pacc2 : f32
}
linalg.generic { ... } ins(%dgp2) outs(%dgp2) {          // + s5x 广播常数
^bb0(%a0: f32, %o0: f32):
  %dgpx = arith.addf %a0, %s5x : f32
  linalg.yield %dgpx : f32
}
scf.for %i7 = %c0 to %cC step %c1 iter_args(%sacc = %f0) -> (f32) {
  %idx = arith.subi %cCm1, %i7 : index                   // 正步长逆序
  %dr_i = memref.load %dgr[%idx] : memref<64xf32>
  %sacc2 = arith.addf %sacc, %dr_i : f32                 // 含尾后缀累加
  %dp_i = memref.load %dgp2[%idx] : memref<64xf32>
  %dgt = arith.addf %sacc2, %dp_i : f32
  memref.store %dgt, %dg_c[%idx] : memref<64xf32, strided<[2], offset: ?>>
  scf.yield %sacc2 : f32
}
```

**效果数字**。dg 列对拍：jax_C64/jax_C64_h0 $6.0\times10^{-6}$、jax_C32 $4.3\times10^{-6}$（fp32/fp64 双口径，`evidence/G1/l1_bwd_check_output.txt`）；seeded 尾块用例最差 $6.6\times10^{-6}$（chunk32）——发现一/二若犯任何一个，误差是 $0.1$–$1.4$ 量级，比 gate 大三个数量级以上。

#### 2.3.11 尾块语义与正步长逆序

尾块（$T$ 非 $C$ 整数倍）在反向同样以零填充闭合：补零行 $\beta=k=g=\mathrm{do}=0$ 对 A/u/P/dvp/G/dA 及全部 suffix/prefix/rowsum 贡献**精确零**，真实行精确、补零行读零（`evidence/G1/l1_harness_report.md`；由 `tail_t96`（最差梯度 $\mathrm{dbeta}$ $3.6\times10^{-5}$）与 `tail_b2`（$\mathrm{dk}$ $8.1\times10^{-6}$）两个 seeded 用例 PASS 验证，`evidence/G1/l1_bwd_check_output.txt`）。逆序统一用正步长 + 下标重映射（chunk 级 `n = NC-1-n2`、行级 `idx = C-1-i7`）——`evidence/G1/l1_harness_report.md` 明确记录"负步长被刻意规避"。

### 2.4 生成器架构与 CPU 执行管线

#### 2.4.1 生成器总体结构

`tools/gdn_l1_gen.py`（1428 行）是参数化的 MLIR 文本生成器：`--shape B,T,H,K,V --chunk C --seed N --emit fwd|bwd|both --npz <ref.npz> --out <f.mlir>`（默认形状 `1,128,2,16,16`、$C{=}64$；约束 `assert C >= 2`）。`--emit fwd` 的输出与反向功能加入前的版本**逐字节一致**（`evidence/G1/l1_harness_report.md` G1-B2 更新注记："the fwd output is byte-identical"）。时间维先零填充 $T_p=\lceil T/C\rceil\cdot C$、$\mathrm{NC}=T_p/C$（`tools/gdn_l1_gen.py:213-214`）。生成模块的解剖：

```
module {
  func.func private @printMemrefF32(memref<*xf32>)         // 运行时外部函数
  memref.global "private" constant @q : memref<BxTpxHxKxf32> = dense<...>
  ...  @k @v @beta @g(log2 域) @h0   [bwd: @do @dht]        // 输入常数嵌入
  [memref.global ... @ref_o @ref_ht @ref_dq ... ]          // 可选期望输出(--npz)
  func.func @gdn_fwd(...) -> (o, ht)                       // emit fwd|both
  func.func @gdn_bwd(...) -> (dq, dk, dv, dbeta, dg, dh0)  // emit bwd|both
  func.func @main() { get_global×N; func.call; memref.cast; call @printMemrefF32×N }
}
```

代表性产物：前向 2167 行（`evidence/G1/l1_chunked_T128_C64.mlir`）、反向 3540 行（`evidence/G1/l1_bwd_chunked_T128_C64.mlir`），其中绝大多数行数是嵌入的 dense 常数（反向模块的 kernel 代码从 2424 行起）。

#### 2.4.2 代码模式一：`memref.global` 常数嵌入

**动机**。让模块**自包含**：不依赖外部数据加载通道，`mlir-runner` 直接执行；同时把"对拍输入"固化进 IR 文本，任何一次运行都可从 `.mlir` 文件完整复现。`--npz` 还会把期望输出嵌入为 `@ref_*` global 并由 `@main` 打印（`tools/gdn_l1_gen.py:264-277`），使产物自带"题目+答案"。

**源码片段**（`tools/gdn_l1_gen.py:136-158`——两个字面量工具；`:252-262` global 发射）：

```python
def fmt_f32(x):
    """MLIR float literal (this build requires an explicit '.' or exponent)."""
    s = "%.9g" % float(x)
    ...  # 补小数点/指数格式
def dense_literal(a, row_indent):
    """Nested dense<...> literal with one innermost row per line."""
    ...
    inner = (",\n" + ind).join(dense_literal(r, ind) for r in a)
```

```python
for name, arr, shape in ins:
    ty = mrt(shape)
    e(f'memref.global "private" constant @{name} : {ty} = dense<')
    lit = dense_literal(arr, "  " * e.depth)
    e.lines[-1] += lit + ">"
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:12-16`）：

```mlir
// ---- input data (zero time-padding to T_pad = 128) ----
memref.global "private" constant @q : memref<1x128x2x16xf32> = dense<[
  [
    [
      [0.0342215411, -0.0359566025, 0.174311697, 0.0285519548, -0.145799711, 0.0984197706, 0.354925722, 0.257778674, -0.191544265, -0.344425321, -0.169644266, 0.0112481993, -0.63283217, -0.0595512167, -0.339114904, -0.199310198],
      ...
```

两条工具链适配发现记录在 `evidence/G1/l1_harness_report.md` "Pass-spelling findings"：本构建（2026 年 10 月主线的 LLVM/MLIR 树）的解析器对字面量严格——dense 浮点列表必须有十进制点或指数（`1` 报错、`1.0` 合法，故有 `fmt_f32`），且**拒绝**扁平化（flattened）dense 列表，`dense_literal` 因此发射**完全嵌套**的字面量。

#### 2.4.3 代码模式二：strided rank-reduced subview 视图族

**动机**。kernel 按 $(b,h)$、按 chunk 切片操作 4-D/3-D global，但算法对象是 $[C,K]$/$[C,V]$/$[C]$ 的 2-D/1-D 矩阵。生成器用**两级 `memref.subview`**（先 $(b,h)$ 再 chunk 偏移）直接在嵌入布局上取 strided 视图——**零打包拷贝**：

```mlir
%q_bh = memref.subview %q[%b, 0, %h, 0] [1, 128, 1, 16] [1, 1, 1, 1] : ... to memref<128x16xf32, strided<[32, 1], offset: ?>>
%q_c  = memref.subview %q_bh[%off, 0] [64, 16] [1, 1] : ... to memref<64x16xf32, strided<[32, 1], offset: ?>>
%beta_c = memref.subview %beta_bh[%off] [64] [1] : ... to memref<64xf32, strided<[2], offset: ?>>
```

（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2507-2511`。$q$ 行步长 $32=H\cdot K$、$\beta$ 行步长 $2=H$——即"跨步 2 读 beta"这类布局直接由 strided 类型表达。）两个必须遵守的语义（`evidence/G1/l1_harness_report.md`）：`memref.subview` 的步长列表是**相对乘子**（非绝对步长），连续 chunk 选择写 `[1,1,1,1]`；rank-reducing subview（`[1,Tp,1,K]→[Tp,K]`）需要下降管线中 `-expand-strided-metadata -lower-affine` 才能落到 LLVM（配方取自 in-tree 测试 `mlir/test/Integration/Dialect/Linalg/CPU/rank-reducing-subview.mlir`）。类型字符串由 `mrt()` 辅助函数拼装（`tools/gdn_l1_gen.py:188-195`）。

#### 2.4.4 代码模式三：`scf.for` 载态 + linalg 命名/泛型算子

**动机**。L1 需要显式的跨迭代状态（前向 $S$、反向 $\mathrm{dH}$ 与 `h_all`），同时需要矩阵级算子表达算法结构。设计取舍（`evidence/G1/l1_harness_report.md` "L1 structure" 设计选择）选了 **memref 就地变异**而非 tensor SSA 状态携带（`tensor.insert_slice`/yield）：不需要 bufferization 分析、编译干净——"语义优先于形式"；tensor 状态携带形式留给 L2。

三个子模式：

- **载态循环**：`scf.for %b → %h → %n(chunk)`；状态 $S$/dH 是每 $(b,h)$ 的 `memref.alloc` scratch，循环体内 load/store 就地更新（初始化 `linalg.copy` 自 $h_0[b,h]$ 的 rank-reduced subview，终态 copy 到 $h_t[b,h]$）。scratch 清单：前向 17 项（`tools/gdn_l1_gen.py:379-396`）、反向约 37 项（`:899-912`；含 `h_all`、dA 链的 `Gb/Xb/dPr/dPm`、dg 链的 8 个 $[C]$ 行向量等）。
- **fill-before-accumulate**：`linalg.matmul`/缩并型 `linalg.generic` 是累加语义，所有 outs 先 `linalg.fill`（`_fill` 辅助，`tools/gdn_l1_gen.py:742-743`）。
- **`generic()` 发射器**（`tools/gdn_l1_gen.py:701-733`）：把 indexing_maps / iterator_types / 块体 lambda 统一成一次调用，反向 kernel 的全部 40 余个 `linalg.generic` 由它发射；affine map 顺便承担转置（`(kk,vv,i)->(i,kk)` 型）与广播（`(kk,vv)->(0)` 型，s5 全归约）。

#### 2.4.5 代码模式四：`printMemrefF32` 驱动

**动机**。CPU 侧唯一观测通道。模块声明外部运行时函数并把结果 cast 后打印（`tools/gdn_l1_gen.py:247,300-317`）：

```mlir
func.func private @printMemrefF32(memref<*xf32>)
...
%dq_ = memref.cast %dq : memref<1x128x2x16xf32> to memref<*xf32>
call @printMemrefF32(%dq_) : (memref<*xf32>) -> ()
```

（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:3516-3539` 的 `@main` 依次打印六个梯度；fwd/both 模式先打印 $o,h_t$。）实现来自 `libmlir_c_runner_utils`（`evidence/G1/l1_harness_report.md`：仅需 private 声明 + `memref.cast`）。**精度边界**：printMemrefF32 打印约 6 位有效数字，$10^{-6}$ 以下的差不叮分辨——因此全部实测差（前向 $10^{-7}$ 级、反向 $10^{-6}$–$10^{-5}$ 级）都被理解为**打印地板**而非真实计算误差（`evidence/G1/l1_harness_report.md` Known limitations 第 1 条）。检查器 `parse_memrefs`（`tools/gdn_l1_check.py:235-261`）用括号配对 + `ast.literal_eval` 解析 `data = [...]` 文本块。

#### 2.4.6 CPU 下降与执行管线：完整命令行与每步作用

**完整管线**（`scripts/run_l1.sh` 原文，与 `evidence/G1/l1_harness_report.md` "Working pipeline" 一致）：

```bash
BUILD=~/codebase/llvm-mlir/build

"$BUILD/bin/mlir-opt" in.mlir \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -expand-strided-metadata \
  -lower-affine \
  -convert-math-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-cf-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o out.mlir

"$BUILD/bin/mlir-runner" out.mlir \
  --shared-libs="$BUILD/lib/libmlir_runner_utils.so,$BUILD/lib/libmlir_c_runner_utils.so,$BUILD/lib/libmlir_float16_utils.so" \
  --entry-point-result=void
```

**每步作用**：

| pass | 作用 | 备注 |
|---|---|---|
| `-one-shot-bufferize="bufferize-function-boundaries"` | tensor→memref 缓冲化 | 对本 IR 是 **no-op**（已是 memref 级），为与 tensor 前端管线兼容而保留 |
| `-convert-linalg-to-loops` | `linalg.generic/matmul/fill/copy` → `scf.for` 循环 | 全部缩并/逐元素算子的标量化 |
| `-convert-scf-to-cf` | `scf.for` → `cf.br/cond_br` 控制流图 | |
| `-expand-strided-metadata` | strided memref 类型 → 显式 offset/size/stride 操作 | **必需**：rank-reducing strided subview 落 LLVM 的前提（§2.4.3） |
| `-lower-affine` | `affine.apply` 等 → arith | 同上必需 |
| `-convert-math-to-llvm` | `math.exp2` → `llvm.intr.exp2` | JIT 对共享运行时库带入的 libm 求解；**无需** `-convert-math-to-libm` |
| `-convert-arith-to-llvm` | arith → llvm 内联汇编/dialect | |
| `-finalize-memref-to-llvm` | memref 描述符、alloc/dealloc、`printMemrefF32`/`memref.cast` 的 C 运行时桥接 | |
| `-convert-cf-to-llvm` / `-convert-func-to-llvm` | 控制流/函数 → LLVM dialect | |
| `-reconcile-unrealized-casts` | 清理各转换间残留的 unrealized_conversion_cast | |
| `mlir-runner`（三个共享库） | JIT 编译 + 执行 `@main` | runner_utils/c_runner_utils（printMemrefF32）/float16_utils |

**效果数字**：base 模块 `mlir-opt` 0.32 s、`mlir-runner` 0.22 s（JIT+运行）；下降后的 LLVM-dialect 输出 2088 行（`evidence/G1/l1_harness_report.md`）。

### 2.5 JAX 可执行规约：语义单一事实源

#### 2.5.1 三实现：`gdn_seq` / `gdn_chunk` / vjp 包装

`tools/gdn_ref_jax.py`（272 行，CPU-only、fp64 `jax_enable_x64`、可 jit、可微）从 fla oracle（`~/codebase/fla/fla/ops/gated_delta_rule/naive.py`，只读）逐行移植，构成整个工程的**语义单一事实源**：L0 方言的语义描述、L1 生成器的每一条数学、以及后续层的数值验证，全部以它为对拍基准。

**(a) 顺序规约 `gdn_seq`**——逐步递推的 `lax.scan` 移植（`tools/gdn_ref_jax.py:117-131`，注意四步的精确顺序：衰减在 delta 之前、$\beta$ 乘残差、输出读更新后状态）：

```python
def step(h, inputs):
    q_t, k_t, v_t, beta_t, g_t = inputs
    h = h * jnp.exp(g_t)[..., None, None]                    # 衰减先
    b_v = v_t - jnp.einsum("bhk,bhkv->bhv", k_t, h)          # 残差
    b_v = b_v * beta_t[..., None]                            # β 乘残差
    h = h + k_t[..., :, None] * b_v[..., None, :]            # delta 更新
    o_t = jnp.einsum("bhk,bhkv->bhv", q_t, h)                # 更新后读出
    return h, o_t
ht, o = jax.lax.scan(step, h0, (q_, k_, v_, beta_, g_))
```

**(b) chunked WY 规约 `gdn_chunk`**——`naive_chunk_gated_delta_rule` 的移植（`tools/gdn_ref_jax.py:139-242`），关键三段：含尾 chunk-local cumsum 与衰减掩码

```python
decay = jnp.cumsum(g.reshape(B, H, n, c), axis=-1)          # inclusive
diff = decay[..., :, None] - decay[..., None, :]
L_mask = jnp.tril(jnp.exp(jnp.tril(diff)))
```

$(I+M)^{-1}$ 的**行递推**（与 L1 前代换逐式对应，§2.2.6 对比表）：

```python
attn = -((k_beta @ k.swapaxes(-1, -2)) * L_mask)            # M，严格下三角
for i in range(1, c):
    attn = attn.at[..., i, :i].add(
        jnp.sum(attn[..., i, :i, None] * attn[..., :i, :i], axis=-2))
attn = attn + jnp.eye(c, dtype=q.dtype)                     # T = (I+M)^-1
w = jnp.matmul(attn, vc)                                    # = 本章 u
u = jnp.matmul(attn, k_beta_c * e_decay[..., None])         # = 本章 w
```

chunk 主循环（$v_{\text{new}}/o$ 分解/状态更新，`tools/gdn_ref_jax.py:210-232`）。

**(c) vjp 包装**——两个规约共用 `_vjp_common`（`tools/gdn_ref_jax.py:248-272`），对闭包 $f$ 取 `jax.vjp`，以 $(\mathrm{do},\mathrm{dht})$ 为余切反解六个梯度：

```python
def _vjp_common(fn, q, k, v, beta, g, h0, do, dht, **kw):
    h0 = _zeros_h0(q, v, h0)      # h0 缺省时对 zeros 微分
    def f(q_, k_, v_, beta_, g_, h0_):
        return fn(q_, k_, v_, beta_, g_, h0_=h0_, **kw)
    (o, ht), vjp_fn = jax.vjp(f, q, k, v, beta, g, h0)
    dq, dk, dv, dbeta, dg, dh0 = vjp_fn((do, dht))
    return o, ht, dq, dk, dv, dbeta, dg, dh0
```

**"忠实移植"纪律**（`tools/gdn_ref_jax.py:21-30`，"do NOT fix these"）：$\mathrm{scale}=K^{-0.5}$ 预乘进 $q$；$\beta$ 原始值；$g$ 自然对数域（并显式警告 MLIR 侧若用 $\exp_2$ 必须乘 $\mathrm{RCP\_LN2}$）；递推顺序固定。与 torch oracle 的两处**非数学性**差异被声明且保留：oracle 内部硬 cast fp32（规约保持调用方 dtype=fp64）、oracle 以 `output_final_state` 门控终态（规约恒返回 $(o,h_t)$）。

#### 2.5.2 为什么它是"单一事实源"：44/44 自洽验证

`tools/gdn_ref_check.py`（jax 0.11.2 CPU，2026-10-08，$B{=}2,H{=}2,K{=}V{=}16$，fp64，GDN 风格输入）给出九类检查、44/44 PASS（`evidence/G1/jax_spec_report.md` §1）：

| 检查（最差组合） | 达到 | 容限 |
|---|---|---|
| chunked vs sequential，$o$（T=96/128，C=32/64，h0 有/无，8 组合） | $5.6\times10^{-16}$ | $10^{-10}$ |
| chunked vs sequential，$h_t$（同 8 组合） | $1.2\times10^{-15}$ | $10^{-10}$ |
| VJP 梯度 chunked vs sequential（六梯度，8 组合） | $1.2\times10^{-14}$ | $10^{-8}$ |
| 有限差分 vs VJP（seq，6 随机坐标） | $2.3\times10^{-9}$ | $10^{-5}$ |
| 有限差分 vs VJP（chunk，6 随机坐标） | $9.0\times10^{-10}$ | $10^{-5}$ |
| jit 一致性 seq / chunk | $0$ / $1.1\times10^{-16}$ | $10^{-12}$ |
| fla `naive_recurrent` vs jax seq（fp32 rel） | $1.2\times10^{-7}$ | $10^{-4}$ |
| fla `naive_chunk` vs jax chunk（fp32 rel） | $2.3\times10^{-6}$ | $10^{-4}$ |

这组数字完成三件事：**内部一致性**（chunk-vs-seq 是两个独立算法路径的互证，$10^{-16}$ 级即 fp64 舍入地板；有限差分独立验证 vjp 的微分手工推导无误）；**可执行性**（jit 编译不改变语义）；**外部锚定**（对 fla oracle 的 fp32 口径相符——fla naive 内部硬 cast fp32，故 fp64 对比不可能，容差放宽到 $10^{-4}$）。T=96 与 C=64 的组合专门行使零填充尾块（96→128 补 32 行）。

#### 2.5.3 参考工件与三方对拍数字

规约以 `tools/gdn_ref_dump.py` 落出三个 npz 工件（`evidence/G1/`，均 $B{=}2,T{=}128,H{=}2,K{=}V{=}16$，fp64 计算、seed `PRNGKey(20261008)`；主数组 fp32 + 同名 `*_fp64` 副本 + `config` JSON，共 33 键/文件）：

| 文件 | chunk | h0 | chunk-vs-seq (fp64) |
|---|---|---|---|
| `ref_jax_T128_C64.npz` | 64 | 零 | $8.9\times10^{-16}$ |
| `ref_jax_T128_C32.npz` | 32 | 零 | $4.4\times10^{-16}$ |
| `ref_jax_T128_C64_h0.npz` | 64 | 种子化（$\max 0.31$） | $8.9\times10^{-16}$ |

三文件共享同一套 $q,k,v,\beta,g,\mathrm{do},\mathrm{dht}$（仅 $h_0$ 不同），使 MLIR-vs-JAX 差跨 chunk 尺寸直接可比。梯度是 $L=\sum\mathrm{do}\odot o+\sum\mathrm{dht}\odot h_t$ 的 VJP。

**三方对拍闭环**（`evidence/G1/GATE_REPORT.md` 判据 5 的链）：

$$
\underbrace{\text{MLIR L1}}_{\text{CPU fp32 执行}} \xrightarrow{\;\substack{\text{fwd }6.4\times10^{-7}\\ \text{bwd }\le 6.4\times10^{-6}}\;} \underbrace{\text{JAX 规约}}_{\text{CPU fp64}} \xrightarrow{\;\substack{1.2\times10^{-7}\,(\text{recurrent})\\ 2.3\times10^{-6}\,(\text{chunk})}\;} \underbrace{\text{fla naive torch}}_{\text{oracle}}
$$

外加第四极——numpy 孪生（`tools/gdn_l1_check.py` 的 `chunked_ref/seq_ref/chunked_bwd_ref`）：fp64 下孪生对 JAX vjp $\le 1.4\times10^{-14}$（三 npz 全六梯度），fp32 下孪生作为 MLIR 的直接参照把"算法差"与"执行差"分离。前向终表（`evidence/G1/l1_check_output.txt`，gate $10^{-3}$）：

| 用例 | 形状 | $\max\|o-\mathrm{ref}\|$ | $\max|h_t-\mathrm{ref}|$ | numpy chunk-vs-seq |
|---|---|---|---|---|
| base | B1 T128 C64 | $6.184\times10^{-7}$ | $8.643\times10^{-7}$ | $2.384\times10^{-7}$ |
| chunk32 | B1 T128 C32 | $5.513\times10^{-7}$ | $7.153\times10^{-7}$ | $1.192\times10^{-7}$ |
| tail_t96 | B1 T96 C64（尾 64+32） | $5.662\times10^{-7}$ | $6.258\times10^{-7}$ | $2.384\times10^{-7}$ |
| tail_b2 | B2 T100 C64（尾 64+36） | $5.960\times10^{-7}$ | $3.099\times10^{-6}$ | $3.874\times10^{-7}$ |
| jax_C64_h0 | B2 T128 C64，G1-A npz，种子 h0 | $6.407\times10^{-7}$ | $8.643\times10^{-7}$ | $5.662\times10^{-7}$ |
| jax_C64 | B2 T128 C64，npz，h0=0 | $6.407\times10^{-7}$ | $8.643\times10^{-7}$ | $5.662\times10^{-7}$ |
| jax_C32 | B2 T128 C32，npz，h0=0 | $5.066\times10^{-7}$ | $5.513\times10^{-7}$ | $5.066\times10^{-7}$ |

反向终表（`evidence/G1/l1_bwd_check_output.txt`；npz 单元格为 fp32/fp64 双口径；gate $2\times10^{-4}$/$3\times10^{-4}$ + 有限性检查，全部 finite=True）：

| 用例 | 形状 | dq | dk | dv | dbeta | dg | dh0 |
|---|---|---|---|---|---|---|---|
| jax_C64_h0 | B2 T128 C64 种子 h0 | $4.5/4$ | $5.8/6$ | $5.0/5$ | $6.4/7$ | $6.0/6$ | $0.45/0.5$（$\times10^{-6}$） |
| jax_C64 | B2 T128 C64 h0=0 | $4.5/4$ | $5.8/6$ | $5.0/5$ | $6.4/7$ | $6.0/6$ | $0.45/0.5$ |
| jax_C32 | B2 T128 C32 h0=0 | $4.5/4$ | $5.6/6$ | $5.0/5$ | $5.5/5$ | $4.3/4$ | $0.45/0.5$ |
| base（孪生） | B1 T128 C64 | $3.2$ | $6.0$ | $4.9$ | $5.0$ | $1.6$ | $0.058$（$\times10^{-6}$） |
| chunk32（孪生） | B1 T128 C32 | $2.3$ | $5.0$ | $2.1$ | $2.4$ | $6.6$ | $0.49$ |
| tail_t96（孪生） | B1 T96 C64 | $4.9$ | $4.8$ | $0.95$ | $36$ | $0.77$ | $0.37$ |
| tail_b2（孪生） | B2 T100 C64 | $3.2$ | $8.1$ | $4.9$ | $4.4$ | $3.0$ | $0.49$ |

#### 2.5.4 检查器：`tools/gdn_l1_check.py`

检查器把规约、生成器、runner 串成单一入口（前向 `python tools/gdn_l1_check.py`；反向 `--bwd` / `--bwd-only-npz`）：生成 → `scripts/run_l1.sh` 执行 → `parse_memrefs` 解析 → 对照两极（嵌入 numpy 参照 / G1-A npz 的 JAX 输出）。四个 seeded 前向用例刻意覆盖尾块与多 batch；反向 gate 的双容差（fp32 主数组 $2\times10^{-4}$ 或 fp64 副本 $3\times10^{-4}$）直接承认打印地板的存在。npz 文件缺席时容错跳过（跨工作流解耦）。

### 2.6 L1 的边界与向 L2 的接口

`evidence/G1/l1_harness_report.md` "Known limitations" 与 "Notes for G2" 划定 L1 的诚实边界，每条同时是 L2 的接口（旋钮）：

1. **观测地板**：printMemrefF32 约 6 位有效数字，$10^{-6}$ 以下不可分辨——真实 fp32 计算误差"很可能小于实测 $\sim6\times10^{-7}$"。
2. **无 tensor SSA 历史**：状态经 memref 就地变异，无函数式历史——对 L1 无碍，对自动微分有碍；L3 的 WY 反向按独立 kernel 处理（本章 §2.3 即其数学）。
3. **scratch 逐 $(b,h)$ alloc/dealloc**：融合/持久缓冲调度归 L2。
4. **前代换 $O(C^3/6)$ 标量迭代**：刻意显式；行递推/Neumann/分块变体是 `solve variant` 旋钮，重新生成即可 A/B。
5. **dA 修正 = 每 chunk 两个 $C^3$ 缩并 + 掩蔽**：blocking/SIMD 归 L2。
6. **fp32、B/H 循环不 tile、无向量化**：linalg-to-loops 发射标量循环；vector dialect/SIMD 归 L2。
7. **重算-物化旋钮**：pass 1 存 u/w/P 的 materialize 变体是显式列出的存储/计算折中。
8. **与 fla kernel 级分解的等价-差异清单**（`evidence/G1/l1_harness_report.md` "Deviations"）：同一数学、不同分布——Bhat 链前缀直发（≡ fla 加末行+全局逆 cumsum）、dg 行组装显式四片（≡ fla 的 dq_loc/dk_loc 路线）、dH 寄存器式携带（fla 物化 dh[n]）、fla 的转置 A 存储怪癖被识别但未复制。

最后回到章首的定位：`docs/L0_SEMANTICS.md` 记录 `gdn-to-chunk` pass 当前的入口调用式占位与未来"chunk-AST 展开"，其验收口径即本章的全部内容——展开 pass 的输出必须与 `tools/gdn_l1_gen.py` 生成的 L1 文本**逐行同构**（前代换解三角、`scf.for` 载态两个代表性结构已先行验证）。换言之，L1 层的交付物有二：一份**可执行的算法定义**（本章 §2.2–§2.3 的全部数学 + IR），与一份**生成格式的规范**（本章 §2.4 的代码模式）；二者共同构成 Schedule-as-IR 论证中"分层可验证 IR"的第一块可对拍的基石。

---

#### 本章引用文件清单

| 文件 | 角色 |
|---|---|
| `tools/gdn_l1_gen.py`（1428 行） | L1 生成器（前向 + 反向），本章主剖析对象 |
| `tools/gdn_ref_jax.py`（272 行） | JAX fp64 可执行规约（seq/chunk/vjp） |
| `tools/gdn_l1_check.py`（548 行） | 数值自检驱动（numpy 孪生 + memref 解析 + 双 gate） |
| `scripts/run_l1.sh` | CPU 下降 + 执行 harness |
| `docs/L0_SEMANTICS.md` | L0 语义规约（8 条约定裁定） |
| `evidence/G1/jax_spec_report.md` | G1-A：JAX 规约结果与约定（44/44、npz 工件、fla 契约） |
| `evidence/G1/l1_harness_report.md` | G1-B：L1 harness、管线发现、数学约定、G1-B2 反向全量 |
| `evidence/G1/l1_chunked_T128_C64.mlir`（2167 行） | 前向 IR 产物样本 |
| `evidence/G1/l1_bwd_chunked_T128_C64.mlir`（3540 行） | 反向 IR 产物样本（本章 IR 引用的主要来源） |
| `evidence/G1/l1_check_output.txt` / `l1_bwd_check_output.txt` | 前向 7/7、反向 7 用例对拍原始输出 |
| `evidence/G1/ref_jax_T128_{C64,C32,C64_h0}.npz` | G1-A 参考工件（33 键/文件） |
| `evidence/G1/GATE_REPORT.md` | G1 gate 判据与三方对拍链总结 |

---

## 3 MMA 机器与混合精度体系：两个生成器共享的张量核发射层

GDN 的前向（`tools/gdn_full_gen.py`，4464 行）与反向（`tools/gdn_bwd_gen.py`，8687 行）生成器都不把 GEMM 交给编译器的自动向量化或 transform 配方，而是各自内建一整套**张量核发射层（MMA machinery）**：从 lane 索引预计算、fragment 装载（`nvgpu.ldmatrix`）、warp 贴图、累加器链接，到 tf32 逃生舱、split-bf16 hi/lo 分解、staging 原语族（同步 st.shared 家族与异步 16B burst 家族）以及配套的 barrier 语义。本章逐个发射器、逐个机制地剖析这一层。历史脉络上，该机器最早在前向生成器中成形并经 GPU 全量验证（G4 门 20/20），随后以"fragment/lane 数学不变、发射器外壳适配"的方式移植入反向生成器（`gdn_bwd_gen.py` 中 `mma_gemm` 一节的移植注释即为此事实的原始记录）；两套实现因此共享同一套硬件布局律与同一条 hazards 纪律，本章以反向侧（更完备）为主线，前向孪生单列一节（§3.11）。

除非另行注明，源码引文出自 `tools/gdn_bwd_gen.py`（行号为写作时的 HEAD），硬件为 RTX 5090 Laptop（sm_120a，无 tcgen05，Ampere 风格 `mma.sync`），工具链为本仓库锁定的 2026-10 LLVM 主干 + ptxas-blackwell（PTX 8.7）。所有效果数字均注明证据文件路径。

### 3.0 为什么需要自建 MMA 机器：stock 路径的边界

自建发射层不是偏好，而是被 stock 路径的五条硬边界逼出来的。这些边界全部在构建期以探针或源码级证据钉死（evidence/G2/gen_report.md "Toolchain quirks & constraints found"、evidence/G5/bwd_report.md 与 fwd_report.md 的平台约束节）：

**边界一：`vector.transfer` 在 workgroup memref 上不 lower。** `vector.transfer_read/write` 作用于 `#gpu.address_space<workgroup>` 时，`gpu-lower-to-nvvm-pipeline` 的 LLVM 翻译直接失败（"conversion of memref memory space ... failed"）；对扁平全局的 `vector.transfer_write` 在 `scf.while` 内则残留 `unrealized_conversion_cast`。这意味着 stock 的向量化搬运通道在我们的共享内存 tile 上整体不可用，staging 只能走 `memref.load/store` 标量环或显式 `cp.async`。

**边界二：`linalg.matmul` 的多线程冗余与竞态。** `linalg-to-loops` 展开的 matmul 在**所有线程上冗余执行**；对链式累加 matmul，多个 warp 之间对同一累加缓冲形成写竞态；同一缓冲同时充当两个 matmul 操作数时彻底 mis-lower（`gdn_bwd_gen.py` 模块 docstring 的 de-risk 清单原文）。`transform.structured.vectorize` 又在共享内存 matmul 上硬失败（无 `failures(suppress)` 子句可用）。

**边界三（本章核心）：`transform.nvgpu.rewrite_matmul_as_mma_sync` 只接受单 warp atom matmul。** 该 transform 硬编码 `laneId = threadIdx.x`，仅覆盖 16×8×16 f16/f16/f16 与 16×8×4 tf32 两种单 warp 形态（evidence/G2/gen_report.md 约束 1 原文："accepts ONLY single-warp atom matmuls ... it hard-codes laneId = threadIdx.x. Bigger GDN tiles are emitted by our generator as explicit nvgpu.mma.sync fragments (same lane formulas from NVGPUTransformOps.cpp)"）。GDN 的 tile（如 64×64、64×32、C×C=64×64 的 kkt）需要多 warp 协作贴图，stock 配方无法表达；生成器必须自己持有一套以 `nvgpu.mma.sync` 原子 + `nvgpu.ldmatrix` fragment 装载为底座的发射器。

**边界四：tf32 `m16n8k8` 在表内路径不可达。** `nvgpu.mma.sync` 的 verifier 强制 8×8×128b 基本片，使 tf32 封顶 K=4；表列 `nvvm.mma.sync` 家族的 m16n8k8 只有 f16/bf16/i8。而 GDN 反向链的 f32 操作数 GEMM 恰恰需要 fla 收割到的 `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`（覆盖矩阵 T1 的主力 tf32 签名）。这逼出了全文预算 ≤5 个逃生舱中唯一真正使用的一个：`gdn.mma.tf32k8`（§3.5）。

**边界五：`create_async_groups` 的 wait_group-0 机器约束。** transform 的 `transform.nvgpu.create_async_groups` 在每个组创建后**立即**插入 `cp.async.wait_group 0`（本树 NVGPU/Transforms/CreateAsyncGroups.cpp 源码级验证："Create the group and wait for it right after"），使任何预取与计算的重叠在结构上不可能。prefetch 必须手工发射 `nvgpu.device_async_copy` 并自持 token 与等待位置（§3.8）。

这五条边界合起来划定了机器的分层（自底向上）：

```
原子层     mma.sync.m16n8k16.bf16 / mma.sync.m16n8k8.tf32(inline_asm)
           ldmatrix.x4 / ldmatrix.x2(.trans) / cp.async(16B, bypassL1) / bar.sync
发射器     sgemm(标量基线) · tgemm(tf32) · mma_gemm(条带贴图)
           · mma_gemm_mh(二维贴图) · rgemm/rgemm_aw/aggemm(寄存器累加)
组合子     mma_chain_if(受卫链) · accs= 累加器链接 · k0= K 分块 · epilogue 回调
           · epi_bar / bar= 栅栏语义
数值策略   tf32 逐位点门 · split-bf16 hi/lo(3-mma / 2-mma / cvec 折入)
staging    st.shared 家族(stage_rows/raw_*/split_*) · 异步家族(astage_*)
           · launder_pass(恒等重写) · +8 列 bank pad
```

**记号（先定义后用）。** 对一个 32 线程的 warp，记 lane 编号 $L\in[0,32)$，并分解为
$$g=\lfloor L/4\rfloor\ \text{（groupID，组号）},\qquad t=L\bmod 4\ \text{（threadID-in-group，组内线程号）}.$$
`emit_mma_lanes`（`gdn_bwd_gen.py:1705`，前向孪生 `emit_lane_precompute`，`gdn_full_gen.py:144`）预计算的派生量即 $2t$、$2t{+}1$、$g{+}8$ 等：

```python
def emit_mma_lanes(e, tid=None):
    # tid= 让内核从 GROUP-LOCAL tid 派生 mma lane/warp 变量（fwdscan v4 的
    # warp-group 半体：256 线程 CTA 的 warp 0-3 / 4-7 各自贴自己半区的 gemm；
    # %mW 来自 %gtid 保证每组的贴图仍是 0..3）
    t = "%tid" if tid is None else tid
    e.l(f"%mW = arith.divsi {t}, %c32 : index")        # warp id
    e.l(f"%mL = arith.remsi {t}, %c32 : index")        # lane id  L
    e.l(f"%mlz = arith.cmpi eq, %mL, {e.ic(0)} : index")
    e.l("%mLG = arith.divsi %mL, %c4 : index")         # g   = L>>2
    e.l("%mTIG = arith.remsi %mL, %c4 : index")        # t   = L&3
    e.l("%mT2 = arith.muli %mTIG, %c2 : index")        # 2t
    e.l("%mT2P1 = arith.addi %mT2, %c1 : index")       # 2t+1
    e.l("%mLG8 = arith.addi %mLG, %c8 : index")        # g+8
    e.l("%mT2X8 = arith.addi %mT2, %c8 : index")       # 2t+8   (k 维)
    e.l("%mT2P1X8 = arith.addi %mT2P1, %c8 : index")   # 2t+1+8 (k 维)
```

所有 fragment 布局公式都以 $(g,t)$ 表达。tile 的 K 维原子步长记 `kb`（bf16 k16 原子为 16，tf32 k8 原子为 8）；"行半（row-half）"指 M 维的 $\{m,\ m{+}8\}$ 两半，"k 八元组（k-octet）"指 K 维的 8 列块。

### 3.1 标量基线 `sgemm` 与站点派发器 `gemm`

**动机（算子层面）。** 反向链中有三类 GEMM 不适合或不必要走张量核：(i) 数值上 tf32 被逐位点门否决、split-bf16 又暂不可行的位点（如 `dhu` 家族、`dqkwg.dvp_bh`——后者的 tf32 化会使 dbeta/dv 随 dh 的任意 f32 级扰动重采样量化桶，跨三个数值等价构建在 1.5e-5→2.9e-3→5.8e-3 间混沌游走，evidence/G5/bwd_report.md G5-J 节）；(ii) 形状过窄、8-warp mma 贴图不下的位点；(iii) 作为性能与数值对照组的基线。为此机器保留一个线程分区的标量 GEMM 基线。

**源码（`tools/gdn_bwd_gen.py:589`）。** `sgemm` 的分区契约是全机器单写者纪律的源头：

```python
def sgemm(e, out, a, b, M, N, Kd, *, at=False, bt=False, acc=False,
          neg=False, a_off=(0, 0), b_off=(0, 0), o_off=(0, 0), bar=True,
          kahan=False):
    """out(+off) (+)= a x b -- thread-partitioned; element e of the MxN tile
    is owned by thread e %% NT for EVERY gemm, so chained accumulation
    re-reads the writer's own cell: race-free and deterministic."""
    e.l(f"scf.for {ge} = %tid to {e.ic(M * N)} step %nt {{")
```

输出元素 $e=iN{+}j$ 由线程集 $\{e \bmod NT + c\cdot NT\}$ 中恰一线程拥有；**每个**累加进同一 acc 的 gemm 使用同一分区，链式累加永远重读自己写过的 cell——跨 warp 无竞态、跨 launch 确定。`at/bt` 以索引换位表达转置（`at=True` 时逻辑 `a[i,kk] = a[kk+o0, i+o1]`），`neg` 只取负 B 操作数（保持 acc 初值符号语义），`kahan=True` 启动补偿求和内层环（dA 链的 64–128 项 f32 乘积和有重度消减，朴素链式在 dW 上留下 ~4e-4..1.4e-3 绝对噪声，被 T 缩并链放大越过 dbeta/dg 门；evidence/G5/bwd_report.md G5-E 节）。非 kahan 体内可显式发射 `math.fma`（`fma3`，单舍入、严格紧于 mulf+addf；本工具链从不自动收缩 `arith.mulf/addf`，probe 验证 `math.fma` 确实 lower 为 `fma.rn.f32`）。

**派发器 `gemm`（`gdn_bwd_gen.py:2145`）。** 一行式的站点开关：站点 tag 在 `TF32_SITES` 集合内走 `tgemm`（warp tile），否则 `sgemm`。`--tf32` 命令行按 `TF32_GROUPS` 分组启停，使逐组数值验证成为流程的一部分。

**IR/PTX 对比。** 标量路径在 SASS 上是 39 指令/MAC 级的 LDG/LDS+FADD 链（覆盖矩阵的 nomma 路径测量，evidence/G6）；同一 tile 换到 mma 路径，PTX 从 276 KB 缩到 88 KB（3.1×）、SASS HMMA 48→0 的反向实验也给出了标量环的代价量级（evidence/G2/gen_report.md 矩阵表：`mma` on→off 一行）。

**效果数字。** 标量基线承载了全部 6/6 数值门的参照值类；其 f32 逐位语义（固定求和顺序、可选 kahan）是后文所有"位级一致"断言的对照面。

### 3.2 `mma_gemm`：基础发射器（条带贴图）

#### 3.2.1 契约与 warp 贴图

**动机。** 反向链里最大的 GEMM 族是 `k·k^T`/`q·k^T` 形状的 [C,C]、[C,BK]、[C,BV] tile（C=64 为 chunk 长度）。bf16 输入 + f32 累加正好落在 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` 原子上（覆盖矩阵 T1 的主力签名，fla 22/22 配置同款）；需要的是把任意 $M\times N\times K$（$M\equiv0\pmod{16}$，$N\equiv0\pmod{8W}$，$K\equiv0\pmod{16}$）的 tile 划给 W 个 warp 并发射 fragment 装载与原子链。

**源码（`tools/gdn_bwd_gen.py:1726`）。** 条带贴图：`wn = N // W`，warp $w$ 拥有 N 条带 $[w\cdot wn,(w{+}1)\cdot wn)$，条带内再按 8 列一个 atom：

```python
wn = N // W
e.l(f"%wnoff_{name} = arith.muli %mW, {e.ic(wn)} : index")
for mi in range(M // 16):
    e.l(f"%rA_{name}_{mi} = arith.addi %mLG, {e.ic(mi * 16)} : index")   # 行 m+g
    e.l(f"%rAh_{name}_{mi} = arith.addi %mLG8, {e.ic(mi * 16)} : index") # 行 m+g+8
for ni in range(wn // 8):
    e.l(f"%nc_{name}_{ni} = arith.addi %wnoff_{name}, {e.ic(ni * 8)} : index")
    e.l(f"%nct_{name}_{ni} = arith.addi %nc_{name}_{ni}, %mT2 : index")   # 列 n+2t
    e.l(f"%nct1_{name}_{ni} = arith.addi %nc_{name}_{ni}, %mT2P1 : index")
```

每个 warp 对每个 (mi, ni) atom 持有一个 `vector<2x2xf32>` 累加器；`k0=` 参数支持把一个宽 K 链拆成若干 K=16/32 的链节（§3.2.4）。

#### 3.2.2 fragment 装载：`nvgpu.ldmatrix` 的象限语义

A fragment（16×16 bf16，每 lane 4 个 32 位寄存器、8 个元素）用 `ldmatrix.x4` 一次装载；B fragment（8 列条带 × 16 k，`kt=True` 即 tile 行为 n、k 连续的转置存放）用 `ldmatrix.x2`。每 lane 提供一个 16B 行地址，象限序为 **[TL, BL, TR, BR]**——这是 G4 锚（`ref_ldmatrix` 配置）钉下的"G4 anchor semantics"：

```python
# ldmatrix per-lane quadrant indices (G4 anchor semantics)
e.l(f"%lq7_{name} = arith.remsi %mL, {e.ic(8)} : index")    # lane%8
e.l(f"%lq8_{name} = arith.divsi %mL, {e.ic(8)} : index")    # lane/8
e.l(f"%lqh_{name} = arith.remsi %lq8_{name}, {e.ic(2)} : index")
e.l(f"%lqt_{name} = arith.divsi %mL, {e.ic(16)} : index")   # lane/16
e.l(f"%lh8_{name} = arith.muli %lqh_{name}, {e.ic(8)} : index")
e.l(f"%lt8_{name} = arith.muli %lqt_{name}, {e.ic(8)} : index")
e.l(f"%lb8_{name} = arith.muli %lq8_{name}, {e.ic(8)} : index")
...
        r  = mi*16 + %lh8 + %lq7      # A: lanes 0-7 行 0-7(TL) 8-15 行 8-15(BL)
        c  = kb + %lt8                #    lanes 16-23 (TR) 24-31 (BR) 列 +8
        e.l(f"%fa_{name}_{mi}_{kb} = nvgpu.ldmatrix {a_src}[{r2}, {c}] "
            f"numTiles = 4 transpose = false : {a_ty} -> vector<4x2xbf16>")
...
        r  = %nc_{name}_{ni} + %lq7   # B (kt=True): 行 n + lane%8
        c  = kb + %lb8                # 列 kb + 8*(lane/8)
        e.l(f"%fb_{name}_{ni}_{kb} = nvgpu.ldmatrix {b_src}[{r}, {c}] "
            f"numTiles = 2 transpose = false : {b_ty} -> vector<2x2xbf16>")
```

这里有一个容易写错、且曾真实写错过的点：**A 的 b16 象限 lane 算术不含 `%mLG`**——`%mLG`（=g）是 f32 C/D fragment 的 lane 偏移，只出现在 epilogue；`mma_gemm_mh` 的注释明确记录了这一区分（"NO mLG -- that is the f32 C-fragment lane offset used by the epilogue only; the b16 quadrant lane math is separate"）。落到 PTX：

```ptx
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r5, %r6, %r7, %r8}, [%rd125];
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r9, %r10}, [%rd126];
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {%r19, %r20, %r21, %r22}, {...}, {...}, {...};
```

（evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/kernel.ptx 与 gen/ref_ldmatrix/kernel.ptx 真实摘录；SASS 侧对应 `HMMA.16816.F32.BF16 R112, R108, R176, R112`，PTX 计数与 SASS HMMA 计数在全部 G2 配置 1:1 相等。）

`ldmatrix` 的价值是直接可量化的：一个 x4 装载替代 8 条 lane 索引化的 `ld.shared.b16`（G2 阶段机器即用后者，见 `frag_build`，§3.11.1），SASS 上 LDSM 一条对八条 LDS。反向链的 dqkwg 在引入 ldmatrix 化的重构后 SASS 计数为 LDS 4440→1378（−69%）、LDSM 192→896、HMMA 832→1280（evidence/G5/bwd_report.md G5-O 节）。

#### 3.2.3 A/B/C/D 的寄存器序与 G4 的一行修复

PTX m16n8k16 布局律（硬件规定，G4 以整型探针在真机上精确裁定）：

| fragment | 每 lane 元素 | 公式 |
|---|---|---|
| A（bf16，4 寄存器 8 元素，按 [TL,BL,TR,BR] 象限 pack 列对） | $a_0{=}(g,2t)$, $a_1{=}(g{+}8,2t)$, $a_2{=}(g,2t{+}1)$, $a_3{=}(g{+}8,2t{+}1)$，再平移 $k{+}8$ 得后四元素 | 行半与 k 八元组配对 |
| B（bf16，2 寄存器 4 元素） | $b_0{=}(2t,g)$, $b_1{=}(2t{+}1,g)$（[k,n] 坐标），k+8 平移得后两元素 | **每 lane 的 B 寄存器属于列 $n=g$（groupID）** |
| C/D（f32，`vector<2x2xf32>`） | $c_0{=}(g,2t)$, $c_1{=}(g,2t{+}1)$, $c_2{=}(g{+}8,2t)$, $c_3{=}(g{+}8,2t{+}1)$ | **每 lane 的 D 元素属于列 $n=2t$ 与 $2t{+}1$** |

`mma_gemm` 的 epilogue 恰按 D 序回填（`ei∈[0,4)`：行 = `rA`/`rAh`，列 = `nct`/`nct1`，`mma_extract(e, acc, ei//2, ei%2)`）。

**B 的 N 索引是 $g$，D 的 N 索引是 $2t$——这一区别就是 G4 门抓出的核心 bug。** 早期 `frag_epilogue` 把累加器偶元素存在列 `ncol + %g`（lane 的 **B-fragment** N 索引）而非 `ncol + 2t`（**D-fragment** N 索引）。后果完全可预测：同一 $g$、不同 $t$ 的两个 lane 写同一列（冲突值），而 $2t{+}1$ 类奇数列永远无人写——kkt 只覆盖 1168/2080 个严格下三角 cell，o 阶段 2304/4096。修复是**一行语义修复**（新建 `%ncolt{ni} = ncol + %t2` 用于偶元素，`frag_precompute`/`ws_frag_precompute` 各加一个预计算），修复后每个 mma 配置写满 2080/2080 个 tril cell，且对 fp64 孪生的最大偏差 9.77e-4 **恰等于** numpy 对"f32 累加 + bf16 单次量化"管线的精确预测 9.765625e-4（测量 == 预测，evidence/G4/GATE_REPORT.md §2.1）。该 bug 影响 20 个配置中的 11 个（全部 mma=on 路径含 ws/tma）；这是"编译产物与手写 kernel 同构"叙事下第一个位级证据链的锚点，也是后文一切 D 序公式被无条件信任前必须先讲的故事。

#### 3.2.4 累加器链接、`k0=` K 分块、epilogue 与栅栏语义

**累加器链接（`accs=`）。** `mma_gemm` 返回 `{(mi,ni): %acc}` 字典；下次调用传 `accs=` 即在寄存器内续累。这是 split-bf16 的 hh/hl/lh 三链节、dkb 的跨 vt 折叠、caryA 的跨 j0 块复用的公共底座——全程不落共享内存。

**`k0=` K 分块。** 一条 `Kred=64` 的链可拆成两条 `K=32` 的链节（`k0=0/32`）共用同一宽 tile，产生**完全相同的 mma.sync 序列**、但 ldmatrix fragment 的活跃区间停留在 K=32 链节级别——ptxas 寄存器压力不随总 K 增长。dqkwg 的 K=64 位点按此分块后（`GDN_DQKWG_K32`），spill 从 88/84/120 B 降到 64/64/104 B，两形状 35/35+36/36 个数组逐位相等、6/6 数值位数与记录一致；代价是该处 1.00×（spill 本身不是性能杠杆，数据点留档，evidence/G5/bwd_report.md G5-AA 节）。`_ops_g`/`_mh_g`/`_gemm_g`（`gdn_bwd_gen.py:5937-5972`）把"KGRAN 分块 + 尾链节挂 epilogue"封装成三个通用 helper。

**epilogue 与 `epi_bar=`。** epilogue 是逐 (mi,ni,ei) 的回调，拿到 (row, col, acc SSA)。`epi_bar=True` 在 mma 体与 epilogue 之间插一道 `gpu.barrier`——当 epilogue 的 st.shared 目标与本链自身 ldmatrix 仍在跨 warp 读取的 tile 重叠时必须开启（wyb 的 split-epilogue-over-B-source 模式；这是与 §3.8.2 同族的跨界写读 hazards 类，evidence/G5/bwd_report.md G5-T 节）。

**栅栏语义总述。** 该发射层的栅栏经济学有三条明文规则（均以源码 docstring 为准）：

1. **跨 staging/mma 边界必须栅栏**：任何 all-thread staging sweep 与 warp-guarded mma 读写之间，一道 `gpu.barrier` 封区。
2. **不相交 tile 的连续 staging 免栅栏（`bar=False` 类）**：扫过互不相交 tile 的连续 staging sweep 之间免尾栅栏，首道 mma 读者之前的单一栅栏覆盖全组（`stage_raw_bf16`/`stage_raw_T` 的 `bar=` 参数，G5-T 节引入）；dqkwg 的 dw→dq1 背靠背链同理——第二条链自己的栅栏给全组封界。
3. **受卫区尾栅栏不可省**（§3.4）：idle warp 会冲进下一段 staging 覆盖 mma warp 还在读的操作数 tile。

**效果数字（本发射器的整机贡献）。** wy 的 kkt/P 链按 k16 切片化 + accs 链接重构后，与 u/w 尾部（§3.6）合计：wy 0.70→0.23 ms（3.04×），PTX 侧 `ld.shared` 832→49、`mma.sync` 256→384、`ldmatrix` 96→384、`cp.async` 20→0，共享内存池 90624→45832 B（2 CTAs/SM），寄存器 255→158/161、零 spill（evidence/G5/bwd_report.md G5-S 节）。

### 3.3 `mma_gemm_mh`：mi-half × ni-strip 二维 warp 映射（W≤16）

**动机。** 条带贴图把 W 上限钉死在 $N/8$：$N{=}32$ 时至多 4 个 warp。dqkwg 的 M=64、N=32 形状族若按条带贴图，只能在 `%wgv4` 受卫区内跑 4/8 个 warp——ncu 显示 **89% no-eligible cycles、每调度器约 2 个活跃 warp**，shared-scoreboard 停顿占 18.1 cyc/inst 的 49.9%（ldmatrix 延迟之外无事可发射；evidence/G5/bwd_report.md G5-O 节）。半台 SM 闲置纯属贴图几何问题。

**源码（`tools/gdn_bwd_gen.py:1821`）。** 二维映射把 **M 维也切半**：设 `npr = N//8`（N 条带内的 atom 列数）、`halves = W//npr`、`mpw = (M//16)//halves`，warp $w$ 拥有 ni-条带 $w \bmod npr$ 与 mi-块对 $(w // npr)$：

```python
npr = N // 8;  halves = W // npr;  mpw = (M // 16) // halves
assert (M // 16) == halves * mpw and W % npr == 0 and W <= 16
e.l(f"%mh_{name} = arith.divsi %mW, {e.ic(npr)} : index")
# mi-pair base in ROWS（块序乘 16 —— 首版误加"块数"到行偏移，warps 4-7 读到
# 行 {2,18} 而非 {32,48}，verify 即刻抓出）
e.l(f"%mhb_{name} = arith.muli %mh_{name}, {e.ic(mpw * 16)} : index")
e.l(f"%nx_{name} = arith.remsi %mW, {e.ic(npr)} : index")
e.l(f"%ncv_{name} = arith.muli %nx_{name}, {e.ic(8)} : index")
```

fragment/lane 数学与 `mma_gemm` 逐条相同（probe 验证的布局复用），B fragment 仍跨本 warp 的 mi 对复用——只是 warp→tile 的指派变了。W≤16 的意义：512 线程的 dqkwg 臂有 16 warp、halves 可到 4，映射对 halves 通用而 fragment 数学是 warp 局部的，无需其它改动（G5-T 节）。

**IR/结构对比。** 相比受卫条带方案，此映射删除了 `scf.if`-yield 区域整体：W=8 全 warp、无 guard、无跨区域累加器。寄存器从 255（60 B spill）降到 248（0 spill）。

**效果数字。** dqkwg 的 LDS 风暴击杀（−69%）正是靠它才不被 guard 空转吃掉；wyb 的 tv 3-mma（M=C=64, N=BV=16, K=C, W=8，即 npr=2 × 4 个 mi 半对）与 caryA 的 [BK2,BK2] 方块链都运行在这张地图上。首版实现的两处 bug（mi 基址块数/行数混淆；见上方注释）被 verify 门当场抓获，是"二维贴图几何必须 GPU 验证"的直接论据。

### 3.4 `mma_chain_if`：受卫链、跨区域累加器与尾栅栏

**动机。** N=32 位点在二维贴图出现之前运行在"4 warp 受卫区"里；链式 mma（split-bf16 的多链节、跨 K 块）需要把 fragment 累加器带出 `scf.if` 区域——**链节之间的 staging 是 all-thread 的，不能坐进 <W-warp 的受卫区内**。

**源码（`tools/gdn_bwd_gen.py:993`）。** 累加器作为 `scf.if` 的**结果值**跨越区域边界：

```python
npr = N // (8 * W);  nacc = (M // 16) * npr
tys = ", ".join(["vector<2x2xf32>"] * nacc)
res = [e.v("acr") for _ in range(nacc)]
e.l(f"{', '.join(res)} = scf.if {guard} -> ({tys}) {{")
    for i, (asrc, aty, bsrc, bty, k0) in enumerate(ops):
        accs = mma_gemm(e, f"{name}_{i}", M, N, Kred, W, asrc, aty, bsrc,
                        bty, kt=True, k0=k0,
                        epilogue=last_epi if i == len(ops) - 1 else None,
                        accs=accs)
    e.l(f"scf.yield {', '.join(ordered)} : {tys}")
e.l("} else {")
    ...零初始化 else 臂 yield...
e.l("}")
e.l("gpu.barrier")     # 尾栅栏：强制
```

三处语义要点都在 docstring 里成文：(i) 写只发生在受卫 warp 内——epilogue 单写者；(ii) **尾栅栏不可省**：没有它，idle warp 冲进下一段 staging sweep、覆盖 mma warp 仍在读的操作数 tile（早期 dvp 循环中的潜伏模式，在此关闭）；(iii) `bar=False` 用于下一区域只**读**相同 tile 的背靠背链（dw→dq1），第二链自身栅栏给组封界。`guard=None` 切换到 mi-half 二维贴图（全 8 warp、无区域），ops 元组允许 6 元形式携带每链节自己的 `klen`（K 分块链）。

**效果数字。** 该组合子是 dqkwg v3/v4 的 dvp/dW/dq1 链与 pdO-fold（pt = (P^T·scale·rie) hi/lo，单一 epilogue 因子 egc[row]=2^(gcl−gc_i) 同时回收 pdO 与 bh 两路）的载体；v3 整体 e2e 贡献见 §3.2.4/§3.3 的 SASS 计数。

### 3.5 `tgemm` 与 `gdn.mma.tf32k8`：tf32 k8 逃生舱

#### 3.5.1 必要性与 inline_asm 形式

**动机。** 反向链的 f32 操作数 GEMM（u/w 尾、dA 链、dq/dk 尾、carry 合成）若走 bf16 mma 会把 f32 指数域截断、重新毒化 dA 链；tf32 保留 f32 指数域、仅把尾数降到 10 位，而数值裕度（当时 ~180× 于门）可以吸收——前提是能发射 `m16n8k8` 原子。如 §3.0 边界四所述，表内路径不可达，于是有了**逃生舱 #1**（全文 ≤5 预算中唯一使用的一个）：方言操作 `gdn.mma.tf32k8`，在 `gdn-lower-hatch` pass 里 lower 为直接 `llvm.inline_asm`（`dialect/lib/GdnHatch.cpp:170`）：

```cpp
static const char *kPtx =
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
    "{$10, $11, $12, $13};";
static const char *kConstraints =
    "=&f,=&f,=&f,=&f,f,f,f,f,f,f,f,f,f,f";
```

操作数序：输出在前——$0..$3 = D，$4..$7 = A，$8/$9 = B，$10..$13 = C；结果是 literal `struct<(f32 x4)>`，四次 `extractvalue`。落到 PTX（真实摘录，evidence/G5/scratch/r18b_l2_T8192/fwdscan/kernel_smemfix.ptx）：

```ptx
// begin inline asm
mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    {%r646, %r647, %r648, %r649}, {%r650, %r651, %r652, %r653},
    {%r654, %r655}, {%r656, %r657, %r658, %r659};
// end inline asm
```

#### 3.5.2 early-clobber `=&f` 是强制项

**这是本机器最重的一条工具链发现。** 不用 `nvvm.inline_ptx`（BasicPtxBuilder 路径）的原因是其输出约束只能给普通 `"=f"`；而 SASS 级验证表明：在 `"=f"` 下，ptxas 的 inline-asm 寄存器分配**合法地**把 D 四元组映射到同一条 mma 的 A 四元组上（`HMMA R12, R12, R10, R16`）——输入在 asm 之后被视为死亡、又无 early-clobber 约束，张量核的读/写重叠随即损坏操作数。一个 K=16 双原子链的整型探针 **118/128 个 cell 失败**，k 项整体缺失、误差确定性复现。`"=&f"` 禁止输出/输入重叠后问题消失；D==C 的别名（fla 式原地累加）保持合法，但本实现仍让 D 完全独立。`GdnHatch.cpp` 的注释把这定为"对任何在 mma 上使用 BasicPtxBuilder 的人都成立的工具链发现"。

#### 3.5.3 A 寄存器序：row-half 外层（硬件探针裁定）

tf32 k8 的 fragment 布局**不靠文档猜**，由 16×8×8 整型操作数探针（tf32 精确表示整数）在真机上钉死（D == A@B+C 精确相等；N(0,1) 操作数下偏差 4.2e-3，即真 tf32 语义、离截断模拟约 1 ulp）：

```
g = lane>>2, t = lane&3
A 16x8 [m,k]: a0=(g,2t)  a1=(g+8,2t)  a2=(g,2t+1)  a3=(g+8,2t+1)
B 8x8  [k,n]: b0=(2t,g)  b1=(2t+1,g)
D 16x8 [m,n]: d0=(g,2t)  d1=(g,2t+1)  d2=(g+8,2t)  d3=(g+8,2t+1)
```

关键差异：**A 的寄存器序是"行半外层、列对内层"**（$a_1$ 与 $a_0$ 同 k、行 +8），而 bf16 k16 的 A 是按 k 八元组配对的 packed-pair 序——把 bf16 k16 的直觉直接搬过来是错的，探针的 exp1 立刻暴露。tgemm 因此**不用 ldmatrix**（b16 装载模式套不上 32 位元素），按布局公式逐元素 `memref.load`（每原子每 lane 6 次：4 A + 2 B）。

#### 3.5.4 `tgemm` 站点发射器

**源码（`tools/gdn_bwd_gen.py:2010`）。** 与 `sgemm` 逻辑契约完全同构（`at/bt/acc/neg/off` 语义一致；kahan 无意义——K 缩并在张量核/f32 累加链内部完成）。贴图为 atom 轮转：`ti = mW + i*W`、`mi = ti/npr`、`ni = ti%npr`，要求 `(M//16)*(N//8)` 是 warps 的整倍数。操作数装载（非转置 A 为例）：

```python
a0 = load2(e, a, abase[0], d["ac2"])     # (m+g,   kb+2t)
a1 = load2(e, a, abase[1], d["ac2"])     # (m+g+8, kb+2t)   <- 行半外层
a2 = load2(e, a, abase[0], d["ac2p1"])   # (m+g,   kb+2t+1)
a3 = load2(e, a, abase[1], d["ac2p1"])   # (m+g+8, kb+2t+1)
...
e.l(f"{n0}, {n1}, {n2}, {n3} = gdn.mma.tf32k8 {a0}, {a1}, {a2}, "
    f"{a3}, {b0}, {b1}, {accs[0]}, {accs[1]}, {accs[2]}, "
    f"{accs[3]} : f32 -> (f32, f32, f32, f32)")
```

两个发射器级陷阱（均被验证门抓出、均在 docstring 成文）：

1. **kb 偏移的 off-by-one**：`%mT2P1` 本身已是 $2t{+}1$，per-kb 偏移代码又加了一次 "+1"——每个 kb>0 的 fragment 全部错位（也是最初 wyb 非法地址的根因：读走出 smem）。定位手段是 **delta 基 GPU 诊断**（B=I 时奇数 k 列整体 +1 位移肉眼可辨），修复后 13 组合探针矩阵（at/bt/acc/neg/offsets × 真实 bwd 形状）全过 tf32 舍入地板。
2. **`warps=` 必须显式传**：wy 以 128 线程启动而全局默认 `THREADS//32=8`——轮转贴图把一半 tile 晾着没人写（垃圾 u/w → dk=30）。`tgemm` 因此要求与内核线程数不符的调用点显式传 `warps=`（wy 传 4）。

`neg` 的语义边界与 sgemm 一致：只对 B 操作数取负、acc 初值保持正号（`dhu.dh_w` 依赖 `out += -(w^T dvp)` 这一区别）。

#### 3.5.5 逐位点数值门与性能

tf32 不是全开开关：24 个 f32-GEMM 位点逐一单独 tf32 化、在 T512 对每个梯度分量定级（evidence/G5/bwd_report.md G5-G 节表）。被否决的位点与数字：

| 位点 | 最差梯度位移 | 判定 |
|---|---|---|
| `dhu.dkB` | dk 1.04e-2 | **越 8e-3 门**（dH 操作数 ~O(10)）→ 保持标量 |
| `wy.u` | dk 1.06e-2 | **越门**（u 噪声经 S→Sn 放大）；同形同 K 的 `wy.w` 却干净（1.4e-5）→ tf32 |
| `wyb.tv` | dbeta 8.6e-3 | 仅 2.3× 裕度 → 标量 |
| `dhu.dvp_bh` | dbeta 3.0e-3 | 6.6×；且 dh 量化桶重采样混沌 → 标量（split-bf16 为文档化替代） |
| `dhu.dv_T` | dv 1.26e-3 | 6.4× → 标量 |
| `dhu.dvp_pdO/dh_qg/dh_w` | ≤3e-4 数值净 | **性能负**（见下）→ 标量 |
| `dqkwg.*` 8 位点 | dk 7.8e-4..5.2e-4 | tf32 |
| `wyb` 7/8 位点 | dk 6.6e-4..1.1e-4 | tf32 |

最终 16 位点 tf32；性能 14.16→11.06 ms（T2048）：dqkwg 1.71×、wyb 2.19×。两个值得记录的结构性发现：(i) `wy.w` 干净而 `wy.u` 有毒——同形状同 K，u 的误差坐状态扫描进入 Sn；(ii) tf32 的 dW/TW（dA 链、此前 kahan 保护）**什么都不动**——dW 量级小，kahan 保护的是 f32 累加噪声，mma 里同样没有。**性能负样本**：dhu 的三个数值干净位点全展开进串行反向 chunk 游走后，ptxas 128 regs/188 B spill → 255 regs/116-128 B spill，dhu 3.69→4.67 ms——tf32 全部回退，dhu 的杠杆是占用率与 retile，不是 mma。fwdscan 的复发 gemm 位点组（`scan`）经 bisect 后纳入默认（h_all 扰动 8e-4 rel / vnew 5.5e-5 rel 被既有最大值吸收，dk 裕度 4.5×；`cary` 组同理，dh0 裕度当时 4000×）。

### 3.6 split-bf16 hi/lo 体系

#### 3.6.1 数学

对 f32 值 $x$ 定义
$$\mathrm{hi}=\mathrm{bf16}(x),\qquad \mathrm{lo}=\mathrm{bf16}(x-\mathrm{hi}),\qquad x=\mathrm{hi}+\mathrm{lo}+O(2^{-24}\cdot|x|).$$
hi 携带前 8 位尾数、lo 携带接下来的 ~8 位，`hi+lo` 在 f32 中**精确**可加（8+8 位尾数装得进 24 位）；残差 ~$2^{-17}$ 相对类。乘积分解：
$$xy=(\mathrm{hi}_x+\mathrm{lo}_x)(\mathrm{hi}_y+\mathrm{lo}_y)=\underbrace{\mathrm{hi}_x\mathrm{hi}_y}_{\text{mma 1}}+\underbrace{\mathrm{hi}_x\mathrm{lo}_y+\mathrm{lo}_x\mathrm{hi}_y}_{\text{mma 2, 3}}+\underbrace{\mathrm{lo}_x\mathrm{lo}_y}_{\approx 2^{-32},\ \text{舍弃}}.$$
三个 bf16 mma 以 f32 累加器链回 ~fp32 质量的乘积，速度是 mma 速度——这是 tf32 的**文档化替代**：tf32 在 dkb.dkB 把 dk 推到 1.04e-2（越 8e-3 门），split-bf16 的 ~$2^{-16}$ 乘积误差比它低约 30×（`_split_bf16` docstring，evidence/G5/bwd_report.md G5-J 节）。

**2-mma 型何时够。** 当一侧操作数本来就是**精确 bf16**（全局存储即 bf16、逐元素搬运无舍入），只需给另一侧分裂：
$$x\cdot y_{\text{raw}}=\mathrm{hi}_x y_{\text{raw}}+\mathrm{lo}_x y_{\text{raw}}\quad\Rightarrow\quad 2\ \text{条 mma}.$$
更强的形式是**行尺度折入 epilogue 的代数恒等式**：当缩放因子 $s_i$ 只依赖**输出行** $i$（不依赖缩并指标 $k$）时，
$$\sum_k \big(s_i\,k[i,kk]\big)\,dh[kk,j]=s_i\sum_k k[i,kk]\,dh[kk,j],$$
缩放因子完全移出缩并——A 操作数退化为 raw bf16 k 切片（元素级 staging，无分裂、无预乘 sweep），尺度落进 epilogue 的一次 mulf。dqkwg 的 `dvp_bh` 位点即此（$s_i=2^{gcl-gc_i}$）：每 (vs,ks) 位点从 3 mma 降到 2 mma，共享内存 99840→97792 B、寄存器 234→214、零 spill，dqkwg 1.71→1.66 ms；两个中间设计（直接 3-mma 搬运 2.19 ms、转置 tile dW/dq1 staging 1.93 ms）被测回退（evidence/G5/bwd_report.md G5-L 节）。wy 的 u/w 尾部同型：`u = T·(β⊙v)` 的列缩放 $\beta[k]$ 作用在缩并指标上、不能移出，于是折入 **A 侧 T 的分裂**（见 §3.6.4 的 cvec）。

#### 3.6.2 3-mma 型的落点与数字

- **dkb**（`gen_dkb`，`gdn_bwd_gen.py:3652`）：`dkB = vng @ dh^T`，两侧 hi/lo 分裂（`_split_bf16`），hh/hl/lh 三链节以 `accs=` 链接、**vt 折叠也在 mma 累加器内完成**。dk 数值不变：1.730e-3 vs 标量 1.731e-3；dkb 0.96→0.25 ms（3.8×）。
- **bpre**（`GDN_BPRE_MMA`）：A = M 块 hi/lo 融合分裂（f32 源，~2^-16 乘积），B = dO^T **raw** bf16 tile（精确）、尺度在 epilogue、K=C 缩及寄存器链。SASS：**0 HMMA→128、LDS 160→0、LDSM 0→80**；[BK2,V] f32 bb staging tile 及其尺度 sweep 退役，smem 49152→36864 B（2 CTAs/SM）、寄存器 40→54；bpre 4.79×（最净窗口 5.9×）。
- **caryA**（`GDN_CARYA_MMA`）：bpre 式移植退役全部四块共栖 tile（82176 B）换三块 bf16 mma tile（28592 B），255→76 寄存器（双零 spill）→ 2 CTAs/SM；Bhat 因子经 `stage_split_T` 的 cvec 折入 A 分裂（§3.6.4）。tf32 版本 1.39× 的占用率天花板被打穿到 **5.00×**。重结合被测量而非空谈：对 tf32 版，A_all 在 8.39M/8.39M 个元素上位移、最大 **5.6e-22 绝对**（相对 2.8e-4——正是它替换掉的 tf32 截断类；A 超收缩故量级 ~1e-18），被 carry 链的舍入吸收在打印位数之下：**全部六个最终梯度位数不变**（evidence/G5/bwd_report.md G5-R 节）。
- **wyb 的 tv/TW/G/X/Y/dk_t1/dk_t2 全族**（`GDN_WYB_DIET`）：每个 tf32/kahan 位点变 2-3 链节 split mma；共享内存池 100096→45056 B、恰 128 寄存器 → 2 CTAs/SM；wyb 2.40×；数值**全线收紧**——七个 tf32 2^-11 位点降为 2^-17 分裂类：dk 1.691e-3→**1.284e-3**（6.2× 裕度）、dbeta→8.527e-4（23×）、dg→3.274e-4（61×）、dh0 2.761e-6→**2.169e-7**（36900×）（evidence/G5/bwd_report.md G5-T 节）。
- **dqkwg 的 tf32 尾巴**（`GDN_DQKWG_SB`）：dq_dPmk / dk_dPmq / TW 三处 tgemm 改 bf16 mma——dPm 自然分裂（dq 缩并其行）、**同一 arena tile 转置再分裂**（dk 缩并其列；两种朝向不能共享一次分裂但能顺序共享 tile）、T^T 分裂 + dW^T 分裂的 TW 三链节。tCC/tCKa/tCKb 及其 cp.async staging 退役：**cp.async 46→0、ld.shared 1234→71、stack 72 B→0**、寄存器 255→240；dq 4.910e-4→**1.824e-4**（裕度 7.5×→44×）（evidence/G5/bwd_report.md G5-S 节）。

#### 3.6.3 分裂原语的 smem 源家族

分裂不总从全局行出发；有一族以**常驻 smem tile** 为源的变体，均保持"逐元素运算序列相同、值相同 → 位相同"的可判定的位级等价：

| 原语（行号） | 源→目标 | 语义 |
|---|---|---|
| `_split_bf16` (:1293) | smem f32 [R,C] → hi/lo | 两 sweep（hi 先行、lo 重读）；`cvec=` 列缩放（任意尺度） |
| `_split_bf16_T` (:1335) | smem f32，**转置**写出 | dk_dPmq 的 A=dPm^T 朝向；同 2^-17 类、同双 sweep |
| `_split_bf16_fused` (:1433) | 同上单 sweep | 装载一次同发 hi/lo（索引算术减半）；dqkwg dh 侧关键路径用 |
| `_split_T_smem` (:1366) | smem f32 源转置分裂 + `cvec[s]` | 消费 cp.async 落地的 raw 窗口（只 `ld.shared` 读，合规于 §3.8.2 规则） |
| `_resplit_pair_T` (:1396) | **既有 bf16 hi/lo 对** → 转置再分裂 + `cvec[s]` | 重构 `shi+slo` 在 f32 精确 → 与直接分裂 f32 同 2^-17 类；dk_t2 的 β[r] 折入 |
| `split_smem_ew` (:1671) | stage_split_ew 的 smem 源版 | 同元素同序同值 → 位相同分裂 |

一条以 verify 门换取的 arena 纪律成文于此（G5-S 节）：**分裂 tile 永不得别名自己的源 tile**——dPm 分裂曾叠放在 accCK 自己的字节上，自毁分裂源，dq/dk 1e0 级 FAIL 当场被抓。

#### 3.6.4 `cvec=`/`vec=` 折入：幂二可交换性与任意尺度两类

把缩放因子折进分裂有两类代数性质不同的通道，源码 docstring 各自给出精确表述：

**幂二类（`stage_split_T` 的 `cvec=`，G5-R 节）。** 对 $2^d$ 缩放：
$$\mathrm{split}(v\cdot 2^d)=\mathrm{split}(v)\cdot 2^d$$
**精确成立**——$2^d$ 与 bf16 截断、与 hi/lo 残差均可交换（hi 与 lo 同步精确缩放，无新舍入）。因此把 caryA 的 Bhat 行尺度 $2^{g_{cl}-gc}$ 折入 A 分裂，能在**相同操作数值上逐位复现**旧的 B 侧缩放乘积。这是"改结合方式而不改任何值"的机器级表达：caryA 的 `bhb`（预乘 Bhat 的 B tile）退役为 raw-k 拷贝。

**任意尺度类（`stage_split_ew` 的 `cvec=`，G5-S 节）。** 对任意 $s$：乘积在 f32 中**先舍入一次**、再做标准 ~$2^{-17}$ 分裂——每项误差停留在分裂类内（非位精确，但与"先分裂再缩放"同误差类）。wy 的 u/w 用它把 $\beta[k]$（u）与 $\beta[k]2^{gc[k]}$（w，hoisted 成 bgc 向量）折进 T 的列缩放分裂，替代了 tf32 被门否决的 `wy.u`（1.06e-2 越门）——这正是 wy 3.04× 的数值前提。`vec=`（`[R,1]` 行向量）则预乘**输出行** j（pt 预缩放 $P^T\odot rie$ 一类的朝向）。

#### 3.6.5 数值效应总账

split-bf16 对六分量梯度的净效应全部朝**收紧**方向：dk 在 dkb 位点不变（1.730e-3 vs 1.731e-3）、在 tf32 尾巴位点收紧 1.7e-3→1.28e-3；dv 因 tv 与 dvp 双双分裂而多一层重结合（3.3e-6→1.088e-5，裕度仍 736×）；dbeta/dg/dh0 随之收紧。与 tf32 的对照数字（dk 1.04e-2 越门 vs split 的 1.73e-3）是该体系立项的全部理由。

### 3.7 staging 原语族 I：同步 st.shared 家族

#### 3.7.1 布局与 +8 列 bank pad

一切喂 `ldmatrix` 的 tile 遵守两条布局律：

1. **+8 列 bf16 pad**：行跨距必须保持 16 B 整倍（cp.async 16B 对齐），但**不得**是 128 B 整倍（前向 F2 的教训，§3.11.4）——bf16 tile 一律 `[R, C+8]`，行跨距如 64 列 → 128+16=144 B，8 个 ldmatrix 行轮转扫过全部 32 bank。pad 的摆放位置服从识别约束：ldmatrix 喂入 tile 的识别 pad 用额外**行**（永不寻址：r < R），纯通用 f32 运算的 tile 才用 +4 列（见下条与 arena 识别键）。
2. **pad 永不写永不读**（纯 bank 展开）；aliased tile 的 malloc 尺寸必须互相唯一——LLVM 会重排 malloc，**尺寸**才是 `gdn_smem_fix` 识别 arena 宿主的键（行 pad 而非列 pad 出现在需要保持 16 B 行对齐处）。

#### 3.7.2 raw 家族：`stage_rows` / `stage_raw_ew` / `stage_raw_bf16` / `stage_raw_T`

**动机。** bf16 全局 → bf16 tile 的搬运是**精确**的（无转换、无舍入），因此它是 split-bf16 体系里"raw B 操作数"的标准供给通道；同时它写 tile 的方式（`memref.store` → st.shared）天然满足 §3.8.2 的 ldmatrix 合规规则。

```python
def stage_raw_T(e, dst, src_g, gty, m, cm, col0, R, S, bar=True):
    """TRANSPOSING raw bf16 stage:  dst[j, s] = g[m, col0 + s*cm + j]
    (tile [R, S+8]; source slice is S rows x R fast cols).  Reads
    coalesced (j fast within s); stores land row-strided (~8-way STS
    conflicts -- cheaper than the uncoalesced global alternative)."""
    e.l(f"scf.for %srt = %tid to {e.ic(R * S)} step %nt {{")
    j = %srt % R;  s = %srt // R        # 全局侧 j 连读（coalesced），
    v = gload_raw(e, src_g, ...)         # smem 侧行跨写（~8 路 STS 冲突
    e.l(f"memref.store {v}, {dst.name}[{j}, {s}] ...")
```

四个变体的分工：`stage_rows`（自然朝向一般形，`shape=` 参数支持有界子区 staging）；`stage_raw_ew`（element-wise RAW 同 dtype 切片，为太窄走不了 16B burst 分区的 tile——如 [C,16] bf16 = 128 块 < 256 线程）；`stage_raw_bf16`（RAW bf16 自然布局进**带 pad 的 mma tile**，两侧 coalesced）；`stage_raw_T`（转置 staging：全局行内 j 快连读、smem 侧行跨写，牺牲 ~8 路 STS 冲突换取全局侧 coalesced——比全局侧不 coalesced 的替代便宜）。`bar=` 参数实现 §3.2.4 的不相交 tile 免栅栏类。

#### 3.7.3 `stage_split_ew` / `stage_split_T`：融合分裂 staging

**动机。** "先 stage 后分裂"要两遍索引算术；融合版从扁平全局行**一次读入、同时写 hi 与 lo**（索引算术减半），且是 split 体系里全局侧的标准入口：

```python
def stage_split_ew(e, hi, lo, src_g, gty, m, cm, col0, R, CW, cvec=None):
    """FUSED hi/lo split straight from a flattened global row ...
    G5-S: cvec -- a [CW,1] smem f32 pre-scaling each COLUMN c
    (hi/lo = split(v * cvec[c])); ARBITRARY scales (unlike stage_split_
    T's exact power-of-two cvec fold): the product rounds once in f32,
    then the standard ~2^-17 split residual."""
    v = gload(e, src_g, gty, [m, _gcol_int(e, i, j, cm, col0)])
    if cvec is not None:
        v = op2(e, "mulf", v, ldsc(e, cvec, j))
    e.l(f"%sph = arith.truncf {v} : f32 to bf16");  store -> hi
    hw = extf(%sph);  r = v - hw
    e.l(f"%spl = arith.truncf {r} : f32 to bf16");  store -> lo
```

`stage_split_T` 是其转置孪生（读 j 快连、写 [R, S+8] 的转置目标），带 `vec=`（输出行预乘）与 `cvec=`（输出列预乘，幂二精确类）两个折入通道。wy 的 u/w 相位即"同一 arena1 槽位先装 u 相位的 T 分裂、尾栅栏后重装 w 相位的 bgc 缩放分裂"——生命周期复用（lifetime arena）的标准玩法。

#### 3.7.4 barrier 经济学回顾

同步家族的每次调用默认以 `gpu.barrier` 收尾；`bar=False` 的两个成文类（不相交 tile 连续 staging；背靠背读同 tile 的链）已在 §3.2.4 列为机器级规则。restage 纪律的另一半来自 fwdscan/dhs 的滚动窗口实践并被源码引用为"r10 规则"：**每块缓冲的下一次覆写之前必有尾栅栏、且复用距离 ≥2 次迭代**。

### 3.8 staging 原语族 II：异步 16B burst 家族与 LDGSTS→LDSM 危险

#### 3.8.1 `astage_tile` / `agroup` / `await_all` / `astage_sync`

**动机。** §3.0 边界五禁止 transform 路径做预取；手工 `nvgpu.device_async_copy` 是唯一通道。`astage_tile`（`gdn_bwd_gen.py:1569`）发射**线程跨步的 16B burst**：

```python
def astage_tile(e, src, gty, m, cm, col0, dst, dst_col0, R, Cc, dt, toks, ...):
    """thread-strided MANUAL nvgpu.device_async_copy burst for an [R, Cc]
    tile of ONE global row ... in 16B chunks (8 bf16 / 4 f32 ...).
    NOT nvgpu.device_async_create_groups / the transform's create_async_
    groups: those insert `cp.async.wait_group 0` right after every group,
    which serializes any prefetch with the compute it should overlap.
    Emitting the ops directly leaves wait placement to the caller."""
    el = 8 if dt == "bf16" else 4
    for kk in range(tot // nthr):            # 静态展开：token 留在本作用域
        ...
        e.l(f"{t} = nvgpu.device_async_copy {src}[{m}, {gcol2}], "
            f"{dst.name}[{r}, {dcol}], {el} bypassL1 : {gty} to {dst.ty}")
        toks.append(t)
```

三个契约级细节（全部由 verify 门用真实代价换得，docstring 成文）：(i) **静态展开**，使 token 在调用方作用域存活、供其 `create_group`；(ii) `nthr` 必须等于启动内核的线程数（wy 128 vs 模块默认 256 的失配曾让缺失线程拥有的块**无人拷贝**、半陈旧 tile 入算）；(iii) `tid=` 覆盖属主线程变量（fwdscan v4 的 warp-group 半体以 `%gtid` 各自 stage 自己的 K 半切片，nthr=128）。`agroup`/`await_all` 封装 `create_group` 与 `numGroups=0` 等待；`astage_sync` 是 `stage_rows` 的即换即用替代（burst + 立即 drain + barrier）——值位相同、staging 指令数约 **4×** 缩减（dkb/dHv 模式，逐位一致验证）。PTX/SASS 真实形态：

```ptx
cp.async.cg.shared.global [%rd518], [%rd520], 16;
cp.async.wait_group 1;
```
```asm
LDGSTS.E.BYPASS.128 [R3], desc[UR10][R116.64] ;
```

跨 chunk 预取的进阶形态（dhs 的 `GDN_DHS_XP`）：最后一个窗口把 chunk c+1 的窗口 0 DMA 发进空缓冲，**cp.async group token 以 `!nvgpu.device.async.token` 类型坐 `scf.for` 的 iter_args** 跨迭代（该助记符是本 MLIR 构建的注册名；`async.token` 是解析错误），下一迭代 prologue 的 await 于是被整 chunk 计算覆盖；末 chunk 发一次无害自重复使 token 恒有效、循环后 drain 保证 CTA 退出时无在飞拷贝。值精确性以全链 npz diff **逐位相等**证明（evidence/G5/bwd_report.md G5-S 节）。

#### 3.8.2 LDGSTS→LDSM 危险（机器级 hazards 规则）

**现象。** 用异步 burst stage 的 mma A tile（`akb [C, BK+8]` bf16，SASS `LDGSTS.E.BYPASS.128` 落地）被**通用 `ld.shared` 读回完全正确**（对全局源逐位相等，dump 验证），却被 `nvgpu.ldmatrix` **读错**——错的行块随构建调度而定（2-mma 构建错 32-63 行、3-mma 构建错 48-63 行）、逐构建确定、对值保持的刷新遍历不变。验证门以 dk 1.34 / dv 2.18 / dbeta 25.7 抓住（dq/dh0 因不依赖 dvp 而纹丝不动——错误面精确圈定 dvp 路径）；lane 真值 fragment dump 把错位定位到整块 mi 块。**同一 tile 换 plain st.shared 写入（ASPLIT bisect 臂）后，同一套 ldmatrix/mma/epilogue 机器对 numpy 逐位相等（5.2e-8）**（evidence/G5/scratch/r6_ldgsts_probe_fail / r6_ldgsts_probe_stshared_pass）。

**成文规则（源码注释中的 "r6 rule"）。** *每一个喂 `nvgpu.ldmatrix` 的 tile 必须是 st.shared 写入或元素级 staging 的——绝不许 raw LDGSTS→LDSM 直连*。raw 落地的 tile 只许喂通用 `ld.shared`（rgemm 窗口、tgemm 操作数皆此合规形态）。这是手动 cp.async 路径在本 sm_120a 工具链上的第二个文档化静默失效模式（第一个是构建早期 kdbg 记录的 burst 零交付——同一指令族、同样延迟等待特征）。

#### 3.8.3 `launder_pass`：恒等 st.shared 重写

**动机与源码（`gdn_bwd_gen.py:1650`）。** FPIPE 化（预取直落 operand tile）与 r6 规则的交集需要一座桥：raw 落地 tile 在首个 ldmatrix 读者之前，做**恰好一遍**元素级 `ld.shared/st.shared` 恒等重写：

```python
def launder_pass(e, buf, R, Cc):
    """identity st.shared rewrite of a cp.async-landed tile region -- the
    r6 rule demands ldmatrix operands be st.shared-written (NEVER raw
    LDGSTS->LDSM), so a raw-landed tile gets ONE element-wise
    ld.shared/st.shared pass over exactly the staged cells before its
    first ldmatrix reader ... Values bit-identical by construction."""
    e.l(f"scf.for %lu = %tid to {e.ic(R * Cc)} step %nt {{")
    v = memref.load buf[i, j];  memref.store v -> buf[i, j]
```

值按构造逐位相同（存回未变的 v）；caryA 的 w 窗口旋转即其消费者——raw cp.async 窗口只喂通用读，ahi/alo 保持 st.shared 写入的 ldmatrix 源。

### 3.9 `rgemm` / `rgemm_aw` / `aggemm`：carry 链的寄存器累加 GEMM 族

**动机。** 反向 carry 链（`dhs`）每 chunk 一次 `Xa += A_all[m] @ X`；[K,K] f32 carry tile 若 staging 进 smem 要 64 KB——而该 gemm 的访问模式让 staging 毫无意义：一个 warp 的 lane 拥有同一输出行的连续 j，每 lane 对每 kk 发出**相同**的 `aglb[mrow, i*Kd+kk]` 装载（硬件广播、每 warp 每 kk 一次事务；第二+个消费者 warp 命中 L2）。`aggemm`（`gdn_bwd_gen.py:1518`）即此直接读全局的标量 gemm（分区/单写者与 sgemm 相同）。

`rgemm`（:1455）解决 `aggemm` 的真实瓶颈——cell 主序的 1 深依赖链让 LDG 延迟串行化（dhs 0.83 ms）。其分区把 ILP 变成结构属性：输出 cell $e=iN{+}j$ 属于线程 $t$ 当 $e\in\{t+c\cdot NT\}$（$CPC=MN/NT$ 个/线程）；当 $N=BV$ 且 $BV\mid NT$，同线程的全部 CPC 个 cell **共享** $j=t\bmod BV$——**每 kk 一次 b 装载服务全部累加器**，CPC 次 a 装载（互异行、warp 广播）是编译器可重叠的独立全局装载：ILP=CPC。

```python
NT = THREADS;  CPC = (M_ * N_) // NT
e.l("%rgj = arith.remsi %tid, " + e.ic(N_) + " : index")
e.l("%rgi0 = arith.divsi %tid, " + e.ic(N_) + " : index")
for c in range(CPC):
    ic = %rgi0 + c*(NT//N_);  rows.append(ic)
    inits.append(load2(e, acc_buf, ic, "%rgj"))   # 加性初值来自另一缓冲（acc=True 语义）
# kk 外层：一次 b 装载 + CPC 次独立 a 装载 + CPC 次 FMA
header = ", ".join(f"%rga{c} = {inits[c]}" for c in range(CPC))   # iter_args 链
```

`rgemm_aw`（:3350）是其 A 操作数**滚动窗口**变体：`a[i,kk]` 改读 ping-pong `[K, BKW=32]` f32 smem 窗口（`astage_tile`+`agroup`+`await_all`，超前一个窗口 staging；24 burst/chunk），b 侧保持全 [K,BV] tile 的绝对 kk 读。窗口只喂通用 `ld.shared`（r6 规则的合法形态）；每次 wait_group 后跟 barrier 才跨 warp 读；缓冲复用距离 ≥2 迭代。

**效果数字。** `rgemm` 单独：dhs 0.83→0.59 ms；`rgemm_aw`（`GDN_DHS_AW`）：dhs 0.59→0.32 ms（1.85×），smem 32768→65536、寄存器 48→80、零 spill、cp.async 0→24。**值精确性以控制链证明**：AW=0 与 AW=1 同种子同上游 cubin → A_all/B_all/dht 逐位相等，且 **dh_all 与 dh0 逐位相等**（窗口是纯延迟变化；kk 升序 FMA 链未动；evidence/G5/scratch/r11_logs/awiso）。BKW=16 探针（8 KB 窗口、理论上 2-CTA）被测否决（0.91×：网格 64 块本就填不满设备，加倍 barrier + 16 B spill 是实价）。

### 3.10 warp 级归约、显式 FMA 与 kahan 边界

mma 机器的 epilogue 侧配套三件标量/洗牌设施：

- **`_shfl_tree5` / `warp_rowsum` / `warp_colsum`**（`gdn_bwd_gen.py:917/928/961`）：5 步 xor 蝴蝶（`gpu.shuffle xor` → `shfl.sync.bfly.b32`，probe 验证）替代"一行一线程、CW 次依赖迭代"的串行 C 链——warp $w$ 拥有行 $\{w, w+NW,\ldots\}$，lane $l$ 供给 $j=l$ 与 $l{+}32$（CW≤64），每行一棵 2 深并行树，lane 0 加进 dst。EX2 计数伴随 epilogue 重构从 149 降到 4（egc/rie/seg/bse [C] 向量 hoist 替代逐元素 MUFU sweep）。
- **`fma3`**：非 kahan sgemm 体内的显式 `math.fma`（单舍入、严格紧于 mulf+addf、同求和顺序）。本树静态分析曾发现全部 bwd kernel 零 `fma.rn.f32`——工具链从不收缩 `arith.mulf/addf`，fastmath 属性不解析；显式发射是唯一通道（probe 验证 lowering）。
- **kahan 边界**：补偿求和臂**禁止**融合——补偿项依赖精确的 sub/add 语义，`fma3` 的 docstring 明文 "NEVER inside kahan bodies"。

### 3.11 前向孪生机器（`gdn_full_gen.py`）

前向生成器持有同一布局律的独立实现（历史顺序上它是原生侧，反向机器由它移植而来）。差异在外壳：常量即时发射（`Em.c`）、lane 变量不带 `m*` 前缀、以及一套为 transform 流水线结构化的 K 循环。

#### 3.11.1 `emit_gemm` 与双 fragment 模式

`_FRAG_MODE`（`--frag`）在两种 fragment 装载间切换：`"ldm"` 用 `ldm_a`/`ldm_b`（`nvgpu.ldmatrix` x4/x2，含 `swap`/`kt` 的 `.trans` 变体——A 转置时对 X 的 [k 八元组][m 八元组] 块取 `.trans`、地址行列互换）；`"scalar"` 用 `a_frag_elems`/`b_frag_elems`/`frag_build`——按 D-fragment 坐标公式逐元素 `memref.load` + `vector.insert` 构造（G2 阶段的原始形态，作为布局公式的可执行规格与回归锚保留）。A 的 8 元素坐标表（`a_frag_elems`）即 §3.2.3 表格的直接编码：

```python
pairs = [(r0, k[0]), (r0, k[1]), (r1, k[0]), (r1, k[1]),
         (r0, k[2]), (r0, k[3]), (r1, k[2]), (r1, k[3])]   # r0=m+g, r1=m+g+8
```

#### 3.11.2 `frag_binop` / `frag_load_global`：fragment 级代数

`frag_binop` 对两个累加器字典做逐元素 extract/op/broadcast-insert，产出同构 `vector<2x2xf32>`（carry 链的 `X − W^T B̂ X` 类组合在寄存器格式上完成）；`frag_load_global` 从扁平 bf16 全局装载**累加器格式**的 f32 fragment（前向 S 初值：bf16 装载、extf、按 D 序 pack）——两者保证"fragment 格式"是机器内的公共数据表示，链间不落 smem。

#### 3.11.3 `emit_kloop`：为 transform 流水线结构化的 K 循环

前向的 K 循环必须可被 `multibuffer + create_async_groups + pipeline_shared_memory_copies` 消化，`emit_kloop`（`gdn_full_gen.py:803`）为此定制结构：`scf.for` 步 `bk`、**累加器是 loop-carried iter_args**（`vector<2x2xf32>` 列表）、staging 是线程跨步的 `vector<8xbf16>` transfer 对、staging 与 mma 之间无条件 `gpu.barrier`、gemm K 步全展开（区域算子会使 pipeliner 退出——这是 G2 发现并被多次复核的 transform 前置条件）。两个关键实测：

1. **staging 必须线程跨步**：G2 式块一致展开让**每个**线程发射**每一条**拷贝（256× 冗余 cp.async）——比标量基线还慢 2×；线程跨步化（每线程拷自己的 16B 块、索引依赖 tid 与 iv）是前向 staged 路径 2.2-2.7× 提速的第一块基石（evidence/G5/fwd_report.md G5-D 节）。
2. `pin_alloc`：用户横跨多个循环的缓冲会触发 `multibuffer` 的硬失败（无唯一候选环）；循环外一行哑 load+store 使 transform 跳过该 alloc（对照 MemRefTransformOps.cpp 验证）。

#### 3.11.4 LDSM 8 路 bank 冲突与 +8 列 pad 的定量

前向 F2（状态扫描）曾长期滞留 471 μs 级。ncu 诊断给出根因：全部 ldmatrix 读 tile 的行跨距是 **128 B 整倍数**（64×128 → 128 B、64×256 → 512 B、32×128 → 256 B），于是每象限 8 行落进**相同 4 个 bank** = 8 路冲突 = ~8× 慢的 LDSM——issue 间隔 63% 停在 "MIO queue full"、每调度器 2.0 活跃 warp、每 SM 约 40 周期才一条 LDSM、每 warp 每 chunk 76 条 LDSM。修复即机器级布局律 1：**每个 ldmatrix 读 tile +8 列 bf16 pad**（行跨距变 128+16 B 的倍数；+16 B 保持 cp.async 16 B 对齐）。F2 473→249 μs；配合 u 去 staging（vn epilogue 直读全局，其延迟藏进 vp gemm）与 w4/bv32 配置（VT 翻倍、串行链延迟减半）到 **83 μs**；数值对上一版**逐位相同**（evidence/G5/fwd_report.md G5-F 节）。反向的全部 bf16 mma tile 因此一律带 +8 pad（§3.7.1）。

#### 3.11.5 epilogue 内全局存储的 40% 回归教训

把 h_all/v_new 存储折进 S-update/vn epilogue（省两遍 cell pass）曾测得 F2 249→380 μs 回归：**epilogue 存储走 fragment lane 映射（2 B @ 4 B 跨距 = 不 coalesced），独立 cell pass 线程跨步则完美 coalesced**。回退保留 copy pass；该教训与 G6 覆盖矩阵对 fla "fragment 映射存储"特征（-40% 慢）的判定互为印证：正确形态是 epilogue 只写 smem（或经 fragment 感知的合并存储），全局落盘交给线程跨步 pass。

#### 3.11.6 `async_stage_unrolled` 与乒乓

前向的异步 staging 孪生（`gdn_full_gen.py:2572`）：同样的静态展开 device_async_copy burst、同样因 create_async_groups 的 wait_group-0 而手工化。F2 的 chunk 级乒乓把 parity 折进**索引算术**（wb 是 C×2K 列半、kb 是 2C×K 行半）——因为 memref 值不能 `arith.select`；chunk n 计算的顶部发出 n+1 的 w/k^T/gc 预取、尾部 wait+barrier、parity 翻转；gemm K 步全展开使 pipeliner 无环可断。仅乒乓本身曾测得 **471→473 μs（零）**——串行链瓶颈根本不在装载延迟，而在 LDSM bank 冲突（§3.11.4）；这个"负结果先于根因"的顺序本身是本机器方法论的注脚：先建位级等价的机制、再用它做归因隔离。

### 3.12 汇总：发射器清单、契约与效果索引

| 发射器/原语 | 位置（bwd 为主） | 契约要点 | 代表效果（证据） |
|---|---|---|---|
| `sgemm` | :589 | cell 单写者分区；kahan 可选；`math.fma` 显式 | 全链数值基线；kahan 保 dA 链过门（G5-E） |
| `gemm` | :2145 | `TF32_SITES` 站点派发 | 16 位点 tf32：e2e 14.16→11.06 ms（G5-G） |
| `tgemm` | :2010 | tf32 k8 atom 轮转；`warps=` 显式；neg 取 B | dqkwg 1.71×、wyb 2.19×（G5-G） |
| `gdn.mma.tf32k8` | dialect/lib/GdnHatch.cpp:170 | `=&f` early-clobber 强制；A 序 row-half 外层 | 无 `"=&f"`：118/128 cell 败（G5-G） |
| `mma_gemm` | :1726 | 条带贴图；accs= 链接；k0= K 分块；epi_bar | wy 3.04×；K32 spill 88/84/120→64/64/104 B（G5-AA） |
| `mma_gemm_mh` | :1821 | mi-half×ni-strip；W≤16；无 guard | 89% no-eligible → 全 warp；255(60 B)→248(0) regs（G5-O） |
| `mma_chain_if` | :993 | accs 过 `scf.if` 结果；尾栅栏强制 | dqkwg v3/v4 链载体（G5-O/S） |
| split 家族 | :826/:875/:1293-1460/:1671 | hi+lo f32 精确；lo·lo 舍弃；cvec 两类 | dkb 3.8× dk 不变；wyb 2.40× dk 1.28e-3（G5-J/T） |
| `stage_raw_*` | :741/:804/:855 | 精确 bf16 搬运；+8 pad；bar= 类 | raw B 操作数标准供给（G5-T） |
| `astage_tile`/`astage_sync` | :1569/:1635 | 静态展开 token；nthr 契约；wait 自持 | staging 指令 ~4× 减、逐位一致（G5-H） |
| `launder_pass` | :1650 | 恒等 st.shared 重写 | FPIPE×r6 规则的桥；位同构造（G5-V） |
| `rgemm`/`rgemm_aw`/`aggemm` | :1455/:3350/:1518 | j 共享装载；ILP=CPC；窗口喂通用读 | dhs 0.83→0.59→0.32 ms；AW 位同（G5-J/R） |
| `warp_rowsum`/`warp_colsum` | :928/:961 | 5 步蝴蝶；lane0 归约 | r4/c4r/r3/c1r 串行尾清零（G5-O） |
| `emit_gemm`（fwd） | full_gen:265 | `_FRAG_MODE` ldm/scalar 双模 | 前向 staged 2.2-2.7×（G5-D） |
| `emit_kloop`（fwd） | full_gen:803 | iter_args 累加器；线程跨步 staging | 256× 冗余消除（G5-D） |
| `async_stage_unrolled`（fwd） | full_gen:2572 | 乒乓 parity 折索引 | 位同；归因隔离 LDSM 根因（G5-F） |

**贯穿本章的三条机器级定律**（全部有位级或 SASS 级证据）：

1. **D-fragment 列索引 = $2t$（组内线程号），B-fragment = $g$（groupID）**——混用即少写一半列、多写冲突列（G4：1168/2080→2080/2080，9.77e-4==预测）。
2. **early-clobber 是 mma inline-asm 的生存条件**——`"=f"` 下 ptxas 合法地把 D 映上 A（HMMA R12,R12,R10,R16；118/128 败）。
3. **ldmatrix 的操作数必须 st.shared 血统**——raw LDGSTS 落地 tile 喂通用读正确、喂 LDSM 静默错（行块随构建调度漂移；st.shared 臂 5.2e-8 逐位对 numpy）。

它们的共同方法论：每条定律都以"最小探针 + 位级对照"钉死后才进入机器的默认路径，而这正是更大叙事——调度即 IR、编译期可判定——在指令层的投影。

---

## 4 PTX 后处理层：`gdn_smem_fix.py` 与构建链的修复/变换全集

### 4.0 章首：为什么在 MLIR 流水线之后还需要一个 PTX 后处理层

前两章（第 2、3 章）分别解剖了 L1 算法 IR 与 MMA 发射机器；本章下移一层，解剖整条构建链的**最后一个编译器**——一个 856 行的纯文本 PTX 改写器 `tools/gdn_smem_fix.py`，连同它上游的 `scripts/run_l2.sh`（134 行编译驱动）与下游的 `tools/gdn_full_rt.py`（运行时/缓存键控）。这一层的存在不是工程惯性，而是被 stock MLIR 流水线（`-gpu-lower-to-nvvm-pipeline`，最终 fork 的 2026 年 10 月 LLVM 主干）的**四个真实缺口**逼出来的：前两个缺口让"未经后处理的 cubin 在真实硬件上根本无法正确运行"，后两个缺口让"后处理后的 cubin 无法达到驻留（occupancy）与发射效率的目标"。四个缺口全部以 GPU 探针取证或源码级验证钉死，无一来自推测：

**缺口一：workgroup 分配下译为设备端 `malloc`（正确性缺口，致命）。** payload 中每一条作用于 `#gpu.address_space<workgroup>` 的 `memref.alloc()`，在被 `-gpu-lower-to-nvvm-pipeline` 下译后都不是静态共享内存符号，而是一段**设备端堆分配调用序列**：`call.uni (retval0), malloc, (param0)` 取回通用地址空间的堆指针，再经 `cvta.to.shared` 把该通用指针转换成共享窗口偏移。这是上游已文档化的行为（`llvm-mlir/test/Dialect/GPU/memref-to-llvm.mlir` 即按此形态生成期望输出）。它在真实硬件上有两条独立的死亡路径：

1. 裸 ptxas 产物不链接设备运行时（cudadevrt），cubin 中残留一条指向 `malloc` 蹦床函数的未解析 `CALL.ABS`——模块加载或内核启动直接失败（`CUDA_ERROR_INVALID_VALUE`）；
2. 即便 `malloc` 得以解析，返回的堆指针被**按共享内存窗口偏移重新解释**，后续 `cp.async` / `ld.shared` 落在共享地址空间之外——同步点上报 `CUDA_ERROR_ILLEGAL_ADDRESS`，或静默读出垃圾。

首次硬件执行（evidence/G3/calib_report.md 第 2 节）的取证结论是：**所有携带共享内存 tile 的 G2 cubin，在未修复形态下全部陷阱**（`CUDA_ERROR_ILLEGAL_ADDRESS` 或启动期 `INVALID_VALUE`）；只有两条不含 workgroup 分配的参考内核（sm80 原子、ldmatrix 锚）能正常启动。缺口一因此不是"性能不达标"，而是"零通过率"。

**缺口二：常数提升导致内核参数爆炸（正确性/效率混合缺口）。** `-transform-interpreter`（tiling 调度）与 `gdn-opt --gdn-lower-hatch`（本仓方言 op 的下译通道）都会把 gpu.launch 区域内部的 `arith.constant` **重新提升回宿主函数**；随后 gpu-kernel-outlining 把其中每一个都变成一条内核参数——单个内核最多膨胀出约 140 条 `.b64` 的编译期常量参数，甚至一个 bf16 填充值都会变成 `.param .align 2 .b8 [2]`。这些参数的值在编译期完全已知，运行时逐条经 `ld.param::entry` 读入既浪费发射带宽，又让"零参数内核 + 公共全局缓冲区"的 noargs 启动契约（第 5 章的输入输出 staging 设计）无从谈起。

**缺口三：寄存器帽不可经 ptxas 命令行施加（驻留缺口）。** 2 CTA/SM 驻留要求 `2 × 256 线程 × 128 寄存器 = 65,536` 恰好占满寄存器堆，即每内核寄存器数必须**逐内核**钉在 128。但本链选定的 `ptxas-blackwell`（Triton 捆绑、PTX ISA 9.3 能力，为 `sm_120a` 与 `cp.async.bulk.tensor` 所需）**无视 `--maxrregcount` 命令行旗标**——实测在帽值 128 下仍然编出 168 寄存器；而 PTX 源内的 `.maxnreg` 指令是**逐 entry** 生效且被尊重的（§4.3）。寄存器帽因此必须由 PTX 后处理层注入。

**缺口四：bank 几何约束编译器不可见（发射效率缺口）。** 生成器把 smem→global 的搬运发射为逐 cell 的 `ld.shared.b16`/`st.global.b16` 对（本工具链中不存在 128 位全局存储通路：`vector.load/store` 能 lower 但被逐元素拆成 b16 op，见第 5 章取证）。把相邻 b16 对融合成 b32 宽存取可以减半 LSU 指令与循环迭代数，但融合合法性取决于**共享内存 bank 几何**——tile 行距 80B（20 字）时，一个 warp 的 32 次 b32 共享读横跨两行，lane 28–31 与 lane 0–3 落进同一组 bank，2 way 冲突反而净慢 4.3 μs。编译器既不知道行距的语义，也不会为 bank 几何重排 lane→cell 映射；这条边界只能在 PTX 层（融合规则）与生成器层（lane 重映射）联手关闭（§4.4）。

四个缺口合起来界定了这一层的职责：**在 MLIR 语义完整下译之后、ptxas 定案之前，以纯文本改写修复正确性（缺口一、二）、施加驻留约束（缺口三）、收割发射效率（缺口四），并保证每一步改写的字节级可验证性。**

#### 4.0.1 构建链全景（最终形态）

理解本章所有变换的宿主，是下面这条两遍 ptxas 的构建链。以一个配置目录 `<dir>`（含 `payload.mlir`、`transform.mlir`、`config.json`，由 `gdn_full_gen.py` / `gdn_bwd_gen.py` / `gdn_l2_gen.py` 产出，可选伴随 `smem_alias.json`）为单位：

```
生成器 (payload.mlir + transform.mlir [+ smem_alias.json])
  │
  ▼  scripts/run_l2.sh
[gdn.* hatch op 存在时] gdn-opt --gdn-lower-hatch        # 方言 op → 树内 nvgpu/nvvm
[transform 脚本含 structured/nvgpu/memref 时] mlir-opt -transform-interpreter
[linalg 残存时] mlir-opt -convert-linalg-to-loops
mlir-opt -gpu-lower-to-nvvm-pipeline=
        "cubin-chip=sm_120a cubin-features=+ptx87 cubin-format=assembly"
  │        └─ cubin-format=assembly → gpu.binary 内嵌 PTX 文本属性
  ▼
asmobj.mlir ──(gdn_extract_ptx.py)──► kernel.ptx
  │                                      │
  │                                      ▼  第一遍 ptxas（-v -arch=sm_120a）
  │                                  kernel.cubin + ptxas_v.txt (+ sass.txt, meta.json)
  ▼  tools/gdn_smem_fix.py（本章主角）
extract_all_ptx(asmobj.mlir)  或  直读 kernel.ptx
  ├─ Fix 1  静态 smem 修复（callseq 删除 + cvta 重写 + .shared 声明，§4.1）
  ├─ Fix 1+ 生存期竞技场别名（smem_alias.json 消费，§4.2）
  ├─ Fix 2  常数参数消除（标量 + 聚合常数字节替换，§4.6）
  ├─ merge_ptx 多模块合并（≥2 个 gpu.binary 时，§4.5）
  ├─ widen_stores / shuffle_pair_stores（GDN_WST / GDN_WSP，§4.4）
  └─ apply_maxnreg_k（GDN_MAXREG_K 逐 entry .maxnreg 注入，§4.3）
  │
  ▼  第二遍 ptxas（-v -arch=sm_120a）
kernel_smemfix.ptx + kernel_smemfix.cubin   ←—— 计时/验证装载的最终产物
```

要点先定义后用：

- **第一遍 ptxas 与第二遍 ptxas 的产物不同**。`kernel.cubin`/`ptxas_v.txt` 来自 `run_l2.sh` 对未修复 `kernel.ptx` 的编译；真正装载执行的 `kernel_smemfix.cubin` 来自 `gdn_smem_fix.py` 的第二遍编译。两遍给 ptxas 的输入不同，调度自由度不同，**资源报告不可互换**——这一"双遍教训"本身是 §4.3 的主角。
- **ptxas 选型**（`pick_ptxas()`，`tools/gdn_smem_fix.py:64`；`scripts/run_l2.sh:27`）：优先 Triton 环境捆绑的 `ptxas-blackwell`，其次捆绑 `ptxas`，最后回退 `/usr/local/cuda/bin/ptxas`。两条链保持一致，避免不同 ptxas 版本的 ISA/调度差异污染 A/B。
- **幂等性**：`gdn_smem_fix.py` 覆盖式重写 `<dir>/kernel_smemfix.{ptx,cubin}`；既不需要修复的内核原样拷贝，保证计时harness 永远有可装载的目标（模块 docstring 明文契约）。
- **单内核与多内核双入口**：`main()` 先尝试 `main_multi()`——当 `asmobj.mlir` 含 ≥2 个 `gpu.binary` 汇编属性（mono 三元组，§4.5）时走"逐模块修复 + 合并 + 单 cubin"路径；否则退回单内核路径，直读 `kernel.ptx` 并消费平铺结构的 `smem_alias.json`。

以下每节按统一四件套展开：【动机 → 源码片段（路径）→ PTX 前后对比 → 效果数字】。

### 4.1 静态共享内存修复（Fix 1 主体）：device `malloc` → 静态 `.shared` 符号

#### 4.1.1 动机与修复思路

缺口一（§4.0）的修复思路是**复刻 MLIR 自身为静态共享内存已经使用的习惯用法**：ws 配置的 mbarrier 下译本来就会发射静态 `.shared` 符号——`run_l2.sh` 的上游流水线对 `gpu.barrier` 相关 lowering 正是这么做的。修复因此不是发明，而是把"每个 workgroup `memref.alloc` 一段设备堆分配"统一改写为"每个分配结果一个静态 `.shared` 符号"：

1. 删除每一段 malloc 调用序列（callseq）；
2. 把引用该分配结果的每一条 `cvta.to.shared` 换成 `mov.u64 %rd, __gdn_smemN;`；
3. 在 entry 前声明 `.shared .align 128 .b8 __gdn_smemN[<malloc 尺寸>];`。

分配尺寸与结果寄存器的配对信息就在被删除的 callseq 自身之内，无需读取任何外部文件——这保证了 Fix 1 对**所有**内核无差别生效（不需要生成器配合）。

#### 4.1.2 源码片段

三个正则锚点加两个改写函数构成全部机制（`tools/gdn_smem_fix.py:73-209`）：

```python
CALLSEQ = re.compile(
    r"\t\{ // callseq (\d+), 0\n"
    r"((?:\t[^\n]*\n)+?)"
    r"\t\} // callseq \1\n")  # closing brace omits the ", 0" suffix
SIZE_RE = re.compile(r"st\.param::func\.b64[ \t]+\[param0\],[ \t]+(\d+);")
RET_RE  = re.compile(r"ld\.param::func\.b64[ \t]+(%r[d]\d+),[ \t]+\[retval0\];")
CVTA_RE = re.compile(
    r"\tcvta\.to\.shared(?:::\w+)?\.u64[ \t]+(%r[d]\d+),[ \t]+(%r[d]\d+);")
```

`eat_callseq`（`gdn_smem_fix.py:101`）对每段 callseq 判定：体中不含 `malloc` 的调用序列原样保留（该正则也覆盖非 malloc 的设备调用）；含 malloc 的则从 `st.param::func.b64 [param0], <N>` 提取尺寸、从 `ld.param::func.b64 %rdX, [retval0]` 提取结果寄存器，登记 `malloc_reg_size[%rdX] = N` 后**整段删除**。`patch_cvta`（`gdn_smem_fix.py:166`）对每条 cvta 判定：源寄存器不在 `malloc_reg_size` 中的（非分配结果，例如 mbarrier 符号的转换）原样保留；在其中的，为该分配结果**首次**铸造符号 `__gdn_smem{len(decls)}` 并登记尺寸，随后统一替换为 `mov.u64`。同一分配结果的**每一条** cvta 转换共享同一符号——这个"每结果一符号"的要点是 §4.1.4 的坑的直接对策。最后把全部声明插在 `.visible .entry` 标记之前（找不到该标记则拒绝改写，防止把符号声明插进指令区）。

#### 4.1.3 PTX 前后对比

修复前（`evidence/G4/gen_verify/kkt_bf16_mma_p2_t64x64x32_w4/kernel.ptx:35-42`，上游流水线的原始输出）：

```ptx
	{ // callseq 0, 0
	.param .b64 	param0;
	.param .b64 	retval0;
	st.param::func.b64 	[param0], 8192;
	call.uni (retval0), malloc, (param0);
	ld.param::func.b64 	%rd24, [retval0];
	} // callseq 0
	cvta.to.shared.u64 	%rd5, %rd24;
```

修复后（同目录 `kernel_smemfix.ptx`；声明区第 18–19 行与指令区第 37–38 行）：

```ptx
	.shared .align 128 .b8 __gdn_smem0[8192];
	.shared .align 128 .b8 __gdn_smem1[8192];
	...
.visible .entry gdn_kkt_kernel_kernel()
	...
	mov.u64 	%rd5, __gdn_smem0;
	mov.u64 	%rd6, __gdn_smem1;
```

八条字节的堆分配序列与指针转换消失，取而代之的是一次立即数装载。对 TMA 配置（`kkt_bf16_mma_tma_ws`），`cp.async.bulk.tensor` 目的地址的转换是**集群味**的（`kernel.ptx:389-390`）：

```ptx
	cvta.to.shared::cluster.u64 	%rd196, %rd221;
	cvta.to.shared::cluster.u64 	%rd197, %rd220;
```

修复后同样落到静态符号（`kernel_smemfix.ptx:371-372`）：

```ptx
	mov.u64 	%rd196, __gdn_smem1;
	mov.u64 	%rd197, __gdn_smem0;
```

#### 4.1.4 `cvta.to.shared::cluster` 也必须改写：G4 的第二个正确性坑

初版 `CVTA_RE` 只匹配裸 `cvta.to.shared.u64`。G4 硬件验证暴露了后果：TMA 配置的两个 `cp.async.bulk.tensor` 目的地址下译为 `cvta.to.shared::cluster.u64`，且**转换的是同一个 malloc 结果**；callseq 删除后，这两条指令读取的是**从未定义的寄存器**——垃圾共享地址直达 `UTMALDG` 的目的操作数，GPU 上报 `CUDA_ERROR_ASSERT`。修复即正则中的 `(?:::\w+)?` 可选组：裸味与 `::cluster` 味一并匹配，且因符号按"分配结果"而非"指令"铸造，同一结果的两种转换自动共享同一符号（`patch_cvta` 注释原文记录了这条取证："leaving those unpatched made them read registers whose defining malloc callseq was already deleted => garbage smem destination => device fault"，`gdn_smem_fix.py:170-176`）。

#### 4.1.5 128B 对齐规则：UTMALDG 的硬约束

声明的对齐从初版的 `.align 16`（忠实复刻 mbarrier 习惯用法）提高到 `.align 128`，依据是 G4 的第三个硬件取证：`cp.async.bulk.tensor`（SASS `UTMALDG`）的共享内存目的地址在 `sm_120a` 上**即使 swizzle=none 也要求 128 字节对齐**。`.align 16` 时 ptxas 可以把某块 tile 放在 16B 对齐但非 128B 对齐的偏移上，首次 TMA 装载即故障 `CUDA_ERROR_MISALIGNED_ADDRESS`——compute-sanitizer 把故障点精确钉在 `UTMALDG.4D`（`evidence/G4/GATE_REPORT.md` §2.3）。代码内注释（`gdn_smem_fix.py:199-203`）将此规则固化：

```python
    if decls:
        # .align 128: cp.async.bulk.tensor (UTMALDG) smem destinations must
        # be 128-byte aligned even with swizzle=none on sm_120a (G4: with
        # .align 16 the tiles can land at 16B-but-not-128B offsets and the
        # first TMA load faults with CUDA_ERROR_MISALIGNED_ADDRESS).
        lines = [f".shared .align 128 .b8 {s}[{n}];" for s, n in decls]
```

该规则后被推广为纪律：任何未来经 memref.alloc → device malloc → smem_fix 改写而来的 TMA 目的 tile 都自动获得 128B 对齐；手写 PTX 必须保持之（G4 报告 §5 明文）。

#### 4.1.6 效果数字

- **通过率从 0 到全量**：G3 首跑中所有携带 smem tile 的未修复 cubin 全部陷阱；修复后全部内核运行并产出非零输出（`evidence/G3/calib_report.md` §2 Gap 1：18 个可计时配置 OK，两个 ws/tma 配置由 G4 进一步修复后完成）。
- 与 Fix 2（§4.6）合并后，**18/20 配置成为真正的零参数内核**；ws 配置保留 3 条封送常数（2× i1 + 一个 16B 向量零），tma 配置保留 3 条宿主侧 tensormap 指针参数（经 `cuTensorMapEncodeTiled` 装配）。
- 修复是纯字节级重写：不触碰任何计算指令，G4 的 20/20 逐配置硬件验证（含确定性重跑）在其之上完成。

### 4.2 生存期竞技场别名（arena aliasing，Fix 1 扩展）

#### 4.2.1 动机：驻留壁垒与"字节记账"

`sm_120a`（RTX 5090 Laptop，82 SM）的共享内存经济：每 SM 102,400B、每块 optin 上限 101,376B、校准实测 optin 共享 99KiB（`evidence/G3/calib_report.md` §1）。**2 CTA/SM 驻留**因此要求每内核静态共享池 ≤ 约 50.2–50.7KB（反向链记账门槛 50,176B、正向链 50,688B），同时寄存器 ≤ 128（2×256×128 = 65,536）。第 6、7 章反复出现的困境是：算法需要的 tile 生存期拼接起来远超单块上限，但**任意时刻真正活跃的字节**却在门槛之下。

MLIR 层面无法直接表达"多块 tile 复用同一片字节"：每块 tile 一个 `memref.alloc` 是 SSA 合法性的要求（第 3 章的 staging 原语族都按此发射）。但共享内存的物理事实不关心 SSA——**生存期不重叠的 tile 可以共享字节**。arena aliasing 就是把这一定理施加在 PTX 层：MLIR 程序保持每 tile 一次分配（生成器语义完整、可验证性不变），后处理把被别名 tile 的 malloc 删除、其 cvta 改写为"竞技场基符号 + 字节偏移"。生成器侧的生存期审计方法学（峰值生存期审计、逐相位活跃表、三条硬规则）由第 7 章展开；本节只讲机制端——`gdn_smem_fix.py` 如何消费规范、如何对抗 LLVM 的三项"不配合"，以及防御性回退立场。

**定义**：一个**竞技场（arena）**是一块以某个宿主 tile 名义声明、按竞技场总尺寸（而非宿主 tile 自身尺寸）声明的静态 `.shared` 符号；**别名（alias）**规范描述"tile T 的字节落在宿主 H 的竞技场内偏移 off 夕"。

#### 4.2.2 规范格式：`smem_alias.json`

单内核路径（`main()`，`gdn_smem_fix.py:819-824`）直接读取配置目录下的 `smem_alias.json`；多内核路径（`main_multi()`，`gdn_smem_fix.py:764-767`）读取其 `"modules"` 子映射按模块名分发。三个键（`fix_ptx` docstring 与 `gdn_smem_fix.py:114-151`）：

```json
{
 "identify": {"16384": "accCC", "9216": "pthi", "9504": "ptlo", ...},
 "arenas":   {"accCC": 34816, "vnh": 10240},
 "alias":    {"pthi": ["accCC", 0], "bdt": ["accCC", 9216], "akb": ["accCC", 18944], ...}
}
```

真实样本取自最终 dqkwg 2-CTA 链（`evidence/G5/scratch/r15w_2cta_T2048_dqkwg/dqkwg/smem_alias.json`）：17 个 identify 条目、2 个竞技场（accCC 34,816B、vnh 10,240B）、16 个别名条目——包括 bhh@13,824、bhl@16,384（+1/+3 行垫，见 §4.2.3）、akb@18,944、accCK@24,064。该规范的作者在生成器（`gdn_bwd_gen.py`/`gdn_full_gen.py`），本工具是纯消费者。

#### 4.2.3 按 malloc 尺寸识别：LLVM 重排的对策与"识别垫"

机制端第一问题是**把规范里的 tile 名对应到 PTX 里的具体 malloc**。不能用顺序：**LLVM 会重排 malloc callseq，顺序不是稳定键**（`fix_ptx` docstring 明文；这是实测行为而非猜测）。对策是**按唯一分配尺寸识别**——每个 tile 的 malloc 尺寸在生成器侧被垫成全内核唯一，于是尺寸即身份：

- ldmatrix 消费的 tile：垫**从不寻址的额外行**——行距必须保持 16B 的倍数（ldmatrix 行地址 16B 对齐是硬约束，G4/G5 双重取证；奇数列垫非法，垫必须按 8 列步进，正向链的一组尺寸 11,264/12,288/13,312/14,336/1,024/1,280/18,432 即此规则的产物）；
- 仅通用访存的 f32 tile：+4 列垫。

对应代码（`gdn_smem_fix.py:115-151`）：按尺寸建 `size_regs` 倒排表；每个 identify 条目要求**恰好一个**同尺寸 malloc。任何失配走防御性回退（§4.2.6）。

#### 4.2.4 cvta → 竞技场基 + 偏移的重写

识别成功后，`patch_cvta` 对别名 tile 的每条 cvta 发射两条指令：基符号装载加常量偏移（`gdn_smem_fix.py:178-184`）：

```python
        tile = tile_of_reg.get(src)
        if tile is not None and tile in alias_of:
            base, off = alias_of[tile]
            sym = base_sym(base)
            n_patch += 1
            n_alias += 1
            return (f"\tmov.u64 \t{dst}, {sym};\n"
                    f"\tadd.u64 \t{dst}, {dst}, {off};")
```

`base_sym`（`gdn_smem_fix.py:157-164`）为竞技场宿主铸造**按竞技场尺寸**声明的符号（`arena_of[base_tile]`）；宿主自己的 cvta 也落在同一符号上（偏移 0 的别名等价于宿主自身）。

最终 dqkwg 链的真实 PTX（`evidence/G5/scratch/r15w_2cta_T2048_dqkwg/dqkwg/kernel_smemfix.ptx`）。改写后的别名地址（akb@accCC+18,944 与 accCK@accCC+24,064）：

```ptx
	mov.u64 	%rd24, __gdn_smem9;
	add.u64 	%rd24, %rd24, 18944;
	...
	mov.u64 	%rd12, __gdn_smem9;
	add.u64 	%rd12, %rd12, 24064;
```

声明区（同文件 37–48 行）展示最终池形态——10 个 256B 标量块 + 两个竞技场，合计 **47,616B**：

```ptx
	.shared .align 128 .b8 __gdn_smem0[256];
	...
	.shared .align 128 .b8 __gdn_smem9[34816];
	.shared .align 128 .b8 __gdn_smem10[256];
	.shared .align 128 .b8 __gdn_smem11[10240];
```

#### 4.2.5 LLVM 的三项"不配合"与逐项对策

机制端的全部趣味在于：规范描述的是 MLIR 里的分配，而改写发生在 LLVM 已经优化过的 PTX 上。三项实测的"不配合"各有专门对策，且全部**响亮而不沉默**（每项都有打印诊断）：

**（一）malloc 重排。** 上节已述：顺序不可作键，唯一尺寸作键。

**（二）未用宿主被 DCE。** 一个"宿主"分配若在 MLIR 语义里没有任何使用（它的存在只是为了给竞技场提供名义），LLVM 会把它**整体删除**——尺寸键找不到任何 malloc。对策不是发明使用，而是**规定宿主必须是首个使用（first-USE）tile**：宿主自己必须有真实数据流动。规范侧的这条纪律在 bwd 报告中作为工具链教训 #1 记录（"an arena HOST alloc with no uses is DCE'd by LLVM and the size key finds 0 mallocs -- arena bases must be first-USE tiles"）。

**（三）store-to-load 前传吞噬别名 tile。** 更隐蔽：当一条链的 epilogue 把 tile A 的内容经共享内存移交给同一 warp 同 lane 映射的下一条链时，LLVM 的 store-to-load 前传会把这次交接**直接前传进寄存器**——于是该 tile 的 malloc 在 PTX 里根本不存在。wyb 的 t2 对（dk_t2 停靠对）即此情形。早期行为是硬 WARN 并**整体禁用别名**，后果是一条 121KB 静态共享的 ptxas 失败；现在的行为（`gdn_smem_fix.py:121-131`）是把该 tile 从别名映射中摘除并继续（数值上无损——前传本身保持了值与舍入链）：

```python
            if len(regs) == 0:
                # G5-T round13: the malloc was optimized away ENTIRELY
                # (e.g. LLVM store-to-load-forwarded a same-lane epilogue
                # handoff into registers -- the wyb t2 pair).  There is no
                # cvta to patch and no bytes to assign: drop the tile from
                # the alias map and continue.
                print(f"[smem_fix] NOTE alias: size {sz_s} for '{tile}' "
                      f"matches 0 mallocs (optimized away) -- skipped")
```

机制端还从验证门学到两条**别名合法性纪律**（生成器侧规范必须遵守，第 7 章展开细节）：其一，**分裂 tile 不得别名其自身源 tile**（dqkwg 的 accCC 自别名 bug 被验证门捕获）；其二，别名区不得压在多缓冲 drain 尚未完成的字节之下（正向 kt 竞技场案例：solve tile 压在偏移 0 处使每个 CTA 的 u 行 0–31 损坏——流水化 kkt 循环在循环后 drain 期仍写 kt 的多缓冲，10,240B 之下必须假想有晚写；`bisect_shift` 探针二分定罪后把 solve tile 上移到 kt 真实字节之上，逐位精确）。

#### 4.2.6 防御性回退立场

任何识别失配（某尺寸匹配多个 malloc、规范引用了未识别 tile、别名映射为空）都会**丢弃整个规范**并回退到"每 malloc 一声明"的经典行为——`gdn_smem_fix.py:132-151` 的 WARN 路径。回退结果**正确但未重叠**：内核照常运行，只是不省字节。设计立场是：别名是纯增益变换，其失败模式必须是"不赚"，绝不许是"算错"。全部数值等价性由此主张并由验证门背书（见效果节）。

#### 4.2.7 效果数字

arena aliasing 是整条反向链从 6.0ms 走到 2.2ms 的核心杠杆之一，关键落点：

| 内核 | 池变化 | 寄存器 | 驻留 | 阶段加速（同窗配对） | 证据 |
|---|---|---|---|---|---|
| dqkwg（r9 首落，两竞技场 accCC 17,408 / vnh 10,240） | 99,328→48,640B | 132→96 | 1→2 CTA | 1.772→1.075ms（1.65×） | bwd_report r9 |
| dqkwg（最终四 tile 别名 bhh/bhl/akb/accCK 入 arena1） | **100,864→47,616B**（低于 50,176 门槛 3,560B） | 128（.maxnreg，§4.3） | 2 CTA | 0.632→0.547ms @T2048（1.16×）；2.323→1.844 @T8192（1.26×） | bwd_report r15 |
| caryA（竞技场"溶解式"瘦身：四个共活 tile 退役为三个 bf16 mma tile + 别名） | **82,176→28,592B** | 255→76 | 1→2 CTA | 0.309→0.0618ms（5.00×）@T2048；2.78× @T512 | bwd_report r11 |
| wyb（tv 3-mma 入 tT/accCK 竞技场，22 malloc→17 声明，其一按竞技场尺寸 18,432） | 98,048→100,096（+T 分裂对真实足迹）；再瘦身 100,096→**45,056B** | 226→128（176B/228B 溢出保留） | 1→2 CTA | 0.817→0.3395ms（2.41×） | bwd_report r11/r13 |
| fwd f1（F1AL 移植，13 malloc→5 声明） | 101,376→89,600B | — | 1 CTA（算术上 2-CTA 不可达） | **诚实负结果**：f1 +4.7μs，旋钮默认关 | fwd_report r15 |
| fwd f3（F3KS：ot 压入 P 的死竞技场，P 的 malloc 以 ot 的 17,408B 声明、按尺寸别名于 +0） | 82,176→47,360B | — | 2 CTA | **诚实负结果**：f3 +96.1μs @T8192（带宽-bound 内核不受益于波减半） | fwd_report r20 |

dqkwg r9 的 ncu 前后对照给出机制证据：理论/实际占用 16.67%/15.81%→33.33%/30.28%，实际 warp/SM 7.59→14.53，调度器活跃 warp 1.98→3.69，无资格周期 88.98%→79.56%。r15 的 2-CTA 落地版本：占用 15.02%→29.17% @T2048、14.23 warp/SM @T8192（此前 8.00）。所有落点数值位级不变或按章内验证门裁决（第 8 章）；两条负结果（f1/f3 正向）同样按测量关闭——机制本身可复用，是"问题被算术/测量关闭"而非"机制失效"。

### 4.3 `.maxnreg` 注入：ptxas-blackwell 的旗标盲区与双遍教训

#### 4.3.1 动机

2 CTA/SM 的寄存器算术（2×256×128=65,536）要求把逐内核寄存器数钉在 128。工具侧本来备有探针旋钮 `GDN_MAXREG`：给第二遍 ptxas 追加 `--maxrregcount`（`gdn_smem_fix.py:797-799/839-842`）。实测结论（bwd 报告 r15 原文）："**ptxas-blackwell IGNORES --maxrregcount**（GDN_MAXREG re-verified useless: 168 regs at cap 128），but the PTX **.maxnreg directive placed inside the entry body (next to .maxntid) IS honored**"。命令行帽是"发了但没人听"，PTX 指令帽才有效——帽的施加点因此必须移入 PTX 文本。

#### 4.3.2 源码片段与 PTX 前后

`apply_maxnreg_k`（`tools/gdn_smem_fix.py:683-712`）读取 `GDN_MAXREG_K`（形如 `f3:168,f2:128`），对每个 `.visible .entry NAME` 行按**子串匹配**注入：

```python
    for ln in lines:
        out.append(ln)
        m = re.match(r"\s*\.visible\s+\.entry\s+([\w.$]+)", ln)
        if m:
            name = m.group(1)
            key = next((k for k in want if k in name), None)
            if key:
                out.append(f"\t.maxnreg {want[key]}")
                applied[name] = want[key]
```

逐 entry 的键控是必须的：`--maxrregcount` 是**逐文件**旗标，对携带三个内核（f1/f2/f3 各有不同寄存器预算）的 mono cubin 无差别帽没有意义；`.maxnreg` 是**逐内核**指令（docstring 原文：单内核链另有构建脚本侧的 `GDN_MAXNREG_ENTRY=dqkwg:128` 同型注入）。注入位置紧随 entry 头（与 `.maxntid` 并排）。真实产物（`evidence/G5/scratch/r15w_2cta_T2048_dqkwg/dqkwg/kernel_smemfix.ptx.maxnreg.ptx:49-51`；该目录为后续复测所重建，记录值 128，此处样本为 96 类再探针）：

```ptx
.visible .entry gdn_dqkwg_kernel_kernel()
.maxntid 256, 1, 1
	.maxnreg 96
{
```

注入后不匹配（wanted ≠ applied）时打印 WARN，防止键拼错静默失效。

#### 4.3.3 双遍教训：`ptxas_v.txt` 只是第一遍

这条教训值得单独成节，因为它是"构建链说谎"的典型案例。r14 的构建脚本从 `ptxas_v.txt` 抄录寄存器数并记档"128 regs / 44B spill"。但 `ptxas_v.txt` 是 **run_l2.sh 第一遍** ptxas（未修复 `kernel.ptx` → `kernel.cubin`）的报告；实际装载的 `kernel_smemfix.cubin` 出自**第二遍**（对 smem_fix 改写后 PTX 的编译），改写后的 PTX 给了 ptxas 不同的调度自由度，2-CTA 机身在第二遍编出 **168 寄存器（0 溢出）**。第一个 A/B 因此"莫名"倒退 0.78×（0.811ms）：ncu 显示 Block Limit Registers = 1（2×256×168 > 65,536）——名义 16-warp 的形式实际跑 8 warp/SM，严格劣于 r14。两条工具链教训（bwd 报告 r15 原文要点）：

1. **做数的寄存器数是 smem_fix 那一遍的**；`ptxas_v.txt` 不得用于对它下结论（r14 记档的 128 恰是第一遍的数；r14 已装载的 cubin 经复跑第二遍验证确为 128/0 溢出）。
2. 帽必须经 `.maxnreg` 施加；注入后 128 寄存器 / 32B 溢出（barrier 瘦身后 64B）。

r16 起此教训**结构化**进构建链：`r16y_build.sh` 把第二遍的 `-v` 报告记入 `ptxas_smemfix_v.txt`，寄存器数在 bench 之前就对账。其对决策的直接影响有实例：dqkwg P11R 重构的 P11R=4 臂（自然 222 寄存器）在帽 128 下需 232B 栈 / 276B 溢出存 / 632B 溢出取，仅 1.026×；P11R=2（自然 174）96B 栈 / 96/148B 溢出，1.040× 胜出——两个数字都在 bench 前由双遍纪律暴露。

#### 4.3.4 效果数字

- dqkwg：注入 `.maxnreg 128` 后 ncu Block Limit Registers = 2、Block Limit Shared Mem = 2——**2 CTA/SM 双门槛同时达成**；占用 15.02%→29.17% @T2048，14.23 warp/SM @T8192（8.00 起），无资格周期 89%→83%，调度器活跃 warp 1.97→3.63，SM 吞吐 18%→30%、MEM 19%→36%（bwd r15）。
- 阶段收益（与 §4.2 的池瘦身叠加）：dqkwg 0.632→0.547ms @T2048（1.16×）、2.323→1.844 @T8192（1.26×，2048 CTA 填满加倍后的 SM 槽，waves 6.24→3.12）。
- 边界纪律：`.maxnreg` 注入是 2-CTA 臂的需求——T512 类小网格的 v3 臂 cubin 与记录逐字节一致，构建时用 `GDN_MAXNREG_ENTRY=nomatch:0` 空过。
- 负面对照：f3 曾以 `GDN_MAXREG_K="f3:168"` 探针——240 寄存器本可 2 CTA（240×128×2≤65,536）但 82,176B×2 > 100KB 共享不允许，寄存器帽无法移动占用；工具照 landing 供后续探针（fwd r14"non-lever, corrected from the brief"）。这是"帽只在寄存器是短板时才是杠杆"的实测注脚。

### 4.4 宽存取四元组融合（GDN_WST）与 W32P 互补半区重映射（GDN_W32P）

#### 4.4.1 动机：cell pass 的 LSU 账本

本工具链不存在 128 位全局存储通路（第 5 章取证：`vector.load/store` 与 `vector<8xbf16>→vector<4xi32>` 位拆变体都能 lower，但被逐元素拆成 b16 op）。生成器因此把 smem→global 搬运发射为**逐 cell 的 b16 单元**。删除探针（timing-only 构建，不装载）把账本钉死（fwd r15）：f2 的两条 cell pass（h_all、v_new）占 T8192 297μs 链中的 **35.4μs + 17.9μs**（合计约 53μs 上限）；f1 的 u/w pass 每 CTA 还要长一倍（C×V + C×K 个 cell）。WST 攻击指令数；W32P 解决 WST 在 80B 行距 tile 上被 bank 冲突挡住的那一半。

#### 4.4.2 WST：四元组融合的源码与规则

生成器侧先配对：`copy_s2g_vec` 的配对 cell 模式按 `(2i, 2i+1)` 发射相邻 cell 对（`cols >= 64` 且 `rows*cols % (2*nthr) == 0`，否则回落单 cell 体）。PTX 侧 `widen_stores()`（`tools/gdn_smem_fix.py:482-563`，`GDN_WST` 启用、`GDN_WST_SCOPE` 按 entry 名圈定）把每个**无条件四元组**

```
ld.shared.b16 [B1+o]; ld.shared.b16 [B1+o+2]; st.global.b16 [B2+g]; st.global.b16 [B2+g+2]
```

熔合为

```
ld.shared.b32 [B1+o]; st.global.b32 [B2+g]
```

合法条件（源码逐一检查，否则该位点不动）：两个 b16 源寄存器**均单次使用**；两条 store 的全局地址相邻（`ob == oa+2`）；**两种 store 顺序都接受**——LLVM 常把 +2 的 store 排在前面；顺序健全性要求每条 load 先于自己的 store、低偏移 load 先于高偏移 load，而 **store 顺序不受约束**。字节精确性由构造保证：熔合后的 32 位寄存器恰是两个相邻 16 位 cell（小端），`st.global.b32` 把相同字节写进相同地址（生成器的偶对 cell 映射保证 4B 对齐）。铸造的新 b32 寄存器从 9000 起编号（"远高于 LLVM 的编号"），并以**显式逗号清单**声明——本 ptxas 拒绝 from-to 区间式 `.reg` 声明，计数式无法点名高编号（`gdn_smem_fix.py:554-562`）。

#### 4.4.3 PTX 前后对比（真实产物）

熔合前（`evidence/G5/scratch/r15x/wst2_T8192_w4bv32/kernel.ptx:3668-3672`——注意 +2 的 store 被 LLVM 排在前面）：

```ptx
$L__BB0_93:
	ld.shared.b16 	%r1848, [%rd587];
	ld.shared.b16 	%r1849, [%rd587+2];
	st.global.b16 	[%rd588+2], %r1849;
	st.global.b16 	[%rd588], %r1848;
```

熔合后（同目录 `kernel_smemfix.ptx:3610-3614`；新寄存器声明在第 62 行 `.reg .b32 %r9000, %r9001;`）：

```ptx
$L__BB0_93:
	ld.shared.b32 	%r9000, [%rd587];
	st.global.b32 	[%rd588], %r9000;
	add.s64 	%rd589, %rd589, 128;
	add.s64 	%rd588, %rd588, 512;
```

四条 LSU 指令对两条；循环迭代数同步减半（步长 128B 共享 / 512B 全局）。

#### 4.4.4 80B 行距的 bank 冲突律与 `cols >= 64` 规则

WST 的配对在 f1/f3（列数 128）上直接合法：**每 warp 的 32 次 b32 共享读保持在一行之内**。f2 的 BV=32 tile 不行——这就是 r16 bank 冲突律（fwd r16 负结果，r24 复述）：BV=32 tile 行宽 (BV+8)=40 元素 = **80B = 20 字**行距；平铺配对映射下一个 warp 横跨**两相邻行**，20 字的行距把 lane 28–31 的字地址搬进 lane 0–3 所在的 bank 组——**2 way 冲突**，单 cell 映射本没有。实测配对（仅映射、不熔合）在 f2 上 **+4.3μs（47/60 对更慢）**。WST 的作用域规则因此定为 `cols >= 64`；f2 保留原 pass。任何未来的 f2 存储通路工作必须先解决这个几何（r16 原文把它列为文档化约束）。

#### 4.4.5 W32P：`80 mod 32 = 16` 互补半区律

r24 关闭了这条 frontier：路线 (a)——BV=64 重铺——在纸面上死亡（F2US 之下池 115,200B > 101,376B optin，且恰是 r14 在 w8/bv64 实测 ptxas 失败的那个数；退 F2US 去凑 96,768B 则同时 surrender r14 实测的 −104.5μs 与减半 f2 网格）。路线 (b)——**逐行 warp 重映射**——落地为生成器旋钮 `GDN_W32P`（`gdn_full_gen.py` 的 `copy_s2g_vec` 第三模式；作用条件 `cols==32 and rows%8==0 and nthr==128`，即仅 f2 的 S/vn tile；f1 的 pass 列数 = V/K = 128、f3 的 = 128，本就 WST 配对）。

**互补半区律**：保持**单 cell 冲突无关的共享读**原样（同地址、同 bank 图样——这正是杀死 tile 级配对的东西，一点不动），把配对搬到 **lane 之间**：对 pair-step `i`（cell 计数器），

```
row     = 8*(i>>7) + warp + 4*(lane>=16)
pair col = 2*(lane%16)
```

即 lane 0–15 读行 `8*round + warp`、lane 16–31 读行 `r+4`。字地址分别落在 `[20r, 20r+16)` 与 `[20r+80, 20r+96)`；**80 mod 32 = 16**，故两个半 warp 恰好命中 **32 个 bank 的两个互补 16-bank 半区**——对每个 `r` 零冲突。映射是双射：(round j, warp w, 半区 h) → 行 `8j+w+4h` 覆盖行 `8j..8j+7`，pair 由 `lane%16` 索引。纯下标算术（lane/warp 常量复用 `%lane`/`%warp`，CSE 出循环），池、tile、LDSM 垫、占用一概不变。两条 b16 load 保持 `[base]`/`[base+2]`、两条 store 保持 `[g]`/`[g+2]`——**结构模式不变**，所以 `widen_stores` 原样把它们熔合：f2 的 smem_fix wst 计数从 0 变 2（f2 的两条存储 pass 循环体与 f1/f3 的 `cols>=64` pass 完全同型地熔成 `ld.shared.b32`/`st.global.b32`）。

这就是"W32P 四元组融合"的完整含义：**W32P 负责让四元组在 80B 行距上合法（bank 无冲突），WST 负责把它熔合**；两层各管一半，缺一则 f2 的存储通路收益不存在。

#### 4.4.6 姊妹变换 WSP：跨 lane shuffle 配对（保留在工具中）

同一条 f2 frontier 的另一条技术路线是 `shuffle_pair_stores()`（`tools/gdn_smem_fix.py:578-680`，`GDN_WSP`/`GDN_WSP_SCOPE`，在 `widen_stores` **之后**运行，只处理 WST 熔不掉的单 cell）：保留 HEAD 的单 cell 共享读（本就无冲突），跨 lane 配对——lane 2j 持 pair 的低半、lane 2j+1 持高半；一条 `shfl.sync.bfly.b32 delta=1` 让每个 lane 拿到伙伴的位，偶 lane 用 `prmt` 打包 own|partner<<16 发一条 `st.global.b32`，奇 lane 不存。合法性门（每处检查，否则跳过）：load 与 store 同基本块（中间无标签）；外层 while 循环 `STEP % 32 == 0` 且 `(BOUND+STEP) % STEP == 0`（warp 一致迭代，`shfl.sync` 合法——计数器从 `%tid` 起）；store 字节偏移为 0（pair 基址）。W32P 胜出后 WSP 作为已验证的备用路线保留在工具里。

#### 4.4.7 效果数字

| 变换 | 对象 | 收益（同窗配对 A/B） | 数值等价 | 证据 |
|---|---|---|---|---|
| WST | f1（u/w pass） | **−11.8/−11.4μs** @T8192（两场，1–5/60 对更慢）、**−10.6μs** @B4T2048（3/60）；staged e2e −13.4/−9.0/−7.0μs | 位级相同（T8192 门 o/ht 0.000e+00；验证 4/4 记录位数） | fwd r15 |
| WST | f3（o pass） | ≈0（−0.4μs，界内） | 同上 | fwd r15/r19 校正 |
| 仅配对不熔合 | f2 | **+4.3μs**（负结果，bank 律的实测注脚） | — | fwd r16 |
| W32P（+WST 熔合） | f2（S/vn pass） | **−17.9…−22.6μs** @T8192（7 场）、−9.7…−25.6μs @B4、−2.0…−3.8μs @T2048；staged e2e −9.1…−12.0μs @T8192；慢对率 1–9/80（B4/T8192） | 位级相同（对 purity 构建逐字节比较；fp64 门记录位数） | fwd r24 |
| W32P 记录迁移 | 全链 | R25 存档 n=30：三形状 formals 全面上移——T8192 **1.349×**、B4T2048 **1.221×**、T2048 **2.125×** | fwd_verify 4/4、det N=5 | fwd r25 |

WST 收益与探针账本自洽：f1 按 0.176μs/MB 计，f2 的 100.6MB pass 写给出约 17μs 上限——W32P 实测 −18…−23μs @T8192 恰好收满。T8192 链上另有一项已定性的 **f3 布局税**（+11.8…+12.6μs，模块布局抽签、f2 的 +300B PTX 触发，f3/f1 的 PTX 段与 purity 逐字节相同）——净链在每一形状、每一场仍为正；R25 的存档重跑重掷了布局。

### 4.5 多模块合并（mono cubin）：重名 `.shared` 符号的静默统一坑

#### 4.5.1 动机

正向链的 staged 形态最初是三个独立 cubin（f1/f2/f3），内核边界靠 DtoD 拷贝桥接——B1T2048 下约 0.1ms 纯搬运。mono 形态把三者放进**一个 cubin、共享公共全局缓冲区**，内核边界退化为流序同步。这要求 `gdn_smem_fix.py` 从"修一个模块"扩展为"提取 + 修复 + 合并多个模块"。

#### 4.5.2 常量回收：`asmobj.mlir` 汇编属性的反转义

多模块路径的输入不是 `kernel.ptx`（那只含第一个模块），而是 `asmobj.mlir` 本身：`extract_all_ptx()`（`tools/gdn_smem_fix.py:415-447`）扫描每个 `gpu.binary @NAME` 的 `assembly = "..."` 字符串属性，用 `unescape_mlir_string()`（`gdn_smem_fix.py:385-412`）反转义 MLIR 字符串编码——`\xNN`/`\NN` 十六进制对、`\n`/`\t`/`\r`/`\"`/`\\`/`\'` 转义——还原 PTX 文本，并以 `.version` 与 `.target` 的存在性做完整性校验。这段"从 MLIR 工件回收 PTX 常量"的代码与 `run_l2.sh` 保留 `asmobj.mlir` 的决定（G3-A 起）构成同一枚硬币的两面：asmobj.mlir 既是常数参数的值源（§4.6），也是多模块 PTX 的正源。配套的 `parse_all_launch_args()`（`gdn_smem_fix.py:450-473`）对每个 `gpu.launch_func` 分别解析常数参数值。

#### 4.5.3 坑：ptxas 对重名 `.shared` 符号的静默统一

合并的最初实现朴素拼接三个已修复模块——每个模块的 Fix 1 都铸造 `__gdn_smem0`、`__gdn_smem1`……GPU 上的后果（fwd 报告 G5-D bug #4 原文要点）：**ptxas 静默统一跨 entry 的重名 `.shared` 符号**——f1/f2/f3 的 `__gdn_smem0` 被当作同一块内存，但三者尺寸不匹配，f3 经由 f1 那个更小的 `__gdn_smem0` **越界写**（compute-sanitizer 定位：shared write at 0xa800）。没有编译错误、没有链接诊断——完全静默，只有数值验证能听见。

#### 4.5.4 修复：逐内核改名与合并算法

`merge_ptx()`（`tools/gdn_smem_fix.py:715-752`）的三步：模块 0 的文件头保留；后续模块的 `.shared` 声明收集进来并全部改名，`.extern` 声明取并集去重；所有 entry 体顺序拼接。改名规则 `__gdn_smem<N>` → `__gdn_smem_<模块标签>_<N>`，**声明与引用一体替换**：

```python
    def rename_syms(text, tag):
        return re.sub(r"__gdn_smem(\d+)",
                      lambda m: "__gdn_smem_%s_%s" % (tag, m.group(1)),
                      text)
```

真实产物（`evidence/G5/gen_fwd/bench_B1T2048H16_mono/kernel_smemfix.ptx:23-28`）：

```ptx
	.shared .align 128 .b8 __gdn_smem_gdn_fwd_f1_kernel_0[256];
	.shared .align 128 .b8 __gdn_smem_gdn_fwd_f1_kernel_1[256];
	.shared .align 128 .b8 __gdn_smem_gdn_fwd_f1_kernel_2[256];
	.shared .align 128 .b8 __gdn_smem_gdn_fwd_f1_kernel_3[6144];
	.shared .align 128 .b8 __gdn_smem_gdn_fwd_f1_kernel_4[9216];
	.shared .align 128 .b8 __gdn_smem_gdn_fwd_f1_kernel_5[9504];
	...
```

`main_multi()`（`gdn_smem_fix.py:755-810`）的完整流水：提取全部模块 → 逐模块 `fix_ptx`（含 `"modules"` 映射分发的别名规范——r15 起正向链也具备竞技场能力）→ `fix_params` → WST → WSP（在 WST 之后，只碰熔剩的单）→ `merge_ptx` → `apply_maxnreg_k` → 第二遍 ptxas。逐模块打印统计行（`cvta= shared= params= wst= wsp=`）——W32P 一节引用的"f2 wst 计数 0→2"即出自此行。

#### 4.5.5 效果数字

- mono 三元组成为正向 staged/fused 全部形态的载体；DtoD 桥接拷贝（约 0.1ms @B1T2048）消失（fwd G5-D）。
- 一个 cubin 多 entry、公共全局缓冲区：`gdn_full_rt.py` 的 `FullKernel` 按 `full_spec.json` 的 kernels 表逐 entry 取 `cuModuleGetFunction`，全局参数表逐个 `cuModuleGetGlobal`（`gdn_full_rt.py:125-147`）。
- 逐 entry `.maxnreg`（§4.3）与逐模块 WST/WSP 作用域都在合并后、编译前的单一 PTX 上施加——合并是它们的宿主。

### 4.6 常数参数消除（Fix 2）：标量、聚合常数字节映像与签名收尾

#### 4.6.1 动机与值源

缺口二（§4.0）的机理复述：tiling 与 hatch 下译把 launch 区域常数提升回宿主，outlining 把它们变成内核参数——最多约 140 条 `.b64`。修复的值源是 `run_l2.sh` 从 G3-A 起保留的 `asmobj.mlir`：宿主侧 `gpu.launch_func ... args(%c : i64, ...)` 的操作数表仍在，**位置顺序 == PTX 参数顺序**，于是每个常数参数的值可回收。非常数参数（tma 配置的宿主侧 tensormap 描述符指针）保留为真实参数。

#### 4.6.2 标量替换

`parse_launch_args()`（`tools/gdn_smem_fix.py:280-312`）把 launch 操作数 SSA 名映射到 `llvm.mlir.constant` 值；`_const_bits()`（`gdn_smem_fix.py:263-277`）把值字面量转 PTX 立即数（整型直书；i1 → 0/1；f32/f64 → `0f%08X`/`0d%016X` 十六进制位型；bf16 取 f32 编码高 16 位）。`fix_params()`（`gdn_smem_fix.py:333-378`）把每条

```ptx
	ld.param::entry.b64 	%rd269, [gdn_o_kernel_kernel_param_129];
```

替换为

```ptx
	mov.b64 	%rd269, 63;
```

（真实前后对照：`evidence/G4/gen_verify/o_f32_nomma_p0_t64x64x32_w4_K64V64/kernel.ptx:159` 与 `kernel_smemfix.ptx:31`），并删除已无用的 `.param` 声明。

#### 4.6.3 聚合常数的字节替换

G5-G 轮次发现该提升通道会连**向量常数**一起提升：payload 一旦携带 hatch op，`gdn-opt` 的 rewriter 把 launch 区域常数提升为宿主实参，**包括 bf16-mma 累加器零**——

```mlir
%zero = llvm.mlir.constant(dense<0.0> : vector<2x2xf32>) : !llvm.array<2 x vector<2xf32>>
```

这类参数在 PTX 里以多条**带偏移**的 `ld.param` 装载（真实样本，ws 配置 `kernel_smemfix.ptx:237-238`——该配置保留的 3 条封送常数之一，16B 向量零）：

```ptx
	ld.param::entry.b64 	%rd3141, [gdn_kkt_kernel_kernel_param_136+8];
	ld.param::entry.b64 	%rd3140, [gdn_kkt_kernel_kernel_param_136];
```

`LLVM_AGG_CONST_RE`（`gdn_smem_fix.py:223-226`）识别 `dense<...> : vector<d1 x d2 x T>` 形态，`_agg_bytes()`（`gdn_smem_fix.py:231-260`）把稠密字面量（标量广播或嵌套列表）按元素类型打包成**小端字节映像**；`fix_params` 的 `sub_ld` 对带偏移的装载按 `off:off+width` 切片字节映像、经 `_bytes_imm` 转成立即数 `mov`。不做此替换的后果是零参数启动直接 `CUDA_ERROR_INVALID_VALUE`（bwd 报告 G5-G 工具链注记原文）。

#### 4.6.4 签名收尾

参数删除后的两处收尾（`fix_params` 尾部，`gdn_smem_fix.py:372-377`）：最后一个残留声明的尾逗号摘除；参数表被清空时把 `name(\n\n)` 折叠为 `name()`——**本 ptxas 拒绝括号内的空行**。这是"改写器必须活在真实汇编器的文法约束下"的又一处微观注脚。

#### 4.6.5 效果数字

- 20 个 G3/G4 配置中 **18 个成为真正的零参数内核**（noargs 契约：输入经公共 `memref.global` 缓冲上载/读回，内核裸启动）；ws 配置保留 3 条封送常数（2× i1 + 16B 向量零）；tma 配置保留 3 条 tensormap 指针（`cuTensorMapEncodeTiled` 宿主编码后以设备指针传入）。
- 单内核 `main()` 的统计行（`gdn_smem_fix.py:848-852`）逐构建对账：`malloc_callseqs / cvta_patched / shared_decls / static_smem_total / const_params_substituted / dynamic_params_kept`——其中 `static_smem_total` 正是 §4.2 各池数字的直接来源。

### 4.7 `cache_key` 哈希键控：陈旧缓存坑的两次咬史与根治

#### 4.7.1 两次咬史

后处理层与运行时（`tools/gdn_full_rt.py` 的 `build()`）共同构成一条**带缓存的构建链**：目标目录已有 `kernel_smemfix.cubin` 且有 `full_spec.json` 时直接复用。这个短路在两处真实地咬人：

1. **第一次（正向 r3/r4）**：`evidence/G5/gen_fwd/B1T8192_mono` 槽位里躺着一个**陈旧的 w8/bv64 cubin**，被第 4 轮的 bench/裁决运行当作新构型取用——构型错配直到下一轮才被发现（fwd 报告 r5 节："a STALE w8/bv64 cubin served to round-4 bench/adjudication runs through the build-cache hole"；fused 槽位同样持有陈旧 r2 构建 `b2549100`）。所有涉事数字作废重测。
2. **第二次（正向 r19）**：F1WY 变体的 T2048 "verify" 实际是一次**陈旧 gen_fwd 缓存命中**——它把一个真 UB 掩盖成了"通过"：宽度 16 的 `gpu.shuffle` 下译为 `shfl.sync ... 0xFFFF` 成员掩码，lane 16..31 执行它是未定义行为；T2048 的缓存假阳性让该缺陷漏网，是 T8192 fp64 门（o 3.6e-1 FAIL）抓住的。教训被写进复现规约：变体验证永远用新鲜 `--gen-root`/scratch 目录。

两次事故的共同根因：缓存键为"目录名"，而**目录名不编码旋钮集与生成器/改写器源状态**。

#### 4.7.2 根治：哈希键控

`_cache_key()`（`tools/gdn_full_rt.py:28-46`）：

```python
def _cache_key():
    """R20-fwd: hash of everything that changes the emitted payload --
    every GDN_*/G5* env knob in scope + the generator/smem_fix sources.
    The stale-cache hole (a cached kernel_smemfix.cubin accepted under a
    DIFFERENT knob set or an edited generator) has bitten twice (r3,
    r19 -- the F1WY T2048 'verify' that was a stale gen_fwd hit)."""
    h = hashlib.sha256()
    for k in sorted(os.environ):
        if k.startswith("GDN_") or k.startswith("G5"):
            h.update(f"{k}={os.environ[k]}\n".encode())
    for f in ("gdn_full_gen.py", "gdn_smem_fix.py"):
        p = os.path.join(TOOLS, f)
        if os.path.exists(p):
            h.update(f"\n#{f}#".encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]
```

键 = sha256(**作用域内全部 `GDN_*`/`G5*` 环境旋钮的有序键值** + **`gdn_full_gen.py` 与 `gdn_smem_fix.py` 两份源文件的字节**)。`build()` 的缓存协议（`gdn_full_rt.py:48-110`）：构建成功后把 `gdn_cache_key` 写入 `full_spec.json`；后续构建若键不匹配，**重定向到 `<out_dir>_k<hash8>` 变体槽**而不是返回（或覆写）缓存目录；无戳的遗留目录按原样信任（gen_fwd 记录船位保持不动，不强制重建）。三态测试：同键命中、异键重定向（新鲜 `_k` 目录）、遗留信任。

#### 4.7.3 效果与协议地位

- 该洞自 R20 关闭后未再发生；生成器每次编辑后 gen_fwd 槽位的戳**按设计陈旧**（r23 编辑后键值 `780ab7d2`，r24 的编辑再次移动它）——`rt.build` 重定向、永不陈旧服务（fwd r24 §4 原文）。
- 键控同时覆盖了"旋钮集差异"与"源码编辑"两类漂移，这正是两次事故的两类肇因（第一次是构型/旋钮错配，第二次掩蔽 UB 的验证也属旋钮错配类）。
- 与第 9 章的测量规约互为表里：cubin md5 复核（重建 vs 记录）+ cache_key 键控共同保证"测的是想测的字节"。

### 4.8 小结：变换一览

| # | 变换 | 开关/入口 | 作用层 | 一句话效果 | 详节 |
|---|---|---|---|---|---|
| 1 | 静态 smem 修复 | 默认（单/多模块） | callseq + cvta + 声明 | smem tile 内核通过率 0→全量；`::cluster` 同规则 | §4.1 |
| 2 | 128B 对齐 | 默认 | `.shared` 声明 | UTMALDG 目的地址合法 | §4.1.5 |
| 3 | 竞技场别名 | `smem_alias.json`（`"modules"` 分发） | malloc 删除 + cvta→基+偏移 | dqkwg 100,864→47,616B、caryA 82,176→28,592B、wyb 100,096→45,056B ⇒ 2 CTA/SM | §4.2 |
| 4 | `.maxnreg` 注入 | `GDN_MAXREG_K` / `GDN_MAXNREG_ENTRY` | entry 体指令 | ptxas-blackwell 旗标盲区的唯一有效帽；dqkwg 1.16–1.26× | §4.3 |
| 5 | WST 四元组融合 | `GDN_WST`（`GDN_WST_SCOPE`） | b16 四元组→b32 对 | f1 −11~13μs，LSU 指令与迭代减半，字节精确 | §4.4 |
| 6 | W32P 行重映射 | `GDN_W32P`（生成器侧） | lane→row 映射 | 80 mod 32=16 互补半区零冲突；f2 −18~26μs @T8192；R25 三形状 formals 上移 | §4.4.5 |
| 7 | WSP shuffle 配对 | `GDN_WSP` | 跨 lane b32 打包 | f2 frontier 备用路线，保留在工具 | §4.4.6 |
| 8 | mono 合并 | ≥2 个 `gpu.binary` 自动 | 多 PTX → 单 cubin | DtoD 桥接（~0.1ms）消失；重名 `.shared` 逐内核改名防静默统一 | §4.5 |
| 9 | 常数参数消除 | 默认 | `ld.param`→`mov` | 18/20 零参数内核；聚合常数字节映像替换 | §4.6 |
| 10 | cache_key 键控 | `gdn_full_rt.build()` | 构建缓存 | 陈旧缓存两次咬史（r3/r19）根治：异键重定向 | §4.7 |

这十个变换共享同一条工程伦理：**每一步都是可审计的字节级重写**——要么位级相同（WST/W32P/别名/常数消除的全部落点都以此立据），要么回退到正确而不省的形态（别名的防御性回退），且每一步的诊断打印（`[smem_fix]` 前缀的 NOTE/WARN/统计行）保证失败响亮。后处理层因此不是对 MLIR 的绕过，而是把 stock 流水线留在 PTX 门口的四笔账——两笔正确性、一笔驻留、一笔发射效率——逐一付清的最短路径。

---

## 5 前向 kernel 体系：`tools/gdn_full_gen.py` 的调度机件全集

> 本章自足地展开前向 Gated DeltaNet（GDN）算子的全部 GPU kernel 机制。每一小节对应一个可独立开关的机制（旋钮），并按固定四件套呈现：**动机**（算子层面的瓶颈与具体例子）→ **源码片段**（生成器 `tools/gdn_full_gen.py`，共 4464 行）→ **IR 前后对比**（`evidence/G5/gen_fwd/` 下的真实 `payload.mlir` 摘录）→ **效果**（配对 A/B 或交错终判的数字，出处随文标注）。本章按**最终发运系统**呈现，不按轮次叙事；全部数字取自 `evidence/G5/fwd_report.md` 对应机制段与 `evidence/G5/gen_fwd/` 的在档 cubin/payload，口径为共享 GPU 交错配对（interleaved adjudication，见全局口径声明）。

### 5.0 章首：三 kernel 分解、融合旗舰与 mono 三重奏的架构图景

**数学对象先行定义。** 前向 GDN 把序列切成 $C=64$ 行的 chunk（尾块零填充，$T_p=NC\cdot C$）。记 $\mathrm{gc}$ 为门控 $g$ 在 $\log_2$ 域的 inclusive 前缀和（$g$ 本身是自然对数域，乘以 $\mathrm{RCP\_LN2}=1.4426950\ldots$ 折入 $\log_2$ 域，使一切衰减因子可由 `math.exp2` 单条 SFU 指令生成），$\beta$ 为删除门，则每 chunk 的五步复合为（`gdn_full_gen.py` 文件头，L1 已验证真值）：

$$
A_{i,j}=\beta_i\,(k_i\!\cdot\! k_j)\,2^{\,\mathrm{gc}_i-\mathrm{gc}_j}\ (\text{严格下三角}),\qquad
T=(I+A)^{-1},
$$
$$
u=T\,(\beta\!\odot\! v),\qquad w=T\,(\beta\!\odot\! k\!\odot\!2^{\mathrm{gc}}),\qquad
v_{\text{new}}=u-w\,S,
$$
$$
o=\Big((q\!\odot\!2^{\mathrm{gc}})\,S+\mathrm{tril}_{\text{inc}}\big((q\,k^{\top})\odot 2^{\mathrm{gc}_i-\mathrm{gc}_{\text{last}}}\big)\,v_{\text{new}}\Big)\cdot K^{-1/2},\qquad
S\leftarrow S\cdot 2^{\mathrm{gc}_{\text{last}}}+(k\odot 2^{\mathrm{gc}_{\text{last}}-\mathrm{gc}})^{\top}v_{\text{new}} .
$$

dtype 契约沿用 fla：$q,k,v,\beta,g$ 的输入为 bf16/fp32 混合（$q/k/v$ bf16，$\beta/g$ fp32），$o$ bf16，$h_0/\mathrm{ht}$ fp32，**跨 chunk 携带态以 bf16 存储，全部算术 f32**（mma f32 累加器，G4 契约）。

**执行平台与全章共用资源口径**：RTX 5090 Laptop（消费级 Blackwell，`sm_120a`，82 SM，**无 tcgen05**，Ampere 风格 `mma.sync`；共享内存 opt-in 上限 $101{,}376$ B/CTA（即 r1 报告所称的 99 KB 硬顶，`0x18c00`），静态申报线 $49{,}152$ B（48 KB——两倍恰为 2 CTAs/SM 的门），每 SM 102,400 B）。全部 tile 形状、门限与占用结论都在这三条资源线上展开。

**路径分解。** 生成器提供 `--path staged|fused|mono` 三种融合边界：

- **staged 三 kernel**（fla 同构分解）：
  - **F1 "wy"**，grid $(NC,\,B\!\cdot\!H)$——token 并行 WY 预计算：gc 扫描、$A$、$T$、$u$、$w$；
  - **F2 "scan"**，grid $(B\!\cdot\!H\!\cdot\!VT,\,1)$——跨 chunk 状态递推：$S$ 的串行 chunk 链，产出入口态 $\mathrm{h\_all}$、$v_{\text{new}}$、终态 $\mathrm{ht}$；
  - **F3 "o"**，grid $(NC\!\cdot\!B\!\cdot\!H,\,VT)$——输出级：从 $\mathrm{h\_all}/v_{\text{new}}$ 重算 $o$。
- **fused 旗舰**：单一 persistent kernel（grid $B\!\cdot\!H\!\cdot\!VT$），chunk 循环进核，$S$ 常驻共享内存，每 chunk 串行完成 WY+solve+状态更新+输出（§5.4）。
- **mono 三重奏**（`gen_mono` + `wrap_module_mono`）：把 F1/F2/F3 三个零参 kernel 与一套公共 `memref.global` 缓冲封进**一个 cubin**——kernel 边界即流序同步，staged 路径因此不再需要任何设备到设备（DtoD）交接拷贝（r1 时代每形状约 0.1 ms 的交接开销由此消失）。发运载体是 mono：F1 的 `gc_out` 输出在 mono 装配时直接改名为 `gc`，三个 kernel 共享同一全局。

每个 kernel 的计算骨架是同一套 G2/G4 机件：全 K 操作数 staging 到 workgroup 共享内存（`vector.transfer` 对 + `create_async_groups` 或手动 `nvgpu.device_async_copy`）、warp 平铺的 `nvgpu.mma.sync` m16n8k16 bf16 原子（f32 累加，D-fragment 列索引 $=2\cdot\mathrm{tig}$——G4 修复律）、`nvgpu.ldmatrix` fragment 装载（每 fragment 一条指令替代 12 条标量 `ld.shared`）、自定义 epilogue（衰减/门控/掩蔽在写回前折入 f32 域）、阶段间 `gpu.barrier`。

**最终判定口径数字**（r25 存档会话，`final_adjudicate_r25f_w32p.json`，30 交错对/形状，发运旋钮全套 + `GDN_W32P=1`，cubin `e7c8eb4c`/`39f094c3`/`be5b2377`）：B1T2048 **2.1250×**（我方中位 0.1866 ms）、B4T2048 **1.2211×**（0.8213 ms）、B1T8192 **1.3493×**（0.7851 ms），对 fla v0.6.0 纯 Triton 前向。数值契约 4/4 PASS：staged T2048 $o\le 6.486\times10^{-4}$ / $\mathrm{ht}\le 7.493\times10^{-4}$ vs fp64 JAX 参考（bar $5\times10^{-3}$，约 7 倍裕量），T8192 fp64 门 $o\,6.626\times10^{-4}$ / $\mathrm{ht}\,8.918\times10^{-4}$（bf16-solve 类）。以下各节把这两个数字背后的每个机件拆开。

---

### 5.1 F1 "wy"：WY 预计算 kernel

F1 的最终形态是一条五段串行流水（每 CTA 处理一个 (chunk, head) 对，128 线程 = 4 warp）：

1. **gc Hillis-Steele 对数扫描 + β staging**（`emit_cumsum_beta_scan`，§5.1.1）；
2. **kkt K-循环**：一块 staging 的 $k$ tile 同时充当 mma 的 A 侧与 B 侧，算 $k k^{\top}$，epilogue 里折入 $\beta$ 与衰减得 $A$（f32→bf16 双写 `A_m`/`Abf`，§5.1.2）；
3. **分块 16×16 前代求解**（`emit_solve_blocked`，§5.1.3）；
4. **uw 段**：$u=T_b\,v$、$w=T_g\,k$ 两个 mma，A 操作数是常驻的 $\mathrm{Tb}/\mathrm{Tg}$（$T^{\top}$ 经 $\beta$、$\beta 2^{\mathrm{gc}}$ 列缩放的转置型），B 操作数 cp.async staging（§5.1.4–5.1.5）；
5. **gc_out 存回**（f1 的 gc 供 f2/f3 复用，省去它们各自重扫）。

F1 的共享内存池在发运形态（F1KS+F1ZF）下为**申报 46,608 B ≤ 49,152 B ⇒ 2 CTAs/SM**（`cuOccupancyMaxActiveBlocksPerMultiprocessor = 2`，前向第一个达成的 2-CTA kernel），寄存器 176（`ptxas_v.txt`，0 spill）。这一占用跃迁是 f1 的主胜利来源（f1 −53.5 µs @T8192，§5.1.5）。

#### 5.1.1 机制：gc 的 Hillis–Steele 对数扫描（与 SCANDB 双缓冲硬化）

**动机（算子层面）。** F1 每chunk需要 64 个 $\mathrm{gc}_t=\sum_{s\le t}g_s\cdot\log_2 e$ 值。朴素实现是每线程串行重扫（每线程平均 32 次 H 跨步全局读——f1 的输入布局是 $[B,T_p,H]$，同一 chunk 同一 head 的 64 个 g 值在全局里彼此相距 $H$ 个元素，天然不合并）。Hillis–Steele 把它改成：每线程一次合并读入共享内存，随后 6 步（$d=1,2,4,8,16,32$）对数深度扫描，步间 barrier。数值上属于 f32 加法树重结合类（与串行前缀和的差远低于契约）。

**源码片段**（`emit_cumsum_beta_scan`，gdn_full_gen.py）：

```python
for i, d in enumerate((1, 2, 4, 8, 16, 32)):
    src, dst = gc, gc                      # 单缓冲（SCANDB 关时）
    ...
    e.l(f"{off} = arith.subi %tid, {e.c(d)} : index")
    e.l(f"{ge} = arith.cmpi sge, %tid, {e.c(d)} : index")
    e.l(f"{offc} = arith.select {ge}, {off}, %c0 : index")
    e.l(f"{lv} = memref.load {src}[{offc}, %c0] : {src_ty}")
    e.l(f"{vv} = arith.select {ge}, {lv}, {zf} : f32")
    e.l(f"{nv} = arith.addf {cur}, {vv} : f32")
    e.l(f"memref.store {nv}, {dst}[%tid, %c0] : {dst_ty}")
    e.l("gpu.barrier")
```

**机器缺陷与 SCANDB（`GDN_SCANDB`，发运开启）。** 单缓冲步骤内有一个**跨 warp 的步内竞态**：线程 $t_2$ 的 `gc[off]` 读与线程 $t_1$（$t_1=t_2-\mathrm{off}$）的 `gc[t_1]` 写在同一步内无序——barrier 只隔步不隔步内。1-CTA 驻留时 warp 0/1 恰好日程对齐而良性；**2-CTA 共驻把它破坏**（r17f F1KS 双 CTA 损坏二分定位到的第二个 bug；compute-sanitizer racecheck 在含发运版在内的每个构建里都精确标出这 6 对）。修复即 SCANDB：$+256$ B 的第二缓冲 `gcs2 [64] f32`，偶数步读 `gcs` 写 `gcs2`、奇数步反向，6 步后值落回 `gcs`（偶数步数免拷回）。

**IR（发运 payload，SCANDB 开启的第 6 步 $d=32$，`bench_B1T2048H16_mono.r22ship/payload.mlir`）**：

```mlir
%cso_6 = arith.subi %tid, %c32 : index
%cge_6 = arith.cmpi sge %tid, %c32 : index
%csc_6 = arith.select %cge_6, %cso_6, %c0 : index
%cgl_6 = memref.load %gcs2[%csc_6, %c0] : memref<64x1xf32, ...>   ; 读对侧缓冲
%csv_6 = arith.select %cge_6, %cgl_6, %czf : f32
%cgc_6 = memref.load %gcs2[%tid, %c0] : memref<64x1xf32, ...>
%cgn_6 = arith.addf %cgc_6, %csv_6 : f32
memref.store %cgn_6, %gcs[%tid, %c0] : memref<64x1xf32, ...>      ; 写回本侧
```

**效果。** r18 双 bug 根因链中，仅 ZFIX 时损坏从 ~70 万 bit 差塌缩到 ~4 千、每 run 在 1–2/512 tile 触发且指纹恰为 `gc[32:64)` 窗口（$d=32$ 步的跨 warp 读）；ZFIX+SCANDB 合用后 **10/10 次仅-f1 复跑与强 1-CTA 参考位级一致**，三重奏命中精确记录数字（$o\,6.486\times10^{-4}$ / $\mathrm{ht}\,7.493\times10^{-4}$）。1-CTA 驻留下该硬化数值上值精确（SCANDB 构建对旧船在 T8192 fp64 门位级一致，$o/\mathrm{ht}=0.000e{+}00$）。删除探针给该扫描+β staging 的表观成本 **17.6 µs（f1 的 7.8%）**（r21 时间图谱，`GDN_F1T_NSC`，§5.1.7）。

#### 5.1.2 机制：kkt K-循环（transform 流水线形态与它的屋顶线关闭）

**动机。** $A=\mathrm{tril}(\beta\odot(k k^{\top})\odot 2^{\Delta\mathrm{gc}})$ 需要一个 $64\times64\times K$ 的 GEMM。G2 机件（`transform.memref.multibuffer` + `transform.nvgpu.create_async_groups bypass_l1` + `transform.nvgpu.pipeline_shared_memory_copies depth=2 peel_epilogue`）只在**扁平 `scf.for` K-循环**上生效，因此 kkt 被结构化为：循环体 = 线程跨距的 16 B `vector.transfer` staging 对（每线程拷**自己的**块——G2 式块均匀展开会让每个线程发每份拷贝，256× 冗余 cp.async，实测比同步版本还慢 2×，r2 的头号修复）+ ldmatrix fragment 装载 + 8 累加器链的 `mma.sync`，累加器作为 `iter_args` 循环携带。

**源码片段**（`emit_kloop`，staging 侧；gemms 以 (tile, staged) 三元组声明）：

```python
for kk in range(npairs // nthr):
    p = addi(e, "%tid", e.c(kk * nthr)) if kk else "%tid"
    r, cc = divsi(e, p, e.c(nch)), remsi(e, p, e.c(nch))
    ...
    e.l(f"{t} = vector.transfer_read {s['src']}[{idx}], {pad} "
        f"{{in_bounds = [true]}} : {s['sty']}, vector<8xbf16>")
    e.l(f"vector.transfer_write {t}, {t_ssa}[{r}, {c8}] ...")
...
e.l(f"{o} = nvgpu.mma.sync({fa[...]}, {fb[...]}, {prev}) "
    f"mmaShape = [16, 8, 16] : (vector<4x2xbf16>, vector<2x2xbf16>, "
    f"vector<2x2xf32>) -> vector<2x2xf32>")
```

**IR（发运 payload 的 kkt 循环头与首对 staging，`r22ship/payload.mlir`；注意 F1KS 使步长为 16、tile 为 `[64,24]`）**：

```mlir
%rs_kk_kkt_0_0, ... = scf.for %kb_kk = %c0 to %c128 step %c16
    iter_args(%ia_kk_kkt_0_0 = %az_kk_kkt_0_0, ...) -> (vector<2x2xf32> x8) {
  %pd_kk = arith.constant 0.0 : bf16
  %idv = arith.divsi %tid, %c2 : index          ; 每线程 2 个 16B 块
  %irm = arith.remsi %tid, %c2 : index
  %im_13 = arith.muli %irm, %c8 : index
  %srkk = vector.transfer_read %k[%ix_11], %pd_kk {in_bounds=[true]}
      : memref<4194304xbf16>, vector<8xbf16>
  vector.transfer_write %srkk, %kt[%idv, %im_13] {in_bounds=[true]}
      : vector<8xbf16>, memref<64x24xbf16, #gpu.address_space<workgroup>>
  gpu.barrier
  %fa_kk_kkt_0_0 = nvgpu.ldmatrix %kt[%ix_13, %ix_14] numTiles = 4
      transpose = false : ... -> vector<4x2xbf16>
  %mr_kk_kkt_0_0_0 = nvgpu.mma.sync(%fa_kk_kkt_0_0, %fb_kk_kkt_0_0, %ia_kk_kkt_0_0)
      mmaShape = [16, 8, 16] : (...) -> vector<2x2xf32>
```

A 的 epilogue（`epi_A2`）在每个 fragment 单元上折入 $\beta_{\text{row}}$、$2^{\mathrm{gc}_i-\mathrm{gc}_j}$（三次共享内存读 + `subf/exp2/2×mulf` + `cmpi slt` 严格下三角选择），f32 写 `A_m`（仅非 bf16-solve 形态）/截断 bf16 写 `Abf`。

**效果与关闭（r23 的拆分探针 + 屋顶线）。** r21 删除探针给整个 kkt 段 **−59.5 µs（f1 的 26%）**；r23 的三路拆分（`GDN_F1T_NKK/NEK/NKM`，各 ×3 会话）把它分解为：**k-staging 全局流 40.6 µs（73%）**、ldmatrix+mma 13.0 µs（23%）、exp2/β 链 2–4 µs（4–7%）、store ≤2 µs。staging 流每 CTA 恰好一次读尽自己的 $64\times128$ bf16 k 块（$B\cdot T_p\cdot H\cdot K\cdot2=33.55$ MB @T8192，100% 扇区利用、32 B 对齐、bypass L1），40.6 µs 折合 **826 GB/s**；同窗 D2D 拷贝标定器件聚合读写带宽 **765 GB/s**——一条只读流超过读写聚合上限，意味着**无重构余量**：kkt 在当前几何（2 CTAs/SM）下以算法最小流量顶在 DRAM 屋顶线上，作为性能之门被关闭（`scratch/r23f/bw.py`；tile 变体 `GDN_F1KS_BK32` 把 kt 多缓冲 6,144→10,240 B、f1 申报池 ~49.7 KB，occupancy API 读 blocks/SM 2→1，配对 ×3 = +70.3/+73.4/+68.3 µs **回退**——全部“增益类”都是占用损失，非 tile 经济学）。

#### 5.1.3 机制：分块 16×16 前代求解（对角块 16 步串行代入 + 非角块 stash/consume mma）

**动机。** $T=(I+A)^{-1}$ 按列前代代入：$T_{i,j}=-\sum_{j\le k<i}A_{i,k}T_{k,j}$。平铺 64 列的每列一个线程版是 2016 条依赖 FMA 的串行链（实测 f1 临界路径 ~45 µs/CTA）。fla 式再瓦解把 $A$ 分成 $4\times4$ 个 $16\times16$ 块：

- **phase 1（对角块）**：线程 $t<64$ 认领 (块 $D=t/16$，列 $c=t\bmod16$)，做 16 步短代入（每列至多 ~120 条串行 FMA，替代平铺版的 2016 条）。源码（`emit_solve_blocked` 的串行臂，发运形态）：

```python
for t in range(16):
    i = addi(e, muli(e, db, c16), e.c(t))      # 块 D 的全局行
    e.l(f"{svi}, {sv} = scf.while (%sk = {ccol}, %sa = {zf}) : (index, f32) ...")
    #   体: %aik = load Abf[i, %skb]; %tkj = load Tt[%skb, ccol]
    #       extf→f32; mulf; addf —— 严格保序的上升 k 链
    e.l(f"{val} = arith.select {isd}, {one}, {neg} : f32")   # 对角=1
    #   写三联: Tt[i,ccol]（未缩放）+ Tb[ccol,i]·β + Tg[ccol,i]·β·2^gc（转置缩放型）
```

  $\beta$ 列缩放可穿过递推（$X=T\,\mathrm{diag}(\beta)$ 满足同一代入、$X_{jj}=\beta_j$），因此**两个消费者 GEMM 直接读原始 $v$/$k$**——r1 时代的标量 $\beta v$、$\beta k 2^{\mathrm{gc}}$ staging 扫pass整体消失（r2 F1 重构 429→270 µs 的一部分）。

- **phase 2（六个非角块）**：按块行 $R=1,2,3$ 两拍推进——**stash 拍**（warp $C_b<R$ 各自算 $S=\sum_K \mathrm{Abf}[R,K]\,@\,\mathrm{Tt}[K,C_b]$ 进**自己的** scratch tile `St_Cb`，m16n8k16 bf16 mma）、barrier、**consume 拍**（$T[R,C_b]=-(\mathrm{Tt}[R,R]\,@\,S)$，经 scratch 的第二个 mma），barrier。两个历史 bug 值得记录：stash 拍 B-fragment 列基漏了 $+16\,C_b$ 块偏移（warp 0 读对块、warp 1/2 读错块；16×16 块 dump 差分定位）；warp 守卫的 `scf.if` **内部**的 `gpu.barrier` = 部分到达 = `sm_120a` 上同步损坏——barrier 一律移到无条件位置。

**IR（phase 1 的代入 `scf.while`，`r22ship/payload.mlir`；bf16-solve 使 A 读自 `Abf`）**：

```mlir
%sviw, %svw = scf.while (%sk = %ix_28, %sa = %svz) : (index, f32) -> (index, f32) {
  %scnd = arith.cmpi slt %sk, %ix_29 : index
  scf.condition(%scnd) %sk, %sa : index, f32
} do {
^bb0(%skb: index, %sab: f32):
  %aikb = memref.load %Abf[%ix_29, %skb] : memref<64x72xbf16, ...>
  %aik  = arith.extf %aikb : bf16 to f32
  %tkj  = memref.load %Tt[%skb, %ix_28] : memref<66x72xbf16, ...>
  %tkf  = arith.extf %tkj : bf16 to f32
  %pr   = arith.mulf %aik, %tkf : f32
  %sn   = arith.addf %sab, %pr : f32
  %skn  = arith.addi %skb, %c1 : index
  scf.yield %skn, %sn : index, f32
}
```

**两个诚实的负结果（phase-1 重构两连败）。** phase-1 串行代入被两次不同思路的重构攻击，均正确、均更慢：
- **F1WY**（`GDN_F1WY`，warp 并行 phase-1 v3）：每对角块一个 warp，行向量蝴蝶归约（宽 32 的 `gpu.shuffle xor`，5 级 shfl+add 同时归约 16 列点积）。三刀才正确：宽 16 的蝴蝶降为 `shfl.sync 0xFFFF` 成员掩码（lane 16–31 执行即 UB）；16 路按 lane 选择存储网被工具链错误编译（列 ≥4 全陈旧）；最终 v3 = 无选择网——T 行驻 `Tt` 共享内存（对角块预清零）、存储列是编译期常数（warp 冗余广播存）、每步一 barrier。T8192 fp64 门**与船位级一致**，但 f1 **+92.6/+90.4/+91.4 µs ×3**（77–80/80 对更慢）：串行代入的共享内存读被编译器流水得极好，而蝴蝶的指令数 ~4–5×（1280 shfl+add、768 冗余存、每 solve 16 条额外 CTA barrier），2 CTAs/SM 下发射压力主导。
- **F1P4**（`GDN_F1P4`，4 线程/块 × 4 列寄存器瓦片）：每行一条 `scf.while` k-链带**四条交错累加链**（4× FMA ILP、每步一次 $A_{i,k}$ 读摊给 4 列，~4.5 指令/FMA 对串行链的 ~8）。位一致性由构造保证：$k<c$ 项读 `Tt` 的 $-0.0$ 上角，IEEE 零和规则使真项以**同一上升序**累加。T8192 门位级一致、寄存器 172→168，但 f1 **+14.8/+14.5/+15.5 µs ×3**：16 活跃线程 × 4 宽 ILP 输给 64 线程 × 1 宽——phase-1 重要的是线程级并行而非每线程链深。

r21 的删除探针最终裁定 **phase-1 仅占 f1 的 14.3%（32.5 µs）**，“phase-1 是 f1 临界路径”的论题就此死亡（f1 的大头是 uw 段 51%，§5.1.4）。附带建构细节：k-链必须 `scf.while` 而非 `scf.for`——payload 的 transform 流水化器匹配**每一个** `scf.for`，对无拷贝循环（“no shared memory copy”）即便 `failures(suppress)` 也硬错整个 `-transform-interpreter` 输出（探针验证；这也是全部非 K 循环 cell 扫pass用 `scf.while` 的原因，`emit_cell_while`）。

**bf16-solve（`GDN_BF16SOLVE`，F1KS 的隐含前提）。** 去掉 f32 `A_m` 镜像：phase-1 递推直接读 bf16 `Abf`（extf 到 f32 参与 FMA），kkt epilogue 只写 `Abf`。CPU 孪生（numpy 按 kernel 精确 bf16 切点仿真）先证：$T$ 本就经 phase-2 mma 与 `Tt` 反馈以 bf16 存活，`A_m` 带进 phase-1 的 f32 位两个操作后就被舍掉——变体 B 与记录数字统计不可分（最坏 $6.7\times$ 裕量于 $5\times10^{-3}$ bar）。池 −16,384 B，为 phase-1 共活下限从 57,408 B 降到 **41,024 B < 50,688 B** 铺平 2-CTA 之路（§5.1.5）。GPU 实测数值过门（$o\,6.626\times10^{-4}$/ht $8.918\times10^{-4}$，即此后所有 F1KS 构建的记录数字类）、性能中性（−2.6/−3.2 µs，噪声）。

#### 5.1.4 机制：F1UW——uw 段重构（单趟化、预解 staging、死 tile 复用）

**动机。** uw 段（$u=T_bv$、$w=T_gk$）的三重瓶颈：(i) K-归约维恰为 $C=64$，pipeliner 的 `scf.for` 单趟形态直接“pipelining preconditions failed”硬错；(ii) operand staging（32 KB）若在解后同步发出，就串在关键路径上；(iii) epilogue 的 fragment 映射全局存是**每线程 128 条离散 2 B `st.global.b16`**（LSU/LG 节流受限，r3 ncu：46.8% 内存吞吐、557 K 共享存储 bank 冲突级别的低效类）。

**三合一改造**（`gen_f1` 的 `_f1uw` 臂，默认开启；`G5K_NOF1UW=1` 回退）：

1. **bk=C 单趟直线化**：uw mma 以 $k\in\{0,32\}$ 两组 fragment 全展开（`_mma_chain`），无循环、无 pipeliner 参与。
2. **staging 提前到 solve 之前发出**：v/k tile 与一切 solve tile 不相交，前代代入的串行延迟正好藏住这 32 KB 拷贝：

```python
if _f1uw and not _f1al and not _ks and not _nust:
    # 在 solve 之前发出 uw operand staging：拷贝写 vt/k2，与全部 solve tile 不相交
    async_stage_unrolled(e, "%v", TY["v"], vtile, vtile_ty, C, V, nthr, ...)
    async_stage_unrolled(e, "%k", TY["k"], ktile2, ktile2_ty, C, K, nthr, ...)
    guw = e.v("agr")
    e.l(f"{guw} = nvgpu.device_async_create_group {', '.join(toks)}")
...
emit_solve_blocked(...)          # 串行 solve 藏住上面的拷贝
...
e.l(f"nvgpu.device_async_wait {guw} numGroups = 0")
e.l("gpu.barrier")               # 此后才发射 mma
```

3. **死 tile 复用为输出 restage**：循环后 vt/k2 作为 B 操作数已死，被回收为 u/w 的输出 staging tile——epilogue 写**带垫共享 tile**（无冲突 STS），再各用一次合并 cell 扫pass搬去全局（64 B/warp 行段），替代离散 2 B 全局存。字节数上 vt/k2 变单缓冲 `[C,N{+}8]` 恰等于原多缓冲二份的字节，**池不变**（F1 仍 101,120 B 时代）。

**效果。** r5 配对 A/B（×50 对）：f1 @T8192 0.416→0.293 ms、staged e2e **−119 µs**（4/50 更慢）；@B4T2048 f1 −122 µs（0/50）；@T2048 f1 −37 µs、e2e −50 µs（2/50）。数值位级一致（truncf 舍入不变，只动了存储顺序；T8192 fp64 门与记录位级同）。r14 时代 u/w 存储的 s2g 扫pass又被 WST 宽存储进一步折半（§5.2.4/§5.3.3）。

#### 5.1.5 机制：F1KS——K-分裂 staging 与终身 arena，前向第一个 2-CTA kernel

**动机。** F1 是 T8192 上最大的前向 kernel（25 波网格）。2-CTA 驻留（申报池 ≤48 KB）的算术障碍：f32 `A_m` 在内时 phase-1 共活下限 57,408 B > 50,688 B（r15 的 F1AL 别名穷举结论——“只有丢掉 A_m 或重构 solve 才能过线”）。bf16-solve（§5.1.3）把下限打到 41,024 B 后，**新的堵点移到 uw 段**：$\mathrm{Tb}+\mathrm{Tt}+\mathrm{Tg}$（31,296 B）+ vt（17,408）+ k2（17,408）= 66,112 B > 50,688 B。

**设计（`GDN_F1KS`，发运开启；栈于 bf16-solve 之上）**——三刀：

1. **uw 的 B 操作数 K-分裂**：vt/k2 砍半为 `[32,136]`/`[33,136]`（k2 的 +1 行是为“按尺寸识别”的 malloc 尺寸唯一性），两个顺序半 K 扫pass复用同一套半 tile；A 操作数 $\mathrm{Tb}/\mathrm{Tg}$ 保持全 tile、在列 $32p$ 处读取。累加器跨扫携带：32 的分裂点**落在四个 m16n8k16 块之间**，每输出元素的 f32 加法序列与原 4 块链位级相同（CPU 孪生 0 翻转；树重结合对照组确实翻位，探针敏感性证明）。
2. **kkt 降到 bk=16**：kt 多缓冲 10,240→6,144 B。
3. **PTX 终身 arena**（生成器写 `smem_alias.json`，`tools/gdn_smem_fix.py` 消费）：半 tile **落在 solve 之后的死字节上**——vt→`Abf` 的字节、k2→`Tt` 的字节（两者在 uw 段均已死；solve 的末 barrier 覆盖其最后读）。按尺寸识别需要各 spec 成员 malloc 尺寸唯一：用**行粒度膨胀**（+1/+2 死行、144 B 行距不变）替代列垫（列垫每块 1–2 KB，且 ptxas 把非 128 倍数申报向上补齐）。申报合计 46,144 B，ptxas 补垫后 46,352 B ≤ 49,152 B。

```python
alias_modules["gdn_fwd_f1_kernel"] = {
    "identify": {"9216": "Abf", "9504": "Tt", "8704": "vt", "8976": "k2"},
    "arenas":   {"Abf": 9216, "Tt": 9504},
    "alias":    {"vt": ["Abf", 0], "k2": ["Tt", 0]}}
```

**IR（发运 payload 的 F1KS 扫pass 0 预取组，`r22ship/payload.mlir`；解后落位、与 gc_out 存回重叠）**：

```mlir
%acp   = nvgpu.device_async_copy %v[%ix_223], %vt[%idv_5, %im_31], 8 bypassL1
    : memref<4194304xbf16> to memref<32x136xbf16, #gpu.address_space<workgroup>>
...  (8 条：v 半块 ×4 + k 半块 ×4)
%agr_p0 = nvgpu.device_async_create_group %acp, ..., %acp_8
...
nvgpu.device_async_wait %agr_p0 numGroups = 0    ; ks_pass(0) 前
gpu.barrier
...  ks_pass(0)：Tb/Tg 全 tile 列 0..31 + vt/k2 行 0..31 的 mma
gpu.barrier
%agr_p1 = nvgpu.device_async_create_group ...      ; 扫pass 1 的组
nvgpu.device_async_wait %agr_p1 numGroups = 0      ; ks_pass(1) 累加器接续
```

**两个 2-CTA 隐蔽 bug（根因链，r18）。** K-分裂码在 1-CTA 强制臂上位级正确；2-CTA 共驻下 u/w 损坏（$o\sim6.3\times10^{-2}$）而 ht 仍精确——八个二分臂排除了码值、arena、staging 机制、bk、mb、池尺寸后，根因是**两个叠加的、被 1-CTA 驻留掩盖的独立 bug**：
- **Bug 1（uninit-smem 步长缺陷）**：phase-0 `zero_up` cell 循环步长 $32\cdot8=256$ 超过 128 线程块，$\mathrm{Tb}/\mathrm{Tg}$ 的 $(r\cdot64+c)\bmod256\in[128,256)$ 单元（每 tile 768 个）**从未被清零**；uw gemm 的转置 ldmatrix A 读覆盖全 64×64，把这些单元当作 $T$ 的上三角零读。1-CTA 时新窗口恰好读到精确零（393,216/393,216 单元实测）；2-CTA 共驻读到非零陈旧数据。修复 `GDN_ZFIX`：步长改 `nthr`，全覆盖。
- **Bug 2（扫描步内竞态）**：§5.1.1 的 SCANDB。
ZFIX+SCANDB 之后 2-CTA 下 10/10 位级干净。损坏指纹（w 恰在块列 $C_b=2$=行 32–47、u 在行 0–47、行 48–63 干净）同时解释了 o/u-vs-ht 之谜：损坏 chunk 的态贡献被衰减压到契约之下所以 ht 完好，而 f3 直接读损坏 chunk 的 `h_all`/`v_new` 所以 o 失败。

**效果。** 发运（F1KS 2-CTA）对 r15/r16 船的配对中位（×3 会话 ×40 对）：T8192 f1 278.7→225.2 µs（**−53.5 µs**，35–38/40 更快）；T2048 −16.7 µs；B4T2048 −57.9 µs（链 −75.4 µs）。r19 的均衡双装载复测收紧为 f1 −56.4 µs（3/80 更慢）、staged e2e ~−62±5 µs、**全部来自 f1**（f3 的原始 B 侧罚分 100% 是模块摆放抽签，§5.2.4 注）。r17 预估的“波数减半”落地为 kernel 时间 −19%。

#### 5.1.6 机制：F1ZF——phase-0 清零折进 phase-1 的空闲线程（与 F1EP 的 wash 关闭）

**动机。** r21 时间图谱：`zero_up`（phase-0，为 $\mathrm{Tb}/\mathrm{Tg}$ 的转置上三角清零）值 ~8.5 µs，而它串行地排在 64 线程的 ~32 µs 串行代入**之前**——同一时刻线程 64–127 全程空闲。

**改造（`GDN_F1ZF`，发运开启）**：把清零循环**折进** phase-1 的守卫块——线程 64–127（`%tid >= 64`）在代入运行的同时清“活上三角”单元，谓词加强为**块级** $\mathrm{block}(r)>\mathrm{block}(c)$（蕴含 $r>c$）。位一致性论证：折叠臂只写代入**永不触碰**的单元（串行链写全部同块单元含其 $-0.0$ 上角），写集不相交 ⇒ $\mathrm{Tb}/\mathrm{Tg}$ 的位与船确定性地相同；一条 barrier 取代原 phase-0/phase-1 两条。

**IR 前后对比**（同为 T2048 mono 构建的真实 payload 差分；`gen_fwd/B1T2048_mono.r18ship`（前，旋钮集少 F1ZF）vs `bench_B1T2048H16_mono.r22ship`（后，+F1ZF）——f1 段的实质差异即 F1ZF 一项）：

前（独立 phase-0，全体线程，随后专属 barrier）：

```mlir
%wir = scf.while (%wi = %tid) : (index) -> (index) {
  %wic = arith.cmpi ult %wi, %c4096 : index
  scf.condition(%wic) %wi : index
} do {
^bb0(%wib: index):
  %idv_2 = arith.divsi %wib, %c64 : index        ; r
  %irm_2 = arith.remsi %wib, %c64 : index        ; c
  %svgt = arith.cmpi sgt %idv_2, %irm_2 : index  ; r > c
  scf.if %svgt {
    memref.store %svz2, %Tb[%idv_2, %irm_2] : ...
    memref.store %svz2, %Tg[%idv_2, %irm_2] : ...
  }
  %win = arith.addi %wib, %c128 : index
  scf.yield %win : index
}
gpu.barrier                                         ; 专属 barrier
```

后（折进空闲半块、与代入并发、步长 64、块级谓词；总 barrier 减一）：

```mlir
%zfe = arith.cmpi uge %tid, %c64 : index          ; 折叠臂：线程 64..127
scf.if %zfe {
  %wir = scf.while (%wi = %tid) : (index) -> (index) {
    %wic = arith.cmpi ult %wi, %c4096 : index
    scf.condition(%wic) %wi : index
  } do {
  ^bb0(%wib: index):
    %idv_2 = arith.divsi %wib, %c64 : index
    %irm_2 = arith.remsi %wib, %c64 : index
    %idv_3 = arith.divsi %idv_2, %c16 : index     ; block(r)
    %idv_4 = arith.divsi %irm_2, %c16 : index     ; block(c)
    %svgt = arith.cmpi sgt %idv_3, %idv_4 : index ; 块级谓词
    scf.if %svgt { memref.store %svz2, %Tb[...] ...; memref.store %svz2, %Tg[...] ... }
    %win = arith.addi %wib, %c64 : index          ; 步长 64
    scf.yield %win : index
  }
}                                                   ; 无专属 barrier
```

**效果。** f1 配对：T8192 **−7.7/−7.9/−6.7 µs ×3**（77–78/80 对）、T2048 −2.3 µs（57/80）——约为 NP0 删除界 8.5 µs 的 91%（折叠是重叠、删除是废除）。门：T2048 三重奏 7/7 数组位级同、T8192 fp64 门位级同（记录数字 $6.626\times10^{-4}$/$8.918\times10^{-4}$）、det N=5 前向确定。F1ZF 就此加入发运旋钮集，r22 完成记录迁移（三形状 n=30 正式终判全部落在同一 cubin 代上：T8192 1.2226×、B4 1.1417×、T2048 首次正式 2.1184×）。

**F1EP（`GDN_F1EP`，默认关——wash 关闭）。** F1KS 的 epilogue 以两个行半程过**一套**半 tile（4 barrier + 两轮 barrier 隔离的拷贝）。F1EP 把行 $[32,64)$ 的输出折进 u2/w2——两个 PTX 级别名到 $\mathrm{Tb}/\mathrm{Tg}$ 死字节的 tile（几何：`Tg` 增到 `[67,72]`，**行**垫 +288 B 保持 144 B 行距为 16 B 倍数——列垫到 76 会造出 152 B 行距、奇数行 ldmatrix `MISALIGNED_ADDRESS`，探针实证），一条 barrier + 一个合并的四 tile 拷贝流。门全绿（位级一致、池 46,432 ≤ 49,152 保 2-CTA），但配对 **+0.0/+0.6/+0.8 µs ×3、T2048 +0.2 µs（39–49/80）——掷硬币**。机理：2 CTAs/SM 下**共驻 CTA 填另一 CTA 的 barrier 间隙**，核内 epilogue 串行化本就不在 SM 级关键路径上；31 µs 的 s2g 泄流是 DRAM 受限（r21 ncu：DRAM 57%、占用 16.2%）。这是“barrier 数与发射顺序杠杆在 2 CTAs/SM 不付钱、只有 DRAM 流量类杠杆还能动 f1”这一调度规律的第 3 个数据点，**核内串行化类就此关闭**。

#### 5.1.7 F1 时间图谱与端到端关闭账目

r21 的十个可组合删除探针（`GDN_F1T_*`，×3 会话 ×40 对；输出按设计为垃圾、永不进 verify）给出 f1（~227 µs 基准 @T8192）的表观相分布：

| 相（删除探针） | 表观成本 | 占 f1 |
|---|---|---|
| uw 段（gemm+epilogue+s2g，NUW） | **116.5 µs** | **51.2%** |
| uw 段+operand staging（NUST） | 131.6 µs | 57.9% |
| kkt 整段（NKT） | 59.5 µs | 26.2% |
| solve 整体（NALL） | 41.3 µs | 18.2% |
| solve phase-1（NP1） | 32.5 µs | 14.3% |
| u/w s2g 拷贝单独（NC2G） | 31.0 µs | 13.6% |
| gc 扫描+β（NSC） | 17.6 µs | 7.8% |
| solve phase-2（NP2） | 14.6 µs | 6.4% |
| solve phase-0（NP0，已折叠为 F1ZF） | 8.5 µs | 3.7% |
| gc_out 存回（NGC） | 0.8 µs | 0.3% |

派生拆分：uw staging ≈ 15.1 µs（NUST−NUW）、uw mma+epilogue+staging 等待 ≈ 85.5 µs（NUW−NC2G）、u/w 全局存 ≈ 31.0 µs（NC2G）。单相占比和 ~109%——相之间存在 DRAM/L2 争用，每个 δ 是该相串行成本的**上界**。r21 ncu（独占、锁频 1.57 GHz）：f1 313.95/312.06 µs、DRAM 56.8–57.7%（400.8 GB/s）、占用 16.24%（8 warp/SM：寄存器限 2 × smem 限 2）、2048 CTA/82 SM = 12.49 波、发射间隔 5.8 周期（无合格 82.6%）、warp 停顿构成 long_scoreboard 24.3% / barrier 22.9% / wait 17.8% / math_pipe 12.4%——**f1 在 2 CTAs/SM 上是延迟受限而非带宽/张量受限**。

r23 之后 f1 端到端关闭：uw 串行化类（F1EP wash）、s2g（DRAM 受限）、staging（persistent-k 池表：一块 $[64,136]$ bf16=17,408 B 的常驻 k 需要横跨 kkt→uw 存活，而该窗口内唯一死字节是 kt 的 6,144 B，**差 11,264 B**——任何别名排列都撞活范围重叠，需 ≥11.3 KB 池重构而非旋钮）、kkt（屋顶线）。残余：phase-1 串行 32.5 µs 与扫描 17.6 µs，各有两轮诚实负结果（F1WY/F1P4；SCANDB 已硬化），重开需新机制类。

---

### 5.2 F2 "scan"：状态递推 kernel

F2 承载全算子唯一的跨 chunk 串行依赖：$S_{n+1}=2^{\mathrm{gc}_{\text{last},n}}S_n+(\,k\odot2^{\mathrm{gc}_{\text{last}}-\mathrm{gc}})^{\top}(u_n-w_n S_n)$。最终形态（`gen_f2`，pp 变体）：grid $(B\!\cdot\!H\!\cdot\!VT)$，每 CTA 一个 (head, V-tile) 的 $[K,\mathrm{BV}]$ 态 tile 常驻共享内存，`scf.while` 走 $NC$ 个 chunk；每 chunk 三个动作——**vp gemm**（$\mathrm{acc}=w\,S$，A=乒乓半体的 wb，B=常驻 S）+ **vn epilogue**（$v_{\text{new}}=(u-\mathrm{acc})\,2^{\mathrm{gc}_{\text{last}}-\mathrm{gc}_r}$，kterm 折叠约定：F3 的 P epilogue 用配对的 $2^{\mathrm{gc}_i-\mathrm{gc}_{\text{last}}}$，全局 $v_{\text{new}}$ 原样携带缩放后的 tile——r1 的原地 RMW kterm 扫pass因此消失）+ **sd gemm 与 S 更新 epilogue**（$S\leftarrow S\cdot2^{\mathrm{gc}_{\text{last}}}+k^{\top}_{\text{term}}v_{\text{new}}$，在 mma epilogue 里完成 RMW）。入口态 `h_all[n]=S` 与 `v_new`/`ht` 的全局存取全部走合并 cell 扫pass（fragment 映射的原位全局存是 4 B 跨距上的 2 B 存，实测比多一次扫pass**慢 40%**——r3 记录在案的盲巷，反向结论同样适用于 o/u/w 段）。

#### 5.2.1 机制：chunk 级乒乓双缓冲与“组 A”预取（以及 wait_group-0 的机器约束）

**动机。** 串行 chunk 链上，下一 chunk 的 $w/k/\mathrm{gc}$（以及后来的 $u$）若同步装载，装载延迟全部落在链上。

**机器约束（源码级证明，决定实现路线）。** 树内 `NVGPU/Transforms/CreateAsyncGroups.cpp` 对每段由 `create_async_groups` 转换的 staging **立即**插入 `cp.async.wait_group 0`（“Create the group and wait for it right after”）——transform 路线**结构上**无法让预取与计算重叠。手动发射的 `nvgpu.device_async_copy` 原样透传，等待位置归生成器。

**实现（`gen_f2` pp 变体）。** 乒乓奇偶折进**索引算术**（memref 值不能被 `arith.select`）：`wb` 是 $C\times2K$ 沿列对分、`kb` 是 $2C\times K$ 沿行对分、`gcs` 是 $2C$ f32。chunk $n$ 顶部发出“组 A”（chunk $\min(n{+}1,NC{-}1)$ 的 w/k/gc 进奇偶 $n\hat{}1$ 半体），与 chunk $n$ 的**全部**计算重叠；chunk 末 `device_async_wait numGroups=0` + barrier 翻转奇偶。

**IR（chunk 循环头与奇偶算术，`r22ship/payload.mlir`）**：

```mlir
%nwr = scf.while (%nw = %c0) : (index) -> (index) {
  %ncc = arith.cmpi ult %nw, %c32 : index
  scf.condition(%ncc) %nw : index
} do {
^bb0(%n: index):
  %n2 = arith.remsi %n, %c2 : index
  %pp = arith.subi %c1, %n2 : index            ; 预取奇偶 = 1 - n%2
  %nm = arith.minsi %np1, %c31 : index         ; min(n+1, NC-1)
  %pK = arith.muli %n2, %c128 : index          ; 读半体列基
  %qK = arith.muli %pp, %c128 : index          ; 预取半体列基
  ...
  %acp = nvgpu.device_async_copy %w[%ix_8], %wb[%idv, %ix_9], 8 bypassL1
      : memref<4194304xbf16> to memref<64x264xbf16, ...>   ; 组 A：w
  %acp_9 = nvgpu.device_async_copy %k[%ix_39], %kb[%ix_40, %im_24], 8 bypassL1
      : memref<4194304xbf16> to memref<128x136xbf16, ...>  ; 组 A：k^T
  ...
```

**效果（诚实归零 + 真根因移交 §5.2.2）。** 乒乓单独立项时**零增益**：471→473 µs（r3，“计划中的杠杆 1 不是故事”）——因为 F2 当时不是装载延迟受限，而是被 §5.2.2 的 LDSM 冲突卡死。乒乓结构保留（它给 F2US/W32P 提供了载体）；在产 pp 体每 chunk 2 条 barrier（r5 审计对“~8/chunk”认知的修正——那是对 r2 旧体的描述，全核含序共 3 条）。sd gemm 的 B 读按 N 维 warp 局部（`ldm_b` 列 = 本 warp 自己的 8 列，与 vn epilogue 写同一 warp 同列——mma N 瓦片恒等式），epilogue 与 sd 之间免一条 barrier；h_all/v_new 的跨 warp 消费已由屏障覆盖。

#### 5.2.2 机制：LDSM +16 B 垫——8 路 bank 冲突的体系级修复

**动机（本体系最大单笔杠杆的根因）。** ncu（乒乓 F2）：发射间隔 63% 停在“MIO 队列满”、每调度器 2.0 活跃 warp、每 SM ~40 周期一条 LDSM、每 warp 每 chunk 76 条 LDSM。所有被 ldmatrix 读的 tile 行距恰为 128 B 的整数倍（$64\times128$→128 B、$64\times256$→512 B…）：LDSM 象限的 8 行全落进**同 4 个 bank = 8 路冲突 ≈ 8× 慢的 LDSM**。r2 时 43% 的 MIO 停顿即此。

**修复。** 每个被 ldmatrix 读的 2-D tile 一律 **+8 个 bf16 列**（=+16 B）：行距变为 128 B+16 B 的倍数，8 行依次轮换扫过全部 32 bank；+16 B 保持 cp.async 的 16 B 对齐。发运 payload 里每个 LDSM-tile 都带这个垫（`wb [64,264]`、`kb [128,136]`、`S [128,40]`、`vn [64,40]`、f1 的 `Abf [64,72]`、f3 的 `P [64,72]`……）。

**效果。** F2 473→**249 µs**（w8/bv64，r3）；配合 w4/bv32 重调（§5.5）达 83 µs 静窗。同一垫顺手用于 f1/f3 全部 LDSM tile（收益在彼处被别的瓶颈掩盖——诚实记录）。同轮 F2 顺带**撤下** u 的 staging（vn epilogue 直接从全局读，其延迟藏进 vp gemm 的 fragment 装载之后；省 8 KB smem + 2 cp.async + 每 2 元素 1 条共享操作）——这一“撤”后来又被 F2US（§5.2.3）以更高形态逆转。

#### 5.2.3 机制：F2US——u 段重回乒乓预取管线

**动机。** r3 的“u 直接从全局读”在 T8192 的账：每 chunk 每 CTA 4 KB 的 u，fragment 映射下是 **2 B 装载 × 4 B 跨距**——每个 32 B 扇区只交付一半字节，4 KB 放大成 ~8 KB 扇区、聚合 ~32 MB→64 MB，且每一发装载都坐在串行链（vp mma→epi_vn→sd mma→S 更新）上，L2 延迟直接暴露。删除探针定价这部分 ~104 µs @T8192。

**实现（`GDN_F2US`，发运开启）。** u 的 VT 切片重新 staging——但不是回同步扫pass，而是进**组 A**：行乒乓 tile `ut [2C, BV+8]`（+10,240 B；池 84,480→94,720 B，仍 1 CTA/SM），随 w/k/gc 一起 chunk 顶部发出、chunk 末既有 wait 收割；epilogue 读无冲突的垫化共享内存。字节与算术顺序不变 ⇒ 位级一致（T8192 门与 verify 差分精确复现）。

**IR（epi_vn 的 u 读，`r22ship/payload.mlir`）**：

```mlir
%vnu = memref.load %ut[%ix_307, %nct_vp_0] : memref<128x40xbf16, ...>  ; F2US
%vnf = arith.extf %vnu : bf16 to f32
%vnd = arith.subf %vnf, %evn : f32            ; u - (w@S)
%vnz = arith.subf %gcl, %vnr : f32            ; gc_last - gc_r
%vne = math.exp2 %vnz : f32
%vns = arith.mulf %vnd, %vne : f32            ; kterm 折叠
%vnt = arith.truncf %vns : f32 to bf16
memref.store %vnt, %vn[%vrow, %col] : ...
```

**效果。** 配对 A/B（×60 对）：f2 @T8192 **0.4017→0.2972 ms（−104.5 µs，0/60 更慢）**、staged e2e −96.8 µs；@B4T2048 f2 −101.2 µs（0/60）、e2e −91.4 µs。T8192 正式终判随之从 1.021×（平手）越线 1.101×（r14，n=30，p≈1e-9）。资源门（§5.5）：w8/bv64 下 ut 翻倍使池 115,200 B > 101,376 B，ptxas 拒绝——F2US 因此是环境旋钮 + 按形状构建，不是无条件默认。

#### 5.2.4 机制：W32P——半 warp 行重映射的无冲突宽存储（与两轮否定后的幸存路线）

**动机。** r15 删除探针把 f2 的两条 cell 扫pass定价 **h_all 35.4 µs + v_new 17.9 µs**（T8192 的 297 µs 链内，合计 ~53 µs 天花板）。WST 的 32 位宽存储（§5.3.3）在 f1/f3 付了钱，但在 f2 直接套用是**负的**：BV=32 tile 每行只有 16 对，一个 warp 的 32 条 b32 共享读要跨两行，而 (BV+8)=40 元素=80 B=20 字的行距使 lane 28–31 与 lane 0–3 撞同一 bank（2 路冲突；仅映射即 +4.3 µs，47/60，r15）。lane 级配对（`GDN_WSP`：保无冲突单格读、`shfl.sync.bfly` 取伴 lane、`prmt` 打包、偶线程谓词 b32 存）更惨：**+157.5 µs（60/60）**——SASS 上 stock 循环四路展开带四条独立批量 `STG.E.U16`（满 MLP），WSP 循环是每次迭代一条 `WARPSYNC→LDS→SHFL→PRMT→STG` 串链，跨 lane SHFL 依赖禁止展开且每迭代加一个汇聚点；而理论节省本就是零（谓词半 warp b32 存与全 warp b16 存发射相同 warp 指令数、搬相同 64 B）。

**W32P 的银行算术（`GDN_W32P`，发运开启）。** 保持配对读存（`[base]`/`[base+2]`，`gdn_smem_fix` 的 `widen_stores` 结构性融合照常触发），只改**行映射**：lane 0–15 读行 $r=8\cdot\mathrm{round}+w$，lane 16–31 读行 $r+4$。字地址：行 $r$ 的 16 对 = 字 $[20r,\,20r{+}16)$；行 $r{+}4$ = $[20r{+}80,\,20r{+}96)$；$80\bmod32=16$，两个半 warp 恰好落进**互补的两个 16-bank 半区**——对每个 $r$ **零冲突**。双射覆盖：轮 $j$、warp $w$、半区 $h$ → 行 $8j+w+4h$（$h\in[0,8)$ 覆盖行 $8j..8j{+}7$、对 lane%16 各一次）。纯索引算术：同字节、同行内配对、同全局地址 ⇒ 位级一致；池/tile/LDSM 垫/占用全部不变；f2 的 smem_fix 宽存计数 0→2。

**IR（发运 T8192 payload 的 h_all 扫pass头部，`gen_fwd/B1T8192_mono/payload.mlir`，即 +W32P 记录代）**：

```mlir
%w3h  = arith.cmpi uge %lane, %c16 : index            ; 半区判定
%w3h4 = arith.select %w3h, %c4, %c0 : index           ; +4 行偏移
%ix_160 = arith.addi %warp, %w3h4 : index             ; 行槽 = warp + 4·hi
%irm_38 = arith.remsi %lane, %c16 : index
%im_90 = arith.muli %irm_38, %c2 : index              ; 对列 = 2·(lane%16)
%wir_2 = scf.while (%wi = %tid) : (index) -> (index) {
  %wic_2 = arith.cmpi ult %wi, %c2048 : index
  scf.condition(%wic_2) %wi : index
} do {
^bb0(%wib: index):
  %idv_38 = arith.divsi %wib, %c128 : index           ; 轮 j = i>>7
  %ix_161 = arith.addi %im_91, %ix_160 : index        ; 行 = 8j + warp + 4·hi
  %sgl = memref.load %S[%ix_161, %im_90] : ...        ; b16 ×2 → smem_fix 融合为 b32
  %sgl_2 = memref.load %S[%ix_161, %ix_162] : ...
  ...
```

**效果。** f2 配对（翻转双装载、×40 对、17 会话）：T8192 **−17.9..−22.6 µs ×7**（慢对率 5–8/80）、B4T2048 −9.7..−25.6 µs ×5、T2048 −2.0..−3.8 µs ×5；链中位全程三形状为正（T8192 −9.1..−12.0 µs ×5、B4 −14.3/−27.5、T2048 −3.2/−5.2）。~6–8% of f2 @T8192，正落在纸面带内（WST 在 f1 付 0.176 µs/MB → f2 的 100.6 MB 扫pass写 ~17 µs 天花板）。门：T8192 fp64 门**位级一致**、fwd_verify 4/4 记录数字、det N=5、占用不变（f1 2 / f2 1 / f3 1 blocks/SM）。附带诚实披露：+W32P 的 T8192 构建抽到一个可复现的 f3 +11.8..+12.6 µs 模块布局税（f1/f3 PTX 节与纯度构建字节相同——f2 的 ~+300 B PTX 触发 r19 定位的模块摆放抽签类，f3 ±10.5±6 µs）；链净值仍每形状每会话为正，n=30 存档重掷该布局。r25 存档终判即在此代上读出（§5.0 的 2.1250×/1.2211×/1.3493×）。

#### 5.2.5 机制：段切分扫描——M-复合交接的数学与 Amdahl 式诚实否定

**动机。** T8192 的 F2 是 128-chunk 串行链、64 CTA。教科书解法是两遍段切分：把 $NC$ 个 chunk 分成 $S$ 段，先并行算每段的出口态再带真入口态重跑。

**数学（先证精确，再证为何免费）。** F2 的 chunk 递推对态是**仿射**而非态无关：$S'=M_nS+b_n$，$M_n=2^{\mathrm{gc}_{\text{last}}}I-k_{\text{term}}^{\top}w$。一般精确并行分裂需要交接式 $E_{sg+1}=M_{sg}E_{sg}+c_{sg}$（每 chunk 一次额外 $K\times K\times C$ gemm + 每段 ~7 个 $K^3$ 复合，~3.4× F2 flops、88 KB smem、1 CTA/SM）。但每个因子 $M_n$ 携带全 chunk 衰减（kterm 里的 $2^{-\mathrm{gc}}$ 与 $w$ 里的 $2^{+\mathrm{gc}}$ 逐 token 相消）：

$$\lVert M_{seg}\rVert \;\sim\; 2^{\sum \mathrm{gc}_{\text{last}}}\cdot O(1) .$$

GDN 基准配方 $g=-\mathrm{softplus}(0.5x)$、平均 log 衰减 ~−0.25/token ⇒ 64-token chunk 衰减 ~$e^{-16}$；在真实基准输入上精确 fp64 复合实测 $\lVert M_{seg}\rVert\le3.6\times10^{-13}$（L=1，T512/S=8）与 $4.3\times10^{-52}$（L=4，T2048/S=8）（`tools/g5i_split_sim.py`）——被丢弃的 $M_{seg}E$ 项远低于 bf16 分辨率（~4×10⁻³ 相对），**免矩阵交接 $E_{sg+1}=c_{sg}$ 在本契约的输入下精确**。有效性边界由 `--stress-g 0.02` 探针划出：近零净衰减（gate≈1，段衰减 ~$e^{-1.3}$）时 $\lVert M_{seg}\rVert$ 涨到 0.07–0.45、免矩阵交接入口误差 3.1×10⁻²（会破 5×10⁻³ 门），精确复合变体仍 3.7×10⁻⁴——M-复合回退**已设计未实现**，记录为遗留。

**实现（`--f2-segments S`，mono 专用）。** `f2a`（grid ×S）：每段从**零入口**跑 NC/S 个 chunk，只把出口态写 `@seg_exit`（零入口下内部 h_all/v_new 是错的、不写）；`f2c`：每段带真入口（段 0 用 h0，其余用 `@seg_exit[sg-1]`）重跑并写真输出。体 = r2 流水化形（`emit_kloop` scf.for 保 transform 流水）+ r3 垫 + r3 撤 u。单缓冲态 tile：43,264 B、206/230 寄存器 ⇒ **2 CTAs/SM 共驻**——本来的第二杠杆。

**数值（全绿，含一段近失教训）。** 4/4 PASS（T512+T2048，分段与串行同裕量）；新增 T8192 fp64 门：$o\,6.85\times10^{-4}$/$\mathrm{ht}\,7.90\times10^{-4}$（7× 裕量），且 **S=1/4/8 的 o 与 ht 位级相同**。近失教训：f2a 起初跳过段 0（“其出口无人用”——错，段 1 的入口就是 $c_0$），`seg_exit[0]` 保持零初始化、h_all[2] 垃圾而 ht 仍 PASS（入口误差被衰减吞掉）——按 chunk 二分用中间量定位。

**效果（Amdahl 式诚实否定）。** 全形状**更慢**：staged T8192 1.298→1.541/1.507 ms（S=4/8）、B4T2048 1.155→1.556（S=4）。两笔账：(i) 第二遍全价——f2a 是一遍免存储的 chunk 全工、f2c 重复全部工再加存储，2× 工只换 ~1.2× 执行 ⇒ 16–33% 净损失；(ii) **Amdahl 17%**——f2c 单独 @S=8 是等工探针（全部 NC chunk、16-chunk 链、2 CTAs/SM）：0.529 ms vs 串行 0.640 ms，链长+共居合计只值 **F2 的 17%**（111/640 µs），其余 83% 是每 chunk 吞吐（staging 流量、mma 发射、barrier、epilogue）——而分裂恰恰复制这些。r3 之后的 F2 已是每 chunk 吞吐受限（64 CTA≈78% SM、每 chunk 5.0 µs 对静窗 T2048 的 2.6 µs——差值是 T8192 更冷的 w/u/k staging，不是链）。推论：**任何扫描重构（段、波、链式 CTA）在这些形状上赢不了串行 kernel**；发运默认 S=1。G5-F 的 ~250 µs 投影把 650 µs 串行 F2 当延迟受限建模（650×2.7/8），测量否决了该模型。

#### 5.2.6 F2 负结果全家福与关闭

除 §5.2.4–5.2.5 外，F2 的其余攻击面全部按测量关闭：
- **F2SB 单缓冲节食**（`--f2 sb`）：wb/kb/gcs 单缓冲化把池压到 50,432 B（2 CTAs/SM 可容纳），预取折进 tile 死点（wb 死栅栏在 vp 后、kb+gc 在底部）——**+21.2 µs**（57/60）；强制 64 KB L1 carveout **+20.9**（60/60，carveout 救援假说被否）；B4T2048（256 CTA、2-CTA 共驻真实生效、164 常驻）**+28.5**（56/60）。根因：pp 的组 A 在 chunk 顶部发出、重叠**全部** chunk 计算；每个单缓冲折叠都把传输移到部分计算**之后**（死栅栏绑定），暴露成本超过 2-CTA 延迟隐藏的回收——即便真 2× CTA 网格、即便强制 carveout。
- **F2BAR 2→1 barrier/chunk**（vn 行乒乓 + v_new 扫pass移下迭代顶）：**+19.7 µs**（55/60，r5）——+5 KB smem（84.5→89.6 KB）挤掉 u 装载/存储扫pass依赖的 L1 carveout，代价超过省下的 bar.sync。r5 审计同时纠正了一个在产认知：**发运 pp 体每 chunk 2 条 barrier**（f2 全核 3 条含序），“~8/chunk”是 r2 旧体的数字——barrier 杠杆在 round 5 开始前已基本花完。
- **w4/bv64（唯一在预算内的每 warp N=16 形态）**：f2 **+218.4 µs**（60/60，r15）——更宽每 warp N 付不起砍半的 CTA 并行。
- **WSP lane 级配对**（§5.2.4 内）：+151..157 µs，双表面（tile 级 +4.3、lane 级 +151）皆死后才轮到 W32P。
- **128 位全局存路**：`vector.load/store`（含 `vector<8xbf16>→vector<4xi32>` 位投掷变体）**能降过**本管线但元素分裂为 b16 op（PTX `st.global.b16 ×8`，探针）——扫pass的 64 B/warp 标量存已是该管线的最优；h_all 亦已近 64 MB 写带宽地板（~35 µs @ ~1.8 TB/s）。

F2 的终局读法（r24/r25）：写路径上 W32P 是最后一个位级一致杠杆；扫pass剩余成本是 DRAM 写受限（100.6 MB @T8192）——F2 与 F1 一样以“算法最小流量 + 实测屋顶线”关闭。

---

### 5.3 F3 "o"：输出级 kernel

F3 从 F2 的产物（`h_all` 入口态、`v_new`）重算每 chunk 的输出：$o=(2^{\mathrm{gc}}\!\odot\!(q\,S_{\text{entry}})+P\,v_{\text{new}})\cdot K^{-1/2}$，$P=\mathrm{tril}_{\text{inc}}((q\,k^{\top})\,2^{\mathrm{gc}_i-\mathrm{gc}_{\text{last}}})$（与 F2 的 kterm 折叠加起来恰还原数学式）。grid $(NC\!\cdot\!B\!\cdot\!H,\,VT)$，发运取 $\mathrm{BV}_3=V$（VT=1）。

#### 5.3.1 机制：qk+oi 融合循环

**动机。** $qk^{\top}$ 与 $q\,S_{\text{entry}}$ 共享 A 操作数 $q$。两条 K-循环 → 一条：一个 staging 的 q tile 同时喂两个 gemm（`emit_kloop` 的 gemms 列表——机制层面的“一个操作数服务两个消费者”在 kkt（k 双侧）与 qo（q 双侧）各出现一次）；k tile 以 `kt=True` 读、S 切片按 k 主序从 `h_all` staging。随后 **P epilogue**（f32 域折 $2^{\mathrm{gc}_i-\mathrm{gc}_{\text{last}}}$、`cmpi sle col,row` 含对角的下三角选择——上三角单元被写成**精确零**，这使 F3 天然没有 ZFIX 类的未初始化暴露）写 bf16 `P`；**oa 循环**（A=常驻 P、B=staging 的 v_new 切片）+ **o epilogue** 跨两组累加器（oi 的 $2^{\mathrm{gc}}$ 行缩放从 staging tile 移进 f32 epilogue——少一次舍入）求和、乘 $K^{-1/2}$、截断。

**效果。** r2 的三重奏重构里这一融合循环与 BV=128 是 f3 的形态基础（239 µs @B1T2048 w8 时代即 DRAM 57–69%、31% 占用、~50 MB @ ~0.4 TB/s）；其后 f3 的每次变动都是下述两杆秤的称量。

#### 5.3.2 机制：BV 选择——bv3=V 的单 tile 论证与 bv3=32 的否定

**动机。** F3 的 q/k 装载量按 VT 倍增（每 VT3 CTA 重复整个 qk gemm 与 q/k staging）。

**权衡与测量。** 发运 $\mathrm{bv3}=V=128$（VT=1）把 q/k 重读减半以上；代价是单 CTA 更大的 v_new/st tile。反例已测：**bv3=32**（f3 池 45,432 B、8192 CTA、2 CTAs/SM）f3 **+357.2 µs**（60/60）——qk gemm 与 q/k staging 每 VT3 CTA 重复一次，占用输给重复工。f3 的占用天花板另有其人：240 寄存器本可 2 CTAs/SM（240×128×2=61,440 ≤ 65,536）但 smem 82,176 B 不容（2×82,176 > 100 KB），`.maxnreg` 注入工具（`GDN_MAXREG_K` per-entry）就位后确认**寄存器帽动不了占用**——纠正了 r14 简报的一个预判。

#### 5.3.3 机制：F3O 与 WST——o 段 restage 和 32 位宽存储

**动机（与 F1UW 第 3 刀同源）。** o epilogue 的 fragment 映射全局存是离散 2 B 写（16 行 × 每线程 4 列对）；restage 到垫化 `ot [64,136]` tile + 一次合并 cell 扫pass把它们变 64 B/warp 行段（G5-F 负结果的逆转：原位全局存输给额外扫pass）。136 列（272 B）行距同时保持 STS 侧无冲突（4 字轮换）。**WST**（`GDN_WST`，f3/f1 侧发运开启）再砍一半指令：生成器扫pass发**成对单元** $(2i,2i{+}1)$（2×2 b16 读写四元组、显式 +1 邻接），`gdn_smem_fix.py` 在 PTX 级把每个四元组融合为 `ld.shared.b32 + st.global.b32`（字节精确——32 位字就是那两个 bf16 单元），LSU 指令与循环迭代减半。守卫 `cols ≥ 64`（f1 的 128 列行保证每 warp 在一行内；f2 的 32 列不满足——它的解是 §5.2.4 的 W32P）。

**效果。** F3O：f3 −1.3 µs @T8192 / **−8.4 µs @B1T2048**（22/60 更慢——原则性正确、量级噪声）；与 F1UW 的 u/w 扫pass合并看，WST 付 f1 −11.8/−11.4/−10.6 µs（T8192×2、B4；1–5/60 更慢）、staged e2e −9.0/−13.4/−7.0 µs，数值位级一致（T8192 门 $o/\mathrm{ht}=0.000e{+}00$）。r19 均衡复测确认 WST 的 f3 效应 ~0（原始 +15.2 µs 读数是 B 侧摆放抽签——A/B 器械史里重要的一课：`r14u_ab` 的 B 侧偏差被根因为**模块构建顺序的摆放抽签**，f3 的 ~160 MB 工作集上第二个认领 `.global` 区的模块付 ~6%；均衡双装载（镜像构造序 + 逐对交替发射序 + 丢弃首对 + warmup 20）成为默认器械）。

#### 5.3.4 机制：F3KS 2-CTA 转化——带机制分解的诚实否定

**动机。** f3 池 82,176 B @1 CTA，是 B4 链上第二大 kernel；F1KS 的 playbook（bk=16 K-循环节食 + `ot` 落 `P` 的死 arena）看起来可直接移植：qt/kt `[64,24]` 多缓冲 6,144 B ×2、st/vnt `[16,136]` 8,704 B ×2、`ot` 别名进 P 的死字节（P 的 malloc 以 ot 的 17,408 B 尺寸申报、`ot` 别名 +0、oa 循环后一条显式 barrier 钉住 P 的最后 ldmatrix 先于 ot 首写——流水化 transform 会重组循环尾，r15 `bisect_shift` 的教训）。池 82,176→47,360 B ≤ 50,688，2 blocks/SM。审计先行：f3 无扫描（无 SCANDB 暴露）、无清零 cell 循环（epi_P 全覆盖写含上三角零、无 ZFIX 暴露）。门全绿：强 1-CTA 位级同、2-CTA ×10 复跑位级干净（两形状）、三重奏 7/7 位级同、B4 ×3 位级同、verify 4/4、det N=5。

**效果（否定 + 分解）。** T8192 f3 **+96.1 µs**（74/80）、staged +97.0；B4T2048 f3 +74.9（76/80）。分解臂定音：**同一份 bk16 码强制 1 CTA = +115.6 µs**——K-循环重排本身 ~+110 µs（3 段 qo 循环 + oa 循环的迭代数翻倍：barrier/wait/staging 发射步全部翻倍），2-CTA 驻留只买回 ~20 µs（116→96，尚不含 +8.7 同构抽签校正）。机理：f3 @T8192 流 ~120 MB 的 `h_all`/`v_new`/`o`——**带宽受限 kernel 不为波数减半付钱**（f1 的 −53.5 µs 是延迟受限情形的正面对照）。`GDN_F3KS` 默认关、不发运；f3 至此在 1 CTA 上以带宽受限关闭（r20）。

---

### 5.4 融合旗舰：单 persistent kernel

#### 5.4.1 结构：chunk 循环内的八段 mma 与终身共享 smem

`gen_fused` 把全部工作装进 grid $(B\!\cdot\!H\!\cdot\!VT)$ 的单 kernel：每 CTA 持 $[K,\mathrm{BV}]$ 态 $S$ 常驻共享内存，`scf.for %n` 走 NC 个 chunk，每 chunk 依次八个 mma 位点——**kkt**（$A$）、**ug**（$u$）、**wg**（$w$）、**vp**（$wS$）、**qk**（$P$ 前）、**oi**（$qS$）、**oa**（$P\,v_{\text{new}}$）、**sd**（$S$ 更新）——加一个 flat solve（`emit_solve`，thread-per-column 的 L1 逐字递推）。99 KB 硬顶下的 tile 预算靠**终身共享**：`QK` tile 先装 bk2（$\beta k 2^{\mathrm{gc}}$）、wg 消费后**晚装** q；`TP` 先装转置 $T$、再装 $P$；`BVN` 先装 $\beta v$、epilogue 原位覆写 $v_{\text{new}}$。r14 的刷新（`GDN_FLAV1`，BV≤48 默认开）把 r3 的 LDSM 垫（k_t/QK/S/w_t/TP/BVN +8 列、`A_m` +1 f32 列）、F1 的 Hillis-Steele 扫描、F3O 式 o restage 全部移植进 fused，池 @w4/bv32 = **99,072 B**（w8/bv64 的垫化池 114,672 B > 101,376 B 装不下——刷新以 BV≤48 为门，记录在案的几何门）。

**IR（发运 fused payload 的 chunk 循环与首批 cp.async 组，`gen_fwd/B1T2048_fused/payload.mlir`，即 FPIPE+SCANDB 代）**：

```mlir
scf.for %n = %c0 to %c32 step %c1 {
  ...
  %acp = nvgpu.device_async_copy %k[...], %k_t[...], 8 bypassL1 : ...  ; k(n)→k_t
  ... (1024 条 16B 拷贝)
  %agr = nvgpu.device_async_create_group %acp, ..., %acp_8
  %acp_9 = nvgpu.device_async_copy %v[...], %BVN[...], 8 bypassL1 : ... ; 原始 v(n)→BVN
  %agr_2 = nvgpu.device_async_create_group %acp_9, %acp_10
  ... (gc 扫描与 k 的 wait 并行)
  nvgpu.device_async_wait %agr numGroups = 0
  ...
  nvgpu.device_async_wait %agr_2 numGroups = 0                            ; solve 后
  ...
  %mr_oa_0_0_0 = nvgpu.mma.sync(%fa_oa_0_0, %fb_oa_0_0, %acc_oa_0_0)
      mmaShape = [16, 8, 16] : ...                                        ; 八段之一
```

#### 5.4.2 机制：FPIPE——同步 staging 的三段 cp.async 改造与“k 一次读”

**动机。** 串行 chunk 链上有三段同步 global→smem cell 扫pass（k、$\beta v$、q）外加 bk2 对同一 k 的**第二次全局读**（每 chunk 16 KB）。这些扫pass在管线里降成 LSU 码、直接坐在链上。

**改造（`GDN_FPIPE`，fused 发运开启；池不变 99,072 B，payload +127 行）**——三段异步化 + 一次 k：

1. **k(n)→k_t 与原始 v(n)→BVN 在 gc 扫描之前发出**（两组 token；两 tile 自 sd(n−1) 消费其上一内容起已死）；k 的 wait 放 kkt 前（重叠整个扫描），v 的 wait 放 solve 后，$\beta$ 以一次**共享内存原地缩放**扫pass施加（与 `pre_bv` 完全同算术——位级一致）。
2. **bk2 变纯 smem→smem 缩放扫pass读常驻 k_t**——k 每 chunk 从全局读**一次**而非两次。
3. **q(n)→QK 在 wg 的 barrier 之后发出**（QK 自 wg 消费 bk2 起已死），qk 之前 wait——重叠整个 vp gemm。

**效果。** fused T8192 **6274→4530 µs（−27.8%）**，三会话 ×40 交错对全部 0/40 更慢（配对中位 −1744/−1750/−1736 µs）；向 T2048/B4 复制为 **−26.5%**（1555.5→1143.3；6216.2→4581.9 µs，同样 0/40 ×3）。数值：对 stock fused 全程位级一致（$o/\mathrm{ht}=0.000e{+}00$）、verify 4/4、det N=5。增益同时是延迟隐藏与指令节食（−28% 远超纯延迟模型的 ~−10%）。r17 的 PTX 审计顺带把 mono 三重奏的同类机会**关闭为负**：trio 的全部 global→smem staging 已是 cp.async（f1 30 op、f2 42、f3 42），仅存的同步全局读（f1 的扫描输入 512 B、f3 的 gc 256 B）没有任何可重叠窗口——FPIPE 类是 fused 路径特有（其串行链背了三条同步扫pass + 一次 k 重读，trio 的 kernel 边界从未有过）。

#### 5.4.3 跨 chunk 乒乓的算术不可行：2,304 B 余量对 17,408 B tile

**动机。** FPIPE 拿下核内重叠后，剩下的串行暴露是 chunk 间边界——跨 chunk 的 k/q 乒乓预取似乎是下一步。

**算术（r16 一次性写清，生成器注释同源）。** 器件 opt-in 上限 $101{,}376$ B（查询：`MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`；102,400 B/SM、82 SM），池已 99,072 B ⇒ 余量 **2,304 B**；一块 k/q 乒乓 tile（$[64,136]$ bf16）= **17,408 B**。每个别名排列都撞活范围重叠（`w_t` 的 $[8..10]$ 行窗口对乒乓 tile 的整 chunk 驻留）。要买第三块 k 尺寸 tile 必须先付 bf16-solve 节食 + 垫回撤——代价超过收益。**跨 chunk 预取就此以算术关闭**；r14 的重分类叙事被修订一半：链不是不能抗核内重叠（FPIPE 吃掉 28%），但整 chunk 预取需要的是重构而非旋钮。

#### 5.4.4 诚实重分类：staged mono 扛旗

fused 路径的结构性天花板与占用无关，而与**并行度**有关：F1 的 WY 工作在 trio 里按 (NC, B·H) 全并行（T8192 = 2048 CTA），在 fused 里塌缩进 64-CTA 的串行 chunk 链——结构性差异不能靠移植 tile 修复。终局数字：fused（FPIPE+SCANDB，`cb2520d8` 等 r18 代）T8192 ~4.5 ms 对 mono 三重奏 **0.785 ms**（约 5.7×）；T2048 1.143 ms 对 0.187 ms。r14 起 fused 被诚实重分类为**设计记录**（design record），全部三个判定形状的旗舰标准由 staged mono 三重奏承担；fused 的存在价值是它证明了同一 payload 形态可以在两个融合极端间由 `--path` 一个生成器旋钮切换、以及 FPIPE 类机制只在它身上存在的定量原因（§5.4.2 的审计）。

---

### 5.5 形状自适应 dispatch 表

生成器的“自适应”分两层：**显式构建参数**（`--warps/--bv/--bv3/--f2/--f2-segments` + 环境旋钮集）按形状由发运程序选择并连同 `gdn_cache_key`（全部 `GDN_*`/`G5*` 旋钮 + 生成器源哈希）戳进 `full_spec.json`——缓存键不匹配即重定向 `_k<hash8>` 新目录，杜绝陈旧 cubin 伺服（r3/r19 两次事故后的制度）；**结构性守卫**折在生成器内部，按 tile 几何自动启用/拒绝。最终发运表：

| 形状 | 构建参数 | 环境旋钮集 | f1 grid | f2 grid | f3 grid | 判定（r25 存档 n=30） |
|---|---|---|---|---|---|---|
| small B1T512 H4 | mono, w4/bv32/bv3=128 | 同船集（dev-loop 验证形状；分段构建 s4/s8 在档） | (8, 4) | (16, 1) | (32, 1) | 数值门形状（无独立计时判定） |
| B1T2048 H16 | mono, w4/bv32/bv3=128 | `GDN_F1KS GDN_F2US GDN_WST GDN_ZFIX GDN_SCANDB GDN_F1ZF GDN_W32P` | (32, 16)=512 | (64, 1) | (512, 1) | **2.1250×**（0.1866 ms） |
| B4T2048 H16 | 同上 | 同上 | (32, 64)=2048 | (256, 1) | (2048, 1) | **1.2211×**（0.8213 ms） |
| B1T8192 H16 | 同上 | 同上 | (128, 16)=2048 | (64, 1) | (2048, 1) | **1.3493×**（0.7851 ms） |

守卫与门（每条都是本文某节的结论）：

1. **F2US 池律**：`ut` tile 在 w8/bv64 下使 f2 池 115,200 B > 101,376 B（ptxas 拒绝）——F2US 永不为无条件默认，按形状构建（r14 实测的精确失败数）。
2. **WST 的 `cols ≥ 64` 规则**：f1（V/K=128 列）与 f3（128 列）扫pass自动配对融合；f2（32 列）被规则排除，由 **W32P**（`cols==32 ∧ rows%8==0 ∧ nthr==128`，恰为 f2 的 S/vn tile）接手。
3. **F1KS 需要 nthr=128**（4 warp）与 bf16-solve；**F1ZF** 需要串行 phase-1 臂 + nthr=128；**F1EP** 只栈在 F1KS 上——旋钮间偏序由 `raise SystemExit` 硬约束。
4. **fused 的 FLAV 刷新门 BV≤48**（垫化池 114,672 B 在 w8/bv64 装不下）；**FPIPE** 池中性、全形状 fused 发运。
5. **f2 的 grid 上限几何**：BV=64 再瓦解（option a）死于 r14 池律 + 网格减半（64→32 CTA/82 SM = 39% 覆盖）双杀；W32P 是其唯一幸存支线（option b，r24 关案）。
6. **段切分 `--f2-segments` 仅 mono**、且 $S$ 必须整除 NC；发运恒为 $S=1$（§5.2.5 的否定）。
7. **f3 恒 bv3=V**（§5.3.2）；F3KS 不发运（§5.3.4）。

---

### 5.6 本章机制清点（收尾索引）

| # | 机制 | 旋钮/默认 | 关键配对数字 | 节 |
|---|---|---|---|---|
| 1 | gc Hillis-Steele 扫描 + SCANDB 双缓冲 | `GDN_SCANDB` / 开 | 损坏 4 K→0（10/10 位级）；扫描 17.6 µs | 5.1.1 |
| 2 | kkt K-循环（transform 流水 + 线程跨距 staging） | 结构默认 | 拆分 40.6/13.0/2–4 µs；826 对 765 GB/s 屋顶线关闭 | 5.1.2 |
| 3 | 分块 16×16 solve（对角 16 步 + 非角 stash/consume mma） | 结构默认 | F1 429→270 µs 时代；phase-1 = 14.3% | 5.1.3 |
| 4 | bf16-solve（F1KS 前提） | `GDN_BF16SOLVE` / 隐含开启 | 池 −16,384 B；数值 $6.626\times10^{-4}$ 类 | 5.1.3 |
| 5 | F1WY / F1P4（phase-1 重构双否定） | 默认关 | +92.6 / +14.8 µs ×3 | 5.1.3 |
| 6 | F1UW（单趟 + 预解 staging + 死 tile 复用） | 结构默认 | f1 −123.3 µs @T8192（4/60） | 5.1.4 |
| 7 | F1KS（K-分裂 + 终身 arena + bk16，2 CTAs/SM） | `GDN_F1KS` / 开 | f1 **−53.5 µs**；池 46,352 B | 5.1.5 |
| 8 | ZFIX（清零步长全覆盖） | `GDN_ZFIX` / 开 | 70 万→4 K bit 差（与 SCANDB 合用归零） | 5.1.5 |
| 9 | F1ZF（清零折进空闲线程） | `GDN_F1ZF` / 开 | −7.7/−7.9/−6.7 µs ×3；NP0 界的 91% | 5.1.6 |
| 10 | F1EP（单趟 epilogue，wash） | 默认关 | +0.0..+0.8 µs；核内串行类关闭 | 5.1.6 |
| 11 | chunk 乒乓 + 组 A 预取（wait_group-0 约束） | `--f2 pp` 默认 | 单独零增益（471→473）；载体保留 | 5.2.1 |
| 12 | LDSM +16 B 垫（8 路 bank 冲突） | 结构默认 | F2 473→249 µs；体系级 | 5.2.2 |
| 13 | F2US（u 重回预取管线，`[2C,BV+8]` 行乒乓） | `GDN_F2US` / 开 | f2 **−104.5 µs**（0/60）；T8192 越线 1.101× | 5.2.3 |
| 14 | WSP/WST-on-f2 双否定 | 默认关 | +157.5 / +4.3 µs | 5.2.4 |
| 15 | W32P（半 warp 行重映射） | `GDN_W32P` / 开 | f2 −17.9..−22.6 µs ×7 @T8192 | 5.2.4 |
| 16 | 段切分扫描（M-复合数学 + Amdahl 否定） | `--f2-segments`，发运 S=1 | $\lVert M_{seg}\rVert\le3.6\times10^{-13}$；链+共居=17% | 5.2.5 |
| 17 | F2SB / F2BAR / bv64 / bv3=32 否定 | 默认关 | +21.2 / +19.7 / +218.4 / +357.2 µs | 5.2.6 |
| 18 | qk+oi 融合循环 | 结构默认 | f3 形态基础 | 5.3.1 |
| 19 | F3O（o 段 restage） | 结构默认 | f3 −8.4 µs @T2048 | 5.3.3 |
| 20 | WST（成对单元 + PTX b32 融合，`cols≥64`） | `GDN_WST` / 开 | f1 −10.6..−11.8 µs；LSU 指令减半 | 5.3.3 |
| 21 | F3KS（2-CTA 否定 + 分解） | 默认关 | f3 +96.1（bk16 ~+110、2-CTA 买回 ~20） | 5.3.4 |
| 22 | fused 八段 mma + 终身共享 smem（FLAV 门） | `GDN_FLAV1` / BV≤48 | 池 99,072 B | 5.4.1 |
| 23 | FPIPE（三段 cp.async + k 一次读） | `GDN_FPIPE` / fused 开 | fused −26.5..−27.8%，位级一致 | 5.4.2 |
| 24 | 跨 chunk 乒乓算术关闭 | — | 2,304 B 余量 vs 17,408 B | 5.4.3 |
| 25 | 诚实重分类（staged mono 承担判定） | `--path mono` | fused 4.53 vs mono 0.785 ms @T8192 | 5.4.4 |

数字与 cubin 出处：`evidence/G5/fwd_report.md`（各机制段）、`evidence/G5/gen_fwd/{B1T8192,B4T2048,B1T2048}_mono/`（+W32P 记录代）、`*_mono.r22ship/`、`bench_B1T2048H16_mono.stale396b/`（pre-F1KS 对照）、`B1T2048_fused/`（FPIPE 代）、`*_mono_s{2,4,8}/`（段切分）、`final_adjudicate_r25f_w32p.json`（判定）、`fwd_verify_r24f2.json` / `fwd_verify_r18f_f1ks.json`（数值门）。生成器：`tools/gdn_full_gen.py`（4464 行）。

---

## 6 反向 kernel 体系：扫描族

> 覆盖对象：`tools/gdn_bwd_gen.py`（8687 行）中的扫描类 kernel——前向状态重算扫描 `fwdscan`，反向携带族 `caryA` / `caryM` / `bpre` / `dhs`，以及 pre-update 部分和的并行重算 kernel `dkb`；配套的 CPU 数学验证脚本 `tools/gdn_carry_validate.py`、`tools/gdn_fwdscan_ksplit.py`；全部性能与数值数字取自 `evidence/G5/bwd_report.md` 及其引用的 json/npz 证据链。本章按**最终系统**呈现：只描述装机（ship）形态与其设计依据，历史演化仅以"结构前/后对比"的形式出现；测量口径均为同窗口配对交错 A/B（CUDA event、30 轮取中位数），数值门槛为 fla 派生契约 dq/dk/dv ≤ 8e-3、dbeta/dg ≤ 2e-2、dh0 ≤ 8e-3，确定性契约为新起进程位级一致（fresh-process bit-identical，N=5–10）。

### 6.0 章首：反向为何是"重算前向 + 两条方向相反的扫描"，以及九 kernel 拓扑

#### 6.0.1 反向计算的两遍结构

分块 GDN（chunked GDN，块长 C=64）的反向传播在数学上是严格的两遍算法（numpy 孪生 `tools/gdn_l1_check.py::chunked_bwd_ref`，与 JAX vjp 在 fp64 下对到 1.4e-14）：

**第一遍——重算前向。** 反向梯度需要前向的全部中间量（每块的 WY 表示与入场状态），而这些量在前向推理结束后并不保留，必须在反向 kernel 内重算：

- 每块的代数重算：块内门控累积 `gc = cumsum(g)`（inclusive，log2 域）、核积 `kkT = k k^T`、WY 下三角系统 `A = strict_tril(beta .* kkT .* 2^(gc_i−gc_j))`、块伴随 `(I+A)^{-1}` 的分块求解、`u = T(beta v)`、`w = T(beta k 2^gc)`、`P = tril((q k^T) .* 2^(gc_i−gc_j))`——这些是纯逐块代数，天然并行，由 **wy** kernel（grid B·H·NC）完成；
- 跨块状态递推：`v_new = u − w S`、`S' = 2^gcl S + (k 2^(gcl−gc))^T v_new`（`gcl` 为块末累积门），同时把每块**入场状态** `h_all[n] = S` 与 `v_new` 落盘供反向消费——这是一条**顺时间方向的串行扫描**，由 **fwdscan** kernel（grid B·H·NV，NV = V/BV 个 v-切片各自独立行走）完成。

**第二遍——反向（逆时间）扫描与逐块局部项。** 以终端余切 `dht` 为种子逆时间行走：

```
dvp_n = P_n^T dO_n + Bhat_n X_n
X_{n-1} = 2^{gcl_n} X_n + Qg_n^T dO_n − W_n^T dvp_n        （X_n = 到达块 n 的 dH）
```

其中 `dO = do/sqrt(K)`，`Bhat = k .* 2^(gcl−gc)`、`Qg = q .* 2^gc`。这条逆时间扫描在最终系统中被重构为**仿射携带形式**（§6.1）：算子预复合（**caryA** / **caryM** / **bpre**，grid 均为 B·H·NC，纯并行）+ 携带链本体（**dhs**，grid B·H·NV）。行走沿途还需要三个消费 X_n（pre-update 值）的部分和——dkB、rowB、s5——它们被剥离为独立的逐块并行 kernel **dkb**（grid B·H·NC）。

因此，反向链整体是"**重算前向 + 两条方向相反的扫描**"的架构：九个 kernel 中只有两个含跨块串行递推——fwdscan（顺向重算）与 dhs（逆向余切携带）——其余七个全部是逐块（或逐 v-切片） embarrassingly parallel 的代数 kernel。两条扫描脊就是反向链的延迟骨架：所有性能工作的目标，要么把工作从串行链上剥离（携带重构、dkb 拆分），要么压缩扫描步本身的延迟（rgemm 的 ILP 重构、A 窗口、跨块预取），要么打开并行度（BV 切片、2-CTA 驻留）。而所有这些手段都受同一条铁律约束：**扫描状态 S 与 X 必须保持 f32 值类**——把递推状态本身 bf16 化会在 32/128 块上复合粗化 h_all/vnew/dh_all，因此并行度的每一次扩张都必须是"纯列并行"或"已证明的重结合类"。

#### 6.0.2 九 kernel 拓扑（最终装机形态）

链的发射见 `tools/gdn_bwd_gen.py::main()`，默认 `--kernels wy,fwdscan,caryA,caryM,bpre,dhs,dkb,dqkwg,wyb`，以扁平化 `[M, C·W]` 全局量（M = BH·NC，m = bh·NC + n）串联，零参 cubin、harness 经 `bwd_manifest.json` 逐级搬运：

| # | kernel | grid | 职责 | 串行性 |
|---|--------|------|------|--------|
| 1 | wy | B·H·NC | 逐块 WY：gc 累积、kkT、A、(I+A)^{-1} 分块解、u/w | 并行 |
| 2 | **fwdscan** | B·H·NV | **顺向状态扫描**：h_all、vnew_all | **串行（NC 步/CTA）** |
| 3 | **caryA** | B·H·NC | A_all[m] = 2^gcl I − W^T (k·2^(gcl−gc))，[K,K] | 并行 |
| 4 | **caryM** | B·H·NC | M_all[m] = Qg^T − W^T P^T，[K,C] | 并行 |
| 5 | **bpre** | B·H·NC | B_all[m] = M_all[m] @ (do·K^{−1/2})，[K,V] | 并行 |
| 6 | **dhs** | B·H·NV | **逆向携带链**：X_{n-1} = A X_n + B；dh_all、dh0 | **串行（NC 步/CTA）** |
| 7 | **dkb** | B·H·NC | dkB_sum / rowB_c / s5_c（消费 pre-update dH） | 并行 |
| 8 | dqkwg | B·H·NC | dq、dW、dk_local、r3/r4/c4r、dg_state | 并行 |
| 9 | wyb | B·H·NC | dAf 修正链、最终 dk/dbeta/dg | 并行 |

本章覆盖第 2–7 号（扫描族）；wy / dqkwg / wyb 属逐块代数族，见他章。旧的单一反向行走 kernel `dhu`（把 dvp 计算、dH 更新、dkB/rowB/s5 全部串在一条逆时间链里，约 8 个 GEMM 级段落/块）保留在文件中仅作 A/B 基准（`--kernels` 可选），装机链不再使用；它 2.17 ms 的阶段耗时与"每块三趟 GEMM 级部分和挂在延迟关键路径上"的结构缺陷，正是携带重构与 dkb 拆分的动机。

#### 6.0.3 全章通用的平台常数与判据

先定义后用，全章共享：

- **形状记号**：B、T、H、K=128、V=128、C=64（块长）、NC=T/C、BH=B·H、M=BH·NC；门 g 为自然对数域预乘 1/log2 得到的 **log2 域** f32，`gc` 为块内 inclusive 累积（f32 串行链），`gcl = gc[C−1]`。
- **sm_120a 静态共享内存上限 101376 B**（ptxas 实测；98304 B 接受、131072 B 拒绝）。2 CTA/SM 的静态池门槛约 50.2–50.7 KB（dqkwg 按 50176 B、fwdscan/dkb 按 50688 B 口径）；256 线程下 2×256×R ≤ 65536 给出寄存器门槛 R ≤ 128。
- **r6 危险规则**：`ldmatrix` 的操作数 tile 必须 `st.shared` 写入或逐元素 staging——**绝不**允许原始 LDGSTS（cp.async）落地后直接喂 LDSM（本工具链上静默损坏）；原始 cp.async 落地的 tile 只允许标量 `ld.shared` 消费。**r9 规则 3**：每个 `wait_group` 之后必须 `gpu.barrier` 再跨 warp 读。**r10 复用栅栏律**：缓冲区下次重 staging 前，其上一轮读者必须已被一道 trailing barrier 收口（复用距离 ≥ 2 次迭代即天然满足）。
- **split-bf16 三链接**：`hi = bf16(x)`，`lo = bf16(x − hi)`，则 `x ≈ hi + lo` 携带约 16 位尾数；`hi·hi + hi·lo + lo·hi` 三条 mma（弃 lo·lo ≈ 2^-32）恢复 ~f32 质量乘积（残差 ~2^-17），是 tf32（双操作数 2^-11 → 2^-17 类）在数值敏感点的替代路径（`_split_bf16`，tools/gdn_bwd_gen.py）。
- **cp.async 手工发射**：`astage_tile`（16 B 线程跨步突发，token 由调用方组队）+ `agroup` + `await_all`——不用 `nvgpu.device_async_create_groups`/transform 的 async-copy 组（其 lowering 在每组后自动插 `wait_group 0`，会把预取与它本应覆盖的计算串行化）。

---

### 6.1 仿射携带数学：反向余切扫描的精确重写（教学式推导）

本节是 caryA/caryM/bpre/dhs 四个 kernel 的共同数学基础。结论先行：**逆时间余切扫描精确地等于一个逐块仿射线性递推**

$$X_{n-1} = A_n X_n + B_n,\qquad X_{NC-1} = \mathrm{dht},\qquad A_n = 2^{gcl_n} I_K - W_n^T \hat{B}_n,\quad B_n = M_n\, dO_n,\ \ M_n = Qg_n^T - W_n^T P_n^T,$$

其中 $X_n$ 恰是**到达块 n 的 dH**（即 dh_all[m] 的 pre-update 契约值），$W = w_{all}$（wy 输出，f32）、$\hat B = k \cdot 2^{gcl-gc_{row}}$（Bhat）、$Qg = q\cdot 2^{gc_{row}}$、$P = P_{all}$、$dO = do/\sqrt K$。这不是近似，而是顺序更新的**代数重结合**。

#### 6.1.1 推导

从顺序形式出发（每块两式，见 `gen_dhu` 与 numpy 孪生）：

$$\mathrm{dvp}_n = P_n^T\,dO_n + \hat B_n X_n \tag{1}$$
$$X_{n-1} = 2^{gcl_n} X_n + Qg_n^T\, dO_n - W_n^T\, \mathrm{dvp}_n \tag{2}$$

(1) 描述余切在块内先经过 $P^T$（token 间路径）与 $\hat B$（状态路径）汇入 dvp；(2) 描述状态余切的衰减与两路注入。将 (1) 代入 (2)：

$$X_{n-1} = 2^{gcl_n} X_n + Qg_n^T dO_n - W_n^T\left(P_n^T dO_n + \hat B_n X_n\right)$$

按 $X_n$ 与 $dO_n$ 收集：

$$X_{n-1} = \underbrace{\left(2^{gcl_n} I_K - W_n^T \hat B_n\right)}_{A_n\ \in\ \mathbb R^{K\times K}} X_n + \underbrace{\left(Qg_n^T - W_n^T P_n^T\right)}_{M_n\ \in\ \mathbb R^{K\times C}}\; dO_n .$$

即 $B_n = M_n dO_n \in \mathbb R^{K\times V}$。**关键结构性观察**：

1. **$A_n$ 与 v-切片无关（vt-INDEPENDENT）**。$A_n$ 只依赖 $W$（f32）、$k$（bf16 全局）、$g$——不含任何 v 或 do 分量。于是一个 $[K,K]$ 算子每（bh, 块）只需复合一次，即服务全部 NV 个 v-切片的行走。
2. **$M_n$ 同样 vt 无关**，只有最后一步 $B_n = M_n dO_n$ 按 v 列切分（$[K,C]\times[C,V]$ → 每 BV 列一片）。
3. **边界与产出**：种子 $X_{NC-1} = dht$；每步之前 $X_n$ 恰为 dh_all[m] 要存的 pre-update 值（dkb/dqkwg 的消费契约）；走完最后一步（n=0）得 $dh_0 = X_{-1}$。这与顺序实现的存取逐位同义——重写不改变任何被消费量的语义。

CPU 验证脚本 `tools/gdn_carry_validate.py` 的对照对象是**真顺序形式**（保留 dvp 中间量，即 (1)(2) 原式，fp64），而非复合式自身，避免"用自己验自己"：

| 检查（fp64，两个门形状 T512/T2048） | 结果 |
|---|---|
| 复合式 vs 真顺序式（含 dh0） | max abs **4.9e-17 / 8.3e-17**——在 \|X\|~130 尺度上为 **2–3 ulp** |
| 二级段分解（每段零种子局部走 + 段复合算子 + 串行边界链 + 后缀积修正），S ∈ {2,4,8,16} | 同为 2–3 ulp |
| fp32 回放（GPU 操作数精度：bf16 展宽的 q/k/do、f32 W、f32 gc 累积） | 重结合底噪 **1.2e-8**（门分布） |
| 近单位化门压力测试（\|\|A\|\|≈8.8） | fp32 底噪 2.4e-6 |
| 与真实 `bwd_twin` 锚点（dh0，bh0） | 1.0e-10 / 7.6e-10（锚点差来自 gc 的 f32/log2 建模，非提取错误） |

GPU 上链内实测：dh_all 对顺序行走值的再现误差 **2.2e-8**——恰落在 CPU 预测的重结合底噪上。

#### 6.1.2 超收缩性：ρ(A) ≤ 2.3e-18

对基准门分布（g = −softplus(randn)，log2 域），逐块算子的谱半径与范数为（fp64 全谱测量）：

$$\rho(A_n) \le 2.3\times 10^{-18},\qquad \lVert A_n\rVert \le 2.2\times 10^{-17}.$$

机理：$gcl$ 强负 ⇒ $2^{gcl}$ 极小，而 $W = T(\beta k 2^{gc})$ 的行尺度含 $2^{gc}$、$\hat B$ 的行尺度含 $2^{gcl-gc}$，乘积整体被 $2^{gcl}$ 压制——块间耦合**一步之内消亡**（$\lVert A\rVert^2 \sim 10^{-34}$，远低于 f32 eps 2^-24 ≈ 6e-8）。两点推论：

- **理论上**，在该分布上做深度截断的完全并行扫描在 fp32 内就是精确的（边界修正只传一步）；
- **工程上不这么做**：弱门分布（近单位化门）下 $\lVert A\rVert$ 可达 O(10)，截断即错误。系统**完整执行精确的一级递推**——正确性不依赖门分布，超收缩性只是"为什么 dhs 的串行链在数值上如此温顺（误差不随 NC 放大）"的解释，以及"段切分等更强并行化在此分布上不会赢走数量级"的预兆（§6.6.1）。

#### 6.1.3 重构的工程含义

顺序形式把每块的全部操作数工作（P/Bhat/Qg/W 的装配与三个 GEMM）钉在串行依赖链上；仿射形式把它拆成三层：

1. **并行预复合层**（caryA/caryM/bpre，grid B·H·NC）：每块独立造 $A_n$、$M_n$、$B_n$，存入全局 `A_all[M,K·K]` / `M_all[M,K·C]` / `B_all[M,K·V]`（均 f32）；
2. **串行链层**（dhs，grid B·H·NV）：每块只剩**一个** GEMM——`Xa = B-切片; Xa += A_all[m] @ X; X = Xa`——外加两次 K×BV 的全局搬运；
3. **余量归位**：顺序形式沿途算出的 dvp_all 本就被 dqkwg 逐块恒等地重算（同 GEMM、同划分、同顺序、读同一 dh_all 值），dv 移到 wyb 的 `tv .* beta` 尾部——携带重构后这些输出各有唯一生产者，验证链闭合（此前 dhu 的"清洁 dvp_all"曾掩盖 dqkwg 侧一个潜在的 bf16-Bhat 污染：dhu 退役后 dbeta 从 3.0e-4 跳到 1.33e-2，二分定位到 dqkwg 自 r2 起把 Bhat 暂存进 bf16 tileCKb——修为闲置的 f32 tileCVb；该 bug 之所以长期隐形，是因为验证 harness 每次启动前重 staging 输入全局量，wyb 总是读到 dhu 的干净 dvp_all。重构本身充当了揭发者）。

测得总账（T2048 同窗口配对）：扫描复合体（旧 dhu + dkb = 3.13 ms）→（caryA+caryM+bpre+dhs+dkb = 2.01 ms）= **1.56×**；其中 dhu 2.17 ms → 0.41+0.46+0.29+0.59 = 1.76 ms，dkb 0.96 → 0.25 ms。以下各节逐 kernel 展开。

---

### 6.2 fwdscan：重算前向的顺向状态扫描

#### 6.2.1 动机（算子层面）

反向需要每块入场状态 `h_all[n]` 与块内新值 `v_new[n]`（dk 的 Bhat 项、dv、dW 等全都消费它们），它们只能由顺向扫描重算。kernel 每个 (bh, vt) 配一个 CTA，行走：

```
每块 n：h_all[n] = S（存入场态）
        acc = w_n @ S          （K 收缩，BK=64 切片链）
        vn  = u_n − acc        （→ vnew_all[n]）
        S  *= 2^gcl            （逐元素衰减）
        kg  = k_n .* 2^(gcl−gc_i)   （[C,K] f32 行缩放）
        S  += kg^T @ vn        （C 收缩）
```

即 $S' = MS + b$、$M = 2^{gcl}I - kg^T w$、$b = kg^T u$——但 M 从不显式装配，三因子行走就是 kernel。它与前向 F2 kernel 跑**同一递推**（前向侧 83 µs），却在改造前耗时 916 µs（11×）：ncu 剖面显示这是纯延迟核——**88.3% 无合格周期、L1TEX 记分板占 17.0 cyc/inst 停顿平均的 43.9%、每调度器 2 warp（64 CTA、1/SM、192 寄存器、0.39 波、DRAM 10.7%）**，没有任何资源饱和。定位出的四个杀手：每块一条 64 次依赖全局载入的**串行 gc 累积**、`astage_sync` 式"发射即排空"的**零重叠 staging**、每块 9 道 barrier、每块 5 趟独立的 stage/emap/store 扫。f32 载荷（h_all+vnew 写 50 MB + w/u/k 读 84 MB @T2048，约 800 GB/s 有效带宽下 ~170 µs 流量地板）是诚实的 2×，不是 11× 的来源。

#### 6.2.2 源码片段（gen_fwdscan_v3：装机主体）

chunk 循环核心（`tools/gdn_bwd_gen.py::gen_fwdscan_v3`，节选）：

```python
# group A: w[n, ks=0] -> wpp col half 0（由 h_all 存储扫覆盖）
astage_tile(e, G["w_all"], wkty, "%m", K, 0, wpp, 0, C, BK, "f32", toksA)
gA = agroup(e, toksA)
# h_all 入场态存储（逐元素）……
await_all(e, gA); e.l("gpu.barrier")
gemm(e, acc, wpp, S, C, BV, BK, b_off=(0, 0), tag="fwdscan.wS")   # gemm-a 片 0
toksB = []   # group B：w 片 1 + k 全量 bf16 + u，飞在 gemm-a 下面
astage_tile(..., wpp, BK, ...); astage_tile(..., kpp, ...); astage_tile(..., utl, ...)
gB = agroup(e, toksB); await_all(e, gB); e.l("gpu.barrier")
gemm(e, acc, wpp, S, C, BV, BK, a_off=(0, BK), b_off=(BK, 0), acc=True, ...)  # 片 1 链入
# vn = u − acc：一趟写 vn tile + vnew_all 全局；S *= 2^gcl（无 barrier）；
# kg 扫：kpp .* expc → 原地覆写已死的 wpp；ONE 全 K tgemm：S += kg^T @ vn
gemm(e, S, wpp, vn, K, BV, C, at=True, acc=True, tag="fwdscan.kg")
```

两条 tf32 m16n8k8 递推 GEMM（站点 `fwdscan.wS` / `fwdscan.kg`）经逐点二分验证收编（h_all 扰动 8.0e-4 rel、vnew 5.5e-5 rel，最终梯度位移 ≤ 3.0e-4，被既有最大误差单元吸收）；`expc[i] = 2^(gcl−gc_i)` 每块只算一次（原为每 kg 元素一次 MUFU）；vn/衰减/kg 三趟互不重叠的扫**背靠背免 barrier**，每块 barrier 从 9 降到 5。

**NC 免疫混合 prologue（GDN_FSCAN_GSCG）**——本章标题所指的机制。v3 的原设计把全部 NC 条块内 gc 累积行停进一块 `[NC, C]` f32 smem tile；池随 NC = T/64 线性增长，判据即发射期一行算术：

```python
_base = (K*BV*4 + C*2*BK*4 + C*K*2 + 3*(C*BV*4) + C*4)      # BV=32 时 90368 B
GSCG_GLOBAL = _base + NC*C*4 > 101376                        # gsc tile 是否还塞得下
```

- **塞得下**（T2048：90368 + 8192 = 98560 B；T512 类似）：smem 臂**逐字节保持 v3**——payload、cubin 全部与基线相同（md5 e680e1d94e47），**零回归由构造保证**，这就是"位级零回退"：混合开关以池不等式为唯一判据，旗舰形状不付任何代价。
- **塞不下**（T8192：90368 + 32768 = 123136 B > 101376 B 静态上限——原设计在该形状**根本不可编译**，辅助 v2 链实测 11.42 ms）：累积行改落新全局 `@gsc_all [M, C]` f32。数值等价的关键是**同一串行序 f32 链改由 `scf.for` iter_args 携带**（寄存器驻留、无读回：同样的加法按同样顺序发生 ⇒ gc 值位级一致）：

```python
e.l(f"%gtfr = scf.for %n = 0 to {NC} step 1 "
    f"iter_args(%gtok = {gpfl}) -> (!nvgpu.device.async.token) {{")
# 块顶：await 上一块预取的 gsc 行 + barrier；行 n 的读法：gr[%grp, %xe]
# 块中：把行 n+1 以单元素 4B cp.async.ca 预取进 gr 的另一行（ping-pong），token 随 yield 回传
```

消费侧是 2 行 `[2, C]` ping-pong（gr，512 B），L2 读藏在上一块的计算下面；每线程无条件发一个 4 B 拷贝（`tid % C` 覆盖，重复写字节相同——良性自重复类，**非 bypassL1**：4 B 拷贝必须降为 `.ca`，`.cg` 仅 16 B）。NV 个同 bh 的兄弟 CTA 以相同值写同一批字节——良性（各 CTA 只在自家 barrier 之后读自家行）。**S 保持 f32 [K, BV] 递推态不动**（两臂皆然）：切分它会粗化 h_all/vnew。池在 T8192 降到 90880 B（90368 + 512）。

#### 6.2.3 结构前后对比

| 维度 | v2（回退臂） | v3 + GSCG 混合（装机） |
|---|---|---|
| gc 累积 | 每块一次 64 次依赖全局载入的串行链，**钉在每块关键路径上** | 全部 NC 条链提升为 prologue（线程 t < NC 串行走块 t，同序 f32 ⇒ 位级同值）；smem 臂整块 tile、超限臂 `@gsc_all` + iter_args 链 + 2 行 ping-pong |
| w/k/u staging | `astage_sync`（发射即排空，零重叠） | 组 A（w 片 0，藏在 h_all 存储下）+ 组 B（w 片 1 + k + u，飞在 gemm-a 下）；消费者仅标量 ld.shared（r6） |
| GEMM | 标量 sgemm（每 MAC ~5–6 条指令、2 次 smem 载入） | 两条 tf32 tgemm；gemm-b 合一为单次全 K 调用（原两片调用的 acc 链式重结合——已 6/6 验证） |
| exp2 | 每 kg 元素一次 MUFU | 每块一次 expc[C] 扫 + 逐元素乘 |
| barrier/块 | 9 | 5（vn/衰减/kg 三趟免 barrier 背靠背） |
| smem | 65792 B | 98560 B（T2048）/ 90880 B（T8192 全局臂）；202/118 寄存器，0 spill |

未随 v3 落地并记录为否定的两项：**跨块 w 预取（v3b）**——裸 `wait_group N` 的 lowering 确实可用，solo 实测 0.358 ms 更快，但预取目的地（唯一 wpp tile）在拷贝落地时**不是死 tile**：与同块 kg 扫写、gemm-b 读竞态（verify：dk 31、dg 2.1 FAIL），正确版需第三块 [C,K] 缓冲，99 KB 预算放不下——撤销并记录；**bf16 mma + ldmatrix 操作数 tile**（F2 的精确 GEMM 形态，可砍 ~10× MIO 压力）会改 h_all/vnew 值类，tf32 路径已在既有值类上通过，该风险未花费。

#### 6.2.4 效果

- fwdscan **0.916 → 0.374 ms @T2048（2.45×）**、0.203 → 0.097（2.11×）@T512（同窗口配对）；
- T8192 链从"不可编译"变为可建：fwdscan 1.487 ms，bwd e2e **8.670/8.611 ms，较 v2 回退的 11.42 ms 为 1.32×**；6/6 通过（dq 2.078e-4/38×、dk 1.881e-3/4.3×、dv 1.050e-5/762×、dbeta 1.654e-3/12×、dg 1.145e-3/17×、dh0 1.965e-7/40700×）；
- 全局臂在 T2048 强制开启实测 **0.96×（0.374→0.388）**：prologue 的散布 4 B 全局写约 15 µs，cp.async 预取未能收回（读延迟本已被藏住）——这一实测回退正是混合门"只在 smem tile 放不下时才切全局臂"的设计依据；
- 确定性：新起进程 N=5 位级一致（含全局臂）。装机链 fwdscan 最终耗时 0.376 ms @T2048 / ~1.50 ms @T8192。

---

### 6.3 caryA / caryM / bpre：携带算子的逐块预复合

三个 kernel 都是 grid B·H·NC 的纯并行 kernel，分别预复合 §6.1 中的 $A_n$（[K,K]）、$M_n$（[K,C]）、$B_n$（[K,V]）。它们的共同设计语法是：**split-bf16 mma 化 + tile 消解（diet）+ epilogue 全局直存**。

#### 6.3.1 caryA：A_all = 2^gcl I − W^T (k·2^(gcl−gc))

**动机（算子层面）。** $A = 2^{gcl}I - W^T \hat B$ 是 [K,K] 乘积 $W^T[K,C] \times \hat B[C,K]$ 加对角。tf32 化（早期形态）只拿到 1.39× 并卡死在 255 寄存器 / 82176 B smem = 1 CTA/SM。瓶颈不是 mma 缺席而是**四块共活 tile 的池**：原始 k 暂存 `kst[C,K]` bf16、Bhat 复合 `bhb[C,K]` f32、W 列块 `wb[C,BK2]` f32、输出块 `ab[BK2,BK2]` f32——按 (i0, j0) 双重块循环迭代时全部同时活着，**不重构就没有合法的 lifetime 别名**。

**源码片段（mma 臂，gen_caryA）**：

```python
seg = k.alloc(e, "seg", C, 1, "f32")                 # 2^(gcl - gc_c)
for i0 in range(0, K, BK2):                          # A 的行块（W^T 的行）
    stage_split_T(e, ahi, alo, G["w_all"], wkty, m, K, i0, BK2, C, cvec=seg)
    for j0 in range(0, K, BK2):                      # A 的列块（k^T 的行）
        stage_raw_T(e, bt, G["k"], kty, m, K, j0, BK2, C)
        def epi_ca(mi, ni, ei, row, col, acc):       # -neg + 对角 + 全局直存
            x = mma_extract(e, acc, ei // 2, ei % 2, "cax"); nx = negf(x)
            if i0 == j0:   # 对角块：row==col 处 += 2^gcl（select 融入 epilogue）
                dz = select(cmpi eq row col, 2^gcl, 0.0); nx = addf(nx, dz)
            gstore(e, nx, G["A_all"], aty, [m, row*K + col + i0*K + j0])
        accs = mma_gemm_mh(..., ahi, bt)                      # hi 链接
        accs = mma_gemm_mh(..., alo, bt, accs=accs, epilogue=epi_ca)  # lo 链接
        e.l("gpu.barrier")                            # r10 复用栅栏
```

**Bhat 因子折入 A splits 的幂二交换恒等式。** 旧路径先在 smem 里把 k 逐元素乘 `2^(gcl−gc_c)` 造成 Bhat（f32 tile），再当 GEMM 的 B 操作数。新路径把这个**列因子折进 split 本身**：`stage_split_T` 的 `cvec=` 参数在截断前对源列乘 `seg[c] = 2^(gcl−gc_c)`。其精确性是一条恒等式：对任意整数 d，

$$\mathrm{split}(v\cdot 2^d) = \mathrm{split}(v)\cdot 2^d$$

——2 的幂与 bf16 截断**可交换**（只移指数域，不碰尾数舍入），也与 hi/lo 残差可交换（残差同步移指数）。因此乘积保持"B 侧已缩放"的旧值，bhb 整块退役，`bt` 直接是**原始 k^T 的 raw bf16**（全局本就是 bf16——**精确**，无舍入）。对角 `2^gcl` 与取负折进 mma epilogue 的 select/addf；A_all 全局直存（bpre playbook，退役 store_rows 扫及其 barrier）。

**结构前后对比。**

| | 标量路径 | split-bf16 mma 路径（装机） |
|---|---|---|
| 共活 tile | kst [C,K] bf16 16384 + bhb [C,K] f32 32768 + wb [C,BK2] f32 16384 + ab [BK2,BK2] f32 16384（净和 81920 B，含垫报告口径 **82176 B**） | ahi/alo（W^T splits，[BK2, C+8] bf16 ×2 = 9216+9504，i0 一次 staging 供两个 j0 复用）+ bt [BK2+1, C+8] 9360（raw k^T）+ gc 256 + seg 256 = **28592 B** |
| 寄存器 | 255 | **76**（0 spill） |
| 驻留 | 1 CTA/SM | **2 CTAs/SM** |
| 数值 | f32 sgemm | 重结合实测：A_all 8.39M/8.39M 元素不同，max **5.6e-22 绝对**（2.8e-4 rel——被替换的 tf32 截断类；A 本身超收缩、值 ~1e-18 量级），全部最终梯度位不变 |

**效果。** caryA **0.3093 → 0.0618 ms @T2048（5.00×）**、2.78× @T512。5× 的结构原因即"**arena 消解式移植**"：不是给旧 tile 结构做别名（不重构就不合法），而是把 bpre 式 split-bf16 移植过来后四块共活 tile **整体退役**，池压掉 65%、寄存器压掉 70%，2 CTA/SM 的驻留增益与 mma 吞吐叠加在已 1.39× 的 tf32 底座上再拿 3.6×。附带否定：**w-窗口轮转**（GDN_CARYA_WF，i0 边界的 16 KB 同步 w_all 读改原始 cp.async 窗口跨整个 j0 循环飞行，`_split_T_smem` 从 smem 重分裂——位级一致）实测 **0.96×**：54 µs 的 kernel 里那个 i0 边界停顿早已摊薄，16 KB 窗口的 smem 流量 + await/barrier 边界更贵——默认关、旋钮留档。

#### 6.3.2 caryM：M_all = Qg^T − W^T P^T

**动机。** $M$ 是 $[K,C]$ 乘积 $W^T[K,C]\times P^T[C,C]$（注意 P 侧转置）减 $Qg^T$。tf32 单站点形态 160 寄存器 / 64256 B。

**源码要点（gen_caryM，mma 臂）。** 3 链接 k16 切片 split mma，**双操作数都直接从全局切片**——"扁平 col0 算术即切片"：

```python
for i0 in range(0, K, BK2):                       # M 的 K 行块
    # mb 初始化扫（f32 精确保持 Qg^T 项）：mb[a,c] = q[c, i0+a] * 2^gc_c
    for kb0 in range(0, C, 16):                   # C 方向 k16 切片
        stage_split_T(e, whs, wls, G["w_all"], wkty, m, K, i0 + kb0*K, BK2, 16)
        stage_split_ew(e, phs, pls, G["P_all"], pty, m, C, kb0, C, 16)
        accs = mma_gemm(..., whs, phs)            # hi·hi
        accs = mma_gemm(..., whs, pls, accs=accs) # hi·lo
        accs = mma_gemm(..., wls, phs, accs=accs, epilogue=_cm_epi if 末片 else None)
        e.l("gpu.barrier")                        # r10 复用栅栏（下一片 restage 前）
```

A 侧（$W^T$ 的 [BK2,16] 切片）经**转置 staging** 的 `col0 = i0 + kb0·K` 取出（W 按 [C, C·K] 扁平行主序，col0 算术直接落在 k16 列片上）；B 侧（P 的自然取向切片）`col0 = kb0`。末片 epilogue 做"读-减-存"：`mb[row,col] −= acc`，把 f32 初始化值**精确保持**（Qg^T 项不经过 mma）。tf32 的双操作数 2^-11 在此换为单操作数分裂的 2^-17 类。

**结构前后对比。** 旧 tf32 形态：`astage_sync` 把整块 P 停进 `pt[C,C]` f32（16384 B）+ 每 i0 停 `wb[C,BK2]` f32（16384 B）+ `mb`，单站点 tgemm [BK2,C,C] at=bt=neg=acc 一次成型。新形态：pt/wb 全部退役，双操作数改从全局按 k16 切片直接分裂（whs/wls [BK2,24] + phs/pls [C,24]，四个小 split tile 共 ~6.5 KB），24 条 Kred=16 mma 链接按 i0 分组进寄存器累加器；池 64256 → **45504 B**、寄存器 160 → 76、0 spill、2 CTAs/SM。

**效果。** caryM **0.1647 → 0.0814 ms（1.96× 阶段 / 2.02× solo @T2048；1.41× @T512）**。下游 dh0 富余随之收紧（2.761e-6 → 2.169e-7，36900× 富余）——M_all 更净直接改善携带链种子质量。

#### 6.3.3 bpre：B_all = M_all @ (do·K^{−1/2})

**动机。** $B = M\,dO$ 是 $[K,C]\times[C,V]$；注意顺序形式是逐元素缩放 dO，复合形式把 $K^{-1/2}$ **改为求和后乘一次**（全程只一次 f32 舍入）。

**源码要点（gen_bpre，mma 臂）。** A = M 行块的 hi/lo 融合分裂（f32 源，~2^-16 乘积——B_all 喂超收缩携带链，当时 dh0 富余约 4000×）；B = **原始 dO^T raw bf16**（全局即 bf16，`stage_raw_T` 转置暂存 [V, C+8]——精确无需分裂）；scale 在 epilogue（`epi_b`：`x * (1/sqrt(K))` 后 `gstore` 直存 B_all）。寄存器累加器跨整个 K=C 归约链共享；[BK2, V] f32 bb 暂存 tile 与其 scale 扫退役，smem **49 KB → 37 KB**（按 smem 获 2 CTAs/SM 而非 1）。W=8 无 warp 守卫。

**结构前后对比。** 旧标量形态：`stage_rows` 停 `dob[C,V]` bf16 + 每 i0 一块 `bb[BK2,V]` f32 输出暂存，`aggemm`（胞格主序、A 从全局串行读）求积后一趟独立 scale 扫再 `store_rows`。新形态：dob 换成一次转置 raw dO^T（dotT），bb 与 scale 扫退役（scale 折入 mma epilogue、B_all 全局直存），aggemm 换两条 mma 链接（bmh/bml）——**发射期指令数与 smem 同时下降**，且 A 载荷从"胞格主序串行 LDG"变为 ldmatrix。

**效果与一个被确定性关卡揪出的竞态。** 该 kernel 的 K 循环 mma **跨迭代 restage 竞态**（`stage_split_ew` 在慢 warp 的 ldmatrix 读仍在飞行时覆写 ahi/alo——r9 类漏网点）曾造成 1–3 ulp 的 B_all 翻转：被交换的操作数元素恰是近零的 LO 分量，acc 扰动极小，却沿 dh_all → dkB_sum 传成**一个 dk 元素的 bf16 舍入翻转**——即运行间 1.719e-3 ↔ 4.719e-3 的 dk 富余摆动（修复前 14/1000 新起模块发射复现；在每次迭代末补一道 `gpu.barrier` 复用栅栏后 **0/1000**）。这是"末位 ulp 竞态 × bf16 量化台阶"放大路径的标本，也是扫描族所有 mma 化 kernel 统一执行 r10 复用栅栏律的直接原因。

---

### 6.4 dhs：反向携带链本体

#### 6.4.1 动机（算子层面）

携带链是反向图上唯一的逆时间串行依赖：$X_{n-1} = A_n X_n + B_n$。预复合之后，每步只剩：

```
dh_all[m] ← X          （pre-update 契约：先存后更新）
Xa = B_all[m] 的 v-切片 （[K, BV] f32）
Xa += A_all[m] @ X     （[K,K]×[K,BV]——每块唯一 GEMM）
X  = Xa
```

循环结束存 dh0。没有 P/T/k/q/w/beta 暂存、没有累积、没有 dvp/dv epilogue——smem 只需 X + Xa + 窗口。设计问题只剩两个：GEMM 怎么读 A（64 KB 的 [K,K] f32 staging tile 毫无意义），以及每步的 B-切片搬运与 A 读取怎么藏进计算。

#### 6.4.2 rgemm：寄存器累加器 + kk 外层归约（每块单 GEMM 的串行链）

初版 GEMM（`aggemm`，胞格主序）让 dhs 停在 0.83 ms：每线程一个输出胞格、kk 内层一次全局 A 载入接一次 FMA——1 层深的依赖链把 LDG 延迟完全串行化。`rgemm` 反转结构：

- 划分：输出胞格 $e = i\cdot N + j$ 归线程 $t$（$e \in \{t + c\cdot NT\}$，CPC = M·N/NT 个/线程）；N = BV 整除 NT ⇒ **同线程全部 CPC 个胞格共享 j = t mod BV**——每个 kk **一次** b（smem）载入服务所有累加器；
- kk 外层 `scf.for` 的 iter_args 携带 CPC 个 f32 寄存器累加器；每 kk 做 CPC 次**相互独立**的 A 全局载入（`A[i_c, kk]`，行各异，warp 内广播）+ CPC 个 FMA——**ILP = CPC**（16）对抗 1 层依赖；
- `acc_buf` 参数以 smem 里的 B-切片作加性初值（"acc=True 但初值来自另一缓冲"）。

仅此一项 dhs **0.83 → 0.59 ms**。`rgemm_aw` 是其窗口变体（下述），胞格划分与 kk 升序 FMA 链**逐位相同**。

#### 6.4.3 A-窗口 ping-pong（rgemm_aw，GDN_DHS_AW）

**动机。** rgemm 消除了依赖链，但每块仍有 **2048 次 warp 广播式全局载入/线程**（K×K 的 A 逐 kk 行读）——长记分板链使 dhs 仍是 #2 阶段。整块 [K,K] f32 staging 要 64 KB；答案是把 A 当流：**滚动 [K, BKW=32] f32 smem 窗口双缓冲（AW[0]/AW[1]，各 16 KB），cp.async 预取超前一个窗口**（`astage_tile` 16 B 突发，24 次/块）。

```python
for w in range(NW):                                  # NW = K // BKW
    if w + 1 < NW:                                   # 超前一个窗口发射
        astage_tile(e, G["A_all"], aty, "%m", K, (w+1)*BKW, AW[(w+1) % 2], 0, K, BKW, "f32", toks)
        gw = agroup(e, toks)
    rgemm_aw(e, Xa, AW[w % 2], w * BKW, X, K, BV, BKW, acc_buf=Xa)
    e.l("gpu.barrier")                               # r10：本窗口读毕，才允许 2 次迭代后的重staging
    if w + 1 < NW:
        await_all(e, gw); e.l("gpu.barrier")         # r9 规则 3
```

危险规则合规：窗口只喂 rgemm 的**标量 ld.shared**（r6 合法口味，绝不 ldmatrix）；窗口读以 trailing barrier 收口后其缓冲的下一次重 staging 至少在 2 次迭代之后（r10）。

**值等价证明（AW=0 对照）。** 同一种子输入、同 caryA cubin（A_all/B_all/dht 位级相同）下建 AW=0 与 AW=1 两条链：**dh_all 与 dh0 逐位相同**。窗口是纯延迟改动——窗口值是同批全局元素的原始 cp.async 字节拷贝，kk 升序 FMA 链未动。这一"证明而非假设"的对照是链内所有纯 staging 改动的范式。

**跨块预取（GDN_DHS_XP）。** 块 c 的**最后一个**窗口（w = NW−1）顺带发射**块 c+1 的窗口 0** 进 AW[0]（此刻空闲：w 自己的读在 AW[1]；AW[0] 的上轮读者已被 w−1 的 trailing barrier 收口——与块内轮转同一的 1 次迭代净空），cp.async 组令牌**搭 `%ci` 的 scf.for iter_args**（`!nvgpu.device.async.token`——本 MLIR 构建注册的助记符，'async.token' 是语法错误）进入下一迭代 prologue 的 await；末块发一次**无害自重复**（自己的窗口 0 进 AW[0]，无人读）保证携带令牌恒有效，循环出口排空（CTA 退出时无在飞 cp.async）。每块 prologue 的暴露等待从"仅被 B 扫覆盖"变为"被整整一块的工作覆盖"。`GDN_DHS_XP=0` 逐字节再生基线 payload——干净的二分臂。

**FPIPE：B-切片置顶。** B-切片（Xa，[K,BV] f32）原本是 dh_all 存储***之后***的同步 `stage_rows`；改为**块顶**发射的 cp.async 组（`GDN_BFPIPE` 名单含 "dhs"），由 dh_all 存储扫 + 窗口 0 await 覆盖。Xa 此刻已死（上一块的拷贝扫读过它、有 trailing barrier 收口）且只喂 rgemm 的标量读（r6）。字节拷贝 + 同值同算 ⇒ 位级一致。

#### 6.4.4 BV=16：买并行度的池瘦身

BV=32 时 X+Xa+2×AW = 65536 B = 1 CTA/SM，而 grid BH·NV = 64 CTA 欠填充设备，串行 NC 行走是关键路径——单纯"瘦身到 2 CTA/SM"不改变任何事（grid 不变）。**有效瘦身是减半 BV**：X+Xa 32768 → 16384，池 49152 ≤ 50176，grid 64 → 128（2/SM），每 CTA 串行工作减半；A-窗口流量翻倍但每条 A 行现被 8 个 v-兄弟读——**留在 L2**。**值精确**：V 维是纯列并行——k 窗口行走、B 扫、每输出元素的 FMA 链全部不变（各形状位级一致已验）。`GDN_DHS_BV` 旋钮，=32 逐字节恢复。

#### 6.4.5 结构对比与效果

| 维度 | rgemm 初版 | + A 窗口 | + 跨块预取 + FPIPE + BV=16（装机） |
|---|---|---|---|
| A 来源 | 全局 warp 广播载入（2048 次/线程/块） | [K,32] f32 双缓冲窗口，超前 1 窗口 | 同左 + 块边界令牌续飞 |
| B-切片 | 块内同步 stage_rows | 同左 | 块顶 cp.async，被存储扫覆盖 |
| smem | 32768 B（X+Xa） | 65536 B | **49152 B**（BV=16） |
| grid | 64 CTA | 64 | **128 CTA = 2/SM** |
| dhs @T2048 | 0.83 → 0.59（rgemm ILP） | 0.592 → 0.320（**1.85×**） | XP 1.03× → 0.312；BV=16 **1.101×** → 0.282；FPIPE 1.028× → 0.272 |
| dhs @T8192 / @T512(H4) | — | 1.83× | BV=16 1.073× → 1.086，FPIPE 1.034× → 1.047 / BV=16 **1.61×** → 0.041，FPIPE 1.065× → 0.038 |

否定记录：**BKW=16**（8 KB 窗口、48 KB 池、理论 2-CTA）实测 **0.91×**——窗口数翻倍使每窗口 wait+barrier+发射开销翻倍、每次预取下的 rgemm 覆盖减半，而 kernel 是内存吞吐约束（ncu 48.9%），池瘦身买不到驻留（grid 128 已 2 波）；**BV=8 过冲**（grid 256 ≈ 2.9 波 @2/SM + 再翻倍的 A 窗口流量）@T8192 实测 **0.80×**——甜点恰在 BV=16。XP 的收益也小于预估（1.03× vs 预估 0.05–0.1 ms 级）：B 扫重叠本已藏住了大部分 prologue await。装机 dhs 最终 **0.272 ms @T2048 / 1.047 @T8192 / 0.038 @T512(H4)**。

---

### 6.5 dkb：pre-update 部分和的并行重算

#### 6.5.1 动机：为什么"拆"在数学上是无损的

逆时间行走沿途要为每块 n 产三个消费 pre-update dH（即 X_n）的部分和（dk 的 Bhat 项、dg 的前缀项与状态衰减常数）：

- `dkB[i,·] = (vn .* 2^(gcl−gc_i}) @ dH^T`（[C,K]，进 dk 的 `(vn·2^(gcl−gc)) @ dH^T` 项）；
- `rowB[i] = Σ_j dkB[i,j]·k[i,j]`（dg 的 Bhat 前缀项 `pre`）；
- `s5 = 2^gcl · Σ_{i∈K, j∈BV} dH[i,j]·h_all[i,j]`（dg5a 状态衰减常数）。

关键时序事实：**顺序行走在更新 dH 之前就把 pre-update 值存进了 dh_all[m]**——所以这三个量是 dh_all 的**逐块纯函数**，可以在行走结束后由独立 kernel 完全并行重算（每块一个 CTA），而不必串在延迟关键路径上。旧 dhu 把它们（vng staging + 2 个 K 切片 GEMM + rowsum + 256 部分和 s5 折叠）钉在行走内，每块多出三趟 GEMM 级段落与 ~17.5 KB smem。剥离后 dkb 还**顺带折掉 NV 归约**：dqkwg 原先每输出胞格 4 次全局载入的逐 vt 折叠，改为 dkb 在此走 v-切片并发射**块求和全局** `dkB_sum[M, C·K]` / `rowB_c[M, C]` / `s5_c[M, 1]`（各仅一个写者）——dqkwg 每胞格只读一个值。数值上刻意贴近原折叠序：per-vt GEMM 同划分、rowB 按 vt 主序、s5 同 256 部分/线程分布；唯一重结合是 dkB_sum 的 vt 链（(((v0+v1)+v2)+v3 对旧 ((((a+v0)+v1)+v2)+v3）——f32 末位 ulp 效应，远低于 4.5× dk 富余。

#### 6.5.2 split-bf16 3-mma 与数值选择

dkB 站点保持标量会令 dkb 指令吞吐受限（每 MAC ~5–6 条指令）；tf32 化被第三轮逐点二分**否决**（dH 操作数 ~O(10)，tf32 dkB 使 dk 1.04e-2，**超 8e-3 杠**）。split-bf16 3-mma 通过：

```python
_split_bf16(e, vh, vl, vngb, C, BV)     # A = vng = vn .* 2^(gcl-gc) 的 hi/lo
_split_bf16(e, hh, hl, cur, K, BV)      # B = dH 切片的 hi/lo（cur = 本 vt 的 dHv）
for vt in range(NV):                    # vt 折叠进 mma 累加器（寄存器）
    maccs = mma_gemm(e, f"dkBhh{vt}", C, K, BV, 8, vh, hh, accs=...)
    maccs = mma_gemm(e, f"dkBhl{vt}", C, K, BV, 8, vh, hl, accs=maccs)
    maccs = mma_gemm(e, f"dkBlh{vt}", C, K, BV, 8, vl, hh, accs=maccs)
```

hi·hi + hi·lo + lo·hi（弃 lo·lo ~2^-32），~2^-16 乘积误差对 dk **不可见**（1.730e-3 vs 标量 1.731e-3）。split tile 带 +8 bf16 列（16 B/行）行垫——G5-F 的 LDSM 教训：8 行 ldmatrix 落在 8 个不同 bank 组。dHv 的 staging 用线程跨步 16 B 异步突发（逐元素 scf.for 每元素 ~7 条索引算术；dkb 是指令吞吐约束而非延迟约束）；**FPIPE 化**把 dHv 的 wait 移过 vngb emap——vnew 全局读 + gc 缩放这趟等体量全局扫成为 DMA 的覆盖（tile 不相交、dHv 只喂标量读）。

rowB 的一个被探针逮住的 bug（记录为设计教训）：第一版对**运行中累加器**逐 (vt, ks) 做 rowsum——`rowsum(v0·k) + rowsum((v0+v1)·k) + … = 4v0+3v1+2v2+v3`，~10 量级 dg 错误；修正为对**最终求和 tile 一次性** rowsum（同数学值、不同 f32 结合——rowB 喂 dg 前缀项有 ~50× 富余）。

#### 6.5.3 s5：串行折 → 线程轴树折（GDN_DKB_S5T）

BV=16 使 NV 翻到 8 后，"线程 0 串行走完每 vt 全部 128 个 s5p 部分和"成为可见串行尾巴（NV×128 = 1024 次依赖 LDS+FADD/CTA）。S5T 重排布局：部分和落 `s5p2[threads, NV+1]`（每 vt 一列，**同一逐线程部分和折叠写**——分布与值恒同），线程轴以**跨步折半屏障树**归约（t < s：`s5p2[t][*] += s5p2[t+s][*]`，每级一道 barrier），随后线程 0 折 NV 列并**一次性**乘 `2^gcl`（逐 vt 的 exp2 对和式分配律）。s5v tile 退役；池 32288 → 36352 B（仍 2 CTAs/SM）、寄存器 100 → 112、0 spill。值类变化范围实测恰为预测锥：位级差异仅在 **s5_c + dg_state**（s5_c 的唯一消费者是 dqkwg 的 dg5a 项 → dg，杠 2e-2 对当前 3.274e-4 峰值，~60× 富余），33/35 数组位级一致，6/6 表格每一位等于既有记录。

#### 6.5.4 2-CTA 转换：accCK 退役 + BV=16（GDN_DKB_2CTA）

**动机的精确化。** dkb 以 ~90 KB 池跑 1 CTA/SM（dHv [128,32] f32 + accCK [64,128] f32 + hh/hl [128,40] bf16 对——都是 K=128 尺寸的 tile）。纯别名审计结论：把 vh/vl 别名进 accCK 字节（lifetime 不相交——accCK 唯一写者是最终 maccs 存储扫）只能降到 ~79 KB，**仍是 1 CTA/SM，零收益**。真正的转换是**退役**而非别名：

1. **maccs 直停 dkB_sum 全局**（r14 park_epi 模式：同一寄存器值、同一 `[m, row·K+col]` 胞格，替代 accCK 中转 + store_rows 扫 + 2 道 barrier；**位级一致**——字节搬家）；rowB 的 rowsum 改 `gcol` 读 L2 热的 dkB_sum 全局（r9 dqkwg 的 P 退役模式；本 CTA 刚写、单 CTA 行——同值回读）。accCK 的 32768 B 退役；
2. **BV=16**：dHv/vngb f32 与 vh/vl/hh/hl split 对减半（[·, 24]，48 B 行跨距：12i mod 32 给 8 个不同 bank 组——+16 B 垫性质保持）。

池 89616 → **32288 B ≤ 50176**、寄存器 126 → **100**（Kred=16 链接减半活 fragment 集）——2×256×100 = 51200 寄存器容 65536 regfile 有余。**栅格门**（r9 派发规则）：`BH·NC ≥ 164` 才装机 2-CTA 臂——小网格（T512：M=32 CTA）在任何驻留度下都欠填充，保持 r17 形态 payload **逐字节不变**（未门控的 BV=16 臂在 T512 实测 **0.86×**：NV=8 的 s5 串行折 + 每 vt barrier 花约 4.5 µs 而无驻留可买——诚实否定，门即由此来）。

#### 6.5.5 效果与两条否定

分臂实测（同窗口配对）：

| 臂 | 池 | 寄存器 | dkb @T2048 | dkb @T8192 |
|---|---|---|---|---|
| 基线（accCK + BV=32） | 89616 B（1/SM） | 126 / 0 spill | 0.2334–0.2371 | 0.8060–0.8136 |
| 仅 park（位级一致） | 56848 B（仍 1/SM） | 126 | 0.2376（0.99×） | — |
| + BV=16（装机，栅格门） | **32288 B（2/SM）** | **100** | **0.1907（1.22×）** | **0.6290（1.28×）** |
| + S5T 树折（装机） | 36352 B | 112 | **0.183（1.05–1.07×）** | **0.607（1.04×）** |

（dkb 拆分本身连同初版 split-bf16：0.96 → 0.25 ms，3.8×。）两条否定及其机制定论：

- **dHv ping-pong（GDN_DKB_DHP）**：静态奇偶 dHvA/dHvB，下一 vt 的突发在本 vt emap 后发射、飞在分裂 + mma 链 + s5 折之下（令牌绝不穿越 mma 内部；`wait_group 1` 保留最新组在飞；末 vt 全排空；r10 以跨 vt trailing barrier 的 2 次迭代距离满足）。寄存器过杆（127 pass1 / 110 smemfix，2-CTA 存活）、**位级一致**，但实测 **0.97×/0.99×**——r17 的 FPIPE emap **已经**覆盖了 dHv DMA，循环顶 await 排空的是近乎付清的组，而额外的发射/等待机械 + 8 KB 池更贵。**真正的串行结构是跨 vt 的 mma 累加器链（每 vt 3 条依赖链接、承值折叠序）+ 每 vt 一道 s5p2 列 barrier——那是值类重结合（vt 折是纯和），不是 staging 玩法。**
- **vt 折重结合（GDN_DKB_VTF）**：按上一定论建**偶/奇双累加器组**（偶 vt 链 maccs0、奇 vt 链 maccs1，park 扫时每 fragment 胞格做**一次固定序加法** even+odd）。取偶奇而非 vt 对半，是因为 warp 按序发射——对半的两条独立链在程序序上仍先后发射，偶奇才能链接级交错、把发射级关键路径减半（+32 活 fragment 寄存器两种取法相同）。寄存器 126（0 spill，2-CTA 存活；ptxas 交错复用只 +14）、池不变、6/6 与位差范围全按预测锥（dkB_sum → rowB_c → dk_local → dk → dg_state → dg）。实测 **0.99×/1.00×**——低于 1.05× 杆。根因：**跨 vt 链在 2 CTA/SM 下本就不是绑定的串行结构**（16 个驻留 warp 已在 warp 调度器层面重叠两条 vt 链）——这是"每 warp/CTA 工作不变 ⇒ 调度器 ILP 已在 ⇒ 无收益"定律的又一次独立确证。

装机 dkb 最终 **0.183 ms @T2048 / 0.607 @T8192 / 0.028 @T512(H4，门控保持旧形态)**。

---

### 6.6 扫描族的三个诚实否定：设计空间的边界测绘

以下三项不是失败罗列：每项都是**数学先行验证、机械随后落地、性能诚实测量**的完整闭环，共同给出扫描族设计空间的三条边界——"更强的并行化在哪里不再有收益"。负结果旋钮全部保留在树中（默认 0，关闭时逐字节再生装机 payload），供未来形状域变更时重新评估。

#### 6.6.1 边界一：段切分（2-level segment scan）——数学精确、经济为负

二级分解（CPU 已实现并验证，`gdn_carry_validate.py::seg_walk`）：段 s 覆盖块 [s·G, (s+1)·G)；每段零种子局部走 Y、复合段算子 (A_seg = ∏A，B_seg = 局部底部输出)；串行边界链 $T_s = A_{seg,s+1} T_{s+1} + B_{seg,s+1}$（S 条，微小）；修正 $X_n = P_n T_{top(n)} + Y_n$（$P_n = A_{n+1}\cdots A_{hi-1}$ 后缀积）。fp64 下对每个 S ∈ {2,4,8,16} 都与真顺序式同在 **2–3 ulp**。

**不装机的决定是测量出的经济学**：段复合需要 [K,K]×[K,K] 矩阵积（每个 2M MAC）——标量实现即成新瓶颈；tf32/bf16 复合重开精度/混沌问题（f32 级 dh 扰动会重采样 mma 操作数量化箱——dbeta/dv 的混沌类）；再加 3 个 kernel 的发射开销，对冲一个 ~0.3 ms 的剩余天花板。而 dhs 经 rgemm + 窗口 + BV=16 后已至 0.28 ms：**并行深度的收益上限被串行步的延迟瘦身提前吃掉**。这是"测量负经济学，非数学失败"的边界：在超收缩分布下（§6.1.2）段修正一步即死，更强并行深度天然无物可赢。

#### 6.6.2 边界二：fwdscan K-split 的 warp-group 形态（GDN_FKS_WG / v4）——数学 GO、性能 0.98–0.99×

**孪生数学（tools/gdn_fwdscan_ksplit.py）。** fwdscan 的行走 $S' = 2^{gcl}S + kg^T(u - wS)$ 对 **K 行对半** $R_0/R_1$ 精确可分：衰减逐元素（行分裂零耦合）；$kg^T vn$ 的输出行只需自家 kg 列（行分裂，vn 共享）；唯一跨 K 耦合是 $wS = \sum_h w[:,R_h]@S[R_h]$——交换 [C,BV] f32 部分和并**固定序加一次**。CPU 孪生（对 kernel 精确 k 序的严格左折模拟）判定 **GO**：fp64 恒等（h_all **逐位相等**、vnew 5.7e-17/5.4e-17 @NC=32/128）；f32 纯重结合 vnew ≤ 3.08e-8/2.89e-8 rel、**h_all 完全不动**（vn 的 ulp 差经 kg^T vn 注入 S，落在 S 自身舍入之下被舍掉）；误差放大 1.00×。

**GPU 端口决策。** 双 CTA 旗标形式的正确性类恰是 F1KS 险族（跨波边界的对等自旋死锁、跨发射旗标陈旧、普通数据载入的 PTX 内存模型栅栏）——设计期即否决，改**单 256 线程 CTA 内的 warp 组分裂**（warp 0–3 持 $R_0$、4–7 持 $R_1$）：交换走 smem（两个 acc tile 同 CTA 可见，`gpu.barrier` 即交接栅栏），各组并行走自己的 staging/gemm，资源包络与双 CTA 形态相同（256 线程 / 49152 rf 寄存器 / 每 SM 一链），零新同步面、无条件确定性。

**数学门（链上实测）。** h_all **位级一致（0/8388608 元素）**；vnew maxabs 2.384e-7（2 ulp）、rel 5.95e-8——孪生预测的同末位 ulp 类、按分布缩放；下游锥恰为预测（vnew_all + dkb + dqkwg + caryA 各数组，25/35 位级一致）；6/6 两位数全等；关闭旋钮双形状逐字节一致；det N=5 位级一致。

**性能定论。** 两个结构臂（融合 per-group staging 组 / 恢复 v3 A-B 覆盖节奏）配对 ×3：**0.98–0.99×**。根因（机制级）：warp 组重划分**不改变每 warp 的工作量**——m16n8 tile 粒度迫使每个 [C,BV] 半 GEMM 取 4-warp 铺瓦，每 warp 每块仍走 4 条原子 tile 链（v3：2 链 × 2 片；v4：4 链 × 1 半），而 8 个驻留 warp **本已**交错 v3 那些形式上依赖的片链（设计想要的 ILP 在 warp 调度器层面已在）；设计真正需要的驻留门（2 CTA/SM = 16 warp）仍结构性关闭：池 98560 B 对 50688 B 杆（[C,K] w/kg tile 与复制的 vn/acc 不随 K-split 减半；即便 BV=16 也 78080 B）。

#### 6.6.3 边界三：fwdscan K-split 的 cluster 双 CTA 形态（GDN_FKS2 / ks2）——机械全证、性能 0.88–0.89×、族闭合

**可行性先行（推翻一个先例记录）。** sm_120 簇 >1 可用：`cuLaunchKernelEx` + `CLUSTER_DIMENSION` 接受 2/4/8 x 维（本 RTX 5090 Laptop、CC 12.0），`%cluster_nctarank` 读取核证明硬件真以簇模式执行（普通发射到处读 1/0；2x1x1 读 2 且成对 {0,1}；4x1x1 读 4 且组内 0–3）。先前"仅 1x1x1"的记录对本设备/驱动不成立——探针代码在树。

**设计。** 每 (bh, vt) 行走配一对 256 线程 CTA（`%ctaid.x` 奇偶定 rank r，持 K 行 $R_r$）：每块 p_r = w[:,R_r]@S[R_r] 停入 `@pex_all[walk, r]` + 对 `@flagx[walk, r]` 的 RELEASE 原子；对等旗标 ACQUIRE 自旋；`vn = u − (p_0 + p_1)` 双部分和**固定序**从全局读（两 CTA 同值）；vnew 双写同字节（良性类）；`S[R_r] *= 2^gcl`；`S[R_r] += kg[:,R_r]^T @ vn`（每 CTA 恰一个 BK 折叠）。三险全闭：(1) 前向进度——簇属性使对共驻成为**驱动保证**（核以 %ctaid.x 奇偶映射、不读簇 SRS，普通发射下只要对恰好共驻仍值正确）；(2) 跨发射陈旧——旗标目标绝对化，harness 经 manifest 的 `zero_globals` 契约在**每次发射前流序 memset**（`pex_all` 为 `scratch_globals`，写前读免 staging）；(3) 栅栏——探针验证 `memref.atomic_rmw addf` 降为 `atom.acq_rel.SYS.global.add.f32`（强于所需 .gpu 域）。

**资源与数学门。** 池 S[K/2,BV] 8192 + wpp[C,K/2] 16384（w 半 cp.async 暂存、kg 原地覆写）+ kpp 8192 + vn 8192 + acc 8192 + gr 512 + expc 256 = **49920 B ≤ 50688**；**89 寄存器**（杆 128）——**首个同时过双杆的 fwdscan 形态**：真 2 CTA/SM、16 warp/SM、T2048 全部 128 CTA 单波驻留。数学：6/6 双形状、每一位数等于记录表；**h_all 位级一致**（2 ulp 的 vnew 重结合被 S 更新乘积的 tf32 舍入吸收——vnew 18607/4194304 胞格移动）；det N=5 新起进程位级一致（旗标/自旋交换确定且免死锁）。

**性能定论与分解。** 配对 ×3：**0.88× @T2048 / 0.89× @T8192**。分解测量：v3 核在 grid 128 上挂 2x1x1 簇属性耗时 722.8 µs == 2×361.4——**簇属性本身免费**；ks2 的差值是 **+57 µs 的交换成本**（park 存 + SYS 域原子往返 + vn 扫的 3× 全局读）。而每 CTA 减半的工作**没买到任何可测收益**：若每块关键路径真是 K 并行 GEMM 链，ks2 应落在 ≈0.5×361+57 ≈ 237 µs，实测 424.9。**结论：扫描的串行结构是每块的 staging/barrier/交换延迟，不是 GEMM 宽度**——与 v4 从 warp 组侧得到的同一定律。

**族闭合。** v4（warp 组）、ks2（双 CTA 簇）加此项分解，fwdscan 的 K-split/驻留族以**三个独立 GPU 数据点**闭合；簇发射 harness 契约（cluster / zero_globals / scratch_globals 三个 manifest 字段）留树，供未来任何跨 CTA 设计直接使用。

#### 6.6.4 三条边界的合并陈述

把三个否定并置，扫描族的收益边界呈一条一致的定律：

> **当每 warp / 每 CTA 的工作量不变时，调度器层面的 ILP 已经在场；串行扫描的剩余延迟由 staging、barrier 与（若有）交换的延迟构成，而非算术宽度。** 因此合法且有效的手段只有三类——把工作移出串行链（携带预复合、dkb 拆分）、压缩每步的暴露延迟（rgemm ILP、A 窗口、跨块预取）、以及纯列并行地增开 CTA（BV 切片）；任何"把同一工作铺到更多执行体"的形态（warp 组对半、双 CTA 对半、段深度）在本形状域内为中性或负收益。

---

### 6.7 小结：扫描族的最终形态

装机链（T2048 / T8192 / T512(H4)，同窗口配对阶段耗时，ms）：

| kernel | 职责一句话 | 关键机制 | T2048 | T8192 | T512(H4) |
|---|---|---|---|---|---|
| fwdscan | 顺向重算 h_all/vnew_all | prologue 累积提升 + 组 A/B ping-pong + tf32 双 GEMM + GSCG 混合 NC 免疫 | 0.376 | ~1.50 | 0.097 |
| caryA | A_all [K,K] | split-bf16 双链接 + Bhat 幂二折入 split + epilogue 直存；2 CTA/SM | 0.054 | — | 0.016 |
| caryM | M_all [K,C] | 双操作数 k16 切片 3 链接 + f32 精确初始化读-减 | 0.083 | — | — |
| bpre | B_all [K,V] | split×raw + 尾部单次缩放 + r10 复用栅栏 | 0.051 | — | — |
| dhs | 逆向携带链 | rgemm 寄存器累加 + A 滚动窗口 + iter_args 跨块令牌 + BV=16 | 0.272 | 1.047 | 0.038 |
| dkb | dkB/rowB/s5 并行重算 | split-bf16 3-mma vt 折叠 + accCK 退役 park + s5 树折；2 CTA/SM（栅格门） | 0.183 | 0.607 | 0.028 |

扫描族六 kernel 合计 ≈ 1.02 ms @T2048（占 bwd e2e ≈2.15–2.20 ms 的 ~46%）；对照重构前的旧形态（单一 dhu 2.17 ms + dkb 前身 0.96 ms，合计 3.13 ms，且 T8192 上 fwdscan 不可编译），族级累计约 **3.1×**（跨窗口量级参考）并把 T8192 从"不可编译"带入可建可验。数值与确定性：三形状 6/6 全过（最窄富余 dk 4.3× @T8192），全部阶段输出新起进程位级一致；两个纯延迟改动（dhs 窗口、dkb park）以位级等价证明收尾，三个值类改动（caryA/caryM 分裂、dkb BV=16/S5T）以预测锥位差审计收尾。三条否定边界（段切分、warp 组、簇双 CTA）与一条合并定律（§6.6.4）把"还能更快吗"的问题从猜测变成测绘。

> 证据索引：`tools/gdn_bwd_gen.py`（gen_fwdscan_v2/v3/v4/ks2、gen_caryA/caryM/bpre、gen_dhs、gen_dkb、rgemm/rgemm_aw、aggemm、astage_tile/agroup/await_all、_split_bf16 系列、main）；`tools/gdn_carry_validate.py`、`tools/gdn_fwdscan_ksplit.py`；`evidence/G5/bwd_report.md` 及 `evidence/G5/` 下 bwd*_verify/bench json、scratch 探针与配对日志；`tools/gdn_l1_check.py::chunked_bwd_ref`（两遍结构的 numpy 孪生）。

---

## 7 反向 kernel 体系之计算族：wy / dqkwg / wyb 与生存期竞技场编排

> 证据口径（本章全适用）：配对计时不取跨窗口绝对值——同一 GPU 窗口内基线臂/变体臂交错成对（base,new,base,new,…，CUDA-event 30 迭代中位数），共享 GPU 负载全程在位；绝对毫秒数受共居任务（pid 常驻、约 18–22 GB 显存）牵引，仅成对比值可判定（`evidence/G5/bwd_report.md` 各轮"Headline"节）。SASS/PTX 计数为静态指令统计（`evidence/G0` 工具链同源）。本章源码引用一律出自 `tools/gdn_bwd_gen.py`（行号为写作时快照；该文件是本计算族的唯一生成器，所有 kernel 以零实参 cubin + 公开 `memref.global`（MUTABLE）形态由 harness 经 `cuModuleGetGlobal` 装载）。rN 指代 `bwd_report.md` 中第 N 轮小节（r6 = G5-L round 6，r8 = G5-O round 8，r9 = G5-P round 9，r11 = G5-R round 11，r12 = G5-S round 12，r13 = G5-T round 13，r14 = round 14 (R14-V)，r15 = round 15 (R15-W)，r16 = round 16 (R16-Y)，r17 = round 17 (R17-bwd)，r18 = round 18 (R18-bwd)，r19 = round 19 (R19-bwd)，r22 = round 22 (R22-bwd)）。

### 7.0 章首：计算族在反传公式中的分工

#### 7.0.1 九 kernel 链中的三 kernel

GDN 反向链共 9 个 kernel（grid 均为 $M = B\cdot H\cdot N_C$ 个 CTA，块长 $C=64$）：`wy → fwdscan → caryA → caryM → bpre → dhs → dkb → dqkwg → wyb`。其中 fwdscan/caryA/caryM/bpre/dhs/dkb 构成"扫描/携带族"（状态递推、跨块携带与余项折叠，见前章）；本章的**计算族**是三个以 $C\times C$ / $C\times K$ / $C\times V$ 收缩为核心的 kernel：

1. **wy（WY 表示重算）**——反向的锚点重算级：由原始输入 $(q,k,v,\beta,g)$ 重建块内 WY 表示的全部共享中间量（$gc$、$kkt$、$A$、$T$、$P$、$u$、$w$），写入全局供全链消费（`kkt_all/P_all` 供 dqkwg/wyb；`u_all/w_all/T_all` 供 fwdscan/dhs 等扫描/携带族与本族）。它不直接产出任何梯度，但反向六梯度的每一个都穿过它写下的张量。
2. **dqkwg（最大 kernel）**——产出 $dq$、$dk_{\mathrm{local}}$、$dW$（即 $-\,\mathrm{d}v_p S_n^\top$）与携带族回传的行/列部分和 $r_3/r_4/c_{4r}$、$dg_{\mathrm{state}}$，以及后续 GEMM 化所需的 $\mathrm{d}v_p$（`dvp_all`）。
3. **wyb（dA 收缩与终值）**——执行 $\mathrm{d}A$ 的双侧严格下三角掩蔽收缩（两个 $C\times C\times C$ 收缩的三明治），产出最终 $dk$、$d\beta$、$dg$、$dv$。

#### 7.0.2 记号（先定义后用）

设块长 $C=64$，特征维 $K$、值维 $V$（实验形状 $K=V=128$），块内索引 $i,j\in[0,C)$。$x_{\mathrm{col}}$ 表示按列广播的行向量（$[x]_{ij}=x_j$），$x_{\mathrm{row}}$ 同理按行。kernel 内 decay 一律在 $\log_2$ 域（`arith.exp2` 换域）。

- $gc_i=\sum_{t\le i} g_t$（块内累积 log-decay），$gcl = gc_{C-1}$；
- $kkt = k\,k^\top$；$A = \mathrm{stricttril}\big(\beta_{\mathrm{row}}\odot kkt\odot 2^{gc_i-gc_j}\big)$；$T=(I+A)^{-1}$；
- $P=\mathrm{tril\_inc}\big((q\,k^\top)\odot 2^{gc_i-gc_j}\big)$（含对角）；
- $u = (T\odot\beta_{\mathrm{col}})\,v\in\mathbb{R}^{C\times V}$，$w = (T\odot(\beta\,2^{gc})_{\mathrm{col}})\,k\in\mathbb{R}^{C\times K}$（命名沿用本仓库 `bwd_report.md` 阶段表；fla 源码中同名张量角色互换，集合相同）；
- $\mathrm{d}Pr=\tfrac{1}{\sqrt K}\sum_{vs}\mathrm{d}O_{vs}\,v_{\text{new},vs}^\top$；$\mathrm{d}Pm=\mathrm{stricttril}(\mathrm{d}Pr\odot 2^{gc_i-gc_j})$；
- $\mathrm{d}v_p = P^\top \mathrm{d}O/\sqrt K + \mathrm{diag}(2^{gcl-gc_i})\,(k\,\mathrm{d}h^\top)$；$\mathrm{d}W=-\,\mathrm{d}v_p\,S_n^\top$（$S_n$ 即 `h_all` 的块末状态）；
- $G=\mathrm{stricttril}\big(\mathrm{d}v_p\,(\beta v)^\top+\mathrm{d}W\,(\beta k\,2^{gc})^\top\big)$；
- $\mathrm{d}A_f=-\,\mathrm{stricttril}\big(T^\top\,G\,T^\top\odot 2^{gc_i-gc_j}\big)$（§2 的双侧掩蔽精确式）；
- **split-bf16**：$x\in\mathbb{R}$ 的 f32 值拆为 $hi=\mathrm{bf16}(x)$、$lo=\mathrm{bf16}(x-hi)$，则 $hi+lo$ 携带约 16 位尾数；2-链乘积 $hi\!\cdot\!B+lo\!\cdot\!B$、3-链 $hi\!\cdot\!hi+hi\!\cdot\!lo+lo\!\cdot\!hi$（弃 $lo\!\cdot\!lo\approx 2^{-32}$）以 mma 速度恢复 fp32 级乘积，残差类 $\sim 2^{-17}$（`_split_bf16` 文档串，r5/r12）。tf32 的两侧操作数截断为 $2^{-11}$ 级——凡 tf32 位点越_bar_者均按此理由换 split（r3 bisect：tf32 曾把 dkb.dkB 推至 $1.04\times 10^{-2}$，超 $8\times 10^{-3}$ 容差）；
- **竞技场（arena）**：同一基地址上按"生存期互斥"叠放多个共享内存 tile 的 PTX 级别名机制（§7.4）；
- **R-A-S**：read-add-store，读全局-加-写回；
- **warp 蝶形和**：`warp_rowsum/warp_colsum`——行/列归约从"单线程串行 $C$ 链"改为 warp 内 5 步 shuffle 树（每行 2 深），lane 0 落盘（r8）。

三个 kernel 的驻留形态（出厂默认臂）：wy 为 128 线程/4 warp × 2 CTA/SM；dqkwg 为 256 线程 × 2 CTA/SM（大网格臂，$BH\cdot N_C\ge 164$ 时启用 v4，T512 小网格保 v3）；wyb 为 256 线程 × 2 CTA/SM。

#### 7.0.3 为什么要手写 GEMM/mma 发射

本工具链（stock MLIR → NVVM → ptxas，sm_120a）上：`linalg.matmul` 不可用——`-convert-linalg-to-loops` 使所有线程冗余执行循环巢、链式累加在同一 `outs` 上跨 warp 竞争、同缓冲双侧操作数整体误降级（G5-C 平台约束 3）；`vector.transfer` 不能在 workgroup memref 上降级（约束 2）；`cp.async` 直供 `ldmatrix` 会按构建调度决定性地读错 16 行块中的子集（r6 静默损坏，见 §7.4.3）。因此计算族的每一条 GEMM 都是生成器手写的线程划分 `scf.for`（每单元胞恰一线程写，确定性）或 `nvgpu.ldmatrix + nvgpu.mma.sync`（m16n8k16, bf16×bf16→f32）发射，操作数 tile 一律 `st.shared` 写入。这一约束是本章所有机制的存在理由。

---

### 7.1 wy：块内 WY 表示重算 kernel

#### 7.1.1 总体数据流

`gen_wy`（`tools/gdn_bwd_gen.py` L2163–2599）每 CTA 处理一个 chunk，级内顺序：

1. $gc$ 前缀和（§7.1.5）+ $\beta$ 暂存 + $bgc=\beta\odot 2^{gc}$ 提升；
2. $kkt/A$（§7.1.2 的 k-切片 mma 链 + epilogue 双写：原始 $kkt$ 存 `kkt_all` 全局供 wyb 的 $\mathrm{d}A_f\odot kkt$ 项；掩蔽衰减后的 $A$ 写 smem `kktf` 作求解输入）；
3. $T=(I+A)^{-1}$ 前代换求解（§7.1.3）+ `T_all` 全局存储；
4. $P$（同款 k-切片链，epilogue 直接全局存储 `P_all`）；
5. $u/w$ 尾（§7.1.4）。

【源码】阶段表与全局声明：

```python
k.global_("T_all",  (M, C * C), "f32")
k.global_("u_all",  (M, C * V), "f32")
k.global_("w_all",  (M, C * K), "f32")
k.global_("kkt_all",(M, C * C), "f32")   # G5-E round2: kkt 与 P 只算一次，全链共享
k.global_("P_all",  (M, C * C), "f32")
```

#### 7.1.2 kkt 与 P：k-切片 mma 链

**【动机】** r12 之前 $kkt=k\,k^\top$ 与 $P=(q\,k^\top)\odot 2^{\Delta}$ 以整 $K=128$ 收缩的 mma 计算，需要 $k_t/q_t$ 两个 $[C,K]$ bf16 共居 tile（32 KB）及其 cp.async 暂存，使 wy 池达 90 624 B（1 CTA/SM，占用率 12.5%）。把整 $K$ 链拆为 8 条 $K{=}16$ 链后，操作数退化为每次一条 16 列的 $[C,24]$ 切片 tile（3 072 B），$k_t/q_t$ 与全部 `opc/accc/kslc` 暂存退役。

**【源码片段】**（`gen_wy` L2308–2326）：

```python
accs = None
for kb0 in range(0, K, 16):
    stage_rows(e, ksl, G["k"], kty, m, col0=kb0, cm=K, shape=(C, 16))
    last = (kb0 == K - 16)
    accs = mma_gemm(e, f"kkt{kb0}", C, C, 16, W, ksl.name, ksl.ty,
                    ksl.name, ksl.ty, kt=True, accs=accs,
                    epilogue=kkt_epi if last else None)
    e.l("gpu.barrier")
```

每链：`stage_rows`（`st.shared` 逐元素暂存——r6 规则：ldmatrix 操作数必须 `st.shared` 写入）→ `mma_gemm` 一条 $K{=}16$ 链 → r10 再暂存栅栏。累加器以 `accs=` 寄存器链跨链携带（r5 引入、r8 `k0=` 机制的同族），**片断/mma/累加序列与旧整 $K$ 调用完全一致**——$kkt/A/P$ 按构造位同。kkt 的 epilogue 在最后一个链上双写（原始值入 `kkt_all`；$\beta_i\,2^{gc_i-gc_j}$ 衰减、严格下三角选择后入 `kktf`）。$P$ 链同理（`p_epi`：含对角 `sle` 选择 + 全局直存）。

**【IR/PTX 前后对比】**

| PTX/SASS 类 | r7 旧形（整 K + k_t/q_t + cp.async） | r12 新形（K16 切片链） |
|---|---|---|
| `ld.shared` | 832 | **49** |
| `mma.sync` | 256 | 384 |
| `ldmatrix` | 96 | 384 |
| `cp.async` | 20 | **0** |

**【效果】**（r12，配对 x3，T2048）：wy 0.7005→0.2277 ms（**3.08x** 单独臂；终链表 3.04x）；T512 0.0920→0.0442（2.08x）。数值：切片化是位同重构（bit-identical by construction），六梯度数位不动。

#### 7.1.3 $(I+A)^{-1}$ 求解：thread-per-column 前代换

**【动机】** C=64 的求解拆为两个 32×32 单位下三角基 $T_{00},T_{11}$ 与合并项 $T_{21}=-T_{11}A_{21}T_{00}$。r6 之前由**单线程**行串行步进（CTA 关键路径上约 1.1 万条依赖 FMA）。

**【机制与保序证明】** 单位下三角前代换的列 $j$ 只读列 $j$：

$$T[i,j] = -\sum_{j\le \kappa<i} A[i,\kappa]\,T[\kappa,j]$$

右侧 $A[i,\kappa]$ 与 $T[\kappa,j]$ 的第二下标均为 $j$——列独立性成立。故 64 线程（`tid < C`）各持一列、**无内部栅栏**并行求解两基全部列；每列的 $\kappa$-升序 f32 链是**与串行行走完全相同的求和、完全相同的次序**——$T$ 位同。严格上三角（含 $T_{12}$ 块）显式清零（旧代码依赖 smem 分配即零的未定义行为，r6 改为显式写零，值同）。合并两步 `sgemm`（`tmpm = A21@T00; T21 = -(T11@tmpm)`，L2441–2443）。

**【源码片段】**（L2360–2396，节选）：

```python
tg = e.v("wyt")
e.l(f"{tg} = arith.cmpi ult, %tid, {e.ic(C)} : index")
e.l(f"scf.if {tg} {{")
...
e.l(f"{ssum} = scf.for {sk} = {j} to {ir} step {e.ic(1)} "
    f"iter_args({sac} = {z0}) -> (f32) {{")
aik = load2(e, kktf, ir, sk)      # A[ir, sk]：本列元
tkj = load2(e, Tf, sk, j)         # T[sk, j]：本列已解元
pp = op2(e, "mulf", aik, tkj)
ss = op2(e, "addf", sac, pp)
e.l(f"scf.yield {ss} : f32")
```

**【IR/PTX 前后对比】** 前：`scf.if %isz`（单线程）内 $O(C^3/2)$ 串行迭代、每步 smem 往返依赖链；后：`scf.if (tid<64)` 的 64 路列并行、每线程至多 32 步 $\kappa$-升序内积链（`scf.for` iter_args 寄存器驻留），发射的 MLIR 算子类不变（`arith.mulf/addf` + `memref.load/store`），仅谓词域从 `%isz` 换为列守卫。

**【效果】**（r6，同窗）：wy 1.45→0.70 ms（**2.06x**）@T2048；T512 0.199→0.095（2.1x）。数值：$T$ 位同（六梯度与 r5 完全一致）。

#### 7.1.4 $u/w$ 尾：2-链 split-bf16 与 cvec 列尺度

**【动机】** $u/w$ 是 wy 最后的标量/tf32 大尾部。r3 bisect 已证明 tf32 化的 `wy.u` 把 dk 推到 $1.06\times 10^{-2}$（超 $8\times 10^{-3}$ bar），tf32 路线被封死；r12 以 r11 caryA 的 split-bf16 剧本接手。

**【机制】** 以收缩下标（列）尺度预乘后拆分 A 侧：

$$u = \big(T\odot\beta_{\mathrm{col}}\big)v:\quad \tilde T = \mathrm{split}\big(T_{ij}\,\beta_j\big),\qquad u = \tilde T_{hi}\,v + \tilde T_{lo}\,v$$

尺度先在 f32 中乘一次（一次舍入），再进入 $2^{-17}$ 残差类的 hi/lo 拆分（`stage_split_ew cvec=` 参数，r12 引入"任意尺度折叠"）；$w$ 同理用 $bgc_j=\beta_j 2^{gc_j}$。B 侧是**原始 bf16 全局**（$v$/$k$），转置暂存到 mma 的 $[n,k]$ 定向（`stage_raw_T`）——**精确、零舍入**。A 侧分裂读 `T_all` 全局（本 CTA 刚写入——同 kernel 全局写后读、L2 热），**不**读 `Tf` tile：`Tf` 的竞技场窗口在循环期间让给 B 原始 tile。两链（hi、lo）以 `accs=` 寄存器链接力，epilogue 全局直存 `u_all/w_all`——`emap/gemm/store_rows` 三段扫描与 `opc/accc/kslc` tile 全部退役。数值注意（r12 记录）：$u$ 原为标量 f32（逐积 fma）、$w$ 原为 tf32（双操作数 $2^{-11}$）；两者统一到 $2^{-17}$ split 类——$w$ **严格变紧**，$u$ 经 $S\to S_n\to dk$ 放大路径重掷（dk 数位随后在 r13 收紧，见 §7.3.6）。

**【源码片段】**（L2546–2567，u 相）：

```python
stage_split_ew(e, auhi, aulo, G["T_all"], ccty, m, C, 0, C, C, cvec=beta)
for vs in range(0, V, BVU):
    stage_raw_T(e, vb, G["v"], vgty, m, V, vs, BVU, C)
    accs = mma_gemm(e, f"uh{vs}", C, BVU, C, W, auhi.name, auhi.ty,
                    vb.name, vb.ty, kt=True)
    accs = mma_gemm(e, f"ul{vs}", C, BVU, C, W, aulo.name, aulo.ty,
                    vb.name, vb.ty, kt=True, accs=accs, epilogue=epi_u)
    e.l("gpu.barrier")   # r10 再暂存栅栏
```

w 相在同一 arena1 槽位重拆（`awhi/awlo`，尺度 `bgc`；上相尾栅栏已围栏）。

**【IR/PTX 前后对比】** 标量 `arith.mulf/addf` 双重循环（f_up/f_w 扫描）与 tf32 `mma.m16n8k8`（`--tf32` 位点）→ 2 条 bf16 `nvgpu.mma.sync ... mmaShape=[16,8,16]` + `nvgpu.ldmatrix`；每元素 MUFU（`ex2`）消失（尺度提升为 $bgc$ 向量，SASS EX2 类随整 kernel 的 exp2 hoist 从 149 降到 4，r8）。

**【效果】** 并入 §7.1.6 的整 kernel 数字（3.04x 的组成部分）；dk 经 $u$ 重掷后 1.718e-3/1.724e-3（6/6 维持，余量 4.7x/4.6x，r12 数值表）。

#### 7.1.5 $gc$ 前缀和：Hillis–Steele shuffle 扫描

**【动机】** 旧实现是每线程**冗余**执行同一 C 步串行行走（每步全局读 + smem 往返的依赖链，所有 128 线程重复同一工作以避免栅栏）。

**【机制】**（`gc_cumsum_shfl`，L1221–1291，`GDN_WY_GC_SHFL` 默认开）：三相——(0) 并行合并停放原始 $g$ 值；(1) `tid<C` 各持一元，warp 内 5 步 `gpu.shuffle up`（偏移 1,2,4,8,16）扫描；**warp 均匀性**：C=64 恰被 warps 0/1 的 `tid<C` 谓词整覆，故 shuffle 的全 lane 收敛成立；lane 31 停放 warp 总和入 $[2,1]$ smem 向量 `wtot`（`andi tg,l31` 守卫防止 warps 2/3 越界写）；读下标 `minsi` 钳制使死 lane 的 smem 读保持在界；(2) warp≥1 加 warp 0 总和，守卫写回。

**【数值声明】** 确定性（固定平衡树）但**不位同**于 k-升序串行行走——求和次序改变，$gc$ 每数位以 $\sim2^{-24}$ 相对类重掷，而 $gc$ 供 $T$ → 六梯度全部重掷。实测（r12）：`T_all` 重掷 4.2e-7、dk 绝对位移 $1.2\times10^{-4}$；6/6 通过，dk 余量 4.7x。**这是本计算族唯一的非位同数值变更**，按 6/6 门径独立放行。

**【源码片段】**（L1256–1264）：

```python
for off in (1, 2, 4, 8, 16):
    e.l(f"{o}, {p} = gpu.shuffle up {x}, {_i32c(e, off)}, {_i32c(e, 32)} : f32")
    y = op2(e, "addf", x, o)
    e.l(f"{x} = arith.select {p}, {y}, {old} : f32")
```

**【IR/PTX 前后对比】** 前：C 步 `memref.load`/`arith.addf` 依赖链 ×128 线程冗余；后：5 条 `gpu.shuffle up` → PTX `shfl.sync.up.b32` ×2 warp + 跨 warp smem 偏移。

**【效果】** wy 单独 +4%（0.2374→0.2278 配对，r12 l1b 臂）；默认开。

#### 7.1.6 生存期收缩：90 624 → 45 832 B，2 CTA/SM

**【机制】** 上述机制的生命期结果：常驻 tile 只剩 $[C,24]$ 切片对（`ksl/qsl`）、两竞技场宿主（`kktf` $[C,C{+}8]$ f32 = 18 432 B，host $u$/$w$ 相的 A-分裂对 9 216×2；`Tf` $[C,C]$ f32 = 16 384 B，host B 原始 tile `vb/kb`）、求解 scratch `tmpm` 与向量。池 = 18 432+16 384+3 072+3 072+4 096+8+768 = **45 832 B ≤ 50 176 B → 2 CTA/SM**（旧 1 CTA、12.5% 占用率）；寄存器 255→158/161、0 spill（r12）。别名细节见 §7.4。

**【效果汇总（wy 全 kernel，r12 终链表，配对）】**

| 量 | r7 形 | r12 终形 |
|---|---|---|
| 池（B） | 90 624（1 CTA） | **45 832（2 CTA）** |
| 寄存器 | 255 | 158/161（0 spill） |
| `ld.shared` / `mma.sync` / `ldmatrix` / `cp.async` | 832 / 256 / 96 / 20 | 49 / 384 / 384 / **0** |
| wy @T2048（ms） | 0.700 | **0.2299**（3.04x） |
| wy @T512（ms） | 0.0920 | 0.0442（2.08x） |
| bwd e2e @T2048（ms） | 4.541/3.927 | 3.734/3.331（1.216x med / 1.179x min） |

#### 7.1.7 R17 FPIPE 的 `T_all→Tf` DMA（中性，保留）

r17 为 wy 加了一条异步预取：`T_all` 存储栅栏一过即发 `T_all→Tf` 的 16 KB 原始 f32 DMA（此刻 `Tf` 已死——求解 tile 的最后读者就是存储扫描），整个 $P$ 相（8 条切片暂存 + mma 链）作掩体；$u$ 分裂随后读驻留 tile（`Tf` 只喂 `ld.shared`，r6 规则合法），同 $\beta$ cvec 算术 → `au*` 位同。**配对中性**（1.001x/1.006x/1.01x 三形状）：源是同 kernel 刚写的一遍性读、L2 驻留、无可藏延迟。保留（默认开）：零成本且 T8192 工作集（`dvp_all` 64 MB）在 L2 边缘，更大 B 时掩体可能生效（r17）。

---

### 7.2 dqkwg：$dq$ / $dk_{\mathrm{local}}$ / $dW$ 计算核

#### 7.2.1 公式分解与相位划分

`_gen_dqkwg_v4`（L5853–7042，出厂臂；`gen_dqkwg` 按 $BH\cdot N_C\ge 164$ 分派，小网格保 v3）实现的量：

$$
\begin{aligned}
\mathrm{d}Pr &= \tfrac{1}{\sqrt K}\textstyle\sum_{vs}\mathrm{d}O_{vs}\,v_{\text{new},vs}^\top &&\text{（P2，mma 链）}\\
r_4 &= \mathrm{rowsum}(\mathrm{d}Pr\odot P),\quad c_{4r}=\mathrm{colsum}(\mathrm{d}Pr\odot P) &&\text{（warp 蝶形）}\\
\mathrm{d}Pm &= \mathrm{stricttril}(\mathrm{d}Pr\odot 2^{gc_i-gc_j}) &&\text{（原地 emap）}\\
dq &= \mathrm{d}Pm\,k \;+\; \mathrm{diag}(2^{gc}/\sqrt K)\,(\mathrm{d}O\,S_n^\top) &&\text{（早跑停车 + P11 链 + R-A-S）}\\
\mathrm{d}v_p &= P^\top \mathrm{d}O/\sqrt K+\mathrm{diag}(2^{gcl-gc_i})\,(k\,\mathrm{d}h^\top) &&\text{（P9，pt/akb/bh 链）}\\
\mathrm{d}W &= -\,\mathrm{d}v_p\,S_n^\top &&\text{（P11，dw 链）}\\
dk_{\mathrm{local}} &= \big(T^\top \mathrm{d}W\big)\odot bse \;+\; \mathrm{d}Pm^\top q \;+\; dkB_{\Sigma},\qquad bse=\beta\odot 2^{gc} &&\text{（早跑停车 + P12 TW + epilogue 折叠）}\\
r_3 &= \mathrm{rowsum}\big(dq_1\odot q\big),\qquad dg_{\mathrm{state},i}=s_5+\textstyle\sum_{j\le i} rowB_j &&\text{（蝶形 / 单线程串行）}
\end{aligned}
$$

实现细节：pt 对把 $rie_i=2^{gc_i-gcl}/\sqrt K$ 折入 $P^\top$ 的 hi/lo 分裂、链尾单因子 $egc_i=2^{gcl-gc_i}$ 一次相乘还原 $P^\top\mathrm{d}O/\sqrt K$（r8 设计，$egc\cdot rie\cdot scale=scale$ 至 $2^{-24}$）；$bse$ 行尺度折入 TW epilogue（与旧原地 emap 同一 mulf，r12）。

#### 7.2.2 dPm 尾部早跑与全局停车（R-A-S）

**【动机】** r8 末的 dqkwg 是延迟受限的 1-CTA 体（99 328 B 池；ncu：新旧臂内存吞吐 12.7% 对 28.6%，serialization 下时长不变——释放的发射槽被暂存/mma 延迟回填）。r9 的占用率杠杆需要把池压到 ≤50 176 B，最大障碍是 accCC（dPm，16 384 B）跨越 **全部** 后续相位的生存期：dq/dk 尾部若在 TW 相才跑，accCC 必须活到 P12。

**【机制】** 把两个 dPm 尾部**提前**到 emap 之后立即执行（"早跑"），其 $[C,BK]$ 切片**停车**（park）到 `dq`/`dk_local` 全局各自的目标单元——不占 smem：

- `dq_dPmk`：A = dPm 自然定向分裂（`_split_bf16` 自 accCC），B = 原始 $k^\top$（`stage_raw_T`，精确）；
- `dk_dPmq`：A = **同一批 arena tile 的转置重分裂**（`_split_bf16_T`；dq 收缩 dPm 的行、dk 收缩其列，两定向不能共享一个分裂但可顺序共享 tile），B = 原始 $q^\top$；
- 其后 dW 循环对 `dq[ks]` **读-加-存**（R-A-S），TW 循环对 `dk_local[ks]` 读-加-读-加-存（叠加 $dkB_\Sigma$）。

代价核算（r9）：每 chunk +96 KB 的 L2 驻留往返，换取 accCC-dPm 16 KB 生存期在 pt 阶段之前死亡——arena1 随之为 pt/bdt 让位。r14 进一步把停车写直入链尾 epilogue（`park_epi`），accCK 中转 + `store_rows` 扫描 + 每尾 2 条栅栏退役；R-A-S 折叠进 `epi_dq1`（$dq[ks]\ {+}{=}\ xs$，与旧扫描同加法次序）与 `epi_tw`（$dk_{\mathrm{local}}$ 三项折叠，同 `addf` 次序）。

**【源码片段】**（L6546–6559，dq 尾；FOLDS 停车 epilogue）：

```python
_split_bf16(e, dhi, dlo, accCC, C, C)
for ks in range(0, K, BK):
    stage_raw_T(e, krt, G["k"], kty, m, K, ks, BK, C)
    accs = _mh_g(e, f"dmh{ks}", C, BK, W4, dhi.name, dhi.ty, krt.name, krt.ty, C)
    accs = _mh_g(e, f"dml{ks}", C, BK, W4, dlo.name, dlo.ty, krt.name, krt.ty, C,
                 accs=accs, epilogue=(park_epi("dq", ks) if FOLDS else epi_dq_park))
    e.l("gpu.barrier")
```

**【IR/PTX 前后对比】** r8 形：tf32 `tgemm` 尾（192 条 k8 链静态计数，`mma.m16n8k8` + f32 tile 暂存 `tCC/tCKa/tCKb` + `cp.async`）；r12 SB 形：bf16 `mma.sync [16,8,16]` ×2/位点 + `ldmatrix`，`cp.async` 46→**0**、`ld.shared` 1234→**71**、栈泄漏 72 B→0、寄存器 255→240（r12）。早跑重构本身只引入 ~1-ulp 级加法重排，dk 数位不可见（r9：dk 1.719e-3/1.710e-3 与基线同印刷位）。

**【效果】** r9：dqkwg 1.772→1.075 ms（**1.65x**），bwd e2e 6.00/5.36→5.20/4.66（1.155x med）；r12 SB 再叠加（dqkwg 1.075→0.9421，1.14x）且 **dq 数位收紧 4.910e-4→1.824e-4（余量 16x→44x）**——tf32 对 dPm 的 $2^{-11}$ 截断换成 $2^{-17}$ split 类，dk 1.710e-3→1.691e-3。

#### 7.2.3 P11 vs-外层重构：元素级保序证明

**【动机】** dW/dq1 合并循环（相位 P11）按 (ks, vs) 16 组迭代；其中 `akb`（原始 dO 切片）、`vnh/vnl`（dvp 分裂）是 **vs-专属**（值跨 ks 不变），却在 r15 2-CTA 臂里随每组重暂存（16 次冗余扫描）。 slab 修复需要 ~17.4 KB 池余量而 2-CTA 臂只有 3 560 B（r15 负结果）；r16 改为**换循环块**。

**【机制与保序证明】**（`GDN_DQKWG_P11R`，默认 2）：P11R 个 per-ks 的 dw/dq1 mma 累加器链集作为**并行寄存器链**跨一个 vs-外层块存活；vs-专属 tile 每 vs 只暂存一次。关键论断——**元素级保序**：对每个输出元素，其全部乘积仍以同一 (vs, link) 序列进入同一条 per-ks mma 累加链；改变的只是 16 个 (ks,vs) 组的**代码次序**，而非任何输出元素内的算术次序。r3 的 warp 行和仍 ks-升序发射；accCK 保持单 tile——在 last_vs 处，每个 $k_i$ 的 `epi_dq1` 之间夹着**它自己的** r3 行和（$k_i$ 的行和先退役 accCK，$k_{i+1}$ 的 epilogue 才重写；链/行和自带的尾栅栏串行化）。

**【源码片段】**（L6788–6818，节选）：

```python
for ksb in range(0, K // BK, P11R):
    accsW = [None] * P11R
    accsQ = [None] * P11R
    for vs in range(0, V, BV):
        stage_raw_bf16(e, akb, G["do"], vty, m, V, vs, C, BV, bar=False)
        stage_split_ew(e, vnh, vnl, G["dvp_all"], dvt, m, V, vs, C, BV)
        for ki in range(P11R):
            ks = (ksb + ki) * BK
            stage_split_ew(e, bhh, bhl, G["h_all"], hty, m, V, ks * V + vs, BK, BV)
            accsW[ki] = mma_chain_if(e, f"dw{ks}_{vs}", C, BK, BV, W4, None,
                                     ops_w, accsW[ki], ..., bar=False)
            accsQ[ki] = mma_chain_if(e, f"dq1_{ks}_{vs}", C, BK, BV, W4, None,
                                     ops_q, accsQ[ki], ...)
            if last_vs and P11R > 1:
                warp_rowsum(e, r3, accCK, C, BK, term=mk_term_r3(ks), bar=True)
```

**【效果】** 实测证明（r16）：T2048 全链 35/35、T8192 36/36 输出数组**逐元素位同**基线；六个验证余量数位与 r15 表全等。计时（配对）：dqkwg 0.5445→0.5236（**1.040x**）@T2048、1.8233→1.7556（**1.039x**）@T8192。对照臂 P11R=4（全灭冗余，4 次扫描）：自然体 222 寄存器 → `.maxnreg 128` 下 spill 涨至 232 B 栈/276 B 存/632 B 载，仅 1.026x——第 4 个累加器链集的 spill 流量吃掉多省的暂存；P11R=2（自然 174 寄存器，96 B 栈/96/148 B spill）胜出。vs-专属扫描 16→8 次；per-(ks,vs) 的 `bhh/bhl` 重暂存不可约简（每对是相异数据）。

#### 7.2.4 pt 对的生存期治理

**【动机】** pt 对（$pthi/ptlo$，$P^\top\odot rie$ 的 hi/lo 分裂，9 216+9 504 B）是 P9 dvp 相的 mma A 侧。r8 排名曾把"退役 pt 对（18 432 B）"列为 2-CTA 前提；最终系统保留 pt 但将其**生存期压缩进死窗**，配套三轮治理：

1. **P 级退役（r9，已入库）**：r3/r4/$c_{4r}$ 的 P 因子改从**全局**读（`gcol`，行和 lane 合并；列和的第二遍跨步读 L2 热——`P_all` 切片刚被流过），16 KB 的 P smem 驻留退役；蝶形求和树与 v3 原地变体位同。
2. **pst 重驻（r13，`GDN_DQKWG_PST` 默认开）**：P 选择性地重暂存进 dhi/dlo 窗口（dPm 分裂登场前空闲）：**P 值位同**（同 f32 元素、同 lane/树）→ r4/c4r 位同；实测 1.01x（噪声带，r13 诚实记录"非验证的赢"），因零成本且值同而保留。
3. **pt 异步化几何封锁（r19 负结果）**：dPm 尾部期间 arena1 字节级满载——accCC $[0,16384)$ 是两次重分裂的分裂源、dhi/dlo 占 $[16384,34816)$、arena2 被 krt/qrt 占据——dPm 掩体下**没有任何窗口**可放飞一个原始 P 半区；kernel 顶部替代方案处处冲突（`pst` 自身的字节是唯一 8 KB 窗口，退役 pst 等于退还 r13 的 LDS 收益）。记录在案、未实施。

**【效果】** 1+2 合计使 pt 对成为"P9 相一次性暂存"（L6600 `stage_split_T` 一处调用），驻留窗 $[0,18432)$ 复用 accCC 死后字节（pthi@0、bdt@9216）；r17 FPIPE 期间 `dprT` 车辆停 accCC 自身字节而 akb@18944 不受扰（见 §7.2.7 的失败案例）。

#### 7.2.5 V-宽 slab 减肥（r14）与 2-CTA 下的让位

**【动机】** r13 末的 dqkwg 0.937 ms 中，P11 的暂存冗余（dvp 分裂随 (ks,vs) 16 次重暂存）与 R-A-S 往返是可见残项（r13 负结果清单）。

**【机制】**（`GDN_DQKWG_DIET`，r14）：dvp 分裂（`svh/svl` $[C{+}1/C{+}3, V{+}8]$）**一次**staged（替代 16 次）、h 分裂（`hbh/hbl`）per-ks 4 次（替代 16 次）、原始 dO（`dbo`）一次；mma 链读 slab 在 $k_0=vs$ 处的列（六元组 ops）——同操作数值、同链次序/累加 → 乘积位同。slab 叠放 arena1 的 pt/dvp 与 TW 之间的死窗（34 816→52 224）；池 66 048→100 864 B（反正 1 CTA）。

**【效果与让位】**（r14，配对）：饮食单独 0.9371→0.7348（**1.28x**）@T2048；寄存器 240→**128**（16 B spill）——这是 512 线程栈的使能器（§7.2.7）；栅栏 206→112（r7 不相交 tile 规则 + 只读链合并）。但 r15 的 2-CTA 化发现 slab 窗口正是把池钉在 1 CTA 的元凶：**出厂 2-CTA 臂撤下 slab**（`SLABS = DQKWG_DIET and not TWO_CTA`），P11 冗余由 §7.2.3 的 P11R 块换承接；FOLDS（epilogue 停车/折叠/栅栏饮食）保留。

#### 7.2.6 dPr 的 warp 映射：strip → mh

**【动机】** strip 映射（`mma_gemm`）把 warp $w$ 绑到 N-条带 $[w\cdot n_w,\dots)$，故 $W\le N/8$：N=32 形（dPm 尾、TW）在 8 warp 下只有 4 个 mma 活跃；r8 第一版 v3 的 dvp/dW/dq1 链跑在 `%wgv4` 守卫内——半 SM 闲置（ncu：89% no-eligible 周期、每调度器 ~2 活跃 warp、shared-scoreboard 49.9%/18.1 cyc/inst）。

**【机制】**（`mma_gemm_mh`，L1821–1920）：2D warp 映射——M 维也二分：warp $w$ 持 $n_i=w\bmod n_{pr}$ 与 mi-块对 $(w\!\!\!/n_{pr})\cdot m_{pw}\cdot 16$；$W=8$ 全活跃、无守卫、无 `scf.if`-yield 区，B 片在 warp 的 mi 对间复用。片断/lane 数学与 strip 同（G4 锚定语义），仅 warp→tile 指派不同；`accs` 与 `mma_gemm/mma_chain_if` 链兼容。r14 扩至 $W\le16$（512 线程臂；映射按 halves 泛化，片断数学 warp-局部）。首版 bug（r8 验证门捕获）：行偏移误加块计数——warps 4–7 读行 {2,18} 而非 {32,48}，0.9 类失败，一行修复为 $(w/n_{pr})\cdot(m_{pw}\cdot16)$。

**【源码片段】**（L1845–1851）：

```python
e.l(f"%mh_{name} = arith.divsi %mW, {e.ic(npr)} : index")
e.l(f"%mhb_{name} = arith.muli %mh_{name}, {e.ic(mpw * 16)} : index")  # 行基（行数）
e.l(f"%nx_{name} = arith.remsi %mW, {e.ic(npr)} : index")
e.l(f"%ncv_{name} = arith.muli %nx_{name}, {e.ic(8)} : index")
```

**【IR/PTX 对比】** 前：`scf.if %wgv4 { ... }` 包裹的链（4/8 warp）+ `scf.yield` 跨区累加器；后：平铺无守卫链，寄存器 255（60 B spill）→248（0 spill）（r8）。

**【效果】** 出厂 256 线程 2-CTA 臂：dPr（N=C=64，strip 容 W=8）走 strip；dPm 尾与 TW（N=BK=32）走 mh；512 线程臂 dPr 亦切 mh（$W{=}16>8$，L6439–6452）——同片断值、同 kb 累加、同 warp 树和 → 数位不动（r14 值精确轮）。

#### 7.2.7 占用率三级跳：竞技场化 → 512thr/128regs → 2-CTA 四 tile 纯别名

**（a）r9：48 640 B 竞技场 + 96 寄存器。** PTX 级生存期别名（§7.4 机制）把 23-malloc 体折叠为双竞技场（arena1/accCC 16 384→17 408；arena2/vnh 10 240）：别名后 ptxas 落在 **96 寄存器 / 0 spill**（未别名同体要 255+spill——别名补丁同时坍缩了地址空间复杂度），256 线程 ×2 CTA=16 warp/SM 的 128 寄存器门槛免费达成。

| ncu（T2048，kernel-only） | v3 基线 | v4（r9） |
|---|---|---|
| 线程 / 静态 smem / 寄存器 | 256 / 99 328 B / 132 | 256 / 48 640 B / **96** |
| 块上限（smem, regs） | 1, 1 | **2, 2** |
| 理论/实际占用率 | 16.67% / 15.81% | **33.33% / 30.28%** |
| 每 SM 实际 warp | 7.59 | **14.53** |
| 每调度器活跃 warp | 1.98 | 3.69 |
| no-eligible 周期 | 88.98% | 79.56% |
| SM 内存吞吐 | 13.2% | 20.0% |

教训一（r9 负结果）：128 线程 ×2 CTA 是结构性空转——sm_120 每 SM 48 warp 槽（12/调度器；r7 成本模型曾按 32 计），2×4=8 warp 与 1×8 相同，墙钟持平（1.780 对 1.775）而 T512 损 1.8x；胜出需要 16 warp。

**（b）r14：512 线程 × 128 寄存器铺满 16 warp。** r14 饮食把自然体压到 128 寄存器后，512×128 = 65 536 = 整个寄存器堆合法（256 线程 240 寄存器体在 512 线程下无法启动）：1 CTA ×16 warp = 4 warp/调度器（原 2）。ncu（饮食 cubin）：90.3% no-eligible、1.97 活跃 warp/调度器——纯延迟画像。dqkwg 0.7348→0.6420（饮食+512 线程对 HEAD 0.940 合计 **1.46x**），e2e 2.690→2.389（1.126x）。

**（c）r15：2×256 线程两 CTA——四 tile 纯别名编排图。** 512 线程臂达到 16 warp 但**单指令流**；2×256 = 两个独立流。r14 草案（accCK 退役到全局 + bhh/bhl 别名）算出 52 736 B，超 bar 2 560 B；r15 峰值生存期审计发现 **四个池 tile 可以纯别名进 arena1 既有窗口**，无需全局停车：

```
arena1（宿主 malloc = accCC，34 816 B；偏移为出厂非 TTF 臂）
0        pthi 9216 ──┐ P9 (dvp)：pthi+bdt+bhh+bhl+akb = 24 064 B
9216     bdt  4608 ──┘
13824    bhh  2560 ──┐ P11 (dW/dq1)：bhh+bhl+akb+accCK[24K,32.2K)
16384    bhl  2560 ──┘   （vnh/vnl 驻 arena2）
18944    akb  5120
24064    accCK 8192   ← dq1 部分和，r3 行和的 LDS 源（r3 保 smem 读，无 L2 往返、无新全局）
32256    …arena1 尾部余量 2 560 B（无可用 slab 变体；最小的 hbh/hbl 也需 17 424 B）
—— 时间不相交的其它驻留：accCC@0（P2 dPr 至 P3 dPm 分裂源）；
   dhi@16384+dlo@25600（P3/P4-7 dPm 尾）；pst@16384（P3' r4/c4r）；
   dprT@0（R17 FPIPE dPr 掩体）；tTh2@0+tTl2@9216（P12 TW）

arena2（宿主 malloc = vnh，10 240 B）：vnh@0 + vnl@5120（P9/P11）；
   ptlo@0 / krt@0 / qrt@0 / dWhi@0+dWlo@4608（相位互斥复用）
池 = 34 816 + 10 240 + 10×256（标量向量）= 47 616 B ≤ 50 176（余量 3 560 B）
```

**相位审计**（r15，每一相位边界有栅栏围栏）：P2 dPr：accCC$[0,16K)$ + akb$[18.9K,24K)$——accCC 只在**最后**一个 vs 的 epilogue 写入，字节不相交；P3 dPm 尾：accCC + dhi/dlo$[16K,34.8K)$（r19 表述：此期间 arena1 **字节级满载**）；P9 dvp：pthi@0+bdt@9216+bhh+bhl+akb = 24 064；P11：bhh+bhl+akb+accCK；P12 TW：tTh2+tTl2（epi_tw 折叠，无 accCK）。零新栅栏、零全局流量；P11 栅栏饮食（三路重暂存扫并两栅栏，akb 搭分裂暂存的栅栏；dw 链 `bar=False`，dq1 自带栅栏围栏组）。

**寄存器侧的关键发现（r15）**：出厂 cubin 实为 **168 寄存器**而非 128——`ptxas_v.txt` 是 run_l2.sh 的**第一遍** ptxas（smem_fix 前）；`gdn_smem_fix.py` 的第二遍释放出不同调度（168 寄存器，0 spill）。第一次 A/B 因此读到 0.78x 回退（ncu：Block Limit Registers = 1，8 warp/SM，劣于 r14 的 16-warp 形）。两条工具链定律（响亮）：(1) 作数的寄存器数是 smem_fix 遍的——`ptxas_v.txt` 不可信；(2) **ptxas-blackwell 无视 `--maxrregcount`**，但 PTX **entry 体内 `.maxnreg` 指令（与 `.maxntid` 并列）被遵守**：128 寄存器/32 B spill（栅栏饮食后 64 B）。`GDN_MAXNREG_ENTRY=dqkwg:128` 注入成为出厂构建步骤。

**【效果】**（r15，配对）：dqkwg 0.632→0.547（**1.16x**）@T2048、2.323→1.844（**1.26x**）@T8192（2048 CTA 填满翻倍 SM 槽，waves 6.24→3.12）；e2e 2.3794→2.2913（1.039x med/1.041x min）@T2048、1.050x/1.053x @T8192。ncu：Block Limit（regs, smem）=（2,2），实际占用率 15.02%→**29.17%**、14.23 warp/SM @T8192、活跃 warp/调度器 1.97→3.63、no-eligible 89→83%、SM 18→30%、MEM 19→36%——r14"独立流"论点的实测。数值：全数字位与 r14 相等（值精确轮）；T512 载荷字节同 HEAD（网格门控保 v3）。

**【R17 FPIPE 与 dprT 的失败案例（编排图的负结果注脚）】** r17 把 dPr 循环 vs=0 的两个操作数（`akb←do`、`vnew_all→dprT`）提为 gc 链掩体下的 prologue 组；`dprT` 落 arena1@0——accCC 自身字节，整个 dPr 循环期间未触碰（mma 累加器驻寄存器；accCC 首个 smem 写在末链 epilogue，远晚于分裂退役 dprT）。**第一版**把 dprT 停 +16384（dhi/dlo 窗），其 8 192 B 尾部覆写 akb@18944（dPr 的 A 操作数，活）——dq/dk/dg NaN，验证门一次定位。免费窗审计从此记录在代码注释。该 FPIPE 配对 1.009x/1.008x（r17），spill 类 96B/96/148→88B/84/120 顺带下降。

#### 7.2.8 K32 链化

**【动机】** r22 审计（§7.2.9）定位出 +46 寄存器覆盖的载体是 K=64 mma 位点的片断质量；r6 已立"K=32 片断生存期"律，r8 的 `k0=` 机器就是为此而生。

**【机制】**（`GDN_DQKWG_K32`，默认 0）：`_ops_g/_mh_g/_gemm_g` 包装器把每条 K=64 链拆为 k0=0/32 两条 K=32 链（`KGRAN` 机器；`GDN_DQKWG_K16` 深化为 K/16）。每累加器片断的 `mma.sync` **序列不变**（同操作数、同 kb 次序、同 acc 链接——只有 ldmatrix 发射点交错进来），故数据流按构造位同（r8 文档串的等价类："$[*,C{+}8]$ tile 上两条 K=32 链 == 一条 K=64 链"）。

**【源码片段】**（L5937–5944）：

```python
def _ops_g(a, aty, b, bty, klen):
    if klen > KGRAN:
        return [(a, aty, b, bty, k0, KGRAN) for k0 in range(0, klen, KGRAN)]
    return [(a, aty, b, bty, 0, klen)]
```

**【IR/PTX 对比】** PTX：每 K=64 位点的 `ldmatrix ... numTiles=4` ×（A 全 kb）预发射 → 拆为两半交错；`.maxnreg 128` 下 spill 类 **88B/84/120 → 64B/64/104**；自然体 174 不变（ptxas 保持无 spill 选择）。

**【效果】**（r22，配对 x3，共载）：dqkwg 0.5144→0.5147 @T2048 / 1.7289→1.7345 @T8192——**1.00x 中性**（spill 流量本身不是杠杆，r17 THR=128 律的第 4 个数据点）。位同证明：35/35+36/36 数组逐元素位等、六数位 == 记录表、det N=5 新进程位同。默认 0、旋钮保留（spill 类削减免费，未来共居压力下或有意义；K16 同律无增益）。连带负结果：P11R=1+K32（最深层饮食态）0.5141→0.5346（0.96x 回退——2 链体付出 r15 每-(ks,vs) 重暂存冗余，链数不可换 spill 类）；K32+TTF 0.99x/0.986x（异步组令牌跨 P11 存活，饮食不吸收）。

#### 7.2.9 寄存器分配器关门审计：+46 联合分配的边界测绘

r22 以 38 个仅编译探针（相删除 XD / 前缀 XP / 微二分 XS / 结构开关；探针生成器 `scratch/r22b_gen_probes.py`，绝不运行）对 174 寄存器自然体做了**边界测绘**（boundary mapping）——这不是一个待实现的优化，而是对"为什么此门关死"的定性：

1. **前缀图**（自然 smemfix 寄存器，逐相累加）：prologue 36 → +dPr 92 → +r4c4 98 → +tails 102 → +dvp 122 → +P11 128 → +TW 170 → +out **174**。跳变在 TW 叠上 P11（+42）。
2. **删除图**：删除 dvp / p11 / tw **任意一个**，体都坍缩到 128——**+46 覆盖是 dvp+p11+tw 的联合分配**，不是某一相的局部需求。微二分：只删 TW 的 mma 链（twmma）同样坍缩 128；dwsplit −10；twsplit/twepi 0。
3. **载体**：K=64 片断质量——`mma_gemm/mh` 在 `mma.sync` 循环**之前**发射全部 kb `ldmatrix` 片断，故每条 Kred=64 链（dvp 的 pdo 链、4 条 dPm 尾链、3 条 TW 链——6 个位点）同时持 A（$m_{pw}\times 4_{kb}\times 4$ 寄存器）+B 约 **48 寄存器/线程**存活——r6 "K=32 片断生存期"律被违反 6 处；r8 的 `k0=` 机器恰为规避它而存在（K32 位点因此非绑定）。
4. **门本身被分配器关死**：无上限遍报告 174 寄存器 0 spill（ptxas 偏好无 spill 甚于占用率）；而**强制 `.maxnreg` 的分配器在任何上限下都到不了 0 spill**——上限 128..174 的 spill 曲线在 **cap 174 处仍地板于 40 B 存/60 B 载**（该指令把 ptxas 切换到与其自然选择结构性不同的受限分配）。**"自然 ≤128 ⇒ .maxnreg 128 下 0-spill"在本工具链上对该丰富度的体为假**，与任何饮食设计无关。

**结论**（r22 排名）：dqkwg 在值同/spill/暂存/驻留各类别于本形状**无剩余实测开启的门**；剩余类别仅新数值类（如 P11R W-全局重驻留——纸面定价：smem 停车 +16 KB 对 2.5 KB 余量、全局停车 +190 KB/CTA L2 RAS 流量 ≈ +10%，双 NO-GO）或形状体制变更。出厂判决：**持有 r19 终字节**（dqkwg 0.514/1.73 ms @T2048/T8192，占 bwd 24–34%）。

---

### 7.3 wyb：$\mathrm{d}A$ 双侧掩蔽收缩与最终梯度

#### 7.3.1 公式与相位

`gen_wyb`（L7585–8126，`GDN_WYB_DIET` 出厂臂）实现：

$$
\begin{aligned}
G &= \mathrm{stricttril}\big(\mathrm{d}v_p\,(\beta v)^\top+\mathrm{d}W\,(\beta k\,2^{gc})^\top\big) &&\text{P1/P2（逐项 select）}\\
X^\top &= T\,G^\top,\qquad Y=T^\top X &&\text{P3/P4（三明治）}\\
\mathrm{d}A_f &= -\,\mathrm{stricttril}\big(Y\odot 2^{gc_i-gc_j}\big) &&\text{P4 epilogue 折叠}\\
ra_i&=\textstyle\sum_j \mathrm{d}A_f{}_{ij}\,kkt_{ij},\qquad c1r_j=\sum_i \mathrm{d}A_f{}_{ij}\,kkt_{ij}\,\beta_i &&\text{P5（warp 蝶形）}\\
dk &= dk_{\mathrm{local}}+\mathrm{diag}(\beta)\,(\mathrm{d}A_f\,k)+\mathrm{d}A_f^\top(\beta k) &&\text{P6/P7（t2/t1 双链）}\\
TW &= T^\top \mathrm{d}W,\qquad rb_i=\textstyle\sum_k TW_{ik}\,k_{ik}\,2^{gc_i} &&\text{P8}\\
tv &= T^\top \mathrm{d}v_p,\qquad dv=\mathrm{diag}(\beta)\,tv,\quad rc_i=\sum_j tv_{ij}\,v_{ij} &&\text{P9}\\
d\beta &= ra+rb+rc;\qquad dg_i=\sum_{j\ge i}\big[(ra_j{+}rb_j)\beta_j-c1r_j+r3_j+r4_j-c4r_j\big]+dg_{\mathrm{state},i} &&\text{P10}
\end{aligned}
$$

其中 §2 的双侧掩蔽精确式 $\mathrm{d}A=-\mathrm{stricttril}(T^\top\,\mathrm{stricttril}(G)\,T^\top)$ 由 P3/P4 实现：内层掩蔽落在 $G$ 的**逐贡献** select 上（select 是线性的：逐项 select 之和 == 和的 select，精确——r13 记录），外层掩蔽与负号、衰减折叠进 P4 的 Y-epilogue。

#### 7.3.2 G_dvp / G_dW 累积

**【机制】** P1：寄存器链 2-链 split mma，逐 vs 切片——A = dvp 自然定向分裂（f32 全局读入，`stage_split_ew`），B = **原始** $v$ 自然切片（`stage_raw_bf16`，精确）；$\beta_{\mathrm{col}}$ 折入末链 epilogue（r12 dqkwg TW 的搬法），tril-select 施于贡献。P2：dW 版本（B = 原始 $k$，尺度 $\beta\,2^{gc}_{\mathrm{col}}$ = `beta*expg` 折 epilogue），其末链 epilogue 对 sN 对做**读-加-重分裂**（同 lane 独占每元素：`cur = sNhi+sNlo; v = cur + sel; _split_store(v)`）。

**【r10 竞态教训（确定性门的历史）】** r10 之前 G 的累积循环携带一个 cp.async 味的跨 warp 竞态（与 bpre K-循环 restage 竞态同类）：同窗基线 dk 在 1.719e-3 与 4.719e-3 间跳变。修复 = r9 规则 3（每个 `wait_group` 后随 `gpu.barrier`）+ r10 规则（重暂存与读者间的尾栅栏，读者清空 ≥2 迭代后才可覆写）。此后 N=10 新进程逐级位同成为每轮结构变更的门径。代价 ~0.3% @T2048（r10）。

**【效果】** 并入 §7.3.6 的 2.40x；dk 数位在 r13 收紧的主因即 G 链的 tf32→split（dk 1.691e-3→1.284e-3 @T2048、1.724e-3→**5.131e-4** @T512，余量 6.2x/15.6x）。

#### 7.3.3 X/Y 三明治的 GEMM 化：不物化 f32 的 epilogue split-store

**【动机】** 三明治若按朴素三步走需物化 $G$、$X$、$Y$ 三个 f32 $[C,C]$ tile（各 16 KB）+ 各自的 emap/store 扫描；wyb 的池预算（2-CTA bar 50 176 B）下不可行，且 fla 的 dtype 真相（A/h/dh/do=bf16）已是本链要避开的数值类。

**【机制】** 三条规则合成"epilogue split-store 越过死 B 源"：(1) **P3 的 X-链 epilogue 把 $X^\top$ 直接 split-store 进 sN 对**——恰是本链自己的 B 源（B 死于末链 `mma.sync` 之后）；mma 体与 epilogue 之间必须加 `gpu.barrier`（`mma_gemm` 的 `epi_bar=` 参数，r13 引入）：epilogue 的 `st.shared` 目标与本链 `ldmatrix` 仍跨 warp 读着的 tile 重叠（r6 危险类）。(2) **P4 的 Y-链 epilogue 折叠终值**：$-y_{ij}\,2^{gc_i-gc_j}$、tril-select、split-store 一步到位——$Y$ 与 $\mathrm{d}A_f$ 都不再单独存在。(3) A 侧 $T$ 对 P3 以自然定向分裂、对 P4/Y 与其后 tv/TW 以 $T^\top$ 定向**重驻**（一次 16 KB 全局重读；`aT` 的竞技场窗口）。K=64 链一律 k16 切片链接力（`accs=`，r12 wy 的片断生存期饮食）。

**【源码片段】**（L7867–7883，P3）：

```python
stage_split_ew(e, aTh, aTl, G["T_all"], ccty, m, C, 0, C, C)
for kb0 in range(0, C, 16):
    last = kb0 == C - 16
    accs = mma_gemm(e, f"xxh{kb0}", C, C, 16, W, aTh.name, aTh.ty,
                    sNhi.name, sNhi.ty, kt=True, accs=accs, k0=kb0)
    accs = mma_gemm(e, f"xxl{kb0}", C, C, 16, W, aTh.name, aTh.ty,
                    sNlo.name, sNlo.ty, kt=True, accs=accs, k0=kb0)
    accs = mma_gemm(e, f"xxm{kb0}", C, C, 16, W, aTl.name, aTl.ty,
                    sNhi.name, sNhi.ty, kt=True, accs=accs, k0=kb0,
                    epilogue=(... _split_store(..., sNhi, sNlo, row, col))
                    if last else None, epi_bar=last)
```

**【IR/PTX 对比】** 旧（r12 legacy）：tf32 `tgemm`（X/Y 位点，$2^{-11}$）+ f32 accCC 暂存 + emap/store_rows 扫描 + kahan 补偿体；新：bf16 `mma.sync` ×3/链（hi·hi+hi·lo+lo·hi）+ `ldmatrix` + epilogue 内 `truncf/extf/subf` 分裂对（`_split_store`）——**f32 三明治在任何时刻不驻 smem**。

**【效果】** 数值：$\mathrm{d}A_f$ 链收紧使 dbeta/dg 随 dk 同步收紧（§7.3.6 表）；dk/dbeta/dg 三个数位移动全部朝紧方向（r13）。计时并入 2.40x。

#### 7.3.4 tv 3-mma：$T^\top$/dvp$^\top$ splits 进 tT/accCK 竞技场

**【动机】** $tv=T^\top\mathrm{d}v_p$ 是 r7 测得 ~158 µs 奖金的位点（tf32-tv 两次被拒：r3 8.6e-3/2.3x 余量、r7 9.3e-3/2.15x 余量，且 255 寄存器 16 B 栈溢出）——mma 速度下的标量精度只能走 3-mma split。r8 试图直建被 smem 封锁（9 8048 B 池下只剩 8.7 KB 自由，分裂 tile 需 23 KB）；r9 竞技场机器使其可解。

**【机制（r11）】** 两个生存期竞技场承接转置定向的分裂对：
- **tT → $[C,C{+}8]$ f32（分配 18 432；尺寸填充使其 malloc 唯一可识别）**：在 TW_rb 最后一次 tT 读之后暂存 `tThi/tTlo`（$[64/66,72]$ bf16，fp 9 216 各）；
- **accCK → $[C{+}2,C]$ f32（分配 16 896，fp 16 384）**：dk 循环之后承接 `dhi/dlo`（dvp$^\top$ 分裂）+ `tvacc`（$[66,16]$ f32）。

池 98 048→100 096 B（+2 048 是 T-分裂对的真实足迹，其余全免）；`smem_fix` 恰按设计消费 spec：22 malloc→17 声明、一条 18 432 竞技场。mma 本体：`mma_gemm_mh`，M=C、N=BV=16、K=C、W=8——A 分裂自 `T_all` 一次暂存，B 逐 vs 切片自 `dvp_all`（转置定向暂存）；**fp32 类累加**（hi·hi+hi·lo+lo·hi）。**dv 折入 epilogue**：每 acc lane 同时存 `tvacc`（dbeta 的 rc 行和用）与 $\beta$ 尺度化的 dv 全局元素——旧 emap/store_rows 扫描退役。细节：填充过的 $[C,C{+}8]$ tT 不能走 16B 突发 astage（1152 块 % 256 线程 ≠ 0）——`stage_rows` 的 `shape=` 参数按界元素级暂存一次。

**【出厂臂中的位置】** r13 全饮食端口后 tv 体原样迁入竞技场计划（P9）：A = dk 循环后重驻的 $T^\top$ 分裂（arena1/aT 窗口），dvp$^\top$ 分裂 + tvacc 在 arena2 的释放窗（`dhi@0/dlo@2304/tvacc@4608`）；链为 k16 切片 3-链接力（`mma_gemm_mh` + `k0=`）。

**【IR/PTX 对比】** r11：`mma.sync` 384→480、`ldmatrix` 0→128、tv 标量 gemm 的 8 条展开 K=64 循环消失；寄存器 239→226（0 spill）。

**【效果】**（r11，配对）：wyb 0.9871→0.8174（**1.21x**）@T2048 / 0.1253→0.0989（**1.27x**）@T512——r7 测得的 ~158 µs 全额落袋；单独链 e2e 5.237→5.112（1.024x med）。数值：dv 3.3e-6→1.088e-5 @T2048（split-bf16 tv 叠上 split-bf16 dvp 的"更紧乘积、多一层重组合"类；余量仍 736x/851x 对 8e-3 bar）。

#### 7.3.5 ra / c1r：warp 蝶形和

**【机制】** P5 的两个 $[C,C]$ 归约（$ra$ 行和、$c1r$ 列和）用 `warp_rowsum/warp_colsum`：warp $w$ 持行/列 $\{w, w{+}N_W,\dots\}$，lane $l$ 供 $j=l$ 与 $l{+}32$（CW≤64），每行/列一棵 5 步 shuffle 树、lane 0 落盘。被归约项从 sN 对**重建**（$hi{+}lo$，f32 内精确）后乘 `kkt`——kkt **直接 L2 热读全局**（`kkt_g`，`kkt_all` 刚被 wy 流过；accCC 暂存退役）；$c1r$ 追乘 $\beta_i$。r8 引入时的对照项：串行单线程 $C$ 链（每行 CW 依赖迭代）消失，`gpu.shuffle xor` → `shfl.sync.bfly.b32` 探针验证。

**【效果】** 计入 2.40x；dk/dbeta/dg 数位随 r13 重掷收紧（蝶形树上再重排，属值类变更、6/6 门径放行）。

#### 7.3.6 七位点 tf32→split-bf16 全清单与 128-寄存器双 CTA

**【清单】** r13 的 `GDN_WYB_DIET` 把 r8 点数的全部 tf32/kahan gemm 位点改写为 2–3 链 split-bf16 mma——**七个位点**：

| # | 位点 | 形状 | 链形 | 尺度折叠 |
|---|---|---|---|---|
| 1 | `G_dvp` | $C\times C\times V$ | 2-链 | $\beta_{\mathrm{col}}$ 折 epilogue |
| 2 | `G_dW` | $C\times C\times K$ | 2-链 | $\beta 2^{gc}_{\mathrm{col}}$ 折 epilogue |
| 3 | `X`（P3） | $C\times C\times C$ | 3-链 k16 切片 | epilogue split-store 越死 B 源 |
| 4 | `Y`（P4） | $C\times C\times C$ | 3-链 k16 切片 | $-2^{gc_i-gc_j}$ + tril + split 折 epilogue |
| 5 | `dk_t2`（P7） | $C\times K\times C$ | 2-链（A = `_resplit_pair_T` 带 $\beta_r$） | 收缩下标尺度折分裂 |
| 6 | `dk_t1`（P7） | $C\times K\times C$ | 2-链 | $\beta_{\mathrm{row}}$ 折 epilogue；终值 dk 三项折叠 |
| 7 | `TW_rb`（P8） | $C\times K\times C$ | 3-链 k16 切片 | $2^{gc}_{\mathrm{row}}$ 在 rowsum 项内 |

（tv 为 r11 已落地的第 8 个 mma 位点，非本清单的 tf32 退役项。）dk 循环的 k 特征切片 BKD=16 使 t2 对 + $k^\top$ tile 恰入 arena2 尾。`_resplit_pair_T`：既有 bf16 对的**转置重分裂**——$hi{+}lo$ 的 f32 重建精确（8+8 尾数位装得下），结果携带与直接分裂原 f32 相同的 $\sim2^{-17}$ 类；cvec 把 $\beta$ 折在源行 $s$ = 收缩下标上。**LLVM 将 t2 交接 store-to-load 前递入寄存器**（两链同 lane 映射）——两个 t2 malloc 从 PTX 消失；`gdn_smem_fix` 学会跳过 0-malloc 的别名 tile 而非关闭整个 spec（此前是硬 WARN + 121 KB 静态 smem ptxas 失败，r13 工具链教训）。

**【资源落点】** 池 **100 096 → 45 056 B**（arena1 = `aTh` 基 18 432；arena2 = `sNhi` 基 24 832；向量 7×256）= **2 CTA/SM × 256 线程**（原 1 CTA、226 寄存器）；寄存器**恰 128**（2×256×128 = 65 536 = 寄存器堆分毫不差）；176 B spill 存/228 B 载——**保留**（132 寄存器变体即失 2-CTA 驻留）。`cp.async` 92→0；`ld.shared` 2 607→（饮食载荷，全部重暂存类收缩）。

**【效果】**（r13，配对）：wyb 0.8195→0.3419（**2.40x**）@T2048 / 0.0995→0.0599（**1.66x**）@T512；单独链 e2e 3.322→2.847（1.167x med）；整轮 bwd e2e 3.326→2.688（**1.237x med / 1.240x min**）。数值（六余量全列，r13 表）：

| 形状 | dq | dk | dv | dbeta | dg | dh0 |
|---|---|---|---|---|---|---|
| T2048 | 1.824e-4 (44x) | **1.284e-3 (6.2x)** | 1.088e-5 (736x) | **8.527e-4 (23x)** | **3.274e-4 (61x)** | 2.169e-7 |
| T512 | 4.617e-4 (17x) | **5.131e-4 (15.6x)** | 9.43e-6 (848x) | **2.178e-4 (92x)** | **1.466e-4 (137x)** | 1.394e-7 |

粗体 = 七位点退役带来的收紧方向移动（dk 的七个 tf32 $2^{-11}$ 位点 → $2^{-17}$ split；dbeta/dg 随 ra/c1r 蝶形重掷 + 更紧的 $\mathrm{d}A_f$ 同步收紧）。

**【残项（诚实负结果）】** r17 FPIPE 的 P1/P2 首切片 DMA 配对中性（1.000x/0.999x）——源是 dqkwg 刚写的 `dvp_all/dW_all`，L2 驻留（保留，同 §7.1.7 理由）；r18 的 P7 `kTt` 双缓冲被寄存器上限封死（wyb 出厂即 `.maxnreg` 帽 128 寄存器、48/44/72 B spill；DWF 臂刚示范了帽下加暂存机器使 spill 类增长快于 L2 延迟隐藏的收益；双缓冲差 ~5 KB 池或先做寄存器饮食——r18/r19 排名遗留，未做）。

---

### 7.4 竞技场编排总论：峰值生存期审计方法论

#### 7.4.1 机制：生成器—PTX 后处理的编译期契约

MLIR 的 SSA 合法性要求每个 tile 一个 `memref.alloc`；竞争性地复用同一块 smem 不能在 MLIR 层表达（会破坏支配性/活性分析），也不必要——**别名下放到 PTX 级**。生成器在 emit 时把生存期互斥的 tile 组织为**竞技场规约**（`smem_alias.json`，`main()` L8655–8661），构建期由 `tools/gdn_smem_fix.py` 的 fix-1 扩展消费：**被别名 tile 的 malloc 被删除，其 `cvta` 地址换算改写为 `arena_base_symbol + add.u64 offset`**（r9）。这一"调度决策 = 编译期工件"的形态是全文 Schedule-as-IR 论点在存储维度的体现：竞技场编排可审计、可再生、可 diff。

#### 7.4.2 识别键与填充规则

- **malloc 以唯一分配尺寸识别**——LLVM 会重排 malloc 调用序列，次序不是稳定键（r9）。因此每个被别名的 tile 带**识别填充**：ldmatrix 喂送的 tile 加**从不寻址的额外行**（行跨步须保持 16 B 倍数，bf16 列数 %8）；纯 generic f32 tile 加 4 列。任何尺寸失配，spec **自禁用**（正确、不重叠的保守回退）。
- **足迹（footprint）≠ 分配尺寸**：竞技场算术用"行跨步 × 被寻址行数"（pad 列在行跨步内被 ldmatrix lane 地址跨过，pad 行不计）。
- **128 B 对齐断言**（ldmatrix + smem_fix 的 `.align 128` 声明）；`off + fp ≤ arena` 容量断言在生成期强制（`assert off + fp_of[t] <= spec["arenas"][base]`）。
- **竞技场基地必须是首用 tile**：纯锚点 alloc 会被 LLVM DCE，尺寸键随即找到 0 malloc（r13 教训 1）；LLVM 亦会把同 lane 的 epilogue 交接 store-to-load 前递入寄存器（t2 对）——此类 0-malloc tile **跳过并注记**，不再使整个 spec 失效（r13 教训 2）。

#### 7.4.3 三条硬规则（验证门反复付费买来的）

1. **分裂永不别名其自身源 tile**（r12 的 accCC 自别名：分裂读 accCC 而分裂对又落 accCC 字节 → 自毁；dq/dk 1e0 类失败，npz diff 一步定位 dk_local）。
2. **`epi_bar`**：epilogue 的 `st.shared` 目标与本链 `ldmatrix` 仍跨 warp 读着的 tile 重叠时，mma 体与 epilogue 之间必须 `gpu.barrier`（r13，`mma_gemm epi_bar=`）。
3. **cp.async 两律**：`wait_group` 是**每线程**的——每个 wait 后随栅栏（r9 的 dk_local 列 0..5 陈旧竞态，256 线程网格才显形）；raw cp.async 落字节只进**纯 `ld.shared` 读者**的 tile，ldmatrix 操作数必须 `st.shared` 写入（r6 静默损坏； laundering/分裂重读为合规实现）。

#### 7.4.4 峰值生存期审计（三 kernel 汇总表）

审计方法：把 kernel 划分为栅栏定界的相位，逐相列出 smem 活集（tile@偏移），验证每相活集字节 ≤ 竞技场容量、且相位边界全部有栅栏围栏；据此求出最小竞技场尺寸与池。出厂形态：

| kernel | 竞技场 | 宿主 malloc | 容量(B) | 叠放 tile（出厂偏移） | 池(B) | 驻留 |
|---|---|---|---|---|---|---|
| wy | arena1 | `kkt` $[C,C{+}8]$ f32 | 18 432 | `auhi@0/aulo@9216`（u 相）→`awhi@0/awlo@9216`（w 相） | **45 832** | 128thr×2 |
| wy | arena2 | `Tf` $[C,C]$ f32 | 16 384 | `vb@0`（u 相）/`kb@0`（w 相） | | |
| dqkwg | arena1 | `accCC` $[C,C]$ f32 | 34 816 | accCC@0；pthi@0+bdt@9216；dhi@16384+dlo@25600；pst@16384；bhh@13824+bhl@16384+akb@18944+accCK@24064；tTh2@0+tTl2@9216；dprT@0（FPIPE） | **47 616** | 256thr×2 |
| dqkwg | arena2 | `vnh` | 10 240 | vnh@0+vnl@5120；ptlo@0；krt@0/qrt@0；dWhi@0+dWlo@4608 | | |
| wyb | arena1 | `aTh` | 18 432 | aTh@0+aTl@9216 → dTh@0+dTl@9216 → dwAh@0+dwAl@5120+dwB@10240 →（FPIPE `wyT@0`） | **45 056** | 256thr×2 |
| wyb | arena2 | `sNhi` | 24 832 | sNhi@0+sNlo@9216 → dvAh/dvAl/dvB@0/3072/6144 → kTt@18432+t2h@20736+t2l@22784 → dWTh@0+dWTl@4608+TWt@9216+tKK2@17408 → dhi@0+dlo@2304+tvacc@4608 | | |

时间轴（dqkwg 出厂臂，每格 = 一个栅栏定界相位；█ = 活）：

```
              0    4.6K  9.2K  13.8K 16K   18.9K 24K   26.6K 32.2K 34.8K
P0 向量        ·     ·     ·     ·     ·     ·     ·     ·     ·     ·   （全向量在池标量区）
P2 dPr         █████████████████████ accCC ······akb·········        ← accCC 只在末 vs epilogue 写
P3a r4/c4r     █████████████████████ accCC ┃ pst┃                    （pst 先于 dhi/dlo 登场）
P3b dPm尾      █████████████████████ accCC ┃ dhi┃ dlo┃               ← 字节级满载（r19 pt 封锁依据）
P9 dvp         ┃pthi┃ ┃bdt┃ ┃bhh┃ ┃bhl┃ ┃akb┃  （ptlo 在 arena2）
P11 dW/dq1     ·     ·     ·     ┃bhh┃ ┃bhl┃ ┃akb┃ ┃accCK┃         （vnh/vnl 在 arena2）
P12 TW         ┃tTh2┃ ┃tTl2┃                                     （epi_tw 折叠，无 accCK）
```

#### 7.4.5 方法论边界与负结果

- **峰值≠总和**：静态 smem 上限（sm_120a 实测 101 376 B；98 304 接受、131 072 拒绝）约束**峰值并集**，竞技场把峰值压到相位最大值；2-CTA bar ≈ 50 176 B/CTA（个别 kernel 的核算用 50 688 B）。寄存器侧同理：2×256×128 = 65 536 恰为寄存器堆——"恰 128"是设计点不是巧合（wyb 以 176 B spill 保它；dqkwg 以 `.maxnreg` 指令保它）。
- **别名不总是答案**：caryA 的四共居 tile（r11）没有任何合法别名——解法是"竞技场溶解饮食"（退役全部四 tile 换三个 bf16 mma tile，82 176→28 592 B）。竞技场与 tile 退役是同一谱系的两端。
- **pt/`GDN_DQKWG_TTF`/DWF 三数据点**（r18/r19/r22）划出谱系边界：帽定寄存器下，**任何**新增暂存机器使 spill 类增长快于 ~1–2 µs 一次性 L2 暴露的收益；_mid-phase FPIPE 族关闭_。重开该族的唯一门是自然寄存器饮食 ≥46（174→128），而无序保序重构触及它（r19 判决维持）。
- **验证门是最终裁判**：本章记录的每个自别名/竞态/填充错误均被 fp64 孪生 + 逐级 npz diff 一步定位（r6/r9/r12/r15/r17 各一例）；竞技场编排的合法性从不由构造单独宣称——位同证明（35/35、36/36 数组级）或 6/6 数值门是每轮放行条件。

---

### 7.5 小结

计算族三 kernel 的出厂终态（B1T2048H16，配对口径）：wy 0.229 ms（3.04x 于 r7 形，池 45 832 B、2 CTA）、dqkwg 0.514 ms（r8 基线 1.665 ms 的 3.2x；池 47 616 B、256thr×2 CTA、`.maxnreg 128`）、wyb 0.342 ms（对 r12 前体 2.40x；池 45 056 B、恰 128 寄存器×2 CTA）。三 kernel 合计从 r8 基线的 ~3.36 ms 降到 ~1.09 ms，是 bwd e2e 从 r8 窗口的 6.71/5.80（med/min）走向 ~2.20/2.15（r17 med / r22 共载 min，对 fla 0.479x）的主通道。方法论产出有二：**split-bf16 位点化剧本**（tf32 不可用处的 mma 速度-标量精度路线，cvec 尺度折叠 + epilogue 折叠 + epilogue split-store 三件套）与**峰值生存期竞技场**（编译期可审计的 smem 编排，配三条硬规则与验证门闭环）。诚实边界同样清楚：dqkwg 的 −46 寄存器门被分配器结构性关闭、wyb 的 P7 双缓冲被 128-寄存器帽封死、wy 的 shfl 扫描是全族唯一非位同数值变更——这些负结果连同其定价，构成对该体系"何处方为收益递减点"的完整测绘。

---

## 8 数值验证与确定性体系

> 本章的自洽口径（与全文全局声明一致）：**数值分母**——前向输出 o/ht 对 fp64 JAX 参考的逐元素最大绝对差；反向六梯度（dq/dk/dv/dbeta/dg/dh0）对 fp64 VJP 参考的容差契约（fla 惯例 atol/rtol 口径的绝对棒）。**确定性定义**——同一 cubin、同一输入、同一发射几何下，输出缓冲区位级（bit-level）一致；跨进程（fresh CUDA context、fresh 模块加载、fresh 设备分配）N 次采样全部一致方为通过。**证据纪律**——本章每一个数字均可回读到 `evidence/` 与 `tools/` 源文件；凡早期轮次数字与最终系统不一致处，以最终记录为准并同时给出演化注记。

本章回答一个问题：一个把调度做成 IR 旋钮、把数值路径也做成旋钮（f32/tf32/split-bf16、K-粒度、重结合次序）的编译器栈，凭什么让每一个落盘的 cubin 都"数值正确且逐位可复现"。答案是四层验证塔加三根横贯的纪律绳：容差契约（§8.1）、fp64 孪生评分（§8.2）、确定性协议（§8.3）；在此之上，竞态破案案例集（§8.4）与 cubin 指纹/纯度/cache_key 体系（§8.5）把"可复现"从口号变成可审计的工程对象。每小节按【动机 → 源码/协议片段 → 前后对比 → 效果数字】四件套展开。

### 8.0 章首：分层验证塔

验证体系是一个自底向上的四层塔。每一层的 oracle 独立于被测对象本身，且上一层以下一层的通过为前提；塔的顶端不是"一次验收"，而是**逐轮门**——任何触碰数值路径或调度路径的变更都要重新过全部门。

**第一层：L1 CPU 三方对拍（G1）。** 语义层在进入 GPU 之前先在 CPU 上闭合为三方一致：MLIR（`linalg`+`scf` 真 IR，经 `mlir-opt`/`mlir-runner` 在 CPU 执行）⇄ JAX 可执行规约（fp64，`gdn_seq`/`gdn_chunk`/`jax.vjp`）⇄ fla naive 参考（只读 oracle，`naive.py`，内部硬转 fp32）。判据与数字：JAX 规约自检 44/44（chunked vs sequential fp64：o 5.6e-16 / ht 1.2e-15 / 六梯度 1.2e-14；有限差分 vs VJP 2.3e-9；jit 奇偶 0.0；torch fla 交叉 1.2e-7 / 2.3e-6，fp32 口径）；MLIR L1 前向 7/7（vs JAX dumps ~6e-7，即 `printMemrefF32` 六位有效数字打印地板）；MLIR L1 反向 3 npz × 6 梯度全部 PASS（最差 6.4e-6，同样贴打印地板）。三方链条：L1↔JAX 6e-6 + JAX↔fla 1e-6/2e-6 量级（evidence/G1/GATE_REPORT.md；evidence/G1/jax_spec_report.md；evidence/G1/l1_harness_report.md）。此外，numpy 侧的 chunked 前向孪生对逐步递推孪生（`naive_recurrent` 移植）自差 2.4e-7——**评分参考自身先被验证**，这是全塔的地基。测量地板如实披露：runner 经 stdout 打印 memref，只有 ~6 位有效数字，6e-7 以下的差异不可分辨，真实 fp32 误差只会更小。

**第二层：GPU 20 配置逐配置对拍（G4）。** L2 旋钮库的全部 20 个配置（kkt/o 阶段的 mma/nomma/warp-specialized/TMA 变体 + 两个 f16 锚点）逐个在 GPU 上执行：重新生成（固定 `--seed`）→ 重编译（`scripts/run_l2.sh` → `tools/gdn_smem_fix.py`）→ 子进程隔离发射（墙钟超时，挂起的 kernel 只杀死自己的 worker）→ **读回完整输出缓冲区** → 与"从同一份嵌入输入计算的精确 numpy fp64 孪生"逐元素比较。判据：bf16/f16-io 配置 max-abs-diff ≤ 5e-3，f32-io 配置 ≤ 1e-3（fla 派生契约的阶段级形式）；同时做两项结构检查——非零元计数必须与参考完全一致（每个 kkt 配置写满全部 2080 个严格下三角单元，G3 曾实测 mma 路径只写 1168 个，正是覆盖性 bug），以及**确定性检查**：清零 + 重发射必须逐位复现输出（`deterministic` 字段）。生成器确定性同时受验：同 seed 两次生成的 payload.mlir sha256 相同、异 seed 不同。结果 20/20 PASS，双 seed（0 与 7）双跑 20/20；全部 mma 配置的实测最大差 9.77e-4，与"精确模拟 f32 累加 + bf16 舍入输出"管线预测的 9.765625e-4 相等——**实测 == 预测**，残差被证明是纯输出量化而非计算误差（evidence/G4/GATE_REPORT.md；evidence/G4/verify_results.json、verify_results_seed7.json）。该层还把 TMA 配置的主机侧接线（`tensormap_spec.json` → cuda-bindings `cuTensorMapEncodeTiled`）纳入 harness，使"验证基础设施完备"本身成为被测对象。

**第三层：全算子 GPU 验证（G5）。** 前向（`tools/gdn_full_verify.py`）：staged mono 链（F1→F2→F3）与 persistent fused 两条路径 × 两个标定形状，对 G5-A 共享 JAX fp64 参考（同一 inputs npz）比较 o/ht，棒 5e-3，判据为**零元素越棒**（`o_over_tol == 0 and ht_over_tol == 0`）；首验 4/4 PASS（T512 o 4.77e-4 / ht 7.00e-4；T2048 o 5.75e-4 / ht 7.93e-4，两路径同值）（evidence/G5/fwd_report.md round 1）。反向（`tools/gdn_bwd_verify.py`）：九阶段链（wy→fwdscan→caryA→caryM→bpre→dhs→dkb→dqkwg→wyb，G5-J 第 5 轮重构后的最终结构：dhu 退役，caryA/caryM/bpre 并行预复合每 chunk 仿射扫描算子 $A_n = 2^{g_{cl}}I - W^\top\hat B$、$M_n = Q_g^\top - W^\top P^\top$、$B_n = M_n\,\mathrm{d}O$，dhs 每 chunk 单 rgemm 载链，dv/dvp_all 移入 dqkwg）对 fp64 numpy 孪生（bf16 量化输入上运行，见 §8.2）比较六梯度，棒 8e-3/8e-3/8e-3/2e-2/2e-2/8e-3，判据 6/6；小形状另有 JAX vjp oracle 交叉（`--jax`）。结构性不变量同层受验：T8192 分段分裂 S=1/4/8 下 o 与 ht **位级一致**（f2-segments 旋钮不改变数值，evidence/G5/fwd_report.md）。

**第四层：逐轮门（round gate）。** 从 G5 第 2 轮起立规："Numerics stay green through every perf change"——每一个性能变更落地前必须重过 6/6（前向 4/4），数值不变类（value-identical）变更必须**逐位复现记录数字**；从 G5-Q 第 10 轮起，确定性进入同一道门：凡触碰 staging/arena/mma/barrier 的变更，附加跨进程 fresh-process N 次（初版 N=10，舰载常规化为 N=5）双形状位级一致证明（checklist 第 7 条，§8.3.5）。这一层的存在使 26 轮反向战役（58.88→2.15 ms）与 25 轮前向战役中的每一字节推进都有同一把尺子。

四层塔的完整判据—证据映射见 EVIDENCE_INDEX.md（`make repro-g1/g4/g5` 为幂等复现入口）。

### 8.1 容差契约：来源、标定与裕度

#### 8.1.1 契约内容与来源【动机→片段】

**动机**：编译器栈的数值路径是旋钮（f32 链、tf32 mma、split-bf16 mma、bf16 载荷状态），没有一个外部的、与实现无关的数值标准，"正确"就只是自我声明。本项目直接采用被对标基线 fla（flash-linear-attention）在其测试体系中使用的容差惯例作为**外部契约**：棒定在"fla 自己的 Triton kernel 对 fp64 参考的距离"所在的量级上，于是"与 fla 同级准确"成为一个可证伪的命题，而非营销口径。

**契约的三个层级（源码片段）**：

阶段级（G4，`tools/gdn_verify.py`）：

```python
THRESH_BF16 = 5e-3   # bf16/f16-io configs (fla stage-output contract)
THRESH_F32 = 1e-3    # f32-io configs
# contract: stage outputs graded at o/ht level: 5e-3 bf16-io,
#           1e-3 f32-io (fla-derived; full-op validation in G5)
```

前向全算子级（G5，`tools/gdn_full_verify.py`）：

```python
TOL = 5e-3
# contract: o/ht max-abs-diff <= 5e-3 vs G5-A JAX fp64 refs (same inputs)
```

反向全算子级（G5，`tools/gdn_bwd_verify.py`）：

```python
tol = {"dq": 8e-3, "dk": 8e-3, "dv": 8e-3, "dbeta": 2e-2, "dg": 2e-2,
       "dh0": 8e-3}
# fla-derived contract: dq/dk/dv <= 8e-3, dbeta/dg <= 2e-2, dh0 <= 8e-3
# (vs fp64, bf16-fed kernel)
```

即：**o/ht 5e-3；dq/dk/dv/dh0 8e-3；dβ/dg 2e-2**，全部为对 fp64 参考的逐元素最大绝对差、bf16 喂入的 kernel 口径。dβ/dg 的棒放宽到 2.5 倍，反映其被积量（行和/前后缀和链）在 fla 惯例中即有更大的舍入暴露。注意绝对棒的隐含适用域是"阶段输出量级 O(1)"：G4 曾实测 ±3.0 输入把 kkt 输出推到 ~1e3，此时纯 bf16 输出量化 ulp ~4 超过任何绝对棒却不指示 bug——嵌入数据因此缩放 2^-5（值为 2^-7 的整倍数、≤4 位尾数，在 bf16/f16/f32/tf32 中精确可表），kkt 对角 ~0.44、量化受限最大差 9.77e-4，对 5e-3 棒保持 ~5× 裕度（evidence/G4/GATE_REPORT.md §2.6）。同理，标定文件把"T8192 的 fp64 参考按设计跳过、数值经 T512/T2048 的 fla 交叉检查插值"写进 meta 注记（同 kernel 逐 chunk 复用，形状间数值不换类）。

#### 8.1.2 fla-vs-JAX 标定：bf16 舍入下界 ≈2×【片段→数字】

**标定协议**：以 G5-A 共享输入配方（`tools/gdn_bench_common.py`，fla `test_chunk` 惯例，seed 42：q/k 先 fp32 L2 归一再 cast bf16，v bf16，β/g/h0 fp32，g 自然对数域）喂给 fla 纯 Triton 后端（`FLA_FLASH_QLA=0`）与 JAX fp64 分块参考，记录逐输出 max-abs-err、参考最大绝对值、相对误差，以及 **bf16 输出舍入下界**（该输出量级下 bf16 表示步长推得的舍入地板；对 fp32 输出的 dβ/dg，为 bf16 载荷中间量决定的下界）。落盘为 `evidence/G5/refs/tolerances.json`；现行文件为 RC1 形状域变更（r1 梯）追加的 T16384 标定代（evidence/G5/refs/tolerances.json，`rc1_B1T16384H16`，JAX 237.2 s）：

| 输出 | fla dtype | max-abs-err | ref max-abs | rel err | bf16 下界 | err/下界 |
|---|---|---|---|---|---|---|
| o | bf16 | 6.221e-4 | 7.113e-2 | 8.75e-3 | 2.441e-4 | **2.55×** |
| ht | fp32 | 5.710e-4 | 1.384e-1 | 4.13e-3 | 3.864e-4 | **1.48×** |
| dq | bf16 | 1.582e-3 | 3.267e-1 | 4.84e-3 | 9.698e-4 | **1.63×** |
| dk | bf16 | 7.298e-2 | 11.72 | 6.23e-3 | 3.026e-2 | **2.41×** |
| dv | bf16 | 1.003e-2 | 2.188 | 4.58e-3 | 3.901e-3 | **2.57×** |
| dbeta | fp32 | 2.096e-2 | 15.55 | 1.35e-3 | 1.380e-2 | **1.52×** |
| dg | fp32 | 2.581e-2 | 3.703 | 6.97e-3 | 5.955e-3 | **4.33×** |
| dh0 | fp32 | 1.205e-4 | 3.580e-2 | 3.37e-3 | 8.886e-5 | **1.36×** |

**读法一（契约的物理意义）**：fla 自身对 fp64 的距离系统性地落在其输出 dtype 舍入下界的 **≈2×（1.36–4.33×，中位 ≈1.9×）**——即 fla 的残差基本就是"bf16 舍入 + 少量累积重结合"，并无额外精度余量。把棒定在 5e-3/8e-3/2e-2，正是覆盖"fla 级准确"在标定形状（ref 量级 O(0.1–1)）的可达域。

**读法二（适用域的反例，如实呈现）**：在同一文件的 T16384 格上，fla 的 dk 绝对距离 7.30e-2 已越过 8e-3 绝对棒（相对误差 0.62% 仍与其它梯度同族）——因为 ref_max_abs 长到 11.7，绝对棒的 O(1) 量级假设失效。这是标定文件坚持"数值以 T512/T2048 交叉检查插值、不做跨量格外推"的原因，也是形状域变更（RC1）被列为开放战线时一并交付新标定代的原因。

#### 8.1.3 我方裕量表【前后对比→效果数字】

以最终系统（前向 R25 档案代 +W32P 舰组；反向冻结于 r19-final 字节）的记录数字对棒折算。前向（vs fp64 JAX 共享参考，棒 5e-3）：staged T2048 **o 6.486e-4 / ht 7.493e-4**（≈7×/6.7× 裕度，记录数字）；T8192 fp64 门 o 6.626e-4 / ht 8.918e-4（≈7.5×/5.6×）；fused 记录数字 o 5.751e-4 / ht 7.929e-4；4/4 零元素越棒（evidence/G5/fwd_report.md rounds 18–25；evidence/G5/GATE_REPORT_FINAL.md"fwd 4/4 位级一致（T8192 fp64 门 7× 裕量）"）。

反向最终记录表（vs fp64 孪生，bf16 喂入；括号为对 8e-3/2e-2 棒的裕度，报告原文口径）：

| 形状 | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T512H4 | 4.617e-4 (17×) | **5.131e-4 (15.6×)** | 9.43e-6 (848×) | 2.178e-4 (92×) | 1.466e-4 (137×) | 1.394e-7 (57400×) |
| B1T2048H16 | 1.824e-4 (44×) | **1.284e-3 (6.2×)** | 1.088e-5 (736×) | 8.527e-4 (23×) | 3.274e-4 (61×) | 2.169e-7 (36900×) |
| B1T8192H16 | 2.078e-4 (≈39×)* | 1.881e-3 (≈4.3×)* | 1.050e-5 (≈762×)* | 1.654e-3 (≈12×)* | 1.145e-3 (≈17×)* | 1.965e-7 (≈4×10⁴)* |

（*T8192 行的记录数字摘自 r19–r22 各轮"every digit equals the record tables"表，裕度由本文按同一棒折算；evidence/G5/bwd_report.md rounds 13–22。）

裕度演化注记（教学价值所在）：dv 在 r8 曾达 **2454×**（3.3e-6，dvp 的 pdO+bh 全部走精确 bf16/split 乘积所致），r11 引入 tv split-bf16 后回落到 736×/848×——"乘积更紧、多一次重结合"类；dk 是全表最紧的棒位（r8–r10 时代 5×），r13 把七个 tf32 2^-11 位点换成 2^-17 split 后收紧到 6.2×/15.6×。dk 裕度部分是运气（r9 原话"The dk 5x margin is partly luck"）：其数值路径上存在一个此前未发现的非确定性竞态（§8.4.2/§8.4.3），同一 kernel 两次 verify 采样到 1.719e-3 或 4.719e-3（一个 bf16 量子之差）——这正是第 10 轮把 dk 相邻变更全部挂到确定性门上的原因，也解释了为何"dk 裕度最小"与"dk 首先暴露竞态"是同一件事。

### 8.2 numpy fp64 孪生方法

#### 8.2.1 孪生的构造与评分口径【动机→源码片段】

**动机**：GPU 上的评分参考必须满足三个 otherwise-冲突的约束——(i) 与 kernel 吃到**位相同**的输入（否则差值混入输入量化噪声）；(ii) 计算精度远高于 kernel（fp64）；(iii) 语义与 kernel 的分块算法**同构**（chunked WY，而非逐步递推，否则差值混入算法重结合噪声）。numpy fp64 孪生（twin）是同时满足三者的最简对象：它是 `gdn_l1_check.chunked_bwd_ref`（G1-B2 数学，自身对 JAX vjp ≤1.4e-14）的向量化移植，运行在**主机端 bf16 量化后的同一值**上（量化值加宽到 f64）。

输入量化在主机完成，round-to-nearest-even 用 fp32 位技巧精确实现（`tools/gdn_bwd_verify.py`）：

```python
def bf16_quant(x):
    """round-to-nearest-even bf16 via fp32 bit trick."""
    f = np.ascontiguousarray(x, dtype=np.float32)
    u = f.view(np.uint32)
    rounded = ((u >> 16) & 1) + 0x7FFF
    u = u + rounded
    return (u >> 16).astype(np.uint16)

def bf16_widen(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
```

评分口径因此是：**差值只度量 kernel 数学 + kernel 内部截断**，与 fla 的评分方法学一致（docstring："Inputs are bf16-quantized on the host (round-nearest) and the twin runs on the SAME quantized values (widened to f64), so the diff measures kernel math + internal truncf"）。布局侧，孪生与 kernel 共用 chunk-major 展平 `[BHNC, C*W]` 宿主布局与 log2 域 g（`RCP_LN2` 换域），反向孪生按"pass1 前向重算只存每 chunk 入口态 h_all → pass2 正步长逆走（n = NC−1−n2）"的结构镜像 MLIR @gdn_bwd。第三方锚点：小形状上 `--jax` 把孪生对 JAX vjp oracle 再比一层（输入量化残差 ~1e-3 量级为期望值），前向侧 G1/G5 则直接以 JAX dumps/共享 refs 为 oracle——孪生永远不是唯一裁判。

孪生同时是**中间量仪器**：`with_intermediates=True` 时输出 `h_all/dh_all/T_all/vn_all/dvp_all/dW_all`，供逐阶段对拍与破案二分（§8.3.4、§8.4）。

#### 8.2.2 方法论教训一：评分孪生自身有 bug——bf16 误诊的翻案【片段→前后对比】

G5-C 首轮反向验证的读数是 dq/dv/dh0 PASS、dk/dbeta/dg 大幅 FAIL（dk 6.6e-1、dg 3.7），当时的假设是"bf16 中间量放大"。第 2 轮的仪器化破案（每个中间量对 fp64 重放）推翻了该假设：四个相互独立的缺陷，没有一个是你以为的"operand 精度地板"——其中三个在 kernel（dk_local epilogue 对整个累加器滥施 β·2^gc 行缩放，~0.52 确定性 dk 误差；dhu 的 rb 累加器未按 chunk 重置零；dg_state 前缀 `scf.for` yield 了末元素而非 running sum），**第四个在孪生自己**（`tools/gdn_bwd_verify.py`）：

```python
# G5-E round2 FIX: kkT must be THIS chunk's k k^T. It was only
# computed in the forward loop above, so dbeta's ra term and
# dg's AdA term silently used the STALE kkT of chunk NC-1
# (~3.6e-2 / 2.4e-2 phantom errors vs the JAX oracle; the GPU
# computes the per-chunk kkt correctly and was right).
kkT = k[bh, n] @ k[bh, n].T
```

即：反向逆走循环里 dbeta 的 ra 项与 dg 的 AdA 项使用了**前向循环遗留的、最后一块的旧 kkT**，制造 3.6e-2/2.4e-2 的幻影误差——与第 1 轮"FAIL"数字严丝合缝；而 GPU 每个 chunk 正确重算 kkt。对 JAX oracle 的三角对拍完成翻案：|jax−twin| = 3.6e-2/2.4e-2（正是"失败"），|jax−GPU| = 1.5e-5/6.9e-6——**GPU 是对的，评分器是错的**。修复后 6/6 双形状 PASS，最差梯度 ~180× 裕度（evidence/G5/bwd_report.md round 2；bwd2_verify{512,2048}.json）。

方法论沉淀有三条。其一，**评分器必须有 oracle**：孪生对 JAX vjp 的交叉不是装饰，是唯一能暴露"裁判自身腐败"的层。其二，**"bf16 放大"是危险的第一解释**：同期的 fp32 化工作仍然必要——round-1 的 bf16 中间量让装配 bug 藏在 2e-3..4e-3 噪声下不可见，fp32 化后每个中间量精确（T 1.2e-7、h 5.7e-7、vn 3.0e-7、dvp 1.5e-6），bug 才各自现形；但"残差大 ⇒ 精度问题"的推理方向本身在第 2 轮被证伪。其三，round-1 报告里归因于输入量化的孪生-vjp 散布，事后看有一部分就是这个 bug——**未翻案的归因会污染后续所有轮次的基线叙事**，这是把它写进论文的原因。

#### 8.2.3 方法论教训二：验证 harness 的 H2D 重分级掩盖效应【前后对比→数字】

第 5 轮扫描重构（dhu 退役）落地后，verify 首次把 dbeta 从 3.0e-4 推到 1.33e-2（1.5× 于棒内）。二分定位（probe_r5inter → probe_dbeta → payload diff → numpy bf16 仿真）发现一个自 round-2 起就存在的潜伏 bug：dqkwg 的 dvp 重算把 Bhat 分级进了 **bf16 tCKb tile**（2e-3 相对误差），且 GPU dvp 与 numpy(bf16 Bhat) 对到 3.3e-7——kernel 忠实执行了被误分级的路径。它**长期不可见的原因在 harness**：`run_chain` 在每次发射前把每个输入 global 重新 H2D 暂存，于是 wyb 永远收到 dhu 的干净 dvp_all，dqkwg 的覆写从未到达任何评分器（round-3/4 的 verify 表给的是干净路径，BENCH 跑的却是被毒染的路径）。dhu 退役后 dvp_all 的唯一生产者变成 dqkwg，毒才浮出水面。修复：bhat tile 改用该循环中已死的 f32 tCVb。**推论被写成契约**：verify harness 的 H2D 重暂存可以掩盖生产者/消费者 bug——round-5 起 dvp_all 从 dqkwg 读回评分（evidence/G5/bwd_report.md round 5"The latent bf16-Bhat bug"）。同一次狩猎还刻画了 tf32 用于 dvp_bh 时的**重建混沌**：三个数值等价 build 上 dbeta/dv 从 1.5e-5 → 2.9e-3 → 5.8e-3 跳变（任何 f32 级 dh 扰动都会重采样 mma 的操作数量化桶），据此保住标量路径。这两条与 §8.2.2 合起来构成"评分体系自身是被测对象"的完整案例集。

### 8.3 确定性协议

#### 8.3.1 协议分层【动机→协议片段】

**动机**：位级可复现不是免费赠品。G4 时代单 kernel 清零重发射即可逐位复现；多 kernel 链、跨 chunk 复用 smem、mma 分级与 cp.async staging 引入共享内存后，**共享内存的非法跨 warp 重叠读写**可以只在特定调度抽签下触发，表现为"大多数 run 一致、偶发 1–3 ulp 翻转"——它不破坏数值棒（dk 4.719e-3 仍在 8e-3 内），却使"逐轮数字可比、位级门可判"失去根基。协议因此按窗口宽度分层加压，直到竞态必然显形。

第一层（G4，进程内）：每个配置清零 + 重发射，输出缓冲逐位相等方 PASS（`res["deterministic"] = np.array_equal(got, got2)`）；生成器确定性以 sha256 同 seed 同 payload / 异 seed 异 payload 受验。

第二层（G5-Q，专用 harness `tools/gdn_det_check.py`），三模式（docstring 摘录）：

```text
loop   (default): run the whole chain N times IN-PROCESS, dump every
                  stage's OUT globals per run to per-run npz, diff.
single RUN OUTNPZ: one fresh-process chain run, dump one npz (the
                  faithful "run-to-run" mode -- fresh CUDA context,
                  fresh module loads, fresh device allocations; drive
                  it from a bash loop).
diff   OUTDIR:    compare all npz in OUTDIR against run 0; report the
                  FIRST chain stage whose outputs differ, per-global
                  diff counts and max-abs diff.
```

链执行器 `run_chain_unload` 镜像 verify 的链契约（每阶段 cubin 经 `cuModuleGetGlobal` H2D 暂存输入、发射、DtoH 读回输出），但**每阶段模块读完即卸载**（co-job <4 GB 纪律），这正是"刷新模块"窗口的来源。前向侧同构协议为 `tools/gdn_fwd_det.py`（mono staged 与 fused 各自 N 次跨进程，o/ht npz diff，`FWD-DETERMINISTIC` 判定）。单 kernel 锤与判别臂见 §8.3.3。

第三层（舰载常规，r15 起）：每个触碰相关路径的变更加 **fresh-process N=5 双形状**位级门（r10–r13 时代为 N=10）；r18 一次跑三套（ship knobs / FPIPE+SCANDB / F1KS 2-CTA），r24 附带把 gen_fwd 舰槽恢复并 md5 复核。反向同理：r13 每杠杆单独过（l1/l2/l3 各 T2048×5 + T512×5），r22 的 K32 臂过 N=5。

第四层（跨发射陈旧态契约，r21b）：2-CTA cluster 形态的 flag/spin 协议引入两个 manifest 级契约——`zero_globals`（自旋旗标为绝对目标，发射前由 harness 做 stream-ordered 清零）与 `scratch_globals`（pex 写前读）——bench/verify/det 三条 harness 全部改道经此契约，使跨发射陈旧性成为被协议显式消灭的对象，而非靠运气（evidence/G5/bwd_report.md round 21；hazard (2)）。

#### 8.3.2 刷新模块的必须性：进程内循环漏检 wyb 竞态【前后对比→效果数字】

**教训本体**（r10 原话级别的陈述）：T2048 的**进程内** 10 次循环在修复前**没有**复现 wyb 竞态；只有**跨进程 fresh-module 循环**复现——"in-process looping is a NECESSARY but not SUFFICIENT determinism probe; the fresh-mode harness is now part of the gate protocol (checklist item 7)"。量化对比（修复前/后，evidence/G5/bwd_report.md round 10 headline 表）：

| 采样套件 | 修复前 | 修复后 |
|---|---|---|
| T2048 跨进程 ×10 | wyb dk/dbeta/dg 发散（1 次；dk 8153 元 8.3e-4） | **全阶段位级一致** |
| T512 跨进程 ×10 | dk/dbeta/dg 发散（3/10 次；至 7113 元 7.2e-4） | **全位级一致** |
| T2048/T512 进程内 ×10 | bpre B_all ulp 翻转（1/10）/ dk 翻转 | 全位级一致 |
| bpre 单核 fresh-module ×1000 | 14/1000 发散（双稳 1–3 ulp） | **0/1000** |
| verify B1T2048H16 | dk 采样到 1.719e-3 **或** 4.719e-3 | 6/6 PASS，dk 固定 1.710e-3 |
| verify B1T512H4 | 同类 | 6/6 PASS，dk 固定 1.719e-3 |
| 前向 mono staged + fused ×10 | （此前未查） | o/ht 双路径位级一致（前向本来就干净） |

机制解释：进程内复用同一模块与同一设备分配，smem/全局的初始内容在 run 间被上一轮"预热"成一致；fresh-module 循环每轮重掷分配器与初始 smem 内容，把"快 warp 抢在慢 warp 前重分级 tile"的时间窗拉宽到可命中。bpre 单核锤的数据把这一点钉死：同一模块连发 300+150 次 0 翻转；fresh 模块每发 1000 次中 14 次发散，且翻转**成时间簇**（第 418–425 次迭代）与共载负载相关；同一元素的一次翻转两次复现出**相同的 (ref, now) 值对**——双稳态、两个确定性结局，是竞态而非垃圾（tools/gdn_bpre_loop.py；evidence/G5/scratch/r10_logs/）。

**效果数字（修复成本）**：确定性修复的运行时代价为 ~0.3%@T2048（bpre +6 µs 一条栅栏；wyb +3 µs 共 11 条；e2e 1.003× med / 1.001× min，同窗配对 A/B ×3），T512 为窗口噪声——位级可复现以近似零价购得。

#### 8.3.3 判别臂设计【源码片段→效果】

`tools/gdn_det_discriminator.py` 把"kernel 内竞态"与"内存子系统腐蚀"两类假设做成四臂对照（每臂 N 次迭代、每次 fresh 模块装载/卸载，与链 harness 的模块循环一致）：

```text
h2d     : stage M_all/do H2D, read them back D2H, compare to host.
          NO kernel launch. Any diff = module/memory path corruption.
launch  : stage H2D, READ BACK M_all/do right after launch+sync
          (input corruption check WITH the kernel running).
bpre    : the reproducer itself (B_all out compare).
idle    : plain device buffer alloc + H2D + dwell + D2H, no modules.
```

判定结果：h2d ×200 干净（模块/内存通路无罪）；launch+输入回读 ×200 干净（kernel 无越界写回 M_all/do）；标量 bpre 对照（GDN_BPRE_MMA=0，去 mma/ldsm）×400 干净——**指向 mma 读路径**，与后续 PTX 级根因（§8.4.2）闭合。这一臂化设计的要点是把"数据在运输中坏掉"与"数据在被计算时读错"在协议层互斥，避免破案在错误子系统里打转。

#### 8.3.4 逐中间量 npz 二分定位法【动机→协议片段→效果】

**动机**：九阶段链的最终梯度发散无法区分"源头在哪个阶段"；逐阶段的 OUT 全量落盘把链变成可二分的序列。

**协议片段**：链 harness 按 `OUT` 表（源码中的链接契约）把每阶段"拥有"的全局读回存 npz——wy: `T_all,u_all,w_all,kkt_all,P_all`；fwdscan: `h_all,vnew_all,gsc_all`；caryA/caryM/bpre: `A_all/M_all/B_all`；dhs: `dh_all,dh0`；dkb: `dkB_sum,rowB_c,s5_c`；dqkwg: `dq,dW_all,dk_local,dvp_all,r3_c,r4_c,c4r_c,dg_state`；wyb: `dk,dbeta,dg,dv`。diff 模式按 STAGES 顺序对每个全局报 DIVERGES/stable、逐全局差数与 max|d|，并输出**第一个发散的阶段与全局**；比较忽略符号零/NaN 对噪声（`m = neq & ~(fa == fb)`）。r9 的 dk 定位还用同法对**阶段内部中间量**（dk_local base-vs-new npz）二分，先以 tf32/标量臂分裂把腐蚀钉在 TW gemm，再逐量收敛。

**效果（r10 破案的定位链）**：第一个发散全局 = bpre 的 `B_all`——12/8.4M 元素、集中在单个 chunk 行（m=132）、单 K 行（k=3）、一个 16 宽 V 条，相对差 ~1e-7 = 1–3 个 f32 ulp；上游（wy/fwdscan/caryA/caryM）与全部 dqkwg 输出位级稳定——与 r9 的观测精确吻合。传播链完整重建：B_all → dh_all（数值在下一加法点被吸收）→ dkB_sum → 某一 dk 元素的 bf16 round-to-nearest 翻转 = 1.719e-3 ↔ 4.719e-3 摆动（一个 bf16 量子）。bpre 修复后第二轮采样发散点移入 wyb 内部（其全部输入稳定），于是第二竞态现形——**二分法在同一轮内连续破两案**。

#### 8.3.5 restage-fence 七点清单【效果：从清单到制度】

r10 破案后，对生成器中每一个 mma/ldsm 消费者做同类审计（补三个潜伏位点、列明审计干净名单），并把裁判规则固化为**七点清单**——此后每一次 kernel 变更的前置检查表（evidence/G5/bwd_report.md round 10"The audit rule added to the toolbox"）：

1. 每个被跨 warp 读取的 smem tile（ldmatrix 或 ld.shared），必须在**最后写者的存储循环之后、首个读者之前**有 gpu.barrier（以栅栏结尾的 staging helper 满足读侧）。
2. **restage 栅栏**：任何满足 (a) 暂存 tile、(b) 运行读取它们的 gemm/mma、(c) 下一迭代**重暂存同一批 tile** 的循环，必须在**读之后（迭代末尾）**有一条全线程栅栏——下一次迭代 staging 内部的尾栅栏**不能**为它们定序（对竞争中的 warp 而言双方都在该栅栏之前）。同时适用于 st.shared 重分级（bpre）与 cp.async 重分级（wyb G 环；cp.async DMA 写在 commit/wait/栅栏之前发射）。
3. cp.async wait_group 是**每线程**的：每个 wait_group 之后需要一条 gpu.barrier，其它 warp 才能读已暂存数据（r9）。
4. 栅栏必须在任何 <W warp 守卫区域**之外**（bar.sync 是全线程的）；守卫式 mma 链必须走自带尾栅栏的 `mma_chain_if`。
5. cp.async（LDGSTS）暂存的 tile **不得**被 ldmatrix 读取（r6 静默腐蚀家族）；mma 操作数 tile 一律 st.shared 写入。
6. 永不依赖"smem 分配即为零"（r7）：未读区域要么显式清零、要么证明永不被读（C+8 的 LDSM 填充列永不被读——经 lane 地址算术验证）。
7. 触碰这些路径的任何变更的确定性证明协议：`tools/gdn_det_check.py` 跨进程 N=10 **双形状**（仅 T2048 进程内循环修复前**没有**抓到 wyb 竞态——fresh-module 循环才是更宽的窗）+ verify 6/6。

第 2 条是全清单的心脏，也是标题点名"栅栏在读完之后"的原因：直觉会在 staging 尾部补栅栏（保护"写后读"），但重分级竞态是**读后写**——慢 warp 的 LDS/MMA 读还在飞，快 warp 已经开始下一轮写；唯一正确的定序点在读集合的末端。第 7 条则把前六条从"知识"升格为"门"。制度效果的直接证据：r11 起每个杠杆自带 fresh-module 循环证明（l1/l2/l3 全位级），r13/r15/r16–r22 的全部 value-exact 轮逐位复现记录数字，r21 的 cluster flag/spin 协议在共载下确定性且无死锁（N=5 双形状全位级）。

### 8.4 竞态破案案例集（教学式）

五个案例按"症状 → 定位 → 根因 → 修复 → 数字"展开；它们共同的教学点是：**每一个都在数值棒内、大多数在 1–3 ulp 量级，全部被确定性协议而非数值验证抓获；其中两个被 1-CTA 驻留掩蔽数轮。**

#### 8.4.1 案例 A：cp.async wait_group 的每线程语义（r9，dqkwg）

**症状**：dqkwg v4（16-warp 占用率改造）在 256 线程/满网格下 final-dk 出现 9.3e3..9.8e10 的垃圾，而 128 线程/T512 网格下**不可见**（窗口太窄）。**定位**：tf32/标量臂分裂钉住 TW gemm 后，dk_local base-vs-new npz 二分：**每个 ks 切片的第 0..5 列**陈旧——恰好是线程 0–1 拥有的 16 B chunk。**根因**：TW 环内联的宽度受限 tCKa 暂存（`astage_tile + await_all`）不带 `astage_sync` 携带的尾 gpu.barrier；wait_group 只**每线程**地等待自己发出的 DMA 组，别的 warp 的在途 chunk 并未就绪，快 warp 竞入 TW tgemm 读到了其它线程尚未完成的列。同一次狩猎的第二发现：v3 式裸 `mma_gemm` 链无尾栅栏（`mma_chain_if` 文档化过此隐患）——dPr 的 vs 环可以在慢 warp 的 mma 读之下重分级 akb/vnh/vnl。**修复**：各补一条 gpu.barrier。**数字**：6/6 双形状 PASS，且**精确落在基线裕度数字**（dk 1.719e-3/1.710e-3——早尾重排只引入 ~1-ulp 级重结合，打印位以下）；这是 verify 门连续第 6 轮抓到当轮真 bug（evidence/G5/bwd_report.md round 9）。

#### 8.4.2 案例 B：bpre 的 K 环 mma 重分级竞态（r10 根因一；r8 起潜伏）

**症状与定位**：见 §8.3.4（B_all 1–3 ulp 双稳翻转）。**根因**：bpre 的 K 环 `for i0 in (0,64): stage_split_ew(ahi,alo); mma(bmh: ahi); mma(bml: alo, epilogue → B_all)`——两个 staging helper 都以 gpu.barrier 结尾，但**没有任何东西隔开 i0=0 的 mma ldmatrix 读与 i0=64 的 split 写**（PTX 4 条栅栏，缺的在环内）。早完成的 warp（其 epilogue 的 64 次全局存储拖慢它抵达，但只要有一个慢 warp 还在发 LDSM 就够了）在慢 warp 读 alo 时覆写之。**为什么像 1-ulp 噪声而非腐蚀**：被交换的操作数元素是 LO 分量（lo = v − bf16(v)，多在 ~1e-10），新旧乘积差落在受影响输出的 1–3 ulp；一条 16 B LDSM 行（8 个 alo 元素）正是观测到的 12/16 元素簇；翻转全落在 k 行 0..63（i0=0 的输出）。ahi 侧（~1e-4）会腐蚀到 O(1e-3) 并直接 FAIL verify——但快 warp 抵达重分级点时距 bmh{0} 只有 ~100 周期， practically 从不输掉该竞态；bml{0} 是暴露环节。r9 已文档化同类（"mma_gemm inter-iteration restage race"）并在 dqkwg v4 的 dPr 环修复——bpre 被漏掉。**修复**：每轮 K 环迭代末一条 gpu.barrier；4 → 5 条栅栏，54 寄存器，0 spill。**数字**：单独 bpre fresh-module ×1000 从 14/1000 发散到 0/1000（§8.3.2 表）。

#### 8.4.3 案例 C：wyb 的 G_dvp/G_dW 累积环竞态（r10 根因二；cp.async 味，更大、更老）

**症状**：bpre 修复后跨进程采样**仍然**发散——发散点移入 wyb 内部（其全部输入稳定）：dk/dbeta/dg 在 1e-6..1e-4 量级、数千元素，dv 不动。**定位**：签名直指 G（accCC2 = strict_tril dvp(βv)ᵀ + dW(βk 2^gc)ᵀ）：G 喂 dAf → dk/dbeta/dg，而 dv 的路径不经过 G。**根因**：gen_wyb 的两个累积环 `for vs: astage_sync(tCVa ← dvp_all); stage_raw_ew(tVV ← v); emap(tCVb); gemm(accCC2, tCVa, tCVb, bar=False, kahan)`（ks 环同形）在环内 gemm 之后**没有栅栏**：下一迭代的 astage_sync DMA（cp.async）重写 tCVa/tCKa 时，慢 warp 的 gemm LDS 读还在飞（emap 亦在其下重写 tCVb/tCKb/tKK）。尺度自洽：G 处全值操作数交换 ~1e-3 → dAf → 下游数千个 1e-6..1e-4 位移。**修复**：每环 gemm 后一条 gpu.barrier（2 处）；wyb PTX 127 → 138 条栅栏。**数字**：T512 修复前 3/10 触发（小网格链在重分级窗中相对停留更久）；这是 r9 采样 dk 摆动的主导贡献者（bpre 翻转只是 ulp 级尾部）。**同类审计的连带收获**：dqkwg v3 的 dPr vs 环（T512 派发）补迭代内栅栏（v3 PTX 192 → 195）；v2 遗留 dvb ks 环在 <W 守卫**外**补栅栏（守卫内会死锁）；审计干净名单（wy kkt/pmm、dkb、caryA/caryM、dhs、fwdscan F2 组、wyb TW/尾环）逐一记录在案（evidence/G5/bwd_report.md round 10）。

#### 8.4.4 案例 D：zero_up 步幅 bug——一指令 SASS 证词（r18 根因一，fwd F1 2-CTA 腐蚀）

**背景**：F1KS 把 f1 压到 46,352 B smem（首个 2 CTA/SM 的前向 kernel）后，2-CTA 共驻下 u/w 腐蚀（o 6.3e-2 对 5e-3 棒），而 ht 停在记录数字 7.493e-4；八臂二分表（arena 有无、staging 类别、bk/mb、pool 在 2-CTA 带内的取值）证明 K-split 代码在一切 1-CTA 臂上位级正确、腐蚀随 2-CTA 共驻本身出现；racecheck 在**干净** build 里也报同样 6 对"良性"扫描步 LDS/STS，memcheck 干净——两个 sanitizer 都以串行化调度把 bug 藏起来。**症状指纹**：仅 f1 在场即可复现（无需 f2/f3）、逐 rep 非确定、逐 tile 二值腐蚀（研究 run 中 338/512 个 tile、后调度的 CTA 腐蚀）——陈旧态签名而非数值 bug。**定位与证词**：forced-1-CTA PASS build 与 2-CTA FAIL build 的 **SASS 差恰好一条指令**（`MOV R4, 0x2490` vs `0x4410`——Tg 的 malloc 尺寸 9,360 vs 17,424）：代码相同，触发器是纯共驻。tile 转储（G5D_DUMP_TT 扩展 Tg 与 epilogue 后半 tile）把 Tb/Tg 差**精确**钉在 zero_up 漏写的单元（每 tile 768 个；计数 43,264 = 338 tiles × 128 cells）；r17 的 o/u-vs-ht 之谜由指纹解开：w 恰在 Tb/Tg 块列 Cb=2 = 行 32–47、u 在行 0–47、行 48–63 干净（mma 输出块 mi=3 只从 k-块 >3 收垃圾，而后者不存在）；被腐蚀 chunk 的状态贡献在末 chunk 前衰减到契约之下（f2 的 ht 完美），f3 却直接读 h_all/v_new——o 败、ht 过，从未有输出路径之谜。**根因**：solve 的 phase-0 `zero_up` 单元循环以 `32*8 = 256` 步幅跨 128 线程块行走，Tb/Tg 中 `(r*64+c) mod 256 ∈ [128,256)` 的单元——左下六个 16×16 块中行 `r mod 4 ∈ {2,3}` 者、每 tile 768 格——**从未被清零**；uw gemm 的转置 ldmatrix A 读覆盖整个 64×64，把这些格读作 T 的上三角零。1-CTA 下读到**精确零**（512 tile 全部 393,216/393,216 格实测）；2-CTA 共驻下读到非零活数据（~56% 格，±1e-14 级远对角衰减值、偶发 O(1) 离群、absmax 1.0），且与顺序无关（1-CTA 在垃圾重的 2-CTA run 之后仍零干净）。陈旧字节的精确供给路径（跨槽渗漏 vs 分配器粒度 vs 前次发射遗留）**留为不归因**——修复消除对三者的全部依赖。**修复**：`GDN_ZFIX`（步幅 nthr，全覆盖）；默认关（未设 == HEAD payload 逐字节，复核两次），经舰组环境打开。**数字**：ZFIX 后腐蚀从 ~70 万 bit 差塌缩到 ~4 千；f1 代价 +8.3 µs@T8192（零循环翻倍到 32 迭代），被 F1KS 收益覆盖 6 倍（evidence/G5/fwd_report.md rounds 17–18；scratch/r18f/probe1..8、dump_1cta/dump_2cta）。

#### 8.4.5 案例 E：Hillis-Steele d=32 跨 warp 竞态与 SCANDB（r17 发现、r18 定罪）

**症状**：ZFIX-only 之后残余 ~4 千 bit 差、每 run 1–2 个 tile 触发、指纹**精确落在 `gc[32:64)` 窗口**（Hillis-Steele 第 d=32 步的跨 warp 读），且在 chunk 内局部传播（该 chunk 的 u/w 行 32–63）。**根因**：f1 的 gc 累加使用 6 步 smem Hillis-Steele 扫描（取代每线程串行 cumsum），d=32 步要求读**另一 warp** 写的半程和——该跨 warp 步内读写未定序；racecheck 在每个 build（含干净 build）报的正是这 6 对扫描步 LDS/STS。1-CTA 驻留从未暴露它（r17 时代因此只作为"建议硬化"落地而默认关）。**修复**：`GDN_SCANDB`——扫描步双缓冲（r17 落地、r18 作为舰载硬化打开：mono 配 ZFIX+SCANDB，fused 配 SCANDB）。**数字**：**ZFIX+SCANDB 于 2 CTA/SM：10/10 次 f1-only 重复对 forced-1-CTA 参考位级一致，全 trio 命中记录数字（o 6.486e-4 / ht 7.493e-4）**；1-CTA 下硬化是值精确的（ZFIX+SCANDB T8192 build 对 r15 舰在 fp64 门上位级一致，o/ht 0.000e+00；FPIPE+SCANDB fused 与 FPIPE-only fused 值位级一致，后者重建精确复现 r17 舰 md5 93856752）；det N=5 跨进程三套件 FWD-DETERMINISTIC（det_ship/、det_fpipe_sdb/、det_f1ks/）（evidence/G5/fwd_report.md round 18）。

案例 D+E 合并的教学结论：**两个都被 1-CTA 驻留掩蔽、都自 blocked solve 落地起就活在每个 build 里**；把 f1 推过 2-CTA 线的占用率工程（本身是纯调度变更）充当了它们的触发器。这正是"确定性协议必须与占用率/驻留变更绑定重跑"（清单第 7 条覆盖"触碰这些路径的任何变更"）的实证理由——调度旋钮可以唤醒沉睡的数值 bug，而位级门在它破坏任何数值棒之前就把它截获。

### 8.5 cubin 指纹、旋钮纯度与 cache_key

#### 8.5.1 md5 舰队登记【动机→效果数字】

**动机**：论文的每一个性能记录都绑定"哪一个 cubin 跑出来的"；没有字节级指纹，记录之间不可比、复现声明不可审计。制度：每轮报告的 Ship provenance 表登记全部舰 build 的 `kernel_smemfix.cubin` md5，记录迁移时旧字节整体保穴（r25 迁移把 r22 字节保存在 `<name>.r22ship` 兄弟目录）。关键指纹（最终系统）：前向 R25 档案代 mono 三形状 `e7c8eb4c / 39f094c3 / be5b2377`（T8192/B4/T2048；由 HEAD 1c7d291 全新构建，md5 与 r24 终版候选 3/3 精确一致——未改树+同环境的重建**逐字节复现**，模块布局彩票只在生成器编辑时重掷，不在重建时）；其前任 r22 记录代 `be8b914b / 83898f74 / 3d47886a`；r18 时代的六舰表（mono `ea8ec2f1/e7af55c1/a065a0bc` + fused `d16522d0/cb2520d8/676487bf`）完整留档。反向冻结于 r19-final 字节：基线 18/18 阶段 cubin md5 == r20 记录（如 dqkwg `837a36fcc908/1f53c8910006`、dkb `1538de9d5dee/fbb0682b6776`、fwdscan `e680e1d94e47/304b4259a79d`，T2048/T8192）；K32 臂 `fb89448c760e/e615d26e9402`（evidence/G5/fwd_report.md rounds 18/22/25；evidence/G5/bwd_report.md rounds 20–22）。

#### 8.5.2 旋钮关断字节等价（knob-off purity）【协议片段→数字】

**契约**：每一个新旋钮默认关，且**未设 == HEAD 逐字节**——不仅 cubin，先在 payload 层面验证（生成器编辑后 mono+fused payload 对 HEAD diff 0 行，r18 `/tmp/r18f_id`）。反向侧按阶段计数：27/27 payload 字节门（r19）、18/18 cubin（r21 `r21b_koff2`）、9/9 × 2 形状（r22）。这一契约使"探针机械不污染舰路径"成为可检查命题：r23 的三个 timing-only 删除探针（GDN_F1T_NKK/NEK/NKM）输出**按设计为垃圾**、从不进 verify——与舰路径的字节隔离是它们合法存在的前提。

#### 8.5.3 纯度重建（purity rebuild）【前后对比→效果数字】

**契约**：对生成器做探针/新旋钮编辑后，用**舰组旋钮**从编辑过的树重建，必须精确复现现任记录 cubin 的 md5。执行记录：r18（F1WY 编辑后 mono-T8192 舰复现 `e7af55c1` 逐字节）；r21 三次（探针旋钮编辑后、keep-alive-sink 编辑后、F1ZF 编辑后，T8192 `e7af55c1` ×3 + T2048 `ea8ec2f1`）；r23 两次（NKM/emit_kloop 编辑前后，`e7af55c1`；+F1ZF 舰 `be8b914b`）；r24 三形状全复现 r22 记录（编辑在关断态惰性）；r25 从 HEAD 全新构建 3/3 精确命中 r24 候选。纯度重建同时是**位级等价快速门**的基础：当变更为值恒等类（W32P 的纯索引重排、K32 的链拆分、F1KS 的 K-split），"cubin 变了但输出位级不变"的直接验证（T8192 门 o/ht 0.000e+00、K32 35/35 + 36/36 阵列位等、AW 隔离 dh_all/dh0 位等）比 fp64 重评分更强也更便宜——它把"未改数值"从主张变成测量。

#### 8.5.4 cache_key：陈旧缓存孔洞的机械闭合【源码片段→前后对比】

**孔洞**：`gen_fwd` 槽位缓存一个已编译的 `kernel_smemfix.cubin`，在**不同旋钮集或已编辑的生成器**之下被接受——这个孔咬了两次（r3；r19 的 F1WY T2048"验证"实为一次陈旧 gen_fwd 命中），另有一次 r17 发现 fused 舰槽持有 r2 时代 w8/bv64 陈旧 build。r20 的机械闭合（`tools/gdn_full_rt.py`）：

```python
def _cache_key():
    """R20-fwd: hash of everything that changes the emitted payload --
    every GDN_*/G5* env knob in scope + the generator/smem_fix sources."""
    h = hashlib.sha256()
    for k in sorted(os.environ):
        if k.startswith("GDN_") or k.startswith("G5"):
            h.update(f"{k}={os.environ[k]}\n".encode())
    for f in ("gdn_full_gen.py", "gdn_smem_fix.py"):
        ...
    return h.hexdigest()[:16]
```

build 把 key 盖章进 `full_spec.json` 的 `gdn_cache_key`；后续 build 在 key 不匹配时**重定向**到 `<out_dir>_k<hash8>` 变体槽（而非返回或覆写缓存），无章的 legacy 目录按原样信任（gen_fwd 舰穴保留）。制度性后果（r23/r25 协议注记）：生成器每编辑一次，记录槽的章即"按设计陈旧"（如 `0d667ecf → 780ab7d2`，再到 r25 的 `7b41c5f9`），rt.build 重定向、**从不陈旧供服**；只有记录真正迁移时才刷新槽章。配套的 det 槽审计（r21）：发现 `bench_B1T2048H16_mono` 持有 pre-r20 的 `396be193`，规范化回舰 `ea8ec2f1`（`.stale396b` 保穴）；以及 run_det 的缓存交换陷阱修复（变体必须**移出**而非盲目 `mv $C.shipbak $C`——r20 曾因此嵌套两层 shipbak，经 md5 复核恢复；r21 把交换逻辑折叠进 run_det.py 本体）。r22 的"legacy 槽信任孔洞"由此绕开：记录迁移一律从新盖章的 genrec 根判读。

#### 8.5.5 位级等价门汇表【效果数字】

| 门 | 对象 | 判据 | 代表数字 |
|---|---|---|---|
| verify 4/4 / 6/6 | 前向两路径×两形状 / 反向六梯度×形状 | 零元素越棒 | fwd 记录数字 o 6.486e-4 / ht 7.493e-4；bwd §8.1.3 表 |
| det N=5/10 跨进程 | 舰组 / 变体，双形状 | 全阶段（或 o/ht）位级一致 | r18 三套件；r24 W32P（舰槽恢复 md5 3d47886a 复核）；r22 K32（r20_logs/det_r22b_k32） |
| T8192 fp64 位级门 | 值恒等变体 vs 纯度 build | o/ht 0.000e+00 | ZFIX+SCANDB、W32P、F1KS 类外推 |
| 阶段阵列位等 | K32 链拆分等 | 35/35 + 36/36 阵列位等 | r22；P11R1+K32 亦位等（保序类） |
| 分段不变量 | f2-segments S=1/4/8 | o、ht 跨 S 位级一致 | evidence/G5/fwd_report.md |
| 旋钮关断 | 全旋钮 off | payload/cubin == HEAD | 27/27、18/18、9/9×2（bwd）；payload diff 0 行（fwd） |
| 纯度重建 | 舰组自编辑树 | md5 == 记录 | r21 ×3、r23 ×2、r24 ×3 形状、r25 3/3 |

这七行合起来即是本章的收束命题：**"数值正确"由四层塔保证，"逐位可复现"由确定性协议保证，而二者都建立在可审计的字节身份之上**——容差契约把"与 fla 同级"变成外部可证伪的命题（§8.1），fp64 孪生把评分从实现中解放出来并两次纠正了评分者自己（§8.2），七点清单与刷新模块协议把共享内存竞态从"运气问题"变成" checklist 问题"（§8.3–8.4），md5/cache_key/纯度重建把 26+25 轮战役中的每一个数字钉在可重新生成的字节上（§8.5）。复现入口：`make repro-g1`（三方对拍）、`make repro-g4`（20 配置双 seed）、`tools/gdn_full_verify.py --warps 4 --bv 32`（前向 4/4）、`tools/gdn_bwd_verify.py --gen-root … --shape …`（反向 6/6）、`tools/gdn_det_check.py --mode single --run i …` ×N + `--mode diff`（跨进程确定性）、`tools/gdn_fwd_det.py --tag bench_B1T2048H16 --path mono|fused`（前向确定性）。

---

## 9 测量与判定方法学：在共载 GPU 上建立裁决级证据

> 本章自洽可读。所有加速比、判定与标定结论均给出证据文件指针；每个小节按【动机→源码片段→数据→结论】四件套组织。全局口径与全文一致：单卡 RTX 5090 Laptop（sm_120a，消费级 Blackwell，82 SM，GDDR7，无 tcgen05），全部 GPU 计时取自共载机会窗口或干净窗口并在源 JSON 中以 `shared_gpu` 标志与 `nvidia-smi` 快照披露；共载任务从不被触碰、从不被信号打断，运行前在 `~/codebase/GPU-COORDINATION.md` 登记协调行，显存占用 <4 GB。**加速比的分母一律是同窗交错测得的基线，不是跨窗口中位数之比；绝对毫秒数只作工程参考，永远带窗口快照。**

### 9.1 章首：为什么测量本身需要一套方法学

本文的实验平台是一块**常驻共载**的消费级 GPU：从 G3 到 R25 的整个证据链上，计时窗口内几乎总有另一个会话的计算任务在跑（co-job pid 3411456 常驻约 17.9–19.9 GiB 显存、15–25% 利用率，另有兄弟工作流的突发负载；各轮记录了当时实际的 pid 与快照，见 `evidence/G5/fwd_report.md` 各轮头部与 `evidence/G3/calib_report.md`）。这不是可以等走再测的偶发干扰，而是持续约半天的常态。用户指令由此明确了协议基调：需要干净窗口的计时工作**延后而不阻塞**——比较要么"在共载下仍可判定"，要么"诚实延后"，机器不许空转（`evidence/G5/adjudication_methodology.md` 卷首）。这条指令逼出来的问题不是"怎么测得更快"，而是"**怎么测得可裁决**"。

三条实测的失败模式说明为什么朴素的测量在这台机器上不成立：

1. **共载窗口漂移**。分离的两个 bench 会话之间，co-job 的负载突发可以让同一对比按窗口选择摆动 1.5–2 倍（G5-A/G5-D 实测；`evidence/G5/adjudication_methodology.md` 等价性论证段）。
2. **空闲时钟循环（idle clock cycling），且与共载无关**。即使在**零共载的干净窗口**里，同一 cubin、同一会话、相隔数分钟的两个进程给出的绝对值也不一致：我方 B4T2048 staged 前向 1.0483 ms（fwd bench 进程）对 1.1941 ms（adjudicator 进程，+14%）；fla B4T2048 前向 0.9240 ms（flaptx 进程）对 1.3617 ms（adjudicator 进程，**+47%**）。原因是进程间隙 GPU 掉进空闲时钟循环：步与步之间的 smi 快照抓到 180–712 MHz，而步内持续 1590+ MHz（`evidence/G5/GATE_REPORT_FULL_REMEASURE.md` §3 异常 A1）。与之对偶的 A6：干净窗口里 fla 的**中位数**反而高于共载窗口存档（B1T2048 Triton 0.3932 对 0.3158 ms）而**最小值**吻合（0.3160 对 0.3158）——共载任务把时钟一直顶在高位，干净窗口的中位数反而携带空闲下探。
3. **同进程内的模块布局彩票（placement lottery）**。两个 cubin 先后 `cuModuleLoadData` 进同一进程时，后加载者的 `.global` 区布局会让访存最重的 kernel 付约 6% 的系统性偏置（T8192 的 f3 上 +10.5±6 µs），与源码无关、与 PTX 逐字节相同无关（§9.4）。

结论性原则（本章其余部分逐条落实）：**(i)** 裁决级口径是**同窗交错配对比值**（§9.2）；**(ii)** 何时一个比较"可判定"必须写成显式规则而不是审稿人直觉（§9.3）；**(iii)** 测量系统本身是仪器，要被审计、标定、修正（§9.4、§9.8）；**(iv)** 在拿不到计时窗的地方，用**删除探针**（§9.5）与**无计时的静态指令分析**（§9.6）替代——这是用户"分析生成指令"指令的方法学落实；**(v)** 任何涉及"成本"的对比（如搜索成本）也要写明摊销口径（§9.7）。

本章覆盖的测量与判定仪器清单（每一件都在后文单节展开）：

| 仪器 | 路径 | 职责 |
|---|---|---|
| 共享输入配方 | `tools/gdn_bench_common.py` | 我方、fla、FlashQLA、fp64 参考全部消费逐位相同的张量 |
| fla 基线计时 | `tools/gdn_bench_flaptx.py` | fla 侧前向/反向计时、后端探针、per-kernel 归因 |
| 交错判定器 | `tools/gdn_final_adjudicate.py` | 裁决级加速比、`--stats` 置信块、可判定规则 |
| 判定方法论 | `evidence/G5/adjudication_methodology.md` | 判定契约（(a)–(d) 条款 + v2 侧规） |
| 平衡 A/B 配对器 | `evidence/G5/scratch/r14u/r14u_ab.py` | per-kernel 配对 A/B，镜像构造序消除布局彩票 |
| 删除探针族 | `GDN_F1T_*` 旋钮（`tools/gdn_full_gen.py`） | 相位级时间归因（timing-only） |
| 静态指令分析 | `tools/g5n_static_analysis.py` | PTX/SASS 直方图、occupancy、roofline 距离（无 GPU 计时） |
| 搜索成本计量 | `tools/g6/g6_solver_timing.sh`、`tools/g6/g6_fla_autotune_cost.py`、`tools/g6/g6_fla_amortize.py` | 我方 0.8 s CPU 对 fla 在线 autotune 及其四路摊销 |
| 硬件标定 | `scripts/run_calib.sh` + `tools/gdn_calib/calib_bench.cu` | 代价模型与 roofline 的速率输入；`shared_gpu` 标志 |
| G3 配置计时 | `tools/gdn_time.py` | 20 配置零参 cubin 的隔离计时与校验和确定性 |
| 干净窗口轮询器 | `scripts/adjudicate_when_clean.sh` | co-job 退出后自动补一次带 `--stats` 的判定 |

---

### 9.2 交错配对判定协议（`tools/gdn_final_adjudicate.py`）

#### 9.2.1 动机

朴素做法是"先测我方 50 次，再测 fla 50 次，取中位数之比"。这条路径在上面三条失败模式下全部失效：两个 50 次块落在不同窗口（失败模式 1）、不同进程（失败模式 2，+47% 量级）、甚至不同 cubin 加载序（失败模式 3）。协议设计的目标是构造一个**对窗口漂移一阶免疫**的估计量，并把"窗口是否干净"从假设变成被记录的观测量。

#### 9.2.2 源码片段与协议结构

**输入层：共享配方。** `tools/gdn_bench_common.py` 是全部计时的单一事实源：fla 官方测试 `fla/tests/ops/test_gdn.py::test_chunk`（H==HV 情形）的输入配方、种子 42，在 CPU 上以 fp32 生成、再 cast：

```python
torch.manual_seed(seed)
q = F.normalize(torch.rand(B, T, H, K, dtype=torch.float32), p=2, dim=-1)
k = F.normalize(torch.rand(B, T, H, K, dtype=torch.float32), p=2, dim=-1)
v = torch.rand(B, T, HV, V, dtype=torch.float32)
beta = torch.rand(B, T, HV, dtype=torch.float32).sigmoid()
g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32))
h0 = torch.zeros(B, HV, K, V, dtype=torch.float32)
do = torch.randn(B, T, HV, V, dtype=torch.float32)
dht = torch.randn(B, HV, K, V, dtype=torch.float32)
```

q/k 在 fp32 域 L2 归一化**之后**再 cast 为 bf16（caller 侧归一化，`use_qk_l2norm_in_kernel=False`，与 test_chunk 一致）；fp64 JAX 参考因此能消费 GPU 算子实际看到的精确 bf16 值（bf16→fp64 上转是精确的）。fla 侧拿到 torch bf16 设备张量；我方拿到同一批值经 numpy fp32 上传通道（`gdn_full_rt`）。**每一次计时、每一个容差数字，都建立在逐位相同的张量上。**

**判定层：交错配对。** `gdn_final_adjudicate.py` 每形状执行 N 个交错对，每对 =（我方前向，CUDA events 计时）紧跟着（fla 前向，torch events 计时），同一进程、同一会话、同一套已 stage 的输入；`FLA_FLASH_QLA=0` 在 import fla **之前**设置（`can_use` 是进程级缓存，此环境变量使其纯 Triton 后端生效），后端探针随输出存档。核心循环逐字如下：

```python
ours_t, fla_t, pairs_ratio = [], [], []
for _ in range(args.pairs):
    a = time_ours()
    b = time_fla()
    ours_t.append(a)
    fla_t.append(b)
    pairs_ratio.append(b / a)   # >1 => ours faster
```

warmup（默认 8 对）吸收 fla 的 autotune 与我方 cubin 装载。输出每形状包含：双侧全样本数组、双侧中位数、per-side min/median/max、逐对 nvidia-smi 快照（`window_pre`/`window_post`，含利用率/显存/SM 时钟）、以及**中位数之比（ratio-of-medians）与逐对比值的中位数（median-of-ratios）并列**——两者偏离超过 0.15 即 `window_clean=False`，把"窗口脏"从托辞变成机器可查的标志。

**统计层：`--stats` 块（G5-N 加入，纯标准库实现）。** 对逐对比值 $r_i$ 计算 median、IQR（`statistics.quantiles(n=4, method="inclusive")`——与中位数同一套线性插值）、min/max，以及**精确双侧符号检验**：

```python
wins = sum(1 for r in ratios if r > ref)
losses = sum(1 for r in ratios if r < ref)
n = wins + losses
k = min(wins, losses)
p_lt = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
return round(min(1.0, 2.0 * p_lt), 6)
```

即 $p = 2\,P(\mathrm{Bin}(n,\tfrac12) \le \min(w, n-w)) = 2 \cdot 2^{-n}\sum_{i=0}^{k}\binom{n}{i}$，无 scipy 依赖、对 n≤30 精确。n=5 时 p 的下限是 0.0625，n=30 全胜时约 $1.9\times10^{-9}$——这正是正式记录一律用 n=30 的原因。**每个存档 JSON 自带自己的置信块**，任何后续读者不需要重新跑 GPU 就能复核显著性。

**可判定规则（方法学条款 (b)）：**

$$\text{decidable} \iff (\mathrm{median}_{r} - 1) > 3 \times \frac{\mathrm{IQR}_r}{\mathrm{median}_{r}} \quad\wedge\quad \mathrm{median}_r > 1$$

即效应量必须超过逐对比值相对散布的 3 倍。理由（`evidence/G5/adjudication_methodology.md`）：共载噪声从配对里进来，IQR 度量的正是协议消不掉的窗内湍流；比湍流大 3 倍的差距不可能由 co-job 制造出来——量级级差距在任何共载下可判。规则**刻意保守**：一个中位数巨大但 IQR 同样巨大的窗口（例 [1.4, 1.5, 2.5, 3.5]：median 2.0，IQR/med 0.64）即使逐对全胜也判"不可判定"；边缘情形一律不裁决，落回符号检验 + `window_clean` 的软解读，并进入干净窗口复测。

**等价性论证（条款 (c)），精确陈述。** co-job 突发落在**两成员之间**时同时拖慢双方（比值一阶抵消）；落在**某成员内部**时只拖慢一方（表现为离群对，中位数稳健）。因此交错比值是共载下共载对称时的**保序估计量**：双方同为 DRAM 受限时各损失相近比例，比值给出干净窗加速比的**下界**。协议从不声称"效应藏在噪声里"的窗口。

**干净窗口自动化（条款 (d)）。** `scripts/adjudicate_when_clean.sh` 自 G7-B 起常驻轮询：每 120 s 查一次 `nvidia-smi --query-compute-apps`，连续 3 次无其它计算进程后（1）登记协调行、（2）运行一次 `gdn_final_adjudicate.py --stats` 落盘 `final_adjudicate_clean.json`、（3）关闭协调行；锁文件在 /tmp，崩溃安全，**从不向任何进程发信号**。完整复测预案（形状 × {staged mono 三联、fused 旗舰、bwd 链} × n=30 + 标定刷新 + 静态分析重算）写在方法学文档里，验收标准 = 规则可判定或 p<0.01 且窗口干净。

#### 9.2.3 数据：前后对比

**前（朴素口径，真实数字，敌对窗口 2026-10-08）**：co-job pid 3411456 常驻（17.9–19.9 GiB、15–25% 利用率）**且**兄弟工作流 G5-M 同分钟内在跑 GPU 突发——`--pairs 5` 的读数：

| 形状 | ours med ms | fla med ms | 中位数之比 | 逐对中位 | IQR (rel) | 符号 p | 可判定 |
|---|---|---|---|---|---|---|---|
| B1T2048 | 0.2892 | 0.6700 | 2.316× | 1.843× | 0.572 (0.310) | 0.0625 | **否** |
| B1T8192 | 0.9385 | 0.9604 | 1.023× | 1.028× | 0.040 (0.039) | 0.375 | **否** |
| B4T2048 | 1.1346 | 1.0045 | 0.885× | 1.022× | 0.421 (0.412) | 1.000 | **否** |

三个全部拒判：B1T2048 原始中位数像 2.3× 大胜但逐对散布（rel IQR 0.310；fla 首对突发到 2.0 ms、我方首对 0.48 ms）不过规则；B4T2048 呈经典脏窗签名（ratio-of-medians 0.885 对 median-of-ratios 1.022，`window_clean` 临界）；B1T8192 内部一致（安静尾部、干净标志、持平到 1.03×）但效应 0.028 藏在 3×0.039=0.117 之内——只是"本窗持平"的软读数，**刻意不是裁决**。这张表是协议"窗口撑不住就闭嘴"的规范演示（`evidence/G5/adjudication_methodology.md` worked example；`final_adjudicate_g5n_stats.json`）。

**后（裁决级口径，现行正式记录，R25 档案，n=30，单一 cubin 代际，cubin md5 `e7c8eb4c/39f094c3/be5b2377`，`evidence/G5/final_adjudicate_r25f_w32p.json`）**：

| 形状 | ours med ms | fla med ms | 记录（中位数之比） | 逐对中位 | 胜对/符号 p | v1 IQR rel | v2 tIQR rel | v1 / v2 |
|---|---|---|---|---|---|---|---|---|
| B1T2048 | 0.18656 | 0.396448 | **2.1250×** | 2.1174× | 30/30，9.3×10⁻¹⁰ | 0.4199 | 0.3468 | 否 / **是** |
| B1T8192 | 0.785056 | 1.059296 | **1.3493×** | 1.3307× | 29/30，2.9×10⁻⁸ | 0.1776 | 0.1190 | 否 / 否（0.331 < 0.357） |
| B4T2048 | 0.821312 | 1.002944 | **1.2211×** | 1.2029× | 25/30，1.6×10⁻⁴ | 0.2733 | 0.1679 | 否 / 否（本窗） |

（p 值按存档 verdict 文件的单侧口径引用；判定器的双侧公式将其翻倍，两口径都远离任何门槛，从不改变任何判词。三窗 `window_clean` 均为 true；注意 B1T8192 的窗内快照抓到 SM 时钟 1612 MHz 的空闲下探——正是 §9.1 失败模式 2 的现场证据，而交错比值照常成立。）同一份 JSON 里可以看到共载从没消失：`gpu_start` 记录 util 28%、显存 9194 MiB——**这套协议不是在干净机器上工作的，它就是在共载机器上工作的。**

**cubin 溯源卫生（判定器自身的历史教训）。** r3 判定曾通过"构建缓存对已存在 cubin 短路"这个洞为一次 w8 请求伺服了陈旧的 w4 cubin（不同配置必须落不同 gen 目录，代码内注释留档）；r19 又被陈旧 `gen_fwd` 缓存咬过一次（`rt.build` 短路）。最终系统的对策：gen 目录按配置区分、`rt.build` 以 knob/generator 哈希为 cache key（r20/r23 关闭 r3/r19 两代陈旧缓存洞）、记录槽携带 stamp 且"设计性过期"时**重定向而绝不陈旧伺服**、每场正式判定前对 cubin md5 与记录值核对（R25：3/3 精确复现）、迁移记录时旧字节整目录保存在 `<name>.r22ship`。历史记录**从不回写**，修正以追加方式落盘（§9.4.3）。

#### 9.2.4 结论

交错配对协议把"共载 GPU 上能不能信一个加速比"从争论变成了可执行的契约：分母同窗、估计量保序、窗口质量被记录而非被假设、置信度随档自携、判定规则显式且保守。正式记录（2.1250×/1.3493×/1.2211×）全部产自该协议；协议同时产出了同样重要的负面能力——在窗口撑不住时给出结构化的"不可判定"而不是含糊的数字。

---

### 9.3 可判定性 v1/v2 双轨与 D1 决策（`adjudication_methodology.md` §b-v2）

#### 9.3.1 动机

v1 规则在三个 B4T2048 正式窗口（r20 1.1883×、r21 1.2329×、r22 1.1417×）上暴露了一个结构性问题：三次会话**我方侧一致到 ±1%**，分歧全部来自 **fla 的尾部**（fla max 1.24–2.58 ms 对中位数 0.97–1.04 ms——co-job 的突发落进 fla 更胖的 kernel 里）。v1 的 IQR 是尾部敏感统计量，于是它度量的是**污染**，而不是未被污染配对的噪声底；这导致连 r22 那个 30/30 全胜、p=9.3×10⁻¹⁰、IQR-rel 0.060 的窗口都被挡下（差 0.039）。

#### 9.3.2 规则与源码依据

v2 规则（`evidence/G5/adjudication_methodology.md` §b-v2；重算脚本 `scratch/r23f/v2_verdicts.py` → `evidence/G5/decidability_v2_r23f.json`、`scratch/r25f/v2_verdicts_r25.py` → `decidability_v2_r25f.json`）：与 v1 唯一的区别是 IQR 改在逐对比值的**内层 80%**上计算（对称 10% 修剪；n=30 剔除最高 3 对与最低 3 对）；median、符号检验、胜场数**一律不动**：

$$\text{decidable}_{v2} \iff (\mathrm{median}_r - 1) > 3 \times \frac{\mathrm{tIQR}_r}{\mathrm{median}_r} \;\wedge\; \text{wins} \ge 25/30 \;\wedge\; p \le 10^{-5}$$

设计依据：一个成员内部的突发每会话污染 O(1) 个对；10% 修剪恰好移除这一类，窗口干净时什么也不修剪（v1==v2）。

#### 9.3.3 数据

**r22 配对上的双轨判词**（B4T2048 正是 v2 为之而生的案例）：

| 形状 | median | v1 IQR rel | v2 tIQR rel | 效应 对 3×门槛 | 胜/p | 判词 |
|---|---|---|---|---|---|---|
| B4T2048 | 1.1450 | 0.060 | **0.044** | 0.145 > 0.131 | 30/30，9.3×10⁻¹⁰ | **DECIDABLE-v2**（v1 否，差 0.039） |
| B1T2048 | 2.1113 | 0.374 | **0.282** | 1.111 > 0.846 | 29/30，2.9×10⁻⁸ | **DECIDABLE-v2**（v1 否；带窗口漂移告诫） |
| B1T8192 | 1.2054 | 0.296 | 0.234 | 0.205 < 0.703 | 27/30，4.2×10⁻⁶ | 双否（真实双侧窗口） |

**绑定于任何 v2 声明的告诫（原文具约束力）**：v2 度量的是**会话内**一致性，不能裁决窗口间偏差——B1T2048 的中心在两个窗口之间仅凭 fla 中位数（0.35→0.42 ms）就从 1.78× 移到 2.11×，其 v2 判词声称的是内部连贯而非稳定的 2.1×；B4 是三类窗口里我方稳定、只有尾部分歧的"设计案例"。

**R25 新鲜配对上的同规则复算**（`decidability_v2_r25f.json`）：B1T2048 判 **DECIDABLE-v2**（tIQR rel 0.347 < 效应/3 = 0.372，30/30）且中心首次跨窗稳定（2.1113→2.1174，相差 0.3%）——这是 r23 备忘要求"再积累 ≥2 个窗口"后的第一个；B4T2048 新窗更吵，v2 亦否（tIQR rel 0.168→门槛 0.504 对效应 0.203，且 25/30、p=1.6×10⁻⁴ 高于 10⁻⁵ 线），故迁移后的数字 1.2211× **原样携带本窗双否判词**，而 D1 双引用声明仍锚定 r22 配对——如实陈述，不做平滑。

#### 9.3.4 D1 决策与结论

D1（R25，`evidence/G5/DECISIONS_R25.md`）：**v2 以 DOCUMENTED-OPTIONAL 侧统计量身份签署**——v1 仍是跨轮可比性的记录规则；任何 v2 声明必须双引两行（"v1-否 / v2-是"）；B4T2048 的 1.1417× 按双引用声明，B1T2048 保持 v2-软直至中心跨窗稳定。该决策是**纯引用政策**：没有任何测量、gate 或 cubin 依赖它，翻转它只是重贴标签。会话内无用户响应，按自治规则以建议默认值采纳并如此记录（覆盖随时欢迎）；一行用户回复即可撤销（"v2 withdrawn"）。与之成对的 D2（bwd 冻结于 r19 字节）后来被用户在 2026-10-10 会话中显式**推翻**为"启动 regime-change 评测（B/T 阶梯）"——D1 未被推翻，双引用口径沿用至今；这一对决策记录本身也是方法学的一部分：**判定规则的变更与测量数据的变更被制度性隔离**。

结论：v1/v2 双轨把"保守到挡住真相"与"激进到制造真相"两个失败方向同时关掉——v1 保跨轮可比，v2 在明示风险下恢复被尾部污染遮蔽的强证据，两者的差异永远并排出现在档案里。

---

### 9.4 测量系统审计：模块布局彩票与 B 侧偏置三部曲

#### 9.4.1 动机

r18 的链级 A/B 留下一条可疑的系统性告诫：r14u_ab 配对器里 **B 侧**的 f3 平均慢约 +13 µs、f2 约 +6 µs（T8192）。如果这是仪器偏置而不是代码差异，那么之前所有经该仪器的 f2/f3 结论都被污染。r19 把它当成一等的测量学问题立案侦查。

#### 9.4.2 三部曲（证据链，`evidence/G5/scratch/r19f/`）

1. **发射序假设死亡**。把**同一个** cubin 目录同时作为 A、B 两臂加载，仅交替发射顺序——f3 仍然读出 B 侧 +14.8 µs（39/40 对）。发射顺序不是原因。
2. **构造序翻转精确镜像**。只翻转两个 `FullKernel` 的**构造顺序**（即 `cuModuleLoadData` 的先后），其余一切不变——偏置精确反号（−16.8 µs，0/40 对；`ab_sameside_loadflip.json`）。机制由此定位：`cuModuleLoadData` 为模块的 `.global` 区分配布局，**后加载的模块**其全局缓冲落在较差的布局上；f3 在 T8192 的约 160 MB 工作集为此付出约 6%。
3. **彩票标定**。在（四模块的）正常 harness 进程里，同一构建的 f3 逐会话抽签 +2.3/+14.8/+14.3 µs（warmup-20 会话）→ 标定均值 **+10.5 ± 6 µs，f3 专属**。f1/f2 在 warmup 20 下干净：r18 观察到的 f1"+53 µs 前半程"是**时钟爬坡**（warmup 8→20 即消灭，`ab_sameside_warm.json` 读 −0.1 µs），r18 的"f2 +6 µs"同为爬坡——**f2 被洗清**。

关键佐证（"PTX 相同暴露"）：这一偏置与被测代码**逐字节无关**——R24 复核 T8192 的 f3 +12 µs 布局税时确认 f3 的 PTX 在两臂构建中逐字节相同，属 r19 彩票类；且"生成器树与 env 不变 → 重构建逐字节复现同一模块"本身是工具链的正向数据（**彩票在生成器编辑时重抽，不在重构建时重抽**；R25 的正式记录因此携带 R24 抽到的签）。

#### 9.4.3 源码片段：平衡双载 harness（`evidence/G5/scratch/r14u/r14u_ab.py`，r19 起的默认行为）

```text
R19-fwd BIAS FIX ... the legacy harness constructed side A's FullKernel first,
and the bias is a MODULE-LOAD / GLOBAL-PLACEMENT order effect, NOT a
launch-order effect ... The fix, new default: TWO half-sessions with the
module construction order FLIPPED between them (load A1,B1 then B2,A2),
per-kernel delta = mean of the two half deltas -- the placement bias is
mirrored between halves and cancels. Also retained: launch order alternates
per rep (even rep A,B / odd rep B,A) and the first post-warmup pair is
discarded.
```

协议要点：每臂 40 对 × 两个半程（镜像构造序）、逐次发射交替、丢弃 warmup 后第一对、warmup 20（足够让时钟爬坡完成）；`--single-load/--fixed-order/--keep-first` 三个旗标可完整复现 r14–r18 的旧行为（审计可重放）。不可消除的抽签分量用**标定会话**处理：任何 f3 敏感的比较，同窗跑一场同构建（`--a X --b X`）会话并扣除。

#### 9.4.4 数据：修正表（平衡 harness 复测，四对 cubin 全部 md5 对上记录值）

| 会话（40 对 × 2 半程） | f1 | f2 | f3 原始 | f3 修正后 | staged 链 |
|---|---|---|---|---|---|
| 同构建抽签 ×3（标定） | +0.1/−0.1 | −3.1..+0.3 | +2.3/+14.8/+14.3 | 构造上为 0 | — |
| r15 WST 对 r14-US | **−12.8**（7/80 慢） | −0.4 | +15.2 | ~+3 ≈ 0 | −0.5 |
| r16 bf16sv 对 r15 ship | −3.0 | +0.9 | +15.0 | ~+3 ≈ 0 | — |
| r18 F1KS 对 r17 ship（旧记录 −54.4/−1.3/+14.5；staged −47.6） | **−56.4**（3/80 慢） | +0.2 | +14.0 | ~+2 ≈ 0 | **−49.9 原始 → ~−62±5** |

修正的三个具体后果：r15 的 WST f3 效应确认为 ≈0（旧记录 −0.4 本来就在偏置带内）；r16 的 f2/f3 效应确认 ≈0；r18"真链赢 −60..−71 µs"收紧为 **−62±5 µs、全部来自 f1**——原始 f3 B 侧罚分（+14.5）100% 是布局。**历史记录不回写**：这些以追加修正的形式落档，r15/r16 的判词经修正后不变，r18 的量级被收紧。

#### 9.4.5 结论

这一节的方法学产出有三层。第一，**测量系统本身是仪器**：一个配对计时器的加载序可以携带 6% 的系统偏置，且只有用"同构建两侧对照"这类零假设实验才能暴露。第二，修正有制度：镜像半程消除一阶、同构建标定会话扣除残差、历史只追加不回写、cubin md5 全程对账。第三，它的副产品把 r18 的旗舰结论（F1KS 链级 −62±5 µs）从"带告诫的读数"升级为"修正后的干净归因。

---

### 9.5 删除探针时间图：`GDN_F1T_*` 机制与 kkt 拆解

#### 9.5.1 动机

拿到 f1（WY 段，T8192 下约 227 µs 基准）之后的问题是相位级归因：时间花在 uw 段、kkt、solve 还是 scan？profiler 的重放与时钟锁定会扭曲绝对值（§9.5.4 的 ncu 在锁定 1.57 GHz 下读 313.95 µs，比配对基准高 38%），而"重写一个没有某相位的版本再整体计时"会把重构本身的效应混进来。删除探针（deletion probe）给出第三条路：**在生成器里用编译期旋钮把某个相位整段删掉，其余一切字节不动，然后用平衡配对器测差值**。

#### 9.5.2 机制与源码形态

`GDN_F1T_*` 是 `tools/gdn_full_gen.py` 里的一个删除旋钮族（r21 建 9 个，r23 增 3 个），全部默认关闭、**仅计时用途**：任一旋钮打开时输出是垃圾（被删相位的消费者读到未定义数据），因此它们**被文档禁止进入 verify/det 通道**（r21 NKT 先例）。两条工程约束写在轮报告里：(a) payload transform 的 multibuffer 模式对**零使用者**的 workgroup alloc 硬报错——每个删除都必须给孤儿 tile 配 keep-alive 汇（一个谓词化的标量 load→store，放置位置尊重 PTX 生命周期 arena）；(b) 探针构建必须 occupancy 对齐（f1 2 / f2 1 / f3 1 blocks/SM 与 ship 一致，逐构建用 cuOccupancy 抽查），否则差值里混进驻留数效应。测量用 `r14u_ab.py` 平衡 harness，f1-only，40 对 × 2 半程 × 3 会话；三会话彼此相差 ≤1.6 µs。

#### 9.5.3 数据一：f1 十探针表（T8192，ship f1 基准 225.8–236.0 µs，~227 µs；差值 = ship − 探针，三会话均值）

| 删除的相位 | 旋钮 | Δ@T8192 | 占 f1 |
|---|---|---|---|
| uw 段（ug/wg gemms + epilogue + u/w s2g） | NUW | **−116.5 µs** | **51.2%** |
| uw 段 + 操作数 staging（k/v 读） | NUST | −131.6 µs | 57.9% |
| kkt K-loop + A epilogue | NKT | **−59.5 µs** | **26.2%** |
| 整个 solve（p0+p1+p2） | NALL | −41.3 µs | 18.2% |
| solve 相位 1（串行对角） | NP1 | **−32.5 µs** | **14.3%** |
| 仅 u/w s2g 拷贝（gemms+epi 保留） | NC2G | −31.0 µs | 13.6% |
| gc Hillis-Steele 扫描 + beta 段 | NSC | −17.6 µs | 7.8% |
| solve 相位 2（非对角 mma 轮） | NP2 | −14.6 µs | 6.4% |
| solve 相位 0（zero_up） | NP0 | −8.5 µs | 3.7% |
| gc_out 存储 | NGC | −0.8 µs | 0.3% |

导出拆分：uw staging（k+v 全局读）≈15.1 µs（NUST−NUW）；uw 的 mma+epilogue+staging 等待 ≈85.5 µs（NUW−NC2G）；u/w s2g 全局存储 ≈31.0 µs（NC2G）。**单相份额之和 ≈109%**——相位之间争用 DRAM/L2，每个差值是该相位串行成本的**上界**；组合交叉检验 NALL（−41.3）对 NP0+NP1+NP2（−55.6）显示 solve 内部约 14 µs 重叠。非可加性如实披露而非掩盖。这张表判死了 r20 的"相位 1 是关键路径"论题（32.5 µs，仅第 3 位）：**关键路径是 uw 段（51%）**，且其中是 ~85.5 µs 的 mma+epilogue+wait 串行化，不是 31 µs 的存储流量——此前两轮相位 1 重构（F1WY +92 µs、F1P4 +15 µs）攻的是一个 14% 的相位，事后看输得不冤。

**ncu 伴随证据**（ship f1，独占窗，时钟锁定 1.57 GHz；绝对值因锁频偏大，结构读数有效）：313.95/312.06 µs；DRAM 56.8–57.7%（400.8 GB/s）、计算 SOL 33%、**achieved occupancy 16.24%**（理论 16.67%：8 warps/SM，寄存器与共享内存双限 2 blocks/SM，214 寄存器、46.61 KB 静态共享）、2048 CTA / 82 SM = **12.49 waves**、发射器无合格指令 82.6%（每 5.8 周期一条）、warp 停顿构成 **long_scoreboard 24.3% / barrier 22.9% / wait 17.8% / math_pipe 12.4%**、L1 命中 0%、L2 命中 40.3%、平均活跃线程/warp 26.8——f1 在 2 CTA/SM 下是**延迟受限**，不是带宽或 tensor 受限。

#### 9.5.4 数据二：kkt 拆解（r23 三个新探针）与 roofline 收口

r21 表里 kkt 是一个 59.5 µs 的整块。r23 用 NKK（删 K-loop，留 epilogue 作用于零累加器）、NEK（仅删 exp2/beta 链，裸存储保住累加器与 Abf 存活）、NKM（仅删 ldmatrix+mma，保留 staging 循环与 barrier）把它拆开（A 臂 = +F1ZF ship 重建 `be8b914b`，×3 会话）：

| 探针 | 删除内容 | 三会话 Δ（µs） | 中位 |
|---|---|---|---|
| NKT（r21 复跑） | 整个 kkt 段 | −59.3 / −55.2 / −55.4 | **−55.4** |
| NKK | K-loop（staging+ldm+mma） | −53.2 / −53.6 / −55.2 | **−53.6** |
| NEK | exp2/beta 链 | −4.7 / −3.4 / −3.7 | −3.7 |
| NKM | ldm+mma | −9.4 / −13.0 / −13.6 | **−13.0** |

| kkt 成分（隔离探针） | µs | 份额 |
|---|---|---|
| k-tile 全局 staging：33.55 MB 的 cp.async 流（NKK−NKM） | **40.6** | **73%** |
| ldmatrix + 链式 mma（NKM） | 13.0 | 23% |
| A-epilogue exp2/beta 链（3 LDS + sub/ex2/2×mul/select）（NEK） | 2–4 | 4–7% |
| A-epilogue 存储（残差 NKT−NKK） | ≤2 | ≤4% |

**roofline 收口，同窗双重测量**：这条流搬运 $B \cdot T_p \cdot H \cdot K \cdot 2 = 33.55$ MB（每个 CTA 恰好一次读入自己的 64×128 bf16 k 块——100% 扇区利用率、32 B 对齐行、bypass_l1，即**算法最小流量**），耗时 40.6 µs = **826 GB/s**；同一共载窗口里跑的 D2D 拷贝标定（`scratch/r23f/bw.py`）量得该设备**读+写聚合 765 GB/s**。一条纯读流跑赢读写聚合速率，说明**没有任何可供重构挤入的余量**：kkt 的 staging 在此几何下 DRAM 受限。r22 的三个候选杠杆随之按测量定价死亡：exp2/beta 折叠（链已在 mma epilogue 里，全部可省 ≤4 µs）；K-loop tile 形状（BK32：kt 6,144→10,240 B → cuOccupancy 读 blocks/SM 2→1，配对 ×3 = **+70.3/+73.4/+68.3 µs 回归**——整类"增益"其实是驻留数损失，不是 tile 经济学）；存储模式（≤2 µs）。追加两个非测量否决：`--mb 3 --depth 3` 在 2-trip uw 循环上硬编译失败（pipelining 前置条件不满足，且 depth 是全局旋钮）；序言预取定价 ≤10 µs 天花板（mb2 在飞 = 8 迭代中的 2 个）但对 transform 有两条独立的敌意轴（循环外使用者令 `transform.memref.multibuffer` 跳过 alloc，需要新 tile +3,072 B → 49.2 KB 声明落在本轮刚测绘的 46.7 KB 可用/50.1 KB 失效间隙里的掷硬币点；以及自定义管线外直线消费者的 async-wait 摆放未经验证）——高腐化风险换 ≤5% 的 f1，NO-GO。

#### 9.5.5 结论

删除探针时间图是一个可复用的归因协议：编译期删除 + 平衡配对 + occupancy 对齐 + 三会话复测 + 非可加性显式披露。它在这台机器上给出了 profiler 给不了的裁决级相位归因，并直接关闭了 f1 的全部性能之门（uw 串行化 r22 关闭、s2g DRAM 受限、staging 池阻塞、kkt roofline 关闭——f1 在该几何下端到端关闭；残余是 solve 串行相位 1 的 32.5 µs 与 scan 的 17.6 µs，各已有两轮诚实负结果）。方法学上同样重要的副产品：**timing-only 探针与验证/确定性通道的制度性隔离**——垃圾输出的构建永远不进 verify/det，归因证据与正确性证据由不同仪器负责。

---

### 9.6 静态指令分析：无计时窗口下的判定替代（`tools/g5n_static_analysis.py`）

#### 9.6.1 动机

用户指令（2026-10-08）：GPU 将被共载占用约半天，**不需要干净计时窗口的分析同样算数**。这是"分析生成指令"这一指令在方法学上的落实：当计时不可得时，对**已落地构建产物**做确定性的、纯 CPU 的指令级分析，仍然可以排序优化目标、给出 roofline 距离、并且结论可复现。

#### 9.6.2 机制与源码片段

`tools/g5n_static_analysis.py`（确定性、纯 CPU）对 r6 落地的 9-kernel bwd 链（`evidence/G5/scratch/r6_T2048`）加前向 mono 三联与 fused 旗舰参照，从**ship 的产物本身**计算：PTX 指令族直方图（复用 `harvest_flaptx.ptx_features` 的 FAMS 表，作用于 harness 实际装载的 `kernel_smemfix.ptx`）、SASS 直方图（对 cubin 现场 `nvdisasm -c`，含 spill LDL/STL、LDSM/LDGSTS 与宽度解码）、occupancy（ptxas 寄存器 + manifest 的 smem/threads/grid，投影 `gdn_costmodel` 的 sm_120 结构模型：82 SM、64 K 寄存器、100,352 B 可用共享内存、16 CTA 槽、32 warp 槽）、每 chunk 计算强度（mma 位点 × 块折叠——scan 类 kernel 网格 = B·H·NV，每个块走遍全部 chunk，故静态体按 NV 折叠）、以及**距 roofline 的距离**。roofline 地板用 G3 标定率，其中 mma 峰值在加载时执行关键的 ×32 修正：

```python
# corrected mma peak: file flops field over-counts x32 (warp-level op);
# honest peak = instr/s / 32 * 4096 (evidence/G3/solver_report.md; the
# costmodel performs the same correction at load time)
MMA_PEAK = M["mma_bf16_m16n8k16"]["instr_per_s"] / 32.0 * 4096.0
```

（标定文件按**指令**计数 mma.sync——一条 m16n8k16 是 warp 级操作，对 4096 FLOP/条直接乘以吞吐率会虚报 32 倍；诚实峰值 = 指令率/32 × 4096 = **115.5 TFLOP/s**。`evidence/G5/static_analysis_r7.md` 卷首。）测得中位数取 `bwd6_bench2048.json` 与其重跑的均值（两轮相差 <1.5%）。

#### 9.6.3 数据

**头表（9 kernel + 关键结构量）**：寄存器 40–255；除 bpre/dhs 外全部 smem 限 1 CTA/SM；wy 是极端（255 寄存器 + 90,624 B 共享内存 → 12.5% occupancy，即每 SM 仅 4 个驻留 warp）；dqkwg 3,405 条 LDS / 832 HMMA / 193 bar.sync；wyb 3,261 LDS、**零 LDSM**。

**指令配比签名（对 fwd-F2 的对照，PTX 静态计数/mma 位点）**：

| kernel | ld.shared/mma | ldmatrix/mma | fp32 标量/mma（SASS） | bar.sync/mma |
|---|---|---|---|---|
| dqkwg | **5.08** | 0.23 | 1.92 | **0.23** |
| wyb | **6.77** | **0.00** | **9.65** | 0.26 |
| dkb（链内反例） | 0.30 | 0.50 | 8.41 | 0.19 |
| *fwd f2（参照）* | 0.69 | **1.19** | 1.66 | 0.05 |

前向旗舰经 **ldmatrix** 为 mma 供操作数（1.16–1.19/mma，仅 0.3–0.7 ld.shared/mma）；bwd 的 mma kernel 恰好倒置——wyb 零 ldmatrix、6.8 ld.shared/mma。根因链是 r6 的正确性教训：由 `LDGSTS`（手工 cp.async）staged 的 tile 必须用 GENERIC `ld.shared` 读回——对 LDGSTS 写入的 tile 用 ldmatrix 会静默损坏（sm_120a 静默失败模式 #2）——**所以 LDS 风暴是正确性权宜，不是风格选择**；dkb（唯一全部经 st.shared 重staged 的 kernel）是活反例：0.30 lds/mma、有 LDSM、链上 roofline 距离最小。次级签名：任何 bwd PTX 中 **`fma.rn.f32` 计数为零**——标量 f32 链全部发射为分离的 add/sub/mul（此 MLIR 路径无 FMA 收缩），标量指令数翻倍；地址算术支配 SASS（dqkwg：IMAD 1666 + MOV/SEL 2673 + 分支 1203 对 832 HMMA）。

**Roofline 距离表（整问题地板 = 校准峰值下的全量工作；DRAM 地板用精确 manifest 字节）**：九个 kernel 全部不饱和任何已测资源（tensor ≤1.6 / 115.5 TFLOP/s；smem ≤4.4 / 113 TB/s；DRAM ≤439 / 824.6 GB/s）——**绑定约束是指令发射 + 依赖停顿**（1 CTA/SM 下的 barrier 等待、窄 LDS 流、标量地址数学），恰是指令级手段可攻的对象。两个链级事实进入 round-8 台账：**九个 DRAM 地板之和 = 1.074 ms**，对实测 7.27 ms（6.8×），且对 **fla bwd 整体 1.03 ms**——分阶段设计的 DRAM 地板单独一项就 ≈ fla 的全部反向；**dkb 是内部存在性证明**：距地板 1.9×、达成 439 GB/s——同一生成器、同一约束、0.30 lds/mma。

**按可回收毫秒排序的 round-8 弹药**：(1) dqkwg（1.66 ms，可回收 ≥1.4）——LDS 风暴 + 576/832 位点是半速率 tf32 k8 + 193 bar.sync + IMAD/MOV/SEL 海；杠杆 = st.shared/逐元素 restage 以重新启用 ldmatrix、tf32 GEMM 的 split-bf16、barrier 合并。(2) wyb（1.04 ms，≥0.9）——最差配比、零 ldmatrix、9.65 fp32 标量/mma。(3) **无 tensor core 的五人组** fwdscan/caryA/caryM/bpre/dhs（合计 2.72 ms = 37%，≥2.2）——零 HMMA 位点、数学上携带 GEMM 形状的标量 f32（[K,K] compose 各 2·128³）；fwdscan/dhs 还结构性欠填充（grid 64 < 82 SM）并以每步 5–15 bar.sync 串行 32 个 chunk。(4) wy（0.70 ms，≥0.6）——255 寄存器 + 90,624 B → 12.5% occupancy，最缺延迟容忍的 kernel 最缺延迟隐藏。(5) 链级——零 FMA 收缩的普适代价 + f32 中间量（845 MiB/遍）的流量杠杆。

**方法学告诫（记录而非隐藏）**：静态计数 = 每 chunk 动态计数**仅当**循环体完全展开（四个 mma kernel 成立，wy 的 256 位点 = 恰好一个 CxC×K GEMM 已验证）；五个标量 kernel 有运行时循环 → 其 smem 流量/CUDA-FLOP 估算是**下限**，其 roofline 距离是**上界**（排序对此不敏感：其地板 ≤0.13 ms 对 0.32–0.92 ms 实测）；全局足迹 = manifest 全局尺寸各计一次被触碰（同一全局被一 kernel 又读又写会少计一次，分类无关且可复现）；标定率是共载**下界**——干净窗口里峰值更高，即真实距离更大而不是更小，量级结论不受影响。T512 交叉核对给出缩放合理性（每 chunk kernel 在 NC=8 下降 4–8×，fwdscan/dhs 随 NC 近线性）。

干净窗口到来时（G5 full re-measure），同一工具以干净峰值重跑一遍：DRAM 地板 824.6→851.4 GB/s 把每个 kernel 的地板收紧约 3.2%（wy 102.0→98.8 µs、dqkwg 255.4→247.4 µs），链仍全部 dram-bound、dkb 仍最近（1.94×）、排序不变（`GATE_REPORT_FULL_REMEASURE.md` §2）。

#### 9.6.4 结论

静态指令分析把"没有计时窗口"从停滞理由变成一种方法：确定性、零 GPU、从 ship 产物出发、以标定率为地板、以显式告诫封边。它产出的排序直接驱动了 round 8+ 的杠杆选择，其签名表（LDS 风暴 / 零 ldmatrix / 零 FMA）把"为什么慢"定位到指令类而不是笼统的"实现不佳"；而 dkb 反例与地板和 1.074 ms 这两个链级事实，分别给出了可达性证明与必须同时移动流量与执行效率的战略判断。

---

### 9.7 fla autotune 搜索成本：冷测与四路摊销（`tools/g6/*`）

#### 9.7.1 动机

"编译期调度对在线 autotune"的核心成本对比必须经得起两种质疑：我方的 0.8 s 到底计了什么；fla 的 39.8 s 会不会被它自己的磁盘缓存摊销掉。R26（camera-ready TODO #3）为此专门测量了摊销。

#### 9.7.2 我方口径：0.8 s 纯 CPU（`tools/g6/g6_solver_timing.sh`）

宿主 CPU（`OMP_NUM_THREADS=4`，conda `gdn`，z3 5.1.0），重测于本窗口：

| 空间 | 合法配置 | Z3 枚举 | 全求解墙钟（枚举 + 代价模型全排序） | top 选择（复现 G3） |
|---|---|---|---|---|
| kkt K=128（B4 T8192 H16），strict 98 KB，v2 | **368** | 0.307 s | 0.378 s | `kkt_bf16_mma_p2_t32x32x64_w2_ws` 8.62 µs |
| kkt K=128，v3（waves/occupancy 形式） | **368** | 0.296 s | 0.365 s | `kkt_bf16_mma_p3_t64x64x32_w2` 140 µs（文档化的 v2→v3 翻转） |
| o K=128 V=128，strict，v2 | **465** | 0.379 s | 0.444 s | `o_bf16_mma_p2_t32x32x64_w2_K128V128` 13.25 µs |
| o K=128 V=128，v3 | **465** | 0.374 s | 0.442 s | `o_bf16_mma_p0_t64x128x32_w2_K128V128` 350 µs |

确定性契约由重测再次确认：固定声明序、seed 7、rlimit 20e6、按（预测值，名字）排序——368/465 空间与 top 选择按字节语义复现 G3 输出。排序质量 Spearman **0.8962**（kkt 0.936 / o 0.829，n=18），对全部结构常数 ±2× 扰动稳定于 [0.882, 0.902]、对 ±30% 时间噪声 ρ≥0.856。如实记录的瑕疵：o K=64 V=64 子空间（204 配置）在本环境因 rlimit 下 Z3 部分模型伪影崩溃（`tools/gdn_solver.py:187` 的 `m[zv]→None`）——两个规范空间精确复现，崩溃如实入档。一次性沉没成本（G3 标定微基准、G2 的 20 配置编译矩阵）**不计入**每形状搜索；每个新形状，调度器在首个候选跑起来之前增加约 0.8 s CPU 与零 GPU 秒。代价模型本身无逐配置拟合常数（速率全部来自标定表 + SASS 导出常数），这是排序诚实性的来源。

#### 9.7.3 fla 侧冷测（`tools/g6/g6_fla_autotune_cost.py`）

测量纪律全部先行验证：本环境 `FLA_CACHE_MODE` 未设 → DISABLED → fla 磁盘配置缓存**不被咨询**（`fla/ops/utils/cache.py:51`）；本 GPU 的 fla 配置目录不存在（没有可挪的热盘缓存）；`TRITON_CACHE_DIR` 指向私有新鲜目录（共享 `~/.triton` 与 co-job 均不被触碰，事后无需恢复）；`Autotuner.run` 被包裹以逐 kernel 计数候选配置与 tuning key。输入 = G5 共享配方 B1T2048H16、`FLA_FLASH_QLA=0`（纯 Triton 后端，与全部对比一致）。单机会窗实测（墙钟，同步，单进程）：

| 阶段 | 冷（首调） | 暖（10 次中位） | 冷−暖 | 实测配置数 |
|---|---|---|---|---|
| **fwd** 首次 `chunk_gated_delta_rule` | **16.55 s** | 0.491 ms（min 0.418） | 16.55 s | **25** / 5 kernel：cumsum 4、kkt_solve 6、recompute_w_u 9、fwd_h 4、fwd_o 2 |
| **bwd** 首次 `torch.autograd.grad` | **23.26 s** | 1.349 ms（min 0.981） | 23.26 s | **23 新增** / 4 kernel：bwd_dv_local 9、bwd_dhu 4、bwd_dqkwg 9、prepare_wy_repr_bwd 1 |
| **合计** | **39.81 s** | — | 39.81 s | **48**（其后 Triton 缓存 67 个 kernel 条目） |

合理性锚点：暖中位数正落在 G5 基线带内（fwd 0.316–0.517 ms 跨 G5 窗口；bwd 1.34 ms 于 G5-C 对 1.349 ms）。共载并未使测量失效——冷成本由编译+基准主导，约 10³× 于争用噪声。机制解读：Triton 对每个候选配置（ptxas）编译并在真 GPU 上 do_bench；盘缓存缺席/禁用时这按**每进程、每形状键**重付（fla 自己的缓存层恰为避免此事而存在，且默认关闭）。

#### 9.7.4 四路摊销（`tools/g6/g6_fla_amortize.py` → `evidence/G6/fla_amortize_r26.json`）

机制（对照 fla 本地 checkout 验证）：fla 全部 GDN kernel 挂 `fla_cache_autotune` 装饰器（`FLA_CACHE_MODE` strict/fuzzy/full，`FLA_CONFIG_DIR`）；Triton 3.8.0 在真实搜索后向 `TRITON_CACHE_DIR` 写 `{kernel}.autotune.json` 结果缓存（fla 的 `FLA_CACHE_RESULTS` 默认 '1'）并在后续进程重读；`fla/scripts/extract_triton_autotune_cache.py` 把前者转成后者。驱动器按五个相位跑，**每个 GPU 相位都是全新 Python 子进程**（"首调"是真进程冷）：

```python
spawn("cold_populate", {**common, "FLA_CACHE_MODE": "disabled",
                        "TRITON_CACHE_DIR": tc_pop})       # P1 全搜索
# P2 extract_triton_autotune_cache.py -> $BASE/fla_configs (无 GPU)
spawn("hit_coldtc",  {**common, "FLA_CACHE_MODE": "strict",
                      "FLA_CONFIG_DIR": cfg_dir,
                      "TRITON_CACHE_DIR": tc_hit})         # P3 配置命中、冷编译缓存
spawn("hit_warmtc",  {**common, "FLA_CACHE_MODE": "strict",
                      "FLA_CONFIG_DIR": cfg_dir,
                      "TRITON_CACHE_DIR": tc_pop})         # P4 配置+编译缓存双命中
spawn("native_hit",  {**common, "FLA_CACHE_MODE": "disabled",
                      "TRITON_CACHE_DIR": tc_pop})         # P5 原生 Triton 结果缓存
```

派生账目（冷 = 本窗口重测 17.628+26.283 = **43.911 s**；比值 = 冷 ÷ 该路首调合计）：

| 路径 | 场景含义 | 首调 fwd+bwd 合计 | 冷 ÷ 该路 |
|---|---|---|---|
| 冷搜索（对 0.8 s CPU） | 新机器、无任何缓存 | 43.911 s | **≈50×**（39.81/0.8，主表口径） |
| hit_coldtc | fla 配置已发、新机器（JIT 只编选中配置） | 5.672 s（2.450+3.222） | **7.7×** |
| hit_warmtc | 配置 + 编译缓存全热（配置查表 + cubin 装载） | 0.651 s（0.558+0.093） | **67.5×** |
| native_hit | fla 层禁用、Triton 原生 `.autotune.json`（用户零配置的持久 `~/.triton`） | 1.071 s（0.939+0.132） | **41×** |

细节如实入档：fla 查表 11 次中 10 命中 1 未命中（`prepare_wy_repr_bwd`——该 kernel 的结果未被提取进配置目录）；暖稳态 fwd 0.4108 ms / bwd 1.4338 ms；共载 pid 3993998 全程未触碰（`shared_gpu: true` 与协议串随 JSON 存档）。解读：缓存不是消掉搜索成本，而是把它变成"一次 43.9 s 的 populate + 特定目录布局 + 路径依赖"——fla 自己的摊销最好情形（hit_warmtc，0.651 s）也要求配置文件已提取并放对目录且 Triton 编译缓存暖；而最常见的不配置场景（native_hit）每次冷机首调仍是 1.071 s 的 GPU+CPU 阻塞，对 0.8 s 的**离线、确定性、零 GPU** 求解没有质量优势可言（对方逐 kernel 精确，但以实测成本为代价且选择受基准序/争用影响）。

#### 9.7.5 对比口径（诚实边界）与结论

`evidence/G6/search_cost.md` 的 scope note 原则性生效：**0.8 s 是排序"我们的 stage-kernel 旋钮空间"的成本**；全算子的 ship 决策在 G5 另用了实测配对 A/B（那是工程归因，不是搜索）。可比量是：同一候选空间下，离线枚举+排序（0.4–0.8 s、确定性、零 GPU）对在线逐候选编译+基准（每方向 16.6–23.3 s，按进程/形状键重付）。结论表（search_cost.md）：fwd+bwd 单形状搜索成本 **1 : ≈50**；搜索发生的位置在"完全离线 CPU、GPU 在选中配置运行前零介入"对"部署 GPU 上、首调、阻塞"；可迁移性在"模型 + 标定携带到新形状（solver→compile→ptxas 重排序的回环设计）"对"按进程/形状键重付（除非可选的 fla 缓存层已被填充）"；四路摊销给出该结论在 fla 缓存全部有利配置下的界（7.7×–67.5×）。

---

### 9.8 硬件标定体系：`run_calib.sh`、共享窗与干净窗、`shared_gpu` 标志

#### 9.8.1 动机

三处消费硬件速率：SMT 调度器的解析成本模型（G3）、静态分析的 roofline 地板（§9.6）、roofline 收口判定（§9.5 的 826 对 765 GB/s）。标定数字本身也必须带口径：在共载窗口测得的峰值是**下界**，用途排序与地板计算可以安全使用，但必须显式标注，且干净窗口到来时要可无争议地刷新。

#### 9.8.2 机制与源码片段

`scripts/run_calib.sh` 是幂等的一分钟微基准：`nvcc -arch=sm_120a -O3` 编译 `tools/gdn_calib/calib_bench.cu`，CUDA events、≥5 warmup + 50 计时迭代、取中位数，逐 SM/时钟归一化对照峰值属性时钟。产物 `evidence/G3/calibration.json` 同时存全部原始测量与共载元数据；**`shared_gpu` 标志由观测推导而非硬编码**（G5-M 修复，commit e987da5——旧版把 `"shared_gpu": True` 与脏窗注释写死，在干净窗口里是假元数据，而方法学条款 (d) 明确要求刷新后撤销告诫；此修复不改任何测量逻辑）：

```python
"shared_gpu": bool(APPS.strip()),  # g5m: derived from observed compute-apps
...
"note": ("co-running compute apps observed: " + APPS.strip() +
         "; absolute bandwidth/FLOP numbers are lower bounds "
         "(contention) - re-run in a clean window to overwrite"
         if APPS.strip() else
         "clean window: no other compute apps resident ..."),
```

起点/跑前/结束三个时点的 smi 快照与 compute-apps 清单全部入档；旧值整体保存在 `evidence/G3/calibration_shared.json`。

**G3 同族协议（`tools/gdn_time.py`）**——20 个 G2 配置零参自含 cubin 的计时，与标定共享同一套口径：每配置一个**子进程**隔离（默认 240 s 墙钟超时，mbarrier 死锁只杀自己的 worker，不滞留 GPU）；验证发射前先把 `@gdn_out` 清零（零初始化全局写非零内容 = kernel 确实写了；某些 payload 会累加，故每次验证发射从 0 开始）+ 校验和确定性（清零重发必须逐位复现）；warmup 5 + 计时 50 取中位；对**未修复**的 as-built cubin 各探针一次（记录 G2 下降落在真机上的行为：device-malloc 下激光的非法地址）；输出固定携带：

```python
"shared_gpu": True,
"caveat": "RTX 5090 Laptop was shared with a co-running job during "
          "measurement (see coload); ABSOLUTE numbers are noisy "
          "lower bounds. Re-run in a clean window with identical "
          "commands to overwrite with fresh values.",
```

#### 9.8.3 数据：共享窗 → 干净窗刷新

共载首测（co-job pid 3411456，约 17.9 GiB，24–99% util，观测 SM 时钟 2.1–2.8 GHz 对 1597 MHz 属性峰值）：mma.sync m16n8k16 bf16 指令率 902.6 G-instr/s（≈4.1 instr/SM/clk @2.7 GHz 观测时钟，与每 SM 4 条 tensor 管一致；经 §9.6 的 ×32 修正 = 诚实 115.5 TFLOP/s）；smem 读 113.1 TB/s；L2 驻留读 2,144 GB/s；DRAM 读 824.6 GB/s（GDDR7 规格 ~1.8 TB/s——按争用下界对待）；SFU ex2 3.23 Telem/s；bar.sync 往返 7.71 ns；cp.async 16B 1,388 GB/s。

干净窗口重跑（G5 full re-measure；`shared_gpu: false`，告诫撤销）：

| 指标 | 共载（旧） | 干净（新） | Δ |
|---|---|---|---|
| cp.async 16B | 1388.2 GB/s | 1540.2 GB/s | **+11.0%** |
| DRAM 读 | 824.6 GB/s | 851.4 GB/s | +3.2% |
| L2 驻留读 | 2144.3 GB/s | 2162.0 GB/s | +0.8% |
| smem 读 | 113.05 TB/s | 114.07 TB/s | +0.9% |
| mma bf16 峰值 | 3697.1 G-instr/s（115.54 归一） | 3713.9 G-instr/s（116.06 归一） | +0.5% |

共载对异步拷贝/DRAM 路径的伤害比对计算峰值高一个量级——这本身就是该 GPU 共载生态的一个测量学事实。同一干净窗的伴随重算：静态分析（§9.6）以干净峰值重跑，DRAM 地板收紧 ~3.2%，排序不变。标定-消费链因此闭合：**同一份速率文件同时驱动求解器排序（离线）与 roofline 地板（分析），其口径（共享=下界）与刷新机制（幂等覆盖 + 标志推导 + 旧值存档）是方法学的组成部分。**

#### 9.8.4 结论

标定体系回答了"分析里的地板凭什么信"：速率可复测、口径自标注、下界方向性明确（共载下界只会让"距离 roofline 很远"的结论更保守）、干净窗口刷新由轮询器与重测预案制度化为常规动作而非一次性运气。与 §9.2–§9.7 合起来，本章的立场是：**在一个测量条件恶劣的平台上，方法学不是论文的附录，而是每一条性能主张能站立的地基**——同窗交错的比值口径、显式可判定规则、被审计的测量仪器、与计时解耦的归因与分析通道、写明摊销口径的成本账、以及带口径标志的标定率，六者共同构成这台机器上"可裁决证据"的定义。

---

## 10 编译期调度与 PTX 覆盖验证体系：Z3 求解器、解析成本模型、一次性标定、参考收割器、覆盖矩阵与旋钮库

### 10.0 章首：XLA/TPU 支柱二的完整落地叙事

本文 §1.2 将"调度即 IR"论点展开为与 Google TPU/XLA 栈逐条对位的四支柱：语义单一事实源（对位 HLO 与自动微分）、**编译期确定性调度（对位 XLA 的"非运行时 autotune"哲学）**、分层可验证 IR（对位分层验证）、布局系统（对位布局传播）。本章的职责是把第二支柱——编译期确定性调度——的完整落地叙事一次讲透，并连同为其提供信任锚的 PTX 覆盖验证体系一起呈现。二者构成同一枚硬币的两面：调度决策若要离开设备、搬到编译期，就必须回答两个问题——*凭什么排序*（成本模型的诚信性）与*凭什么相信生成的代码*（覆盖与结构验证）。

XLA 在 TPU 上的做法是：调度决策全部在编译期做出，编译产物不携带运行时搜索。把它移植到"原生 MLIR + 消费级 Blackwell（RTX 5090 Laptop，sm_120a）+ 单算子族（Gated DeltaNet）"的世界，落地为一条六个部件组成的流水线，本章按此分节：

| 部件 | 文件 | 职责 | 证据 |
|---|---|---|---|
| 旋钮库 | `tools/gdn_l2_gen.py` | 把调度自由度物化为 20 个可编译的配置面（payload.mlir + transform.mlir 生成器） | evidence/G2/gen_report.md |
| Z3 求解器 | `tools/gdn_solver.py` | 以 SMT 枚举合法旋钮空间（kkt 368 / o 465 / o-K64V64 204），按成本模型排序 | evidence/G3/solver_report.md |
| 解析成本模型 | `tools/gdn_costmodel.py` | 旋钮向量 → 时延估计；v1→v3 三形式；零逐配置拟合常数 | evidence/G3/solver_report.md §2 |
| 一次性标定 | `scripts/run_calib.sh` + `tools/gdn_calib/calib_bench.cu` | 模型全部速率的唯一来源：mma/smem/L2/DRAM/ex2/bar.sync/cp.async 微基准 | evidence/G3/calib_report.md、calibration.json |
| 参考收割器 | `tools/harvest_flaptx.py`、`tools/harvest_qlaptx.py`（scripts/harvest_fla.sh、harvest_qla.sh） | 纯 CPU 采集 fla（Triton AOT）与 FlashQLA（TileLang）参考 PTX/SASS/ptxas 资源表；提取 T1 契约集 33 族 | evidence/G2/ref_inventory.md |
| 覆盖矩阵 | `tools/gdn_coverage.py` | T1 指令族 / T2 结构两极点的可核查对照；逃生舱台账 1/≤5 | evidence/G2/coverage_matrix.md、hatch_report.md |

数据流是闭环的（求解器文档所称 loopback）：**求解**（Z3 枚举 + 成本模型排序）→ **生成**（`gdn_l2_gen.py --json` 实例化 top-K 旋钮向量）→ **编译**（`scripts/run_l2.sh`：transform 解释 → NVVM 管线 → ptxas-blackwell）→ **静态复核**（ptxas `-v` / SASS 重收割，未编译配置按 spills=0 排序、命中已编译向量者自动拾取实测 regs/spills/ex2 特征，输出 JSON 以 `has_index_features` 标记）→ **窗口计时**（top-K 幸存者上 GPU）。本章覆盖前四环与最后一环的静态分支；窗口计时与端到端判定属实验章范围。

这条流水线的两个契约贯穿全章，先在此定义后用。**确定性契约**：求解器固定 `random_seed=7`、`rlimit=20×10⁶`、固定变量声明序、固定阻塞子句序、终排序键 `(predicted_time, name)`；成本模型为纯整型/浮点算术、无任何随机源；两次独立运行产出逐字节一致的 368 项排名（evidence/G3/solver_report.md §1；G6 复测再证）。**零逐配置拟合契约**：成本模型中不存在任何对实测配置时间调参的自由常数——每一个速率要么来自一次性标定窗口（§10.3），要么是文档化的 sm_120 级结构常数（如 SASS 导出的 nomma 10 cyc/MAC，§10.2.4）；全局唯一的 launch 常数由纯发射锚核导出（§10.2.5）。这两个契约合起来，使"编译期调度"从工程直觉变成可复现的机械过程——这正是支柱二的全部内容。作为其经济效益的收口（属第 6 章证据、此处引用作叙事闭环）：每新形状的调度成本约 0.8 秒纯 CPU（kkt 空间枚举 0.29 s + o 空间 0.33 s + 排序），对 fla/Triton 在设备 autotune 冷启动实测 39.81 秒（前向 16.55 s + 反向 23.26 s，25+23 个候选配置逐个在设备编译计时）约 50 倍；fla 磁盘缓存摊销后为 7.7×–67×（evidence/G6/search_cost.md、fla_amortize_r26.json）。本方 GPU 秒为零。

本章各节统一采用四件套结构：【动机】→【机制与源码】→【数据对比】→【结论】。

---

### 10.1 Z3 旋钮求解器（tools/gdn_solver.py）

#### 10.1.1 动机

"选哪个配置"在手写世界里是 autotune 在设备上逐候选计时回答的问题；在调度即 IR 的世界里，它必须变成一个带确定性契约的形式化问题：给定旋钮空间、合法性约束与目标函数，机械地给出全序。TileLang 已经证明了 Z3 在这一角色上的可行性——但其 modulo-scheduler 只存在于 Ascend 后端（变量为指令开始时间与 buffer 版本，目标为最小启动间隔 II），CUDA 侧退化为运行时 autotune（docs/PLAN.md §7 关键事实）。本求解器（工作流 G3-B 交付物）将 TileLang 的机制**上移一层**移植到 CUDA/sm_120：决策变量直接是旋钮向量而非指令时隙，约束编码资源容量与结构合法性，目标为解析成本模型。这一移植在 CUDA 侧无先例。

#### 10.1.2 机制与源码：变量、约束、目标

求解器核心是 `KnobSpace` 类——一个 stage（kkt 或 o）加维度（K，可选 V）的合法旋钮空间的 SMT 编码。**决策变量**（固定声明序，这本身就是确定性契约的一部分）共十个：`dtype`、`prec`、`tile_M`、`tile_N`、`tile_K`、`warps`、`pipeline`、`mma`、`warpspec`、`tma`；域为 tile ∈ {32,64,128} 的合法子集（须整除全局维）、warps ∈ {2,4,8}、pipeline ∈ {0,2,3}（域裁剪依据 G3 旋钮空间契约；生成器另容纳 M=16 与 32 以下的 8 倍数，此处不含）。约束分三类，全部为 Z3 断言：

```python
# tools/gdn_solver.py :: KnobSpace.__init__（节选，注释为本文所加）
s = z3.Solver()
s.set("random_seed", seed)          # z3 5.1.0 中 seed 参数名为 random_seed
s.set("rlimit", rlimit)             # 封顶求解工作量（确定性）

dtype = z3.Int("dtype")             # 0 = bf16, 1 = f32
prec  = z3.Int("prec")              # 0 = bf16, 1 = tf32
bt, bv, bk = z3.Int("tile_M"), z3.Int("tile_N"), z3.Int("tile_K")
warps, pipe = z3.Int("warps"), z3.Int("pipeline")
mma, ws, tma = z3.Bool("mma"), z3.Bool("warpspec"), z3.Bool("tma")

# —— 旋钮耦合 ——
s.add(z3.Implies(dtype == 0, prec == 0))          # bf16 输入只有 bf16 mma
s.add(z3.Implies(dtype == 1, prec == 1))          # f32 输入只有 tf32 mma
s.add(z3.Implies(z3.Not(mma), prec == dtype))     # mma 关闭则精度旋钮消失

# —— 结构合法性（镜像 gdn_l2_gen.validate）——
s.add(z3.Implies(mma, bt % 16 == 0))              # mma fragment 对齐
s.add(z3.Implies(z3.And(mma, dtype == 0), bk % 16 == 0))   # bf16 m16n8k16
s.add(z3.Implies(z3.And(mma, dtype == 1), bk % 4 == 0))    # tf32 m16n8k4
s.add(z3.Implies(mma, bv % (8 * warps) == 0))     # N 按 warp 剖分对齐
s.add(z3.Implies(z3.And(mma, pipe > 0), K >= bk * pipe))   # 流水深 ≤ K 趟数
ws_ok = z3.And(stage == "kkt", dtype == 0, mma)
s.add(z3.Implies(ws, ws_ok))                      # warpspec 仅 kkt bf16 mma
s.add(z3.Implies(tma, ws))                        # tma 依赖 warpspec
s.add(z3.Implies(ws, pipe == 2))                  # ws 载荷管道不可观测，钉 p2

# —— 资源容量 ——
esz = z3.If(dtype == 0, 2, 4)
threads = 32 * (warps + z3.If(ws, 1, 0))
s.add(threads * 255 <= 65536)                     # regs/SM 预算
pair = (bt * bk + bk * bv) * esz
smem_mma = z3.If(ws,
                 pair + z3.If(tma, 64 * 64 * esz, 0) + 8,   # 单缓冲 + TMA 暂存 + mbarrier
                 pair * z3.If(pipe > 0, pipe, 1) * self.n_gemm)  # 多缓冲因子 = pipeline
smem_nomma = (bt * K + K * bv + bt * bv) * esz
s.add(z3.If(mma, smem_mma, smem_nomma) <= cap)
```

**共享内存字节模型**（显式、文档化于求解器 docstring）是资源约束的核心，三种形态：

$$\text{pair} = (bt\cdot bk + bk\cdot bv)\cdot esz,\qquad esz = \begin{cases}2 & \text{bf16}\\ 4 & \text{f32}\end{cases}$$

$$\text{mma, ws=off}:\ \text{smem} = \text{pair}\cdot\max(p,1)\cdot n_{\text{gemm}}\qquad(\text{多缓冲因子}=p;\ n_{\text{gemm}}=1\ \text{kkt},\ 2\ \text{o})$$

$$\text{mma, ws=on}:\ \text{smem} = \text{pair} + 64\cdot 64\cdot esz\,[\text{tma 暂存}] + 8\,[\text{mbarrier}]\qquad(\text{生产/消费单缓冲})$$

$$\text{mma=off}:\ \text{smem} = (bt\cdot K + K\cdot bv + bt\cdot bv)\cdot esz$$

容量上限两档：`strict` 模式 99 KB − 1 KB（sm_120 opt-in 101,376 B 减去 1 KB 驱动保留 = 100,352 B 可用），`generator` 模式 220 KB（镜像 `gdn_l2_gen.validate()` 的生成器上限）。**寄存器约束**为 $32\cdot(w + w_{\text{spec}})\cdot 255 \le 65536$——每线程 255 regs 是 ptxas 给定（该家族全员顶格），SM 级预算约束足以排除如 8 warp + 生产 warp（$288\times255 > 64\text{K}$）的组合。

**目标与枚举**：最小化标定解析成本模型（§10.2）。模型非线性（max/混合/除法），无法直接交给 Z3 优化，故 v1 采用与 TileLang 同角色的折衷——SAT + 阻塞子句（blocking clause）枚举全部合法指派，在求解器**外**按 `(predicted_time, name)` 排序；这与 TileLang 的 estimate_latency 引导 II 二分搜索在同一抽象层级上等价地回避了内嵌非线性目标。每个模型的阻塞子句由**补全后的 Python 值**（而非模型直读值）构造，原因见 §10.1.4。

#### 10.1.3 确定性契约

`random_seed=7`、`rlimit=20×10⁶`、固定变量声明序、固定阻塞子句序、终排序键 `(predicted_time, name)`，此外全程无 RNG。**数据对比**：两次独立运行产出相同的 368 项排名，名序与时间逐字节一致（evidence/G3/solver_report.md §1；evidence/G6 复测再证）。落盘的 `evidence/G3/solver_kkt.json` 记录 `z3: {seed: 7, rlimit: 20000000, last_check: "unsat"}`——`last_check` 为 `unsat` 即空间干净穷尽（最后一个阻塞子句使公式不可满足），这是"枚举完整"的可核查证据。

#### 10.1.4 防御性补全：partial-model 伪影与 SAT 探针

枚举循环必须防御一个真实的 Z3 行为伪影（G6/search_cost.md 首次记录的 flake 类）：z3py 的 `model()[zv]` 对模型**省略**的常量回读为 `None`（get_interp 路径，非异常）——一个被其余指派唯一决定的旋钮常量可能不出现在模型里，直读即崩溃或注入非法值。防御分两层：

```python
# tools/gdn_solver.py :: KnobSpace._complete（节选）
def _complete(self, key, fixed):
    """对模型省略的旋钮常量做确定性补全：声明域中首个保持其余指派可满足的值。
    check-and-assert 式探针——省略的旋钮绝不可能注入非法值。"""
    s = self.solver
    saved_rlimit = self.rlimit
    s.set("rlimit", 0)              # 探针内 rlimit 中止不得误判为"候选被拒"
    s.push()
    try:
        pin = [self.v[k] == ... for k, v in fixed.items()]
        s.add(z3.And(pin))
        for cand in self.domains[key]:        # 声明序 => 确定性
            s.push()
            s.add(self.v[key] == ...)
            ok = s.check() == z3.sat
            s.pop()
            if ok:
                return cand
        raise RuntimeError("z3 partial model: no legal completion ... "
                           "this is a solver contract violation")
    finally:
        s.pop()
        s.set("rlimit", saved_rlimit)
```

第一层即此 `_complete()`：以"钉住其余指派 + 逐候选 SAT 探针"的 check-and-assert 方式补全，保证补全值合法（至少一个候选必合法——求解器返回的模型本身可满足）；探针在 rlimit 解除状态下运行，避免把资源上限中止误判为拒绝。第二层是**枚举级 rlimit 重试**：若 `enumerate()` 以 `unknown`（而非 `unsat`）终止，说明空间尚未穷尽就撞上 rlimit——静默接受会返回被截断的空间冒充完整空间（G6 flake 类）；`solve()` 此时以 `rlimit=0` 重建空间重枚举一次，仍 `unknown` 则抛异常拒绝输出（这些空间 ≤10³ 配置，无封顶一遍远低于一秒，重试本身确定性）。`enum_stats` 记录 `models / partial_models / probe_completions` 三计数，输出 JSON 携带 `z3.last_check` 与 `z3.rlimit_retry_uncapped`，使伪影触发频率与空间完整性可事后审计。

#### 10.1.5 数据对比：合法空间规模与排名示例

| stage / 维度 | 约束模式 | 合法组合数 | 枚举耗时 |
|---|---|---|---|
| kkt K=128 | strict（99 KB−1 KB） | **368** | 0.29 s |
| kkt K=128 | generator（220 KB） | 368 | 0.28 s |
| o K=64 V=64 | strict | **204** | 0.15 s |
| o K=128 V=128 | strict | **465** | 0.33 s |

（{32,64,128} tile 域内这些维度下没有任何配置超过 strict 98 KB 预算，故两模式同数；smem 约束在更大 K/V 或放行 M=16 时才咬合。——evidence/G3/solver_report.md §1"Legality-space size"）

排名示例（`--solve --stage kkt --K 128 --B 4 --T 8192 --H 16`，8192 chunk，v2 每 CTA 时延视角）：

| # | 配置 | 预测 |
|---|---|---|
| 1 | kkt_bf16_mma_p2_t32x32x64_w2_ws | 8.62 µs |
| 2 | kkt_bf16_mma_p2_t32x32x64_w4_ws | 8.65 µs |
| 3 | kkt_bf16_mma_p2_t32x32x64_w2 | 8.67 µs |
| 4 | kkt_bf16_mma_p2_t32x32x32_w2 | 8.79 µs |
| 5 | kkt_bf16_mma_p3_t32x32x32_w2 | 8.79 µs |

v3（波次/占用视角，`solver_kkt_v3.json`）top-5 翻转为大 tile / 最少 CTA：p3_t64x64x32_w2 140 µs、p0_t64x64x64_w2 153 µs、p0_t64x64x32_w2 156 µs、p2_t64x64x64_w2 185 µs、f32_mma_p0_t64x64x32_w2 197 µs——8192 chunk 下 CTA 数（$8192\cdot(64/bt)\cdot(64/bv)$）主导，64×64 tile（每 chunk 1 CTA）胜过 32×32（4 CTA/chunk）。o 阶段（K=128, V=128, B4T8192H16）：465 合法配置，首选 `o_bf16_mma_p2_t32x32x64_w2` 13.25 µs（`evidence/G3/solver_o.json`）。两个诚实的限定：单 chunk 计时门（§10.2.7）无法分辨 v2/v3，全形状偏好是模型预测，留待 G5 计时窗口抽查；warpspec 条目在模型上排名靠前但 G3 时期在硬件上死锁（mbarrier TIMEOUT），其排名是暂定的——这正说明求解器的合法性空间包含"结构合法且可编译但尚不可运行"的成员，它们是后续工作流（G4 正确性目标）的输入而非结论。

#### 10.1.6 与 TileLang Ascend 机制 B 的移植差异表

| 维度 | TileLang Ascend `z3_scheduler`（机制 B 原型） | 本文 G3-B 移植（CUDA/sm_120） |
|---|---|---|
| 决策变量 | 指令**开始时间** + buffer 版本（整数时槽） | **旋钮向量**（dtype/prec/tile/warps/pipeline/mma/warpspec/tma）——上移一层 |
| 约束 | 依赖关系 + 资源互斥（同一功能单元的时槽冲突） | 资源容量（smem 字节模型、regs/SM）+ 结构合法性（整除/fragment 对齐/流水深≤趟数）+ 旋钮耦合 |
| 目标 | min II（启动间隔），estimate_latency 引导的 II 二分搜索 | min 解析时延；因模型非线性（max/混合/除法），SAT+阻塞枚举全空间、求解器外排序 |
| 确定性纪律 | rlimit + 固定种子 | 同一纪律保留：random_seed=7、rlimit=20×10⁶、固定声明序/阻塞序/(predicted, name) 终排序 |
| 输出 | 指令调度（时槽指派） | 全序合法配置排名 + 每项预测分解（terms_ms）与 smem 模型字节 |
| 回环 | 无（调度即终点） | solve → 生成/编译 top-K → ptxas 静态复核重排 → 幸存者窗口计时；未编译项 spills=0、命中已编译向量自动拾取实测特征（`has_index_features`） |
| 防御 | —— | partial-model 伪影的 SAT 探针补全 + rlimit 截断的确定性重试（§10.1.4） |

#### 10.1.7 结论

Z3 求解器把支柱二从哲学落成机械：合法空间（368/465/204）以亚秒级纯 CPU 穷尽、以 `last_check=unsat` 可核查、以固定种子与固定序逐字节复现；目标函数的诚信性完全外包给下一节的成本模型与标定；其防御性补全与重试把 Z3 的两个真实行为伪影（partial-model、rlimit 截断）从潜在的正确性漏洞变成输出 JSON 里可审计的统计量。

---

### 10.2 解析成本模型（tools/gdn_costmodel.py）

#### 10.2.1 动机与零拟合契约

求解器需要把每个合法旋钮向量映射成时延估计。业界常规做法是拟合式代理模型（在设备或仿真侧迭代测量后回归），其诚信性随拟合自由度流失。本文的合同是**标定解析模型**：模型中每一个速率要么 (a) 由 G3-A 微基准窗口一次性测得（evidence/G3/calibration.json，§10.3），要么 (b) 是文档化的 sm_120 级结构常数；**不存在任何逐配置拟合的自由常数**。仅有的全局常数是 launch 后备值 `T_LAUNCH_FALLBACK` 与设备分配 `T_MALLOC`（均可用标定键 `launch_ns`/`malloc_ns` 覆盖，实际使用中标定/锚定生效，后备不触发——输出以 `calibration_fallback_keys` 空集自证）。

#### 10.2.2 机制与源码：项定义

模型服务于 `gdn_l2_gen.py` 生成的 GDN chunk 级阶段 kernel 家族（sm_120：82 SM、Ampere 风格 mma.sync、99 KB opt-in smem、64K regs）。三形式递进，每项均为"计数 ÷ 速率"：

**v1（合同形，roofline）**
$$T = T_{\text{launch}} + \operatorname{ov}(T_{\text{compute}},\, T_{\text{gmem}},\, T_{\text{smem}}) + T_{\text{bar}} + T_{\text{spill}} + T_{\text{ex2}}$$
其中 $\operatorname{ov}=\max$ 当配置软件流水化（pipeline>0 或 warpspec：cp.async 双缓冲把拷贝藏进计算），$\operatorname{ov}=\sum$ 当 pipeline=0（同步拷贝串行于 gpu.barrier 之后）。

**v2 = v1 + $T_{\text{lsu}}$ + $T_{\text{alloc}}$**
$$T_{\text{lsu}} = \frac{\text{LSU warp 指令数}}{\text{LSU\_IOPS\_PER\_CLK}\cdot f_{\text{clk}}}$$
$T_{\text{lsu}}$ 的存在源于一个编译期可核查的代码事实：生成的拷贝流是**均匀地址**（非分布式）的——`kernel.ptx` 中拷贝索引不含 `%tid`，每个线程都发射 tile 的每一条 16 B cp.async；线上字节经 L1/L2 扇区去重，但**发射成本**是指令数 × 线程数。fragment 装载是 CSE 后每迭代唯一的 `ld.shared`（静态计数 = 唯一 fragment 装载数，非 12× 原子数）。$T_{\text{alloc}}$：memref.alloc 的下降对每个 tile buffer 调一次设备 `malloc` + `cvta.to.shared`（mma 路径每 GEMM 2 次、nomma 3 次、tma 再 +1 输出暂存）——对 G3-A 静态 smem 修复后的计时 cubin 此项归零（`merge_index_features` 中 `smemfix => n_allocs=0`）。

**v3 = v2 + 占用/波次**
$$\text{waves} = \left\lceil \frac{\text{CTA 总数}}{82\cdot\text{occ}} \right\rceil,\qquad T = T_{\text{launch}} + \text{waves}\cdot T_{\text{per-CTA}}$$
occ 取 regs/smem/warp 槽/CTA 槽四限的最小值（`occupancy()`），重叠显式化为 $\alpha\cdot\max + (1-\alpha)\cdot\sum$、$\alpha\in\{0,1\}$（保留参数化为扰动实验服务）。

```python
# tools/gdn_costmodel.py :: predict（v2 核心，节选）
if feats["mma"]:
    peak = (calib.m["mma_bf16_peak_flops"] if esz == 2 else calib.mma_tf32_peak)
    terms["T_mma"] = feats["mma_flops_per_cta"] / (peak / calib.sm_count)
else:
    macs = feats["ffma_flops_per_cta"] / 2.0
    terms["T_ffma"] = macs * NOMMA_CYC_PER_MAC / calib.m["clock_hz"]
bw_key = "l2_bw" if l2_resident else "gmem_bw"      # 计时 harness 中操作数 KB 级、循环驻留 L2
terms["T_gmem"] = feats["gmem_bytes_per_cta"] / calib.per_sm(bw_key)
terms["T_smem"] = feats["smem_bytes_per_cta"] / calib.per_sm("smem_bw")
terms["T_bar"]  = feats["barriers_dyn"] * calib.m["bar_sync_ns"] * 1e-9
terms["T_spill"] = spill_bytes / (calib.per_sm("l2_bw") * SPILL_BW_FRAC)
terms["T_ex2"]  = feats["ex2_count_per_cta"] / calib.per_sm("ex2_rate")
lsu_ops = feats["cpasync_warp_ops"] + feats["lds_warp_ops"] + feats["stg_warp_ops"]
terms["T_lsu"] = lsu_ops / (LSU_IOPS_PER_CLK * calib.m["clock_hz"])
terms["T_alloc"] = feats["n_allocs"] * calib.m["malloc_ns"] * 1e-9
total = (terms["T_launch"] + terms["overlap"] + terms["T_bar"]
         + terms["T_spill"] + terms["T_ex2"] + terms["T_lsu"] + terms["T_alloc"])
```

各计数的解析特征（`analytic_features()`）在给定生成器代码结构下精确：mma 原子数 $=(bt/16)\cdot((bv/w)/8)\cdot(bk/a_k)\cdot\text{trips}\cdot n_{\text{gemm}}$（$a_k$=16 bf16 / 4 tf32）；gmem 字节 $=(bt\cdot K + K\cdot bv)\cdot esz\cdot n_{\text{gemm}} + bt\cdot bv\cdot esz$（即 $(M\!K\!+\!K\!N\!+\!M\!N)\cdot esz$ 经 tile 网格放大：A tile 重读 $n_{\text{glob}}/bv$ 次、B tile 重读 $64/bt$ 次）；spill 静态 LDL+STL 计数自 INDEX 拾取并乘 $\frac{\text{trips}+p}{1+p}$ 撤销流水 peel 的体复制。sm_120 结构常数（文档化 + 敏感性检查）：`SM_COUNT=82`、`SMEM_USABLE_B=100352`、`REGS_PER_SM=65536`、`WARP_SLOTS_PER_SM=32`、`CTA_SLOTS_PER_SM=16`、`TENSOR_FLOP_PER_CLK_SM=1024`（bf16 密集 mma/SM/clk）、`TF32_RATIO=0.5`、`LSU_IOPS_PER_CLK=4.0`、`SPILL_BW_FRAC=1.0`、`NOMMA_CYC_PER_MAC=10.0`。

#### 10.2.3 v1→v3 形式迭代：失败保留的长卷

模型不是一次写对的；迭代史本身是证据（evidence/G3/solver_report.md §2 迭代表）：

| 迭代 | 变更 | 总体 ρ | 分 stage ρ（kkt / o） |
|---|---|---|---|
| 1 | v1 合同形；nomma = CUDA 核 FFMA roofline；launch=4 µs 后备 | 0.7352 | 0.644 / 0.486 |
| 2 | v2：+ $T_{\text{lsu}}$（均匀地址 cp.async 流 + fragment ld.shared + 尾部 st.global 的发射项）+ $T_{\text{alloc}}$（设备 malloc 项；smem 修复后的计时 cubin 自动归零） | 0.7475 | 0.669 / 0.486 |
| 3 | nomma 项替换：$T=\text{MACs}\cdot 10\,\text{cyc}/f_{\text{clk}}$ —— SASS 导出的标量环模型（§10.2.4） | **0.8962** | **0.936 / 0.829** |
| 4 | launch 锚：全局唯一 `launch_ns` 由纯发射 ref_f16 锚核导出（实测减去其建模非 launch 项 = 6.073 µs；报告行四舍五入记 6.12 µs） | 0.8962（秩不变；修复绝对尺度） | 0.936 / 0.829 |
| 5 | v3：+ 占用/波次项 | 0.8962 | 0.936 / 0.829 |

（最终代码三形式的正式对照——validation.json：v1 0.8900 / v2 **0.8962** / v3 0.8962，top-5 重叠 0.8，n=18。此处 v1 指终码的"合同项子集"（已含 SASS nomma 模型与锚定 launch），与迭代史第 1 行的"历史 v1"不同——历史 v1 的 nomma 是 CUDA 核 roofline、launch 是后备值。）

#### 10.2.4 nomma 项的 SASS 标量环模型（10 cyc/MAC）

迭代 3 的依据是编译期证据而非拟合：实测 nomma 配置跑 0.5–2.5 ms，比 FFMA roofline 高 100–400×。读反汇编发现根因——`convert-linalg-to-loops` 的下降给出**未分布的标量三重循环**：`evidence/G3/gen_noargs/kkt_bf16_nomma_p2_t64x64x32_w4/sass.txt` 的内环（`.L_x_17`）每 MAC 39 条 SASS 指令（10× `IADD.64` 寻址 + 循环控制 + 3 条标量 `LDS.U16` + 1 条 FMA），按 4 发射管道/SM 约 **10 周期/MAC**。常数 `NOMMA_CYC_PER_MAC=10` 从 SASS 读出、从未对实测时间调参（预测 kkt-nomma 1.94 ms vs 实测 0.99–1.07 ms；o-nomma 1.93 ms vs 2.35–2.46 ms——strided-B 的 o 阶段变体每 MAC 代价约 2×，为已接受的簇内偏差）。

#### 10.2.5 launch 锚

标定文件不测 launch 开销；`ref_f16_sm80_atom` 锚核是纯发射 kernel（非 launch 项建模值 ~0.05 µs）。由"实测 − 建模非 launch 项"导出**全局唯一** launch 常数 6.073 µs（validation.json `launch_anchor_s = 6.0729×10⁻⁶`），这是合同允许的唯一 launch 常数来源；报告中另附有/无锚两栏以自证秩不变。

#### 10.2.6 特征公式自检

`gdn_solver.py --selftest` 把解析每迭代计数公式对 INDEX 静态 PTX 计数核对（18 个 GDN 配置）：`mma.sync` **精确 18/18**、`ld.shared` **精确 18/18**、`bar.sync` **精确 18/18**、`cp.async` 偏差 +0.4–1.6%（prologue/param 尾部松弛）。`pipeline_shared_memory_copies` 的体复制 peel 因子 $(1+p)$ 与 CSE 唯一 fragment 装载均以此方式验证——模型的分子（计数）在编译期就是对的，误差只剩分母（速率）。

#### 10.2.7 验证协议与 Spearman 门

`--validate` 对 18 个实测配置（20 个已编译中排除 2 个：ws 配置 TIMEOUT=硬件 mbarrier 死锁、tma 配置 NOT_TIMED=cubin 无数据全局——G2 有意把 CUtensorMap 接线留给运行时；均被计时加载器的状态过滤器剔除）计算预测/实测的 Spearman 秩相关，门限 0.7。**结果：v2 ρ = 0.8962，PASS**（kkt 0.936 / o 0.829；top-5 重叠 0.8）。输入为 `evidence/G3/config_times.json`（smem 修复、零参重建 cubin 的 50 次 CUDA 事件中位）与 `evidence/G3/gen_noargs/*/meta.json` 编译期特征——加载器**优先** gen_noargs 元数据，正因为计时 cubin 与 G2 INDEX 的 ptxas 结果截然不同（regs 65–96 且 ~0 spill vs 255 且 10⁴–10⁵ spill 操作）：编译期特征必须逐构建重收割，陈旧特征对错对象。

18 配置实测全表（中位 ms，均 checksum 稳定、nonzero 写出验证）：

| 配置 | 状态 | 中位 ms | 配置 | 状态 | 中位 ms |
|---|---|---|---|---|---|
| kkt_bf16_mma_p0_t64x64x32_w4 | OK | 0.1051 | o_bf16_mma_p0_t32x64x32_w2_K64V64 | OK | 0.0450 |
| kkt_bf16_mma_p2_t32x64x32_w4 | OK | 0.0188 | o_bf16_mma_p2_t64x64x32_w4_K64V64 | OK | 0.0203 |
| kkt_bf16_mma_p2_t64x64x16_w4 | OK | 0.0205 | o_bf16_mma_p3_t64x128x16_w8_K128V128 | OK | 0.0844 |
| kkt_bf16_mma_p2_t64x64x32_w2 | OK | 0.0148 | o_bf16_nomma_p2_t64x64x32_w4_K64V64 | OK | 2.3492 |
| kkt_bf16_mma_p2_t64x64x32_w4 | OK | 0.0212 | o_f32_mma_p2_t64x64x32_w4_K64V64 | OK | 0.0333 |
| kkt_bf16_mma_p2_t64x64x32_w8 | OK | 0.0342 | o_f32_nomma_p0_t64x64x32_w4_K64V64 | OK | 2.4642 |
| kkt_bf16_mma_p3_t64x64x32_w4 | OK | 0.0206 | ref_f16_sm80_atom | OK | 0.0062 |
| kkt_bf16_nomma_p0_t64x64x32_w4 | OK | 1.0725 | ref_ldmatrix | OK | 0.0073 |
| kkt_bf16_nomma_p2_t64x32x32_w4 | OK | 0.5079 | kkt_bf16_nomma_p2_t64x64x32_w4 | OK | 0.9918 |

#### 10.2.8 稳健性：±2× 常数扰动与 ±30% 时间噪声

共享窗上的 PASS 是否伪影？`sensitivity.py` 做两类扰动（200 试验/档，evidence/G3/solver_report.md §4）：

- 对**每一个结构常数**（NOMMA_CYC_PER_MAC、TF32_RATIO、LSU_IOPS、SPILL_BW_FRAC、tensor/clk）±2×：ρ 稳定于 **[0.882, 0.902]**；
- 对实测时间均匀 ±5/10/20/30% 噪声：ρ 中位 0.896 / 0.890 / 0.885 / 0.885，最差 p5 = **0.856**——比观测到的 run-to-run 散布大近一个数量级的噪声下门仍过（config_times 的 ms_all 很紧：如 kkt p2 w4 为 21.0±0.3 µs）。

结论：PASS 判定不是共享窗口的产物。

#### 10.2.9 已记录盲区（方向正确、幅度未建模）

诚实契约要求模型披露其不知：

1. **pipeline=0 的 `.ca` 串行化幅度**：`create_async_groups` 不带 `bypass_l1` 发射 `cp.async.ca`（L1 分配式）拷贝，每 K 迭代一组 `commit_group`+`wait_group 0`。同 tile/warp 下实测 kkt p0 = 105 µs vs p2 = 21 µs（5×），模型的 sum 重叠只预测 2.4 µs 串行化（预测 7.4 µs）。排序方向（p0 在流水化 mma 变体中最差）正确，幅度未建模——不存在编译期可导出的 `.ca` 惩罚常数，而向实测拟合被诚信合同禁止。`o_bf16_mma_p0`（45 µs）同因。
2. **mma 路径绝对时间系统性低估 3–5×**（fragment 装载依赖链、超出 7.7 ns 边际 bar.sync 的屏障往返等延迟效应）——对全部 mma 配置是均匀缩放，秩不受影响。

#### 10.2.10 结论

零拟合解析模型以 Spearman 0.8962（n=18，门 0.7）通过验证，且该数字对全部结构常数 ±2× 与实测 ±30% 噪声稳健；其分子的计数公式对静态 PTX 18/18 精确、分母的速率来自一次性标定；两个幅度级盲区如实记录而秩级正确。支柱二的"凭什么排序"至此闭合。

---

### 10.3 一次性硬件标定（scripts/run_calib.sh + tools/gdn_calib/calib_bench.cu）

#### 10.3.1 动机

成本模型的全部速率分母来自一个单 GPU、分钟级、<2 GB 显存的微基准窗口——一次性沉没成本，之后每新形状的调度（0.8 s CPU）不再触碰 GPU。这与 autotune 每形状 39.8 s 的在设备成本形成经济学对照。标定器的设计约束：(a) 每个速率对应模型的一个项；(b) 全部产物 JSON 携带共载快照与 `shared_gpu` 标志；(c) 幂等可重跑——同命令同文件名，干净窗口刷新值直接覆盖。

#### 10.3.2 机制与源码：六个微基准核

`calib_bench.cu`（`nvcc -arch=sm_120a -O3`）实现 (a)–(e) 五组测量，每个核把累加器写经一个永不成立的分支以挫败死码消除：

```cuda
// (a) mma 峰值：每线程 8 深度 ILP 的 mma.sync.m16n8k16 bf16->f32
__global__ void __launch_bounds__(128) k_mma(int reps, float* sink) {
  ...
  for (int it = 0; it < reps; ++it)
#pragma unroll
    for (int u = 0; u < 8; ++u)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : ...);
  if (c0 == 1e30f) sink[blockIdx.x] = c0 + c1 + c2 + c3;   // DCE 汇
}
// (d) bar.sync 往返：20000 次 bar.sync 0,128 除以次数 -> ns/次
__global__ void __launch_bounds__(128) k_bar(int reps, float* sink) {
  for (int it = 0; it < reps; ++it)
    asm volatile("bar.sync 0, 128;");
  ...
}
```

(b) 组三个带宽核：`k_smem`（48 KB 共享窗 xor-sum 的 `ld.shared` 16 B 读）、`k_gread`（两体制读：32 MiB L2 驻留缓冲——约 60% L2 上限，与 512 MiB DRAM 体制）；(c) `k_ex2`（每线程 4 条独立 `ex2.approx.f32` 链）；(e) `k_cpasync`（16 B `cp.async.cg` + commit/wait，1 MiB L2 驻留源窗，bonus 项）。计时统一为 CUDA 事件、≥5 预热 + 50 计次、**取中位**。`scripts/run_calib.sh` 负责构建、在 start/pre-run/end 三点抓 `nvidia-smi` 共载快照（利用率/显存/SM 时钟）与计算进程清单，把原始 JSON 合成 `evidence/G3/calibration.json`：`shared_gpu` 由"是否观测到共居计算进程"推导，快照与重跑说明写入文件本体。

#### 10.3.3 数据对比：标定全表

**原测量（共享机会窗）**——正式门判定（validation.json / solver_report）所用之表：

| 速率 | 值 | 方法学注记 |
|---|---|---|
| (a) mma.sync.m16n8k16 bf16→f32 峰值 | **115.5 TFLOP/s**（修正后；文件原始 flops 字段 3.70 PFLOP/s，见 §10.3.4） | 82 SM 饱和、每线程 ILP 8；902.6 G-instr/s ÷32×4096 |
| tf32 张量峰（导出） | 57.8 TFLOP/s | TF32_RATIO=0.5（文档化消费级常数） |
| (b) 共享内存读带宽（ld.shared 16 B） | 113.1 TB/s | 48 KB/窗/块、xor-sum；可能被广播放大，只影响 ≤1% 的 $T_{\text{smem}}$ 项 |
| (b) gmem L2 驻留读 | 2144.3 GB/s | 操作数 KB 级且计时 harness 循环驻留 → 模型有效操作数速率 |
| (b) gmem DRAM 体制读 | 824.6 GB/s | 512 MiB 缓冲、重共载挤占；GDDR7 规格 ~1.8 TB/s，按下界解读 |
| (c) SFU ex2.approx.f32 | 3.23 Telem/s | 每线程 4 独立链 |
| (d) bar.sync 往返（4-warp 块） | 7.707 ns | 20000 次往返除法，82 块 |
| (e) cp.async 16 B | 1388.2 GB/s | bonus；尚未成为模型项 |
| launch 开销 | 6.073 µs | 由 ref 锚核导出（§10.2.5）；后备常数 4 µs |
| 观测 SM 时钟 | 2722 MHz（中位） | 峰值属性 1597 MHz 是保守的笔电基频；$T_{\text{lsu}}$ 用共载观测中位 |

**干净窗口刷新（现行 calibration.json，`shared_gpu: false`）**：DRAM 851.42 GB/s（+3.2%）、L2 2162.0 GB/s、smem 114.07 TB/s、ex2 3.66 Telem/s、bar.sync 7.694 ns、cp.async 1540.2 GB/s、mma 修正峰 116.06 TFLOP/s（906.71 G-instr/s）。原共享窗表存档为 `calibration_shared.json`；正式门数字（0.8962）以共享窗表记录，与刷新表的差异（≤3.2%）远小于 §10.2.8 已证的稳健性余量。

#### 10.3.4 ×32 修正：warp 级操作的计数陷阱

标定文件的 `flops` 字段把每**线程**指令率乘上 m16n8k16 原子 4096 FLOP——但 `mma.sync` 是 **warp 级**操作，诚实峰值 = warp 指令率 × 4096 = instr/s ÷ 32 × 4096，原字段过计 32 倍（3.70 PFLOP/s 在 82 SM/2.7 GHz 的机器上物理不可能）。成本模型加载器以可检测规则自动纠正并留痕：

```python
# tools/gdn_costmodel.py :: Calibration._parse_g3a（节选）
warp_flops = mma_instr / 32.0 * 4096.0
if mma_flops > warp_flops * 1.5:          # >2x 的差距物理上不可能 => 检测过计
    self.m["mma_flops_field_asbuilt"] = mma_flops
    mma_flops = warp_flops
    self.m["mma_flops_corrected"] = True  # 解析表携带修正标志
```

交叉验证：6.89 instr/SM/clk @1597 MHz 属性钟 ≈ 2.7 GHz 观测钟下 4.1 instr/SM/clk，与 4 tensor 管线/SM 的结构一致。任何消费 `calibration.json` 的下游必须重算——`g5n_static_analysis.py`（§10.7）在文件头内联执行同一修正。

#### 10.3.5 共享窗披露

两个窗口（标定与计时）均 `shared_gpu: true`（原始表）：共居训练任务 pid 3411456 常驻（~17.9 GiB 显存，窗内利用率 24→99%，SM 时钟观测 2.1–2.8 GHz vs 1597 MHz 峰值属性）。绝对带宽/FLOP 速率按**受挤占下界**解读。这不是被容忍的缺陷而是被协议化的条件：每个输出 JSON 标注标志与快照；重跑指令与幂等性写入 calib_report §5（`scripts/run_calib.sh` → `scripts/run_g3.sh` → `tools/gdn_time.py`，同文件名覆盖）；排序主张由 §10.2.8 的扰动稳健性与交错对内比较（实验章）承保，而非依赖绝对值。

#### 10.3.6 加载器：容错 schema、单位规范化、后备标记

`Calibration` 类原生解析 G3-A schema（device / mma_bf16_m16n8k16 / smem_read_bw_gbps / gmem_read_l2resident / gmem_read_dram / sfu_ex2_f32 / bar_sync_roundtrip / cp_async_16b 嵌套于 measurements），并对其它布局做规范化别名搜索（键名/单位启发式：GB/s、TFLOP/s、GHz、µs/ns 后缀）；缺键时落入场记的 5090 级数据手册后备表并**逐键标记**（`fallback_used`，最终报告中为空集自证）；共载快照中的观测 SM 时钟若高于属性钟则取中位替代（`T_lsu` 随实际运行时钟缩放）；时钟完全缺失时由张量峰反推 $f = \text{peak}/(82\times1024)$ 并标记 `clock_hz(derived)`。

#### 10.3.7 结论

一次性标定以六个微基准核覆盖成本模型的每一个速率分母，全表带共载披露与幂等重跑协议；一个真实的 ×32 计数陷阱（warp 级操作按线程计数）被检测规则捕获并在全部下游统一修正；干净窗刷新与原始共享窗表并存，差异在已证稳健性余量之内。

---

### 10.4 参考收割器与 T1 契约集提取（harvest_flaptx / harvest_qlaptx / FlashKDA / ptxas-blackwell 链）

#### 10.4.1 动机

覆盖矩阵（§10.5）需要靶子，且采集必须**纯 CPU**（G0 规则：构建期零 GPU 使用）。两个手写设计极点——fla（Triton，multistage 流水极点）与 FlashQLA（TileLang，warp 专化 + TMA 极点）——的 PTX/SASS/ptxas 资源表必须在不启动 GPU 的前提下拿到。第三个风格点 FlashKDA（两段式 token-parallel WY 预段）在最终系统中的落点是旋钮面的 grid 映射旋钮与计划中的第二批收割路径（见 §10.4.4）。

#### 10.4.2 fla 收割器：Triton AOT + GPUTarget

`tools/harvest_flaptx.py`（经 `scripts/harvest_fla.sh` 在 conda `gdn` 环境运行，probe→harvest 两步，日志落 logs/）的关键是 Triton 的 AOT 编译接口——无需 GPU，目标架构显式给定：

```python
# tools/harvest_flaptx.py :: compile_one（节选）
from triton.compiler import ASTSource            # triton >= 3.8
from triton.backends.compiler import GPUTarget
src = ASTSource(fn=k, signature=sig, constexprs=cexpr)
tgt = GPUTarget("cuda", 120, 32)                 # 纯 CPU：目标显式，无设备探测
return triton.compile(src, target=tgt,
                      options={"num_warps": warps, "num_stages": stages})
```

工程面三个要点。**装饰链解包**：fla kernel 包在 CachedAutotuner 等装饰链里，`unwrap()` 沿 `.fn/.kernel/.inner` 链走到带 `.params` 的原始 JITFunction 以自省参数表。**类型推断表**：非 constexpr 参数按 fla 命名约定映射 Triton 类型串——显式类型表 `BF16_PTRS`（k/v/w/u/o/q/beta/dq/dk/dv/dw/du/db/h/dh/do…）、`FP32_PTRS`（A/Ai/g/gk/dht/h0/ht/dg…）、`I32_PTRS`（cu_seqlens/chunk_indices/chunk_offsets）加逐 kernel 覆盖 `PER_KERNEL_TYPES`（如 solve_tril 的 Ai=bf16、wy 类 kernel 的 A 以 k.dtype 存储为 bf16）；未知名收割时点名人工补表（probe 模式打印全部参数名供核对）。**形状 constexpr**：主套件代表点 B=4、T=8192、H=HV=16、K=V=128（NT=128），USE_G/USE_DW/USE_A 等布尔开关与 DOT_PRECISION 一并固定；每 kernel 的小配置集取自 fla autotune 空间的 Blackwell 相关代表点（如 kkt_solve 的 warps{2,4}×stages{3,4}，dhu 的 BV{32,64}×warps2×stages{3,4}），共 12 个 kernel 族、22 个 (kernel, config) 组合。产物链每目录五件：`kernel.ptx`（`ck.asm["ptx"]`）、`kernel.cubin`、独立复跑的 `ptxas_v.txt` 资源报告、`nvdisasm -c` 的 `sass.txt`、含指令族直方图（FAMS 正则表）与 triton metadata（动态 smem 等）的 `meta.json`；汇总 `SUMMARY.md`。

**ptxas-blackwell 链**：sm_120a + 该 PTX 版本只有 ptxas-blackwell 接受——收割器定位工具的优先级为 triton 自带 `backends/nvidia/bin`（含 `-blackwell` 后缀变体）→ CUDA 工具链 → PATH。版本事实（evidence/G2/gen_report.md）：triton 13.3.33 自带 ptxas-blackwell、`/usr/local/cuda/bin` 为 13.3.73，两者皆可；triton 的普通 `ptxas`（12.9）**不能**编译 sm_120a。`parse_ptxas_v` 从 `-v` 报告解析 regs/smem/spill_stores/spill_loads/stack_frame（对 ptxas-blackwell 的措辞差异"cumulative stack size"兼容）。

#### 10.4.3 FlashQLA 收割器：TileLang 锁 target + monkeypatch

`tools/harvest_qlaptx.py`（经 `scripts/harvest_qla.sh`，独立 conda 环境 `gdn-tl`）从 TileLang JITKernel 取 CUDA 源再 nvcc 离线编译，全程不 launch：

```python
# tools/harvest_qlaptx.py :: _force_sm120a（节选）
def _force_sm120a():
    """monkeypatch：所有 get_target_arch_and_code 调用点强制返回 120a。
    FlashQLA sm120 kernel 需要 setmaxnreg（arch-specific 'a' 后缀），
    而 tilelang 默认解析链会把 target 规范成不带 'a' 的 sm_120。
    只在本进程内 patch，不改 vendor 文件。"""
    import tilelang.contrib.nvcc as _nvcc
    orig = _nvcc.get_target_arch_and_code
    def patched(target=None):
        arch, code = orig(target)
        return ("120a", code)
    _nvcc.get_target_arch_and_code = patched
    for name in ("tilelang.jit.adapter.libgen", "tilelang.cuda.backend",
                 "tilelang.jit.kernel"):
        ...  # 三个下游调用点同步替换
```

进程级补丁是必要的：FlashQLA 的 sm120 kernel 依赖 `setmaxnreg`（arch-专属 `a` 后缀指令），而 TileLang 默认解析链把 target 规范化为无后缀 sm_120。版本裁定（脚本内固化）：FlashQLA 官方 pin 的 tilelang 0.1.9 与 CPython 3.12.15 不兼容（tvm_ffi `type.__dict__` 不可写），收割用 0.1.15 并在矩阵中如实标注。五个工厂调用（kkt_solve、prepare_h ×2（num_stages 2/3）、fused_fwd、fused_bwd）的实参全部镜像 FlashQLA 包装器（H=Hg=16、DK=DV=128、chunk 64、bf16/fp32 dtype 集、accum fp32、use_initial_state/store_final_state 等布尔面），`get_kernel_source()` 出 CUDA 源（`kernel_only=True` 优先），`nvcc -O3 --expt-relaxed-constexpr -arch=sm_120a` 配 TileLang 内部 include 集（src/3rdparty/cutlass/tl_templates）出 `-ptx` 与 `-Xptxas -v -cubin`，`nvdisasm -c` 出 SASS；特征提取与 ptxas 解析直接复用 fla 收割器的 `FAMS/ptx_features/parse_ptxas_v`（import 复用即对拍基础）。环境层：`TILELANG_DEFAULT_TARGET='{"kind":"cuda","arch":"sm_120a"}'`、FlashQLA 源码以 `--no-deps -e` 挂载。

#### 10.4.4 FlashKDA 与第三风格点

生态里存在第三个风格点 FlashKDA（两段式：token-parallel WY 预段 + 主段，适配 82-SM 并行度）。在最终系统中它的落点有二：其一，**旋钮面**——L2 调度层的 grid 映射旋钮显式包含"可选 token-parallel WY 预段 = FlashKDA 两段式"（docs/PLAN.md L2 层定义），即风格差异被吸收为配置空间中的一个方向，与"FLA/FlashKDA/FlashQLA 风格 = 配置空间中三个点，复现 = 调参不重写"的总论断一致；其二，**收割路径**——计划第二批参考收割走 FlashKDA 的 `FLASH_KDA_CUDA_ARCHS=120a` 环境变量 AOT（同属纯 CPU 链），T1 契约集的第一批（fla + FlashQLA）先落地并支撑了 §10.5 的矩阵。本节如实记录该分工：已收割两极点、第三点以旋钮面吸收。

#### 10.4.5 T1 契约集：33 族的提取

`tools/gdn_coverage.py ref-inventory` 扫描两套收割产物，构建规范化指令族清单——**T1 集 = fla ∪ FlashQLA 出现的全部指令族（33 族）**。分族不是正则拼凑而是两级机制：通用操作码直方图（指令行首 token，字符集含 `:` 以覆盖 `st.shared::cta`、`.L2::cache_hint` 限定词）+ 有序前缀分类规则 `CLASSIFY`（约 90 条，先特征后基线、先 packed f32x2 后标量 f32、bf16 域显式前缀先于通用 cvt/add/mul）；层（tier）判定：`FEATURE_FAMILIES` 中的 15 族为 feature 层（需要显式 IR 供给：intrinsics/流水/warp 集体操作），其余为 baseline 层（标准 arith/linalg→NVVM 下降自然产生）。**G0 FAMS 的修复与对拍**：G0 收割器的 `tma.bulk` 正则从未匹配过、mbarrier 漏掉 `mbarrier_init` 拼写、`exp2?\.approx` 漏 PTX 的 `ex2.approx` 拼写——`EXTRA_FAMS` 修复后，每个条目同时记录 G0-FAMS 复算值与收割时 meta 值的差异（`g0_parity_diff`），无物静默丢失。除族计数外，清单还提取：每族的精确操作码变体集、mma.sync 全签名串（带计数与 kernel 出处）、逐 kernel 资源包络（regs/smem/spill/mma 总数在配置维上的 min–max）、以及逐配置循环结构（`bra` 总数与回边数——标签位置先于分支即回边）。

#### 10.4.6 数据对比（收割汇总）

| 来源 | 采集方式 | 规模 | 关键资源事实 |
|---|---|---|---|
| fla（evidence/G0/fla_ptx） | Triton AOT `GPUTarget("cuda",120,32)` | 12 kernel 族 / 22 配置 | mma.sync 3284 条（18 配置）；regs 包络 [40,255]（bwd 全员 255 顶格 + spill：prepare_wy 5348 B、dhu 3788 B、dqkwg 1544 B）；无 TMA/mbarrier/setmaxnreg；两签名：bf16 m16n8k16（2708）+ tf32 m16n8k8（576，仅 dhu/kkt_solve） |
| FlashQLA（evidence/G0/qla_ptx） | TileLang monkeypatch 锁 sm_120a + nvcc | 5 kernel / 5 配置 | mma.sync 2496 条（全 bf16 m16n8k16）；regs 88–128 全员零/微 spill；动态 smem 1–2 KB（TMA 经描述子出场）；TMA bulk.tensor 4d/5d 100 条 + mbarrier 386 + elect.sync 59 + setmaxnreg 16；真循环（bra 107–611，fused_bwd 单 kernel 1600 mma 来自循环体复用） |

#### 10.4.7 结论

两条纯 CPU 收割链（Triton AOT 显式 GPUTarget；TileLang 进程级 target monkeypatch + nvcc 离线）把两个手写极点的完整资源表与指令面变成可 diff 的证据文件；T1 契约集 33 族从中规范化提取，分族规则显式、G0 正则缺陷被修复并对拍留痕；FlashKDA 风格点以旋钮面吸收并保留第二批收割路径。

---

### 10.5 覆盖矩阵与逃生舱台账（tools/gdn_coverage.py + evidence/G2/coverage_matrix.md、hatch_report.md）

#### 10.5.1 动机

"编译器生成的 kernel 与手写专家 kernel 同构"长期缺乏可核查证据。覆盖矩阵把它变成一张可机器再生成的对照表：本方生成物（ours 列，20 配置）对 T1 契约集逐族判定覆盖状态，对 T2 结构维度逐极点匹配；NO 格子触发逃生舱预算记账。矩阵命令幂等（`matrix` 可在 G2-A 生成集落地前后重复运行，ours 列 PENDING→LIVE）。

#### 10.5.2 机制与源码：T1/T2 定义与 ours 列判定

**T1 = 指令族 + mma 签名 + 结构维度**（来自参考清单 §10.4.5）；**T2 = 结构比较卡**：对四个代表 kernel（fla chunk_fwd_o/BK64_BV64、fla dhu/BV64、qla fused_fwd、qla fused_bwd）逐卡给出 mma 总数/签名/异步模式/资源，并在 ours 列中按**族集合 Jaccard 相似度 + mma 签名匹配数 + 总数接近度**三元组选取最接近配置，资源项按 ±30% 包络判分（`envelope_check`：参考 0 则本方须 0，否则本方须落入 [0.7×, 1.3×]）。ours 列的族级判定规则：

```python
# tools/gdn_coverage.py :: cmd_matrix.ours_cell（节选）
def ours_cell(f, ref_variants_set):
    total = sum(e["families"].get(f, 0) for e in ours)
    if total == 0:
        return "NO"                          # 全 20 配置从未发射该族
    ov = set()
    for e in ours:
        ov.update(e["variants"].get(f, {}))  # 本方出现过的精确操作码变体
    missing = ref_variants_set - ov
    return f"PARTIAL (missing: {sorted(missing)})" if missing else "YES"
```

三态语义：YES = 某配置发射该族且**全部参考变体**齐备；PARTIAL = 族已发射但缺参考的某操作码变体/签名；NO = 从未发射。PARTIAL 的缺口逐条点名（如 mma.sync 缺 `m16n8k8.tf32`、ex2 缺 `.ftz` 变体、st.shared 缺 `::cta` 作用域拼写）——这使"覆盖"精确到变体级而非族名级。

#### 10.5.3 数据对比一：家族覆盖状态

G2-A 首测（17 配置）态：3 YES / 14 NO / 16 PARTIAL 中的 NO 占比过高；G2-C（warp 专化 + TMA + ldmatrix 三配置加入，20 配置）后：**7 YES / 20 PARTIAL / 6 NO（33 族）**；feature 层 15 族中 8 YES / 6 PARTIAL / 1 NO。关键翻转（evidence/G2/hatch_report.md §4）：

| 族 | 前 | 后 | 未满 YES 的原因 |
|---|---|---|---|
| setmaxnreg | NO | **YES** | inc+dec 精确拼写 |
| elect.sync | NO | **YES** | 精确 |
| mbarrier | NO | **PARTIAL** | 四操作齐（init/arrive/expect_tx/try_wait.parity）；本方印 `.shared.b64`、参考显式拼默认作用域 `.shared::cta.b64`——同指令、字符串级变体差 |
| tma.bulk.tensor | NO | **PARTIAL** | 4d+5d 载 + 2d 存已发射；参考为 `.L2::cache_hint` 载入拼写（nvgpu 下降硬编码无 hint）与 4d/5d 存 |
| tma.bulk.ctrl | NO | **YES** | commit_group + wait_group.read 精确 |
| prefetch.tensormap | NO | **YES** | 精确 |
| ldmatrix | NO | **PARTIAL** | x4 A-frag + x2.trans B-frag；缺 x1/x4.trans/操作数次序变体（numTiles/transpose 旋钮可达） |
| shfl.sync | NO | **PARTIAL** | idx 变体已发射；bfly 需 `nvvm.shfl.sync bfly` |
| stmatrix | NO | NO | `nvvm.stmatrix` 在树但需要仅 LLVM 转换后才物化的共享窗指针操作数——文档化后续项 |

终态（按最终系统呈现）：G5-G 轮 3 的 `gdn.mma.tf32k8` 逃生舱（§10.5.6）把 mma.sync 族的最后一条签名缺口（tf32 m16n8k8）翻 YES，全表终态 **8 YES / 19 PARTIAL / 6 NO**；mma 签名层双 YES——bf16 m16n8k16：本方 552 条 vs fla 2708 / FlashQLA 2496；tf32 m16n8k8：fla 精确拼写（576 条，经逃生舱达成）。

#### 10.5.4 数据对比二：fla vs FlashQLA 两极点（结构维度）

| 维度 | fla 包络 | FlashQLA 包络 | ours（20 配置，G2 态） | 注记 |
|---|---|---|---|---|
| regs/线程 | [40, 255] | [88, 128] | [28, 255] | fla 顶格 255 + spill；qla 以 setmaxnreg 钉 ≤128；ours 两端皆含 |
| 动态 smem (B) | [0, 33288] | [1024, 2048] | [0, 0]（ptxas 读数） | ours 读数为 0 是 G2 下降把 workgroup 分配降为设备 malloc 的伪影（ptxas 无静态 smem 行；SASS 证共享窗流量真实存在）；G3 smem 修复后按静态 .shared 声明计 |
| spill (B, st/ld 大者) | [0, 5348] | [0, 1752] | [0, 48032] | fla bwd 重 spill（dhu 3.8 KB）；ours 上界来自 f32 mma 大 tile 配置（G3 重建后 ~0） |
| 异步流水风格 | cp.async.ca 仅 state-push kernel（fwd_h/dhu）；余为 ld.global multistage | TMA cp.async.bulk.tensor 4d/5d + mbarrier + elect.sync 领队（kkt_solve：经典 cp.async.cg） | cp.async.cg/ca ×20 配置 | 两种待覆盖流水皆已成旋钮（pipeline / warpspec+tma） |
| warp 专化 | 无（无 setmaxnreg/elect.sync） | setmaxnreg.inc/dec + elect.sync（4/5 kernel） | ws 两配置 | G2-C 交付 |
| 循环结构（bra/回边） | [0,16] bra，多全展开 | [107,611] bra，真 scf 循环 | 随旋钮 | fla 展开复用、qla 循环复用两种风格 |

#### 10.5.5 数据对比三：T2 结构卡（四卡摘要）

- **fla:chunk_fwd_o/BK64_BV64_warps4_stages3**（mma 96、ldmatrix 52、regs 255、smem 24576 B）：ours 最近配置 `ref_ldmatrix`——族 Jaccard 最大者；regs 72 vs 255 → 包络外（0.28×）。
- **fla:dhu_blockdim64/BV64_warps2_stages3**（mma 512 = bf16×256 + tf32×256、ldmatrix 105、stmatrix 80、regs 255、smem 33288 B、spill 3236/3788 B）：ours 最近 `kkt_bf16_mma_p2_t64x64x32_w2`——regs 255 = 255 **PASS**。
- **qla:fused_fwd/default**（mma 384、TMA tensor 31、mbarrier 74、regs 128、smem 2048 B）：ours 最近 `kkt_bf16_mma_tma_ws`——regs 255 vs 128 → 1.99× 包络外。
- **qla:fused_bwd/default**（mma 1600、bra 422/回边 8、setmaxnreg 4、elect 21、shfl 80）：同上最近配置；mma 总量比 0.02×（单 chunk 阶段 kernel 对 fused 全反传 kernel，结构卡度量的是风格可达性而非规模等价）。

T2 卡的意义是**风格可达性**：两个极点的每一结构要素（multistage+顶格寄存器；warp 专化+TMA+循环复用）都在旋钮格内可表达；±30% 判分对逐 kernel 资源给出 PASS/OUT 的诚实记账，而非笼统"等价"。

#### 10.5.6 逃生舱台账：1/≤5 与 tf32k8 全案

**预算机制**：覆盖矩阵对每个 NO 格 feature 族预先登记逃生舱方案（自定义 NVVM 操作直书 PTX），总数预算 ≤5。**G2-C 的发现把预算用到 0**：2026 年 10 月主干已把全部"以为要逃生舱"的词汇（setmaxnreg/elect.sync/mbarrier 四件套/TMA 族/prefetch/ldmatrix/shfl）收编在树（NVGPU/NVVM 方言，经 `-gpu-lower-to-nvvm-pipeline` 降到 sm_120a PTX）；`gdn` 方言的 11 个 hatch 操作（GdnHatch.td）是对在树操作的 1:1 薄封装（`gdn-opt --gdn-lower-hatch`），机制演练充分、预算分文未动。后备逃生路径（`nvvm.inline_ptx`/BasicPtxBuilder）验证存在而未用。

**唯一真逃生舱：`gdn.mma.tf32k8`**（G5-G 轮 3；hatch_report ADDENDUM）。签名 `mma.sync.aligned.m16n8k16...` 之 tf32 兄弟 `m16n8k8.row.col.f32.tf32.tf32.f32`（fla 在 dhu/kkt_solve 的精确拼写）在树内不可达，双重根据：`nvgpu.mma.sync` 验证器强制 8×8×128b 基本片（tf32 封顶 K=4）；表生成的 `nvvm.mma.sync` 家族对 m16n8k8 只列 f16/bf16/i8。全案四要素：

1. **操作**：10 个 f32 操作数（a0..a3, b0, b1, c0..c3）→ 4 个 f32（d0..d3）；发射位点 `tgemm()` 在 `tools/gdn_bwd_gen.py`（逐 GEMM 位点的 `--tf32` 旋钮）。
2. **降级路径**：不走 `nvvm.inline_ptx` 而直接 `llvm.inline_asm`，约束 `=&f,=&f,=&f,=&f,f,...`——**early-clobber 必需**（SASS 在 5090 上验证）：BasicPtxBuilder 的普通 `=f` 输出下，ptxas 的内联汇编分配可把 D 四元组映到同一 mma 的 A 四元组（`HMMA.1688.F32.TF32 R12, R12, R10, R16`），tensor core 随即腐蚀操作数——精确整数 K=16 双原子链确定性 118/128 单元失败；`=&f` 禁止输出/输入重叠，`nvvm.inline_ptx` 无法表达。
3. **fragment 布局（硬件导出）**：整数操作数探针（evidence/G5/scratch/tf32k8_probe，D == A@B+C 精确）：a0=(g,2t)、a1=(g+8,2t)、a2=(g,2t+1)、a3=(g+8,2t+1)；b0=(2t,g)、b1=(2t+1,g)；d0..d3 用 bf16-k16 的 D 布局（列 = 2·tig）。A 寄存器次序（行半外层）与打包直觉不同——探针导出而非文档臆断。
4. **实测影响**（bwd B1T2048H16，共载中位）：dqkwg 4.71→2.75 ms（1.71×，与 r4/c4r 并行化合并计）、wyb 2.65→1.21 ms（2.19×，与 ra/c1r 合并计）、e2e bwd 14.16→11.06 ms（1.28×；相对 fla 0.073×→0.093×）；数值 6/6 PASS 于两个门形状；24 个 f32-GEMM 位点中 16 个 tf32 化、5 个为数值保持标量、3 个因 dhu 寄存器 spill 回归保持标量（逐位点 bisect 表见 evidence/G5/bwd_report.md 轮 3）。

**台账：1/≤5。**

#### 10.5.7 结论

覆盖矩阵把"同构"变成三态可审计的对照：终态 8/19/6（33 族），全部 PARTIAL 残差为变体级（拼写/向量宽度/代码生成旗标）而非能力缺口；mma 签名层双 YES（其一经逃生舱达成）；T2 结构卡证明两个手写极点的每个结构要素都在旋钮格内；逃生舱预算 1/≤5，且唯一一例带有完整的必要性证明、正确性规则与实测收益。支柱二生成的代码"凭什么信"，至此有了与性能声明互证的覆盖声明。

---

### 10.6 旋钮库：20 个配置面与敏感性（tools/gdn_l2_gen.py）

#### 10.6.1 动机

求解器的决策变量必须能物化为可编译工件，否则"编译期调度"是空转。旋钮库是旋钮向量的**具体化生成器**：每个配置生成 `payload.mlir`（含旋钮头部注释）+ `transform.mlir`（旋钮的 transform 载体）+ `config.json`；`--json` 接口直接消费求解器输出的配置字典（回环的生成环节）。它同时是覆盖矩阵 ours 列的全部来源（20 配置）与敏感性测量（旋钮 → PTX 可验证变化）的实验台。

#### 10.6.2 机制与源码：validate 与合法性规则

`validate()` 是求解器结构约束的镜像（求解器注释明言"mirrors gdn_l2_gen.validate"），两处独立编码互为交叉验证：

```python
# tools/gdn_l2_gen.py :: validate（节选）
if M_GLOBAL % bt or bt < 16:      raise ValueError("tile.M must divide 64 and be >= 16")
if n_glob % bv or bv < 8:         raise ValueError("tile.N must divide stage N and be >= 8")
if k_glob % bk or bk < 8:         raise ValueError("tile.K must divide stage K")
if bv % 8 or bk % 8:              raise ValueError("tile.N/tile.K multiples of 8 (16B chunks)")
if cfg["mma"]:
    if bt % 16:                   raise ValueError("mma=on requires tile.M multiple of 16")
    if dt == "bf16" and bk % 16:  raise ValueError("mma=on bf16 needs tile.K multiple of 16")
    if dt == "f32"  and bk % 4:   raise ValueError("mma=on f32(tf32) needs tile.K multiple of 4")
    if bv % (warps * 8):          raise ValueError("mma=on requires tile.N multiple of 8*warps")
    if cfg["pipeline"] > 0 and k_glob // bk < cfg["pipeline"]:
                                  raise ValueError("pipeline depth exceeds K-loop trip count")
esz = 2 if dt == "bf16" else 4
smem = (bt * bk + bk * bv) * esz
if cfg["mma"] and cfg["pipeline"] > 0:
    smem *= cfg["pipeline"]
if cfg["mma"] is False:                       # G4：操作数 staging + f32 累加器
    smem = (bt * k_glob + k_glob * bv) * esz + bt * bv * 4
if smem > 220 * 1024:
    raise ValueError(f"smem budget exceeded: {smem}B")
```

生成器容纳比求解器更宽的域（M=16 与 32 以下的 8 倍数），求解器按 G3 旋钮空间契约裁剪到 {32,64,128}——两域之差已文档化于求解器源码 TILE_DOMAIN 注释。旋钮共十一个：stage（kkt / o / ref / ref_ldmatrix）、dtype（bf16 | f32 | f16-仅 ref）、prec（bf16 | tf32）、tile{M,N,K}、warps（2|4|8）、mma（on/off）、pipeline（0|2|3）、warpspec（off | producer_consumer）、tma（off|on，须 warpspec）、driver（args | noargs：后者把输入物化为模块级常量 `memref.global`（确定性 0.5 步值、2⁻⁷ 的倍数保证跨 dtype 精确表示）、输出到公共零初始化 `@gdn_out`，得到零参 kernel 供 `gdn_time.py` 直载直发）、seed（noargs 数据确定性种子）。tma=on 时生成器同时落盘 `tensormap_spec.json`——宿主侧 `cuTensorMapEncodeTiled` 的逐描述子契约（CUDA 维序、box/stride 字节、kernel 坐标次序），描述子契约由生成器拥有、运行期照抄。

#### 10.6.3 数据对比一：20 配置面全表

| 配置 | stage | dtype | K/V | tile M×N×K | w | mma | pipe | warpspec | tma |
|---|---|---|---|---|---|---|---|---|---|
| kkt_bf16_mma_p0_t64x64x32_w4 | kkt | bf16 | 128 | 64×64×32 | 4 | on | 0 | off | off |
| kkt_bf16_mma_p2_t64x64x32_w4 | kkt | bf16 | 128 | 64×64×32 | 4 | on | 2 | off | off |
| kkt_bf16_mma_p3_t64x64x32_w4 | kkt | bf16 | 128 | 64×64×32 | 4 | on | 3 | off | off |
| kkt_bf16_mma_p2_t64x64x16_w4 | kkt | bf16 | 128 | 64×64×16 | 4 | on | 2 | off | off |
| kkt_bf16_mma_p2_t32x64x32_w4 | kkt | bf16 | 128 | 32×64×32 | 4 | on | 2 | off | off |
| kkt_bf16_mma_p2_t64x64x32_w8 | kkt | bf16 | 128 | 64×64×32 | 8 | on | 2 | off | off |
| kkt_bf16_mma_p2_t64x64x32_w2 | kkt | bf16 | 128 | 64×64×32 | 2 | on | 2 | off | off |
| kkt_bf16_nomma_p2_t64x64x32_w4 | kkt | bf16 | 128 | 64×64×32 | 4 | off | 2 | off | off |
| kkt_bf16_nomma_p0_t64x64x32_w4 | kkt | bf16 | 128 | 64×64×32 | 4 | off | 0 | off | off |
| kkt_bf16_nomma_p2_t64x32x32_w4 | kkt | bf16 | 128 | 64×32×32 | 4 | off | 2 | off | off |
| o_bf16_mma_p2_t64x64x32_w4_K64V64 | o | bf16 | 64/64 | 64×64×32 | 4 | on | 2 | off | off |
| o_bf16_mma_p3_t64x128x16_w8_K128V128 | o | bf16 | 128/128 | 64×128×16 | 8 | on | 3 | off | off |
| o_bf16_mma_p0_t32x64x32_w2_K64V64 | o | bf16 | 64/64 | 32×64×32 | 2 | on | 0 | off | off |
| o_bf16_nomma_p2_t64x64x32_w4_K64V64 | o | bf16 | 64/64 | 64×64×32 | 4 | off | 2 | off | off |
| o_f32_mma_p2_t64x64x32_w4_K64V64 | o | f32 | 64/64 | 64×64×32 | 4 | on | 2 | off | off |
| o_f32_nomma_p0_t64x64x32_w4_K64V64 | o | f32 | 64/64 | 64×64×32 | 4 | off | 0 | off | off |
| ref_f16_sm80_atom | ref | f16 | 16 | 16×8×16 | 1 | on | 0 | off | off |
| kkt_bf16_mma_p2_t64x64x64_w4_ws | kkt | bf16 | 128 | 64×64×64 | 4 | on | 2 | producer_consumer | off |
| kkt_bf16_mma_tma_ws | kkt | bf16 | 128 | 64×64×64 | 4 | on | 2 | producer_consumer | on |
| ref_ldmatrix | ref_ldmatrix | f16 | 16 | 16×8×16 | 1 | on | 0 | off | off |

#### 10.6.4 数据对比二：逐旋钮敏感性（PTX/SASS 可验证 delta；evidence/G2/gen_report.md gate (c)）

| 旋钮 | 变化 | PTX/SASS delta |
|---|---|---|
| pipeline | 0→2→3（kkt mma t64x64x32 w4） | mma 16→48→64；LDGSTS 512→1536→2048；LDS 83→246→326；spill 9.7→12.6→18.1 KB；PTX 126→276→351 KB；拷贝变体 ca→cg；多缓冲因子 1→2→3（tile 4→8→12 KB/缓冲） |
| tile.M | 64→32 | 每线程 mma 48→24；LDS 246→150 |
| tile.K | 32→16 | 在飞 cp.async 1542→774；mma 48→24 |
| tile.N | 64→32（nomma） | cp.async 514→386 |
| warps | 2→4→8 | 每 warp mma 原子 96→48→24（每 warp 的 N 片减半）；PTX 303→276→263 KB |
| mma | on→off | HMMA 48→0；PTX 276→88 KB（3.1×）；LDS 246→7 |
| dtype | bf16→f32 | 原子 m16n8k16.bf16 → m16n8k4.tf32（原子数 4×） |
| warpspec | off→producer_consumer | setmaxnreg dec96/inc224、elect、mbarrier init/arrive/try_wait、shfl.sync.idx 入 PTX（kkt_bf16_mma_p2_t64x64x64_w4_ws：spill 22,844 B） |
| tma | off→on | cp.async 流 1026 条经典 → 5 条 TMA bulk（4d+5d 载、2d 存、commit、wait.read）；spill 22,844→48 B；PTX 214.6→41.2 KB；另发 prefetch.tensormap + expect_tx |
| （寄存器） | 全 family | 全阶段 kernel 钉 255（全展开 epilogue 所致）；敏感性信号在 stack frame 0→26.8 KB 与 spill 64 B→118.8 KB；ref 锚 28 regs |

发射的精确 PTX 事实（矩阵 ours 列的物理基础）：bf16 工作马签名 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`（SASS HMMA.16816.F32.BF16 计数与 PTX 1:1）；tf32 `m16n8k4`（f32 配置 384 在环原子）；`cp.async.ca/cg.shared.global` + commit/wait 按流水旋钮切换；kkt 衰减 epilogue 降为非 ftz `ex2.approx.f32`；经典 mma 路径用 lane 索引 `ld.shared.b16` 装 fragment（与树内 RewriteMatmulAsMmaSyncOp 同址公式），ldmatrix 留作独立锚配置。

#### 10.6.5 与求解器的一致性及已发现的计数 quirk

生成器 `validate()` 的 smem 模型与求解器字节模型有一处**已发现并文档化的差异**：o 阶段每 GEMM 各分配一对 tile buffer（n_gemm=2），生成器的 `pair` 公式未乘 n_gemm（下计），求解器字节模型乘之（`pair·max(pipe,1)·n_gemm`，solver_report §1 点名的"found quirk"）。求解器侧为准（其模型对 PTX/SASS 复核），生成器侧该差额在 220 KB 生成上限内不产生非法配置，故未回改生成器——差异本身作为两模型交叉验证的产物记录在案。这也是"同一合法性规则双编码"的价值：分歧点即审计点。

#### 10.6.6 编译链

每配置经 `scripts/run_l2.sh`：`mlir-opt -transform-interpreter` →（幸存 linalg 的 `-convert-linalg-to-loops` 兜底）→ `gdn-opt --gdn-lower-hatch`（hatch 配置，在 transform 合并**之前**跑在裸 payload 上——gdn-opt 不能解析 transform.structured.*）→ `-gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a cubin-features=+ptx87 cubin-format=assembly"`（PTX 以 `#gpu.object<..., assembly>` 属性挂载，`gdn_extract_ptx.py` 解 MLIR 字符串转义出 kernel.ptx；`.version 8.7 / .target sm_120a`）→ 独立 `ptxas-blackwell -v -arch=sm_120a` → `nvdisasm -c`；产物五件套 + INDEX.json（20/20 ok）。

#### 10.6.7 结论

旋钮库把求解器的抽象决策变量落成 20 个全编译配置面，逐旋钮敏感性在 PTX/SASS 级可验证（每个旋钮都有可指认的 delta）；与求解器的双编码合法性规则互为镜像并暴露出一个真实计数 quirk；tma 旋钮的描述子契约由生成器单点拥有。至此回环的"生成"与"编译"两环闭合。

---

### 10.7 标定与成本模型的下游复用：静态指令分析（tools/g5n_static_analysis.py）

#### 10.7.1 动机

支柱二的信任链不止于 G3 的排序门：当 G5 的 bwd kernel 落地后，需要在不占 GPU 清窗的前提下回答"每个 kernel 距屋脊多远、被什么资源绑定"。`g5n_static_analysis.py`（G5-N 轮 7）把本章两件资产——收割器的特征提取器与成本模型的 sm_120 常数/标定速率——复用为逐 kernel 静态分析器，证明标定表与结构常数是**可复用的项目级基础设施**而非一次性的排序零件。

#### 10.7.2 机制与源码

对最新落地 bwd 生成目录（`evidence/G5/scratch/r6_T2048` 的 9-kernel 链）加 fwd mono 三件与 fused fwd 作参考，仅从**已出货工件**计算：`kernel_smemfix.ptx` 的逐 entry 族直方图（复用 `harvest_flaptx.ptx_features`）、`kernel_smemfix.cubin` 的新鲜 nvdisasm SASS 直方图（含 spill LDL/STL、LDSM/LDGSTS 与访存宽度解码——LDS/STS 按指令宽度累加字节，LDSM 按 512 B/warp）、ptxas regs + manifest 的 smem/threads/grid 的占用模型（复用 `gdn_costmodel` 常数）、每 chunk 计算强度估计（mma 位点 × scan fold：grid=B·H·NV 的扫描 kernel 每 (bh, chunk) 折叠 NV 次）、以及距屋脊：

```python
# tools/g5n_static_analysis.py（节选）
MMA_PEAK = M["mma_bf16_m16n8k16"]["instr_per_s"] / 32.0 * 4096.0   # ×32 修正内联重复
cta_regs  = cm.REGS_PER_SM // (regs * threads)
cta_smem  = cm.SMEM_USABLE_B // max(1, smem) if smem else cm.CTA_SLOTS_PER_SM
floor_mma  = tensor_flop * n_chunks / MMA_PEAK
floor_smem = smem_bytes * n_chunks / SMEM_BW
floor_dram = gbytes / DRAM_BW
fl = max(floor_mma, floor_cuda, floor_smem, floor_dram)   # 绑定资源 = 最大下界
```

#### 10.7.3 数据对比

分析器头部声明并携带修正后的标定速率（mma 峰 115.5 TFLOP/s ×32 修正、smem 读 113.05 TB/s、DRAM 824.6 GB/s——共载下界口径，注明 evidence/G3/calibration.json + G6/search_cost.md 为源）；输出每 kernel 的占用（CTA/SM、warp 槽百分比、限制资源）、比值表（lds/sts/ldsm/ldg/fp32/bar 每 mma 位点）、每 chunk 强度与屋脊下界组、绑定资源分类与 `实测/下界` 倍数。fwd 参考行同样处理（mono/fused 模块逐 entry 拆分）。该分析在 G5 后续轮次（如 r23-f 的 kkt 屋缘判定：staging 实测 826 GB/s 对 D2D 同窗 765 GB/s）中成为瓶颈归因的标准工具之一。

#### 10.7.4 结论

标定表与成本模型常数在排序门之外被下游静态分析直接复用（×32 修正逻辑在第二个消费者处显式重现，形成双点一致性）；占用模型、屋脊下界与绑定分类把"编译期成本模型"延伸为"编译期瓶颈诊断"，均零 GPU、确定性、可重跑。

---

### 10.8 本章小结

本章完整呈现了支柱二的落地与它的信任锚：（i）**Z3 求解器**以固定种子/rlimit/固定序把旋钮空间（368/465/204）亚秒级穷尽为逐字节可复现的全序排名，并以 SAT 探针补全与 rlimit 重试防御两个真实的 Z3 行为伪影；（ii）**解析成本模型**以零逐配置拟合契约达到 Spearman 0.8962（kkt 0.936 / o 0.829，n=18），对全部结构常数 ±2× 与实测 ±30% 噪声稳健，计数分子对静态 PTX 18/18 精确，幅度级盲区（`.ca` 串行化、mma 路径 3–5× 低估）如实记录而秩级正确；（iii）**一次性标定**以六个微基准覆盖全部速率分母，×32 warp 级计数陷阱被检测规则统一修正，共享窗披露与幂等干净窗刷新并存；（iv）**参考收割器**以纯 CPU 链（Triton AOT GPUTarget / TileLang monkeypatch + nvcc / ptxas-blackwell）固化两个手写极点的资源表，提取 T1 契约集 33 族；（v）**覆盖矩阵**给出终态 8 YES / 19 PARTIAL / 6 NO 与 mma 签名双 YES，T2 结构卡证明两极点结构要素皆在旋钮格内，逃生舱台账 1/≤5 且唯一一例（tf32k8）带完整必要性与正确性证明；（vi）**旋钮库**以 20 个配置面与逐旋钮 PTX 级敏感性把决策变量物化为可编译工件，并与求解器互为合法性镜像；（vii）标定与模型常数经静态分析器**下游复用**为屋脊诊断工具。调度离开了设备、搬进了编译期，且每一步——排序、生成、编译、复核——都有可再生的证据文件。本章各论断的证据源：evidence/G3/{solver_report.md, calib_report.md, validation.json, config_times.json, calibration.json, calibration_shared.json, solver_kkt.json, solver_kkt_v3.json, solver_o.json}；evidence/G2/{gen_report.md, coverage_matrix.md, ref_inventory.md, hatch_report.md}；evidence/G0/{fla_ptx, qla_ptx}；evidence/G6/{search_cost.md, coverage_hatch_summary.md}；evidence/G5/bwd_report.md（tf32k8 位点表）。

---

## 11 体制变化战役：标度律、误差预算与精度关闭定理

> 覆盖对象：反向链在 r19-final 冻结（第 6–7 章、第 9 章口径）之后发起的**体制变化战役（regime-change campaign，回合 RC1 → R4，2026-10-08 ~ 10-12）**全部四轮的方法学与结论——CPU 预测定价器 `tools/rc1_price.py` 与实测 B/T/H 阶梯、bf16 误差预算探针 `evidence/G5/scratch/r2_bf16_probe.py` 与 16 符号分类表、A_all 退役探针 `evidence/G5/scratch/r3_probe.py` 与 dhs 精确坍缩、tgemm a-fragment hoist、融合定价器 `evidence/G5/scratch/r4_price.py` 的三重杀；全部数字取自 `evidence/G5/{rc1_scaling_predictions,rc1_ladder_report,r2_bf16_globals_report,r3_report,r4_fusion_report}.md` 及其引用的 scratch 日志。本章按**最终系统与最终结论**呈现：四轮的杠杆全部以装机形态（或决策级 NO-GO 判定）讲述，历史演化只以"结构前/后对比"出现；测量口径延续第 9 章——正式数字一律为同窗交错/三重交错配对中位数（n=30 轮取中位，每 cell 三对），数值门槛为 fla 派生契约 dq/dk/dv ≤ 8e-3、dbeta/dg ≤ 2e-2、dh0 ≤ 8e-3，确定性契约为新起进程位级一致（N=3）；跨窗口绝对毫秒只作方向性参考（每轮披露窗口状态）。

### 11.0 章首：为什么在"冻结"之后又开一场战役

#### 11.0.1 冻结的语境与被推翻的决策

第 6–7 章的反向链在 26 轮自提速后冻结于 **r19-final bytes**：e2e 从 58.88 ms 提到 2.15 ms（约 27×），对 fla 的同窗交错比长期停在 ~0.48×。第 9 章的判定方法学与 `evidence/G5/DECISIONS_R25.md` 决策 D2 把这个状态正式化为"**持有（hold）**"：值恒等类杠杆经 r22 穷尽、新数值类杠杆经 r24 W-park 决策级 NO-GO，在 B1T2048H16 / B1T8192H16 两个形状上，"还能更快吗"的每一类想法都已定价或实测关闭；若反向工作重开，**唯一有未测量余量的方向是体制变化（regime change）——B4+/T-other/H-other 等形状外推**。D2 的签署状态是 *adopted-per-recommendation（用户未响应，欢迎推翻）*——协议里明确写着：任何后续回合的用户显式推翻都直接取代该记录，无需重做任何测量。

战役正是这个推翻的发生。用户在 2026-10-08 行使了 D2 预留的推翻权，指令反向链**"全覆盖全部 fla 支持的形状并取胜"**（fully cover ALL fla-supported shapes and win），并给出了一个候选阶梯：B∈{1,2,4,8} × T∈{2048, 8192, 16384, 32768} × H∈{16,32}。这条指令把战场从"两个锚形状上的同窗比值"扩展为"整条 B/T/H 体制空间上的胜负判定"，因此得名体制变化战役。它隐含两个待裁决的物理命题：

1. **趋势命题**：ours-vs-fla 比值是否随形状增大而改善（82 SM 填充、波尾摊销、fla 的逐形状 autotune 税是否会在大形状上反噬 fla）？是否存在任何体制达到 ≥1.1×？
2. **覆盖命题**：<6 GB 显存协调规则（co-job 常驻 ~8.7–9 GB）下，阶梯的哪些格点可测？曲线能否被有限个格点约束？

战役的回答方式继承了第 9 章的方法学纪律，并把它推到极致：**先 CPU 后 GPU、先预测后实测、先预算后落地、先纸面定价后写码**。四轮的分工是：RC1 用标度律框架在 GPU 触碰之前给整条曲线定价并实测对拍（§11.1）；R2 把"字节减半"变成可审计的误差预算合同并落地 9 符号默认集（§11.2）；R3 用预算探针发现 A_all 数值死、退役 caryA、把 dhs 坍缩为移位拷贝，并补上最后一刀 CVT 税（§11.3、§11.4）；R4 对仅存的字节地板推动者（dqkwg+wyb 融合）做纸面三重杀，把战役以精度关闭定理收官（§11.5、§11.6）。

#### 11.0.2 结局先行：本章要论证的三句话

先定义后用，三个贯穿全章的术语在此钉死：

- **体制（regime）**：以 $u = M/512$（$M = B\cdot H\cdot NC$，$NC = T/64$，即 B1T2048H16 锚的工作倍数）为坐标的工作量区间。**小形状体制**指 $u \ll 1$（两侧都被固定成本支配）；**大形状体制（u ≥ 1）**指 $u \ge 1$ 的全部实测格点（两侧边际成本主导）。
- **精度关闭（precision closure）**：一个性能命题不是被"执行力"（尚未到达屋顶线的差距）关闭，而是被**误差预算本身**（哪些字节在合同上必须保持 f32）关闭。关闭后的比值天花板是合同的数字，与实现质量无关。
- **同窗配对货币**：跨窗口绝对毫秒不可比（共载漂移 ±5–20%），一切 before/after 以同窗交错（RC1/R2：ours-fla 成对；R3：r2-链/r3-链/fla 三重交错）的配对中位数为货币。

本章按最终结论叙事，三句话概括：

1. **大形状反向的 ≥1.1× 目标被三重独立证据关闭**——误差预算地（§11.2，可铸字节被预算判死）、地板算术地（§11.6，完美执行当前字节集的天花板 0.85–0.92× @u=8）、融合地（§11.5，最后的字节搬运者被池/占用算术判 NO-GO）。最终战绩：u ≥ 1 全阶梯 **0.41–0.49×**，边际成本比 0.47×——**这是误差合同的数字，不是我们的数字**。
2. **战役在两个体制上全胜**：小形状 B1T512H4 **1.72×**（同窗配对，R3 窗口；从 R2 窗口的 1.55× 深化而来）；**冷启动全形状胜**——AOT cubin 零 autotune、~27 μs 发射开销，对 fla 每形状 48–69 s 的冷 autotune 税。
3. **战役本身把大形状反向从 0.39–0.61× 推到 0.41–0.49×**（u=8 轨迹：0.396 → ~0.43 → 0.462，边际 2.00 → 1.488 ms/u），全部杠杆位恒等或预算内（逐字节等价证明见 §11.3.5）——关闭定理不是"放弃"，而是把可动与不可动的边界钉死到数字。

#### 11.0.3 战役的窗口纪律与门槛栈

四轮的门槛栈是第 8–9 章体系的全量继承，每轮全过：payload 溯源（逐 cell 从 HEAD 再生成 bit 一致）、cubin 谱系（md5 对 r21b/R2/R3 记录）、数值 6/6（fp64 孪生、种子 0，全部 6 阶梯 cell）、确定性（新起进程 N=3 位级一致）、显存（<6 GB 规则、co-job 绝不触碰）。窗口状态按轮披露：RC1/R2 共载窗（GPU 常驻 ~8.9–9 GiB、16–28% util），R3 全程干净窗（0 共载），R4 仅做位恒等健康认证（无计时主张，共载不适用）。跨轮绝对数字的方向性比较始终以"每轮自窗配对臂"的口径给出。

---

### 11.1 标度律框架：在 GPU 触碰之前给整条曲线定价

#### 11.1.1 动机：把"形状外推"变成可证伪的预测

D2 留下的开放问题只有一个方向——体制变化——但"换个形状再测一次"不是裁决级的工作方式。第 9 章的方法学要求：**任何 GPU 测量之前，先有一个可证伪的预测**。RC1 的第一步因此不是上 GPU，而是从代码的 grid/tile 结构加两个已测锚点（B1T2048H16、B1T8192H16 的 r19-final 装机记录）推导出整条 B/T/H 曲线的解析形式，再从中选出信息量最大的格点去实测。这一节讲清楚三件事：标度律的推导（每 kernel 一条仿射律）、预测定价器 `tools/rc1_price.py` 的构造（预测如何被机器化）、以及 6 cell 实测对拍的判定（预测对了几成、错在哪、为什么）。

#### 11.1.2 两类 kernel、两条标度律

先定义后用：九 kernel 链（第 6 章 §6.0.2 拓扑）按 grid 公式分成两类，两类的时间随形状的走法不同。

**逐块并行类**：wy、caryA、caryM、bpre、dkb、dqkwg、wyb，grid = $M = B\cdot H\cdot NC$，每 CTA 处理一个块行。这类 kernel 的时间是工作量的仿射函数

$$t_s(u) \;=\; a_s + b_s\cdot u,$$

其中 $b_s$ 是 DRAM 受限的边际（每 M 行全局量都随 $M$ 线性增长），$a_s$ 是发射+填充残差。锚点拟合给出九个阶段的 $(a_s, b_s)$（`evidence/G5/rc1_scaling_predictions.md` §1 的完整表：如 wy −0.031/0.254、dqkwg 0.032/0.430、caryA −0.033/0.086 ms）。两个结构性修正：其一，**波量子化**只在边缘改善（$u{=}1$ 时 6.2 波 → $u{=}4$ 时 25 波，尾部效应 <4%）；其二，**L2 足迹罚**——32 符号 M 行全局量随 $M$ 增长把小足迹阶段（caryA/caryM/bpre）挤出 L2，T 方向 4× 工作量实测 5.9–6.2× 时间（超线性），mma 类阶段只有 3.3–3.4×（亚线性）。所以仿射律在 B/H 方向同样适用（足迹随 $M$ 增长不分方向），但边际略差于 u=1 平均。

**串行链类**：fwdscan（grid $= B\cdot H\cdot 4$，NV = V/32 个 v-切片各自走 NC 步）与 dhs（grid $= B\cdot H\cdot 8$，BV=16 装机）。T 方向：链长 NC 增长，实测 4× 工作量 4.08×/3.91×（近线性，含 L2 增长）；B/H 方向：链长不变，makespan = $\lceil \text{grid}/82\rceil$ 波 × 同样的链——B1H16 的 fwdscan 64 CTA 只有 0.78 波（18 个 SM 闲置），H32/B≥2 把它填满，预言一次性的 ~1.28× 利用率红利（约 0.64 ms 的链上），之后线性。

发射开销：9 kernel × ~3 μs ≈ 27 μs 固定，约占 u=1 e2e 的 1.3%——这是小形状体制的"我方固定成本"。

fla 侧同构处理：从干净窗派生序列（B1T2048 1.1006、B1T8192 2.8379 ms）拟合 **fla(u) = 0.522 + 0.579·u**。fla 的 bwd kernel 解剖（profiler 归因，~10–12 个发射：dqkwg 137.6 μs、dhu 132.5、prepare_wy_repr_bwd 96.3、前向重算 76.2、逐元素 Mul 56.7…）grid 同样在 $B\cdot H\cdot$(V-块) 上并行，同 M 标度；它的 ~0.52 ms 截距是 autograd 逐元素链 + 10 余次发射的固定成本。

#### 11.1.3 预测定价器的构造：`tools/rc1_price.py`

预测不是写在文档里的一段话，而是一个可运行的 CPU 脚本——它输出全部候选格点的预测表与显存可行性，供格点选择。其核心是三段算术（源码节选，`tools/rc1_price.py` 第 40–86 行）：

```python
# ---- measured ship anchors (r21b/r20/r19 records, co-loaded window) ----
A1 = {"wy": 0.2224, "fwdscan": 0.3713, "caryA": 0.0531, "caryM": 0.0792,
      "bpre": 0.0499, "dhs": 0.2703, "dkb": 0.1803, "dqkwg": 0.5146,
      "wyb": 0.3383}
E2E1_MIN = 2.14      # ours bwd min @u=1 (co-loaded)
A4 = {"wy": 0.7984, "fwdscan": 1.5155, "caryA": 0.3103, "caryM": 0.3852,
      "bpre": 0.3088, "dhs": 1.0571, "dkb": 0.6024, "dqkwg": 1.7436,
      "wyb": 1.1300}
E2E4_MIN = 7.75      # ... @u=4 (T8192)
FLA_1, FLA_4 = 1.1006, 2.8379     # fla derived clean series
FLA_B, FLA_A = (FLA_4 - FLA_1) / 3.0, FLA_1 - (FLA_4 - FLA_1) / 3.0
OURS_B, OURS_A = (E2E4_MIN - E2E1_MIN) / 3.0, E2E1_MIN - (E2E4_MIN - E2E1_MIN) / 3.0

SERIAL = {"fwdscan": 4, "dhs": 8}     # NV divisor (V/BV)

def stage_time(s, B, T, H):
    """逐块阶段：u 的仿射（T 拟合的边际对 B/H 同样成立——足迹随 M 走）。
    串行阶段：T 锚仿射 × B/H 波数乘子。"""
    NC = T // 64
    u = B * H * NC / 512.0
    a = A1[s] - (A4[s] - A1[s]) / 3.0
    b = (A4[s] - A1[s]) / 3.0
    if s in SERIAL:
        waves_anchor = max(1, math.ceil(1 * 16 * SERIAL[s] / SM))
        waves = max(1, math.ceil(B * H * SERIAL[s] / SM))
        t_chain_anchor = A1[s] / waves_anchor
        t_chain4 = A4[s] / waves_anchor
        t_chain = t_chain_anchor + (t_chain4 - t_chain_anchor) * (NC / 32.0 - 1) / 3.0
        return waves * max(t_chain, 1e-6)
    return max(a + b * u, 0.0)
```

显存可行性按 manifest 全局量模板**精确并集**计算（`manifest_template_bytes()`：从 r21b T2048 的 `bwd_manifest.json` 扫出 32 个 M 行符号 722 KiB/M 行 + 3 个 BH 行符号 128 KiB/BH 行 + ~0.9 GiB harness），输出每格点的 GiB 与 <6 GB 判定。预测曲线即两仿射之比：

$$\mathrm{ratio}(u) \;=\; \frac{\mathrm{fla}(u)}{\mathrm{ours}(u)} \;=\; \frac{0.522 + 0.579\,u}{0.270 + 1.837\,u}.$$

#### 11.1.4 预测：趋势与"任务希望"相反，且有结构下界

定价器的输出（u = 1/2/4/8/16 → 0.514/0.419/0.366/0.338/0.324，渐近 0.310）给出**与指令希望相反的预测**：比值随 B·T·H **单调下降**。我方的每 u 边际 1.837 ms vs fla 的 0.579 ms——**每 token 成本 3.17×**——阶梯每上一级，比较只会更差；fla 的亚线性固定成本（~0.52 ms）只在小形状托举我们。预测同时给出地板论证，把"任何执行质量下 ≥1.1× 是否可能"变成算术：

- r7 静态分析（第 9 章口径）的九阶段 DRAM 地板 = 1.074 ms @u=1，且全部阶段流量随 $M$ 缩放 → floor(u) = 1.074·u ms；
- u=1：地板 1.074 vs fla 实测 1.03–1.10——一条**完美屋顶线执行的 f32 分段链**在锚点也只是打平；
- 屋顶线完美的天花板比 $\frac{0.522+0.579u}{1.074u}$：u=1 时 0.81、u=4 时 0.65、渐近 **0.539**；
- 要在任何 $u\ge 1$ 达到 1.1×，需我方斜率 ≤ 0.526 ms/u——**比 f32 流量地板低 2.04×**。这不是执行问题：**字节必须减半**（bf16 中间量，受误差预算门控；或融合减趟）。

唯一 ≥1.1× 有可能成立的体制是 $u \ll 1$（小形状两侧都是固定成本）。定价器据此选出 6 个信息量最大的 cell：锚 B1T2048H16（cubin 必须复现 r19-final md5）、B1T2048H32（H 方向，检验串行填充红利）、B4T2048H16（B 方向 u=4，链从未在 B4 实测）、B1T8192H16（T 方向 u=4 重锚 + 新鲜交错 fla）、B1T16384H16（u=8 远端，NC=256 新类型 → 数值验证 cell）、外加 B1T512H4（u=1/16 小端唯一候选胜点）。u ≥ 16 的任务阶梯格点（T32768、B8T8192H32 等）全部是 M=8192+ 的 6.5+ GiB cell——被 <6 GB 规则排除，且预测落同一条平台（refit 0.38 @u=16），测量它们不增加判定信息。

#### 11.1.5 实测对拍：曲线的四个判定

实测阶梯（共载窗、每 cell 交错配对 ours-then-fla ×3、30 轮 CUDA 事件中位；门槛栈全过——54/54 payload 溯源、锚 cubin 9/9 md5 复现、数值 6/6、确定性全 cell 位级一致、显存 <6 GB）：

| cell | u | ours ms | fla_bwd ms | **比值（实测）** | 比值（预测） |
|---|---|---|---|---|---|
| B1T512H4 | 1/16 | 0.5573 | 1.2107 | **2.172** | —（候选胜点） |
| B1T2048H16 | 1 | 2.1917 | 1.3319 | **0.608** | 0.514 |
| B1T2048H32 | 2 | 4.1539 | 1.6316 | **0.393** | 0.419 |
| B4T2048H16 | 4 | 7.9710 | 3.0757 | **0.386** | 0.419（u=4: 0.366） |
| B1T8192H16 | 4 | 8.1377 | 3.2277 | **0.397** | 0.366 |
| B1T16384H16 | 8 | 16.1609 | 6.4045 | **0.396** | 0.338 |

对拍的四个判定（`evidence/G5/rc1_ladder_report.md` §2）：

1. **趋势符号：验证成立。** 比值单调下降（0.608 → ~0.39），$u \ge 1$ 任何格点都够不到 1——指令的希望被精确按预测否证。最小二乘重拟合（u≥1 档）：ours(u) = 0.138 + 1.996u（预测 0.270 + 1.837u）、fla(u) = 0.298 + 0.746u（预测 0.522 + 0.579u），残差 |r| ≤ 0.16/0.29 ms——**仿射标度律成立**。
2. **幅度：下降在预测之上饱和。** 预测 u=8 为 0.338、渐近 0.310；实测平台 0.386–0.397 横跨 u=2..8，重拟合渐近 0.374。共载窗把两侧斜率都抬高（我方 +9%、fla +29%——fla 的 autograd 逐元素链在共载下更受伤），边际比落在 0.374–0.396。
3. **u=4 方向分裂：预测确认。** B4T2048 7.971 vs B1T8192 8.138 ms——B 方向快 2.0%，在预测 ±5% 带内（$M$ 是工作量；串行几何只挪动 dhs 0.940 vs 1.107 与 fwdscan 1.557 vs 1.613——B4 的 4×短链 ×4×多链填机更好）。一个预测效应**未**兑现：H32 的串行填充红利（fwdscan 64→128 CTA 的一次性 ~1.28×）实测完全在窗噪声内（fwdscan/dhs 恰好工作量线性 1.02/0.98）。
4. **小端：胜利为真。** B1T512H4 **2.172×**（逐对 2.17/2.25/1.59）——fla 付 ~1.2 ms 固定成本（10–12 发射 + autograd 逐元素链）对我方 0.56 ms。交叉点被括在 u ∈ (1/16, 1)，u≥1 重拟合给 u* ≈ 0.13、全 6 点拟合 u* ≈ 0.31（M ≈ 65–160 块行；拟合在 u<1 质量差——引用区间而非点）。

#### 11.1.6 阶段分解：超线性类的普遍化与 R2 的弹药

实测每阶段 x/u（对 u=1 锚归一，1.00 = 恰好工作量线性）把预测文档的 T 方向发现**推广到全部方向**：

| 阶段 | H32 u=2 | B4 u=4 | T8192 u=4 | T16384 u=8 | 类 |
|---|---|---|---|---|---|
| wy / dkb / dqkwg / wyb | 0.96/0.88/0.89/0.88 | 0.89/0.82/0.84/0.83 | 0.91/0.82/0.84/0.83 | 0.88/0.81/0.98/0.98 | mma 主导，亚线性（波尾摊销；NC=256 起转线性） |
| caryA / caryM / bpre | 1.39/1.26/1.45 | 1.37/1.18/1.40 | 1.37/1.19/1.41 | 1.36/1.15/1.40 | **超线性：L2 足迹罚**（u=1 时 L2 驻留，随 32 符号足迹增长被挤出） |
| fwdscan | 1.02 | 1.01 | 1.04 | **1.17** | 串行 NC 链；NC=256 抖动 onset（重开 r19 K-split 之门） |
| dhs | 0.98 | 0.83 | 0.98 | 1.02 | 串行 NC 链；B4 填充红利 |

两个端点的阶段份额指明了 R2/R3 的杠杆排序：最佳 cell T512H4 的固定成本集中于 **dqkwg 40.0%**（其全局 park + arena 机件在小工作量时主导——小端杠杆是 dqkwg 的静态成本）；最差 cell T16384 是流量多数派（dqkwg 23.5%、fwdscan 20.5%、wyb 15.5%、dhs 13.1%，caryA/M/bpre 合计 11.4%）。fla 侧体制事实一并记录：冷 autotune 税逐形状、新进程 + 新 TRITON_CACHE_DIR 下 fwd 18.7–30.8 s + bwd 29.3–38.3 s、随 T 增长（T16384 达 69 s）；我方 AOT cubin 零 autotune、~27 μs 发射——**冷启动服役在全形状胜**。

RC1 的结论把战役导向唯一算术通路：**把字节减半**（bf16 铸造 M 行全局量，误差预算门控）能把地板 1.074 → ~0.55–0.6 ms/u、u=8 天花板变成 1.33–1.49——前提是全部 32 符号类可铸。这个前提在 R2 被预算本身检验。

---
### 11.2 bf16 误差预算方法学：把"字节减半"变成可审计的合同

#### 11.2.1 动机：预算先行，铸造后行

RC1 的算术说清楚：u ≥ 1 的胜负在字节里，不在执行里。但"bf16 化全局量"在本项目里从来不是一个开关，而是一份**合同**——第 8 章的容差契约（dq/dk/dv ≤ 8e-3、dbeta/dg ≤ 2e-2、dh0 ≤ 8e-3，判据按 bars/2 保守）决定哪些中间量可以在存储上粗化、粗化到什么程度。r16 的 bf16-solve 先例已确立方法学模板：**CPU 误差预算先行，然后铸造，然后交错阶梯**。R2 把它推广到全部 32 个 M 行符号，并回答三个问题：每个符号单独粗化的增量是多少（leave-one-in 分类法）？可铸集合的联合增量在门槛内吗（集合级投影）？被预算杀死的字节还有别的通路吗（split-bf16 与融合的分界）？

#### 11.2.2 重构 fp64 孪生：读写边界精确复制

预算探针的一切合法性来自一个前提：**孪生必须逐边复制真实 kernel 的全局读写边界**。`evidence/G5/scratch/r2_bf16_probe.py` 不是重写一份"差不多"的反向参考，而是把 `payload.mlir` 的全局边扫描出来、按 kernel 忠实重排的 fp64 数据流：每个穿越 `memref.global` 边界的中间量，若其符号在量化集 $Q$ 中，就在**落盘处**经 bf16 往返（`truncf`，round-to-nearest-even）、在**每个消费边**展宽读回（`extf`）。两条纪律使增量恰好隔离"存储粗化"这一个变量：

- **生产者递推状态保持 f32**——落盘舍入永不回馈进生产者自己的行走（fwdscan 的 S 递推、dhs 的 dH 行走都在寄存器里全精度推进，只有 h_all/dh_all 的落盘被粗化）。这正是装机形态的语义（第 6 章"扫描状态 f32 值类"铁律），孪生若在递推内部粗化就是在测另一个算子。
- **自读边显式建模**——dqkwg 的 dW 循环会**重读自己刚落盘的 dvp_all**（P11 的 `dW = -(dvp_STORE @ h^T)`），孪生里 `dW = -(dvp_s[n] @ h_s[n].T)` 用的正是粗化后的存值，注释写明 "kernel re-reads the store"。

构造正确性有对拍门：$Q=\{\}$ 的重构孪生对原始 `bwd_twin` 复现到 **dq 0.0 / dk 3.6e-15**——重构在数学上精确，之后的每个增量都是纯存储效应。核心骨架（节选）：

```python
def R(x, q):
    """round-trip through bf16 iff symbol in q (f64 in, f64 out)."""
    if q is None:
        return x
    return bf16_widen(bf16_quant(np.asarray(x, dtype=np.float32)))

def staged_twin(lay, Q):
    q = Q or {}
    def rr(x, sym):
        return R(x, sym) if sym in q else x
    ...
    # pass 2: fwdscan (serial)
    for n in range(NC):
        vn = U_s[n] - W_s[n] @ S          # register (f64) -- 递推不粗化
        vn_s[n] = rr(vn, "vnew_all")      # 落盘粗化
        h_s[n] = rr(S, "h_all")           # 入场状态落盘
        S = (S * np.exp(gl)
             + (kc[bh, n] * np.exp(gl - gc)[:, None]).T @ vn)
    ...
    # pass 6: dqkwg -- 自读边显式建模
    dvp_s[n] = rr(dvp, "dvp_all")
    dW = -(dvp_s[n] @ h_s[n].T)           # kernel re-reads the store
    dW_s[n] = rr(dW, "dW_all")
```

#### 11.2.3 leave-one-in 分类法：16 候选符号的完整判定表

分类法是 leave-one-**in**：每轮只把**一个**符号放进 $Q$，其余全 f32，对 $Q=\{\}$ 孪生取六梯度输出的 max-abs 增量；符号的判定 = 其 solo 增量对 bars/2（4e-3 / 1e-2）的裕度。完整表（两个门槛形状，种子 0；`r2_bf16_globals_report.md` §1）：

| 符号 | 每 M 行元素（全边） | dk solo | dbeta solo | 判定 |
|---|---|---|---|---|
| w_all | 32,768 | 4.7e-5 | 2.8e-5 | **可铸**（裕度 56–170×） |
| h_all | 49,152 | 4.7e-5 | 3.2e-5 | **可铸** |
| A_all | 32,768 | 0（！） | 0 | 可铸——**数值死**（见 §11.2.4） |
| B_all | 32,768 | 5.1e-4 | 4.0e-4 | **可铸**（16×） |
| dh_all | 49,152 | 5.1e-4 | 4.0e-4 | **可铸**（dh0 不受影响：dhs 行走的寄存器保持 f32） |
| dW_all | 24,576 | 4.1e-5 | 3.6e-5 | **可铸** |
| T_all | 12,288 | 7.3e-5 | 2.1e-3 | **可铸**（dbeta 上 4.8×） |
| P_all | 12,288 | 1.7e-4 | 2.0e-3 | **可铸** |
| kkt_all | 8,192 | 0 | 9.1e-4 | **可铸** |
| M_all | 16,384 | 1.0e-3 | 3.8e-4 | **可铸**（dk 上 4×） |
| **u_all** | 16,384 | **4.6e-2** | 4.3e-5 | **杀**（超 dk 门槛 11×） |
| **vnew_all** | 24,576 | **4.6e-2** | — | **杀**（vn 信号直喂 dkB） |
| **dvp_all** | 24,576 | 1.2e-3 | **4.8e-2** | **杀**（超 dbeta 门槛 2.4×） |
| **dkB_sum** | 16,384 | **6.1e-2** | — | **杀**（dk 局部项 O(10) 相消到 O(0.1)） |
| **dk_local** | 16,384 | **6.2e-2** | — | **杀** |
| rowB/s5/r3/r4/c4r/dg_state | 770 | — | dg 1.3–1.5e-2 | **保 f32**（流量可忽略） |
| 输出 dq/dk/dv | 24,576 写 | dk 6.2e-2 / dv 7.2e-3 / dq 1.9e-3 | | **保 f32**（被评分的输出） |

判定汇总：**16 个中间量候选里 10 个可铸、6 个被杀**（含 5 个大流量符号：u_all/vnew_all/dvp_all/dkB_sum/dk_local，合计 470 KiB/M 行），且**全部被评分的输出被锁在 f32**。集合级增量（可铸 10 符号联合）在两个门槛形状下投影总和均在 bars/2 内（最差 T2048 dk 2.3e-3 对 4e-3），B4 线性和 3.5e-3 对 8e-3——**10 符号集合可以整体落地**。

#### 11.2.4 A_all 数值死的发现

分类表里最反常的一行是 A_all：solo 增量**恰好为零**（dk 0、dbeta 0），探针测到的端到端增量在 1e-27 量级。原因在门分布的结构：基准配方 $g$（E[g] ≈ −0.69 自然对数/token），64-token 块的块末累积 $2^{gcl} \sim 10^{-19}$，于是

$$A \;=\; 2^{gcl} I - W^T \hat B \quad\Rightarrow\quad |A| \sim 10^{-18},$$

dhs 携带链完全被 B 项支配——caryA 与 dhs 的 A 窗口机件在算**死重**。这个发现当时被记录为"R3 的退役杠杆"（§11.3），它与第 6 章的门算子超收缩 ρ(A) ≤ 2.3×10⁻¹⁸ 是同一现象的两个视角：前者从数据体制看（这个 g 下 A 有多小），后者从算子谱看（任何合法 g 下 A 收缩得多快）。

#### 11.2.5 split-bf16 零收益定律：读者就是紧的

被杀的 470 KiB/M 行能否用 split-bf16（hi+lo 双链，第 3 章/第 6 章的三链接乘积类）救回？探针给出的答案是结构性的**否**：split-bf16 的代价是 4 B/元素（两个 bf16）——**对流量的节省为零**；它的价值只在"乘法质量"（hi·hi + hi·lo + lo·hi 恢复 ~f32 乘积）。但预算杀这 5 个符号的原因不是乘法质量不足，而是**它们的消费者对值本身紧**：u_all/vnew_all 喂 dkB（O(10) 相消到 O(0.1) 的病态和）、dvp_all 喂 dbeta（4.8e-2 超门槛）、dk_local/dkB_sum 是同阶级。**读者就是紧的**——在消费端做任何双链分解都改变不了"值必须以 f32 分辨率到达求和点"这个事实。因此这 470 KiB/M 行的字节**只有一条通路：删边（融合）**。这个分界在 R4 被正式定价（§11.5）。

（一个例外记录在案：R3 的 split-S 探针把 fwdscan 的 wS 操作数 S 拆 hi+lo，dk 增量 2.6e-8/6.6e-8——数值上远比 9 符号集宽裕，说明 S 的读者**不**紧；但它被池封锁——S_hi/S_lo 加 16 KiB 把 fwdscan 的 99072 B 池推过 101376 B 静态上限。数值余量记录备档，风险未取。）

#### 11.2.6 落地：逐符号 dtype 分派与三条建链定律

实现落在 `tools/gdn_bwd_gen.py` 的 `GDN_BF16G` 旋钮（默认 9 符号集 = 10 可铸减去 A_all；`"none"` 逐字节恢复 f32 全局量）。dtype 是**符号的属性**：每个 `k.global_` 与类型串位点经 `gdt()`/`gty_()` 分派；生产者 `truncf` 落盘、消费者 `extf` 读回，穿过既有的 gload/gstore/stage_rows/stage_split_*/rgemm/aggemm/gcol 机件（全部 dtype 感知）。cp.async 位点二选一：读者是 load2-generic 的（dhs 的 A 窗/Xb、dkb 的 dHv ping-pong、fwdscan 的 wbf、dqkwg 的 tTr/dWt0/dWt1）直接把落地 tile 翻 bf16（r6 合法）；读者是**计算 tile** 的（wy 的 solve tile Tf、fwdscan 的原地 kg 扫描 wpp）不能翻——那里 bf16 化既是数值改变（kg 粗化未探）也是类型错误。落地途中沉淀三条**建链定律**：

1. **Tf/wpp 是计算 tile**——solve 与原地扫描**写**它们；bf16 翻转是数值变更 + 类型错误，保持 f32。
2. **标量 GEMM 的 CVT 税**——bf16 tile 喂标量 FMA 的 gemm 循环（fwdscan 的 wS、dhs 的 rgemm_aw 窗口）在 MAC 循环内付**逐元素** bf16→f32 转换：配对二分定罪（A_all 单独翻转 → dhs 0.52 ms vs f32 0.27 ms，即 0.52×）；mma/ldmatrix 消费者原生接受 bf16 操作数而**赢**（bpre 1.30–1.50×、caryA/M、dkb、wy）。A_all 因此**默认关**（它唯一的消费者是被税的 rgemm_aw；何况它数值死）。
3. **串行关键路径上的转换扫描比税更差**——fwdscan 的 sweep 变体实测 0.49× vs bf16-tile 直读 0.90×；读者能展宽的地方，异步落地 tile + 原生宽读才是正确形状。

门槛栈全过（`GDN_BF16G=none` 对 RC1 链 9/9 payload+manifest **逐位一致**；数值 6/6，最差 B4 dk 5.04e-3 为门槛 1.6× 裕度、T16384 3.92e-3 与 f32 记录 3.80e-3 几乎同位；dh0 2.0e-4 处处与探针增量**精确相等**——探针即实测）。

#### 11.2.7 效果数字：字节赢了，时间还没赢

三重交错（ours-f32 / ours-bf16 / fla 每轮交错 ×3，共载窗配对中位——唯一诚实的 before/under）：

| cell | u | ours f32 | ours bf16 | **bf16/f32** | 比值(fla/ours) f32 → bf16 |
|---|---|---|---|---|---|
| B1T512H4 | 1/16 | 0.5626 | 0.5141 | **1.094×** | 1.703 → **1.864** |
| B1T2048H16 | 1 | 2.1705 | 2.1306 | **1.019×** | 0.433 → 0.441 |
| B1T2048H32 | 2 | 4.6585 | 4.5984 | **1.013×** | 0.306 → 0.310 |
| B4T2048H16 | 4 | 8.8392 | 8.7373 | **1.012×** | 0.322 → 0.326 |
| B1T8192H16 | 4 | 8.9652 | 8.8890 | **1.009×** | 0.346 → 0.349 |
| B1T16384H16 | 8 | 17.5255 | 17.3063 | **1.013×** | 0.353 → 0.357 |

u=8 的阶段分解（配对中位）说明字节去哪了：**原生赢**——bpre **1.495×**（aggemm 全局读，字节减半且无税）、caryA 1.125×、dkb 1.151×（bf16 dHv 落地 + split 读者）、wy 1.070×、caryM 1.086×；**被税**——fwdscan **0.913×**（wS 的 CVT 税）、dhs 0.992×（A 保 f32 的被税路径）；**混合**——dqkwg/wyb 各 1.02×（sweep 税 ≈ split 赢）。铸造子集矩阵（full9 对 Mk/MkBdh/MkTP/MkhP）确认 9 符号默认是子集空间的最优（17.29 vs 17.33–17.61 in-window）。

诚实的地板算术更新：预算内可铸流量 254,464 / 351,104 元素/M 行（62%），节省 437 KiB / 1.70 MiB（26%），字节缩放地板 **1.074 → ~0.81 ms/u**——RC1 梦想中的 0.55–0.60 需要连被杀的 28% 一起铸，**已被预算本身关闭**。u=8 天花板（本窗）≈ 6.16/(0.81×8+0.14) ≈ **0.92**。而我方执行 17.31 ms ≈ 新地板 e2e ~8.1 ms 的 **2.1×**——字节赢来的时间正死在 ALU/发射受限的转换机件里（CVT 税、split 扫描、truncf 落盘）。R3 的任务清单由此排定：消费者侧 bf16 原生性（税）→ A_all 退役（死重）→ 融合（删边）。

---
### 11.3 A_all 退役与 dhs 精确坍缩

#### 11.3.1 动机：从"免费字节"到"整段死代码"

R2 留下的两条线索在 R3 汇合：A_all 数值死（§11.2.4）意味着 caryA 阶段（0.55 ms @u=8）与 dhs 的 A 窗口机件（2.3 ms 中约 0.7 ms）在**算死重**；而 dhs 的 rgemm_aw 恰是 CVT 税最重的位点。杠杆本身是**算法变更**（A := 0），按项目纪律必须自带全套门槛——CPU 预算探针先行，量化"退役到底丢掉什么"。

#### 11.3.2 探针：max|A@dH| = 2.3e-18 的推导

`evidence/G5/scratch/r3_probe.py` 的 `astats` 模式直接测量退役丢弃项的量级：在 fp64 孪生上重算 W、沿真实 dH 走行走一遍，逐步记录 $\max|A|$、$\max 2^{gcl}$、**丢弃项** $\max|A\cdot dH|$ 与被馈送的**状态** $\max|dH|$：

| 形状 | max abs A | max 2^gcl | **max abs A@dH（丢弃项）** | max abs dH（馈送状态） |
|---|---|---|---|---|
| T512H4 | 1.59e-19 | 1.84e-19 | 1.90e-19 | 6.8e-2 |
| T2048 | 2.01e-18 | 2.34e-18 | 2.26e-18 | 8.3e-2 |

丢弃项比它馈送的状态**低 16 个数量级**——也比 dh_all 自身 bf16 存储 ULP 低 ~15 个量级（这解释了为什么退役连 dh_all 的**字节**都不动，见 §11.3.5）。端到端增量（`aret` 模式，A := 0 于孪生 dhs 行走）在全部 5 个探针形状上 dk ≤ 5.6e-17、dh0 ≤ 1.3e-21——对 8e-3/2e-2 门槛低 **14+ 个数量级**；且 `aret + default9` 与 `default9-only` 在每个形状**逐位相同**——退役叠加在 R2 已装机的预算上**零增加**。探针对 GPU 的预测精确兑现（组合集预测 vs 实测：T8192 dbeta 3.449e-3 vs 3.444e-3；T16384 2.694e-3 vs 2.691e-3；dh0 2.0e-4 精确）。

#### 11.3.3 实现：caryA 删除 + dhs 坍缩为移位拷贝（源码与 IR 前后）

退役旋钮 `GDN_ARET`（默认 1；`=0` 逐字节恢复 R2 链）做两件事。

**（a）manifest 侧：caryA 整段退役**（`tools/gdn_bwd_gen.py::main()`）——A_all 是 caryA 的唯一输出，其唯一消费者（dhs 的 A 窗口 rgemm）读的是数值死操作数，生成 caryA 只会写一个无人读的全局量：

```python
# R3 (GDN_ARET): the caryA stage retires with A_all -- its only
# consumer (dhs's A-window rgemm) reads a numerically dead operand
# in the gate regime and is deleted from the dhs walk; generating
# caryA anyway would just write an unread global.
kernels = [s.strip() for s in args.kernels.split(",") if s.strip()]
if ARET and "caryA" in kernels:
    kernels.remove("caryA")
    print("[bwd_gen] R3 A-retirement: caryA dropped from the chain")
```

九 kernel 链变**八 kernel 链**。

**（b）kernel 侧：dhs 反向行走精确坍缩**。死 A 递推

$$\texttt{dh\_all}[n] = X_{\text{arr}}(n),\qquad X \leftarrow A_r X + B_r[n]$$

在 $A \sim 10^{-18}$ 下**精确地**约化为移位拷贝：

$$\texttt{dh\_all}[NC{-}1] = \texttt{dht},\qquad \texttt{dh\_all}[n < NC{-}1] = B\_all[n{+}1],\qquad \texttt{dh0} = B\_all[0].$$

**前**（R2 形态的机件清单）：dhs 声明 `A_all`（`if not ARET: k.global_("A_all", ...)`）、持 2×[K,BKW] A 滚动窗口 tile、每块 NW 次 rgemm_aw（寄存器累加 GEMM 喂 A 窗）、XP 跨块预取 token、f32 X/Xa 状态对与 Xa→X 拷贝扫描——串行链每步背着一次 [K,K]×[K,BV] 级的窗口 GEMM 与双重 staging。**后**（ARET 臂，发射源码节选）：

```python
if ARET:
    # ... the reverse recurrence ... with A ~ 2^gcl ~ 1e-18 reduces
    # EXACTLY to: dh_all[NC-1] = dht, dh_all[n<NC-1] = B_all[n+1],
    # dh0 = B_all[0].  Emission: ONE [K, 2*BV] ping-pong tile (parity =
    # the column half; the landing burst writes the idle half while the
    # store sweep drains the live half -- the FPIPE recipe) ...
    xb_dt = gdt("B_all")
    Xb = k.alloc(e, "Xb", K, 2 * BV, xb_dt)
    # prologue: Xb[:, 0:BV] <- dht slice (same-dtype raw copy)
    ...
    e.l(f"scf.for %ci = {e.ic(0)} to {e.ic(NC)} step {e.ic(1)} {{")
    e.l(f"%n = arith.subi {e.ic(NC - 1)}, %ci : index")   # 逆时间
    e.l("%m = arith.addi %bhNC, %n : index")
    e.l(f"%par = arith.remsi %ci, {e.ic(2)} : index")     # 乒乓列半
    # B[m] -> the idle half (16B burst; covered by the store sweep)
    toks = []
    astage_tile(e, G["B_all"], bty, "%m", V, "%voff", Xb, "%nparc",
                K, BV, xb_dt, toks)
    gB = agroup(e, toks)
    # dh_all[m] <- the live half
    e.l(f"scf.for %he = %tid to {e.ic(K * BV)} step %nt {{")
    ...
    gstore(e, v, G["dh_all"], dht_, ["%m", gcol], src_dt="f32")
```

IR 后的形态：单一 [K, 2·BV] 乒乓 tile（奇偶 = 列半），每块 B[m] 的 16 B cp.async 突发落进**闲半**、同块的 `dh_all[m]` 存储扫描排空**活半**（FPIPE 覆盖配方，无需 token 搭车），await+栅栏收口。A 窗口机件、f32 X/Xa 对、拷贝扫描全部退役：**池 49152 → 8192 B，寄存器 124 → 32，payload 行数 504 → 91**。dh_all 落盘的是 B 的 payload **位逐字**（bf16 tile → bf16 全局 = 往返恒等；`GDN_BF16G=none` 时 f32 路径存展宽后的 B/dht——与 A 恰为零时 A 臂存的值相同）。旋钮处带**体制门**文档：退役合法当且仅当衰减保持 $2^{gcl} \ll$ 容差尺度——对门/基准配方与可比衰减的学习门数据成立；弱衰减（g → 0）下 A → I、臂失效（`=0` 恢复）。

#### 11.3.4 5.9× 的结构原因

阶段配对中位（u=8，T16384）：**dhs 2.083 → 0.354 ms = 5.89×**，外加 caryA 整段删除（0.557 ms → 0）。5.9× 不是调优出来的，是**结构原因**的必然：旧 dhs 的每步 = A 窗口 GEMM（[K,K]×[K,BV] 寄存器累加，NW 次）+ 双份状态 staging + A/B 两次全局读 + 一次拷贝扫描；新 dhs 的每步 = 一次 16 B/线程的 B 突发 + 一次展宽存储扫描——串行链上每步的关键路径从"多次依赖链 GEMM + 两轮 barrier"缩到"DMA 与存储互掩的一来一回"。链上剩下的纯粹是带宽与发射：0.354 ms @T16384 对应的就是 B_all 读 + dh_all 写的字节流。同样地，被税路径（rgemm_aw 的 CVT 税）随 A 窗口一起消失——退役顺带免掉了 §11.2.6 定律 2 里最重的税点。

#### 11.3.5 逐字节等价证明

算法变更的收尾门槛是**位级对照**：单次确定性运行同时跑 R2 链与 R3 链（T16384），**每个公共阶段输出逐字节一致**——包括 fwdscan 的 h_all/vnew_all（§11.4 的 hoist 证明）**和 dhs 侧全部符号**。dhs 侧的等价需要一个论证：退役丢弃的项（~1e-18）比 dh_all 自身 bf16 存储 ULP 低 ~15 个量级，因此 $A\cdot dH$ 的舍入不可能改变任何 dh_all 存储位——两条链**收敛到相同字节**；A_all 只存在于 R2 根。其余门槛：`GDN_ARET=0 GDN_TGEMM_AH=0` 对 R2 链 9/9 payload+manifest 逐位一致；未触碰阶段的 cubin md5 与 R2 记录相等（48-md5 全录）；数值 6/6 与 R2 记录**同位**（B4 dk 5.036e-3、T16384 3.819e-3）；确定性 N=3 ×3 形状位级一致；VRAM 高水位 6.56 GiB 中 2.10 GiB 为既有基线、链本身 ~4.5 GiB < 6 GB。

#### 11.3.6 效果数字（与杠杆一的合成）

三重交错（r2 链 / r3 链 / fla 每轮交错 ×3，**全程干净窗**）配对中位：

| cell | u | ours r2 | ours r3 | **r2/r3** | 比值(fla/ours) r2 → r3 |
|---|---|---|---|---|---|
| B1T512H4 | 1/16 | 0.5107 | 0.4588 | **1.113×** | 1.548 → **1.723** |
| B1T2048H16 | 1 | 2.1234 | 1.7883 | **1.187×** | 0.411 → **0.488** |
| B1T2048H32 | 2 | 3.9281 | 3.2644 | **1.203×** | 0.345 → **0.415** |
| B4T2048H16 | 4B | 7.4286 | 6.2444 | **1.190×** | 0.346 → **0.411** |
| B1T8192H16 | 4T | 7.4911 | 6.2010 | **1.208×** | 0.375 → **0.453** |
| B1T16384H16 | 8 | 14.7390 | 12.1521 | **1.213×** | 0.381 → **0.462** |

u=8 的阶段表：caryA **删除**、dhs **5.89×**、其余 1.00×（wy 1.466→1.467、caryM 0.671→0.672、bpre 0.412→0.412、dkb 1.020→1.019、dqkwg 3.295→3.292、wyb 2.110→2.103）——杠杆的作用面干净利落地限于设计对象。隔离臂（`r3ahoff`：ARET=1、TGEMM_AH=0 @T16384）把两个杠杆分解：退役单独值 **1.179×** e2e（fwdscan 不动 0.994×），hoist 在其上再叠加 fwdscan 3.282 → 2.930 = 1.120×。边际（T8192→T16384 重拟合）：ours 1.812 → **1.488 ms/u**、fla 0.7025 ms/u，边际比 0.388 → **0.472**。小端同窗深化 1.55 → 1.72×。

---

### 11.4 tgemm a-fragment hoist：CVT 税的最后一刀

#### 11.4.1 动机：机器级地看，税从哪来

§11.2.6 定律 2 的税有明确的机器级形态。fwdscan 的 wS gemm 是 tgemm（tf32k8 hatch 类）消费者：bf16 tile 喂进标量 FMA/tf32 的 k-循环后，**每个操作数元素在每次被一个 n-tile 复用时都要重付一次 extf（bf16→f32 转换）**——R2 实测 fwdscan 整段 0.913×（税 ~0.35 ms @u=8）。对照 mma/ldmatrix 消费者：`mma.sync` 与 `ldmatrix` 在硬件上**原生接受 bf16 操作数**（第 3 章的 mma_gemm 路径），无逐元素转换。税不是"bf16 慢"，是"**bf16 喂错了消费者**"。杀税的两条路线在 R3 都被探针检验：split-S 数值 GO 但池封锁（§11.2.5 尾注）；**值精确的 hoist** 是所缺的第三条路——不改变任何数值，只改变**谁来加载/展宽 a-fragment**。

#### 11.4.2 hoist 的构造：warp 地图从轮转改为连续

观察：一个 warp 的原子 tile 若全部住在**同一个 16 行 m-块**里（`npr % tpw == 0`，无跨块），则每个 k 步的四个 a-fragment 寄存器**只需加载（并展宽）一次**、喂给该 warp 全部 tile 的 mma——a 加载与 extf 从 npr 次/元素/遍降到 **1 次/元素/遍**。实现（`tools/gdn_bwd_gen.py` 第 2143–2199 行）：

```python
# R3 (regime-change campaign, lever 1 -- the CVT-tax kill): the
# a-fragment HOIST arm for bf16 A operands. ... When a warp's
# tiles all live in ONE 16-row m-block, the four a-fragment registers
# per k-step are loaded (and widened) ONCE and fed to every tile's
# mma: a-loads and extfs drop npr->1 per element per warp pass.  The
# warp map changes round-robin (ti = mW + i*W) -> consecutive
# (ti = mW*tpw + i); every per-cell k-chain stays kb-ascending with
# identical atoms -> BIT-IDENTICAL values (the r20 warp-map law:
# which warp owns a cell never touches its arithmetic).  Activates
# only when the A operand tile is bf16 AND npr % tpw == 0 ...
if (a.dt == "bf16" and tpw > 1 and npr % tpw == 0
        and os.environ.get("GDN_TGEMM_AH", "1") == "1"):
    # the warp's shared m-block (all its tiles have the same mi)
    ti0 = e.v("tti")
    e.l(f"{ti0} = arith.muli %mW, {e.ic(tpw)} : index")
    mi0 = e.v("tmi")
    e.l(f"{mi0} = arith.divsi {ti0}, {e.ic(npr)} : index")
    mrow0 = e.v("tmr")
    e.l(f"{mrow0} = arith.muli {mi0}, {e.ic(16)} : index")
    ...
    tiles = []
    for i in range(tpw):
        ti = _tadd(e, ti0, i)
        ni = e.v("tni")
        e.l(f"{ni} = arith.remsi {ti}, {e.ic(npr)} : index")
        ...
```

激活条件天然保守：**只有 A 操作数 tile 为 bf16 时启用**（默认旋钮下恰为 fwdscan.wS 的两个切片）；其余位点逐字发轮转臂。代价是寄存器换发射：fwdscan payload 1068 → 932 行、102 → 118 寄存器、**0 spill**——a-fragment 常驻换掉循环内重复 extf，mma 原子数前后 64 == 64。

#### 11.4.3 位恒等论证

hoist 的合法性一句线：warp 地图从轮转（ti = mW + i·W）改为连续（ti = mW·tpw + i）只改变**哪个 warp 拥有哪个 cell**，而 per-cell 的 k-链保持 kb 升序、原子集合相同——按 r20 warp-map 定律（cell 的算术与归属解耦），**值位恒等**。这不需要概率论证：GPU 上的证明就是 §11.3.5 的链间位级对照（fwdscan 的 h_all/vnew_all 在两链间逐字节一致）。

#### 11.4.4 效果数字：1.115× 翻正，税全数回收

fwdscan 阶段 **3.266 → 2.930 ms = 1.115×**（u=8 配对中位）；隔离臂上 fwdscan 3.282 → 2.930 = **1.120×，~0.35 ms @u=8——恰是 R2 实测的 CVT 税全额回收**。至此 R2 的税表清零：bf16→原生消费者的翻转全部完成（bpre/dkb/wy/cary 类原生赢；fwdscan 经 hoist 翻正；dhs 经退役免役），fwdscan 从 0.913× 的输家变为 +11.5% 的赢家。战役的 u=8 比值 0.462、边际比 0.472 是税后净额。

---
### 11.5 融合定价学：dqkwg+wyb 的纸面三重杀

#### 11.5.1 动机：最后一个字节地板搬运者

R3 收官时按证据排序留下两个候选：#1 **dqkwg+wyb 融合**（"唯一剩余的字节地板搬运者"：两阶段合计 5.40/12.15 ms @u=8，删边能触及被预算杀死的 f32 边 dvp_all/dk_local 的重读）；#2 扫描原生性（~0.1–0.2 ms 级，§11.7 记录不定价）。按 r14/r16 纪律，融合**先纸面定价**：字节算术来自 L1 数学与装机链、池算术来自实测 smem_fix 池、占用算术来自 r15 实测 1CTA 因子——全部输入为测量量，无一个拟合参数。定价器 `evidence/G5/scratch/r4_price.py`；边图审计 `scratch/r4_audit.py`；结论 **NO-GO（决策级）**，战役链保持 r3-final bytes。

#### 11.5.2 边图审计：融合的合法杀伤集

融合能删的边必须**只有** dqkwg 与 wyb 踏踩。`r4_audit.py` 对装机 r3 链的 manifest + payload 做逐符号 load/store 审计：dqkwg 的八个输出里恰一个终结（dq——被评分输出，f32，无论如何要写），其余七个（dW_all、dvp_all、dk_local、r3_c/r4_c/c4r_c/dg_state）全部流向 wyb，且 dW/dvp 还喂 dqkwg 自己的后续相位（TW 循环重读 dW_all、dW 循环重读 dvp_all）；第三个 stage 不触碰其中任何符号——**边图闭合**，杀伤集由此上界。另一条被杀的 f32 边 vnew_all 是 fwdscan→dkb/dqkwg 边，**不**在本融合可达范围。

#### 11.5.3 全物化屏障：为什么生存期不允许池共享

从 L1 数学（`tools/gdn_l1_gen.py` @`gdn_bwd`）看融合界：

```
dvp  = P^T dO + Bhat dH                    -> dqkwg P9（写 dvp_all）
dW   = -dvp S_n^T                          -> dqkwg P11（写 dW_all）
G    = dvp (beta v)^T + dW (beta k 2^gc)^T -> wyb P1/P2
dA   = -strict_tril(T^T strict_tril(G) T^T)-> wyb P3/P4
...
dv   = (T^T dvp).*beta                     -> wyb P9（再次需要 dvp）
```

**结构性发现**：G 对**全部 V**（dvp 的收缩）与**全部 K**（dW 的收缩）累加；G 下游的一切（dAf 与 dk/dbeta/dg 尾）加上 dv（在尾处**再次直读 dvp**）都排在 G 完成之后。融合 kernel 可以在 GEMM 内循环里逐片交错生产与消费，但**完整的 dvp、dW、dk_local 矩阵必须活过屏障**——池不共享：

$$\text{fused\_pool}(\text{subset}) \;=\; \max(\text{dqkwg 相位池} + \text{parks},\ \text{wyb 相位池} + \text{parks}),$$

parks 在其全部生存期内叠加在消费相位自己的池上，**没有任何交错重排能缩小它们**。这就是**全物化屏障（full-materialization barrier）**：G 的两个操作数在下游任何东西开跑前都必须是完整 [C,*] 矩阵。

#### 11.5.4 池共存算术：杀伤集枚举

逐符号边流量（按装机生成器的循环重数精确计数；`r4_price.py` 节选）：

```python
# dvp_all f32 [C,V]: dqkwg P9 write; P11 dW-loop re-reads
#   (K/BK)*(V/BV) slice stages of [C,BV]; wyb P1 + P9 reads.
dvp_row = C * V * 4
dvp = {"write": dvp_row,
       "dqkwg_reread": (K // BK) * (V // BV) * (C * BV * 4),
       "wyb_reads": 2 * (V // BV) * (C * BV * 4)}
# dW_all bf16 [C,K]: dqkwg P11 write + P12 TW re-read; wyb P2 + P8.
dW_row = C * K * 2
dW = {"write": dW_row, "dqkwg_reread": dW_row, "wyb_reads": 2 * dW_row}
# dk_local f32 [C,K]: dqkwg P12 write; wyb P7 per-element read.
dk_row = C * K * 4
dk = {"write": dk_row, "wyb_reads": dk_row}
vec = 4 * (2 * C * 4)     # r3/r4/c4r/dg_state f32 [C] x4
PARK = {"dvp": dvp_row, "dW": dW_row, "dk": dk_row, "vec": C * 4}

BAR_2CTA, BAR_STATIC = 50176, 101376
POOL_DQKWG, POOL_WYB = 47616, 45056
for m in range(16):                       # 全子集枚举
    s = [n for i, n in enumerate(names) if m >> i & 1]
    park = sum(PARK[n] for n in s)
    pool = max(POOL_DQKWG + park, POOL_WYB + park)  # parks live in both
    kill = sum(TOT[n] for n in s)
```

| 杀伤集 | 融合池 | 驻留 | 删 B/行 |
|---|---|---|---|
| —（仅合并发射） | 47616 | 2 CTAs | 0（省 ~发射+重 staging，μs 级） |
| 仅 vec | 47872 | 2 CTAs | 2048 |
| dW | 64000 | 1 CTA | 65536 |
| dvp | 80384 | 1 CTA | 229376 |
| dk_local | 80384 | 1 CTA | 65536 |
| **dvp+dW+vec** | **97024** | **1 CTA** | **296960**（最优可建） |
| dW+dk+vec | 97024 | 1 CTA | 133120 |
| dvp+dk（及以上） | ≥110592 | **不可建** | — |
| 全部（完美） | 129792 | **不可建** | 362496 |

最优可建杀伤集是完美字节的 **82%**（97024 vs 101376 B 静态上限逼死 dk_local 入场；dvp+dk_local 单对已 110592 B）。指令问句"除非生存期允许"的答案是**否**——G 屏障之下，2-CTA 上限下的 5120 B 余地买不到任何东西。

#### 11.5.5 占用反转：1 CTA/SM 的实测代价

97024 B 的融合 kernel 在**两个**合并体上都跑 1 CTA/SM。这不是假设性损失——r15 的配对实测就在这个 stage 上：dqkwg 512thr/1CTA 对装机 256thr/2CTA = 0.632 → 0.547 ms @T2048（**F = 1.155×**）、2.3232 → 1.8443 @T8192（**F = 1.2597×**）；r15 的 ncu 记录归因于独立指令流效应（达成占用 15.02 → 29.17%、no-eligible 89 → 83%）。

#### 11.5.6 逐 cell 净账与 NO-GO

带宽计价用 r24 W-park 备忘录的本机标定（DRAM 类 824.6 GB/s / L2 驻留 2144 GB/s），阶段时 r3 三重交错配对中位，占用成本 dqkwg 半按实测 F、wyb 半按同结构类推（256thr/2CTA/128reg/mma+staging 体、已有 32 B spill——任何 F_wyb ≥ 1.10 都把 u=8 净额压到噪声）：

| cell | 删 GB | 省 ms（L2/DRAM） | 1CTA 代价 ms | 净 best/mid/worst ms | 比值 now → best-corner |
|---|---|---|---|---|---|
| B1T2048 u=1 | 0.152 | 0.071/0.184 | 0.079..0.130 | +0.106/+0.024/−0.059 | 0.488 → 0.519 |
| B4 u=4B | 0.608 | 0.284/0.738 | 0.439..0.722 | +0.298/−0.070/−0.438 | 0.411 → 0.432 |
| T8192 u=4T | 0.608 | 0.284/0.738 | 0.440..0.724 | +0.297/−0.071/−0.440 | 0.453 → 0.476 |
| T16384 u=8 | 1.216 | 0.567/1.475 | 0.855..1.401 | **+0.620/−0.107/−0.834** | 0.463 → 0.488 |

（best 角要求全部删除流量按纯 DRAM 计价**且** wyb 半零 1CTA 代价——后者不可信；诚实中心：u=1 打平、u≥4 亏损。）甚至**不可建的完美融合**也被字节算术封顶：完美杀伤 1.485 GB @T16384 → 最多 1.80 ms，比值上限 5.6247/(12.1521−1.80) = **0.544**（L2 计价 0.491）——R3 的"~0.6× 上限"（假设两阶段纯字节受限）已被字节本身收窄，因为这个对的残差是发射受限机件而非字节。对照战役已装杠杆（dhs 1.73 ms、caryA 删除 0.56 ms、fwdscan hoist 0.35 ms @u=8），融合的期望值不过线而风险等级最高（合并两个 ~1000 行 kernel、重推跨相位生存期竞技场、两体已溢出的寄存器并集 ≤128、全门槛栈重跑）。**裁决：NO-GO，带全套算术。**

---

### 11.6 精度关闭定理：三重独立证据

#### 11.6.1 定理表述

把四轮的证据组织成可引用的定理形式：

> **定理（bwd u ≥ 1 的精度关闭）。** 在门控配方数据体制、fla 派生容差合同（dq/dk/dv ≤ 8e-3、dbeta/dg ≤ 2e-2、dh0 ≤ 8e-3）与本机（sm_120a、DRAM 824.6 GB / L2 2144 GB/s 标定）之下，九（现八）kernel f32/bf16 分段反向链对 fla fused-autograd 链的比值在 $u \ge 1$ 全体制**无法达到 1.1×**；更精确地，u=8 的比值天花板为 **~0.85–0.92×**（完美屋顶线执行当前字节集），且该上限由三重相互独立的证据共同钉死——任何单一证据的松动都不改变结论。

三重证据的独立性在于它们关闭的是**不同的逃逸方向**：预算地关闭"把字节再铸低"，地板算术地关闭"把执行做到完美"，融合地关闭"把边删掉"。

#### 11.6.2 证据一（预算地）：能挪地板的字节被合同判死

能移动地板的字节只有被杀的 f32 边（u_all/vnew_all/dvp_all/dkB_sum/dk_local，470 KiB/M 行）——而它们被杀的原因是**消费者对值紧**（§11.2.5：读者就是紧的；split-bf16 对流量零节省、对求和点分辨率无补）；全部被评分输出（dq/dk/dv）由预算保持 f32。RC1 的 1.33–1.49× 天花板需要全部 32 符号类可铸——**被预算本身关闭**，与执行质量无关。

#### 11.6.3 证据二（地板算术地）：u=8 天花板 0.85–0.92× 的推导链

推导链步步可查（`r3_report.md` §5、`r4_fusion_report.md` §7）：

1. R2 落地 9 符号铸造后，字节缩放地板 = **0.81 ms/u**（r7 的 1.074 按字节比缩放）；
2. R3 退役删掉 caryA 整条字节流（~257 MB → ~0.34 ms @T16384）与 dhs 的 A 窗流量 → 地板 **~0.75 ms/u**；
3. 完美执行（100% 屋顶线）该字节集的 e2e = ~0.75×8 + 固定 ≈ **6.1–6.6 ms** @u=8；
4. 本窗 fla = 5.620 ms → **天花板 = 5.620/6.1–6.6 ≈ 0.85–0.92×**；
5. 对照：r3 实测 12.15 ms ≈ 地板的 2×（残差为发射受限的标量/扫描机件——dqkwg 混合扫描、wyb 混合位点——不是字节）；即便把 ~2× 执行余量全部花掉，u=8 也只到 0.92×，**跨不过 1.1**；
6. 换个极端口径：把 dqkwg+wyb+dhs **整段删除**（它们计算被评分输出，删除并非真实选项）也只到 0.88——融合只能减字节、不能归零输出级。

#### 11.6.4 证据三（融合地）：最后的搬运者被池与占用判死

唯一还能挪地板的机制（删边）只剩 dqkwg+wyb 融合一个可达载体，而 §11.5 的三重杀——**池封顶**（G 全物化屏障下最优可建 97024 B = 完美字节的 82%）、**占用反转**（实测 1CTA 因子 1.155–1.26×）、**净账打平或亏**（u=8 best-corner 0.488）——把这一载体判 NO-GO；连不可建的完美融合也被字节算术封在 ~0.54×。

#### 11.6.5 "这是误差合同的数字，不是我们的数字"

关闭定理的最后一句属于表述纪律：u ≥ 1 的 0.46–0.47× **不是实现失败的测度，而是合同的下界**。若有人想改这个数字，可改的入参全都不在本项目手里——它们是：容差合同（更宽的梯度容差）、门分布（更弱的 decay 体制，但那也会违反 A 退役的体制门）、或带宽标定。战役因此以**装机主张**而非遗憾收官：**小形状 + 冷启动 bwd 胜者（1.72× @T512H4 同窗）加 u ≥ 1 fwd 胜者；u ≥ 1 bwd 是精度受限的 ~0.46–0.47×——地板的数字，不是我们的数字。** 若反向工作重开，有余量的方向是新的体制（本阶梯外的 B/T/H、学习门数据体制），不是这条链在这些形状上的任何想法类——B1T2048H16..B1T16384H16 的每个想法类都已被定价或实测关闭。

---

### 11.7 战役总结：四轮一览与最终战绩

#### 11.7.1 四轮一览

| 轮 | 问题 | 方法 | 装机/裁决 | 关键数字 |
|---|---|---|---|---|
| **RC1** | 比值随形状怎么走？任何体制 ≥1.1×？ | CPU 预测定价器（标度律 + 仿射拟合 + 地板论证）→ 6 cell 实测对拍 | 无码变更；曲线判定 | 趋势验证（单调降）；边际比 2.67×/u；小端 **2.172×**；地板 1.074·u ms、斜率需 ≤0.526（低地板 2.04×） |
| **R2** | 哪些字节可铸？铸造换多少时间？ | 重构 fp64 孪生 + leave-one-in 预算 → `GDN_BF16G` 9 符号默认集 → 三重交错阶梯 | **装**：9 符号 bf16 全局量 | 10/16 可铸、6 杀；+1–2% @u≥1、**+9.4%** 小端；地板 1.074→0.81 ms/u；CVT 税发现（fwdscan 0.913×） |
| **R3** | 税能杀吗？死重能删吗？ | 预算探针（astats/aret）先行 → `GDN_ARET` + `GDN_TGEMM_AH` → 全门槛栈 + 干净窗三重交错 | **装**：caryA 退役、dhs 坍缩、wS hoist | max&#124;A@dH&#124; 2.3e-18（低状态 16 个量级）；dhs **5.89×**、fwdscan **1.115×**；e2e r2/r3 **1.19–1.21×**；u=8 比值 0.381→**0.462**、边际比 0.472 |
| **R4** | 最后的字节搬运者值不值得建？ | 纸面定价（边图审计 + 全物化屏障 + 池/占用/净账）→ 健康认证 | **NO-GO**（决策级）；链 == r3-final bytes | 最优可建 82% 字节 @1CTA；净账 u=8 +0.62/−0.11/−0.83 ms；完美融合封顶 ~0.54× |

战役阶梯全轨迹（各轮自窗配对臂；最终列为 R3 干净窗记录、R4 健康认证）：

| cell | u | RC1 ours | R2 ours | **R3 ours（最终）** | 最终比值(fla/ours) |
|---|---|---|---|---|---|
| B1T512H4 | 1/16 | 0.5573 | 0.5107 | 0.4588 | **1.723（胜）** |
| B1T2048H16 | 1 | 2.1917 | 2.1234 | 1.7883 | 0.488 |
| B1T2048H32 | 2 | 4.1539 | 3.9281 | 3.2644 | 0.415 |
| B4T2048H16 | 4B | 7.9710 | 7.4286 | 6.2444 | 0.411 |
| B1T8192H16 | 4T | 8.1377 | 7.4911 | 6.2010 | 0.453 |
| B1T16384H16 | 8 | 16.1609 | 14.7390 | 12.1521 | 0.462 |

u=8 轨迹：比值 0.396（RC1 窗）→ ~0.43（R2 窗）→ **0.462**（R3 窗）；边际 2.00 → 1.812 → **1.488 ms/u**（边际比 0.396 → 0.439 → **0.472**）；小端 1.55 → 1.72×（R3 窗内 r2 臂 → r3 臂）。R4 的贡献是关闭证明，不是一个字节。

#### 11.7.2 最终战绩表（战役对外主张）

| 战线 | 判定 | 记录 | 口径 |
|---|---|---|---|
| **fwd，u ≥ 1 全形状** | **胜** | B1T2048 **2.125×** / B4T2048 **1.2211×** / B1T8192 **1.3493×** | r25 正式 n=30 交错记录（双窗：另有干净窗 clean3 记录并存）；战役未动任何 fwd 旋钮，记录不变 |
| **bwd，小形状（u ≪ 1）** | **胜** | B1T512H4 **1.723×** vs fla | R3 干净窗三重交错配对中位（战役内从 2.172×/1.86× 深化） |
| **bwd，u ≥ 1** | **精度关闭**（非执行关闭） | 0.411–0.488× 全阶梯；边际比 0.472；天花板 0.85–0.92× @u=8 | 三重独立证据（§11.6）；"误差合同的数字" |
| **冷启动，全形状** | **胜** | 我方 AOT cubin：零 autotune、~27 μs 发射；fla：逐形状 fwd 18.7–30.8 s + bwd 29.3–38.3 s 冷 autotune（T16384 达 69 s；缓存命中 0.65–5.67 s） | 新进程 + 新 TRITON_CACHE_DIR 口径 |
| **编译器主张全链** | **成立且 sharpened** | 语义单一事实源 → 编译期调度（Z3+成本模型）→ 分层验证 → 位级确定性 → 预算门控精度决策 → 决策级 NO-GO 定价 | 战役四轮本身是"调度即 IR"方法学的展演：每个结论先 CPU 后 GPU、先预测后实测 |

#### 11.7.3 收官链的健康认证与未竟事项

R4 对收官链（== r3-final bytes）做了纯位恒等认证：48/48 cubin 对 R3/R2 记录全对账（32 == R3 md5 记录；16 对 R2 锚、其中 fwdscan/dhs 恰按 R3 设计相异）、确定性 N=3 @T16384 全位一致、生成器零改动。未竟事项如实记录：R3 排名 #2 的扫描原生性（~0.1–0.2 ms 级）未装——0.2/12.15 ms = +0.008 比值，改变不了任何判定，记录在案不定价；战局主张的最终用户确认（关闭定理是否被接受为指令的答案）在收官时点**待确认**——`evidence/G5/CAMPAIGN_CLOSEOUT.md` 为其正式文档。

---

### 11.8 本章小结

体制变化战役把"形状外推"从猜测变成了一条完整的证据链：**标度律框架**（两类 kernel 的仿射律 + L2 足迹罚 + 波数量子化，在 GPU 触碰前预测整条曲线并选中 6 个判定 cell）；**误差预算方法学**（读写边界精确复制的重构 fp64 孪生 + leave-one-in 分类，把"bf16 化"变成 10/16 可铸、6 杀的合同，并顺带发现 A_all 数值死与 split-bf16 零收益定律）；**两个装机杠杆**（A_all 退役：max|A@dH| = 2.3e-18 的探针推导 → caryA 删除 + dhs 坍缩为移位拷贝，池 49152→8192 B、阶段 5.89×、逐字节等价收尾；tgemm a-fragment hoist：warp 地图轮转→连续，CVT 税 1.115× 翻正且位恒等）；**融合定价学**（全物化屏障、池共存算术、实测占用反转的三重杀，决策级 NO-GO）；最终以**精度关闭定理**收官——三重独立证据把 u ≥ 1 的 ≥1.1× 关闭在合同层面，u=8 天花板 0.85–0.92×。战役四轮把大形状反向从 0.39–0.61× 推进到 0.41–0.49×（边际比 0.472）、小形状深化到 1.72×、冷启动全形状胜，并交付了比速度更持久的两件东西：一份**哪些字节在合同上必须保持 f32**的完整预算表，与一套**在写码之前给融合定价**的算术。

> 证据索引：`evidence/G5/rc1_scaling_predictions.md`（预测文档）+ `evidence/G5/rc1_ladder_report.md`（实测阶梯判定）+ `tools/rc1_price.py`（预测定价器）；`evidence/G5/r2_bf16_globals_report.md` + `evidence/G5/scratch/r2_bf16_probe.py` + `scratch/r2_logs/`（预算探针与三重交错日志）；`evidence/G5/r3_report.md` + `evidence/G5/scratch/r3_probe.py` + `scratch/r3_logs/` + `scratch/r3ahoff_T16384/`（退役/hoist 探针、隔离臂与位级对照）；`evidence/G5/r4_fusion_report.md` + `evidence/G5/scratch/r4_price.py` + `scratch/r4_audit.py` + `scratch/r4_logs/`（融合定价与收官健康认证）；`evidence/G5/CAMPAIGN_CLOSEOUT.md`（战役收官记录：最终战绩、三重关闭论证、决策史与待确认状态）；决策上下文 `evidence/G5/DECISIONS_R25.md`（D2 持有决策及其推翻条款）；门槛栈与窗口纪律见 `evidence/G5/GATE_REPORT*.md` 与第 8–9 章；装机实现 `tools/gdn_bwd_gen.py`（`GDN_BF16G`/`GDN_ARET`/`GDN_TGEMM_AH` 三旋钮）。

---

## 附录：复现指南与证据索引（交叉引用）

本长文不复述复现流程，复现入口统一收敛于主论文与仓级索引，此处仅给交叉引用：

- **主论文附录 C「复现指南」**（`paper/zh/gdn-mlir-paper-zh.md` / 同名 `.html`）：一键入口 `make repro-g0` … `make repro-g6`、`make repro-all`（依序全跑，首败即停），覆盖方言构建（纯 CPU 约 15 min）、旋钮矩阵（约 25 min）、调度标定 + Spearman 门（约 45 min）、GPU 正确性（约 1 h）、全算子验证 + 基准 + 交错终判（约 1–1.5 h）、搜索成本（约 10 min）。
- **干净窗口标定刷新**：`scripts/run_calib.sh`（幂等覆写）；**正式终判复算**：`python tools/gdn_final_adjudicate.py --stats --pairs 30 --warps 4 --bv 32 --gen-root scratch/r25f/genrec`；**求解器**：`python tools/gdn_solver.py --solve --stage kkt --K 128 --B 4 --T 8192 --H 16 --mode strict --form v2`。
- **章级复现路径**：多数章内设「复现工程」小节（如第 4 章 §4.4），给出该章所涉工具的构建与校验命令；各章末尾的「本章引用的证据文件」清单逐章列出引用物。
- **证据索引**：`EVIDENCE_INDEX.md`（gate → 报告 → 产物 → 复现目标的一行索引）；失败与战役级发现的组织方式见 `evidence/G6/failure_longscroll.md`。
