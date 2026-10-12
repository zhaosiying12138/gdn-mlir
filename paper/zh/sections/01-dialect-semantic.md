# 1 方言与语义层（L0）与逃生舱 op 族

## 1.1 章首：这一层为何存在

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

## 1.2 `gdn.fwd`：前向语义入口

### 1.2.1 动机（算子层面，含例子）

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

### 1.2.2 TableGen 源码逐行讲解

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

### 1.2.3 Verifier 源码逐行讲解

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

### 1.2.4 真实 IR 展示（generic 语法）

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

## 1.3 `gdn.bwd`：反向语义入口

### 1.3.1 动机（算子层面，含例子）

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

### 1.3.2 TableGen 源码逐行讲解

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

### 1.3.3 Verifier 与真实 IR

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

## 1.4 `GdnConfigAttr`：七个语义旋钮

### 1.4.1 动机与定义

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

### 1.4.2 七旋钮逐个讲解

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

### 1.4.3 assembly 语法与真实用例

`assemblyFormat` 生成的属性拼写（冒烟测试与全部生成器均用此形式）：

```mlir
#gdn.config<64, true, false, true, true, true, false>
       ^chunk  ^g   ^gk   ^nq   ^init  ^store ^varlen
```

七个参数以逗号分隔、尖括号包裹，按定义序排列。属性在 op 上以 `config = #gdn.config<...>` 出现（1.2.4 已见）；`gdn-to-chunk` 改写后它以 `gdn.config = #gdn.config<...>` 键名**原样透传**到入口调用上（1.6 节），键名 `gdn.config` 即方言名 + mnemonic 的标准展开。

---

## 1.5 GdnHatch 逃生舱 op 族：11 个薄封装 + 1 个真逃生舱

### 1.5.0 总览：逃生舱机制与"零预算发现"

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

### 1.5.1 `gdn.setmaxnreg`：warp 级寄存器再分配

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

### 1.5.2 `gdn.elect`：warp 领袖选举

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

### 1.5.3 `gdn.mbarrier.init`：mbarrier 初始化

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

### 1.5.4 `gdn.mbarrier.arrive`：到达操作

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

### 1.5.5 `gdn.mbarrier.arrive.expect_tx`：带预期事务字节数的到达

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

### 1.5.6 `gdn.mbarrier.try_wait.parity`：相位奇偶等待

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

### 1.5.7 `gdn.tma.async.load`：TMA 异步张量装载

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

### 1.5.8 `gdn.tma.async.store`：TMA 异步张量写回

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

### 1.5.9 `gdn.tma.prefetch`：描述符预取

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

### 1.5.10 `gdn.bulk.commit_group`：提交 bulk 异步组

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

### 1.5.11 `gdn.bulk.wait_group`：等待 bulk 异步组

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

### 1.5.12 `gdn.mma.tf32k8`：唯一的真逃生舱

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

### 1.5.13 `gdn-lower-hatch` pass 与整体下降流

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

## 1.6 `gdn-to-chunk` pass：L0 → L1 展开通道

### 1.6.0 pass 总体与两个 rewrite 模式

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

### 1.6.1 `FwdToLowering`：`gdn.fwd` → `call @__gdn_l1_fwd_chunk`

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

### 1.6.2 `BwdToLowering`：`gdn.bwd` → `call @__gdn_l1_bwd_chunk`

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

## 1.7 gdn-opt 工具与外建构建体系

### 1.7.1 `gdn-opt`：方言的 mlir-opt 副本

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

### 1.7.2 外建（out-of-tree）MLIR 项目结构

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

### 1.7.3 表生成：include/CMakeLists.txt

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

## 1.8 小结

本章完整解剖了 GDN-MLIR 的语义底座：两个 L0 语义 op（`gdn.fwd`/`gdn.bwd`，ODS + verifier + generic 语法 + 真实 IR 全展示）、一个七旋钮语义属性（`GdnConfigAttr`，assembly 序即定义序）、11 个零预算逃生舱薄封装与唯一真逃生舱 `gdn.mma.tf32k8`（预算 1/≤5，携带 early-clobber 与 fragment 布局两条硬件验证发现，反向 e2e 14.16→11.06 ms 的实测载体）、两个 L0→L1 改写模式（`FwdToLowering`/`BwdToLowering`，真实 before/after IR）与 12 个 hatch 下降模式（`gdn-lower-hatch`）、以及支撑这一切的外建构建体系（三 .td 九表生成、双 include 路径、MLIROptLib 工具）。这一层的全部设计服务于同一命题：语义单一事实源必须先于调度自由度存在——L0 之上，L1 的算法结构、L2 的 Z3 调度、L3 的流水与 mma 才有可验证的下降起点。

（本章引用的证据文件：`evidence/G1/GATE_REPORT.md`、`evidence/G2/gen_report.md`、`evidence/G2/hatch_report.md`（含 ADDENDUM）、`evidence/G5/bwd_report.md`、`evidence/G5/scratch/tf32k8_probe/meta.json`、`evidence/G2/scratch/g2c/smoke_hatch.mlir`、`evidence/G0/fla_ptx`、`docs/L0_SEMANTICS.md`、`docs/PLAN.md`；源文件均在 `dialect/` 与 `scripts/` 下，路径已在正文逐一标注。）
