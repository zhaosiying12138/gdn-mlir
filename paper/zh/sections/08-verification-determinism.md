# 8 数值验证与确定性体系

> 本章的自洽口径（与全文全局声明一致）：**数值分母**——前向输出 o/ht 对 fp64 JAX 参考的逐元素最大绝对差；反向六梯度（dq/dk/dv/dbeta/dg/dh0）对 fp64 VJP 参考的容差契约（fla 惯例 atol/rtol 口径的绝对棒）。**确定性定义**——同一 cubin、同一输入、同一发射几何下，输出缓冲区位级（bit-level）一致；跨进程（fresh CUDA context、fresh 模块加载、fresh 设备分配）N 次采样全部一致方为通过。**证据纪律**——本章每一个数字均可回读到 `evidence/` 与 `tools/` 源文件；凡早期轮次数字与最终系统不一致处，以最终记录为准并同时给出演化注记。

本章回答一个问题：一个把调度做成 IR 旋钮、把数值路径也做成旋钮（f32/tf32/split-bf16、K-粒度、重结合次序）的编译器栈，凭什么让每一个落盘的 cubin 都"数值正确且逐位可复现"。答案是四层验证塔加三根横贯的纪律绳：容差契约（§8.1）、fp64 孪生评分（§8.2）、确定性协议（§8.3）；在此之上，竞态破案案例集（§8.4）与 cubin 指纹/纯度/cache_key 体系（§8.5）把"可复现"从口号变成可审计的工程对象。每小节按【动机 → 源码/协议片段 → 前后对比 → 效果数字】四件套展开。

## 8.0 章首：分层验证塔

验证体系是一个自底向上的四层塔。每一层的 oracle 独立于被测对象本身，且上一层以下一层的通过为前提；塔的顶端不是"一次验收"，而是**逐轮门**——任何触碰数值路径或调度路径的变更都要重新过全部门。

**第一层：L1 CPU 三方对拍（G1）。** 语义层在进入 GPU 之前先在 CPU 上闭合为三方一致：MLIR（`linalg`+`scf` 真 IR，经 `mlir-opt`/`mlir-runner` 在 CPU 执行）⇄ JAX 可执行规约（fp64，`gdn_seq`/`gdn_chunk`/`jax.vjp`）⇄ fla naive 参考（只读 oracle，`naive.py`，内部硬转 fp32）。判据与数字：JAX 规约自检 44/44（chunked vs sequential fp64：o 5.6e-16 / ht 1.2e-15 / 六梯度 1.2e-14；有限差分 vs VJP 2.3e-9；jit 奇偶 0.0；torch fla 交叉 1.2e-7 / 2.3e-6，fp32 口径）；MLIR L1 前向 7/7（vs JAX dumps ~6e-7，即 `printMemrefF32` 六位有效数字打印地板）；MLIR L1 反向 3 npz × 6 梯度全部 PASS（最差 6.4e-6，同样贴打印地板）。三方链条：L1↔JAX 6e-6 + JAX↔fla 1e-6/2e-6 量级（evidence/G1/GATE_REPORT.md；evidence/G1/jax_spec_report.md；evidence/G1/l1_harness_report.md）。此外，numpy 侧的 chunked 前向孪生对逐步递推孪生（`naive_recurrent` 移植）自差 2.4e-7——**评分参考自身先被验证**，这是全塔的地基。测量地板如实披露：runner 经 stdout 打印 memref，只有 ~6 位有效数字，6e-7 以下的差异不可分辨，真实 fp32 误差只会更小。

**第二层：GPU 20 配置逐配置对拍（G4）。** L2 旋钮库的全部 20 个配置（kkt/o 阶段的 mma/nomma/warp-specialized/TMA 变体 + 两个 f16 锚点）逐个在 GPU 上执行：重新生成（固定 `--seed`）→ 重编译（`scripts/run_l2.sh` → `tools/gdn_smem_fix.py`）→ 子进程隔离发射（墙钟超时，挂起的 kernel 只杀死自己的 worker）→ **读回完整输出缓冲区** → 与"从同一份嵌入输入计算的精确 numpy fp64 孪生"逐元素比较。判据：bf16/f16-io 配置 max-abs-diff ≤ 5e-3，f32-io 配置 ≤ 1e-3（fla 派生契约的阶段级形式）；同时做两项结构检查——非零元计数必须与参考完全一致（每个 kkt 配置写满全部 2080 个严格下三角单元，G3 曾实测 mma 路径只写 1168 个，正是覆盖性 bug），以及**确定性检查**：清零 + 重发射必须逐位复现输出（`deterministic` 字段）。生成器确定性同时受验：同 seed 两次生成的 payload.mlir sha256 相同、异 seed 不同。结果 20/20 PASS，双 seed（0 与 7）双跑 20/20；全部 mma 配置的实测最大差 9.77e-4，与"精确模拟 f32 累加 + bf16 舍入输出"管线预测的 9.765625e-4 相等——**实测 == 预测**，残差被证明是纯输出量化而非计算误差（evidence/G4/GATE_REPORT.md；evidence/G4/verify_results.json、verify_results_seed7.json）。该层还把 TMA 配置的主机侧接线（`tensormap_spec.json` → cuda-bindings `cuTensorMapEncodeTiled`）纳入 harness，使"验证基础设施完备"本身成为被测对象。

**第三层：全算子 GPU 验证（G5）。** 前向（`tools/gdn_full_verify.py`）：staged mono 链（F1→F2→F3）与 persistent fused 两条路径 × 两个标定形状，对 G5-A 共享 JAX fp64 参考（同一 inputs npz）比较 o/ht，棒 5e-3，判据为**零元素越棒**（`o_over_tol == 0 and ht_over_tol == 0`）；首验 4/4 PASS（T512 o 4.77e-4 / ht 7.00e-4；T2048 o 5.75e-4 / ht 7.93e-4，两路径同值）（evidence/G5/fwd_report.md round 1）。反向（`tools/gdn_bwd_verify.py`）：九阶段链（wy→fwdscan→caryA→caryM→bpre→dhs→dkb→dqkwg→wyb，G5-J 第 5 轮重构后的最终结构：dhu 退役，caryA/caryM/bpre 并行预复合每 chunk 仿射扫描算子 $A_n = 2^{g_{cl}}I - W^\top\hat B$、$M_n = Q_g^\top - W^\top P^\top$、$B_n = M_n\,\mathrm{d}O$，dhs 每 chunk 单 rgemm 载链，dv/dvp_all 移入 dqkwg）对 fp64 numpy 孪生（bf16 量化输入上运行，见 §8.2）比较六梯度，棒 8e-3/8e-3/8e-3/2e-2/2e-2/8e-3，判据 6/6；小形状另有 JAX vjp oracle 交叉（`--jax`）。结构性不变量同层受验：T8192 分段分裂 S=1/4/8 下 o 与 ht **位级一致**（f2-segments 旋钮不改变数值，evidence/G5/fwd_report.md）。

**第四层：逐轮门（round gate）。** 从 G5 第 2 轮起立规："Numerics stay green through every perf change"——每一个性能变更落地前必须重过 6/6（前向 4/4），数值不变类（value-identical）变更必须**逐位复现记录数字**；从 G5-Q 第 10 轮起，确定性进入同一道门：凡触碰 staging/arena/mma/barrier 的变更，附加跨进程 fresh-process N 次（初版 N=10，舰载常规化为 N=5）双形状位级一致证明（checklist 第 7 条，§8.3.5）。这一层的存在使 26 轮反向战役（58.88→2.15 ms）与 25 轮前向战役中的每一字节推进都有同一把尺子。

四层塔的完整判据—证据映射见 EVIDENCE_INDEX.md（`make repro-g1/g4/g5` 为幂等复现入口）。

## 8.1 容差契约：来源、标定与裕度

### 8.1.1 契约内容与来源【动机→片段】

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

### 8.1.2 fla-vs-JAX 标定：bf16 舍入下界 ≈2×【片段→数字】

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

### 8.1.3 我方裕量表【前后对比→效果数字】

以最终系统（前向 R25 档案代 +W32P 舰组；反向冻结于 r19-final 字节）的记录数字对棒折算。前向（vs fp64 JAX 共享参考，棒 5e-3）：staged T2048 **o 6.486e-4 / ht 7.493e-4**（≈7×/6.7× 裕度，记录数字）；T8192 fp64 门 o 6.626e-4 / ht 8.918e-4（≈7.5×/5.6×）；fused 记录数字 o 5.751e-4 / ht 7.929e-4；4/4 零元素越棒（evidence/G5/fwd_report.md rounds 18–25；evidence/G5/GATE_REPORT_FINAL.md"fwd 4/4 位级一致（T8192 fp64 门 7× 裕量）"）。

反向最终记录表（vs fp64 孪生，bf16 喂入；括号为对 8e-3/2e-2 棒的裕度，报告原文口径）：

| 形状 | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T512H4 | 4.617e-4 (17×) | **5.131e-4 (15.6×)** | 9.43e-6 (848×) | 2.178e-4 (92×) | 1.466e-4 (137×) | 1.394e-7 (57400×) |
| B1T2048H16 | 1.824e-4 (44×) | **1.284e-3 (6.2×)** | 1.088e-5 (736×) | 8.527e-4 (23×) | 3.274e-4 (61×) | 2.169e-7 (36900×) |
| B1T8192H16 | 2.078e-4 (≈39×)* | 1.881e-3 (≈4.3×)* | 1.050e-5 (≈762×)* | 1.654e-3 (≈12×)* | 1.145e-3 (≈17×)* | 1.965e-7 (≈4×10⁴)* |

（*T8192 行的记录数字摘自 r19–r22 各轮"every digit equals the record tables"表，裕度由本文按同一棒折算；evidence/G5/bwd_report.md rounds 13–22。）

裕度演化注记（教学价值所在）：dv 在 r8 曾达 **2454×**（3.3e-6，dvp 的 pdO+bh 全部走精确 bf16/split 乘积所致），r11 引入 tv split-bf16 后回落到 736×/848×——"乘积更紧、多一次重结合"类；dk 是全表最紧的棒位（r8–r10 时代 5×），r13 把七个 tf32 2^-11 位点换成 2^-17 split 后收紧到 6.2×/15.6×。dk 裕度部分是运气（r9 原话"The dk 5x margin is partly luck"）：其数值路径上存在一个此前未发现的非确定性竞态（§8.4.2/§8.4.3），同一 kernel 两次 verify 采样到 1.719e-3 或 4.719e-3（一个 bf16 量子之差）——这正是第 10 轮把 dk 相邻变更全部挂到确定性门上的原因，也解释了为何"dk 裕度最小"与"dk 首先暴露竞态"是同一件事。

## 8.2 numpy fp64 孪生方法

### 8.2.1 孪生的构造与评分口径【动机→源码片段】

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

### 8.2.2 方法论教训一：评分孪生自身有 bug——bf16 误诊的翻案【片段→前后对比】

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

### 8.2.3 方法论教训二：验证 harness 的 H2D 重分级掩盖效应【前后对比→数字】

第 5 轮扫描重构（dhu 退役）落地后，verify 首次把 dbeta 从 3.0e-4 推到 1.33e-2（1.5× 于棒内）。二分定位（probe_r5inter → probe_dbeta → payload diff → numpy bf16 仿真）发现一个自 round-2 起就存在的潜伏 bug：dqkwg 的 dvp 重算把 Bhat 分级进了 **bf16 tCKb tile**（2e-3 相对误差），且 GPU dvp 与 numpy(bf16 Bhat) 对到 3.3e-7——kernel 忠实执行了被误分级的路径。它**长期不可见的原因在 harness**：`run_chain` 在每次发射前把每个输入 global 重新 H2D 暂存，于是 wyb 永远收到 dhu 的干净 dvp_all，dqkwg 的覆写从未到达任何评分器（round-3/4 的 verify 表给的是干净路径，BENCH 跑的却是被毒染的路径）。dhu 退役后 dvp_all 的唯一生产者变成 dqkwg，毒才浮出水面。修复：bhat tile 改用该循环中已死的 f32 tCVb。**推论被写成契约**：verify harness 的 H2D 重暂存可以掩盖生产者/消费者 bug——round-5 起 dvp_all 从 dqkwg 读回评分（evidence/G5/bwd_report.md round 5"The latent bf16-Bhat bug"）。同一次狩猎还刻画了 tf32 用于 dvp_bh 时的**重建混沌**：三个数值等价 build 上 dbeta/dv 从 1.5e-5 → 2.9e-3 → 5.8e-3 跳变（任何 f32 级 dh 扰动都会重采样 mma 的操作数量化桶），据此保住标量路径。这两条与 §8.2.2 合起来构成"评分体系自身是被测对象"的完整案例集。

## 8.3 确定性协议

### 8.3.1 协议分层【动机→协议片段】

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

### 8.3.2 刷新模块的必须性：进程内循环漏检 wyb 竞态【前后对比→效果数字】

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

### 8.3.3 判别臂设计【源码片段→效果】

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

### 8.3.4 逐中间量 npz 二分定位法【动机→协议片段→效果】

**动机**：九阶段链的最终梯度发散无法区分"源头在哪个阶段"；逐阶段的 OUT 全量落盘把链变成可二分的序列。

**协议片段**：链 harness 按 `OUT` 表（源码中的链接契约）把每阶段"拥有"的全局读回存 npz——wy: `T_all,u_all,w_all,kkt_all,P_all`；fwdscan: `h_all,vnew_all,gsc_all`；caryA/caryM/bpre: `A_all/M_all/B_all`；dhs: `dh_all,dh0`；dkb: `dkB_sum,rowB_c,s5_c`；dqkwg: `dq,dW_all,dk_local,dvp_all,r3_c,r4_c,c4r_c,dg_state`；wyb: `dk,dbeta,dg,dv`。diff 模式按 STAGES 顺序对每个全局报 DIVERGES/stable、逐全局差数与 max|d|，并输出**第一个发散的阶段与全局**；比较忽略符号零/NaN 对噪声（`m = neq & ~(fa == fb)`）。r9 的 dk 定位还用同法对**阶段内部中间量**（dk_local base-vs-new npz）二分，先以 tf32/标量臂分裂把腐蚀钉在 TW gemm，再逐量收敛。

**效果（r10 破案的定位链）**：第一个发散全局 = bpre 的 `B_all`——12/8.4M 元素、集中在单个 chunk 行（m=132）、单 K 行（k=3）、一个 16 宽 V 条，相对差 ~1e-7 = 1–3 个 f32 ulp；上游（wy/fwdscan/caryA/caryM）与全部 dqkwg 输出位级稳定——与 r9 的观测精确吻合。传播链完整重建：B_all → dh_all（数值在下一加法点被吸收）→ dkB_sum → 某一 dk 元素的 bf16 round-to-nearest 翻转 = 1.719e-3 ↔ 4.719e-3 摆动（一个 bf16 量子）。bpre 修复后第二轮采样发散点移入 wyb 内部（其全部输入稳定），于是第二竞态现形——**二分法在同一轮内连续破两案**。

### 8.3.5 restage-fence 七点清单【效果：从清单到制度】

r10 破案后，对生成器中每一个 mma/ldsm 消费者做同类审计（补三个潜伏位点、列明审计干净名单），并把裁判规则固化为**七点清单**——此后每一次 kernel 变更的前置检查表（evidence/G5/bwd_report.md round 10"The audit rule added to the toolbox"）：

1. 每个被跨 warp 读取的 smem tile（ldmatrix 或 ld.shared），必须在**最后写者的存储循环之后、首个读者之前**有 gpu.barrier（以栅栏结尾的 staging helper 满足读侧）。
2. **restage 栅栏**：任何满足 (a) 暂存 tile、(b) 运行读取它们的 gemm/mma、(c) 下一迭代**重暂存同一批 tile** 的循环，必须在**读之后（迭代末尾）**有一条全线程栅栏——下一次迭代 staging 内部的尾栅栏**不能**为它们定序（对竞争中的 warp 而言双方都在该栅栏之前）。同时适用于 st.shared 重分级（bpre）与 cp.async 重分级（wyb G 环；cp.async DMA 写在 commit/wait/栅栏之前发射）。
3. cp.async wait_group 是**每线程**的：每个 wait_group 之后需要一条 gpu.barrier，其它 warp 才能读已暂存数据（r9）。
4. 栅栏必须在任何 <W warp 守卫区域**之外**（bar.sync 是全线程的）；守卫式 mma 链必须走自带尾栅栏的 `mma_chain_if`。
5. cp.async（LDGSTS）暂存的 tile **不得**被 ldmatrix 读取（r6 静默腐蚀家族）；mma 操作数 tile 一律 st.shared 写入。
6. 永不依赖"smem 分配即为零"（r7）：未读区域要么显式清零、要么证明永不被读（C+8 的 LDSM 填充列永不被读——经 lane 地址算术验证）。
7. 触碰这些路径的任何变更的确定性证明协议：`tools/gdn_det_check.py` 跨进程 N=10 **双形状**（仅 T2048 进程内循环修复前**没有**抓到 wyb 竞态——fresh-module 循环才是更宽的窗）+ verify 6/6。

第 2 条是全清单的心脏，也是标题点名"栅栏在读完之后"的原因：直觉会在 staging 尾部补栅栏（保护"写后读"），但重分级竞态是**读后写**——慢 warp 的 LDS/MMA 读还在飞，快 warp 已经开始下一轮写；唯一正确的定序点在读集合的末端。第 7 条则把前六条从"知识"升格为"门"。制度效果的直接证据：r11 起每个杠杆自带 fresh-module 循环证明（l1/l2/l3 全位级），r13/r15/r16–r22 的全部 value-exact 轮逐位复现记录数字，r21 的 cluster flag/spin 协议在共载下确定性且无死锁（N=5 双形状全位级）。

## 8.4 竞态破案案例集（教学式）

五个案例按"症状 → 定位 → 根因 → 修复 → 数字"展开；它们共同的教学点是：**每一个都在数值棒内、大多数在 1–3 ulp 量级，全部被确定性协议而非数值验证抓获；其中两个被 1-CTA 驻留掩蔽数轮。**

### 8.4.1 案例 A：cp.async wait_group 的每线程语义（r9，dqkwg）

**症状**：dqkwg v4（16-warp 占用率改造）在 256 线程/满网格下 final-dk 出现 9.3e3..9.8e10 的垃圾，而 128 线程/T512 网格下**不可见**（窗口太窄）。**定位**：tf32/标量臂分裂钉住 TW gemm 后，dk_local base-vs-new npz 二分：**每个 ks 切片的第 0..5 列**陈旧——恰好是线程 0–1 拥有的 16 B chunk。**根因**：TW 环内联的宽度受限 tCKa 暂存（`astage_tile + await_all`）不带 `astage_sync` 携带的尾 gpu.barrier；wait_group 只**每线程**地等待自己发出的 DMA 组，别的 warp 的在途 chunk 并未就绪，快 warp 竞入 TW tgemm 读到了其它线程尚未完成的列。同一次狩猎的第二发现：v3 式裸 `mma_gemm` 链无尾栅栏（`mma_chain_if` 文档化过此隐患）——dPr 的 vs 环可以在慢 warp 的 mma 读之下重分级 akb/vnh/vnl。**修复**：各补一条 gpu.barrier。**数字**：6/6 双形状 PASS，且**精确落在基线裕度数字**（dk 1.719e-3/1.710e-3——早尾重排只引入 ~1-ulp 级重结合，打印位以下）；这是 verify 门连续第 6 轮抓到当轮真 bug（evidence/G5/bwd_report.md round 9）。

### 8.4.2 案例 B：bpre 的 K 环 mma 重分级竞态（r10 根因一；r8 起潜伏）

**症状与定位**：见 §8.3.4（B_all 1–3 ulp 双稳翻转）。**根因**：bpre 的 K 环 `for i0 in (0,64): stage_split_ew(ahi,alo); mma(bmh: ahi); mma(bml: alo, epilogue → B_all)`——两个 staging helper 都以 gpu.barrier 结尾，但**没有任何东西隔开 i0=0 的 mma ldmatrix 读与 i0=64 的 split 写**（PTX 4 条栅栏，缺的在环内）。早完成的 warp（其 epilogue 的 64 次全局存储拖慢它抵达，但只要有一个慢 warp 还在发 LDSM 就够了）在慢 warp 读 alo 时覆写之。**为什么像 1-ulp 噪声而非腐蚀**：被交换的操作数元素是 LO 分量（lo = v − bf16(v)，多在 ~1e-10），新旧乘积差落在受影响输出的 1–3 ulp；一条 16 B LDSM 行（8 个 alo 元素）正是观测到的 12/16 元素簇；翻转全落在 k 行 0..63（i0=0 的输出）。ahi 侧（~1e-4）会腐蚀到 O(1e-3) 并直接 FAIL verify——但快 warp 抵达重分级点时距 bmh{0} 只有 ~100 周期， practically 从不输掉该竞态；bml{0} 是暴露环节。r9 已文档化同类（"mma_gemm inter-iteration restage race"）并在 dqkwg v4 的 dPr 环修复——bpre 被漏掉。**修复**：每轮 K 环迭代末一条 gpu.barrier；4 → 5 条栅栏，54 寄存器，0 spill。**数字**：单独 bpre fresh-module ×1000 从 14/1000 发散到 0/1000（§8.3.2 表）。

### 8.4.3 案例 C：wyb 的 G_dvp/G_dW 累积环竞态（r10 根因二；cp.async 味，更大、更老）

**症状**：bpre 修复后跨进程采样**仍然**发散——发散点移入 wyb 内部（其全部输入稳定）：dk/dbeta/dg 在 1e-6..1e-4 量级、数千元素，dv 不动。**定位**：签名直指 G（accCC2 = strict_tril dvp(βv)ᵀ + dW(βk 2^gc)ᵀ）：G 喂 dAf → dk/dbeta/dg，而 dv 的路径不经过 G。**根因**：gen_wyb 的两个累积环 `for vs: astage_sync(tCVa ← dvp_all); stage_raw_ew(tVV ← v); emap(tCVb); gemm(accCC2, tCVa, tCVb, bar=False, kahan)`（ks 环同形）在环内 gemm 之后**没有栅栏**：下一迭代的 astage_sync DMA（cp.async）重写 tCVa/tCKa 时，慢 warp 的 gemm LDS 读还在飞（emap 亦在其下重写 tCVb/tCKb/tKK）。尺度自洽：G 处全值操作数交换 ~1e-3 → dAf → 下游数千个 1e-6..1e-4 位移。**修复**：每环 gemm 后一条 gpu.barrier（2 处）；wyb PTX 127 → 138 条栅栏。**数字**：T512 修复前 3/10 触发（小网格链在重分级窗中相对停留更久）；这是 r9 采样 dk 摆动的主导贡献者（bpre 翻转只是 ulp 级尾部）。**同类审计的连带收获**：dqkwg v3 的 dPr vs 环（T512 派发）补迭代内栅栏（v3 PTX 192 → 195）；v2 遗留 dvb ks 环在 <W 守卫**外**补栅栏（守卫内会死锁）；审计干净名单（wy kkt/pmm、dkb、caryA/caryM、dhs、fwdscan F2 组、wyb TW/尾环）逐一记录在案（evidence/G5/bwd_report.md round 10）。

### 8.4.4 案例 D：zero_up 步幅 bug——一指令 SASS 证词（r18 根因一，fwd F1 2-CTA 腐蚀）

**背景**：F1KS 把 f1 压到 46,352 B smem（首个 2 CTA/SM 的前向 kernel）后，2-CTA 共驻下 u/w 腐蚀（o 6.3e-2 对 5e-3 棒），而 ht 停在记录数字 7.493e-4；八臂二分表（arena 有无、staging 类别、bk/mb、pool 在 2-CTA 带内的取值）证明 K-split 代码在一切 1-CTA 臂上位级正确、腐蚀随 2-CTA 共驻本身出现；racecheck 在**干净** build 里也报同样 6 对"良性"扫描步 LDS/STS，memcheck 干净——两个 sanitizer 都以串行化调度把 bug 藏起来。**症状指纹**：仅 f1 在场即可复现（无需 f2/f3）、逐 rep 非确定、逐 tile 二值腐蚀（研究 run 中 338/512 个 tile、后调度的 CTA 腐蚀）——陈旧态签名而非数值 bug。**定位与证词**：forced-1-CTA PASS build 与 2-CTA FAIL build 的 **SASS 差恰好一条指令**（`MOV R4, 0x2490` vs `0x4410`——Tg 的 malloc 尺寸 9,360 vs 17,424）：代码相同，触发器是纯共驻。tile 转储（G5D_DUMP_TT 扩展 Tg 与 epilogue 后半 tile）把 Tb/Tg 差**精确**钉在 zero_up 漏写的单元（每 tile 768 个；计数 43,264 = 338 tiles × 128 cells）；r17 的 o/u-vs-ht 之谜由指纹解开：w 恰在 Tb/Tg 块列 Cb=2 = 行 32–47、u 在行 0–47、行 48–63 干净（mma 输出块 mi=3 只从 k-块 >3 收垃圾，而后者不存在）；被腐蚀 chunk 的状态贡献在末 chunk 前衰减到契约之下（f2 的 ht 完美），f3 却直接读 h_all/v_new——o 败、ht 过，从未有输出路径之谜。**根因**：solve 的 phase-0 `zero_up` 单元循环以 `32*8 = 256` 步幅跨 128 线程块行走，Tb/Tg 中 `(r*64+c) mod 256 ∈ [128,256)` 的单元——左下六个 16×16 块中行 `r mod 4 ∈ {2,3}` 者、每 tile 768 格——**从未被清零**；uw gemm 的转置 ldmatrix A 读覆盖整个 64×64，把这些格读作 T 的上三角零。1-CTA 下读到**精确零**（512 tile 全部 393,216/393,216 格实测）；2-CTA 共驻下读到非零活数据（~56% 格，±1e-14 级远对角衰减值、偶发 O(1) 离群、absmax 1.0），且与顺序无关（1-CTA 在垃圾重的 2-CTA run 之后仍零干净）。陈旧字节的精确供给路径（跨槽渗漏 vs 分配器粒度 vs 前次发射遗留）**留为不归因**——修复消除对三者的全部依赖。**修复**：`GDN_ZFIX`（步幅 nthr，全覆盖）；默认关（未设 == HEAD payload 逐字节，复核两次），经舰组环境打开。**数字**：ZFIX 后腐蚀从 ~70 万 bit 差塌缩到 ~4 千；f1 代价 +8.3 µs@T8192（零循环翻倍到 32 迭代），被 F1KS 收益覆盖 6 倍（evidence/G5/fwd_report.md rounds 17–18；scratch/r18f/probe1..8、dump_1cta/dump_2cta）。

### 8.4.5 案例 E：Hillis-Steele d=32 跨 warp 竞态与 SCANDB（r17 发现、r18 定罪）

**症状**：ZFIX-only 之后残余 ~4 千 bit 差、每 run 1–2 个 tile 触发、指纹**精确落在 `gc[32:64)` 窗口**（Hillis-Steele 第 d=32 步的跨 warp 读），且在 chunk 内局部传播（该 chunk 的 u/w 行 32–63）。**根因**：f1 的 gc 累加使用 6 步 smem Hillis-Steele 扫描（取代每线程串行 cumsum），d=32 步要求读**另一 warp** 写的半程和——该跨 warp 步内读写未定序；racecheck 在每个 build（含干净 build）报的正是这 6 对扫描步 LDS/STS。1-CTA 驻留从未暴露它（r17 时代因此只作为"建议硬化"落地而默认关）。**修复**：`GDN_SCANDB`——扫描步双缓冲（r17 落地、r18 作为舰载硬化打开：mono 配 ZFIX+SCANDB，fused 配 SCANDB）。**数字**：**ZFIX+SCANDB 于 2 CTA/SM：10/10 次 f1-only 重复对 forced-1-CTA 参考位级一致，全 trio 命中记录数字（o 6.486e-4 / ht 7.493e-4）**；1-CTA 下硬化是值精确的（ZFIX+SCANDB T8192 build 对 r15 舰在 fp64 门上位级一致，o/ht 0.000e+00；FPIPE+SCANDB fused 与 FPIPE-only fused 值位级一致，后者重建精确复现 r17 舰 md5 93856752）；det N=5 跨进程三套件 FWD-DETERMINISTIC（det_ship/、det_fpipe_sdb/、det_f1ks/）（evidence/G5/fwd_report.md round 18）。

案例 D+E 合并的教学结论：**两个都被 1-CTA 驻留掩蔽、都自 blocked solve 落地起就活在每个 build 里**；把 f1 推过 2-CTA 线的占用率工程（本身是纯调度变更）充当了它们的触发器。这正是"确定性协议必须与占用率/驻留变更绑定重跑"（清单第 7 条覆盖"触碰这些路径的任何变更"）的实证理由——调度旋钮可以唤醒沉睡的数值 bug，而位级门在它破坏任何数值棒之前就把它截获。

## 8.5 cubin 指纹、旋钮纯度与 cache_key

### 8.5.1 md5 舰队登记【动机→效果数字】

**动机**：论文的每一个性能记录都绑定"哪一个 cubin 跑出来的"；没有字节级指纹，记录之间不可比、复现声明不可审计。制度：每轮报告的 Ship provenance 表登记全部舰 build 的 `kernel_smemfix.cubin` md5，记录迁移时旧字节整体保穴（r25 迁移把 r22 字节保存在 `<name>.r22ship` 兄弟目录）。关键指纹（最终系统）：前向 R25 档案代 mono 三形状 `e7c8eb4c / 39f094c3 / be5b2377`（T8192/B4/T2048；由 HEAD 1c7d291 全新构建，md5 与 r24 终版候选 3/3 精确一致——未改树+同环境的重建**逐字节复现**，模块布局彩票只在生成器编辑时重掷，不在重建时）；其前任 r22 记录代 `be8b914b / 83898f74 / 3d47886a`；r18 时代的六舰表（mono `ea8ec2f1/e7af55c1/a065a0bc` + fused `d16522d0/cb2520d8/676487bf`）完整留档。反向冻结于 r19-final 字节：基线 18/18 阶段 cubin md5 == r20 记录（如 dqkwg `837a36fcc908/1f53c8910006`、dkb `1538de9d5dee/fbb0682b6776`、fwdscan `e680e1d94e47/304b4259a79d`，T2048/T8192）；K32 臂 `fb89448c760e/e615d26e9402`（evidence/G5/fwd_report.md rounds 18/22/25；evidence/G5/bwd_report.md rounds 20–22）。

### 8.5.2 旋钮关断字节等价（knob-off purity）【协议片段→数字】

**契约**：每一个新旋钮默认关，且**未设 == HEAD 逐字节**——不仅 cubin，先在 payload 层面验证（生成器编辑后 mono+fused payload 对 HEAD diff 0 行，r18 `/tmp/r18f_id`）。反向侧按阶段计数：27/27 payload 字节门（r19）、18/18 cubin（r21 `r21b_koff2`）、9/9 × 2 形状（r22）。这一契约使"探针机械不污染舰路径"成为可检查命题：r23 的三个 timing-only 删除探针（GDN_F1T_NKK/NEK/NKM）输出**按设计为垃圾**、从不进 verify——与舰路径的字节隔离是它们合法存在的前提。

### 8.5.3 纯度重建（purity rebuild）【前后对比→效果数字】

**契约**：对生成器做探针/新旋钮编辑后，用**舰组旋钮**从编辑过的树重建，必须精确复现现任记录 cubin 的 md5。执行记录：r18（F1WY 编辑后 mono-T8192 舰复现 `e7af55c1` 逐字节）；r21 三次（探针旋钮编辑后、keep-alive-sink 编辑后、F1ZF 编辑后，T8192 `e7af55c1` ×3 + T2048 `ea8ec2f1`）；r23 两次（NKM/emit_kloop 编辑前后，`e7af55c1`；+F1ZF 舰 `be8b914b`）；r24 三形状全复现 r22 记录（编辑在关断态惰性）；r25 从 HEAD 全新构建 3/3 精确命中 r24 候选。纯度重建同时是**位级等价快速门**的基础：当变更为值恒等类（W32P 的纯索引重排、K32 的链拆分、F1KS 的 K-split），"cubin 变了但输出位级不变"的直接验证（T8192 门 o/ht 0.000e+00、K32 35/35 + 36/36 阵列位等、AW 隔离 dh_all/dh0 位等）比 fp64 重评分更强也更便宜——它把"未改数值"从主张变成测量。

### 8.5.4 cache_key：陈旧缓存孔洞的机械闭合【源码片段→前后对比】

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

### 8.5.5 位级等价门汇表【效果数字】

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
