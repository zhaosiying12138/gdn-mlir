# abstract.md — GDN-MLIR 论文摘要（G7-A 交付物，2026-10-09）

> 双语成稿：英文版用于投稿；中文版存档。数字全部锁定到证据文件（见文末溯源）。

---

## English (submission version, ~200 words)

High-performance kernels for linear-attention operators such as Gated DeltaNet
remain hand-written territory, with scheduling knowledge frozen inside Triton
or TileLang sources and re-paid as on-device autotuning at every new shape. We
argue for **schedule-as-IR**: a stock-MLIR progressive-lowering stack for the
Gated DeltaNet forward and backward passes in which every scheduling decision —
tiling, pipeline depth, warp specialization, fusion boundaries, numeric paths —
is a tunable IR knob (transform-dialect scripts plus dialect attributes),
resolved at compile time by a deterministic Z3 enumeration over a calibrated
analytical cost model with no per-configuration fitted constants. Instead of
trusting the generated code, we validate it structurally: a PTX-level coverage
proof spanning a 33-family instruction contract and both hand-written design
poles (fla/Triton multistage; FlashQLA/TileLang warp-specialized TMA), with an
escape-hatch budget of one custom op out of five allowed. On a consumer
Blackwell GPU (RTX 5090 Laptop, sm_120a), the forward pass beats fla's Triton
kernels on all three benchmark shapes under same-window interleaved
adjudication — 1.592× at B1T2048, 1.05–1.19× at B4T2048 (window variance
disclosed), and 1.253–1.312× at B1T8192 — while offline scheduling ranks
configurations at Spearman 0.8962 and costs 0.8 s of CPU versus 39.8 s of
on-device autotuning (≈50×). The backward pass reaches 0.126× of fla after a
five-round 7.2× self-acceleration; we report its full bottleneck accounting as
an honest open front.

---

## 中文（存档版）

面向 Gated DeltaNet 一类线性注意力算子的高性能 kernel 长期处于手写领地：调度知识
被冻结在 Triton 或 TileLang 源码之中，并以逐新形状的在设备 autotune 形式反复支付。
本文提出**调度即 IR（schedule-as-IR）**：一套面向 Gated DeltaNet 前向与反向的原生
（stock）MLIR 渐进下降栈，其中全部调度决策——分块、流水深度、warp 专化、融合边界、
数值路径——均为可调 IR 旋钮（transform dialect 脚本 + 方言属性），由编译期确定性
Z3 枚举在一次性标定、零逐配置拟合常数的解析成本模型上求解。我们不以"信任生成代码"
了事，而是对其进行结构化验证：PTX 级覆盖证明，涵盖 33 族指令契约与两个手写设计
极点（fla/Triton 多级流水极点与 FlashQLA/TileLang warp 专化 TMA 极点），逃生舱预算
五个自定义 op 仅用其一。在消费级 Blackwell GPU（RTX 5090 Laptop，sm_120a）上，
前向以同窗口交错终判在全部三个基准形状上超过 fla 的 Triton kernel——B1T2048
1.592×、B4T2048 1.05–1.19×（窗口方差已披露）、B1T8192 1.253–1.312×——同时离线
调度以 Spearman 0.8962 的精度排序配置，搜索成本 0.8 秒纯 CPU，对比在设备
autotune 的 39.8 秒（约 50×）。反向在五轮 7.2× 自提速后达到 fla 的 0.126×；我们
将其完整瓶颈账目作为诚实的开放战线如实报告。

---

## 数字溯源（审稿自查用）

| 摘要论断 | 数字 | 证据文件 |
|---|---|---|
| fwd 3/3 ≥1.1×（交错终判） | B1T2048 1.592×；B4T2048 1.05–1.19×（两窗口，配对 A/B −148 µs，fla 中位方差）；B1T8192 1.253×/1.312×（ours 1.126 ms） | `evidence/G5/GATE_REPORT_FINAL.md`；`evidence/G5/final_adjudicate_g5k_run{1,2}.json` |
| fwd 数值 | o ≤6.85e-4 / ht ≤7.93e-4 vs fp64（4/4） | `evidence/G5/fwd_verify_round5.json` |
| 排序质量 | Spearman 0.8962（kkt 0.936 / o 0.829，n=18） | `evidence/G3/validation.json`；`evidence/G3/solver_report.md` |
| 搜索成本 | 我方 ~0.8 s CPU（kkt 368 / o 465 配置全解 0.378/0.444 s）vs fla 39.81 s 在设备（fwd 16.55 s/25 配置 + bwd 23.26 s/23 配置）≈50× | `evidence/G6/search_cost.md`；`evidence/G6/fla_autotune_cost.json`；`evidence/G6/scratch/solver_timing_summary.json` |
| 覆盖 | T1 33 族 8 YES/19 PARTIAL/6 NO（feature 层 15 族 8/6/2）；T2 两极点皆可表达且硬件验证；hatch 1/≤5 | `evidence/G6/coverage_hatch_summary.md`；`evidence/G2/coverage_matrix.md` |
| bwd | 8.15 ms vs fla 同窗 1.03 ms = 0.126×；五轮 58.88→8.15 ms = 7.2× | `evidence/G5/GATE_REPORT_FINAL.md`；`evidence/G5/bwd5_bench2048.json` |
| 平台 | RTX 5090 Laptop，sm_120a，82 SM，无 tcgen05 | `docs/PLAN.md` §7；`evidence/G3/calib_report.md` |

口径声明（摘要脚注候选）：全部 GPU 计时取自共享机会窗口（源 JSON 标注
`shared_gpu: true`）；加速比一律以同窗口交错对内比较为分母；B4T2048 的 1.05–1.19×
区间为两终判窗口的完整披露（我方中位稳定 0.980–1.004 ms，波动归因 fla 中位方差，
另以受控配对 A/B −148 µs 与稳定 min 交叉验证）。
