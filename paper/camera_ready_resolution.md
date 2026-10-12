# Camera-ready TODO 决议文档（G7 十条清点）

Workstream **R26**，2026-10-10。输入：`evidence/G7/GATE_REPORT.md` 判据 5 的十条
camera-ready 缺口清单（"clean-window 判定、FlashQLA bwd 比较、fla autotune 摊销口径、
单卡单架构、形状覆盖窄、无模型级 e2e、成本模型绝对值盲区+Z3 子空间 flake、旗舰正式
计时、无 bib、DRAM 标定为受扰下界"）。编号采用 GATE_REPORT 顺序 1–10（注：R26 任务
书将 #9/#10 对调——其 #10=bib；本文按 GATE_REPORT 原序，#9=无 bib、#10=DRAM 标定，
内容一一对应）。本轮新增测量均在 opportunistic 共享窗口（co-job
pid 3993998，后由监督进程原位重启为 3998048，~8.7–9.0 GB 常驻，从未触碰；共享
`~/.triton`/`~/.tilelang` 从未触碰）。

## 状态总表

| # | TODO | 状态 | 证据指针 |
|---|---|---|---|
| 1 | 干净窗口正式终判（W32P ship，n=30 三形状 + DRAM 刷新） | **OPEN-needs-window**（本轮窗口未开：co-job 即时重启接管）；既有干净/空闲窗口证据链已足论文主张 | `evidence/G5/final_adjudicate_clean.json`（10-09 干净窗口）、`evidence/G5/GATE_REPORT_FULL_REMEASURE.md`（10-10 空闲窗口全量重测）、`evidence/G5/fwd_report.md` round 25（W32P ship 正式 n=30 存档：T8192 1.3493× / B4 1.2211× / T2048 2.1250×） |
| 2 | FlashQLA bwd 比较 | **CLOSED-by-this-round**（更正"出厂无 bwd"的记录：源码有完整 sm120 bwd，出厂默认在本机不可启动，shim 下可跑，取得同窗口 spot 数据点） | `evidence/G7/flashqla_bwd_probe.json`、`tools/g7_flashqla_bwd_probe.py`、`evidence/G5/baseline_report.md` ADDENDUM(R26) |
| 3 | fla autotune 摊销口径（50× 的诚实口径） | **CLOSED-by-this-round**（四口径实测：冷搜索 43.9 s / strict+冷编译 5.67 s / strict+热编译 0.65 s / Triton 自带缓存 1.07 s） | `evidence/G6/fla_amortize_r26.json`、`tools/g6/g6_fla_amortize.py`、`evidence/G6/fla_autotune_cost.json`（存档冷数 39.81 s） |
| 4 | 单卡单架构范围 | **OPEN-paper-work**（句子就绪，见下） | `paper/outline.md` §6 已有雏形；本轮给出定稿句 |
| 5 | 形状覆盖窄 | **OPEN-paper-work**（句子就绪） | `docs/L0_SEMANTICS.md`（LVA/varlen/h0 已定义未硬验证）；`evidence/G5/adjudication_methodology.md`（三正式形状） |
| 6 | 无模型级 e2e | **OPEN-paper-work**（句子就绪） | `paper/contributions.md`（全部主张 kernel 级）；`evidence/G6/failure_longscroll.md` §Amdahl 反噬条目 |
| 7 | 成本模型绝对值盲区 + Z3 子空间 flake | **OPEN-paper-work**（披露句就绪，flake 已在 G6 公开记录） | `evidence/G3/calib_report.md` §2（.ca p0 盲区）、`evidence/G6/search_cost.md`（flake 段）、`tools/gdn_solver.py:187` |
| 8 | 旗舰正式计时状态 | **CLOSED-earlier**（r14 诚实重分类已覆盖：staged 链承载旗舰判据，fused 7.2× 负结果带机理披露） | `evidence/G5/fwd_report.md` round 14 "honest reclassify" 段 + round 25 正式 n=30 |
| 9 | 无 bib | **OPEN-needs-user**（纯写作任务，无测量可做；引用键已留位） | `paper/{outline,abstract,contributions}.md` |
| 10 | DRAM 标定为受扰下界 | **CLOSED-earlier**（10-10 空闲窗口刷新已 de-caveat：DRAM +3.2% / cp.async +11%；句子就绪） | `evidence/G5/GATE_REPORT_FULL_REMEASURE.md` §2、`evidence/G3/calibration.json`（`shared_gpu:false`） |

---

## #2 FlashQLA bwd：决议（更正而非 N/A）

**事实链**（`evidence/G7/flashqla_bwd_probe.json`，env `gdn-tl`，tilelang 0.1.15，
B1T2048H16K128V128，共享窗口）：

1. 本机 checkout（shallow clone @ v0.1.3，da06429，2026-10-09 01:01 克隆，早于
   G5-A 会话）**源码含完整 sm120 bwd 链**（`blackwell_sm120/{fused_bwd,prepare_h,
   cp_bwd}.py`；公共 `chunk_gated_delta_rule_bwd` 已接线，`fused_gdr_bwd` 非 None）。
   G5-A 报告 §2 的"SM120 forward-only (no backward)"作为源码陈述是错的，作为
   **出厂默认可运行性**陈述是对的——见下。已在
   `evidence/G5/baseline_report.md` 追加 R26 ADDENDUM 更正。
2. 出厂默认在 sm_120 **不可启动**：autograd 路径内含的 fwd kernel 请求 112,640 B
   动态共享内存 > sm_120 101,376 B opt-in 上限（`InternalError: Failed to set the
   allowed dynamic shared memory size to 112640`；tilelang 0.1.15——上游 5090 数据
   用 0.1.13 测得）。auto_cp=True/False 两档同样失败。
3. 在 G5-A 已披露的 **fwd block_DV=64 shim** 下（bwd kernel 未改动）bwd 完整运行：
   全算子 `torch.autograd.grad` 温态中位 **1.4348 ms**（min 1.3275，n=10，梯度
   finite）；首调 TileLang JIT 编译 36.9 s（一次性）。同窗口锚点：fla Triton bwd
   温态 1.349 ms（G6 冷/温测量）→ FlashQLA sm120 bwd 与 fla **近似持平**（1.06×
   慢）；我方 staged bwd ~2.15 ms（r16–r22 min co-loaded）= 对 FlashQLA **0.67×**、
   对 fla 0.48×。
4. 口径限定：单窗口 spot 数据（无交错判定、无 JAX 梯度交叉验证）；TileLang 代表性
   的**正式**数据点仍是 G5-A 的 fwd 交错比较；论文 bwd 诚实线（我方反向落后）不变，
   现在同时有 fla 与 FlashQLA 两个参照。

**可直接粘贴段（中文）**：

> FlashQLA 的 SM120 反向在源码层完整存在（v0.1.3），但在本机出厂配置下无法启动——
> autograd 链内的前向 kernel 请求 112,640 B 动态共享内存，超出 sm_120 的 101,376 B
> opt-in 上限（tilelang 0.1.15）。在已披露的前向 block_DV=64 适配（不改动反向
> kernel）下，其全算子反向在 B1T2048 上温态中位 1.43 ms，与 fla Triton 反向同窗口
> 近似持平（1.06×）；我方 staged 反向 2.15 ms，同时落后两个参照（0.48×/0.67×）。
> 该点为共享窗口单点测量（无交错判定与梯度交叉验证），TileLang 代表性的正式数据点
> 仍为 §5.1 的前向交错比较。

**Paste-ready (EN)**: FlashQLA's SM120 backward exists in source (v0.1.3) but cannot
launch under shipped defaults on our machine -- the forward kernels inside the
autograd chain request 112,640 B of dynamic shared memory, above the sm_120
101,376 B opt-in limit (tilelang 0.1.15).  With the disclosed forward
block_DV=64 adaptation (backward kernels unmodified), its full-op backward runs
at 1.43 ms warm median on B1T2048, on par with fla Triton's backward in the same
window (1.06x); our staged backward at 2.15 ms trails both references
(0.48x/0.67x).  This is a single shared-window spot measurement (no interleaved
adjudication, no gradient cross-check); the formal TileLang-representative
datapoint remains the forward interleaved comparison of Sec. 5.1.

## #3 fla autotune 摊销：测量与论文句

**测量协议**（`tools/g6/g6_fla_amortize.py` → `evidence/G6/fla_amortize_r26.json`）：
每阶段独立新进程；fla 侧缓存机制全部核实（`fla/ops/utils/cache.py` 的
`FLA_CACHE_MODE`/`FLA_CONFIG_DIR`；triton 3.8 `Autotuner.check_disk_cache` 的
`.autotune.json` 结果缓存，fla 默认 `FLA_CACHE_RESULTS=1` 开启；
`scripts/extract_triton_autotune_cache.py` 把 Triton 结果缓存转成 fla 发行配置）。
共享 `~/.triton` 未触碰（全部 scratch 目录）。

| 口径（B1T2048，fwd+bwd 首调合计） | 墙钟 | 相对冷搜索 |
|---|---|---|
| 冷搜索（FLA_CACHE_MODE=disabled，全新 Triton 缓存；fwd 17.63 s + bwd 26.28 s） | **43.91 s** | 1×（G6 存档 39.81 s，同量级，本轮更重 co-load） |
| fla 发行配置命中（strict + FLA_CONFIG_DIR），Triton 编译缓存**冷** | **5.67 s** | **7.7×** |
| fla 发行配置命中，Triton 编译缓存**热** | **0.65 s** | **67.5×** |
| Triton 自带 `.autotune.json` 持久缓存（fla 侧零配置，默认 `~/.triton` 即得） | **1.07 s** | **41.0×** |
| 温态稳态（进程内第 2–11 次） | fwd 0.41 ms / bwd 1.43 ms | — |

命中路径验证：命中阶段 **0 个** autotune benchmark 运行（阶段 Triton 缓存内
`.autotune.json` 计数 0）；FLA 配置查找 10 命中 + 1 无害 miss
（`prepare_wy_repr_bwd_kernel` 单候选核，无结果文件可抽，走 Triton 单配置免测路径）。
冷搜索内部归属：首调 fwd 17.63 s 中 16.11 s 在 Autotuner 基准循环内（搜索主导）。

**可直接粘贴段（中文）**：

> 搜索成本比 ≈50× 为冷启动口径：fla/Triton 的首调搜索按进程、按形状键重复支付
> （B1T2048 fwd+bwd 实测 39.8 s，重负载窗口复测 43.9 s，同量级）。fla 自带磁盘缓存
> 可摊销该成本，实测新进程首调（fwd+bwd 合计）为：Triton 持久结果缓存下 1.07 s
> （41×）；fla 发行配置层（FLA_CACHE_MODE=strict）且编译缓存冷 5.67 s（7.7×）、
> 热 0.65 s（67×）。因此诚实表述为"50× 首调、每缓存配置集摊销 7.7×–67×"——其前提
> 是首个进程已支付全量搜索且部署固定缓存目录；我们的 0.8 s 离线求解无需任何缓存
> 基础设施即为 0.8 s，且以固定 seed/rlimit 确定性复现。

**Paste-ready (EN)**: The ~50x search-cost ratio is a cold-start accounting:
fla/Triton re-pays first-call search per process and per shape key (39.8 s
measured for B1T2048 fwd+bwd; 43.9 s re-measured in a heavier co-load window,
same class).  fla's own disk caches amortize it -- fresh-process first call
(fwd+bwd): 1.07 s with Triton's persistent result cache (41x); 5.67 s (7.7x)
with fla's shipped-config layer (FLA_CACHE_MODE=strict) on a cold compile
cache, 0.65 s (67x) warm.  The fair claim is "50x first-call; 7.7x-67x
amortized per cached configuration set", noting that the first process pays
the full search and the deployment must pin a cache directory, whereas our
0.8 s offline solve needs no cache infrastructure and reproduces
deterministically under fixed seed/rlimit.

## #4–#10 就绪披露句（按 GATE_REPORT 原序 #9=无 bib、#10=DRAM）

**#4 单卡单架构（OPEN-paper-work）**

> 本文全部结果在单卡消费级 Blackwell（RTX 5090 Laptop，sm_120a，无 tcgen05、以
> mma.sync 为上限的架构现实）上取得；无 H100/B200 对照，数据中心 Blackwell 的
> tcgen05 词汇未纳入旋钮格——L3 层向新 mma 族的扩展是已界定的未来工作。

**#5 形状覆盖窄（OPEN-paper-work）**

> 实验形状刻意收窄：三个正式判定形状（B1T2048 / B4T2048 / B1T8192，H=16、
> K=V=128）加 T512 验证级；LVA/varlen/非零 h0 在 L0 语义有定义但未硬证据验证，
> 其它 head-dim 无正式判定——新形状的调度决策由离线求解器外推（loopback 设计），
> 依赖 §5.1 的排序稳健性证据。

**#6 无模型级 e2e（OPEN-paper-work）**

> 本工作的交付与主张均为 kernel 级：无模型级端到端（训练吞吐/困惑度）结果；
> kernel 加速向端到端的折算受 Amdahl 约束（线性注意力层占全模型的比重未测），
> 由未来工作承接。

**#7 成本模型盲区 + Z3 flake（OPEN-paper-work）**

> 成本模型的已知边界：模型不含逐配置拟合常数，但 `.ca` p0 串行化幅值等效应未建模
> （盲区，见标定报告）；Spearman 0.8962 度量的是与 18 个实测配置的排序一致性而非
> 绝对时延精度（全部结构常数 ±2× 承动下 ρ∈[0.882,0.902]）；Z3 在 o K=64V=64
> 子空间存在 rlimit 下 partial-model flake，两个主空间（kkt 368 / o 465 配置）以
> 固定 seed+rlimit 字节级复现，flake 公开记录而非隐藏。

**#8 旗舰正式计时（CLOSED-earlier，r14 诚实重分类）**

> 旗舰形状（B1T8192）的正式计时按诚实重分类主张：staged 三 kernel 链承载旗舰判据
> （0.79 ms 级），fused 单 kernel 在移植全部可移植优化后仍慢 7.2×（6.35 ms）——
> chunk 串行链无法吸收 WY 阶段 2048-CTA 的并行性，这是结构差异而非 tile 工程
> 缺口；论文以 staged 数字主张，fused 作为带机理的负结果完整披露。

**#9 无 bib（OPEN-needs-user）**

> 参考文献表未随骨架交付（正文引用键已留位）；投稿前补齐——纯写作任务，无测量
> 成分。

**#10 DRAM 标定受扰下界（CLOSED-earlier）**

> 标定常数为单窗口实测，且任何 GPU 标定天然是受扰下界：干净窗口刷新将 DRAM 峰值
> 824.6→851.4 GB/s（+3.2%，cp.async 路径 +11%）；模型的用法是排序与相对账目
> （±2× 常数承动检验覆盖该不确定性），不主张绝对时延。

## #1 干净窗口终判：本轮窗口记录

R26 全程轮询 `nvidia-smi --query-compute-apps`（含一段 60 s 节律 × 25 样本的连续
记录，09:05–09:29）。观察：co-job 为监督式服务（fp4vla `serve_recovery.py`），
本会话内重启 4 次（pid 3993998 → 3998048 → 4001138 → 4003437 → 4005511），
每次退出后 ≤60 s 即原位重启（两次捕获单样本 EMPTY：09:17、09:28，随即新 pid 接管）。
连续干净样本最多 1 个，从未达到 3 —— **窗口未打开，clean3 未执行**。论文主张不受阻：
交错同窗纪律本身为共存设计（`evidence/G5/adjudication_methodology.md`），且已有
10-09 干净窗口判（`final_adjudicate_clean.json`）、10-10 空闲窗口全量重测
（`GATE_REPORT_FULL_REMEASURE.md`）与 r25 W32P 正式 n=30 存档
（T8192 1.3493× / B4 1.2211× / T2048 2.1250×，`fwd_report.md` round 25）。
clean3 命令留档（窗口开启时先登记协调行）：
`GDN_ZFIX=1 GDN_SCANDB=1 GDN_F1ZF=1 GDN_W32P=1 python tools/gdn_final_adjudicate.py --pairs 30 --warps 4 --bv 32 --stats --gen-root scratch/r25f/genrec --out evidence/G5/final_adjudicate_clean3.json`（前置
`scripts/run_calib.sh`）。

---
*R26 交付物：`evidence/G6/fla_amortize_r26.json`、`evidence/G7/flashqla_bwd_probe.json`、
`tools/g6/g6_fla_amortize.py`、`tools/g7_flashqla_bwd_probe.py`、
`evidence/G5/baseline_report.md` ADDENDUM、本文件。DECISIONS_R25.md 未触碰
（D1/D2 维持 adopted-per-recommendation）。*
