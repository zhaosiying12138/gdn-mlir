# figures_plan.md — 图表清单（G7-A 交付物，2026-10-09）

> 8 件图表（6 图 + 2 数学框/热图），每件注明：呈现内容、面板结构、数据源文件、
> 生成脚本/落盘位置约定。数据源一律为 evidence/ 下的已落盘文件；绘图脚本将置于
> `tools/paper/figN_*.py`，产出 `paper/figures/figN.{pdf,png}` + 生成用的中间
> `.json`/`.csv`（数值不手敲）。

---

## F1（图，§3.1/§3.2）旋钮格 knob-lattice 示意图

- **内容**：L0 语义（`gdn.fwd/bwd` + GdnConfigAttr）→ L1 算法结构（chunked WY
  DAG、solve 变体、recompute/materialize）→ L2 调度（tiles、grid 映射、warps、
  融合边界）→ L3 流水/向量化（multibuffer、async groups、pipeliner、mma.sync、
  warp 专化、补列）→ L4 目标（sm_120a cubin）的五层格；右侧纵列标注每层 oracle
  （JAX fp64 / CPU linalg 对拍 / ptxas+SASS 静态复核 / fp64 孪生逐元素 / fla 容差）；
  左侧纵列标注 Z3 求解器与成本模型作用于 L2/L3 旋钮向量的位置，以及 escape-hatch
  从 L3 直插 L4 的唯一通道（`gdn.mma.tf32k8`）。
- **画法**：分层框图（TikZ），旋钮以层内小标签群表示；四支柱以底色带标注。
- **数据源**：`docs/PLAN.md` §2（层定义与旋钮清单）、`dialect/`（属性面）、
  `evidence/G3/solver_report.md` §1（求解器变量即旋钮向量）、
  `evidence/G6/coverage_hatch_summary.md` §3（hatch 通道）。

## F2（图，§5.2）T1 覆盖热图（33 族 × 三方）

- **内容**：行 = 33 个指令族（feature 层 15 族置顶，baseline 层 18 族折叠为底部
  分组条）；列 = fla / FlashQLA / ours(final)；单元格色阶 = YES / PARTIAL(变体级) /
  NO / -（该参考不含）；ours 列对 mma.sync 标注 hatch#1 翻转（PARTIAL→YES），
  对 stmatrix 标注"测量判非所需"角标。附右侧窄条：T2 两极点资源包络（我方 regs
  包络 [28,255] vs fla [40,255]；setmaxnreg spill 22,844 B→48 B vs FlashQLA 零
  spill 级）。
- **数据源**：`evidence/G6/coverage_hatch_summary.md` §1a–1d（终态矩阵 + 计数）、
  `evidence/G2/coverage_matrix.md` §1/§3/§5（逐族行与资源包络）、
  `evidence/G0/fla_ptx/SUMMARY.md`、`evidence/G0/qla_ptx/SUMMARY.md`（参考侧计数）。
- **注**：色阶图例必须写明 PARTIAL 的定义（族已发射、缺变体拼写/宽度），防止
  "19 个 PARTIAL"被误读为能力缺口。

## F3（图，§5.3）Spearman 散点：预测 vs 实测（18 配置）

- **内容**：x = 成本模型预测（µs，对数轴），y = 实测中位（µs，对数轴），18 个
  stage 配置按 stage（kkt/o）与 knob（mma/pipeline/warps）着色/标记；标注
  Spearman 0.8962（kkt 0.936 / o 0.829）；旁嵌小图：稳健性曲线（结构常数 ±2× →
  ρ∈[0.882,0.902]；计时噪声 ±5/10/20/30% → ρ 中位 0.896/0.890/0.885/0.885，
  p5 0.856）。虚线参考 y=x 揭示 mma 路径 3–5× 系统性低估（秩不变的披露）。
- **数据源**：`evidence/G3/validation.json`（18 配置预测/实测对 + 敏感性试验）、
  `evidence/G3/config_times.json`（实测中位）、`evidence/G3/solver_report.md` §2/§4。

## F4（图，§5.3）搜索成本对比条形图

- **内容**：左组 = 我方（kkt 368 配置枚举 0.307 s / 全解 0.378 s；o 465 配置
  0.379/0.444 s；fwd+bwd 合计 ~0.8 s，纯 CPU，GPU 秒 = 0）；右组 = fla 在设备
  autotune（fwd 冷 16.55 s / 25 配置 / 5 kernel；bwd 冷 23.26 s / 23 配置 / 4
  kernel；合计 39.81 s / 48 配置）；对数时间轴；顶部标注 ≈50×。副面板：暖中位
  自锚定（我方测得的 fla 暖中位 0.491/1.349 ms 落于 G5 基线带 0.316–0.517 /
  1.34 ms）证明该测量的可信性。
- **数据源**：`evidence/G6/search_cost.md` §a/§b 两表、
  `evidence/G6/scratch/solver_timing_summary.json`、`evidence/G6/fla_autotune_cost.json`。
- **注**：脚注写明口径——fla 侧为其设计内首调成本（`FLA_CACHE_MODE` 未设 =
  DISABLED，按进程按形状键重付）；我方 0.8 s 为 stage 旋钮空间排序成本，
  full-op 发运另用配对 A/B（工程归因非搜索）。

## F5（图，§5.1）前向加速比条形 + 交错对分布

- **内容**：主面板：3 形状 × 加速比条（B1T2048 1.592×；B4T2048 两窗口 1.052×/
  1.189× 以双条+区间须呈现；B1T8192 1.253/1.312×），1.0× 与 1.1× 判决线；副
  面板 A：B4T2048 的交错对级分布（60 对 per-pair 比值直方图，两窗口分色）+
  配对 A/B −148 µs 标注 + 我方/基线中位与 min 稳定性表；副面板 B：五轮轨迹
  staged fwd 1.597→0.663→0.262→0.215/0.373 ms（杠杆标注：K-循环流水 2.2–2.7×、
  LDSM 补列、w4/bv32、F1UW）。
- **数据源**：`evidence/G5/GATE_REPORT_FINAL.md`（终判表）、
  `evidence/G5/final_adjudicate_g5k_run{1,2}.json`（交错对原始值）、
  `evidence/G5/final_adjudicate_w4bv32_run{1,2}.json`、`evidence/G5/final_adjudicate_g5i_run1.json`、
  `evidence/G5/fwd_bench_round5.json`、`evidence/G6/knob_ablation.md` §2（轨迹与杠杆）。
- **注**：每条形脚注标窗口口径与 `shared_gpu` 披露；B4T2048 的方差解剖引用
  `evidence/G6/failure_longscroll.md` §10。

## F6（图，§5.5）反向五轮轨迹长卷线图

- **内容**：x = 轮次（r1–r5），y = e2e ms（对数轴）：58.88 → 14.16 → 11.06 →
  8.94 → 8.15（B1T2048H16），每段标注本轮杠杆（mma 化 4.2× / tf32 hatch 1.28× /
  dkb split + async staging / dhs 携带扫描重构 + split-bf16 3.8×）；水平参考线 =
  fla 同窗 1.03 ms（标注最终 0.126×）；右端叠柱状分解终态瓶颈账目（dqkwg 1.71 /
  wy 1.45+ / wyb 1.04 / dhs 0.59（天花板 ~0.3 标注）/ dkb 0.25 / 其余）。阴影带
  标注各轮主要负结果（tf32-dhu 双负、async 零投递悬案、段分裂 Amdahl 反驳——
  后者为 fwd 但同轮时间线，可作时间轴下方的事件轨）。
- **数据源**：`evidence/G5/GATE_REPORT_FINAL.md`（轨迹与账目）、
  `evidence/G5/bwd{2,3,4,5}_bench2048.json`（各轮原始计时）、
  `evidence/G5/bwd_report.md`（逐轮杠杆与负结果）、`evidence/G6/knob_ablation.md` §2。

## F7（图，§5.4）两级消融热图（旋钮 × 效应量）

- **内容**：级一（stage 级）：行 = 旋钮（mma、pipeline、warps、tile M/K/N、dtype、
  warpspec、tma），列 = 效应量（HMMA 数、LDGSTS、LDS、spill KB、PTX KB、GPU µs
  比），单元格 = 对数色阶的前后比；关键数字定格（mma 开关 50–100×、p0→p2 5×、
  w2<w4<w8 反直觉、f32/tf32 1.6×）。级二（full-op 级）：行 = 杠杆（K-循环流水、
  staging 线程跨距、ping-pong、LDSM 补列、w4/bv32、段分裂、barrier 2→1、F1UW、
  tf32 位点、split-bf16、dhs 重构、rgemm 序），列 = Δms 与判定（win/negative/
  neutral/structural 色点）——负结果行不隐藏，与 win 同图呈现。
- **数据源**：`evidence/G6/knob_ablation.md` §1/§2（全表）、`evidence/G2/gen/INDEX.json`、
  `evidence/G3/config_times.json`、`evidence/G5/fwd_bench_round{2,3,4,5}.json`、
  `evidence/G5/bwd{2,3,4,5}_bench2048.json`。

## F8（数学框，§2.1/§3.6）dA 双侧公式与携带扫描框

- **内容**：双栏数学框（LaTeX box，非数据图）。左框《dA 的精确反传》：
  $$\mathrm{d}A = -\operatorname{stricttril}\!\big(T^\top \operatorname{stricttril}(G)\,T^\top\big),\quad
    G = \mathrm{d}v_p(\beta v)^\top + \mathrm{d}W(\beta k e^{gc})^\top,\quad
    \mathrm{d}W = -\mathrm{d}v_p\, S_n^\top$$
  注记：先掩蔽再两侧乘 $T^\top$；单侧修正式对 $C\ge 3$ 不成立（符号级验证）；
  与 fla `prepare_wy_repr_bwd`（计入转置 A 存储）等价。右框《反向携带的仿射形与
  超收缩》：$X_{n-1}=(2^{g_{cl}}I - W^\top\hat{B})X_n + (Q_g^\top - W^\top P^\top)
  \mathrm{d}O$；实测对真序贯迹 2–3 ulp（fp64，全分段 + 近单位门 stress）；
  $\rho(A_n)\le 2.3\times10^{-18}$、$\|A_n\|\le 2.2\times10^{-17}$（弱门安全的
  结构性保证）。
- **数据源**：`evidence/G1/GATE_REPORT.md` §数学成果（dA 公式与验证）、
  `evidence/G1/l1_harness_report.md`（推导与符号验证）、`evidence/G5/GATE_REPORT_FINAL.md`
  §bwd r5（携带式与 ulp/收缩界实测）。

---

## 表格清单（正文表，随章节排版；非图版）

| 表 | 位置 | 内容 | 数据源 |
|---|---|---|---|
| T1 | §3.4 | sm_120 一次性标定表（8 速率 + 平台参数 + shared_gpu 注） | `evidence/G3/calib_report.md` §1、`evidence/G3/calibration.json` |
| T2 | §3.3 | Z3 变量/约束/目标规格表 + 合法空间规模（368/465）与确定性契约 | `evidence/G3/solver_report.md` §1 |
| T3 | §5.1 | 前向终判表（3 形状 × 加速比 + 两窗口披露 + 数值列） | `evidence/G5/GATE_REPORT_FINAL.md` |
| T4 | §5.6 | 数值总账三层表（语义闭合 44/44 与 6e-7；stage 20/20 与 9.77e-4==预测；full-op 4/4 与 6/6） | `evidence/G1/GATE_REPORT.md`、`evidence/G4/GATE_REPORT.md`、`evidence/G5/GATE_REPORT_FINAL.md` |
| T5 | §4.3/附录 C | 开放发现账本表（7 条 stock-MLIR 缺陷：现象/探针/根因/修复/验证） | `evidence/G4/GATE_REPORT.md` §2、`evidence/G3/calib_report.md` §2、`evidence/G6/failure_longscroll.md` |

## 制作约定

1. 绘图脚本只读 evidence/（或其派生的中间 json），禁止手敲数值；每个图目录留
   `data.json` 快照保证可审计。
2. 所有含 GPU 计时的图/表统一脚注："shared opportunistic window; medians flagged
   `shared_gpu` in source JSONs; ratios are within-window interleaved unless noted."
3. 色盲安全色板；对数轴必须标刻度值；每图 caption 自含口径（判据、分母、窗口数）。
