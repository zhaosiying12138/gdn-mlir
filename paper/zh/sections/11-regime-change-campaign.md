# 11 体制变化战役：标度律、误差预算与精度关闭定理

> 覆盖对象：反向链在 r19-final 冻结（第 6–7 章、第 9 章口径）之后发起的**体制变化战役（regime-change campaign，回合 RC1 → R4，2026-10-08 ~ 10-12）**全部四轮的方法学与结论——CPU 预测定价器 `tools/rc1_price.py` 与实测 B/T/H 阶梯、bf16 误差预算探针 `evidence/G5/scratch/r2_bf16_probe.py` 与 16 符号分类表、A_all 退役探针 `evidence/G5/scratch/r3_probe.py` 与 dhs 精确坍缩、tgemm a-fragment hoist、融合定价器 `evidence/G5/scratch/r4_price.py` 的三重杀；全部数字取自 `evidence/G5/{rc1_scaling_predictions,rc1_ladder_report,r2_bf16_globals_report,r3_report,r4_fusion_report}.md` 及其引用的 scratch 日志。本章按**最终系统与最终结论**呈现：四轮的杠杆全部以装机形态（或决策级 NO-GO 判定）讲述，历史演化只以"结构前/后对比"出现；测量口径延续第 9 章——正式数字一律为同窗交错/三重交错配对中位数（n=30 轮取中位，每 cell 三对），数值门槛为 fla 派生契约 dq/dk/dv ≤ 8e-3、dbeta/dg ≤ 2e-2、dh0 ≤ 8e-3，确定性契约为新起进程位级一致（N=3）；跨窗口绝对毫秒只作方向性参考（每轮披露窗口状态）。

## 11.0 章首：为什么在"冻结"之后又开一场战役

### 11.0.1 冻结的语境与被推翻的决策

第 6–7 章的反向链在 26 轮自提速后冻结于 **r19-final bytes**：e2e 从 58.88 ms 提到 2.15 ms（约 27×），对 fla 的同窗交错比长期停在 ~0.48×。第 9 章的判定方法学与 `evidence/G5/DECISIONS_R25.md` 决策 D2 把这个状态正式化为"**持有（hold）**"：值恒等类杠杆经 r22 穷尽、新数值类杠杆经 r24 W-park 决策级 NO-GO，在 B1T2048H16 / B1T8192H16 两个形状上，"还能更快吗"的每一类想法都已定价或实测关闭；若反向工作重开，**唯一有未测量余量的方向是体制变化（regime change）——B4+/T-other/H-other 等形状外推**。D2 的签署状态是 *adopted-per-recommendation（用户未响应，欢迎推翻）*——协议里明确写着：任何后续回合的用户显式推翻都直接取代该记录，无需重做任何测量。

战役正是这个推翻的发生。用户在 2026-10-08 行使了 D2 预留的推翻权，指令反向链**"全覆盖全部 fla 支持的形状并取胜"**（fully cover ALL fla-supported shapes and win），并给出了一个候选阶梯：B∈{1,2,4,8} × T∈{2048, 8192, 16384, 32768} × H∈{16,32}。这条指令把战场从"两个锚形状上的同窗比值"扩展为"整条 B/T/H 体制空间上的胜负判定"，因此得名体制变化战役。它隐含两个待裁决的物理命题：

1. **趋势命题**：ours-vs-fla 比值是否随形状增大而改善（82 SM 填充、波尾摊销、fla 的逐形状 autotune 税是否会在大形状上反噬 fla）？是否存在任何体制达到 ≥1.1×？
2. **覆盖命题**：<6 GB 显存协调规则（co-job 常驻 ~8.7–9 GB）下，阶梯的哪些格点可测？曲线能否被有限个格点约束？

战役的回答方式继承了第 9 章的方法学纪律，并把它推到极致：**先 CPU 后 GPU、先预测后实测、先预算后落地、先纸面定价后写码**。四轮的分工是：RC1 用标度律框架在 GPU 触碰之前给整条曲线定价并实测对拍（§11.1）；R2 把"字节减半"变成可审计的误差预算合同并落地 9 符号默认集（§11.2）；R3 用预算探针发现 A_all 数值死、退役 caryA、把 dhs 坍缩为移位拷贝，并补上最后一刀 CVT 税（§11.3、§11.4）；R4 对仅存的字节地板推动者（dqkwg+wyb 融合）做纸面三重杀，把战役以精度关闭定理收官（§11.5、§11.6）。

### 11.0.2 结局先行：本章要论证的三句话

先定义后用，三个贯穿全章的术语在此钉死：

- **体制（regime）**：以 $u = M/512$（$M = B\cdot H\cdot NC$，$NC = T/64$，即 B1T2048H16 锚的工作倍数）为坐标的工作量区间。**小形状体制**指 $u \ll 1$（两侧都被固定成本支配）；**大形状体制（u ≥ 1）**指 $u \ge 1$ 的全部实测格点（两侧边际成本主导）。
- **精度关闭（precision closure）**：一个性能命题不是被"执行力"（尚未到达屋顶线的差距）关闭，而是被**误差预算本身**（哪些字节在合同上必须保持 f32）关闭。关闭后的比值天花板是合同的数字，与实现质量无关。
- **同窗配对货币**：跨窗口绝对毫秒不可比（共载漂移 ±5–20%），一切 before/after 以同窗交错（RC1/R2：ours-fla 成对；R3：r2-链/r3-链/fla 三重交错）的配对中位数为货币。

本章按最终结论叙事，三句话概括：

1. **大形状反向的 ≥1.1× 目标被三重独立证据关闭**——误差预算地（§11.2，可铸字节被预算判死）、地板算术地（§11.6，完美执行当前字节集的天花板 0.85–0.92× @u=8）、融合地（§11.5，最后的字节搬运者被池/占用算术判 NO-GO）。最终战绩：u ≥ 1 全阶梯 **0.41–0.49×**，边际成本比 0.47×——**这是误差合同的数字，不是我们的数字**。
2. **战役在两个体制上全胜**：小形状 B1T512H4 **1.72×**（同窗配对，R3 窗口；从 R2 窗口的 1.55× 深化而来）；**冷启动全形状胜**——AOT cubin 零 autotune、~27 μs 发射开销，对 fla 每形状 48–69 s 的冷 autotune 税。
3. **战役本身把大形状反向从 0.39–0.61× 推到 0.41–0.49×**（u=8 轨迹：0.396 → ~0.43 → 0.462，边际 2.00 → 1.488 ms/u），全部杠杆位恒等或预算内（逐字节等价证明见 §11.3.5）——关闭定理不是"放弃"，而是把可动与不可动的边界钉死到数字。

### 11.0.3 战役的窗口纪律与门槛栈

四轮的门槛栈是第 8–9 章体系的全量继承，每轮全过：payload 溯源（逐 cell 从 HEAD 再生成 bit 一致）、cubin 谱系（md5 对 r21b/R2/R3 记录）、数值 6/6（fp64 孪生、种子 0，全部 6 阶梯 cell）、确定性（新起进程 N=3 位级一致）、显存（<6 GB 规则、co-job 绝不触碰）。窗口状态按轮披露：RC1/R2 共载窗（GPU 常驻 ~8.9–9 GiB、16–28% util），R3 全程干净窗（0 共载），R4 仅做位恒等健康认证（无计时主张，共载不适用）。跨轮绝对数字的方向性比较始终以"每轮自窗配对臂"的口径给出。

---

## 11.1 标度律框架：在 GPU 触碰之前给整条曲线定价

### 11.1.1 动机：把"形状外推"变成可证伪的预测

D2 留下的开放问题只有一个方向——体制变化——但"换个形状再测一次"不是裁决级的工作方式。第 9 章的方法学要求：**任何 GPU 测量之前，先有一个可证伪的预测**。RC1 的第一步因此不是上 GPU，而是从代码的 grid/tile 结构加两个已测锚点（B1T2048H16、B1T8192H16 的 r19-final 装机记录）推导出整条 B/T/H 曲线的解析形式，再从中选出信息量最大的格点去实测。这一节讲清楚三件事：标度律的推导（每 kernel 一条仿射律）、预测定价器 `tools/rc1_price.py` 的构造（预测如何被机器化）、以及 6 cell 实测对拍的判定（预测对了几成、错在哪、为什么）。

### 11.1.2 两类 kernel、两条标度律

先定义后用：九 kernel 链（第 6 章 §6.0.2 拓扑）按 grid 公式分成两类，两类的时间随形状的走法不同。

**逐块并行类**：wy、caryA、caryM、bpre、dkb、dqkwg、wyb，grid = $M = B\cdot H\cdot NC$，每 CTA 处理一个块行。这类 kernel 的时间是工作量的仿射函数

$$t_s(u) \;=\; a_s + b_s\cdot u,$$

其中 $b_s$ 是 DRAM 受限的边际（每 M 行全局量都随 $M$ 线性增长），$a_s$ 是发射+填充残差。锚点拟合给出九个阶段的 $(a_s, b_s)$（`evidence/G5/rc1_scaling_predictions.md` §1 的完整表：如 wy −0.031/0.254、dqkwg 0.032/0.430、caryA −0.033/0.086 ms）。两个结构性修正：其一，**波量子化**只在边缘改善（$u{=}1$ 时 6.2 波 → $u{=}4$ 时 25 波，尾部效应 <4%）；其二，**L2 足迹罚**——32 符号 M 行全局量随 $M$ 增长把小足迹阶段（caryA/caryM/bpre）挤出 L2，T 方向 4× 工作量实测 5.9–6.2× 时间（超线性），mma 类阶段只有 3.3–3.4×（亚线性）。所以仿射律在 B/H 方向同样适用（足迹随 $M$ 增长不分方向），但边际略差于 u=1 平均。

**串行链类**：fwdscan（grid $= B\cdot H\cdot 4$，NV = V/32 个 v-切片各自走 NC 步）与 dhs（grid $= B\cdot H\cdot 8$，BV=16 装机）。T 方向：链长 NC 增长，实测 4× 工作量 4.08×/3.91×（近线性，含 L2 增长）；B/H 方向：链长不变，makespan = $\lceil \text{grid}/82\rceil$ 波 × 同样的链——B1H16 的 fwdscan 64 CTA 只有 0.78 波（18 个 SM 闲置），H32/B≥2 把它填满，预言一次性的 ~1.28× 利用率红利（约 0.64 ms 的链上），之后线性。

发射开销：9 kernel × ~3 μs ≈ 27 μs 固定，约占 u=1 e2e 的 1.3%——这是小形状体制的"我方固定成本"。

fla 侧同构处理：从干净窗派生序列（B1T2048 1.1006、B1T8192 2.8379 ms）拟合 **fla(u) = 0.522 + 0.579·u**。fla 的 bwd kernel 解剖（profiler 归因，~10–12 个发射：dqkwg 137.6 μs、dhu 132.5、prepare_wy_repr_bwd 96.3、前向重算 76.2、逐元素 Mul 56.7…）grid 同样在 $B\cdot H\cdot$(V-块) 上并行，同 M 标度；它的 ~0.52 ms 截距是 autograd 逐元素链 + 10 余次发射的固定成本。

### 11.1.3 预测定价器的构造：`tools/rc1_price.py`

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

### 11.1.4 预测：趋势与"任务希望"相反，且有结构下界

定价器的输出（u = 1/2/4/8/16 → 0.514/0.419/0.366/0.338/0.324，渐近 0.310）给出**与指令希望相反的预测**：比值随 B·T·H **单调下降**。我方的每 u 边际 1.837 ms vs fla 的 0.579 ms——**每 token 成本 3.17×**——阶梯每上一级，比较只会更差；fla 的亚线性固定成本（~0.52 ms）只在小形状托举我们。预测同时给出地板论证，把"任何执行质量下 ≥1.1× 是否可能"变成算术：

- r7 静态分析（第 9 章口径）的九阶段 DRAM 地板 = 1.074 ms @u=1，且全部阶段流量随 $M$ 缩放 → floor(u) = 1.074·u ms；
- u=1：地板 1.074 vs fla 实测 1.03–1.10——一条**完美屋顶线执行的 f32 分段链**在锚点也只是打平；
- 屋顶线完美的天花板比 $\frac{0.522+0.579u}{1.074u}$：u=1 时 0.81、u=4 时 0.65、渐近 **0.539**；
- 要在任何 $u\ge 1$ 达到 1.1×，需我方斜率 ≤ 0.526 ms/u——**比 f32 流量地板低 2.04×**。这不是执行问题：**字节必须减半**（bf16 中间量，受误差预算门控；或融合减趟）。

唯一 ≥1.1× 有可能成立的体制是 $u \ll 1$（小形状两侧都是固定成本）。定价器据此选出 6 个信息量最大的 cell：锚 B1T2048H16（cubin 必须复现 r19-final md5）、B1T2048H32（H 方向，检验串行填充红利）、B4T2048H16（B 方向 u=4，链从未在 B4 实测）、B1T8192H16（T 方向 u=4 重锚 + 新鲜交错 fla）、B1T16384H16（u=8 远端，NC=256 新类型 → 数值验证 cell）、外加 B1T512H4（u=1/16 小端唯一候选胜点）。u ≥ 16 的任务阶梯格点（T32768、B8T8192H32 等）全部是 M=8192+ 的 6.5+ GiB cell——被 <6 GB 规则排除，且预测落同一条平台（refit 0.38 @u=16），测量它们不增加判定信息。

### 11.1.5 实测对拍：曲线的四个判定

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

### 11.1.6 阶段分解：超线性类的普遍化与 R2 的弹药

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
## 11.2 bf16 误差预算方法学：把"字节减半"变成可审计的合同

### 11.2.1 动机：预算先行，铸造后行

RC1 的算术说清楚：u ≥ 1 的胜负在字节里，不在执行里。但"bf16 化全局量"在本项目里从来不是一个开关，而是一份**合同**——第 8 章的容差契约（dq/dk/dv ≤ 8e-3、dbeta/dg ≤ 2e-2、dh0 ≤ 8e-3，判据按 bars/2 保守）决定哪些中间量可以在存储上粗化、粗化到什么程度。r16 的 bf16-solve 先例已确立方法学模板：**CPU 误差预算先行，然后铸造，然后交错阶梯**。R2 把它推广到全部 32 个 M 行符号，并回答三个问题：每个符号单独粗化的增量是多少（leave-one-in 分类法）？可铸集合的联合增量在门槛内吗（集合级投影）？被预算杀死的字节还有别的通路吗（split-bf16 与融合的分界）？

### 11.2.2 重构 fp64 孪生：读写边界精确复制

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

### 11.2.3 leave-one-in 分类法：16 候选符号的完整判定表

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

### 11.2.4 A_all 数值死的发现

分类表里最反常的一行是 A_all：solo 增量**恰好为零**（dk 0、dbeta 0），探针测到的端到端增量在 1e-27 量级。原因在门分布的结构：基准配方 $g$（E[g] ≈ −0.69 自然对数/token），64-token 块的块末累积 $2^{gcl} \sim 10^{-19}$，于是

$$A \;=\; 2^{gcl} I - W^T \hat B \quad\Rightarrow\quad |A| \sim 10^{-18},$$

dhs 携带链完全被 B 项支配——caryA 与 dhs 的 A 窗口机件在算**死重**。这个发现当时被记录为"R3 的退役杠杆"（§11.3），它与第 6 章的门算子超收缩 ρ(A) ≤ 2.3×10⁻¹⁸ 是同一现象的两个视角：前者从数据体制看（这个 g 下 A 有多小），后者从算子谱看（任何合法 g 下 A 收缩得多快）。

### 11.2.5 split-bf16 零收益定律：读者就是紧的

被杀的 470 KiB/M 行能否用 split-bf16（hi+lo 双链，第 3 章/第 6 章的三链接乘积类）救回？探针给出的答案是结构性的**否**：split-bf16 的代价是 4 B/元素（两个 bf16）——**对流量的节省为零**；它的价值只在"乘法质量"（hi·hi + hi·lo + lo·hi 恢复 ~f32 乘积）。但预算杀这 5 个符号的原因不是乘法质量不足，而是**它们的消费者对值本身紧**：u_all/vnew_all 喂 dkB（O(10) 相消到 O(0.1) 的病态和）、dvp_all 喂 dbeta（4.8e-2 超门槛）、dk_local/dkB_sum 是同阶级。**读者就是紧的**——在消费端做任何双链分解都改变不了"值必须以 f32 分辨率到达求和点"这个事实。因此这 470 KiB/M 行的字节**只有一条通路：删边（融合）**。这个分界在 R4 被正式定价（§11.5）。

（一个例外记录在案：R3 的 split-S 探针把 fwdscan 的 wS 操作数 S 拆 hi+lo，dk 增量 2.6e-8/6.6e-8——数值上远比 9 符号集宽裕，说明 S 的读者**不**紧；但它被池封锁——S_hi/S_lo 加 16 KiB 把 fwdscan 的 99072 B 池推过 101376 B 静态上限。数值余量记录备档，风险未取。）

### 11.2.6 落地：逐符号 dtype 分派与三条建链定律

实现落在 `tools/gdn_bwd_gen.py` 的 `GDN_BF16G` 旋钮（默认 9 符号集 = 10 可铸减去 A_all；`"none"` 逐字节恢复 f32 全局量）。dtype 是**符号的属性**：每个 `k.global_` 与类型串位点经 `gdt()`/`gty_()` 分派；生产者 `truncf` 落盘、消费者 `extf` 读回，穿过既有的 gload/gstore/stage_rows/stage_split_*/rgemm/aggemm/gcol 机件（全部 dtype 感知）。cp.async 位点二选一：读者是 load2-generic 的（dhs 的 A 窗/Xb、dkb 的 dHv ping-pong、fwdscan 的 wbf、dqkwg 的 tTr/dWt0/dWt1）直接把落地 tile 翻 bf16（r6 合法）；读者是**计算 tile** 的（wy 的 solve tile Tf、fwdscan 的原地 kg 扫描 wpp）不能翻——那里 bf16 化既是数值改变（kg 粗化未探）也是类型错误。落地途中沉淀三条**建链定律**：

1. **Tf/wpp 是计算 tile**——solve 与原地扫描**写**它们；bf16 翻转是数值变更 + 类型错误，保持 f32。
2. **标量 GEMM 的 CVT 税**——bf16 tile 喂标量 FMA 的 gemm 循环（fwdscan 的 wS、dhs 的 rgemm_aw 窗口）在 MAC 循环内付**逐元素** bf16→f32 转换：配对二分定罪（A_all 单独翻转 → dhs 0.52 ms vs f32 0.27 ms，即 0.52×）；mma/ldmatrix 消费者原生接受 bf16 操作数而**赢**（bpre 1.30–1.50×、caryA/M、dkb、wy）。A_all 因此**默认关**（它唯一的消费者是被税的 rgemm_aw；何况它数值死）。
3. **串行关键路径上的转换扫描比税更差**——fwdscan 的 sweep 变体实测 0.49× vs bf16-tile 直读 0.90×；读者能展宽的地方，异步落地 tile + 原生宽读才是正确形状。

门槛栈全过（`GDN_BF16G=none` 对 RC1 链 9/9 payload+manifest **逐位一致**；数值 6/6，最差 B4 dk 5.04e-3 为门槛 1.6× 裕度、T16384 3.92e-3 与 f32 记录 3.80e-3 几乎同位；dh0 2.0e-4 处处与探针增量**精确相等**——探针即实测）。

### 11.2.7 效果数字：字节赢了，时间还没赢

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
## 11.3 A_all 退役与 dhs 精确坍缩

### 11.3.1 动机：从"免费字节"到"整段死代码"

R2 留下的两条线索在 R3 汇合：A_all 数值死（§11.2.4）意味着 caryA 阶段（0.55 ms @u=8）与 dhs 的 A 窗口机件（2.3 ms 中约 0.7 ms）在**算死重**；而 dhs 的 rgemm_aw 恰是 CVT 税最重的位点。杠杆本身是**算法变更**（A := 0），按项目纪律必须自带全套门槛——CPU 预算探针先行，量化"退役到底丢掉什么"。

### 11.3.2 探针：max|A@dH| = 2.3e-18 的推导

`evidence/G5/scratch/r3_probe.py` 的 `astats` 模式直接测量退役丢弃项的量级：在 fp64 孪生上重算 W、沿真实 dH 走行走一遍，逐步记录 $\max|A|$、$\max 2^{gcl}$、**丢弃项** $\max|A\cdot dH|$ 与被馈送的**状态** $\max|dH|$：

| 形状 | max abs A | max 2^gcl | **max abs A@dH（丢弃项）** | max abs dH（馈送状态） |
|---|---|---|---|---|
| T512H4 | 1.59e-19 | 1.84e-19 | 1.90e-19 | 6.8e-2 |
| T2048 | 2.01e-18 | 2.34e-18 | 2.26e-18 | 8.3e-2 |

丢弃项比它馈送的状态**低 16 个数量级**——也比 dh_all 自身 bf16 存储 ULP 低 ~15 个量级（这解释了为什么退役连 dh_all 的**字节**都不动，见 §11.3.5）。端到端增量（`aret` 模式，A := 0 于孪生 dhs 行走）在全部 5 个探针形状上 dk ≤ 5.6e-17、dh0 ≤ 1.3e-21——对 8e-3/2e-2 门槛低 **14+ 个数量级**；且 `aret + default9` 与 `default9-only` 在每个形状**逐位相同**——退役叠加在 R2 已装机的预算上**零增加**。探针对 GPU 的预测精确兑现（组合集预测 vs 实测：T8192 dbeta 3.449e-3 vs 3.444e-3；T16384 2.694e-3 vs 2.691e-3；dh0 2.0e-4 精确）。

### 11.3.3 实现：caryA 删除 + dhs 坍缩为移位拷贝（源码与 IR 前后）

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

### 11.3.4 5.9× 的结构原因

阶段配对中位（u=8，T16384）：**dhs 2.083 → 0.354 ms = 5.89×**，外加 caryA 整段删除（0.557 ms → 0）。5.9× 不是调优出来的，是**结构原因**的必然：旧 dhs 的每步 = A 窗口 GEMM（[K,K]×[K,BV] 寄存器累加，NW 次）+ 双份状态 staging + A/B 两次全局读 + 一次拷贝扫描；新 dhs 的每步 = 一次 16 B/线程的 B 突发 + 一次展宽存储扫描——串行链上每步的关键路径从"多次依赖链 GEMM + 两轮 barrier"缩到"DMA 与存储互掩的一来一回"。链上剩下的纯粹是带宽与发射：0.354 ms @T16384 对应的就是 B_all 读 + dh_all 写的字节流。同样地，被税路径（rgemm_aw 的 CVT 税）随 A 窗口一起消失——退役顺带免掉了 §11.2.6 定律 2 里最重的税点。

### 11.3.5 逐字节等价证明

算法变更的收尾门槛是**位级对照**：单次确定性运行同时跑 R2 链与 R3 链（T16384），**每个公共阶段输出逐字节一致**——包括 fwdscan 的 h_all/vnew_all（§11.4 的 hoist 证明）**和 dhs 侧全部符号**。dhs 侧的等价需要一个论证：退役丢弃的项（~1e-18）比 dh_all 自身 bf16 存储 ULP 低 ~15 个量级，因此 $A\cdot dH$ 的舍入不可能改变任何 dh_all 存储位——两条链**收敛到相同字节**；A_all 只存在于 R2 根。其余门槛：`GDN_ARET=0 GDN_TGEMM_AH=0` 对 R2 链 9/9 payload+manifest 逐位一致；未触碰阶段的 cubin md5 与 R2 记录相等（48-md5 全录）；数值 6/6 与 R2 记录**同位**（B4 dk 5.036e-3、T16384 3.819e-3）；确定性 N=3 ×3 形状位级一致；VRAM 高水位 6.56 GiB 中 2.10 GiB 为既有基线、链本身 ~4.5 GiB < 6 GB。

### 11.3.6 效果数字（与杠杆一的合成）

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

## 11.4 tgemm a-fragment hoist：CVT 税的最后一刀

### 11.4.1 动机：机器级地看，税从哪来

§11.2.6 定律 2 的税有明确的机器级形态。fwdscan 的 wS gemm 是 tgemm（tf32k8 hatch 类）消费者：bf16 tile 喂进标量 FMA/tf32 的 k-循环后，**每个操作数元素在每次被一个 n-tile 复用时都要重付一次 extf（bf16→f32 转换）**——R2 实测 fwdscan 整段 0.913×（税 ~0.35 ms @u=8）。对照 mma/ldmatrix 消费者：`mma.sync` 与 `ldmatrix` 在硬件上**原生接受 bf16 操作数**（第 3 章的 mma_gemm 路径），无逐元素转换。税不是"bf16 慢"，是"**bf16 喂错了消费者**"。杀税的两条路线在 R3 都被探针检验：split-S 数值 GO 但池封锁（§11.2.5 尾注）；**值精确的 hoist** 是所缺的第三条路——不改变任何数值，只改变**谁来加载/展宽 a-fragment**。

### 11.4.2 hoist 的构造：warp 地图从轮转改为连续

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

### 11.4.3 位恒等论证

hoist 的合法性一句线：warp 地图从轮转（ti = mW + i·W）改为连续（ti = mW·tpw + i）只改变**哪个 warp 拥有哪个 cell**，而 per-cell 的 k-链保持 kb 升序、原子集合相同——按 r20 warp-map 定律（cell 的算术与归属解耦），**值位恒等**。这不需要概率论证：GPU 上的证明就是 §11.3.5 的链间位级对照（fwdscan 的 h_all/vnew_all 在两链间逐字节一致）。

### 11.4.4 效果数字：1.115× 翻正，税全数回收

fwdscan 阶段 **3.266 → 2.930 ms = 1.115×**（u=8 配对中位）；隔离臂上 fwdscan 3.282 → 2.930 = **1.120×，~0.35 ms @u=8——恰是 R2 实测的 CVT 税全额回收**。至此 R2 的税表清零：bf16→原生消费者的翻转全部完成（bpre/dkb/wy/cary 类原生赢；fwdscan 经 hoist 翻正；dhs 经退役免役），fwdscan 从 0.913× 的输家变为 +11.5% 的赢家。战役的 u=8 比值 0.462、边际比 0.472 是税后净额。

---
## 11.5 融合定价学：dqkwg+wyb 的纸面三重杀

### 11.5.1 动机：最后一个字节地板搬运者

R3 收官时按证据排序留下两个候选：#1 **dqkwg+wyb 融合**（"唯一剩余的字节地板搬运者"：两阶段合计 5.40/12.15 ms @u=8，删边能触及被预算杀死的 f32 边 dvp_all/dk_local 的重读）；#2 扫描原生性（~0.1–0.2 ms 级，§11.7 记录不定价）。按 r14/r16 纪律，融合**先纸面定价**：字节算术来自 L1 数学与装机链、池算术来自实测 smem_fix 池、占用算术来自 r15 实测 1CTA 因子——全部输入为测量量，无一个拟合参数。定价器 `evidence/G5/scratch/r4_price.py`；边图审计 `scratch/r4_audit.py`；结论 **NO-GO（决策级）**，战役链保持 r3-final bytes。

### 11.5.2 边图审计：融合的合法杀伤集

融合能删的边必须**只有** dqkwg 与 wyb 踏踩。`r4_audit.py` 对装机 r3 链的 manifest + payload 做逐符号 load/store 审计：dqkwg 的八个输出里恰一个终结（dq——被评分输出，f32，无论如何要写），其余七个（dW_all、dvp_all、dk_local、r3_c/r4_c/c4r_c/dg_state）全部流向 wyb，且 dW/dvp 还喂 dqkwg 自己的后续相位（TW 循环重读 dW_all、dW 循环重读 dvp_all）；第三个 stage 不触碰其中任何符号——**边图闭合**，杀伤集由此上界。另一条被杀的 f32 边 vnew_all 是 fwdscan→dkb/dqkwg 边，**不**在本融合可达范围。

### 11.5.3 全物化屏障：为什么生存期不允许池共享

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

### 11.5.4 池共存算术：杀伤集枚举

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

### 11.5.5 占用反转：1 CTA/SM 的实测代价

97024 B 的融合 kernel 在**两个**合并体上都跑 1 CTA/SM。这不是假设性损失——r15 的配对实测就在这个 stage 上：dqkwg 512thr/1CTA 对装机 256thr/2CTA = 0.632 → 0.547 ms @T2048（**F = 1.155×**）、2.3232 → 1.8443 @T8192（**F = 1.2597×**）；r15 的 ncu 记录归因于独立指令流效应（达成占用 15.02 → 29.17%、no-eligible 89 → 83%）。

### 11.5.6 逐 cell 净账与 NO-GO

带宽计价用 r24 W-park 备忘录的本机标定（DRAM 类 824.6 GB/s / L2 驻留 2144 GB/s），阶段时 r3 三重交错配对中位，占用成本 dqkwg 半按实测 F、wyb 半按同结构类推（256thr/2CTA/128reg/mma+staging 体、已有 32 B spill——任何 F_wyb ≥ 1.10 都把 u=8 净额压到噪声）：

| cell | 删 GB | 省 ms（L2/DRAM） | 1CTA 代价 ms | 净 best/mid/worst ms | 比值 now → best-corner |
|---|---|---|---|---|---|
| B1T2048 u=1 | 0.152 | 0.071/0.184 | 0.079..0.130 | +0.106/+0.024/−0.059 | 0.488 → 0.519 |
| B4 u=4B | 0.608 | 0.284/0.738 | 0.439..0.722 | +0.298/−0.070/−0.438 | 0.411 → 0.432 |
| T8192 u=4T | 0.608 | 0.284/0.738 | 0.440..0.724 | +0.297/−0.071/−0.440 | 0.453 → 0.476 |
| T16384 u=8 | 1.216 | 0.567/1.475 | 0.855..1.401 | **+0.620/−0.107/−0.834** | 0.463 → 0.488 |

（best 角要求全部删除流量按纯 DRAM 计价**且** wyb 半零 1CTA 代价——后者不可信；诚实中心：u=1 打平、u≥4 亏损。）甚至**不可建的完美融合**也被字节算术封顶：完美杀伤 1.485 GB @T16384 → 最多 1.80 ms，比值上限 5.6247/(12.1521−1.80) = **0.544**（L2 计价 0.491）——R3 的"~0.6× 上限"（假设两阶段纯字节受限）已被字节本身收窄，因为这个对的残差是发射受限机件而非字节。对照战役已装杠杆（dhs 1.73 ms、caryA 删除 0.56 ms、fwdscan hoist 0.35 ms @u=8），融合的期望值不过线而风险等级最高（合并两个 ~1000 行 kernel、重推跨相位生存期竞技场、两体已溢出的寄存器并集 ≤128、全门槛栈重跑）。**裁决：NO-GO，带全套算术。**

---

## 11.6 精度关闭定理：三重独立证据

### 11.6.1 定理表述

把四轮的证据组织成可引用的定理形式：

> **定理（bwd u ≥ 1 的精度关闭）。** 在门控配方数据体制、fla 派生容差合同（dq/dk/dv ≤ 8e-3、dbeta/dg ≤ 2e-2、dh0 ≤ 8e-3）与本机（sm_120a、DRAM 824.6 GB / L2 2144 GB/s 标定）之下，九（现八）kernel f32/bf16 分段反向链对 fla fused-autograd 链的比值在 $u \ge 1$ 全体制**无法达到 1.1×**；更精确地，u=8 的比值天花板为 **~0.85–0.92×**（完美屋顶线执行当前字节集），且该上限由三重相互独立的证据共同钉死——任何单一证据的松动都不改变结论。

三重证据的独立性在于它们关闭的是**不同的逃逸方向**：预算地关闭"把字节再铸低"，地板算术地关闭"把执行做到完美"，融合地关闭"把边删掉"。

### 11.6.2 证据一（预算地）：能挪地板的字节被合同判死

能移动地板的字节只有被杀的 f32 边（u_all/vnew_all/dvp_all/dkB_sum/dk_local，470 KiB/M 行）——而它们被杀的原因是**消费者对值紧**（§11.2.5：读者就是紧的；split-bf16 对流量零节省、对求和点分辨率无补）；全部被评分输出（dq/dk/dv）由预算保持 f32。RC1 的 1.33–1.49× 天花板需要全部 32 符号类可铸——**被预算本身关闭**，与执行质量无关。

### 11.6.3 证据二（地板算术地）：u=8 天花板 0.85–0.92× 的推导链

推导链步步可查（`r3_report.md` §5、`r4_fusion_report.md` §7）：

1. R2 落地 9 符号铸造后，字节缩放地板 = **0.81 ms/u**（r7 的 1.074 按字节比缩放）；
2. R3 退役删掉 caryA 整条字节流（~257 MB → ~0.34 ms @T16384）与 dhs 的 A 窗流量 → 地板 **~0.75 ms/u**；
3. 完美执行（100% 屋顶线）该字节集的 e2e = ~0.75×8 + 固定 ≈ **6.1–6.6 ms** @u=8；
4. 本窗 fla = 5.620 ms → **天花板 = 5.620/6.1–6.6 ≈ 0.85–0.92×**；
5. 对照：r3 实测 12.15 ms ≈ 地板的 2×（残差为发射受限的标量/扫描机件——dqkwg 混合扫描、wyb 混合位点——不是字节）；即便把 ~2× 执行余量全部花掉，u=8 也只到 0.92×，**跨不过 1.1**；
6. 换个极端口径：把 dqkwg+wyb+dhs **整段删除**（它们计算被评分输出，删除并非真实选项）也只到 0.88——融合只能减字节、不能归零输出级。

### 11.6.4 证据三（融合地）：最后的搬运者被池与占用判死

唯一还能挪地板的机制（删边）只剩 dqkwg+wyb 融合一个可达载体，而 §11.5 的三重杀——**池封顶**（G 全物化屏障下最优可建 97024 B = 完美字节的 82%）、**占用反转**（实测 1CTA 因子 1.155–1.26×）、**净账打平或亏**（u=8 best-corner 0.488）——把这一载体判 NO-GO；连不可建的完美融合也被字节算术封在 ~0.54×。

### 11.6.5 "这是误差合同的数字，不是我们的数字"

关闭定理的最后一句属于表述纪律：u ≥ 1 的 0.46–0.47× **不是实现失败的测度，而是合同的下界**。若有人想改这个数字，可改的入参全都不在本项目手里——它们是：容差合同（更宽的梯度容差）、门分布（更弱的 decay 体制，但那也会违反 A 退役的体制门）、或带宽标定。战役因此以**装机主张**而非遗憾收官：**小形状 + 冷启动 bwd 胜者（1.72× @T512H4 同窗）加 u ≥ 1 fwd 胜者；u ≥ 1 bwd 是精度受限的 ~0.46–0.47×——地板的数字，不是我们的数字。** 若反向工作重开，有余量的方向是新的体制（本阶梯外的 B/T/H、学习门数据体制），不是这条链在这些形状上的任何想法类——B1T2048H16..B1T16384H16 的每个想法类都已被定价或实测关闭。

---

## 11.7 战役总结：四轮一览与最终战绩

### 11.7.1 四轮一览

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

### 11.7.2 最终战绩表（战役对外主张）

| 战线 | 判定 | 记录 | 口径 |
|---|---|---|---|
| **fwd，u ≥ 1 全形状** | **胜** | B1T2048 **2.125×** / B4T2048 **1.2211×** / B1T8192 **1.3493×** | r25 正式 n=30 交错记录（双窗：另有干净窗 clean3 记录并存）；战役未动任何 fwd 旋钮，记录不变 |
| **bwd，小形状（u ≪ 1）** | **胜** | B1T512H4 **1.723×** vs fla | R3 干净窗三重交错配对中位（战役内从 2.172×/1.86× 深化） |
| **bwd，u ≥ 1** | **精度关闭**（非执行关闭） | 0.411–0.488× 全阶梯；边际比 0.472；天花板 0.85–0.92× @u=8 | 三重独立证据（§11.6）；"误差合同的数字" |
| **冷启动，全形状** | **胜** | 我方 AOT cubin：零 autotune、~27 μs 发射；fla：逐形状 fwd 18.7–30.8 s + bwd 29.3–38.3 s 冷 autotune（T16384 达 69 s；缓存命中 0.65–5.67 s） | 新进程 + 新 TRITON_CACHE_DIR 口径 |
| **编译器主张全链** | **成立且 sharpened** | 语义单一事实源 → 编译期调度（Z3+成本模型）→ 分层验证 → 位级确定性 → 预算门控精度决策 → 决策级 NO-GO 定价 | 战役四轮本身是"调度即 IR"方法学的展演：每个结论先 CPU 后 GPU、先预测后实测 |

### 11.7.3 收官链的健康认证与未竟事项

R4 对收官链（== r3-final bytes）做了纯位恒等认证：48/48 cubin 对 R3/R2 记录全对账（32 == R3 md5 记录；16 对 R2 锚、其中 fwdscan/dhs 恰按 R3 设计相异）、确定性 N=3 @T16384 全位一致、生成器零改动。未竟事项如实记录：R3 排名 #2 的扫描原生性（~0.1–0.2 ms 级）未装——0.2/12.15 ms = +0.008 比值，改变不了任何判定，记录在案不定价；战局主张的最终用户确认（关闭定理是否被接受为指令的答案）在收官时点**待确认**——`evidence/G5/CAMPAIGN_CLOSEOUT.md` 为其正式文档。

---

## 11.8 本章小结

体制变化战役把"形状外推"从猜测变成了一条完整的证据链：**标度律框架**（两类 kernel 的仿射律 + L2 足迹罚 + 波数量子化，在 GPU 触碰前预测整条曲线并选中 6 个判定 cell）；**误差预算方法学**（读写边界精确复制的重构 fp64 孪生 + leave-one-in 分类，把"bf16 化"变成 10/16 可铸、6 杀的合同，并顺带发现 A_all 数值死与 split-bf16 零收益定律）；**两个装机杠杆**（A_all 退役：max|A@dH| = 2.3e-18 的探针推导 → caryA 删除 + dhs 坍缩为移位拷贝，池 49152→8192 B、阶段 5.89×、逐字节等价收尾；tgemm a-fragment hoist：warp 地图轮转→连续，CVT 税 1.115× 翻正且位恒等）；**融合定价学**（全物化屏障、池共存算术、实测占用反转的三重杀，决策级 NO-GO）；最终以**精度关闭定理**收官——三重独立证据把 u ≥ 1 的 ≥1.1× 关闭在合同层面，u=8 天花板 0.85–0.92×。战役四轮把大形状反向从 0.39–0.61× 推进到 0.41–0.49×（边际比 0.472）、小形状深化到 1.72×、冷启动全形状胜，并交付了比速度更持久的两件东西：一份**哪些字节在合同上必须保持 f32**的完整预算表，与一套**在写码之前给融合定价**的算术。

> 证据索引：`evidence/G5/rc1_scaling_predictions.md`（预测文档）+ `evidence/G5/rc1_ladder_report.md`（实测阶梯判定）+ `tools/rc1_price.py`（预测定价器）；`evidence/G5/r2_bf16_globals_report.md` + `evidence/G5/scratch/r2_bf16_probe.py` + `scratch/r2_logs/`（预算探针与三重交错日志）；`evidence/G5/r3_report.md` + `evidence/G5/scratch/r3_probe.py` + `scratch/r3_logs/` + `scratch/r3ahoff_T16384/`（退役/hoist 探针、隔离臂与位级对照）；`evidence/G5/r4_fusion_report.md` + `evidence/G5/scratch/r4_price.py` + `scratch/r4_audit.py` + `scratch/r4_logs/`（融合定价与收官健康认证）；`evidence/G5/CAMPAIGN_CLOSEOUT.md`（战役收官记录：最终战绩、三重关闭论证、决策史与待确认状态）；决策上下文 `evidence/G5/DECISIONS_R25.md`（D2 持有决策及其推翻条款）；门槛栈与窗口纪律见 `evidence/G5/GATE_REPORT*.md` 与第 8–9 章；装机实现 `tools/gdn_bwd_gen.py`（`GDN_BF16G`/`GDN_ARET`/`GDN_TGEMM_AH` 三旋钮）。
