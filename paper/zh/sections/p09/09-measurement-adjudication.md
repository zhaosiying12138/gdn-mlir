# 9 测量与判定方法学：在共载 GPU 上建立裁决级证据

> 本章自洽可读。所有加速比、判定与标定结论均给出证据文件指针；每个小节按【动机→源码片段→数据→结论】四件套组织。全局口径与全文一致：单卡 RTX 5090 Laptop（sm_120a，消费级 Blackwell，82 SM，GDDR7，无 tcgen05），全部 GPU 计时取自共载机会窗口或干净窗口并在源 JSON 中以 `shared_gpu` 标志与 `nvidia-smi` 快照披露；共载任务从不被触碰、从不被信号打断，运行前在 `~/codebase/GPU-COORDINATION.md` 登记协调行，显存占用 <4 GB。**加速比的分母一律是同窗交错测得的基线，不是跨窗口中位数之比；绝对毫秒数只作工程参考，永远带窗口快照。**

## 9.1 章首：为什么测量本身需要一套方法学

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

## 9.2 交错配对判定协议（`tools/gdn_final_adjudicate.py`）

### 9.2.1 动机

朴素做法是"先测我方 50 次，再测 fla 50 次，取中位数之比"。这条路径在上面三条失败模式下全部失效：两个 50 次块落在不同窗口（失败模式 1）、不同进程（失败模式 2，+47% 量级）、甚至不同 cubin 加载序（失败模式 3）。协议设计的目标是构造一个**对窗口漂移一阶免疫**的估计量，并把"窗口是否干净"从假设变成被记录的观测量。

### 9.2.2 源码片段与协议结构

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

### 9.2.3 数据：前后对比

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

### 9.2.4 结论

交错配对协议把"共载 GPU 上能不能信一个加速比"从争论变成了可执行的契约：分母同窗、估计量保序、窗口质量被记录而非被假设、置信度随档自携、判定规则显式且保守。正式记录（2.1250×/1.3493×/1.2211×）全部产自该协议；协议同时产出了同样重要的负面能力——在窗口撑不住时给出结构化的"不可判定"而不是含糊的数字。

---

## 9.3 可判定性 v1/v2 双轨与 D1 决策（`adjudication_methodology.md` §b-v2）

### 9.3.1 动机

v1 规则在三个 B4T2048 正式窗口（r20 1.1883×、r21 1.2329×、r22 1.1417×）上暴露了一个结构性问题：三次会话**我方侧一致到 ±1%**，分歧全部来自 **fla 的尾部**（fla max 1.24–2.58 ms 对中位数 0.97–1.04 ms——co-job 的突发落进 fla 更胖的 kernel 里）。v1 的 IQR 是尾部敏感统计量，于是它度量的是**污染**，而不是未被污染配对的噪声底；这导致连 r22 那个 30/30 全胜、p=9.3×10⁻¹⁰、IQR-rel 0.060 的窗口都被挡下（差 0.039）。

### 9.3.2 规则与源码依据

v2 规则（`evidence/G5/adjudication_methodology.md` §b-v2；重算脚本 `scratch/r23f/v2_verdicts.py` → `evidence/G5/decidability_v2_r23f.json`、`scratch/r25f/v2_verdicts_r25.py` → `decidability_v2_r25f.json`）：与 v1 唯一的区别是 IQR 改在逐对比值的**内层 80%**上计算（对称 10% 修剪；n=30 剔除最高 3 对与最低 3 对）；median、符号检验、胜场数**一律不动**：

$$\text{decidable}_{v2} \iff (\mathrm{median}_r - 1) > 3 \times \frac{\mathrm{tIQR}_r}{\mathrm{median}_r} \;\wedge\; \text{wins} \ge 25/30 \;\wedge\; p \le 10^{-5}$$

设计依据：一个成员内部的突发每会话污染 O(1) 个对；10% 修剪恰好移除这一类，窗口干净时什么也不修剪（v1==v2）。

### 9.3.3 数据

**r22 配对上的双轨判词**（B4T2048 正是 v2 为之而生的案例）：

| 形状 | median | v1 IQR rel | v2 tIQR rel | 效应 对 3×门槛 | 胜/p | 判词 |
|---|---|---|---|---|---|---|
| B4T2048 | 1.1450 | 0.060 | **0.044** | 0.145 > 0.131 | 30/30，9.3×10⁻¹⁰ | **DECIDABLE-v2**（v1 否，差 0.039） |
| B1T2048 | 2.1113 | 0.374 | **0.282** | 1.111 > 0.846 | 29/30，2.9×10⁻⁸ | **DECIDABLE-v2**（v1 否；带窗口漂移告诫） |
| B1T8192 | 1.2054 | 0.296 | 0.234 | 0.205 < 0.703 | 27/30，4.2×10⁻⁶ | 双否（真实双侧窗口） |

**绑定于任何 v2 声明的告诫（原文具约束力）**：v2 度量的是**会话内**一致性，不能裁决窗口间偏差——B1T2048 的中心在两个窗口之间仅凭 fla 中位数（0.35→0.42 ms）就从 1.78× 移到 2.11×，其 v2 判词声称的是内部连贯而非稳定的 2.1×；B4 是三类窗口里我方稳定、只有尾部分歧的"设计案例"。

**R25 新鲜配对上的同规则复算**（`decidability_v2_r25f.json`）：B1T2048 判 **DECIDABLE-v2**（tIQR rel 0.347 < 效应/3 = 0.372，30/30）且中心首次跨窗稳定（2.1113→2.1174，相差 0.3%）——这是 r23 备忘要求"再积累 ≥2 个窗口"后的第一个；B4T2048 新窗更吵，v2 亦否（tIQR rel 0.168→门槛 0.504 对效应 0.203，且 25/30、p=1.6×10⁻⁴ 高于 10⁻⁵ 线），故迁移后的数字 1.2211× **原样携带本窗双否判词**，而 D1 双引用声明仍锚定 r22 配对——如实陈述，不做平滑。

### 9.3.4 D1 决策与结论

D1（R25，`evidence/G5/DECISIONS_R25.md`）：**v2 以 DOCUMENTED-OPTIONAL 侧统计量身份签署**——v1 仍是跨轮可比性的记录规则；任何 v2 声明必须双引两行（"v1-否 / v2-是"）；B4T2048 的 1.1417× 按双引用声明，B1T2048 保持 v2-软直至中心跨窗稳定。该决策是**纯引用政策**：没有任何测量、gate 或 cubin 依赖它，翻转它只是重贴标签。会话内无用户响应，按自治规则以建议默认值采纳并如此记录（覆盖随时欢迎）；一行用户回复即可撤销（"v2 withdrawn"）。与之成对的 D2（bwd 冻结于 r19 字节）后来被用户在 2026-10-10 会话中显式**推翻**为"启动 regime-change 评测（B/T 阶梯）"——D1 未被推翻，双引用口径沿用至今；这一对决策记录本身也是方法学的一部分：**判定规则的变更与测量数据的变更被制度性隔离**。

结论：v1/v2 双轨把"保守到挡住真相"与"激进到制造真相"两个失败方向同时关掉——v1 保跨轮可比，v2 在明示风险下恢复被尾部污染遮蔽的强证据，两者的差异永远并排出现在档案里。

---

## 9.4 测量系统审计：模块布局彩票与 B 侧偏置三部曲

### 9.4.1 动机

r18 的链级 A/B 留下一条可疑的系统性告诫：r14u_ab 配对器里 **B 侧**的 f3 平均慢约 +13 µs、f2 约 +6 µs（T8192）。如果这是仪器偏置而不是代码差异，那么之前所有经该仪器的 f2/f3 结论都被污染。r19 把它当成一等的测量学问题立案侦查。

### 9.4.2 三部曲（证据链，`evidence/G5/scratch/r19f/`）

1. **发射序假设死亡**。把**同一个** cubin 目录同时作为 A、B 两臂加载，仅交替发射顺序——f3 仍然读出 B 侧 +14.8 µs（39/40 对）。发射顺序不是原因。
2. **构造序翻转精确镜像**。只翻转两个 `FullKernel` 的**构造顺序**（即 `cuModuleLoadData` 的先后），其余一切不变——偏置精确反号（−16.8 µs，0/40 对；`ab_sameside_loadflip.json`）。机制由此定位：`cuModuleLoadData` 为模块的 `.global` 区分配布局，**后加载的模块**其全局缓冲落在较差的布局上；f3 在 T8192 的约 160 MB 工作集为此付出约 6%。
3. **彩票标定**。在（四模块的）正常 harness 进程里，同一构建的 f3 逐会话抽签 +2.3/+14.8/+14.3 µs（warmup-20 会话）→ 标定均值 **+10.5 ± 6 µs，f3 专属**。f1/f2 在 warmup 20 下干净：r18 观察到的 f1"+53 µs 前半程"是**时钟爬坡**（warmup 8→20 即消灭，`ab_sameside_warm.json` 读 −0.1 µs），r18 的"f2 +6 µs"同为爬坡——**f2 被洗清**。

关键佐证（"PTX 相同暴露"）：这一偏置与被测代码**逐字节无关**——R24 复核 T8192 的 f3 +12 µs 布局税时确认 f3 的 PTX 在两臂构建中逐字节相同，属 r19 彩票类；且"生成器树与 env 不变 → 重构建逐字节复现同一模块"本身是工具链的正向数据（**彩票在生成器编辑时重抽，不在重构建时重抽**；R25 的正式记录因此携带 R24 抽到的签）。

### 9.4.3 源码片段：平衡双载 harness（`evidence/G5/scratch/r14u/r14u_ab.py`，r19 起的默认行为）

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

### 9.4.4 数据：修正表（平衡 harness 复测，四对 cubin 全部 md5 对上记录值）

| 会话（40 对 × 2 半程） | f1 | f2 | f3 原始 | f3 修正后 | staged 链 |
|---|---|---|---|---|---|
| 同构建抽签 ×3（标定） | +0.1/−0.1 | −3.1..+0.3 | +2.3/+14.8/+14.3 | 构造上为 0 | — |
| r15 WST 对 r14-US | **−12.8**（7/80 慢） | −0.4 | +15.2 | ~+3 ≈ 0 | −0.5 |
| r16 bf16sv 对 r15 ship | −3.0 | +0.9 | +15.0 | ~+3 ≈ 0 | — |
| r18 F1KS 对 r17 ship（旧记录 −54.4/−1.3/+14.5；staged −47.6） | **−56.4**（3/80 慢） | +0.2 | +14.0 | ~+2 ≈ 0 | **−49.9 原始 → ~−62±5** |

修正的三个具体后果：r15 的 WST f3 效应确认为 ≈0（旧记录 −0.4 本来就在偏置带内）；r16 的 f2/f3 效应确认 ≈0；r18"真链赢 −60..−71 µs"收紧为 **−62±5 µs、全部来自 f1**——原始 f3 B 侧罚分（+14.5）100% 是布局。**历史记录不回写**：这些以追加修正的形式落档，r15/r16 的判词经修正后不变，r18 的量级被收紧。

### 9.4.5 结论

这一节的方法学产出有三层。第一，**测量系统本身是仪器**：一个配对计时器的加载序可以携带 6% 的系统偏置，且只有用"同构建两侧对照"这类零假设实验才能暴露。第二，修正有制度：镜像半程消除一阶、同构建标定会话扣除残差、历史只追加不回写、cubin md5 全程对账。第三，它的副产品把 r18 的旗舰结论（F1KS 链级 −62±5 µs）从"带告诫的读数"升级为"修正后的干净归因。

---

## 9.5 删除探针时间图：`GDN_F1T_*` 机制与 kkt 拆解

### 9.5.1 动机

拿到 f1（WY 段，T8192 下约 227 µs 基准）之后的问题是相位级归因：时间花在 uw 段、kkt、solve 还是 scan？profiler 的重放与时钟锁定会扭曲绝对值（§9.5.4 的 ncu 在锁定 1.57 GHz 下读 313.95 µs，比配对基准高 38%），而"重写一个没有某相位的版本再整体计时"会把重构本身的效应混进来。删除探针（deletion probe）给出第三条路：**在生成器里用编译期旋钮把某个相位整段删掉，其余一切字节不动，然后用平衡配对器测差值**。

### 9.5.2 机制与源码形态

`GDN_F1T_*` 是 `tools/gdn_full_gen.py` 里的一个删除旋钮族（r21 建 9 个，r23 增 3 个），全部默认关闭、**仅计时用途**：任一旋钮打开时输出是垃圾（被删相位的消费者读到未定义数据），因此它们**被文档禁止进入 verify/det 通道**（r21 NKT 先例）。两条工程约束写在轮报告里：(a) payload transform 的 multibuffer 模式对**零使用者**的 workgroup alloc 硬报错——每个删除都必须给孤儿 tile 配 keep-alive 汇（一个谓词化的标量 load→store，放置位置尊重 PTX 生命周期 arena）；(b) 探针构建必须 occupancy 对齐（f1 2 / f2 1 / f3 1 blocks/SM 与 ship 一致，逐构建用 cuOccupancy 抽查），否则差值里混进驻留数效应。测量用 `r14u_ab.py` 平衡 harness，f1-only，40 对 × 2 半程 × 3 会话；三会话彼此相差 ≤1.6 µs。

### 9.5.3 数据一：f1 十探针表（T8192，ship f1 基准 225.8–236.0 µs，~227 µs；差值 = ship − 探针，三会话均值）

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

### 9.5.4 数据二：kkt 拆解（r23 三个新探针）与 roofline 收口

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

### 9.5.5 结论

删除探针时间图是一个可复用的归因协议：编译期删除 + 平衡配对 + occupancy 对齐 + 三会话复测 + 非可加性显式披露。它在这台机器上给出了 profiler 给不了的裁决级相位归因，并直接关闭了 f1 的全部性能之门（uw 串行化 r22 关闭、s2g DRAM 受限、staging 池阻塞、kkt roofline 关闭——f1 在该几何下端到端关闭；残余是 solve 串行相位 1 的 32.5 µs 与 scan 的 17.6 µs，各已有两轮诚实负结果）。方法学上同样重要的副产品：**timing-only 探针与验证/确定性通道的制度性隔离**——垃圾输出的构建永远不进 verify/det，归因证据与正确性证据由不同仪器负责。

---

## 9.6 静态指令分析：无计时窗口下的判定替代（`tools/g5n_static_analysis.py`）

### 9.6.1 动机

用户指令（2026-10-08）：GPU 将被共载占用约半天，**不需要干净计时窗口的分析同样算数**。这是"分析生成指令"这一指令在方法学上的落实：当计时不可得时，对**已落地构建产物**做确定性的、纯 CPU 的指令级分析，仍然可以排序优化目标、给出 roofline 距离、并且结论可复现。

### 9.6.2 机制与源码片段

`tools/g5n_static_analysis.py`（确定性、纯 CPU）对 r6 落地的 9-kernel bwd 链（`evidence/G5/scratch/r6_T2048`）加前向 mono 三联与 fused 旗舰参照，从**ship 的产物本身**计算：PTX 指令族直方图（复用 `harvest_flaptx.ptx_features` 的 FAMS 表，作用于 harness 实际装载的 `kernel_smemfix.ptx`）、SASS 直方图（对 cubin 现场 `nvdisasm -c`，含 spill LDL/STL、LDSM/LDGSTS 与宽度解码）、occupancy（ptxas 寄存器 + manifest 的 smem/threads/grid，投影 `gdn_costmodel` 的 sm_120 结构模型：82 SM、64 K 寄存器、100,352 B 可用共享内存、16 CTA 槽、32 warp 槽）、每 chunk 计算强度（mma 位点 × 块折叠——scan 类 kernel 网格 = B·H·NV，每个块走遍全部 chunk，故静态体按 NV 折叠）、以及**距 roofline 的距离**。roofline 地板用 G3 标定率，其中 mma 峰值在加载时执行关键的 ×32 修正：

```python
# corrected mma peak: file flops field over-counts x32 (warp-level op);
# honest peak = instr/s / 32 * 4096 (evidence/G3/solver_report.md; the
# costmodel performs the same correction at load time)
MMA_PEAK = M["mma_bf16_m16n8k16"]["instr_per_s"] / 32.0 * 4096.0
```

（标定文件按**指令**计数 mma.sync——一条 m16n8k16 是 warp 级操作，对 4096 FLOP/条直接乘以吞吐率会虚报 32 倍；诚实峰值 = 指令率/32 × 4096 = **115.5 TFLOP/s**。`evidence/G5/static_analysis_r7.md` 卷首。）测得中位数取 `bwd6_bench2048.json` 与其重跑的均值（两轮相差 <1.5%）。

### 9.6.3 数据

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

### 9.6.4 结论

静态指令分析把"没有计时窗口"从停滞理由变成一种方法：确定性、零 GPU、从 ship 产物出发、以标定率为地板、以显式告诫封边。它产出的排序直接驱动了 round 8+ 的杠杆选择，其签名表（LDS 风暴 / 零 ldmatrix / 零 FMA）把"为什么慢"定位到指令类而不是笼统的"实现不佳"；而 dkb 反例与地板和 1.074 ms 这两个链级事实，分别给出了可达性证明与必须同时移动流量与执行效率的战略判断。

---

## 9.7 fla autotune 搜索成本：冷测与四路摊销（`tools/g6/*`）

### 9.7.1 动机

"编译期调度对在线 autotune"的核心成本对比必须经得起两种质疑：我方的 0.8 s 到底计了什么；fla 的 39.8 s 会不会被它自己的磁盘缓存摊销掉。R26（camera-ready TODO #3）为此专门测量了摊销。

### 9.7.2 我方口径：0.8 s 纯 CPU（`tools/g6/g6_solver_timing.sh`）

宿主 CPU（`OMP_NUM_THREADS=4`，conda `gdn`，z3 5.1.0），重测于本窗口：

| 空间 | 合法配置 | Z3 枚举 | 全求解墙钟（枚举 + 代价模型全排序） | top 选择（复现 G3） |
|---|---|---|---|---|
| kkt K=128（B4 T8192 H16），strict 98 KB，v2 | **368** | 0.307 s | 0.378 s | `kkt_bf16_mma_p2_t32x32x64_w2_ws` 8.62 µs |
| kkt K=128，v3（waves/occupancy 形式） | **368** | 0.296 s | 0.365 s | `kkt_bf16_mma_p3_t64x64x32_w2` 140 µs（文档化的 v2→v3 翻转） |
| o K=128 V=128，strict，v2 | **465** | 0.379 s | 0.444 s | `o_bf16_mma_p2_t32x32x64_w2_K128V128` 13.25 µs |
| o K=128 V=128，v3 | **465** | 0.374 s | 0.442 s | `o_bf16_mma_p0_t64x128x32_w2_K128V128` 350 µs |

确定性契约由重测再次确认：固定声明序、seed 7、rlimit 20e6、按（预测值，名字）排序——368/465 空间与 top 选择按字节语义复现 G3 输出。排序质量 Spearman **0.8962**（kkt 0.936 / o 0.829，n=18），对全部结构常数 ±2× 扰动稳定于 [0.882, 0.902]、对 ±30% 时间噪声 ρ≥0.856。如实记录的瑕疵：o K=64 V=64 子空间（204 配置）在本环境因 rlimit 下 Z3 部分模型伪影崩溃（`tools/gdn_solver.py:187` 的 `m[zv]→None`）——两个规范空间精确复现，崩溃如实入档。一次性沉没成本（G3 标定微基准、G2 的 20 配置编译矩阵）**不计入**每形状搜索；每个新形状，调度器在首个候选跑起来之前增加约 0.8 s CPU 与零 GPU 秒。代价模型本身无逐配置拟合常数（速率全部来自标定表 + SASS 导出常数），这是排序诚实性的来源。

### 9.7.3 fla 侧冷测（`tools/g6/g6_fla_autotune_cost.py`）

测量纪律全部先行验证：本环境 `FLA_CACHE_MODE` 未设 → DISABLED → fla 磁盘配置缓存**不被咨询**（`fla/ops/utils/cache.py:51`）；本 GPU 的 fla 配置目录不存在（没有可挪的热盘缓存）；`TRITON_CACHE_DIR` 指向私有新鲜目录（共享 `~/.triton` 与 co-job 均不被触碰，事后无需恢复）；`Autotuner.run` 被包裹以逐 kernel 计数候选配置与 tuning key。输入 = G5 共享配方 B1T2048H16、`FLA_FLASH_QLA=0`（纯 Triton 后端，与全部对比一致）。单机会窗实测（墙钟，同步，单进程）：

| 阶段 | 冷（首调） | 暖（10 次中位） | 冷−暖 | 实测配置数 |
|---|---|---|---|---|
| **fwd** 首次 `chunk_gated_delta_rule` | **16.55 s** | 0.491 ms（min 0.418） | 16.55 s | **25** / 5 kernel：cumsum 4、kkt_solve 6、recompute_w_u 9、fwd_h 4、fwd_o 2 |
| **bwd** 首次 `torch.autograd.grad` | **23.26 s** | 1.349 ms（min 0.981） | 23.26 s | **23 新增** / 4 kernel：bwd_dv_local 9、bwd_dhu 4、bwd_dqkwg 9、prepare_wy_repr_bwd 1 |
| **合计** | **39.81 s** | — | 39.81 s | **48**（其后 Triton 缓存 67 个 kernel 条目） |

合理性锚点：暖中位数正落在 G5 基线带内（fwd 0.316–0.517 ms 跨 G5 窗口；bwd 1.34 ms 于 G5-C 对 1.349 ms）。共载并未使测量失效——冷成本由编译+基准主导，约 10³× 于争用噪声。机制解读：Triton 对每个候选配置（ptxas）编译并在真 GPU 上 do_bench；盘缓存缺席/禁用时这按**每进程、每形状键**重付（fla 自己的缓存层恰为避免此事而存在，且默认关闭）。

### 9.7.4 四路摊销（`tools/g6/g6_fla_amortize.py` → `evidence/G6/fla_amortize_r26.json`）

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

### 9.7.5 对比口径（诚实边界）与结论

`evidence/G6/search_cost.md` 的 scope note 原则性生效：**0.8 s 是排序"我们的 stage-kernel 旋钮空间"的成本**；全算子的 ship 决策在 G5 另用了实测配对 A/B（那是工程归因，不是搜索）。可比量是：同一候选空间下，离线枚举+排序（0.4–0.8 s、确定性、零 GPU）对在线逐候选编译+基准（每方向 16.6–23.3 s，按进程/形状键重付）。结论表（search_cost.md）：fwd+bwd 单形状搜索成本 **1 : ≈50**；搜索发生的位置在"完全离线 CPU、GPU 在选中配置运行前零介入"对"部署 GPU 上、首调、阻塞"；可迁移性在"模型 + 标定携带到新形状（solver→compile→ptxas 重排序的回环设计）"对"按进程/形状键重付（除非可选的 fla 缓存层已被填充）"；四路摊销给出该结论在 fla 缓存全部有利配置下的界（7.7×–67.5×）。

---

## 9.8 硬件标定体系：`run_calib.sh`、共享窗与干净窗、`shared_gpu` 标志

### 9.8.1 动机

三处消费硬件速率：SMT 调度器的解析成本模型（G3）、静态分析的 roofline 地板（§9.6）、roofline 收口判定（§9.5 的 826 对 765 GB/s）。标定数字本身也必须带口径：在共载窗口测得的峰值是**下界**，用途排序与地板计算可以安全使用，但必须显式标注，且干净窗口到来时要可无争议地刷新。

### 9.8.2 机制与源码片段

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

### 9.8.3 数据：共享窗 → 干净窗刷新

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

### 9.8.4 结论

标定体系回答了"分析里的地板凭什么信"：速率可复测、口径自标注、下界方向性明确（共载下界只会让"距离 roofline 很远"的结论更保守）、干净窗口刷新由轮询器与重测预案制度化为常规动作而非一次性运气。与 §9.2–§9.7 合起来，本章的立场是：**在一个测量条件恶劣的平台上，方法学不是论文的附录，而是每一条性能主张能站立的地基**——同窗交错的比值口径、显式可判定规则、被审计的测量仪器、与计时解耦的归因与分析通道、写明摊销口径的成本账、以及带口径标志的标定率，六者共同构成这台机器上"可裁决证据"的定义。
