# 10 编译期调度与 PTX 覆盖验证体系：Z3 求解器、解析成本模型、一次性标定、参考收割器、覆盖矩阵与旋钮库

## 10.0 章首：XLA/TPU 支柱二的完整落地叙事

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

## 10.1 Z3 旋钮求解器（tools/gdn_solver.py）

### 10.1.1 动机

"选哪个配置"在手写世界里是 autotune 在设备上逐候选计时回答的问题；在调度即 IR 的世界里，它必须变成一个带确定性契约的形式化问题：给定旋钮空间、合法性约束与目标函数，机械地给出全序。TileLang 已经证明了 Z3 在这一角色上的可行性——但其 modulo-scheduler 只存在于 Ascend 后端（变量为指令开始时间与 buffer 版本，目标为最小启动间隔 II），CUDA 侧退化为运行时 autotune（docs/PLAN.md §7 关键事实）。本求解器（工作流 G3-B 交付物）将 TileLang 的机制**上移一层**移植到 CUDA/sm_120：决策变量直接是旋钮向量而非指令时隙，约束编码资源容量与结构合法性，目标为解析成本模型。这一移植在 CUDA 侧无先例。

### 10.1.2 机制与源码：变量、约束、目标

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

### 10.1.3 确定性契约

`random_seed=7`、`rlimit=20×10⁶`、固定变量声明序、固定阻塞子句序、终排序键 `(predicted_time, name)`，此外全程无 RNG。**数据对比**：两次独立运行产出相同的 368 项排名，名序与时间逐字节一致（evidence/G3/solver_report.md §1；evidence/G6 复测再证）。落盘的 `evidence/G3/solver_kkt.json` 记录 `z3: {seed: 7, rlimit: 20000000, last_check: "unsat"}`——`last_check` 为 `unsat` 即空间干净穷尽（最后一个阻塞子句使公式不可满足），这是"枚举完整"的可核查证据。

### 10.1.4 防御性补全：partial-model 伪影与 SAT 探针

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

### 10.1.5 数据对比：合法空间规模与排名示例

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

### 10.1.6 与 TileLang Ascend 机制 B 的移植差异表

| 维度 | TileLang Ascend `z3_scheduler`（机制 B 原型） | 本文 G3-B 移植（CUDA/sm_120） |
|---|---|---|
| 决策变量 | 指令**开始时间** + buffer 版本（整数时槽） | **旋钮向量**（dtype/prec/tile/warps/pipeline/mma/warpspec/tma）——上移一层 |
| 约束 | 依赖关系 + 资源互斥（同一功能单元的时槽冲突） | 资源容量（smem 字节模型、regs/SM）+ 结构合法性（整除/fragment 对齐/流水深≤趟数）+ 旋钮耦合 |
| 目标 | min II（启动间隔），estimate_latency 引导的 II 二分搜索 | min 解析时延；因模型非线性（max/混合/除法），SAT+阻塞枚举全空间、求解器外排序 |
| 确定性纪律 | rlimit + 固定种子 | 同一纪律保留：random_seed=7、rlimit=20×10⁶、固定声明序/阻塞序/(predicted, name) 终排序 |
| 输出 | 指令调度（时槽指派） | 全序合法配置排名 + 每项预测分解（terms_ms）与 smem 模型字节 |
| 回环 | 无（调度即终点） | solve → 生成/编译 top-K → ptxas 静态复核重排 → 幸存者窗口计时；未编译项 spills=0、命中已编译向量自动拾取实测特征（`has_index_features`） |
| 防御 | —— | partial-model 伪影的 SAT 探针补全 + rlimit 截断的确定性重试（§10.1.4） |

### 10.1.7 结论

Z3 求解器把支柱二从哲学落成机械：合法空间（368/465/204）以亚秒级纯 CPU 穷尽、以 `last_check=unsat` 可核查、以固定种子与固定序逐字节复现；目标函数的诚信性完全外包给下一节的成本模型与标定；其防御性补全与重试把 Z3 的两个真实行为伪影（partial-model、rlimit 截断）从潜在的正确性漏洞变成输出 JSON 里可审计的统计量。

---

## 10.2 解析成本模型（tools/gdn_costmodel.py）

### 10.2.1 动机与零拟合契约

求解器需要把每个合法旋钮向量映射成时延估计。业界常规做法是拟合式代理模型（在设备或仿真侧迭代测量后回归），其诚信性随拟合自由度流失。本文的合同是**标定解析模型**：模型中每一个速率要么 (a) 由 G3-A 微基准窗口一次性测得（evidence/G3/calibration.json，§10.3），要么 (b) 是文档化的 sm_120 级结构常数；**不存在任何逐配置拟合的自由常数**。仅有的全局常数是 launch 后备值 `T_LAUNCH_FALLBACK` 与设备分配 `T_MALLOC`（均可用标定键 `launch_ns`/`malloc_ns` 覆盖，实际使用中标定/锚定生效，后备不触发——输出以 `calibration_fallback_keys` 空集自证）。

### 10.2.2 机制与源码：项定义

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

### 10.2.3 v1→v3 形式迭代：失败保留的长卷

模型不是一次写对的；迭代史本身是证据（evidence/G3/solver_report.md §2 迭代表）：

| 迭代 | 变更 | 总体 ρ | 分 stage ρ（kkt / o） |
|---|---|---|---|
| 1 | v1 合同形；nomma = CUDA 核 FFMA roofline；launch=4 µs 后备 | 0.7352 | 0.644 / 0.486 |
| 2 | v2：+ $T_{\text{lsu}}$（均匀地址 cp.async 流 + fragment ld.shared + 尾部 st.global 的发射项）+ $T_{\text{alloc}}$（设备 malloc 项；smem 修复后的计时 cubin 自动归零） | 0.7475 | 0.669 / 0.486 |
| 3 | nomma 项替换：$T=\text{MACs}\cdot 10\,\text{cyc}/f_{\text{clk}}$ —— SASS 导出的标量环模型（§10.2.4） | **0.8962** | **0.936 / 0.829** |
| 4 | launch 锚：全局唯一 `launch_ns` 由纯发射 ref_f16 锚核导出（实测减去其建模非 launch 项 = 6.073 µs；报告行四舍五入记 6.12 µs） | 0.8962（秩不变；修复绝对尺度） | 0.936 / 0.829 |
| 5 | v3：+ 占用/波次项 | 0.8962 | 0.936 / 0.829 |

（最终代码三形式的正式对照——validation.json：v1 0.8900 / v2 **0.8962** / v3 0.8962，top-5 重叠 0.8，n=18。此处 v1 指终码的"合同项子集"（已含 SASS nomma 模型与锚定 launch），与迭代史第 1 行的"历史 v1"不同——历史 v1 的 nomma 是 CUDA 核 roofline、launch 是后备值。）

### 10.2.4 nomma 项的 SASS 标量环模型（10 cyc/MAC）

迭代 3 的依据是编译期证据而非拟合：实测 nomma 配置跑 0.5–2.5 ms，比 FFMA roofline 高 100–400×。读反汇编发现根因——`convert-linalg-to-loops` 的下降给出**未分布的标量三重循环**：`evidence/G3/gen_noargs/kkt_bf16_nomma_p2_t64x64x32_w4/sass.txt` 的内环（`.L_x_17`）每 MAC 39 条 SASS 指令（10× `IADD.64` 寻址 + 循环控制 + 3 条标量 `LDS.U16` + 1 条 FMA），按 4 发射管道/SM 约 **10 周期/MAC**。常数 `NOMMA_CYC_PER_MAC=10` 从 SASS 读出、从未对实测时间调参（预测 kkt-nomma 1.94 ms vs 实测 0.99–1.07 ms；o-nomma 1.93 ms vs 2.35–2.46 ms——strided-B 的 o 阶段变体每 MAC 代价约 2×，为已接受的簇内偏差）。

### 10.2.5 launch 锚

标定文件不测 launch 开销；`ref_f16_sm80_atom` 锚核是纯发射 kernel（非 launch 项建模值 ~0.05 µs）。由"实测 − 建模非 launch 项"导出**全局唯一** launch 常数 6.073 µs（validation.json `launch_anchor_s = 6.0729×10⁻⁶`），这是合同允许的唯一 launch 常数来源；报告中另附有/无锚两栏以自证秩不变。

### 10.2.6 特征公式自检

`gdn_solver.py --selftest` 把解析每迭代计数公式对 INDEX 静态 PTX 计数核对（18 个 GDN 配置）：`mma.sync` **精确 18/18**、`ld.shared` **精确 18/18**、`bar.sync` **精确 18/18**、`cp.async` 偏差 +0.4–1.6%（prologue/param 尾部松弛）。`pipeline_shared_memory_copies` 的体复制 peel 因子 $(1+p)$ 与 CSE 唯一 fragment 装载均以此方式验证——模型的分子（计数）在编译期就是对的，误差只剩分母（速率）。

### 10.2.7 验证协议与 Spearman 门

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

### 10.2.8 稳健性：±2× 常数扰动与 ±30% 时间噪声

共享窗上的 PASS 是否伪影？`sensitivity.py` 做两类扰动（200 试验/档，evidence/G3/solver_report.md §4）：

- 对**每一个结构常数**（NOMMA_CYC_PER_MAC、TF32_RATIO、LSU_IOPS、SPILL_BW_FRAC、tensor/clk）±2×：ρ 稳定于 **[0.882, 0.902]**；
- 对实测时间均匀 ±5/10/20/30% 噪声：ρ 中位 0.896 / 0.890 / 0.885 / 0.885，最差 p5 = **0.856**——比观测到的 run-to-run 散布大近一个数量级的噪声下门仍过（config_times 的 ms_all 很紧：如 kkt p2 w4 为 21.0±0.3 µs）。

结论：PASS 判定不是共享窗口的产物。

### 10.2.9 已记录盲区（方向正确、幅度未建模）

诚实契约要求模型披露其不知：

1. **pipeline=0 的 `.ca` 串行化幅度**：`create_async_groups` 不带 `bypass_l1` 发射 `cp.async.ca`（L1 分配式）拷贝，每 K 迭代一组 `commit_group`+`wait_group 0`。同 tile/warp 下实测 kkt p0 = 105 µs vs p2 = 21 µs（5×），模型的 sum 重叠只预测 2.4 µs 串行化（预测 7.4 µs）。排序方向（p0 在流水化 mma 变体中最差）正确，幅度未建模——不存在编译期可导出的 `.ca` 惩罚常数，而向实测拟合被诚信合同禁止。`o_bf16_mma_p0`（45 µs）同因。
2. **mma 路径绝对时间系统性低估 3–5×**（fragment 装载依赖链、超出 7.7 ns 边际 bar.sync 的屏障往返等延迟效应）——对全部 mma 配置是均匀缩放，秩不受影响。

### 10.2.10 结论

零拟合解析模型以 Spearman 0.8962（n=18，门 0.7）通过验证，且该数字对全部结构常数 ±2× 与实测 ±30% 噪声稳健；其分子的计数公式对静态 PTX 18/18 精确、分母的速率来自一次性标定；两个幅度级盲区如实记录而秩级正确。支柱二的"凭什么排序"至此闭合。

---

## 10.3 一次性硬件标定（scripts/run_calib.sh + tools/gdn_calib/calib_bench.cu）

### 10.3.1 动机

成本模型的全部速率分母来自一个单 GPU、分钟级、<2 GB 显存的微基准窗口——一次性沉没成本，之后每新形状的调度（0.8 s CPU）不再触碰 GPU。这与 autotune 每形状 39.8 s 的在设备成本形成经济学对照。标定器的设计约束：(a) 每个速率对应模型的一个项；(b) 全部产物 JSON 携带共载快照与 `shared_gpu` 标志；(c) 幂等可重跑——同命令同文件名，干净窗口刷新值直接覆盖。

### 10.3.2 机制与源码：六个微基准核

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

### 10.3.3 数据对比：标定全表

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

### 10.3.4 ×32 修正：warp 级操作的计数陷阱

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

### 10.3.5 共享窗披露

两个窗口（标定与计时）均 `shared_gpu: true`（原始表）：共居训练任务 pid 3411456 常驻（~17.9 GiB 显存，窗内利用率 24→99%，SM 时钟观测 2.1–2.8 GHz vs 1597 MHz 峰值属性）。绝对带宽/FLOP 速率按**受挤占下界**解读。这不是被容忍的缺陷而是被协议化的条件：每个输出 JSON 标注标志与快照；重跑指令与幂等性写入 calib_report §5（`scripts/run_calib.sh` → `scripts/run_g3.sh` → `tools/gdn_time.py`，同文件名覆盖）；排序主张由 §10.2.8 的扰动稳健性与交错对内比较（实验章）承保，而非依赖绝对值。

### 10.3.6 加载器：容错 schema、单位规范化、后备标记

`Calibration` 类原生解析 G3-A schema（device / mma_bf16_m16n8k16 / smem_read_bw_gbps / gmem_read_l2resident / gmem_read_dram / sfu_ex2_f32 / bar_sync_roundtrip / cp_async_16b 嵌套于 measurements），并对其它布局做规范化别名搜索（键名/单位启发式：GB/s、TFLOP/s、GHz、µs/ns 后缀）；缺键时落入场记的 5090 级数据手册后备表并**逐键标记**（`fallback_used`，最终报告中为空集自证）；共载快照中的观测 SM 时钟若高于属性钟则取中位替代（`T_lsu` 随实际运行时钟缩放）；时钟完全缺失时由张量峰反推 $f = \text{peak}/(82\times1024)$ 并标记 `clock_hz(derived)`。

### 10.3.7 结论

一次性标定以六个微基准核覆盖成本模型的每一个速率分母，全表带共载披露与幂等重跑协议；一个真实的 ×32 计数陷阱（warp 级操作按线程计数）被检测规则捕获并在全部下游统一修正；干净窗刷新与原始共享窗表并存，差异在已证稳健性余量之内。

---

## 10.4 参考收割器与 T1 契约集提取（harvest_flaptx / harvest_qlaptx / FlashKDA / ptxas-blackwell 链）

### 10.4.1 动机

覆盖矩阵（§10.5）需要靶子，且采集必须**纯 CPU**（G0 规则：构建期零 GPU 使用）。两个手写设计极点——fla（Triton，multistage 流水极点）与 FlashQLA（TileLang，warp 专化 + TMA 极点）——的 PTX/SASS/ptxas 资源表必须在不启动 GPU 的前提下拿到。第三个风格点 FlashKDA（两段式 token-parallel WY 预段）在最终系统中的落点是旋钮面的 grid 映射旋钮与计划中的第二批收割路径（见 §10.4.4）。

### 10.4.2 fla 收割器：Triton AOT + GPUTarget

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

### 10.4.3 FlashQLA 收割器：TileLang 锁 target + monkeypatch

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

### 10.4.4 FlashKDA 与第三风格点

生态里存在第三个风格点 FlashKDA（两段式：token-parallel WY 预段 + 主段，适配 82-SM 并行度）。在最终系统中它的落点有二：其一，**旋钮面**——L2 调度层的 grid 映射旋钮显式包含"可选 token-parallel WY 预段 = FlashKDA 两段式"（docs/PLAN.md L2 层定义），即风格差异被吸收为配置空间中的一个方向，与"FLA/FlashKDA/FlashQLA 风格 = 配置空间中三个点，复现 = 调参不重写"的总论断一致；其二，**收割路径**——计划第二批参考收割走 FlashKDA 的 `FLASH_KDA_CUDA_ARCHS=120a` 环境变量 AOT（同属纯 CPU 链），T1 契约集的第一批（fla + FlashQLA）先落地并支撑了 §10.5 的矩阵。本节如实记录该分工：已收割两极点、第三点以旋钮面吸收。

### 10.4.5 T1 契约集：33 族的提取

`tools/gdn_coverage.py ref-inventory` 扫描两套收割产物，构建规范化指令族清单——**T1 集 = fla ∪ FlashQLA 出现的全部指令族（33 族）**。分族不是正则拼凑而是两级机制：通用操作码直方图（指令行首 token，字符集含 `:` 以覆盖 `st.shared::cta`、`.L2::cache_hint` 限定词）+ 有序前缀分类规则 `CLASSIFY`（约 90 条，先特征后基线、先 packed f32x2 后标量 f32、bf16 域显式前缀先于通用 cvt/add/mul）；层（tier）判定：`FEATURE_FAMILIES` 中的 15 族为 feature 层（需要显式 IR 供给：intrinsics/流水/warp 集体操作），其余为 baseline 层（标准 arith/linalg→NVVM 下降自然产生）。**G0 FAMS 的修复与对拍**：G0 收割器的 `tma.bulk` 正则从未匹配过、mbarrier 漏掉 `mbarrier_init` 拼写、`exp2?\.approx` 漏 PTX 的 `ex2.approx` 拼写——`EXTRA_FAMS` 修复后，每个条目同时记录 G0-FAMS 复算值与收割时 meta 值的差异（`g0_parity_diff`），无物静默丢失。除族计数外，清单还提取：每族的精确操作码变体集、mma.sync 全签名串（带计数与 kernel 出处）、逐 kernel 资源包络（regs/smem/spill/mma 总数在配置维上的 min–max）、以及逐配置循环结构（`bra` 总数与回边数——标签位置先于分支即回边）。

### 10.4.6 数据对比（收割汇总）

| 来源 | 采集方式 | 规模 | 关键资源事实 |
|---|---|---|---|
| fla（evidence/G0/fla_ptx） | Triton AOT `GPUTarget("cuda",120,32)` | 12 kernel 族 / 22 配置 | mma.sync 3284 条（18 配置）；regs 包络 [40,255]（bwd 全员 255 顶格 + spill：prepare_wy 5348 B、dhu 3788 B、dqkwg 1544 B）；无 TMA/mbarrier/setmaxnreg；两签名：bf16 m16n8k16（2708）+ tf32 m16n8k8（576，仅 dhu/kkt_solve） |
| FlashQLA（evidence/G0/qla_ptx） | TileLang monkeypatch 锁 sm_120a + nvcc | 5 kernel / 5 配置 | mma.sync 2496 条（全 bf16 m16n8k16）；regs 88–128 全员零/微 spill；动态 smem 1–2 KB（TMA 经描述子出场）；TMA bulk.tensor 4d/5d 100 条 + mbarrier 386 + elect.sync 59 + setmaxnreg 16；真循环（bra 107–611，fused_bwd 单 kernel 1600 mma 来自循环体复用） |

### 10.4.7 结论

两条纯 CPU 收割链（Triton AOT 显式 GPUTarget；TileLang 进程级 target monkeypatch + nvcc 离线）把两个手写极点的完整资源表与指令面变成可 diff 的证据文件；T1 契约集 33 族从中规范化提取，分族规则显式、G0 正则缺陷被修复并对拍留痕；FlashKDA 风格点以旋钮面吸收并保留第二批收割路径。

---

## 10.5 覆盖矩阵与逃生舱台账（tools/gdn_coverage.py + evidence/G2/coverage_matrix.md、hatch_report.md）

### 10.5.1 动机

"编译器生成的 kernel 与手写专家 kernel 同构"长期缺乏可核查证据。覆盖矩阵把它变成一张可机器再生成的对照表：本方生成物（ours 列，20 配置）对 T1 契约集逐族判定覆盖状态，对 T2 结构维度逐极点匹配；NO 格子触发逃生舱预算记账。矩阵命令幂等（`matrix` 可在 G2-A 生成集落地前后重复运行，ours 列 PENDING→LIVE）。

### 10.5.2 机制与源码：T1/T2 定义与 ours 列判定

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

### 10.5.3 数据对比一：家族覆盖状态

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

### 10.5.4 数据对比二：fla vs FlashQLA 两极点（结构维度）

| 维度 | fla 包络 | FlashQLA 包络 | ours（20 配置，G2 态） | 注记 |
|---|---|---|---|---|
| regs/线程 | [40, 255] | [88, 128] | [28, 255] | fla 顶格 255 + spill；qla 以 setmaxnreg 钉 ≤128；ours 两端皆含 |
| 动态 smem (B) | [0, 33288] | [1024, 2048] | [0, 0]（ptxas 读数） | ours 读数为 0 是 G2 下降把 workgroup 分配降为设备 malloc 的伪影（ptxas 无静态 smem 行；SASS 证共享窗流量真实存在）；G3 smem 修复后按静态 .shared 声明计 |
| spill (B, st/ld 大者) | [0, 5348] | [0, 1752] | [0, 48032] | fla bwd 重 spill（dhu 3.8 KB）；ours 上界来自 f32 mma 大 tile 配置（G3 重建后 ~0） |
| 异步流水风格 | cp.async.ca 仅 state-push kernel（fwd_h/dhu）；余为 ld.global multistage | TMA cp.async.bulk.tensor 4d/5d + mbarrier + elect.sync 领队（kkt_solve：经典 cp.async.cg） | cp.async.cg/ca ×20 配置 | 两种待覆盖流水皆已成旋钮（pipeline / warpspec+tma） |
| warp 专化 | 无（无 setmaxnreg/elect.sync） | setmaxnreg.inc/dec + elect.sync（4/5 kernel） | ws 两配置 | G2-C 交付 |
| 循环结构（bra/回边） | [0,16] bra，多全展开 | [107,611] bra，真 scf 循环 | 随旋钮 | fla 展开复用、qla 循环复用两种风格 |

### 10.5.5 数据对比三：T2 结构卡（四卡摘要）

- **fla:chunk_fwd_o/BK64_BV64_warps4_stages3**（mma 96、ldmatrix 52、regs 255、smem 24576 B）：ours 最近配置 `ref_ldmatrix`——族 Jaccard 最大者；regs 72 vs 255 → 包络外（0.28×）。
- **fla:dhu_blockdim64/BV64_warps2_stages3**（mma 512 = bf16×256 + tf32×256、ldmatrix 105、stmatrix 80、regs 255、smem 33288 B、spill 3236/3788 B）：ours 最近 `kkt_bf16_mma_p2_t64x64x32_w2`——regs 255 = 255 **PASS**。
- **qla:fused_fwd/default**（mma 384、TMA tensor 31、mbarrier 74、regs 128、smem 2048 B）：ours 最近 `kkt_bf16_mma_tma_ws`——regs 255 vs 128 → 1.99× 包络外。
- **qla:fused_bwd/default**（mma 1600、bra 422/回边 8、setmaxnreg 4、elect 21、shfl 80）：同上最近配置；mma 总量比 0.02×（单 chunk 阶段 kernel 对 fused 全反传 kernel，结构卡度量的是风格可达性而非规模等价）。

T2 卡的意义是**风格可达性**：两个极点的每一结构要素（multistage+顶格寄存器；warp 专化+TMA+循环复用）都在旋钮格内可表达；±30% 判分对逐 kernel 资源给出 PASS/OUT 的诚实记账，而非笼统"等价"。

### 10.5.6 逃生舱台账：1/≤5 与 tf32k8 全案

**预算机制**：覆盖矩阵对每个 NO 格 feature 族预先登记逃生舱方案（自定义 NVVM 操作直书 PTX），总数预算 ≤5。**G2-C 的发现把预算用到 0**：2026 年 10 月主干已把全部"以为要逃生舱"的词汇（setmaxnreg/elect.sync/mbarrier 四件套/TMA 族/prefetch/ldmatrix/shfl）收编在树（NVGPU/NVVM 方言，经 `-gpu-lower-to-nvvm-pipeline` 降到 sm_120a PTX）；`gdn` 方言的 11 个 hatch 操作（GdnHatch.td）是对在树操作的 1:1 薄封装（`gdn-opt --gdn-lower-hatch`），机制演练充分、预算分文未动。后备逃生路径（`nvvm.inline_ptx`/BasicPtxBuilder）验证存在而未用。

**唯一真逃生舱：`gdn.mma.tf32k8`**（G5-G 轮 3；hatch_report ADDENDUM）。签名 `mma.sync.aligned.m16n8k16...` 之 tf32 兄弟 `m16n8k8.row.col.f32.tf32.tf32.f32`（fla 在 dhu/kkt_solve 的精确拼写）在树内不可达，双重根据：`nvgpu.mma.sync` 验证器强制 8×8×128b 基本片（tf32 封顶 K=4）；表生成的 `nvvm.mma.sync` 家族对 m16n8k8 只列 f16/bf16/i8。全案四要素：

1. **操作**：10 个 f32 操作数（a0..a3, b0, b1, c0..c3）→ 4 个 f32（d0..d3）；发射位点 `tgemm()` 在 `tools/gdn_bwd_gen.py`（逐 GEMM 位点的 `--tf32` 旋钮）。
2. **降级路径**：不走 `nvvm.inline_ptx` 而直接 `llvm.inline_asm`，约束 `=&f,=&f,=&f,=&f,f,...`——**early-clobber 必需**（SASS 在 5090 上验证）：BasicPtxBuilder 的普通 `=f` 输出下，ptxas 的内联汇编分配可把 D 四元组映到同一 mma 的 A 四元组（`HMMA.1688.F32.TF32 R12, R12, R10, R16`），tensor core 随即腐蚀操作数——精确整数 K=16 双原子链确定性 118/128 单元失败；`=&f` 禁止输出/输入重叠，`nvvm.inline_ptx` 无法表达。
3. **fragment 布局（硬件导出）**：整数操作数探针（evidence/G5/scratch/tf32k8_probe，D == A@B+C 精确）：a0=(g,2t)、a1=(g+8,2t)、a2=(g,2t+1)、a3=(g+8,2t+1)；b0=(2t,g)、b1=(2t+1,g)；d0..d3 用 bf16-k16 的 D 布局（列 = 2·tig）。A 寄存器次序（行半外层）与打包直觉不同——探针导出而非文档臆断。
4. **实测影响**（bwd B1T2048H16，共载中位）：dqkwg 4.71→2.75 ms（1.71×，与 r4/c4r 并行化合并计）、wyb 2.65→1.21 ms（2.19×，与 ra/c1r 合并计）、e2e bwd 14.16→11.06 ms（1.28×；相对 fla 0.073×→0.093×）；数值 6/6 PASS 于两个门形状；24 个 f32-GEMM 位点中 16 个 tf32 化、5 个为数值保持标量、3 个因 dhu 寄存器 spill 回归保持标量（逐位点 bisect 表见 evidence/G5/bwd_report.md 轮 3）。

**台账：1/≤5。**

### 10.5.7 结论

覆盖矩阵把"同构"变成三态可审计的对照：终态 8/19/6（33 族），全部 PARTIAL 残差为变体级（拼写/向量宽度/代码生成旗标）而非能力缺口；mma 签名层双 YES（其一经逃生舱达成）；T2 结构卡证明两个手写极点的每个结构要素都在旋钮格内；逃生舱预算 1/≤5，且唯一一例带有完整的必要性证明、正确性规则与实测收益。支柱二生成的代码"凭什么信"，至此有了与性能声明互证的覆盖声明。

---

## 10.6 旋钮库：20 个配置面与敏感性（tools/gdn_l2_gen.py）

### 10.6.1 动机

求解器的决策变量必须能物化为可编译工件，否则"编译期调度"是空转。旋钮库是旋钮向量的**具体化生成器**：每个配置生成 `payload.mlir`（含旋钮头部注释）+ `transform.mlir`（旋钮的 transform 载体）+ `config.json`；`--json` 接口直接消费求解器输出的配置字典（回环的生成环节）。它同时是覆盖矩阵 ours 列的全部来源（20 配置）与敏感性测量（旋钮 → PTX 可验证变化）的实验台。

### 10.6.2 机制与源码：validate 与合法性规则

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

### 10.6.3 数据对比一：20 配置面全表

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

### 10.6.4 数据对比二：逐旋钮敏感性（PTX/SASS 可验证 delta；evidence/G2/gen_report.md gate (c)）

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

### 10.6.5 与求解器的一致性及已发现的计数 quirk

生成器 `validate()` 的 smem 模型与求解器字节模型有一处**已发现并文档化的差异**：o 阶段每 GEMM 各分配一对 tile buffer（n_gemm=2），生成器的 `pair` 公式未乘 n_gemm（下计），求解器字节模型乘之（`pair·max(pipe,1)·n_gemm`，solver_report §1 点名的"found quirk"）。求解器侧为准（其模型对 PTX/SASS 复核），生成器侧该差额在 220 KB 生成上限内不产生非法配置，故未回改生成器——差异本身作为两模型交叉验证的产物记录在案。这也是"同一合法性规则双编码"的价值：分歧点即审计点。

### 10.6.6 编译链

每配置经 `scripts/run_l2.sh`：`mlir-opt -transform-interpreter` →（幸存 linalg 的 `-convert-linalg-to-loops` 兜底）→ `gdn-opt --gdn-lower-hatch`（hatch 配置，在 transform 合并**之前**跑在裸 payload 上——gdn-opt 不能解析 transform.structured.*）→ `-gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a cubin-features=+ptx87 cubin-format=assembly"`（PTX 以 `#gpu.object<..., assembly>` 属性挂载，`gdn_extract_ptx.py` 解 MLIR 字符串转义出 kernel.ptx；`.version 8.7 / .target sm_120a`）→ 独立 `ptxas-blackwell -v -arch=sm_120a` → `nvdisasm -c`；产物五件套 + INDEX.json（20/20 ok）。

### 10.6.7 结论

旋钮库把求解器的抽象决策变量落成 20 个全编译配置面，逐旋钮敏感性在 PTX/SASS 级可验证（每个旋钮都有可指认的 delta）；与求解器的双编码合法性规则互为镜像并暴露出一个真实计数 quirk；tma 旋钮的描述子契约由生成器单点拥有。至此回环的"生成"与"编译"两环闭合。

---

## 10.7 标定与成本模型的下游复用：静态指令分析（tools/g5n_static_analysis.py）

### 10.7.1 动机

支柱二的信任链不止于 G3 的排序门：当 G5 的 bwd kernel 落地后，需要在不占 GPU 清窗的前提下回答"每个 kernel 距屋脊多远、被什么资源绑定"。`g5n_static_analysis.py`（G5-N 轮 7）把本章两件资产——收割器的特征提取器与成本模型的 sm_120 常数/标定速率——复用为逐 kernel 静态分析器，证明标定表与结构常数是**可复用的项目级基础设施**而非一次性的排序零件。

### 10.7.2 机制与源码

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

### 10.7.3 数据对比

分析器头部声明并携带修正后的标定速率（mma 峰 115.5 TFLOP/s ×32 修正、smem 读 113.05 TB/s、DRAM 824.6 GB/s——共载下界口径，注明 evidence/G3/calibration.json + G6/search_cost.md 为源）；输出每 kernel 的占用（CTA/SM、warp 槽百分比、限制资源）、比值表（lds/sts/ldsm/ldg/fp32/bar 每 mma 位点）、每 chunk 强度与屋脊下界组、绑定资源分类与 `实测/下界` 倍数。fwd 参考行同样处理（mono/fused 模块逐 entry 拆分）。该分析在 G5 后续轮次（如 r23-f 的 kkt 屋缘判定：staging 实测 826 GB/s 对 D2D 同窗 765 GB/s）中成为瓶颈归因的标准工具之一。

### 10.7.4 结论

标定表与成本模型常数在排序门之外被下游静态分析直接复用（×32 修正逻辑在第二个消费者处显式重现，形成双点一致性）；占用模型、屋脊下界与绑定分类把"编译期成本模型"延伸为"编译期瓶颈诊断"，均零 GPU、确定性、可重跑。

---

## 10.8 本章小结

本章完整呈现了支柱二的落地与它的信任锚：（i）**Z3 求解器**以固定种子/rlimit/固定序把旋钮空间（368/465/204）亚秒级穷尽为逐字节可复现的全序排名，并以 SAT 探针补全与 rlimit 重试防御两个真实的 Z3 行为伪影；（ii）**解析成本模型**以零逐配置拟合契约达到 Spearman 0.8962（kkt 0.936 / o 0.829，n=18），对全部结构常数 ±2× 与实测 ±30% 噪声稳健，计数分子对静态 PTX 18/18 精确，幅度级盲区（`.ca` 串行化、mma 路径 3–5× 低估）如实记录而秩级正确；（iii）**一次性标定**以六个微基准覆盖全部速率分母，×32 warp 级计数陷阱被检测规则统一修正，共享窗披露与幂等干净窗刷新并存；（iv）**参考收割器**以纯 CPU 链（Triton AOT GPUTarget / TileLang monkeypatch + nvcc / ptxas-blackwell）固化两个手写极点的资源表，提取 T1 契约集 33 族；（v）**覆盖矩阵**给出终态 8 YES / 19 PARTIAL / 6 NO 与 mma 签名双 YES，T2 结构卡证明两极点结构要素皆在旋钮格内，逃生舱台账 1/≤5 且唯一一例（tf32k8）带完整必要性与正确性证明；（vi）**旋钮库**以 20 个配置面与逐旋钮 PTX 级敏感性把决策变量物化为可编译工件，并与求解器互为合法性镜像；（vii）标定与模型常数经静态分析器**下游复用**为屋脊诊断工具。调度离开了设备、搬进了编译期，且每一步——排序、生成、编译、复核——都有可再生的证据文件。本章各论断的证据源：evidence/G3/{solver_report.md, calib_report.md, validation.json, config_times.json, calibration.json, calibration_shared.json, solver_kkt.json, solver_kkt_v3.json, solver_o.json}；evidence/G2/{gen_report.md, coverage_matrix.md, ref_inventory.md, hatch_report.md}；evidence/G0/{fla_ptx, qla_ptx}；evidence/G6/{search_cost.md, coverage_hatch_summary.md}；evidence/G5/bwd_report.md（tf32k8 位点表）。
