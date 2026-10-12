# 论文骨架 outline.md — GDN-MLIR（G7-A 交付物，2026-10-09）

> 本文件是论文的逐节骨架：每一小节给出内容要点（3–8 条）与喂给它的证据文件。
> 纪律：**每一个实验性论断必须标注其 evidence/ 源文件**；正文数字一律以证据文件为准
> （本骨架中的数字已逐一回读核对到下列文件，凡与早期计划文档不一致处，证据文件胜出）。
> 写作规则（作者自约束，全文适用）：严格学术体、零日志腔；贡献亮点量化到具体对象
> （写"1.592×@B1T2048 交错终判"，不写"显著提升"）；实验章为连续探索长卷——同一主题
> 单节展开、失败保留为推进动力，不做成功罗列；核心机制贡献点级展开，禁一句带过、
> 禁括号化；公式以可渲染 LaTeX 给出；黑话先定义后用；协议与账目自洽（分母口径、
> 窗口口径、共享 GPU 披露全部写明）；负结果与开放战线如实呈现。

- 题目（EN）：*Schedule-as-IR: Compile-Time Deterministic Scheduling for the Gated
  DeltaNet Operator on Stock MLIR, Validated to the PTX Level*
- 题目（CN）：《调度即 IR：基于原生 MLIR 的 Gated DeltaNet 算子编译期确定性调度及其 PTX 级验证》
- 目标会议：MLSys / ASPLOS / CGO（按 PLAN.md G7）
- 试验平台：RTX 5090 Laptop（消费级 Blackwell，sm_120a，82 SM，GDDR7，无 tcgen05）
- 全局口径声明（写入正文每张表的脚注）：
  1. **交错终判（interleaved adjudication）**：同一 GPU 窗口内我方与基线交替成对计时
     （per-pair 比较取多数/中位），由 `tools/gdn_final_adjudicate.py` 执行——除非另有说明，
     一切加速比的分母是**同窗口交错测得的基线**，不是跨窗口中位数之比。
  2. **共享 GPU 披露**：全部 GPU 计时取自机会窗口（共居任务 pid 3411456 常驻，约 18 GB
     显存、22–99% 利用率），所有源 JSON 标注 `shared_gpu: true`；绝对毫秒数按受挤占的
     下界解读，加速比以交错对内比较消解共载漂移；干净窗口幂等重跑协议已留
     （`evidence/G3/calib_report.md` §5）。
  3. **数值分母**：前向输出对 fp64 JAX 参考的逐元素最大绝对差；反向六梯度对 fp64
     VJP 参考的容差契约（fla 惯例 atol/rtol）。

---

## Abstract（全文成稿见 abstract.md，约 200 词）

要点（数字锁定）：
- 论点：linear-attention 算子 kernel 工程长期是手写领地；我们把**调度本身做成 IR 旋钮**——
  原生（stock）MLIR 渐进下降栈，transform dialect 脚本 + 方言属性即参数化工件；
  编译期确定性 SMT（Z3）+ 一次性标定解析成本模型完成调度决策，无运行时 autotune。
- 验证方法：PTX 级覆盖证明（T1 指令族 33 族 + T2 结构两极点）对标手写 SOTA
  （fla Triton / FlashQLA TileLang）；端到端交错终判。
- 结果：前向 3/3 形状 ≥1.1×（B1T2048 1.592×；B4T2048 1.05–1.19× 两窗口，配对 A/B
  −148 µs，波动归因基线中位方差；B1T8192 1.253/1.312×）；排序质量 Spearman 0.8962；
  搜索成本 0.8 s 离线 CPU vs fla 39.8 s 在设备 autotune（≈50×）；逃生舱预算 1/≤5。
- 诚实边界：反向 0.126×（8.15 ms vs fla 1.03 ms 同窗），附五轮 7.2× 自提速轨迹与
  全瓶颈账目，作为方法论可解释性的开放战线呈现。
- 证据：`evidence/G5/GATE_REPORT_FINAL.md`、`evidence/G6/GATE_REPORT.md`、
  `evidence/G3/GATE_REPORT.md`。

---

## 1 Introduction（约 2 页）

### 1.1 问题：算子工程是手写领地
- 现代 linear-attention（Gated DeltaNet, GDN）前向/反向 kernel 由 Triton / TileLang /
  CUTLASS 手写专家kernel占据：调度知识（tile 形状、流水深度、warp 专化、融合边界）
  固化在 kernel 源码与 autotune 空间里，不可作为可变换、可验证、可复用的编译器 IR。
- 代价一：调参成本在部署侧支付——fla/Triton 首次调用按 shape 键在设备上编译并计时
  每个候选配置：实测 fwd 16.55 s（25 配置/5 kernel）、bwd 23.26 s（23 配置/4 kernel），
  每进程每形状重付（`evidence/G6/search_cost.md` §b）。
- 代价二：表达能力被 DSL 边界锁死——TileLang 的 Z3 modulo 调度器只存在于 Ascend 后端，
  CUDA 侧退化为 autotune（`docs/PLAN.md` §7 关键事实）。
- 代价三：验证缺失——"编译器生成的 kernel 与手写 kernel 等价/同构"缺乏可核查证据，
  性能声明无法与覆盖声明互证。

### 1.2 论点：Schedule-as-IR
- 定义（先定义后用）：**调度即 IR**——把 kernel 的全部调度自由度（tile、流水、warp
  专化、融合边界、数值宽度、访存路径）表达为原生 MLIR 的 transform dialect 脚本参数
  与方言属性，调度决策 = 在合法性约束下对解析成本模型求最小，由编译期确定性 SMT
  求解器执行；不引入新 IR 层、不 fork 编译器、不手写 PTX（预算化逃生舱除外）。
- 与 XLA/TPU 四支柱的对位（`docs/PLAN.md` §0）：语义单一事实源（`gdn` 方言 L0 +
  JAX 可执行规约 + vjp）↔ HLO+autodiff；编译期确定性调度（Z3 + 一次性标定）↔ 非运行时
  autotune；分层可验证 IR（L0–L4 每层 oracle）↓ 分层验证；布局系统（smem/寄存器
  layout 旋钮）↓ 布局传播。
- 论文主张的完整证据链 = 正例（前向胜）+ 可解释的开阔地（反向五轮轨迹与瓶颈账目），
  二者共同支撑"调度即 IR"（`evidence/G5/GATE_REPORT_FINAL.md` §bwd 五轮账目）。

### 1.3 贡献 C1–C4（全文见 contributions.md）
- C1 覆盖证明方法论：T1 指令族 33 族契约 + T2 两极点结构匹配；最终 8 YES/19 PARTIAL/
  6 NO（feature 层 15 族中 8/6/2），逃生舱 1/≤5（`evidence/G6/coverage_hatch_summary.md`）。
- C2 编译期 SMT 调度：Z3 枚举 368/465 合法配置逐字节确定性复现；排序 Spearman 0.8962
  （kkt 0.936 / o 0.829，n=18）；搜索成本 ≈50× 低于在设备 autotune
  （`evidence/G3/GATE_REPORT.md`、`evidence/G6/search_cost.md`）。
- C3 前向端到端胜利：3/3 形状 ≥1.1× vs fla Triton（交错终判口径，见 abstract 数字）；
  数值 4/4（o ≤6.85e-4、ht ≤7.93e-4 vs fp64；T8192 分段分裂间位级一致）
  （`evidence/G5/GATE_REPORT_FINAL.md`、`fwd_verify_round5.json`）。
- C4 开放发现账本：stock-MLIR 工具链缺陷清单（workgroup alloc→device malloc、
  常量捕获爆炸、wait_group-0 机器约束、ptxas D-onto-A 操作数腐蚀与 early-clobber 规则、
  mma fragment 布局律、TMA 128B 对齐/维序契约、H2D 重分级掩盖效应），每条以探针在
  硬件验证（`evidence/G6/failure_longscroll.md`、`evidence/G4/GATE_REPORT.md`）。

### 1.4 结果预览与诚实边界
- 前向胜利表（1 行/形状）+ 反向一行（0.126×，指向 §5.5 长卷）。
- 明确声明：单 GPU（sm_120a 消费级 Blackwell）、共享窗口口径、单算子范围（§6 展开）。

---

## 2 Background & Related Work（约 2 页）

### 2.1 GDN 算子与 chunked WY 数学
- 逐步递推定义（gate 先于 delta 更新的次序是约定的一部分）：
  $$h_t = e^{g_t}\,h_{t-1} + \beta_t\, k_t\left(v_t^\top - k_t^\top e^{g_t}\, h_{t-1}\right),\quad o_t = q_t^\top h_t$$
- chunked WY 表示（FLA 同构阶段 DAG）：块内构造
  $$M[i,j] = -\beta_i\,(k_i\!\cdot\! k_j)\,e^{cd_i-cd_j}\ (i>j),\qquad T = (I+M)^{-1}$$
  （前代换求逆），$w = T(\beta v)$，$u = T(\beta k\, e^{cd})$；跨块状态递推
  $$S_n = e^{cd_{\mathrm{last}}}\,S_{n-1} + (k \odot e^{cd_{\mathrm{last}}-cd})^\top (w - u\,S_{n-1})$$
  输出 = 块内注意力项 + 状态项（对角保留、同 decay 因子，仅严格上三角掩蔽）。
- 反向结构：dh 逆推 → 重算 h/w/u → dqkwg → dv/dv_local → dy_bwd（与 fla v0.6.0 阶段
  同构）；dA 的**双侧掩蔽**精确公式（贡献点，展开见 §3.6 与 figures_plan.md F8）：
  $$\mathrm{d}A = -\operatorname{stricttril}\!\big(T^\top\,\operatorname{stricttril}(G)\,T^\top\big),\qquad
    G = \mathrm{d}v_p\,(\beta v)^\top + \mathrm{d}W\,(\beta k\, e^{gc})^\top,\quad \mathrm{d}W = -\mathrm{d}v_p\, S_n^\top$$
  单侧修正式对 C≥3 不成立（符号级验证）；与 fla `prepare_wy_repr_bwd` 计入其转置
  A 存储后等价（`evidence/G1/GATE_REPORT.md` §数学成果、`docs/L0_SEMANTICS.md`）。
- 术语定义：chunk（块长 C）、WY 表示、decay 门 g（自然对数域，kernel 内 exp2 换域）、
  差分状态 S、GVA/varlen 等仅作属性面定义（本文实验不覆盖 GVA/varlen，见 §6）。

### 2.2 手写 kernel 生态与两个设计极点
- fla（flash-linear-attention，Triton）：multistage `ld.global`/`cp.async` 路线，反向
  kernel 全部 255 regs 顶格 + spill（prepare_wy 4808 B …）；G0 收割 22/22 配置资源表；
  dtype 真相（A/h/dh/do=bf16，ht/dg=fp32，solve_tril fp32→bf16）
  （`evidence/G0/GATE_REPORT.md`、`evidence/G0/fla_ptx/SUMMARY.md`）。
- FlashQLA（TileLang）：warp 专化极点——setmaxnreg+elect.sync+mbarrier+TMA，全部
  ≤128 regs 零 spill（fused_bwd 单 kernel 1600 mma.sync）；5/5 kernel 资源表
  （`evidence/G0/qla_ptx/SUMMARY.md`）。
- 两极点 = 本文 T2 结构覆盖的靶（§5.2）；sm_120 在 FlashQLA 侧仅 fwd bf16 支持，
  无快 GDN 反向——这正是本文反向对标选 fla Triton 的原因（`docs/PLAN.md` §7）。

### 2.3 编译与调度系统（点名对比）
- TileLang：tile DSL + TVM 分析器；Z3 modulo-scheduler 仅 Ascend 后端（变量=开始时间
  +buffer 版本，目标 min II）。本文把该机制**上移一层**移植到 CUDA/sm_120：决策变量=
  旋钮向量，约束=资源容量+结构合法性，目标=解析时延；CUDA 侧无先例
  （`docs/PLAN.md` §3/§7、`evidence/G3/solver_report.md` §1）。
- FlexLA：linear-attention 家族 kernel 生成器——覆盖面广但调度靠模板+搜索，非
  stock-MLIR IR 旋钮、无 SMT 约束求解、无 PTX 覆盖证明（gap：表达即模板，验证即计时）。
- Hexcute / TileLoom：层级 tile 调度框架——调度是显式编程模型而非可求最小化的目标，
  不提供确定性编译期搜索与硬件覆盖对拍。
- CAKE：自改进步 kernel 优化闭环（失败→约束规则库/IR 原语/归因表）——本文采用其
  失败→规则纪律（`failure_longscroll.md` 的组织方式），但搜索由 SMT 显式枚举替代。
- PEAK / BaCO：代价模型代理 +（贝叶斯）autotuning 的代码参数优化——在设备或仿真
  侧迭代测量；本文成本模型**零逐配置拟合常数**、一次性标定、离线确定性，与拟合式
  代理形成方法论对照（`evidence/G3/solver_report.md` §2–3）。
- 空白陈述（gap statement，§2 收束）：上述系统无一同时做到「原生 MLIR 渐进下降 +
  编译期确定性 SMT 调度 + PTX 级覆盖验证」；本文填的正是这个三角。

### 2.4 MLIR 栈与 transform dialect
- transform dialect 作为参数化工件（script = artifact）；NVGPU/NVVM lowering 面；
  2026-10 主干已收录 setmaxnreg/elect.sync/mbarrier/TMA/ldmatrix——"逃生舱前提失效"
  是本文的好消息叙事（`evidence/G2/GATE_REPORT.md` §头条发现 1）。
- 树内 sm80 transform-mma 集成测试路径为本文起步锚（5090 真跑通过，带 teardown 瑕疵
  备注）（`evidence/G0/sm80_smoke.txt`）。

---

## 3 Design: Schedule-as-IR（约 3.5 页）

### 3.1 总体架构：四支柱
- 图 1（knob-lattice，见 figures_plan.md F1）：L0 语义 → L1 算法结构 → L2 调度 →
  L3 流水/向量化 → L4 目标；每层有独立 oracle（§3.6）。
- 支柱展开（每个一段，禁括号化）：语义单一事实源；编译期确定性调度；分层可验证；
  布局系统。对应 `docs/PLAN.md` §0 四条映射。

### 3.2 旋钮格（knob lattice）L0–L4 表
- 表：层 | 旋钮 | 载体 | oracle。L0：dtype/chunk_size/use_g/use_gk/h0/β-sigmoid/
  varlen/GVA（GdnConfigAttr）；L1：chunk∈{16,32,64}、solve 变体、recompute-vs-
  materialize、decay 域（log2/exp2）、state 精度、门融合位；L2：BT/BK/BV/BC、grid
  映射（rows×V-split×两遍 token-并行 WY）、82-SM 并行度适配、num_warps、融合边界
  （FLA 6-kernel 任意切分 ↔ 单 kernel persistent 两档）；L3：multibuffer factor、
  create_async_groups/rewrite_copy_as_tma、pipeline_shared_memory_copies depth、
  rewrite_matmul_as_mma_sync（m16n8k16 bf16→fp32；sm_120 无 tcgen05）、手动 warp
  专化（warp 词汇+setmaxregister）、swizzle/补列；L4：`gpu-lower-to-nvvm-pipeline
  ="cubin-chip=sm_120a"`。
- 论断：FLA/FlashKDA/FlashQLA 的风格差异 = 配置空间中三个点；复现 = 调参不重写
  （`docs/PLAN.md` §2；两极点实证 `evidence/G2/coverage_matrix.md` §3–5）。

### 3.3 Z3 公式化：变量、约束、目标、确定性
- 变量（固定声明序）：dtype、prec、tile_M/N/K、warps、pipeline、mma、warpspec、tma；
  域 tile∈{32,64,128}、warps∈{2,4,8}、pipeline∈{0,2,3}。
- 约束三类：结构合法性（整除、mma fragment 对齐 $M\equiv 0 \pmod{16}$、bf16
  $K\equiv 0\pmod{16}$、$N \equiv 0 \pmod{8\cdot\text{warps}}$、流水深 ≤ K 循环趟数）；
  旋钮耦合（warpspec⇒kkt∧bf16∧mma；tma⇒warpspec）；资源容量——smem 字节模型
  $$\text{smem} = (bt\cdot bk + bk\cdot bv)\cdot esz \cdot \max(p,1)\cdot n_{\text{gemm}} \le 98\,\text{KB},\qquad
    32\,(w + w_{\text{spec}})\cdot 255 \le 65536$$
- 目标：最小化解析成本模型；模型非线性（max/混合/除法）→ SAT + blocking 子句枚举
  全合法空间（kkt=368 / o=465），在求解器外排序——TileLang 的 rlimit+固定种子
  确定性纪律被保留。
- 确定性契约：`random_seed=7`、`rlimit=20e6`、固定变量/阻塞序、终排序键
  `(predicted, name)`；两次独立运行 368 项排名逐字节一致（G6 复测再证）
  （`evidence/G3/solver_report.md` §1、`evidence/G6/search_cost.md` §a）。
- 回环设计：解 → transform 脚本实例化 → 编译 → ptxas `-v`/SASS 静态复核 →（窗口）
  top-K 计时；未编译配置 spills=0 排序，命中已编译向量自动拾取实测特征。

### 3.4 解析成本模型与一次性标定
- 模型形式（v2）：
  $$T = T_{\text{launch}} + \operatorname{ov}\!\big(T_{\text{tensor}} \mid T_{\text{ffma}},\ T_{\text{gmem}},\ T_{\text{smem}}\big) + T_{\text{lsu}} + T_{\text{bar}} + T_{\text{spill}} + T_{\text{ex2}}$$
  $\operatorname{ov}=\max$（流水化）或 $\sum$（p0 串行同步拷贝）；nomma 项替换为
  SASS 导出标量环模型 $T_{\text{nomma}} = \text{MACs}\cdot 10\,\text{cyc}/f_{\text{clk}}$
  （39 指令/3 LDS 内环，编译期证据非拟合）；v3 增加 waves/占用项
  $\lceil \text{CTAs}/(82\cdot\text{occ})\rceil$；launch 常数 6.12 µs 由 ref 锚核导出。
- 标定表（单 GPU 窗口、分钟级、<2 GB，一次性沉没成本）：mma.sync m16n8k16
  bf16→fp32 峰值 115.5 TFLOP/s（线程级 ×32 过计已修正）；smem 读 113.1 TB/s；
  gmem L2 驻留读 2144 GB/s / DRAM 825 GB/s（受共载挤占下界）；SFU ex2 3.23 Telem/s；
  bar.sync 往返 7.71 ns；cp.async 16B 1388 GB/s（`evidence/G3/calib_report.md` §1、
  `evidence/G3/calibration.json`）。
- 零拟合契约与自检：速率全部来自标定表或文献级 sm_120 结构常数，**无逐配置拟合**；
  静态特征公式 vs PTX 计数 18/18 精确（mma/ld.shared/bar.sync 计数，cp.async 偏差
  ≤1.6%）（`evidence/G3/solver_report.md` §2）。
- 模型迭代长卷（论文素材，失败保留）：v1 0.7352 → v2 +LSU-issue/device-alloc 项
  0.7475 → nomma 项 SASS 化 **0.8962** → launch 锚定（秩不变）→ v3 waves（单 CTA
  测点集不可分辨；全形状生效，v2↔v3 top-5 翻转 = 占用项在工作，G5 可检验预测）
  （`evidence/G3/GATE_REPORT.md` §模型迭代长卷）。
- 已记录盲点（如实）：p0 的 `.ca` 串行化幅度未建模（实测 5×，模型仅 2.4 µs）；
  mma 路径绝对时间系统性低估 3–5×（秩不变）；DRAM 带宽为共载下界。

### 3.5 逃生舱机制与账本
- 机制：GdnHatch 方言 1:1 薄封装 → 在树 nvgpu/nvvm op（11 op，机制已演练、预算
  未动）；真逃生舱 = `gdn.mma.tf32k8` → `llvm.inline_asm`（early-clobber `=&f`）。
- 账本（1/≤5）：唯一使用的理由（NVVM 表生成限制：nvgpu.mma.sync 强制 8x8x128b
  基本片使 tf32 封顶 K=4；表列 nvvm.mma.sync 的 m16n8k8 仅 f16/bf16/i8）；不用
  early-clobber 时 ptxas 合法地把 D 四元组映射到 A 四元组（SASS `HMMA R12,R12,R10,R16`），
  K=16 链确定性 118/128 单元失败——fragment 布局由硬件探针导出（a0=(g,2t)…、
  d 的 N=2·tig 律）；实测收益 bwd e2e 14.16→11.06 ms（1.28×）、dqkwg 1.71×、
  wyb 2.19×，16 个 GEMM 位点，6/6 数值过门（`evidence/G6/coverage_hatch_summary.md` §3、
  `evidence/G2/hatch_report.md` ADDENDUM、`dialect/include/Gdn/GdnHatch.td`）。
- 叙事升级：计划中的其余 hatch 族（setmaxnreg/elect.sync/mbarrier/TMA/prefetch/
  ldmatrix/shfl）2026-10 主干已全部在树——原生 MLIR 表达力边界已推进到覆盖
  TileLang/CUTLASS 级 warp 专化词汇表；唯一的真 hatch 是 NVVM 表生成限制而非调度
  限制（`evidence/G2/GATE_REPORT.md` §头条发现 1）。

### 3.6 验证塔与 dA 公式
- 分层 oracle：L0↔JAX 规约（fp64）；L1 CPU linalg 执行 ↔ JAX dumps；L2 编译 +
  ptxas/SASS 静态资源复核；L3/L4 GPU 上 fp64 孪生逐元素对拍（同嵌入输入）；
  e2e fla 容差契约。语义闭合三角：MLIR-L1 ⇄ JAX ⇄ fla naive（`evidence/G1/GATE_REPORT.md`）。
- dA 双侧公式与 dg 链约定（展开成正文一段 + F8 数学框）：先掩蔽再两侧乘 $T^\top$；
  dg 的 o_intra 项配 raw $(\mathrm{d}O\,v_n^\top)$；Bhat 链 dg 项 = 严格前缀和；
  反传两遍结构（pass1 只存入口态，pass2 正步长逆走 chunk）。
- 反向扫描的仿射携带形（§5.5 的数学基础）：
  $$X_{n-1} = \big(2^{g_{cl}} I - W^\top \hat{B}\big)\,X_n + \big(Q_g^\top - W^\top P^\top\big)\,\mathrm{d}O$$
  对真序贯迹 2–3 ulp（fp64）；附赠结构发现：门算子超收缩
  $\rho(A_n)\le 2.3\times 10^{-18}$、$\|A_n\|\le 2.2\times 10^{-17}$——弱门场景安全性
  的结构性保证（`evidence/G5/GATE_REPORT_FINAL.md` §bwd、`evidence/G5/bwd_report.md` r5）。

---

## 4 Implementation（约 2 页）

### 4.1 代码栈组成
- LLVM Oct-2026 worktree 有限构建（mlir-opt/mlir-runner/cuda-runtime/Python 绑定，
  1.9 GB Release）；`gdn` 方言（ODS + 展开 pass + gdn-opt）；生成器
  `tools/gdn_l2_gen.py`（payload+transform 参数化）；执行链 `scripts/run_l2.sh` →
  `tools/gdn_smem_fix.py` → `tools/gdn_time.py`/`tools/gdn_verify.py`；终判器
  `tools/gdn_final_adjudicate.py`；求解器 `tools/gdn_solver.py` + `tools/gdn_costmodel.py`
  （`evidence/G0/GATE_REPORT.md`、`evidence/G2/gen_report.md`）。
- PTX 收割方法（基线侧）：fla Triton AOT `GPUTarget("cuda",120,32)` 自带 ptxas `-v`；
  FlashQLA TileLang 错 target + `get_kernel_source()` + nvcc -ptx；SASS 特征提取器
  nvdisasm；ptxas `-v` 文法（regs/smem/spill）（`evidence/G0/GATE_REPORT.md`）。

### 4.2 sm_120 现实（消费级 Blackwell 的编译目标真相）
- 无 tcgen05：消费级 Blackwell 用 Ampere 风格 `mma.sync`（cluster 1×1×1；99 KB
  smem/CTA opt-in，静态上限 101376 B；82 SM）（`docs/PLAN.md` §7、`evidence/G5/fwd_report.md`）。
- mma 管线配方 = multibuffer + async groups + pipeline_shared_memory_copies +
  rewrite_matmul_as_mma_sync；树内样例 sm80 transform-mma-sync-matmul 集成测试
  为模板（`evidence/G0/sm80_smoke.txt`）。
- transform 拼写现状（Oct-2026）：`tile_using_for`、`multibuffer <factor,skip_analysis>`、
  `create_async_groups bypass_l1`（否则 `.ca`）、pipeliner 要求扁平 K 循环体、smem
  matmul 不能走 vectorize（`evidence/G2/GATE_REPORT.md` §过程沉淀）。

### 4.3 工具链缺陷发现与修复（开放发现账本，C4 的展开）
每条：现象→探针→根因→修复→验证（全部硬件复现）：
1. **workgroup memref.alloc → device malloc + cvta**：`-gpu-lower-to-nvvm-pipeline` 把
   workgroup 空间 alloc 降为设备侧 malloc 调用 + 通用堆指针 cvta；裸 ptxas cubin 上
   CALL.ABS 蹦床未解析 → 首跑即 `CUDA_ERROR_ILLEGAL_ADDRESS`。修复 = PTX 级静态
   smem 重写（`tools/gdn_smem_fix.py`：删 malloc 调用序列、`mov.u64 %rd, __gdn_smemN`、
   `.shared .align 16 .b8 name[size]`——与 MLIR 自身为 mbarrier 发的惯用式相同）
   （`evidence/G3/calib_report.md` §2 Gap 1）。
2. **常量捕获爆炸**：tiling/tiling-hoist 把 launch 区域常量抬回宿主 → 最多 ~140 个
   `.b64` kernel 参数（一个 bf16 pad 变 `.param .align 2 .b8[2]`）；同工具解析宿主
   `gpu.launch_func args(...)` 逐参替换 `ld.param::entry` 为立即数，18/20 配置成为
   真 zero-arg kernel（`evidence/G3/calib_report.md` §2 Gap 2）。
3. **wait_group-0 机器约束（源码级证明）**：`create_async_groups` 转换的每段 staging
   后**立即** `cp.async.wait_group 0`（`NVGPU/Transforms/CreateAsyncGroups.cpp`）——
   transform 路线结构上永远无法让预取与计算重叠；手动 `device_async_copy` 原样透传，
   wait 位置归我方（ping-pong 零增益负结果的根因，§5.5/失败账本 §3）
   （`evidence/G6/failure_longscroll.md` §3）。
4. **ptxas D-onto-A 操作数腐蚀与 early-clobber 规则**：见 §3.5——无 `=&f` 时
   ptxas 合法复用 A 寄存器为 D，确定性 118/128 失败；带探针的 fragment 布局推导。
5. **mma fragment 布局律**：D-fragment 列索引 = `2·threadID_in_group`（非 B-fragment
   的 groupID）；错用致 kkt 1168/2080 单元覆盖（G3 疑点 → G4 定位修复，实测 ==
   bf16 量化仿真预测 9.765625e-4）；ldmatrix 需 per-lane 象限地址（uniform 基址 =
   行广播 A）（`evidence/G4/GATE_REPORT.md` §2.1/§2.5）。
6. **TMA 接线三缺口**：宿主端 CUtensorMap 编码契约（秩/CUDA 序 globalDim/boxDim）、
   维序契约（坐标 K 最内；B tile 必须 K 连续布局）、`.shared` 目的 128B 对齐
   （16B 对齐时 ptxas 放 16B-非-128B 偏移 → `MISALIGNED_ADDRESS`，compute-sanitizer
   定位 `UTMALDG.4D`）（`evidence/G4/GATE_REPORT.md` §2.3）。
7. **验证 harness 的 H2D 重分级掩盖效应**：逐级重初始化中间量的 harness 掩盖了
   生产者/消费者契约破坏（bf16-B̂ 潜伏 bug 三轮不可见）——教训升级为端到端验证规约
   （`evidence/G6/failure_longscroll.md` §12）。
- 其余 12+6 条负结果与教训（staging 冗余、pipeliner suppress 子句、Amdahl 反驳、
  L1 carveout、tf32 双负、重建混沌、树扫描负经济学、async 零投递悬案、窗口方差、
  bf16 误诊+grading-twin bug 等）完整保留于失败长卷并编入 §5 各节作为推进动力
  （`evidence/G6/failure_longscroll.md`）。

### 4.4 复现工程
- 全链 CPU 可复现（编译、求解、静态覆盖）；GPU 步骤幂等脚本 + 证据目录结构
  （payload/transform/ptx/cubin/ptxas_v/sass per config）；G6 补充测量工具
  `tools/g6/g6_fla_autotune_cost.py` + `g6_solver_timing.sh`（`evidence/G6/GATE_REPORT.md`）。

---

## 5 Evaluation：连续探索长卷（约 5–6 页）

> 章首协议段（先行声明，全章共用）：交错终判定义、共享 GPU 披露、每表分母口径、
> 基线版本（fla v0.6.0 纯 Triton 路径 `FLA_FLASH_QLA=0`、Triton AOT 收割口径）、
> 形状集 B1T2048/B4T2048/B1T8192（H16，K=V=128，bf16 io/fp32 门）。

### 5.1 前向对 fla Triton 的端到端胜利
- 交错终判表（正式判定口径）：B1T2048 **1.592×**（第 4 轮 1.292×→第 5 轮 F1UW）；
  B4T2048 **1.05–1.19×**（两窗口；受控 A/B 实测 −148 µs；我方中位稳 0.980–1.004 ms，
  波动全部来自 fla 中位方差 1.056 vs 1.165 ms，fla min 稳 0.865–0.875）；B1T8192
  **1.253×/1.312×**（第 4 轮 1.021× 平手→过线；我方 1.126 ms，f1 .471/f2 .576/f3 .431）
  （`evidence/G5/GATE_REPORT_FINAL.md`、`final_adjudicate_g5k_run{1,2}.json`）。
- 方差披露段：单窗中位数之比对有噪对手夸大确定性——本文用 per-pair 交错判据 +
  配对 A/B + 稳定 min 三重交叉；B4T2048 案例完整解剖为方法学段落
  （`evidence/G6/failure_longscroll.md` §10）。
- 前向五轮轨迹：staged 1.597→0.663→0.262（静窗）→0.215 ms（r5 配对 A/B）/0.373 ms
  （r5 共载中位）；各轮杠杆归因（K-循环 transform 流水 2.2–2.7×、LDSM 补列
  473→249 µs、w4/bv32 重调、F1UW −123 µs@T8192）内联展开（`evidence/G5/fwd_report.md`
  各轮、`evidence/G6/knob_ablation.md` §2）。
- 第 5 轮决胜解剖：预测的 F2 barrier 削减被受控 A/B 否决（2→1 bar 反 +19.7 µs，+5 KB
  smem 挤占 L1 carveout）——真杠杆是 F1 的 uw 段重构；附审计纠错（在产 F2 实为
  2 bar/chunk；round-4 曾被陈旧 cubin 喂料，已清重建）（`evidence/G5/GATE_REPORT_FINAL.md`）。
- 数值：4/4 staged+fused；o ≤6.85e-4 / ht ≤7.93e-4 vs fp64（T8192 fp64 门 7× 裕量；
  分段分裂 S=1/4/8 位级一致）（`fwd_verify_round5.json`、`fwd_verify_round4_t8192.json`）。

### 5.2 PTX 级覆盖矩阵与逃生舱（T1+T2）
- T1 特征层终态表（15 族）：8 YES / 6 PARTIAL / 2 NO；mma.sync 经 hatch #1
  PARTIAL→YES；全部残差为变体级（拼写/宽度/旗标）而非能力缺口；stmatrix 为唯一
  特征级 NO 且经测量判为错误模式（fragment 映射存储 +40% 慢）（`evidence/G6/coverage_hatch_summary.md` §1、§4）。
- T1 签名层：bf16 m16n8k16（我方 552 条/10 配置 vs fla 2708/qla 2496）+ tf32
  m16n8k8（hatch #1，fla 的精确拼写）双 YES；33 族总账 8/19/6（G2 收官 7/20/6 →
  hatch 翻转）（同上 §1d）。
- T2 两极点陈述：fla 极点（multistage、255 regs 包络 [40,255] ⊂ 我方 [28,255]）与
  FlashQLA 极点（warp 专化 + TMA：5 个 TMA op 替换 1026 条经典拷贝；setmaxnreg
  dec96/inc224 把 spill 22,844 B→48 B = FlashQLA 量级）**皆可表达且经硬件验证**
  （`evidence/G2/coverage_matrix.md` §3–5、`evidence/G4/GATE_REPORT.md`）。
- 逃生舱账本 1/≤5 + 剩余缺口"非 GDN 所需"评估表（cache_hint/ftz/变体拼写等）
  （`evidence/G6/coverage_hatch_summary.md` §3–4）。
- 覆盖≠性能的边界声明：PTX 匹配是必要非充分；静态覆盖为 gate，性能以 §5.1 计时为准。

### 5.3 编译期调度质量：排序精度与搜索成本
- 排序精度：Spearman **0.8962**（kkt 0.936 / o 0.829，n=18 实测配置）；稳健性——
  全部结构常数 ±2× → ρ∈[0.882,0.902]；实测 ±30% 噪声 → ρ≥0.856；静态特征自检
  18/18 精确（`evidence/G3/validation.json`、`evidence/G3/solver_report.md` §2/§4）。
- 模型迭代长卷（0.7352→0.7475→0.8962，见 §3.4）在此以图呈现；v2↔v3 top-5 翻转
  的占用项预测在 G5 全形状可检验（`evidence/G3/solver_kkt{,_v3}.json`）。
- 搜索成本对照表（核心论据）：我方 0.30/0.38 s 枚举 + ~0.4 s 全解（kkt 368/o 465
  配置，纯 CPU、确定性、零 GPU 秒）vs fla 在设备 autotune 实测一次：fwd 冷
  16.55 s（25 配置/5 kernel）、bwd 冷 23.26 s（23 配置/4 kernel）、fwd+bwd 合计
  39.81 s（48 配置）——**≈50×**；暖中位自锚定回 G5 基线带（0.491 vs 0.316–0.517 ms；
  1.349 vs 1.34 ms）证明测得可信（`evidence/G6/search_cost.md`、
  `evidence/G6/fla_autotune_cost.json`、`evidence/G6/scratch/solver_timing_summary.json`）。
- 诚实范围注：0.8 s 是**我方 stage 旋钮空间**的排序成本；full-op 发运决策另用 GPU
  配对 A/B（工程归因，非搜索）；可比量 = 同候选空间下离线枚举排序 vs 在设备逐个
  编译计时；fla 的可选磁盘缓存层默认关闭（`FLA_CACHE_MODE` 未设=DISABLED），首调
  成本按进程按形状键重付（`evidence/G6/search_cost.md` §b 方法注记 + 结论表）。
- 记录在案的环境毛刺：o K64V64 子空间在 rlimit 上限下 Z3 部分模型伪影崩溃
  （`tools/gdn_solver.py:187`）——披露不隐藏；两个规范空间逐字节复现。

### 5.4 消融矩阵（旋钮→效应，两级）
- 级一（G2 stage kernel 编译期研究，20 配置，PTX/SASS/资源三层 + 18 个 GPU 计时）：
  mma 开关 50–100×（nomma=未分布标量循环，39 SASS 指令/MAC）；pipeline 0→2 5×
  （105→21 µs；`.ca` 串行化为模型最大盲点）；warps 反直觉（w2 14.9 < w4 21.2 <
  w8 34.2 µs——均匀地址拷贝流按 warp 重复发射）；tile 尺度两难（小 tile 赢单 CTA、
  大 tile 赢全形状=v2↔v3 翻转的占用项）；dtype f32/tf32 ≈1.6× bf16；warpspec+tma
  资源包络（spill 22.8 KB→48 B）（`evidence/G6/knob_ablation.md` §1、
  `evidence/G3/config_times.json`、`evidence/G2/gen_report.md`）。
- 级二（G5 full-op 杠杆 A/B，配对交错）：K-循环 transform 流水 2.2–2.7×；块均匀
  staging 2× 慢（256× 冗余拷贝，负→修）；ping-pong 零增益（根因 wait_group-0）；
  LDSM +8 列补列 = fwd r3 决胜杠杆（473→249 µs；8 路 bank 冲突、63% MIO 停顿）；
  段分裂扫描 16–33% 损失（Amdahl：链+共居仅值 F2 的 17%）——诚实反驳；
  F2 barrier 2→1 +19.7 µs（L1 carveout 代价）；F1UW −123 µs；bwd 侧 mma 化 4.2×、
  tf32 hatch 1.28×、split-bf16 3-mma 3.8×（dk 数值不变 1.730e-3 而 tf32 超 bar）、
  dhs 重构 1.56×、rgemm 访存序 0.83→0.59 ms；BV=64 retile 资源墙（~150 KB > 
  101376 B）（`evidence/G6/knob_ablation.md` §2 全表）。
- 交叉验证段：cost model 对 18 个计时重建的 Spearman 0.8962 = 本表旋钮→成本方向
  与求解器排序同源（零拟合）（`evidence/G3/validation.json`）。
- 级间交互教训（论文独有论点）：旋钮在 stage 尺度正确、在 full-op 尺度有毒
  （block-uniform staging 案例）——两级消融不可缺一（`evidence/G6/failure_longscroll.md` §1）。

### 5.5 反向：五轮开放战线长卷（诚实主景）
- 轨迹：58.88 → 14.16 → 11.06 → 8.94 → **8.15 ms**（B1T2048H16；累计 **7.2×**
  自提速；vs fla 同窗 1.03 ms = **0.126×**）；数值 6/6 双形状全程在容差
  （`evidence/G5/GATE_REPORT_FINAL.md`、`bwd5_bench2048.json`）。
- 逐轮杠杆（每轮一段，含该轮负结果）：
  r1→r2 mma 化 wy + fp32 链 + 线程分区 staging（58.88→14.16，4.2×）——同轮解剖出
  四个独立缺陷，**皆非**起初假设的 bf16 操作数放大（grading-twin 本身有错的认知论
  段落：错误参考制造出你最不怀疑的误差签名；fp64 级 |jax−twin| 3.6e-2 vs
  |jax−GPU| 1.5e-5）（`evidence/G5/bwd_report.md` r2、`evidence/G6/failure_longscroll.md` §11）。
  r2→r3 tf32 hatch 16 位点（14.16→11.06，1.28×）+ tf32-in-dhu 双负（占用悬崖
  2→1 CTA/SM + 误差经携带态复合 dk 1.09e-2 > 8e-3——逐位点干净 ≠ 可发运）
  （`evidence/G5/bwd_report.md` r3/r4）。
  r3→r4 dhu split→dkb + async 16B staging（11.13→8.94）+ async 零投递悬案（OPEN：
  同样的 burst 在隔离探针位级正确、在真实游走中全零，确定性复现，根因未明，门留
  `GDN_PF_K/GDN_PF_QW` 默认关）（`evidence/G5/bwd_report.md` r4 §2、longscroll §9）。
  r4→r5 携带扫描重构出厂（caryA/caryM/bpre 预复合 + 每 chunk 单 rgemm 0.59 ms；
  dhu 退役）+ split-bf16 3-mma dkb（0.96→0.25 ms）+ dv 零成本折进 wyb；携带数学
  对真序贯迹 2–3 ulp，全分段+近单位门 stress 下成立；超收缩发现（§3.6）
  （`evidence/G5/GATE_REPORT_FINAL.md` §bwd r5）。
- 剩余差距的实测排序（账目）：dqkwg 1.71（dvp_bh 需 ≤4-warp mma 规划的 split-bf16）
  > wy 1.45+（串行 solve 基）> wyb 1.04 > dhs 0.59（树扫描负经济学已证，天花板
  ~0.3 ms）；判定：达标在当前会话不可行——fla bwd 是成熟 Triton 工程多轮调优产物，
  我方编译栈已证 7.2× 可压缩性与全部瓶颈归属，剩余 8× 需多轮算法+占用率工程
  （F2 式 ≤49.5 KB 双 CTA 重设计级别）（`evidence/G5/GATE_REPORT_FINAL.md` §五轮账目）。
- 未出厂但已验证的负经济学：tree/look-back 扫描 CPU 验证通过（2–3 ulp）但复合成本
  +3 launch vs ~0.3 ms 串行链天花板——"更多并行"必须对实测串行地板清偿自身开销
  （`evidence/G6/failure_longscroll.md` §8）。
- 叙事定位段（章末）：fwd 胜利 + bwd 轨迹+账目 = "调度即 IR"主张的完整证据链
  （正例 + 可解释的开阔地）；0.126× 是本文的诚实部分而非污点。

### 5.6 数值正确性总账
- 语义闭合（G1）：JAX 规约 44/44；MLIR L1 ↔ JAX ~6e-7（fwd）/ 6 梯度 ≤6.4e-6；
  三角闭合 MLIR⇄JAX⇄fla naive（1.2e-7/2.3e-6 fp32 口径）
  （`evidence/G1/GATE_REPORT.md`、`evidence/G1/jax_spec_report.md`）。
- stage 级（G4）：20/20 配置过 fla 衍生契约（bf16-io 5e-3 / fp32-io 1e-3），最大
  9.77e-4 == 纯 bf16 输出量化的 numpy 仿真预测（measured==predicted）；seed-7 置换
  复跑 20/20（非单置换过拟合）；mma 路径修复后 2080/2080 精确覆盖
  （`evidence/G4/GATE_REPORT.md`、`verify_results{,_seed7}.json`）。
- full-op（G5）：fwd 4/4（§5.1 数字）；bwd 6/6 双形状，最差梯度 ~180× 裕量
  （dq 5.2e-7 … dh0 1.1e-8 @T512）（`evidence/G5/bwd_report.md`、`bwd5_verify*.json`）。
- 误差复合律（方法学发现）：tf32 逐位点干净边界**不复合**（经携带态跨 32 反向块
  传播）；重建混沌位点（dvp_bh：三次数值等价构建 dbeta 1.5e-5→2.9e-3→5.8e-3）须
  单构建 bisect 之外的多构建鉴定——保标量（`evidence/G6/failure_longscroll.md` §6–7）。

---

## 6 Discussion & Limitations（约 1 页）

- 反向差距的定性：0.126× 不是方法论失败而是工程轮次差距（§5.5 账目）；编译栈的
  可压缩性（7.2×/五轮）与瓶颈归属完整性是本方法在该方向的可交付结论。
- 共享 GPU 与窗口口径：全部绝对数为受挤占下界；交错协议为共载设计；干净窗口
  幂等重跑协议在档、正式干净窗终判为相机就绪 TODO（GATE_REPORT_FINAL 明示）。
- 单 GPU / 单架构范围：仅 RTX 5090 Laptop（sm_120a）；无 H100/B200 数据；数据中心
  Blackwell（tcgen05 词汇）未触及——旋钮格的 L3 层需扩展新 mma 族，属未来工作。
- 单算子范围：GDN fwd+bwd；GVA/varlen/h0 等属性在 L0 已定义但未硬件验证；无模型级
  （训练吞吐）端到端。
- LLVM 版本耦合：全部拼写与"在树"结论对 Oct-2026 主干负责；transform API 漂移与
  stock-MLIR 缺口的修复上游化是开源计划的一部分（§4.3 清单即上游 issue 草稿）。
- 覆盖的边界：T1/T2 为静态必要证据；性能主张只由 §5.1 计时承担；成本模型只承担
  排序质量（绝对刻度 3–5× 低估已披露）。
- 基线公平性：fla autotune 计为其设计内成本（首调、按进程按形状键）；若用户预填充
  fla 可选磁盘缓存层，50× 为首次部署口径——已在 §5.3 范围注中写明。

## 7 Conclusion（约 0.5 页）

- 论点回顾：调度即 IR——把 kernel 调度自由度放进原生 MLIR 旋钮、由编译期确定性
  SMT + 零拟合标定模型决策、以 PTX 覆盖与交错终判双轨验证。
- 数字回顾：fwd 3/3 ≥1.1×（1.592/1.05–1.19/1.253–1.312×）；Spearman 0.8962；
  搜索 ≈50×（0.8 s vs 39.8 s）；覆盖 feature 层 8/6/2、hatch 1/≤5；bwd 0.126× +
  7.2× 轨迹如实呈现。
- 开放战线：反向剩余 8×（dqkwg split-bf16 ≤4-warp 规划、双 CTA 重设计）、
  async 零投递悬案、tcgen05 词汇扩展、干净窗口终判补跑。
- 工件承诺：证据树（evidence/G0–G6）+ 复现脚本随文发布。

---

## 附录（计划）

- A 术语与约定表（docs/L0_SEMANTICS.md 8 条裁定的论文版）。
- B 证据索引：论断→文件映射表（本骨架的机器可读版）。
- C 失败长卷全文（`evidence/G6/failure_longscroll.md` 的成文版，12+6 条）。
- D 复现手册（G0–G6 各 gate 的命令与产物路径）。

---

## 写作纪律自检清单（成稿时逐项过）

1. 每个"×"数字旁有窗口口径与分母口径脚注。
2. 每个机制（Z3 公式化、成本模型、hatch、覆盖判据、交错终判）≥1 段展开，无括号化。
3. 负结果在正文出现处均带数字与去向（reverted/kept-behind-flag/open）。
4. 黑话表：WY、chunk、decay 门、交错终判、T1/T2、hatch、nomma、LDSM、carveout、
   超收缩——首次出现前定义。
5. 相关工作逐点名（FlexLA/TileLang/Hexcute/TileLoom/CAKE/PEAK/BaCO）+ 差距陈述。
6. 公式全部以 `$$…$$` 可渲染形式给出（本文稿已示范）。

---

## G5 post-G7 补遗：R7–R25 长卷的评估章映射（R25 交付，2026-10-08）

G7 收官后 G5 战线又跑了 **19 轮优化（R7–R25，fwd/bwd 双线交错，全程共载交错口径、
cubin md5 键控可溯）**。本节把该长卷映射进上面 §4/§5 的骨架，使论文与跑过 G7 的
战役保持同步；正文成稿时按下列条目扩写（数字均已在引用文件中落地）。

### → §5.1（前向终判表更新：正式记录的三代迁移）

- 正式记录表（formal record table）现为三代：r19/r20 单形记录 → **r22 三形 n=30
  --stats 单一代 cubin 代**（B1T8192 1.2226× 27/30 p=8e-06 `be8b914b`；B4T2048
  1.1417× 30/30 p=0.0 `83898f74`；B1T2048 首个 formal 2.1184× 29/30 `3d47886a`；
  `evidence/G5/final_adjudicate_r22f_f1zf.json`）→ **R25 W32P 迁移代**（cubins
  `e7c8eb4c/39f094c3/be5b2377`，`evidence/G5/final_adjudicate_r25f_w32p.json`，
  本轮档案化 n=30；old→new 对照表在 `evidence/G5/fwd_report.md` round 25）。
- 可判性双轨（v1/v2）：v1（全 30 对 IQR）保持记录规则；v2（内 80% 截尾）为
  DOCUMENTED-OPTIONAL 侧统计（`evidence/G5/adjudication_methodology.md`；R25 决策
  记录 `evidence/G5/DECISIONS_R25.md` D1，adopted-per-recommendation）。B4T2048 的
  1.1417× 按 **"v1-no / v2-yes" 双引**；B1T2048 保持 v2-soft（窗口漂移警告）。
- 19 轮前向杠杆长卷（每轮含负结果归档）：F1KS 2-CTA + ZFIX/SCANDB 双竞态根因
  （r17/r18）、WST 宽存（r15）、F2US 预取管分期（r14）、F1ZF 零折（r21）、
  GDN_W32P 无冲突按行 warp 重排（r24：f2 −9.7..−25.6 µs B4 / −17.9..−22.6 µs T8192
  配对，值恒等）——全部在 `evidence/G5/fwd_report.md` r6–r25。
- 诚实负结果群（进 §5.1/§5.4 交叉引用）：F1EP/FKS2/F3KS/F1P4/mb3-depth3/BV=64
  retile/BK32 等，全部量化死因在案；f1 已端到端关闭（r23：staging 826 GB/s 实测
  屋顶 = 算法最小流量）、f2 写通道关闭（r24）、f3 带宽界（r20）。

### → §5.5（反向长卷延伸 + 双重穷尽证明，本文的 bwd 完备性论证）

- 轨迹延长：8.15 ms（r5）→ 6.73（r7）→ … → **~2.15 ms 共载最小 @B1T2048H16**
  （0.479× vs fla 同窗；自 r1 的 58.88 ms 累计 ~27× 自提速）；数值 6/6 双形状全程
  记录位数（`evidence/G5/bwd_report.md` r6–r22）。
- **双重穷尽证明（dual-exhaustion proof；R24/R25 决策冻结——`DECISIONS_R25.md`
  D2，adopted-per-recommendation）**：(i) 值恒等类 r22 穷尽（dqkwg 分配器审计 38
  编译探针：forced-.maxnreg 任意 cap 皆非零 spill；K32 链分裂第 4 数据点；
  VTF/DHP/ks2/F1EP 家族负结果）；(ii) 新数值类 r24 穷尽（W-park 旗舰**决策级
  NO-GO**：数值可容——bf16 部分泊车 65–1025× 裕量、f32 重结合 ~1e-9——而 pool
  2,560 B 头寸 vs ≥16,384 B 最小泊车、全局泊车净 +49..147 KB/chunk = 0.83–0.98×
  REGR、FREE 天花板 1.02× 低于一切发运门槛）→ **r19-final bytes 冻结**
  （`evidence/G5/bwd_wpark_decision_memo.md` + `wpark_twin_T{2048,8192}.json`、
  `wpark_arith_out.txt`）。
- 剩余 ~8× 的定性由"多轮工程差距"升级为："两类想法空间已被量化穷尽；唯一未测
  余地是 regime 变更（B4+/T-other/H-other 形状外推）"——进 §6 Discussion 的 bwd
  段落。

### → §4.3（工具链发现账本追加，C4 的延续）

- r20：rt.build 陈旧缓存信任洞（gen 目录 cache_key 键控闭合——r3 曾由此洞喂入
  陈旧 cubin）；r21：G0 "sm_120 仅 1×1×1 簇" 论断被设备实测推翻（cluster 2/4/8
  可用、attr 开销 ~0）；r22：forced-.maxnreg 分配器定律（cap 174 下地板 40/60 B
  spill——自然 ≤128 寄存器 ≠ 零 spill）；r23：--mb3/--depth3 在 2-trip uw 循环上
  流水前置条件硬失败；r19/r24：模块布局彩票特征化（f3 ±10.5±6 µs 类，PTX 字节
  恒等而布局重排；R25 三形重建 md5 全数复现 = 工具链输出确定性的正面数据点）。

### 协议注

- 全部 R7–R25 计时为共载交错口径（平衡两负载 harness）；干净窗口终判 hook 仍在档
  （`scripts/adjudicate_when_clean.sh`，co-job 退出后相机启动）；每代正式记录带
  cubin md5 与窗口快照（JSON 内 `gpu_start/gpu_end`、`window_pre/post` 字段）。
