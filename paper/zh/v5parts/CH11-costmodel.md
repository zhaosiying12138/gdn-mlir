## 十一、解析成本模型：计数除以速率

求解器把十维旋钮的合法取值空间整个枚举出来（kkt 阶段 368 个、o 阶段 465 个配置），枚举本身不区分好坏；给每个配置一个可信的时延预测、从而排出次序，是成本模型的职责。这个函数有三个硬约束。其一，纯 CPU 可算：每个新形状的调度预算是亚秒级。其二，确定性：同一输入必须给出逐字节相同的排序，否则"两次完整编译逐字节一致"的承诺失效。其三，零逐配置拟合：模型里只要有一个对着实测配置时间调参的自由常数，排序就退化为对训练集的记忆，对新形状失去效力。解析成本模型同时满足三者：每一项都写成"分子为编译期可精确知道的计数、分母为第十章标定速率"的除法。术语约定：解析模型指由封闭公式与常数表计算、不含拟合参数的模型；屋顶线（roofline）指以各资源项的最大者为时延估算的粗粒度形式。

### 1. 零逐配置拟合契约与自证机制

契约原文写在模型文件头：

```
every rate is either (a) measured once by the G3-A microbenchmark window
(evidence/G3/calibration.json) or (b) a documented sm_120-class structural
constant. There are NO per-config fitted free constants.
```
［tools/gdn_costmodel.py:9-11，docstring 逐字］

契约的执行不靠口头声明：验证程序把"哪些键没有从标定文件取得、退回了占位表"逐键落盘为 calibration_fallback_keys，任何人可查。本次验证该清单里唯一在列的是 malloc_ns（设备分配常数，占位 2000 ns）；而 18 个被计时配置全部经过静态共享内存修复、分配次数归零，T_alloc 一项在全部 18 行预测里恒为 0.0［evidence/G3/validation.json:37-39 及各配置 terms_ms；归零逻辑 tools/gdn_costmodel.py:607-611］——没有任何一个回退速率参与过排序数值。发射常数同样有出处：它由参考核实测推导（第十章第二节），推导完成后从回退清单移除［tools/gdn_solver.py:433-446］。

结构常数一共五个，全部带文档与出处：标量环每 MAC 10 周期（SASS 推导，见第三节）、LSU 每时钟每 SM 4 条 warp 指令、tf32 对 bf16 张量速率折算比 0.5、溢出流量按 L2 每 SM 份额计、张量每 SM 每时钟 1024 FLOP［tools/gdn_costmodel.py:120-136］。它们全部经过正负两倍的扰动敏感度检验（第十二章第三节）。

### 2. 三形式递进：v1 屋顶线、v2 发射项、v3 波次

模型有三个累积的形式，定义文档与代码同文件，以下为 docstring 真文：

```
v1  (contract form, roofline)
    T = T_launch + overlap(T_compute, T_gmem, T_smem) + T_bar + T_spill + T_ex2
    overlap = max(...) when the config is software-pipelined
    (pipeline>0 or warpspec producer/consumer: cp.async double buffering
    hides the copy behind compute), sum(...) when pipeline=0 (sync copies
    serialized behind gpu.barrier).
v2  v1 + T_lsu + T_alloc
    T_lsu   = issue-bound term for the LSU instruction stream. The
              generated code issues UNIFORM-ADDRESS (non-distributed)
              global->shared copies: every thread executes every
              16B cp.async of the tile (verified in kernel.ptx: copy
              indices are block-uniform, no %tid in the address chains).
              Wire bytes are deduped by L1/L2 sectors, but the ISSUE cost
              is instructions x threads. Fragment ld.shared loads are
              CSE'd unique-per-iteration loads (verified: LDS static count
              = unique fragment loads, not 12x atoms).
    T_alloc = the memref.alloc lowering calls device malloc() for each
              tile buffer and cvta.to.shared's the result (verified in
              kernel.ptx: 2 mallocs per GEMM in the mma path, 3 allocs in
              the nomma path, +1 output staging buffer for tma=on).
              Cost = n_allocs * malloc_ns (single global constant).
v3  v2 + occupancy/waves + overlap blend
    waves = ceil(total_ctas / (82 * ctas_per_sm)); ctas_per_sm from
            regs/smem/warp-slot limits; per-wave time = v2 per-CTA time.
    overlap becomes alpha*max + (1-alpha)*sum with alpha = 1.0 for
            pipeline>0 / warpspec, 0.0 for pipeline=0 (same as v1/v2 split,
            kept explicit for perturbation experiments).
```
［tools/gdn_costmodel.py:18-44，逐字］

v1 是契约形式：发射常数，加计算、全局访存、共享访存三项的重叠，加栅栏、溢出、ex2 三项的串联。重叠的取法编码一个调度事实——流水配置（pipeline 大于 0 或 warp 专化）的双缓冲把复制藏进计算，取最大值；p0 配置的同步复制被屏障串行化，取和。v2 补上发射视角：导线上的字节会被 L1/L2 扇区去重，但发射成本按指令乘线程计，均匀地址复制流的这一项不可忽略；T_alloc 对修复后的被测 cubin 自动为零。v3 把波次（waves，CTA 总数除以全机可驻留数的向上取整）乘上 v2 的每 CTA 时间，驻留数由寄存器、共享内存、warp 槽三重上限的最小者给出［occupancy 函数，647-657 行］。三形式的递进有完整的效果档案（ρ 为对 18 个实测配置的 Spearman 秩相关，其含义与门槛在第十二章定义，此处只看相对变化）：

表：模型形式递进档案

| 步骤 | 变更 | 整体 ρ | kkt ρ / o ρ |
|---|---|---|---|
| 1 | v1 屋顶线；标量环按 CUDA 核屋顶线 | 0.7352 | 0.644 / 0.486 |
| 2 | 加 LSU 发射项与分配项（即上文英文展品里的两个 T 项） | 0.7475 | 0.669 / 0.486 |
| 3 | 标量环项改为每 MAC 10 周期 | 0.8962 | 0.936 / 0.829 |
| 4 | 发射锚定标（6.12 µs） | 0.8962 | 秩不变，定绝对尺度 |
| 5 | v3 波次项 | 0.8962 | 0.936 / 0.829 |

［evidence/G3/solver_report.md §2 的迭代记录，数值逐行转写］

### 3. 张量核主项与标量环项

计算项分 mma 与标量环两个分支，各占一段代码（本展品沿用主文）：

```python
if feats["mma"]:
    peak = calib.m["mma_bf16_peak_flops"] if esz == 2 else calib.mma_tf32_peak
    terms["T_mma"] = feats["mma_flops_per_cta"] / (peak / calib.sm_count)
else:  # 标量环：每 MAC 按发射/延迟受限建模，速率=SASS 推导的 10 cyc/MAC
    macs = feats["ffma_flops_per_cta"] / 2.0
    terms["T_ffma"] = macs * NOMMA_CYC_PER_MAC / calib.m["clock_hz"]
```
［tools/gdn_costmodel.py:689-700；分子为封闭公式 2·bt·bv·K·n_gemm，已对静态 PTX 计数自检］

分子的封闭公式住在 analytic_features（462 行起）：每 CTA 的 mma FLOP 数、访存字节、栅栏次数、ex2 个数全部由旋钮取值与形状经算术得出；分母把全机峰值折成每 SM 份额（除以 82），tf32 按折算比减半。标量环分支的 10 周期常数有一段来历：mma 关闭时，convert-linalg-to-loops 下降出一条未分布到线程的标量三重循环，SASS 里每个 MAC 的最内层体是 39 条指令——10 组 64 位地址加法、循环控制、3 条标量 LDS、1 条 FMA——按每 SM 4 条发射管道折合约 10 周期每 MAC；常数从反汇编读出、写死并注释出处，从未对实测时间调参［tools/gdn_costmodel.py:128-136；evidence/G3/gen_noargs/kkt_bf16_nomma_p2_t64x64x32_w4/sass.txt 循环 .L_x_17］。最初的 CUDA 核屋顶线形式把这条路径低估 100–400 倍（实测 0.5–2.5 ms），因为它把延迟受限的标量环当成吞吐受限的向量乘；递进档案第 3 步的 ρ 跃升全部来自这一处替换。

### 4. 溢出项：计数来源的三层结构

溢出项展示计数来源的分层：能算就算（封闭公式），算不了就读报告（ptxas 资源报告），报告缺失再数指令（SASS 静态计数）。第二十九章讲过报告与反汇编如何落盘为每配置的 meta.json，这里给出消费端（本展品沿用主文并扩出回退行）：

```python
spills = ((mm.get("spill_stores") or 0)
          + (mm.get("spill_loads") or 0))
if spills == 0:
    spills = (so.get("LDL") or 0) + (so.get("STL") or 0)    # SASS 计数回退
if spills:
    # undo peeled-body duplication: static ~= body*(1+pipe),
    # dynamic ~= body*(trips+pipe)
    pipe = max(feats["pipe"], 0)
    dyn = spills * (feats["trips"] + pipe) / (1 + pipe)
    feats["spill_dyn_ops_per_cta"] = dyn * feats["threads"]
...
spill_bytes = feats["spill_dyn_ops_per_cta"] * 4.0
terms["T_spill"] = spill_bytes / (calib.per_sm("l2_bw") * SPILL_BW_FRAC)
```
［tools/gdn_costmodel.py:616-625、711-712，节选；SASS 回退即 618-619 行］

其一，静态计数到动态次数的换算：软件流水把循环体复制出 peel 副本，静态计数约等于体乘（1 加 pipe），动态次数约等于体乘（trips 加 pipe），比值把复制抵消。其二，取数优先级：加载器优先采用 G3 重建（被计时 cubin）的 meta，G2 编译索引只作回退——两次编译的寄存器分配可以合法地不同（一边 65–96 寄存器零溢出，另一边 255 寄存器万级溢出），给错版本的计数等于给错分子［tools/gdn_solver.py:449-452；tools/gdn_costmodel.py:597-606；第二十九章双遍陷阱］。

### 5. 分子的精确性自检

"分子在编译期精确"是一个被验证的命题。验证办法：把封闭公式对 18 个配置的静态 PTX 计数逐一对照——mma.sync 计数 18/18 精确，ld.shared 18/18 精确，bar.sync 18/18 精确，cp.async 偏差正 0.4% 至正 1.6%（prologue 与参数收尾的松弛）［evidence/G3/solver_report.md §2；自检入口 tools/gdn_solver.py:525-528 起］。ld.shared 的精确命中依赖一个生成器事实：公共子表达式消除把片段装载去重成每迭代唯一装载，静态计数恰等于公式值；流水的 peel 复制因子也经同一路径验证。至此模型的误差只剩物理来源——重叠的幅度、依赖链的长度——它们不触碰计数与速率的契约，其影响留在秩层面度量（第十二章）。
