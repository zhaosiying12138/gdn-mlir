## 二十五、反向机器设计：重算、双向扫描与仿射携带

反向状态梯度的旧发射器把每个分块（chunk）的约八段 GEMM 级工作放在同一条逆时间链上：装配 P、Bhat、Qg、W，计算 dvp 的两项，更新 dH，再顺手累积 dk、dg 所需的部分和。后一块必须等前一块完成，增加线程块（CTA）也无法切开这段依赖。真正昂贵的是链上被迫串行的工作种类，GEMM 数量只是其可见表象。

内核代号先固定：终态链为 `wy → fwdscan → caryM → bpre → dhs → dkb → dqkwg → wyb`，依次负责 WY 重算、前向扫描、M/B 预计算、逆向携带、部分和重算与梯度组装；`dhu` 是早期串行链的名称。第 4 节给出每个内核的输入输出职责，读代码时可以先按这张图定位。

### 1. 从顺序更新提取携带算子

令 Xₙ 为到达 chunk n、尚未执行本块更新的 dH。旧顺序形式可写成：

```text
dvpₙ   = Pₙᵀ dOₙ + Bhatₙ Xₙ
Xₙ₋₁  = 2^gclₙ Xₙ + Qgₙᵀ dOₙ − Wₙᵀ dvpₙ
```

把第一式代入第二式，利用矩阵乘法的结合律与分配律，分别收集 Xₙ 和 dOₙ：

```text
Xₙ₋₁ = AₙXₙ + Bₙ
Aₙ = 2^gclₙ I − WₙᵀBhatₙ
Mₙ = Qgₙᵀ − WₙᵀPₙᵀ
Bₙ = MₙdOₙ
```

这里的 gcl 是 chunk 末行的 log2 域累积衰减。外部语义仍是 decay = e^g、g≤0；内部先将 g 乘 log2(e)，再用 exp2，因此算子中才出现 2^gcl。重结合保留了每个数学乘积，但浮点求值次序已经改变，代数等价仍需独立数值对照。

CPU 验证器特意保留一条经过 dvp 的真顺序路径，避免用同一个复合公式验证自身：

```python
for n in reversed(range(NC)):
    Xs[n] = X
    if W_list is None:
        X = A_list[n] @ X + M_list[n] @ dOs[n]
    else:
        gcl = gc[n][-1]
        Bhat = k[n] * np.exp2(gcl - gc[n])[:, None]
        Qg = q[n] * np.exp2(gc[n])[:, None]
        dvp = P_list[n].T @ dOs[n] + Bhat @ X
        X = (np.exp2(gcl) * X + Qg.T @ dOs[n] - W_list[n].T @ dvp)

# carry_walk 中的独立复合路径
B_list = [M @ dOs[n] for n, M in enumerate(M_list)]
X = dht.copy()
for n in reversed(range(NC)):
    Xs[n] = X
    X = A_list[n] @ X + B_list[n]
```

[tools/gdn_carry_validate.py:121-145，节选并省略公共初始化与返回语句]

读点时先追踪 `Xs[n] = X`：它固定了更新前（pre-update）数据合同。`W_list` 分支经过 dvp，`carry_walk` 只消费 A、M、B，两者拥有独立的乘加分组。这样才能把 4.9e-17（T512）和 8.3e-17（T2048）的 fp64 最大偏差解释为重结合的舍入差，在约 130 的 X 范数上相当于 2–3 ulp。fp32 回放预测底噪 1.2e-8，GPU 链内实测 2.2e-8，说明目标机结果也落在预判量级。[tools/gdn_carry_validate.py:111-145；evidence/G5/bwd_report.md:699-715,856-857]

### 2. 超收缩与 v 切片复用回答不同问题

基准门分布下，逐块算子的谱半径 ρ(Aₙ)≤2.3e-18，范数也不超过 2.2e-17。谱半径描述渐近放大能力，范数补充一步增长的界；两者合起来说明跨块耦合在这条数据分布上极弱，复合误差不会沿长链积累。Aₙ 只依赖 W、k、g，不含 v 或 dO，因此一个 K×K 算子只需复合一次，便可服务所有 v 列切片，预复合成本能随切片数摊薄。

这两条性质不能互换使用。仿射重结合对任意门分布在代数上成立，超收缩只属于测得的门分布；弱衰减时 A 可接近单位算子，近酉（unitary）压力例的范数曾达到约 8.8。当前默认分支利用强衰减进一步退役 caryA 和 A_all，dhs 随之退化为 B_all/dht 的移位携带；`GDN_ARET=0` 恢复完整 A 递推。把退役条件写成生成器开关，正是为了让分布假设保持可见。[tools/gdn_bwd_gen.py:382-402]

验证器还检查了 2、4、8、16 段的二级扫描：段内以零状态局部行走，段边界串行复合，再用后缀算子修正，fp64 偏差同为 2–3 ulp。[tools/gdn_carry_validate.py:148-190] 数学可行并未自动带来设备收益。段算子需要额外的 K×K 复合，边界修正增加内核与全局中间量，而强衰减已经削弱了长链耦合的收益空间。保留一级精确链、压缩每一步暴露延迟，因而不是保守选择；它由工作量和实测共同决定。

### 3. 预复合移出串行链，部分和改为独立重算

caryM 以 B·H·NC 个独立 CTA 计算 Mₙ。这里先给后文反复出现的 split-bf16 一个最小定义：把 x 拆成 `hi=bf16(x)` 与 `lo=bf16(x−hi)`，用 `hi·hi + hi·lo + lo·hi` 三条 bf16 mma 恢复更多尾数；hi/lo 的数学与误差账在第二十七章展开。当前实现先放 Qgᵀ，再用 split-bf16 三链减去 WᵀPᵀ，最后一个 k16 切片的收尾（epilogue）完成减法。以下片段位于 `for i0 in range(0, K, BK2)` 内，此时 mb 已含 Qgᵀ：

```python
# mb[a, c] = q[c, i0+a] 2^gc_c - sum_r W[r, i0+a] P[c, r]
def _cm_epi(mi, ni, ei, row, col, acc):
    x = mma_extract(e, acc, ei // 2, ei % 2, "cmx")
    cur = load2(e, mb, row, col)
    e.l(f"memref.store {op2(e, 'subf', cur, x)}, "
        f"{mb.name}[{row}, {col}] : {mb.ty}")

accs = None
for kb0 in range(0, C, 16):
    # 省略 W 与 P 的 hi/lo staging
    accs = mma_gemm(e, f"cmh{i0}_{kb0}", BK2, C, 16, 8,
                    whs.name, whs.ty, phs.name, phs.ty, accs=accs)
    accs = mma_gemm(e, f"cml{i0}_{kb0}", BK2, C, 16, 8,
                    whs.name, whs.ty, pls.name, pls.ty, accs=accs)
    accs = mma_gemm(e, f"cmm{i0}_{kb0}", BK2, C, 16, 8,
                    wls.name, wls.ty, phs.name, phs.ty, accs=accs,
                    epilogue=_cm_epi if kb0 == C - 16 else None)
```

[tools/gdn_bwd_gen.py:3390-3427，保留累加与 epilogue，省略初始化和切片暂存]

读这段代码要区分 `mb` 的初值、三链的部分积与最后一次减法。`accs` 跨 k16 切片保留同一数学累加，epilogue 只在末尾执行，避免每片重复减 Qgᵀ。由此可以判断 caryM 对 chunk 间状态没有读写依赖，适合按 M=B·H·NC 并行，也能确认 M 的生成与 v 切片无关。

bpre 再把 Mₙ 乘到 dOₙ 上。原始 dO 已是 bf16，只需拆 M 侧；两条 mma 对应 M_hi·dO 与 M_lo·dO，K 的平方根倒数尺度只在最后的 epilogue 执行一次：

```python
def epi_b(mi, ni, ei, row, col, acc):
    x = mma_extract(e, acc, ei // 2, ei % 2, "bex")
    xs = op2(e, "mulf", x, scale)
    # 省略全局列下标计算
    gstore(e, xs, G["B_all"], bty, [m, gcc], src_dt="f32")

for i0 in range(0, K, BK2):
    stage_split_ew(e, ahi, alo, G["M_all"], mty, m, C, i0 * C, BK2, C)
    accs = mma_gemm(e, f"bmh{i0}", BK2, V, C, 8, ahi.name, ahi.ty,
                    dotT.name, dotT.ty)
    accs = mma_gemm(e, f"bml{i0}", BK2, V, C, 8, alo.name, alo.ty,
                    dotT.name, dotT.ty, accs=accs, epilogue=epi_b)
```

[tools/gdn_bwd_gen.py:3510-3528，省略 epilogue 的地址展开]

这里读的是职责边界：bpre 消费 M_all、产出 B_all，不知道 Xₙ，也不保存跨 chunk 累加器。尺度只在 `epi_b` 出现一次，说明它位于缩并之后。于是 caryM 与 bpre 都能按块并行，dhs 的完整形式每步只剩 B 切片加 A@X 的一次 GEMM。

dkb 能独立，是因为 dh_all 保存的恰为 pre-update Xₙ。旧链在更新前消费该值形成 dkB、rowB、s5；新链先存轨迹，dkb 再按 chunk 重算这些纯函数，并在本地折叠 v 切片归约。dhu 的串行部分和因此消失，dqkwg 每格也只读一个已归约值。[tools/gdn_bwd_gen.py:3995-4016]

这里交换的是计算位置与存储流量。dh_all 必须先写后读，caryM、bpre 也要生成真实全局张量；收益来自这些访问脱离逆时间依赖，并可按块、按列并行。dkb 的一个 CTA 负责一个 chunk，v 方向归约仍在固定顺序中完成，因而并行化没有求助于无序原子加。审查这类改写时，应同时核对输入是否确为 pre-update、归约顺序是否写入合同，以及新增全局读写是否仍小于临界路径上的延迟收益。

### 4. 八内核终态与退役边界

当前发布链的数据流和职责如下，箭头表示全局中间量的生产与消费顺序：

| 顺序 | 内核 | 主要职责 |
|---|---|---|
| 1 | wy | 重算 WY、P 等块内因子，供双向扫描与六梯度使用 |
| 2 | fwdscan | 顺向重算 h_all 与 v_new，提供块入口前向状态 |
| 3 | caryM | 逐块生成 M_all |
| 4 | bpre | 逐块生成 B_all |
| 5 | dhs | 逆向保存 pre-update dh_all，并产生 dh0 |
| 6 | dkb | 独立重算 dkB、rowB、s5，折叠 v 切片 |
| 7 | dqkwg | 组装 dq、dk_local、dW_all、dvp_all 等中间梯度 |
| 8 | wyb | 完成 dA 双侧掩蔽收缩，写出 dk、dβ、dg、dv |

caryA 的退役直接体现在发射清单，而非表格外的一条备注：

```python
kernels = [s.strip() for s in args.kernels.split(",") if s.strip()]
if ARET and "caryA" in kernels:
    kernels.remove("caryA")
```

[tools/gdn_bwd_gen.py:9081-9084；ARET 默认值见 402 行，省略日志打印]

读这段逻辑可确认，默认配置即使收到含 caryA 的候选列表，也会在生成前将其删除；没有消费者的 A_all 不再落盘。历史报告中的九内核链包含 caryA，可用于核对仿射改写和后续分阶段测量的来源，不能用来描述当前发射数。dhs 的 A 窗口、dkb 的 split-bf16 等优化也应按所测构建理解，而非无条件叠加到退役分支。

历史含 caryA 的九 kernel 构建在 T2048 的归档汇总中，扫描复杂体由 3.13 ms 降到 2.01 ms（1.56×），dkb 单件由 0.96 ms 降至 0.25 ms。数字背后的核心收益是把操作数装配和部分和移出逆时间临界路径；并行预复合仍要付真实 GEMM 与全局流量成本，不能按删除指令数估算收益，也不能用这个历史合计直接替代当前八 kernel 的扫描族时间。

2.01 ms 只覆盖扫描复杂体的阶段账，3.13/2.01 不能直接当成端到端加速比。扫描复杂体只是端到端的一部分，其余时间仍在 wy、fwdscan、dqkwg、wyb。dhs 保留 NC 步逆向依赖，说明优化后的机器仍有明确串行地板；继续增加预复合并行度不会缩短这段地板。性能判断因此需要逐阶段求和与端到端交错计时相互印证，还要记录 caryA 是否在场，避免把不同清单的阶段项机械相加。

### 5. 算法知识落在生成器中

这次改写由 Python 发射函数、A_all/M_all/B_all/dh_all 的全局合同和 CPU fp64 验证器共同实现。仓库没有能识别任意 MLIR 递推、证明其仿射结构并自动重排的通用 pass，也没有把 pre-update 协议推导成通用分析。生成器承载的是这个算子的算法级规格；MLIR 负责表达发射后的循环、存储与 GPU 操作。可迁移的是推导和验证方法，套用到其他算子仍需重新建立携带方程、消费时点和数值门。
