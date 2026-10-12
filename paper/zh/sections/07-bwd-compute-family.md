# 7 反向 kernel 体系之计算族：wy / dqkwg / wyb 与生存期竞技场编排

> 证据口径（本章全适用）：配对计时不取跨窗口绝对值——同一 GPU 窗口内基线臂/变体臂交错成对（base,new,base,new,…，CUDA-event 30 迭代中位数），共享 GPU 负载全程在位；绝对毫秒数受共居任务（pid 常驻、约 18–22 GB 显存）牵引，仅成对比值可判定（`evidence/G5/bwd_report.md` 各轮"Headline"节）。SASS/PTX 计数为静态指令统计（`evidence/G0` 工具链同源）。本章源码引用一律出自 `tools/gdn_bwd_gen.py`（行号为写作时快照；该文件是本计算族的唯一生成器，所有 kernel 以零实参 cubin + 公开 `memref.global`（MUTABLE）形态由 harness 经 `cuModuleGetGlobal` 装载）。rN 指代 `bwd_report.md` 中第 N 轮小节（r6 = G5-L round 6，r8 = G5-O round 8，r9 = G5-P round 9，r11 = G5-R round 11，r12 = G5-S round 12，r13 = G5-T round 13，r14 = round 14 (R14-V)，r15 = round 15 (R15-W)，r16 = round 16 (R16-Y)，r17 = round 17 (R17-bwd)，r18 = round 18 (R18-bwd)，r19 = round 19 (R19-bwd)，r22 = round 22 (R22-bwd)）。

## 7.0 章首：计算族在反传公式中的分工

### 7.0.1 九 kernel 链中的三 kernel

GDN 反向链共 9 个 kernel（grid 均为 $M = B\cdot H\cdot N_C$ 个 CTA，块长 $C=64$）：`wy → fwdscan → caryA → caryM → bpre → dhs → dkb → dqkwg → wyb`。其中 fwdscan/caryA/caryM/bpre/dhs/dkb 构成"扫描/携带族"（状态递推、跨块携带与余项折叠，见前章）；本章的**计算族**是三个以 $C\times C$ / $C\times K$ / $C\times V$ 收缩为核心的 kernel：

1. **wy（WY 表示重算）**——反向的锚点重算级：由原始输入 $(q,k,v,\beta,g)$ 重建块内 WY 表示的全部共享中间量（$gc$、$kkt$、$A$、$T$、$P$、$u$、$w$），写入全局供全链消费（`kkt_all/P_all` 供 dqkwg/wyb；`u_all/w_all/T_all` 供 fwdscan/dhs 等扫描/携带族与本族）。它不直接产出任何梯度，但反向六梯度的每一个都穿过它写下的张量。
2. **dqkwg（最大 kernel）**——产出 $dq$、$dk_{\mathrm{local}}$、$dW$（即 $-\,\mathrm{d}v_p S_n^\top$）与携带族回传的行/列部分和 $r_3/r_4/c_{4r}$、$dg_{\mathrm{state}}$，以及后续 GEMM 化所需的 $\mathrm{d}v_p$（`dvp_all`）。
3. **wyb（dA 收缩与终值）**——执行 $\mathrm{d}A$ 的双侧严格下三角掩蔽收缩（两个 $C\times C\times C$ 收缩的三明治），产出最终 $dk$、$d\beta$、$dg$、$dv$。

### 7.0.2 记号（先定义后用）

设块长 $C=64$，特征维 $K$、值维 $V$（实验形状 $K=V=128$），块内索引 $i,j\in[0,C)$。$x_{\mathrm{col}}$ 表示按列广播的行向量（$[x]_{ij}=x_j$），$x_{\mathrm{row}}$ 同理按行。kernel 内 decay 一律在 $\log_2$ 域（`arith.exp2` 换域）。

- $gc_i=\sum_{t\le i} g_t$（块内累积 log-decay），$gcl = gc_{C-1}$；
- $kkt = k\,k^\top$；$A = \mathrm{stricttril}\big(\beta_{\mathrm{row}}\odot kkt\odot 2^{gc_i-gc_j}\big)$；$T=(I+A)^{-1}$；
- $P=\mathrm{tril\_inc}\big((q\,k^\top)\odot 2^{gc_i-gc_j}\big)$（含对角）；
- $u = (T\odot\beta_{\mathrm{col}})\,v\in\mathbb{R}^{C\times V}$，$w = (T\odot(\beta\,2^{gc})_{\mathrm{col}})\,k\in\mathbb{R}^{C\times K}$（命名沿用本仓库 `bwd_report.md` 阶段表；fla 源码中同名张量角色互换，集合相同）；
- $\mathrm{d}Pr=\tfrac{1}{\sqrt K}\sum_{vs}\mathrm{d}O_{vs}\,v_{\text{new},vs}^\top$；$\mathrm{d}Pm=\mathrm{stricttril}(\mathrm{d}Pr\odot 2^{gc_i-gc_j})$；
- $\mathrm{d}v_p = P^\top \mathrm{d}O/\sqrt K + \mathrm{diag}(2^{gcl-gc_i})\,(k\,\mathrm{d}h^\top)$；$\mathrm{d}W=-\,\mathrm{d}v_p\,S_n^\top$（$S_n$ 即 `h_all` 的块末状态）；
- $G=\mathrm{stricttril}\big(\mathrm{d}v_p\,(\beta v)^\top+\mathrm{d}W\,(\beta k\,2^{gc})^\top\big)$；
- $\mathrm{d}A_f=-\,\mathrm{stricttril}\big(T^\top\,G\,T^\top\odot 2^{gc_i-gc_j}\big)$（§2 的双侧掩蔽精确式）；
- **split-bf16**：$x\in\mathbb{R}$ 的 f32 值拆为 $hi=\mathrm{bf16}(x)$、$lo=\mathrm{bf16}(x-hi)$，则 $hi+lo$ 携带约 16 位尾数；2-链乘积 $hi\!\cdot\!B+lo\!\cdot\!B$、3-链 $hi\!\cdot\!hi+hi\!\cdot\!lo+lo\!\cdot\!hi$（弃 $lo\!\cdot\!lo\approx 2^{-32}$）以 mma 速度恢复 fp32 级乘积，残差类 $\sim 2^{-17}$（`_split_bf16` 文档串，r5/r12）。tf32 的两侧操作数截断为 $2^{-11}$ 级——凡 tf32 位点越_bar_者均按此理由换 split（r3 bisect：tf32 曾把 dkb.dkB 推至 $1.04\times 10^{-2}$，超 $8\times 10^{-3}$ 容差）；
- **竞技场（arena）**：同一基地址上按"生存期互斥"叠放多个共享内存 tile 的 PTX 级别名机制（§7.4）；
- **R-A-S**：read-add-store，读全局-加-写回；
- **warp 蝶形和**：`warp_rowsum/warp_colsum`——行/列归约从"单线程串行 $C$ 链"改为 warp 内 5 步 shuffle 树（每行 2 深），lane 0 落盘（r8）。

三个 kernel 的驻留形态（出厂默认臂）：wy 为 128 线程/4 warp × 2 CTA/SM；dqkwg 为 256 线程 × 2 CTA/SM（大网格臂，$BH\cdot N_C\ge 164$ 时启用 v4，T512 小网格保 v3）；wyb 为 256 线程 × 2 CTA/SM。

### 7.0.3 为什么要手写 GEMM/mma 发射

本工具链（stock MLIR → NVVM → ptxas，sm_120a）上：`linalg.matmul` 不可用——`-convert-linalg-to-loops` 使所有线程冗余执行循环巢、链式累加在同一 `outs` 上跨 warp 竞争、同缓冲双侧操作数整体误降级（G5-C 平台约束 3）；`vector.transfer` 不能在 workgroup memref 上降级（约束 2）；`cp.async` 直供 `ldmatrix` 会按构建调度决定性地读错 16 行块中的子集（r6 静默损坏，见 §7.4.3）。因此计算族的每一条 GEMM 都是生成器手写的线程划分 `scf.for`（每单元胞恰一线程写，确定性）或 `nvgpu.ldmatrix + nvgpu.mma.sync`（m16n8k16, bf16×bf16→f32）发射，操作数 tile 一律 `st.shared` 写入。这一约束是本章所有机制的存在理由。

---

## 7.1 wy：块内 WY 表示重算 kernel

### 7.1.1 总体数据流

`gen_wy`（`tools/gdn_bwd_gen.py` L2163–2599）每 CTA 处理一个 chunk，级内顺序：

1. $gc$ 前缀和（§7.1.5）+ $\beta$ 暂存 + $bgc=\beta\odot 2^{gc}$ 提升；
2. $kkt/A$（§7.1.2 的 k-切片 mma 链 + epilogue 双写：原始 $kkt$ 存 `kkt_all` 全局供 wyb 的 $\mathrm{d}A_f\odot kkt$ 项；掩蔽衰减后的 $A$ 写 smem `kktf` 作求解输入）；
3. $T=(I+A)^{-1}$ 前代换求解（§7.1.3）+ `T_all` 全局存储；
4. $P$（同款 k-切片链，epilogue 直接全局存储 `P_all`）；
5. $u/w$ 尾（§7.1.4）。

【源码】阶段表与全局声明：

```python
k.global_("T_all",  (M, C * C), "f32")
k.global_("u_all",  (M, C * V), "f32")
k.global_("w_all",  (M, C * K), "f32")
k.global_("kkt_all",(M, C * C), "f32")   # G5-E round2: kkt 与 P 只算一次，全链共享
k.global_("P_all",  (M, C * C), "f32")
```

### 7.1.2 kkt 与 P：k-切片 mma 链

**【动机】** r12 之前 $kkt=k\,k^\top$ 与 $P=(q\,k^\top)\odot 2^{\Delta}$ 以整 $K=128$ 收缩的 mma 计算，需要 $k_t/q_t$ 两个 $[C,K]$ bf16 共居 tile（32 KB）及其 cp.async 暂存，使 wy 池达 90 624 B（1 CTA/SM，占用率 12.5%）。把整 $K$ 链拆为 8 条 $K{=}16$ 链后，操作数退化为每次一条 16 列的 $[C,24]$ 切片 tile（3 072 B），$k_t/q_t$ 与全部 `opc/accc/kslc` 暂存退役。

**【源码片段】**（`gen_wy` L2308–2326）：

```python
accs = None
for kb0 in range(0, K, 16):
    stage_rows(e, ksl, G["k"], kty, m, col0=kb0, cm=K, shape=(C, 16))
    last = (kb0 == K - 16)
    accs = mma_gemm(e, f"kkt{kb0}", C, C, 16, W, ksl.name, ksl.ty,
                    ksl.name, ksl.ty, kt=True, accs=accs,
                    epilogue=kkt_epi if last else None)
    e.l("gpu.barrier")
```

每链：`stage_rows`（`st.shared` 逐元素暂存——r6 规则：ldmatrix 操作数必须 `st.shared` 写入）→ `mma_gemm` 一条 $K{=}16$ 链 → r10 再暂存栅栏。累加器以 `accs=` 寄存器链跨链携带（r5 引入、r8 `k0=` 机制的同族），**片断/mma/累加序列与旧整 $K$ 调用完全一致**——$kkt/A/P$ 按构造位同。kkt 的 epilogue 在最后一个链上双写（原始值入 `kkt_all`；$\beta_i\,2^{gc_i-gc_j}$ 衰减、严格下三角选择后入 `kktf`）。$P$ 链同理（`p_epi`：含对角 `sle` 选择 + 全局直存）。

**【IR/PTX 前后对比】**

| PTX/SASS 类 | r7 旧形（整 K + k_t/q_t + cp.async） | r12 新形（K16 切片链） |
|---|---|---|
| `ld.shared` | 832 | **49** |
| `mma.sync` | 256 | 384 |
| `ldmatrix` | 96 | 384 |
| `cp.async` | 20 | **0** |

**【效果】**（r12，配对 x3，T2048）：wy 0.7005→0.2277 ms（**3.08x** 单独臂；终链表 3.04x）；T512 0.0920→0.0442（2.08x）。数值：切片化是位同重构（bit-identical by construction），六梯度数位不动。

### 7.1.3 $(I+A)^{-1}$ 求解：thread-per-column 前代换

**【动机】** C=64 的求解拆为两个 32×32 单位下三角基 $T_{00},T_{11}$ 与合并项 $T_{21}=-T_{11}A_{21}T_{00}$。r6 之前由**单线程**行串行步进（CTA 关键路径上约 1.1 万条依赖 FMA）。

**【机制与保序证明】** 单位下三角前代换的列 $j$ 只读列 $j$：

$$T[i,j] = -\sum_{j\le \kappa<i} A[i,\kappa]\,T[\kappa,j]$$

右侧 $A[i,\kappa]$ 与 $T[\kappa,j]$ 的第二下标均为 $j$——列独立性成立。故 64 线程（`tid < C`）各持一列、**无内部栅栏**并行求解两基全部列；每列的 $\kappa$-升序 f32 链是**与串行行走完全相同的求和、完全相同的次序**——$T$ 位同。严格上三角（含 $T_{12}$ 块）显式清零（旧代码依赖 smem 分配即零的未定义行为，r6 改为显式写零，值同）。合并两步 `sgemm`（`tmpm = A21@T00; T21 = -(T11@tmpm)`，L2441–2443）。

**【源码片段】**（L2360–2396，节选）：

```python
tg = e.v("wyt")
e.l(f"{tg} = arith.cmpi ult, %tid, {e.ic(C)} : index")
e.l(f"scf.if {tg} {{")
...
e.l(f"{ssum} = scf.for {sk} = {j} to {ir} step {e.ic(1)} "
    f"iter_args({sac} = {z0}) -> (f32) {{")
aik = load2(e, kktf, ir, sk)      # A[ir, sk]：本列元
tkj = load2(e, Tf, sk, j)         # T[sk, j]：本列已解元
pp = op2(e, "mulf", aik, tkj)
ss = op2(e, "addf", sac, pp)
e.l(f"scf.yield {ss} : f32")
```

**【IR/PTX 前后对比】** 前：`scf.if %isz`（单线程）内 $O(C^3/2)$ 串行迭代、每步 smem 往返依赖链；后：`scf.if (tid<64)` 的 64 路列并行、每线程至多 32 步 $\kappa$-升序内积链（`scf.for` iter_args 寄存器驻留），发射的 MLIR 算子类不变（`arith.mulf/addf` + `memref.load/store`），仅谓词域从 `%isz` 换为列守卫。

**【效果】**（r6，同窗）：wy 1.45→0.70 ms（**2.06x**）@T2048；T512 0.199→0.095（2.1x）。数值：$T$ 位同（六梯度与 r5 完全一致）。

### 7.1.4 $u/w$ 尾：2-链 split-bf16 与 cvec 列尺度

**【动机】** $u/w$ 是 wy 最后的标量/tf32 大尾部。r3 bisect 已证明 tf32 化的 `wy.u` 把 dk 推到 $1.06\times 10^{-2}$（超 $8\times 10^{-3}$ bar），tf32 路线被封死；r12 以 r11 caryA 的 split-bf16 剧本接手。

**【机制】** 以收缩下标（列）尺度预乘后拆分 A 侧：

$$u = \big(T\odot\beta_{\mathrm{col}}\big)v:\quad \tilde T = \mathrm{split}\big(T_{ij}\,\beta_j\big),\qquad u = \tilde T_{hi}\,v + \tilde T_{lo}\,v$$

尺度先在 f32 中乘一次（一次舍入），再进入 $2^{-17}$ 残差类的 hi/lo 拆分（`stage_split_ew cvec=` 参数，r12 引入"任意尺度折叠"）；$w$ 同理用 $bgc_j=\beta_j 2^{gc_j}$。B 侧是**原始 bf16 全局**（$v$/$k$），转置暂存到 mma 的 $[n,k]$ 定向（`stage_raw_T`）——**精确、零舍入**。A 侧分裂读 `T_all` 全局（本 CTA 刚写入——同 kernel 全局写后读、L2 热），**不**读 `Tf` tile：`Tf` 的竞技场窗口在循环期间让给 B 原始 tile。两链（hi、lo）以 `accs=` 寄存器链接力，epilogue 全局直存 `u_all/w_all`——`emap/gemm/store_rows` 三段扫描与 `opc/accc/kslc` tile 全部退役。数值注意（r12 记录）：$u$ 原为标量 f32（逐积 fma）、$w$ 原为 tf32（双操作数 $2^{-11}$）；两者统一到 $2^{-17}$ split 类——$w$ **严格变紧**，$u$ 经 $S\to S_n\to dk$ 放大路径重掷（dk 数位随后在 r13 收紧，见 §7.3.6）。

**【源码片段】**（L2546–2567，u 相）：

```python
stage_split_ew(e, auhi, aulo, G["T_all"], ccty, m, C, 0, C, C, cvec=beta)
for vs in range(0, V, BVU):
    stage_raw_T(e, vb, G["v"], vgty, m, V, vs, BVU, C)
    accs = mma_gemm(e, f"uh{vs}", C, BVU, C, W, auhi.name, auhi.ty,
                    vb.name, vb.ty, kt=True)
    accs = mma_gemm(e, f"ul{vs}", C, BVU, C, W, aulo.name, aulo.ty,
                    vb.name, vb.ty, kt=True, accs=accs, epilogue=epi_u)
    e.l("gpu.barrier")   # r10 再暂存栅栏
```

w 相在同一 arena1 槽位重拆（`awhi/awlo`，尺度 `bgc`；上相尾栅栏已围栏）。

**【IR/PTX 前后对比】** 标量 `arith.mulf/addf` 双重循环（f_up/f_w 扫描）与 tf32 `mma.m16n8k8`（`--tf32` 位点）→ 2 条 bf16 `nvgpu.mma.sync ... mmaShape=[16,8,16]` + `nvgpu.ldmatrix`；每元素 MUFU（`ex2`）消失（尺度提升为 $bgc$ 向量，SASS EX2 类随整 kernel 的 exp2 hoist 从 149 降到 4，r8）。

**【效果】** 并入 §7.1.6 的整 kernel 数字（3.04x 的组成部分）；dk 经 $u$ 重掷后 1.718e-3/1.724e-3（6/6 维持，余量 4.7x/4.6x，r12 数值表）。

### 7.1.5 $gc$ 前缀和：Hillis–Steele shuffle 扫描

**【动机】** 旧实现是每线程**冗余**执行同一 C 步串行行走（每步全局读 + smem 往返的依赖链，所有 128 线程重复同一工作以避免栅栏）。

**【机制】**（`gc_cumsum_shfl`，L1221–1291，`GDN_WY_GC_SHFL` 默认开）：三相——(0) 并行合并停放原始 $g$ 值；(1) `tid<C` 各持一元，warp 内 5 步 `gpu.shuffle up`（偏移 1,2,4,8,16）扫描；**warp 均匀性**：C=64 恰被 warps 0/1 的 `tid<C` 谓词整覆，故 shuffle 的全 lane 收敛成立；lane 31 停放 warp 总和入 $[2,1]$ smem 向量 `wtot`（`andi tg,l31` 守卫防止 warps 2/3 越界写）；读下标 `minsi` 钳制使死 lane 的 smem 读保持在界；(2) warp≥1 加 warp 0 总和，守卫写回。

**【数值声明】** 确定性（固定平衡树）但**不位同**于 k-升序串行行走——求和次序改变，$gc$ 每数位以 $\sim2^{-24}$ 相对类重掷，而 $gc$ 供 $T$ → 六梯度全部重掷。实测（r12）：`T_all` 重掷 4.2e-7、dk 绝对位移 $1.2\times10^{-4}$；6/6 通过，dk 余量 4.7x。**这是本计算族唯一的非位同数值变更**，按 6/6 门径独立放行。

**【源码片段】**（L1256–1264）：

```python
for off in (1, 2, 4, 8, 16):
    e.l(f"{o}, {p} = gpu.shuffle up {x}, {_i32c(e, off)}, {_i32c(e, 32)} : f32")
    y = op2(e, "addf", x, o)
    e.l(f"{x} = arith.select {p}, {y}, {old} : f32")
```

**【IR/PTX 前后对比】** 前：C 步 `memref.load`/`arith.addf` 依赖链 ×128 线程冗余；后：5 条 `gpu.shuffle up` → PTX `shfl.sync.up.b32` ×2 warp + 跨 warp smem 偏移。

**【效果】** wy 单独 +4%（0.2374→0.2278 配对，r12 l1b 臂）；默认开。

### 7.1.6 生存期收缩：90 624 → 45 832 B，2 CTA/SM

**【机制】** 上述机制的生命期结果：常驻 tile 只剩 $[C,24]$ 切片对（`ksl/qsl`）、两竞技场宿主（`kktf` $[C,C{+}8]$ f32 = 18 432 B，host $u$/$w$ 相的 A-分裂对 9 216×2；`Tf` $[C,C]$ f32 = 16 384 B，host B 原始 tile `vb/kb`）、求解 scratch `tmpm` 与向量。池 = 18 432+16 384+3 072+3 072+4 096+8+768 = **45 832 B ≤ 50 176 B → 2 CTA/SM**（旧 1 CTA、12.5% 占用率）；寄存器 255→158/161、0 spill（r12）。别名细节见 §7.4。

**【效果汇总（wy 全 kernel，r12 终链表，配对）】**

| 量 | r7 形 | r12 终形 |
|---|---|---|
| 池（B） | 90 624（1 CTA） | **45 832（2 CTA）** |
| 寄存器 | 255 | 158/161（0 spill） |
| `ld.shared` / `mma.sync` / `ldmatrix` / `cp.async` | 832 / 256 / 96 / 20 | 49 / 384 / 384 / **0** |
| wy @T2048（ms） | 0.700 | **0.2299**（3.04x） |
| wy @T512（ms） | 0.0920 | 0.0442（2.08x） |
| bwd e2e @T2048（ms） | 4.541/3.927 | 3.734/3.331（1.216x med / 1.179x min） |

### 7.1.7 R17 FPIPE 的 `T_all→Tf` DMA（中性，保留）

r17 为 wy 加了一条异步预取：`T_all` 存储栅栏一过即发 `T_all→Tf` 的 16 KB 原始 f32 DMA（此刻 `Tf` 已死——求解 tile 的最后读者就是存储扫描），整个 $P$ 相（8 条切片暂存 + mma 链）作掩体；$u$ 分裂随后读驻留 tile（`Tf` 只喂 `ld.shared`，r6 规则合法），同 $\beta$ cvec 算术 → `au*` 位同。**配对中性**（1.001x/1.006x/1.01x 三形状）：源是同 kernel 刚写的一遍性读、L2 驻留、无可藏延迟。保留（默认开）：零成本且 T8192 工作集（`dvp_all` 64 MB）在 L2 边缘，更大 B 时掩体可能生效（r17）。

---

## 7.2 dqkwg：$dq$ / $dk_{\mathrm{local}}$ / $dW$ 计算核

### 7.2.1 公式分解与相位划分

`_gen_dqkwg_v4`（L5853–7042，出厂臂；`gen_dqkwg` 按 $BH\cdot N_C\ge 164$ 分派，小网格保 v3）实现的量：

$$
\begin{aligned}
\mathrm{d}Pr &= \tfrac{1}{\sqrt K}\textstyle\sum_{vs}\mathrm{d}O_{vs}\,v_{\text{new},vs}^\top &&\text{（P2，mma 链）}\\
r_4 &= \mathrm{rowsum}(\mathrm{d}Pr\odot P),\quad c_{4r}=\mathrm{colsum}(\mathrm{d}Pr\odot P) &&\text{（warp 蝶形）}\\
\mathrm{d}Pm &= \mathrm{stricttril}(\mathrm{d}Pr\odot 2^{gc_i-gc_j}) &&\text{（原地 emap）}\\
dq &= \mathrm{d}Pm\,k \;+\; \mathrm{diag}(2^{gc}/\sqrt K)\,(\mathrm{d}O\,S_n^\top) &&\text{（早跑停车 + P11 链 + R-A-S）}\\
\mathrm{d}v_p &= P^\top \mathrm{d}O/\sqrt K+\mathrm{diag}(2^{gcl-gc_i})\,(k\,\mathrm{d}h^\top) &&\text{（P9，pt/akb/bh 链）}\\
\mathrm{d}W &= -\,\mathrm{d}v_p\,S_n^\top &&\text{（P11，dw 链）}\\
dk_{\mathrm{local}} &= \big(T^\top \mathrm{d}W\big)\odot bse \;+\; \mathrm{d}Pm^\top q \;+\; dkB_{\Sigma},\qquad bse=\beta\odot 2^{gc} &&\text{（早跑停车 + P12 TW + epilogue 折叠）}\\
r_3 &= \mathrm{rowsum}\big(dq_1\odot q\big),\qquad dg_{\mathrm{state},i}=s_5+\textstyle\sum_{j\le i} rowB_j &&\text{（蝶形 / 单线程串行）}
\end{aligned}
$$

实现细节：pt 对把 $rie_i=2^{gc_i-gcl}/\sqrt K$ 折入 $P^\top$ 的 hi/lo 分裂、链尾单因子 $egc_i=2^{gcl-gc_i}$ 一次相乘还原 $P^\top\mathrm{d}O/\sqrt K$（r8 设计，$egc\cdot rie\cdot scale=scale$ 至 $2^{-24}$）；$bse$ 行尺度折入 TW epilogue（与旧原地 emap 同一 mulf，r12）。

### 7.2.2 dPm 尾部早跑与全局停车（R-A-S）

**【动机】** r8 末的 dqkwg 是延迟受限的 1-CTA 体（99 328 B 池；ncu：新旧臂内存吞吐 12.7% 对 28.6%，serialization 下时长不变——释放的发射槽被暂存/mma 延迟回填）。r9 的占用率杠杆需要把池压到 ≤50 176 B，最大障碍是 accCC（dPm，16 384 B）跨越 **全部** 后续相位的生存期：dq/dk 尾部若在 TW 相才跑，accCC 必须活到 P12。

**【机制】** 把两个 dPm 尾部**提前**到 emap 之后立即执行（"早跑"），其 $[C,BK]$ 切片**停车**（park）到 `dq`/`dk_local` 全局各自的目标单元——不占 smem：

- `dq_dPmk`：A = dPm 自然定向分裂（`_split_bf16` 自 accCC），B = 原始 $k^\top$（`stage_raw_T`，精确）；
- `dk_dPmq`：A = **同一批 arena tile 的转置重分裂**（`_split_bf16_T`；dq 收缩 dPm 的行、dk 收缩其列，两定向不能共享一个分裂但可顺序共享 tile），B = 原始 $q^\top$；
- 其后 dW 循环对 `dq[ks]` **读-加-存**（R-A-S），TW 循环对 `dk_local[ks]` 读-加-读-加-存（叠加 $dkB_\Sigma$）。

代价核算（r9）：每 chunk +96 KB 的 L2 驻留往返，换取 accCC-dPm 16 KB 生存期在 pt 阶段之前死亡——arena1 随之为 pt/bdt 让位。r14 进一步把停车写直入链尾 epilogue（`park_epi`），accCK 中转 + `store_rows` 扫描 + 每尾 2 条栅栏退役；R-A-S 折叠进 `epi_dq1`（$dq[ks]\ {+}{=}\ xs$，与旧扫描同加法次序）与 `epi_tw`（$dk_{\mathrm{local}}$ 三项折叠，同 `addf` 次序）。

**【源码片段】**（L6546–6559，dq 尾；FOLDS 停车 epilogue）：

```python
_split_bf16(e, dhi, dlo, accCC, C, C)
for ks in range(0, K, BK):
    stage_raw_T(e, krt, G["k"], kty, m, K, ks, BK, C)
    accs = _mh_g(e, f"dmh{ks}", C, BK, W4, dhi.name, dhi.ty, krt.name, krt.ty, C)
    accs = _mh_g(e, f"dml{ks}", C, BK, W4, dlo.name, dlo.ty, krt.name, krt.ty, C,
                 accs=accs, epilogue=(park_epi("dq", ks) if FOLDS else epi_dq_park))
    e.l("gpu.barrier")
```

**【IR/PTX 前后对比】** r8 形：tf32 `tgemm` 尾（192 条 k8 链静态计数，`mma.m16n8k8` + f32 tile 暂存 `tCC/tCKa/tCKb` + `cp.async`）；r12 SB 形：bf16 `mma.sync [16,8,16]` ×2/位点 + `ldmatrix`，`cp.async` 46→**0**、`ld.shared` 1234→**71**、栈泄漏 72 B→0、寄存器 255→240（r12）。早跑重构本身只引入 ~1-ulp 级加法重排，dk 数位不可见（r9：dk 1.719e-3/1.710e-3 与基线同印刷位）。

**【效果】** r9：dqkwg 1.772→1.075 ms（**1.65x**），bwd e2e 6.00/5.36→5.20/4.66（1.155x med）；r12 SB 再叠加（dqkwg 1.075→0.9421，1.14x）且 **dq 数位收紧 4.910e-4→1.824e-4（余量 16x→44x）**——tf32 对 dPm 的 $2^{-11}$ 截断换成 $2^{-17}$ split 类，dk 1.710e-3→1.691e-3。

### 7.2.3 P11 vs-外层重构：元素级保序证明

**【动机】** dW/dq1 合并循环（相位 P11）按 (ks, vs) 16 组迭代；其中 `akb`（原始 dO 切片）、`vnh/vnl`（dvp 分裂）是 **vs-专属**（值跨 ks 不变），却在 r15 2-CTA 臂里随每组重暂存（16 次冗余扫描）。 slab 修复需要 ~17.4 KB 池余量而 2-CTA 臂只有 3 560 B（r15 负结果）；r16 改为**换循环块**。

**【机制与保序证明】**（`GDN_DQKWG_P11R`，默认 2）：P11R 个 per-ks 的 dw/dq1 mma 累加器链集作为**并行寄存器链**跨一个 vs-外层块存活；vs-专属 tile 每 vs 只暂存一次。关键论断——**元素级保序**：对每个输出元素，其全部乘积仍以同一 (vs, link) 序列进入同一条 per-ks mma 累加链；改变的只是 16 个 (ks,vs) 组的**代码次序**，而非任何输出元素内的算术次序。r3 的 warp 行和仍 ks-升序发射；accCK 保持单 tile——在 last_vs 处，每个 $k_i$ 的 `epi_dq1` 之间夹着**它自己的** r3 行和（$k_i$ 的行和先退役 accCK，$k_{i+1}$ 的 epilogue 才重写；链/行和自带的尾栅栏串行化）。

**【源码片段】**（L6788–6818，节选）：

```python
for ksb in range(0, K // BK, P11R):
    accsW = [None] * P11R
    accsQ = [None] * P11R
    for vs in range(0, V, BV):
        stage_raw_bf16(e, akb, G["do"], vty, m, V, vs, C, BV, bar=False)
        stage_split_ew(e, vnh, vnl, G["dvp_all"], dvt, m, V, vs, C, BV)
        for ki in range(P11R):
            ks = (ksb + ki) * BK
            stage_split_ew(e, bhh, bhl, G["h_all"], hty, m, V, ks * V + vs, BK, BV)
            accsW[ki] = mma_chain_if(e, f"dw{ks}_{vs}", C, BK, BV, W4, None,
                                     ops_w, accsW[ki], ..., bar=False)
            accsQ[ki] = mma_chain_if(e, f"dq1_{ks}_{vs}", C, BK, BV, W4, None,
                                     ops_q, accsQ[ki], ...)
            if last_vs and P11R > 1:
                warp_rowsum(e, r3, accCK, C, BK, term=mk_term_r3(ks), bar=True)
```

**【效果】** 实测证明（r16）：T2048 全链 35/35、T8192 36/36 输出数组**逐元素位同**基线；六个验证余量数位与 r15 表全等。计时（配对）：dqkwg 0.5445→0.5236（**1.040x**）@T2048、1.8233→1.7556（**1.039x**）@T8192。对照臂 P11R=4（全灭冗余，4 次扫描）：自然体 222 寄存器 → `.maxnreg 128` 下 spill 涨至 232 B 栈/276 B 存/632 B 载，仅 1.026x——第 4 个累加器链集的 spill 流量吃掉多省的暂存；P11R=2（自然 174 寄存器，96 B 栈/96/148 B spill）胜出。vs-专属扫描 16→8 次；per-(ks,vs) 的 `bhh/bhl` 重暂存不可约简（每对是相异数据）。

### 7.2.4 pt 对的生存期治理

**【动机】** pt 对（$pthi/ptlo$，$P^\top\odot rie$ 的 hi/lo 分裂，9 216+9 504 B）是 P9 dvp 相的 mma A 侧。r8 排名曾把"退役 pt 对（18 432 B）"列为 2-CTA 前提；最终系统保留 pt 但将其**生存期压缩进死窗**，配套三轮治理：

1. **P 级退役（r9，已入库）**：r3/r4/$c_{4r}$ 的 P 因子改从**全局**读（`gcol`，行和 lane 合并；列和的第二遍跨步读 L2 热——`P_all` 切片刚被流过），16 KB 的 P smem 驻留退役；蝶形求和树与 v3 原地变体位同。
2. **pst 重驻（r13，`GDN_DQKWG_PST` 默认开）**：P 选择性地重暂存进 dhi/dlo 窗口（dPm 分裂登场前空闲）：**P 值位同**（同 f32 元素、同 lane/树）→ r4/c4r 位同；实测 1.01x（噪声带，r13 诚实记录"非验证的赢"），因零成本且值同而保留。
3. **pt 异步化几何封锁（r19 负结果）**：dPm 尾部期间 arena1 字节级满载——accCC $[0,16384)$ 是两次重分裂的分裂源、dhi/dlo 占 $[16384,34816)$、arena2 被 krt/qrt 占据——dPm 掩体下**没有任何窗口**可放飞一个原始 P 半区；kernel 顶部替代方案处处冲突（`pst` 自身的字节是唯一 8 KB 窗口，退役 pst 等于退还 r13 的 LDS 收益）。记录在案、未实施。

**【效果】** 1+2 合计使 pt 对成为"P9 相一次性暂存"（L6600 `stage_split_T` 一处调用），驻留窗 $[0,18432)$ 复用 accCC 死后字节（pthi@0、bdt@9216）；r17 FPIPE 期间 `dprT` 车辆停 accCC 自身字节而 akb@18944 不受扰（见 §7.2.7 的失败案例）。

### 7.2.5 V-宽 slab 减肥（r14）与 2-CTA 下的让位

**【动机】** r13 末的 dqkwg 0.937 ms 中，P11 的暂存冗余（dvp 分裂随 (ks,vs) 16 次重暂存）与 R-A-S 往返是可见残项（r13 负结果清单）。

**【机制】**（`GDN_DQKWG_DIET`，r14）：dvp 分裂（`svh/svl` $[C{+}1/C{+}3, V{+}8]$）**一次**staged（替代 16 次）、h 分裂（`hbh/hbl`）per-ks 4 次（替代 16 次）、原始 dO（`dbo`）一次；mma 链读 slab 在 $k_0=vs$ 处的列（六元组 ops）——同操作数值、同链次序/累加 → 乘积位同。slab 叠放 arena1 的 pt/dvp 与 TW 之间的死窗（34 816→52 224）；池 66 048→100 864 B（反正 1 CTA）。

**【效果与让位】**（r14，配对）：饮食单独 0.9371→0.7348（**1.28x**）@T2048；寄存器 240→**128**（16 B spill）——这是 512 线程栈的使能器（§7.2.7）；栅栏 206→112（r7 不相交 tile 规则 + 只读链合并）。但 r15 的 2-CTA 化发现 slab 窗口正是把池钉在 1 CTA 的元凶：**出厂 2-CTA 臂撤下 slab**（`SLABS = DQKWG_DIET and not TWO_CTA`），P11 冗余由 §7.2.3 的 P11R 块换承接；FOLDS（epilogue 停车/折叠/栅栏饮食）保留。

### 7.2.6 dPr 的 warp 映射：strip → mh

**【动机】** strip 映射（`mma_gemm`）把 warp $w$ 绑到 N-条带 $[w\cdot n_w,\dots)$，故 $W\le N/8$：N=32 形（dPm 尾、TW）在 8 warp 下只有 4 个 mma 活跃；r8 第一版 v3 的 dvp/dW/dq1 链跑在 `%wgv4` 守卫内——半 SM 闲置（ncu：89% no-eligible 周期、每调度器 ~2 活跃 warp、shared-scoreboard 49.9%/18.1 cyc/inst）。

**【机制】**（`mma_gemm_mh`，L1821–1920）：2D warp 映射——M 维也二分：warp $w$ 持 $n_i=w\bmod n_{pr}$ 与 mi-块对 $(w\!\!\!/n_{pr})\cdot m_{pw}\cdot 16$；$W=8$ 全活跃、无守卫、无 `scf.if`-yield 区，B 片在 warp 的 mi 对间复用。片断/lane 数学与 strip 同（G4 锚定语义），仅 warp→tile 指派不同；`accs` 与 `mma_gemm/mma_chain_if` 链兼容。r14 扩至 $W\le16$（512 线程臂；映射按 halves 泛化，片断数学 warp-局部）。首版 bug（r8 验证门捕获）：行偏移误加块计数——warps 4–7 读行 {2,18} 而非 {32,48}，0.9 类失败，一行修复为 $(w/n_{pr})\cdot(m_{pw}\cdot16)$。

**【源码片段】**（L1845–1851）：

```python
e.l(f"%mh_{name} = arith.divsi %mW, {e.ic(npr)} : index")
e.l(f"%mhb_{name} = arith.muli %mh_{name}, {e.ic(mpw * 16)} : index")  # 行基（行数）
e.l(f"%nx_{name} = arith.remsi %mW, {e.ic(npr)} : index")
e.l(f"%ncv_{name} = arith.muli %nx_{name}, {e.ic(8)} : index")
```

**【IR/PTX 对比】** 前：`scf.if %wgv4 { ... }` 包裹的链（4/8 warp）+ `scf.yield` 跨区累加器；后：平铺无守卫链，寄存器 255（60 B spill）→248（0 spill）（r8）。

**【效果】** 出厂 256 线程 2-CTA 臂：dPr（N=C=64，strip 容 W=8）走 strip；dPm 尾与 TW（N=BK=32）走 mh；512 线程臂 dPr 亦切 mh（$W{=}16>8$，L6439–6452）——同片断值、同 kb 累加、同 warp 树和 → 数位不动（r14 值精确轮）。

### 7.2.7 占用率三级跳：竞技场化 → 512thr/128regs → 2-CTA 四 tile 纯别名

**（a）r9：48 640 B 竞技场 + 96 寄存器。** PTX 级生存期别名（§7.4 机制）把 23-malloc 体折叠为双竞技场（arena1/accCC 16 384→17 408；arena2/vnh 10 240）：别名后 ptxas 落在 **96 寄存器 / 0 spill**（未别名同体要 255+spill——别名补丁同时坍缩了地址空间复杂度），256 线程 ×2 CTA=16 warp/SM 的 128 寄存器门槛免费达成。

| ncu（T2048，kernel-only） | v3 基线 | v4（r9） |
|---|---|---|
| 线程 / 静态 smem / 寄存器 | 256 / 99 328 B / 132 | 256 / 48 640 B / **96** |
| 块上限（smem, regs） | 1, 1 | **2, 2** |
| 理论/实际占用率 | 16.67% / 15.81% | **33.33% / 30.28%** |
| 每 SM 实际 warp | 7.59 | **14.53** |
| 每调度器活跃 warp | 1.98 | 3.69 |
| no-eligible 周期 | 88.98% | 79.56% |
| SM 内存吞吐 | 13.2% | 20.0% |

教训一（r9 负结果）：128 线程 ×2 CTA 是结构性空转——sm_120 每 SM 48 warp 槽（12/调度器；r7 成本模型曾按 32 计），2×4=8 warp 与 1×8 相同，墙钟持平（1.780 对 1.775）而 T512 损 1.8x；胜出需要 16 warp。

**（b）r14：512 线程 × 128 寄存器铺满 16 warp。** r14 饮食把自然体压到 128 寄存器后，512×128 = 65 536 = 整个寄存器堆合法（256 线程 240 寄存器体在 512 线程下无法启动）：1 CTA ×16 warp = 4 warp/调度器（原 2）。ncu（饮食 cubin）：90.3% no-eligible、1.97 活跃 warp/调度器——纯延迟画像。dqkwg 0.7348→0.6420（饮食+512 线程对 HEAD 0.940 合计 **1.46x**），e2e 2.690→2.389（1.126x）。

**（c）r15：2×256 线程两 CTA——四 tile 纯别名编排图。** 512 线程臂达到 16 warp 但**单指令流**；2×256 = 两个独立流。r14 草案（accCK 退役到全局 + bhh/bhl 别名）算出 52 736 B，超 bar 2 560 B；r15 峰值生存期审计发现 **四个池 tile 可以纯别名进 arena1 既有窗口**，无需全局停车：

```
arena1（宿主 malloc = accCC，34 816 B；偏移为出厂非 TTF 臂）
0        pthi 9216 ──┐ P9 (dvp)：pthi+bdt+bhh+bhl+akb = 24 064 B
9216     bdt  4608 ──┘
13824    bhh  2560 ──┐ P11 (dW/dq1)：bhh+bhl+akb+accCK[24K,32.2K)
16384    bhl  2560 ──┘   （vnh/vnl 驻 arena2）
18944    akb  5120
24064    accCK 8192   ← dq1 部分和，r3 行和的 LDS 源（r3 保 smem 读，无 L2 往返、无新全局）
32256    …arena1 尾部余量 2 560 B（无可用 slab 变体；最小的 hbh/hbl 也需 17 424 B）
—— 时间不相交的其它驻留：accCC@0（P2 dPr 至 P3 dPm 分裂源）；
   dhi@16384+dlo@25600（P3/P4-7 dPm 尾）；pst@16384（P3' r4/c4r）；
   dprT@0（R17 FPIPE dPr 掩体）；tTh2@0+tTl2@9216（P12 TW）

arena2（宿主 malloc = vnh，10 240 B）：vnh@0 + vnl@5120（P9/P11）；
   ptlo@0 / krt@0 / qrt@0 / dWhi@0+dWlo@4608（相位互斥复用）
池 = 34 816 + 10 240 + 10×256（标量向量）= 47 616 B ≤ 50 176（余量 3 560 B）
```

**相位审计**（r15，每一相位边界有栅栏围栏）：P2 dPr：accCC$[0,16K)$ + akb$[18.9K,24K)$——accCC 只在**最后**一个 vs 的 epilogue 写入，字节不相交；P3 dPm 尾：accCC + dhi/dlo$[16K,34.8K)$（r19 表述：此期间 arena1 **字节级满载**）；P9 dvp：pthi@0+bdt@9216+bhh+bhl+akb = 24 064；P11：bhh+bhl+akb+accCK；P12 TW：tTh2+tTl2（epi_tw 折叠，无 accCK）。零新栅栏、零全局流量；P11 栅栏饮食（三路重暂存扫并两栅栏，akb 搭分裂暂存的栅栏；dw 链 `bar=False`，dq1 自带栅栏围栏组）。

**寄存器侧的关键发现（r15）**：出厂 cubin 实为 **168 寄存器**而非 128——`ptxas_v.txt` 是 run_l2.sh 的**第一遍** ptxas（smem_fix 前）；`gdn_smem_fix.py` 的第二遍释放出不同调度（168 寄存器，0 spill）。第一次 A/B 因此读到 0.78x 回退（ncu：Block Limit Registers = 1，8 warp/SM，劣于 r14 的 16-warp 形）。两条工具链定律（响亮）：(1) 作数的寄存器数是 smem_fix 遍的——`ptxas_v.txt` 不可信；(2) **ptxas-blackwell 无视 `--maxrregcount`**，但 PTX **entry 体内 `.maxnreg` 指令（与 `.maxntid` 并列）被遵守**：128 寄存器/32 B spill（栅栏饮食后 64 B）。`GDN_MAXNREG_ENTRY=dqkwg:128` 注入成为出厂构建步骤。

**【效果】**（r15，配对）：dqkwg 0.632→0.547（**1.16x**）@T2048、2.323→1.844（**1.26x**）@T8192（2048 CTA 填满翻倍 SM 槽，waves 6.24→3.12）；e2e 2.3794→2.2913（1.039x med/1.041x min）@T2048、1.050x/1.053x @T8192。ncu：Block Limit（regs, smem）=（2,2），实际占用率 15.02%→**29.17%**、14.23 warp/SM @T8192、活跃 warp/调度器 1.97→3.63、no-eligible 89→83%、SM 18→30%、MEM 19→36%——r14"独立流"论点的实测。数值：全数字位与 r14 相等（值精确轮）；T512 载荷字节同 HEAD（网格门控保 v3）。

**【R17 FPIPE 与 dprT 的失败案例（编排图的负结果注脚）】** r17 把 dPr 循环 vs=0 的两个操作数（`akb←do`、`vnew_all→dprT`）提为 gc 链掩体下的 prologue 组；`dprT` 落 arena1@0——accCC 自身字节，整个 dPr 循环期间未触碰（mma 累加器驻寄存器；accCC 首个 smem 写在末链 epilogue，远晚于分裂退役 dprT）。**第一版**把 dprT 停 +16384（dhi/dlo 窗），其 8 192 B 尾部覆写 akb@18944（dPr 的 A 操作数，活）——dq/dk/dg NaN，验证门一次定位。免费窗审计从此记录在代码注释。该 FPIPE 配对 1.009x/1.008x（r17），spill 类 96B/96/148→88B/84/120 顺带下降。

### 7.2.8 K32 链化

**【动机】** r22 审计（§7.2.9）定位出 +46 寄存器覆盖的载体是 K=64 mma 位点的片断质量；r6 已立"K=32 片断生存期"律，r8 的 `k0=` 机器就是为此而生。

**【机制】**（`GDN_DQKWG_K32`，默认 0）：`_ops_g/_mh_g/_gemm_g` 包装器把每条 K=64 链拆为 k0=0/32 两条 K=32 链（`KGRAN` 机器；`GDN_DQKWG_K16` 深化为 K/16）。每累加器片断的 `mma.sync` **序列不变**（同操作数、同 kb 次序、同 acc 链接——只有 ldmatrix 发射点交错进来），故数据流按构造位同（r8 文档串的等价类："$[*,C{+}8]$ tile 上两条 K=32 链 == 一条 K=64 链"）。

**【源码片段】**（L5937–5944）：

```python
def _ops_g(a, aty, b, bty, klen):
    if klen > KGRAN:
        return [(a, aty, b, bty, k0, KGRAN) for k0 in range(0, klen, KGRAN)]
    return [(a, aty, b, bty, 0, klen)]
```

**【IR/PTX 对比】** PTX：每 K=64 位点的 `ldmatrix ... numTiles=4` ×（A 全 kb）预发射 → 拆为两半交错；`.maxnreg 128` 下 spill 类 **88B/84/120 → 64B/64/104**；自然体 174 不变（ptxas 保持无 spill 选择）。

**【效果】**（r22，配对 x3，共载）：dqkwg 0.5144→0.5147 @T2048 / 1.7289→1.7345 @T8192——**1.00x 中性**（spill 流量本身不是杠杆，r17 THR=128 律的第 4 个数据点）。位同证明：35/35+36/36 数组逐元素位等、六数位 == 记录表、det N=5 新进程位同。默认 0、旋钮保留（spill 类削减免费，未来共居压力下或有意义；K16 同律无增益）。连带负结果：P11R=1+K32（最深层饮食态）0.5141→0.5346（0.96x 回退——2 链体付出 r15 每-(ks,vs) 重暂存冗余，链数不可换 spill 类）；K32+TTF 0.99x/0.986x（异步组令牌跨 P11 存活，饮食不吸收）。

### 7.2.9 寄存器分配器关门审计：+46 联合分配的边界测绘

r22 以 38 个仅编译探针（相删除 XD / 前缀 XP / 微二分 XS / 结构开关；探针生成器 `scratch/r22b_gen_probes.py`，绝不运行）对 174 寄存器自然体做了**边界测绘**（boundary mapping）——这不是一个待实现的优化，而是对"为什么此门关死"的定性：

1. **前缀图**（自然 smemfix 寄存器，逐相累加）：prologue 36 → +dPr 92 → +r4c4 98 → +tails 102 → +dvp 122 → +P11 128 → +TW 170 → +out **174**。跳变在 TW 叠上 P11（+42）。
2. **删除图**：删除 dvp / p11 / tw **任意一个**，体都坍缩到 128——**+46 覆盖是 dvp+p11+tw 的联合分配**，不是某一相的局部需求。微二分：只删 TW 的 mma 链（twmma）同样坍缩 128；dwsplit −10；twsplit/twepi 0。
3. **载体**：K=64 片断质量——`mma_gemm/mh` 在 `mma.sync` 循环**之前**发射全部 kb `ldmatrix` 片断，故每条 Kred=64 链（dvp 的 pdo 链、4 条 dPm 尾链、3 条 TW 链——6 个位点）同时持 A（$m_{pw}\times 4_{kb}\times 4$ 寄存器）+B 约 **48 寄存器/线程**存活——r6 "K=32 片断生存期"律被违反 6 处；r8 的 `k0=` 机器恰为规避它而存在（K32 位点因此非绑定）。
4. **门本身被分配器关死**：无上限遍报告 174 寄存器 0 spill（ptxas 偏好无 spill 甚于占用率）；而**强制 `.maxnreg` 的分配器在任何上限下都到不了 0 spill**——上限 128..174 的 spill 曲线在 **cap 174 处仍地板于 40 B 存/60 B 载**（该指令把 ptxas 切换到与其自然选择结构性不同的受限分配）。**"自然 ≤128 ⇒ .maxnreg 128 下 0-spill"在本工具链上对该丰富度的体为假**，与任何饮食设计无关。

**结论**（r22 排名）：dqkwg 在值同/spill/暂存/驻留各类别于本形状**无剩余实测开启的门**；剩余类别仅新数值类（如 P11R W-全局重驻留——纸面定价：smem 停车 +16 KB 对 2.5 KB 余量、全局停车 +190 KB/CTA L2 RAS 流量 ≈ +10%，双 NO-GO）或形状体制变更。出厂判决：**持有 r19 终字节**（dqkwg 0.514/1.73 ms @T2048/T8192，占 bwd 24–34%）。

---

## 7.3 wyb：$\mathrm{d}A$ 双侧掩蔽收缩与最终梯度

### 7.3.1 公式与相位

`gen_wyb`（L7585–8126，`GDN_WYB_DIET` 出厂臂）实现：

$$
\begin{aligned}
G &= \mathrm{stricttril}\big(\mathrm{d}v_p\,(\beta v)^\top+\mathrm{d}W\,(\beta k\,2^{gc})^\top\big) &&\text{P1/P2（逐项 select）}\\
X^\top &= T\,G^\top,\qquad Y=T^\top X &&\text{P3/P4（三明治）}\\
\mathrm{d}A_f &= -\,\mathrm{stricttril}\big(Y\odot 2^{gc_i-gc_j}\big) &&\text{P4 epilogue 折叠}\\
ra_i&=\textstyle\sum_j \mathrm{d}A_f{}_{ij}\,kkt_{ij},\qquad c1r_j=\sum_i \mathrm{d}A_f{}_{ij}\,kkt_{ij}\,\beta_i &&\text{P5（warp 蝶形）}\\
dk &= dk_{\mathrm{local}}+\mathrm{diag}(\beta)\,(\mathrm{d}A_f\,k)+\mathrm{d}A_f^\top(\beta k) &&\text{P6/P7（t2/t1 双链）}\\
TW &= T^\top \mathrm{d}W,\qquad rb_i=\textstyle\sum_k TW_{ik}\,k_{ik}\,2^{gc_i} &&\text{P8}\\
tv &= T^\top \mathrm{d}v_p,\qquad dv=\mathrm{diag}(\beta)\,tv,\quad rc_i=\sum_j tv_{ij}\,v_{ij} &&\text{P9}\\
d\beta &= ra+rb+rc;\qquad dg_i=\sum_{j\ge i}\big[(ra_j{+}rb_j)\beta_j-c1r_j+r3_j+r4_j-c4r_j\big]+dg_{\mathrm{state},i} &&\text{P10}
\end{aligned}
$$

其中 §2 的双侧掩蔽精确式 $\mathrm{d}A=-\mathrm{stricttril}(T^\top\,\mathrm{stricttril}(G)\,T^\top)$ 由 P3/P4 实现：内层掩蔽落在 $G$ 的**逐贡献** select 上（select 是线性的：逐项 select 之和 == 和的 select，精确——r13 记录），外层掩蔽与负号、衰减折叠进 P4 的 Y-epilogue。

### 7.3.2 G_dvp / G_dW 累积

**【机制】** P1：寄存器链 2-链 split mma，逐 vs 切片——A = dvp 自然定向分裂（f32 全局读入，`stage_split_ew`），B = **原始** $v$ 自然切片（`stage_raw_bf16`，精确）；$\beta_{\mathrm{col}}$ 折入末链 epilogue（r12 dqkwg TW 的搬法），tril-select 施于贡献。P2：dW 版本（B = 原始 $k$，尺度 $\beta\,2^{gc}_{\mathrm{col}}$ = `beta*expg` 折 epilogue），其末链 epilogue 对 sN 对做**读-加-重分裂**（同 lane 独占每元素：`cur = sNhi+sNlo; v = cur + sel; _split_store(v)`）。

**【r10 竞态教训（确定性门的历史）】** r10 之前 G 的累积循环携带一个 cp.async 味的跨 warp 竞态（与 bpre K-循环 restage 竞态同类）：同窗基线 dk 在 1.719e-3 与 4.719e-3 间跳变。修复 = r9 规则 3（每个 `wait_group` 后随 `gpu.barrier`）+ r10 规则（重暂存与读者间的尾栅栏，读者清空 ≥2 迭代后才可覆写）。此后 N=10 新进程逐级位同成为每轮结构变更的门径。代价 ~0.3% @T2048（r10）。

**【效果】** 并入 §7.3.6 的 2.40x；dk 数位在 r13 收紧的主因即 G 链的 tf32→split（dk 1.691e-3→1.284e-3 @T2048、1.724e-3→**5.131e-4** @T512，余量 6.2x/15.6x）。

### 7.3.3 X/Y 三明治的 GEMM 化：不物化 f32 的 epilogue split-store

**【动机】** 三明治若按朴素三步走需物化 $G$、$X$、$Y$ 三个 f32 $[C,C]$ tile（各 16 KB）+ 各自的 emap/store 扫描；wyb 的池预算（2-CTA bar 50 176 B）下不可行，且 fla 的 dtype 真相（A/h/dh/do=bf16）已是本链要避开的数值类。

**【机制】** 三条规则合成"epilogue split-store 越过死 B 源"：(1) **P3 的 X-链 epilogue 把 $X^\top$ 直接 split-store 进 sN 对**——恰是本链自己的 B 源（B 死于末链 `mma.sync` 之后）；mma 体与 epilogue 之间必须加 `gpu.barrier`（`mma_gemm` 的 `epi_bar=` 参数，r13 引入）：epilogue 的 `st.shared` 目标与本链 `ldmatrix` 仍跨 warp 读着的 tile 重叠（r6 危险类）。(2) **P4 的 Y-链 epilogue 折叠终值**：$-y_{ij}\,2^{gc_i-gc_j}$、tril-select、split-store 一步到位——$Y$ 与 $\mathrm{d}A_f$ 都不再单独存在。(3) A 侧 $T$ 对 P3 以自然定向分裂、对 P4/Y 与其后 tv/TW 以 $T^\top$ 定向**重驻**（一次 16 KB 全局重读；`aT` 的竞技场窗口）。K=64 链一律 k16 切片链接力（`accs=`，r12 wy 的片断生存期饮食）。

**【源码片段】**（L7867–7883，P3）：

```python
stage_split_ew(e, aTh, aTl, G["T_all"], ccty, m, C, 0, C, C)
for kb0 in range(0, C, 16):
    last = kb0 == C - 16
    accs = mma_gemm(e, f"xxh{kb0}", C, C, 16, W, aTh.name, aTh.ty,
                    sNhi.name, sNhi.ty, kt=True, accs=accs, k0=kb0)
    accs = mma_gemm(e, f"xxl{kb0}", C, C, 16, W, aTh.name, aTh.ty,
                    sNlo.name, sNlo.ty, kt=True, accs=accs, k0=kb0)
    accs = mma_gemm(e, f"xxm{kb0}", C, C, 16, W, aTl.name, aTl.ty,
                    sNhi.name, sNhi.ty, kt=True, accs=accs, k0=kb0,
                    epilogue=(... _split_store(..., sNhi, sNlo, row, col))
                    if last else None, epi_bar=last)
```

**【IR/PTX 对比】** 旧（r12 legacy）：tf32 `tgemm`（X/Y 位点，$2^{-11}$）+ f32 accCC 暂存 + emap/store_rows 扫描 + kahan 补偿体；新：bf16 `mma.sync` ×3/链（hi·hi+hi·lo+lo·hi）+ `ldmatrix` + epilogue 内 `truncf/extf/subf` 分裂对（`_split_store`）——**f32 三明治在任何时刻不驻 smem**。

**【效果】** 数值：$\mathrm{d}A_f$ 链收紧使 dbeta/dg 随 dk 同步收紧（§7.3.6 表）；dk/dbeta/dg 三个数位移动全部朝紧方向（r13）。计时并入 2.40x。

### 7.3.4 tv 3-mma：$T^\top$/dvp$^\top$ splits 进 tT/accCK 竞技场

**【动机】** $tv=T^\top\mathrm{d}v_p$ 是 r7 测得 ~158 µs 奖金的位点（tf32-tv 两次被拒：r3 8.6e-3/2.3x 余量、r7 9.3e-3/2.15x 余量，且 255 寄存器 16 B 栈溢出）——mma 速度下的标量精度只能走 3-mma split。r8 试图直建被 smem 封锁（9 8048 B 池下只剩 8.7 KB 自由，分裂 tile 需 23 KB）；r9 竞技场机器使其可解。

**【机制（r11）】** 两个生存期竞技场承接转置定向的分裂对：
- **tT → $[C,C{+}8]$ f32（分配 18 432；尺寸填充使其 malloc 唯一可识别）**：在 TW_rb 最后一次 tT 读之后暂存 `tThi/tTlo`（$[64/66,72]$ bf16，fp 9 216 各）；
- **accCK → $[C{+}2,C]$ f32（分配 16 896，fp 16 384）**：dk 循环之后承接 `dhi/dlo`（dvp$^\top$ 分裂）+ `tvacc`（$[66,16]$ f32）。

池 98 048→100 096 B（+2 048 是 T-分裂对的真实足迹，其余全免）；`smem_fix` 恰按设计消费 spec：22 malloc→17 声明、一条 18 432 竞技场。mma 本体：`mma_gemm_mh`，M=C、N=BV=16、K=C、W=8——A 分裂自 `T_all` 一次暂存，B 逐 vs 切片自 `dvp_all`（转置定向暂存）；**fp32 类累加**（hi·hi+hi·lo+lo·hi）。**dv 折入 epilogue**：每 acc lane 同时存 `tvacc`（dbeta 的 rc 行和用）与 $\beta$ 尺度化的 dv 全局元素——旧 emap/store_rows 扫描退役。细节：填充过的 $[C,C{+}8]$ tT 不能走 16B 突发 astage（1152 块 % 256 线程 ≠ 0）——`stage_rows` 的 `shape=` 参数按界元素级暂存一次。

**【出厂臂中的位置】** r13 全饮食端口后 tv 体原样迁入竞技场计划（P9）：A = dk 循环后重驻的 $T^\top$ 分裂（arena1/aT 窗口），dvp$^\top$ 分裂 + tvacc 在 arena2 的释放窗（`dhi@0/dlo@2304/tvacc@4608`）；链为 k16 切片 3-链接力（`mma_gemm_mh` + `k0=`）。

**【IR/PTX 对比】** r11：`mma.sync` 384→480、`ldmatrix` 0→128、tv 标量 gemm 的 8 条展开 K=64 循环消失；寄存器 239→226（0 spill）。

**【效果】**（r11，配对）：wyb 0.9871→0.8174（**1.21x**）@T2048 / 0.1253→0.0989（**1.27x**）@T512——r7 测得的 ~158 µs 全额落袋；单独链 e2e 5.237→5.112（1.024x med）。数值：dv 3.3e-6→1.088e-5 @T2048（split-bf16 tv 叠上 split-bf16 dvp 的"更紧乘积、多一层重组合"类；余量仍 736x/851x 对 8e-3 bar）。

### 7.3.5 ra / c1r：warp 蝶形和

**【机制】** P5 的两个 $[C,C]$ 归约（$ra$ 行和、$c1r$ 列和）用 `warp_rowsum/warp_colsum`：warp $w$ 持行/列 $\{w, w{+}N_W,\dots\}$，lane $l$ 供 $j=l$ 与 $l{+}32$（CW≤64），每行/列一棵 5 步 shuffle 树、lane 0 落盘。被归约项从 sN 对**重建**（$hi{+}lo$，f32 内精确）后乘 `kkt`——kkt **直接 L2 热读全局**（`kkt_g`，`kkt_all` 刚被 wy 流过；accCC 暂存退役）；$c1r$ 追乘 $\beta_i$。r8 引入时的对照项：串行单线程 $C$ 链（每行 CW 依赖迭代）消失，`gpu.shuffle xor` → `shfl.sync.bfly.b32` 探针验证。

**【效果】** 计入 2.40x；dk/dbeta/dg 数位随 r13 重掷收紧（蝶形树上再重排，属值类变更、6/6 门径放行）。

### 7.3.6 七位点 tf32→split-bf16 全清单与 128-寄存器双 CTA

**【清单】** r13 的 `GDN_WYB_DIET` 把 r8 点数的全部 tf32/kahan gemm 位点改写为 2–3 链 split-bf16 mma——**七个位点**：

| # | 位点 | 形状 | 链形 | 尺度折叠 |
|---|---|---|---|---|
| 1 | `G_dvp` | $C\times C\times V$ | 2-链 | $\beta_{\mathrm{col}}$ 折 epilogue |
| 2 | `G_dW` | $C\times C\times K$ | 2-链 | $\beta 2^{gc}_{\mathrm{col}}$ 折 epilogue |
| 3 | `X`（P3） | $C\times C\times C$ | 3-链 k16 切片 | epilogue split-store 越死 B 源 |
| 4 | `Y`（P4） | $C\times C\times C$ | 3-链 k16 切片 | $-2^{gc_i-gc_j}$ + tril + split 折 epilogue |
| 5 | `dk_t2`（P7） | $C\times K\times C$ | 2-链（A = `_resplit_pair_T` 带 $\beta_r$） | 收缩下标尺度折分裂 |
| 6 | `dk_t1`（P7） | $C\times K\times C$ | 2-链 | $\beta_{\mathrm{row}}$ 折 epilogue；终值 dk 三项折叠 |
| 7 | `TW_rb`（P8） | $C\times K\times C$ | 3-链 k16 切片 | $2^{gc}_{\mathrm{row}}$ 在 rowsum 项内 |

（tv 为 r11 已落地的第 8 个 mma 位点，非本清单的 tf32 退役项。）dk 循环的 k 特征切片 BKD=16 使 t2 对 + $k^\top$ tile 恰入 arena2 尾。`_resplit_pair_T`：既有 bf16 对的**转置重分裂**——$hi{+}lo$ 的 f32 重建精确（8+8 尾数位装得下），结果携带与直接分裂原 f32 相同的 $\sim2^{-17}$ 类；cvec 把 $\beta$ 折在源行 $s$ = 收缩下标上。**LLVM 将 t2 交接 store-to-load 前递入寄存器**（两链同 lane 映射）——两个 t2 malloc 从 PTX 消失；`gdn_smem_fix` 学会跳过 0-malloc 的别名 tile 而非关闭整个 spec（此前是硬 WARN + 121 KB 静态 smem ptxas 失败，r13 工具链教训）。

**【资源落点】** 池 **100 096 → 45 056 B**（arena1 = `aTh` 基 18 432；arena2 = `sNhi` 基 24 832；向量 7×256）= **2 CTA/SM × 256 线程**（原 1 CTA、226 寄存器）；寄存器**恰 128**（2×256×128 = 65 536 = 寄存器堆分毫不差）；176 B spill 存/228 B 载——**保留**（132 寄存器变体即失 2-CTA 驻留）。`cp.async` 92→0；`ld.shared` 2 607→（饮食载荷，全部重暂存类收缩）。

**【效果】**（r13，配对）：wyb 0.8195→0.3419（**2.40x**）@T2048 / 0.0995→0.0599（**1.66x**）@T512；单独链 e2e 3.322→2.847（1.167x med）；整轮 bwd e2e 3.326→2.688（**1.237x med / 1.240x min**）。数值（六余量全列，r13 表）：

| 形状 | dq | dk | dv | dbeta | dg | dh0 |
|---|---|---|---|---|---|---|
| T2048 | 1.824e-4 (44x) | **1.284e-3 (6.2x)** | 1.088e-5 (736x) | **8.527e-4 (23x)** | **3.274e-4 (61x)** | 2.169e-7 |
| T512 | 4.617e-4 (17x) | **5.131e-4 (15.6x)** | 9.43e-6 (848x) | **2.178e-4 (92x)** | **1.466e-4 (137x)** | 1.394e-7 |

粗体 = 七位点退役带来的收紧方向移动（dk 的七个 tf32 $2^{-11}$ 位点 → $2^{-17}$ split；dbeta/dg 随 ra/c1r 蝶形重掷 + 更紧的 $\mathrm{d}A_f$ 同步收紧）。

**【残项（诚实负结果）】** r17 FPIPE 的 P1/P2 首切片 DMA 配对中性（1.000x/0.999x）——源是 dqkwg 刚写的 `dvp_all/dW_all`，L2 驻留（保留，同 §7.1.7 理由）；r18 的 P7 `kTt` 双缓冲被寄存器上限封死（wyb 出厂即 `.maxnreg` 帽 128 寄存器、48/44/72 B spill；DWF 臂刚示范了帽下加暂存机器使 spill 类增长快于 L2 延迟隐藏的收益；双缓冲差 ~5 KB 池或先做寄存器饮食——r18/r19 排名遗留，未做）。

---

## 7.4 竞技场编排总论：峰值生存期审计方法论

### 7.4.1 机制：生成器—PTX 后处理的编译期契约

MLIR 的 SSA 合法性要求每个 tile 一个 `memref.alloc`；竞争性地复用同一块 smem 不能在 MLIR 层表达（会破坏支配性/活性分析），也不必要——**别名下放到 PTX 级**。生成器在 emit 时把生存期互斥的 tile 组织为**竞技场规约**（`smem_alias.json`，`main()` L8655–8661），构建期由 `tools/gdn_smem_fix.py` 的 fix-1 扩展消费：**被别名 tile 的 malloc 被删除，其 `cvta` 地址换算改写为 `arena_base_symbol + add.u64 offset`**（r9）。这一"调度决策 = 编译期工件"的形态是全文 Schedule-as-IR 论点在存储维度的体现：竞技场编排可审计、可再生、可 diff。

### 7.4.2 识别键与填充规则

- **malloc 以唯一分配尺寸识别**——LLVM 会重排 malloc 调用序列，次序不是稳定键（r9）。因此每个被别名的 tile 带**识别填充**：ldmatrix 喂送的 tile 加**从不寻址的额外行**（行跨步须保持 16 B 倍数，bf16 列数 %8）；纯 generic f32 tile 加 4 列。任何尺寸失配，spec **自禁用**（正确、不重叠的保守回退）。
- **足迹（footprint）≠ 分配尺寸**：竞技场算术用"行跨步 × 被寻址行数"（pad 列在行跨步内被 ldmatrix lane 地址跨过，pad 行不计）。
- **128 B 对齐断言**（ldmatrix + smem_fix 的 `.align 128` 声明）；`off + fp ≤ arena` 容量断言在生成期强制（`assert off + fp_of[t] <= spec["arenas"][base]`）。
- **竞技场基地必须是首用 tile**：纯锚点 alloc 会被 LLVM DCE，尺寸键随即找到 0 malloc（r13 教训 1）；LLVM 亦会把同 lane 的 epilogue 交接 store-to-load 前递入寄存器（t2 对）——此类 0-malloc tile **跳过并注记**，不再使整个 spec 失效（r13 教训 2）。

### 7.4.3 三条硬规则（验证门反复付费买来的）

1. **分裂永不别名其自身源 tile**（r12 的 accCC 自别名：分裂读 accCC 而分裂对又落 accCC 字节 → 自毁；dq/dk 1e0 类失败，npz diff 一步定位 dk_local）。
2. **`epi_bar`**：epilogue 的 `st.shared` 目标与本链 `ldmatrix` 仍跨 warp 读着的 tile 重叠时，mma 体与 epilogue 之间必须 `gpu.barrier`（r13，`mma_gemm epi_bar=`）。
3. **cp.async 两律**：`wait_group` 是**每线程**的——每个 wait 后随栅栏（r9 的 dk_local 列 0..5 陈旧竞态，256 线程网格才显形）；raw cp.async 落字节只进**纯 `ld.shared` 读者**的 tile，ldmatrix 操作数必须 `st.shared` 写入（r6 静默损坏； laundering/分裂重读为合规实现）。

### 7.4.4 峰值生存期审计（三 kernel 汇总表）

审计方法：把 kernel 划分为栅栏定界的相位，逐相列出 smem 活集（tile@偏移），验证每相活集字节 ≤ 竞技场容量、且相位边界全部有栅栏围栏；据此求出最小竞技场尺寸与池。出厂形态：

| kernel | 竞技场 | 宿主 malloc | 容量(B) | 叠放 tile（出厂偏移） | 池(B) | 驻留 |
|---|---|---|---|---|---|---|
| wy | arena1 | `kkt` $[C,C{+}8]$ f32 | 18 432 | `auhi@0/aulo@9216`（u 相）→`awhi@0/awlo@9216`（w 相） | **45 832** | 128thr×2 |
| wy | arena2 | `Tf` $[C,C]$ f32 | 16 384 | `vb@0`（u 相）/`kb@0`（w 相） | | |
| dqkwg | arena1 | `accCC` $[C,C]$ f32 | 34 816 | accCC@0；pthi@0+bdt@9216；dhi@16384+dlo@25600；pst@16384；bhh@13824+bhl@16384+akb@18944+accCK@24064；tTh2@0+tTl2@9216；dprT@0（FPIPE） | **47 616** | 256thr×2 |
| dqkwg | arena2 | `vnh` | 10 240 | vnh@0+vnl@5120；ptlo@0；krt@0/qrt@0；dWhi@0+dWlo@4608 | | |
| wyb | arena1 | `aTh` | 18 432 | aTh@0+aTl@9216 → dTh@0+dTl@9216 → dwAh@0+dwAl@5120+dwB@10240 →（FPIPE `wyT@0`） | **45 056** | 256thr×2 |
| wyb | arena2 | `sNhi` | 24 832 | sNhi@0+sNlo@9216 → dvAh/dvAl/dvB@0/3072/6144 → kTt@18432+t2h@20736+t2l@22784 → dWTh@0+dWTl@4608+TWt@9216+tKK2@17408 → dhi@0+dlo@2304+tvacc@4608 | | |

时间轴（dqkwg 出厂臂，每格 = 一个栅栏定界相位；█ = 活）：

```
              0    4.6K  9.2K  13.8K 16K   18.9K 24K   26.6K 32.2K 34.8K
P0 向量        ·     ·     ·     ·     ·     ·     ·     ·     ·     ·   （全向量在池标量区）
P2 dPr         █████████████████████ accCC ······akb·········        ← accCC 只在末 vs epilogue 写
P3a r4/c4r     █████████████████████ accCC ┃ pst┃                    （pst 先于 dhi/dlo 登场）
P3b dPm尾      █████████████████████ accCC ┃ dhi┃ dlo┃               ← 字节级满载（r19 pt 封锁依据）
P9 dvp         ┃pthi┃ ┃bdt┃ ┃bhh┃ ┃bhl┃ ┃akb┃  （ptlo 在 arena2）
P11 dW/dq1     ·     ·     ·     ┃bhh┃ ┃bhl┃ ┃akb┃ ┃accCK┃         （vnh/vnl 在 arena2）
P12 TW         ┃tTh2┃ ┃tTl2┃                                     （epi_tw 折叠，无 accCK）
```

### 7.4.5 方法论边界与负结果

- **峰值≠总和**：静态 smem 上限（sm_120a 实测 101 376 B；98 304 接受、131 072 拒绝）约束**峰值并集**，竞技场把峰值压到相位最大值；2-CTA bar ≈ 50 176 B/CTA（个别 kernel 的核算用 50 688 B）。寄存器侧同理：2×256×128 = 65 536 恰为寄存器堆——"恰 128"是设计点不是巧合（wyb 以 176 B spill 保它；dqkwg 以 `.maxnreg` 指令保它）。
- **别名不总是答案**：caryA 的四共居 tile（r11）没有任何合法别名——解法是"竞技场溶解饮食"（退役全部四 tile 换三个 bf16 mma tile，82 176→28 592 B）。竞技场与 tile 退役是同一谱系的两端。
- **pt/`GDN_DQKWG_TTF`/DWF 三数据点**（r18/r19/r22）划出谱系边界：帽定寄存器下，**任何**新增暂存机器使 spill 类增长快于 ~1–2 µs 一次性 L2 暴露的收益；_mid-phase FPIPE 族关闭_。重开该族的唯一门是自然寄存器饮食 ≥46（174→128），而无序保序重构触及它（r19 判决维持）。
- **验证门是最终裁判**：本章记录的每个自别名/竞态/填充错误均被 fp64 孪生 + 逐级 npz diff 一步定位（r6/r9/r12/r15/r17 各一例）；竞技场编排的合法性从不由构造单独宣称——位同证明（35/35、36/36 数组级）或 6/6 数值门是每轮放行条件。

---

## 7.5 小结

计算族三 kernel 的出厂终态（B1T2048H16，配对口径）：wy 0.229 ms（3.04x 于 r7 形，池 45 832 B、2 CTA）、dqkwg 0.514 ms（r8 基线 1.665 ms 的 3.2x；池 47 616 B、256thr×2 CTA、`.maxnreg 128`）、wyb 0.342 ms（对 r12 前体 2.40x；池 45 056 B、恰 128 寄存器×2 CTA）。三 kernel 合计从 r8 基线的 ~3.36 ms 降到 ~1.09 ms，是 bwd e2e 从 r8 窗口的 6.71/5.80（med/min）走向 ~2.20/2.15（r17 med / r22 共载 min，对 fla 0.479x）的主通道。方法论产出有二：**split-bf16 位点化剧本**（tf32 不可用处的 mma 速度-标量精度路线，cvec 尺度折叠 + epilogue 折叠 + epilogue split-store 三件套）与**峰值生存期竞技场**（编译期可审计的 smem 编排，配三条硬规则与验证门闭环）。诚实边界同样清楚：dqkwg 的 −46 寄存器门被分配器结构性关闭、wyb 的 P7 双缓冲被 128-寄存器帽封死、wy 的 shfl 扫描是全族唯一非位同数值变更——这些负结果连同其定价，构成对该体系"何处方为收益递减点"的完整测绘。
