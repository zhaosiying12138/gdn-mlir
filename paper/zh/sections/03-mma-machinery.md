# 3 MMA 机器与混合精度体系：两个生成器共享的张量核发射层

GDN 的前向（`tools/gdn_full_gen.py`，4464 行）与反向（`tools/gdn_bwd_gen.py`，8687 行）生成器都不把 GEMM 交给编译器的自动向量化或 transform 配方，而是各自内建一整套**张量核发射层（MMA machinery）**：从 lane 索引预计算、fragment 装载（`nvgpu.ldmatrix`）、warp 贴图、累加器链接，到 tf32 逃生舱、split-bf16 hi/lo 分解、staging 原语族（同步 st.shared 家族与异步 16B burst 家族）以及配套的 barrier 语义。本章逐个发射器、逐个机制地剖析这一层。历史脉络上，该机器最早在前向生成器中成形并经 GPU 全量验证（G4 门 20/20），随后以"fragment/lane 数学不变、发射器外壳适配"的方式移植入反向生成器（`gdn_bwd_gen.py` 中 `mma_gemm` 一节的移植注释即为此事实的原始记录）；两套实现因此共享同一套硬件布局律与同一条 hazards 纪律，本章以反向侧（更完备）为主线，前向孪生单列一节（§3.11）。

除非另行注明，源码引文出自 `tools/gdn_bwd_gen.py`（行号为写作时的 HEAD），硬件为 RTX 5090 Laptop（sm_120a，无 tcgen05，Ampere 风格 `mma.sync`），工具链为本仓库锁定的 2026-10 LLVM 主干 + ptxas-blackwell（PTX 8.7）。所有效果数字均注明证据文件路径。

## 3.0 为什么需要自建 MMA 机器：stock 路径的边界

自建发射层不是偏好，而是被 stock 路径的五条硬边界逼出来的。这些边界全部在构建期以探针或源码级证据钉死（evidence/G2/gen_report.md "Toolchain quirks & constraints found"、evidence/G5/bwd_report.md 与 fwd_report.md 的平台约束节）：

**边界一：`vector.transfer` 在 workgroup memref 上不 lower。** `vector.transfer_read/write` 作用于 `#gpu.address_space<workgroup>` 时，`gpu-lower-to-nvvm-pipeline` 的 LLVM 翻译直接失败（"conversion of memref memory space ... failed"）；对扁平全局的 `vector.transfer_write` 在 `scf.while` 内则残留 `unrealized_conversion_cast`。这意味着 stock 的向量化搬运通道在我们的共享内存 tile 上整体不可用，staging 只能走 `memref.load/store` 标量环或显式 `cp.async`。

**边界二：`linalg.matmul` 的多线程冗余与竞态。** `linalg-to-loops` 展开的 matmul 在**所有线程上冗余执行**；对链式累加 matmul，多个 warp 之间对同一累加缓冲形成写竞态；同一缓冲同时充当两个 matmul 操作数时彻底 mis-lower（`gdn_bwd_gen.py` 模块 docstring 的 de-risk 清单原文）。`transform.structured.vectorize` 又在共享内存 matmul 上硬失败（无 `failures(suppress)` 子句可用）。

**边界三（本章核心）：`transform.nvgpu.rewrite_matmul_as_mma_sync` 只接受单 warp atom matmul。** 该 transform 硬编码 `laneId = threadIdx.x`，仅覆盖 16×8×16 f16/f16/f16 与 16×8×4 tf32 两种单 warp 形态（evidence/G2/gen_report.md 约束 1 原文："accepts ONLY single-warp atom matmuls ... it hard-codes laneId = threadIdx.x. Bigger GDN tiles are emitted by our generator as explicit nvgpu.mma.sync fragments (same lane formulas from NVGPUTransformOps.cpp)"）。GDN 的 tile（如 64×64、64×32、C×C=64×64 的 kkt）需要多 warp 协作贴图，stock 配方无法表达；生成器必须自己持有一套以 `nvgpu.mma.sync` 原子 + `nvgpu.ldmatrix` fragment 装载为底座的发射器。

**边界四：tf32 `m16n8k8` 在表内路径不可达。** `nvgpu.mma.sync` 的 verifier 强制 8×8×128b 基本片，使 tf32 封顶 K=4；表列 `nvvm.mma.sync` 家族的 m16n8k8 只有 f16/bf16/i8。而 GDN 反向链的 f32 操作数 GEMM 恰恰需要 fla 收割到的 `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`（覆盖矩阵 T1 的主力 tf32 签名）。这逼出了全文预算 ≤5 个逃生舱中唯一真正使用的一个：`gdn.mma.tf32k8`（§3.5）。

**边界五：`create_async_groups` 的 wait_group-0 机器约束。** transform 的 `transform.nvgpu.create_async_groups` 在每个组创建后**立即**插入 `cp.async.wait_group 0`（本树 NVGPU/Transforms/CreateAsyncGroups.cpp 源码级验证："Create the group and wait for it right after"），使任何预取与计算的重叠在结构上不可能。prefetch 必须手工发射 `nvgpu.device_async_copy` 并自持 token 与等待位置（§3.8）。

这五条边界合起来划定了机器的分层（自底向上）：

```
原子层     mma.sync.m16n8k16.bf16 / mma.sync.m16n8k8.tf32(inline_asm)
           ldmatrix.x4 / ldmatrix.x2(.trans) / cp.async(16B, bypassL1) / bar.sync
发射器     sgemm(标量基线) · tgemm(tf32) · mma_gemm(条带贴图)
           · mma_gemm_mh(二维贴图) · rgemm/rgemm_aw/aggemm(寄存器累加)
组合子     mma_chain_if(受卫链) · accs= 累加器链接 · k0= K 分块 · epilogue 回调
           · epi_bar / bar= 栅栏语义
数值策略   tf32 逐位点门 · split-bf16 hi/lo(3-mma / 2-mma / cvec 折入)
staging    st.shared 家族(stage_rows/raw_*/split_*) · 异步家族(astage_*)
           · launder_pass(恒等重写) · +8 列 bank pad
```

**记号（先定义后用）。** 对一个 32 线程的 warp，记 lane 编号 $L\in[0,32)$，并分解为
$$g=\lfloor L/4\rfloor\ \text{（groupID，组号）},\qquad t=L\bmod 4\ \text{（threadID-in-group，组内线程号）}.$$
`emit_mma_lanes`（`gdn_bwd_gen.py:1705`，前向孪生 `emit_lane_precompute`，`gdn_full_gen.py:144`）预计算的派生量即 $2t$、$2t{+}1$、$g{+}8$ 等：

```python
def emit_mma_lanes(e, tid=None):
    # tid= 让内核从 GROUP-LOCAL tid 派生 mma lane/warp 变量（fwdscan v4 的
    # warp-group 半体：256 线程 CTA 的 warp 0-3 / 4-7 各自贴自己半区的 gemm；
    # %mW 来自 %gtid 保证每组的贴图仍是 0..3）
    t = "%tid" if tid is None else tid
    e.l(f"%mW = arith.divsi {t}, %c32 : index")        # warp id
    e.l(f"%mL = arith.remsi {t}, %c32 : index")        # lane id  L
    e.l(f"%mlz = arith.cmpi eq, %mL, {e.ic(0)} : index")
    e.l("%mLG = arith.divsi %mL, %c4 : index")         # g   = L>>2
    e.l("%mTIG = arith.remsi %mL, %c4 : index")        # t   = L&3
    e.l("%mT2 = arith.muli %mTIG, %c2 : index")        # 2t
    e.l("%mT2P1 = arith.addi %mT2, %c1 : index")       # 2t+1
    e.l("%mLG8 = arith.addi %mLG, %c8 : index")        # g+8
    e.l("%mT2X8 = arith.addi %mT2, %c8 : index")       # 2t+8   (k 维)
    e.l("%mT2P1X8 = arith.addi %mT2P1, %c8 : index")   # 2t+1+8 (k 维)
```

所有 fragment 布局公式都以 $(g,t)$ 表达。tile 的 K 维原子步长记 `kb`（bf16 k16 原子为 16，tf32 k8 原子为 8）；"行半（row-half）"指 M 维的 $\{m,\ m{+}8\}$ 两半，"k 八元组（k-octet）"指 K 维的 8 列块。

## 3.1 标量基线 `sgemm` 与站点派发器 `gemm`

**动机（算子层面）。** 反向链中有三类 GEMM 不适合或不必要走张量核：(i) 数值上 tf32 被逐位点门否决、split-bf16 又暂不可行的位点（如 `dhu` 家族、`dqkwg.dvp_bh`——后者的 tf32 化会使 dbeta/dv 随 dh 的任意 f32 级扰动重采样量化桶，跨三个数值等价构建在 1.5e-5→2.9e-3→5.8e-3 间混沌游走，evidence/G5/bwd_report.md G5-J 节）；(ii) 形状过窄、8-warp mma 贴图不下的位点；(iii) 作为性能与数值对照组的基线。为此机器保留一个线程分区的标量 GEMM 基线。

**源码（`tools/gdn_bwd_gen.py:589`）。** `sgemm` 的分区契约是全机器单写者纪律的源头：

```python
def sgemm(e, out, a, b, M, N, Kd, *, at=False, bt=False, acc=False,
          neg=False, a_off=(0, 0), b_off=(0, 0), o_off=(0, 0), bar=True,
          kahan=False):
    """out(+off) (+)= a x b -- thread-partitioned; element e of the MxN tile
    is owned by thread e %% NT for EVERY gemm, so chained accumulation
    re-reads the writer's own cell: race-free and deterministic."""
    e.l(f"scf.for {ge} = %tid to {e.ic(M * N)} step %nt {{")
```

输出元素 $e=iN{+}j$ 由线程集 $\{e \bmod NT + c\cdot NT\}$ 中恰一线程拥有；**每个**累加进同一 acc 的 gemm 使用同一分区，链式累加永远重读自己写过的 cell——跨 warp 无竞态、跨 launch 确定。`at/bt` 以索引换位表达转置（`at=True` 时逻辑 `a[i,kk] = a[kk+o0, i+o1]`），`neg` 只取负 B 操作数（保持 acc 初值符号语义），`kahan=True` 启动补偿求和内层环（dA 链的 64–128 项 f32 乘积和有重度消减，朴素链式在 dW 上留下 ~4e-4..1.4e-3 绝对噪声，被 T 缩并链放大越过 dbeta/dg 门；evidence/G5/bwd_report.md G5-E 节）。非 kahan 体内可显式发射 `math.fma`（`fma3`，单舍入、严格紧于 mulf+addf；本工具链从不自动收缩 `arith.mulf/addf`，probe 验证 `math.fma` 确实 lower 为 `fma.rn.f32`）。

**派发器 `gemm`（`gdn_bwd_gen.py:2145`）。** 一行式的站点开关：站点 tag 在 `TF32_SITES` 集合内走 `tgemm`（warp tile），否则 `sgemm`。`--tf32` 命令行按 `TF32_GROUPS` 分组启停，使逐组数值验证成为流程的一部分。

**IR/PTX 对比。** 标量路径在 SASS 上是 39 指令/MAC 级的 LDG/LDS+FADD 链（覆盖矩阵的 nomma 路径测量，evidence/G6）；同一 tile 换到 mma 路径，PTX 从 276 KB 缩到 88 KB（3.1×）、SASS HMMA 48→0 的反向实验也给出了标量环的代价量级（evidence/G2/gen_report.md 矩阵表：`mma` on→off 一行）。

**效果数字。** 标量基线承载了全部 6/6 数值门的参照值类；其 f32 逐位语义（固定求和顺序、可选 kahan）是后文所有"位级一致"断言的对照面。

## 3.2 `mma_gemm`：基础发射器（条带贴图）

### 3.2.1 契约与 warp 贴图

**动机。** 反向链里最大的 GEMM 族是 `k·k^T`/`q·k^T` 形状的 [C,C]、[C,BK]、[C,BV] tile（C=64 为 chunk 长度）。bf16 输入 + f32 累加正好落在 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` 原子上（覆盖矩阵 T1 的主力签名，fla 22/22 配置同款）；需要的是把任意 $M\times N\times K$（$M\equiv0\pmod{16}$，$N\equiv0\pmod{8W}$，$K\equiv0\pmod{16}$）的 tile 划给 W 个 warp 并发射 fragment 装载与原子链。

**源码（`tools/gdn_bwd_gen.py:1726`）。** 条带贴图：`wn = N // W`，warp $w$ 拥有 N 条带 $[w\cdot wn,(w{+}1)\cdot wn)$，条带内再按 8 列一个 atom：

```python
wn = N // W
e.l(f"%wnoff_{name} = arith.muli %mW, {e.ic(wn)} : index")
for mi in range(M // 16):
    e.l(f"%rA_{name}_{mi} = arith.addi %mLG, {e.ic(mi * 16)} : index")   # 行 m+g
    e.l(f"%rAh_{name}_{mi} = arith.addi %mLG8, {e.ic(mi * 16)} : index") # 行 m+g+8
for ni in range(wn // 8):
    e.l(f"%nc_{name}_{ni} = arith.addi %wnoff_{name}, {e.ic(ni * 8)} : index")
    e.l(f"%nct_{name}_{ni} = arith.addi %nc_{name}_{ni}, %mT2 : index")   # 列 n+2t
    e.l(f"%nct1_{name}_{ni} = arith.addi %nc_{name}_{ni}, %mT2P1 : index")
```

每个 warp 对每个 (mi, ni) atom 持有一个 `vector<2x2xf32>` 累加器；`k0=` 参数支持把一个宽 K 链拆成若干 K=16/32 的链节（§3.2.4）。

### 3.2.2 fragment 装载：`nvgpu.ldmatrix` 的象限语义

A fragment（16×16 bf16，每 lane 4 个 32 位寄存器、8 个元素）用 `ldmatrix.x4` 一次装载；B fragment（8 列条带 × 16 k，`kt=True` 即 tile 行为 n、k 连续的转置存放）用 `ldmatrix.x2`。每 lane 提供一个 16B 行地址，象限序为 **[TL, BL, TR, BR]**——这是 G4 锚（`ref_ldmatrix` 配置）钉下的"G4 anchor semantics"：

```python
# ldmatrix per-lane quadrant indices (G4 anchor semantics)
e.l(f"%lq7_{name} = arith.remsi %mL, {e.ic(8)} : index")    # lane%8
e.l(f"%lq8_{name} = arith.divsi %mL, {e.ic(8)} : index")    # lane/8
e.l(f"%lqh_{name} = arith.remsi %lq8_{name}, {e.ic(2)} : index")
e.l(f"%lqt_{name} = arith.divsi %mL, {e.ic(16)} : index")   # lane/16
e.l(f"%lh8_{name} = arith.muli %lqh_{name}, {e.ic(8)} : index")
e.l(f"%lt8_{name} = arith.muli %lqt_{name}, {e.ic(8)} : index")
e.l(f"%lb8_{name} = arith.muli %lq8_{name}, {e.ic(8)} : index")
...
        r  = mi*16 + %lh8 + %lq7      # A: lanes 0-7 行 0-7(TL) 8-15 行 8-15(BL)
        c  = kb + %lt8                #    lanes 16-23 (TR) 24-31 (BR) 列 +8
        e.l(f"%fa_{name}_{mi}_{kb} = nvgpu.ldmatrix {a_src}[{r2}, {c}] "
            f"numTiles = 4 transpose = false : {a_ty} -> vector<4x2xbf16>")
...
        r  = %nc_{name}_{ni} + %lq7   # B (kt=True): 行 n + lane%8
        c  = kb + %lb8                # 列 kb + 8*(lane/8)
        e.l(f"%fb_{name}_{ni}_{kb} = nvgpu.ldmatrix {b_src}[{r}, {c}] "
            f"numTiles = 2 transpose = false : {b_ty} -> vector<2x2xbf16>")
```

这里有一个容易写错、且曾真实写错过的点：**A 的 b16 象限 lane 算术不含 `%mLG`**——`%mLG`（=g）是 f32 C/D fragment 的 lane 偏移，只出现在 epilogue；`mma_gemm_mh` 的注释明确记录了这一区分（"NO mLG -- that is the f32 C-fragment lane offset used by the epilogue only; the b16 quadrant lane math is separate"）。落到 PTX：

```ptx
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r5, %r6, %r7, %r8}, [%rd125];
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r9, %r10}, [%rd126];
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {%r19, %r20, %r21, %r22}, {...}, {...}, {...};
```

（evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/kernel.ptx 与 gen/ref_ldmatrix/kernel.ptx 真实摘录；SASS 侧对应 `HMMA.16816.F32.BF16 R112, R108, R176, R112`，PTX 计数与 SASS HMMA 计数在全部 G2 配置 1:1 相等。）

`ldmatrix` 的价值是直接可量化的：一个 x4 装载替代 8 条 lane 索引化的 `ld.shared.b16`（G2 阶段机器即用后者，见 `frag_build`，§3.11.1），SASS 上 LDSM 一条对八条 LDS。反向链的 dqkwg 在引入 ldmatrix 化的重构后 SASS 计数为 LDS 4440→1378（−69%）、LDSM 192→896、HMMA 832→1280（evidence/G5/bwd_report.md G5-O 节）。

### 3.2.3 A/B/C/D 的寄存器序与 G4 的一行修复

PTX m16n8k16 布局律（硬件规定，G4 以整型探针在真机上精确裁定）：

| fragment | 每 lane 元素 | 公式 |
|---|---|---|
| A（bf16，4 寄存器 8 元素，按 [TL,BL,TR,BR] 象限 pack 列对） | $a_0{=}(g,2t)$, $a_1{=}(g{+}8,2t)$, $a_2{=}(g,2t{+}1)$, $a_3{=}(g{+}8,2t{+}1)$，再平移 $k{+}8$ 得后四元素 | 行半与 k 八元组配对 |
| B（bf16，2 寄存器 4 元素） | $b_0{=}(2t,g)$, $b_1{=}(2t{+}1,g)$（[k,n] 坐标），k+8 平移得后两元素 | **每 lane 的 B 寄存器属于列 $n=g$（groupID）** |
| C/D（f32，`vector<2x2xf32>`） | $c_0{=}(g,2t)$, $c_1{=}(g,2t{+}1)$, $c_2{=}(g{+}8,2t)$, $c_3{=}(g{+}8,2t{+}1)$ | **每 lane 的 D 元素属于列 $n=2t$ 与 $2t{+}1$** |

`mma_gemm` 的 epilogue 恰按 D 序回填（`ei∈[0,4)`：行 = `rA`/`rAh`，列 = `nct`/`nct1`，`mma_extract(e, acc, ei//2, ei%2)`）。

**B 的 N 索引是 $g$，D 的 N 索引是 $2t$——这一区别就是 G4 门抓出的核心 bug。** 早期 `frag_epilogue` 把累加器偶元素存在列 `ncol + %g`（lane 的 **B-fragment** N 索引）而非 `ncol + 2t`（**D-fragment** N 索引）。后果完全可预测：同一 $g$、不同 $t$ 的两个 lane 写同一列（冲突值），而 $2t{+}1$ 类奇数列永远无人写——kkt 只覆盖 1168/2080 个严格下三角 cell，o 阶段 2304/4096。修复是**一行语义修复**（新建 `%ncolt{ni} = ncol + %t2` 用于偶元素，`frag_precompute`/`ws_frag_precompute` 各加一个预计算），修复后每个 mma 配置写满 2080/2080 个 tril cell，且对 fp64 孪生的最大偏差 9.77e-4 **恰等于** numpy 对"f32 累加 + bf16 单次量化"管线的精确预测 9.765625e-4（测量 == 预测，evidence/G4/GATE_REPORT.md §2.1）。该 bug 影响 20 个配置中的 11 个（全部 mma=on 路径含 ws/tma）；这是"编译产物与手写 kernel 同构"叙事下第一个位级证据链的锚点，也是后文一切 D 序公式被无条件信任前必须先讲的故事。

### 3.2.4 累加器链接、`k0=` K 分块、epilogue 与栅栏语义

**累加器链接（`accs=`）。** `mma_gemm` 返回 `{(mi,ni): %acc}` 字典；下次调用传 `accs=` 即在寄存器内续累。这是 split-bf16 的 hh/hl/lh 三链节、dkb 的跨 vt 折叠、caryA 的跨 j0 块复用的公共底座——全程不落共享内存。

**`k0=` K 分块。** 一条 `Kred=64` 的链可拆成两条 `K=32` 的链节（`k0=0/32`）共用同一宽 tile，产生**完全相同的 mma.sync 序列**、但 ldmatrix fragment 的活跃区间停留在 K=32 链节级别——ptxas 寄存器压力不随总 K 增长。dqkwg 的 K=64 位点按此分块后（`GDN_DQKWG_K32`），spill 从 88/84/120 B 降到 64/64/104 B，两形状 35/35+36/36 个数组逐位相等、6/6 数值位数与记录一致；代价是该处 1.00×（spill 本身不是性能杠杆，数据点留档，evidence/G5/bwd_report.md G5-AA 节）。`_ops_g`/`_mh_g`/`_gemm_g`（`gdn_bwd_gen.py:5937-5972`）把"KGRAN 分块 + 尾链节挂 epilogue"封装成三个通用 helper。

**epilogue 与 `epi_bar=`。** epilogue 是逐 (mi,ni,ei) 的回调，拿到 (row, col, acc SSA)。`epi_bar=True` 在 mma 体与 epilogue 之间插一道 `gpu.barrier`——当 epilogue 的 st.shared 目标与本链自身 ldmatrix 仍在跨 warp 读取的 tile 重叠时必须开启（wyb 的 split-epilogue-over-B-source 模式；这是与 §3.8.2 同族的跨界写读 hazards 类，evidence/G5/bwd_report.md G5-T 节）。

**栅栏语义总述。** 该发射层的栅栏经济学有三条明文规则（均以源码 docstring 为准）：

1. **跨 staging/mma 边界必须栅栏**：任何 all-thread staging sweep 与 warp-guarded mma 读写之间，一道 `gpu.barrier` 封区。
2. **不相交 tile 的连续 staging 免栅栏（`bar=False` 类）**：扫过互不相交 tile 的连续 staging sweep 之间免尾栅栏，首道 mma 读者之前的单一栅栏覆盖全组（`stage_raw_bf16`/`stage_raw_T` 的 `bar=` 参数，G5-T 节引入）；dqkwg 的 dw→dq1 背靠背链同理——第二条链自己的栅栏给全组封界。
3. **受卫区尾栅栏不可省**（§3.4）：idle warp 会冲进下一段 staging 覆盖 mma warp 还在读的操作数 tile。

**效果数字（本发射器的整机贡献）。** wy 的 kkt/P 链按 k16 切片化 + accs 链接重构后，与 u/w 尾部（§3.6）合计：wy 0.70→0.23 ms（3.04×），PTX 侧 `ld.shared` 832→49、`mma.sync` 256→384、`ldmatrix` 96→384、`cp.async` 20→0，共享内存池 90624→45832 B（2 CTAs/SM），寄存器 255→158/161、零 spill（evidence/G5/bwd_report.md G5-S 节）。

## 3.3 `mma_gemm_mh`：mi-half × ni-strip 二维 warp 映射（W≤16）

**动机。** 条带贴图把 W 上限钉死在 $N/8$：$N{=}32$ 时至多 4 个 warp。dqkwg 的 M=64、N=32 形状族若按条带贴图，只能在 `%wgv4` 受卫区内跑 4/8 个 warp——ncu 显示 **89% no-eligible cycles、每调度器约 2 个活跃 warp**，shared-scoreboard 停顿占 18.1 cyc/inst 的 49.9%（ldmatrix 延迟之外无事可发射；evidence/G5/bwd_report.md G5-O 节）。半台 SM 闲置纯属贴图几何问题。

**源码（`tools/gdn_bwd_gen.py:1821`）。** 二维映射把 **M 维也切半**：设 `npr = N//8`（N 条带内的 atom 列数）、`halves = W//npr`、`mpw = (M//16)//halves`，warp $w$ 拥有 ni-条带 $w \bmod npr$ 与 mi-块对 $(w // npr)$：

```python
npr = N // 8;  halves = W // npr;  mpw = (M // 16) // halves
assert (M // 16) == halves * mpw and W % npr == 0 and W <= 16
e.l(f"%mh_{name} = arith.divsi %mW, {e.ic(npr)} : index")
# mi-pair base in ROWS（块序乘 16 —— 首版误加"块数"到行偏移，warps 4-7 读到
# 行 {2,18} 而非 {32,48}，verify 即刻抓出）
e.l(f"%mhb_{name} = arith.muli %mh_{name}, {e.ic(mpw * 16)} : index")
e.l(f"%nx_{name} = arith.remsi %mW, {e.ic(npr)} : index")
e.l(f"%ncv_{name} = arith.muli %nx_{name}, {e.ic(8)} : index")
```

fragment/lane 数学与 `mma_gemm` 逐条相同（probe 验证的布局复用），B fragment 仍跨本 warp 的 mi 对复用——只是 warp→tile 的指派变了。W≤16 的意义：512 线程的 dqkwg 臂有 16 warp、halves 可到 4，映射对 halves 通用而 fragment 数学是 warp 局部的，无需其它改动（G5-T 节）。

**IR/结构对比。** 相比受卫条带方案，此映射删除了 `scf.if`-yield 区域整体：W=8 全 warp、无 guard、无跨区域累加器。寄存器从 255（60 B spill）降到 248（0 spill）。

**效果数字。** dqkwg 的 LDS 风暴击杀（−69%）正是靠它才不被 guard 空转吃掉；wyb 的 tv 3-mma（M=C=64, N=BV=16, K=C, W=8，即 npr=2 × 4 个 mi 半对）与 caryA 的 [BK2,BK2] 方块链都运行在这张地图上。首版实现的两处 bug（mi 基址块数/行数混淆；见上方注释）被 verify 门当场抓获，是"二维贴图几何必须 GPU 验证"的直接论据。

## 3.4 `mma_chain_if`：受卫链、跨区域累加器与尾栅栏

**动机。** N=32 位点在二维贴图出现之前运行在"4 warp 受卫区"里；链式 mma（split-bf16 的多链节、跨 K 块）需要把 fragment 累加器带出 `scf.if` 区域——**链节之间的 staging 是 all-thread 的，不能坐进 <W-warp 的受卫区内**。

**源码（`tools/gdn_bwd_gen.py:993`）。** 累加器作为 `scf.if` 的**结果值**跨越区域边界：

```python
npr = N // (8 * W);  nacc = (M // 16) * npr
tys = ", ".join(["vector<2x2xf32>"] * nacc)
res = [e.v("acr") for _ in range(nacc)]
e.l(f"{', '.join(res)} = scf.if {guard} -> ({tys}) {{")
    for i, (asrc, aty, bsrc, bty, k0) in enumerate(ops):
        accs = mma_gemm(e, f"{name}_{i}", M, N, Kred, W, asrc, aty, bsrc,
                        bty, kt=True, k0=k0,
                        epilogue=last_epi if i == len(ops) - 1 else None,
                        accs=accs)
    e.l(f"scf.yield {', '.join(ordered)} : {tys}")
e.l("} else {")
    ...零初始化 else 臂 yield...
e.l("}")
e.l("gpu.barrier")     # 尾栅栏：强制
```

三处语义要点都在 docstring 里成文：(i) 写只发生在受卫 warp 内——epilogue 单写者；(ii) **尾栅栏不可省**：没有它，idle warp 冲进下一段 staging sweep、覆盖 mma warp 仍在读的操作数 tile（早期 dvp 循环中的潜伏模式，在此关闭）；(iii) `bar=False` 用于下一区域只**读**相同 tile 的背靠背链（dw→dq1），第二链自身栅栏给组封界。`guard=None` 切换到 mi-half 二维贴图（全 8 warp、无区域），ops 元组允许 6 元形式携带每链节自己的 `klen`（K 分块链）。

**效果数字。** 该组合子是 dqkwg v3/v4 的 dvp/dW/dq1 链与 pdO-fold（pt = (P^T·scale·rie) hi/lo，单一 epilogue 因子 egc[row]=2^(gcl−gc_i) 同时回收 pdO 与 bh 两路）的载体；v3 整体 e2e 贡献见 §3.2.4/§3.3 的 SASS 计数。

## 3.5 `tgemm` 与 `gdn.mma.tf32k8`：tf32 k8 逃生舱

### 3.5.1 必要性与 inline_asm 形式

**动机。** 反向链的 f32 操作数 GEMM（u/w 尾、dA 链、dq/dk 尾、carry 合成）若走 bf16 mma 会把 f32 指数域截断、重新毒化 dA 链；tf32 保留 f32 指数域、仅把尾数降到 10 位，而数值裕度（当时 ~180× 于门）可以吸收——前提是能发射 `m16n8k8` 原子。如 §3.0 边界四所述，表内路径不可达，于是有了**逃生舱 #1**（全文 ≤5 预算中唯一使用的一个）：方言操作 `gdn.mma.tf32k8`，在 `gdn-lower-hatch` pass 里 lower 为直接 `llvm.inline_asm`（`dialect/lib/GdnHatch.cpp:170`）：

```cpp
static const char *kPtx =
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
    "{$10, $11, $12, $13};";
static const char *kConstraints =
    "=&f,=&f,=&f,=&f,f,f,f,f,f,f,f,f,f,f";
```

操作数序：输出在前——$0..$3 = D，$4..$7 = A，$8/$9 = B，$10..$13 = C；结果是 literal `struct<(f32 x4)>`，四次 `extractvalue`。落到 PTX（真实摘录，evidence/G5/scratch/r18b_l2_T8192/fwdscan/kernel_smemfix.ptx）：

```ptx
// begin inline asm
mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    {%r646, %r647, %r648, %r649}, {%r650, %r651, %r652, %r653},
    {%r654, %r655}, {%r656, %r657, %r658, %r659};
// end inline asm
```

### 3.5.2 early-clobber `=&f` 是强制项

**这是本机器最重的一条工具链发现。** 不用 `nvvm.inline_ptx`（BasicPtxBuilder 路径）的原因是其输出约束只能给普通 `"=f"`；而 SASS 级验证表明：在 `"=f"` 下，ptxas 的 inline-asm 寄存器分配**合法地**把 D 四元组映射到同一条 mma 的 A 四元组上（`HMMA R12, R12, R10, R16`）——输入在 asm 之后被视为死亡、又无 early-clobber 约束，张量核的读/写重叠随即损坏操作数。一个 K=16 双原子链的整型探针 **118/128 个 cell 失败**，k 项整体缺失、误差确定性复现。`"=&f"` 禁止输出/输入重叠后问题消失；D==C 的别名（fla 式原地累加）保持合法，但本实现仍让 D 完全独立。`GdnHatch.cpp` 的注释把这定为"对任何在 mma 上使用 BasicPtxBuilder 的人都成立的工具链发现"。

### 3.5.3 A 寄存器序：row-half 外层（硬件探针裁定）

tf32 k8 的 fragment 布局**不靠文档猜**，由 16×8×8 整型操作数探针（tf32 精确表示整数）在真机上钉死（D == A@B+C 精确相等；N(0,1) 操作数下偏差 4.2e-3，即真 tf32 语义、离截断模拟约 1 ulp）：

```
g = lane>>2, t = lane&3
A 16x8 [m,k]: a0=(g,2t)  a1=(g+8,2t)  a2=(g,2t+1)  a3=(g+8,2t+1)
B 8x8  [k,n]: b0=(2t,g)  b1=(2t+1,g)
D 16x8 [m,n]: d0=(g,2t)  d1=(g,2t+1)  d2=(g+8,2t)  d3=(g+8,2t+1)
```

关键差异：**A 的寄存器序是"行半外层、列对内层"**（$a_1$ 与 $a_0$ 同 k、行 +8），而 bf16 k16 的 A 是按 k 八元组配对的 packed-pair 序——把 bf16 k16 的直觉直接搬过来是错的，探针的 exp1 立刻暴露。tgemm 因此**不用 ldmatrix**（b16 装载模式套不上 32 位元素），按布局公式逐元素 `memref.load`（每原子每 lane 6 次：4 A + 2 B）。

### 3.5.4 `tgemm` 站点发射器

**源码（`tools/gdn_bwd_gen.py:2010`）。** 与 `sgemm` 逻辑契约完全同构（`at/bt/acc/neg/off` 语义一致；kahan 无意义——K 缩并在张量核/f32 累加链内部完成）。贴图为 atom 轮转：`ti = mW + i*W`、`mi = ti/npr`、`ni = ti%npr`，要求 `(M//16)*(N//8)` 是 warps 的整倍数。操作数装载（非转置 A 为例）：

```python
a0 = load2(e, a, abase[0], d["ac2"])     # (m+g,   kb+2t)
a1 = load2(e, a, abase[1], d["ac2"])     # (m+g+8, kb+2t)   <- 行半外层
a2 = load2(e, a, abase[0], d["ac2p1"])   # (m+g,   kb+2t+1)
a3 = load2(e, a, abase[1], d["ac2p1"])   # (m+g+8, kb+2t+1)
...
e.l(f"{n0}, {n1}, {n2}, {n3} = gdn.mma.tf32k8 {a0}, {a1}, {a2}, "
    f"{a3}, {b0}, {b1}, {accs[0]}, {accs[1]}, {accs[2]}, "
    f"{accs[3]} : f32 -> (f32, f32, f32, f32)")
```

两个发射器级陷阱（均被验证门抓出、均在 docstring 成文）：

1. **kb 偏移的 off-by-one**：`%mT2P1` 本身已是 $2t{+}1$，per-kb 偏移代码又加了一次 "+1"——每个 kb>0 的 fragment 全部错位（也是最初 wyb 非法地址的根因：读走出 smem）。定位手段是 **delta 基 GPU 诊断**（B=I 时奇数 k 列整体 +1 位移肉眼可辨），修复后 13 组合探针矩阵（at/bt/acc/neg/offsets × 真实 bwd 形状）全过 tf32 舍入地板。
2. **`warps=` 必须显式传**：wy 以 128 线程启动而全局默认 `THREADS//32=8`——轮转贴图把一半 tile 晾着没人写（垃圾 u/w → dk=30）。`tgemm` 因此要求与内核线程数不符的调用点显式传 `warps=`（wy 传 4）。

`neg` 的语义边界与 sgemm 一致：只对 B 操作数取负、acc 初值保持正号（`dhu.dh_w` 依赖 `out += -(w^T dvp)` 这一区别）。

### 3.5.5 逐位点数值门与性能

tf32 不是全开开关：24 个 f32-GEMM 位点逐一单独 tf32 化、在 T512 对每个梯度分量定级（evidence/G5/bwd_report.md G5-G 节表）。被否决的位点与数字：

| 位点 | 最差梯度位移 | 判定 |
|---|---|---|
| `dhu.dkB` | dk 1.04e-2 | **越 8e-3 门**（dH 操作数 ~O(10)）→ 保持标量 |
| `wy.u` | dk 1.06e-2 | **越门**（u 噪声经 S→Sn 放大）；同形同 K 的 `wy.w` 却干净（1.4e-5）→ tf32 |
| `wyb.tv` | dbeta 8.6e-3 | 仅 2.3× 裕度 → 标量 |
| `dhu.dvp_bh` | dbeta 3.0e-3 | 6.6×；且 dh 量化桶重采样混沌 → 标量（split-bf16 为文档化替代） |
| `dhu.dv_T` | dv 1.26e-3 | 6.4× → 标量 |
| `dhu.dvp_pdO/dh_qg/dh_w` | ≤3e-4 数值净 | **性能负**（见下）→ 标量 |
| `dqkwg.*` 8 位点 | dk 7.8e-4..5.2e-4 | tf32 |
| `wyb` 7/8 位点 | dk 6.6e-4..1.1e-4 | tf32 |

最终 16 位点 tf32；性能 14.16→11.06 ms（T2048）：dqkwg 1.71×、wyb 2.19×。两个值得记录的结构性发现：(i) `wy.w` 干净而 `wy.u` 有毒——同形状同 K，u 的误差坐状态扫描进入 Sn；(ii) tf32 的 dW/TW（dA 链、此前 kahan 保护）**什么都不动**——dW 量级小，kahan 保护的是 f32 累加噪声，mma 里同样没有。**性能负样本**：dhu 的三个数值干净位点全展开进串行反向 chunk 游走后，ptxas 128 regs/188 B spill → 255 regs/116-128 B spill，dhu 3.69→4.67 ms——tf32 全部回退，dhu 的杠杆是占用率与 retile，不是 mma。fwdscan 的复发 gemm 位点组（`scan`）经 bisect 后纳入默认（h_all 扰动 8e-4 rel / vnew 5.5e-5 rel 被既有最大值吸收，dk 裕度 4.5×；`cary` 组同理，dh0 裕度当时 4000×）。

## 3.6 split-bf16 hi/lo 体系

### 3.6.1 数学

对 f32 值 $x$ 定义
$$\mathrm{hi}=\mathrm{bf16}(x),\qquad \mathrm{lo}=\mathrm{bf16}(x-\mathrm{hi}),\qquad x=\mathrm{hi}+\mathrm{lo}+O(2^{-24}\cdot|x|).$$
hi 携带前 8 位尾数、lo 携带接下来的 ~8 位，`hi+lo` 在 f32 中**精确**可加（8+8 位尾数装得进 24 位）；残差 ~$2^{-17}$ 相对类。乘积分解：
$$xy=(\mathrm{hi}_x+\mathrm{lo}_x)(\mathrm{hi}_y+\mathrm{lo}_y)=\underbrace{\mathrm{hi}_x\mathrm{hi}_y}_{\text{mma 1}}+\underbrace{\mathrm{hi}_x\mathrm{lo}_y+\mathrm{lo}_x\mathrm{hi}_y}_{\text{mma 2, 3}}+\underbrace{\mathrm{lo}_x\mathrm{lo}_y}_{\approx 2^{-32},\ \text{舍弃}}.$$
三个 bf16 mma 以 f32 累加器链回 ~fp32 质量的乘积，速度是 mma 速度——这是 tf32 的**文档化替代**：tf32 在 dkb.dkB 把 dk 推到 1.04e-2（越 8e-3 门），split-bf16 的 ~$2^{-16}$ 乘积误差比它低约 30×（`_split_bf16` docstring，evidence/G5/bwd_report.md G5-J 节）。

**2-mma 型何时够。** 当一侧操作数本来就是**精确 bf16**（全局存储即 bf16、逐元素搬运无舍入），只需给另一侧分裂：
$$x\cdot y_{\text{raw}}=\mathrm{hi}_x y_{\text{raw}}+\mathrm{lo}_x y_{\text{raw}}\quad\Rightarrow\quad 2\ \text{条 mma}.$$
更强的形式是**行尺度折入 epilogue 的代数恒等式**：当缩放因子 $s_i$ 只依赖**输出行** $i$（不依赖缩并指标 $k$）时，
$$\sum_k \big(s_i\,k[i,kk]\big)\,dh[kk,j]=s_i\sum_k k[i,kk]\,dh[kk,j],$$
缩放因子完全移出缩并——A 操作数退化为 raw bf16 k 切片（元素级 staging，无分裂、无预乘 sweep），尺度落进 epilogue 的一次 mulf。dqkwg 的 `dvp_bh` 位点即此（$s_i=2^{gcl-gc_i}$）：每 (vs,ks) 位点从 3 mma 降到 2 mma，共享内存 99840→97792 B、寄存器 234→214、零 spill，dqkwg 1.71→1.66 ms；两个中间设计（直接 3-mma 搬运 2.19 ms、转置 tile dW/dq1 staging 1.93 ms）被测回退（evidence/G5/bwd_report.md G5-L 节）。wy 的 u/w 尾部同型：`u = T·(β⊙v)` 的列缩放 $\beta[k]$ 作用在缩并指标上、不能移出，于是折入 **A 侧 T 的分裂**（见 §3.6.4 的 cvec）。

### 3.6.2 3-mma 型的落点与数字

- **dkb**（`gen_dkb`，`gdn_bwd_gen.py:3652`）：`dkB = vng @ dh^T`，两侧 hi/lo 分裂（`_split_bf16`），hh/hl/lh 三链节以 `accs=` 链接、**vt 折叠也在 mma 累加器内完成**。dk 数值不变：1.730e-3 vs 标量 1.731e-3；dkb 0.96→0.25 ms（3.8×）。
- **bpre**（`GDN_BPRE_MMA`）：A = M 块 hi/lo 融合分裂（f32 源，~2^-16 乘积），B = dO^T **raw** bf16 tile（精确）、尺度在 epilogue、K=C 缩及寄存器链。SASS：**0 HMMA→128、LDS 160→0、LDSM 0→80**；[BK2,V] f32 bb staging tile 及其尺度 sweep 退役，smem 49152→36864 B（2 CTAs/SM）、寄存器 40→54；bpre 4.79×（最净窗口 5.9×）。
- **caryA**（`GDN_CARYA_MMA`）：bpre 式移植退役全部四块共栖 tile（82176 B）换三块 bf16 mma tile（28592 B），255→76 寄存器（双零 spill）→ 2 CTAs/SM；Bhat 因子经 `stage_split_T` 的 cvec 折入 A 分裂（§3.6.4）。tf32 版本 1.39× 的占用率天花板被打穿到 **5.00×**。重结合被测量而非空谈：对 tf32 版，A_all 在 8.39M/8.39M 个元素上位移、最大 **5.6e-22 绝对**（相对 2.8e-4——正是它替换掉的 tf32 截断类；A 超收缩故量级 ~1e-18），被 carry 链的舍入吸收在打印位数之下：**全部六个最终梯度位数不变**（evidence/G5/bwd_report.md G5-R 节）。
- **wyb 的 tv/TW/G/X/Y/dk_t1/dk_t2 全族**（`GDN_WYB_DIET`）：每个 tf32/kahan 位点变 2-3 链节 split mma；共享内存池 100096→45056 B、恰 128 寄存器 → 2 CTAs/SM；wyb 2.40×；数值**全线收紧**——七个 tf32 2^-11 位点降为 2^-17 分裂类：dk 1.691e-3→**1.284e-3**（6.2× 裕度）、dbeta→8.527e-4（23×）、dg→3.274e-4（61×）、dh0 2.761e-6→**2.169e-7**（36900×）（evidence/G5/bwd_report.md G5-T 节）。
- **dqkwg 的 tf32 尾巴**（`GDN_DQKWG_SB`）：dq_dPmk / dk_dPmq / TW 三处 tgemm 改 bf16 mma——dPm 自然分裂（dq 缩并其行）、**同一 arena tile 转置再分裂**（dk 缩并其列；两种朝向不能共享一次分裂但能顺序共享 tile）、T^T 分裂 + dW^T 分裂的 TW 三链节。tCC/tCKa/tCKb 及其 cp.async staging 退役：**cp.async 46→0、ld.shared 1234→71、stack 72 B→0**、寄存器 255→240；dq 4.910e-4→**1.824e-4**（裕度 7.5×→44×）（evidence/G5/bwd_report.md G5-S 节）。

### 3.6.3 分裂原语的 smem 源家族

分裂不总从全局行出发；有一族以**常驻 smem tile** 为源的变体，均保持"逐元素运算序列相同、值相同 → 位相同"的可判定的位级等价：

| 原语（行号） | 源→目标 | 语义 |
|---|---|---|
| `_split_bf16` (:1293) | smem f32 [R,C] → hi/lo | 两 sweep（hi 先行、lo 重读）；`cvec=` 列缩放（任意尺度） |
| `_split_bf16_T` (:1335) | smem f32，**转置**写出 | dk_dPmq 的 A=dPm^T 朝向；同 2^-17 类、同双 sweep |
| `_split_bf16_fused` (:1433) | 同上单 sweep | 装载一次同发 hi/lo（索引算术减半）；dqkwg dh 侧关键路径用 |
| `_split_T_smem` (:1366) | smem f32 源转置分裂 + `cvec[s]` | 消费 cp.async 落地的 raw 窗口（只 `ld.shared` 读，合规于 §3.8.2 规则） |
| `_resplit_pair_T` (:1396) | **既有 bf16 hi/lo 对** → 转置再分裂 + `cvec[s]` | 重构 `shi+slo` 在 f32 精确 → 与直接分裂 f32 同 2^-17 类；dk_t2 的 β[r] 折入 |
| `split_smem_ew` (:1671) | stage_split_ew 的 smem 源版 | 同元素同序同值 → 位相同分裂 |

一条以 verify 门换取的 arena 纪律成文于此（G5-S 节）：**分裂 tile 永不得别名自己的源 tile**——dPm 分裂曾叠放在 accCK 自己的字节上，自毁分裂源，dq/dk 1e0 级 FAIL 当场被抓。

### 3.6.4 `cvec=`/`vec=` 折入：幂二可交换性与任意尺度两类

把缩放因子折进分裂有两类代数性质不同的通道，源码 docstring 各自给出精确表述：

**幂二类（`stage_split_T` 的 `cvec=`，G5-R 节）。** 对 $2^d$ 缩放：
$$\mathrm{split}(v\cdot 2^d)=\mathrm{split}(v)\cdot 2^d$$
**精确成立**——$2^d$ 与 bf16 截断、与 hi/lo 残差均可交换（hi 与 lo 同步精确缩放，无新舍入）。因此把 caryA 的 Bhat 行尺度 $2^{g_{cl}-gc}$ 折入 A 分裂，能在**相同操作数值上逐位复现**旧的 B 侧缩放乘积。这是"改结合方式而不改任何值"的机器级表达：caryA 的 `bhb`（预乘 Bhat 的 B tile）退役为 raw-k 拷贝。

**任意尺度类（`stage_split_ew` 的 `cvec=`，G5-S 节）。** 对任意 $s$：乘积在 f32 中**先舍入一次**、再做标准 ~$2^{-17}$ 分裂——每项误差停留在分裂类内（非位精确，但与"先分裂再缩放"同误差类）。wy 的 u/w 用它把 $\beta[k]$（u）与 $\beta[k]2^{gc[k]}$（w，hoisted 成 bgc 向量）折进 T 的列缩放分裂，替代了 tf32 被门否决的 `wy.u`（1.06e-2 越门）——这正是 wy 3.04× 的数值前提。`vec=`（`[R,1]` 行向量）则预乘**输出行** j（pt 预缩放 $P^T\odot rie$ 一类的朝向）。

### 3.6.5 数值效应总账

split-bf16 对六分量梯度的净效应全部朝**收紧**方向：dk 在 dkb 位点不变（1.730e-3 vs 1.731e-3）、在 tf32 尾巴位点收紧 1.7e-3→1.28e-3；dv 因 tv 与 dvp 双双分裂而多一层重结合（3.3e-6→1.088e-5，裕度仍 736×）；dbeta/dg/dh0 随之收紧。与 tf32 的对照数字（dk 1.04e-2 越门 vs split 的 1.73e-3）是该体系立项的全部理由。

## 3.7 staging 原语族 I：同步 st.shared 家族

### 3.7.1 布局与 +8 列 bank pad

一切喂 `ldmatrix` 的 tile 遵守两条布局律：

1. **+8 列 bf16 pad**：行跨距必须保持 16 B 整倍（cp.async 16B 对齐），但**不得**是 128 B 整倍（前向 F2 的教训，§3.11.4）——bf16 tile 一律 `[R, C+8]`，行跨距如 64 列 → 128+16=144 B，8 个 ldmatrix 行轮转扫过全部 32 bank。pad 的摆放位置服从识别约束：ldmatrix 喂入 tile 的识别 pad 用额外**行**（永不寻址：r < R），纯通用 f32 运算的 tile 才用 +4 列（见下条与 arena 识别键）。
2. **pad 永不写永不读**（纯 bank 展开）；aliased tile 的 malloc 尺寸必须互相唯一——LLVM 会重排 malloc，**尺寸**才是 `gdn_smem_fix` 识别 arena 宿主的键（行 pad 而非列 pad 出现在需要保持 16 B 行对齐处）。

### 3.7.2 raw 家族：`stage_rows` / `stage_raw_ew` / `stage_raw_bf16` / `stage_raw_T`

**动机。** bf16 全局 → bf16 tile 的搬运是**精确**的（无转换、无舍入），因此它是 split-bf16 体系里"raw B 操作数"的标准供给通道；同时它写 tile 的方式（`memref.store` → st.shared）天然满足 §3.8.2 的 ldmatrix 合规规则。

```python
def stage_raw_T(e, dst, src_g, gty, m, cm, col0, R, S, bar=True):
    """TRANSPOSING raw bf16 stage:  dst[j, s] = g[m, col0 + s*cm + j]
    (tile [R, S+8]; source slice is S rows x R fast cols).  Reads
    coalesced (j fast within s); stores land row-strided (~8-way STS
    conflicts -- cheaper than the uncoalesced global alternative)."""
    e.l(f"scf.for %srt = %tid to {e.ic(R * S)} step %nt {{")
    j = %srt % R;  s = %srt // R        # 全局侧 j 连读（coalesced），
    v = gload_raw(e, src_g, ...)         # smem 侧行跨写（~8 路 STS 冲突
    e.l(f"memref.store {v}, {dst.name}[{j}, {s}] ...")
```

四个变体的分工：`stage_rows`（自然朝向一般形，`shape=` 参数支持有界子区 staging）；`stage_raw_ew`（element-wise RAW 同 dtype 切片，为太窄走不了 16B burst 分区的 tile——如 [C,16] bf16 = 128 块 < 256 线程）；`stage_raw_bf16`（RAW bf16 自然布局进**带 pad 的 mma tile**，两侧 coalesced）；`stage_raw_T`（转置 staging：全局行内 j 快连读、smem 侧行跨写，牺牲 ~8 路 STS 冲突换取全局侧 coalesced——比全局侧不 coalesced 的替代便宜）。`bar=` 参数实现 §3.2.4 的不相交 tile 免栅栏类。

### 3.7.3 `stage_split_ew` / `stage_split_T`：融合分裂 staging

**动机。** "先 stage 后分裂"要两遍索引算术；融合版从扁平全局行**一次读入、同时写 hi 与 lo**（索引算术减半），且是 split 体系里全局侧的标准入口：

```python
def stage_split_ew(e, hi, lo, src_g, gty, m, cm, col0, R, CW, cvec=None):
    """FUSED hi/lo split straight from a flattened global row ...
    G5-S: cvec -- a [CW,1] smem f32 pre-scaling each COLUMN c
    (hi/lo = split(v * cvec[c])); ARBITRARY scales (unlike stage_split_
    T's exact power-of-two cvec fold): the product rounds once in f32,
    then the standard ~2^-17 split residual."""
    v = gload(e, src_g, gty, [m, _gcol_int(e, i, j, cm, col0)])
    if cvec is not None:
        v = op2(e, "mulf", v, ldsc(e, cvec, j))
    e.l(f"%sph = arith.truncf {v} : f32 to bf16");  store -> hi
    hw = extf(%sph);  r = v - hw
    e.l(f"%spl = arith.truncf {r} : f32 to bf16");  store -> lo
```

`stage_split_T` 是其转置孪生（读 j 快连、写 [R, S+8] 的转置目标），带 `vec=`（输出行预乘）与 `cvec=`（输出列预乘，幂二精确类）两个折入通道。wy 的 u/w 相位即"同一 arena1 槽位先装 u 相位的 T 分裂、尾栅栏后重装 w 相位的 bgc 缩放分裂"——生命周期复用（lifetime arena）的标准玩法。

### 3.7.4 barrier 经济学回顾

同步家族的每次调用默认以 `gpu.barrier` 收尾；`bar=False` 的两个成文类（不相交 tile 连续 staging；背靠背读同 tile 的链）已在 §3.2.4 列为机器级规则。restage 纪律的另一半来自 fwdscan/dhs 的滚动窗口实践并被源码引用为"r10 规则"：**每块缓冲的下一次覆写之前必有尾栅栏、且复用距离 ≥2 次迭代**。

## 3.8 staging 原语族 II：异步 16B burst 家族与 LDGSTS→LDSM 危险

### 3.8.1 `astage_tile` / `agroup` / `await_all` / `astage_sync`

**动机。** §3.0 边界五禁止 transform 路径做预取；手工 `nvgpu.device_async_copy` 是唯一通道。`astage_tile`（`gdn_bwd_gen.py:1569`）发射**线程跨步的 16B burst**：

```python
def astage_tile(e, src, gty, m, cm, col0, dst, dst_col0, R, Cc, dt, toks, ...):
    """thread-strided MANUAL nvgpu.device_async_copy burst for an [R, Cc]
    tile of ONE global row ... in 16B chunks (8 bf16 / 4 f32 ...).
    NOT nvgpu.device_async_create_groups / the transform's create_async_
    groups: those insert `cp.async.wait_group 0` right after every group,
    which serializes any prefetch with the compute it should overlap.
    Emitting the ops directly leaves wait placement to the caller."""
    el = 8 if dt == "bf16" else 4
    for kk in range(tot // nthr):            # 静态展开：token 留在本作用域
        ...
        e.l(f"{t} = nvgpu.device_async_copy {src}[{m}, {gcol2}], "
            f"{dst.name}[{r}, {dcol}], {el} bypassL1 : {gty} to {dst.ty}")
        toks.append(t)
```

三个契约级细节（全部由 verify 门用真实代价换得，docstring 成文）：(i) **静态展开**，使 token 在调用方作用域存活、供其 `create_group`；(ii) `nthr` 必须等于启动内核的线程数（wy 128 vs 模块默认 256 的失配曾让缺失线程拥有的块**无人拷贝**、半陈旧 tile 入算）；(iii) `tid=` 覆盖属主线程变量（fwdscan v4 的 warp-group 半体以 `%gtid` 各自 stage 自己的 K 半切片，nthr=128）。`agroup`/`await_all` 封装 `create_group` 与 `numGroups=0` 等待；`astage_sync` 是 `stage_rows` 的即换即用替代（burst + 立即 drain + barrier）——值位相同、staging 指令数约 **4×** 缩减（dkb/dHv 模式，逐位一致验证）。PTX/SASS 真实形态：

```ptx
cp.async.cg.shared.global [%rd518], [%rd520], 16;
cp.async.wait_group 1;
```
```asm
LDGSTS.E.BYPASS.128 [R3], desc[UR10][R116.64] ;
```

跨 chunk 预取的进阶形态（dhs 的 `GDN_DHS_XP`）：最后一个窗口把 chunk c+1 的窗口 0 DMA 发进空缓冲，**cp.async group token 以 `!nvgpu.device.async.token` 类型坐 `scf.for` 的 iter_args** 跨迭代（该助记符是本 MLIR 构建的注册名；`async.token` 是解析错误），下一迭代 prologue 的 await 于是被整 chunk 计算覆盖；末 chunk 发一次无害自重复使 token 恒有效、循环后 drain 保证 CTA 退出时无在飞拷贝。值精确性以全链 npz diff **逐位相等**证明（evidence/G5/bwd_report.md G5-S 节）。

### 3.8.2 LDGSTS→LDSM 危险（机器级 hazards 规则）

**现象。** 用异步 burst stage 的 mma A tile（`akb [C, BK+8]` bf16，SASS `LDGSTS.E.BYPASS.128` 落地）被**通用 `ld.shared` 读回完全正确**（对全局源逐位相等，dump 验证），却被 `nvgpu.ldmatrix` **读错**——错的行块随构建调度而定（2-mma 构建错 32-63 行、3-mma 构建错 48-63 行）、逐构建确定、对值保持的刷新遍历不变。验证门以 dk 1.34 / dv 2.18 / dbeta 25.7 抓住（dq/dh0 因不依赖 dvp 而纹丝不动——错误面精确圈定 dvp 路径）；lane 真值 fragment dump 把错位定位到整块 mi 块。**同一 tile 换 plain st.shared 写入（ASPLIT bisect 臂）后，同一套 ldmatrix/mma/epilogue 机器对 numpy 逐位相等（5.2e-8）**（evidence/G5/scratch/r6_ldgsts_probe_fail / r6_ldgsts_probe_stshared_pass）。

**成文规则（源码注释中的 "r6 rule"）。** *每一个喂 `nvgpu.ldmatrix` 的 tile 必须是 st.shared 写入或元素级 staging 的——绝不许 raw LDGSTS→LDSM 直连*。raw 落地的 tile 只许喂通用 `ld.shared`（rgemm 窗口、tgemm 操作数皆此合规形态）。这是手动 cp.async 路径在本 sm_120a 工具链上的第二个文档化静默失效模式（第一个是构建早期 kdbg 记录的 burst 零交付——同一指令族、同样延迟等待特征）。

### 3.8.3 `launder_pass`：恒等 st.shared 重写

**动机与源码（`gdn_bwd_gen.py:1650`）。** FPIPE 化（预取直落 operand tile）与 r6 规则的交集需要一座桥：raw 落地 tile 在首个 ldmatrix 读者之前，做**恰好一遍**元素级 `ld.shared/st.shared` 恒等重写：

```python
def launder_pass(e, buf, R, Cc):
    """identity st.shared rewrite of a cp.async-landed tile region -- the
    r6 rule demands ldmatrix operands be st.shared-written (NEVER raw
    LDGSTS->LDSM), so a raw-landed tile gets ONE element-wise
    ld.shared/st.shared pass over exactly the staged cells before its
    first ldmatrix reader ... Values bit-identical by construction."""
    e.l(f"scf.for %lu = %tid to {e.ic(R * Cc)} step %nt {{")
    v = memref.load buf[i, j];  memref.store v -> buf[i, j]
```

值按构造逐位相同（存回未变的 v）；caryA 的 w 窗口旋转即其消费者——raw cp.async 窗口只喂通用读，ahi/alo 保持 st.shared 写入的 ldmatrix 源。

## 3.9 `rgemm` / `rgemm_aw` / `aggemm`：carry 链的寄存器累加 GEMM 族

**动机。** 反向 carry 链（`dhs`）每 chunk 一次 `Xa += A_all[m] @ X`；[K,K] f32 carry tile 若 staging 进 smem 要 64 KB——而该 gemm 的访问模式让 staging 毫无意义：一个 warp 的 lane 拥有同一输出行的连续 j，每 lane 对每 kk 发出**相同**的 `aglb[mrow, i*Kd+kk]` 装载（硬件广播、每 warp 每 kk 一次事务；第二+个消费者 warp 命中 L2）。`aggemm`（`gdn_bwd_gen.py:1518`）即此直接读全局的标量 gemm（分区/单写者与 sgemm 相同）。

`rgemm`（:1455）解决 `aggemm` 的真实瓶颈——cell 主序的 1 深依赖链让 LDG 延迟串行化（dhs 0.83 ms）。其分区把 ILP 变成结构属性：输出 cell $e=iN{+}j$ 属于线程 $t$ 当 $e\in\{t+c\cdot NT\}$（$CPC=MN/NT$ 个/线程）；当 $N=BV$ 且 $BV\mid NT$，同线程的全部 CPC 个 cell **共享** $j=t\bmod BV$——**每 kk 一次 b 装载服务全部累加器**，CPC 次 a 装载（互异行、warp 广播）是编译器可重叠的独立全局装载：ILP=CPC。

```python
NT = THREADS;  CPC = (M_ * N_) // NT
e.l("%rgj = arith.remsi %tid, " + e.ic(N_) + " : index")
e.l("%rgi0 = arith.divsi %tid, " + e.ic(N_) + " : index")
for c in range(CPC):
    ic = %rgi0 + c*(NT//N_);  rows.append(ic)
    inits.append(load2(e, acc_buf, ic, "%rgj"))   # 加性初值来自另一缓冲（acc=True 语义）
# kk 外层：一次 b 装载 + CPC 次独立 a 装载 + CPC 次 FMA
header = ", ".join(f"%rga{c} = {inits[c]}" for c in range(CPC))   # iter_args 链
```

`rgemm_aw`（:3350）是其 A 操作数**滚动窗口**变体：`a[i,kk]` 改读 ping-pong `[K, BKW=32]` f32 smem 窗口（`astage_tile`+`agroup`+`await_all`，超前一个窗口 staging；24 burst/chunk），b 侧保持全 [K,BV] tile 的绝对 kk 读。窗口只喂通用 `ld.shared`（r6 规则的合法形态）；每次 wait_group 后跟 barrier 才跨 warp 读；缓冲复用距离 ≥2 迭代。

**效果数字。** `rgemm` 单独：dhs 0.83→0.59 ms；`rgemm_aw`（`GDN_DHS_AW`）：dhs 0.59→0.32 ms（1.85×），smem 32768→65536、寄存器 48→80、零 spill、cp.async 0→24。**值精确性以控制链证明**：AW=0 与 AW=1 同种子同上游 cubin → A_all/B_all/dht 逐位相等，且 **dh_all 与 dh0 逐位相等**（窗口是纯延迟变化；kk 升序 FMA 链未动；evidence/G5/scratch/r11_logs/awiso）。BKW=16 探针（8 KB 窗口、理论上 2-CTA）被测否决（0.91×：网格 64 块本就填不满设备，加倍 barrier + 16 B spill 是实价）。

## 3.10 warp 级归约、显式 FMA 与 kahan 边界

mma 机器的 epilogue 侧配套三件标量/洗牌设施：

- **`_shfl_tree5` / `warp_rowsum` / `warp_colsum`**（`gdn_bwd_gen.py:917/928/961`）：5 步 xor 蝴蝶（`gpu.shuffle xor` → `shfl.sync.bfly.b32`，probe 验证）替代"一行一线程、CW 次依赖迭代"的串行 C 链——warp $w$ 拥有行 $\{w, w+NW,\ldots\}$，lane $l$ 供给 $j=l$ 与 $l{+}32$（CW≤64），每行一棵 2 深并行树，lane 0 加进 dst。EX2 计数伴随 epilogue 重构从 149 降到 4（egc/rie/seg/bse [C] 向量 hoist 替代逐元素 MUFU sweep）。
- **`fma3`**：非 kahan sgemm 体内的显式 `math.fma`（单舍入、严格紧于 mulf+addf、同求和顺序）。本树静态分析曾发现全部 bwd kernel 零 `fma.rn.f32`——工具链从不收缩 `arith.mulf/addf`，fastmath 属性不解析；显式发射是唯一通道（probe 验证 lowering）。
- **kahan 边界**：补偿求和臂**禁止**融合——补偿项依赖精确的 sub/add 语义，`fma3` 的 docstring 明文 "NEVER inside kahan bodies"。

## 3.11 前向孪生机器（`gdn_full_gen.py`）

前向生成器持有同一布局律的独立实现（历史顺序上它是原生侧，反向机器由它移植而来）。差异在外壳：常量即时发射（`Em.c`）、lane 变量不带 `m*` 前缀、以及一套为 transform 流水线结构化的 K 循环。

### 3.11.1 `emit_gemm` 与双 fragment 模式

`_FRAG_MODE`（`--frag`）在两种 fragment 装载间切换：`"ldm"` 用 `ldm_a`/`ldm_b`（`nvgpu.ldmatrix` x4/x2，含 `swap`/`kt` 的 `.trans` 变体——A 转置时对 X 的 [k 八元组][m 八元组] 块取 `.trans`、地址行列互换）；`"scalar"` 用 `a_frag_elems`/`b_frag_elems`/`frag_build`——按 D-fragment 坐标公式逐元素 `memref.load` + `vector.insert` 构造（G2 阶段的原始形态，作为布局公式的可执行规格与回归锚保留）。A 的 8 元素坐标表（`a_frag_elems`）即 §3.2.3 表格的直接编码：

```python
pairs = [(r0, k[0]), (r0, k[1]), (r1, k[0]), (r1, k[1]),
         (r0, k[2]), (r0, k[3]), (r1, k[2]), (r1, k[3])]   # r0=m+g, r1=m+g+8
```

### 3.11.2 `frag_binop` / `frag_load_global`：fragment 级代数

`frag_binop` 对两个累加器字典做逐元素 extract/op/broadcast-insert，产出同构 `vector<2x2xf32>`（carry 链的 `X − W^T B̂ X` 类组合在寄存器格式上完成）；`frag_load_global` 从扁平 bf16 全局装载**累加器格式**的 f32 fragment（前向 S 初值：bf16 装载、extf、按 D 序 pack）——两者保证"fragment 格式"是机器内的公共数据表示，链间不落 smem。

### 3.11.3 `emit_kloop`：为 transform 流水线结构化的 K 循环

前向的 K 循环必须可被 `multibuffer + create_async_groups + pipeline_shared_memory_copies` 消化，`emit_kloop`（`gdn_full_gen.py:803`）为此定制结构：`scf.for` 步 `bk`、**累加器是 loop-carried iter_args**（`vector<2x2xf32>` 列表）、staging 是线程跨步的 `vector<8xbf16>` transfer 对、staging 与 mma 之间无条件 `gpu.barrier`、gemm K 步全展开（区域算子会使 pipeliner 退出——这是 G2 发现并被多次复核的 transform 前置条件）。两个关键实测：

1. **staging 必须线程跨步**：G2 式块一致展开让**每个**线程发射**每一条**拷贝（256× 冗余 cp.async）——比标量基线还慢 2×；线程跨步化（每线程拷自己的 16B 块、索引依赖 tid 与 iv）是前向 staged 路径 2.2-2.7× 提速的第一块基石（evidence/G5/fwd_report.md G5-D 节）。
2. `pin_alloc`：用户横跨多个循环的缓冲会触发 `multibuffer` 的硬失败（无唯一候选环）；循环外一行哑 load+store 使 transform 跳过该 alloc（对照 MemRefTransformOps.cpp 验证）。

### 3.11.4 LDSM 8 路 bank 冲突与 +8 列 pad 的定量

前向 F2（状态扫描）曾长期滞留 471 μs 级。ncu 诊断给出根因：全部 ldmatrix 读 tile 的行跨距是 **128 B 整倍数**（64×128 → 128 B、64×256 → 512 B、32×128 → 256 B），于是每象限 8 行落进**相同 4 个 bank** = 8 路冲突 = ~8× 慢的 LDSM——issue 间隔 63% 停在 "MIO queue full"、每调度器 2.0 活跃 warp、每 SM 约 40 周期才一条 LDSM、每 warp 每 chunk 76 条 LDSM。修复即机器级布局律 1：**每个 ldmatrix 读 tile +8 列 bf16 pad**（行跨距变 128+16 B 的倍数；+16 B 保持 cp.async 16 B 对齐）。F2 473→249 μs；配合 u 去 staging（vn epilogue 直读全局，其延迟藏进 vp gemm）与 w4/bv32 配置（VT 翻倍、串行链延迟减半）到 **83 μs**；数值对上一版**逐位相同**（evidence/G5/fwd_report.md G5-F 节）。反向的全部 bf16 mma tile 因此一律带 +8 pad（§3.7.1）。

### 3.11.5 epilogue 内全局存储的 40% 回归教训

把 h_all/v_new 存储折进 S-update/vn epilogue（省两遍 cell pass）曾测得 F2 249→380 μs 回归：**epilogue 存储走 fragment lane 映射（2 B @ 4 B 跨距 = 不 coalesced），独立 cell pass 线程跨步则完美 coalesced**。回退保留 copy pass；该教训与 G6 覆盖矩阵对 fla "fragment 映射存储"特征（-40% 慢）的判定互为印证：正确形态是 epilogue 只写 smem（或经 fragment 感知的合并存储），全局落盘交给线程跨步 pass。

### 3.11.6 `async_stage_unrolled` 与乒乓

前向的异步 staging 孪生（`gdn_full_gen.py:2572`）：同样的静态展开 device_async_copy burst、同样因 create_async_groups 的 wait_group-0 而手工化。F2 的 chunk 级乒乓把 parity 折进**索引算术**（wb 是 C×2K 列半、kb 是 2C×K 行半）——因为 memref 值不能 `arith.select`；chunk n 计算的顶部发出 n+1 的 w/k^T/gc 预取、尾部 wait+barrier、parity 翻转；gemm K 步全展开使 pipeliner 无环可断。仅乒乓本身曾测得 **471→473 μs（零）**——串行链瓶颈根本不在装载延迟，而在 LDSM bank 冲突（§3.11.4）；这个"负结果先于根因"的顺序本身是本机器方法论的注脚：先建位级等价的机制、再用它做归因隔离。

## 3.12 汇总：发射器清单、契约与效果索引

| 发射器/原语 | 位置（bwd 为主） | 契约要点 | 代表效果（证据） |
|---|---|---|---|
| `sgemm` | :589 | cell 单写者分区；kahan 可选；`math.fma` 显式 | 全链数值基线；kahan 保 dA 链过门（G5-E） |
| `gemm` | :2145 | `TF32_SITES` 站点派发 | 16 位点 tf32：e2e 14.16→11.06 ms（G5-G） |
| `tgemm` | :2010 | tf32 k8 atom 轮转；`warps=` 显式；neg 取 B | dqkwg 1.71×、wyb 2.19×（G5-G） |
| `gdn.mma.tf32k8` | dialect/lib/GdnHatch.cpp:170 | `=&f` early-clobber 强制；A 序 row-half 外层 | 无 `"=&f"`：118/128 cell 败（G5-G） |
| `mma_gemm` | :1726 | 条带贴图；accs= 链接；k0= K 分块；epi_bar | wy 3.04×；K32 spill 88/84/120→64/64/104 B（G5-AA） |
| `mma_gemm_mh` | :1821 | mi-half×ni-strip；W≤16；无 guard | 89% no-eligible → 全 warp；255(60 B)→248(0) regs（G5-O） |
| `mma_chain_if` | :993 | accs 过 `scf.if` 结果；尾栅栏强制 | dqkwg v3/v4 链载体（G5-O/S） |
| split 家族 | :826/:875/:1293-1460/:1671 | hi+lo f32 精确；lo·lo 舍弃；cvec 两类 | dkb 3.8× dk 不变；wyb 2.40× dk 1.28e-3（G5-J/T） |
| `stage_raw_*` | :741/:804/:855 | 精确 bf16 搬运；+8 pad；bar= 类 | raw B 操作数标准供给（G5-T） |
| `astage_tile`/`astage_sync` | :1569/:1635 | 静态展开 token；nthr 契约；wait 自持 | staging 指令 ~4× 减、逐位一致（G5-H） |
| `launder_pass` | :1650 | 恒等 st.shared 重写 | FPIPE×r6 规则的桥；位同构造（G5-V） |
| `rgemm`/`rgemm_aw`/`aggemm` | :1455/:3350/:1518 | j 共享装载；ILP=CPC；窗口喂通用读 | dhs 0.83→0.59→0.32 ms；AW 位同（G5-J/R） |
| `warp_rowsum`/`warp_colsum` | :928/:961 | 5 步蝴蝶；lane0 归约 | r4/c4r/r3/c1r 串行尾清零（G5-O） |
| `emit_gemm`（fwd） | full_gen:265 | `_FRAG_MODE` ldm/scalar 双模 | 前向 staged 2.2-2.7×（G5-D） |
| `emit_kloop`（fwd） | full_gen:803 | iter_args 累加器；线程跨步 staging | 256× 冗余消除（G5-D） |
| `async_stage_unrolled`（fwd） | full_gen:2572 | 乒乓 parity 折索引 | 位同；归因隔离 LDSM 根因（G5-F） |

**贯穿本章的三条机器级定律**（全部有位级或 SASS 级证据）：

1. **D-fragment 列索引 = $2t$（组内线程号），B-fragment = $g$（groupID）**——混用即少写一半列、多写冲突列（G4：1168/2080→2080/2080，9.77e-4==预测）。
2. **early-clobber 是 mma inline-asm 的生存条件**——`"=f"` 下 ptxas 合法地把 D 映上 A（HMMA R12,R12,R10,R16；118/128 败）。
3. **ldmatrix 的操作数必须 st.shared 血统**——raw LDGSTS 落地 tile 喂通用读正确、喂 LDSM 静默错（行块随构建调度漂移；st.shared 臂 5.2e-8 逐位对 numpy）。

它们的共同方法论：每条定律都以"最小探针 + 位级对照"钉死后才进入机器的默认路径，而这正是更大叙事——调度即 IR、编译期可判定——在指令层的投影。
