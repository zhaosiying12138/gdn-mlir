# 5 前向 kernel 体系：`tools/gdn_full_gen.py` 的调度机件全集

> 本章自足地展开前向 Gated DeltaNet（GDN）算子的全部 GPU kernel 机制。每一小节对应一个可独立开关的机制（旋钮），并按固定四件套呈现：**动机**（算子层面的瓶颈与具体例子）→ **源码片段**（生成器 `tools/gdn_full_gen.py`，共 4464 行）→ **IR 前后对比**（`evidence/G5/gen_fwd/` 下的真实 `payload.mlir` 摘录）→ **效果**（配对 A/B 或交错终判的数字，出处随文标注）。本章按**最终发运系统**呈现，不按轮次叙事；全部数字取自 `evidence/G5/fwd_report.md` 对应机制段与 `evidence/G5/gen_fwd/` 的在档 cubin/payload，口径为共享 GPU 交错配对（interleaved adjudication，见全局口径声明）。

## 5.0 章首：三 kernel 分解、融合旗舰与 mono 三重奏的架构图景

**数学对象先行定义。** 前向 GDN 把序列切成 $C=64$ 行的 chunk（尾块零填充，$T_p=NC\cdot C$）。记 $\mathrm{gc}$ 为门控 $g$ 在 $\log_2$ 域的 inclusive 前缀和（$g$ 本身是自然对数域，乘以 $\mathrm{RCP\_LN2}=1.4426950\ldots$ 折入 $\log_2$ 域，使一切衰减因子可由 `math.exp2` 单条 SFU 指令生成），$\beta$ 为删除门，则每 chunk 的五步复合为（`gdn_full_gen.py` 文件头，L1 已验证真值）：

$$
A_{i,j}=\beta_i\,(k_i\!\cdot\! k_j)\,2^{\,\mathrm{gc}_i-\mathrm{gc}_j}\ (\text{严格下三角}),\qquad
T=(I+A)^{-1},
$$
$$
u=T\,(\beta\!\odot\! v),\qquad w=T\,(\beta\!\odot\! k\!\odot\!2^{\mathrm{gc}}),\qquad
v_{\text{new}}=u-w\,S,
$$
$$
o=\Big((q\!\odot\!2^{\mathrm{gc}})\,S+\mathrm{tril}_{\text{inc}}\big((q\,k^{\top})\odot 2^{\mathrm{gc}_i-\mathrm{gc}_{\text{last}}}\big)\,v_{\text{new}}\Big)\cdot K^{-1/2},\qquad
S\leftarrow S\cdot 2^{\mathrm{gc}_{\text{last}}}+(k\odot 2^{\mathrm{gc}_{\text{last}}-\mathrm{gc}})^{\top}v_{\text{new}} .
$$

dtype 契约沿用 fla：$q,k,v,\beta,g$ 的输入为 bf16/fp32 混合（$q/k/v$ bf16，$\beta/g$ fp32），$o$ bf16，$h_0/\mathrm{ht}$ fp32，**跨 chunk 携带态以 bf16 存储，全部算术 f32**（mma f32 累加器，G4 契约）。

**执行平台与全章共用资源口径**：RTX 5090 Laptop（消费级 Blackwell，`sm_120a`，82 SM，**无 tcgen05**，Ampere 风格 `mma.sync`；共享内存 opt-in 上限 $101{,}376$ B/CTA（即 r1 报告所称的 99 KB 硬顶，`0x18c00`），静态申报线 $49{,}152$ B（48 KB——两倍恰为 2 CTAs/SM 的门），每 SM 102,400 B）。全部 tile 形状、门限与占用结论都在这三条资源线上展开。

**路径分解。** 生成器提供 `--path staged|fused|mono` 三种融合边界：

- **staged 三 kernel**（fla 同构分解）：
  - **F1 "wy"**，grid $(NC,\,B\!\cdot\!H)$——token 并行 WY 预计算：gc 扫描、$A$、$T$、$u$、$w$；
  - **F2 "scan"**，grid $(B\!\cdot\!H\!\cdot\!VT,\,1)$——跨 chunk 状态递推：$S$ 的串行 chunk 链，产出入口态 $\mathrm{h\_all}$、$v_{\text{new}}$、终态 $\mathrm{ht}$；
  - **F3 "o"**，grid $(NC\!\cdot\!B\!\cdot\!H,\,VT)$——输出级：从 $\mathrm{h\_all}/v_{\text{new}}$ 重算 $o$。
- **fused 旗舰**：单一 persistent kernel（grid $B\!\cdot\!H\!\cdot\!VT$），chunk 循环进核，$S$ 常驻共享内存，每 chunk 串行完成 WY+solve+状态更新+输出（§5.4）。
- **mono 三重奏**（`gen_mono` + `wrap_module_mono`）：把 F1/F2/F3 三个零参 kernel 与一套公共 `memref.global` 缓冲封进**一个 cubin**——kernel 边界即流序同步，staged 路径因此不再需要任何设备到设备（DtoD）交接拷贝（r1 时代每形状约 0.1 ms 的交接开销由此消失）。发运载体是 mono：F1 的 `gc_out` 输出在 mono 装配时直接改名为 `gc`，三个 kernel 共享同一全局。

每个 kernel 的计算骨架是同一套 G2/G4 机件：全 K 操作数 staging 到 workgroup 共享内存（`vector.transfer` 对 + `create_async_groups` 或手动 `nvgpu.device_async_copy`）、warp 平铺的 `nvgpu.mma.sync` m16n8k16 bf16 原子（f32 累加，D-fragment 列索引 $=2\cdot\mathrm{tig}$——G4 修复律）、`nvgpu.ldmatrix` fragment 装载（每 fragment 一条指令替代 12 条标量 `ld.shared`）、自定义 epilogue（衰减/门控/掩蔽在写回前折入 f32 域）、阶段间 `gpu.barrier`。

**最终判定口径数字**（r25 存档会话，`final_adjudicate_r25f_w32p.json`，30 交错对/形状，发运旋钮全套 + `GDN_W32P=1`，cubin `e7c8eb4c`/`39f094c3`/`be5b2377`）：B1T2048 **2.1250×**（我方中位 0.1866 ms）、B4T2048 **1.2211×**（0.8213 ms）、B1T8192 **1.3493×**（0.7851 ms），对 fla v0.6.0 纯 Triton 前向。数值契约 4/4 PASS：staged T2048 $o\le 6.486\times10^{-4}$ / $\mathrm{ht}\le 7.493\times10^{-4}$ vs fp64 JAX 参考（bar $5\times10^{-3}$，约 7 倍裕量），T8192 fp64 门 $o\,6.626\times10^{-4}$ / $\mathrm{ht}\,8.918\times10^{-4}$（bf16-solve 类）。以下各节把这两个数字背后的每个机件拆开。

---

## 5.1 F1 "wy"：WY 预计算 kernel

F1 的最终形态是一条五段串行流水（每 CTA 处理一个 (chunk, head) 对，128 线程 = 4 warp）：

1. **gc Hillis-Steele 对数扫描 + β staging**（`emit_cumsum_beta_scan`，§5.1.1）；
2. **kkt K-循环**：一块 staging 的 $k$ tile 同时充当 mma 的 A 侧与 B 侧，算 $k k^{\top}$，epilogue 里折入 $\beta$ 与衰减得 $A$（f32→bf16 双写 `A_m`/`Abf`，§5.1.2）；
3. **分块 16×16 前代求解**（`emit_solve_blocked`，§5.1.3）；
4. **uw 段**：$u=T_b\,v$、$w=T_g\,k$ 两个 mma，A 操作数是常驻的 $\mathrm{Tb}/\mathrm{Tg}$（$T^{\top}$ 经 $\beta$、$\beta 2^{\mathrm{gc}}$ 列缩放的转置型），B 操作数 cp.async staging（§5.1.4–5.1.5）；
5. **gc_out 存回**（f1 的 gc 供 f2/f3 复用，省去它们各自重扫）。

F1 的共享内存池在发运形态（F1KS+F1ZF）下为**申报 46,608 B ≤ 49,152 B ⇒ 2 CTAs/SM**（`cuOccupancyMaxActiveBlocksPerMultiprocessor = 2`，前向第一个达成的 2-CTA kernel），寄存器 176（`ptxas_v.txt`，0 spill）。这一占用跃迁是 f1 的主胜利来源（f1 −53.5 µs @T8192，§5.1.5）。

### 5.1.1 机制：gc 的 Hillis–Steele 对数扫描（与 SCANDB 双缓冲硬化）

**动机（算子层面）。** F1 每chunk需要 64 个 $\mathrm{gc}_t=\sum_{s\le t}g_s\cdot\log_2 e$ 值。朴素实现是每线程串行重扫（每线程平均 32 次 H 跨步全局读——f1 的输入布局是 $[B,T_p,H]$，同一 chunk 同一 head 的 64 个 g 值在全局里彼此相距 $H$ 个元素，天然不合并）。Hillis–Steele 把它改成：每线程一次合并读入共享内存，随后 6 步（$d=1,2,4,8,16,32$）对数深度扫描，步间 barrier。数值上属于 f32 加法树重结合类（与串行前缀和的差远低于契约）。

**源码片段**（`emit_cumsum_beta_scan`，gdn_full_gen.py）：

```python
for i, d in enumerate((1, 2, 4, 8, 16, 32)):
    src, dst = gc, gc                      # 单缓冲（SCANDB 关时）
    ...
    e.l(f"{off} = arith.subi %tid, {e.c(d)} : index")
    e.l(f"{ge} = arith.cmpi sge, %tid, {e.c(d)} : index")
    e.l(f"{offc} = arith.select {ge}, {off}, %c0 : index")
    e.l(f"{lv} = memref.load {src}[{offc}, %c0] : {src_ty}")
    e.l(f"{vv} = arith.select {ge}, {lv}, {zf} : f32")
    e.l(f"{nv} = arith.addf {cur}, {vv} : f32")
    e.l(f"memref.store {nv}, {dst}[%tid, %c0] : {dst_ty}")
    e.l("gpu.barrier")
```

**机器缺陷与 SCANDB（`GDN_SCANDB`，发运开启）。** 单缓冲步骤内有一个**跨 warp 的步内竞态**：线程 $t_2$ 的 `gc[off]` 读与线程 $t_1$（$t_1=t_2-\mathrm{off}$）的 `gc[t_1]` 写在同一步内无序——barrier 只隔步不隔步内。1-CTA 驻留时 warp 0/1 恰好日程对齐而良性；**2-CTA 共驻把它破坏**（r17f F1KS 双 CTA 损坏二分定位到的第二个 bug；compute-sanitizer racecheck 在含发运版在内的每个构建里都精确标出这 6 对）。修复即 SCANDB：$+256$ B 的第二缓冲 `gcs2 [64] f32`，偶数步读 `gcs` 写 `gcs2`、奇数步反向，6 步后值落回 `gcs`（偶数步数免拷回）。

**IR（发运 payload，SCANDB 开启的第 6 步 $d=32$，`bench_B1T2048H16_mono.r22ship/payload.mlir`）**：

```mlir
%cso_6 = arith.subi %tid, %c32 : index
%cge_6 = arith.cmpi sge %tid, %c32 : index
%csc_6 = arith.select %cge_6, %cso_6, %c0 : index
%cgl_6 = memref.load %gcs2[%csc_6, %c0] : memref<64x1xf32, ...>   ; 读对侧缓冲
%csv_6 = arith.select %cge_6, %cgl_6, %czf : f32
%cgc_6 = memref.load %gcs2[%tid, %c0] : memref<64x1xf32, ...>
%cgn_6 = arith.addf %cgc_6, %csv_6 : f32
memref.store %cgn_6, %gcs[%tid, %c0] : memref<64x1xf32, ...>      ; 写回本侧
```

**效果。** r18 双 bug 根因链中，仅 ZFIX 时损坏从 ~70 万 bit 差塌缩到 ~4 千、每 run 在 1–2/512 tile 触发且指纹恰为 `gc[32:64)` 窗口（$d=32$ 步的跨 warp 读）；ZFIX+SCANDB 合用后 **10/10 次仅-f1 复跑与强 1-CTA 参考位级一致**，三重奏命中精确记录数字（$o\,6.486\times10^{-4}$ / $\mathrm{ht}\,7.493\times10^{-4}$）。1-CTA 驻留下该硬化数值上值精确（SCANDB 构建对旧船在 T8192 fp64 门位级一致，$o/\mathrm{ht}=0.000e{+}00$）。删除探针给该扫描+β staging 的表观成本 **17.6 µs（f1 的 7.8%）**（r21 时间图谱，`GDN_F1T_NSC`，§5.1.7）。

### 5.1.2 机制：kkt K-循环（transform 流水线形态与它的屋顶线关闭）

**动机。** $A=\mathrm{tril}(\beta\odot(k k^{\top})\odot 2^{\Delta\mathrm{gc}})$ 需要一个 $64\times64\times K$ 的 GEMM。G2 机件（`transform.memref.multibuffer` + `transform.nvgpu.create_async_groups bypass_l1` + `transform.nvgpu.pipeline_shared_memory_copies depth=2 peel_epilogue`）只在**扁平 `scf.for` K-循环**上生效，因此 kkt 被结构化为：循环体 = 线程跨距的 16 B `vector.transfer` staging 对（每线程拷**自己的**块——G2 式块均匀展开会让每个线程发每份拷贝，256× 冗余 cp.async，实测比同步版本还慢 2×，r2 的头号修复）+ ldmatrix fragment 装载 + 8 累加器链的 `mma.sync`，累加器作为 `iter_args` 循环携带。

**源码片段**（`emit_kloop`，staging 侧；gemms 以 (tile, staged) 三元组声明）：

```python
for kk in range(npairs // nthr):
    p = addi(e, "%tid", e.c(kk * nthr)) if kk else "%tid"
    r, cc = divsi(e, p, e.c(nch)), remsi(e, p, e.c(nch))
    ...
    e.l(f"{t} = vector.transfer_read {s['src']}[{idx}], {pad} "
        f"{{in_bounds = [true]}} : {s['sty']}, vector<8xbf16>")
    e.l(f"vector.transfer_write {t}, {t_ssa}[{r}, {c8}] ...")
...
e.l(f"{o} = nvgpu.mma.sync({fa[...]}, {fb[...]}, {prev}) "
    f"mmaShape = [16, 8, 16] : (vector<4x2xbf16>, vector<2x2xbf16>, "
    f"vector<2x2xf32>) -> vector<2x2xf32>")
```

**IR（发运 payload 的 kkt 循环头与首对 staging，`r22ship/payload.mlir`；注意 F1KS 使步长为 16、tile 为 `[64,24]`）**：

```mlir
%rs_kk_kkt_0_0, ... = scf.for %kb_kk = %c0 to %c128 step %c16
    iter_args(%ia_kk_kkt_0_0 = %az_kk_kkt_0_0, ...) -> (vector<2x2xf32> x8) {
  %pd_kk = arith.constant 0.0 : bf16
  %idv = arith.divsi %tid, %c2 : index          ; 每线程 2 个 16B 块
  %irm = arith.remsi %tid, %c2 : index
  %im_13 = arith.muli %irm, %c8 : index
  %srkk = vector.transfer_read %k[%ix_11], %pd_kk {in_bounds=[true]}
      : memref<4194304xbf16>, vector<8xbf16>
  vector.transfer_write %srkk, %kt[%idv, %im_13] {in_bounds=[true]}
      : vector<8xbf16>, memref<64x24xbf16, #gpu.address_space<workgroup>>
  gpu.barrier
  %fa_kk_kkt_0_0 = nvgpu.ldmatrix %kt[%ix_13, %ix_14] numTiles = 4
      transpose = false : ... -> vector<4x2xbf16>
  %mr_kk_kkt_0_0_0 = nvgpu.mma.sync(%fa_kk_kkt_0_0, %fb_kk_kkt_0_0, %ia_kk_kkt_0_0)
      mmaShape = [16, 8, 16] : (...) -> vector<2x2xf32>
```

A 的 epilogue（`epi_A2`）在每个 fragment 单元上折入 $\beta_{\text{row}}$、$2^{\mathrm{gc}_i-\mathrm{gc}_j}$（三次共享内存读 + `subf/exp2/2×mulf` + `cmpi slt` 严格下三角选择），f32 写 `A_m`（仅非 bf16-solve 形态）/截断 bf16 写 `Abf`。

**效果与关闭（r23 的拆分探针 + 屋顶线）。** r21 删除探针给整个 kkt 段 **−59.5 µs（f1 的 26%）**；r23 的三路拆分（`GDN_F1T_NKK/NEK/NKM`，各 ×3 会话）把它分解为：**k-staging 全局流 40.6 µs（73%）**、ldmatrix+mma 13.0 µs（23%）、exp2/β 链 2–4 µs（4–7%）、store ≤2 µs。staging 流每 CTA 恰好一次读尽自己的 $64\times128$ bf16 k 块（$B\cdot T_p\cdot H\cdot K\cdot2=33.55$ MB @T8192，100% 扇区利用、32 B 对齐、bypass L1），40.6 µs 折合 **826 GB/s**；同窗 D2D 拷贝标定器件聚合读写带宽 **765 GB/s**——一条只读流超过读写聚合上限，意味着**无重构余量**：kkt 在当前几何（2 CTAs/SM）下以算法最小流量顶在 DRAM 屋顶线上，作为性能之门被关闭（`scratch/r23f/bw.py`；tile 变体 `GDN_F1KS_BK32` 把 kt 多缓冲 6,144→10,240 B、f1 申报池 ~49.7 KB，occupancy API 读 blocks/SM 2→1，配对 ×3 = +70.3/+73.4/+68.3 µs **回退**——全部“增益类”都是占用损失，非 tile 经济学）。

### 5.1.3 机制：分块 16×16 前代求解（对角块 16 步串行代入 + 非角块 stash/consume mma）

**动机。** $T=(I+A)^{-1}$ 按列前代代入：$T_{i,j}=-\sum_{j\le k<i}A_{i,k}T_{k,j}$。平铺 64 列的每列一个线程版是 2016 条依赖 FMA 的串行链（实测 f1 临界路径 ~45 µs/CTA）。fla 式再瓦解把 $A$ 分成 $4\times4$ 个 $16\times16$ 块：

- **phase 1（对角块）**：线程 $t<64$ 认领 (块 $D=t/16$，列 $c=t\bmod16$)，做 16 步短代入（每列至多 ~120 条串行 FMA，替代平铺版的 2016 条）。源码（`emit_solve_blocked` 的串行臂，发运形态）：

```python
for t in range(16):
    i = addi(e, muli(e, db, c16), e.c(t))      # 块 D 的全局行
    e.l(f"{svi}, {sv} = scf.while (%sk = {ccol}, %sa = {zf}) : (index, f32) ...")
    #   体: %aik = load Abf[i, %skb]; %tkj = load Tt[%skb, ccol]
    #       extf→f32; mulf; addf —— 严格保序的上升 k 链
    e.l(f"{val} = arith.select {isd}, {one}, {neg} : f32")   # 对角=1
    #   写三联: Tt[i,ccol]（未缩放）+ Tb[ccol,i]·β + Tg[ccol,i]·β·2^gc（转置缩放型）
```

  $\beta$ 列缩放可穿过递推（$X=T\,\mathrm{diag}(\beta)$ 满足同一代入、$X_{jj}=\beta_j$），因此**两个消费者 GEMM 直接读原始 $v$/$k$**——r1 时代的标量 $\beta v$、$\beta k 2^{\mathrm{gc}}$ staging 扫pass整体消失（r2 F1 重构 429→270 µs 的一部分）。

- **phase 2（六个非角块）**：按块行 $R=1,2,3$ 两拍推进——**stash 拍**（warp $C_b<R$ 各自算 $S=\sum_K \mathrm{Abf}[R,K]\,@\,\mathrm{Tt}[K,C_b]$ 进**自己的** scratch tile `St_Cb`，m16n8k16 bf16 mma）、barrier、**consume 拍**（$T[R,C_b]=-(\mathrm{Tt}[R,R]\,@\,S)$，经 scratch 的第二个 mma），barrier。两个历史 bug 值得记录：stash 拍 B-fragment 列基漏了 $+16\,C_b$ 块偏移（warp 0 读对块、warp 1/2 读错块；16×16 块 dump 差分定位）；warp 守卫的 `scf.if` **内部**的 `gpu.barrier` = 部分到达 = `sm_120a` 上同步损坏——barrier 一律移到无条件位置。

**IR（phase 1 的代入 `scf.while`，`r22ship/payload.mlir`；bf16-solve 使 A 读自 `Abf`）**：

```mlir
%sviw, %svw = scf.while (%sk = %ix_28, %sa = %svz) : (index, f32) -> (index, f32) {
  %scnd = arith.cmpi slt %sk, %ix_29 : index
  scf.condition(%scnd) %sk, %sa : index, f32
} do {
^bb0(%skb: index, %sab: f32):
  %aikb = memref.load %Abf[%ix_29, %skb] : memref<64x72xbf16, ...>
  %aik  = arith.extf %aikb : bf16 to f32
  %tkj  = memref.load %Tt[%skb, %ix_28] : memref<66x72xbf16, ...>
  %tkf  = arith.extf %tkj : bf16 to f32
  %pr   = arith.mulf %aik, %tkf : f32
  %sn   = arith.addf %sab, %pr : f32
  %skn  = arith.addi %skb, %c1 : index
  scf.yield %skn, %sn : index, f32
}
```

**两个诚实的负结果（phase-1 重构两连败）。** phase-1 串行代入被两次不同思路的重构攻击，均正确、均更慢：
- **F1WY**（`GDN_F1WY`，warp 并行 phase-1 v3）：每对角块一个 warp，行向量蝴蝶归约（宽 32 的 `gpu.shuffle xor`，5 级 shfl+add 同时归约 16 列点积）。三刀才正确：宽 16 的蝴蝶降为 `shfl.sync 0xFFFF` 成员掩码（lane 16–31 执行即 UB）；16 路按 lane 选择存储网被工具链错误编译（列 ≥4 全陈旧）；最终 v3 = 无选择网——T 行驻 `Tt` 共享内存（对角块预清零）、存储列是编译期常数（warp 冗余广播存）、每步一 barrier。T8192 fp64 门**与船位级一致**，但 f1 **+92.6/+90.4/+91.4 µs ×3**（77–80/80 对更慢）：串行代入的共享内存读被编译器流水得极好，而蝴蝶的指令数 ~4–5×（1280 shfl+add、768 冗余存、每 solve 16 条额外 CTA barrier），2 CTAs/SM 下发射压力主导。
- **F1P4**（`GDN_F1P4`，4 线程/块 × 4 列寄存器瓦片）：每行一条 `scf.while` k-链带**四条交错累加链**（4× FMA ILP、每步一次 $A_{i,k}$ 读摊给 4 列，~4.5 指令/FMA 对串行链的 ~8）。位一致性由构造保证：$k<c$ 项读 `Tt` 的 $-0.0$ 上角，IEEE 零和规则使真项以**同一上升序**累加。T8192 门位级一致、寄存器 172→168，但 f1 **+14.8/+14.5/+15.5 µs ×3**：16 活跃线程 × 4 宽 ILP 输给 64 线程 × 1 宽——phase-1 重要的是线程级并行而非每线程链深。

r21 的删除探针最终裁定 **phase-1 仅占 f1 的 14.3%（32.5 µs）**，“phase-1 是 f1 临界路径”的论题就此死亡（f1 的大头是 uw 段 51%，§5.1.4）。附带建构细节：k-链必须 `scf.while` 而非 `scf.for`——payload 的 transform 流水化器匹配**每一个** `scf.for`，对无拷贝循环（“no shared memory copy”）即便 `failures(suppress)` 也硬错整个 `-transform-interpreter` 输出（探针验证；这也是全部非 K 循环 cell 扫pass用 `scf.while` 的原因，`emit_cell_while`）。

**bf16-solve（`GDN_BF16SOLVE`，F1KS 的隐含前提）。** 去掉 f32 `A_m` 镜像：phase-1 递推直接读 bf16 `Abf`（extf 到 f32 参与 FMA），kkt epilogue 只写 `Abf`。CPU 孪生（numpy 按 kernel 精确 bf16 切点仿真）先证：$T$ 本就经 phase-2 mma 与 `Tt` 反馈以 bf16 存活，`A_m` 带进 phase-1 的 f32 位两个操作后就被舍掉——变体 B 与记录数字统计不可分（最坏 $6.7\times$ 裕量于 $5\times10^{-3}$ bar）。池 −16,384 B，为 phase-1 共活下限从 57,408 B 降到 **41,024 B < 50,688 B** 铺平 2-CTA 之路（§5.1.5）。GPU 实测数值过门（$o\,6.626\times10^{-4}$/ht $8.918\times10^{-4}$，即此后所有 F1KS 构建的记录数字类）、性能中性（−2.6/−3.2 µs，噪声）。

### 5.1.4 机制：F1UW——uw 段重构（单趟化、预解 staging、死 tile 复用）

**动机。** uw 段（$u=T_bv$、$w=T_gk$）的三重瓶颈：(i) K-归约维恰为 $C=64$，pipeliner 的 `scf.for` 单趟形态直接“pipelining preconditions failed”硬错；(ii) operand staging（32 KB）若在解后同步发出，就串在关键路径上；(iii) epilogue 的 fragment 映射全局存是**每线程 128 条离散 2 B `st.global.b16`**（LSU/LG 节流受限，r3 ncu：46.8% 内存吞吐、557 K 共享存储 bank 冲突级别的低效类）。

**三合一改造**（`gen_f1` 的 `_f1uw` 臂，默认开启；`G5K_NOF1UW=1` 回退）：

1. **bk=C 单趟直线化**：uw mma 以 $k\in\{0,32\}$ 两组 fragment 全展开（`_mma_chain`），无循环、无 pipeliner 参与。
2. **staging 提前到 solve 之前发出**：v/k tile 与一切 solve tile 不相交，前代代入的串行延迟正好藏住这 32 KB 拷贝：

```python
if _f1uw and not _f1al and not _ks and not _nust:
    # 在 solve 之前发出 uw operand staging：拷贝写 vt/k2，与全部 solve tile 不相交
    async_stage_unrolled(e, "%v", TY["v"], vtile, vtile_ty, C, V, nthr, ...)
    async_stage_unrolled(e, "%k", TY["k"], ktile2, ktile2_ty, C, K, nthr, ...)
    guw = e.v("agr")
    e.l(f"{guw} = nvgpu.device_async_create_group {', '.join(toks)}")
...
emit_solve_blocked(...)          # 串行 solve 藏住上面的拷贝
...
e.l(f"nvgpu.device_async_wait {guw} numGroups = 0")
e.l("gpu.barrier")               # 此后才发射 mma
```

3. **死 tile 复用为输出 restage**：循环后 vt/k2 作为 B 操作数已死，被回收为 u/w 的输出 staging tile——epilogue 写**带垫共享 tile**（无冲突 STS），再各用一次合并 cell 扫pass搬去全局（64 B/warp 行段），替代离散 2 B 全局存。字节数上 vt/k2 变单缓冲 `[C,N{+}8]` 恰等于原多缓冲二份的字节，**池不变**（F1 仍 101,120 B 时代）。

**效果。** r5 配对 A/B（×50 对）：f1 @T8192 0.416→0.293 ms、staged e2e **−119 µs**（4/50 更慢）；@B4T2048 f1 −122 µs（0/50）；@T2048 f1 −37 µs、e2e −50 µs（2/50）。数值位级一致（truncf 舍入不变，只动了存储顺序；T8192 fp64 门与记录位级同）。r14 时代 u/w 存储的 s2g 扫pass又被 WST 宽存储进一步折半（§5.2.4/§5.3.3）。

### 5.1.5 机制：F1KS——K-分裂 staging 与终身 arena，前向第一个 2-CTA kernel

**动机。** F1 是 T8192 上最大的前向 kernel（25 波网格）。2-CTA 驻留（申报池 ≤48 KB）的算术障碍：f32 `A_m` 在内时 phase-1 共活下限 57,408 B > 50,688 B（r15 的 F1AL 别名穷举结论——“只有丢掉 A_m 或重构 solve 才能过线”）。bf16-solve（§5.1.3）把下限打到 41,024 B 后，**新的堵点移到 uw 段**：$\mathrm{Tb}+\mathrm{Tt}+\mathrm{Tg}$（31,296 B）+ vt（17,408）+ k2（17,408）= 66,112 B > 50,688 B。

**设计（`GDN_F1KS`，发运开启；栈于 bf16-solve 之上）**——三刀：

1. **uw 的 B 操作数 K-分裂**：vt/k2 砍半为 `[32,136]`/`[33,136]`（k2 的 +1 行是为“按尺寸识别”的 malloc 尺寸唯一性），两个顺序半 K 扫pass复用同一套半 tile；A 操作数 $\mathrm{Tb}/\mathrm{Tg}$ 保持全 tile、在列 $32p$ 处读取。累加器跨扫携带：32 的分裂点**落在四个 m16n8k16 块之间**，每输出元素的 f32 加法序列与原 4 块链位级相同（CPU 孪生 0 翻转；树重结合对照组确实翻位，探针敏感性证明）。
2. **kkt 降到 bk=16**：kt 多缓冲 10,240→6,144 B。
3. **PTX 终身 arena**（生成器写 `smem_alias.json`，`tools/gdn_smem_fix.py` 消费）：半 tile **落在 solve 之后的死字节上**——vt→`Abf` 的字节、k2→`Tt` 的字节（两者在 uw 段均已死；solve 的末 barrier 覆盖其最后读）。按尺寸识别需要各 spec 成员 malloc 尺寸唯一：用**行粒度膨胀**（+1/+2 死行、144 B 行距不变）替代列垫（列垫每块 1–2 KB，且 ptxas 把非 128 倍数申报向上补齐）。申报合计 46,144 B，ptxas 补垫后 46,352 B ≤ 49,152 B。

```python
alias_modules["gdn_fwd_f1_kernel"] = {
    "identify": {"9216": "Abf", "9504": "Tt", "8704": "vt", "8976": "k2"},
    "arenas":   {"Abf": 9216, "Tt": 9504},
    "alias":    {"vt": ["Abf", 0], "k2": ["Tt", 0]}}
```

**IR（发运 payload 的 F1KS 扫pass 0 预取组，`r22ship/payload.mlir`；解后落位、与 gc_out 存回重叠）**：

```mlir
%acp   = nvgpu.device_async_copy %v[%ix_223], %vt[%idv_5, %im_31], 8 bypassL1
    : memref<4194304xbf16> to memref<32x136xbf16, #gpu.address_space<workgroup>>
...  (8 条：v 半块 ×4 + k 半块 ×4)
%agr_p0 = nvgpu.device_async_create_group %acp, ..., %acp_8
...
nvgpu.device_async_wait %agr_p0 numGroups = 0    ; ks_pass(0) 前
gpu.barrier
...  ks_pass(0)：Tb/Tg 全 tile 列 0..31 + vt/k2 行 0..31 的 mma
gpu.barrier
%agr_p1 = nvgpu.device_async_create_group ...      ; 扫pass 1 的组
nvgpu.device_async_wait %agr_p1 numGroups = 0      ; ks_pass(1) 累加器接续
```

**两个 2-CTA 隐蔽 bug（根因链，r18）。** K-分裂码在 1-CTA 强制臂上位级正确；2-CTA 共驻下 u/w 损坏（$o\sim6.3\times10^{-2}$）而 ht 仍精确——八个二分臂排除了码值、arena、staging 机制、bk、mb、池尺寸后，根因是**两个叠加的、被 1-CTA 驻留掩盖的独立 bug**：
- **Bug 1（uninit-smem 步长缺陷）**：phase-0 `zero_up` cell 循环步长 $32\cdot8=256$ 超过 128 线程块，$\mathrm{Tb}/\mathrm{Tg}$ 的 $(r\cdot64+c)\bmod256\in[128,256)$ 单元（每 tile 768 个）**从未被清零**；uw gemm 的转置 ldmatrix A 读覆盖全 64×64，把这些单元当作 $T$ 的上三角零读。1-CTA 时新窗口恰好读到精确零（393,216/393,216 单元实测）；2-CTA 共驻读到非零陈旧数据。修复 `GDN_ZFIX`：步长改 `nthr`，全覆盖。
- **Bug 2（扫描步内竞态）**：§5.1.1 的 SCANDB。
ZFIX+SCANDB 之后 2-CTA 下 10/10 位级干净。损坏指纹（w 恰在块列 $C_b=2$=行 32–47、u 在行 0–47、行 48–63 干净）同时解释了 o/u-vs-ht 之谜：损坏 chunk 的态贡献被衰减压到契约之下所以 ht 完好，而 f3 直接读损坏 chunk 的 `h_all`/`v_new` 所以 o 失败。

**效果。** 发运（F1KS 2-CTA）对 r15/r16 船的配对中位（×3 会话 ×40 对）：T8192 f1 278.7→225.2 µs（**−53.5 µs**，35–38/40 更快）；T2048 −16.7 µs；B4T2048 −57.9 µs（链 −75.4 µs）。r19 的均衡双装载复测收紧为 f1 −56.4 µs（3/80 更慢）、staged e2e ~−62±5 µs、**全部来自 f1**（f3 的原始 B 侧罚分 100% 是模块摆放抽签，§5.2.4 注）。r17 预估的“波数减半”落地为 kernel 时间 −19%。

### 5.1.6 机制：F1ZF——phase-0 清零折进 phase-1 的空闲线程（与 F1EP 的 wash 关闭）

**动机。** r21 时间图谱：`zero_up`（phase-0，为 $\mathrm{Tb}/\mathrm{Tg}$ 的转置上三角清零）值 ~8.5 µs，而它串行地排在 64 线程的 ~32 µs 串行代入**之前**——同一时刻线程 64–127 全程空闲。

**改造（`GDN_F1ZF`，发运开启）**：把清零循环**折进** phase-1 的守卫块——线程 64–127（`%tid >= 64`）在代入运行的同时清“活上三角”单元，谓词加强为**块级** $\mathrm{block}(r)>\mathrm{block}(c)$（蕴含 $r>c$）。位一致性论证：折叠臂只写代入**永不触碰**的单元（串行链写全部同块单元含其 $-0.0$ 上角），写集不相交 ⇒ $\mathrm{Tb}/\mathrm{Tg}$ 的位与船确定性地相同；一条 barrier 取代原 phase-0/phase-1 两条。

**IR 前后对比**（同为 T2048 mono 构建的真实 payload 差分；`gen_fwd/B1T2048_mono.r18ship`（前，旋钮集少 F1ZF）vs `bench_B1T2048H16_mono.r22ship`（后，+F1ZF）——f1 段的实质差异即 F1ZF 一项）：

前（独立 phase-0，全体线程，随后专属 barrier）：

```mlir
%wir = scf.while (%wi = %tid) : (index) -> (index) {
  %wic = arith.cmpi ult %wi, %c4096 : index
  scf.condition(%wic) %wi : index
} do {
^bb0(%wib: index):
  %idv_2 = arith.divsi %wib, %c64 : index        ; r
  %irm_2 = arith.remsi %wib, %c64 : index        ; c
  %svgt = arith.cmpi sgt %idv_2, %irm_2 : index  ; r > c
  scf.if %svgt {
    memref.store %svz2, %Tb[%idv_2, %irm_2] : ...
    memref.store %svz2, %Tg[%idv_2, %irm_2] : ...
  }
  %win = arith.addi %wib, %c128 : index
  scf.yield %win : index
}
gpu.barrier                                         ; 专属 barrier
```

后（折进空闲半块、与代入并发、步长 64、块级谓词；总 barrier 减一）：

```mlir
%zfe = arith.cmpi uge %tid, %c64 : index          ; 折叠臂：线程 64..127
scf.if %zfe {
  %wir = scf.while (%wi = %tid) : (index) -> (index) {
    %wic = arith.cmpi ult %wi, %c4096 : index
    scf.condition(%wic) %wi : index
  } do {
  ^bb0(%wib: index):
    %idv_2 = arith.divsi %wib, %c64 : index
    %irm_2 = arith.remsi %wib, %c64 : index
    %idv_3 = arith.divsi %idv_2, %c16 : index     ; block(r)
    %idv_4 = arith.divsi %irm_2, %c16 : index     ; block(c)
    %svgt = arith.cmpi sgt %idv_3, %idv_4 : index ; 块级谓词
    scf.if %svgt { memref.store %svz2, %Tb[...] ...; memref.store %svz2, %Tg[...] ... }
    %win = arith.addi %wib, %c64 : index          ; 步长 64
    scf.yield %win : index
  }
}                                                   ; 无专属 barrier
```

**效果。** f1 配对：T8192 **−7.7/−7.9/−6.7 µs ×3**（77–78/80 对）、T2048 −2.3 µs（57/80）——约为 NP0 删除界 8.5 µs 的 91%（折叠是重叠、删除是废除）。门：T2048 三重奏 7/7 数组位级同、T8192 fp64 门位级同（记录数字 $6.626\times10^{-4}$/$8.918\times10^{-4}$）、det N=5 前向确定。F1ZF 就此加入发运旋钮集，r22 完成记录迁移（三形状 n=30 正式终判全部落在同一 cubin 代上：T8192 1.2226×、B4 1.1417×、T2048 首次正式 2.1184×）。

**F1EP（`GDN_F1EP`，默认关——wash 关闭）。** F1KS 的 epilogue 以两个行半程过**一套**半 tile（4 barrier + 两轮 barrier 隔离的拷贝）。F1EP 把行 $[32,64)$ 的输出折进 u2/w2——两个 PTX 级别名到 $\mathrm{Tb}/\mathrm{Tg}$ 死字节的 tile（几何：`Tg` 增到 `[67,72]`，**行**垫 +288 B 保持 144 B 行距为 16 B 倍数——列垫到 76 会造出 152 B 行距、奇数行 ldmatrix `MISALIGNED_ADDRESS`，探针实证），一条 barrier + 一个合并的四 tile 拷贝流。门全绿（位级一致、池 46,432 ≤ 49,152 保 2-CTA），但配对 **+0.0/+0.6/+0.8 µs ×3、T2048 +0.2 µs（39–49/80）——掷硬币**。机理：2 CTAs/SM 下**共驻 CTA 填另一 CTA 的 barrier 间隙**，核内 epilogue 串行化本就不在 SM 级关键路径上；31 µs 的 s2g 泄流是 DRAM 受限（r21 ncu：DRAM 57%、占用 16.2%）。这是“barrier 数与发射顺序杠杆在 2 CTAs/SM 不付钱、只有 DRAM 流量类杠杆还能动 f1”这一调度规律的第 3 个数据点，**核内串行化类就此关闭**。

### 5.1.7 F1 时间图谱与端到端关闭账目

r21 的十个可组合删除探针（`GDN_F1T_*`，×3 会话 ×40 对；输出按设计为垃圾、永不进 verify）给出 f1（~227 µs 基准 @T8192）的表观相分布：

| 相（删除探针） | 表观成本 | 占 f1 |
|---|---|---|
| uw 段（gemm+epilogue+s2g，NUW） | **116.5 µs** | **51.2%** |
| uw 段+operand staging（NUST） | 131.6 µs | 57.9% |
| kkt 整段（NKT） | 59.5 µs | 26.2% |
| solve 整体（NALL） | 41.3 µs | 18.2% |
| solve phase-1（NP1） | 32.5 µs | 14.3% |
| u/w s2g 拷贝单独（NC2G） | 31.0 µs | 13.6% |
| gc 扫描+β（NSC） | 17.6 µs | 7.8% |
| solve phase-2（NP2） | 14.6 µs | 6.4% |
| solve phase-0（NP0，已折叠为 F1ZF） | 8.5 µs | 3.7% |
| gc_out 存回（NGC） | 0.8 µs | 0.3% |

派生拆分：uw staging ≈ 15.1 µs（NUST−NUW）、uw mma+epilogue+staging 等待 ≈ 85.5 µs（NUW−NC2G）、u/w 全局存 ≈ 31.0 µs（NC2G）。单相占比和 ~109%——相之间存在 DRAM/L2 争用，每个 δ 是该相串行成本的**上界**。r21 ncu（独占、锁频 1.57 GHz）：f1 313.95/312.06 µs、DRAM 56.8–57.7%（400.8 GB/s）、占用 16.24%（8 warp/SM：寄存器限 2 × smem 限 2）、2048 CTA/82 SM = 12.49 波、发射间隔 5.8 周期（无合格 82.6%）、warp 停顿构成 long_scoreboard 24.3% / barrier 22.9% / wait 17.8% / math_pipe 12.4%——**f1 在 2 CTAs/SM 上是延迟受限而非带宽/张量受限**。

r23 之后 f1 端到端关闭：uw 串行化类（F1EP wash）、s2g（DRAM 受限）、staging（persistent-k 池表：一块 $[64,136]$ bf16=17,408 B 的常驻 k 需要横跨 kkt→uw 存活，而该窗口内唯一死字节是 kt 的 6,144 B，**差 11,264 B**——任何别名排列都撞活范围重叠，需 ≥11.3 KB 池重构而非旋钮）、kkt（屋顶线）。残余：phase-1 串行 32.5 µs 与扫描 17.6 µs，各有两轮诚实负结果（F1WY/F1P4；SCANDB 已硬化），重开需新机制类。

---

## 5.2 F2 "scan"：状态递推 kernel

F2 承载全算子唯一的跨 chunk 串行依赖：$S_{n+1}=2^{\mathrm{gc}_{\text{last},n}}S_n+(\,k\odot2^{\mathrm{gc}_{\text{last}}-\mathrm{gc}})^{\top}(u_n-w_n S_n)$。最终形态（`gen_f2`，pp 变体）：grid $(B\!\cdot\!H\!\cdot\!VT)$，每 CTA 一个 (head, V-tile) 的 $[K,\mathrm{BV}]$ 态 tile 常驻共享内存，`scf.while` 走 $NC$ 个 chunk；每 chunk 三个动作——**vp gemm**（$\mathrm{acc}=w\,S$，A=乒乓半体的 wb，B=常驻 S）+ **vn epilogue**（$v_{\text{new}}=(u-\mathrm{acc})\,2^{\mathrm{gc}_{\text{last}}-\mathrm{gc}_r}$，kterm 折叠约定：F3 的 P epilogue 用配对的 $2^{\mathrm{gc}_i-\mathrm{gc}_{\text{last}}}$，全局 $v_{\text{new}}$ 原样携带缩放后的 tile——r1 的原地 RMW kterm 扫pass因此消失）+ **sd gemm 与 S 更新 epilogue**（$S\leftarrow S\cdot2^{\mathrm{gc}_{\text{last}}}+k^{\top}_{\text{term}}v_{\text{new}}$，在 mma epilogue 里完成 RMW）。入口态 `h_all[n]=S` 与 `v_new`/`ht` 的全局存取全部走合并 cell 扫pass（fragment 映射的原位全局存是 4 B 跨距上的 2 B 存，实测比多一次扫pass**慢 40%**——r3 记录在案的盲巷，反向结论同样适用于 o/u/w 段）。

### 5.2.1 机制：chunk 级乒乓双缓冲与“组 A”预取（以及 wait_group-0 的机器约束）

**动机。** 串行 chunk 链上，下一 chunk 的 $w/k/\mathrm{gc}$（以及后来的 $u$）若同步装载，装载延迟全部落在链上。

**机器约束（源码级证明，决定实现路线）。** 树内 `NVGPU/Transforms/CreateAsyncGroups.cpp` 对每段由 `create_async_groups` 转换的 staging **立即**插入 `cp.async.wait_group 0`（“Create the group and wait for it right after”）——transform 路线**结构上**无法让预取与计算重叠。手动发射的 `nvgpu.device_async_copy` 原样透传，等待位置归生成器。

**实现（`gen_f2` pp 变体）。** 乒乓奇偶折进**索引算术**（memref 值不能被 `arith.select`）：`wb` 是 $C\times2K$ 沿列对分、`kb` 是 $2C\times K$ 沿行对分、`gcs` 是 $2C$ f32。chunk $n$ 顶部发出“组 A”（chunk $\min(n{+}1,NC{-}1)$ 的 w/k/gc 进奇偶 $n\hat{}1$ 半体），与 chunk $n$ 的**全部**计算重叠；chunk 末 `device_async_wait numGroups=0` + barrier 翻转奇偶。

**IR（chunk 循环头与奇偶算术，`r22ship/payload.mlir`）**：

```mlir
%nwr = scf.while (%nw = %c0) : (index) -> (index) {
  %ncc = arith.cmpi ult %nw, %c32 : index
  scf.condition(%ncc) %nw : index
} do {
^bb0(%n: index):
  %n2 = arith.remsi %n, %c2 : index
  %pp = arith.subi %c1, %n2 : index            ; 预取奇偶 = 1 - n%2
  %nm = arith.minsi %np1, %c31 : index         ; min(n+1, NC-1)
  %pK = arith.muli %n2, %c128 : index          ; 读半体列基
  %qK = arith.muli %pp, %c128 : index          ; 预取半体列基
  ...
  %acp = nvgpu.device_async_copy %w[%ix_8], %wb[%idv, %ix_9], 8 bypassL1
      : memref<4194304xbf16> to memref<64x264xbf16, ...>   ; 组 A：w
  %acp_9 = nvgpu.device_async_copy %k[%ix_39], %kb[%ix_40, %im_24], 8 bypassL1
      : memref<4194304xbf16> to memref<128x136xbf16, ...>  ; 组 A：k^T
  ...
```

**效果（诚实归零 + 真根因移交 §5.2.2）。** 乒乓单独立项时**零增益**：471→473 µs（r3，“计划中的杠杆 1 不是故事”）——因为 F2 当时不是装载延迟受限，而是被 §5.2.2 的 LDSM 冲突卡死。乒乓结构保留（它给 F2US/W32P 提供了载体）；在产 pp 体每 chunk 2 条 barrier（r5 审计对“~8/chunk”认知的修正——那是对 r2 旧体的描述，全核含序共 3 条）。sd gemm 的 B 读按 N 维 warp 局部（`ldm_b` 列 = 本 warp 自己的 8 列，与 vn epilogue 写同一 warp 同列——mma N 瓦片恒等式），epilogue 与 sd 之间免一条 barrier；h_all/v_new 的跨 warp 消费已由屏障覆盖。

### 5.2.2 机制：LDSM +16 B 垫——8 路 bank 冲突的体系级修复

**动机（本体系最大单笔杠杆的根因）。** ncu（乒乓 F2）：发射间隔 63% 停在“MIO 队列满”、每调度器 2.0 活跃 warp、每 SM ~40 周期一条 LDSM、每 warp 每 chunk 76 条 LDSM。所有被 ldmatrix 读的 tile 行距恰为 128 B 的整数倍（$64\times128$→128 B、$64\times256$→512 B…）：LDSM 象限的 8 行全落进**同 4 个 bank = 8 路冲突 ≈ 8× 慢的 LDSM**。r2 时 43% 的 MIO 停顿即此。

**修复。** 每个被 ldmatrix 读的 2-D tile 一律 **+8 个 bf16 列**（=+16 B）：行距变为 128 B+16 B 的倍数，8 行依次轮换扫过全部 32 bank；+16 B 保持 cp.async 的 16 B 对齐。发运 payload 里每个 LDSM-tile 都带这个垫（`wb [64,264]`、`kb [128,136]`、`S [128,40]`、`vn [64,40]`、f1 的 `Abf [64,72]`、f3 的 `P [64,72]`……）。

**效果。** F2 473→**249 µs**（w8/bv64，r3）；配合 w4/bv32 重调（§5.5）达 83 µs 静窗。同一垫顺手用于 f1/f3 全部 LDSM tile（收益在彼处被别的瓶颈掩盖——诚实记录）。同轮 F2 顺带**撤下** u 的 staging（vn epilogue 直接从全局读，其延迟藏进 vp gemm 的 fragment 装载之后；省 8 KB smem + 2 cp.async + 每 2 元素 1 条共享操作）——这一“撤”后来又被 F2US（§5.2.3）以更高形态逆转。

### 5.2.3 机制：F2US——u 段重回乒乓预取管线

**动机。** r3 的“u 直接从全局读”在 T8192 的账：每 chunk 每 CTA 4 KB 的 u，fragment 映射下是 **2 B 装载 × 4 B 跨距**——每个 32 B 扇区只交付一半字节，4 KB 放大成 ~8 KB 扇区、聚合 ~32 MB→64 MB，且每一发装载都坐在串行链（vp mma→epi_vn→sd mma→S 更新）上，L2 延迟直接暴露。删除探针定价这部分 ~104 µs @T8192。

**实现（`GDN_F2US`，发运开启）。** u 的 VT 切片重新 staging——但不是回同步扫pass，而是进**组 A**：行乒乓 tile `ut [2C, BV+8]`（+10,240 B；池 84,480→94,720 B，仍 1 CTA/SM），随 w/k/gc 一起 chunk 顶部发出、chunk 末既有 wait 收割；epilogue 读无冲突的垫化共享内存。字节与算术顺序不变 ⇒ 位级一致（T8192 门与 verify 差分精确复现）。

**IR（epi_vn 的 u 读，`r22ship/payload.mlir`）**：

```mlir
%vnu = memref.load %ut[%ix_307, %nct_vp_0] : memref<128x40xbf16, ...>  ; F2US
%vnf = arith.extf %vnu : bf16 to f32
%vnd = arith.subf %vnf, %evn : f32            ; u - (w@S)
%vnz = arith.subf %gcl, %vnr : f32            ; gc_last - gc_r
%vne = math.exp2 %vnz : f32
%vns = arith.mulf %vnd, %vne : f32            ; kterm 折叠
%vnt = arith.truncf %vns : f32 to bf16
memref.store %vnt, %vn[%vrow, %col] : ...
```

**效果。** 配对 A/B（×60 对）：f2 @T8192 **0.4017→0.2972 ms（−104.5 µs，0/60 更慢）**、staged e2e −96.8 µs；@B4T2048 f2 −101.2 µs（0/60）、e2e −91.4 µs。T8192 正式终判随之从 1.021×（平手）越线 1.101×（r14，n=30，p≈1e-9）。资源门（§5.5）：w8/bv64 下 ut 翻倍使池 115,200 B > 101,376 B，ptxas 拒绝——F2US 因此是环境旋钮 + 按形状构建，不是无条件默认。

### 5.2.4 机制：W32P——半 warp 行重映射的无冲突宽存储（与两轮否定后的幸存路线）

**动机。** r15 删除探针把 f2 的两条 cell 扫pass定价 **h_all 35.4 µs + v_new 17.9 µs**（T8192 的 297 µs 链内，合计 ~53 µs 天花板）。WST 的 32 位宽存储（§5.3.3）在 f1/f3 付了钱，但在 f2 直接套用是**负的**：BV=32 tile 每行只有 16 对，一个 warp 的 32 条 b32 共享读要跨两行，而 (BV+8)=40 元素=80 B=20 字的行距使 lane 28–31 与 lane 0–3 撞同一 bank（2 路冲突；仅映射即 +4.3 µs，47/60，r15）。lane 级配对（`GDN_WSP`：保无冲突单格读、`shfl.sync.bfly` 取伴 lane、`prmt` 打包、偶线程谓词 b32 存）更惨：**+157.5 µs（60/60）**——SASS 上 stock 循环四路展开带四条独立批量 `STG.E.U16`（满 MLP），WSP 循环是每次迭代一条 `WARPSYNC→LDS→SHFL→PRMT→STG` 串链，跨 lane SHFL 依赖禁止展开且每迭代加一个汇聚点；而理论节省本就是零（谓词半 warp b32 存与全 warp b16 存发射相同 warp 指令数、搬相同 64 B）。

**W32P 的银行算术（`GDN_W32P`，发运开启）。** 保持配对读存（`[base]`/`[base+2]`，`gdn_smem_fix` 的 `widen_stores` 结构性融合照常触发），只改**行映射**：lane 0–15 读行 $r=8\cdot\mathrm{round}+w$，lane 16–31 读行 $r+4$。字地址：行 $r$ 的 16 对 = 字 $[20r,\,20r{+}16)$；行 $r{+}4$ = $[20r{+}80,\,20r{+}96)$；$80\bmod32=16$，两个半 warp 恰好落进**互补的两个 16-bank 半区**——对每个 $r$ **零冲突**。双射覆盖：轮 $j$、warp $w$、半区 $h$ → 行 $8j+w+4h$（$h\in[0,8)$ 覆盖行 $8j..8j{+}7$、对 lane%16 各一次）。纯索引算术：同字节、同行内配对、同全局地址 ⇒ 位级一致；池/tile/LDSM 垫/占用全部不变；f2 的 smem_fix 宽存计数 0→2。

**IR（发运 T8192 payload 的 h_all 扫pass头部，`gen_fwd/B1T8192_mono/payload.mlir`，即 +W32P 记录代）**：

```mlir
%w3h  = arith.cmpi uge %lane, %c16 : index            ; 半区判定
%w3h4 = arith.select %w3h, %c4, %c0 : index           ; +4 行偏移
%ix_160 = arith.addi %warp, %w3h4 : index             ; 行槽 = warp + 4·hi
%irm_38 = arith.remsi %lane, %c16 : index
%im_90 = arith.muli %irm_38, %c2 : index              ; 对列 = 2·(lane%16)
%wir_2 = scf.while (%wi = %tid) : (index) -> (index) {
  %wic_2 = arith.cmpi ult %wi, %c2048 : index
  scf.condition(%wic_2) %wi : index
} do {
^bb0(%wib: index):
  %idv_38 = arith.divsi %wib, %c128 : index           ; 轮 j = i>>7
  %ix_161 = arith.addi %im_91, %ix_160 : index        ; 行 = 8j + warp + 4·hi
  %sgl = memref.load %S[%ix_161, %im_90] : ...        ; b16 ×2 → smem_fix 融合为 b32
  %sgl_2 = memref.load %S[%ix_161, %ix_162] : ...
  ...
```

**效果。** f2 配对（翻转双装载、×40 对、17 会话）：T8192 **−17.9..−22.6 µs ×7**（慢对率 5–8/80）、B4T2048 −9.7..−25.6 µs ×5、T2048 −2.0..−3.8 µs ×5；链中位全程三形状为正（T8192 −9.1..−12.0 µs ×5、B4 −14.3/−27.5、T2048 −3.2/−5.2）。~6–8% of f2 @T8192，正落在纸面带内（WST 在 f1 付 0.176 µs/MB → f2 的 100.6 MB 扫pass写 ~17 µs 天花板）。门：T8192 fp64 门**位级一致**、fwd_verify 4/4 记录数字、det N=5、占用不变（f1 2 / f2 1 / f3 1 blocks/SM）。附带诚实披露：+W32P 的 T8192 构建抽到一个可复现的 f3 +11.8..+12.6 µs 模块布局税（f1/f3 PTX 节与纯度构建字节相同——f2 的 ~+300 B PTX 触发 r19 定位的模块摆放抽签类，f3 ±10.5±6 µs）；链净值仍每形状每会话为正，n=30 存档重掷该布局。r25 存档终判即在此代上读出（§5.0 的 2.1250×/1.2211×/1.3493×）。

### 5.2.5 机制：段切分扫描——M-复合交接的数学与 Amdahl 式诚实否定

**动机。** T8192 的 F2 是 128-chunk 串行链、64 CTA。教科书解法是两遍段切分：把 $NC$ 个 chunk 分成 $S$ 段，先并行算每段的出口态再带真入口态重跑。

**数学（先证精确，再证为何免费）。** F2 的 chunk 递推对态是**仿射**而非态无关：$S'=M_nS+b_n$，$M_n=2^{\mathrm{gc}_{\text{last}}}I-k_{\text{term}}^{\top}w$。一般精确并行分裂需要交接式 $E_{sg+1}=M_{sg}E_{sg}+c_{sg}$（每 chunk 一次额外 $K\times K\times C$ gemm + 每段 ~7 个 $K^3$ 复合，~3.4× F2 flops、88 KB smem、1 CTA/SM）。但每个因子 $M_n$ 携带全 chunk 衰减（kterm 里的 $2^{-\mathrm{gc}}$ 与 $w$ 里的 $2^{+\mathrm{gc}}$ 逐 token 相消）：

$$\lVert M_{seg}\rVert \;\sim\; 2^{\sum \mathrm{gc}_{\text{last}}}\cdot O(1) .$$

GDN 基准配方 $g=-\mathrm{softplus}(0.5x)$、平均 log 衰减 ~−0.25/token ⇒ 64-token chunk 衰减 ~$e^{-16}$；在真实基准输入上精确 fp64 复合实测 $\lVert M_{seg}\rVert\le3.6\times10^{-13}$（L=1，T512/S=8）与 $4.3\times10^{-52}$（L=4，T2048/S=8）（`tools/g5i_split_sim.py`）——被丢弃的 $M_{seg}E$ 项远低于 bf16 分辨率（~4×10⁻³ 相对），**免矩阵交接 $E_{sg+1}=c_{sg}$ 在本契约的输入下精确**。有效性边界由 `--stress-g 0.02` 探针划出：近零净衰减（gate≈1，段衰减 ~$e^{-1.3}$）时 $\lVert M_{seg}\rVert$ 涨到 0.07–0.45、免矩阵交接入口误差 3.1×10⁻²（会破 5×10⁻³ 门），精确复合变体仍 3.7×10⁻⁴——M-复合回退**已设计未实现**，记录为遗留。

**实现（`--f2-segments S`，mono 专用）。** `f2a`（grid ×S）：每段从**零入口**跑 NC/S 个 chunk，只把出口态写 `@seg_exit`（零入口下内部 h_all/v_new 是错的、不写）；`f2c`：每段带真入口（段 0 用 h0，其余用 `@seg_exit[sg-1]`）重跑并写真输出。体 = r2 流水化形（`emit_kloop` scf.for 保 transform 流水）+ r3 垫 + r3 撤 u。单缓冲态 tile：43,264 B、206/230 寄存器 ⇒ **2 CTAs/SM 共驻**——本来的第二杠杆。

**数值（全绿，含一段近失教训）。** 4/4 PASS（T512+T2048，分段与串行同裕量）；新增 T8192 fp64 门：$o\,6.85\times10^{-4}$/$\mathrm{ht}\,7.90\times10^{-4}$（7× 裕量），且 **S=1/4/8 的 o 与 ht 位级相同**。近失教训：f2a 起初跳过段 0（“其出口无人用”——错，段 1 的入口就是 $c_0$），`seg_exit[0]` 保持零初始化、h_all[2] 垃圾而 ht 仍 PASS（入口误差被衰减吞掉）——按 chunk 二分用中间量定位。

**效果（Amdahl 式诚实否定）。** 全形状**更慢**：staged T8192 1.298→1.541/1.507 ms（S=4/8）、B4T2048 1.155→1.556（S=4）。两笔账：(i) 第二遍全价——f2a 是一遍免存储的 chunk 全工、f2c 重复全部工再加存储，2× 工只换 ~1.2× 执行 ⇒ 16–33% 净损失；(ii) **Amdahl 17%**——f2c 单独 @S=8 是等工探针（全部 NC chunk、16-chunk 链、2 CTAs/SM）：0.529 ms vs 串行 0.640 ms，链长+共居合计只值 **F2 的 17%**（111/640 µs），其余 83% 是每 chunk 吞吐（staging 流量、mma 发射、barrier、epilogue）——而分裂恰恰复制这些。r3 之后的 F2 已是每 chunk 吞吐受限（64 CTA≈78% SM、每 chunk 5.0 µs 对静窗 T2048 的 2.6 µs——差值是 T8192 更冷的 w/u/k staging，不是链）。推论：**任何扫描重构（段、波、链式 CTA）在这些形状上赢不了串行 kernel**；发运默认 S=1。G5-F 的 ~250 µs 投影把 650 µs 串行 F2 当延迟受限建模（650×2.7/8），测量否决了该模型。

### 5.2.6 F2 负结果全家福与关闭

除 §5.2.4–5.2.5 外，F2 的其余攻击面全部按测量关闭：
- **F2SB 单缓冲节食**（`--f2 sb`）：wb/kb/gcs 单缓冲化把池压到 50,432 B（2 CTAs/SM 可容纳），预取折进 tile 死点（wb 死栅栏在 vp 后、kb+gc 在底部）——**+21.2 µs**（57/60）；强制 64 KB L1 carveout **+20.9**（60/60，carveout 救援假说被否）；B4T2048（256 CTA、2-CTA 共驻真实生效、164 常驻）**+28.5**（56/60）。根因：pp 的组 A 在 chunk 顶部发出、重叠**全部** chunk 计算；每个单缓冲折叠都把传输移到部分计算**之后**（死栅栏绑定），暴露成本超过 2-CTA 延迟隐藏的回收——即便真 2× CTA 网格、即便强制 carveout。
- **F2BAR 2→1 barrier/chunk**（vn 行乒乓 + v_new 扫pass移下迭代顶）：**+19.7 µs**（55/60，r5）——+5 KB smem（84.5→89.6 KB）挤掉 u 装载/存储扫pass依赖的 L1 carveout，代价超过省下的 bar.sync。r5 审计同时纠正了一个在产认知：**发运 pp 体每 chunk 2 条 barrier**（f2 全核 3 条含序），“~8/chunk”是 r2 旧体的数字——barrier 杠杆在 round 5 开始前已基本花完。
- **w4/bv64（唯一在预算内的每 warp N=16 形态）**：f2 **+218.4 µs**（60/60，r15）——更宽每 warp N 付不起砍半的 CTA 并行。
- **WSP lane 级配对**（§5.2.4 内）：+151..157 µs，双表面（tile 级 +4.3、lane 级 +151）皆死后才轮到 W32P。
- **128 位全局存路**：`vector.load/store`（含 `vector<8xbf16>→vector<4xi32>` 位投掷变体）**能降过**本管线但元素分裂为 b16 op（PTX `st.global.b16 ×8`，探针）——扫pass的 64 B/warp 标量存已是该管线的最优；h_all 亦已近 64 MB 写带宽地板（~35 µs @ ~1.8 TB/s）。

F2 的终局读法（r24/r25）：写路径上 W32P 是最后一个位级一致杠杆；扫pass剩余成本是 DRAM 写受限（100.6 MB @T8192）——F2 与 F1 一样以“算法最小流量 + 实测屋顶线”关闭。

---

## 5.3 F3 "o"：输出级 kernel

F3 从 F2 的产物（`h_all` 入口态、`v_new`）重算每 chunk 的输出：$o=(2^{\mathrm{gc}}\!\odot\!(q\,S_{\text{entry}})+P\,v_{\text{new}})\cdot K^{-1/2}$，$P=\mathrm{tril}_{\text{inc}}((q\,k^{\top})\,2^{\mathrm{gc}_i-\mathrm{gc}_{\text{last}}})$（与 F2 的 kterm 折叠加起来恰还原数学式）。grid $(NC\!\cdot\!B\!\cdot\!H,\,VT)$，发运取 $\mathrm{BV}_3=V$（VT=1）。

### 5.3.1 机制：qk+oi 融合循环

**动机。** $qk^{\top}$ 与 $q\,S_{\text{entry}}$ 共享 A 操作数 $q$。两条 K-循环 → 一条：一个 staging 的 q tile 同时喂两个 gemm（`emit_kloop` 的 gemms 列表——机制层面的“一个操作数服务两个消费者”在 kkt（k 双侧）与 qo（q 双侧）各出现一次）；k tile 以 `kt=True` 读、S 切片按 k 主序从 `h_all` staging。随后 **P epilogue**（f32 域折 $2^{\mathrm{gc}_i-\mathrm{gc}_{\text{last}}}$、`cmpi sle col,row` 含对角的下三角选择——上三角单元被写成**精确零**，这使 F3 天然没有 ZFIX 类的未初始化暴露）写 bf16 `P`；**oa 循环**（A=常驻 P、B=staging 的 v_new 切片）+ **o epilogue** 跨两组累加器（oi 的 $2^{\mathrm{gc}}$ 行缩放从 staging tile 移进 f32 epilogue——少一次舍入）求和、乘 $K^{-1/2}$、截断。

**效果。** r2 的三重奏重构里这一融合循环与 BV=128 是 f3 的形态基础（239 µs @B1T2048 w8 时代即 DRAM 57–69%、31% 占用、~50 MB @ ~0.4 TB/s）；其后 f3 的每次变动都是下述两杆秤的称量。

### 5.3.2 机制：BV 选择——bv3=V 的单 tile 论证与 bv3=32 的否定

**动机。** F3 的 q/k 装载量按 VT 倍增（每 VT3 CTA 重复整个 qk gemm 与 q/k staging）。

**权衡与测量。** 发运 $\mathrm{bv3}=V=128$（VT=1）把 q/k 重读减半以上；代价是单 CTA 更大的 v_new/st tile。反例已测：**bv3=32**（f3 池 45,432 B、8192 CTA、2 CTAs/SM）f3 **+357.2 µs**（60/60）——qk gemm 与 q/k staging 每 VT3 CTA 重复一次，占用输给重复工。f3 的占用天花板另有其人：240 寄存器本可 2 CTAs/SM（240×128×2=61,440 ≤ 65,536）但 smem 82,176 B 不容（2×82,176 > 100 KB），`.maxnreg` 注入工具（`GDN_MAXREG_K` per-entry）就位后确认**寄存器帽动不了占用**——纠正了 r14 简报的一个预判。

### 5.3.3 机制：F3O 与 WST——o 段 restage 和 32 位宽存储

**动机（与 F1UW 第 3 刀同源）。** o epilogue 的 fragment 映射全局存是离散 2 B 写（16 行 × 每线程 4 列对）；restage 到垫化 `ot [64,136]` tile + 一次合并 cell 扫pass把它们变 64 B/warp 行段（G5-F 负结果的逆转：原位全局存输给额外扫pass）。136 列（272 B）行距同时保持 STS 侧无冲突（4 字轮换）。**WST**（`GDN_WST`，f3/f1 侧发运开启）再砍一半指令：生成器扫pass发**成对单元** $(2i,2i{+}1)$（2×2 b16 读写四元组、显式 +1 邻接），`gdn_smem_fix.py` 在 PTX 级把每个四元组融合为 `ld.shared.b32 + st.global.b32`（字节精确——32 位字就是那两个 bf16 单元），LSU 指令与循环迭代减半。守卫 `cols ≥ 64`（f1 的 128 列行保证每 warp 在一行内；f2 的 32 列不满足——它的解是 §5.2.4 的 W32P）。

**效果。** F3O：f3 −1.3 µs @T8192 / **−8.4 µs @B1T2048**（22/60 更慢——原则性正确、量级噪声）；与 F1UW 的 u/w 扫pass合并看，WST 付 f1 −11.8/−11.4/−10.6 µs（T8192×2、B4；1–5/60 更慢）、staged e2e −9.0/−13.4/−7.0 µs，数值位级一致（T8192 门 $o/\mathrm{ht}=0.000e{+}00$）。r19 均衡复测确认 WST 的 f3 效应 ~0（原始 +15.2 µs 读数是 B 侧摆放抽签——A/B 器械史里重要的一课：`r14u_ab` 的 B 侧偏差被根因为**模块构建顺序的摆放抽签**，f3 的 ~160 MB 工作集上第二个认领 `.global` 区的模块付 ~6%；均衡双装载（镜像构造序 + 逐对交替发射序 + 丢弃首对 + warmup 20）成为默认器械）。

### 5.3.4 机制：F3KS 2-CTA 转化——带机制分解的诚实否定

**动机。** f3 池 82,176 B @1 CTA，是 B4 链上第二大 kernel；F1KS 的 playbook（bk=16 K-循环节食 + `ot` 落 `P` 的死 arena）看起来可直接移植：qt/kt `[64,24]` 多缓冲 6,144 B ×2、st/vnt `[16,136]` 8,704 B ×2、`ot` 别名进 P 的死字节（P 的 malloc 以 ot 的 17,408 B 尺寸申报、`ot` 别名 +0、oa 循环后一条显式 barrier 钉住 P 的最后 ldmatrix 先于 ot 首写——流水化 transform 会重组循环尾，r15 `bisect_shift` 的教训）。池 82,176→47,360 B ≤ 50,688，2 blocks/SM。审计先行：f3 无扫描（无 SCANDB 暴露）、无清零 cell 循环（epi_P 全覆盖写含上三角零、无 ZFIX 暴露）。门全绿：强 1-CTA 位级同、2-CTA ×10 复跑位级干净（两形状）、三重奏 7/7 位级同、B4 ×3 位级同、verify 4/4、det N=5。

**效果（否定 + 分解）。** T8192 f3 **+96.1 µs**（74/80）、staged +97.0；B4T2048 f3 +74.9（76/80）。分解臂定音：**同一份 bk16 码强制 1 CTA = +115.6 µs**——K-循环重排本身 ~+110 µs（3 段 qo 循环 + oa 循环的迭代数翻倍：barrier/wait/staging 发射步全部翻倍），2-CTA 驻留只买回 ~20 µs（116→96，尚不含 +8.7 同构抽签校正）。机理：f3 @T8192 流 ~120 MB 的 `h_all`/`v_new`/`o`——**带宽受限 kernel 不为波数减半付钱**（f1 的 −53.5 µs 是延迟受限情形的正面对照）。`GDN_F3KS` 默认关、不发运；f3 至此在 1 CTA 上以带宽受限关闭（r20）。

---

## 5.4 融合旗舰：单 persistent kernel

### 5.4.1 结构：chunk 循环内的八段 mma 与终身共享 smem

`gen_fused` 把全部工作装进 grid $(B\!\cdot\!H\!\cdot\!VT)$ 的单 kernel：每 CTA 持 $[K,\mathrm{BV}]$ 态 $S$ 常驻共享内存，`scf.for %n` 走 NC 个 chunk，每 chunk 依次八个 mma 位点——**kkt**（$A$）、**ug**（$u$）、**wg**（$w$）、**vp**（$wS$）、**qk**（$P$ 前）、**oi**（$qS$）、**oa**（$P\,v_{\text{new}}$）、**sd**（$S$ 更新）——加一个 flat solve（`emit_solve`，thread-per-column 的 L1 逐字递推）。99 KB 硬顶下的 tile 预算靠**终身共享**：`QK` tile 先装 bk2（$\beta k 2^{\mathrm{gc}}$）、wg 消费后**晚装** q；`TP` 先装转置 $T$、再装 $P$；`BVN` 先装 $\beta v$、epilogue 原位覆写 $v_{\text{new}}$。r14 的刷新（`GDN_FLAV1`，BV≤48 默认开）把 r3 的 LDSM 垫（k_t/QK/S/w_t/TP/BVN +8 列、`A_m` +1 f32 列）、F1 的 Hillis-Steele 扫描、F3O 式 o restage 全部移植进 fused，池 @w4/bv32 = **99,072 B**（w8/bv64 的垫化池 114,672 B > 101,376 B 装不下——刷新以 BV≤48 为门，记录在案的几何门）。

**IR（发运 fused payload 的 chunk 循环与首批 cp.async 组，`gen_fwd/B1T2048_fused/payload.mlir`，即 FPIPE+SCANDB 代）**：

```mlir
scf.for %n = %c0 to %c32 step %c1 {
  ...
  %acp = nvgpu.device_async_copy %k[...], %k_t[...], 8 bypassL1 : ...  ; k(n)→k_t
  ... (1024 条 16B 拷贝)
  %agr = nvgpu.device_async_create_group %acp, ..., %acp_8
  %acp_9 = nvgpu.device_async_copy %v[...], %BVN[...], 8 bypassL1 : ... ; 原始 v(n)→BVN
  %agr_2 = nvgpu.device_async_create_group %acp_9, %acp_10
  ... (gc 扫描与 k 的 wait 并行)
  nvgpu.device_async_wait %agr numGroups = 0
  ...
  nvgpu.device_async_wait %agr_2 numGroups = 0                            ; solve 后
  ...
  %mr_oa_0_0_0 = nvgpu.mma.sync(%fa_oa_0_0, %fb_oa_0_0, %acc_oa_0_0)
      mmaShape = [16, 8, 16] : ...                                        ; 八段之一
```

### 5.4.2 机制：FPIPE——同步 staging 的三段 cp.async 改造与“k 一次读”

**动机。** 串行 chunk 链上有三段同步 global→smem cell 扫pass（k、$\beta v$、q）外加 bk2 对同一 k 的**第二次全局读**（每 chunk 16 KB）。这些扫pass在管线里降成 LSU 码、直接坐在链上。

**改造（`GDN_FPIPE`，fused 发运开启；池不变 99,072 B，payload +127 行）**——三段异步化 + 一次 k：

1. **k(n)→k_t 与原始 v(n)→BVN 在 gc 扫描之前发出**（两组 token；两 tile 自 sd(n−1) 消费其上一内容起已死）；k 的 wait 放 kkt 前（重叠整个扫描），v 的 wait 放 solve 后，$\beta$ 以一次**共享内存原地缩放**扫pass施加（与 `pre_bv` 完全同算术——位级一致）。
2. **bk2 变纯 smem→smem 缩放扫pass读常驻 k_t**——k 每 chunk 从全局读**一次**而非两次。
3. **q(n)→QK 在 wg 的 barrier 之后发出**（QK 自 wg 消费 bk2 起已死），qk 之前 wait——重叠整个 vp gemm。

**效果。** fused T8192 **6274→4530 µs（−27.8%）**，三会话 ×40 交错对全部 0/40 更慢（配对中位 −1744/−1750/−1736 µs）；向 T2048/B4 复制为 **−26.5%**（1555.5→1143.3；6216.2→4581.9 µs，同样 0/40 ×3）。数值：对 stock fused 全程位级一致（$o/\mathrm{ht}=0.000e{+}00$）、verify 4/4、det N=5。增益同时是延迟隐藏与指令节食（−28% 远超纯延迟模型的 ~−10%）。r17 的 PTX 审计顺带把 mono 三重奏的同类机会**关闭为负**：trio 的全部 global→smem staging 已是 cp.async（f1 30 op、f2 42、f3 42），仅存的同步全局读（f1 的扫描输入 512 B、f3 的 gc 256 B）没有任何可重叠窗口——FPIPE 类是 fused 路径特有（其串行链背了三条同步扫pass + 一次 k 重读，trio 的 kernel 边界从未有过）。

### 5.4.3 跨 chunk 乒乓的算术不可行：2,304 B 余量对 17,408 B tile

**动机。** FPIPE 拿下核内重叠后，剩下的串行暴露是 chunk 间边界——跨 chunk 的 k/q 乒乓预取似乎是下一步。

**算术（r16 一次性写清，生成器注释同源）。** 器件 opt-in 上限 $101{,}376$ B（查询：`MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`；102,400 B/SM、82 SM），池已 99,072 B ⇒ 余量 **2,304 B**；一块 k/q 乒乓 tile（$[64,136]$ bf16）= **17,408 B**。每个别名排列都撞活范围重叠（`w_t` 的 $[8..10]$ 行窗口对乒乓 tile 的整 chunk 驻留）。要买第三块 k 尺寸 tile 必须先付 bf16-solve 节食 + 垫回撤——代价超过收益。**跨 chunk 预取就此以算术关闭**；r14 的重分类叙事被修订一半：链不是不能抗核内重叠（FPIPE 吃掉 28%），但整 chunk 预取需要的是重构而非旋钮。

### 5.4.4 诚实重分类：staged mono 扛旗

fused 路径的结构性天花板与占用无关，而与**并行度**有关：F1 的 WY 工作在 trio 里按 (NC, B·H) 全并行（T8192 = 2048 CTA），在 fused 里塌缩进 64-CTA 的串行 chunk 链——结构性差异不能靠移植 tile 修复。终局数字：fused（FPIPE+SCANDB，`cb2520d8` 等 r18 代）T8192 ~4.5 ms 对 mono 三重奏 **0.785 ms**（约 5.7×）；T2048 1.143 ms 对 0.187 ms。r14 起 fused 被诚实重分类为**设计记录**（design record），全部三个判定形状的旗舰标准由 staged mono 三重奏承担；fused 的存在价值是它证明了同一 payload 形态可以在两个融合极端间由 `--path` 一个生成器旋钮切换、以及 FPIPE 类机制只在它身上存在的定量原因（§5.4.2 的审计）。

---

## 5.5 形状自适应 dispatch 表

生成器的“自适应”分两层：**显式构建参数**（`--warps/--bv/--bv3/--f2/--f2-segments` + 环境旋钮集）按形状由发运程序选择并连同 `gdn_cache_key`（全部 `GDN_*`/`G5*` 旋钮 + 生成器源哈希）戳进 `full_spec.json`——缓存键不匹配即重定向 `_k<hash8>` 新目录，杜绝陈旧 cubin 伺服（r3/r19 两次事故后的制度）；**结构性守卫**折在生成器内部，按 tile 几何自动启用/拒绝。最终发运表：

| 形状 | 构建参数 | 环境旋钮集 | f1 grid | f2 grid | f3 grid | 判定（r25 存档 n=30） |
|---|---|---|---|---|---|---|
| small B1T512 H4 | mono, w4/bv32/bv3=128 | 同船集（dev-loop 验证形状；分段构建 s4/s8 在档） | (8, 4) | (16, 1) | (32, 1) | 数值门形状（无独立计时判定） |
| B1T2048 H16 | mono, w4/bv32/bv3=128 | `GDN_F1KS GDN_F2US GDN_WST GDN_ZFIX GDN_SCANDB GDN_F1ZF GDN_W32P` | (32, 16)=512 | (64, 1) | (512, 1) | **2.1250×**（0.1866 ms） |
| B4T2048 H16 | 同上 | 同上 | (32, 64)=2048 | (256, 1) | (2048, 1) | **1.2211×**（0.8213 ms） |
| B1T8192 H16 | 同上 | 同上 | (128, 16)=2048 | (64, 1) | (2048, 1) | **1.3493×**（0.7851 ms） |

守卫与门（每条都是本文某节的结论）：

1. **F2US 池律**：`ut` tile 在 w8/bv64 下使 f2 池 115,200 B > 101,376 B（ptxas 拒绝）——F2US 永不为无条件默认，按形状构建（r14 实测的精确失败数）。
2. **WST 的 `cols ≥ 64` 规则**：f1（V/K=128 列）与 f3（128 列）扫pass自动配对融合；f2（32 列）被规则排除，由 **W32P**（`cols==32 ∧ rows%8==0 ∧ nthr==128`，恰为 f2 的 S/vn tile）接手。
3. **F1KS 需要 nthr=128**（4 warp）与 bf16-solve；**F1ZF** 需要串行 phase-1 臂 + nthr=128；**F1EP** 只栈在 F1KS 上——旋钮间偏序由 `raise SystemExit` 硬约束。
4. **fused 的 FLAV 刷新门 BV≤48**（垫化池 114,672 B 在 w8/bv64 装不下）；**FPIPE** 池中性、全形状 fused 发运。
5. **f2 的 grid 上限几何**：BV=64 再瓦解（option a）死于 r14 池律 + 网格减半（64→32 CTA/82 SM = 39% 覆盖）双杀；W32P 是其唯一幸存支线（option b，r24 关案）。
6. **段切分 `--f2-segments` 仅 mono**、且 $S$ 必须整除 NC；发运恒为 $S=1$（§5.2.5 的否定）。
7. **f3 恒 bv3=V**（§5.3.2）；F3KS 不发运（§5.3.4）。

---

## 5.6 本章机制清点（收尾索引）

| # | 机制 | 旋钮/默认 | 关键配对数字 | 节 |
|---|---|---|---|---|
| 1 | gc Hillis-Steele 扫描 + SCANDB 双缓冲 | `GDN_SCANDB` / 开 | 损坏 4 K→0（10/10 位级）；扫描 17.6 µs | 5.1.1 |
| 2 | kkt K-循环（transform 流水 + 线程跨距 staging） | 结构默认 | 拆分 40.6/13.0/2–4 µs；826 对 765 GB/s 屋顶线关闭 | 5.1.2 |
| 3 | 分块 16×16 solve（对角 16 步 + 非角 stash/consume mma） | 结构默认 | F1 429→270 µs 时代；phase-1 = 14.3% | 5.1.3 |
| 4 | bf16-solve（F1KS 前提） | `GDN_BF16SOLVE` / 隐含开启 | 池 −16,384 B；数值 $6.626\times10^{-4}$ 类 | 5.1.3 |
| 5 | F1WY / F1P4（phase-1 重构双否定） | 默认关 | +92.6 / +14.8 µs ×3 | 5.1.3 |
| 6 | F1UW（单趟 + 预解 staging + 死 tile 复用） | 结构默认 | f1 −123.3 µs @T8192（4/60） | 5.1.4 |
| 7 | F1KS（K-分裂 + 终身 arena + bk16，2 CTAs/SM） | `GDN_F1KS` / 开 | f1 **−53.5 µs**；池 46,352 B | 5.1.5 |
| 8 | ZFIX（清零步长全覆盖） | `GDN_ZFIX` / 开 | 70 万→4 K bit 差（与 SCANDB 合用归零） | 5.1.5 |
| 9 | F1ZF（清零折进空闲线程） | `GDN_F1ZF` / 开 | −7.7/−7.9/−6.7 µs ×3；NP0 界的 91% | 5.1.6 |
| 10 | F1EP（单趟 epilogue，wash） | 默认关 | +0.0..+0.8 µs；核内串行类关闭 | 5.1.6 |
| 11 | chunk 乒乓 + 组 A 预取（wait_group-0 约束） | `--f2 pp` 默认 | 单独零增益（471→473）；载体保留 | 5.2.1 |
| 12 | LDSM +16 B 垫（8 路 bank 冲突） | 结构默认 | F2 473→249 µs；体系级 | 5.2.2 |
| 13 | F2US（u 重回预取管线，`[2C,BV+8]` 行乒乓） | `GDN_F2US` / 开 | f2 **−104.5 µs**（0/60）；T8192 越线 1.101× | 5.2.3 |
| 14 | WSP/WST-on-f2 双否定 | 默认关 | +157.5 / +4.3 µs | 5.2.4 |
| 15 | W32P（半 warp 行重映射） | `GDN_W32P` / 开 | f2 −17.9..−22.6 µs ×7 @T8192 | 5.2.4 |
| 16 | 段切分扫描（M-复合数学 + Amdahl 否定） | `--f2-segments`，发运 S=1 | $\lVert M_{seg}\rVert\le3.6\times10^{-13}$；链+共居=17% | 5.2.5 |
| 17 | F2SB / F2BAR / bv64 / bv3=32 否定 | 默认关 | +21.2 / +19.7 / +218.4 / +357.2 µs | 5.2.6 |
| 18 | qk+oi 融合循环 | 结构默认 | f3 形态基础 | 5.3.1 |
| 19 | F3O（o 段 restage） | 结构默认 | f3 −8.4 µs @T2048 | 5.3.3 |
| 20 | WST（成对单元 + PTX b32 融合，`cols≥64`） | `GDN_WST` / 开 | f1 −10.6..−11.8 µs；LSU 指令减半 | 5.3.3 |
| 21 | F3KS（2-CTA 否定 + 分解） | 默认关 | f3 +96.1（bk16 ~+110、2-CTA 买回 ~20） | 5.3.4 |
| 22 | fused 八段 mma + 终身共享 smem（FLAV 门） | `GDN_FLAV1` / BV≤48 | 池 99,072 B | 5.4.1 |
| 23 | FPIPE（三段 cp.async + k 一次读） | `GDN_FPIPE` / fused 开 | fused −26.5..−27.8%，位级一致 | 5.4.2 |
| 24 | 跨 chunk 乒乓算术关闭 | — | 2,304 B 余量 vs 17,408 B | 5.4.3 |
| 25 | 诚实重分类（staged mono 承担判定） | `--path mono` | fused 4.53 vs mono 0.785 ms @T8192 | 5.4.4 |

数字与 cubin 出处：`evidence/G5/fwd_report.md`（各机制段）、`evidence/G5/gen_fwd/{B1T8192,B4T2048,B1T2048}_mono/`（+W32P 记录代）、`*_mono.r22ship/`、`bench_B1T2048H16_mono.stale396b/`（pre-F1KS 对照）、`B1T2048_fused/`（FPIPE 代）、`*_mono_s{2,4,8}/`（段切分）、`final_adjudicate_r25f_w32p.json`（判定）、`fwd_verify_r24f2.json` / `fwd_verify_r18f_f1ks.json`（数值门）。生成器：`tools/gdn_full_gen.py`（4464 行）。
