## 二十七、精度也是调度约束：Split-BF16 的三链与代数折扣

反向 GEMM 常消费 fp32 中间量，而大数相消会把读取侧的舍入放大到输出。精度选择因此会改变 tile 数、共享内存池、MMA 数量和寄存器生存期（live range）。Split-BF16 用一个 hi/lo 分量对保留更多尾数，再以三条 bf16 张量核乘法和 fp32 累加器恢复足够细的乘积分辨率；它带来的调度成本必须与数值收益一起核算。

### 1. 指数域相同，尾数分段

bf16、tf32、fp32 都有 8 位指数；显式尾数分别为 7、10、23 位，计入隐含位后的有效精度分别为 8、11、24 位。直接转 bf16 损失的是有效数字，动态范围并没有因指数位减少而同比缩小。

对正规范围内的 fp32 数 x 定义：

```text
hi = bf16(x)
lo = bf16(x − float32(hi))
```

hi 提供首段约 8 位有效数字，lo 接着表示残差。两个分量的有效位段合计仍能放进 fp32 的 24 位精度，所以 hi+lo 这次 fp32 加法自身可以精确完成；hi+lo 与原值 x 之间仍保留第二次 bf16 截断的误差。这个区分决定了后续误差账不能写成“完全恢复原值”。

以绝对值位于 [1,2) 的数为例，单路 bf16 的相邻间隔是 2^-7，hi 的最大舍入距离约为 2^-8。残差进入更小的指数区间后再由 lo 量化，有效尾数得以延伸约 8 位，距离 fp32 的 24 位仍有差距。反向中间量还常由若干 O(10) 的项相消为 O(0.1)，直接 bf16 暂存曾在相关位点造成 4.6e-2–6.9e-2 的绝对误差。分量对的价值正是在这种相消前保留细粒度，而非扩大指数范围。
两个分量数相乘时展开为：

```text
(ah + al)(bh + bl)
= ah·bh + ah·bl + al·bh + al·bl
```

前三项各走一条 bf16 MMA，累加器保持 fp32。al·bl 是被三链舍弃的二阶小项；单值 split 的表示残差可到约 2^-17，而三链省略项和分量舍入通常按约 2^-16 的乘积误差类估计。dkb 的端到端 dk 从 tf32 的 1.04e-2 降至 1.73e-3，约 6 倍；2^-11 与 2^-16 的量级差只是误差类比较，不能直接当成最终输出倍率。两个理论量级也不能当成所有输出的固定比值，最终输出仍须经过端到端容差门。这里不能把某个被省略项孤立成全部误差来源，三链之后的每一次累加仍在 fp32 网格上舍入。

### 2. 两遍拆分把共享数据依赖写清楚

`_split_bf16` 第一遍生成 hi 并写入共享内存，第二遍重读原值和 hi，再生成 lo。两道 `gpu.barrier` 分别隔开 hi 写读与 hi/lo 的后续消费：

```python
e.l(f"scf.for %sh = %tid to {e.ic(R * Cc)} step %nt {{")
# 省略 i、j 的线程跨步下标
v = load2(e, src, i, j)
if cvec is not None:
    v = op2(e, "mulf", v, ldsc(e, cvec, j))
e.l(f"%sht = arith.truncf {v} : f32 to bf16")
e.l(f"memref.store %sht, {hi.name}[{i}, {j}] : {hi.ty}")
e.depth -= 1
e.l("}")
e.l("gpu.barrier")
e.l(f"scf.for %sl = %tid to {e.ic(R * Cc)} step %nt {{")
# 省略同一组 i、j 下标
v = load2(e, src, i, j)
if cvec is not None:
    v = op2(e, "mulf", v, ldsc(e, cvec, j))
h = load2(e, hi, i, j)
r = op2(e, "subf", v, h)
e.l(f"%slt = arith.truncf {r} : f32 to bf16")
e.l(f"memref.store %slt, {lo.name}[{i}, {j}] : {lo.ty}")
e.depth -= 1
e.l("}")
e.l("gpu.barrier")
```

[tools/gdn_bwd_gen.py:1398-1425，省略循环深度调整和重复下标展开]

读这段发射器要跟踪三个值：f32 的 v、bf16 的 hi、由 v−hi 得到的 lo。`truncf` 出现两次，第二次的输入已是残差，所以 lo 不会简单重复 hi。第一道屏障把共享 hi 的生产与消费分开，第二道屏障让完整分量对先于 MMA 的跨 warp 读取。即使两遍当前采用相同线程映射，显式屏障也把可见性合同固定在 IR 中，避免后续调度改变映射后引入隐蔽竞态。

可选 cvec 在拆分前执行一次 f32 缩放，适用于任意尺度；此时恒等式作用对象是“缩放后再舍入”的 v，两条分量共享同一次缩放误差。拆分本身还需要一对带 16B 垫行的 bf16 tile，资源消耗已经进入了第二十六章的 bank 与驻留账。

### 3. 三链调用必须落到三组真实操作数

dkb 对 vnew 与 dh 各生成一对 hi/lo，再把三条 GEMM 串入同一逻辑累加器。可选 VTF 分支为不同 v 切片准备两组累加器；默认顺序仍保持 hh、hl、lh：

```python
_split_bf16(e, vh, vl, vngb, C, BV)
_split_bf16(e, hh, hl, cur, K, BV)
if VTF:
    tgt = maccs1 if (vt % 2 == 1) else maccs0
maccs = mma_gemm(e, f"dkBhh{vt}", C, K, BV, 8, vh.name, vh.ty,
                 hh.name, hh.ty, accs=tgt if VTF else maccs)
maccs = mma_gemm(e, f"dkBhl{vt}", C, K, BV, 8, vh.name, vh.ty,
                 hl.name, hl.ty, accs=maccs)
maccs = mma_gemm(e, f"dkBlh{vt}", C, K, BV, 8, vl.name, vl.ty,
                 hh.name, hh.ty, accs=maccs)
```

[tools/gdn_bwd_gen.py:4246-4258]

读调用顺序即可还原代数：hh 使用 vh·hh，hl 使用 vh·hl，lh 使用 vl·hh，没有第四条 ll。`accs=maccs` 让后一条链接收前一条的 fp32 累加结果。VTF 的 tgt 只改变 v 切片求和的结合方式，不改变每个切片内部的三项分解；它会引入另一种浮点求值次序，必须单独过数值与确定性验证。

真实 PTX 能把这种对应关系核对到寄存器组：

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {%r336, %r337, %r338, %r339}, {%r280, %r281, %r282, %r283},
    {%r332, %r333}, {%r300, %r301, %r302, %r303};
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {%r340, %r341, %r342, %r343}, {%r280, %r281, %r282, %r283},
    {%r334, %r335}, {%r304, %r305, %r306, %r307};
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {%r344, %r345, %r346, %r347}, {%r284, %r285, %r286, %r287},
    {%r332, %r333}, {%r308, %r309, %r310, %r311};
```

[evidence/G5/scratch/r21b_ks2_T8192/dkb/kernel.ptx:336-350]

A 片段 {%r280–283} 为 v_hi，{%r284–287} 为 v_lo；B 片段 {%r332,333} 与 {%r334,335} 分别为 h_hi、h_lo。三条因此恰为 hi·hi、hi·lo、lo·hi。C 操作数组承接上一 k 段对应链的 D，说明 fp32 累加跨缩并段延续；本段 D 使用新寄存器号属于目标码寄存器分配，追踪依赖时不能只比较寄存器名字。

dkb 的实测把数值与性能同时落地：内核由 0.96 ms 降至 0.25 ms，dk 最大误差为 1.73e-3，与标量 fp32 路径的 1.731e-3 几乎重合；同一位点的 tf32 曾达到 1.04e-2，越过 8e-3 门限。这说明三链的乘积误差控制在该输出上足够，但不能外推为所有位点都足够。

{{FIG5}}

### 4. 两条代数性质各删去一条链

若尺度只依赖输出行 i、与缩并下标 kk 无关，就有 `Σ s_i·a_i,kk·b_kk,j = s_i·Σ a_i,kk·b_kk,j`。dvp_bh 的 k 原本已是 bf16，行尺度可移到 epilogue，只需拆 dh 侧：

```python
gr = ldsc(e, gc, row)
d = op2(e, "subf", gcl, gr)
xs = op2(e, "mulf", x, exp2f(e, d))

accm = mma_gemm(e, f"dvbh{vs}_{ks}", C, BV, BK, 4,
                a_src, a_ty, bhi.name, bhi.ty)
accm = mma_gemm(e, f"dvbl{vs}_{ks}", C, BV, BK, 4,
                a_src, a_ty, blo.name, blo.ty,
                accs=accm, epilogue=epi_bh)
```

[tools/gdn_bwd_gen.py:7768-7770,7784-7788，省略 epi_bh 的累加写回]

读点是 a_src 指向原始 bf16 k，bhi/blo 只来自 dh，尺度在两条链结束后才乘。三链由此降为两条，少一对 A 侧 split tile，也少一次分量乘积。尺度移出缩并会改变一次 f32 舍入的位置，所以“代数可移”仍需输出误差验证；原始 k 无需再次量化则是该项优化的精确性基础。

第二条恒等式针对 2 的幂尺度。在不发生上溢、下溢或次正规损失的范围内，乘 2^d 只平移指数，不改变有效数字的舍入格点：

```text
split(v·2^d) = split(v)·2^d
```

`stage_split_T` 在转置暂存时把该尺度折进 hi 与 lo：

```python
v = gload(e, src_g, gty, [m, _gcol_int(e, i, j, cm, col0)])
if cvec is not None:
    x = ldsc(e, cvec, s)
    v = op2(e, "mulf", v, x)
if vec is not None:
    x = ldsc(e, vec, j)
    v = op2(e, "mulf", v, x)
e.l(f"%spuh = arith.truncf {v} : f32 to bf16")
e.l(f"memref.store %spuh, {hi.name}[{j}, {s}] : {hi.ty}")
hw = e.v("spuw")
e.l(f"{hw} = arith.extf %spuh : bf16 to f32")
r = op2(e, "subf", v, hw)
e.l(f"%spul = arith.truncf {r} : f32 to bf16")
e.l(f"memref.store %spul, {lo.name}[{j}, {s}] : {lo.ty}")
```

[tools/gdn_bwd_gen.py:956-969；恒等式合同见 939-949 行]

这段 fused helper 在寄存器中保留 hi 的 widened 值，一次扫描便能算 lo；它与前述两遍共享源 helper 的形状不同，读代码时不应机械套用同一实现。历史 caryA 利用 2 的幂恒等式把 Bhat 行尺度折进 A 侧 split，B 侧退化为原始 bf16 k 转置，四个共存 tile 随之退役。归档结果为 5.00× 提速、共享池 82,176B→28,592B、寄存器 255→76。[evidence/G5/bwd_report.md:1827-1853] caryA 已在当前默认门控分支退役，这些数字用于解释代数折扣的收益来源，不属于八内核终态的现行资源表。

### 5. 数值、性能、流量要分别结账

**数值账**看最终输出与参考的距离。乘积残差只是预算入口，后续相消、前缀和与携带链会放大或吸收它。tf32 在 dkb 的超门结果说明 10 位尾数没有普遍适用性；其他位点若输入原本就是 bf16，tf32 反而增加无收益的量化。Split-BF16 也应逐链验证，尤其在改变 v 切片结合顺序或移动尺度时。

**性能账**包括三条 MMA、两次分量暂存、屏障和累加器生存期。它的对照对象是标量 fp32 的每乘加多条指令，以及特定位点可用的 tf32；三链的乘积吞吐必须按三次发射核算，不能由单条 bf16 MMA 的峰值直接推算。行尺度与 2 的幂折叠删除一条链，值更大的收益常来自同步减少 tile、寄存器和驻留压力。

**流量账**没有随尾数补偿下降。一个 fp32 值变成两个 bf16，合计仍为 4B；写入和读取都要搬运 hi、lo 两路，共享内存还要容纳两份 tile 与垫列。Split-BF16 改善乘积精度并启用张量核，不节省存储字节，也不能缓解由中间量流量决定的带宽下界。把三本账并列后，精度选择便显出其调度含义：每一次数值改写都要重新核对 MMA 数、池、寄存器、屏障和输出裕度。
