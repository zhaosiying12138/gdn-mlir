## 二十四、前向机器设计：三 Kernel 边界与瓶颈驱动的优化

前向算子同时含有 chunk 内代数、跨 chunk 状态递推和大流量输出重算。三者的依赖方向与资源需求不同，机器设计先把它们拆成 F1、F2、F3，再让每个 kernel 的 grid、共享内存和流水结构服从自己的瓶颈。chunk 长 C=64，NC=T/C；V 方向按 BV 切成 VT 份。

### 1. 三个 kernel 各自拥有一条清晰的依赖边界

| kernel | grid | 核心职责 | 并行结构 |
|---|---|---|---|
| F1（wy） | (NC, B·H) | gc 扫描、kkt、A/T 求解、u/w 预计算 | 每个 CTA 处理一个 chunk 与一个 batch-head |
| F2（scan） | (B·H·VT, 1) | 状态跨 chunk 递推，写 h_all、v_new 与终态 | 不同 batch-head 和 V-tile 并行，chunk 链在 CTA 内串行 |
| F3（o） | (NC·B·H, VT) | 由 h_all、v_new、q/k 重算并写输出 o | chunk 与 V-tile 恢复并行 |

F1 的五段职责直接写在生成器契约中[tools/gdn_full_gen.py:1827-1838]；F2 的 group-A 预取、vp、vn、sd 和状态更新顺序也有明确定义[tools/gdn_full_gen.py:2631-2649]。F3 将 qk 与 oi 放在同一 K-loop，再让驻留 P 与 v_new 做 oa，输出含行尺度与最终缩放[tools/gdn_full_gen.py:3687-3695]。这条边界把唯一的全序列递推限制在 F2，F1、F3 都能利用 NC 方向的并行度。

单 cubin 多入口（mono）形态把三个零参数入口和一套公共全局缓冲装进一个 cubin。F1 的 `gc_out` 在装配时改名为 `gc`，三个入口按 F1、F2、F3 的顺序生成[tools/gdn_full_gen.py:3909-3921]。同步合同只有几行：

```python
def wrap_module_mono(kernels, params, meta):
    """One module, one set of public globals, three zero-arg kernels.
    Kernel boundaries are the stream-order syncs; the staged path then runs
    without any device-to-device handoff copies."""
```

[tools/gdn_full_gen.py:3925-3928]

同一 stream 的 kernel 边界提供完成顺序：F2 启动时 F1 的公共全局写已经可见，F3 同理等待 F2。公共缓冲取消分立方块之间的 device-to-device 交接拷贝，kernel 边界保留算法所需的可见性；同步没有搬进复杂的跨 CTA 信号协议。

### 2. fused 的 4.53ms 解释了并行度为何优先于融合

`--path fused` 是持久（persistent）单 kernel：grid 为 B·H·VT，状态 S 驻留共享内存，chunk 循环在核内完成全部工作。它减少入口边界并保住状态，却也把 F1 的 chunk 并行折进每条状态链。T8192、B1、H16、BV32 时，三 kernel 的 F1 有 128×16=2048 个 CTA；fused 只有 16×4=64 个 CTA，每个还要串行走过 128 个 chunk。搬运同样的 tile 或增加局部异步流水，无法消除这项结构性差距[evidence/G5/fwd_report.md:982-994]。

归档 fused 经核内异步 staging 后达到 4.530ms，输出保持位级一致，资源池仍为 99,072B[evidence/G5/fwd_report.md:1393-1415]；最终 mono 在正式测量中为 0.7851ms@T8192[evidence/G5/fwd_report.md:3299-3308]。两个数字来自不同归档窗口，适合说明设计量级，不应伪装成同一次配对实验。4.53 对 0.785 的差距与 grid 分析相互印证：减少 kernel 边界带来的收益，不足以弥补 WY 工作失去 chunk 级并行。

fused 仍有保留价值。它提供状态全程驻留的融合极限参照，可独立检验核内 staging、数值与确定性，也能量化“再多核内重叠还能买多少时间”。正式性能路径由三 kernel 承担，fused 则让融合边界成为可测量对象，而非只凭设计偏好争论。

### 3. F1KS：46,352B 换来双 CTA，收益来自延迟隐藏

F1KS 将三类空间压缩叠加起来：求解阶段去掉 f32 的 A 镜像，uw 的 v/k 操作数按 K 分成两个半程并复用半尺寸 tile，求解结束的 Abf/Tt 字节再作为后两者的生存期竞技场；kkt 的 bk 降为 16，同步缩小多缓冲 k tile。声明合计 46,144B，经 ptxas 对齐补 padding 后报告 46,352B，低于 49,152B 的静态双 CTA 门槛[tools/gdn_full_gen.py:1933-1943,4399-4409]。加入扫描双缓冲的硬化版本报告 46,608B，仍在同一驻留档内[evidence/G5/fwd_report.md:1834-1840]，两个数字对应不同缓冲配置，不能混写。

半 K 的真实 payload 使用 32×136 的 vt：

```mlir
%acp = nvgpu.device_async_copy %v[%ix_223], %vt[%idv_5, %im_31], 8 bypassL1
  : memref<16777216xbf16> to memref<32x136xbf16, #gpu.address_space<workgroup>>
%acp_2 = nvgpu.device_async_copy %v[%ix_227], %vt[%idv_6, %im_33], 8 bypassL1
  : memref<16777216xbf16> to memref<32x136xbf16, #gpu.address_space<workgroup>>
```

[evidence/G5/gen_fwd/B1T8192_mono/payload.mlir:2255-2264，后续拷贝省略]

这里要看目的 tile 已缩到 32 行，源地址仍按原逻辑单元计算，装载保持 16B 对齐。`136` 不是任意 padding：它由 K=128 加 8 列垫行组成；K=32 的切分边界位于完整 m16n8k16 原子之间。两个半程复用同一组 `[32,136]` tile，accumulator 从第一半程传入第二半程，f32 加法顺序不变，因此位级等价有构造依据[tools/gdn_full_gen.py:1939-1943]。空间收益来自复用和缩小缓冲区，不靠改变归约顺序换取。

把累加关系写成伪代码会更清楚：

```text
acc = 0
for half in [0, 1]:
    stage_half_tile(half)              # 复用同一个 [32,136] tile
    for kb in range(half*32, (half+1)*32, 16):
        acc = mma(acc, A[kb:kb+16], B[kb:kb+16])
```

`acc` 不在半程边界清零，两个半程的 k 区间也不重叠；这正是“缩小 tile 而不改变数学归约”的检查点。

驻留翻倍有效，是因为 F1 的画像属于延迟受限。独占剖析记录 long_scoreboard 占 24.3%、barrier 占 22.9%，DRAM 利用约 56.8%，调度器 No-Eligible 达 82.6%；更多独立 CTA 能填补等待与同步空洞[evidence/G5/fwd_report.md:2457-2466]。T8192 上 F1 从 278.7µs 降至 225.2µs，减少 53.5µs[evidence/G5/fwd_report.md:1834-1840]。数字验证的是“延迟受限时提高驻留”这条机制，不能推出所有 kernel 双 CTA 都会提速。

### 4. F1ZF：把 8.5µs 清零工作放进空闲线程

F1 的分块求解先清零转置输出中代入不会写入的单元，再由前 64 个线程处理对角块递推。递推期间线程 64–127 空闲，清零却原本排在它前面并独占一段屏障间隔。基础清零函数保持简单：

```python
def zero_up(i):
    r = divsi(e, i, cC)
    c = remsi(e, i, cC)
    gt = e.v("svgt")
    e.l(f"{gt} = arith.cmpi sgt, {r}, {c} : index")
    e.l(f"scf.if {gt} {{")
    e.l(f"  memref.store {zbf}, {Tb}[{r}, {c}] : {Tb_ty}")
    e.l(f"  memref.store {zbf}, {Tg}[{r}, {c}] : {Tg_ty}")
    e.l("}")
```

[tools/gdn_full_gen.py:1366-1375]

F1ZF 让空闲的上半线程并发执行这项工作，并用一条 barrier 取代原来的两段分隔。折叠后的块级谓词只写串行代入不触碰的单元；代入路径会写同块全部单元，包括负零上角，所以两个写集保持不相交，输出位级一致[tools/gdn_full_gen.py:1341-1358]。

删除 phase-0 的探针给出约 8.5µs 上限，代表整段工作完全消失；并发折叠在 T8192 的三次配对读数为 −7.7、−7.9、−6.7µs，约兑现上限的 91%[evidence/G5/fwd_report.md:2473-2488]。余量来自清零仍然消耗的指令、存储和同步资源。删除用于估价，折叠是在估价之内重新安排执行，两种数字的关系正好检查优化是否接近其理论边界。

### 5. F2 垫行：先消除共享内存的几何放大

F2 曾把大量时间花在 `ldmatrix` 的共享内存读侧。共享内存有 32 个 bank，每个 4B 宽；128B 行距会让同一组 8 行反复落进相同的 4-bank 组，形成 8 路冲突。为每个被 `ldmatrix` 读取的二维 bf16 tile 增加 8 列，即每行多 16B，行使起点逐行旋转 4 个 bank，8 行可覆盖全部 32 个 bank；16B 增量也保住异步拷贝的对齐[tools/gdn_full_gen.py:2676-2683]。

这项修改没有减少矩阵运算，却撤掉了每次片段装载的硬件串行化。归档 w8/bv64 配置中，F2 从 473µs 降至 249µs[evidence/G5/fwd_report.md:340-356]。它说明“DRAM 流量不大”仍可能存在存储瓶颈：共享内存 bank 几何会在数据抵达寄存器之前放大延迟，剖析中的 MIO queue full 比全算法 FLOP 数更能指出修复位置。

### 6. F2US：把 u 段从串行全局读放回乒乓预取

F2 的 vn 尾声曾按 MMA fragment 映射直接从全局读取 u。每个 lane 读 2B、相邻有效单元呈 4B 跨距，32B 扇区只有一半字节被使用；每 chunk、每 CTA 的 4KB 逻辑数据由此放大为约 8KB 扇区流量，而且装载延迟位于 vp→vn→sd 的串行链上[evidence/G5/fwd_report.md:996-1012]。

F2US 增加 `[2C,BV+8]` 的 ut 行乒乓 tile，让 u 与 w、k、gc 共用 chunk 顶部发出的 group-A 异步预取，在 chunk 末统一等待。空间从 84,480B 增至 94,720B，仍保持单 CTA；读侧变为带垫共享内存：

```mlir
%vnu = memref.load %ut[%ix_311, %nct_vp_0]
  : memref<128x40xbf16, #gpu.address_space<workgroup>>
%vnf = arith.extf %vnu : bf16 to f32
%vnd = arith.subf %vnf, %evn : f32
```

[evidence/G5/gen_fwd/B1T8192_mono/payload.mlir:4157-4159]

`u − (w@S)` 的 extf、subf 和后续运算顺序未变，改变的是同一字节的来源与到达时间。T8192 的 F2 从 0.4017ms 降至 0.2972ms，减少 104.5µs；同批归档的形状判定从中位数比值 1.021 提升到 1.101[evidence/G5/fwd_report.md:954-964,1060-1069]。这项优化同时移除扇区放大与串行延迟暴露，因此收益远高于“多了一次共享拷贝”的直观成本。

它仍受形状约束。w8/bv64 下 ut 加倍，池达到 115,200B，超过 101,376B 的每块 opt-in 上限，ptxas 会拒绝；所以 F2US 按构建形状启用，不能成为无条件默认[evidence/G5/fwd_report.md:1014-1022]。

### 7. 删除探针决定下一步该看哪一段时间

优化对象来自时间去向，而非指令表面。删除探针保留同步骨架和共享内存存活结构，移除某段计算或存储，再与出货形态交错计时；这些变体输出无效，只能用于定价[tools/gdn_full_gen.py:1842-1886]。T8192、约 227µs 的 F1 基线上，归档读数如下：

| 被删除的阶段 | 表观时间 |
|---|---:|
| uw：gemm、尾声与 u/w 写回 | 116.5µs |
| kkt：K-loop 与 A 尾声 | 59.5µs |
| solve 整体 | 41.3µs |
| gc 扫描与 beta 阶段 | 17.6µs |
| solve phase-0 | 8.5µs |

[evidence/G5/fwd_report.md:2429-2448]

各项不能直接相加。删除一段也会移除它与其他阶段的带宽、发射和延迟重叠，所以读数是表观成本与串行成本上限；合计超过总时间正好暴露这种重叠。可靠的用途是排序：uw 是最大段，kkt 次之，单独攻击一条小规模尾声很难改变整核。

{{FIG4}}

进一步拆分 kkt，k-tile staging 搬运算法所需 33.55MB，表观 40.6µs，折合 826GB/s；同一共载窗口的 D2D 读写合计标定为 765GB/s[evidence/G5/fwd_report.md:2887-2917]。在当前几何下，这段流已贴近器件带宽屋顶，且字节数是算法最小流量。继续做 tile 微调之前，必须先回答还能减少哪些字节，或者能否改变计算结构。

### 8. F3 的负收益封住了“驻留越高越好”的直觉

F3KS 也把共享池从 82,176B 压到 47,360B，达到双 CTA，位级、确定性和重复运行检查均通过；性能却反向变化：T8192 的 F3 增加 96.1µs。将同一份 bk16 代码强制回单 CTA，增加量为 115.6µs，说明双 CTA 只挽回约 20µs，K-loop 改造本身付出了约 110µs。更多 barrier、wait 和 staging 发射随迭代数增加，覆盖了驻留收益[evidence/G5/fwd_report.md:2231-2258]。

F3 在 T8192 搬运约 120MB 的 h_all、v_new 与 o，属于带宽受限。增加驻留可以隐藏延迟，却不能减少这些字节，也不能免除新增循环管理指令。F1KS 的 −53.5µs 与 F3KS 的 +96.1µs 因此并不矛盾：前者给延迟空洞填入独立工作，后者在带宽路径上增加开销。

瓶颈驱动的方法到此处闭合。延迟受限时购买并行驻留，共享读侧冲突时修改几何，串行链上的窄读改为提前批量 staging，算法最小流量贴近屋顶时停止微调。负收益变体保留数值证据和默认关闭的开关，让下一次设计能从已测边界出发，而不必重新相信一条未经定价的直觉。
