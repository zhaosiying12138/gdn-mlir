## 三十三、数据通路：staging、乒乓、warp 专化与 TMA

数据从全局内存流进共享内存、再进张量核，这条通路的每个环节——装载指令的选择、缓冲的组织、完成的通知方式、发射指令的主体——都藏着结构性选择，选错一处的代价动辄是整段流水失效。最终系统的数据通路分四组机制：手动异步 staging、乒乓预取、数据完整性规则、warp 专化与 TMA。术语先定义：staging（全局到共享内存的搬运）、cp.async（异步拷贝指令，每条搬 16 字节）、乒乓（双缓冲按奇偶交替读写）、warp（32 线程的执行单位）、TMA（Tensor Memory Accelerator，按张量描述符整块搬运的硬件单元）、mbarrier（共享内存中的到达-等待屏障对象）。

### 1. 手动 cp.async：发射与等待位置的归属

异步搬运的第一选择本是树内 transform 的 create_async_groups，但它对每段 staging 在组建后立即插入排空等待——预取与计算的重叠在结构上不可能。生成器因此直接发射 nvgpu.device_async_copy：每个线程按跨步下标认领一个 16 字节块，token 交调用方分组，等待位置归调用方决定：

```
el = 8 if dt == "bf16" else 4            # 每块 8 个 bf16 = 16B
nch = Cc // el;  tot = R * nch
for kk in range(tot // nthr):
    p = tid if kk == 0 else tid + kk * nthr          # 线程跨步认领
    r  = p // nch;  cc = p % nch                     # 行 / 块列
    gcol = r * cm + cc * el                          # 全局列地址
    t = nvgpu.device_async_copy src[m, gcol], dst[r, dcol], el bypassL1
    toks.append(t)                                   # token 交调用方分组等待
```

［tools/gdn_bwd_gen.py:1687-1714（astage_tile 节选）；组队与等待封装在 1728-1740 行］

落到真实 PTX，地址按线程跨步加 16、32、48、换行加 256——每线程一次搬自己认领的块：

```
cp.async.ca.shared.global [%rd5],  [%rd549], 16;
cp.async.ca.shared.global [%rd14], [%rd549+16], 16;
cp.async.ca.shared.global [%rd15], [%rd549+32], 16;
cp.async.ca.shared.global [%rd16], [%rd549+48], 16;
cp.async.ca.shared.global [%rd17], [%rd549+256], 16;
```

［evidence/G4/gen_verify/kkt_bf16_mma_p0_t64x64x32_w4/kernel.ptx:606-610；bypassL1 对应 .cg 变体见同目录 p2 配置］

三条发射纪律以真实代价换得。其一，发射必须静态展开，token 才在调用方作用域存活。其二，线程数必须与启动核一致——失配曾让部分块无人拷贝、半陈旧 tile 入算。其三，排空等待是每线程语义，每个 wait 之后必须 barrier，跨 warp 的读者才能看见落盘数据。

### 2. 乒乓预取的两个实例

实例一，前向 F2 的组 A。串行 chunk 链上，chunk n 顶部发出 chunk n+1 的 w、k、gc（后来加 u）装载，与 chunk n 的全部计算重叠；chunk 末统一等待加屏障，翻转奇偶。奇偶折进索引算术（memref 值不能按条件选择，能选择的只有下标）：

```
%n2 = arith.remsi %nw, %c2 : index
%pp = arith.subi %c1, %n2 : index            ; 预取奇偶 = 1 − n%2
```

［evidence/G5/gen_fwd/B1T8192_mono/payload.mlir:3718-3719；发射侧 issue_group_A 在 tools/gdn_full_gen.py:2731-2744，w、k、u、gc 四个 tile 同组发出］

实例二，反向携带链的 A 窗口。dhs 每步一个 K×K 乘 K×BV 的 GEMM，整块 A staging 要 64KB 共享内存；有效形态是把 A 当流处理：滚动 K×32 的 f32 窗口双缓冲，超前一个窗口预取，窗口只喂标量共享读。窗口之前还有一步指令级并行的铺垫：输出胞格按线程划分后每线程持有 16 个独立累加器，一次 b 装载服务全部累加器、16 次 a 装载相互独立可重叠——dhs 先靠它从 0.83 降到 0.59 毫秒，窗口再把 0.59 降到 0.32：

```
for w in range(NW):                                  # NW = K // BKW
    if w + 1 < NW:                                   # 超前一个窗口发射
        astage_tile(e, G["A_all"], aty, "%m", K, (w+1)*BKW, AW[(w+1) % 2], ...)
        gw = agroup(e, toks)
    rgemm_aw(e, Xa, AW[w % 2], w * BKW, X, K, BV, BKW, acc_buf=Xa)
    e.l("gpu.barrier")                               # 本窗口读毕才允许重 staging
    if w + 1 < NW:
        await_all(e, gw); e.l("gpu.barrier")
```

［tools/gdn_bwd_gen.py:3571-3585（rgemm_aw 契约）；窗口轮转发射在 gen_dhs 内］

效果：dhs 从 0.59 降到 0.32 毫秒。值等价以对照链证明——同一上游 cubin、窗口开关两种构建，dh_all 与 dh0 逐位相同：窗口值是同批全局元素的原始字节拷贝，乘加链的次序未动。进阶形态是跨块预取：最后一个窗口把下一块窗口 0 发进此刻空闲的缓冲，异步组的 token 以循环携带变量进入下一迭代，下一块开头的等待被整整一块的工作覆盖。两个实例共享一条复用栅栏律：缓冲的下一次重写之前，上一轮的读者必须已被一道尾栅栏收口，复用距离不足两次迭代的写法要显式补栅栏——共享缓冲的合法性从来以读毕为前提。

### 3. launder_pass：数据完整性的边界规则

本工具链存在一条静默失效模式：cp.async 落地的 tile 经普通标量共享读读回完全正确（对全局源逐位相等，转储验证），经 ldmatrix 读回错误——错的行块随构建调度而定、逐构建确定、对数值容差不可见（验证门以 dk 1.34、dv 2.18、dβ 25.7 的量级抓住）。成文规则由此立下：每个喂 ldmatrix 的 tile 必须由标量共享写或逐元素 staging 写入；异步直落的 tile 只许喂标量共享读。launder_pass 是两个世界的桥——在首个 ldmatrix 读者之前，对已落盘区域做恰好一遍恒等重写：

```
def launder_pass(e, buf, R, Cc):
    """identity st.shared rewrite of a cp.async-landed tile region -- the
    r6 rule demands ldmatrix operands be st.shared-written (NEVER raw
    LDGSTS->LDSM), so a raw-landed tile gets ONE element-wise
    ld.shared/st.shared pass over exactly the staged cells before its
    first ldmatrix reader ... Values bit-identical by construction."""
    e.l(f"scf.for %lu = %tid to {e.ic(R * Cc)} step %nt {{")
    v = memref.load buf[i, j];  memref.store v -> buf[i, j]
```

［tools/gdn_bwd_gen.py:1743-1757，循环体节选；值按构造逐位相同——存回的就是读出的 v］

### 4. warp 专化与 TMA：生产者-消费者分工

流水线的极端形态是分工进 warp：一个生产者 warp 只做装载与屏障，其余 warp 跑 mma 主循环。四件词汇各司其职。寄存器再分配用 setmaxnreg：生产者降到 96、消费者升到 224——warp 专化成立的必要条件。领袖选举用 elect：TMA 发射与事务计数要求单线程执行，warp 集体语义里先选出领袖。完成通知用 mbarrier 的字节记账：领袖在到达的同时增加屏障的预期事务字节数，TMA 硬件在数据落完后以完成字节数销账，两数相抵屏障才翻相。装载本身用 TMA：按张量描述符一次搬整块 tile。

```
gdn.setmaxnreg dec 96
%el = gdn.elect -> i1
scf.if %el {
    gdn.tma.prefetch %descA : ...
    gdn.mbarrier.arrive.expect_tx %bar, %c8192 : ...
    gdn.tma.async.load %descA[%k0, %c0, %c0, %c0], %bar to %sa : ...
    gdn.tma.async.load %descB[%k0, %c0, %c0, %c0, %c0], %bar to %sb : ...
}
```

［tools/gdn_l2_gen.py:585-593（gen_payload_ws 的 TMA 分支，t64x64x32 配置下两块 tile 合计 8192 字节）；消费者侧为 setmaxnreg inc 224 加相位奇偶等待］

协议有一个被真机死锁换来的修正：屏障必须是纯数据就绪屏障、初始化计数为 1——领袖是每步唯一的到达者；早期版本让全部消费者也到达，相位完成依赖的到达恰好被它等待的屏障挡住，死锁。修正后消费者只等待。

效果与代价同表：warp 专化的 kkt 配置在 255 寄存器下溢出 22,844B；TMA 变体把装载从一千余条 cp.async 换成 5 条 bulk 指令，溢出降到 48B，PTX 从 214.6KB 缩到 41.2KB。CUtensorMap 的维序契约由生成器持有：描述符的维度序是 CUDA 序（最内维在前），下译对 kernel 坐标不做反转、宿主编码助手把 memref 维序反转为 CUDA 维序，因此 kernel 侧坐标必须按原始 PTX 序发射；维度、box、跨距、坐标次序的完整契约落盘为 tensormap_spec.json，运行期照抄执行［tools/gdn_l2_gen.py:1226-1240（tensormap_spec），描述符三份：A 四维、B 五维、输出二维］。数据通路的最后一环由此闭合：装载指令、缓冲奇偶、完成通知、发射主体四个选择互相咬合，任何一环退回树内默认形态，重叠即消失。
