## 二十六、数据通路与存储几何：流水、Bank 与驻留的联合核算

一个 tile 从全局内存到张量核，要依次回答四个问题：哪些线程发出哪些字节，数据何时对其他线程束（warp）可见，共享内存地址落在哪些存储体（bank），以及这块生命周期会抬高多少驻留成本。四项账目相互制约。搬得更快可能增大缓冲，缓冲压缩可能改变行距，寄存器上限又可能把驻留收益兑换成溢出。

### 1. 线程跨步暂存与完成协议

暂存（staging）的基本单元是 16B，对应 8 个 bf16 或 4 个 f32。生成器采用线程跨步（thread-strided）认领：tid 决定首个块，后续块每次跨过 nthr 个位置，异步令牌（token）留在调用方作用域，等待点由流水结构决定。

```python
el = 8 if dt == "bf16" else 4
nch = Cc // el
tot = R * nch
assert tot % nthr == 0, f"astage: {tot} chunks % {nthr} threads"
for kk in range(tot // nthr):
    p = tid if kk == 0 else _tadd(e, tid, kk * nthr)
    # 省略 r、cc 与全局/共享列下标计算
    t = e.v("acp")
    e.l(f"{t} = nvgpu.device_async_copy {src}[{m}, {gcol2}], "
        f"{dst.name}[{r}, {dcol}], {el} bypassL1 : {gty} to {dst.ty}")
    toks.append(t)

# astage_sync 的组队与排空
g = agroup(e, toks)
await_all(e, g)
e.l("gpu.barrier")
```

[tools/gdn_bwd_gen.py:1687-1714,1735-1740，省略地址展开]

读这段代码先看 `p`，再看 `tot // nthr`，最后看 wait 后的屏障。设 `nch = Cc // el`，线性块编号为 `p = tid + kk*nthr`，则 `r = p // nch` 是行，`cc = p % nch` 是块内列；源全局列为 `r*cm + cc*el + col0`，目的共享列为 `cc*el + dst_col0`。例如 `nthr=128、tid=1` 时，kk=0、1、2 分别认领 p=1、129、257；只要 `tot % nthr == 0`，这些跨步序列恰好覆盖全部块且不重复。`p` 同时决定写集覆盖和全局合并访问；nthr 若与启动线程数不符，会有 16B 块无人认领。块统一（block-uniform）反例更能说明下标的分量：若 256 个线程都遍历全部 tot 个块，同一数据会被发射 256 次；若保留 tot/nthr 次循环却让所有线程使用同一个 p，则只有 1/256 的块被覆盖，每块又重复 256 次。前者浪费带宽，后者还会让陈旧数据进入计算。

`nvgpu.device_async_wait` 只排空调用线程所等待的异步组，其他 warp 的读者需要 `gpu.barrier` 建立 CTA 级先后关系。生成器绕开会在组队后立即排空的默认 transform，是为了把等待移到计算之后；省掉这道显式屏障则会破坏可见性合同。[tools/gdn_bwd_gen.py:1662-1680]

### 2. 乒乓缓冲把等待藏进有效工作

双缓冲按奇偶区分当前阅读窗口与下次写入窗口。完整仿射递推分支（`GDN_ARET=0`）的 dhs 将 K×K 的 A 拆成 K×32 滚动窗口，避免一次驻留 64KB；计算窗口 w 时，窗口 w+1 已在途中：

```python
for w in range(NW):
    if w + 1 < NW:
        toks = []
        astage_tile(e, G["A_all"], aty, "%m", K, (w + 1) * BKW,
                    AW[(w + 1) % 2], 0, K, BKW, gdt("A_all"), toks)
        gw = agroup(e, toks)
    rgemm_aw(e, Xa, AW[w % 2], w * BKW, X, K, BV, BKW,
             acc_buf=(Xb if (w == 0 and Xb is not None) else Xa))
    e.l("gpu.barrier")
    if w + 1 < NW:
        await_all(e, gw)
        e.l("gpu.barrier")
```

[tools/gdn_bwd_gen.py:3891-3906]

读点在两道屏障的不同职责。`rgemm_aw` 后的屏障封住当前窗口的共享读，允许它在两次迭代后重写；wait 后的屏障发布下一窗口的异步落盘。`AW[w % 2]` 与 `AW[(w+1) % 2]` 保证两事件落在不同半区，kk 递增的 FMA 次序保持不变。真实对照中 dh_all 与 dh0 逐位相同，dhs 从 0.59 ms 降至 0.32 ms。[evidence/G5/bwd_report.md:1801-1825]

当前默认的 caryA 退役分支不再需要 A 窗口，改用 Xb 的两个列半区携带 B_all/dht；流水原则仍然成立，池与内核清单却属于另一种构建。[tools/gdn_bwd_gen.py:3671-3689,3717-3752] 这也说明乒乓只是缓冲复用协议，预取距离、读者结束点和寄存器压力必须一起核算。

### 3. LDGSTS 到 LDSM 之间的数据完整性规则

这套 sm_120a 工具链出现过一种静默失效：cp.async（SASS 的 LDGSTS）落盘的 tile 经普通标量共享读完全正确，经 ldmatrix（LDSM）读回却出现构建相关的错行。数值差可大到 dk 1.34、dv 2.18，容差门能够报警，却不能修复数据通路。生成器规定，LDSM 操作数必须由标量共享写产生；异步落盘区域在首个 LDSM 读者前执行一次恒等重写：

```python
e.l(f"scf.for %lu = %tid to {e.ic(R * Cc)} step %nt {{")
e.depth += 1
i, j = e.v("lui"), e.v("luj")
e.l(f"{i} = arith.divsi %lu, {e.ic(Cc)} : index")
e.l(f"{j} = arith.remsi %lu, {e.ic(Cc)} : index")
v = e.v("luv")
e.l(f"{v} = memref.load {buf.name}[{i}, {j}] : {buf.ty}")
e.l(f"memref.store {v}, {buf.name}[{i}, {j}] : {buf.ty}")
e.depth -= 1
e.l("}")
e.l("gpu.barrier")
```

[tools/gdn_bwd_gen.py:1751-1761]

`launder_pass` 读什么就原样写回什么，因此值位级不变；它改变的是共享写路径。末尾屏障让全部恒等写先于 LDSM。由此可以判断，在本文记录的工具链与复现条件下，异步 tile 只能直接喂普通 `ld.shared`，需要 LDSM 时必须经过这类重写或改用逐元素 staging。这是当前工具链下的保守规避，不是通用 MLIR 或 PTX 的禁令；版本变化后应先用最小复现重新判定。

### 4. Bank 几何要分别核算读侧与写侧

共享内存有 32 个 bank，每个宽 4B；地址相差 128B 整数倍的两个字落入同一 bank。ldmatrix 一次读取 8 行、每行 16B。若行距是 128B 的倍数，8 个行起点反复命中同一组 4 个 bank，形成 8 路冲突。给 bf16 tile 增加 8 列，即把行距增加 16B，行起点每次移动 4 个 bank，8 行可扫过 32 个 bank，同时保持 16B 对齐。dkb 的 `vh/vl/hh/hl` 因而都声明为 `BV+8` 列，垫列从不进入 MMA 写集。[tools/gdn_bwd_gen.py:4147-4160] 前向 F2 的同类修复将 473 µs 降至 249 µs。

写侧的 WST 把相邻两个 bf16 配对为一次 32 位读写，指令与循环迭代减半。直接套到 32 列、80B 行距的 F2 tile 却反慢 4.3 µs：每行只有 16 对，平铺映射让一个 warp 跨两行，lane 28–31 与 lane 0–3 撞 bank。W32P 保留配对字节，只改 lane 到行的双射：

```python
if (os.environ.get("GDN_W32P") and cols == 32 and rows % 8 == 0
        and nthr == 128):
    hi16 = e.v("w3h")
    e.l(f"{hi16} = arith.cmpi uge, %lane, {e.c(16)} : index")
    hi4 = e.v("w3h4")
    e.l(f"{hi4} = arith.select {hi16}, {e.c(4)}, {e.c(0)} : index")
    hcol = addi(e, "%warp", hi4)
    c0 = muli(e, remsi(e, "%lane", e.c(16)), e.c(2))

    def body3(i):
        k = divsi(e, i, e.c(nthr))
        row = addi(e, muli(e, k, e.c(8)), hcol)
        rr = addi(e, row_base, row) if row_base is not None else row
        # 省略同地址的 v0/v1 配对读写
```

[tools/gdn_full_gen.py:1036-1059，省略配对内存操作]

读这段下标可还原 `row = 8*(i>>7) + warp + 4*(lane>=16)`，列基为 `2*(lane%16)`。低半 warp 访问行 r 的 16 对，高半访问行 r+4；80B 等于 20 个字，而行号差 4 带来 80 个字的位移，80 mod 32 = 16，两半区恰好覆盖互补的 16-bank 集合。该映射在七个场景减少 17.9–22.6 µs，且每行、每对仍只写一次。[tools/gdn_full_gen.py:986-1002]

{{FIG11}}

### 5. 生存期别名压缩静态池

生存期（lifetime）结束后的 tile 可以把字节交给后续 tile。memref 的 SSA 值唯一并不禁止底层内存别名；本项目选择在 PTX 后处理阶段按生成器声明的布局合址，并用尺寸唯一性识别 malloc，识别失配便整体回退到无别名布局。一个简洁的前向布局表是：

```python
alias_modules["gdn_fwd_f1_kernel"] = {
    "identify": {"9216": "Abf", "9504": "Tt",
                 "8704": "vt", "8976": "k2"},
    "arenas": {"Abf": 9216, "Tt": 9504},
    "alias": {"vt": ["Abf", 0], "k2": ["Tt", 0]},
}
```

[tools/gdn_full_gen.py:4430-4435]

读表时先核对 identify 的尺寸能否唯一锚定，再检查 vt、k2 的首次写入是否晚于基 tile 的最后读取。arenas 给出保留字节，alias 给出折叠位置；它能让静态申报反映同时活跃的数据，而非所有 tile 的生命周期拼接。屏障必须封住跨界读写，异步落盘还要遵守上一节的消费规则。

三个反向资源池的归档压缩结果如下；caryA 一项属于其退役前的历史构建：

| 内核 | 压缩前 | 压缩后 | 主要手段 |
|---|---|---|---|
| dqkwg | 100,864B | 47,616B | 双竞技场别名与 epilogue 停车 |
| caryA | 82,176B | 28,592B | split 改写并退役四个共存 tile |
| wyb | 100,096B | 45,056B | 双竞技场别名与暂存退役 |

{{FIG6}}

三者在各自测量构建中达到每 SM 两个 CTA。dqkwg 的每 SM warp 数由 7.59 升至 14.53；逐入口 `.maxnreg` 固定 128 后，另一次实测占用由 15.02% 升至 29.17%，warp 数由 8.00 升至 14.23。读取这些数字要同时检查最终 smem_fix 后那遍 ptxas 报告，第一遍资源数可能对应尚未发布的 cubin。[evidence/G5/bwd_report.md:2508-2601]

`.maxnreg` 是 PTX 入口上的静态寄存器上限，ptxas 据此安排驻留；`gdn.setmaxnreg` 则在 kernel 内做 warp 间再分配，warp 专化时生产者降到 96、消费者升到 224。前者影响 CTA 能否驻留，后者影响已驻留 warp 的可用寄存器与溢出，二者不能互相替代。

寄存器账可以直接手算：256 线程、每线程 128 寄存器时，两个 CTA 需要 2×256×128 = 65,536 个寄存器，恰好达到寄存器文件上限。若最终调度实际需要 168 个，驻留便跌回一个 CTA；强行压低上限又可能引入本地内存溢出。静态池、寄存器、spill 字节和实测活跃 warp 必须一起阅读，任何单项都不足以宣布驻留优化成功。
驻留也不是普遍收益。F3 在 T8192 约流动 120MB，属于带宽受限；双 CTA 版本实测反慢 96.1 µs。分解账显示，同代码强制单 CTA 慢 115.6 µs，其中 K 循环重排约付出 110 µs，驻留只买回约 20 µs。[evidence/G5/fwd_report.md:2231-2258]

通用 MLIR 能检查类型、地址和部分资源约束，却没有这份 GDN 调度中的 tile 消费指令、lane 映射、异步 token 距离和门控分支知识。理论上可另写带目标 bank 模型与活性分析的专用 pass；现有通用管线无法仅凭 IR 自动推出这些几何事实。生成器把它们与发射决策放在同处，再用 ptxas、SASS、数值门和实测画像复核，才使每次地址改写同时满足正确性与资源预算。
