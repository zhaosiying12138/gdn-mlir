## 十七、成本模型的物理量：标定、计数、速率与占用

编译期排序要在毫秒尚未发生之前比较数百个配置。形状、tile 和循环趟数可以给出工作量，工作量却不能直接决定次序：同样一字节，在 L2 与 DRAM 中的代价不同；同样一条指令，标量循环与张量核也走不同流水线。独立速率来源因而处在模型的底层。它把计数转换成时间，也让新形状可以复用同一套物理量，而无须为每个配置拟合一条曲线。

### 1. 六个微基准测量什么

一次性标定（calibration）用微基准（microbenchmark）隔离单一硬件能力。`k_mma` 测 bf16 张量峰值，`k_smem` 测共享内存 16B 读取，`k_gread` 以 32MiB 与 512MiB 两种缓冲区分别测 L2 驻留和 DRAM 读取，`k_ex2` 测 SFU 的 `ex2.approx.f32` 吞吐，`k_bar` 测 128 线程块的 `bar.sync` 往返，`k_cpasync` 测 16B 异步复制吞吐。计时采用 CUDA 事件，至少 5 次预热、50 次测量并取中位数；网格按 82 个 SM 乘 8 块铺开。入口脚本声明整段窗口约一分钟、显存低于 2GB，重跑覆盖同名结果，并保存三次 GPU 状态快照与计算进程清单。[tools/gdn_calib/calib_bench.cu:132-154,171-214][scripts/run_calib.sh:1-10,19-39]

张量基准的主体很短，却同时展示了测量与计数两个要点：

```cuda
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
// 外层 reps，内层展开 8 条独立累加链；另有永不成立的 sink 分支。
double instr = (double)grid8 * 128 * 2048 * 8;
```

[tools/gdn_calib/calib_bench.cu:33-50,203-214，两段节选]

读这段代码要看三点：m16n8k16 原子含 4096 FLOP，八路展开用于填满指令级并行，`sink` 分支阻止死代码消除。末行的 `128` 又暴露了口径风险：它按线程累计指令，而 `mma.sync` 由整个 warp 协同完成。微基准的真实值只有连同计数单位一起保存，才能成为可靠分母。

G3 排序验证归档的共载窗口给出下表；tf32 峰值由 bf16 的 0.5 倍结构比推导，发射基准则由参考核单独锚定。[evidence/G3/validation.json:16-35][evidence/G3/calibration_shared.json:48-89]

| 物理量 | 归档值 |
|---|---|
| bf16 / tf32 张量峰值 | 115.5 / 57.8 TFLOP/s |
| 共享内存读带宽 | 113.1 TB/s |
| L2 / DRAM 读带宽 | 2144.3 / 824.6 GB/s |
| SFU ex2 吞吐 | 3.23 Telem/s |
| bar.sync 往返 | 7.707 ns |
| cp.async 16B 吞吐 | 1388 GB/s |
| kernel 发射基准 | 6.073 µs |

这些值属于同一轮验证，不能与另一窗口的测量随意拼接。当前 `calibration.json` 已是空闲窗口重测，其中 DRAM 为 851.4 GB/s；824.6 与 851.4 因此代表两种测量条件。[evidence/G3/calibration.json:16-23,62-67]

微基准的适用范围也由构造方式决定。32MiB 缓冲小于 64MiB L2，重复计时测到驻留带宽；512MiB 缓冲才把访问推向 DRAM。共享内存基准中的规则访问可能受广播优待，所以 113.1 TB/s 不能被解释为任意 bank 模式的保证。共载快照记录利用率、显存和时钟，让带宽下降可以追溯；中位数抑制单次抖动，最小值与最大值仍保留窗口离散程度。独立速率来源由此带有设备、访问形态和测量窗口三项限定，远比一条规范峰值更接近待排序内核的实际条件。

### 2. warp 级修正与速率契约

原始 `flops` 字段曾达到 3.70 PFLOP/s，恰好保留了逐线程计数造成的 32 倍膨胀。加载器用独立保存的 `instr_per_s` 重算，并留下修正旗标：

```python
warp_flops = mma_instr / 32.0 * 4096.0
if mma_flops > warp_flops * 1.5:
    self.m["mma_flops_field_asbuilt"] = mma_flops
    mma_flops = warp_flops
    self.m["mma_flops_corrected"] = True
```

[tools/gdn_costmodel.py:343-356，节选]

这里读的是单位守恒：32 条 lane 共同完成一条 warp 指令，不能再乘 32。除以 32 后得到 115.5 TFLOP/s，与每 SM 四条张量流水线的时钟归一相互印证。其余速率也有各自单位：带宽是 B/s，ex2 是 elem/s，屏障与发射是 ns。标定表因此包含物理量和单位纪律；cp.async 的 1388 GB/s 在当前模型中保留为测量项，并未单独充当 `T_lsu` 的分母。[evidence/G3/solver_report.md:146-168]

### 3. 六个时间项如何由计数除以速率

`predict` 把同一配置的特征转换成资源时间。下面几处赋值是模型的核心：

```python
terms["T_mma"] = feats["mma_flops_per_cta"] / (peak / calib.sm_count)
macs = feats["ffma_flops_per_cta"] / 2.0
terms["T_ffma"] = macs * NOMMA_CYC_PER_MAC / calib.m["clock_hz"]
terms["T_gmem"] = feats["gmem_bytes_per_cta"] / calib.per_sm(bw_key)
terms["T_smem"] = feats["smem_bytes_per_cta"] / calib.per_sm("smem_bw")
lsu_ops = (feats["cpasync_warp_ops"] + feats["lds_warp_ops"]
           + feats["stg_warp_ops"])
terms["T_lsu"] = lsu_ops / (LSU_IOPS_PER_CLK * calib.m["clock_hz"])
terms["T_spill"] = spill_bytes / (calib.per_sm("l2_bw") * SPILL_BW_FRAC)
```

[tools/gdn_costmodel.py:689-724，按项节选]

`T_mma` 的分子是每 CTA 的 `2*bt*bv*K*n_gemm`，分母是全机张量峰值的每 SM 份额。`T_ffma` 服务 no-MMA 标量三重循环：MAC 数乘 SASS 读出的 10 cyc/MAC，再除以实际时钟。旧 CUDA 核屋顶线忽略这条未分布循环的发射与依赖限制，曾低估 100–400 倍。

`T_gmem` 统计 A、B tile 与输出的逻辑字节，按 L2 驻留或 DRAM 选择速率；`T_smem` 统计 CSE 后唯一 fragment 装载的字节。两者都除以 82 得到每 SM 份额。`T_lsu` 改问发射压力：cp.async、ld.shared 与 st.global 的 warp 指令总数，除以每 SM 每时钟 4 条 LSU 指令和时钟频率。均匀地址复制可能在导线上被扇区去重，发射次数却不会因此消失。`T_spill` 则把动态 LDL/STL 操作乘 4B，再除以每 SM 的 L2 带宽份额。[tools/gdn_costmodel.py:46-94,514-555]

以 bt=bv=64、K=128、单个 GEMM 的 bf16 kkt 为例，张量项分子为 2×64×64×128=1,048,576 FLOP。每 SM 分母为 115.5 TFLOP/s 除以 82，得到约 0.744 µs，与验证文件的该项一致。全局字节的分子为 40,960B，采用 L2 驻留速率时约为 1.566 µs。手算能暴露两类常见错误：把全机速率直接除进单 CTA 工作量会把时间缩小 82 倍，把线程级 FLOP 再乘进分子又会放大 32 倍。两项的相对大小还提示，在流水路径的 max 重叠中，此配置由全局访存项主导。[evidence/G3/validation.json:163-179]

此外还有三项串联成本：动态屏障次数乘 7.707 ns 得 `T_bar`，kkt 的 `bt*bv` 个 ex2 除以 SFU 每 SM 速率得 `T_ex2`，分配次数乘 `malloc_ns` 得 `T_alloc`。发射常数 6.073 µs 来自 `ref_f16` 实测减去其模型内非发射项，是全局锚点。[tools/gdn_solver.py:430-447][evidence/G3/validation.json:16]

### 4. 静态自检、动态溢出与资源审计

解析计数需要与真实代码对照。G3 的 18 个 GDN 配置自检中，`mma.sync`、`ld.shared`、`bar.sync` 静态计数均为 18/18 精确命中，cp.async 因序言与参数收尾有 +0.4% 至 +1.6% 偏差。软件流水的 peel 复制因子也在这项检查内；它验证了公式确实描述生成代码，而非仅凭数学形状猜测。[evidence/G3/solver_report.md:120-125][tools/gdn_solver.py:525-593]

溢出计数多走一步。优先读取被计时 cubin 的 ptxas 报告，报告缺失时回退到 SASS 的 LDL/STL 静态计数，再消除 peel 对静态与动态次数的不同影响：

```python
spills = (so.get("LDL") or 0) + (so.get("STL") or 0)
pipe = max(feats["pipe"], 0)
dyn = spills * (feats["trips"] + pipe) / (1 + pipe)
feats["spill_dyn_ops_per_cta"] = dyn * feats["threads"]
spill_bytes = feats["spill_dyn_ops_per_cta"] * 4.0
```

[tools/gdn_costmodel.py:616-625,710-712，节选]

这里能判断静态指令数如何进入动态时间项，也能看出版本必须一致：另一遍编译可能有不同寄存器分配，旧 cubin 的溢出计数不能套到新 cubin 上。自检也不等于资源账已经完备。当前 no-MMA 成本模型仍把 accumulator 按 `esz` 计入 `(bt*K + K*bv + bt*bv)*esz`，生成器却明确分配 f32 accumulator；末项应为 `bt*bv*4`。用真实合法配置 o、bf16、K=192、V=128、tile.M=64、tile.N=128、tile.K=64、mma=off 对账：旧式为 `(64*192 + 192*128 + 64*128)*2 = 90,112 B`，新式为 `(64*192 + 192*128)*2 + 64*128*4 = 106,496 B`，后者超过 strict 的 100,352 B，而生成器的 220 KiB 门槛仍接受。G3/G6 的 368、465 应作为相应报告与约束快照阅读，这个差异正是重新审计合法域的入口。[tools/gdn_costmodel.py:557-568][tools/gdn_l2_gen.py:130-138,768-799]

### 5. v1、v2、v3 与占用波次

三种形式逐步增加机制。v1 用屋顶线（roofline）组合计算、全局访存和共享访存：有软件流水或 warp 专化时取最大值，同步 p0 路径取和，再串联发射、屏障、溢出与 ex2。v2 加入 LSU 发射项和设备分配项。v3 再加入占用（occupancy）与波次（waves），让整形状的 CTA 数进入总时间。

```python
by_regs = regs_per_sm // (threads * regs)
by_smem = max(1, smem_usable // max(1, feats["smem_static_model"]))
by_warps = WARP_SLOTS_PER_SM // warps
occ = max(1, min(by_regs, by_smem, by_warps, CTA_SLOTS_PER_SM))
waves = max(1, math.ceil(total_ctas / (calib.sm_count * occ)))
```

[tools/gdn_costmodel.py:647-657,731-740，两段节选]

占用取寄存器、共享内存、warp 槽与 CTA 槽四项限制的最小值；波次是总 CTA 除以全机驻留容量后向上取整。单 chunk 验证中波次为 1，所以 v2 与 v3 秩相同；大形状下小 tile 产生更多 CTA，二者可能翻转头部。占用模型给出可检验预测，仍需全形状实测检查。

最后要区分参数纪律与预测能力。模型没有逐配置拟合常数，仍可以选择 v1/v2/v3、替换有 SASS 依据的结构常数，并保留明显绝对误差。该次验证的 `calibration_fallback_keys` 明确只有 `malloc_ns`；静态共享内存修复使 18 个配置的 `n_allocs` 和 `T_alloc` 全为零，因此这个后备值未参与有效排序数值。后备清单、逐项时间和秩指标应一起阅读，单看任何一个都不足以审定模型。[tools/gdn_costmodel.py:597-611,715-741][evidence/G3/validation.json:37-39,73-94]
