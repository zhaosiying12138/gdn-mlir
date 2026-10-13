## 十、一次性标定：六个微基准与标定表

编译期调度把配置排序整个搬到编译时刻，排序的依据是解析成本模型（第十一章）；模型的每一项都是"指令计数除以硬件速率"，速率这个分母必须有一个可信来源。候选来源有三：厂商数据手册、公开的第三方基准、自己测。对 sm_120a 这块消费级笔记本芯片，第一条路给不出完整的逐指令吞吐表，第二条路给的是桌面卡数值；更实质的困难在于部署窗口的 GPU 常与其它计算任务共载，可用带宽与时钟都偏离标称。项目因此设立一次性标定（calibration）：六个微基准 kernel 在一个 GPU 窗口内测出模型需要的全部速率，构成标定表。术语约定：标定指用专用微基准测量硬件速率常数的一次性动作；微基准指只测一种速率、其余因素压到最小的最小 kernel；共载指测量窗口内 GPU 上同时驻留其它计算任务。标定的全部代价是单次沉没成本：一个约一分钟的窗口、低于 2GB 的显存，此后每个新形状的调度只消费这张表。

### 1. 为什么编译期排序必须有自己的速率来源

成本模型的契约禁止逐配置拟合（第十一章第一节），速率因此必须整批来自某处。占位速率表与实测的对照说明问题有多大。加载器内置的 FALLBACK 表（注释明写为数据手册级估计、非测量，仅在标定文件缺键时启用并逐键点名）与实测对照：全局内存读带宽占位 1792 GB/s，实测 824.6 GB/s，占位值是实测的 2.2 倍；共享内存读带宽占位 20 TB/s，实测 113.1 TB/s，占位值不足实测的五分之一；bar.sync 占位 30 ns，实测 7.707 ns，占位值是实测的 3.9 倍［tools/gdn_costmodel.py:142-153；evidence/G3/calibration.json］。误差方向不一、幅度跨数倍——时间项等于计数除以速率，分母这样失真会把各资源项的权重关系整个搅乱，排序随之失真。

规范带宽同样不可用：这块卡的 GDDR7 规范带宽约 1.8 TB/s，实测共载窗口 824.6 GB/s、干净窗口 851.4 GB/s［evidence/G3/calib_report.md §1］。调度要预测的恰是实际窗口里的执行，标称数字对它没有约束力。反方向的要求又有限度：第十二章的扰动实验表明全部结构常数正负两倍缩放时秩相关仍稳定在 0.882–0.902。标定的设计目标由此定形——速率必须实测，精度达到排序容差即可，测量条件必须随数据一并披露。

### 2. 六个微基准：各测什么、怎么写

六个 kernel 全部住在一个 276 行的 CUDA 源文件里［tools/gdn_calib/calib_bench.cu］，共享三条写作纪律。其一，每个 kernel 把累加结果经一条永不成立的分支写出，防止编译器死代码消除删掉整段被测代码（文件头 12-13 行明写此意图）。其二，计时用 CUDA 事件，5 次以上预热后计 50 次、取中位数（time_kernel，132-154 行）。其三，网格取 82 个 SM 乘 8 块以饱和全机（178 行）。

（a）mma 峰值。k_mma 每线程维护 8 条独立的 mma 累加链填满指令级并行，一条 warp 指令恰为一个 4096 FLOP 的 m16n8k16 原子：

```cuda
__global__ void __launch_bounds__(128) k_mma(int reps, float* sink) {
  uint32_t a0 = 0x3f803f80u + threadIdx.x, a1 = a0 ^ 0x00ff00ffu;
  uint32_t a2 = a0 + 1, a3 = a1 + 1;
  uint32_t b0 = 0xbf80bf80u + blockIdx.x, b1 = b0 ^ 0x00010001u;
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  for (int it = 0; it < reps; ++it) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  if (c0 == 1e30f) sink[blockIdx.x] = c0 + c1 + c2 + c3;
}
```
［tools/gdn_calib/calib_bench.cu:35-51，逐字；末行即防死码消除的永不成立分支］

（b）共享内存读带宽：k_smem 在 48KB 窗口做 16 字节 uint4 读取加 xor 归约，带宽等于移动字节除以时间［54-68 行］。（c）全局读带宽的两种形态共用 k_gread：32MiB 缓冲小于 64MiB L2，测得 L2 驻留读带宽；512MiB 缓冲强制走 DRAM，测得 DRAM 形态；缓冲构造在 186-193 行［71-81 行］。（d）SFU ex2 吞吐：k_ex2 每线程 4 条独立 ex2.approx.f32 依赖链、8 路展开［84-97 行］。（e）bar.sync 往返：k_bar 在 128 线程块内连发 20000 次 bar.sync 0,128，总时间除以次数得单次 7.707 ns［100-104 行］。（f）cp.async 吞吐：k_cpasync 以 16 字节粒度从 1MiB 的 L2 驻留源窗拷入共享内存，每轮 commit 加 wait，测得 1388 GB/s［107-125 行］。

第七个量是 kernel 发射基准，它没有专属微基准：参考核 ref_f16（单个 m16n8k16 原子的纯发射 kernel）的实测时间减去其模型内全部非发射项，余项即发射常数 6.073 µs——这是契约允许的唯一全局发射常数，推导在求解器验证入口完成［tools/gdn_solver.py:433-446；evidence/G3/validation.json 的 launch_anchor_s］。

### 3. 运行协议：分钟级、低显存、幂等与共载口径

run_calib.sh 是唯一入口，头部注释声明全部代价：一个窗口约一分钟、显存低于 2GB、同路径输出直接覆盖——幂等，重跑即用新窗口的新值覆写旧表［scripts/run_calib.sh:1-4］。共载口径的披露协议：脚本在启动前、运行前、结束后抓三次 nvidia-smi 快照（利用率、显存、SM 时钟三元组），外加一次计算进程清单，据此置 shared_gpu 布尔，并把四组观测连同一句口径注记写进 calibration.json：

```bash
snap() {
  nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm \
    --format=csv,noheader,nounits | head -1
}
COL0="$(snap)"            # 启动前
# ……编译并运行 calib_bench……
COL2="$(snap)"            # 结束后
APPS="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
  | tr '\n' ';')"
# shared_gpu = bool(APPS)；快照与注记组装为 coload 块落盘
```
［scripts/run_calib.sh:19-39、54-66，节选；nvcc 编译与基准运行两行略］

这套协议的产出是：每个速率都能追溯到测量当时的 GPU 状态。本仓有过两种真实口径：初次标定与一个 24–99% 利用率的共载训练任务同窗，数值按"受挤占下界"记录；其后在干净窗口重跑，DRAM 读带宽 824.6 升到 851.4 GB/s，两个口径并列进表。时钟口径同理：峰值属性 1597 MHz 属笔记本保守基值，窗口内观察时钟 2.1–2.8 GHz；加载器优先采用共载快照的中位观察时钟（本例 2722 MHz），因为发射类时间项随实际时钟缩放［tools/gdn_costmodel.py:323-341］。

### 4. ×32 计数陷阱：warp 级操作按线程计数

标定文件 mma 段的 flops 字段写着 3.70 PFLOP/s。这个数物理不可能：它高于修正后真实峰值 32 倍，也远超任何已发布 GPU。根因在计数口径：峰值计算把每条 mma 指令按线程记数（网格 × 128 线程 × 次数 × 展开），而 mma.sync 是 warp 级操作——32 条 lane 共同执行一条指令、共同产出一个 4096 FLOP 的原子；逐线程计数把 FLOP 放大了 32 倍。诚实峰值等于指令率除以 32 再乘 4096，即 115.5 TFLOP/s。两条独立证据闭合这个修正。其一，指令率对属性时钟归一为每 SM 每时钟 6.89 条，换算到观察时钟 2.72 GHz 后约 4.05 条——恰等于每 SM 四条张量流水线的微架构配置［evidence/G3/calib_report.md §1 行 (a)］。其二，修正后的峰值进入模型，mma 路径预测与实测的差距落回可解释区间（第十二章盲区清单给出残余偏差的方向与幅度）。修复放在加载器一侧自动执行并留痕：

```python
if mma_flops and mma_instr:
    # G3-A's flops field multiplies per-THREAD instruction counts
    # by 4096 (m16n8k16 atom FLOPs), but mma.sync is a WARP-level
    # op: the honest peak = warp instructions/s x 4096 =
    # instr_per_s / 32 x 4096. Detect the discrepancy (a >2x gap
    # is physically impossible otherwise) and recompute.
    warp_flops = mma_instr / 32.0 * 4096.0
    if mma_flops > warp_flops * 1.5:
        self.m["mma_flops_field_asbuilt"] = mma_flops
        mma_flops = warp_flops
        self.m["mma_flops_corrected"] = True
```
［tools/gdn_costmodel.py:346-356，逐字；原值存 mma_flops_field_asbuilt、修正打 mma_flops_corrected 旗标］

任何消费 calibration.json 的程序都必须做这次重算——这条作为硬件与工作流事实记录在案（evidence/G3/solver_report.md §6）。一般化守则：凡按线程计数的指标遇到 warp 级指令，先问除没除 32。

### 5. 标定值全表

标定表是成本模型全部分母的唯一来源（数值沿用主文表 5）：

表：sm_120 一次性标定表

| 量 | 标定值 |
|---|---|
| mma bf16 峰值 | 115.5 TFLOP/s（tf32 按 2:1 折算 57.8） |
| 共享内存读带宽 | 113.1 TB/s |
| L2 / DRAM 带宽 | 2144.3 / 824.6 GB/s（空闲窗口 851.4） |
| SFU ex2 吞吐 | 3.23 Telem/s |
| bar.sync | 7.707 ns |
| cp.async | 1388 GB/s |
| kernel 发射基准 | 6.073 µs |

表内每个值的测量条件——共载或干净窗口、缓冲规模、预热与迭代次数、块形——都记录在 evidence/G3/calibration.json 的 method 与各测量段；复跑用同一条脚本覆写同名文件。分母自此齐备，下一章把计数除上去。
