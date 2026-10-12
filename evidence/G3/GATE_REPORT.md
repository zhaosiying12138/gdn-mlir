# G3 Gate 报告（sm_120 标定 + Z3 求解器 + 解析成本模型）—— ✅ 达成（2026-10-09，commits 13b8865/9713821）

## Gate 判据核验（docs/PLAN.md G3）

| # | 判据 | 状态 | 证据 |
|---|---|---|---|
| 1 | sm_120 一次性标定（1 个 GPU 窗口，分钟级 <2GB） | ✅ | scripts/run_calib.sh → evidence/G3/calibration.json（opportunistic 窗口已按协议登记+完结；**shared_gpu=true 全量标注**，干净窗口可幂等重跑） |
| 2 | Z3 knob 求解器 | ✅ | tools/gdn_solver.py：knob 向量变量 + smem 字节模型(≤98KB)/regs/对齐/结构合法性断言；SAT+blocking 枚举，**固定 seed+rlimit，两次运行逐字节一致**；合法空间 kkt=368 / o=204~465，~0.3s 枚举完 |
| 3 | 校准解析成本模型（无逐配置拟合常数） | ✅ | tools/gdn_costmodel.py：速率全部来自标定表或文献级 sm_120 结构常数；--selftest 静态特征 vs PTX 计数 18/18 精确（cp.async 偏差 ≤1.6%） |
| 4 | **排序 gate：已测点 Spearman ≥ 0.7** | ✅ **ρ = 0.8962**（kkt 0.936 / o 0.829，n=18） | evidence/G3/validation.json + solver_report.md |

## 标定表（RTX 5090 Laptop，82 SM，L2 64MiB，共享窗口口径=噪声下界）

| 指标 | 值 |
|---|---|
| mma.sync.m16n8k16 bf16→f32 | 3.70 PFLOP/s（≈4.1 instr/SM/clk，与 4 条张量管线吻合）* |
| smem 读带宽（ld.shared 16B） | 113.1 TB/s |
| gmem L2 驻留读 / DRAM 读 | 2144 / 825 GB/s（后者受共训挤压，规格 ~1.8TB/s） |
| SFU ex2.approx | 3.23 Telem/s（24.7 elem/SM/clk） |
| bar.sync 往返（128 线程） | 7.71 ns |
| cp.async 16B | 1388 GB/s |

*G3-B 复核修正：calibration.json 的 flops 字段线程级多计 ×32（装载器已改算 115.5 TFLOP/s 有效口径——比率用于模型不受影响）；干净窗口重跑时以 microbench 修正。

## 模型迭代长卷（论文素材）

v1 合同式 roofline **0.7352** → v2 +LSU-issue/device-alloc 项 **0.7475** → nomma 项替换为 **SASS 导出标量环模型**（内环 39 指令/3 LDS ⇒ 10 cyc/MAC，编译期证据非拟合）**0.8962** → launch 锚定（ref kernel 6.07µs，rank 不变量）→ v3 waves/occupancy（单 CTA 测点集不可分辨；全形状生效，v2↔v3 top-5 翻转=占用项在工作，**G5 可检验预测**）。
稳健性：全部结构常数 ±2× → ρ∈[0.882, 0.902]；实测 ±30% 噪声 → ρ≥0.856。

## 首批实测洞见（G4/G5 的路标）

1. **mma 开关主导（~50-100×）**：nomma 路径=未分布标量 linalg 循环（0.5-2.5ms vs 0.015-0.105ms）——生产路径必须走 mma 或修 nomma 降低。
2. **pipeline p2/p3 ≫ p0**（105→21µs；.ca 串行化是模型最大已记录盲点，方向正确）。
3. **少 warp 反而快**（w2 14.9 < w4 21.2 < w8 34.2µs；均匀地址拷贝按 warp 重复发射）。
4. 硬件首跑揭出两个真缺口：**MLIR workgroup memref.alloc 降为 device malloc + cvta**（首跑即 ILLEGAL_ADDRESS；已建 PTX 级静态 smem 修复 tools/gdn_smem_fix.py，修复后 18/18 可跑）——这是可上报的 stock-MLIR 缺口；ws 配置 mbarrier 骨架死锁 + TMA 宿主描述符未接线 = **G4 目标**。
5. kkt mma 路径数值疑点（非零元 1168 vs linalg 2080）——G4 数值验证清单第一项。

## 结论

**G3 达成**：编译期确定性调度闭环成形（Z3 合法性枚举 + 标定成本模型排序 ρ=0.896，全离线、确定性、无逐配置拟合）。
进入 **G4：GPU 正确性**（修 ws/tma 硬件缺口 + 全配置数值对拍 JAX 参考 + kkt 数值疑点排查；fla 容差契约）。
