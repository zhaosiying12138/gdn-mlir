# G5 Gate 报告——第 3 轮（2026-10-09，commits d8ed168/c6239c7）

## 本轮战果

| 方向 | 第2轮 | 第3轮 | vs fla（交错终判暂读） |
|---|---|---|---|
| **fwd staged B1T2048** | 0.663ms | **0.262ms（静默单读）/ 0.44-0.46ms（共载中位）** | **1.14×/1.83×（两轮交错暂读）——首次越过 1.1×** |
| fwd B4T2048 | 1.636ms | **1.17ms** | 1.00×/1.36× |
| fwd B1T8192 | 2.450ms | **1.34ms** | 0.87×/1.04×（segment-split 杠杆已量化 ~250µs） |
| bwd B1T2048 | 14.16ms | **11.06ms**（min 9.57） | 0.093× |

数值：fwd 4/4 位级一致（每杠杆后复验）；bwd 6/6 双形状（最紧 dk 4.5× 裕量）。

## 决定性发现（本轮）

1. **F2 真根因 = 8 路 LDSM bank 冲突**（所有 ldmatrix 读 tile 行距 128B 整数倍 → 每象限 8 行落同 4 bank；ncu MIO 满 63%）——**+16B padding 修复，F2 471→249µs**；叠加 w4/bv32（V-tile 翻倍至 64 CTA）→ **83µs（5.7×）**。预置的 ping-pong 本身零收益（诚实负结果）——且源码级发现 **transform 的 create_async_groups 在每组后立即插 wait_group 0，永不重叠**（必须手动 nvgpu.device_async_copy 路线）。
2. **首个真实逃生舱启用（hatch 1/≤5）**：`gdn.mma.tf32k8` 经 `llvm.inline_asm` 发出 fla 精确指令；两项硬件级发现——**tf32-k8 的 A-fragment 寄存器序是 row-half-outer**（硬件整数探针裁定，文档直觉序错误）；**early-clobber `=&f` 强制**（普通 `=f` 下 ptxas 把 D 映到同 mma 的 A quad，张量核确定性毁操作数；nvvm.inline_ptx 的 BasicPtxBuilder 无法表达 & → 直落 llvm.inline_asm）。
3. **逐位点数值 bisect**（24 个 f32-GEMM 位点单点启用测试）：16 位点 tf32 化；5 位点为数值保标量（dhu.dkB/wy.u 单点即把 dk 顶破 8e-3 bar——同形状的 wy.w 却干净，S→Sn 路径不对称）；3 位点为性能保标量。全表入库。
4. 诚实负结果：dhu 的 tf32 化在串行逆走中寄存器爆到 255+spill（3.69→4.67ms，回退）；BV=64 retile 需 ~150KB smem 超静态限（且 H16 下 NV=2 CTA 反而更少）——真实杠杆=spill-free tile 重规划/buffer 复用重构/BV=16。

## 判据状态（第 3 轮后）

| # | 判据 | 状态 |
|---|---|---|
| 5 | e2e ≥1.1× | **fwd B1T2048 ✅（暂读 1.14×）**；B4T2048 边缘（1.00-1.36×）；B1T8192 未达（0.87-1.04×，杠杆已列）；**bwd 未达（0.093×）** |
| 6 | 单 kernel 旗舰 | 建成+数值过；staged 已过线，旗舰随共通修复同步 |

## 第 4 轮任务单（全部有实测排序）

1. **bwd dhu 3.76ms 串行逆走**（最大单点）：spill-free mma tile 重规划 或 buffer 复用重构（96KB→腾出 tf32 位点空间）。
2. bwd dqkwg 2.75ms（全局 staging+dkB NV-fold）> wy 1.8ms（串行 32×32 solve 基）。
3. fwd B1T8192 segment-split（composed-M 两遍，预估 ~250µs）。
4. 干净窗口 `tools/gdn_final_adjudicate.py` 交错终判（脚本就绪，含脏窗口标记与 smi 快照）。
