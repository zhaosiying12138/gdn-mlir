# G5 附录——第 6 轮（bwd 冲线，G7 后追加；2026-10-09，commit f07fd2e）

## 战果

| 项 | 第5轮 | 第6轮 | 备注 |
|---|---|---|---|
| wy solve 基 | 1.45ms | **0.70ms（2.06×）** | thread-per-column；T 位级一致（同 k 升序 f32 链）；显式上三角清零替代 smem-alloc-as-zero 依赖 |
| dqkwg dvp_bh | 1.71ms | **1.66ms（1.03×）** | 落地为 2-mma 型（行尺度 2^(gcl−gc) 折入 epilogue → A=原始 bf16 k 切片免 a 侧拆分；b 侧单遍融合拆分；≤4-warp 受守卫 mma 规划；234→214 regs 0 spill）；两个中途设计实测回退已录（3-mma 移植 2.19ms、转置 dW/dq1 staging 1.93ms） |
| dhs 预取 | 0.59ms | 未出厂 | 负经济学实测：B 级<5% 每 chunk 耗时，rgemm 全局 A 读链（2048 warp 广播 load/thread/chunk）主导——下一杠杆=A 侧（A_all smem/压缩） |
| **e2e bwd** | **8.21/7.20** | **7.27 med / 6.41 min（复跑 7.36/6.42）** | **0.126×→0.142×**；T512 1.139→1.067ms |

数值：6/6 双形状逐步保持（第5轮量级：dk 4.5× 裕量等）；wy 改动零位移、split-bf16 仅动 dbeta 1.413→1.407e-3。

## 重大工具链发现（论文素材，探针 r6_ldgsts_probe_{fail,stshared_pass} 已入库）

**sm_120a 手动 cp.async 的第二类静默失败**：经 `LDGSTS.E.BYPASS.128` 写入的 smem tile，用通用 `ld.shared` 读回**正确**，用 `nvgpu.ldmatrix` 读回**错误**——损坏限定于构建调度相关的 16 行块子集（行 32-63 或 48-63，确定性、逐构建变化），任何通用 load dump 均不可见（本次是 6/6 门失败+lane-true fragment dump 才定位到整 mi 块级错误）；同一 tile 经 `st.shared` 写入则位级精确（5.2e-8）。既有链上全部 mma 操作数 tile 都是 st.shared 写入——dkb/wy 从未触雷的原因。修复=逐元素 bf16 staging。**r4 的 kdbg 零读取为同族兄弟症状**（同指令族+延迟等待），悬案等级从"未解"降为"已定性家族"。

## 剩余排位（0.142×→1× 需再 -7.1ms）

fwdscan 0.92 + wyb 1.04（现最大阶段，本轮无未试结构杠杆）＞ dqkwg 1.66 残余（操作数 staging+串行列和+16 切片 dW/dq1）＞ dhs 0.59（rgemm A 读延迟，A 侧杠杆已文档化）＞ wy 0.70 尾部（wy.u tf32-barred，split-bf16 候选）。

## 干净窗口终判状态

轮询器运行中（120s 间隔×3 连续干净采样触发；co-job pid 3411456 仍驻留）。触发后自动落 `final_adjudicate_clean.json` 并关闭协调行。交错双跑口径（正式）不受影响。
