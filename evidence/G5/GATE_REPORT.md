# G5 Gate 报告（决战计时）—— 第 1 轮/N：全算子上 GPU 完成，性能未达标，根因已量化（2026-10-09，commits dcb246f/6df3f26/ea03059）

## Gate 判据核验（docs/PLAN.md G5：e2e ≥1.1× fla + 单 kernel 旗舰 + fwd 横比 FlashQLA）

| # | 判据 | 状态 | 证据 |
|---|---|---|---|
| 1 | 基线（纯 Triton 口径 FLA_FLASH_QLA=0） | ✅ | evidence/G5/fla_baseline.json：fwd 0.316/0.873/0.844ms，fwdbwd 1.346/4.036/3.873ms（B1T2048/B1T8192/B4T2048, H16 KV128 bf16）；bwd 靶子=dqkwg ~20%+dhu ~18%+wy_bwd ~10%+重算 ~12% |
| 2 | fwd 横比 FlashQLA | ✅（参照完成） | flashqla_fwd.json：FlashQLA fwd 0.139/0.404/0.355ms（=fla 的 2.2×）；顺带披露其 sm120 两个启动 bug（export 缺失 + smem 超限需 block_DV=64） |
| 3 | **前向全算子 GPU：数值** | ✅ | fwd_report.md：staged 与 persistent 融合旗舰双路径，B1T512H4 与 B1T2048H16 全 PASS（o ≤5.75e-4、ht ≤7.93e-4，bar 5e-3，裕量 7-10×，vs JAX fp64 同种子参考） |
| 4 | 反向全算子 GPU：数值 | 🟡 3/6 | bwd_report.md：dq 2.3e-3✓ dv 3.8e-3✓ dh0 1.4e-4✓；dk 0.66/dβ 0.036/dg 3.7 超 bar——中间量全对（T 2e-4、h_all 3e-3、dvp/dW 5e-3），**根因=bf16 底噪经 dA C×C×C 链放大 ~64×**（fp64 重算 GPU 自身中间量可复现 kernel 至 9e-4）；修复路径已文档化（dA 链 fp32 累加/关键中间 fp32 化） |
| 5 | **e2e ≥1.1× fla** | ❌ 第 1 轮 | fwd staged 0.16-0.20×、fused 0.10-0.18×；bwd 0.02×（标量 GEMM 发射器） |
| 6 | 单 kernel persistent 旗舰 | ✅ 已建成（数值过）/性能待迭代 | 每 chunk 8 个 mma GEMM 阶段、状态常驻 smem 96.5KB（贴 99KB 上限，lifetime 共享 tile） |

## 第 1 轮结论：结构性落后 ~8×，根因全部量化（第 2 轮的弹药）

逐阶段 vs fla 自家 kernel（F1 0.44 / F2 0.56 / F3 0.35ms vs fla 49/71/42µs）——**均匀慢 ~8× = 系统性调度缺口而非单点**：
1. **无 cp.async 流水**：单缓冲整 K 分级搬运，~15 bar.sync/chunk（→ ping-pong 多缓冲 + create_async_groups；G4 mbarrier 协议是现成底座——本轮为保正确性显式降级，是第 2 轮头号杠杆）。
2. **小 tile 延迟bound GEMM**（→ K 循环 retile 接 transform 流水 multibuffer）。
3. **串行 chunk 链**：fused 仅 32-64 CTA，82 SM 一半闲置（→ chunk 并行 scan 切分 + V-tiling）。
4. **标量逐列 solve ~5-7µs/chunk**（→ warp 并行前代换）。
5. bwd 更是标量 GEMM 发射器完全没用张量核（→ 用 G4 验证过的 fragment 机制 mma 化——G2 旋钮库全部现成）。

**可行性判断**：8× 缺口 = 上述 1-4 的乘积效应，其中每一项都有 G2-G4 已验证的机器对应（pipeline 旋钮在 stage kernel 上实测 5×、mma vs 标量 50-100×）。第 2 轮目标：fwd staged 先过 1.1×（杠杆 1+2+4），随后 fused 与 bwd mma 化。

## 工程战果（本轮沉淀）

- 全算子 GPU 生成管线贯通：gdn_full_gen.py（fwd staged+ fused）/ gdn_bwd_gen.py（bwd 5-kernel）+ runtime/verify/bench 三件套；零参数 cubin+公共全局的驱动模式固化。
- 6+4 个真 bug 修复留档（module 三层嵌套杀 outlining、transfer-to-workgroup 必须 create_async_groups、ldmatrix 分型寻址、solve 的 OOB warp 守卫+蝴蝶归约算法错误换 L1 标量递推、linalg.matmul 别名操作数竞态、sm_120a 静态 smem 101376B 硬限、双 vtc ht 存储……全带复现命令）。
- 容差标定（fla bf16 vs JAX fp64 全在舍入下界 ~2× 内）——契约成立且我们 fwd 已在同等水平。

## 计划

G5 第 2 轮（继续，直到达标或证据显示平台性上限）：①ping-pong+async 流水（fwd F1/F3 先行）②solve warp 并行③scan chunk 并行切分④bwd mma 化⑤dk/dβ/dg 的 dA 链 fp32 化。计时终判仍需干净窗口（同窗口交错口径已备）。
