# G5 干净窗口正式终判归档（2026-10-09 23:39-23:42，co-job 退出后自动触发）

机制：`scripts/adjudicate_when_clean.sh` 轮询器（120s×3 连续干净采样）→ 自动登记协调行（adjclean1009154535）→ 一次性交错终判 → `evidence/G5/final_adjudicate_clean.json` → 行关闭。窗口质量：CLEAN（全程无共载）。

## 正式裁定（fwd，staged，vs fla Triton 纯口径）

| 形状 | ours | fla | 加速比 | 判定 |
|---|---|---|---|---|
| B1T2048 | 0.2916ms | 0.3760ms | **1.289×**（逐对 1.271×） | ✅ **正式 ≥1.1×** |
| B1T8192 | 0.9440ms | 1.1444ms | **1.212×**（逐对 1.214×） | ✅ **正式 ≥1.1×** |
| B4T2048 | 0.9750ms | 0.9228ms | **0.947×**（逐对 0.952×） | ❌ 未达（需 ~16%） |

**修正**：脏窗下的 B4T2048 读数（1.05-1.19×）偏乐观；干净口径为 0.947×。fwd 正式成绩单=**2/3 形状 ≥1.1×，B4T2048 诚实未过**——缺口已归因（F1/F2/F3 的 round-5 leftovers：F2 ≤49.5KB 双 CTA 重设计、F3 寄存器上限占用、F1 残余 0.47ms@T8192 类问题在 B4 同族），排入后续轮任务单。

## 待办（GPU 已 idle）

1. 第 13 轮（wyb+dqkwg，进行中）落地后：**全量补测**（all shapes × {staged fwd, fused flagship, bwd} × 30 reps + run_calib.sh 标定刷新 + 静态分析重算）——按 evidence/G5/adjudication_methodology.md 的完整规格。
2. B4T2048 fwd 的 ~16% 缺口 → 下一轮 fwd 任务单首位。
3. 终报与论文数字表以本干净口径为准更新。
