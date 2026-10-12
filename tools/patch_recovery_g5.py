#!/usr/bin/env python3
"""G5 第1轮 RECOVERY 表更新（幂等）。"""
import pathlib

P = pathlib.Path("/home/zhaosiying/codebase/gdn-mlir/RECOVERY.md")
s = P.read_text()
old = "| G5 GPU 计时（超 fla） | 见 docs/PLAN.md G5 | **next**（需干净窗口） | |"
new = ("| G5-A 基线+参考 | gdn_bench_flaptx.py 等 | **done** | fla_baseline.json 等 |\n"
       "| G5-B 前向全算子 GPU | gdn_full_gen.py | **数值✅（staged+旗舰）/性能 0.16-0.20×（第2轮迭代中）** | fwd_report.md |\n"
       "| G5-C 反向全算子 GPU | gdn_bwd_gen.py | **数值 3/6（dA链bf16放大，修复路径已定）/性能 0.02×（待mma化）** | bwd_report.md |\n"
       "| G5 gate | — | **第1轮/N：判据5未达，迭代继续** | evidence/G5/GATE_REPORT.md |")
if old in s:
    s = s.replace(old, new)
elif "第1轮/N" not in s:
    # 表尾兜底追加
    lines = s.rstrip().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("|"):
            lines[i + 1:i + 1] = [x for x in new.splitlines()]
            break
    s = "\n".join(lines) + "\n"
P.write_text(s)
print("recovery updated")
