#!/usr/bin/env bash
# 现场状态（WSL 重启后第一个命令）
set -u
ROOT="$HOME/codebase/gdn-mlir"
echo "=== gdn-mlir 现场状态 $(date +%F_%T) ==="
echo "--- 资源 ---"
free -g | head -2
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null || echo "(nvidia-smi 不可用)"
df -h /home | tail -1
echo "--- conda envs ---"
ls "$HOME/miniforge3/envs" 2>/dev/null
echo "--- gdn env 包 ---"
"$HOME/miniforge3/envs/gdn/bin/python" -c "import torch" 2>/dev/null && echo torch-ok || echo torch-missing
"$HOME/miniforge3/envs/gdn/bin/python" -c "import z3" 2>/dev/null && echo z3-ok || echo z3-missing
echo "--- LLVM/MLIR ---"
WT="$HOME/codebase/llvm-mlir"
[ -d "$WT/.git" ] || [ -f "$WT/.git" ] && echo "worktree: 存在 $(git -C "$WT" log --oneline -1 2>/dev/null)" || echo "worktree: 未创建"
[ -f "$WT/build/build.ninja" ] && echo "configure: 已配置" || echo "configure: 未配置"
[ -x "$WT/build/bin/mlir-opt" ] && echo "mlir-opt: 已构建 $(du -sh "$WT/build" 2>/dev/null | cut -f1)" || echo "mlir-opt: 未构建"
[ -x "$WT/build/bin/mlir-runner" ] && echo "mlir-runner: 已构建" || echo "mlir-runner: 未构建"
echo "--- 重构建槽位 ---"
[ -d /tmp/gdnmlir_heavy.lock ] && echo "占用中" || echo "空闲"
echo "--- watchdog 尾部 ---"
tail -5 "$ROOT/logs/watchdog.log" 2>/dev/null || echo "(无日志)"
echo "--- 阶段表见 RECOVERY.md ---"
