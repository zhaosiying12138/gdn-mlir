#!/usr/bin/env bash
# conda env `gdn`（G0-3）。phase A 秒级（build_llvm 的依赖）；phase B 慢（torch/fla/z3/jax）。
# 用法: env_gdn.sh [a|b|all]   恢复 = 重跑（pip 已装自动跳过）
set -euo pipefail
PHASE="${1:-all}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx gdn; then
  conda create -y -n gdn python=3.12
fi
conda activate gdn
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [ "$PHASE" = a ] || [ "$PHASE" = all ]; then
  pip install -q pybind11 "nanobind>=2.9,<3" numpy
  python -c "import pybind11, numpy; print('[env_gdn] phase A ok:', pybind11.__version__, numpy.__version__)"
fi

if [ "$PHASE" = b ] || [ "$PHASE" = all ]; then
  pip install -q torch --index-url https://download.pytorch.org/whl/cu130
  if [ ! -d "$HOME/codebase/fla" ]; then
    git clone --depth 1 https://github.com/fla-org/flash-linear-attention.git "$HOME/codebase/fla"
  fi
  ( cd "$HOME/codebase/fla" && pip install -q -e ".[cuda]" )
  pip install -q z3-solver "jax[cpu]"
  python - <<'PY'
import torch, triton, z3, jax, numpy as np
print("[env_gdn] phase B ok:",
      "torch", torch.__version__, "| triton", triton.__version__,
      "| z3", z3.get_version_string(), "| jax", jax.__version__)
PY
fi
