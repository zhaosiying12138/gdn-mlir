#!/usr/bin/env bash
# gdn 方言构建（小，独立 build 目录；幂等）
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/resource_guard.sh"
ROOT="$HOME/codebase/gdn-mlir"
SRC="$ROOT/dialect"
BUILD="$ROOT/build-dialect"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate gdn

if [ ! -f "$BUILD/build.ninja" ]; then
  cmake -G Ninja -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DMLIR_DIR="$HOME/codebase/llvm-mlir/build/lib/cmake/mlir"
fi

rg_gate_heavy
J=$(rg_jobs); (( J > 6 )) && J=6   # 小项目，少占预算
export PATH=/usr/local/cuda/bin:$PATH
set +e
ninja -C "$BUILD" -j "$J" gdn-opt &
NPID=$!
rg_watchdog_start "$NPID"
wait "$NPID"; RC=$?
set -e
ls -la "$BUILD/bin/gdn-opt" || exit "$RC"
echo "[build_dialect] DONE"
