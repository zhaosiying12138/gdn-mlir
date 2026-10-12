#!/usr/bin/env bash
# 幂等 LLVM/MLIR 构建（G0-4）。恢复 = 直接重跑本脚本（worktree/配置/增量编译全部跳过已完成部分）。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/resource_guard.sh"
export PATH=/usr/local/cuda/bin:$PATH   # nvcc/ptxas：configure 与 ninja 都需要

SRC_REPO="$HOME/codebase/llvm-project"   # 现有 clone（riscv-yushuxin-v2 分支，绝不触碰）
WT="$HOME/codebase/llvm-mlir"            # 新 worktree（新分支 gdnmlir ← origin/main）
BUILD="$WT/build"

echo "[build_llvm] stage1: worktree（不触碰现有 riscv 分支）"
cd "$SRC_REPO"
if [ ! -e "$WT/.git" ] && [ ! -f "$WT/.git" ]; then
  git fetch origin main
  git worktree add "$WT" -b gdnmlir origin/main
fi
git -C "$WT" log --oneline -1 || true

echo "[build_llvm] stage2: configure（python 绑定=conda gdn）"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate gdn
if [ ! -f "$BUILD/build.ninja" ] || ! grep -q "MLIR_RUN_CUDA_SM80_TESTS:BOOL=ON" "$BUILD/CMakeCache.txt" 2>/dev/null; then
  cmake -G Ninja -S "$WT/llvm" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DMLIR_RUN_CUDA_SM80_TESTS=ON \
    -DLLVM_INCLUDE_TESTS=ON \
    -DCUDACXX=/usr/local/cuda/bin/nvcc \
    -Dnanobind_DIR="$(python -m nanobind --cmake_dir 2>/dev/null || echo '')" \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir 2>/dev/null || echo '')" || { rm -rf "$BUILD"; echo "[build_llvm] configure 失败，已清空 build 目录（修复后重跑）"; exit 1; }
fi

echo "[build_llvm] stage3: ninja（动态 -j，watchdog 保护）"
rg_gate_heavy
J=$(rg_jobs); (( J > 10 )) && J=10
echo "[build_llvm] jobs=$J avail=$(rg_avail_gb)G start=$(date +%F_%T)"
export PATH=/usr/local/cuda/bin:$PATH
set +e
ninja -C "$BUILD" -j "$J" mlir-opt mlir-runner FileCheck count not mlir_cuda_runtime &
NPID=$!
rg_watchdog_start "$NPID"
wait "$NPID"; RC=$?
set -e
if (( RC != 0 )); then
  echo "[build_llvm] ninja rc=$RC（watchdog 或中断所致则直接重跑本脚本续编）"
  exit "$RC"
fi

echo "[build_llvm] stage4: Python 绑定目标（名称兼容两代）"
ninja -C "$BUILD" -j "$J" MLIRPythonModules 2>/dev/null || \
  ninja -C "$BUILD" -j "$J" mlir-python-modules 2>/dev/null || \
  echo "[build_llvm] WARN: python targets 目标名未命中，稍后用 ninja -t targets 自查"

echo "[build_llvm] DONE $(date +%F_%T)"
ls -la "$BUILD/bin/mlir-opt" "$BUILD/bin/mlir-runner" 2>/dev/null || true
du -sh "$BUILD" || true
