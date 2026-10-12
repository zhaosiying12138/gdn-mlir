#!/usr/bin/env bash
# G0-4b/G0-7: 树内 sm80 transform-mma 集成测试在 5090 冒烟（GPU 共享窗口，秒级、<500MB 显存）
# 证明 MLIR mma.sync 全链路（transform → nvgpu → nvvm → cubin → 运行）在本机可用。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
BUILD="$HOME/codebase/llvm-mlir/build"
OUT="$ROOT/evidence/G0/sm80_smoke.txt"
mkdir -p "$ROOT/evidence/G0"

echo "[smoke] GPU 现状（协调协议检查）:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true

# cuda runtime 库（若缺则小量补建，不动 -j 预算）
ninja -C "$BUILD" -j4 MLIRCUDARuntime 2>/dev/null || ninja -C "$BUILD" -j4 mlir_cuda_runtime 2>/dev/null || true
ls -la "$BUILD/lib/libmlir_cuda_runtime.so" || { echo "缺少 libmlir_cuda_runtime.so"; exit 2; }

T="$HOME/codebase/llvm-mlir/mlir/test/Integration/GPU/CUDA/TensorCore/sm80/transform-mma-sync-matmul-f16-f16-accum.mlir"

# 路 A: lit 直跑（需要测试装置齐全；Unsupported 视为未跑，转手动管线）
if [ -x "$BUILD/bin/llvm-lit" ]; then
  echo "[smoke] 方式A: llvm-lit"
  LITOUT=$(timeout 240 "$BUILD/bin/llvm-lit" -sv "$T" 2>&1 | tee "$OUT")
  if echo "$LITOUT" | grep -q "PASS" && ! echo "$LITOUT" | grep -qE "Unsupported|Failed"; then
    echo "SMOKE PASS (lit)"; exit 0
  fi
  echo "[smoke] lit 未真正通过（Unsupported/Failed），转手动管线"
fi

# 路 B: 手动管线（chip 改 sm_120a）
echo "[smoke] 方式B: 手动管线 cubin-chip=sm_120a"
ERASE=""
if "$BUILD/bin/mlir-opt" --help 2>/dev/null | grep -q erase-schedule; then
  ERASE="-test-transform-dialect-erase-schedule"
fi
set -o pipefail
"$BUILD/bin/mlir-opt" "$T" \
  -transform-interpreter $ERASE \
  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a cubin-features=+ptx87" \
| "$BUILD/bin/mlir-runner" \
    --shared-libs="$BUILD/lib/libmlir_cuda_runtime.so,$BUILD/lib/libmlir_runner_utils.so,$BUILD/lib/libmlir_c_runner_utils.so" \
    --entry-point-result=void \
| tee "$OUT"
echo "SMOKE PASS (manual)"
