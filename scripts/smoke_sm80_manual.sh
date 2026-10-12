#!/usr/bin/env bash
# sm80 transform-mma 冒烟：手动管线（固定参数，不依赖 lit）
set -uo pipefail
BUILD="$HOME/codebase/llvm-mlir/build"
T="$HOME/codebase/llvm-mlir/mlir/test/Integration/GPU/CUDA/TensorCore/sm80/transform-mma-sync-matmul-f16-f16-accum.mlir"
OUT="$HOME/codebase/gdn-mlir/evidence/G0/sm80_smoke.txt"
export PATH=/usr/local/cuda/bin:$PATH
echo "[smoke-B] GPU 现状:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true

set -o pipefail
"$BUILD/bin/mlir-opt" "$T" \
  -transform-interpreter \
  -test-transform-dialect-erase-schedule \
  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a cubin-features=+ptx87" \
| "$BUILD/bin/mlir-runner" \
    --shared-libs="$BUILD/lib/libmlir_cuda_runtime.so,$BUILD/lib/libmlir_runner_utils.so,$BUILD/lib/libmlir_c_runner_utils.so,$BUILD/lib/libmlir_float16_utils.so" \
    --entry-point-result=void \
| tee "$OUT"
rc=$?
echo "[smoke-B] rc=$rc"
exit $rc
