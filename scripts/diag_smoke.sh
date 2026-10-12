#!/usr/bin/env bash
# 复现 runner abort 取头部错误
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH
B="$HOME/codebase/llvm-mlir/build"
T="$HOME/codebase/llvm-mlir/mlir/test/Integration/GPU/CUDA/TensorCore/sm80/transform-mma-sync-matmul-f16-f16-accum.mlir"
"$B/bin/mlir-opt" "$T" \
  -transform-interpreter \
  -test-transform-dialect-erase-schedule \
  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a cubin-features=+ptx87" \
  > /tmp/smoke_payload.mlir
echo "payload size: $(wc -c < /tmp/smoke_payload.mlir)"
"$B/bin/mlir-runner" /tmp/smoke_payload.mlir \
  --shared-libs="$B/lib/libmlir_cuda_runtime.so,$B/lib/libmlir_runner_utils.so,$B/lib/libmlir_c_runner_utils.so,$B/lib/libmlir_float16_utils.so" \
  --entry-point-result=void 2>&1 | head -10
echo "rc=$?"
