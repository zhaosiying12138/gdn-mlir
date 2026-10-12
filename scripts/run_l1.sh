#!/usr/bin/env bash
# run_l1.sh -- G1-B: lower an L1 chunked-GDN .mlir module to LLVM and run it on
# CPU via mlir-runner. Prints the kernel memrefs (o, then ht) to stdout.
#
# Usage: scripts/run_l1.sh <input.mlir> [output.lowered.mlir]
#
# Pipeline notes (see evidence/G1/l1_harness_report.md):
#  - -expand-strided-metadata -lower-affine are REQUIRED for the strided
#    rank-reduced subviews the generator emits (recipe from
#    mlir/test/Integration/Dialect/Linalg/CPU/rank-reducing-subview.mlir).
#  - math.exp2 lowers via -convert-math-to-llvm to llvm.intr.exp2, which the
#    JIT resolves against libm pulled in by the shared runtime libs.
set -euo pipefail

BUILD="$HOME/codebase/llvm-mlir/build"
IN="${1:?usage: run_l1.sh <in.mlir> [out.mlir]}"
OUT="${2:-${IN%.mlir}.llvm.mlir}"

OPT="$BUILD/bin/mlir-opt"
RUNNER="$BUILD/bin/mlir-runner"
LIBS="$BUILD/lib/libmlir_runner_utils.so,$BUILD/lib/libmlir_c_runner_utils.so,$BUILD/lib/libmlir_float16_utils.so"

"$OPT" "$IN" \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -expand-strided-metadata \
  -lower-affine \
  -convert-math-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-cf-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o "$OUT"

"$RUNNER" "$OUT" --shared-libs="$LIBS" --entry-point-result=void
