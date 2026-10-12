#!/usr/bin/env bash
# G3-A: generate --driver noargs variants of all G2 configs + compile + smem fix.
# CPU-only (no GPU). Idempotent: wipes and regenerates each target dir.
#
# Usage: scripts/run_g3.sh [gen-root]
#   gen-root defaults to evidence/G3/gen_noargs

set -euo pipefail
REPO=~/codebase/gdn-mlir
cd "$REPO"

ROOT="${1:-evidence/G3/gen_noargs}"
mkdir -p "$ROOT"

ok=0; fail=0
for cfg in evidence/G2/gen/*/config.json; do
  d=$(dirname "$cfg")
  name=$(basename "$d")
  out="$ROOT/$name"
  rm -rf "$out"
  echo "=== $name ==="
  if python3 tools/gdn_l2_gen.py --json "$cfg" --driver noargs --out "$out" \
      && scripts/run_l2.sh "$out" > "$out/build.log" 2>&1 \
      && python3 tools/gdn_smem_fix.py "$out" >> "$out/build.log" 2>&1; then
    tail -2 "$out/build.log" | head -1
    ok=$((ok+1))
  else
    echo "FAILED: see $out/build.log"
    tail -5 "$out/build.log" || true
    fail=$((fail+1))
  fi
done
echo "=== run_g3: ok=$ok fail=$fail under $ROOT ==="
