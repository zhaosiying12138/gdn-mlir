#!/bin/bash
# G6: re-run the G3 Z3 solver for wall-time + config counts (CPU only).
# Writes evidence/G6/scratch/solver_{stage}_{form}.json + a summary json.
set -euo pipefail
cd "$HOME/codebase/gdn-mlir"
export OMP_NUM_THREADS=4
mkdir -p evidence/G6/scratch
PY="$HOME/miniforge3/envs/gdn/bin/python"

run_one () {  # stage form V
  local stage="$1" form="$2" V="$3"
  local out="evidence/G6/scratch/solver_${stage}_${form}.json"
  local t0 t1
  t0=$(date +%s.%N)
  if [ "$stage" = "kkt" ]; then
    $PY tools/gdn_solver.py --solve --stage kkt --K 128 --B 4 --T 8192 --H 16 \
        --mode strict --form "$form" --top 5 --out "$out" > "/tmp/g6_solve_${stage}_${form}.log" 2>&1
  else
    $PY tools/gdn_solver.py --solve --stage o --K 128 --V "$V" --B 4 --T 8192 --H 16 \
        --mode strict --form "$form" --top 5 --out "$out" > "/tmp/g6_solve_${stage}_${form}.log" 2>&1
  fi
  t1=$(date +%s.%N)
  echo "${stage} ${form} V=${V} wall=$(echo "$t1 $t0" | awk '{printf "%.3f", $1-$2}')s"
}

echo "== G6 solver timing (RTX host CPU; OMP_NUM_THREADS=4) =="
run_one kkt v2 ""
run_one kkt v3 ""
# NOTE: --stage o --V 64 (204-config subspace from the G3 report table)
# crashes in this environment with a Z3 partial-model artifact
# (m[zv] -> None under rlimit=20e6; tools/gdn_solver.py:187) -- the two
# canonical spaces (kkt K=128, o K=128 V=128) reproduce exactly.
run_one o v2 128
run_one o v3 128

$PY - <<'PYEOF'
import json, glob
summary = []
for p in sorted(glob.glob("evidence/G6/scratch/solver_*.json")):
    if p.endswith("solver_timing_summary.json"):
        continue
    d = json.load(open(p))
    summary.append({
        "file": p,
        "stage": d["stage"], "form": d["model_form"],
        "dims": d["dims"],
        "space_size": d["space_size"],
        "enumeration_seconds": d["enumeration_seconds"],
        "best": d["ranking"][0]["name"],
        "best_predicted_ms": d["ranking"][0]["predicted_ms"],
    })
json.dump({"runs": summary}, open("evidence/G6/scratch/solver_timing_summary.json", "w"), indent=1)
for s in summary:
    print(s)
PYEOF
