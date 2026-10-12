#!/usr/bin/env bash
# scripts/repro.sh -- G7-B one-command-per-gate reproduction driver (G0..G6).
#
# Wires the EXISTING idempotent stage drivers (scripts/*.sh + tools/*.py) into
# per-gate targets.  No driver is reimplemented here: every step is exactly the
# command the corresponding gate report documents as its reproduce entry
# (gate -> report -> artifacts: EVIDENCE_INDEX.md).
#
# Usage:
#   scripts/repro.sh <g0|g1|g2|g3|g4|g5|g6|all|list> [--keep-going]
#   make repro-g0 .. repro-g6 | repro-all | repro-all-keepgoing
#
# Behaviour:
#   * every step prints a banner (command, CPU/GPU marker, expected wall time)
#     and tee's its output to logs/repro/<gate>.log
#   * every gate ends with a PASS/FAIL summary.  FAIL = any step exit code != 0
#     OR a post-run JSON verdict check failing.  Several harness tools always
#     exit 0 by design (gdn_full_verify / gdn_full_bench / gdn_final_adjudicate
#     / gdn_time / gdn_l2_batch / run_g3.sh); for those the freshly written
#     JSON / log tail is inspected instead of the exit code.
#   * `all` runs G0->G6 in order; by default it STOPS at the first FAIL;
#     --keep-going continues and prints the full table at the end.
#   * regenerable repro outputs land in evidence/repro/<gate>/ (gitignored).
#     Drivers that own canonical evidence paths overwrite them in place --
#     that is their documented idempotent behaviour (run_calib.sh, run_g3.sh,
#     gdn_time.py, gdn_verify.py, g6_fla_autotune_cost.py); the landed numbers
#     stay in git history (e.g. `git checkout -- evidence/G3` to restore).
#   * GPU steps only PRINT an nvidia-smi compute-apps snapshot for the log
#     (shared-GPU discipline: ~/codebase/GPU-COORDINATION.md -- register your
#     window there before long GPU gates on a shared machine).  The G5 FINAL
#     adjudication officially wants a CLEAN window: use
#     scripts/adjudicate_when_clean.sh for that.
#
# Fresh-machine prerequisites (see README "Quickstart"):
#   conda env `gdn`          (make env-a && make env-b)
#   CUDA toolkit             /usr/local/cuda (nvcc + ptxas on PATH via drivers)
#   LLVM/MLIR build          make llvm   (~/codebase/llvm-mlir/build)
#   clang (CMake C/C++ compiler for the two CMake builds)
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$HOME/miniforge3/envs/gdn/bin/python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PATH="/usr/local/cuda/bin:$PATH"
mkdir -p logs/repro evidence/repro

LOG=""          # current gate log (logs/repro/<gate>.log)
STEP_RC=0
KEEP_GOING=0

# ---------------------------------------------------------------- helpers ---
banner() { # gate marker wall cmd...
  local gate="$1" mark="$2" wall="$3"; shift 3
  echo
  echo "===[repro.$gate]== $mark"
  echo "    cmd    : $*"
  echo "    expect : ~$wall"
  echo "    log    : $LOG"
}

step() { # gate marker wall cmd...
  local gate="$1" mark="$2" wall="$3"; shift 3
  banner "$gate" "$mark" "$wall" "$@"
  "$@" 2>&1 | tee -a "$LOG"
  STEP_RC=${PIPESTATUS[0]}
  if [ "$STEP_RC" -ne 0 ]; then
    echo "===[repro.$gate]== STEP FAILED rc=$STEP_RC: $*" | tee -a "$LOG"
  fi
  return "$STEP_RC"
}

gpu_snapshot() {
  { echo "--- nvidia-smi compute-apps $(date +%F_%T) ---"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
  } | tee -a "$LOG"
}

# Generic JSON verdict walker: FAIL if any status/verdict/overall cell holds a
# bad value (covers gdn_full_verify / gdn_full_bench / bwd reports).
check_json() { # file what
  $PY - "$1" "$2" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys
BAD = {"FAIL", "WORKER_CRASH", "TIMEOUT", "ERROR", "CRASH", "fail"}
def walk(o, hits):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, str) and k in ("status", "verdict", "overall") and v in BAD:
                hits.append(f"{k}={v}")
            walk(v, hits)
    elif isinstance(o, list):
        for v in o:
            walk(v, hits)
hits = []
try:
    walk(json.load(open(sys.argv[1])), hits)
except Exception as e:  # noqa: BLE001
    print(f"[repro] JSON CHECK {sys.argv[2]}: CANNOT PARSE {sys.argv[1]}: {e}")
    sys.exit(1)
print(f"[repro] JSON CHECK {sys.argv[2]}: {sys.argv[1]}: "
      f"{len(hits)} bad status cell(s)" + (f" {hits[:8]}" if hits else ""))
sys.exit(1 if hits else 0)
PYEOF
  return ${PIPESTATUS[0]}
}

# file-exists check for tools that print their verdict to stdout only.
check_file() { # file what
  if [ -s "$1" ]; then
    echo "[repro] FILE CHECK $2: $1 present ($(wc -c <"$1") bytes)" | tee -a "$LOG"
    return 0
  fi
  echo "[repro] FILE CHECK $2: $1 MISSING/EMPTY" | tee -a "$LOG"
  return 1
}

# run_g3.sh always exits 0; its own summary line carries the verdict.
check_run_g3() {
  local line
  line="$(grep -a "=== run_g3:" "$LOG" | tail -1)"
  echo "[repro] run_g3 summary: $line" | tee -a "$LOG"
  case "$line" in *"ok=20 fail=0"*) return 0;; esac
  return 1
}

# gdn_time.py writes {results: {name: {status: OK|...}}} and exits 0.
check_gdn_time() { # json
  $PY - "$1" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
ok = sum(1 for r in d["results"].values() if r.get("status") == "OK")
n = len(d["results"])
# G3's gate accepted 18/20 on the shared GPU (ws/tma later fixed in G4);
# a healthy rerun should now time all 20.
print(f"[repro] gdn_time: {ok}/{n} configs timed OK "
      f"(gate bar: >=18; G4-fixed ws/tma should make it 20)")
sys.exit(0 if ok >= 18 else 1)
PYEOF
  return ${PIPESTATUS[0]}
}

# gdn_l2_batch.py exits 0; INDEX.json carries ok/total counts.
check_l2_index() { # json
  $PY - "$1" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[repro] l2_batch: ok_count={d['ok_count']} total_count={d['total_count']}")
sys.exit(0 if d["ok_count"] == d["total_count"] == 20 else 1)
PYEOF
  return ${PIPESTATUS[0]}
}

# gdn_verify.py exits 0; PASS = every config's n_over_threshold == 0 (20 expected).
check_g4() { # json
  $PY - "$1" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
res = d.get("results", {})
np_ = sum(1 for r in res.values() if r.get("n_over_threshold") == 0)
bad = [k for k, r in res.items() if r.get("n_over_threshold") not in (0, None)]
print(f"[repro] gdn_verify: {np_}/{len(res)} configs within threshold "
      f"(expect 20/20)" + (f"; over-threshold: {bad}" if bad else ""))
sys.exit(0 if np_ >= 20 and not bad else 1)
PYEOF
  return ${PIPESTATUS[0]}
}

# gdn_final_adjudicate.py exits 0; a fresh default run (--shapes
# B1T2048,B1T8192,B4T2048) must carry all 3 shapes with finite speedups
# (the >=1.1x claim itself is adjudicated on a CLEAN window via
# scripts/adjudicate_when_clean.sh; here we verify the run completed).
check_adjudicate() { # json
  $PY - "$1" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
res = d.get("results", {})
ok = len(res) >= 3
for name, r in sorted(res.items()):
    sp = r.get("speedup_ratio_of_medians")
    if not isinstance(sp, (int, float)):
        print(f"[repro] adjudicate: {name}: MISSING/INCOMPLETE")
        ok = False
    else:
        print(f"[repro] adjudicate: {name}: speedup {sp:.3f}x "
              f"(window {'clean' if r.get('window_clean') else 'NOISY'})")
missing = {"B1T2048", "B1T8192", "B4T2048"} - set(res)
if missing:
    print(f"[repro] adjudicate: missing shapes {sorted(missing)}")
    ok = False
sys.exit(0 if ok else 1)
PYEOF
  return ${PIPESTATUS[0]}
}

# Build one bwd chain: generate (gdn_bwd_gen.py) + compile every manifest
# stage through the existing drivers (scripts/run_l2.sh + tools/gdn_smem_fix.py)
# -- the same wiring as evidence/G5/scratch/r4_build_chain.sh, but driven off
# the manifest so the round-5 stage list (caryA/caryM/bpre/dhs) is picked up
# from the generator itself.
build_bwd_chain() { # out_dir shape
  local out_dir="$1" shape="$2" kernels k rc
  echo "[repro.g5] bwd chain gen -> $out_dir (shape $shape)" | tee -a "$LOG"
  "$PY" tools/gdn_bwd_gen.py --shape "$shape" --out "$out_dir" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}; [ "$rc" -ne 0 ] && return "$rc"
  kernels="$("$PY" -c "import json;print(' '.join(s['dir'] for s in json.load(open('$out_dir/bwd_manifest.json'))['stages']))")" \
    || return 1
  for k in $kernels; do
    echo "[repro.g5] bwd build stage $k" | tee -a "$LOG"
    bash scripts/run_l2.sh "$out_dir/$k" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}; [ "$rc" -ne 0 ] && return "$rc"
    "$PY" tools/gdn_smem_fix.py "$out_dir/$k" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}; [ "$rc" -ne 0 ] && return "$rc"
  done
  return 0
}

gate_summary() { # gate status expect
  echo
  echo "===[repro.$1]=================================================="
  echo "    $1: $2"
  echo "    expected: $3"
  echo "==============================================================[repro.$1]"
}

# ------------------------------------------------------------------ gates ---
g0() {
  LOG="logs/repro/g0.log"; : > "$LOG"
  echo "===[repro.g0]== infra + reference harvest (mixed: 1 GPU smoke, rest CPU)" | tee -a "$LOG"
  echo "    expected total: ~2-2.5 h COLD (LLVM build dominates: ~90 min); ~15 min warm" | tee -a "$LOG"
  step g0 "conda env phase A (pybind11/nanobind/numpy) [CPU, seconds]" "10 s" \
    bash scripts/env_gdn.sh a || return 1
  step g0 "conda env phase B (torch cu130 / fla / z3 / jax[cpu]) [CPU+network, 15-30 min cold, no-op when installed]" "20 min" \
    bash scripts/env_gdn.sh b || return 1
  step g0 "LLVM/MLIR worktree + build (mlir-opt/runner/cuda-runtime) [CPU, ~90 min cold, incremental ~min]" "90 min" \
    bash scripts/build_llvm.sh || return 1
  gpu_snapshot
  step g0 "in-tree sm80 transform-mma smoke on real GPU [GPU, ~1 min]" "1 min" \
    bash scripts/smoke_sm80_manual.sh || return 1
  step g0 "fla Triton reference PTX/SASS harvest, 22 configs [CPU, ~10 min]" "10 min" \
    bash scripts/harvest_fla.sh || return 1
  step g0 "FlashQLA TileLang reference harvest, 5 kernels [CPU, ~15 min cold env]" "15 min" \
    bash scripts/harvest_qla.sh || return 1
  check_file evidence/G0/fla_ptx/SUMMARY.md "g0 fla harvest summary" || return 1
  check_file evidence/G0/qla_ptx/SUMMARY.md "g0 qla harvest summary" || return 1
  gate_summary g0 "PASS" "harvests 22/22 + 5/5; gate report evidence/G0/GATE_REPORT.md"
}

g1() {
  LOG="logs/repro/g1.log"; : > "$LOG"
  echo "===[repro.g1]== dialect + semantics + CPU three-way cross-check (all CPU)" | tee -a "$LOG"
  step g1 "JAX executable spec self-consistency suite [CPU]" "5 min" \
    "$PY" tools/gdn_ref_check.py || return 1
  step g1 "MLIR L1 forward numeric check (7 cases vs JAX) [CPU]" "3 min" \
    "$PY" tools/gdn_l1_check.py || return 1
  step g1 "MLIR L1 backward numeric check (3 npz x 6 grads) [CPU]" "3 min" \
    "$PY" tools/gdn_l1_check.py --bwd || return 1
  step g1 "gdn dialect build + gdn-opt smoke [CPU]" "3 min" \
    bash scripts/build_dialect.sh || return 1
  gate_summary g1 "PASS" "JAX spec 44/44; L1 fwd 7/7 (~6e-7); bwd 3npz x 6 grads; dialect smoke green"
}

g2() {
  LOG="logs/repro/g2.log"; : > "$LOG"
  echo "===[repro.g2]== knob library + PTX coverage matrix (all CPU)" | tee -a "$LOG"
  step g2 "generate + compile the 20-config knob matrix (PTX/SASS/ptxas) [CPU, 10-30 min]" "20 min" \
    "$PY" tools/gdn_l2_batch.py || return 1
  check_l2_index evidence/G2/gen/INDEX.json || return 1
  step g2 "coverage analysis: T1 family inventory + matrix + t1-set [CPU]" "2 min" \
    "$PY" tools/gdn_coverage.py all || return 1
  check_file evidence/G2/coverage_matrix.md "g2 coverage matrix" || return 1
  gate_summary g2 "PASS" "20/20 configs compile; T1 feature tier covered; both poles expressible; hatch 0/5"
}

g3() {
  LOG="logs/repro/g3.log"; : > "$LOG"
  echo "===[repro.g3]== calibration + timed configs + Z3 solver (1 CPU gen + 2 GPU + 1 CPU)" | tee -a "$LOG"
  step g3 "regenerate noargs variants of the 20 G2 configs + compile [CPU; WIPES+regenerates evidence/G3/gen_noargs]" "20 min" \
    bash scripts/run_g3.sh || true
  check_run_g3 || return 1
  gpu_snapshot
  step g3 "sm_120 calibration microbench [GPU, ~1 min, <2GB; OVERWRITES evidence/G3/calibration.json]" "1 min" \
    bash scripts/run_calib.sh || return 1
  step g3 "timed GPU execution of the 20 configs [GPU, 10-30 min; OVERWRITES evidence/G3/config_times.json]" "20 min" \
    "$PY" tools/gdn_time.py || return 1
  check_gdn_time evidence/G3/config_times.json || return 1
  step g3 "Z3 solver + calibrated cost model ranking validation (Spearman gate) [CPU; reads the three canonical files above]" "2 min" \
    "$PY" tools/gdn_solver.py --validate || return 1
  gate_summary g3 "PASS" "Spearman rho >= 0.7 (landed run: 0.8962; solver deterministic byte-identical)"
}

g4() {
  LOG="logs/repro/g4.log"; : > "$LOG"
  echo "===[repro.g4]== GPU numeric correctness of the 20 configs (GPU)" | tee -a "$LOG"
  gpu_snapshot
  step g4 "regenerate + recompile + verify all 20 configs vs fp64 twins, seed 0 [GPU short bursts, <2GB; OVERWRITES evidence/G4/verify_results.json]" "30 min" \
    "$PY" tools/gdn_verify.py --seed 0 || return 1
  check_g4 evidence/G4/verify_results.json || return 1
  step g4 "second data permutation, seed 7 [GPU]" "30 min" \
    "$PY" tools/gdn_verify.py --seed 7 --out evidence/repro/G4/verify_results_seed7.json || return 1
  check_g4 evidence/repro/G4/verify_results_seed7.json || return 1
  gate_summary g4 "PASS" "20/20 PASS dual-seed (bf16/f16-io <= 5e-3, f32-io <= 1e-3 vs fp64 twins)"
}

g5() {
  LOG="logs/repro/g5.log"; : > "$LOG"
  local R5="evidence/repro/G5"
  mkdir -p "$R5"
  echo "===[repro.g5]== full-op fwd+bwd verify + bench + adjudication (GPU; ~1-1.5 h total)" | tee -a "$LOG"
  echo "    NOTE: the official clean-window adjudication is scripted separately:" | tee -a "$LOG"
  echo "          scripts/adjudicate_when_clean.sh (G5 verdict deferred to it on shared GPUs)" | tee -a "$LOG"
  gpu_snapshot
  step g5 "fwd numeric verify, staged + fused paths (self-builds evidence/G5/gen_fwd kernels) [GPU]" "20 min" \
    "$PY" tools/gdn_full_verify.py --out "$R5/fwd_verify.json" || return 1
  check_json "$R5/fwd_verify.json" "g5 fwd verify" || return 1
  build_bwd_chain "$R5/gen_bwd_T512" "1,512,4,128,128" || return 1
  build_bwd_chain "$R5/gen_bwd_T2048" "1,2048,16,128,128" || return 1
  step g5 "bwd numeric verify, small shape (6 grads vs fp64 twin; exit code is the verdict) [GPU]" "5 min" \
    "$PY" tools/gdn_bwd_verify.py --gen-root "$R5/gen_bwd_T512" --shape 1,512,4,128,128 --out "$R5/bwd_verify_T512.json" || return 1
  step g5 "bwd numeric verify, bench shape [GPU]" "10 min" \
    "$PY" tools/gdn_bwd_verify.py --gen-root "$R5/gen_bwd_T2048" --shape 1,2048,16,128,128 --out "$R5/bwd_verify_T2048.json" || return 1
  step g5 "fwd e2e bench vs fla baselines (3 shapes, self-builds) [GPU]" "30 min" \
    "$PY" tools/gdn_full_bench.py --out "$R5/fwd_bench.json" || return 1
  check_json "$R5/fwd_bench.json" "g5 fwd bench" || return 1
  step g5 "bwd e2e bench vs fla baseline @B1T2048H16 [GPU]" "10 min" \
    "$PY" tools/gdn_bwd_bench.py --gen-root "$R5/gen_bwd_T2048" --shape 1,2048,16,128,128 --out "$R5/bwd_bench_T2048.json" || return 1
  check_json "$R5/bwd_bench_T2048.json" "g5 bwd bench" || return 1
  step g5 "INTERLEAVED final adjudication vs fla (3 shapes x 5 pairs) [GPU; clean window preferred]" "20 min" \
    "$PY" tools/gdn_final_adjudicate.py --out "$R5/final_adjudicate.json" || return 1
  check_adjudicate "$R5/final_adjudicate.json" || return 1
  gate_summary g5 "PASS" "fwd 3/3 >= 1.1x interleaved (landed: 1.592/1.05-1.19/1.253-1.312x); bwd honest miss 0.126x with 7.2x trajectory (numbers re-adjudicated on a clean window)"
}

g6() {
  LOG="logs/repro/g6.log"; : > "$LOG"
  echo "===[repro.g6]== paper-grade evidence chain: search-cost measurements (1 CPU + 1 GPU)" | tee -a "$LOG"
  step g6 "Z3 solver wall-time + config counts rerun (deterministic, CPU) [writes evidence/G6/scratch/]" "5 min" \
    bash tools/g6/g6_solver_timing.sh || return 1
  check_file evidence/G6/scratch/solver_timing_summary.json "g6 solver timing summary" || return 1
  gpu_snapshot
  step g6 "fla on-device autotune cost, cold vs warm, B1T2048 fwd+bwd [GPU; own TRITON_CACHE_DIR, co-job safe; OVERWRITES evidence/G6/fla_autotune_cost.json]" "2 min" \
    "$PY" tools/g6/g6_fla_autotune_cost.py || return 1
  check_file evidence/G6/fla_autotune_cost.json "g6 fla autotune cost" || return 1
  gate_summary g6 "PASS" "ours ~0.8 s CPU deterministic vs fla ~39.8 s GPU autotune per shape fwd+bwd (~50x)"
}

# ------------------------------------------------------------------ main ----
GATES="g0 g1 g2 g3 g4 g5 g6"

gate_desc() {
  case "$1" in
    g0) echo "infra + env + LLVM build + sm80 GPU smoke + fla/FlashQLA PTX harvest [mixed, ~2-2.5 h cold]";;
    g1) echo "JAX spec + MLIR L1 fwd/bwd + dialect build [CPU, ~15 min]";;
    g2) echo "20-config knob matrix + PTX coverage analysis [CPU, ~25 min]";;
    g3) echo "noargs regen + calibration + 20-config timing + Spearman gate [2 GPU steps, ~45 min]";;
    g4) echo "20-config GPU numeric verify, dual seed [GPU, ~1 h]";;
    g5) echo "full-op fwd/bwd verify + bench + interleaved adjudication [GPU, ~1-1.5 h]";;
    g6) echo "search-cost vs autotune: solver timing + fla autotune cost [CPU + 1 GPU step, ~10 min]";;
  esac
}

run_gate() {
  local g="$1" rc=0
  echo
  echo "############################################################"
  echo "#  REPRO GATE $g : $(gate_desc "$g")"
  echo "############################################################"
  "$g" || rc=$?
  if [ "$rc" -ne 0 ]; then
    gate_summary "$g" "FAIL (rc=$rc) -- see $LOG" ""
  fi
  return "$rc"
}

ARGS=()
for a in "$@"; do
  if [ "$a" = "--keep-going" ]; then KEEP_GOING=1; else ARGS+=("$a"); fi
done
CMD="${ARGS[0]:-}"

case "$CMD" in
  list)
    echo "repro gates (in order):"
    for g in $GATES; do echo "  $g : $(gate_desc "$g")"; done
    echo "  all : G0->G6 in order (default: stop on first FAIL; --keep-going to continue)"
    exit 0 ;;
  g0|g1|g2|g3|g4|g5|g6)
    run_gate "$CMD"; exit $? ;;
  all)
    declare -A STATUS
    OVERALL=0
    for g in $GATES; do
      if run_gate "$g"; then STATUS[$g]=PASS; else
        STATUS[$g]=FAIL; OVERALL=1
        if [ "$KEEP_GOING" -ne 1 ]; then
          echo "[repro.all] STOPPING at first FAIL ($g). Rerun with --keep-going to continue." >&2
          break
        fi
      fi
    done
    echo
    echo "#################### REPRO SUMMARY ####################"
    for g in $GATES; do
      printf "  %-4s %s\n" "$g" "${STATUS[$g]:-skipped}"
    done
    echo "  overall: $([ "$OVERALL" -eq 0 ] && echo PASS || echo FAIL) (keep_going=$KEEP_GOING)"
    echo "#######################################################"
    exit "$OVERALL" ;;
  *)
    sed -n '2,40p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
    echo "usage: scripts/repro.sh <g0|g1|g2|g3|g4|g5|g6|all|list> [--keep-going]"
    exit 2 ;;
esac
