#!/usr/bin/env bash
# scripts/adjudicate_when_clean.sh -- G5 FINAL clean-window adjudication hook (G7-B).
#
# The G5 forward verdict (3/3 shapes >= 1.1x vs fla, interleaved same-window
# measurement) is robust to co-load, but the official CLEAN-window rerun was
# deferred while a co-job holds the GPU (see ~/codebase/GPU-COORDINATION.md).
# This hook polls nvidia-smi compute-apps; when no other compute process has
# been resident for ADJ_CLEAN_RUNS consecutive samples, it:
#   1. registers a coordination row (protocol step 2) BEFORE launching,
#   2. runs tools/gdn_final_adjudicate.py ONCE
#      -> evidence/G5/final_adjudicate_clean.json,
#   3. marks the coordination row done (or failed).
#
# Usage:
#   scripts/adjudicate_when_clean.sh --check   # dry run: window state only, no run
#   scripts/adjudicate_when_clean.sh           # poll + run once
#
# Env (optional):
#   ADJ_POLL_SEC   (default 120)  seconds between polls
#   ADJ_CLEAN_RUNS (default 3)    consecutive clean samples required
#
# Lock: /tmp/gdn_adjudicate_clean.lock (tmpfs by design: crash => auto-gone).
# Discipline: NEVER kills or signals any co-job process. Long-running: keep it
# in a harness background task holding the WSL session (nohup does not
# survive wsl.exe session exit -- RECOVERY.md lesson #4).
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$HOME/miniforge3/envs/gdn/bin/python"
COORD="$HOME/codebase/GPU-COORDINATION.md"
OUT="evidence/G5/final_adjudicate_clean.json"
LOG="logs/adjudicate_clean.log"
LOCK="/tmp/gdn_adjudicate_clean.lock"
POLL_SEC="${ADJ_POLL_SEC:-120}"
CLEAN_RUNS="${ADJ_CLEAN_RUNS:-3}"

mkdir -p logs evidence/G5

compute_apps() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
}

log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

register_row() { # TAG -> appends a scheduling row to GPU-COORDINATION.md
  {
    echo ""
    echo "| $(date '+%m-%d %H:%M') | gdn-mlir G5-final ($1) | clean-window fwd adjudication via scripts/adjudicate_when_clean.sh: interleaved ours-vs-fla, 3 shapes x 5 pairs, ~15-20 min, <4GB | <4GB | running |"
  } >> "$COORD"
  log "coordination row registered ($1) in $COORD"
}

close_row() { # TAG status  (sed delimiters: @, replacement text contains /)
  sed -i "/$1/ s@| running |@| $2 (see $LOG) |@" "$COORD"
  log "coordination row ($1) -> $2"
}

TAG="adjclean$(date +%m%d%H%M%S)"

case "${1:-}" in
  --check)
    echo "=== adjudicate_when_clean --check $(date +%F_%T) ==="
    apps="$(compute_apps)"
    if [ -z "$apps" ]; then
      echo "compute-apps: (none) -- window CLEAN; the poller would run the adjudication now."
      exit 0
    else
      echo "compute-apps resident:"
      echo "$apps"
      echo "-- window BUSY; the poller would keep waiting (${POLL_SEC}s interval, ${CLEAN_RUNS} consecutive clean samples required)."
      exit 1
    fi
    ;;
esac

if mkdir "$LOCK" 2>/dev/null; then
  trap 'rmdir "$LOCK" 2>/dev/null' EXIT
else
  log "another adjudicate_when_clean instance holds $LOCK -- exiting"
  exit 2
fi

log "polling for a clean GPU window (interval ${POLL_SEC}s, need ${CLEAN_RUNS} consecutive clean samples)"
streak=0
while :; do
  apps="$(compute_apps)"
  if [ -z "$apps" ]; then
    streak=$((streak + 1))
    log "clean sample ${streak}/${CLEAN_RUNS}"
  else
    [ "$streak" -gt 0 ] && log "window busy again (resident: $(echo "$apps" | head -2 | tr '\n' ' ')) -- streak reset"
    streak=0
  fi
  [ "$streak" -ge "$CLEAN_RUNS" ] && break
  sleep "$POLL_SEC"
done

log "CLEAN WINDOW acquired -- registering coordination row and launching the one-shot adjudication"
register_row "$TAG"

set +e
"$PY" tools/gdn_final_adjudicate.py --out "$OUT" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

if [ "$RC" -eq 0 ] && [ -s "$OUT" ]; then
  close_row "$TAG" "done"
  log "SUCCESS: $OUT"
  "$PY" - "$OUT" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
for name, r in d.get("results", {}).items():
    print(f"  {name}: ours {r['ours_median_ms']:.4f} ms vs fla "
          f"{r['fla_median_ms']:.4f} ms -> {r['speedup_ratio_of_medians']:.3f}x "
          f"(window {'clean' if r.get('window_clean') else 'NOISY'})")
PYEOF
  exit 0
else
  close_row "$TAG" "failed"
  log "FAILED rc=$RC (row marked failed; rerun this script after fixing)"
  exit 3
fi
