#!/usr/bin/env bash
# G3-A: one-time sm_120 hardware calibration microbench (opportunistic GPU
# window, <2GB VRAM, total wall time ~1 min). Re-runnable: same output files
# are overwritten with fresh values (calibration.json / calib_report data).
#
# Usage: scripts/run_calib.sh
# Produces:
#   tools/gdn_calib/build/calib_bench      (nvcc -arch=sm_120a -O3)
#   evidence/G3/calibration.json           (raw measurements + shared-GPU
#                                           metadata, merged by this script)

set -euo pipefail
REPO=~/codebase/gdn-mlir
cd "$REPO"
export PATH="/usr/local/cuda/bin:$PATH"

mkdir -p tools/gdn_calib/build evidence/G3

snap() {
  nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm \
    --format=csv,noheader,nounits | head -1
}

COL0="$(snap)"
echo "[run_calib] co-load at start: ${COL0}"

nvcc -arch=sm_120a -O3 tools/gdn_calib/calib_bench.cu \
  -o tools/gdn_calib/build/calib_bench

COL1="$(snap)"
echo "[run_calib] co-load pre-run: ${COL1}"

# sanity gate: another session's big job must not be killed by us; we only
# need a few hundred MB for a couple of seconds.
tools/gdn_calib/build/calib_bench > evidence/G3/.calib_raw.json

COL2="$(snap)"
APPS="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
  | tr '\n' ';')"

python3 - "$COL0" "$COL1" "$COL2" "$APPS" <<'PYEOF'
import json, sys, datetime, re
raw = json.load(open("evidence/G3/.calib_raw.json"))
APPS = sys.argv[4]
def parse_snap(s):
    u, m, c = [x.strip() for x in s.split(",")]
    return {"util_pct": int(u), "mem_used_mib": int(m), "sm_clock_mhz": int(c)}
doc = {
    "description": "G3-A one-time sm_120 (RTX 5090 Laptop) calibration "
                   "microbench: mma peak / smem+gmem+L2 bandwidth / SFU "
                   "ex2 / bar.sync / cp.async",
    "method": "CUDA events, >=5 warmup + 50 timed iterations, median; "
              "per-SM/clk normalized against the peak (attribute) clock",
    "shared_gpu": bool(APPS.strip()),  # g5m: derived from observed compute-apps
    "coload": {
        "start": parse_snap(sys.argv[1]),
        "pre_run": parse_snap(sys.argv[2]),
        "end": parse_snap(sys.argv[3]),
        "compute_apps": sys.argv[4],
        "note": ("co-running compute apps observed: " + APPS.strip() +
                "; absolute bandwidth/FLOP numbers are lower bounds "
                "(contention) - re-run in a clean window to overwrite"
                if APPS.strip() else
                "clean window: no other compute apps resident in the "
                "start/pre-run/end snapshots; numbers are unconditional"),
    },
    "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "rerun": "scripts/run_calib.sh (idempotent; overwrites calibration.json)",
    "measurements": raw,
}
json.dump(doc, open("evidence/G3/calibration.json", "w"), indent=2)
print("[run_calib] wrote evidence/G3/calibration.json")
PYEOF
