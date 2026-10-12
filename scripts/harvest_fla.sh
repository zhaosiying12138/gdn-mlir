#!/usr/bin/env bash
# G0-5: fla 参考实现 PTX/SASS/ptxas 收割（纯 CPU）
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate gdn
mkdir -p logs evidence/G0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python tools/harvest_flaptx.py probe 2>&1 | tee logs/harvest_flaptx_probe.log
python tools/harvest_flaptx.py harvest 2>&1 | tee logs/harvest_flaptx.log
