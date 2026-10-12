#!/usr/bin/env bash
# G0-6: FlashQLA(TileLang) 参考收割环境 + PTX 收割（纯 CPU）
# 独立 conda env gdn-tl，避免与 gdn env 依赖冲突。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
source "$HOME/miniforge3/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx gdn-tl; then
  conda create -y -n gdn-tl python=3.12
fi
conda activate gdn-tl
export PIP_DISABLE_PIP_VERSION_CHECK=1

# 版本裁定：tilelang 0.1.9（FlashQLA 官方 pin）的 tvm_ffi 与 CPython 3.12.15 不兼容
# （type.__dict__ 不可写，CPython 行为变更）→ 收割用 0.1.15（可 import，sm120 kernel
# 需要 harvest_qlaptx.py 里的 sm_120a monkeypatch）。PTX 覆盖矩阵以 0.1.15 为参考并如实标注。
TLVER=0.1.15
need_install=0
[ "$(python -c 'import tilelang;print(tilelang.__version__)' 2>/dev/null)" = "$TLVER" ] || need_install=1
if [ "$need_install" = 1 ]; then
  conda deactivate; conda env remove -y -n gdn-tl; conda create -y -n gdn-tl python=3.12; conda activate gdn-tl
  pip install -q torch --index-url https://download.pytorch.org/whl/cu130
  pip install -q "tilelang==$TLVER"
  python -c "import tilelang; print('tilelang', tilelang.__version__)"
fi

if [ ! -d "$HOME/codebase/FlashQLA/.git" ]; then
  git clone --depth 1 https://github.com/QwenLM/FlashQLA.git "$HOME/codebase/FlashQLA"
fi
pip install -q --no-deps -e "$HOME/codebase/FlashQLA" 2>/dev/null || echo "[env-qla] FlashQLA 源码模式可用（无需安装包体）"

python - <<'PY'
import torch, tilelang
print("[env-qla] ok torch", torch.__version__, "| tilelang", tilelang.__version__)
PY

# 收割（target 锁定 sm_120，get_kernel_source → nvcc -ptx → ptxas -v → SASS）
export TILELANG_DEFAULT_TARGET='{"kind":"cuda","arch":"sm_120a"}'
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
python "$ROOT/tools/harvest_qlaptx.py" 2>&1 | tee -a "$ROOT/logs/harvest_qla.log"
