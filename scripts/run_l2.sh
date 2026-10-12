#!/usr/bin/env bash
# G2-A compile one knob config to PTX/SASS for sm_120a. CPU-only (no GPU).
#
# Usage: scripts/run_l2.sh <config-dir>
#   <config-dir> must contain payload.mlir, transform.mlir, config.json
#   (as produced by tools/gdn_l2_gen.py --gen NAME --out <config-dir>).
#
# Produces in <config-dir>:
#   kernel.ptx      PTX text (extraction method: gpu-lower-to-nvvm-pipeline
#                   cubin-format=assembly -> gpu.binary assembly attr)
#   kernel.cubin    ptxas -arch=sm_120a object
#   ptxas_v.txt     ptxas -v resource report
#   sass.txt        nvdisasm -c disassembly
#   meta.json       config + ptxas metrics + PTX feature histogram
set -euo pipefail

REPO=~/codebase/gdn-mlir
MLIR_OPT=~/codebase/llvm-mlir/build/bin/mlir-opt
GDN_OPT=~/codebase/gdn-mlir/build-dialect/bin/gdn-opt
TRITON_BIN=~/miniforge3/envs/gdn/lib/python3.12/site-packages/triton/backends/nvidia/bin
export PATH="/usr/local/cuda/bin:$PATH"

DIR="${1:?usage: run_l2.sh <config-dir>}"
cd "$DIR"

# pick tools: prefer ptxas-blackwell (PTX ISA 9.3-capable), then CUDA 13.3
PTXAS="$TRITON_BIN/ptxas-blackwell"
[ -x "$PTXAS" ] || PTXAS="$TRITON_BIN/ptxas"
[ -x "$PTXAS" ] || PTXAS="$(command -v ptxas)"
NVDISASM="$TRITON_BIN/nvdisasm"
[ -x "$NVDISASM" ] || NVDISASM="$(command -v nvdisasm)"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[run_l2] $(basename "$PWD")"
# 1. merge payload + transform and run the transform interpreter.
# G2-C: payloads with gdn.* hatch ops are first lowered to in-tree
# nvgpu/nvvm ops with our dialect tool (gdn-opt parses the payload but not
# the transform script, so lower BEFORE merging).
if grep -qE 'gdn\.(setmaxnreg|elect|mbarrier|tma|bulk|mma)' payload.mlir; then
  echo "[run_l2] gdn hatch ops -> in-tree nvgpu/nvvm (gdn-opt --gdn-lower-hatch)"
  "$GDN_OPT" payload.mlir --gdn-lower-hatch -o "$TMP/payload.gdn.mlir"
else
  cp payload.mlir "$TMP/payload.gdn.mlir"
fi

# G2-C note: an identity transform script (direct-ops payloads: gdn hatch /
# ldmatrix configs) skips -transform-interpreter entirely — the interpreter
# wraps a bare top-level func into `module {}`, and that extra nesting level
# breaks the gpu-lower-to-nvvm-pipeline on these payloads ("conversion of
# memref memory space #gpu.address_space<workgroup> ... failed").
if grep -qE 'transform\.(structured|nvgpu|memref)' transform.mlir; then
  cat "$TMP/payload.gdn.mlir" transform.mlir > "$TMP/combined.mlir"
  "$MLIR_OPT" "$TMP/combined.mlir" -transform-interpreter -o "$TMP/sched.mlir"
else
  echo "[run_l2] identity transform script -> skip -transform-interpreter"
  cp "$TMP/payload.gdn.mlir" "$TMP/sched.mlir"
fi

# 2. linalg fallback (mma=off path keeps linalg.matmul after tiling)
if grep -q "linalg.matmul" "$TMP/sched.mlir"; then
  echo "[run_l2] linalg survivor -> -convert-linalg-to-loops"
  "$MLIR_OPT" "$TMP/sched.mlir" -convert-linalg-to-loops -o "$TMP/sched2.mlir"
else
  cp "$TMP/sched.mlir" "$TMP/sched2.mlir"
fi

# 3. NVVM lowering + PTX emission (gpu.binary assembly attr)
"$MLIR_OPT" "$TMP/sched2.mlir" -test-transform-dialect-erase-schedule \
  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a cubin-features=+ptx87 cubin-format=assembly" \
  -o "$TMP/asmobj.mlir"

# 4. extract PTX text
python3 "$REPO/tools/gdn_extract_ptx.py" "$TMP/asmobj.mlir" kernel.ptx
# G3-A: keep the post-pipeline MLIR (host gpu.launch_func operand list =
# kernel parameter values; parsed by tools/gdn_smem_fix.py)
cp "$TMP/asmobj.mlir" asmobj.mlir

# 5. ptxas resource report + cubin
"$PTXAS" -v -arch=sm_120a kernel.ptx -o kernel.cubin > ptxas_v.txt 2>&1

# 6. SASS
"$NVDISASM" -c kernel.cubin > sass.txt

# 7. meta.json (config + metrics + PTX feature histogram)
python3 - "$REPO" <<'PYEOF'
import json, re, sys, os, subprocess
repo = sys.argv[1]
sys.path.insert(0, os.path.join(repo, "tools"))
from harvest_flaptx import ptx_features, parse_ptxas_v  # noqa: E402

cfg = json.load(open("config.json"))
ptx = open("kernel.ptx").read()
ptxas_txt = open("ptxas_v.txt").read()
sass = open("sass.txt").read()

meta = dict(cfg)
meta["metrics"] = parse_ptxas_v(ptxas_txt)
# dynamic smem: report any .extern .shared declaration size if present
m = re.search(r"\.extern \.shared \.align \d+ \.b8 \w+\[(\d+)\]", ptx)
meta["metrics"]["smem_dynamic"] = int(m.group(1)) if m else 0
feat = ptx_features(ptx)
# harvest FAMS pattern (\bexp2?\.approx) misses the actual PTX `ex2.approx`
feat["ex2.approx"] = len(re.findall(r"\bex2\.approx[\w.]*", ptx))
meta["ptx_features"] = feat
sass_ops = {}
for op in ("HMMA", "LDGSTS", "LDS", "STS", "LDL", "STL", "BAR.SYNC",
           "LDSM", "EX2"):
    sass_ops[op] = len(re.findall(r"\b" + op.replace(".", r"\.") + r"\b", sass))
meta["sass_ops"] = sass_ops
meta["ptx_size_bytes"] = len(ptx.encode())
meta["sass_size_bytes"] = len(sass.encode())
meta["sass_line_count"] = sass.count("\n")
# exact mma signatures for the T1 checklist
meta["mma_signatures"] = sorted(set(re.findall(r"mma\.sync\.aligned[.\w]+", ptx)))
meta["ex2_variants"] = sorted(set(re.findall(r"ex2\.approx[.\w]*", ptx)))
meta["ldmatrix"] = len(re.findall(r"\bldmatrix\b", ptx))
meta["ptx_version"] = (re.search(r"\.version (\S+)", ptx) or [None, "?"])[1] \
    if re.search(r"\.version (\S+)", ptx) else "?"
meta["tools"] = {
    "ptxas": subprocess.run([os.environ.get("PTXAS_BIN", "ptxas"), "--version"],
                            capture_output=True, text=True).stdout.strip().splitlines()[-1],
}
meta["extraction_method"] = ("gpu-lower-to-nvvm-pipeline cubin-format=assembly "
                             "-> gpu.binary #gpu.object assembly attr -> python unescape")
json.dump(meta, open("meta.json", "w"), indent=2)
print("[run_l2] meta.json: regs=%s smem=%s mma=%d cp.async=%d" % (
    meta["metrics"].get("registers"), meta["metrics"].get("smem_static"),
    feat.get("mma.sync", 0), feat.get("cp.async", 0)))
PYEOF

echo "[run_l2] done: kernel.ptx kernel.cubin ptxas_v.txt sass.txt meta.json"
