#!/usr/bin/env python3
"""G5-H micro-probe: does MANUAL nvgpu.device_async_copy into a NONZERO
column offset of a [64,64] bf16 smem tile deliver the right bytes (and
survive wait_group 0 + barrier) on this toolchain?  Stage cols 32..64 from
a global row, read back with the same +32 offset, store to an output global.
"""
import json, os, sys

sys.path.insert(0, os.path.expanduser("~/codebase/gdn-mlir/tools"))
import gdn_bwd_gen as GEN

OUT = os.path.expanduser("~/codebase/gdn-mlir/evidence/G5/scratch/r4_pfprobe")

GEN.THREADS = 256  # astage distributes by module THREADS; keep in sync

k = GEN.Kernel("pfp", gx=1, threads=256)
k.global_("A", (1, 8192), "bf16")
k.global_("B", (1, 8192), "bf16")
e = GEN.E()
G = GEN.emit_header(e, k)
tile = k.alloc(e, "tile", 64, 64, "bf16")
SRC_OFF = int(os.environ.get("PFP_SRC_OFF", "32"))
DST_OFF = int(os.environ.get("PFP_DST_OFF", "32"))
toks = []
GEN.astage_tile(e, G["A"], "memref<1x8192xbf16>", "%c0", 128, SRC_OFF,
                tile, DST_OFF, 64, 32, "bf16", toks)
g = GEN.agroup(e, toks)
GEN.await_all(e, g)
e.l("gpu.barrier")
# read back with the PROVEN emap + store_rows helpers (no hand-written loop)
tile2 = k.alloc(e, "tile2", 64, 32, "bf16")
GEN.emap(e, tile2, None, None, "%c0", 0, lambda e, v, i, j: v,
         smem_src=tile, off=(0, DST_OFF))
GEN.store_rows(e, tile2, G["B"], "memref<1x8192xbf16>", "%c0",
               col0=DST_OFF, cm=128)
GEN.emit_footer(e)
txt = GEN.module_text(k, e)
os.makedirs(f"{OUT}/pfp", exist_ok=True)
open(f"{OUT}/pfp/payload.mlir", "w").write(txt)
open(f"{OUT}/pfp/transform.mlir", "w").write(GEN.IDENTITY_TRANSFORM)
json.dump({"name": "pfp", "grid": [1, 1], "threads": 256,
           "smem_bytes": k.smem_bytes()},
          open(f"{OUT}/pfp/config.json", "w"), indent=2)
print(f"[probe_pf] src_off={SRC_OFF} dst_off={DST_OFF}: wrote payload")
