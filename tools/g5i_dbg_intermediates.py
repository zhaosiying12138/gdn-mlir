#!/usr/bin/env python3
"""g5i_dbg_intermediates.py -- diff h_all/v_new/ht between two mono configs.

Builds/loads evidence/G5/gen_fwd/<tag>_mono{, suffix}, runs both, diffs
per-chunk h_all and v_new to localize segment-split bugs.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.expanduser("~/codebase/gdn-mlir")
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np
import cuda.bindings.driver as cu
import gdn_full_rt as rt

tag = sys.argv[1] if len(sys.argv) > 1 else "small_B1T512H4"
sfx = sys.argv[2] if len(sys.argv) > 2 else "_s4"
warps, bv = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 3 else ("4", "32")

inp = dict(np.load(os.path.join(REPO, f"evidence/G5/refs/{tag}_inputs.npz")))
B, T, H, K = inp["q_bf16"].shape
V = inp["v_bf16"].shape[-1]
C = 64
Tp = ((T + C - 1) // C) * C
NC = Tp // C
shape = (B, T, H, K, V, C, NC, Tp)

rt._ck(cu.cuInit(0), "cuInit")
e, dev = cu.cuDeviceGet(0); rt._ck((e,), "dev")
e, ctx = cu.cuDevicePrimaryCtxRetain(dev); rt._ck((e,), "ctx")
rt._ck(cu.cuCtxSetCurrent(ctx), "ctx")
stream = cu.CUstream(0)


def run(dirname):
    km = rt.FullKernel(os.path.join(REPO, "evidence/G5/gen_fwd", dirname))
    rt.stage_inputs(km, inp, shape)
    for kn in [k["name"] for k in km.spec.get("kernels", [])]:
        km.launch(stream, kn)
    rt._ck(cu.cuStreamSynchronize(stream), "sync")
    return km


ka = run(f"{tag}_mono")
kb = run(f"{tag}_mono{sfx}")

for nm, elems, dt in (("h_all", B * H * NC * K * V, "bf16"),
                      ("v_new", B * H * NC * C * V, "bf16"),
                      ("ht", B * H * K * V, "f32"),
                      ("u", B * H * NC * C * V, "bf16"),
                      ("w", B * H * NC * C * K, "bf16")):
    a = ka.readout(nm, dt, elems)
    b = kb.readout(nm, dt, elems)
    d = np.abs(a - b)
    print(f"{nm:6s} max {d.max():.3e}", end="")
    if nm in ("h_all", "v_new"):
        # per (bh, chunk) localization
        per = d.reshape(B * H, NC, -1).max(axis=2)
        bad = np.argwhere(per > 1e-3)
        print(f"  bad chunks (>1e-3): {len(bad)}",
              [tuple(x) for x in bad[:12]], end="")
    print()
