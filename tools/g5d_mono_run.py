#!/usr/bin/env python3
"""G5-D: mono module bisect runner."""
import os
import sys

REPO = os.path.expanduser("~/codebase/gdn-mlir")
sys.path.insert(0, os.path.join(REPO, "tools"))
import gdn_full_rt as rt  # noqa: E402
import numpy as np  # noqa: E402
import cuda.bindings.driver as cu  # noqa: E402

tag = "small_B1T512H4"
d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/g5d_gen/small_B1T512H4_mono"
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
km = rt.FullKernel(d)
print("entries:", list(km.fns))
rt.stage_inputs(km, inp, shape)
seq = sys.argv[2].split(",") if len(sys.argv) > 2 else ["f1", "f2", "f3"]
for s in seq:
    km.launch(stream, f"gdn_fwd_{s}")
    try:
        rt._ck(cu.cuStreamSynchronize(stream), "sync " + s)
        print("ok", s)
    except Exception as ex:  # noqa: BLE001
        print("FAULT", s, ex)
        break
# quick nonzero check
for n in ("u", "w", "h_all", "v_new", "o"):
    if n in km.globals:
        p = [q for q in km.spec["params"] if q["name"] == n][0]
        arr = km.readout(n, p["dtype"], min(p["elems"], 4096))
        print(n, "nonzero:", int((arr != 0).sum()), "/", len(arr))
