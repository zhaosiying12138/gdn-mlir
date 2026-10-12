#!/usr/bin/env python3
"""G5-D: time the mono staged path e2e + per kernel."""
import os
import statistics
import sys

REPO = os.path.expanduser("~/codebase/gdn-mlir")
sys.path.insert(0, os.path.join(REPO, "tools"))
import gdn_full_rt as rt  # noqa: E402
import numpy as np  # noqa: E402
import cuda.bindings.driver as cu  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "bench_B1T2048H16"
d = sys.argv[2] if len(sys.argv) > 2 else \
    "/tmp/g5d_gen/bench_B1T2048H16_mono"
inp = dict(np.load(os.path.join(REPO, f"evidence/G5/refs/{tag}_inputs.npz")))
B, T, H, K = inp["q_bf16"].shape
V = inp["v_bf16"].shape[-1]
C = 64
Tp = ((T + C - 1) // C) * C
NC = Tp // C
rt._ck(cu.cuInit(0), "cuInit")
e, dev = cu.cuDeviceGet(0); rt._ck((e,), "dev")
e, ctx = cu.cuDevicePrimaryCtxRetain(dev); rt._ck((e,), "ctx")
rt._ck(cu.cuCtxSetCurrent(ctx), "ctx")
stream = cu.CUstream(0)
km = rt.FullKernel(d)
rt.stage_inputs(km, inp, (B, T, H, K, V, C, NC, Tp))
_, e0 = cu.cuEventCreate(0)
_, e1 = cu.cuEventCreate(0)


def t(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(iters):
        cu.cuEventRecord(e0, stream)
        fn()
        cu.cuEventRecord(e1, stream)
        rt._ck(cu.cuEventSynchronize(e1), "ev")
        _, ms = cu.cuEventElapsedTime(e0, e1)
        ts.append(ms)
    return statistics.median(ts) * 1000


def e2e():
    km.launch(stream, "gdn_fwd_f1")
    km.launch(stream, "gdn_fwd_f2")
    km.launch(stream, "gdn_fwd_f3")
    rt._ck(cu.cuStreamSynchronize(stream), "s")


print(f"e2e {t(e2e):.1f}us")
for kk in ("f1", "f2", "f3"):
    print(f"{kk} {t(lambda kk=kk: (km.launch(stream, 'gdn_fwd_' + kk), cu.cuStreamSynchronize(stream))):.1f}us")
