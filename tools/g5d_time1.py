#!/usr/bin/env python3
"""G5-D: time ONE staged kernel (already built dir) with proper inputs."""
import os
import sys

REPO = os.path.expanduser("~/codebase/gdn-mlir")
sys.path.insert(0, os.path.join(REPO, "tools"))
import gdn_full_rt as rt  # noqa: E402
import numpy as np  # noqa: E402
import cuda.bindings.driver as cu  # noqa: E402

d = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "bench_B1T2048H16"
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
kf = rt.FullKernel(d)
rt.stage_inputs(kf, inp, shape)
_, e0 = cu.cuEventCreate(0)
_, e1 = cu.cuEventCreate(0)
for _ in range(10):
    kf.launch(stream)
rt._ck(cu.cuStreamSynchronize(stream), "w")
ts = []
for _ in range(50):
    cu.cuEventRecord(e0, stream)
    kf.launch(stream)
    cu.cuEventRecord(e1, stream)
    rt._ck(cu.cuEventSynchronize(e1), "ev")
    _, ms = cu.cuEventElapsedTime(e0, e1)
    ts.append(ms)
import statistics
print(f"{d}: median={statistics.median(ts)*1000:.1f}us min={min(ts)*1000:.1f}us")
