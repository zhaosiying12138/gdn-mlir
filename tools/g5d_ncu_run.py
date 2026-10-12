#!/usr/bin/env python3
"""G5-D: single-kernel launcher for ncu profiling (staged trio)."""
import os
import sys

REPO = os.path.expanduser("~/codebase/gdn-mlir")
sys.path.insert(0, os.path.join(REPO, "tools"))
import gdn_full_rt as rt  # noqa: E402
import numpy as np  # noqa: E402
import cuda.bindings.driver as cu  # noqa: E402

kk = sys.argv[1]            # f1|f2|f3
tag = sys.argv[2] if len(sys.argv) > 2 else "bench_B1T2048H16"
gen_root = sys.argv[3] if len(sys.argv) > 3 else "/tmp/g5d_gen"

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
kf = rt.FullKernel(os.path.join(gen_root, f"{tag}_{kk}"))
rt.stage_inputs(kf, inp, shape)
for _ in range(3):
    kf.launch(stream)
    rt._ck(cu.cuStreamSynchronize(stream), "s")
print("done", kk)
