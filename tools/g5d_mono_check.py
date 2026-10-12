#!/usr/bin/env python3
"""Compare mono kernels' outputs against the standalone trio, per buffer."""
import os
import sys

REPO = os.path.expanduser("~/codebase/gdn-mlir")
sys.path.insert(0, os.path.join(REPO, "tools"))
import gdn_full_rt as rt  # noqa: E402
import numpy as np  # noqa: E402
import cuda.bindings.driver as cu  # noqa: E402

tag = "small_B1T512H4"
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

mono = rt.FullKernel("/tmp/g5d_mono64")
rt.stage_inputs(mono, inp, shape)
mono.launch(stream, "gdn_fwd_f1")
rt._ck(cu.cuStreamSynchronize(stream), "s1")
mono.launch(stream, "gdn_fwd_f2")
rt._ck(cu.cuStreamSynchronize(stream), "s2")
mono.launch(stream, "gdn_fwd_f3")
rt._ck(cu.cuStreamSynchronize(stream), "s3")

names = [("u", "bf16", B * H * NC * C * V),
         ("w", "bf16", B * H * NC * C * K),
         ("gc", "f32", B * H * NC * C),
         ("h_all", "bf16", B * H * NC * K * V),
         ("v_new", "bf16", B * H * NC * C * V),
         ("ht", "f32", B * H * K * V),
         ("o", "bf16", B * Tp * H * V)]
ref = dict(np.load(os.path.join(REPO, f"evidence/G5/refs/{tag}_ref.npz")))
ro = ref.get("ref__o_f64", ref.get("ref__o_f32"))
rh = ref.get("ref__ht_f64", ref.get("ref__ht_f32"))
for n, dt, el in names:
    a = mono.readout(n, dt, el)
    if n == "o":
        got = a.reshape(B, Tp, H, V)[:, :T]
        print(f"o: maxdiff={np.abs(got - ro).max():.2e}")
    elif n == "ht":
        got = a.reshape(B, H, K, V)
        print(f"ht: maxdiff={np.abs(got - rh).max():.2e}")
    else:
        print(f"{n}: nonzero={int((a != 0).sum())}/{el} "
              f"absmax={np.abs(a).max():.3f}")
