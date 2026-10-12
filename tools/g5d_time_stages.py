#!/usr/bin/env python3
"""G5-D round 2: per-stage timing probe for the staged trio (CUDA events
around each kernel launch, median of N)."""
import json
import os
import statistics
import sys

REPO = os.path.expanduser("~/codebase/gdn-mlir")
sys.path.insert(0, os.path.join(REPO, "tools"))
import gdn_full_rt as rt  # noqa: E402


def run(tag, gen_root, warps, bv, iters=50, warmup=10):
    import cuda.bindings.driver as cu
    import numpy as np
    inp, ref = None, None
    d = os.path.join(REPO, "evidence/G5/refs")
    inp = dict(np.load(os.path.join(d, f"{tag}_inputs.npz")))
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

    kf = {}
    for kk in ("f1", "f2", "f3"):
        dd = rt.build(["--path", "staged", "--kernel", kk,
                       "--shape", f"{B},{T},{H},{K},{V}",
                       "--warps", str(warps), "--bv", str(bv)],
                      os.path.join(gen_root, f"{tag}_{kk}"))
        kf[kk] = rt.FullKernel(dd)
        rt.stage_inputs(kf[kk], inp, shape)

    def once():
        kf["f1"].launch(stream)
        rt._ck(cu.cuStreamSynchronize(stream), "s1")
        rt.staged_handoff(kf["f1"], kf["f2"],
                          (("u", "u"), ("w", "w"), ("gc", "gc_out")))
        kf["f2"].launch(stream)
        rt._ck(cu.cuStreamSynchronize(stream), "s2")
        rt.staged_handoff(kf["f2"], kf["f3"],
                          (("h_all", "h_all"), ("v_new", "v_new"),
                           ("gc", "gc")))
        kf["f3"].launch(stream)
        rt._ck(cu.cuStreamSynchronize(stream), "s3")

    def time_fn(fn):
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(iters):
            _, e0 = cu.cuEventCreate(0)
            _, e1 = cu.cuEventCreate(0)
            cu.cuEventRecord(e0, stream)
            fn()
            cu.cuEventRecord(e1, stream)
            rt._ck(cu.cuEventSynchronize(e1), "ev")
            _, ms = cu.cuEventElapsedTime(e0, e1)
            ts.append(ms)
        return statistics.median(ts)

    out = {"shape": tag, "warps": warps, "bv": bv, "gpu": rt.smi_snapshot()}
    out["f1_ms"] = round(time_fn(lambda: (kf["f1"].launch(stream),
                                          cu.cuStreamSynchronize(stream))), 6)
    out["f2_ms"] = round(time_fn(lambda: (kf["f2"].launch(stream),
                                          cu.cuStreamSynchronize(stream))), 6)
    out["f3_ms"] = round(time_fn(lambda: (kf["f3"].launch(stream),
                                          cu.cuStreamSynchronize(stream))), 6)
    out["e2e_ms"] = round(time_fn(once), 6)
    return out


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "bench_B1T2048H16"
    gen_root = sys.argv[2] if len(sys.argv) > 2 else "/tmp/g5d_gen"
    warps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    bv = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    r = run(tag, gen_root, warps, bv)
    print(json.dumps(r, indent=1))
