#!/usr/bin/env python3
"""gdn_fwd_det.py -- G5-Q round10: forward-chain run-to-run determinism.

Runs the staged mono forward chain (F1->F2[a/c]->F3, the shipped gdn_full
path) OR the fused kernel N times -- fresh process per run (fresh module
loads, mirroring the bwd harness), dumping o/ht (+ u/w/h_all counts) per
run; diff mode reports flips.  Usage:
  run:   python tools/gdn_fwd_det.py --tag bench_B1T2048H16 --out run0.npz
  diff:  python tools/gdn_fwd_det.py --diff OUTDIR
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.expanduser("~/codebase/gdn-mlir")
sys.path.insert(0, os.path.join(REPO, "tools"))
import gdn_full_rt as rt  # noqa: E402


def one_run(tag, path, warps, bv, f2s):
    import cuda.bindings.driver as cu
    inp, ref = load_case(tag)
    B, T, H, K = inp["q_bf16"].shape
    V = inp["v_bf16"].shape[-1]
    C = 64
    Tp = ((T + C - 1) // C) * C
    NC = Tp // C
    shape = (B, T, H, K, V, C, NC, Tp)

    rt._ck(cu.cuInit(0), "init")
    e, dev = cu.cuDeviceGet(0); rt._ck((e,), "dev")
    e, ctx = cu.cuDevicePrimaryCtxRetain(dev); rt._ck((e,), "ctx")
    rt._ck(cu.cuCtxSetCurrent(ctx), "ctx")
    stream = cu.CUstream(0)

    gen_args = ["--path", path, "--shape", f"{B},{T},{H},{K},{V}",
                "--warps", str(warps), "--bv", str(bv), "--bv3", str(V)]
    if f2s > 1:
        gen_args += ["--f2-segments", str(f2s)]
    suffix = "" if f2s == 1 else f"_s{f2s}"
    d = rt.build(gen_args, os.path.join(
        REPO, "evidence/G5/gen_fwd", f"{tag}_{path}{suffix}"))
    km = rt.FullKernel(d)
    rt.stage_inputs(km, inp, shape)
    if path == "mono":
        for kn in [k["name"] for k in km.spec.get("kernels", [])]:
            km.launch(stream, kn)
    else:
        km.launch(stream)
    rt._ck(cu.cuStreamSynchronize(stream), "sync")
    out = {
        "o": km.readout("o", "bf16", B * Tp * H * V),
        "ht": km.readout("ht", "f32", B * H * K * V),
    }
    rt._ck(cu.cuModuleUnload(km.mod), "unload")
    rt._ck(cu.cuDevicePrimaryCtxRelease(dev), "release")
    return out


def load_case(tag):
    dd = os.path.join(REPO, "evidence/G5/refs")
    inp = dict(np.load(os.path.join(dd, f"{tag}_inputs.npz")))
    ref = dict(np.load(os.path.join(dd, f"{tag}_ref.npz")))
    return inp, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bench_B1T2048H16")
    ap.add_argument("--path", default="mono", choices=["mono", "fused"])
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--bv", type=int, default=64)
    ap.add_argument("--f2-segments", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--diff", default=None)
    args = ap.parse_args()

    if args.diff:
        files = sorted(os.listdir(args.diff))
        base = dict(np.load(os.path.join(args.diff, files[0])))
        bad = 0
        for f in files[1:]:
            cur = np.load(os.path.join(args.diff, f))
            for k in ("o", "ht"):
                neq = base[k] != cur[k]
                if neq.any():
                    bad += 1
                    print(f"{f} {k}: {int(neq.sum())}/{neq.size} elems")
        print("FWD-DETERMINISTIC" if bad == 0 else f"FWD-DIVERGES ({bad})")
        return 0 if bad == 0 else 1

    out = one_run(args.tag, args.path, args.warps, args.bv,
                  args.f2_segments)
    np.savez(args.out, o=out["o"], ht=out["ht"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
