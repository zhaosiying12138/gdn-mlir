#!/usr/bin/env python3
"""gdn_bpre_loop.py -- isolate the bpre nondeterminism: launch the bpre
cubin in a tight loop against captured M_all/do inputs, reading back
B_all every iteration. Reports every iteration whose B_all differs from
iteration 0 (element count, max abs diff, first few indices).

--fresh: fresh module load + H2D staging + unload per iteration (emulate
the chain harness module cycle; fresh device global allocations and
fresh smem initial content per launch).
--save PATH: save iter0 B_all (the deterministic reference) to npz."""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

import cuda.bindings.driver as cu


def ck(res, what):
    err = res[0] if isinstance(res, tuple) else res
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what}: {err}")
    return res


def stage_globals(mod, M_all, do, stream):
    for sym, arr in (("M_all", M_all), ("do", do)):
        e, ptr, sz = cu.cuModuleGetGlobal(mod, sym.encode()); ck((e,), sym)
        ck(cu.cuMemcpyHtoD(ptr, np.frombuffer(arr.tobytes(), dtype=np.uint8),
                           arr.nbytes), f"h2d {sym}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--inputs", required=True,
                    help="npz with M_all and do (a full-chain det npz)")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--save", default=None)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    src = np.load(args.inputs)
    M_all = np.ascontiguousarray(src["M_all"])
    do = np.ascontiguousarray(src["do"])
    M, KC = M_all.shape
    print(f"M_all {M_all.shape} {M_all.dtype}, do {do.shape} {do.dtype}")

    stream = cu.CUstream(0)
    ck(cu.cuInit(0), "init")
    e, dev = cu.cuDeviceGet(0); ck((e,), "dev")
    e, ctx = cu.cuDevicePrimaryCtxRetain(dev); ck((e,), "ctx")
    ck(cu.cuCtxSetCurrent(ctx), "setctx")

    raw = open(os.path.join(args.stage_dir, "kernel_smemfix.cubin"), "rb").read()
    ptx = open(os.path.join(args.stage_dir, "kernel_smemfix.ptx")).read()
    kname = re.search(r"\.visible \.entry (\w+)\(", ptx).group(1)

    mod0 = None
    if not args.fresh:
        e, mod0 = cu.cuModuleLoadData(raw); ck((e,), "load")
        e, fn0 = cu.cuModuleGetFunction(mod0, kname.encode()); ck((e,), "fn")
        stage_globals(mod0, M_all, do, stream)
        e, bptr0, _ = cu.cuModuleGetGlobal(mod0, b"B_all"); ck((e,), "B_all")

    B = np.zeros(M * KC * 4, dtype=np.uint8)
    Bref = None
    ndiff = 0
    for i in range(args.n):
        if args.fresh:
            e, mod_i = cu.cuModuleLoadData(raw); ck((e,), "load")
            e, fn_i = cu.cuModuleGetFunction(mod_i, kname.encode())
            ck((e,), "fn")
            stage_globals(mod_i, M_all, do, stream)
            e, bptr_i, _ = cu.cuModuleGetGlobal(mod_i, b"B_all")
            ck((e,), "B_all")
            ck(cu.cuLaunchKernel(fn_i, args.grid, 1, 1, args.block, 1, 1, 0,
                                 stream, None, 0), "launch")
            ck(cu.cuStreamSynchronize(stream), "sync")
            ck(cu.cuMemcpyDtoH(B, bptr_i, B.nbytes), "d2h")
            ck(cu.cuModuleUnload(mod_i), "unload")
        else:
            ck(cu.cuLaunchKernel(fn0, args.grid, 1, 1, args.block, 1, 1, 0,
                                 stream, None, 0), "launch")
            ck(cu.cuStreamSynchronize(stream), "sync")
            ck(cu.cuMemcpyDtoH(B, bptr0, B.nbytes), "d2h")
        Bv = B.view(np.float32)
        if Bref is None:
            Bref = Bv.copy()
            if args.save:
                np.savez(args.save, B_all=Bv.reshape(M, KC),
                         M_all=M_all, do=do)
                print(f"saved reference to {args.save}")
        else:
            neq = Bv != Bref
            if neq.any():
                ndiff += 1
                idx = np.argwhere(neq).ravel()
                dmax = np.abs(Bv[neq] - Bref[neq]).max()
                mrows = sorted(set((idx // KC).tolist()))
                print(f"iter {i}: {neq.sum()} elems differ, "
                      f"max|d|={dmax:.3e}, rows(m)={mrows[:8]}")
                for t in idx[:6]:
                    print(f"    flat {t} -> m={t//KC} col={t%KC}"
                          f" ref={Bref[t]:.9e} now={Bv[t]:.9e}")
    print(f"done: {ndiff}/{args.n} iterations diverged from iter0")
    if mod0 is not None:
        ck(cu.cuModuleUnload(mod0), "unload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
