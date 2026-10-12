#!/usr/bin/env python3
"""gdn_det_discriminator.py -- G5-Q round10: separate kernel-bug races
from memory-subsystem corruption in the bpre nondeterminism.

Arms (each N iterations, fresh module load/unload per iteration, same
as the chain harness):
  h2d     : stage M_all/do H2D, read them back D2H, compare to host.
            NO kernel launch. Any diff = module/memory path corruption.
  launch  : stage H2D, READ BACK M_all/do right after launch+sync
            (input corruption check WITH the kernel running).
  bpre    : the reproducer itself (B_all out compare).
  idle    : plain device buffer alloc + H2D + dwell + D2H, no modules.
"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--arm", required=True,
                    choices=["h2d", "launch", "bpre", "idle"])
    ap.add_argument("-n", type=int, default=150)
    args = ap.parse_args()

    src = np.load(args.inputs)
    M_all = np.ascontiguousarray(src["M_all"])
    do = np.ascontiguousarray(src["do"])
    M, KC = M_all.shape

    stream = cu.CUstream(0)
    ck(cu.cuInit(0), "init")
    e, dev = cu.cuDeviceGet(0); ck((e,), "dev")
    e, ctx = cu.cuDevicePrimaryCtxRetain(dev); ck((e,), "ctx")
    ck(cu.cuCtxSetCurrent(ctx), "setctx")

    if args.arm == "idle":
        buf = np.random.default_rng(7).standard_normal(
            256 * 1024 * 1024).astype(np.float32)  # 1 GiB
        raw = buf.tobytes()
        e, dbuf = cu.cuMemAlloc(1024 * 1024 * 1024); ck((e,), "alloc")
        rb = np.zeros_like(buf)
        for i in range(args.n):
            ck(cu.cuMemcpyHtoD(dbuf, np.frombuffer(raw, dtype=np.uint8),
                               len(raw)), "h2d")
            import time
            time.sleep(20)
            ck(cu.cuMemcpyDtoH(rb.view(np.uint8), dbuf, len(raw)), "d2h")
            neq = rb != buf
            print(f"iter {i}: {neq.sum()} flipped elems")
        return 0

    raw = open(os.path.join(args.stage_dir, "kernel_smemfix.cubin"),
               "rb").read()
    ptx = open(os.path.join(args.stage_dir, "kernel_smemfix.ptx")).read()
    kname = re.search(r"\.visible \.entry (\w+)\(", ptx).group(1)

    ndiff = 0
    for i in range(args.n):
        e, mod = cu.cuModuleLoadData(raw); ck((e,), "load")
        e, fn = cu.cuModuleGetFunction(mod, kname.encode()); ck((e,), "fn")
        ptrs = {}
        for sym, arr in (("M_all", M_all), ("do", do)):
            e, ptr, sz = cu.cuModuleGetGlobal(mod, sym.encode())
            ck((e,), sym)
            ck(cu.cuMemcpyHtoD(ptr, np.frombuffer(arr.tobytes(),
                                                  dtype=np.uint8),
                               arr.nbytes), f"h2d {sym}")
            ptrs[sym] = (ptr, arr)
        if args.arm == "launch":
            ck(cu.cuLaunchKernel(fn, 512, 1, 1, 256, 1, 1, 0, stream,
                                 None, 0), "launch")
            ck(cu.cuStreamSynchronize(stream), "sync")
        if args.arm == "bpre":
            ck(cu.cuLaunchKernel(fn, 512, 1, 1, 256, 1, 1, 0, stream,
                                 None, 0), "launch")
            ck(cu.cuStreamSynchronize(stream), "sync")
            e, bptr, _ = cu.cuModuleGetGlobal(mod, b"B_all"); ck((e,), "B")
            B = np.zeros(M * KC * 4, dtype=np.uint8)
            ck(cu.cuMemcpyDtoH(B, bptr, B.nbytes), "d2h B")
        # read the INPUTS back and compare
        for sym, (ptr, arr) in ptrs.items():
            rb = np.zeros(arr.nbytes, dtype=np.uint8)
            ck(cu.cuMemcpyDtoH(rb, ptr, arr.nbytes), f"d2h {sym}")
            if not np.array_equal(rb, np.frombuffer(arr.tobytes(),
                                                    dtype=np.uint8)):
                neq = rb != np.frombuffer(arr.tobytes(), dtype=np.uint8)
                print(f"iter {i}: {sym} INPUT flipped: {neq.sum()} elems")
                ndiff += 1
        ck(cu.cuModuleUnload(mod), "unload")
    print(f"arm {args.arm}: {ndiff}/{args.n} iterations had input "
          f"corruption")
    return 0


if __name__ == "__main__":
    sys.exit(main())
