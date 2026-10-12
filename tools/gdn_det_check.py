#!/usr/bin/env python3
"""gdn_det_check.py -- G5-Q round10 run-to-run determinism harness.

Root-cause probe for the pre-existing dk nondeterminism (r9 finding:
final dk swings 1.719e-3 -> 4.719e-3 across identical verify runs).

Modes:
  loop   (default): run the whole chain N times IN-PROCESS, dump every
                    stage's OUT globals per run to per-run npz, diff.
  single RUN OUTNPZ: one fresh-process chain run, dump one npz (the
                    faithful "run-to-run" mode -- fresh CUDA context,
                    fresh module loads, fresh device allocations; drive
                    it from a bash loop).
  diff   OUTDIR:    compare all npz in OUTDIR against run 0; report the
                    FIRST chain stage whose outputs differ, per-global
                    diff counts and max-abs diff.

Device memory discipline: each stage's module is UNLOADED right after
its outputs are read back (co-job on the GPU; <4GB rule). The chain
semantics allow this: stage globals live per-module and the host
re-uploads downstream inputs (the chaining contract).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/codebase/gdn-mlir/tools"))
from gdn_bwd_verify import (STAGES, OUT, make_data, layout)  # noqa: E402


def run_chain_unload(gen_root, lay):
    """Mirror gdn_bwd_verify.run_chain but cuModuleUnload each stage."""
    import cuda.bindings.driver as cu

    man = json.load(open(os.path.join(gen_root, "bwd_manifest.json")))

    def ck(res, what):
        err = res[0] if isinstance(res, tuple) else res
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{what}: {err}")
        return res

    stream = cu.CUstream(0)
    ck(cu.cuInit(0), "init")
    e, dev = cu.cuDeviceGet(0)
    ck((e,), "dev")
    e, ctx = cu.cuDevicePrimaryCtxRetain(dev)
    ck((e,), "ctx")
    ck(cu.cuCtxSetCurrent(ctx), "setctx")

    host = {}
    for nm in ("q", "k", "v", "do"):
        host[nm] = lay[nm]
    host["beta"] = lay["beta"].astype(np.float32)
    host["g"] = lay["g_log2"].astype(np.float32)
    host["h0"] = lay["h0"]
    host["dht"] = lay["dht"]

    for st in man["stages"]:
        name = st["name"]
        d = os.path.join(gen_root, st["dir"])
        raw = open(os.path.join(d, "kernel_smemfix.cubin"), "rb").read()
        e, mod = cu.cuModuleLoadData(raw)
        ck((e,), f"load {name}")
        ptx = open(os.path.join(d, "kernel_smemfix.ptx")).read()
        kname = re.search(r"\.visible \.entry (\w+)\(", ptx).group(1)
        e, fn = cu.cuModuleGetFunction(mod, kname.encode())
        ck((e,), f"fn {name}")
        gx, gy, _gz = st["grid"]
        bx = st["block"][0]
        for sym, ty in st["globals"].items():
            if sym in OUT[name]:
                continue
            if sym in st.get("zero_globals", []):
                continue    # r21b scratch (spin flags): memset pre-launch
            if sym in st.get("scratch_globals", []):
                continue    # r21b scratch (pex): write-before-read
            if sym not in host:
                raise RuntimeError(f"{name}: input global @{sym} not staged")
            e, ptr, sz = cu.cuModuleGetGlobal(mod, sym.encode())
            ck((e,), f"global {name}/{sym}")
            arr = host[sym]
            ck(cu.cuMemcpyHtoD(ptr, np.frombuffer(arr.tobytes(), dtype=np.uint8),
                               arr.nbytes), f"h2d {name}/{sym}")
        # r21b scratch contract: stream-ordered zero of the spin flags
        for sym in st.get("zero_globals", []):
            e, ptr, sz = cu.cuModuleGetGlobal(mod, sym.encode())
            ck((e,), f"zero global {name}/{sym}")
            ck(cu.cuMemsetD8Async(ptr, 0, sz, stream),
               f"zero {name}/{sym}")
        cl = st.get("cluster")
        if cl:
            cfg = cu.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = gx, gy, 1
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = bx, 1, 1
            cfg.sharedMemBytes = 0
            cfg.hStream = stream
            cfg.numAttrs = 1
            a = cu.CUlaunchAttribute()
            a.id = (cu.CUlaunchAttributeID
                    .CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)
            a.value.clusterDim.x = cl[0]
            a.value.clusterDim.y = cl[1] if len(cl) > 1 else 1
            a.value.clusterDim.z = cl[2] if len(cl) > 2 else 1
            cfg.attrs = [a]
            ck(cu.cuLaunchKernelEx(cfg, fn, 0, 0), f"launchEx {name}")
        else:
            ck(cu.cuLaunchKernel(fn, gx, gy, 1, bx, 1, 1, 0, stream,
                                 None, 0), f"launch {name}")
        ck(cu.cuStreamSynchronize(stream), f"sync {name}")
        for sym in OUT[name]:
            if sym not in st["globals"]:
                continue    # r14: knob-off arms lack gsc_all
            ty = dict(st["globals"])[sym]
            rows, cols = (int(x) for x in ty.split("<")[1].split("x")[0:2])
            dt = "f32" if ty.endswith("f32>") else "bf16"
            esz = 4 if dt == "f32" else 2
            e, ptr, _sz = cu.cuModuleGetGlobal(mod, sym.encode())
            ck((e,), f"global out {name}/{sym}")
            buf = np.zeros(rows * cols * esz, dtype=np.uint8)
            ck(cu.cuMemcpyDtoH(buf, ptr, rows * cols * esz),
               f"d2h {name}/{sym}")
            if dt == "bf16":
                host[sym] = buf.view(np.uint16).reshape(rows, cols)
            else:
                host[sym] = buf.view(np.float32).reshape(rows, cols)
        ck(cu.cuModuleUnload(mod), f"unload {name}")
    ck(cu.cuDevicePrimaryCtxRelease(dev), "release")
    return {k: v for k, v in host.items() if not k.startswith("_")}


def dump_npz(host, path):
    arrs = {}
    for k, v in host.items():
        arrs[k] = v
    np.savez(path, **arrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", required=True)
    ap.add_argument("--shape", default="1,2048,16,128,128")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["loop", "single", "diff"],
                    default="loop")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--run", type=int, default=0, help="single mode run idx")
    ap.add_argument("--out", default=None, help="npz (single) or dir")
    args = ap.parse_args()

    B, T, H, K, V = (int(x) for x in args.shape.split(","))

    if args.mode == "diff":
        outdir = args.out
        files = sorted([f for f in os.listdir(outdir) if f.endswith(".npz")],
                       key=lambda f: int(re.search(r"run(\d+)", f).group(1)))
        base = dict(np.load(os.path.join(outdir, files[0])))
        order = STAGES
        first_bad = None
        print(f"comparing {len(files)} runs vs {files[0]}")
        for st in order:
            for sym in OUT[st]:
                if sym not in base:
                    continue
                runs_bad = []
                for f in files[1:]:
                    cur = np.load(os.path.join(outdir, f))
                    if sym not in cur.files:
                        continue
                    a, b = base[sym], cur[sym]
                    if a.shape != b.shape or a.dtype != b.dtype:
                        runs_bad.append((f, "shape/dtype"))
                        continue
                    neq = a != b
                    if neq.any():
                        if a.dtype == np.uint16:
                            fa = (a.astype(np.uint32) << 16).view(np.float32)
                            fb = (b.astype(np.uint32) << 16).view(np.float32)
                        else:
                            fa, fb = a.astype(np.float64), b.astype(np.float64)
                        m = neq & ~(  # ignore sign-of-zero / nan-pair noise
                            (fa == fb))
                        dmax = float(np.abs(np.where(neq, fa - fb, 0.0)).max())
                        runs_bad.append((f, f"{int(neq.sum())}/{neq.size}"
                                         f" elems, max|d|={dmax:.3e}"))
                tag = "DIVERGES" if runs_bad else "stable"
                print(f"  [{st:8s}] {sym:12s} {tag}")
                for f, why in runs_bad:
                    print(f"      {f}: {why}")
                if runs_bad and first_bad is None:
                    first_bad = (st, sym)
        if first_bad:
            print(f"\nFIRST diverging output: stage {first_bad[0]} "
                  f"global {first_bad[1]}")
        else:
            print("\nALL stage outputs bit-identical across runs")
        return 0

    data = make_data(B, T, H, K, V, args.seed)
    lay = layout(data, B, T, H, K, V, args.chunk)

    if args.mode == "single":
        host = run_chain_unload(args.gen_root, lay)
        out = args.out or f"det_run{args.run}.npz"
        dump_npz(host, out)
        print(f"wrote {out}")
        return 0

    # loop mode
    outdir = args.out or "."
    os.makedirs(outdir, exist_ok=True)
    for i in range(args.n):
        host = run_chain_unload(args.gen_root, lay)
        dump_npz(host, os.path.join(outdir, f"run{i}.npz"))
        print(f"run {i} done", flush=True)
    print("loop complete; now diff (in-process):")
    # reuse diff logic by shelling out
    import subprocess
    sys.exit(subprocess.call(
        [sys.executable, __file__, "--mode", "diff", "--gen-root",
         args.gen_root, "--shape", args.shape, "--out", outdir]))


if __name__ == "__main__":
    sys.exit(main())
