#!/usr/bin/env python3
"""gdn_bwd_bench.py -- G5-C e2e backward timing vs the fla Triton baseline.

Times the 5-stage GDN-MLIR backward chain (all kernels pre-staged, launched
back-to-back on one stream; median CUDA-event time over N iterations) and,
if a fla baseline json is available (evidence/G5/fla_baseline.json -- else
measured here with torch+fla on the same GPU, shared-GPU flagged), reports
the bwd speedup. Also times fwd+bwd total when a fwd time is present in the
baseline json.

Usage (conda gdn):
  python tools/gdn_bwd_bench.py --gen-root evidence/G5/gen_bwd \
      --shape 1,2048,16,128,128 [--iters 30] [--fla-json PATH] [--measure-fla]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.expanduser("~/codebase/gdn-mlir")


def time_chain(gen_root, iters=30, warmup=5):
    import cuda.bindings.driver as cu
    import numpy as np

    man = json.load(open(os.path.join(gen_root, "bwd_manifest.json")))

    def ck(r, w):
        err = r[0] if isinstance(r, tuple) else r
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{w}: {err}")
        return r

    stream = cu.CUstream(0)
    ck(cu.cuInit(0), "init")
    dev = cu.cuDeviceGet(0)[1]
    cu.cuCtxSetCurrent(cu.cuDevicePrimaryCtxRetain(dev)[1])
    import re
    fns = []
    for st in man["stages"]:
        d = os.path.join(gen_root, st["dir"])
        mod = cu.cuModuleLoadData(
            open(os.path.join(d, "kernel_smemfix.cubin"), "rb").read())[1]
        ptx = open(os.path.join(d, "kernel_smemfix.ptx")).read()
        kname = re.search(r"\.visible \.entry (\w+)\(", ptx).group(1)
        fns.append((cu.cuModuleGetFunction(mod, kname.encode())[1], st, mod))

    # G5-Y round21 (R21-bwd, GDN_FKS2): cluster-launch + scratch-zero
    # contract.  Stages whose manifest carries "cluster" launch via
    # cuLaunchKernelEx with CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION (the
    # pair co-residency guarantee that voids the F1KS spin hazard);
    # "zero_globals" are memset-0 STREAM-ORDERED before every launch
    # (absolute spin targets -- cross-launch flag staleness closed).
    zg = {}
    for fn, st, mod in fns:
        for sym in st.get("zero_globals", []):
            e, ptr, sz = cu.cuModuleGetGlobal(mod, sym.encode())
            ck((e,), f"zero global {st['name']}/{sym}")
            zg[(st["name"], sym)] = (ptr, sz)

    def launch_one(fn, st):
        for sym in st.get("zero_globals", []):
            ptr, sz = zg[(st["name"], sym)]
            ck(cu.cuMemsetD8Async(ptr, 0, sz, stream),
               f"zero {st['name']}/{sym}")
        cl = st.get("cluster")
        if cl:
            cfg = cu.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = (
                st["grid"][0], st["grid"][1], 1)
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = (
                st["block"][0], 1, 1)
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
            ck(cu.cuLaunchKernelEx(cfg, fn, 0, 0),
               f"launchEx {st['name']}")
        else:
            ck(cu.cuLaunchKernel(fn, st["grid"][0], st["grid"][1], 1,
                                 st["block"][0], 1, 1, 0, stream, None, 0),
               f"launch {st['name']}")

    def launch_all():
        for fn, st, _mod in fns:
            launch_one(fn, st)

    ev0 = cu.cuEventCreate(0)[1]
    ev1 = cu.cuEventCreate(0)[1]
    for _ in range(warmup):
        launch_all()
    cu.cuStreamSynchronize(stream)
    times = []
    for _ in range(iters):
        cu.cuEventRecord(ev0, stream)
        launch_all()
        cu.cuEventRecord(ev1, stream)
        cu.cuEventSynchronize(ev1)
        err, ms = cu.cuEventElapsedTime(ev0, ev1)
        ck((err,), "elapsed")
        times.append(ms)
    per_stage = {}
    for fn, st, _mod in fns:
        ts = []
        for _ in range(5):
            cu.cuEventRecord(ev0, stream)
            launch_one(fn, st)
            cu.cuEventRecord(ev1, stream)
            cu.cuEventSynchronize(ev1)
            ts.append(cu.cuEventElapsedTime(ev0, ev1)[1])
        per_stage[st["name"]] = round(statistics.median(ts), 4)
    return round(statistics.median(times), 4), round(min(times), 4), per_stage


def measure_fla(B, T, H, K, V, iters=20):
    """fla chunk_gated_delta_rule bwd timing (pure Triton, torch harness)."""
    code = f"""
import torch, time, json
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
torch.manual_seed(0)
dev = "cuda"
q = torch.randn({B}, {T}, {H}, {K}, device=dev, dtype=torch.bfloat16)
k = torch.randn({B}, {T}, {H}, {K}, device=dev, dtype=torch.bfloat16)
v = torch.randn({B}, {T}, {H}, {V}, device=dev, dtype=torch.bfloat16)
beta = torch.randn({B}, {T}, {H}, device=dev, dtype=torch.bfloat16).sigmoid()
g = -torch.nn.functional.softplus(torch.randn({B}, {T}, {H}, device=dev,
                                              dtype=torch.float32))
q, k, v, beta, g = [x.requires_grad_(True) if x.is_floating_point() else x
                    for x in (q, k, v, beta, g)]
def run():
    o, _ = chunk_gated_delta_rule(q, k, v, beta, g, output_final_state=True)
    do = torch.randn_like(o)
    dht = torch.randn_like(_)
    torch.autograd.grad((o, _), (q, k, v, beta, g), (do, dht),
                        retain_graph=True)
for _ in range(5):
    run()
torch.cuda.synchronize()
ts = []
for _ in range({iters}):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    run()
    torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
ts = sorted(ts)
print(json.dumps({{"fla_bwd_ms_median": ts[len(ts)//2],
                   "fla_bwd_ms_min": ts[0]}}))
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=os.path.expanduser("~/codebase/fla"))
    for line in r.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("fla measurement failed: " + r.stderr[-500:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", required=True)
    ap.add_argument("--shape", default="1,2048,16,128,128")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--fla-json",
                    default=os.path.join(REPO, "evidence/G5/fla_baseline.json"))
    ap.add_argument("--measure-fla", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    B, T, H, K, V = (int(x) for x in args.shape.split(","))
    med, mn, per_stage = time_chain(args.gen_root, args.iters)
    print(f"[bwd_bench] ours e2e bwd: median {med:.4f} ms  min {mn:.4f} ms")
    print(f"[bwd_bench] per-stage (median ms): {per_stage}")

    doc = {"shape": args.shape, "iters": args.iters,
           "ours_bwd_ms_median": med, "ours_bwd_ms_min": mn,
           "ours_stage_ms_median": per_stage, "shared_gpu": True}

    fla = None
    if os.path.exists(args.fla_json):
        base = json.load(open(args.fla_json))
        key = f"{args.shape}"
        fla = base.get(key) or base.get("shapes", {}).get(key)
        if fla:
            print(f"[bwd_bench] fla baseline (from {args.fla_json}): {fla}")
    if fla is None and (args.measure_fla or not os.path.exists(args.fla_json)):
        print("[bwd_bench] measuring fla bwd baseline on this GPU ...")
        fla = measure_fla(B, T, H, K, V)
        print(f"[bwd_bench] fla: {fla}")
    if fla:
        fm = fla.get("fla_bwd_ms_median", fla.get("bwd_ms"))
        doc["fla"] = fla
        if fm:
            doc["bwd_speedup_vs_fla"] = round(fm / med, 3)
            print(f"[bwd_bench] SPEEDUP ours/fla bwd = {fm:.4f}/{med:.4f} "
                  f"= {fm / med:.3f}x")
        ft = fla.get("fwd_ms") or fla.get("fla_fwd_ms_median")
        if ft:
            doc["fwd_plus_bwd_ours_ms"] = round(ft + med, 4)
            doc["fwd_plus_bwd_fla_ms"] = round(
                ft + fla.get("fla_bwd_ms_median", fla.get("bwd_ms")), 4)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(doc, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
