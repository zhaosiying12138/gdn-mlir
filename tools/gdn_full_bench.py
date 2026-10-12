#!/usr/bin/env python3
"""gdn_full_bench.py -- G5-B full-op forward timing vs the fla baselines.

Per bench shape (B1T2048 / B1T8192 / B4T2048, H16 K=V=128): builds the
staged trio + the persistent fused kernel (gdn_full_gen -> run_l2 ->
smem_fix), stages the shared G5-A inputs (evidence/G5/refs npz when
present, else the shared tools/gdn_bench_common.py recipe, seed 42),
times each path e2e with CUDA events (median of 50 after 10 warmups):

  staged: F1 -> DtoD handoff -> F2 -> DtoD handoff -> F3  (the DtoD
          copies are part of our cubin-per-kernel design and are INCLUDED
          in the timed e2e; noted in the report)
  fused : ONE persistent kernel launch

Prints a speedup table vs evidence/G5/fla_baseline.json (G5-A; fla
chunk_gated_delta_rule fwd, triton + default backends).  Per-config
subprocess isolation with wall timeouts (shared GPU protocol).

Usage:
  python tools/gdn_full_bench.py [--shapes B1T2048,B1T8192,B4T2048]
      [--warps 4] [--bv 64] [--iters 50] [--warmup 10]
      [--out evidence/G5/fwd_bench.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

REPO = os.path.expanduser("~/codebase/gdn-mlir")
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)

import gdn_full_rt as rt  # noqa: E402

SHAPES = {
    "B1T2048": (1, 2048, 16, 128, 128),
    "B1T8192": (1, 8192, 16, 128, 128),
    "B4T2048": (4, 2048, 16, 128, 128),
}
TAGS = {"B1T2048": "bench_B1T2048H16"}  # G5-A input npz available


def load_inputs(name, B, T, H, K, V):
    import numpy as np
    tag = TAGS.get(name)
    p = os.path.join(REPO, f"evidence/G5/refs/{tag}_inputs.npz")
    if tag and os.path.exists(p):
        d = np.load(p)
        return {k: d[k] for k in ("q_bf16", "k_bf16", "v_bf16", "beta", "g",
                                  "h0")}
    # shared recipe fallback (same generator that produced the G5-A npzs)
    import torch  # noqa: F401
    sys.path.insert(0, TOOLS)
    import gdn_bench_common as bc  # noqa: E402
    case = bc.gen_case(B, T, H, H, K, V, seed=bc.SEED)

    def f32(t):
        return t.detach().cpu().float().numpy()

    return {"q_bf16": f32(case["q"]), "k_bf16": f32(case["k"]),
            "v_bf16": f32(case["v"]), "beta": f32(case["beta"]),
            "g": f32(case["g"]), "h0": f32(case["h0"])}


def run_bench(name, gen_root, warps, bv, iters, warmup, f2s=1):
    import cuda.bindings.driver as cu
    import numpy as np

    B, T, H, K, V = SHAPES[name]
    C = 64
    Tp = ((T + C - 1) // C) * C
    NC = Tp // C
    shape = (B, T, H, K, V, C, NC, Tp)
    inp = load_inputs(name, B, T, H, K, V)

    rt._ck(cu.cuInit(0), "cuInit")
    e, dev = cu.cuDeviceGet(0); rt._ck((e,), "dev")
    e, ctx = cu.cuDevicePrimaryCtxRetain(dev); rt._ck((e,), "ctx")
    rt._ck(cu.cuCtxSetCurrent(ctx), "ctx")
    stream = cu.CUstream(0)

    out = {"name": name, "shape": {"B": B, "T": T, "H": H, "K": K, "V": V},
           "knobs": {"warps": warps, "bv": bv, "f2_segments": f2s},
           "gpu": rt.smi_snapshot()}

    # ---- staged path (G5-D round 2: mono cubin, shared globals, no DtoD;
    # G5-I round 4: f2s>1 -> the spec kernel list is f1/f2a/f2c/f3) ----
    gen_args = ["--path", "mono", "--shape", f"{B},{T},{H},{K},{V}",
                "--warps", str(warps), "--bv", str(bv), "--bv3", str(V)]
    if f2s > 1:
        gen_args += ["--f2-segments", str(f2s)]
    suffix = "" if f2s == 1 else f"_s{f2s}"
    km = rt.FullKernel(rt.build(gen_args,
                                os.path.join(gen_root,
                                             f"{name}_mono{suffix}")))
    rt.stage_inputs(km, inp, shape)
    knames = [k["name"] for k in km.spec.get("kernels", [])]

    def staged_once():
        for kn in knames:
            km.launch(stream, kn)
        rt._ck(cu.cuStreamSynchronize(stream), "sm")

    # ---- fused path ----
    d = rt.build(["--path", "fused", "--shape", f"{B},{T},{H},{K},{V}",
                  "--warps", str(warps), "--bv", str(bv)],
                 os.path.join(gen_root, f"{name}_fused"))
    kfus = rt.FullKernel(d)
    rt.stage_inputs(kfus, inp, shape)

    _, ev0 = cu.cuEventCreate(0)
    _, ev1 = cu.cuEventCreate(0)

    def timeit(fn):
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(iters):
            cu.cuEventRecord(ev0, stream)
            fn()
            cu.cuEventRecord(ev1, stream)
            rt._ck(cu.cuEventSynchronize(ev1), "evsync")
            _, ms = cu.cuEventElapsedTime(ev0, ev1)
            times.append(ms)
        return {"median_ms": round(statistics.median(times), 6),
                "min_ms": round(min(times), 6),
                "p25_ms": round(statistics.quantiles(times, n=4)[0], 6),
                "p75_ms": round(statistics.quantiles(times, n=4)[2], 6),
                "mean_ms": round(statistics.fmean(times), 6)}

    def fused_once():
        kfus.launch(stream)
        rt._ck(cu.cuStreamSynchronize(stream), "sf")

    out["staged"] = timeit(staged_once)
    for kn in knames:
        short = kn.replace("gdn_fwd_", "")
        out[short] = timeit(lambda kn=kn: (km.launch(stream, kn),
                                           rt._ck(cu.cuStreamSynchronize(stream),
                                                  "sk")))
    out["f2_total"] = None
    if "f2a" in [k.replace("gdn_fwd_", "") for k in knames]:
        out["f2_total"] = timeit(
            lambda: (km.launch(stream, "gdn_fwd_f2a"),
                     km.launch(stream, "gdn_fwd_f2c"),
                     rt._ck(cu.cuStreamSynchronize(stream), "sk2")))
    out["kernel_names"] = knames
    out["fused"] = timeit(fused_once)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="B1T2048,B1T8192,B4T2048")
    ap.add_argument("--gen-root", default=os.path.join(
        REPO, "evidence/G5/gen_fwd"))
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--bv", type=int, default=64)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--f2-segments", type=int, default=1,
                    help="G5-I: F2 segment-split count (mono staged path)")
    ap.add_argument("--out", default=os.path.join(REPO,
                                                  "evidence/G5/fwd_bench.json"))
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-args", default=None)
    args = ap.parse_args()

    if args.worker:
        name, gr, w, bv, it, wm, f2s = args.worker_args.split(",")
        try:
            r = run_bench(name, gr, int(w), int(bv), int(it), int(wm),
                          f2s=int(f2s))
        except Exception as ex:  # noqa: BLE001
            r = {"name": name, "status": "WORKER_CRASH",
                 "error": f"{type(ex).__name__}: {ex}"}
        print("RESULT:" + json.dumps(r))
        return 0

    doc = {"what": "G5-B full-op GDN forward: staged trio + persistent "
                   "fused kernel, CUDA events median-of-N, shared GPU",
           "f2_segments": args.f2_segments,
           "gpu_start": rt.smi_snapshot(), "results": {}}
    baselines = {}
    bl_path = os.path.join(REPO, "evidence/G5/fla_baseline.json")
    if os.path.exists(bl_path):
        bl = json.load(open(bl_path))
        for mode in ("triton", "default"):
            for s, e in bl["fla"].get(mode, {}).items():
                fwd = e.get("fwd") or {}
                if fwd.get("median_ms"):
                    baselines[f"fla_{mode}_{s}"] = fwd["median_ms"]
    doc["fla_baselines_median_ms"] = baselines

    for name in args.shapes.split(","):
        print(f"[gdn_full_bench] {name} ...", flush=True)
        try:
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--worker",
                 "--worker-args",
                 f"{name},{args.gen_root},{args.warps},{args.bv},"
                 f"{args.iters},{args.warmup},{args.f2_segments}"],
                capture_output=True, text=True, timeout=args.timeout, cwd=REPO)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT:")]
            r = (json.loads(line[-1][7:]) if line else
                 {"name": name, "status": "WORKER_CRASH",
                  "error": (p.stderr or p.stdout)[-600:]})
        except subprocess.TimeoutExpired:
            r = {"name": name, "status": "TIMEOUT",
                 "error": f"worker > {args.timeout}s"}
        doc["results"][name] = r
        st, fu = r.get("staged", {}), r.get("fused", {})
        print(f"    staged median={st.get('median_ms')}ms  "
              f"fused median={fu.get('median_ms')}ms", flush=True)

    doc["gpu_end"] = rt.smi_snapshot()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)

    # speedup table
    print("\n=== fwd e2e medians vs fla (shared GPU, noisy lower bounds) ===")
    print(f"{'shape':10s} {'ours staged':>12s} {'ours fused':>12s} "
          f"{'fla triton':>11s} {'fla def':>9s} {'fused/triton':>13s}")
    for name in args.shapes.split(","):
        r = doc["results"].get(name, {})
        st = (r.get("staged") or {}).get("median_ms")
        fu = (r.get("fused") or {}).get("median_ms")
        bt = baselines.get(f"fla_triton_{name}")
        bd = baselines.get(f"fla_default_{name}")
        sp = f"{bt / fu:.2f}x" if (bt and fu) else "-"
        print(f"{name:10s} {str(st):>12s} {str(fu):>12s} {str(bt):>11s} "
              f"{str(bd):>9s} {sp:>13s}")
    print(f"[gdn_full_bench] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
