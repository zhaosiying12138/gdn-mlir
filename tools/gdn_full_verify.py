#!/usr/bin/env python3
"""gdn_full_verify.py -- G5-B full-op forward numeric validation.

Runs BOTH paths (staged F1->F2->F3 and persistent fused) on the small
shape (B1 T512 H4 K=V=128) and on B1T2048H16, against the G5-A shared
JAX fp64 references (evidence/G5/refs/<tag>_ref.npz, produced from the
SAME inputs npz; poll first, these landed during G5-B).  Tolerance: the
fla contract from the workstream brief, o/ht <= 5e-3 ABS vs fp64 ref.

Inputs are the G5-A shared recipe (tools/gdn_bench_common.py, fla
test_chunk convention: q/k/v bf16 values, beta/g/h0 fp32, g NATURAL-log).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.expanduser("~/codebase/gdn-mlir")
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)

import gdn_full_rt as rt  # noqa: E402

TOL = 5e-3


def load_case(tag):
    d = os.path.join(REPO, "evidence/G5/refs")
    import numpy as np
    inp = dict(np.load(os.path.join(d, f"{tag}_inputs.npz")))
    ref = dict(np.load(os.path.join(d, f"{tag}_ref.npz")))
    B, T, H, K = inp["q_bf16"].shape
    V = inp["v_bf16"].shape[-1]
    return inp, ref, (B, T, H, K, V)


def run_paths(shape_tag, gen_root, warps, bv, timeout=600, f2s=1):
    """Worker body: build kernels, run both paths, return result dict."""
    import cuda.bindings.driver as cu
    import numpy as np

    inp, ref, (B, T, H, K, V) = load_case(shape_tag)
    C = 64
    Tp = ((T + C - 1) // C) * C
    NC = Tp // C
    shape = (B, T, H, K, V, C, NC, Tp)

    rt._ck(cu.cuInit(0), "cuInit")
    e, dev = cu.cuDeviceGet(0); rt._ck((e,), "dev")
    e, ctx = cu.cuDevicePrimaryCtxRetain(dev); rt._ck((e,), "ctx")
    rt._ck(cu.cuCtxSetCurrent(ctx), "ctx")
    stream = cu.CUstream(0)

    results = {}

    def compare(tagr, o_flat, ht_flat):
        ro_key = "ref__o_f64" if "ref__o_f64" in ref else "ref__o_f32"
        rh_key = "ref__ht_f64" if "ref__ht_f64" in ref else "ref__ht_f32"
        ro = ref[ro_key]
        rh = ref[rh_key]
        gotm = o_flat.reshape(B, Tp, H, V)[:, :T]
        refm = ro
        do_ = np.abs(gotm.astype(np.float64) - refm)
        htm = ht_flat.reshape(B, H, K, V)
        dh = np.abs(htm.astype(np.float64) - rh)
        results[tagr] = {
            "o_max_abs_diff": float(do_.max()),
            "ht_max_abs_diff": float(dh.max()),
            "o_over_tol": int((do_ > TOL).sum()),
            "ht_over_tol": int((dh > TOL).sum()),
            "o_total": int(refm.size), "ht_total": int(refm.size),
            "ref_keys": (ro_key, rh_key),
        }

    # ---- staged path (G5-D round 2: mono cubin, shared globals, no DtoD;
    # G5-I round 4: with f2s>1 the spec's kernel list is f1/f2a/f2c/f3) ----
    try:
        gen_args = ["--path", "mono", "--shape", f"{B},{T},{H},{K},{V}",
                    "--warps", str(warps), "--bv", str(bv), "--bv3", str(V)]
        if f2s > 1:
            gen_args += ["--f2-segments", str(f2s)]
        suffix = "" if f2s == 1 else f"_s{f2s}"
        d = rt.build(gen_args,
                     os.path.join(gen_root, f"{shape_tag}_mono{suffix}"))
        km = rt.FullKernel(d)
        rt.stage_inputs(km, inp, shape)
        for kn in [k["name"] for k in km.spec.get("kernels", [])]:
            km.launch(stream, kn)
        rt._ck(cu.cuStreamSynchronize(stream), "sync mono")
        o = km.readout("o", "bf16", B * Tp * H * V)
        ht = km.readout("ht", "f32", B * H * K * V)
        compare("staged", o, ht)
        results["staged"]["intermediates"] = {
            "u_nonzero": int((km.readout("u", "bf16",
                                         B * H * NC * C * V) != 0).sum()),
            "w_nonzero": int((km.readout("w", "bf16",
                                         B * H * NC * C * K) != 0).sum()),
            "h_all_nonzero": int((km.readout(
                "h_all", "bf16", B * H * NC * K * V) != 0).sum()),
        }
    except Exception as e:  # noqa: BLE001
        results["staged"] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}

    # ---- fused path ----
    try:
        d = rt.build(["--path", "fused", "--shape", f"{B},{T},{H},{K},{V}",
                      "--warps", str(warps), "--bv", str(bv)],
                     os.path.join(gen_root, f"{shape_tag}_fused"))
        kfus = rt.FullKernel(d)
        rt.stage_inputs(kfus, inp, shape)
        kfus.launch(stream)
        rt._ck(cu.cuStreamSynchronize(stream), "sync fused")
        o = kfus.readout("o", "bf16", B * Tp * H * V)
        ht = kfus.readout("ht", "f32", B * H * K * V)
        compare("fused", o, ht)
    except Exception as e:  # noqa: BLE001
        results["fused"] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}

    for p in ("staged", "fused"):
        if p in results and "o_max_abs_diff" in results[p]:
            r = results[p]
            r["status"] = ("PASS" if r["o_over_tol"] == 0
                           and r["ht_over_tol"] == 0 else "FAIL")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", default=os.path.join(REPO, "evidence/G5/gen_fwd"))
    ap.add_argument("--out", default=os.path.join(REPO, "evidence/G5/fwd_verify.json"))
    ap.add_argument("--shapes", default="small_B1T512H4,bench_B1T2048H16")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--bv", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--f2-segments", type=int, default=1,
                    help="G5-I: F2 segment-split count (mono staged path)")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-args", default=None)
    args = ap.parse_args()

    if args.worker:
        tag, gr, w, bv, f2s = args.worker_args.split(",")
        try:
            r = run_paths(tag, gr, int(w), int(bv), f2s=int(f2s))
        except Exception as e:  # noqa: BLE001
            r = {"status": "WORKER_CRASH", "error": f"{type(e).__name__}: {e}"}
        print("RESULT:" + json.dumps(r))
        return 0

    doc = {"tolerance": TOL, "contract": "o/ht max-abs-diff <= 5e-3 vs G5-A "
           "JAX fp64 refs (same inputs)", "gpu": rt.smi_snapshot(),
           "f2_segments": args.f2_segments, "results": {}}
    for tag in args.shapes.split(","):
        print(f"[gdn_full_verify] {tag} ...", flush=True)
        try:
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--worker",
                 "--worker-args",
                 f"{tag},{args.gen_root},{args.warps},{args.bv},"
                 f"{args.f2_segments}"],
                capture_output=True, text=True, timeout=args.timeout, cwd=REPO)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT:")]
            r = (json.loads(line[-1][7:]) if line else
                 {"status": "WORKER_CRASH", "error": (p.stderr or p.stdout)[-800:]})
        except subprocess.TimeoutExpired:
            r = {"status": "TIMEOUT", "error": f"worker > {args.timeout}s"}
        doc["results"][tag] = r
        for path in ("staged", "fused"):
            rr = r.get(path, {})
            print(f"    {path}: {rr.get('status')} "
                  f"o_diff={rr.get('o_max_abs_diff')} ht_diff={rr.get('ht_max_abs_diff')}",
                  flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"[gdn_full_verify] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
