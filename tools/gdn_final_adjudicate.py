#!/usr/bin/env python3
"""gdn_final_adjudicate.py -- G5 FINAL clean-window adjudication harness.

One session, INTERLEAVED fla-vs-ours forward measurement per shape:

    for each shape: 5 x { ours fwd, fla fwd } alternating, CUDA events,
                     medians, speedup

The interleave is the point: on this shared GPU, a co-job's load bursts
between two SEPARATE bench sessions can fake a 1.5-2x swing either way
(measured in G5-A/G5-D).  Alternating the two contenders inside the same
window, sample by sample, makes the speedup ratio robust to everything
except a burst INSIDE one interleaved pair.

Inputs: the shared G5-A recipe (tools/gdn_bench_common.py, fla
test_chunk convention, seed 42) -- identical tensors for both sides
(torch bf16 on device for fla; the same values through the numpy fp32
upload path of gdn_full_rt for ours).

fla dispatch: FLA_FLASH_QLA=0 set BEFORE import (pure-Triton backend,
the beat target); backend probe recorded in the output.

ours: the staged mono trio (gdn_fwd_f1/f2/f3, round-3 generator) via
gdn_full_rt; kernels are rebuilt only if missing (--force to rebuild).

Window quality is recorded, not assumed: nvidia-smi snapshots before /
after each shape, per-sample min/median/max for BOTH sides, and the
median-of-ratios (per-pair speedups) alongside the ratio-of-medians --
if the two diverge the window was dirty.

Usage (conda env gdn):
  python tools/gdn_final_adjudicate.py [--shapes B1T2048,B1T8192,B4T2048]
      [--pairs 5] [--warps 8] [--bv 64] [--stats]
      [--out evidence/G5/final_adjudicate.json] [--force]

--stats (G5-N): adds a per-shape "stats" block over the per-pair speedup
ratios -- median / IQR / min / max, exact two-sided sign-test p vs
ratio 1.0, and the order-of-magnitude decidability rule
(decidable = (median-1) > 3 * IQR/median, ties/IQR=0 handled; a gap that
large cannot be co-load noise -- see evidence/G5/adjudication_methodology.md).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys

REPO = os.path.expanduser("~/codebase/gdn-mlir")
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)

# dispatch env must be set before fla import (can_use is process-cached)
os.environ["FLA_FLASH_QLA"] = "0"

import gdn_full_rt as rt  # noqa: E402


def smi_snapshot():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,"
             "clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        u, m, tot, clk = r.stdout.strip().split("\n")[0].split(",")
        return {"util_pct": int(u.strip()), "mem_used_mib": int(m.strip()),
                "mem_total_mib": int(tot.strip()),
                "sm_clock_mhz": int(clk.strip())}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# -- G5-N: per-pair ratio statistics ----------------------------------------

def sign_test_p(ratios, ref=1.0):
    """Exact two-sided sign-test p for median(ratios) == ref.

    Wins/losses vs ref (ties dropped); p = 2 * P(Bin(n, .5) <= min(w,
    n-w)), capped at 1. Exact for n <= ~20 (pairs here are 5-30); no
    scipy dependency. None if every pair is a tie.
    """
    wins = sum(1 for r in ratios if r > ref)
    losses = sum(1 for r in ratios if r < ref)
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    p_lt = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return round(min(1.0, 2.0 * p_lt), 6)


def pair_stats(ratios):
    """median / IQR / min / max / sign-test p / decidability.

    IQR from statistics.quantiles(n=4, method="inclusive") -- the same
    linear interpolation the median uses, defined for n >= 2. The
    decidability rule (adjudication_methodology.md (b)): the window's
    co-load noise cannot explain an effect (median - 1) larger than
    3x the relative spread IQR/median.
    """
    med = statistics.median(ratios)
    if len(ratios) >= 2:
        q1, _, q3 = statistics.quantiles(ratios, n=4, method="inclusive")
    else:
        q1 = q3 = ratios[0]
    iqr = q3 - q1
    iqr_rel = (iqr / med) if med else float("inf")
    effect = med - 1.0
    decidable = bool(effect > 0 and (iqr_rel == 0 or
                                     effect > 3.0 * iqr_rel))
    return {
        "n_pairs": len(ratios),
        "median_ratio": round(med, 4),
        "iqr": round(iqr, 4),
        "iqr_over_median": round(iqr_rel, 4),
        "min_ratio": round(min(ratios), 4),
        "max_ratio": round(max(ratios), 4),
        "sign_test_p_vs_1": sign_test_p(ratios),
        "decidable_rule": "(median-1) > 3*IQR/median",
        "decidable": decidable,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shapes", default="B1T2048,B1T8192,B4T2048")
    ap.add_argument("--pairs", type=int, default=5,
                    help="interleaved (ours, fla) pairs per shape")
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--bv", type=int, default=64)
    ap.add_argument("--bv3", type=int, default=None)
    ap.add_argument("--f2-segments", type=int, default=1,
                    help="G5-I round 4: F2 segment-split count (1=serial)")
    ap.add_argument("--stats", action="store_true",
                    help="G5-N: add per-pair ratio stats (median/IQR/"
                         "min/max/sign-test p/decidability) per shape")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--gen-root", default=os.path.join(
        REPO, "evidence/G5/gen_fwd"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "evidence/G5/final_adjudicate.json"))
    ap.add_argument("--force", action="store_true",
                    help="force kernel rebuild")
    args = ap.parse_args()

    import numpy as np
    import torch  # noqa: F401  (env gdn)
    import gdn_bench_common as C  # noqa: E402
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: E402
    from fla.ops.gated_delta_rule.backends.flash_qla import (
        FlashQLABackend)  # noqa: E402
    import cuda.bindings.driver as cu  # noqa: E402

    probe = {
        "FLA_FLASH_QLA": os.environ.get("FLA_FLASH_QLA", "<unset>"),
        "flash_qla_importable": __import__("importlib.util", fromlist=[
            "util"]).find_spec("flash_qla") is not None,
        "flash_qla_backend_can_use": FlashQLABackend.can_use(),
        "torch": torch.__version__,
    }

    rt._ck(cu.cuInit(0), "cuInit")
    e, dev = cu.cuDeviceGet(0); rt._ck((e,), "dev")
    e, ctx = cu.cuDevicePrimaryCtxRetain(dev); rt._ck((e,), "ctx")
    rt._ck(cu.cuCtxSetCurrent(ctx), "ctx")
    stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)

    doc = {
        "what": "G5 final adjudication: interleaved fla-vs-ours fwd, "
                "5 pairs/shape, CUDA events, shared-recipe inputs",
        "probe": probe,
        "knobs": {"warps": args.warps, "bv": args.bv,
                  "f2_segments": args.f2_segments},
        "gpu_start": smi_snapshot(),
        "results": {},
    }

    for name in args.shapes.split(","):
        shape = C.bench_shape_by_name(name)
        B, T, H, HV, K, V = (shape["B"], shape["T"], shape["H"],
                             shape["HV"], shape["K"], shape["V"])
        scale = K ** -0.5
        case = C.gen_case(B, T, H, HV, K, V, seed=C.SEED)
        io = C.to_device_bench(case, device="cuda",
                               dtype=C.DTYPE_BENCH)
        inp_np = {"q_bf16": case["q"].detach().cpu().float().numpy(),
                  "k_bf16": case["k"].detach().cpu().float().numpy(),
                  "v_bf16": case["v"].detach().cpu().float().numpy(),
                  "beta": case["beta"].detach().cpu().float().numpy(),
                  "g": case["g"].detach().cpu().float().numpy(),
                  "h0": case["h0"].detach().cpu().float().numpy()}

        # ---- ours: staged mono trio ----
        Cc = 64
        Tp = ((T + Cc - 1) // Cc) * Cc
        NC = Tp // Cc
        mshape = (B, T, H, K, V, Cc, NC, Tp)
        # distinct gen dir per config: the build cache short-circuits on an
        # existing cubin (the round-3 adjudication served stale w4 cubins
        # for a w8 request through exactly this hole)
        suffix = "" if args.f2_segments == 1 else f"_s{args.f2_segments}"
        gen_args = ["--path", "mono", "--shape", f"{B},{T},{H},{K},{V}",
                    "--warps", str(args.warps), "--bv", str(args.bv),
                    "--bv3", str(args.bv3 or V)]
        if args.f2_segments > 1:
            gen_args += ["--f2-segments", str(args.f2_segments)]
        km = rt.FullKernel(rt.build(
            gen_args,
            os.path.join(args.gen_root, f"{name}_mono{suffix}"),
            force=args.force))
        rt.stage_inputs(km, inp_np, mshape)
        knames = [k["name"] for k in km.spec.get("kernels", [])]

        def ours_fwd():
            for kn in knames:
                km.launch(stream, kn)
            rt._ck(cu.cuStreamSynchronize(stream), "sm")

        def fla_fwd():
            with torch.no_grad():
                chunk_gated_delta_rule(
                    q=io["q"], k=io["k"], v=io["v"], g=io["g"],
                    beta=io["beta"], scale=scale,
                    initial_state=io["h0"], output_final_state=True)
            torch.cuda.synchronize()

        # warmups (absorbs fla autotune + our cubin load)
        for _ in range(args.warmup):
            ours_fwd()
            fla_fwd()

        _, ev0 = cu.cuEventCreate(0)
        _, ev1 = cu.cuEventCreate(0)

        def time_ours():
            cu.cuEventRecord(ev0, stream)
            for kn in knames:
                km.launch(stream, kn)
            cu.cuEventRecord(ev1, stream)
            rt._ck(cu.cuEventSynchronize(ev1), "ev")
            _, ms = cu.cuEventElapsedTime(ev0, ev1)
            return ms

        def time_fla():
            s = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad():
                chunk_gated_delta_rule(
                    q=io["q"], k=io["k"], v=io["v"], g=io["g"],
                    beta=io["beta"], scale=scale,
                    initial_state=io["h0"], output_final_state=True)
            en.record()
            torch.cuda.synchronize()
            return s.elapsed_time(en)

        snap_pre = smi_snapshot()
        ours_t, fla_t, pairs_ratio = [], [], []
        for _ in range(args.pairs):
            a = time_ours()
            b = time_fla()
            ours_t.append(a)
            fla_t.append(b)
            pairs_ratio.append(b / a)   # >1 => ours faster
        snap_post = smi_snapshot()

        r = {
            "shape": {k: shape[k] for k in ("name", "B", "T", "H", "K", "V")},
            "ours_samples_ms": [round(x, 6) for x in ours_t],
            "fla_samples_ms": [round(x, 6) for x in fla_t],
            "ours_median_ms": round(statistics.median(ours_t), 6),
            "fla_median_ms": round(statistics.median(fla_t), 6),
            "speedup_ratio_of_medians": round(
                statistics.median(fla_t) / statistics.median(ours_t), 4),
            "speedup_median_of_ratios": round(
                statistics.median(pairs_ratio), 4),
            "ours_min_ms": round(min(ours_t), 6),
            "fla_min_ms": round(min(fla_t), 6),
            "window_pre": snap_pre,
            "window_post": snap_post,
        }
        if args.stats:
            r["stats"] = pair_stats(pairs_ratio)
        r["window_clean"] = (
            abs(r["speedup_ratio_of_medians"]
                - r["speedup_median_of_ratios"]) < 0.15)
        doc["results"][name] = r
        st = r.get("stats")
        st_s = (f"  IQR {st['iqr']:.4f} (rel {st['iqr_over_median']:.3f}) "
                f"p={st['sign_test_p_vs_1']} "
                f"{'DECIDABLE' if st['decidable'] else 'co-load-sensitive'}"
                ) if st else ""
        print(f"[adjudicate] {name}: ours {r['ours_median_ms']:.4f}ms  "
              f"fla {r['fla_median_ms']:.4f}ms  "
              f"speedup {r['speedup_ratio_of_medians']:.3f}x "
              f"(per-pair median {r['speedup_median_of_ratios']:.3f}x, "
              f"{'clean' if r['window_clean'] else 'NOISY'}){st_s}",
              flush=True)

    doc["gpu_end"] = smi_snapshot()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"[adjudicate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
