#!/usr/bin/env python3
"""RC1: per-cell fla chunk_gated_delta_rule baseline (pure-Triton discipline).

Phases (one process per invocation; conda gdn):
  --phase cold: TRITON_CACHE_DIR MUST point at a fresh empty dir (caller).
      Times the FIRST fwd call and FIRST autograd.grad (compile+autotune
      included) -- the per-shape cold autotune tax, G6 methodology.
  --phase warm: 5 warmup rounds, then 30 reps each of (a) fwd no_grad,
      (b) autograd.grad only (graph pre-built, retain_graph=True),
      (c) fwd+grad together.  CUDA-event medians; bwd = the (b) series
      (cross-checked against (c)-(a) derived).

Inputs: tools/gdn_bench_common.gen_case (seed 42) -- bit-identical to the
refs and every other G5/RC1 measurement.  FLA_FLASH_QLA=0 set before fla
import.  Usage:
  python tools/rc1_flacell.py --shape 1,2048,32,128,128 --tag H32 \
      --phase warm --out evidence/G5/scratch/rc1_logs/fla_H32.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("FLA_FLASH_QLA", "0")   # BEFORE fla import
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch  # noqa: E402

import gdn_bench_common as C  # noqa: E402


def ev_time(fn, iters, warmup=5):
    """Median/min CUDA-event ms over iters calls of fn()."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return round(statistics.median(ts), 4), round(min(ts), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, help="B,T,H,K,V")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--phase", choices=("cold", "warm"), required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    B, T, H, K, V = (int(x) for x in args.shape.split(","))
    scale = K ** -0.5

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    case = C.gen_case(B, T, H, H, K, V, seed=C.SEED)
    io = C.to_device_bench(case, requires_grad=False)
    gi = {k: (v.detach().clone().requires_grad_(True)
              if k in ("q", "k", "v", "beta", "g") else v)
          for k, v in io.items()}

    def fwd(inp):
        return chunk_gated_delta_rule(
            q=inp["q"], k=inp["k"], v=inp["v"], g=inp["g"],
            beta=inp["beta"], scale=scale, initial_state=None,
            output_final_state=True)

    doc = {"tag": args.tag, "shape": args.shape, "phase": args.phase,
           "fla_dispatch": "FLA_FLASH_QLA=0 (pure Triton)",
           "torch": torch.__version__,
           "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR", "default"),
           "shared_gpu": True}

    if args.phase == "cold":
        assert os.environ.get("TRITON_CACHE_DIR"), \
            "cold phase needs a fresh TRITON_CACHE_DIR"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o, ht = fwd(gi)
        torch.cuda.synchronize()
        t_fwd_cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        torch.autograd.grad(
            (o * io["do"]).sum() + (ht * io["dht"]).sum(),
            (gi["q"], gi["k"], gi["v"], gi["beta"], gi["g"]))
        torch.cuda.synchronize()
        t_bwd_cold = time.perf_counter() - t0
        # second-call warm sanity
        t0 = time.perf_counter()
        o2, _ = fwd(gi)
        torch.cuda.synchronize()
        t_fwd_warm2 = time.perf_counter() - t0
        doc.update({"fwd_cold_s": round(t_fwd_cold, 2),
                    "bwd_cold_s": round(t_bwd_cold, 2),
                    "fwd_second_call_s": round(t_fwd_warm2, 4)})
        print(json.dumps(doc))
        if args.out:
            json.dump(doc, open(args.out, "w"), indent=1)
        return 0

    # warm phase ----------------------------------------------------------- #
    o, ht = fwd(gi)
    lossfn = lambda o_, ht_: (o_ * io["do"]).sum() + (ht_ * io["dht"]).sum()  # noqa: E731
    grads = torch.autograd.grad(lossfn(o, ht),
                                (gi["q"], gi["k"], gi["v"], gi["beta"],
                                 gi["g"]), retain_graph=True)

    f_med, f_min = ev_time(lambda: fwd(io), args.iters)
    b_med, b_min = ev_time(
        lambda: torch.autograd.grad(lossfn(o, ht),
                                    (gi["q"], gi["k"], gi["v"], gi["beta"],
                                     gi["g"]), retain_graph=True),
        args.iters)
    fb_med, fb_min = ev_time(
        lambda: torch.autograd.grad(
            lossfn(*fwd(gi)),
            (gi["q"], gi["k"], gi["v"], gi["beta"], gi["g"])),
        args.iters)
    doc.update({"fla_fwd_ms_median": f_med, "fla_fwd_ms_min": f_min,
                "fla_bwd_ms_median": b_med, "fla_bwd_ms_min": b_min,
                "fla_fwdbwd_ms_median": fb_med, "fla_fwdbwd_ms_min": fb_min,
                "fla_bwd_derived_ms": round(fb_med - f_med, 4)})
    print(json.dumps(doc))
    if args.out:
        json.dump(doc, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
