#!/usr/bin/env python3
"""GDN-MLIR G5-A -- FlashQLA (TileLang) forward baseline on the G5 bench shapes.

Runs in conda env `gdn-tl` (tilelang 0.1.15, torch cu130, flash_qla editable at
~/codebase/FlashQLA).  FlashQLA on SM120 (RTX 5090 Laptop) ships a bf16 FORWARD
kernel only -> fwd-only timing, no autograd, on the same shapes/inputs as
tools/gdn_bench_flaptx.py (shared recipe via tools/gdn_bench_common.py:
H=HV=16, K=V=128, bf16 q/k/v, fp32 beta/g, zeros h0, scale=K**-0.5,
output_final_state=True).

CUDA events, >=10 warmup (first call absorbs TileLang JIT; wall time recorded),
>=50 reps, median.  Per-kernel profile once (top 5).  Unsupported shapes are
skipped with a note.  initial_state dtype is tried fp32-first (what fla's
dispatcher passes through), falling back to bf16, then to None.

Output: evidence/G5/flashqla_fwd.json

Usage (conda env gdn-tl):
  python tools/gdn_bench_flashqlaptx.py --out evidence/G5/flashqla_fwd.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser("~/codebase/gdn-mlir")

sys.path.insert(0, HERE)
import torch  # noqa: E402

import gdn_bench_common as C  # noqa: E402
import flash_qla  # noqa: E402
import tilelang  # noqa: E402

# --- sm120 runtime shim (upstream export bug, NOT a patch to FlashQLA) ------ #
# chunk/cp/preprocess.py: cp_preprocess_fwd/bwd do
#     from flash_qla.ops.gated_delta_rule.chunk import (..., aggregate_card_state, ...)
# at the top of the function, but the sm120 branch of chunk/__init__.py
# deliberately does not export aggregate_card_state (it only exists for
# SM90/SM100 inter+intra CP; on sm120 _assert_inter_intra_supported accepts
# None for pure-inter/pure-intra).  Without the symbol the PUBLIC entry point
# raises ImportError on every sm120 call.  Injecting None restores the
# documented sm120 semantics; recorded in the JSON as sm120_aggregate_shim.
import flash_qla.ops.gated_delta_rule.chunk as _fq_chunk  # noqa: E402
SM120_SHIM = False
if not hasattr(_fq_chunk, "aggregate_card_state"):
    _fq_chunk.aggregate_card_state = None
    SM120_SHIM = True


_BDV_STATE = {"orig": None}


def _set_fwd_block_dv(bdv):
    """Force (bdv=int) or clear (bdv=None) the block_DV knob of the sm120 fwd
    kernel -- a real parameter of tilelang_fused_chunk_gdr_fwd, hardcoded to
    128 at the fused_fwd.py call site.  With tilelang 0.1.15 the shipped
    block_DV=128 config needs 112,640 B dynamic smem > the sm120 101,376 B
    (99KB) opt-in limit and cannot launch; block_DV=64 halves the DV-scaling
    smem buffers (~77KB) and launches.  Upstream's own 5090 numbers
    (benchmark_results_5090.txt) were taken with tilelang 0.1.13.  Fully
    disclosed in the JSON as fwd_block_dv; outputs are cross-checked against
    fla Triton fwd on the same inputs (evidence/G5/refs/).
    """
    import flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120.fused_fwd as ff
    if _BDV_STATE["orig"] is None:
        _BDV_STATE["orig"] = ff.tilelang_fused_chunk_gdr_fwd
    orig = _BDV_STATE["orig"]
    if bdv is None:
        ff.tilelang_fused_chunk_gdr_fwd = orig
        return

    def wrapped(*a, **kw):
        kw["block_DV"] = bdv
        return orig(*a, **kw)

    ff.tilelang_fused_chunk_gdr_fwd = wrapped


def smi_snapshot() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"<smi failed: {e}>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "evidence/G5/flashqla_fwd.json"))
    ap.add_argument("--shapes", default="B1T2048,B1T8192,B4T2048")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "FlashQLA bench needs CUDA (env gdn-tl)"
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)

    out = {"meta": {
        "what": "FlashQLA (TileLang) chunk_gated_delta_rule fwd baselines (G5-A)",
        "gpu": smi_snapshot(),
        "gpu_capability": f"sm_{cap[0]}{cap[1]}",
        "torch": torch.__version__,
        "tilelang": tilelang.__version__,
        "flash_qla": os.path.dirname(os.path.dirname(flash_qla.__file__)),
        "limits": "SM120: fwd bf16 only (no backward) -- fwd-only benchmark",
        "sm120_aggregate_shim": SM120_SHIM,
        "sm120_chunk_size": getattr(_fq_chunk, "CHUNK_SIZE", None),
        "call_kwargs": ("auto_cp=True (FlashQLA default, same as fla's dispatcher "
                        "passes), use_qk_l2norm_in_kernel=False (q/k "
                        "pre-normalized, test_chunk recipe)"),
        "shared_gpu_note": ("opportunistic window; co-job ~17.9GB resident -- "
                            "absolute times noisy by design"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": args.warmup, "reps": args.reps,
    }, "flashqla_fwd": {}}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for shape_name in args.shapes.split(","):
        shape = C.bench_shape_by_name(shape_name)
        B, T, H, HV, K, V = (shape["B"], shape["T"], shape["H"], shape["HV"],
                             shape["K"], shape["V"])
        scale = K ** -0.5
        entry = {"shape": {k: shape[k] for k in ("name", "B", "T", "H", "HV", "K", "V")},
                 "dtype": "bfloat16 fwd (q/k/v), beta/g fp32", "scale": scale}

        if not (K == 128 and V == 128):
            entry["skipped"] = f"FlashQLA requires K=V=128, got K={K}, V={V}"
            out["flashqla_fwd"][shape_name] = entry
            continue

        try:
            case = C.gen_case(B, T, H, HV, K, V, seed=C.SEED)
            io = C.to_device_bench(case, device=dev, dtype=C.DTYPE_BENCH)
            # Attempt chain, shipped config first:
            #   1. auto_cp=True (what fla's dispatcher passes) -- fails on sm120:
            #      intra-CP preprocess kernels request 110KB dynamic smem > 99KB
            #      opt-in limit;
            #   2. auto_cp=False dense with the shipped block_DV=128 fwd kernel --
            #      also 112,640B > 101,376B under tilelang 0.1.15 (upstream 5090
            #      numbers were taken with tilelang 0.1.13);
            #   3. auto_cp=False dense with the kernel's own block_DV parameter
            #      forced to 64 (~77KB, launches; disclosed as fwd_block_dv=64).
            h0_mode = auto_cp_mode = None
            bdv_mode = 128
            o = ht = None
            last = None
            for bdv in (128, 64):
                _set_fwd_block_dv(bdv if bdv != 128 else None)
                for acp in (True, False):
                    for h0_key in ("fp32", "bf16", "none"):
                        h0 = None if h0_key == "none" else io["h0"].to(
                            torch.bfloat16 if h0_key == "bf16" else torch.float32)
                        try:
                            with torch.no_grad():
                                o, ht = flash_qla.chunk_gated_delta_rule(
                                    q=io["q"], k=io["k"], v=io["v"], g=io["g"],
                                    beta=io["beta"], scale=scale,
                                    initial_state=h0, output_final_state=True,
                                    auto_cp=acp)
                            h0_mode, auto_cp_mode, bdv_mode = h0_key, acp, bdv
                            break
                        except Exception as e:
                            last = f"{type(e).__name__}: {e}"
                    if h0_mode is not None:
                        break
                if h0_mode is not None:
                    break
            if h0_mode is None:
                raise RuntimeError(f"all auto_cp/h0/block_DV modes failed; "
                                   f"last: {last}")
            entry["initial_state_mode"] = h0_mode
            entry["auto_cp"] = auto_cp_mode
            entry["fwd_block_dv"] = bdv_mode
            if auto_cp_mode is False:
                entry["auto_cp_note"] = ("auto_cp=True (shipped default) fails on "
                                         "sm120: intra-CP preprocess kernels request "
                                         "110KB dynamic smem > 99KB/CTA opt-in; dense "
                                         "auto_cp=False path timed instead")
            if bdv_mode != 128:
                entry["block_dv_note"] = ("shipped block_DV=128 fwd kernel needs "
                                          "112,640B dynamic smem > 101,376B sm120 "
                                          "opt-in under tilelang 0.1.15 (upstream 5090 "
                                          "results used tilelang 0.1.13); timed with "
                                          "the kernel's own block_DV parameter forced "
                                          "to 64 (~77KB). Outputs cross-checked vs "
                                          "fla Triton fwd (see refs/)")
            entry["fwd_output_finite"] = bool(torch.isfinite(o.float()).all().item())

            # dump o/ht once for cross-backend (fla-vs-FlashQLA) comparison
            if shape_name == "B1T2048":
                refdir = os.path.join(os.path.dirname(args.out), "refs")
                os.makedirs(refdir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(refdir, "flashqla_B1T2048_fwd.npz"),
                    o=o.detach().to(torch.float32).cpu().numpy(),
                    ht=ht.detach().to(torch.float32).cpu().numpy())

            def call():
                with torch.no_grad():
                    return flash_qla.chunk_gated_delta_rule(
                        q=io["q"], k=io["k"], v=io["v"], g=io["g"],
                        beta=io["beta"], scale=scale,
                        initial_state=None if h0_mode == "none" else h0,
                        output_final_state=True, auto_cp=auto_cp_mode)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            call()
            torch.cuda.synchronize()
            entry["first_call_s"] = round(time.perf_counter() - t0, 3)  # TileLang JIT
            for _ in range(args.warmup - 1):
                call()
            torch.cuda.synchronize()
            times = []
            for _ in range(args.reps):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                call()
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            entry["fwd"] = {
                "median_ms": statistics.median(times),
                "mean_ms": statistics.fmean(times),
                "min_ms": min(times), "max_ms": max(times),
                "p25_ms": statistics.quantiles(times, n=4)[0],
                "p75_ms": statistics.quantiles(times, n=4)[2],
                "reps": args.reps, "warmup": args.warmup,
                "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            }

            # per-kernel profile (top 5)
            try:
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(3):
                        call()
                    torch.cuda.synchronize()
                evs = [e for e in prof.key_averages()
                       if getattr(e, "self_device_time_total", 0) > 0]
                evs.sort(key=lambda e: -e.self_device_time_total)
                total = sum(e.self_device_time_total for e in evs) or 1.0
                entry["kernels_fwd"] = [{
                    "kernel": e.key[:160],
                    "count_per_iter": e.count / 3,
                    "self_us_per_iter": round(e.self_device_time_total / 3, 1),
                    "pct": round(100.0 * e.self_device_time_total / total, 2),
                } for e in evs[:5]]
            except Exception as e:
                entry["kernels_fwd"] = {"error": f"{type(e).__name__}: {e}"}

        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        out["flashqla_fwd"][shape_name] = entry
        f = entry.get("fwd", {})
        print(f"[gdn_bench_flashqlaptx] {shape_name}: "
              f"{f.get('median_ms', entry.get('error', entry.get('skipped', '?')))}",
              flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gdn_bench_flashqlaptx] wrote {args.out}")


if __name__ == "__main__":
    main()
