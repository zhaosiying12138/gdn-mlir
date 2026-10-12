#!/usr/bin/env python3
"""GDN-MLIR G5-A -- fla `chunk_gated_delta_rule` baseline timing harness (GPU).

Times fla's chunked GDN op on the G5 bench shapes
(B1T2048 / B1T8192 / B4T2048, H=HV=16, K=V=128, bf16 io, scale=K**-0.5,
test_chunk input recipe via tools/gdn_bench_common.py) in TWO dispatch modes:

  * triton  -- FLA_FLASH_QLA=0 set INSIDE the child process via os.environ
               BEFORE importing fla (pure-Triton backend; the G5 beat-target
              口径).  The backend probe asserts FlashQLABackend.can_use() is
               False, i.e. the env var really disables the TileLang backend
               (fla/ops/backends/__init__.py BaseBackend.is_enabled reads
               os.environ at first use; can_use is @cache-per-process, hence
               one child process per mode).
  * default -- FLA_FLASH_QLA unset; record whatever fla's dispatcher does.
               NOTE: on this box (env gdn) the flash_qla package is not
               importable, so default dispatch falls back to the same Triton
               path -- recorded honestly via the backend probe; the
               FlashQLA-TileLang fwd numbers come from
               tools/gdn_bench_flashqlaptx.py in env gdn-tl.

Per shape x mode, a worker subprocess measures:
  * fwd-only  (torch.no_grad, output_final_state=True)   -- inference口径
  * fwd+bwd   (torch.autograd.grad of (o*do).sum()+(ht*dht).sum())
with >=10 warmup (also absorbs fla/Triton autotune; first-call wall time is
recorded) and >=50 timed reps, CUDA events, median reported.
Once per shape x mode, torch.profiler captures a per-kernel breakdown
(top kernels, self device time) for fwd and for fwd+bwd; the per-kernel
bwd attribution = fwdbwd_total - fwd_total per kernel name.

Output: evidence/G5/fla_baseline.json (merged over modes/shapes).

Usage (conda env gdn):
  python tools/gdn_bench_flaptx.py --out evidence/G5/fla_baseline.json \
      [--shapes B1T2048,B1T8192,B4T2048] [--modes triton,default]
      [--warmup 10] [--reps 50]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser("~/codebase/gdn-mlir")
JSON_MARK = "###G5JSON###"


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


# --------------------------------------------------------------------------- #
# worker (one process per (mode, shape))
# --------------------------------------------------------------------------- #
def run_worker(mode: str, shape_name: str, warmup: int, reps: int) -> None:
    # dispatch environment must be set BEFORE importing fla (can_use is cached)
    if mode == "triton":
        os.environ["FLA_FLASH_QLA"] = "0"
    else:
        os.environ.pop("FLA_FLASH_QLA", None)

    sys.path.insert(0, HERE)
    import torch  # noqa: E402

    import gdn_bench_common as C  # noqa: E402

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: E402
    from fla.ops.gated_delta_rule.backends.flash_qla import FlashQLABackend  # noqa: E402
    from fla.utils import device as fla_device  # noqa: E402
    import fla  # noqa: E402
    import triton  # noqa: E402

    # backend probe: prove the env var actually disables the TileLang backend
    probe = {
        "mode": mode,
        "FLA_FLASH_QLA": os.environ.get("FLA_FLASH_QLA", "<unset>"),
        "flash_qla_importable": importlib.util.find_spec("flash_qla") is not None,
        "flash_qla_backend_enabled": FlashQLABackend.is_enabled(),
        "flash_qla_backend_can_use": FlashQLABackend.can_use(),
    }

    shape = C.bench_shape_by_name(shape_name)
    B, T, H, HV, K, V = (shape["B"], shape["T"], shape["H"], shape["HV"],
                         shape["K"], shape["V"])
    scale = K ** -0.5
    dev = fla_device

    case = C.gen_case(B, T, H, HV, K, V, seed=C.SEED)
    io = C.to_device_bench(case, device=dev, dtype=C.DTYPE_BENCH)
    io_g = C.to_device_bench(case, device=dev, dtype=C.DTYPE_BENCH,
                             requires_grad=True)

    result = {
        "shape": {k: shape[k] for k in ("name", "B", "T", "H", "HV", "K", "V")},
        "dtype": "bfloat16 (q/k/v/do; beta/g/h0/dht fp32, per test_chunk)",
        "scale": scale,
        "backend_probe": probe,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "fla": fla.__version__ if hasattr(fla, "__version__") else "<n/a>",
    }

    def call_fwd(inp):
        with torch.no_grad():
            return chunk_gated_delta_rule(
                q=inp["q"], k=inp["k"], v=inp["v"], g=inp["g"], beta=inp["beta"],
                scale=scale, initial_state=inp["h0"], output_final_state=True)

    def call_fwdbwd(inp):
        q, k, v, beta, g, h0 = (inp["q"], inp["k"], inp["v"], inp["beta"],
                                inp["g"], inp["h0"])
        o, ht = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=h0, output_final_state=True)
        loss = (o * inp["do"]).sum() + (ht * inp["dht"]).sum()
        return torch.autograd.grad(loss, (q, k, v, beta, g, h0))

    def timed(fn, arg):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        fn(arg)  # first call: autotune/compile
        torch.cuda.synchronize()
        first_call_s = time.perf_counter() - t0
        for _ in range(warmup - 1):
            fn(arg)
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn(arg)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        peak_mb = torch.cuda.max_memory_allocated() / 2**20
        return {
            "median_ms": statistics.median(times),
            "mean_ms": statistics.fmean(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "p25_ms": statistics.quantiles(times, n=4)[0],
            "p75_ms": statistics.quantiles(times, n=4)[2],
            "reps": reps, "warmup": warmup,
            "first_call_s": round(first_call_s, 3),
            "peak_alloc_mb": round(peak_mb, 1),
        }

    def profile_topk(fn, arg, topk=10, iters=3):
        from torch.profiler import ProfilerActivity, profile
        fn(arg)  # ensure warm
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn(arg)
            torch.cuda.synchronize()
        evs = [e for e in prof.key_averages()
               if getattr(e, "self_device_time_total", 0) > 0]
        evs.sort(key=lambda e: -e.self_device_time_total)
        total = sum(e.self_device_time_total for e in evs) or 1.0
        rows = []
        for e in evs[:topk]:
            rows.append({
                "kernel": e.key[:160],
                "count_per_iter": e.count / iters,
                "self_us_per_iter": round(e.self_device_time_total / iters, 1),
                "self_us_per_call": round(
                    e.self_device_time_total / max(e.count, 1), 1),
                "pct": round(100.0 * e.self_device_time_total / total, 2),
            })
        return {"iters": iters, "total_self_us_per_iter": round(total / iters, 1),
                "top": rows}

    for phase, fn, arg in (("fwd", call_fwd, io), ("fwdbwd", call_fwdbwd, io_g)):
        try:
            result[phase] = timed(fn, arg)
        except Exception as e:  # record and continue (spec: bwd may throw)
            result[phase] = {"error": f"{type(e).__name__}: {e}"}
    for phase, fn, arg in (("kernels_fwd", call_fwd, io),
                           ("kernels_fwdbwd", call_fwdbwd, io_g)):
        try:
            result[phase] = profile_topk(fn, arg)
        except Exception as e:
            result[phase] = {"error": f"{type(e).__name__}: {e}"}

    # bwd kernel attribution: per-kernel fwdbwd-minus-fwd self time
    try:
        kf = result["kernels_fwd"]["top"]
        kb = result["kernels_fwdbwd"]["top"]
        fmap = {r["kernel"]: r["self_us_per_iter"] for r in kf}
        attr = []
        for r in kb:
            d = round(r["self_us_per_iter"] - fmap.get(r["kernel"], 0.0), 1)
            if d > 0:
                attr.append({"kernel": r["kernel"], "bwd_self_us_per_iter": d,
                             "pct_of_fwdbwd": r["pct"]})
        attr.sort(key=lambda x: -x["bwd_self_us_per_iter"])
        result["kernels_bwd_attribution"] = attr[:10]
    except Exception as e:
        result["kernels_bwd_attribution"] = {"error": f"{type(e).__name__}: {e}"}

    print(JSON_MARK + json.dumps(result))


# --------------------------------------------------------------------------- #
# parent
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "evidence/G5/fla_baseline.json"))
    ap.add_argument("--shapes", default="B1T2048,B1T8192,B4T2048")
    ap.add_argument("--modes", default="triton,default")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-mode")
    ap.add_argument("--worker-shape")
    args = ap.parse_args()

    if args.worker:
        assert args.worker_mode and args.worker_shape
        run_worker(args.worker_mode, args.worker_shape, args.warmup, args.reps)
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged = {"meta": {
        "what": "fla chunk_gated_delta_rule baselines (G5-A)",
        "gpu": smi_snapshot(),
        "shared_gpu_note": ("opportunistic window; co-job pid 3411456 "
                            "~17.9GB resident -- absolute times noisy by design"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": args.warmup, "reps": args.reps,
        "shapes": args.shapes, "modes": args.modes,
    }, "fla": {}}

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "4"

    for mode in args.modes.split(","):
        merged["fla"][mode] = {}
        for shape in args.shapes.split(","):
            print(f"[gdn_bench_flaptx] mode={mode} shape={shape} ...",
                  flush=True)
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--worker", "--worker-mode", mode, "--worker-shape", shape,
                   "--warmup", str(args.warmup), "--reps", str(args.reps)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=3600, env=env)
                line = next((l for l in r.stdout.splitlines()
                             if l.startswith(JSON_MARK)), None)
                if line is None:
                    raise RuntimeError(
                        f"no result; rc={r.returncode}; stderr tail: "
                        + "\n".join(r.stderr.splitlines()[-8:]))
                merged["fla"][mode][shape] = json.loads(line[len(JSON_MARK):])
                print(f"[gdn_bench_flaptx] ok "
                      f"(fwd {merged['fla'][mode][shape].get('fwd', {}).get('median_ms', '?')} ms)",
                      flush=True)
            except Exception as e:
                merged["fla"][mode][shape] = {"error": f"{type(e).__name__}: {e}"}
                print(f"[gdn_bench_flaptx] FAILED: {e}", flush=True)

    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[gdn_bench_flaptx] wrote {args.out}")


if __name__ == "__main__":
    main()
