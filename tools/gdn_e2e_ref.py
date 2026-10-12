#!/usr/bin/env python3
"""GDN-MLIR G5-A -- JAX fp64 reference data at the G5 bench shapes (+ fla dump).

For each REF case (small_B1T512H4 fast dev loop, bench_B1T2048H16 bench shape;
T8192 fp64 skipped by design -- see baseline_report.md):
  1. generate inputs ONCE with the shared bench recipe
     (tools/gdn_bench_common.py = fla test_chunk recipe, seed 42) on CPU;
     q/k/v/do are baked to bf16 (exactly what the GPU ops consume),
     beta/g/h0/dht stay fp32;
  2. run tools/gdn_ref_jax.py gdn_chunk (chunk_size=64) + gdn_chunk_vjp in
     fp64 on the upcast bf16 values -> reference {o, ht, dq, dk, dv, dbeta,
     dg, dh0}, saved as f64 / f32 / bf16-as-f32 casts;
  3. on the small case, cross-check the chunked spec against the sequential
     spec (gdn_seq / gdn_seq_vjp) at fp64 -- spec self-consistency;
  4. run fla's chunk_gated_delta_rule on GPU (pure-Triton口径: FLA_FLASH_QLA=0
     set via os.environ BEFORE importing fla) on the SAME tensors, fwd
     (no_grad) + autograd.grad bwd, and dump o/ht/grads for direct
     fla-vs-JAX tolerance calibration at bench shapes.

Outputs (per case):
  evidence/G5/refs/<name>_inputs.npz   -- the exact bench inputs
  evidence/G5/refs/<name>_ref.npz      -- JAX f64/f32/bf16 refs + fla outputs
  evidence/G5/refs/tolerances.json     -- max-err tables (fla vs JAX fp64)

Usage (conda env gdn, CPU JAX + one short GPU window):
  python tools/gdn_e2e_ref.py --cases small_B1T512H4,bench_B1T2048H16 \
      --outdir evidence/G5/refs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser("~/codebase/gdn-mlir")
sys.path.insert(0, HERE)

# pure-Triton口径 BEFORE fla import (gdn_bench_common imports no fla)
os.environ.setdefault("FLA_FLASH_QLA", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gdn_bench_common as C  # noqa: E402
import gdn_ref_jax as R  # noqa: E402  (tools/gdn_ref_jax.py; enables x64)

OUTS = ("o", "ht", "dq", "dk", "dv", "dbeta", "dg", "dh0")


def t64(t_bf16_or_f32: torch.Tensor) -> "R.jnp.ndarray":  # noqa: F821
    return R.jnp.asarray(t_bf16_or_f32.to(torch.float64).numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(c["name"] for c in C.REF_CASES))
    ap.add_argument("--outdir", default=os.path.join(REPO, "evidence/G5/refs"))
    ap.add_argument("--chunk-size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cases = {c["name"]: c for c in C.REF_CASES}
    tol = {"meta": {
        "what": "fla(Triton) vs JAX fp64 tolerance calibration (G5-A)",
        "recipe": "tools/gdn_bench_common.py gen_case (fla test_chunk, seed 42)",
        "jax": R.jax.__version__,
        "chunk_size": args.chunk_size,
        "fla_dispatch": "FLA_FLASH_QLA=0 (pure Triton)",
        "note": "T8192 fp64 refs skipped by design; numerics interpolate via "
                "fla cross-checks at T512/T2048 (same kernels per chunk)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, "cases": {}}

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # after env

    for name in args.cases.split(","):
        cs = cases[name]
        B, T, H, HV, K, V = (cs["B"], cs["T"], cs["H"], cs["HV"],
                             cs["K"], cs["V"])
        scale = K ** -0.5
        print(f"[gdn_e2e_ref] case {name}: B{B} T{T} H{H} KV{K} chunk"
              f"={args.chunk_size}", flush=True)

        # ---- 1. inputs (shared recipe; bake bf16 exactly as the GPU sees) -- #
        case = C.gen_case(B, T, H, HV, K, V, seed=C.SEED)
        baked = dict(
            q=case["q"].to(torch.bfloat16), k=case["k"].to(torch.bfloat16),
            v=case["v"].to(torch.bfloat16),
            beta=case["beta"].to(torch.float32), g=case["g"].to(torch.float32),
            h0=case["h0"].to(torch.float32),
            do=case["do"].to(torch.bfloat16), dht=case["dht"].to(torch.float32))
        C.save_case_npz(baked, os.path.join(args.outdir, f"{name}_inputs.npz"))

        # ---- 2. JAX fp64 chunked fwd + vjp bwd ----------------------------- #
        q64, k64, v64 = t64(baked["q"]), t64(baked["k"]), t64(baked["v"])
        beta64, g64 = t64(baked["beta"]), t64(baked["g"])
        do64, dht64 = t64(baked["do"]), t64(baked["dht"])
        t0 = time.perf_counter()
        ref = dict(zip(OUTS, R.gdn_chunk_vjp(
            q64, k64, v64, beta64, g64, scale=scale,
            chunk_size=args.chunk_size, do=do64, dht=dht64)))
        jax_s = round(time.perf_counter() - t0, 1)
        print(f"[gdn_e2e_ref] jax chunk+vjp fp64 done in {jax_s}s", flush=True)

        # ---- 3. small-case spec cross-check (chunk vs sequential) ---------- #
        cross = None
        if T <= 1024:
            seq = dict(zip(OUTS, R.gdn_seq_vjp(
                q64, k64, v64, beta64, g64, scale=scale, do=do64, dht=dht64)))
            cross = {k: float(np.max(np.abs(np.asarray(ref[k]) - np.asarray(seq[k]))))
                     for k in OUTS}
            worst = max(cross.values())
            print(f"[gdn_e2e_ref] chunk-vs-seq fp64 max|d| = {worst:.3e}",
                  flush=True)
            assert worst < 1e-8, f"spec self-check failed: {worst}"

        # ---- 4. fla GPU outputs on the SAME tensors ------------------------ #
        dev = "cuda"
        io = {key: baked[key].to(dev) for key in baked}
        io_g = {key: (io[key].detach().clone().requires_grad_(True)
                      if key in ("q", "k", "v", "beta", "g", "h0") else io[key])
                for key in io}
        # fwd under grad mode: the SAME call provides o/ht for the dump and
        # the graph for autograd.grad (values identical to the no_grad fwd)
        o_fl, ht_fl = chunk_gated_delta_rule(
            q=io_g["q"], k=io_g["k"], v=io_g["v"], g=io_g["g"],
            beta=io_g["beta"], scale=scale, initial_state=io_g["h0"],
            output_final_state=True)
        grads = torch.autograd.grad(
            (o_fl * io_g["do"]).sum() + (ht_fl * io_g["dht"]).sum(),
            (io_g["q"], io_g["k"], io_g["v"], io_g["beta"], io_g["g"], io_g["h0"]))
        fl = {"o": o_fl, "ht": ht_fl, "dq": grads[0], "dk": grads[1],
              "dv": grads[2], "dbeta": grads[3], "dg": grads[4], "dh0": grads[5]}

        # ---- 5. tolerance calibration + npz dump --------------------------- #
        arrs, tolrows = {}, {}
        for key in OUTS:
            a64 = np.asarray(ref[key])
            a32 = a64.astype(np.float32)
            abf = torch.from_numpy(a32).to(torch.bfloat16).to(torch.float32).numpy()
            arrs[f"ref__{key}_f64"] = a64
            arrs[f"ref__{key}_f32"] = a32
            arrs[f"ref__{key}_bf16"] = abf
            fv = fl[key].detach().to(torch.float64).cpu().numpy()
            arrs[f"fla__{key}"] = fv
            max_abs = float(np.max(np.abs(fv - a64)))
            scale_n = float(np.max(np.abs(a64)))
            tolrows[key] = {
                "fla_dtype": str(fl[key].dtype),
                "max_abs_err": max_abs,
                "ref_max_abs": scale_n,
                "rel_err": (max_abs / scale_n) if scale_n > 0 else None,
                "bf16_out_floor": float(np.max(np.abs(abf - a64))),
            }
            print(f"[gdn_e2e_ref]   {key:6s} max_abs={max_abs:.3e} "
                  f"rel={tolrows[key]['rel_err']:.3e} "
                  f"bf16_floor={tolrows[key]['bf16_out_floor']:.3e}", flush=True)

        np.savez_compressed(os.path.join(args.outdir, f"{name}_ref.npz"), **arrs)
        tol["cases"][name] = {
            "shape": {k: cs[k] for k in ("B", "T", "H", "HV", "K", "V")},
            "scale": scale, "jax_seconds": jax_s,
            "chunk_vs_seq_fp64_maxdiff": cross, "fla_vs_jax": tolrows,
        }

    with open(os.path.join(args.outdir, "tolerances.json"), "w") as f:
        json.dump(tol, f, indent=2)
    print(f"[gdn_e2e_ref] wrote {args.outdir}/tolerances.json")


if __name__ == "__main__":
    main()
