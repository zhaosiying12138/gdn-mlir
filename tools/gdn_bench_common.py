#!/usr/bin/env python3
"""GDN-MLIR G5-A -- shared bench-case definitions and input generation.

Single source of truth for the G5 showdown baselines.  The fla Triton harness
(tools/gdn_bench_flaptx.py), the FlashQLA harness (tools/gdn_bench_flashqlaptx.py)
and the JAX fp64 reference generator (tools/gdn_e2e_ref.py) all build inputs
through this module, so every timing and every tolerance number is measured on
bit-identical tensors.

Input recipe = fla/tests/ops/test_gdn.py::test_chunk (H == HV case), seed 42:

    torch.manual_seed(42)
    q = F.normalize(torch.rand(B, T, H, K), p=2, dim=-1)      # fp32, THEN cast
    k = F.normalize(torch.rand(B, T, H, K), p=2, dim=-1)
    v = torch.rand(B, T, H, V)
    beta = torch.rand(B, T, H).sigmoid()                      # fp32, in (0, 1)
    g = F.logsigmoid(torch.rand(B, T, H))                     # fp32, negative log-decay
    h0 = torch.zeros(B, H, K, V)                              # fp32 (test_chunk)
    do = torch.randn_like(v)                                  # cotangent for o
    dht = torch.randn_like(h0)                                # cotangent for ht

scale is always K ** -0.5 (passed explicitly; = the op default), and
output_final_state=True.  q/k are L2-normalized OUTSIDE the op (caller side),
i.e. use_qk_l2norm_in_kernel=False, exactly like test_chunk.

Generation happens on CPU in fp32 (q/k normalized in fp32 *before* the bf16
cast) so the fp64 JAX reference can consume the exact bf16 values the GPU ops
see (bf16 -> fp64 upcast is exact).

This module must stay importable in BOTH conda envs (`gdn`: torch+fla+jax)
and `gdn-tl`: torch+tilelang+flash_qla) -> import torch only, never fla.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# shapes
# --------------------------------------------------------------------------- #
# Main bench shapes (docs/PLAN.md G5): B in {1, 4} first (8 later),
# T in {2048, 8192}, H = HV = 16, K = V = 128, bf16 io.
H_BENCH = 16
KV_BENCH = 128
DTYPE_BENCH = torch.bfloat16
SEED = 42

BENCH_SHAPES = [
    {"name": "B1T2048", "B": 1, "T": 2048, "H": H_BENCH, "HV": H_BENCH,
     "K": KV_BENCH, "V": KV_BENCH},
    {"name": "B1T8192", "B": 1, "T": 8192, "H": H_BENCH, "HV": H_BENCH,
     "K": KV_BENCH, "V": KV_BENCH},
    {"name": "B4T2048", "B": 4, "T": 2048, "H": H_BENCH, "HV": H_BENCH,
     "K": KV_BENCH, "V": KV_BENCH},
]

# Reference-data cases: the big bench shape whose fp64 chunked JAX spec is
# still tractable on CPU, plus a small fast-dev-loop case.
# (T=8192 fp64 is intentionally skipped -- numerics interpolate via the
#  fla cross-checks at these two resolutions; see evidence/G5/baseline_report.md)
REF_CASES = [
    {"name": "small_B1T512H4", "B": 1, "T": 512, "H": 4, "HV": 4,
     "K": KV_BENCH, "V": KV_BENCH},
    {"name": "bench_B1T2048H16", "B": 1, "T": 2048, "H": H_BENCH, "HV": H_BENCH,
     "K": KV_BENCH, "V": KV_BENCH},
    # RC1 (regime-change r1): the T16384 ladder cell's fp64 ref (new NC=256
    # class; JAX CPU ~5 min; fla section needs a scratch TRITON_CACHE_DIR).
    {"name": "rc1_B1T16384H16", "B": 1, "T": 16384, "H": H_BENCH,
     "HV": H_BENCH, "K": KV_BENCH, "V": KV_BENCH},
]


def bench_shape_by_name(name: str) -> dict:
    for s in BENCH_SHAPES:
        if s["name"] == name:
            return s
    raise KeyError(f"unknown bench shape {name!r}; have {[s['name'] for s in BENCH_SHAPES]}")


# --------------------------------------------------------------------------- #
# input generation (fla test_chunk recipe, CPU fp32, seeded)
# --------------------------------------------------------------------------- #
def gen_case(B: int, T: int, H: int, HV: int, K: int, V: int,
             seed: int = SEED) -> dict:
    """Generate one bench case on CPU in fp32 (fla test_chunk recipe).

    Returns fp32 CPU tensors; q/k are ALREADY l2-normalized (caller-side
    normalization, use_qk_l2norm_in_kernel=False).  Cast q/k/v/do to the io
    dtype only at the very end (to_device_bench / bake recipe below).
    """
    torch.manual_seed(seed)
    q = F.normalize(torch.rand(B, T, H, K, dtype=torch.float32), p=2, dim=-1)
    k = F.normalize(torch.rand(B, T, H, K, dtype=torch.float32), p=2, dim=-1)
    v = torch.rand(B, T, HV, V, dtype=torch.float32)
    beta = torch.rand(B, T, HV, dtype=torch.float32).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32))
    h0 = torch.zeros(B, HV, K, V, dtype=torch.float32)
    do = torch.randn(B, T, HV, V, dtype=torch.float32)
    dht = torch.randn(B, HV, K, V, dtype=torch.float32)
    return dict(q=q, k=k, v=v, beta=beta, g=g, h0=h0, do=do, dht=dht)


def to_device_bench(case: dict, device: str = "cuda",
                    dtype: torch.dtype = DTYPE_BENCH,
                    requires_grad: bool = False) -> dict:
    """Move a gen_case dict to device in the bench io convention.

    q/k/v/do -> io dtype (bf16); beta/g/h0/dht stay fp32 (exactly what
    test_chunk feeds the op: fp32 gates, fp32 initial state/cotangent).
    """
    out = dict(
        q=case["q"].to(device=device, dtype=dtype),
        k=case["k"].to(device=device, dtype=dtype),
        v=case["v"].to(device=device, dtype=dtype),
        beta=case["beta"].to(device=device),
        g=case["g"].to(device=device),
        h0=case["h0"].to(device=device),
        do=case["do"].to(device=device, dtype=dtype),
        dht=case["dht"].to(device=device),
    )
    if requires_grad:
        for key in ("q", "k", "v", "beta", "g", "h0"):
            out[key] = out[key].detach().clone().requires_grad_(True)
    return out


# --------------------------------------------------------------------------- #
# npz persistence (bf16 stored exactly as its fp32 upcast; numpy has no bf16)
# --------------------------------------------------------------------------- #
def _to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


def save_case_npz(case: dict, path: str) -> None:
    """Save a (possibly bf16-cast) case dict; dtypes preserved via naming."""
    arrs = {}
    for key, t in case.items():
        arrs[key if t.dtype == torch.float32 else f"{key}_bf16"] = _to_f32(t)
    np.savez_compressed(path, **arrs)


def load_case_npz(path: str) -> dict:
    z = np.load(path)
    out = {}
    for key in z.files:
        if key.endswith("_bf16"):
            out[key[:-5]] = torch.from_numpy(z[key]).to(torch.bfloat16)
        else:
            out[key] = torch.from_numpy(z[key])
    return out
