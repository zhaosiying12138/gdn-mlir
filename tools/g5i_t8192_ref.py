#!/usr/bin/env python3
"""g5i_t8192_ref.py -- G5-I: T8192 fp64 JAX reference (one-off, CPU).

Same pipeline as the G5-A refs: shared recipe inputs
(tools/gdn_bench_common.gen_case, seed 42) with q/k/v bf16-ROUNDED, fed
to the JAX fp64 chunked spec (tools/gdn_ref_jax.gdn_chunk).  Writes:

  evidence/G5/refs/bench_B1T8192H16_inputs.npz   (q_bf16..h0, do/dht-free)
  evidence/G5/refs/bench_B1T8192H16_ref.npz      (ref__o_f32, ref__ht_f32)

T8192 fp64 chunked on CPU runs in minutes; acceptable once.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

REPO = os.path.expanduser("~/codebase/gdn-mlir")
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np
import torch  # noqa: F401
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import gdn_bench_common as bc  # noqa: E402
from gdn_ref_jax import gdn_chunk  # noqa: E402

B, T, H, K, V = 1, 8192, 16, 128, 128
case = bc.gen_case(B, T, H, H, K, V, seed=bc.SEED)


def bf16r(x):
    u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u32.astype(np.uint64) + 0x7FFF + ((u32 >> 16) & 1)) >> 16
    return (r.astype(np.uint32) << 16).view(np.float32)


q = bf16r(case["q"].detach().cpu().float().numpy())
k = bf16r(case["k"].detach().cpu().float().numpy())
v = bf16r(case["v"].detach().cpu().float().numpy())
beta = case["beta"].detach().cpu().float().numpy()
g = case["g"].detach().cpu().float().numpy()
h0 = case["h0"].detach().cpu().float().numpy()

out = os.path.join(REPO, "evidence/G5/refs")
os.makedirs(out, exist_ok=True)
np.savez(os.path.join(out, "bench_B1T8192H16_inputs.npz"),
         q_bf16=q, k_bf16=k, v_bf16=v, beta=beta, g=g, h0=h0)
print("[g5i_t8192_ref] inputs written; running fp64 chunked spec ...",
      flush=True)

o, ht = gdn_chunk(jnp.array(q), jnp.array(k), jnp.array(v),
                  jnp.array(beta), jnp.array(g), h0=jnp.array(h0),
                  chunk_size=64)
o = np.asarray(o, dtype=np.float64)
ht = np.asarray(ht, dtype=np.float64)
np.savez(os.path.join(out, "bench_B1T8192H16_ref.npz"),
         ref__o_f64=o.astype(np.float32), ref__ht_f64=ht.astype(np.float32))
print(f"[g5i_t8192_ref] done: o {o.shape} ht {ht.shape}; "
      f"|o|max {np.abs(o).max():.3f} |ht|max {np.abs(ht).max():.3f}")
