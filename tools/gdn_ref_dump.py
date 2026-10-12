"""
GDN-MLIR G1-A -- dump reference artifacts for the MLIR workstream (B).

For B=2, T=128, H=2, K=16, V=16 (fp64 compute) writes, per config, an npz with
  inputs : q, k, v, beta, g, h0, do, dht        (do/dht are the seeded
             VJP cotangents = gradient seeds)
  outputs: o, ht
  grads  : dq, dk, dv, dbeta, dg, dh0           (VJP of L = <do,o> + <dht,ht>)
Primary keys are fp32 casts (the comparison target for the MLIR fp32 path);
identical fp64 arrays are stored under "<name>_fp64" for tighter comparisons.

Files (~/codebase/gdn-mlir/evidence/G1/):
  ref_jax_T128_C64.npz     chunk_size=64, h0=None (zeros)
  ref_jax_T128_C32.npz     chunk_size=32, h0=None (zeros)
  ref_jax_T128_C64_h0.npz  chunk_size=64, h0=seeded nonzero

All three share the SAME seeded q,k,v,beta,g,do,dht (h0 differs).

Run:  python tools/gdn_ref_dump.py
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import json
import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdn_ref_jax import gdn_chunk, gdn_chunk_vjp, gdn_seq, make_gdn_inputs  # noqa: E402

B, T, H, K, V = 2, 128, 2, 16, 16
SCALE = float(K ** -0.5)  # 0.25 -- naive.py default (q pre-scaled by this)
OUT_DIR = Path(__file__).resolve().parent.parent / "evidence" / "G1"

CONFIGS = [
    ("ref_jax_T128_C64.npz", 64, False),
    ("ref_jax_T128_C32.npz", 32, False),
    ("ref_jax_T128_C64_h0.npz", 64, True),
]

key = jax.random.PRNGKey(20261008)
inp_none = make_gdn_inputs(key, B, T, H, K, V, with_h0=False, dtype=jnp.float64)
inp_h0 = make_gdn_inputs(key, B, T, H, K, V, with_h0=True, dtype=jnp.float64)
# identical q..dht across files; only h0 differs
for nm in ("q", "k", "v", "beta", "g", "do", "dht"):
    assert float(jnp.max(jnp.abs(inp_none[nm] - inp_h0[nm]))) == 0.0

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'FILE':<28} {'ARRAYS':>7} {'fp32 max|x|':>12}  notes")
print("-" * 78)
for fname, C, with_h0 in CONFIGS:
    inp = inp_h0 if with_h0 else inp_none
    o, ht, dq, dk, dv, dbeta, dg, dh0 = gdn_chunk_vjp(
        inp["q"], inp["k"], inp["v"], inp["beta"], inp["g"], h0=inp["h0"],
        chunk_size=C, do=inp["do"], dht=inp["dht"])

    # sanity: chunked agrees with sequential in fp64 on the dumped data
    o_s, ht_s = gdn_seq(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g"],
                        h0=inp["h0"])
    dseq = float(max(jnp.max(jnp.abs(o - o_s)), jnp.max(jnp.abs(ht - ht_s))))

    h0 = (inp["h0"] if inp["h0"] is not None
          else np.zeros((B, H, K, V), dtype=np.float64))
    arrays = dict(q=inp["q"], k=inp["k"], v=inp["v"], beta=inp["beta"],
                  g=inp["g"], h0=h0, do=inp["do"], dht=inp["dht"],
                  o=o, ht=ht, dq=dq, dk=dk, dv=dv, dbeta=dbeta, dg=dg, dh0=dh0)
    payload = {}
    for nm, x in arrays.items():
        x = np.asarray(x, dtype=np.float64)
        payload[nm] = x.astype(np.float32)  # primary: fp32 comparison target
        payload[nm + "_fp64"] = x
    payload["config"] = np.array(json.dumps({
        "B": B, "T": T, "H": H, "K": K, "V": V, "chunk_size": C,
        "scale": SCALE, "compute_dtype": "float64", "storage_dtype": "float32",
        "loss": "L = sum(do*o) + sum(dht*ht);  dq..dh0 = dL/d(q,k,v,beta,g,h0)",
        "conventions": {
            "g": "natural-log per-token decay, applied as exp(g)",
            "beta": "raw (sigmoid already applied by caller)",
            "q": "caller l2-normalized; op pre-scales q by scale=K**-0.5",
            "state": "h[b,h,k,v]; o[b,t,h,v]",
        },
        "h0": "zeros" if not with_h0 else "seeded nonzero",
        "generator": "tools/gdn_ref_dump.py, jax.random.PRNGKey(20261008)",
    }))
    path = OUT_DIR / fname
    np.savez(path, **payload)
    f32max = max(float(np.max(np.abs(payload[nm]))) for nm in arrays)
    print(f"{fname:<28} {len(payload):>7} {f32max:>12.4e}  "
          f"chunk-vs-seq(fp64)={dseq:.2e}")
print("-" * 78)
print(f"written to {OUT_DIR}")
