"""
GDN-MLIR G1-A -- self-consistency suite for the JAX GDN spec (tools/gdn_ref_jax.py).

CPU-only. fp64 (chunked vs sequential, VJP cross-check, finite differences)
plus a torch cross-check against fla's naive_recurrent_gated_delta_rule /
naive_chunk_gated_delta_rule in fp32 (the oracle hard-casts to fp32 internally).

Run:  python tools/gdn_ref_check.py     (exit code 0 iff every row PASSes)
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import functools
import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdn_ref_jax import (  # noqa: E402
    gdn_chunk,
    gdn_chunk_vjp,
    gdn_seq,
    gdn_seq_vjp,
    make_gdn_inputs,
)

ROWS = []


def check(name, value, tol):
    ok = bool(value < tol)
    ROWS.append((name, value, tol, ok))
    return ok


B, H, K, V = 2, 2, 16, 16
COMBOS = [(96, 32), (96, 64), (128, 32), (128, 64)]  # (T, chunk_size)
# (96,64) exercises the zero-padded tail chunk; the others are exact multiples.

master = jax.random.PRNGKey(12345)
inputs = {}
for T, C in COMBOS:
    for h0flag in (False, True):
        key, master = jax.random.split(master)
        inputs[(T, C, h0flag)] = make_gdn_inputs(
            key, B, T, H, K, V, with_h0=h0flag, dtype=jnp.float64
        )

# --------------------------------------------------------------------------- #
# 1) chunked vs sequential: o and final_state, max abs diff < 1e-10 (fp64)
# --------------------------------------------------------------------------- #
for (T, C, h0flag), inp in inputs.items():
    o_s, ht_s = gdn_seq(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g"],
                        h0=inp["h0"])
    o_c, ht_c = gdn_chunk(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g"],
                          h0=inp["h0"], chunk_size=C)
    tag = f"T={T} C={C} h0={'y' if h0flag else 'n'}"
    check(f"chunk-vs-seq o    {tag}",
          float(jnp.max(jnp.abs(o_c - o_s))), 1e-10)
    check(f"chunk-vs-seq ht   {tag}",
          float(jnp.max(jnp.abs(ht_c - ht_s))), 1e-10)

# --------------------------------------------------------------------------- #
# 2) VJP grads: chunked vs sequential, all grads max abs diff < 1e-8 (fp64)
# --------------------------------------------------------------------------- #
for (T, C, h0flag), inp in inputs.items():
    res_s = gdn_seq_vjp(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g"],
                        h0=inp["h0"], do=inp["do"], dht=inp["dht"])
    res_c = gdn_chunk_vjp(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g"],
                          h0=inp["h0"], chunk_size=C, do=inp["do"],
                          dht=inp["dht"])
    worst = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(res_s[2:], res_c[2:]))
    check(f"vjp chunk-vs-seq T={T} C={C} h0={'y' if h0flag else 'n'}",
          worst, 1e-8)

# --------------------------------------------------------------------------- #
# 3) finite differences on 6 random coordinates (one per tensor), < 1e-5
#    loss = sum(do * o) + sum(dht * ht)
# --------------------------------------------------------------------------- #
def fd_check(tag, fwd, inp, chunk_size=None, eps=1e-6, seed=7):
    arrays = {n: np.asarray(inp[n], dtype=np.float64)
              for n in ("q", "k", "v", "beta", "g")}
    arrays["h0"] = (np.zeros((B, H, K, V), dtype=np.float64)
                    if inp["h0"] is None
                    else np.asarray(inp["h0"], dtype=np.float64))
    do, dht = jnp.asarray(inp["do"]), jnp.asarray(inp["dht"])
    kw = {} if chunk_size is None else dict(chunk_size=chunk_size)

    def loss(a):
        o, ht = fwd(a["q"], a["k"], a["v"], a["beta"], a["g"], h0=a["h0"], **kw)
        return jnp.sum(a["do"] * o) + jnp.sum(a["dht"] * ht)

    def scalar(a):
        a = dict(a, do=do, dht=dht)
        return float(loss(a))

    def loss_args(q, k, v, beta, g, h0):
        return loss(dict(q=q, k=k, v=v, beta=beta, g=g, h0=h0, do=do, dht=dht))

    names = ("q", "k", "v", "beta", "g", "h0")
    grads = jax.grad(loss_args, argnums=tuple(range(6)))(
        *[jnp.asarray(arrays[n]) for n in names])

    rng = np.random.default_rng(seed)
    worst = 0.0
    for gi, n in enumerate(names):
        idx = tuple(int(i) for i in rng.integers(0, arrays[n].shape))
        lp, lm = dict(arrays), dict(arrays)
        lp[n] = arrays[n].copy(); lp[n][idx] += eps
        lm[n] = arrays[n].copy(); lm[n][idx] -= eps
        fd = (scalar({m: jnp.asarray(x) for m, x in lp.items()})
              - scalar({m: jnp.asarray(x) for m, x in lm.items()})) / (2 * eps)
        ad = float(grads[gi][idx])
        worst = max(worst, abs(fd - ad))
    check(f"finite-diff vs vjp {tag} (6 coords)", worst, 1e-5)


fd_inp = inputs[(96, 64, True)]  # tail chunk + h0 present
fd_check("seq   T=96     ", gdn_seq, fd_inp)
fd_check("chunk T=96 C=64", gdn_chunk, fd_inp, chunk_size=64)

# --------------------------------------------------------------------------- #
# 4) jit parity (functions must be jittable; scale/C bound via functools.partial)
# --------------------------------------------------------------------------- #
inp = inputs[(96, 64, False)]
o_e, ht_e = gdn_seq(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g"], h0=None)
o_j, ht_j = jax.jit(gdn_seq)(inp["q"], inp["k"], inp["v"], inp["beta"],
                             inp["g"], h0=None)
check("jit parity seq   T=96",
      float(max(jnp.max(jnp.abs(o_j - o_e)), jnp.max(jnp.abs(ht_j - ht_e)))),
      1e-12)
o_ec, ht_ec = gdn_chunk(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g"],
                        h0=None, chunk_size=64)
jit_chunk = jax.jit(functools.partial(gdn_chunk, chunk_size=64))
o_jc, ht_jc = jit_chunk(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g"],
                        h0=None)
check("jit parity chunk T=96 C=64",
      float(max(jnp.max(jnp.abs(o_jc - o_ec)), jnp.max(jnp.abs(ht_jc - ht_ec)))),
      1e-12)

# --------------------------------------------------------------------------- #
# 5) torch cross-check vs fla naive oracle (fp32: oracle hard-casts to fp32)
# --------------------------------------------------------------------------- #
import torch  # noqa: E402

torch.set_num_threads(4)
from fla.ops.gated_delta_rule.naive import (  # noqa: E402
    naive_chunk_gated_delta_rule,
    naive_recurrent_gated_delta_rule,
)


def rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


for (T, C, h0flag) in [(96, 64, False), (96, 64, True), (128, 32, False),
                       (128, 64, True)]:
    inp = inputs[(T, C, h0flag)]
    # JAX in fp32, matching the oracle's internal dtype
    f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)  # noqa: E731
    q32, k32, v32 = f32(inp["q"]), f32(inp["k"]), f32(inp["v"])
    b32, g32 = f32(inp["beta"]), f32(inp["g"])
    h032 = None if inp["h0"] is None else f32(inp["h0"])
    o_s, ht_s = gdn_seq(q32, k32, v32, b32, g32, h0=h032)
    o_c, ht_c = gdn_chunk(q32, k32, v32, b32, g32, h0=h032, chunk_size=C)

    tq = torch.from_numpy(np.asarray(inp["q"], dtype=np.float32))
    tk = torch.from_numpy(np.asarray(inp["k"], dtype=np.float32))
    tv = torch.from_numpy(np.asarray(inp["v"], dtype=np.float32))
    tb = torch.from_numpy(np.asarray(inp["beta"], dtype=np.float32))
    tg = torch.from_numpy(np.asarray(inp["g"], dtype=np.float32))
    th0 = None if inp["h0"] is None else torch.from_numpy(
        np.asarray(inp["h0"], dtype=np.float32))
    # recurrent: signature (q, k, v, beta, g, ...)
    o_tr, ht_tr = naive_recurrent_gated_delta_rule(
        tq, tk, tv, tb, tg, initial_state=th0, output_final_state=True)
    # chunked: signature (q, k, v, g, beta, ...)  <- note the swap!
    o_tc, ht_tc = naive_chunk_gated_delta_rule(
        tq, tk, tv, tg, tb, chunk_size=C, initial_state=th0,
        output_final_state=True)

    tag = f"T={T} C={C} h0={'y' if h0flag else 'n'}"
    check(f"torch-seq   vs jax-seq   o  {tag} (rel)",
          rel_err(o_s, o_tr.numpy()), 1e-4)
    check(f"torch-seq   vs jax-seq   ht {tag} (rel)",
          rel_err(ht_s, ht_tr.numpy()), 1e-4)
    check(f"torch-chunk vs jax-chunk o  {tag} (rel)",
          rel_err(o_c, o_tc.numpy()), 1e-4)
    check(f"torch-chunk vs jax-chunk ht {tag} (rel)",
          rel_err(ht_c, ht_tc.numpy()), 1e-4)

# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
print()
hdr = f"{'CHECK':<56} {'VALUE':>12} {'TOL':>8}  RESULT"
print(hdr)
print("-" * len(hdr))
fails = 0
for name, value, tol, ok in ROWS:
    val_s = f"{value:.3e}"
    tol_s = f"{tol:.0e}"
    print(f"{name:<56} {val_s:>12} {tol_s:>8}  {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1
print("-" * len(hdr))
print(f"TOTAL: {len(ROWS)} checks, {fails} failed")
if fails:
    sys.exit(1)
print("ALL PASS")
