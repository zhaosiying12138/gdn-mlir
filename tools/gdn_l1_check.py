#!/usr/bin/env python3
"""gdn_l1_check.py -- numeric self-check driver for the L1 chunked-GDN MLIR.

For each test config this script:
  1. generates seeded input data (same code path as gdn_l1_gen.py),
  2. emits the .mlir module via tools/gdn_l1_gen.py,
  3. lowers + executes it on CPU via scripts/run_l1.sh (mlir-opt -> mlir-runner),
  4. parses the printed memrefs (o, ht) from the runner stdout,
  5. compares against an embedded NUMPY reference of the SAME chunked
     algorithm (ported from fla naive_chunk_gated_delta_rule, log2-domain g),
     which is itself cross-checked against the sequential recurrence
     (naive_recurrent_gated_delta_rule port) to validate the chunked math.

PASS gate: max abs diff < 1e-3 for both o and final state ht.
"""

import ast
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdn_l1_gen as gen  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS_ATOL = 1e-3


# ----------------------------------------------------------------------------
# numpy references (mirror of the MLIR math; see gdn_l1_gen.py conventions)
# ----------------------------------------------------------------------------

def _pad(x, Tp):
    pw = [(0, 0)] * x.ndim
    pw[1] = (0, Tp - x.shape[1])
    return np.pad(x, pw)


def chunked_ref(q, k, v, beta, g, h0, C):
    """Chunked forward-substitution variant, fp32, NATURAL-log g (fla/G1-A
    convention; the MLIR kernel computes exp2(g*log2 e) == exp(g)).

    Returns o [B,T,H,V] (unpadded) and ht [B,H,K,V].
    """
    f32 = np.float32
    B, T, H, K = q.shape
    V = v.shape[-1]
    Tp = ((T + C - 1) // C) * C
    q, k, v = _pad(q, Tp), _pad(k, Tp), _pad(v, Tp)
    beta, g = _pad(beta, Tp), _pad(g, Tp)
    scale = f32(1.0 / np.sqrt(K))
    o = np.zeros((B, Tp, H, V), f32)
    ht = np.zeros((B, H, K, V), f32)
    for b in range(B):
        for h in range(H):
            S = h0[b, h].astype(f32).copy()
            for n in range(Tp // C):
                sl = slice(n * C, (n + 1) * C)
                kc = k[b, sl, h]
                qc = q[b, sl, h]
                vc = v[b, sl, h]
                bc = beta[b, sl, h]
                gc = np.cumsum(g[b, sl, h]).astype(f32)  # inclusive, chunk-local
                # A strict lower, Tm = (I+A)^-1 by forward substitution
                A = np.tril(
                    bc[:, None] * (kc @ kc.T) * np.exp(gc[:, None] - gc[None, :]),
                    -1).astype(f32)
                Tm = np.zeros((C, C), f32)
                for i in range(C):
                    Tm[i, i] = f32(1.0)
                    for j in range(i):
                        s = f32(0.0)
                        for kk in range(j, i):
                            s += A[i, kk] * Tm[kk, j]
                        Tm[i, j] = -s
                u = (Tm @ (bc[:, None] * vc)).astype(f32)
                w = (Tm @ (bc[:, None] * kc * np.exp(gc)[:, None])).astype(f32)
                vn = (u - w @ S).astype(f32)
                M = np.tril((qc @ kc.T) * np.exp(gc[:, None] - gc[None, :]),
                            0).astype(f32)  # diagonal included
                oc = ((qc * np.exp(gc)[:, None]) @ S + M @ vn).astype(f32)
                o[b, sl, h] = oc * scale
                S = (S * np.exp(gc[-1])
                     + (kc * np.exp(gc[-1] - gc)[:, None]).T @ vn).astype(f32)
            ht[b, h] = S
    return o[:, :T, :, :], ht


def seq_ref(q, k, v, beta, g, h0):
    """Sequential recurrence port of naive_recurrent_gated_delta_rule
    (natural-log g): validates the chunked derivation."""
    f32 = np.float32
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = f32(1.0 / np.sqrt(K))
    qs = q * scale
    o = np.zeros((B, T, H, V), f32)
    ht = np.zeros((B, H, K, V), f32)
    for b in range(B):
        for h in range(H):
            S = h0[b, h].astype(f32).copy()
            for t in range(T):
                kt = k[b, t, h]
                vt = v[b, t, h].copy()
                S = (S * np.exp(g[b, t, h])).astype(f32)
                bv = (vt - kt @ S).astype(f32)
                bv = (bv * beta[b, t, h]).astype(f32)
                S = (S + np.outer(kt, bv)).astype(f32)
                o[b, t, h] = qs[b, t, h] @ S
            ht[b, h] = S
    return o, ht


# ----------------------------------------------------------------------------
# numpy backward twin (mirror of the MLIR @gdn_bwd; G1-B2)
# ----------------------------------------------------------------------------

def _chunk_core(qc, kc, vc, bc, gc, S):
    """Forward chunk intermediates: A, T=(I+A)^-1 (row recurrence -- the
    algebraic equivalent of the MLIR forward substitution), u, w, vn, P."""
    C = len(gc)
    A = np.tril(bc[:, None] * (kc @ kc.T) * np.exp(gc[:, None] - gc[None, :]), -1)
    T = np.eye(C)
    for i in range(1, C):
        T[i, :i] = -(A[i, :i] @ T[:i, :i])
    U = T @ (bc[:, None] * vc)
    W = T @ (bc[:, None] * kc * np.exp(gc)[:, None])
    vn = U - W @ S
    P = np.tril((qc @ kc.T) * np.exp(gc[:, None] - gc[None, :]), 0)
    return A, T, U, W, vn, P


def chunked_bwd_ref(q, k, v, beta, g, h0, do, dht, C, dtype=np.float32):
    """Chunked WY backward, numpy twin of the MLIR @gdn_bwd (natural-log g).

    Per chunk (reverse walk, dH = cotangent of the state LEAVING the chunk,
    seeded from dht): dO = do*scale; dvp = P^T dO + Bhat dH;
    dH' = e^gcl dH + (q e^gc)^T dO - W^T dvp; dW = -(dvp S^T);
    G = dvp (beta v)^T + dW (beta k e^gc)^T;
    dA = -strict_tril(T^T strict_tril(G) T^T); dAf = dA .* e^(gc_i-gc_j);
    dq/dk/dv/dbeta/dg assembled per the G1-B2 report.  Validated against the
    G1-A JAX vjp dumps to <= 1.4e-14 (fp64) on all three npz files."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    Tp = ((T + C - 1) // C) * C
    NC = Tp // C
    scale = dtype(1.0 / np.sqrt(K))
    q, k, v = _pad(q, Tp), _pad(k, Tp), _pad(v, Tp)
    beta, g, do = _pad(beta, Tp), _pad(g, Tp), _pad(do, Tp)
    dq = np.zeros((B, Tp, H, K), dtype)
    dk = np.zeros((B, Tp, H, K), dtype)
    dv = np.zeros((B, Tp, H, V), dtype)
    dbeta = np.zeros((B, Tp, H), dtype)
    dg = np.zeros((B, Tp, H), dtype)
    dh0 = np.zeros((B, H, K, V), dtype)
    m_inc = np.tril(np.ones((C, C)))
    m_str = np.tril(np.ones((C, C)), -1)

    for b in range(B):
        for h in range(H):
            # pass 1: forward recompute, storing per-chunk entry states
            S = h0[b, h].astype(dtype).copy()
            h_all = np.zeros((NC, K, V), dtype)
            for n in range(NC):
                sl = slice(n * C, (n + 1) * C)
                gc = np.cumsum(g[b, sl, h]).astype(dtype)
                h_all[n] = S
                _, _, _, W, vn, _ = _chunk_core(q[b, sl, h], k[b, sl, h],
                                                v[b, sl, h], beta[b, sl, h], gc, S)
                S = (S * np.exp(gc[-1])
                     + (k[b, sl, h] * np.exp(gc[-1] - gc)[:, None]).T @ vn).astype(dtype)
            # pass 2: reverse chunk walk
            dH = dht[b, h].astype(dtype).copy()
            for n in reversed(range(NC)):
                sl = slice(n * C, (n + 1) * C)
                qc, kc, vc, bc = (q[b, sl, h], k[b, sl, h], v[b, sl, h],
                                  beta[b, sl, h])
                gc = np.cumsum(g[b, sl, h]).astype(dtype)
                gcl = gc[-1]
                Sn = h_all[n]
                kkT = kc @ kc.T
                D = np.exp(gc[:, None] - gc[None, :])
                _, Tm, U, W, vn, P = _chunk_core(qc, kc, vc, bc, gc, Sn)
                Bhat = kc * np.exp(gcl - gc)[:, None]
                Qg = qc * np.exp(gc)[:, None]
                Kbg = bc[:, None] * kc * np.exp(gc)[:, None]
                Vb = bc[:, None] * vc

                dO = do[b, sl, h] * scale
                dPm = (dO @ vn.T) * D * m_inc
                dvp = P.T @ dO + Bhat @ dH
                dW = -(dvp @ Sn.T)
                G = dvp @ Vb.T + dW @ Kbg.T
                dAf = -(Tm.T @ (G * m_str) @ Tm.T) * D * m_str
                TW = Tm.T @ dW
                tv = Tm.T @ dvp
                dq1 = (dO * np.exp(gc)[:, None]) @ Sn.T

                dq[b, sl, h] = dq1 + dPm @ kc
                dk[b, sl, h] = (dAf @ kc) * bc[:, None] \
                    + dAf.T @ (kc * bc[:, None]) + TW * (bc * np.exp(gc))[:, None] \
                    + (vn * np.exp(gcl - gc)[:, None]) @ dH.T + dPm.T @ qc
                dv[b, sl, h] = tv * bc[:, None]
                dbeta[b, sl, h] = (dAf * kkT).sum(1) \
                    + (TW * (kc * np.exp(gc)[:, None])).sum(1) + (tv * vc).sum(1)

                # dg: suffix sums + Bhat prefix + state-decay constant.
                # NOTE: the o_intra term uses the RAW (dO vn^T) .* P -- dPm
                # already carries the decay e^(gc_i-gc_j) and multiplying by P
                # again would double-count it (the g-derivative lives in P).
                AdA = dAf * kkT * bc[:, None]
                dgr = AdA.sum(1) - AdA.sum(0)
                dgr += (TW * Kbg).sum(1)
                dgr += (dq1 * qc).sum(1)
                PdP = (dO @ vn.T) * P
                dgr += PdP.sum(1) - PdP.sum(0)
                rowB = ((vn * np.exp(gcl - gc)[:, None]) @ dH.T * kc).sum(1)
                dg5a = np.exp(gcl) * (dH * Sn).sum()
                suf = np.cumsum(dgr[::-1])[::-1]
                pre = np.concatenate([[dtype(0)], np.cumsum(rowB)[:-1]])
                dg[b, sl, h] = suf + pre + dg5a

                dH = np.exp(gcl) * dH + Qg.T @ dO - W.T @ dvp
            dh0[b, h] = dH
    return (dq[:, :T], dk[:, :T], dv[:, :T], dbeta[:, :T], dg[:, :T], dh0)


# ----------------------------------------------------------------------------
# runner output parsing
# ----------------------------------------------------------------------------

def parse_memrefs(text):
    """Extract every printed 'data = [...]' blob as an np.float32 array."""
    out = []
    idx = 0
    while True:
        p = text.find("data =", idx)
        if p < 0:
            break
        b = text.find("[", p)
        if b < 0:
            break
        depth = 0
        end = -1
        for i in range(b, len(text)):
            c = text[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            break
        out.append(np.array(ast.literal_eval(text[b:end + 1]), dtype=np.float32))
        idx = end + 1
    return out


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def run_case(name, B, T, H, K, V, C, seed, workdir):
    data = gen.make_data(B, T, H, K, V, seed)
    text = gen.emit_module(data, B, T, H, K, V, C, meta=f"seed={seed}")
    inpath = os.path.join(workdir, f"{name}.mlir")
    with open(inpath, "w") as f:
        f.write(text)

    r = subprocess.run(["bash", os.path.join(REPO, "scripts", "run_l1.sh"), inpath],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{name}] RUN FAILED (rc={r.returncode})")
        print("--- stderr tail ---")
        print("\n".join(r.stderr.splitlines()[-30:]))
        return None

    ms = parse_memrefs(r.stdout)
    if len(ms) < 2:
        print(f"[{name}] PARSE FAILED: only {len(ms)} memrefs in runner stdout")
        return None
    o_m, ht_m = ms[0], ms[1]

    o_ref, ht_ref = chunked_ref(data["q"], data["k"], data["v"],
                                data["beta"], data["g"], data["h0"], C)
    o_seq, ht_seq = seq_ref(data["q"], data["k"], data["v"],
                            data["beta"], data["g"], data["h0"])
    algo_check = float(np.max(np.abs(o_ref - o_seq)))
    algo_ht = float(np.max(np.abs(ht_ref - ht_seq)))

    exp_o_shape = (B, ((T + C - 1) // C) * C, H, V)
    assert o_m.shape == exp_o_shape, f"o shape {o_m.shape} != {exp_o_shape}"
    assert ht_m.shape == (B, H, K, V), f"ht shape {ht_m.shape}"
    diff_o = float(np.max(np.abs(o_m[:, :T, :, :] - o_ref)))
    diff_ht = float(np.max(np.abs(ht_m - ht_ref)))
    ok = diff_o < PASS_ATOL and diff_ht < PASS_ATOL
    return {
        "name": name, "shape": f"B{B} T{T} H{H} K{K} V{V} C{C}",
        "diff_o": diff_o, "diff_ht": diff_ht,
        "algo_o": algo_check, "algo_ht": algo_ht, "ok": ok,
    }


def run_npz_case(name, npz_path, workdir):
    """Cross-workstream case: embed the G1-A JAX dump inputs, run MLIR, and
    compare against the dump's own fp32 o/ht (fp64 compute, cast)."""
    data, refs = gen.load_npz(npz_path)
    B, T, H, K = data["q"].shape
    V = data["v"].shape[-1]
    C = 64 if "_C64" in os.path.basename(npz_path) else 32
    text = gen.emit_module(data, B, T, H, K, V, C, refs=refs,
                           meta="npz=" + os.path.basename(npz_path))
    inpath = os.path.join(workdir, f"{name}.mlir")
    with open(inpath, "w") as f:
        f.write(text)
    r = subprocess.run(["bash", os.path.join(REPO, "scripts", "run_l1.sh"), inpath],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{name}] RUN FAILED (rc={r.returncode})")
        print("\n".join(r.stderr.splitlines()[-30:]))
        return None
    ms = parse_memrefs(r.stdout)
    if len(ms) < 2:
        print(f"[{name}] PARSE FAILED: {len(ms)} memrefs")
        return None
    o_m, ht_m = ms[0], ms[1]
    diff_o = float(np.max(np.abs(o_m[:, :T, :, :] - refs["o"])))
    diff_ht = float(np.max(np.abs(ht_m - refs["ht"])))
    o_ref, _ = chunked_ref(data["q"], data["k"], data["v"],
                           data["beta"], data["g"], data["h0"], C)
    diff_onp = float(np.max(np.abs(o_m[:, :T, :, :] - o_ref)))
    return {
        "name": name, "shape": f"B{B} T{T} H{H} K{K} V{V} C{C}",
        "diff_o": diff_o, "diff_ht": diff_ht,
        "algo_o": diff_onp, "algo_ht": float("nan"),
        "ok": diff_o < PASS_ATOL and diff_ht < PASS_ATOL,
    }


GRADS = ("dq", "dk", "dv", "dbeta", "dg", "dh0")
BWD_ATOL = 2e-4       # vs fp32 primary arrays (print floor ~6 sig digits)
BWD_ATOL64 = 3e-4     # vs *_fp64 arrays


def _finite(*arrs):
    return bool(all(np.all(np.isfinite(a)) for a in arrs))


def run_bwd_case(name, B, T, H, K, V, C, seed, workdir):
    """Seeded backward case: MLIR @gdn_bwd vs the numpy backward twin."""
    data = gen.make_bwd_data(gen.make_data(B, T, H, K, V, seed), seed)
    text = gen.emit_module(data, B, T, H, K, V, C, meta=f"seed={seed}", emit="bwd")
    inpath = os.path.join(workdir, f"{name}_bwd.mlir")
    with open(inpath, "w") as f:
        f.write(text)
    r = subprocess.run(["bash", os.path.join(REPO, "scripts", "run_l1.sh"), inpath],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{name}] RUN FAILED (rc={r.returncode})")
        print("\n".join(r.stderr.splitlines()[-30:]))
        return None
    ms = parse_memrefs(r.stdout)
    if len(ms) < 6:
        print(f"[{name}] PARSE FAILED: {len(ms)} memrefs")
        return None
    ref = chunked_bwd_ref(data["q"], data["k"], data["v"], data["beta"],
                          data["g"], data["h0"], data["do"], data["dht"], C)
    diffs = {}
    for i, gn in enumerate(GRADS):
        got = ms[i]
        exp = ref[i]
        if gn != "dh0":            # runner prints padded Tp rows; twin unpads
            got = got[:, :T]
        assert got.shape == exp.shape, \
            f"{gn}: mlir {got.shape} != twin {exp.shape}"
        diffs[gn] = float(np.max(np.abs(got - exp)))
    fin = _finite(*ms[:6])
    ok = fin and all(d < BWD_ATOL for d in diffs.values())
    return {"name": name, "shape": f"B{B} T{T} H{H} K{K} V{V} C{C}",
            "grads": diffs, "finite": fin, "ok": ok}


def run_bwd_npz_case(name, npz_path, workdir):
    """G1-A npz backward case: embed inputs + cotangents, run MLIR @gdn_bwd,
    compare all six gradients against the dump's JAX-vjp grads."""
    data, refs = gen.load_npz(npz_path, bwd=True)
    B, T, H, K = data["q"].shape
    V = data["v"].shape[-1]
    C = 64 if "_C64" in os.path.basename(npz_path) else 32
    text = gen.emit_module(data, B, T, H, K, V, C, refs=refs,
                           meta="npz=" + os.path.basename(npz_path), emit="bwd")
    inpath = os.path.join(workdir, f"{name}_bwd.mlir")
    with open(inpath, "w") as f:
        f.write(text)
    r = subprocess.run(["bash", os.path.join(REPO, "scripts", "run_l1.sh"), inpath],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{name}] RUN FAILED (rc={r.returncode})")
        print("\n".join(r.stderr.splitlines()[-30:]))
        return None
    ms = parse_memrefs(r.stdout)
    if len(ms) < 6:
        print(f"[{name}] PARSE FAILED: {len(ms)} memrefs")
        return None
    got = dict(zip(GRADS, ms[:6]))
    d64 = np.load(npz_path)
    diffs = {}
    ok = True
    for gn in GRADS:
        if gn not in refs:
            continue
        mlir = got[gn]
        ref32 = refs[gn]
        ref64 = d64[gn + "_fp64"]
        if gn == "dh0":
            d32 = float(np.max(np.abs(mlir - ref32)))
            d64d = float(np.max(np.abs(mlir.astype(np.float64) - ref64)))
        else:
            m = mlir[:, :T] if mlir.ndim == 3 else mlir[:, :T, :, :]
            d32 = float(np.max(np.abs(m - ref32)))
            d64d = float(np.max(np.abs(m.astype(np.float64) - ref64)))
        diffs[gn] = (d32, d64d)
        ok &= (d32 < BWD_ATOL or d64d < BWD_ATOL64)
    ok &= _finite(*ms[:6])
    return {"name": name, "shape": f"B{B} T{T} H{H} K{K} V{V} C{C}",
            "grads": diffs, "finite": _finite(*ms[:6]), "ok": ok}


def main_bwd(only_npz=False):
    """G1-B2 backward gate: 3 npz x 6 grads vs JAX vjp (+ seeded tail cases
    vs the numpy twin for the zero-padded tail-chunk path)."""
    results = []
    with tempfile.TemporaryDirectory(prefix="gdn_l1b_") as wd:
        if not only_npz:
            for c in (("base", 1, 128, 2, 16, 16, 64, 0),
                      ("chunk32", 1, 128, 2, 16, 16, 32, 1),
                      ("tail_t96", 1, 96, 2, 16, 16, 64, 2),
                      ("tail_b2", 2, 100, 2, 16, 16, 64, 3)):
                res = run_bwd_case(*c, wd)
                if res is None:
                    sys.exit(1)
                results.append(res)
        npzs = [
            ("jax_C64_h0", "ref_jax_T128_C64_h0.npz"),
            ("jax_C64", "ref_jax_T128_C64.npz"),
            ("jax_C32", "ref_jax_T128_C32.npz"),
        ]
        for nm, fn in npzs:
            p = os.path.join(REPO, "evidence", "G1", fn)
            if not os.path.exists(p):
                continue
            res = run_bwd_npz_case(nm, p, wd)
            if res is None:
                sys.exit(1)
            results.append(res)

    print()
    hdr = (f"{'case':<12} {'shape':<24} " + " ".join(f"{g:>11}" for g in GRADS)
           + "  finite  verdict")
    print(hdr)
    all_ok = True
    for r in results:
        cells = []
        for g in GRADS:
            v = r["grads"].get(g)
            if v is None:
                cells.append(" " * 11)
            elif isinstance(v, tuple):
                cells.append(f"{v[0]:>6.1e}/{v[1]:<4.0e}" if v[1] < 10
                             else f"{v[0]:>11.1e}")
            else:
                cells.append(f"{v:>11.1e}")
        verdict = "PASS" if r["ok"] else "FAIL"
        all_ok &= r["ok"]
        print(f"{r['name']:<12} {r['shape']:<24} " + " ".join(cells)
              + f"  {str(r['finite']):<6}  {verdict}")
    print()
    print("gate: max abs diff < %.0e vs fp32 refs (or < %.0e vs *_fp64); "
          "no NaN/Inf" % (BWD_ATOL, BWD_ATOL64))
    if not all_ok:
        sys.exit(1)
    print("ALL PASS")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bwd", action="store_true",
                    help="run the G1-B2 backward checks (seeded vs numpy twin "
                         "+ G1-A npz vs JAX vjp grads) instead of the fwd checks")
    ap.add_argument("--bwd-only-npz", action="store_true",
                    help="with --bwd: only the three G1-A npz cases")
    args = ap.parse_args()

    if args.bwd:
        return main_bwd(only_npz=args.bwd_only_npz)

    cases = [
        ("base", 1, 128, 2, 16, 16, 64, 0),
        ("chunk32", 1, 128, 2, 16, 16, 32, 1),
        ("tail_t96", 1, 96, 2, 16, 16, 64, 2),   # 96 = 64 + 32: tail chunk
        ("tail_b2", 2, 100, 2, 16, 16, 64, 3),   # B=2, 100 = 64 + 36: tail
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="gdn_l1_") as wd:
        for c in cases:
            res = run_case(*c, wd)
            if res is None:
                sys.exit(1)
            results.append(res)

        # optional cross-workstream check against G1-A dumps (tolerate absence)
        npzs = [
            ("jax_C64_h0", "ref_jax_T128_C64_h0.npz"),
            ("jax_C64", "ref_jax_T128_C64.npz"),
            ("jax_C32", "ref_jax_T128_C32.npz"),
        ]
        for nm, fn in npzs:
            p = os.path.join(REPO, "evidence", "G1", fn)
            if os.path.exists(p):
                res = run_npz_case(nm, p, wd)
                if res is not None:
                    results.append(res)

    print()
    print(f"{'case':<10} {'shape':<26} {'max|o-mlir-ref|':>16} "
          f"{'max|ht-mlir-ref|':>17} {'numpy chunk-vs-seq':>20}  verdict")
    all_ok = True
    for r in results:
        verdict = "PASS" if r["ok"] else "FAIL"
        all_ok &= r["ok"]
        algo = r["algo_o"] if np.isnan(r["algo_ht"]) else max(r["algo_o"], r["algo_ht"])
        print(f"{r['name']:<10} {r['shape']:<26} {r['diff_o']:>16.3e} "
              f"{r['diff_ht']:>17.3e} {algo:>20.3e}  {verdict}")
    print()
    print("gate: max abs diff < %.0e for o and ht (fp32 CPU)" % PASS_ATOL)
    if not all_ok:
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
