#!/usr/bin/env python3
"""gdn_carry_validate.py -- G5-J round5 CPU validation of the parallel
[K,K]-carry reformulation of the bwd reverse cotangent scan.

CLAIM (from the L1 bwd math / the dhu emitter): the reverse scan is an
affine linear recurrence per chunk n,

    X_{n-1} = A_n X_n + B_n,     X_{NC-1} = dht,

with X_n = dH ARRIVING at chunk n (exactly what dhu stores to dh_all[m]
PRE-update and what dkb/dqkwg consume), and

    A_n = 2^gcl I_K  -  W^T Bhat            ([K,K]; vt-INDEPENDENT)
    B_n = M_n dO_n,  M_n = Qg^T - W^T P^T    ([K,V] per v-slice)

where W = w_all (wy output), Bhat = k .* 2^(gcl-gc_row), Qg = q .* 2^gc_row,
dO = do/sqrt(K), P = P_all.  This follows algebraically from the sequential
update  dH' = 2^gcl dH + Qg^T dO - W^T (P^T dO + Bhat dH).

This script validates, per (shape, seed), against the sequential trajectory
of the fp64 twin (gdn_bwd_verify.bwd_twin math):

  1. CARRY   : X from the composed recurrence (fp64 -> machine precision?)
  2. SEGMENT : the 2-level decomposition actually implementable on GPU --
              per-segment zero-seed local walks Y, per-segment composed
              (A_seg, B_seg), a serial boundary chain T_s, correction
              X_n = P_n T_s + Y_n with per-chunk suffix products P_n.
              (fp64 -> machine precision for S in {2,4,8,16})
  3. fp32 replay of 1+2 with the GPU operand precisions (bf16-widened q/k/do,
              f32 W from the f32 wy chain) -> the re-association noise floor
              the GPU kernels will see (informative only; the real gate is
              gdn_bwd_verify.py).
  4. conditioning: spectral radii of A_n, ||A_n||, 2^gcl range, ||X_n||, and
              the correction magnitudes ||P_n|| (do zero-seed segment walks
              stay bounded?  does P_n decay?).

Usage (conda gdn):
  python tools/gdn_carry_validate.py --shape 1,512,4,128,128 [--seed 0]
  python tools/gdn_carry_validate.py --shape 1,2048,16,128,128
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_bwd_verify import (make_data, layout, bf16_widen,  # noqa: E402
                            bwd_twin)

RCP_LN2 = 1.4426950408889634


def build_chunk_operators(lay, bh, dt):
    """Per-chunk (A_n, M_n, dO_n) in dtype dt, from the SAME operands the GPU
    kernels see: q/k/do bf16-widened, W/P f32-widened-from-f32-chain, gc from
    the f32 g cumsum.  Returns lists over n + gcl array.  Also returns the
    forward-recurrence quantities needed by the twin's sequential walk."""
    md = lay["_meta"]
    B, H, K, V, C = md["B"], md["H"], md["K"], md["V"], md["C"]
    NC = md["NC"]

    def widen_chunks(u16, W_):
        return bf16_widen(u16).reshape(md["BH"], NC, C, W_)

    q = widen_chunks(lay["q"], K)[bh].astype(dt)
    k = widen_chunks(lay["k"], K)[bh].astype(dt)
    do = widen_chunks(lay["do"], V)[bh].astype(dt)
    beta = lay["beta"].astype(np.float64).reshape(md["BH"], NC, C)[bh].astype(dt)
    # g: the GPU computes gc as an f32 cumsum of the f32 log2-domain g, then
    # widens to fp64-free f32 exp2 arithmetic; mirror with f32 cumsum.
    g_log2 = lay["g_log2"].astype(np.float32).reshape(md["BH"], NC, C)[bh]
    gc32 = np.cumsum(g_log2, axis=-1).astype(np.float32)
    gc = gc32.astype(dt)
    scale = dt(1.0 / K ** 0.5)

    # forward pass (twin's fwd loop) to get W = T(beta k 2^gc), P, and the
    # exact f32-class Tm; in fp64 mode this matches bwd_twin bit-for-bit in
    # structure.
    m_inc = np.tril(np.ones((C, C)))
    A_list, M_list, W_list, P_list = [], [], [], []
    gcl_arr = np.empty(NC, dtype=dt)
    for n in range(NC):
        gcn = gc[n]
        gcl = gcn[-1]
        gcl_arr[n] = gcl
        kkT = k[n] @ k[n].T
        A = np.tril(beta[n][:, None] * kkT *
                    np.exp2(gcn[:, None] - gcn[None, :]), -1)
        Tm = np.eye(C)
        for i in range(1, C):
            Tm[i, :i] = -(A[i, :i] @ Tm[:i, :i])
        W = Tm @ (beta[n][:, None] * k[n] * np.exp2(gcn)[:, None])
        P = np.tril((q[n] @ k[n].T) * np.exp2(gcn[:, None] - gcn[None, :]), 0)
        Bhat = k[n] * np.exp2(gcl - gcn)[:, None]
        Qg = q[n] * np.exp2(gcn)[:, None]
        Aop = np.exp2(gcl) * np.eye(K) - W.T @ Bhat
        Mop = Qg.T - W.T @ P.T
        A_list.append(Aop.astype(dt))
        M_list.append(Mop.astype(dt))
        W_list.append(W.astype(dt))
        P_list.append(P.astype(dt))
    dOs = (do * scale).astype(dt)
    dht = bf16_widen(lay["dht"]).reshape(md["BH"], K, V)[bh].astype(dt)
    return A_list, M_list, dOs, dht, gcl_arr, W_list, P_list, gc, beta, k, q


def seq_walk(A_list, M_list, dOs, dht, dt, W_list=None, P_list=None,
             gc=None, beta=None, k=None, q=None):
    """Sequential reference in the TRUE dhu/twin form:
       dvp = P^T dO + Bhat X;  X' = 2^gcl X + Qg^T dO - W^T dvp
    (X[n] recorded PRE-update = the dh_all contract).  If the W/P/gc
    operands are not given, falls back to the composed expression -- the
    caller passes them for the honest re-association comparison."""
    NC = len(A_list)
    X = dht.copy()
    Xs = np.empty((NC,) + dht.shape, dtype=dt)
    for n in reversed(range(NC)):
        Xs[n] = X
        if W_list is None:
            X = A_list[n] @ X + M_list[n] @ dOs[n]
        else:
            gcl = gc[n][-1]
            Bhat = k[n] * np.exp2(gcl - gc[n])[:, None]
            Qg = q[n] * np.exp2(gc[n])[:, None]
            dvp = P_list[n].T @ dOs[n] + Bhat @ X
            X = (np.exp2(gcl) * X + Qg.T @ dOs[n] - W_list[n].T @ dvp)
    return Xs, X  # trajectory + dh0


def carry_walk(A_list, M_list, dOs, dht, dt):
    """Identical recurrence -- written as the pure composed form the GPU scan
    will use (B precomputed).  Numerically the re-association appears only
    through B = M dO vs the sequential (Qg^T dO - W^T (P^T dO))."""
    NC = len(A_list)
    B_list = [M @ dOs[n] for n, M in enumerate(M_list)]
    X = dht.copy()
    Xs = np.empty((NC,) + dht.shape, dtype=dt)
    for n in reversed(range(NC)):
        Xs[n] = X
        X = A_list[n] @ X + B_list[n]
    return Xs, X, B_list


def seg_walk(A_list, B_list, dht, S, dt):
    """2-level segment scan (the GPU step-B structure):
       seg s covers chunks [s*G, (s+1)*G);  walk order: seg S-1 first.
       T_s      = state ARRIVING at the first-walked chunk of seg s
                  (= X_{hi_s - 1});  T_{S-1} = dht.
       local    : Y zero-seed walk inside the segment (all chunks)
       suffix   : P_n = A_{n+1} .. A_{hi_s-1}   (P_{hi_s-1} = I)
       boundary : T_{s} = A_seg_{s+1} T_{s+1} + B_seg_{s+1}
                  (A_seg = P_{lo}, B_seg = local bottom output Y_{lo-1})
       correct  : X_n = P_n @ T_{top(n)} + Y_n
    """
    NC = len(A_list)
    assert NC % S == 0, f"NC={NC} not divisible by S={S}"
    G = NC // S
    K, V = dht.shape
    Ys = np.empty((NC, K, V), dtype=dt)
    Xs = np.empty((NC, K, V), dtype=dt)
    Psuf = [None] * NC
    A_seg = [None] * S
    B_seg = np.zeros((S, K, V), dtype=dt)
    for s in range(S):                    # parallel on GPU
        lo, hi = s * G, (s + 1) * G
        Y = np.zeros((K, V), dtype=dt)
        P = np.eye(K, dtype=dt)
        for n in range(hi - 1, lo - 1, -1):
            Ys[n] = Y
            Psuf[n] = P
            Y = A_list[n] @ Y + B_list[n]
            P = A_list[n] @ P
        A_seg[s] = P                       # = P_{lo-1}: seg's full product
        B_seg[s] = Y                       # zero-seed bottom output
    # boundary chain (serial over S on GPU; tiny)
    T = np.empty((S, K, V), dtype=dt)
    T[S - 1] = dht
    for s in range(S - 2, -1, -1):
        T[s] = A_seg[s + 1] @ T[s + 1] + B_seg[s + 1]
    # correction (parallel over chunks on GPU)
    for s in range(S):
        lo, hi = s * G, (s + 1) * G
        for n in range(lo, hi):
            Xs[n] = Psuf[n] @ T[s] + Ys[n]
    dh0 = A_seg[0] @ T[0] + B_seg[0]
    return Xs, dh0, A_seg, B_seg


def conditioning(A_list, Xs, dht, dt):
    NC = len(A_list)
    rhos = np.array([np.abs(np.linalg.eigvals(A)).max() for A in A_list])
    norms = np.array([np.linalg.norm(A) for A in A_list])
    xs = np.array([np.linalg.norm(X) for X in Xs])
    return rhos, norms, xs


def run_shape(shape, chunk, seed, seg_list, out=None):
    B, T, H, K, V = (int(x) for x in shape.split(","))
    data = make_data(B, T, H, K, V, seed)
    lay = layout(data, B, T, H, K, V, chunk)
    md = lay["_meta"]
    BH, NC = md["BH"], md["NC"]
    report = {"shape": shape, "chunk": chunk, "seed": seed,
              "BH": BH, "NC": NC, "modes": {}}

    # ---- fp64: sequential twin walk vs composed vs segments ----
    max_seq = max_carry = 0.0
    max_seg = {S: 0.0 for S in seg_list}
    cond = {"rho": [], "Anorm": [], "Xnorm": []}
    for bh in range(BH):
        ops = build_chunk_operators(lay, bh, np.float64)
        A_list, M_list, dOs, dht = ops[0], ops[1], ops[2], ops[3]
        W_list, P_list, gc, beta, k, q = ops[5], ops[6], ops[7], ops[8], \
            ops[9], ops[10]
        Xseq, dh0s = seq_walk(A_list, M_list, dOs, dht, np.float64,
                              W_list=W_list, P_list=P_list, gc=gc, beta=beta,
                              k=k, q=q)
        Xcom, dh0c, B_list = carry_walk(A_list, M_list, dOs, dht, np.float64)
        max_seq = max(max_seq, np.abs(Xseq - Xcom).max(),
                      np.abs(dh0s - dh0c).max())
        max_carry = max(max_carry, np.abs(Xseq - Xcom).max())
        for S in seg_list:
            Xsg, dh0g, _, _ = seg_walk(A_list, B_list, dht, S, np.float64)
            max_seg[S] = max(max_seg[S], np.abs(Xseq - Xsg).max(),
                             np.abs(dh0s - dh0g).max())
        rhos, norms, xs = conditioning(A_list, Xseq, dht, np.float64)
        cond["rho"].append(rhos)
        cond["Anorm"].append(norms)
        cond["Xnorm"].append(xs)
    rho = np.concatenate(cond["rho"])
    an = np.concatenate(cond["Anorm"])
    xn = np.concatenate(cond["Xnorm"])
    report["fp64"] = {
        "carry_vs_seq_max_abs": float(max_carry),
        "segment_vs_seq_max_abs": {f"S={S}": float(v)
                                   for S, v in max_seg.items()},
        "rho_A_min_med_max": [float(rho.min()), float(np.median(rho)),
                              float(rho.max())],
        "normA_max": float(an.max()),
        "normX_max": float(xn.max()),
    }

    # ---- anchor: my fp64 sequential walk vs the REAL twin (bwd_twin) ----
    # Expected ~1e-6 (NOT machine eps): the GPU's gc is an f32 cumsum of the
    # f32 log2-domain g, which this script models; bwd_twin cumsums the
    # natural-log g in fp64.  This anchor confirms the operator EXTRACTION
    # (A_n/M_n from q/k/g/beta/do/w/P) is the twin's math, not a lookalike.
    from gdn_bwd_verify import bwd_twin as _twin
    ref = _twin(lay)
    ops0 = build_chunk_operators(lay, 0, np.float64)
    Xs0, dh00 = seq_walk(ops0[0], ops0[1], ops0[2], ops0[3], np.float64,
                         W_list=ops0[5], P_list=ops0[6], gc=ops0[7],
                         beta=ops0[8], k=ops0[9], q=ops0[10])
    report["fp64"]["seq_vs_realtwin_dh0_bh0"] = float(
        np.abs(dh00 - ref["dh0"][0]).max())

    # ---- fp32 replay (GPU operand precisions; noise floor estimate) ----
    max_c32 = 0.0
    max_s32 = {S: 0.0 for S in seg_list}
    ref64 = None
    for bh in range(min(BH, 4)):          # fp64 sequential baseline per bh
        ops64 = build_chunk_operators(lay, bh, np.float64)
        Xs64, _ = seq_walk(ops64[0], ops64[1], ops64[2], ops64[3],
                           np.float64, W_list=ops64[5], P_list=ops64[6],
                           gc=ops64[7], beta=ops64[8], k=ops64[9], q=ops64[10])
        ops = build_chunk_operators(lay, bh, np.float32)
        A_list, M_list, dOs, dht = ops[0], ops[1], ops[2], ops[3]
        Xc32, _, B_list = carry_walk(A_list, M_list, dOs, dht, np.float32)
        max_c32 = max(max_c32, np.abs(Xs64 - Xc32).max())
        for S in seg_list:
            Xg32, _, _, _ = seg_walk(A_list, B_list, dht, S, np.float32)
            max_s32[S] = max(max_s32[S], np.abs(Xs64 - Xg32).max())
    report["fp32"] = {
        "carry_vs_fp64seq_max_abs": float(max_c32),
        "segment_vs_fp64seq_max_abs": {f"S={S}": float(v)
                                       for S, v in max_s32.items()},
    }

    # ---- verdict ----
    eps64 = 1e-12
    verdict = "EXACT" if max_carry < eps64 else (
        "APPROX" if max_carry < 1e-8 else "FALSE")
    segv = {f"S={S}": ("EXACT" if v < eps64 else "APPROX" if v < 1e-8
                       else "FALSE") for S, v in max_seg.items()}
    report["verdict"] = {"carry": verdict, "segments": segv}
    print(json.dumps(report, indent=2))
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(report, open(out, "w"), indent=2)
        print(f"wrote {out}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1,512,4,128,128")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segments", default="2,4,8,16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seg_list = [int(x) for x in args.segments.split(",") if x]
    NC = (int(args.shape.split(",")[1]) + args.chunk - 1) // args.chunk
    seg_list = [S for S in seg_list if NC % S == 0] or [1]
    print(f"[carry_val] shape={args.shape} NC={NC} segments={seg_list}")
    run_shape(args.shape, args.chunk, args.seed, seg_list, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
