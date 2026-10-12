#!/usr/bin/env python3
"""g5i_split_sim.py -- G5-I round 4: CPU emulation of the F2 segment-split.

Two questions answered BEFORE any GPU work:
  1. MATH: is the 3-pass split (A: zero-entry exits; M: per-segment
     transition composition M_seg = prod(2^gcl I - kterm^T w); B: stitch
     E_{i+1} = M_i E_i + c_i; C: re-run with true entries) EXACTLY equal
     to the serial scan?  (The brief's "matmul-free 2-pass" claim is
     tested too: chaining zero-entry exits directly is WRONG because the
     recurrence is affine in S, not state-independent.)
  2. NUMERICS: does the bf16-rounded kernel emulation (mirroring the
     exact op order/dtype cut points of the planned MLIR kernels) stay
     under the 5e-3 contract at T512/T2048?

Run: python tools/g5i_split_sim.py [--tag small_B1T512H4] [--segs 2,4,8]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.expanduser("~/codebase/gdn-mlir")


def bf16(x):
    """Round fp32/f64 array to bf16 (round-to-nearest-even), back to f32."""
    u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u32.astype(np.uint64) + 0x7FFF + ((u32 >> 16) & 1)) >> 16
    return (r.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def chunk_wy(qc, kc, vc, betac, gc):  # all fp64 [C,K]/[C,K]/[C,V]/[C]/[C] (log2-domain gc)
    """F1 per chunk (fp64 'exact'): returns u, w, gc (log2), kterm-free."""
    C, K = qc.shape
    A = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            if j < i:
                A[i, j] = betac[i] * (kc[i] @ kc[j]) * 2.0 ** (gc[i] - gc[j])
    T = np.eye(C)
    for j in range(C):           # forward substitution, T = (I+A)^-1
        for i in range(j + 1, C):
            T[i, j] = -A[i, j:i] @ T[j:i, j]
    u = T @ (betac[:, None] * vc)
    w = T @ (betac[:, None] * kc * 2.0 ** gc[:, None])
    return u, w


def state_chunk(S, u, w, k, gcl, gc, emulate_bf16):
    """One F2 chunk.  Returns (S_new, v_new_scaled).  Mirrors the kernel:
    vp = w @ S (f32 acc, bf16 operands); vn = (u - vp) * 2^(gcl-gc) -> bf16;
    sd = k^T @ vn (f32 acc, bf16 operands); S = bf16(S*2^gcl + sd)."""
    if emulate_bf16:
        vp = bf16(w).astype(np.float64) @ bf16(S)
        vn = bf16((u - vp) * 2.0 ** (gcl - gc)[:, None])
        sd = bf16(k).astype(np.float64).T @ vn
        S2 = bf16(bf16(S) * 2.0 ** gcl + sd)
        return S2, vn
    vp = w @ S
    vn = (u - vp) * 2.0 ** (gcl - gc)[:, None]
    sd = k.T @ vn
    return 2.0 ** gcl * S + sd, vn


def run_case(tag, seg_list, emulate_bf16, stress_g=0.0):
    inp = dict(np.load(os.path.join(REPO, f"evidence/G5/refs/{tag}_inputs.npz")))
    q, k, v = inp["q_bf16"], inp["k_bf16"], inp["v_bf16"]
    beta, g, h0 = inp["beta"], inp["g"], inp["h0"]
    if stress_g:
        # adversarial validity probe: near-zero log-decays (gate ~ 1).  The
        # matmul-free handoff is exact only when segments carry net decay.
        rng = np.random.default_rng(7)
        g = -stress_g * np.abs(rng.standard_normal(g.shape))
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = 64
    Tp = ((T + C - 1) // C) * C
    NC = Tp // C
    RCP = 1.4426950408889634
    pad3 = lambda x: np.pad(x, ((0, 0), (0, Tp - T), (0, 0)))
    pad4 = lambda x: np.pad(x, ((0, 0), (0, Tp - T), (0, 0), (0, 0)))
    q, k, v = pad4(q), pad4(k), pad4(v)
    beta, g = pad3(beta), pad3(g)
    scale = K ** -0.5

    # ---- per-chunk WY precompute (F1, fp64) ----
    U = np.zeros((B, H, NC, C, V))
    W = np.zeros((B, H, NC, C, K))
    GC = np.zeros((B, H, NC, C))
    GCL = np.zeros((B, H, NC))
    for b in range(B):
        for h in range(H):
            for n in range(NC):
                sl = slice(n * C, (n + 1) * C)
                gg = g[b, sl, h] * RCP
                gc = np.cumsum(gg)
                U[b, h, n], W[b, h, n] = chunk_wy(
                    q[b, sl, h] * scale, k[b, sl, h], v[b, sl, h],
                    beta[b, sl, h], gc)
                GC[b, h, n] = gc
                GCL[b, h, n] = gc[-1]

    def outputs(h_all, v_new):  # F3 (fp64) -> o, ht
        o = np.zeros((B, T, H, V))
        for b in range(B):
            for h in range(H):
                for n in range(NC):
                    sl = slice(n * C, (n + 1) * C)
                    gc = GC[b, h, n]
                    qn = q[b, sl, h] * scale
                    # v_new carried in the kernel/sim is kterm-FOLDED
                    # (scaled by 2^(gcl-gc_j)), so P uses 2^(gc_i-gcl):
                    P = np.tril((qn @ k[b, sl, h].T) * 2.0 ** (gc[:, None] - GCL[b, h, n]), 0)
                    oi = (qn * 2.0 ** gc[:, None]) @ h_all[b, h, n]
                    oa = P @ v_new[b, h, n]
                    t0 = min(T, (n + 1) * C)
                    if t0 > n * C:
                        o[b, n * C:t0, h] = (oi + oa)[: t0 - n * C]
        return o

    def serial(S0):
        S = S0.copy()
        h_all = np.zeros((B, H, NC, K, V))
        v_new = np.zeros((B, H, NC, C, V))
        for b in range(B):
            for h in range(H):
                St = S0[b, h].copy()
                for n in range(NC):
                    h_all[b, h, n] = St
                    St, vn = state_chunk(St, U[b, h, n], W[b, h, n],
                                         k[b, n * C:(n + 1) * C, h],
                                         GCL[b, h, n], GC[b, h, n], emulate_bf16)
                    v_new[b, h, n] = vn
                S[b, h] = St
        return S, h_all, v_new

    ref = dict(np.load(os.path.join(REPO, f"evidence/G5/refs/{tag}_ref.npz")))
    ro = ref.get("ref__o_f64", ref.get("ref__o_f32"))
    rh = ref.get("ref__ht_f64", ref.get("ref__ht_f32"))

    S0 = h0.astype(np.float64)
    if emulate_bf16:
        S0 = bf16(S0)
    ht_s, ha_s, vn_s = serial(S0)
    o_s = outputs(ha_s, vn_s)
    base = {
        "serial_o_vs_ref": float(np.abs(o_s[:, :T] - ro).max()),
        "serial_ht_vs_ref": float(np.abs(ht_s - rh).max()),
    }

    out = {"tag": tag, "bf16": emulate_bf16, "serial": base, "splits": {}}
    for NS in seg_list:
        if NC % NS:
            continue
        L = NC // NS
        # pass A: zero-entry exits c_sg  (+ naive matmul-free chain, to test
        # the brief's claim)
        cexit = np.zeros((B, H, NS, K, V))
        for b in range(B):
            for h in range(H):
                for sg in range(NS):
                    St = np.zeros((K, V))
                    for n in range(sg * L, (sg + 1) * L):
                        St, _ = state_chunk(St, U[b, h, n], W[b, h, n],
                                            k[b, n * C:(n + 1) * C, h],
                                            GCL[b, h, n], GC[b, h, n],
                                            emulate_bf16)
                    cexit[b, h, sg] = St
        # pass M: compose transitions per segment (ALWAYS audited in fp64;
        # the bf16 cut points only affect the emulate_bf16 stitch below)
        Mseg = np.zeros((B, H, NS, K, K))
        for b in range(B):
            for h in range(H):
                for sg in range(NS):
                    M = np.eye(K)
                    for n in range(sg * L, (sg + 1) * L):
                        gcl = GCL[b, h, n]
                        kt = k[b, n * C:(n + 1) * C, h] * 2.0 ** (gcl - GC[b, h, n])[:, None]
                        Mn = 2.0 ** gcl * np.eye(K) - kt.T @ W[b, h, n]
                        M = Mn @ M
                    Mseg[b, h, sg] = M
        # pass B: stitch
        E = np.zeros((B, H, NS, K, V))
        for b in range(B):
            for h in range(H):
                E[b, h, 0] = S0[b, h]
                for sg in range(NS - 1):
                    if emulate_bf16:
                        E[b, h, sg + 1] = bf16(bf16(Mseg[b, h, sg]).astype(np.float64) @ bf16(E[b, h, sg]) + bf16(cexit[b, h, sg]))
                    else:
                        E[b, h, sg + 1] = Mseg[b, h, sg] @ E[b, h, sg] + cexit[b, h, sg]
        # pass C: re-run with true entries
        ht_c = np.zeros_like(S0)
        ha_c = np.zeros((B, H, NC, K, V))
        vn_c = np.zeros((B, H, NC, C, V))
        for b in range(B):
            for h in range(H):
                for sg in range(NS):
                    St = E[b, h, sg].copy()
                    for n in range(sg * L, (sg + 1) * L):
                        ha_c[b, h, n] = St
                        St, vn = state_chunk(St, U[b, h, n], W[b, h, n],
                                             k[b, n * C:(n + 1) * C, h],
                                             GCL[b, h, n], GC[b, h, n],
                                             emulate_bf16)
                        vn_c[b, h, n] = vn
                    ht_c[b, h] = St
        o_c = outputs(ha_c, vn_c)
        # max transition magnitude: bounds the DROPPED term ||M_sg @ E_sg||
        # in the matmul-free stitch (E_{sg+1} = c_sg)
        mmax = float(np.abs(Mseg).max())
        mrow = float(max(np.abs(Mseg[b, h, sg]).sum(axis=1).max()
                         for b in range(B) for h in range(H)
                         for sg in range(NS)))
        out["splits"][NS] = {
            "o_vs_ref": float(np.abs(o_c[:, :T] - ro).max()),
            "ht_vs_ref": float(np.abs(ht_c - rh).max()),
            "o_vs_serial": float(np.abs(o_c - o_s).max()),
            "max_abs_Mseg": mmax,
            "max_inf_norm_Mseg": mrow,
        }
        # the naive matmul-free chain (brief's literal scheme), fp-only check
        Enaive = np.zeros((B, H, NS, K, V))
        for b in range(B):
            for h in range(H):
                Enaive[b, h, 0] = S0[b, h]
                for sg in range(NS - 1):
                    Enaive[b, h, sg + 1] = cexit[b, h, sg]   # WRONG on purpose
        ht_n = np.zeros_like(S0)
        ha_n = np.zeros((B, H, NC, K, V))
        vn_n = np.zeros((B, H, NC, C, V))
        for b in range(B):
            for h in range(H):
                for sg in range(NS):
                    St = Enaive[b, h, sg].copy()
                    for n in range(sg * L, (sg + 1) * L):
                        ha_n[b, h, n] = St
                        St, vn = state_chunk(St, U[b, h, n], W[b, h, n],
                                             k[b, n * C:(n + 1) * C, h],
                                             GCL[b, h, n], GC[b, h, n], False)
                        vn_n[b, h, n] = vn
        o_n = outputs(ha_n, vn_n)
        out["splits"][NS]["naive_matmul_free_o_vs_ref"] = float(
            np.abs(o_n[:, :T] - ro).max())
        out["splits"][NS]["naive_E_max_delta"] = float(
            np.abs(Enaive[:, :, 1:] - E[:, :, 1:]).max()) if NS > 1 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="small_B1T512H4,bench_B1T2048H16")
    ap.add_argument("--segs", default="2,4,8")
    ap.add_argument("--stress-g", type=float, default=0.0,
                    help="adversarial probe: replace g by -x*|N(0,1)|")
    args = ap.parse_args()
    for tag in args.tags.split(","):
        for embf in (False, True):
            r = run_case(tag, [int(s) for s in args.segs.split(",")], embf,
                         stress_g=args.stress_g)
            print(f"== {tag}  bf16-emulation={embf}"
                  + (f"  STRESS g=-{args.stress_g}|N(0,1)|" if args.stress_g
                     else ""))
            print(f"   serial : o {r['serial']['serial_o_vs_ref']:.3e}  "
                  f"ht {r['serial']['serial_ht_vs_ref']:.3e}")
            for ns, d in r["splits"].items():
                print(f"   S={ns:<3} : o {d['o_vs_ref']:.3e}  ht {d['ht_vs_ref']:.3e}"
                      f"  o-vs-serial {d['o_vs_serial']:.3e}"
                      f"  ||Mseg||inf {d['max_inf_norm_Mseg']:.3e}"
                      f"  | naive(matmul-free) o {d['naive_matmul_free_o_vs_ref']:.3e}"
                      f"  naiveEdelta {d['naive_E_max_delta']:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
