#!/usr/bin/env python3
"""r24b: the P11R W-park decision study on the fp64 numpy twin (CPU-only).

The dqkwg P11 dW/dq1 loop accumulates  dW = -(dvp @ Sn^T)  per (ks, vs)
block via split-bf16 mma links.  A "W-park" parks the per-ks dW
accumulators OUT of registers so the vs-only tiles (akb/vnh/vnl) stage
once per vs for ALL ks (the P11R=4 full kill) without 4 live register
sets.  This script derives the NUMERICS of every park flavor on the
twin, at both bench shapes:

  exact      : fp64 dW (ground truth)
  current    : split-bf16 3-link mma per (ks,vs), f32 acc threaded across
               vs in the link-interleaved order (the shipped P11R=2
               association)
  wpark_f32  : per-vs partials from a ZERO f32 acc, summed vs-ascending
               at the end -- the pure RE-ASSOCIATION an f32 park buys
               (park-resume of raw f32 bits is bit-identical by
               construction and is NOT simulated)
  wpark_bf16 : per-vs partials PARKED AS BF16 (the traffic-halving
               park), summed in f32 -- the only variant that changes
               VALUES, not just their rounding order

Everything downstream of dW (G = dvp@Vb.T + dW@Kbg.T, TW = Tm.T@dW,
dAf -> dk/dbeta/dg) runs in fp64 per variant on a SHARED reverse walk
(the dH recurrence is dW-independent), so the graded-output deltas
isolate the dW-class error exactly.  dq/dv/dh0 have no dW term and are
reported as 0 by construction.

Usage: python wpark_twin.py [--shape 1,2048,16,128,128] [--out json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.expanduser("~/codebase/gdn-mlir")
sys.path.insert(0, os.path.join(REPO, "tools"))

from gdn_bwd_verify import (RCP_LN2, bf16_quant, bf16_widen, layout,  # noqa
                            make_data)

C = 64
BV = 32          # dqkwg BV (the P11 vs-block)
BK = 32          # dqkwg BK (the P11 ks-block)


def f32(x):
    return np.asarray(x, dtype=np.float32)


def split(x):
    """f32 -> (hi, lo) widened f32, the stage_split_ew recipe."""
    x = f32(x)
    hi = bf16_widen(bf16_quant(x)).astype(np.float32)
    lo = f32(x - hi)
    lo = bf16_widen(bf16_quant(lo)).astype(np.float32)
    return hi, lo


def dW_current(dvp, Sn, K_, V_):
    """Shipped association: f32 acc threaded over (vs, link), 3 links."""
    vnh, vnl = split(dvp)                 # [C, V]
    acc = np.zeros((C, K_), dtype=np.float32)
    for vs in range(0, V_, BV):
        snh, snl = split(Sn[:, vs:vs + BV])        # [K, BV] slices
        a_h, a_l = vnh[:, vs:vs + BV], vnl[:, vs:vs + BV]
        for ks in range(0, K_, BK):
            bh, bl = snh[ks:ks + BK], snl[ks:ks + BK]
            # ops_w order: vnh.bhh, vnh.bhl, vnl.bhh
            acc[:, ks:ks + BK] = f32(
                acc[:, ks:ks + BK] + f32(a_h @ bh.T))
            acc[:, ks:ks + BK] = f32(
                acc[:, ks:ks + BK] + f32(a_h @ bl.T))
            acc[:, ks:ks + BK] = f32(
                acc[:, ks:ks + BK] + f32(a_l @ bh.T))
    return acc


def dW_wpark(dvp, Sn, K_, V_, park_bf16):
    """W-park association: per-vs partials from zero, vs-sum at end."""
    vnh, vnl = split(dvp)
    dw = np.zeros((C, K_), dtype=np.float32)
    for vs in range(0, V_, BV):
        snh, snl = split(Sn[:, vs:vs + BV])
        a_h, a_l = vnh[:, vs:vs + BV], vnl[:, vs:vs + BV]
        pvs = np.zeros((C, K_), dtype=np.float32)
        for ks in range(0, K_, BK):
            bh, bl = snh[ks:ks + BK], snl[ks:ks + BK]
            pvs[:, ks:ks + BK] = f32(a_h @ bh.T)          # 0 + l0
            pvs[:, ks:ks + BK] = f32(
                pvs[:, ks:ks + BK] + f32(a_h @ bl.T))     # + l1
            pvs[:, ks:ks + BK] = f32(
                pvs[:, ks:ks + BK] + f32(a_l @ bh.T))     # + l2
        if park_bf16:
            pvs = bf16_widen(bf16_quant(pvs)).astype(np.float32)
        dw = f32(dw + pvs)
    return dw


def run(shape, seed=0):
    B, T, H, K_, V_ = shape
    data = make_data(B, T, H, K_, V_, seed)
    lay = layout(data, B, T, H, K_, V_, C)
    md = lay["_meta"]
    NC, BH = md["NC"], md["BH"]

    def wch(u16, W):
        return bf16_widen(u16).reshape(BH, NC, C, W)

    q, k, v = wch(lay["q"], K_), wch(lay["k"], K_), wch(lay["v"], V_)
    do = wch(lay["do"], V_)
    beta = lay["beta"].astype(np.float64).reshape(BH, NC, C)
    g = (lay["g_log2"].astype(np.float64) / RCP_LN2).reshape(BH, NC, C)
    h0 = bf16_widen(lay["h0"]).reshape(BH, K_, V_)
    dht = bf16_widen(lay["dht"]).reshape(BH, K_, V_)
    scale = 1.0 / K_ ** 0.5

    m_inc = np.tril(np.ones((C, C)))
    m_str = np.tril(np.ones((C, C)), -1)
    outs = {vn: {o: np.zeros((BH, NC, C, Wn))
                 for o, Wn in (("dk", K_), ("dbeta", 1), ("dg", 1))}
            for vn in ("exact", "current", "wpark_f32", "wpark_bf16")}
    dW_stats = {vn: 0.0 for vn in outs}
    dw_ref_mag = 0.0

    for bh in range(BH):
        # ---- shared forward (the twin's) --------------------------------
        S = h0[bh].copy()
        h_all = np.zeros((NC, K_, V_))
        T_all = np.zeros((NC, C, C))
        W_all = np.zeros((NC, C, K_))
        vn_all = np.zeros((NC, C, V_))
        for n in range(NC):
            gc = np.cumsum(g[bh, n])
            kkT = k[bh, n] @ k[bh, n].T
            A = np.tril(beta[bh, n][:, None] * kkT *
                        np.exp(gc[:, None] - gc[None, :]), -1)
            Tm = np.eye(C)
            for i in range(1, C):
                Tm[i, :i] = -(A[i, :i] @ Tm[:i, :i])
            U = Tm @ (beta[bh, n][:, None] * v[bh, n])
            W = Tm @ (beta[bh, n][:, None] * k[bh, n]
                      * np.exp(gc)[:, None])
            vn = U - W @ S
            h_all[n] = S
            T_all[n], W_all[n], vn_all[n] = Tm, W, vn
            S = (S * np.exp(gc[-1])
                 + (k[bh, n] * np.exp(gc[-1] - gc)[:, None]).T @ vn)

        # ---- shared reverse walk (dH is dW-independent) -----------------
        dH = dht[bh].copy()
        for n in reversed(range(NC)):
            qc, kc, vc, bc = q[bh, n], k[bh, n], v[bh, n], beta[bh, n]
            gc = np.cumsum(g[bh, n])
            gcl = gc[-1]
            kkT = kc @ kc.T
            Sn = h_all[n]
            D = np.exp(gc[:, None] - gc[None, :])
            Tm = T_all[n]
            vn = vn_all[n]
            P = np.tril((qc @ kc.T) * D, 0)
            Bhat = kc * np.exp(gcl - gc)[:, None]
            Qg = qc * np.exp(gc)[:, None]
            Kbg = bc[:, None] * kc * np.exp(gc)[:, None]
            Vb = bc[:, None] * vc
            dO = do[bh, n] * scale
            dPm = (dO @ vn.T) * D * m_inc
            dvp = P.T @ dO + Bhat @ dH
            dq1 = (dO * np.exp(gc)[:, None]) @ Sn.T

            # dW variants (dvp/Sn widened f32 for the kernel-side sims)
            dvp32, Sn32 = f32(dvp), f32(Sn)
            dws = {
                "exact": -(dvp @ Sn.T),
                "current": -dW_current(dvp32, Sn32, K_, V_).astype(
                    np.float64),
                "wpark_f32": -dW_wpark(dvp32, Sn32, K_, V_,
                                       False).astype(np.float64),
                "wpark_bf16": -dW_wpark(dvp32, Sn32, K_, V_,
                                        True).astype(np.float64),
            }
            dw_ref_mag = max(dw_ref_mag, float(np.abs(dws["exact"]).max()))
            for vn_, dw_ in dws.items():
                dW_stats[vn_] = max(dW_stats[vn_],
                                    float(np.abs(dw_ - dws["exact"]).max()))
                G = dvp @ Vb.T + dw_ @ Kbg.T
                TW = Tm.T @ dw_
                dAf = -(Tm.T @ (G * m_str) @ Tm.T) * D * m_str
                dk = ((dAf @ kc) * bc[:, None]
                      + dAf.T @ (kc * bc[:, None])
                      + TW * (bc * np.exp(gc))[:, None]
                      + (vn * np.exp(gcl - gc)[:, None]) @ dH.T
                      + dPm.T @ qc)
                dbeta = ((dAf * kkT).sum(1)
                         + (TW * (kc * np.exp(gc)[:, None])).sum(1)
                         + ((Tm.T @ dvp) * vc).sum(1))
                dgr = (dAf * kkT * bc[:, None])
                dgr = dgr.sum(1) - dgr.sum(0)
                dgr += (TW * Kbg).sum(1)
                dgr += (dq1 * qc).sum(1)
                PdP = (dO @ vn.T) * P
                dgr += PdP.sum(1) - PdP.sum(0)
                rowB = ((vn * np.exp(gcl - gc)[:, None]) @ dH.T
                        * kc).sum(1)
                dg5a = np.exp(gcl) * (dH * Sn).sum()
                suf = np.cumsum(dgr[::-1])[::-1]
                pre = np.concatenate([[0.0], np.cumsum(rowB)[:-1]])
                outs[vn_]["dk"][bh, n] = dk
                outs[vn_]["dbeta"][bh, n, :, 0] = dbeta
                outs[vn_]["dg"][bh, n, :, 0] = suf + pre + dg5a
            dH = np.exp(gcl) * dH + Qg.T @ dO - W_all[n].T @ dvp

    res = {"shape": list(shape), "|dW|max": dw_ref_mag,
           "dW_delta_vs_exact": dW_stats, "graded": {}}
    for vn_ in outs:
        for o in ("dk", "dbeta", "dg"):
            d = float(np.abs(outs[vn_][o] - outs["current"][o]).max())
            res["graded"][f"{vn_}|{o}|vs_current"] = d
            de = float(np.abs(outs[vn_][o] - outs["exact"][o]).max())
            res["graded"][f"{vn_}|{o}|vs_exact"] = de
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1,2048,16,128,128")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    shape = tuple(int(x) for x in args.shape.split(","))
    res = run(shape)
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
