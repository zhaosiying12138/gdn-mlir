#!/usr/bin/env python3
"""G5-X round 19 (R19-bwd): fwdscan K-split-across-CTAs -- CPU twin + verdict.

THE LEVER (the r18/r19 ranking's "only big lever, new numerics class"):
fwdscan is 0.373/1.475 ms (T2048/T8192), 2 CTAs/SM blocked STRUCTURALLY
(pool bar, the r16 verdict).  The K-split variant parallelizes the SCAN
STATE instead of the residency: split the [K, BV] f32 state S across TWO
co-resident CTAs by K-rows (each owns K/2 rows of S + the matching k/w
column slices), exchanging one [C, BV] f32 partial per chunk.

THE MATH (the derivation this script verifies).  Per chunk n the kernel
walks, in f32, exactly:

    h_all[n] = S                                  (store)
    acc       = w_n @ S                           (K-contraction,
                                                  BK=64 slice-chained)
    vn        = u_n - acc                         -> vnew_all[n]
    S        *= 2^gcl                             (element-wise)
    kg        = k_n * 2^(gcl - gc_i)              ([C, K], f32 scale)
    S        += kg^T @ vn                         (C-contraction)

i.e. S' = M S + b with M = 2^gcl I - kg^T w, b = kg^T u -- but M is
NEVER formed; the three-factor walk is the kernel.  Decompose over
K-row halves R_0 = [0, K/2), R_1 = [K/2, K):

  * factor 2 (acc):      w S = sum_h w[:, R_h] @ S[R_h]  -- the ONE
    cross-K coupling.  Each CTA folds its half; the [C, BV] partials
    are exchanged; acc = partial_0 + partial_1 (fixed order).
  * factor 4 (decay):    element-wise -- row-split, no coupling.
  * factor 6 (kg^T vn):  output rows R_h need ONLY kg[:, R_h] and the
    SHARED vn -- row-split, no coupling.  (The mission's "C x C
    products stay replicated or split by the k-halves": on the GPU the
    w/kg tiles are [C, K/2] slices, exactly half the bytes; vn is
    replicated by the exchange.)

So in EXACT arithmetic the split walk is the identity.  In f32 the ONLY
change is the acc association: the fused kernel folds slice 0 then
folds slice 1 INTO it ((((A + b1) + b2) + ...), BK=64 = K/2 so each CTA
folds exactly ONE slice), while the split folds each slice from 0 and
adds the partials once (A + ((b1 + b2) + ...)) -- the classic
re-association, 2^-24-class per element.  Everything else (values,
element-wise ops, the kg^T vn fold order per S row) is IDENTICAL.

VERDICT CRITERIA:
  GO needs (1) the fp64 split walk == the fp64 sequential walk at
  machine precision (the identity check), and (2) the f32 split walk's
  error vs the fp64 reference the same class as the f32 sequential
  walk's (the re-association does not amplify the scan's error growth
  over NC chunks).

Usage: python tools/gdn_fwdscan_ksplit.py [--T 2048]
Prints the tables + VERDICT.  Self-contained (no GPU).
"""
import argparse
import numpy as np

RCP_LN2 = np.float64(1.0) / np.log(np.float64(2.0))


def make_inputs(B, T, H, K, V, C, seed=1234):
    """The gdn_bwd_verify.make_data distribution classes (documented
    stand-in): l2-normalized k (bf16-quantized like the chain), sigmoid
    beta, -softplus g, 0.1-normal h0."""
    rng = np.random.default_rng(seed)
    NC = T // C

    def l2n(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    k = l2n(rng.standard_normal((B, T, H, K))).astype(np.float32)
    v = rng.standard_normal((B, T, H, V)).astype(np.float32)
    beta = (1 / (1 + np.exp(-rng.standard_normal((B, T, H))))).astype(np.float32)
    g = (-np.logaddexp(0.0, 0.5 * rng.standard_normal((B, T, H)))).astype(np.float32)
    h0 = (0.1 * rng.standard_normal((B, H, K, V))).astype(np.float32)

    def chunkify(x, W):
        pad = (-T) % C
        if pad:
            x = np.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
        return x.reshape(B, H, NC, C, W)

    kc = chunkify(k, K)                    # bf16 inputs, like the chain
    ku = (kc.view(np.uint32) >> 16).astype(np.uint16)
    k64 = (ku.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    vc = chunkify(v, V)
    bc = chunkify(beta, 1)[..., 0].astype(np.float64)
    gc = (chunkify(g, 1)[..., 0].astype(np.float64) * RCP_LN2)
    g32 = gc.astype(np.float32)            # the chain stores g_log2 f32
    h0_64 = h0.astype(np.float64)
    h0_32 = (h0.view(np.uint32) >> 16).astype(np.uint16)
    h0_32 = (h0_32.astype(np.uint32) << 16).view(np.float32)
    return k64, vc.astype(np.float64), bc, g32, h0_32, NC


def wy_solve(kn, vn, betan, gn64):
    """The wy kernel's per-chunk Tm/U/W solve, fp64 (bwd_twin's math;
    U/W are the chain's wy outputs -- f32 in the chain, rounded by the
    caller)."""
    C, K = kn.shape
    gc = np.cumsum(gn64)
    kkT = kn @ kn.T
    A = np.tril(betan[:, None] * kkT * np.exp(gc[:, None] - gc[None, :]), -1)
    Tm = np.eye(C)
    for i in range(1, C):
        Tm[i, :i] = -(A[i, :i] @ Tm[:i, :i])
    U = Tm @ (betan[:, None] * vn)
    W = Tm @ (betan[:, None] * kn * np.exp(gc)[:, None])
    return U, W


def fold_gemm(A, B, k0, k1, acc=None, dt=np.float32):
    """k-ascending STRICT left fold of A[:, k0:k1] @ B[k0:k1, :] in dt
    (the scalar gemm emission's order).  acc != None folds INTO acc:
    (((acc + b1) + b2) + ...) -- the acc=True slice chain.  Emulated
    exactly with np.add.accumulate (sequential, no pairwise) over an
    explicit outer product [rows, k, cols]."""
    P = (A[:, k0:k1, None] * B[None, k0:k1, :]).astype(dt)  # [M, k', N]
    if acc is not None:
        P = np.concatenate([acc[:, None, :].astype(dt), P], axis=1)
    return np.add.accumulate(P, axis=1)[:, -1, :]


def walk(kn, vn_all, bn, gn32, h0, NC, C, K, V, BV, dt, nsplit=1):
    """One fwdscan walk over NC chunks for ONE (bh, vt) view (BV of the
    V columns), simulating the kernel's op order at precision dt.
    nsplit=1: the sequential kernel (BK=64 slice chain).  nsplit=2/4:
    the K-split -- each virtual CTA owns K/nsplit S rows, folds its acc
    slice(s) from 0, partials sum in FIXED h-ascending pairwise-tree
    order (the 2-CTA design's single add is the nsplit=2 line)."""
    S = h0[:, :BV].astype(dt).copy()       # [K, BV]
    h_all = np.zeros((NC, K, BV), dt)
    vn_out = np.zeros((NC, C, BV), dt)
    BK = 64
    for n in range(NC):
        U, W = wy_solve(kn[n], vn_all[n], bn[n], gn32[n].astype(np.float64))
        # the chain's f32 staging of the wy outputs
        u = U[:, :BV].astype(dt)
        w = W.astype(dt)
        k = kn[n].astype(np.float32).astype(dt)
        # the kernel's own serial gc cumsum at dt (gc_cumsum chain)
        gc = np.add.accumulate(gn32[n].astype(dt))
        gcl = gc[-1]
        h_all[n] = S
        if nsplit == 1:
            acc = fold_gemm(w, S, 0, BK, dt=dt)
            for kk in range(BK, K, BK):     # the slice chain, acc=True
                acc = fold_gemm(w, S, kk, kk + BK, acc=acc, dt=dt)
        else:
            parts = []
            for h in range(nsplit):
                # each CTA's half = ONE fold from 0 (K/nsplit <= BK=64
                # for every split of K=128; larger halves would chain
                # BK slices inside the CTA, same as the nsplit=1 line)
                r0 = h * K // nsplit
                r1 = r0 + K // nsplit
                parts.append(fold_gemm(w, S, r0, r1, dt=dt))
            lvl = parts                     # pairwise fixed-order tree
            while len(lvl) > 1:
                lvl = [lvl[i] + lvl[i + 1] for i in range(0, len(lvl), 2)]
            acc = lvl[0]
        vn = u - acc
        vn_out[n] = vn
        # S *= 2^gcl -- the kernel's exp2f at dt
        dlx = np.exp2(np.array(gcl, dtype=dt))
        S = S * dlx
        # expc[i] = 2^(gcl - gc[i]) at dt; kg = k * expc (f32 mulf)
        expc = np.exp2((gcl - gc).astype(dt))
        kg = k * expc[:, None].astype(dt)
        # the S-update's C-fold: c-ascending per S row (the tgemm order)
        P = (kg.T[:, :, None] * vn[None, :, :]).astype(dt)  # [K, C, BV]
        add = np.add.accumulate(P, axis=1)[:, -1, :]
        S = S + add
    return h_all, vn_out


def relerr(x, ref):
    d = np.abs(x.astype(np.float64) - ref.astype(np.float64))
    scale = max(np.abs(ref.astype(np.float64)).max(), 1e-30)
    return d.max() / scale, d.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--BV", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    K, V, C, BV = args.K, args.V, args.C, args.BV

    k64, v64, b64, g32, h0_32, NC = make_inputs(
        args.B, args.T, args.H, K, V, C, args.seed)
    # one (bh, vt) view: the state walk the kernel runs per CTA
    kn, vn, bn, gn = k64[0, 0], v64[0, 0], b64[0, 0], g32[0, 0]
    h0n = h0_32[0, 0]                      # [K, V]

    print(f"fwdscan K-split CPU twin: T={args.T} NC={NC} K={K} V={V} "
          f"C={C} BV={BV} (one bh/vt view; seed {args.seed})")

    # ---- (1) identity check: fp64 sequential vs fp64 split ------------
    h_seq64, vn_seq64 = walk(kn, vn, bn, gn, h0n.astype(np.float64),
                             NC, C, K, V, BV, np.float64, nsplit=1)
    results = {}
    for ns in (2, 4):
        h_sp, vn_sp = walk(kn, vn, bn, gn, h0n.astype(np.float64),
                           NC, C, K, V, BV, np.float64, nsplit=ns)
        eh = relerr(h_sp, h_seq64)
        ev = relerr(vn_sp, vn_seq64)
        results[f"fp64/{ns}"] = (eh[0], ev[0])
        print(f"  [identity, fp64] split-{ns}: h_all rel {eh[0]:.3e} "
              f"(abs {eh[1]:.3e})  vnew rel {ev[0]:.3e}")

    # ---- (2) f32 re-association cost -----------------------------------
    h_seq32, vn_seq32 = walk(kn, vn, bn, gn, h0n, NC, C, K, V, BV,
                             np.float32, nsplit=1)
    e_seq_h = relerr(h_seq32, h_seq64)
    e_seq_v = relerr(vn_seq32, vn_seq64)
    print(f"  [f32 sequential vs fp64] h_all rel {e_seq_h[0]:.3e}  "
          f"vnew rel {e_seq_v[0]:.3e}   (the SHIPPED kernel's class)")
    for ns in (2, 4):
        h_sp, vn_sp = walk(kn, vn, bn, gn, h0n, NC, C, K, V, BV,
                           np.float32, nsplit=ns)
        e_sp_h = relerr(h_sp, h_seq64)
        e_sp_v = relerr(vn_sp, vn_seq64)
        d_h = relerr(h_sp, h_seq32)
        d_v = relerr(vn_sp, vn_seq32)
        print(f"  [f32 split-{ns} vs fp64]     h_all rel {e_sp_h[0]:.3e} "
              f"({e_sp_h[0]/e_seq_h[0]:.2f}x seq)  vnew rel {e_sp_v[0]:.3e} "
              f"({e_sp_v[0]/e_seq_v[0]:.2f}x seq)")
        print(f"  [f32 split-{ns} vs f32 seq]  h_all rel {d_h[0]:.3e}"
              f"   vnew rel {d_v[0]:.3e}   (the PURE re-association)")
        results[f"f32/{ns}"] = (e_sp_h[0], e_sp_v[0])

    # ---- verdict --------------------------------------------------------
    ident_ok = all(r[0] < 1e-12 and r[1] < 1e-12
                   for kk, r in results.items() if kk.startswith("fp64"))
    ampl2 = results["f32/2"][0] / e_seq_h[0]
    ampl2v = results["f32/2"][1] / e_seq_v[0]
    verdict = "GO" if ident_ok and ampl2 < 3.0 and ampl2v < 3.0 else "NO-GO"
    print(f"\nVERDICT: {verdict}")
    print(f"  identity (fp64): "
          f"{'MACHINE-PRECISION EXACT' if ident_ok else 'BROKEN'}")
    print(f"  2-way f32 error amplification vs sequential: "
          f"h_all {ampl2:.2f}x  vnew {ampl2v:.2f}x")
    print("  (the downstream 6-grad tolerances sit 4 orders above these")
    print("   maxima; the GPU port must still pass the 6/6 chain verify)")


if __name__ == "__main__":
    main()
