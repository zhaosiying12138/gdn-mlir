#!/usr/bin/env python3
"""RC1 (regime-change r1) -- CPU-only predictive pricing of the bwd B/T/H ladder.

Model per stage (from tools/gdn_bwd_gen.py grid/tile structure + the two
measured ship anchors B1T2048H16 / B1T8192H16):

  NC = T/64 (chunk C=64);  M = B*H*NC (total chunk-rows);  u = M/512
    (= the work multiple of the B1T2048H16 anchor)

  grid formulas (manifest r21b_base_T2048 + gen dispatch):
    wy,caryA,caryM,bpre,dkb,dqkwg,wyb : grid = M        (per-chunk parallel)
    fwdscan : grid = B*H*4   (NV = V/32; serial NC-chain per CTA)
    dhs     : grid = B*H*8   (BV=16 ship;  serial NC-chain per CTA)

  per-chunk stages: t_s(u) = a_s + b_s*u   (affine fit on the 2 anchors;
      b_s is the DRAM-bound marginal, a_s the launch+fill residual)
  serial stages:    T-direction multiplier from the anchors (includes the
      L2 per-chunk growth); B/H-direction = ceil(grid/82) wave count
      (each CTA's chain length is NC, unchanged by B/H)

fla bwd model: affine on the same u from the clean-window derived series
(B1T2048 1.1006, B1T8192 2.8379): fla(u) = 0.5215 + 0.5791*u.  A second
basis (co-load window, the r18-r22 'fla 1.03' convention) is reported
alongside for the u=1 anchor.

VRAM: exact union of the manifest globals template (per-M-row bytes),
recomputed per cell; harness overhead 0.9 GiB assumed (CUDA context +
python + staging).

Outputs a ranked cell table (predicted ours/fla ratio) + VRAM feasibility
to stdout; JSON to --out.
"""
from __future__ import annotations

import argparse
import json
import math
import re

# ---- measured ship anchors (r21b/r20/r19 records, co-loaded window) -------- #
# B1T2048H16 per-stage (min-basis, r22b_p11r1k32_base jsons)
A1 = {"wy": 0.2224, "fwdscan": 0.3713, "caryA": 0.0531, "caryM": 0.0792,
      "bpre": 0.0499, "dhs": 0.2703, "dkb": 0.1803, "dqkwg": 0.5146,
      "wyb": 0.3383}
E2E1_MIN = 2.14      # ours bwd min @u=1 (co-loaded)
# B1T8192H16 per-stage (r21b_vtf_T8192 base jsons)
A4 = {"wy": 0.7984, "fwdscan": 1.5155, "caryA": 0.3103, "caryM": 0.3852,
      "bpre": 0.3088, "dhs": 1.0571, "dkb": 0.6024, "dqkwg": 1.7436,
      "wyb": 1.1300}
E2E4_MIN = 7.75

SM = 82
# fla bwd derived clean series (GATE_REPORT_FULL_REMEASURE: fwdbwd - fwd)
FLA_1, FLA_4 = 1.1006, 2.8379
FLA_B, FLA_A = (FLA_4 - FLA_1) / 3.0, FLA_1 - (FLA_4 - FLA_1) / 3.0
# ours e2e affine on the same two points
OURS_B, OURS_A = (E2E4_MIN - E2E1_MIN) / 3.0, E2E1_MIN - (E2E4_MIN - E2E1_MIN) / 3.0

SERIAL = {"fwdscan": 4, "dhs": 8}     # NV divisor (V/BV)
# occupancy (CTA/SM) from smem pools: r21b manifest smem bytes
OCC = {"wy": 1, "fwdscan": 1, "caryA": 2, "caryM": 2, "bpre": 2,
       "dhs": 1, "dkb": 2, "dqkwg": 1, "wyb": 1}


def stage_time(s, B, T, H):
    """Predicted stage ms.  Per-chunk stages: affine in u (T-fit; the same
    marginal applies to B/H since footprint scales with M either way).
    Serial stages: T-anchor affine * B/H wave-count multiplier."""
    NC = T // 64
    u = B * H * NC / 512.0
    a = A1[s] - (A4[s] - A1[s]) / 3.0
    b = (A4[s] - A1[s]) / 3.0
    if s in SERIAL:
        # anchor direction is T; the fit already carries chain length NC.
        # B/H: chain length unchanged -> makespan scales with wave count
        # rel. to the anchor's (B1H16, already >= 1 wave at H16 for dhs,
        # 0.78 wave for fwdscan -- use the exact anchor wave counts).
        waves_anchor = max(1, math.ceil(1 * 16 * SERIAL[s] / SM))
        waves = max(1, math.ceil(B * H * SERIAL[s] / SM))
        # per-chain time at anchor NC: t_chain = anchor_t / waves_anchor
        t_chain_anchor = A1[s] / waves_anchor
        t_chain4 = A4[s] / waves_anchor      # T8192: same grid -> same waves
        # chain time grows with NC per the T-anchor marginal:
        t_chain = t_chain_anchor + (t_chain4 - t_chain_anchor) * (NC / 32.0 - 1) / 3.0
        return waves * max(t_chain, 1e-6)
    return max(a + b * u, 0.0)


def manifest_template_bytes():
    """Union bytes per M-row and per BH-row from the r21b T2048 manifest."""
    import os
    mf = os.path.expanduser(
        "~/codebase/gdn-mlir/evidence/G5/scratch/r21b_base_T2048/"
        "bwd_manifest.json")
    man = json.load(open(mf))
    m_rows = {}    # symbol -> bytes per M-row
    bh_rows = {}   # symbol -> bytes per BH-row (h0/dht/dh0)
    M = man["cfg"]["B"] * man["cfg"]["H"] * man["cfg"]["NC"]
    BH = man["cfg"]["B"] * man["cfg"]["H"]
    for st in man["stages"]:
        for sym, ty in st["globals"].items():
            mt = re.match(r"memref<(\d+)x(\d+)x(f32|bf16|f64)>$", ty)
            if not mt:
                continue
            rows, cols, dt = int(mt.group(1)), int(mt.group(2)), mt.group(3)
            wid = 4 if dt == "f32" else 2
            if rows == M:
                m_rows[sym] = max(m_rows.get(sym, 0), cols * wid)
            elif rows == BH:
                bh_rows[sym] = max(bh_rows.get(sym, 0), cols * wid)
            else:
                raise AssertionError((sym, ty))
    return m_rows, bh_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    per_row, per_bh = manifest_template_bytes()
    row_bytes = sum(per_row.values())
    bh_bytes = sum(per_bh.values())
    print(f"# union globals: {len(per_row)} M-row symbols "
          f"({row_bytes/1024:.0f} KiB/M-row) + {len(per_bh)} BH-row symbols "
          f"({bh_bytes/1024:.0f} KiB/BH-row)")

    cells = []
    for B in (1, 2, 4, 8):
        for T in (2048, 8192, 16384, 32768):
            for H in (16, 32):
                NC = T // 64
                M = B * H * NC
                u = M / 512.0
                stages = {s: round(stage_time(s, B, T, H), 4)
                          for s in A1}
                ours = OURS_A + OURS_B * u
                fla = FLA_A + FLA_B * u
                vram_gib = (M * row_bytes + B * H * bh_bytes) / 2**30 + 0.9
                cells.append({
                    "B": B, "T": T, "H": H, "NC": NC, "M": M, "u": round(u, 2),
                    "ours_pred_ms": round(ours, 3),
                    "fla_pred_ms": round(fla, 3),
                    "ratio_pred": round(fla / ours, 4),
                    "stages_pred": stages,
                    "stage_sum": round(sum(stages.values()), 3),
                    "vram_gib": round(vram_gib, 2),
                    "feasible_15g": vram_gib < 15.0,
                    "feasible_6g_rule": vram_gib < 6.0,
                })
    cells.sort(key=lambda c: -c["ratio_pred"])
    print(f"\n# ranked by predicted ratio (fla/ours, clean-derived basis);"
          f" ours-affine a={OURS_A:.3f} b={OURS_B:.3f} |"
          f" fla a={FLA_A:.3f} b={FLA_B:.3f} | asymptote"
          f" {FLA_B/OURS_B:.3f}")
    print(f"{'cell':22s} {'u':>5s} {'ours':>8s} {'fla':>7s} {'ratio':>6s}"
          f" {'VRAM':>6s} {'<6G':>4s}")
    for c in cells:
        nm = f"B{c['B']}T{c['T']}H{c['H']}"
        print(f"{nm:22s} {c['u']:5.1f} {c['ours_pred_ms']:8.2f}"
              f" {c['fla_pred_ms']:7.2f} {c['ratio_pred']:6.3f}"
              f" {c['vram_gib']:6.2f} {'ok' if c['feasible_6g_rule'] else 'NO':>4s}")
    if args.out:
        json.dump({"model": {
                      "ours_affine": [round(OURS_A, 4), round(OURS_B, 4)],
                      "fla_affine": [round(FLA_A, 4), round(FLA_B, 4)],
                      "asymptote_ratio": round(FLA_B / OURS_B, 4),
                      "per_row_bytes": row_bytes,
                      "anchors": {"u1": A1, "u4": A4,
                                  "e2e1_min": E2E1_MIN,
                                  "e2e4_min": E2E4_MIN}},
                   "cells": cells},
                  open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
