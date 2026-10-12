# RC1: B/T/H ladder -- predictive pricing of the bwd regime (CPU-only, pre-GPU)

Regime-change campaign round 1 (user override of the bwd-hold, 2026-10-08).
Question on trial: does bwd's ours-vs-fla ratio improve at larger shapes
(82-SM fill, staged-traffic amortization, fla's per-shape autotune tax),
and does any regime reach >=1.1x?  This document prices the ladder on CPU
BEFORE any GPU work, from the code's grid/tile structure plus the two
measured ship anchors.  Tool: `tools/rc1_price.py` (staging copy
`.gdn-mlir-staging/rc1/rc1_price.py`), raw table `/tmp/rc1_price.json`.

## 1. Scaling-law derivation (per stage, from tools/gdn_bwd_gen.py)

Notation: `NC = T/64` (chunk C=64), `M = B*H*NC` (chunk-rows),
`u = M/512` = work multiple of the B1T2048H16 anchor.

Grid formulas (manifest + gen dispatch, HEAD f09a673):

| stage | grid | class | occupancy (CTA/SM, smem-limited) |
|---|---|---|---|
| wy, caryA, caryM, bpre, dkb, dqkwg, wyb | `M` | per-chunk parallel | wy/dqkwg/wyb 1; caryA/caryM/bpre/dkb 2 |
| fwdscan | `B*H*(V/32)` = `B*H*4` | **serial NC-chain per CTA** | 1 |
| dhs | `B*H*(V/16)` = `B*H*8` | **serial NC-chain per CTA** | 1 (49152 B pool at BV=16) |

Consequences:

* **Per-chunk stages** are workload-linear in `u`; wave quantization
  (grid/(82*occ) waves) improves only at the margin (6.2 -> 25 waves from
  u=1 to u=4, i.e. a <4% tail effect).  Footprint (all globals scale with
  M) drives an L2 penalty that makes the *marginal* rate per u slightly
  WORSE than the u=1 average: measured T-direction stage ratios at 4x
  work are 3.3-3.4x for the mma stages (dkb 3.35, dqkwg 3.39, wyb 3.34)
  but **5.9-6.2x for caryA/caryM/bpre** (small-footprint stages whose
  u=1 time is L2-boosted).
* **Serial stages** (fwdscan, dhs) scale with chain length NC at fixed
  B,H: measured 4.08x / 3.91x at 4x T.  B/H direction: chain length
  unchanged, makespan = ceil(grid/82) waves of the same chain (fwdscan
  B1H16 grid 64 = 0.78 waves -- 18 SMs idle; H32/B>=2 fill it, a ONE-TIME
  ~1.28x utilization bonus on 0.64 ms of the chain, then linear).
* Launch overhead: 9 kernels x ~3 us ~= 27 us fixed, ~1.3% of u=1 e2e.

Measured ship anchors (co-loaded window, r19-final bytes; per-stage from
r22b_p11r1k32_base / r21b_vtf_T8192 base runs):

| stage | u=1 (ms) | u=4 (ms) | ratio | affine a,b (ms) |
|---|---|---|---|---|
| wy | 0.2224 | 0.7984 | 3.59 | -0.031, 0.254 |
| fwdscan | 0.3713 | 1.5155 | 4.08 | -0.077, 0.381 |
| caryA | 0.0531 | 0.3103 | 5.85 | -0.033, 0.086 |
| caryM | 0.0792 | 0.3852 | 4.86 | -0.023, 0.102 |
| bpre | 0.0499 | 0.3088 | 6.19 | -0.037, 0.086 |
| dhs | 0.2703 | 1.0571 | 3.91 | -0.022, 0.262 |
| dkb | 0.1803 | 0.6024 | 3.34 | -0.027, 0.141 |
| dqkwg | 0.5146 | 1.7436 | 3.39 | 0.032, 0.430 |
| wyb | 0.3383 | 1.1300 | 3.34 | 0.043, 0.264 |
| **e2e** | **2.14 (min)** | **7.75 (min)** | **3.62** | **0.270, 1.837** |

fla bwd (derived fwdbwd-fwd, clean window, GATE_REPORT_FULL_REMEASURE):
1.1006 @u=1, 2.8379 @u=4 -> affine **0.522 + 0.579*u**.

## 2. The predicted curve (verdict-grade hypothesis)

```
ratio(u) = fla(u)/ours(u) = (0.522 + 0.579 u) / (0.270 + 1.837 u)
u:      1      2      4      8     16    -> inf
ratio: 0.514  0.419  0.366  0.338  0.324 -> 0.310
```

**The prediction is the OPPOSITE of the mission's hope: the ratio
DECLINES monotonically with B*T*H.**  fla's sublinearity (fixed ~0.52 ms)
props up OUR ratio at small shapes; ours' marginal cost per u is
1.837 ms vs fla's 0.579 ms (**3.17x per-token**), so every step up the
ladder makes the comparison worse, not better.  The 82-SM fill and
wave-tail amortization recover at most ~0.1-0.15 ms one-time (the
fwdscan/dhs underfill at B1H16), while the L2-footprint effect on
caryA/caryM/bpre actively raises our marginal rate.

### The floor argument (why >=1.1x is structurally closed at u >= 1)

r7 static analysis: sum of per-stage DRAM floors = 1.074 ms @u=1, and
ALL stage traffic scales with M -> floor(u) = 1.074*u ms (at the
co-load 824.6 GB/s calibration; both sides scale together in a clean
window).  Then:

* at u=1: floor 1.074 > fla measured 1.03-1.10 -- a PERFECTLY
  roofline-executed f32 staged chain merely TIES fla at the anchor;
* the ceiling ratio at roofline perfection is
  (0.522+0.579u)/(1.074u): 0.81 at u=1, 0.65 at u=4, **0.539
  asymptotically**;
* to reach ratio 1.1 anywhere with u >= 1 requires our slope <= 0.526
  ms/u -- **2.04x below the f32-traffic floor**.  Not an execution
  problem: the bytes have to halve (bf16 intermediates ~2x, gated on the
  error budget, or fewer passes via fusion).

The only regime where >=1.1x is even plausible is SMALL B*T*H (u << 1),
where both sides are fixed-cost-bound: ours T512H4 measured 0.499 ms
(r16-y) vs fla small-shape fixed ~0.4-0.6 ms (its 0.52 ms intercept
includes ~10 kernel launches + autograd elementwise chains).  A bonus
paired probe at B1T512H4 is included in the measured ladder to anchor
the small end of the curve.

## 3. Ranked cells + VRAM feasibility (union-of-globals exact)

Union globals: 32 M-row symbols (722 KiB/M-row) + 3 BH-row symbols
(128 KiB/BH-row: h0/dht bf16, dh0 f32) + ~0.9 GiB harness/context.
VRAM rule: <6 GB per run.

| cell | u | ours pred | fla pred | ratio pred | VRAM GiB | feasible |
|---|---|---|---|---|---|---|
| B1T2048H16 (anchor) | 1 | 2.14 | 1.10 | 0.514 | 1.25 | yes |
| B1T2048H32 | 2 | 4.01 | 1.68 | 0.419 | 1.61 | yes |
| B2T2048H16 | 2 | 4.01 | 1.68 | 0.419 | 1.61 | yes |
| B1T8192H16 | 4 | 7.75 | 2.84 | 0.366 | 2.31 | yes |
| B4T2048H16 | 4 | 7.75 | 2.84 | 0.366 | 2.32 | yes |
| B1T16384H16 | 8 | 15.23 | 5.15 | 0.338 | 3.72 | yes |
| any u=16 cell (M=8192) | 16 | 30.19 | 9.79 | 0.324 | 6.5 | **NO (<6G rule)** |
| any u=32 cell | 32 | 60.11 | 19.05 | 0.317 | 12.2 | NO |
| u>=64 | 64+ | -- | -- | ->0.311 | 23+ | NO |

The u>=16 rungs of the mission ladder (T32768, B8T8192H32, ...) are all
M=8192+ cells at 6.5+ GiB -- excluded by the <6 GB coordination rule
(and co-job ~8.7 GB resident).  The curve is well-constrained by
u in {1,2,4,8}: predicted 0.514/0.419/0.366/0.338 vs asymptote 0.310.

### Measured-ladder cell selection (top-4 predicted + anchor + small-end)

1. **B1T2048H16** -- anchor; cubins must reproduce the r19-final md5s.
2. **B1T2048H32** -- best predicted non-anchor cell; H-direction (tests
   the serial-stage fill bonus: fwdscan 64->128, dhs 128->256 CTAs).
3. **B4T2048H16** -- u=4 B-direction; current chain never measured at
   B4 (r13 data is 6 rounds of levers stale); serial-wave regime.
4. **B1T8192H16** -- u=4 T-direction re-anchor with fresh interleaved fla.
5. **B1T16384H16** -- u=8 far end; NEW NC=256 class -> the numerics
   verification cell (JAX fp64 ref extension + 6/6 twin verify + det).
6. *(bonus)* **B1T512H4** -- u=1/16 small end; the only regime where
   ratio >= 1 is plausible; anchors the curve's left side.

## 4. fla-side regime facts carried into the measurement design

* Autotune tax (G6/R26): cold fwd 16.6 s + cold bwd 23.3 s @B1T2048
  (25 cfg/5 kernels fwd, 23 cfg/4 kernels bwd); fresh-process cache-hit
  5.67 s strict+cold-compile, 0.65 s strict+warm.  The tax is PER SHAPE
  and grows with shape (each config-bench call scales with T); it is
  part of the regime story (cold-start serving) but NOT of the warm
  paired ratio -- both recorded.
* fla bwd kernel anatomy @B1T2048 (profiler attribution): dqkwg 137.6 us,
  dhu 132.5, prepare_wy_repr_bwd 96.3, fwd-h recompute 76.2, elementwise
  Mul 56.7, recompute_w_u 30.8, dv_local 25.6, + remainder -> ~10-12
  launches.  Its grids parallelize over B*H*(V-blocks); same M-scaling.

## 5. What the ladder must decide

1. Sign of the trend: measured ratio(u) at u in {1/16, 1, 2, 4T, 4B, 8}
   vs the predicted decline 0.514 -> 0.338.
2. Direction split at u=4 (B4T2048 vs B1T8192): the serial-stage wave
   quantization + fill bonus predicts B4 within +-5% of T8192.
3. The asymptote estimate: refit ours/fla marginals on the measured
   u=1..8 rungs; compare to 1.837/0.579 = 3.17x per-token.
4. The small end: B1T512H4 ratio >= 1.1 or not -- the only candidate
   regime for a win, sizing the round-2 "small-shape serving" option
   against the traffic-halving (bf16 intermediates) program.
