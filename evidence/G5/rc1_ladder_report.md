# RC1-m: measured B/T ladder -- the scaling-curve verdict

Regime-change campaign round 1, measured phase.  The predictions agent died
on rate limits after building the chains and starting the measurement
sweep (commit e3b932b); this session validated every salvaged artifact,
closed the remaining gates, and adjudicated the prediction.  Prediction
doc: `evidence/G5/rc1_scaling_predictions.md`.  Raw logs:
`evidence/G5/scratch/rc1_logs/` (ab x3 + cold + verify + det per cell),
chains `evidence/G5/scratch/rc1_<tag>/`.

Window: co-loaded (GPU job ~8.9-9.0 GiB resident, 16-28% util), fresh
process per measurement, interleaved paired ours-then-fla x3 per cell
(`scratch/rc1_abfla.sh`), 30-rep CUDA-event medians.  Paired ratios are
the currency; absolute times are window-inflated ~5-20% vs idle.

## 0. Gate record (all six cells)

| gate | result |
|---|---|
| payload provenance | 54/54 payload.mlir bit-identical to regeneration from HEAD `gdn_bwd_gen.py` (all 6 cells x 9 stages) |
| cubin lineage | B1T2048 and T8192: 9/9 md5 == r21b record dirs (r19-final bytes); T512H4 records are payload-only, closed by fresh gates below |
| numerics 6/6 | PASS at ALL 6 cells (fp64 twin, seed 0): B1T2048 / H32 / B4 / T8192 / T16384 / T512H4 -- worst grad dk 3.8e-3 @T16384 vs tol 8e-3 |
| determinism | det N=3 fresh-process, ALL stage outputs bit-identical: B1T2048, H32, B4, T16384 (salvaged runs, adjudicated here) + T512H4 (fresh) |
| dispatch | no gen changes needed: the r14/r16 NC/BH gates cover NC=256 (T16384) and BH=64 (B4); B4/T8192 share 7/9 cubin md5s (M-only stages), fwdscan/dhs differ (serial NC/BH) |
| VRAM | all runs < 6 GB (max cell T16384 ~3.7 GiB union); co-job pid 4076546 never touched |

The RC1 dh0 B>1 reshape fix (`gdn_bwd_verify.py`, twin emits flat [BH,K,V])
is exercised by the B4 cell: dh0 3.0e-7 PASS.

## 1. The measured curve

u = M/512, M = B*H*(T/64).  ours = 9-kernel chain e2e median (30 reps);
fla = autograd.grad-only median, `FLA_FLASH_QLA=0` pure Triton.

| cell | u | ours ms | fla_bwd ms | **ratio (meas)** | ratio (pred) | per-pair |
|---|---|---|---|---|---|---|
| B1T512H4 | 1/16 | 0.5573 | 1.2107 | **2.172** | -- (plausible-win) | 2.174 / 2.246 / 1.588 |
| B1T2048H16 | 1 | 2.1917 | 1.3319 | **0.608** | 0.514 | 0.729 / 0.474 / 0.610 |
| B1T2048H32 | 2 | 4.1539 | 1.6316 | **0.393** | 0.419 | 0.372 / 0.389 / 0.393 |
| B4T2048H16 | 4 | 7.9710 | 3.0757 | **0.386** | 0.419 (u=4: 0.366) | 0.410 / 0.360 / 0.304 |
| B1T8192H16 | 4 | 8.1377 | 3.2277 | **0.397** | 0.366 | 0.396 / 0.381 / 0.490 |
| B1T16384H16 | 8 | 16.1609 | 6.4045 | **0.396** | 0.338 | 0.399 / 0.469 / 0.390 |

Affine refits (least squares):

* u>=1 rungs: **ours(u) = 0.138 + 1.996u** (pred 0.270 + 1.837u),
  **fla(u) = 0.298 + 0.746u** (pred 0.522 + 0.579u);
  residuals: ours |r| <= 0.16 ms, fla |r| <= 0.29 ms -- the affine law
  holds.
* all 6 points: ours 0.259 + 1.973u, fla 0.654 + 0.680u.
* **marginal per-token cost ratio ours/fla = 2.67x** (pred 3.17x);
  asymptote ratio = 0.746/1.996 = **0.374** (pred 0.310).

## 2. The verdict on the prediction

**(1) Sign of the trend: VALIDATED.**  ratio(u) declines monotonically in
the medians from 0.608 @u=1 to ~0.39 @u>=2 and never reaches 1 anywhere
with u >= 1.  The mission's hope (ratio improving with shape) is refuted
exactly as predicted: fla's sublinear fixed cost props us up only at
small shapes; our marginal cost per unit work is 2.67x fla's.

**(2) Magnitude: the decline SATURATES above the prediction.**  Predicted
0.338 @u=8 and asymptote 0.310; measured plateau 0.386-0.397 across
u=2..8, refit asymptote 0.374.  Both slopes came in higher than
predicted in this co-load window (ours 1.996 vs 1.837, +9%; fla 0.746 vs
0.579, +29% -- fla's autograd elementwise chains suffer more under
co-load), and the ratio of marginals lands at 0.374-0.396.  The u=1
anchor: ours predicted 2.14 vs measured 2.19 (ours-side prediction
excellent); fla predicted 1.10 vs measured 1.33 (window).

**(3) Direction split at u=4: prediction CONFIRMED.**  B4T2048 7.971 ms
vs B1T8192 8.138 ms -- B-direction 2.0% faster, inside the predicted
+-5% band (M is the workload; serial-chain geometry only shifts dhs
0.940 vs 1.107 and fwdscan 1.557 vs 1.613 -- B4's 4x-shorter NC=32
chains with 4x more parallel chains fill the machine better).  One
predicted effect did NOT materialize: the H32 serial-stage fill bonus
(fwdscan 64->128 CTAs was to buy a one-time ~1.28x on the chain) --
measured fwdscan/dhs at H32 are exactly workload-linear (1.02/0.98);
the bonus is inside the window noise.

**(4) The floor argument: STRUCTURALLY CONFIRMED.**  r7 DRAM floor
1.074*u ms (f32 staged traffic, clean-window calibration).  Against THIS
window's fla: ceiling ratio = fla/floor = 0.72-0.76 at u=2..8; at u=1 in
a clean window (fla 1.03-1.10) the ceiling is 0.96-1.02 -- parity at
best for a PERFECTLY roofline-executed f32 chain.  Reaching 1.1x at u=8
needs ours <= 5.82 ms = 0.73 ms/u = **1.47x BELOW the f32-traffic
floor**: not an execution problem, the bytes must halve.  Separately,
ours' own execution gap is real: measured marginal 2.00 ms/u = 1.86x own
floor -- there is up to ~1.9x execution headroom, but even spending all
of it cannot cross 1.1 at u >= 2 (ceiling 0.72-0.76).

**(5) The small end: THE WIN IS REAL.**  B1T512H4 measures **2.172x**
(per-pair 2.17/2.25/1.59) -- ours 0.557 ms vs fla 1.211 ms.  This is the
one regime where the 9-kernel chain beats fla, and it is decisive, not
marginal: fla pays ~1.2 ms of fixed cost (10-12 kernel launches +
autograd elementwise chains) against our 0.56 ms fixed cost.  Crossover:
bracketed between u=1/16 (2.17) and u=1 (0.61); the u>=1 refit puts
u* ~ 0.13, the all-6 fit u* ~ 0.31 (M ~ 65-160 chunk-rows, i.e.
B1T~256-1024 H16-class cells).  Fit quality below u=1 is poor (only one
rung) -- quote the bracket, not the point.

## 3. Per-stage decomposition (the WHY, for round 2)

Scaling class per stage, measured x/u normalized vs the u=1 anchor
(1.00 = exactly workload-linear):

| stage | H32 u=2 | B4 u=4 | T8192 u=4 | T16384 u=8 | class |
|---|---|---|---|---|---|
| wy | 0.96 | 0.89 | 0.91 | 0.88 | mma-major, sublinear (wave tails amortize) |
| dkb | 0.88 | 0.82 | 0.82 | 0.81 | mma-major, sublinear |
| dqkwg | 0.89 | 0.84 | 0.84 | 0.98 | mma-major, sublinear -> linear at NC=256 |
| wyb | 0.88 | 0.83 | 0.83 | 0.98 | mma-major, sublinear -> linear at NC=256 |
| caryA | 1.39 | 1.37 | 1.37 | 1.36 | **superlinear: L2-footprint penalty** |
| caryM | 1.26 | 1.18 | 1.19 | 1.15 | superlinear (same) |
| bpre | 1.45 | 1.40 | 1.41 | 1.40 | superlinear (same) |
| fwdscan | 1.02 | 1.01 | 1.04 | **1.17** | serial NC-chain; thrash onset at NC=256 |
| dhs | 0.98 | 0.83 | 0.98 | 1.02 | serial NC-chain; B4 fill bonus |

The predictions doc's T-direction finding (caryA/caryM/bpre 5.9-6.2x at
4x work) GENERALIZES to the B and H directions: these small-footprint
stages are L2-resident at u=1 and fall out of L2 as the 32-symbol
M-row global footprint grows.  Their combined share is 9-11% at every
rung, so the superlinearity is visible but not dominant.

Stage shares at the ladder's ends (medians of the 3 ab runs):

* **best cell T512H4 (2.17x)**: dqkwg **40.0%**, fwdscan 16.6%, wyb
  11.2%, dhs 6.9% -- fixed-cost concentrated in dqkwg (its global park +
  arena machinery dominates when work is 1/16 of anchor).  The small-end
  lever is dqkwg's static cost, not traffic.
* **worst cell T16384 (0.396x)**: dqkwg 23.5%, fwdscan **20.5%**, wyb
  15.5%, dhs 13.1%, wy 9.2%, dkb 6.8%, caryA/M/bpre 11.4% -- the
  traffic-majority regime.  fwdscan is the fastest-growing stage
  (superlinear 1.17x at NC=256; the r19-r21 K-split family was closed
  at NC=32/128 -- NC=256 is new territory and reopens that door).

fla-side regime facts (recorded, part of the story):

* Cold autotune tax per shape, fresh process + fresh TRITON_CACHE_DIR:
  fwd 18.7-30.8 s, bwd 29.3-38.3 s, growing with T (T16384: 30.8+38.3 =
  69 s; G6's quieter-window B1T2048 record was 16.6+23.3 s).  Second
  call after cold: 1-20 ms.  Ours: AOT cubins, zero autotune, ~27 us
  launch overhead -- a cold-start serving win at EVERY shape.
* fla bwd series cross-check: grad-only median vs derived (fwdbwd-fwd)
  agree within 10-15% at every cell (e.g. T16384 6.40 vs 6.11) -- the
  b-series is the honest comparable; both are recorded in the logs.

## 4. What this means for "fully cover ALL fla-supported shapes and win"

The honest answer, now measured:

1. **bwd at u >= 1 (T2048-class and up at H16/H32): we lose, and the
   loss is structural.**  0.39-0.61x across the whole measured ladder,
   floor-closed at 0.72-0.76 ceiling even for roofline-perfect f32
   execution.  No tuning round closes this; the architecture's 9-kernel
   f32 staged chain moves ~2x the bytes per token fla's fused Triton
   autograd does (2.67x per-token cost measured, ~1.9x of that is
   execution above own floor, the rest is the traffic itself).
2. **bwd at u << 1 (small B*T*H): we win decisively** -- 2.17x at
   B1T512H4, crossover bracketed at u ~ 0.1-0.3.  Plus zero cold-start
   tax (fla pays 48-69 s per shape before first warm call).
3. **fwd at u >= 1: we already win** (standing r25 formal records:
   2.125x B1T2048 / 1.221x B4T2048 / 1.349x B1T8192; unchanged by this
   round -- no fwd knobs moved).
4. The u>=16 cells of the mission ladder (T32768, B8T8192H32, ...) are
   M=8192+ cells excluded by the <6 GB rule (6.5+ GiB) and would land on
   the same plateau (refit 0.38 at u=16) -- measuring them adds no
   decision information.

## 5. Round-2 design material (ranked by this ladder's evidence)

1. **Halve the bytes: bf16 the M-row globals** (the 32-symbol, 722
   KiB/M-row class; error-budget-gated, 6/6 + det + bit-exact gates per
   the r16 bf16-solve precedent).  The bf16-castable M-row class
   dominates the floor, so floor 1.074 -> ~0.55-0.6 ms/u (f32-only
   symbols remain): the ceiling at u=8 becomes ~6.40/4.4-4.8 =
   **1.33-1.49** -- the only route that ARITHMETICALLY opens u>=1 wins.
   Must land with execution tightening (ours currently 1.86x above own
   floor; needs <= ~1.2-1.35x of the bf16 floor to read 1.1 at u=8).
2. **Execution tightening toward own floor** (independent of #1):
   the caryA/caryM/bpre superlinear class (1.15-1.45x/u) is pure
   footprint penalty -- bf16 halves it for free; fwdscan's NC=256
   thrash onset (1.17x/u, 20.5% share at T16384) reopens the r19
   K-split/residency family CLOSED at NC<=128.
3. **Fusion passes**: the dqkwg+wyb+dhs trio is 52% of u=8 e2e; the r21
   arena/alias machinery is the merge substrate.  Kernel-count reduction
   also cuts the ~0.14 ms fixed intercept (9 launches + restage).
4. **The regime pivot option (no code)**: position the current
   architecture as the small-shape + cold-start-serving bwd (u < 0.3)
   plus the u>=1 fwd winner; leave u>=1 bwd to round 2's structural
   change.  This is what the measured curve supports today.

## Appendix: artifacts

* Chains: `scratch/rc1_{B1T2048,H32,B4,T8192,T16384,T512H4}/`
  (payload provenance-verified vs HEAD; anchor md5s: B1T2048/T8192 9/9
  == r21b records; B4==T8192 on wy/caryA/caryM/bpre/dkb/dqkwg/wyb --
  M-only stages identical, serial stages differ).
* Logs: `scratch/rc1_logs/` -- ab_<tag>_{ours,fla}_{1,2,3}.json,
  cold_<tag>.json, verify_<tag>.{txt,json}, det_<tag>/run{0,1,2}.npz,
  curve_collected.json (this table).
* Harness: `tools/rc1_flacell.py` (fla cell baseline, cold+warm),
  `tools/rc1_price.py` (CPU pricing), `tools/gdn_bench_common.py`
  (rc1 ref case), `tools/gdn_bwd_verify.py` (RC1 dh0 B>1 reshape),
  refs `evidence/G5/refs/rc1_B1T16384H16_{inputs,ref}.npz`.
