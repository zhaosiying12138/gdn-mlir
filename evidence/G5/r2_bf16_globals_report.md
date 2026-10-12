# R2: bf16 the M-row globals -- the error-budget-gated byte halving

Regime-change campaign round 2 (workstream R2; coordination row
`gdn-mlir R2`).  RC1's #1 lever ("halve the bytes: bf16 the M-row
globals", predicted ceiling 1.33-1.49 @u=8) taken through the r16
bf16-solve precedent: CPU error budget FIRST, then the cast, then the
interleaved ladder.

**Headline: the error budget admits 10 of 16 candidate symbols (and
every graded output is locked out); the implemented 9-symbol default
(sans the CVT-taxed A_all) improves EVERY ladder cell -- paired
+1.9%/+1.3%/+1.2%/+0.9%/+1.3% at u=1/2/4B/4T/8 and +9.4% at the
small end (B1T512H4 ratio vs fla 1.70 -> 1.86) -- but the u>=1 ratios
move only 0.43->0.44 / 0.32->0.33 / 0.35->0.36: the byte halving does
not yet convert to time because the u>=1 stages are ALU/issue-bound in
their conversion machinery, not DRAM-bound.  The 1.33-1.49 ceiling
required the FULL 32-symbol class including the probe-KILLED symbols;
under this tolerance regime that arithmetic route is closed by the
budget itself.**

Artifacts: probe `scratch/r2_bf16_probe.py` + logs `scratch/r2_logs/`
(probe_*.json, verify_*.json, det_*/, abboth_*), chains
`scratch/r2_{B1T2048,T512H4,B4,T8192,T16384,H32}/`, harness
`scratch/r2_{gen,build,verify,det,abboth,pair,subset}.sh`.

## 1. The CPU error budget (probe: scratch/r2_bf16_probe.py)

Restructured fp64 twin of the 9-stage dataflow with the kernels' EXACT
global read/write edges (payload.mlir global scans; dqkwg's self
re-read of dvp_all/dW_all modeled).  Validation: the restructured twin
with Q={} reproduces the original `bwd_twin` to 3.6e-15 (dq 0.0,
dk 3.6e-15) -- the restructure is mathematically exact.  Each candidate
round-trips through bf16 at its store (truncf) and is read back
widened (extf) at every consumer edge; producer recurrence state stays
f32 (store rounding never feeds back into the walk).

Leave-one-in at the gate shapes (solo increments vs the
unquantized twin; bars/2 = 4e-3 / 1e-2):

| symbol | elems/M-row (all edges) | dk solo | dbeta solo | verdict |
|---|---|---|---|---|
| w_all | 32,768 | 4.7e-5 | 2.8e-5 | castable (56-170x) |
| h_all | 49,152 | 4.7e-5 | 3.2e-5 | castable |
| A_all | 32,768 | 0 (!) | 0 | castable -- numerically DEAD (see below) |
| B_all | 32,768 | 5.1e-4 | 4.0e-4 | castable (16x) |
| dh_all | 49,152 | 5.1e-4 | 4.0e-4 | castable (dh0 UNAFFECTED: the dhs walk keeps f32 registers) |
| dW_all | 24,576 | 4.1e-5 | 3.6e-5 | castable |
| T_all | 12,288 | 7.3e-5 | 2.1e-3 | castable (4.8x on dbeta) |
| P_all | 12,288 | 1.7e-4 | 2.0e-3 | castable |
| kkt_all | 8,192 | 0 | 9.1e-4 | castable |
| M_all | 16,384 | 1.0e-3 | 3.8e-4 | castable (4x on dk) |
| **u_all** | 16,384 | **4.6e-2** | 4.3e-5 | **KILLED** (11x over the dk bar) |
| **vnew_all** | 24,576 | **4.6e-2** | -- | **KILLED** (the vn signal feeds dkB directly) |
| **dvp_all** | 24,576 | 1.2e-3 | **4.8e-2** | **KILLED** (2.4x OVER the dbeta bar) |
| **dkB_sum** | 16,384 | **6.1e-2** | -- | **KILLED** (dk-local terms are O(10) cancelling to O(0.1)) |
| **dk_local** | 16,384 | **6.2e-2** | -- | **KILLED** |
| rowB/s5/r3/r4/c4r/dg_state | 770 | -- | dg 1.3-1.5e-2 | stay f32 (negligible traffic) |
| outputs dq/dk/dv | 24,576 write | dk 6.2e-2 / dv 7.2e-3 / dq 1.9e-3 | | **stay f32** (graded) |

Set-level increments (the implemented 10-symbol castable set):
dk 4.5e-4 / 1.0e-3 / 1.3e-3 / 1.7e-3 and dbeta 1.3e-3 / 2.0e-3 /
5.2e-3 / 2.7e-3 at T512H4 / T2048 / B4 / T16384.  Projected totals
(kernel + increment) hold under bar/2 at both GATE shapes (worst: dk
2.3e-3 of 4e-3 at T2048) and under the bar everywhere (worst: B4 dk
3.5e-3 linear-sum of 8e-3).

**Finding (paper material): A_all is numerically dead in this data
regime.**  With the gate recipe's g (E[g] ~ -0.69 natural-log per
token), 2^gcl ~ 1e-19 per 64-token chunk: every entry of
A = 2^gcl I - W^T Bhat is ~1e-18, and the probe measures a 1e-27-class
end-to-end increment from rounding it.  The dhs carry chain is
B-dominated; caryA + the A-window machinery compute dead weight (the
R3 retirement lever).

Split-bf16 (hi+lo) analysis for the marginal class: every reader of
u_all/vnew_all/dkB_sum/dk_local is precision-tight, so hi+lo (4B, no
saving) buys nothing -- the "half the useful traffic reduction" route
is empty here; their bytes are only reachable by REMOVING the edges
(fusion, R3).

## 2. Implementation (tools/gdn_bwd_gen.py)

Knob `GDN_BF16G` (default = the 9-symbol set; "none" restores the f32
globals).  Global dtype is a property of the symbol (per-symbol
`gdt()`/`gty_()` dispatch at every k.global_ and type-string site);
producers truncf at store and consumers extf at read through the
existing gload/gstore/stage_rows/stage_split_*/rgemm/aggemm/gcol
machinery (all dtype-aware).  cp.async sites: bf16 landing tiles where
the readers are load2-generic (dhs A-windows/Xb, dkb dHv ping-pong,
fwdscan wbf, dqkwg tTr/dWt0/dWt1, wyb wyT-f32-because-dvp-shares-it);
converting sweeps (new `stage_rows_off`) where the dst is a COMPUTE
tile (wy's solve tile Tf; fwdscan's in-place kg tile wpp).

Three build-law findings on the way (all in the round's logs):
* **Tf and wpp are compute tiles** -- the solve and the in-place kg
  sweep write them; a bf16 flip there is a numerics change (kg
  quantization is unprobed) AND a type error.  They stay f32.
* **The scalar-gemm CVT tax**: bf16 tiles feeding scalar-FMA gemm
  loops (fwdscan's wS, dhs's rgemm_aw windows) pay a per-element
  bf16->f32 convert inside the MAC loop -- measured 0.49x..0.91x on
  those stages (the paired bisect: A_all alone -> dhs 0.52ms vs 0.27
  f32).  mma/ldmatrix consumers take bf16 operands natively and WIN
  (bpre 1.30-1.50x, caryA/M, dkb, wy).  A_all is therefore
  DEFAULT-OFF (its only consumer is the taxed rgemm_aw; and it is
  numerically dead anyway).
* Converting sweeps on serial critical paths are worse than the CVT
  tax (fwdscan sweep variant measured 0.49x vs 0.90x for the
  bf16-tile direct read); the async landing tile + native-width read
  is the right shape wherever the reader can widen.

## 3. Gate record

| gate | result |
|---|---|
| knob-off bit identity | GDN_BF16G=none: 9/9 payload.mlir + bwd_manifest.json BIT-IDENTICAL to the RC1 chains (r19-final bytes) |
| payload provenance | all 6 r2 chains regenerated from the shipped generator; cubin md5s recorded in `scratch/r2_logs/md5_record.txt` (54 stage md5s; e.g. B1T2048 fwdscan ddf24ae855bc, dqkwg deadc496ab38, wyb d5767430bd70) |
| numerics 6/6 | PASS at ALL 6 ladder cells (fp64 twin, seed 0).  Worst digits: dk 5.04e-3 @B4 (1.6x under bar), 3.92e-3 @T8192, 3.82e-3 @T16384 (the f32 record was 3.80e-3 -- the A_off build is digit-identical to the 10-symbol build, A was dead); dbeta 5.17e-3 @B4 (3.9x); dh0 2.0e-4 everywhere (probe-exact: the probe's 2.04e-4 increment IS the measured digit) |
| determinism | det N=3 fresh-process, ALL stage outputs bit-identical: B1T2048, T16384, B4 (the structural-change rule) |
| VRAM | union at T16384 ~= 3.1 GiB (RC1's 3.7 minus the 156 KiB/M-row footprint cut) + harness -- < 6 GB; co-job (pids 4094539 -> 4109809 over the round) never touched |
| probe validation | GPU measured increments match the probe: dv/dg/dh0 digits EQUAL the probe increments (1.9e-4/7.1e-4/1.9e-4 @T2048); dk/dbeta land in the RSS band (the tf32 rebuild-resample effect) |

## 4. The measured ladder (triple-interleaved f32/bf16/fla x3, paired)

`abboth` runs: ours-f32 (rc1 chain) / ours-bf16 (r2 chain) / fla
interleaved per iteration -- the only honest before/under under co-load
drift (the window moved twice during the round; per-stage absolutes
varied up to 12% between windows, paired ratios are the currency).

| cell | u | ours f32 | ours bf16 | **bf16/f32** | ratio(fla/ours) f32 -> bf16 |
|---|---|---|---|---|---|
| B1T512H4 | 1/16 | 0.5626 | 0.5141 | **1.094x** | 1.703 -> **1.864** |
| B1T2048H16 | 1 | 2.1705 | 2.1306 | **1.019x** | 0.433 -> 0.441 |
| B1T2048H32 | 2 | 4.6585 | 4.5984 | **1.013x** | 0.306 -> 0.310 |
| B4T2048H16 | 4 | 8.8392 | 8.7373 | **1.012x** | 0.322 -> 0.326 |
| B1T8192H16 | 4 | 8.9652 | 8.8890 | **1.009x** | 0.346 -> 0.349 |
| B1T16384H16 | 8 | 17.5255 | 17.3063 | **1.013x** | 0.353 -> 0.357 |

(One lighter-window paired sample at T16384 read 0.899x with the same
per-stage winners/losers -- window sensitivity ~+-10% on the delta;
the stage-level pattern below is stable across all windows.)

Stage decompositions at u=8 (T16384, paired medians):

| stage | f32 | bf16 | spd | class |
|---|---|---|---|---|
| wy | 1.570 | 1.467 | **1.070x** | mma stores+reads: native win |
| fwdscan | 3.341 | 3.658 | **0.913x** | scalar-gemm CVT tax (the wS gemm) |
| caryA | 0.620 | 0.551 | **1.125x** | split-bf16 mma reads |
| caryM | 0.728 | 0.670 | **1.086x** | split-bf16 mma reads |
| bpre | 0.613 | 0.410 | **1.495x** | aggemm global reads: bytes halved, no tax |
| dhs | 2.294 | 2.313 | 0.992x | B/DMA par; A kept f32 (taxed path) |
| dkb | 1.172 | 1.018 | **1.151x** | bf16 dHv landings + split readers |
| dqkwg | 3.901 | 3.824 | 1.020x | mixed (sweeps tax ~ split wins) |
| wyb | 2.455 | 2.404 | 1.021x | mixed |

Cast-subset matrix at T16384 (Mk / MkBdh / MkTP / MkhP / full9):
full9 beats every trim (17.29 vs 17.33-17.61 in-window) -- the
fwdscan w-tax is real but dropping w costs more (wy/cary wins) than it
saves; the 9-symbol default is the measured optimum among the subsets.

## 5. Floor analysis update (the honest arithmetic)

Castable traffic under the budget: 254,464 elems/M-row of the 351,104
intermediate edges (62%; the implemented set) -- 437 KiB/M-row of the
1.70 MiB total chain traffic saved (26%).  Model floor (scaling the
r7 1.074 ms/u by bytes): **~0.81 ms/u** (was 1.074; the full-class
dream was 0.55-0.60 -- it required casting the probe-KILLED 28% too).
Ceiling at u=8: 6.16/(0.81*8+0.14) ~= **0.92** in this window (the
1.33-1.49 RC1 ceiling is not reachable by storage casts under this
error budget -- closed by the budget, not by execution).

Execution vs the new floor: ours 17.31 ms = ~2.1x the ~8.1 ms
bf16-floor e2e -- the ALU/issue-bound conversion machinery (CVT tax,
split sweeps, truncf stores) is where the byte win currently goes to
die.  The round's +1-2% is the first slice; the headroom is real and
now quantified per-stage.

## 6. R3 ranking (by this round's evidence)

1. **Consumer-side bf16 nativity** (the CVT tax): fwdscan's wS gemm
   (0.86-0.91x, ~0.5 ms @u=8) and the split-sweep readers need
   mma-native or widen-once paths; dhs's A-window rgemm likewise
   (0.57x if A is cast).  This is the difference between "bytes
   halved" and "time halved".
2. **The A_all retirement** (algorithm change, own gates): A ~ 1e-18
   in this regime -- caryA (0.55 ms) + the dhs A-window machinery
   (~0.7 ms of the 2.3) compute dead weight @u=8.  Same class: the
   dhs B-gemm's contribution through the ~zero carry.
3. **Fusion passes** (RC1 #3, now sharper): the probe-killed symbols
   (u_all/vnew_all/dvp_all/dkB_sum/dk_local, 470 KiB/M-row) are only
   reachable by REMOVING edges -- dqkwg+wyb+dhs is 52% of u=8 e2e and
   the r21 arena is the merge substrate.
4. **The regime pivot stands** (RC1 #4): with the small end now at
   1.86x vs fla (deepened by R2's fixed-cost cuts), the shipped
   architecture is the small-shape + cold-start bwd plus the u>=1 fwd
   winner, pending R3's consumer-nativity work.
