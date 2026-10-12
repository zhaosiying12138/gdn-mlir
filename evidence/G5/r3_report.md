# R3: consumer-side bf16 nativity + the A_all/caryA retirement -- the regime-change campaign verdict

Regime-change campaign round 3 (workstream R3; coordination row
`gdn-mlir R3`).  R2's ranked levers taken in order: (1) the CVT tax in
fwdscan's wS gemm, (2) the A_all/caryA retirement.  CPU budget probe
FIRST (both routes), then implementation, then the full gate stack and
the triple-interleaved paired ladder.

**Headline: both levers LAND.  The A-retirement deletes the caryA stage
outright and collapses dhs to a shifted bf16 copy (5.9x on the stage,
+the whole caryA stage); the fwdscan wS tgemm a-fragment hoist kills
half the conversion tax with ZERO value change (bit-identical outputs,
on-GPU proven).  Paired r2/r3: +11.3% / +18.7% / +20.3% / +19.0% /
+20.8% / +21.3% at u=1/16/1/2/4B/4T/8 -- every surviving stage output
BIT-IDENTICAL to the r2 chain.  ratio(fla/ours) at u=8: 0.381 -> 0.462
(same-window paired arms).  The ratio still cannot cross 1.1x -- and the
round's floor arithmetic shows it CANNOT at any execution quality: the
precision-constrained byte floor itself caps u=8 at ~0.85-0.92x.  The
campaign's final answer: bwd u>=1 is closed at the precision-constrained
floor.**

Window disclosure: the whole ladder ran in a CLEAN window (0 co-load
compute apps throughout; GPU 0%/2154MiB before each arm).  All
before/after numbers are same-window triple-interleaved PAIRED medians
(r2-chain / r3-chain / fla interleaved per iteration x3).

Artifacts: probe `scratch/r3_probe.py` + `scratch/r3_logs/probe_*.json`;
chains `scratch/r3_{B1T2048,T512H4,B4,T8192,T16384,H32}/` +
`scratch/r3ahoff_T16384/` (the hoist isolation arm); harness
`scratch/r3_{gen,build,buildahoff,verify,det,abboth,cmp,bitcmp}.sh`
(staged via .gdn-mlir-staging/r3/).

## 1. The CPU budget probe (probe FIRST -- scratch/r3_probe.py)

The R2 staged fp64 twin extended with two levers; every increment vs
the Q={} twin at seed 0.

**(a) The A-magnitude regime gate (astats).**  What A actually
contributes to the dhs chain, measured on the twin (max over all
chunks/rows):

| shape | max abs A | max 2^gcl | max abs A@dH (the dropped term) | max abs dH (the fed state) |
|---|---|---|---|---|
| T512H4 | 1.59e-19 | 1.84e-19 | 1.90e-19 | 6.8e-2 |
| T2048 | 2.01e-18 | 2.34e-18 | 2.26e-18 | 8.3e-2 |

The dropped terms are ~16 orders below the state they feed -- and ~15
orders below the bf16 store ULP of dh_all itself (which is why the
retirement's dh_all BYTES do not move; see the gate record).

**(b) The A-retirement increment (aret: A := 0 in the twin's dhs walk),
solo at every ladder shape:**

| shape | dq | dk | dv | dbeta | dg | dh0 |
|---|---|---|---|---|---|---|
| T512H4 | 0 | 2.8e-17 | 4.3e-19 | 1.4e-17 | 6.9e-18 | 2.6e-23 |
| T2048 | 0 | 5.6e-17 | 6.9e-18 | 5.6e-17 | 2.8e-17 | 1.3e-21 |
| B4 | 0 | 5.6e-17 | 3.5e-18 | 5.6e-17 | 1.0e-17 | 6.8e-21 |
| T8192 | 0 | 2.8e-17 | 2.6e-18 | 2.8e-17 | 3.5e-18 | 3.4e-21 |
| T16384 | 0 | 2.8e-17 | 3.5e-18 | 2.8e-17 | 6.9e-18 | 3.4e-21 |

14+ orders under the 8e-3/2e-2 bars at EVERY shape.  `aret +
default9-cast-set` equals `default9-only` at every printed digit at
every shape -- the retirement adds NOTHING on top of R2's already-shipped
error budget.  Probe-vs-GPU: the combined set's predicted digits land on
the measured verify digits (T8192 dbeta 3.449e-3 pred vs 3.444e-3
measured, dh0 2.015e-4 EXACT; T16384 dbeta 2.694e-3 vs 2.691e-3, dh0
2.039e-4 EXACT).

**(c) The fwdscan wS split-S probe (the OTHER lever-1 route).  S split
once per use into a bf16 hi+lo pair (S ~ hi+lo to 2^-16 rel; register
recurrence stays full-precision):** dk solo 2.6e-8 / 6.6e-8 at
T512H4/T2048 -- 5 orders under the default9 set's own increments.
Numerically GO.  **But pool-blocked**: the S_hi/S_lo pair adds 16KiB to
fwdscan's 99072B pool -- over the 101376B sm_120a static bar (the GSCG
park already holds the floor at T>=8192).  The route that shipped
instead is value-EXACT (below), so this probe documents the unused
numerics headroom, not a taken risk.

## 2. Implementation (tools/gdn_bwd_gen.py)

**Lever 2 -- GDN_ARET (default 1; =0 restores the R2 chain
byte-for-byte).**  Two changes:
* `main()` drops `caryA` from the stage list -- its only output A_all
  has no consumer left.
* `gen_dhs` ARET arm: the reverse walk `dh_all[n] = X_arr(n),
  X <- A_r@X + B_r[n]` with A ~ 1e-18 reduces EXACTLY to
  `dh_all[NC-1] = dht, dh_all[n<NC-1] = B_all[n+1], dh0 = B_all[0]`.
  Emission: ONE [K, 2*BV] ping-pong tile (parity = the column half);
  per chunk: B[m]'s 16B cp.async burst lands in the idle half while the
  store sweep drains the live half to dh_all (the FPIPE cover recipe),
  then await+barrier.  The A-window machinery (2x[K,BKW] tiles, NW
  rgemm_aws per chunk, the XP cross-chunk token), the f32 X/Xa pair,
  and the Xa->X copy sweep all retire: **pool 49152 -> 8192B, 124 -> 32
  registers, 504 -> 91 payload lines.**  dh_all stores the B payload
  bits VERBATIM (bf16 tile -> bf16 global is the round-trip identity).
* REGIME GATE (documented at the knob): valid iff the decay keeps
  2^gcl << tolerance scale -- true for the gate/benchmark recipe and
  any learned-gate data with comparable decay; with weak decay (g -> 0)
  A -> I and the arm is invalid (=0 restores).

**Lever 1 -- GDN_TGEMM_AH (default 1): the tgemm a-fragment HOIST arm
for bf16 A operands.**  A bf16 tile feeding the tf32k8 hatch pays a
per-element extf inside the k-loop's operand loads, once per reusing
n-tile (R2: fwdscan 0.913x).  When a warp's atom tiles all live in one
16-row m-block (npr % tpw == 0), the four a-fragment registers per
k-step load (and widen) ONCE and feed every tile's mma: a-loads/extf
drop npr->1 per element per warp pass.  The warp map changes
round-robin (ti = mW + i*W) -> consecutive (ti = mW*tpw + i); per-cell
k-chains stay kb-ascending with identical atoms -> bit-identical
values.  Activates ONLY where the A operand tile is bf16 (under default
knobs: exactly fwdscan.wS's two slices).  fwdscan: 1068 -> 932 payload
lines, 102 -> 118 regs, 0 spill.  64 mma atoms before == 64 after.

## 3. Gate record

| gate | result |
|---|---|
| knob-off bit identity | GDN_ARET=0 GDN_TGEMM_AH=0: 9/9 payload.mlir + bwd_manifest.json BIT-IDENTICAL to the r2 chains |
| untouched stages | dqkwg/wyb/dkb/wy/caryM/bpre cubin md5s == the R2 record (e.g. B1T2048 dqkwg deadc496ab38, wyb d5767430bd70); full 48-md5 record in `scratch/r3_logs/md5_record.txt` |
| numerics 6/6 | PASS at ALL 6 ladder cells (fp64 twin, seed 0).  Digits == the R2 records: B4 dk 5.036e-3 (r2: 5.04e-3), T16384 dk 3.819e-3 (r2: 3.82e-3), T8192 3.922e-3 (r2: 3.92e-3), dh0 2.0e-4 everywhere (probe-exact) |
| determinism | det N=3 fresh-process, ALL stage outputs bit-identical: B1T2048, T16384, B4 (the structural-change rule) |
| r2-vs-r3 chain bit-compare | single det runs both chains at T16384: EVERY common stage output byte-identical -- incl. fwdscan's h_all/vnew_all (the hoist proof) AND the dhs-side symbols: the retirement's dropped terms are 15 orders under dh_all's own bf16 store ULP, so the chain converges to identical bytes; A_all exists only in the r2 root |
| VRAM | high-water 6.56 GiB TOTAL during the T16384 bench, of which 2.10 GiB is the pre-existing baseline -> ours ~4.5 GiB < 6 GB; no co-job present the whole round (clean window) |

## 4. The measured ladder (triple-interleaved r2/r3/fla x3, paired medians)

| cell | u | ours r2 | ours r3 | **r2/r3** | ratio(fla/ours) r2 -> r3 |
|---|---|---|---|---|---|
| B1T512H4 | 1/16 | 0.5107 | 0.4588 | **1.113x** | 1.548 -> **1.723** |
| B1T2048H16 | 1 | 2.1234 | 1.7883 | **1.187x** | 0.411 -> **0.488** |
| B1T2048H32 | 2 | 3.9281 | 3.2644 | **1.203x** | 0.345 -> **0.415** |
| B4T2048H16 | 4B | 7.4286 | 6.2444 | **1.190x** | 0.346 -> **0.411** |
| B1T8192H16 | 4T | 7.4911 | 6.2010 | **1.208x** | 0.375 -> **0.453** |
| B1T16384H16 | 8 | 14.7390 | 12.1521 | **1.213x** | 0.381 -> **0.462** |

Stage decomposition at u=8 (T16384, paired medians):

| stage | r2 | r3 | spd | class |
|---|---|---|---|---|
| wy | 1.466 | 1.467 | 1.00x | untouched |
| fwdscan | 3.266 | 2.930 | **1.115x** | the wS hoist (see isolation below) |
| caryA | 0.557 | -- | **deleted** | the retirement |
| caryM | 0.671 | 0.672 | 1.00x | untouched |
| bpre | 0.412 | 0.412 | 1.00x | untouched |
| dhs | 2.083 | 0.354 | **5.89x** | the walk collapsed to a shifted copy |
| dkb | 1.020 | 1.019 | 1.00x | untouched |
| dqkwg | 3.295 | 3.292 | 1.00x | untouched |
| wyb | 2.110 | 2.103 | 1.00x | untouched |

**Hoist isolation (the r3ahoff arm: ARET=1, TGEMM_AH=0, T16384):** e2e
r2/ahoff = 1.179x with fwdscan 0.994x (unchanged, as designed); r3 vs
ahoff on fwdscan: 3.282 -> 2.930 = **1.120x, ~0.35 ms @u=8 -- the whole
R2-measured CVT tax, recovered**, with bit-identical outputs.  The
retirement alone is worth 1.179x; the two levers compose to 1.213x.

Marginals (T8192 -> T16384 refit): ours 1.812 -> **1.488 ms/u**, fla
0.7025 ms/u, ratio-of-marginals 0.388 -> **0.472**.  Small end
DEEPENED: T512H4 1.72x vs fla (R2: 1.86 in its window; both windows'
arms re-measured here: r2-arm 1.55 -> r3-arm 1.72).

## 5. The close-out arithmetic (why 1.1x is closed -- the floor itself)

At u=8 in this window: fla 5.620 ms; crossing 1.1x needs ours <= 5.11 ms.

1. **Perfect execution of the CURRENT byte set cannot cross.**  R2's
   byte-scaled floor was 0.81 ms/u (the 9-symbol cast set); R3 removes
   caryA's whole byte stream (~257 MB -> ~0.34 ms @T16384) and dhs's
   A-window traffic, putting the floor at ~0.75 ms/u -> **~6.1-6.6 ms
   e2e at u=8 = ceiling ratio ~0.85-0.92** even at 100%-of-roofline
   execution of exactly these edges.  ours-r3 12.15 ms is ~2x that
   floor -- the residual is issue-bound scalar/sweep machinery (dqkwg's
   mixed sweeps, wyb's mixed sites), not bytes.
2. **Even deleting dqkwg+wyb+dhs OUTRIGHT (R2's #3 fusion class, 47% of
   the new e2e) caps at 0.88** (12.15 - 5.75 = 6.40) -- and those
   stages compute the graded outputs, so deletion is not a real move;
   fusion reduces their bytes but cannot zero them.
3. The bytes that COULD move the floor are the probe-KILLED f32 edges
   (u_all/vnew_all/dvp_all/dkB_sum/dk_local, 470 KiB/M-row) -- only
   reachable by REMOVING edges (fusion), not by casting; and the graded
   outputs dq/dk/dv stay f32 by the budget.  The 1.33-1.49 RC1 dream
   needed the full 32-symbol cast class the error budget killed (R2's
   verdict, unchanged).

**Campaign verdict: the bwd u>=1 regime change is CLOSED AT THE
PRECISION-CONSTRAINED FLOOR.**  Rounds RC1 -> R2 -> R3 moved u=8 from
~0.396 to ~0.46 (window-matched: each round's own-window paired arms)
and the marginal ratio to 0.472; the remaining ~2.4x to 1.1x is not
reachable by storage precision (budget), consumer nativity (this
round: the last taxed mma feed is now native), algorithm retirement
(this round: the deadest weight is gone), or any fusion short of
deleting the output-producing stages.  The shipped architecture claim
stands and sharpens: **the small-shape + cold-start bwd winner
(1.72x vs fla at T512H4, deepened from 1.55 same-window) plus the u>=1
fwd winner; the u>=1 bwd is a precision-constrained ~0.46-0.47x vs fla
and that is the floor's number, not ours.**

## 6. R4 ranking (if the campaign reopens)

1. **dqkwg+wyb fusion** (the only byte-floor mover left; 5.40 ms of
   12.15 @u=8): removes the probe-killed vnew_all/dvp_all re-reads;
   the r21 arena is the substrate.  Ceiling even at a perfect 2x:
   ~0.6x -- a margin play, not a regime change.
2. **dqkwg/wyb sweep nativity** (the R2 "mixed" stages: converting
   sweeps tax ~ split wins, 1.02x each): same hoist-class surgery on
   the rgemm readers; ~0.1-0.2 ms class.
3. **fwdscan 2.93 ms**: now native-fed; the r19-r21 K-split family is
   closed; the residual is the serial walk itself.
