# R4: the dqkwg+wyb fusion paper price -- NO-GO; the regime-change campaign closes at the precision-constrained floor

Regime-change campaign round 4, the FINAL ranked lever (workstream R4;
coordination row `gdn-mlir R4`).  R3's ranking put the dqkwg+wyb fusion
first ("the only remaining byte-floor mover, ceiling ~0.6x@u=8") and
sweep nativity second (~0.1-0.2 ms class).  Per the r14/r16 discipline,
the fusion was PAPER-PRICED FIRST: byte arithmetic from the L1 math and
the shipped chains, pool arithmetic from the measured smem_fix pools,
occupancy arithmetic from the r15-measured 1CTA factors.  The paper
says NO-GO, and the round closes the campaign with the final chain
unchanged (the r3-final bytes, health-certified fresh this round:
48/48 cubins accounted vs the R3/R2 records, det N=3 at T16384
all-bit-identical).

**Headline: the fusion is pool-capped at 82% of its own byte ceiling and
occupancy-inverted by the r15-measured 1CTA factor.  The G-accumulator
barrier G = dvp.(beta v)^T + dW.(beta k 2^gc)^T is a full-materialization
boundary -- both operands are complete [C,*] matrices before anything
downstream runs -- so no phase interleave can shrink the parks: killing
the dvp/dW/dk_local edges requires +32K/+16K/+32K B of smem that must be
live across essentially the whole fused wyb phase (dvp is re-needed at
the dv tail), on top of wyb's 45056 B aliased-minimum pool.  The best
BUILDABLE kill-set {dvp, dW, vec} = 97024 B pool forces 1 CTA/SM on a
merged body where both halves ship at 2 CTAs/SM today; the measured
dqkwg 1CTA factor alone (1.155x @T2048 / 1.2597x @T8192, r15 paired)
costs 0.08-0.86 ms against a 0.07-1.48 ms saving band (L2..DRAM bounds).
The net is a wash centered on zero at every ladder cell; even the
impossible best corner (all traffic priced at pure DRAM AND the wyb half
paying ZERO 1CTA penalty) moves u=8 only 0.4629 -> ~0.488 -- a margin
play under every bar this campaign has shipped at, for the highest
engineering-complexity class it has (a 2-kernel merge).  NO-GO, decision
grade.**

Window disclosure: the round opened with a transient co-app (pid 16816,
8.5 GB, 0% util) that cleared before the only GPU touch; the health cert
below ran in the clean baseline window (0% util, 2154 MiB before, no
co-load apps).  The cert is bit-identity only -- no timing claims are
made this round, so co-load sensitivity does not apply.

Artifacts: pricing `scratch/r4_price.py` -> `scratch/r4_logs/price_out.txt`;
edge audit `scratch/r4_audit.py`; health cert `scratch/r4_logs/det_T16384/`
+ the md5 table below; harness staged via
`.gdn-mlir-staging/r4/` (Windows side).

## 1. The fusion boundary from the L1 math (tools/gdn_l1_gen.py @gdn_bwd)

The backward's per-chunk local-term stage (dqkwg) and its consumer
(dA-correction + final tails, wyb) split the PASS-2 math of the chunked
reference exactly as follows (gdn_l1_gen.py emit_bwd_kernel):

```
dvp  = P^T dO + Bhat dH                     -> dqkwg P9 (writes dvp_all)
dW   = -dvp S_n^T                           -> dqkwg P11 (writes dW_all)
G    = dvp (beta v)^T + dW (beta k 2^gc)^T   -> wyb P1/P2
dA   = -strict_tril(T^T strict_tril(G) T^T)  -> wyb P3/P4
dAf  = dA .* 2^(gc_i-gc_j)
dq   = (dO S^T).*2^gc + dPm k               -> dqkwg P10 (TERMINAL, graded)
dk   = ... (dAf k, dAf^T beta k, TW = T^T dW, vn dH^T, dPm^T q, dkB)
                                                 -> dqkwg P12 partial + wyb P7/P8
dv   = (T^T dvp).*beta                      -> wyb P9 (needs dvp AGAIN)
dbeta/dg = rowsum/suffixsum chains over dAf, TW, T^T dvp, r3/r4/c4r, dg_state
                                                 -> wyb P5/P8/P9/P10
```

So of dqkwg's eight outputs, exactly ONE is terminal (dq -- a graded
output, f32 by the error budget, written regardless); the other seven
(dW_all, dvp_all, dk_local, r3_c, r4_c, c4r_c, dg_state) flow to wyb --
and dW/dvp additionally feed dqkwg's own later phases (the TW loop
re-reads dW_all; the dW loop re-reads dvp_all).  The edge map is CLOSED:
a payload audit of the shipped r3 chains (scratch/r4_audit.py) confirms
no third stage touches any of the seven symbols -- dqkwg and wyb are the
entire consumer set, so a fusion's kill-set is bounded by these edges
and nothing else.  (vnew_all -- the other probe-killed f32 edge -- is a
fwdscan->dkb/dqkwg edge and is NOT reachable by this fusion.)

**The structural finding (why lifetimes do NOT allow pool sharing):**
G accumulates over ALL of V (dvp's contraction) and ALL of K (dW's), and
everything downstream of G (dAf, then dk/dbeta/dg tails) plus dv (which
re-reads dvp directly at the very tail) sits behind G's completion.  A
fused kernel could interleave production and consumption slice-by-slice
for the GEMM inner loops, but it still must hold the COMPLETE dvp
(produced at dqkwg-P9, last read at wyb-P9), dW (P11 -> wyb-P8) and
dk_local (P12 -> wyb-P7) matrices at the barrier.  The parks are
additive to the consumer phase's own pool for their whole lifetime:
fused_pool(subset) = max(dqkwg_phase_pool + parks, wyb_phase_pool +
parks).  There is no interleaved restructure that shrinks them.

## 2. The byte price (per M-row; C=64, K=V=128; ladder shapes)

Edge traffic of the seven shared symbols, counted with loop multiplicity
exactly as the shipped generators emit them (dtypes per the R2 cast;
dW_all is bf16 storage, dvp_all/dk_local stay f32 -- probe-killed):

| symbol | edges (per M-row) | traffic B/row | smem park B |
|---|---|---|---|
| dvp_all f32 [C,V] | dqkwg P9 write 32768; dqkwg P11 dW-loop re-read 16x[C,BV] slices 131072; wyb P1 32768; wyb P9 32768 | **229376** | 32768 |
| dW_all bf16 [C,K] | dqkwg P11 write 16384; dqkwg P12 TW re-read 16384; wyb P2 16384; wyb P8 16384 | **65536** | 16384 |
| dk_local f32 [C,K] | dqkwg P12 write 32768; wyb P7 read 32768 | **65536** | 32768 |
| r3/r4/c4r/dg_state | write + read | **2048** | 256 |
| **perfect kill-set** | | **362496** (~354 KiB) | 82176 |

Pricing bounds (the r24 W-park calibration, measured on this device):
DRAM-class 824.6 GB/s, L2-resident 2144 GB/s.  At T16384 (M=4096) the
perfect kill is 1.485 GB -> **0.69 ms (L2) .. 1.80 ms (DRAM)** of the
12.1521 ms e2e; even the UNBUILDABLE perfect fusion therefore tops out
at ratio(fla/ours) = 5.6247/(12.1521-1.80) = **0.544**, and at the L2
bound 0.491 -- i.e. R3's "~0.6x ceiling" (which assumed a perfect 2x on
the whole 5.395 ms pair, i.e. that the stages are byte-bound) was
generous: the byte arithmetic itself caps the fusion class at ~0.54x,
because the pair's residual is issue-bound machinery, not bytes.

## 3. The pool price (measured pools; the kill-set enumeration)

Shipped facts (ptxas smem_fix pass, r3_T16384 chains): dqkwg 47616 B
smem / 128 regs / 148-228 B spill; wyb 45056 B / 128 regs / 32 B spill;
both at 2 CTAs/SM (256 thr x 2 x 128 regs = the regfile; the 2-CTA bar
is 50176 B/CTA, static bar 101376 B).  Every subset of the kill-set
(scratch/r4_logs/price_out.txt for the full 16-row table):

| kill-set | fused pool | residency | kills B/row |
|---|---|---|---|
| -(launch-only fusion) | 47616 | 2 CTAs | 0 (edges stay; saves ~launch+gc-restage, ~us) |
| vec only | 47872 | 2 CTAs | 2048 |
| dW | 64000 | 1 CTA | 65536 |
| dvp | 80384 | 1 CTA | 229376 |
| dk_local | 80384 | 1 CTA | 65536 |
| **dvp+dW+vec** | **97024** | **1 CTA** | **296960** (best buildable) |
| dW+dk+vec | 97024 | 1 CTA | 133120 |
| dvp+dk (+..) | >=110592 | UNBUILDABLE | -- |
| all (perfect) | 129792 | UNBUILDABLE | 362496 |

The mission's "unless lifetimes allow" question is answered NO by the
G barrier of section 1: the 5120 B of headroom under the 2-CTA bar buys
nothing, and the best buildable kill-set is 82% of the perfect bytes at
1 CTA/SM.  (dk_local cannot join dvp under the static bar; the
dvp+dk_local pair alone is 110592 B.)

## 4. The occupancy price and the per-cell net

The 1 CTA/SM cost is not hypothetical -- it is the r15 paired
measurement on this exact stage: dqkwg 512thr/1CTA vs the shipped
256thr/2CTA = 0.632 -> 0.547 ms @T2048 (**F = 1.155x**) and
2.3232 -> 1.8443 @T8192 (**F = 1.2597x**); the r15 ncu record attributes
it to the independent-instruction-streams effect (achieved occupancy
15.02 -> 29.17%, no-eligible 89 -> 83%).  A fused kernel at 97024 B runs
1 CTA on BOTH merged bodies.  The net per ladder cell (stage times =
the r3 triple-interleaved paired medians; full table in
scratch/r4_logs/price_out.txt):

| cell | kill GB | saving ms (L2/DRAM) | 1CTA cost ms (dqkwg half .. +wyb half) | net best / mid / worst ms | ratio now -> best-corner |
|---|---|---|---|---|---|
| B1T2048 u=1 | 0.152 | 0.071 / 0.184 | 0.079 .. 0.130 | +0.106 / +0.024 / -0.059 | 0.4881 -> 0.5188 |
| B4 u=4B | 0.608 | 0.284 / 0.738 | 0.439 .. 0.722 | +0.298 / -0.070 / -0.438 | 0.4113 -> 0.4320 |
| T8192 u=4T | 0.608 | 0.284 / 0.738 | 0.440 .. 0.724 | +0.297 / -0.071 / -0.440 | 0.4531 -> 0.4759 |
| T16384 u=8 | 1.216 | 0.567 / 1.475 | 0.855 .. 1.401 | **+0.620 / -0.107 / -0.834** | 0.4629 -> 0.4877 |

The "best" corner requires ALL removed traffic to price at pure DRAM
bandwidth AND the wyb half to pay ZERO 1CTA penalty -- the latter is not
credible (wyb is the same structural class: 256 thr, 2 CTAs, 128 regs,
mma+staging body, already 32 B spilling; any F_wyb >= 1.10 shrinks the
u=8 net to <= +0.06 ms = noise).  The honest center of the band is a
WASH at u=1 and a LOSS at u>=4.  Against the campaign's shipped-lever
class (R3: dhs 1.73 ms, caryA delete 0.56 ms, fwdscan hoist 0.35 ms) the
fusion's expected value does not clear the bar, while its risk class is
the project's highest (merging two ~1000-line kernels, re-deriving the
cross-phase lifetime arena, landing the union body under 128 regs with
both bodies already spilling, full gate stack re-run).

**VERDICT: NO-GO, with the arithmetic.**  The lever R3 ranked #1 is
closed by pool + occupancy math built entirely from measured quantities.
R3's #2 (sweep nativity, ~0.1-0.2 ms class) remains unshipped by the
same standard: 0.2 ms of 12.15 ms is +0.008 ratio -- it cannot change
the campaign verdict and is noted here for the record, not priced.

## 5. The final-chain health cert (the close-out chain == r3-final bytes)

| check | result |
|---|---|
| md5 audit | 48/48 cubins across the six r3 chains accounted: 32 == `scratch/r3_logs/md5_record.txt` (B4/T8192/T16384/H32; e.g. T16384 dqkwg e07ba815c922, wyb d4b4a7379394); the other 16 (B1T2048/T512H4, absent from that file) cross-checked against the R2 record -- every UNTOUCHED stage byte-equal (dqkwg deadc496ab38 / wyb d5767430bd70, the R3 gate anchors), fwdscan/dhs differing exactly as R3 designed (hoist + ARET) |
| determinism | det N=3 fresh-process at T16384: ALL stage outputs bit-identical (scratch/r4_logs/det_T16384/) |
| generator | UNCHANGED this round (no knobs touched; HEAD == R3's b8d3fac payload emission) |
| VRAM | cert runs in the clean baseline window (2154 MiB before, no co-load apps); chain working set ~3.1 GiB < 6 GB rule |
| window | transient co-app (pid 16816, 8.5 GB, 0% util) present at round start CLEARED before the cert; no timing claims made this round |

## 6. The campaign's final ladder (RC1 -> R2 -> R3 -> R4)

R4 ships nothing; the ladder is R3's, unchanged.  For the record, the
full campaign chain (each round's own-window paired arms; R4 = the
paper close-out):

| cell | u | RC1 ours | R2 ours | R3 ours (= final) | final ratio(fla/ours) |
|---|---|---|---|---|---|
| B1T512H4 | 1/16 | 0.5573 | 0.5107 | 0.4588 | **1.723 (WIN)** |
| B1T2048H16 | 1 | 2.1917 | 2.1234 | 1.7883 | 0.488 |
| B1T2048H32 | 2 | 4.1539 | 3.9281 | 3.2644 | 0.415 |
| B4T2048H16 | 4B | 7.9710 | 7.4286 | 6.2444 | 0.411 |
| B1T8192H16 | 4T | 8.1377 | 7.4911 | 6.2010 | 0.453 |
| B1T16384H16 | 8 | 16.1609 | 14.7390 | 12.1521 | 0.462 |

(RC1/R2/R3 columns are each round's own-window paired medians -- cross-
window comparison is directional only, per the campaign's window
discipline; the FINAL column is the R3 clean-window record, health-
certified fresh this round.)

Campaign trajectory at u=8: ratio 0.396 (RC1 window) -> ~0.43 (R2
window) -> 0.462 (R3 window); marginals 2.00 -> 1.812 -> 1.488 ms/u
(ratio-of-marginals 0.396 -> 0.439 -> 0.472); small end deepened
1.55 -> 1.72x (r2-arm -> r3-arm, R3 window).  R4's contribution is the
closure proof below, not a byte.

## 7. The definitive close-out statement

**The bwd u>=1 regime change is CLOSED AT THE PRECISION-CONSTRAINED
FLOOR, now on three independent grounds.**

1. **The budget ground (R2/R3):** the only bytes that could move the
   floor are the probe-KILLED f32 edges (u_all/vnew_all/dvp_all/
   dkB_sum/dk_local, 470 KiB/M-row); every graded output is f32 by the
   error budget; the 32-symbol cast class that the 1.33-1.49 RC1 dream
   needed was killed by the budget itself.
2. **The floor arithmetic (R3):** perfect execution of the current byte
   set caps u=8 at ~0.85-0.92x; deleting dqkwg+wyb+dhs OUTRIGHT caps at
   0.88.  The remaining ~2.4x to 1.1x is not reachable by storage
   precision, consumer nativity (R3 shipped the last native feed), or
   algorithm retirement (R3 deleted the deadest weight).
3. **The fusion ground (R4, this round):** the last byte-floor mover --
   the dqkwg+wyb fusion -- is pool-capped at 82% of its own byte
   ceiling (97024 vs 101376 B forces dropping dk_local), forces 1 CTA/SM
   against a measured 1.155-1.26x occupancy inversion, and nets a wash
   at best (buildable corner ~0.49x @u=8; even the unbuildable perfect
   fusion caps at ~0.54x).  NO-GO with the arithmetic.

The shipped architecture claim stands as the campaign's result: **the
small-shape + cold-start bwd winner (1.72x vs fla at B1T512H4,
window-matched) plus the u>=1 fwd winner; the u>=1 bwd is a
precision-constrained ~0.46-0.47x vs fla, and that is the floor's
number, not ours.**  If bwd work ever reopens, the directions with
unmeasured headroom are regime changes (B/T/H shapes outside this
ladder, learned-gate data regimes), not this chain at these shapes --
every idea class at B1T2048H16..B1T16384H16 is now priced or measured
closed.

## 8. Round record

* Coordination row `gdn-mlir R4` registered (GPU-COORDINATION.md).
* No code changes; `tools/gdn_bwd_gen.py` untouched; the r6/r10/r19/r22
  standing rules bind trivially (no new payload bytes).
* Provenance: cubin md5 audit 48/48 accounted (32 vs the R3 record, 16
  vs the R2 anchors with the two expected R3 stage deltas -- the bwd
  provenance rule; the fwd-side gdn_cache_key stamping does not apply to
  the bwd chain).
* Commit: this report + scratch/r4_* artifacts.
