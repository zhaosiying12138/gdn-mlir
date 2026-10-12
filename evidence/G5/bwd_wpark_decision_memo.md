# DECISION MEMO -- bwd P11R W-park: the quantified material for the three-way choice

Round 24 (R24-bwd, CPU-only by design) | HEAD de779b5 | 2026-10-08
Artifacts: `scratch/r24b/wpark_twin.py` (the fp64-twin W-park study,
seed 0, both bench shapes; raw JSONs `wpark_twin_T{2048,8192}.json`) and
`scratch/r24b/wpark_arith.py` (the exact-integer pool/traffic/time
arithmetic; output `wpark_arith_out.txt`).  Every input number cites its
round: dqkwg stage medians 0.5144/1.7289 ms (r22 paired x3), pool
47,616 B and the 2-CTA bar 50,176 B (r15), staging sweep geometry
(gdn_bwd_gen.py v4 P11 loop), DRAM 824.6 GB/s / L2-resident 2,144 GB/s
(static-r7 co-loaded lower bounds), P11R 1->2 measured gain 1.040x
(r16-Y), register/spill laws (r22 audit).

## 0. The question this answers

The r22 round-23 list closed every value-identical / spill / staging /
residency door at B1T2048H16 / B1T8192H16 and named ONE remaining bwd
class: new-numerics, with the P11R W-park as the flagship candidate
(priced there in one line: smem-park pool-blocked, global-park ~+10%).
This memo turns that line into decision material: the exact design, the
numpy-twin error bound against the current 6/6 margins, the pool and
traffic arithmetic against the measured 47.6 KB pool, the perf ceiling,
and the numerical-contract impact.

## 1. The W-park design (what parks W, where)

The dqkwg P11 dW/dq1 loop computes, per chunk m,

    dW = -(dvp @ Sn^T)        [C,K], accumulated per (ks, vs) block
                              (BK=BV=32 -> 4x4 blocks) as 3 split-bf16
                              mma links (vnh.bhh + vnh.bhl + vnl.bh)

with the vs-only tiles (akb = dO slice, vnh/vnl = dvp splits) staged
once per vs and shared by the P11R=2 ks chains (r16-Y).  The FULL
staging kill (stage them once per vs for ALL FOUR ks) requires four
per-ks accumulator sets live across the vs loop = 64 regs/thread on top
of the 174-reg natural body -- measured 1.026x only, spills eat it
(r16-Y P11R=4).  The W-park moves those accumulators out of registers:

* **WHERE W PARKS**: per-ks [C,BK] f32 park tiles -- one per ks chain.
  smem flavor: 4 tiles x 8,192 B = 32,768 B (f32) or 16,384 B (bf16);
  global flavor: read-modify-write the per-chunk dW_all windows
  (3 park-writes + 3 resume-reads per ks, [C,BK] f32 each).
* **THE RE-ASSOCIATION IT IMPLIES**: two numerical flavors.
  (a) *f32 park-resume* -- park the exact f32 accumulator bits, resume
  the same mma chain: per-output-element product order is unchanged,
  bit-identical by construction (the accCK hop pattern already proves
  the mechanism in-tree).  Not the interesting case.
  (b) *parked partials* -- each (ks,vs) computes its 3-link partial
  from a ZERO accumulator, parks it, and the vs partials are summed at
  last_vs.  Per element this re-groups
  `((((0+l0)+l1)+l2 + l0')+l1')+l2')...` into
  `sum_vs [ ((0+l0)+l1)+l2 ]_vs` -- a genuine fp re-association.
  Parking the partials as **bf16** (halves park traffic) additionally
  ROUNDS each parked partial -- the only flavor that changes values.

## 2. The error bound on the numpy twin (both shapes)

`wpark_twin.py` runs the full fp64 twin (gdn_bwd_verify math, seed 0)
with FOUR dW variants propagated through a SHARED reverse walk -- the
dH recurrence is dW-independent, so the graded deltas isolate the dW
class exactly.  The kernel-side variants simulate the shipped split-bf16
link structure in f32 (hi = bf16(x); lo = bf16(x-hi), the
stage_split_ew recipe).  dq/dv/dh0 have no dW term (0 by construction);
the dW cone is {dk, dbeta, dg} via TW = Tm^T dW, G = ... + dW Kbg^T,
dAf.

| quantity | T2048 | T8192 |
|---|---|---|
| \|dW\|max (context) | 2.92 | 3.54 |
| dW error, current association vs fp64 | 2.22e-5 | 2.05e-5 |
| dW error, wpark_f32 vs fp64 | 2.23e-5 | 2.00e-5 |
| dW error, wpark_bf16 vs fp64 | **7.7e-3** | **6.0e-3** |
| NEW graded error, wpark_f32 vs current: dk / dbeta / dg | 3.2e-9 / 1.4e-9 / 1.1e-9 | 4.4e-9 / 2.4e-9 / 1.5e-9 |
| NEW graded error, wpark_bf16 vs current: dk / dbeta / dg | **4.3e-5 / 2.7e-5 / 1.9e-5** | **9.4e-5 / 4.4e-5 / 3.1e-5** |

Against the r19-final margins and the fla contract bars:

| output | bar | margin T2048 / T8192 | slack T2048 / T8192 | wpark_bf16 adds | headroom used |
|---|---|---|---|---|---|
| dk | 8e-3 | 1.284e-3 / 1.881e-3 | 6.72e-3 / 6.12e-3 | 4.3e-5 / 9.4e-5 | **157x / 65x under** |
| dbeta | 2e-2 | 8.527e-4 / 1.654e-3 | 1.91e-2 / 1.83e-2 | 2.7e-5 / 4.4e-5 | 712x / 413x under |
| dg | 2e-2 | 3.274e-4 / 1.145e-3 | 1.97e-2 / 1.89e-2 | 1.9e-5 / 3.1e-5 | 1025x / 618x under |

Readings:
* The error is PER-CHUNK (dW is produced and consumed within the
  chunk): the T8192 deltas (4x the chunks) are the same order as T2048
  -- no cross-chunk accumulation.  (Measured, not assumed.)
* The pure re-association (f32 parks) is numerically a NON-EVENT:
  ~1e-9 on graded outputs, six orders under the margins.
* Even the lossy bf16 park stays 65-1025x under the bars and costs at
  most 5% of the current dk margin.  The r2-era worry ("a bf16 floor
  on dW is 10x worse than the dbeta bar") does NOT transfer to the
  partial-park form: rounding four partials that sum to the f32 dW
  bounds the perturbation to ~0.5-1 bf16 quantum of the partial
  magnitudes, ~2.6e-3 relative on |dW|~3, and the dAf/TW propagation
  attenuates it (~8x, the r2 sensitivity datum).
* Caveat, stated: the downstream here is fp64 (class isolation).  On
  GPU, dW's consumers are themselves split-bf16 gemms; the perturbed
  dW enters through their operand splits at the same attenuation.  The
  twin-prediction -> GPU match precedent is r5 (twin 2.2e-8 predicted /
  2.2e-8 measured for the fwdscan re-association class).

## 3. The pool / traffic arithmetic (vs the measured 47.6 KB pool)

**smem-park: DEAD on pool, every flavor.**

| park flavor | bytes | pool becomes | vs 50,176 B 2-CTA bar |
|---|---|---|---|
| dw-only, 4 ks, f32 | 32,768 | 80,384 | DEAD (1 CTA/SM) |
| dw-only, 4 ks, bf16 | 16,384 | 64,000 | DEAD (1 CTA/SM) |
| dw+dq1, f32 | 65,536 | 113,152 | DEAD (> 101,376 per-block optin) |
| reuse the accCK window | 8,192 | 55,808 | DEAD; and one window cannot host 4 sets |

Headroom is 2,560 B -- under ONE park tile (8,192 B) in the best case.
Falling back to 1 CTA/SM surrenders the r15 2-CTA conversion measured
at 1.46x (0.937 -> 0.642 ms): an instant -32% on the stage.  The r22
"+16KB vs 2.5KB headroom" line is confirmed and sharpened: the MINIMUM
realizable park is +16,384 B (bf16, dw-only).

**global-park: priced, net NEGATIVE traffic.**

Per CTA per chunk: the full kill saves 4 staging sweeps x 12,288 B of
global reads (akb 4,096 + dvp split source 8,192) = 49,152 B (plus
49,152 B of smem writes).  The f32 park-resume adds 3W+3R of [C,BK] f32
per ks x 4 ks = 196,608 B (the r22 "~+190KB", derived exactly); the
bf16 partial-park adds 98,304 B.  Net: f32 +147,456 B, bf16 +49,152 B
per chunk-CTA -- and the added traffic is 8 KB read-modify-write
windows (RAS-hostile) replacing streaming reads.

Aggregate / time (M = BH*NC CTAs; co-loaded calibration bounds):

| flavor, shape | net added traffic | added time (L2 2,144 / DRAM 824.6 GB/s) | dqkwg verdict |
|---|---|---|---|
| f32, T2048 | +75.5 MB | +35 / +92 us on 514 us | 0.936x / 0.849x REGR |
| bf16, T2048 | +25.2 MB | +12 / +31 us | 0.978x / 0.944x REGR |
| f32, T8192 | +302.0 MB | +141 / +366 us on 1,729 us | 0.925x / 0.825x REGR |
| bf16, T8192 | +100.7 MB | +47 / +122 us | 0.974x / 0.934x REGR |

The r22 "~+10%" estimate is confirmed at the DRAM end (15-18% f32,
7% bf16).  Even the L2-optimistic bound is a regression.

## 4. The perf ceiling (best case, park FREE)

The only thing the W-park buys is the P11R 2->4 staging-kill remainder.
r16-Y measured P11R 1->2 (killing EIGHT redundant sweeps) at 1.040x;
2->4 kills four more -- at most half the gain, and only if the park
costs literally nothing:

* T2048: dqkwg 514 -> ~504 us (**1.021x**), e2e 2.16 -> 2.15 ms
  (**-0.48%**).
* T8192: dqkwg 1,729 -> ~1,695 us (**1.020x**), e2e 7.68 -> 7.65 ms
  (**-0.44%**).

That ceiling is (a) below the ~1.04x-at-stage bar the r16 P11R=2 lever
needed to ship, (b) below the 1.05x chain bar, and (c) UNBUILDABLE --
every realizable park is a 0.83-0.98x REGR (section 3).  The register
side offers no secondary win either: the r22 audit closed the
forced-cap allocator door (never 0-spill at any cap) and K32 proved
spill-class cuts are not a lever (4th data point), so the park's
register relief (4 sets -> 1) has no measurable value to redeem it.

## 5. Numerical-contract impact

**No tolerance-relaxation sign-off is required by either flavor.**
The f32 re-association is ~1e-9 on graded outputs; the bf16 partial-park
is 65-1025x under every bar and <= 5% of the tightest current margin
(dk @T8192).  Both would verify 6/6 inside the existing fla-derived
contract.  (The bf16 global-scratch park would additionally need a
harness note -- a new internal scratch global -- but dW_all itself
already exists as the park target for the f32 flavor, so even that is
avoidable.)  The blocker is exclusively performance and pool.

## 6. Recommendation

**NO-GO, now decision-grade.**  The W-park is numerically admissible
and structurally dead: pool headroom 2,560 B vs a >= 16,384 B minimum
park; global parking costs 1-3x more traffic than the staging it saves;
and the unbuildable FREE variant tops out at 1.02x on a stage that is
19-24% of the chain -- under every bar this project has ever shipped a
lever at.  For the three-way choice: the "new-numerics (W-park)" branch
of the r22 round-23 list is CLOSED at these shapes with quantified
cause; nothing in this memo disturbs the r19-final bytes or the hold
recommendation.  If bwd work ever reopens, the only directions with
unmeasured headroom are regime changes (B4+/T-other/H-other shapes) --
at B1T2048H16/B1T8192H16 the remaining idea classes are, to the best of
the project's current knowledge, exhausted.
