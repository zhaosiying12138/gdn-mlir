# Tolerance sensitivity: the dk 8e-3 -> 2e-2 sweep (decision material for the pending bwd u>=1 choice)

CPU-side, non-destructive analysis for the PENDING user decision (the
bwd u>=1 AskUserQuestion, 4 attempts unanswered; standing directive
"全覆盖全部 shape 都要战胜它").  No GPU touched; no shipped code or
records modified (`scratch/r2_bf16_probe.py` imported READ-ONLY, md5
8e6889e78731ffbbe3b17ffd8897e24d unchanged; tools/ clean).  This
document does not pick a side -- it prices the tolerance axis so
either answer to the pending question is decision-grade.

The question the sweep answers: R2 classified 16 M-row symbols into
10 bf16-castable / 6 killed under the shipped contract (dq/dk/dv <=
8e-3, dbeta/dg <= 2e-2, dh0 <= 8e-3).  If the dk contract were relaxed
to 1.0e-2 / 1.2e-2 / 1.5e-2 / 2.0e-2 -- which killed symbols come
back, what bytes move, and where does the u=8 floor ceiling land?

**Headline: NOTHING returns anywhere in the surveyed band.  The killed
symbols' dk solo increments sit at 4.6-6.9e-2 across ALL four ladder
shapes (5.7-8.7x the current 8e-3 contract, 4.6-6.9x even the solo bar
of the 2.0e-2 tier) -- the band is a factor of ~6-9 too tight.  The
first unlock (u_all+vnew_all) needs dk >= 6.6e-2 at set level (8.2x);
the full five-symbol class needs dk >= 1.19e-1 (14.9x) AND dbeta >=
4.8e-2 (2.4x, a different axis: dvp_all is dbeta-killed, not dk);
and even the full unlock only moves the u=8 perfect-execution ceiling
from 0.916 to 1.08-1.15 (conservative vs multiplicity-honest byte
count).  Tolerance relaxation inside 2e-2 buys zero bytes; tolerance
relaxation to the unlock tiers buys a ceiling that still requires the
~2x execution gap to close before any ladder cell actually flips.**

Artifacts: `scratch/tol_sweep/` -- `tol_sweep.py` (the sweep script;
imports the R2 staged twin verbatim), `sweep_{B1T512H4,B1T2048,B4T2048,
B1T16384}.json` (the new 4-shape measurements), `tier_out.txt/.json`
(the classification + floor arithmetic).  Staged via
`.gdn-mlir-staging/tol/` (Windows side).

## 1. Method

The R2 probe's staged fp64 twin (`scratch/r2_bf16_probe.py`) is
imported unchanged: same seed-0 recipe, same bf16 round-trip edges
(truncf at store / extf at read), same increments-comparison
methodology.  Cross-check: the T512H4/T2048 runs reproduce the R2
records DIGIT-IDENTICALLY (u_all solo dk 5.084e-2/4.567e-2 = the R2
gate json; cast10 set 4.508e-4/1.012e-3 and the dvp set = the R2
classify json).  New in this round: the 5 killed-symbol solos and the
expanded set-level increments at ALL FOUR set-level shapes (R2 had
solos only at the two gate shapes), plus the tier classification and
the floor arithmetic under each tolerance tier.

Classification discipline (R2's, held fixed): solo increment <= tier/2
(leaves headroom for the rest of the set); shipped-set requirement is
the set-level projection kernel-digit + set-increment <= tier
(linear-sum, conservative).  Kernel digits: dk {T512H4 8.9e-4 (bounded
by the measured r3 total), T2048 1.3e-3, B4 2.2e-3 (R2's projections
minus probe increments), T16384 3.8e-3 (the r19 f32 record)}; dbeta
~1e-4 (the R2/R3 probe-vs-GPU validation showed measured dbeta digits
equal the probe increments).  These +-2e-3 uncertainties do not move
any verdict below (the increments in play are 4.6e-2..1.2e-1).

## 2. The measured increments (the new data)

Killed-symbol SOLO increments, all four shapes (max-abs vs the
unquantized twin; the two axes that ever bind):

| symbol | T512H4 | T2048 | B4 | T16384 | worst |
|---|---|---|---|---|---|
| u_all dk | 5.08e-2 | 4.57e-2 | **6.37e-2** | 4.93e-2 | 6.37e-2 @B4 |
| u_all dg | 3.17e-3 | 1.01e-2 | **2.19e-2** | 1.55e-2 | 2.19e-2 @B4 |
| vnew_all dk | = u_all | = u_all | = u_all | = u_all | 6.37e-2 |
| dvp_all dk | 1.12e-3 | 1.24e-3 | 3.64e-3 | 1.50e-3 | 3.64e-3 (fits 8e-3 NOW) |
| dvp_all dv | 5.41e-3 | 5.67e-3 | 6.58e-3 | 5.91e-3 | 6.58e-3 (fits 8e-3 NOW) |
| dvp_all dbeta | 1.99e-2 | **4.76e-2** | 4.42e-2 | 2.57e-2 | 4.76e-2 @T2048 |
| dkB_sum dk | 6.04e-2 | 6.14e-2 | 6.24e-2 | 6.07e-2 | 6.24e-2 |
| dk_local dk | 5.93e-2 | 6.23e-2 | **6.94e-2** | 6.16e-2 | 6.94e-2 |

Readings: (a) the dk-axis solos are FLAT across shapes (+-20%, worst
at B4) -- shape-scaling does not open a back door; (b) u_all and
vnew_all are numerically the SAME symbol (identical digits: the vn
signal feeds dkB directly); (c) dvp_all's binding axis is dbeta
(2.4x the contract), NOT dk -- its dk/dv digits already fit the
CURRENT contract; (d) u/vnew's dg solo at B4 (2.19e-2) pokes 10% over
the held 2e-2 dg contract (the uvn SET dg stays under: 1.84e-2).

SET-level increments (cast10 baseline; the shipping arithmetic):

| set | T512H4 | T2048 | B4 | T16384 |
|---|---|---|---|---|
| cast10 (shipped class) | dk 4.51e-4 | 1.01e-3 | 1.31e-3 | 1.70e-3 |
| +u_all+vnew_all | dk 5.08e-2 | 4.57e-2 | **6.38e-2** | 4.93e-2 |
| +dkB_sum+dk_local | dk 9.78e-2 | 1.12e-1 | **1.15e-1** | 1.04e-1 |
| +dvp_all | db 1.99e-2 | **4.77e-2** | 4.44e-2 | 2.67e-2 (dk <= 3.6e-3) |
| +all five | dk 9.56e-2 / db 1.99e-2 | 1.12e-1 / 4.77e-2 | **1.17e-1 / 4.44e-2** / dg 2.002e-2 | 1.04e-1 / 2.67e-2 |

New structural fact: the dk pair is SUPERADDITIVE in set -- solos
6.2e-2 + 6.9e-2 (max-based, not summed) but the pair lands at
1.15e-1 @B4: the two round-trips hit the same cancelling dk sums
(dk-local terms are O(10) cancelling to O(0.1), the R2 finding) and
their errors align.  And the all5 set's dg increment at B4 lands at
2.002e-2 -- 0.1% OVER the held 2e-2 dg contract (a dg tier of 2.1e-2
clears it).

## 3. The tier table (the mission's sweep)

dk contract swept; dbeta/dg/dv contracts HELD; solo bar = tier/2:

| dk tier | solo bar | killed symbols that return | +bytes/M-row | floor ms/u | u=8 e2e floor | u=8 ceiling | u=1 ceiling |
|---|---|---|---|---|---|---|---|
| 8.0e-3 (shipped) | 4.0e-3 | none (the R2 verdict) | 0 | 0.750 | 6.14 | 0.916 | 0.981 |
| 1.0e-2 | 5.0e-3 | **none** | 0 | 0.750 | 6.14 | 0.916 | 0.981 |
| 1.2e-2 | 6.0e-3 | **none** | 0 | 0.750 | 6.14 | 0.916 | 0.981 |
| 1.5e-2 | 7.5e-3 | **none** | 0 | 0.750 | 6.14 | 0.916 | 0.981 |
| 2.0e-2 | 1.0e-2 | **none** | 0 | 0.750 | 6.14 | 0.916 | 0.981 |

The band is flat because the nearest killed dk solo (u/vnew 6.37e-2
@B4) is 6.4x the widest solo bar in the band.  A 2.5x contract
relaxation (8e-3 -> 2e-2) admits zero new bytes and moves no ceiling.
Floors: the R3/R4 byte-floor model (total chain traffic 1.70 MiB/M-row;
r7 1.074 ms/u scaled by remaining bytes; the R3 record 0.75 ms/u after
the shipped 9-symbol cast + ARET; fixed e2e term 0.14 ms; fla = the
final clean-window records 5.6247 ms @u=8 / 0.8729 @u=1).

## 4. The unlock ladder and the relaxation -> win curve

What each symbol actually needs (worst shape):

| unlock | set-level tier requirement (kernel + increment) | strict solo-bar reading | bytes unlocked B/M-row (r2 count / r4 count) |
|---|---|---|---|
| u_all + vnew_all | **dk >= 6.6e-2** (8.2x) | dk >= 1.27e-1, dg >= 4.4e-2 | 81,920 / 81,920 |
| dkB_sum + dk_local | **dk >= 1.17e-1** (14.6x; superadditive) | dk >= 1.25-1.39e-1 | 65,536 / 65,536 |
| dvp_all | **dbeta >= 4.8e-2** (2.4x; dk/dv already fit) | dbeta >= 9.5e-2 | 49,152 / 114,688 (16x re-reads) |
| all five | **dk >= 1.19e-1 AND dbeta >= 4.8e-2** (+ dg 2.1e-2 hair @B4) | -- | 196,608 / 262,144 |

Byte counts: the killed class' f32 edge traffic per M-row is 384 KiB
(r2 count: u_all 2 edges [C,V], vnew 3, dkB 2 [C,K], dkl 2, dvp 3) to
512 KiB (r4 count: the R4 payload audit prices dvp_all at 229,376 B --
dqkwg re-reads its own store 16x in slice stages -- vs 98,304 at
single-count).  Casting halves whatever is counted.

The win curve (u=8; u=1 in parens; ceiling = perfect-execution bound
of exactly these edges):

| unlocked set | Snew B/row | floor ms/u | u=8 e2e floor | ceiling @u=8 (@u=1) |
|---|---|---|---|---|
| none (shipped) | 0 | 0.750 | 6.14 | 0.916 (0.981) |
| uvn only | 81,920 | 0.701 | 5.75 | 0.979 (1.038) |
| dkpair only | 65,536 | 0.711 | 5.82 | 0.966 (1.026) |
| dvp only (r4 count) | 114,688 | 0.681 | 5.59 | 1.007 (1.063) |
| dk-axis four (no dvp) | 147,456 | 0.661 | 5.43 | 1.036 (1.090) |
| all five (r2 count) | 196,608 | 0.632 | 5.19 | 1.083 (1.131) |
| all five (r4 count) | 262,144 | 0.592 | 4.88 | **1.153 (1.192)** |
| + graded outputs too (the RC1 dream class) | 294,912 | 0.572 | 4.72 | 1.192 (1.225) |

**The 1.1x line at u=8 needs Snew >= ~213 KB/row (208 KiB) beyond the
shipped set** (floor <= 0.6217 ms/u).  Sensitivities: the fixed-term
band (0 -> 0.14 ms) moves the bar 184 -> 213 KB; the fla window (+-10%,
the campaign's window discipline) moves it proportionally.  So:

* A dk-ONLY relaxation can NEVER reach 1.1 at u=8 -- the four dk-gated
  symbols total 144 KiB < the 184-213 KiB bar (ceiling caps at 1.036).
  Crossing requires dvp_all too, i.e. a dbeta relaxation to >= 4.8e-2
  regardless of what dk does.
* With dvp admitted, the conservative count (192 KiB) is BORDERLINE
  (1.083 at fixed=0.14; 1.11 at fixed=0) and the multiplicity-honest
  count (256 KiB) clears (1.153).  There is no partial subset that
  clears robustly: uvn+dvp(r4) = 196 KiB, dkpair+dvp(r4) = 180 KiB --
  all under or at the bar's edge.

## 5. The two caveats that bound the whole exercise

1. **A ceiling is not a measurement.**  The floors above assume 100%
   of the byte model at perfect execution.  The shipped chain runs at
   ~2x its floor (12.15 vs 6.14 ms @u=8; the R3 finding: the residual
   is issue-bound sweep/conversion machinery, not bytes), and R2's
   measured lesson is that u>=1 stages converted a 26% byte cut into
   only +1-2% time.  Even the full unlock (ceiling 1.153) leaves
   realized time at ~2x floor = ~9.8 ms = ratio ~0.58 until the
   execution gap closes.  Tolerance moves the CEILING, not the
   leaderboard.
2. **The fusion interplay (R4).**  Casting dvp_all/dk_local and
   removing their edges are substitutes: the R4 fusion kill-set prices
   those same edges (229,376 + 65,536 B/row); a cast halves the
   remaining fusion prize, and the fusion's NO-GO verdict (pool cap +
   1CTA occupancy inversion) is unchanged by tolerance.  The two
   levers should not be double-counted.

## 6. The decision paragraph (costs/gains of each option -- no side taken)

The pending choice is whether the bwd u>=1 regime-change campaign
reopens on the tolerance axis.  The arithmetic above prices it:

* **Hold the shipped contract (dk/dv 8e-3, dbeta/dg 2e-2):** zero
  cost, zero new bytes; the R3/R4 close-out verdict stands unchanged
  (u=8 ceiling 0.916, measured 0.462; small end 1.72x).  Nothing in
  this sweep disturbs that record.
* **Relax dk inside the surveyed band (<= 2.0e-2):** pure contract
  cost, ZERO return -- measured flat across all four shapes and all
  four tiers.  This option is closed by the data, not by judgment.
* **Relax to the unlock tiers (dk >= ~1.2e-1, dbeta >= ~4.8e-2,
  i.e. ~15x/2.4x the shipped bars):** admits the full five-symbol
  killed class (192-256 KiB/M-row), moves the u=8 perfect-execution
  ceiling to 1.08-1.15 and the u=1 ceiling to 1.13-1.19.  The price:
  the end-to-end dk error contract loosens ~15x (projected worst-set
  digits ~1.2e-1 vs the shipped 5.0e-3 measured worst), dbeta ~2.4x
  (~4.8e-2 vs 5.2e-3), a dg hair at B4 (2.1e-2), PLUS the ~2x
  execution-gap closure that must land before ANY cell flips -- the
  ceiling crossing is necessary, not sufficient.  Partial unlocks
  (uvn at 8.2x, or any dk-only relaxation) do NOT reach 1.1 anywhere.

In one sentence for the pending question: the tolerance axis is a
step function, not a dial -- nothing until ~8x on dk, no 1.1x at u=8
without ~15x on dk AND ~2.4x on dbeta together, and even then only as
a perfect-execution ceiling that the issue-bound residual has not yet
earned.  Whichever way the user decides, the numbers above are the
trade.

## 7. Provenance

* CPU only; conda `gdn` (numpy 2.5.3); WSL; no GPU allocations, no
  timing claims, no window exposure.
* `scratch/r2_bf16_probe.py` and all shipped records UNTOUCHED (the
  twin is imported; md5 8e6889e78731ffbbe3b17ffd8897e24d).
* Inputs: R2 probe records (gate/classify jsons; digit-reproduced
  here), the r3 verify digits (kernel-digit anchors), the R3/R4 floor
  model constants (1.70 MiB/M-row, 1.074 ms/u, 0.75 ms/u, fixed 0.14,
  fla 5.6247/0.8729), the R4 edge audit (dvp multiplicity).
* Commit: this report + `scratch/tol_sweep/`.
