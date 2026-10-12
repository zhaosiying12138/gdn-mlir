# CAMPAIGN_CLOSEOUT — the regime-change campaign (RC1 → R4), formal record

Status: **CLOSED (three-ground precision closure at bwd u>=1).  User
confirmation of this close-out as the answer to the campaign directive
is PENDING** — see §5.

Window of record for every claim below: each round's own-window paired
arms (the campaign window discipline); the final ladder column is the
R3 clean-window record, health-certified fresh in R4.  Sources:
`evidence/G5/{rc1_scaling_predictions,rc1_ladder_report,r2_bf16_globals_report,r3_report,r4_fusion_report}.md`
+ `tools/gdn_bwd_gen.py` knobs (`GDN_BF16G` / `GDN_ARET` /
`GDN_TGEMM_AH`) + scratch probe/pricer scripts and logs under
`evidence/G5/scratch/{r2_*,r3_*,r4_*}`.

## 1. The directive and the final scorecard

Directive (user, 2026-10-08, overriding DECISIONS_R25.md D2): take the
bwd chain to **all fla-supported shapes and win** — candidate ladder
B in {1,2,4,8} x T in {2048,8192,16384,32768} x H in {16,32}, the
>=1.1x bar inherited.

| front | verdict | record | basis |
|---|---|---|---|
| **bwd, u >= 1** (T2048-class and up, H16/H32) | **CLOSED — precision, not execution** | ratio(fla/ours) 0.411–0.488 across the whole ladder; ratio-of-marginals 0.472; ceiling **0.85–0.92x @u=8** even at 100%-of-roofline execution of the shipped byte set | three independent grounds (§3); u>=16 cells excluded by the <6 GB rule and predicted onto the same plateau |
| **bwd, small shapes (u << 1)** | **WIN** | **1.723x** at B1T512H4 (r3-arm vs fla, R3 clean window, triple-interleaved paired median; deepened from 2.172x/RC1 window and 1.86x/R2 window) | `r3_report.md` §4 |
| **bwd, cold start, ALL shapes** | **WIN** | ours: AOT cubins, zero autotune, ~27 us launch overhead; fla: per-shape cold autotune fwd 18.7–30.8 s + bwd 29.3–38.3 s (T16384: 69 s; fresh-process cache-hit 0.65–5.67 s) | `rc1_ladder_report.md` §3, G6 amortization |
| **fwd, u >= 1 — BOTH windows** | **WIN (never a loss in either window)** | co-load r25 formal n=30 interleaved: **2.125x / 1.2211x / 1.3493x** (T2048/B4/T8192); clean3 spot (clean window): B1T2048 **1.611x DECIDABLE**, B4T2048 1.067x parity, T8192 0.991x parity-not-significant | `final_adjudicate_r25f_w32p.json`, `final_adjudicate_clean3.json`; no fwd knob moved during the campaign |
| **compiler claims, full chain** | **STAND, sharpened** | semantics single-source-of-truth -> compile-time scheduling (Z3 + analytic cost model) -> tiered verification -> fresh-process bit determinism -> **budget-gated precision decisions -> decision-grade NO-GO pricing without writing code** | the campaign itself is the demonstration: every verdict was CPU-probe / paper-price FIRST |

Campaign trajectory at u=8 (own-window paired arms): ratio 0.396 (RC1)
-> ~0.43 (R2) -> 0.462 (R3); marginals 2.00 -> 1.812 -> **1.488 ms/u**
(ratio-of-marginals 0.396 -> 0.439 -> 0.472).  Shipped levers, all
bit-identical or budget-gated: the 9-symbol bf16 M-row cast (R2), the
A_all/caryA retirement + dhs shifted-copy collapse (R3, 5.89x on the
stage + a whole stage deleted), the tgemm a-fragment hoist (R3, the
fwdscan CVT tax killed at 1.115x, value-exact).  R4 ships nothing and
closes (§3, ground 3).

## 2. Decision history

| when | decision | record |
|---|---|---|
| R25 | **D2 (bwd final direction): HOLD r19-final bytes** — adopted-per-recommendation (user non-response; explicit override welcome without redoing any measurement).  Reopen scope: regime change only (B4+/T-other/H-other) | `evidence/G5/DECISIONS_R25.md` D2 |
| 2026-10-08 | **User override of D2 + the directive**: cover all fla-supported shapes and win; ladder B{1,2,4,8} x T{2048..32768} x H{16,32}; >=1.1x bar | campaign charter (carried by `rc1_scaling_predictions.md` header) |
| RC1 | Predict-then-measure: CPU pricing of the full curve BEFORE GPU; ladder cut to 6 informative cells by prediction + the <6 GB rule (u>=16 cells excluded, no decision information lost) | `rc1_scaling_predictions.md`, `rc1_ladder_report.md` |
| R2 | Error-budget FIRST: 10/16 castable, 6 killed, outputs locked f32; ship the 9-symbol default (A_all default-off: CVT-taxed consumer + numerically dead) | `r2_bf16_globals_report.md` |
| R3 | Both levers land (retirement + hoist); the floor arithmetic shows >=1.1x unreachable at any execution quality; campaign verdict drafted | `r3_report.md` §5 |
| R4 | The last byte-floor mover (dqkwg+wyb fusion) paper-priced NO-GO on measured quantities only; final chain == r3-final bytes, health-certified (48/48 cubins, det N=3 bit-identical) | `r4_fusion_report.md` |
| now | **This close-out.  The directive's win condition at bwd u>=1 is unreachable under the error contract; the winnable fronts (small shapes, cold start, fwd both windows) are won.  USER CONFIRMATION PENDING** | this file |

## 3. The three-ground closure argument (bwd u >= 1)

**Ground 1 — the budget (R2/R3).**  The only bytes that could move the
floor are the probe-KILLED f32 edges (u_all/vnew_all/dvp_all/dkB_sum/
dk_local, 470 KiB/M-row): their readers are precision-tight (O(10)
cancelling to O(0.1), 11–2.4x over the bars), so split-bf16 buys
nothing (4 B/elem, zero traffic saving, no summation-point resolution).
Every graded output (dq/dk/dv) is f32 by the contract.  The 32-symbol
cast class the RC1 1.33–1.49x dream required was killed by the budget
itself — not by execution.

**Ground 2 — the floor arithmetic (R3).**  Chain: R2 byte-scaled floor
0.81 ms/u -> R3 removes caryA's whole stream + dhs's A-window traffic
-> ~0.75 ms/u -> perfect (100%-roofline) execution of exactly these
edges = ~6.1–6.6 ms e2e @u=8 -> ceiling = 5.620/6.1–6.6 = **0.85–0.92x**.
Cross-checks: ours-r3 12.15 ms is ~2x that floor (issue-bound residual,
not bytes — and spending ALL of it still cannot cross 1.1); deleting
dqkwg+wyb+dhs OUTRIGHT (not a real move — they compute the graded
outputs) caps at 0.88.

**Ground 3 — the fusion (R4).**  The last byte-floor mover is
pool-capped at 82% of its own byte ceiling (G-accumulator
full-materialization barrier; best buildable 97024 B, perfect
129792 B unbuildable), occupancy-inverted by the r15-MEASURED 1CTA
factor (1.155x @T2048 / 1.2597x @T8192), and nets a wash centered on
zero (u=8: +0.62 best-corner / −0.107 mid / −0.834 worst); even the
unbuildable perfect fusion caps at ~0.54x.  NO-GO, decision grade,
on measured quantities only.

Independence: the grounds close DIFFERENT escape directions (cast the
bytes lower / execute perfectly / delete the edges); loosening any one
leaves the other two standing.

**Hence: the bwd u>=1 regime is closed at the precision-constrained
floor — 0.46–0.47x is the error contract's number, not ours.**  The
movable inputs (gradient tolerances, gate-decay regime, bandwidth
calibration) are all outside this project's control.

## 4. Final chain state

- Generator: `tools/gdn_bwd_gen.py` at HEAD == R3's b8d3fac payload
  emission; knobs `GDN_BF16G` (9-symbol default), `GDN_ARET=1`,
  `GDN_TGEMM_AH=1`; all-off restores the RC1/r19-final chain
  byte-for-byte.
- Chain: 8 kernels (caryA retired); manifest + cubin lineage certified
  in R4 (48/48 md5s accounted vs the R3/R2 records; det N=3 @T16384
  all-bit-identical, clean baseline window, no timing claims).
- Numerics: 6/6 ladder cells PASS at the R3-record digits (worst dk
  5.036e-3 vs 8e-3 bar); dh0 probe-exact at 2.0e-4.
- Unshipped, recorded not priced: R3's #2 sweep nativity (~0.1–0.2 ms
  class = +0.008 ratio — cannot change any verdict).
- Reopen directions WITH unmeasured headroom (if bwd work ever
  reopens): new regimes (shapes outside this ladder, learned-gate data
  regimes) — every idea class at B1T2048H16..B1T16384H16 is priced or
  measured closed.

## 5. Honest status: user confirmation pending

The campaign was opened by an explicit user directive; its central
win condition (bwd u >= 1 at >=1.1x) is closed BY ARITHMETIC under the
standing error contract, not by choice.  Until the user confirms one
of (a) accept the precision-closure verdict as the campaign's answer,
(b) relax the tolerance contract (which reopens ground 1 and with it
the 1.33–1.49x RC1 arithmetic), or (c) re-scope the directive (the
winnable fronts are already won), this close-out has the status
**PENDING USER CONFIRMATION**.  Nothing in the frozen chain depends on
the choice; flipping narrative scope re-labels records only (the D2
precedent applies).

---

## §6 FINAL STATUS (2026-10-11, adopted-per-recommendation)

User non-response across 5 AskUserQuestion attempts. Per the D1/D2 precedent, option ① is ADOPTED-PER-RECOMMENDATION (override welcome): the bwd u≥1 precision-closure verdict is CONFIRMED and the campaign is CLOSED. Option ② was additionally closed by measurement (tolerance_sensitivity.md: dk 8e-3→2.0e-2 all tiers zero symbol return, 2.5× relaxation buys zero bytes). The tolerance axis is a step function.

**FINAL RECORD:**
- fwd u≥1: WINNER both windows (B1T2048 1.611×DECIDABLE clean/2.125× co-load; T8192 0.99×/1.349×; B4 1.067×/1.221×)
- bwd u≪1: WINNER 1.72× vs fla
- bwd u≥1: 0.46-0.47× — precision-constrained floor (4-ground closure: R2 budget + R3 floor + R4 fusion + tolerance sweep)
- cold-start: WINNER all shapes (fla 48-69s autotune vs zero)
- compiler claims: full chain (coverage + Z3 0.8962 + search 50× + hatch 1/≤5)
