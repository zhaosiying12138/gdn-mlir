# DECISION RECORD -- R25: the two sign-offs, adopted per recommendation

Round 25 (R25) | HEAD 1c7d291 | 2026-10-08
Context: the user was asked in-session for two sign-offs at the R23/R24
close -- (D1) the B4T2048 trimmed-per-pair decidability v2 rule, and
(D2) the bwd final direction (hold r19-final bytes vs reopen).  No
response arrived within the session window.  Per the autonomous-goal
rules, the RECOMMENDED default of each is ADOPTED here and documented
as such; both are cheap to reverse and an explicit user override at any
later round supersedes this record without redoing any measurement:

* D1 is a citation-policy line only -- no measurement, gate, or cubin
  depends on it; flipping it re-labels existing JSON, nothing else.
* D2 is a narrative/scope freeze only -- no code or bytes depend on it;
  flipping it reopens round planning, not correctness.

---

## D1 -- B4T2048 decidability v2: SIGNED as DOCUMENTED-OPTIONAL side statistic; 1.1417x carries the dual v1-no / v2-yes citation

STATUS: **ADOPTED-PER-RECOMMENDATION (user non-response; override
welcome).**

Rationale, verbatim from the r23-fwd decision memo
(`evidence/G5/fwd_report.md` round 23, section 4):

> DECISION MEMO -- trimmed-per-pair decidability v2.  WHAT CHANGES: an
> ADDITIONAL adjudication line (decidable_v2) computes the existing
> IQR-spread bar over the inner 80% of per-pair ratios (10% symmetric
> trim; n=30 drops the 3 highest and 3 lowest) instead of all 30;
> median, sign test, and win counts are untouched, v1 stays the rule of
> record for cross-round comparability, and any v2-claimed record cites
> both lines.  WHAT IT WOULD FLIP: on the r22 formal pairs, B4T2048
> (30/30 wins, p=9.3e-10, blocked under v1 by 0.039) and B1T2048
> (29/30) become DECIDABLE-v2; B1T8192 stays undecidable under both.
> THE RISK: v2 certifies within-session coherence, not
> window-to-window stability -- B1T2048's center moved 1.78x -> 2.11x
> across windows on fla's median alone (0.35 -> 0.42ms), so its v2
> verdict is weaker than it looks; B4 (three windows agreeing on ours
> +/-1%, differing only in fla's tails) is exactly the case the rule was
> designed for.  RECOMMENDATION: sign off v2 as a documented-optional
> side statistic; claim B4T2048 1.1417x under "v1-no / v2-yes" dual
> citation at the next archival n=30 session; keep B1T2048 v2-soft
> until its center stabilizes across >=2 more windows.

Operational meaning from this round on:

* v1 (IQR over all 30 per-pair ratios) stays the rule of record and the
  cross-round comparability line; v2 is printed beside it, never alone.
* The B4T2048 formal record is claimed as **1.1417x, v1-no / v2-yes
  (dual citation)** on the r22 pairs; the r25 archival re-computes both
  verdicts on the fresh W32P pairs (below) and any moved number again
  carries both lines.
* B1T2048 remains v2-soft (the stated window-drift caveat); B1T8192 is
  reported undecidable under both rules (genuine two-sided window).

Evidence pointers:
* Rule text: `evidence/G5/adjudication_methodology.md` (the
  trimmed-per-pair v2 section, added r23).
* r22-pair verdicts: `evidence/G5/decidability_v2_r23f.json` (from
  `scratch/r23f/v2_verdicts.py`, reproducing the r22 section-5 numbers).
* Fresh-pair verdicts (this round, W32P ships):
  `evidence/G5/decidability_v2_r25f.json` over
  `evidence/G5/final_adjudicate_r25f_w32p.json`.

## D2 -- bwd final direction: HOLD the r19-final bytes; the dual-exhaustion proof (value-identical r22 + new-numerics r24) is the paper narrative

STATUS: **ADOPTED-PER-RECOMMENDATION (user non-response; override
welcome).**

Rationale, verbatim from the r24-bwd decision memo
(`evidence/G5/bwd_wpark_decision_memo.md` section 6):

> **NO-GO, now decision-grade.**  The W-park is numerically admissible
> and structurally dead: pool headroom 2,560 B vs a >= 16,384 B minimum
> park; global parking costs 1-3x more traffic than the staging it
> saves; and the unbuildable FREE variant tops out at 1.02x on a stage
> that is 19-24% of the chain -- under every bar this project has ever
> shipped a lever at.  For the three-way choice: the "new-numerics
> (W-park)" branch of the r22 round-23 list is CLOSED at these shapes
> with quantified cause; nothing in this memo disturbs the r19-final
> bytes or the hold recommendation.  If bwd work ever reopens, the only
> directions with unmeasured headroom are regime changes (B4+/T-other/
> H-other shapes) -- at B1T2048H16/B1T8192H16 the remaining idea
> classes are, to the best of the project's current knowledge,
> exhausted.

What this freezes:

* The bwd chain stays the **r19-final bytes** (r20-r22 spot/det gates
  re-verified them; no later round changed a shipped byte).
* The paper narrative for bwd is the **dual-exhaustion proof**:
  1. *value-identical class* -- exhausted at r22 (dqkwg allocator
     audit, K32 split, VTF/DHP/ks2/f1ep-family honest negatives;
     `evidence/G5/bwd_report.md` round 22, commit 5b70903).
  2. *new-numerics class* -- exhausted at r24 (the W-park flagship
     priced NO-GO decision-grade: numerically admissible, perf/pool
     dead; `evidence/G5/bwd_wpark_decision_memo.md` + raw
     `wpark_twin_T{2048,8192}.json`, `wpark_arith_out.txt`).
* bwd perf claims stay honest-miss (0.126x formal; 7.2x five-round
  trajectory) with the completeness argument above as the deliverable.

Evidence pointers:
* `evidence/G5/bwd_report.md` rounds 19-22 (the exhaustion chain).
* `evidence/G5/bwd_wpark_decision_memo.md` (the quantified closure).
* `paper/outline.md` sections 5.5/6 (the honest-miss framing this
  decision keeps in force; post-G7 addendum now carries the longscroll
  mapping).

---

## Override procedure

Either decision flips by a one-line user reply at any round: D1 by
"v2 withdrawn" (claims revert to v1-only wording; no re-measurement),
D2 by "bwd reopened" (new round planning under the regime-change scope
the memo names; the frozen bytes remain the last-good reference).

---
## OVERRIDE (2026-10-10, user in-session)

**D2 OVERRIDDEN by user**: bwd does NOT hold — **LAUNCH THE REGIME-CHANGE EVALUATION** (B/T ladder, larger-shape extension; engineering scope ~= one new G5 direction). D1 not overridden (v1/v2 dual citation stands). The dual-exhaustion proof remains valid *at the original three shapes* and enters the paper as the small-shape chapter; the ladder campaign extends the frontier.
