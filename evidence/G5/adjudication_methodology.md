# G5: no-clean-window adjudication methodology (formalized G5-N, 2026-10-08)

User directive this formalizes: the GPU is co-occupied (~half a day,
co-job pid 3411456 resident, plus sibling workstream bursts); timing
work that needs a clean window is deferred, not blocked — comparisons
are either **decidable despite co-load** or **honestly deferred**, and
nothing idles. This document is the contract for which is which.

Harness: `tools/gdn_final_adjudicate.py` (interleaved paired-median
protocol; this G5-N pass adds `--stats`). Worked example with real CI
numbers: `evidence/G5/final_adjudicate_g5n_stats.json` (run registered
in `~/codebase/GPU-COORDINATION.md`, row `gdn-mlir G5-N`, <4 GB,
co-job untouched).

## The interleaved paired-median protocol (as implemented)

Per shape: N interleaved pairs, each pair = (ours fwd timed with CUDA
events, fla fwd timed with torch events), same session, same staged
inputs (shared G5-A recipe, seed 42). Outputs per shape: per-pair
ratios r_i = fla_i / ours_i (>1 = ours faster), both medians, per-side
min/median/max, nvidia-smi snapshots before/after, and the
ratio-of-medians vs median-of-ratios divergence check (`window_clean`).

**Equivalence argument (c), stated precisely.** A co-job burst slows
BOTH contenders when it lands between the two members of a pair, and
slows exactly one when it lands inside a member. On this GPU we
measured (G5-A/G5-D) separate-session comparisons swinging 1.5-2x with
window choice alone; the interleaved ratio cancels the between-pair
drift to first order because consecutive samples see statistically the
same co-load. What interleaving cannot cancel is a burst INSIDE one
member — that shows up as an outlier pair, which is why the protocol
reports medians (outlier-robust) and now the per-pair spread (below),
not means. The ratio remains an ORDER-PRESERVING estimator under
symmetric co-load: both sides lose roughly the same fraction to memory
pressure, so the ratio estimates a lower bound of the clean-window
speedup when both kernels are comparably dram-bound — and this
document's decidability rule only ever claims a comparison when the
effect dwarfs the measured co-load noise, never when it hides inside it.

## (a) N-pair confidence — the `--stats` block

`--stats` adds, per shape, over the per-pair ratios:

* median, IQR (quartiles by the same linear interpolation as the
  median), min, max;
* an exact two-sided **sign test** vs r = 1.0 (wins/losses, ties
  dropped, p = 2·P(Bin(n,½) ≤ min(w, n−w))); exact binomial, no scipy;
  n = 5 gives a floor of p = 0.0625, n = 30 reaches ~1.9e-9 — which is
  why the idle-time re-measure (below) uses 30 reps;
* all of it lands in the JSON (`stats` key), so every archived run
  carries its own confidence.

## (b) The order-of-magnitude decidability rule

A comparison is **decidable in a dirty window** iff

    (median_ratio − 1) > 3 × (IQR / median_ratio)   [and median > 1]

i.e. the effect exceeds 3x the relative per-pair spread. Rationale:
co-load noise enters through the pairs (the IQR measures exactly the
within-window co-load turbulence the protocol cannot cancel); a gap 3x
larger than that turbulence cannot be manufactured by the co-job —
order-of-magnitude gaps are decidable regardless of co-load (the user
directive's rule). Borderline cases are NOT adjudicated: they fall to
the sign test + `window_clean` heuristic for a soft read, and to the
idle-time re-measure for the verdict. The rule is deliberately
conservative: a huge median with proportionally huge IQR (e.g. [1.4,
1.5, 2.5, 3.5] → median 2.0, IQR/med 0.64) stays "not decidable" even
though every pair won.

## Worked example (real numbers, dirty window, 2026-10-08)

Window: co-job pid 3411456 resident (17.9-19.9 GiB, 15-25% util) AND
sibling workstream G5-M running its round-7 GPU bursts in the same
minutes — i.e. the hostile case. `--pairs 5`:

| shape | ours med ms | fla med ms | ratio-of-med | pair median | IQR (rel) | sign p | decidable |
|---|---|---|---|---|---|---|---|
| B1T2048 | 0.2892 | 0.6700 | 2.316x | 1.843x | 0.572 (0.310) | 0.0625 | **no** |
| B1T8192 | 0.9385 | 0.9604 | 1.023x | 1.028x | 0.040 (0.039) | 0.375 | **no** |
| B4T2048 | 1.1346 | 1.0045 | 0.885x | 1.022x | 0.421 (0.412) | 1.000 | **no** |

Reading, exactly as the methodology prescribes: **none of the three is
decidable in this window** — B1T2048's raw medians look like a 2.3x win
but the per-pair spread (0.31 relative IQR; fla's first pair burst to
2.0 ms, ours' first to 0.48 ms) fails the rule; B4T2048 shows the
classic dirty signature (ratio-of-medians 0.885 vs median-of-ratios
1.022, `window_clean` borderline); B1T8192 is internally consistent
(quiet tail, clean flag, parity-to-1.03x) but the effect (0.028) is
inside 3x its own spread (0.117) — a soft "parity in this window",
NOT a verdict, and deliberately not one: the G5 gate verdict (1.25-1.31x
interleaved, `GATE_REPORT_FINAL.md`) was measured in a quieter window
and stands until superseded by the idle-time re-measure. This table is
the canonical demonstration that the protocol refuses to speak when the
window cannot support it.

## (d) Idle-time full re-measurement plan

Two layers, both already wired:

**Poller (running since G7-B): `scripts/adjudicate_when_clean.sh`.**
Polls `nvidia-smi --query-compute-apps` every 120 s; after 3
consecutive samples with no other compute process resident it (1)
registers a coordination row, (2) runs
`python tools/gdn_final_adjudicate.py --stats` ONCE →
`evidence/G5/final_adjudicate_clean.json`, (3) closes the row. Lock in
/tmp (crash-safe); it never signals any process. The upgrade here means
the clean-window run automatically carries the CI block.

**Complete re-measure (what a full idle window covers, for whoever
holds it):** all shapes x {fwd staged mono trio, fwd fused flagship,
bwd chain} x **30 reps**, interleaved vs fla, with `--stats`
(n = 30 pushes the sign-test floor to ~2e-9 and the IQR-based rule to a
real CI):

```
python tools/gdn_final_adjudicate.py --stats --pairs 30 \
    --shapes B1T2048,B1T8192,B4T2048 \
    --out evidence/G5/final_adjudicate_clean30.json     # staged fwd
python tools/gdn_final_adjudicate.py --stats --pairs 30 \
    --gen-root evidence/G5/gen_fwd  --shapes ...        # fused flagship
python tools/gdn_bwd_bench.py                            # bwd vs fla
bash scripts/run_calib.sh                                # calibration refresh
```

plus a calibration refresh (`scripts/run_calib.sh`, idempotent,
overwrites `evidence/G3/calibration.json` and removes the
`shared_gpu: true` lower-bound caveat) — the same pass then re-runs
`tools/g5n_static_analysis.py` so the round-7 roofline distances
re-compute against clean-window peaks. Acceptance for a shape:
decidable=true under the rule (or p < 0.01 with clean flags); anything
else stays deferred. Budget: ~30-40 min GPU, <4 GB, one row in
`~/codebase/GPU-COORDINATION.md`.

## Version note

`--stats` is additive: the JSON gains a `stats` key per shape; all
pre-G5-N fields and their semantics are unchanged, so archived runs
remain comparable. Statistics are dependency-free (stdlib
`statistics` + `math.comb`).

## (b-v2) Trimmed-per-pair decidability -- DOCUMENTED-OPTIONAL, PENDING USER SIGN-OFF (r21 #4 -> r22 §5 -> R23-fwd)

**Status: additive side-rule only. v1 above stays THE rule for every
cross-round comparison; nothing in any archived JSON or record changes.
A record claimed under v2 must cite both v1 and v2 lines.**  Sign-off
was asked at r22 and cannot be granted mid-run, so this section exists
to make the rule reviewable -- applying it is a per-record editorial
choice, not a default.

Motivation (three B4T2048 windows: r20 1.1883x / r21 1.2329x / r22
1.1417x): the sessions agree on ours-side to +/-1% and disagree only
through fla's TAILS (fla max 1.24-2.58 ms vs medians 0.97-1.04 ms; the
co-job's bursts land inside fla's fatter kernels). The v1 IQR is a
tail-sensitive statistic, so it measures the CONTAMINATION, not the
noise floor of the uncontaminated pairs -- blocking every window
including r22's 30/30, p=9.3e-10, IQR-rel-0.060 one (missed by 0.039).

Rule (v2): identical to v1 except the IQR is computed over the INNER
80% of per-pair ratios (symmetric 10% trim; n=30 drops the 3 highest
and 3 lowest pairs). Median, sign test, and win counts are untouched:

    decidable_v2 = (median_ratio - 1) > 3 x trimmed_IQR / median_ratio
                   AND wins >= 25/30 AND p <= 1e-05

A burst inside one member contaminates O(1) pairs per session; the 10%
trim removes exactly that class, and trims nothing when the window is
clean (v1 == v2 then).

Computed on the r22 raw pairs (final_adjudicate_r22f_f1zf.json;
recomputed and verified r23 from the raw arrays -- scratch/r23f/
v2_verdicts.py + v2_verdicts.json):

| shape | median | v1 IQR rel | v2 tIQR rel | effect vs 3x bar | wins / p | verdict |
|---|---|---|---|---|---|---|
| B4T2048 | 1.1450 | 0.060 | **0.044** | 0.145 > 0.131 | 30/30, 9.3e-10 | **DECIDABLE-v2** (v1: no, by 0.039) |
| B1T2048 | 2.1113 | 0.374 | **0.282** | 1.111 > 0.846 | 29/30, 2.9e-08 | **DECIDABLE-v2** (v1: no) |
| B1T8192 | 1.2054 | 0.296 | 0.234 | 0.205 < 0.703 | 27/30, 4.2e-06 | no (both; genuine two-sided window) |

STATED CAVEAT (binding for any v2 claim): v2 measures WITHIN-session
consistency. It cannot rule on window-to-window bias -- B1T2048's
center moved 1.78x -> 2.11x between windows on fla's median alone
(0.35 -> 0.42 ms), so its v2 verdict claims internal coherence, not a
stable 2.1x. B4 is the case v2 was designed for: three windows, ours
stable, only the tails disagree.
