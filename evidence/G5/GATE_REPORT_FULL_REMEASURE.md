# G5 FULL RE-MEASURE (clean window) — 2026-10-10 00:07-00:5x

The formal idle-window measurement event per
`evidence/G5/adjudication_methodology.md` (d): GPU verified idle at start
(0 compute apps, 0% util; coordination row `gdn-mlir full-remeasure`,
`~/codebase/GPU-COORDINATION.md`), no other process touched; 0 compute
apps re-checked before each GPU step and once mid-run (recorded in the
JSONs' smi snapshots — `window_pre`/`window_post` show only our own
contexts, clocks 1590-1597 MHz sustained / 180-712 MHz idle dips between
steps). Measurement only — no kernel/generator optimization edits. HEAD
throughout: a7a5603 (G5-T round 13).

Artifacts (this run, all same clean window):

| file | what |
|---|---|
| `evidence/G3/calibration_shared.json` | the OLD co-load-lowered calibration (copy, unchanged) |
| `evidence/G3/calibration.json` | clean-window refresh (`shared_gpu: false` — now derived from observed compute-apps, see harness note below) |
| `evidence/G5/fwd_bench_clean.json` | ours fwd staged trio + fused, 30 reps, CUDA events |
| `evidence/G5/fla_baseline_clean.json` | fla chunk_gated_delta_rule fwd + fwd+bwd, triton+default, 50 reps |
| `evidence/G5/bwd_bench_clean.json` | ours bwd 9-kernel chain (fresh HEAD rebuild), 30 reps, + in-session fla bwd |
| `evidence/G5/final_adjudicate_clean2.json` | interleaved adjudication, 30 pairs/shape, `--stats` |
| `evidence/G5/scratch/g5m_bwd_*` | fresh bwd chain builds (B1T2048, B4T2048 flagship; B1T8192_v2 fallback) |
| `evidence/G5/scratch/g5n/static_analysis_g5m_clean_calib.json` | r7 roofline distances recomputed vs clean peaks |

## 1. Master table (clean window, interleaved verdicts + component tables)

### 1a. fwd — INTERLEAVED adjudication (the verdict-grade comparison)

`tools/gdn_final_adjudicate.py --stats --pairs 30`, same cubins + knobs
request as the archived `final_adjudicate_clean.json` (w8/bv64/f2s1
request; see knob provenance note). fla = pure-Triton dispatch
(FLA_FLASH_QLA=0, probe recorded in the JSON).

| shape | ours fwd med | fla fwd med | speedup (ratio-of-med) | per-pair med | IQR (rel) | sign p | vs archived 5-pair | verdict |
|---|---|---|---|---|---|---|---|---|
| B1T2048 | 0.2894 ms | 0.4236 ms | **1.464x** | 1.410x | 0.323 (0.229) | ~1.9e-9 | 1.289x | **WIN (formal: p<0.01, clean)** |
| B1T8192 | 0.9775 ms | 0.9984 ms | **1.021x** | 1.034x | 0.080 (0.077) | 0.0987 | 1.212x | **PARITY — not significant** |
| B4T2048 | 1.1941 ms | 1.3617 ms | **1.140x** | 1.130x | 0.099 (0.087) | ~1.9e-9 | 0.947x | **WIN (formal: p<0.01, clean; all 30 pairs > 1, min 1.049)** |

* `window_clean = true` on all three shapes (ratio-of-medians vs
  median-of-ratios agree to <= 0.054); clocks pinned at 1590-1597 MHz
  during measurement; nvidia-smi shows no foreign process at any
  snapshot.
* Decidability rule ((median-1) > 3*IQR/median) is NOT met on any shape
  — including B1T2048's 1.464x with p~1e-9. In THIS window the IQR
  measures intrinsic kernel jitter (fla's own min/med spread is 24% at
  B1T2048; one pair hit 3.16x), not co-load. The rule stays conservative
  as designed; acceptance therefore rests on the documented alternative:
  p < 0.01 with clean flags. B1T8192 meets neither -> honestly deferred
  (parity band 0.94-1.23 per pair).
* **Formal score after re-measure: 2/3 shapes >= 1.1x — same count as
  the archived verdict but a DIFFERENT SET: B4T2048 flips 0.947x ->>
  1.140x (all 30 pairs won), B1T8192 drops 1.212x -> 1.021x (p=0.099).**

### 1b. ours fwd components (`fwd_bench_clean.json`, 30 reps, CUDA events)

| shape | staged trio med (min) | f1 | f2 | f3 | fused med (min) | staged/fla-triton* | fused/fla-triton* |
|---|---|---|---|---|---|---|---|
| B1T2048 | 0.3087 (0.2970) | 0.1250 | 0.1233 | 0.0776 | 1.7610 (1.7244) | 1.274x | 0.223x |
| B1T8192 | 0.9992 (0.9735) | 0.3426 | 0.4462 | 0.3301 | 7.5196 (7.3985) | 0.991x | 0.132x |
| B4T2048 | 1.0483 (1.0084) | 0.4621 | 0.3761 | 0.3127 | 3.7893 (3.7420) | 0.881x | 0.244x |

*non-interleaved (separate bench process vs flaptx process) — shown for
stage decomposition only, NOT verdict-grade (see anomaly A1). The staged
trio is the shipped flagship; the persistent fused kernel remains the
older round-2 design and loses everywhere (known; unchanged since G5-D).

### 1c. fla baselines (`fla_baseline_clean.json`, same window, 50 reps)

| shape | triton fwd med (min) | triton fwdbwd med | triton bwd (derived) | default fwd med | default fwdbwd med |
|---|---|---|---|---|---|
| B1T2048 | 0.3932 (0.3160) | 1.4938 | 1.1006 | 0.4134 | 1.5233 |
| B1T8192 | 0.9906 (0.9044) | 3.8285 | 2.8379 | 0.9605 | 3.7622 |
| B4T2048 | 0.9240 (0.8405) | 3.5251 | 2.6010 | 0.9504 | 3.5701 |

### 1d. bwd (`bwd_bench_clean.json`, 30 reps, fresh HEAD chains)

| shape | chain | ours bwd med (min) | fla bwd measured* | ratio* | fla bwd derived** | ratio** |
|---|---|---|---|---|---|---|
| B1T2048 | flagship r13 (rebuild) | 2.6889 (2.6754) | 1.2994 | 0.483x | 1.1006 | 0.409x |
| B4T2048 | flagship r13 (rebuild) | 9.7585 (9.7390) | 3.2118 | 0.329x | 2.6010 | 0.267x |
| B1T8192 | **v2-fwdscan fallback** (see A4) | 11.4216 (11.4007) | 3.4908 | 0.306x | 2.8379 | 0.248x |

*`--measure-fla` in-session (seed-0 torch recipe; times fwd + autograd,
i.e. the tool's documented "fla bwd" protocol). **fwdbwd-fwd from the
clean triton baselines — the r13-report-comparable derivation (r13
quoted 0.383x @B1T2048 against the shared-window fla numbers). bwd
remains an honest miss vs fla on every shape (~0.25-0.48x).

Per-stage medians @B1T2048 (reproduces round 13 exactly):
wy 0.2246, fwdscan 0.3754, caryA 0.0523, caryM 0.0796, bpre 0.0511,
dhs 0.3121, dkb 0.2409, dqkwg 0.9370, wyb 0.3386 — e2e 2.6889/2.6754 vs
r13's 2.688/2.674. B4T2048 stages: wy 0.7936, fwdscan 1.4541, caryA
0.3139, caryM 0.3849, bpre 0.3066, dhs 1.1803, dkb 0.8598, dqkwg
3.4335, wyb 1.1305.

## 2. Calibration refresh: old (shared) vs new (clean)

`scripts/run_calib.sh` overwritten in place; old values preserved as
`evidence/G3/calibration_shared.json`.

| metric | shared (old) | clean (new) | delta |
|---|---|---|---|
| cp.async 16B | 1388.2 GB/s | 1540.2 GB/s | **+11.0%** |
| DRAM read | 824.6 GB/s | 851.4 GB/s | +3.2% |
| L2-resident read | 2144.3 GB/s | 2162.0 GB/s | +0.8% |
| smem read | 113.05 TB/s | 114.07 TB/s | +0.9% |
| mma bf16 peak | 3697.1 TFLOP/s (115.54 norm) | 3713.9 TFLOP/s (116.06 norm) | +0.5% |

The co-load was hurting the async-copy / DRAM paths an order of
magnitude more than the compute peak; the G3 "lower-bound" caveat is
removed (`shared_gpu: false`, note updated).

Harness note (minimal fix, documented): `scripts/run_calib.sh` hardcoded
`"shared_gpu": True` + a dirty-window note; in a clean window that is
false metadata and the methodology explicitly expects the refresh to
drop the caveat. The script now derives `shared_gpu` and the note from
the observed compute-apps snapshots. No measurement logic changed.
Committed separately with this event.

Companion pass: `tools/g5n_static_analysis.py` re-run against the clean
peaks (same gen/bench inputs as the archived r7 analysis; output
`evidence/G5/scratch/g5n/static_analysis_g5m_clean_calib.json`): DRAM
floor 824.6 -> 851.4 GB/s tightens every kernel's floor by ~3.2%
(e.g. wy floor 102.0 -> 98.8 us, dqkwg 255.4 -> 247.4 us); the chain
stays dram-bound everywhere, dkb remains closest to roofline (1.94x),
then bpre 4.65x / dhs 4.90x — the round-14 ranking is unchanged.

## 3. Anomalies & honest disclosures

* **A1 — separate-process absolutes swing even in a clean window.**
  Same cubins, same session, minutes apart: ours B4T2048 staged = 1.0483
  (fwd bench) vs 1.1941 (adjudicator, +14%); fla B4T2048 fwd = 0.9240
  (flaptx) vs 1.3617 (adjudicator, +47%). Cause: idle clock cycling
  between processes (smi snapshots catch 180-712 MHz between steps vs
  1590+ sustained inside). This is the G5-A/G5-D lesson reproduced with
  zero co-load — only the interleaved ratio is verdict-grade, which is
  why 1a (not 1b/1c cross-products) carries the formal numbers.
* **A2 — B1T8192 fwd win not reproduced.** Archived 1.212x (5 pairs) ->
  1.021x (30 pairs, p=0.0987, per-pair range 0.944-1.235): ours 3.5%
  slower AND fla 13% faster than the archived window. With n=30, stats
  and pinned clocks this run supersedes: **B1T8192 is parity**, formally
  deferred (needs the r14 fwdscan/dqkwg levers or a T8192-specific F2
  config to clear 1.1x).
* **A3 — B4T2048 flips to a formal win** (0.947x archived -> 1.140x,
  all 30 pairs > 1, min ratio 1.049). The archived 5-pair loss was the
  noisy read; the earlier dirty-window optimism correction over-corrected
  on this shape.
* **A4 — flagship bwd chain cannot build at T8192**: `gdn_bwd_gen.py`
  asserts `fwdscan: smem 123136B > 101376B` (ptxas sm_120a static limit).
  Root cause: the v3 fwdscan prologue cumsum buffer `gsc [NC,C] f32`
  scales with NC=T/64 (32 KB at NC=128 vs 8 KB at NC=32). This is the
  already-sketched r14 per-half N-split scope. The auxiliary
  GDN_SCAN_PP=0 chain (round-6 v2 fwdscan, 49408 B, 189 regs) builds and
  is timed above, clearly marked NON-flagship; its fwdscan costs 3.115
  ms vs v3's 1.454 ms at the same per-BH work (B4T2048).
* **A5 — B4T2048 bwd numerics not re-verified in this pass**: the r13
  6/6 gates cover T512/T2048; B4 shares the per-BH T2048 structure
  (identical generator knobs, grid x4). This is a timing-only event; a
  B4T2048 verify is queued as normal round work.
* **A6 — fla clean-window fwd medians sit ABOVE the shared-window
  archive** (e.g. B1T2048 triton 0.3932 vs 0.3158) while mins match
  (0.3160 vs 0.3158): the co-job kept clocks boosted; in a clean window
  the median carries idle-dip excursions. Another reason ratios, not
  absolutes, are the currency.
* **Knob provenance** (from `gen_fwd/*/meta.json`): B1T2048_mono and
  B4T2048_mono serve w8/bv64 builds; B1T8192_mono serves the shipped
  w4/bv32+F1UW build; all `_fused` dirs are w8/bv64. The w8/bv64 request
  used here matches the archived clean-adjudication request exactly, so
  the cubins measured are identical to the ones behind 1.289/1.212/0.947
  (the archived run's T8192 "w8" label also served the w4/bv32 cubin —
  the documented build-cache short-circuit).

## 4. Bottom line

Clean-window formal event complete. fwd: **B1T2048 1.464x and B4T2048
1.140x formally >= 1.1x (p ~ 1.9e-9, clean windows, 30 interleaved
pairs); B1T8192 honestly parity (1.021x, p=0.0987)** — score 2/3 with
the winning set changed vs the archive. bwd: honest miss on all shapes
(0.27-0.48x vs fla depending on derivation), reproducing r13's 2.689 ms
@B1T2048 exactly; T8192 flagship bwd blocked by fwdscan static smem
(r14 scope). Calibration de-caveated: DRAM +3.2%, cp.async +11%, compute
peaks within 0.5% — the G3/G5n roofline conclusions are unchanged.
