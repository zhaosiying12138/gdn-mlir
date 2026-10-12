# G6 Gate report (paper-grade evidence chain) — 达成 (2026-10-09)

Workstream G6: synthesis of the landed G0–G5 evidence tree into the four
paper artifacts, plus one supplementary GPU measurement (fla's on-device
autotune cost). All numbers cite their source files; where sources
disagree the corrected value is used and flagged (the G3 `flops` ×32
thread-vs-warp over-count — corrected to 115.5 TFLOP/s in
`evidence/G3/solver_report.md` §3 and used nowhere else).

## Artifact inventory (evidence/G6/)

| artifact | what it is | key sources |
|---|---|---|
| `knob_ablation.md` | THE ablation table: §1 G2 stage-kernel compile-time knob sensitivity (mma/pipeline/tile/warps/dtype/warpspec/tma), §2 G5 full-op lever A/Bs (pipeline, staging, LDSM padding, retunes, segment-split, barriers, F1UW, tf32 sites, split-bf16, carry scan) — every row: delta + where + verdict | G2 gen_report, G3 config_times/solver_report, G5 fwd/bwd reports |
| `search_cost.md` | compile-time scheduling vs on-device autotune: our Z3 enumeration re-timed this gate (368 kkt / 465 o configs, 0.30/0.38 s enum, ~0.4 s full solve, CPU-only, deterministic) vs fla's autotune MEASURED once on GPU (fwd cold 16.55 s / 25 configs; bwd cold 23.26 s / 23 configs; warm 0.49/1.35 ms cross-checking the G5 baselines) | `scratch/solver_timing_summary.json`, `fla_autotune_cost.json` |
| `coverage_hatch_summary.md` | final T1 family coverage (8 YES/19 PARTIAL/6 NO of 33; feature tier 8/6/2 of 15; mma.sync flipped PARTIAL→YES by hatch #1), the T2 both-poles statement (fla multistage pole + FlashQLA warp-spec pole, spills 22.8 KB→48 B), escape-hatch ledger 1/≤5, remaining-gaps table with "not needed for GDN" assessments | G2 coverage_matrix/hatch_report, G5 bwd round-3 addendum |
| `failure_longscroll.md` | the five-round honest negatives, 12 numbered entries + 6 one-liners: staging redundancy, broken pipeliner clause, ping-pong zero-gain + wait_group-0, segment-split Amdahl refutation, barrier/L1-carveout, tf32-dhu double negative, tf32-dvp_bh chaos, tree-scan economics, the OPEN async zero-delivery mystery, B4T2048 window variance, the bf16 misdiagnosis + grading-twin bug, the H2D-masked bf16-B̂ bug | G5 fwd/bwd reports, G2 gen_report |

Supplementary measurement tooling: `tools/g6/g6_fla_autotune_cost.py` +
`tools/g6/g6_solver_timing.sh` (rerun-able); GPU window registered and
closed in `~/codebase/GPU-COORDINATION.md` (co-job pid 3411456 untouched;
own TRITON_CACHE_DIR, nothing restored because nothing shared was moved).

## Headline numbers (what a paper abstract would quote)

| claim | number | source |
|---|---|---|
| fwd vs fla-triton, interleaved adjudication | **B1T2048 1.592×; B4T2048 1.05–1.19×; B1T8192 1.253/1.312×** (all three cross 1.1×) | `evidence/G5/GATE_REPORT_FINAL.md`, `final_adjudicate_g5k_run{1,2}.json` |
| fwd numerics | 4/4 staged+fused, o ≤6.85e-4 / ht ≤7.93e-4 vs fp64 (T8192 gate: bit-identical across segment splits) | `fwd_verify_round5.json`, `fwd_verify_round4_t8192.json` |
| bwd honest trajectory | 58.88→**8.15 ms** @B1T2048H16 over five rounds (**7.2× self-acceleration**), 6/6 grads in tolerance both shapes, **0.126× of fla** — reported as the honest miss | `bwd5_bench2048.json`, `evidence/G5/GATE_REPORT_FINAL.md` |
| offline scheduler ranking quality | Spearman **0.8962** (kkt 0.936 / o 0.829, n=18; robust ±2× constants and ±30% noise) | `evidence/G3/validation.json` |
| search cost: ours (offline CPU) vs fla (on-device autotune) | **~0.8 s vs 39.8 s per shape fwd+bwd (≈50×)**; fla fwd alone 16.55 s / 25 configs, bwd 23.26 s / 23 configs; ours deterministic, zero GPU | `evidence/G6/scratch/solver_timing_summary.json`, `evidence/G6/fla_autotune_cost.json` |
| coverage | T1 feature-tier full modulo variant spellings (mma.sync PARTIAL→YES via hatch); T2 both design poles expressible + hardware-verified; escape-hatch budget **1/≤5** (`gdn.mma.tf32k8`, measured 1.28× e2e bwd) | `coverage_hatch_summary.md` §1–3 |
| the one true hatch | `mma.sync...m16n8k8...tf32` unreachable in-tree (NVVM tablegen limit); early-clobber `llvm.inline_asm` mandatory (ptxas D-onto-A operand corruption otherwise) | `evidence/G2/hatch_report.md` ADDENDUM, `dialect/include/Gdn/GdnHatch.td` |

## Process notes

* The Z3 rerun reproduced G3 byte-semantically (368/465 spaces, same top
  picks); one recorded environment flake: the o K=64 V=64 subspace crashes
  with a Z3 partial-model artifact under the rlimit cap
  (`tools/gdn_solver.py:187`) — documented in `search_cost.md`, not
  silently dropped.
* fla autotune measurement succeeded under co-load (cold cost ~10³× the
  contention noise); the fallback protocol (count-from-source only) was
  not needed. Warm medians land on the G5 baseline bands (0.49 vs
  0.316–0.517 ms fwd; 1.349 vs 1.34 ms bwd) — the measurement is
  self-anchored.
* No new numbers were invented: every cell in the four artifacts traces to
  a landed evidence file or to this gate's two measurement scripts.

## Verdict

**G6 达成**: the evidence chain is paper-grade and consolidated —
ablation matrix, search-cost account (with the one supplementary GPU
measurement), coverage+hatch ledger, and the failure longscroll are in
`evidence/G6/`, committed together with the tooling.
