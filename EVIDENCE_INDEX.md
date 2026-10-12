# EVIDENCE_INDEX — gate → report → headline artifacts

One line per gate: canonical GATE_REPORT, the headline artifacts behind its
verdict, and the reproduction target that regenerates them
(`scripts/repro.sh`, wired to the existing idempotent drivers).
Note: the root-level `GATE_REPORT.md` is a stale copy of the G0 report — the
canonical reports are the `evidence/G*/GATE_REPORT*.md` files below.

| gate | verdict & headline | GATE_REPORT | headline artifacts | repro |
|---|---|---|---|---|
| **G0** infra + reference harvest | LLVM/MLIR build + sm80 GPU smoke + fla 22/22 + FlashQLA 5/5 PTX/SASS resource tables | `evidence/G0/GATE_REPORT.md` | `evidence/G0/sm80_smoke.txt`, `evidence/G0/fla_ptx/SUMMARY.md`, `evidence/G0/qla_ptx/SUMMARY.md` | `make repro-g0` |
| **G1** dialect + semantics | JAX spec 44/44; MLIR L1 fwd 7/7 (~6e-7 vs JAX); bwd 3npz × 6 grads; dialect smoke green | `evidence/G1/GATE_REPORT.md` | `evidence/G1/jax_spec_report.md`, `evidence/G1/l1_harness_report.md`, `evidence/G1/l1_check_output.txt`, `evidence/G1/l1_bwd_check_output.txt`, `evidence/G1/ref_jax_T128_C{32,64,64_h0}.npz` | `make repro-g1` |
| **G2** knob library + PTX coverage | 20/20 configs compile; T1 feature tier covered; both poles expressible (setmaxnreg spill 22.8KB→48B); hatch 0/5 | `evidence/G2/GATE_REPORT.md` | `evidence/G2/gen/INDEX.json`, `evidence/G2/coverage_matrix.md`, `evidence/G2/hatch_report.md`, `evidence/G2/gen_report.md`, `evidence/G2/ref_inventory.{md,json}` | `make repro-g2` |
| **G3** calibration + Z3 solver | Spearman **ρ=0.8962** (gate ≥0.7); solver deterministic (byte-identical reruns); 18/20 timed (shared-GPU) | `evidence/G3/GATE_REPORT.md` | `evidence/G3/validation.json`, `evidence/G3/solver_report.md`, `evidence/G3/calibration.json`, `evidence/G3/config_times.json`, `evidence/G3/solver_{kkt,o,kkt_v3}.json`, `evidence/G3/calib_report.md` | `make repro-g3` |
| **G4** GPU correctness | 20/20 configs PASS dual-seed vs exact fp64 twins (bf16-io ≤5e-3 / f32-io ≤1e-3) | `evidence/G4/GATE_REPORT.md` | `evidence/G4/verify_results.json`, `evidence/G4/verify_results_seed7.json`, `evidence/G4/gen_verify/<config>/` (payload/PTX/cubin/build.log per config) | `make repro-g4` |
| **G5** full-op timing | **fwd 3/3 ≥1.1× vs fla** (1.592 / 1.05–1.19 / 1.253–1.312×, interleaved); bwd honest miss **0.126×** (7.2× trajectory); fwd numerics 4/4 | `evidence/G5/GATE_REPORT_FINAL.md` (+ rounds: `GATE_REPORT{,_R2,_R3,_R4}.md`) | `evidence/G5/final_adjudicate_g5k_run{1,2}.json`, `evidence/G5/fwd_verify_round5.json`, `evidence/G5/fwd_verify_round4_t8192.json`, `evidence/G5/bwd5_bench2048.json`, `evidence/G5/fwd_report.md`, `evidence/G5/bwd_report.md`, `evidence/G5/baseline_report.md`, `evidence/G5/fla_baseline.json`, `evidence/G5/flashqla_fwd.json` | `make repro-g5` (clean-window verdict: `scripts/adjudicate_when_clean.sh`) |
| **G6** evidence synthesis | search-cost **~50×** (0.8 s CPU vs 39.8 s GPU autotune); coverage+hatch ledger **1/≤5**; 12-entry failure longscroll | `evidence/G6/GATE_REPORT.md` | `evidence/G6/search_cost.md`, `evidence/G6/knob_ablation.md`, `evidence/G6/coverage_hatch_summary.md`, `evidence/G6/failure_longscroll.md`, `evidence/G6/fla_autotune_cost.json`, `evidence/G6/scratch/solver_timing_summary.json` | `make repro-g6` |
| **G7** paper + open-source | paper draft (G7-A); repro targets + README + this index (G7-B) | — | `paper/`, `scripts/repro.sh`, `EVIDENCE_INDEX.md` | `make repro-all` |
| **RC1–R4** regime-change campaign (post-G5 bwd addendum, 2026-10) | bwd u≥1 **precision-closed** 0.411–0.488× vs fla (three independent grounds; u=8 ceiling 0.85–0.92× = the error contract's number); small-shape **B1T512H4 1.723×** WIN; cold-start all shapes WIN (AOT vs fla 48–69 s/shape autotune); shipped knobs `GDN_BF16G`/`GDN_ARET`/`GDN_TGEMM_AH` (chain == r3-final bytes, 8 kernels); deep-dive chapter `paper/zh/sections/11-regime-change-campaign.md` | `evidence/G5/CAMPAIGN_CLOSEOUT.md` (close-out; **user confirmation pending**) | `evidence/G5/rc1_scaling_predictions.md`, `evidence/G5/rc1_ladder_report.md`, `evidence/G5/r2_bf16_globals_report.md`, `evidence/G5/r3_report.md`, `evidence/G5/r4_fusion_report.md`, `tools/rc1_price.py`, `evidence/G5/scratch/{r2_bf16_probe,r3_probe,r4_price}.py`, `evidence/G5/scratch/{r2,r3,r4}_logs/` | CPU probes: `python3 scratch/r2_bf16_probe.py gate` etc.; ladders per round report (interleaved paired A/B ×3, fresh process) |

Cross-cutting: recovery/state ledger `RECOVERY.md` · resource rules
`CONTRIBUTING.md` · plan `docs/PLAN.md` · conventions `docs/L0_SEMANTICS.md` ·
GPU coordination protocol (dev box) `~/codebase/GPU-COORDINATION.md` ·
regime-change campaign close-out `evidence/G5/CAMPAIGN_CLOSEOUT.md`
(decision history D2-override → RC1→R4, three-ground closure, final
scorecard) · deep-dive mega-paper (11 chapters incl. campaign §11)
`paper/zh/gdn-mlir-deep-dive.{md,html}`.

Measured sizes (2026-10-09, working tree): `evidence/` 698 MB total —
G5 562 MB (refs 438 + scratch 96 + gen_fwd 28), G4 37, G3 36, G2 34, G0 28,
G1 3.1, G6 0.1 MB. Regeneration paths per tree: see README "Repository size
& evidence policy".
