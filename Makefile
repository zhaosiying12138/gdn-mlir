.DEFAULT_GOAL := status

status:
	@bash scripts/status.sh
env-a:
	@bash scripts/env_gdn.sh a
env-b:
	@bash scripts/env_gdn.sh b
llvm:
	@bash scripts/build_llvm.sh 2>&1 | tee -a logs/build_llvm.log
smoke-sm80:
	@bash scripts/smoke_sm80.sh 2>&1 | tee logs/smoke_sm80.log
harvest-fla:
	@bash scripts/harvest_fla.sh 2>&1 | tee -a logs/harvest_fla.log
harvest-qla:
	@bash scripts/harvest_qla.sh 2>&1 | tee -a logs/harvest_qla.log

.PHONY: status env-a env-b llvm smoke-sm80 harvest-fla harvest-qla

# ===========================================================================
# G7-B reproduction targets -- one command per gate, G0->G6, wiring the
# EXISTING idempotent drivers via scripts/repro.sh (step banners: command,
# CPU/GPU marker, expected wall time; per-gate PASS/FAIL from exit codes +
# post-run JSON verdict checks).  Gate -> report -> artifacts: EVIDENCE_INDEX.md
# ===========================================================================

repro-list: ## list the reproduction gates and their cost/markers
	@bash scripts/repro.sh list

repro-g0: ## G0 infra: env-a/env-b + LLVM build + sm80 GPU smoke + fla/FlashQLA PTX harvest (CPU; ~2-2.5h cold, 1 GPU minute)
	@bash scripts/repro.sh g0

repro-g1: ## G1 semantics: JAX spec 44/44 + MLIR L1 fwd/bwd + dialect build+smoke (CPU; ~15 min)
	@bash scripts/repro.sh g1

repro-g2: ## G2 knob matrix: 20-config gen+compile + PTX coverage analysis (CPU; ~25 min)
	@bash scripts/repro.sh g2

repro-g3: ## G3 scheduling: noargs regen + sm_120 calibration + 20-config GPU timing + Spearman gate (2 GPU steps; ~45 min)
	@bash scripts/repro.sh g3

repro-g4: ## G4 GPU correctness: 20-config verify vs fp64 twins, dual seed (GPU; ~1h)
	@bash scripts/repro.sh g4

repro-g5: ## G5 full-op: fwd/bwd verify + fwd/bwd bench + interleaved adjudication (GPU; ~1-1.5h)
	@bash scripts/repro.sh g5

repro-g6: ## G6 search-cost: Z3 solver timing (CPU) + fla autotune cost (GPU) (~10 min)
	@bash scripts/repro.sh g6

repro-all: ## G0->G6 in order, STOP on first FAIL, summary table at the end
	@bash scripts/repro.sh all

repro-all-keepgoing: ## G0->G6 in order, CONTINUE past failures, full table at the end
	@bash scripts/repro.sh all --keep-going

adjudicate-check: ## dry-run: is the GPU window clean for the final adjudication?
	@bash scripts/adjudicate_when_clean.sh --check

.PHONY: repro-list repro-g0 repro-g1 repro-g2 repro-g3 repro-g4 repro-g5 repro-g6 repro-all repro-all-keepgoing adjudicate-check
