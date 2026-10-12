# G6: compile-time scheduling vs on-device autotune — the search-cost account

Question the paper must answer: what does it COST to pick kernel
configurations? Our stack answers offline, on CPU, with a deterministic
Z3 enumeration + calibrated analytical cost model. The reference stack
(fla/Triton) answers on-device, by compiling and benchmarking every
candidate config at first call, per shape key, per process.

## (a) Our side: fully offline CPU

Re-measured for this gate (this window, host CPU of the RTX 5090 laptop,
`OMP_NUM_THREADS=4`, conda `gdn`, z3 5.1.0). Runner:
`tools/g6/g6_solver_timing.sh`; raw outputs
`evidence/G6/scratch/solver_{kkt,o}_{v2,v3}.json` + summary
`evidence/G6/scratch/solver_timing_summary.json`.

| space | legal configs | Z3 enumeration | full solve wall (enum + cost-model rank of every config) | top pick (reproduces G3) |
|---|---|---|---|---|
| kkt K=128 (B4 T8192 H16), strict 98 KB, v2 | **368** | 0.307 s | 0.378 s | `kkt_bf16_mma_p2_t32x32x64_w2_ws` 8.62 µs |
| kkt K=128, v3 (waves/occupancy form) | **368** | 0.296 s | 0.365 s | `kkt_bf16_mma_p3_t64x64x32_w2` 140 µs (the documented v2→v3 flip) |
| o K=128 V=128, strict, v2 | **465** | 0.379 s | 0.444 s | `o_bf16_mma_p2_t32x32x64_w2_K128V128` 13.25 µs |
| o K=128 V=128, v3 | **465** | 0.374 s | 0.442 s | `o_bf16_mma_p0_t64x128x32_w2_K128V128` 350 µs |

* Determinism contract re-confirmed by the rerun: fixed declaration order,
  seed 7, rlimit 20e6, sort on (predicted, name) — the 368/465 spaces and
  top picks reproduce the G3 outputs byte-semantically
  (`evidence/G3/solver_kkt.json`, `evidence/G3/solver_o.json`,
  `evidence/G3/solver_report.md` §1).
* Ranking quality: Spearman **0.8962** vs 18 measured configs (kkt 0.936 /
  o 0.829), robust to ±2× on every structural constant (ρ∈[0.882,0.902])
  and ±30% time noise (ρ≥0.856) (`evidence/G3/validation.json`).
* Caveat, recorded: the o K=64 V=64 subspace (204 configs in the G3 table)
  crashes in this environment with a Z3 partial-model artifact under the
  rlimit cap (`m[zv]→None`, `tools/gdn_solver.py:187`); the two canonical
  spaces above reproduce exactly. Flaked run documented here, not hidden.
* One-time sunk costs, NOT part of per-shape search: the G3 calibration
  microbenches (GPU, one-off; `evidence/G3/calib_report.md`) and the 20-config
  G2 compile matrix (CPU ptxas/mlir-opt; `evidence/G2/gen_report.md`).
  Per NEW shape, the scheduler adds ~0.8 s of CPU (kkt+o solve) and zero
  GPU seconds before the first candidate runs.
* The cost model itself has no per-config fitted constants (rates from the
  calibration table + SASS-derived constants;
  `evidence/G3/solver_report.md` §2–3), which is what keeps the ranking
  honest (the known blind spot — the `.ca` p0 serialization magnitude —
  is documented, not fitted).

## (b) fla side: on-device autotune, measured once (the G6 GPU measurement)

Single opportunistic GPU window (2026-10-09, co-job pid 3411456 resident,
~18.2 GB, 22% util at start — smi snapshots inside the json; our process
<4 GB, fresh own `TRITON_CACHE_DIR` so the shared Triton cache and the
co-job were untouched; nothing to restore afterwards). Harness:
`tools/g6/g6_fla_autotune_cost.py`; raw:
`evidence/G6/fla_autotune_cost.json`.

Method notes (all verified before measuring):
* `FLA_CACHE_MODE` is unset in this environment → `DISABLED` → fla's
  disk config cache is **not consulted**; every process pays full Triton
  autotune on first call (`fla/ops/utils/cache.py:51` FlaCacheMode). The
  fla config dir for this GPU (`fla/configs/NVIDIA_GeForce_RTX_5090_Laptop_GPU`)
  does not exist — there is no warm disk cache to move aside; the operative
  caches are the per-process Autotuner benchmark cache and the Triton
  compile cache (the latter bypassed per-process via `TRITON_CACHE_DIR`).
* Inputs: the exact G5 shared recipe (`tools/gdn_bench_common.py`, fla
  test_chunk, seed 42, bf16 io / fp32 gates), B1T2048H16 K=V=128,
  `FLA_FLASH_QLA=0` — i.e. the pure-Triton backend all our comparisons
  used (`tools/gdn_bench_flaptx.py`).
* `Autotuner.run` was wrapped to count, per kernel, the configs in the
  (pruned) candidate list at call time and the tuned keys.

Measured (wall, sync'd, single process):

| phase | cold (first call) | warm (median of 10) | cold−warm | configs benchmarked |
|---|---|---|---|---|
| **fwd** first `chunk_gated_delta_rule` | **16.55 s** | 0.491 ms (min 0.418) | **16.55 s** | **25** across 5 kernels: chunk_local_cumsum 4, fwd_kkt_solve 6, recompute_w_u 9, fwd_h 4, fwd_o 2 |
| **bwd** first `torch.autograd.grad` | **23.26 s** | 1.349 ms (min 0.981) | **23.26 s** | **23 new** across 4 kernels: bwd_dv_local 9, bwd_dhu 4, bwd_dqkwg 9, prepare_wy_repr_bwd 1 |
| **fwd+bwd total** | **39.81 s** | — | **39.81 s** | **48** (67 Triton cache kernel entries after both) |

Sanity anchors: the warm medians land exactly on the G5 baseline bands —
fla fwd median 0.316–0.517 ms across G5 windows
(`evidence/G5/fla_baseline.json`, `evidence/G5/fwd_report.md` r3 §2), and
fla bwd measured 1.34 ms in G5-C (`evidence/G5/bwd_report.md` timing
table) vs 1.349 ms here. The co-job did NOT make the measurement unusable
(the cold cost is compile+benchmark dominated, ~10³× the contention noise),
so the fallback protocol (count-only) was not needed.

Interpretation: Triton compiles each candidate config (ptxas) and
benchmarks it (do_bench warmup+reps) on the live GPU; with the disk cache
absent/disabled this is re-paid **per process and per shape key** — every
new shape re-runs a subset of the benchmarking (fuzzy keys aside, fla's
own cache layer exists precisely to avoid this and is off by default).

## Conclusion table

| | ours (compile-time scheduling) | fla/Triton (on-device autotune) |
|---|---|---|
| search cost, one shape (fwd) | ~0.4 s CPU (kkt solve; stage space 368) | 16.55 s GPU+CPU (25 configs, 5 kernels) |
| search cost, fwd+bwd one shape | ~0.8 s CPU (kkt+o solves) | 39.81 s (48 configs) |
| **ratio (fwd+bwd)** | **1 : ~50** | — |
| where it runs | fully offline CPU; GPU untouched until the chosen config runs | on the deployment GPU, first call, blocking |
| determinism | fixed seed/rlimit, byte-identical reruns (verified again this gate) | benchmark-order/timing dependent, contention-sensitive |
| transferability | model + calibration carry to new shapes (loopback design: solver → compile → ptxas re-rank) | re-paid per process/shape key (unless the optional FLA cache layer is populated) |
| selection quality | Spearman 0.8962 on the stage spaces; full-op shipping configs additionally confirmed by paired GPU A/B (G5) | exact for its own kernel (measured) — the gold standard per kernel, at the measured cost |

Honest scope note: the 0.8 s is the cost of ranking OUR stage-kernel knob
space; the full-op shipping decisions in G5 additionally used measured
paired A/Bs on GPU (the per-lever table in `knob_ablation.md` §2) — that
is engineering attribution, not search. The comparable quantity is: given
the same candidate space, enumerate+rank offline (0.4–0.8 s, deterministic,
no GPU) vs compile+benchmark every candidate on-device (16.6–23.3 s per
direction, re-paid per process).
