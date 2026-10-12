# G3-B: Z3 knob solver + calibrated analytical cost model

Deliverables: `tools/gdn_solver.py` (Z3 enumeration + ranking + `--validate`)
and `tools/gdn_costmodel.py` (analytical model). This report documents the
model forms, the Spearman gate result against the G3-A measured times, the
legality-space size, the ranked-config example, the calibration table, and
the shared-GPU robustness argument.

**Gate verdict: PASS — Spearman 0.8962 (threshold 0.7), n=18 measured
configs** (of 20 built; `kkt_bf16_mma_p2_t64x64x64_w4_ws` TIMEOUT =
hardware mbarrier deadlock, `kkt_bf16_mma_tma_ws` NOT_TIMED = cubin has no
data globals because G2 deliberately left CUtensorMap wiring to the runtime;
both are excluded by the status filter in the times loader).

Validation inputs: `evidence/G3/config_times.json` (median-of-50 CUDA-event
times of the smem-fixed, zero-arg rebuilds), compile-time features from
`evidence/G3/gen_noargs/*/meta.json` (preferred: these are the cubins
actually timed; the G2 `evidence/G2/gen/INDEX.json` entries are used only as
fallback), rates from `evidence/G3/calibration.json`.

---

## 1. Solver mechanism (PLAN.md §3 port)

TileLang's Ascend `z3_scheduler` used start-time/buffer-version integer
variables + dependence/resource-exclusion constraints + II binary search.
Our v1 port is one level up, as planned: the *decision variables are the
knob vector*, the constraints encode resource capacities and structural
legality, and the objective (analytical latency) ranks the enumerated legal
space:

* **Variables** (Z3, fixed declaration order): `dtype`, `prec`, `tile_M`,
  `tile_N`, `tile_K`, `warps`, `pipeline`, `mma`, `warpspec`, `tma`.
  Domains: tile {32,64,128} (legal subsets), warps {2,4,8}, pipeline
  {0,2,3}.
* **Constraints** (assertions, all in `KnobSpace`):
  - structural legality mirroring `gdn_l2_gen.validate()`: `tile.M | 64`,
    `tile.N | N_glob`, `tile.K | K`, 8-multiples, mma fragment alignment
    (`M % 16 == 0`, bf16: `K % 16 == 0`, tf32: `K % 4 == 0`,
    `N % (8*warps) == 0`), pipeline depth ≤ K-loop trip count;
  - knob coupling: `mma off ⇒ no separate precision knob`;
    `warpspec ⇒ (kkt ∧ bf16 ∧ mma)`; `tma ⇒ warpspec`; warpspec payloads
    use an identity transform script so `pipeline` is pinned to the p2
    convention there (not observable — avoids duplicate code);
  - **smem byte model** (explicit, documented in the solver docstring):
    `pair = (bt·bk + bk·bv)·esz`;
    mma non-ws: `smem = pair · max(pipe,1) · n_gemm` (multibuffer factor =
    pipeline; the o stage allocates one pair per GEMM — note the G2
    `validate()` undercounts this by n_gemm, a found quirk);
    warpspec: `pair + 64·64·esz (tma staging) + 8 (mbarrier)` (single
    buffered);
    nomma: `(bt·K + K·bv + bt·bv)·esz`.
    Constraint `smem ≤ 99KB − 1KB` (mode `strict`, sm_120 opt-in minus
    reserved) or `≤ 220KB` (mode `generator`, mirrors the generator);
  - **regs**: `32·(warps + warpspec) · 255 ≤ 65536` — forbids e.g.
    warps=8 + producer warp (288·255 > 64K).
* **Objective**: minimize the calibrated analytical cost model. The model
  is non-linear (max/blend, divisions), so v1 enumerates every legal
  assignment with SAT + blocking clauses (the TileLang `rlimit`+
  fixed-seed determinism discipline kept) and ranks outside the solver.
* **Determinism**: `z3` solver `random_seed=7`, `rlimit=20e6`, fixed
  variable order, fixed blocking order, final sort on
  `(predicted_time, name)`. Two independent runs produced identical
  368-entry rankings (verified byte-identical name order and times).
* **Loopback** (not in this deliverable's scope but designed for):
  solve → `gdn_l2_gen.py --json` → compile → ptxas re-rank → GPU window
  timing of top-K. Un-compiled configs are ranked with spills=0 (the one
  compile-time feature that needs the loopback); configs whose knob vector
  matches a compiled entry automatically pick up its measured
  regs/spills/ex2 features (`has_index_features` in the output JSON).

### Legality-space size

| stage / dims | mode | legal combos | enum time |
|---|---|---|---|
| kkt K=128 | strict (99KB−1KB) | **368** | 0.29 s |
| kkt K=128 | generator (220KB) | 368 | 0.28 s |
| o K=64 V=64 | strict | 204 | 0.15 s |
| o K=128 V=128 | strict | 465 | 0.33 s |

(Within the {32,64,128} tile domain nothing exceeds the strict 98KB budget
at these dims, so strict == generator here; the smem constraint bites for
larger K/V or if M=16 tiles are admitted.)

---

## 2. Cost model forms and the iteration history

All term definitions are documented in `gdn_costmodel.py` (module
docstring). Every rate is either measured by G3-A calibration or a
documented sm_120-class structural constant; **no per-config fitted
constants**. The single global launch constant is derived from the pure-
launch `ref_f16` anchor (see iteration 4).

Baseline v1 (contract form):
`T = T_launch + overlap(T_mma|T_ffma, T_gmem, T_smem) + T_bar + T_spill + T_ex2`,
`overlap = max` when pipelined (pipeline>0 or warpspec), `sum` when
pipeline=0 (serialized sync copies).

| it | change | overall ρ | per-stage ρ (kkt / o) |
|---|---|---|---|
| 1 | v1 as contracted; nomma = CUDA-core FFMA roofline; launch=4µs fallback | 0.7352 | 0.644 / 0.486 |
| 2 | v2: + `T_lsu` (warp-instruction issue term for the uniform-address cp.async stream + fragment `ld.shared` + epilogue stores) + `T_alloc` (device-malloc term, auto-zero for the smem-fixed timed cubins) | 0.7475 | 0.669 / 0.486 |
| 3 | nomma term replaced: `T = MACs · 10 cyc / clock` — SASS-derived scalar-loop model (see below) | **0.8962** | 0.936 / 0.829 |
| 4 | launch anchor: single global `launch_ns = 6.12µs` derived from the measured `ref_f16` anchor minus its modeled non-launch terms | 0.8962 (rank-invariant; fixes absolute scale) | 0.936 / 0.829 |
| 5 | v3: + occupancy/waves term (`waves = ceil(CTAs/(82·occupancy))`, occupancy from regs/smem/warp-slot limits) | 0.8962 | 0.936 / 0.829 |

Iteration 3 rationale (compile-time evidence, not fitted): the measured
nomma configs run 0.5–2.5 ms — 100–400× above the FFMA roofline. The
`convert-linalg-to-loops` lowering emits an *undistributed scalar triple
loop*: the innermost body in `evidence/G3/gen_noargs/kkt_bf16_nomma_p2_
t64x64x32_w4/sass.txt` (loop `.L_x_17`) is 39 SASS instructions — 10×
`IADD.64` addressing + loop control + 3 scalar `LDS.U16` + 1 FMA — per MAC,
i.e. ~10 cycles/MAC at 4 issue pipes/SM. The constant `NOMMA_CYC_PER_MAC=10`
is read off the SASS, never tuned against the measured times (predicted
kkt-nomma 1.94 ms vs measured 0.99–1.07 ms; o-nomma 1.93 ms vs 2.35–2.46 ms
— the strided-B o-stage variant pays ~2× more per MAC, an accepted
within-cluster imprecision).

Feature-formula verification (`gdn_solver.py --selftest`, analytic vs
INDEX static PTX counts, 18 GDN configs): `mma.sync` counts **exact 18/18**,
`ld.shared` **exact 18/18**, `bar.sync` **exact 18/18**, `cp.async` within
+0.4–1.6% (prologue/param epilogue slack). The peel factor `(1+pipeline)`
for `pipeline_shared_memory_copies` body duplication and the CSE'd unique
fragment loads were both verified this way.

### Remaining known blind spots (documented, direction-correct)

* **pipeline=0 `.ca` serialization**: `create_async_groups` without
  `bypass_l1` emits `cp.async.ca` (L1-allocating) copies with a single
  `commit_group`+`wait_group 0` per K-iteration. Measured kkt p0 = 105 µs
  vs p2 = 21 µs (5×) on identical tiles/warps; the model's sum-overlap
  only predicts 2.4 µs of serialization (predicted 7.4 µs). The ranking
  direction (p0 worst among pipelined mma variants) is correct; the
  magnitude is not modeled — no compile-time-derivable constant for the
  `.ca` penalty was available and fitting one to the measured times was
  ruled out by the honesty contract.
* The `.ca` effect similarly explains `o_bf16_mma_p0` (45 µs measured).
* Absolute mma-path times are systematically under-predicted ~3–5×
  (latency-bound effects: fragment-load dependency chains, barrier
  round-trips beyond the 7.7 ns marginal `bar.sync` cost). Ranking is
  unaffected (a uniform scale on all mma configs).

---

## 3. Calibration table used (evidence/G3/calibration.json)

RTX 5090 Laptop GPU (shared; co-running training job at 24–99% util during
the window), 82 SMs, L2 64 MB, smem opt-in 101376 B/CTA (→ 100352 usable),
65536 regs/SM, observed SM clock 2722 MHz (peak attribute 1597 MHz is a
conservative laptop base; the coload-observed median is used for `T_lsu`).

| rate | value | note |
|---|---|---|
| mma bf16 m16n8k16 peak | 115.5 TFLOP/s | recomputed: file's `flops` field = per-thread instr/s × 4096 (over-counts ×32 — `mma.sync` is warp-level); honest peak = instr/s ÷ 32 × 4096. Flagged `mma_flops_corrected` in the parsed table. |
| tf32 tensor peak (derived) | 57.8 TFLOP/s | TF32_RATIO = 0.5 (documented consumer-class constant) |
| smem read bw | 113 TB/s | as-measured; likely broadcast-inflated (uniform-address bench), only affects the ≤1% `T_smem` term |
| gmem DRAM read | 824.6 GB/s | contended lower bound |
| gmem L2-resident read | 2144.3 GB/s | operands are KB-scale and loop-resident in the timing harness → this is the effective operand rate |
| SFU ex2 | 3.23e12 elem/s | |
| bar.sync (marginal) | 7.707 ns | round-trip/20000, 4-warp block |
| cp.async 16B | 1388.2 GB/s | bonus; not yet a model term |
| launch overhead | 6.12 µs | derived from the ref anchor (iteration 4); fallback constant 4 µs |

Structural constants (sm_120-class, sensitivity-checked): LSU issue 4
warp-inst/clk/SM; spill traffic at the L2 per-SM share; nomma 10 cyc/MAC
(SASS-derived); tensor:f32 consumer ratio 2:1 (only used for the unused-
after-iteration-3 roofline path).

---

## 4. Shared-GPU caveat and ranking robustness

`shared_gpu: true` for both calibration and timing (co-load snapshots in
both JSONs; util 24→99% during the calibration window, ~20–31% during
timing). Absolute numbers are contended lower bounds. Robustness argument
(`sensitivity.py`, 200-trial noise injection + ±2× constant perturbation):

* ±2× on every structural constant (NOMMA_CYC_PER_MAC, TF32_RATIO,
  LSU_IOPS, SPILL_BW_FRAC, tensor/clk): ρ stays in **0.882–0.902**;
* uniform ±5/10/20/30% noise on measured times: ρ median 0.896/0.890/
  0.885/0.885, worst p5 = **0.856** — the gate would still pass at 0.7
  under noise nearly an order of magnitude larger than the observed
  run-to-run spread (ms_all in config_times is tight: e.g. kkt p2 w4
  21.0±0.3 µs).

Conclusion: the PASS verdict is not an artifact of the shared window.

---

## 5. Ranked-config example — kkt, B=4, T=8192, H=16 (8192 chunks)

`python tools/gdn_solver.py --solve --stage kkt --K 128 --B 4 --T 8192
--H 16 --mode strict --form v2` → `evidence/G3/solver_kkt.json`
(368 legal configs, ranked; full list in the JSON):

v2 (per-CTA latency view — matches what was validated):

| # | config | predicted |
|---|---|---|
| 1 | kkt_bf16_mma_p2_t32x32x64_w2_ws | 8.62 µs |
| 2 | kkt_bf16_mma_p2_t32x32x64_w4_ws | 8.65 µs |
| 3 | kkt_bf16_mma_p2_t32x32x64_w2 | 8.67 µs |
| 4 | kkt_bf16_mma_p2_t32x32x32_w2 | 8.79 µs |
| 5 | kkt_bf16_mma_p3_t32x32x32_w2 | 8.79 µs |

v3 (waves/occupancy view — the full-shape mechanism; `solver_kkt_v3.json`):
top-5 flips to large tiles / minimal CTAs: p3_t64x64x32_w2 140 µs,
p0_t64x64x64_w2 153 µs, p0_t64x64x32_w2 156 µs, p2_t64x64x64_w2 185 µs,
f32_mma_p0_t64x64x32_w2 197 µs — at 8192 chunks the CTA count
(`8192·(64/bt)·(64/bv)`) dominates, so 64×64 tiles (1 CTA/chunk) beat
32×32 (4 CTAs/chunk) despite the worse per-CTA copy stream. The measured
gate (single-chunk kernels) cannot distinguish v2/v3; the full-shape
preference is a model prediction to be spot-checked in the G5 timing
window. Caveat: warpspec entries rank high on the model but currently
deadlock on hardware (see §6).

o-stage (K=128,V=128, B=4,T=8192,H=16): 465 legal configs, top pick
`o_bf16_mma_p2_t32x32x64_w2` 13.25 µs (`solver_o.json`).

---

## 6. Knob dominance (what G4/G5 must know)

Mean term shares over the 18 measured configs (v2): overlap 48.8%,
nomma-loop 48.7%, spill 1.6%, everything else ≤1%. Ranked by measured
impact on cost:

1. **mma on/off dwarfs everything**: the nomma (linalg→loops) path is
   50–100× slower (0.5–2.5 ms vs 15–105 µs). Root cause is codegen — an
   undistributed scalar loop — not CUDA-core throughput. Any G4/G5 kernel
   must stay on the mma path; fixing the nomma lowering would need
   vectorization + thread distribution in the transform script.
2. **pipeline depth**: p2/p3 ≫ p0 (21 vs 105 µs measured on identical
   tiles): `.cg`+multibuffered pipelining vs `.ca`+serialized. The `.ca`
   penalty magnitude is the model's largest blind spot (§2).
3. **warps**: fewer warps win on these kernels (w2 14.9 µs < w4 21.2 <
   w8 34.2 measured, same tile): the uniform-address copy stream is issued
   redundantly by every warp, so copy issue scales with warps. (A
   distributed-copy lowering would invert this.)
4. **tile size**: context-dependent — small tiles win per-CTA (fewer
   per-thread copies), large tiles win at full shapes (fewer CTAs/waves);
   the v2↔v3 flip is exactly the occupancy term doing its job.
5. **dtype/prec**: f32/tf32 ≈ 1.6× bf16 at equal shape (33 vs 20 µs).
6. ex2, barriers, spills (in the timed rebuilds) are minor (≤2%) at these
   sizes.

Hardware/workflow facts discovered (feeding G4/G5):

* The G2 as-built cubins (device-malloc "smem") **illegal-address on real
  hardware**; only the `kernel_smemfix.cubin` static-smem rebuilds run.
  G4 must build on the smemfix path (or fix the NVVM lowering).
* warpspec producer/consumer deadlocks on HW (mbarrier TIMEOUT);
  tma=on cubins carry no data globals (host CUtensorMap wiring is a
  runtime contract G2 left open). Both are G4 correctness targets; the
  solver's legality space includes them (they are structurally legal and
  compile) — their model ranking is provisional until they run.
* The G3 rebuild's ptxas results differ sharply from the G2 INDEX (regs
  65–96 & ~0 spills vs 255 & 10⁴–10⁵ spill ops): compile-time features
  must be re-harvested per build (the loader prefers `gen_noargs` metas
  for exactly this reason).
* Calibration `flops` field over-counts ×32 (thread- vs warp-level mma
  instructions) — any consumer of `calibration.json` must recompute
  (already reported to G3-A; our loader corrects it with a flag).
