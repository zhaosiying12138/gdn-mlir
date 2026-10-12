# G6: the knob→effect ablation matrix (project-wide)

The single ablation table for the paper. Every row carries the measured
delta, where the knob applied (stage kernels vs full-op), and the verdict.
Two evidence classes:

* **§1 G2 stage-kernel sensitivity** — the compile-time knob study: 20
  generated single-stage configs, effects measured in PTX/SASS/resource
  counters AND (for 18 of them) in CUDA-event GPU time by G3.
  Sources: `evidence/G2/gen_report.md` (table + §gate (c)),
  `evidence/G2/gen/INDEX.json`, `evidence/G3/config_times.json`,
  `evidence/G3/solver_report.md` (§2 blind spots, §6 dominance),
  `evidence/G3/calib_report.md`.
* **§2 G5 full-op lever A/Bs** — the shipped forward/backward kernels;
  every lever measured by paired interleaved A/B or same-window medians
  (co-job resident throughout; flagged in the source jsons).
  Sources: `evidence/G5/fwd_report.md` (§round-2/3/4/5),
  `evidence/G5/bwd_report.md` (§round-2/3/4/5), per-round bench jsons
  named inline.

Verdicts: **win** (shipped, positive measured), **negative** (measured
regression, reverted), **neutral** (no effect beyond noise), **structural**
(effect visible at compile level only / correctness-gated).

GPU context for all §2 numbers: shared RTX 5090 Laptop (sm_120a, 82 SM,
static smem cap 101376 B/CTA — `evidence/G5/fwd_report.md` lesson 3,
`evidence/G5/bwd_report.md` constraint 4).

---

## 1. G2 stage-kernel sensitivity (compile-time knob study)

Knob deltas measured on the generated stage kernels (kkt = the K·Kᵀ
factor-solve stage; o = the output stage). The three counters are PTX
statics (`mma.sync` / `cp.async*` / `ld.shared`) and ptxas `-v` resources;
timing rows are median-of-50 CUDA events on the smem-fixed rebuilds
(`evidence/G3/config_times.json`).

| knob | delta measured | where | verdict |
|---|---|---|---|
| **mma on→off** (same tile p2 64x64x32 w4) | HMMA 48→0; PTX 276→88 KB (3.1×); LDS 246→7; GPU: mma path 15–105 µs vs nomma 0.5–2.5 ms (**50–100×**) | kkt+o stage kernels; root cause = undistributed scalar linalg loop (39 SASS instr/MAC, ~10 cyc/MAC) | **win, decisive** — every production kernel must be mma-path (`evidence/G2/gen_report.md` gate (c); `evidence/G3/solver_report.md` §6.1) |
| **pipeline 0→2→3** (kkt mma t64x64x32 w4) | mma 16/48/64; LDGSTS 512/1536/2048; LDS 83/246/326; spill 9.7/12.6/18.1 KB; PTX 126/276/351 KB; copy variant ca→cg; multibuffer 1→2→3 | kkt stage kernel | **win (2≥3 per-byte)**; GPU: p0 105 µs vs p2 21 µs (**5×**) — the `.ca` serialization is the cost model's largest recorded blind spot (`evidence/G2/gen_report.md` gate (c); `evidence/G3/solver_report.md` §2) |
| **tile.M 64→32** (p2 64x64x32 w4) | per-thread mma 48→24; LDS 246→150 | kkt | **structural** — small tiles win per-CTA (fewer per-thread copies), large tiles win full-shape (fewer CTAs/waves): the v2→v3 ranking flip is exactly the occupancy term (`evidence/G3/solver_report.md` §5–6.4) |
| **tile.K 32→16** (p2) | in-flight cp.async 1542→774; mma 48→24 | kkt | **structural, trade** — halves in-flight copy stream with half the K-reuse |
| **tile.N 64→32** (nomma) | cp.async 514→386 | kkt | **structural** |
| **warps 2→4→8** (p2 64x64x32) | per-warp mma atoms 96/48/24; PTX 303/276/263 KB; GPU: w2 14.9 < w4 21.2 < w8 34.2 µs (same tile) | kkt stage kernel | **negative-at-stage (fewer warps win)**: the uniform-address copy stream is issued redundantly by every warp — a distributed-copy lowering would invert this (`evidence/G2/gen_report.md` gate (c); `evidence/G3/solver_report.md` §6.3) |
| **dtype bf16→f32** | atom m16n8k16.bf16 → m16n8k4.tf32 (4× atom count; o_f32: 384 atoms, 118.8 KB spills); GPU f32/tf32 ≈ 1.6× bf16 at equal shape (33 vs 20 µs) | o stage kernel | **structural** — f32 accum on bf16 inputs is the workhorse; tf32 costs 4× atoms + spill (`evidence/G2/gen_report.md`; `evidence/G3/solver_report.md` §6.5) |
| **warpspec off→producer_consumer** (kkt p2 t64x64x64 w4) | classic-cg producer + mbarrier + elect; spill 22.8 KB | kkt stage kernel | **structural, hardware-gated**: deadlocked on HW at G3 (mbarrier TIMEOUT); protocol fixed in G4 (count=1, elect-arrive) — never a stage-level win (`evidence/G2/hatch_report.md`; `evidence/G3/solver_report.md` HW facts) |
| **tma off→on** (warpspec) | cp.async 1026 classic → **5 TMA ops**; PTX 214.6→41.2 KB; spills 22844→48 B (the setmaxnreg effect) | kkt stage kernel | **win at compile level** (FlashQLA-pole expressivity + spill); runtime wiring contract landed in G4 (tensormap descriptors, host-side) (`evidence/G2/hatch_report.md` §3; `evidence/G4/GATE_REPORT.md`) |
| **warpspec+tma resource envelope** | setmaxnreg dec96/inc224 holds regs ≤128-class: spill 22.8 KB→48 B = FlashQLA-level vs fla's 255-regs+spill | kkt stage kernel | **win (T2 pole expressible)** (`evidence/G2/hatch_report.md` §3; `evidence/G0/GATE_REPORT.md` qla table) |

Cross-check of the whole matrix: the G3 cost model ranks the 18 timed
rebuilds at Spearman **0.8962** (kkt 0.936 / o 0.829) with zero per-config
fitted constants — the knob→cost direction the table asserts is the same
one the solver ranks (`evidence/G3/validation.json`,
`evidence/G3/solver_report.md` §1).

## 2. G5 full-op lever A/Bs (shipped kernels, paired/interleaved)

Forward = staged trio F1 (WY) / F2 (scan) / F3 (o), mono cubin, w4/bv32
shipped; backward = 9-kernel chain (wy, fwdscan, caryA, caryM, bpre, dhs,
dkb, dqkwg, wyb). Baselines and per-round jsons as cited.

| lever (knob) | measured delta | stage / shape | verdict |
|---|---|---|---|
| **K-loops through the G2 transform pipeline** (multibuffer-2 + async groups + pipeliner) | staged fwd 1.597→0.663 ms B1T2048; 2.4×/2.2×/2.7× across 3 shapes | full fwd, r1→r2 | **win** (`evidence/G5/fwd_report.md` r2 table; `fwd_bench_round2.json`) |
| **thread-strided vs block-uniform staging** | block-uniform unrolled staging made EVERY thread issue EVERY copy (256× redundant cp.async) → measured 2× SLOWER than r1 before the fix | full fwd, r2 | **negative→fix** (the failure entry is `failure_longscroll.md` §1) (`evidence/G5/fwd_report.md` r2 §1.1) |
| **chunk ping-pong double-buffer (manual device_async_copy)** | F2 471→473 µs — **zero**; and the transform route can never overlap (create_async_groups waits group-0 immediately; source-verified in NVGPU/Transforms/CreateAsyncGroups.cpp) | F2 B1T2048, r3 | **neutral (kept for structure)** — the chunk chain was not load-latency-bound (`fwd_report.md` r3 §1a) |
| **LDSM padding (+8 bf16 cols on every ldmatrix-read tile)** | F2 473→249 µs (w8/bv64), then w4/bv32 → **83 µs quiet / staged 262 µs**; root cause 8-way LDSM bank conflicts, 63% MIO-stall | F1/F2/F3, r3 | **win, THE fwd r3 lever** (`fwd_report.md` r3 §1b–1c; `fwd_bench_round3.json`) |
| **warps×BV retune (w8/bv64 → w4/bv32 for F2)** | F2 smem 96.7→84.5 KB; VT doubles → 64 CTAs; F2 267→83 µs (quiet) | F2, r3 | **win (shipped config)** (`fwd_report.md` r3 §1c) |
| **segment-split scan (`--f2-segments S`)** | staged S=1 vs S=4: B1T2048 0.448 vs 0.478; B1T8192 1.298 vs 1.541/1.507 (S=4/8) = **16–33% LOSS**; Amdahl probe: chain+co-residency worth only 17% of F2 | F2, r4 | **negative, honest refutation** (`fwd_report.md` r4 §4; `fwd_bench_round4.json`) |
| **F2 barrier 2→1/chunk** | **+19.7 µs** (55/60 pairs slower); +5 KB smem (84.5→89.6) shrinks the L1 carveout the u-loads/store passes rely on | F2 B1T8192, r5 | **negative, REJECTED** (kept behind `G5K_F2BAR=1`) (`fwd_report.md` r5 §2; `scratch/g5k_ab_*.json`) |
| **F1UW (uw-section restage: bk=C straight-line, staging before solve, u/w through reused tiles, coalesced cell pass)** | f1 −123.3 µs @T8192 (4/60 slower), −149 µs staged B4T2048, −50 µs B1T2048; **B1T8192 adjudication 1.021→1.253/1.312×** | F1/full fwd, r5 | **win, THE r5 lever (shipped, default on)** (`fwd_report.md` r5 §2–3, §5) |
| **F1 A_m STS pad [64,64]→[64,65]** | −0.8 µs, 28/60 | F1, r5 | **neutral** (conflicts real but off critical path; `G5K_F1PAD=1`) |
| **F3 o-epilogue restage** | −1.3 µs f3 @T8192 (22/60), −8.4 µs f3 @B1T2048 | F3, r5 | **keep (noise-level positive, principled)** |
| **in-epilogue global stores (fused h_all/v_new into epilogues)** | F2 249→380 µs (**+40%**) — fragment-lane stores are 2B@4B-stride uncoalesced vs thread-strided cell passes | F2, r3 | **negative, reverted** (`fwd_report.md` r3 §1d) |
| **mma-ization of wy (bf16 m16n8k16 + ldmatrix) + fp32 chain + thread-partitioned staging** | bwd e2e 58.88→14.16 ms (**4.2×**); wy 3.50→1.59, dhu 24.48→3.69 | full bwd, r1→r2 | **win, the correctness+speed round** (`bwd_report.md` r2; `bwd2_bench2048.json`) |
| **tf32 (m16n8k8 via hatch #1) at f32-GEMM sites** | per-site bisect: 16 of 24 sites tf32'd; dqkwg 4.71→2.75 ms (1.71×), wyb 2.65→1.21 ms (2.19×); e2e 14.16→11.06 ms (1.28×); 6/6 numerics at both shapes | dqkwg/wyb, bwd r3 | **win (with 8 scalar holdouts, below)** (`bwd_report.md` r3; `bwd3_bench2048.json`) |
| **tf32 inside dhu (3 clean sites)** | r3: regs 128→255, spill 188→116 B, dhu 3.69→4.67 ms; r4 retest post-split: 246 regs → occupancy 2→1 CTAs/SM, 2.70→3.05 ms AND margins don't compose (dk 1.09e-2 > 8e-3 bar) | dhu, bwd r3+r4 | **double negative — structural** (no site fits ≤128 regs) (`bwd_report.md` r3 "did NOT work" + r4 §"did NOT work") |
| **async 16B staging chain-wide (`astage_tile`, manual cp.async bursts)** | ~4× fewer staging instructions per byte; part of 11.13→8.94 ms round | whole bwd chain, r4 | **win** (`bwd_report.md` r4 §2) |
| **dhu split → dkb kernel (chunk-summed partials)** | dhu 96.0→74.2 KB smem, −3 GEMM passes/chunk; dkb adds 0.96 ms but dqkwg 2.74→1.55; dhu 3.68→2.27 (1.62×); e2e 11.13→8.94 ms | bwd scan complex, r4 | **win** (`bwd_report.md` r4 §1,§perf) |
| **split-bf16 3-mma (dkB = hi+lo bf16 splits)** | dkb 0.96→0.25 ms (**3.8×**); dk numerics unchanged (1.730e-3 vs 1.731e-3 scalar; tf32 was 1.04e-2 over-bar) | dkb, bwd r5 | **win — the correct mma-for-f32 use case** (`bwd_report.md` r5 §4) |
| **dhu→dhs carry-scan restructure (X_{n-1}=A_n X_n+B_n, precomposed A/M/B + one rgemm/chunk)** | scan complex 3.13→2.01 ms (1.56×); dhu 2.27 retired → cary*+dhs 1.76; e2e 8.98→8.15 ms; carry math exact to 2–3 ulp fp64 | bwd, r5 | **win** (`bwd_report.md` r5; `bwd5_bench2048.json`) |
| **rgemm access order (cell-major → kk-outer, CPC=16 sharing one b-load)** | dhs 0.83→0.59 ms | dhs, bwd r5 | **win** (LDG-latency overlap) (`bwd_report.md` r5 §2) |
| **BV=64 retile of dhu** | does not fit: BV=32 already 96 KB of 101376 B; BV=64 needs ~150 KB+ | bwd, r3 | **negative (resource wall)** (`bwd_report.md` r3 negatives) |

Trajectory context for the §2 rows (B1T2048H16): fwd staged
1.597→0.663→0.262 (quiet)→0.215 (r5 paired-A/B) / 0.373 ms (r5 co-job
bench median) across r1→r2→r3→r5; bwd 58.88→14.16→11.06→8.94→**8.15 ms**
(r1→r5, 7.2× self-acceleration, 0.126× of fla). Adjudicated fwd vs
fla-triton: B1T2048 **1.592×**, B4T2048 **1.05–1.19×**, B1T8192
**1.253/1.312×** (`evidence/G5/GATE_REPORT_FINAL.md`;
`final_adjudicate_g5k_run1/2.json`).

Related artifacts: the honest negatives behind the "negative" verdicts are
expanded, with numbers and lessons, in `evidence/G6/failure_longscroll.md`.
