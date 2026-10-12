# G5-N: static instruction analysis of the landed bwd kernels (round-7, CPU-only)

Why this exists (user directive, 2026-10-08): the GPU is co-occupied for
~half a day; analysis that needs no clean timing window still counts.
This is the **analysis-without-timing** pass over the SHIPPED round-6
artifacts: PTX/SASS opcode histograms, occupancy, per-chunk compute
intensity, and distance-to-roofline per kernel — to rank
instruction-level inefficiency and aim rounds 8+.

Artifacts analyzed (freshest per-kernel cubins, landed chain):
`evidence/G5/scratch/r6_T2048/` — the round-6 9-kernel bwd
(wy, fwdscan, caryA, caryM, bpre, dhs, dkb, dqkwg, wyb; commit 455eef7;
the r4_*/r5_* scratch dirs are superseded by it). Histograms are computed
from `kernel_smemfix.ptx` + fresh `nvdisasm -c kernel_smemfix.cubin`
(the cubin the harness actually loads). References: the fwd mono trio +
fused flagship from `evidence/G5/gen_fwd/B1T2048_{mono,fused}`.

Tool: `tools/g5n_static_analysis.py` (deterministic, CPU-only; PTX
families via the importable `harvest_flaptx.ptx_features` FAMS table);
raw dump: `evidence/G5/scratch/g5n/static_analysis_r7.json`.
Measured per-stage medians: mean of `bwd6_bench2048.json` +
`bwd6_bench2048_rerun.json` (shared GPU; medians stable across the two
runs to <1.5%).

Calibration: `evidence/G3/calibration.json` with the CORRECTED rates
(G3 solver_report §"remaining blind spots": the file's `flops` field
over-counts x32 — `mma.sync` is warp-level; honest bf16 mma peak =
instr/s / 32 * 4096 = **115.5 TFLOP/s**). smem read 113.05 TB/s, DRAM
824.6 GB/s, L2-resident 2144 GB/s, bar.sync 7.7 ns — all measured UNDER
co-load (lower bounds), flagged `shared_gpu: true` in the file.

Shape: B1 T2048 H16 K=V=128 C=64 → 512 chunks (BH=16 x NC=32), NV=4.

## 1. Headline numbers

| kernel | meas ms | regs | smem B | occ (CTA/SM, limit) | HMMA sites/chunk (shape mix) | SASS LDS | LDSM | BAR | spill |
|---|---|---|---|---|---|---|---|---|---|
| wy      | 0.704 | **255** | 90624 | **12.5%** (1, smem) | 256 (128 bf16 k16 + **128 tf32 k8**) | 1000 | 96 | 37 | 0 |
| fwdscan | 0.917 | 64  | 65792 | 25% (1, smem) | **0** | 501 | 0 | 15 | 16 B |
| caryA   | 0.412 | 48  | 82176 | 25% (1, smem) | **0** | 611 | 0 | 18 | 4 B |
| caryM   | 0.478 | 40  | 65792 | 25% (1, smem) | **0** | 198 | 0 | 11 | 0 |
| bpre    | 0.320 | 40  | 49152 | 50% (2, smem) | **0** | 160 | 0 | 7 | 0 |
| dhs     | 0.588 | 48  | 32768 | 75% (3, smem) | **0** | 44 | 0 | 5 | 0 |
| dkb     | 0.249 | 128 | 89616 | 25% (1, smem) | 192 (bf16 k16) | 568 | 96 | 37 | 0 |
| dqkwg   | 1.663 | 234 | 97792 | 25% (1, smem) | 832 (**576 tf32 k8** + 256 bf16) | **3405** | 192 | **193** | 0 |
| wyb     | 1.040 | 234 | 91648 | 25% (1, smem) | 384 (**all tf32 k8**) | **3261** | **0** | 99 | 0 |

Occupancy = warps-resident / 32 warp slots (project sm_120 model,
`gdn_costmodel`: 82 SMs, 64K regs, 100352 B usable smem, 16 CTA slots,
32 warp slots). Every kernel is **smem-limited to 1 CTA/SM** except
bpre/dhs; wy is the extreme case: 255 regs and 90624 B smem pin it to
4 resident warps on an SM (12.5%).

## 2. Instruction-mix ratios (the fwd-F2 comparison)

PTX static counts, per mma site (`--` = no mma in kernel):

| kernel | ld.shared/mma | st.shared/mma | **ldmatrix/mma** | ld.global/mma | fp32-scalar-op/mma (SASS) | bar.sync/mma |
|---|---|---|---|---|---|---|
| wy     | 3.25 | 0.47 | 0.38 | 0.04 | 3.79  | 0.14 |
| fwdscan | -- | -- | -- | -- | (657 scalar ops, 0 tensor) | -- |
| caryA   | -- | -- | -- | -- | (707)  | -- |
| caryM   | -- | -- | -- | -- | (337)  | -- |
| bpre    | -- | -- | -- | -- | (272)  | -- |
| dhs     | -- | -- | -- | -- | (128)  | -- |
| dkb     | 0.30 | 0.34 | 0.50 | 0.07 | 8.41  | 0.19 |
| dqkwg   | **5.08** | 0.98 | 0.23 | 0.08 | 1.92  | **0.23** |
| wyb     | **6.77** | 0.99 | **0.00** | 0.11 | **9.65** | 0.26 |
| *fwd f1* | 0.82 | 2.20 | 1.06 | 0.02 | 3.84 | 0.22 |
| *fwd f2* | 0.69 | 0.77 | **1.19** | 0.27 | 1.66 | 0.05 |
| *fwd f3* | 0.09 | 0.16 | 0.63 | 0.01 | 1.46 | 0.12 |
| *fwd fused* | 0.29 | 0.55 | 1.16 | 0.04 | 1.81 | 0.06 |

The bwd signature, quantified: the fwd flagship stages mma operands
**through ldmatrix** (1.16-1.19 ldmatrix per mma, only 0.3-0.7
ld.shared per mma); the bwd mma kernels invert that — wyb has **zero**
ldmatrix and 6.8 ld.shared per mma, dqkwg 5.1 with 0.23 ldmatrix. Every
bwd ld.shared is a narrow `ld.shared.b32` (wy PTX: 828/832 are b32).
Root cause chain (r6 hazard, `bwd_report.md` r6 / GATE_REPORT_R6): tiles
staged by `LDGSTS` (manual cp.async: dqkwg has 96, wyb 36 LDGSTS sites)
must be read back by GENERIC ld.shared — ldmatrix on an LDGSTS-written
tile silently corrupts (the sm_120a silent-failure mode #2) — so the
LDS storm on exactly those tiles is a correctness workaround, not a
styling choice. dkb (the one kernel that restages everything through
st.shared) is the counterexample: 0.30 lds/mma, LDSM present, and the
smallest roofline distance of the chain (below).

Secondary signatures, all kernels: **zero `fma.rn.f32` in any bwd PTX**
— scalar f32 chains emit as separate add/sub/mul.rn.f32 (no FMA
contraction on this MLIR path; wy PTX: 115 mul + adds), doubling scalar
instruction counts; address arithmetic dominates SASS (dqkwg: IMAD 1666
+ MOV/SEL 2673 + branch 1203 vs 832 HMMA — the thread-partitioned
`scf.for` GEMMs keep 64-bit index math in the loop bodies).

## 3. Per-chunk compute intensity and distance-to-roofline

mma sites/chunk are exact for the four mma kernels (fully unrolled
bodies; wy's 256 static sites = exactly one CxCxK GEMM, the G2 selftest
anchor). Global footprint/chunk is exact (manifest globals). smem
traffic and CUDA-core FLOP are static-code estimates and are
**under-counts for the five loop-carried scalar kernels** (fwdscan,
caryA/caryM, bpre, dhs have runtime inner loops; their static bodies are
tiny — dhs 584 SASS total — while the r6 report documents ~4.2K
ops/thread/chunk for dhs's rgemm); math-based FLOP for those (from the
stage formulas) is noted below.

Floors = whole-problem work at calibrated peak rates (dram floor uses
the exact manifest bytes; rates are co-load lower bounds):

| kernel | footprint MiB (total) | tensor MFLOP/chunk | dram floor ms | largest floor | meas ms | **distance** | achieved DRAM GB/s |
|---|---|---|---|---|---|---|---|
| wy      | 80.2 | 0.786 (site-mix) | 0.102 | dram | 0.704 | 6.9x  | 120 |
| fwdscan | 88.6 | 0 (math: ~4.2 scalar) | 0.113 | dram | 0.917 | 8.1x  | 101 |
| caryA   | 56.1 | 0 (math: ~4.2/boundary) | 0.071 | dram | 0.412 | 5.8x  | 143 |
| caryM   | 48.1 | 0 (math: ~4.2/boundary) | 0.061 | dram | 0.478 | 7.8x  | 106 |
| bpre    | 56.0 | 0 (math: ~2.1) | 0.071 | dram | 0.320 | 4.5x  | 183 |
| dhs     | 97.5 | 0 (math: ~6.3) | 0.124 | dram | 0.588 | 4.8x  | 174 |
| dkb     | 104.3 | 0.786 | 0.133 | dram | 0.249 | **1.9x** | **439** |
| dqkwg   | 200.9 | 2.228 (site-mix) | 0.255 | dram | 1.663 | 6.5x  | 127 |
| wyb     | 113.0 | 0.786 (tf32) | 0.144 | dram | 1.040 | 7.2x  | 114 |

No kernel saturates ANY measured resource (tensor <= 1.6 TF of 115.5;
smem <= 4.4 TB/s of 113; DRAM <= 439 GB/s of 824.6 lower-bound) — the
binding constraint is **instruction issue + dependency stalls** (barrier
waits at 1 CTA/SM, narrow LDS streams, scalar address math), which is
exactly what an instruction-level pass can attack. Structural evidence:
scan kernels launch grid=64 blocks on an 82-SM part (fwdscan/dhs
under-fill the device by construction and serialize NC=32 chunk steps
with 5-15 bar.sync per step).

Two aggregate facts for the round-8 ledger:

* **Sum of dram floors = 1.074 ms** vs measured 7.27 ms (6.8x) — and
  vs **fla bwd 1.03 ms**: the 9-kernel staged design already moves
  enough bytes that its DRAM floor alone is ~fla's whole bwd. Traffic
  (845 MiB/pass, f32 intermediates by the G5-E correctness decision)
  and execution efficiency BOTH have to move to pass fla.
* dkb is the in-house existence proof: 1.9x from its floor at 439 GB/s
  achieved — same generator, same constraints, 0.30 lds/mma.

## 4. Ranked instruction-level inefficiency list (round-8 ammunition)

Ranked by recoverable-ms (measured − largest floor) x lever clarity:

1. **dqkwg (1.66 ms, recoverable ≥1.4)** — LDS storm: 3405 LDS / 832
   HMMA (5.1 lds/mma vs fwd-F2 0.69), ldmatrix only 0.23/mma because
   the mma operand tiles are LDGSTS-staged (96 cp.async sites) and the
   r6 hazard forces generic readback. Plus 576/832 mma sites are tf32
   m16n8k8 (half-rate vs bf16 k16), 193 bar.sync, IMAD 1666 + MOV/SEL
   2673 + branch 1203. Levers: st.shared/element-wise restage of the
   A/B tiles (the r6 fix pattern) to re-enable ldmatrix; split-bf16 for
   the tf32 GEMMs (the dvp_bh precedent: 3-mma→2-mma + raw-bf16 A);
   barrier coalescing (the F2 2→1/chunk lever, measured +19.7 us there,
   re-check under the new mix).
2. **wyb (1.04 ms, recoverable ≥0.9)** — worst mix: 6.8 lds/mma, ZERO
   ldmatrix (all operand tiles LDGSTS-or-scalar staged), 9.65 fp32
   scalar ops per mma (the dAf epilogue + dk/dbeta/dg reductions are
   per-thread f32 chains with no FMA contraction), 99 bar.sync. Levers:
   same restage→ldmatrix fix; split-bf16 the all-tf32 contraction (384
   k8 sites → k16 half-count at same error class as dqkwg's landed
   split); warp-level reductions (redux/shfl) for dbeta/dg.
3. **The no-tensor-core five — fwdscan, caryA, caryM, bpre, dhs
   (2.72 ms = 37% of bwd, recoverable ≥2.2)** — zero HMMA sites across
   all five; math says they carry ~2.1-6.3 MFLOP/chunk of GEMM-shaped
   scalar f32 (the [K,K] composes are 2·128³ each). fwdscan/dhs also
   under-fill the device (grid 64 < 82 SMs, 1 CTA/SM) and serialize 32
   chunk steps behind 5-15 bar.sync each. Levers: mma-ize the compose
   GEMMs (split-bf16, the documented "compose is the candidate"); segment
   scan to shorten the serial chain; raise blocks-per-SM for the scan
   kernels (smem 65 KB → 1 CTA/SM caps latency hiding at 8 warps).
4. **wy (0.70 ms, recoverable ≥0.6)** — 255 regs + 90624 B smem → 12.5%
   occupancy (4 resident warps/SM): the least latency-tolerant kernel
   has the least latency hiding. Half its mma sites are tf32 k8 (kkt).
   Levers: register-pressure pass on the solve (the r6 column-parallel
   solve is the reg hog), bf16 mma for kkt, then occupancy doubles.
5. **Chain-wide (no single owner)** — (a) zero FMA contraction: every
   scalar f32 MAC costs 2 instructions + extra register traffic on a
   path that is issue-bound; a generator-level `a*b+c` → fma fold pays
   everywhere. (b) f32 intermediates: 845 MiB/pass; the dk numerics
   margin is 4.5x at T2048 — bf16 halves of the big tiles (dh_all,
   h_all, dW_all: 32+32+16 MiB per pass each) are the traffic lever,
   gated on the error budget, not on execution.

## 5. Method notes / caveats (recorded, not hidden)

* Static counts = dynamic per-chunk counts only for fully-unrolled
  bodies (true for the four mma kernels; verified: wy's 256 sites ==
  one CxCxK GEMM in m16n8k16+m16n8k8 units). The five scalar kernels
  have runtime loops → their smem-traffic/CUDA-FLOP estimates here are
  under-counts; their roofline distances are therefore UPPER bounds
  (the ranking is insensitive: their floors are <=0.13 ms vs 0.32-0.92
  measured).
* Global footprint = sum of manifest global sizes (each touched at least
  once); a global both read and written by one kernel would under-count
  by that global's size once. Classification-free and reproducible.
* Calibration rates are co-load lower bounds (shared_gpu: true); in a
  clean window the peaks are higher, i.e. true distances are LARGER,
  not smaller. Order-of-magnitude conclusions unaffected.
* fwd reference smem per entry was not re-derived (single-module meta);
  its ratio rows need only PTX counts, which are per-entry exact.

## 6. Repro

```
conda activate gdn && cd ~/codebase/gdn-mlir
OMP_NUM_THREADS=4 python tools/g5n_static_analysis.py \
  --gen evidence/G5/scratch/r6_T2048 \
  --bench evidence/G5/bwd6_bench2048.json \
          evidence/G5/bwd6_bench2048_rerun.json \
  --out evidence/G5/scratch/g5n/static_analysis_r7.json
```

T512 cross-check (scaling sanity, `bwd6_bench512.json`): per-chunk
kernels drop 4-8x at NC=8 (footprint 26 MiB passes fit the 64 MB L2);
fwdscan/dhs scale ~linearly with NC (0.25/0.21x) — consistent with the
serial-scan + dram-floor picture above.
