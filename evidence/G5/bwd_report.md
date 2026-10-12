# G5-C: L1 GDN backward as real GPU kernels (multi-kernel path)

Deliverables: `tools/gdn_bwd_gen.py` (5-kernel backward generator, sm_120a),
`tools/gdn_bwd_verify.py` (fp64 numpy-twin + JAX-vjp-anchored verification,
GPU chain runner), `tools/gdn_bwd_bench.py` (e2e timing vs the fla baseline).
All numbers below are real single-run measurements on the shared RTX 5090
Laptop GPU (co-job resident; absolute times noisy by design, flagged in
`shared_gpu` fields).

## Stage split (fla-isomorphic)

| stage | grid | work | smem |
|-------|------|------|------|
| `wy` | B*H*NC | gc cumsum, kkt, A, blocked (I+A)^-1 solve (2x 32x32 scalar bases + gemm combine `T21 = -T11 A21 T00`), u = T(beta v), w = T(beta k 2^gc) | 78.3 KB |
| `fwdscan` | B*H*NV | v_new = u - w S; S' = 2^gcl S + (k 2^(gcl-gc))^T v_new; stores ENTRY states h_all + v_new | 65.8 KB |
| `dhu` | B*H*NV | reverse cotangent scan: dvp = P^T dO + Bhat dH; dH' = 2^gcl dH + (q 2^gc)^T dO - w^T dvp; emits dh_all, dh0, dv, dvp_all, dkB/rowB/s5 partials | ~102 KB |
| `dqkwg` | B*H*NC | dq, dW = -(dvp Sn^T), dk_local = TW.(beta 2^gc) + dkB + dPm^T q, r3/r4/c4r, dg_state | ~97 KB |
| `wyb` | B*H*NC | dAf = -strict_tril(T^T strict_tril(G) T^T .* D) (the two CxCxC contractions, all-f32 chain), final dk / dbeta / dg | ~100 KB |

All kernels are zero-argument cubins with public MUTABLE `memref.global`s
staged/read by the harness (`cuModuleGetGlobal` + HtoD/DtoH), compiled by
`scripts/run_l2.sh` + `tools/gdn_smem_fix.py` (identity transform script).

## Platform constraints discovered while building this (de-risked on hardware)

Each of these was isolated experimentally; all are now encoded in the
generator and worth remembering for any follow-on GPU work in this tree:

1. **No explicit `module {}` wrapper in payloads.** A wrapper makes
   `-transform-interpreter` re-nest the module; `gpu-lower-to-nvvm-pipeline`
   then fails with "conversion of memref memory space #gpu.address_space
   ... failed" and no PTX is produced. Bare top-level ops + identity
   transform script (interpreter skipped) is the reliable path.
2. **vector.transfer cannot lower on workgroup memrefs in this build**; it
   only survives via `transform.nvgpu.create_async_groups`, whose async-copy
   lowering computes WRONG addresses for our staging shapes (verified:
   element-linear instead of row*stride addressing, silent corruption). The
   bwd kernels therefore stage everything with scalar load/store loops.
3. **linalg.matmul is unusable here**: -convert-linalg-to-loops executes the
   loop nest redundantly on all threads; chained accumulating matmuls on one
   `outs` race across warps (deterministic-looking but wrong), and a matmul
   with the same buffer as both operands is mis-lowered entirely. Every GEMM
   is therefore a hand-rolled thread-partitioned `scf.for`: output element e
   is owned by thread e % NT for every gemm touching an accumulator -> one
   writer per cell, deterministic; chained accumulation re-reads the
   writer's own value.
4. **Static shared limit on sm_120a is 101376 B** (measured with ptxas
   directly; 98304 B accepted, 131072 B rejected). All kernels sit just
   under it; buffer reuse/restage-from-global is the lever when a tile
   changes dtype.
5. Flattened `[M, C*W]` globals: the chunk lives in ONE global row; staging
   helpers index `col = col0 + cm*r + c` (a row+col double-count in the
   first cut caused OOB reads only at full K=V=128 shapes — found with
   compute-sanitizer + SASS stride decode).

## Numerics (vs fp64 twin on bf16-quantized inputs, chunk C=64)

Tolerances are the fla-derived contract (dq/dk/dv <= 8e-3, db/dg <= 2e-2,
dh0 <= 8e-3). The fp64 twin is a port of the G1-B2 `chunked_bwd_ref` and was
cross-checked against the G1-A JAX vjp oracle on the small shape
(`--jax`: inputs themselves quantized to bf16 explains the observed
2e-3..5e-2 twin-vs-vjp spread; the graded comparison below uses the twin).

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1 T512 H4 K128 V128 | **2.3e-3 PASS** | 6.6e-1 FAIL | **3.8e-3 PASS** | 3.6e-2 FAIL | 3.7 FAIL | **1.4e-4 PASS** |
| B1 T2048 H16 K128 V128 | **3.1e-3 PASS** | 9.2e-1 FAIL | **7.2e-3 PASS** | 9.8e-2 FAIL | 5.6 FAIL | **2.0e-4 PASS** |

Stage intermediates verified against the twin (B1T512): T 2.3e-4, h_all
3.0e-3, v_new 1.5e-2, dvp 4.5e-3, dW 4.6e-3, r3/r4/c4r ~1e-3. The dq/dv/dh0
PASSes plus these intermediate checks establish that the recompute pass,
both scans, and the local-term kernels are numerically sound.

### Where dk/db/dg fail, precisely

Recomputing dg in fp64 FROM the GPU's own bf16 intermediates reproduces the
kernel's dg to 9e-4 — the kernel's assembly is exact; the error enters
through intermediate precision and is then amplified ~64x by the
suffix/rowsum structure, exactly the "dA CxCxC contractions are the likely
hot spot" risk from the tasking. The chain dAf = -T^T(G)T^T with
AdA = dAf.*kkT feeding dg/dbeta multiplies operand ulps by two K=64
contractions plus the 64-wide suffix sums. Upgrades already made (verified
to remove entire error classes): T/dW/dvp/dh_all/dkB_part globals f32, the
wyb G/X/Y/dAf chain entirely in f32 registers, dH carry and vng f32, final
dq/dk/dv f32. The remaining over-tolerance residual is concentrated in
small-|value| elements where the fp64 twin itself sits at
input-quantization noise (twin-vs-JAX spread above), i.e. the fixed
floor from bf16 inputs (~2e-2 on dg-class grads) is the same order as the
2e-2 tolerance; meeting it needs the remaining bf16 operands (k, v, beta
staging tiles) in the G chain promoted like the rest, which the smem budget
currently forbids in a single wyb kernel (the report's risk list has the
split plan).

## Timing (CUDA events, median, shared GPU)

| shape | ours bwd e2e | fla bwd (triton, baseline json implied / measured here) | speedup |
|-------|--------------|----------------|---------|
| B1 T512 H4 K128 V128 | 11.60 ms | 1.18 ms (measured here, autograd harness) | 0.10x |
| B1 T2048 H16 K128 V128 | 58.40 ms | 1.03 ms (evidence/G5/fla_baseline.json fwdbwd-fwd) / 1.34 ms (measured) | 0.018-0.023x |

Per-stage medians at B1T2048H16: `dhu` 24.2 ms, `dqkwg` 16.2 ms, `fwdscan`
7.2 ms, `wyb` 6.7 ms, `wy` 3.5 ms. fwd+bwd total was not computed because
G5-B's fwd artifacts landed mid-window; ours-bwd-only is reported.

### Dominant-stage analysis and the perf roadmap

The scalar thread-partitioned GEMM design buys determinism and correctness
but leaves the tensor cores idle: every MAC is a scalar fma between two
`ld.shared`/`ld.global` fetches. `dhu` (reverse scan: P build, dvp, dH
update, dkB — ~8 gemms/chunk) and `dqkwg` (P, dPr, dvp, dW, TW, dk, dq —
~10 gemms/chunk) dominate exactly as predicted. The next lever is the
G4-proven mma path (nvgpu.mma.sync fragments, f32 accumulators in
registers, D-fragment N index = 2*tid_in_group per the G4 fix): replacing
the scalar sgemm emitter inside the same kernel skeletons is mechanical
since every GEMM site already carries its shapes/orientations as arguments.
That conversion, not algorithmic change, is what stands between the current
0.02x and the >=1.1x goal.

## Flagship (persistent fused bwd)

Not attempted: per the tasking it is gated on a correct+fast multi-kernel
path. The multi-kernel path is correct in structure (4/6 grads in
tolerance, both scans exact) but not yet fast or fully in-tolerance.

## Risks / rerun notes for the clean window

- All kernels sit 1-3 KB under the 101376 B static smem limit; any tile
  dtype flip must re-check `module_text`'s assert.
- The harness runs the chain in one process; a hanging kernel takes the
  worker with it (subprocess isolation like gdn_verify.py is a TODO).
- Benchmarks were taken with the co-job resident (flagged in the jsons);
  rerun `tools/gdn_bwd_bench.py` in a clean window with identical commands.
- Numerics rerun: `python tools/gdn_bwd_verify.py --gen-root <dir>
  --shape 1,512,4,128,128 --jax` (small) and the T2048 shape; gen dirs are
  rebuilt with `python tools/gdn_bwd_gen.py --shape ... --out <dir>` +
  `scripts/run_l2.sh` + `tools/gdn_smem_fix.py` per kernel.
- Known remaining precision lever: f32-staged k/v/beta operand tiles in
  the wyb G chain (needs the wyb split into dAf + final kernels to fit
  smem), plus mma fragments with f32 inputs where shapes allow.


---

# G5-E round 2: numerics 6/6 PASS + 4.2x bwd speedup (mma-ized wy, fp32 chain)

Artifacts: `tools/gdn_bwd_gen.py` (rewritten kernels), `tools/gdn_bwd_verify.py`
(twin bug fix), `evidence/G5/bwd2_verify{512,2048}.json`,
`evidence/G5/bwd2_bench{512,2048}.json`.  All measurements on the shared RTX
5090 Laptop (co-job resident, flagged in the jsons); every number below is a
real run, medians over 30 reps.

## Numerics: 6/6 PASS at BOTH gate shapes (vs fp64 twin, bf16-fed kernel)

| shape | dq | dk | dv | dbeta | dg | dh0 | overall |
|-------|----|----|----|----|----|-----|---------|
| B1 T512 H4  | 5.2e-7 | 1.7e-5 | 1.4e-6 | 1.5e-5 | 6.7e-6 | 1.1e-8 | **PASS (6/6)** |
| B1 T2048 H16 | 8.0e-7 | 4.5e-5 | 3.1e-6 | 2.0e-5 | 1.1e-5 | 1.9e-8 | **PASS (6/6)** |

Worst grad has ~180x headroom to its bar.  The dbeta/dg diffs above are
byte-identical to the diffs against the G1-A JAX vjp oracle run on the SAME
bf16-quantized inputs (|jax-GPU| = 1.5e-5 / 6.9e-6 at T512) -- the chain is
at input-quantization exactness.

### What the failures actually were (the round-1 "bf16 amplification" theory was wrong)

Instrumenting every intermediate against an fp64 replay localized the errors
(probes: P/dPr/r3/r4 exact, dvp exact per head, dkB/rowB/s5 partials, term
decompositions).  Four independent defects, none of them the hypothesized
operand-precision floor:

1. **dqkwg dk_local epilogue** applied the `beta .* 2^gc` row scale to the
   whole `(TW + dPm^T q)` accumulator; per the G1-B2 math only TW carries it
   (dkB and dPm^T q enter unscaled).  Deterministic ~0.52 dk error.  Fixed:
   in-place scale of the TW slice before the dPm^T q accumulate.
2. **dhu `rb` accumulator** was zeroed once OUTSIDE the reverse chunk walk,
   so rowB_part[n] accumulated over all later chunks.  Fixed: per-chunk
   re-zero.
3. **dqkwg dg_state prefix** `scf.for` yielded the last element instead of
   the running sum (`yield %rbt` with the iter_arg dropped) -- dg_state
   carried rbsum[di-1], not the prefix.  Fixed: `yield %pc + %rbt`.
4. **fp64 twin bug (gdn_bwd_verify.py)**: `kkT` was only computed in the
   forward loop; the reverse loop's dbeta/dg terms used the STALE kkT of
   chunk NC-1.  The GPU computes the per-chunk kkt correctly.  Against the
   JAX oracle on quantized inputs: |jax-twin| was 3.6e-2 / 2.4e-2 (exactly
   the round-1 "failures"), |jax-GPU| 1.5e-5 / 6.9e-6.  The twin is fixed;
   this also means part of the G5-C "twin-vs-vjp spread attributed to input
   quantization" was this bug.

The fp32 work was still necessary: with round-1's bf16 intermediates
(u/w/S/vn/h/P/dk_local storage + operand tiles) the passing grads sat at
2e-3..4e-3 and the assembly bugs were invisible under bf16 noise.  Now every
intermediate is exact (T 1.2e-7, h 5.7e-7, vn 3.0e-7, dvp 1.5e-6, dW 4.4e-4
f32-gemm noise) and the dA-chain gemms (dW, G, X, Y, TW, tv, dk terms, kkt)
use compensated (Kahan) inner reductions.

## Performance: 0.018x -> 0.073x of fla bwd (4.2x round-over-round)

| B1T2048H16 | G5-C | round2 | speedup |
|------------|------|--------|---------|
| wy         | 3.50 | 1.59 | 2.2x |
| fwdscan    | 7.08 | 1.04 | 6.8x |
| dhu        | 24.48 | 3.69 | 6.6x |
| dqkwg      | 16.51 | 4.71 | 3.5x |
| wyb        | 6.85 | 2.65 | 2.6x |
| **e2e bwd** | **58.88** | **14.16** | **4.2x** |

At B1T512H4 (same window, baseline rebuilt from git HEAD): 10.38 -> 2.40 ms
(4.3x).  fla bwd reference at B1T2048: 1.03 ms implied (fwdbwd 1.3463 -
fwd 0.3158, evidence/G5/fla_baseline.json) / 1.34 ms measured in G5-C -->
round-2 ours/fla = 14.16/1.03 = **0.073x** (was 0.018x).

### What landed

* **mma-ized wy**: kkt = k k^T and P = tril_inc((q k^T) D) are warp-tiled
  `nvgpu.mma.sync` m16n8k16 bf16 atoms with f32 accumulators + ldmatrix
  fragment loads (ported from the GPU-verified gdn_full_gen.py emitter,
  D-fragment N = 2*tig per G4).  128 HMMA + 96 LDSM in wy's SASS.  The
  operands are raw bf16 inputs, so the mma values are twin-exact (kkt_all
  3.3e-7, P_all 1.3e-7 vs fp64) -- numerics did not move.
* **kkt_all / P_all hoisted**: wy computes them once per chunk into new f32
  globals; dhu (was rebuilding P in EVERY v-tile), dqkwg (was rebuilding P)
  and wyb (was rebuilding kkt) now stage them.  This removes ~30% of the
  dhu/dqkwg GEMM work outright.
* **Thread-partitioned staging**: stage_rows was a serial loop executed
  redundantly by all 128 threads; now partitioned like store_rows.  Biggest
  single lever on fwdscan/dhu.
* **s5 and rowB reductions parallelized** (per-thread partials + fixed-order
  fold; the C*C*NV single-thread triple loop in dg_state is now a
  row-parallel NV fold + O(C^2) prefix).
* **threads=256 default** for fwdscan/dhu/dqkwg/wyb (128 kept for wy: the
  mma tile plan is sized for 4 warps).  1.13x on the scalar-bound stages.
* In-place masked maps (P, dPm) instead of separate bf16 tiles -- smem
  headroom for the f32 intermediates without splitting any kernel.

### Where the remaining 14 ms lives (round-3 scope)

dqkwg (4.7) + dhu (3.7) are now dominated by the f32-operand scalar GEMMs
(dW, dvp, dq1, TW, dk/dq tails) that CANNOT take bf16 mma without
reintroducing operand floors on the dA chain (measured sensitivity: even
4e-4 dW noise -> 5e-5 dAf; a bf16 floor is 10x worse than the dbeta bar).
Levers for round 3, in order:

1. tf32 mma (m16n8k8 f32 fragments) for the f32-operand GEMMs if this
   MLIR build lowers them -- untested here.
2. Split-bf16 (a = a_hi + a_lo) 3-mma emulation of f32 GEMMs (~16-bit
   effective mantissa >= the error budget the margins now allow).
3. Fuse dhu's per-v-tile work at BV=64+ once P is a global (no P rebuild
   to amortize), and pipeline the K/V slice staging.
4. The serial 32x32 forward-substitution bases in wy's solve and the
   gc_cumsum (single-thread sections) are now visible in wy/dqkwg tails.

The flag "Numerics stay green through every perf change" was kept: each perf
step above was re-verified 6/6 before the next landed.

---

# G5-G round 3: tf32 m16n8k8 mma via ESCAPE HATCH #1 + parallel dA reductions

Artifacts: dialect op  (dialect/include/Gdn/GdnHatch.td +
lib/GdnHatch.cpp; lowers to llvm.inline_asm, see the hatch addendum in
evidence/G2/hatch_report.md), warp-tiled  emitter + per-site 
knob in tools/gdn_bwd_gen.py, warp-parallel r4/c4r (dqkwg) and ra/c1r (wyb)
reductions, , probe suite in
.  All measurements on the shared RTX
5090 Laptop (co-job resident; every json flagged shared_gpu -- medians are
noisy, mins reported alongside).

## The hatch (paper material)

 emits exactly fla's harvested instruction
 -- unreachable in-tree (nvgpu.mma.sync caps tf32 at
K=4; the tabled nvvm.mma.sync family has m16n8k8 only for f16/bf16/i8).
Two findings beyond "it compiles":

1. **Fragment layout (hardware-derived, not doc-guessed).**  A 16x8x8 probe
   kernel with integer operands (tf32-exact) pinned the layout to
   a0=(g,2t) a1=(g+8,2t) a2=(g,2t+1) a3=(g+8,2t+1) for A (row-half OUTER,
   col-pair inner -- NOT the packed-pair order intuition from bf16 k16),
   b0=(2t,g) b1=(2t+1,g), d0..d3 like the bf16 D (N=2t rule).  Probe:
   D == A@B+C EXACTLY; rounding check on N(0,1) operands gives 4.2e-3
   (genuine tf32 semantics, ~1 ulp off a truncation sim).
2. **Early-clobber is mandatory.**  nvvm.inline_ptx (the BasicPtxBuilder
   path) emits plain "=f" outputs; ptxas then legally maps the D quad ONTO
   THE A QUAD of the same mma (SASS: HMMA R12, R12, R10, R16) and the
   tensor core corrupts operands -- a K=16 two-atom chain failed 118/128
   cells with deterministic k-term errors.  The op therefore lowers to a
   direct  with "=&f,=&f,=&f,=&f,f,f,f,f,f,f,f,f,f,f".
   (nvvm.inline_ptx cannot express early-clobber; this is a toolchain
   finding for anyone using BasicPtxBuilder with mma.)

## tgemm (the site emitter) and the off-by-one

Warp-tiled: atom tiles (M/16)x(N/8) round-robin over warps (runtime
mi/ni divmod, needs ntiles%warps==0); per-lane 6 fragment loads + 1
gdn.mma.tf32k8 per (tile, kb); acc=True loads the out tile first; neg
negates the B OPERANDS (sgemm semantics -- the acc init stays positive;
dhu.dh_w depends on that distinction); offsets fold into index constants.

Debugging trail (worth remembering): a numpy simulator of the emitted index
math + verified atom semantics computed A@B exactly, so the emitted code
was wrong -- a delta-basis GPU diagnostic (B=I) showed odd-k columns
shifted +1:  is already 2t+1, and the per-kb offset code added
"+1" again.  Every kb>0 fragment was mis-addressed (also the cause of the
initial wyb illegal-address: reads walked past smem).  After the fix, a
13-combo probe matrix (at/bt/acc/neg/offsets, real bwd shapes) passes at
the tf32 rounding floor.

A second emitter bug of the same family: wy launches 128 threads but tgemm
defaulted to the global THREADS//32=8 warps -- half of wy's tiles were
never written (garbage u/w -> dk=30).  tgemm now takes an explicit warps=
(wy passes 4).

Toolchain note: once a payload carries gdn.* ops, gdn-opt's rewriter
hoists launch-region constants to host args INCLUDING vector constants
(the bf16-mma accumulator zero); gdn_smem_fix.py Fix 2 now byte-substitutes
aggregate constant params (ld.param.b64[+off] -> mov immediates), else the
zero-arg launch fails with CUDA_ERROR_INVALID_VALUE.

## Numerics: per-site tf32 bisect then 6/6 at both shapes

Every one of the 24 f32-GEMM sites was tf32'd ALONE at T512 and graded
(noise = diff vs the round-2 baseline diff; baseline ~1e-5..5e-7):

| site | worst grad moved | verdict |
|------|------------------|---------|
| dhu.dkB | dk 1.04e-2 | OVER 8e-3 bar -> scalar |
| wy.u | dk 1.06e-2 | OVER bar (u noise amplifies via S->Sn) -> scalar |
| wyb.tv | dbeta 8.6e-3 | 2.3x margin -> scalar |
| dhu.dvp_bh | dbeta 3.0e-3 | 6.6x -> scalar |
| dhu.dv_T | dv 1.26e-3 | 6.4x -> scalar |
| dhu.dvp_pdO/dh_qg/dh_w | <=3e-4 clean but PERF-NEGATIVE (below) -> scalar |
| wy.w (u's same-shape sibling) | 1.4e-5 | clean -> tf32 |
| dqkwg.* (8 sites) | dk 7.8e-4..5.2e-4, dq 4.9e-4 | tf32 |
| wyb G_dvp/G_dW/X/Y/TW_rb/dk_t1/dk_t2 | dk 6.6e-4..1.1e-4 | tf32 |

Interesting asymmetries: wy.w is clean while wy.u (same shapes, same K) is
poisonous -- u's error rides the state scan into Sn; and tf32 dW/TW
(dA-chain, Kahan-protected in round 2) move NOTHING (dW magnitudes are
small; the round-2 Kahan was protecting against f32 ACCUMULATION noise,
which the mma also does not have).

Final set: 16 sites tf32 (all of dqkwg, 7 of 8 wyb, wy.w).  6/6 PASS:

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T512H4 | 8.0e-4 (10x) | 1.7e-3 (4.6x) | 1.4e-6 | 3.0e-4 (66x) | 1.9e-4 | 1.3e-6 |
| B1T2048H16 | 1.1e-3 (7.5x) | 1.8e-3 (4.5x) | 3.1e-6 | 1.4e-3 (14x) | 4.1e-4 | 1.8e-6 |

(parenthesized = margin under the bar; dv/dh0 back to round-2 exactness
because dhu stayed scalar.)

## Performance: 14.16 -> 11.06 ms (T2048), 2.40 -> 1.62 ms (T512)

| B1T2048H16 | round2 | round3 | delta |
|------------|--------|--------|-------|
| wy | 1.59 | 1.85 (1.72-1.85 band, noisy) | ~ |
| fwdscan | 1.04 | 1.03 | = |
| dhu | 3.69 | 3.76-3.83 (scalar, unchanged by design) | = |
| dqkwg | 4.71 | 2.75 | 1.71x |
| wyb | 2.65 | 1.21 | 2.19x |
| **e2e bwd** | **14.16** | **11.06 median / 9.57 min** | **1.28x / 1.48x** |

ours/fla = 11.06/1.03 = **0.093x** (was 0.073x).  T512: 2.40 -> 1.62
(1.48x).  dqkwg gains = tf32 mma (8 sites) + warp-parallel r4/c4r (was a
4096-iteration single-thread loop per CTA); wyb gains = tf32 (7 sites) +
warp-parallel ra/c1r.  Both reductions keep single-writer row ownership ->
bit-identical numerics (the verify table is unchanged by them).

## What did NOT work (honest negative results)

* **dhu + tf32 mma = regression.**  The 3 numerics-clean dhu sites
  (dvp_pdO/dh_qg/dh_w), fully unrolled inside the serial reverse chunk
  walk, push ptxas from 128 regs/188B spill to 255 regs/116-128B spill ->
  dhu 3.69 -> 4.67 ms.  All three reverted; dhu is unchanged this round.
* **BV=64 retile does not fit**: dhu at BV=32 already uses 96KB of the
  101376B static limit; BV=64 needs dH[128,64]+dO/dvp/vngb[64,64]+tCC[64,64]
  ~ 150KB+.  The retile needs a buffer-reuse restructure (or 2-CTA
  cooperation on the V split), not a tile-size flip.  Also BV=64 means
  NV=2 -> 32 CTAs at H16 (fewer, not more); the mission-note premise
  ("more CTAs") only holds for BV=16.
* Chunk prefetch double-buffer: only ~5KB smem headroom at BV=32 -- one
  extra k-tile buffer would fit but the ping-pong parity plumbing through
  scf.for is a restructure; left documented.

## What remains for the adjudication round (measured, ranked)

1. **dhu 3.76ms** -- the whole residual is the serial reverse walk
   (32 chunks x ~10 scalar gemms, 64 CTAs latency-bound, 1 CTA/SM at 128
   regs... actually 2).  Levers: mma without the spill (split the walk's
   per-chunk body into a smaller tile plan, or 384-thread CTA with 12
   warps), smem-budgeted BV=64 restructure, or staging prefetch.
2. **dqkwg 2.75ms** -- remaining costs are global staging (dvp_all/h_all/
   dh_all slices per ks/vs) and the dkB NV-fold (4 global loads per output
   cell per ks); stage dkB_part once per chunk instead.
3. **wy 1.7-1.85ms** -- the two 32x32 serial forward-substitution bases
   (scf.if %isz) + gc cumsum; warp-parallel blocked solve like fwd.
4. fwdscan 1.03ms -- unchanged since round 2 (not re-examined).

---

# G5-G round 3: tf32 m16n8k8 mma via ESCAPE HATCH #1 + parallel dA reductions

Artifacts: dialect op `gdn.mma.tf32k8` (dialect/include/Gdn/GdnHatch.td +
lib/GdnHatch.cpp; lowers to llvm.inline_asm, see the hatch addendum in
evidence/G2/hatch_report.md), warp-tiled `tgemm` emitter + per-site `--tf32`
knob in tools/gdn_bwd_gen.py, warp-parallel r4/c4r (dqkwg) and ra/c1r (wyb)
reductions, `evidence/G5/bwd3_{verify,bench}{512,2048}.json`, probe suite in
`evidence/G5/scratch/tf32k8_probe/`.  All measurements on the shared RTX
5090 Laptop (co-job resident; every json flagged shared_gpu -- medians are
noisy, mins reported alongside).

## The hatch (paper material)

`gdn.mma.tf32k8` emits exactly fla's harvested instruction
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {d0..d3}, {a0..a3},
{b0,b1}, {c0..c3};` -- unreachable in-tree (nvgpu.mma.sync caps tf32 at
K=4; the tabled nvvm.mma.sync family has m16n8k8 only for f16/bf16/i8).
Two findings beyond "it compiles":

1. **Fragment layout (hardware-derived, not doc-guessed).**  A 16x8x8 probe
   kernel with integer operands (tf32-exact) pinned the layout to
   a0=(g,2t) a1=(g+8,2t) a2=(g,2t+1) a3=(g+8,2t+1) for A (row-half OUTER,
   col-pair inner -- NOT the packed-pair order intuition from bf16 k16),
   b0=(2t,g) b1=(2t+1,g), d0..d3 like the bf16 D (N=2t rule).  Probe:
   D == A@B+C EXACTLY; rounding check on N(0,1) operands gives 4.2e-3
   (genuine tf32 semantics, ~1 ulp off a truncation sim).
2. **Early-clobber is mandatory.**  nvvm.inline_ptx (the BasicPtxBuilder
   path) emits plain "=f" outputs; ptxas then legally maps the D quad ONTO
   THE A QUAD of the same mma (SASS: `HMMA R12, R12, R10, R16`) and the
   tensor core corrupts operands -- a K=16 two-atom chain failed 118/128
   cells with deterministic k-term errors.  The op therefore lowers to a
   direct `llvm.inline_asm` with `=&f,=&f,=&f,=&f,f,f,f,f,f,f,f,f,f,f`.
   (nvvm.inline_ptx cannot express early-clobber; toolchain finding for
   anyone using BasicPtxBuilder with mma.)

## tgemm (the site emitter) and the off-by-one

Warp-tiled: atom tiles (M/16)x(N/8) round-robin over warps (runtime
mi/ni divmod, needs ntiles%warps==0); per-lane 6 fragment loads + 1
gdn.mma.tf32k8 per (tile, kb); acc=True loads the out tile first; neg
negates the B OPERANDS (sgemm semantics -- the acc init stays positive;
dhu.dh_w depends on that distinction); offsets fold into index constants.

Debugging trail (worth remembering): a numpy simulator of the emitted index
math + the verified atom semantics computed A@B exactly, so the EMITTED code
was wrong -- a delta-basis GPU diagnostic (B=I) showed odd-k columns
shifted +1: `%mT2P1` is already 2t+1, and the per-kb offset code added
"+1" again.  Every kb>0 fragment was mis-addressed (also the cause of the
initial wyb illegal-address: reads walked past smem).  After the fix, a
13-combo probe matrix (at/bt/acc/neg/offsets, real bwd shapes) passes at
the tf32 rounding floor.

A second emitter bug of the same family: wy launches 128 threads but tgemm
defaulted to the global THREADS//32=8 warps -- half of wy's tiles were
never written (garbage u/w -> dk=30).  tgemm now takes an explicit warps=
(wy passes 4).

Toolchain note: once a payload carries gdn.* ops, gdn-opt's rewriter
hoists launch-region constants to host args INCLUDING vector constants
(the bf16-mma accumulator zero); gdn_smem_fix.py Fix 2 now byte-substitutes
aggregate constant params (ld.param.b64[+off] -> mov immediates), else the
zero-arg launch fails with CUDA_ERROR_INVALID_VALUE.

## Numerics: per-site tf32 bisect then 6/6 at both shapes

Every one of the 24 f32-GEMM sites was tf32'd ALONE at T512 and graded
(noise = diff vs the round-2 baseline diff; baseline ~1e-5..5e-7):

| site | worst grad moved | verdict |
|------|------------------|---------|
| dhu.dkB | dk 1.04e-2 | OVER 8e-3 bar -> scalar |
| wy.u | dk 1.06e-2 | OVER bar (u noise amplifies via S->Sn) -> scalar |
| wyb.tv | dbeta 8.6e-3 | 2.3x margin -> scalar |
| dhu.dvp_bh | dbeta 3.0e-3 | 6.6x -> scalar |
| dhu.dv_T | dv 1.26e-3 | 6.4x -> scalar |
| dhu.dvp_pdO / dh_qg / dh_w | <=3e-4 clean but PERF-NEGATIVE (below) | scalar |
| wy.w (u's same-shape sibling) | 1.4e-5 | clean -> tf32 |
| dqkwg.* (8 sites) | dk 7.8e-4..5.2e-4, dq 4.9e-4 | tf32 |
| wyb G_dvp/G_dW/X/Y/TW_rb/dk_t1/dk_t2 | dk 6.6e-4..1.1e-4 | tf32 |

Interesting asymmetries: wy.w is clean while wy.u (same shapes, same K) is
poisonous -- u's error rides the state scan into Sn; and tf32 dW/TW
(dA-chain, Kahan-protected in round 2) move NOTHING (dW magnitudes are
small; the round-2 Kahan was protecting against f32 ACCUMULATION noise,
which the mma also does not have).

Final set: 16 sites tf32 (all of dqkwg, 7 of 8 wyb, wy.w).  6/6 PASS:

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T512H4 | 8.0e-4 (10x) | 1.7e-3 (4.6x) | 1.4e-6 | 3.0e-4 (66x) | 1.9e-4 | 1.3e-6 |
| B1T2048H16 | 1.1e-3 (7.5x) | 1.8e-3 (4.5x) | 3.1e-6 | 1.4e-3 (14x) | 4.1e-4 | 1.8e-6 |

(parenthesized = margin under the bar; dv/dh0 back to round-2 exactness
because dhu stayed scalar.)

## Performance: 14.16 -> 11.06 ms (T2048), 2.40 -> 1.62 ms (T512)

| B1T2048H16 | round2 | round3 | delta |
|------------|--------|--------|-------|
| wy | 1.59 | 1.72-1.85 (noisy band) | ~ |
| fwdscan | 1.04 | 1.03 | = |
| dhu | 3.69 | 3.76-3.83 (scalar, unchanged by design) | = |
| dqkwg | 4.71 | 2.75 | 1.71x |
| wyb | 2.65 | 1.21 | 2.19x |
| **e2e bwd** | **14.16** | **11.06 median / 9.57 min** | **1.28x / 1.48x** |

ours/fla = 11.06/1.03 = **0.093x** (was 0.073x).  T512: 2.40 -> 1.62
(1.48x).  dqkwg gains = tf32 mma (8 sites) + warp-parallel r4/c4r (was a
4096-iteration single-thread loop per CTA); wyb gains = tf32 (7 sites) +
warp-parallel ra/c1r.  Both reductions keep single-writer row ownership ->
bit-identical numerics (the verify table is unchanged by them).

## What did NOT work (honest negative results)

* **dhu + tf32 mma = regression.**  The 3 numerics-clean dhu sites
  (dvp_pdO/dh_qg/dh_w), fully unrolled inside the serial reverse chunk
  walk, push ptxas from 128 regs/188B spill to 255 regs/116-128B spill ->
  dhu 3.69 -> 4.67 ms.  All three reverted; dhu is unchanged this round.
* **BV=64 retile does not fit**: dhu at BV=32 already uses 96KB of the
  101376B static limit; BV=64 needs dH[128,64] + dO/dvp/vngb[64,64] +
  tCC[64,64] ~ 150KB+.  The retile needs a buffer-reuse restructure (or
  2-CTA cooperation on the V split), not a tile-size flip.  Also BV=64
  means NV=2 -> 32 CTAs at H16 (fewer, not more); the "more CTAs" premise
  only holds for BV=16.
* Chunk prefetch double-buffer: only ~5KB smem headroom at BV=32 -- one
  extra k-tile buffer would fit, but ping-pong parity plumbing through the
  scf.for reverse walk is a restructure; left documented for round 4.

## What remains for the adjudication round (measured, ranked)

1. **dhu 3.76ms** -- the whole residual is the serial reverse walk
   (32 chunks x ~10 scalar gemms, 64 CTAs latency-bound).  Levers: mma
   without the spill (smaller per-thread tile plan or 384-thread CTA with
   12 warps and 3-4 sites re-enabled), the smem-budgeted BV=64 restructure,
   or staging prefetch.
2. **dqkwg 2.75ms** -- remaining costs are global staging (dvp_all/h_all/
   dh_all slices per ks/vs) and the dkB NV-fold (4 global loads per output
   cell per ks); stage dkB_part once per chunk instead.
3. **wy 1.7-1.85ms** -- the two 32x32 serial forward-substitution bases
   (scf.if %isz) + gc cumsum; warp-parallel blocked solve like fwd.
4. fwdscan 1.03ms -- unchanged since round 2 (not re-examined).

---

# G5-H round 4: dhu split (dkb kernel) + chain-wide async 16B staging

Artifacts: `tools/gdn_bwd_gen.py` (6-kernel chain: wy, fwdscan, dhu, **dkb**
(new), dqkwg, wyb), `tools/gdn_bwd_verify.py` (stage contract updated),
`tools/ab_probe.py` (per-stage A/B diff harness), `tools/probe_dkb.py`,
`tools/probe_pf{,_run}.py` (cp.async micro-probes),
`evidence/G5/bwd4_{verify,bench}{512,2048}.json`,
`evidence/G5/scratch/r4_final_*` (dhu/dkb payloads + ptxas reports).
All measurements on the shared RTX 5090 Laptop (co-job pid 3411456 resident,
~17.9GB, flagged in the jsons); same-window baseline rebuilt from git HEAD
at round start (11.13 ms e2e, per-stage within 2% of the round-3 numbers).

## What landed

1. **dkb split kernel (the dhu restructure).**  The dkB/rowB/s5 partials
   consumed the walk's PRE-update dH -- which the walk stores to
   `dh_all[m]` *before* updating -- so they recompute exactly from globals
   in a new parallel per-chunk kernel (grid B*H*NC, 57.9KB smem, 59 regs).
   dkb also folds the NV reduction dqkwg used to do per output cell (4
   dkB_part global loads per dk cell): it emits CHUNK-SUMMED globals
   (dkB_sum/rowB_c/s5_c) and dqkwg reads one value per cell.  dhu drops
   ~17.5KB smem + 3 GEMM-class passes per walk chunk (96.0 -> 78.3KB, then
   74.2KB after the prefetch restructure below).
   *Debugging trail (probe-caught, worth remembering): the first cut
   rowsummed the RUNNING dkB accumulator per (vt, ks) slice -- rowsum(v0.k)
   + rowsum((v0+v1).k) + ... = 4v0+3v1+2v2+v3, a 1.1e1 dg error localized
   by `tools/probe_dkb.py` comparing rowB_c against (dkB_sum x k).rowsum;
   rowB is now one rowsum over the final summed tile (same mathematical
   value, different f32 association -- dg margin ~50x).*
2. **Chain-wide thread-strided 16B async staging.**  `astage_tile` emits
   MANUAL `nvgpu.device_async_copy` bursts (8 bf16 / 4 f32 per chunk;
   NOT create_async_groups -- G5-F's wait_group-0 lesson) + group + wait
   under caller control; `astage_sync` is the drop-in for same-dtype
   `stage_rows`.  Converted: dhu's P stage + dkb's dHv, dqkwg's P/T/vnew/
   dh_all/dvp_all/h_all/dW_all/q/k stages, wyb's dvp/T/dW stages,
   fwdscan's u/w stages, wy's raw k_t/q_t tiles.  ~4x fewer staging
   instructions per byte.
   *Bug caught by the verify gate: wy launches 128 threads against the
   module's THREADS=256 -- the burst distributes chunks over 256 threads,
   so wy's k_t/q_t came up HALF-STALE (chunks owned by threads 128-255
   never copied).  astage now takes an explicit nthr = the kernel's
   thread count.  The 6/6 gate is what makes this class of silent
   corruption visible; bench-only iteration would have shipped it at
   "1.48ms wy".*
3. **dhu in-walk prefetch (the survivors).**  T restages tCC IN PLACE via
   async right after P's last consumer (dvp_pdO) and is drained before
   dv_T; do prefetches into a bf16 staging tile right after the dO map and
   drains at the body end; dO is emapped from smem instead of global.  The
   separate dvp copy tile is GONE (accCB *is* dvp through loop 2 -- the dh
   gemms accumulate into accKB and never touch accCB; dv_T writes the
   then-free dO and the beta scale maps it in place).  All bit-identical
   (verified against the split build to 0.0 diff).
4. dqkwg reg pressure fell 197 -> 240... (see jsons) with async staging;
   registers now 128/59/240/226 for dhu/dkb/dqkwg/wyb, 0 spill everywhere.

## Performance: 11.13 -> 8.94 ms at B1T2048H16 (1.25x), 1.81 -> 1.28 at T512

Same-window baseline -> round 4 (median CUDA-event, 30 reps):

| B1T2048H16 | base (r4 start) | round4 | delta |
|------------|------|--------|-------|
| wy         | 1.72 | 1.45 | 1.19x |
| fwdscan    | 1.04 | 0.91 | 1.14x |
| dhu        | 3.68 | 2.27 | 1.62x |
| dkb (new)  | -    | 0.96 | - |
| dqkwg      | 2.74 | 1.55 | 1.77x |
| wyb        | 1.21 | 0.95 | 1.27x |
| **e2e bwd**| **11.13** | **8.94 median / 7.76 min** | **1.25x / 1.43x** |

T512: 1.81 -> 1.28 ms (1.41x; same-window baseline; the round-3 recorded
1.62 sits inside shared-GPU noise).  ours/fla bwd at T2048 =
8.94/1.03 = **0.115x** (was 0.093x).  Every per-stage number above is the
median of the saved bwd4_bench jsons; re-run stability 8.94-9.03 ms.

## Numerics: 6/6 PASS at BOTH shapes, byte-identical to round 3

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T512H4 | 8.0e-4 | 1.7e-3 | 1.4e-6 | 3.0e-4 | 1.9e-4 | 1.3e-6 |
| B1T2048H16 | 1.1e-3 | 1.8e-3 | 3.1e-6 | 1.4e-3 | 4.1e-4 | 1.8e-6 |

dq/dk/dv/dbeta/dh0 are the SAME VALUES as round 3 (the split + async
staging + prefetch survivors are value-preserving; verified per-stage with
tools/ab_probe.py 0.0 diffs before benching).  dg moved 4.13e-4 <- 4.12e-4
(rowB re-association only).  dk margin 4.5x, dbeta 14x unchanged.

## What did NOT work (honest negative results)

* **dhu tf32 re-enablement: still negative on BOTH axes after the split
  freed the register budget.**  dhu.dvp_pdO+dh_qg+dh_w together compile
  spill-free at 246 regs (round 3: 255 + 116B spill) -- but (a) 246 regs
  halves occupancy (2 -> 1 CTAs/SM at 256 threads), dhu 2.70 -> 3.05 ms;
  (b) the round-3 per-site-clean margins do NOT compose: combined, dk goes
  to 1.09e-2 > the 8e-3 bar (dh_w's tf32 error rides the dH carry across
  all 32 reverse chunks and dvp_pdO feeds dvp -> dW -> dk).  dhu stays
  scalar; no single site can stay under 128 regs, so the occupancy cliff
  is structural for mma inside this walk.
* **k/q/w slice ping-pong prefetch: async bursts deliver ZERO DATA inside
  the real walk.**  ks/qslc read back all-zero via an @kdbg dump at the
  slice-1 read point (dvp loses the whole Bhat@dH term; dh loses qg^T dO),
  while the IDENTICAL astage_tile bursts into column-offset halves verify
  PERFECTLY in an isolated probe kernel (evidence/G5/scratch/r4_pfprobe:
  src_off/dst_off in {0,32} all bit-exact) and the full-tile bursts
  (dkb/dHv, dhu P/T/do, all of dqkwg/wyb/fwdscan/wy) are bit-identical
  in-chain.  Deterministic, invariant to wait placement (issue-top vs
  issue-end) and to which of k/qw is enabled.  Root cause NOT found this
  round -- an interaction between the walk's live-buffer set and the
  cp.async lowering in this build.  Gates left in the generator
  (GDN_PF_K/GDN_PF_QW, default off) with the probes committed for the
  follow-up.
* The P prefetch variant (async into tCC at the chunk tail, token carried
  across scf.for iterations) was abandoned pre-implementation:
  !nvgpu.device.async.token as an scf.for iter_arg through the NVVM
  lowering is unproven; P stages blocking via astage_sync instead.

## Stage ranking after round 4 (what's genuinely left)

1. **dhu 2.27 + dkb 0.96 = 3.2ms** -- the reverse scan is now the plain
   serial algorithm with fast staging; the next lever is algorithmic
   (chunk-level dH carry parallelization: dH_n = (2^gcl I - w^T Bhat)
   dH_{n+1} + ..., a [K,K] per-chunk carry + tree/look-back scan -- the
   fla-class approach), or the cp.async zero-delivery root-cause to get
   the k/q/w prefetch back.  dkb is instruction-bound on the scalar dkB
   gemm (tf32 over-bar by 4x on dk; split-bf16 3-mma emulation is the
   documented unexplored path).
2. **wy 1.45ms** -- the two 32x32 serial forward-substitution bases + the
   single-thread gc cumsum (the round-4 staging conversion was applied;
   the warp-parallel solve/cumsum restructure is untouched -- it changes
   add order, so it needs its own verify cycle).
3. dqkwg 1.55ms -- tf32'd gemms + async staging; residual is the dvp
   recompute (P^T dO + Bhat dh per chunk) and the 16-slice dW loop.
4. fwdscan 0.91, wyb 0.95 -- no longer ranked bottlenecks.

Repro: `python tools/gdn_bwd_gen.py --shape 1,2048,16,128,128 --out <dir>`
+ `scripts/run_l2.sh` + `tools/gdn_smem_fix.py` per kernel (or
`bash evidence/G5/scratch/r4_build_chain.sh`
 recipe); verify `tools/gdn_bwd_verify.py --gen-root <dir> --shape ...`;
bench `tools/gdn_bwd_bench.py`.

---

# G5-J round 5: the [K,K]-carry reverse scan (dhu retired) + split-bf16 dkB

Artifacts:  (CPU validation, evidence/G5/scratch/
r5_carry_val{512,2048}.json),  (9-kernel chain: wy,
fwdscan, **caryA, caryM, bpre, dhs**, dkb, dqkwg, wyb; dhu kept for A/B),
 /  (the debugging trail),
,
 (same-window round-4 baseline rebuilt from git
8c99f6f), build recipe .  All
measurements on the shared RTX 5090 Laptop (co-job pid 3411456 resident,
flagged in the jsons).

## The math verdict: EXACT (and the scan is hyper-contractive)

The reverse cotangent scan is the affine recurrence per chunk n
    X_{n-1} = A_n X_n + B_n,   X_{NC-1} = dht,   X_n = dh_all[m] (PRE-update)
    A_n = 2^gcl I_K - W^T Bhat          ([K,K], vt-INDEPENDENT)
    B_n = M_n dO_n,  M_n = Qg^T - W^T P^T
-- an algebraic re-association of the sequential update.  Validated
(tools/gdn_carry_validate.py) against the TRUE sequential form (through
dvp, not the composed expression):

| check | value |
|---|---|
| composed vs sequential, fp64 | 4.9e-17 (T512) / 8.3e-17 (T2048) -- 2-3 ulps at |X|~130 |
| 2-level segment scan (zero-seed local walks + suffix products + composed boundary chain), fp64, every S in {2,4,8,16} | same 2-3 ulps |
| fp32 replay floor (GPU operand precisions) | 1.2e-8 (gate distribution) / 2.4e-6 (near-unitary-gate stress, ||A||~8.8) |
| conditioning, gate distribution | rho(A_n) <= 2.3e-18, ||A_n|| <= 2.2e-17 |
| anchor vs the real bwd_twin (dh0, bh0) | 1.0e-10 / 7.6e-10 |

So the carry math is exact for ANY gate distribution, AND for the
benchmark distribution (-softplus(randn)) the per-chunk operators are
hyper-contractive: cross-chunk coupling dies within one chunk step
(||A||^2 ~ 1e-34, far under fp32 eps).  A depth-truncated fully-parallel
scan would be exact-in-fp32 HERE but wrong for weak gates -- not shipped;
the exact 1-level chain is what runs.

## What landed

1. **caryA/caryM/bpre** (grid B*H*NC, all parallel): precompose A_all
   [M,K,K], M_all [M,K,C], B_all [M,K,V] per chunk.  64.5-82KB smem, 40-48
   regs.  A enters consumers straight from GLOBAL inside their gemms
   (warp-broadcast row loads -- a [K,K] f32 staging tile would cost 64KB
   for nothing).
2. **dhs** (grid B*H*NV): the serial chain reduced to ONE gemm per chunk
   (Xa = B-slice; Xa += A_all[m] @ X; X = Xa) -- 32KB smem total, no
   P/T/k/q/w/beta staging, no cumsum, no dvp/dv epilogue in the walk.
   The gemm is : register accumulators, kk-outer reduction, CPC=16
   cells/thread all sharing one b (smem) load per kk with 16 INDEPENDENT
   global A loads to overlap (the naive cell-major  serialized the
   LDG latency: dhs 0.83 -> 0.59ms from this alone).
3. **dv moved to wyb** (dv = tv .* beta off wybs

---

# G5-J round 5: the [K,K]-carry reverse scan (dhu retired) + split-bf16 dkB

Artifacts: `tools/gdn_carry_validate.py` (CPU validation, evidence/G5/scratch/
r5_carry_val{512,2048}.json), `tools/gdn_bwd_gen.py` (9-kernel chain: wy,
fwdscan, **caryA, caryM, bpre, dhs**, dkb, dqkwg, wyb; dhu kept for A/B),
`tools/probe_r5inter.py` / `tools/probe_dbeta.py` (the debugging trail),
`evidence/G5/bwd5_{verify,bench}{512,2048}.json`,
`bwd5_r4_samebench2048.json` (same-window round-4 baseline rebuilt from git
8c99f6f), build recipe `evidence/G5/scratch/r5_build_chain.sh`.  All
measurements on the shared RTX 5090 Laptop (co-job pid 3411456 resident,
flagged in the jsons).

## The math verdict: EXACT (and the scan is hyper-contractive)

The reverse cotangent scan is the affine recurrence per chunk n

    X_{n-1} = A_n X_n + B_n,   X_{NC-1} = dht,   X_n = dh_all[m] (PRE-update)
    A_n = 2^gcl I_K - W^T Bhat          ([K,K], vt-INDEPENDENT)
    B_n = M_n dO_n,  M_n = Qg^T - W^T P^T

-- an algebraic re-association of the sequential update.  Validated
(tools/gdn_carry_validate.py) against the TRUE sequential form (through
dvp, not the composed expression):

| check | value |
|---|---|
| composed vs sequential, fp64 | 4.9e-17 (T512) / 8.3e-17 (T2048) -- 2-3 ulps at \|X\|~130 |
| 2-level segment scan (zero-seed local walks + suffix products + composed boundary chain), fp64, every S in {2,4,8,16} | same 2-3 ulps |
| fp32 replay floor (GPU operand precisions) | 1.2e-8 (gate distribution) / 2.4e-6 (near-unitary-gate stress, \|\|A\|\|~8.8) |
| conditioning, gate distribution | rho(A_n) <= 2.3e-18, \|\|A_n\|\| <= 2.2e-17 |
| anchor vs the real bwd_twin (dh0, bh0) | 1.0e-10 / 7.6e-10 |

So the carry math is exact for ANY gate distribution, AND for the
benchmark distribution (-softplus(randn)) the per-chunk operators are
hyper-contractive: cross-chunk coupling dies within one chunk step
(\|\|A\|\|^2 ~ 1e-34, far under fp32 eps).  A depth-truncated
fully-parallel scan would be exact-in-fp32 HERE but wrong for weak gates
-- not shipped; the exact 1-level chain is what runs.

## What landed

1. **caryA/caryM/bpre** (grid B*H*NC, all parallel): precompose A_all
   [M,K,K], M_all [M,K,C], B_all [M,K,V] per chunk.  64.5-82KB smem, 40-48
   regs.  A enters consumers straight from GLOBAL inside their gemms
   (warp-broadcast row loads -- a [K,K] f32 staging tile would cost 64KB
   for nothing).
2. **dhs** (grid B*H*NV): the serial chain reduced to ONE gemm per chunk
   (Xa = B-slice; Xa += A_all[m] @ X; X = Xa) -- 32KB smem total, no
   P/T/k/q/w/beta staging, no cumsum, no dvp/dv epilogue in the walk.
   The gemm is `rgemm`: register accumulators, kk-outer reduction, CPC=16
   cells/thread all sharing one b (smem) load per kk with 16 INDEPENDENT
   global A loads to overlap (the naive cell-major `aggemm` serialized the
   LDG latency: dhs 0.83 -> 0.59ms from this alone).
3. **dv moved to wyb** (dv = tv .* beta off wyb's existing tv = T^T dvp
   gemm -- zero new gemms) and dvp_all's real producer is now dqkwg (it
   always recomputed it identically; see the bug below for why that
   mattered).
4. **dkb split-bf16 3-mma** (lever 1): dkB = vng @ dh^T per vt via
   x = hi + lo bf16 splits (3 mma_gemm rounds, vt-folded in the mma
   accumulators, +16B-padded bf16 split tiles per the G5-F LDSM lesson).
   dk UNCHANGED (1.730e-3 vs 1.731e-3 scalar -- the ~2^-16 product error
   is invisible; tf32 was 1.04e-2 = over-bar in round 3).  dkb 0.96 ->
   0.25ms (3.8x).

## The latent bf16-Bhat bug (found by the restructure; paper material)

Round-5's first verify moved dbeta 3.0e-4 -> 1.33e-2 (1.5x under the bar).
Bisect trail (probe_r5inter -> probe_dbeta -> payload diff -> numpy
bf16-emulation): dqkwg's dvp recompute had staged Bhat into the **bf16
tCKb** tile since round 2 (2e-3 relative; visible only at the seed chunk
where \|Bhat dh\| ~ O(1): GPU dvp == numpy(bf16 Bhat) to 3.3e-7).  It was
invisible because run_chain re-stages every input global H2D before each
launch: wyb always received **dhu's** clean dvp_all, and dqkwg's overwrite
never reached a verifier (the round-3/4 verify tables graded the clean
path; the BENCH runs used the poisoned one).  With dhu retired, dvp_all's
only producer is dqkwg and the poison surfaced.  Fix: the bhat tile is the
f32 tCVb (dead in that loop); probes committed.  Corollary: the verify
harness's H2D re-staging can mask producer/consumer bugs -- the round-5
contract (dvp_all read back from dqkwg) closes this one.

Related finding: tf32 on dvp_bh makes dbeta/dv CHAOTIC (1.5e-5 -> 2.9e-3
-> 5.8e-3 across three numerically equivalent builds) because any
f32-level dh perturbation resamples the mma's operand quantization bins.
dvp_bh stays scalar (dbeta stable at 1.4e-3-class); dvp_pdO (no dh
operand) is tf32.  Split-bf16 (the dkb machinery) is the documented path
to mma speed at scalar accuracy there -- its out tile [C,BV=32] is too
narrow for 8-warp mma_gemm as-is.

## Performance: 8.98 -> 8.15 ms same-window (1.10x), 1.28 -> 1.16 at T512

Same-window A/B (round-4 chain rebuilt from git 8c99f6f in this window:
8.98 median / 7.78 min, per-stage within 2% of round 4):

| B1T2048H16 | r4 (same window) | round5 | delta |
|---|---|---|---|
| wy | 1.45 | 1.45 | = |
| fwdscan | 0.92 | 0.91 | = |
| dhu -> caryA+caryM+bpre+dhs | 2.17 | 0.41+0.46+0.29+0.59 = 1.76 | 1.23x |
| dkb | 0.96 | 0.25 | 3.8x |
| dqkwg | 1.55 | 1.71 | 0.91x (scalar dvp_bh) |
| wyb | 0.96 | 1.05 | 0.91x (dv fold) |
| **e2e bwd** | **8.98 / 7.78** | **8.15 / 7.19** | **1.10x / 1.08x** |

scan complex (dhu+dkb = 3.13) -> (cary*+dhs+dkb = 2.01): **1.56x**.
T512: 1.28 -> 1.16 ms.  ours/fla bwd = 1.03/8.15 = **0.126x** (was
0.115x).  9 kernel launches vs 6.

## Numerics: 6/6 PASS at BOTH shapes (round-4 value class, robust)

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T512H4 | 8.0e-4 (10x) | 1.7e-3 (4.6x) | 6.9e-5 | 6.9e-4 (29x) | 1.9e-4 | 1.3e-6 |
| B1T2048H16 | 1.1e-3 (7.5x) | 1.8e-3 (4.5x) | 9.6e-5 | 1.4e-3 (14x) | 4.2e-4 | 1.8e-6 |

dh_all (the carry chain) reproduces the sequential walk's values to
2.2e-8 on GPU (probe_r5inter; the CPU-predicted re-association floor).

## What did NOT happen (honest)

* **Segment/tree scan not shipped.**  The validated 2-level machinery
  (segment-local walks + composed boundary correction) was implemented and
  verified on CPU, but on GPU the 1-level chain is already
  latency-light (dhs 0.59ms; rgemm closed most of the gap) while the
  segment version needs [K,K]x[K,K] composes (2M MACs each, scalar = the
  new bottleneck; tf32/bf16 composes re-open the chaos/precision
  questions) + 3 more kernels of launch overhead for a remaining ~0.3ms
  ceiling.  Measured negative economics, not a math failure.
* **Lever 2 (async zero-delivery bisect) not attempted this round**: the
  mystery lived in dhu's k/q/w slice ping-pong -- dhu is retired from the
  default chain, so the suspected code path no longer runs.  The new dhs
  does a synchronous B-slice stage per chunk; an async B ping-pong there
  is the successor idea (its own risk profile), left with the r4 probes.

## Ranking after round 5

1. dqkwg 1.71ms -- dvp_bh scalar (split-bf16 needs a <=4-warp or wider-tile
   mma plan), dW/h_all staging, 16-slice dW loop.
2. wy 1.45ms -- the serial 32x32 solve bases + gc cumsum (unchanged since
   round 3).
3. wyb 1.05ms / fwdscan 0.91ms.
4. The scan complex is 1.76ms with dhs at 0.59 -- the serial floor of 32
   x 1 rgemm step; the segment scan is the next algorithmic lever if the
   composes can be made cheap+exact (split-bf16 compose is the candidate).

Repro: `python tools/gdn_bwd_gen.py --shape ... --out DIR` +
`bash evidence/G5/scratch/r5_build_chain.sh DIR`; verify
`tools/gdn_bwd_verify.py --gen-root DIR --shape ...`; bench
`tools/gdn_bwd_bench.py`; GDN_DKB_SPLIT=0 restores the scalar dkB.

---

# G5-L round 6: wy column-parallel solve + dvp_bh split-bf16 (2-mma) + the cp.async->ldmatrix hazard

Artifacts: tools/gdn_bwd_gen.py (wy thread-per-column solve bases;
dqkwg dvp_bh split-bf16 with the row scale factored into the epilogue),
evidence/G5/scratch/bwd6_{verify,bench}{512,2048}.json +
bwd6_bench2048_rerun.json (stability) + bwd6_r5samebench512.json
(same-window round-5 T512 baseline rebuilt from git HEAD), gen dirs
scratch/r6_T{512,2048}, build recipe scratch/r6_build_chain.sh, and
the hazard probes scratch/r6_ldgsts_probe_fail /
r6_ldgsts_probe_stshared_pass.  All measurements on the shared RTX 5090
Laptop (co-job pid 3411456 resident; same-window baselines rebuilt from
git HEAD at round start: 8.21 ms median / 7.20 min @T2048, 1.139 ms
@T512).

## What landed

1. **wy thread-per-column solve (lever 2).**  The two 32x32
   forward-substitution bases ran as ONE thread's row-serial walk
   (~11K dependent FMAs on the CTA critical path; C=64 -> two 32x32
   bases + the T21 combine).  A unit-lower-triangular solve's column j
   depends only on column j (T[i,j] = -sum_{j<=k<i} A[i,k] T[k,j] reads
   its own column exclusively), so 64 threads now solve all columns of
   both bases in parallel with NO internal barriers.  The per-column
   k-ascending f32 chains are the same sums in the same order as the
   serial walk -> T is BIT-IDENTICAL (all six grad values unchanged).
   The never-written strict upper triangle (incl. the T12 block), which
   the old code left to smem-alloc-as-zero, is now explicitly zeroed
   (same values, no reliance on alloc state).  wy 1.45 -> 0.70 ms
   (2.06x); at T512 0.199 -> 0.095 ms.
2. **dqkwg dvp_bh split-bf16, 2-mma form (lever 1).  The row scale
   factors out**: sum_k (k[i,kk] 2^(gcl-gc[i])) dh[kk,j] =
   2^(gcl-gc[i]) sum_k k[i,kk] dh[kk,j] -- so the mma A operand is the
   RAW bf16 k slice (element-wise staged, no split, no Bhat emap) and
   the scale lands in the epilogue.  Only the dh^T side needs the hi/lo
   split (fused single sweep), giving TWO mma per (vs,ks) site instead
   of three.  The "<=4-warp mma plan": N=BV=32 admits W=4; warps 4-7
   skip the guarded scf.if region while warps 0-3 own the [C,BV]
   fragment tiles; all staging stays all-thread; each site's 2-mma
   chain + epilogue live in one region (region-local accumulator SSAs,
   accCB accumulates through smem keeping the scalar chain's
   (((pdO+bh0)+bh1)+...) association).  smem 99840 -> 97792 B (tTf
   retired; TW restages into tCC), 234 -> 214 regs, 0 spill.
   dqkwg 1.71 -> 1.66 ms.  Two intermediate designs were measured and
   reverted: the direct 3-mma port (v1: stage+emap+double-split per
   site, 2.19 ms) and the transposed-tile dW/dq1 staging (1.93 ms).
3. **dhs prefetch: measured negative economics, not shipped.**  The B
   slice stage is ~130 instr/thread against rgemm's ~4.2K ops/thread
   per chunk (<5%); dhs is dominated by rgemm's global A-read latency
   (2048 warp-broadcast A-loads per thread per chunk), which a B
   ping-pong cannot hide.  The remaining dhs lever is A-side
   (smem-staged/compressed A_all), out of round scope.

## The cp.async -> ldstmatrix silent-corruption hazard (lever 3 root-cause artifact; paper material)

Staging the mma A tile (akb [C, BK+8] bf16) with the round-4 MANUAL
nvgpu.device_async_copy burst (SASS: LDGSTS.E.BYPASS.128) delivered
data that GENERIC ld.shared reads back CORRECTLY (bit-exact vs the
global source, dump-verified) but nvgpu.ldmatrix reads WRONG for a
build-schedule-dependent subset of the 16-row A blocks -- rows 32-63 in
the 2-mma builds, rows 48-63 in the 3-mma build.  Deterministic per
build; invariant to a value-preserving refresh pass; the 6/6 gate
caught it as dk 1.34 / dv 2.18 / dbeta 25.7 while dq/dh0
(dvp-independent) stayed at baseline values.  A lane-true fragment
dump (every lane stores its own D cell) localized the wrongness to
whole mi blocks; the identical tile written by plain st.shared (the
ASPLIT bisect arm, scratch/r6_ldgsts_probe_stshared_pass) is bit-exact
vs numpy (5.2e-8) through the SAME ldmatrix/mma/epilogue machinery.
Every mma operand tile elsewhere in the chain is st.shared-written (the
split tiles), so dkb/wy never trip it.  Fix shipped: the element-wise
bf16 raw copy (~8 elems/thread, no conversion).  This is a second
documented silent-failure mode of the manual-cp.async path on this
sm_120a toolchain (the r4 kdbg zero-reads were the first); both live in
the same instruction family with deferred waits.  Probes:
scratch/r6_ldgsts_probe_fail (cp.async-written A, fails),
..._stshared_pass (st.shared A, passes), debug gated by GDN_MMA_DBG=1.

## Performance: 8.21 -> 7.27 ms same-window at B1T2048H16 (1.13x), 1.139 -> 1.067 at T512

| B1T2048H16 | r5 (same window) | round6 | delta |
|---|---|---|---|
| wy | 1.448 | 0.705 | 2.06x |
| fwdscan | 0.913 | 0.917 | = |
| caryA+caryM+bpre | 1.14 | 1.21 | = (noise) |
| dhs | 0.588 | 0.588 | = |
| dkb | 0.244 | 0.250 | = |
| dqkwg | 1.712 | 1.663 | 1.03x |
| wyb | 1.040 | 1.041 | = |
| **e2e bwd** | **8.21 / 7.20** | **7.27 / 6.41 (rerun 7.36/6.42)** | **1.13x med / 1.12x min** |

T512: 1.139 -> 1.067 ms (1.07x; wy 2.1x, dqkwg 1.08x).  ours/fla bwd =
1.03/7.27 = **0.142x** (was 0.126x).  Same 9-kernel chain, same
launch count.

## Numerics: 6/6 PASS at BOTH shapes, round-5 value class

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T512H4 | 8.0e-4 (10x) | 1.7e-3 (4.6x) | 6.9e-5 | 6.9e-4 (29x) | 1.9e-4 | 1.3e-6 |
| B1T2048H16 | 1.1e-3 (7.5x) | 1.8e-3 (4.5x) | 9.6e-5 | 1.4e-3 (14x) | 4.2e-4 | 1.8e-6 |

The wy solve is bit-identical (identical grad values to round 5 at
T512; T2048 differs only through dvp: dbeta 1.413e-3 -> 1.407e-3, the
split-bf16 signature; dk/dq/dv/dg/dh0 unchanged to 4 digits).  All
margins >= 2x (tightest: dk 4.5x).

## Ranking after round 6 (what remains)

1. fwdscan 0.92 + wyb 1.04 -- now the biggest non-scan-complex stages;
   both are staging/gemm-mix kernels with no un-attempted structural
   lever this round.
2. dqkwg 1.66 -- residual: 16 element-wise operand stages per dvp site
   (akb + transposed dh + fused split), the r4/c4r/c1r serial column
   sums, the 16-slice dW/dq1 loops.
3. dhs 0.59 -- A-side latency (the rgemm global A-read chain); the
   documented next lever is an smem/compressed A_all restructure.
4. wy 0.70 -- the tail (kkt/P mma + u/w scalar gemms + stores); wy.u is
   tf32-barred (dk 1.06e-2 over-bar), split-bf16 is the candidate.

Repro: python tools/gdn_bwd_gen.py --shape ... --out DIR +
bash evidence/G5/scratch/r6_build_chain.sh DIR; verify
tools/gdn_bwd_verify.py --gen-root DIR --shape ...; bench
tools/gdn_bwd_bench.py.  Kill switches: GDN_WY_PAR=0 restores the
serial solve; GDN_DVP_SPLIT=0 restores scalar dvp_bh; GDN_V2_ASPLIT=1
is the st.shared-A bisect arm; GDN_MMA_DBG=1 adds the lane-true
fragment/tile dumps.


---

# G5-M round 7: the fwdscan F2 port + wy staging reuse

Artifacts: tools/gdn_bwd_gen.py (gen_fwdscan_v3 pipeline body +
gen_fwdscan_v2 fallback, scan tf32 group, wyb tKK/tVV staging reuse +
stage_raw_ew helper + GDN_TV_KAHAN knob), evidence/G5/scratch/
r7_{base,final,pp,ppb,pp16,wybs,wybtv,tf}_T* gen dirs (the bisect
trail), bwd7_{verify,bench}{512,2048}.json + bwd7_r6samebench2048.json,
gen dirs r7_T{512,2048}, build recipe r7_build_chain.sh.  All
measurements on the shared RTX 5090 Laptop (co-job pid 3411456 resident;
same-window baselines rebuilt from git HEAD 455eef7 at round start:
7.379/6.408 ms median/min @T2048, fwdscan 0.916, wyb 1.041 -- within
noise of the round-6 table).

## The fwdscan-twin gap analysis (why 916us when the fwd F2 twin runs 83us)

fwdscan recomputes the SAME recurrence as the forward F2 kernel (the
per-chunk S update + h_all/vnew stores).  Round 6 left it at 910us vs
F2 83us (11x).  Ranked causes, each measured or ncu-verified:

1. **The two sgemms were scalar element-per-thread loops** (2 smem
   loads + ~5-6 issued instr per MAC): tf32-mma-ing them alone moved
   fwdscan 0.970 -> 0.789 (solo, same window) -- i.e. the gemms were
   ~180us, NOT the ~830us a naive instruction-count reading suggests.
2. **Pure latency execution**: ncu on the tf32 stage-1 build: 88.3%
   no-eligible cycles, L1TEX-scoreboard 43.9% of the 17.0 cyc/inst
   stall average, 2 warps/scheduler (64 CTAs, 1/SM, 192 regs), 0.39
   waves, DRAM 10.7%, nothing saturated.  The killers: a SERIAL
   per-chunk gc cumsum (64 dependent global-load iterations, EVERY
   chunk), synchronous astage_sync staging (issue + immediate drain
   = zero overlap), 9 barriers/chunk, and 5 separate stage/emap/store
   passes per chunk.
3. **f32 payload** (the honest 2x, not the 11x): h_all+vnew writes
   50MB + w/u/k reads 84MB @T2048 vs F2 bf16 ~67MB total.  At the
   measured ~800GB/s effective this is a ~170us floor; the kernel sat
   5x above it for reason 2, not bandwidth.

## What landed (fwdscan; solo same-window numbers, paired e2e below)

| change | fwdscan solo (ms) |
|---|---|
| round-6 HEAD | 0.920 |
| + gemm dispatch (tags fwdscan.wS/kg) + fused u-read/vn/vnew sweep + fused k-scale sweep | 0.970 (scalar) / 0.789 (tf32) |
| + ALL-chunk gc cumsum hoisted to a prologue (thread t<NC walks chunk t serially -- the SAME serial-order f32 chain, gc values bit-identical) + w/k/u ping-pong prefetch groups (the F2 group-A pattern; scalar-ld.shared consumers only, per the r6 LDGSTS->LDSM rule) + gemm-b collapsed to one full-K tgemm + 5 barriers/chunk | 0.398 |
| + per-chunk exp2 hoist (expc[i] = 2^(gcl-gc_i); was one MUFU per kg element) + hoisted decay scale | **0.381** (bench per-stage 0.374) |

chain @T2048 (paired, interleaved with the HEAD baseline): fwdscan
0.916 -> 0.374 (**2.45x**); @T512 (vs the untouched r6_T512 chain)
0.203 -> 0.097 (**2.11x**).  smem 65792 -> 98560B, 202 regs, 0 spill,
same 64-CTA grid, tf32 on both recurrence gemms (numerics below).

## What did NOT land (honest negatives, fwdscan)

* **Cross-chunk w prefetch (v3b)**: the wait lowering is a BARE
  cp.async.wait_group N (source-verified in-tree; the group operand is
  unused by the lowering), so waits-by-count work without tokens
  crossing the scf.for boundary -- and the design measured 0.358 solo
  (40us better).  REVERTED: the prefetch destination (the single wpp
  tile) is NOT dead when the copy lands -- it races the same chunk
  kg-sweep writes and gemm-b reads (verify: dk 31, dg 2.1 FAIL).  A
  correct version needs a third [C,K] buffer, which does not fit the
  99KB smem budget next to S/wpp/kpp/utl/vn/acc/gsc.  Documented as
  the next fwdscan lever together with the bf16-ldsm operand path.
* **BV=16** (128 CTAs, fills all 82 SMs): 0.523 vs 0.398 -- w/k
  staging traffic doubles (every vt-CTA restages the same tiles) and
  the second wave half-fills.  BV=32 stays.
* **bf16 mma + ldmatrix operand tiles** (the exact F2 gemm form):
  would cut MIO pressure ~10x (ncu post-v3: mio_throttle 2.88 cyc/inst
  is the top stall) but changes the h_all/vnew value class; the tf32
  tgemm path passed at the round-6 value class, so that risk was not
  spent this round.  Ranked round-8 candidate.
* 128-bit stores: no such lowering path exists (the G5-K re-probe
  stands).

## wy staging reuse (the wyb half)

ncu on wyb: long_scoreboard 2.76 cyc/inst (global loads) + mio 2.19 +
wait 2.39; 512 CTAs, 92KB smem (2 CTAs/SM by smem).  Landed: raw k/v
bf16 slice tiles (tKK/tVV) staged once per slice and reused by every
k/v-reading sweep (f_up / f_kb2 / term_rb / f_bk / the dk raw-k copy /
term_rc -- all were per-element GLOBAL loads); exp2(gc_i) hoisted per
chunk (expg); wyb.tv kahan OFF (its K=64 reduction has no cancellation
history; dv moved 9.592e-5 -> 9.591e-5, everything else to 4 digits).
The FIRST build FAILED verify (dk 1.466, dbeta 3.5e-2): the rb/dk
loops read tKK at ks-slices it was not restaged for -- the G_dW loop
had left only its LAST slice resident.  Fixed by restaging tKK per ks
in both loops; verify caught it, no masking.  wyb 1.041 -> 0.988
@T2048 (1.05x); 0.1274 -> 0.1253 @T512 (1.02x -- only 128 CTAs there,
the loads were already latency-hidden).

* **wyb.tv tf32 re-tested and REJECTED**: in today chain it is 6/6
  PASS but dbeta lands at 9.3e-3 = 2.15x margin (r3 had measured the
  same site at 8.6e-3 / 2.3x and kept it scalar) and the build spills
  (255 regs, 16B frame).  The mma path at scalar accuracy is the
  3-mma split-bf16 form; it needs TRANSPOSED operands (T^T and dvp^T
  tiles in the mma B orientation -- stage_ht-style transposing sweeps,
  or transposed-global contracts written by wy/dqkwg).  Designed this
  round, not built; the measured prize is ~158us (wyb 0.99 -> 0.85 at
  tf32-tv speed).

## Performance: 7.36 -> 6.73 ms same-window @B1T2048H16 (1.09x med / 1.10x min); 1.092 -> 0.918 @T512 (1.19x med)

| B1T2048H16 | HEAD baseline (this window) | round7 | delta |
|---|---|---|---|
| wy | 0.705 | 0.705 | = |
| fwdscan | 0.916 | 0.374 | 2.45x |
| caryA+caryM+bpre | 1.21 | 1.15 | = (noise) |
| dhs | 0.591 | 0.594 | = |
| dkb | 0.248 | 0.250 | = |
| dqkwg | 1.666 | 1.669 | = |
| wyb | 1.041 | 0.987 | 1.05x |
| **e2e bwd** | **7.36 / 6.41** | **6.73 / 5.81** | **1.09x med / 1.10x min** |

(The round-6 report carried 7.27/6.41; today same-window HEAD
re-measure is 7.31-7.44 median across 6 interleaved runs -- co-job
noise; ratios are same-window paired.)  T512: 1.092/0.977 ->
0.918/0.826 (1.19x med / 1.18x min).  ours/fla bwd = 1.03/6.73 =
**0.153x** (was 0.142x).  Same 9-kernel chain, same launch count.

## Numerics: 6/6 PASS at BOTH shapes; round-6 value class kept

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T512H4 | 8.0e-4 (10x) | 1.7e-3 (4.6x) | 6.9e-5 (116x) | 6.9e-4 (29x) | 1.9e-4 (104x) | 1.3e-6 |
| B1T2048H16 | 1.1e-3 (7.5x) | 1.8e-3 (4.5x) | 9.6e-5 (83x) | 1.4e-3 (14x) | 4.2e-4 (48x) | 1.8e-6 |

The fwdscan tf32 sites (fwdscan.wS / fwdscan.kg) perturb h_all by
8.0e-4 rel and vnew by 5.5e-5 rel (payload dump A/B at T512) ->
final-grad shifts <= 3.0e-4 (dq 2.99e-4, dk 2.47e-4, dbeta 2.7e-5)
that absorb into the EXISTING max-error cells: the table is unchanged
to 4 digits from round 6 (dk 1.791e-3 vs 1.8e-3).  The prologue
cumsum is bit-identical by construction; the gemm-b single-call
re-association also lands under the printed digits.  The sites are
bisect-vetted into TF32_DEFAULT (group "scan") from this round.
Kill switches: GDN_SCAN_PP=0 (stage-1 body), --tf32 without the scan
group, GDN_TV_KAHAN=1.

## Ranking after round 7 (round-8 scope)

1. **dqkwg 1.67** -- unchanged; the 16 element-wise operand stages per
   dvp site, r4/c4r column sums, the 16-slice dW/dq1 loops.
2. **wyb 0.99** -- the tv 3-mma split-bf16 port (transposed-operand
   staging; ~158us measured prize) + X/Y-chain instruction weight.
3. **caryA+caryM+bpre 1.15** (was 1.21) -- three parallel per-chunk
   precompose kernels; gemm/stage restructure candidates.
4. **dhs 0.59** -- A-side latency (the rgemm global A-read chain);
   smem ping-pong A tiles.
5. **fwdscan 0.37 remainder** -- ncu: mio_throttle top stall at 2
   warps/scheduler; the bf16-ldsm operand path (10x fewer MIO ops;
   needs split-bf16 machinery + transposed tiles) or a 2-CTA
   residency (regs <= 128) close toward the ~170us traffic floor.
6. **wy 0.70 tail / dkb 0.25**.

Repro: python tools/gdn_bwd_gen.py --shape ... --out DIR + bash
evidence/G5/scratch/r7_build_chain.sh DIR; verify tools/gdn_bwd_verify.py
--gen-root DIR --shape ...; bench tools/gdn_bwd_bench.py.


# G5-O round 8: dqkwg st.shared-ldsm operands + warp sums; carry mma; explicit FMA

Artifacts: tools/gdn_bwd_gen.py (dqkwg v3 + mma_gemm_mh 2D warp map +
mma_chain_if + stage_raw_bf16/stage_split_ew/stage_raw_T/stage_split_T +
warp_rowsum/warp_colsum + fma3; bpre split-bf16 mma; cary tf32 group; sgemm
math.fma), evidence/G5/scratch/r8_{base,new}_T{512,2048} gen dirs +
r8_build_chain.sh + r8b_* bisect dirs (the verify-caught bug trail) +
r8b_{base,new}{512,2048}_*.json paired benches, bwd8_verify{512,2048}.json.
All measurements on the shared RTX 5090 Laptop (co-job pid 3411456 resident
~19h; same-window baselines rebuilt from git HEAD 35bdd32 at round start:
6.65-6.71/5.80 ms median/min @T2048 -- within noise of the round-7 table's
6.73).  Probes this round (CPU+GPU-verified): gpu.shuffle xor ->
shfl.sync.bfly.b32, math.fma -> fma.rn.f32 (the static-r7 "zero fma.rn.f32
in every bwd kernel" finding is fixable with EXPLICIT emission -- the
toolchain never contracts arith mulf/addf, and fastmath attrs do not parse).

## Performance: 6.71 -> 5.97 ms same-window @B1T2048H16 (1.125x med / 1.084x min); T512 1.00-1.18x med / 1.117x min

| B1T2048H16 | HEAD baseline (this window) | round8 | delta |
|---|---|---|---|
| wy | 0.703 | 0.700 | = |
| fwdscan | 0.375 | 0.374 | = |
| caryA | 0.434 | 0.312 | **1.39x** (tf32) |
| caryM | 0.460 | 0.161 | **2.86x** (tf32) |
| bpre | 0.296 | 0.062 | **4.79x** (5.9x in the cleanest window; split-bf16 mma + 2 CTA/SM) |
| dhs | 0.592 | 0.592 | = |
| dkb | 0.246 | 0.246 | = |
| dqkwg | 1.665 | 1.773 | **0.94x REGRESSION** (see below -- kept ON: e2e with v3 5.97 vs ~6.09 with the old dqkwg and all other levers) |
| wyb | 0.987 | 0.985 | = |
| **e2e bwd** | **6.714 / 5.800** | **5.969 / 5.349** | **1.125x med / 1.084x min** |

ours/fla bwd = 1.03/5.97 = **0.173x** (was 0.153x).  T512 (final window):
0.850/0.818 -> 0.846/0.732 (1.004x med -- the small-shape median is
launch-dominated and window-noisy; an earlier same-day window measured
1.175x med; 1.117x min is stable).  Same 9-kernel chain, same launch count.

## What landed

* **bpre split-bf16 mma (4.8-5.9x)**: A = M-block hi/lo fused splits
  (f32 source, ~2^-16 products), B = dO^T RAW bf16 tile (bf16 global ->
  bf16 tile: exact, no split), scale in the epilogue, register-chain over
  the whole K=C reduction, B_all stored global-direct.  The [BK2,V] f32 bb
  staging tile and its scale sweep are gone; smem 49152 -> 36864 B (2
  CTAs/SM by smem instead of 1); regs 40 -> 54; SASS: 0 HMMA -> 128, LDS
  160 -> 0, LDSM 0 -> 80.
* **caryA/caryM tf32 (1.39x / 2.86x)**: the [K,K]/[K,C] carry composes
  were the scalar thread-partitioned sgemm loops (static r7: zero HMMA
  across the no-tensor-core five).  One-line site-group additions
  ("cary"): both read their operands naturally (at/bt addressings are
  smem-natural for these shapes).  caryA sits at 255 regs / 0 spill and
  82176 B smem (1 CTA/SM) -- the register/smem pass is its round-9 lever;
  caryM lands at 160 regs.
* **dqkwg v3 -- the designed round-8 lever (0.94x wall, LDS storm killed)**:
  static r7 measured 3405 LDS / 832 HMMA (5.1 lds/mma).  Now: **LDS 4440
  -> 1378 SASS (-69%)**, EX2 149 -> 4, HMMA 832 -> 1280, LDSM 192 -> 896,
  248 regs, 0 spill, smem 97792 -> 99328 B.  Structure (all per the r6
  hazard rule -- every ldmatrix-fed tile is st.shared-written, the
  LDGSTS tiles tCKa/tCKb feed only tf32 generic loads):
  - dPr: split-bf16 mma, A = RAW dO slice (exact bf16), B = vnew hi/lo
    fused splits, register-chain across all 4 vs slices, scale in the
    epilogue (W=8, unguarded).
  - dvp: pdO FOLDED into the bh chain -- pt = (P^T .* scale .* rie) hi/lo
    with rie = 2^(gc_i - gcl), so the single epilogue factor
    egc[row] = 2^(gcl - gc_i) recovers pdO (egc*rie*scale = scale to
    2^-24) while the bh part keeps its exact r6 form; bdt = dO^T raw
    (pdO's B, exact); per-vs chain = 2 K=64 pdO links + 4 ks x 2 bh
    links; ONE epilogue stores the dvp_all slice global-direct (the
    accCB tile + store_rows sweep retired).
  - dW + dq1 MERGED (ks,vs) loops sharing one Sn hi/lo B-tile set per
    (ks,vs) (round 7 staged the same h_all slice TWICE); dW A = dvp hi/lo
    fused splits, epilogue negates + stores dW_all direct; dq1 A = raw dO
    with the (scale * 2^gc) row scale in the epilogue.
  - TW/dk_dPmq/dq_dPmk stay tf32 (their [C,C]-class operand splits do
    not fit next to the mma tiles in the 101 KB budget; 192/576 static
    tf32 links remain); the TW row-scale sweep reads the hoisted bse =
    beta * 2^gc vector (no per-element MUFU).
  - r4/c4r (prod-in-place pass into tCC, then warp-butterfly row/col
    sums) and r3 (over accCK x q-global, coalesced per lane): the serial
    one-thread-per-row C-chains are gone (gpu.shuffle xor -> 5-step bfly
    trees, 2-deep per row).
  - exp2 hoists: egc/rie/seg/bse [C] vectors replace per-element MUFU
    sweeps in f_do/f_dog/the TW scale sweep (SASS EX2 149 -> 4).
* **mma_gemm_mh (the 2D warp map)**: the strip map caps W at N/8 = 4 for
  N=32, so the first v3 cut ran every dvp/dW/dq1 mma inside a %wgv4
  guard -- HALF the SM idle; ncu showed 89% no-eligible cycles, ~2
  active warps/scheduler, shared-scoreboard 49.9% of 18.1 cyc/inst.  The
  new map splits M in halves as well (warp w owns ni = w%4 and the
  mi-block pair (w//4)*16): W=8, NO guard, NO scf.if-yield regions, B
  fragments still reused across the warp's mi pair.  regs 255 (60 B
  spill) -> 248 (0 spill).
* **math.fma in the non-kahan sgemm bodies** (GDN_FMA=1): single-rounding,
  strictly tighter than mulf+addf, same sum order; kahan bodies excluded
  BY DESIGN (compensation must not contract).  wyb PTX 0 -> 8 fma.rn.f32
  static sites (its tv sgemm); wyb wall 0.987 -> 0.985 (noise -- the
  in-chain non-kahan scalar surface is small now that everything heavy
  is tf32/mma; the win is forward-facing for any new scalar gemm).

## The verify gate caught FOUR bugs this round (5 rounds in a row it earns its keep)

1. bpre A-split staged at global col0=i0 instead of i0*C -- block 0
   coincidentally correct, blocks k>=64 read the wrong M_all columns
   (dh0/dk/dv/dbeta 0.5-class FAIL; the per-k-row probe showed k<64
   exact at 1e-7, k>=64 wrong).
2. dqkwg v3's dq_dPmk port wrongly carried at=True: dq = dPm @ k reads
   dPm NATURAL (only dk_dPmq is the transposed form dPm^T @ q); dq
   0.5-class FAIL until the one-line fix.
3. The pdO fold dropped the old site's 1/sqrt(K) dO-side scale (the
   probe's own numpy reference had the same omission -- the twin
   comparison exposed it; fix: bake scale into rie).
4. mma_gemm_mh's first cut added a BLOCK-count base to ROW offsets
   (warps 4-7 read A rows {2,18} instead of {32,48}); 0.9-class FAIL,
   fixed to rows (w//npr)*(mpw*16).

## What did NOT land (honest negatives)

* **dqkwg wall 0.94x despite the -69% LDS kill**.  ncu A/B (new vs HEAD):
  12.7% vs 28.6% memory throughput, ~same duration under serialization --
  the kernel is LATENCY-bound at 1 CTA/SM (99328 B smem, 8 warps = 2 per
  scheduler, occupancy 15.8%): the issue slots freed by the LDS storm
  were refilled by the element-wise staging sweeps (the st.shared hazard
  rule's price vs the old 16B LDGSTS bursts) and the larger mma+LDSM
  streams, and there is no second CTA to hide either.  The K=64 pdO-link
  re-chunk and the mihalf no-guard map were both measured neutral-to-
  0.94x.  KEPT ON: e2e 5.97 with v3 vs ~6.09 with the old dqkwg + all
  other levers (the -0.11 ms dqkwg regression is smaller than the
  fwdscan-class wins it enables; and the mma path + 2D map are now
  verify-proven machinery).  The r9 lever is OCCUPANCY: retire the pt
  pair (18432 B) by splitting the TW tf32 tails onto the freed slots
  (or bf16-halving accCC/tCC) to get under 50176 B -> 2 CTAs/SM.
* **wyb tv 3-mma: designed, SMEM-BLOCKED**.  Needs the T^T hi/lo pair
  [64,72] bf16 (18432 B) + dvpT splits [16,72]x2 (4608 B) against 8704 B
  free at wyb's 98048 B -- the X/Y/TW_rb tf32 chain's tT/tCKa/tCKb/accCK
  tiles own the rest.  The unblock (this round's design note in the
  generator comments): stage T TRANSPOSED once (tTt) so X reads bt=False
  and Y/TW_rb/tv read at=False -- ALL FOUR sites become natural reads of
  one tile -- then retire the 16 KB tT f32 and the splits fit.  Prize
  unchanged from r7's measurement (~158 us).
* **T512 median 1.004x in the final window** (1.175x in an earlier
  window): the small-shape chain is launch/latency-dominated and the
  co-job makes medians swing; the min-based 1.117x is the stable figure.
* caryA's 255-reg/82 KB state (1.39x, not the 2.5x+ its tensor-core FLOP
  ratio suggests): needs the same occupancy treatment as dqkwg.

## Numerics: 6/6 PASS at BOTH shapes; margins improved at dq/dv

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T512H4 | 4.6e-4 (17x) | 1.7e-3 (5x) | 1.4e-6 (5559x) | 3.0e-4 (66x) | 2.1e-4 (97x) | 2.4e-6 |
| B1T2048H16 | 4.9e-4 (16x) | 1.7e-3 (5x) | 3.3e-6 (2454x) | 1.4e-3 (14x) | 4.5e-4 (44x) | 2.8e-6 |

vs round 7: dq 1.1e-3 -> 4.9e-4 (7.5x -> 16x margin; the dPm^T fix + the
dq1 epilogue scale), dv 9.6e-5 -> 3.3e-6 (83x -> 2454x: dvp's pdO+bh now
runs entirely through exact-bf16/split products), dk/dbeta/dg/dh0 at the
round-6/7 value class (the carry tf32 sites perturb the hyper-contractive
chain below the printed digits; dh0 2.8e-6 unchanged to 2 digits).

## Ranking after round 8 (round-9 scope)

1. **dqkwg occupancy** -- smem 99328 -> <=50176 B (retire pt pair /
   bf16-halve the [C,C] f32 tiles) -> 2 CTAs/SM: converts the LDS-kill +
   mihalf machinery into the ~1.4 ms recoverable the static analysis
   ranks it at (0.94x -> target 1.2x+).
2. **wyb tv 3-mma via the T-transposed restage** (all-natural tTt reads;
   frees the 16 KB tT f32; ~158 us prize measured at tf32-tv speed).
3. **dhs 0.59** -- A-side latency (the rgemm global A-read chain); smem
   ping-pong A tiles.
4. **caryA register/smem pass** (255 regs / 82176 B -> 2 CTA/SM; the
   tf32 conversion is otherwise unbounded by its 1.39x).
5. **wy 0.70 tail / fwdscan remainder / dqkwg TW-tail splits** (192 tf32
   links left).

Repro: python tools/gdn_bwd_gen.py --shape ... --out DIR + bash
evidence/G5/scratch/r8_build_chain.sh DIR; verify tools/gdn_bwd_verify.py
--gen-root DIR --shape ...; bench tools/gdn_bwd_bench.py.  Knobs:
GDN_DQKWG_V3=0 (round-7 dqkwg), GDN_BPRE_MMA=0 (scalar bpre), GDN_FMA=0,
bisect arms GDN_DQKWG_PD0/GDN_DQKWG_BH0/GDN_DQKWG_DUMP.


# G5-P round 9: the dqkwg 16-warp occupancy conversion (lifetime-arena smem aliasing)

Artifacts: tools/gdn_bwd_gen.py (dqkwg v4 + smem_alias.json emission +
shape-adaptive dispatch), tools/gdn_smem_fix.py (fix-1 extension: arena
aliasing -- aliased mallocs deleted, cvtas patched to arena+offset;
GDN_MAXREG probe knob), tools/ncu_one_r9.py (single-kernel ncu launcher),
evidence/G5/scratch/r9_{base,v4,v4b,T512f,T2048f}_* gen dirs +
r9_logs/*.json (paired benches + verifies + the bisect trail), this
section.  Same-window baselines rebuilt from git HEAD 87a36d0 at round
start (6.0-6.7/5.35 med/min across windows -- co-job load heavier than
round-8 windows; all ratios below are same-window interleaved pairs).

## Headline: dqkwg 1.772 -> 1.075 ms (1.65x); bwd e2e 6.00/5.36 -> 5.20/4.66 @B1T2048H16 (1.155x med / 1.150x min)

ours/fla bwd = 1.03/5.20 = **0.198x** (was 0.173x).  T512 keeps v3
(the conversion needs the grid to fill the device; see honest
negatives) -- T512 e2e 0.979x med / 1.000x min = noise.

| B1T2048H16 | HEAD baseline | round9 | delta |
|---|---|---|---|
| wy | 0.6994 | 0.6994 | = |
| fwdscan | 0.3752 | 0.3731 | = |
| caryA | 0.2874 | 0.3271 | 0.88x (noise; the base window itself swung 0.28-0.38) |
| caryM | 0.1604 | 0.1734 | = (noise) |
| bpre | 0.0497 | 0.0720 | = (noise; same cubin both sides) |
| dhs | 0.5919 | 0.5917 | = |
| dkb | 0.2481 | 0.2516 | = |
| **dqkwg** | **1.7724** | **1.0749** | **1.65x** |
| wyb | 0.9869 | 0.9848 | = |
| **e2e bwd** | **6.001/5.355** | **5.197/4.657** | **1.155x med / 1.150x min** |

## What landed: HOW the smem got under 50,176 B (three pieces)

1. **PTX-level lifetime-arena aliasing** (the new machinery,
   gdn_smem_fix.py fix-1 extension).  MLIR keeps one memref.alloc per
   tile (SSA-legal); at PTX level the ALIASED tiles' mallocs are
   deleted and their cvta conversions become arena_base_symbol +
   add.u64 offset.  Mallocs are identified by UNIQUE ALLOC SIZES (LLVM
   reorders malloc callseqs -- order is not a stable key);
   identification pads are extra never-addressed ROWS for ldmatrix
   tiles (row stride must stay a multiple of 16B) and +4 cols for
   generic-only f32 tiles; on any size mismatch the spec self-disables
   (correct, unoverlapped).  Arenas: accCC region 17408 B hosts {dPm,
   T-stage, pthi, bdt@+pthi}; vnh region 10240 B hosts {vnh, vnl@+vnh,
   ptlo, tCKa, tCKb}.
2. **Structure** (the lifetimes that make the arenas legal): the P
   stage is retired (r4/c4r warp-butterfly sums read P from GLOBAL --
   same lane/tree, bit-identical products); the dPm tails run EARLY
   (dq_dPmk/dk_dPmq park their [C,BK] slices in the dq/dk_local
   GLOBALS right after the emap; the dW loop read-add-stores dq[ks],
   the TW loop read-adds dk_local[ks] -- +96 KB/chunk of L2-resident
   round trips, the price for killing accCC-dPm's 16 KB liveness
   before the pt stage); the TW loop drops its q stage.
3. **2 CTAs x 256 threads = 16 warps/SM.**  ptxas lands at **96
   registers / 0 spill** on the aliased PTX (the 23-malloc unaliased
   build wants 255 regs + spill at the same body -- the arena patch
   also collapsed the address-space complexity; the 128-reg cap that
   gates 2-CTA residency is free).

ncu before/after (T2048, kernel-only):

| metric | v3 baseline | v4 round9 |
|---|---|---|
| block / static smem / regs | 256 thr / 99328 B / 132 | 256 thr / 48640 B / 96 |
| block limit (smem, regs) | 1, 1 | 2, 2 |
| theoretical / achieved occupancy | 16.67% / 15.81% | **33.33% / 30.28%** |
| achieved warps per SM | 7.59 | **14.53** |
| active warps per scheduler | 1.98 | 3.69 |
| no-eligible cycles | 88.98% | 79.56% |
| SM memory throughput | 13.2% | 20.0% |

## The verify gate caught the round's real bug (6th round running)

**cp.async wait_group is PER-THREAD.**  The TW loop's inline
width-restricted tCKa stage (an astage_tile + await_all WITHOUT the
trailing gpu.barrier that astage_sync carries) let warps race into the
TW tgemm reading tCKa columns owned by OTHER threads' in-flight
chunks.  Symptom trail: dk_local cols 0..5 of EVERY ks slice stale
(exactly the 16B chunks owned by threads 0-1), 9.3e3..9.8e10 final-dk
garbage at 256thr/full grids, INVISIBLE at 128thr/T512 grids (window
too narrow) -- bisected via dk_local base-vs-new npz diffs after a
tf32/scalar arm-split pinned the corruption to the TW gemm.  One
gpu.barrier fixes it; 6/6 PASS both shapes at EXACTLY the baseline
margin digits (dk 1.719e-3/1.710e-3; the early-tail restructure adds
only ~1-ulp-order reassociation, invisible at the printed digits).
A second latent find from the same hunt: v3-style plain mma_gemm
chains carry NO trailing barrier (mma_chain_if documents the hazard
and has one) -- the dPr vs-loop could restage akb/vnh/vnl under slower
warps' mma reads; v4 adds a per-iteration barrier.

## Honest negatives

* **The first conversion design (128thr x 2 CTAs) was a structural
  no-op.**  sm_120 has 48 warp slots/SM (12/scheduler -- the r7
  costmodel assumed 32): 2x4 warps = the SAME 8 warps as 1x8, and the
  measured wall was a wash (1.780 vs 1.775 @T2048) while T512 lost
  1.8x (halved warps, no residency gain at grid 32).  The win needed
  16 warps (256thr x 2 CTAs = regs<=128), which the arena-aliased PTX
  made free (96 regs).  The honest 128-thread v4 with both races
  closed measures 0.57x at T512 -- its earlier "wash" was riding the
  illegal warp overlap.
* **T512 keeps v3** (GDN_DQKWG_V4 auto-off below 164 blocks): the
  conversion only pays when the grid fills the device; small-shape
  chains underfill at any residency.  T512 e2e = noise-level 0.979x
  med / 1.000x min.
* **The shipped chain has a PRE-EXISTING nondeterministic dk path**:
  same-window baseline verify runs sampled final dk 1.719e-3 /
  1.719e-3 / 4.719e-3 (and the 256-thread timing amplified it to
  ~1.1e-2 in early arms before the real bug above was found).  dq/dv/
  dbeta/dg/dh0 were stable in every run; dqkwg outputs were
  bit-stable across runs.  The dk 5x margin is partly luck -- finding
  this race is round-10 #1 (it gates every dk-adjacent change).
* **wyb tv 3-mma and the dhs A-cache did not land** (the race hunt
  consumed the round's lever-2/3 budget; designs stand -- the r8 wyb
  blocker was 8.7 KB free vs 23 KB needed, and the r9 arena machinery
  now makes the tTt restage + split-tile packing straightforward).
* ncu serialized-replay Duration is near-equal (2.72 vs 2.78 ms) and
  NOT representative of the wall (replay serializes the very
  residency the conversion exploits); the paired wall bench is the
  criterion.

## Numerics: 6/6 PASS at BOTH shapes; baseline digits kept exactly

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T512H4 | 4.6e-4 (17x) | 1.7e-3 (5x) | 1.4e-6 (5559x) | 3.0e-4 (66x) | 2.1e-4 (97x) | 2.4e-6 |
| B1T2048H16 | 4.9e-4 (16x) | 1.7e-3 (5x) | 3.3e-6 (2454x) | 1.4e-3 (14x) | 4.5e-4 (44x) | 2.8e-6 |

(T512 runs the v3 dqkwg -- identical to round 8 by construction.)

## Ranking after round 9 (round-10 scope)

1. **The pre-existing dk race** -- baseline final-dk swings 1.7e-3 ->
   4.7e-3 run-to-run (worse under co-load); localize with the r9
   npz-diff harness (host readback after every stage), then fix.
   Every dk-margin-consuming change is gated on it.
2. **wyb tv 3-mma split-bf16** (~158us prize, r8 design): stage T
   transposed ONCE (all four consumers natural-read), retire the 16 KB
   tT f32, and arena-alias the T^T/dvp split tiles -- the r8 smem
   blocker dissolves under the r9 machinery.
3. **dhs 0.59** -- A-side latency (the rgemm global A-read chain); smem
   rolling-window A tiles, st.shared per the r6 hazard rule.
4. **caryA 255-reg/82 KB** -- the arena-aliasing + 2-CTA treatment
   that paid 1.65x on dqkwg (its tf32 mma is landed; occupancy is the
   cap).
5. **dqkwg tail tf32->split-bf16** (192 links) -- the arena budget
   (48640 of 50176 B) has no headroom for [C,C] split pairs today;
   needs pt-slot recycling or the TW restructure first.

Repro: python tools/gdn_bwd_gen.py --shape 1,2048,16,128,128 --out DIR
+ bash evidence/G5/scratch/r8_build_chain.sh DIR (for the 256thr
dqkwg, re-assemble its smemfix with ptxas --maxrregcount 128 -- see
the r9 build scripts in evidence/G5/scratch/r9_logs); verify
tools/gdn_bwd_verify.py --gen-root DIR --shape ...; bench
tools/gdn_bwd_bench.py.  Knobs: GDN_DQKWG_V4=0 (v3 always),
GDN_DQKWG_THR (default 256), GDN_DQKWG_PD0/BH0/DUMP bisect arms,
GDN_MAXREG (smem_fix probe), GDN_BWD_DBG/GDN_DQKWG_DUMP unchanged.


# G5-Q round 10: the pre-existing dk nondeterminism -- root-caused and closed (TWO races)

Artifacts: tools/gdn_det_check.py (whole-chain run-to-run determinism
harness: per-run npz of EVERY stage's OUT globals, in-process loop /
fresh-process single / diff modes; stage modules unloaded after readback
-- the <4GB co-job rule), tools/gdn_bpre_loop.py (single-kernel fresh-
module hammer), tools/gdn_det_discriminator.py (h2d/launch/bpre/idle
arms), tools/gdn_fwd_det.py (forward o/ht determinism),
evidence/G5/scratch/r10_logs/* (all probe logs + npz sets),
evidence/G5/scratch/{r10_chain,r10_chain_T512} (fixed chains),
evidence/G5/bwd10_verify{2048,512}.json, this section.

## Headline: N=10 bit-identical, both shapes, fresh-process; 6/6 PASS at exactly the baseline digits; cost ~0

| suite | pre-fix | post-fix |
|---|---|---|
| T2048 fresh-process x10 | wyb dk/dbeta/dg diverge (1 run; dk 8153 elems 8.3e-4) | **ALL stage outputs bit-identical** |
| T512 fresh-process x10 | dk/dbeta/dg diverge (3/10 runs; up to 7113 elems 7.2e-4) | **ALL bit-identical** |
| T2048/T512 in-process x10 | bpre B_all ulp-flips (1/10) / dk flips | **ALL bit-identical** |
| bpre alone, fresh-module x1000 | 14/1000 divergent (bistable 1-3 ulp) | **0/1000** |
| bpre scalar control (no mma) x400 | 0/400 | -- |
| verify B1T2048H16 | dk sampled 1.719e-3 or 4.719e-3 run-to-run | **6/6 PASS, dk 1.710e-3 fixed digits** |
| verify B1T512H4 | (same class) | **6/6 PASS, dk 1.719e-3 fixed digits** |
| FWD mono staged + fused x10 (B1T2048H16) | not previously checked | **o/ht bit-identical both paths** (forward was already clean) |

## The hunt (how the corruption was localized)

1. **Per-stage npz bisect** (gdn_det_check): FIRST diverging global =
   bpre's B_all -- 12/8.4M elems in ONE chunk row (m=132), one K row
   (k=3), a 16-wide V slab, reldiff ~1e-7 = 1-3 f32 ulp.  Everything
   upstream (wy/fwdscan/caryA/caryM) and ALL dqkwg outputs bit-stable --
   matching r9's observation exactly.  Propagation: dh_all (values
   identical to B_all's, absorbed at the next add) -> dkB_sum -> one dk
   element's bf16 round-to-nearest flips = the 1.719e-3 <-> 4.719e-3
   margin swing (one bf16 quantum apart).
2. **Isolation**: bpre alone, same module, 300+150 launches: 0 flips.
   Fresh module per launch (chain harness's module cycle): 14/1000.
   Flips come in temporal BURSTS (iters 418-425 of the 1000) --
   co-job-load-correlated; one flip (m=228 col=267) reproduced the SAME
   (ref, now) value pair twice = BISTABLE, two deterministic outcomes,
   a race and not garbage.
3. **Discriminator arms**: h2d-only (no launch) x200 clean (module/
   memory path innocent); launch+input-readback x200 clean (no OOB
   writes into M_all/do); scalar-bpre control (GDN_BPRE_MMA=0, no
   mma/ldsm) x400 clean -> the mma read path.

## Root cause #1: bpre's K-loop mma restage race (the r9 class, in bpre since r8)

`for i0 in (0, 64): stage_split_ew(ahi, alo); mma(bmh: ahi); mma(bml:
alo, epilogue -> B_all)`.  Both staging helpers end in gpu.barrier, but
NOTHING separates iteration i0=0's mma ldmatrix READS of ahi/alo from
i0=64's split STORES to the same tiles (PTX: 4 barriers; the missing
one is inside the loop).  A fast warp finishing bml{0} early (its
epilogue's 64 global stores make it slow to get there, but a slow warp
still issuing LDSMs is all it takes) overwrites alo under slower warps'
reads.

**Why it looked like 1-ulp noise and not corruption**: the swapped
operand elements are the LO-split values (lo = v - bf16(v), many
~1e-10), so old-vs-new product deltas land at ~1e-10 = 1-3 ulp of the
affected outputs; one 16-byte LDSM row (8 alo elements) = the observed
12-of-16-element clusters; all flips land in k rows 0..63 = i0=0's
outputs.  The ahi side (hi values ~1e-4) would corrupt at O(1e-3) and
fail verify outright -- it practically never loses the race because the
fast warp is only ~100 cycles past bmh{0} when it reaches the restage;
bml{0} is the exposed link.  r9 documented exactly this class
("mma_gemm inter-iteration restage race") and fixed it in dqkwg v4's
dPr loop -- bpre was missed.

**Fix**: one `gpu.barrier` at the end of each K-loop iteration
(tools/gdn_bwd_gen.py gen_bpre).  4 -> 5 barriers, 54 regs, 0 spill.

## Root cause #2: wyb's G_dvp/G_dW accumulation loops (cp.async flavor, bigger, older)

With bpre fixed, fresh-process runs STILL diverged -- now INSIDE wyb
(all its inputs stable): dk/dbeta/dg at 1e-6..1e-4 over thousands of
elements, dv untouched.  Signature points at G (accCC2 = strict_tril
dvp(beta v)^T + dW(beta k 2^gc)^T): G feeds dAf -> dk/dbeta/dg; dv's
path never touches G.  gen_wyb's two accumulation loops
`for vs: astage_sync(tCVa <- dvp_all); stage_raw_ew(tVV <- v);
emap(tCVb); gemm(accCC2, tCVa, tCVb, bar=False, kahan)` and the same
shape `for ks: ... gemm(..., tCKa, tCKb, bar=False, ...)` have NO
barrier after the gemm inside the loop: the NEXT iteration's
astage_sync DMA (cp.async) rewrites tCVa/tCKa while slower warps' gemm
LDS reads are in flight (and emap rewrites tCVb/tCKb/tKK under them).
Scale-consistent: full operand value swaps at G ~1e-3 -> dAf ->
thousands of downstream 1e-6..1e-4 shifts.

**Fix**: one `gpu.barrier` after each loop's gemm (2 sites).  wyb PTX
127 -> 138 barriers.  T512 fired it 3/10 pre-fix (small-grid chains
spend relatively more time in the restage windows); this is almost
certainly the dominant contributor to r9's sampled dk swings (the bpre
flips were the ulp-level tail).

## Same-class latent sites closed in the same pass (audit of every mma/ldsm consumer)

* **dqkwg v3 dPr vs-loop** (the T512 dispatch): identical missing
  per-iteration barrier -- r9 fixed only the v4 copy.  Fixed (v3 PTX
  192 -> 195 barriers; T512 dqkwg md5 changes, T2048 keeps v4).
* **dqkwg v2 tail dvb ks-loop** (legacy GDN_DQKWG_V3=0 path, not
  emitted by default): guarded mma + all-thread restage (mma_chain_if's
  documented "idle warps race ahead" pattern on a raw mma_gemm) --
  barrier added outside the scf.if (a barrier inside a <W guard
  deadlocks).
* Audited-clean (no change): wy kkt/pmm (single-shot, disjoint tail
  tiles), dkb (astage_sync's barrier orders the inter-iteration
  restage; _split_bf16's hi/lo loops each barrier-terminated), dqkwg
  v4 dPr (r9 barrier) and all v3/v4 dvp/dW/dq1 chains (mma_chain_if
  carries the trailing barrier), caryA/caryM (scalar/tf32 `gemm`
  bar=True default), dhs (rgemm reads globals directly -- no smem
  operands to restage), fwdscan F2 groups, wyb TW/tail loops
  (explicit barriers already).

## The audit rule added to the toolbox (checklist for every future kernel change)

1. Every smem tile read cross-warp (ldmatrix OR ld.shared) needs a
   gpu.barrier AFTER the last writer's store loop and BEFORE the first
   reader (staging helpers ending in a barrier satisfy the read side).
2. **The restage fence**: any loop that (a) stages tiles, (b) runs
   gemm/mma reading them, and (c) RE-stages the SAME tiles next
   iteration needs an all-thread barrier AFTER the reads (end of
   iteration) -- a trailing barrier inside the NEXT iteration's staging
   does NOT order them (both sides are pre-barrier for the racing
   warp).  Applies to st.shared restage (bpre) AND cp.async restage
   (wyb G loops; cp.async DMA writes are issued before their commit/
   wait/barrier).
3. cp.async wait_group is per-thread: every wait_group needs a
   gpu.barrier before other warps read the staged data (r9).
4. A barrier must be OUTSIDE any <W-warp guard region (bar.sync is
   all-thread); guarded mma chains use mma_chain_if (which owns the
   trailing barrier).
5. cp.async(LDGSTS)-staged tiles must NOT be ldmatrix-read (r6 silent-
   corruption family); mma operand tiles are st.shared-written.
6. Never rely on smem-alloc-as-zero (r7): unread regions are zeroed or
   provably never read (the C+8 LDSM pads are never read -- verified
   against the lane-address arithmetic).
7. Determinism proof protocol for any change touching these paths:
   tools/gdn_det_check.py fresh-process N=10 BOTH shapes (the T2048
   in-process loop alone did NOT catch the wyb race pre-fix --
   fresh-module cycling is the wider window) + verify 6/6.

## Numerics: 6/6 PASS both shapes, identical digits to round 9

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T512H4 | 4.6e-4 (17x) | 1.7e-3 (5x) | 1.4e-6 (5559x) | 3.0e-4 (66x) | 2.1e-4 (97x) | 2.4e-6 |
| B1T2048H16 | 4.9e-4 (16x) | 1.7e-3 (5x) | 3.3e-6 (2454x) | 1.4e-3 (14x) | 4.5e-4 (44x) | 2.8e-6 |

The barriers are numerics-neutral by construction (same values, same
order -- they only remove illegal overlapping reads); the printed
digits confirm (dk 1.710e-3/1.719e-3 are the r9 "clean draw" values;
the 4.719e-3 draw is gone).

## Forward chain: determinism confirmed clean

gdn_full mono staged (F1->F2->F3) and fused, B1T2048H16, N=10 fresh
processes each: o/ht bit-identical (tools/gdn_fwd_det.py).  No action
needed.

## Perf: the determinism fixes cost ~0.3% at T2048, noise at T512 (same-window paired A/B x3, interleaved)

| B1T2048H16 | HEAD 0b5a354 | round10 | ratio |
|---|---|---|---|
| bpre | 0.0541 | 0.0600 | 0.90x (+6us: the barrier) |
| wyb | 0.9846 | 0.9875 | 1.00x (+3us: 11 barriers) |
| all other stages | = | = | 1.00-1.01x (noise) |
| **e2e bwd med/min** | **5.235 / 4.662** | **5.218 / 4.658** | **1.003x med / 1.001x min** |

T512: e2e med 0.796 -> 0.758 (window noise; mins 0.731 vs 0.734 = dead
even).  ours/fla bwd = 1.03/5.218 = 0.198x (unchanged from r9).

## Honest negatives

* **The round-9 ranked perf levers were NOT reached** (wyb tv 3-mma via
  tTt + arenas ~158us, dhs A-cache 0.59ms, caryA 255-reg arena
  treatment): the correctness mission consumed the round (reproduction,
  isolation, discriminator arms, two root causes, the same-class audit
  of every mma/ldsm consumer in the generator, and the N=10x2-shape
  proof protocol).  Designs stand as ranked in round 9; the dk gate
  they were blocked on is now CLOSED (this was the ranking's explicit
  #1: "every dk-margin-consuming change is gated on it").
* The T2048 in-process 10-run loop alone did NOT reproduce the wyb race
  pre-fix (only fresh-process module cycling did) -- in-process looping
  is a NECESSARY but not SUFFICIENT determinism probe; the fresh-mode
  harness is now part of the gate protocol (checklist item 7).
* bpre's ahi-side (gross-corruption) variant of the restage race was
  never observed firing in any sample (only the alo/bml side); the fix
  closes both by construction.

## Ranking after round 10 (round-11 scope)

1. **wyb tv 3-mma split-bf16 via the tTt restage + r9 arenas** (~158us
   prize, r8/r9 design unchanged; the dk determinism gate is now clear
   -- this lever is unblocked).
2. **dhs A-cache** (0.59ms; A_all rolling smem window; r6 hazard rule
   + the r10 checklist now applies to its introduction).
3. **caryA arena + 2-CTA conversion** (255 regs / 82176 B; the dqkwg
   r9 playbook).
4. **dqkwg tail tf32->split-bf16** (192 links; needs pt-slot recycling
   or the TW restructure first).
5. Fresh-process determinism N=10 both shapes added to EVERY future
   round's verify gate (checklist item 7; ~4min wall).

Repro: python tools/gdn_det_check.py --gen-root DIR --shape
1,2048,16,128,128 --mode loop -n 10 --out OUT (in-process) and --mode
single --run i --out OUT/runi.npz x10 + --mode diff --out OUT
(fresh-process); hammers: tools/gdn_bpre_loop.py [--fresh]; forward:
tools/gdn_fwd_det.py --tag bench_B1T2048H16 --path mono|fused;
verify/bench unchanged.  Gen dirs: evidence/G5/scratch/r10_chain (T2048,
bpre+wyb changed), r10_chain_T512 (bpre+wyb+dqkwg-v3 changed); probe
logs evidence/G5/scratch/r10_logs/.


# G5-R round 11: the three ranked levers land (wyb tv 3-mma, dhs A-window, caryA mma diet)

Artifacts: tools/gdn_bwd_gen.py (gen_wyb tv split-bf16 3-mma + the tT/accCK
lifetime arenas; gen_dhs GDN_DHS_AW rolling A-window + rgemm_aw; gen_caryA
GDN_CARYA_MMA split-bf16 mma + the stage_split_T cvec column-scale fold;
stage_rows shape= param), evidence/G5/scratch/r11_{base,l1,l23}_{T2048,T512}
gen dirs (l1 = lever 1 alone, l23 = all three; the six untouched kernels are
the round-start baseline cubins, md5-checked) + r11_{b16,aw0}_T2048 probe
chains + r11_logs/* (paired bench jsons x3 interleaved per shape, det npz
sets, the AW-isolation npzs) + r11_{build_one,det_fresh,ab_bench,freshness}.sh
helpers, this section.  Same-window baselines rebuilt from git HEAD ac771a1
at round start (5.207/4.657 med/min first sample -- within noise of the r10
table's 5.218/4.658).

## Headline: bwd e2e 5.209/4.662 -> 4.526/3.930 @B1T2048H16 (1.151x med / 1.186x min); T512 0.786/0.735 -> 0.724/0.626 (1.086x med / 1.175x min)

ours/fla bwd = 1.03/4.526 = **0.228x** (was 0.198x).  All ratios
same-window interleaved pairs x3 (base,new,base,new,... 30-iter CUDA-event
medians).

| B1T2048H16 | HEAD baseline | round11 | ratio | note |
|---|---|---|---|---|
| wy | 0.6985 | 0.7006 | = | untouched (md5 SAME) |
| fwdscan | 0.3731 | 0.3733 | = | untouched |
| caryA | 0.3093 | 0.0618 | **5.00x** | lever 3 |
| caryM | 0.1602 | 0.1619 | = | untouched (window noise band) |
| bpre | 0.0495 | 0.0840 | (0.59x) | identical cubin -- co-job noise (see below) |
| dhs | 0.5921 | 0.3197 | **1.85x** | lever 2 |
| dkb | 0.2537 | 0.2502 | = | untouched |
| dqkwg | 1.0735 | 1.0731 | = | untouched (v4 @T2048, v3 @T512 as r9/r10) |
| wyb | 0.9874 | 0.8178 | **1.21x** | lever 1 |
| **e2e bwd** | **5.2087 / 4.6615** | **4.5264 / 3.9303** | **1.151x med / 1.186x min** | |

Stage-delta accounting closes: caryA -0.248 + dhs -0.272 + wyb -0.170 =
-0.690 ms vs e2e med delta -0.682 ms.  bpre's 0.59x is a WINDOW artifact on
a byte-identical binary (r9 documented its 0.049-0.084 swing class; the l1
window measured the same cubin at 0.0514).  T512: caryA 0.0456 -> 0.0164
(2.78x), dhs 0.1230 -> 0.0671 (1.83x), wyb 0.1253 -> 0.0989 (1.27x); T512
med is launch-noise-dominated as ever (runs 0.636-0.754), the stable figure
is min: 0.7354 -> 0.6259 = 1.175x.

## Lever 1: wyb tv 3-mma via lifetime arenas (wyb 1.21x/1.27x; the r7-measured ~158us prize collected in full)

The r8 smem blocker (8.7 KB free vs 23 KB of split tiles at wyb's 98048 B)
dissolved WITHOUT the tTt restage: the r9 arena machinery hosts the splits
in tiles whose lifetimes end first.

* **Arenas**: tT -> [C, C+8] f32 (alloc 18432; the size-pad makes its
  malloc uniquely identifiable, the r9 rule) hosts tThi/tTlo ([64,72] +
  [66,72] bf16, fp 9216 each, staged after TW_rb's last tT read);
  accCK -> [C+2, C] f32 (alloc 16896, fp 16384) hosts dhi/dlo ([16/18,72]
  bf16) + tvacc ([66,16] f32) after the dk loop.  Pool 98048 -> 100096 B
  (the +2048 is the T-split pair's true footprint; everything else is
  free).  smem_fix consumed the spec (smem_alias.json) exactly as
  designed: 22 mallocs -> 17 decls, one at arena size 18432.
* **The mma**: tv = T^T @ dvp as hi*hi + hi*lo + lo*hi links
  (mma_gemm_mh, M=C, N=BV=16, K=C, W=8 -- npr=2 x 4 mi-halves, all 8
  warps, no guard); A splits staged ONCE from T_all, B splits per
  vs-slice from dvp_all (stage_split_T transposing stages).  fp32-class
  accumulate -- the numerics class tf32-tv was rejected at (2.15x) in r7.
* **dv folded into the epilogue** (the r5 beta-fold, epilogue-side): each
  acc lane stores tvacc (for dbeta's rc rowsum) AND the beta-scaled dv
  global element -- the old emap/store_rows passes are gone.
* regs 239 -> 226 (0 spill), mma.sync 384 -> 480, ldmatrix 0 -> 128; the
  tv scalar gemm's 8 unrolled K=64 loops are gone.
* Detail: the padded [C, C+8] tT cannot take the 16B-burst astage (1152
  chunks % 256 threads != 0) -- stage_rows gained a shape= param and
  stages the bounded [C, C] region element-wise (one sweep, once).

Solo same-window numbers (l1 chain): wyb 0.9871 -> 0.8174 (1.21x), e2e
5.237/4.662 -> 5.112/4.495 (1.024x med / 1.037x min).

## Lever 2: dhs A-cache -- the rolling smem window (1.85x/1.83x; 0.59 -> 0.32 ms)

rgemm's 2048 warp-broadcast global loads/thread/chunk (the long-scoreboard
chain that made dhs the #2 stage) are gone: the kk-outer
register-accumulator gemm now reads A[i, kk] from ping-pong [K, BKW=32] f32
smem windows, cp.async-staged one window ahead (astage_tile + agroup +
await_all -- the r7 fwdscan prefetch-group pattern; 24 bursts/chunk).

* Hazard compliance: the window feeds GENERIC ld.shared only (the r6
  rule's legal flavor -- never ldmatrix); every wait_group is followed by
  gpu.barrier before cross-warp reads (r9 rule 3); each window's reads are
  fenced by a trailing barrier before that buffer's next restage, which
  is always >= 2 iterations later (r10 rule 2).
* **Value-exactness PROVEN, not assumed**: an AW=0 control chain (same
  seeded inputs, same caryA cubin -> bit-identical A_all/B_all/dht) vs
  the AW=1 chain: dh_all AND dh0 BIT-IDENTICAL (r11_logs/awiso).  The
  window is a pure latency change; the kk-ascending FMA chain is
  untouched.
* smem 32768 -> 65536 (X + Xa + 2x16KB), regs 48 -> 80, 0 spill, cp.async
  0 -> 24.
* **BKW=16 probe REJECTED**: 8KB windows (48KB total, 2-CTA-capable)
  measured dhs 0.352 vs 0.320 (0.91x) -- the grid (64 blocks) underfills
  the device so the second CTA/SM buys nothing while the doubled barrier
  count + 16B spill cost real time (r11_b16_T2048 chain kept as
  evidence).  BKW=32 is the default (GDN_DHS_BKW knob).

## Lever 3: caryA -- the split-bf16 mma diet (5.00x/2.78x; the r8 1.39x cap broken)

The "arena treatment" became the arena-DISSOLVING diet: a bpre-style port
retires ALL FOUR co-live tiles (no alias is legal without restructuring)
for three bf16 mma tiles.

* Retired: kst [C,K] bf16 + bhb [C,K] f32 + wb [C,BK2] f32 + ab
  [BK2,BK2] f32 = 82176 B.  Added: ahi/alo ([64/66, 72] bf16, W^T
  splits, staged once per i0 and REUSED by both j0 blocks) + bt ([65,72]
  bf16, k^T RAW -- the global IS bf16, exact) + gc + seg = 28592 B.
  **82176 -> 28592 B and 255 -> 76 regs (0 spill both) -> 2 CTAs/SM**
  (the r9 dqkwg playbook; 1.65x there, 3.6x incremental on the already-
  1.39x tf32 base here).
* **The Bhat factor folds into the A splits**: the new stage_split_T
  cvec= param column-scales the split by seg[c] = 2^(gcl-gc_c).  Exact
  (a power-of-two commutes with bf16 truncation AND the hi/lo residual:
  split(v*2^d) == split(v)*2^d), so the products keep the old
  B-side-scaled values; bhb becomes a raw-k copy.
* The bpre epilogue playbook: -neg + the diagonal 2^gcl (an i0==j0
  row==col select) + A_all global-direct stores.  mma.sync 128 (the tf32
  count) -> 128 (coincidence of shapes), ldmatrix 0 -> 144.
* **Reassociation measured, not hand-waved**: vs the tf32 build, A_all
  differs in 8.39M/8.39M elems at max 5.6e-22 ABSOLUTE (2.8e-4 rel -- the
  tf32 truncation class it replaces; A is hyper-contracting so its values
  are ~1e-18-scale) -- absorbed below the carry chain's rounding: every
  FINAL grad digit is unchanged (dk 1.710e-3/1.719e-3, dbeta, dg, dq,
  dh0 all exactly the r10 values).

## Numerics: 6/6 PASS BOTH shapes; margins

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T512H4 | 4.6e-4 (17x) | 1.7e-3 (5x) | 9.4e-6 (851x) | 3.0e-4 (66x) | 2.1e-4 (97x) | 2.4e-6 |
| B1T2048H16 | 4.9e-4 (16x) | 1.7e-3 (5x) | 1.1e-5 (736x) | 1.4e-3 (14x) | 4.5e-4 (44x) | 2.8e-6 |

The ONLY digit moves are dv (3.3e-6 -> 1.088e-5 @T2048, 1.4e-6 -> 9.4e-6
@T512): dv = tv*beta now stacks split-bf16 tv on the split-bf16 dvp (r8)
-- margins 736x/851x vs the 8e-3 bar, the same "tighter products, one
more reassociation" class r8 measured when dvp went split (dv then went
83x -> 2454x TIGHTER).  dk/dbeta/dg/dq/dh0 are the r10 digits exactly.

## Determinism: N=10 fresh-process, BOTH shapes, after EACH structural change

| gate | chain | result |
|---|---|---|
| l1 (wyb arena+restage+mma) | T2048 x10 / T512 x10 | ALL stage outputs bit-identical |
| l23 (dhs window + caryA mma) | T2048 x10 / T512 x10 | ALL stage outputs bit-identical |
| AW isolation | AW=0 vs AW=1, same inputs | dh_all/dh0 bit-identical |

The r10 protocol (checklist item 7) was applied per-lever, not once at the
end: every restage/arena/mma introduction carried its own fresh-module
cycling proof.

## Honest negatives

* **bpre 0.59x in the final T2048 window** -- identical cubin (md5 SAME
  both sides), the documented co-job swing class; the accounting above
  closes without it and the l1 window measured 0.0514 on the same binary.
* **dhs cross-chunk prefetch designed, NOT landed**: issuing chunk c+1's
  window-0 during chunk c's last window needs the cp.async group token
  carried through scf.for iter_args; the per-chunk prologue await remains
  the exposed serial latency (~1 window stage per chunk).  Round-12
  candidate.
* **wyb stays 1 CTA/SM** (pool 100096 > 50176): the tv win is
  issue-slot/latency, not occupancy.  A wyb occupancy pass needs accCC
  retirement or the tf32-tail diet -- a deeper restructure than this
  round's budget.
* **caryA's knob-off (GDN_CARYA_MMA=0) path is NOT payload-identical to
  HEAD**: the kst/bhb/wb/ab allocs moved after gc_cumsum (ahead of the
  mma branch) -- same tiles, same ops, same 82176 B, but different alloc
  ORDER in the MLIR text.  wyb's and dhs's knob-off payloads regenerate
  BIT-IDENTICAL to HEAD (diff-verified) -- clean bisect arms; caryA's is
  semantically identical, textually reordered.

## Ranking after round 11 (round-12 scope)

1. **wy 0.70** -- now the largest stage untouched since r7's k/v staging
   reuse; the twin-solve body is scalar-heavy (static r7 class).
2. **dqkwg 1.07 tail tf32->split-bf16** (192 links; the r9 pt-slot
   recycling / TW restructure prerequisite).
3. **dhs cross-chunk window prefetch** (group-token iter_args; the
   remaining prologue latency, ~0.05-0.1 ms class).
4. **wyb occupancy** (accCC/tf32-tail diet toward <=50176 B).
5. **dkb 0.25 / fwdscan remainder / caryM polish**.

Repro: python tools/gdn_bwd_gen.py --shape ... --out DIR (knobs
GDN_WYB_TV_MMA/GDN_DHS_AW/GDN_DHS_BKW/GDN_CARYA_MMA, all default on);
bash evidence/G5/scratch/r11_build_one.sh DIR <kernel>; verify
tools/gdn_bwd_verify.py --gen-root DIR --shape ...; determinism
evidence/G5/scratch/r11_det_fresh.sh DIR <shape> <tag> [N]; paired bench
evidence/G5/scratch/r11_ab_bench.sh <base> <new> <shape> <tag>; cubin
freshness r11_freshness.sh.  Fixed chains: r11_l23_T2048 / r11_l23_T512.


# G5-S round 12: wy mma+shfl diet, dhs cross-chunk prefetch, dqkwg tf32-tail splits

Artifacts: tools/gdn_bwd_gen.py (gen_wy GDN_WY_MMA k-slice/split-mma diet +
GDN_WY_GC_SHFL Hillis-Steele scan; gen_dhs GDN_DHS_XP cross-chunk group-token
iter_args prefetch; _gen_dqkwg_v4 GDN_DQKWG_SB split-bf16 early-tails+TW;
_split_bf16_T + stage_split_ew/_split_bf16 cvec params), chains
evidence/G5/scratch/r12_{base,l1,l1b,l3,l4}_T2048 (l1 = wy diet alone, l1b =
+shfl, l3 = +dhs XP, l4 = +dqkwg SB = the round-12 final, regenerating
r12_final_T2048 bit-identically), r12_{base,l1,final}_T512, r12_logs/
(interleaved paired bench jsons x3 per comparison, det npz sets) +
r12_{build_all,det_fresh,ab_bench}.sh helpers, this section.  Baselines
rebuilt fresh from git HEAD 28a709a at round start: payload/config/PTX
bit-identical to the shipped r11_l23 chain (only the dqkwg cubin md5 differs
-- ptxas embeds a nondeterministic header; PTX identical).

## Headline: bwd e2e 4.541/3.927 -> 3.734/3.331 @B1T2048H16 (1.216x med / 1.179x min); T512 1.052x med / 1.083x min

ours/fla bwd = 1.03/3.734 = **0.276x** (was 0.228x).  All ratios
same-window interleaved pairs x3 (base,new,base,new,... 30-iter CUDA-event
medians).

| B1T2048H16 | HEAD 28a709a | round12 | ratio | note |
|---|---|---|---|---|
| wy | 0.7000 | 0.2299 | **3.04x** | lever 1 (mma diet + shfl) |
| fwdscan | 0.3730 | 0.3733 | = | untouched (md5 SAME) |
| caryA | 0.0526 | 0.0539 | = | untouched |
| caryM | 0.1655 | 0.1602 | = | untouched (noise band) |
| bpre | 0.0510 | 0.0635 | (0.80x) | identical cubin -- co-job swing (same class as r11's bpre 0.59x; other windows measured 0.0515) |
| dhs | 0.3199 | 0.3117 | **1.03x** | lever 3 |
| dkb | 0.2476 | 0.2482 | = | untouched |
| dqkwg | 1.0750 | 0.9421 | **1.14x** | lever 2 (v4 @T2048) |
| wyb | 0.8174 | 0.8177 | = | untouched |
| **e2e bwd** | **4.5414 / 3.9275** | **3.7342 / 3.3307** | **1.216x med / 1.179x min** | |

Stage-delta sum: wy -0.470 + dqkwg -0.133 + dhs -0.008 - caryM -0.005 +
bpre +0.013 = **-0.603 ms** vs e2e med delta -0.807 ms; the residual
~0.2 ms is window-level (the per-stage medians ride the co-job noise that
also produced bpre's 0.80x on a byte-identical cubin; the e2e runs
themselves are tight: 3.7247-3.7369).  T512 (final window): wy 0.0920 ->
0.0442 (2.08x), dhs 0.0686 -> 0.0659 (1.04x), dqkwg = (keeps v3 per the
r9 ship policy), e2e 1.052x med / 1.083x min (launch-noise-dominated as
ever).

## Lever 1: wy -- the split-mma tile diet + the shfl scan (3.04x/2.08x; the last big single stage retired)

The r8 static verdict (255 regs, 90624 B smem, 12.5% occupancy) dissolved
the caryA way: every co-live [C,K]/[C,C] staging tile retired for
slice/split bf16 mma tiles.

* **kkt/P as k-sliced chains**: the whole-K mma links become 8 K=16
  links over a [C,24] bf16 slice (stage_rows st.shared restage per link,
  r10-fenced; accs= register chaining) -- SAME fragment/mma/acc sequence
  -> kkt/A/P bit-identical by construction.  k_t/q_t [C,K] (32 KB) and
  their cp.async stages retire (wy cp.async 20 -> 0).
* **u/w as 2-link split-bf16 mma** (the r11 caryA playbook on the
  tf32-barred tail): A = T_all COLUMN-scaled by the contraction-index
  factor (beta[k] for u; bgc[k] = beta[k] 2^gc[k] for w, hoisted) and
  split hi/lo via the new stage_split_ew cvec= (arbitrary-scale fold: the
  product rounds once in f32, then the 2^-17 split residual -- vs the
  r3 tf32 bisect that put tf32 wy.u OVER the dk bar at 1.06e-2); B = the
  RAW bf16 global (v/k), transposing-staged to the [n, k] orientation --
  exact, no rounding at all.  The A splits read the GLOBAL T_all (L2-hot,
  this CTA just wrote it -- the same-kernel read-after-global-write
  pattern dqkwg's dq/dk_local tails run on), NOT the Tf tile: Tf's arena
  hosts the B tiles during the loops.  opc/accc/kslc and every
  emap/store_rows sweep retire; the epilogues store u_all/w_all
  global-direct.
* **Lifetimes -> pool 90624 -> 45832 B = 2 CTAs/SM** (kktf pads to
  [C,C+8] f32 18432 hosting the two A-split pairs 18432 EXACT, u phase
  then w phase; Tf hosts the B raw tiles; ksl/qsl/tmpm/vectors own).
  regs 255 -> 158/161, 0 spill.  PTX: ld.shared 832 -> 49, mma.sync
  256 -> 384, ldmatrix 96 -> 384.
* **gc cumsum -> Hillis-Steele** (GDN_WY_GC_SHFL): coalesced parallel
  park, then threads < C each own one element, a 5-step gpu.shuffle-up
  scan per warp (warps 0/1 fully covered by tid<C -- warp-uniform, so
  the shuffle's full-lane convergence holds), warp totals crossed
  through a [2,1] smem vector, lane-31 parks guarded (andi tg,l31) so
  warps 2/3 never write wtot OOB, load index clamped (minsi) so the
  dead-lane smem reads stay in bounds.  DETERMINISTIC (fixed balanced
  tree) but NOT bit-identical to the serial walk: T_all re-rolls at
  4.2e-7 (measured), dk at 1.2e-4 abs -- 6/6 holds, dk margin 4.7x.
  wy +4% on its own (0.2374 -> 0.2278 paired); ON by default.

Solo numbers (each same-window paired): l1 (diet alone) wy 0.7005 ->
0.2277 (3.08x), e2e 4.536 -> 3.896 (1.164x); l1b (+shfl) wy 0.2374 ->
0.2278 (1.04x).

## Lever 3: dhs cross-chunk window prefetch (1.03x/1.04x; the r11 design lands)

The last window (w = NW-1) of chunk c issues chunk c+1's window-0 DMA into
AW[0] (free: w's own reads live on AW[1]; AW[0]'s last reads were fenced by
w-1's trailing barrier -- the same 1-iteration clearance as the in-chunk
rotation), and the cp.async group token rides the %ci scf.for's iter_args
(!nvgpu.device.async.token -- the mnemonic this MLIR build registers;
'async.token' is a parse error) into the next iteration's prologue await.
The last chunk issues a HARMLESS self-duplicate (its own window 0; nobody
reads it) so the carried token is always valid; the loop result is drained
post-loop (no cp.async in flight at CTA exit).  The exposed per-chunk
prologue await now hides behind a full chunk instead of just the B sweep.
**Value-exactness PROVEN**: the full l3-vs-l1b stage-output npz diff is
bit-identical (every global) -- only the DMA issue timing moved.  cp.async
24 -> 30, regs 80 -> 79.  Measured 0.3199 -> 0.3117 (1.03x): the r11
"~0.05-0.1 ms class" estimate was optimistic; the B-sweep overlap was
already hiding most of the prologue.

## Lever 2: dqkwg v4 tails tf32 -> split-bf16 (1.14x stage; dq margin 2.7x TIGHTER -- the 192 tf32 links are gone)

The three tf32 tgemms (dq_dPmk / dk_dPmq / TW -- the r8-counted 192 tf32
k8 tail links) become bf16 mma; tCC/tCKa/tCKb and their cp.async stages
retire (dqkwg cp.async 46 -> 0, ld.shared 1234 -> 71, stack 72B -> 0,
regs 255 -> 240).

* **dq_dPmk**: 2-link mma, A = dPm natural split (_split_bf16 from the
  accCC smem tile), B = raw k^T (stage_raw_T, exact).  tf32's 2^-11 dPm
  rounding -> 2^-17.
* **dk_dPmq**: needs dPm^T as A -- the SAME arena tiles are RE-SPLIT
  transposed (new _split_bf16_T) after the dq loop, B = raw q^T.  The
  two orientations cannot share one split (dq contracts dPm's rows, dk
  its columns) but share the tiles sequentially.
* **TW**: 3-link (T^T splits staged once via the transposing
  stage_split_T + dW^T splits per ks; hh+hl+lh, lo*lo dropped), bse row
  scale folded into the epilogue (same one mulf as the old in-place
  emap).  tf32 was 2^-11 on BOTH operands.
* **Arena discipline**: the dPm splits sit ABOVE accCC in arena1
  (16384+18432 = 34816) -- aliasing them to accCC's own bytes
  self-corrupts the split's source (verify-caught instantly: dq/dk
  1e0-class FAIL, isolated to dk_local in one npz diff; the first fix
  attempt then lost the dke dkB fold to a bad indent -- caught as
  dk_local == park-only at every 32nd block; both fixed).  pthi/bdt
  (earlier phase) and tTh2/tTl2 (later) reuse [0, 18432) after accCC
  dies.  Pool 48640 -> 66048 B: **1 CTA/SM either way** (the 255-reg
  body was reg-capped before and stays reg-capped at 240; the smem
  growth buys the mma operands, not occupancy).
* T512 keeps v3 (r9 ship policy: small grids underfill at any residency)
  -- the v3 tf32 tails remain there, unchanged digits.

## Numerics: 6/6 PASS BOTH shapes; margins IMPROVE

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T2048H16 | **1.824e-4 (44x)** | **1.691e-3 (4.7x)** | 1.088e-5 (736x) | 1.433e-3 (14x) | 4.227e-4 (47x) | 2.761e-6 (2900x) |
| B1T512H4 | 4.617e-4 (17x) | 1.724e-3 (4.6x) | 9.43e-6 (848x) | 3.095e-4 (65x) | 2.085e-4 (96x) | 2.348e-6 |

Digit moves vs r11, all in the TIGHTER direction: dq 4.910e-4 -> 1.824e-4
(the dq_dPmk tf32 2^-11 noise gone), dk 1.710e-3 -> 1.691e-3 (passes
through +7e-6 at the wy split and +1e-6 at the shfl re-roll, then -3e-5
at the TW/tail splits).  dv/dbeta/dg/dh0 unchanged (their paths never
touch the tails; dW_all/dvp_all bit-identical in the l3-vs-l4 diff).

## Determinism: fresh-process cycling per lever, both shapes

| gate | chain | result |
|---|---|---|
| l1 (wy diet) | T2048 x10 / T512 x10 | ALL stage outputs bit-identical |
| l1b (+shfl scan) | T2048 x5 | bit-identical |
| l3 (+dhs XP) | T2048 x5 | bit-identical; l3-vs-l1b VALUE-EXACT (bit-identical globals) |
| l4 (+dqkwg SB) = final | T2048 x5 / T512 x5 | bit-identical |

Knob-off bisect arms: GDN_DHS_XP=0 regenerates the HEAD dhs payload
BYTE-identical; GDN_WY_MMA=0 regenerates HEAD wy modulo ONE redundant
gpu.barrier (value-identical); GDN_DQKWG_SB=0 regenerates HEAD dqkwg with
the tCC alloc reordered (the r11 caryA class -- same tiles, same ops,
different malloc order).  All final chains regenerate bit-identically
from the committed generator (r12_final_T{2048,512} verified).

## Honest negatives

* **bpre 0.80x in the final T2048 window** on a byte-identical cubin
  (the documented co-job swing; other windows measured 0.0515).
* **dhs cross-chunk prefetch is worth 1.03x, not the estimated
  0.05-0.1 ms** -- the B-sweep overlap was already hiding the prologue
  await; the residual win is real but small.
* **dqkwg pool grows 48640 -> 66048 B** (the dPm/T/dW split operands
  need arena1 at 34816): theoretical 2-CTA smem eligibility is gone, but
  the kernel was reg-capped at 255 regs (1 CTA) before and lands at 240
  (1 CTA) -- no occupancy change, and the 72B stack leak is gone.
* **Two verify-caught bugs on the dqkwg port** (the accCC self-alias
  split; the dke fold lost to indentation) -- both fixed before any
  timing was recorded; documented above so the next round knows the
  arena rules: a split must NEVER alias its own source tile.
* **stage-delta accounting leaves ~0.2 ms** of the e2e med gain beyond
  the per-stage medians (window noise; e2e runs themselves tight).

## Ranking after round 12 (round-13 scope)

1. **wyb 0.82** -- now the largest stage (the dA-chain twin; accCC
   retirement / tf32-tail diet toward <= 50176 B = the r11 #4 carried
   forward, now the top single-stage prize).
2. **dqkwg 0.94 v4 body** -- the scalar r4/c4r/term_P global-P loops and
   the dW/dq1 merged mma chains; the tf32 storm is gone, the remainder
   is staging + the read-add-store round trips.
3. **fwdscan 0.373 / caryM 0.160 polish**.
4. **dkb 0.248 / dhs remainder 0.31**.

Repro: python tools/gdn_bwd_gen.py --shape ... --out DIR (knobs
GDN_WY_MMA/GDN_WY_GC_SHFL/GDN_DHS_AW/GDN_DHS_BKW/GDN_DHS_XP/GDN_DQKWG_SB/
GDN_DQKWG_V4, all default on); verify tools/gdn_bwd_verify.py --gen-root
DIR --shape ...; determinism evidence/G5/scratch/r12_det_fresh.sh DIR
<shape> <tag> [N]; paired bench evidence/G5/scratch/r12_ab_bench.sh
<base> <new> <shape> <tag>.  Fixed chains: r12_final_T2048 (== r12_l4),
r12_final_T512.


# G5-T round 13: the wyb split-bf16 diet (2 CTAs/SM), caryM mma, dqkwg P-stage

Artifacts: tools/gdn_bwd_gen.py (gen_wyb GDN_WYB_DIET full-port dispatcher +
_legacy fallback; gen_caryM GDN_CARYM_MMA k16-sliced split mma; _resplit_pair_T
helper; mma_gemm epi_bar= fence param; _gen_dqkwg_v4 GDN_DQKWG_PST P restage),
tools/gdn_smem_fix.py (aliased-tile "0 mallocs = optimized away" skip),
chains evidence/G5/scratch/r13_{base,l1,l2,l3(final)}_T2048 +
r13_{base,l3(final)}_T512 (l1 = wyb diet alone, l2 = +caryM, l3 = +dqkwg
P-stage = the round-13 final), r13_logs/ (interleaved paired bench jsons x3
per comparison + det npz sets) + r13_{build_all,det_fresh,ab_bench}.sh
helpers, this section.  Baselines rebuilt fresh from git HEAD b5b9150 at
round start.

## Headline: bwd e2e 3.326/3.316 -> 2.688/2.674 @B1T2048H16 (1.237x med / 1.240x min); T512 1.067x med / 1.090x min

ours/fla bwd = 1.03/2.688 = **0.383x** (was 0.276x).  All ratios
same-window interleaved pairs x3 (base,new,base,new,... 30-iter CUDA-event
medians), co-job resident throughout.

| B1T2048H16 | HEAD b5b9150 | round13 | ratio | note |
|---|---|---|---|---|
| wy | 0.2255 | 0.2251 | = | untouched |
| fwdscan | 0.3771 | 0.3763 | = | untouched (design notes below) |
| caryA | 0.0574 | 0.0545 | = | untouched (noise band) |
| caryM | 0.1631 | 0.0833 | **1.96x** | lever 2 (split mma + 2 CTAs) |
| bpre | 0.0565 | 0.0525 | = | untouched (noise band) |
| dhs | 0.3131 | 0.3137 | = | untouched |
| dkb | 0.2466 | 0.2473 | = | untouched |
| dqkwg | 0.9475 | 0.9369 | 1.01x | lever 3 (P-stage; noise band) |
| wyb | 0.8195 | 0.3419 | **2.40x** | lever 1 (the diet) |
| **e2e bwd** | **3.3265 / 3.3164** | **2.6881 / 2.6735** | **1.237x med / 1.240x min** | |

Stage-delta sum: wyb -0.478 + caryM -0.080 + dqkwg -0.011 - caryA/bpre
-0.007 = **-0.575 ms** vs e2e med delta -0.638 ms (residual ~0.06 ms =
window noise; e2e runs tight at 2.6735-2.7163).  T512 (final window):
wyb 0.0995 -> 0.0599 (1.66x), caryM 0.0291 -> 0.0207 (1.41x), e2e
1.067x med / 1.090x min (launch-noise-dominated as ever; bpre/caryA
"1.1x" cells there are the same class of window noise as r11/r12's).

## Lever 1: wyb -- the full split-bf16 port + two-arena diet (2.40x/1.66x; the r11 #4 / r12 #1 prize lands)

Every tf32/kahan gemm site (G_dvp / G_dW / X / Y / dk_t1 / dk_t2 / TW_rb
-- the r8-counted tf32 tails) becomes 2-3-link split-bf16 mma; accCC (the
kkt staging) and every cp.async/emap staging tile retire; pool 100096 ->
45056 B at exactly 128 regs = **2 CTAs/SM x 256 thr** (was 1 CTA, 226
regs).  ld.shared 2607 -> (diet payload), cp.async 92 -> 0.

* **G = tril(dvp (beta v)^T + dW (beta k 2^gc)^T)**: register-chained
  2-link mma per slice -- A = dvp/dW natural splits (stage_split_ew from
  the f32 globals), B = the RAW bf16 v/k natural slice (stage_raw_bf16,
  exact); beta[col] (and beta*2^gc[col]) fold at the chain epilogue (the
  r12 dqkwg TW move), tril-select per contribution (select is linear, so
  per-term select == select of the sum, exactly).  The dW-phase epilogue
  read-add-resplits the dvp-phase splits in place (single lane owns each
  element).
* **X^T = T G^T, then Y = T^T X**: the sandwich never materializes f32 --
  the X-chain's epilogue split-stores X^T straight over its own dead
  B-source (sN), and the Y-chain's epilogue folds dAf = -tril(Y *
  2^(gc_i - gc_j)) the same way.  A-side T is staged NATURAL for X^T and
  re-staged T^T for Y (one 16 KB global re-read; aT's arena window).
  K=64 chains run as k16-sliced links (accs= chaining, the r12 wy
  fragment-live-range diet).
* **dk loop (BKD=16)**: dk_t2 first (A = dAf^T re-splits of the sN pair
  via the new _resplit_pair_T with beta folded on the contraction index,
  B = raw k^T), t2 parked as a bf16 pair; then dk_t1 (A = dAfN splits,
  same k^T B) whose epilogue folds the final dk = dk_local + t1*beta +
  t2 store (the r12 reorder note: addf commutes, values exact).
  **LLVM store-to-load-forwards the t2 handoff into registers** (same
  lane map on both chains) -- the two t2 mallocs vanish from the PTX and
  gdn_smem_fix now SKIPS aliased tiles with 0 mallocs instead of
  disabling the whole spec (the bf16 rounding chain survives; values
  identical).
* **ra/c4r -> warp-butterfly sums** reading kkt straight from the
  L2-hot global (the accCC staging retires); expg/bgc-class hoists kept.
* **TW_rb / tv**: the r11 tv 3-mma body moves into the arena plan
  verbatim (A = the T^T splits re-staged after dk; dvp^T splits + tvacc
  in the freed arena2); TW = 3-link (T^T splits x dW^T splits), rowsum
  reads v/k from globals.
* **Arena map** (arena1 = aT/dT/dwA windows 18432, arena2 = sN + dk-tail
  24832): every epilogue that overwrites a tile the same chain reads is
  fenced by mma_gemm's new epi_bar= barrier (the r6 hazard class); every
  restage keeps the r10 trailing-fence discipline.
* 128 regs exactly (2x256x128 = the 64K regfile; 176 B spill stores /
  228 B spill loads -- mild, kept: the alternative reg bump loses the
  2-CTA residency).

Solo: l1 (wyb diet alone) wyb 0.8172 -> 0.3395 (2.41x), e2e 3.322 ->
2.847 (1.167x med).

## Lever 2: caryM -- tf32 tgemm -> k16-sliced split mma (1.96x/1.41x)

The single caryM.m site ([BK2,C,C], at=bt=neg=acc) becomes a 3-link
split mma with BOTH operands sliced straight from the globals: A = W^T
splits via the transposing stage with **flat col0 = i0 + kb*K (the col0
arithmetic IS the k16 slice)**, B = P-natural splits via col0 = kb; mb's
init sweep and read-subtract epilogue keep the f32 init values exact.
pt/wb retire; 160 -> 76 regs, 0 spill; pool 64256 -> 45504 B = 2 CTAs/SM.
Solo: caryM 0.1647 -> 0.0814 (2.02x).

## Lever 3: dqkwg P-stage (1.01x -- HONEST: noise band, kept as value-identical)

r4/c4r's P factor re-staged from the global into the dhi/dlo arena
window (free until the dPm splits): P VALUES identical (same f32
elements, same lane/tree) -> r4/c4r bit-identical, verified (l3 digits ==
l2 digits exactly).  Measured 0.9444 -> 0.9361 (1.01x, inside the 2% "="
band): **the r9 L2-hot parking was already sufficient**; kept ON because
it is free and value-identical, documented as NOT a verified win.  The
deeper v4-body items (the dW/dq1 merged-loop staging redundancy, the
dq/dk_local read-add-store round trips, 205 bar.sync) were analyzed and
deliberately left -- see honest negatives.

## Numerics: 6/6 PASS BOTH shapes; margins IMPROVE

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T2048H16 | 1.824e-4 (44x) | **1.284e-3 (6.2x)** | 1.088e-5 (736x) | **8.527e-4 (23x)** | **3.274e-4 (61x)** | **2.169e-7 (36900x)** |
| B1T512H4 | 4.617e-4 (17x) | **5.131e-4 (15.6x)** | 9.43e-6 (848x) | **2.178e-4 (92x)** | **1.466e-4 (137x)** | **1.394e-7 (57400x)** |

Digit moves vs r12, all TIGHTER: dk 1.691e-3 -> 1.284e-3 @T2048 and
1.724e-3 -> 5.131e-4 @T512 (the seven tf32 2^-11 sites -> 2^-17 splits);
dbeta/dg tighten with them (ra/c1r re-roll onto warp trees + the tighter
dAf); dh0 2.761e-6 -> 2.169e-7 (caryM's M_all -> carry chain tightens).
dq/dv digits unchanged (their paths never touch the three levers).

## Determinism: fresh-process cycling, both shapes, per chain

| gate | chain | result |
|---|---|---|
| l1 (wyb diet) | T2048 x5 / T512 x5 | ALL stage outputs bit-identical |
| l2 (+caryM) | T2048 x5 / T512 x5 | bit-identical |
| l3 (+dqkwg P) = final | T2048 x5 / T512 x5 | bit-identical; l3-vs-l2 VALUE-EXACT (dqkwg digits identical) |

Knob-off bisect arms: GDN_WYB_DIET=0 and GDN_CARYM_MMA=0 regenerate the
HEAD payloads BYTE-identically (checked vs r13_base payloads);
GDN_DQKWG_PST=0 differs only by the unused pst alloc (the r11 alloc-order
class; PTX-equivalent).  The final chains regenerate bit-identically from
the committed generator (md5-checked for wyb/caryM/dqkwg).

## Honest negatives

* **dqkwg v4 body largely unattempted** beyond the P-stage: the dW/dq1
  merged loop restages the dvp splits per (ks, vs) (4x redundant), the
  dq/dk_local parks + read-add-stores round-trip through L2, 205
  bar.sync; the loop-swap needed to kill the dvp redundancy blows up the
  register accs.  dqkwg 0.937 is now 35% of the whole bwd -- round-14 #1.
* **fwdscan not attempted**: the kg gemm's operand is [C, K]-wide (the
  [*, 136] bf16 pair = 36864 B) and S (32 KB f32) + accCK + gsc already
  pin the kernel over the 2-CTA threshold -- the port needs the per-half
  N-split design (two N=64 chains with B-half splits emitted by the kg
  sweep) sketched this round; S must stay f32 (bf16-splitting the
  recurrence state would COARSen h_all/vnew after 32 chunks -- the r7
  value-class warning, in the tightening direction everywhere else but
  compounding here).  fwdscan 0.376 = round-14 #2.
* **wyb spills**: 176 B stores / 228 B loads at exactly 128 regs.  The
  regfile binds 2x256x128 = 65536 exactly; a 132-reg variant would need
  1 CTA.  Kept the spills.
* **the P-stage is not a win** (1.01x, noise band) -- kept only because
  value-identical and free.
* **two new toolchain lessons, both loud-not-silent**: (1) an arena HOST
  alloc with no uses is DCE'd by LLVM and the size key finds 0 mallocs --
  arena bases must be first-USE tiles; (2) LLVM store-to-load-forwards
  same-lane epilogue handoffs (the t2 pair) straight into registers --
  gdn_smem_fix now skips such tiles with a NOTE instead of disabling
  aliasing (previously a hard WARN + a 121 KB static-smem ptxas failure).
* stage-vs-e2e accounting residual ~0.06 ms (window noise; the co-job
  stayed resident the whole round, no clean-window blocking).

## Ranking after round 13 (round-14 scope)

1. **dqkwg 0.937** -- now 35% of bwd e2e: the dW/dq1 loop restructure
   (dvp-split restage redundancy, park/RAS round trips, barrier diet
   205), the r4/c4r term reads, reg 240 cap.
2. **fwdscan 0.376** -- the per-half N-split split-bf16 port (design
   above; S stays f32).
3. **dhs 0.314 / dkb 0.247 / wy 0.225**.
4. **wyb 0.342 polish** (the 176 B spills; the dg suffix walk) / caryM
   0.083 residual.

Repro: python tools/gdn_bwd_gen.py --shape ... --out DIR (knobs
GDN_WY_MMA/GDN_WY_GC_SHFL/GDN_DHS_AW/GDN_DHS_BKW/GDN_DHS_XP/GDN_DQKWG_SB/
GDN_DQKWG_V4/GDN_WYB_DIET/GDN_CARYM_MMA/GDN_DQKWG_PST, all default on);
verify tools/gdn_bwd_verify.py --gen-root DIR --shape ...; determinism
evidence/G5/scratch/r13_det_fresh.sh DIR <shape> <tag> [N]; paired bench
evidence/G5/scratch/r13_ab_bench.sh <base> <new> <shape> <tag>.  Fixed
chains: r13_l3_T2048 (== r13_final_T2048), r13_l3_T512 (==
r13_final_T512).

# G5-T round 14 (R14-V): fwdscan NC-proof prologue (T8192 chain BUILDABLE) + the dqkwg round-trip/barrier diet -> 512-thread conversion

Date: 2026-10-08 | clean idle window discipline (0 compute apps checked
before every bench; coordination row `gdn-mlir R14-V`).  Baselines
rebuilt fresh from git HEAD e987da5 at round start (bwd payloads
byte-identical to the r13 final; the fwd-side commits 10c673e/2a2b382
do not touch the bwd generator).  Artifacts: tools/gdn_bwd_gen.py
(GDN_FSCAN_GSCG hybrid + GDN_DQKWG_DIET + the GDN_DQKWG_THR=512
default; mma_gemm_mh W<=16; stage helpers / mma_chain_if bar= params),
tools/gdn_bwd_verify.py + tools/gdn_det_check.py (gsc_all in
OUT["fwdscan"], skip-if-absent), chains evidence/G5/scratch/
r14v_{base,final}_{T2048,T512} + r14v_final_T8192 + r14v_l1_T8192
(fwdscan-lever solo), r14_logs/ (interleaved paired bench jsons x3 +
det npz sets + T8192 benches), r14_{build_all,ab_bench,det_fresh}.sh,
r14v_prof.py (ncu chain launcher).  cubin md5: fwdscan T2048
base==final e680e1d94e47 (the hybrid keeps HEAD), T8192 304b4259a79d;
dqkwg T2048 6c3042981c2e -> 49573b09264f, T512 7de32d2a07c3 (HEAD v3),
T8192 641ddec6c6fd.

## Headline: bwd e2e 2.690/2.681 -> 2.389/2.378 @B1T2048H16 (1.126x med / 1.128x min); T512 1.00x; T8192 chain BUILDABLE at 8.670/8.611 ms

ours/fla bwd = 1.03/2.389 = **0.431x** (was 0.383x).  All ratios
same-window interleaved pairs x3 (base,new,base,new,... 30-iter
CUDA-event medians), co-job idle throughout (0 compute apps at every
pre-run check).

| B1T2048H16 | HEAD e987da5 | round14 | ratio | note |
|---|---|---|---|---|
| wy | 0.2232 | 0.2258 | = | untouched (noise band) |
| fwdscan | 0.3751 | 0.3761 | = | lever 1 (hybrid: HEAD-identical payload at this shape BY DESIGN) |
| caryA | 0.0532 | 0.0544 | = | untouched (noise band) |
| caryM | 0.0812 | 0.0834 | = | untouched (noise band) |
| bpre | 0.0552 | 0.0514 | = | untouched (noise band) |
| dhs | 0.3122 | 0.3125 | = | untouched |
| dkb | 0.2441 | 0.2469 | = | untouched |
| dqkwg | 0.9405 | 0.6420 | **1.46x** | lever 2 (diet 1.28x + 512-thr stack) |
| wyb | 0.3413 | 0.3408 | = | untouched |
| **e2e bwd** | **2.6902 / 2.6813** | **2.3893 / 2.3780** | **1.126x med / 1.128x min** | |

Stage-delta: dqkwg -0.299 ms vs e2e med delta -0.301 ms (the whole
gain is the one stage; everything else inside the 2-3% window-noise
band).  T512: every payload byte-identical to HEAD (fwdscan hybrid
picks the smem arm; dqkwg v3 by the small-grid policy) -> the A/B
pair is a base-vs-base noise calibration, 0.998x.

## Lever 1: fwdscan NC-proof prologue -- the T8192 chain is BUILDABLE (fwdscan 1.487 ms, e2e 8.670/8.611 ms, 6/6)

The v3 prologue parked all NC per-chunk gc-cumsum rows in a [NC, C]
f32 smem tile: fwdscan pool 98560 B at T2048 -> **123136 B at T8192 >
the 101376 B static limit (chain UNBUILDABLE; the auxiliary v2 chain
timed 11.42 ms non-flagship)**.  GDN_FSCAN_GSCG ships a HYBRID:

* while the static pool still fits, the smem tile stays EXACTLY v3
  (T2048/T512 payloads byte-identical to HEAD -- zero regression by
  construction, cubin md5 e680e1d94e47 on both sides of the A/B);
* past the limit the rows move to a new @gsc_all [M, C] f32 global:
  the same one-thread-per-row serial-order f32 chain carried in an
  scf.for iter_arg (register-resident, no read-back -> gc values
  bit-identical), consumed through a 2-row cp.async ping-pong (gr,
  512 B) whose wait lands at the chunk top under the previous chunk's
  compute (the dhs-XP token-riding pattern; unconditional 4 B
  cp.async.ca per thread -- the benign self-duplicate class, NOT
  bypassL1: 4 B copies must lower to .ca).  The NV sibling CTAs of
  one bh write the same bytes with the same values (benign; each CTA
  reads only its own rows after its own bar.sync).  Pool at T8192:
  123136 -> 90880 B.
* S stays the f32 [K, BV] recurrence state in both arms (splitting it
  would coarsen h_all/vnew -- only the PROLOGUE park moves).

HONEST negative: the global arm measured 0.96x (0.374 -> 0.388) when
force-enabled at T2048 -- the prologue's scattered 4 B global stores
cost ~15 us and the cp.async prefetch did NOT recover it (read
latency was already hidden).  That measured regression is WHY the arm
picks itself only where the smem tile cannot fit; the flagship shapes
keep HEAD.  T8192 (NC=128, global arm): fwdscan 1.4741 ms on the
lever-only chain (r14v_l1_T8192, HEAD dqkwg), 1.4872 ms on the final
chain; **T8192 bwd e2e 8.6701/8.6110 ms -- the first v3-path build,
1.32x vs the 11.42 ms v2 fallback**, 6/6 numerics (dq 2.078e-4 38x,
dk 1.881e-3 4.3x, dv 1.050e-5 762x, dbeta 1.654e-3 12x, dg 1.145e-3
17x, dh0 1.965e-7 40700x).

## Lever 2: dqkwg 0.940 -> 0.642 (1.46x): the diet (1.28x) + the 512-thread stack

GDN_DQKWG_DIET (all value-identical -- every digit of the 6/6 table
below is UNCHANGED from r13):

* **V-wide slabs for the dW/dq1 (ks, vs) loop** (the r13 sketched
  restage-redundancy kill): dvp splits (svh/svl [C+1/C+3, V+8], staged
  ONCE instead of 16x), h splits (hbh/hbl [BK+2/BK+6, V+8], per-ks 4x
  instead of per-(ks,vs) 16x), raw dO (dbo [C, V+8], once instead of
  16x).  The mma links read slab columns at k0=vs (the existing
  6-tuple ops) -- same operand VALUES, same link order/accumulation
  -> bit-identical products.  The slabs alias arena1 grown 34816 ->
  52224 (the dead window between pt/dvp and TW); pool 66048 -> 100864
  B (reg-capped 1 CTA either way).
* **R-A-S folds into mma epilogues** (same f32 op ORDER ->
  bit-identical): dq_dPmk/dk_dPmq park straight from the link
  epilogues (the accCK hop + store_rows sweep + 2 barriers x8
  retire); dq[ks] += accCK folds into epi_dq1; dk_local = TW.*bse +
  dPmq + dkB folds into epi_tw (accCK dies with TW).
* **Barrier diet** (the r7 disjoint-tile rule + read-only chain
  merging): 206 -> 112 bar.sync.

Diet solo (GDN_DQKWG_THR=256): dqkwg 0.9371 -> 0.7348 (1.28x), e2e
2.6903 -> 2.4858 (1.082x).  Resources: **240 -> 128 regs** (16 B
spill).

**The 512-thread conversion (GDN_DQKWG_THR default 512)**: the diet's
128-reg landing makes 512x128 = 65536 = the whole regfile legal (the
256-thr HEAD body sat at 240 regs -> 512-thr would not even launch).
1 CTA/SM x 16 warps = 4 warps/scheduler (was 2).  ncu on the diet
cubin (r14v_prof.py): 90.3% no-eligible, 1.97 active
warps/scheduler, SM 18% / MEM 19% -- pure latency.  The mma maps
extend: mma_gemm_mh W<=16 (the map is generic in halves; the
fragment/lane math is warp-local), and the dPr links switch to the mh
map when W>8 (the strip map caps at N/8 warps).  Same fragment
values, same kb accumulation, same warp-tree sums -> **every 6/6
digit unchanged**.  128 regs / 44 B spill at the 512-thr unroll.  dqkwg
0.7348 -> 0.6420 (**1.46x total vs HEAD**), e2e 2.690 -> 2.389
(1.126x).

## Numerics: 6/6 PASS BOTH shapes; digits IDENTICAL to r13 (a value-exact round)

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| B1T2048H16 | 1.824e-4 (44x) | 1.284e-3 (6.2x) | 1.088e-5 (736x) | 8.527e-4 (23x) | 3.274e-4 (61x) | 2.169e-7 (36900x) |
| B1T512H4 | 4.617e-4 (17x) | 5.131e-4 (15.6x) | 9.43e-6 (848x) | 2.178e-4 (92x) | 1.466e-4 (137x) | 1.394e-7 (57400x) |

T2048 digits == r13 exactly (both levers value-exact); T512 payloads
byte-identical to HEAD (inherited).  T8192 6/6 (table in lever 1).

## Determinism: fresh-process cycling, bit-identical

| chain | runs | result |
|---|---|---|
| final T2048 (diet + 512thr; fwdscan HEAD arm) | x5 | ALL stage outputs bit-identical |
| final T512 (== HEAD payloads) | x5 | bit-identical |
| final T8192 (fwdscan global arm + 512thr) | x3 | bit-identical |

Knob-off bisect arms: GDN_FSCAN_GSCG=0 and GDN_DQKWG_DIET=0 +
GDN_DQKWG_THR=256 regenerate the HEAD payloads byte-identically
(checked fwdscan @T2048/T512, dqkwg @T2048); GDN_DQKWG_THR=512
WITHOUT the diet cannot launch (240 regs x 512 threads > 65536) --
the diet is the 512-thr enabler.

## Honest negatives + process lessons

* **a bogus flat A/B almost buried the diet**: an early l2 A/B read
  dqkwg 1.00x -- that chain dir had been assembled with
  `cp -r base/* new/` AFTER building the variant, which OVERWROTE the
  variant dqkwg with the base cubin (the l2 cubin md5 == base).  The
  same clobber faked the first 512-thr probe.  Caught by cubin md5
  when the final fresh-build A/B disagreed; the corrected solo
  numbers: diet 1.28x, diet+512thr 1.46x.  LESSON (loud): assemble
  variant chains by copying the UNCHANGED stages only, and md5-compare
  every A/B pair before believing a flat result.
* **fwdscan@T2048 not improved**: the hybrid deliberately keeps HEAD
  there (the global-arm 0.96x measured regression); the 2-CTA
  split-bf16 kg port (r13 sketch) remains untouched.
* **T512 untouched by construction** (v3 dqkwg by the small-grid
  policy; fwdscan smem arm).
* dqkwg 512-thr: 44 B spills at exactly 128 regs; pool 100864 B sits
  512 B under the static cap (fine at 1 CTA, but the r15 2-CTA push
  must shrink, not grow, the pool).
* the diet's slab half is orthogonal to the timing win (the 1.28x is
  folds + barrier diet + reg landing; staging was never the bound) --
  kept for the structural simplification + the 128-reg enable.

## Ranking after round 14 (round-15 scope)

1. **dqkwg 2-CTA/SM (<= 50176 B)**: the remaining 0.642 ms is still
   27% of bwd.  The 512-thr arm reaches 16 warps but ONE instruction
   stream; 2 x 256-thr CTAs = independent streams.  The diet-minus-
   slabs pool is 66048; with accCK retired (r3 term reads the dq1
   partial from a global park) + bhh/bhl aliased into arena1's dead
   dvp window: ~52736 = **2560 B over** the 50176 bar -- a pt diet /
   krt+qrt window reuse closes it.  The 128-reg landing (this round)
   is the other half of the prerequisite.
2. **fwdscan 0.376**: the split-bf16 kg port (two N=32 chains, B-half
   splits from the kg sweep; the r13 sketch) toward 2 CTAs/SM; the
   global prologue arm now exists for T8192+.
3. **dhs 0.311 / dkb 0.247 / wyb 0.341 polish** (dqkwg drops to #3 of
   the stage table after the diet).
4. **T8192 fleet**: fwdscan 1.487 / dqkwg 2.294 / dhs 1.176 dominate;
   port the 512-thr/diet lessons there (dhs is the next 1-CTA latency
   suspect).

Repro: python tools/gdn_bwd_gen.py --shape ... --out DIR (knobs
GDN_WY_MMA/GDN_WY_GC_SHFL/GDN_DHS_AW/GDN_DHS_BKW/GDN_DHS_XP/GDN_DQKWG_SB/
GDN_DQKWG_V4/GDN_WYB_DIET/GDN_CARYM_MMA/GDN_DQKWG_PST/GDN_FSCAN_GSCG/
GDN_DQKWG_DIET/GDN_DQKWG_THR, all default on / 512); verify
tools/gdn_bwd_verify.py --gen-root DIR --shape ...; determinism
evidence/G5/scratch/r14_det_fresh.sh DIR <shape> <tag> [N]; paired
bench evidence/G5/scratch/r14_ab_bench.sh <base> <new> <shape> <tag>.
Fixed chains: r14v_final_T2048, r14v_final_T512, r14v_final_T8192
(baselines r14v_base_T2048/T512; fwdscan solo r14v_l1_T8192).

# G5-U round 15 (R15-W): dqkwg 2-CTA/SM lands -- pool 100864 -> 47616 B, .maxnreg 128, dqkwg 0.632 -> 0.547 @T2048 and 2.323 -> 1.844 @T8192

Date: 2026-10-08 | clean idle window discipline (0 compute apps checked
before every bench, incl. the r15w_ab_bench.sh hard refusal guard;
coordination row `gdn-mlir R15-W`).  Baselines rebuilt fresh from git
HEAD 4d5916d at round start; the base cubin md5s reproduce the r14
record exactly (dqkwg T2048 49573b09264f, T8192 641ddec6c6fd) -- the
rebuilt-vs-record check the r14 clobber lesson mandates.  Artifacts:
tools/gdn_bwd_gen.py (GDN_DQKWG_2CTA default on), chains
evidence/G5/scratch/r15w_{base,2cta,final}_* + r15w_2cta_*_dqkwg,
r15_logs/ (interleaved paired jsons x3 per pair + det npz sets + final
singles), scripts r15w_{build,assemble,ab_bench,det,ncu,regs,ptxas}.sh.
New cubin md5: dqkwg T2048 49573b09264f -> f618413ac53a, T8192
641ddec6c6fd -> b42395cf722e; every other stage byte-identical
(assemble script md5-verifies each pair).

## Headline: bwd e2e 2.3794/2.3688 -> 2.2913/2.2796 @B1T2048H16 paired (1.039x med / 1.041x min); T8192 8.822/8.778 -> 8.399/8.333 (1.050x / 1.053x); T512 untouched

ours/fla bwd = 1.03/2.2887 = **0.450x** (was 0.431x).  Final singles:
T2048 2.2887/2.2841 (dqkwg 0.5464), T8192 8.2669/8.2477 (dqkwg 1.8281),
T512 0.5271/0.5224 (== HEAD payloads; the A/B is a base-vs-base
calibration, 0.998x).

| B1T2048H16 | HEAD 4d5916d | round15 | ratio | note |
|---|---|---|---|---|
| dqkwg | 0.6323 | 0.5466 | **1.16x** | the 2-CTA conversion (lever 1) |
| all 8 others | -- | -- | = | inside the 1-2% noise band, every pair |
| **e2e bwd** | **2.3794 / 2.3688** | **2.2913 / 2.2796** | **1.039x / 1.041x** | |

Stage-delta: dqkwg -0.086 ms vs e2e med delta -0.088 ms (the whole gain
is the one stage).  T8192: dqkwg 2.3232 -> 1.8443 (**1.26x**, e2e med
delta -0.423 of -0.423) -- the 2-CTA win is BIGGER at T8192 (2048 CTAs
fill the doubled SM slots; waves 6.24 -> 3.12).

## Lever 1: the 2560B battle resolved WITHOUT the global park -- akb/accCK/bhh/bhl all alias into arena1 (pool 47616, 3560B UNDER the bar)

The r14 sketch (accCK retired to a global park + bhh/bhl aliased ->
52736, 2560 over) is improved on: the peak-lifetime audit shows FOUR
tiles fit arena1's existing 34816 window, so accCK STAYS in smem (r3's
rowsum keeps its LDS read -- no L2 round trip, no new global):

* diet FOLDS kept (park_epi, epi_dq1 RAS, epi_tw dk_local fold -- the
  r14-shipped value-exact epilogues), V-wide SLABS dropped: arena1 back
  to 34816, the dW/dq1 loop returns to the per-(ks,vs) restage arm
  (same operand VALUES, same link order/accumulation -> bit-identical
  products).
* alias remap (zero new barriers, zero global traffic): bhh@13824,
  bhl@16384 (+1/+3 row pads -- the smem_fix size-identification key
  needs unique alloc sizes; pads never addressed), akb@18944 (the
  dPr/dvp/dq1 raw dO/k A-tile), accCK@24064 (the dq1 partial, [C,BK]
  f32 -- smem-resident as before).
* the phase audit (every boundary barrier-fenced): P2 dPr accCC[0,16K)+
  akb[18.9K,24K) (accCC written only in the LAST vs epilogue --
  disjoint bytes); P3 accCC+pst@16384; P4-7 accCC+dhi+dlo; P9 dvp
  pthi@0+bdt@9216+bhh+bhl+akb = 24064; P11 dW/dq1 bhh+bhl+akb+
  accCK[24K,32.2K) (vnh/vnl stay arena2); P12 TW tTh2+tTl2 (epi_tw
  folds -- no accCK).  Everything under 34816.
* pool = arena1 34816 + arena2 10240 + 10 x 256B scalars = **47616 <=
  50176** with 3560B headroom (no slab variant fits the headroom; the
  smallest useful one, hbh/hbl per-ks slabs, needs 17424).
* P11 barrier diet (the r7 disjoint rule): the three restage sweeps
  (akb / vnh+vnl / bhh+bhl -- three disjoint windows) drop to two
  barriers (akb rides the split stages' barrier) + dw chain bar=False
  (dq1 only reads the same tiles; its own bar fences the restage).
  Measured = noise (kept for structure; 0.547 -> 0.550 same-band).

## THE enabling find: the shipped cubin was 168 regs, not 128 -- ptxas_v.txt reads the WRONG pass

The r14 build greps ptxas_v.txt -- that file is run_l2.sh's FIRST
ptxas pass (kernel.ptx -> kernel.cubin, pre-smem_fix).  The SHIPPED
kernel_smemfix.cubin comes from gdn_smem_fix.py's SECOND ptxas pass,
which frees ptxas to a different schedule: the 2CTA body lands **168
regs** there (0 spills) while the first pass reports 128.  The first
A/B caught it: dqkwg REGRESSED 0.78x (0.811) -- ncu showed Block Limit
Registers = 1 (2x256x168 > 65536): 8 warps/SM, strictly worse than the
r14 16-warp form.  TWO toolchain lessons, loud:

* the register number that matters is the smem_fix pass's; ptxas_v.txt
  must not be trusted for it (r14's "128 regs / 44B spill" record was
  the first pass's; the r14 SHIPPED cubin really is 128 -- verified
  this round by re-running the fix pass on the base: 128/0-spill).
* ptxas-blackwell IGNORES --maxrregcount (GDN_MAXREG re-verified
  useless: 168 regs at cap 128), but the PTX **.maxnreg directive
  placed inside the entry body (next to .maxntid) IS honored**: 128
  regs / 32B spills (64B after the barrier diet).  r15w_build.sh
  injects it post-smem_fix (GDN_MAXNREG_ENTRY=dqkwg:128) and recompiles
  -- the repro path for the round's cubins.

With 2x256x128 = 65536 legal: ncu Block Limit Registers = 2, Block
Limit Shared Mem = 2 -- **2 CTAs/SM achieved** at both shapes (achieved
occupancy 15.02% -> 29.17% @T2048, 14.23 warps/SM @T8192 vs 8.00
before; no-eligible 89 -> 83%, active warps/scheduler 1.97 -> 3.63,
SM 18 -> 30%, MEM 19 -> 36% -- the independent-streams thesis of the
r14 ranking, measured).

## Numerics: 6/6 at ALL THREE shapes; digits IDENTICAL to r14 (value-exact round)

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| T2048 | 1.824e-4 (44x) | 1.284e-3 (6.2x) | 1.088e-5 (736x) | 8.527e-4 (23x) | 3.274e-4 (61x) | 2.169e-7 (36900x) |
| T512 | 4.617e-4 (17x) | 5.131e-4 (15.6x) | 9.43e-6 (848x) | 2.178e-4 (92x) | 1.466e-4 (137x) | 1.394e-7 (57400x) |
| T8192 | 2.078e-4 (38x) | 1.881e-3 (4.3x) | 1.050e-5 (762x) | 1.654e-3 (12x) | 1.145e-3 (17x) | 1.965e-7 (40700x) |

Every digit equals the r14 tables (T2048 == r13 == r14; T8192 == the
r14 global-arm table).  T512 payloads byte-identical to HEAD (all 9
stages, diff -r clean).  Knob-off bisect: GDN_DQKWG_2CTA=0 regenerates
the HEAD dqkwg payload.mlir AND smem_alias.json byte-identically.

## Determinism: fresh-process cycling, bit-identical

| chain | runs | result |
|---|---|---|
| final T2048 (2CTA + folds, 256thr) | x5 | ALL stage outputs bit-identical |
| final T8192 (2CTA + fwdscan global arm) | x3 | bit-identical |
| final T512 (== HEAD payloads) | -- | inherited (r14 x5) |

## Honest negatives + the fwdscan verdict (round-15 #2 CLOSED as structurally unreachable)

* **fwdscan 2-CTA is NOT reachable with this kernel structure**: the
  pool math is decisive -- S [K,BV] f32 (16384B) + wpp [C,2BK] f32
  (32768B) ALONE exceed the 50176B bar, and split-bf16 saves no bytes
  on f32-VALUED operands (hi+lo bf16 == f32 width; the r12/r13 sketch
  premise "halves the staging" holds only for bf16-EXACT sources --
  k is the one such operand and it is ALREADY staged bf16 as kpp).  A
  real 2-CTA fwdscan needs a K-split-across-CTAs redesign with a
  cross-CTA reduction of S -- a new numerics class (coarsened state
  handoff), out of single-round scope.  The numerics-only port
  (tf32->split-bf16 kg, ~2^-11 -> 2^-17) remains possible but changes
  h_all/vnew digits chain-wide for an estimated 0.02-0.03 ms on a
  0.376 ms stage -- poor risk/reward this round.
* the first 2CTA A/B WITHOUT the reg fix read 0.78x (the 168-reg cubin
  at 1 CTA) -- kept in r15_logs/l1_2cta_T2048 as the negative control.
* dhs profiled (ncu): 65540B / 1 CTA but SM 48.9% = MEM 48.9% --
  throughput-bound, NOT the latency picture that paid 1.46x for dqkwg;
  a -15364B pool diet is the round-16 ask, with a smaller expected
  multiple.
* dqkwg is still 24% of bwd (0.547/2.291); 64B spills at 128 regs; the
  P11 vs-only restage redundancy (vnh/vnl/akb staged 4x -- the slabs'
  job) is back and no slab variant fits the 3560B headroom.

## Ranking after round 15 (round-16 scope)

1. **dqkwg P11 order-preserving redundancy kill**: vnh/vnl/akb are
   staged per-(ks,vs) with values constant across ks -- the slab fix
   needs ~17.4KB the pool cannot give; candidates: a vs-major loop
   restructure (changes accumulation order -> new digits, full
   re-verify) or a [C,2BV] double-buffer pair (20480B -- needs the
   bhh/bhl window re-plumbed, 8.5KB over).  Prize: part of the ~0.1 ms
   restage cost.
2. **dhs pool diet (<= 50176)**: 65.5KB -> need -15.4KB; balanced
   kernel, expect 1.1-1.2x at best (the 2-CTA thesis pays when
   latency-bound).
3. **T8192 residuals**: fwdscan 1.498 (global arm) / wy 0.800 /
   wyb 1.131 / dkb 0.856 -- none touched this round; dhs 1.175 follows
   item 2.
4. **wyb 0.341 / dkb 0.247 polish** (unchanged from r14's list).

Repro: GDN_MAXNREG_ENTRY=dqkwg:128 bash evidence/G5/scratch/
r15w_build.sh <gen-root> [dqkwg-only] (the .maxnreg injection is
REQUIRED -- plain gdn_smem_fix yields the 168-reg cubin); gen
python tools/gdn_bwd_gen.py --shape ... --out DIR (knobs ... all prior
plus GDN_DQKWG_2CTA, default 1; GDN_DQKWG_THR default 256 under 2CTA);
verify tools/gdn_bwd_verify.py --gen-root DIR --shape ...; determinism
evidence/G5/scratch/r15w_det.sh DIR <shape> <tag> [N]; paired bench
evidence/G5/scratch/r15w_ab_bench.sh <base> <new> <shape> <tag>;
occupancy evidence/G5/scratch/r15w_ncu.sh DIR <shape> dqkwg <log>.
Fixed chains: r15w_2cta_T2048 / r15w_2cta_T8192 / r15w_final_T512
(baselines r15w_base_*; cubin-md5 assembly via r15w_assemble.sh).

# G5-U round 16 (R16-Y): dqkwg P11 vs-outer restructure + dhs BV=16 parallelism diet -- bwd 2.286 -> 2.228 ms @T2048, 8.150 -> 7.976 @T8192, 0.522 -> 0.499 @T512

Date: 2026-10-08 | clean idle window discipline (the r15w_ab_bench.sh
hard refusal guard carried into r16y_ab_bench.sh; coordination row
`gdn-mlir R16-Y`).  Baselines rebuilt fresh from git HEAD 289717f at
round start: ALL 18 cubin md5s (9 stages x T2048/T8192) reproduce the
r15 record exactly (dqkwg f618413ac53a / b42395cf722e, dhs
706a3f7a45a9 / 82fddbec2a4f, ...); HEAD 289717f touched only
gdn_full_gen.py/gdn_smem_fix.py (R15-X), gdn_bwd_gen.py byte-identical
to 680f5b6.  The T512 record chain is B1T512**H4** (the r13-r15
convention; my first T512 rebuild used H16 and failed the md5 gate --
caught by the rebuilt-vs-record check, the r14 clobber lesson paying
for itself again).  Artifacts: tools/gdn_bwd_gen.py (GDN_DQKWG_P11R
default 2, GDN_DHS_BV default 16, GDN_SCAN_BV knob default off),
chains evidence/G5/scratch/r16y_{base,final}_T{2048,8192,512} +
r16y_l{1,2,3}_* variants, logs r16_logs/, scripts r16y_{build,
assemble,ab_bench,det,ncu,verify,base,base512,fix512,fix512final,l1,
l2,l3,final,final_bench,md5s}.sh.

## Headline: bwd e2e 2.2860/2.2771 -> 2.2281/2.2202 med/min @B1T2048H16 paired (1.026x med AND min); T8192 8.1502 -> 7.9759 (1.022x, first dip under 8 ms); T512(H4) 0.5218 -> 0.4991 (1.045x)

ours/fla bwd = 1.03/2.2281 = **0.462x** (was 0.450x).  Final singles:
T2048 2.2323/2.2239 (dqkwg 0.5220, dhs 0.2805), T8192 7.9867/7.9689
(dqkwg 1.7552, dhs 1.0900), T512 0.4979/0.4938 (dhs 0.0408).  The two
levers are DIFFERENT STAGES (dqkwg + dhs) so the wins stack; every
other stage inside the 1-2% noise band with byte-identical cubins
(the assemble script md5-verifies each pair).

| lever | stage | T2048 | T8192 | T512(H4) |
|---|---|---|---|---|
| 1: P11 vs-outer (P11R=2) | dqkwg | 0.5445->0.5236 **1.040x** | 1.8233->1.7556 **1.039x** | v3 arm, cubin == r15 record |
| 2: dhs BV=16 | dhs | 0.3105->0.2820 **1.101x** | 1.1656->1.0863 **1.073x** | 0.0652->0.0406 **1.61x** |
| 3: fwdscan BV=16 | fwdscan | 0.3822->0.4955 **0.77x REGRESSION** | 1.4755->1.9211 **0.77x REGRESSION** | -- not shipped |

## Lever 1: the P11 restructure IS order-preserving (the r15 "changes accumulation order" worry was wrong at the element level)

The r15 ranking feared a vs-major restructure changes accumulation
order.  It does not, per OUTPUT element: the dw/dq1 chains keep ONE
accumulator set per ks threaded across the vs loop -- P11R=2 holds TWO
per-ks sets live across a vs-outer block, so every output element's
products still arrive in the same (vs, link) sequence into the same
mma accumulate chain; only the CODE order of the 16 (ks,vs) groups
changes.  r3's warp rowsums still fire ks-ascending, and accCK stays
ONE tile because at last_vs each ks's epi_dq1 fires with its OWN r3
rowsum BETWEEN them (ki's rowsum retires accCK before ki+1's epilogue
rewrites it; the chains'/rowsum's own trailing barriers serialize).
Measured proof: all 35 (T2048) / 36 (T8192) chain output arrays
BIT-IDENTICAL to the base chain (det-check single runs compared
element-wise), six verify margins digit-equal to the r15 tables.

The staging kill: akb/vnh/vnl are vs-only (values constant across ks)
but were restaged per (ks,vs) = 16 sweeps each; P11R=2 stages them
once per vs within each 2-ks block (8 sweeps; the per-(ks,vs) bhh/bhl
stages are irreducible -- every pair is distinct data).  P11R=4 (4
sets, full kill, 4 sweeps) was built and measured first: natural regs
222 (vs the 172 body) -> at the mandatory .maxnreg 128 cap: 232B
stack / 276B spill stores / 632B spill loads, and dqkwg only 1.026x
@T2048 -- the 4th accumulator set's spill traffic eats more staging
than it saves.  P11R=2 (natural 174 regs -> 96B stack / 96/148B
spills) wins: 1.040x.  The double-pass reg discipline (r16y_build.sh
records the smem_fix pass into ptxas_smemfix_v.txt -- the r15 find,
now structural in the build) surfaced both numbers pre-bench.

## Lever 2: dhs BV=16 -- the pool diet that actually buys parallelism

The r15 framing was "diet to <= 50176 for 2 CTA/SM", but dhs's grid
is BH*NV = 64 CTAs REGARDLESS of T (the serial NC-chunk walk lives in
the CTA) -- a pure diet would fit 2 CTAs/SM and change nothing at
T2048/T8192.  The diet that pays halves BV instead: X+Xa [K,BV] f32
32768 -> 16384, pool 65536 -> 49152 <= 50176, grid 64 -> 128 (2/SM),
regs 123 -> 109.  VALUE-EXACT: the V dim is pure column parallelism
(the k-window walk, the B sweep and every per-element FMA chain
unchanged) -- bit-identical outputs verified at every shape.  The
doubled A-window traffic (each A_all row now read by 8 v-siblings
instead of 4) stays in L2; dhs 1.10x @T2048, 1.07x @T8192, and 1.61x
@T512(H4) where the BV=32 grid was only 16 CTAs on 68 SMs (the
underfill was worst exactly where NC is small).  BV=8 overshoots
(grid 256 = ~2 waves at 2/SM + doubled-again A traffic): 0.80x
@T8192 -- honest negative; the sweet spot is BV=16.

## Lever 3 honest negatives: fwdscan BV=16 regresses 0.77x; wy/wyb/dkb have NO unported dispatch arms

fwdscan BV=16 (GDN_SCAN_BV, verified 6/6 + bit-identical BEFORE
benching): 1.4755 -> 1.9211 @T8192 (0.77x) and 0.3822 -> 0.4955
@T2048 (0.77x).  Diagnosis: dhs's per-chunk work is BV-scaled (the
A@X gemm output is [K,BV]; its K*K A operand streams via cp.async
windows that hide), but fwdscan's per-chunk critical path is dominated
by BV-INDEPENDENT work (the kg/wpp gemms and the CxC solve class) --
doubling the v-slice count doubles that work per device with no more
SMs.  The knob stays in the tree (default off) as the recorded probe.

wy/wyb/dkb: audited for shape-adaptive dispatch -- there is NONE in
the bwd gen beyond dqkwg's v3/v4 gate at BH*NC >= 164 (T8192 M=2048
already takes v4; T512/H4 M=32 keeps v3) and fwdscan's pool-driven
GSCG arm (built FOR T8192).  The four T8192 residuals share the T2048
code path; there is no T2048-class win sitting behind a gate to port.
Their polish is new-kernel work, listed for round 17.

## Numerics: 6/6 at ALL THREE shapes; every digit equals the r15 tables (value-exact round)

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| T2048 | 1.824e-4 | 1.284e-3 | 1.088e-5 | 8.527e-4 | 3.274e-4 | 2.169e-7 |
| T8192 | 2.078e-4 | 1.881e-3 | 1.050e-5 | 1.654e-3 | 1.145e-3 | 1.965e-7 |
| T512(H4) | 4.617e-4 | 5.131e-4 | 9.430e-6 | 2.178e-4 | 1.466e-4 | 1.394e-7 |

All six margins >= 4.3x under the bars at every shape (min = dk 4.3x
@T8192, unchanged from r15).  Both levers bit-identical: full-chain
output arrays compared element-wise vs the base chain -- 35/35 (l1,
l2, l3 @T2048), 36/36 (@T8192), 35/35 (final T512).  Knob-off bisect:
GDN_DQKWG_P11R=1 + GDN_DHS_BV=32 regenerates the HEAD dqkwg/dhs
payloads byte-identically at every shape (T512's dqkwg is the v3 arm
-- P11R does not touch it; the shipped T512 dqkwg cubin IS the r15
record 7de32d2a07c3, the .maxnreg injection is a 2CTA-arm need only).

## Determinism: fresh-process cycling, bit-identical

| chain | runs | result |
|---|---|---|
| l1 P11R=2 T2048 | x5 | ALL stage outputs bit-identical |
| l2 dhs BV=16 T2048 | x5 | ALL stage outputs bit-identical |
| final T8192 | x3 | ALL stage outputs bit-identical |
| final T512(H4) | x3 | ALL stage outputs bit-identical |

## Ranking after round 16 (round-17 scope)

1. **dhs 0.280 @T2048 / 1.081 @T8192**: the serial NC-walk is now a
   2-CTA/SM BV=16 kernel; the remaining critical path is the per-chunk
   A-window walk (cp.async, BV-independent) -- a BKW=16 window (pool
   49152 -> 40960, 3/SM class) or a two-chunk software pipeline (the
   XP token already rides iter_args) are the next cuts.
2. **fwdscan 0.371/1.470**: the structurally-closed 2-CTA verdict
   stands; the BV probe says the win must come from the BV-INDEPENDENT
   per-chunk work (the kg sweep / CxC solve), not the v-slicing.
3. **dqkwg 0.524/1.753**: P11's remaining restage redundancy is the
   bhh/bhl pair (irreducible without slabs); the 96B spill class at
   P11R=2 is the next reg budget to hunt.
4. **wyb 0.34/1.13 / dkb 0.24/0.86 / wy 0.22/0.79**: no dispatch gates
   exist; new-kernel work (wyb's dA chain is the biggest).

Repro: gen python tools/gdn_bwd_gen.py --shape ... --out DIR (new
knobs GDN_DQKWG_P11R default 2, GDN_DHS_BV default 16, GDN_SCAN_BV
default unset; all prior knobs unchanged); build
GDN_MAXNREG_ENTRY=dqkwg:128 bash evidence/G5/scratch/r16y_build.sh
<gen-root> [stages] (the smem_fix-pass regs land in
ptxas_smemfix_v.txt; T512-class v3 chains build with
GDN_MAXNREG_ENTRY=nomatch:0); verify r16y_verify.sh <base> <variant>
<T> [H=4 for T512] (6/6 + bit-diff vs base); paired
r16y_ab_bench.sh <base> <new> <shape> <tag>; determinism
r16y_det.sh <root> <shape> <tag> [N].  Fixed chains:
r16y_l1_p11r{2,4}_T{2048,8192}, r16y_l2_bv{8,16}_T{2048,8192},
r16y_l3_bv16_T{2048,8192}, r16y_final_T{2048,8192,512} (baselines
r16y_base_T{2048,8192,512}; new cubin md5s: dqkwg 00a68bd70fd7 /
1e4465d0fac2 / (T512: unchanged 7de32d2a07c3), dhs 95385a0a298f /
af489c0dd445 / 850e648a4429).

# G5-V round 17 (R17-bwd): bwd-side FPIPE -- dkb 1.06x / dhs 1.03-1.04x / dqkwg 1.01x, bwd 2.228 -> 2.202 ms @T2048, 7.976 -> 7.864 @T8192

Date: 2026-10-08 | clean idle window discipline (the r17b_ab_bench.sh
hard refusal guard; coordination row `gdn-mlir R17-bwd`, sibling
R17-fwd owns gdn_full_gen.py).  Baselines rebuilt fresh from git HEAD
26c2693 at round start: ALL 18 cubin md5s (9 stages x T2048/T8192)
reproduce the r16 record exactly (dqkwg 00a68bd70fd7 / 1e4465d0fac2,
dhs 95385a0a298f / af489c0dd445, dkb 70d7115ff11f / 46693d368d5f, wyb
40227e98d561 / 17fce9e4d3de, ...); T512(H4) reproduces the SHIPPED HEAD
chain (r16y_final records; dhs 850e648a4429 -- note r16y_base_T512's
dhs 709050b7 is the pre-BV16 payload, the final chain is the record).
Artifacts: tools/gdn_bwd_gen.py (new knob GDN_BFPIPE, default
"wy,dhs,dkb,dqkwg,wyb"; '' regenerates HEAD byte-identically), chains
evidence/G5/scratch/r17b_{base,final}_T{2048,8192,512}, logs r17_logs/,
scripts r17b_{build,verify,det,ab_bench,base,base512,assemble,md5s,
bitdiff,final,smoke,bisect}.sh.

## Headline: bwd e2e 2.2305 -> 2.2015 med @B1T2048H16 paired (1.013x med AND min); T8192 7.9776 -> 7.8644 (1.014x); T512(H4) 0.4982 -> 0.4946 (1.007x)

ours/fla bwd = 1.03/2.2015 = **0.468x** (was 0.462x).  Final stage
singles (T2048): dhs 0.2718, dkb 0.2295, dqkwg 0.5177; T8192: dhs
1.0468, dkb 0.8001, dqkwg 1.7385.  Untouched stages (fwdscan/caryA/
caryM/bpre) inside the noise band with byte-identical cubins.

| lever (all under GDN_BFPIPE) | stage | T2048 | T8192 | T512(H4) |
|---|---|---|---|---|
| dkb: dHv wait moved past the vngb emap | dkb | 0.2454->0.2321 **1.058x** | 0.8542->0.8001 **1.064x** | 0.0277->0.0275 1.01x |
| dhs: B-slice (Xa) DMA at chunk top | dhs | 0.2796->0.2719 **1.028x** | 1.0829->1.0468 **1.034x** | 0.0410->0.0378 **1.065x** |
| dqkwg: dPr vs=0 prologue groups | dqkwg | 0.5231->0.5184 **1.009x** | 1.7505->1.7372 **1.008x** | v3 arm, cubin == r15 record |
| wy: T_all->Tf DMA under the P phase | wy | 1.001x neutral | 1.006x ~ | 1.01x ~ |
| wyb: P1/P2 first-slice DMAs | wyb | 1.000x neutral | 0.999x ~ | 1.000x ~ |

## Lever 1: the FPIPE audit + what actually paid

Audited all 9 kernels for SYNCHRONOUS global->smem staging compute could
overlap (r4's astage_sync and r11+/r12's windows were already async;
fwdscan v3 carries r7's group-A/B ping-pong + the r14 GSCG arm -- left
alone).  The port adds ONE emission discipline, strictly inside the
project's hazard rules: raw cp.async lands bytes only in tiles whose
post-wait readers are SCALAR ld.shared passes (the r6 LDGSTS->LDSM
rule); every tile that feeds ldmatrix gets its values via an
element-wise st.shared pass after the wait -- the split stages re-read
the resident f32 temp (split_smem_ew: same truncf/extf/subf chain, same
order), and RAW-landed bf16 operand tiles get launder_pass (an identity
ld.shared/st.shared rewrite of exactly the staged cells -- the fwd
FPIPE in-place-scale pattern without the scale).  Two new helpers,
~50 lines total; no transform changes; every conversion is a byte-copy
plus identical arithmetic on identical values.

* **dkb (the biggest single win, 1.06x)**: per vt the dHv burst's
  wait_group sat BETWEEN the DMA and the vngb emap (the vnew global
  read + gc scale).  Moving the wait past the emap makes the vngb sweep
  the cover -- two equal-size global sweeps that were serial now
  overlap; disjoint tiles, generic-readers-only, zero pool change.
* **dhs**: the per-chunk B-slice (Xa, [K,BV] f32) was a sync
  stage_rows between the dh_all store sweep and the window walk.  It
  now issues as a cp.async group at the CHUNK TOP (before the store
  sweep), draining after the window-0 await -- the store sweep plus
  the cross-chunk window wait cover it.  Xa is dead at issue (the
  previous chunk's copy sweep read it, barrier-fenced) and feeds
  rgemm's generic ld.shared only.
* **dqkwg**: the dPr loop's vs=0 operands (akb <- do, vnew_all ->
  dprT) issue as two groups right after the gc cumsum; the beta stage
  + egc/rie/seg/bse computes + zero loops cover them; the waits land
  at the gc barrier, then launder(akb) + split(dprT -> vnh/vnl).
  dprT is a NEW aliased tile ([C+1,BV] f32, alloc 8320B uid) living
  at arena1 offset 0 -- accCC's own bytes, untouched through the whole
  dPr loop because the mma accumulator lives in REGISTERS (accCC's
  first smem write is the LAST link's epilogue, long after the split
  retired dprT).  The first cut parked dprT at +16384 (the dhi/dlo
  window) and its 8192B tail OVERWROTE akb@18944 (the dPr A operand,
  live!) -- dq/dk/dg NaN, verify-caught, bisected to dqkwg in one
  pass; the free-window audit is now recorded in the code comment.
  Incidental: the .maxnreg-128 spill class DROPPED 96B/96/148B (r16
  record) -> 88B stack/84/120B.
* **wy / wyb (neutral, honest)**: wy issues T_all -> Tf (16KB f32 raw)
  right after the T_all store barrier; the ENTIRE P phase covers it;
  the u-split (beta cvec) then reads Tf.  wyb issues P1's vs=0 dvp
  slice before the P0 vector phase, and P2's ks=0 dW slice right after
  the P1-vs0 split (covered by the P1 tail); both land in arena1's
  dead window (aTh+0, wyT [C+3,BK] f32 8576B uid, fp 8192, zero pool
  delta).  Both VALUE-EXACT but perf-neutral at these shapes: the
  sources are one-shot reads of buffers the same kernel (wy) or the
  immediately-preceding kernel (dqkwg's dvp_all/dW_all) just wrote --
  L2-resident, no latency to hide.  The fwd FPIPE's -28% came from
  COLD HBM k/v per chunk; the bwd's per-iteration re-stages (dkb/dhs,
  every vt/chunk) are where the cold traffic actually was, and that is
  exactly where the wins landed.  Kept (default on): they are free and
  the T8192 working set (dvp_all 64MB) is at the L2 edge -- the covers
  may matter at larger B.

## Lever 2 honest negative: 0-spill was reachable and is NOT worth it

GDN_DQKWG_THR=128 under 2CTA: 2x128 threads x 256 regs = the regfile,
so the 255-reg natural body ships UNPINNED with 24B stack/20/28B spill
(the only measured 0-spill-class configuration of this kernel).
Measured 0.5178 -> 0.5206 @T2048 (0.995x): the halved thread count
doubles the staging sweep issue counts and the W=4 mma strip map's
per-thread fragment load, and eats more than the 84/120B spill traffic
saves.  The shipped 256-thr/.maxnreg-128 arm keeps its (now smaller)
88/84/120B spill class -- spill elimination per se is not a lever here;
a real cut needs >= 46 natural regs out of the body (174 -> 128), which
no order-preserving restructure examined this round reaches (the P11
accumulator sets are only 2x8 regs each beyond the pre-P11 172-reg
baseline that ALREADY spilled).

## Lever 3 honest negative: dhs BKW=16 windows regress 0.91x

GDN_DHS_BKW=16 (pool 49152 -> 40960, NW 2 -> 4, verified 6/6 first):
dhs 0.272 -> 0.300 @T2048.  Doubling the window count doubles the
per-window wait+barrier+issue overhead while halving the rgemm cover
under each prefetch -- and the kernel is MEM-throughput-bound (r15 ncu:
48.9%), so the pool diet buys no residency (grid 128 CTAs already = 2
waves at 2/SM).  The Xa-at-chunk-top conversion (lever 1) is the
staging win the window walk had left.  The deeper 2-chunk-ahead XP
variant hits the same bandwidth ceiling.

## Lever 4 (audit finding, not shipped): dkb's 2CTA conversion needs accCK retirement

dkb runs 1 CTA/SM at ~90KB pool (dHv [128,32]f32 + accCK [64,128]f32 +
hh/hl [128,40]bf16 pairs -- the K=128-sized tiles).  Aliasing vh/vl
into accCK's bytes (lifetime-disjoint: accCK's only writer is the final
maccs store sweep) only reaches 79KB -- still 1/SM for zero benefit
(built, measured pool, reverted).  The real conversion is the r9 dqkwg
P-retirement playbook: park the maccs straight to dkB_sum global (the
r14 park_epi pattern), rowB's rowsum reading the L2-hot global via
gcol, THEN accCK's 32768B retire and BV=16 lands the pool at ~32.5KB
= 2 CTAs/SM.  Risks: NV=8 doubles the serial single-thread s5 folds
(a re-association-class change if restructured) and the Kred=16 links.
New-kernel class -> round-18 list.

## Numerics: 6/6 at ALL THREE shapes; every digit equals the r16 tables (value-exact round)

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| T2048 | 1.824e-4 | 1.284e-3 | 1.088e-5 | 8.527e-4 | 3.274e-4 | 2.169e-7 |
| T8192 | 2.078e-4 | 1.881e-3 | 1.050e-5 | 1.654e-3 | 1.145e-3 | 1.965e-7 |
| T512(H4) | 4.617e-4 | 5.131e-4 | 9.430e-6 | 2.178e-4 | 1.466e-4 | 1.394e-7 |

Full-chain bit-diff vs the rebuilt base: **35/35 arrays BIT-IDENTICAL**
@T2048 (final chain, incl. the launder/split passes).  Knob-off
bisect: GDN_BFPIPE= regenerates all 9 HEAD payloads byte-identically
at T2048 (re-checked after the final edit set).

## Determinism: fresh-process cycling, bit-identical

| chain | runs | result |
|---|---|---|
| final T2048 | x5 | ALL stage outputs bit-identical |
| final T8192 | x3 | ALL stage outputs bit-identical |
| final T512(H4) | x3 | ALL stage outputs bit-identical |

## Ranking after round 17 (round-18 scope)

1. **dkb 2CTA conversion** (lever 4's math): maccs->dkB_sum park +
   rowB gcol + BV=16, pool ~32.5KB, 2/SM; the dhs BV=16 precedent
   (1.10x@T2048, 1.61x@T512-underfill) is the model; watch the NV=8
   serial s5 folds.
2. **dqkwg mid-phase FPIPE depth**: the pt (16KB P_all^T) / tTh2+tTl2
   (16KB T_all^T) transposing splits and the per-ks dWhi/dWlo restage
   remain sync; their temp windows only open with a phase reorder
   (dvp/P11 boundaries) -- the r17 arena free-window audit is the map.
3. **fwdscan 0.373/1.475**: unchanged verdict (BV-independent per-chunk
   work; the 2-CTA pool bar is structural).
4. **wyb 0.340/1.128**: P3-P9 loop-internal prefetch is ping-pong class
   (needs pool); P7's kTt per-ks transposing restage is the biggest
   remaining sync staging.
5. **caryA w-window rotation**: 16KB f32 temp, ~21KB pool headroom
   exists (2CTA at 28.6KB) -- the one carry with cheap headroom.

Repro: gen `python tools/gdn_bwd_gen.py --shape ... --out DIR` (new
knob GDN_BFPIPE default "wy,dhs,dkb,dqkwg,wyb", '' = HEAD bytes); build
`GDN_MAXNREG_ENTRY=dqkwg:128 bash evidence/G5/scratch/r17b_build.sh
<gen-root> [stages]` (T512-class: GDN_MAXNREG_ENTRY=nomatch:0); verify
`r17b_verify.sh <base> <variant> <T>` (6/6 + bit-diff) or
`r17b_bitdiff.sh <base> <var> <T> [H]`; paired `r17b_ab_bench.sh <base>
<new> <shape> <tag>`; determinism `r17b_det.sh <root> <shape> <tag>
[N]`.  Fixed chains: r17b_base_T{2048,8192,512} + r17b_final_T{2048,
8192,512} + the r17b_l{1,2,3,4}_T2048 lever variants; md5s:
r17b_final_T2048 wy 5c4b834a6a56 / dhs 28be305aaf71 / dkb e3ba601bd1b3 /
dqkwg 837a36fcc908 / wyb f387689e228c (fwdscan/caryA/caryM/bpre ==
base); T8192 wy 412aea5a9875 / dhs fbf11c2bcd70 / dkb 00170d083153 /
dqkwg 1f53c8910006 / wyb d1da694efc93; T512 wy 7620830405e2 / dhs
30e2a5cac5ea / dkb 2ed728923ef1 / dqkwg 7de32d2a07c3 (== r15 record,
v3 arm untouched) / wyb a8517cd6f321.

# G5-W round 18 (R18-bwd): the dkb 2-CTA conversion — dkb 1.20x/1.29x, bwd 2.202 -> 2.16 @T2048 / 7.864 -> 7.68 @T8192, T512 grid-gated byte-identical

Date: 2026-10-08 | **CO-LOAD DISCIPLINE** (a co-job owned the GPU the
entire round -- never touched, never blocked on; every conclusion below
is a same-window PAIRED INTERLEAVED stage ratio x3; the e2e med columns
carry co-job noise -- byte-identical cubins swung 0.72x-1.07x there,
only stage-level deltas against the same-window base are decidable).
Baselines rebuilt fresh from git HEAD 30086e4 at round start: ALL 18
cubin md5s (9 stages x T2048/T8192) reproduce the r17 final records
exactly; the T512(H4) base built from the pristine HEAD gen reproduces
the r17 T512 chain.  Coordination row `gdn-mlir R18-bwd` in
~/codebase/GPU-COORDINATION.md.  Artifacts: tools/gdn_bwd_gen.py (new
knobs GDN_DKB_2CTA default 1, GDN_DKB_BV default grid-gated 16/32,
GDN_DQKWG_DWF default 0, GDN_CARYA_WF default 0; all-off regenerates
HEAD byte-identically at all three shapes), chains
evidence/G5/scratch/r18b_{base,final,l1,l2,l3,l4}_T*, logs r18_logs/,
scripts r18b_*.sh.

## Headline: dkb 0.2334 -> 0.1907 (1.22x) @T2048 / 0.8060 -> 0.6290 (1.28x) @T8192 paired — the r17-list #1 lands

Projected idle e2e (paired deltas on the r17 records): **2.2015 ->
~2.16 ms @B1T2048H16; 7.864 -> ~7.68 @T8192**; ours/fla bwd
0.468x -> ~0.477x.  T512(H4): the conversion is grid-gated OFF (M=32
CTAs underfills the device at any residency -- the same r9 dqkwg-v4
dispatch rule, BH*NC >= 164), so the T512 payload stays the r17 chain
BYTE-IDENTICALLY (dkb 2ed728923ef1); the ungated BV=16 arm measured
0.86x there (honest negative: the NV=8 serial s5 folds + per-vt
barriers cost ~4.5us with no occupancy to pay for them).

## Lever 1 (the designed route): maccs -> dkB_sum park + rowB gcol + BV=16 -> pool 32288B = 2 CTAs/SM

The r17 §lever-4 playbook, executed as bisected:

| arm | payload | pool | regs (smemfix) | dkb @T2048 | dkb @T8192 |
|---|---|---|---|---|---|
| base (HEAD) | accCK smem + store_rows + rowsum(accCK) | 89616B (1 CTA/SM) | 126 / 0 spill | 0.2334-0.2371 | 0.8060-0.8136 |
| l1: park only (BV=32) | maccs park straight to dkB_sum (r14 park_epi pattern: same x, same [m, row*K+col] cells); rowB rowsum reads the L2-hot global via gcol (r9 P pattern); accCK 32768B RETIRED; the store_rows sweep + its barrier retire | 56848B (1 CTA/SM) | 126 / 0 spill | 0.2376 (0.99x) | — |
| l2: + BV=16 (SHIP, grid-gated) | NV 4 -> 8; dHv/vngb halve; vh/vl/hh/hl [*,24] (48B stride: 12i mod 32 gives 8 distinct bank groups — the +16B-pad property survives) | **32288B <= 50176 -> 2 CTAs/SM** | **100** / 0 spill | **0.1907 (1.22x)** | **0.6290 (1.28x)** |

* **l1 is BIT-IDENTICAL across 35/35 arrays** (a byte move: the same
  register values stored to global instead of smem; rowB reads the same
  values back through L2) — the park mechanics were validated for free.
  Perf-neutral at 1 CTA/SM (0.99x): the smem hop it removed was cheap;
  the win is entirely the occupancy.
* **l2 is a value variant, 6/6 verified**: BV=16 re-associates the mma
  vt chain (the same 24 m16n8k16 sub-dots, interleaved hh/hl/lh per
  16-wide vt instead of 6-link per-32 groups) and the s5 per-thread
  partial assignment.  Every 6/6 table digit EQUALS the r17 record
  (the maxima are tf32-site-dominated; the ulp re-associations do not
  move them) — dk margin stays 6.2x @T2048 / 15.6x headroom-class @T8192.
  Bit-diff vs base: dkB_sum 67% of elems, dk 2%, rowB_c 83%, dg_state
  84%, s5_c 96% — all last-ulp-class, downstream of the two designed
  re-associations, exactly the predicted scope (nothing outside the
  dkb->dqkwg->wyb value cone differs).
* Regs 126 -> 100 (Kred=16 links halve the live fragment set) — the
  2x256x100 = 51200-reg co-residency fits the 65536 regfile with the
  margin the 126-reg BV=32 form lacked.
* The r17 risk callout ("NV=8 doubles the serial single-thread s5
  folds") measured benign at the winning shapes: the folds ride under
  the doubled per-vt tiling's mma work; no re-association-class s5
  restructure was needed (none was attempted — the mission bar was
  value-exact only for fwdscan).

## Lever 2 honest negative: dqkwg mid-phase FPIPE — the dW-restage arm 0.98x, default OFF

The r17 leftover list (pt / tTh2+tTl2 / per-ks dW restage) audited
against the arena free-window map:
* **pt and tTh2/tTl2 stay SYNC, window-closed**: pt's alias home is
  accCC@0 (LIVE through the whole dPm loop) and ptlo@arena2 collides
  with krt/qrt; tT's 16KB raw temp would exactly-fill arena1's
  [18432, 34816) dead window and evict the dW double-buffer.  Both
  need the phase reorder the r17 audit flagged -- not attempted (the
  one-shot 16KB stagings are ~1-2us exposures on a 520us kernel).
* **The per-ks dW restage WAS converted** (GDN_DQKWG_DWF): raw
  [C,BK] f32 slices double-buffered as cp.async into arena1's
  [18432, 34816) dead window (26624+8192 = 34816 = arena1 EXACTLY,
  zero pool delta; dWt0/dWt1 padded to unique identification sizes),
  ks=0's DMA covered by the one-time tT split, the transposing split
  re-read from smem (_split_bf16_T -- same truncf/extf/subf chain,
  r6-clean: the raw temps feed generic ld.shared only).  BIT-IDENTICAL
  outputs (35/35).  **Measured 0.5158 -> 0.5266 @T2048 paired
  (0.98x REGR)**: the natural body grows 174 -> 220 regs, so under the
  shipped .maxnreg-128 the spill class grows 88/84/120B -> 136/136/208B,
  and the split becomes TWO smem sweeps where stage_split_T was one
  fused global-reading sweep.  The pipelining hides only ~2us of
  L2-hot latency per iteration (dW_all is same-kernel-written).
  Knob kept (=1) for the record; default OFF; the shipped dqkwg cubin
  stays the r17 record (837a36fcc908 / 1f53c8910006 / 7de32d2a07c3).

## Lever 4 honest negatives: caryA w-window 0.96x; wyb P7 kTt blocked by the reg cap

* **caryA w-window rotation** (GDN_CARYA_WF): i0=BK2's 16KB w_all
  slice cp.async'd across i0=0's whole j0 loop, split from smem by a
  new fused transposing split helper (_split_T_smem: stage_split_T's
  exact truncf/extf/subf + seg fold, read side ld.shared).  Pool
  28592 -> 44976 (still 2 CTAs), BIT-IDENTICAL outputs.  **Measured
  caryA 0.0541 -> 0.0564 @T2048 paired (0.96x REGR)**: the one
  i0-boundary stall it hides was already amortized in the 54us kernel;
  the 16KB window's smem traffic + the await/barrier boundary cost
  more.  Default OFF, knob kept.
* **wyb P7 kTt restage**: not attempted -- wyb ships at the .maxnreg
  cap (128 regs, 48/44/72B spill), and the DWF arm just demonstrated
  that adding staging machinery under a reg cap grows the spill class
  faster than L2-latency hiding pays.  Needs pool headroom for a
  double-buffer (5120B short) or a reg diet first -> round-19 list.

## fwdscan: untouched (the r16/r17 verdict stands)

r16's negative showed the per-chunk path dominated by BV-INDEPENDENT
kg/wpp/CxC work; the r15 K-split-across-CTAs idea for the scan itself
remains the deferred new-numerics-class.  No clean value-exact variant
was found this round (the kg split-bf16 idea is a numerics tighten,
not a staging win; the CxC elementwise is already vector-wide).

## Numerics: 6/6 at ALL THREE shapes; every digit equals the r17 tables

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| T2048 | 1.824e-4 | 1.284e-3 | 1.088e-5 | 8.527e-4 | 3.274e-4 | 2.169e-7 |
| T8192 | 2.078e-4 | 1.881e-3 | 1.050e-5 | 1.654e-3 | 1.145e-3 | 1.965e-7 |
| T512(H4) | 4.617e-4 | 5.131e-4 | 9.430e-6 | 2.178e-4 | 1.466e-4 | 1.394e-7 |

(T512 payload is byte-identical to r17 by the grid gate.)  The T2048
bit-diff scope: 7 arrays differ, all last-ulp-class inside the
dkb->dqkwg->wyb value cone (dkB_sum/dk/dk_local/rowB_c/s5_c/dg/
dg_state), 28/35 arrays bit-identical.  Knob-off bisect:
GDN_DKB_2CTA=0 GDN_DKB_BV=32 GDN_DQKWG_DWF=0 GDN_CARYA_WF=0
regenerates the HEAD base payloads byte-identically (9/9 stages).

## Determinism: fresh-process cycling, bit-identical

| chain | runs | result |
|---|---|---|
| final T2048 | x5 | ALL stage outputs bit-identical |
| final T8192 | x3 | ALL stage outputs bit-identical |
| final T512(H4) | x3 | ALL stage outputs bit-identical (== base payload) |

## Ranking after round 18 (round-19 scope)

1. **dqkwg 0.518/1.747** (now 34% of T2048): the mid-phase stagings
   need the PHASE REORDER (pt past the dPm loop; tT split moved into
   P11's dead window) or the P11 slab revival under a smaller pool --
   the DWF lesson prices every added staging at its spill cost first.
2. **fwdscan 0.373/1.475**: the structurally-closed verdict stands;
   the K-split-across-CTAs scan remains the only big lever (new
   numerics class).
3. **dkb residuals**: 0.197/0.630 -- the NV=8 s5 folds (8x256
   single-thread adds) are now the visible serial tail; a value-class
   fold restructure has margin (dh0 bar 8e-3 vs 2.2e-7) if round-19
   wants it.
4. **wyb 0.34/1.13**: P7 kTt double-buffer needs ~5KB pool or a reg
   diet below the 128 cap (the spill-class gate).
5. **caryA/caryM/bpre/wy/dhs**: closed at these shapes (dhs A-window
   XP-2 remains the only listed residual, r16 #1 note).

Repro: gen `python tools/gdn_bwd_gen.py --shape ... --out DIR` (new
knobs GDN_DKB_2CTA default 1 grid-gated, GDN_DKB_BV default 16-if-big,
GDN_DQKWG_DWF default 0, GDN_CARYA_WF default 0; all-off == HEAD
bytes); build `GDN_MAXNREG_ENTRY=dqkwg:128 bash evidence/G5/scratch/
r18b_build.sh <gen-root> [stages]` (T512-class: GDN_MAXNREG_ENTRY=
nomatch:0); verify `r18b_verify.sh <base> <variant> <T> [H]` (6/6 +
bit-diff); paired `r18b_ab_bench.sh <base> <new> <shape> <tag>`
(CO-LOAD tolerant: prints the co-job line, paired-ratio mode);
determinism `r18b_det.sh <root> <shape> <tag> [N]`.  Fixed chains:
r18b_base_T{2048,8192,512} (== r17 records, all 27 md5s) +
r18b_final_T{2048,8192,512} + the r18b_l{1,2,3,4}_T2048 lever arms;
md5s: final dkb 506856583563 @T2048 / a0ba6cf5e927 @T8192 /
2ed728923ef1 @T512 (== base; every other stage == base).

# G5-X round 19 (R19-bwd): dqkwg phase-reorder CLOSED (TTF 0.99x spill-priced, pt arena-blocked) + fwdscan K-split CPU GO + dkb s5 tree SHIP (1.05-1.07x/1.04x)

Date: 2026-10-08 | **CO-LOAD DISCIPLINE** (a co-job owned the GPU the
entire round -- pids 3888342/3898781, ~8.9GB resident; never touched,
never blocked on; every conclusion is a same-window PAIRED INTERLEAVED
stage ratio x3; the e2e med columns carry co-job noise, only
stage-level deltas on CHANGED cubins are decidable).  Baselines
rebuilt fresh from git HEAD 0aa4b37 at round start: ALL 18 cubin md5s
(9 stages x T2048/T8192) reproduce the r18 final records exactly.
Coordination row `gdn-mlir R19-bwd` in ~/codebase/GPU-COORDINATION.md.
Artifacts: tools/gdn_bwd_gen.py (new knobs GDN_DKB_S5T default 1
grid-gated, GDN_DQKWG_TTF default 0; all-off regenerates HEAD
byte-identically, 9/9 stages), tools/gdn_fwdscan_ksplit.py (the
K-split CPU twin, NEW), chains evidence/G5/scratch/r19b_{base,final,
knoboff,l1,l2}_T*, logs r19_logs/, scripts r19b_*.sh.

## Headline: dkb 0.1930 -> 0.1827-0.1832 (1.05-1.07x) @T2048 / 0.6313 -> 0.6071-0.6080 (1.04x) @T8192 paired

Projected idle e2e (paired deltas on the r18 records): **~2.16 ->
~2.15 ms @B1T2048H16; ~7.68 -> ~7.656 @T8192**; ours/fla bwd
~0.477x -> ~0.479x (the r18 1.03ms basis).  T512(H4): grid-gated OFF
(the DKB2 dispatch rule) -- the T512 payload stays the r18 chain
BYTE-IDENTICALLY (dkb 2ed728923ef1, verify + 35/35 bit-diff vs the
r18 T512 base).

## Lever 1 (the r18-list #1, the designed phase reorder): honest negative, family CLOSED

The r18 lesson priced every staging at its spill cost FIRST; the r19
arena audit did exactly that, phase by phase (shipped arm:
TWO_CTA+SB+FOLDS, P11R=2, DWF off):

* **pt is GEOMETRICALLY BLOCKED.**  During the dPm tails (the dq/dk
  park loops) arena1 is byte-exact FULL: accCC [0,16384) is the split
  source through BOTH re-splits, dhi/dlo occupy [16384,34816), and
  arena2 holds krt/qrt -- there is NO window for a raw P half to be
  in flight under the dPm cover.  The kernel-top alternative (issue
  with the _fpdq prologue groups, covered by gc+dPr+r4/c4r) collides
  everywhere: [0,16384) is dprT then accCC's last-vs epilogue;
  [16384,33280) is pst (the P restage) through r4/c4r; the only 8KB
  free-until-P11 window IS pst's own bytes (retiring pst = paying
  back the r13 LDS win).  Documented, not attempted.
* **tT (GDN_DQKWG_TTF) was the exact-fit conversion -- and it works
  value-exactly but loses on registers.**  The r17 leftover, unblocked
  by DWF's OFF default: relocate bhh/bhl/akb/accCK to [16384,34816)
  (offsets 16384/18944/21504/26624; bhh boundary-exact against the
  new tTr@0), land the raw T_all into tTr@arena1@0 ([0,16384) =
  pthi/bdt's bytes, TIME-DISJOINT: pt's last reader is the dvp loop's
  final ops_pd link, fenced by its trailing barrier before the
  P11-top issue), await+barrier after P11, split from smem into
  tTh2/tTl2@16384/25600 (disjoint from the tTr source; zero pool
  delta, all aliased, arena1 stays 34816).  One-shot prologue class
  (the _fpdq pattern), NOT loop-carried.  **BIT-IDENTICAL 35/35, 6/6
  digits == the r18 tables.**  **Measured 0.5184 -> 0.5241 @T2048
  paired (0.99x)**: the natural body grows 220 -> 232 pass1 regs and
  the .maxnreg-128 spill class 88/84/120B -> 128/132/232B -- the
  async GROUP TOKEN stays live ACROSS the whole P11 loop (the
  register-hottest phase), and that pinned liveness costs more than
  the ~1-2us of hidden one-shot T_all latency pays.  Knob kept
  (=1), default 0.
* **The dqkwg mid-phase FPIPE family is now CLOSED on three data
  points** (DWF 0.98x loop-carried, TTF 0.99x one-shot, pt blocked):
  at the pinned .maxnreg 128, ANY added staging machinery grows the
  spill class faster than the r17-audited ~1-2us one-shot exposures
  pay.  The only door back in is a natural-reg diet (174 -> ~128) --
  no order-preserving restructure examined reaches it (the r17
  verdict stands).

## Lever 2 (the research deliverable): fwdscan K-split-across-CTAs -- CPU verdict **GO**

fwdscan stays 0.372-0.385/1.538-1.576 ms (2 CTAs/SM structurally
barred, the r16 verdict).  The K-split parallelizes the SCAN STATE
itself: S [K,BV] f32 split by K-rows across two co-resident CTAs per
(bh,vt), one [C,BV] f32 partial exchange per chunk.  **The
decomposition** (tools/gdn_fwdscan_ksplit.py docstring carries the
full derivation): the kernel's three-factor walk S' = 2^gcl*S +
kg^T(u - w@S) splits exactly over K-row halves -- the decay is
element-wise (row-split), the kg^T vn output rows need only their own
kg columns (row-split, vn shared), and the ONE cross-K coupling
(w@S = sum_h w[:,R_h] @ S[R_h]) resolves by exchanging the [C,BV]
partials and adding in fixed order.  In f32 the ONLY change is the
acc association (fused folds slice1 INTO slice0's accumulator;
split folds each half from 0 and adds once) -- 2^-24-class.

**The twin** (strict left-fold f32 simulation of the kernel's exact
k-order via np.add.accumulate; fp64 walks for the identity):

| check | T2048 (NC=32) | T8192 (NC=128) | seeds |
|---|---|---|---|
| fp64 split-2/4 vs fp64 seq (identity) | h_all **0.0**, vnew 5.7e-17 | h_all **0.0**, vnew 5.4e-17 | 3/3 exact |
| f32 split-2 vs f32 seq (PURE re-assoc) | vnew 3.08e-8 rel, h_all **0.0** | vnew 2.89e-8 rel, h_all **0.0** | 3/3 |
| f32 split vs fp64 seq (error class) | **1.00x** the sequential's | **1.00x** | 3/3 |

The identity is machine-precision exact (the h_all walks are
BIT-EQUAL at fp64); the re-association costs <= 1 ulp of vnew and
MOVES h_all NOT AT ALL in f32 (the vn ulp-delta feeds S through
kg^T vn, lands far below S's own rounding, and rounds away); error
amplification over the scan is 1.00x at NC=32 AND NC=128.  **GO.**

**GPU port (round-20 item, not attempted this round):** grid
BH*NV*2; each CTA owns S rows R_h + w/kg [C,K/2] column slices +
half the h_all store rows; per chunk both CTAs fold their acc half
(BK=64 = K/2: each CTA folds exactly ONE BK slice), exchange partials
through a global scratch with a flag/spin protocol (decoupled-
lookback class), add h-ascending (fixed order = the twin's tree),
compute vn identically, update their S rows.  Deferred for two
honest reasons: the exchange sits ON the scan critical path (NC
serial global round-trips must amortize against a latency-bound
kernel), and the 2-CTA co-residency + flag-protocol correctness
class is exactly the r18-fwd F1KS corruption saga (masked-at-1-CTA
bugs + races that only fire at co-residency) -- it needs its own
hardened round.  The math is now proven; the port is engineering.

## Lever 3 (the r18 residual list): GDN_DKB_S5T SHIPS -- the s5 serial fold becomes a parallel tree

The r18 BV=16 ship doubled the per-vt SINGLE-THREAD serial folds
(thread 0 walks all 128 s5p partials per vt: NV=8 x 128 = 1024
dependent LDS+FADD ops per CTA -- the visible serial tail).  S5T
re-lays the partials [threads, NV] (s5p2, one column per vt, written
by the SAME per-thread partial fold -- identical distribution and
values) and folds the THREAD axis as a 7-stage strided smem tree
(t < s: s5p2[t][*] += s5p2[t+s][*], one barrier per stage), then
thread 0 folds the NV column sums and scales once by 2^gcl (the
per-vt exp2 distributes over the sum).  s5v retires; pool 32288 ->
36352B (still 2 CTAs/SM); regs 100 -> 112, 0 spill.

| session (paired x3) | dkb @T2048 | dkb @T8192 |
|---|---|---|
| r18 record (base rebuilt) | 0.1930-0.1954 | 0.6313-0.6350 |
| S5T lever arm | **0.1832 (1.05x)** | **0.6071 (1.04x)** |
| S5T final-chain confirm | **0.1827 (1.07x)** | **0.6080 (1.04x)** |

Value-class change with the predicted scope: bit-diff vs base =
**s5_c (472/512 @T2048, 1885/2048 @T8192) + dg_state (1978/32768,
8006/131072) ONLY** -- 33/35 arrays bit-identical, every 6/6 table
digit EQUALS the r18 records (s5_c's only consumer is dqkwg's dg5a
term -> dg, bar 2e-2 vs the 3.274e-4 max, ~60x margin).

## Numerics: 6/6 at ALL THREE shapes; every digit equals the r18 tables

| shape | dq | dk | dv | dbeta | dg | dh0 |
|-------|----|----|----|----|----|-----|
| T2048 | 1.824e-4 | 1.284e-3 | 1.088e-5 | 8.527e-4 | 3.274e-4 | 2.169e-7 |
| T8192 | 2.078e-4 | 1.881e-3 | 1.050e-5 | 1.654e-3 | 1.145e-3 | 1.965e-7 |
| T512(H4) | 4.617e-4 | 5.131e-4 | 9.430e-6 | 2.178e-4 | 1.466e-4 | 1.394e-7 |

(T512 payload byte-identical to r18 by the grid gate -- verified
35/35 vs the r18 T512 base.)  Knob-off bisect: GDN_DKB_S5T=0
(with GDN_DQKWG_TTF off by default) regenerates the HEAD base
payloads byte-identically, 9/9 stages @T2048.

## Determinism: fresh-process cycling, bit-identical

| chain | runs | result |
|---|---|---|
| final T2048 | x5 | ALL stage outputs bit-identical |
| final T8192 | x3 | ALL stage outputs bit-identical |
| final T512(H4) | x3 | ALL stage outputs bit-identical (== r18 payload) |

## Ranking after round 19 (round-20 scope)

1. **fwdscan K-split GPU port** (CPU GO this round; the exchange
   design is in the twin's docstring; budget the F1KS-class hardening
   -- flag protocol, co-residency gates, det N=5 at 2 CTAs).
2. **dkb post-tree residuals**: 0.183/0.607 -- the per-vt barrier
   count (8 sequential, one per dHv restage) is now the visible
   serial structure; a 2-buffer dHv ping-pong would need ~2.5KB pool
   (36352 -> ~38.9KB, still 2 CTAs) -- reg-class check first.
3. **wyb 0.34/1.13**: P7 kTt double-buffer needs ~5KB pool or a reg
   diet below the 128 cap (unchanged r18 item; the spill-class gate).
4. **dqkwg 0.516-0.526**: mid-phase FPIPE CLOSED (three data points);
   the only door is a >=46-natural-reg diet (174 -> 128) -- no
   order-preserving candidate identified; or accept and move on.
5. **caryA/caryM/bpre/wy/dhs**: closed at these shapes (unchanged).

Repro: gen `python tools/gdn_bwd_gen.py --shape ... --out DIR` (new
knobs GDN_DKB_S5T default 1 grid-gated, GDN_DQKWG_TTF default 0;
all-off == HEAD bytes); CPU twin `python tools/gdn_fwdscan_ksplit.py
[--T 2048|8192]`; build `GDN_MAXNREG_ENTRY=dqkwg:128 bash
evidence/G5/scratch/r19b_build.sh <gen-root> [stages]` (T512-class:
GDN_MAXNREG_ENTRY=nomatch:0); verify `r19b_verify.sh <base> <variant>
<T> [H]` (6/6 + bit-diff); paired `r19b_ab_bench.sh <base> <new>
<shape> <tag>` (CO-LOAD tolerant); determinism `r19b_det.sh <root>
<shape> <tag> [N]`.  Fixed chains: r19b_base_T{2048,8192} (== r18
records, all 18 md5s) + r19b_final_T{2048,8192,512} + r19b_l1_T2048
(TTF arm) + r19b_l2_T{2048,8192} (S5T lever arms) +
r19b_knoboff_T2048; md5s: final dkb 1538de9d5dee @T2048 /
fbb0682b6776 @T8192 / 2ed728923ef1 @T512 (== base; every other
stage == base).

# G5-X round 20 (R20-bwd): fwdscan K-split GPU port -- MATH PROVEN ON-GPU, PERF HONEST NEGATIVE (warp-group 0.98-0.99x) + dkb dHv ping-pong honest negative (0.97x/0.99x)

Date: 2026-10-08 | CO-LOAD DISCIPLINE (a co-job owned the GPU all
round -- pids 3912354/3916562/3922799 across windows, never touched,
never blocked on; every conclusion is a same-window PAIRED INTERLEAVED
stage ratio x3; only stage-level deltas on CHANGED cubins are
decidable).  Baselines rebuilt fresh from git HEAD 3accc73 at round
start: 16/18 cubin md5s == the r19 base records and dkb ==
1538de9d5dee/fbb0682b6776 (the r19 FINAL S5T md5s -- HEAD includes
R19-b), so the working baseline IS the r19-final chain; fwdscan
e680e1d94e47/304b4259a79d.  Coordination row `gdn-mlir R20-bwd` in
~/codebase/GPU-COORDINATION.md.  Artifacts: tools/gdn_bwd_gen.py (new
knobs GDN_FKS_WG default 0, GDN_DKB_DHP default 0; both grid/shape
gated; all-off regenerates HEAD byte-identically -- payload byte gate
27/27 across 9 stages x T2048/T8192/T512), chains evidence/G5/scratch/
r20b_{base,l1,l2,l3,koff,koff3,defoff,final}_T*, logs r20_logs/,
scripts r20b_*.sh.

## The port decision: the two-CTA flag form REJECTED at design time; the WARP-GROUP form built

The r19 twin design (two co-resident CTAs per (bh,vt), [C,BV] f32
partials through a global scratch with a flag/spin protocol) carries
exactly the F1KS hazard family, now with three concrete failure modes
priced BEFORE writing it: (1) co-residency-dependent FORWARD PROGRESS
-- a pair straddling a wave boundary spins forever while its peer
never starts, and the CO-LOAD can force exactly that split by taking
SM slots (82 SMs, grid 128 at 2 CTAs/SM needs every slot); (2)
cross-launch flag staleness (stream-ordered kernel completion makes
kernel-side resets safe, but only with another protocol arm); (3) the
PTX memory model fencing of plain data loads after an acquire flag
(memref.atomic_rmw DOES lower to acq_rel llvm.atomicrmw here -- the
primitive exists -- but the harness contract would also need scratch
globals, breaking the every-input-staged rule in gdn_bwd_verify/
gdn_det_check).  ALL of it evaporates by keeping ONE 256-thread CTA
and splitting the EXISTING 8 warps into two groups (warps 0-3 = K-rows
R_0, warps 4-7 = R_1): the exchange rides SMEM (P_0/P_1 in sibling
acc tiles, the vn sweep adds once in fixed order), gpu.barrier IS the
r10 fence, and the values are concurrency-independent by construction
(the mma k-chains are warp-map-independent).  Same resource envelope
as the 2-CTA form (256 thr, 49152 rf regs, one chain per SM), zero
new sync surface, no harness changes, unconditional determinism.
Grid unchanged (BH*NV), pool 98560B == v3 (vn retires in place over
utl, acc0+acc1 replace acc), T8192 keeps the GSCG global route
verbatim; 210/208 pass1 -> 114/106 smemfix-pass regs, 0 spill.

## The math gate: GO on-chain (the r19 deliverable, now GPU-proven)

GDN_FKS_WG=1 (v4) vs the HEAD r15-hybrid, full chains (wy feeds
bit-identical inputs; fwdscan outputs isolate the re-association):

* **h_all BIT-IDENTICAL, 0/8388608 elements** (and absent from the
  T8192 diff list) -- the twin's "h_all does not move" prediction
  holds exactly on chain data; h_all stays f32-exact.
* **vnew_all maxabs 2.384e-7 (2 ulps), rel 5.95e-8** vs the twin's
  3.08e-8 on its own distribution -- the same last-ulp re-association
  class, distribution-scaled.
* Downstream scope EXACTLY the predicted cone: vnew_all + dkb
  (dkB_sum/dk/dk_local/rowB_c/dg_state) + dqkwg (dg/dq) + caryA
  (c4r_c/r4_c); 25/35 arrays bit-identical @T2048.
* **6/6 PASS both shapes; every digit == the r19 record tables**
  (T2048 1.824e-4/1.284e-3/1.088e-5/8.527e-4/3.274e-4/2.169e-7; T8192
  2.078e-4/1.881e-3/1.050e-5/1.654e-3/1.145e-3/1.965e-7); margins
  6-61x.  Knob-off byte-identical BOTH shapes (e680e1d94e47 /
  304b4259a79d).  det N=5 fresh-process: ALL stage outputs
  bit-identical.

## The perf verdict: honest NEGATIVE, root-caused

Two structural arms, paired x3 each (co-load ratio discipline):

| arm | fwdscan @T2048 | fwdscan @T8192 |
|---|---|---|
| HEAD (base rebuilt) | 0.3729-0.3765 | 1.5015-1.5080 |
| v4a (one fused per-group staging group) | 0.3787 (0.99x) | 1.5321 (0.98x) |
| v4b (v3's A/B cover cadence restored; values bit-identical to v4a) | 0.3831 (0.98x) | 1.5205 (0.99x) |

Root cause: the warp-group re-partition leaves PER-WARP WORK
UNCHANGED -- the m16n8 tile granularity forces a 4-warp tiling of
each [C,BV] half-gemm, so every warp still walks 4 atom-tile chains
per chunk (v3: 2 chains x 2 slices; v4: 4 chains x 1 half), and the 8
resident warps ALREADY interleave v3's formerly-dependent slice
chains (the ILP the design wanted was present at warp-scheduler
level).  The occupancy door the r19 design actually needed -- 2
CTAs/SM = 16 warps -- stays STRUCTURALLY BARRED: pool 98560B vs the
50688B 2-CTA bar (the [C,K] w/kg tiles and the replicated vn/acc do
not halve with the K-split; even BV=16 lands at 78080B).  Verdict:
GDN_FKS_WG kept, default 0; the K-split math port is PROVEN and
parked for any future door into fwdscan residency.

## Lever 2: dkb dHv ping-pong (GDN_DKB_DHP) -- honest negative

The r19-list #2.  Reg check FIRST per the r19 law: the arm lands at
127 pass1 / 110 smemfix regs (bar: 128 for 2x256thr) -- the 2-CTA
residency SURVIVES; pool 36352 -> 44544B (+8192 dHvB; bar 50688).
Implementation: static-parity dHvA/dHvB (the vt loop is a Python
unroll), the NEXT vt's bursts issue right after this vt's emap and
fly under the splits/mma-chain/s5 fold (token never crossing the mma
interior -- the r19 TTF lesson), the loop-top wait drains the
previous group with numGroups=1 keeping the newest in flight
(cp.async.wait_group semantics; the LAST vt uses a full drain -- a
numGroups=1 there would skip its own wait), r10 fence satisfied by
distance-2 buffer reuse across each vt's trailing barrier.
**BIT-IDENTICAL outputs vs HEAD** (pure cover change).  **Measured
dkb 0.1811 -> 0.1863 @T2048 (0.97x REGR) / 0.6044 -> 0.6087 @T8192
(0.99x)**: the r17 FPIPE emap was ALREADY covering the dHv DMA, so
the loop-top await drained a nearly-paid group while the extra
issue/wait machinery + 8KB of pool cost more.  The REAL serial
structure is the cross-vt mma ACCUMULATOR CHAIN (3 dependent links
per vt, value-bearing fold order) + the per-vt s5p2 column barriers
-- a value-class re-association (the vt fold is a pure sum), round-21
class.  Default 0, knob kept.

## Not attempted (the r19 ranking's own gates)

wyb P7 kTt (needs ~5KB pool or a sub-128 reg diet; the spill-class law
now has THREE closed data points) and the dqkwg >=46-natural-reg diet
(no order-preserving candidate identified).  e2e unchanged: co-loaded
min ~2.15ms @B1T2048 all arms; the r19 basis (~2.15 idle-projected,
ours/fla ~0.479x) stands -- the shipped chain stays the r19-final
bytes (27/27 payload byte gate, det N=5 bit-identical, 6/6 spot PASS).

## Ranking after round 20 (round-21 scope)

1. **dkb vt-fold re-association** (0.181/0.604): the mma accumulator
   chain across vt is a pure sum -- two accumulator sets (vt halves)
   + one fixed-order add is the SAME re-assoc class this round proved
   safe twice (K-split h_all-exact; S5T margins ~60x); halves the
   serial mma chain per CTA.
2. **fwdscan residency door** (0.373/1.475): the K-split math is
   parked and proven; any future lever must buy SMEM (a [C,K]-tile
   halving or the 2-CTA form with cluster-launch co-scheduling --
   cuLaunchKernelEx + CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION would
   make the pair co-residency a DRIVER GUARANTEE, voiding the F1KS
   forward-progress hazard; the flag protocol primitive is confirmed
   available: memref.atomic_rmw -> acq_rel atomicrmw).
3. **dqkwg 0.518/1.75**: unchanged (the reg-diet door).
4. **wyb 0.34/1.13 / caryA / caryM / bpre / dhs / wy**: closed at these
   shapes (unchanged).

Repro: gen `python tools/gdn_bwd_gen.py --shape ... --out DIR` (new
knobs GDN_FKS_WG default 0, GDN_DKB_DHP default 0; all-off == HEAD
bytes 27/27 x 3 shapes); build `GDN_MAXNREG_ENTRY=dqkwg:128 bash
evidence/G5/scratch/r20b_build.sh <gen-root> [stages]`; verify
`r20b_verify.sh <base> <variant> <T> [H]`; paired `r20b_ab_bench.sh
<base> <new> <shape> <tag>`; determinism `r20b_det.sh <root> <shape>
<tag> [N]`.  Fixed chains: r20b_base_T{2048,8192} (== the r19-final
records; 16/18 == r19 base + dkb == r19 final) + r20b_l1_T{2048,8192}
(v4a) + r20b_l2_T{2048,8192} (v4b) + r20b_l3_T{2048,8192} (DHP arm) +
r20b_koff{,3}_T* + r20b_defoff_T* + r20b_final_T* (payloads == base);
md5s: every stage == the r19-final records (dkb 1538de9d5dee @T2048 /
fbb0682b6776 @T8192; fwdscan e680e1d94e47 / 304b4259a79d).
## Round 21 (R21-bwd): all three r20-list levers resolved -- honest negatives across the board; the sm_120 cluster door OPENED (G0 claim disproven, harness contract landed); the fwdscan K-split family CLOSED with three GPU data points

Base rebuilt at HEAD 2bea03f: 18/18 stage cubin md5s == the r19-final
records (r21b_base_T{2048,8192}).  Nothing ships this round; the chain
stays the r19-final bytes; e2e unchanged: co-loaded min ~2.15 ms
@B1T2048 all arms (ours/fla ~0.479x, the r19 basis).  The round's
deliverables are the two lever verdicts below plus a NEW capability
finding that outlives them (real thread-block clusters on sm_120 +
the cuLaunchKernelEx harness contract), and two more numerics-class
proofs at record digits.

## Lever 1: dkb vt-fold re-association (GDN_DKB_VTF) -- honest negative

The r20-list #1.  The cross-vt mma accumulator chain (NV x 3 = 24
dependent mma.sync links per CTA) re-associated as TWO accumulator
sets -- EVEN vts build maccs0, ODD vts maccs1, one fixed-order
fragment-level add (even-fold + odd-fold) at the park sweep.  EVEN/
ODD, not the ranking's vt-halves: warp issue is IN ORDER, so two
half-chains still ISSUE sequentially (all of set A's links precede
set B's in program order); even/odd interleaves the chains
link-by-link, halving the ISSUE-level critical path at the SAME
+fragment-reg cost either way.  Reg check FIRST (the r19 law): 126
pass1 / 126 smemfix regs, 0 spill -- the 2-CTA residency SURVIVES
(v3-era 112 + 14, not +32: ptxas interleaved and reused).  Pool
unchanged (fragments are registers; 36352 B).  6/6 PASS both shapes,
every digit == the record tables; bit-diff scope exactly the predicted
cone (dkB_sum -> rowB_c -> dk_local -> dk -> dg_state -> dg).  Knob-off
== HEAD bytes.  **Perf: dkb 0.1808 -> 0.1825 @T2048 (0.99x) /
0.6024 -> 0.6027 @T8192 (1.00x) paired x3** -- below the 1.05x bar.
Root cause: the cross-vt chain was NOT the binding serial structure
at 2 CTAs/SM (16 resident warps already overlap the two vt chains at
warp-scheduler level) -- the third independent confirmation of the
law the r20 v4 arm established (per-warp/per-CTA work unchanged =>
scheduler ILP already present => no gain).  Default 0, knob kept.

## Lever 2, route (a): the [C,K] w/kg tile halving -- priced NO-GO on paper

The bar: same-kernel 2 CTAs/SM needs pool <= 50688 B from 98560
(T2048, BV=32).  Priced cuts, in the mission's own terms:

* kpp (the r15 k-bf16-exact law): kg = k x 2^(gcl-gc) is a POWER-OF-
  TWO exponent shift -- exactly representable in bf16 -- so the kg
  tile can live IN kpp's own bytes (in-place bf16 scale).  This
  frees wpp's dual-use kg role but saves ZERO bytes by itself: the
  kg result must still exist somewhere, and its natural home (the
  dead w tile) is already what v3 does.
* wpp: gemm-a consumes it BK-slice by BK-slice; shrinking to ONE
  [C,BK] slice (-16384 B) serializes the group-A/group-B staging
  cover -- the v2->v3 FPIPE cadence exists precisely because the
  uncovered DMA was measured expensive.  The mission's "hi/lo
  split-bf16 halves it at 2x tile count cost" prices WORSE than
  zero: 2 x [C,K/2] bf16 == the f32 tile's bytes (split-bf16 never
  halves bytes for a full-precision operand; it only swaps the mma
  operand dtype) while DOUBLING gemm-a's mma count.
* gsc -> GSCG global arm: -7680 B, r14-proven, safe but small.

Sum of the safe cuts: ~24 KB; the pool lands ~74-80 KB -- the bar
needs 47.9 KB.  Closing the remaining gap REQUIRES BV 32->16 (doubles
the walk count with the r18-measured per-walk overhead class), the
gemm-b swap to split-bf16 operands (a NEW numerics class ON the scan
state -- its own CPU twin), and gemm-b K-half streaming (DOUBLES the
per-chunk serial gemm links in a latency-bound kernel).  Five
confounded changes, three of them measured-negative or new-numerics,
to buy an occupancy hypothesis -- NO-GO without a single A/B.  The
same hypothesis is testable through route (b) with ONE twin-proven
value change; route (b) was built instead.

## Lever 2, route (b): 2-CTA K-split + cluster-launch co-scheduling (GDN_FKS2) -- math + machinery PROVEN, perf honest negative, family CLOSED

**The feasibility check FIRST (the mission's CHECK): sm_120 clusters
>1 ARE available -- the G0 "1x1x1 only" claim is DISPROVEN on this
device/driver.**  Two probes (evidence/G5/scratch/r21b_cluster_probe.py,
cluster_rank_probe.{py,ptx}): cuLaunchKernelEx with
CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION accepts 2/4/8 x-dim on this RTX
5090 Laptop (CC 12.0, driver accepts all), and a %cluster_nctarank/
%cluster_ctarank reading kernel proves the hardware executes REAL
cluster mode: plain launch reads nctarank=1/ctarank=0 everywhere;
2x1x1 reads nctarank=2 with pairwise ranks {0,1}; 4x1x1 reads
nctarank=4 with per-group ranks 0-3.  Whatever G0 observed, it does
not bind this device -- documented, with the probe code in the tree.

The r19 twin's design as designed, with the r20 hazard list closed:
pair of 256-thread CTAs per (bh,vt) walk, rank r = %ctaid.x % 2
owning K-rows R_r; p_r = w[:,R_r] @ S[R_r] parks to @pex_all[walk,r]
+ a RELEASE atomic on @flagx[walk,r]; an ACQUIRE spin on the peer
flag; vn = u - (p_0 + p_1) with BOTH partials read from the global in
FIXED ORDER (identical values on both CTAs); S[R_r] *= 2^gcl;
S[R_r] += kg[:,R_r]^T @ vn.  Hazard (1) forward progress: the
cluster attribute makes pair co-residency a DRIVER GUARANTEE (the
F1KS wave-boundary spin deadlock is void; the kernel maps ranks by
%ctaid.x parity -- no cluster SRS reads, a plain launch is still
value-correct whenever the pair co-resides).  Hazard (2) cross-launch
staleness: absolute targets (chunk n spins for n+1) from a harness
memset of @flagx STREAM-ORDERED before every launch (the new
zero_globals manifest contract; bench/verify/det all route through
it; pex_all is write-before-read scratch -- the scratch_globals
contract).  Hazard (3) fencing: probe-verified lowering --
memref.atomic_rmw addf -> atom.acq_rel.SYS.global.add.f32 (stronger
than the .gpu scope the message-passing pattern needs); scf.while ->
bra loop.  Pool: S [K/2,BV] 8192 + wpp [C,K/2] f32 16384 (w half
staged, kg overwrites in place) + kpp bf16 8192 + vn 8192 + acc 8192
+ gr 512 + expc 256 = **49920 B <= the 50688 B 2-CTA bar** (u reads
the global in the vn sweep -- the utl tile does not fit; gsc rides
the GSCG arm at all shapes).  **89 smemfix regs (bar 128), 0 spill**
-- the first fwdscan form ever under BOTH bars: true 2 CTAs/SM, 16
warps/SM, all 128 CTAs of the T2048 grid resident in one wave.

**Math gate: 6/6 PASS both shapes, every digit == the record tables;
h_all BIT-IDENTICAL** (the 2-ulp vn re-association is absorbed by
the tf32 rounding of the S-update products -- vnew_all moves in
18607/4194304 cells, the downstream cone is vnew_all + dkb + dqkwg +
caryA as predicted).  det N=5 fresh-process: ALL stage outputs
bit-identical, both shapes -- the flag protocol is deterministic and
deadlock-free under co-load.

**Perf: fwdscan 0.3762 -> 0.4275 @T2048 (0.88x REGR) / 1.4975 ->
1.6897 @T8192 (0.89x REGR) paired x3** (the T8192 dqkwg 0.89x in the
same window is co-load noise -- that cubin is md5-identical to base).
Decomposition (r21b_cluster_overhead.py): the v3 kernel at grid 128
under a 2x1x1 cluster attribute costs 722.8 us == 2 x 361.4 -- the
cluster attribute itself is FREE; the ks2 delta is +57 us of EXCHANGE
(park stores + sys-scope atomic RTTs + 3x global reads in the vn
sweep).  And the halved per-CTA work bought NOTHING measurable: had
the per-chunk critical path been the K-parallel gemm chain, ks2
would land near 0.5 x 361 + 57 ~ 237 us; measured 424.9.  The scan's
serial structure is the per-chunk STAGING/BARRIER/EXCHANGE latency,
not the gemm width -- the SAME law the r20 v4 arm found from the
warp-group side.  Verdict: GDN_FKS2 default 0, knob kept; with v4
(warp-group), ks2 (two-CTA cluster) and this decomposition, the
fwdscan K-split/residency family is CLOSED at these shapes on three
independent GPU data points.  The cluster-launch harness contract
stays landed for any future cross-CTA design.

## Numerics: 6/6 at BOTH shapes on BOTH arms; every digit equals the record tables

T2048 1.824e-4/1.284e-3/1.088e-5/8.527e-4/3.274e-4/2.169e-7; T8192
2.078e-4/1.881e-3/1.050e-5/1.654e-3/1.145e-3/1.965e-7 -- VTF and ks2
both reproduce the r19-final digits exactly (both re-associations
sit below the record-table resolution; ks2's h_all is bit-identical
outright).  Knob-off byte-identical (final-generator control
r21b_koff2: all 9 x 2 cubins == base).

## Determinism: fresh-process N=5, bit-identical (ks2 arm, both shapes)

ALL stage outputs bit-identical across runs -- including through the
flag/spin exchange (fixed-order p_0 + p_1 on both CTAs; benign
same-value vnew double-writer, the gsc sibling class).

## Ranking after round 21 (round-22 scope)

1. **dqkwg 0.518/1.75 -- the reg diet (the -46-natural-regs door) via
   P11R=2-class accumulator restructuring.**  The ONLY remaining
   measured-open door.  Price the r21 lesson FIRST: the VTF arm
   showed +14 fragment regs survive at 2 CTAs but parallel chains buy
   no stage time -- the diet must come from RETIRING register
   lifetime (restructuring), not from adding interleave.
2. dkb 0.181/0.604: the s5p2 per-vt column barriers (NV=8/chunk
   walk) are the last visible serial structure -- a column-tree or
   batched-barrier form, value-identical class, if a diet opens.
3. fwdscan 0.373/1.475: CLOSED (v4 + ks2 + decomposition; strike
   from all future rankings at these shapes).
4. wyb / caryA / caryM / bpre / dhs / wy: closed at these shapes.

Repro: gen `GDN_DKB_VTF=1|GDN_FKS2=1 python tools/gdn_bwd_gen.py
--shape ... --out DIR` (new knobs, both default 0; all-off == HEAD
bytes); build `bash evidence/G5/scratch/r20b_build.sh <gen-root>` (or
the r21b_*.sh drivers in scratch: base/vtf/ks2/koff2 chains +
verify + A/B + det); cluster probes
`r21b_cluster_probe.py <chain-root> caryM` and
`cluster_rank_probe.py cluster_rank_probe.ptx`; overhead
decomposition `r21b_cluster_overhead.py <base> <ks2>`.  Fixed chains:
r21b_base_T{2048,8192} (== r19-final), r21b_vtf_T*, r21b_ks2_T*,
r21b_koff2_T* (== base); md5s: base == the r20 records 18/18; ks2/vtf
non-fwdscan/dkb stages == base (single-knob arms).


# G5-AA round 22 (R22-bwd): the dqkwg reg-diet door is ALLOCATOR-CLOSED (the audit finding); GDN_DQKWG_K32 lands bit-identical at 1.00x; the r19 staging re-prices stay honest negatives; dkb s5p2 column barriers priced NO-GO on pool

Base rebuilt at HEAD cd88c0a: 18/18 stage cubin md5s == the r20 records
(r22b_base_T{2048,8192}; dqkwg 837a36fcc908 / 1f53c8910006).  Nothing
ships default-on this round; the chain stays the r19-final bytes; e2e
unchanged: co-loaded min ~2.15 ms @B1T2048 (ours/fla ~0.479x, the r19
basis).  The round delivers the diet VERDICT with its cause (a
toolchain law, not a design miss), one bit-identical knob (K32), two
re-priced r19 levers, and the dkb barrier pricing.

## The audit (compile-only; probe generator at scratch/r22b_gen_probes.py, drivers r22b_regaudit*.sh / r22b_structmatrix.sh / r22b_spillcurve.sh / r22b_spillsites.sh)

Phase-deletion (XD), prefix (XP), micro-bisect (XS) and
structural-toggle probes, 38 compile points at T2048:

* PREFIX map (natural smemfix regs): prologue 36 -> +dPr 92 ->
  +r4c4 98 -> +tails 102 -> +dvp 122 -> +P11 128 -> +TW 170 -> +out
  174.  The jump is TW-on-top-of-P11 (+42).
* DELETION map: deleting ANY ONE of dvp / p11 / tw collapses the
  body to 128 -- the +46 overlay is a JOINT dvp+p11+tw allocation,
  not one phase's local need.  Micro: dropping ONLY the TW mma links
  (twmma) also collapses to 128; dwsplit -10; twsplit/twepi 0.
* K=64 FRAGMENT MASS is the overlay's carrier: mma_gemm/mh emit ALL
  kb ldmatrix fragments BEFORE the mma.sync loop, so each Kred=64
  link (the dvp pdo links, the 4 dPm-tail links, the 3 TW links --
  6 sites) holds A(mpw x 4kb x 4regs)+B ~48 regs/thread live at
  once.  This is the r6 "K=32 fragment live-range" law violated 6x;
  the r8 k0= machinery exists precisely to avoid it.
* THE DOOR ITSELF IS CLOSED BY THE ALLOCATOR: the uncapped pass
  reports 174 regs 0-spill (ptxas prefers no-spill over occupancy),
  while the FORCED-.maxnreg allocator NEVER reaches 0 spill at ANY
  cap -- the spill curve at caps 128..174 floors at 40B stores/60B
  loads AT CAP 174 (the directive switches ptxas to a constrained
  allocation that structurally differs from its natural choice).
  "natural <= 128 => 0-spill at .maxnreg 128" is FALSE on this
  toolchain for a body this rich: the r17 door premise does not
  hold, independent of any diet.

## Lever 1: GDN_DQKWG_K32 -- the K=64 links become chained K=32 links.  BIT-IDENTICAL, 1.00x, spill class cut

Every K=64 mma site splits into two chained K=32 links (k0=0/32) via
the _ops_g/_mh_g/_gemm_g wrappers (KGRAN machinery; GDN_DQKWG_K16
deepens to K/16).  The mma.sync SEQUENCE per accumulator fragment is
IDENTICAL -- same operands, same kb order, same acc chaining; only
the ldmatrix issue points interleave in -- so the dataflow is
bit-identical by construction (the r8 docstring equivalence class).

* Regs (double-pass): pass1 220 | smemfix natural 174 (unchanged --
  ptxas keeps its no-spill choice) | .maxnreg-128 spill class
  88B/84/120 -> **64B/64/104**.  K16 adds NOTHING (64/104 too): the
  K=32 sites are not binding.
* Numerics: 6/6 PASS both shapes, every digit == the record tables;
  stage outputs **BIT-IDENTICAL vs base 35/35 @T2048 and 36/36
  @T8192** (the class proof on-device).  det N=5 fresh-process: all
  stage outputs bit-identical.
* Bytes: knob-off == HEAD 9/9 stages x2 shapes (r22b_koff2, built
  with the SHIPPED generator); T512 grid-gated identical (the v3
  dispatch; NOTE the 7de32d2a07c3 T512 record is the r15-era
  unpinned build -- any r20b_build.sh T512 rebuild is 7b0e280e907f
  at BOTH knob states, a harness-history artifact, not a leak).
* **Perf: dqkwg 0.5144 -> 0.5147 @T2048 / 1.7289 -> 1.7345 @T8192
  paired x3 co-loaded (1.00x both)** -- the -20B/-16B spill cut buys
  nothing measurable: the r17 THR=128 spill law's 4th data point.
  dkb/caryA/caryM/bpre swing in the same windows is co-load noise
  (those cubins are md5-identical between arms).
* Verdict: default 0, knob kept (the spill-class cut is free and
  may matter under future co-residency pressure; K16 same).

## Lever 2: the re-prices at the diet point (the mission item 3)

* **P11R=1 + K32** (spill 40B/72 -- the deepest diet state): dqkwg
  0.5141 -> 0.5346 (0.96x REGR) -- the 2-chain body pays the r15
  per-(ks,vs) restage redundancy; the chain count cannot be traded
  for the spill class.  BIT-IDENTICAL outputs, 6/6.
* **K32 + TTF** (the r19 one-shot tT conversion, spill 124B/188 vs
  TTF-alone 128B/232): dqkwg 0.5146 -> 0.5198 @T2048 / 1.7322 ->
  1.7569 @T8192 (0.99x / 0.986x REGR) -- the async group token stays
  live across the P11 loop and the diet does not absorb it.
  BIT-IDENTICAL, 6/6 both shapes.  TTF stays default 0; the r19
  verdict unchanged.
* **DWF at K32** (compile probe only): capped spill 136B/136/208 --
  IDENTICAL to DWF alone; no headroom change, the r18 verdict
  stands without a re-measure.

## Lever 3: dkb s5p2 column barriers -- priced NO-GO on pool (value-identical class)

The per-vt barrier (NV=8/chunk walk) is the r10 restage fence for a
SIX-tile set (dHv 8192 + vngb 4096 + vh/vl 3072x2 + hh/hl 6144x2 =
30720B at BV=16): the next vt's split/emap/stage writes race this
vt's mma/s5 readers.  A batched-barrier (batch=2) form needs the set
2-vt-disjoint = +30720B against a 10240B window (2-CTA bar 50688 -
pool 40448).  Every under-bar subset leaves the fence: hh/hl-only
parity (+12288) exceeds alone; dHv-only parity (+8192, fits) removes
NO barrier (vh/vl/hh/hl/vngb reuse still per-vt) and is exactly the
r20 DHP arm shape (measured 0.97x).  A column-tree through
warp-shuffles would drop the 7 post-loop tree barriers but is a
value-class change (the S5T tree order is load-bearing), outside the
ranking's own "value-identical ... if a diet opens" gate -- and the
diet did not open.  NO-GO; dkb stays 0.180-0.181/0.600-0.604 (today
0.1802-0.1806/0.6006-0.6024 in-window, cubins == records).

## Numerics: 6/6 at both shapes on every arm; every digit equals the record tables

T2048 1.824e-4/1.284e-3/1.088e-5/8.527e-4/3.274e-4/2.169e-7; T8192
2.078e-4/1.881e-3/1.050e-5/1.654e-3/1.145e-3/1.965e-7 (K32, K32+TTF,
P11R1+K32 all reproduce the r19-final digits exactly; K32/K32+TTF
outputs bit-identical outright, P11R1+K32 bit-identical too -- the
r16 order-preserving P11R class).

## Determinism: fresh-process N=5, bit-identical (K32 arm, T2048)

ALL stage outputs bit-identical across 5 runs (r20_logs/det_r22b_k32).

## Ranking after round 22 (round-23 scope)

The bwd chain at B1T2048H16/B1T8192H16 has NO remaining
measured-open door in the value-identical / spill / staging /
residency classes:

1. **dqkwg 0.514/1.73: CLOSED at these shapes.**  The -46 door is
   allocator-closed (this round); spill is not a lever (4 data
   points); the staging family is closed (r18/r19 + this round's
   re-prices); P11R=2 is the measured optimum (chain count vs
   restage).  Remaining classes are NEW-NUMERICS only (e.g. a P11R
   W-park re-association -- priced: smem-park is pool-blocked
   (+16KB vs 2.5KB headroom), global-park costs ~+190KB/CTA L2 RAS
   traffic ~= +10% -- NO-GO on paper) or regime changes (B4+, other
   T/H).
2. dkb 0.180/0.601: CLOSED (column barriers pool-blocked; cross-vt
   mma chain not binding, VTF 0.99x; s5 tree already parallel).
3. fwdscan / wyb / caryA / caryM / bpre / dhs / wy: closed (struck
   in r21).
4. If a round-23 opens: the honest candidates are cross-kernel (the
   K32/K16 fragment-live-range law applied to the OTHER mma-heavy
   stages -- dhs/dkb/wyb natural regs are under their caps, so it
   buys nothing there) or shape-regime work (B4/T-other spot
   ladders).  The default recommendation: bwd optimization at these
   shapes is DONE; hold the r19-final bytes.

Repro: gen `GDN_DQKWG_K32=1 [GDN_DQKWG_K16=1] [GDN_DQKWG_TTF=1]
[GDN_DQKWG_P11R=1] python tools/gdn_bwd_gen.py --shape ... --out DIR`
(new knobs, all default 0; all-off == HEAD bytes 9/9 x2 shapes,
r22b_koff2); build/verify/A/B/det `evidence/G5/scratch/r20b_*` (the
r20 harness); audit drivers `r22b_regaudit.sh` (XD/XP phase probes),
`r22b_regaudit2.sh` (XS micro), `r22b_structmatrix.sh` (toggles),
`r22b_spillcurve.sh` (caps 128..174 + spill-site map),
`r22b_k32probe.sh` / `r22b_armprobe.sh` (arm compiles),
`r22b_k32gate.sh` / `r22b_reprice.sh` / `r22b_cleancheck.sh` (the
gates); probe generator `r22b_gen_probes.py` (compile-only, NEVER
run).  Fixed chains: r22b_base_T{2048,8192} (== r20 records 18/18),
r22b_k32_T*, r22b_k32ttf_T*, r22b_p11r1k32_T2048, r22b_koff2_T*
(== base), r22b_k32c_T* (the clean-generator control, == k32);
dqkwg md5s: base 837a36fcc908/1f53c8910006, k32
fb89448c760e/e615d26e9402.
