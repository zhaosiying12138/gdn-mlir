# G4 GATE: all 20 configs numerically correct on the GPU vs exact fp64 references

Date: 2026-10-09 | GPU: RTX 5090 Laptop (sm_120a), shared/opportunistic window
(co-load pid 3411456 untouched) | Harness: `tools/gdn_verify.py`, evidence in
`evidence/G4/verify_results.json`, per-config build dirs in
`evidence/G4/gen_verify/<name>/`.

## Verdict: 20/20 PASS

Every config was regenerated (`gdn_l2_gen.py --driver noargs --seed 0`),
recompiled through the existing chain (`scripts/run_l2.sh` ->
`tools/gdn_smem_fix.py`), launched on the GPU in a subprocess-isolated worker
with a wall timeout, and its FULL output buffer read back and compared
element-wise against an exact numpy fp64 reference twin computed from the
SAME embedded inputs (`gdn_l2_gen.noargs_val(seed)`). Generator determinism
verified: same seed -> byte-identical payload (sha256), different seed ->
different data (`generator_determinism` in the json).

Thresholds (fla-derived contract, stage outputs graded at o/ht level):
**bf16/f16-io configs: max-abs-diff <= 5e-3; f32-io configs: <= 1e-3.**
Full-op fwd/bwd fla-tolerance validation happens in G5 with the complete
kernel; these stage-level bars are the G4 gate.

## 1. Per-config results

| config | max abs diff | threshold | nonzero/total | deterministic | verdict |
|---|---|---|---|---|---|
| `kkt_bf16_mma_p0_t64x64x32_w4` | 0.000976562 | 0.005 | 2080/4096 | yes | PASS |
| `kkt_bf16_mma_p2_t64x64x32_w4` | 0.000976562 | 0.005 | 2080/4096 | yes | PASS |
| `kkt_bf16_mma_p3_t64x64x32_w4` | 0.000976562 | 0.005 | 2080/4096 | yes | PASS |
| `kkt_bf16_mma_p2_t64x64x16_w4` | 0.000976562 | 0.005 | 2080/4096 | yes | PASS |
| `kkt_bf16_mma_p2_t32x64x32_w4` | 0.000976562 | 0.005 | 2080/4096 | yes | PASS |
| `kkt_bf16_mma_p2_t64x64x32_w8` | 0.000976562 | 0.005 | 2080/4096 | yes | PASS |
| `kkt_bf16_mma_p2_t64x64x32_w2` | 0.000976562 | 0.005 | 2080/4096 | yes | PASS |
| `kkt_bf16_nomma_p2_t64x64x32_w4` | 0.000976562 | 0.005 | 2080/4096 | yes | PASS |
| `kkt_bf16_nomma_p0_t64x64x32_w4` | 0.000976562 | 0.005 | 2080/4096 | yes | PASS |
| `kkt_bf16_nomma_p2_t64x32x32_w4` | 0.000976562 | 0.005 | 2080/4096 | yes | PASS |
| `o_bf16_mma_p2_t64x64x32_w4_K64V64` | 0.000488281 | 0.005 | 4096/4096 | yes | PASS |
| `o_bf16_mma_p3_t64x128x16_w8_K128V128` | 0.000976562 | 0.005 | 8192/8192 | yes | PASS |
| `o_bf16_mma_p0_t32x64x32_w2_K64V64` | 0.000488281 | 0.005 | 4096/4096 | yes | PASS |
| `o_bf16_nomma_p2_t64x64x32_w4_K64V64` | 0.000488281 | 0.005 | 4096/4096 | yes | PASS |
| `o_f32_mma_p2_t64x64x32_w4_K64V64` | 0 | 0.001 | 4096/4096 | yes | PASS |
| `o_f32_nomma_p0_t64x64x32_w4_K64V64` | 0 | 0.001 | 4096/4096 | yes | PASS |
| `ref_f16_sm80_atom` | 0 | 0.005 | 128/128 | yes | PASS |
| `kkt_bf16_mma_p2_t64x64x64_w4_ws` | 0.000976562 | 0.005 | 2080/4096 | yes | PASS |
| `kkt_bf16_mma_tma_ws` | 0.000976562 | 0.005 | 2080/4096 | yes | PASS |
| `ref_ldmatrix` | 0 | 0.005 | 128/128 | yes | PASS |

A second full run with `--seed 7` (different data permutation, fresh
reference twins) also passes 20/20 (`evidence/G4/verify_results_seed7.json`)
-- the gate is not overfit to one data permutation.

All launches are single-shot; every kkt config writes EXACTLY the 2080
strict-tril cells the reference has (G3 measured 1168 on the mma path --
that under-coverage was the bug, see 2.1). `deterministic` = zero + relaunch
reproduced the output buffer bit-for-bit.

## 2. Root causes found and fixed (all in `tools/gdn_l2_gen.py` unless noted)

### 2.1 kkt/o mma-path numeric discrepancy (the G3 "1168 vs 2080" flag) -- D-fragment column misindexing

`frag_epilogue` stored the accumulator's even elements (`extract[0][0]`,
`extract[1][0]`) at column `ncol + %g` -- the lane's **B-fragment** N index
(groupID) -- instead of the **accumulator/D-fragment** N index
`2*threadID_in_group` (`ncol + %t2`). PTX m16n8k16 layouts: B fragment
holds N=groupID per lane, but D holds N=2*tig and 2*tig+1 (c0..c3 =
C[g][2t], C[g][2t+1], C[g+8][2t], C[g+8][2t+1]). Result: two lanes wrote
conflicting values to the same columns while other columns were never
written (kkt: 1168/2080 cells covered; o: 2304/4096). Fix: new
`%ncolt{ni} = ncol + %t2` used for even elements in the epilogue (one-line
semantic fix in `frag_epilogue` + precompute in both `frag_precompute` and
`ws_frag_precompute`). After the fix every mma config writes all 2080 kkt
tril cells / all o cells and matches the fp64 twin at 9.77e-4 (== pure bf16
output quantization; numpy simulation of the exact f32-accum + bf16-round
pipeline predicts 9.765625e-4 -- measured == predicted).

This bug affected 11 of the 20 configs (all mma=on paths incl. ws/tma).

### 2.2 ws deadlock (`kkt_bf16_mma_p2_t64x64x64_w4_ws` G3 TIMEOUT) -- mbarrier arrival-count chicken-and-egg

G2's protocol initialized the single mbarrier with count `32*warps` and had
(a) the producer's elected leader arrive once and (b) EVERY consumer thread
arrive after its mma. Phase i could therefore only complete after the
consumers' post-mma arrivals -- but those same consumers were blocked in
`mbarrier.try_wait.parity` waiting for exactly that phase: deadlock at
iteration 0 (phase 0 needs 128 arrivals, gets 1). Additionally the
arrive/wait raced `mbarrier.init` (no barrier between init and first use),
and the epilogue ran on the producer warp too, whose `wnoff = (warp-1)*wn`
is negative -> negative store columns (OOB writes).

Fix (single-buffer-correct protocol):
* the mbarrier is a pure DATA-READY barrier, `init count = 1`; the
  producer-elect is the ONLY arriver (tma=on: `arrive.expect_tx` + the two
  TMA loads' `complete_tx::bytes` credits complete the phase; tma=off:
  plain arrive after the cp.async group wait);
* consumers only `try_wait.parity` (loop-carried parity, flipped per
  iteration -- this part of G2 was correct and is unchanged);
* `gpu.barrier` right after `gdn.mbarrier.init` (init visibility) and the
  loop-end `gpu.barrier` continues to provide single-buffer reuse safety
  (producer cannot overwrite the tile consumers are reading);
* epilogue guarded to consumer warps (`scf.if %iscons`).

Verified on hardware: completes, 2080/2080 cells, 9.77e-4 vs fp64 twin,
bit-deterministic across relaunches.

### 2.3 TMA wiring (`kkt_bf16_mma_tma_ws` G3 NOT_TIMED) -- three independent gaps

1. **No host encode existed.** The compile-level lowering emits host calls
   to `mgpuTensorMapEncodeTiledMemref` (never executed; the cubin contains
   only the kernel) plus three `.u64 .ptr` kernel params. G4: the generator
   now OWNS the descriptor contract -- `--driver noargs` writes
   `tensormap_spec.json` (rank, CUDA-order globalDim/globalStrides/boxDim
   per descriptor) and `gdn_verify.py` encodes the CUtensorMaps with
   cuda-bindings `cuTensorMapEncodeTiled` exactly per spec, uploads them to
   256B-aligned device memory, and passes the device pointers through the
   three ptr params (each kernelParams entry -> host u64 holding the device
   pointer).
2. **Dim-order contract** (derived from the in-tree runtime helper
   `mgpuTensorMapEncodeTiledMemref`, CudaRuntimeWrappers.cpp: it REVERSES
   memref dim order into CUDA order for globalDim/boxDim/strides, while
   tensormap load/store COORDINATES pass through the pipeline UNREVERSED
   into `cp.async.bulk.tensor` c0..cN, c0 = innermost). G2's payload put the
   K offset in the LAST coordinate slot (an outer, size-1 dim) while any
   valid encode of row-major k must have K innermost (= c0); it also laid
   the B smem tile out as [bk x bv] (N-contiguous), which TMA cannot fill
   from row-major k (N is not contiguous in global). Fixed: CUDA-ordered
   coordinates (K first) in the payload; B tile is a k-rows [bv x bk] tile
   (K contiguous) read with the kt=True b_frag pattern; descA/descB are
   rank-4/5 maps over the k global with K innermost, descS rank-2 over the
   output.
3. **PTX repair gaps for the TMA instructions** (`tools/gdn_smem_fix.py`):
   (a) the TMA load destinations lower to `cvta.to.shared::cluster.u64` of
   the device-malloc result -- the old CVTA_RE only matched plain
   `cvta.to.shared.u64`, so after the malloc callseq was deleted the two
   TMA destinations read never-defined registers (garbage smem address ->
   CUDA_ERROR_ASSERT); now matched, with one static `.shared` symbol
   per malloc result shared by all its cvtas; (b) the static shared tiles
   are now declared `.align 128` -- `UTMALDG` (cp.async.bulk.tensor)
   destinations must be 128B-aligned even with swizzle=none on sm_120a
   (with `.align 16` ptxas placed a tile at a 16B-but-not-128B offset ->
   CUDA_ERROR_MISALIGNED_ADDRESS, pinpointed with compute-sanitizer at
   `UTMALDG.4D`).

Also fixed in `gdn_time.py`: `parse_ptx_params` now matches `.param .u64
.ptr[.align N]` declarations (the tensormap params were invisible to the
launcher before).

Verified on hardware: completes, 2080/2080, 9.77e-4, deterministic.

### 2.4 nomma (linalg) path: bf16 accumulation missed the 5e-3 bar (numerics, not wiring)

G2's nomma payload accumulated `linalg.matmul outs(...)` in **bf16**.
Sequential bf16 accumulation over K=128 rounds away ~1e-2 on the kkt
diagonal: exact simulation of the payload's arithmetic (bf16 mul, bf16
running sum, f32 decay, bf16 store) gives max diff 0.028 vs the fp64 twin
with 165/2080 tril cells above 5e-3 -- the config could never meet the fla
bar. Fix: the acc tile is f32 and `linalg.matmul ins(bf16,bf16) outs(f32)`
-- this tree's named op auto-promotes (verified against
`-convert-linalg-to-loops`: extf operands -> f32 mul/add -> f32 store), so
the payload keeps G2's byte-identical copy/staging structure and only the
accumulator widens. (A `linalg.generic` with an explicit cast region would
be more self-documenting but breaks `gpu-lower-to-nvvm-pipeline` on
workgroup memrefs in this tree: "conversion of memref memory space ...
failed" -- bisected to the generic op itself.) After the fix the nomma
configs match the twin at the same 9.77e-4 as the mma path.

### 2.5 ref_ldmatrix anchor: uniform ldmatrix base = row-broadcast A

G2's payload passed `nvgpu.ldmatrix %sa[%c0, %c0]` (a uniform base). The
in-tree lowering (`MmaLdMatrixOpToNVVM`) passes the ONE per-lane address
register through with no row spreading, so all 32 lanes addressed the same
16B row -> the mma saw a row-broadcast A (GPU output: all 16 output rows
identical, max diff 0.032). Fix: the payload computes the per-lane
quadrant addresses for the m16n8k16 A fragment ([TL, BL, TR, BR]: lane L
-> sa[L%8 + 8*((L/8)%2)][8*(L/16)]) and the x2-trans B rows (lane L ->
sb[L%16]). After the fix the anchor is EXACT (max diff 0.0, like the
non-ldmatrix sm80 anchor).

### 2.6 Input scale + explicit seed (harness prerequisites)

G3's +-3.0 (0.5-step) inputs drive kkt stage outputs to ~1e3, where pure
bf16 output quantization (ulp ~ 4 at |C| ~ 1e3) exceeds ANY absolute
tolerance without indicating a bug. The fla contract's absolute bars assume
O(1) stage outputs, so embedded data is scaled by 2^-5 (values = multiples
of 2^-7 with <=4-bit mantissas: exactly representable in bf16/f16/f32/tf32;
kkt diag ~ 0.44, bf16 output ulp ~ 2e-3, quantization-limited max diff
9.77e-4 < 5e-3 with ~5x margin). `--seed N` (default 0) makes the data
permutation explicit and deterministic; same seed -> identical payload/
inputs (verified by sha256), and the fp64 twins are computed from the same
`noargs_val(i, j, seed)` the payload embeds.

## 3. Evidence chain per config

`evidence/G4/gen_verify/<name>/`: payload.mlir, transform.mlir,
config.json (+ tensormap_spec.json for the tma config), kernel.ptx,
kernel.cubin, asmobj.mlir, kernel_smemfix.ptx/.cubin, build.log, ptxas_v.txt,
sass.txt. `evidence/G4/verify_results.json`: per-config max_abs_diff,
threshold, over-threshold count, nonzero counts vs reference nonzero counts,
determinism flag, worker wall time, plus the generator-determinism probe
(same-seed sha256 equality, seed-changes-data check).

## 4. fla-contract mapping (for G5)

* G4 grades STAGE outputs at o/ht level with absolute bars (5e-3 bf16-io /
1e-3 fp32-io) -- all 20 pass with the largest observed margin ~5x (9.77e-4
vs 5e-3; f32/tf32 and f16-atom stages are exact or ~1e-7).
* The full-op fwd/bwd fla-tolerance validation (atol/rtol per
  flash-linear-attention conventions on complete kernel chains) is G5's
  gate with the complete kernel; G4's twins + harness
  (`tools/gdn_verify.py`, `--seed`) are directly reusable there.
* Numerics contract now uniform across paths: every compute path
  accumulates in f32 (mma f32 accumulators; nomma f32 acc tile;
  ldmatrix/atom anchors exact), inputs are io-dtype-exact, outputs are
  single-quantized to the io dtype.

## 5. Remaining risks / notes for G5

* **nomma pipeline knob**: the nomma transform keeps G2's
  tile_using_for + create_async_groups; the copies still convert to
  cp.async where the transfer-pair pattern matches (514 cp.async in the
  kkt nomma PTX). p0 vs p2 nomma PTX differ only via bypass_l1 (as in G2).
* **ws/tma structural simplifications kept from G2-C** (documented, not
  correctness-relevant): single smem buffer (no multibuffered ping-pong;
  loop-end bar.sync serializes producer/consumer), no producer
  back-pressure beyond that barrier, producer passes accumulators through
  the scf.if. G5's fused kernel should move to double-buffering for
  overlap; the mbarrier protocol here is the correct single-buffer base.
* **TMA smem alignment**: any future TMA destination allocated via
  memref.alloc -> device malloc -> smem_fix rewrite needs the 128B static
  alignment (now automatic in gdn_smem_fix.py); hand-written PTX should
  keep it.
* **Timing**: G3's config_times.json stands for the 18 previously-running
  configs; the ws/tma entries there (TIMEOUT/NOT_TIMED) are superseded by
  G4 -- both now complete in <120s wall (per-launch time in worker wall_s
  only; a proper timing re-run is G5's option via the unchanged
  tools/gdn_time.py).
* Shared-GPU: all measurements in an opportunistic window (<2GB, short
  bursts, per-config subprocess isolation); co-job untouched; row in
  ~/codebase/GPU-COORDINATION.md marked done.

## 6. Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
~/miniforge3/envs/gdn/bin/python tools/gdn_verify.py --seed 0
#   -> evidence/G4/verify_results.json (regenerates + recompiles + reruns)
~/miniforge3/envs/gdn/bin/python tools/gdn_verify.py --seed 7 \
    --out evidence/G4/verify_results_seed7.json   # data-permutation check
```
