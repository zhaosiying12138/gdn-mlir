# G5-B GATE: full-op GDN FORWARD as real GPU kernels — staged trio + persistent fused flagship

Date: 2026-10-09 | GPU: RTX 5090 Laptop (sm_120a, **99KB/CTA smem cap** — see
lesson 3), shared/opportunistic window (co-job pid 3411456 ~17.9GB resident,
untouched) | conda `gdn`, existing compile chain (`scripts/run_l2.sh` ->
`tools/gdn_smem_fix.py` -> cuda-bindings launcher).

Deliverables:

* `tools/gdn_full_gen.py` — full-op forward generator: **staged** path
  (F1 "wy" token-parallel WY precompute / F2 "scan" state recurrence /
  F3 "o" output stage — the fla-isomorphic 3-kernel split) and the
  **persistent fused flagship** (ONE kernel, grid `B*H*V-tiles`, chunk loop
  in-kernel, state resident in smem).  Whole-K operand staging to workgroup
  smem, warp-tiled `nvgpu.mma.sync` m16n8k16 bf16 with f32 accumulators,
  **`nvgpu.ldmatrix` fragment loads** (validated per flavor, lesson 4),
  the L1 forward-substitution solve, and custom epilogues.  Knobs:
  `--warps` (2/4/8), `--bv` (V-tile per CTA), `--frag ldm|scalar`.
* `tools/gdn_full_rt.py` — build + launch runtime (zero-arg cubins with
  public `memref.global` buffers, upload/download via `cuModuleGetGlobal`,
  staged-path DtoD handoffs).
* `tools/gdn_full_verify.py` — numeric validation vs the G5-A shared refs.
* `tools/gdn_full_bench.py` — CUDA-events median-of-50 e2e timing + fla
  speedup table (reads G5-A `fla_baseline.json`).

Evidence: `evidence/G5/fwd_verify*.json` (numerics), `evidence/G5/fwd_bench.json`
(timing), `evidence/G5/gen_fwd/<shape>_<kernel>/` (payload/PTX/SASS/cubins).

## 1. Verdict

**Numerics: PASS on both paths, both validated shapes.** fla contract
(o/ht max-abs-diff <= 5e-3 vs the G5-A JAX fp64/f32 refs, same inputs):

| shape | path | o max-abs-diff | ht max-abs-diff | status |
|---|---|---|---|---|
| B1 T512 H4 KV128 | staged (F1->F2->F3) | **4.77e-4** | **7.00e-4** | PASS |
| B1 T512 H4 KV128 | fused | **4.75e-4** | **7.00e-4** | PASS |
| B1 T2048 H16 KV128 | staged | **5.75e-4** | **7.93e-4** | PASS |
| B1 T2048 H16 KV128 | fused | **5.75e-4** | **7.93e-4** | PASS |

(shipped config warps=8 bv=64, `fwd_verify_w8.json`; identical diffs at
warps=4 in `fwd_verify.json`.)  Margins ~7-10x under the 5e-3 bar — the
residual is bf16 storage of T/u/w/v_new/state plus bf16 mma operands, per
the fla dtype contract (bf16 io + bf16 carried state, ALL math f32).

**Performance: honest miss of the 1.1x fla stretch goal.** Shipped config
(warps=8, BV=64, ldmatrix): best B1T2048 staged 1.29-1.60ms / fused
1.82-2.01ms across passes vs fla pure-Triton fwd median 0.316ms =>
**0.16x-0.25x fla** (fla is ~4-6x faster).  Section 3 has the breakdown
and where the gap lives.

## 2. What was built (and the five real bugs found on the way)

Math is THE L1 verified truth (chunked fwd from `tools/gdn_l1_gen.py`):
gc = inclusive cumsum of `g*log2(e)`; A = strict_tril(beta_i (k_i.k_j)
2^(gc_i-gc_j)); T = (I+A)^-1 by forward substitution; u = T@(beta.v),
w = T@(beta.k.2^gc); v_new = u - w@S; o = ((q.2^gc)@S +
tril_inc((q k^T) 2^D) @ v_new) * K^-0.5; S = S 2^gc_last +
(k.2^(gc_last-gc))^T @ v_new.  dtype contract per G5-A shared recipe
(q/k/v bf16, beta/g/h0 fp32, g NATURAL-log; ht fp32 out; o bf16 out).

Both paths per chunk run 8 mma GEMM stages (kkt, T@bv, T@bk2, w@S, qk,
P@v_new, qg@S, kT-term @ v_new) at 64x{64,BV}x{64..128} with custom
epilogues (decay/beta/mask for A and P, qg/kterm in-place row scaling,
state update S <- S*2^gc_last + kterm^T v_new in the epilogue).

Bugs found and fixed (each verified by an isolated GPU probe before
re-integration — probes kept in this gate's debugging trail):

1. **Module nesting**: a `module {}` wrapper around the payload +
   transform module triple-nests after `-transform-interpreter` and the
   nvvm pipeline then fails kernel outlining ("conversion of memref
   memory space workgroup failed").  Fix: payloads are TOP-LEVEL ops
   (G2 format).  (The G2-C lesson, rediscovered at full-op scale.)
2. **`vector.transfer` to workgroup smem is NOT legalizable** by the
   plain pipeline — `transform.nvgpu.create_async_groups` (the async
   transform) is MANDATORY for the staging loops (G2 always ran it; the
   identity transform silently drops the kernel).
3. **Laptop sm_120a caps smem at 99KB/CTA** (0x18c00, not the 227KB of
   the desktop part): the fused kernel's naive tile set is 148KB.  Fix:
   lifetime-shared smem buffers (bk2 then late-staged q in one tile;
   transposed-T then P in one tile; beta.v then v_new in one tile) =>
   96.5KB.
4. **ldmatrix flavor semantics** were derived and validated per-flavor
   with dump probes (tile val=16r+c, decode lane elements): A-normal
   x4 (row = 16mi + 8*half + lane%8, col = 16k + 8*(lane/16)), A-swapped
   x4 **.trans** (the transposed-T and kterm reads), B kt-True x2
   no-trans, B kt-False x2 **.trans**.  One ldmatrix per fragment
   replaces 12 scalar ld.shared.
5. **Solve correctness**: (a) a warp guard `warp < 32` (should be `< 2`)
   sent warps 2-3 into the solve with out-of-range columns — OOB smem
   writes corrupting neighboring tiles (found with compute-sanitizer:
   "Invalid __shared__ write ... thread (64,0,0)"); (b) the
   warp-per-column shfl butterfly was algorithmically wrong (it summed
   one partial per LANE, i.e. across 32 DIFFERENT columns); replaced by
   the obviously-correct thread-per-column scalar recurrence (L1 order
   preserved, bf16 transposed-T tile consumed by the mmas via the
   swapped-fragment read).  (c) the ht epilogue double-added the V-tile
   offset (vtc) — exactly half of ht was garbage for V-tile 1.
6. **Staged-path handoff**: each kernel is a separate cubin with its own
   globals; F2/F3 consume F1/F2 outputs via device-to-device
   `cuMemcpyDtoD` (~50us of the e2e, included in timings).

## 3. Timing (shared GPU; medians of 50, 10 warmups; lower bounds)

| shape | ours staged (ms) | ours fused (ms) | fla triton (ms) | fla default (ms) | staged/fla | fused/fla |
|---|---|---|---|---|---|---|
| B1T2048 H16 KV128 | 1.597 | 2.010 | 0.316 | 0.347 | 0.20x | 0.16x |
| B1T8192 H16 KV128 | 5.398 | 8.567 | 0.873 | 0.997 | 0.16x | 0.10x |
| B4T2048 H16 KV128 | 4.476 | 4.585 | 0.844 | 1.022 | 0.19x | 0.18x |

(one pass, `evidence/G5/fwd_bench.json`, w8/bv64; an earlier same-config
pass measured staged 1.29ms on B1T2048 — shared-window noise ~20%.
Scaling notes: the FUSED kernel degrades with NC (serial chunk chain:
T8192 has NC=128) and nearly ties the staged path at B4T2048 where its
64-CTA grid finally fills the 82 SMs.)

**Per-stage breakdown (B1T2048, w8):** F1 (WY, 512 CTAs) 0.44ms,
F2 (scan, 32 CTAs) 0.56ms, F3 (o, 1024 CTAs) 0.35ms, handoffs 0.05ms.
fla's own profile for the same shape (G5-A `fla_baseline.json`
kernels_fwd): h-scan 71us, o 42us, recompute_w_u 27us, kkt_solve 22us
(167us self-time total).  **We are uniformly ~8x slower per stage** —
the gap is not algorithmic (same GEMM count/flops), it is execution:

* our whole-K staging is single-buffered with a gpu.barrier between
  every phase (~15 barriers/chunk in the fused kernel); Triton's loops
  are cp.async double-buffered with 2-3 stage pipelines.
* ldmatrix removed the fragment-load instructions but the GEMMs remain
  issue/latency-bound at these tiny tile sizes (64x64x.. per chunk) —
  the fused kernel's chunk chain is fully serial per CTA (60us/chunk
  at 32 CTAs; half the 82 SMs idle at V=128/BV=64).
* the scalar solve costs ~5-7us/chunk on one thread (the correct
  warp-parallel variant needs per-column cross-lane reduction of
  k-slices of the SAME column — designed, not re-attempted after the
  butterfly lesson).
* knob sweep (warps x BV): w8/bv64 best; BV=32 (more CTAs for the
  fused kernel) LOSES — the WY pre-pass is duplicated per V-tile and
  dominates.

**What would close the gap (unstarted, honest):** K-loop re-tiling of
the GEMMs so `transform.nvgpu.pipeline_shared_memory_copies`
multibuffering applies (the G2 machinery), cp.async double-buffered
chunk prefetch in the fused kernel (the "ping-pong" knob in the brief —
the mbarrier protocol from G4 is the right base), warp-parallel solve,
and a chunk-parallel scan split (fla-style two-level pass) to lift the
32-CTA ceiling on F2.  Each is real engineering; with all of them the
roofline (4.0 GFLOP @ ~57 TFLOPs bf16 ~= 70us + traffic) suggests
parity is reachable, but we do not claim it.

## 4. Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
~/miniforge3/envs/gdn/bin/python tools/gdn_full_verify.py --warps 8   # numerics
~/miniforge3/envs/gdn/bin/python tools/gdn_full_bench.py --warps 8    # timing vs fla
```


---

# G5-D round 2: the staged path through the G2 transform pipeline

Date: 2026-10-09/10 | same shared-GPU window discipline (co-job pid 3411456
untouched; ncu/sanitizer runs opportunistic).

## Verdict

**Numerics: PASS everywhere, unchanged margins** (o/ht <= 7.5e-4 vs the G5-A
fp64 refs, bar 5e-3): `evidence/G5/fwd_verify_round2.json`
(small_B1T512H4 + bench_B1T2048H16, staged mono + fused).

**Perf: staged fwd 0.48x / 0.36x / 0.55x fla-triton on
B1T2048/B1T8192/B4T2048** (from 0.20x/0.16x/0.19x in round 1) --
`evidence/G5/fwd_bench_round2.json`, medians of 30, shared GPU:

| shape | r1 staged | r2 staged | r2 f1 | r2 f2 | r2 f3 | fla triton | r2/fla |
|---|---|---|---|---|---|---|---|
| B1T2048 H16 | 1.597ms | **0.663ms** | 270us | 471us | 239us | 0.316ms | 0.48x |
| B1T8192 H16 | 5.398ms | **2.450ms** | 609us | 1479us | 464us | 0.873ms | 0.36x |
| B4T2048 H16 | 4.476ms | **1.636ms** | 620us | 876us | 459us | 0.844ms | 0.55x |

(2.4x/2.2x/2.7x faster than round 1; per-kernel medians include ~5-10us
launch+sync overhead each. The 1.1x goal is NOT reached; gap analysis below.)

## What was done (each lever re-verified numerically before the next)

1. **Every K-loop now runs through the G2 transform pipeline**
   (multibuffer-2 + create_async_groups(bypass_l1) +
   pipeline_shared_memory_copies depth-2 peel): F1 kkt + fused ug/wg loop;
   F2 vp + sd loops; F3 fused qk+oi loop + oa loop.  Thread-STRIDED
   staging (each thread copies its own 16B chunks; the transfer index is
   tid- and iv-dependent) -- the G2-style block-uniform unrolled staging
   makes EVERY thread issue EVERY copy (256x redundant cp.async) and was
   measured 2x SLOWER than round 1 before this fix.
2. **F1 restructure**: Hillis-Steele gc cumsum (6 smem steps; was a serial
   per-thread re-scan of H-strided global g); the beta/beta*2^gc scalings
   folded INTO the solve outputs (Tb = T^T diag(beta), Tg = T^T
   diag(beta*2^gc)) so the consumer GEMMs read RAW v/k via cp.async --
   the round-1 scalar beta*v and beta*k*2^gc staging passes are gone.
3. **Blocked 16x16 solve (the fla retile)**: T[R][C] = -(T[R][R] @
   sum_K Abf[R][K]@Tt[K][C]); diagonal blocks by 16-length
   thread-per-column substitution (f32 A), off-diagonal blocks by
   m16n8k16 (bf16 Abf), stash/consume phases with per-Cb scratch tiles.
   The flat 64-column solve was the F1 critical path (~45us/CTA serial:
   2016 dependent FMAs); the blocked form has ~120.  F1 429 -> 270us.
4. **F2 restructure**: kterm fold (vn scaled by 2^(gcl-gc_r) in the v_new
   epilogue; F3's P epilogue uses the matching 2^(gc_i-gcl) form so global
   v_new carries the scaled tile as-is -- the round-1 kterm in-place RMW
   pass is gone); u staged as one cp.async tile (was 4096 scalar fragment
   loads); h_all/v_new stores coalesced.
5. **F3 restructure**: ONE pipelined loop computes qk AND oi (a single
   staged q tile feeds both mmas); the 2^gc row scaling moved from a
   staged q tile into the f32 epilogue (one less rounding); oa loop reads
   resident P + staged v_new slices; BV=128 (VT=1) halves q/k re-reads.
6. **Mono staged module**: F1/F2/F3 in ONE cubin with shared public
   globals (kernel boundaries are the stream-order syncs) -- the round-1
   DtoD handoff copies (~0.1ms at B1T2048) are gone.  Required extending
   gdn_smem_fix to extract + fix + MERGE the three PTX modules.

## Bugs found on the way (all root-caused with GPU probes)

1. pipeline_shared_memory_copies failures(suppress) STILL kills the
   -transform-interpreter output on any non-copy scf.for (probe-verified
   on a G2 payload + dummy loop).  ALL non-K cell passes and the F2 chunk
   loop are therefore scf.WHILE (not matched by ops{["scf.for"]}).
   scf.while quirks: do-region needs an explicit ^bb0 header with fresh
   arg names, condition yields ALL loop args, args==results types.
2. Buffers with users in several different loops (F2 S/vn/utile/gcs)
   hit multiBuffer()'s hard-fail (no single candidate loop) ->
   transform crash.  A one-line dummy non-loop load+store (pin_alloc)
   makes the transform op skip them.
3. vector.transfer on WORKGROUP memrefs does not lower (G2 constraint 5,
   re-confirmed for READS); vector.transfer_write to a flat GLOBAL inside
   scf.while leaves an unrealized_conversion_cast.  All smem->global
   passes are scalar (coalesced by construction).
4. Mono merge: ptxas silently UNIFIES duplicate .shared symbol names
   across entries -- f1/f2/f3 __gdn_smem0 aliased with mismatched
   sizes -> f3 wrote OOB (compute-sanitizer: shared write at 0xa800).
   The merge now renames per-kernel symbols.
5. Mono wiring: f1 wrote @gc_out while f2/f3 read @gc (the trio bridged
   them with a DtoD copy) -> all-zero gc downstream.  Aliased in gen_mono.
6. Blocked solve: the stash-phase B-fragment column base missed the
   +16*Cb block offset (warp 0 read the right Tt blocks, warps 1/2 the
   wrong ones; per-16x16-block dump diffing found it).  Also: gpu.barrier
   INSIDE warp-guarded scf.if = partial arrivals = broken sync; barriers
   are now unconditional between stash/consume phases.
7. Debug-tool trap worth recording: dumping a per-CTA smem tile to a
   global without a CTA-indexed base races across CTAs -- the dump read
   ARBITRARY chunks' values, which first looked like "A_m is wrong".

## Profile state (ncu, B1T2048)

* F1 270us: latency-bound (DRAM 10%, SM 13%, occupancy 17% = 1 CTA/SM:
  89KB smem + 255 regs).  Post-blocked-solve the kkt/uw loops + epilogues
  dominate; 512 CTAs = 6.25 waves.
* F2 471us: pure latency (DRAM 11%, SM 7%): the serial 32-chunk chain at
  ~14.6us/chunk on 32 CTAs (B1H16 x VT2).  THE remaining bottleneck.
* F3 239us (BV=128): DRAM 57-69% at 31% occupancy (2 CTAs/SM by
  regs/smem); q/k/h_all/v_new traffic ~50MB at ~0.4TB/s achieved.

## Gap analysis vs 1.1x (honest)

fla fwd = 316us at B1T2048; we are at 663us.  Needed: ~2.4x more.

* F2 (471us vs fla h-scan 71us): the chunk recurrence is still a serial
  dependency chain per (b,h,vtile).  Real fixes, in impact order:
  (a) chunk-level double-buffering (prefetch chunk n+1's w/u/k tiles
  during chunk n compute -- the G4 mbarrier ping-pong protocol is the
  proven base; the transform pipeliner cannot cross the chunk while);
  (b) the 2-level segment split (per-chunk transition M_c = 2^gcl I -
  kterm^T w is S-independent; compose per segment, short stitch, rerun
  segments with entry states) -> 4-8x CTAs; (c) BV=32 needs W=4 which
  currently fails the pipeliner preconditions (unresolved).
* F1 (270us vs fla wy+kkt_solve 49us): 1 CTA/SM occupancy; the solve is
  no longer the chain, the two pipelined loops + 4 epilogue/store phases
  per chunk are.  Splitting F1 into F1a (kkt+solve) / F1b (ug/wg) or
  2-CTA residency (smem 89KB -> <49KB by sharing Tb/Tg with Tt) are the
  next levers.
* F3 (239us vs fla o 42us): DRAM-bound at 31% occupancy; register
  pressure (255) is the limiter; the o-store epilogue is scalar 2B
  (could smem-stage + coalesce).

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
~/miniforge3/envs/gdn/bin/python tools/gdn_full_verify.py --warps 8   # round-2 numerics
~/miniforge3/envs/gdn/bin/python tools/gdn_full_bench.py --warps 8    # round-2 timing
# per-stage probe:
~/miniforge3/envs/gdn/bin/python tools/g5d_time_mono.py bench_B1T2048H16 evidence/G5/gen_fwd/bench_B1T2048H16_mono
```

---

# G5-F round 3: the F2 attack -- ping-pong, the REAL root cause, and 1.1x-fla at the flagship

Date: 2026-10-08 | same shared-GPU window discipline (co-job pid 3411456
untouched; all runs short-burst, <4GB; ncu opportunistic).

## Verdict

**F2 at B1T2048: 471us -> 83us (quiet window, 5.7x); staged fwd e2e
0.663 -> 0.262ms** (single-session tight read; interleaved-vs-fla
per-pair medians 1.14x-1.83x across two windows).  Numerics 4/4 PASS,
**bit-identical diffs to round 2** (o/ht <= 7.5e-4 vs the fp64 refs):
 (w8/bv64) and 
(shipped config).

The 1.1x-fla target at the flagship: **met on the interleaved criterion**
(B1T2048 per-pair speedup vs fla-triton 1.143x and 1.827x in two
adjudication windows; absolute medians ours 0.285-0.289ms vs the 0.287ms
target under co-job load, 0.262ms in the quiet read).  B4T2048 at parity
(1.00x-1.36x), B1T8192 still behind (0.87x-1.04x).  Final clean-window
adjudication:  (see 5).

## 1. What actually happened (three findings, one blind alley)

### 1a. Ping-pong alone: NOTHING (the planned lever 1 was not the story)

Chunk-level double buffering exactly per the brief: manual
 bursts (w/k/gc for chunk n+1 into
parity-indexed halves of wide tiles) issued at the TOP of chunk n's
compute,  + barrier at the end, gemm
K-steps fully UNROLLED (no scf.for left in F2 -> nothing for the
pipeliner to break on).  Parity is folded into INDEX ARITHMETIC (wb is
Cx2K halves-along-columns, kb is 2CxK halves-along-rows) because memrefs
cannot be arith.select-ed.

Machine constraint that forced the manual route (source-verified in
this LLVM tree, NVGPU/Transforms/CreateAsyncGroups.cpp): the transform
creates a group and **waits for it right after** -- every
create_async_groups-converted staging is IMMEDIATELY followed by
cp.async.wait_group 0, so the transform route can never overlap a
prefetch with compute.  Manual device_async_copy ops pass through
untouched and the wait placement is ours.

Numerics: bit-identical PASS.  Timing: **471us -> 473us.  Zero.**
The chunk chain was not load-latency-bound after all.

### 1b. THE root cause: 8-way LDSM bank conflicts (MIO saturation)

ncu on the ping-pong F2: still 63% of issue interval stalled
"MIO queue full", 2.0 active warps/scheduler, one LDSM every ~40 cycles
per SM.  Instruction mix: 76 LDSM/chunk/warp.  All ldmatrix-read tiles
had row strides that are 128B MULTIPLES (64x128 -> 128B, 64x256 ->
512B, 32x128 -> 256B...): the 8 rows of every LDSM quadrant land in the
SAME 4 banks = 8-way conflict = ~8x-slow LDSM.  This was true of the
round-2 tiles too (the 43%-MIO stall in the round-2 profile was this).

Fix: **+8 bf16 columns of padding on every ldmatrix-read tile**
(row strides become 128B+16B multiples; 8 rows rotate through all 32
banks; +16B keeps cp.async 16B alignment).  F2 also DE-STAGED u (the vn
epilogue loads it straight from global; its latency hides behind the vp
gemm and it frees 8KB smem + 2 cp.async + 2 shared ops per element).
Result: **F2 473 -> 249us at w8/bv64.**  The same padding was applied to
every F1/F3 ldm-read tile (marginal there -- see 4).

### 1c. Shipped config: warps=4 BV=32 for F2 (lever 3)

BV=32 doubles V-tiles per CTA (VT=4 -> 64 CTAs at B1T2048, still within
the 82 SMs) and halves the per-chunk gemm/epilogue work; the serial
chain latency halves.  F2 96.7KB -> 84.5KB smem.  **F2 83us, staged
262us (quiet window)**; numerics bit-identical again.

### 1d. Blind alley, recorded: in-epilogue global stores REGRESS 40%

Folding the h_all store into the S-update epilogue and the v_new store
into the vn epilogue (saving two cell passes) measured F2 249 -> 380us:
epilogue stores follow the FRAGMENT lane mapping (2B at 4B stride =
uncoalesced), while the separate cell passes are thread-strided and
perfectly coalesced.  Reverted; the copy passes stay.

## 2. Numbers

Per-stage medians (shared window; ncu-free, CUDA events, n=30):

| shape | r2 staged | r3 w8/bv64 | r3 w4/bv32 (shipped) | r3 f2 (w8) | r3 f2 (w4) |
|---|---|---|---|---|---|
| B1T2048 | 0.663 | 0.462 (min .412) | 0.445-0.459 loaded / **0.262 quiet** | 0.267 (min .129) | 0.083 quiet |
| B1T8192 | 2.450 | 1.366 (min 1.268) | 1.335-1.348 | 0.648 | 0.651 |
| B4T2048 | 1.636 | 1.199 (min 1.114) | 1.172-1.187 | 0.495 | 0.498 |

(; w4/bv32 medians are co-job-loaded -- the
quiet single-session read is recorded inside the same file.  The
loaded-window f2/f3 at w4 sit at ~0.25/0.22 medians; the quiet read
f2=0.083ms is the capability number.)

INTERLEAVED adjudication (tools/gdn_final_adjudicate.py, 5 pairs/shape,
ours-then-fla alternating, per-pair median speedup):

| shape | run 1 | run 2 (different window) |
|---|---|---|
| B1T2048 | **1.143x** (flagged NOISY) | **1.827x** (clean) |
| B1T8192 | 0.867x | 1.036x |
| B4T2048 | 1.000x | 1.357x |

(; both runs measured the SAME
w4/bv32 cubins -- run 2 requested w8/bv64 but the build cache served
the w4 cubins; annotated in the file.  Ours-median at B1T2048 was
0.285-0.289ms in both, vs fla 0.316-0.517 across windows.)

## 3. F2 final form (tools/gdn_full_gen.py --f2 pp, default)

Per chunk (scf.while, everything else unrolled): group-A cp.async
prefetch of chunk min(n+1,NC-1) w/k/gc into wb/kb/gcs halves [parity
n^1] -> gcl load -> coalesced h_all entry-state pass -> vp gemm (A=wb
half, B=S) -> vn epilogue (u from global, exp2 fold, writes vn) ->
barrier -> coalesced v_new pass -> sd gemm (A=kb half swapped, B=vn) ->
S-update epilogue -> wait group A -> barrier.  3 barriers/chunk (was
~8).  All tiles +8-col padded.   keeps the round-2 kernel.

## 4. F1/F3 (untouched logic, same padding; honest non-results)

F1: 270 -> 260-285us (noise-level; LSU/LG-throttle-bound now, NOT
LDSM-bound: ncu shows 557K shared-STORE bank conflicts + 46.8% memory
throughput; the solve/epilogue scalar STS pattern is the next real
lever -- a warp-shuffle store swizzle or 128-bit STS via vector.load
experiments).  F3: unchanged within noise (DRAM-bound at 31% occupancy,
as in round 2).

## 5. Adjudication machinery (for the final clean window)

: per shape, 5x{ours fwd, fla fwd}
alternating pairs, CUDA events, reports ratio-of-medians AND
median-of-per-pair-ratios (divergence = dirty window flag), smi
snapshots, full sample arrays; shared-recipe inputs bit-identical for
both sides; FLA_FLASH_QLA=0 set pre-import with backend probe recorded.
Provisional runs above; the final adjudication re-runs this in a clean
window.

## 6. Next levers (quantified)

* F2 83us vs fla h-scan 71us at B1T2048: essentially closed at the
  flagship; the residual is the first-chunk prefetch + prologue.
* B1T8192 (f2 0.65ms, 128-chunk chain, 64 CTAs): THE remaining case for
  the segment-split scan (lever 2, unstarted): 2-pass with composed
  per-segment transitions M_seg = prod(2^gcl I - kterm^T w) needs one
  extra KxKxC gemm/chunk + 7 K^3 matmuls/segment (fp32-composed, bf16
  composition risks the 5e-3 gate) -- at 8 chunks/segment the chain
  drops 4x for ~2.7x the work; ~250us projected at B1T8192.
* F1: scalar-store swizzle (557K STS conflicts) + the round-2 noted
  F1a/F1b split or 2-CTA residency.
* F3: register cap (255) for 3+ CTAs/SM at BV=128.

## Reproduce



---

# G5-F round 3: the F2 attack -- ping-pong, the REAL root cause, and 1.1x-fla at the flagship

Date: 2026-10-08 | same shared-GPU window discipline (co-job pid 3411456
untouched; all runs short-burst, <4GB; ncu opportunistic).

## Verdict

**F2 at B1T2048: 471us -> 83us (quiet window, 5.7x); staged fwd e2e
0.663 -> 0.262ms** (single-session tight read; interleaved-vs-fla
per-pair medians 1.14x-1.83x across two windows).  Numerics 4/4 PASS,
**bit-identical diffs to round 2** (o/ht <= 7.5e-4 vs the fp64 refs):
`fwd_verify_round3.json` (w8/bv64) and `fwd_verify_round3_w4bv32.json`
(shipped config).

The 1.1x-fla target at the flagship: **met on the interleaved criterion**
(B1T2048 per-pair speedup vs fla-triton 1.143x and 1.827x in two
adjudication windows; absolute medians ours 0.285-0.289ms vs the 0.287ms
target under co-job load, 0.262ms in the quiet read).  B4T2048 at parity
(1.00x-1.36x), B1T8192 still behind (0.87x-1.04x).  Final clean-window
adjudication: `tools/gdn_final_adjudicate.py` (see 5).

## 1. What actually happened (three findings, one blind alley)

### 1a. Ping-pong alone: NOTHING (the planned lever 1 was not the story)

Chunk-level double buffering exactly per the brief: manual
`nvgpu.device_async_copy` bursts (w/k/gc for chunk n+1 into
parity-indexed halves of wide tiles) issued at the TOP of chunk n's
compute, `device_async_wait numGroups=0` + barrier at the end, gemm
K-steps fully UNROLLED (no scf.for left in F2 -> nothing for the
pipeliner to break on).  Parity is folded into INDEX ARITHMETIC (wb is
Cx2K halves-along-columns, kb is 2CxK halves-along-rows) because memrefs
cannot be arith.select-ed.

Machine constraint that forced the manual route (source-verified in
this LLVM tree, NVGPU/Transforms/CreateAsyncGroups.cpp): the transform
creates a group and **waits for it right after** -- every
create_async_groups-converted staging is IMMEDIATELY followed by
cp.async.wait_group 0, so the transform route can never overlap a
prefetch with compute.  Manual device_async_copy ops pass through
untouched and the wait placement is ours.

Numerics: bit-identical PASS.  Timing: **471us -> 473us.  Zero.**
The chunk chain was not load-latency-bound after all.

### 1b. THE root cause: 8-way LDSM bank conflicts (MIO saturation)

ncu on the ping-pong F2: still 63% of issue interval stalled
"MIO queue full", 2.0 active warps/scheduler, one LDSM every ~40 cycles
per SM.  Instruction mix: 76 LDSM/chunk/warp.  All ldmatrix-read tiles
had row strides that are 128B MULTIPLES (64x128 -> 128B, 64x256 ->
512B, 32x128 -> 256B...): the 8 rows of every LDSM quadrant land in the
SAME 4 banks = 8-way conflict = ~8x-slow LDSM.  This was true of the
round-2 tiles too (the 43%-MIO stall in the round-2 profile was this).

Fix: **+8 bf16 columns of padding on every ldmatrix-read tile**
(row strides become 128B+16B multiples; 8 rows rotate through all 32
banks; +16B keeps cp.async 16B alignment).  F2 also DE-STAGED u (the vn
epilogue loads it straight from global; its latency hides behind the vp
gemm and it frees 8KB smem + 2 cp.async + 2 shared ops per element).
Result: **F2 473 -> 249us at w8/bv64.**  The same padding was applied to
every F1/F3 ldm-read tile (marginal there -- see 4).

### 1c. Shipped config: warps=4 BV=32 for F2 (lever 3)

BV=32 doubles V-tiles per CTA (VT=4 -> 64 CTAs at B1T2048, still within
the 82 SMs) and halves the per-chunk gemm/epilogue work; the serial
chain latency halves.  F2 96.7KB -> 84.5KB smem.  **F2 83us, staged
262us (quiet window)**; numerics bit-identical again.

### 1d. Blind alley, recorded: in-epilogue global stores REGRESS 40%

Folding the h_all store into the S-update epilogue and the v_new store
into the vn epilogue (saving two cell passes) measured F2 249 -> 380us:
epilogue stores follow the FRAGMENT lane mapping (2B at 4B stride =
uncoalesced), while the separate cell passes are thread-strided and
perfectly coalesced.  Reverted; the copy passes stay.

## 2. Numbers

Per-stage medians (shared window; CUDA events, n=30):

| shape | r2 staged | r3 w8/bv64 | r3 w4/bv32 (shipped) | r3 f2 (w8) | r3 f2 (w4) |
|---|---|---|---|---|---|
| B1T2048 | 0.663 | 0.462 (min .412) | 0.445-0.459 loaded / **0.262 quiet** | 0.267 (min .129) | 0.083 quiet |
| B1T8192 | 2.450 | 1.366 (min 1.268) | 1.335-1.348 | 0.648 | 0.651 |
| B4T2048 | 1.636 | 1.199 (min 1.114) | 1.172-1.187 | 0.495 | 0.498 |

(`fwd_bench_round3.json`; w4/bv32 medians are co-job-loaded -- the
quiet single-session read is recorded inside the same file.  The
loaded-window f2/f3 at w4 sit at ~0.25/0.22 medians; the quiet read
f2=0.083ms is the capability number.)

INTERLEAVED adjudication (tools/gdn_final_adjudicate.py, 5 pairs/shape,
ours-then-fla alternating, per-pair median speedup):

| shape | run 1 | run 2 (different window) |
|---|---|---|
| B1T2048 | **1.143x** (flagged NOISY) | **1.827x** (clean) |
| B1T8192 | 0.867x | 1.036x |
| B4T2048 | 1.000x | 1.357x |

(`final_adjudicate_w4bv32_run{1,2}.json`; both runs measured the SAME
w4/bv32 cubins -- run 2 requested w8/bv64 but the build cache served
the w4 cubins; annotated in the file.  Ours-median at B1T2048 was
0.285-0.289ms in both, vs fla 0.316-0.517 across windows.)

## 3. F2 final form (tools/gdn_full_gen.py --f2 pp, default)

Per chunk (scf.while, everything else unrolled): group-A cp.async
prefetch of chunk min(n+1,NC-1) w/k/gc into wb/kb/gcs halves [parity
n^1] -> gcl load -> coalesced h_all entry-state pass -> vp gemm (A=wb
half, B=S) -> vn epilogue (u from global, exp2 fold, writes vn) ->
barrier -> coalesced v_new pass -> sd gemm (A=kb half swapped, B=vn) ->
S-update epilogue -> wait group A -> barrier.  3 barriers/chunk (was
~8).  All tiles +8-col padded.  `--f2 r2` keeps the round-2 kernel.

## 4. F1/F3 (untouched logic, same padding; honest non-results)

F1: 270 -> 260-285us (noise-level; LSU/LG-throttle-bound now, NOT
LDSM-bound: ncu shows 557K shared-STORE bank conflicts + 46.8% memory
throughput; the solve/epilogue scalar STS pattern is the next real
lever -- a warp-shuffle store swizzle or 128-bit STS via vector.load
experiments).  F3: unchanged within noise (DRAM-bound at 31% occupancy,
as in round 2).

## 5. Adjudication machinery (for the final clean window)

`tools/gdn_final_adjudicate.py`: per shape, 5x{ours fwd, fla fwd}
alternating pairs, CUDA events, reports ratio-of-medians AND
median-of-per-pair-ratios (divergence = dirty-window flag), smi
snapshots, full sample arrays; shared-recipe inputs bit-identical for
both sides; FLA_FLASH_QLA=0 set pre-import with the backend probe
recorded.  Provisional runs above; the final adjudication re-runs this
in a clean window.

## 6. Next levers (quantified)

* F2 83us vs fla h-scan 71us at B1T2048: essentially closed at the
  flagship; the residual is the first-chunk prefetch + prologue.
* B1T8192 (f2 0.65ms, 128-chunk chain, 64 CTAs): THE remaining case for
  the segment-split scan (lever 2, unstarted): 2-pass with composed
  per-segment transitions M_seg = prod(2^gcl I - kterm^T w) needs one
  extra KxKxC gemm/chunk + 7 K^3 matmuls/segment (fp32-composed; bf16
  composition risks the 5e-3 gate) -- at 8 chunks/segment the chain
  drops 4x for ~2.7x the work; ~250us projected at B1T8192.
* F1: scalar-store swizzle (557K STS conflicts) + the round-2 noted
  F1a/F1b split or 2-CTA residency.
* F3: register cap (255) for 3+ CTAs/SM at BV=128.

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
~/miniforge3/envs/gdn/bin/python tools/gdn_full_verify.py --warps 4 --bv 32   # numerics (shipped cfg)
~/miniforge3/envs/gdn/bin/python tools/gdn_full_bench.py --warps 4 --bv 32    # timing vs fla
~/miniforge3/envs/gdn/bin/python tools/gdn_final_adjudicate.py --warps 4 --bv 32   # interleaved
```


---

# G5-I round 4: the F2 segment-split -- built, proven exact, honestly refuted as a perf lever

Date: 2026-10-08 | same shared-GPU window discipline (co-job pid 3411456
resident, untouched; util 17-23% during runs; all runs short-burst <4GB).

## Verdict (three sentences)

**The segment-split scan is implemented (`--f2-segments S`), numerically
green everywhere -- including a new T8192 fp64 gate where o/ht are
BIT-IDENTICAL across S=1/4/8 -- and it makes every shape SLOWER** (T8192
f2 0.640 -> 0.780-0.817ms; staged 1.298 -> 1.507-1.541ms), because after
the round-3 ping-pong kernel F2 is per-chunk-THROUGHPUT-bound, not
chain-latency-bound: equal-work chain-shortening (8x shorter chains + 2
CTAs/SM co-residency) recovers only 17% of F2's runtime, far less than
the ~100% extra pass the 2-pass split costs.  **Provisional adjudication
(shipped serial config, interleaved, clean windows): B1T2048 1.292x,
B4T2048 1.191x (both cross 1.1x), B1T8192 1.021x (per-pair 0.982x --
parity, does NOT cross).**  Best S per shape: **1 everywhere**; the
shipped default is unchanged (round-3 serial F2).

## 1. The math first (one correction to the round-4 brief)

The F2 chunk recurrence is AFFINE in the state, not state-independent:
`v_new = u - w@S` makes the per-chunk map `S' = 2^gcl S + kterm^T(u - w@S)`
= `M_n S + b_n` with `M_n = 2^gcl I - kterm^T w`.  A general exact
parallel split therefore needs the composed linear part in the handoff:
`E_{sg+1} = M_seg E_sg + c_sg` (G5-F's plan: one extra KxKxC gemm/chunk +
~7 K^3 composes/segment, ~3.4x F2 flops -- and 88KB smem, 1 CTA/SM, after
which the compose kernel alone rivals the whole serial F2).  The brief's
"matmul-free handoff" (`E_{sg+1} = c_sg`) is NOT exact in general.

It IS exact here, with enormous margin, and for a structural reason:
every factor M_n = 2^gcl(I - K_n) carries the full chunk decay (the
2^-gc in kterm and the 2^+gc in w cancel per token), so
`||M_seg|| ~ 2^(sum gcl over the segment) x O(||prod(I-K)||)`.  The GDN
bench recipes draw `g = -softplus(0.5x)`, mean log-decay ~ -0.25/token,
so a 64-token chunk decays by ~e^-16 and ||M_seg|| is measured
(`tools/g5i_split_sim.py`, exact fp64 composition over the actual bench
inputs) at **<= 3.6e-13 for L=1 segments (T512, S=8) and 4.3e-52 for L=4
(T2048, S=8)** -- the dropped term `M_seg @ E` is far below bf16
resolution (~4e-3 relative).  The CPU bf16 kernel-emulation of the split
(they are `evidence/G5/g5i_split_sim.txt`):

| case | exact split (with M) vs serial | matmul-free vs fp64 ref |
|---|---|---|
| T512/T2048, S=2/4/8, bf16-emulated | 0.0 (bit-equal) | o 4.1-8.7e-5 (serial bf16: 1.8-2.1e-4) |

Validity boundary, documented by the `--stress-g 0.02` probe
(`g5i_split_sim_stress.txt`): with near-zero net decay (gate ~ 1,
segment decay ~ e^-1.3) ||M_seg|| grows to 0.07-0.45, the matmul-free
entry error reaches 3.1e-2 (would breach the 5e-3 gate), while the
exact-composition variant stays at 3.7e-4 vs serial.  The matmul-free
split is therefore exact CONDITIONAL on per-segment net decay -- which
every input in this project's contract has; inputs with gate ~ 1 need
the M-composition fallback (designed above, NOT implemented -- recorded
as a leftover).

## 2. What was built

`tools/gdn_full_gen.py --f2-segments S` (mono path; default 1 = the
round-3 serial kernel, bit-for-bit):

* **f2a** (pass A, grid B*H*VT*S): every segment runs its NC/S chunks
  from a ZERO entry; per chunk = the r2 pipelined body (emit_kloop
  scf.for K-loops, transform-multibuffered) with the r3 +8-col LDSM pads
  and the r3 de-staged u; writes ONLY the segment exit to `@seg_exit`
  (interior h_all/v_new are wrong under a zero entry and are not
  written).
* **f2c** (pass C, same grid): re-runs every segment with its TRUE entry
  (h0 for segment 0, `@seg_exit[sg-1]` otherwise) and writes the real
  outputs (h_all entry states, v_new, ht from the last segment).
* Single-buffered state tiles: **43,264B smem, 206/230 regs -> 2
  CTAs/SM co-residency** (the serial pp kernel is 84.5KB -> 1 CTA/SM);
  this was the intended silent second lever (chain-latency hiding).
* One real bug found and fixed on the way (per-chunk bisection with
  `tools/g5i_dbg_intermediates.py`): pass A initially skipped segment 0
  ("its exit is unused") -- wrong, segment 1's entry IS c_0; seg_exit[0]
  stayed zero-init and h_all[2] was garbage while ht still passed
  (entry errors decay away), a nicely instructive near-miss.

Harness: verify/bench/adjudicate now launch whatever kernel list the
spec declares and take `--f2-segments`; per-config gen dirs
(`*_mono_s{S}`) close the stale-cubin hole that bit the round-3
adjudication run 2.

## 3. Numerics (green after every step, as required)

* `fwd_verify_round4_s4.json` / `..._s8.json`: **4/4 PASS** at
  T512+T2048 (staged split + fused), o <= 6.6e-4, ht <= 7.9e-4 vs the
  fp64 refs -- same margins as the serial kernel (5e-3 bar).
* **New T8192 fp64 gate**: `tools/g5i_t8192_ref.py` (one-off CPU JAX
  chunked spec on the shared-recipe inputs, bf16-rounded q/k/v) wrote
  `evidence/G5/refs/bench_B1T8192H16_{inputs,ref}.npz`;
  `tools/g5i_t8192_check.py`
  (`fwd_verify_round4_t8192.json`): o 6.85e-4 / ht 7.90e-4 vs fp64
  (PASS, 7x margin), **o and ht BIT-IDENTICAL across S=1/4/8** (the
  decayed entry deltas vanish below bf16 resolution, so the split
  reproduces the serial kernel bit-for-bit -- stronger than the brief's
  "identical when possible"), and vs fla fwd: o 9.8e-4 / ht 1.1e-3
  (bf16-scale agreement, bar 1e-2).

## 4. Performance: the honest negative (Amdahl included)

Same-window medians (n=30, co-job resident; `fwd_bench_round4.json`):

| shape | staged S=1 | staged S=2 | staged S=4 | staged S=8 | f2 S=1 | f2 S=4 | f2 S=8 |
|---|---|---|---|---|---|---|---|
| B1T2048 | **0.448** | 0.481 | 0.478 | - | 0.267 | 0.295 | - |
| B4T2048 | **1.155** | 1.556 | 1.532 | - | 0.492 | 0.777 | - |
| B1T8192 | **1.298** | - | 1.541 | 1.507 | 0.640 | 0.817 | 0.780 |

Per-kernel at T8192 (S=8): f1 0.618, **f2a 0.428, f2c 0.529**
(f2a+f2c back-to-back 0.780), f3 0.438.  Two readings:

1. **The split's second pass costs more than parallelism recovers.**
   f2a is a full chunk-work pass (compute-only: no h_all/v_new stores --
   which is why it beats f2c by ~100us of store traffic); f2c repeats
   the work plus the stores.  2x work at ~1.2x better execution =
   a 16-33% LOSS at every shape.
2. **The chain is no longer the binding constraint -- the Amdahl
   number.**  f2c ALONE at S=8 is the equal-work probe (all NC chunks,
   16-chunk chains instead of 128, 2 CTAs/SM instead of 1): 0.529ms vs
   the serial 0.640ms.  Chain length + co-residency together are worth
   **17% of F2** (111us of 640); the other 83% is per-chunk throughput
   (staging traffic, mma issue, ~8 barriers/chunk, epilogues) -- all of
   which the split duplicates.  G5-F's ~250us projection modeled the
   650us serial F2 as latency-bound (650 x 2.7 / 8); the measurement
   refutes the model: at T8192 the round-3 pp kernel already runs F2 at
   its per-chunk throughput bound (64 CTAs, ~78% of SMs, per-chunk
   5.0us vs the quiet-T2048 2.6us -- the delta is T8192's colder w/u/k
   staging, not the chain).

Corollary: no scan restructure (segments, waves, chained CTAs) can beat
the serial kernel at these shapes; only reducing per-chunk cost can
(fewer barriers, larger effective tiles, store coalescing -- the r3
leftovers list stands).

## 5. Provisional adjudication (shipped config, interleaved)

`evidence/G5/final_adjudicate_g5i_run1.json` (5 pairs/shape, ours-then-
fla, CUDA events, FLA_FLASH_QLA=0, shared-recipe inputs; co-job
resident the whole time -- window quality honestly flagged, see
snapshots in the file):

| shape | ours med | fla med | ratio-of-medians | per-pair median | window |
|---|---|---|---|---|---|
| B1T2048 | 0.2846ms | 0.3676ms | **1.292x** | 1.289x | clean |
| B1T8192 | 1.1286ms | 1.1527ms | **1.021x** | 0.982x | clean |
| B4T2048 | 0.9874ms | 1.1761ms | **1.191x** | 1.146x | clean |

B1T2048 and B4T2048 cross 1.1x provisionally; **B1T8192 sits at parity
(0.98-1.02x) and does NOT cross** -- closing it needs ~130us somewhere
in f1+f2+f3 per-chunk costs, not a scan split.  (fla's medians in this
window are slower than the G5-A baseline file -- the co-job hits fla
too; that is exactly what the interleaved criterion is for.)

## 6. Leftovers

* F2 per-chunk throughput (the real 83%): barrier count (8/chunk in the
  r2-style body; the pp body's 3), h_all/v_new store coalescing at
  source, larger BK slices; F1 (0.56-0.62ms at T8192) is now the
  biggest single kernel and keeps its r3 leftovers (STS swizzle,
  F1a/F1b split); F3 register cap.
* The M-composition fallback for near-unity-decay inputs: designed
  (88KB smem forces a 2-CTA column-half split of the compose; K^3 work
  ~1.5x F2 base) but NOT implemented -- the matmul-free split's
  validity condition (net per-segment decay) is checked by
  `g5i_split_sim.py`, which shippers should run if the input contract
  ever changes.
* A clean-window final adjudication rerun remains pending (co-job
  still resident at press time).

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
~/miniforge3/envs/gdn/bin/python tools/g5i_split_sim.py            # split math + bounds (+ --stress-g 0.02)
~/miniforge3/envs/gdn/bin/python tools/gdn_full_verify.py --warps 4 --bv 32 --f2-segments 8   # 4/4
~/miniforge3/envs/gdn/bin/python tools/g5i_t8192_check.py --segs 1,4,8 --fla                   # T8192 gate
~/miniforge3/envs/gdn/bin/python tools/gdn_full_bench.py --warps 4 --bv 32 --f2-segments 8    # the negative result
~/miniforge3/envs/gdn/bin/python tools/gdn_final_adjudicate.py --warps 4 --bv 32              # interleaved
```


---

# G5-K round 5: closing B1T8192 -- the F1 store restage is the whole game

Date: 2026-10-09 | same shared-GPU window discipline (co-job pid 3411456
resident, untouched; util 17-28% during runs; all runs short-burst <4GB).
Knobs: w4/bv32/bv3=V, mono trio, F2 = round-3 serial pp (unchanged).

## Verdict

**B1T8192 crosses: interleaved adjudication 1.253x (5 pairs) / 1.312x
(8 pairs, second window) vs fla -- from round 4's 1.021x/0.982x parity.**
B1T2048 1.592x, B4T2048 1.052x-1.189x across two windows (ours stable
at 0.98-1.00ms; the swing is fla's median, 1.06-1.17ms, its min is
0.86-0.88ms in every window).  The lever that did it is in F1, not F2:
the u/w epilogue's fragment-mapped 2B global stores were the F1 binding
constraint; restaging them through the reused uw stage tiles costs zero
extra smem and takes F1 from 0.59 to 0.47ms at T8192 (-123us measured by
paired A/B).  Every F2 lever tried this round is an honest negative --
recorded below with numbers.

## 0. What the audit actually found (correcting the brief)

* **The shipped pp F2 body has 2 gpu.barrier/chunk, not ~8.**  Generated
  payload + SASS: f2 = 3 total (2 in the chunk while + prologue), f1 =
  23, f3 = 7(+1 new).  The "~8/chunk" figure describes the r2-style body
  (gen_f2_r2 / the seg kernels), which is NOT the shipped kernel.  The
  barrier lever was therefore mostly spent before round 5 began.
* F1 sits at 98.8KB smem of the 99KB cap (101,120/101,376B) -- only 256B
  of headroom, which constrains every F1 idea (below).
* `evidence/G5/gen_fwd/B1T8192_mono` was a STALE w8/bv64 cubin served to
  round-4 bench/adjudication runs through the build-cache hole (config
  mismatch discovered this round); all round-5 numbers use fresh gen
  dirs, and the dir now holds the shipped w4/bv32+F1UW build.

## 1. Probe: no 128-bit global store path exists in this toolchain

`vector.load`/`vector.store` (and a `vector<8xbf16> -> vector<4xi32>`
bitcast variant) LOWER through run_l2.sh but element-split to b16 ops
(`st.global.b16 x8` in PTX; probe: scratch, /tmp/g5k_probe).  The
"vectorize the h_all/v_new cell passes" direction is closed: the passes'
64B/warp scalar stores are already the best this pipeline emits.

## 2. Per-lever attribution (interleaved A/B, new harness)

Absolute medians swing 5-10% with the co-job, so every lever was
measured by `evidence/G5/scratch/g5k_ab.py`: both cubins loaded in one
process, single-kernel launches alternated A,B x50-60 with CUDA events,
pairwise deltas + sign counts (jsons: scratch/g5k_ab_*.json).

| lever | B1T8192 delta | pairs | verdict |
|---|---|---|---|
| F2 barrier 2->1/chunk (vn row-ping-pong + v_new pass moved to top of next iteration; sd B-reads proven warp-local in N) | **+19.7us** | 55/60 slower | **REJECTED**: the +5KB smem (84.5->89.6KB) shrinks the L1 carveout the u-loads/store passes rely on; costs more than the saved bar.sync.  Code kept behind G5K_F2BAR=1. |
| F1 A_m store-tile pad [64,64]->[64,65] f32 (kills the 8-way STS pattern behind r3's 557K conflicts; +256B = exact 101,376B fit) | -0.8us | 28/60 | **non-result**: conflicts are real but off the critical path.  G5K_F1PAD=1. |
| F3 o-epilogue restage (padded ot tile + coalesced cell pass) | -1.3us f3 | 22/60 | keep (default on): noise-level positive, principled, B1T2048 f3 -8.4us. |
| **F1UW** (below) | **-123.3us f1** | **4/60 slower** | **SHIPPED, default on** |

## 3. The winning lever: F1UW (uw section restructure)

Three changes in one, all in gen_f1's uw stage (u = T(beta v),
w = T(beta k 2^gc) gemms):

1. **bk=C (64), single trip, straight-line**: the uw gemms' K-reduction
   is exactly C=64, so the scf.for degenerates.  (The 1-trip scf.for
   form makes pipeline_shared_memory_copies hard-error with "pipelining
   preconditions failed" -- the section is emitted gen_f2-style instead:
   manual thread-strided nvgpu.device_async_copy + fully unrolled mma.)
2. **Staging issued BEFORE the solve**: the v/k tiles are disjoint from
   every solve tile, so the forward-substitution's serial latency hides
   the 32KB copy that previously stalled the kloop's first iteration.
3. **u/w outputs restaged through the (now dead) stage tiles**: vt/k2
   become [C, N+8] single-buffered -- exactly the bytes their
   [32,136]x2 multibuffered selves cost, so F1 stays at 101,120B total.
   The epilogues write the padded tiles (conflict-free STS), and one
   coalesced cell pass per tile moves them to global (64B/warp row
   segments), replacing 128 scattered STG.U16 per thread (the r3 ncu
   diagnosis: LSU/LG-throttle-bound, "46.8% memory throughput").

Paired A/B (vs the round-4 generator, fresh cubins):

| shape | f1 | staged e2e | pairs (f1 slower) |
|---|---|---|---|
| B1T8192 | 0.416 -> 0.293 (quiet-window single-kernel) | 1.113 -> 0.993 (**-119us**) | 4/50 |
| B4T2048 | 0.415 -> 0.293 | 1.168 -> 1.019 (**-149us**) | 0/50 |
| B1T2048 | 0.121 -> 0.084 | 0.265 -> 0.215 (**-50us**) | 2/50 |

(f2/f3 move <10us in either direction across shapes -- side effects of
changed L2 state, not kernel changes.)

## 4. Numerics (green after every step, as required)

* `fwd_verify_round5.json` (staged+fused, T512+T2048): **4/4 PASS**,
  o <= 6.6e-4 / ht <= 7.9e-4 vs the fp64 refs -- BIT-IDENTICAL diff
  values to the round-4 kernels (the restage preserves the truncf
  rounding; only store ordering changed).
* T8192 fp64 gate rerun against the SHIPPED cubin after catching the
  stale-cubin hole (`gen_fwd/B1T8192_mono` purged + rebuilt):
  **o 6.846e-4 / ht 7.902e-4, PASS** (7x margin), same values as round 4.

## 5. Final numbers

Bench (`fwd_bench_round5.json`, co-job window, median-of-40): B1T2048
staged 0.373 (f1 0.233 / f2 0.275 / f3 0.185), B1T8192 staged 1.126
(f1 0.471 / f2 0.576 / f3 0.431), B4T2048 1.170.

Interleaved adjudication (5x{ours,fla} alternating, CUDA events,
shared-recipe inputs, FLA_FLASH_QLA=0; `final_adjudicate_g5k_run1/2.json`):

| shape | run 1 (5 pairs) | run 2 (8 pairs) | round 4 |
|---|---|---|---|
| B1T2048 | **1.592x** | - | 1.292x |
| B1T8192 | **1.253x** | **1.312x** (per-pair 1.304) | 1.021x (0.982x) |
| B4T2048 | 1.052x | **1.189x** (per-pair 1.184) | 1.191x |

B4T2048's split reading is fla's MEDIAN variance (1.056 vs 1.165 across
the two windows; fla min stable at 0.865-0.875), not an ours-side
regression: ours medians are 0.980-1.004ms vs round 4's 0.987ms while
the paired A/B shows ours -148us at this shape.  All windows flagged
clean in-file; co-job resident throughout (the interleaved criterion).

## 6. Leftovers (measured, ranked)

* F2 at T8192 is still 83% per-chunk throughput (r4's Amdahl): the
  barrier lever is now measured NEGATIVE (L1-carveout cost), the store
  passes are already coalesced with no 128-bit path available, and
  2-ahead prefetch needs 4-way ping-pong buffers that do not fit smem.
  What remains: the h_all/v_new pass instruction count itself, and the
  64-CTA/12.5%-warp occupancy (a <=49.5KB F2 redesign for 2 CTAs/SM).
* F3: DRAM-read-bound (r3 diagnosis unchanged); the o-restage is in but
  worth ~nothing at T8192 -- the next lever is occupancy (register cap).
* F1 residual 0.47ms at T8192: the solve and the kkt kloop; A_m STS pad
  available (G5K_F1PAD) but measured off-path.

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir   (F1UW + F3O are generator defaults)
~/miniforge3/envs/gdn/bin/python tools/gdn_full_verify.py --warps 4 --bv 32   # 4/4
~/miniforge3/envs/gdn/bin/python tools/g5i_t8192_check.py --segs 1            # T8192 gate
~/miniforge3/envs/gdn/bin/python tools/gdn_full_bench.py --warps 4 --bv 32    # timing
~/miniforge3/envs/gdn/bin/python tools/gdn_final_adjudicate.py --warps 4 --bv 32   # interleaved
# per-lever A/B attribution (both cubins, interleaved single-kernel pairs):
~/miniforge3/envs/gdn/bin/python evidence/G5/scratch/g5k_ab.py --shape B1T8192 \
    --a <base_dir> --b <variant_dir> --reps 60 --only f1,f2,f3,staged
# reversions: G5K_NOF1UW=1 (uw kloop back to r2 pipelined form),
#             G5K_NOF3O=1 (o epilogue direct stores); opt-ins: G5K_F2BAR, G5K_F1PAD
```
---

# G5 round 14 (R14-U): fwd T8192 pushed over 1.1x -- the F2 u-staging lever; the 2-CTA question closed by measurement

Date: 2026-10-08/10 | clean idle window discipline (0 compute apps at
every pre-run check; window snapshots in the JSONs; coordination row
`gdn-mlir R14-U`).  HEAD through the round: e987da5 (rebuilt baselines);
shipped at the end of this round: mono trio with **GDN_F2US** for the
T8192 and B4T2048 builds.

## Verdict

**B1T8192 formally crosses: interleaved adjudication 1.101x (per-pair
median 1.104x, 30 pairs, p ~ 1e-9, clean window) -- from 1.021x
(p=0.099) parity.  ours 0.8835 ms vs fla 0.9729 ms.**  The lever is in
F2 (again not where the brief pointed): the v_new epilogue's
fragment-mapped 2B global u-loads were costing ~104us of F2 at T8192;
staging u through a row-ping-ponged cp.async tile (riding the existing
group-A prefetch) removes the amplification from the serial chain --
f2 0.4017 -> 0.2972 ms, staged e2e 0.9551 -> 0.8583 ms in the paired
A/B (0/60 pairs slower).  Every OTHER angle this round is an honest
negative, and together they CLOSE the r5-documented "<=49.5KB 2-CTA
F2 redesign" question:

* **B1T2048 1.333x** (p ~ 1e-9, 30 pairs; archived formal 1.464x; same
  w8/bv64 cubin as the full re-measure, ours 0.2889 vs 0.2894 -- no
  regression).
* **B4T2048: shipped the US build and read 1.094x** (per-pair 1.097x,
  p ~ 1e-9, 30 pairs, ours 0.9171 ms) -- decisively > 1 but a hair
  under the 1.1 bar in THIS window; the archived formal verdict 1.140x
  (all 30 pairs > 1) stands as the shape's record.  Pre-ship (w8/bv64,
  identical cubin to the re-measure) the same formal run read 0.994x
  with ours 0.9923 ms vs the re-measure's 1.1941 ms absolute -- the
  swing is fla-side median variance under idle-clock cycling (A1/A6),
  ours-side the cubin only got faster (paired A/B for the US ship:
  staged -91.4 us, 0/60 slower).  Across windows B4T2048 straddles the
  bar with ours cubins monotonically improving; reported as an honest
  near-miss, not a win claim.
* **Flagship (criterion 6): honest reclassify, path (b) of the
  brief.**  The fused kernel took the portable staged wins (LDSM +8
  pads everywhere incl. a +1 f32 A_m store pad, the Hillis-Steele gc
  log-scan, the F3O-style coalesced o restage; gated to BV<=48 because
  the padded pool 114,672B does not fit the w8/bv64 build):
  **7.52 -> 6.35 ms @T8192 (1.18x), numerics 4/4, det N=5.**  It still
  loses 7.2x to the staged trio (0.88 ms) because the chunk chain is
  serial per CTA: F1's WY work parallelizes 2048-wide in the trio (grid
  (NC, B*H)) but collapses to the 64-CTA chain in the fused kernel --
  that structural difference is not fixable by porting tiles; it needs
  chunk-level WY/scan pipelining (r15).  The staged mono trio carries
  the flagship criterion (the full-remeasure's de-facto reading, now
  explicit).

## 1. The winning lever: F2US (u through the prefetch pipe)

The r3 F2 loads u straight from GLOBAL in the v_new epilogue
(`epi_vn`), one bf16 load per accumulator element under the mma
fragment lane map: 2B loads at 4B stride within 16B windows -- every
32B sector delivers half its bytes, and each load sits on the serial
per-chunk chain (vp mma -> epi_vn -> sd mma -> S update).  At T8192
that is 4 KB/chunk/CTA amplified to ~8 KB of sectors, ~32 MB -> 64 MB
aggregate, with L2-latency exposure on the chain.  F2US stages the u
VT-slice into a row-ping-ponged smem tile `[2C, BV+8]` (+10,240B; F2
pool 84,480 -> 94,720B, still 1 CTA/SM) via the SAME manual
`nvgpu.device_async_copy` burst as w/k/gc -- issued in group A at the
TOP of the chunk (overlapping ALL of the chunk's compute), drained by
the existing end-of-chunk wait+barrier.  The epilogue then reads
conflict-free padded smem.  Values are bit-identical (same bytes, same
arithmetic order -- the T8192 gate and the verify diffs reproduce r5
exactly).

Knob: `GDN_F2US=1` (gen-time env; `GDN_F2US` unset regenerates HEAD
payloads byte-identically).  Shipped build provenance (md5 of
kernel_smemfix.cubin): `B1T8192_mono` (w4/bv32+US)
`b574395a5a66d063aa099f90761c080c`, `B4T2048_mono` (w4/bv32+US)
`89a42d5f8a838d8421fa9c3163012e63`, `B1T2048_mono` (w8/bv64, untouched
re-measure cubin) `bf56dde8ada28178b569558f09fcc7bb`.  NOTE: US cannot
be the unconditional default -- at w8/bv64 the ut tile pushes the pool
to 115,200B > 101,376B (ptxas fail); hence the env-knob + per-shape
build request, exactly the documented T8192 knob-provenance pattern.

## 2. The closed question: <=49.5KB 2-CTA F2, measured four ways

The r5 leftover said "2 CTA/SM halves the serial chain exposure per
SM" (the r4 f2c S=8 equal-work probe: 17% of F2 from shorter chains +
co-residency).  Round 14 built the real thing and measured:

| angle | f2 delta (paired, 60 pairs) | slower-pairs | verdict |
|---|---|---|---|
| F2SB diet: single-buffered wb/kb/gcs (pool 84,480 -> **50,432B** = 2 CTAs/SM fit), prefetch folded at tile-death (wb after the post-vp barrier = overlaps sd+S-epi; kb+gc at the bottom = overlaps the next top passes), 3 barriers/chunk | **+21.2 us** | 57/60 | REJECTED |
| F2SB + forced 64KB L1 carveout (`CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`, the r14u_ab.py probe) | **+20.9 us** | 60/60 | the L1-carveout rescue hypothesis is refuted |
| F2SB @B4T2048 (grid 256 CTAs -> 2 CTAs/SM co-residency ACTUALLY ENGAGES, 164 resident) | **+28.5 us** | 56/60 | co-residency does not pay for the lost top-issued full-chunk prefetch overlap |
| bv3=32 f3 split (f3 pool 45,432B -> 2 CTAs/SM, 8192 CTAs) | f3 **+357.2 us** | 60/60 | the qk gemm + q/k staging duplicate per VT3 CTA; occupancy loses to duplicated work |

Root cause of the negatives: the pp kernel's group-A prefetch issued
at the TOP of chunk n overlaps ALL of chunk n's compute; every
single-buffered fold moves the transfer AFTER part of that compute
(death-fence-bound), and the exposure costs more than 2-CTA latency
hiding recovers -- even with a real 2x-CTA grid (B4) and even with a
forced 64KB carveout.  The diet code is kept behind `--f2 sb` /
`GDN_F2SB=1` (default off).

**F1 2-CTA diet: non-viable by arithmetic** -- the <=50.7KB pool
requires dropping the F1UW restage tiles (vt/k2, 34,816B -- the r5
-123us win) and/or the blocked-solve tile set (A_m f32 16,384 + Abf
9,216 + Tt 9,216 + St 2,304 + Tb/Tg 18,432); no combination keeps the
measured wins under the cap.  Documented, not attempted.

**F3 maxnreg: a non-lever, corrected from the brief** -- the merged
cubin reads f3 = 240 regs / 82,176B smem: registers would ALLOW 2
CTAs/SM (240 x 128 x 2 = 61,440 <= 65,536) but smem does not (2 x
82,176 > 100KB), so a register cap cannot move occupancy.  Tooling
landed anyway for future probes: `tools/gdn_smem_fix.py` now accepts
`GDN_MAXREG_K="f3:168"` (per-ENTRY `.maxnreg` injection -- the ptxas
`--maxrregcount` flag is per-file, useless for the 3-kernel mono
cubin).

## 3. Per-lever attribution (interleaved paired A/B, 60 pairs each)

| lever | shape | kernel delta | staged e2e delta | pairs slower |
|---|---|---|---|---|
| F2SB diet | B1T8192 | f2 +21.2us | +32.7us | 57/60 |
| F2SB + carveout64 | B1T8192 | f2 +20.9us | +30.7us | 60/60 |
| F2SB | B4T2048 | f2 +28.5us | +34.9us | 56/60 |
| bv3=32 | B1T8192 | f3 +357.2us | +345.9us | 60/60 |
| **F2US** | **B1T8192** | **f2 -104.5us (0.4017->0.2972)** | **-96.8us (0.9551->0.8583)** | **0/60** |
| **F2US** | **B4T2048** | **f2 -101.2us (0.4365->0.3353)** | **-91.4us (0.9890->0.8976)** | **0/60** |

(jsons: `scratch/r14u/ab_f2sb_T8192.json`, `ab_f2sb_carve64_T8192.json`,
`ab_f2sb_B4T2048.json`, `ab_bv3x_T8192.json`, `ab_f2us_T8192.json`,
`ab_f2us_B4T2048.json`).

## 4. Numerics + determinism (green after every step)

* `scratch/r14u/fwd_verify_r14u_f2sb.json`, `_flav.json`, `_f2us.json`:
  **4/4 PASS each** (o <= 6.6e-4 / ht <= 7.9e-4 vs the fp64 refs) --
  staged diffs BIT-IDENTICAL to the r5 records for every variant (the
  diet/restage preserves values; only store order changed).
* T8192 fp64 gate (fresh-built cubins): base o 6.846e-4 / ht 7.902e-4
  PASS; F2SB and F2US outputs **BIT-IDENTICAL** to base
  (`scratch/r14u/r14u_t8192_gate.py` runs recorded in the logs).
* Determinism N=5 fresh-process (`tools/gdn_fwd_det.py`): mono trio
  with GDN_F2US and the refreshed fused kernel both
  **FWD-DETERMINISTIC** (`scratch/r14u/det*/`).
* Knob-off arms: `GDN_F2US` unset / `GDN_FLAV0=1` / `--f2 pp`
  regenerate the HEAD payloads byte-identically (generator diff is
  strictly additive).

## 5. Final adjudication (30 interleaved pairs/shape, --stats, clean window)

`scratch/r14u/final_adjudicate_r14u.json` (+ `_b4ship.json`):

| shape | ours med | fla med | ratio-of-med | per-pair med | p | vs re-measure |
|---|---|---|---|---|---|---|
| B1T2048 | 0.2889 | 0.3853 | **1.333x** | 1.320x | ~1e-9 | was 1.464x (win re-confirmed; ours cubin unchanged, fla median faster this window) |
| **B1T8192** | **0.8835** | 0.9729 | **1.101x** | **1.104x** | ~1e-9 | **was 1.021x parity -> CROSSES** |
| B4T2048 (pre-ship cubin) | 0.9923 | 0.9865 | 0.994x | 0.974x | 0.58 | was 1.140x; ours absolute improved, fla-side swing (A1/A6) |
| B4T2048 (US ship) | 0.9171 | 1.0037 | **1.094x** | 1.097x | ~1e-9 | archived 1.140x stands; this window a hair under |

Decidability rule still not met on any shape (IQR measures intrinsic
fla jitter even clean); acceptance rests on p < 0.01 + clean flags, as
in the re-measure.  T8192's 1.101x is the round target met.

## 6. Flagship refresh details

`gen_fused` changes (default ON for BV<=48, `GDN_FLAV0=1` reverts):
+8-bf16-col pads on k_t/QK/S/w_t/TP/BVN (+1 f32 col on A_m) -- the r3
LDSM root-cause fix; `emit_cumsum_beta_scan` (Hillis-Steele, the F1
form) replaces the serial per-thread gc cumsum; o epilogue restaged
through a padded ot tile + one coalesced cell pass.  Pool @w4/bv32:
99,072B.  The w8/bv64 fused build keeps the round-2 design (padded
pool 114,672B > 101,376B -- the refresh does not fit there; documented
gate).  Bench: 7.5196 -> 6.3549 ms @T8192
(`scratch/r14u/fwd_bench_r14u_flav_T8192.json`).

## 7. Round-15 list (measured, ranked)

1. **B1T2048 US ship candidate**: w4/bv32+US at T2048 untested as an
   e2e A/B (F2US scales with NC, expect ~-25-30us); current w8/bv64 is
   1.333-1.464x so the risk is low, the upside small.
2. **F2 residual 0.297ms @T8192**: next is the h_all/v_new scalar
   store-pass instruction count (2B stores, no 128-bit path exists) and
   the 4-warp mma issue width; the chain is now store/issue-bound.
3. **F1 2-CTA via arena aliasing**: alias vt/k2 onto the post-solve
   dead tiles (Abf/Tt/St) with the r13 smem_alias.json machinery --
   the fwd chain has no alias-spec writer yet (bwd-side only); a real
   ~50KB F1 would halve the 25-wave grid @T8192.
4. **Fused pipelining**: chunk-level WY-vs-scan overlap (producer
   warps / async stages) is the only route to a fused flagship that
   beats the trio; 6.35ms has ~5.5us/chunk of serial solve+staging to
   hide.
5. bwd-side r14 items (sibling workstream).

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
GDN_F2US=1 ~/miniforge3/envs/gdn/bin/python tools/gdn_full_gen.py --path mono \
    --shape 1,8192,16,128,128 --warps 4 --bv 32 --bv3 128 --out <dir>   # the shipped T8192 build
GDN_F2US=1 ~/miniforge3/envs/gdn/bin/python tools/gdn_full_verify.py --warps 4 --bv 32  # 4/4
~/miniforge3/envs/gdn/bin/python evidence/G5/scratch/r14u/r14u_t8192_gate.py \
    --dirs base=<HEAD dir>,ship=<US dir>                                 # T8192 gate + bit-identical
~/miniforge3/envs/gdn/bin/python tools/gdn_fwd_det.py --tag bench_B1T2048H16 \
    --path mono --warps 4 --bv 32 --out det.npz                          # determinism (GDN_F2US=1 build)
~/miniforge3/envs/gdn/bin/python tools/gdn_final_adjudicate.py --stats --pairs 30 \
    --warps 8 --bv 64                                                    # formal (cache serves per-shape builds)
# paired A/B: scratch/r14u/r14u_ab.py (--shape --a --b --reps --only; CARVEOUT_A/B env)
# negatives: --f2 sb (diet), GDN_F2SB/GDN_MAXREG_K knobs; fused revert: GDN_FLAV0=1
```

# G5 round 15 (R15-X): B4T2048 ship decided in-session; WST wide-store cell passes; F1 arena machinery ported + honestly closed

Date: 2026-10-10/11 | clean idle window discipline (0 compute apps at
every pre-run check; snapshots in the JSONs; coordination row
`gdn-mlir R15-X`).  HEAD through the round: 4d5916d; shipped at the end
of this round: mono trio with **GDN_F2US + GDN_WST** for the T8192 and
B4T2048 builds (B1T2048 keeps its untouched w8/bv64 re-measure cubin).

## Verdict

* **B4T2048 ship decision: the US build wins, by same-session paired
  measurement, not by fla-window luck.**  Two independent 36-pair
  sessions (`scratch/r15x/r15x_b4ship.py`: both ours builds + fla in
  ONE window, launch order rotated [fla, A, B] per iteration so
  position bias cancels over 12 full rotations):
  * session 1 (fla median 1.1432 ms): pre-US (w8/bv64, the re-measure
    cubin `006c5234`) 1.1132x (per-pair 1.1209x, p=7e-05); US (w4/bv32,
    `89a42d5f`) **1.2226x** (per-pair 1.2375x, p ~ 1e-9, min pair
    1.0472 -- all 36 pairs > 1).
  * session 2 (fla median 1.0110 ms): pre-US 0.9638x (per-pair 0.981x,
    p=0.87 -- parity); US **1.0696x** (per-pair 1.0744x, p ~ 1e-9, min
    pair 1.0047 -- again all pairs > 1).
  * Direct same-window A-vs-B paired delta (immune to fla swings):
    **US faster by 92.5 us (35/36) and 104.1 us (36/36)** -- 71/72 pairs
    over both sessions.
  The r14 "fla-side median swing" thesis is confirmed directly: pre-US
  straddles 1.0 with fla's window (0.96x-1.11x) while US stays > 1 in
  every pair of both windows.  DECISION: **US is the ship**; the
  archived formal 1.140x stands as the shape's record, this round's
  dual-build measurement is the provenance for WHICH cubin.
* **WST (wide-store cell passes) ships on top of US**: the f1 u/w and
  f3 o smem->global passes emit PAIRED cells (2i, 2i+1) in the
  generator and `gdn_smem_fix.py` fuses each ld.shared.b16 x2 +
  st.global.b16 x2 quartet into one `ld.shared.b32` + `st.global.b32`
  (byte-exact; the 32-bit word IS the two bf16 cells).  f1
  **-11.8/-11.4/-10.6 us** (T8192 x2, B4T2048; 1-5/60 pairs slower),
  staged e2e **-13.4/-9.0/-7.0 us**, numerics BIT-IDENTICAL (T8192 gate
  o/ht diffs 0.000e+00), det N=5 FWD-DETERMINISTIC.
* **T8192 spot adjudication after the ship refresh: 1.1309x** (12
  interleaved pairs, p=4.9e-4, clean; ours 0.8440 ms vs r14's formal
  0.8835 ms) -- up from the r14 formal 1.101x, WST visible e2e.
  B1T2048 spot 1.2293x (untouched cubin, window-consistent with the
  1.333x formal); B4T2048 spot 1.0846x (fla-slow window; the ship
  decision above is the number that matters for the shape).
* **F2 store-pass attack: half negative, half closed by geometry.**
  Deletion probes put the two f2 cell passes at **h_all 35.4 us +
  v_new 17.9 us** of the 297 us T8192 chain (`ab_probe_hall/vnew.json`;
  ~53 us combined ceiling).  The b32 pair attack on f2 measured
  **+4.3 us** (47/60): f2's BV=32 tiles hold only 16 pairs per row, so
  a warp's 32 b32 shared loads span two rows and the 80B (20-word) row
  stride puts lanes 28-31 on the same banks as lanes 0-3 -- 2-way
  conflicts the single-cell mapping does not have.  The pair rule is
  therefore cols >= 64 (f1's 128-col rows keep every warp inside one
  row); f2 keeps its original passes.  The mma issue-width item
  (r14 #2) is CLOSED by measurement: w4/bv64 (the only in-budget
  per-warp N=16 config; US cannot fit there) reads f2 **+218.4 us**
  (60/60) -- wider per-warp N does not pay for the halved CTA
  parallelism over the serial chunk chain.
* **F1 arena-alias diet: machinery PORTED and PROVEN, diet measured
  NEGATIVE -- knob OFF, question closed by arithmetic.**
  `gdn_smem_fix.py main_multi` now consumes `smem_alias.json`
  (module-keyed "modules" map; the lifetime-arena machinery was
  single-kernel-path-only before) and `gdn_full_gen.py` writes the
  fwd-side spec under `GDN_F1AL=1` (unique-size 8-col-granular tile
  pads, vt's cp.async group moved to the solve's phase-1|2 boundary
  via a mid_hook, k2's to post-solve, vt->A_m arena, k2->Abf arena,
  solve tiles->kt arena).  Two toolchain finds on the way: (a)
  odd-column pads are ILLEGAL (ldmatrix needs 16B-aligned row
  addresses -> row strides must be multiples of 16B -> pads in
  8-column steps); (b) aliasing the solve tiles at kt-arena OFFSET 0
  corrupts u rows 0-31 in every CTA (the pipelined kkt loop still
  writes kt's multibuffer around the post-loop drain) -- the tiles
  must sit ABOVE kt's real bytes (`bisect_shift` probe, bit-exact).
  Final: pool 101,376 -> **89,600B**, values bit-identical (T8192
  gate), but f1 **+4.7 us** (52/60), staged +5.9 us -- the post-solve
  k2 exposure with zero occupancy payoff.  The 2-CTA target is
  unreachable by aliasing: phase-1 co-holds gcs+bss + A_m + Abf + the
  solve-output arena ~= 57.4KB (unpadded) > the 50.7KB 2-CTA cap;
  breaking it needs the A_m-free (bf16-solve) algorithm change, not
  tile packing.

## 1. The B4T2048 ship decision (the round's #1)

`scratch/r15x/preUS_B4T2048_w8bv64` is the re-measure cubin extracted
from git (10c673e, md5 `006c5234e6ca722a3c311a6bde92903e`); the US
side is r14's ship (`89a42d5f...`).  One session, three contenders,
rotated order (12 rotations of [fla, A, B] = 36 samples/side):

| session | fla med | pre-US med | US med | pre-US vs fla | US vs fla | paired US-preUS |
|---|---|---|---|---|---|---|
| 1 | 1.1432 | 1.0270 | 0.9351 | 1.1132x (p=7e-05) | **1.2226x** (p~1e-9, min 1.047) | **-92.5 us (35/36)** |
| 2 | 1.0110 | 1.0490 | 0.9451 | 0.9638x (p=0.87) | **1.0696x** (p~1e-9, min 1.005) | **-104.1 us (36/36)** |

(jsons: `b4ship_adjudication.json`, `_run2.json`; windows clean,
sm_clock 1620; IQR-based decidability still false -- intrinsic fla
jitter, as in every round; the PAIRED column is the decision basis.)

## 2. WST: the wide-store cell passes (the round's win)

The r15 deletion probes quantified the cell passes (timing-only builds,
`GDN_F2_PROBE`, not shipped): h_all 35.4us / v_new 17.9us of f2's 297us
at T8192; f1's u/w passes are twice as long per CTA (C x V + C x K
cells).  WST attacks the instruction count: the generator's
`copy_s2g_vec` gains a paired-cell mode (GDN_WST=1; cols >= 64 and
rows*cols % (2*nthr) == 0, else the original single-cell body) and
`gdn_smem_fix.py` gains `widen_stores()` (same env; scoped per entry
via GDN_WST_SCOPE): every unconditional quartet
`ld.shared.b16 [B1+o]; ld.shared.b16 [B1+o+2]; st.global.b16 [B2+g];
st.global.b16 [B2+g+2]` (single-use b16 regs, both orders of the two
stores -- LLVM schedules the +2 store first) fuses to
`ld.shared.b32 [B1+o]; st.global.b32 [B2+g]` -- a byte-exact round
trip, half the LSU instructions and half the loop iterations.  The
minted b32 registers are declared as an explicit comma list (bounded
`.reg` ranges reject new names; the from-to decl form is rejected by
this ptxas).

| A/B (60 pairs, vs r14 US ship) | f1 | f2 | f3 | staged |
|---|---|---|---|---|
| WST @T8192 (run1/run2) | **-11.8 / -11.4 us** (1-5/60 slower) | +2.5* / -0.0 | -0.4 / -0.4 | **-9.0 / -13.4 us** |
| WST @B4T2048 | **-10.6 us** (3/60) | -0.1 | -3.1 | **-7.0 us** |

*f2 is byte-identical code under WST (excluded by the cols>=64 rule);
the +2.5us reading did not reproduce (-0.0 in the fresh session) --
cubin layout noise.

The f2 side of the lever is the honest negative: pair mapping alone
(no fusion) measured +4.3us (47/60) on f2 -- the 16-pairs-per-row
geometry (BV=32 tiles) forces every warp's b32 shared loads across two
rows of an 80B-stride tile and the 20-word row stride banks lanes
28-31 onto lanes 0-3's banks.  Documented as the constraint any future
f2 store-pass work must solve (warp-level pairing via shuffle with
single-cell conflict-free loads, or a >=64-col row geometry).

## 3. F1 arena-alias: the port, the two bugs, the closure

The bwd's lifetime-arena machinery existed only on the single-kernel
smem_fix path; `main_multi` (the fwd trio's path) never read a spec.
Round 15 ports it (module-keyed specs) and writes the first fwd-side
spec (GDN_F1AL=1).  Mechanical summary: 13 f1 mallocs -> 5 decls
(256+256 gcs/bss, kt-arena 53,248, A_m-arena 17,408 hosting vt,
Abf-arena 18,432 hosting k2); the kt-arena holds the solve tiles ABOVE
kt's real 10,240 multibuffer bytes (Tb@10,240, Tt@23,552, St@35,840..,
Tg@38,912).

Findings, in the order the GPU reported them:
* odd-column pads -> CUDA_ERROR_MISALIGNED_ADDRESS (ldmatrix row
  addresses must stay 16B-aligned; pads moved to 8-column steps,
  sizes 11,264/12,288/13,312/14,336/1,024/1,280/18,432);
* solve tiles at kt-arena offset 0 -> u rows 0-31 wrong in EVERY CTA,
  deterministic (racecheck clean; `bisect_vkonly` proved the vt/k2
  arenas + moved copy issue bit-exact; `bisect_shift` proved the
  kt-region overlap was the corruptor) -- the pipelined kkt loop's
  cp.async discipline still touches kt's buffers around the post-loop
  drain; anything aliased under 10,240B must assume late writes;
* the diet itself: pool 101,376 -> 89,600B, bit-identical outputs, but
  f1 +4.7us (52/60) / staged +5.9us -- k2's post-solve copy exposure
  (vt keeps phase-2 overlap via the mid_hook) with no occupancy change
  buys nothing.  GDN_F1AL stays OFF; the machinery is landed and
  reusable (bwd-style arenas now available to any fwd kernel).
* 2-CTA arithmetic, final: phase 1 co-holds gcs+bss (512) + A_m
  (16,384) + Abf (9,216) + the solve-output tiles (Tb/Tt/St/Tg
  31,296 unpadded) = 57,408B > 50,688B.  Every alias-permutation hits
  this floor; only dropping A_m (bf16 solve -- a numerics experiment,
  not a packing change) or restructuring the solve gets under it.

## 4. Numerics + determinism (green after every step)

* `scratch/r15x/fwd_verify_r15x_wst.json`: **4/4 PASS** (staged o
  6.598e-4 / ht 7.493e-4 @B1T2048 -- the r5/r14 record digits exactly;
  the store fusion is byte-exact).
* T8192 fp64 gate: ship (US+WST) vs r14 base **BIT-IDENTICAL**
  (o/ht 0.000e+00); the F1AL variant separately BIT-IDENTICAL.
* Determinism: `scratch/r15x/det_wst/` N=5 fresh-process
  **FWD-DETERMINISTIC** (mono w4/bv32 US+WST, bench_B1T2048H16).
* Knob arms: `GDN_WST`/`GDN_F1AL`/`GDN_F2US` unset regenerates the
  HEAD payload byte-identically (verified against the 4d5916d
  generator's output for the T8192 shape).

## 5. Ship provenance + spot adjudication

Shipped cubin md5s (kernel_smemfix.cubin): `B1T8192_mono` (w4/bv32 +
F2US + WST) `5b9fcf48344e742ad963db83b071875c`, `B4T2048_mono` (same
knobs) `6df1497e26555e83f793952dee53dbd5`, `B1T2048_mono` (untouched
w8/bv64) `bf56dde8ada28178b569558f09fcc7bb`.

| shape (spot, 12 pairs --stats) | ours | fla | ratio | per-pair | p | clean |
|---|---|---|---|---|---|---|
| B1T2048 | 0.2869 | 0.3527 | 1.2293x | 1.2159x | 4.9e-4 | yes |
| **B1T8192** | **0.8440** | 0.9544 | **1.1309x** | 1.1286x | 4.9e-4 | yes |
| B4T2048 | 0.8788 | 0.9531 | 1.0846x | 1.0861x | 4.9e-4 | yes |

(`spot_adjudicate_r15x.json`; the B4 verdict-of-record remains the
archived 1.140x formal + this round's same-session paired ship
decision; T8192's formal re-adjudication at 30 pairs is queued for the
next clean window if wanted -- the spot reads above it.)

## 6. Round-16 list (measured, ranked)

1. **F2 store passes, take 2**: warp-level pairing (each thread keeps
   the conflict-free single-cell shared loads; shuffle the odd lane's
   value in and store one packed b32 per pair) -- sidesteps the
   80B-stride bank geometry that killed tile-level pairing (+4.3us);
   ceiling is the measured 53us (h_all 35.4 + v_new 17.9).
2. **Fused chunk-level pipelining** (r14 #4, still open): producer
   warps / async stages to overlap the ~5.5us/chunk serial
   solve+staging -- the only route to a fused flagship that beats the
   trio (6.35ms @T8192).
3. **F1 A_m-free (bf16) solve probe**: the only 2-CTA route under the
   57.4KB phase-1 floor; a pure numerics experiment (forward
   substitution from Abf), gate-measurable in one build.
4. **B1T2048 US+WST refresh**: untested e2e as a build (low risk,
   small upside; the shape's record 1.333-1.464x is on the untouched
   w8/bv64 cubin).
5. bwd-side r15 items (sibling workstream).

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
GDN_F2US=1 GDN_WST=1 ~/miniforge3/envs/gdn/bin/python tools/gdn_full_gen.py     --path mono --shape 1,8192,16,128,128 --warps 4 --bv 32 --bv3 128     --out <dir>          # the shipped T8192 build (B4: --shape 4,2048,...)
bash scripts/run_l2.sh <dir>
GDN_WST=1 ~/miniforge3/envs/gdn/bin/python tools/gdn_smem_fix.py <dir>
GDN_F2US=1 GDN_WST=1 ~/miniforge3/envs/gdn/bin/python tools/gdn_full_verify.py     --warps 4 --bv 32 --gen-root <fresh>   # 4/4
# B4 ship decision: evidence/G5/scratch/r15x/r15x_b4ship.py --pairs 36
# paired A/B: evidence/G5/scratch/r14u/r14u_ab.py (as r14)
# F1AL (off by default): GDN_F1AL=1 in the gen env + smem_fix reads the
#   generated smem_alias.json automatically (main_multi "modules" map)
# knobs off -> HEAD payloads byte-identically (GDN_WST/GDN_F1AL/GDN_F2US)
```

# G5 round 16 (R16-Z): fused flagship 6.27 -> 4.53ms (FPIPE intra-chunk async staging); B1T2048 US+WST ship (spot 1.957x); f2 shuffle-pair store measured dead; F1 bf16-solve bounded GO + implemented (neutral, knob off)

Date: 2026-10-08 | clean idle window discipline (compute-apps checked
before every GPU run; coordination row `gdn-mlir R16-Z`).  HEAD through
the round: 289717f (all A/B baselines REBUILT at this commit; the T8192
mono rebuild reproduced the r15 ship cubin md5 `5b9fcf48` bit-for-bit,
proving the modified generator/smem_fix are knob-off byte-identical).

## Verdict

* **FPIPE (lever 2, the round's headline): the fused T8192 flagship
  drops 6274 -> 4530 us (-27.8%), three sessions of 40 interleaved
  pairs each, 0/40 slower in every session** (-1744 / -1750 / -1736 us
  paired medians).  Under the r14 "reclassify or pipeline" bar and
  under the round's <5ms target.  Mechanism: the three synchronous
  global->smem cell passes of the serial chunk chain (k, beta*v, q) +
  bk2's SECOND global read of the same k become MANUAL thread-strided
  `nvgpu.device_async_copy` groups (the F2US pattern) with waits at
  the first consumer: k(n) and RAW v(n) issue BEFORE the gc scan (the
  tiles are dead since sd(n-1)), beta lands via an in-place smem scale
  after the solve, q(n) issues right after wg's barrier (QK dead once
  wg consumed bk2) and is waited before qk; bk2 reads the resident
  k_t (k is read from global ONCE per chunk, not twice).  Outputs
  BIT-IDENTICAL to the stock fused kernel (o/ht 0.000e+00 full-run
  compare), 4/4 verify at the record digits, det N=5
  FWD-DETERMINISTIC, pool unchanged 99,072B, knob `GDN_FPIPE`
  (default off; knob-off regen byte-identical to HEAD).  Shipped as
  `B1T8192_fused` cubin `ae2c4ea4d39686c88ef49591b8c6da63` (the cache
  entry held the stale r2 w8/bv64 build `b2549100`).
  Cross-chunk ping-pong stays blocked by arithmetic: the device cap is
  MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 101,376B (102,400B/SM, 82 SMs --
  queried, `scratch/r16z` probe), leaving 2,304B over the 99,072B
  pool; a k/q pong tile is 17,408B and every aliasing permutation hits
  a live-range overlap (w_t's [8..10] window vs the pong's whole-chunk
  residency; documented in the gen comment).  The r14 reclassify story
  is therefore HALF-revised: the chain did not fight intra-chunk
  overlap at all -- it ate 28% -- but full chunk-ahead prefetch
  remains a restructure (a third k-sized tile must be bought with the
  bf16-solve diet + pad stripbacks that cost more than they pay).
* **B1T2048 US+WST (lever 4): SHIPPED, and the shape jumps to a 1.957x
  spot.**  The r15 ship decision never tested this shape e2e; the
  w4/bv32 + GDN_F2US + GDN_WST rebuild vs the untouched w8/bv64 cubin
  (`bf56dde8`, 3 sessions x 60 pairs): f1 -39.2/-39.3 us (0-0/60
  slower), f2 -51.1/-51.6 us (2-0/60), f3 -11.3/-11.2/-11.0 us
  (3-0-2/60), staged e2e **-77.5 / -76.3 / -78.1 us** (0.288 ->
  0.210 ms, -27%).  Numerics 4/4 at the exact record digits (staged o
  6.599e-4 / ht 7.493e-4 -- the r5/r14/r15 digits; WST is byte-exact
  and the tile-config change does not touch arithmetic), det N=5
  FWD-DETERMINISTIC.  New ship cubin
  `809754953615bce837ec4505b3462b7d`.  Post-ship spot adjudication
  (12 pairs --stats): **1.957x** (per-pair 1.945x, IQR 0.081 ->
  DECIDABLE -- the first decidably-clean shape read of the project;
  fla 0.4121 ms was slow-side this window, but even the conservative
  per-pair median doubles the r15 spot 1.229x).  All three knobs that
  r15 shipped for T8192/B4T2048 now apply to all three verdict shapes.
* **F2 warp-shuffle pair store (lever 1): measured decisively NEGATIVE
  and closed with the mechanism in SASS.**  Implementation
  (`shuffle_pair_stores`, env `GDN_WSP`, smem_fix-side, generator
  untouched): keep the conflict-free single-cell `ld.shared.b16`,
  `shfl.sync.bfly.b32 delta=1` the partner lane's bits in, `prmt`
  0x5410 packs own|partner, predicate on even tid, `st.global.b32` at
  the pair-aligned address; loop-invariant predicate hoisted to the
  entry.  Exactly the two f2 sites transform (h_all + v_new), f1/f3
  untouched (WST fuses theirs first), T8192 gate BIT-IDENTICAL
  (o/ht 0.000e+00 -- the packing is byte-exact).  A/B: f2 **+157.5 us
  (60/60 slower)**, confirm +151.2 us (30/30).  Why (SASS, `.L_x_8`):
  the stock loop unrolls x4 with four independent batched `STG.E.U16`
  (+0x400 strides -- full memory-level parallelism), while the WSP
  loop is a per-iteration serial chain `WARPSYNC.ALL; LDS.U16 ->
  SHFL.BFLY -> PRMT -> @!P0 STG.E` -- the cross-lane SHFL dependency
  forbids unrolling and adds a warp convergence point per iteration.
  The theorized saving is also void: a predicated half-warp b32 store
  issues the SAME warp-instruction count, moves the SAME 64B and the
  same 2 sectors as the full-warp b16 store -- zero LSU slots saved.
  With h_all already near the 64MB-write bandwidth floor (~35us at
  ~1.8TB/s), the r15 "warp-level pairing" idea is now measured dead at
  both attack surfaces (tile-level +4.3us r15, lane-level +151us r16).
  `GDN_WSP` stays as a documented off-knob.
* **F1 bf16-solve (lever 3): CPU probe GO, GPU implemented, perf
  NEUTRAL -- knob off; the 2-CTA arithmetic moves one floor down.**
  CPU probe first (numpy twin of the staged chain with the kernel's
  exact bf16 cut points, `scratch/r16z/probe_bf16solve.py`): the twin
  CALIBRATES to the record (variant A = today's f32-A phase 1 reads:
  T2048 o 7.318e-4 / ht 7.493e-4 -- the kernel's own record digit) and
  variant B (bf16-rounded solve A reads) is statistically
  indistinguishable: o 6.523e-4 / ht 7.493e-4, worst case 6.7x under
  the 5e-3 bar, 3.3x under the 2.5e-3 probe gate.  The reason T does
  not erode: T was already bf16 through phase 2 (Abf mma) and the Tt
  feedback -- the f32 bits A_m carried into phase 1 were rounded away
  two ops later.  GPU knob `GDN_BF16SOLVE` (gen_f1): A_m not
  allocated, phase-1 recurrences load Abf + extf, kkt epilogue stores
  only Abf; pool 101,120 -> 84,736B; T8192 gate PASS (o 6.626e-4 / ht
  8.918e-4 vs base 6.846e-4 / 7.902e-4) and A/B neutral (f1 -2.6 /
  -3.2 us, 11/60 and 7/30 pairs -- noise; no occupancy change, the
  pool is still far above the 50,688B 2-CTA cap).  The phase-1
  co-live floor DOES drop under the cap with the diet (57,408 ->
  41,024B < 50,688B) -- but the uw stage is the new blocker: it
  co-holds Tb+Tt+Tg (31,296B) + vt (17,408B) + k2 (17,408B) =
  66,112B > 50,688B.  Round-17 item: K-split staging of the uw loop's
  Tb/Tg A-operands.  Knob OFF (not stackable with GDN_F1AL).

## 1. FPIPE details (what exactly changed)

`gen_fused` under `GDN_FPIPE=1` (payload +127 lines, pool unchanged):
per chunk, BEFORE the gc scan: `async_stage_unrolled` k(n)->k_t (1024
16B copies) + v(n)->BVN RAW (256 copies), two `create_group` tokens;
`device_async_wait` gk + barrier before kkt; after the solve: wait gv
+ barrier + an in-place beta scale over BVN (the exact pre_bv
arithmetic, sourced from smem instead of global -- bit-identical);
bk2 becomes a smem->smem scaled pass reading k_t (was a SECOND 16KB
global read per chunk); after wg's barrier: q(n)->QK async (1024
copies), waited + barrier'd immediately before the qk gemm (overlaps
vp).  Hazards: k_t/BVN are dead since sd(n-1) consumed their previous
content inside chunk n-1; QK is dead once wg consumed bk2.  The
stock path's synchronous staging lowered to vector-transfer/scalar
LSU code that sat on the serial chain; cp.async removes it from the
issue path entirely -- which is why the win is latency-hiding AND
instruction-diet at once (-28%, not the ~-10% a pure latency model
predicts).

## 2. Adjudication table (post-ship spot, 12 interleaved pairs --stats)

| shape | ours | fla | ratio | per-pair | IQR rel | flags |
|---|---|---|---|---|---|---|
| B1T2048 | 0.2106 | 0.4121 | **1.957x** | 1.945x | 0.081 | clean DECIDABLE |
| B1T8192 | 0.8476 | 1.1856 | 1.399x | 1.310x | 0.381 | clean, co-load-sensitive (fla slow window) |
| B4T2048 | 0.8824 | 0.9700 | 1.099x | 1.098x | 0.071 | co-load-sensitive |

(`spot_adjudicate_r16z.json`; T8192/B4 cubins unchanged from r15 --
their read is a no-regression check; B1T2048 is the new ship.  fla's
T8192 median swung 0.95 -> 1.19 ms vs the r15 window; the per-pair
1.310x is the robust number and still above the r15 spot 1.131x.)

## 3. Numerics + determinism (green after every step)

* Ships: B1T2048 US+WST `fwd_verify_r16z_2048uswst.json` **4/4 PASS**
  (staged o 6.599e-4 / ht 7.493e-4, fused 5.751e-4 / 7.929e-4 -- the
  record digits); FPIPE `fwd_verify_r16z_fpipe.json` **4/4 PASS**
  (fused digits unchanged -- the staging refactor is value-exact);
  det N=5 fresh-process FWD-DETERMINISTIC for both ships
  (`det_2048uswst/`, `det_fpipe/`).
* Gates: WSP bit-identical (T8192 o/ht 0.000e+00); BF16SOLVE PASS
  (o 6.626e-4 / ht 8.918e-4, the measured CPU-bound erosion 6.6e-4-
  class exactly); FPIPE full-run o/ht 0.000e+00 vs stock fused.
* Knob-off arms byte-identical: mono payload == HEAD payload, fused
  payload == HEAD payload (`/tmp/knoboff_*` diffs empty); the T8192
  mono baseline rebuild reproduced ship md5 `5b9fcf48`.

## 4. Ship provenance

| build | knobs | cubin md5 (kernel_smemfix.cubin) |
|---|---|---|
| B1T2048_mono (NEW) | w4/bv32 + F2US + WST | `809754953615bce837ec4505b3462b7d` |
| B1T8192_mono (r15, unchanged) | w4/bv32 + F2US + WST | `5b9fcf48344e742ad963db83b071875c` |
| B4T2048_mono (r15, unchanged) | w4/bv32 + F2US + WST | `6df1497e26555e83f793952dee53dbd5` |
| B1T8192_fused (NEW) | w4/bv32 + FPIPE | `ae2c4ea4d39686c88ef49591b8c6da63` |

(retired: B1T2048 w8/bv64 `bf56dde8...` -- beaten by -77us same-window
paired; stale fused w8/bv64 `b2549100...`.)

## 5. Round-17 list (measured, ranked)

1. **FPIPE propagation**: port the same intra-chunk async staging to
   the T2048/B4 fused builds and the mono trio's remaining synchronous
   passes (f3's q/h_all staging, f1's k2/vt post-solve copies already
   async); the -28% says the pattern is worth sweeping everywhere.
2. **F1 2-CTA endgame**: bf16-solve (landed, knob) + uw-stage
   restructure -- K-split Tb/Tg staging (mma A-operands re-staged per
   32-col slice over the freed A_m bytes) to get the uw co-live under
   the 50,688B cap; the phase-1 floor already clears (41,024B).
3. **f2 cell passes**: closed at both attack surfaces; any future work
   needs the >=64-col row geometry (tile layout change), not store
   instruction selection.
4. **bwd-side FPIPE** (sibling workstream): the fused-fwd -28% suggests
   the bwd chain's synchronous stagings have the same headroom.
5. Re-run the formal 30-pair T8192 adjudication in a clean window (the
   spot 1.399x rides a slow fla window; per-pair 1.310x is the robust
   floor and the r15 formal 1.101x needs the refresh).

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
# ships:
GDN_F2US=1 GDN_WST=1 ~/miniforge3/envs/gdn/bin/python tools/gdn_full_gen.py \
    --path mono --shape 1,2048,16,128,128 --warps 4 --bv 32 --bv3 128 --out <d2048>
bash scripts/run_l2.sh <d2048>; GDN_WST=1 ~/miniforge3/envs/gdn/bin/python tools/gdn_smem_fix.py <d2048>
GDN_FPIPE=1 ~/miniforge3/envs/gdn/bin/python tools/gdn_full_gen.py \
    --path fused --shape 1,8192,16,128,128 --warps 4 --bv 32 --out <dfused>
bash scripts/run_l2.sh <dfused>; ~/miniforge3/envs/gdn/bin/python tools/gdn_smem_fix.py <dfused>
# fused A/B: evidence/G5/scratch/r16z/fused_ab.py --a <stock> --b <fpipe> --reps 40
# trio A/B:  evidence/G5/scratch/r14u/r14u_ab.py (as r14/r15)
# gates:     evidence/G5/scratch/r14u/r14u_t8192_gate.py --dirs base=<b>,var=<v>
# verify:    GDN_FPIPE=1 (or the ship knobs) tools/gdn_full_verify.py --warps 4 --bv 32 --gen-root <fresh>
# det:       GDN_FPIPE=1 tools/gdn_fwd_det.py --path fused --warps 4 --bv 32 (x5 + --diff)
# knobs: GDN_WSP (f2 shuffle-pair store, NEGATIVE +151us), GDN_BF16SOLVE
#   (numerics-GO, perf-neutral), GDN_FPIPE (SHIPPED for fused T8192);
#   all default off; unset == HEAD payloads byte-identically
```

# G5 round 17 (R17-fwd): FPIPE propagates -26% to every fused build; formal T8192 record 1.170x (30 pairs, decidable); the F1 2-CTA endgame built, proven bit-exact at 1 CTA, and honestly blocked by an unresolved 2-CTA-only corruption

Date: 2026-10-08 | clean idle window for the formal run (compute-apps
checked; window snapshots in the JSONs; coordination row `gdn-mlir
R17-fwd`).  HEAD through the round: 26c2693 (all four r16 ship cubin
md5s re-verified at round start; knob-off generator payloads
byte-identical after every edit, re-checked twice).

## Verdict

* **Formal T8192 re-adjudication (R16-Z #5): the record moves 1.101x ->
  1.170x, 30 interleaved pairs, ALL 30 pairs > 1 (min pair 1.1188x),
  sign-test p = 0.0, DECIDABLE by the methodology rule** (IQR rel
  0.040 < the 3x bar), window clean (spot `final_adjudicate_r17f_
  t8192_n30.json`).  ours 0.8272ms vs fla 0.9681ms; the robust
  per-pair median is 1.152x.  The FPIPE-era cubin is now formally the
  record on the flagship shape; this supersedes the r14 1.101x formal
  and the r16 1.399x/1.310x spot (which rode a slow fla window).
* **FPIPE propagation (R16-Z #1a): SHIPPED for the fused path at
  B1T2048 and B4T2048 -- the -28% T8192 result replicates at -26.5% on
  both shapes, three sessions of 40 interleaved pairs each, 0/40
  slower in every session, outputs bit-identical (o/ht 0.000e+00).**
  B1T2048 fused: 1555.5/1554.7/1555.9 -> 1143.3/1143.4/1143.3 us
  (paired medians -412.4/-411.8/-412.2 us).  B4T2048 fused:
  6216.2 -> 4581.9/4580.9/4583.5 us (paired -1635.6/-1629.0/-1628.4
  us).  4/4 verify at the exact record digits under GDN_FPIPE, det
  N=5 FWD-DETERMINISTIC, pool unchanged 99,072B.  New ship cubins:
  B1T2048_fused `93856752dc53980421a60bef9aad851b`, B4T2048_fused
  `b5d549349b8ce6f5477d33985e4ebabc` (the gen_fwd cache entries held
  stale r2 w8/bv64 builds).  The fused path remains a DESIGN RECORD,
not a verdict path -- the mono trio still carries all three shapes
  (fused 1.14ms vs trio 0.21ms at T2048).
* **Mono trio sync-staging audit (R16-Z #1b): CLOSED NEGATIVE -- there
  is nothing left to FPIPE.**  At PTX level every global->smem staging
  in the trio is already cp.async: f1 30 ops (kkt k staging through
  the transform pipeline + the r5 manual vt/k2 groups), f2 42 (the r3
  group-A w/k/gc prefetch + r14 F2US u tiles), f3 42 (q/k/h_all/v_new
  through the pipelined K-loops).  The only synchronous global loads
  left are f1's scan inputs (g/beta, 512B, consumed immediately --
  no overlap window exists before them) and f3's gc (256B, overlaps
  the qo pipeline's first-iteration prefetch naturally).  The FPIPE
  -26..-28% class was fused-path-specific: its serial chunk chain
  carried three synchronous cell passes + a second k global read that
  the trio's kernel boundaries never had.  The bwd side (sibling
  workstream) is where the pattern still has untouched surface.
* **F1 2-CTA endgame (R16-Z #2): built and closed HONEST-BLOCKED.**
  The CPU twin probe ran FIRST (order-preservation question): the uw
  gemm's per-element fp32 accumulation sequence is UNCHANGED by the
  32-split (the boundary falls BETWEEN m16n8k16 blocks; the carried
  accumulator preserves the exact add order) -- twin A vs B
  bit-identical (0 flips) while a tree-reassociation control C DOES
  flip bits (3+1), proving the probe's sensitivity; GO at infinite
  margin (`scratch/r17f/ksplit_probe.json`).  The GPU implementation
  (GDN_F1KS, stacks on GDN_BF16SOLVE): vt/k2 halved to [32,136]/
  [33,136] half-K tiles reused across two sequential passes,
  post-solve landing into Abf/Tt's bytes via the PTX lifetime arena
  (identify-by-size made cheap by ROW-granular inflation: +1/+2 dead
  rows at 144B each instead of F1AL's 1-2KB column pads; St2 drops
  its +8 pad), kkt at bk=16 (kt multibuffer 10,240 -> 6,144B).
  **f1 pool 101,120 -> 46,352B, cuOccupancyMaxActiveBlocksPerMulti-
  Processor = 2 blocks/SM (the first fwd kernel ever at 2 CTAs).**
  Numerics: the K-split code is BIT-IDENTICAL to bf16-solve at 1
  CTA/SM (u/w 0/4,194,304 bit-diffs, two independent builds).  BUT at
  2-CTA co-residency u/w corrupt (o 6.3e-2 vs the 5e-3 bar; ht stays
  at the exact record digit 7.493e-4) through a mechanism eight bisect
  arms could not pin (table below).  Knob OFF (default off; knob-off
  payloads byte-identical).  The machinery + the bisect table are the
  round-18 starting point.

## 1. The 2-CTA corruption bisect table (eight arms, all T2048)

| arm | pool | CTAs/SM | staging | result |
|---|---|---|---|---|
| mono bf16-solve (r16 knob) | 84,736 | 1 | cp.async pre-solve | PASS o 6.486e-4 |
| mono F1KS, no arena (staged f1 + mono) | 64,144 | 1 | cp.async 2-pass | PASS, bit-identical to bf16sv |
| mono F1KS + arena | 46,352 | 2 | cp.async 2-pass | FAIL o 6.35e-2 |
| ... forced 1 CTA (Tg +56 dead rows) | 54,416 | 1 | cp.async 2-pass | PASS, bit-identical |
| ... +20 rows instead | 49,232 | 2 | cp.async 2-pass | FAIL o 5.3e+02 |
| ... synchronous scalar staging | 46,352 | 2 | NO cp.async | FAIL o 6.49e-2 |
| ... kkt mb=1 (kt 3,072B) | 43,280 | 2 | cp.async 2-pass | FAIL (NaN class) |
| ... kkt bk=32 + St pad shaves | 49,936 | 2 | cp.async 2-pass | FAIL o 4.9e+01 |
| ... + GDN_SCANDB double-buffered scan | 46,608 | 2 | cp.async 2-pass | FAIL o 6.5-6.8e-2 (rep-varying) |

Established by the table: the K-split mma/staging/epilogue code is
value-exact (1-CTA arms); the corruption is 2-CTA co-residency
itself, independent of the arena, the staging mechanism, bk, mb, and
pool size within the 2-CTA band (the effective 2-CTA line on this
part measured between 49,936 and 50,432B reported smem).  NOT
established: the mechanism.  racecheck flags the same 6 benign-looking
scan-step LDS/STS pairs in the CLEAN builds; memcheck is clean (the 3
reported errors are host-side lookup failures); both sanitizers
serialize scheduling and hide the bug.  The scan-step cross-warp
intra-step race hypothesis (GDN_SCANDB, landed) was falsified as the
sole cause.  The standing puzzle: f2 produces a PERFECT ht (7.493e-4,
the record digit) while the same run's u/w read back corrupted -- if
u/w were truly corrupt when f2 read them, ht could not be perfect.
This o/u-vs-ht inconsistency is the round-18 thread to pull: it
points at f1's u/w global writes racing with something AFTER f2's
consumption, or at the readback path -- not at the compute.

## 2. Adjudication table

| shape (run) | ours | fla | ratio | per-pair | p | flags |
|---|---|---|---|---|---|---|
| **B1T8192 (formal, 30 pairs)** | 0.8272 | 0.9681 | **1.170x** | 1.152x | 0.0 | clean, DECIDABLE (IQR rel 0.040) |
| B1T2048 (spot, 12 pairs) | 0.2073 | 0.3483 | 1.680x | 1.671x | 6.3e-3 | co-load-sensitive |
| B4T2048 (spot, 12 pairs) | 0.8815 | 0.9825 | 1.115x | 1.117x | 3.9e-2 | co-load-sensitive |

(`final_adjudicate_r17f_t8192_n30.json`, `spot_adjudicate_r17f.json`;
mono ship cubins UNCHANGED this round -- the spots re-read the r16
ships; B1T2048's r16 spot 1.957x rode a slow fla window, this window's
robust per-pair read is 1.671x; B4's archived 1.140x formal stands.)

## 3. Numerics + determinism (green after every step)

* FPIPE fused T2048/B4: `fwd_verify_r17f_fpipe.json` **4/4 PASS**
  (staged o 6.599e-4 / ht 7.493e-4, fused 5.751e-4 / 7.929e-4 -- the
  record digits); outputs bit-identical to stock fused in all six A/B
  sessions; det N=5 fresh-process FWD-DETERMINISTIC
  (`det_fpipe_2048/`); B4 numerics via bit-identity (no fp64 ref npz
  for B4 -- the r15/r16 pattern).
* F1KS: CPU twin GO (above); GPU K-split code bit-identical at 1 CTA
  (two builds x u/w arrays); the 2-CTA arms are the corruption study,
  not ships; knob-off payload byte-identical to HEAD (checked after
  every generator edit round, twice).
* All four r16 ship cubins re-verified at round start and unchanged at
  round end (md5s in section 4).

## 4. Ship provenance

| build | knobs | cubin md5 (kernel_smemfix.cubin) |
|---|---|---|
| B1T2048_mono (r16, unchanged) | w4/bv32 + F2US + WST | `809754953615bce837ec4505b3462b7d` |
| B1T8192_mono (r15, unchanged) | w4/bv32 + F2US + WST | `5b9fcf48344e742ad963db83b071875c` |
| B4T2048_mono (r15, unchanged) | w4/bv32 + F2US + WST | `6df1497e26555e83f793952dee53dbd5` |
| B1T8192_fused (r16, unchanged) | w4/bv32 + FPIPE | `ae2c4ea4d39686c88ef49591b8c6da63` |
| B1T2048_fused (NEW) | w4/bv32 + FPIPE | `93856752dc53980421a60bef9aad851b` |
| B4T2048_fused (NEW) | w4/bv32 + FPIPE | `b5d549349b8ce6f5477d33985e4ebabc` |

(new knobs landed, all default off, unset == HEAD byte-identically:
GDN_F1KS (+ GDN_F1KS_F1CTA/_BK32/_SYNC diagnostic arms),
GDN_SCANDB -- a real race fix for the Hillis-Steele scan steps,
recommended hardening for the r18 ships even though 1-CTA residency
has never exposed it.)

## 5. Round-18 list (measured, ranked)

1. **The f1 2-CTA corruption root-cause** -- the lever with the
   measured 25-wave halve on the table (f1 0.47ms is the largest fwd
   kernel @T8192).  Start from the o/u-vs-ht inconsistency: instrument
   WHEN u/w go bad (dump u/w from f2's perspective via a checksum
   kernel between f1 and f2); skeleton-repro (scan+solve+uw only);
   SASS-diff the 1-CTA-forced vs 2-CTA cubins (should be identical --
   if so, the trigger is pure scheduling).  The bisect table in
   section 1 rules out: code values, arena, staging class, bk, mb,
   the scan race alone.
2. **GDN_SCANDB as ship hardening**: the scan's cross-warp intra-step
   race is real (racecheck's 6 pairs in every build incl. the ships);
   1-CTA residency masks it.  A/B the ships + SCANDB (expect
   bit-identical + ~0 perf) and fold into the r18 ship refresh.
3. **bwd-side FPIPE** (sibling workstream, unchanged from r16's list).
4. **f2 >=64-col row geometry** (r16 #3, still last).
5. If 1 lands: re-run the formal T8192 adjudication with the 2-CTA f1
   (the 1.170x record would move again; f1 is ~55% of the chain).

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
# ships (mono trio unchanged; fused FPIPE now on all three shapes):
GDN_F2US=1 GDN_WST=1 ~/miniforge3/envs/gdn/bin/python tools/gdn_full_gen.py \
    --path mono --shape 1,8192,16,128,128 --warps 4 --bv 32 --bv3 128 --out <dmono>
GDN_FPIPE=1 ~/miniforge3/envs/gdn/bin/python tools/gdn_full_gen.py \
    --path fused --shape 1,2048,16,128,128 --warps 4 --bv 32 --out <dfused>  # B4: 4,2048,...
bash scripts/run_l2.sh <dir>; GDN_WST=1 <python> tools/gdn_smem_fix.py <dir>  # mono only needs WST
# formal T8192 n=30 (clean window):
~/miniforge3/envs/gdn/bin/python tools/gdn_final_adjudicate.py --shapes B1T8192 \
    --pairs 30 --stats --warps 4 --bv 32 --bv3 128 --out <json>
# fused FPIPE A/B (any shape):
~/miniforge3/envs/gdn/bin/python evidence/G5/scratch/r17f/fused_ab2.py \
    --a <stock> --b <fpipe> --shape 1,2048,16,128,128 --reps 40
# F1KS 2-CTA study (knob OFF by default; the corruption repro):
GDN_F1KS=1 ~/miniforge3/envs/gdn/bin/python tools/gdn_full_gen.py --path mono \
    --shape 1,2048,16,128,128 --warps 4 --bv 32 --bv3 128 --out <dks>   # smem_alias.json auto
bash scripts/run_l2.sh <dks>; GDN_WST=1 <python> tools/gdn_smem_fix.py <dks>
~/miniforge3/envs/gdn/bin/python evidence/G5/scratch/r17f/occ_probe.py --dir <dks>
# CPU twin K-split probe: evidence/G5/scratch/r17f/probe_ksplit.py [small_B1T512H4]
# bisect arms: bisect*_f1ks.sh in scratch/r17f (correct baselines matter:
#   F1KS implies bf16-solve -- compare against a GDN_BF16SOLVE=1 build)
```

# G5 round 18 (R18-fwd): the F1 2-CTA corruption ROOT-CAUSED (two stacked masked-at-1-CTA bugs: an uninit-smem stride bug + the scan race), ZFIX+SCANDB shipped as hardening, and the F1KS 2-CTA build SHIPPED on all three mono shapes (f1 -19%/-21%/-21% paired; T8192 spot 1.239x)

Date: 2026-10-08 | shared/opportunistic window (co-job pid 3861537
~8.9GB resident, untouched; window snapshots in the JSONs; coordination
row ).  HEAD through the round: 30086e4; knob-off
generator payloads byte-identical after both edits (mono+fused diff 0
lines vs the HEAD generator, /tmp/r18f_id).

## Verdict

* **The F1 2-CTA corruption is ROOT-CAUSED: two independent bugs, both
  masked at 1-CTA residency, both live in every build since the blocked
  solve landed.**  Bug 1 (found this round): the solves

# G5 round 18 (R18-fwd): the F1 2-CTA corruption ROOT-CAUSED (two stacked masked-at-1-CTA bugs: an uninit-smem stride bug + the scan race), ZFIX+SCANDB shipped as hardening, and the F1KS 2-CTA build SHIPPED on all three mono shapes (f1 -19%/-21%/-21% paired; T8192 spot 1.239x)

Date: 2026-10-08 | shared/opportunistic window (co-job pid 3861537
~8.9GB resident, untouched; window snapshots in the JSONs; coordination
row `gdn-mlir R18-fwd`).  HEAD through the round: 30086e4; knob-off
generator payloads byte-identical after both edits (mono+fused diff 0
lines vs the HEAD generator, /tmp/r18f_id).

## Verdict

* **The F1 2-CTA corruption is ROOT-CAUSED: two independent bugs, both
  masked at 1-CTA residency, both live in every build since the blocked
  solve landed.**  Bug 1 (found this round): the solve's phase-0
  `zero_up` cell loop strides `32*8 = 256` over a 128-thread block,
  so Tb/Tg cells with `(r*64+c) mod 256 in [128,256)` -- rows
  `r mod 4 in {2,3}` of the six lower-left 16x16 blocks, 768
  cells/tile -- are NEVER zeroed; the uw gemms' transposed ldmatrix
  A-reads cover the full 64x64, so they read those cells as T's
  upper-triangle zeros.  At 1 CTA they read EXACT ZERO (measured:
  393,216/393,216 cells across all 512 tiles); at 2-CTA co-residency
  they read nonzero live-kernel data (~56% of cells, +-1e-14-class
  far-off-diagonal decay values with rare O(1) outliers, absmax 1.0)
  -- order-independent (1-CTA stays zero-clean even immediately after
  a garbage-heavy 2-CTA run), f1-only reproducible (no f2/f3 in the
  process at all), and the SASS-diff between the PASS forced-1-CTA
  build and the FAIL 2-CTA build is exactly ONE instruction (`MOV R4,
  0x2490` vs `0x4410` -- Tg's malloc size 9,360 vs 17,424): the
  trigger is pure co-residency staleness, the code is identical.
  Fix: `GDN_ZFIX` (stride `nthr`, full coverage).  Bug 2 (found r17,
  now PROVEN the residual): the Hillis-Steele scan's cross-warp
  intra-step race -- under ZFIX-only the corruption collapses from
  ~700K to ~4K bit-diffs, fires in 1-2 of 512 tiles per run, exactly
  the `gc[32:64)` window (the d=32 step's cross-warp read), and
  propagates chunk-locally (u/w rows 32-63 of that chunk).  Fix:
  `GDN_SCANDB` (r17).  **ZFIX+SCANDB at 2 CTA/SM: 10/10 f1-only reps
  bit-identical to the forced-1-CTA reference and the full trio hits
  the exact record digits (o 6.486e-4 / ht 7.493e-4).**
* **The r17 o/u-vs-ht puzzle is solved by the fingerprint: the
  uninit-cell corruption is per-CTA binary (338/512 tiles in the study
  run) and row-structured (w exactly at Tb/Tg block-column Cb=2 = rows
  32-47, u at rows 0-47, rows 48-63 clean because mma output block
  mi=3 receives garbage only from k-blocks > 3, which do not exist).
  f2's ht stays perfect because the corrupted chunks' state
  contributions decay below the contract by the final chunk, while f3
  reads the corrupted chunks' h_all/v_new DIRECTLY -- o fails, ht
  passes.  No output-path mystery was ever there.**
* **F1KS 2-CTA SHIPPED on all three mono shapes** (knobs w4/bv32 +
  F2US + WST + ZFIX + SCANDB + F1KS; f1 pool 46,608B reported,
  cuOccupancyMaxActiveBlocksPerMultiProcessor = 2): paired medians
  vs the r15/r16 ships, interleaved x3 sessions of 40 pairs each --
  T8192: f1 278.7 -> 225.2us (**-53.5us, 35-38/40 faster**), f2 -0.4,
  chain **-50.9us median**; T2048: f1 78.6 -> 61.9us (**-16.7us**),
  chain -13.0us; B4T2048: f1 281.1 -> 223.9us (**-57.9us, 36-40/40
  faster**), chain **-75.4us median (-8%)**.  The r17 estimate (f1
  halves its wave exposure) measured out at -19% kernel time.
* **Ship hardening ON: ZFIX+SCANDB for the mono trio, SCANDB for the
  fused trio.**  At 1 CTA the hardening is VALUE-EXACT: the
  ZFIX+SCANDB T8192 build is BIT-IDENTICAL to the r15 ship at the
  fp64 gate (o/ht 0.000e+00), and FPIPE+SCANDB fused is bit-identical
  to FPIPE-only fused (whose rebuild reproduced the r17 ship md5
  93856752... exactly).  Cost: f1 +8.3us @T8192 (the zero loop
  doubles to 32 iterations) -- covered 6x by the F1KS win.
* **T8192 spot adjudication with the new ship: ours 0.7890ms vs fla
  0.9779ms = 1.239x (per-pair median 1.213x, 12 pairs, p=4.9e-4,
  co-loaded window)** -- ours moved 0.8272 -> 0.7890 vs the r17
  formal record window; the formal n=30 refresh is queued for a clean
  window (round-19 #1).
* **Harness caveat exposed: the r14u_ab A/B harness has a systematic
  B-SIDE bias** (~+13us on f3, ~+6us on f2 at T8192) proven by a
  swap-sides session (the r15 ship's f3 measured 0.2786ms on side B
  and 0.2663ms on side A in the same window; PTX of f2/f3 identical
  between builds).  f1 shows no side bias.  All f2/f3 deltas this
  round are reported with this correction; the true T8192 chain win
  is -60 to -71us (raw -47.6/-58.3/-50.9 with f3's bias on B).  Fix
  queued (round-19 #3).

## 1. The root-cause chain (how it fell, in order)

1. **f1-only repro** (probe1): the corruption needs NO f2/f3 --
   launching f1 alone at 2 CTA reproduces it; nondeterministic
   rep-to-rep; gc (the scan output) essentially clean (16 bits in one
   rep -- bug 2's rare firing, initially mistaken for noise).
2. **16x16 block fingerprint** (probe2/3): column-uniform, row-
   structured; per-(head,chunk) map = per-tile BINARY corruption with
   a wave structure (338/512 tiles, later-scheduled CTAs corrupt) --
   a stale-state signature, not a value bug.
3. **SASS-diff** forced-1-CTA vs 2-CTA builds: ONE instruction (the
   Tg malloc size) -- pure scheduling trigger, per the r18 plan.
4. **Tile dumps** (probe4/5, G5D_DUMP_TT extended with Tg + the
   post-epilogue half tiles): Tb/Tg diffs land EXACTLY on the cells
   zero_up misses (768/tile; counts 43,264 = 338 tiles x 128 cells);
   1-CTA reads zeros in ALL of them, 2-CTA reads garbage (order-
   independent, probe6).  (The tt dump's "corruption" is an alias
   artifact: k2 restages into Tt's bytes, so the kernel-end dump
   reads w-data -- recognized and discounted.)
5. **GDN_ZFIX** (stride nthr): 700K -> 4K bit-diffs, one rep fully
   clean; the residual fingerprint (gc[32:64), 1-2 tiles) = the r17
   scan race.  **GDN_ZFIX+GDN_SCANDB: 10/10 clean, record digits.**
   The precise provider path of the stale bytes under co-residency
   (cross-slot bleed vs allocator granularity vs prior-launch
   leftovers) is left unattributed -- the fix removes the dependence
   on all of them.  The r17 bisect table reinterpreted: every 2-CTA
   arm had BOTH bugs live; the mb=1 NaN arm = uninit cells reading
   bigger garbage; the "rep-varying 6.5-6.8e-2" = bug 2 modulating on
   top of bug 1's deterministic footprint.

## 2. Adjudication + perf table

| run | ours | fla / base | ratio / delta | pairs | flags |
|---|---|---|---|---|---|
| B1T8192 spot (12 pairs) | 0.7890ms | 0.9779 | **1.239x** (per-pair 1.213x) | 12 | co-load-sensitive, p=4.9e-4 |
| T8192 f1 A/B s1/s2/s3 | 278.4-279.0us | 224.0-225.6 | **-53.4/-53.5/-54.4us** | 40x3 | 35-38/40 faster |
| T8192 chain A/B | 0.8939-0.9676 | 0.8463-0.9093 | **-47.6/-58.3/-50.9us** | 40x3 | f3 side-bias on B; true ~-60..-71 |
| T2048 f1 / chain | 78.6 / 207.2us | 61.9 / 194.2 | **-16.7 / -13.0us** | 40 | clean |
| B4T2048 f1 / chain s1/s2/s3 | 280.9-289.2 | 223.0-230.9 | **-57.2/-57.9/-58.3us; chain -72.4/-75.4/-83.3us** | 40x3 | 36-40/40 faster; sweep |
| hardening-only T8192 (ZFIX+SCANDB @1 CTA) | f1 +8.3us | f2/f3 PTX-identical | chain +29.4us raw (~+16 corrected) | 40 | superseded by F1KS ship |

(jsons: spot_adjudicate_r18f.json, ab_f1ksuw_T8192_s1..3.json,
ab_swap_T8192.json, ab_f1ksuw_T2048_s1.json, ab_f1ksuw_B4_s1..3.json,
ab_ship_T8192_s1.json, ab_ship_T2048_s1.json; the f1ksu sessions with
the WST/F2US-incomplete build are superseded by f1ksuw.)

## 3. Numerics + determinism (green after every step)

* fwd_verify 4/4 PASS x3 knob sets: ship knobs (staged o 6.599e-4 /
  ht 7.493e-4 -- record digits), FPIPE+SCANDB (fused record digits
  5.751e-4 / 7.929e-4), F1KS full set (staged o 6.486e-4 / ht 7.493e-4
  -- the bf16-solve class) -- fwd_verify_r18f_{ship,fpipe_sdb,f1ks}.json.
* T8192 fp64 gate: ZFIX+SCANDB ship BIT-IDENTICAL to the r15 ship
  (o/ht 0.000e+00); F1KS PASS o 6.626e-4 / ht 8.918e-4 vs base
  6.846e-4 / 7.902e-4 (the r16 bf16-solve measured class).
* B4T2048: F1KS vs the r15 B4 ship o max 7.324e-4 / ht 4.883e-4
  (identical to the T8192 gate deltas), rep-to-rep 0 diffs x3.
* det N=5 fresh-process FWD-DETERMINISTIC x3 suites: ship knobs
  (det_ship/), FPIPE+SCANDB (det_fpipe_sdb/), F1KS 2-CTA (det_f1ks/).
* 2-CTA corruption study artifacts: probe1..8 + dump_1cta/dump_2cta +
  zfix_2cta/zs_2cta in scratch/r18f/ (the localization evidence).

## 4. Ship provenance

| build | knobs | cubin md5 (kernel_smemfix.cubin) |
|---|---|---|
| B1T2048_mono (NEW) | w4/bv32 + F2US + WST + ZFIX + SCANDB + F1KS | `ea8ec2f1dc1318d139863c95228179af` |
| B1T8192_mono (NEW) | same | `e7af55c1a89bdd3b1108c905a1fbe893` |
| B4T2048_mono (NEW) | same | `a065a0bcf1b0c4c710bed6c21f63190c` |
| B1T2048_fused (NEW) | w4/bv32 + FPIPE + SCANDB | `d16522d094c9d56f59001389f12ad827` |
| B1T8192_fused (NEW) | same | `cb2520d8f942edd3e086cfe396c1ab4e` |
| B4T2048_fused (NEW) | same | `676487bfc2bf4af190f54ca4612c1ee9` |

(retired: the r15/r16/r17 mono trio 5b9fcf48 / 80975495 / 6df1497e and
the r17 fused trio ae2c4ea4 / 93856752 / b5d54934 -- the FPIPE-only
rebuild reproduced 93856752 exactly before the SCANDB refresh, and
FPIPE+SCANDB is bit-identical in values.  New knobs: GDN_ZFIX -- the
zero_up stride fix; default OFF (unset == HEAD payload byte-identical,
re-verified); SHIPPED ON via the ship knob set.  GDN_F1KS / GDN_SCANDB
now SHIPPED ON for the mono trio; SCANDB on for fused.)

## 5. Round-19 list (measured, ranked)

1. **Formal T8192 n=30 refresh in a clean window** with the F1KS ship
   (spot 1.239x / per-pair 1.213x; the 1.170x formal record moves).
2. **f1's remaining serial chain**: the 25-wave halving realized only
   -19% kernel time -- the per-CTA serial solve (phase-1 recurrence +
   phase-2 R=1..3) is now f1's critical path; levers: warp-parallel
   phase-1 (2-CTA makes CTA-level parallelism abundant), uw-stage
   double-buffer across the two half-K passes.
3. **Fix the r14u_ab B-side bias** (rotate sides per rep); re-read the
   f2/f3 paired deltas of r15-r17 that sat inside the +-13us band.
4. **f3 2-CTA conversion** (the F1KS playbook: K-split + arena; f3 is
   82,176B at 1 CTA, the chain's second-largest kernel @B4).
5. **f2 >=64-col row geometry** (r16 #3, still last, untouched).

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
# ships (mono trio: F1KS 2-CTA + hardening):
GDN_F1KS=1 GDN_F2US=1 GDN_WST=1 GDN_ZFIX=1 GDN_SCANDB=1 \
    ~/miniforge3/envs/gdn/bin/python tools/gdn_full_gen.py --path mono \
    --shape 1,8192,16,128,128 --warps 4 --bv 32 --bv3 128 --out <dmono>
bash scripts/run_l2.sh <dmono>; GDN_WST=1 <python> tools/gdn_smem_fix.py <dmono>
# fused trio (FPIPE + SCANDB):
GDN_FPIPE=1 GDN_SCANDB=1 <python> tools/gdn_full_gen.py --path fused \
    --shape 1,2048,16,128,128 --warps 4 --bv 32 --out <dfused>
bash scripts/run_l2.sh <dfused>; <python> tools/gdn_smem_fix.py <dfused>
# verify (same env as the build): tools/gdn_full_verify.py --warps 4 --bv 32
# det (same env): evidence/G5/scratch/r18f/run_det.py <tag> <path> <KNOBS> <out> 5
# gates: evidence/G5/scratch/r14u/r14u_t8192_gate.py --dirs base=<b>,var=<v>
# trio A/B: evidence/G5/scratch/r14u/r14u_ab.py --shape B1T8192 --a <A> --b <B> --reps 40
#   (NOTE the B-side f3 bias; run a swap session when f2/f3 deltas matter)
# corruption study (r18 evidence): scratch/r18f/probe1..8.py, dump_1cta/
#   dump_2cta (G5D_DUMP_TT=1 builds), zfix_2cta, zs_2cta
# knobs: GDN_ZFIX (zero_up stride; SHIPPED ON in ship set), GDN_SCANDB
#   (SHIPPED ON), GDN_F1KS (SHIPPED ON for mono), GDN_F1KS_F1CTA/_BK32/
#   _SYNC (diagnostic arms), GDN_FPIPE (fused), GDN_F2US/GDN_WST;
#   all default off; unset == HEAD payloads byte-identically
```

# G5 round 19 (R19-fwd): the formal record refreshed (T8192 1.231x, 30/30); the r14u_ab B-side bias ROOT-CAUSED as a module-placement lottery (not launch order) + the corrections; F1WY warp-parallel phase-1 landed correct-but-slower (honest negative, knob off); f3 2-CTA audited + designed

Date: 2026-10-08/10 | shared/opportunistic window (co-job pid 3888342
~8.9GB exited mid-round, replaced by a lighter ~26%-util child pid
3889795; never touched; window snapshots in the JSONs; coordination row
`gdn-mlir R19-fwd`).  HEAD through the round: 0aa4b37 -> 6873f2e
(R19-b sibling landed mid-round); the ship-knob rebuild reproduced the
r18 mono-T8192 ship cubin md5 `e7af55c1...` bit-for-bit with the new
default-off GDN_F1WY code in the generator (the knob-off proof).

## Verdict

* **The formal T8192 record moves: 1.170x (r17, 30 pairs) -> 1.2314x
  ratio-of-medians / 1.208x per-pair median, 30/30 interleaved pairs
  faster, sign-test p = 0.0** with the r18 F1KS ship cubin
  (`e7af55c1`), n=30, --stats (final_adjudicate_r19f_T8192_s2.json;
  ours 0.8116ms vs fla 0.9994ms, window 15-33% util, sm_clock 2782).
  A first n=30 session in the dirtier window (the old co-job exiting
  mid-round) read 1.4254x/1.3928x per-pair, 27/30, p=8e-06, NOT
  decidable (IQR rel 0.249, one pair 0.83x / one 1.89x -- co-load
  bursts INSIDE pairs; kept as the robustness check, s1 json).  The
  refresh magnitude is consistent with the r18 spot (1.239x/1.213x);
  decidability remains narrowly unmet on the IQR rule (0.208 vs
  3x0.085=0.255) due to one 2.17x burst-inflated pair.
* **Spots for the record table (12 pairs, --stats, same window):
  B1T2048 1.7842x (per-pair 1.7902x, p=0.0063) -- up from the 1.680x
  spot; B4T2048 1.5936x (per-pair 1.5101x, p=0.00049) -- up from the
  1.115x-era read** (final_adjudicate_r19f_spots.json; the F1KS ship
  moved B4's f1 -57.9us in r18 and the window was clean).
* **The r14u_ab B-side bias is ROOT-CAUSED: a module-construction-
  order PLACEMENT lottery on the memory-heaviest kernel, NOT launch
  order.**  Evidence chain (scratch/r19f): (1) same-cubin-on-both-
  sides with launch-order alternation still reads f3 B +14.8us
  (39/40) -- the r18/r19-mission launch-order hypothesis is dead;
  (2) flipping ONLY the FullKernel construction order mirrors it
  exactly (-16.8us, 0/40 -- ab_sameside_loadflip.json): whichever
  cuModuleLoadData claims its .global region second pays ~6% on f3's
  ~160MB working set at T8192; (3) in 4-module (balanced-harness)
  processes the draw is lottery-like: same-build f3 draws +2.3/+14.8/
  +14.3us (warmup-20 sessions) -> calibration mean **+10.5 +- 6us,
  f3-only**; f1/f2 are clean at warmup 20 (the f1 "+53us half-1" was
  clock ramp: warmup 8 -> 20 kills it, ab_sameside_warm.json
  -0.1us).  **f2 is exonerated: the r18 "+6us f2" was ramp, not
  placement.**
* **Harness fix (r14u_ab.py, default behavior): TWO half-sessions
  with mirrored module construction order (A1,B1 then B2,A2),
  per-rep launch-order alternation, first post-warmup pair discarded,
  warmup 20** -- deltas reported per half + mean-of-halves + pooled;
  --single-load/--fixed-order/--keep-first reproduce the legacy
  behavior.  Same-build validation: f1 +0.1/-0.1, f2 -3.1..+0.3, f3
  +14.8/+14.3 (the uncancellable draw component; corrected for by
  the calibration protocol: for any f3-sensitive comparison run a
  same-build session in the same window and subtract).
* **Corrections (balanced harness, all four cubins md5-verified
  against their recorded values; r14-US T8192 rebuilt to `b574395a`
  exactly):**  r15 WST vs r14-US: f1 -12.8 (7/80 slower), f2 -0.4,
  f3 +15.2 raw -> ~+3 corrected ~= 0 (recorded -0.4: verdict
  unchanged, WST's f3 store-widening is perf-neutral), staged -0.5.
  r16 bf16-solve vs r15 ship: f1 -3.0, f2 +0.9, f3 +15.0 raw ->
  ~+3 corrected ~= 0 (recorded +0.4: unchanged).  r18 F1KS vs r17
  ship: f1 **-56.4us (3/80 slower)**, f2 +0.2, f3 +14.0 raw -> ~+2
  corrected ~= 0, staged -49.9 raw -> **~-62 +- 5us corrected** --
  the r18 estimate "-60 to -71us" is confirmed and tightened to the
  f1-only win (f3 contributes ~0; the entire raw f3 B-side penalty
  was placement).  Recorded history is NOT rewritten; these are
  appended corrections.
* **F1WY (warp-parallel solve phase-1) is a correct-buildable honest
  negative: f1 +92.6/+90.4/+91.4us x3 sessions (77-80/80 pairs
  slower)** vs the F1KS ship at T8192.  Three cuts to correctness:
  (a) width-16 gpu.shuffle lowers to `shfl.sync ... 0xFFFF` membermask
  -- lanes 16..31 executing it is UB (PTX rule; found by the T8192
  gate after the T2048 "verify" turned out to be a stale gen_fwd
  cache hit -- the rt.build short-circuit hole struck again; use
  fresh --gen-root/scratch dirs for variant verifies); (b) the
  16-way select-by-lane store mux had a span-progression bug (mids
  {8,24,40..} -- a deterministic column permutation, cols >= 4, rows
  3+ of u 95% corrupt; numpy twin exact -- MY bug, found by the
  G5D_DUMP_TT probe fingerprint); (c) even with the mux fixed the
  toolchain miscompiled the select(r-update)+mux web (stale identity
  values read back).  **v3 design that passes: NO lane-indexed select
  webs** -- T rows live in the Tt smem tile (diag blocks pre-zeroed,
  the r18 NaN-proofing lesson), the store column is a compile-time c
  (warp-redundant broadcast stores), the butterfly is width-32 with
  lanes 16..31 contributing masked-A exact zeros, one gpu.barrier per
  step.  Correctness: T8192 fp64 gate PASS **BIT-IDENTICAL to the
  ship** (o/ht 0.000e+00); T2048 TT-dump probe: Tb/Tg 26/20 cells of
  2M flip at bf16-ulp (max 1.2e-7; the tree-order fp32 dots); verify
  4/4 PASS (staged T2048 o 6.486e-4 / ht 7.493e-4 -- the record
  digits); det N=5 fresh-process FWD-DETERMINISTIC; regs 174, pool
  unchanged, 2 CTA/SM intact.  Perf verdict: the legacy serial
  substitution's smem loads pipeline far better than the r18 estimate
  assumed, while the butterfly costs ~4-5x the instructions (1280
  shfl+add, 768 redundant stores, 16 extra CTA barriers per solve);
  at 2 CTA/SM that issue pressure dominates.  **GDN_F1WY stays
  default-off; not shipped.**  Round-20 direction: 4-threads-per-
  block x 4-columns register scheme -- no shuffles, no barriers, 4x
  ILP on the serial chain.
* **f3 2-CTA: audited + designed, build queued (round-20 #1).**
  Pool today 82,176B @1 CTA (qt/kt [64,40] + st/vnt [32,136] at
  multibuffer x2 = 55,296B, P [64,72] 9,216B, ot [64,136] 17,408B,
  gcs 256B).  Diet to ~47.4KB <= 50,688: (1) bk=16 K-loops (the F1KS
  kkt move) -12.9KB; (2) ot into P's dead arena (P dies at the oa
  epilogue, ot is written after it -- the identify-by-size PTX alias
  the f1 path already uses) -9.2KB.  Audit clean: f3's P-tile upper
  cells are exact-zero-written by epi_P over the full CxC (select
  col<=row, run_epilogue coverage); f3 has NO scans (no SCANDB-class
  exposure); no zero-init cell loops (no ZFIX-class stride exposure);
  the residual risk is co-residency scheduling -- apply the F1KS
  playbook gate first (forced-1-CTA bit-clean, then the 2-CTA
  bit-clean x10 reps).

## 1. Adjudication table (formal + spots, --stats)

| run | ours | fla | ratio | per-pair | pairs | flags |
|---|---|---|---|---|---|---|
| **B1T8192 formal n=30 (s2, the record)** | 0.8116ms | 0.9994 | **1.2314x** | 1.2080x | 30/30>1, p=0.0 | IQR rel 0.085; one 2.17x burst pair blocks the decidability rule |
| B1T8192 n=30 (s1, robustness) | 0.8199 | 1.1688 | 1.4254x | 1.3928x | 27/30, p=8e-06 | dirty window (old co-job exiting); co-load-sensitive |
| B1T2048 spot 12p | 0.1965 | 0.3506 | **1.7842x** | 1.7902x | p=0.0063 | was 1.680x |
| B4T2048 spot 12p | 0.8242 | 1.3134 | **1.5936x** | 1.5101x | p=0.00049 | was ~1.115x-era |

(jsons: final_adjudicate_r19f_T8192{,_s2}.json,
final_adjudicate_r19f_spots.json; cubin `e7af55c1...` via the
gen_fwd cache, verified before both sessions.)

## 2. The bias investigation + corrections table

| session (balanced harness, 40 reps x2 halves) | f1 | f2 | f3 raw | f3 corrected | staged |
|---|---|---|---|---|---|
| same-build draw1/draw2/draw3 | +0.1/-0.1 | -3.1..+0.3 | +2.3/+14.8/+14.3 | 0 by construction | -- |
| r15 WST vs r14-US (recorded -11.8/-11.4; +2.5*/-0.0; -0.4) | **-12.8** (7/80 slow) | -0.4 | +15.2 | ~+3 ~= 0 | -0.5 |
| r16 bf16sv vs r15 ship (recorded -2.6; -1.2; +0.4) | -3.0 | +0.9 | +15.0 | ~+3 ~= 0 | -- |
| r18 F1KS vs r17 ship (recorded -54.4; -1.3; +14.5; staged -47.6) | **-56.4** (3/80 slow) | +0.2 | +14.0 | ~+2 ~= 0 | **-49.9 raw -> ~-62+-5** |

Corrections applied to the RECORD (appended, history untouched):
* r15: WST f3 effect confirmed ~= 0 (recorded -0.4; the -0.4 was
  inside the bias band all along).  f1 -11.8/-11.4 -> -12.8 stands.
* r16: bf16-solve f2/f3 effects confirmed ~0 (recorded values stand).
* r18: the "true chain win -60 to -71us" estimate tightens to
  **-62 +- 5us, entirely f1**; the raw f3 B-side penalty (+14.5) was
  100% placement.  f2's recorded -1.3 stands (no f2 bias exists).

(cubin provenance for the pairs: r14us_T8192 `b574395a` (rebuilt,
exact), wst2_T8192_w4bv32 `5b9fcf48` (exact), r16z/base `5b9fcf48`,
bf16sv `e53e213b`, f1ksuw `e7af55c1`, shiprepro `e7af55c1` (exact).)

## 3. F1WY lever numbers (the honest negative)

| arm | f1 delta @T8192 | pairs slower | notes |
|---|---|---|---|
| v1 width-16 shfl | -- | -- | UB (membermask 0xFFFF); T8192 gate o 3.6e-1 FAIL; masked at T2048 by a stale-cache verify |
| v2 mux-span fix | -- | -- | still o 4.5e-1 FAIL (select-web miscompile) |
| **v3 (the knob)** | **+92.6 / +90.4 / +91.4us** | 77/80, 78/80, 80/80 | halves consistent [+85..+96]; gate BIT-IDENTICAL; verify 4/4 record digits; det N=5 |

Per-step cost accounting (why it loses): 1 A-load + 16 Tt loads +
16 muls + 80 shfl + 80 adds + ~35 sels + 48 broadcast stores + 1
CTA barrier, x16 steps, x all 4 warps -- vs the legacy's 2-warp
latency-chained while-loops whose loads the compiler pipelines.

## 4. Numerics + determinism (green after every step)

* fwd_verify 4/4 PASS on REAL F1WY builds (fresh gen-root
  scratch/r19f/gv_f1wy -- the stale-cache lesson): staged small
  6.382e-4/6.385e-4, staged T2048 **6.486e-4 / 7.493e-4 (record
  digits)**, fused unchanged (F1WY is mono-only).
* T8192 fp64 gate: F1WY **BIT-IDENTICAL to the r18 ship** (o/ht
  0.000e+00 vs f1ksuw_T8192 base).
* T2048 G5D_DUMP_TT probe (probe_ttdump.py): Tb 26 / Tg 20 cells of
  2,097,152 differ (bf16-ulp, max 1.2e-7 / 7.5e-9); vtd IDENTICAL.
* det N=5 fresh-process FWD-DETERMINISTIC (det_f1wy/, gen_fwd cache
  swap protocol documented in the round log).
* Ship provenance re-verified: ship-knob rebuild from the edited
  generator reproduces `e7af55c1` exactly (GDN_F1WY default-off is
  byte-non-perturbing); r14-US rebuild reproduces `b574395a`.

## 5. Ship provenance

No ship change this round (the r18 mono/fused trios stand).  New
knob: **GDN_F1WY** (default off; warp-parallel solve phase-1 v3;
correct, bit-identical at the gate, -92us-class perf regression --
available for the round-20 v4 to build on).  Tools changed:
tools/gdn_full_gen.py (+emit_solve_p1_wy, phase-1 branch),
evidence/G5/scratch/r14u/r14u_ab.py (the balanced two-load harness).

## 6. Round-20 list (measured, ranked)

1. **f3 2-CTA conversion** per the audited design: bk=16 K-loops +
   ot@P's dead arena -> ~47.4KB, 2 CTA/SM; forced-1-CTA bit-clean
   gate first, then 2-CTA x10 bit-clean reps (the F1KS playbook;
   f3 is 82,176B @1 CTA, the chain's second-largest kernel).
2. **f1 phase-1 v4**: 4-threads-per-block x 4-columns register
   scheme (no shuffles, no barriers, no select webs; 4x ILP on the
   serial chain; the F1WY post-mortem direction).
3. **B4T2048 formal adjudication** in a clean window (the 1.594x
   spot says the record can move far above the 1.115x-era number).
4. **f2 >=64-col row geometry** (r16 #3, still last, untouched).
5. Protocol debt: fold the placement-calibration (same-build f3
   session) into any f3-sensitive A/B; consider an rt.build
   gen-dir keying on knob-set hashes (the stale-cache hole has now
   bitten twice: r3, r19).

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
# formal record (s2) + robustness (s1) + spots:
python tools/gdn_final_adjudicate.py --shapes B1T8192 --pairs 30 \
    --warps 4 --bv 32 --stats --out <json>       # cache = gen_fwd ship
python tools/gdn_final_adjudicate.py --shapes B1T2048,B4T2048 \
    --pairs 12 --warps 4 --bv 32 --stats --out <json>
# balanced A/B (new default: mirrored construction orders):
python evidence/G5/scratch/r14u/r14u_ab.py --shape B1T8192 \
    --a <ref> --b <var> --reps 40 [--only f1,f2,f3,staged]
#   f3 reads carry the placement draw: run a same-build session
#   (--a X --b X) for the window's calibration; f1/f2 are clean.
# corrections provenance: r14us_T8192 (GDN_F2US=1 w4/bv32) = b574395a;
#   wst2 (r15x) = 5b9fcf48; r16z/base = 5b9fcf48; bf16sv = e53e213b;
#   f1ksuw = e7af55c1 (the ship)
# F1WY arm (correct, slower; default off):
GDN_F1KS=1 GDN_F2US=1 GDN_WST=1 GDN_ZFIX=1 GDN_SCANDB=1 GDN_F1WY=1 \
    <python> tools/gdn_full_gen.py --path mono --shape 1,8192,16,128,128 \
    --warps 4 --bv 32 --bv3 128 --out <d>; bash scripts/run_l2.sh <d>; \
    GDN_WST=1 <python> tools/gdn_smem_fix.py <d>
# gates: evidence/G5/scratch/r14u/r14u_t8192_gate.py --dirs base=<b>,var=<v>
# verify (REAL variant builds -- fresh gen-root, never the shared cache):
GDN_...=1 GDN_F1WY=1 <python> tools/gdn_full_verify.py \
    --gen-root <fresh_dir> --warps 4 --bv 32
# det: evidence/G5/scratch/r18f/run_det.py <tag> mono <KNOBS> <out> 5
#   (gen_fwd cache swap required for variant knobs -- see round log)
# bias study: scratch/r19f/{ab_sameside_*.json, probe_f1wy.py,
#   probe_ttdump.py}; f1wy builds in scratch/r19f/{f1wy,dump}_*
# knobs: GDN_F1WY (warp-parallel solve phase-1; DEFAULT OFF, perf
#   negative this round), all prior knobs unchanged
```


# G5 round 20 (R20-fwd): the f3 2-CTA conversion built bit-clean and measured NEGATIVE (honest close: the bk16 kloop diet costs ~+110us, the 2-CTA win is only ~+20 -- f3 is bandwidth-bound); f1 phase-1 v4 correct-and-lean but negative (the serial chain wins again); the B4T2048 formal record moves to 1.188x (30/30)

Date: 2026-10-08 | shared/opportunistic window (co-job pid 3908172
~8.9GB resident throughout, 23-41% util bursts, never touched; window
snapshots in the JSONs; coordination row `gdn-mlir R20-fwd`).  HEAD at
round start: 3accc73; the ship-knob rebuild from the edited generator
reproduced the r18 mono-T8192 ship cubin `e7af55c1...` exactly (the
knob-off proof), and the knob-off payload diff was byte-identical to
HEAD after BOTH generator edits (mono T2048, x2).

## Verdict

* **The audited f3 2-CTA design is BUILT, BIT-CLEAN, and measured
  NEGATIVE -- the door is closed by measurement, with the mechanism.**
  The r19 design landed exactly as specified (GDN_F3KS, default off):
  both K-loops at bk=16 (qt/kt [64,24] multibuffer 6,144B each, st/vnt
  [16,136] 8,704B each; mma order unchanged -- bk16 splits BETWEEN the
  m16n8k16 blocks) + ot in P's dead arena (P's malloc declared at ot's
  17,408B, identify-by-size alias at +0, one explicit barrier after
  the oa loop pinning every warp's last P ldmatrix before the first ot
  store).  f3 pool 82,176 -> 47,360B, cuOccupancyMaxActiveBlocks
  2/SM (f1/f2 untouched: 46,608@2 / 94,720@1); the F1CTA diag arm
  (+7 dead rows on st -> 51,200B) forces 1/SM.  The F1KS-playbook
  gates ALL green: forced-1-CTA BIT-IDENTICAL to the ship (T2048 trio
  7/7 arrays; T8192 fp64 gate o/ht 0.000e+00), 2-CTA f3-only x10 reps
  bit-clean at BOTH T2048 and T8192 (rep-to-rep deterministic), full
  trio 7/7 bit-identical, B4 x3 o/ht max 0.0, verify 4/4 record
  digits, det N=5.  Perf (balanced two-load harness): T8192 f3
  **+96.1us (74/80 slower)**, staged +97.0; B4T2048 f3 **+74.9us
  (76/80)**.  The DECOMPOSITION arm settles the mechanism: the SAME
  bk16 code forced to 1 CTA is **+115.6us** -- the kloop restructure
  itself costs ~+110us on f3 (2x iterations over the 3-stage qo loop
  + the oa loop: double the barrier/wait/staging-issue steps), and
  2-CTA residency buys back only ~+20us (116 -> 96, before the +8.7
  same-build placement-draw correction).  f3 streams ~120MB of
  h_all/v_new/o at T8192 -- a bandwidth-bound kernel does not pay for
  wave-halving (f1's -53.5us win was the latency-bound case).
  **GDN_F3KS stays default-off; not shipped.**
* **f1 phase-1 v4 (GDN_F1P4, default off) is correct, lean, and
  negative: f1 +14.8/+14.5/+15.5us x3 sessions (70-76/80 slower).**
  The design: 4 threads per diagonal block x 4 REGISTER-TILED columns
  -- per row an scf.while k-chain with FOUR interleaved accumulator
  chains (4x FMA ILP, one shared A[i,k] load per step amortized over
  4 columns, ~4.5 instr/FMA vs the serial chain's ~8), each thread
  touching ONLY its own 4 columns (no shuffles, no barriers, no
  lane-indexed select webs -- the three F1WY correctness cuts).
  Bit-identity by construction: the k<c terms read Tt's -0.0 uppers
  and (+0.0)+(+-0.0) leaves the accumulator +0.0 (IEEE zero-sum), so
  the REAL terms accumulate in the SAME ascending order as the serial
  chain -- T8192 gate BIT-IDENTICAL to the ship, T2048 trio 7/7
  IDENTICAL, verify 4/4 record digits, det N=5, regs 172 -> 168, pool
  unchanged, 2 CTA/SM intact.  One build lesson: the k-loop had to be
  scf.while, NOT scf.for -- the payload transform's
  pipeline_shared_memory_copies matches every scf.for and errors
  "no shared memory copy" on copy-free loops (why the serial chain
  was a while all along).  The perf verdict is the F1WY post-mortem
  thesis confirmed a second way: 16 active threads x 4-wide ILP loses
  to 64 threads x 1-wide -- the serial substitution's smem loads
  pipeline so well that phase-1's THREAD-level parallelism, not its
  per-thread chain, is what matters.  **GDN_F1P4 default-off; the
  phase-1-as-critical-path estimate is now dead: round-21 must
  re-instrument WHERE f1's 225us actually goes.**
* **The B4T2048 formal record moves: 1.1883x ratio-of-medians /
  1.1843x per-pair, 30/30 interleaved pairs faster, sign-test p=0.0**
  (final_adjudicate_r20f_B4T2048.json; ours 0.8472ms vs fla 1.0067ms,
  window clean, pre/post 3% util).  NOT decidable on the IQR rule
  ((median-1)=0.188 < 3x0.120=0.361; one 0.92x pair, max 1.77x --
  the swing is fla-side: fla min 0.825ms this session vs the 1.31ms
  median of the r19 1.594x spot window).  Formal history: ~1.115x-era
  (r14: 1.094x window / 1.140x archived) -> 1.188x.
* **Protocol debt paid: the rt.build stale-cache hole is closed.**
  builds now stamp full_spec.json with gdn_cache_key = sha256(all
  GDN_*/G5* env knobs + the gdn_full_gen/gdn_smem_fix sources); a
  later build under a different knob-set/generator redirects to
  <dir>_k<hash8> instead of returning (or clobbering) the cached
  cubin -- legacy un-stamped dirs are trusted (the gen_fwd ships keep
  their slots).  Tested: same-knob cache hit, different-knob redirect
  (fresh _k dir), legacy trust.  This is the hole that bit r3 and
  r19 (the F1WY T2048 "verify" that was a stale gen_fwd hit).  The
  f3 placement-calibration protocol (same-build session per window)
  was followed this round for every f3-sensitive read.

## 1. Adjudication table

| run | ours | fla | ratio | per-pair | pairs | flags |
|---|---|---|---|---|---|---|
| **B4T2048 formal n=30 (the record)** | 0.8472ms | 1.0067 | **1.1883x** | 1.1843x | 30/30>1, p=0.0 | IQR rel 0.120; not decidable; min pair 0.92x / max 1.77x |
| B1T8192 formal (r19 s2, stands) | 0.8116 | 0.9994 | 1.2314x | 1.2080x | 30/30, p=0.0 | unchanged this round |
| B1T2048 spot (r19, stands) | 0.1965 | 0.3506 | 1.7842x | 1.7902x | 12p | unchanged this round |

(json: final_adjudicate_r20f_B4T2048.json; cubin `a065a0bc...` = the
r18 B4 mono ship via the gen_fwd cache, md5-verified.)

## 2. Lever numbers

| lever | arm | delta @T8192 | pairs slower | verdict |
|---|---|---|---|---|
| f3 2-CTA (GDN_F3KS) | 2-CTA vs ship | f3 **+96.1us**, staged +97.0 | 74/80 | NEGATIVE (draw-corrected ~+87) |
| f3 2-CTA decomposition | SAME code @forced-1-CTA | f3 **+115.6us** | 76/80 | the bk16 loops cost ~+110; 2-CTA buys ~+20 |
| f3 2-CTA @B4T2048 | 2-CTA vs ship | f3 **+74.9us**, staged +73.0 | 76/80 | consistent |
| same-build calibration | f3ks vs f3ks | f1 +0.2, f2 -0.2, f3 +8.7 (halves 6.2/11.1) | 75/80 | the uncancellable draw |
| f1 p1 v4 (GDN_F1P4) | vs ship x3 | f1 **+14.8/+14.5/+15.5us** | 70-76/80 | NEGATIVE, consistent halves [14..16] |
| f1 p1 v4 x3 | f2 / f3 raw | -0.4..+0.2 / +8.7..+15.8 | -- | f2 cubin-identical; f3 raw = placement draw |

(f3ks cubins: f3ks_T8192 `684b8845`, f3ks_1cta_T8192 `c0687a7d`,
f3ks_T2048 `80ed2302`, f3ks_1cta_T2048 `a70f6943`, f3ks_B4T2048
`a9872637`, f3ks_1cta_B4T2048 `97b64322`; f1p4_T8192 `a2ca837f`,
f1p4_T2048 `07970799`.  jsons: ab_f3ks_T8192_s1, ab_f3ks1cta_T8192_s1,
ab_f3ks_B4_s1, ab_same_T8192, ab_f1p4_T8192_s1..3 -- scratch/r20f.)

## 3. Numerics + determinism (green after every step)

* knob-off purity: mono payload byte-identical to HEAD x2 (after the
  f3 edit and again after the f1p4 edit); ship-knob rebuild from the
  edited generator reproduces `e7af55c1` exactly.
* GDN_F3KS gates: forced-1-CTA trio 7/7 IDENTICAL vs ship (T2048) and
  T8192 fp64 gate BIT-IDENTICAL (o/ht 0.000e+00, PASS digits
  6.626e-4/8.918e-4); 2-CTA f3-only x10 IDENTICAL + rep-deterministic
  at T2048 AND T8192; full trio 7/7 IDENTICAL; B4 x3 o/ht max 0.0
  deterministic; fwd_verify 4/4 PASS record digits
  (fwd_verify_r20f_f3ks.json: staged T2048 6.486e-4/7.493e-4); det
  N=5 fresh-process FWD-DETERMINISTIC (det_f3ks/).
* GDN_F1P4 gates: T8192 fp64 gate BIT-IDENTICAL to ship; T2048 trio
  7/7 IDENTICAL + deterministic; fwd_verify 4/4 record digits
  (fwd_verify_r20f_f1p4.json); det N=5 FWD-DETERMINISTIC (det_f1p4/).
* probe gotcha (documented for future rounds): an f3-only relaunch in
  a FRESH process reads ZERO gc/h_all/v_new -- rt.stage_inputs zeroes
  every non-input global; the f3-only corruption study must reuse ONE
  module load (trio once, then f3 relaunches).  scratch/r20f/
  probe_f3.py implements this (plus --shape T2048/T8192/B4T2048).

## 4. Ship provenance

No ship change this round (the r18 mono/fused trios stand; every A/B
was vs the md5-verified r18 ship cubins e7af55c1 / a065a0bc /
ea8ec2f1, and the HEAD-rebuild reproduced e7af55c1 exactly).  New
knobs, both DEFAULT OFF, unset == HEAD byte-identically:
* **GDN_F3KS** -- the f3 2-CTA conversion (bk16 kloops + ot@P's dead
  arena, pool 47,360B, 2 CTA/SM).  Correct everywhere, perf negative.
  Diagnostic arm **GDN_F3KS_F1CTA** (+7 dead rows on st -> 51,200B,
  1 CTA/SM; the SASS-diff-constant trick, values bit-identical).
* **GDN_F1P4** -- the f1 solve phase-1 v4 (4 threads/block x 4-column
  register tiles, scf.while k-chains with 4-way FMA ILP).  Correct,
  bit-identical, leaner per-FMA, perf negative.
* tools/gdn_full_gen.py: gen_f3 knob branch + the merged per-kernel
  smem_alias.json writer (F1KS+F3KS stack in one mono build);
  emit_solve_blocked: the F1P4 branch.
* tools/gdn_full_rt.py: the _cache_key/gdn_cache_key build stamping
  + _k<hash8> variant-dir redirect (legacy dirs trusted).

## 5. Round-21 list (measured, ranked)

1. **f1 critical-path re-audit**: two phase-1 restructures measured
   negative (F1WY +92us, F1P4 +15us) -- the "phase-1 is f1's
   critical path" estimate is dead.  Instrument the real 225us
   (kkt loop? phase-2 R=1..3 mma chain? uw stage? zero/scan phases?)
   before any more solve work; the tt-dump probe machinery extends.
2. **B4T2048 decidability**: 1.188x formal is 30/30 but IQR-blocked
   by fla-side swing (min pair 0.92x); a calmer window rerun could
   make it decidable.
3. **f2 >=64-col row geometry** (r16 #3, untouched, still last).
4. f3 alternatives to the closed bk16 door: K-split-across-CTAs (the
   r19-bwd fwdscan GO design) if f3 must shrink its wave exposure;
   otherwise f3 is done at 1 CTA (bandwidth-bound).
5. Protocol: the det cache-swap trap must mv the variant OUT, not
   `mv $C.shipbak $C` blind (this round nested a shipbak two deep
   when run_det left $C populated; recovered, md5-verified) -- fold
   the swap into run_det.py itself.

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
# f3 2-CTA arm (correct, slower; default off):
GDN_F1KS=1 GDN_F2US=1 GDN_WST=1 GDN_ZFIX=1 GDN_SCANDB=1 GDN_F3KS=1 \
    <python> tools/gdn_full_gen.py --path mono --shape 1,8192,16,128,128 \
    --warps 4 --bv 32 --bv3 128 --out <d>; bash scripts/run_l2.sh <d>; \
    GDN_WST=1 <python> tools/gdn_smem_fix.py <d>
#   + GDN_F3KS_F1CTA=1 = the forced-1-CTA gate arm
# f1 phase-1 v4 arm (correct, slower; default off): ship knobs + GDN_F1P4=1
# gates: evidence/G5/scratch/r14u/r14u_t8192_gate.py --dirs base=<ship>,var=<d>
# f3-only corruption study (ONE module load):
#   evidence/G5/scratch/r20f/probe_f3.py --shape T2048 --one-cta <1cta> \
#       --two-cta <2cta> --reps 10
# B4 formal record: python tools/gdn_final_adjudicate.py --shapes B4T2048 \
#     --pairs 30 --warps 4 --bv 32 --stats --out <json>
# balanced A/B: evidence/G5/scratch/r14u/r14u_ab.py --shape B1T8192 \
#     --a <A> --b <B> --reps 40   (f3 reads carry the placement draw:
#     run --a X --b X same-build calibration in the window and subtract)
# det (cache swap): swap gen_fwd/<tag>_mono aside, run
#   evidence/G5/scratch/r18f/run_det.py <tag> mono "<KNOBS>" <out> 5 4 32
# knobs: GDN_F3KS / GDN_F3KS_F1CTA / GDN_F1P4 (all DEFAULT OFF,
#   perf negatives this round), all prior knobs unchanged
```

# G5 round 21 (R21-fwd): the f1 critical-path TIME-MAP is built (10 deletion probes x3 + ncu: uw stage 51%, kkt 26%, solve p1 14% -- the phase-1-as-critical-path thesis is DEAD); GDN_F1ZF (phase-0 folded into phase-1's idle threads) bit-identical, f1 -7.4us T8192 x3 / -2.3us T2048; B4T2048 rerun 1.2329x, still IQR-blocked (fla-side swing)

Date: 2026-10-08 | shared/opportunistic window (co-job ~8.9-9.2GB
resident throughout, 0-33% util bursts, never touched; window snapshots
in every json; coordination row `gdn-mlir R21-fwd`).  HEAD at round
start: 2bea03f.  Knob-off purity: the ship-knob rebuild from the edited
generator reproduced `e7af55c1` exactly at T8192 **x3** (after the
probe-knob edit, after the keep-alive-sink edit, after the F1ZF edit)
and `ea8ec2f1` at T2048 — every GDN_F1T_*/GDN_F1ZF gate is
default-off and inert.

## Verdict

* **The f1 time-map (mission #1) is delivered.**  Ten composable
  deletion probes (GDN_F1T_*, new, default off, timing-only — outputs
  are garbage under any of them) were built on the ship knob set at
  T8192 and A/B'd x3 sessions each (r14u_ab balanced harness, f1-only,
  40 pairs x2 halves; ship f1 median 225.8-236.0us across sessions,
  ~227us basis).  Apparent phase costs (delta = ship - probe, mean of
  3 sessions, all three sessions within +-1.6us of each other):

| phase deleted | knob | delta @T8192 | share of f1 |
|---|---|---|---|
| uw stage (ug/wg gemms + epilogue + u/w s2g) | NUW | **-116.5us** | **51.2%** |
| uw stage + operand staging (k/v reads) | NUST | -131.6us | 57.9% |
| kkt K-loop + A epilogue | NKT | **-59.5us** | **26.2%** |
| whole solve (p0+p1+p2) | NALL | -41.3us | 18.2% |
| solve phase 1 (serial diagonal) | NP1 | **-32.5us** | **14.3%** |
| u/w s2g copies ONLY (gemms+epi kept) | NC2G | -31.0us | 13.6% |
| gc Hillis-Steele scan + beta stage | NSC | -17.6us | 7.8% |
| solve phase 2 (off-diag mma rounds) | NP2 | -14.6us | 6.4% |
| solve phase 0 (zero_up) | NP0 | -8.5us | 3.7% |
| gc_out store | NGC | -0.8us | 0.3% |

  Derived splits: **uw staging (k+v global reads) ~= 15.1us**
  (NUST-NUW); **uw gemms+epilogue+staging-wait ~= 85.5us**
  (NUW-NC2G); **u/w s2g global stores ~= 31.0us** (NC2G).  The
  single-phase shares sum to ~109% -- phases contend (DRAM/L2), so
  each delta is an UPPER bound on the phase's serial cost; the
  composition cross-check NALL (-41.3) vs NP0+NP1+NP2 (-55.6) shows
  ~14us of intra-solve overlap.
* **ncu (ship f1, exclusive, clocks locked @1.57GHz): 313.95/312.06us,
  DRAM 56.8-57.7% (400.8 GB/s), compute SOL 33%, occupancy 16.24%
  achieved** (theoretical 16.67%: 8 warps/SM — Block-Limit-Registers=2
  AND Block-Limit-SharedMem=2, 214 regs/thread, 46.61KB static smem),
  grid 2048 CTAs / 82 SMs = **12.49 waves**, scheduler No-Eligible
  82.6% (an issue every 5.8 cycles), warp-stall mix per issued inst:
  **long_scoreboard 24.3%, barrier 22.9%, wait 17.8%, math_pipe
  12.4%**, short_scoreboard 5.9%, branch 3.1%; L1 hit 0%, L2 hit
  40.3%, avg active threads/warp 26.8.  f1 is LATENCY-bound at 2
  CTAs/SM, not bandwidth- or tensor-bound (u/w+k/v ~= 128MB moved per
  launch ~= 57% of DRAM).
* **The R20 question is answered: phase-1 is NOT f1's critical path**
  (32.5us, #3).  The uw stage is (116.5us, 51%) — and within it the
  ~85.5us mma+epilogue+wait serialization, not the 31us of store
  traffic.  Both prior phase-1 restructures (F1WY +92, F1P4 +15)
  attacked a 14% phase; the round-22 levers attack the 51% one.
* **GDN_F1ZF SHIPPED (bit-identical, won x3): solve phase 0 folded
  into phase-1's idle threads.**  Threads 64..127 (idle through the
  ~32us serial substitution) zero the transposed-upper cells
  concurrently; the folded predicate block(r)>block(c) (implies r>c)
  writes ONLY cells the substitution never touches (it writes every
  same-block cell incl. its -0.0 uppers), so write sets are disjoint —
  and ONE barrier replaces the phase-0/phase-1 pair.  Gates: T2048
  trio 7/7 arrays IDENTICAL to ship; T8192 fp64 gate BIT-IDENTICAL
  (o/ht 0.000e+00) with record digits 6.626e-4/8.918e-4; fwd_verify
  4/4 PASS staged T2048 6.486e-4/7.493e-4 (the record digits,
  fwd_verify_r21f_f1zf.json); det N=5 FWD-DETERMINISTIC (det_f1zf/).
  Perf (f1-only, interleaved): **T8192 -7.7/-7.9/-6.7us x3 sessions
  (77-78/80 pairs), T2048 -2.3us (57/80)** — ~91% of the NP0
  deletion bound (the fold overlaps, the delete abolishes).
  GDN_F1ZF joins the canonical ship knob set
  (F1KS+F2US+WST+ZFIX+SCANDB+**F1ZF**) for every build from now on;
  the standing formal records stay pinned to the r18 cubins until
  re-adjudicated (round-22 #1).  Cubins: f1zf_T8192 `be8b914b`,
  f1zf_T2048 `3d47886a` (scratch/r21f).
* **B4T2048 decidability rerun (mission #2): 1.2329x ratio-of-medians
  / 1.2308x per-pair, 27/30 pairs, sign-test p=8e-06 — STILL NOT
  DECIDABLE, and the window was NOT calmer** (IQR 0.2364 / rel 0.192
  vs r20's 0.120; fla-side swing: fla min 0.936ms / max 2.58ms in the
  same session; ours tight 0.824-0.850ms; window_pre 20% util, post
  33%).  (median-1)=0.231 < 3x0.192=0.576.  The **1.1883x record
  (r20) stands**; this session centers at 1.23x — both windows agree
  on the ~1.19-1.23x center, and the blocker is fla's own tail, not
  our spread.
* Protocol debt (r20 list #5) PAID: run_det.py now owns the gen_fwd
  cache swap (idempotent aside/restore with md5 prints, try/finally,
  variant preserved at <out>_variant — OUT of the npz dir, which the
  first cut broke --diff on).  Exercised by this round's det_f1zf run.
  AUDIT FINDING while det-ing: the legacy det slot
  gen_fwd/bench_B1T2048H16_mono held a STALE non-ship cubin
  (`396be193`, un-stamped, pre-r20 build; the exact hole the r20
  cache keys closed) — preserved as `.stale396b` and the slot
  normalized to the true ship `ea8ec2f1`.

## 1. Adjudication table

| run | ours | fla | ratio | per-pair | pairs | flags |
|---|---|---|---|---|---|---|
| B1T8192 formal (r19 s2, stands) | 0.8116 | 0.9994 | **1.2314x** | 1.2080x | 30/30, p=0.0 | unchanged; F1ZF migration queued r22 |
| **B4T2048 rerun n=30 (r21f)** | 0.8418 | 1.0378 | 1.2329x | 1.2308x | 27/30, p=8e-06 | IQR rel 0.192; NOT decidable; does NOT supersede the 1.1883x record (same cubin `a065a0bc`, calmer-window attempt failed — fla swing) |
| B1T2048 spot (r19, stands) | 0.1965 | 0.3506 | 1.7842x | 1.7902x | 12p | unchanged |

(json: final_adjudicate_r21f_B4T2048.json.)

## 2. Lever numbers

| lever | arm | delta @T8192 f1 | pairs faster | verdict |
|---|---|---|---|---|
| **GDN_F1ZF zero-fold** | f1zf vs ship s1 | **-7.7us** | 78/80 | SHIP (bit-identical; joins knob set) |
| GDN_F1ZF s2 | | -7.9us | 77/80 | |
| GDN_F1ZF s3 | | -6.7us | 77/80 | |
| GDN_F1ZF @T2048 | | -2.3us | 57/80 | consistent |
| (bound) NP0 delete | probe | -8.5us | 79/80 | the fold realizes ~91% |

(probe deltas: table in Verdict; jsons ab_probe_*_s1..3 + ab_f1zf_*
+ ab_nc2g_* -- scratch/r21f, all .gitignored by policy, scripts
checked in alongside.)

## 3. Numerics + determinism (green after every step)

* knob-off purity: e7af55c1 reproduced x3 at T8192 (after each
  generator edit), ea8ec2f1 at T2048.
* F1ZF gates: T2048 trio 7/7 IDENTICAL (gate_t2048.py);
  T8192 fp64 gate BIT-IDENTICAL o/ht 0.000e+00, PASS digits
  6.626e-4/8.918e-4 (r14u_t8192_gate.py); fwd_verify 4/4 PASS
  (small 512: staged 6.382e-4/6.385e-4, fused 4.752e-4/6.997e-4;
  T2048: staged 6.486e-4/7.493e-4 — the record digits —, fused
  5.751e-4/7.929e-4); det N=5 fresh-process FWD-DETERMINISTIC.
* GDN_F1T_* probes are timing-only (garbage outputs); they never
  touch verify/det paths and are barred from them by doc.
* Build lesson (documented for probe rounds): the payload transform's
  multibuffer pattern HARD-ERRORS on a workgroup alloc with ZERO
  users — every deletion that orphans a tile (NKT->kt, NP2->St*,
  NUST->vt/k2, NALL->Tt) needs a keep-alive sink; sinks are one
  predicated scalar load->gc_out store each, placed to respect the
  PTX lifetime arena (Tt pre-solve, vt/k2 at kernel end — the alias
  windows open post-solve; pools verified unchanged).

## 4. Ship provenance

* tools/gdn_full_gen.py (R21-fwd's file): the GDN_F1T_* deletion-probe
  family (NP0/NP1/NP2 in emit_solve_blocked; NSC/NKT/NGC/NUW/NUST/
  NC2G in gen_f1; keep-alive sinks; ALL default off, timing-only) +
  **GDN_F1ZF** (the phase-0 fold; default off, in the canonical ship
  knob list from this round).
* evidence/G5/scratch/r18f/run_det.py: the swap protocol fix (r20 #5).
* evidence/G5/scratch/r21f/: build/probe/gate/summarize scripts +
  all ab_*.json + ncu_ship_f1_T8192.txt + ncu_stalls_ship_T8192.csv
  (gitignored scratch by policy; cited by this section).
* gen_fwd/bench_B1T2048H16_mono: normalized to the true ship
  (`ea8ec2f1`); the stale pre-r20 cubin preserved as
  bench_B1T2048H16_mono.stale396b (audit trail).
* Standing record ships unchanged (e7af55c1 / a065a0bc / ea8ec2f1).

## 5. Round-22 list (measured, ranked)

1. **F1ZF record migration**: rebuild the gen_fwd record ships with
   the +F1ZF knob set and re-record the three formals (expected
   ~+0.3-1% e2e: f1 -7.4us on an 812us T8192 chain, -2.3us on B4).
2. **uw-stage serialization (~85.5us, the map's #1)**: two concrete
   designs — (a) single-pass u/w epilogue staging the second row-half
   in Tb/Tg's dead bytes (both dead after the uw ldm reads) via the
   PTX arena, halving the restage barriers+copies; (b) issue uw
   pass-1's cp.async group EARLIER, into kt's dead post-kkt bytes
   (6,144B; a [32,40]bf16 half of the k2 read = 2,560B fits) to hide
   its ~15us under the solve.  Both need arena-spec edits + lifetime
   proofs (the r18 F1KS playbook).
3. **k double-read (kkt -59.5 + uw staging -15.1 both stream k)**:
   persistent-k is pool-blocked at the 2-CTA line; only viable with a
   pool shrink elsewhere — do the pool budget table first.
4. **B4 decidability**: two sessions now agree on the center but the
   fla-side tail blocks the IQR rule; propose (methodology sign-off
   needed) a trimmed-per-pair adjudication or n=50 archival session.
5. f2 >=64-col row geometry (r16 #3, untouched, still last).

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
# ship knob set FROM THIS ROUND (F1ZF joined):
#   GDN_F1KS=1 GDN_F2US=1 GDN_WST=1 GDN_ZFIX=1 GDN_SCANDB=1 GDN_F1ZF=1
# f1 time-map probes (timing-only; ANY GDN_F1T_* makes garbage):
S=evidence/G5/scratch/r21f
for K in NSC NKT NP0 NP1 NP2 NGC NUW NUST NC2G; do
  GDN_F1KS=1 GDN_F2US=1 GDN_WST=1 GDN_ZFIX=1 GDN_SCANDB=1 \
    GDN_F1T_$K=1 <python> tools/gdn_full_gen.py --path mono \
    --shape 1,8192,16,128,128 --warps 4 --bv 32 --bv3 128 --out $D; \
    bash scripts/run_l2.sh $D; GDN_WST=1 <python> tools/gdn_smem_fix.py $D
done
# A/B: scratch/r14u/r14u_ab.py --shape B1T8192 --a <ship> --b <probe> \
#     --only f1 --reps 40   (x3 sessions; see scratch/r21f/run_ab.sh)
# ncu sections/metrics: see scratch/r21f/ncu_ship_f1_T8192.txt and
#   ncu_stalls_ship_T8192.csv (launcher ncu_launch.py <dir> f1 12 T8192)
# F1ZF gates: scratch/r21f/gate_t2048.py <ship_T2048> <f1zf_T2048>;
#   scratch/r14u/r14u_t8192_gate.py --dirs base=<ship>,var=<f1zf>;
#   fwd_verify --gen-root <fresh> with the +F1ZF env;
#   det (swap now built into run_det.py):
#   scratch/r18f/run_det.py bench_B1T2048H16 mono \
#     "GDN_F1KS=1,GDN_F2US=1,GDN_WST=1,GDN_ZFIX=1,GDN_SCANDB=1,GDN_F1ZF=1" \
#     <out> 5 4 32
# B4 rerun: python tools/gdn_final_adjudicate.py --shapes B4T2048 \
#     --pairs 30 --warps 4 --bv 32 --stats --out <json>  (ship env)
# knobs: GDN_F1ZF (ship set member; default off in code),
#   GDN_F1T_{NP0,NP1,NP2,NSC,NKT,NGC,NUW,NUST,NC2G} (timing probes,
#   never verified), all prior knobs unchanged
```
# G5 round 22 (R22-fwd): the F1ZF record migration DONE -- all three formals are now n=30 on ONE cubin generation (B1T2048's first formal: 2.1184x); the uw epilogue single-pass (GDN_F1EP) built bit-identical through Tb/Tg's dead bytes and measured a WASH (+0.0..+0.8us x3) -- intra-kernel serialization class CLOSED at 2 CTAs/SM; design (b) priced NO-GO; the k-double-read pool budget table closes persistent-k; B4 trimmed-per-pair decidability proposal

Date: 2026-10-08 | shared/opportunistic window (co-job ~8.9-10.9GB
resident throughout, 0-54% util bursts, never touched; window snapshots
in every json; coordination row `gdn-mlir R22-fwd` in
~/codebase/GPU-COORDINATION.md).  HEAD at round start: cd88c0a.
Knob-off purity: the ship-knob rebuild (NO F1ZF) from HEAD reproduced
all three r18 record md5s -- `e7af55c1` / `a065a0bc` / `ea8ec2f1`
(the B4 rebuild-repro is a FIRST; r20/r21 only ever reproduced
T8192/T2048) -- and after the F1EP generator edit landed the no-knob
path reproduced `e7af55c1` again (pur2; the edit is inert when off).

## Verdict

* **Mission #1, the F1ZF record migration: DONE.**  All three record
  ships rebuilt from HEAD cd88c0a with the canonical knob set
  (F1KS+F2US+WST+ZFIX+SCANDB+F1ZF): T8192 `be8b914b`, T2048
  `3d47886a` (both md5-IDENTICAL to r21's f1zf cubins -- same
  generator + knobs, reproducibility confirmed), B4T2048 `83898f74`
  (new; B4 had never been built +F1ZF).  The +F1ZF adjudications ran
  from a FRESH stamped gen root (scratch/r22f/genrec, cache_key
  0d667ecf): the legacy un-stamped gen_fwd slots would have been
  "trusted" and silently served the r18 cubins -- the exact legacy
  hole the r20 cache keys left.  The gen_fwd slots themselves are
  migrated at round end (see section 4), r18 bytes preserved.
* **All three formal records are n=30 --stats interleaved
  adjudications on the SAME cubin generation** (B1T2048's formal had
  been spot-only, 1.784x/12 pairs, since r19).  One session, three
  shapes, 30 pairs each (final_adjudicate_r22f_f1zf.json):

| shape | ours | fla | ratio | per-pair | pairs | IQR rel | decidable |
|---|---|---|---|---|---|---|---|
| B1T8192 | 0.7905ms | 0.9665 | **1.2226x** | 1.2054x | 27/30, p=8e-06 | 0.296 | no (fla tail; min pair 0.80x) |
| B4T2048 | 0.8481 | 0.9682 | **1.1417x** | 1.1450x | 30/30, p=0.0 | **0.060** | no -- by 0.039 (0.142 < 3x0.060=0.180) |
| B1T2048 | 0.1980 | 0.4194 | **2.1184x** | 2.1113x | 29/30, p=0.0 | 0.374 | no (fla tails to 1.26ms vs 0.42 median) |

  Ours-side is tight everywhere (T8192 0.782-1.046 with one co-job
  burst sample; B4 0.807-0.913; T2048 0.191-0.272); every IQR block
  is fla-side tail, the same blocker r20/r21 documented.  Durable
  content: the BEST formal ours-median at T8192 yet (0.7905ms vs
  r19's 0.8116 on the r18 cubins; F1ZF is worth ~-8us of the -21us,
  the rest is window draw), B4 ours-side UNCHANGED (0.8481 vs r20's
  0.8472 -- the F1ZF win is sub-us at B4's per-launch f1 share), and
  B4's IQR rel 0.060 is the TIGHTEST B4 window ever recorded (r20
  0.120 / r21 0.192) -- 30/30, p=0.0, blocked only because fla's
  median sat fast.  The ratio-center MOVES are fla-side (fla medians
  0.9665/0.9682/0.4194 this session vs 0.9994/1.0067/0.3506 in the
  r19/r20 windows) -- records are window-bound estimates; the cubin
  generation + ours-side medians are what carry.
* **Mission #2, the uw-stage serialization attack: both r21 arena-spec
  designs priced, (a) BUILT, and the class is now CLOSED by
  measurement.**
  * **(a) GDN_F1EP (built, bit-identical, WASH)**: the ship's F1KS
    epilogue stores+drains u/w in two row-halves through ONE half-tile
    set -- 4 barriers + 2 barrier-separated copy rounds, the second
    half's stores idling behind the first half's drain.  F1EP stores
    rows [32,64) into u2/w2, two new tiles PTX-aliased over **Tb/Tg's
    dead bytes** (both A operands die at ks_pass(1)'s last ldm, which
    its trailing barrier covers): ONE barrier + one merged 4-tile
    copy stream.  Geometry: Tb/Tg join the spec as arena bases; Tg
    grows [65,72]->[67,72] (+288B, ROWS -- the first cut padded COLS
    to 76, making the row stride 152B, not a 16B multiple: ldmatrix
    on odd rows faulted CUDA_ERROR_MISALIGNED_ADDRESS, probe-verified
    and fixed); u2 [34,136] / w2 [35,136] (row pads exist only for
    identify-by-size uniqueness; extents <= their bases; mallocs
    deleted at PTX level, zero pool cost).  Pool: decl 46,432B vs
    ship 46,144B, padded ~46.7KB <= 49,152 -> 2 CTAs/SM kept.  All
    gates green (below).  Perf: **+0.0/+0.6/+0.8us x3 sessions
    T8192, +0.2us T2048, pairs 39-49/80 -- a coin flip.**  Mechanistic
    reading: at 2 CTAs/SM the CO-RESIDENT CTA fills one CTA's barrier
    gaps, so intra-CTA epilogue serialization was never on the
    SM-level critical path; the 31us s2g drain is DRAM-bound (r21
    ncu: DRAM 57%, occupancy 16.2%).  Third data point for the
    co-residency/scheduler law (after r20-b VTF, r21-b ks2):
    barrier-count and issue-order levers do not pay at 2 CTAs/SM;
    only DRAM-traffic-class levers can move f1 now.  GDN_F1EP stays
    DEFAULT OFF.
  * **(b) early pass-1 cp.async into kt's dead bytes: NO-GO on
    arithmetic.**  Pass-1's group moves 16,384B (v+k rows [32,64));
    kt's dead post-kkt arena is 6,144B -> at most 37.5% fits.  An
    ldm-clean split needs 16-row granular pieces ([16,136] = 4,352B):
    only ONE operand's half fits (8,704 > 6,144 for both).  Upper
    bound ~15.1us x 4,352/16,384 = **~4.0us** for a two-source ldm_b
    restructure + a second wait group + a third arena member --
    strictly worse complexity-per-us than (a) which itself measured
    zero.  Not built; closed.
* **The k double-read pool budget table (r21 #3's precondition) closes
  persistent-k**:

| tile (F1KS ship) | bytes | live window |
|---|---|---|
| kt multibuffer (bk16, mb2) | 6,144 | kkt loop; dead post-kkt |
| Abf [64,72] | 9,216 | kkt epi -> solve p1; dead post-solve (vt's arena) |
| Tt [66,72] | 9,504 | solve; dead post-solve (k2's arena) |
| Tb [65,72] | 9,360 | solve -> uw ldm; dead at epilogue (u2's arena) |
| Tg [65,72] | 9,360 | solve -> uw ldm; dead at epilogue (w2's arena) |
| St0/St1/St2 | 768/768/512 | solve rounds R=1..3 |
| vt [32,136] / k2 [33,136] | 8,704/8,976 | uw staging + epilogue restage |
| gcs/bss | 256/256 | scan -> gc_out |
| **decl total** | **46,144** | padded 46,352 <= 49,152 (2 CTA/SM line) |

  A persistent-k tile [64,136]bf16 = **17,408B** must be live
  kkt -> uw, spanning the solve; the ONLY dead bytes in that window
  are kt's 6,144B: **11,264B short**.  Nothing else dies mid-solve
  (St/Tb/Tg/Abf/Tt are solve-live; vt/k2's arenas open only
  post-solve).  Persistent-k stays CLOSED at 2 CTAs/SM; reopening
  needs a >=11.3KB pool restructure (St elimination or solve
  re-geometry), not a knob.

## 1. Adjudication table (the migration, old -> new)

| run | ours | fla | ratio | per-pair | pairs | flags |
|---|---|---|---|---|---|---|
| **B1T8192 formal (r22f, the record)** | 0.7905 | 0.9665 | **1.2226x** | 1.2054x | 27/30, p=8e-06 | cubin be8b914b; WAS 1.2314x/1.2080x 30/30 (r19 s2, e7af55c1) |
| **B4T2048 formal (r22f, the record)** | 0.8481 | 0.9682 | **1.1417x** | 1.1450x | 30/30, p=0.0 | cubin 83898f74; WAS 1.1883x/1.1843x 30/30 (r20, a065a0bc); IQR rel 0.120 -> 0.060, still not decidable (by 0.039) |
| **B1T2048 FORMAL n=30 (r22f, first formal)** | 0.1980 | 0.4194 | **2.1184x** | 2.1113x | 29/30, p=0.0 | cubin 3d47886a; WAS spot-only 1.7842x/1.7902x 12p (r19, ea8ec2f1); IQR rel 0.374 (fla tails), not decidable |

(json: final_adjudicate_r22f_f1zf.json; every pair of sessions agrees
ours-side and disagrees only on fla's window -- see Verdict.)

## 2. Lever numbers

| lever | arm | delta @T8192 f1 | pairs faster | verdict |
|---|---|---|---|---|
| GDN_F1EP s1 | f1ep vs ship(+F1ZF) | **+0.0us** | 42/80 | WASH (bit-identical, knob off) |
| GDN_F1EP s2 | | +0.6us | 49/80 | |
| GDN_F1EP s3 | | +0.8us | 45/80 | |
| GDN_F1EP @T2048 | | +0.2us | 39/80 | consistent |

(cubins: f1ep_T8192 `8f266ab8`, f1ep_T2048 `2b505b60`; jsons
ab_f1ep_* -- scratch/r22f.  A-side = the genrec +F1ZF ship, i.e. a
HEAD rebuild; f2/f3 unaffected -- only gen_f1 changed.)

## 3. Numerics + determinism (green after every step)

* Purity: no-F1ZF ship rebuild == e7af55c1 / a065a0bc / ea8ec2f1
  (pre-edit); post-F1EP-edit knob-off == e7af55c1 (pur2).
* +F1ZF record ships: B4T2048 identity gate x3 -- o/ht max 0.000e+00,
  ndiff 0/17.8M + 0/1.05M, rep-to-rep deterministic
  (b4_check_f1zf.py); fwd_verify 4/4 PASS record digits (staged T2048
  6.486e-4/7.493e-4; fwd_verify_r22f_f1zf.json); the T8192 fp64 gate,
  T2048 7/7 and det N=5 transfer from r21 (same cubin md5s).
* GDN_F1EP gates: T2048 trio 7/7 IDENTICAL vs the +F1ZF ship;
  T8192 fp64 gate BIT-IDENTICAL (o/ht 0.000e+00, PASS digits
  6.626e-4/8.918e-4); fwd_verify 4/4 PASS record digits
  (fwd_verify_r22f_f1ep.json); det N=5 fresh-process
  FWD-DETERMINISTIC (det_f1ep; ship slot restored + md5-verified).
* F1EP probe gotchas (documented for future arena rounds): (1) arena
  tile COL pads must keep row strides 16B-multiples or ldmatrix
  faults on odd rows (MISALIGNED_ADDRESS -- the 152B-stride cut);
  (2) `export "A=1 B=1"` is one var in bash -- the first F1EP build
  silently built without the knob (caught by the md5 equaling the
  F1ZF ship; scripts now echo the knob env).

## 4. Ship provenance

* **The gen_fwd record slots are MIGRATED to the +F1ZF ships** (the
  r21 stale-slot pattern, whole-dir preservation):
  B1T8192_mono e7af55c1 -> `be8b914b`, B4T2048_mono a065a0bc ->
  `83898f74`, B1T2048_mono ea8ec2f1 -> `3d47886a`, and the det slot
  bench_B1T2048H16_mono ea8ec2f1 -> `3d47886a`; every r18 slot kept
  whole at `<name>.r18ship/` (git-tracked, like .stale396b).  The
  slots carry full_spec gdn_cache_key 0d667ecf; after any future
  generator edit the stamps go stale-by-design and rt.build redirects
  to _k<hash> dirs (verified-equivalent rebuild, never a stale
  serve).  The fused slots stay r18-era (different chain, untouched).
* tools/gdn_full_gen.py (R22-fwd's file): **GDN_F1EP** (default off)
  -- the single-pass epilogue + the 4-member F1KS arena spec (u2/w2
  over Tb/Tg) + the Tg [67,72] row-pad.  Correct, bit-identical,
  perf wash; kept for the arena playbook.
* evidence/G5/final_adjudicate_r22f_f1zf.json -- the three n=30
  formals.  scratch/r22f/ (gitignored policy): build/gate/A-B/det
  scripts + jsons + b4_check_f1zf.py + f1_only.py probe.

## 5. B4T2048 decidability -- trimmed-per-pair rule PROPOSAL (r21 #4; methodology sign-off required, no measurement change)

Three B4 windows now (r20 1.1883x / r21 1.2329x / r22f 1.1417x) agree
on ours-side to +/-1% and disagree only through fla's tails (fla max
1.24-2.58ms vs medians 0.97-1.04ms; the co-job's bursts land INSIDE
fla's fatter kernels).  The v1 rule blocks all three sessions because
the IQR -- a tail-sensitive statistic -- measures the CONTAMINATION,
not the noise floor of the uncontaminated pairs.  Proposal (v2,
additive, both always reported):

* **trimmed decidability**: compute the rule's IQR over the INNER
  80% of per-pair ratios (symmetric 10% trim), leaving median,
  sign-test, and pair counts untouched.  A burst inside a member
  contaminates O(1) pairs per session; 10% trim removes exactly that
  class while trimming nothing when the window is clean (v1 == v2
  then).  decidable_v2 = (median-1) > 3 x trimmed_IQR/median AND
  wins >= 25/30 AND p <= 1e-05.
* Computed on this round's raw pairs (no rerun needed; n=30, 10%
  trim = 3 pairs/side): **B4T2048 DECIDABLE-v2** (trimmed med 1.1450,
  tIQR rel 0.0436; effect 0.145 > 3x0.131, wins 30/30, p=0.0) and
  **B1T2048 DECIDABLE-v2** (2.1113, 0.2821; 1.111 > 0.846, 29/30);
  B1T8192 stays undecidable (0.205 < 0.703 -- 27/30 and a genuine
  two-sided window).  CAVEAT, stated up front: v2 measures
  WITHIN-session consistency; it cannot rule on window-to-window
  bias -- B1T2048's center moved 1.78x -> 2.11x between windows on
  fla's median alone (0.35 -> 0.42ms), so its v2 decidability claims
  internal coherence, not a stable 2.1x.  B4's three windows agree on
  ours to +/-1% and disagree only in the tails -- exactly the case
  v2 is for.
* Ask: sign-off on v2 as an ADDITIONAL line in
  adjudication_methodology.md (v1 unchanged for cross-round
  comparability); records claimed under v2 cite both.

## 6. Round-23 list (measured, ranked)

1. **f1's next phase is kkt (59.5us, 26%)**: the uw serialization
   class is closed (F1WY +92 / F1P4 +15 / F1EP ~0 -- co-residency
   absorbs intra-CTA order), the s2g is DRAM-bound, staging is
   pool-blocked.  Split the kkt deletion probe (K-loop vs A-epilogue)
   to localize, then attack the winning half (the exp2/beta epilogue
   math chain is the visible candidate).
2. **B4 decidability sign-off** (section 5) + one n=30 archival
   session under v1+v2.
3. The 4-member arena playbook (u2/w2 over Tb/Tg) is proven and
   reusable -- fold into any future epilogue/store restructure.
4. f2 >=64-col row geometry (r16 #3, untouched, still last).
5. Protocol note: genrec stamps go stale after every generator edit
   (redirect+rebuild by design); refresh slot stamps only when a
   record moves.

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
# ship knob set (unchanged from r21):
#   GDN_F1KS=1 GDN_F2US=1 GDN_WST=1 GDN_ZFIX=1 GDN_SCANDB=1 GDN_F1ZF=1
# the three n=30 formals (record json):
GDN_F1KS=1 GDN_F2US=1 GDN_WST=1 GDN_ZFIX=1 GDN_SCANDB=1 GDN_F1ZF=1 \
    <python> tools/gdn_final_adjudicate.py \
    --shapes B1T8192,B4T2048,B1T2048 --pairs 30 --warps 4 --bv 32 \
    --stats --out <json>     # serves the migrated gen_fwd slots
# F1EP arm (correct, wash; default off): ship knobs + GDN_F1EP=1
# F1EP gates: scratch/r21f/gate_t2048.py <ship_T2048> <f1ep_T2048>;
#   scratch/r14u/r14u_t8192_gate.py --dirs base=<ship>,var=<f1ep>;
#   gdn_full_verify --gen-root <fresh> with +F1EP env;
#   scratch/r18f/run_det.py bench_B1T2048H16 mono \
#     "GDN_F1KS=1,GDN_F2US=1,GDN_WST=1,GDN_ZFIX=1,GDN_SCANDB=1,GDN_F1ZF=1,GDN_F1EP=1" \
#     <out> 5 4 32
# B4 identity gate: scratch/r22f/b4_check_f1zf.py <shipB4> <varB4>
# A/B: scratch/r14u/r14u_ab.py --shape B1T8192 --a <ship> --b <f1ep> \
#     --only f1 --reps 40   (x3 sessions)
# r18 record cubins preserved at gen_fwd/<name>_mono.r18ship/
# knobs: GDN_F1EP (default OFF, perf wash this round), all prior
#   knobs unchanged
```

# G5 round 23 (R23-fwd): the kkt split map -- the stage is 96% K-loop, the K-loop is 73% k-staging, and the staging runs AT the measured device roofline (826 vs 765 GB/s D2D in the same window) -> kkt CLOSED as a perf door; all three r22 lever candidates priced dead (chain ~2-4us total, tile shape breaks 2-CTA occupancy, store pattern sub-2us); B4 decidability v2 documented-optional with verified verdicts

Date: 2026-10-08 | shared/opportunistic window (co-job ~8.7GB resident
throughout, 0-54% util bursts, never touched; window snapshots in every
json; coordination row `gdn-mlir R23-fwd` in
~/codebase/GPU-COORDINATION.md).  HEAD at round start: 5b70903.
Knob-off purity: the ship-knob rebuild (NO F1ZF) reproduced the r18
record md5 `e7af55c1` from the edited tree TWICE (before and after the
NKM/emit_kloop edit) -- the probe machinery is inert when off; the
+F1ZF ship rebuild reproduced `be8b914b` (the r21/r22 T8192 record
cubin) from the edited tree.

## Verdict

* **Mission #1, the kkt deletion-probe split: DONE, and it kills the
  r22 lever list by measurement.**  Three new probes extend the
  GDN_F1T_* set (NKK = delete the K-loop keep the epilogue; NEK =
  delete the exp2/beta chain keep the bare store; NKM = delete
  ldm+mma keep the staging loop), each A/B'd x3 sessions vs the
  rebuilt +F1ZF ship on T8192.  The r21 map's single kkt lump
  (59.5us) splits as:

| kkt component (probe that isolates it) | us | share |
|---|---|---|
| k-tile global staging, cp.async stream of 33.55MB (NKK - NKM) | **40.6** | 73% |
| ldmatrix + chained mma (NKM: -13.0) | 13.0 | 23% |
| A-epilogue exp2/beta chain: 3 LDS + subf/ex2/2xmul/select (NEK: -3.7) | 2-4 | 4-7% |
| A-epilogue stores to Abf (residual NKT-NKK) | <=2 | <=4% |
| **total kkt (NKT: -55.4 this window; r21 said 59.5)** | **55.4** | 100% |

  The r22 hypothesis "the exp2/beta chain is the visible target" is
  DISPROVEN -- the ENTIRE A epilogue (chain + stores, the thing the
  fold/store-pattern levers would restructure) is 2-4us of a 55us
  stage.  The stage is 96% K-loop, and the K-loop is 73% the k-read
  stream.
* **The staging is at the memory roofline, measured two ways.**  The
  stream moves B*Tp*H*K*2 = 33.55MB (each CTA reads its own 64x128
  bf16 k chunk exactly once -- 100% sector utilization, 32B-aligned
  rows, bypass_l1) in 40.6us = **826 GB/s**; a D2D copy calibration
  run in the SAME co-loaded window measured the device at **765 GB/s
  aggregate (read+write)** (scratch/r23f/bw.py).  A read-only stream
  beating the r+w aggregate means there is no headroom to restructure
  into: the kkt staging is DRAM-bound at this shape.  kkt is CLOSED
  as a perf door at the current geometry (2 CTAs/SM, 33.55MB
  algorithmic-minimum k traffic).
* **Every r22 lever candidate priced (none builds a win):**
  1. *exp2/beta fold into the mma epilogue*: the chain IS already in
     the mma epilogue; the probe says there is <=4us of anything to
     save.  DEAD (measured).
  2. *K-loop tile shape* (bk 16->32, pre-existing GDN_F1KS_BK32 arm):
     kt grows 6,144 -> 10,240B (mb2), f1 decl ~49.7KB, and the
     occupancy API reads **blocks/SM 2 -> 1** (scratch/r23f/occ.py);
     paired x3 = **+70.3/+73.4/+68.3us REGR** -- the whole gain class
     is the occupancy loss, not tile economics.  DEAD (pool).  bk32
     at mb1 (pool-neutral 5,120B) would serialize staging<->compute
     in a loop whose cost is 73% staging -- priced NO-GO on paper.
  3. *A-epilogue store pattern*: stores are <=2us of the stage.
     DEAD (measured).
  4. (this round's own candidate) *deeper pipeline* (--mb 3 --depth
     3): the transform hard-fails ("pipelining preconditions failed")
     on the uw loop -- a 2-trip loop cannot pipeline at depth 3, and
     the knob is global (no per-loop depth).  DEAD (transform).
     mb3+depth2 changes nothing (in-flight = depth).
  5. *prologue prefetch of the first staged k tiles into the scan
     window* (the only overlap left: the 17.6us Hillis-Steele scan is
     compute/latency-bound while DRAM idles): priced, NOT built.
     Ceiling = mb2 in-flight = 2 of 8 iterations = ~10us; the
     implementation is transform-hostile on two independent axes
     (a pre-loop user of kt makes transform.memref.multibuffer SKIP
     the alloc -> the remaining loop loses double-buffering, so the
     prefetch needs a NEW tile kt0 +3,072B -> decl 49,216B, an
     occupancy coin-flip in the (46.7KB ok, 50.1KB broken) gap this
     round just mapped; and the async-group WAIT placement for a
     straight-line consumer outside the pipelined-loop pattern is
     unproven in this custom pipeline).  ~10us ceiling for a
     high-corruption-risk restructure: NO-GO at these odds; revisit
     only if a round needs <=5% of f1 and has nothing better.
* **f1 is now closed-end-to-end at this geometry**: uw serialization
  closed r22 (F1EP wash), s2g DRAM-bound, staging pool-blocked
  (persistent-k table), kkt roofline-closed (this round); the
  residuals are the solve's serial phase-1 (32.5us; F1WY +92 /
  F1P4 +15 honest negatives) and the scan (17.6us).  The fwd perf
  frontier moves to f2/f3.
* **Mission #2, B4T2048 trimmed-per-pair decidability v2: applied as
  DOCUMENTED-OPTIONAL, verdicts computed, memo below.**  The rule is
  now a section in adjudication_methodology.md (v1 unchanged as the
  primary; any v2 claim cites both), and the verdicts were recomputed
  from the r22 raw arrays (scratch/r23f/v2_verdicts.py ->
  evidence/G5/decidability_v2_r23f.json), reproducing the r22 §5
  numbers exactly: **B4T2048 DECIDABLE-v2** (med 1.1450, tIQR rel
  0.044, effect 0.145 > 3x0.131, 30/30, p=9.3e-10 -- v1 blocks it by
  0.039), **B1T2048 DECIDABLE-v2** (2.1113, 0.282, 1.111 > 0.846,
  29/30), B1T8192 no under both (0.205 < 0.703, genuine two-sided
  window).  No measurement changed anywhere.
* f2 >=64-col row geometry (r16 #3): untouched again (explicitly the
  "if time" item; the kkt split + arms consumed the round).  Carried.

## 1. The kkt split table (f1 @B1T8192, paired interleaved A/B x3)

| probe (ship+F1ZF+one GDN_F1T_*) | deleted | deltas us (s1/s2/s3) | median |
|---|---|---|---|
| NKT (r21, re-run) | whole kkt stage | -59.3 / -55.2 / -55.4 | **-55.4** |
| NKK (new) | K-loop (staging+ldm+mma), epilogue on zero accs | -53.2 / -53.6 / -55.2 | **-53.6** |
| NEK (new) | exp2/beta chain only, bare store keeps accs+Abf live | -4.7 / -3.4 / -3.7 | **-3.7** |
| NKM (new) | ldm+mma only, staging loop + barriers kept | -9.4 / -13.0 / -13.6 | **-13.0** |

(A-side = scratch/r23f/ship2_T8192 = be8b914b rebuilt from the edited
tree; jsons ab_probe_*_T8192_s{1,2,3}.json.  Derived: staging =
NKK-NKM = 40.6; epilogue = NKT-NKK = 1.8 (vs NEK's direct 3.7 --
deletion probes are non-additive at the ~2us scale; the class reading
"epilogue <= 4us" is safe under either).

## 2. Lever arms priced this round

| arm | knobs | result | verdict |
|---|---|---|---|
| tile shape | GDN_F1KS_BK32=1 | +70.3/+73.4/+68.3us x3; blocks/SM 2->1 (cuOccupancy API); kt 6,144->10,240B | DEAD (occupancy) |
| deeper pipeline | --mb 3 --depth 3 | compile error: pipelining preconditions failed (uw 2-trip loop at depth 3); global knob | DEAD (transform) |
| staging roofline | (calibration) | stream 826 GB/s vs D2D 765 GB/s r+w, same window | AT THE WALL |
| prologue prefetch | (paper) | <=10us ceiling (mb2 = 2/8 iters into the 17.6us scan window); multibuffer-skip + wait-placement + 49.2KB occupancy coin-flip | NO-GO, priced |

## 3. Numerics + determinism (green after every step)

* Purity x2: no-F1ZF ship rebuild from the edited tree == e7af55c1
  (pre-NKM-edit pur_, post-edit pur2_); +F1ZF ship == be8b914b.
* fwd_verify 4/4 PASS on a FRESH gen root under the +F1ZF ship env
  from the edited generator (staged T2048 o 6.486e-4 / ht 7.493e-4 =
  the record digits; fwd_verify_r23f.json).
* The GDN_F1T_NKK/NEK/NKM probes are timing-only deletion probes
  (outputs GARBAGE by design; never verified -- the r21 NKT
  precedent).  No ship-path byte changed this round; the +F1ZF
  record ships and their r22 gates (T8192 fp64 BIT-IDENTICAL, T2048
  7/7, B4 identity x3, det N=5) transfer on the same cubin md5s.
* Occupancy spot-check of every probe build: f1 2 / f2 1 / f3 1
  blocks/SM, same as the ship (the probe pool stays at or under the
  ship's; NKK/NKM shrink kt's multibuffer).

## 4. B4T2048 decidability v2 -- the decision memo (for the user)

DECISION MEMO -- trimmed-per-pair decidability v2.  WHAT CHANGES: an
ADDITIONAL adjudication line (decidable_v2) computes the existing
IQR-spread bar over the inner 80% of per-pair ratios (10% symmetric
trim; n=30 drops the 3 highest and 3 lowest) instead of all 30;
median, sign test, and win counts are untouched, v1 stays the rule of
record for cross-round comparability, and any v2-claimed record cites
both lines.  WHAT IT WOULD FLIP: on the r22 formal pairs, B4T2048
(30/30 wins, p=9.3e-10, blocked under v1 by 0.039) and B1T2048
(29/30) become DECIDABLE-v2; B1T8192 stays undecidable under both.
THE RISK: v2 certifies within-session coherence, not window-to-window
stability -- B1T2048's center moved 1.78x -> 2.11x across windows on
fla's median alone (0.35 -> 0.42ms), so its v2 verdict is weaker than
it looks; B4 (three windows agreeing on ours +/-1%, differing only in
fla's tails) is exactly the case the rule was designed for.
RECOMMENDATION: sign off v2 as a documented-optional side statistic;
claim B4T2048 1.1417x under "v1-no / v2-yes" dual citation at the next
archival n=30 session; keep B1T2048 v2-soft until its center
stabilizes across >=2 more windows.

## 5. Round-24 list (measured, ranked)

1. **f2 >=64-col row geometry** (r16 #3, carried): the fwd frontier
   after f1's end-to-end closure (this round).
2. **B4 v2 sign-off decision + one n=30 archival session under v1+v2
   dual citation** (the memo above; the r22 item, now unblocked).
3. f1's residual non-DRAM time (solve phase-1 serial 32.5us + scan
   17.6us) has two honest-negative attack rounds (F1WY/F1P4; SCANDB
   hardened) -- only revisit with a NEW mechanism class, not another
   warp-mapping.
4. Protocol: the gen_fwd record slots carry cache_key 0d667ecf; this
   round's generator edit moves the key to 780ab7d2 -> the stamps are
   stale-by-design and rt.build will redirect to _k-prefixed fresh
   dirs (verified-equivalent rebuild, never a stale serve; refresh
   slot stamps only if a record ever moves).

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
# ship knob set (unchanged): GDN_F1KS=1 GDN_F2US=1 GDN_WST=1
#   GDN_ZFIX=1 GDN_SCANDB=1 (+ GDN_F1ZF=1 for the record ships)
# the kkt split probes (timing-only, garbage outputs, x3 sessions):
#   A-side: ship+F1ZF rebuild == be8b914b
#   GDN_F1T_NKK=1 (K-loop deleted)  GDN_F1T_NEK=1 (chain deleted)
#   GDN_F1T_NKM=1 (ldm+mma deleted) GDN_F1T_NKT=1 (r21, whole stage)
# build: gdn_full_gen.py --path mono --shape 1,8192,16,128,128 \
#   --warps 4 --bv 32 --bv3 128 + knobs; scripts/run_l2.sh;
#   GDN_WST=1 gdn_smem_fix.py; then
#   scratch/r14u/r14u_ab.py --shape B1T8192 --a <ship> --b <probe> \
#       --only f1 --reps 40   (x3)
# tile-shape arm: ship knobs + GDN_F1KS_BK32=1 (occupancy 2->1, REGR)
# v2 verdicts: scratch/r23f/v2_verdicts.py (from
#   final_adjudicate_r22f_f1zf.json raw arrays) ->
#   evidence/G5/decidability_v2_r23f.json
# knobs: GDN_F1T_NKK/NEK/NKM (new, default OFF, timing probes only),
#   all prior knobs unchanged; nothing shipped this round
```


# G5 round 24 (R24-fwd): the f2 >=64-col row geometry CLOSED -- option (a) dead on paper (the r14-measured 115,200B pool law), option (b) the GDN_W32P conflict-free per-row warp remap BUILT, bit-identical, f2 -18..-26us x15 sessions, chain-positive at all three shapes; + the bwd W-park decision memo (R24-bwd, CPU)

Date: 2026-10-08 | shared/opportunistic window (co-job ~8.9GB resident
throughout, never touched; interleaved paired ratios only; coordination
row  in ~/codebase/GPU-COORDINATION.md).  HEAD at round
start: de779b5.  Knob-off purity: the ship-knob rebuilds from the
edited tree reproduce ALL THREE r22 formal-record cubins -- T8192
, B4T2048 , B1T2048  (the generator edit
is inert with the knob off, at every shape).

## Verdict

* **The r16 #3 frontier item is resolved.**  The two routes the r16
  negative left open were priced on paper first, per the mission:
  1. *BV=64 re-tile (option a): DEAD on arithmetic, no build.*  With
     F2US the pool is 115,200B > the 101,376B per-block optin -- not a
     projection: r14 measured EXACTLY this number fail ptxas at
     w8/bv64 (the ut tile doubles to [2C,72]).  Dropping F2US to fit
     (96,768B, 1 CTA/SM) surrenders the measured F2US win (f2 -104.5us
     T8192 / -101.2 B4, 0/60 slower, r14) AND halves the f2 grid (VT
     4->2: 64->32 CTAs on 82 SMs at T8192/B1T2048 = 39% SM coverage
     with an unchanged 1 CTA/SM residency) -- and r3s


# G5 round 24 (R24-fwd): the f2 >=64-col row geometry CLOSED -- option (a) dead on paper (the r14-measured 115,200B pool law), option (b) the GDN_W32P conflict-free per-row warp remap BUILT, bit-identical, f2 -18..-26us across 17 sessions, chain-positive at all three shapes; + the bwd W-park decision memo (R24-bwd, CPU)

Date: 2026-10-08 | shared/opportunistic window (co-job ~8.9GB resident
throughout, never touched; interleaved paired ratios only; coordination
row `gdn-mlir R24` in ~/codebase/GPU-COORDINATION.md).  HEAD at round
start: de779b5.  Knob-off purity: the ship-knob rebuilds from the
edited tree reproduce ALL THREE r22 formal-record cubins -- T8192
`be8b914b`, B4T2048 `83898f74`, B1T2048 `3d47886a` (the generator edit
is inert with the knob off, at every shape).

## Verdict

* **The r16 #3 frontier item is resolved.**  The two routes the r16
  negative left open were priced on paper first, per the mission:
  1. *BV=64 re-tile (option a): DEAD on arithmetic, no build.*  With
     F2US the pool is 115,200B > the 101,376B per-block optin -- not a
     projection: r14 measured EXACTLY this number fail ptxas at
     w8/bv64 (the ut tile doubles to [2C,72]).  Dropping F2US to fit
     (96,768B, 1 CTA/SM) surrenders the measured F2US win (f2 -104.5us
     T8192 / -101.2 B4, 0/60 slower, r14) AND halves the f2 grid (VT
     4->2: 64->32 CTAs on 82 SMs at T8192/B1T2048 = 39% SM coverage
     with an unchanged 1 CTA/SM residency) -- and r3's BV=64 wins were
     on fwd-F2-serial, not this pp kernel.  NO-GO.
  2. *Per-row warp remap (option b): the survivor, SHIPPED as
     GDN_W32P.*  The r16 bank-conflict law: a BV=32 tile row is
     (BV+8)=40 elems = 80B = 20 words stride; under the flat WST pair
     map a warp spans 2 CONSECUTIVE rows and the 20-word stride banks
     lanes 28-31 onto lanes 0-3 (2-way conflict, the r16 +4.3us
     mapping-only measurement).  The W32P remap: lanes 0-15 read row
     8*round + warp, lanes 16-31 read row r+4 -- word addresses
     [20r, 20r+16) and [20r+80, 20r+96); 80 mod 32 = 16, so the two
     half-warps hit the two COMPLEMENTARY 16-bank halves: zero
     conflicts for every r.  Bijective (round j, warp w, hi h -> row
     8j+w+4h covers rows 8j..8j+7, pair lane%16), pure index
     arithmetic, value-identical: pool, tiles, LDSM pads, occupancy
     all unchanged; f2's smem_fix wst count 0 -> 2 (the two store-pass
     loop bodies fuse to ld.shared.b32/st.global.b32 exactly like
     f1/f3's cols>=64 passes).
* **f2 kernel win, paired interleaved A/B vs the purity builds
  (flipped two-load harness, reps 40, 17 sessions across both
  emissions):**

| shape | f2 delta (sessions) | slower-pair rate | staged e2e delta |
|---|---|---|---|
| B1T8192 | **-17.9/-18.3/-18.8/-19.3/-21.5/-21.8/-22.6 us** (x7) | 5-8/80 | **-9.1..-12.0 us** (x5) |
| B4T2048 | -9.7/-10.0/-10.1/-13.2/-25.6 us (x5) | 1-9/80 | **-14.3 / -27.5 us** |
| B1T2048 | -2.0/-2.1/-2.1/-3.2/-3.8 us (x5) | 31-39/80 | **-3.2 / -5.2 us** |

  ~6-8% of f2 at T8192, right in the paper band (the r15 deletion
  probes put the two passes at 53.3us; WST on f1 paid 0.176us/MB ->
  ~17us ceiling on f2's 100.6MB of pass writes).
* **The T8192 chain tax, characterized**: the arm cubins draw a
  REPRODUCIBLE f3 +11.8..+12.6us at T8192 (stable across sessions,
  same sign in BOTH flip halves, on BOTH arm builds -- f3/f1 PTX
  sections are byte-IDENTICAL to purity; it is the module-layout
  lottery of the r19 calibration, f3 +-10.5+-6us, triggered by f2's
  +~300B PTX).  B4/T2048 draws are noise-range (both signs).  Net
  chain: positive at every shape, every session.  The next archival
  n=30 re-rolls this layout; the f2 kernel win is the stable
  quantity.
* **Gates (final emission, all green)**: T8192 fp64 gate PASS at
  record digits (o 6.626e-4 / ht 8.918e-4) and **BIT-IDENTICAL**
  vs the purity build; fwd_verify 4/4 PASS on a fresh gen root at the
  record digits (T2048 staged o 6.486e-4 / ht 7.493e-4;
  fwd_verify_r24f2.json); det N=5 FWD-DETERMINISTIC (ship slot
  restored, md5 3d47886a); occupancy unchanged (f1 2 / f2 1 / f3 1
  blocks/SM at all three shapes).  Arm cubins (final emission):
  T8192 e7c8eb4c, B4 39f094c3, T2048 be5b2377.
* **Ship status**: GDN_W32P joins the ship knob set (default 0 in
  code; the documented ship env gains GDN_W32P=1) -- the F1ZF
  precedent: records stay r22-pinned until the next archival n=30
  session re-adjudicates on the new cubins.
* **R24-bwd (the second mission half, CPU-only)**: the P11R W-park
  decision memo is delivered at evidence/G5/bwd_wpark_decision_memo.md
  (raw: wpark_twin_T{2048,8192}.json, wpark_arith_out.txt).  Headline:
  numerically ADMISSIBLE (the f32 re-association is ~1e-9 on graded
  outputs; the bf16 partial-park adds <=9.4e-5 on dk -- 65-1025x under
  every bar, no tolerance-relaxation sign-off needed) but PERF-DEAD:
  smem park >=16,384B vs 2,560B headroom (pool law; the 1-CTA fallback
  costs the measured 1.46x 2-CTA conversion), global park nets
  +147KB/chunk f32 (+49KB bf16) against the 49KB the full staging kill
  saves (0.83-0.98x REGR), and the FREE-park ceiling is 1.02x on a
  19-24% stage = -0.5% e2e, under every bar this project ships at.
  NO-GO, decision-grade; the r22 new-numerics class closes.

## 1. The option-(a) pool table (paper, no build)

| tile | bv32 (ship) | bv64 | note |
|---|---|---|---|
| S [K,BV+8] bf16 | 10,240 | 18,432 | |
| ut (F2US) [2C,BV+8] | 10,240 | 18,432 | the killer |
| vn [C,BV+8] | 5,120 | 9,216 | |
| wb / kb / gcs | 33,792/34,816/512 | same | K-tiles, BV-free |
| **pool** | **94,720 (1 CTA/SM)** | **115,200** | **> 101,376 optin = the r14-measured ptxas fail** |
| without F2US | 84,480 | 96,768 | fits 1 CTA/SM; loses -104.5us (r14) + halves the grid |

Device: 82 SMs, 101,376B/block optin, 102,400B/SM (queried this
round).  f2 grids: 64 (T8192/T2048), 256 (B4) CTAs.

## 2. The W32P remap (what the knob does)

copy_s2g_vec gains a third mode (GDN_W32P=1 and cols==32 and rows%8==0
and nthr==128 -- f2's S/vn tiles only; f1's passes are cols=V/K=128 and
f3's are cols=128, both already WST-paired): per pair-step i (the
while-loop cell counter), row = 8*(i>>7) + warp + 4*(lane>=16), pair
col = 2*(lane%16) -- the lane constants reuse %lane/%warp and CSE out
of the loop.  The two b16 loads stay [base]/[base+2] and the two
stores [g]/[g+2], so gdn_smem_fix widen_stores fuses them unchanged
(structural pattern).  The first emission computed the lane constants
from %tid (6 lines/pass); the final emission reuses %lane/%warp (4
lines/pass) -- both value-identical, both measured, the leaner one
ships.

## 3. Numerics + determinism (green after every step)

* Purity x3 shapes: no-W32P rebuilds == the r22 record cubins
  (be8b914b / 83898f74 / 3d47886a), pre- and post-final-emission.
* T8192 gate: variant vs purity BIT-IDENTICAL (o/ht 0.000e+00), fp64
  PASS at record digits; the gate ran on BOTH emissions.
* fwd_verify 4/4 on a FRESH gen root under ship+W32P (the edited
  generator end-to-end): staged T2048 o 6.486e-4 / ht 7.493e-4 = the
  record digits; fused/small also PASS.
* det N=5 fresh-process: FWD-DETERMINISTIC; the gen_fwd ship slot
  restored and md5-verified (3d47886a), variant preserved at
  scratch/r24f/det_r24f2_variant (be5b2377).
* Occupancy probe on every final arm: f1 2 / f2 1 / f3 1 blocks/SM.

## 4. Round-25 list (measured, ranked)

1. **B4 v1+v2 sign-off + the archival n=30 session** (the r23 #2,
   still the gate): if it runs, adjudicate the W32P ship on fresh
   formals (ship env + GDN_W32P=1, new record cubins
   e7c8eb4c/39f094c3/be5b2377) -- the T8192 f3 layout tax re-rolls
   with the new build; the f2 kernel win is the stable base.
2. f1's residual non-DRAM time (solve p1 32.5us serial + scan 17.6us):
   unchanged from r23 -- new mechanism classes only.
3. f2's next class, if ever reopened: the passes now run at half the
   LSU instructions; the remaining pass cost is DRAM-write-bound
   (100.6MB at T8192) -- no value-identical lever left there either.
4. Protocol: the gen_fwd slot stamps remain stale-by-design
   (cache_key 780ab7d2 after the r23 edit; this round's generator edit
   moves it again -- rt.build redirects, never stale-serves).

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
# ship knob set + the new member: GDN_F1KS=1 GDN_F2US=1 GDN_WST=1
#   GDN_ZFIX=1 GDN_SCANDB=1 GDN_F1ZF=1 GDN_W32P=1
# builds (per shape): scratch/r24f/{build_r24.sh,build_r24b.sh,
#   finalize_r24.sh}; purity == the r22 record md5s at every shape
# gates: scratch/r14u/r14u_t8192_gate.py --dirs base=<pur>,w32p=<arm>;
#   gdn_full_verify --gen-root <fresh> with +W32P (fwd_verify_r24f2.json);
#   scratch/r18f/run_det.py bench_B1T2048H16 mono "<ship,W32P>" <out> 5 4 32
# A/B: scratch/r14u/r14u_ab.py --shape {B1T8192,B4T2048,B1T2048}
#   --a <pur> --b <arm> --only f2 --reps 40   (x3; full-trio jsons
#   ab_w32p{,2}_full_*.json carry the staged lines)
# occupancy: scratch/r23f/occ.py <arm dirs>
# the bwd memo: evidence/G5/bwd_wpark_decision_memo.md;
#   wpark_twin.py / wpark_arith.py (CPU-only) + their JSONs at
#   evidence/G5/
# knobs: GDN_W32P (new, default OFF; =1 in the documented ship env),
#   all prior knobs unchanged
```

# G5 round 25 (R25-fwd): the W32P record migration -- the archival n=30 lands on the +W32P ships, all three formals move up (T8192 1.349x / B4 1.221x / T2048 2.125x); the two held sign-offs adopted per recommendation and recorded; bwd frozen at the r19-final bytes

Date: 2026-10-08 | shared/opportunistic window (co-job ~8.9GB resident
throughout, 25% util at session start, never touched; interleaved
ratios only; coordination row `gdn-mlir R25` in
~/codebase/GPU-COORDINATION.md).  HEAD at round start: 1c7d291.

## Decisions first (the round's gate items)

* **D1 -- B4T2048 decidability v2: SIGNED as DOCUMENTED-OPTIONAL side
  statistic**, and **D2 -- bwd final direction: HOLD the r19-final
  bytes with the dual-exhaustion proof as the paper narrative.**  Both
  were asked in-session at the R23/R24 close with no response; per the
  autonomous-goal rules the RECOMMENDED defaults are adopted and
  documented as ADOPTED-PER-RECOMMENDATION (user non-response; override
  welcome) in `evidence/G5/DECISIONS_R25.md`, with the verbatim memo
  rationale quoted there (fwd_report round-23 section 4;
  bwd_wpark_decision_memo.md section 6).  D1 is citation-policy only,
  D2 is a scope freeze only -- neither disturbs a measurement.

## The archival session

* **Build**: the +W32P ship cubins rebuilt FRESH from HEAD 1c7d291
  (ship env `GDN_F1KS=1 GDN_F2US=1 GDN_WST=1 GDN_ZFIX=1 GDN_SCANDB=1
  GDN_F1ZF=1 GDN_W32P=1`, gen + run_l2 + smem_fix, cache keys stamped
  7b41c5f9) into the fresh record gen-root `scratch/r25f/genrec`.
  **md5 3/3 EXACT vs the r24 final-emission candidates** -- T8192
  `e7c8eb4c`, B4 `39f094c3`, T2048 `be5b2377`.  Notable: the T8192 f3
  module-layout lottery did NOT re-roll (unchanged tree + env = byte-
  identical module); the r24 tax draw carries into the record
  generation -- and the build's bit-reproducibility is itself a
  positive toolchain datum (the layout lottery fires on generator
  edits, not on rebuilds).
* **Numerics**: fwd_verify 4/4 PASS on a fresh gen root under the
  +W32P ship env at the RECORD DIGITS (staged T2048 o 6.486e-4 /
  ht 7.493e-4; `scratch/r25f/fwd_verify_r25f.json`).  The r24 gate
  battery transfers on the identical md5s (T8192 fp64 gate
  BIT-IDENTICAL at record digits, det N=5 FWD-DETERMINISTIC -- the
  r24 det ran on the be5b2377 variant, byte-identical to this ship).
* **The formal n=30 --stats, interleaved, all three shapes on the one
  cubin generation** (`tools/gdn_final_adjudicate.py --pairs 30
  --warps 4 --bv 32 --stats --gen-root scratch/r25f/genrec`;
  `final_adjudicate_r25f_w32p.json`; window 8941MiB/25% co-load):

| shape | ours med (r22 -> r25) | fla med (r22 -> r25) | record old -> NEW | per-pair med | wins, p | IQR rel v1 | tIQR rel v2 | v1 / v2 (fresh pairs) |
|---|---|---|---|---|---|---|---|---|
| B1T8192 | 0.7905 -> 0.7851 ms | 0.9665 -> 1.0593 ms | 1.2226x -> **1.3493x** | 1.3307x | 29/30, p=2.9e-08 | 0.178 | 0.119 | no / no (0.331 < 0.357 bar) |
| B4T2048 | 0.8481 -> 0.8213 ms | 0.9682 -> 1.0029 ms | 1.1417x -> **1.2211x** | 1.2029x | 25/30, p=1.6e-04 | 0.273 | 0.168 | no / no (this window) |
| B1T2048 | 0.1980 -> 0.1866 ms | 0.4194 -> 0.3964 ms | 2.1184x -> **2.1250x** | 2.1174x | 30/30, p=9.3e-10 | 0.420 | 0.347 | no / **YES** |

  All three windows internally clean (|ratio-of-medians - per-pair|
  < 0.15).  Fresh-pair verdicts: `scratch/r25f/v2_verdicts_r25.py` ->
  `evidence/G5/decidability_v2_r25f.json`.

## Honest readings

* **The ours-side medians improved at every shape** -- 0.7905 -> 0.7851
  (T8192), 0.8481 -> 0.8213 (B4, -26.8us = squarely inside r24's
  paired B4 staged chain band -14.3..-27.5us), 0.1980 -> 0.1866
  (T2048).  The W32P chain win is visible in the ours side exactly
  where r24 measured it, on the formal instrument.
* **The ratio-of-medians gains are partly fla-side window swing**:
  this was a heavier co-load window than r22's (fla medians moved
  0.9665 -> 1.0593 at T8192, 0.9682 -> 1.0029 at B4).  The per-pair
  interleaved ratios (the robust line) moved less: 1.2054 -> 1.3307 /
  1.1450 -> 1.2029 / 2.1113 -> 2.1174.  Both lines are recorded.
* **B4T2048 decidability, dual citation per D1**: the v1-no / v2-yes
  claim stands on the r22 pairs (`decidability_v2_r23f.json`, blocked
  by 0.039 under v1); the FRESH pairs draw a noisier window -- v2-no
  with two failed conditions (tIQR rel 0.168 -> bar 0.504 vs effect
  0.203; and 25/30 wins with p=1.6e-04 above the 1e-05 line).  The
  moved number 1.2211x therefore carries BOTH verdicts of its own
  session (v1-no / v2-no, noisier window) while the D1 dual-citation
  claim remains anchored to the r22 pairs -- stated, not smoothed.
* **B1T2048's v2 verdict strengthens**: DECIDABLE-v2 this window
  (30/30, p=9.3e-10, tIQR rel 0.347 < effect/3 = 0.372) AND the
  center is now stable across two formal windows (2.1113 -> 2.1174,
  0.3% apart) -- the r23 memo wanted ">=2 more windows" before
  un-softening; this is the first of them, recorded as such.
* **Slot migration**: the gen_fwd record slots moved to the +W32P
  ships (`B1T8192_mono` e7c8eb4c, `B4T2048_mono` 39f094c3,
  `B1T2048_mono` + `bench_B1T2048H16_mono` be5b2377), the r22 bytes
  preserved whole at `<name>.r22ship` siblings, stamps refreshed
  (7b41c5f9; stale-by-design at the next generator edit -- rt.build
  redirects, never stale-serves).  Pattern = the r22 migration.

## Round-26 list

None scheduled.  At these shapes the fwd frontier is closed the same
way bwd is: f1 end-to-end (r23: staging at the 826GB/s measured
roofline = algorithmic-minimum traffic), f2 write-path (r24: W32P was
the last value-identical lever; the passes are DRAM-write-bound), f3
bandwidth-bound (r20).  Reopen triggers: a NEW mechanism class, or a
regime change (B4+/T-other/H-other shapes) -- the same scope D2 names
for bwd.  Paper-side: the R7-R25 longscroll is mapped into the
evaluation chapter skeleton (paper/outline.md "G5 post-G7 addendum").

## Reproduce

```bash
# conda env gdn; ~/codebase/gdn-mlir
# decisions: evidence/G5/DECISIONS_R25.md (D1 citation policy, D2 bwd freeze)
# build+stamp the fresh genrec (3 shapes, ship env + GDN_W32P=1):
#   bash evidence/G5/scratch/r25f/build_r25.sh   (md5s vs r24 candidates)
# verify 4/4 (fresh root, ship+W32P env):
#   bash evidence/G5/scratch/r25f/run_gates_r25.sh
# THE formal n=30 --stats x3 shapes (interleaved, co-loaded window):
#   bash evidence/G5/scratch/r25f/run_formal_r25.sh
#   -> evidence/G5/final_adjudicate_r25f_w32p.json
# dual v1/v2 verdicts on the fresh pairs:
#   python evidence/G5/scratch/r25f/v2_verdicts_r25.py
#   -> evidence/G5/decidability_v2_r25f.json
# record-slot migration (r22 bytes preserved at .r22ship):
#   bash evidence/G5/scratch/r25f/migrate_slots_r25.sh
```
