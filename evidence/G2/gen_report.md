# G2-A: L2/L3 transform knob library + sm_120a PTX matrix — gen report

Workstream G2-A (GDN-MLIR). Deliverables:
- `tools/gdn_l2_gen.py` — the knob library: emits `payload.mlir` + `transform.mlir`
  (+ `config.json`) per knob config. `--list-configs` prints the built-in matrix.
- `tools/gdn_extract_ptx.py` — PTX extraction from `gpu.binary` assembly attrs.
- `tools/gdn_l2_batch.py` — matrix driver; emits `evidence/G2/gen/INDEX.json`.
- `scripts/run_l2.sh <config-dir>` — one-config compile chain:
  `mlir-opt -transform-interpreter` → (`-convert-linalg-to-loops` fallback) →
  `-test-transform-dialect-erase-schedule -gpu-lower-to-nvvm-pipeline=
  "cubin-chip=sm_120a cubin-features=+ptx87 cubin-format=assembly"` →
  PTX extract → `ptxas-blackwell -v -arch=sm_120a` → `nvdisasm -c` →
  `{kernel.ptx, kernel.cubin, ptxas_v.txt, sass.txt, meta.json}`.
- `evidence/G2/gen/<config>/` × 17 + `evidence/G2/gen/INDEX.json` (17/17 ok).
- `evidence/G2/scratch/` — pipeline-lock evidence (sm80 smoke, minimal bf16-mma
  and async-pipe probes, run scripts).

**CPU only**: mlir-opt / ptxas-blackwell / nvdisasm; mlir-runner (GPU) never invoked.

## PTX extraction method (chosen, with fallback chain explored)

Primary (used): `-gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a cubin-features=+ptx87
cubin-format=assembly"` attaches the PTX text as `#gpu.object<..., assembly = "...">`
on `gpu.binary`; `tools/gdn_extract_ptx.py` unescapes the MLIR string (\\XX hex
escapes) to `kernel.ptx`. Emitted PTX is `.version 8.7 / .target sm_120a`.

Fallbacks explored:
- `cubin-format=ptx` — rejected by this build ("Invalid format"; valid:
  `offloading, llvm, assembly, isa, binary, bin, fatbinary, fatbin`).
- Default fatbin → `cuobjdump -ptx`: works (fatbin embedded in gpu.binary) but
  needs binary extraction from the .mlir attr; assembly format is strictly simpler.
- `-gpu-module-to-binary="format=..."` standalone pass also accepts `assembly`.

SASS: own `ptxas-blackwell -v -arch=sm_120a` (triton 13.3.33; /usr/local/cuda/bin
ptxas is 13.3.73 — both accept sm_120a + PTX 8.7; triton's plain `ptxas` is 12.9
and CANNOT) then `nvdisasm -c` (triton bin or CUDA 13.3).

## The matrix (17 configs, all compiled ok)

mma = PTX `mma.sync` count (== SASS HMMA count in every config);
cp = PTX `cp.async*` count (SASS LDGSTS + commit/wait lines);
regs/stack/spill from `ptxas -v`; LDS/STS = SASS shared-window ops;
ex2 = `ex2.approx.f32` count; PTX KB = kernel.ptx size.

| config | regs | stack B | spill B | mma | cp.async | LDGSTS | LDS | ex2 | PTX KB |
|---|---|---|---|---|---|---|---|---|---|
| kkt_bf16_mma_p0_t64x64x32_w4 | 255 | 4544 | 9712 | 16 | 514(ca) | 512 | 83 | 32 | 125.7 |
| kkt_bf16_mma_p2_t64x64x32_w4 | 255 | 3808 | 12628 | 48 | 1542(cg) | 1536 | 246 | 32 | 276.3 |
| kkt_bf16_mma_p3_t64x64x32_w4 | 255 | 4080 | 18056 | 64 | 2056(cg) | 2048 | 326 | 32 | 350.8 |
| kkt_bf16_mma_p2_t64x64x16_w4 | 255 | 2360 | 8120 | 24 | 774(cg) | 768 | 126 | 32 | 162.6 |
| kkt_bf16_mma_p2_t32x64x32_w4 | 255 | 3912 | 7268 | 24 | 1158(cg) | 1152 | 150 | 16 | 254.4 |
| kkt_bf16_mma_p2_t64x64x32_w8 | 255 | 3768 | 12392 | 24 | 1542(cg) | 1536 | 222 | 16 | 262.6 |
| kkt_bf16_mma_p2_t64x64x32_w2 | 255 | 3680 | 11708 | 96 | 1542(cg) | 1536 | 294 | 63 | 303.2 |
| kkt_bf16_nomma_p2_t64x64x32_w4 | 255 | 1168 | 2332 | 0 | 514(cg) | 512 | 7 | 1 | 88.0 |
| kkt_bf16_nomma_p0_t64x64x32_w4 | 255 | 1168 | 2332 | 0 | 514(ca) | 512 | 7 | 1 | 88.0 |
| kkt_bf16_nomma_p2_t64x32x32_w4 | 255 | 0 | 0 | 0 | 386(cg) | 384 | 7 | 1 | 85.9 |
| o_bf16_mma_p2_t64x64x32_w4_K64V64 | 255 | 10216 | 33848 | 96 | 3084(cg) | 3072 | 492 | 0 | 683.2 |
| o_bf16_mma_p3_t64x128x16_w8_K128V128 | 255 | 8360 | 40328 | 64 | 3088(cg) | 3072 | 332 | 0 | 647.3 |
| o_bf16_mma_p0_t32x64x32_w2_K64V64 | 255 | 1992 | 6456 | 32 | 772(ca) | 768 | 134 | 0 | 179.4 |
| o_bf16_nomma_p2_t64x64x32_w4_K64V64 | 255 | 0 | 0 | 0 | 1028(cg) | 1024 | 13 | 0 | 196.1 |
| o_f32_mma_p2_t64x64x32_w4_K64V64 | 255 | 26760 | 118848 | 384 | 6156(cg) | 6144 | 492 | 0 | 1314.0 |
| o_f32_nomma_p0_t64x64x32_w4_K64V64 | 255 | 24 | 64 | 0 | 2052(ca) | 2048 | 13 | 0 | 362.0 |
| ref_f16_sm80_atom (rewrite op) | 28 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 3.2 |

MMA signatures observed (exact):
- `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` — all bf16 mma configs
  (the T1 workhorse signature; 10 configs) and in SASS as HMMA.16816.F32.BF16.
- `mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32` — f32/tf32 configs
  (o_f32_mma, 384 in-loop atoms).
- `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` — the ref anchor, produced
  by the in-tree `transform.nvgpu.rewrite_matmul_as_mma_sync` itself (sm80
  recipe retargeted to sm_120a).

cp.async exact spellings: `cp.async.ca.shared.global` (pipeline=0, no bypass_l1),
`cp.async.cg.shared.global` (pipeline>=2, bypass_l1), plus
`cp.async.commit_group` / `cp.async.wait_group` — matching the T1 expectation.

exp2: the kkt decay epilogue lowers to `ex2.approx.f32` (NON-ftz variant, like
qla). ftz is a future codegen flag; noted for the L3 knob follow-up.

ldmatrix: NOT emitted (0 everywhere). Our mma path loads fragments with
lane-indexed `ld.shared.b16` loads (same addressing as RewriteMatmulAsMmaSyncOp
emits); ldmatrix would require nvgpu.ldmatrix-based fragment loading — tracked
as an L3 knob candidate, not forced here.

## Gate status

- **(a) PASS** — 10 bf16 configs emit `mma.sync...f32.bf16.bf16.f32` (f32 accum
  on bf16 inputs); SASS HMMA counts match PTX counts 1:1 (e.g. 48/48).
- **(b) PASS** — 16/17 configs emit cp.async groups (commit/wait); only the f16
  atom ref (no smem staging) has none, as designed.
- **(c) PASS** — knob sensitivity, per knob (all deltas measured in the table):
  - `pipeline` 0→2→3 (kkt mma t64x64x32 w4): mma 16→48→64, LDGSTS 512→1536→2048,
    LDS 83→246→326, spills 9.7KB→12.6KB→18.1KB, PTX 126→276→351 KB; copy variant
    ca→cg; multibuffer factor 1→2→3 (smem tile 4KB→8KB→12KB per buffer).
  - `tile.M` 64→32: per-thread mma 48→24, LDS 246→150.
  - `tile.K` 32→16: in-flight cp.async 1542→774, mma 48→24.
  - `tile.N` 64→32 (nomma): cp.async 514→386.
  - `warps` 2→4→8: per-warp mma atoms 96→48→24 (each warp's N-slice halves),
    PTX size 303→276→263 KB.
  - `mma` on→off: HMMA 48→0, PTX 276→88 KB (3.1×), LDS 246→7.
  - `dtype` bf16→f32: atom m16n8k16.bf16 → m16n8k4.tf32 (4× atom count).
  - Registers: pinned at 255 (cap) for all full-stage kernels due to fully
    unrolled epilogues; the sensitivity signal lives in stack frame
    (0B→26.8KB) and spill bytes (64B→118.8KB); ref anchor uses 28 regs.
- **(d) PASS** — zero GPU usage: only mlir-opt/ptxas/nvdisasm (CPU binaries);
  mlir-runner never invoked (G0 rule respected).

## Exact transform-op spellings that worked (this Oct-2026 LLVM)

```
transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
transform.structured.match ops{["scf.for"]} in %arg0 : ...
transform.structured.match ops{["memref.alloc"]} in %arg0 : ... -> !transform.op<"memref.alloc">
transform.memref.multibuffer %allocs <factor = 2, skip_analysis> : (!transform.op<"memref.alloc">) -> !transform.any_op
transform.nvgpu.create_async_groups %func bypass_l1 : (!transform.any_op) -> (!transform.any_op)   // bypass_l1 -> cp.async.cg; without -> cp.async.ca
transform.nvgpu.pipeline_shared_memory_copies failures(propagate) %loop depth = 2 peel_epilogue : (!transform.any_op) -> !transform.any_op
transform.nvgpu.rewrite_matmul_as_mma_sync %matmul : (!transform.any_op) -> ()
transform.structured.tile_using_for %mms tile_sizes [16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
```
Critical ordering: `create_async_groups` invalidates handles into the func —
re-match `scf.for` AFTER it. `multibuffer` without `skip_analysis` fails on
transfer-write staging (analysis only accepts `memref.copy` overwrites).

## Toolchain quirks & constraints found (binding for later workstreams)

1. `rewrite_matmul_as_mma_sync` accepts ONLY single-warp atom matmuls
   (16x8x16 f16/f16/f16, 16x8x4 tf32) — it hard-codes laneId = threadIdx.x.
   Bigger GDN tiles are emitted by our generator as explicit nvgpu.mma.sync
   fragments (same lane formulas from NVGPUTransformOps.cpp).
2. `mma.sync.aligned.m16n8k8...tf32` is NOT reachable through nvgpu.mma.sync:
   the verifier enforces the 8x8x128b fundamental tile => K=4 for tf32 (the
   NVVM tabled op lists m16n8k8 only for f16/bf16/i8). KNOWN GAP vs the T1
   signature list; we emit m16n8k4.tf32 (f32 accum preserved).
3. The nvvm pipeline does not lower linalg: mma=off keeps one linalg.matmul
   (tiled by `tile_using_for`); `transform.structured.vectorize` HARD-fails on
   these shared-memory matmuls (no failures(suppress) clause on that op), so
   run_l2.sh lowers survivors with `-convert-linalg-to-loops`.
4. `pipeline_shared_memory_copies` requires a FLAT loop body (any op with a
   region inside the K-loop makes it bail) — hence all copies and mma atoms are
   generator-unrolled in the loop body.
5. `vector.transfer_write` to `#gpu.address_space<workgroup>` does not lower
   through this pipeline (LLVM translation fails: "conversion of memref memory
   space ... failed") — pipeline=0 configs therefore still use
   `create_async_groups` WITHOUT bypass_l1 (cp.async.ca, single buffer, no
   pipelining) instead of sync ld.global/st.shared staging.
6. ptxas-blackwell `-v` on these kernels reports no static smem bytes: the
   workgroup allocs lower to generic allocas + `cvta.to.shared` (no `.shared`
   globals in PTX). SASS confirms shared-window traffic anyway (LDGSTS/LDS).
   smem sensitivity is therefore tracked via LDS/LDGSTS counts and the
   multibuffer factor rather than the ptxas smem line.
7. ptxas-blackwell report phrase differs from plain ptxas ("3808 bytes
   cumulative stack size"); harvest_flaptx's `parse_ptxas_v` still parses
   regs/spills fine. Its FAMS pattern `\bexp2?\.approx` misses PTX `ex2.approx`
   (PTX spells it ex2, not exp2) — run_l2.sh adds `ex2.approx` separately.
8. `arith.cmpi` requires `pred, lhs, rhs` (comma after predicate) in this build.
   `memref.transpose` requires `(d0, d1) -> (d1, d0)` with an affine-map result.
9. wsl.exe quoting from Windows Git Bash strips `$vars` — all WSL work goes
   through script files in `evidence/G2/scratch/` (G0 rule, re-confirmed).

## INDEX.json

`evidence/G2/gen/INDEX.json` — per-config entries with `name`, `config`,
`path`, `ptx_path`/`ptx` (relative), `regs`/`smem`/`spills`, `ptxas` block
(ptx_features + metrics + mma_signatures + sass_ops), `status`, `seconds`.
G2-B's merge script can consume `configs[].ptx|ptx_path|path` and each config
dir ships its own `meta.json`.

## G2-C addendum: warp-spec + TMA + ldmatrix configs (3 new, total 20/20 ok)

Extended by G2-C (see `hatch_report.md` for the full story):

- knobs: `warpspec: off|producer_consumer`, `tma: off|on` (plus the
  `ref_ldmatrix` stage); `scripts/run_l2.sh` lowers `gdn.*` hatch ops via
  `gdn-opt --gdn-lower-hatch` before the standard chain.
- **hatch count 0/5**: every family (setmaxnreg/elect/mbarrier/TMA/prefetch)
  lowers to in-tree nvgpu/nvvm ops.

| config | regs | spill B | mma | cp.async | PTX KB |
|---|---|---|---|---|---|
| kkt_bf16_mma_p2_t64x64x64_w4_ws | 255 | 22844 | 32 | 1026 (cg classic) | 214.6 |
| kkt_bf16_mma_tma_ws | 255 | 48 | 32 | 5 (TMA bulk) | 41.2 |
| ref_ldmatrix | 72 | 0 | 1 | 0 | 10.1 |

- `kkt_bf16_mma_p2_t64x64x64_w4_ws`: producer warp issues classic
  `nvgpu.device_async_copy` (bypassL1 -> cp.async.cg) + create_group +
  wait_group 0; elected leader arrives on the mbarrier; consumers
  (4 warps, setmaxnreg inc 224 vs producer dec 96) try_wait.parity + mma.
- `kkt_bf16_mma_tma_ws`: producer's elected leader does
  prefetch.tensormap + arrive.expect_tx + `nvgpu.tma.async.load` (4d A, 5d B)
  + TMA 2d store epilogue + bulk commit/wait.read. Descriptors are created
  host-side (opaque at compile level; G4 wires runtime).
- `ref_ldmatrix`: single f16 m16n8k16 atom with `nvgpu.ldmatrix` fragment
  loads (x4; x2 .trans) + per-lane ld.global.b16/st.shared.b16 staging ->
  `ldmatrix.sync.aligned.m8n8.x4.shared.b16` +
  `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` in PTX (family NO->PARTIAL).
- Coverage effect (coverage_matrix.md, ours column): setmaxnreg YES,
  elect.sync YES, tma.bulk.ctrl YES, prefetch.tensormap YES, mbarrier
  PARTIAL (`.shared.b64` vs ref `.shared::cta.b64` spelling only),
  tma.bulk.tensor PARTIAL (no `.L2::cache_hint` loads / 4d-5d stores),
  shfl.sync PARTIAL (idx emitted). ours family coverage 3 YES/14 NO ->
  7 YES/6 NO of 33. tf32 m16n8k8 signature still unreachable through
  tabled nvgpu/nvvm mma ops (would be the first real inline-PTX hatch use).
