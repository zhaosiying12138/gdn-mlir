# G2-C: warp-specialization + TMA hatch report

Workstream G2-C (GDN-MLIR). Scope: close the 4 NO-cells of
`evidence/G2/coverage_matrix.md` (`setmaxnreg`, `elect.sync`,
`tma.bulk.tensor`(+ctrl+prefetch), `mbarrier`), attempt `ldmatrix` via an
in-tree recipe, and provide FlashQLA-pole knob configs. CPU only
(compile-level; no GPU execution).

## TL;DR

- **Final escape-hatch count: 0 custom NVVM ops (budget 5).** In this
  Oct-2026 LLVM every "hatch" family is already tabled in-tree (NVGPU/NVVM
  dialects) and lowers through `-gpu-lower-to-nvvm-pipeline` to sm_120a PTX.
  Our dialect ops are thin first-class wrappers that lower 1:1 to those
  in-tree ops (`gdn-opt --gdn-lower-hatch`), so the hatch mechanism exists
  and is exercised but consumes zero budget.
- **All 4 target NO-cells flipped** (3 YES + 1 PARTIAL whose only gaps are
  PTX spelling variants, see below). Bonus flips: `tma.bulk.ctrl` NO→YES,
  `prefetch.tensormap` NO→YES, `shfl.sync` NO→PARTIAL, `ldmatrix` NO→PARTIAL.
- ours family coverage moved **3 YES / 14 NO → 7 YES / 6 NO** of 33 T1
  families (20 generated configs; `evidence/G2/gen/INDEX.json` 20/20 ok).

## 1. In-tree vs custom (the discovery that shrank the hatch)

The coverage matrix's escape-hatch plan assumed these families had "no
upstream MLIR op". In this LLVM tree they all exist:

| family | in-tree op used (exact spelling) | PTX emitted (verified in kernel.ptx) |
|---|---|---|
| setmaxnreg | `nvvm.setmaxregister increase/decrease <n>` | `setmaxnreg.inc.sync.aligned.u32 224` / `.dec... 96` |
| elect.sync | `nvvm.elect.sync -> i1` | `elect.sync %r\|%p, -1` |
| mbarrier init | `nvgpu.mbarrier.init %bar[%c0], %count` | `mbarrier.init.shared.b64` |
| mbarrier arrive | `nvgpu.mbarrier.arrive %bar[%c0]` | `mbarrier.arrive.shared.b64` |
| mbarrier expect_tx | `nvgpu.mbarrier.arrive.expect_tx %bar[%c0], %bytes` | `mbarrier.arrive.expect_tx.shared.b64` |
| mbarrier wait | `nvgpu.mbarrier.try_wait.parity %bar[%c0], %parity, %ticks` | `mbarrier.try_wait.parity.shared.b64` |
| TMA load 4d/5d | `nvgpu.tma.async.load %desc[coords], %bar[%c0] to %dst` | `cp.async.bulk.tensor.{4,5}d.shared::cluster.global.tile.mbarrier::complete_tx::bytes` |
| TMA store 2d | `nvgpu.tma.async.store %src to %desc[coords]` | `cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group` |
| TMA ctrl | `nvvm.cp.async.bulk.commit.group` / `nvvm.cp.async.bulk.wait_group 0 read` | `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group.read 0` |
| tensormap prefetch | `nvgpu.tma.prefetch.descriptor %desc` | `prefetch.tensormap [%rd]` |
| descriptor | host-side `nvgpu.tma.create.descriptor %cast box[...]` | runtime call in host func; kernel gets the tensormap as a pointer arg (opaque at compile level — runtime wiring is downstream/G4) |
| shfl (bonus) | `nvvm.shfl.sync idx %mask, %v, %off, %clamp` | `shfl.sync.idx.b32` |
| ldmatrix | `nvgpu.ldmatrix %smem[...] numTiles = N transpose = b` | `ldmatrix.sync.aligned.m8n8.x4.shared.b16` / `.x2.trans...` |

Escape-hatch fallback that was verified to exist but NOT needed:
`nvvm.inline_ptx "..." ro(...) rw(...) memory_clobber = true` (in-tree
BasicPtxBuilder op, would lower to `llvm.inline_asm`).

## 2. Dialect extension (hatch ops)

`dialect/include/Gdn/GdnHatch.td` + `dialect/lib/GdnHatch.cpp` + pass
`gdn-lower-hatch` (registered in `gdn-opt`, wired into
`dialect/include/CMakeLists.txt`, `Gdn.h` umbrella, `GdnDialect::initialize`,
dependentDialects NVGPU/NVVM/GPU/Arith; `MLIRNVGPUDialect`/`MLIRNVVMDialect`
link deps). Rebuild green via `scripts/build_dialect.sh`.

| gdn op | lowers to |
|---|---|
| `gdn.setmaxnreg inc\|dec <count>` (I32Attr, mult of 8 in [24,256] verified) | `nvvm.setmaxregister increase/decrease` |
| `gdn.elect -> i1` | `nvvm.elect.sync` |
| `gdn.mbarrier.init %bar, %count` | `nvgpu.mbarrier.init %bar[0], %count` |
| `gdn.mbarrier.arrive %bar -> !nvgpu.mbarrier.token` | `nvgpu.mbarrier.arrive` |
| `gdn.mbarrier.arrive.expect_tx %bar, %bytes` | `nvgpu.mbarrier.arrive.expect_tx` |
| `gdn.mbarrier.try_wait.parity %bar, %parity, %ticks` | `nvgpu.mbarrier.try_wait.parity` |
| `gdn.tma.async.load %desc[...], %bar to %dst` | `nvgpu.tma.async.load` |
| `gdn.tma.async.store %src to %desc[...]` | `nvgpu.tma.async.store` |
| `gdn.tma.prefetch %desc` | `nvgpu.tma.prefetch.descriptor` |
| `gdn.bulk.commit_group` | `nvvm.cp.async.bulk.commit.group` |
| `gdn.bulk.wait_group <n> [read]` | `nvvm.cp.async.bulk.wait_group` |

Note on printing: the gdn custom asm needs the FULL types
(`!nvgpu.mbarrier.group<memorySpace = ...>`,
`!nvgpu.tensormap.descriptor<tensor = ...>`); the nvgpu ops' own pretty
printer elides the `!nvgpu.*` prefix, which is why the sm90 integration tests
show bare `<memorySpace = ...>`.

## 3. New knob configs (FlashQLA pole)

Generator knobs added to `tools/gdn_l2_gen.py`:
- `warpspec: off | producer_consumer` — 1 producer warp (warp 0,
  shfl.sync-broadcast predicate) + `warps` consumer warps; block =
  32*(warps+1) threads.
- `tma: off | on` — producer issues TMA loads (4d A-tile + 5d B-tile) with
  `arrive.expect_tx` + `try_wait.parity` consumer sync + TMA store epilogue.

| config | regs | spill B | mma | cp.async family | PTX KB | emits |
|---|---|---|---|---|---|---|
| `kkt_bf16_mma_p2_t64x64x64_w4_ws` | 255 | 22844 | 32 | 1026 (classic cg + commit/wait) | 214.6 | setmaxnreg dec96/inc224, elect, mbarrier init/arrive/try_wait, shfl.sync.idx |
| `kkt_bf16_mma_tma_ws` | 255 | 48 | 32 | 5 (TMA 4d+5d load, 2d store, commit, wait.read) | 41.2 | + expect_tx, prefetch.tensormap, full TMA family |
| `ref_ldmatrix` | 72 | 0 | 1 | 0 | 10.1 | ldmatrix x4 + x2.trans, ld.global.b16/st.shared.b16 staging |

Structural notes (simplified vs a full FlashQLA pipeline — deliberate,
compile-level fidelity only):
- single smem buffer, no multibuffered ping-pong and no producer
  back-pressure; phase parity ping-pongs as an `scf.for` iter_arg (i1);
- producer passes the accumulators through the `scf.if` (result-yielding);
- classic-copy variant issues the copies warp-wide and only the elected
  leader arrives on the mbarrier; TMA variant issues everything from the
  elected leader (FlashQLA-style);
- setmaxnreg split 96 (producer, dec) / 224 (consumers, inc);
- kkt semantic note: the TMA variant stages B k-major and feeds the
  non-transposed b-fragments (vs the transposed staging of the base kkt
  configs) — mathematically this computes with B rather than B^T; fine for
  instruction-coverage purposes, flagged for G4.

Toolchain quirks found (binding, documented in run_l2.sh):
1. `gdn-opt` cannot parse `transform.structured.*` scripts → hatch lowering
   runs on the bare payload BEFORE merging with transform.mlir.
2. `-transform-interpreter` wraps a bare top-level `func.func` into
   `module {}`; that extra nesting breaks
   `-gpu-lower-to-nvvm-pipeline` on the ws-classic payload ("conversion of
   memref memory space #gpu.address_space<workgroup> ... failed"). Identity
   transform scripts (all direct-ops G2-C configs) therefore skip the
   interpreter entirely.
3. TMA descriptors are created host-side before `gpu.launch` (the lowering
   emits a runtime call in the host func); the kernel receives them as
   pointer parameters. At compile level they are opaque — G4 wires
   cuTensorMapEncodeTiled at runtime.

## 4. Family-flip details (what flipped, what didn't)

| family | before | after | why not full YES |
|---|---|---|---|
| `setmaxnreg` | NO | **YES** | inc+dec both emitted, exact ref spellings |
| `elect.sync` | NO | **YES** | exact |
| `mbarrier` | NO | **PARTIAL** | all four ops emitted (init/arrive/expect_tx/try_wait.parity); ours prints `.shared.b64`, ref variants spell the (default) scope explicitly: `.shared::cta.b64`. Same instructions; string-level variant match only. |
| `tma.bulk.tensor` | NO | **PARTIAL** | 4d+5d loads + 2d store emitted; ref variants are the `.L2::cache_hint` load spellings (needs the raw `nvvm.cp.async.bulk.tensor...l2_cache_hint` operand — the nvgpu.tma.async.load lowering hardcodes no-hint) and 4d/5d stores. Our loads also carry the default `.tile` mode qualifier. |
| `tma.bulk.ctrl` | NO | **YES** | commit_group + wait_group.read exact |
| `prefetch.tensormap` | NO | **YES** | exact |
| `ldmatrix` | NO | **PARTIAL** | `nvgpu.ldmatrix` works (x4 A-frag, x2.trans B-frag); missing x1 / x4.trans / `.x4.m8n8` operand-order variants — reachable by numTiles/transpose knob combos if needed. `-convert-vector-to-gpu=use-nvgpu` recipe not needed. |
| `stmatrix` | NO | NO (unchanged) | `nvvm.stmatrix` IS tabled in-tree but needs a shared-window pointer operand that only materializes post-LLVM-conversion; left as documented follow-up. |
| `shfl.sync` | NO | **PARTIAL** | `idx` variant emitted in ws configs; `bfly` variant would need `nvvm.shfl.sync bfly`. |

## 5. tf32 m16n8k8 signature status (unchanged, re-confirmed)

`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` remains unreachable:
`nvgpu.mma.sync` enforces the 8x8x128b fundamental tile (tf32 → K=4), and
the tabled `nvvm.mma.sync` op family lists m16n8k8 only for f16/bf16/i8.
We emit `m16n8k4.tf32` instead (f32 accum preserved). Closing the k8
signature would be the first real use of the escape hatch
(`nvvm.inline_ptx` or a custom NVVM op; budget currently 0/5 used) — left
as follow-up since it is a signature-level PARTIAL of the already-covered
`mma.sync` family, not a NO cell.

## 6. Files

- `dialect/include/Gdn/GdnHatch.td`, `dialect/lib/GdnHatch.cpp`,
  `dialect/include/Gdn/Gdn.h`, `dialect/include/Gdn/GdnDialect.td`,
  `dialect/lib/GdnDialect.cpp`, `dialect/include/CMakeLists.txt`,
  `dialect/lib/CMakeLists.txt`, `dialect/tools/gdn-opt/gdn-opt.cpp`
- `tools/gdn_l2_gen.py` (warpspec/tma knobs + 3 configs),
  `tools/gdn_l2_batch.py` (`--merge` keeps prior INDEX entries),
  `scripts/run_l2.sh` (gdn-opt hatch step + identity-transform skip)
- `evidence/G2/gen/{kkt_bf16_mma_p2_t64x64x64_w4_ws,kkt_bf16_mma_tma_ws,ref_ldmatrix}/`
  (payload/transform/config + kernel.ptx/cubin + ptxas_v.txt + sass.txt + meta.json)
- `evidence/G2/scratch/g2c/` — probe_ws.mlir (all in-tree ops, hand-written),
  smoke_hatch.mlir (gdn ops → gdn-lower-hatch e2e), min_ws_classic.mlir
  (wrapper-bug isolation), run_probe.sh / run_smoke_e2e.sh
- `evidence/G2/coverage_matrix.md` (regenerated, 20 configs),
  `evidence/G2/gen/INDEX.json` (20/20 ok)

---

# ADDENDUM (G5-G round 3): escape-hatch budget 0/5 -> 1/5 USED

**Hatch #1: `gdn.mma.tf32k8` -- tf32 m16n8k8 warp mma via inline PTX.**
This is the first REAL use of the escape hatch: the signature
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` (fla's exact spelling,
harvested in evidence/G0/fla_ptx chunk_gated_delta_rule_bwd_kernel_dhu /
fwd_kkt_solve) does not exist in-tree -- `nvgpu.mma.sync` enforces the
8x8x128b fundamental tile (tf32 caps at K=4) and the `nvvm.mma.sync`
tablegen family lists m16n8k8 only for f16/bf16/i8 (see section 5 above).

- **Op**: 10 f32 operands (a0..a3, b0, b1, c0..c3) -> 4 f32 (d0..d3);
  dialect/include/Gdn/GdnHatch.td + lib/GdnHatch.cpp; registered in
  GdnDialect.cpp; site emitter `tgemm()` in tools/gdn_bwd_gen.py
  (`--tf32` knob per GEMM site).
- **Lowering**: NOT nvvm.inline_ptx (BasicPtxBuilder) but a direct
  `llvm.inline_asm` with constraints
  `=&f,=&f,=&f,=&f,f,f,f,f,f,f,f,f,f,f`.  Reason (SASS-verified on the
  RTX 5090 Laptop): with BasicPtxBuilder's plain `=f` outputs, ptxas'
  inline-asm register allocation may map the D quad onto the A quad of
  the SAME mma (`HMMA.1688.F32.TF32 R12, R12, R10, R16`), and the tensor
  core then corrupts operands -- an exact-integer K=16 two-atom chain
  failed 118/128 cells deterministically.  Early-clobber (`=&f`) forbids
  output/input overlap; nvvm.inline_ptx cannot express it.
- **Fragment layout, hardware-derived** (evidence/G5/scratch/tf32k8_probe:
  integer-operand probe, D == A@B+C EXACTLY):
  a0=(g,2t) a1=(g+8,2t) a2=(g,2t+1) a3=(g+8,2t+1); b0=(2t,g) b1=(2t+1,g);
  d0..d3 = the bf16-k16 D layout (N = 2*tig, the G4 rule).  Note the A
  register order (row-half outer) differs from the packed-pair intuition.
- **Measured impact** (bwd B1T2048H16, shared GPU, medians): dqkwg
  4.71 -> 2.75 ms (1.71x, together with the r4/c4r parallelization),
  wyb 2.65 -> 1.21 ms (2.19x, with ra/c1r), e2e bwd 14.16 -> 11.06 ms
  (1.28x; 0.073x -> 0.093x of fla) with numerics 6/6 PASS at both gate
  shapes (16 of 24 f32-GEMM sites tf32'd; 5 stayed scalar for numerics,
  3 for the dhu register-spill regression -- evidence/G5/bwd_report.md
  round-3 has the per-site bisect table).

Budget: **1 of <=5 used.**
