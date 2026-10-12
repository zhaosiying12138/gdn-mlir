# G2 coverage matrix (T1: instruction families + mma signatures + structure)

Generated 2026-10-08 19:13:51 UTC. ours-column state: **LIVE** (20 generated configs from `evidence/G2/gen/INDEX.json`)

Legend: YES = emitted by some config of that column; PARTIAL = family emitted but a reference opcode variant/signature is missing; NO = never emitted; - = not present in that reference.

## 1. Instruction families

| family | tier | fla | FlashQLA | ours | escape-hatch / plan (for NO cells) |
|---|---|---|---|---|---|
| `mma.sync` | feature | YES (3284) | YES (2496) | PARTIAL (missing: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32) | MLIR transform-matmul -> nvvm.mma.sync; m16n8k16.row.col.f32.bf16.bf16.f32 needs bf16 dot lowering; m16n8k8 tf32 sig needs tf32 precision knob (G2-A knob) |
| `ldmatrix` | feature | YES (1182) | YES (1016) | PARTIAL (missing: ldmatrix.sync.aligned.m8n8.x1.shared.b16, ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16, ldmatrix.sync.aligned.x4.m8n8.shared.b16, ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16) | gpu.ldmatrix / nvvm.ldmatrix (upstream NVVM dialect); .trans variant needs transpose flag |
| `stmatrix` | feature | YES (283) | YES (212) | NO | nvvm.stmatrix (upstream NVVM main; x2/x4/.trans variants) |
| `cp.async.classic` | feature | YES (28) | YES (8) | YES | gpu.async_copy (nvvm.cp.async.ca/cg) via MLIR async-copy pipelining |
| `cp.async.ctrl` | feature | YES (36) | YES (2) | PARTIAL (missing: cp.async.wait_all) | gpu.commit_group / gpu.wait_group (upstream) |
| `tma.bulk.tensor` | feature | - | YES (100) | PARTIAL (missing: cp.async.bulk.tensor.4d.global.shared::cta.bulk_group, cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint, cp.async.bulk.tensor.5d.global.shared::cta.bulk_group, cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint) | ESCAPE-HATCH: nvvm.cp.async.bulk.tensor.{3,4,5}d + nvvm.tensormap descriptors — no upstream MLIR op; counts vs <=5 budget (or G2-C manual pattern) |
| `tma.bulk.ctrl` | feature | - | YES (50) | YES | bundled with tma.bulk.tensor hatch (nvvm.cp.async.bulk.commit_group/wait_group.read) |
| `mbarrier` | feature | - | YES (386) | PARTIAL (missing: mbarrier.arrive.shared::cta.b64, mbarrier.expect_tx.shared::cta.b64, mbarrier.init.shared::cta.b64, mbarrier.try_wait.parity.shared::cta.b64) | ESCAPE-HATCH: nvvm.mbarrier.{init,arrive,expect_tx,try_wait.parity} — no upstream MLIR op; async pipeline G2-C |
| `elect.sync` | feature | - | YES (59) | YES | ESCAPE-HATCH: nvvm.elect.sync (warp-leader election for TMA issue) — G2-C |
| `setmaxnreg` | feature | - | YES (16) | YES | nvvm.setmaxnreg.inc/dec — warp specialization; PLANNED as G2-C deliverable |
| `prefetch.tensormap` | feature | - | YES (22) | YES | rides along with tma.bulk.tensor hatch (nvvm.prefetch.tensormap) |
| `bar.sync` | feature | YES (697) | YES (203) | YES | gpu.barrier -> bar.sync 0 (free) |
| `barrier` | feature | - | YES (39) | NO | gpu.barrier (free) |
| `shfl.sync` | feature | YES (966) | YES (80) | PARTIAL (missing: shfl.sync.bfly.b32) | gpu.shuffle (nvvm.shfl.sync) bfly/idx — upstream |
| `ex2.approx` | feature | YES (549) | YES (74) | PARTIAL (missing: ex2.approx.ftz.f32) | math.exp2 -> llvm.nvvm.ex2.approx.f32 under fastmath; .ftz variant needs ftz codegen flag |
| `ld.global` | baseline | YES (4237) | YES (581) | PARTIAL (missing: ld.global.b32, ld.global.nc.b16, ld.global.nc.b32, ld.global.nc.v2.b32, ld.global.nc.v4.b32, ld.global.v4.b32) | standard lowering; .v2/.vN vector width via vectorization |
| `st.global` | baseline | YES (1898) | YES (505) | PARTIAL (missing: st.global.b64, st.global.v2.b32, st.global.v4.b32) | standard lowering |
| `ld.shared` | baseline | YES (1624) | YES (992) | PARTIAL (missing: ld.shared.b64, ld.shared.v2.b16, ld.shared.v2.b32, ld.shared.v2.b64, ld.shared.v4.b16, ld.shared.v4.b32) | standard lowering (memref -> shared) |
| `st.shared` | baseline | YES (3754) | YES (1018) | PARTIAL (missing: st.shared.b64, st.shared.v2.b16, st.shared.v2.b32, st.shared.v4.b32, st.shared::cta.b16, st.shared::cta.b32, st.shared::cta.v2.b16, st.shared::cta.v2.b32, st.shared::cta.v4.b16, st.shared::cta.v4.b32) | standard lowering |
| `ld.param` | baseline | YES (167) | YES (481) | PARTIAL (missing: ld.param.b32, ld.param.b64, ld.param.s32) | kernel ABI, free |
| `fma.f32` | baseline | YES (835) | - | NO | arith/mulf+add fusion -> fma.rn.f32 (free) |
| `math.f32x2` | baseline | YES (1002) | YES (1547) | NO | vector<f32> x2 packed math (free via vector type) |
| `cvt.bf16f32` | baseline | YES (4088) | YES (2276) | PARTIAL (missing: add.bf16, add.bf16x2, add.rn.f32.bf16, cvt.rn.bf16x2.f32, fma.rn.bf16x2, mul.bf16, mul.bf16x2, sub.rn.f32.bf16) | arith.extf/truncf on bf16 (free; bf16x2 packing via vector<2xbf16>) |
| `math.f32` | baseline | YES (4154) | YES (762) | NO |  |
| `math.int` | baseline | YES (12785) | YES (3738) | PARTIAL (missing: bfe.s32, bfe.u32, bfe.u64, div.s32, mad.lo.s32, mad.wide.s32, mad.wide.u32, max.s32, min.s32, min.s64, min.u64, mul.hi.u32, mul.wide.s32, mul.wide.u16, neg.s64) | arith integer lowering (free) |
| `logic.shift` | baseline | YES (7534) | YES (7441) | PARTIAL (missing: and.b32, and.pred, not.b32, or.b16, or.pred, shl.b16, shr.s32, shr.u16, xor.b16, xor.b32) | arith bitwise/shift (free) |
| `cvt.int` | baseline | YES (802) | YES (2541) | PARTIAL (missing: cvt.s64.s32, cvt.u16.u32, cvt.u32.u16) | arith ext/trunc (free) |
| `cvta` | baseline | - | YES (120) | PARTIAL (missing: cvta.param.u64) | address spaces, free |
| `setp` | baseline | YES (1917) | YES (1794) | PARTIAL (missing: setp.eq.b32, setp.eq.b64, setp.ge.s32, setp.ge.u32, setp.gt.s32, setp.gt.u32, setp.gt.u64, setp.le.s32, setp.le.u32, setp.lt.s32, setp.lt.u32, setp.lt.u64) | arith.cmp (free) |
| `selp` | baseline | YES (2529) | YES (187) | PARTIAL (missing: selp.b64) | arith.select (free) |
| `mov` | baseline | YES (11775) | YES (6227) | PARTIAL (missing: mov.b16, mov.s32, mov.u16) | trivial (free) |
| `mov.pred` | baseline | YES (79) | - | NO | trivial (free) |
| `bra` | baseline | YES (108) | YES (2273) | YES | scf.for/if lowering (free) |

ours family coverage: 7 YES / 20 PARTIAL / 6 NO of 33 T1 families.

## 2. mma.sync signatures

| signature | fla | FlashQLA | ours | note |
|---|---|---|---|---|
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | 2708 | 2496 | YES (552) | bf16 workhorse (both refs) |
| `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` | 576 | 0 | NO (ours mma total 938) | tf32 kkt/dot precision (fla only) |

## 3. Structural dimensions

| dimension | fla envelope | FlashQLA envelope | ours | note |
|---|---|---|---|---|
| registers/thread | [40, 255] | [88, 128] | [28, 255] over 20 configs | fla rides the 255-reg cap with spills; qla pins <=128 via setmaxnreg |
| dynamic smem (B) | [0, 33288] | [1024, 2048] | [0, 0] over 20 configs | sm_120 CTA limit 100KB/128KB; qla tiny (2KB) because TMA stages out of smem descriptors |
| spills (B, max of st/ld) | [0, 5348] | [0, 1752] | [0, 48032] over 20 configs | fla bwd kernels spill heavily (dhu 3.8KB); fused_fwd/bwd ~1.1-1.3KB |
| async pipeline style | cp.async.ca only in state-push kernels (fwd_h/dhu); else ld.global multistage | TMA cp.async.bulk.tensor 4d/5d + mbarrier + elect.sync leader (kkt_solve: classic cp.async.cg) | cp.async.cg/ca x20-configs | two distinct pipelines to cover |
| warp specialization | none (no setmaxnreg/elect.sync) | setmaxnreg.inc/dec + elect.sync in 4/5 kernels | cp.async.cg/ca x20-configs | G2-C deliverable (nvvm.setmaxnreg) |
| loop structure (bra / back-edges) | [0, 16] bra, mostly unrolled | [107, 611] bra, real scf loops | cp.async.cg/ca x20-configs | qla re-uses loop bodies (fused_bwd 1600 mma from ~looped bodies); fla unrolls |

## 4. Escape-hatch budget (<= 5 custom NVVM ops)

No NO-cell feature families — escape-hatch budget untouched.

## 5. T2 structural comparison cards

### fla:chunk_fwd_kernel_o/BK64_BV64_warps4_stages3

- mma.sync total **96** (est. per loop body ~96 over 1 back-edge loop(s); bra=1)
- signatures: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 x96
- async pattern: cp.async.ca=0, ctrl(commit/wait)=0, TMA tensor=0, TMA ctrl=0, mbarrier=0, ld.global=146(of which .nc 0)
- resources: regs=255, smem=24576B, spill st/ld=96/96B
- ldmatrix=52, stmatrix=4, setmaxnreg=0, elect.sync=0, shfl=0
- ours closest config: **ref_ldmatrix** (`evidence/G2/gen/ref_ldmatrix/kernel.ptx`)
  - regs 72 vs ref 255 -> OUT (0.28x); smem None vs 24576 -> n/a (±30%)
  - mma total 1 vs ref 96 (0.01x); family Jaccard basis for 'closest' selection

### fla:chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64/BV64_warps2_stages3

- mma.sync total **512** (est. per loop body ~512 over 1 back-edge loop(s); bra=2)
- signatures: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 x256; mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 x256
- async pattern: cp.async.ca=6, ctrl(commit/wait)=8, TMA tensor=0, TMA ctrl=0, mbarrier=0, ld.global=640(of which .nc 0)
- resources: regs=255, smem=33288B, spill st/ld=3236/3788B
- ldmatrix=105, stmatrix=80, setmaxnreg=0, elect.sync=0, shfl=0
- ours closest config: **kkt_bf16_mma_p2_t64x64x32_w2** (`evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w2/kernel.ptx`)
  - regs 255 vs ref 255 -> PASS; smem None vs 33288 -> n/a (±30%)
  - mma total 96 vs ref 512 (0.19x); family Jaccard basis for 'closest' selection

### qla:fused_fwd/default

- mma.sync total **384** (est. per loop body ~32 over 12 back-edge loop(s); bra=611)
- signatures: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 x384
- async pattern: cp.async.ca=0, ctrl(commit/wait)=0, TMA tensor=31, TMA ctrl=24, mbarrier=74, ld.global=186(of which .nc 186)
- resources: regs=128, smem=2048B, spill st/ld=1160/1272B
- ldmatrix=136, stmatrix=44, setmaxnreg=4, elect.sync=18, shfl=0
- ours closest config: **kkt_bf16_mma_tma_ws** (`evidence/G2/gen/kkt_bf16_mma_tma_ws/kernel.ptx`)
  - regs 255 vs ref 128 -> OUT (1.99x); smem 8 vs 2048 -> OUT (0.00x) (±30%)
  - mma total 32 vs ref 384 (0.08x); family Jaccard basis for 'closest' selection

### qla:fused_bwd/default

- mma.sync total **1600** (est. per loop body ~200 over 8 back-edge loop(s); bra=422)
- signatures: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 x1600
- async pattern: cp.async.ca=0, ctrl(commit/wait)=0, TMA tensor=41, TMA ctrl=6, mbarrier=194, ld.global=77(of which .nc 76)
- resources: regs=128, smem=2048B, spill st/ld=1098/1752B
- ldmatrix=712, stmatrix=104, setmaxnreg=4, elect.sync=21, shfl=80
- ours closest config: **kkt_bf16_mma_tma_ws** (`evidence/G2/gen/kkt_bf16_mma_tma_ws/kernel.ptx`)
  - regs 255 vs ref 128 -> OUT (1.99x); smem 8 vs 2048 -> OUT (0.00x) (±30%)
  - mma total 32 vs ref 1600 (0.02x); family Jaccard basis for 'closest' selection

