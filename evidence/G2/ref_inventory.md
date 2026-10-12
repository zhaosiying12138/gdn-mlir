# G2-B reference PTX inventory (T1 set source)

Generated 2026-10-08 19:13:51 UTC. Sources: `evidence/G0/fla_ptx` (22 configs, Triton sm_120), `evidence/G0/qla_ptx` (5 kernels, TileLang sm_120a).

## 1. Instruction families (family x which reference kernels use it)

Tier `feature` = needs an explicit IR affordance (intrinsics/pipelining); `baseline` = falls out of standard arith/linalg->NVVM lowering.

| family | tier | fla cfgs (count) | fla kernels | qla kernels (count) | exact opcode variants |
|---|---|---|---|---|---|
| `mma.sync` | feature | 18 (3284) | 9: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (2496) | mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32<br>mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 |
| `ldmatrix` | feature | 20 (1182) | 11: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (1016) | ldmatrix.sync.aligned.m8n8.x1.shared.b16<br>ldmatrix.sync.aligned.m8n8.x4.shared.b16<br>ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16<br>ldmatrix.sync.aligned.x4.m8n8.shared.b16<br>ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 |
| `stmatrix` | feature | 12 (283) | 8: chunk_bwd_dqkwg, chunk_fwd_o, cgdr_bwd_dhu_blockdim64, ... | fused_bwd, fused_fwd, prepare_h, prepare_h_s3 (212) | stmatrix.sync.aligned.m8n8.x2.shared.b16<br>stmatrix.sync.aligned.m8n8.x4.shared.b16<br>stmatrix.sync.aligned.m8n8.x4.trans.shared.b16<br>stmatrix.sync.aligned.x4.m8n8.shared.b16 |
| `cp.async.classic` | feature | 4 (28) | cgdr_bwd_dhu_blockdim64, cgdr_fwd_h_blockdim64 | kkt_solve (8) | cp.async.ca.shared.global<br>cp.async.cg.shared.global |
| `cp.async.ctrl` | feature | 4 (36) | cgdr_bwd_dhu_blockdim64, cgdr_fwd_h_blockdim64 | kkt_solve (2) | cp.async.commit_group<br>cp.async.wait_all<br>cp.async.wait_group |
| `tma.bulk.tensor` | feature | 0 (0) | - | fused_bwd, fused_fwd, prepare_h, prepare_h_s3 (100) | cp.async.bulk.tensor.4d.global.shared::cta.bulk_group<br>cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint<br>cp.async.bulk.tensor.5d.global.shared::cta.bulk_group<br>cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint |
| `tma.bulk.ctrl` | feature | 0 (0) | - | fused_bwd, fused_fwd, prepare_h, prepare_h_s3 (50) | cp.async.bulk.commit_group<br>cp.async.bulk.wait_group.read |
| `mbarrier` | feature | 0 (0) | - | fused_bwd, fused_fwd, prepare_h, prepare_h_s3 (386) | mbarrier.arrive.shared::cta.b64<br>mbarrier.expect_tx.shared::cta.b64<br>mbarrier.init.shared::cta.b64<br>mbarrier.try_wait.parity.shared::cta.b64 |
| `elect.sync` | feature | 0 (0) | - | fused_bwd, fused_fwd, prepare_h, prepare_h_s3 (59) | elect.sync |
| `setmaxnreg` | feature | 0 (0) | - | fused_bwd, fused_fwd, prepare_h, prepare_h_s3 (16) | setmaxnreg.dec.sync.aligned.u32<br>setmaxnreg.inc.sync.aligned.u32 |
| `prefetch.tensormap` | feature | 0 (0) | - | fused_bwd, fused_fwd, prepare_h, prepare_h_s3 (22) | prefetch.tensormap |
| `bar.sync` | feature | 21 (697) | 12: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (203) | bar.sync |
| `barrier` | feature | 0 (0) | - | fused_bwd, fused_fwd, prepare_h, prepare_h_s3 (39) | fence.mbarrier_init.release.cluster<br>fence.proxy.async.shared::cta |
| `shfl.sync` | feature | 11 (966) | 6: chunk_bwd_dqkwg, cgdr_fwd_kkt_solve, merge_16x16_to_32x32_inverse, ... | fused_bwd (80) | shfl.sync.bfly.b32<br>shfl.sync.idx.b32 |
| `ex2.approx` | feature | 17 (549) | 8: chunk_bwd_dqkwg, chunk_fwd_o, cgdr_bwd_dhu_blockdim64, ... | fused_bwd, fused_fwd, prepare_h, prepare_h_s3 (74) | ex2.approx.f32<br>ex2.approx.ftz.f32 |
| `ld.global` | baseline | 22 (4237) | 12: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (581) | ld.global.b16<br>ld.global.b32<br>ld.global.nc.b16<br>ld.global.nc.b32<br>ld.global.nc.v2.b32<br>ld.global.nc.v4.b32<br>ld.global.v4.b32 |
| `st.global` | baseline | 22 (1898) | 12: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (505) | st.global.b16<br>st.global.b32<br>st.global.b64<br>st.global.v2.b32<br>st.global.v4.b32 |
| `ld.shared` | baseline | 21 (1624) | 12: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (992) | ld.shared.b16<br>ld.shared.b32<br>ld.shared.b64<br>ld.shared.v2.b16<br>ld.shared.v2.b32<br>ld.shared.v2.b64<br>ld.shared.v4.b16<br>ld.shared.v4.b32 |
| `st.shared` | baseline | 21 (3754) | 12: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (1018) | st.shared.b16<br>st.shared.b32<br>st.shared.b64<br>st.shared.v2.b16<br>st.shared.v2.b32<br>st.shared.v4.b32<br>st.shared::cta.b16<br>st.shared::cta.b32<br>st.shared::cta.v2.b16<br>st.shared::cta.v2.b32<br>st.shared::cta.v4.b16<br>st.shared::cta.v4.b32 |
| `ld.param` | baseline | 22 (167) | 12: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (481) | ld.param.b32<br>ld.param.b64<br>ld.param.s32 |
| `fma.f32` | baseline | 13 (835) | 8: chunk_bwd_dqkwg, chunk_fwd_o, cgdr_bwd_dhu_blockdim64, ... | - (0) | fma.rn.f32 |
| `math.f32x2` | baseline | 13 (1002) | 7: chunk_bwd_dqkwg, cgdr_bwd_dhu_blockdim64, cgdr_fwd_h_blockdim64, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (1547) | add.f32x2<br>add.rn.f32x2<br>fma.rn.f32x2<br>mul.f32x2<br>mul.rn.f32x2<br>sub.f32x2<br>sub.rn.f32x2 |
| `cvt.bf16f32` | baseline | 22 (4088) | 12: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (2276) | add.bf16<br>add.bf16x2<br>add.rn.f32.bf16<br>cvt.f32.bf16<br>cvt.rn.bf16.f32<br>cvt.rn.bf16x2.f32<br>fma.rn.bf16x2<br>mul.bf16<br>mul.bf16x2<br>sub.rn.f32.bf16 |
| `math.f32` | baseline | 19 (4154) | 10: chunk_bwd_dqkwg, chunk_fwd_o, cgdr_bwd_dhu_blockdim64, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (762) | add.f32<br>mul.f32<br>neg.f32<br>sub.f32 |
| `math.int` | baseline | 22 (12785) | 12: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (3738) | add.s32<br>add.s64<br>bfe.s32<br>bfe.u32<br>bfe.u64<br>div.s32<br>div.u32<br>mad.lo.s32<br>mad.lo.s64<br>mad.wide.s32<br>mad.wide.u32<br>max.s32<br>min.s32<br>min.s64<br>min.u64<br>mul.hi.u32<br>mul.lo.s32<br>mul.lo.s64<br>mul.wide.s32<br>mul.wide.u16<br>mul.wide.u32<br>neg.s32<br>neg.s64<br>sub.s32<br>sub.s64 |
| `logic.shift` | baseline | 22 (7534) | 12: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (7441) | and.b16<br>and.b32<br>and.b64<br>and.pred<br>not.b32<br>not.pred<br>or.b16<br>or.b32<br>or.b64<br>or.pred<br>shl.b16<br>shl.b32<br>shl.b64<br>shr.s32<br>shr.u16<br>shr.u32<br>shr.u64<br>xor.b16<br>xor.b32 |
| `cvt.int` | baseline | 22 (802) | 12: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (2541) | cvt.s64.s32<br>cvt.u16.u32<br>cvt.u32.u16<br>cvt.u32.u64<br>cvt.u64.u32 |
| `cvta` | baseline | 0 (0) | - | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (120) | cvta.param.u64<br>cvta.to.global.u64 |
| `setp` | baseline | 22 (1917) | 12: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (1794) | setp.eq.b32<br>setp.eq.b64<br>setp.ge.s32<br>setp.ge.u32<br>setp.gt.s32<br>setp.gt.s64<br>setp.gt.u32<br>setp.gt.u64<br>setp.le.s32<br>setp.le.u32<br>setp.le.u64<br>setp.lt.s32<br>setp.lt.s64<br>setp.lt.u32<br>setp.lt.u64<br>setp.ne.b32<br>setp.ne.b64 |
| `selp` | baseline | 21 (2529) | 12: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, kkt_solve, prepare_h_s3 (187) | selp.b16<br>selp.b32<br>selp.b64<br>selp.f32 |
| `mov` | baseline | 22 (11775) | 12: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (6227) | mov.b16<br>mov.b32<br>mov.b64<br>mov.s32<br>mov.u16<br>mov.u32 |
| `mov.pred` | baseline | 17 (79) | 9: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | - (0) | mov.pred |
| `bra` | baseline | 22 (108) | 12: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | fused_bwd, fused_fwd, kkt_solve, prepare_h, prepare_h_s3 (2273) | bra<br>bra.uni<br>ret |

## 2. mma.sync signatures (unique full strings, with counts)

| signature | fla | fla kernels | qla | qla kernels |
|---|---|---|---|---|
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | 2708 | 9: chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o, ... | 2496 | 5: fused_bwd, fused_fwd, kkt_solve, ... |
| `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` | 576 | cgdr_bwd_dhu_blockdim64, cgdr_fwd_kkt_solve | 0 | - |

## 3. Resource envelopes per kernel (regs / dynamic smem / max spill / mma totals over configs)

| reference:kernel | configs | regs | smem (B) | spill max (B) | mma.sync total |
|---|---|---|---|---|---|
| fla:chunk_bwd_kernel_dqkwg | ['warps4_stages2', 'warps4_stages3'] | 255-255 | 32776-32776 | 1544 | 192-192 |
| fla:chunk_bwd_kernel_dv_local | ['warps4_stages3'] | 222-222 | 16384-16384 | 0 | 32-32 |
| fla:chunk_fwd_kernel_o | ['BK32_BV32_warps2_stages3', 'BK64_BV64_warps4_stages3'] | 255-255 | 10240-24576 | 824 | 80-96 |
| fla:chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 | ['BV32_warps2_stages4', 'BV64_warps2_stages3'] | 255-255 | 29452-33288 | 3788 | 256-512 |
| fla:chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | ['BV32_warps2_stages4', 'BV64_warps2_stages3'] | 255-255 | 17164-25096 | 764 | 128-256 |
| fla:chunk_gated_delta_rule_fwd_kkt_solve_kernel | ['warps2_stages3', 'warps4_stages3', 'warps4_stages4'] | 255-255 | 5120-6144 | 212 | 140-140 |
| fla:chunk_scaled_dot_kkt_fwd_kernel | ['BK128_warps8_stages3', 'BK64_warps4_stages3'] | 64-194 | 16384-32768 | 20 | 32-32 |
| fla:merge_16x16_to_32x32_inverse_kernel | ['warps4_stages2'] | 56-56 | 3072-3072 | 0 | 0-0 |
| fla:merge_16x16_to_64x64_inverse_kernel | ['warps4_stages2'] | 246-246 | 10240-10240 | 0 | 0-0 |
| fla:prepare_wy_repr_bwd_kernel | ['warps2_stages4', 'warps4_stages3'] | 255-255 | 17408-24576 | 5348 | 288-576 |
| fla:recompute_w_u_fwd_kernel | ['warps2_stages3', 'warps4_stages2'] | 255-255 | 16384-16384 | 444 | 64-128 |
| fla:solve_tril_16x16_kernel | ['warps1_stages4', 'warps4_stages2'] | 40-66 | 0-256 | 0 | 0-0 |
| qla:fused_bwd | ['default'] | 128-128 | 2048-2048 | 1752 | 1600-1600 |
| qla:fused_fwd | ['default'] | 128-128 | 2048-2048 | 1272 | 384-384 |
| qla:kkt_solve | ['default'] | 88-88 | 1024-1024 | 0 | 64-64 |
| qla:prepare_h | ['default'] | 128-128 | 2048-2048 | 0 | 224-224 |
| qla:prepare_h_s3 | ['default'] | 128-128 | 2048-2048 | 8 | 224-224 |

## 4. Structural notes (from PTX loop structure)

- fla:chunk_bwd_kernel_dqkwg/warps4_stages2: bra=1, loop back-edges=1
- fla:chunk_bwd_kernel_dqkwg/warps4_stages3: bra=1, loop back-edges=1
- fla:chunk_bwd_kernel_dv_local/warps4_stages3: bra=1, loop back-edges=1
- fla:chunk_fwd_kernel_o/BK32_BV32_warps2_stages3: bra=1, loop back-edges=1
- fla:chunk_fwd_kernel_o/BK64_BV64_warps4_stages3: bra=1, loop back-edges=1
- fla:chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64/BV32_warps2_stages4: bra=2, loop back-edges=1
- fla:chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64/BV64_warps2_stages3: bra=2, loop back-edges=1
- fla:chunk_gated_delta_rule_fwd_kernel_h_blockdim64/BV32_warps2_stages4: bra=2, loop back-edges=1
- fla:chunk_gated_delta_rule_fwd_kernel_h_blockdim64/BV64_warps2_stages3: bra=2, loop back-edges=1
- fla:chunk_gated_delta_rule_fwd_kkt_solve_kernel/warps2_stages3: bra=16, loop back-edges=5
- fla:chunk_gated_delta_rule_fwd_kkt_solve_kernel/warps4_stages3: bra=15, loop back-edges=5
- fla:chunk_gated_delta_rule_fwd_kkt_solve_kernel/warps4_stages4: bra=15, loop back-edges=5
- fla:chunk_scaled_dot_kkt_fwd_kernel/BK128_warps8_stages3: bra=0, loop back-edges=0
- fla:chunk_scaled_dot_kkt_fwd_kernel/BK64_warps4_stages3: bra=1, loop back-edges=1
- fla:merge_16x16_to_32x32_inverse_kernel/warps4_stages2: bra=4, loop back-edges=2
- fla:merge_16x16_to_64x64_inverse_kernel/warps4_stages2: bra=8, loop back-edges=4
- fla:prepare_wy_repr_bwd_kernel/warps2_stages4: bra=3, loop back-edges=3
- fla:prepare_wy_repr_bwd_kernel/warps4_stages3: bra=3, loop back-edges=3
- fla:recompute_w_u_fwd_kernel/warps2_stages3: bra=2, loop back-edges=2
- fla:recompute_w_u_fwd_kernel/warps4_stages2: bra=2, loop back-edges=2
- fla:solve_tril_16x16_kernel/warps1_stages4: bra=2, loop back-edges=1
- fla:solve_tril_16x16_kernel/warps4_stages2: bra=2, loop back-edges=1
- qla:fused_bwd/default: bra=422, loop back-edges=8
- qla:fused_fwd/default: bra=611, loop back-edges=12
- qla:kkt_solve/default: bra=107, loop back-edges=0
- qla:prepare_h/default: bra=400, loop back-edges=11
- qla:prepare_h_s3/default: bra=400, loop back-edges=11
