| kernel | config | regs | smem_dyn | spill_st | spill_ld | mma.sync | cp.async | exp.approx |
|---|---|---|---|---|---|---|---|---|
| chunk_bwd_kernel_dqkwg | {'warps': 4, 'stages': 2} | 255 | 32776 | 1544 | 1440 | 192 | 0 | 0 |
| chunk_bwd_kernel_dqkwg | {'warps': 4, 'stages': 3} | 255 | 32776 | 1544 | 1440 | 192 | 0 | 0 |
| chunk_bwd_kernel_dv_local | {'warps': 4, 'stages': 3} | 222 | 16384 | 0 | 0 | 32 | 0 | 0 |
| chunk_fwd_kernel_o | {'BK': 32, 'BV': 32, 'warps': 2, 'stages': 3} | 255 | 10240 | 824 | 748 | 80 | 0 | 0 |
| chunk_fwd_kernel_o | {'BK': 64, 'BV': 64, 'warps': 4, 'stages': 3} | 255 | 24576 | 96 | 96 | 96 | 0 | 0 |
| chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 | {'BV': 32, 'warps': 2, 'stages': 4} | 255 | 29452 | 2156 | 2400 | 256 | 18 | 0 |
| chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 | {'BV': 64, 'warps': 2, 'stages': 3} | 255 | 33288 | 3236 | 3788 | 512 | 14 | 0 |
| chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | {'BV': 32, 'warps': 2, 'stages': 4} | 255 | 17164 | 380 | 384 | 128 | 18 | 0 |
| chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | {'BV': 64, 'warps': 2, 'stages': 3} | 255 | 25096 | 716 | 764 | 256 | 14 | 0 |
| chunk_gated_delta_rule_fwd_kkt_solve_kernel | {'warps': 2, 'stages': 3} | 255 | 5120 | 212 | 212 | 140 | 0 | 0 |
| chunk_gated_delta_rule_fwd_kkt_solve_kernel | {'warps': 4, 'stages': 3} | 255 | 6144 | 40 | 40 | 140 | 0 | 0 |
| chunk_gated_delta_rule_fwd_kkt_solve_kernel | {'warps': 4, 'stages': 4} | 255 | 6144 | 40 | 40 | 140 | 0 | 0 |
| chunk_scaled_dot_kkt_fwd_kernel | {'BK': 128, 'warps': 8, 'stages': 3} | 64 | 32768 | 20 | 20 | 32 | 0 | 0 |
| chunk_scaled_dot_kkt_fwd_kernel | {'BK': 64, 'warps': 4, 'stages': 3} | 194 | 16384 | 0 | 0 | 32 | 0 | 0 |
| merge_16x16_to_32x32_inverse_kernel | {'warps': 4, 'stages': 2} | 56 | 3072 | 0 | 0 | 0 | 0 | 0 |
| merge_16x16_to_64x64_inverse_kernel | {'warps': 4, 'stages': 2} | 246 | 10240 | 0 | 0 | 0 | 0 | 0 |
| prepare_wy_repr_bwd_kernel | {'warps': 2, 'stages': 4} | 255 | 17408 | 4808 | 5348 | 576 | 0 | 0 |
| prepare_wy_repr_bwd_kernel | {'warps': 4, 'stages': 3} | 255 | 24576 | 1556 | 2092 | 288 | 0 | 0 |
| recompute_w_u_fwd_kernel | {'warps': 2, 'stages': 3} | 255 | 16384 | 444 | 444 | 128 | 0 | 0 |
| recompute_w_u_fwd_kernel | {'warps': 4, 'stages': 2} | 255 | 16384 | 328 | 328 | 64 | 0 | 0 |
| solve_tril_16x16_kernel | {'warps': 1, 'stages': 4} | 66 | 0 | 0 | 0 | 0 | 0 | 0 |
| solve_tril_16x16_kernel | {'warps': 4, 'stages': 2} | 40 | 256 | 0 | 0 | 0 | 0 | 0 |
