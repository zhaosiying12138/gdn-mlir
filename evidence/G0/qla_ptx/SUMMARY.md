| kernel | regs | smem | spill | mma.sync | cp.async | tma.bulk | setmaxnreg |
|---|---|---|---|---|---|---|---|
| kkt_solve | 88 | 1024 | 0 | 64 | 10 | 0 | 0 |
| prepare_h | 128 | 2048 | 0 | 224 | 24 | 0 | 4 |
| prepare_h_s3 | 128 | 2048 | 8 | 224 | 24 | 0 | 4 |
| fused_fwd | 128 | 2048 | 1160 | 384 | 55 | 0 | 4 |
| fused_bwd | 128 | 2048 | 1098 | 1600 | 47 | 0 | 4 |
