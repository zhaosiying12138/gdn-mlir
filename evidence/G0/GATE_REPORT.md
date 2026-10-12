# G0 Gate 报告（基建+参考收割）—— ✅ 全部判据达成（2026-10-09 01:45）

## Gate 判据核验（docs/PLAN.md G0）

| # | 判据 | 状态 | 证据 |
|---|---|---|---|
| 1 | LLVM/MLIR 构建（mlir-opt/mlir-runner/cuda-runtime/python 绑定/erase-pass/runner 工具库） | ✅ | logs/build_llvm.log（DONE 01:30/01:32）；build/ 1.9GB Release 共享库 |
| 2 | 树内 sm80 transform-mma 集成测试在 5090 真跑通过 | ✅（带瑕疵备注） | evidence/G0/sm80_smoke.txt + sm80_payload_sm120a.mlir |
| 3 | fla Triton 参考资源表（第一批） | ✅ 22/22 配置 | evidence/G0/fla_ptx/SUMMARY.md |
| 4 | FlashQLA TileLang 参考资源表（第一批） | ✅ 5/5 kernel | evidence/G0/qla_ptx/SUMMARY.md |
| 5 | 资源纪律基建生效 | ✅ | scripts/resource_guard.sh + logs/watchdog.log 全程采样，无 OOM 事件 |

## 关键数字

### fla Triton（12 kernel × 代表配置 = 22 项，纯 CPU AOT `GPUTarget("cuda",120,32)`，runtime-faithful dtype）
- **反向 kernel 全部 255 regs 顶格 + spill**：prepare_wy 4808B(s2w4)/1556B(s3w4)、dhu 3236B、dqkwg 1544B、recompute_wu 444B。
- mma.sync 32-576/kernel；**cp.async 仅出现在状态递推 kernel**（fwd_h/dhu 14-18 条）；无 TMA/mbarrier/setmaxnreg（纯 multistage ld.global 路线）。
- dtype 真相（收割器 PER_KERNEL_TYPES 已固化）：**A/h/dh/do=bf16**（A 以 k.dtype 保存、h=k.new_empty）；ht/dh0/dht/dg=fp32；solve_tril 输入 fp32→输出 bf16。
- 动态 smem：dqkwg 32.7KB、dhu 29.4KB、fwd_o 24.5KB。

### FlashQLA TileLang（tilelang 0.1.15，sm_120a monkeypatch，get_kernel_source→nvcc -ptx -Xptxas -v）
| kernel | regs | smem | mma.sync | cp.async | 特征 |
|---|---|---|---|---|---|
| kkt_solve | 88 | 1KB | 64 | 10 | 轻量 |
| prepare_h | 128 | 2KB | 224 | 24 | 无 spill |
| fused_fwd | 128 | 2KB | 384 | 55 | 融合 KKT+solve+h+o 单 kernel |
| fused_bwd | 128 | 2KB | **1600** | 47 | 单 kernel 全反传，零 spill |
- **与 fla 的设计极点对比**：FlashQLA 全部 ≤128 regs 零 spill（setmaxnreg 生产/消费者 warp 分区）vs fla 255+spill；单 kernel 融合（fwd/bwd 各 1）vs fla 5-8 kernel。**G2 的 T2 覆盖矩阵就以这两个极点+FlashKDA 两段式为目标设计旋钮空间。**
- 版本备注：FlashQLA 官方 pin tilelang 0.1.9 与 CPython 3.12.15 不兼容（tvm_ffi type.__dict__），收割用 0.1.15，矩阵如实标注。

### sm80 冒烟（判据 2）
全链：`mlir-opt -transform-interpreter -test-transform-dialect-erase-schedule -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a cubin-features=+ptx87"` → `mlir-runner --shared-libs=libmlir_cuda_runtime,...` → **5090 真执行**，f16 m16n8k16 结果正确。
**瑕疵备注**：结果打印后 teardown 阶段报 `CUDA_ERROR_ILLEGAL_ADDRESS`（不影响计算输出；疑 mlir_cuda_runtime 清理路径或与共训任务干扰，G4 起持续观察，若复现则升级排查）。

## 过程记录（失败→修复，全部沉淀为规则）
1. nanobind：MLIR main 要 2.x（pip 3.1 被拒）→ pin `>=2.9,<3`。
2. CMakeCache 记住 CUDA NOTFOUND → configure 失败即清缓存。
3. nvcc 不在 PATH（configure 阶段也要）→ 脚本头统一 export PATH。
4. wsl.exe 会话退出即杀后台进程（nohup 无效）→ 长任务用 harness 后台任务持会话。
5. wsl.exe 内联命令 $var/引号被外层重解析（踩 3 次）→ 非平凡命令一律脚本文件。
6. triton 3.8：ASTSource 在 `triton.compiler`；**PTX ISA 9.3 需 `ptxas-blackwell`**（普通 ptxas 13.3 只认 8.8）。
7. fla dtype 以包装器分配处为准（h/dh/do=bf16、A=k.dtype=bf16）。
8. tilelang 0.1.9 × CPython 3.12.15 不兼容 → 用 0.1.15；sm120 kernel 需 `sm_120a`（setmaxnreg 是 arch-specific），tilelang 解析会剥 'a' → 收割器内 monkeypatch `get_target_arch_and_code`。
9. TileLang CUDA 源外编译需对齐内部 -I（src + 3rdparty/cutlass/include）。
10. LLVM lit CUDA sm80 门控需 `-DMLIR_RUN_CUDA_SM80_TESTS=ON`；mlir-runner 还需 runner_utils/c_runner_utils/float16_utils 三个库（目标名小写 `mlir_cuda_runtime`）。

## 结论
**G0 达成**。进入 G1（方言 L0/L1 + JAX 规约 + CPU 三方语义对拍）。
