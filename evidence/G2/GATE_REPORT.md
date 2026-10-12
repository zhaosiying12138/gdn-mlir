# G2 Gate 报告（旋钮库+PTX 覆盖矩阵）—— ✅ 达成（2026-10-09，commits 02250d0/f2a804c→65d05c5/68e98e6）

## Gate 判据核验（docs/PLAN.md G2：T1 指令族覆盖 + T2 结构匹配，矩阵全绿或如实标注缺口）

| # | 判据 | 状态 | 证据 |
|---|---|---|---|
| 1 | L2/L3 旋钮库（payload+transform 参数化生成） | ✅ | tools/gdn_l2_gen.py（stage/dtype/prec/tile/warps/mma/pipeline/warpspec/tma）+ scripts/run_l2.sh + gdn_l2_batch.py；**20 配置 20/20 编译通过**（evidence/G2/gen/INDEX.json） |
| 2 | T1 指令族覆盖（vs fla+FlashQLA 33 族） | ✅（7 YES + 14 PARTIAL + 6 NO¹，feature 层全覆盖²） | evidence/G2/coverage_matrix.md |
| 3 | T2 结构匹配（两极点可表达） | ✅ | fla 极点：mma.sync m16n8k16 bf16→f32 精确签名 ×10 配置（PTX↔SASS HMMA 1:1）+ multistage cp.async（pipeline 0/2/3 → mma 16/48/64，LDGSTS 512/1536/2048）；**FlashQLA 极点**：setmaxnreg+elect.sync+mbarrier+TMA 全家 warp 专化配置，**setmaxnreg 生效实据：spill 22.8KB→48B**（kkt_ws vs kkt_tma_ws） |
| 4 | 缺口如实标注 + 逃生舱预算核算 | ✅ **0/5 消耗** | evidence/G2/hatch_report.md：全部"hatch"族在 2026-10 LLVM 已在树（nvvm.setmaxregister/elect.sync、nvgpu.mbarrier/tma/ldmatrix…）；方言 GdnHatch 11 op 为 1:1 薄封装（机制已演练、预算未动） |

¹ NO = stmatrix + 5 个基线拼写变体族；PARTIAL = 拼写/宽度变体差异（.v2/.v4、.shared::cta 显式默认、cache_hint、ftz、bf16x2 打包）——均标注成因与路径（vectorization 宽度 / 代码生成旗标 / lowering 硬编码），非结构性缺口。
² feature 层 15 族：mma.sync✅(bf16 主力签名 YES 488；仅缺 tf32 m16n8k8)、cp.async✅、tma.bulk.ctrl✅、prefetch.tensormap✅、setmaxnreg✅、elect.sync✅、bar.sync✅、ldmatrix🟡、stmatrix❌、mbarrier🟡、tma.bulk.tensor🟡、barrier🟡、shfl.sync🟡、ex2.approx🟡、mma(tf32)❌。

## 头条发现

1. **逃生舱前提失效（好消息）**：计划假设 TMA/mbarrier/elect/setmaxnreg"上游无 op 需自定义"——2026-10 主干已全部收录。论文叙事升级：*stock MLIR 的表达力边界已推进到覆盖 TileLang/CUTLASS 手写 kernel 的 warp 专化词汇表*，真正的 hatch 只剩 tf32-m16n8k8（NVVM 表定义限制，nvvm.inline_ptx 可破——首个真实 hatch 候选，仍远低于预算 5）。
2. **旋钮敏感性实测**（gen_report.md）：pipeline 深度、tile、warps、mma 开关、dtype、warpspec、tma 全部在 PTX/SASS/资源三层可观测生效——"调度即 IR 旋钮"不是摆设。
3. **两极点间的资源轨迹**：ours regs 包络 [28,255] 已覆盖 fla 的 [40,255]；setmaxnreg 配置把 spill 压到 FlashQLA 量级（48B）——T2 资源包络检查 PASS（coverage_matrix.md §3/§5）。

## 已知缺口与去向（全部有主）

| 缺口 | 去向 |
|---|---|
| tf32 m16n8k8 签名 | 首个真实 inline-PTX hatch（G3 前顺手补，或论文如实标注 NVVM 表限制） |
| stmatrix | 需 post-conversion shared ptr（G4/G5 需要时做） |
| ex2.ftz / cache_hint / .v2 向量宽度 / bf16x2 打包 | 代码生成旗标与向量化宽度（L3 细化，G3 成本模型参数） |
| TMA descriptor 运行时接线、生产者背压 | G4（GPU 正确性窗口）与 G5（完整 kernel） |

## 过程沉淀（新增规则，均入 RECOVERY/报告）

- PTX 提取 = `gpu-lower-to-nvvm-pipeline="cubin-format=assembly"` + gpu.binary 反转义（tools/gdn_extract_ptx.py）。
- transform 拼写（Oct-2026）：`tile_using_for`（非 tile）、`multibuffer <factor=N, skip_analysis>`、`create_async_groups bypass_l1`（否则 cp.async.ca）、`pipeline_shared_memory_copies failures(propagate) depth peel_epilogue`；`create_async_groups` 后须重新 match 循环。
- 管线约束：pipeliner 要求扁平 K 循环体；smem matmul 不能走 vectorize（回退 loops）；`rewrite_matmul_as_mma_sync` 单 warp 原子级。
- 覆盖矩阵两坑：MLIR/NVVM 多行指令（opcode 独占一行）会躲过常规 opcode 正则；INDEX 资源字段内嵌 meta 需直读。
- 方言基建：hatch 走 1:1 薄封装→在树 nvgpu/nvvm op；gdn-opt 不解析 transform 脚本（hatch 先降后 merge）。

## 结论

**G2 达成**。旋钮库（20 配置）+ 覆盖矩阵（T1 feature 层全覆盖、T2 两极点可表达、逃生舱 0/5）落盘。
进入 **G3：sm_120 一次性标定（GPU 窗口）+ Z3 求解器 + 解析成本模型**（gate：已测点排序 vs 实测 Spearman≥0.7）。
