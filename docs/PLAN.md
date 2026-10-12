# GDN-MLIR 总计划（2026-10-09 已批准，单一事实源）

> 对计划的任何变更先改本文件并 commit，再动代码。

## 0. 愿景、定位与论文主张

终极交付 = 顶会级 AI systems 论文 + 开源仓。核心主张：**不手写 PTX、以 transform dialect 为调度 IR 的 stock MLIR + SMT 约束求解 + 校准解析成本模型**，在 GDN（线性注意力代表算子）上做到：超 fla Triton fwd/bwd、fwd 冲 FlashQLA、sm_120 首个快反向，且用 **PTX 级覆盖证明**（T1+T2）证明旋钮空间覆盖各家手写设计。空白：FlexLA/TileLang/Hexcute/TileLoom/CAKE/PEAK 均未做"调度即 IR 的 stock-MLIR 渐进下降 + SMT 编译期调度 + PTX 覆盖验证"。

**对标 Google TPU/XLA 栈四支柱 → 对应物**：
1. 语义单一事实源（HLO+autodiff）→ `gdn` 方言 L0 + **JAX 可执行规约**（jnp chunk 前向 + `jax.vjp` 推导反向，fp64 紧验证）
2. 编译期确定性调度（非运行时 autotune）→ **Z3 modulo-scheduler（TileLang Ascend 机制移植到 CUDA/sm_120）+ 一次性硬件标定表**（一个 GPU 窗口采集 sm_120 解析延迟参数，之后全离线求解，设备只验 top-K）
3. 分层可验证 IR → L0-L4 每层 oracle/verifier/lit 回归
4. 布局系统 → smem/寄存器 layout 旋钮族

CAKE 自改进闭环：失败 → 约束规则库/IR 原语（BasicPtxBuilder）/归因表。**escape-hatch 预算**（自定义 NVVM op 数）作为"stock MLIR 极限"论文指标，目标 ≤5。

## 0.5 资源纪律与 WSL 崩溃恢复（铁律，见 CONTRIBUTING.md，优先级最高）

线程预算 clamp(空闲GB÷2.5, 4, 12)、watchdog STOP/CONT/TERM、全局唯一重构建槽位、实验限线程、幂等分 stage、产物只落盘、RECOVERY.md 恢复、GPU 协调协议并行生效。

## 1. 工作模式

- **CPU-first**：Phase A 全程无 GPU——PTX 覆盖证明 + Z3 求解 + 静态排序（ptxas `-v` + SASS 指令组合，纯 CPU）；Phase B 只用 GPU 机会窗口（正确性共享跑 <2GB；标定一次；计时只排空闲窗口，遵守 GPU-COORDINATION.md）。
- **goal mode**：G0-G7 每个 goal = spec + exit criteria + 证据落盘，gate 间自主迭代，只在 gate 通过/需决策时汇报。
- **多 agent 并行**：WS0-WS5，CPU 预算主会话统配。

## 2. 分层 IR 与旋钮空间

- **L0 语义**：`gdn.fwd`/`gdn.bwd`（q,k,v,β,g[,h0]→o,ht[,dq,dk,dv,dβ,dg,dh0]）；属性：dtype/chunk_size/use_g/use_gk/h0/l2norm-in-kernel/β-sigmoid/varlen/GVA。
- **L1 算法结构**（与 FLA 同构阶段 DAG）：KKT→三角求逆→w/u→h 递推（scf.for 载 h）→o；bwd：dh 反递推→重算 h/w/u→dqkwg→dv/dv_local→wy_bwd。旋钮：chunk∈{16,32,64}、solve 变体（前代换/Schur 合并/融合 KKT+solve）、recompute-vs-materialize、decay 域(log2/exp2)、state 精度(fp32/bf16)、gate 融合位置。
- **L2 调度**（transform 脚本=参数化工件）：BT/BK/BV/BC、grid 映射（rows×V-split×可选 token-parallel WY 预段=FlashKDA 两段式，82-SM 并行度适配）、num_warps、**融合边界**（单 kernel…FLA 6 kernel 任意切分；旗舰=fwd/bwd 各严格 1 launch persistent）。
- **L3 流水/向量化**：`transform.memref.multibuffer(factor)`、`transform.nvgpu.create_async_groups`/`rewrite_copy_as_tma`、`pipeline_shared_memory_copies(depth)`、`transform.loop.pipeline`、`transform.nvgpu.rewrite_matmul_as_mma_sync`（m16n8k16 bf16→fp32；sm_120 无 tcgen05，mma.sync 即上限）、手动 warp 专化（warp 谓词+setmaxregister，NVDSL Ch5 模式）、swizzle。**只做 sm_120**。
- **L4 目标**：`gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a"`。
- FLA/FlashKDA/FlashQLA 风格 = 配置空间中三个点，复现=调参不重写。

## 3. Z3 求解器（TileLang 机制 B 的 CUDA 移植+创新）

- **变量**：knob 向量（tiles、warps、stages、buffer 多版本数、融合边界、warp 分区）。
- **约束**：chunk 递推依赖链、流水线资源互斥（LSU/MMA/SFU）、smem=f(tiles,stages)≤99KB/CTA、regs≤255/thread·64K/SM、mma fragment 对齐、chunk 对角块结构、TMA 对齐。
- **目标**：最小化解析模型估算延迟（标定表×静态流量/占用模型；用户注解可覆盖）。
- **确定性**：rlimit+固定 seed。**回灌**：解→transform 脚本实例化→编译→ptxas -v/SASS 静态复核→（窗口期）设备计时；失败→规则库。
- 论文对比：vs fla_cache_autotune（运行时试跑）与 TileLang CUDA（无求解器）——"SMT+标定模型的编译期调度用 ~1/100 GPU 时间达到 ≥ 设备 autotune 的配置质量"。

## 4. 里程碑 G0-G7（全 gate 制，证据落盘）

- **G0 基建+参考收割（1-2 天，纯 CPU）**：LLVM worktree+受限构建；conda env gdn（torch cu130/fla[cuda]/z3/jax[cpu]）；watchdog/RECOVERY 基建；参考 PTX 收割：fla 经 Triton AOT `GPUTarget("cuda",120,32)`（自带 ptxas `-v`）、FlashQLA 经 TileLang 锁 target+`get_kernel_source()`+nvcc -ptx、（第二批）FlashKDA `FLASH_KDA_CUDA_ARCHS=120a`、kingsleykimm nvcc -ptx；SASS 特征提取器（nvdisasm）。**Gate**：第一批（fla+FlashQLA）PTX/SASS/ptxas 资源表落盘；树内 sm80 transform-mma 集成测试跑通。
- **G1 方言+语义（2-3 天，纯 CPU）**：L0/L1 ODS+展开 pass；JAX 规约（chunk fwd + vjp bwd，fp64）；CPU linalg 全梯度对拍（JAX fp64 1e-6 级 + fla naive fp32 1e-4 交叉）。**Gate**：三方语义一致。
- **G2 旋钮→PTX 覆盖矩阵（3-5 天，纯 CPU，核心）**：T1=指令族覆盖（mma.sync 形状/精度、cp.async/TMA、mbarrier、setmaxregister、exp2.approx 等）；T2=结构匹配（每参考关键 kernel 存在旋钮组合使 mma 形状/异步模式/流水结构等价、ptxas 资源包络±30% 同档）。第一批 fla+FlashQLA，第二批 FlashKDA+kingsleykimm。**Gate**：矩阵全绿（缺口→补 IR 原语[计 escape-hatch 预算]或如实标注）。
- **G3 标定+求解器（3-4 天，CPU+1 个 GPU 窗口）**：sm_120 标定 microbench（一次窗口，分钟级，<2GB）；Z3+解析成本模型+静态排序 → per-shape top-K。**Gate**：已测参考点排序 vs 实测 Spearman≥0.7 起步。
- **G4 GPU 正确性（1-2 天，共享窗口短跑）**：多 kernel 与单 kernel 路径 bf16 全过 fla 容差（o/ht 5e-3、dq/dk/dv 8e-3、dβ/dg 2e-2 vs naive fp32 参考缓存）。**Gate**：主套件+GVA 扩展全过。
- **G5 GPU 计时（2-4 天，排窗口）**：e2e 超 fla fwd/fwdbwd（`FLA_FLASH_QLA=0` 纯 Triton 口径，双口径记录）；单 kernel 旗舰；fwd 与 FlashQLA SM120 横比。**Gate**：fwd/bwd ≥1.1× fla；未达则归因迭代（允许多轮）。
- **G6 消融+证据链（2-3 天）**：knob 消融、搜索成本 vs Triton autotune、覆盖矩阵表、escape-hatch 预算、失败路径长卷。
- **G7 论文+开源收官**：论文骨架（MLSys/ASPLOS/CGO）、复现脚本、文档。

## 5. 并行工作流（多 agent）

WS0 基建｜WS1 参考 PTX 收割+特征提取器｜WS2 方言+L1+JAX 规约｜WS3 L2/L3 旋钮库+覆盖矩阵｜WS4 Z3 模型+成本模型｜WS5 GPU 窗口执行器。WS2/WS3 依赖 G0 的 LLVM，其余可齐头。

## 6. 风险与对策

1. PTX 匹配≠性能（必要非充分）→ 静态覆盖为 gate、性能以窗口计时为准。
2. MLIR sm_120 mma.sync 成熟度 → G0 即跑树内 sm80 集成测试（同指令族）。
3. 单 kernel bwd 寄存器压力 → recompute-vs-materialize 旋钮退化。
4. Z3 排序不准 → 标定迭代+注解逃生舱；排序质量是 G6 消融素材（负结果入长卷）。
5. GPU 窗口稀缺 → 正确性共享跑、标定一次、计时 top-K。
6. CPU/内存争用 → §0.5 铁律。

## 7. 关键已核实事实（10-08/09 调研）

- sm_120 无 tcgen05（消费级 Blackwell 用 Ampere 风格 mma.sync，cluster 1x1x1，99KB smem/CTA opt-in，82 SM）。
- FLA v0.6.0 阶段结构（fwd: KKT→solve_tril→w/u→fwd_h→fwd_o；bwd: dhu→重算→dqkwg→dv/dv_local→wy_bwd）；Blackwell bwd 钉 warps=2/stages=4（issue #913）；sm_120 无快 GDN 反向（FlashQLA SM120 仅 fwd bf16）。
- 纯 CPU 编译路径全部核实：Triton AOT（GPUTarget 无 GPU、自带 ptxas 恒 -v）、TileLang 锁 target+get_kernel_source、FlashKDA 需显式 FLASH_KDA_CUDA_ARCHS、kingsleykimm 头文件 nvcc -ptx、ptxas -v 字段（regs/smem/spill）、nvdisasm SASS。
- TileLang Z3 双机制：TVM 分析器可证性（全后端）+ Ascend-only modulo-scheduler（变量=start 时间+buffer 版本、约束=依赖+pipe 互斥+容量、目标=min II、estimate_latency 解析模型、二分搜 II、解回灌 buffer_versions）；CUDA 侧与 autotune 均无 Z3 → 我们的移植即贡献。
- NVDSL=mlir/test/Examples/NVGPU（Python-on-MLIR JIT 模板）；mma 管线配方=multibuffer+async groups+pipeline_shared_memory_copies+rewrite_matmul_as_mma_sync；集成测试样例 mlir/test/Integration/GPU/CUDA/TensorCore/sm80/transform-mma-sync-matmul-*.mlir。
