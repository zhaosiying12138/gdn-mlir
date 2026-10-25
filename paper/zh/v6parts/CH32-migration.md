## 三十二、迁移到下一个 AI 算子：重写什么、复用什么、如何复现

迁移从语义差异开始。新算子若也有分块递推、矩阵乘和跨阶段状态，GDN-MLIR 的下降与验证骨架可以延续；数学规约、数据流和资源关系必须重新证明。可直接执行的工作簿如下，每一步都以工件完成与否作为前进条件。

### 1. 九步迁移工作簿

| 步骤 | 要回答的问题 | 可检查产出 |
|---|---|---|
| 语义规约 | 数学、形状、状态、别名、精度与边界条件是什么 | fp64 可执行规约、小形状反例、前反向输出合同；GDN 类门控须单列对数域和底数转换 |
| op/config | 调用者需要声明什么，配置合法域在哪里 | ODS、属性、trait、verifier、合法与非法 IR；诊断范围不冒充完整数值证明 |
| L1 | 能否先得到与 GPU 无关的可执行算法 | memref 或 tensor 算法 IR、CPU 生成器、与规约及参考实现的交叉误差；标注 bufferize 实际生效或为 no-op |
| 调度参数 | 哪些选择会改变 tile、流水、融合和指令形态 | 旋钮 schema、默认值、单位、非法组合、逐 stage 配置样例 |
| 约束/成本 | 合法空间能否穷尽，资源与排序如何计算 | Z3 变量与约束、unsat 完备性记录、资源公式、标定表、排序相关性与盲区 |
| 目标探针 | IR 词汇怎样落到真实指令和 ABI | 锁定 target 的 PTX/cubin、nvdisasm SASS、ptxas 资源、最小硬件探针；内联扩展附操作数布局证明 |
| 验证矩阵 | 哪些形状、类型、种子与执行方式必须通过 | 前反向数值、容差标定、双种子、fresh-process 逐位一致性、sanitizer 与失败样本 |
| 性能与边界 | 哪个窗口支持什么结论，负结果由什么造成 | 30 对交错 JSON、两个加速比、IQR/decidable、双窗口、冷启动、流量或融合边界 |
| 证据包 | 第三方怎样从命令回到原始产物 | 环境版本、命令、配置哈希、原始日志、报告与证据索引；每个数字可追到文件行或 JSON 字段 |

工作簿还要贴合当前实现的真实边界。`gdn-to-chunk` 目前只把 `gdn.fwd/bwd` 改为 `func.call @__gdn_l1_*`，透传 config 并复用函数声明；完整 L1 文本由 `tools/gdn_l1_gen.py` 生成。迁移时可以复用入口改写模式，不能据此宣称 pass 已自动展开全部 chunk AST。[dialect/lib/GdnPasses.cpp:23-90；scripts/repro.sh:248-260]

当前 L1 已是 f32、原地状态的 memref 级 CPU 标量 IR，`run_l1.sh` 中的 one-shot-bufferize 对这类输入实际为 no-op，保留它是为了将来的 tensor frontend。[scripts/run_l1.sh:23-37] 数值域也要写入规约：外部 GDN 语义为 decay=e^g 且 g≤0，内部先计算 g_log2=g·log2(e)，再调用 `math.exp2`；两种底数不可混写。[docs/L0_SEMANTICS.md:7-20；tools/gdn_l1_gen.py:16-20,223-236]

读这张表应沿依赖方向检查：语义规约没有闭合，后续 PTX 只能证明发射；资源公式未对账，求解结果再快也不能进入生成；缺少原始样本，性能表无法复审。每行留下的产出恰好说明当前可以主张什么，也暴露尚不能主张什么。

### 2. 复用按机制分类

| 原样复用 | 参数化复用 | 必须重写 |
|---|---|---|
| MLIR 的 rewrite、canonicalization、Dialect Conversion 与 PassManager 基础设施 | gdn-opt 注册、ODS/config 外壳、CPU/GPU pass 管线 | 新 op 的数学语义、状态合同及 verifier 中的领域规则 |
| Z3 固定序阻塞枚举方法、符号检验与配对统计算法、resource_guard | 旋钮声明、标定表、形状分派、收割 kernel 清单与目标架构 | L1 算法体、stage 划分、全局读写边、生存期池及相应的资源公式 |
| nvdisasm、ptxas、CUDA event 与证据目录的组织方式 | transform payload、验证 harness 的形状/dtype/容差参数、T1/T2 参考配置 | 新精度或布局的 fragment 映射、误差预算、参考基线、性能下界与融合价格 |

“原样”指机制及其通用实现可以直接采用，“参数化”仍要求检查单位、默认值和假设是否适用。尤其是成本模型：硬件标定可以沿用测量方法，操作数流量、复用次数和驻留关系随算子改变。复用比例无需预先承诺，三类清单本身就是迁移设计的验收材料。

求解范围也应保持分层。当前 Z3 负责 stage-kernel 的十维旋钮合法域，成本模型负责候选排序，完整算子的最终选择还使用实测 A/B。SAT 或枚举完备只证明所声明变量与约束下的合法性，不能替代端到端最优性证明。迁移新增旋钮时，应同步更新声明域、阻塞枚举、资源公式和实测复核，避免只改生成器而让求解器继续接受旧空间。[tools/gdn_solver.py:163-174；README.md:258-294]

### 3. G0–G6 复现入口

```bash
cd ~/codebase/gdn-mlir
make status
make repro-list
make repro-g0
make repro-g1
make repro-g2
make repro-g3
make repro-g4
make repro-g5
make repro-g6
# 以下命令替代逐门执行，首个失败即停止
make repro-all
```
[Makefile:27-55；scripts/repro.sh:346-412]

读命令先看依赖：G0 准备环境、LLVM、sm_80 上游冒烟与两家收割物，其中冒烟在芯片重定向为 sm_120a 后于 5090 运行，用来验证同 mma.sync 指令族的工具链；G1/G2 可在 CPU 复核语义和覆盖，G3 依赖标定与设备计时，G4 验证配置，G5 才进入全算子测量，G6 复核搜索成本。`repro-all` 的顺序保留这些依赖，逐门日志和 JSON 判据比进程退出码更能说明通过原因。[scripts/repro.sh:225-345]

参考环境为 WSL2、conda 环境 gdn、CUDA 13.x，以及计算能力至少 12 的 NVIDIA GPU；纯 CPU 门可在没有该 GPU 的机器上先运行。部分驱动会更新规范证据路径，恢复或对比旧数字前要保存当前工作树，不能把“幂等”误解为所有历史文件都永不覆盖。[scripts/repro.sh:23-38；README.md:418-430]

资源纪律与命令同等重要。并行度按空闲内存每作业 2.5 GB 估算并限制在 4–12；watchdog 在空闲低于 6 GB 时暂停、回升到 8 GB 时恢复、低于 3 GB 时终止己方任务；全局重构建使用单一槽位。GPU 工作先登记窗口，正确性短跑可共享且控制在 2 GB 内，正式计时使用空闲窗口，绝不终止他人进程。持久产物留在仓库文件系统，长任务按 stage 与内容哈希断点续跑。[scripts/resource_guard.sh:14-57；CONTRIBUTING.md:3-12]

### 4. 八项结课实践

| 实践 | 任务 | 可检查产出 |
|---|---|---|
| 1. 方言 | 定义一个小算子、config 属性与 verifier | ODS、合法/非法 IR、诊断文本；逐项说明 verifier 未检查的语义 |
| 2. rewrite/conversion | 写一个局部 rewrite，再把自定义 op 渐进转换到标准方言 | pass 源码、前后 MLIR、未匹配反例、转换后类型与合法性检查 |
| 3. CPU pipeline | 用 linalg/scf/memref 实现数值核并执行 CPU 管线 | 分阶段 IR、JIT 结果、fp64 误差；明确 bufferize 的真实作用 |
| 4. transform | 用 payload/handle 写出两种调度 | 两份 transform 工件、对应 MLIR 差异、tile/流水参数与资源变化 |
| 5. Z3 | 建模旋钮、资源约束并以阻塞子句穷举 | 变量表、约束、unsat 日志、确定性枚举；no-MMA 示例须按 f32 累加器核对字节与 strict 上限 |
| 6. GPU MMA | 从 GPU IR 到 PTX/SASS，验证一种 MMA 布局 | 精确 mma 签名、SASS、ptxas 寄存器/共享内存/溢出、整型 fragment 硬件探针 |
| 7. 数值—PTX—资源闭环 | 改一处精度或布局并同时审计三类后果 | 数值误差、PTX/SASS diff、资源 diff、双种子与 fresh-process 哈希，形成同一证据包 |
| 8. 迁移 | 选择一个结构相近算子走通九步工作簿 | 语义到证据包的完整目录、三列复用清单、T1/T2、双窗口结果和至少一项边界分析 |

这些任务覆盖 AI 编译器开发中最常用的核心能力，不把“80%”写成对上游全部 API 的虚假覆盖率。完成第 1–4 项即可独立走通方言、rewrite/conversion、CPU pipeline 与 transform；第 5–8 项再进入 Z3、GPU、数值—资源—性能联合审计和迁移。每项都以可检查产物验收，不以读过的 API 名称验收。

第五项练习可用一个真实合法配置检查模型与实现：o、bf16、K=192、V=128、tile.M=64、tile.N=128、tile.K=64、mma=off。当前生成器的 no-MMA 累加器为 f32，按当前实现应为 106,496 B；旧资源公式为 90,112 B，会把该点误放进 100,352 B 的 strict 上限，而生成器的 220 KiB 上限仍可接受。发现这类差异应先修正模型，再解释合法空间计数。[tools/gdn_solver.py:147-160；tools/gdn_l2_gen.py:130-138,769-778]

### 5. 血统、对照与许可

机制来源与参照关系需要分开。fla 提供 Triton 性能基线和 T1/T2 一极，FlashQLA 提供 TileLang 手写 warp 专化的另一极；两者都是外部对照实现。TileLang 的 Ascend 后端带来 Z3 编译期调度机制，本仓把决策变量从指令时槽上移为旋钮向量。XLA/TPU 提供语义单一事实源、编译期确定调度与分层验证的哲学参照。CAKE 以 agent 写 IR 和自改进闭环为代表，提供方法对照；本仓采用约束求解与确定性证据链，不据此主张代码血统。[README.md:432-443]

本仓代码、文档与证据树采用 Apache License 2.0。[LICENSE:1-6；README.md:450-452] fla 与 FlashQLA 均为 MIT 许可，在此只作对照基准，归档的 PTX 收割物不含其源码。再分发它们的源码、修改或相关声明时，仍需遵守各自许可并保留版权与许可文本；本仓的 Apache-2.0 标识不会改变外部项目的许可。迁移所得证据包应同时记录上游版本、补丁和许可来源，使技术血缘与法律边界都可回查。
