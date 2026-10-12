# contributions.md — C1–C4（G7-A 交付物，2026-10-09）

> 写作纪律：每条贡献一段成文、量化到具体对象与口径；括号内仅为数字溯源路径，
> 成稿时移入引用与脚注。四条合成论文 §1.3 的贡献列表与 Introduction 收束段。

---

## C1（方法论）PTX 级覆盖证明：编译器输出对手写 SOTA 的结构化验证契约

本文为"编译器生成的 kernel 与手写专家 kernel 同构"给出了第一个可核查的验证方法
论，而不是一句性能持平的断言。方法分两层。**T1（指令族覆盖）**：从 fla（纯 Triton
AOT 收割，22/22 配置）与 FlashQLA（TileLang 错 target 收割，5/5 kernel）的 PTX/SASS
中提取 33 族指令契约，逐族判定我方 20 配置生成物的覆盖状态；最终账目 8 YES /
19 PARTIAL / 6 NO，其中 feature 层 15 族为 8/6/2，且全部 PARTIAL 残差均为变体级
（拼写、向量宽度、代码生成旗标）而非能力缺口——包括 mma 签名层双 YES（bf16
m16n8k16：我方 552 条 vs fla 2708 / FlashQLA 2496；tf32 m16n8k8 经逃生舱翻
PARTIAL→YES，即 fla 的精确拼写）。**T2（结构匹配）**：以两个手写设计极点为靶——
fla 的 multistage/255-regs-with-spill 极点（我方寄存器包络 [28,255] 覆盖 fla 的
[40,255]）与 FlashQLA 的 warp 专化 TMA 极点（setmaxnreg dec96/inc224 将 spill 从
22,844 B 压至 48 B，5 个 TMA op 替换 1026 条经典拷贝）——两极点在旋钮格内皆可
表达、皆经硬件数值验证（G4 20/20）。覆盖契约由**逃生舱预算**封顶：五个允许的自
定义 NVVM op 仅消耗一个（`gdn.mma.tf32k8`），使"原生 MLIR 无需 fork 即可复现手写
kernel 词汇表"成为带预算的定量主张而非修辞。证据：
`evidence/G6/coverage_hatch_summary.md`（终态汇总）、`evidence/G2/coverage_matrix.md`
（逐族矩阵）、`evidence/G2/hatch_report.md`（账本）、`evidence/G4/GATE_REPORT.md`
（硬件验证）。

## C2（系统机制）编译期确定性 SMT 调度：Z3 旋钮求解 + 零拟合标定解析成本模型

本文把 TileLang 仅存在于 Ascend 后端的 Z3 modulo-scheduler 机制上移一层移植到
CUDA/sm_120：决策变量直接是旋钮向量（dtype、prec、tile M/N/K、warps、pipeline、
mma、warpspec、tma，固定声明序），约束编码结构合法性、旋钮耦合与资源容量
（smem 字节模型 ≤98 KB、寄存器 $32\,(w+w_{\text{spec}})\cdot255\le65536$），目标为
最小化解析成本模型 $T = T_{\text{launch}} + \operatorname{ov}(T_{\text{tensor}},
T_{\text{gmem}}, T_{\text{smem}}) + T_{\text{lsu}} + T_{\text{bar}} + T_{\text{spill}} +
T_{\text{ex2}}$（v3 增 waves/占用项）。模型的一切速率来自一次性 GPU 标定表
（mma.sync bf16 峰值 115.5 TFLOP/s、smem 读 113.1 TB/s、bar.sync 7.71 ns 等，单窗口
分钟级 <2 GB）或 SASS 导出的结构常数（nomma 路径 10 cyc/MAC，读自反汇编的 39
指令内环），**零逐配置拟合常数**；静态特征公式对 PTX 计数 18/18 精确。定量结果：
对 18 个实测配置的排序 Spearman **0.8962**（kkt 0.936 / o 0.829），对全部结构常数
±2× 扰动稳定于 [0.882, 0.902]、对 ±30% 计时噪声 ≥0.856；合法空间（kkt 368 / o 465
配置）以固定 seed+rlimit 逐字节确定性复现。搜索成本账目：每新形状 ~0.8 s 纯 CPU
（枚举+全解 0.378/0.444 s）对比 fla Triton 在设备 autotune 实测一次的 39.81 s
（fwd 16.55 s/25 配置 + bwd 23.26 s/23 配置，每进程每形状键重付），**≈50×**，且
我方 GPU 秒为零、决策可复现。证据：`evidence/G3/solver_report.md`、
`evidence/G3/validation.json`、`evidence/G3/calib_report.md`、`evidence/G6/search_cost.md`、
`evidence/G6/fla_autotune_cost.json`。

## C3（端到端结果）前向在全部基准形状上胜过 fla Triton，及支撑该主张的交错终判方法学

在消费级 Blackwell（RTX 5090 Laptop，sm_120a，无 tcgen05、mma.sync 为上限的架构
现实）上，我方前向（staged 三 kernel F1/F2/F3 + fused 路径，mono cubin）在与
fla v0.6.0 纯 Triton 路径（`FLA_FLASH_QLA=0`）的**同窗口交错终判**下于 3/3 基准
形状越过 1.1×：B1T2048 **1.592×**（第 4 轮 1.292× → 第 5 轮 F1UW uw 段重构
−123 µs@T8192 量级的杠杆族）；B4T2048 **1.05–1.19×**（两终判窗口完整披露——我方
中位稳定于 0.980–1.004 ms，区间宽度的全部来源是 fla 的中位方差 1.056 vs 1.165 ms，
另以受控配对 A/B 实测 −148 µs 与稳定 min 0.865–0.875 ms 交叉验证）；B1T8192
**1.253×/1.312×**（第 4 轮 1.021× 平手 → 第 5 轮过线；我方 1.126 ms）。数值契约
同时达成：4/4 配置 o ≤6.85e-4、ht ≤7.93e-4（vs fp64 JAX 参考），T8192 在分段分裂
S=1/4/8 间位级一致。方法学贡献在于判分协议本身：单窗中位数之比对有噪对手会夸大
确定性，本文的交错 per-pair 判据 + 配对 A/B + 稳定 min 三重交叉，把 B4T2048 的窗口
摆动从"我方回归的表象"正确归因为基线方差——该案例本身作为共享 GPU 环境下性能
主张的诚实性协议写入论文。证据：`evidence/G5/GATE_REPORT_FINAL.md`、
`evidence/G5/final_adjudicate_g5k_run{1,2}.json`、`evidence/G5/fwd_verify_round5.json`、
`evidence/G6/failure_longscroll.md` §10。

## C4（开放发现账本）stock-MLIR 工具链缺陷的定位、探针验证与修复——可上游化的知识产出

构建该栈的过程产出一批此前未被记录的 stock-MLIR/ptxas 缺陷，每条均以最小探针在
硬件上验证、给出根因与修复，构成论文的独立知识贡献与开源上游化清单：
(1) **workgroup memref.alloc → device malloc**：`-gpu-lower-to-nvvm-pipeline` 将
workgroup 空间分配降为设备侧 malloc 调用 + 通用指针 cvta，裸 cubin 上未解析的
CALL.ABS 蹦床使一切含 smem tile 的配置首跑即 `CUDA_ERROR_ILLEGAL_ADDRESS`——修复
为 PTX 级静态 smem 重写（`tools/gdn_smem_fix.py`），18/18 配置由死转生；
(2) **常量捕获爆炸**：tiling 把 launch 区域常量抬回宿主，kernel 参数最多膨胀至
~140 个 `.b64`（一个 bf16 pad 成为 `.param .align 2 .b8[2]`），同工具按宿主
`gpu.launch_func` 实参序逐参折叠为立即数；(3) **wait_group-0 机器约束**（源码级
证明，`NVGPU/Transforms/CreateAsyncGroups.cpp`）：`create_async_groups` 转换的每段
staging 必然紧跟 `cp.async.wait_group 0`，transform 路线结构性无法重叠预取与计算
——这解释了我们与一切后续使用者的 ping-pong 零增益；(4) **ptxas D-onto-A 操作数
腐蚀与 early-clobber 规则**：inline-PTX 的 mma 若不声明 early-clobber `=&f`，
ptxas 合法地把 D 四元组映射上 A 四元组（SASS `HMMA R12,R12,R10,R16`），K=16 链
确定性 118/128 失败；连同硬件探针导出的 fragment 布局律（D 列索引 = 2·tig，而非
B 的 groupID——错用致 1168/2080 单元欠覆盖），为 mma 内联汇编写下正确性规则；
(5) **TMA 三重接线缺口**（宿主 CUtensorMap 编码契约、坐标维序须 K 最内 + B tile
K 连续布局、`.shared` 目的强制 128B 对齐否则 `MISALIGNED_ADDRESS`）；(6) **验证
harness 的 H2D 重分级掩盖效应**（逐级重初始化中间量使 bf16-B̂ 污染三轮不可见，
教训升级为端到端验证规约）。以上与 12+6 条负结果长卷（含 tf32 误差经携带态复合、
重建混沌位点、Amdahl 反驳、async 零投递悬案）共同构成"失败保留为推进动力"的
完整账本。证据：`evidence/G6/failure_longscroll.md`、`evidence/G4/GATE_REPORT.md`
§2、`evidence/G3/calib_report.md` §2、`evidence/G2/hatch_report.md` ADDENDUM。

---

## 成稿注意事项

- C1–C4 在 Introduction 的呈现顺序即此顺序（方法论→机制→结果→知识产出），
  每条压缩为 3–4 句的量化列表项，指向本文件的全幅段落。
- 反向 0.126× + 7.2× 轨迹**不列入贡献**（避免过度主张），在 §5.5 与 §6 作为
  诚实开放战线呈现；Introduction 结果预览中单列一行并明示口径。
- 每条贡献的数字在成稿时逐一回读证据文件复核（本文件数字已对 2026-10-09 收官
  证据树核对）。
