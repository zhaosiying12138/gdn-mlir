
# v6 教程写作合同（内部文件，不进入发布正文）

标题：从一段 IR 到 GPU 内核：GDN-MLIR 实战教程
副标题：用真实的 Gated DeltaNet 编译器理解 MLIR 方言、pass、结构化 IR、调度、Z3 与 NVVM

读者路径：IR 读写 → 方言/pass → GDN 数学与 CPU L1 → linalg/scf/memref/转换 → transform 调度 → Z3/成本模型 → GPU/NVVM/PTX → 机器优化 → 验证/覆盖/性能/边界/迁移。

总量目标：32 章，正文约 6–9 万中文字（不含代码），每章一个核心能力。宁可少而深，不按固定模板凑字。每章 1400–2800 中文字，复杂章可到 3500。每章 2–5 个真实展品，代码后必须解释“看什么、为什么、能判断什么”。

## 章节分配（文件名即新章号）
- CH01-04：IR/SSA/Region；方言 ODS/属性/trait；verifier/诊断/测试；rewrite/greedy/canonicalization/CSE/DCE/folding。
- CH05-08：PassManager/Analysis/可观察性；外建工程与 gdn-opt；GDN 数学；L1 可执行规约与三方闭合。
- CH09-12：linalg；scf/cf；tensor→memref 与 11-pass CPU 管线；Dialect Conversion 与渐进下降。
- CH13-16：transform dialect；生成器/十维参数/端到端配置；Z3 变量/约束/合法域；阻塞枚举/防御/确定性。
- CH17-20：标定与成本模型；排序质量与搜索成本；GPU/NVVM 下译；nvgpu/NVVM op 合同。
- CH21-24：inline asm 与唯一扩展；PTX 后处理四变换；ptxas/SASS/sanitizer 审计；前向三 kernel 与瓶颈。
- CH25-28：反向扫描与仿射携带；数据通路/存储几何/驻留；split-bf16；验证塔与容差。
- CH29-32：T1/T2 覆盖；测量与完整性能；反向精度边界；迁移、复现、相关工作与结课实践。

## 递进原则
1. 每章开头直接进入一个技术对象、现象或问题，禁止“本章解决……”“接下来我们将……”等元话语。
2. 先建立概念，再给真实实现；概念不能只在结尾补定义。每个新术语首现给中文+英文（必要时缩写）。
3. 不把每章机械写成“存在意义/直觉/机制/效果”四段。可以用 IR 反例、手算、前后对照、协议状态机、资源账或故障排查自然推进。
4. 重要代码必须真实，标 `[路径:行号]`；省略处明确。通用 MLIR 小例可新写，但标【通用补充】或【练习】，不得冒充项目代码。
5. 区分四类内容：项目实现、上游 pass 的真实行为、通用 MLIR 补充、归档测量。不得把项目未写的 Analysis、DialectConversion、canonicalization、FileCheck 套件说成项目已有。
6. 禁止轮次流水账、黑话、自创词、emoji、空泛宣传、绝对化“全部/彻底/完美”；禁“不是 X，而是 Y”句式。直接写结论和条件。
7. 保留必要的负结果与边界，但按机制解释，不写过程流水账。
8. 小节使用 `### 1.`、`### 2.` 递增；章标题使用 `## 一、` 等中文数字。表格内不放 LaTeX/上下标公式，公式放表外代码块或 Unicode 文本。
9. 交叉引用只用“第 N 章”（中文数字）且必须核对新章号；图占位用 `{{FIG1}}` 等，只放在指定章节。不要自己编图号。

## 事实锁定（必须遵守）
A. g 的自然对数域：外部语义 decay = exp(g)，g≤0；内部按 fla 习惯把 g_log2 = g*log2(e) 后用 math.exp2。不要再写“外部直接 2^g”。如需简式，写 e^g 并说明内部 exp2 实现。
B. 当前 gdn-to-chunk 只做 gdn.fwd/bwd → func.call @__gdn_l1_* 的入口改写、config 透传和声明去重；完整 L1 由 tools/gdn_l1_gen.py 生成。L0-L4 是设计与验证分层，不要声称所有生产路径都由该 pass 自动展开到 cubin。
C. 当前 L1 已是 memref 级 f32/原地状态/CPU 标量 IR；run_l1.sh 的 one-shot-bufferize 对该输入实际 no-op，为将来 tensor frontend 保留。memref 的 SSA 值唯一不禁止底层内存别名；PTX 生存期别名是项目选择，不是 MLIR 原理禁令。
D. verifier 只验证部分结构/配置合法域，不证明完整 shape、g 符号、归一化或数值语义；数值由 JAX fp64、fla naive、MLIR L1 三方交叉及后续 GPU 验证负责。
E. split-bf16：bf16 与 fp32 都是 8 位指数，差异在尾数精度；hi+lo 保留更多尾数，lo·lo 省略，残差约 2^-17 量级；两条 bf16 不节省存储流量，不称“免费 fp32”。tf32 尾数 10 位（含隐含位口径按文章统一），不要把问题写成指数域截断。
F. 终态反向链按当前生成器/README 为 8-kernel（wy、fwdscan、caryM、bpre、dhs、dkb、dqkwg、wyb）；历史报告出现 9-kernel 时注明是旧状态，不混用。
G. Z3 的范围是 stage-kernel 的 10 维旋钮空间，完整算子最终决策还使用实测 A/B；0.8s CPU/形状对 39.81s 设备首搜的对比只针对同一候选空间的排序成本。保留 Spearman 0.8962、n=18、扰动与绝对误差边界，不神化模型。
H. 求解器与生成器存在 no-MMA 资源公式对账问题：solver/costmodel 用 `(bt*K + K*bv + bt*bv)*esz`，当前 generator 的 no-MMA accumulator 为 f32，应把最后一项写为 `bt*bv*4`。例如 K=128、bt=bv=128 时旧值 98,304B，按当前生成器为 131,072B，超过 strict 100,352B。不要把 368/465/204 写成当前所有路径无条件完备的合法空间；可写成 G3 报告记录，并将该差异作为审计练习。若主会话后续提供修正计数，以新计数为准。
I. 性能表使用 JSON 的两个不同估计量：speedup_ratio_of_medians 与 speedup_median_of_ratios。共载正式 B1T2048=2.1250/2.1174、B4T2048=1.2211/1.2029、B1T8192=1.3493/1.3307；三者 stats.decidable=false。空闲 clean3：B1T2048=1.6106/1.6140 且 decidable=true，B4T2048=1.0669/1.0719 且 false，B1T8192=0.9912/0.9857 且 false。不要把中位数比值与逐对比值中位数混称，不写 p=0 的概率绝对为零；写记录字段或“p 低于记录精度”。
J. 反向大形状 0.41–0.49× 与理想执行上限 0.85–0.92× 要区分：实测与理想流量下界仍有约 2× 执行余量；合同分析说明即使兑现执行余量也难达到 1.1×，不要把全部差距归因于合同。
K. FlashQLA 前向为 sm_120 参照，当前本方前向落后约 1.4–1.5×（窗口口径不同）；反向发布状态缺 export、smem 超限，垫片后 1.435ms@B1T2048 与 fla 相当。不要写“全面优于所有实现”。
L. sm_80 冒烟：上游 sm_80 测试文件，chip 重定向 sm_120a 后在 5090 上运行，因同 mma.sync 指令族而验证工具链；不写只测 sm_80。

## 旧材料使用
可读 paper/zh/v5parts/CH*.md 获取真实展品与数字，但不得整段拼接旧文。优先使用同一真实代码的更短片段，补充原理、推导、反例和读者可操作检查。v5 中所有 R* 轮次引用只留在证据路径，不进正文。

## 图计划（占位按出现顺序，构建时自动编号）
FIG1 在 CH30：前向双窗口加速比（沿用真实 SVG）。
FIG2 在 CH30：反向负载阶梯（沿用真实 SVG）。
FIG3 在 CH18：搜索成本对数条（沿用真实 SVG）。
FIG4 在 CH24：F1 分段成本（沿用真实 SVG）。
FIG5 在 CH27：split-bf16 误差条（沿用真实 SVG）。
FIG6 在 CH26：别名池压缩（沿用真实 SVG）。
FIG7 在 CH29：T1 覆盖堆叠条（沿用真实 SVG）。
FIG8 在 CH31：反向流量下界（沿用真实 SVG）。
FIG9 在 CH01：IR 层级与 use-def 图解（新 SVG，必须有真实代码对应）。
FIG10 在 CH09：linalg indexing map 的 KKT 实例（新 SVG，带公式与下标）。
FIG11 在 CH26：W32P bank 几何（新 SVG，含 80 mod 32=16 与 lane 地址区间）。
FIG12 在 CH13：payload/transform/handle 与解释器流程（新 SVG，含真实 15 行工件对应动作）。
如时间不足，优先保留 FIG1-8 和 FIG9-11；FIG12 可用真实代码表替代，不强行画空图。

## 结课能力目标（不写“覆盖上游 80% API”）
读者能独立完成：读写并验证 IR；定义方言/属性/verifier；写 rewrite 与转换 pass；组织 pipeline 并观察 IR；写 linalg+scf CPU 数值核并 bufferize/JIT；用 transform 改调度；用 Z3 建模并证明枚举完备；从标定计数构造排序模型并评估盲区；从 GPU IR 到 PTX/SASS；处理 fragment/async/TMA 合同；建立数值/确定性/性能验证矩阵；迁移到结构相近的 AI 算子并列出复用/重写清单。
