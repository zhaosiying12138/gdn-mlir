## 三十八、覆盖证明：PTX 收割与 T1/T2

第一章立下的表达力主张——旋钮空间在指令与结构两层覆盖两家手写参考——是一句可以作弊的话，除非"覆盖"被翻译成可数的对象：把 fla 与 FlashQLA 的产物离线收割成 PTX 与 SASS，逐指令族清点比较（T1），再按结构要素配对比较（T2）。术语先行：收割，指在纯 CPU 上把参考实现的 GPU 代码编译导出为可数文本的过程；指令族，指同一条硬件指令的全部拼写变体归并成的一族；三态，指每个族被判为 YES（已发射且变体齐备）、PARTIAL（已发射但缺个别变体）或 NO（从未发射）。覆盖证明与第三十六、三十七章的验证塔互补：验证塔回答数值对不对，覆盖证明回答旋钮空间够不够宽。

### 1. 纯 CPU 收割路径

直觉：要比较指令清单，必须拿到两家的真实 PTX，而不能拿文档或记忆。机制：fla 侧走 Triton 的提前编译接口，目标锁定后编译产物直接落盘；FlashQLA 侧经 TileLang 锁定 target 后导出 kernel 源；SASS 统一经 nvdisasm 反汇编。收割不占 GPU，覆盖这道门因此在纯 CPU 上可复现——这层性质并非为了省钱：收割物是覆盖主张的全部证据，证据的采集链路若依赖某台 GPU 的现场状态，主张就随环境漂移；纯 CPU 路径让本章的覆盖结论与第三十六章的字节级可复现主张共用同一条纪律。第三个风格点 FlashKDA 不单独收割：它的 token 并行 WY 预段设计已被网格映射旋钮显式吸收，属 L2 层的结构可达性而非指令层问题。fla 侧的目标锁定一行：

```
from triton.backends.compiler import GPUTarget
tgt = GPUTarget("cuda", 120, 32)
```
［tools/harvest_flaptx.py:260-262；驱动脚本 scripts/harvest_fla.sh；TileLang 侧见 scripts/harvest_qla.sh:38（target 锁定 sm_120 后取 kernel 源、ptxas -v、反汇编）］

收割物自带资源日志（Triton 提前编译顺带产出 ptxas -v 报告），每个参考 kernel 一行账。fla 侧样例（寄存器顶格与溢出的普遍性一望而知）：

```
| chunk_gated_delta_rule_bwd_kernel_dhu | 255 | 33288 | 3236 | 3788 | 512 | 14 |
| chunk_gated_delta_rule_fwd_kernel_h    | 255 | 17164 |  380 |  384 | 256 | 14 |
| prepare_wy_repr_bwd_kernel             | 255 | 17408 | 4808 | 5348 | 576 |  0 |
```
［evidence/G0/fla_ptx/SUMMARY.md，列序为配置/寄存器/动态共享/写溢出/读溢出/mma 数/异步复制数，节选三行］

### 2. 两极点画像

两家参考构成调度谱系的两个极端，收割数据给出各自的画像：

| 维度 | fla 包络 | FlashQLA 包络 |
|---|---|---|
| 每线程寄存器 | 40–255（多数顶格加溢出） | 88–128（setmaxnreg 固定） |
| 溢出字节 | 0–5348 | 0–1752 |
| 异步流水 | 多级 ld.global，状态类 kernel 用 cp.async | TMA 加 mbarrier 加 elect.sync 领导者 |
| warp 专化 | 无 | setmaxnreg 加 elect.sync，5 个 kernel 中 4 个 |
| 循环结构 | 多数全展开（bra 0–16） | 真循环（bra 107–611） |

［evidence/G2/coverage_matrix.md:56-64；FlashQLA 侧原始计数 evidence/G0/qla_ptx/SUMMARY.md——fused_bwd 单 kernel 1600 条 mma 来自约 200 条循环体乘 8 个回边］

画像的用途：覆盖的目标从抽象的"两家"变成具体的两份资源与风格包络。旋钮空间若能生成落在这两个包络内的配置，表达力主张就有了可检验的形态。

### 3. T1：指令族三态

T1 在指令族层清点：33 个族（15 个特性层加 18 个基线层），每族对照两家收割物判三态。终态：全部 33 族 8 YES / 19 PARTIAL / 6 NO；特性层 15 族 8 YES / 6 PARTIAL / 1 NO［evidence/G6/coverage_hatch_summary.md §1d；v4 第八节同数］。这个计数里有一格值得注明来历：mma.sync 族从 PARTIAL 翻到 YES，靠的正是第 4 节那张签名表的 tf32 一行——唯一逃生舱把 fla 的 K=8 精确拼写补齐之前，计数是 7/20/6。两个关键读法。其一，全部 PARTIAL 的残差属于指令变体而非能力缺口——缺的是操作码拼写、向量宽度、旗标这类变体（ldmatrix 的 x1 与 x4 转置形态、cp.async 的 wait_all 变体、mbarrier 的显式作用域拼写），每格都标注了对应的旋钮可达路径；有一格的残差干脆是命名层面的事实（基线层的 barrier 族：本方经 gpu.barrier 发出的 bar.sync 0 与参考的 barrier 指令同名异写，族级 NO 是记号产物）。其二，唯一的特性层 NO 是 stmatrix：它按测量关闭——参考实现用它做片段态存储，本方 epilogue 走合并的线程跨步 cell 通道，把片段态映射存储的替代方案实测过，慢 40%（F2 从 249 到 380 微秒），族级 NO 是错误模式被测量否决，与拼写能力无关［evidence/G6/coverage_hatch_summary.md §4；evidence/G5/fwd_report.md r3 §1d］。

{{FIG7}}

基线层的 18 族经标准下降自然触达（ld/st.global、ld/st.shared、cvt、setp、bra 等），其中 4 个 NO 属于本方 kernel 不需要的自由融合与打包（fma.f32、math.f32x2 等），不构成 GDN 依赖。

### 4. mma 签名双 YES 表

T1 内嵌一张更细的子表：mma 指令按完整签名清点（签名含形状、精度与布局，是张量核发射能力的最小单位）。

| mma 指令签名 | fla | FlashQLA | 本方 |
|---|---|---|---|
| bf16 m16n8k16 | 2708 条 | 2496 条 | 552 条，YES |
| tf32 m16n8k8 | 576 条 | 0 条 | YES（经唯一逃生舱，fla 精确拼写） |

［evidence/G6/coverage_hatch_summary.md §1b；条数为收割与生成 PTX 中的实际计数］

本方是三者中唯一同时发射两个家族签名的生成方。tf32 那一行是逃生舱 gdn.mma.tf32k8 的全部意义：树内 NVVM 表列封顶 K=4，fla 的 K=8 拼写只能经内联汇编触达，其工程全案（早占用约束、A 片寄存器序的硬件探针）见第二十六章。预算 5 个、支出 1 个——其余计划中的直通 op（setmaxnreg、elect、mbarrier、TMA、prefetch）经查全部已在 2026-10 主干表列，薄封装即可下降，这正是"预算克制"的证据结构。

### 5. T2：结构匹配与风格可达性限度

T2 回答结构层：参考 kernel 的调度形态是否在旋钮空间内可生成。方法：四张结构卡（fla 两个 kernel 加 FlashQLA 两个 fused kernel），每张按指令族 Jaccard 相似度、mma 签名匹配与指令总数接近度，从本方 20 个配置中选最近邻，再对寄存器、共享内存、溢出三项资源以正负 30% 包络判分。结论：fla 极点（多级流水加 cp.async 加 255 寄存器顶格）与 FlashQLA 极点（warp 专化加 TMA 加单 kernel 融合）的每个结构要素都在旋钮空间内可生成——前者由基础 mma 配置覆盖（寄存器钉在 255 上限、栈帧 0 至 26.8KB 落在 fla 包络内、流水深度 0/2/3 逐档扫过，对应 LDGSTS 计数 512/1536/2048），后者由 warp 专化配置覆盖（setmaxnreg 把溢出从 22,844B 压到 48B、5 个 TMA 操作替换 1026 条经典复制、elect 领导者 mbarrier 协议，且经 G4 硬件验证——先解决死锁协议与主机侧张量映射编码，两极点各自拿到上机通过的证据）［evidence/G2/coverage_matrix.md §3-5；evidence/G4/GATE_REPORT.md］。

T2 同时如实报告度量限度：它度量风格的可达性，规模差异由资源包络单独判分。FlashQLA 的 fused_bwd 单 kernel 装载 1600 条 mma（约 200 条循环体乘 8 个回边），与本方单 chunk kernel 相差 50 倍——该卡资源判定为 OUT，结构要素全部在格内：

```
### qla:fused_bwd/default
- mma.sync total 1600 (est. per loop body ~200 over 8 back-edge loop(s); bra=422)
- async pattern: TMA tensor=41, mbarrier=194, ld.global=77(of which .nc 76)
- resources: regs=128, smem=2048B, spill st/ld=1098/1752B
- ours closest config: kkt_bf16_mma_tma_ws
  - regs 255 vs ref 128 -> OUT (1.99x); mma total 32 vs ref 1600 (0.02x)
```
［evidence/G2/coverage_matrix.md:104-121，节选］

读法：规模之差来自"单 kernel 全融合"与"按阶段拆分"两种结构选择，属设计取舍；T2 的主张从来限于结构要素（TMA、mbarrier、elect、寄存器钉位、真循环）逐项可生成，这一主张四张卡全部成立。

覆盖证明的总效果可以一句话收拢：表达力主张（第一章第一条）经 T1 与 T2 拆成两份可数的账——指令层 33 族三态加 mma 签名双 YES，结构层两极点四卡全过——而支撑这两份账的收割物、比较脚本与逐格注释全部入库，覆盖结论与验证塔的数值结论一样，是离线可复算的证据文件。这一性质直接服务第四十一章的复现门：覆盖分析挂在纯 CPU 的 G2 门下，任何人重跑都能得到同一张三态表。
