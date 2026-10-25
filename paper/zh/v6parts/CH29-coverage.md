## 二十九、覆盖如何成为证据：PTX 收割、T1/T2 与扩展预算

“能表达手写内核”只有落到可数的机器产物上，才可以检验。这里的收割（harvest）是把参考实现编译成 PTX、SASS 与资源日志，再从文本提取指令族和结构特征。T1 检查指令与精确签名，T2 检查调度结构；数值验证仍由第二十八章的矩阵承担，性能则由第三十章另行裁决。

### 1. 把参考实现变成离线文本

fla 的 Triton 内核使用提前编译（Ahead-of-Time，AOT）接口。目标三元组明确写成 CUDA、计算能力 120、warp 宽度 32，避免宿主机自动探测改变产物。

```python
def compile_one(k, sig: dict, cexpr: dict, warps: int, stages: int):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    src = ASTSource(fn=k, signature=sig, constexprs=cexpr)
    tgt = GPUTarget("cuda", 120, 32)
    return triton.compile(
        src, target=tgt,
        options={"num_warps": warps, "num_stages": stages})
```
[tools/harvest_flaptx.py:257-263，函数换行排版，语句保持原意]

读这段代码要看 `target` 与两个调度选项：`num_warps`、`num_stages` 会改变指令和资源，因而属于收割坐标的一部分。锁定目标后，编译、PTX 保存和 `ptxas -v` 都可在 CPU 上完成；这能判断产物对应哪个架构，却不能代替该架构上的数值运行。

FlashQLA 走另一条确定性路径：驱动固定 `TILELANG_DEFAULT_TARGET` 为 `sm_120a`，由 TileLang 导出 kernel 源，再经 NVCC 生成 PTX/cubin；`nvdisasm -c kernel.cubin` 负责反汇编。[scripts/harvest_qla.sh:38-42；tools/harvest_qlaptx.py:117-133,155-166] 两条路径都无需执行 GPU kernel。纯 CPU 的含义是采集可离线复算，并不表示目标架构、编译器版本和补丁可以省略；这些信息必须与 PTX 一同归档。

收割的最小单位是“kernel 名加完整配置”，不能只有函数名。相同 Triton 函数在不同 warps、stages 或 tile 下会产生不同指令清单；FlashQLA 的工具链补丁也会改变可生成的目标。每个目录因此同时保存 PTX、cubin、SASS 与日志，汇总表只充当索引。复查者应能从汇总行回到对应文本，无需再次凭函数名编译一个可能不同的配置。

资源日志也有独立用途。收割器从 `ptxas -v` 提取寄存器、静态共享内存、溢出读写和栈帧，这些字段构成后面的包络与 T2 判分材料。[tools/harvest_flaptx.py:246-253] PTX 适合核对虚拟指令签名，SASS 适合观察后端实际映射；两者分层保存，可以避免把 PTX 静态条数误当成最终机器指令的动态执行次数。

### 2. 两个参考极点

两家实现提供了差异足够大的资源与调度包络。包络是一组观测范围，描述参考实现实际占据了哪些设计点。

| 维度 | fla 包络 | FlashQLA 包络 |
|---|---|---|
| 每线程寄存器 | 40–255，常见顶格与溢出 | 88–128，由 setmaxnreg 约束 |
| 最大读或写溢出 | 0–5348 B | 0–1752 B |
| 异步装载 | 多级 ld.global；状态 kernel 使用 cp.async | TMA、mbarrier、elect.sync 领导者 |
| warp 专化 | 无 | 五个 kernel 中四个使用 |
| 循环形态 | 多数展开，bra 为 0–16 | 真循环，bra 为 107–611 |
[evidence/G2/coverage_matrix.md:54-63；evidence/G0/fla_ptx/SUMMARY.md:1-24；evidence/G0/qla_ptx/SUMMARY.md:1-7]

读表要同时看指令和资源。255 寄存器配溢出、无专化、展开循环勾勒出 fla 一极；固定寄存器、TMA 协议和真循环勾勒出 FlashQLA 另一极。由此可以把“覆盖两家”拆成具体结构问题：能否生成多级流水，能否形成领导者—屏障协议，能否承受相应的寄存器与溢出状态。表中没有耗时，所以不能从中推出任何一方的速度。

### 3. T1：指令族、变体与精确签名

T1 把同义拼写归并为指令族，再用三态记录生成集：YES 表示某配置已经发射且参考变体齐备，PARTIAL 表示族已出现但缺参考中的个别变体，NO 表示生成集中未出现。33 族的账目为 8 YES、19 PARTIAL、6 NO；feature 层 15 族按 G6 表逐行重算为 7 YES、6 PARTIAL、2 NO。[README.md:302-311]

PARTIAL 要读到变体粒度。例如 `ldmatrix` 缺少 x1 与部分 x4 转置形态，`cp.async.ctrl` 缺 `wait_all`，`mbarrier` 差显式 CTA 作用域拼写。它们说明“族名相同”仍可能掩盖操作数、向量宽度或旗标差异，也给出可逐项复查的残差清单。feature 层的两个 NO 要分开读：`stmatrix` 在 NVVM 已有词汇，但本方 epilogue 的片段映射存储方案实测从 249µs 增至 380µs，时间变为 1.526×、约增加 53%，因此按测量关闭；`barrier` 的 NO 是命名归并结果，`gpu.barrier` 已覆盖所需屏障语义。它们都不构成能力缺口。这里的 NO 保存了有证据的设计选择。[evidence/G6/coverage_hatch_summary.md:96-103]

三态计数没有“覆盖率百分比”的含义。19 个 PARTIAL 各自缺的内容不同，6 个 NO 也要区分测量否决、命名归并与当前 kernel 不需要的融合或打包。一个严谨单元格至少应能回答四个问题：参考中出现了什么，本方哪个配置出现过什么，残差精确到哪条拼写，关闭它的依据是什么。缺少任何一项，YES 也可能只是粗粒度同名。

结论还受生成集范围约束。这里的 YES 是 G2 的 20 个配置与 G5-G/当前生产生成物联合形成的存在性证明，并不声称任意配置、任意形状都会走同一路径。新增参考指令时，应先扩充收割合同，再判断需要新增旋钮、调整下降，还是确认当前算子没有这项需求。

{{FIG7}}

指令族仍不足以描述张量核，必须继续比较包含形状、输入精度和累加类型的完整签名。

| mma 签名 | 本方 | fla | FlashQLA |
|---|---|---|---|
| bf16 m16n8k16，f32 累加 | 552 条，YES | 2708 条 | 2496 条 |
| tf32 m16n8k8，f32 累加 | YES | 576 条 | 0 条 |
[evidence/G6/coverage_hatch_summary.md:36-41]

读表时，条数描述被清点文本的规模，YES 描述精确签名是否出现，两者不能混成覆盖率。bf16 行证明三家共享主力签名；tf32 行证明本方能发出 fla 的 K=8 拼写，同时 FlashQLA 收割物没有这一签名。精确匹配比“都用了 mma”更能约束实现能力。

静态签名计数也不能直接换算计算量。循环体会重复执行，不同 kernel 的网格和 tile 又不同，552 对 2708 并不代表工作量之比。签名 YES 只把问题推进到操作数布局与数值：寄存器顺序错误时，文本仍可能完全正确，所以 tf32 扩展还保留 early-clobber 约束和整型硬件探针作为配套证据。[evidence/G6/coverage_hatch_summary.md:79-87]

### 4. T2：结构最近邻与规模边界

T2 选四张结构卡：fla 的 `chunk_fwd_kernel_o`、`bwd_kernel_dhu`，以及 FlashQLA 的 `fused_fwd`、`fused_bwd`。每张卡从本方 20 个配置中，按指令族 Jaccard 相似度、mma 签名匹配和指令总数接近度寻找最近邻；寄存器、共享内存、溢出另用正负 30% 包络判分。[README.md:313-317；evidence/G2/coverage_matrix.md:69-113]

这里的 30% 是预先声明的结构匹配容差，不能解释为统计置信区间。Jaccard 相似度看指令族集合，无法表达依赖顺序；mma 签名与异步协议、循环、资源卡补足了这部分信息。最近邻只负责找一张便于人工比较的配置，OUT 必须保留，否则匹配过程会把最有诊断价值的差异隐藏掉。四张卡覆盖两个设计极点，也不代表所有可能的手写实现都已纳入合同。

最近邻用于读结构，不用于排名。fla 极点所需的 255 寄存器、0/2/3 级流水和 cp.async，FlashQLA 极点所需的 setmaxnreg、TMA、mbarrier、elect.sync 与真循环，都能在旋钮空间生成；后一类配置还经过硬件数值验证。结构要素逐项可达，并不保证某个最近邻落入全部资源包络。

```text
### qla:fused_bwd/default
- mma.sync total **1600** (est. per loop body ~200 over 8 back-edge loop(s); bra=422)
- resources: regs=128, smem=2048B, spill st/ld=1098/1752B
- ours closest config: **kkt_bf16_mma_tma_ws**
  - regs 255 vs ref 128 -> OUT (1.99x); smem 8 vs 2048 -> OUT (0.00x) (±30%)
  - mma total 32 vs ref 1600 (0.02x); family Jaccard basis for 'closest' selection
```
[evidence/G2/coverage_matrix.md:104-113，省略签名、异步与其他资源计数行]

这段卡先读 1600 的来源：约 200 条循环体指令经八个回边复用，文本计数描述融合规模。再读两个 OUT 所揭示的边界：最近配置的寄存器超出包络，单 chunk kernel 与全融合反向 kernel 的 mma 规模相差约 50 倍。T2 能确认 TMA、屏障、专化和循环等要素存在，不能据此声称阶段拆分与单 kernel 融合拥有相同资源或性能。

### 5. 扩展预算限制证明成本

覆盖还应交代为此扩展了多少编译器词汇。项目自设的审计预算允许至多 5 个自定义 NVVM op，当前实现实际使用 `gdn.mma.tf32k8` 一个 inline-asm 扩展：它源于 G2 记录中的 NVVM 词表缺口与当时的 K=4 产物。当前上游 NVVM 和 NVGPU→NVVM 已能表达 TF32 m16n8k8，因此这笔支出不能被解释为 MLIR 的永久表达力边界；是否迁移到上游路径，仍需重新验证类型、布局、数值和性能。setmaxnreg、elect、mbarrier、TMA、prefetch 等其余计划词汇已能经当前主干表列下降。[evidence/G6/coverage_hatch_summary.md:79-94；../llvm-mlir/mlir/include/mlir/Dialect/LLVMIR/NVVMOps.td:2860-2862]

预算使表达力主张带有价格：若无限制添加私有指令，覆盖可以通过扩张词汇获得；只支出一个扩展，说明大部分结构仍由通用 IR 与下降路径承担。T1/T2 到此证明的是指令表达力与参考风格可达。性能等价还涉及依赖链、占用、流量和测量窗口，必须由独立计时证据回答。
