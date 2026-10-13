## 二十六、当表内没有你要的指令：inline_asm 逃生舱工程

任何编译器的指令词表都有边界。MLIR 的 NVVM 方言由 TableGen 表驱动生成，表里没有的 PTX 指令，无论调度多正确都发不出来。这时唯一的出路是 inline asm（内联汇编）：在 IR 里嵌一段原始 PTX 文本，把操作数与结果经约束字符串接线。本编译器给自定义 op 定了不超过 5 个的逃生舱预算，实际支出 1 个——gdn.mma.tf32k8。方法论的核心立场：逃生舱越小越好，但每一个都要带必要性论证、缺陷定位与硬件探针三件套。

### 1. 必要性：两条 stock 路线都到不了 m16n8k8

反向链的 fp32 GEMM 需要 tf32 的 mma.sync.aligned.m16n8k8 拼写——这正是从 fla 收割到的精确签名（表 6：tf32 m16n8k8，fla 576 条）。为什么必须 tf32：反向多数 GEMM 有一侧操作数是 fp32 中间量，直接 bf16 mma 会在读取时破坏精度，而 tf32 保留 f32 的指数域、只把尾数降到 10 位，是张量核三档精度里唯一既走硬件乘加又保住指数范围的档位。stock 路线一（nvgpu.mma.sync）：verifier 的 8×8×128b 基本片把 tf32 封顶在 K=4（第二十五章），本方 stock 路径实际只能发出 m16n8k4——真实存证：

```
mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32
    {%r6, %r7, %r8, %r9},
    {%r2, %r3},
    {%r4},
    {%r5, %r5, %r5, %r5};
```
［evidence/G4/gen_verify/o_f32_mma_p2_t64x64x32_w4_K64V64/kernel.ptx:6229-6233，o_f32 配置的 stock 产物，逐字］

stock 路线二（NVVM TableGen 的 nvvm.mma.sync 家族）：m16n8k8 形态只登记了 f16/bf16/i8 操作数，没有 tf32。表内不可达属于上游表的限制，与调度能力无关；于是在 gdn 方言里定义 gdn.mma.tf32k8（吃 4 个 A 标量、2 个 B 标量、4 个 C 标量，产 4 个 D 标量），由下降层翻译成 inline asm。

### 2. 约束字母：=f、=&f 与 BasicPtxBuilder 的表达力边界

llvm.inline_asm 的约束字符串沿用 GCC 体系：等号开头的字母声明一个输出约束，f 表示 32 位浮点寄存器，& 是 early-clobber（提前写坏）前缀——它告诉寄存器分配器"此输出在全部输入被读走之前就会被写入，禁止与任何输入共用寄存器"。没有 & 的输出，分配器合法地可以把它分到某个已死输入的寄存器上。这段常识在 mma 上变成正确性问题：D 四元组若被分到同一条 mma 的 A 四元组上，张量核读写重叠、操作数被自身结果覆盖。真机 SASS 上观察到 ptxas 恰恰这么分（HMMA R12, R12, R10, R16——第一个操作数与目的同名），K=16 的双原子链 128 个单元里 118 个确定性失败、每个错值都恰等于缺若干 k 项。修正是四个输出全部加 &：

```
static const char *kPtx =
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
    "{$10, $11, $12, $13};";
static const char *kConstraints =
    "=&f,=&f,=&f,=&f,f,f,f,f,f,f,f,f,f,f";
```
［dialect/lib/GdnHatch.cpp:174-179，逐字；完整下降 170-210 行，结果为 literal struct 由 4 条 extractvalue 解出］

树内的 nvvm.inline_ptx（BasicPtxBuilder 路径）表达不了这个修正——它的输出约束写死为普通 =f，没有 early-clobber 入口。所以 gdn.mma.tf32k8 绕过它、直接构造 LLVM::InlineAsmOp。这条对任何在 mma 上使用 inline asm 的实现都成立，是独立于本项目的工具链发现（第四十章发现 11）。

### 3. 探针：整型逐位与片段序

逃生舱接对了约束还不够，还要证明发射的片段序（每 lane 的操作数对应矩阵哪个元素）与硬件一致。探针设计分两层。第一层整型逐位：tf32 的 10 位尾数可精确表示小整数，于是用整型操作数上机，D 与 A、B、C 的整数乘加必须逐位相等——任何片段错位都表现为缺项或错位项，零容错。16×8×8 探针由此锁定了寄存器序：

```
# Hardware-verified fragment layout (evidence/G5/scratch/tf32k8_probe,
# exact integer match): g = lane>>2, t = lane&3
#   A 16x8: a0=(g,2t) a1=(g+8,2t) a2=(g,2t+1) a3=(g+8,2t+1)   [m, k]
#   B 8x8 : b0=(2t,g) b1=(2t+1,g)                             [k, n]
#   D 16x8: d0=(g,2t) d1=(g,2t+1) d2=(g+8,2t) d3=(g+8,2t+1)   [m, n]
# (NOTE: the A register order is row-half OUTER, col-pair inner -- NOT the
#  packed-pair order of bf16 k16's A; probe exp1 caught this.)
```
［tools/gdn_bwd_gen.py:2029-2036，注释逐字；探针套件 evidence/G5/scratch/tf32k8_probe/］

A 的四个寄存器是行半外层、列对内层（a0 与 a1 同列对不同行半），与 bf16 m16n8k16 的 k 打包直觉相反——这条若靠文档猜必错，靠探针一次锁定。探针套件本身值得展开：每个组合是一个独立 kernel——全局缓冲装随机数、共享内存装片段、按候选布局发射 hatch 调用、结果写回——由运行器对 numpy 参考逐元素核对，共覆盖 13 个站点实际用到的 (at, bt, acc, neg, offs) 组合（evidence/G5/scratch/tf32k8_probe/tgemm_probe_gen.py:18-32），越界由 compute-sanitizer 与非法地址错误兜底。第二层语义确认：换 N(0,1) 随机输入测舍入偏差，实测 4.2e-3，与 tf32 截断模拟差约 1 ulp，证实硬件执行的是真 tf32 乘加而非某种降级路径（evidence/G5/bwd_report.md:277-286）。118/128 的失败指纹同时承担了诊断职能：确定性缺 k 项指向读写重叠（约束问题），随机性错位才指向片段序问题——两类缺陷由两种探针分开裁决。

### 4. 位点门控：逐点数值验证与性能否决

逃生舱放行的最后一级控制按位点（GEMM 调用位置）进行：反向共 24 个 fp32 GEMM 位点，逐点数值验证后放行 16 个。验证的尺子是逐位点对照阈值——每个候选位点单独启用，整链对 fp64 参考评差，误差以该位点喂养的梯度合同（dq/dk/dv 不超过 8e-3、dβ/dg 不超过 2e-2）裁决。两类否决都有真实案例：wy.u 位点误差 1.06e-2 超出 dk 容差 8e-3，数值否决（u 的噪声经状态递推放大到 dk 的路径是机制原因，同形状的 wy.w 位点误差 1.4e-5 干净放行）；dhu 位点数值干净但完全展开的 tgemm 本体把 ptxas 推到 255 寄存器加 116B 溢出（对照标量路径 128 寄存器加 188B 溢出），内核 3.69 毫秒劣化到 4.67，性能否决、整体回退。管理方式在生成器里显式化为两张集合表：TF32_GROUPS 按组枚举全部候选位点、TF32_EXCLUDE 列否决清单并注明每条否决的理由，差集即默认放行集（tools/gdn_bwd_gen.py:2049-2100）。发射侧，每条放行位点落成一条标量化的 hatch 调用：

```
e.l(f"{n0}, {n1}, {n2}, {n3} = gdn.mma.tf32k8 {a0}, {a1}, {a2}, "
    f"{a3}, {b0}, {b1}, {accs[0]}, {accs[1]}, {accs[2]}, "
    f"{accs[3]} : f32 -> (f32, f32, f32, f32)")
```
［tools/gdn_bwd_gen.py:2310-2313，tgemm 每 (tile, kb) 发射一条；累加器经 accs 列表跨 k 段串接］

效果：放行位点把所在阶段端到端从 14.16 降到 11.06 毫秒。预算 5 个、支出 1 个；支出的这一个带着完整的三件套——必要性（两条 stock 路线封死）、缺陷定位（=f 的读写重叠与 118/128 指纹）、硬件探针（row-half-outer 序与 N(0,1) 语义确认）——构成"逃生舱工程"的全部要素。
