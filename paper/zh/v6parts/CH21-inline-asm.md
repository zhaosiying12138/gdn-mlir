## 二十一、词表之外：Inline Asm 与唯一自定义指令扩展

`gdn.mma.tf32k8` 是自定义指令扩展预算中唯一的实际支出：预算至多 5 个，使用 1 个。它通过内联汇编（inline asm）发出目标指令，接收 4 个 A、2 个 B、4 个 C 的 f32 标量，返回 4 个 f32 的 D 标量[dialect/include/Gdn/GdnHatch.td:185-193]。一个 op 获得这项支出的前提，不能停在“PTX 手册里有这条指令”；词表缺口、寄存器合同、片段布局、整链数值和资源成本必须依次成立。

### 1. 必要性必须按版本与产物复核

反向计算里不少 GEMM 直接消费 f32 中间量。bf16、tf32、fp32 都有 8 位指数；显式尾数分别为 7、10、23 位，计入隐含位后的有效精度分别为 8、11、24 位。直接转 bf16 会再次量化已经承载误差的中间量。目标指令因而是 `mma.sync.aligned.m16n8k8...tf32`，无须把输入改成 bf16。

先看当前的 NVGPU verifier：`shapeK` 表示基本片大小，tf32 的 shapeK=4，`kTile=k/shapeK` 允许多个基本片，所以 shapeK 不能被读成 K 的上限。当前 NVGPU→NVVM 也按 gemmShape 逐片创建 NVVM MMA；上游测试已经直接给出 `mmaShape=[16,8,8]`[../llvm-mlir/mlir/test/Dialect/NVGPU/mma-sync-f32-to-tf32.mlir:13-18；verifier 的基本片处理见 ../llvm-mlir/mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp:218-264]。本项目 G2 建立扩展时核对的 NVVM 记录则是另一版本：当时的 m16n8k8 组合只覆盖 f16、bf16 和整型，未提供所需 TF32 形式[evidence/G2/gen_report.md:125-134]。这两条事实必须按版本分开。

当前上游 `llvm-mlir/mlir/include/mlir/Dialect/LLVMIR/NVVMOps.td:2860-2862` 已把 TF32 的 `GEOM<16,8,4>` 与 `GEOM<16,8,8>` 都列入 `tf32_mma_ops`；`NVGPUDialect.cpp` 的基本片处理也说明 K 可以通过 kTile 组合扩大。项目当时实际能发出的 TF32 原子仍是 K=4，下面摘自 stock `o_f32` 配置的真实产物：

```ptx
mov.b32 %r5, 0f00000000;
mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32
    {%r6, %r7, %r8, %r9},
    {%r2, %r3},
    {%r4},
    {%r5, %r5, %r5, %r5};
```

[evidence/G4/gen_verify/o_f32_mma_p2_t64x64x32_w4_K64V64/kernel.ptx:6228-6233]

读这段产物要看三处：形状字段确为 `m16n8k4`，A/B 类型确为 `tf32`，累加与结果仍为 `f32`。它记录的是当时生成器实际选择的形态，不能用来证明当前上游只能 K=4。当前上游已能表达 TF32 m16n8k8；本实现仍保留 `gdn.mma.tf32k8` 与 inline asm，是否迁移到上游路径，需要重新检查类型、布局、数值和性能，而不能把 G2 的缺口写成 MLIR 的永久限制。

### 2. 约束字符串同时描述类型与寄存器生存期

当前下降直接创建 `LLVM::InlineAsmOp`。PTX 文本、约束和操作数顺序如下；结构体与四条 `extractvalue` 的后续代码在此省略：

```cpp
static const char *kPtx =
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
    "{$10, $11, $12, $13};";
static const char *kConstraints =
    "=&f,=&f,=&f,=&f,f,f,f,f,f,f,f,f,f,f";
auto structTy =
    LLVM::LLVMStructType::getLiteral(ctx, {f32, f32, f32, f32});
SmallVector<Value, 10> args{op.getA0(), op.getA1(), op.getA2(),
                            op.getA3(), op.getB0(), op.getB1(),
                            op.getC0(), op.getC1(), op.getC2(),
                            op.getC3()};
auto asmOp = LLVM::InlineAsmOp::create(
    rewriter, loc, structTy, ValueRange(args),
    rewriter.getStringAttr(kPtx), rewriter.getStringAttr(kConstraints),
    /*has_side_effects=*/UnitAttr::get(ctx),
    /*is_align_stack=*/UnitAttr(),
    /*tail_call_kind=*/
    LLVM::TailCallKindAttr::get(ctx, LLVM::TailCallKind::None),
    /*convergent=*/UnitAttr(),
    /*asm_dialect=*/
    LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
    /*operand_attrs=*/ArrayAttr());
```

[dialect/lib/GdnHatch.cpp:174-199，省略前面的 loc、ctx、f32 定义]

约束从左到右先声明 4 个输出，再声明 10 个输入，所以 `$0..$3` 是 D，`$4..$7` 是 A，`$8/$9` 是 B，`$10..$13` 是 C。`f` 请求 32 位浮点寄存器，`=` 表示输出，`&` 表示提前破坏（early-clobber）：分配器不得把该输出放进任一输入寄存器。普通 `=f` 允许在输入最后一次使用后复用其寄存器；对一条同时读取整组 A、B、C 并写整组 D 的 MMA，这种复用会让同一原子内的读写集合相交。

内联汇编对 LLVM 是一个带合同的指令黑盒。框外的十个 SSA 输入和结构体结果仍参与优化，框内的矩阵语义、lane 映射和并发读写顺序则不会由 LLVM 自动推导。下降把四个结果组成 literal struct，再用四条 `extractvalue` 接回独立 f32 SSA 值，避免借助临时内存交换片段。这样做也把审计责任划得很清楚：类型与生存期由约束表达，矩阵含义只能交给探针验证。

缺陷有明确的机器指纹。普通 `=f` 下，ptxas 曾把 D 四元组映射到同一指令的 A 四元组，SASS 出现 `HMMA R12, R12, R10, R16`；K=16 的双原子链有 118/128 个单元稳定出错，错值恰好缺少完整的 k 项[dialect/lib/GdnHatch.cpp:152-169]。四个 `=&f` 将这个分配空间排除。硬件允许的原地累加语义与这里更强的分配约束也要分开：实现选择让 D 与所有输入保持不同。旧 `nvvm.inline_ptx` 的 BasicPtxBuilder 路径只生成普通 `=f`，无法表达这项约束，所以当前实现没有走那条捷径。

### 3. 片段布局要用可精确判决的数据上机

约束正确只保证寄存器不互相覆盖，不能保证每个 lane 拿到的矩阵元素正确。片段（fragment）是 warp 指令把矩阵元素分摊到各 lane 寄存器的映射；本项目用 `g = lane >> 2`、`t = lane & 3` 表示其坐标：

```text
A 16x8: a0=(g,2t)   a1=(g+8,2t)
        a2=(g,2t+1) a3=(g+8,2t+1)
B 8x8 : b0=(2t,g)   b1=(2t+1,g)       [k,n]
D 16x8: d0=(g,2t)   d1=(g,2t+1)
        d2=(g+8,2t) d3=(g+8,2t+1)     [m,n]
```

[tools/gdn_bwd_gen.py:2030-2036]

A 的顺序是“行半外层、列对内层”：a0 与 a1 先跨两个八行半区，a2/a3 才走到相邻 k 列。bf16 m16n8k16 常让人按打包列对类推寄存器顺序，照搬到 TF32 K=8 会错位。这里的价值在于每个寄存器都有坐标，发射器可以做索引断言，探针也能逐单元指出错在哪一组。

整型探针先把舍入从判决中移除：

```python
A = rng.integers(0, 8, (16, 8)).astype(np.float32)
B = rng.integers(0, 8, (8, 8)).astype(np.float32)
C = rng.integers(0, 8, (16, 8)).astype(np.float32)
Dref = A.astype(np.int64) @ B.astype(np.int64) + C.astype(np.int64)
assert Dref.max() < 2048  # tf32-exact
# ... 发射一个 warp 并读回 D ...
if np.array_equal(D, Dref):
    print("[tf32k8_probe] PASS: D == A@B + C EXACTLY (16x8x8 atom)")
```

[evidence/G5/scratch/tf32k8_probe/tf32k8_probe_run.py:28-33,49-57，省略 CUDA 装载]

小整数及其乘加结果都落在 TF32 可精确表示的范围内，因此期望是逐位相等；任何缺项、交换或跨行错位都无法被容差吞掉。探针还把两类故障的指纹分开了：early-clobber 缺失表现为稳定缺少若干完整 k 项，片段顺序错误则会在坐标映射下呈现系统性错位。若一开始就用随机浮点和宽松误差门，这两类问题都可能被误读成正常舍入。第二层再换 `N(0,1)` 输入，与 f64 参考和截断 TF32 模拟分别比较[evidence/G5/scratch/tf32k8_probe/tf32k8_probe_run.py:58-84]。归档结果为对 f64 最大误差 4.2e-3，与截断模拟相差约一个 ulp 的量级[evidence/G5/bwd_report.md:275-287]。前者确认真实舍入误差存在，后者确认误差符合 TF32 语义；两次试验合起来，才能区分“恰好算对的错误精度路径”和真正的 TF32 MMA。

原子探针之后还有发射器探针。tgemm 要处理 A/B 转置、是否累加、是否取负和三类偏移，正确的单原子坐标仍可能被外层索引算错。套件覆盖实际使用的 13 种组合，每种都生成独立 kernel，并以 NumPy 参考逐元素核对；它曾定位到奇数 k 列被额外加一的错误[evidence/G5/bwd_report.md:291-311]。这层验证把“指令本身正确”与“生成器在每种调用方式下都正确”分开，也使后续位点准入有一个可复用的正确基座。

### 4. 位点准入选址的是误差传播与机器资源

原子正确不代表每个 GEMM 位点都该启用。最初的 24 个 f32 GEMM 候选逐一单独接入，在 T512 上观察整条反向链相对参考的梯度误差；16 个通过数值与性能门后进入放行集[evidence/G5/bwd_report.md:319-347]。这个 24→16 是该候选集的审计截面，后续新增的其他分组不能倒算进同一分母。

逐位点试验比一次全开更有诊断力。单开某个位点时，整链误差的变化可以与该位置的传播路径对应；若所有位点同时开启，多个扰动会混合，无法知道应当撤回哪一处。这里衡量的是最终 dq、dk、dv、dbeta、dg、dh0 等输出合同，而非孤立 GEMM 的最大误差。局部误差很小仍可能在递推中放大，局部舍入可见也可能被后续数据尺度吸收。

否决理由展现了局部舍入如何沿算法传播。`wy.u` 使 dk 误差达到 1.06e-2，超过 8e-3 合同，因为 u 的扰动会经状态扫描进入 Sn；同形状、同 K 的 `wy.w` 只带来 1.4e-5，获得放行。`dhu` 的三个数值干净位点则撞上资源墙：完全展开的 tgemm 位于反向串行 chunk 循环内，ptxas 从 128 寄存器、188B 溢出变为 255 寄存器、116–128B 溢出，阶段时间从 3.69ms 增至 4.67ms，遂保留标量路径[evidence/G5/bwd_report.md:321-340,369-374]。

回退是普通发射分支：

```python
if tag is not None and tag in TF32_SITES:
    tgemm(e, out, a, b, M, N, Kd, at=at, bt=bt, acc=acc, neg=neg,
          a_off=a_off, b_off=b_off, o_off=o_off, bar=bar, tag=tag,
          warps=warps)
else:
    sgemm(e, out, a, b, M, N, Kd, at=at, bt=bt, acc=acc, neg=neg,
          a_off=a_off, b_off=b_off, o_off=o_off, bar=bar, kahan=kahan)
```

[tools/gdn_bwd_gen.py:2338-2349]

`TF32_SITES` 按位点决定走向，未入选及被否决的位置继续调用语义相同的 `sgemm`。这使数值否决与性能否决都无需改变算子结构。放行集由候选分组减去显式排除项构成，排除项旁记录了越界误差或资源回退原因[tools/gdn_bwd_gen.py:2073-2100]。配置因此也能接受审计：看到某个标签，就能判断它为何走张量核，看到另一个标签，也能确认标量回退仍在编译覆盖之内。

性能判断不能只看单条 MMA 的理论吞吐。完全展开会增加寄存器活跃集合和指令文本，位于串行循环时还会重复暴露这些成本；占用、溢出、循环结构与归约方式需要一起计时。启用 tgemm 与配套 warp 归约改造后，归档端到端中位数从 14.16ms 降至 11.06ms；报告同时把 dqkwg、wyb 的收益归因于 MMA 和归约并行化，不能把全部差额记到这一条指令名下[evidence/G5/bwd_report.md:352-367]。

### 5. 一条扩展对应一条完整检查链

必要性由项目建立扩展时的 NVVM 记录与真实 stock PTX 证明；当前上游已能表达 TF32 m16n8k8，是否迁移需要重新做类型、布局、数值和性能验证。正确性由 early-clobber、SASS 指纹和整型片段探针证明；数值语义由随机舍入试验与逐位点整链误差证明；性能由资源报告和端到端计时共同裁决；任何位点不满足条件，都能回到标量实现。五道检查各自回答不同问题，任何一道通过都不能替代下一道。预算 1/5 只是本实现自设的审计规则，不代表当前上游的永久表达力边界；词表外代码必须携带足够证据，并且始终存在可发射、可验证的回退路径。
