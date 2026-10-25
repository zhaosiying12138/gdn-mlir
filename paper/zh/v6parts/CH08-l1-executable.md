## 八、L1 是一份可执行规约：生成、CPU 运行与三方闭合

`gdn-to-chunk` 当前留下的边界是一次普通函数调用。真实冒烟输出中，私有声明与前向调用具有以下形状：

```mlir
func.func private @__gdn_l1_fwd_chunk(tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2xf32>, tensor<2x128x2xf32>, tensor<2x2x16x16xf32>) -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
%0:2 = call @__gdn_l1_fwd_chunk(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {gdn.config = #gdn.config<64, true, false, true, true, true, false>} : (tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2xf32>, tensor<2x128x2xf32>, tensor<2x2x16x16xf32>) -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
```

[paper/zh/gdn-mlir-deep-dive.md:974-984，已记录的真实 stdout；生成逻辑见 dialect/lib/GdnPasses.cpp:26-58]

这次改写建立声明、传递操作数与结果类型、原样保留 `gdn.config`，并去重同一模块里的声明。声明没有函数体，张量也没有在这里展开。当前可运行的完整 L1 由 `tools/gdn_l1_gen.py` 直接生成，其入口名为 `@gdn_fwd` 或 `@gdn_bwd`；L1 数值 harness 并未把上述 `@__gdn_l1_*` 声明链接到生成器产物。分清这两个符号，才能准确理解现有验证覆盖的边界。

### 1. Python 发射器把算法规约物化为 MLIR

生成器的 `E` 类管理文本行、缩进、区域结束和 SSA 名分配，形状与 chunk 则经 f-string 注入模板：

```python
class E:
    """Tiny line emitter with fresh SSA names and indentation."""
    def __init__(self):
        self.lines = []
        self.depth = 0
        self.n = 0
    def __call__(self, s):
        self.lines.append("  " * self.depth + s)
    def v(self, hint=""):
        self.n += 1
        return "%" + hint + str(self.n)
    def block(self, header):
        self(header + " {")
        self.depth += 1
    def end(self):
        self.depth -= 1
        self("}")
    def text(self):
        return "\n".join(self.lines) + "\n"
```

[tools/gdn_l1_gen.py:161-185]

`emit_module` 接收未补零的数据和 B、T、H、K、V、C，先把时间维补到 C 的整数倍，再把输入写成私有常量 `memref.global`。`--npz` 可以嵌入 JAX 的输入、初始状态、余切和参考结果；`--emit fwd|bwd|both` 选择内核；`@main` 调用内核并通过 `printMemrefF32` 打印结果[tools/gdn_l1_gen.py:202-334,1391-1424]。发射器因此同时提供算法文本、可重复输入和观察入口。自然对数 g 到 log2 计算域的转换也发生在嵌入常量之前，MLIR 内部只接收转换后的 g[tools/gdn_l1_gen.py:217-255]。

种子数据由 NumPy 的 `default_rng(seed)` 产生，发射顺序由 Python 代码顺序决定[tools/gdn_l1_gen.py:66-82]。同版本生成器、同参数和同输入文件因此可以得到可逐字节比较的文本，适合用 diff 审查算法改动。内嵌数据也使模块成为独立测试程序，而非等待外部调用方传参的通用库：输出 memref 使用补零后的时间尺寸，检查器在比较前裁去 padding 行[tools/gdn_l1_check.py:283-305]。这项边界决定了证据文件既能独立运行，也需要由检查器解释其形状。

逐行文本发射的价值在于产物可直接审查：前代换循环、严格下三角谓词、状态写回顺序都与生成代码一一对应。它是一份可执行规约，提供未来 C++ 展开应对照的算法结构；当前入口 pass 的注释也明确把完整展开留待后续，而没有声称已经由 pass 生成这些循环[dialect/lib/GdnPasses.cpp:1-7]。

### 2. L1 直接书写 memref、f32 与原地状态

真实生成物中的前向签名、输出分配和状态初始化如下：

```mlir
func.func @gdn_fwd(%q: memref<1x128x2x16xf32>, %k: memref<1x128x2x16xf32>, %v: memref<1x128x2x16xf32>, %beta: memref<1x128x2xf32>, %g: memref<1x128x2xf32>, %h0: memref<1x2x16x16xf32>) -> (memref<1x128x2x16xf32>, memref<1x2x16x16xf32>) {
  // ……常量与输出 %o、%ht 的分配……
  scf.for %b = %c0 to %cB step %c1 {
    scf.for %h = %c0 to %cH step %c1 {
      %S = memref.alloc() : memref<16x16xf32>
      // ……A、T、u、w 等 scratch 分配……
      %h0h = memref.subview %h0[%b, %h, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1] : memref<1x2x16x16xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
      linalg.copy ins(%h0h : memref<16x16xf32, strided<[16, 1], offset: ?>>) outs(%S : memref<16x16xf32>)
      // ……chunk 循环……
```

[evidence/G1/l1_chunked_T128_C64.mlir:1863-1905，省略处已标出]

每个 `(b,h)` 都取得一块 K×V 的 `%S`，从 h0 的降秩视图复制入口状态，并在 chunk 循环中原地更新。这些 memref 的元素类型都是 f32；`linalg.matmul`、逐元素 `linalg.generic` 和显式 `scf.for` 共同表达第七章的数学。`memref.subview` 保留源布局的 stride 与动态 offset，无需为每个 head 或 chunk 复制一份连续输入。

memref 句柄仍是 SSA 值，但 SSA 的唯一赋值约束作用于句柄，不禁止句柄指向的内存被读写，也不禁止两个句柄指向重叠存储。`%S` 作为 `linalg.generic` 的输入与输出、A 缓冲在生命周期结束后复用，都属于内存层语义。原地状态减少了 L1 对 tensor 缓冲化分析的依赖，同时把读写顺序明确留给循环与内存操作[tools/gdn_l1_gen.py:629-677]。

阅读这类 IR 时，累加初值与写回位置同样重要。生成器在 `linalg.matmul` 或收缩型 `linalg.generic` 前对 outs 缓冲执行 `linalg.fill`：matmul named op 按 outs 累加，收缩型 generic 也在块体中显式使用累加参数；漏掉清零会把上一 chunk 的残留带入当前结果[tools/gdn_l1_gen.py:433-477,511-536]。状态更新则先算完独立的 `%prod`，再用旧 `%S` 与 `%prod` 写回 `%S`，从数据依赖上保持“右侧使用入口状态”的约定。可执行规约的检查对象因此包括分配、填充、循环边界、负载和存储，函数签名相同并不足以判定两份实现等价。

因此，管线中的 `one-shot-bufferize` 对当前输入实际为空操作：输入已经处于 memref 级。保留该 pass 是为了让未来 tensor 前端复用同一命令行，不应把当前 L1 描述成由 tensor 自动缓冲化而来[evidence/G1/l1_harness_report.md:50-68]。

### 3. 在 CPU 上观察输出

`run_l1.sh` 把生成器文本下降并交给 `mlir-runner`。完整的十一条 pass 顺序与每条职责在第十一章展开，这里只保留实际调用方式：

```bash
scripts/run_l1.sh /tmp/gdn_l1.mlir /tmp/gdn_l1.llvm.mlir
```

[scripts/run_l1.sh:19-37]

这条路径执行真实 CPU 标量 IR，不调用 GPU，也不经过 `gdn-to-chunk`。它验证生成器、下降配方、运行库和算法文本能够共同产生数值结果；未来接入 tensor 前端、其他缓冲化策略或 GPU 后端，都需要各自新增证据。

`OUT` 缺省为输入文件名后缀改成 `.llvm.mlir`，降低后的 IR 与 runner 的标准输出因而可以分别保存[scripts/run_l1.sh:15-21]。`gdn_l1_check.py` 的检查对象是 stdout 中每个 `data = [...]` 块：它将文本恢复成 NumPy 数组，再检查形状、裁剪时间 padding、计算最大绝对差，不比较 LLVM 文本[tools/gdn_l1_check.py:235-260,268-305]。所以“编译成功”“打印出数组”“数值在容差内”是三个独立检查点；只保留终端最后一行 PASS 会丢失定位失败所需的中间证据。

### 4. 三方交叉验证各自回答一个问题

JAX fp64 规约、fla naive 与 MLIR L1 形成的证据链包含不同精度和不同实现。反向规约使用向量-雅可比积（vector-Jacobian product，VJP）核对六类输入梯度。归档结果可整理为：

| 对照 | 范围与精度 | 归档最差记录 | 主要检查对象 |
|---|---|---|---|
| JAX chunk 对 sequential | 8 种形状、chunk、h0 组合，fp64 | o 5.6e-16；ht 1.2e-15 | 前向代数等价 |
| JAX chunk VJP 对 sequential VJP | 同组组合，六类梯度，fp64 | 1.2e-14 | 反向代数一致性 |
| 有限差分对 JAX VJP | 每条路径随机六坐标，fp64 | sequential 2.3e-9；chunk 9.0e-10 | 自动微分的独立数值核对 |
| JAX 对 fla naive | 四种组合，fp32 相对误差 | recurrent 1.2e-7；chunk 2.3e-6 | 移植约定与独立参考一致 |
| MLIR L1 对 JAX dumps | 三份 npz，f32 CPU | o 6.407e-7；ht 8.643e-7 | 生成 IR、下降与运行 |
| MLIR L1 反向对 JAX VJP | 三份 npz 乘六类梯度 | 6.4e-6，按 fp32 主数组 | 反向 IR 与状态反递推 |

[evidence/G1/jax_spec_report.md:12-32；evidence/G1/l1_check_output.txt:2-12；evidence/G1/l1_bwd_check_output.txt:2-12]

JAX 套件合计 44/44 PASS，覆盖前向、VJP、有限差分、JIT 一致性和 fla 交叉检查[tools/gdn_ref_check.py:47-85,87-204]。MLIR 前向七个用例均通过，常见输出差约为 6e-7；反向三份 JAX 数据的六类梯度也均通过。按 `*_fp64` 数组比较时，dbeta 的对应最大记录为 7e-6；这些数字都处于 `printMemrefF32` 的打印精度附近，不能解释为更高精度的真实计算误差上界[evidence/G1/l1_harness_report.md:75-99,274-298]。

带种子的尾块反向用例与 NumPy 孪生相比还出现 3.6e-5 的最大差，仍低于 2e-4 验收阈值[evidence/G1/l1_bwd_check_output.txt:3-12]。容差、观测值和打印地板是三个不同量：容差决定测试是否接受，观测值来自当前输出通道，打印地板限制该通道还能分辨多小的差异。

这些结果支持的结论都有范围。chunk 与 sequential 的高度一致检查代数重排；有限差分只抽查所选坐标，未对其余坐标给出结论；fla naive 内部强制 fp32，不能提供 fp64 对照；MLIR CPU 结果不覆盖 GPU 内核、bf16 路径或任意形状。方言 verifier 也只检查部分结构与配置合法域，不完整证明 shape 匹配、g≤0、归一化或数值语义[dialect/lib/GdnDialect.cpp:43-68]。

表中的数字还不能直接相加成一条严格误差上界。JAX 对 fla 使用 fp32 相对误差，MLIR 检查主要报告最大绝对差，fp64 规约与 f32 输出又处于不同舍入阶段。三方闭合的意义在于独立实现共享同一输入和约定，并在各自精度下相符；若需要新形状或新后端的误差界，必须重新规定范数、参考精度、输出通道和容差，而不能沿用这些数字作形式证明。

### 5. L0–L4 是设计与验证分层

方言描述中的层次依次是：L0 的 `gdn.fwd/gdn.bwd` 语义入口，L1 的 chunk 算法结构，L2 的切块与调度，L3 的流水和矩阵指令，L4 的 NVVM 与目标架构[dialect/include/Gdn/GdnDialect.td:13-23]。这些层次帮助安排规约、变换和验证责任。当前 `gdn-to-chunk` 只完成 L0 到 L1 外部入口的改写，完整 L1 来自 Python 生成器，后续 GPU 工件由其他生成器与脚本分阶段产生。仓库中没有一条把 L0 自动连续展开到 cubin 的单一 pass，也没有理由用 L1 CPU 结果替代后续各层验证。

生成器还提供了后续展开的差分基准。新的 C++ 展开若接入 `@__gdn_l1_*` 边界，可以固定同一份 npz，对比中间缓冲、最终 o/ht 和六类梯度，逐层定位偏离；入口声明去重与 config 透传则由方言冒烟单独检查。这样，call 合同、算法展开和后续调度可以分别获得证据，无需等待一条长管线的各阶段完成后再猜测故障位置。

第九章将拆解 KKT 与 P 的 `linalg` 下标映射，第十章追踪 `scf.for` 载态及其向 `cf` 的下降，第十一章再系统比较 tensor 与 memref，并复核这里的十一条 CPU pass。当前生成物已经位于 memref 层这一事实，会成为理解 bufferize 实际工作范围的重要基线。
