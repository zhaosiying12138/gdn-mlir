## 十一、Tensor 到 Memref：Bufferize 真正决定了什么

张量（tensor）值只有形状、元素类型和值语义；一次“修改”会产生新的 tensor 值，旧值的后续读取仍看到旧内容。内存引用（memref）把计算放到具体存储位置，load/store 通过地址读写内容。缓冲化（bufferize）需要为原来的值计算选择存储、分配、拷贝和复用方式，同时保持可观察结果不变。

这段过渡改变了优化的提问方式。tensor 数据流主要追问某个值由谁产生、被谁使用；memref 数据流还要追问某次 load 与先前哪次 store 指向同一位置，以及两个写操作是否重叠。一个 tensor 结果对应一次 alloc 只是常见选择，分析也可能复用某个输入或中间缓冲。反过来，看到两条 alloc 也不能推断后续变换永远不会引入别名，机器层的存储分配还可以另有布局策略。

拷贝同样不能按数量孤立评价。若旧 tensor 值已经没有任何读取者，原地写通常可以保留结果语义；若函数边界、其他返回值或后续 op 仍观察旧值，就需要保留原存储或另建副本。缓冲化把这些条件落实到具体 op，后续优化才能基于明确的内存效果继续工作。

### 1. 值、位置、布局与地址空间

memref 的 SSA 句柄只有一个定义点，这个性质约束句柄的 def-use，不限制底层存储是否重叠。两个句柄可以引用同一缓冲，一个句柄也可以同时出现在 `ins` 与 `outs` 中。别名（alias）因此是内存分析和变换必须处理的事实。不能根据 SSA 名称不同就推断存储不相交，也不能仅凭类型相同就推断可以复用。

几个 op 对应不同的存储决定。`memref.alloc` 产生新缓冲的句柄，CPU 堆缓冲通常在生命周期末端匹配 `memref.dealloc`，目标管理的内存空间则遵循相应释放合同；`memref.copy` 复制内容，保留源缓冲的独立值；`memref.subview` 建立原存储上的视图，本身不复制元素。复用已有缓冲可以减少分配和流量，但只有在旧值的所有读取完成后才保持 tensor 的值语义。缓冲化的核心检查正是这些读取、写入与生存期关系。

视图类型还记录布局。偏移（offset）给出起点相对基地址的位置，步长（stride）给出各维下标增加一时跨过多少元素，动态 offset 或 stride 在类型中写作 `?`。真实 L1 的连续两层切片展示了这些信息怎样随视图传播：

```mlir
%k_bh = memref.subview %k[%b, 0, %h, 0] [1, 128, 1, 16] [1, 1, 1, 1]
  : memref<1x128x2x16xf32>
    to memref<128x16xf32, strided<[32, 1], offset: ?>>
%k_c = memref.subview %k_bh[%off, 0] [64, 16] [1, 1]
  : memref<128x16xf32, strided<[32, 1], offset: ?>>
    to memref<64x16xf32, strided<[32, 1], offset: ?>>
```

[evidence/G1/l1_bwd_chunked_T128_C64.mlir:2494,2508；排版有折行]

读 `%k_bh` 的形状和步长可以判断：时间维每前进一个元素要跨过 H×K=32 个 f32，特征维保持连续；`offset:?` 表示偏移依赖 `%b`、`%h`，并非运行时缺少值。第二层选择 64 行后仍保留 `[32,1]`，说明 chunk 视图没有变成紧凑的 64×16 副本。最后的步长列表 `[1,1]` 是相对源步长的倍率。KKT 直接读取这个 strided 视图，因此转置来自 linalg 映射，物理布局仍由 memref 类型描述。

布局之外还有内存空间（memory space）。它说明存储属于哪个目标地址空间，影响可执行的访问 op 和后续下译。项目的 GPU 生成器把 tile 类型明确标为 workgroup：

```python
sa_ty = f"memref<{bt}x{bk}x{dt}, #gpu.address_space<workgroup>>"
sb_ty = (f"memref<{bv}x{bk}x{dt}, #gpu.address_space<workgroup>>" if kt else
         f"memref<{bk}x{bv}x{dt}, #gpu.address_space<workgroup>>")
```

[tools/gdn_l2_gen.py:422-424]

这里要区分形状、元素类型、步长和地址空间四种信息；workgroup 标注不会自动把全局输入复制到共享内存，实际分配、拷贝与同步仍需相应 IR。CPU L1 未附此属性，使用默认内存空间。

### 2. 当前 L1 上的 one-shot-bufferize 是 no-op

`one-shot-bufferize` 通过缓冲化接口和数据分析统一处理 tensor 数据流，决定哪些结果可原地写入、何处需要分配或拷贝；`bufferize-function-boundaries` 还转换函数边界的 tensor 类型。当前 GDN L1 从生成器起点已经包含 `memref.alloc/subview/load/store`，算法直接使用 f32 memref 和原地状态，所以这条 pass 对该输入没有 tensor 可转换，实际为 no-op。保留它是为了让将来的 tensor 前端接入同一管线，不能据此声称现有 L1 曾经由 tensor 自动展开。

完整的 11 条命令如下，顺序构成输入合同：

```bash
"$OPT" "$IN" \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -expand-strided-metadata \
  -lower-affine \
  -convert-math-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-cf-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o "$OUT"
```

[scripts/run_l1.sh:23-35；no-op 记录见 evidence/G1/l1_harness_report.md:60-62]

逐条看输入、输出和相互依赖：

| 顺序与 pass | 输入到输出 | 需要它的原因及顺序约束 |
|---|---|---|
| 1. one-shot-bufferize | tensor 计算及边界转为 memref 计算及边界，按需引入分配和拷贝 | 给后续循环与 LLVM 下译提供位置语义；当前 L1 空转，放在最前便于接入 tensor 前端 |
| 2. convert-linalg-to-loops | linalg op 转为 scf 循环、memref 访问和标量计算 | 在缓冲化后处理目标存储，并把迭代空间展开，供下一条处理 SCF |
| 3. convert-scf-to-cf | scf.for、if、while 转为基本块、块参数和 cf 分支 | 接收上一条产生的循环；先消除结构化控制，后续才能统一转换 CF |
| 4. expand-strided-metadata | subview、形状折叠和元数据提取等复杂 memref op 展开为显式的 offset、size、stride 计算与较简单操作 | 当前降秩 strided subview 需要它；它使用 affine 构造，故排在 lower-affine 前，也排在 memref 最终下译前 |
| 5. lower-affine | affine.apply 变为 arith 算术，affine.for、if 变为 SCF | 消除上一步及输入中的仿射构造，让 arith 下译看到普通标量运算 |
| 6. convert-math-to-llvm | math 运算转为 LLVM intrinsic、运算或相应调用 | 本例把 math.exp2 降为 llvm.intr.exp2；在最终函数与控制流下译前处理数学语义 |
| 7. convert-arith-to-llvm | arith 整数、index 和浮点运算转为 LLVM 方言运算 | 接收 lower-affine 与各循环展开产生的标量算术，避免遗留高层算术 |
| 8. finalize-memref-to-llvm | memref 描述符、全局量、分配释放和访存转为 LLVM 类型、指令与运行时调用 | 复杂视图已展开后再完成存储下译；函数签名转换也需要一致的存储类型表示 |
| 9. convert-cf-to-llvm | cf.br、cf.cond_br 等转为 LLVM 基本块与分支 | 放在 SCF 已降为 CF 之后，接管最终控制流 |
| 10. convert-func-to-llvm | func 函数、调用、返回和签名转为 LLVM 对应形式 | 在操作数与 memref 表示明确后统一函数边界，形成可调用的 LLVM 方言模块 |
| 11. reconcile-unrealized-casts | 折叠往返后类型相同的 unrealized_conversion_cast 链 | 等各方言转换都接通后再清理过渡桥；它不能补救缺失的算术、函数或内存转换 |

`lower-affine` 可能产生新的 `scf.for` 或 `scf.if`，当前命令却已经完成 `convert-scf-to-cf`。现有 L1 所需路径可以沿此顺序通过；若前端加入 `affine.for` 或 `affine.if`，就要检查它出现的位置及 lower-affine 的产物，可能需要把 lower-affine 提前，或在其后再次执行 SCF 到 CF。判断依据是每一步后的真实 IR 与转换目标合法性，不能只凭 pass 名称复用顺序。

### 3. JIT 入口与 libm

下降后的模块仍包含 LLVM 方言和运行时符号，`mlir-runner` 负责即时编译（JIT）并调用入口：

```bash
LIBS="$BUILD/lib/libmlir_runner_utils.so,$BUILD/lib/libmlir_c_runner_utils.so,$BUILD/lib/libmlir_float16_utils.so"
"$RUNNER" "$OUT" --shared-libs="$LIBS" --entry-point-result=void
```

[scripts/run_l1.sh:20-21,37]

`runner_utils` 与 `c_runner_utils` 提供打印等运行支持，`float16_utils` 提供半精度辅助；共享运行库同时把 libm 带入符号解析环境。当前 `math.exp2` 经 `llvm.intr.exp2` 后由 JIT 对 libm 解析，无需另加 `convert-math-to-libm`。`--entry-point-result=void` 与生成器 main 的返回约定一致。JIT 能执行只说明整条 ABI、符号与 IR 下译链闭合，数值正确性仍由独立规约对拍确认。

### 4. 【通用补充】真实 one-shot-bufferize 输出

下面的 tensor 输入同时返回插入后的新值 `%y` 与原值 `%x`。这个例子是教程补充，用来观察旧值仍被读取时的存储决策，不属于项目的 GDN 内核：

```mlir
func.func @insert_keep_old(%x: tensor<4xf32>, %v: f32)
    -> (tensor<4xf32>, tensor<4xf32>) {
  %i = arith.constant 1 : index
  %y = tensor.insert %v into %x[%i] : tensor<4xf32>
  return %y, %x : tensor<4xf32>, tensor<4xf32>
}
```

对这份输入实际执行 `mlir-opt -one-shot-bufferize="bufferize-function-boundaries"`，输出为：

```mlir
module {
  func.func @insert_keep_old(%arg0: memref<4xf32, strided<[?], offset: ?>>, %arg1: f32)
      -> (memref<4xf32>, memref<4xf32, strided<[?], offset: ?>>) {
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() alignment = 64 : memref<4xf32>
    memref.copy %arg0, %alloc : memref<4xf32, strided<[?], offset: ?>> to memref<4xf32>
    memref.store %arg1, %alloc[%c1] : memref<4xf32>
    %cast = memref.cast %alloc : memref<4xf32> to memref<4xf32, strided<[?], offset: ?>>
    return %alloc, %arg0 : memref<4xf32>, memref<4xf32, strided<[?], offset: ?>>
  }
}
```

[paper/zh/v6examples/insert_keep_old.bufferized.mlir:1-9；由上述命令对 insert_keep_old.mlir 实际生成]

第二个返回值仍要求 `%x` 的原内容，所以直接在 `%x` 上 store 会破坏结果。pass 新分配 `%alloc`，先完整 copy，再对副本写入；`%arg0` 原样返回，函数边界的 tensor 类型则换成带动态 stride/offset 的 memref。若旧值没有后续使用且接口允许写，分析可能选择复用并减少拷贝；判断条件来自使用链与生存期，而非操作数名称。这个真实输出也把“当前 L1 已是 memref 而 bufferize 空转”和“真正 tensor 前端需要分配与复制”区分开了。

各转换 pass 之间还需要一致的类型合同与合法性判定。第十二章将用同一组脚本和项目自有两个改写 pass 检查渐进式下降的实际边界。
