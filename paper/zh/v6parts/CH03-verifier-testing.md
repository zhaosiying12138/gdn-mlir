## 三、先让非法程序报错：Verifier、诊断与 IR 测试

验证器（verifier）把非法 IR 挡在编译早期，并给出能定位、能回归的诊断。它能保证的范围取决于实际检查代码：通过验证表示已满足已实现的结构合同，不能由此推出算子的数学与数值行为正确。

### 1. ODS 约束与 C++ 检查接续工作

ODS 先生成机械约束，例如 `AnyRankedTensor` 要求有秩张量，`I32Attr` 要求 32 位整数属性，`Variadic<Index>` 要求每个坐标都是 `index`。C++ verifier 接着检查需要遍历多个字段或表达合法域的条件。

```cpp
static LogicalResult verifyConfig(gdn::GdnConfigAttr config, Operation *op) {
  if (!llvm::is_contained(gdn::kAllowedChunkSizes, config.getChunkSize()))
    return op->emitError() << "chunk_size must be one of {16,32,64}, got "
                           << config.getChunkSize();
  return success();
}

LogicalResult gdn::GdnFwdOp::verify() {
  if (failed(verifyConfig(getConfig(), *this)))
    return failure();
  auto q = mlir::dyn_cast<RankedTensorType>(getQ().getType());
  auto v = mlir::dyn_cast<RankedTensorType>(getV().getType());
  auto beta = mlir::dyn_cast<RankedTensorType>(getBeta().getType());
  if (!q || q.getRank() != 4)
    return emitError() << "q must be rank-4 [B,T,H,K]";
  if (!v || v.getRank() != 4)
    return emitError() << "v must be rank-4 [B,T,H,V]";
  if (!beta || beta.getRank() != 3)
    return emitError() << "beta must be rank-3 [B,T,HV]";
  return success();
}
```

[dialect/lib/GdnDialect.cpp:43-62]

合法 chunk 集合来自 `kAllowedChunkSizes = {16, 32, 64}` [dialect/include/Gdn/Gdn.h:31-32]，前向与反向都调用 `verifyConfig`。前向额外检查 q、v 为秩 4、beta 为秩 3；反向的 `verify()` 在第 65-69 行只复核配置。`emitError()` 把消息附着到操作的源位置，返回的 `failure()` 沿 `LogicalResult` 传给框架。解析后的验证和 pass 管线中的验证都使用同一套钩子，错误因此能指向具体 IR 行。

这里能看到明确的未覆盖区域：k、g、h0 的秩，跨操作数的维度一致性，元素类型组合，配置开关之间的依赖均未完整核对；`g≤0`、归一化、衰减顺序与梯度公式也不在 verifier 中证明。测试设计必须针对这些空白另建数值与语义检查。

有效的结构测试需要正负例成组。chunk 的三个合法值都应各有一次正常解析，15、48、128 可分别覆盖不在集合、落在常见倍率但仍不在集合、超出集合的情况。仅测一个非法值不能证明集合边界完整，仅测合法值也无法发现 verifier 根本没有被调用。每个负例还应断言关键消息与位置，防止测试因另一个无关解析错误而偶然通过。

自定义 verifier 与 ODS 约束的先后顺序也值得控制。若 ODS 已能保证操作数是张量，C++ 可以专注于秩和跨字段关系；若需要给出比通用类型错误更清楚的领域诊断，就应在定义中保留足够的信息。所有分支都应返回明确的 `LogicalResult`，避免检查失败后仍继续访问不成立的类型或字段。

### 2. 硬件入口更需要早期范围诊断

硬件直通操作把指令约束前移到 IR 层，避免非法参数直到汇编阶段才以底层报错出现。

```cpp
LogicalResult GdnSetmaxnregOp::verify() {
  if (getCount() < 24 || getCount() > 256 || getCount() % 8 != 0)
    return emitError() << "setmaxnreg count must be a multiple of 8 in "
                          "[24, 256], got "
                       << getCount();
  return success();
}

static LogicalResult verifyTmaCoords(Operation *op, ValueRange coords) {
  if (coords.empty() || coords.size() > 5)
    return op->emitError() << "tma coordinates must be 1..5 dims, got "
                           << coords.size();
  return success();
}

LogicalResult GdnTmaAsyncLoadOp::verify() {
  return verifyTmaCoords(*this, getCoordinates());
}
```

[dialect/lib/GdnHatch.cpp:246-263；store 的同构检查在 265-267 行]

`setmaxnreg` 的 count 必须同时满足闭区间 `[24, 256]` 与 8 的倍数：24、32、256 合法，16、25、260 非法。枚举属性已经限制方向为已声明值，C++ 再检查数值范围与步进。TMA load 和 store 复用坐标检查，零个或六个坐标都会报错，一至五个通过这一层；描述符内容、张量形状、运行时越界与异步完成协议仍需后续合同保证。

### 3. 期望诊断与结构输出是两种 IR 断言

`--verify-diagnostics` 用于负例。源码在预期出错的位置写 `// expected-error {{消息片段}}`，也可用 `expected-error@+1` 指向下一行；预期诊断缺失或出现额外诊断都会使测试失败。它检查诊断级别、位置和消息匹配，适合固定 chunk 越界、TMA 零坐标这类行为。正常输入无需错误注释，过度宽泛的消息匹配则会放过误诊。

FileCheck 面向命令标准输出。`CHECK`、`CHECK-SAME`、`CHECK-NOT` 等指令匹配文本片段和正则捕获，适合检查 roundtrip 后属性仍在、改写后调用符号存在、某个高层操作已经消失。它不会执行 IR，也不会自行比较浮点数组。`lit` 是测试运行器，负责发现测试、展开 substitutions、执行 `RUN` 行并汇总状态；FileCheck 通常只是其中一条命令。三者的职责可以组合，不能用一个 `PASS` 代替所有层级。

写结构断言时应匹配合同，少匹配打印细节。用变量捕获一个 SSA 名称再检查其使用，比固定 `%0` 更耐打印重编号；`CHECK-LABEL` 可隔离函数，`CHECK-DAG` 可表达一组无序出现项，`CHECK-NOT` 可约束相邻匹配之间不能出现的文本。另一方面，只写 `CHECK: "gdn.fwd"` 无法证明属性正确，只写函数名也无法证明调用参数顺序。断言应覆盖会支撑后续改写的关键结构。

诊断测试同样需要精度。普通 `expected-error` 检查消息片段，`expected-error-re` 才显式使用正则；位置设计符把期望绑定到相应源码行。一个文件中多个负例常配合 `--split-input-file` 隔离，防止第一个非法操作影响后续解析。错误消息可以包含实际值与期望值，例如 count 为 25，但测试最好保留稳定的领域短语，避免把完整格式字符串固化成脆弱快照。

### 4. 数值测试与 GPU 冒烟回答不同问题

真实 L1 检查先生成并执行 MLIR，再把输出数组与参考实现逐元素比较。

```python
diff_o = float(np.max(np.abs(o_m[:, :T, :, :] - o_ref)))
diff_ht = float(np.max(np.abs(ht_m - ht_ref)))
ok = diff_o < PASS_ATOL and diff_ht < PASS_ATOL
return {
    "name": name, "shape": f"B{B} T{T} H{H} K{K} V{V} C{C}",
    "diff_o": diff_o, "diff_ht": diff_ht,
    "algo_o": algo_check, "algo_ht": algo_ht, "ok": ok,
}
```

[tools/gdn_l1_check.py:299-306]

这里判定的是最大绝对误差是否越界，IR 是否漂亮并不影响结果。相邻的 `run_npz_case` 使用 JAX 转储参考值，反向检查还覆盖六个梯度 [tools/gdn_l1_check.py:309-341、345-385]。文档、JAX fp64、fla naive 与生成的 MLIR L1 通过可执行交叉测试约束同一份数值语义，verifier 不承担这层证明。容差、有限值和 shape 都必须单独判定。

一个结构完全合法的程序仍可能算错。外部语义的 `g` 位于自然对数域，内部需要乘 `log2(e)` 后调用 `math.exp2`；漏掉比例变换不会自动改变秩、类型或操作数个数，verifier 因而可以接受错误程序，FileCheck 也能继续匹配到 `math.exp2`。只有执行并比较输出才会暴露这类偏差。数值测试还应拒绝 NaN 和 Inf，并按任务记录绝对误差、相对误差或位级条件；仅检查进程返回码不足以判定正确。

参考链也要避免共同误解。让同一个错误公式同时充当生成器和参考，只会得到稳定的一致错误，所以项目保留 JAX、fla naive 与 MLIR L1 的独立对照。容差用于界定可接受的浮点差异，不能用来掩盖持续的符号错误或衰减顺序错误。

GPU 冒烟属于另一类集成测试，专门检查 transform→NVVM→cubin→运行链；sm_80 输入文件如何重定向到 sm_120a、runner 的输出如何核对，在第十九章展开。这里先固定分工：结构测试看 IR，数值测试看输出，二者不能互相替代。

### 5. 【练习】构造一个可执行的 L0 回归小例

当前 `dialect/test/` 只有 `smoke_l0.mlir`，没有配置完整的项目自有 lit/FileCheck 与 `verify-diagnostics` 套件。以下内容是新练习，需要在临时文件中运行，并非仓库已有测试；命令假定 `gdn-opt` 与 `FileCheck` 已在 PATH 中。

```mlir
// RUN: gdn-opt %s | gdn-opt | FileCheck %s
// RUN: gdn-opt %s --gdn-to-chunk | FileCheck %s --check-prefix=CALL

!qt = tensor<1x4x1x8xf32>
!bt = tensor<1x4x1xf32>
!st = tensor<1x1x8x8xf32>

func.func @roundtrip(%q: !qt, %k: !qt, %v: !qt,
                     %beta: !bt, %g: !bt) -> (!qt, !st) {
  %o, %ht = "gdn.fwd"(%q, %k, %v, %beta, %g)
      <{config = #gdn.config<16, true, false, false, false, true, false>}>
      : (!qt, !qt, !qt, !bt, !bt) -> (!qt, !st)
  return %o, %ht : !qt, !st
}

// CHECK: "gdn.fwd"
// CHECK: #gdn.config<16, true, false, false, false, true, false>
// CALL: func.func private @__gdn_l1_fwd_chunk
// CALL: call @__gdn_l1_fwd_chunk
// CALL: #gdn.config<16, true, false, false, false, true, false>
```

第一条 RUN 完成解析、打印、再次解析与结构匹配，第二条固定当前的 L1 调用边界及配置透传。另取一个只保留正常输入的副本，把 chunk 从 16 改成 48，并在 `gdn.fwd` 前一行加入 `// expected-error@+1 {{chunk_size must be one of}}`，再以 `gdn-opt --verify-diagnostics` 运行，就得到一个负例。这个练习仍不验证 GDN 数值；要覆盖数值，必须执行生成的 L1 并与独立参考比较。
