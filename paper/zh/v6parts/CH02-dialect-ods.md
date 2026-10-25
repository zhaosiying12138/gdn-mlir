## 二、方言把语义写成合同：ODS、属性、Trait 与接口

方言（Dialect）为一组操作、类型与属性提供命名空间和注册边界。声明式定义还要回答更细的问题：哪些值来自数据流，哪些选择固定在编译期，哪些结构可以机械检查，哪些能力要暴露给通用 pass。

### 1. 方言声明与 TableGen 生成物

`gdn` 方言先声明自己的名字、C++ 命名空间和依赖方言。

```tablegen
def Gdn_Dialect : Dialect {
  let name = "gdn";
  let summary = "Gated DeltaNet (gated delta rule) operator dialect";
  let cppNamespace = "::gdn";
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = [
    "tensor::TensorDialect", "linalg::LinalgDialect", "scf::SCFDialect",
    "func::FuncDialect", "mlir::NVVM::NVVMDialect", "nvgpu::NVGPUDialect",
    "gpu::GPUDialect", "arith::ArithDialect",
  ];
}
```

[dialect/include/Gdn/GdnDialect.td:13-35；描述字段省略]

`name = "gdn"` 对应 `gdn.fwd` 的前缀，`cppNamespace` 决定生成 C++ 类的位置。`dependentDialects` 保证加载本方言时准备下降所需的方言，并不表示每个操作都会使用这些词汇。`useDefaultAttributePrinterParser` 接入方言属性的默认打印与解析入口。`GdnDialect::initialize()` 仍要明确调用 `addAttributes` 和 `addOperations`，把生成类注册进方言 [dialect/lib/GdnDialect.cpp:28-36]。

操作定义规约（ODS，Operation Definition Specification）用 TableGen 记录这些声明。构建规则把 `.td` 转成 C++ 声明与实现：

```cmake
set(LLVM_TARGET_DEFINITIONS Gdn/GdnOps.td)
mlir_tablegen(GdnOps.h.inc -gen-op-decls)
mlir_tablegen(GdnOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS Gdn/GdnHatch.td)
mlir_tablegen(GdnHatchOps.h.inc -gen-op-decls)
mlir_tablegen(GdnHatchOps.cpp.inc -gen-op-defs)
mlir_tablegen(GdnHatchEnums.h.inc -gen-enum-decls)
mlir_tablegen(GdnHatchEnums.cpp.inc -gen-enum-defs)
```

[dialect/include/CMakeLists.txt:7-15]

生成代码承担访问器、builder、通用解析打印、属性存储与约束检查等样板，例如 `getQ()`、`getConfig()`。`GdnAttrDefs` 与方言本身也在同文件前段生成。TableGen 不会替作者写出完整数值语义，`hasVerifier` 对应的 C++ 检查体仍由项目维护。

生成的 `.inc` 是构建产物，修改它会在下次生成时丢失。需要改变操作签名时，应修改 `.td` 并检查生成的声明；需要改变非法输入的判定时，则修改手写 verifier。ODS 还生成 adaptor，供通用代码在不构造具体 op wrapper 的情况下访问操作数与属性。具体 op wrapper、adaptor 和底层 `Operation*` 是三种抽象层级，前两者提供强类型便利，最后一种保留遍历任意 IR 的通用性。

声明错误与输入错误发生在不同阶段。dag 写错、参数类型无法识别属于 TableGen 或 C++ 构建问题；某段 IR 的秩不符合合同则由运行中的 verifier 报告。把这两类问题分开，能避免用测试输入去掩盖一个本来就无法生成的定义。

### 2. `dag` 把类型约束、名字和槽位绑定

ODS 的参数列表使用有向无环图记录（dag）语法，每个 `约束:$名字` 同时供代码生成与模式匹配使用。前向和反向定义展示了固定、可选输入及属性的分工。

```tablegen
def GdnFwdOp : GdnOp<"fwd", [Pure]> {
  let arguments = (ins
    AnyRankedTensor:$q, AnyRankedTensor:$k, AnyRankedTensor:$v,
    AnyRankedTensor:$beta, AnyRankedTensor:$g,
    Optional<AnyRankedTensor>:$h0,
    GdnConfigAttr:$config
  );
  let results = (outs AnyRankedTensor:$o, AnyRankedTensor:$ht);
  let hasVerifier = 1;
}

def GdnBwdOp : GdnOp<"bwd", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyRankedTensor:$q, AnyRankedTensor:$k, AnyRankedTensor:$v,
    AnyRankedTensor:$beta, AnyRankedTensor:$g,
    Optional<AnyRankedTensor>:$h0,
    Optional<AnyRankedTensor>:$A,
    AnyRankedTensor:$dout,
    Optional<AnyRankedTensor>:$dht,
    GdnConfigAttr:$config
  );
  let results = (outs AnyRankedTensor:$dq, AnyRankedTensor:$dk, AnyRankedTensor:$dv,
                        AnyRankedTensor:$dbeta, AnyRankedTensor:$dg, AnyRankedTensor:$dh0);
  let hasVerifier = 1;
}
```

[dialect/include/Gdn/GdnOps.td:14-58；description 省略]

`ins` 中的张量项是操作数，代表有 use-def 边的 SSA 输入；`outs` 定义结果。`AnyRankedTensor` 只约束为有秩张量，秩、维度关系与元素类型仍有更细的合同。`config` 是属性，没有 SSA 定义点，适合保存编译期选择。`Optional<AnyRankedTensor>` 允许对应槽缺失，生成的访问器返回可空值；`Variadic<T>` 则接收任意数量的连续值。

反向操作有三个可选张量槽，单看一个扁平操作数列表无法可靠判断 `h0`、`A`、`dht` 各自是否存在。`AttrSizedOperandSegments` 为每个 ODS 操作数槽记录实际段长，解析器和访问器据此切分。当前生成接口使用 `operandSegmentSizes`，并兼容旧文本拼写 `operand_segment_sizes`；普通固定槽记 1，可选槽记 0 或 1。前向只有一个尾随可选张量，总数已经足以消歧，因此没有附加该 Trait。

段长记录属于操作结构，不能被当成随意命名的普通属性省略。九个张量槽依次记录段长后，`getH0()`、`getA()`、`getDht()` 才能返回对应值；若手工构造 IR 时给出的段长总和与实际 operands 数不一致，生成的验证逻辑会拒绝它。`Optional` 限制每槽最多一个值，`Variadic` 允许一段含多个值，两者都依靠槽边界保证访问器稳定。

可选操作数与可选属性也要分开。前者改变 SSA 实参列表，后者只改变属性是否存在；`GdnBulkWaitOp` 的 `OptionalAttr<UnitAttr>:$read` 属于后者 [dialect/include/Gdn/GdnHatch.td:150-159]。把可选标志写成 UnitAttr 可以省去运行时布尔值，同时让 pretty 格式用 `read` 关键字表达它。

hatch 是本项目给硬件入口 op 的分组名，不是 MLIR 的标准术语；这里只借 setmaxnreg 与 TMA 的定义展示 ODS 和 assembly，机器协议在第二十章展开。hatch 定义补上了可变参数与自定义操作格式的实例。

```tablegen
def GdnSetmaxnregOp : GdnHatchOp<"setmaxnreg"> {
  let arguments = (ins I32Attr:$count, GdnRegDirAttr:$direction);
  let assemblyFormat = "enum($direction) $count attr-dict";
  let hasVerifier = 1;
}

def GdnTmaAsyncLoadOp : GdnHatchOp<"tma.async.load"> {
  let arguments = (ins AnyType:$dst, AnyType:$descriptor, AnyType:$barrier,
                       Variadic<Index>:$coordinates);
  let assemblyFormat = "$descriptor `[` $coordinates `]` `,` $barrier `to` $dst attr-dict `:` type($descriptor) `,` type($barrier) `->` type($dst)";
  let hasVerifier = 1;
}
```

[dialect/include/Gdn/GdnHatch.td:32-44、102-115；description 省略]

`coordinates` 是可变长 SSA 操作数段，`count` 与 `direction` 是属性。`assemblyFormat` 中 `$coordinates` 引用参数，反引号给出字面标点，`type(...)` 控制类型打印；它定义操作的 pretty 形式。`Variadic<Index>` 已限制每个坐标为 `index`，坐标个数为一至五则由 C++ verifier 补充。

### 3. `GdnConfigAttr` 固定七个编译期参数

属性也可以用 `AttrDef` 获得强类型参数和稳定文本形式。

```tablegen
def GdnConfigAttr : AttrDef<Gdn_Dialect, "GdnConfig"> {
  let mnemonic = "config";
  let parameters = (ins
    "int":$chunk_size,
    "bool":$use_g,
    "bool":$use_gk,
    "bool":$normalize_q,
    "bool":$use_initial_state,
    "bool":$store_final_state,
    "bool":$is_varlen
  );
  let assemblyFormat = "`<` $chunk_size `,` $use_g `,` $use_gk `,` $normalize_q `,` $use_initial_state `,` $store_final_state `,` $is_varlen `>`";
}
```

[dialect/include/Gdn/GdnDialect.td:50-67；summary 与 description 省略]

七个字段依次控制 chunk 长度、标量门、逐通道门、q 归一化、初始状态、终态输出和变长序列。它们挂在操作上，生成 `getChunkSize()` 等强类型访问器，并可原样透传到下降后的调用。张量形状与元素类型属于类型系统，q、k、v 等运行时数据属于操作数，这组离散配置属于属性；把配置伪装成 SSA 输入会引入不存在的数据依赖。

这组语义配置还要与调度参数分开审查。`chunk_size` 会固定 L1 算法结构，边界开关会改变输入输出行为；tile、warp 数和流水深度只应改变同一语义合同的实现方式。修改前者需要重新执行语义与数值对照，修改后者仍需验证等价性并检查资源与性能。两者都能成为编译期属性，是否改变算子合同决定了验证责任，存储载体本身无法替代这层分类。

属性对象在上下文中不可变，操作却可以通过 `setAttr` 换挂另一个属性值。改写因此能更新配置，但不能原地修改某个共享属性对象而影响无关操作。逐字段访问器让这种更新保持类型安全，打印出的固定顺序也便于复核。

`mnemonic = "config"` 与所属方言共同形成 `#gdn.config<...>`。这里的 `assemblyFormat` 只定义属性内部的七个字段及标点，不改变 `gdn.fwd` 的操作语法。属性在 MLIRContext 中驻留并按值比较，适合缓存与差异检查；驻留优化不会替配置检查合法域。

### 4. Trait、接口与 verifier 各守一层合同

特征（Trait）提供可组合的静态事实或实现钩子。这里的 `Pure` 在当前 MLIR 中展开为 `AlwaysSpeculatable` 与 `NoMemoryEffect`，并通过内存效果接口告知通用变换：该操作可推测执行且不访问内存。它支持死代码消除、公共子表达式消除等优化，但不证明数学公式正确，也不允许忽略结果类型与 SSA 关系。

操作接口（OpInterface）向异构操作暴露统一查询，例如类型推导或内存效果查询；调用方面向接口编程，具体操作提供实现。verifier 则返回成功或失败，专门拒绝非法 IR。当前 `GdnOps.td` 与 `GdnHatch.td` 没有声明项目自定义操作接口，`hasVerifier = 1` 只生成验证钩子；包含接口头文件或链接接口库不能当作已经实现接口的证据。

结构合同也有明确边界。ODS 与 verifier 能检查部分类型、秩、配置和硬件参数，完整 shape 关系、g 的非正约束、归一化位置及数值公式仍由 `docs/L0_SEMANTICS.md`、JAX fp64、fla naive 与 L1 生成器共同维护。阅读 ODS 时还要校正至少两处旧注释：`GdnOps.td:21` 与 `GdnDialect.td:53-55` 仍写着 log2-domain；现行语义规定外部 `g` 为自然对数域，`decay = exp(g)` 且 `g≤0`，内部乘 `log2(e)` 后使用 `math.exp2` [docs/L0_SEMANTICS.md:9-10；dialect/include/Gdn/GdnDialect.td:53-55]。

最后，声明存在不等于下降已经展开。当前 `gdn-to-chunk` 只把 `gdn.fwd/bwd` 改成 `func.call @__gdn_l1_*`，透传 `gdn.config` 并去重声明；完整 L1 来自生成器 [dialect/lib/GdnPasses.cpp:38-85]。第四章将分析这段改写本身。
