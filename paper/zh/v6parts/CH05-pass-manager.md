## 五、许多 Pass 如何协同：PassManager、Analysis 与可观察性

一条重写规则即使局部正确，仍可能因运行锚点不合适、嵌套顺序含混或分析缓存过期而出错。PassManager（pass 管理器）负责把多条 pass 组成有作用域的执行序列；AnalysisManager（分析管理器）负责在这条序列中计算、复用并失效分析结果。两者连接的是 IR 的生命周期，而不只是函数调用顺序。

pass 与 rewrite pattern 的遍历层级也需要区分。pattern 决定局部匹配与替换，pass 决定这次遍历从哪枚 op 获得授权，PassManager 再决定不同 pass 的先后和嵌套。把这三层分开后，`walk`、贪心驱动器和函数级适配器就不会被误当成同一种调度机制。

### 1. OperationPass 的锚点限定作用域

`OperationPass<OpT>` 的模板参数是锚点（anchor）：pass 在一枚指定类型的 op 上运行，`getOperation()` 也返回对应类型。`gdn-to-chunk` 选择 `ModuleOp`，因为它一边改写嵌套在函数里的 `gdn.fwd/gdn.bwd`，一边向父模块插入被调函数声明。

```cpp
struct GdnToChunkPass
    : public PassWrapper<GdnToChunkPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GdnToChunkPass)
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGdnToChunkPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  StringRef getArgument() const override { return "gdn-to-chunk"; }
};
```

[dialect/lib/GdnPasses.cpp:93-105]

模块锚点使声明插入与调用改写都位于同一棵授权修改的子树内。`ensureL1Entry` 先查符号，再在模块体开头建立私有声明；贪心驱动器随后可以处理模块里的多个前向或反向 op，而不会重复声明同名函数[dialect/lib/GdnPasses.cpp:26-36,38-85]。若把锚点改成 `func::FuncOp`，向父模块写声明就越过了函数 pass 的作用域，也会破坏函数级管线可并行处理独立函数的假设。锚点选择因此同时回答“收到哪枚 op”“允许修改哪棵子树”“会被执行多少次”。

锚点还决定分析的自然归属。模块级符号表分析应看到整个模块，函数级支配分析只应依赖当前函数；函数子管线可由适配器分发到多枚函数，模块 pass 则形成它们之间的串行边界。把需要跨函数写符号表的逻辑塞进函数 pass，既扩大实际修改范围，也使缓存该放在哪一层变得含混。`gdn-to-chunk` 的模块选择把改写权限、符号生命周期和分析归属放在了同一层。

### 2. 嵌套管线把执行次序写成结构

`OpPassManager` 保存同一锚点下的一串 pass，`nest<OpT>()` 在其下建立子管线；跨锚点时，适配器遍历匹配的内层 op 并调度子管线。文本形式直接把这套树写出来：

```bash
build-dialect/bin/gdn-opt dialect/test/smoke_l0.mlir \
  --pass-pipeline='builtin.module(gdn-to-chunk,func.func(canonicalize,cse),gdn-lower-hatch)' \
  --dump-pass-pipeline -o /tmp/gdn.pipeline.mlir
```

[dialect/tools/gdn-opt/gdn-opt.cpp:15-30；dialect/lib/GdnPasses.cpp:102-110；dialect/lib/GdnHatch.cpp:223-240]

外层先在模块锚点运行 `gdn-to-chunk`，再对每个 `func.func` 运行规范化与公共子表达式消除，最后回到模块层运行 hatch 下降。显式 `--pass-pipeline` 按文本建树，解析期间关闭隐式嵌套；漏写 `func.func(...)` 且锚点不匹配时会直接报错[上游 llvm-mlir/mlir/lib/Pass/PassRegistry.cpp:607-617,1002-1027]。`--dump-pass-pipeline` 用来核对实际树形，避免凭命令行外观猜测作用域。对本例运行后，日志同时给出 canonicalize 的配置，能看到 `cse-between-iterations=false` 这一默认值：

```text
Pass Manager with 3 passes:
builtin.module(
  gdn-to-chunk,
  func.func(
    canonicalize{cse-between-iterations=false ...},
    cse
  ),
  gdn-lower-hatch
)
```

[paper/zh/v6examples/pass-pipeline.txt:1-10；`...` 表示省略其余默认参数]

显式管线与逐项 pass 旗标属于两种输入方式，同一次调用不能混用；上游解析器会对这种组合报错[上游 llvm-mlir/mlir/lib/Pass/PassRegistry.cpp:1002-1026]。调试顺序变化时，应比较完整文本树和模块边界，而不只比较旗标列表。尤其是函数子管线前后的两个模块 pass，它们之间可能已发生各函数上的改写；将最后的模块 pass 移到函数子管线之前，观察到的符号和 IR 都可能不同。

逐项传 pass 旗标时，`MlirOptMain` 从解析所得根 op 的名字创建 PassManager，并采用 `Nesting::Implicit`，随后应用通用命令行选项、装配管线并执行[上游 llvm-mlir/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:591-604]。项目的 L1 CPU 脚本正使用这种形式：

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

[scripts/run_l1.sh:19-35]

这里 `$OPT` 是上游 `mlir-opt`，因为生成后的 L1 只含树内方言。顺序具有语义：结构化计算先降为循环，控制流、数学运算、memref 和函数再分别降到 LLVM 可表达的形式，最后清理过渡 cast。项目没有另写一个 C++ 驱动来组装这条链；`gdn-opt` 也只向 `MlirOptMain` 注册方言与 pass[dialect/tools/gdn-opt/gdn-opt.cpp:15-30]。第六章会说明两个可执行文件的链接关系，第八章再检查这条 CPU 链的数值证据。

### 3. AnalysisManager 的懒计算、缓存与失效

分析（analysis）是从当前 IR 计算出的只读视图，例如操作计数、支配关系或别名信息。`getAnalysis<T>()` 首次请求时构造 `T` 并放入当前锚点的 AnalysisMap；后续请求命中同一缓存。构造发生在请求时，管线启动不会预先计算任何分析。缓存命中也不会再次调用分析构造器[上游 llvm-mlir/mlir/include/mlir/Pass/AnalysisManager.h:132-175,193-215,287-368]。

缓存键同时包含分析类型与所依附的 operation，而非全局只有一份 T。嵌套 AnalysisManager 为子 op 建立独立 AnalysisMap，父级与子级可通过明确的接口查询，不能把某函数的分析结果误用到另一函数。`getChildAnalysis` 会按需建立子级结果，`getCachedParentAnalysis` 只读取已经存在的父级缓存[上游 llvm-mlir/mlir/include/mlir/Pass/AnalysisManager.h:299-360]。这种布局与 PassManager 的嵌套树相对应，也使失效可以限制在受影响的 IR 层级。

每条 pass 结束后，PassManager 根据该 pass 声明的 `PreservedAnalyses` 清理缓存。默认规则很保守：未明确保留的分析失效，下一次 `getAnalysis<T>()` 重新计算。`markAnalysesPreserved<T>()` 是对分析语义的承诺，`markAllAnalysesPreserved()` 则承诺每份已缓存分析仍有效。只改属性而不改变操作计数时可以保留 CountOps；增加一条 call 后仍保留它，就会让后续 pass 读到旧计数。分析依赖也需要进入失效条件，上游单元测试展示了依赖分析同时检查自身与依赖项是否保留的做法[上游 llvm-mlir/mlir/unittests/Pass/AnalysisManagerTest.cpp:173-204]。

【通用补充】下面的 CountOps 展示完整的最小协议。它使用上游 `Operation::walk` 遍历当前 op 及其嵌套 op；walk 是 IR 遍历接口，不负责 pass 调度或缓存管理。上游默认采用后序遍历，也允许按具体 op 类型筛选[上游 llvm-mlir/mlir/include/mlir/IR/Operation.h:813-850]。

```cpp
struct CountOpsAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CountOpsAnalysis)
  explicit CountOpsAnalysis(Operation *root) {
    root->walk([&](Operation *) { ++count; });
  }
  size_t count = 0;
};

struct CountingPass
    : PassWrapper<CountingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CountingPass)
  void runOnOperation() override {
    auto &counts = getAnalysis<CountOpsAnalysis>();
    llvm::outs() << counts.count << "\n";
    // 仅当本 pass 没有改变 op 数量时保留；改写 op 数量时删除下一行。
    markAnalysesPreserved<CountOpsAnalysis>();
  }
};
```

【通用补充，接口依据：上游 llvm-mlir/mlir/include/mlir/Pass/Pass.h:228-260；上游 llvm-mlir/mlir/include/mlir/Pass/AnalysisManager.h:82-97】

失效例可以直接沿这条协议推演。第一个 pass 请求 CountOps，得到 N；只读 pass 明确保留后，下一个读者复用 N。某条改写 pass 新增两个 op 且不作保留声明，PassManager 删除缓存，后续读者重新 walk，得到 N+2。若改写 pass 错标保留，后续读者仍看到 N。`getCachedAnalysis<T>()` 只查询现有缓存，适合观察或可选优化，不能替代需要结果时的 `getAnalysis<T>()`；缓存被清理后，旧引用也不应继续使用。

项目另有一个外部 Python 求解器：它读取阶段、形状和旋钮域，构造约束并输出排名文件，输入中没有 `Operation*`，生命周期由 Python 进程和显式脚本调用管理[tools/gdn_solver.py:1-26,81-110]。它不是 MLIR Analysis，不注册进 AnalysisManager，也不随 IR pass 自动失效或保留。当前项目没有自写 CountOps 这类 MLIR Analysis；项目机制是 `MlirOptMain` 加命令行管线，上游在幕后提供分析基础设施。具体求解器模型在第十五至十八章展开。

### 4. 用插桩（instrumentation）观察执行，而非猜测执行

`PassInstrumentation` 在管线、pass 和分析的前后提供回调，另有 pass 失败回调。IR 打印与计时都复用这套观察点；分析回调只包围真正的构造过程，因此还能区分缓存命中与重新计算[上游 llvm-mlir/mlir/include/mlir/Pass/PassInstrumentation.h:25-86；上游 llvm-mlir/mlir/include/mlir/Pass/AnalysisManager.h:195-214]。项目未安装自定义 instrumentation，直接使用 `MlirOptMain` 注册的命令行能力即可。

```bash
build-dialect/bin/gdn-opt dialect/test/smoke_l0.mlir \
  --gdn-to-chunk --verify-each \
  --mlir-print-ir-after-all \
  --mlir-pass-statistics \
  -o /tmp/gdn.after.mlir 2>/tmp/gdn.pass.log
```

[dialect/test/smoke_l0.mlir:1-15；上游 llvm-mlir/mlir/lib/Pass/PassManagerOptions.cpp:36-85,145-175]

`--mlir-print-ir-after-all` 在每条 pass 后打印 IR，日志写入 stderr，与 `-o` 的最终 IR 分开；只需要变化快照时可加 `--mlir-print-ir-after-change`。嵌套函数管线下若要始终打印顶层模块，可配 `--mlir-print-ir-module-scope` 与 `--mlir-disable-threading`，避免并发修改期间的模块级读取[上游 llvm-mlir/mlir/lib/Pass/IRPrinting.cpp:49-77,80-138；上游 llvm-mlir/mlir/lib/Pass/PassManagerOptions.cpp:166-174]。

`--mlir-pass-statistics` 输出各 pass 已注册的 `Pass::Statistic` 计数，并可按 pipeline 或合并列表显示[上游 llvm-mlir/mlir/lib/Pass/PassStatistics.cpp:28-55,105-160]。它没有自动推导逐模式命中率的义务；自定义 pass 未登记计数器时，应结合 IR 前后快照判断改写是否发生。`--verify-each` 则在每条变换后检查 IR 合法性，使“结构已损坏”和“数值语义已偏离”尽早分成两类问题。

几种观察通道应交叉使用。IR 快照定位哪条 pass 改变了哪个 op，统计计数确认已登记动作发生了多少次，`--mlir-timing` 帮助发现耗时集中在变换还是分析构造。统计为零并不自动证明 pass 无效，它也可能只做了该 pass 未计数的工作；IR 文本一致也不自动证明数值测试充分。把日志、最终 IR 和第八章的 CPU 数值检查、第二十八章的完整验证矩阵放在同一证据路径中，才能从“执行过什么”进一步追到“结果是否可信”。
