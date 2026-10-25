## 四、改写如何收敛：RewritePattern、贪心驱动与规范化

重写模式（RewritePattern）把局部 IR 变化写成可复用规则，驱动器负责选择规则、跟踪变化并处理后续暴露的机会。局部规则的正确性与整次改写的收敛性相互依赖，必须同时检查。

### 1. `FwdToLowering` 一次只替换一个语义入口

前向下降模式完整展示了从匹配到结果重接线的过程。

```cpp
struct FwdToLowering : OpRewritePattern<gdn::GdnFwdOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(gdn::GdnFwdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> args{op.getQ(), op.getK(), op.getV(), op.getBeta(),
                            op.getG()};
    if (op.getH0())
      args.push_back(op.getH0());
    auto fnTy = rewriter.getFunctionType(ValueRange(args).getTypes(),
                                         op.getResultTypes());
    if (auto module = op->getParentOfType<ModuleOp>())
      ensureL1Entry(module, rewriter, kL1FwdSym, fnTy);
    auto callee = rewriter.getStringAttr(kL1FwdSym);
    auto call = rewriter.create<func::CallOp>(loc, callee,
                                              op.getResultTypes(), args);
    call->setAttr("gdn.config", op.getConfig());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};
```

[dialect/lib/GdnPasses.cpp:38-59]

`OpRewritePattern<gdn::GdnFwdOp>` 让驱动器只向该模式递交匹配类型的操作。进入 `matchAndRewrite` 后，模式应只在不适用且尚未改写时返回 `failure()`；完成变化才返回 `success()`。五个必备操作数先进入实参向量，可选 `h0` 按槽位存在性追加，函数类型直接由实参和原结果类型构造。这里没有展开 chunk 算法，目标仍是 L1 外部调用。

`PatternRewriter` 是修改 IR 的受控入口。`create` 插入 `func.call` 并继承原位置，`setAttr` 透传配置，`replaceOp` 把旧结果的每个使用改接到新 call 的对应结果，再删除旧操作。保留 `Location` 能让后续诊断回溯到源位置；沿用 `getResultTypes()` 则把类型变化留给明确的转换机制。绕开 rewriter 直接改操作数或删除节点，会使驱动器错过需要重新处理的值与操作。

替换前必须检查结果数量与类型是否对应。旧操作可能已被嵌套区域中的用户使用，也可能和模块中其他操作共享操作数；`replaceOp` 只替换旧 results 的使用，不会复制或删除这些输入值。新 call 仍需支配所有原使用点，在原操作附近创建它正好维持这一条件。若把新操作插入到不相干的区域，单纯的 API 调用成功也不能保证最终 IR 合法。

模式集在交给驱动器前会被冻结，驱动器可以据此建立匹配索引。模式本体应把可变状态放在单次调用或外部只读上下文中，避免把前一个匹配的 op 指针留到下一个匹配使用。旧 op 被替换后，其地址可能立即失效，后续通知也只能使用 rewriter 提供的新句柄和值。

### 2. `ensureL1Entry` 让声明生成幂等

call 需要可解析的符号。多个 `gdn.fwd` 共享一个入口时，声明应在模块级按需创建。

```cpp
static func::FuncOp ensureL1Entry(ModuleOp module, PatternRewriter &rewriter,
                                  StringRef sym, FunctionType fnTy) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(sym))
    return existing;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto decl = rewriter.create<func::FuncOp>(
      module.getLoc(), sym, fnTy);
  decl.setPrivate();
  return decl;
}
```

[dialect/lib/GdnPasses.cpp:26-36]

先执行 `lookupSymbol`，同名声明已存在便直接复用，第二个匹配不会产生重复定义。`InsertionGuard` 保存并恢复调用方的插入点，使“到模块首部插入声明”不干扰旧操作附近的 call 创建。`setPrivate()` 给出私有外部声明，函数体由后续 L1 实体提供。这个查询只比较符号名，未核对既有声明的 `fnTy`；同名入口必须维持一致签名，否则后续 call 验证仍会暴露冲突。

### 3. 无结果硬件操作用 `eraseOp` 完成替换

有些下降没有结果可重接线。`setmaxnreg` 先建立等价的 NVVM 操作，再删除旧包装。

```cpp
struct SetmaxnregLowering : OpRewritePattern<GdnSetmaxnregOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GdnSetmaxnregOp op,
                                PatternRewriter &rewriter) const override {
    auto action = op.getDirection() == GdnRegDir::inc
                      ? NVVM::SetMaxRegisterAction::increase
                      : NVVM::SetMaxRegisterAction::decrease;
    NVVM::SetMaxRegisterOp::create(rewriter, op.getLoc(),
                                   (uint32_t)op.getCount(), action);
    rewriter.eraseOp(op);
    return success();
  }
};
```

[dialect/lib/GdnHatch.cpp:29-41]

枚举属性 `inc/dec` 被翻译为 NVVM 的 `increase/decrease`，count 与位置随新操作保留。`eraseOp` 在这里属于已经建立替代语义后的显式下降步骤，不能类推出“无结果操作都可删除”。`setmaxnreg` 会改变 warp 寄存器预算且未标 `Pure`；简单死代码消除不能因为它没有 results 就把寄存器效果抹去。对有结果操作，`replaceOpWithNewOp` 可同时完成构建和替换，hatch 文件中的 `ElectLowering` 即采用这一形式 [dialect/lib/GdnHatch.cpp:43-50]。

### 4. 贪心驱动只处理 worklist 中的局部机会

两个项目 pass 都把模式集交给同一个上游驱动器。

```cpp
void populateGdnToChunkPatterns(RewritePatternSet &patterns) {
  patterns.add<FwdToLowering, BwdToLowering>(patterns.getContext());
}

struct GdnToChunkPass
    : public PassWrapper<GdnToChunkPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GdnToChunkPass)
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGdnToChunkPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
```

[dialect/lib/GdnPasses.cpp:89-105；pass 元数据省略]

`applyPatternsGreedily` 在模块各区域内维护操作 worklist，按模式收益尝试候选规则。成功改写后，新操作以及受影响的值生产者和使用者会重新进入处理过程；队列为空且没有可匹配模式时到达不动点。`gdn-lower-hatch` 在第 215-230 行以同样方式注册十二个模式并调用驱动器 [dialect/lib/GdnHatch.cpp:215-230]。

对 `ModuleOp` 调用的是操作重载：它处理模块的各个区域，模块中的 `func.func` 及其嵌套区域会被覆盖，但不会把模块自身当作一个可删除的根节点。默认 scope 限制可以修改的范围，严格模式还能限制哪些已有操作重新入队。理解范围很重要，在嵌套 pass 中以为整个模块都会被扫描，可能造成模式始终不触发。

模式收益影响优先尝试顺序，worklist 通知决定改写后还能看到哪些新机会。局部成功不要求模式集与顺序无关；若需要确定性产物，应避免依赖指针地址或未固定的遍历顺序来决定输出。对于本项目的两个下降，目标操作每成功一次就消失，进展条件相对直接，复杂规范化模式则需要更仔细地排除循环。

默认驱动还执行操作折叠、简单死代码消除和常量去重。当前 MLIR 可在迭代之间启用完整公共子表达式消除，项目这两个调用没有打开该配置。驱动返回失败可表示达到迭代或改写预算后仍未收敛，`signalPassFailure()` 把问题上升为 pass 失败。

收敛只说明给定模式集在当前配置下没有继续命中的局部机会，不提供全局最优保证。模式 A 创建的模式 B 输入会促进收敛；A 与 B 若互相重建对方，或某模式在不改变 IR 时仍返回成功，就可能反复触发。调试时应检查每次成功是否减少目标操作、缩短表达式或建立严格可测的进展，并用收敛测试暴露循环。

### 5. fold、CSE、DCE 与 canonicalization 的交界

【通用补充】同一个输入分别执行三条命令，避免把组合结果误记成某一个 pass 的全部作用。输入包含常量算式、两次相同加法和一条无使用的乘法：

```mlir
func.func @simplify(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %k = arith.addi %c2, %c4 : i32
  %a = arith.addi %x, %k : i32
  %b = arith.addi %x, %k : i32
  %dead = arith.muli %a, %a : i32
  %r = arith.addi %a, %b : i32
  return %r : i32
}
```

[paper/zh/v6examples/simplify.mlir:1-10]

`-canonicalize` 的真实输出把 `2+4` 折成 6，删除了 `%dead`，但仍保留两个相同加法：

```mlir
%c6_i32 = arith.constant 6 : i32
%0 = arith.addi %arg0, %c6_i32 : i32
%1 = arith.addi %arg0, %c6_i32 : i32
%2 = arith.addi %0, %1 : i32
return %2 : i32
```

[paper/zh/v6examples/simplify.canonicalize.mlir:2-6；函数包装省略]

`-cse` 的真实输出合并了相同加法，却没有把 `2+4` 折成常量；该实现同时清理了因此不再需要的操作：

```mlir
%c2_i32 = arith.constant 2 : i32
%c4_i32 = arith.constant 4 : i32
%0 = arith.addi %c2_i32, %c4_i32 : i32
%1 = arith.addi %arg0, %0 : i32
%2 = arith.addi %1, %1 : i32
return %2 : i32
```

[paper/zh/v6examples/simplify.cse.mlir:2-7；函数包装省略]

`-trivial-dce` 的真实输出只删除无使用的 `%dead`，两个相同加法和两个常量都保留：

```mlir
%c2_i32 = arith.constant 2 : i32
%c4_i32 = arith.constant 4 : i32
%0 = arith.addi %c2_i32, %c4_i32 : i32
%1 = arith.addi %arg0, %0 : i32
%2 = arith.addi %arg0, %0 : i32
%3 = arith.addi %1, %2 : i32
return %3 : i32
```

[paper/zh/v6examples/simplify.dce.mlir:2-8；函数包装省略]

折叠由操作自己的钩子识别常量或恒等情形；CSE 在支配与效果条件允许时合并等价计算；DCE 删除无使用且无副作用的操作。canonicalization pass 收集方言注册的规范化模式，并复用贪心驱动、折叠和局部清理，所以它不是“只做折叠”的别名。当前 gdn 方言没有自写 fold、canonicalization 或 DCE 钩子，项目两个贪心 pass 也未打开完整 CSE 配置；本例全部依赖上游 arith 与通用 pass。

### 6. 普通 rewrite 与 DialectConversion 的边界

`FwdToLowering` 和 hatch 模式直接创建已知目标操作，源结果类型大多沿用，目标集合也不需要逐步声明合法与非法。普通 `OpRewritePattern` 因而足够表达这些一对一或一对多的局部替换。

当下降需要类型转换、目标合法性判定、源与目标之间的 materialization，以及部分或整体转换的提交管理时，应使用 DialectConversion 的 `ConversionPattern`、`TypeConverter` 与 `ConversionTarget`。它处理的是另一组基础设施，不能把 `replaceOp` 的局部成功当作整个类型系统已经转换完成。具体转换合同留到第十二章。
