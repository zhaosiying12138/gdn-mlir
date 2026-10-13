## 十五、写第一个重写 pass

方言声明了 op 的形态，编译器的本体却是改写：把 IR 从一层的形态变换为下一层的形态，每次变换封装为一个 pass。MLIR 把单个 op 的局部改写抽象为 OpRewritePattern（op 重写模式），把模式的批量执行抽象为 pass 与贪心驱动器。本章以 gdn-to-chunk——L0 语义 op 到 L1 入口调用的改写——为教材。

### 1. 直觉：一个模式只管一种 op

贪心改写驱动器的工作方式是：给定一组模式，对 IR 维护一个待处理 op 的工作队列，逐个出队、按模式类型索引找到候选模式依次尝试，命中即改写；改写产生的新 op 与受影响的使用方重新入队，直至队列耗尽、IR 不再变化——驱动到不动点为止。因此模式的粒度是"一种 source op 到任意 target 形态"的局部变换，模式之间彼此独立、可自由组合，正确的模式集合在任意初始 IR 上收敛到同一下降形态。gdn-to-chunk 需要两个模式：gdn.fwd 到前向 L1 入口调用、gdn.bwd 到反向 L1 入口调用，两个模式装入同一个集合，一个 pass 全部生效。

### 2. matchAndRewrite：签名与契约

```
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
      ensureL1Entry(module, rewriter, kL1FwdSym, fnTy);   // 按需生成声明
    auto callee = rewriter.getStringAttr(kL1FwdSym);
    auto call = rewriter.create<func::CallOp>(loc, callee,
                                              op.getResultTypes(), args);
    // 语义属性原样透传（L1 消费方按此选择算法变体）
    call->setAttr("gdn.config", op.getConfig());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};
```
［dialect/lib/GdnPasses.cpp:38-59；kL1FwdSym 常量在 23 行，BwdToLowering 同构于 61-85 行］

签名本身携带三条契约。第一，模板参数 gdn::GdnFwdOp 限定本模式只匹配这种 op，驱动器据此建索引，无需为其它 op 浪费一次虚调用。第二，matchAndRewrite 把"匹配"与"改写"合并为一个函数：进入函数即视为匹配成功，函数内部直接完成改写并返回 success()；判断不适用时返回 failure()，IR 保持原状，驱动器转试其它模式。第三，成员函数带 const 修饰，模式必须无状态——所有上下文通过参数 op 与 rewriter 传递，这是模式可被多线程驱动器并发复用的前提。函数体的读法顺下来就是一张翻译表：组装调用参数（五个必备操作数依序放入，可选的 h0 按有无追加），从参数与结果类型推导函数签名，确保被调符号存在，创建 call，透传属性，最后整体替换。第十三章讲过的 Optional 访问器（if (op.getH0())）在这里兑现为参数表的条件延长——AttrSizedOperandSegments 消解的槽歧义，在改写侧表现为按槽有无组装实参。

### 3. PatternRewriter 的使用规则：create、replaceOp、eraseOp

对 IR 的一切修改必须经过 rewriter 完成，这是 PatternRewriter 存在的理由：它把每次修改通知给驱动器与观察者，保证改写过程中的中间状态可被跟踪、可被取消。三个动词覆盖全部用法。create 在当前插入点生成新 op——上文的 rewriter.create<func::CallOp>(loc, callee, ...) 以位置、被调者、结果类型与实参四元组构建调用。replaceOp 接受旧 op 与一组新值，原子地完成"旧 op 的每个结果替换为新值、使用方重接线、旧 op 删除"——上文的 rewriter.replaceOp(op, call.getResults()) 即把 gdn.fwd 的两个结果整体移交给 call 的两个结果。eraseOp 单纯删除 op，用于无结果的直通 op 下降：

```
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
［dialect/lib/GdnHatch.cpp:29-41；tma.async.load 的下降在同文件 101-110 行——补一个零偏移谓词后直接替换为树内 nvgpu::TmaAsyncLoadOp，无 PTX 内联］

这段展品同时展示了枚举属性的翻译：GdnRegDir 的 inc/dec 映射到树内 NVVM 的 increase/decrease。此外还有一步到位的组合动词 replaceOpWithNewOp——先建新 op 再替换旧 op，hatch 下降大量使用（如 ElectLowering 以 rewriter.replaceOpWithNewOp<NVVM::ElectSyncOp> 完成无参翻译，GdnHatch.cpp:43-51）。使用纪律有三条。其一，旧 op 在 replaceOp 调用之前保持有效，中途不得直接 delete；新 op 一律经 rewriter（或树内 op 的静态 create 重载）生成，绕过 rewriter 的构建会逃出改写跟踪，使驱动器的增量状态失效。其二，位置信息要传递：新 op 应携带旧 op 的 Location（两个展品里的 op.getLoc() 即此用途），改写后 IR 一旦在某层验证报错，诊断仍能指回最初的源位置，而不是指向凭空生成的无名 call。其三，改写保持类型不动：FwdToLowering 里 call 的结果类型直接取 op.getResultTypes()，下降层没有重述或改写任何类型的权力——类型变化属于后续显式 pass 的职责，混入重写模式会让改写前后的 IR 难以对账。

### 4. ensureL1Entry：为什么在 ModuleOp 上按需声明

call 引用符号 __gdn_l1_fwd_chunk，符号必须在可达的符号表里有声明，否则验证报未定义被调者。声明的生成遵循按需原则：

```
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
［dialect/lib/GdnPasses.cpp:26-36］

四个设计点。第一，先查后建：lookupSymbol 在模块符号表上按名查询，命中即返回——模式对同一个模块里的每个 gdn.fwd 都会触发，若无查询，第二个 op 改写时会生成重名函数声明，验证直接拒绝。第二，挂在 ModuleOp 上：call 的符号解析沿最近的包围符号表向上走，模块是放置跨函数声明的正确层级；每模块至多一份声明，幂等由此成立。第三，InsertionGuard 是异常安全的插入点管理：构造时保存当前位置、析构时恢复，函数体内的 setInsertionPointToStart 只影响本次插入，不污染调用方模式的插入状态。第四，setPrivate 把声明标为私有外部——无函数体，定义由 L1 侧实体在后续阶段提供，恰与"下降通道薄边界"的定位一致：本 pass 只建立调用边界，展开留给下层。函数类型 fnTy 每次按实参现推：h0 有无改变签名元数，前向与反向各自持有独立符号（kL1FwdSym 与 kL1BwdSym，GdnPasses.cpp:23-24）。

### 5. 改写前后的真实 IR

输入是第十三章与第十四章用过的冒烟文件（dialect/test/smoke_l0.mlir:9-13 的 generic 形式 gdn.fwd，携带 #gdn.config<64, true, false, true, true, true, false>）。对它实际运行 gdn-opt --gdn-to-chunk，真实输出：

```
module {
  func.func private @__gdn_l1_fwd_chunk(tensor<2x128x2x16xf32>, ..., tensor<2x2x16x16xf32>)
      -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
  func.func @smoke(%arg0: ..., %arg5: tensor<2x2x16x16xf32>) -> (...) {
    %0:2 = call @__gdn_l1_fwd_chunk(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
        {gdn.config = #gdn.config<64, true, false, true, true, true, false>}
        : (...) -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
    return %0#0, %0#1 : ...
  }
}
```
［build-dialect/bin/gdn-opt --gdn-to-chunk 的真实 stdout，类型串有删节标记］

对照可见三件事全部按设计发生：私有声明出现在模块首（ensureL1Entry 的插入点），gdn.fwd 被替换为 call 且结果直接供给 return，config 属性原样落在 call 上。这条输出同时是回归测试的黄金样本——方言端到端（parse 到 verify 到 rewrite 到 print）可测试，语义层因此可以被单独 dump 与对照。

### 6. pass 的注册：纯 C++ 路线

读 dialect/lib/GdnPasses.cpp 完整文件核实，本仓的 pass 注册走纯 C++ 路线，未使用 TableGen 的 PASS_DEF 生成：

```
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
  StringRef getDescription() const override {
    return "Lower gdn.fwd/gdn.bwd (L0) to L1 chunk-AST entry calls";
  }
};

std::unique_ptr<Pass> createGdnToChunkPass() {
  return std::make_unique<GdnToChunkPass>();
}
```
［dialect/lib/GdnPasses.cpp:93-110；populateGdnToChunkPatterns 在 89-91 行］

结构分四层。PassWrapper 混入引用计数与类型标识样板，MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID 为 RTTI 提供唯一类型号。模板参数 `OperationPass<ModuleOp>` 声明本 pass 的作用单位是模块——runOnOperation 每次收到一个 ModuleOp，getOperation() 的返回类型随之确定。runOnOperation 的标准三步是：组装模式集（populate 函数把上文的两个模式加入集合，供 pass 与其它驱动器复用）、applyPatternsGreedily 执行贪心改写、失败则 signalPassFailure 报告 pass 级错误。getArgument 与 getDescription 提供命令行身份：--gdn-to-chunk 旗标与帮助文本由此生成。工厂函数以 unique_ptr<Pass> 返回实例，接入工具的注册入口。这套骨架在仓内第二个 pass 上原样复用：gdn-lower-hatch 的包装类逐行同构（getArgument 返回 gdn-lower-hatch），其 populate 函数一次性装入全部十二个直通下降模式——从 setmaxnreg 到 mma.tf32k8 ［dialect/lib/GdnHatch.cpp:215-240］；hatch 侧的 populate 与工厂声明集中在伞头 Gdn.h:35-36，gdn-to-chunk 的工厂则在 gdn-opt.cpp:11-13 就地声明，供注册入口共用。

```
int main(int argc, char **argv) {
  registerAllPasses();
  registerPass([]() -> std::unique_ptr<Pass> {
    return gdn::createGdnToChunkPass();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return gdn::createGdnLowerHatchPass();
  });

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<gdn::GdnDialect>();

  return asMainReturnCode(MlirOptMain(
      argc, argv, "GDN dialect optimizer driver\n", registry));
}
```
［dialect/tools/gdn-opt/gdn-opt.cpp:15-30］

registerPass 把工厂挂入全局注册表，registerAllPasses 先行装入全部树内 pass——gdn-opt 因此同时拥有树内重武器与自有 pass（工程细节见第十六章）。与 TableGen 路线（写 Passes.td、由 PASS_DEF 生成类骨架）相比，纯 C++ 直写少一层代码生成，代价是手写类型标识等样板；两条路线生成的 pass 在注册后行为无差别，选择属于工程口味，本仓全部自有 pass 走的都是纯 C++ 这一种。
