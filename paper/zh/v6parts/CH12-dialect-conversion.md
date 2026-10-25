## 十二、渐进式下降与 Dialect Conversion：合法性、类型和过渡 Cast

方言转换（Dialect Conversion）把“改写哪些 op”扩展为三份合同：目标终态允许什么、源类型怎样变成目标类型、每个非法 op 用哪些模式替换。渐进式下降可以暂时保留高层与低层 op 的混合物，但每个转换点都要维持 SSA 使用双方的类型一致，最终还要满足目标方言的合法性要求。

### 1. 合法性、模式和类型是三份独立合同

转换目标（ConversionTarget）描述合法性。`addLegalDialect` 或 `addLegalOp` 声明可以保留的终态 op，`addIllegalOp` 或 `addIllegalDialect` 声明必须消除的 op，`addDynamicallyLegalOp` 则按具体实例的类型、属性或操作数判断。例如，同一个容器 op 可以只在结果类型已转换后合法。具体操作上的设置优先于整个方言的设置；未显式分类的 unknown op 还可另设动态合法规则。

转换模式（ConversionPattern）负责消除非法 op。`OpConversionPattern<T>` 的 `matchAndRewrite` 多一个 adaptor：被匹配 op 仍描述原操作数，adaptor 提供当前已重映射、符合 TypeConverter 预期的操作数。类型发生变化时使用 adaptor 能避免把新类型值直接塞回旧 op。模式可以先生成仍不合法的中间 op，再由其他模式接续转换；驱动器按目标合法性决定何时完成。

类型转换器（TypeConverter）只回答类型映射，包括一对一和一对多，不在 `addConversion` 回调里生成 IR。函数与区域签名还要相应转换块参数。若替换值与尚未转换的使用方类型不同，框架会请求物化（materialization）：source materialization 把新替换值转回旧 source 类型，target materialization 把尚未转换的值转成模式期待的 target 类型。物化回调可以生成真实转换 op，也可以在过渡阶段生成 cast。

`applyPartialConversion` 允许原本合法或允许保留的 unknown op 留下，但显式非法 op 仍须成功处理；`applyFullConversion` 要求整个转换根内的 op 都满足给定目标。full 的严格程度取决于目标定义，若目标把其他方言也声明为合法，成功并不表示只剩一种方言。

这三份合同可以分别定位故障。报出某个 op 仍非法，首先检查目标分类及是否存在能匹配该实例的模式；模式匹配却在 adaptor 处得到意外类型，检查 TypeConverter 与物化规则；转换在仍有旧类型用户时失败，检查 source materialization 是否能构造所需值。把所有 op 都标成合法可以掩盖第一类问题，却不能完成预期下降，因此合法目标本身也需要用输出 IR 审计。

区域签名是另一处常见边界。只替换 op 结果而未转换函数参数、返回类型或基本块参数，使用链仍会在边界处断开。反过来，只改签名而不改写调用者，也会留下类型不一致。渐进式下降允许这些边界暂时由桥连接，但每一座桥都应有明确的后续消除路径。

### 2. 项目调用的上游转换

L1 脚本末段把数学、算术、内存、控制流和函数分别交给上游 pass，最后清理过渡 cast：

```bash
  -convert-math-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-cf-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
```

[scripts/run_l1.sh:29-34]

读这些名字时要把调用者与实现者分开。脚本在项目中，ConversionTarget、TypeConverter 和 ConversionPattern 的实现由对应上游 pass 封装；项目没有为这条 L1 命令另写一套合法性框架。各 pass 的先后安排让上一阶段的输出成为下一阶段的合法输入，最后的 reconcile 只有在生产者与消费者转换都完成后才能看清哪些 cast 已成无操作链。

GPU 侧的 L2 还包含调度解释与 NVVM pipeline；它同样由上游转换 pass 组成，但与 transform 工件的关系在第十三、十九章展开。这里先固定转换框架的边界：IR 合法化属于 Dialect Conversion，PTX 文本提取和 ptxas 不属于它。

### 3. 自有 pass 使用贪心改写

`gdn-to-chunk` 的前向模式直接继承 `OpRewritePattern`，pass 主体调用 `applyPatternsGreedily`：

```cpp
struct FwdToLowering : OpRewritePattern<gdn::GdnFwdOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(gdn::GdnFwdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> args{op.getQ(), op.getK(), op.getV(),
                            op.getBeta(), op.getG()};
    if (op.getH0()) args.push_back(op.getH0());
    auto fnTy = rewriter.getFunctionType(ValueRange(args).getTypes(),
                                         op.getResultTypes());
    if (auto module = op->getParentOfType<ModuleOp>())
      ensureL1Entry(module, rewriter, kL1FwdSym, fnTy);
    auto callee = rewriter.getStringAttr(kL1FwdSym);
    auto call = rewriter.create<func::CallOp>(
        loc, callee, op.getResultTypes(), args);
    call->setAttr("gdn.config", op.getConfig());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};
// 省略 BwdToLowering 与声明辅助函数
if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
  signalPassFailure();
```

[dialect/lib/GdnPasses.cpp:38-59,93-100；声明去重见 26-35 行，反向模式见 61-85 行]

这里能判断当前实现只把 `gdn.fwd` 替换为 `func.call @__gdn_l1_fwd_chunk`，准备参数与结果类型，透传 `gdn.config`；`ensureL1Entry` 先查已有符号，避免重复声明。它没有展开 KKT、前代换和状态递推，也没有 TypeConverter 或 ConversionTarget。完整 L1 文本由 `tools/gdn_l1_gen.py` 生成。贪心驱动也没有“源方言必须全部消失”的全局目标，未匹配 op 可以保留，成功结果不能替代合法性审计。

`gdn-lower-hatch` 使用相同的驱动方式，单个 Elect 模式是一对一替换：

```cpp
struct ElectLowering : OpRewritePattern<GdnElectOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GdnElectOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<NVVM::ElectSyncOp>(
        op, rewriter.getI1Type(), Value());
    return success();
  }
};
// populateGdnHatchLoweringPatterns 注册 12 个模式
if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
  signalPassFailure();
```

[dialect/lib/GdnHatch.cpp:43-50,215-230]

读继承关系和 pass 主体即可判断：这里同样没有自定义 DialectConversion、动态合法目标或 TypeConverter。对该例而言，输入输出类型简单，普通 rewriter 已足够；需要协调多个 op 的签名、允许混合终态或处理一对多类型时，转换框架提供的合同才有直接用途。上游转换能力仍可由脚本调用，这与项目自有 C++ 是否实现该框架是两件事。

### 4. 过渡 cast 与 pass 排序

类型转换经过多个 pass 时，生产端可能已变成新类型，消费端仍期待旧类型。`builtin.unrealized_conversion_cast` 提供临时的类型桥，使混合 IR 在转换过程中保持可表达；它没有为目标机器定义一套任意类型间的运行时转换语义。需要永久保留的转换应使用有明确语义的 op。

以一个旧值 `x:!old` 为例。生产端转换后得到 `x_new:!new`，未转换的消费者仍需 `!old`，source materialization 要在二者之间提供旧类型值。稍后消费者也开始转换，它的模式期待 `!new`，target materialization 又为尚未替换的其他输入提供目标类型。驱动器可能在此过程中给出不同的过渡 SSA 值，模式不应假定 adaptor 一定就是生产者最初 replaceOp 时提供的那个值；应依赖它保证的目标类型。

这也解释了为何 cast 数量本身不能作为失败指标。两个独立 pass 各自插入桥时，中间出现额外 cast 是正常过渡状态；完成消费者转换后，一些桥会折叠，一些则指出仍未接通的类型边界。需要检查的是残留 cast 的两端类型、生产者、消费者以及后续负责它们的 pass。

`reconcile-unrealized-casts` 清理的是沿 cast 链往返后回到相同类型的情况，例如 A→B→C→A 可直接接回原值。它不能证明任意 A→B 与 B→A 在数值上互逆，也不能替缺失的 conversion pattern 补齐语义。因此清理 pass 通常放在相关转换之后；若其后又改变函数签名或 memref 表示，还需重新检查是否产生新桥。动态合法谓词、签名转换与清理顺序应围绕实际残留 IR 设计。

### 5. 【通用补充】用内置类型完成一次转换

下面用 MLIR 自带的 `index` 与 `i64` 演示类型转换，不再引入未定义的 demo 方言。它只是类型与过渡 IR 的教学例，不是 GDN 的实现，也不把 index 的平台相关宽度和 i64 的固定宽度混为语义等价。`ConversionTarget` 允许 `arith.addi` 在结果为 i64 时保留，模式负责把 index 形式改写到这个终态：

```cpp
TypeConverter types;
types.addConversion([](Type type) { return type; });
types.addConversion([](IndexType type) -> Type {
  return IntegerType::get(type.getContext(), 64);
});
auto bridge = [](OpBuilder &b, Type type, ValueRange inputs, Location loc) {
  if (inputs.size() != 1) return Value();
  return UnrealizedConversionCastOp::create(b, loc, type, inputs).getResult(0);
};
types.addSourceMaterialization(bridge);
types.addTargetMaterialization(bridge);

struct AddI64 : OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      arith::AddIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddIOp>(
        op, adaptor.getOperands(), rewriter.getI64Type());
    return success();
  }
};

ConversionTarget target(ctx);
target.addLegalDialect<arith::ArithDialect>();
target.addDynamicallyLegalOp<arith::AddIOp>([](arith::AddIOp op) {
  return op.getResult().getType().isInteger(64);
});
RewritePatternSet patterns(&ctx);
patterns.add<AddI64>(types, &ctx);
if (failed(applyPartialConversion(module, target, std::move(patterns))))
  signalPassFailure();
```

【通用补充，依据上游 Dialect Conversion API；`module`、上下文与 pass 注册略】

`addLegalDialect<arith::ArithDialect>()` 给出默认可保留的算术方言，针对 `arith.addi` 的动态规则再收紧到“结果已经是 i64”。`applyPartialConversion` 允许其他暂时合法的 op 留下，但被规则覆盖的 index 形式必须改写。`adaptor.getOperands()` 提供已按目标类型准备的输入，避免把旧 index 操作数直接塞回新 op。

过渡效果可以用一个 `index` 返回值的消费者观察：

```mlir
// 转换前：addi 的结果仍被 index 的 return 使用
%sum = arith.addi %a, %b : index
return %sum : index

// 概念上的过渡形态：两个输入先物化到 i64，输出再桥回 index
%a64 = unrealized_conversion_cast %a : index to i64
%b64 = unrealized_conversion_cast %b : index to i64
%sum64 = arith.addi %a64, %b64 : i64
%sum = unrealized_conversion_cast %sum64 : i64 to index
return %sum : index
```

source materialization 负责给仍需旧类型的消费者提供 index 值；target materialization 负责给新模式提供 i64 输入。若后面还有真实的 index→i64 数值转换 op，应把桥换成那个有语义的 op，而不是把 `unrealized_conversion_cast` 当成运行时指令。完成消费者转换后，`reconcile-unrealized-casts` 才能清理可消除的往返桥；`applyFullConversion` 则要求根内所有剩余 op 都满足目标，不能只把 addi 一个模式跑完就宣布完成。

### 6. L0–L4 设计与真实调用边界

L0 的 `gdn.fwd/bwd` 表达 tensor 语义，L1 的生成器给出可执行 chunk 算法，L2 由 payload、transform 工件和配置承载调度，L3 使用机器相关 op，L4 落到 sm_120a cubin。这个分层说明各类决策的归属与验证位置。

真实连通范围应按代码陈述：`gdn-to-chunk` 当前完成 L0 入口调用改写、config 透传和声明去重；`tools/gdn_l1_gen.py` 独立生成完整 memref 级 L1；`run_l2.sh` 消费生成好的 payload/transform/config，按需先运行 `gdn-lower-hatch`，再解释调度并调用上游 NVVM pipeline，最后由 ptxas 汇编。L0–L4 是设计与验证分层，现有代码没有提供一条从任意 `gdn.fwd` 自动展开全部算法、选择调度并产出 cubin 的统一转换链。
