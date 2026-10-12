// GDN L0→L1 展开通道。
//
// 完整的 chunk-AST 展开（KKT→前代换→w/u→h 递推→o；bwd 同构 FLA 阶段）随 G2
// 旋钮管线落地——届时本 pass 发出的 IR 与 tools/gdn_l1_gen.py 生成的 L1 文本
// 逐行同构（生成器即本 pass 的可执行规格）。
// 当前版本：把 gdn.fwd/gdn.bwd 改写为对 __gdn_l1_* 外部函数的调用（携带全部
// 语义属性），保证方言端到端（parse→verify→rewrite→print）可测试。
#define GET_OP_CLASSES
#define GET_ATTRDEF_CLASSES
#include "Gdn/Gdn.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace gdn {

namespace {

static constexpr const char *kL1FwdSym = "__gdn_l1_fwd_chunk";
static constexpr const char *kL1BwdSym = "__gdn_l1_bwd_chunk";

struct FwdToLowering : OpRewritePattern<gdn::GdnFwdOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(gdn::GdnFwdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> args{op.getQ(), op.getK(), op.getV(), op.getBeta(),
                            op.getG()};
    if (op.getH0())
      args.push_back(op.getH0());
    auto callee = rewriter.getStringAttr(kL1FwdSym);
    auto call = rewriter.create<func::CallOp>(loc, callee,
                                              op.getResultTypes(), args);
    // 语义属性原样透传（L1 消费方按此选择算法变体）
    call->setAttr("gdn.config", op.getConfig());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct BwdToLowering : OpRewritePattern<gdn::GdnBwdOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(gdn::GdnBwdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> args{op.getQ(), op.getK(), op.getV(), op.getBeta(),
                            op.getG()};
    if (op.getH0())
      args.push_back(op.getH0());
    if (op.getA())
      args.push_back(op.getA());
    args.push_back(op.getDout());
    if (op.getDht())
      args.push_back(op.getDht());
    auto call = rewriter.create<func::CallOp>(
        loc, rewriter.getStringAttr(kL1BwdSym), op.getResultTypes(), args);
    call->setAttr("gdn.config", op.getConfig());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

} // namespace

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
  StringRef getArgument() const override { return "gdn-to-chunk"; }
  StringRef getDescription() const override {
    return "Lower gdn.fwd/gdn.bwd (L0) to L1 chunk-AST entry calls";
  }
};

std::unique_ptr<Pass> createGdnToChunkPass() {
  return std::make_unique<GdnToChunkPass>();
}

} // namespace gdn
