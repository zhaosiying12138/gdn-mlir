// G2-C: gdn hatch ops -> in-tree NVGPU/NVVM ops (1:1 rewrites). Pass:
// gdn-lower-hatch (registered in gdn-opt).
// G5-G round 3: ONE real escape hatch added -- gdn.mma.tf32k8 lowers to
// nvvm.inline_ptx (tf32 m16n8k8 mma.sync is not tabled in-tree).  Hatch
// budget: 1 of <=5 used.
#define GET_OP_CLASSES
#define GET_ATTRDEF_CLASSES
#include "Gdn/Gdn.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace gdn {

namespace {

static Value czero(PatternRewriter &rewriter, Location loc) {
  return arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
}

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

struct ElectLowering : OpRewritePattern<GdnElectOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GdnElectOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<NVVM::ElectSyncOp>(op, rewriter.getI1Type(),
                                                   Value());
    return success();
  }
};

struct MbarrierInitLowering : OpRewritePattern<GdnMbarrierInitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GdnMbarrierInitOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<nvgpu::MBarrierInitOp>(
        op, op.getBarrier(), op.getCount(), czero(rewriter, op.getLoc()),
        Value());
    return success();
  }
};

struct MbarrierArriveLowering : OpRewritePattern<GdnMbarrierArriveOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GdnMbarrierArriveOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<nvgpu::MBarrierArriveOp>(
        op, op.getToken().getType(), op.getBarrier(),
        czero(rewriter, op.getLoc()));
    return success();
  }
};

struct MbarrierArriveExpectTxLowering
    : OpRewritePattern<GdnMbarrierArriveExpectTxOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult
  matchAndRewrite(GdnMbarrierArriveExpectTxOp op,
                  PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<nvgpu::MBarrierArriveExpectTxOp>(
        op, op.getBarrier(), op.getTxcount(), czero(rewriter, op.getLoc()),
        Value());
    return success();
  }
};

struct MbarrierTryWaitParityLowering
    : OpRewritePattern<GdnMbarrierTryWaitParityOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult
  matchAndRewrite(GdnMbarrierTryWaitParityOp op,
                  PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<nvgpu::MBarrierTryWaitParityOp>(
        op, op.getBarrier(), op.getParity(), op.getTicks(),
        czero(rewriter, op.getLoc()));
    return success();
  }
};

struct TmaAsyncLoadLowering : OpRewritePattern<GdnTmaAsyncLoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GdnTmaAsyncLoadOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<nvgpu::TmaAsyncLoadOp>(
        op, op.getDst(), op.getBarrier(), op.getDescriptor(),
        op.getCoordinates(), czero(rewriter, op.getLoc()), Value(), Value());
    return success();
  }
};

struct TmaAsyncStoreLowering : OpRewritePattern<GdnTmaAsyncStoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GdnTmaAsyncStoreOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<nvgpu::TmaAsyncStoreOp>(
        op, op.getSrc(), op.getDescriptor(), op.getCoordinates(), Value());
    return success();
  }
};

struct TmaPrefetchLowering : OpRewritePattern<GdnTmaPrefetchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GdnTmaPrefetchOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<nvgpu::TmaPrefetchOp>(op, op.getDescriptor(),
                                                      Value());
    return success();
  }
};

struct BulkCommitLowering : OpRewritePattern<GdnBulkCommitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GdnBulkCommitOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<NVVM::CpAsyncBulkCommitGroupOp>(op);
    return success();
  }
};

struct BulkWaitLowering : OpRewritePattern<GdnBulkWaitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult
  matchAndRewrite(GdnBulkWaitOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<NVVM::CpAsyncBulkWaitGroupOp>(
        op, rewriter.getI32IntegerAttr(op.getGroup()),
        op.getRead() ? UnitAttr::get(op.getContext()) : UnitAttr());
    return success();
  }
};

// ESCAPE HATCH #1 (G5-G round 3): tf32 m16n8k8 mma.sync -- not reachable
// through nvgpu.mma.sync (tf32 caps at K=4 there) or the tabled
// nvvm.mma.sync family (m16n8k8 only f16/bf16/i8).  Emitted as a direct
// llvm.inline_asm with EARLY-CLOBBERED outputs.
//
// Why not nvvm.inline_ptx (the BasicPtxBuilder path): its output
// constraints are plain "=f".  SASS-verified on the RTX 5090 Laptop:
// with "=f", ptxas' inline-asm register allocation is free to map the
// D quad ONTO THE A QUAD OF THE SAME mma (HMMA R12, R12, R10, R16) --
// inputs are considered dead after the asm, no early-clobber, and the
// tensor core's read/write overlap then corrupts the operands (exact
// integer probes come back short by whole k-terms; a K=16 two-atom
// chain failed 118/128 cells).  "=&f" forbids output/input overlap;
// D==C aliasing (fla-style in-place accumulation) remains legal but we
// keep D fully distinct.
//
// Operand order: outputs first -> $0..$3 = D, $4..$7 = A, $8/$9 = B,
// $10..$13 = C.  Result is a literal struct<(f32 x4)>; 4 extractvalue.
struct MmaTf32K8Lowering : OpRewritePattern<GdnMmaTf32K8Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult
  matchAndRewrite(GdnMmaTf32K8Op op, PatternRewriter &rewriter) const override {
    static const char *kPtx =
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
        "{$10, $11, $12, $13};";
    static const char *kConstraints =
        "=&f,=&f,=&f,=&f,f,f,f,f,f,f,f,f,f,f";
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Type f32 = rewriter.getF32Type();
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
    Value d0 = LLVM::ExtractValueOp::create(rewriter, loc, asmOp->getResult(0),
                                            ArrayRef<int64_t>(0));
    Value d1 = LLVM::ExtractValueOp::create(rewriter, loc, asmOp->getResult(0),
                                            ArrayRef<int64_t>(1));
    Value d2 = LLVM::ExtractValueOp::create(rewriter, loc, asmOp->getResult(0),
                                            ArrayRef<int64_t>(2));
    Value d3 = LLVM::ExtractValueOp::create(rewriter, loc, asmOp->getResult(0),
                                            ArrayRef<int64_t>(3));
    rewriter.replaceOp(op, {d0, d1, d2, d3});
    return success();
  }
};

} // namespace

void populateGdnHatchLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<SetmaxnregLowering, ElectLowering, MbarrierInitLowering,
               MbarrierArriveLowering, MbarrierArriveExpectTxLowering,
               MbarrierTryWaitParityLowering, TmaAsyncLoadLowering,
               TmaAsyncStoreLowering, TmaPrefetchLowering, BulkCommitLowering,
               BulkWaitLowering, MmaTf32K8Lowering>(patterns.getContext());
}

struct GdnLowerHatchPass
    : public PassWrapper<GdnLowerHatchPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GdnLowerHatchPass)
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGdnHatchLoweringPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  StringRef getArgument() const override { return "gdn-lower-hatch"; }
  StringRef getDescription() const override {
    return "Lower gdn warp-spec/TMA hatch ops to in-tree nvgpu/nvvm ops";
  }
};

std::unique_ptr<Pass> createGdnLowerHatchPass() {
  return std::make_unique<GdnLowerHatchPass>();
}

//===----------------------------------------------------------------------===//
// Hatch op verifiers
//===----------------------------------------------------------------------===//

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

LogicalResult GdnTmaAsyncStoreOp::verify() {
  return verifyTmaCoords(*this, getCoordinates());
}

} // namespace gdn
