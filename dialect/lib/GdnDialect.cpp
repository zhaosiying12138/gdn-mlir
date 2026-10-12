#define GET_OP_CLASSES
#define GET_ATTRDEF_CLASSES
#include "Gdn/Gdn.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

#include "GdnDialect.cpp.inc"
#include "GdnHatchEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "GdnAttrDefs.cpp.inc"
#define GET_OP_CLASSES
#include "GdnOps.cpp.inc"
#define GET_OP_CLASSES
#include "GdnHatchOps.cpp.inc"

void gdn::GdnDialect::initialize() {
  addAttributes<GdnConfigAttr, GdnRegDirAttr>();
  addOperations<GdnFwdOp, GdnBwdOp>();
  // G2-C warp-spec/TMA hatch ops（lowered via gdn-lower-hatch）
  addOperations<gdn::GdnSetmaxnregOp, gdn::GdnElectOp, gdn::GdnMbarrierInitOp,
                gdn::GdnMbarrierArriveOp, gdn::GdnMbarrierArriveExpectTxOp,
                gdn::GdnMbarrierTryWaitParityOp, gdn::GdnTmaAsyncLoadOp,
                gdn::GdnTmaAsyncStoreOp, gdn::GdnTmaPrefetchOp,
                gdn::GdnBulkCommitOp, gdn::GdnBulkWaitOp, gdn::GdnMmaTf32K8Op>();
}

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

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

LogicalResult gdn::GdnBwdOp::verify() {
  if (failed(verifyConfig(getConfig(), *this)))
    return failure();
  return success();
}
