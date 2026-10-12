#ifndef GDN_DIALECT_GDN_H
#define GDN_DIALECT_GDN_H

#include <memory>

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

// 生成代码（本方言体量小：伞头直接展开完整声明，免去各 TU 宏纪律）
#include "GdnDialect.h.inc"
#include "GdnHatchEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "GdnAttrDefs.h.inc"
// GdnOps.h.inc / GdnHatchOps.h.inc #undef the GET_* macros; re-arm each time.
#define GET_OP_CLASSES
#include "GdnOps.h.inc"
#define GET_OP_CLASSES
#include "GdnHatchOps.h.inc"

namespace gdn {

// chunk 合法集合（L1 算法结构旋钮）
constexpr int kAllowedChunkSizes[] = {16, 32, 64};

// G2-C hatch lowering（gdn.* -> in-tree nvgpu/nvvm）
void populateGdnHatchLoweringPatterns(mlir::RewritePatternSet &patterns);
std::unique_ptr<mlir::Pass> createGdnLowerHatchPass();

} // namespace gdn

#endif // GDN_DIALECT_GDN_H
