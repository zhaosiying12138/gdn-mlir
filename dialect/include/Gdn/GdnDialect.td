#ifndef GDN_DIALECT_TD
#define GDN_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// GDN 方言（L0 语义层）：Gated DeltaNet / gated delta rule 算子的语义入口。
// 语义唯一事实源 = docs/L0_SEMANTICS.md + tools/gdn_ref_jax.py（JAX 可执行规约）。
//===----------------------------------------------------------------------===//

def Gdn_Dialect : Dialect {
  let name = "gdn";
  let summary = "Gated DeltaNet (gated delta rule) operator dialect";
  let description = [{
    L0 semantic entry for chunk_gated_delta_rule forward/backward.
    Progressive lowering: L0 (gdn.fwd/gdn.bwd) -> L1 chunk-AST
    (KKT -> tril-solve -> w/u -> h recurrence -> o) -> L2 tiling/scheduling
    (transform dialect) -> L3 pipelining/mma -> L4 nvvm/sm_120a.
    G2-C hatch ops (gdn.setmaxnreg/elect/mbarrier.*/tma.*/bulk.*) lower to
    in-tree nvgpu/nvvm ops via the gdn-lower-hatch pass.
  }];
  let cppNamespace = "::gdn";
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = [
    "tensor::TensorDialect",
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "func::FuncDialect",
    "mlir::NVVM::NVVMDialect",
    "nvgpu::NVGPUDialect",
    "gpu::GPUDialect",
    "arith::ArithDialect",
  ];
}

// G2-C hatch: setmaxnreg direction（enum 声明随 -gen-enum-decls 于 GdnHatch.td
// 生成 GdnHatchEnums.h.inc；Attr 类随 GdnDialect.td 的 -gen-attrdef-* 生成）
def GdnRegDirInc : I32EnumAttrCase<"inc", 0>;
def GdnRegDirDec : I32EnumAttrCase<"dec", 1>;
def GdnRegDir : I32EnumAttr<"GdnRegDir", "GDN setmaxnreg direction",
  [GdnRegDirInc, GdnRegDirDec]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::gdn";
}
def GdnRegDirAttr : EnumAttr<Gdn_Dialect, GdnRegDir, "reg_dir">;

// 语义/算法旋钮束（L0 即可调；逐项含义见 docs/L0_SEMANTICS.md）
def GdnConfigAttr : AttrDef<Gdn_Dialect, "GdnConfig"> {
  let mnemonic = "config";
  let summary = "GDN algorithm knobs";
  let description = [{
    chunk_size ∈ {16,32,64}; use_g (scalar log2 decay); use_gk (per-channel);
    normalize_q; use_initial_state; store_final_state; is_varlen.
  }];
  let parameters = (
    ins
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

#endif // GDN_DIALECT_TD
