#ifndef GDN_OPS_TD
#define GDN_OPS_TD

include "Gdn/GdnDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class GdnOp<string mnemonic, list<Trait> traits = []>
    : Op<Gdn_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// gdn.fwd
//===----------------------------------------------------------------------===//

def GdnFwdOp : GdnOp<"fwd", [Pure]> {
  let summary = "GDN forward (semantic entry)";
  let description = [{
    q : tensor<B x T x H  x K>   (L2-normalized if config.normalize_q)
    k : tensor<B x T x H  x K>
    v : tensor<B x T x HV x V>
    beta : tensor<B x T x HV>    (write strength)
    g    : tensor<B x T x HV>    (log2-domain decay, <= 0)
    h0   : optional tensor<B x HV x K x V> f32
    -> o : tensor<B x T x HV x V>, ht : tensor<B x HV x K x V> f32
    Scale = K^{-1/2} applied to normalized q. See docs/L0_SEMANTICS.md.
  }];
  let arguments = (ins
    AnyRankedTensor:$q, AnyRankedTensor:$k, AnyRankedTensor:$v,
    AnyRankedTensor:$beta, AnyRankedTensor:$g,
    Optional<AnyRankedTensor>:$h0,
    GdnConfigAttr:$config
  );
  let results = (outs AnyRankedTensor:$o, AnyRankedTensor:$ht);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// gdn.bwd
//===----------------------------------------------------------------------===//

def GdnBwdOp : GdnOp<"bwd", [Pure, AttrSizedOperandSegments]> {
  let summary = "GDN backward (semantic entry)";
  let description = [{
    Gradients of gdn.fwd w.r.t. q,k,v,beta,g,h0 given upstream do (optional dht).
    A : optional saved WY factor (recompute-vs-reuse knob; absent => recompute).
    Outputs dq,dk,dv (io dtype), dbeta,dg (f32), dh0 (f32, iff h0 given).
  }];
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

#endif // GDN_OPS_TD
