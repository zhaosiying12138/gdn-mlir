#ifndef GDN_HATCH_TD
#define GDN_HATCH_TD

// G2-C: warp-specialization escape-hatch ops (hatch = our spelling of PTX
// features the NVGPU/NVVM dialects already table in this Oct-2026 LLVM).
//
// Every op here lowers 1:1 (lib/GdnHatch.cpp, pass `gdn-lower-hatch`) to an
// IN-TREE op, so the custom-NVVM-escape-hatch budget consumed = 0:
//   gdn.setmaxnreg            -> nvvm.setmaxregister (inc/dec)
//   gdn.elect                 -> nvvm.elect.sync
//   gdn.mbarrier.init         -> nvgpu.mbarrier.init
//   gdn.mbarrier.arrive       -> nvgpu.mbarrier.arrive
//   gdn.mbarrier.arrive.expect_tx -> nvgpu.mbarrier.arrive.expect_tx
//   gdn.mbarrier.try_wait.parity  -> nvgpu.mbarrier.try_wait.parity
//   gdn.tma.async.load        -> nvgpu.tma.async.load
//   gdn.tma.async.store       -> nvgpu.tma.async.store
//   gdn.tma.prefetch          -> nvgpu.tma.prefetch.descriptor
//   gdn.bulk.commit_group     -> nvvm.cp.async.bulk.commit.group
//   gdn.bulk.wait_group       -> nvvm.cp.async.bulk.wait_group

include "Gdn/GdnDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class GdnHatchOp<string mnemonic, list<Trait> traits = []>
    : Op<Gdn_Dialect, mnemonic, traits>;

// GdnRegDir / GdnRegDirAttr live in GdnDialect.td (attrdef backend emits the
// Attr class from the dialect td; enum decls are emitted from this file).

def GdnSetmaxnregOp : GdnHatchOp<"setmaxnreg"> {
  let summary = "warp register redistribution (PTX setmaxnreg.{inc,dec}.sync.aligned.u32)";
  let description = [{
    Warp-specialization hatch: raises (inc) or lowers (dec) the per-warp
    register budget. Producer warps dec to hand registers to consumer warps
    (which inc). Count must be a multiple of 8 in [24, 256]. Lowers to the
    in-tree `nvvm.setmaxregister` (LLVM intrinsics
    nvvm_setmaxnreg_{inc,dec}_sync_aligned_u32).
  }];
  let arguments = (ins I32Attr:$count, GdnRegDirAttr:$direction);
  let assemblyFormat = "enum($direction) $count attr-dict";
  let hasVerifier = 1;
}

def GdnElectOp : GdnHatchOp<"elect"> {
  let summary = "warp-leader election (PTX elect.sync)";
  let description = [{
    Elects one predicated active leader thread of the warp; result is true on
    the leader. Used to single-thread TMA issue / mbarrier arrive.expect_tx.
    Lowers to the in-tree `nvvm.elect.sync` (default membermask 0xFFFFFFFF).
  }];
  let results = (outs I1:$pred);
  let assemblyFormat = "attr-dict `->` type($pred)";
}

def GdnMbarrierInitOp : GdnHatchOp<"mbarrier.init"> {
  let summary = "mbarrier initialization (PTX mbarrier.init.shared.b64)";
  let description = [{
    Initializes the mbarrier object (an `!nvgpu.mbarrier.group<...>` value
    from `nvgpu.mbarrier.create`) with the given arrival count. Lowers to the
    in-tree `nvgpu.mbarrier.init` at barrier id 0.
  }];
  let arguments = (ins AnyType:$barrier, Index:$count);
  let assemblyFormat = "$barrier `,` $count attr-dict `:` type($barrier)";
}

def GdnMbarrierArriveOp : GdnHatchOp<"mbarrier.arrive"> {
  let summary = "mbarrier arrive-on (PTX mbarrier.arrive.shared.b64)";
  let description = [{
    Performs an arrive-on operation on the mbarrier and returns an
    `!nvgpu.mbarrier.token`. Lowers to the in-tree `nvgpu.mbarrier.arrive`.
  }];
  let arguments = (ins AnyType:$barrier);
  let results = (outs AnyType:$token);
  let assemblyFormat = "$barrier attr-dict `:` type($barrier) `->` type($token)";
}

def GdnMbarrierArriveExpectTxOp : GdnHatchOp<"mbarrier.arrive.expect_tx"> {
  let summary = "mbarrier arrive with expected tx-count (PTX mbarrier.arrive.expect_tx.shared.b64)";
  let description = [{
    Arrives on the mbarrier and increases its expected async-transaction
    byte count by $txcount (used with TMA loads completing via
    mbarrier::complete_tx::bytes). Lowers to the in-tree
    `nvgpu.mbarrier.arrive.expect_tx`.
  }];
  let arguments = (ins AnyType:$barrier, Index:$txcount);
  let assemblyFormat = "$barrier `,` $txcount attr-dict `:` type($barrier)";
}

def GdnMbarrierTryWaitParityOp : GdnHatchOp<"mbarrier.try_wait.parity"> {
  let summary = "mbarrier phase-parity wait (PTX mbarrier.try_wait.parity.shared.b64)";
  let description = [{
    Potentially-blocking test of whether the mbarrier completed the phase of
    the given parity (false = even/first phase, true = odd/second phase).
    Lowers to the in-tree `nvgpu.mbarrier.try_wait.parity`.
  }];
  let arguments = (ins AnyType:$barrier, I1:$parity, Index:$ticks);
  let assemblyFormat = "$barrier `,` $parity `,` $ticks attr-dict `:` type($barrier)";
}

def GdnTmaAsyncLoadOp : GdnHatchOp<"tma.async.load"> {
  let summary = "TMA async tensor load (PTX cp.async.bulk.tensor.Nd...mbarrier::complete_tx::bytes)";
  let description = [{
    Loads the tile addressed by $coordinates through the tensor-map
    descriptor (host-side `nvgpu.tma.create.descriptor`; opaque at compile
    level — runtime wiring is downstream's job) into the workgroup-memref
    $dst, completing on the mbarrier. Lowers to the in-tree
    `nvgpu.tma.async.load`.
  }];
  let arguments = (ins AnyType:$dst, AnyType:$descriptor, AnyType:$barrier,
                       Variadic<Index>:$coordinates);
  let assemblyFormat = "$descriptor `[` $coordinates `]` `,` $barrier `to` $dst attr-dict `:` type($descriptor) `,` type($barrier) `->` type($dst)";
  let hasVerifier = 1;
}

def GdnTmaAsyncStoreOp : GdnHatchOp<"tma.async.store"> {
  let summary = "TMA async tensor store (PTX cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group)";
  let description = [{
    Stores the workgroup-memref tile $src to the global tensor through the
    tensor-map descriptor (bulk async-group completion; pair with
    gdn.bulk.commit_group / gdn.bulk.wait_group). Lowers to the in-tree
    `nvgpu.tma.async.store`.
  }];
  let arguments = (ins AnyType:$src, AnyType:$descriptor,
                       Variadic<Index>:$coordinates);
  let assemblyFormat = "$src `to` $descriptor `[` $coordinates `]` attr-dict `:` type($src) `->` type($descriptor)";
  let hasVerifier = 1;
}

def GdnTmaPrefetchOp : GdnHatchOp<"tma.prefetch"> {
  let summary = "tensormap prefetch (PTX prefetch.tensormap)";
  let description = [{
    Prefetches the cache line containing the tensor-map descriptor. Lowers
    to the in-tree `nvgpu.tma.prefetch.descriptor`.
  }];
  let arguments = (ins AnyType:$descriptor);
  let assemblyFormat = "$descriptor attr-dict `:` type($descriptor)";
}

def GdnBulkCommitOp : GdnHatchOp<"bulk.commit_group"> {
  let summary = "commit cp.async.bulk group (PTX cp.async.bulk.commit_group)";
  let description = [{
    Commits all prior uncommitted cp.async.bulk operations into a bulk
    async-group. Lowers to the in-tree `nvvm.cp.async.bulk.commit.group`.
  }];
  let assemblyFormat = "attr-dict";
}

def GdnBulkWaitOp : GdnHatchOp<"bulk.wait_group"> {
  let summary = "wait cp.async.bulk group (PTX cp.async.bulk.wait_group[.read])";
  let description = [{
    Waits until $group or fewer of the most recent bulk async-groups are
    pending; with `read`, waits only until sources have been read. Lowers to
    the in-tree `nvvm.cp.async.bulk.wait_group`.
  }];
  let arguments = (ins I32Attr:$group, OptionalAttr<UnitAttr>:$read);
  let assemblyFormat = "$group (`read` $read^)? attr-dict";
}

def GdnMmaTf32K8Op : GdnHatchOp<"mma.tf32k8", [Pure]> {
  let summary = "tf32 m16n8k8 warp mma (PTX mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32) -- ESCAPE HATCH #1";
  let description = [{
    G5-G round 3, REAL ESCAPE HATCH #1 of <=5: the tf32 m16n8k8 mma.sync
    signature does not exist in-tree (nvgpu.mma.sync enforces the 8x8x128b
    fundamental tile -> tf32 caps at K=4; the nvvm.mma.sync tablegen family
    lists m16n8k8 only for f16/bf16/i8 -- evidence/G2/gen_report.md).  Lowered
    to `nvvm.inline_ptx` emitting exactly:
      mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
        {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};
    (the exact spelling fla harvests into chunk_gated_delta_rule_bwd_kernel_
    dhu / fwd_kkt_solve -- evidence/G0/fla_ptx).

    Scalar f32 in/out (operands are f32 values; the tensor core rounds them
    to tf32 internally, 10-bit mantissa, f32 exponent range).  Fragment
    layout HARDWARE-DERIVED (G5-G probe exp1+exp2, evidence/G5/scratch/
    tf32k8_probe -- exact integer match; note the A register order differs
    from the packed-pair intuition of bf16 k16):
      groupID g = laneid >> 2, threadID_in_group t = laneid mod 4
      A (16x8): a0=(g,2t) a1=(g+8,2t) a2=(g,2t+1) a3=(g+8,2t+1)
      B (8x8, col): b0=(2t,g) b1=(2t+1,g)                        [k, n]
      C/D (16x8): d0=(g,2t) d1=(g,2t+1) d2=(g+8,2t) d3=(g+8,2t+1)
    D-fragment N index = 2*t (the G4 rule, same as the bf16 m16n8k16 D).
  }];
  let arguments = (ins F32:$a0, F32:$a1, F32:$a2, F32:$a3,
                       F32:$b0, F32:$b1,
                       F32:$c0, F32:$c1, F32:$c2, F32:$c3);
  let results = (outs F32:$d0, F32:$d1, F32:$d2, F32:$d3);
  let assemblyFormat = [{
    $a0 `,` $a1 `,` $a2 `,` $a3 `,` $b0 `,` $b1 `,`
    $c0 `,` $c1 `,` $c2 `,` $c3 attr-dict `:` type($a0) `->` `(` type($d0) `,`
    type($d1) `,` type($d2) `,` type($d3) `)`
  }];
}

#endif // GDN_HATCH_TD
