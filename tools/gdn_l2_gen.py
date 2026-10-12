#!/usr/bin/env python3
"""GDN L2/L3 transform knob library (G2-A).

Generates a knob-configured GDN chunk-stage kernel family as MLIR
(payload.mlir + transform.mlir) for the LLVM/MLIR transform-dialect GPU
pipeline, targeting sm_120a (Blackwell) via offline PTX/SASS compilation.

Stages
------
  "kkt" : C[64,64] = tril-masked (k[64,K] @ k^T) with exp2 decay epilogue
  "o"   : o[64,V]  = (q[64,K] @ h[K,V]) + (A[64,K] @ v[K,V])  (two chained GEMMs)
  "ref" : sm80-style single-atom f16 matmul lowered by the in-tree
          transform.nvgpu.rewrite_matmul_as_mma_sync (pipeline anchor)

Knobs (each verifiably changes the emitted PTX)
-----------------------------------------------
  dtype    : bf16 | f32 | f16(ref only)      (io dtype; mma path accumulates f32)
  prec     : bf16 | tf32 (mma precision for dtype=f32; tf32 -> m16n8k4 atoms;
             m16n8k8.tf32 is NOT reachable through nvgpu.mma.sync - known gap)
  tile     : {M,N,K} block tile (BT/BV/BK)   (loop bounds, smem sizes, atom counts)
  warps    : 2|4|8 (block = 32*warps threads; mma path splits N across warps)
  mma      : True  -> warp-tile nvgpu.mma.sync tensor-core code (bf16 m16n8k16
                      with f32 accum; f32 -> tf32 m16n8k4);
             False -> linalg.matmul staged through shared memory, tiled +
                      vectorized by the transform script (plain vector path).
  pipeline : 0 (sync copies) | 2 | 3 (async copies; the mma path additionally
             multibuffers the smem tiles and software-pipelines the K loops via
             transform.nvgpu.pipeline_shared_memory_copies).

Toolchain notes (validated; see evidence/G2/gen_report.md):
  * transform.nvgpu.rewrite_matmul_as_mma_sync only supports single-warp,
    atom-sized matmuls (16x8x16 f16/f16/f16 and 16x8k4 tf32). The mma=on GDN
    stages therefore emit nvgpu.mma.sync fragments directly in the payload,
    using the same lane-index formulas as RewriteMatmulAsMmaSyncOp
    (mlir/lib/Dialect/NVGPU/TransformOps/NVGPUTransformOps.cpp).
  * The sm_120a gpu-lower-to-nvvm-pipeline does not lower linalg; mma=off
    leaves one linalg.matmul that the transform script tiles+vectorizes;
    the batch runner lowers any survivor with -convert-linalg-to-loops.
  * transform.nvgpu.pipeline_shared_memory_copies requires flat loop bodies:
    all copies and mma fragments are unrolled in the K-loop body.

Usage:
  gdn_l2_gen.py --list-configs
  gdn_l2_gen.py --gen NAME --out DIR [--driver args|noargs]
  gdn_l2_gen.py --json cfg.json --out DIR [--driver args|noargs]

  --driver noargs (G3-A): self-contained timing variant - inputs become
  module-level constant memref.globals (deterministic 0.5-step values), the
  output goes to a public zero-init memref.global @gdn_out, and the emitted
  gpu.launch kernel therefore takes NO runtime arguments. tools/gdn_time.py
  loads the cubin, launches it 0-arg, and reads @gdn_out back through
  cuModuleGetGlobal for the non-zero/checksum sanity check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

M_GLOBAL = 64   # GDN chunk rows per stage
KKT_N = 64      # kkt output width


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def validate(cfg: dict) -> None:
    stage = cfg["stage"]
    if stage not in ("kkt", "o", "ref", "ref_ldmatrix"):
        raise ValueError(f"unknown stage {stage}")
    if stage == "ref":
        if cfg.get("dtype") != "f16":
            raise ValueError("ref stage is f16-only")
        return
    if stage == "ref_ldmatrix":
        if cfg.get("dtype") != "f16":
            raise ValueError("ref_ldmatrix stage is f16-only")
        return
    dt = cfg["dtype"]
    if dt not in ("bf16", "f32"):
        raise ValueError("dtype must be bf16|f32")
    cfg.setdefault("prec", "bf16" if dt == "bf16" else "tf32")
    if cfg["prec"] not in ("bf16", "tf32"):
        raise ValueError("prec must be bf16|tf32")
    if cfg["prec"] == "bf16" and dt == "f32":
        raise ValueError("prec=bf16 requires dtype=bf16")
    warps = cfg["warps"]
    if warps not in (2, 4, 8):
        raise ValueError("warps must be 2|4|8")
    cfg.setdefault("warpspec", "off")
    cfg.setdefault("tma", "off")
    if cfg["warpspec"] not in ("off", "producer_consumer"):
        raise ValueError("warpspec must be off|producer_consumer")
    if cfg["tma"] not in ("off", "on"):
        raise ValueError("tma must be off|on")
    if cfg["warpspec"] == "producer_consumer":
        if stage != "kkt" or not cfg["mma"] or dt != "bf16":
            raise ValueError("warpspec=producer_consumer needs kkt bf16 mma=on")
    if cfg["tma"] == "on" and cfg["warpspec"] != "producer_consumer":
        raise ValueError("tma=on requires warpspec=producer_consumer")
    if cfg["pipeline"] not in (0, 2, 3):
        raise ValueError("pipeline must be 0|2|3")
    tile = cfg["tile"]
    bt, bv, bk = tile["M"], tile["N"], tile["K"]
    dims = cfg["dims"]
    n_glob = KKT_N if stage == "kkt" else dims["V"]
    k_glob = dims["K"]
    if M_GLOBAL % bt or bt < 16:
        raise ValueError("tile.M must divide 64 and be >= 16")
    if n_glob % bv or bv < 8:
        raise ValueError("tile.N must divide stage N and be >= 8")
    if k_glob % bk or bk < 8:
        raise ValueError("tile.K must divide stage K")
    if bv % 8 or bk % 8:
        raise ValueError("tile.N/tile.K multiples of 8 (16B chunks)")
    if cfg["mma"]:
        if bt % 16:
            raise ValueError("mma=on requires tile.M multiple of 16")
        if dt == "bf16" and bk % 16:
            raise ValueError("mma=on bf16 needs tile.K multiple of 16")
        if dt == "f32" and bk % 4:
            raise ValueError("mma=on f32(tf32) needs tile.K multiple of 4")
        if bv % (warps * 8):
            raise ValueError("mma=on requires tile.N multiple of 8*warps")
        if cfg["pipeline"] > 0 and k_glob // bk < cfg["pipeline"]:
            raise ValueError("pipeline depth exceeds K-loop trip count")
    esz = 2 if dt == "bf16" else 4
    smem = (bt * bk + bk * bv) * esz
    if cfg["mma"] and cfg["pipeline"] > 0:
        smem *= cfg["pipeline"]
    if cfg["mma"] is False:
        # G4: G2 operand staging + f32 accumulator (see gen_payload_linalg)
        smem = (bt * k_glob + k_glob * bv) * esz + bt * bv * 4
    if smem > 220 * 1024:
        raise ValueError(f"smem budget exceeded: {smem}B")


# --------------------------------------------------------------------------
# emission helpers
# --------------------------------------------------------------------------

class Buf:
    def __init__(self):
        self.lines = []

    def l(self, s: str = ""):
        self.lines.append(s)

    def text(self):
        return "\n".join(self.lines) + "\n"


def chunk(dt):
    return 8 if dt == "bf16" else 4


def atom_k(dt):
    return 16 if dt == "bf16" else 4


# --------------------------------------------------------------------------
# G3-A: --driver noargs (self-contained timing variant)
# --------------------------------------------------------------------------

# G4: input scale. G3 used +-3.0 (0.5-step) inputs, which drive kkt stage
# outputs to ~1e3; bf16 output quantization alone (ulp ~4 at |C|~1e3) then
# exceeds ANY absolute tolerance without indicating a bug. The fla-derived
# G4 contract grades stage outputs with ABSOLUTE thresholds (5e-3 bf16-io /
# 1e-3 fp32-io) assuming O(1) stage outputs, so the embedded data is scaled
# by 2^-5: outputs stay O(1) (kkt diag ~0.44) where pure bf16 output
# quantization is <= ~1e-3. Values stay multiples of 2^-7 (4-bit mantissa)
# -> exactly representable in bf16/f16/f32/tf32.
NOARGS_SCALE = 0.03125


def noargs_val(i: int, j: int, seed: int = 0) -> float:
    """Deterministic input value (pure function of (i, j, seed)); a multiple
    of 2^-7 so it is exactly representable in f16/bf16/f32/tf32 (checksum
    stable across dt; fp64 reference twins see the exact same values).
    seed != 0 permutes the data deterministically (generator determinism
    check: same seed -> identical payload)."""
    a, b = 31 + seed, 17 + 7 * seed
    return ((i * a + j * b) % 13 - 6) * 0.5 * NOARGS_SCALE


def emit_global_const(b, sym, rows, cols, dt, seed=0):
    """Module-level constant memref.global with a nested dense literal
    (one source row per line). Private: kernel-internal, no host lookup."""
    b.l(f'memref.global "private" constant @{sym} : memref<{rows}x{cols}x{dt}>'
        f" = dense<[")
    for r in range(rows):
        vals = ", ".join(repr(noargs_val(r, c, seed)) for c in range(cols))
        sep = "," if r < rows - 1 else ""
        b.l(f"  [{vals}]{sep}")
    b.l("]>")


def emit_global_out(b, rows, cols, dt):
    """Public mutable zero-init output global. tools/gdn_time.py reads it
    back via cuModuleGetGlobal(@gdn_out) for the non-zero/checksum check.
    (zero init => any non-zero content proves the kernel actually wrote.)"""
    b.l(f"memref.global @gdn_out : memref<{rows}x{cols}x{dt}> = dense<0.0>")


def copy_pair(b, src_ref, src_ty, src_r, src_c, dst_ref, dst_ty, dst_r, dst_c,
              dt, pad, tag):
    ch = chunk(dt)
    b.l(f"    %{tag} = vector.transfer_read {src_ref}[{src_r}, {src_c}], {pad} "
        f"{{in_bounds = [true]}} : {src_ty}, vector<{ch}x{dt}>")
    b.l(f"    vector.transfer_write %{tag}, {dst_ref}[{dst_r}, {dst_c}] "
        f"{{in_bounds = [true]}} : vector<{ch}x{dt}>, {dst_ty}")


def frag_build(b, elems, dt, vshape, tag, src_ref, src_ty):
    """elems: [(row_v, col_v)] in vector linear order (values incl. leading %)."""
    r0, c0 = elems[0]
    b.l(f"    %{tag}_0 = memref.load {src_ref}[{r0}, {c0}] : {src_ty}")
    b.l(f"    %{tag}_b0 = vector.broadcast %{tag}_0 : {dt} to {vshape}")
    dims = [int(x) for x in vshape.replace("vector<", "").split("x")[:-1]]
    n = 1
    for d in dims:
        n *= d
    cur = f"%{tag}_b0"
    for i in range(1, n):
        r, c = elems[i]
        idx, rem = [], i
        for d in reversed(dims):
            idx.append(rem % d)
            rem //= d
        idx = list(reversed(idx))
        b.l(f"    %{tag}_{i} = memref.load {src_ref}[{r}, {c}] : {src_ty}")
        b.l(f"    %{tag}_b{i} = vector.insert %{tag}_{i}, {cur} "
            f"[{', '.join(map(str, idx))}] : {dt} into {vshape}")
        cur = f"%{tag}_b{i}"
    return cur


def a_frag(mi, kk, dt):
    if dt == "bf16":
        kb = kk * 16
        return [(f"%rowA{mi}", f"%ka{kb}_0"), (f"%rowA{mi}", f"%ka{kb}_1"),
                (f"%rowAh{mi}", f"%ka{kb}_0"), (f"%rowAh{mi}", f"%ka{kb}_1"),
                (f"%rowA{mi}", f"%ka{kb}_2"), (f"%rowA{mi}", f"%ka{kb}_3"),
                (f"%rowAh{mi}", f"%ka{kb}_2"), (f"%rowAh{mi}", f"%ka{kb}_3")]
    kb = kk * 4
    return [(f"%rowA{mi}", f"%kaf{kb}"), (f"%rowAh{mi}", f"%kaf{kb}")]


def b_frag(ni, kk, dt, kt):
    if dt == "bf16":
        kb = kk * 16
        if kt:
            return [(f"%ncolg{ni}", f"%kb{kb}_0"), (f"%ncolg{ni}", f"%kb{kb}_1"),
                    (f"%ncolg{ni}", f"%kb{kb}_2"), (f"%ncolg{ni}", f"%kb{kb}_3")]
        return [(f"%kb{kb}_0", f"%ncolg{ni}"), (f"%kb{kb}_1", f"%ncolg{ni}"),
                (f"%kb{kb}_2", f"%ncolg{ni}"), (f"%kb{kb}_3", f"%ncolg{ni}")]
    kb = kk * 4
    return [(f"%ncolg{ni}", f"%kbf{kb}")] if kt else [(f"%kbf{kb}", f"%ncolg{ni}")]


def frag_precompute(b, cfg):
    bt, bv, bk = cfg["tile"]["M"], cfg["tile"]["N"], cfg["tile"]["K"]
    dt, warps = cfg["dtype"], cfg["warps"]
    wn = bv // warps
    for mi in range(bt // 16):
        b.l(f"    %rowA{mi} = arith.addi %g, %c{mi * 16} : index")
        b.l(f"    %rowAh{mi} = arith.addi %g8, %c{mi * 16} : index")
    for ni in range(wn // 8):
        b.l(f"    %ncol{ni} = arith.addi %wnoff, %c{ni * 8} : index")
        # %ncolg is the lane's N index for the B fragment: PTX m16n8k16 B
        # layout holds column groupID (= %g) per lane.
        b.l(f"    %ncolg{ni} = arith.addi %ncol{ni}, %g : index")
        # G4 fix: %ncolt is the lane's N index for the D/accumulator fragment:
        # PTX m16n8 D layout holds columns 2*threadID_in_group (= %t2) and
        # %t2+1 per lane -- NOT groupID. The G2 epilogue stored even elements
        # at %ncolg (the B-fragment index), misindexing half the outputs
        # (kkt mma wrote 1168/2080 tril cells; o mma 2304/4096).
        b.l(f"    %ncolt{ni} = arith.addi %ncol{ni}, %t2 : index")
        b.l(f"    %ncolgp1{ni} = arith.addi %ncol{ni}, %t2p1 : index")
    ak = atom_k(dt)
    for kk in range(bk // ak):
        kb = kk * ak
        if dt == "bf16":
            for sfx, lv in (("_0", "%t2"), ("_1", "%t2p1"),
                            ("_2", "%t2x8"), ("_3", "%t2p1x8")):
                b.l(f"    %ka{kb}{sfx} = arith.addi %c{kb}, {lv} : index")
                b.l(f"    %kb{kb}{sfx} = arith.addi %c{kb}, {lv} : index")
        else:
            b.l(f"    %kaf{kb} = arith.addi %c{kb}, %tig : index")
            b.l(f"    %kbf{kb} = arith.addi %c{kb}, %tig : index")


def mma_atoms(b, cfg, sa, sa_ty, sb, sb_ty, kt, prev_names, gi):
    """One K-iteration of unrolled mma fragments. prev_names: list of value
    names (with %) carrying accumulators; returns new names."""
    bt, bv, bk = cfg["tile"]["M"], cfg["tile"]["N"], cfg["tile"]["K"]
    dt, warps = cfg["dtype"], cfg["warps"]
    wn = bv // warps
    ak = atom_k(dt)
    av = "vector<4x2xbf16>" if dt == "bf16" else "vector<2x1xf32>"
    bv_ = "vector<2x2xbf16>" if dt == "bf16" else "vector<1x1xf32>"
    tf = " tf32Enabled" if dt == "f32" else ""
    out = []
    k = 0
    for mi in range(bt // 16):
        for ni in range(wn // 8):
            prev = prev_names[k]
            for kk in range(bk // ak):
                t = f"g{gi}f{mi}_{ni}_{kk}"
                af = frag_build(b, a_frag(mi, kk, dt), dt, av, f"{t}a", sa, sa_ty)
                bf = frag_build(b, b_frag(ni, kk, dt, kt), dt, bv_, f"{t}b", sb,
                                sb_ty)
                b.l(f"    %{t}r = nvgpu.mma.sync({af}, {bf}, {prev}) "
                    f"mmaShape = [16, 8, {ak}]{tf} : ({av}, {bv_}, "
                    f"vector<2x2xf32>) -> vector<2x2xf32>")
                prev = f"%{t}r"
            out.append(prev)
            k += 1
    return out


def frag_zero(b, cfg, pfx):
    bt, bv = cfg["tile"]["M"], cfg["tile"]["N"]
    wn = bv // cfg["warps"]
    b.l("    %zfv = arith.constant 0.0 : f32")
    names = []
    for mi in range(bt // 16):
        for ni in range(wn // 8):
            n = f"{pfx}{mi}_{ni}"
            b.l(f"    %{n} = vector.broadcast %zfv : f32 to vector<2x2xf32>")
            names.append(f"%{n}")
    return names


def frag_epilogue(b, cfg, accs, out_mem, out_ty, decay):
    bt, bv = cfg["tile"]["M"], cfg["tile"]["N"]
    wn = bv // cfg["warps"]
    dt = cfg["dtype"]
    b.l("    %epz = arith.constant 0.0 : f32")
    if decay:
        b.l("    %gam = arith.constant -6.25e-02 : f32")
    k = 0
    for mi in range(bt // 16):
        for ni in range(wn // 8):
            acc = accs[k]
            for e in range(4):
                # PTX m16n8 accumulator fragment: extract[i][j] holds
                # C[g + 8*i][2*tig + j]; rows g / g+8, cols 2*tig / 2*tig+1.
                rowv = f"%rowA{mi}" if e < 2 else f"%rowAh{mi}"
                colv = f"%ncolt{ni}" if e % 2 == 0 else f"%ncolgp1{ni}"
                t = f"ep{mi}_{ni}_{e}"
                b.l(f"    %{t}r = arith.addi {rowv}, %m0 : index")
                b.l(f"    %{t}c = arith.addi {colv}, %n0base : index")
                b.l(f"    %{t}x = vector.extract {acc}[{e // 2}, {e % 2}] "
                    f": f32 from vector<2x2xf32>")
                if decay:
                    b.l(f"    %{t}d = arith.subi %{t}r, %{t}c : index")
                    b.l(f"    %{t}di = arith.index_cast %{t}d : index to i32")
                    b.l(f"    %{t}df = arith.sitofp %{t}di : i32 to f32")
                    b.l(f"    %{t}dg = arith.mulf %{t}df, %gam : f32")
                    b.l(f"    %{t}e = math.exp2 %{t}dg : f32")
                    b.l(f"    %{t}m = arith.mulf %{t}x, %{t}e : f32")
                    b.l(f"    %{t}q = arith.cmpi sle, %{t}c, %{t}r : index")
                    b.l(f"    %{t}s = arith.select %{t}q, %{t}m, %epz : f32")
                    val = f"%{t}s"
                else:
                    val = f"%{t}x"
                if dt == "f32":
                    b.l(f"    memref.store {val}, {out_mem}[%{t}r, %{t}c] : "
                        f"{out_ty}")
                else:
                    b.l(f"    %{t}t = arith.truncf {val} : f32 to {dt}")
                    b.l(f"    memref.store %{t}t, {out_mem}[%{t}r, %{t}c] : "
                        f"{out_ty}")
            k += 1


def copy_tile(b, cfg, gi, a_glob, a_ty, b_glob, b_ty, sa, sa_ty, sb, sb_ty, kt):
    """Per-iteration global->shared copies of the operand tiles (unrolled)."""
    bt, bv, bk = cfg["tile"]["M"], cfg["tile"]["N"], cfg["tile"]["K"]
    dt = cfg["dtype"]
    ch = chunk(dt)
    pad = f"%pad{dt}"
    for c in range(bk // ch):
        b.l(f"    %ks{gi}_{c} = arith.addi %k{gi}, %c{c * ch} : index")
    for r in range(bt):
        for c in range(bk // ch):
            copy_pair(b, a_glob, a_ty, f"%grow{r}", f"%ks{gi}_{c}",
                      sa, sa_ty, f"%c{r}", f"%c{c * ch}", dt, pad,
                      f"vA{gi}_{r}_{c}")
    if kt:
        for r in range(bv):
            for c in range(bk // ch):
                copy_pair(b, b_glob, b_ty, f"%nrow{r}", f"%ks{gi}_{c}",
                          sb, sb_ty, f"%c{r}", f"%c{c * ch}", dt, pad,
                          f"vB{gi}_{r}_{c}")
    else:
        for kr in range(bk):
            b.l(f"    %kbrow{gi}_{kr} = arith.addi %k{gi}, %c{kr} : index")
            for c in range(bv // ch):
                copy_pair(b, b_glob, b_ty, f"%kbrow{gi}_{kr}", f"%bcol{c}",
                          sb, sb_ty, f"%c{kr}", f"%c{c * ch}", dt, pad,
                          f"vB{gi}_{kr}_{c}")


# --------------------------------------------------------------------------
# payload: mma path
# --------------------------------------------------------------------------

def gen_payload_mma(b, cfg, stage):
    dt = cfg["dtype"]
    bt, bv, bk = cfg["tile"]["M"], cfg["tile"]["N"], cfg["tile"]["K"]
    k_glob = cfg["dims"]["K"]
    n_glob = KKT_N if stage == "kkt" else cfg["dims"]["V"]
    warps = cfg["warps"]
    wn = bv // warps
    kt = stage == "kkt"
    n_gemm = 2 if stage == "o" else 1
    sa_ty = f"memref<{bt}x{bk}x{dt}, #gpu.address_space<workgroup>>"
    sb_ty = (f"memref<{bv}x{bk}x{dt}, #gpu.address_space<workgroup>>" if kt else
             f"memref<{bk}x{bv}x{dt}, #gpu.address_space<workgroup>>")
    a_ty = f"memref<64x{k_glob}x{dt}>"
    b_ty = (f"memref<64x{k_glob}x{dt}>" if kt else
            f"memref<{k_glob}x{n_glob}x{dt}>")
    for gi in range(n_gemm):
        b.l(f"    %sa{gi} = memref.alloc() : {sa_ty}")
        b.l(f"    %sb{gi} = memref.alloc() : {sb_ty}")
    b.l(f"    %pad{dt} = arith.constant 0.0 : {dt}")
    frag_precompute(b, cfg)
    gemms = ([("%kg", "%kg")] if kt else [("%qg", "%hg"), ("%ag", "%vg")])
    prev = frag_zero(b, cfg, "z")
    for gi, (ag, bg) in enumerate(gemms):
        ia = [f"%ia{gi}_{i}" for i in range(len(prev))]
        resn = [f"%lr{gi}_{i}" for i in range(len(prev))]
        b.l(f"    {', '.join(resn)} = scf.for %k{gi} = %c0 to %c{k_glob} "
            f"step %c{bk} iter_args({', '.join(f'{a} = {p}' for a, p in zip(ia, prev))})"
            f" -> ({', '.join(['vector<2x2xf32>'] * len(prev))}) {{")
        copy_tile(b, cfg, gi, ag, a_ty, bg, b_ty, f"%sa{gi}", sa_ty,
                  f"%sb{gi}", sb_ty, kt)
        b.l("      gpu.barrier")
        newv = mma_atoms(b, cfg, f"%sa{gi}", sa_ty, f"%sb{gi}", sb_ty, kt,
                         ia, gi)
        b.l("      gpu.barrier")
        b.l(f"      scf.yield {', '.join(newv)} : "
            f"{', '.join(['vector<2x2xf32>'] * len(newv))}")
        b.l("    }")
        prev = resn
    out = "%cg" if kt else "%og"
    out_ty = f"memref<64x{KKT_N if kt else n_glob}x{dt}>"
    frag_epilogue(b, cfg, prev, out, out_ty, decay=kt)


# --------------------------------------------------------------------------
# payload: warp-specialized producer/consumer path (G2-C)
# --------------------------------------------------------------------------

MBAR_TY = "!nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>>"
TOKEN_TY = "!nvgpu.mbarrier.token"


def desc_ty(shape):
    return (f"!nvgpu.tensormap.descriptor<tensor = memref<{shape}xbf16, 3>, "
            f"swizzle = none, l2promo = none, oob = zero, interleave = none>")


def ws_frag_precompute(b, cfg):
    """frag_precompute with the consumer-warp offset (warps 1..N consume)."""
    bt, bv, bk = cfg["tile"]["M"], cfg["tile"]["N"], cfg["tile"]["K"]
    dt, warps = cfg["dtype"], cfg["warps"]
    wn = bv // warps
    b.l("    %cwm1 = arith.subi %warp, %c1 : index")
    b.l(f"    %wnoff = arith.muli %cwm1, %c{wn} : index")
    for mi in range(bt // 16):
        b.l(f"    %rowA{mi} = arith.addi %g, %c{mi * 16} : index")
        b.l(f"    %rowAh{mi} = arith.addi %g8, %c{mi * 16} : index")
    for ni in range(wn // 8):
        b.l(f"    %ncol{ni} = arith.addi %wnoff, %c{ni * 8} : index")
        b.l(f"    %ncolg{ni} = arith.addi %ncol{ni}, %g : index")
        b.l(f"    %ncolt{ni} = arith.addi %ncol{ni}, %t2 : index")
        b.l(f"    %ncolgp1{ni} = arith.addi %ncol{ni}, %t2p1 : index")
    ak = atom_k(dt)
    for kk in range(bk // ak):
        kb = kk * ak
        if dt == "bf16":
            for sfx, lv in (("_0", "%t2"), ("_1", "%t2p1"),
                            ("_2", "%t2x8"), ("_3", "%t2p1x8")):
                b.l(f"    %ka{kb}{sfx} = arith.addi %c{kb}, {lv} : index")
                b.l(f"    %kb{kb}{sfx} = arith.addi %c{kb}, {lv} : index")
        else:
            b.l(f"    %kaf{kb} = arith.addi %c{kb}, %tig : index")
            b.l(f"    %kbf{kb} = arith.addi %c{kb}, %tig : index")


def gen_payload_ws(b, cfg, stage):
    """Warp-specialized kkt (G2-C, FlashQLA-pole): 1 producer warp + `warps`
    consumer warps in one CTA.

    G4 corrected single-buffer mbarrier protocol (the G2 version deadlocked
    on hardware -- G3 TIMEOUT):
      * the mbarrier is a pure DATA-READY barrier with init count 1: the
        producer's elected leader is the ONLY arriver per K-iteration
        (tma=on: gdn.mbarrier.arrive.expect_tx + the two TMA loads'
        complete_tx credits complete the phase; tma=off: plain arrive after
        cp.async group wait). G2 initialized count = 32*warps and ALSO had
        every consumer thread arrive after its mma: phase completion then
        required arrivals that can only happen after the very wait that
        gates them (chicken-and-egg) -> hang. Consumers now only WAIT.
      * buffer-reuse safety for the single smem buffer is provided by the
        loop-end gpu.barrier (all warps), exactly as before.
      * a gpu.barrier after gdn.mbarrier.init makes the initialized barrier
        visible to every warp before the first arrive/wait.
      * the epilogue runs on CONSUMER warps only (G2 let producer warp 0
        run it with wnoff = (warp-1)*wn < 0 -> negative store columns).

    producer (warp 0): gdn.setmaxnreg dec 96; gdn.elect leader issues
      - tma=off: classic global->shared cp.async copies, one committed
        group + wait, then gdn.mbarrier.arrive
      - tma=on:  gdn.tma.prefetch + gdn.mbarrier.arrive.expect_tx +
        gdn.tma.async.load (4d A tile + 5d B tile). TMA coordinate order is
        the RAW PTX/CUDA order (innermost dim first) because the lowering
        passes coordinates through unreversed while the host-side encode
        helper reverses memref dim order into CUDA dim order; the host
        tensormap spec (emitted to tensormap_spec.json by --driver noargs)
        describes both. The B tile is a k-rows tile [bv x bk] (K contiguous)
        exactly like the cp.async path -- TMA fills smem with the innermost
        (K) dimension contiguous, so a B[k][n]-major tile is not producible
        from row-major k and G2's [bk x bv] B tile / non-transposed b_frag
        reads were unfixable as generated.
    consumers (warps 1..N): gdn.setmaxnreg inc 224; per K-iter
      gdn.mbarrier.try_wait.parity on the loop-carried phase parity, the
      mma atoms (kt=True k-rows B fragments), then nothing (no arrive).
    tma=on epilogue stages C in smem (consumers), then thread 0 stores via
    gdn.tma.async.store + gdn.bulk.commit_group / gdn.bulk.wait_group read.
    """
    dt = cfg["dtype"]
    bt, bv, bk = cfg["tile"]["M"], cfg["tile"]["N"], cfg["tile"]["K"]
    k_glob = cfg["dims"]["K"]
    n_glob = KKT_N
    warps = cfg["warps"]
    wn = bv // warps
    tma = cfg["tma"] == "on"
    esz = 2
    txbytes = (bt * bk + bk * bv) * esz
    init_count = 1
    sa2_ty = f"memref<{bt}x{bk}x{dt}, #gpu.address_space<workgroup>>"
    sb2_ty = f"memref<{bv}x{bk}x{dt}, #gpu.address_space<workgroup>>"
    sc2_ty = f"memref<64x{KKT_N}x{dt}, #gpu.address_space<workgroup>>"

    if tma:
        sa4_ty = f"memref<1x1x{bt}x{bk}x{dt}, #gpu.address_space<workgroup>>"
        sb5_ty = (f"memref<1x1x1x{bv}x{bk}x{dt}, "
                  f"#gpu.address_space<workgroup>>")

    b.l(f"    %sa = memref.alloc() : {sa4_ty if tma else sa2_ty}")
    b.l(f"    %sb = memref.alloc() : {sb5_ty if tma else sb2_ty}")
    if tma:
        b.l(f"    %sa2 = memref.collapse_shape %sa [[0, 1, 2], [3]] : {sa4_ty} into {sa2_ty}")
        b.l(f"    %sb2 = memref.collapse_shape %sb [[0, 1, 2, 3], [4]] : {sb5_ty} into {sb2_ty}")
        b.l(f"    %sc2 = memref.alloc() : {sc2_ty}")
    b.l(f"    %pad{dt} = arith.constant 0.0 : {dt}")
    b.l(f"    %bar = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>>")
    b.l("    %w32 = arith.index_cast %warp : index to i32")
    b.l("    %wid = nvvm.shfl.sync idx %neg1, %w32, %c0i, %c31i : i32 -> i32")
    b.l("    %isprod = arith.cmpi eq, %wid, %c0i : i32")
    b.l("    %iscons = arith.cmpi ne, %wid, %c0i : i32")
    b.l("    %isz = arith.cmpi eq, %tid, %c0 : index")
    ws_frag_precompute(b, cfg)
    b.l(f"    scf.if %isz {{")
    b.l(f"      gdn.mbarrier.init %bar, %c{init_count} : {MBAR_TY}")
    b.l("    }")
    b.l("    gpu.barrier")

    accs = frag_zero(b, cfg, "z")
    b.l("    %p0 = arith.constant false")
    iargs = [f"%ia{i}" for i in range(len(accs))]
    resn = [f"%ir{i}" for i in range(len(accs))]
    tylist = ", ".join(["vector<2x2xf32>"] * len(accs))
    b.l(f"    {', '.join(resn)}, %pn = scf.for %k0 = %c0 to %c{k_glob} "
        f"step %c{bk} iter_args({', '.join(f'{a} = {p}' for a, p in zip(iargs, accs))}"
        f", %parity = %p0) -> ({tylist}, i1) {{")
    b.l(f"      {', '.join(f'%if{i}' for i in range(len(accs)))} = scf.if %isprod -> ({tylist}) {{")
    b.l("        gdn.setmaxnreg dec 96")
    if tma:
        b.l("        %el = gdn.elect -> i1")
        b.l("        scf.if %el {")
        b.l(f"          gdn.tma.prefetch %descA : {desc_ty(f'1x1x{bt}x{bk}')}")
        b.l(f"          gdn.mbarrier.arrive.expect_tx %bar, %c{txbytes} : {MBAR_TY}")
        b.l(f"          gdn.tma.async.load %descA[%k0, %c0, %c0, %c0], %bar to %sa : "
            f"{desc_ty(f'1x1x{bt}x{bk}')}, {MBAR_TY} -> {sa4_ty}")
        b.l(f"          gdn.tma.async.load %descB[%k0, %c0, %c0, %c0, %c0], %bar to %sb : "
            f"{desc_ty(f'1x1x1x{bv}x{bk}')}, {MBAR_TY} -> {sb5_ty}")
        b.l("        }")
    else:
        # classic cp.async pipeline: the whole producer warp issues 16B
        # chunk copies (bypassL1 -> cp.async.cg), commits one group and
        # waits; the elected leader then arrives on the mbarrier.
        ch_ = 8 if dt == "bf16" else 4
        for c in range(bk // ch_):
            b.l(f"        %ks0_{c} = arith.addi %k0, %c{c * ch_} : index")
        toks = []
        for r in range(bt):
            for c in range(bk // ch_):
                b.l(f"        %tA{r}_{c} = nvgpu.device_async_copy "
                    f"%kg[%grow{r}, %ks0_{c}], %sa[%c{r}, %c{c * ch_}], {ch_} "
                    f"bypassL1 : memref<64x{k_glob}x{dt}> to {sa2_ty}")
                toks.append(f"%tA{r}_{c}")
        for r in range(bv):
            for c in range(bk // ch_):
                b.l(f"        %tB{r}_{c} = nvgpu.device_async_copy "
                    f"%kg[%nrow{r}, %ks0_{c}], %sb[%c{r}, %c{c * ch_}], {ch_} "
                    f"bypassL1 : memref<64x{k_glob}x{dt}> to {sb2_ty}")
                toks.append(f"%tB{r}_{c}")
        b.l(f"        %acgrp = nvgpu.device_async_create_group {', '.join(toks)}")
        b.l("        nvgpu.device_async_wait %acgrp numGroups = 0")
        b.l("        %el = gdn.elect -> i1")
        b.l("        scf.if %el {")
        b.l(f"          %atok = gdn.mbarrier.arrive %bar : {MBAR_TY} -> {TOKEN_TY}")
        b.l("        }")
    b.l(f"        scf.yield {', '.join(iargs)} : {tylist}")
    b.l("      } else {")
    b.l("        gdn.setmaxnreg inc 224")
    b.l(f"        gdn.mbarrier.try_wait.parity %bar, %parity, %cticks : {MBAR_TY}")
    newv = mma_atoms(b, cfg, "%sa2" if tma else "%sa", sa2_ty,
                     "%sb2" if tma else "%sb", sb2_ty, True, iargs, 0)
    b.l(f"        scf.yield {', '.join(newv)} : {tylist}")
    b.l("      }")
    b.l("      gpu.barrier")
    b.l("      %parn = arith.xori %parity, %true : i1")
    b.l(f"      scf.yield {', '.join(f'%if{i}' for i in range(len(accs)))}, %parn : "
        f"{tylist}, i1")
    b.l("    }")

    if tma:
        b.l("    scf.if %iscons {")
        frag_epilogue(b, cfg, resn, "%sc2", sc2_ty, decay=True)
        b.l("    }")
        b.l("    gpu.barrier")
        b.l("    scf.if %isz {")
        b.l(f"      gdn.tma.async.store %sc2 to %descS[%c0, %c0] : {sc2_ty} -> {desc_ty(f'64x{KKT_N}')}")
        b.l("      gdn.bulk.commit_group")
        b.l("      gdn.bulk.wait_group 0 read")
        b.l("    }")
    else:
        b.l("    scf.if %iscons {")
        frag_epilogue(b, cfg, resn, "%cg", f"memref<64x{KKT_N}x{dt}>",
                      decay=True)
        b.l("    }")


# --------------------------------------------------------------------------
# payload: ref_ldmatrix anchor (G2-C ldmatrix attempt)
# --------------------------------------------------------------------------

def gen_payload_ldmatrix(cfg):
    noargs = cfg.get("driver") == "noargs"
    staging = []
    for c in range(16):
        staging.append(f"      %a{c} = memref.load %lhs[%lane, %c{c}] : memref<16x16xf16>")
        staging.append(f"      memref.store %a{c}, %sa[%lane, %c{c}] : "
                       f"memref<16x16xf16, #gpu.address_space<workgroup>>")
    for c in range(8):
        staging.append(f"      %b{c} = memref.load %rhs[%lane, %c{c}] : memref<16x8xf16>")
        staging.append(f"      memref.store %b{c}, %sb[%lane, %c{c}] : "
                       f"memref<16x16xf16, #gpu.address_space<workgroup>>")
    if noargs:
        gb = Buf()
        emit_global_const(gb, "gdn_in_lhs", 16, 16, "f16",
                          int(cfg.get("seed", 0)))
        emit_global_const(gb, "gdn_in_rhs", 16, 8, "f16",
                          int(cfg.get("seed", 0)))
        emit_global_out(gb, 16, 8, "f32")
        globals_txt = gb.text()
        args = "()"
        # host scope keeps only the launch-dim constants under fresh names
        # (no SSA shadowing); the body's constants are re-defined INSIDE the
        # gpu.launch region so nothing is captured -> zero-arg kernel
        host_consts = ("  %hc1 = arith.constant 1 : index\n"
                       "  %hc32 = arith.constant 32 : index\n")
        d1, d32 = "%hc1", "%hc32"
        region_consts = [f"    %c{c} = arith.constant {c} : index"
                         for c in range(17)]
        gets = ("\n".join([
            "    %lhs = memref.get_global @gdn_in_lhs : memref<16x16xf16>",
            "    %rhs = memref.get_global @gdn_in_rhs : memref<16x8xf16>",
            "    %res = memref.get_global @gdn_out : memref<16x8xf32>",
        ] + region_consts) + "\n")
    else:
        globals_txt = ""
        args = ("(%lhs: memref<16x16xf16>, %rhs: memref<16x8xf16>, "
                "%res: memref<16x8xf32>)")
        host_consts = "".join(f"  %c{c} = arith.constant {c} : index\n"
                              for c in list(range(17)) + [32])
        d1, d32 = "%c1", "%c32"
        gets = ""
    text = f"""{globals_txt}// GDN L2 'ref_ldmatrix' anchor: single f16 m16n8k16 atom whose fragments
// are loaded with in-tree nvgpu.ldmatrix (x4 for A; x2 .trans for B) instead
// of the lane-indexed ld.shared.b16 loads the sm80 rewrite recipe emits.
func.func @gdn_ref_ldmatrix_kernel{args} {{
{host_consts}  gpu.launch blocks(%bx, %by, %bz) in (%gx = {d1}, %gy = {d1}, %gz = {d1})
             threads(%tx, %ty, %tz) in (%bxx = {d32}, %byy = {d1}, %bzz = {d1}) {{
{gets}    %lane = gpu.thread_id x
    %g = arith.divsi %lane, %c4 : index
    %tig = arith.remsi %lane, %c4 : index
    %t2 = arith.muli %tig, %c2 : index
    %t2p1 = arith.addi %t2, %c1 : index
    %g8 = arith.addi %g, %c8 : index
    %padi = arith.constant 0.0 : f16
    %sa = memref.alloc() : memref<16x16xf16, #gpu.address_space<workgroup>>
    %sb = memref.alloc() : memref<16x16xf16, #gpu.address_space<workgroup>>
    %ok = arith.cmpi ult, %lane, %c16 : index
    scf.if %ok {{
      // per-lane ld.global.b16 / st.shared.b16 staging (transfer_write to
      // workgroup does not lower through this pipeline - G2-A constraint 5)
%STAGING%
    }}
    gpu.barrier
    // G4 fix: nvgpu.ldmatrix lowers to ONE per-lane address register
    // (MmaLdMatrixOpToNVVM: getStridedElementPtr(src, indices) -- no
    // built-in per-lane row spread). G2 passed the uniform base %sa[%c0,
    // %c0], so every lane addressed the same 16B row and the mma saw a
    // row-broadcast A (GPU output: all 16 rows identical). The per-lane
    // row/col indices below spell the m16n8k16 A-fragment quadrant order
    // [TL, BL, TR, BR]: lanes 0-7 -> sa[L][0] (TL), 8-15 -> sa[L][0] (BL),
    // 16-23 -> sa[L%8][8] (TR), 24-31 -> sa[8+L%8][8] (BR); B x2 .trans
    // wants k-major rows: lanes 0-7 -> sb[0-7], 8-15 -> sb[8-15].
    %lr8 = arith.remsi %lane, %c8 : index
    %ld8 = arith.divsi %lane, %c8 : index
    %ld8m2 = arith.remsi %ld8, %c2 : index
    %arowoff = arith.muli %ld8m2, %c8 : index
    %ld16 = arith.divsi %lane, %c16 : index
    %acol = arith.muli %ld16, %c8 : index
    %arow = arith.addi %lr8, %arowoff : index
    %af = nvgpu.ldmatrix %sa[%arow, %acol] numTiles = 4 transpose = false : memref<16x16xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %brow = arith.remsi %lane, %c16 : index
    %bf = nvgpu.ldmatrix %sb[%brow, %c0] numTiles = 2 transpose = true : memref<16x16xf16, #gpu.address_space<workgroup>> -> vector<2x2xf16>
    %zf = arith.constant 0.0 : f16
    %zacc = vector.broadcast %zf : f16 to vector<2x2xf16>
    %acc = nvgpu.mma.sync(%af, %bf, %zacc) mmaShape = [16, 8, 16] : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %e0 = vector.extract %acc[0, 0] : f16 from vector<2x2xf16>
    %e1 = vector.extract %acc[0, 1] : f16 from vector<2x2xf16>
    %e2 = vector.extract %acc[1, 0] : f16 from vector<2x2xf16>
    %e3 = vector.extract %acc[1, 1] : f16 from vector<2x2xf16>
    %r0 = arith.addi %g, %c0 : index
    %r1 = arith.addi %g8, %c0 : index
    %x0 = arith.extf %e0 : f16 to f32
    %x1 = arith.extf %e1 : f16 to f32
    %x2 = arith.extf %e2 : f16 to f32
    %x3 = arith.extf %e3 : f16 to f32
    memref.store %x0, %res[%r0, %t2] : memref<16x8xf32>
    memref.store %x1, %res[%r0, %t2p1] : memref<16x8xf32>
    memref.store %x2, %res[%r1, %t2] : memref<16x8xf32>
    memref.store %x3, %res[%r1, %t2p1] : memref<16x8xf32>
    gpu.terminator
  }}
  return
}}
"""
    return text.replace("%STAGING%", "\n".join(staging))


# --------------------------------------------------------------------------
# payload: linalg (mma=off) path
# --------------------------------------------------------------------------

def gen_payload_linalg(b, cfg, stage):
    """mma=off path (G4): G2's staging structure with an f32 accumulator.

    G2 staged whole-K bf16 tiles and accumulated linalg.matmul outs in bf16:
    sequential bf16 accumulation over K=128 rounds away ~1e-2 on the kkt
    diagonal (simulation: 0.028 max diff vs the fp64 twin, 165/2080 tril
    cells over the 5e-3 fla bar) -- a numerical deficiency, not a wiring
    bug. The acc tile is now f32 and linalg.matmul ins(bf16,bf16)
    outs(f32) auto-promotes (extf operands -> f32 mul/add -> f32 store;
    verified against -convert-linalg-to-loops output), matching the mma
    path's f32-accumulation contract. linalg.generic with a manual cast
    region would express this more explicitly but breaks the
    gpu-lower-to-nvvm-pipeline on workgroup memrefs in this tree
    ("conversion of memref memory space ... failed"), so the named op +
    promotion is the load-bearing trick. Operand staging/copy loops are
    byte-identical to G2 (proven to compile); only the acc dtype changes.
    """
    dt = cfg["dtype"]
    bt, bv, bk = cfg["tile"]["M"], cfg["tile"]["N"], cfg["tile"]["K"]
    k_glob = cfg["dims"]["K"]
    ch = chunk(dt)
    kt = stage == "kkt"
    fa_t = f"memref<{bt}x{k_glob}x{dt}, #gpu.address_space<workgroup>>"
    fb_t = (f"memref<{bv}x{k_glob}x{dt}, #gpu.address_space<workgroup>>" if kt
            else f"memref<{k_glob}x{bv}x{dt}, #gpu.address_space<workgroup>>")
    ac_t = f"memref<{bt}x{bv}xf32, #gpu.address_space<workgroup>>"
    btt_t = (f"memref<{k_glob}x{bv}x{dt}, affine_map<(d0, d1) -> "
             f"(d0 + d1 * {k_glob})>, #gpu.address_space<workgroup>>")
    b.l(f"    %fa0 = memref.alloc() : {fa_t}")
    b.l(f"    %fb0 = memref.alloc() : {fb_t}")
    b.l(f"    %acc = memref.alloc() : {ac_t}")
    b.l(f"    %pad{dt} = arith.constant 0.0 : {dt}")
    b.l("    %padf = arith.constant 0.0 : f32")
    b.l(f"    scf.for %zr = %c0 to %c{bt} step %c1 {{")
    b.l(f"      scf.for %zc = %c0 to %c{bv} step %c1 {{")
    b.l(f"        memref.store %padf, %acc[%zr, %zc] : {ac_t}")
    b.l("      }")
    b.l("    }")
    gemms = ([("%kg", "%kg")] if kt else [("%qg", "%hg"), ("%ag", "%vg")])
    for gi, (ag, bg) in enumerate(gemms):
        b.l(f"    scf.for %k{gi} = %c0 to %c{k_glob} step %c{bk} {{")
        for c in range(bk // ch):
            b.l(f"      %lks{gi}_{c} = arith.addi %k{gi}, %c{c * ch} : index")
        for r in range(bt):
            for c in range(bk // ch):
                copy_pair(b, ag, f"memref<64x{k_glob}x{dt}>", f"%grow{r}",
                          f"%lks{gi}_{c}", "%fa0", fa_t, f"%c{r}",
                          f"%lks{gi}_{c}", dt, f"%pad{dt}", f"lvA{gi}_{r}_{c}")
        if kt:
            for r in range(bv):
                for c in range(bk // ch):
                    copy_pair(b, bg, f"memref<64x{k_glob}x{dt}>", f"%nrow{r}",
                              f"%lks{gi}_{c}", "%fb0", fb_t, f"%c{r}",
                              f"%lks{gi}_{c}", dt, f"%pad{dt}",
                              f"lvB{gi}_{r}_{c}")
        else:
            for kr in range(bk):
                b.l(f"      %lkrow{gi}_{kr} = arith.addi %k{gi}, %c{kr} : index")
                for c in range(bv // ch):
                    copy_pair(b, bg, f"memref<{k_glob}x{cfg['dims']['V']}x{dt}>",
                              f"%lkrow{gi}_{kr}", f"%bcol{c}", "%fb0", fb_t,
                              f"%lkrow{gi}_{kr}", f"%bcol{c}", dt,
                              f"%pad{dt}", f"lvB{gi}_{kr}_{c}")
        b.l("      gpu.barrier")
        b.l("    }")
        if kt:
            b.l(f"    %bt0 = memref.transpose %fb0 (d0, d1) -> (d1, d0) : {fb_t} to {btt_t}")
            b.l(f"    linalg.matmul ins(%fa0, %bt0 : {fa_t}, {btt_t}) "
                f"outs(%acc : {ac_t})")
        else:
            b.l(f"    linalg.matmul ins(%fa0, %fb0 : {fa_t}, {fb_t}) "
                f"outs(%acc : {ac_t})")
        b.l("    gpu.barrier")
    out = "%cg" if kt else "%og"
    out_t = f"memref<64x{KKT_N if kt else cfg['dims']['V']}x{dt}>"
    if kt:
        b.l("    %gam = arith.constant -6.25e-02 : f32")
        b.l("    %epz = arith.constant 0.0 : f32")
    b.l(f"    scf.for %er = %c0 to %c{bt} step %c1 {{")
    b.l(f"      scf.for %ec = %c0 to %c{bv} step %c1 {{")
    b.l(f"        %ex = memref.load %acc[%er, %ec] : {ac_t}")
    b.l("        %erow = arith.addi %m0, %er : index")
    b.l("        %ecol = arith.addi %n0base, %ec : index")
    if kt:
        b.l("        %ed = arith.subi %erow, %ecol : index")
        b.l("        %edi = arith.index_cast %ed : index to i32")
        b.l("        %edf = arith.sitofp %edi : i32 to f32")
        b.l("        %edg = arith.mulf %edf, %gam : f32")
        b.l("        %ee = math.exp2 %edg : f32")
        b.l(f"        %em = arith.mulf %ex, %ee : f32")
        b.l("        %eq = arith.cmpi sle, %ecol, %erow : index")
        b.l("        %es = arith.select %eq, %em, %epz : f32")
        val = "%es"
    else:
        val = "%ex"
    if dt == "f32":
        b.l(f"        memref.store {val}, {out}[%erow, %ecol] : {out_t}")
    else:
        b.l(f"        %et = arith.truncf {val} : f32 to {dt}")
        b.l(f"        memref.store %et, {out}[%erow, %ecol] : {out_t}")
    b.l("      }")
    b.l("    }")


# --------------------------------------------------------------------------
# payload top level
# --------------------------------------------------------------------------

def gen_payload_ref(cfg):
    noargs = cfg.get("driver") == "noargs"
    pre, args, gets = "", ("(%lhs: memref<16x16xf16>, %rhs: memref<16x8xf16>, "
                           "%res: memref<16x8xf16>)"), ""
    if noargs:
        gb = Buf()
        emit_global_const(gb, "gdn_in_lhs", 16, 16, "f16",
                          int(cfg.get("seed", 0)))
        emit_global_const(gb, "gdn_in_rhs", 16, 8, "f16",
                          int(cfg.get("seed", 0)))
        emit_global_out(gb, 16, 8, "f16")
        pre = gb.text()
        args = "()"
        gets = ("    %lhs = memref.get_global @gdn_in_lhs : memref<16x16xf16>\n"
                "    %rhs = memref.get_global @gdn_in_rhs : memref<16x8xf16>\n"
                "    %res = memref.get_global @gdn_out : memref<16x8xf16>\n")
    return f"""{pre}// GDN L2 'ref' anchor: single-atom f16 matmul lowered via the in-tree
// transform.nvgpu.rewrite_matmul_as_mma_sync (sm80 recipe retargeted sm_120a).
func.func @gdn_ref_kernel{args} {{
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%bxx = %c32, %byy = %c1, %bzz = %c1) {{
{gets}    linalg.matmul ins(%lhs, %rhs: memref<16x16xf16>, memref<16x8xf16>)
                 outs(%res: memref<16x8xf16>)
    gpu.terminator
  }}
  return
}}
"""


def gen_payload(cfg):
    stage = cfg["stage"]
    if stage == "ref":
        return gen_payload_ref(cfg)
    if stage == "ref_ldmatrix":
        return gen_payload_ldmatrix(cfg)
    dt = cfg["dtype"]
    bt, bv, bk = cfg["tile"]["M"], cfg["tile"]["N"], cfg["tile"]["K"]
    k_glob = cfg["dims"]["K"]
    n_glob = KKT_N if stage == "kkt" else cfg["dims"]["V"]
    warps = cfg["warps"]
    wn = bv // warps if cfg["mma"] else 0
    ch = chunk(dt)

    b = Buf()
    noargs = cfg.get("driver") == "noargs"
    seed = int(cfg.get("seed", 0))
    hdr = json.dumps(cfg, separators=(",", ":"))
    if noargs:
        # G3-A self-contained variant: inputs as module-level constant
        # memref.globals (cloned into the gpu.module by gpu-kernel-outlining),
        # output to a public zero-init @gdn_out global => zero kernel args.
        if stage == "kkt":
            emit_global_const(b, "gdn_in_k", 64, k_glob, dt, seed)
            emit_global_out(b, 64, KKT_N, dt)
        else:
            emit_global_const(b, "gdn_in_q", 64, k_glob, dt, seed)
            emit_global_const(b, "gdn_in_h", k_glob, n_glob, dt, seed)
            emit_global_const(b, "gdn_in_a", 64, k_glob, dt, seed)
            emit_global_const(b, "gdn_in_v", k_glob, n_glob, dt, seed)
            emit_global_out(b, 64, n_glob, dt)
    if stage == "kkt":
        b.l(f"// GDN L2 stage 'kkt': C = tril(k @ k^T) * exp2decay  knobs={hdr}")
        if noargs:
            b.l("func.func @gdn_kkt_kernel() {")
        else:
            b.l(f"func.func @gdn_kkt_kernel(%kg: memref<64x{k_glob}x{dt}>, "
                f"%cg: memref<64x64x{dt}>) {{")
    else:
        b.l(f"// GDN L2 stage 'o': o = (q @ h) + (A @ v)  knobs={hdr}")
        if noargs:
            b.l("func.func @gdn_o_kernel() {")
        else:
            b.l(f"func.func @gdn_o_kernel(%qg: memref<64x{k_glob}x{dt}>, "
                f"%hg: memref<{k_glob}x{n_glob}x{dt}>, "
                f"%ag: memref<64x{k_glob}x{dt}>, "
                f"%vg: memref<{k_glob}x{n_glob}x{dt}>, "
                f"%og: memref<64x{n_glob}x{dt}>) {{")
    consts = {0, 1, 2, 4, 8, 16, 32, 64, bt, bv, bk, k_glob, n_glob,
              32 * warps, n_glob // bv, 64 // bt}
    ws = cfg["warpspec"] == "producer_consumer"
    if ws:
        consts.add(32 * (warps + 1))
        consts.add((bt * bk + bk * bv) * (2 if dt == "bf16" else 4))
        consts.add(32 * warps)
        consts.add(10000000)
    if wn:
        consts.add(wn)
    consts |= set(range(bt))            # %grow offsets
    consts |= set(range(bv))            # %nrow offsets
    consts |= set(range(bk))            # B-tile row offsets
    consts |= set(range(0, bk, ch))     # chunk offsets
    if cfg["mma"]:
        consts |= {mi * 16 for mi in range(bt // 16)}
        consts |= {ni * 8 for ni in range(wn // 8)}
        consts |= {kk * atom_k(dt) for kk in range(bk // atom_k(dt))}
        consts |= set(range(0, bv, ch))  # o-stage bcol chunk offsets
    else:
        consts |= set(range(0, bv, ch))
    const_stmts = [f"%c{c} = arith.constant {c} : index"
                   for c in sorted(consts)]
    if ws:
        const_stmts += [
            "%cticks = arith.constant 10000000 : index",
            "%true = arith.constant true",
            "%false = arith.constant false",
            "%neg1 = arith.constant -1 : i32",
            "%c0i = arith.constant 0 : i32",
            "%c31i = arith.constant 31 : i32",
        ]
    if noargs:
        # Constants live ONLY inside the gpu.launch region (zero captures =>
        # zero-arg kernel; MLIR forbids SSA shadowing so names must not
        # collide). Host scope keeps a minimal renamed set for the launch
        # dims and (tma) the host-side tensormap descriptors.
        b.l(f"  %hg_gx = arith.constant {64 // bt} : index")
        b.l(f"  %hg_gy = arith.constant {n_glob // bv} : index")
        b.l("  %hg_one = arith.constant 1 : index")
        b.l(f"  %hg_btx = arith.constant "
            f"{32 * (warps + 1) if ws else 32 * warps} : index")
        d_gx, d_gy, d_one, d_btx = ("%hg_gx", "%hg_gy", "%hg_one", "%hg_btx")
        if ws and cfg["tma"] == "on":
            b.l("  %hd_one = arith.constant 1 : index")
            b.l(f"  %hd_n = arith.constant {KKT_N} : index")
            b.l(f"  %hd_bt = arith.constant {bt} : index")
            b.l(f"  %hd_bk = arith.constant {bk} : index")
            b.l(f"  %hd_bv = arith.constant {bv} : index")
            dsc = ("%hd_one", "%hd_n", "%hd_bt", "%hd_bk", "%hd_bv")
        else:
            dsc = ("%c1", "%c64", f"%c{bt}", f"%c{bk}", f"%c{bv}")
    else:
        for s in const_stmts:
            b.l(f"  {s}")
        d_gx, d_gy, d_one, d_btx = (f"%c{64 // bt}", f"%c{n_glob // bv}",
                                    "%c1",
                                    f"%c{32 * (warps + 1) if ws else 32 * warps}")
        dsc = ("%c1", "%c64", f"%c{bt}", f"%c{bk}", f"%c{bv}")

    if ws and cfg["tma"] == "on":
        # host-side tensormap descriptors. The compile-level lowering turns
        # each into a host runtime call (mgpuTensorMapEncodeTiledMemref) +
        # a .u64 .ptr kernel parameter; that helper reverses MEMREF dim
        # order into CUDA dim order (innermost first) for globalDim/boxDim,
        # while load/store coordinates pass through UNREVERSED -- so the
        # payload emits CUDA-ordered coordinates (K offset first; see
        # gen_payload_ws). G4's host wiring (tools/gdn_verify.py) encodes
        # the maps directly per the spec written to tensormap_spec.json.
        # noargs mode: host-side views of the module globals (%kgh/%cgh);
        # the kernel reads/writes only through the tensormap params (its
        # data movement no longer references @gdn_in_k/@gdn_out directly,
        # so the verifier stages inputs / reads outputs from the host-side
        # buffers described by the spec).
        kh, ch = ("%kgh", "%cgh") if noargs else ("%kg", "%cg")
        if noargs:
            b.l(f"  %kgh = memref.get_global @gdn_in_k : memref<64x{k_glob}x{dt}>")
            b.l(f"  %cgh = memref.get_global @gdn_out : memref<64x{KKT_N}x{dt}>")
        b.l(f"  %castA = memref.cast {kh} : memref<64x{k_glob}x{dt}> to memref<*x{dt}>")
        b.l(f"  %descA = nvgpu.tma.create.descriptor %castA box[{dsc[0]}, {dsc[0]}, {dsc[2]}, {dsc[3]}]"
            f" : memref<*x{dt}> -> {desc_ty(f'1x1x{bt}x{bk}')}")
        b.l(f"  %descB = nvgpu.tma.create.descriptor %castA box[{dsc[0]}, {dsc[0]}, {dsc[0]}, {dsc[4]}, {dsc[3]}]"
            f" : memref<*x{dt}> -> {desc_ty(f'1x1x1x{bv}x{bk}')}")
        b.l(f"  %castS = memref.cast {ch} : memref<64x{KKT_N}x{dt}> to memref<*x{dt}>")
        b.l(f"  %descS = nvgpu.tma.create.descriptor %castS box[{dsc[1]}, {dsc[1]}]"
            f" : memref<*x{dt}> -> {desc_ty(f'64x{KKT_N}')}")

    b.l(f"  gpu.launch blocks(%bx, %by, %bz) in (%gx = {d_gx}, "
        f"%gy = {d_gy}, %gz = {d_one})")
    b.l(f"             threads(%tx, %ty, %tz) in (%bxx = {d_btx}, "
        f"%byy = {d_one}, %bzz = {d_one}) {{")
    if noargs:
        # region-local global views: no captures => zero runtime kernel args
        if stage == "kkt":
            b.l(f"    %kg = memref.get_global @gdn_in_k : memref<64x{k_glob}x{dt}>")
            b.l(f"    %cg = memref.get_global @gdn_out : memref<64x{KKT_N}x{dt}>")
        else:
            b.l(f"    %qg = memref.get_global @gdn_in_q : memref<64x{k_glob}x{dt}>")
            b.l(f"    %hg = memref.get_global @gdn_in_h : memref<{k_glob}x{n_glob}x{dt}>")
            b.l(f"    %ag = memref.get_global @gdn_in_a : memref<64x{k_glob}x{dt}>")
            b.l(f"    %vg = memref.get_global @gdn_in_v : memref<{k_glob}x{n_glob}x{dt}>")
            b.l(f"    %og = memref.get_global @gdn_out : memref<64x{n_glob}x{dt}>")
        # region-local constants (shadow the host-scope copies)
        for s in const_stmts:
            b.l(f"    {s}")
    b.l("    %bid_x = gpu.block_id x")
    b.l("    %bid_y = gpu.block_id y")
    b.l(f"    %m0 = arith.muli %bid_x, %c{bt} : index")
    b.l(f"    %n0base = arith.muli %bid_y, %c{bv} : index")
    b.l("    %tid = gpu.thread_id x")
    b.l("    %warp = arith.divsi %tid, %c32 : index")
    b.l("    %lane = arith.remsi %tid, %c32 : index")
    b.l("    %g = arith.divsi %lane, %c4 : index")
    b.l("    %tig = arith.remsi %lane, %c4 : index")
    b.l("    %t2 = arith.muli %tig, %c2 : index")
    b.l("    %t2p1 = arith.addi %t2, %c1 : index")
    b.l("    %g8 = arith.addi %g, %c8 : index")
    b.l("    %t2x8 = arith.addi %t2, %c8 : index")
    b.l("    %t2p1x8 = arith.addi %t2p1, %c8 : index")
    if cfg["mma"] and not ws:
        b.l(f"    %wnoff = arith.muli %warp, %c{wn} : index")
    for r in range(bt):
        b.l(f"    %grow{r} = arith.addi %m0, %c{r} : index")
    for r in range(bv):
        b.l(f"    %nrow{r} = arith.addi %n0base, %c{r} : index")
    if stage == "o":
        for c in range(bv // ch):
            b.l(f"    %bcol{c} = arith.addi %n0base, %c{c * ch} : index")

    if ws:
        gen_payload_ws(b, cfg, stage)
    elif cfg["mma"]:
        gen_payload_mma(b, cfg, stage)
    else:
        gen_payload_linalg(b, cfg, stage)

    b.l("    gpu.terminator")
    b.l("  }")
    b.l("  return")
    b.l("}")
    return b.text()


# --------------------------------------------------------------------------
# transform script (knob carrier)
# --------------------------------------------------------------------------

def gen_transform(cfg):
    b = Buf()
    b.l("module attributes {transform.with_named_sequence} {")
    b.l("  transform.named_sequence @__transform_main(%arg0: !transform.any_op "
        "{transform.readonly}) {")
    if (cfg["stage"] == "ref_ldmatrix"
            or cfg.get("warpspec") == "producer_consumer"):
        # direct-ops payloads (gdn hatch + nvgpu.device_async_copy +
        # nvgpu.ldmatrix + nvgpu.mma.sync): nothing to schedule
        pass
    elif cfg["stage"] == "ref":
        b.l('    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0')
        b.l("      : (!transform.any_op) -> !transform.any_op")
        b.l("    transform.nvgpu.rewrite_matmul_as_mma_sync %matmul")
        b.l("      : (!transform.any_op) -> ()")
    elif cfg["mma"]:
        pipe = cfg["pipeline"]
        if pipe > 0:
            b.l('    %allocs = transform.structured.match ops{["memref.alloc"]} in %arg0')
            b.l('      : (!transform.any_op) -> !transform.op<"memref.alloc">')
            b.l(f"    transform.memref.multibuffer %allocs "
                f"<factor = {pipe}, skip_analysis>")
            b.l('      : (!transform.op<"memref.alloc">) -> !transform.any_op')
        b.l('    %func = transform.structured.match ops{["func.func"]} in %arg0')
        b.l("      : (!transform.any_op) -> !transform.any_op")
        if pipe > 0:
            b.l("    transform.nvgpu.create_async_groups %func bypass_l1 : "
                "(!transform.any_op) -> (!transform.any_op)")
            b.l('    %loops = transform.structured.match ops{["scf.for"]} in %arg0')
            b.l("      : (!transform.any_op) -> !transform.any_op")
            b.l(f"    transform.nvgpu.pipeline_shared_memory_copies "
                f"failures(propagate) %loops depth = {pipe} peel_epilogue : "
                f"(!transform.any_op) -> !transform.any_op")
        else:
            # pipeline=0: single-buffer cp.async.ca (L1-allocating) copies,
            # no multibuffering, no software pipelining.
            b.l("    transform.nvgpu.create_async_groups %func : "
                "(!transform.any_op) -> (!transform.any_op)")
    else:
        tm = min(cfg["tile"]["M"], 16)
        tn = min(cfg["tile"]["N"], 16)
        b.l('    %mms = transform.structured.match ops{["linalg.matmul"]} in %arg0')
        b.l("      : (!transform.any_op) -> !transform.any_op")
        b.l(f"    %tiled, %loops:2 = transform.structured.tile_using_for %mms "
            f"tile_sizes [{tm}, {tn}] : (!transform.any_op) -> "
            f"(!transform.any_op, !transform.any_op, !transform.any_op)")
        # NOTE: transform.structured.vectorize hard-fails on the shared-memory
        # matmuls in this tree (no failures(suppress) clause on this op), so
        # the plain-vector path is realized by the runner's
        # -convert-linalg-to-loops + pipeline vector lowering (ld.global.v /
        # fma vector shapes still carry the tile knob).
        if cfg["pipeline"] > 0:
            b.l('    %func2 = transform.structured.match ops{["func.func"]} in %arg0')
            b.l("      : (!transform.any_op) -> !transform.any_op")
            b.l("    transform.nvgpu.create_async_groups %func2 bypass_l1 : "
                "(!transform.any_op) -> (!transform.any_op)")
        else:
            b.l('    %func2 = transform.structured.match ops{["func.func"]} in %arg0')
            b.l("      : (!transform.any_op) -> !transform.any_op")
            b.l("    transform.nvgpu.create_async_groups %func2 : "
                "(!transform.any_op) -> (!transform.any_op)")
    b.l("    transform.yield")
    b.l("  }")
    b.l("}")
    return b.text()


# --------------------------------------------------------------------------
# built-in matrix
# --------------------------------------------------------------------------

def C(name, stage, dtype, K, V, tM, tN, tK, warps, mma, pipe,
      warpspec="off", tma="off"):
    cfg = {"name": name, "stage": stage, "dtype": dtype, "dims": {"K": K, "V": V},
           "tile": {"M": tM, "N": tN, "K": tK}, "warps": warps, "mma": mma,
           "pipeline": pipe, "warpspec": warpspec, "tma": tma}
    validate(cfg)
    return cfg


def builtin_configs():
    return [
        # kkt stage (K=128, N=64): mma on/off x pipe 0/2/3 x tiles x warps
        C("kkt_bf16_mma_p0_t64x64x32_w4", "kkt", "bf16", 128, 64, 64, 64, 32, 4, True, 0),
        C("kkt_bf16_mma_p2_t64x64x32_w4", "kkt", "bf16", 128, 64, 64, 64, 32, 4, True, 2),
        C("kkt_bf16_mma_p3_t64x64x32_w4", "kkt", "bf16", 128, 64, 64, 64, 32, 4, True, 3),
        C("kkt_bf16_mma_p2_t64x64x16_w4", "kkt", "bf16", 128, 64, 64, 64, 16, 4, True, 2),
        C("kkt_bf16_mma_p2_t32x64x32_w4", "kkt", "bf16", 128, 64, 32, 64, 32, 4, True, 2),
        C("kkt_bf16_mma_p2_t64x64x32_w8", "kkt", "bf16", 128, 64, 64, 64, 32, 8, True, 2),
        C("kkt_bf16_mma_p2_t64x64x32_w2", "kkt", "bf16", 128, 64, 64, 64, 32, 2, True, 2),
        C("kkt_bf16_nomma_p2_t64x64x32_w4", "kkt", "bf16", 128, 64, 64, 64, 32, 4, False, 2),
        C("kkt_bf16_nomma_p0_t64x64x32_w4", "kkt", "bf16", 128, 64, 64, 64, 32, 4, False, 0),
        C("kkt_bf16_nomma_p2_t64x32x32_w4", "kkt", "bf16", 128, 64, 64, 32, 32, 4, False, 2),
        # o stage: both GEMMs, V/K variants
        C("o_bf16_mma_p2_t64x64x32_w4_K64V64", "o", "bf16", 64, 64, 64, 64, 32, 4, True, 2),
        C("o_bf16_mma_p3_t64x128x16_w8_K128V128", "o", "bf16", 128, 128, 64, 128, 16, 8, True, 3),
        C("o_bf16_mma_p0_t32x64x32_w2_K64V64", "o", "bf16", 64, 64, 32, 64, 32, 2, True, 0),
        C("o_bf16_nomma_p2_t64x64x32_w4_K64V64", "o", "bf16", 64, 64, 64, 64, 32, 4, False, 2),
        C("o_f32_mma_p2_t64x64x32_w4_K64V64", "o", "f32", 64, 64, 64, 64, 32, 4, True, 2),
        C("o_f32_nomma_p0_t64x64x32_w4_K64V64", "o", "f32", 64, 64, 64, 64, 32, 4, False, 0),
        # ref anchor: the actual in-tree rewrite op (sm80 recipe, f16 atom)
        {"name": "ref_f16_sm80_atom", "stage": "ref", "dtype": "f16",
         "dims": {"K": 16, "V": 8}, "tile": {"M": 16, "N": 8, "K": 16},
         "warps": 1, "mma": True, "pipeline": 0,
         "warpspec": "off", "tma": "off"},
        # G2-C warp-specialization / FlashQLA pole
        C("kkt_bf16_mma_p2_t64x64x64_w4_ws", "kkt", "bf16", 128, 64,
          64, 64, 64, 4, True, 2, warpspec="producer_consumer"),
        C("kkt_bf16_mma_tma_ws", "kkt", "bf16", 128, 64,
          64, 64, 64, 4, True, 2, warpspec="producer_consumer", tma="on"),
        # G2-C ldmatrix attempt: in-tree nvgpu.ldmatrix fragment loads
        {"name": "ref_ldmatrix", "stage": "ref_ldmatrix", "dtype": "f16",
         "dims": {"K": 16, "V": 8}, "tile": {"M": 16, "N": 8, "K": 16},
         "warps": 1, "mma": True, "pipeline": 0,
         "warpspec": "off", "tma": "off"},
    ]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def tensormap_spec(cfg):
    """Host-side CUtensorMap encode spec for tma=on configs (G4 deliverable:
    the generator OWNS the descriptor contract; tools/gdn_verify.py encodes
    the maps with cuda-bindings exactly as specified here).

    Dim order = CUDA order (innermost/fastest-varying first), matching
    cuTensorMapEncodeTiled's globalDim/globalStrides/boxDim and the raw
    cp.async.bulk.tensor coordinate order the kernel emits.
    """
    bt, bv, bk = cfg["tile"]["M"], cfg["tile"]["N"], cfg["tile"]["K"]
    k_glob = cfg["dims"]["K"]
    esz = 2  # bf16 only (tma is kkt bf16)
    row = k_glob * esz                 # bytes per k row (row-major k global)
    plane = 64 * k_glob * esz
    return {
        "config": cfg["name"],
        "dtype": "bfloat16",
        "cuda_tensor_map_dtype": "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
        "swizzle": "CU_TENSOR_MAP_SWIZZLE_NONE",
        "interleave": "CU_TENSOR_MAP_INTERLEAVE_NONE",
        "l2_promotion": "CU_TENSOR_MAP_L2_PROMOTION_NONE",
        "oob_fill": "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE",
        "descriptors": {
            "descA": {
                "rank": 4,
                "source": "input k, row-major [64 x K] bf16",
                "global_dim": [k_glob, 64, 1, 1],          # K innermost
                "global_strides_bytes": [row, plane, plane],
                "box_dim": [bk, bt, 1, 1],                 # K x N tile
                "kernel_coords": "[%k0, 0, 0, 0] (CUDA order: K offset first)",
                "smem_tile": f"[{bt} x {bk}] K-contiguous (k-rows A tile)",
            },
            "descB": {
                "rank": 5,
                "source": "input k, row-major [64 x K] bf16 (viewed as k^T)",
                "global_dim": [k_glob, 64, 1, 1, 1],        # K innermost
                "global_strides_bytes": [row, plane, plane, plane],
                "box_dim": [bk, bv, 1, 1, 1],              # K x N tile
                "kernel_coords": "[%k0, 0, 0, 0, 0] (CUDA order)",
                "smem_tile": f"[{bv} x {bk}] K-contiguous (k-rows B tile, "
                             "transposed b_frag reads)",
            },
            "descS": {
                "rank": 2,
                "source": "output c, row-major [64 x 64] bf16",
                "global_dim": [KKT_N, 64],                  # N innermost
                "global_strides_bytes": [KKT_N * esz],
                "box_dim": [KKT_N, 64],
                "kernel_coords": "[0, 0]",
                "smem_tile": "[64 x 64] N-contiguous",
            },
        },
        "kernel_params": "three .u64 .ptr params (descA, descB, descS), "
                         "each holding a DEVICE pointer to a 64-byte-aligned"
                         " 128-byte CUtensorMap",
    }


def main():
    ap = argparse.ArgumentParser(description="GDN L2/L3 knob library generator")
    ap.add_argument("--list-configs", action="store_true")
    ap.add_argument("--gen", metavar="NAME")
    ap.add_argument("--json", metavar="FILE")
    ap.add_argument("--name")
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=0,
                    help="deterministic data seed for --driver noargs "
                         "embedded inputs (default 0; same seed -> "
                         "identical payload/inputs, see noargs_val)")
    ap.add_argument("--driver", choices=("args", "noargs"), default=None,
                    help="args: kernel takes memref pointer args (G2 default). "
                    "noargs (G3-A): inputs as memref.global constants, output "
                    "to public zero-init @gdn_out global => zero-arg cubin "
                    "launchable by tools/gdn_time.py without host marshalling.")
    args = ap.parse_args()

    cfgs = {c["name"]: c for c in builtin_configs()}
    if args.list_configs:
        print("# GDN L2/L3 built-in knob matrix (payload+transform generator)")
        print(f"# {'name':38s} {'stage':13s} {'dt':4s} {'K':>4s} {'V':>4s} "
              f"{'tile(MxNxK)':>12s} {'w':>2s} {'mma':>5s} {'pipe':>4s} "
              f"{'warpspec':>18s} {'tma':>3s}")
        for c in cfgs.values():
            print(f"  {c['name']:38s} {c['stage']:13s} {c['dtype']:4s} "
                  f"{c['dims']['K']:4d} {c['dims']['V']:4d} "
                  f"{c['tile']['M']:2d}x{c['tile']['N']:3d}x{c['tile']['K']:3d} "
                  f"{c['warps']:2d} {str(c['mma']):>5s} {c['pipeline']:4d} "
                  f"{c['warpspec']:>18s} {c['tma']:>3s}")
        return 0

    cfg = None
    if args.gen:
        if args.gen not in cfgs:
            print(f"error: unknown config '{args.gen}'", file=sys.stderr)
            return 2
        cfg = dict(cfgs[args.gen])
    elif args.json:
        with open(args.json) as f:
            cfg = json.load(f)
        if args.name:
            cfg["name"] = args.name
        validate(cfg)
    else:
        ap.print_help()
        return 2
    cfg["driver"] = args.driver or cfg.get("driver", "args")
    cfg.setdefault("seed", args.seed)
    if cfg["driver"] == "noargs":
        cfg["seed"] = args.seed

    out = args.out
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "payload.mlir"), "w") as f:
        f.write(gen_payload(cfg))
    with open(os.path.join(out, "transform.mlir"), "w") as f:
        f.write(gen_transform(cfg))
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if cfg.get("tma") == "on":
        with open(os.path.join(out, "tensormap_spec.json"), "w") as f:
            json.dump(tensormap_spec(cfg), f, indent=2)
    print(f"generated payload.mlir + transform.mlir + config.json for "
          f"'{cfg['name']}' in {out}"
          + (" + tensormap_spec.json" if cfg.get("tma") == "on" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
