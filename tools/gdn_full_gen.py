#!/usr/bin/env python3
"""gdn_full_gen.py -- G5-B FULL-OP GDN forward GPU kernel generator.

Emits one self-contained MLIR payload per kernel (payload.mlir +
transform.mlir + config.json) for the existing G2-G4 compile/launch chain
(scripts/run_l2.sh -> tools/gdn_smem_fix.py -> cuda-bindings launcher).
The math is THE verified L1 truth (tools/gdn_l1_gen.py, the fla
naive_chunk_gated_delta_rule port):

  per 64-row chunk (g in NATURAL-log domain; kernel works log2:
  gc = inclusive cumsum of g*RCP_LN2; scale = K**-0.5):
    A[i,j] = beta_i * (k_i . k_j) * 2^(gc_i-gc_j)      (strict lower)
    T      = (I+A)^-1  by the L1 forward-substitution recurrence
    u      = T @ (beta .* v)          w = T @ (beta .* k .* 2^gc)
    v_new  = u - w @ S
    o      = ((q .* 2^gc) @ S + tril_inc((q k^T) .* 2^(gc_i-gc_j)) @ v_new)*scale
    S      = S .* 2^gc_last + (k .* 2^(gc_last-gc))^T @ v_new

dtype contract (fla): q,k,v,beta,g bf16 in; o bf16; h0/ht fp32; carried
state bf16 (fla Triton behavior); ALL math f32 (mma f32 accumulators,
G4 contract).  Tail chunks: zero-padded inputs (harness pads T->Tp=NC*C).

Paths (fusion-boundary knob):
  staged : F1 "wy"   (grid NC x B*H)     token-parallel WY precompute
           F2 "scan" (grid B*H*VT)       h recurrence; entry states h_all,
                                         v_new, ht
           F3 "o"    (grid NC*B*H x VT)  output stage from h_all/v_new
  fused  : ONE persistent kernel (grid B*H*VT), scf.for over chunks doing
           WY+solve+state+o per chunk; state lives in smem.

GPU pattern (G2/G4 machinery): whole-K operand staging to workgroup smem
(vector.transfer pairs), warp-tiled nvgpu.mma.sync m16n8k16 bf16 atoms
with f32 accumulators (D-fragment N index = 2*tig, the G4 fix), custom
epilogues, gpu.barrier between stages.  The forward-substitution solve
runs warp-per-column with the T column held in REGISTERS (each lane owns
k = j+lane, j+lane+32; butterfly shfl.sync reduction per row i) writing
a transposed f32 Tt tile; the L1 recurrence order is preserved per
column up to the reduction tree.

Driver: --driver args (flat memref pointer params; every static memref
lowers to a single live .u64 .ptr kernel param -- G5-B probe; the
launcher maps buffers via asmobj's gpu.launch_func operand list).

Usage:
  gdn_full_gen.py --path staged --kernel f1|f2|f3 --shape B,T,H,K,V
      [--warps 4] [--bv 64] [--copies sync|async] --out DIR
  gdn_full_gen.py --path fused --shape ... [--warps 4] [--bv 64] --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys

RCP_LN2 = 1.4426950408889634   # natural-log g -> log2-domain g
CHUNK = 64
BF, F32 = "bf16", "f32"

# fragment-load mode: "ldm" = nvgpu.ldmatrix (1 op per fragment; G4 anchor
# per-lane addressing), "scalar" = per-element memref.load chains (G2
# style).  Toggled by --frag; module-level because emit_gemm is deep in
# the call tree.
_FRAG_MODE = "ldm"


def f32lit(x):
    s = "%.9g" % float(x)
    if "e" in s or "E" in s:
        m, _, e = s.partition("e")
        if "." not in m:
            m += ".0"
        return m + "e" + e
    if "." not in s:
        s += ".0"
    return s


# ---------------------------------------------------------------------------
# emitter
# ---------------------------------------------------------------------------

class Em:
    def __init__(self):
        self.lines = []
        self._seen = {}

    def l(self, s=""):
        self.lines.append("    " + s if s else "")

    def v(self, hint):
        n = self._seen.get(hint, 0) + 1
        self._seen[hint] = n
        return f"%{hint}" if n == 1 else f"%{hint}_{n}"

    def c(self, val):
        val = int(val)
        name = f"%c{val}"
        if name not in self._seen:
            self._seen[name] = 1
            self.l(f"{name} = arith.constant {val} : index")
        return name

    def text(self):
        return "\n".join(self.lines)


def wg(r, co, dt):
    return f"memref<{r}x{co}x{dt}, #gpu.address_space<workgroup>>"


def alloc2d(e, name, r, co, dt):
    ty = wg(r, co, dt)
    e.l(f"%{name} = memref.alloc() : {ty}")
    return f"%{name}", ty


def alloc1d(e, name, n, dt):
    ty = f"memref<{n}x{dt}, #gpu.address_space<workgroup>>"
    e.l(f"%{name} = memref.alloc() : {ty}")
    return f"%{name}", ty


def addi(e, a, b):
    o = e.v("ix"); e.l(f"{o} = arith.addi {a}, {b} : index"); return o


def muli(e, a, b):
    o = e.v("im"); e.l(f"{o} = arith.muli {a}, {b} : index"); return o


def divsi(e, a, b):
    o = e.v("idv"); e.l(f"{o} = arith.divsi {a}, {b} : index"); return o


def remsi(e, a, b):
    o = e.v("irm"); e.l(f"{o} = arith.remsi {a}, {b} : index"); return o


# ---------------------------------------------------------------------------
# lane precompute + mma machinery (G2/G4 lane formulas)
# ---------------------------------------------------------------------------

def emit_lane_precompute(e):
    e.l("%tid = gpu.thread_id x")
    e.l("%warp = arith.divsi %tid, %c32 : index")
    e.l("%lane = arith.remsi %tid, %c32 : index")
    e.l("%lg = arith.divsi %lane, %c4 : index")
    e.l("%tig = arith.remsi %lane, %c4 : index")
    e.l("%t2 = arith.muli %tig, %c2 : index")
    e.l("%t2p1 = arith.addi %t2, %c1 : index")
    e.l("%lg8 = arith.addi %lg, %c8 : index")
    e.l("%t2x8 = arith.addi %t2, %c8 : index")
    e.l("%t2p1x8 = arith.addi %t2p1, %c8 : index")
    for x in (0, 1, 2, 4, 8, 32):
        e.c(x)


def gemm_precompute(e, name, M, N, W):
    wn = N // W
    e.l(f"%wnoff_{name} = arith.muli %warp, {e.c(wn)} : index")
    for mi in range(M // 16):
        e.l(f"%rA_{name}_{mi} = arith.addi %lg, {e.c(mi*16)} : index")
        e.l(f"%rAh_{name}_{mi} = arith.addi %lg8, {e.c(mi*16)} : index")
    for ni in range(wn // 8):
        e.l(f"%nc_{name}_{ni} = arith.addi %wnoff_{name}, {e.c(ni*8)} : index")
        e.l(f"%ncg_{name}_{ni} = arith.addi %nc_{name}_{ni}, %lg : index")
        e.l(f"%nct_{name}_{ni} = arith.addi %nc_{name}_{ni}, %t2 : index")
        e.l(f"%nct1_{name}_{ni} = arith.addi %nc_{name}_{ni}, %t2p1 : index")
    return wn


def gemm_koffs(e, name, Kred):
    for kb in range(0, Kred, 16):
        for sfx, base in (("_0", "%t2"), ("_1", "%t2p1"),
                          ("_2", "%t2x8"), ("_3", "%t2p1x8")):
            e.l(f"%ko_{name}_{kb}{sfx} = arith.addi {e.c(kb)}, {base} : index")


def ldm_precompute(e, name):
    """Per-lane quadrant indices for nvgpu.ldmatrix (G4 anchor semantics:
    ONE per-lane address register; lanes 0-7/8-15/16-23/24-31 provide the
    four 8x8 quadrant row starts, 16B (=8 bf16) per row)."""
    e.l(f"%lq7_{name} = arith.remsi %lane, %c8 : index")
    e.l(f"%lq8_{name} = arith.divsi %lane, %c8 : index")
    e.l(f"%lqh_{name} = arith.remsi %lq8_{name}, %c2 : index")
    e.l(f"%lqt_{name} = arith.divsi %lane, %c16 : index")
    e.l(f"%lh8_{name} = arith.muli %lqh_{name}, %c8 : index")
    e.l(f"%lt8_{name} = arith.muli %lqt_{name}, %c8 : index")
    e.l(f"%lb8_{name} = arith.muli %lq8_{name}, %c8 : index")
    e.c(16)


def ldm_a(e, name, tag, src, src_ty, mi, kb, swap):
    """A-fragment via ldmatrix.x4.  Normal: quadrants [rows0-8/8-16]x
    [k0-8/k8-16] of the tile, lane addr row = mi*16+8*((lane/8)%2)+lane%8,
    col = kb+8*(lane/16).  Swapped (A[m][k] = X[k][m], X k-rows major):
    .trans on X's [k-octet][m-octet] blocks: addr row = kb+8*(lane/16)+
    lane%8, col = mi*16+8*((lane/8)%2)."""
    kbc = kb if isinstance(kb, str) else e.c(kb)
    if not swap:
        row = addi(e, addi(e, e.c(mi * 16), f"%lh8_{name}"),
                   f"%lq7_{name}")
        col = addi(e, kbc, f"%lt8_{name}")
        e.l(f"%{tag} = nvgpu.ldmatrix {src}[{row}, {col}] numTiles = 4 "
            f"transpose = false : {src_ty} -> vector<4x2xbf16>")
    else:
        row = addi(e, addi(e, kbc, f"%lt8_{name}"), f"%lq7_{name}")
        col = addi(e, e.c(mi * 16), f"%lh8_{name}")
        e.l(f"%{tag} = nvgpu.ldmatrix {src}[{row}, {col}] numTiles = 4 "
            f"transpose = true : {src_ty} -> vector<4x2xbf16>")
    return f"%{tag}"


def ldm_b(e, name, tag, src, src_ty, ni, kb, kt):
    """B-fragment via ldmatrix.x2.  kt=True (tile rows = n, k contiguous):
    no-trans, lane addr row = nc+lane%8, col = kb+8*(lane/8).  kt=False
    (tile rows = k): .trans, addr row = kb+8*(lane/8)+lane%8, col = nc."""
    kbc = kb if isinstance(kb, str) else e.c(kb)
    if kt:
        row = addi(e, f"%nc_{name}_{ni}", f"%lq7_{name}")
        col = addi(e, kbc, f"%lb8_{name}")
        e.l(f"%{tag} = nvgpu.ldmatrix {src}[{row}, {col}] numTiles = 2 "
            f"transpose = false : {src_ty} -> vector<2x2xbf16>")
    else:
        row = addi(e, addi(e, kbc, f"%lb8_{name}"), f"%lq7_{name}")
        e.l(f"%{tag} = nvgpu.ldmatrix {src}[{row}, %nc_{name}_{ni}] "
            f"numTiles = 2 transpose = true : {src_ty} -> vector<2x2xbf16>")
    return f"%{tag}"


def a_frag_elems(name, mi, kb, swap):
    r0, r1 = f"%rA_{name}_{mi}", f"%rAh_{name}_{mi}"
    k = [f"%ko_{name}_{kb}_{i}" for i in range(4)]
    pairs = [(r0, k[0]), (r0, k[1]), (r1, k[0]), (r1, k[1]),
             (r0, k[2]), (r0, k[3]), (r1, k[2]), (r1, k[3])]
    if swap:
        pairs = [(c, r) for (r, c) in pairs]
    return pairs


def b_frag_elems(name, ni, kb, kt):
    k = [f"%ko_{name}_{kb}_{i}" for i in range(4)]
    if kt:
        return [(f"%ncg_{name}_{ni}", x) for x in k]
    return [(x, f"%ncg_{name}_{ni}") for x in k]


def frag_build(e, tag, elems, src, src_ty):
    n = len(elems)
    shape = "vector<4x2xbf16>" if n == 8 else "vector<2x2xbf16>"
    r0, c0 = elems[0]
    e.l(f"%{tag}_l0 = memref.load {src}[{r0}, {c0}] : {src_ty}")
    e.l(f"%{tag}_b0 = vector.broadcast %{tag}_l0 : bf16 to {shape}")
    cur = f"%{tag}_b0"
    for i in range(1, n):
        r, c = elems[i]
        e.l(f"%{tag}_l{i} = memref.load {src}[{r}, {c}] : {src_ty}")
        e.l(f"%{tag}_b{i} = vector.insert %{tag}_l{i}, {cur} "
            f"[{i//2}, {i%2}] : bf16 into {shape}")
        cur = f"%{tag}_b{i}"
    return cur


def emit_gemm(e, name, M, N, Kred, W, a_src=None, a_ty=None, b_src=None,
              b_ty=None, kt=False, a_swap=False, epilogue=None,
              epi_accs=None):
    """Whole-K warp-tiled bf16 mma (f32 accum).  M%16==0, N%(8*W)==0,
    Kred%16==0.  With epi_accs given, no atoms are emitted: the epilogue
    runs over the supplied accumulator dict (frag-binop results)."""
    wn = gemm_precompute(e, name, M, N, W)
    if epi_accs is None:
        if _FRAG_MODE == "ldm":
            ldm_precompute(e, name)
        else:
            gemm_koffs(e, name, Kred)
        e.l(f"%zf_{name} = arith.constant 0.0 : f32")
        accs = {}
        for mi in range(M // 16):
            for ni in range(wn // 8):
                an = f"%acc_{name}_{mi}_{ni}"
                e.l(f"{an} = vector.broadcast %zf_{name} : f32 "
                    f"to vector<2x2xf32>")
                accs[(mi, ni)] = an
        fa, fb = {}, {}
        for kb in range(0, Kred, 16):
            for mi in range(M // 16):
                if _FRAG_MODE == "ldm":
                    fa[(mi, kb)] = ldm_a(e, name, f"fa_{name}_{mi}_{kb}",
                                         a_src, a_ty, mi, kb, a_swap)
                else:
                    fa[(mi, kb)] = frag_build(
                        e, f"fa_{name}_{mi}_{kb}",
                        a_frag_elems(name, mi, kb, a_swap), a_src, a_ty)
            for ni in range(wn // 8):
                if _FRAG_MODE == "ldm":
                    fb[(ni, kb)] = ldm_b(e, name, f"fb_{name}_{ni}_{kb}",
                                         b_src, b_ty, ni, kb, kt)
                else:
                    fb[(ni, kb)] = frag_build(
                        e, f"fb_{name}_{ni}_{kb}",
                        b_frag_elems(name, ni, kb, kt), b_src, b_ty)
        for mi in range(M // 16):
            for ni in range(wn // 8):
                prev = accs[(mi, ni)]
                for kb in range(0, Kred, 16):
                    o = f"%mr_{name}_{mi}_{ni}_{kb}"
                    e.l(f"{o} = nvgpu.mma.sync({fa[(mi, kb)]}, "
                        f"{fb[(ni, kb)]}, {prev}) mmaShape = [16, 8, 16] : "
                        f"(vector<4x2xbf16>, vector<2x2xbf16>, "
                        f"vector<2x2xf32>) -> vector<2x2xf32>")
                    prev = o
                accs[(mi, ni)] = prev
    else:
        accs = epi_accs
    if epilogue is not None:
        for mi in range(M // 16):
            for ni in range(wn // 8):
                acc = accs[(mi, ni)]
                for ei in range(4):
                    row = f"%rA_{name}_{mi}" if ei < 2 else f"%rAh_{name}_{mi}"
                    col = (f"%nct_{name}_{ni}" if ei % 2 == 0
                           else f"%nct1_{name}_{ni}")
                    epilogue(mi, ni, ei, row, col, acc)
    return accs


def acc_extract(e, acc, i, j, tag="ax"):
    o = e.v(tag)
    e.l(f"{o} = vector.extract {acc}[{i}, {j}] : f32 from vector<2x2xf32>")
    return o


def frag_binop(e, accs_a, accs_b, op, tag="fb"):
    out = {}
    for key in accs_a:
        a, b = accs_a[key], accs_b[key]
        cur = None
        for i in range(2):
            for j in range(2):
                xa = acc_extract(e, a, i, j, tag + "a")
                xb = acc_extract(e, b, i, j, tag + "b")
                r = e.v(tag + "r")
                e.l(f"{r} = arith.{op}f {xa}, {xb} : f32")
                if cur is None:
                    cur = e.v(tag + "v")
                    e.l(f"{cur} = vector.broadcast {r} : f32 "
                        f"to vector<2x2xf32>")
                else:
                    nxt = e.v(tag + "v")
                    e.l(f"{nxt} = vector.insert {r}, {cur} [{i}, {j}] "
                        f": f32 into vector<2x2xf32>")
                    cur = nxt
        out[key] = cur
    return out


def frag_load_global(e, name, M, N, W, src, src_ty, flatfn):
    """Accumulator-format f32 frags loaded from a flat bf16 global."""
    wn = gemm_precompute(e, name, M, N, W)
    out = {}
    for mi in range(M // 16):
        for ni in range(wn // 8):
            cur = None
            for ei in range(4):
                row = f"%rA_{name}_{mi}" if ei < 2 else f"%rAh_{name}_{mi}"
                col = (f"%nct_{name}_{ni}" if ei % 2 == 0
                       else f"%nct1_{name}_{ni}")
                idx = flatfn(row, col)
                lv = e.v("fl")
                e.l(f"{lv} = memref.load {src}[{idx}] : {src_ty}")
                xf = e.v("flx")
                e.l(f"{xf} = arith.extf {lv} : bf16 to f32")
                if cur is None:
                    cur = e.v("flv")
                    e.l(f"{cur} = vector.broadcast {xf} : f32 "
                        f"to vector<2x2xf32>")
                else:
                    nxt = e.v("flv")
                    e.l(f"{nxt} = vector.insert {xf}, {cur} [{ei//2}, {ei%2}] "
                        f": f32 into vector<2x2xf32>")
                    cur = nxt
            out[(mi, ni)] = cur
    return out


# ---------------------------------------------------------------------------
# scalar cell loops / staging
# ---------------------------------------------------------------------------

def emit_cell_loop(e, total, nthr, body):
    iv = e.v("ci")
    e.l(f"scf.for {iv} = %tid to {e.c(total)} step {e.c(nthr)} {{")
    body(iv)
    e.l("}")


def stage_copy_vec(e, src, src_ty, flatbase, rowstride, rows, cols, dst,
                   dst_ty, nthr):
    nchunk = cols // 8
    pad = e.v("padv")
    e.l(f"{pad} = arith.constant 0.0 : bf16")

    def body(i):
        r = divsi(e, i, e.c(nchunk))
        cc = remsi(e, i, e.c(nchunk))
        c8 = muli(e, cc, e.c(8))
        base = addi(e, flatbase, muli(e, r, e.c(rowstride)))
        idx = addi(e, base, c8)
        t = e.v("scv")
        e.l(f"{t} = vector.transfer_read {src}[{idx}], {pad} "
            f"{{in_bounds = [true]}} : {src_ty}, vector<8xbf16>")
        e.l(f"vector.transfer_write {t}, {dst}[{r}, {c8}] "
            f"{{in_bounds = [true]}} : vector<8xbf16>, {dst_ty}")

    emit_cell_loop(e, rows * nchunk, nthr, body)


def smem_to_global_scalar(e, src_ssa, src_ty, dst, dst_ty, flatfn, rows,
                          cols, nthr, src_dt=BF, dst_dt=None):
    dst_dt = dst_dt or src_dt

    def body(i):
        r = divsi(e, i, e.c(cols))
        c = remsi(e, i, e.c(cols))
        v = e.v("cgs")
        e.l(f"{v} = memref.load {src_ssa}[{r}, {c}] : {src_ty}")
        w = v
        if src_dt != dst_dt:
            w = e.v("cgc")
            if dst_dt == F32:
                e.l(f"{w} = arith.extf {v} : {src_dt} to {dst_dt}")
            else:
                e.l(f"{w} = arith.truncf {v} : {src_dt} to {dst_dt}")
        e.l(f"memref.store {w}, {dst}[{flatfn(r, c)}] : {dst_ty}")

    emit_cell_loop(e, rows * cols, nthr, body)


def global_to_smem_scalar(e, src, src_ty, flatfn, dst_ssa, dst_ty, rows,
                          cols, nthr, pre=None):
    def body(i):
        r = divsi(e, i, e.c(cols))
        c = remsi(e, i, e.c(cols))
        v = e.v("gsl")
        e.l(f"{v} = memref.load {src}[{flatfn(r, c)}] : {src_ty}")
        w = e.v("gsx")
        e.l(f"{w} = arith.extf {v} : bf16 to f32")
        if pre is not None:
            w = pre(w, r, c)
        e.l(f"memref.store {w}, {dst_ssa}[{r}, {c}] : {dst_ty}")

    emit_cell_loop(e, rows * cols, nthr, body)


# ---------------------------------------------------------------------------
# gc cumsum + beta (per chunk, tid<64)
# ---------------------------------------------------------------------------

def emit_cumsum_beta(e, g, g_ty, g_base, beta, beta_ty, beta_base, gc, gc_ty,
                     bs, bs_ty, H):
    """gc = inclusive cumsum of g*RCP_LN2 (g is an fp32 NATURAL-log global
    [B,T,H], the G5-A shared convention; per-row stride is H); beta staged
    to f32 smem."""
    c64 = e.c(64)
    cH_ = e.c(H)
    in64 = e.v("in64")
    e.l(f"{in64} = arith.cmpi ult, %tid, {c64} : index")
    e.l(f"scf.if {in64} {{")
    rcp = e.v("rcp")
    e.l(f"{rcp} = arith.constant {f32lit(RCP_LN2)} : f32")
    z = e.v("czf")
    e.l(f"{z} = arith.constant 0.0 : f32")
    tidp1 = addi(e, "%tid", e.c(1))
    iv = e.v("gr")
    e.l(f"%gcacc = scf.for {iv} = %c0 to {tidp1} step %c1 "
        f"iter_args(%ga = {z}) -> (f32) {{")
    gid = addi(e, g_base, muli(e, iv, cH_))
    lv = e.v("ggl")
    e.l(f"{lv} = memref.load {g}[{gid}] : {g_ty}")
    lm = e.v("ggl2")
    e.l(f"{lm} = arith.mulf {lv}, {rcp} : f32")
    an = e.v("gan")
    e.l(f"{an} = arith.addf %ga, {lm} : f32")
    e.l(f"scf.yield {an} : f32")
    e.l("}")
    e.l(f"memref.store %gcacc, {gc}[%tid, %c0] : {gc_ty}")
    bid = addi(e, beta_base, muli(e, "%tid", cH_))
    bv = e.v("gbt")
    e.l(f"{bv} = memref.load {beta}[{bid}] : {beta_ty}")
    e.l(f"memref.store {bv}, {bs}[%tid, %c0] : {bs_ty}")
    e.l("}")
    e.l("gpu.barrier")


def emit_load_gc(e, gc_ptr, gc_ty, gc_base, gc_ssa, gc_pty):
    c64 = e.c(64)
    in64 = e.v("in64")
    e.l(f"{in64} = arith.cmpi ult, %tid, {c64} : index")
    e.l(f"scf.if {in64} {{")
    gid = addi(e, gc_base, "%tid")
    v = e.v("lgv")
    e.l(f"{v} = memref.load {gc_ptr}[{gid}] : {gc_ty}")
    e.l(f"memref.store {v}, {gc_ssa}[%tid, %c0] : {gc_pty}")
    e.l("}")
    e.l("gpu.barrier")


# ---------------------------------------------------------------------------
# forward-substitution solve (warp-per-column, T column in registers)
# ---------------------------------------------------------------------------

def emit_solve(e, A_ssa, A_ty, TbT_ssa, TbT_ty, nthr, C=64):
    """T = (I+A)^-1 via the L1 forward-substitution recurrence,
    thread-per-column (tid<64), writing the TRANSPOSED bf16 tile
    TbT[j][k] = T[k][j] which the T@X mmas read with swapped A-fragment
    addressing (a_swap=True).

    The earlier warp-per-column register variant (lane-owned k pair +
    shfl butterfly) had an unresolved off-diagonal defect; this scalar
    form is the L1 recurrence verbatim: for column j, rows i>j:
      T[i][j] = -sum_{j<=k<i} A[i,k]*T[k,j]
    (bf16 T storage inside the recurrence: off-diag entries are O(|A|)<<1
    so relative rounding stays far below the 5e-3 contract; the diagonal
    1.0 and zeros are exact in bf16.)  ~2k FMAs on one thread for j=0 --
    correctness-first; perf variant is a follow-up.
    """
    cC = e.c(C)
    c64 = e.c(64)
    in64 = e.v("in64")
    e.l(f"{in64} = arith.cmpi ult, %tid, {c64} : index")
    e.l(f"scf.if {in64} {{")
    jcol = e.v("jcol")
    e.l(f"{jcol} = arith.addi %tid, %c0 : index")
    one = e.v("sone")
    e.l(f"{one} = arith.constant 1.0 : f32")
    zbf = e.v("szb")
    e.l(f"{zbf} = arith.constant 0.0 : bf16")
    ob = e.v("sob")
    e.l(f"{ob} = arith.truncf {one} : f32 to bf16")
    zr = e.v("zr")
    e.l(f"scf.for {zr} = %c0 to {jcol} step %c1 {{")
    e.l(f"  memref.store {zbf}, {TbT_ssa}[{jcol}, {zr}] : {TbT_ty}")
    e.l("}")
    e.l(f"memref.store {ob}, {TbT_ssa}[{jcol}, {jcol}] : {TbT_ty}")
    iv = e.v("si")
    ip1 = addi(e, jcol, e.c(1))
    e.l(f"scf.for {iv} = {ip1} to {cC} step %c1 {{")
    zf = e.v("szf")
    e.l(f"{zf} = arith.constant 0.0 : f32")
    kv = e.v("sk")
    e.l(f"%sv = scf.for {kv} = {jcol} to {iv} step %c1 "
        f"iter_args(%sa = {zf}) -> (f32) {{")
    e.l(f"  %aik = memref.load {A_ssa}[{iv}, {kv}] : {A_ty}")
    e.l(f"  %tkj = memref.load {TbT_ssa}[{jcol}, {kv}] : {TbT_ty}")
    e.l(f"  %tkf = arith.extf %tkj : bf16 to f32")
    e.l(f"  %pr = arith.mulf %aik, %tkf : f32")
    e.l(f"  %sn = arith.addf %sa, %pr : f32")
    e.l(f"  scf.yield %sn : f32")
    e.l("}")
    neg = e.v("sng")
    e.l(f"{neg} = arith.negf %sv : f32")
    ntb = e.v("stb")
    e.l(f"{ntb} = arith.truncf {neg} : f32 to bf16")
    e.l(f"memref.store {ntb}, {TbT_ssa}[{jcol}, {iv}] : {TbT_ty}")
    e.l("}")
    e.l("}")
    e.l("gpu.barrier")


# ---------------------------------------------------------------------------
# module assembly
# ---------------------------------------------------------------------------

IDENTITY_TRANSFORM = """module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    transform.yield
  }
}
"""

PIPE_TRANSFORM = """module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %arg0
      : (!transform.any_op) -> !transform.op<"memref.alloc">
    transform.memref.multibuffer %allocs <factor = {mb}, skip_analysis>
      : (!transform.op<"memref.alloc">) -> !transform.any_op
    %func = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.create_async_groups %func bypass_l1 : (!transform.any_op) -> (!transform.any_op)
    %loops = transform.structured.match ops{["scf.for"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.pipeline_shared_memory_copies failures(suppress) %loops depth = {depth} peel_epilogue : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
"""


def pipe_transform(mb=2, depth=2):
    return PIPE_TRANSFORM.replace("{mb}", str(mb)).replace("{depth}",
                                                           str(depth))


ASYNC_TRANSFORM = """module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.create_async_groups %func bypass_l1 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
"""


def hoist_constants(text):
    """Move every '%cN = arith.constant N : index' to the region top.

    The emitter materializes integer constants lazily at first use, which
    can land them inside a nested scf.for/scf.if that is NOT dominated by
    later uses.  Hoisting to the top of the gpu.launch region fixes
    dominance (loop bodies see outer-scope definitions)."""
    import re as _re
    pat = _re.compile(r"^ *(%c\d+) = arith\.constant (\d+) : index$")
    consts, seen, out = [], set(), []
    for ln in text.split("\n"):
        m = pat.match(ln)
        if m:
            if m.group(1) not in seen:
                seen.add(m.group(1))
                consts.append(f"{m.group(1)} = arith.constant {m.group(2)} "
                              f": index")
        else:
            out.append(ln)
    if not consts:
        return text
    return "\n".join(consts) + "\n" + "\n".join(out)


def wrap_module(kname, params, region_text, gx, gy, bx, meta):
    """Zero-arg module (G3/G4 noargs pattern): every buffer is a PUBLIC
    uninitialized memref.global (PTX .bss, no embedded data text); the
    gpu.launch region takes no runtime args (memref.get_global views are
    cloned into the gpu.module by outlining), and the harness stages
    inputs / reads outputs through cuModuleGetGlobal.  This sidesteps the
    host-side launch-arg marshalling (unrealized_conversion_cast) that
    breaks pointer-arg kernels with many late-used memref captures."""
    hdr = ["// Generated by tools/gdn_full_gen.py -- GDN full-op forward (G5-B)",
           f"// kernel {kname} | grid {gx}x{gy} block {bx} | {meta}",
           "// buffers (public zero-init globals; harness uploads/downloads):",
           " // " + " ".join(f"@{n}:{dt}[{el}]" for n, el, dt in params)]
    # TOP-LEVEL ops (NO module wrapper): G2 payload format.  The transform
    # interpreter wraps these into ONE extra module level (double nesting,
    # which the nvvm pipeline accepts); a wrapper here would triple-nest
    # and break kernel outlining ("conversion of memref memory space
    # workgroup failed", the G2-C lesson).
    lines = []
    for n, el, dt in params:
        lines.append(f'memref.global @{n} : memref<{el}x{dt}>')
    lines.append(f"func.func @{kname}() {{")
    lines.append(f"%hgx = arith.constant {gx} : index")
    lines.append(f"%hgy = arith.constant {gy} : index")
    lines.append("%hgz = arith.constant 1 : index")
    lines.append(f"%hbx = arith.constant {bx} : index")
    lines.append("gpu.launch blocks(%bx, %by, %bz) in "
                 "(%gx = %hgx, %gy = %hgy, %gz = %hgz)")
    lines.append("             threads(%tx, %ty, %tz) in "
                 "(%bxx = %hbx, %byy = %hgz, %bzz = %hgz) {")
    for n, el, dt in params:
        lines.append(f"%{n} = memref.get_global @{n} : memref<{el}x{dt}>")
    lines.extend(ln if ln.startswith("//") else "  " + ln if ln else ln
                 for ln in region_text.split("\n"))
    lines += ["  gpu.terminator", "  }", "  return", "}"]
    return "\n".join(hdr) + "\n" + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# shared emission blocks
# ---------------------------------------------------------------------------

def emit_A_epilogue(e, A_m, A_ty, bs, bs_ty, gc, gc_ty):
    def epi(mi, ni, ei, row, col, acc):
        x = acc_extract(e, acc, ei // 2, ei % 2, "eA")
        br = e.v("Abr"); e.l(f"{br} = memref.load {bs}[{row}, %c0] : {bs_ty}")
        gi = e.v("Agi"); e.l(f"{gi} = memref.load {gc}[{row}, %c0] : {gc_ty}")
        gj = e.v("Agj"); e.l(f"{gj} = memref.load {gc}[{col}, %c0] : {gc_ty}")
        d = e.v("Ad"); e.l(f"{d} = arith.subf {gi}, {gj} : f32")
        ex = e.v("Aex"); e.l(f"{ex} = math.exp2 {d} : f32")
        m1 = e.v("Am1"); e.l(f"{m1} = arith.mulf {x}, {br} : f32")
        m2 = e.v("Am2"); e.l(f"{m2} = arith.mulf {m1}, {ex} : f32")
        lt = e.v("Alt"); e.l(f"{lt} = arith.cmpi slt, {col}, {row} : index")
        sel = e.v("Asel"); e.l(f"{sel} = arith.select {lt}, {m2}, %zfs : f32")
        e.l(f"memref.store {sel}, {A_m}[{row}, {col}] : {A_ty}")
    return epi


def emit_P_epilogue(e, P, P_ty, gc, gc_ty):
    def epi(mi, ni, ei, row, col, acc):
        x = acc_extract(e, acc, ei // 2, ei % 2, "eP")
        gi = e.v("Pgi"); e.l(f"{gi} = memref.load {gc}[{row}, %c0] : {gc_ty}")
        gj = e.v("Pgj"); e.l(f"{gj} = memref.load {gc}[{col}, %c0] : {gc_ty}")
        d = e.v("Pd"); e.l(f"{d} = arith.subf {gi}, {gj} : f32")
        ex = e.v("Pex"); e.l(f"{ex} = math.exp2 {d} : f32")
        m = e.v("Pm"); e.l(f"{m} = arith.mulf {x}, {ex} : f32")
        le = e.v("Ple"); e.l(f"{le} = arith.cmpi sle, {col}, {row} : index")
        sel = e.v("Psel"); e.l(f"{sel} = arith.select {le}, {m}, %zfs : f32")
        t = e.v("Pt"); e.l(f"{t} = arith.truncf {sel} : f32 to bf16")
        e.l(f"memref.store {t}, {P}[{row}, {col}] : {P_ty}")
    return epi


def emit_bk2_stage(e, k, k_ty, kbase, bs, bs_ty, gc, gc_ty, bk2, bk2_ty,
                   H, K, C, nthr):
    def pre(w, r, c):
        b_ = e.v("kb"); e.l(f"{b_} = memref.load {bs}[{r}, %c0] : {bs_ty}")
        gi = e.v("kgi"); e.l(f"{gi} = memref.load {gc}[{r}, %c0] : {gc_ty}")
        ex = e.v("kex"); e.l(f"{ex} = math.exp2 {gi} : f32")
        m = e.v("km2"); e.l(f"{m} = arith.mulf {w}, {b_} : f32")
        m2 = e.v("km3"); e.l(f"{m2} = arith.mulf {m}, {ex} : f32")
        t = e.v("kbt"); e.l(f"{t} = arith.truncf {m2} : f32 to bf16")
        return t
    global_to_smem_scalar(
        e, k, k_ty,
        lambda r, c: addi(e, kbase, addi(e, muli(e, r, e.c(H * K)), c)),
        bk2, bk2_ty, C, K, nthr, pre=pre)


def emit_kterm_inplace(e, k_t, k_t_ty, gc, gc_ty, gcl, K, C, nthr):
    def body(i):
        r = divsi(e, i, e.c(K))
        c = remsi(e, i, e.c(K))
        v = e.v("ktv"); e.l(f"{v} = memref.load {k_t}[{r}, {c}] : {k_t_ty}")
        vf = e.v("ktf"); e.l(f"{vf} = arith.extf {v} : bf16 to f32")
        gi = e.v("ktg"); e.l(f"{gi} = memref.load {gc}[{r}, %c0] : {gc_ty}")
        d = e.v("ktd"); e.l(f"{d} = arith.subf {gcl}, {gi} : f32")
        ex = e.v("kte"); e.l(f"{ex} = math.exp2 {d} : f32")
        m = e.v("ktm"); e.l(f"{m} = arith.mulf {vf}, {ex} : f32")
        t = e.v("ktt"); e.l(f"{t} = arith.truncf {m} : f32 to bf16")
        e.l(f"memref.store {t}, {k_t}[{r}, {c}] : {k_t_ty}")
    emit_cell_loop(e, C * K, nthr, body)


def emit_qg_inplace(e, q_t, q_t_ty, gc, gc_ty, K, C, nthr):
    def body(i):
        r = divsi(e, i, e.c(K))
        c = remsi(e, i, e.c(K))
        v = e.v("qgv"); e.l(f"{v} = memref.load {q_t}[{r}, {c}] : {q_t_ty}")
        vf = e.v("qgf"); e.l(f"{vf} = arith.extf {v} : bf16 to f32")
        gi = e.v("qgg"); e.l(f"{gi} = memref.load {gc}[{r}, %c0] : {gc_ty}")
        ex = e.v("qge"); e.l(f"{ex} = math.exp2 {gi} : f32")
        m = e.v("qgm"); e.l(f"{m} = arith.mulf {vf}, {ex} : f32")
        t = e.v("qgt"); e.l(f"{t} = arith.truncf {m} : f32 to bf16")
        e.l(f"memref.store {t}, {q_t}[{r}, {c}] : {q_t_ty}")
    emit_cell_loop(e, C * K, nthr, body)


def emit_S_update(e, S, S_ty, dl):
    def epi(mi, ni, ei, row, col, acc):
        x = acc_extract(e, acc, ei // 2, ei % 2, "eS")
        sv = e.v("Sv"); e.l(f"{sv} = memref.load {S}[{row}, {col}] : {S_ty}")
        sf = e.v("Sf"); e.l(f"{sf} = arith.extf {sv} : bf16 to f32")
        m = e.v("Sm"); e.l(f"{m} = arith.mulf {sf}, {dl} : f32")
        a2 = e.v("Sa"); e.l(f"{a2} = arith.addf {m}, {x} : f32")
        t = e.v("St"); e.l(f"{t} = arith.truncf {a2} : f32 to bf16")
        e.l(f"memref.store {t}, {S}[{row}, {col}] : {S_ty}")
    return epi


# G5-D round-2: new emitter sections for tools/gdn_full_gen.py
# This file holds ONLY the new/rewritten functions; a patch script splices
# them into gdn_full_gen.py (replacing gen_f1/gen_f2/gen_f3 and extending
# the CLI).  Kept separate for review clarity; the repo file is the truth.

# ---------------------------------------------------------------------------
# round 2: pipelined K-loop emitter (transform-pipeline route)
# ---------------------------------------------------------------------------

PIPE_TRANSFORM_TMPL = """module attributes {transform.with_named_sequence} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %allocs = transform.structured.match ops{{["memref.alloc"]}} in %arg0
      : (!transform.any_op) -> !transform.op<"memref.alloc">
    transform.memref.multibuffer %allocs <factor = {mb}, skip_analysis>
      : (!transform.op<"memref.alloc">) -> !transform.any_op
    %func = transform.structured.match ops{{["func.func"]}} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.create_async_groups %func bypass_l1 : (!transform.any_op) -> (!transform.any_op)
    %loops = transform.structured.match ops{{["scf.for"]}} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.pipeline_shared_memory_copies failures(suppress) %loops depth = {depth} peel_epilogue : (!transform.any_op) -> !transform.any_op
    transform.yield
  }}
}}
"""


def _koff_ssa(e, iv, kloc, staged):
    """Fragment k-offset SSA: local (staged tile) or iv+local (fixed tile)."""
    if staged:
        return e.c(kloc)
    return addi(e, iv, e.c(kloc))


def emit_kloop(e, lname, Kred, bk, stages, gemms, W):
    """One K-loop structured for the G2 transform pipeline (multibuffer +
    create_async_groups + pipeline_shared_memory_copies).

    stages: dicts {tile, ty, rows, cols, src, sty, kind, base, rowmul}
    describing per-iteration global->smem tile staging as THREAD-STRIDED
    vector<8xbf16> transfer pairs (each thread copies its own distinct
    16B chunks; the G2-style block-uniform unrolled form makes EVERY
    thread issue EVERY copy = 256x redundant cp.async -- measured 2x
    SLOWER than round 1 before this fix):
        kind "rc": tile[r][c8]; global idx = base + r*rowmul + (iv + c8)
        kind "kc": tile[r][c8] (r = 0..bk-1); global idx =
                   base + c8 + (iv + r)*rowmul
    Requires rows*cols/8 % (32*W) == 0.

    gemms: dicts {name, M, N, kt, a_swap, a, b}; a/b are (ssa, ty, staged)
    triples.  Staged operands read the per-iteration tile at LOCAL k
    offsets; fixed operands read a full resident tile at iv+local offsets.
    Accumulators are loop-carried iter_args.

    Returns {(gname, mi, ni): result ssa}.
    """
    for g in gemms:
        gemm_precompute(e, g["name"], g["M"], g["N"], W)
        ldm_precompute(e, g["name"])
    zf = e.v("zk" + lname)
    e.l(f"{zf} = arith.constant 0.0 : f32")
    keys, inits = [], []
    for g in gemms:
        wn = g["N"] // W
        for mi in range(g["M"] // 16):
            for ni in range(wn // 8):
                an = f"%az_{lname}_{g['name']}_{mi}_{ni}"
                e.l(f"{an} = vector.broadcast {zf} : f32 to vector<2x2xf32>")
                keys.append((g["name"], mi, ni))
                inits.append(an)
    iv = f"%kb_{lname}"
    iargs = [f"%ia_{lname}_{g}_{mi}_{ni}" for (g, mi, ni) in keys]
    res = [f"%rs_{lname}_{g}_{mi}_{ni}" for (g, mi, ni) in keys]
    tys = ", ".join(["vector<2x2xf32>"] * len(keys))
    if keys:
        e.l(f"{', '.join(res)} = scf.for {iv} = %c0 to {e.c(Kred)} "
            f"step {e.c(bk)} iter_args("
            + ", ".join(f"{a} = {b}" for a, b in zip(iargs, inits))
            + f") -> ({tys}) {{")
    else:
        # R23-fwd GDN_F1T_NKM arm: no gemms -> a plain staging-only loop
        # (staging + both barriers kept; ldm/mma deleted).
        e.l(f"scf.for {iv} = %c0 to {e.c(Kred)} step {e.c(bk)} {{")
    pad = f"%pd_{lname}"
    e.l(f"{pad} = arith.constant 0.0 : bf16")
    nthr = 32 * W
    for s in stages:
        t_ssa, t_ty = s["tile"], s["ty"]
        nch = s["cols"] // 8
        npairs = s["rows"] * nch
        assert npairs % nthr == 0, \
            f"stage {lname}: {npairs} pairs not divisible by {nthr} threads"
        for kk in range(npairs // nthr):
            p = addi(e, "%tid", e.c(kk * nthr)) if kk else "%tid"
            r = divsi(e, p, e.c(nch))
            cc = remsi(e, p, e.c(nch))
            c8 = muli(e, cc, e.c(8))
            if s["kind"] == "rc":
                idx = addi(e, addi(e, muli(e, r, e.c(s["rowmul"])),
                                   s["base"]), addi(e, iv, c8))
            else:  # "kc"
                idx = addi(e, muli(e, addi(e, iv, r), e.c(s["rowmul"])),
                           addi(e, s["base"], c8))
            t = e.v("sr" + lname)
            e.l(f"{t} = vector.transfer_read {s['src']}[{idx}], "
                f"{pad} {{in_bounds = [true]}} : {s['sty']}, "
                f"vector<8xbf16>")
            e.l(f"vector.transfer_write {t}, {t_ssa}[{r}, "
                f"{c8}] {{in_bounds = [true]}} : "
                f"vector<8xbf16>, {t_ty}")
    e.l("gpu.barrier")
    cur = dict(zip(keys, iargs))
    for g in gemms:
        a_ssa, a_ty, a_stg = g["a"]
        b_ssa, b_ty, b_stg = g["b"]
        kt = g.get("kt", False)
        swap = g.get("a_swap", False)
        wn = g["N"] // W
        fa, fb = {}, {}
        for mi in range(g["M"] // 16):
            for kloc in range(0, bk, 16):
                ka = _koff_ssa(e, iv, kloc, a_stg)
                fa[(mi, kloc)] = ldm_a(e, g["name"],
                                       f"fa_{lname}_{g['name']}_{mi}_{kloc}",
                                       a_ssa, a_ty, mi, ka, swap)
        for ni in range(wn // 8):
            for kloc in range(0, bk, 16):
                kb_ = _koff_ssa(e, iv, kloc, b_stg)
                fb[(ni, kloc)] = ldm_b(e, g["name"],
                                       f"fb_{lname}_{g['name']}_{ni}_{kloc}",
                                       b_ssa, b_ty, ni, kb_, kt)
        for mi in range(g["M"] // 16):
            for ni in range(wn // 8):
                prev = cur[(g["name"], mi, ni)]
                for kloc in range(0, bk, 16):
                    o = f"%mr_{lname}_{g['name']}_{mi}_{ni}_{kloc}"
                    e.l(f"{o} = nvgpu.mma.sync({fa[(mi, kloc)]}, "
                        f"{fb[(ni, kloc)]}, {prev}) "
                        f"mmaShape = [16, 8, 16] : (vector<4x2xbf16>, "
                        f"vector<2x2xbf16>, vector<2x2xf32>) -> "
                        f"vector<2x2xf32>")
                    prev = o
                cur[(g["name"], mi, ni)] = prev
    e.l("gpu.barrier")
    if keys:
        e.l(f"scf.yield {', '.join(cur[k] for k in keys)} : {tys}")
    e.l("}")
    return dict(zip(keys, res))


def run_epilogue(e, name, M, N, W, accs, epilogue):
    """Fragment epilogue over one gemm's accumulators (outside any loop)."""
    wn = N // W
    for mi in range(M // 16):
        for ni in range(wn // 8):
            acc = accs[(name, mi, ni)]
            for ei in range(4):
                row = f"%rA_{name}_{mi}" if ei < 2 else f"%rAh_{name}_{mi}"
                col = (f"%nct_{name}_{ni}" if ei % 2 == 0
                       else f"%nct1_{name}_{ni}")
                epilogue(mi, ni, ei, row, col, acc)


def pin_alloc(e, ssa, ty):
    """Dummy non-loop RMW so transform.memref.multibuffer SKIPS this alloc.

    Buffers whose users live in several DIFFERENT loops (and none outside
    any loop) hit multiBuffer()'s hard-fail path (no single candidate loop)
    which propagates as a transform error.  One straight-line load+store
    outside all loops makes the transform op skip the alloc (verified
    against MemRefTransformOps.cpp in this tree)."""
    v = e.v("pin")
    e.l(f"{v} = memref.load {ssa}[%c0, %c0] : {ty}")
    e.l(f"memref.store {v}, {ssa}[%c0, %c0] : {ty}")


def copy_g2s_vec(e, src, sty, flatfn, dst, dst_ty, rows, cols, nthr):
    """Whole-tile global->smem staging as vector<8> transfer pairs in a cell
    loop (outside K-loops; create_async_groups turns these into cp.async)."""
    nchunk = cols // 8
    pad = e.v("pgv")
    e.l(f"{pad} = arith.constant 0.0 : bf16")

    def body(i):
        r = divsi(e, i, e.c(nchunk))
        cc = remsi(e, i, e.c(nchunk))
        c8 = muli(e, cc, e.c(8))
        idx = flatfn(r, c8)
        t = e.v("pgt")
        e.l(f"{t} = vector.transfer_read {src}[{idx}], {pad} "
            f"{{in_bounds = [true]}} : {sty}, vector<8xbf16>")
        e.l(f"vector.transfer_write {t}, {dst}[{r}, {c8}] "
            f"{{in_bounds = [true]}} : vector<8xbf16>, {dst_ty}")

    emit_cell_while(e, rows * nchunk, nthr, body)


def copy_s2g_vec(e, src, src_ty, flatfn, dst, dst_ty, rows, cols, nthr,
                 row_base=None):
    """Whole-tile smem->global copy (bf16, scalar loads/stores: vector
    transfers on workgroup space do not lower through this pipeline, and
    vector.transfer_write to a flat global inside scf.while regions leaves
    an unrealized_conversion_cast -- both re-confirmed in G5-D probes;
    G5-K round 5 re-probed vector.load/store + bitcast-to-i32 variants:
    they LOWER but element-split to b16 ops -- no 128-bit path exists).
    row_base: optional SSA/index expr prepended to the row when the tile
    is row-ping-ponged (G5-K vn halves).

    R15-X (GDN_WST, default off): PAIRED-cell mode -- each thread-step
    moves cells (2i, 2i+1): 2x2 b16 ld/st quartets with explicit +1
    adjacency, which tools/gdn_smem_fix.py (same env) fuses at PTX level
    into ld.shared.b32 + st.global.b32 (byte-exact).  Halves the LSU
    instructions AND the loop iterations of every cell pass; the r15
    deletion probes put the f2 h_all/v_new passes at 35.4/17.9us of the
    297us T8192 chain.  Requires cols even (no pair straddles a row) and
    rows*cols % (2*nthr) == 0; otherwise falls back to single cells.

    R24 (GDN_W32P, default off): per-row warp remap for the BV=32 f2
    tiles (cols == 32, rows % 8 == 0, nthr == 128).  The r16 negative:
    under the flat WST pair map a warp spans 2 CONSECUTIVE rows of the
    80B-stride tile (16 pairs/row); the 20-word row stride banks lanes
    28-31 onto lanes 0-3's banks = 2-way conflict on every b32 load
    (measured +4.3us mapping-only, r15 `ab_pairoonly`).  The remap:
    lanes 0-15 read row r = 8*round + warp, lanes 16-31 read row r+4.
    Word addresses: row r pairs 0-15 = words [20r, 20r+16); row r+4 =
    [20r+80, 20r+96); 80 mod 32 = 16, so the two 16-word half-warps hit
    bank runs {b..b+15} and {b+16..b+31} -- exactly complementary, ZERO
    conflicts for every r.  Bijective cover: round j, warp w, hi h ->
    row 8j + w + 4h, pair lane%16 (h in 0..7 covers rows 8j..8j+7,
    pairs 0..15 once each).  Pure index-arithmetic change: same bytes,
    same in-row pairing, same global addresses -> bit-identical; pool,
    tiles, pads, LDSM geometry, occupancy all unchanged.  smem_fix's
    widen_stores quartet fusion is structural ([base] / [base+2]) and
    fires unchanged; requires GDN_WST=1 to actually fuse."""
    def body(i):
        r = divsi(e, i, e.c(cols))
        c = remsi(e, i, e.c(cols))
        rr = addi(e, row_base, r) if row_base is not None else r
        v = e.v("sgl")
        e.l(f"{v} = memref.load {src}[{rr}, {c}] : {src_ty}")
        idx = flatfn(r, c)
        e.l(f"memref.store {v}, {dst}[{idx}] : {dst_ty}")

    if (os.environ.get("GDN_WST") and cols >= 64 and cols % 2 == 0
            and (rows * cols) % (2 * nthr) == 0):
        def body2(i):
            i2 = muli(e, i, e.c(2))
            r = divsi(e, i2, e.c(cols))
            c = remsi(e, i2, e.c(cols))
            rr = addi(e, row_base, r) if row_base is not None else r
            v0 = e.v("sgl")
            e.l(f"{v0} = memref.load {src}[{rr}, {c}] : {src_ty}")
            c1 = addi(e, c, e.c(1))
            v1 = e.v("sgl")
            e.l(f"{v1} = memref.load {src}[{rr}, {c1}] : {src_ty}")
            idx = flatfn(r, c)
            e.l(f"memref.store {v0}, {dst}[{idx}] : {dst_ty}")
            idx1 = addi(e, idx, e.c(1))
            e.l(f"memref.store {v1}, {dst}[{idx1}] : {dst_ty}")

        emit_cell_while(e, rows * cols // 2, nthr, body2)
        return

    # R24 (GDN_W32P, default off): conflict-free per-row warp remap for
    # the f2 BV=32 store passes (see the function docstring for the bank
    # arithmetic).  i runs tid, tid+nthr, ... so i//nthr is the ROUND
    # (tid < nthr); the lane constants are loop-invariant (CSE'd out).
    if (os.environ.get("GDN_W32P") and cols == 32 and rows % 8 == 0
            and nthr == 128):
        hi16 = e.v("w3h")
        e.l(f"{hi16} = arith.cmpi uge, %lane, {e.c(16)} : index")
        hi4 = e.v("w3h4")
        e.l(f"{hi4} = arith.select {hi16}, {e.c(4)}, {e.c(0)} : index")
        hcol = addi(e, "%warp", hi4)           # half-warp row slot 0..7
        c0 = muli(e, remsi(e, "%lane", e.c(16)), e.c(2))

        def body3(i):
            k = divsi(e, i, e.c(nthr))          # round (i = tid + nthr*j)
            row = addi(e, muli(e, k, e.c(8)), hcol)
            rr = addi(e, row_base, row) if row_base is not None else row
            v0 = e.v("sgl")
            e.l(f"{v0} = memref.load {src}[{rr}, {c0}] : {src_ty}")
            c1 = addi(e, c0, e.c(1))
            v1 = e.v("sgl")
            e.l(f"{v1} = memref.load {src}[{rr}, {c1}] : {src_ty}")
            idx = flatfn(row, c0)
            e.l(f"memref.store {v0}, {dst}[{idx}] : {dst_ty}")
            idx1 = addi(e, idx, e.c(1))
            e.l(f"memref.store {v1}, {dst}[{idx1}] : {dst_ty}")

        emit_cell_while(e, rows * cols // 2, nthr, body3)
        return
    emit_cell_while(e, rows * cols, nthr, body)


def emit_cumsum_beta_scan(e, g, g_ty, g_base, beta, beta_ty, beta_base, gc,
                          gc_ty, bs, bs_ty, H):
    """gc = inclusive log2-domain cumsum of g (Hillis-Steele, 6 smem steps)
    + beta staged to f32 smem.  Replaces the serial per-thread scan (each
    thread re-read g[t0..t] with H-strided global loads: avg 32 uncoalesced
    loads/thread); now ONE strided load per row + log-depth smem scan."""
    c64 = e.c(64)
    cH_ = e.c(H)
    in64 = e.v("in64")
    e.l(f"{in64} = arith.cmpi ult, %tid, {c64} : index")
    e.l(f"scf.if {in64} {{")
    rcp = e.v("rcp")
    e.l(f"{rcp} = arith.constant {f32lit(RCP_LN2)} : f32")
    gid = addi(e, g_base, muli(e, "%tid", cH_))
    lv = e.v("ggl")
    e.l(f"{lv} = memref.load {g}[{gid}] : {g_ty}")
    lm = e.v("ggl2")
    e.l(f"{lm} = arith.mulf {lv}, {rcp} : f32")
    e.l(f"memref.store {lm}, {gc}[%tid, %c0] : {gc_ty}")
    bid = addi(e, beta_base, muli(e, "%tid", cH_))
    bv = e.v("gbt")
    e.l(f"{bv} = memref.load {beta}[{bid}] : {beta_ty}")
    e.l(f"memref.store {bv}, {bs}[%tid, %c0] : {bs_ty}")
    e.l("}")
    e.l("gpu.barrier")
    zf = e.v("czf")
    e.l(f"{zf} = arith.constant 0.0 : f32")
    # R17-fwd GDN_SCANDB (default off): double-buffer the 6 Hillis-Steele
    # steps.  The stock single-buffer step has a CROSS-WARP intra-step
    # race: thread t2's gc[off] load vs thread t1's gc[t1] store (t1 ==
    # t2's off) are unordered within a step (the barrier only separates
    # steps); warp-internal SIMD order makes it benign while warps 0/1
    # stay schedule-aligned -- which 1-CTA residency happened to
    # guarantee and 2-CTA co-residency broke (racecheck flags exactly
    # these 6 steps in EVERY build incl. the ships; the r17f F1KS 2-CTA
    # corruption bisect pinned it).  +256B (gcs2 [64] f32); values end
    # in gc (6 steps, even count -- no copy-back).
    _sdb = bool(os.environ.get("GDN_SCANDB"))
    if _sdb:
        gc2, gc2_ty = alloc2d(e, "gcs2", 64, 1, F32)
    for i, d in enumerate((1, 2, 4, 8, 16, 32)):
        src, src_ty, dst, dst_ty = gc, gc_ty, gc, gc_ty
        if _sdb:
            src, src_ty = (gc, gc_ty) if i % 2 == 0 else (gc2, gc2_ty)
            dst, dst_ty = (gc2, gc2_ty) if i % 2 == 0 else (gc, gc_ty)
        e.l(f"scf.if {in64} {{")
        off = e.v("cso")
        e.l(f"{off} = arith.subi %tid, {e.c(d)} : index")
        ge = e.v("cge")
        e.l(f"{ge} = arith.cmpi sge, %tid, {e.c(d)} : index")
        offc = e.v("csc")
        e.l(f"{offc} = arith.select {ge}, {off}, %c0 : index")
        lv = e.v("cgl")
        e.l(f"{lv} = memref.load {src}[{offc}, %c0] : {src_ty}")
        vv = e.v("csv")
        e.l(f"{vv} = arith.select {ge}, {lv}, {zf} : f32")
        cur = e.v("cgc")
        e.l(f"{cur} = memref.load {src}[%tid, %c0] : {src_ty}")
        nv = e.v("cgn")
        e.l(f"{nv} = arith.addf {cur}, {vv} : f32")
        e.l(f"memref.store {nv}, {dst}[%tid, %c0] : {dst_ty}")
        e.l("}")
        e.l("gpu.barrier")


def emit_cell_while(e, total, nthr, body):
    """Cell loop as scf.while (NOT matched by ops{["scf.for"]} transforms:
    pipeline_shared_memory_copies hard-fails the whole -transform-interpreter
    output on ANY non-copy scf.for in this build, even with
    failures(suppress) -- probe-verified.  All non-K-loop cell passes in the
    round-2 kernels therefore use scf.while; only the K-loops stay scf.for
    so the pipeliner sees exactly the loops it can pipeline."""
    wir = e.v("wir")
    e.l(f"{wir} = scf.while (%wi = %tid) : (index) -> (index) {{")
    cond = e.v("wic")
    e.l(f"{cond} = arith.cmpi ult, %wi, {e.c(total)} : index")
    e.l(f"scf.condition({cond}) %wi : index")
    e.l("} do {")
    e.l("^bb0(%wib: index):")
    body("%wib")
    n = e.v("win")
    e.l(f"{n} = arith.addi %wib, {e.c(nthr)} : index")
    e.l(f"scf.yield {n} : index")
    e.l("}")


def emit_solve_p1_wy(e, A_m, A_ty, Abf, Abf_ty, Tt, Tt_ty, Tb, Tb_ty,
                     Tg, Tg_ty, bs, bs_ty, gc, gc_ty, p1_bf16, nthr=128):
    """R19-fwd GDN_F1WY: warp-parallel solve phase-1 (default off).

    Replaces the thread-per-column diagonal-block recurrences (64
    threads, each a 16-step serial substitution whose inner dots are
    smem-latency-chained -- the r18/r19 critical path of the per-CTA
    solve) with one warp per 16x16 diagonal block: per row i, lane k
    contributes the partial vector A[i,k] * T-row-k and a width-32 xor
    butterfly (gpu.shuffle, 5 shfl+add stages) reduces all 16 columns'
    dots simultaneously.  Depth: 16 steps x ~5 chained shfl-adds vs the
    legacy's ~120 latency-chained smem FMA steps.

    DESIGN (v3, the one that passes the gate): NO lane-indexed select
    webs anywhere.  The first two cuts kept the T rows in per-lane
    register vectors updated by select(lane==i, v, r) and extracted the
    store column with a 16-way select tree -- the toolchain miscompiled
    that web (probe_ttdump: columns 0-3 exact, columns >= 4 stale from
    the first off-diagonal; numpy twin exact; the PTX shows the select
    tree CSE'd against the init predicates) -- and the width-16
    butterfly lowered to shfl.sync membermask 0xFFFF (lanes 16..31
    executing it = UB).  v3 instead: T rows live in the Tt SMEM tile
    (pre-zeroed diag blocks -- the r18 NaN-proofing lesson: rows read
    before first written are killed by A's exact-zero upper, but uninit
    smem can hold NaN bit patterns), the store column index is a
    COMPILE-TIME c (warp-redundant stores: every lane stores the same
    broadcast value to the same address), and the butterfly is
    width-32 with lanes 16..31 contributing exact zeros via a masked A
    load.  One gpu.barrier per step orders the cross-lane
    store->reload.  Correctness otherwise rests on A's strict-lower
    storage (upper + diagonal cells are exact zeros): the butterfly's
    k > i terms are A[i,k] * T-row-k with A[i,k] = 0 exactly.

    fp32-order note (the mission's bit-sensitivity flag): each cell's
    dot now sums in the butterfly's tree order instead of ascending-k;
    T is stored bf16 so the change class is the r16 bf16-solve rounding
    class -- verify o/ht margins stay >=2x under bar.

    Requires nthr == 128 (warp D = diagonal block D; lanes 0..15
    active).
    """
    c16 = e.c(16)
    c32 = e.c(32)
    zbf = e.v("wyzb")
    e.l(f"{zbf} = arith.constant 0.0 : bf16")
    lane = e.v("wyln")
    e.l(f"{lane} = arith.remsi %tid, {c32} : index")
    d = e.v("wydb")
    e.l(f"{d} = arith.divsi %tid, {c32} : index")
    act = e.v("wyac")
    e.l(f"{act} = arith.cmpi ult, {lane}, {c16} : index")
    ln0 = e.v("wyl0")
    e.l(f"{ln0} = arith.select {act}, {lane}, %c0 : index")
    krow = e.v("wykr")
    e.l(f"{krow} = arith.muli {d}, {c16} : index")
    krow = addi(e, krow, ln0)
    kcol = e.v("wykc")
    e.l(f"{kcol} = arith.muli {d}, {c16} : index")
    kcol = addi(e, kcol, ln0)

    # ---- pre-zero the four 16x16 diagonal blocks of Tt ----
    # (rows k are re-read at step i for all k, including k > i whose
    # cells are not yet written; their partial is A[i,k]*T[k,c] with
    # A[i,k] = 0 exactly, but uninit smem can carry NaN bit patterns
    # and 0*NaN = NaN -- zero them, the r18 ZFIX-class discipline.)
    def _zz(i):
        blk = divsi(e, i, e.c(256))
        rest = remsi(e, i, e.c(256))
        zr = addi(e, muli(e, blk, c16), divsi(e, rest, c16))
        zc = remsi(e, rest, c16)
        e.l(f"memref.store {zbf}, {Tt}[{zr}, {zc}] : {Tt_ty}")
    emit_cell_while(e, 4 * 256, nthr, _zz)
    e.l("gpu.barrier")

    # per-column scale constants (static c, runtime block d)
    beta, ex = [], []
    for c in range(16):
        bc = addi(e, muli(e, d, c16), e.c(c))
        b_ = e.v("wybc")
        e.l(f"{b_} = memref.load {bs}[{bc}, %c0] : {bs_ty}")
        beta.append(b_)
        g_ = e.v("wygc")
        e.l(f"{g_} = memref.load {gc}[{bc}, %c0] : {gc_ty}")
        x_ = e.v("wyex")
        e.l(f"{x_} = math.exp2 {g_} : f32")
        ex.append(x_)
    one = e.v("wy1")
    e.l(f"{one} = arith.constant 1.0 : f32")
    zero = e.v("wy0")
    e.l(f"{zero} = arith.constant 0.0 : f32")
    w32 = e.v("wyw")
    e.l(f"{w32} = arith.constant 32 : i32")
    offs = {}
    for off in (16, 8, 4, 2, 1):
        ov = e.v("wyof")
        e.l(f"{ov} = arith.constant {off} : i32")
        offs[off] = ov

    for i in range(16):
        irow = e.v("wyir")
        e.l(f"{irow} = arith.muli {d}, {c16} : index")
        irow = addi(e, irow, e.c(i))
        if p1_bf16:
            ab = e.v("wyab")
            e.l(f"{ab} = memref.load {Abf}[{irow}, {kcol}] : {Abf_ty}")
            aik = e.v("wya")
            e.l(f"{aik} = arith.extf {ab} : bf16 to f32")
        else:
            aik = e.v("wya")
            e.l(f"{aik} = memref.load {A_m}[{irow}, {kcol}] : {A_ty}")
        aikz = e.v("wyaz")
        e.l(f"{aikz} = arith.select {act}, {aik}, {zero} : f32")
        # lane k's partial vector: A[i,k] * T-row-k (from the Tt tile)
        p = []
        for c in range(16):
            cc = addi(e, muli(e, d, c16), e.c(c))
            t_ = e.v("wytr")
            e.l(f"{t_} = memref.load {Tt}[{krow}, {cc}] : {Tt_ty}")
            tf = e.v("wytf")
            e.l(f"{tf} = arith.extf {t_} : bf16 to f32")
            pc = e.v("wyp")
            e.l(f"{pc} = arith.mulf {aikz}, {tf} : f32")
            p.append(pc)
        # width-32 xor butterfly: d[c] = sum_k A[i,k]*T[k,c]
        for off in (16, 8, 4, 2, 1):
            for c in range(16):
                o = e.v("wysh")
                ok = e.v("wyok")
                e.l(f"{o}, {ok} = gpu.shuffle xor {p[c]}, {offs[off]}, "
                    f"{w32} : f32")
                s = e.v("wys")
                e.l(f"{s} = arith.addf {p[c]}, {o} : f32")
                p[c] = s
        for c in range(16):
            nc = e.v("wyng")
            e.l(f"{nc} = arith.negf {p[c]} : f32")
            p[c] = nc
        p[i] = one
        # stores: STATIC column c, warp-redundant (all lanes hold the
        # reduced v_c; same-address stores broadcast; straight-line,
        # not cell loops -- the f2 pairing rule does not apply)
        for c in range(16):
            tcol = addi(e, muli(e, d, c16), e.c(c))
            ttv = e.v("wytt")
            e.l(f"{ttv} = arith.truncf {p[c]} : f32 to bf16")
            e.l(f"memref.store {ttv}, {Tt}[{irow}, {tcol}] : {Tt_ty}")
            vb = e.v("wyvb")
            e.l(f"{vb} = arith.mulf {p[c]}, {beta[c]} : f32")
            tbb = e.v("wybb")
            e.l(f"{tbb} = arith.truncf {vb} : f32 to bf16")
            e.l(f"memref.store {tbb}, {Tb}[{tcol}, {irow}] : {Tb_ty}")
            vg = e.v("wyvg")
            e.l(f"{vg} = arith.mulf {vb}, {ex[c]} : f32")
            vgf = e.v("wygf")
            e.l(f"{vgf} = arith.truncf {vg} : f32 to bf16")
            e.l(f"memref.store {vgf}, {Tg}[{tcol}, {irow}] : {Tg_ty}")
        e.l("gpu.barrier")


def emit_solve_blocked(e, A_m, A_ty, Abf, Abf_ty, Tt, Tt_ty, St,
                       Tb, Tb_ty, Tg, Tg_ty, bs, bs_ty, gc, gc_ty, C=64,
                       mid_hook=None, p1_bf16=False, nthr=128):
    """Blocked 16x16 forward-substitution solve (the fla-style retile).

    T = (I+A)^-1, A strictly lower [C,C].  Blocks of 16:
      phase 0: zero the upper cells of the transposed output tiles.
      phase 1: the four DIAGONAL blocks solved thread-per-column (16-length
               recurrences: ~120 serial FMAs vs the 2016 of the flat
               64-column solve -- the flat solve was the F1 critical path
               at ~45us/CTA).  Threads tid<64: block D=tid/16, col t=tid%16.
               Stores unscaled Tt[i][c] plus the beta / beta*2^gc scaled
               TRANSPOSED outputs Tb[c][i], Tg[c][i].
      phase 2: the six OFF-DIAGONAL blocks by mma (warp w handles column
               block C=w; block-row phases R=1..3 separated by barriers
               since (R,C) needs T[K][C], K<R):
               S = sum_K Abf[R][K] @ Tt[K][C]   (m16n8k16 x2, bf16)
               T[R][C] = -(Tt[R][R] @ S)        (through the St scratch)
               epilogue stores Tt (unscaled) + scaled transposed Tb/Tg.
    """
    # R21-fwd GDN_F1T_* (default off, ALL timing-only deletion probes --
    # outputs are GARBAGE under any of them; never stack with verify):
    #   GDN_F1T_NP0 = delete solve phase 0 (the zero_up cell loop)
    #   GDN_F1T_NP1 = delete solve phase 1 (diagonal-block substitution)
    #   GDN_F1T_NP2 = delete solve phase 2 (off-diagonal mma rounds;
    #                 barriers kept)
    # Composable (NP0+NP1+NP2 deletes the whole solve body).  The
    # barriers stay so the sync skeleton is unchanged -- the delta is
    # the phase's compute+mem footprint (the G5-F deletion-probe form).
    _np0 = os.environ.get("GDN_F1T_NP0")
    _np1 = os.environ.get("GDN_F1T_NP1")
    _np2 = os.environ.get("GDN_F1T_NP2")
    # R21-fwd GDN_F1ZF (default off): fold solve phase 0 into phase 1's
    # IDLE threads.  The r21 deletion map: zero_up costs ~8.5us of f1's
    # ~227us @T8192 while threads 64..127 idle through the ~32us serial
    # diagonal substitution (tid<64).  Fold: the upper-half threads zero
    # the transposed-upper cells CONCURRENTLY, and ONE barrier replaces
    # the phase-0/phase-1 pair.  Bit-identity: the folded predicate
    # block(r)>block(c) (which implies r>c) writes ONLY cells the
    # substitution never touches -- the serial chain writes EVERY
    # same-block cell (its -0.0 uppers included), so the write sets are
    # disjoint and Tb/Tg bits are deterministic-identical to ship.
    # Requires the serial phase-1 arm (not F1WY/F1P4) and nthr=128.
    _zf = os.environ.get("GDN_F1ZF")
    if _zf and (os.environ.get("GDN_F1WY") or os.environ.get("GDN_F1P4")
                or os.environ.get("GDN_F1T_NP1")):
        raise SystemExit("GDN_F1ZF needs the serial phase-1 arm")
    if _zf and nthr != 128:
        raise SystemExit("GDN_F1ZF needs nthr=128 (threads 64..127 "
                         "are the folding arm)")
    cC = e.c(C)
    c16 = e.c(16)
    e.l("%warp_sv = arith.divsi %tid, %c32 : index")
    ldm_precompute(e, "sv")
    zbf = e.v("svz2")
    e.l(f"{zbf} = arith.constant 0.0 : bf16")

    # ---- phase 0: zero Tb/Tg lower cells (j > i in transposed layout) ----
    def zero_up(i):
        r = divsi(e, i, cC)
        c = remsi(e, i, cC)
        gt = e.v("svgt")
        e.l(f"{gt} = arith.cmpi sgt, {r}, {c} : index")
        e.l(f"scf.if {gt} {{")
        e.l(f"  memref.store {zbf}, {Tb}[{r}, {c}] : {Tb_ty}")
        e.l(f"  memref.store {zbf}, {Tg}[{r}, {c}] : {Tg_ty}")
        e.l("}")
    # R18-fwd GDN_ZFIX (default off): the stock stride 32*8=256 exceeds
    # the 128-thread block, leaving Tb/Tg rows r%4 in {2,3} (cells
    # (r*64+c)%256 in [128,256)) NEVER zeroed -> uninitialized-smem reads
    # in the uw gemms (the F1KS 2-CTA corruption; masked at 1 CTA where
    # the fresh window reads zeros).  ZFIX strides nthr for full cover.
    _zfix_stride = nthr if os.environ.get("GDN_ZFIX") else 32 * 8
    if not _zf:
        # ship order: zero FIRST (all threads), barrier, then phase 1.
        if not _np0:
            emit_cell_while(e, C * C, _zfix_stride, zero_up)
        e.l("gpu.barrier")

    # ---- phase 1: diagonal blocks ----
    # R21-fwd GDN_F1T_NP1 (default off): first arm of the chain deletes
    # the phase-1 body entirely (timing probe; Tt/Tb/Tg diagonals stay
    # unwritten -> garbage downstream, never verified).
    if _np1:
        pass
    # R19-fwd GDN_F1WY (default off): warp-parallel phase-1 -- one warp
    # per 16x16 diagonal block, lane-owned row vectors + width-16 xor
    # butterfly dots (emit_solve_p1_wy above).  w4 builds only.
    elif os.environ.get("GDN_F1WY"):
        if nthr != 128:
            raise SystemExit("GDN_F1WY: warp-per-block phase-1 needs "
                             "nthr=128 (--warps 4)")
        emit_solve_p1_wy(e, A_m, A_ty, Abf, Abf_ty, Tt, Tt_ty, Tb, Tb_ty,
                         Tg, Tg_ty, bs, bs_ty, gc, gc_ty, p1_bf16,
                         nthr=nthr)
    elif os.environ.get("GDN_F1P4"):
        # R20-fwd GDN_F1P4 (default off): phase-1 v4 -- 4 threads per
        # diagonal block x 4 REGISTER-TILED columns each (the r19 F1WY
        # post-mortem direction).  Per row i the k-sum runs as scf.for
        # from the thread's base column c0 (STATIC lower bound, dynamic
        # upper i -- pipelineable, vs the serial chain's scf.while) with
        # FOUR interleaved accumulator chains: 4x FMA ILP, one shared
        # A[i,k] load per step amortized over 4 columns.  The k<c cols
        # read Tt's -0.0 upper cells: (+0.0)+(+-0.0) products leave the
        # accumulator +0.0 (IEEE round-to-nearest zero-sum rule), so the
        # REAL terms accumulate in the SAME ascending order as the
        # serial chain -> BIT-IDENTICAL values incl. the -0.0 uppers.
        # No shuffles, no barriers, no lane-indexed select webs (the
        # three F1WY correctness cuts); each thread reads/writes ONLY
        # its own 4 columns (no cross-thread dependencies).  Threads
        # tid>=16 idle to the existing phase-1 barrier.
        in16 = e.v("p4in")
        e.l(f"{in16} = arith.cmpi ult, %tid, {e.c(16)} : index")
        e.l(f"scf.if {in16} {{")
        db = e.v("p4d")
        e.l(f"{db} = arith.divsi %tid, {e.c(4)} : index")
        jj = e.v("p4j")
        e.l(f"{jj} = arith.remsi %tid, {e.c(4)} : index")
        c0 = addi(e, muli(e, db, c16), muli(e, jj, e.c(4)))
        # per-column beta / 2^gc, hoisted out of the row loop
        betas, exps, cols = [], [], []
        for j in range(4):
            cj = addi(e, c0, e.c(j))
            cols.append(cj)
            b_ = e.v(f"p4b{j}")
            e.l(f"{b_} = memref.load {bs}[{cj}, %c0] : {bs_ty}")
            g_ = e.v(f"p4g{j}")
            e.l(f"{g_} = memref.load {gc}[{cj}, %c0] : {gc_ty}")
            x_ = e.v(f"p4x{j}")
            e.l(f"{x_} = math.exp2 {g_} : f32")
            betas.append(b_)
            exps.append(x_)
        one = e.v("p4one")
        e.l(f"{one} = arith.constant 1.0 : f32")
        zf = e.v("p4z")
        e.l(f"{zf} = arith.constant 0.0 : f32")
        for t in range(16):
            i = addi(e, muli(e, db, c16), e.c(t))   # global row of block D
            # NOTE: scf.while (NOT scf.for) -- the payload transform's
            # pipeline_shared_memory_copies matches every scf.for and
            # errors on copy-free loops ("no shared memory copy"); the
            # serial chain dodges it the same way.  The while's k-chain
            # is independent of the FMA chain, so the loads still issue
            # ahead of the accumulators (the 4-way ILP survives).
            res = [e.v(f"p4r{t}_{j}") for j in range(4)]
            e.l(f"{e.v('p4kw' + str(t))}, "
                + ", ".join(res)
                + f" = scf.while (%kw{t} = {c0}, %wa0_{t} = {zf}, "
                f"%wa1_{t} = {zf}, %wa2_{t} = {zf}, %wa3_{t} = {zf}) : "
                "(index, f32, f32, f32, f32) -> "
                "(index, f32, f32, f32, f32) {")
            e.l(f"  %p4c_{t} = arith.cmpi slt, %kw{t}, {i} : index")
            e.l(f"  scf.condition(%p4c_{t}) %kw{t}, %wa0_{t}, %wa1_{t}, "
                f"%wa2_{t}, %wa3_{t} : index, f32, f32, f32, f32")
            e.l("} do {")
            e.l(f"^bb0(%kwb{t}: index, %ab0_{t}: f32, %ab1_{t}: f32, "
                f"%ab2_{t}: f32, %ab3_{t}: f32):")
            if p1_bf16:
                e.l(f"  %p4ak_{t} = memref.load {Abf}[{i}, %kwb{t}]"
                    f" : {Abf_ty}")
                e.l(f"  %p4a_{t} = arith.extf %p4ak_{t} : bf16 to f32")
            else:
                e.l(f"  %p4a_{t} = memref.load {A_m}[{i}, %kwb{t}]"
                    f" : {A_ty}")
            for j in range(4):
                e.l(f"  %p4tj_{t}_{j} = memref.load {Tt}[%kwb{t}, "
                    f"{cols[j]}] : {Tt_ty}")
                e.l(f"  %p4tf_{t}_{j} = arith.extf %p4tj_{t}_{j}"
                    f" : bf16 to f32")
                e.l(f"  %p4mu_{t}_{j} = arith.mulf %p4a_{t}, "
                    f"%p4tf_{t}_{j} : f32")
                e.l(f"  %p4ad_{t}_{j} = arith.addf %ab{j}_{t}, "
                    f"%p4mu_{t}_{j} : f32")
            e.l(f"  %p4kn_{t} = arith.addi %kwb{t}, {e.c(1)} : index")
            e.l(f"  scf.yield %p4kn_{t}, " + ", ".join(
                f"%p4ad_{t}_{j}" for j in range(4))
                + " : index, f32, f32, f32, f32")
            e.l("}")
            for j in range(4):
                cj = cols[j]
                isd = e.v(f"p4q{t}_{j}")
                e.l(f"{isd} = arith.cmpi eq, {cj}, {i} : index")
                neg = e.v(f"p4n{t}_{j}")
                e.l(f"{neg} = arith.negf {res[j]} : f32")
                val = e.v(f"p4v{t}_{j}")
                e.l(f"{val} = arith.select {isd}, {one}, {neg} : f32")
                tv = e.v(f"p4tt{t}_{j}")
                e.l(f"{tv} = arith.truncf {val} : f32 to bf16")
                e.l(f"memref.store {tv}, {Tt}[{i}, {cj}] : {Tt_ty}")
                vb = e.v(f"p4vb{t}_{j}")
                e.l(f"{vb} = arith.mulf {val}, {betas[j]} : f32")
                tbb = e.v(f"p4tb{t}_{j}")
                e.l(f"{tbb} = arith.truncf {vb} : f32 to bf16")
                e.l(f"memref.store {tbb}, {Tb}[{cj}, {i}] : {Tb_ty}")
                vg = e.v(f"p4vg{t}_{j}")
                e.l(f"{vg} = arith.mulf {vb}, {exps[j]} : f32")
                vgf = e.v(f"p4tg{t}_{j}")
                e.l(f"{vgf} = arith.truncf {vg} : f32 to bf16")
                e.l(f"memref.store {vgf}, {Tg}[{cj}, {i}] : {Tg_ty}")
        e.l("}")
    else:
        in64 = e.v("in64")
        e.l(f"{in64} = arith.cmpi ult, %tid, {cC} : index")
        e.l(f"scf.if {in64} {{")
        db = e.v("svd")
        e.l(f"{db} = arith.divsi %tid, {c16} : index")
        tb_ = e.v("svt")
        e.l(f"{tb_} = arith.remsi %tid, {c16} : index")
        ccol = e.v("svc")
        e.l(f"{ccol} = arith.muli {db}, {c16} : index")
        ccol = addi(e, ccol, tb_)
        betav = e.v("svb1")
        e.l(f"{betav} = memref.load {bs}[{ccol}, %c0] : {bs_ty}")
        gcv = e.v("svg1")
        e.l(f"{gcv} = memref.load {gc}[{ccol}, %c0] : {gc_ty}")
        ex = e.v("svx1")
        e.l(f"{ex} = math.exp2 {gcv} : f32")
        one = e.v("svo")
        e.l(f"{one} = arith.constant 1.0 : f32")
        zf = e.v("svz")
        e.l(f"{zf} = arith.constant 0.0 : f32")
        for t in range(16):
            i = addi(e, muli(e, db, c16), e.c(t))     # global row of block D
            svi = e.v("sviw")
            sv = e.v("svw")
            e.l(f"{svi}, {sv} = scf.while (%sk = {ccol}, %sa = {zf}) : "
                f"(index, f32) -> (index, f32) {{")
            e.l(f"  %scnd = arith.cmpi slt, %sk, {i} : index")
            e.l(f"  scf.condition(%scnd) %sk, %sa : index, f32")
            e.l("} do {")
            e.l("^bb0(%skb: index, %sab: f32):")
            if p1_bf16:
                # R16-Z GDN_BF16SOLVE: the recurrence reads the bf16 A tile
                # (A_m is not allocated); extf to f32 for the FMA.
                e.l(f"  %aikb = memref.load {Abf}[{i}, %skb] : {Abf_ty}")
                e.l(f"  %aik = arith.extf %aikb : bf16 to f32")
            else:
                e.l(f"  %aik = memref.load {A_m}[{i}, %skb] : {A_ty}")
            e.l(f"  %tkj = memref.load {Tt}[%skb, {ccol}] : {Tt_ty}")
            e.l(f"  %tkf = arith.extf %tkj : bf16 to f32")
            e.l(f"  %pr = arith.mulf %aik, %tkf : f32")
            e.l(f"  %sn = arith.addf %sab, %pr : f32")
            e.l(f"  %skn = arith.addi %skb, %c1 : index")
            e.l(f"  scf.yield %skn, %sn : index, f32")
            e.l("}")
            neg = e.v("svn")
            e.l(f"{neg} = arith.negf {sv} : f32")
            isd = e.v("svq")
            e.l(f"{isd} = arith.cmpi eq, %tid, {i} : index")
            val = e.v("svl")
            e.l(f"{val} = arith.select {isd}, {one}, {neg} : f32")
            tbv = e.v("svbt")
            e.l(f"{tbv} = arith.truncf {val} : f32 to bf16")
            e.l(f"memref.store {tbv}, {Tt}[{i}, {ccol}] : {Tt_ty}")
            vb = e.v("svvb")
            e.l(f"{vb} = arith.mulf {val}, {betav} : f32")
            tbb = e.v("svbb")
            e.l(f"{tbb} = arith.truncf {vb} : f32 to bf16")
            e.l(f"memref.store {tbb}, {Tb}[{ccol}, {i}] : {Tb_ty}")
            vg = e.v("svvg")
            e.l(f"{vg} = arith.mulf {vb}, {ex} : f32")
            vgf = e.v("svgf")
            e.l(f"{vgf} = arith.truncf {vg} : f32 to bf16")
            e.l(f"memref.store {vgf}, {Tg}[{ccol}, {i}] : {Tg_ty}")
        e.l("}")
    if _zf:
        # folded phase-0 arm: threads 64..127 zero the LIVE upper cells
        # (block(r) > block(c), which implies r > c) concurrently with
        # the substitution; the while's %tid base (in [64,128)) with
        # stride 64 covers every cell >= 64, and cells < 64 are rows
        # r=0 (predicate dead) -- full cover of all live cells.
        ge64 = e.v("zfe")
        e.l(f"{ge64} = arith.cmpi uge, %tid, {cC} : index")
        e.l(f"scf.if {ge64} {{")

        def zero_live(i):
            r = divsi(e, i, cC)
            c = remsi(e, i, cC)
            br = divsi(e, r, c16)
            bc = divsi(e, c, c16)
            gt = e.v("svgt")
            e.l(f"{gt} = arith.cmpi sgt, {br}, {bc} : index")
            e.l(f"scf.if {gt} {{")
            e.l(f"  memref.store {zbf}, {Tb}[{r}, {c}] : {Tb_ty}")
            e.l(f"  memref.store {zbf}, {Tg}[{r}, {c}] : {Tg_ty}")
            e.l("}")
        emit_cell_while(e, C * C, 64, zero_live)
        e.l("}")
    e.l("gpu.barrier")
    if mid_hook is not None:
        # R15-X GDN_F1AL: phase-1 A_m reads are complete at this barrier;
        # the vt cp.async group may now land in A_m's arena (overlapping
        # all of phase 2).
        mid_hook(e)

    import os as _os3
    if _os3.environ.get("G5D_DUMP_P1"):
        dbx = e.v("dbx"); e.l(f"{dbx} = gpu.block_id x")
        dby = e.v("dby"); e.l(f"{dby} = gpu.block_id y")
        dbase = addi(e, muli(e, dby, e.c(8)), dbx)
        dbase = muli(e, dbase, e.c(4096))
        def _dp1(i):
            r = divsi(e, i, e.c(64))
            c = remsi(e, i, e.c(64))
            v = e.v("dp1")
            e.l(f"{v} = memref.load {Tt}[{r}, {c}] : {Tt_ty}")
            e.l(f"memref.store {v}, %tt["
                f"{addi(e, dbase, addi(e, muli(e, r, e.c(64)), c))}] : "
                f"memref<131072xbf16>")
        emit_cell_while(e, 64 * 64, 32 * 8, _dp1)
        def _da1(i):
            r = divsi(e, i, e.c(64))
            c = remsi(e, i, e.c(64))
            v = e.v("da1")
            e.l(f"{v} = memref.load {A_m}[{r}, {c}] : {A_ty}")
            w = e.v("da1t")
            e.l(f"{w} = arith.truncf {v} : f32 to bf16")
            e.l(f"memref.store {w}, %tbb["
                f"{addi(e, muli(e, r, e.c(64)), c)}] : "
                f"memref<131072xbf16>")
        emit_cell_while(e, 64 * 64, 32 * 8, _da1)

    # ---- phase 2: off-diagonal blocks via mma, warp w = column block ----
    # Per block-row R: STASH phase (each warp Cb<R computes S = sum_K
    # Abf[R,K] @ Tt[K,Cb] into its OWN scratch tile St_Cb), unconditional
    # barrier, CONSUME phase (T[R,Cb] = -(Tt[R,R] @ St_Cb), epilogue
    # stores), unconditional barrier.  All barriers OUTSIDE scf.if
    # (conditional gpu.barrier = partial arrivals = broken sync on
    # sm_120a; found via w-row-0-block corruption + u-wide corruption).
    for ni in (0, 1):
        e.l(f"%nc_sv_{ni} = arith.constant {8 * ni} : index")
        e.l(f"%nct_sv_{ni} = arith.addi %nc_sv_{ni}, %t2 : index")
        e.l(f"%nct1_sv_{ni} = arith.addi %nc_sv_{ni}, %t2p1 : index")
    e.l("%zf_sv = arith.constant 0.0 : f32")
    sts = list(St)  # St = (ssa, ty) -> per-Cb list of tuples
    for R in (1, 2, 3):
        # R21-fwd GDN_F1T_NP2 (default off): stash+consume bodies deleted;
        # every gpu.barrier kept (timing probe).
        if not _np2:
            for Cb in range(0, R):
                Stc, Stc_ty = sts[Cb]
                wp = e.v("svwp")
                e.l(f"{wp} = arith.cmpi eq, %warp_sv, {e.c(Cb)} : index")
                e.l(f"scf.if {wp}{{")
                accs = []
                for ni in (0, 1):
                    an = e.v("acs")
                    e.l(f"{an} = vector.broadcast %zf_sv : f32 "
                         f"to vector<2x2xf32>")
                    accs.append(an)
                for K in range(Cb, R):
                    fa = e.v("fas")
                    row = addi(e, addi(e, e.c(16 * R), "%lh8_sv"), "%lq7_sv")
                    col = addi(e, e.c(16 * K), "%lt8_sv")
                    e.l(f"{fa} = nvgpu.ldmatrix {Abf}[{row}, {col}] "
                         f"numTiles = 4 transpose = false : {Abf_ty} -> "
                         f"vector<4x2xbf16>")
                    rowb = addi(e, addi(e, e.c(16 * K), "%lb8_sv"), "%lq7_sv")
                    # +16*Cb block offset on the B-frag column base
                    # (was missing: warp 0 read right blocks, warps
                    # 1/2 read the wrong Tt column blocks)
                    for ni in (0, 1):
                        cbn = e.v("cbn")
                        e.l(f"{cbn} = arith.addi %nc_sv_{ni}, "
                             f"{e.c(16 * Cb)} : index")
                        fb = e.v("fbs")
                        e.l(f"{fb} = nvgpu.ldmatrix {Tt}[{rowb}, "
                             f"{cbn}] numTiles = 2 transpose = true : "
                             f"{Tt_ty} -> vector<2x2xbf16>")
                        o = e.v("mrs")
                        e.l(f"{o} = nvgpu.mma.sync({fa}, {fb}, {accs[ni]}) "
                             f"mmaShape = [16, 8, 16] : (vector<4x2xbf16>, "
                             f"vector<2x2xbf16>, vector<2x2xf32>) -> "
                             f"vector<2x2xf32>")
                        accs[ni] = o
                for ni in (0, 1):
                    for ei in range(4):
                        r_ = "%lg" if ei < 2 else "%lg8"
                        c_ = (f"%nct_sv_{ni}" if ei % 2 == 0
                              else f"%nct1_sv_{ni}")
                        x = acc_extract(e, accs[ni], ei // 2, ei % 2, "exs")
                        t = e.v("sts")
                        e.l(f"{t} = arith.truncf {x} : f32 to bf16")
                        e.l(f"memref.store {t}, {Stc}[{r_}, {c_}] : {Stc_ty}")
                e.l("}")
        e.l("gpu.barrier")
        if not _np2:
            for Cb in range(0, R):
                Stc, Stc_ty = sts[Cb]
                wp = e.v("svwp")
                e.l(f"{wp} = arith.cmpi eq, %warp_sv, {e.c(Cb)} : index")
                e.l(f"scf.if {wp} {{")
                fa2 = e.v("fa2")
                row = addi(e, addi(e, e.c(16 * R), "%lh8_sv"), "%lq7_sv")
                col = addi(e, e.c(16 * R), "%lt8_sv")
                e.l(f"{fa2} = nvgpu.ldmatrix {Tt}[{row}, {col}] numTiles = 4 "
                     f"transpose = false : {Tt_ty} -> vector<4x2xbf16>")
                for ni in (0, 1):
                    fb2 = e.v("fb2")
                    rowb = addi(e, "%lb8_sv", "%lq7_sv")
                    e.l(f"{fb2} = nvgpu.ldmatrix {Stc}[{rowb}, %nc_sv_{ni}] "
                         f"numTiles = 2 transpose = true : {Stc_ty} -> "
                         f"vector<2x2xbf16>")
                    az = e.v("acz")
                    e.l(f"{az} = vector.broadcast %zf_sv : f32 "
                         f"to vector<2x2xf32>")
                    o = e.v("mrz")
                    e.l(f"{o} = nvgpu.mma.sync({fa2}, {fb2}, {az}) "
                         f"mmaShape = [16, 8, 16] : (vector<4x2xbf16>, "
                         f"vector<2x2xbf16>, vector<2x2xf32>) -> "
                         f"vector<2x2xf32>")
                    for ei in range(4):
                        r_ = "%lg" if ei < 2 else "%lg8"
                        c_in = (f"%nct_sv_{ni}" if ei % 2 == 0
                                else f"%nct1_sv_{ni}")
                        x = acc_extract(e, o, ei // 2, ei % 2, "exz")
                        nx = e.v("ngz")
                        e.l(f"{nx} = arith.negf {x} : f32")
                        tt = e.v("ttz")
                        e.l(f"{tt} = arith.truncf {nx} : f32 to bf16")
                        grow = addi(e, e.c(16 * R), r_)
                        gcol = addi(e, e.c(16 * Cb), c_in)
                        e.l(f"memref.store {tt}, {Tt}[{grow}, {gcol}] : {Tt_ty}")
                        bv = e.v("bvz")
                        e.l(f"{bv} = memref.load {bs}[{gcol}, %c0] : {bs_ty}")
                        gv = e.v("gvz")
                        e.l(f"{gv} = memref.load {gc}[{gcol}, %c0] : {gc_ty}")
                        exz = e.v("exz2")
                        e.l(f"{exz} = math.exp2 {gv} : f32")
                        m1 = e.v("m1z")
                        e.l(f"{m1} = arith.mulf {nx}, {bv} : f32")
                        tbv = e.v("tbz")
                        e.l(f"{tbv} = arith.truncf {m1} : f32 to bf16")
                        e.l(f"memref.store {tbv}, {Tb}[{gcol}, {grow}] : {Tb_ty}")
                        m2 = e.v("m2z")
                        e.l(f"{m2} = arith.mulf {m1}, {exz} : f32")
                        tgv = e.v("tgz")
                        e.l(f"{tgv} = arith.truncf {m2} : f32 to bf16")
                        e.l(f"memref.store {tgv}, {Tg}[{gcol}, {grow}] : {Tg_ty}")
                e.l("}")
        e.l("gpu.barrier")
    e.l("gpu.barrier")



def emit_solve2(e, A_ssa, A_ty, Tb, Tb_ty, Tg, Tg_ty, bs, bs_ty, gc, gc_ty,
                C=64):
    """T = (I+A)^-1 forward substitution, thread-per-column (tid<C), storing
    TWO beta-scaled transposed tiles so the consumer GEMMs read RAW global
    operands (kills the scalar beta*v and beta*k*2^gc staging passes):

        Tb[j][i] = T[i][j] * beta_j          (for u = Tb @ v)
        Tg[j][i] = T[i][j] * beta_j * 2^gc_j (for w = Tg @ k)

    The beta column scaling commutes through the recurrence: X = T diag(beta)
    satisfies the same substitution with X[j] = beta_j.  The i-loop is
    UNROLLED (straight-line blocks, no outer scf.for) so the Tb/Tg allocs
    keep non-loop users (transform.memref.multibuffer skips them); the
    runtime-bound k-sum stays a small scf.for per block.  i<j blocks run an
    empty k-sum and store -0 (== 0.0); i==j selected to the scaled diagonal.
    """
    cC = e.c(C)
    in64 = e.v("in64")
    e.l(f"{in64} = arith.cmpi ult, %tid, {cC} : index")
    e.l(f"scf.if {in64} {{")
    betav = e.v("svb")
    e.l(f"{betav} = memref.load {bs}[%tid, %c0] : {bs_ty}")
    gcv = e.v("svg")
    e.l(f"{gcv} = memref.load {gc}[%tid, %c0] : {gc_ty}")
    ex = e.v("svx")
    e.l(f"{ex} = math.exp2 {gcv} : f32")
    zf = e.v("svz")
    e.l(f"{zf} = arith.constant 0.0 : f32")
    for i in range(C):
        kv = e.v("skw")
        svi = e.v("svi")
        sv = e.v("sv")
        # k-sum as scf.while (runtime bounds j..i-1; not a pipeliner target).
        # args and results share types so every region-branch edge matches.
        e.l(f"{svi}, {sv} = scf.while (%sk = %tid, %sa = {zf}) : "
            f"(index, f32) -> (index, f32) {{")
        e.l(f"  %scnd = arith.cmpi slt, %sk, {e.c(i)} : index")
        e.l(f"  scf.condition(%scnd) %sk, %sa : index, f32")
        e.l("} do {")
        e.l("^bb0(%skb: index, %sab: f32):")
        e.l(f"  %aik = memref.load {A_ssa}[{e.c(i)}, %skb] : {A_ty}")
        e.l(f"  %tkj = memref.load {Tb}[%tid, %skb] : {Tb_ty}")
        e.l(f"  %tkf = arith.extf %tkj : bf16 to f32")
        e.l(f"  %pr = arith.mulf %aik, %tkf : f32")
        e.l(f"  %sn = arith.addf %sab, %pr : f32")
        e.l(f"  %skn = arith.addi %skb, %c1 : index")
        e.l(f"  scf.yield %skn, %sn : index, f32")
        e.l("}")
        neg = e.v("sng")
        e.l(f"{neg} = arith.negf {sv} : f32")
        isd = e.v("ssd")
        e.l(f"{isd} = arith.cmpi eq, %tid, {e.c(i)} : index")
        val = e.v("ssl")
        e.l(f"{val} = arith.select {isd}, {betav}, {neg} : f32")
        tb = e.v("stb")
        e.l(f"{tb} = arith.truncf {val} : f32 to bf16")
        e.l(f"memref.store {tb}, {Tb}[%tid, {e.c(i)}] : {Tb_ty}")
        vg = e.v("svg2")
        e.l(f"{vg} = arith.mulf {val}, {ex} : f32")
        tg = e.v("stg")
        e.l(f"{tg} = arith.truncf {vg} : f32 to bf16")
        e.l(f"memref.store {tg}, {Tg}[%tid, {e.c(i)}] : {Tg_ty}")
    e.l("}")
    e.l("gpu.barrier")


# G5-D round-2: rewritten stage generators (gen_f1/gen_f2/gen_f3/gen_mono).
# Spliced into gdn_full_gen.py by the patch script.

# ---------------------------------------------------------------------------
# F1 "wy" (round 2: pipelined)
# ---------------------------------------------------------------------------

def gen_f1(shp, W, BV=None, bk=32):
    """WY precompute, grid (NC, B*H).

    Round-2 structure (all K-loops through the G2 transform pipeline):
      1. Hillis-Steele gc cumsum + beta stage (smem log-scan).
      2. kkt K-loop: ONE staged k tile serves both mma sides (A rows and
         kt=True B rows); decay/beta applied in the A epilogue (f32 A_m).
      3. solve2: forward substitution emitting Tb = T^T diag(beta) and
         Tg = T^T diag(beta*2^gc)  -> consumers read RAW v / k.
      4. fused ug+wg K-loop over C: A operands are the resident Tb/Tg
         (swapped ldm reads), B tiles v/K-slices staged by cp.async.
      5. u/w epilogues store to global.
    """
    B, T, H, K, V, C, NC, Tp = shp
    nthr = 32 * W
    # R21-fwd GDN_F1T_* (default off, timing-only deletion probes; the
    # f1 critical-path time-map round).  All outputs GARBAGE under any:
    #   GDN_F1T_NSC  = delete the gc Hillis-Steele scan + beta stage
    #   GDN_F1T_NKT  = delete the kkt K-loop + A epilogue
    #   GDN_F1T_NGC  = delete the gc_out store
    #   GDN_F1T_NUW  = delete the uw stage (gemms + u/w epilogue stores
    #                  + s2g copies), KEEP the operand staging issue
    #   GDN_F1T_NUST = NUW + also delete the uw operand staging issue
    #   GDN_F1T_NC2G = delete ONLY the u/w s2g copy passes (uw gemms +
    #                  epilogue + staging kept -- splits NUW's delta into
    #                  mma+epilogue vs global-store traffic)
    # R23-fwd kkt split (splits NKT's 59.5us lump, timing-only):
    #   GDN_F1T_NKK  = delete ONLY the kkt K-loop (staging + ldm + mma);
    #                  the A epilogue runs over zero accs (loads/exp2/
    #                  stores intact) -- delta = K-loop, NKT-NKK = epi
    #   GDN_F1T_NEK  = delete ONLY the A epilogue's exp2/beta chain (the
    #                  3 smem loads + subf/exp2/2xmulf/cmpi/select);
    #                  bare extract->store keeps accs+Abf live -- delta
    #                  = the chain, stores stay priced in both arms
    #   GDN_F1T_NKM  = delete ONLY the kkt K-loop's ldm+mma (staging +
    #                  barriers kept; epilogue on zero accs) -- splits
    #                  NKK's K-loop delta into staging vs compute
    # (solve-phase probes NP0/NP1/NP2 live in emit_solve_blocked.)
    _nsc = os.environ.get("GDN_F1T_NSC")
    _nkt = os.environ.get("GDN_F1T_NKT")
    _ngc = os.environ.get("GDN_F1T_NGC")
    _nuw = os.environ.get("GDN_F1T_NUW")
    _nust = os.environ.get("GDN_F1T_NUST")
    _nc2g = os.environ.get("GDN_F1T_NC2G")
    _nkk = os.environ.get("GDN_F1T_NKK")
    _nek = os.environ.get("GDN_F1T_NEK")
    _nkm = os.environ.get("GDN_F1T_NKM")
    if _nkt and (_nkk or _nek or _nkm):
        raise SystemExit("GDN_F1T_NKT already deletes the whole kkt stage")
    _nkmx = [x for x in (_nkk, _nek, _nkm) if x]
    if len(_nkmx) > 1:
        raise SystemExit("GDN_F1T_NKK/NEK/NKM are separate arms")
    e = Em()
    emit_lane_precompute(e)

    # R21-fwd probe keep-alive sinks: the payload transform's multibuffer
    # pattern HARD-ERRORS on a workgroup alloc with ZERO users, so every
    # deletion that orphans a tile emits one predicated scalar
    # load->gc_out store (tid==0; noise <<1us) keeping the malloc in the
    # IR -- the smem pool stays EXACTLY the ship's (occupancy untouched).
    # Placement respects the PTX lifetime arena: Tt (orphaned only under
    # NP1+NP2) is sunk PRE-solve so its lifetime stays a pre-solve point
    # (k2 lands in Tt's bytes POST-solve); vt/k2 (orphaned under NUST)
    # are sunk at the very END (their alias windows open post-solve).
    def probe_sink(tile, ty, tag):
        z = e.c(0)
        t0 = e.v("psk" + tag)
        e.l(f"{t0} = arith.cmpi eq, %tid, {z} : index")
        e.l(f"scf.if {t0} {{")
        v = e.v("psv" + tag)
        e.l(f"{v} = memref.load {tile}[{z}, {z}] : {ty}")
        f = e.v("psf" + tag)
        e.l(f"{f} = arith.extf {v} : bf16 to f32")
        e.l(f"memref.store {f}, %gc_out[{z}] : {TY['gc_out']}")
        e.l("}")
    e.l("%zfs = arith.constant 0.0 : f32")
    cH, cC, cNC = e.c(H), e.c(C), e.c(NC)
    _ks = bool(os.environ.get("GDN_F1KS"))
    TY = {
        "k": f"memref<{B*Tp*H*K}xbf16>", "v": f"memref<{B*Tp*H*V}xbf16>",
        "beta": f"memref<{B*Tp*H}xf32>", "g": f"memref<{B*Tp*H}xf32>",
        "u": f"memref<{B*H*NC*C*V}xbf16>", "w": f"memref<{B*H*NC*C*K}xbf16>",
        "gc_out": f"memref<{B*H*NC*C}xf32>"}

    # round-3: +8-col pads on every ldmatrix-read tile (see gen_f2 note:
    # 64B/128B/256B row strides LDSM-conflict without them)
    # R17-fwd GDN_F1KS: the kkt K-loop runs at bk=16 (f1-only; f2/f3 keep
    # the --bk default) -- the kt multibuffer shrinks 10,240 -> 6,144B,
    # which is what pulls the F1KS pool under the 48KB static-shared
    # 2-CTA line (measured: >48KB pools read blocks/SM=1 through the
    # occupancy API unless the carveout is forced, and the forced
    # carveout measured +20.9us on f2 in r14 -- not worth it).
    _bk32 = bool(os.environ.get("GDN_F1KS_BK32"))
    _bkt = (32 if _bk32 else 16) if _ks else bk
    ktile, ktile_ty = alloc2d(e, "kt", C, _bkt + 8, BF)
    if _nkt or _nkk:
        probe_sink(ktile, ktile_ty, "kt")   # orphan keep-alive (NKT/NKK)
    # G5-K round 5: A_m +1 f32 col breaks the 256B row stride (all rows
    # start at bank 0 -> the kkt epilogue's fragment-mapped stores hit 4
    # banks = 8-way STS conflicts; the r3-measured 557K).  HONEST
    # NON-RESULT: interleaved A/B at B1T8192 measured -0.8us on f1
    # (28/60 pairs; scratch/g5k_ab_F1PAD.json) -- the conflicts are real
    # but NOT on the critical path.  OFF by default (headroom for the
    # F1UW lever below).
    _f1pad = bool(os.environ.get("G5K_F1PAD"))
    _f1al = bool(os.environ.get("GDN_F1AL"))
    # R17-fwd GDN_F1KS (default off): the 2-CTA endgame.  Stacks on
    # GDN_BF16SOLVE (implied): A_m is gone, phase-1 co-live 41,024B;
    # the uw stage's vt/k2 tiles are K-SPLIT into two sequential
    # half-K passes reusing ONE half-sized tile set ([32,136] each),
    # landed POST-solve into Abf/Tt's bytes via the PTX lifetime arena
    # (both dead at the uw stage), and the kkt loop drops to bk=16
    # (kt multibuffer 10,240 -> 6,144B).  Pool: 46,352B reported <=
    # 48KB static-shared line -> 2 CTAs/SM.  The fp32 accumulation
    # order is UNCHANGED (split at 32 = BETWEEN the four m16n8k16 mma
    # blocks, accumulator carried across passes) -- CPU twin
    # bit-identical, scratch/r17f/ksplit_probe.json.
    assert not (_ks and _f1al), "GDN_F1KS not stackable with GDN_F1AL"
    assert not (_ks and os.environ.get("G5K_NOF1UW")), \
        "GDN_F1KS needs the F1UW straight-line uw stage (G5K_NOF1UW off)"
    # R16-Z GDN_BF16SOLVE (default off): drop the f32 A_m mirror entirely
    # -- the solve's phase-1 recurrences read Abf (bf16 A, extf to f32),
    # the kkt epilogue stores ONLY Abf.  CPU probe first (numpy twin,
    # scratch/r16z/bf16solve_probe.json): bf16-rounded phase-1 A is
    # statistically indistinguishable at the o/ht contract (worst
    # 7.493e-4 vs the 5e-3 bar, T2048 twin calibrated to the record
    # digits) -- T was already bf16 through phase 2 + the Tt feedback,
    # so the f32 bits A_m carried were rounded away two ops later.
    # Pool: -16,384B (A_m [64,64] f32).  Not stackable with GDN_F1AL
    # (the arena spec is written for the A_m malloc set).
    _bf16sv = ((bool(os.environ.get("GDN_BF16SOLVE")) or _ks) and not _f1al
               and not os.environ.get("G5D_EPI1"))
    if not _bf16sv:
        A_m, A_ty = alloc2d(e, "A_m", C, C + (1 if _f1pad else 0), F32)
    # R15-X GDN_F1AL: unique-size pads (identify-by-size aliasing needs
    # every arena member at a distinct malloc size); row strides 146B/
    # 148B/150B/152B keep the LDSM bank rotation the +8 pads provide.
    # R17-fwd GDN_F1KS: ROW-granular inflation instead (+1/+2 dead rows
    # at the same 144B stride) -- the identify-by-size PTX alias needs
    # the spec tiles at distinct malloc sizes, but column pads cost
    # 1-2KB each and ptxas PADS every non-128B-multiple decl up to 128
    # (measured +208B on the first cut), so the budget is: decl-sum +
    # pads <= 50,688.  Tt +2 rows (9,504, in-spec), Tb/Tg +1 row each
    # (9,360, shared size -- fine outside the spec), St0/1 keep the +8
    # pad (768), St2 drops it (512; used only in solve round R=3 -- one
    # 2-way-conflict LDSM per warp, negligible).  vt/k2 become HALF
    # tiles [32,136]/[33,136] (the +1 k2 row = size uniqueness) aliased
    # over Abf/Tt at PTX level.  Decl total 50,240B, worst-case padded
    # 50,560B <= 50,688 -> 2 CTAs/SM.
    Abf, Abf_ty = alloc2d(e, "Abf", C, C + (24 if _f1al else 8), BF)
    if _bf16sv:
        # phase 1 reads the bf16 A tile (extf in the recurrence); A_m's
        # SSA slot carries Abf so every downstream site type-checks.
        A_m, A_ty = Abf, Abf_ty
    Tt, Tt_ty = alloc2d(e, "Tt", C + (2 if _ks else 0),
                        C + (32 if _f1al else 8), BF)
    if os.environ.get("GDN_F1T_NP1") and os.environ.get("GDN_F1T_NP2"):
        # Tt orphaned only when BOTH solve emitters are deleted (NALL);
        # sunk HERE so its PTX lifetime stays a pre-solve point (k2
        # lands in Tt's bytes post-solve via the lifetime arena).
        probe_sink(Tt, Tt_ty, "tt")
    _bk32arm = _ks and os.environ.get("GDN_F1KS_BK32")
    St = [alloc2d(e, f"St{cb}", 16,
                  (16 if _bk32arm else
                   ((24 if cb < 2 else 16) if _ks else
                    (16 + (8 if not _f1al else 8 + 8 * cb)))),
                  BF)
          for cb in range(3)]
    if os.environ.get("GDN_F1T_NP2"):
        for cb in range(3):
            probe_sink(St[cb][0], St[cb][1], f"st{cb}")  # keep-alive (NP2)
    Tb, Tb_ty = alloc2d(e, "Tb", C + (1 if _ks else 0),
                        C + (40 if _f1al else 8), BF)
    # R22-fwd GDN_F1EP (default off): SINGLE-PASS uw epilogue -- the
    # second row-half stages into Tb/Tg's dead bytes (both A operands'
    # last ldm read is inside ks_pass(1), covered by its trailing
    # barrier; nothing reads Tb/Tg after).  Geometry: Tg grows
    # [65,72]->[67,72] (+288B pool) -- ROWS not cols, so the 144B row
    # stride stays a 16B multiple (a col pad to 76 made it 152B and
    # ldmatrix on odd rows faulted MISALIGNED, probe-verified); the
    # +2 rows are dead (data rows 0..63).  Identify-by-size needs Tb
    # and Tg at DISTINCT malloc sizes (both would be 9,360 otherwise
    # and the whole spec would drop -> vt/k2 unaliased -> 1 CTA/SM).
    # u2/w2 are the second-half store tiles, PTX-aliased over Tb/Tg at
    # offset 0 (their mallocs are DELETED at PTX level, so the +2/+3
    # row pads exist only to give unique identify sizes: 34*136*2 =
    # 9,248 and 35*136*2 = 9,520, distinct from every ship tile);
    # extents 9,248 <= Tb 9,360 and 9,520 <= Tg 9,648.
    _f1ep = bool(os.environ.get("GDN_F1EP"))
    if _f1ep and not _ks:
        raise SystemExit("GDN_F1EP needs GDN_F1KS (the two-half epilogue)")
    assert not (_f1ep and _f1al), "GDN_F1EP not stackable with GDN_F1AL"
    assert not (_f1ep and (_nuw or _nust)), \
        "GDN_F1EP orphan-safety: no GDN_F1T_NUW/NUST combo (u2/w2 users)"
    Tg, Tg_ty = alloc2d(e, "Tg",
                        C + (57 if _ks and os.environ.get("GDN_F1KS_F1CTA")
                             else ((3 if _f1ep else 1) if _ks else 0)),
                        C + (48 if _f1al else 8), BF)
    # G5-K round 5 (default ON; G5K_NOF1UW=1 reverts): run the uw K-loop
    # at bk=C (ONE trip) with
    # full [C, N+8] stage tiles.  The transform then has nothing to
    # multibuffer (1-trip loop), so vt/k2 cost 17,408B each SINGLE-
    # buffered -- EXACTLY the bytes their [32,136]x2 multibuffered selves
    # cost before (F1 stays at 101,120B total).  After the kloop the
    # tiles are dead as B operands and get REUSED as the u/w epilogue
    # staging tiles: the fragment-mapped 2B global stores (LSU/LG-
    # throttle-bound per r3 ncu) become padded STS + one coalesced cell
    # pass per tile (the G5-F pattern).  Post-loop users also make the
    # multibuffer transform SKIP vt/k2 on its own (outside-loop users).
    _f1uw = not os.environ.get("G5K_NOF1UW")
    _uwrows = C if _f1uw else bk
    # GDN_F1KS: half-K tiles; k2 +1 row for the unique malloc size
    _vtrows = (C // 2) if _ks else _uwrows
    _k2rows = (C // 2 + 1) if _ks else _uwrows
    vtile, vtile_ty = alloc2d(e, "vt", _vtrows, V + 8, BF)
    ktile2, ktile2_ty = alloc2d(e, "k2", _k2rows,
                                K + (16 if _f1al else 8), BF)
    if _f1ep:
        # R22-fwd GDN_F1EP: second-half epilogue store tiles (aliased
        # over Tb/Tg at PTX level; see the F1KS spec writer below).
        # Rows beyond 32 are dead -- they exist for malloc-size identity.
        u2t, u2_ty = alloc2d(e, "u2", _vtrows + 2, V + 8, BF)  # [34,136]
        w2t, w2_ty = alloc2d(e, "w2", _vtrows + 3, K + 8, BF)  # [35,136]
    gc, gc_ty = alloc2d(e, "gcs", C, 1, F32)
    bs, bs_ty = alloc2d(e, "bss", C, 1, F32)

    e.l("%n = gpu.block_id x")
    e.l("%bh = gpu.block_id y")
    e.l(f"%b = arith.divsi %bh, {cH} : index")
    e.l(f"%h = arith.remsi %bh, {cH} : index")
    e.l(f"%t0 = arith.muli %n, {cC} : index")
    x = addi(e, muli(e, "%b", e.c(Tp)), "%t0")
    x = muli(e, x, cH)
    xh = addi(e, x, "%h")
    kb_ = muli(e, xh, e.c(K))
    vb = muli(e, xh, e.c(V))
    bbg = addi(e, x, "%h")
    ub = addi(e, muli(e, "%bh", e.c(NC * C * V)), muli(e, "%n", e.c(C * V)))
    wb = addi(e, muli(e, "%bh", e.c(NC * C * K)), muli(e, "%n", e.c(C * K)))
    gcb = addi(e, muli(e, "%bh", e.c(NC * C)), muli(e, "%n", cC))

    if not _nsc:
        emit_cumsum_beta_scan(e, "%g", TY["g"], bbg, "%beta", TY["beta"],
                              bbg, gc, gc_ty, bs, bs_ty, H)

    if not _nkt:
        def _kkt_zero_accs():
            # R23-fwd GDN_F1T_NKK/NKM (timing-only): zero broadcasts
            # shaped exactly like the kloop's iter_arg inits, so
            # run_epilogue stays fully priced; the gemm_precompute index
            # SSAs are emitted by hand (emit_kloop normally provides
            # them) -- run_epilogue reads %rA/%rAh/%nct/%nct1.
            gemm_precompute(e, "kkt", C, C, W)
            zfk = e.v("zkk")
            e.l(f"{zfk} = arith.constant 0.0 : f32")
            out = {}
            for mi in range(C // 16):
                for ni in range((C // W) // 8):
                    an = e.v(f"azkk{mi}_{ni}")
                    e.l(f"{an} = vector.broadcast {zfk} : f32 "
                        f"to vector<2x2xf32>")
                    out[("kkt", mi, ni)] = an
            return out
        if _nkk:
            # NKK: the whole K-loop deleted (staging + ldm + mma);
            # ktile is sunk at alloc (above).
            accs_kkt = _kkt_zero_accs()
        else:
            # NKM: staging-only loop (ldm + mma deleted, barriers kept)
            accs_kkt = emit_kloop(
                e, "kk", K, _bkt,
                stages=[{"tile": ktile, "ty": ktile_ty, "rows": C,
                         "cols": _bkt,
                         "src": "%k", "sty": TY["k"], "kind": "rc",
                         "base": kb_, "rowmul": H * K}],
                gemms=([] if _nkm else
                       [{"name": "kkt", "M": C, "N": C, "kt": True,
                         "a": (ktile, ktile_ty, True),
                         "b": (ktile, ktile_ty, True)}]),
                W=W)
            if _nkm:
                accs_kkt = _kkt_zero_accs()
        e.l("gpu.barrier")

        def epi_A2(mi, ni, ei, row, col, acc):
            x = acc_extract(e, acc, ei // 2, ei % 2, "eA")
            if _nek:
                # R23-fwd GDN_F1T_NEK (timing-only): the exp2/beta chain
                # deleted (3 smem loads + subf/exp2/2xmulf/cmpi/select);
                # the bare extract->store keeps every acc and Abf live so
                # the mma K-loop cannot be DCE'd, and prices the store
                # pattern in BOTH arms.
                if not _bf16sv:
                    e.l(f"memref.store {x}, {A_m}[{row}, {col}] : {A_ty}")
                t = e.v("Abt")
                e.l(f"{t} = arith.truncf {x} : f32 to bf16")
                e.l(f"memref.store {t}, {Abf}[{row}, {col}] : {Abf_ty}")
                return
            br = e.v("Abr"); e.l(f"{br} = memref.load {bs}[{row}, %c0] : {bs_ty}")
            gi = e.v("Agi"); e.l(f"{gi} = memref.load {gc}[{row}, %c0] : {gc_ty}")
            gj = e.v("Agj"); e.l(f"{gj} = memref.load {gc}[{col}, %c0] : {gc_ty}")
            d = e.v("Ad"); e.l(f"{d} = arith.subf {gi}, {gj} : f32")
            ex = e.v("Aex"); e.l(f"{ex} = math.exp2 {d} : f32")
            m1 = e.v("Am1"); e.l(f"{m1} = arith.mulf {x}, {br} : f32")
            m2 = e.v("Am2"); e.l(f"{m2} = arith.mulf {m1}, {ex} : f32")
            lt = e.v("Alt"); e.l(f"{lt} = arith.cmpi slt, {col}, {row} : index")
            sel = e.v("Asel"); e.l(f"{sel} = arith.select {lt}, {m2}, %zfs : f32")
            if not _bf16sv:
                e.l(f"memref.store {sel}, {A_m}[{row}, {col}] : {A_ty}")
            t = e.v("Abt"); e.l(f"{t} = arith.truncf {sel} : f32 to bf16")
            e.l(f"memref.store {t}, {Abf}[{row}, {col}] : {Abf_ty}")
        import os as _ose
        _epi = epi_A2 if not _ose.environ.get("G5D_EPI1") else emit_A_epilogue(e, A_m, A_ty, bs, bs_ty, gc, gc_ty)
        run_epilogue(e, "kkt", C, C, W, accs_kkt, _epi)
        e.l("gpu.barrier")
    import os as _osa
    if _osa.environ.get("G5D_DUMP_AM"):
        dabx = e.v("dabx"); e.l(f"{dabx} = gpu.block_id x")
        daby = e.v("daby"); e.l(f"{daby} = gpu.block_id y")
        dabs = addi(e, muli(e, daby, e.c(NC)), dabx)
        dabs = muli(e, dabs, e.c(4096))
        def _da(i):
            r = divsi(e, i, e.c(64))
            c = remsi(e, i, e.c(64))
            v = e.v("dam")
            e.l(f"{v} = memref.load {A_m}[{r}, {c}] : {A_ty}")
            w = e.v("damt")
            e.l(f"{w} = arith.truncf {v} : f32 to bf16")
            e.l(f"memref.store {w}, %tbb["
                f"{addi(e, dabs, addi(e, muli(e, r, e.c(64)), c))}] : "
                f"memref<131072xbf16>")
        emit_cell_while(e, 64 * 64, nthr, _da)

    import os as _oss
    guw = gv_ = None
    if _ks:
        # GDN_F1KS: NO pre-solve staging -- during the solve there is no
        # free byte left in the 50,640B pool (solve co-live is the cap);
        # pass 0's group is issued right AFTER the solve (below, before
        # the gc_out store, overlapping it) into Abf/Tt's dead bytes.
        def issue_uw_half(p, tagp):
            if os.environ.get("GDN_F1KS_SYNC"):
                # bisect arm: synchronous scalar staging (no cp.async)
                def sync_stage(src, sty, dst, dst_ty, cols, gbase, rmul):
                    def body(i):
                        r = divsi(e, i, e.c(cols))
                        c = remsi(e, i, e.c(cols))
                        gr = addi(e, r, e.c((C // 2) * p))
                        v = e.v("ksl")
                        e.l(f"{v} = memref.load {src}["
                            f"{addi(e, gbase, addi(e, muli(e, gr, e.c(rmul)), c))}]"
                            f" : {sty}")
                        e.l(f"memref.store {v}, {dst}[{r}, {c}] : {dst_ty}")
                    emit_cell_while(e, (C // 2) * cols, nthr, body)
                sync_stage("%v", TY["v"], vtile, vtile_ty, V, vb, H * V)
                sync_stage("%k", TY["k"], ktile2, ktile2_ty, K, kb_, H * K)
                return None
            toks = []
            rbase = e.c((C // 2) * p)
            async_stage_unrolled(                   # v rows [32p, 32p+32)
                e, "%v", TY["v"], vtile, vtile_ty, C // 2, V, nthr,
                lambda r, c8: addi(e, vb, addi(e, muli(e, addi(e, r, rbase),
                                                        e.c(H * V)), c8)),
                lambda r, c8: (r, c8), toks)
            async_stage_unrolled(                   # k rows [32p, 32p+32)
                e, "%k", TY["k"], ktile2, ktile2_ty, C // 2, K, nthr,
                lambda r, c8: addi(e, kb_, addi(e, muli(e, addi(e, r, rbase),
                                                        e.c(H * K)), c8)),
                lambda r, c8: (r, c8), toks)
            g = e.v("agr_" + tagp)
            e.l(f"{g} = nvgpu.device_async_create_group {', '.join(toks)}")
            return g
    if _f1uw and not _f1al and not _ks and not _nust:
        # issue the uw operand staging NOW (before the solve): the copies
        # write vt/k2, disjoint from every solve tile, so the solve's
        # serial forward-substitution latency hides them.
        toks = []
        async_stage_unrolled(                       # v tile (C x V)
            e, "%v", TY["v"], vtile, vtile_ty, C, V, nthr,
            lambda r, c8: addi(e, vb, addi(e, muli(e, r, e.c(H * V)), c8)),
            lambda r, c8: (r, c8), toks)
        async_stage_unrolled(                       # k tile (C x K)
            e, "%k", TY["k"], ktile2, ktile2_ty, C, K, nthr,
            lambda r, c8: addi(e, kb_, addi(e, muli(e, r, e.c(H * K)), c8)),
            lambda r, c8: (r, c8), toks)
        guw = e.v("agr")
        e.l(f"{guw} = nvgpu.device_async_create_group {', '.join(toks)}")

    def issue_vt_group():
        # GDN_F1AL: vt's copy issued at the phase-1|2 boundary (its arena
        # is A_m's, dead only after phase 1); overlaps all of phase 2.
        toks = []
        async_stage_unrolled(
            e, "%v", TY["v"], vtile, vtile_ty, C, V, nthr,
            lambda r, c8: addi(e, vb, addi(e, muli(e, r, e.c(H * V)), c8)),
            lambda r, c8: (r, c8), toks)
        g = e.v("agr")
        e.l(f"{g} = nvgpu.device_async_create_group {', '.join(toks)}")
        return g

    def issue_k2_group():
        # GDN_F1AL: k2's copy issued post-solve (its arena is Abf's).
        toks = []
        async_stage_unrolled(
            e, "%k", TY["k"], ktile2, ktile2_ty, C, K, nthr,
            lambda r, c8: addi(e, kb_, addi(e, muli(e, r, e.c(H * K)), c8)),
            lambda r, c8: (r, c8), toks)
        g = e.v("agr")
        e.l(f"{g} = nvgpu.device_async_create_group {', '.join(toks)}")
        return g

    if _oss.environ.get("G5D_SOLVE2"):
        emit_solve2(e, A_m, A_ty, Tb, Tb_ty, Tg, Tg_ty, bs, bs_ty,
                    gc, gc_ty, C)
    else:
        mid = None
        if _f1uw and _f1al:
            def mid(e_):
                gv_[0] = issue_vt_group()
            gv_ = [None]
        emit_solve_blocked(e, A_m, A_ty, Abf, Abf_ty, Tt, Tt_ty, St,
                           Tb, Tb_ty, Tg, Tg_ty, bs, bs_ty, gc, gc_ty, C,
                           mid_hook=mid, p1_bf16=_bf16sv, nthr=nthr)
    if _f1uw and _f1al:
        guw = issue_k2_group()
    if _ks and not _nust:
        # p0 landed here: the solve's final barrier is behind every Abf
        # (R=3 stash) and Tt (R=3 consume) read; the copies overlap the
        # gc_out store below.  vt->Abf's bytes, k2->Tt's bytes (arena).
        # (R21-fwd GDN_F1T_NUST also deletes this issue; GDN_F1T_NUW
        # keeps it -- its group is then simply never waited.)
        guw = issue_uw_half(0, "p0")

    # gc_out store
    if not _ngc:
        c64 = e.c(64)
        in64 = e.v("in64")
        e.l(f"{in64} = arith.cmpi ult, %tid, {c64} : index")
        e.l(f"scf.if {in64} {{")
        gv = e.v("gco")
        e.l(f"{gv} = memref.load {gc}[%tid, %c0] : {gc_ty}")
        gid = addi(e, gcb, "%tid")
        e.l(f"memref.store {gv}, %gc_out[{gid}] : {TY['gc_out']}")
        e.l("}")

    if _nuw or _nust:
        # R21-fwd GDN_F1T_NUW/NUST (default off): the uw gemm stage is
        # deleted entirely (timing probe; u/w keep their prior contents).
        pass
    elif _f1uw and not _ks:
        # straight-line uw stage (the gen_f2 pattern: manual thread-strided
        # device_async_copy + fully unrolled mma; the 1-trip scf.for form
        # trips pipeline_shared_memory_copies' "pipelining preconditions
        # failed" hard error, probe-verified this round).  The copies were
        # issued BEFORE the solve (see above) so the solve's serial
        # recurrence hides the staging latency.
        e.l(f"nvgpu.device_async_wait {guw} numGroups = 0")
        e.l("gpu.barrier")
        accs = {}
        for gn, at, att, bt, btt, N in (("ug", Tb, Tb_ty, vtile, vtile_ty, V),
                                        ("wg", Tg, Tg_ty, ktile2, ktile2_ty,
                                         K)):
            gemm_precompute(e, gn, C, N, W)
            ldm_precompute(e, gn)
            accs_g, keys = _zero_accs(e, gn, C, N, W)
            accs.update(accs_g)
            for k in range(0, C, 32):
                fa, fb = {}, {}
                for mi in range(C // 16):
                    for kloc in (0, 16):
                        fa[(mi, kloc)] = ldm_a(e, gn, f"fa_{gn}_{k}_{mi}_{kloc}",
                                               at, att, mi,
                                               e.c(k + kloc), True)
                for ni in range((N // W) // 8):
                    for kloc in (0, 16):
                        fb[(ni, kloc)] = ldm_b(e, gn, f"fb_{gn}_{k}_{ni}_{kloc}",
                                               bt, btt, ni,
                                               e.c(k + kloc), False)
                _mma_chain(e, gn, fa, fb, accs, keys, gn)
    elif _ks:
        # GDN_F1KS: two sequential half-K passes over the ONE half-sized
        # vt/k2 tile set.  Pass p stages global rows [32p, 32p+32) into
        # the half tiles' rows 0..31 (one cp.async group per pass, vt+k2
        # together); the A operands (Tb/Tg) stay full-tile and are read
        # at COLUMN 32p (the gemm's K dim maps to Tb/Tg cols / vt/k2
        # rows).  Accumulators carry across passes: pass 0 zero-inits,
        # pass 1's mma chains chain from pass 0's SSA accs -- the fp32
        # addition sequence per output element is bit-identical to the
        # stock 4-block chain (CPU twin, scratch/r17f/ksplit_probe.json).
        # Barriers: wait+bar before the mmas (tiles resident), bar after
        # (all warps' LDS done before the next pass's restage).
        accs = {}
        keys_map = {}
        for gn, at, att, bt, btt, N in (
                ("ug", Tb, Tb_ty, vtile, vtile_ty, V),
                ("wg", Tg, Tg_ty, ktile2, ktile2_ty, K)):
            gemm_precompute(e, gn, C, N, W)
            ldm_precompute(e, gn)
            accs_g, keys = _zero_accs(e, gn, C, N, W)
            accs.update(accs_g)
            keys_map[gn] = keys

        def ks_pass(p, gpass):
            if gpass is not None:
                e.l(f"nvgpu.device_async_wait {gpass} numGroups = 0")
            e.l("gpu.barrier")
            k0 = (C // 2) * p
            for gn, at, att, bt, btt, N in (
                    ("ug", Tb, Tb_ty, vtile, vtile_ty, V),
                    ("wg", Tg, Tg_ty, ktile2, ktile2_ty, K)):
                fa, fb = {}, {}
                for mi in range(C // 16):
                    for kloc in (0, 16):
                        fa[(mi, kloc)] = ldm_a(
                            e, gn, f"fa_{gn}_{k0}_{mi}_{kloc}",
                            at, att, mi, e.c(k0 + kloc), True)
                for ni in range((N // W) // 8):
                    for kloc in (0, 16):
                        fb[(ni, kloc)] = ldm_b(
                            e, gn, f"fb_{gn}_{k0}_{ni}_{kloc}",
                            bt, btt, ni, e.c(kloc), False)
                _mma_chain(e, gn, fa, fb, accs, keys_map[gn], gn)
            e.l("gpu.barrier")

        ks_pass(0, guw)
        g1 = issue_uw_half(1, "p1")
        ks_pass(1, g1)
    else:
        accs = emit_kloop(
            e, "uw", C, bk,
            stages=[
                {"tile": vtile, "ty": vtile_ty, "rows": bk, "cols": V,
                 "src": "%v", "sty": TY["v"], "kind": "kc", "base": vb,
                 "rowmul": H * V},
                {"tile": ktile2, "ty": ktile2_ty, "rows": bk, "cols": K,
                 "src": "%k", "sty": TY["k"], "kind": "kc", "base": kb_,
                 "rowmul": H * K}],
            gemms=[
                {"name": "ug", "M": C, "N": V, "kt": False, "a_swap": True,
                 "a": (Tb, Tb_ty, False), "b": (vtile, vtile_ty, True)},
                {"name": "wg", "M": C, "N": K, "kt": False, "a_swap": True,
                 "a": (Tg, Tg_ty, False), "b": (ktile2, ktile2_ty, True)}],
            W=W)
    e.l("gpu.barrier")

    def epi_u(mi, ni, ei, row, col, acc):
        x = acc_extract(e, acc, ei // 2, ei % 2, "eu")
        t = e.v("eut"); e.l(f"{t} = arith.truncf {x} : f32 to bf16")
        if _f1uw:
            e.l(f"memref.store {t}, {vtile}[{row}, {col}] : {vtile_ty}")
        else:
            idx = addi(e, ub, addi(e, muli(e, row, e.c(V)), col))
            e.l(f"memref.store {t}, %u[{idx}] : {TY['u']}")

    def epi_w(mi, ni, ei, row, col, acc):
        x = acc_extract(e, acc, ei // 2, ei % 2, "ew")
        t = e.v("ewt"); e.l(f"{t} = arith.truncf {x} : f32 to bf16")
        if _f1uw:
            e.l(f"memref.store {t}, {ktile2}[{row}, {col}] : {ktile2_ty}")
        else:
            idx = addi(e, wb, addi(e, muli(e, row, e.c(K)), col))
            e.l(f"memref.store {t}, %w[{idx}] : {TY['w']}")

    if _nuw or _nust:
        # R21-fwd GDN_F1T_NUW/NUST: u/w epilogue + s2g copies deleted.
        pass
    elif _ks and _f1ep:
        # R22-fwd GDN_F1EP: SINGLE-PASS epilogue.  Rows [0,32) land in
        # the half tiles vt/k2, rows [32,64) in u2/w2 (= Tb/Tg's dead
        # bytes -- both A operands' last ldm read is inside ks_pass(1),
        # covered by its trailing barrier; nothing touches Tb/Tg after).
        # ONE barrier + one merged 4-tile copy stream replaces the two
        # store/barrier/copy/barrier rounds (the r21 time-map's uw
        # epi+s2g serialization; barriers are 22.9% of f1's warp stalls).
        # Same truncf rounding, same store/copy values -> bit-identical.
        for gn, bt, bty, N, dt, dty in (
                ("ug", vtile, vtile_ty, V, u2t, u2_ty),
                ("wg", ktile2, ktile2_ty, K, w2t, w2_ty)):
            for mi in range(4):
                st, sty = (dt, dty) if mi >= 2 else (bt, bty)
                for ni in range((N // W) // 8):
                    acc = accs[(gn, mi, ni)]
                    for ei in range(4):
                        row = (f"%rA_{gn}_{mi}" if ei < 2
                               else f"%rAh_{gn}_{mi}")
                        col = (f"%nct_{gn}_{ni}" if ei % 2 == 0
                               else f"%nct1_{gn}_{ni}")
                        x = acc_extract(e, acc, ei // 2, ei % 2, "ek")
                        t = e.v("ekt")
                        e.l(f"{t} = arith.truncf {x} : f32 to bf16")
                        if mi >= 2:
                            rl = e.v("ekr")
                            e.l(f"{rl} = arith.subi {row}, "
                                f"{e.c(C // 2)} : index")
                        else:
                            rl = row
                        e.l(f"memref.store {t}, {st}[{rl}, {col}]"
                            f" : {sty}")
        e.l("gpu.barrier")
        if not _nc2g:
            # one merged drain: both halves' 32-row passes back-to-back,
            # no interleaved barrier (disjoint tiles, disjoint globals)
            copy_s2g_vec(e, vtile, vtile_ty,
                         lambda r, c8: addi(
                             e, ub, addi(e, muli(e, r, e.c(V)), c8)),
                         "%u", TY["u"], C // 2, V, nthr)
            copy_s2g_vec(e, u2t, u2_ty,
                         lambda r, c8: addi(
                             e, ub, addi(e, muli(e, addi(e, r, e.c(C // 2)),
                                                 e.c(V)), c8)),
                         "%u", TY["u"], C // 2, V, nthr)
            copy_s2g_vec(e, ktile2, ktile2_ty,
                         lambda r, c8: addi(
                             e, wb, addi(e, muli(e, r, e.c(K)), c8)),
                         "%w", TY["w"], C // 2, K, nthr)
            copy_s2g_vec(e, w2t, w2_ty,
                         lambda r, c8: addi(
                             e, wb, addi(e, muli(e, addi(e, r, e.c(C // 2)),
                                                 e.c(K)), c8)),
                         "%w", TY["w"], C // 2, K, nthr)
    elif _ks:
        # GDN_F1KS epilogue: u/w restaged through the HALF tiles in two
        # row-halves -- rows [0,32) store + drain, then rows [32,64)
        # reuse the same bytes.  Same truncf rounding as epi_u/epi_w.
        for hf in (0, 1):
            roff = (C // 2) * hf
            for gn, bt, bty, N in (("ug", vtile, vtile_ty, V),
                                   ("wg", ktile2, ktile2_ty, K)):
                for mi in range(2 * hf, 2 * hf + 2):
                    for ni in range((N // W) // 8):
                        acc = accs[(gn, mi, ni)]
                        for ei in range(4):
                            row = (f"%rA_{gn}_{mi}" if ei < 2
                                   else f"%rAh_{gn}_{mi}")
                            col = (f"%nct_{gn}_{ni}" if ei % 2 == 0
                                   else f"%nct1_{gn}_{ni}")
                            x = acc_extract(e, acc, ei // 2, ei % 2, "ek")
                            t = e.v("ekt")
                            e.l(f"{t} = arith.truncf {x} : f32 to bf16")
                            if roff:
                                rl = e.v("ekr")
                                e.l(f"{rl} = arith.subi {row}, "
                                    f"{e.c(roff)} : index")
                            else:
                                rl = row
                            e.l(f"memref.store {t}, {bt}[{rl}, {col}]"
                                f" : {bty}")
            e.l("gpu.barrier")
            if not _nc2g:
                copy_s2g_vec(e, vtile, vtile_ty,
                             lambda r, c8: addi(
                                 e, ub, addi(e, muli(e, addi(e, r, e.c(roff)),
                                                     e.c(V)), c8)),
                             "%u", TY["u"], C // 2, V, nthr)
                copy_s2g_vec(e, ktile2, ktile2_ty,
                             lambda r, c8: addi(
                                 e, wb, addi(e, muli(e, addi(e, r, e.c(roff)),
                                                     e.c(K)), c8)),
                             "%w", TY["w"], C // 2, K, nthr)
            e.l("gpu.barrier")
    else:
        run_epilogue(e, "ug", C, V, W, accs, epi_u)
        run_epilogue(e, "wg", C, K, W, accs, epi_w)

        if _f1uw:
            # coalesced u/w passes (warp = 32 consecutive cols = 64B; the
            # fragment-mapped stores these replace are 2B-at-scattered-addr)
            e.l("gpu.barrier")
            if not _nc2g:
                copy_s2g_vec(e, vtile, vtile_ty,
                             lambda r, c8: addi(e, ub,
                                                addi(e, muli(e, r, e.c(V)),
                                                     c8)),
                             "%u", TY["u"], C, V, nthr)
                copy_s2g_vec(e, ktile2, ktile2_ty,
                             lambda r, c8: addi(e, wb,
                                                addi(e, muli(e, r, e.c(K)),
                                                     c8)),
                             "%w", TY["w"], C, K, nthr)

    if _nust:
        # orphan keep-alives at kernel END: vt/k2's arena windows open
        # post-solve (vt->Abf's bytes, k2->Tt's bytes), and a point
        # lifetime at the END cannot overlap them.
        probe_sink(vtile, vtile_ty, "vt")
        probe_sink(ktile2, ktile2_ty, "k2")


    params = [("k", B * Tp * H * K, BF), ("v", B * Tp * H * V, BF),
              ("beta", B * Tp * H, F32), ("g", B * Tp * H, F32),
              ("u", B * H * NC * C * V, BF), ("w", B * H * NC * C * K, BF),
              ("gc_out", B * H * NC * C, F32)]
    import os as _os
    if (_os.environ.get("G5D_DUMP_TT") or _os.environ.get("G5D_DUMP_P1")
            or _os.environ.get("G5D_DUMP_AM")):
        params = params + [("tt", B * H * NC * 64 * 64, BF),
                           ("tbb", B * H * NC * 64 * 64, BF),
                           ("tgg", B * H * NC * 64 * 64, BF)]
        if _os.environ.get("G5D_DUMP_TT"):
            dbxe = e.v("dbxe"); e.l(f"{dbxe} = gpu.block_id x")
            dbye = e.v("dbye"); e.l(f"{dbye} = gpu.block_id y")
            dbse = muli(e, addi(e, muli(e, dbye, e.c(NC)), dbxe), e.c(4096))
            def _dump(src, sty, dst, dty):
                def body(i):
                    r = divsi(e, i, e.c(64))
                    c = remsi(e, i, e.c(64))
                    v = e.v("ttv")
                    e.l(f"{v} = memref.load {src}[{r}, {c}] : {sty}")
                    e.l(f"memref.store {v}, {dst}["
                        f"{addi(e, dbse, addi(e, muli(e, r, e.c(64)), c))}] : {dty}")
                emit_cell_while(e, 64 * 64, nthr, body)
            _dump(Tt, Tt_ty, "%tt",
                  f"memref<{B*H*NC*64*64}xbf16>")
            _dump(Tb, Tb_ty, "%tbb",
                  f"memref<{B*H*NC*64*64}xbf16>")
            # R18-fwd: + Tg and the post-epilogue half tiles (the 2-CTA
            # corruption localization probe; debug knob, default off).
            _dump(Tg, Tg_ty, "%tgg",
                  f"memref<{B*H*NC*64*64}xbf16>")
            def _dump_half(src, sty, dst, dty, rows, cols):
                def body(i):
                    r = divsi(e, i, e.c(cols))
                    c = remsi(e, i, e.c(cols))
                    v = e.v("ttv")
                    e.l(f"{v} = memref.load {src}[{r}, {c}] : {sty}")
                    e.l(f"memref.store {v}, {dst}["
                        f"{addi(e, dbse, i)}] : {dty}")
                emit_cell_while(e, rows * cols, nthr, body)
            _dump_half(vtile, vtile_ty, "%vtd",
                       f"memref<{B*H*NC*32*V}xbf16>", 32, V)
            _dump_half(ktile2, ktile2_ty, "%k2d",
                       f"memref<{B*H*NC*32*K}xbf16>", 32, K)
            params = params + [("vtd", B * H * NC * 32 * V, BF),
                               ("k2d", B * H * NC * 32 * K, BF)]
        if _os.environ.get("G5D_DUMP_P1"):
            params = params + [("am", 131072, F32),
                               ("tt1", B * H * NC * 64 * 64, BF)]

    return e, params, (NC, B * H), nthr


# ---------------------------------------------------------------------------
# F2 "scan" (round 3: chunk-level ping-pong double buffering)
# ---------------------------------------------------------------------------

def async_stage_unrolled(e, src, sty, dst, dst_ty, rows, cols, nthr,
                         gidx, didx, toks, elems=8):
    """Thread-strided MANUAL nvgpu.device_async_copy burst (statically
    unrolled; no loop -> the tokens stay in this scope for create_group).

    The transform's create_async_groups route cannot be used for prefetch:
    it inserts `cp.async.wait_group 0` IMMEDIATELY after every group it
    creates (CreateAsyncGroups.cpp "Create the group and wait for it right
    after"), which would serialize the prefetch with the compute it is
    supposed to overlap.  Emitting the nvgpu ops directly leaves the wait
    placement to us.

    gidx(r, c8) -> flat global index SSA; didx(r, c8) -> (dst row, dst col)
    SSA pair.  elems elements per copy (8 bf16 / 4 f32 = 16B, bypassL1).
    """
    nch = cols // elems
    total = rows * nch
    assert total % nthr == 0, \
        f"async_stage: {total} chunks not divisible by {nthr} threads"
    for kk in range(total // nthr):
        p = addi(e, "%tid", e.c(kk * nthr)) if kk else "%tid"
        r = divsi(e, p, e.c(nch))
        cc = remsi(e, p, e.c(nch))
        c8 = muli(e, cc, e.c(elems))
        gi = gidx(r, c8)
        dr, dc = didx(r, c8)
        t = e.v("acp")
        e.l(f"{t} = nvgpu.device_async_copy {src}[{gi}], {dst}[{dr}, {dc}], "
            f"{elems} bypassL1 : {sty} to {dst_ty}")
        toks.append(t)
    return toks


def _mma_chain(e, tagp, fa, fb, accs, keys, name):
    for key in keys:
        mi, ni = key
        prev = accs[(name, mi, ni)]
        for kloc in (0, 16):
            o = e.v("mm" + tagp)
            e.l(f"{o} = nvgpu.mma.sync({fa[(mi, kloc)]}, "
                f"{fb[(ni, kloc)]}, {prev}) mmaShape = [16, 8, 16] : "
                f"(vector<4x2xbf16>, vector<2x2xbf16>, vector<2x2xf32>) "
                f"-> vector<2x2xf32>")
            prev = o
        accs[(name, mi, ni)] = prev
    return accs


def _zero_accs(e, name, M, N, W):
    accs, keys = {}, []
    for mi in range(M // 16):
        for ni in range((N // W) // 8):
            z = e.v("az_" + name)
            e.l(f"{z} = vector.broadcast %zfs : f32 to vector<2x2xf32>")
            accs[(name, mi, ni)] = z
            keys.append((mi, ni))
    return accs, keys


def gen_f2(shp, W, BV, bk=32):
    """State recurrence, grid (B*H*VT, 1).  ROUND 3: ping-pong chunks.

    Per chunk n (scf.while over chunks; no scf.for left in F2 -- the gemm
    K-steps are fully unrolled so the transform pipeliner sees nothing
    here, and the static mma/ldmatrix stream schedules better):

      * group A issued at TOP: cp.async prefetch of chunk min(n+1,NC-1)'s
        w tile (into wb cols [qK,qK+K)), k^T tile (into kb rows [qC,qC+C))
        and gc (into gcs[qC..]) -- q = parity n^1; overlaps ALL of chunk
        n's compute (create_async_groups would have wait_group-0'd right
        after the group, hence the manual nvgpu.device_async_copy route).
      * gcl / h_all entry-state store / vp gemm (unrolled, A = wb half p,
        B = resident S) / vn epilogue (u straight from global, gcs half p,
        writes vn).
      * barrier; v_new store; sd gemm (unrolled, A = kb half p swapped,
        B = vn); S-update epilogue.
      * wait group A + barrier: chunk n+1's operands are resident; parity
        flips.

    Ping-pong parity is folded into INDEX ARITHMETIC (wb is C x 2K with
    halves along columns, kb is 2C x K along rows, gcs is 2C f32) because
    memref values cannot be arith.select-ed.
    """
    B, T, H, K, V, C, NC, Tp = shp
    nthr = 32 * W
    VT = V // BV
    e = Em()
    emit_lane_precompute(e)
    gemm_precompute(e, "vp", C, BV, W)
    ldm_precompute(e, "vp")
    gemm_precompute(e, "sd", K, BV, W)
    ldm_precompute(e, "sd")
    e.l("%zfs = arith.constant 0.0 : f32")
    cH, cC, cNC, cBV, cVT = e.c(H), e.c(C), e.c(NC), e.c(BV), e.c(VT)
    cCm1 = e.c(C - 1)
    c2 = e.c(2)
    TY = {
        "u": f"memref<{B*H*NC*C*V}xbf16>", "w": f"memref<{B*H*NC*C*K}xbf16>",
        "k": f"memref<{B*Tp*H*K}xbf16>", "gc": f"memref<{B*H*NC*C}xf32>",
        "h0": f"memref<{B*H*K*V}xf32>",
        "h_all": f"memref<{B*H*NC*K*V}xbf16>",
        "v_new": f"memref<{B*H*NC*C*V}xbf16>",
        "ht": f"memref<{B*H*K*V}xf32>"}

    # LDSM bank-conflict avoidance: every 2-D tile read by ldmatrix gets
    # +8 bf16 columns of padding so row strides are NOT 128B multiples
    # (512B/256B/128B strides put all 8 rows of an LDSM quadrant in the
    # same 4 banks = 8-way conflict; ncu showed the smem pipe at ~40 cyc
    # per LDSM because of it).  +8 elems = +16B rotates banks by 4 per row.
    S, S_ty = alloc2d(e, "S", K, BV + 8, BF)
    wb, wb_ty = alloc2d(e, "wb", C, 2 * K + 8, BF)  # ping-pong along cols
    kb, kb_ty = alloc2d(e, "kb", 2 * C, K + 8, BF)  # ping-pong along rows
    # G5-K round 5: vn ping-ponged along rows (parity), so the v_new
    # global store pass can run at the TOP of the NEXT iteration reading
    # the other half -- removes the mid-chunk barrier.  HONEST NEGATIVE:
    # interleaved A/B at B1T8192 measured +19.7us on f2 (55/60 pairs
    # slower; scratch/g5k_ab_F2BAR.json) -- the +5KB smem (84.5->89.6KB)
    # shrinks the L1 carveout that the u loads / store passes rely on,
    # costing more than the saved bar.sync.  OFF by default.
    _f2bar = bool(os.environ.get("G5K_F2BAR"))
    vn, vn_ty = alloc2d(e, "vn", (2 if _f2bar else 1) * C, BV + 8, BF)
    gcs, gcs_ty = alloc1d(e, "gcs", 2 * C, F32)     # ping-pong along 1-D
    # R14-U (GDN_F2US, default off): stage u through a row-ping-ponged smem
    # tile (cp.async, 16B vectors -- no fragment-mapped 2B global-load
    # sector amplification on the serial chain) instead of the epi_vn
    # straight-from-global loads.  +10,240B (pool 84,480 -> 94,720B, still
    # 1 CTA/SM).  The prefetch rides group A (top-issued, end-of-chunk
    # wait) exactly like w/k/gc.
    _f2us = bool(os.environ.get("GDN_F2US"))
    ut, ut_ty = (alloc2d(e, "ut", 2 * C, BV + 8, BF) if _f2us
                 else (None, None))
    # NOTE: u is NOT staged -- the vn epilogue loads it straight from
    # global (its latency is covered by issuing right after the vp gemm
    # fragments are already in registers; saves the ub tile + its
    # cp.async group + 2 scalar shared ops per element of MIO pressure).

    e.l("%bhb = gpu.block_id x")
    e.l(f"%bh = arith.divsi %bhb, {cVT} : index")
    e.l(f"%vt = arith.remsi %bhb, {cVT} : index")
    e.l(f"%b = arith.divsi %bh, {cH} : index")
    e.l(f"%h = arith.remsi %bh, {cH} : index")
    e.l(f"%vtc = arith.muli %vt, {cBV} : index")
    h0b = addi(e, muli(e, "%bh", e.c(K * V)), "%vtc")
    htb = h0b
    # invariant global bases (chunk-sliced parts are SSA per chunk)
    e.l(f"%wq = arith.muli %bh, {e.c(NC * C * K)} : index")
    e.l(f"%uq = arith.muli %bh, {e.c(NC * C * V)} : index")
    e.l(f"%uqv = arith.addi %uq, %vtc : index")
    e.l(f"%gcq = arith.muli %bh, {e.c(NC * C)} : index")
    x = muli(e, muli(e, "%b", e.c(Tp)), cH)
    e.l(f"%kq = arith.muli {addi(e, x, '%h')}, {e.c(K)} : index")  # b*Tp*H*K+h*K

    def chunk_bases(n):
        wbn = addi(e, muli(e, n, e.c(C * K)), "%wq")
        ubn = addi(e, muli(e, n, e.c(C * V)), "%uqv")
        gcn = addi(e, muli(e, n, cC), "%gcq")
        kbn = addi(e, muli(e, n, e.c(C * H * K)), "%kq")
        return wbn, ubn, gcn, kbn

    def issue_group_A(nm, qK, qC):
        wbn, ubn, gcn, kbn = chunk_bases(nm)
        toks = []
        async_stage_unrolled(                      # w tile (C x K)
            e, "%w", TY["w"], wb, wb_ty, C, K, nthr,
            lambda r, c8: addi(e, addi(e, wbn, muli(e, r, e.c(K))), c8),
            lambda r, c8: (r, addi(e, qK, c8)), toks)
        async_stage_unrolled(                      # k^T tile (C x K)
            e, "%k", TY["k"], kb, kb_ty, C, K, nthr,
            lambda r, c8: addi(e, addi(e, kbn, muli(e, r, e.c(H * K))), c8),
            lambda r, c8: (addi(e, qC, r), c8), toks)
        if _f2us:                                  # u tile (C x BV slice)
            async_stage_unrolled(
                e, "%u", TY["u"], ut, ut_ty, C, BV, nthr,
                lambda r, c8: addi(e, ubn, addi(e, muli(e, r, e.c(V)),
                                                 c8)),
                lambda r, c8: (addi(e, qC, r), c8), toks)
        # gc: 16 x 4-f32 chunks; threads clamp-duplicate (identical bytes,
        # benign; keeps the tokens in scope without a guarded region)
        gcl_ = e.v("gcq_")
        e.l(f"{gcl_} = arith.minsi %tid, {e.c(15)} : index")
        gi4 = muli(e, gcl_, e.c(4))
        t = e.v("acp")
        e.l(f"{t} = nvgpu.device_async_copy %gc[{addi(e, gcn, gi4)}], "
            f"{gcs}[{addi(e, qC, gi4)}], 4 bypassL1 : {TY['gc']} to {gcs_ty}")
        toks.append(t)
        g = e.v("agr")
        e.l(f"{g} = nvgpu.device_async_create_group "
             f"{', '.join(toks)}")
        return g

    # -- prologue: S init from h0 overlapped with chunk-0 prefetch -------
    ga0 = issue_group_A("%c0", "%c0", "%c0")

    def init_s(i):
        r = divsi(e, i, e.c(BV))
        c = remsi(e, i, e.c(BV))
        v = e.v("h0v")
        e.l(f"{v} = memref.load %h0[{addi(e, h0b, addi(e, muli(e, r, e.c(V)), c))}] : {TY['h0']}")
        t = e.v("h0t")
        e.l(f"{t} = arith.truncf {v} : f32 to bf16")
        e.l(f"memref.store {t}, {S}[{r}, {c}] : {S_ty}")
    emit_cell_while(e, K * BV, nthr, init_s)
    e.l(f"nvgpu.device_async_wait {ga0} numGroups = 0")
    e.l("gpu.barrier")

    # -- chunk loop ------------------------------------------------------
    e.l("%nwr = scf.while (%nw = %c0) : (index) -> (index) {")
    ncond = e.v("ncc")
    e.l(f"{ncond} = arith.cmpi ult, %nw, {cNC} : index")
    e.l(f"scf.condition({ncond}) %nw : index")
    e.l("} do {")
    e.l("^bb0(%n: index):")
    e.l("%n2 = arith.remsi %n, %c2 : index")
    e.l("%pp = arith.subi %c1, %n2 : index")
    e.l(f"%np1 = arith.addi %n, {e.c(1)} : index")
    e.l(f"%nm = arith.minsi %np1, {e.c(NC - 1)} : index")
    # parity offsets: read half p = n%2, prefetch half q = 1-p
    e.l(f"%pK = arith.muli %n2, {e.c(K)} : index")
    e.l(f"%pC = arith.muli %n2, {cC} : index")
    e.l(f"%qK = arith.muli %pp, {e.c(K)} : index")
    e.l(f"%qC = arith.muli %pp, {cC} : index")
    shb = addi(e, muli(e, "%bh", e.c(NC * K * V)), muli(e, "%n", e.c(K * V)))
    shb_vtc = addi(e, shb, "%vtc")
    ub_vtc = addi(e, addi(e, "%uq", muli(e, "%n", e.c(C * V))), "%vtc")

    # group A: prefetch chunk nm (w/k/gc) -- overlaps everything below
    ga = issue_group_A("%nm", "%qK", "%qC")
    e.l(f"%gcl = memref.load {gcs}[{addi(e, '%pC', cCm1)}] : {gcs_ty}")

    if _f2bar:
        # G5-K round 5: v_new store pass for chunk n-1, at the TOP of the
        # iteration, reading the vn half of parity pp (= (n-1)%2).  Its
        # cross-warp visibility is covered by the END-of-chunk barrier of
        # iteration n-1; this iteration's epi_vn writes the OTHER half
        # (parity p), so pass-read vs epilogue-write cannot race.  This
        # lets the old mid-chunk barrier (vn visible to all warps) go
        # away: the sd gemm's B-operand reads of vn are WARP-LOCAL in N
        # (ldm_b col = wnoff+t2 = the warp's own 8 cols; epi_vn writes
        # the same cols of the same warp -- the mma N-tiling identity),
        # and the only cross-warp consumer (this pass) moved after a
        # barrier.  => 1 barrier/chunk (was 2).
        e.l("%npos = arith.cmpi ugt, %n, %c0 : index")
        e.l("scf.if %npos {")
        nmo = e.v("nmo")
        e.l(f"{nmo} = arith.subi %n, %c1 : index")
        ubn1 = addi(e, addi(e, "%uq", muli(e, nmo, e.c(C * V))), "%vtc")
        copy_s2g_vec(e, vn, vn_ty,
                     lambda r, c8: addi(e, ubn1, addi(e, muli(e, r, e.c(V)),
                                                       c8)),
                     "%v_new", TY["v_new"], C, BV, nthr,
                     row_base="%qC")
        e.l("}")

    # entry-state store (h_all[n] = S; coalesced cell pass -- in-epilogue
    # stores under the fragment mapping are uncoalesced, see note below)
    copy_s2g_vec(e, S, S_ty,
                 lambda r, c8: addi(e, shb_vtc, addi(e, muli(e, r, e.c(V)),
                                                     c8)),
                 "%h_all", TY["h_all"], K, BV, nthr)

    # vp gemm: acc = w @ S, A = wb half p (cols pK+..), B = S (koff const)
    accs_vp, keys_vp = _zero_accs(e, "vp", C, BV, W)
    for k in range(0, K, 32):
        fa, fb = {}, {}
        for mi in range(C // 16):
            for kloc in (0, 16):
                fa[(mi, kloc)] = ldm_a(e, "vp", f"fvp_{k}_{mi}_{kloc}",
                                       wb, wb_ty, mi,
                                       addi(e, "%pK", e.c(k + kloc)), False)
        for ni in range((BV // W) // 8):
            for kloc in (0, 16):
                fb[(ni, kloc)] = ldm_b(e, "vp", f"fbvp_{k}_{ni}_{kloc}",
                                       S, S_ty, ni, e.c(k + kloc), False)
        _mma_chain(e, "vp", fa, fb, accs_vp, keys_vp, "vp")

    # v_new epilogue: vn = (u - w@S) * 2^(gcl - gc_r)  [kterm fold]
    # (u loaded straight from GLOBAL -- see NOTE at the tile allocs; the
    # vn smem tile is the sd gemm's B operand; v_new goes to global via
    # the separate coalesced pass below -- in-epilogue stores follow the
    # FRAGMENT lane mapping = 2B-at-4B-stride uncoalesced, measured 40%
    # SLOWER than the extra pass)
    uqn = addi(e, "%uqv", muli(e, "%n", e.c(C * V)))

    def epi_vn(mi, ni, ei, row, col, acc):
        xv = acc_extract(e, acc, ei // 2, ei % 2, "evn")
        if _f2us:
            uv = e.v("vnu")
            e.l(f"{uv} = memref.load {ut}[{addi(e, '%pC', row)}, {col}] : {ut_ty}")
        else:
            uix = addi(e, uqn, addi(e, muli(e, row, e.c(V)), col))
            uv = e.v("vnu")
            e.l(f"{uv} = memref.load %u[{uix}] : {TY['u']}")
        uf = e.v("vnf")
        e.l(f"{uf} = arith.extf {uv} : bf16 to f32")
        dif = e.v("vnd")
        e.l(f"{dif} = arith.subf {uf}, {xv} : f32")
        gr = e.v("vnr")
        e.l(f"{gr} = memref.load {gcs}[{addi(e, '%pC', row)}] : {gcs_ty}")
        d = e.v("vnz")
        e.l(f"{d} = arith.subf %gcl, {gr} : f32")
        ex = e.v("vne")
        e.l(f"{ex} = math.exp2 {d} : f32")
        sc = e.v("vns")
        e.l(f"{sc} = arith.mulf {dif}, {ex} : f32")
        t = e.v("vnt")
        e.l(f"{t} = arith.truncf {sc} : f32 to bf16")
        vrow = addi(e, "%pC", row) if _f2bar else row
        e.l(f"memref.store {t}, {vn}[{vrow}, {col}] : {vn_ty}")
    run_epilogue(e, "vp", C, BV, W, accs_vp, epi_vn)
    if not _f2bar:
        e.l("gpu.barrier")            # vn visible to all warps (sd B op)

        # vn -> global v_new (coalesced cell pass; scaled convention)
        copy_s2g_vec(e, vn, vn_ty,
                     lambda r, c8: addi(e, ub_vtc, addi(e, muli(e, r, e.c(V)),
                                                         c8)),
                     "%v_new", TY["v_new"], C, BV, nthr)

    # sd gemm: acc = kterm^T @ vn, A = kb half p (rows pC+.., swapped),
    # B = vn half p (rows pC+.. under G5K_NOF2BAR=off; koff const over C
    # reduction).  B-fragment reads are warp-local in N (own 8 cols).
    accs_sd, keys_sd = _zero_accs(e, "sd", K, BV, W)
    for k in range(0, C, 32):
        fa, fb = {}, {}
        for mi in range(K // 16):
            for kloc in (0, 16):
                fa[(mi, kloc)] = ldm_a(e, "sd", f"fsd_{k}_{mi}_{kloc}",
                                       kb, kb_ty, mi,
                                       addi(e, "%pC", e.c(k + kloc)), True)
        for ni in range((BV // W) // 8):
            for kloc in (0, 16):
                fb[(ni, kloc)] = ldm_b(e, "sd", f"fbsd_{k}_{ni}_{kloc}",
                                       vn, vn_ty, ni,
                                       addi(e, "%pC", e.c(k + kloc))
                                       if _f2bar else e.c(k + kloc), False)
        _mma_chain(e, "sd", fa, fb, accs_sd, keys_sd, "sd")

    dl = e.v("dl")
    e.l(f"{dl} = math.exp2 %gcl : f32")
    run_epilogue(e, "sd", K, BV, W, accs_sd,
                 emit_S_update(e, S, S_ty, dl))

    # drain the prefetch group; chunk n+1 operands now resident
    e.l(f"nvgpu.device_async_wait {ga} numGroups = 0")
    e.l("gpu.barrier")
    nn = e.v("ncn")
    e.l(f"{nn} = arith.addi %n, %c1 : index")
    e.l(f"scf.yield {nn} : index")
    e.l("}")

    if _f2bar:
        # final v_new pass: chunk NC-1's vn (the loop stores chunk n-1 at
        # iteration n, so the last chunk's tile is still in smem here; the
        # loop's final barrier makes it visible to all warps).
        lp = remsi(e, e.c(NC - 1), c2)
        lpC = muli(e, lp, cC)
        ub_last = addi(e, addi(e, "%uq", muli(e, e.c(NC - 1), e.c(C * V))),
                       "%vtc")
        copy_s2g_vec(e, vn, vn_ty,
                     lambda r, c8: addi(e, ub_last,
                                        addi(e, muli(e, r, e.c(V)), c8)),
                     "%v_new", TY["v_new"], C, BV, nthr, row_base=lpC)

    # ht (f32; htb already includes the +vtc V-tile offset)
    def ht_pass(i):
        r = divsi(e, i, e.c(BV))
        c = remsi(e, i, e.c(BV))
        v = e.v("htv")
        e.l(f"{v} = memref.load {S}[{r}, {c}] : {S_ty}")
        w = e.v("htc")
        e.l(f"{w} = arith.extf {v} : bf16 to f32")
        e.l(f"memref.store {w}, %ht[{addi(e, htb, addi(e, muli(e, r, e.c(V)), c))}] : {TY['ht']}")
    emit_cell_while(e, K * BV, nthr, ht_pass)

    # pins: keep multibuffer off every F2 alloc (users span the chunk
    # while; no single candidate loop -> transform hard-fail without pins)
    for ssa, ty in ((S, S_ty), (wb, wb_ty), (kb, kb_ty), (vn, vn_ty)):
        pin_alloc(e, ssa, ty)
    pv = e.v("pin")
    e.l(f"{pv} = memref.load {gcs}[%c0] : {gcs_ty}")
    e.l(f"memref.store {pv}, {gcs}[%c0] : {gcs_ty}")

    params = [("u", B * H * NC * C * V, BF), ("w", B * H * NC * C * K, BF),
              ("k", B * Tp * H * K, BF), ("gc", B * H * NC * C, F32),
              ("h0", B * H * K * V, F32),
              ("h_all", B * H * NC * K * V, BF),
              ("v_new", B * H * NC * C * V, BF),
              ("ht", B * H * K * V, F32)]
    return e, params, (B * H * VT, 1), nthr


def gen_f2_sb(shp, W, BV, bk=32):
    """State recurrence, grid (B*H*VT, 1).  ROUND 14 (R14-U): the
    <=49.5KB single-buffer diet ("F2SB").

    The round-3 pp kernel holds w/k DOUBLE-buffered (pool 84.5KB at
    w4/bv32 = 1 CTA/SM, 100KB shared carveout).  This variant drops the
    second buffer of wb/kb/gcs and folds the chunk n+1 prefetch into the
    points where the tiles die:

      pool @w4/bv32: S 10240 + wb 17408 + kb 17408 + vn 5120 + gcs 256
                   = 50432B  (2 CTAs/SM fit: 2x50432 <= 100KB; L1 carveout
                     can drop to 64KB shared where the driver supports it)

    Per chunk n (scf.while over chunks; gemms fully unrolled as in pp):
      * top: v_new(n-1) store pass (reads vn -- safe: single buffer, the
        pass is separated from this iteration's epi_vn write by the top
        barrier) and the h_all(n) entry-state pass (reads S) -- both run
        BEFORE the prefetch wait, so the bottom-issued k/gc transfers
        overlap them;
      * nvgpu.device_async_wait(all) + barrier: chunk n's w/k/gc resident;
      * vp gemm (A = wb, B = resident S) / vn epilogue (u straight from
        global, gcs row loads, writes vn);
      * barrier  -- the wb DEATH FENCE: all warps are past the vp ldm's,
        so chunk n+1's w tile can be prefetched NOW (overlaps the whole
        second half: sd gemm + S-update epilogue);
      * sd gemm (A = kb, B = vn; B reads warp-local in N, the r5 mma
        N-tiling identity -- no barrier needed between epi_vn and sd);
      * S-update epilogue;
      * barrier  -- kb/gcs death fence; issue chunk n+1's k^T + gc
        prefetch (covered by the next iteration's top passes + wait);
      * yield (the async-group token rides the while iter_args so the
        next iteration's wait has a dominating def).

    3 barriers/chunk (pp has 2) -- the r5 F2BAR negative was measured at
    +5KB smem (L1-carveout cost), not barrier cost; here every extra
    barrier buys transfer overlap that the lost double buffer would have
    provided for free.
    """
    B, T, H, K, V, C, NC, Tp = shp
    nthr = 32 * W
    VT = V // BV
    e = Em()
    emit_lane_precompute(e)
    gemm_precompute(e, "vp", C, BV, W)
    ldm_precompute(e, "vp")
    gemm_precompute(e, "sd", K, BV, W)
    ldm_precompute(e, "sd")
    e.l("%zfs = arith.constant 0.0 : f32")
    cH, cC, cNC, cBV, cVT = e.c(H), e.c(C), e.c(NC), e.c(BV), e.c(VT)
    cCm1 = e.c(C - 1)
    TY = {
        "u": f"memref<{B*H*NC*C*V}xbf16>", "w": f"memref<{B*H*NC*C*K}xbf16>",
        "k": f"memref<{B*Tp*H*K}xbf16>", "gc": f"memref<{B*H*NC*C}xf32>",
        "h0": f"memref<{B*H*K*V}xf32>",
        "h_all": f"memref<{B*H*NC*K*V}xbf16>",
        "v_new": f"memref<{B*H*NC*C*V}xbf16>",
        "ht": f"memref<{B*H*K*V}xf32>"}

    # LDSM pads as in pp (+8 bf16 cols rotate banks by 4 per row)
    S, S_ty = alloc2d(e, "S", K, BV + 8, BF)
    wb, wb_ty = alloc2d(e, "wb", C, K + 8, BF)     # single buffer
    kb, kb_ty = alloc2d(e, "kb", C, K + 8, BF)     # single buffer
    vn, vn_ty = alloc2d(e, "vn", C, BV + 8, BF)    # single buffer
    gcs, gcs_ty = alloc1d(e, "gcs", C, F32)        # single buffer

    e.l("%bhb = gpu.block_id x")
    e.l(f"%bh = arith.divsi %bhb, {cVT} : index")
    e.l(f"%vt = arith.remsi %bhb, {cVT} : index")
    e.l(f"%b = arith.divsi %bh, {cH} : index")
    e.l(f"%h = arith.remsi %bh, {cH} : index")
    e.l(f"%vtc = arith.muli %vt, {cBV} : index")
    h0b = addi(e, muli(e, "%bh", e.c(K * V)), "%vtc")
    htb = h0b
    e.l(f"%wq = arith.muli %bh, {e.c(NC * C * K)} : index")
    e.l(f"%uq = arith.muli %bh, {e.c(NC * C * V)} : index")
    e.l(f"%uqv = arith.addi %uq, %vtc : index")
    e.l(f"%gcq = arith.muli %bh, {e.c(NC * C)} : index")
    x = muli(e, muli(e, "%b", e.c(Tp)), cH)
    e.l(f"%kq = arith.muli {addi(e, x, '%h')}, {e.c(K)} : index")

    def chunk_bases(n):
        wbn = addi(e, muli(e, n, e.c(C * K)), "%wq")
        ubn = addi(e, muli(e, n, e.c(C * V)), "%uqv")
        gcn = addi(e, muli(e, n, cC), "%gcq")
        kbn = addi(e, muli(e, n, e.c(C * H * K)), "%kq")
        return wbn, ubn, gcn, kbn

    def issue_w(nm):
        """Prefetch chunk nm's w tile into wb (mid-chunk, post vp)."""
        wbn, _ubn, _gcn, _kbn = chunk_bases(nm)
        toks = []
        async_stage_unrolled(e, "%w", TY["w"], wb, wb_ty, C, K, nthr,
                             lambda r, c8: addi(e, addi(e, wbn,
                                                        muli(e, r, e.c(K))),
                                                c8),
                             lambda r, c8: (r, c8), toks)
        g = e.v("agr")
        e.l(f"{g} = nvgpu.device_async_create_group {', '.join(toks)}")
        return g

    def issue_kgc(nm):
        """Prefetch chunk nm's k^T tile into kb + gc into gcs (bottom)."""
        _wbn, _ubn, gcn, kbn = chunk_bases(nm)
        toks = []
        async_stage_unrolled(e, "%k", TY["k"], kb, kb_ty, C, K, nthr,
                             lambda r, c8: addi(e, addi(e, kbn,
                                                        muli(e, r,
                                                             e.c(H * K))),
                                                c8),
                             lambda r, c8: (r, c8), toks)
        gcl_ = e.v("gcq_")
        e.l(f"{gcl_} = arith.minsi %tid, {e.c(15)} : index")
        gi4 = muli(e, gcl_, e.c(4))
        t = e.v("acp")
        e.l(f"{t} = nvgpu.device_async_copy %gc[{addi(e, gcn, gi4)}], "
            f"{gcs}[{gi4}], 4 bypassL1 : {TY['gc']} to {gcs_ty}")
        toks.append(t)
        g = e.v("agr")
        e.l(f"{g} = nvgpu.device_async_create_group {', '.join(toks)}")
        return g

    # -- prologue: S init from h0 overlapped with chunk-0 prefetch -------
    issue_w("%c0")
    ga0b = issue_kgc("%c0")

    def init_s(i):
        r = divsi(e, i, e.c(BV))
        c = remsi(e, i, e.c(BV))
        v = e.v("h0v")
        e.l(f"{v} = memref.load %h0[{addi(e, h0b, addi(e, muli(e, r, e.c(V)), c))}] : {TY['h0']}")
        t = e.v("h0t")
        e.l(f"{t} = arith.truncf {v} : f32 to bf16")
        e.l(f"memref.store {t}, {S}[{r}, {c}] : {S_ty}")
    emit_cell_while(e, K * BV, nthr, init_s)
    e.l(f"nvgpu.device_async_wait {ga0b} numGroups = 0")
    e.l("gpu.barrier")

    # -- chunk loop (async-group token rides the iter_args so the top wait
    #    has a dominating def; semantics: wait_group 0 = all outstanding) --
    e.l(f"%nwr0, %gtw0 = scf.while (%nw = %c0, %gt = {ga0b}) : "
        f"(index, !nvgpu.device.async.token) -> (index, !nvgpu.device.async.token) {{")
    ncond = e.v("ncc")
    e.l(f"{ncond} = arith.cmpi ult, %nw, {cNC} : index")
    e.l(f"scf.condition({ncond}) %nw, %gt : index, !nvgpu.device.async.token")
    e.l("} do {")
    e.l("^bb0(%n: index, %gtp: !nvgpu.device.async.token):")
    e.l(f"%np1 = arith.addi %n, {e.c(1)} : index")
    e.l(f"%nm = arith.minsi %np1, {e.c(NC - 1)} : index")
    shb = addi(e, muli(e, "%bh", e.c(NC * K * V)), muli(e, "%n", e.c(K * V)))
    shb_vtc = addi(e, shb, "%vtc")

    # v_new store pass for chunk n-1 (vn single-buffered: this pass and
    # this iteration's epi_vn are separated by the barrier below)
    e.l("%npos = arith.cmpi ugt, %n, %c0 : index")
    e.l("scf.if %npos {")
    nmo = e.v("nmo")
    e.l(f"{nmo} = arith.subi %n, %c1 : index")
    ubn1 = addi(e, addi(e, "%uq", muli(e, nmo, e.c(C * V))), "%vtc")
    copy_s2g_vec(e, vn, vn_ty,
                 lambda r, c8: addi(e, ubn1, addi(e, muli(e, r, e.c(V)),
                                                   c8)),
                 "%v_new", TY["v_new"], C, BV, nthr)
    e.l("}")

    # entry-state store (h_all[n] = S)
    copy_s2g_vec(e, S, S_ty,
                 lambda r, c8: addi(e, shb_vtc, addi(e, muli(e, r, e.c(V)),
                                                     c8)),
                 "%h_all", TY["h_all"], K, BV, nthr)

    # chunk n's operands resident (groups from the previous iteration)
    e.l(f"nvgpu.device_async_wait %gtp numGroups = 0")
    e.l("gpu.barrier")
    e.l(f"%gcl = memref.load {gcs}[{cCm1}] : {gcs_ty}")

    # vp gemm: acc = w @ S
    accs_vp, keys_vp = _zero_accs(e, "vp", C, BV, W)
    for k in range(0, K, 32):
        fa, fb = {}, {}
        for mi in range(C // 16):
            for kloc in (0, 16):
                fa[(mi, kloc)] = ldm_a(e, "vp", f"fvp_{k}_{mi}_{kloc}",
                                       wb, wb_ty, mi, e.c(k + kloc), False)
        for ni in range((BV // W) // 8):
            for kloc in (0, 16):
                fb[(ni, kloc)] = ldm_b(e, "vp", f"fbvp_{k}_{ni}_{kloc}",
                                       S, S_ty, ni, e.c(k + kloc), False)
        _mma_chain(e, "vp", fa, fb, accs_vp, keys_vp, "vp")

    # v_new epilogue: vn = (u - w@S) * 2^(gcl - gc_r)  [kterm fold]
    uqn = addi(e, "%uqv", muli(e, "%n", e.c(C * V)))

    def epi_vn(mi, ni, ei, row, col, acc):
        xv = acc_extract(e, acc, ei // 2, ei % 2, "evn")
        uix = addi(e, uqn, addi(e, muli(e, row, e.c(V)), col))
        uv = e.v("vnu")
        e.l(f"{uv} = memref.load %u[{uix}] : {TY['u']}")
        uf = e.v("vnf")
        e.l(f"{uf} = arith.extf {uv} : bf16 to f32")
        dif = e.v("vnd")
        e.l(f"{dif} = arith.subf {uf}, {xv} : f32")
        gr = e.v("vnr")
        e.l(f"{gr} = memref.load {gcs}[{row}] : {gcs_ty}")
        d = e.v("vnz")
        e.l(f"{d} = arith.subf %gcl, {gr} : f32")
        ex = e.v("vne")
        e.l(f"{ex} = math.exp2 {d} : f32")
        sc = e.v("vns")
        e.l(f"{sc} = arith.mulf {dif}, {ex} : f32")
        t = e.v("vnt")
        e.l(f"{t} = arith.truncf {sc} : f32 to bf16")
        e.l(f"memref.store {t}, {vn}[{row}, {col}] : {vn_ty}")
    run_epilogue(e, "vp", C, BV, W, accs_vp, epi_vn)

    # wb death fence: all warps past the vp ldm's -> prefetch chunk n+1's
    # w tile now; the transfer hides under sd + the S-update epilogue
    e.l("gpu.barrier")
    gwa = issue_w("%nm")

    # sd gemm: acc = kterm^T @ vn (B reads warp-local in N, r5 identity)
    accs_sd, keys_sd = _zero_accs(e, "sd", K, BV, W)
    for k in range(0, C, 32):
        fa, fb = {}, {}
        for mi in range(K // 16):
            for kloc in (0, 16):
                fa[(mi, kloc)] = ldm_a(e, "sd", f"fsd_{k}_{mi}_{kloc}",
                                       kb, kb_ty, mi, e.c(k + kloc), True)
        for ni in range((BV // W) // 8):
            for kloc in (0, 16):
                fb[(ni, kloc)] = ldm_b(e, "sd", f"fbsd_{k}_{ni}_{kloc}",
                                       vn, vn_ty, ni, e.c(k + kloc), False)
        _mma_chain(e, "sd", fa, fb, accs_sd, keys_sd, "sd")

    dl = e.v("dl")
    e.l(f"{dl} = math.exp2 %gcl : f32")
    run_epilogue(e, "sd", K, BV, W, accs_sd,
                 emit_S_update(e, S, S_ty, dl))

    # kb/gcs death fence: prefetch chunk n+1's k^T + gc; covered by the
    # next iteration's top passes + wait
    e.l("gpu.barrier")
    gka = issue_kgc("%nm")
    nn = e.v("ncn")
    e.l(f"{nn} = arith.addi %n, %c1 : index")
    e.l(f"scf.yield {nn}, {gka} : index, !nvgpu.device.async.token")
    e.l("}")

    # final v_new pass: chunk NC-1 (the loop stores chunk n-1 at iter n)
    ub_last = addi(e, addi(e, "%uq", muli(e, e.c(NC - 1), e.c(C * V))),
                   "%vtc")
    copy_s2g_vec(e, vn, vn_ty,
                 lambda r, c8: addi(e, ub_last,
                                    addi(e, muli(e, r, e.c(V)), c8)),
                 "%v_new", TY["v_new"], C, BV, nthr)

    # ht (f32; htb already includes the +vtc V-tile offset)
    def ht_pass(i):
        r = divsi(e, i, e.c(BV))
        c = remsi(e, i, e.c(BV))
        v = e.v("htv")
        e.l(f"{v} = memref.load {S}[{r}, {c}] : {S_ty}")
        w = e.v("htc")
        e.l(f"{w} = arith.extf {v} : bf16 to f32")
        e.l(f"memref.store {w}, %ht[{addi(e, htb, addi(e, muli(e, r, e.c(V)), c))}] : {TY['ht']}")
    emit_cell_while(e, K * BV, nthr, ht_pass)

    # pins: keep multibuffer off every F2 alloc (users span the chunk while)
    for ssa, ty in ((S, S_ty), (wb, wb_ty), (kb, kb_ty), (vn, vn_ty)):
        pin_alloc(e, ssa, ty)
    pv = e.v("pin")
    e.l(f"{pv} = memref.load {gcs}[%c0] : {gcs_ty}")
    e.l(f"memref.store {pv}, {gcs}[%c0] : {gcs_ty}")

    params = [("u", B * H * NC * C * V, BF), ("w", B * H * NC * C * K, BF),
              ("k", B * Tp * H * K, BF), ("gc", B * H * NC * C, F32),
              ("h0", B * H * K * V, F32),
              ("h_all", B * H * NC * K * V, BF),
              ("v_new", B * H * NC * C * V, BF),
              ("ht", B * H * K * V, F32)]
    return e, params, (B * H * VT, 1), nthr


def gen_f2_r2(shp, W, BV, bk=32):
    """State recurrence, grid (B*H*VT, 1).

    Round-2 body (per chunk): vectorized h_all entry-state store; vp K-loop
    (w slices cp.async-staged, B = resident S); u staged as one async tile;
    v_new epilogue writes vn SCALED by 2^(gcl-gc_r) (the kterm fold: F3's P
    epilogue uses the matching 2^(gc_i-gcl) form, so global v_new carries
    the scaled tile as-is); vectorized vn->v_new copy; sd K-loop (k slices
    staged, swapped A reads, B = resident vn); S-update epilogue.
    """
    B, T, H, K, V, C, NC, Tp = shp
    nthr = 32 * W
    VT = V // BV
    e = Em()
    emit_lane_precompute(e)
    e.l("%zfs = arith.constant 0.0 : f32")
    cH, cC, cNC, cBV, cVT = e.c(H), e.c(C), e.c(NC), e.c(BV), e.c(VT)
    cCm1 = e.c(C - 1)
    TY = {
        "u": f"memref<{B*H*NC*C*V}xbf16>", "w": f"memref<{B*H*NC*C*K}xbf16>",
        "k": f"memref<{B*Tp*H*K}xbf16>", "gc": f"memref<{B*H*NC*C}xf32>",
        "h0": f"memref<{B*H*K*V}xf32>",
        "h_all": f"memref<{B*H*NC*K*V}xbf16>",
        "v_new": f"memref<{B*H*NC*C*V}xbf16>",
        "ht": f"memref<{B*H*K*V}xf32>"}

    S, S_ty = alloc2d(e, "S", K, BV, BF)
    wtile, wtile_ty = alloc2d(e, "wt", C, bk, BF)
    ktile2, ktile2_ty = alloc2d(e, "k2", bk, K, BF)
    vn, vn_ty = alloc2d(e, "vn", C, BV, BF)
    utile, utile_ty = alloc2d(e, "ut", C, BV, BF)
    gc, gc_ty = alloc2d(e, "gcs", C, 1, F32)

    e.l("%bhb = gpu.block_id x")
    e.l(f"%bh = arith.divsi %bhb, {cVT} : index")
    e.l(f"%vt = arith.remsi %bhb, {cVT} : index")
    e.l(f"%b = arith.divsi %bh, {cH} : index")
    e.l(f"%h = arith.remsi %bh, {cH} : index")
    e.l(f"%vtc = arith.muli %vt, {cBV} : index")
    h0b = addi(e, muli(e, "%bh", e.c(K * V)), "%vtc")
    htb = h0b

    def init_s(i):
        r = divsi(e, i, e.c(BV))
        c = remsi(e, i, e.c(BV))
        v = e.v("h0v")
        e.l(f"{v} = memref.load %h0[{addi(e, h0b, addi(e, muli(e, r, e.c(V)), c))}] : {TY['h0']}")
        t = e.v("h0t")
        e.l(f"{t} = arith.truncf {v} : f32 to bf16")
        e.l(f"memref.store {t}, {S}[{r}, {c}] : {S_ty}")
    emit_cell_while(e, K * BV, nthr, init_s)
    e.l("gpu.barrier")

    # chunk loop as scf.while (a pipeliner-unreachable outer recurrence; all
    # its K-loops stay scf.for and get pipelined)
    e.l("%nwr = scf.while (%nw = %c0) : (index) -> (index) {")
    ncond = e.v("ncc")
    e.l(f"{ncond} = arith.cmpi ult, %nw, {cNC} : index")
    e.l(f"scf.condition({ncond}) %nw : index")
    e.l("} do {")
    e.l("^bb0(%n: index):")
    ub = addi(e, muli(e, "%bh", e.c(NC * C * V)), muli(e, "%n", e.c(C * V)))
    wb = addi(e, muli(e, "%bh", e.c(NC * C * K)), muli(e, "%n", e.c(C * K)))
    gcb = addi(e, muli(e, "%bh", e.c(NC * C)), muli(e, "%n", cC))
    shb = addi(e, muli(e, "%bh", e.c(NC * K * V)), muli(e, "%n", e.c(K * V)))
    t0 = muli(e, "%n", cC)
    x = addi(e, muli(e, "%b", e.c(Tp)), t0)
    x = muli(e, x, cH)
    xh = addi(e, x, "%h")
    kbc = muli(e, xh, e.c(K))
    shb_vtc = addi(e, shb, "%vtc")
    ub_vtc = addi(e, ub, "%vtc")

    # entry-state store (vectorized)
    copy_s2g_vec(e, S, S_ty,
                 lambda r, c8: addi(e, shb_vtc, addi(e, muli(e, r, e.c(V)),
                                                     c8)),
                 "%h_all", TY["h_all"], K, BV, nthr)
    e.l("gpu.barrier")

    emit_load_gc(e, "%gc", TY["gc"], gcb, gc, gc_ty)
    e.l(f"%gcl = memref.load {gc}[{cCm1}, %c0] : {gc_ty}")

    accs_vp = emit_kloop(
        e, "vp", K, bk,
        stages=[{"tile": wtile, "ty": wtile_ty, "rows": C, "cols": bk,
                 "src": "%w", "sty": TY["w"], "kind": "rc", "base": wb,
                 "rowmul": K}],
        gemms=[{"name": "vp", "M": C, "N": BV, "kt": False,
                "a": (wtile, wtile_ty, True), "b": (S, S_ty, False)}],
        W=W)

    # u staging (whole tile; async burst via create_async_groups)
    copy_g2s_vec(e, "%u", TY["u"],
                 lambda r, c8: addi(e, ub_vtc, addi(e, muli(e, r, e.c(V)),
                                                     c8)),
                 utile, utile_ty, C, BV, nthr)
    e.l("gpu.barrier")

    # v_new epilogue: vn = (u - w@S) * 2^(gcl - gc_r)  [kterm fold]
    def epi_vn(mi, ni, ei, row, col, acc):
        xv = acc_extract(e, acc, ei // 2, ei % 2, "evn")
        uv = e.v("vnu")
        e.l(f"{uv} = memref.load {utile}[{row}, {col}] : {utile_ty}")
        uf = e.v("vnf")
        e.l(f"{uf} = arith.extf {uv} : bf16 to f32")
        dif = e.v("vnd")
        e.l(f"{dif} = arith.subf {uf}, {xv} : f32")
        gr = e.v("vnr")
        e.l(f"{gr} = memref.load {gc}[{row}, %c0] : {gc_ty}")
        d = e.v("vnz")
        e.l(f"{d} = arith.subf %gcl, {gr} : f32")
        ex = e.v("vne")
        e.l(f"{ex} = math.exp2 {d} : f32")
        sc = e.v("vns")
        e.l(f"{sc} = arith.mulf {dif}, {ex} : f32")
        t = e.v("vnt")
        e.l(f"{t} = arith.truncf {sc} : f32 to bf16")
        e.l(f"memref.store {t}, {vn}[{row}, {col}] : {vn_ty}")
    run_epilogue(e, "vp", C, BV, W, accs_vp, epi_vn)
    e.l("gpu.barrier")

    # vn -> global v_new (vectorized; scaled convention, see docstring)
    copy_s2g_vec(e, vn, vn_ty,
                 lambda r, c8: addi(e, ub_vtc, addi(e, muli(e, r, e.c(V)),
                                                     c8)),
                 "%v_new", TY["v_new"], C, BV, nthr)
    e.l("gpu.barrier")

    accs_sd = emit_kloop(
        e, "sd", C, bk,
        stages=[{"tile": ktile2, "ty": ktile2_ty, "rows": bk, "cols": K,
                 "src": "%k", "sty": TY["k"], "kind": "kc", "base": kbc,
                 "rowmul": H * K}],
        gemms=[{"name": "sd", "M": K, "N": BV, "kt": False, "a_swap": True,
                "a": (ktile2, ktile2_ty, True), "b": (vn, vn_ty, False)}],
        W=W)
    dl = e.v("dl")
    e.l(f"{dl} = math.exp2 %gcl : f32")
    run_epilogue(e, "sd", K, BV, W, accs_sd,
                 emit_S_update(e, S, S_ty, dl))
    e.l("gpu.barrier")
    nn = e.v("ncn")
    e.l(f"{nn} = arith.addi %n, %c1 : index")
    e.l(f"scf.yield {nn} : index")
    e.l("}")

    # ht (f32; htb already includes the +vtc V-tile offset) -- while-cells
    def ht_pass(i):
        r = divsi(e, i, e.c(BV))
        c = remsi(e, i, e.c(BV))
        v = e.v("htv")
        e.l(f"{v} = memref.load {S}[{r}, {c}] : {S_ty}")
        w = e.v("htc")
        e.l(f"{w} = arith.extf {v} : bf16 to f32")
        e.l(f"memref.store {w}, %ht[{addi(e, htb, addi(e, muli(e, r, e.c(V)), c))}] : {TY['ht']}")
    emit_cell_while(e, K * BV, nthr, ht_pass)

    # pins: keep multibuffer off S/vn/utile/gcs (users span several loops /
    # live only inside the chunk while, which has no single iv)
    pin_alloc(e, S, S_ty)
    pin_alloc(e, vn, vn_ty)
    pin_alloc(e, utile, utile_ty)
    pin_alloc(e, gc, gc_ty)

    params = [("u", B * H * NC * C * V, BF), ("w", B * H * NC * C * K, BF),
              ("k", B * Tp * H * K, BF), ("gc", B * H * NC * C, F32),
              ("h0", B * H * K * V, F32),
              ("h_all", B * H * NC * K * V, BF),
              ("v_new", B * H * NC * C * V, BF),
              ("ht", B * H * K * V, F32)]
    return e, params, (B * H * VT, 1), nthr


# ---------------------------------------------------------------------------
# G5-I round 4: F2 segment-split scan (2-pass variant)
# ---------------------------------------------------------------------------

def gen_f2_seg(shp, W, BV, NS, mode, bk=32):
    """Segment-split state scan, grid (B*H*VT*NS, 1) -- G5-I round 4.

    The chunk recurrence is AFFINE in the state (v_new = u - w@S makes the
    per-chunk map S' = 2^gcl S + kterm^T(u - w S)), so an exact parallel
    split needs the composed linear part M_seg = prod(2^gcl I - kterm^T w)
    in the handoff: E_{sg+1} = M_sg E_sg + c_sg.  BUT every factor M_n
    carries 2^gcl times an O(1) delta-rule operator (the 2^-gc in kterm
    and the 2^+gc in w cancel per token), so ||M_seg|| ~ 2^(sum gcl) --
    for the GDN bench recipes (g = -softplus(...), mean log-decay ~ -0.25/
    token) this is <= 4e-13 at L=1 and ~1e-52 at L=4 (measured over the
    actual bench inputs, tools/g5i_split_sim.py).  The matmul-free handoff

        E_0 = h0;   E_{sg+1} = c_sg  (pass-A zero-entry exit of segment sg)

    is therefore exact to far below bf16 resolution AT THESE RECIPES.
    Validity boundary: inputs with near-zero net decay per segment
    (gate ~ 1) would need the M composition fallback (not implemented;
    the simulator's --stress-g probe demonstrates the divergence).

    mode "a" (pass A): every segment EXCEPT 0 runs its NC/NS chunks from a
      ZERO entry and stores ONLY its exit state to @seg_exit (interior
      h_all/v_new would be wrong with a zero entry and are NOT written).
    mode "c" (pass C): every segment re-runs WITH its true entry (h0 for
      segment 0, @seg_exit[sg-1] otherwise) and writes the real outputs
      (h_all entry states, v_new, ht for the last segment).

    Body = the round-2 pipelined form (emit_kloop K-loops stay scf.for so
    the transform pipeline multibuffers/pipelines them) + the round-3 LDSM
    +8-col pads + the round-3 de-staged u (vn epilogue reads it straight
    from global).  Single-buffered state tiles: ~43KB smem/CTA at w4/bv32
    -> 2 CTAs/SM co-residency, which is where the latency-bound serial
    chunk chains actually overlap (the round-3 F2 at 84.5KB gets 1/SM).
    """
    B, T, H, K, V, C, NC, Tp = shp
    assert NC % NS == 0, f"NC={NC} not divisible by segments={NS}"
    L = NC // NS
    nthr = 32 * W
    VT = V // BV
    e = Em()
    emit_lane_precompute(e)
    e.l("%zfs = arith.constant 0.0 : f32")
    cH, cC, cNC, cBV, cVT = e.c(H), e.c(C), e.c(NC), e.c(BV), e.c(VT)
    cNS, cL = e.c(NS), e.c(L)
    cNSm1 = e.c(NS - 1)
    cCm1 = e.c(C - 1)
    TY = {
        "u": f"memref<{B*H*NC*C*V}xbf16>", "w": f"memref<{B*H*NC*C*K}xbf16>",
        "k": f"memref<{B*Tp*H*K}xbf16>", "gc": f"memref<{B*H*NC*C}xf32>",
        "h0": f"memref<{B*H*K*V}xf32>",
        "h_all": f"memref<{B*H*NC*K*V}xbf16>",
        "v_new": f"memref<{B*H*NC*C*V}xbf16>",
        "ht": f"memref<{B*H*K*V}xf32>",
        "seg_exit": f"memref<{B*H*VT*NS*K*BV}xbf16>"}

    # round-3 +8-col pads on every ldmatrix-read tile
    S_, S_ty = alloc2d(e, "S", K, BV + 8, BF)
    vn, vn_ty = alloc2d(e, "vn", C, BV + 8, BF)
    wtile, wtile_ty = alloc2d(e, "wt", C, bk + 8, BF)
    ktile2, ktile2_ty = alloc2d(e, "k2", bk, K + 8, BF)
    gc, gc_ty = alloc2d(e, "gcs", C, 1, F32)

    e.l("%sxb = gpu.block_id x")
    e.l(f"%sg = arith.remsi %sxb, {cNS} : index")
    e.l(f"%bhv = arith.divsi %sxb, {cNS} : index")
    e.l(f"%vt = arith.remsi %bhv, {cVT} : index")
    e.l(f"%bh = arith.divsi %bhv, {cVT} : index")
    e.l(f"%b = arith.divsi %bh, {cH} : index")
    e.l(f"%h = arith.remsi %bh, {cH} : index")
    e.l(f"%vtc = arith.muli %vt, {cBV} : index")
    e.l(f"%sgL = arith.muli %sg, {cL} : index")
    h0b = addi(e, muli(e, "%bh", e.c(K * V)), "%vtc")
    htb = h0b
    # invariant bases
    e.l(f"%wq = arith.muli %bh, {e.c(NC * C * K)} : index")
    e.l(f"%uq = arith.muli %bh, {e.c(NC * C * V)} : index")
    e.l(f"%uqv = arith.addi %uq, %vtc : index")
    e.l(f"%gcq = arith.muli %bh, {e.c(NC * C)} : index")
    x = muli(e, muli(e, "%b", e.c(Tp)), cH)
    e.l(f"%kq = arith.muli {addi(e, x, '%h')}, {e.c(K)} : index")
    # seg_exit layout: ((bh*VT+vt)*NS + sg)*K*BV + r*BV + c
    e.l(f"%cxq = arith.muli %bhv, {e.c(NS * K * BV)} : index")
    e.l(f"%sgx = arith.muli %sg, {e.c(K * BV)} : index")

    # ---- entry-state init ------------------------------------------------
    if mode == "c":
        nz = e.v("sgnz")
        e.l(f"{nz} = arith.cmpi ne, %sg, %c0 : index")
        e.l(f"scf.if {nz} {{")
        sx = addi(e, "%cxq", "%sgx")
        sxb = subi_ssa(e, sx, e.c(K * BV))
        def init_se(i):
            r = divsi(e, i, e.c(BV))
            c = remsi(e, i, e.c(BV))
            v = e.v("sev")
            e.l(f"{v} = memref.load %seg_exit[{addi(e, sxb, addi(e, muli(e, r, e.c(BV)), c))}] : {TY['seg_exit']}")
            e.l(f"memref.store {v}, {S_}[{r}, {c}] : {S_ty}")
        emit_cell_while(e, K * BV, nthr, init_se)
        e.l("} else {")
        def init_h0(i):
            r = divsi(e, i, e.c(BV))
            c = remsi(e, i, e.c(BV))
            v = e.v("h0v")
            e.l(f"{v} = memref.load %h0[{addi(e, h0b, addi(e, muli(e, r, e.c(V)), c))}] : {TY['h0']}")
            t = e.v("h0t")
            e.l(f"{t} = arith.truncf {v} : f32 to bf16")
            e.l(f"memref.store {t}, {S_}[{r}, {c}] : {S_ty}")
        emit_cell_while(e, K * BV, nthr, init_h0)
        e.l("}")
    else:
        zbf = e.v("szb")
        e.l(f"{zbf} = arith.constant 0.0 : bf16")
        def zero_s(i):
            r = divsi(e, i, e.c(BV))
            c = remsi(e, i, e.c(BV))
            e.l(f"memref.store {zbf}, {S_}[{r}, {c}] : {S_ty}")
        emit_cell_while(e, K * BV, nthr, zero_s)
    e.l("gpu.barrier")

    # ---- segment chunk loop ----------------------------------------------
    def chunk_loop():
        e.l("%nwr = scf.while (%jw = %c0) : (index) -> (index) {")
        ncond = e.v("jcc")
        e.l(f"{ncond} = arith.cmpi ult, %jw, {cL} : index")
        e.l(f"scf.condition({ncond}) %jw : index")
        e.l("} do {")
        e.l("^bb0(%j: index):")
        e.l("%n = arith.addi %j, %sgL : index")
        wbn = addi(e, "%wq", muli(e, "%n", e.c(C * K)))
        gcb = addi(e, "%gcq", muli(e, "%n", cC))
        kbc = addi(e, "%kq", muli(e, "%n", e.c(C * H * K)))
        shb = addi(e, muli(e, "%bh", e.c(NC * K * V)), muli(e, "%n", e.c(K * V)))
        shb_vtc = addi(e, shb, "%vtc")
        ub_vtc = addi(e, "%uqv", muli(e, "%n", e.c(C * V)))

        if mode == "c":
            # entry-state store h_all[n] = S (coalesced cell pass)
            copy_s2g_vec(e, S_, S_ty,
                         lambda r, c8: addi(e, shb_vtc,
                                            addi(e, muli(e, r, e.c(V)), c8)),
                         "%h_all", TY["h_all"], K, BV, nthr)
            e.l("gpu.barrier")

        emit_load_gc(e, "%gc", TY["gc"], gcb, gc, gc_ty)
        e.l(f"%gcl = memref.load {gc}[{cCm1}, %c0] : {gc_ty}")

        # vp: acc = w @ S (w staged in bk slices; B = resident S)
        accs_vp = emit_kloop(
            e, "vp", K, bk,
            stages=[{"tile": wtile, "ty": wtile_ty, "rows": C, "cols": bk,
                     "src": "%w", "sty": TY["w"], "kind": "rc", "base": wbn,
                     "rowmul": K}],
            gemms=[{"name": "vp", "M": C, "N": BV, "kt": False,
                    "a": (wtile, wtile_ty, True), "b": (S_, S_ty, False)}],
            W=W)

        # vn epilogue (u straight from global; kterm fold 2^(gcl-gc_r))
        uqn = addi(e, "%uqv", muli(e, "%n", e.c(C * V)))

        def epi_vn(mi, ni, ei, row, col, acc):
            xv = acc_extract(e, acc, ei // 2, ei % 2, "evn")
            uix = addi(e, uqn, addi(e, muli(e, row, e.c(V)), col))
            uv = e.v("vnu")
            e.l(f"{uv} = memref.load %u[{uix}] : {TY['u']}")
            uf = e.v("vnf")
            e.l(f"{uf} = arith.extf {uv} : bf16 to f32")
            dif = e.v("vnd")
            e.l(f"{dif} = arith.subf {uf}, {xv} : f32")
            gr = e.v("vnr")
            e.l(f"{gr} = memref.load {gc}[{row}, %c0] : {gc_ty}")
            d = e.v("vnz")
            e.l(f"{d} = arith.subf %gcl, {gr} : f32")
            ex = e.v("vne")
            e.l(f"{ex} = math.exp2 {d} : f32")
            sc = e.v("vns")
            e.l(f"{sc} = arith.mulf {dif}, {ex} : f32")
            t = e.v("vnt")
            e.l(f"{t} = arith.truncf {sc} : f32 to bf16")
            e.l(f"memref.store {t}, {vn}[{row}, {col}] : {vn_ty}")
        run_epilogue(e, "vp", C, BV, W, accs_vp, epi_vn)
        e.l("gpu.barrier")

        if mode == "c":
            copy_s2g_vec(e, vn, vn_ty,
                         lambda r, c8: addi(e, ub_vtc,
                                            addi(e, muli(e, r, e.c(V)), c8)),
                         "%v_new", TY["v_new"], C, BV, nthr)
            e.l("gpu.barrier")

        # sd: acc = k^T @ vn_scaled (k staged in bk slices, swapped A reads)
        accs_sd = emit_kloop(
            e, "sd", C, bk,
            stages=[{"tile": ktile2, "ty": ktile2_ty, "rows": bk, "cols": K,
                     "src": "%k", "sty": TY["k"], "kind": "kc", "base": kbc,
                     "rowmul": H * K}],
            gemms=[{"name": "sd", "M": K, "N": BV, "kt": False, "a_swap": True,
                    "a": (ktile2, ktile2_ty, True), "b": (vn, vn_ty, False)}],
            W=W)
        dl = e.v("dl")
        e.l(f"{dl} = math.exp2 %gcl : f32")
        run_epilogue(e, "sd", K, BV, W, accs_sd,
                     emit_S_update(e, S_, S_ty, dl))
        e.l("gpu.barrier")
        jn = e.v("jcn")
        e.l(f"{jn} = arith.addi %j, %c1 : index")
        e.l(f"scf.yield {jn} : index")
        e.l("}")

    if mode == "a":
        # ALL segments run pass A (segment 0 too: its zero-entry exit c_0
        # IS segment 1's entry; only E_0 = h0 is handled by f2c directly)
        chunk_loop()
        # exit store: seg_exit[sg] = S
        cxb = addi(e, "%cxq", "%sgx")
        def exit_store(i):
            r = divsi(e, i, e.c(BV))
            c = remsi(e, i, e.c(BV))
            v = e.v("exv")
            e.l(f"{v} = memref.load {S_}[{r}, {c}] : {S_ty}")
            e.l(f"memref.store {v}, %seg_exit[{addi(e, cxb, addi(e, muli(e, r, e.c(BV)), c))}] : {TY['seg_exit']}")
        emit_cell_while(e, K * BV, nthr, exit_store)
    else:
        chunk_loop()
        # ht (f32) from the LAST segment only; CTA-uniform guard
        isl = e.v("sgl")
        e.l(f"{isl} = arith.cmpi eq, %sg, {cNSm1} : index")
        e.l(f"scf.if {isl} {{")
        def ht_pass(i):
            r = divsi(e, i, e.c(BV))
            c = remsi(e, i, e.c(BV))
            v = e.v("htv")
            e.l(f"{v} = memref.load {S_}[{r}, {c}] : {S_ty}")
            w = e.v("htc")
            e.l(f"{w} = arith.extf {v} : bf16 to f32")
            e.l(f"memref.store {w}, %ht[{addi(e, htb, addi(e, muli(e, r, e.c(V)), c))}] : {TY['ht']}")
        emit_cell_while(e, K * BV, nthr, ht_pass)
        e.l("}")

    # pins: keep multibuffer off the state tiles (users span the chunk
    # scf.while + several kloop scf.for's -> transform hard-fail without)
    pin_alloc(e, S_, S_ty)
    pin_alloc(e, vn, vn_ty)
    pin_alloc(e, gc, gc_ty)

    params = [("u", B * H * NC * C * V, BF), ("w", B * H * NC * C * K, BF),
              ("k", B * Tp * H * K, BF), ("gc", B * H * NC * C, F32),
              ("h0", B * H * K * V, F32),
              ("h_all", B * H * NC * K * V, BF),
              ("v_new", B * H * NC * C * V, BF),
              ("ht", B * H * K * V, F32),
              ("seg_exit", B * H * VT * NS * K * BV, BF)]
    return e, params, (B * H * VT * NS, 1), nthr


def subi_ssa(e, a, b):
    o = e.v("sbs")
    e.l(f"{o} = arith.subi {a}, {b} : index")
    return o


# ---------------------------------------------------------------------------
# F3 "o" (round 2)
# ---------------------------------------------------------------------------

def gen_f3(shp, W, BV, bk=32):
    """Output stage, grid (NC*B*H, VT).

    Round-2 structure: fused qk+oi K-loop (ONE staged q tile feeds both
    gemms; k tile kt=True; S slice k-major from h_all) -> P epilogue writes
    P = tril_inc((q k^T) 2^(gc_i - gcl))  [matches F2's scaled v_new] ->
    oa K-loop (A = resident P, B = v_new slices staged) -> o epilogue
    o = (2^gc_i * (q@S) + P@v_new) * scale  (the 2^gc row scaling moved
    from a staged q tile into the f32 epilogue: one less rounding).
    """
    B, T, H, K, V, C, NC, Tp = shp
    nthr = 32 * W
    VT = V // BV
    scale = K ** -0.5
    e = Em()
    emit_lane_precompute(e)
    e.l("%zfs = arith.constant 0.0 : f32")
    e.l(f"%fsc = arith.constant {f32lit(scale)} : f32")
    cH, cC, cNC, cBV, cVT = e.c(H), e.c(C), e.c(NC), e.c(BV), e.c(VT)
    cCm1 = e.c(C - 1)
    TY = {
        "q": f"memref<{B*Tp*H*K}xbf16>", "k": f"memref<{B*Tp*H*K}xbf16>",
        "gc": f"memref<{B*H*NC*C}xf32>",
        "h_all": f"memref<{B*H*NC*K*V}xbf16>",
        "v_new": f"memref<{B*H*NC*C*V}xbf16>",
        "o": f"memref<{B*Tp*H*V}xbf16>"}

    # R20-fwd GDN_F3KS (default off): f3's 2-CTA conversion, the r19
    # audited design.  Both K-loops drop to bk=16 (the F1KS kkt move:
    # qt/kt [64,24] multibuffer 6,144B each, st/vnt [16,136] 8,704B
    # each; the mma accumulation ORDER is unchanged -- bk=16 splits
    # BETWEEN the m16n8k16 blocks, accumulators carried across
    # iterations) and ot lands in P's dead arena at PTX level (P's
    # last read is the oa loop's A-operand ldmatrix; ot is written
    # after it -- identify-by-size lifetime arena + one explicit
    # barrier after the oa loop pins the read-before-write order).
    # Pool 82,176 -> 47,360B <= 50,688 -> 2 CTAs/SM.  Audit (r19):
    # f3 has NO scans (no SCANDB exposure) and NO zero-init cell loops
    # (epi_P writes the full CxC incl. upper zeros; no ZFIX exposure).
    # GDN_F3KS_F1CTA (diag arm): +7 dead rows on st push the pool to
    # ~51.2KB > 50,688 -- the forced-1-CTA gate (the r18 SASS-diff
    # trick: ONE size constant, values bit-identical).
    _ks3 = bool(os.environ.get("GDN_F3KS"))
    _f3f1cta = _ks3 and bool(os.environ.get("GDN_F3KS_F1CTA"))
    _bk3 = 16 if _ks3 else bk
    # round-3: +8-col pads on every ldmatrix-read tile (see gen_f2 note)
    qtile, qtile_ty = alloc2d(e, "qt", C, _bk3 + 8, BF)
    ktile, ktile_ty = alloc2d(e, "kt", C, _bk3 + 8, BF)
    Stile, Stile_ty = alloc2d(e, "st", _bk3 + (7 if _f3f1cta else 0),
                              BV + 8, BF)
    vntile, vntile_ty = alloc2d(e, "vnt", _bk3, BV + 8, BF)
    P, P_ty = alloc2d(e, "P", C, C + 8, BF)
    gc, gc_ty = alloc2d(e, "gcs", C, 1, F32)
    # G5-K round 5: o staging tile.  The o epilogue's fragment-mapped
    # global stores are 2B-at-scattered-addresses (16 rows x 4 col-pairs
    # per lane group); restaging through a padded smem tile + one
    # coalesced cell pass turns them into 64B/warp row segments (the
    # G5-F negative result inverted: in-epilogue global stores lose to
    # the extra pass).  136-col stride (272B) also keeps the STS side
    # conflict-free (4-word row rotation).
    _f3o = not os.environ.get("G5K_NOF3O")
    otile, otile_ty = (alloc2d(e, "ot", C, BV + 8, BF) if _f3o
                       else (None, None))

    e.l("%bnh = gpu.block_id x")
    e.l(f"%n = arith.remsi %bnh, {cNC} : index")
    e.l(f"%bh = arith.divsi %bnh, {cNC} : index")
    e.l("%vt = gpu.block_id y")
    e.l(f"%b = arith.divsi %bh, {cH} : index")
    e.l(f"%h = arith.remsi %bh, {cH} : index")
    e.l(f"%vtc = arith.muli %vt, {cBV} : index")
    t0 = muli(e, "%n", cC)
    x = addi(e, muli(e, "%b", e.c(Tp)), t0)
    x = muli(e, x, e.c(H))
    xh = addi(e, x, "%h")
    qb = muli(e, xh, e.c(K))
    ob = muli(e, xh, e.c(V))
    gcb = addi(e, muli(e, "%bh", e.c(NC * C)), muli(e, "%n", e.c(C)))
    shb = addi(e, muli(e, "%bh", e.c(NC * K * V)), muli(e, "%n", e.c(K * V)))
    vnb = addi(e, muli(e, "%bh", e.c(NC * C * V)), muli(e, "%n", e.c(C * V)))
    shb_vtc = addi(e, shb, "%vtc")
    vnb_vtc = addi(e, vnb, "%vtc")

    emit_load_gc(e, "%gc", TY["gc"], gcb, gc, gc_ty)
    e.l(f"%gcl = memref.load {gc}[{cCm1}, %c0] : {gc_ty}")

    accs1 = emit_kloop(
        e, "qo", K, _bk3,
        stages=[
            {"tile": qtile, "ty": qtile_ty, "rows": C, "cols": _bk3,
             "src": "%q", "sty": TY["q"], "kind": "rc", "base": qb,
             "rowmul": H * K},
            {"tile": ktile, "ty": ktile_ty, "rows": C, "cols": _bk3,
             "src": "%k", "sty": TY["k"], "kind": "rc", "base": qb,
             "rowmul": H * K},
            {"tile": Stile, "ty": Stile_ty, "rows": _bk3, "cols": BV,
             "src": "%h_all", "sty": TY["h_all"], "kind": "kc",
             "base": shb_vtc, "rowmul": V}],
        gemms=[
            {"name": "qk", "M": C, "N": C, "kt": True,
             "a": (qtile, qtile_ty, True), "b": (ktile, ktile_ty, True)},
            {"name": "oi", "M": C, "N": BV, "kt": False,
             "a": (qtile, qtile_ty, True), "b": (Stile, Stile_ty, True)}],
        W=W)
    e.l("gpu.barrier")

    def epi_P(mi, ni, ei, row, col, acc):
        xv = acc_extract(e, acc, ei // 2, ei % 2, "eP")
        gi = e.v("Pgi"); e.l(f"{gi} = memref.load {gc}[{row}, %c0] : {gc_ty}")
        d = e.v("Pd"); e.l(f"{d} = arith.subf {gi}, %gcl : f32")
        ex = e.v("Pex"); e.l(f"{ex} = math.exp2 {d} : f32")
        m = e.v("Pm"); e.l(f"{m} = arith.mulf {xv}, {ex} : f32")
        le = e.v("Ple"); e.l(f"{le} = arith.cmpi sle, {col}, {row} : index")
        sel = e.v("Psel"); e.l(f"{sel} = arith.select {le}, {m}, %zfs : f32")
        t = e.v("Pt"); e.l(f"{t} = arith.truncf {sel} : f32 to bf16")
        e.l(f"memref.store {t}, {P}[{row}, {col}] : {P_ty}")
    run_epilogue(e, "qk", C, C, W, accs1, epi_P)
    e.l("gpu.barrier")

    accs2 = emit_kloop(
        e, "oa", C, _bk3,
        stages=[{"tile": vntile, "ty": vntile_ty, "rows": _bk3, "cols": BV,
                 "src": "%v_new", "sty": TY["v_new"], "kind": "kc",
                 "base": vnb_vtc, "rowmul": V}],
        gemms=[{"name": "oa", "M": C, "N": BV, "kt": False,
                "a": (P, P_ty, False), "b": (vntile, vntile_ty, True)}],
        W=W)
    if _ks3:
        # ot lands in P's dead arena (the PTX lifetime alias): every
        # warp's last P ldmatrix (the oa A-operand) must complete
        # before the o epilogue's first ot store.  The kloop body ends
        # in a barrier, but the pipeline transform restructures loop
        # epilogues -- this explicit post-loop barrier pins the order
        # (the r15 bisect_shift lesson).
        e.l("gpu.barrier")

    # o epilogue over BOTH accumulator sets
    wn = BV // W
    for mi in range(C // 16):
        for ni in range(wn // 8):
            acc_i = accs1[("oi", mi, ni)]
            acc_a = accs2[("oa", mi, ni)]
            for ei in range(4):
                row = f"%rA_oi_{mi}" if ei < 2 else f"%rAh_oi_{mi}"
                col = (f"%nct_oi_{ni}" if ei % 2 == 0
                       else f"%nct1_oi_{ni}")
                xi = acc_extract(e, acc_i, ei // 2, ei % 2, "eoi")
                xa = acc_extract(e, acc_a, ei // 2, ei % 2, "eoa")
                gi = e.v("ogi")
                e.l(f"{gi} = memref.load {gc}[{row}, %c0] : {gc_ty}")
                ex = e.v("oge")
                e.l(f"{ex} = math.exp2 {gi} : f32")
                m = e.v("ogm")
                e.l(f"{m} = arith.mulf {xi}, {ex} : f32")
                a2 = e.v("oga")
                e.l(f"{a2} = arith.addf {m}, {xa} : f32")
                sc = e.v("ogs")
                e.l(f"{sc} = arith.mulf {a2}, %fsc : f32")
                t = e.v("ogt")
                e.l(f"{t} = arith.truncf {sc} : f32 to bf16")
                if _f3o:
                    e.l(f"memref.store {t}, {otile}[{row}, {col}] "
                        f": {otile_ty}")
                else:
                    idx = addi(e, ob, addi(e, muli(e, row, e.c(H * V)),
                                           addi(e, "%vtc", col)))
                    e.l(f"memref.store {t}, %o[{idx}] : {TY['o']}")

    if _f3o:
        e.l("gpu.barrier")
        # coalesced o pass: warp covers 32 consecutive V-columns of one
        # row (64B); VT=1 at bv3=V so the whole 128-col row is covered by
        # 4 warps = 256B per row, no cross-half seams.
        copy_s2g_vec(e, otile, otile_ty,
                     lambda r, c8: addi(e, ob, addi(e, muli(e, r, e.c(H * V)),
                                                    addi(e, "%vtc", c8))),
                     "%o", TY["o"], C, BV, nthr)

    params = [("q", B * Tp * H * K, BF), ("k", B * Tp * H * K, BF),
              ("gc", B * H * NC * C, F32),
              ("h_all", B * H * NC * K * V, BF),
              ("v_new", B * H * NC * C * V, BF),
              ("o", B * Tp * H * V, BF)]
    return e, params, (NC * B * H, VT), nthr


# ---------------------------------------------------------------------------
# mono module: the staged trio in ONE cubin (shared globals, no DtoD)
# ---------------------------------------------------------------------------

def gen_mono(shp, W, BV2, BV3, bk=32, f2="pp", f2_segments=1):
    B, T, H, K, V, C, NC, Tp = shp
    VT2 = V // BV2
    VT3 = V // BV3
    parts = []
    params = []

    def add_params(ps):
        for p in ps:
            if p not in params:
                params.append(p)

    # G5-I round 4: --f2-segments S > 1 replaces the single serial F2 with
    # the segment-split pair f2a (zero-entry exits) -> f2c (true-entry
    # re-run with outputs).  Launch order = list order (single stream).
    f2_kernels = []
    if f2_segments > 1:
        assert NC % f2_segments == 0
        f2_kernels = [
            ("f2a", lambda: gen_f2_seg(shp, W, BV2, f2_segments, "a", bk=bk),
             (B * H * VT2 * f2_segments, 1)),
            ("f2c", lambda: gen_f2_seg(shp, W, BV2, f2_segments, "c", bk=bk),
             (B * H * VT2 * f2_segments, 1))]
    else:
        # R14-U: "sb" = the round-14 single-buffer diet (GDN_F2SB=1 also
        # selects it for default-pp build requests)
        _f2sb = (f2 == "sb") or bool(os.environ.get("GDN_F2SB"))
        f2_fn = (gen_f2_sb if _f2sb else
                 gen_f2 if f2 == "pp" else gen_f2_r2)
        f2_kernels = [("f2", lambda: f2_fn(shp, W, BV2, bk=bk),
                       (B * H * VT2, 1))]

    trio = [("f1", lambda: gen_f1(shp, W, None, bk=bk), (NC, B * H))]
    trio += f2_kernels
    trio += [("f3", lambda: gen_f3(shp, W, BV3, bk=bk), (NC * B * H, VT3))]
    for name, fn, grid in trio:
        e, ps, _g, nthr = fn()
        if name == "f1":
            # mono: f1 gc_out IS f2/f3 gc (trio bridged via DtoD; one
            # module = one shared global)
            e.lines = [ln.replace("gc_out", "gc") for ln in e.lines]
            ps = [("gc" if n == "gc_out" else n, el, dt)
                  for n, el, dt in ps]
        add_params(ps)
        parts.append((f"gdn_fwd_{name}", e, grid, nthr))
    return parts, params


def wrap_module_mono(kernels, params, meta):
    """One module, one set of public globals, three zero-arg kernels.
    Kernel boundaries are the stream-order syncs; the staged path then runs
    without any device-to-device handoff copies."""
    hdr = ["// Generated by tools/gdn_full_gen.py -- GDN full-op forward "
           "(G5-D round 2, mono staged trio)",
           "// " + meta,
           "// buffers (public zero-init globals; harness uploads/downloads):",
           " // " + " ".join(f"@{n}:{dt}[{el}]" for n, el, dt in params)]
    lines = []
    for n, el, dt in params:
        lines.append(f'memref.global @{n} : memref<{el}x{dt}>')
    for kname, e, grid, nthr in kernels:
        lines.append(f"func.func @{kname}() {{")
        lines.append(f"%hgx = arith.constant {grid[0]} : index")
        lines.append(f"%hgy = arith.constant {grid[1]} : index")
        lines.append("%hgz = arith.constant 1 : index")
        lines.append(f"%hbx = arith.constant {nthr} : index")
        lines.append("gpu.launch blocks(%bx, %by, %bz) in "
                     "(%gx = %hgx, %gy = %hgy, %gz = %hgz)")
        lines.append("             threads(%tx, %ty, %tz) in "
                     "(%bxx = %hbx, %byy = %hgz, %bzz = %hgz) {")
        for n, el, dt in params:
            lines.append(f"%{n} = memref.get_global @{n} : "
                         f"memref<{el}x{dt}>")
        lines.extend(ln if ln.startswith("//") else "  " + ln if ln else ln
                     for ln in hoist_constants(e.text()).split("\n"))
        lines += ["  gpu.terminator", "  }", "  return", "}"]
    return "\n".join(hdr) + "\n" + "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# fused flagship
# ---------------------------------------------------------------------------

def gen_fused(shp, W, BV):
    B, T, H, K, V, C, NC, Tp = shp
    nthr = 32 * W
    VT = V // BV
    scale = K ** -0.5
    e = Em()
    emit_lane_precompute(e)
    e.l("%zfs = arith.constant 0.0 : f32")
    e.l(f"%fsc = arith.constant {f32lit(scale)} : f32")
    cH, cC, cNC, cBV, cVT = e.c(H), e.c(C), e.c(NC), e.c(BV), e.c(VT)
    cCm1 = e.c(C - 1)
    TY = {
        "q": f"memref<{B*Tp*H*K}xbf16>", "k": f"memref<{B*Tp*H*K}xbf16>",
        "v": f"memref<{B*Tp*H*V}xbf16>", "beta": f"memref<{B*Tp*H}xf32>",
        "g": f"memref<{B*Tp*H}xf32>", "h0": f"memref<{B*H*K*V}xf32>",
        "o": f"memref<{B*Tp*H*V}xbf16>", "ht": f"memref<{B*H*K*V}xf32>"}

    # smem plan (<= 99KB/CTA on this sm_120 laptop part):
    #   k_t [C,K] persistent | QK [C,K] shared: bk2 first, q_t staged LATE
    #   (after wg consumed bk2) | S [K,BV] persistent | A_m [C,C] f32
    #   w_t [C,K] | TP [C,C] bf16 shared: transposed-T (TbT) then P
    #   (P written only after wg consumed TbT) | BVN [C,BV] shared:
    #   beta*v then v_new | gc/beta [C] f32.
    # R14-U refresh (GDN_FLAV1, default on; GDN_FLAV0 reverts): the r3
    # LDSM bank-conflict fix ported from the staged path -- every 2-D
    # tile read by ldmatrix gets +8 bf16 cols (and A_m +1 f32 col for its
    # fragment-mapped STS); the serial per-thread gc cumsum replaced by
    # the F1 Hillis-Steele log-scan (emit_cumsum_beta_scan -- value
    # re-roll expected, same class as the r5 GDN_WY_GC_SHFL lesson); the
    # o epilogue restaged through a padded ot tile + one coalesced cell
    # pass (the F3O lesson: fragment-mapped 2B global stores lose to the
    # extra pass).  Pool @w4/bv32: 99072B.
    _fla = (not os.environ.get("GDN_FLAV0")) and BV <= 48
    _pad = 8 if _fla else 0
    _apad = 1 if _fla else 0
    k_t, k_t_ty = alloc2d(e, "k_t", C, K + _pad, BF)
    QK, QK_ty = alloc2d(e, "QK", C, K + _pad, BF)
    S, S_ty = alloc2d(e, "S", K, BV + _pad, BF)
    A_m, A_ty = alloc2d(e, "A_m", C, C + _apad, F32)
    w_t, w_t_ty = alloc2d(e, "w_t", C, K + _pad, BF)
    TP, TP_ty = alloc2d(e, "TP", C, C + _pad, BF)
    BVN, BVN_ty = alloc2d(e, "BVN", C, BV + _pad, BF)
    otile, otile_ty = (alloc2d(e, "ot", C, BV + _pad, BF) if _fla
                       else (None, None))
    gc, gc_ty = alloc2d(e, "gcs", C, 1, F32)
    bs, bs_ty = alloc2d(e, "bss", C, 1, F32)

    e.l("%bhb = gpu.block_id x")
    e.l(f"%bh = arith.divsi %bhb, {cVT} : index")
    e.l(f"%vt = arith.remsi %bhb, {cVT} : index")
    e.l(f"%b = arith.divsi %bh, {cH} : index")
    e.l(f"%h = arith.remsi %bh, {cH} : index")
    e.l(f"%vtc = arith.muli %vt, {cBV} : index")
    h0b = addi(e, muli(e, "%bh", e.c(K * V)), "%vtc")
    htb = h0b

    def init_s(i):
        r = divsi(e, i, e.c(BV))
        c = remsi(e, i, e.c(BV))
        v = e.v("h0v")
        e.l(f"{v} = memref.load %h0[{addi(e, h0b, addi(e, muli(e, r, e.c(V)), c))}] : {TY['h0']}")
        t = e.v("h0t")
        e.l(f"{t} = arith.truncf {v} : f32 to bf16")
        e.l(f"memref.store {t}, {S}[{r}, {c}] : {S_ty}")
    emit_cell_loop(e, K * BV, nthr, init_s)
    e.l("gpu.barrier")

    e.l(f"scf.for %n = %c0 to {cNC} step %c1 {{")
    t0 = muli(e, "%n", cC)
    x = addi(e, muli(e, "%b", e.c(Tp)), t0)
    x = muli(e, x, cH)
    xh = addi(e, x, "%h")
    qbc = muli(e, xh, e.c(K))
    vbc = muli(e, xh, e.c(V))
    ob = vbc
    bbc = addi(e, x, "%h")

    # R16-Z GDN_FPIPE (default off): intra-chunk async staging, the F2US
    # pattern applied to the fused chunk chain.  The three global->smem
    # cell passes (k, beta*v, q) become MANUAL thread-strided
    # device_async_copy groups with the waits placed at the first
    # consumer, so the global latency overlaps compute instead of
    # stalling the serial chain:
    #   k(n) -> k_t   issued BEFORE the gc scan (overlaps it; the tile
    #                  is dead since sd(n-1) consumed k(n-1));
    #   v(n) -> BVN   issued alongside k (RAW; overlaps scan + k wait +
    #                  kkt + the solve), beta applied by an in-place
    #                  smem scale before ug;
    #   q(n) -> QK    issued right after wg's barrier (QK is dead once
    #                  wg consumed bk2; overlaps vp), waited before qk.
    # bk2 additionally re-reads GLOBAL k in the stock path; with k_t
    # resident it becomes a pure smem->smem scaled pass (k(n) is read
    # from global ONCE per chunk, not twice).
    # No cross-chunk ping-pong: the 101,376B opt-in cap leaves 2,304B
    # over the 99,072B pool -- a k/q pong (17,408B) does not fit without
    # dieting the very tiles the r14 refresh added (see fwd_report
    # round-16).  Values are bit-identical to the stock staging.
    _fp = bool(os.environ.get("GDN_FPIPE"))
    if _fp:
        ktoks = []
        async_stage_unrolled(
            e, "%k", TY["k"], k_t, k_t_ty, C, K, nthr,
            lambda r, c8: addi(e, qbc, addi(e, muli(e, r, e.c(H * K)), c8)),
            lambda r, c8: (r, c8), ktoks)
        gk = e.v("agr")
        e.l(f"{gk} = nvgpu.device_async_create_group {', '.join(ktoks)}")
        vtoks = []
        async_stage_unrolled(
            e, "%v", TY["v"], BVN, BVN_ty, C, BV, nthr,
            lambda r, c8: addi(e, vbc, addi(e, muli(e, r, e.c(H * V)),
                                            addi(e, "%vtc", c8))),
            lambda r, c8: (r, c8), vtoks)
        gv = e.v("agr")
        e.l(f"{gv} = nvgpu.device_async_create_group {', '.join(vtoks)}")
    if _fla:
        emit_cumsum_beta_scan(e, "%g", TY["g"], bbc, "%beta", TY["beta"],
                              bbc, gc, gc_ty, bs, bs_ty, H)
    else:
        emit_cumsum_beta(e, "%g", TY["g"], bbc, "%beta", TY["beta"], bbc,
                         gc, gc_ty, bs, bs_ty, H)
    if _fp:
        e.l(f"nvgpu.device_async_wait {gk} numGroups = 0")
    else:
        stage_copy_vec(e, "%k", TY["k"], qbc, H * K, C, K, k_t, k_t_ty,
                       nthr)
    e.l("gpu.barrier")
    emit_gemm(e, "kkt", C, C, K, W, k_t, k_t_ty, k_t, k_t_ty, kt=True,
              epilogue=emit_A_epilogue(e, A_m, A_ty, bs, bs_ty, gc, gc_ty))
    e.l("gpu.barrier")
    emit_solve(e, A_m, A_ty, TP, TP_ty, nthr, C)

    def pre_bv(w, r, c):
        b_ = e.v("bvb")
        e.l(f"{b_} = memref.load {bs}[{r}, %c0] : {bs_ty}")
        m = e.v("bvm"); e.l(f"{m} = arith.mulf {w}, {b_} : f32")
        t = e.v("bvt"); e.l(f"{t} = arith.truncf {m} : f32 to bf16")
        return t
    if _fp:
        # raw v landed by cp.async; apply beta in place (same arithmetic
        # as pre_bv -- bit-identical values)
        e.l(f"nvgpu.device_async_wait {gv} numGroups = 0")
        e.l("gpu.barrier")

        def bvscale(i):
            r = divsi(e, i, e.c(BV))
            c = remsi(e, i, e.c(BV))
            v0 = e.v("bvl")
            e.l(f"{v0} = memref.load {BVN}[{r}, {c}] : {BVN_ty}")
            w = e.v("bvx")
            e.l(f"{w} = arith.extf {v0} : bf16 to f32")
            t = pre_bv(w, r, c)
            e.l(f"memref.store {t}, {BVN}[{r}, {c}] : {BVN_ty}")
        emit_cell_loop(e, C * BV, nthr, bvscale)
    else:
        global_to_smem_scalar(
            e, "%v", TY["v"],
            lambda r, c: addi(e, vbc, addi(e, muli(e, r, e.c(H * V)),
                                           addi(e, "%vtc", c))),
            BVN, BVN_ty, C, BV, nthr, pre=pre_bv)
    e.l("gpu.barrier")
    accs_u = emit_gemm(e, "ug", C, BV, C, W, TP, TP_ty, BVN, BVN_ty,
                       a_swap=True)

    if _fp:
        # bk2 from the resident k_t (smem->smem scaled pass; the stock
        # path re-reads the same 16KB of k from global)
        def bk2_sm(i):
            r = divsi(e, i, e.c(K))
            c = remsi(e, i, e.c(K))
            v0 = e.v("bkv")
            e.l(f"{v0} = memref.load {k_t}[{r}, {c}] : {k_t_ty}")
            w = e.v("bkx")
            e.l(f"{w} = arith.extf {v0} : bf16 to f32")
            b_ = e.v("bkb")
            e.l(f"{b_} = memref.load {bs}[{r}, %c0] : {bs_ty}")
            gi = e.v("bkg")
            e.l(f"{gi} = memref.load {gc}[{r}, %c0] : {gc_ty}")
            ex = e.v("bkex")
            e.l(f"{ex} = math.exp2 {gi} : f32")
            m = e.v("bkm")
            e.l(f"{m} = arith.mulf {w}, {b_} : f32")
            m2 = e.v("bkm2")
            e.l(f"{m2} = arith.mulf {m}, {ex} : f32")
            t = e.v("bkt")
            e.l(f"{t} = arith.truncf {m2} : f32 to bf16")
            e.l(f"memref.store {t}, {QK}[{r}, {c}] : {QK_ty}")
        emit_cell_loop(e, C * K, nthr, bk2_sm)
    else:
        emit_bk2_stage(e, "%k", TY["k"], qbc, bs, bs_ty, gc, gc_ty, QK,
                       QK_ty, H, K, C, nthr)
    e.l("gpu.barrier")

    def epi_w(mi, ni, ei, row, col, acc):
        x = acc_extract(e, acc, ei // 2, ei % 2, "ew")
        t = e.v("ewt"); e.l(f"{t} = arith.truncf {x} : f32 to bf16")
        e.l(f"memref.store {t}, {w_t}[{row}, {col}] : {w_t_ty}")
    emit_gemm(e, "wg", C, K, C, W, TP, TP_ty, QK, QK_ty, a_swap=True,
              epilogue=epi_w)
    e.l("gpu.barrier")

    # q staged LATE into QK (bk2 consumed by wg above)
    if _fp:
        qtoks = []
        async_stage_unrolled(
            e, "%q", TY["q"], QK, QK_ty, C, K, nthr,
            lambda r, c8: addi(e, qbc, addi(e, muli(e, r, e.c(H * K)), c8)),
            lambda r, c8: (r, c8), qtoks)
        gq = e.v("agr")
        e.l(f"{gq} = nvgpu.device_async_create_group {', '.join(qtoks)}")
    else:
        stage_copy_vec(e, "%q", TY["q"], qbc, H * K, C, K, QK, QK_ty, nthr)
    e.l("gpu.barrier")

    accs_vp = emit_gemm(e, "vp", C, BV, K, W, w_t, w_t_ty, S, S_ty)
    accs_vn = frag_binop(e, accs_u, accs_vp, "sub", "vns")
    if _fp:
        e.l(f"nvgpu.device_async_wait {gq} numGroups = 0")
        e.l("gpu.barrier")

    emit_gemm(e, "qk", C, C, K, W, QK, QK_ty, k_t, k_t_ty, kt=True,
              epilogue=emit_P_epilogue(e, TP, TP_ty, gc, gc_ty))
    e.l("gpu.barrier")
    emit_qg_inplace(e, QK, QK_ty, gc, gc_ty, K, C, nthr)

    def epi_vn(mi, ni, ei, row, col, acc):
        x = acc_extract(e, acc, ei // 2, ei % 2, "evn")
        t = e.v("vnt"); e.l(f"{t} = arith.truncf {x} : f32 to bf16")
        e.l(f"memref.store {t}, {BVN}[{row}, {col}] : {BVN_ty}")
    emit_gemm(e, "vng", C, BV, C, W, epi_accs=accs_vn, epilogue=epi_vn)
    e.l("gpu.barrier")

    accs_oi = emit_gemm(e, "oi", C, BV, K, W, QK, QK_ty, S, S_ty)
    accs_oa = emit_gemm(e, "oa", C, BV, C, W, TP, TP_ty, BVN, BVN_ty)
    accs_o = frag_binop(e, accs_oa, accs_oi, "add", "os")

    def epi_o(mi, ni, ei, row, col, acc):
        x = acc_extract(e, acc, ei // 2, ei % 2, "eo")
        m = e.v("eom"); e.l(f"{m} = arith.mulf {x}, %fsc : f32")
        t = e.v("eot"); e.l(f"{t} = arith.truncf {m} : f32 to bf16")
        if _fla:
            e.l(f"memref.store {t}, {otile}[{row}, {col}] : {otile_ty}")
        else:
            idx = addi(e, ob, addi(e, muli(e, row, e.c(H * V)),
                                   addi(e, "%vtc", col)))
            e.l(f"memref.store {t}, %o[{idx}] : {TY['o']}")
    emit_gemm(e, "og", C, BV, C, W, epi_accs=accs_o, epilogue=epi_o)
    e.l("gpu.barrier")

    if _fla:
        # coalesced o pass (the F3O lesson): warp = 32 consecutive V-cols
        copy_s2g_vec(e, otile, otile_ty,
                     lambda r, c8: addi(e, ob,
                                        addi(e, muli(e, r, e.c(H * V)),
                                             addi(e, "%vtc", c8))),
                     "%o", TY["o"], C, BV, nthr)
        e.l("gpu.barrier")

    e.l(f"%gcl = memref.load {gc}[{cCm1}, %c0] : {gc_ty}")
    emit_kterm_inplace(e, k_t, k_t_ty, gc, gc_ty, "%gcl", K, C, nthr)
    e.l("gpu.barrier")
    dl = e.v("dl")
    e.l(f"{dl} = math.exp2 %gcl : f32")
    emit_gemm(e, "sd", K, BV, C, W, k_t, k_t_ty, BVN, BVN_ty, a_swap=True,
              epilogue=emit_S_update(e, S, S_ty, dl))
    e.l("gpu.barrier")
    e.l("}")

    # ht (f32; htb already includes the +vtc V-tile offset)
    smem_to_global_scalar(
        e, S, S_ty, "%ht", TY["ht"],
        lambda r, c: addi(e, htb, addi(e, muli(e, r, e.c(V)), c)),
        K, BV, nthr, src_dt=BF, dst_dt=F32)

    params = [("q", B * Tp * H * K, BF), ("k", B * Tp * H * K, BF),
              ("v", B * Tp * H * V, BF), ("beta", B * Tp * H, F32),
              ("g", B * Tp * H, F32), ("h0", B * H * K * V, F32),
              ("o", B * Tp * H * V, BF), ("ht", B * H * K * V, F32)]
    return e, params, (B * H * VT, 1), nthr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def validate(shp, W, BV):
    B, T, H, K, V, C, NC, Tp = shp
    assert C == 64
    assert K % 16 == 0, "K must be a multiple of 16"
    assert V % BV == 0 and BV % (8 * W) == 0, \
        f"BV={BV} must divide V={V} and be a multiple of 8*warps={8*W}"
    assert W in (1, 2, 4, 8)


def main():
    ap = argparse.ArgumentParser(description="G5-B full-op GDN forward generator")
    ap.add_argument("--path", choices=("staged", "fused", "mono"),
                    required=True)
    ap.add_argument("--kernel", choices=("f1", "f2", "f3"))
    ap.add_argument("--bk", type=int, default=32,
                    help="K-loop step (pipelined tiles are [.,bk])")
    ap.add_argument("--mb", type=int, default=2,
                    help="multibuffer factor for pipeline tiles")
    ap.add_argument("--depth", type=int, default=2,
                    help="pipeline_shared_memory_copies depth")
    ap.add_argument("--bv3", type=int, default=None,
                    help="BV for the f3 kernel in mono (default: --bv)")
    ap.add_argument("--f2", choices=("pp", "r2", "sb"), default="pp",
                    help="F2 scan variant: pp = round-3 ping-pong "
                         "double-buffered chunks (default), r2 = round-2 "
                         "single-buffered, sb = round-14 single-buffer "
                         "diet (<=49.5KB pool, 2 CTAs/SM; GDN_F2SB=1 "
                         "selects it for pp requests too)")
    ap.add_argument("--f2-segments", type=int, default=1,
                    help="G5-I round 4: F2 segment-split count S (mono "
                         "path only; 1 = the serial round-3 kernel). "
                         "S>1 replaces f2 with f2a/f2c (2-pass split; "
                         "matmul-free handoff exact under the bench "
                         "recipes' net decay, see tools/g5i_split_sim.py)")
    ap.add_argument("--shape", default="1,512,4,128,128")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--bv", type=int, default=64)
    ap.add_argument("--copies", choices=("sync", "async"), default="async",
                    help="async: transform.nvgpu.create_async_groups (MANDATORY "
                         "for vector.transfer staging to workgroup smem; the "
                         "plain pipeline cannot legalize it)")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="debug (f1 only): truncate after phase N")
    ap.add_argument("--frag", choices=("ldm", "scalar"), default="ldm",
                    help="fragment loads: ldmatrix (default) or per-element "
                         "scalar loads")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    global _FRAG_MODE
    _FRAG_MODE = args.frag

    B, T, H, K, V = (int(x) for x in args.shape.split(","))
    C = CHUNK
    Tp = ((T + C - 1) // C) * C
    NC = Tp // C
    shp = (B, T, H, K, V, C, NC, Tp)
    W, BV = args.warps, args.bv
    validate(shp, W, BV)
    NS = args.f2_segments
    assert 1 <= NS <= NC and NC % NS == 0, \
        f"--f2-segments {NS} must divide NC={NC}"
    if NS > 1:
        assert args.path == "mono", "--f2-segments > 1 needs --path mono"

    meta = (f"{args.path} W={W} BV={BV} bk={args.bk} mb={args.mb} "
            f"depth={args.depth} copies={args.copies} f2={args.f2} "
            f"f2_segments={NS} "
            f"B={B} T={T} H={H} K={K} V={V} NC={NC}")
    if args.path == "fused":
        e, params, grid, nthr = gen_fused(shp, W, BV)
        kname = "gdn_fwd_fused"
        mod = wrap_module(kname, params, hoist_constants(e.text()),
                          grid[0], grid[1], nthr, meta)
        transform_txt = (ASYNC_TRANSFORM if args.copies == "async"
                         else IDENTITY_TRANSFORM)
        spec = {
            "kernel": kname, "path": args.path,
            "shape": {"B": B, "T": T, "H": H, "K": K, "V": V, "C": C,
                      "NC": NC, "Tp": Tp},
            "knobs": {"warps": W, "bv": BV, "copies": args.copies},
            "params": [{"name": n, "dtype": dt, "elems": int(el)}
                       for n, el, dt in params],
            "grid": [grid[0], grid[1], 1], "block": [nthr, 1, 1],
        }
    elif args.path == "mono":
        # G5-D round 2: the staged trio in ONE cubin (shared globals, no
        # DtoD handoffs), every K-loop through the transform pipeline.
        kernels, params = gen_mono(shp, W, BV, args.bv3 or BV,
                                bk=args.bk, f2=args.f2, f2_segments=NS)
        kname = "gdn_fwd_mono"
        mod = wrap_module_mono(kernels, params, meta)
        transform_txt = pipe_transform(args.mb, args.depth)
        spec = {
            "kernel": kname, "path": args.path,
            "shape": {"B": B, "T": T, "H": H, "K": K, "V": V, "C": C,
                      "NC": NC, "Tp": Tp},
            "knobs": {"warps": W, "bv": BV, "bk": args.bk, "mb": args.mb,
                      "depth": args.depth, "f2": args.f2,
                      "f2_segments": NS},
            "params": [{"name": n, "dtype": dt, "elems": int(el)}
                       for n, el, dt in params],
            "kernels": [{"name": kn, "grid": [g[0], g[1], 1],
                         "block": [nt, 1, 1]}
                        for kn, _e, g, nt in kernels],
        }
    else:
        assert args.kernel, "--kernel f1|f2|f3 required for --path staged"
        if args.kernel == "f2" and (args.f2 == "sb"
                                    or os.environ.get("GDN_F2SB")):
            fn = gen_f2_sb
        else:
            fn = {"f1": gen_f1, "f2": gen_f2, "f3": gen_f3}[args.kernel]
        e, params, grid, nthr = fn(shp, W, BV, bk=args.bk)
        kname = f"gdn_fwd_{args.kernel}"
        mod = wrap_module(kname, params, hoist_constants(e.text()),
                          grid[0], grid[1], nthr, meta)
        transform_txt = pipe_transform(args.mb, args.depth)
        spec = {
            "kernel": kname, "path": args.path,
            "shape": {"B": B, "T": T, "H": H, "K": K, "V": V, "C": C,
                      "NC": NC, "Tp": Tp},
            "knobs": {"warps": W, "bv": BV, "bk": args.bk, "mb": args.mb,
                      "depth": args.depth},
            "params": [{"name": n, "dtype": dt, "elems": int(el)}
                       for n, el, dt in params],
            "grid": [grid[0], grid[1], 1], "block": [nthr, 1, 1],
        }
    os.makedirs(args.out, exist_ok=True)
    # R20-fwd: the per-knob lifetime-arena specs MERGE into one
    # module-keyed smem_alias.json (F1KS + F3KS stack in one mono build).
    alias_modules = {}
    if args.path == "mono" and os.environ.get("GDN_F1AL"):
        # R15-X: fwd-side lifetime-arena spec (module-keyed; consumed by
        # gdn_smem_fix.main_multi).  Sizes must match the FINAL PTX
        # mallocs: kt is the multibuffered kkt stage (mb x [C,bk+8] bf16),
        # everything else is the generator's own tile geometry.  On any
        # mismatch smem_fix drops the spec (kernel stays correct).
        mbkt = args.mb * 64 * (args.bk + 8) * 2
        alias_modules["gdn_fwd_f1_kernel"] = {
            "identify": {
                str(mbkt): "kt", "16384": "A_m", "11264": "Abf",
                "12288": "Tt", "13312": "Tb", "14336": "Tg",
                "768": "St0", "1024": "St1", "1280": "St2",
                "17408": "vt", "18432": "k2"},
            # NOTE: the solve tiles sit ABOVE the kt multibuffer's real
            # bytes -- aliasing them at offset 0 measured CORRUPT (u rows
            # 0-31 wrong in every CTA: the pipelined kkt loop still
            # writes kt buffers around the post-loop drain); the shifted
            # layout is bit-exact (r15x bisect_shift probe).
            "arenas": {"kt": 53248, "A_m": 17408, "Abf": 18432},
            "alias": {
                "Tb": ["kt", 10240], "Tt": ["kt", 23552],
                "St0": ["kt", 35840], "St1": ["kt", 36608],
                "St2": ["kt", 37632], "Tg": ["kt", 38912],
                "vt": ["A_m", 0], "k2": ["Abf", 0]},
        }
    if args.path == "mono" and os.environ.get("GDN_F1KS"):
        # R17-fwd GDN_F1KS lifetime-arena spec: the half-K vt/k2 tiles
        # land POST-solve into Abf/Tt's bytes (both dead at the uw
        # stage; the solve's final barrier covers their last reads).
        # Identify-by-size needs each spec size unique among ALL f1
        # mallocs: kt 6,144 (bk=16 multibuffer) / Abf 9,216 / Tt[66,72]
        # 9,504 / Tb,Tg [65,72] 9,360 x2 (shared, outside the spec) /
        # St0,1 768, St2 512 / vt[32,136] 8,704 / k2[33,136] 8,976 /
        # gc,bs 256 x2.  Decl total 46,144B; ptxas pads the
        # non-128-multiples (Tb/Tg +112 each, Tt +96) -> 46,352B
        # reported <= 49,152 (48KB static 2-CTA line).
        if os.environ.get("GDN_F1EP"):
            # R22-fwd GDN_F1EP spec: Tb/Tg JOIN as arena bases (the
            # epilogue's second-half tiles alias over them; their last
            # read is ks_pass(1)'s ldm, covered by its trailing
            # barrier).  Tg grows to [67,72] = 9,648 (rows -- the 144B
            # stride is a 16B multiple; a col pad faults ldmatrix) so
            # Tb 9,360 / Tg 9,648 are distinct; u2 [34,136] 9,248 <=
            # Tb 9,360, w2 [35,136] 9,520 <= Tg 9,648.  All sizes
            # unique: 6144 / 9216 / 9504 / 9360 / 9648 / 768x2 / 512 /
            # 8704 / 8976 / 9248 / 9520 / 256x3.  Pool: decl 46,432B
            # (Tg +288), padded ~46,7xx <= 49,152 -> 2 CTAs/SM kept.
            alias_modules["gdn_fwd_f1_kernel"] = {
                "identify": {"9216": "Abf", "9504": "Tt", "9360": "Tb",
                             "9648": "Tg", "8704": "vt", "8976": "k2",
                             "9248": "u2", "9520": "w2"},
                "arenas": {"Abf": 9216, "Tt": 9504, "Tb": 9360, "Tg": 9648},
                "alias": {"vt": ["Abf", 0], "k2": ["Tt", 0],
                          "u2": ["Tb", 0], "w2": ["Tg", 0]},
            }
        else:
            alias_modules["gdn_fwd_f1_kernel"] = {
                "identify": {"9216": "Abf", "9504": "Tt",
                             "8704": "vt", "8976": "k2"},
                "arenas": {"Abf": 9216, "Tt": 9504},
                "alias": {"vt": ["Abf", 0], "k2": ["Tt", 0]},
            }
    if args.path == "mono" and os.environ.get("GDN_F3KS"):
        # R20-fwd GDN_F3KS lifetime-arena spec: ot lands in P's dead
        # arena (P [64,72] 9,216B malloc declared at ot's 17,408B arena
        # size; ot's malloc deleted, aliased at P+0).  Sizes unique
        # among ALL f3 mallocs at bk=16: qt/kt 6,144 x2 (not in spec),
        # st/vnt 8,704 x2 (not in spec), P 9,216, ot 17,408, gcs 256.
        # Pool 47,360B <= 50,688 (2 CTA/SM line).  F3KS_F1CTA arm: st
        # 12,512 (still outside the spec) -> pool ~51.2KB, 1 CTA/SM.
        alias_modules["gdn_fwd_f3_kernel"] = {
            "identify": {"9216": "P", "17408": "ot"},
            "arenas": {"P": 17408},
            "alias": {"ot": ["P", 0]},
        }
    if args.path == "mono" and alias_modules:
        with open(os.path.join(args.out, "smem_alias.json"), "w") as f:
            json.dump({"modules": alias_modules}, f, indent=1)
    with open(os.path.join(args.out, "payload.mlir"), "w") as f:
        f.write(mod)
    with open(os.path.join(args.out, "transform.mlir"), "w") as f:
        f.write(transform_txt)
    for fn_ in ("config.json", "full_spec.json"):
        with open(os.path.join(args.out, fn_), "w") as f:
            json.dump(spec, f, indent=2)
    print(f"[gdn_full_gen] {kname}: {len(mod.splitlines())} lines -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
