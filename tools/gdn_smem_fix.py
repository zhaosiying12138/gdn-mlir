#!/usr/bin/env python3
"""gdn_smem_fix.py -- G3-A PTX repair for G2 cubins (two fixes).

Fix 1: static shared memory.
The LLVM/MLIR -gpu-lower-to-nvvm-pipeline lowers every memref.alloc() on
#gpu.address_space<workgroup> inside gpu.func to a DEVICE-SIDE
`call.uni malloc` + `cvta.to.shared` of the returned generic heap pointer
(upstream-documented behavior, see
llvm-mlir/mlir/test/Dialect/GPU/memref-to-llvm.mlir).  Two consequences on
real hardware:

  1. the cubin carries an unresolved CALL.ABS to the cudadevrt `malloc`
     trampoline (raw ptxas cubins do not link the device runtime), and
  2. even when malloc resolves, the heap pointer is reinterpreted as a
     shared-memory-window offset, so cp.async / ld.shared on the tile
     buffers touch out-of-range shared addresses -> illegal access /
     garbage.

Repair: pure PTX rewrite replicating what MLIR itself emits for static
shared (the ws-config mbarrier lowering already uses this idiom):

  * delete each malloc callseq,
  * replace `cvta.to.shared.u64 %rdY, %rdX` (where %rdX is a malloc result)
    with `mov.u64 %rdY, __gdn_smemN;`,
  * declare `.shared .align 16 .b8 __gdn_smemN[<malloc size>];`

Fix 2: constant-parameter elimination.
The `-transform-interpreter` (tiling) and `gdn-opt --gdn-lower-hatch` passes
hoist the gpu.launch-region's arith.constants back out to the host func,
where gpu-kernel-outlining turns each of them into a kernel parameter
(up to ~140 .b64 params of compile-time-constant ints).  The post-pipeline
MLIR kept by scripts/run_l2.sh (asmobj.mlir) still contains the host-side
`gpu.launch_func ... args(%c : i64, ...)` operand list, so every constant
parameter's VALUE is recoverable.  Repair:

  * parse asmobj.mlir: map launch_func arg SSA names to
    llvm.mlir.constant values (positional order == PTX param order),
  * in the PTX entry, replace each `ld.param::entry.<ty> %rX, [..param_N]`
    with an immediate `mov` and drop the now-dead `.param` declaration,
  * non-constant args (the tma config's host-side tensormap descriptors)
    are kept as real kernel parameters.

Both fixes re-assemble with the same ptxas used by scripts/run_l2.sh.

Usage:  python3 tools/gdn_smem_fix.py <config-dir>
Writes <config-dir>/kernel_smemfix.ptx and <config-dir>/kernel_smemfix.cubin.
Idempotent (overwrites).  Kernels needing neither fix are copied verbatim so
the timing harness always has a kernel_smemfix.cubin to load.
"""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys

REPO = os.path.expanduser("~/codebase/gdn-mlir")
TRITON_BIN = os.path.expanduser(
    "~/miniforge3/envs/gdn/lib/python3.12/site-packages/triton/backends/nvidia/bin")


def pick_ptxas():
    for cand in (os.path.join(TRITON_BIN, "ptxas-blackwell"),
                 os.path.join(TRITON_BIN, "ptxas"),
                 "/usr/local/cuda/bin/ptxas"):
        if os.path.exists(cand):
            return cand
    return "ptxas"


CALLSEQ = re.compile(
    r"\t\{ // callseq (\d+), 0\n"
    r"((?:\t[^\n]*\n)+?)"
    r"\t\} // callseq \1\n")  # closing brace omits the ", 0" suffix
SIZE_RE = re.compile(r"st\.param::func\.b64[ \t]+\[param0\],[ \t]+(\d+);")
RET_RE = re.compile(r"ld\.param::func\.b64[ \t]+(%r[d]\d+),[ \t]+\[retval0\];")
CVTA_RE = re.compile(
    r"\tcvta\.to\.shared(?:::\w+)?\.u64[ \t]+(%r[d]\d+),[ \t]+(%r[d]\d+);")


def fix_ptx(ptx: str, alias=None):
    """Returns (fixed_ptx, list_of_(sym,size), n_cvta_patched).

    G5-P r9: alias=<spec from gdn_bwd_gen.py smem_alias.json> enables
    LIFETIME-ARENA sharing of static smem across memref.allocs.  The MLIR
    program keeps separate allocs (SSA legality); at PTX level the
    aliased tiles' mallocs are deleted and their cvta conversions point
    at the arena base symbol PLUS a byte offset.  mallocs are identified
    by UNIQUE sizes (LLVM reorders malloc callseqs; order is not stable);
    on any mismatch the spec is DROPPED and the kernel keeps one decl per
    malloc (correct, just not overlapped)."""
    malloc_reg_size = {}
    decls = []
    reg_of_tile = {}
    tile_of_reg = {}
    arena_of = {}      # tile name -> declared arena size
    alias_of = {}      # tile name -> (base tile name, byte offset)

    def eat_callseq(m):
        body = m.group(2)
        if "malloc" not in body:
            return m.group(0)          # not a malloc callseq: keep verbatim
        sm = SIZE_RE.search(body)
        rm = RET_RE.search(body)
        if not sm or not rm:
            return m.group(0)
        malloc_reg_size[rm.group(1)] = int(sm.group(1))
        return ""                       # drop the whole callseq

    out = CALLSEQ.sub(eat_callseq, ptx)

    if alias:
        size_regs = {}
        for reg, sz in malloc_reg_size.items():
            size_regs.setdefault(sz, []).append(reg)
        ok = True
        dropped = set()
        for sz_s, tile in alias.get("identify", {}).items():
            regs = size_regs.get(int(sz_s), [])
            if len(regs) == 0:
                # G5-T round13: the malloc was optimized away ENTIRELY
                # (e.g. LLVM store-to-load-forwarded a same-lane epilogue
                # handoff into registers -- the wyb t2 pair).  There is no
                # cvta to patch and no bytes to assign: drop the tile from
                # the alias map and continue.
                print(f"[smem_fix] NOTE alias: size {sz_s} for '{tile}' "
                      f"matches 0 mallocs (optimized away) -- skipped")
                dropped.add(tile)
                continue
            if len(regs) != 1:
                print(f"[smem_fix] WARN alias: size {sz_s} for '{tile}' "
                      f"matches {len(regs)} mallocs -- aliasing disabled")
                ok = False
                break
            reg_of_tile[regs[0]] = tile
        tile_of_reg = dict(reg_of_tile)
        if ok:
            arena_of = {t: int(v) for t, v in alias.get("arenas", {}).items()}
            alias_of = {t: (a[0], int(a[1]))
                        for t, a in alias.get("alias", {}).items()
                        if t not in dropped}
            need = set(alias_of) | {b for b, _o in alias_of.values()}
            have = set(tile_of_reg.values())
            if not need <= have or not alias_of:
                print(f"[smem_fix] WARN alias: unidentified tiles "
                      f"{sorted(need - have)} -- aliasing disabled")
                ok = False
        if not ok:
            arena_of, alias_of, reg_of_tile, tile_of_reg = {}, {}, {}, {}

    n_patch = 0
    n_alias = 0
    sym_of_reg = {}

    def base_sym(base_tile):
        """symbol for an arena base tile (declared at ARENA size)."""
        breg = next(r for r, t in reg_of_tile.items() if t == base_tile)
        if breg not in sym_of_reg:
            sym = f"__gdn_smem{len(decls)}"
            decls.append((sym, arena_of[base_tile]))
            sym_of_reg[breg] = sym
        return sym_of_reg[breg]

    def patch_cvta(m):
        nonlocal n_patch, n_alias
        dst, src = m.group(1), m.group(2)
        if src not in malloc_reg_size:
            return m.group(0)
        # G4: one static .shared symbol PER MALLOC RESULT, shared by every
        # cvta conversion of it (plain .shared and .shared::cluster alike --
        # the tma config's cp.async.bulk.tensor destinations lower to
        # cvta.to.shared::cluster of the SAME malloc result; leaving those
        # unpatched made them read registers whose defining malloc callseq
        # was already deleted => garbage smem destination => device fault).
        tile = tile_of_reg.get(src)
        if tile is not None and tile in alias_of:
            base, off = alias_of[tile]
            sym = base_sym(base)
            n_patch += 1
            n_alias += 1
            return (f"\tmov.u64 \t{dst}, {sym};\n"
                    f"\tadd.u64 \t{dst}, {dst}, {off};")
        if src in sym_of_reg:
            sym = sym_of_reg[src]
        else:
            sz = malloc_reg_size[src]
            if tile is not None and tile in arena_of:
                sz = arena_of[tile]    # arena base: declare at arena size
            sym = f"__gdn_smem{len(decls)}"
            decls.append((sym, sz))
            sym_of_reg[src] = sym
        n_patch += 1
        return f"\tmov.u64 \t{dst}, {sym};"

    out = CVTA_RE.sub(patch_cvta, out)

    if decls:
        # .align 128: cp.async.bulk.tensor (UTMALDG) smem destinations must
        # be 128-byte aligned even with swizzle=none on sm_120a (G4: with
        # .align 16 the tiles can land at 16B-but-not-128B offsets and the
        # first TMA load faults with CUDA_ERROR_MISALIGNED_ADDRESS).
        lines = [f".shared .align 128 .b8 {s}[{n}];" for s, n in decls]
        marker = out.find(".visible .entry")
        if marker < 0:
            raise SystemExit("no .visible .entry found; refusing to patch")
        out = out[:marker] + "\n".join(lines) + "\n" + out[marker:]
    return out, decls, n_patch


# --------------------------------------------------------------------------
# fix 2: constant-parameter elimination
# --------------------------------------------------------------------------

LLVM_CONST_RE = re.compile(
    r"(%\d+) = llvm\.mlir\.constant\(([^)]*?) : (i64|i32|i1|f32|f64|bf16|f16)\)")
# G5-G round3: hoisted AGGREGATE constants (e.g. the bf16-mma accumulator
# zero `llvm.mlir.constant(dense<0.0> : vector<2x2xf32>) :
# !llvm.array<2 x vector<2xf32>>` -- gdn-opt's rewriter hoists launch-region
# constants to host args once a payload carries hatch ops).  Handled as a
# byte image: ld.param.{b64,b32}[+off] of the param become mov immediates.
LLVM_AGG_CONST_RE = re.compile(
    r"(%\d+) = llvm\.mlir\.constant\(dense<(.+?)> : "
    r"vector<(\d+)\s*x\s*(\d+)\s*x\s*(f32|f16|bf16|i32|i16|i8)>\) : "
    r"!llvm\.array<\3\s*x\s*vector<\4\s*x\s*\5>>")
LAUNCH_RE = re.compile(
    r"gpu\.launch_func\s+@[\w:]+.*?\bargs\(([^)]*)\)", re.S)


def _agg_bytes(dense_lit, elem_ty, d1, d2):
    """dense literal of vector<d1 x d2 x elem_ty> -> little-endian bytes
    (None if unparseable)."""
    lit = dense_lit.strip()
    elems = []
    try:
        if lit.startswith("["):
            tree = json.loads(re.sub(r"-?\d[\d.]*(?:e[+-]?\d+)?",
                                     lambda m: '"%s"' % m.group(0), lit))

            def walk(t):
                if isinstance(t, str):
                    elems.append(float(t))
                elif isinstance(t, list):
                    for v in t:
                        walk(v)
                else:
                    elems.append(float(t))

            walk(tree)
            if len(elems) != d1 * d2:
                return None
        else:
            v = float(lit)
            elems = [v] * (d1 * d2)
    except Exception:
        return None
    fmt = {"f32": "<f", "f16": "<e", "bf16": "<e", "i32": "<i",
           "i16": "<h", "i8": "<b"}[elem_ty]
    return b"".join(struct.pack(fmt, e) for e in elems)


def _const_bits(val_s: str, ty: str):
    """Value literal -> PTX immediate string."""
    val_s = val_s.strip()
    if ty in ("i64", "i32"):
        return str(int(val_s))
    if ty == "i1":
        return "1" if val_s == "true" else "0"
    f = float(val_s)
    if ty == "f32":
        return "0f%08X" % struct.unpack("<I", struct.pack("<f", f))[0]
    if ty == "f64":
        return "0d%016X" % struct.unpack("<Q", struct.pack("<d", f))[0]
    if ty == "bf16":  # top 16 bits of the f32 encoding
        return "0x%04X" % (struct.unpack("<I", struct.pack("<f", f))[0] >> 16)
    return "0x%04X" % struct.unpack("<H", struct.pack("<e", f))[0]


def parse_launch_args(asmobj_path: str):
    """-> (kernel_name, [value-or-None per param]).  Value is an immediate
    string (scalar constant) or a bytes image (aggregate constant); None =
    dynamic (kept as a real kernel parameter)."""
    try:
        txt = open(asmobj_path).read()
    except FileNotFoundError:
        return None, []
    consts = {}
    for m in LLVM_CONST_RE.finditer(txt):
        consts[m.group(1)] = (m.group(2), m.group(3))
    aggs = {}
    for m in LLVM_AGG_CONST_RE.finditer(txt):
        b = _agg_bytes(m.group(2), m.group(5), int(m.group(3)),
                       int(m.group(4)))
        if b is not None:
            aggs[m.group(1)] = b
    m = LAUNCH_RE.search(txt)
    if not m:
        return None, []
    args = [a.strip() for a in m.group(1).split(",") if a.strip()]
    vals = []
    for a in args:
        ssa = a.split(":")[0].strip()
        if ssa in consts:
            vals.append(_const_bits(*consts[ssa]))
        elif ssa in aggs:
            vals.append(aggs[ssa])       # aggregate constant -> byte image
        else:
            vals.append(None)           # dynamic (e.g. tensormap descriptor)
    # kernel name: last component of the launch_func symbol
    nm = re.search(r"gpu\.launch_func\s+@[\w:]*[:@](\w+)", txt)
    return (nm.group(1) if nm else None), vals


PARAM_DECL_RE = re.compile(
    r"^\t\.param (?:\.align \d+ )?\.\w+ (\w+_param_\d+)(?:\[\d+\])?(,?)$", re.M)
LDPARAM_RE = re.compile(
    r"\tld\.param::entry\.(b64|b32|b16|b8|f32|f64)[ \t]+(%\w+),[ \t]+"
    r"\[(\w+_param_\d+)(?:\+(\d+))?\];")

_LD_WIDTH = {"b64": 8, "f64": 8, "b32": 4, "f32": 4, "b16": 2, "b8": 1}


def _bytes_imm(ty, chunk):
    iv = int.from_bytes(chunk, "little")
    if ty == "f32":
        return "0f%08X" % iv
    if ty == "f64":
        return "0d%016X" % iv
    return "0x%0*X" % (len(chunk) * 2, iv)


def fix_params(ptx: str, values):
    """Substitute constant params with immediates; drop their declarations.

    values: positional list (immediate string, bytes image, or None) aligned
    to <kernel>_param_0..N as parsed from asmobj.mlir."""
    if not values:
        return ptx, 0
    # map param name -> value-or-None
    name_val = {}
    subst = 0
    for m in PARAM_DECL_RE.finditer(ptx):
        name = m.group(1)
        idx = int(name.rsplit("_param_", 1)[1])
        v = values[idx] if idx < len(values) else None
        name_val[name] = v

    def sub_ld(m):
        nonlocal subst
        ty, reg, name = m.group(1), m.group(2), m.group(3)
        off = int(m.group(4) or 0)
        v = name_val.get(name)
        if v is None:
            return m.group(0)
        if isinstance(v, (bytes, bytearray)):
            w = _LD_WIDTH[ty]
            chunk = bytes(v[off:off + w])
            if len(chunk) != w:
                return m.group(0)   # out-of-range slice: keep the load
            subst += 1
            return f"\tmov.{ty} \t{reg}, {_bytes_imm(ty, chunk)};"
        subst += 1
        return f"\tmov.{ty} \t{reg}, {v};"

    out = LDPARAM_RE.sub(sub_ld, ptx)

    def drop_decl(m):
        return "" if name_val.get(m.group(1)) is not None else m.group(0)

    out = PARAM_DECL_RE.sub(drop_decl, out)
    # tidy the signature: trailing comma on the last remaining decl, and
    # collapse a fully-emptied parameter list to `name()` (ptxas rejects
    # `name(\n\n)` with blank lines inside the parens)
    out = re.sub(
        r"(\.param [.\w ]+?\w+_param_\d+(?:\[\d+\])?),(\s*\))", r"\1\2", out)
    out = re.sub(r"(\.visible \.entry \w+)\(\s*\)", r"\1()", out)
    return out, subst


# --------------------------------------------------------------------------
# fix 3 (G5-D mono): multi-entry modules
# --------------------------------------------------------------------------

def unescape_mlir_string(s):
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt in "xX" and i + 3 < n:
                out.append(chr(int(s[i + 2:i + 4], 16)))
                i += 4
                continue
            if i + 2 < n and all(ch in "0123456789abcdefABCDEF"
                                 for ch in s[i + 1:i + 3]):
                out.append(chr(int(s[i + 1:i + 3], 16)))
                i += 3
                continue
            mp = {"n": "\n", "t": "\t", "r": "\r", '"': '"',
                  "\\": "\\", "'": "'"}
            if nxt in mp:
                out.append(mp[nxt])
                i += 2
                continue
            out.append(nxt)
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def extract_all_ptx(asmobj_path):
    """-> [(kernel_name, ptx_text)] for every gpu.binary assembly attr."""
    try:
        txt = open(asmobj_path, errors="replace").read()
    except FileNotFoundError:
        return []
    mods = []
    key = 'assembly = "'
    idx = 0
    while True:
        start = txt.find(key, idx)
        if start < 0:
            break
        # nearest preceding gpu.binary @NAME
        b = txt.rfind("gpu.binary @", 0, start)
        name = txt[b + 12:].split()[0] if b >= 0 else f"mod{len(mods)}"
        i = start + len(key)
        buf = []
        while i < len(txt):
            c = txt[i]
            if c == "\\":
                buf.append(txt[i:i + 3])
                i += 3
                continue
            if c == '"':
                break
            buf.append(c)
            i += 1
        text = unescape_mlir_string("".join(buf))
        if ".version" in text and ".target" in text:
            mods.append((name, text))
        idx = i + 1
    return mods


def parse_all_launch_args(asmobj_path):
    """-> {kernel_name: [immediate-or-None]} for every launch_func."""
    try:
        txt = open(asmobj_path, errors="replace").read()
    except FileNotFoundError:
        return {}
    consts = {}
    for m in LLVM_CONST_RE.finditer(txt):
        consts[m.group(1)] = (m.group(2), m.group(3))
    out = {}
    for m in LAUNCH_ALL_RE.finditer(txt):
        kname = m.group("name")
        args = [a.strip() for a in m.group("args").split(",") if a.strip()]
        vals = []
        for a in args:
            ssa = a.split(":")[0].strip()
            vals.append(_const_bits(*consts[ssa]) if ssa in consts else None)
        out[kname] = vals
    return out


LAUNCH_ALL_RE = re.compile(
    r"gpu\.launch_func\s+@[\w:]*(?:[:@](?P<name>\w+))[^\n]*?args\((?P<args>[^)]*)\)",
    re.S)


LD_SH_B16 = re.compile(
    r"\t(ld\.shared(?:::\w+)?\.b16)[ \t]+(%\w+),[ \t]+\[(%\w+)(?:\+(\d+))?\];")
ST_G_B16 = re.compile(
    r"\t(st\.global\.b16)[ \t]+\[(%\w+)(?:\+(\d+))?\],[ \t]+(%\w+);")


def widen_stores(ptx: str, tag: str = ""):
    """GDN_WST: fuse adjacent b16 ld.shared/st.global pairs into b32 ops.

    Returns (ptx, n_fused).  Byte-exact by construction: the fused 32-bit
    register holds exactly the two adjacent 16-bit cells; st.global.b32
    writes the same bytes to the same addresses (4B-aligned by the
    generator's even-pair cell mapping: pairs start at even cells, tile
    row strides and global flat strides are even)."""
    lines = ptx.split("\n")
    ld_def = {}    # b16 srcreg -> (line_idx, base, off)
    uses = {}      # b16 srcreg -> use count
    st_sites = []  # (line_idx, gbase, goff, srcreg)
    for i, ln in enumerate(lines):
        m = LD_SH_B16.match(ln)
        if m:
            ld_def[m.group(2)] = (i, m.group(3), int(m.group(4) or 0))
            continue
        m = ST_G_B16.match(ln)
        if m:
            st_sites.append((i, m.group(2), int(m.group(3) or 0),
                             m.group(4)))
    for _i, _b, _o, r in st_sites:
        uses[r] = uses.get(r, 0) + 1
    cand = {}      # store line -> (ld line, lbase, loff, gbase, goff)
    for i, gb, go, r in st_sites:
        d = ld_def.get(r)
        if d and uses.get(r) == 1:
            cand[i] = (d[0], d[1], d[2], gb, go)
    # group candidate stores by their LOAD address (base, off)
    by_ld = {}
    for st_i, (ld_i, lb, lo, gb, go) in cand.items():
        by_ld.setdefault((lb, lo), []).append((st_i, ld_i, gb, go))
    out = list(lines)
    consumed = set()
    n = 0
    new_reg = [9000]       # fresh 32-bit regs, far above LLVM's numbering
    for (lb, lo) in sorted(by_ld, key=lambda k: (k[0], k[1])):
        if (lb, lo + 2) not in by_ld:
            continue
        for (st_a, ld_a, ga, oa) in by_ld[(lb, lo)]:
            if st_a in consumed or ld_a in consumed:
                continue
            for (st_b, ld_b, gb_, ob) in by_ld[(lb, lo + 2)]:
                if (st_b in consumed or ld_b in consumed
                        or ld_b == ld_a or st_b == st_a):
                    continue
                if gb_ != ga or ob != oa + 2:
                    continue
                # order sanity: each load precedes its own store; the
                # LOW-offset load precedes the high-offset load.  Store
                # order is NOT constrained (LLVM often schedules the +2
                # store first).
                if not (ld_a < st_a and ld_b < st_b and ld_a < ld_b):
                    continue
                rk = "%%r%d" % new_reg[0]
                new_reg[0] += 1
                soff = "+%d" % lo if lo else ""
                # the fused store keeps the LOW global offset of the pair
                go_lo, go_hi = (oa, ob) if oa < ob else (ob, oa)
                if go_hi != go_lo + 2:
                    continue
                goff = "+%d" % go_lo if go_lo else ""
                out[ld_a] = "\tld.shared.b32 \t%s, [%s%s];" % (rk, lb,
                                                                 soff)
                out[st_a] = "\tst.global.b32 \t[%s%s], %s;" % (ga, goff,
                                                                 rk)
                out[ld_b] = None
                out[st_b] = None
                consumed.update((ld_a, ld_b, st_a, st_b))
                n += 1
                break
    out = [x for x in out if x is not None]
    if n and new_reg[0] > 9000:
        # declare the freshly minted b32 regs as an explicit comma list
        # (from-to range decls are rejected by this ptxas; count-form
        # %r<N> decls cannot name high indices)
        decl = "\t.reg .b32 " + ", ".join("%%r%d" % i
                                          for i in range(9000, new_reg[0])) \
            + ";"
        last = max(i for i, ln in enumerate(out) if ln.startswith("\t.reg"))
        out.insert(last + 1, decl)
    return "\n".join(out), n


# R16-Z (GDN_WSP): loop-bound sanity for the shuffle-pair transform.
# The cell-pass while loops lower to:  body; add.s64 %ctr,%ctr,STEP;
# setp.lt.u64 %p,%ctr,BOUND; @%p bra.  Counter starts at %tid, so warp
# uniform iteration (shfl.sync legality) needs total % nthr == 0, i.e.
# (BOUND+STEP) % STEP == 0, and STEP % 32 == 0 (whole warps step
# together).  Sites failing this are left untouched.
_LOOP_STEP = re.compile(r"\tadd\.s64[ \t]+(%\w+),[ \t]*\1,[ \t]*(\d+);")
_LOOP_SETPLT = re.compile(
    r"\tsetp\.lt\.u64[ \t]+(%\w+),[ \t]+(%\w+),[ \t]+(\d+);")
_LOOP_BRA = re.compile(r"\t@(%\w+)[ \t]+bra[ \t]+(\$L__\w+);")


def shuffle_pair_stores(ptx: str, tag: str = ""):
    """GDN_WSP (R16-Z): warp-shuffle paired b32 GLOBAL stores for the f2
    cell passes (BV=32 tiles -- excluded from GDN_WST's quartet fusion by
    the cols>=64 rule because same-thread b32 shared loads there are
    2-way bank-conflicted across the 80B row stride; measured +4.3us,
    r15 `ab_pairoonly`).

    Route: keep the conflict-free SINGLE-cell shared loads exactly as
    HEAD emits them (same addresses, same bank pattern), then pair ACROSS
    LANES: consecutive lanes hold consecutive cells, so lane 2j holds
    cell pair (2k, 2k+1)'s low half and lane 2j+1 the high half.  One
    `shfl.sync.bfly.b32 delta=1` gives every lane its partner's bits;
    even lanes pack own|partner<<16 (prmt) and store ONE st.global.b32
    at their own (pair-aligned, 4B) address; odd lanes store nothing.
    Halves the global-store lane-slots; the shared-load geometry -- the
    thing that killed tile-level pairing -- is untouched.

    Byte-exact by construction: the packed 32-bit word is exactly the
    two adjacent b16 cells (little-endian), written to the same address
    the b16 pair would touch (even-cell byte offset, 4B-aligned).

    Site eligibility (all checked, else the site is skipped):
      * `st.global.b16 [G], %rs` with %rs defined by an `ld.shared.b16`
        earlier in the same basic block (no label in between);
      * the store's enclosing while-loop steps a u64 counter by STEP
        (STEP % 32 == 0) under bound BOUND with (BOUND+STEP) % STEP == 0
        -- warp-uniform iteration, shfl.sync legal;
      * store byte offset 0 (pair base).
    Requires (generator-side, true for every fwd pass): total cells %
    nthr == 0, nthr even (parity(lane) == parity(cell)), even global
    flat stride (V=128), cols even.  Returns (ptx, n_sites)."""
    lines = ptx.split("\n")
    ld_def = {}   # b16 srcreg -> line index of its ld.shared.b16
    for i, ln in enumerate(lines):
        m = LD_SH_B16.match(ln)
        if m:
            ld_def[m.group(2)] = i
    sites = []    # (st_line, ld_line, gbase, srcreg)
    for i, ln in enumerate(lines):
        m = ST_G_B16.match(ln)
        if not m or int(m.group(3) or 0) != 0:
            continue
        r = m.group(4)
        d = ld_def.get(r)
        if d is None or d >= i:
            continue
        # same basic block: no label between load and store
        if any(x.startswith("$L__") for x in lines[d:i]):
            continue
        sites.append((i, d, m.group(2), r))
    out = list(lines)
    replaced = set()
    n = 0
    # fresh registers, above widen_stores' 9000 block
    R_OWN, R_PAR, R_TID, R_AND, R_PK = ("%r9100", "%r9101", "%r9102",
                                        "%r9103", "%r9104")
    P_EV = "%p900"
    for st_i, ld_i, gbase, r in sites:
        if st_i in replaced or ld_i in replaced:
            continue
        # find the enclosing while-loop: counter step + bound + bra
        # within the 10 lines after the store (the cell-pass bodies are
        # tight: ld, st, 3 adds, setp, bra)
        step = bound = None
        for j in range(st_i + 1, min(st_i + 11, len(lines))):
            m = _LOOP_STEP.match(lines[j])
            if m and step is None:
                step = int(m.group(2))
            m = _LOOP_SETPLT.match(lines[j])
            if m and bound is None:
                bound = int(m.group(3))
            if _LOOP_BRA.match(lines[j]):
                break
        if not step or not bound or step % 32 or (bound + step) % step:
            continue
        block = [
            lines[ld_i],
            "\tcvt.u32.u16 \t%s, %s;" % (R_OWN, r),
            "\tshfl.sync.bfly.b32 \t%s, %s, 1, 31, 0xffffffff;"
            % (R_PAR, R_OWN),
            "\tprmt.b32 \t%s, %s, %s, 21520;" % (R_PK, R_OWN, R_PAR),
            "\t@%s st.global.b32 \t[%s], %s;" % (P_EV, gbase, R_PK),
        ]
        # predicate + tid regs are loop-invariant: compute once at the
        # top of the entry body (right after the last .reg decl)
        out[st_i] = None
        out[ld_i] = "\n".join(block)
        replaced.update((st_i, ld_i))
        n += 1
    if not n:
        return ptx, 0
    out = [x for x in out if x is not None]
    pre = [
        "\tmov.u32 \t%s, %%tid.x;" % R_TID,
        "\tand.b32 \t%s, %s, 1;" % (R_AND, R_TID),
        "\tsetp.eq.b32 \t%s, %s, 0;" % (P_EV, R_AND),
    ]
    decl = ("\t.reg .pred %s;\n" % P_EV
            + "\t.reg .b32 " + ", ".join([R_OWN, R_PAR, R_TID, R_AND,
                                          R_PK]) + ";")
    last = max(i for i, ln in enumerate(out) if ln.startswith("\t.reg"))
    out[last + 1:last + 1] = [decl] + pre
    return "\n".join(out), n


def apply_maxnreg_k(ptx):
    """R14-U: GDN_MAXREG_K='f3:168,f2:128' injects per-ENTRY .maxnreg
    directives into the fixed PTX (the ptxas --maxrregcount flag is per
    FILE; the PTX .maxnreg directive is per kernel -- the mono cubin
    carries three kernels with different reg budgets).  Match is by
    substring on the entry name.  Returns (ptx, {entry: cap})."""
    spec = os.environ.get("GDN_MAXREG_K")
    if not spec:
        return ptx, {}
    want = {}
    for part in spec.split(","):
        k, n = part.split(":")
        want[k.strip()] = int(n)
    lines = ptx.split("\n")
    out = []
    applied = {}
    for ln in lines:
        out.append(ln)
        m = re.match(r"\s*\.visible\s+\.entry\s+([\w.$]+)", ln)
        if m:
            name = m.group(1)
            key = next((k for k in want if k in name), None)
            if key:
                out.append(f"\t.maxnreg {want[key]}")
                applied[name] = want[key]
    n_ent = ptx.count(".visible .entry")
    if len(applied) != len(want) or (n_ent and not applied):
        print(f"[smem_fix] WARN maxnreg_k: wanted {want} applied {applied} "
              f"({n_ent} entries) -- check the spec keys")
    return "\n".join(out), applied


def merge_ptx(mods):
    """mods: [(name, fixed ptx)] -> one PTX module.  Header of mod0, union
    of .extern decls, all entry bodies concatenated.  Per-module static
    .shared symbols (__gdn_smemN from fix 1) are RENAMED per kernel:
    ptxas silently unifies duplicate .shared names across entries, which
    aliased the kernels' tiles with mismatched sizes (G5-D: the mono f3
    wrote OOB through f1's smaller __gdn_smem0)."""
    def split(p):
        i = p.find(".visible .entry")
        return p[:i], p[i:]

    def rename_syms(text, tag):
        return re.sub(r"__gdn_smem(\d+)",
                      lambda m: "__gdn_smem_%s_%s" % (tag, m.group(1)),
                      text)

    pre0, body0 = split(mods[0][1])
    parts = [rename_syms(pre0, mods[0][0])]
    seen = set()
    for ln in pre0.splitlines(True):
        s = ln.strip()
        if s.startswith(".extern"):
            seen.add(s)
    for _n, p in mods[1:]:
        pre, _body = split(p)
        for ln in pre.splitlines(True):
            s = ln.strip()
            if s.startswith(".shared"):
                parts.append(rename_syms(ln, _n))
            elif s.startswith(".extern") and s not in seen:
                seen.add(s)
                parts.append(ln)
    parts.append(rename_syms(body0, mods[0][0]))
    for n, p in mods[1:]:
        _pre, body = split(p)
        parts.append("\n")
        parts.append(rename_syms(body, n))
    return "".join(parts)


def main_multi(d):
    """mono path: fix every module, merge, assemble one cubin."""
    mods = extract_all_ptx(os.path.join(d, "asmobj.mlir"))
    if len(mods) < 2:
        return False
    largs = parse_all_launch_args(os.path.join(d, "asmobj.mlir"))
    # R15-X: fwd-side lifetime-arena specs (module-keyed).  The writer
    # lives in gdn_full_gen.py (GDN_F1AL); on any size mismatch fix_ptx
    # drops the spec and the kernel keeps one decl per malloc (correct).
    amods = {}
    ap = os.path.join(d, "smem_alias.json")
    if os.path.exists(ap):
        amods = json.load(open(ap)).get("modules", {})
    fixed = []
    for name, ptx in mods:
        out, decls, n_patch = fix_ptx(ptx, alias=amods.get(name))
        kname, values = None, largs.get(name, [])
        out, n_subst = fix_params(out, values)
        n_wst = 0
        if os.environ.get("GDN_WST"):
            scope = os.environ.get("GDN_WST_SCOPE", "")
            if scope in name:
                out, n_wst = widen_stores(out, name)
        # R16-Z: shuffle-pair stores -- run AFTER widen_stores so only the
        # singles WST could not fuse (f2's BV=32 cell passes) transform.
        n_wsp = 0
        if os.environ.get("GDN_WSP"):
            scope = os.environ.get("GDN_WSP_SCOPE", "")
            if scope in name:
                out, n_wsp = shuffle_pair_stores(out, name)
        fixed.append((name, out))
        print(f"[smem_fix]   {name}: cvta={n_patch} shared={len(decls)} "
              f"params={n_subst} wst={n_wst} wsp={n_wsp}")
    merged = merge_ptx(fixed)
    merged, mrk = apply_maxnreg_k(merged)
    dst_ptx = os.path.join(d, "kernel_smemfix.ptx")
    dst_cub = os.path.join(d, "kernel_smemfix.cubin")
    with open(dst_ptx, "w") as f:
        f.write(merged)
    ptxas = pick_ptxas()
    cmd = [ptxas, "-v", "-arch=sm_120a", dst_ptx, "-o", dst_cub]
    # G5-P r9 experiment knob: cap registers (occupancy probe)
    mr = os.environ.get("GDN_MAXREG")
    if mr:
        cmd += ["--maxrregcount", mr]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"ptxas failed for merged {dst_ptx}")
    if mrk:
        print("[smem_fix] ptxas -v (maxnreg_k):\n" + r.stdout)
    n_ent = merged.count(".visible .entry")
    print(f"[smem_fix] {os.path.basename(d)}: MERGED {len(fixed)} modules, "
          f"{n_ent} entries -> {os.path.basename(dst_cub)}")
    return True


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    if main_multi(d):
        return
    src = os.path.join(d, "kernel.ptx")
    ptx = open(src).read()
    # G5-P r9: lifetime-arena spec (written by gdn_bwd_gen.py alongside
    # config.json); absent -> classic one-decl-per-malloc behavior.
    alias = None
    ap = os.path.join(d, "smem_alias.json")
    if os.path.exists(ap):
        alias = json.load(open(ap))
    out, decls, n_patch = fix_ptx(ptx, alias=alias)
    pool = sum(n for _s, n in decls)
    n_malloc = ptx.count("call.uni")

    kname, values = parse_launch_args(os.path.join(d, "asmobj.mlir"))
    n_dyn = sum(1 for v in values if v is None)
    out, n_subst = fix_params(out, values)

    dst_ptx = os.path.join(d, "kernel_smemfix.ptx")
    dst_cub = os.path.join(d, "kernel_smemfix.cubin")
    with open(dst_ptx, "w") as f:
        f.write(out)
    ptxas = pick_ptxas()
    cmd = [ptxas, "-v", "-arch=sm_120a", dst_ptx, "-o", dst_cub]
    # G5-P r9 experiment knob: cap registers (occupancy probe)
    mr = os.environ.get("GDN_MAXREG")
    if mr:
        cmd += ["--maxrregcount", mr]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"ptxas failed for {dst_ptx}")
    print(f"[smem_fix] {os.path.basename(d)}: malloc_callseqs={n_malloc} "
          f"cvta_patched={n_patch} shared_decls={decls} "
          f"static_smem_total={pool}B "
          f"const_params_substituted={n_subst} dynamic_params_kept={n_dyn} "
          f"-> {os.path.basename(dst_cub)}")


if __name__ == "__main__":
    main()
