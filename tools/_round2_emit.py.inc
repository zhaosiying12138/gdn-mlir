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
    e.l(f"{', '.join(res)} = scf.for {iv} = %c0 to {e.c(Kred)} "
        f"step {e.c(bk)} iter_args("
        + ", ".join(f"{a} = {b}" for a, b in zip(iargs, inits))
        + f") -> ({tys}) {{")
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


def copy_s2g_vec(e, src, src_ty, flatfn, dst, dst_ty, rows, cols, nthr):
    """Whole-tile smem->global copy (bf16, scalar loads/stores: vector
    transfers on workgroup space do not lower through this pipeline, and
    vector.transfer_write to a flat global inside scf.while regions leaves
    an unrealized_conversion_cast -- both re-confirmed in G5-D probes)."""
    def body(i):
        r = divsi(e, i, e.c(cols))
        c = remsi(e, i, e.c(cols))
        v = e.v("sgl")
        e.l(f"{v} = memref.load {src}[{r}, {c}] : {src_ty}")
        idx = flatfn(r, c)
        e.l(f"memref.store {v}, {dst}[{idx}] : {dst_ty}")

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
    for d in (1, 2, 4, 8, 16, 32):
        e.l(f"scf.if {in64} {{")
        off = e.v("cso")
        e.l(f"{off} = arith.subi %tid, {e.c(d)} : index")
        ge = e.v("cge")
        e.l(f"{ge} = arith.cmpi sge, %tid, {e.c(d)} : index")
        offc = e.v("csc")
        e.l(f"{offc} = arith.select {ge}, {off}, %c0 : index")
        lv = e.v("cgl")
        e.l(f"{lv} = memref.load {gc}[{offc}, %c0] : {gc_ty}")
        vv = e.v("csv")
        e.l(f"{vv} = arith.select {ge}, {lv}, {zf} : f32")
        cur = e.v("cgc")
        e.l(f"{cur} = memref.load {gc}[%tid, %c0] : {gc_ty}")
        nv = e.v("cgn")
        e.l(f"{nv} = arith.addf {cur}, {vv} : f32")
        e.l(f"memref.store {nv}, {gc}[%tid, %c0] : {gc_ty}")
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
