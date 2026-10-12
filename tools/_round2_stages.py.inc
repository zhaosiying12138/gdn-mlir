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
    e = Em()
    emit_lane_precompute(e)
    e.l("%zfs = arith.constant 0.0 : f32")
    cH, cC, cNC = e.c(H), e.c(C), e.c(NC)
    TY = {
        "k": f"memref<{B*Tp*H*K}xbf16>", "v": f"memref<{B*Tp*H*V}xbf16>",
        "beta": f"memref<{B*Tp*H}xf32>", "g": f"memref<{B*Tp*H}xf32>",
        "u": f"memref<{B*H*NC*C*V}xbf16>", "w": f"memref<{B*H*NC*C*K}xbf16>",
        "gc_out": f"memref<{B*H*NC*C}xf32>"}

    ktile, ktile_ty = alloc2d(e, "kt", C, bk, BF)
    A_m, A_ty = alloc2d(e, "A_m", C, C, F32)
    Tb, Tb_ty = alloc2d(e, "Tb", C, C, BF)
    Tg, Tg_ty = alloc2d(e, "Tg", C, C, BF)
    vtile, vtile_ty = alloc2d(e, "vt", bk, V, BF)
    ktile2, ktile2_ty = alloc2d(e, "k2", bk, K, BF)
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

    emit_cumsum_beta_scan(e, "%g", TY["g"], bbg, "%beta", TY["beta"], bbg,
                          gc, gc_ty, bs, bs_ty, H)

    accs_kkt = emit_kloop(
        e, "kk", K, bk,
        stages=[{"tile": ktile, "ty": ktile_ty, "rows": C, "cols": bk,
                 "src": "%k", "sty": TY["k"], "kind": "rc", "base": kb_,
                 "rowmul": H * K}],
        gemms=[{"name": "kkt", "M": C, "N": C, "kt": True,
                "a": (ktile, ktile_ty, True), "b": (ktile, ktile_ty, True)}],
        W=W)
    e.l("gpu.barrier")
    run_epilogue(e, "kkt", C, C, W, accs_kkt,
                 emit_A_epilogue(e, A_m, A_ty, bs, bs_ty, gc, gc_ty))
    e.l("gpu.barrier")

    emit_solve2(e, A_m, A_ty, Tb, Tb_ty, Tg, Tg_ty, bs, bs_ty, gc, gc_ty, C)

    # gc_out store
    c64 = e.c(64)
    in64 = e.v("in64")
    e.l(f"{in64} = arith.cmpi ult, %tid, {c64} : index")
    e.l(f"scf.if {in64} {{")
    gv = e.v("gco")
    e.l(f"{gv} = memref.load {gc}[%tid, %c0] : {gc_ty}")
    gid = addi(e, gcb, "%tid")
    e.l(f"memref.store {gv}, %gc_out[{gid}] : {TY['gc_out']}")
    e.l("}")

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
        idx = addi(e, ub, addi(e, muli(e, row, e.c(V)), col))
        e.l(f"memref.store {t}, %u[{idx}] : {TY['u']}")

    def epi_w(mi, ni, ei, row, col, acc):
        x = acc_extract(e, acc, ei // 2, ei % 2, "ew")
        t = e.v("ewt"); e.l(f"{t} = arith.truncf {x} : f32 to bf16")
        idx = addi(e, wb, addi(e, muli(e, row, e.c(K)), col))
        e.l(f"memref.store {t}, %w[{idx}] : {TY['w']}")

    run_epilogue(e, "ug", C, V, W, accs, epi_u)
    run_epilogue(e, "wg", C, K, W, accs, epi_w)

    params = [("k", B * Tp * H * K, BF), ("v", B * Tp * H * V, BF),
              ("beta", B * Tp * H, F32), ("g", B * Tp * H, F32),
              ("u", B * H * NC * C * V, BF), ("w", B * H * NC * C * K, BF),
              ("gc_out", B * H * NC * C, F32)]
    return e, params, (NC, B * H), nthr


# ---------------------------------------------------------------------------
# F2 "scan" (round 2)
# ---------------------------------------------------------------------------

def gen_f2(shp, W, BV, bk=32):
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

    qtile, qtile_ty = alloc2d(e, "qt", C, bk, BF)
    ktile, ktile_ty = alloc2d(e, "kt", C, bk, BF)
    Stile, Stile_ty = alloc2d(e, "st", bk, BV, BF)
    vntile, vntile_ty = alloc2d(e, "vnt", bk, BV, BF)
    P, P_ty = alloc2d(e, "P", C, C, BF)
    gc, gc_ty = alloc2d(e, "gcs", C, 1, F32)

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
        e, "qo", K, bk,
        stages=[
            {"tile": qtile, "ty": qtile_ty, "rows": C, "cols": bk,
             "src": "%q", "sty": TY["q"], "kind": "rc", "base": qb,
             "rowmul": H * K},
            {"tile": ktile, "ty": ktile_ty, "rows": C, "cols": bk,
             "src": "%k", "sty": TY["k"], "kind": "rc", "base": qb,
             "rowmul": H * K},
            {"tile": Stile, "ty": Stile_ty, "rows": bk, "cols": BV,
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
        e, "oa", C, bk,
        stages=[{"tile": vntile, "ty": vntile_ty, "rows": bk, "cols": BV,
                 "src": "%v_new", "sty": TY["v_new"], "kind": "kc",
                 "base": vnb_vtc, "rowmul": V}],
        gemms=[{"name": "oa", "M": C, "N": BV, "kt": False,
                "a": (P, P_ty, False), "b": (vntile, vntile_ty, True)}],
        W=W)

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
                idx = addi(e, ob, addi(e, muli(e, row, e.c(H * V)),
                                       addi(e, "%vtc", col)))
                e.l(f"memref.store {t}, %o[{idx}] : {TY['o']}")

    params = [("q", B * Tp * H * K, BF), ("k", B * Tp * H * K, BF),
              ("gc", B * H * NC * C, F32),
              ("h_all", B * H * NC * K * V, BF),
              ("v_new", B * H * NC * C * V, BF),
              ("o", B * Tp * H * V, BF)]
    return e, params, (NC * B * H, VT), nthr


# ---------------------------------------------------------------------------
# mono module: the staged trio in ONE cubin (shared globals, no DtoD)
# ---------------------------------------------------------------------------

def gen_mono(shp, W, BV, bk=32):
    B, T, H, K, V, C, NC, Tp = shp
    VT = V // BV
    parts = []
    params = []

    def add_params(ps):
        for p in ps:
            if p not in params:
                params.append(p)

    for name, fn, grid in (("f1", gen_f1, (NC, B * H)),
                           ("f2", gen_f2, (B * H * VT, 1)),
                           ("f3", gen_f3, (NC * B * H, VT))):
        e, ps, _g, nthr = fn(shp, W, BV, bk=bk)
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
