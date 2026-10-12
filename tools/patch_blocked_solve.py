#!/usr/bin/env python3
"""Wire the blocked solve into gen_f1 in tools/gdn_full_gen.py (WSL truth)."""
PATH = "tools/gdn_full_gen.py"
src = open(PATH).read()
if "emit_solve_blocked" in src:
    print("already")
    raise SystemExit(0)

old = """    ktile, ktile_ty = alloc2d(e, "kt", C, bk, BF)
    A_m, A_ty = alloc2d(e, "A_m", C, C, F32)
    Tb, Tb_ty = alloc2d(e, "Tb", C, C, BF)
    Tg, Tg_ty = alloc2d(e, "Tg", C, C, BF)"""
new = """    ktile, ktile_ty = alloc2d(e, "kt", C, bk, BF)
    A_m, A_ty = alloc2d(e, "A_m", C, C, F32)
    Abf, Abf_ty = alloc2d(e, "Abf", C, C, BF)
    Tt, Tt_ty = alloc2d(e, "Tt", C, C, BF)
    St, St_ty = alloc2d(e, "St", 16, 16, BF)
    Tb, Tb_ty = alloc2d(e, "Tb", C, C, BF)
    Tg, Tg_ty = alloc2d(e, "Tg", C, C, BF)"""
assert src.count(old) == 1, "alloc anchor"
src = src.replace(old, new)

old = """    e.l("gpu.barrier")
    run_epilogue(e, "kkt", C, C, W, accs_kkt,
                 emit_A_epilogue(e, A_m, A_ty, bs, bs_ty, gc, gc_ty))
    e.l("gpu.barrier")

    emit_solve2(e, A_m, A_ty, Tb, Tb_ty, Tg, Tg_ty, bs, bs_ty, gc, gc_ty, C)"""
new = """    e.l("gpu.barrier")

    def epi_A2(mi, ni, ei, row, col, acc):
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
        t = e.v("Abt"); e.l(f"{t} = arith.truncf {sel} : f32 to bf16")
        e.l(f"memref.store {t}, {Abf}[{row}, {col}] : {Abf_ty}")
    run_epilogue(e, "kkt", C, C, W, accs_kkt, epi_A2)
    e.l("gpu.barrier")

    emit_solve_blocked(e, A_m, A_ty, Abf, Abf_ty, Tt, Tt_ty, St, St_ty,
                       Tb, Tb_ty, Tg, Tg_ty, bs, bs_ty, gc, gc_ty, C)"""
assert src.count(old) == 1, "solve anchor"
src = src.replace(old, new)
open(PATH, "w").write(src)
print("[patch_blocked_solve] ok")
