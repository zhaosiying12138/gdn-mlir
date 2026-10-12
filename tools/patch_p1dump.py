#!/usr/bin/env python3
"""Replace debug plumbing: G5D_DUMP_P1 dumps Tt right after phase 1."""
import re

PATH = "tools/gdn_full_gen.py"
s = open(PATH).read()
if "G5D_DUMP_P1" in s and "_dt1" not in s and "dump Tt right after phase 1" in s:
    print("already")
    raise SystemExit(0)
# remove NO_PHASE2 knob (breaks multibuffer on unused St tiles)
s = s.replace("""    import os as _os2
    for R in ((1, 2, 3) if not _os2.environ.get("G5D_NO_PHASE2")
              else ()):  # debug: G5D_NO_PHASE2 skips phase 2
        for Cb in range(0, R):
            Stc, Stc_ty = sts[Cb]""",
"""    for R in (1, 2, 3):
        for Cb in range(0, R):
            Stc, Stc_ty = sts[Cb]""")
# remove the old end-dump block in gen_f1
m = re.search(
    r"    import os as _os\n    if _os\.environ\.get\(\"G5D_DUMP_TT\"\):.*?\(\"tbb\", B \* H \* NC \* 64 \* 64, BF\)\]\n",
    s, re.S)
assert m, "old dump block"
s = s.replace(m.group(0), """    import os as _os
    if _os.environ.get("G5D_DUMP_TT") or _os.environ.get("G5D_DUMP_P1"):
        params = params + [("tt", B * H * NC * 64 * 64, BF),
                           ("tbb", B * H * NC * 64 * 64, BF)]
        if _os.environ.get("G5D_DUMP_TT"):
            def _dump(src, sty, dst, dty):
                def body(i):
                    r = divsi(e, i, e.c(64))
                    c = remsi(e, i, e.c(64))
                    v = e.v("ttv")
                    e.l(f"{v} = memref.load {src}[{r}, {c}] : {sty}")
                    e.l(f"memref.store {v}, {dst}["
                        f"{addi(e, muli(e, r, e.c(64)), c)}] : {dty}")
                emit_cell_while(e, 64 * 64, nthr, body)
            _dump(Tt, Tt_ty, "%tt",
                  f"memref<{B*H*NC*64*64}xbf16>")
            _dump(Tb, Tb_ty, "%tbb",
                  f"memref<{B*H*NC*64*64}xbf16>")
""")
# insert the phase-1 dump inside emit_solve_blocked after phase 1 barrier
anchor = """        e.l(f"memref.store {vgf}, {Tg}[{ccol}, {i}] : {Tg_ty}")
    e.l("}")
    e.l("gpu.barrier")"""
assert s.count(anchor) == 1, "phase1 tail"
s = s.replace(anchor, anchor + """
    import os as _os3
    if _os3.environ.get("G5D_DUMP_P1"):
        def _dp1(i):
            r = divsi(e, i, e.c(64))
            c = remsi(e, i, e.c(64))
            v = e.v("dp1")
            e.l(f"{v} = memref.load {Tt}[{r}, {c}] : {Tt_ty}")
            e.l(f"memref.store {v}, %tt["
                f"{addi(e, muli(e, r, e.c(64)), c)}] : "
                f"memref<{64*64*32}xbf16>")
        emit_cell_while(e, 64 * 64, 32 * 8, _dp1)""")
open(PATH, "w").write(s)
print("[p1dump] ok")
