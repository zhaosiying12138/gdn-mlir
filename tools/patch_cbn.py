#!/usr/bin/env python3
"""Fix the stash B-fragment column base (+16*Cb) by line surgery."""
p = "tools/gdn_full_gen.py"
lines = open(p).readlines()
done = False
for i, ln in enumerate(lines):
    if "rowb = addi(e, addi(e, e.c(16 * K)" in ln:
        j = i + 1
        while "mrs" not in lines[j]:
            j += 1
        new = [
            lines[i],
            "                # +16*Cb block offset on the B-frag column base\n",
            "                # (was missing: warp 0 read right blocks, warps\n",
            "                # 1/2 read the wrong Tt column blocks)\n",
            "                for ni in (0, 1):\n",
            '                    cbn = e.v("cbn")\n',
            '                    e.l(f"{cbn} = arith.addi %nc_sv_{ni}, "\n',
            '                         f"{e.c(16 * Cb)} : index")\n',
            '                    fb = e.v("fbs")\n',
            '                    e.l(f"{fb} = nvgpu.ldmatrix {Tt}[{rowb}, "\n',
            '                         f"{cbn}] numTiles = 2 transpose = true : "\n',
            '                         f"{Tt_ty} -> vector<2x2xbf16>")\n',
            '                    o = e.v("mrs")\n',
        ]
        lines[i:j + 1] = new
        done = True
        break
assert done, "stash emission not found"
open(p, "w").writelines(lines)
print("[cbn_fix] ok at line", i)
