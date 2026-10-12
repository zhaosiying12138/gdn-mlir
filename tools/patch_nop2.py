#!/usr/bin/env python3
"""Add G5D_NO_PHASE2 knob to emit_solve_blocked (debug bisection)."""
import re

PATH = "tools/gdn_full_gen.py"
s = open(PATH).read()
if "G5D_NO_PHASE2" in s:
    print("already")
    raise SystemExit(0)
# drop broken P1 dump if present
m = re.search(
    r"    import os as _os\n    if _os\.environ\.get\(\"G5D_DUMP_P1\"\):.*?_dt1\)\n",
    s, re.S)
if m:
    s = s.replace(m.group(0), "")
    print("removed p1 dump")
# also drop its params tail
s = s.replace("""        if _os.environ.get("G5D_DUMP_P1"):
            params = params + [("am", 131072, F32),
                               ("tt1f", 131072, F32)]
""", "")
old = "    for R in (1, 2, 3):\n        for Cb in range(0, R):\n            Stc, Stc_ty = sts[Cb]"
new = ("    import os as _os2\n"
       "    for R in ((1, 2, 3) if not _os2.environ.get(\"G5D_NO_PHASE2\")\n"
       "              else ()):  # debug: G5D_NO_PHASE2 skips phase 2\n"
       "        for Cb in range(0, R):\n            Stc, Stc_ty = sts[Cb]")
assert s.count(old) == 1, "R loop anchor"
s = s.replace(old, new)
open(PATH, "w").write(s)
print("[no_p2] ok")
