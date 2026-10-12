#!/usr/bin/env python3
"""G2 结构段修复：ours 三行各自取包络（regs/smem/spills）。幂等。"""
import pathlib

P = pathlib.Path(__file__).resolve().parent.parent / "tools" / "gdn_coverage.py"
s = P.read_text()

old = '''    R = lambda e, k: (e.get("resources") or {}).get(k)  # noqa: E731
    ours_struct = "PENDING"
    if ours_state == "LIVE":
        regs = [R(e, "regs") for e in ours if R(e, "regs") is not None]
        ours_struct = f"regs [{min(regs)}, {max(regs)}] over {len(ours)} configs" if regs else "no ptxas data in INDEX"
    struct_rows = [
        ("registers/thread", fla_env(None, lambda e: R(e, "regs")), qla_env(None, lambda e: R(e, "regs")), ours_struct,
         "fla rides the 255-reg cap with spills; qla pins <=128 via setmaxnreg"),
        ("dynamic smem (B)", fla_env(None, lambda e: R(e, "smem_dynamic")), qla_env(None, lambda e: R(e, "smem_static")), ours_struct,
         "sm_120 CTA limit 100KB/128KB; qla tiny (2KB) because TMA stages out of smem descriptors"),
        ("spills (B, max of st/ld)", fla_env(None, lambda e: max(R(e, "spill_stores") or 0, R(e, "spill_loads") or 0)),
         qla_env(None, lambda e: max(R(e, "spill_stores") or 0, R(e, "spill_loads") or 0)), ours_struct,
         "fla bwd kernels spill heavily (dhu 3.8KB); fused_fwd/bwd ~1.1-1.3KB"),'''

new = '''    R = lambda e, k: (e.get("resources") or {}).get(k)  # noqa: E731

    def ours_env(key):
        if ours_state != "LIVE":
            return "PENDING"
        vals = [R(e, key) for e in ours]
        vals = [v for v in vals if v is not None]
        return f"[{min(vals)}, {max(vals)}] over {len(ours)} configs" if vals else "?"

    ours_regs = ours_env("regs")
    ours_smem = ours_env("smem_dynamic")
    ours_spills = ours_env("spill_stores")
    struct_rows = [
        ("registers/thread", fla_env(None, lambda e: R(e, "regs")), qla_env(None, lambda e: R(e, "regs")), ours_regs,
         "fla rides the 255-reg cap with spills; qla pins <=128 via setmaxnreg"),
        ("dynamic smem (B)", fla_env(None, lambda e: R(e, "smem_dynamic")), qla_env(None, lambda e: R(e, "smem_static")), ours_smem,
         "sm_120 CTA limit 100KB/128KB; qla tiny (2KB) because TMA stages out of smem descriptors"),
        ("spills (B, max of st/ld)", fla_env(None, lambda e: max(R(e, "spill_stores") or 0, R(e, "spill_loads") or 0)),
         qla_env(None, lambda e: max(R(e, "spill_stores") or 0, R(e, "spill_loads") or 0)), ours_spills,
         "fla bwd kernels spill heavily (dhu 3.8KB); fused_fwd/bwd ~1.1-1.3KB"),'''

assert old in s, "struct block not found"
s = s.replace(old, new)
P.write_text(s)
print("patched OK")
