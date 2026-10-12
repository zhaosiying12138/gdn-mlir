#!/usr/bin/env python3
"""G2 merge fix: (1) OP_RE 接受 MLIR/NVVM 的多行指令（opcode 行尾即换行）；
(2) load_ours 优先用 INDEX.json 内嵌 meta 的资源字段（G2-A schema）。幂等。"""
import pathlib

P = pathlib.Path(__file__).resolve().parent.parent / "tools" / "gdn_coverage.py"
s = P.read_text()

old_re = 'OP_RE = re.compile(r"^[ \\t]*(?:@!?%?\\w+[ \\t]+)?([a-z][a-zA-Z0-9_.:]*)(?=[ \\t;(])", re.M)'
new_re = 'OP_RE = re.compile(r"^[ \\t]*(?:@!?%?\\w+[ \\t]+)?([a-z][a-zA-Z0-9_.:]*)(?=[ \\t;(\\n])", re.M)'
assert old_re in s, "OP_RE pattern not found"
s = s.replace(old_re, new_re)

old_add = '''    def add(ptx_path: Path, label: str):
        if not ptx_path.is_file():
            return
        ptx = ptx_path.read_text(errors="replace")
        ours.append({
            "ref": "ours", "kernel": label, "config": label,
            "path": str(ptx_path.relative_to(REPO)),
            **extract_ptx(ptx),
            "resources": load_meta_resources(ptx_path.parent / "meta.json"),
        })'''
new_add = '''    def add(ptx_path: Path, label: str, idx_meta: dict | None = None):
        if not ptx_path.is_file():
            return
        ptx = ptx_path.read_text(errors="replace")
        res = load_meta_resources(ptx_path.parent / "meta.json")
        m = idx_meta or {}
        res = {
            "regs": m.get("registers") or res.get("regs"),
            "smem_static": m.get("smem_static", res.get("smem_static")),
            "smem_dynamic": m.get("smem_dynamic", res.get("smem_dynamic")),
            "spill_stores": m.get("spill_stores", res.get("spill_stores", 0)) or 0,
            "spill_loads": m.get("spill_loads", res.get("spill_loads", 0)) or 0,
            "num_warps": m.get("num_warps", res.get("num_warps")),
        }
        ours.append({
            "ref": "ours", "kernel": label, "config": label,
            "path": str(ptx_path.relative_to(REPO)),
            **extract_ptx(ptx),
            "resources": res,
        })'''
assert old_add in s, "add() not found"
s = s.replace(old_add, new_add)

old_call = '''                for c in cand:
                    if c.is_file():
                        add(c, it.get("name") or it.get("config") or c.stem)
                        break'''
new_call = '''                for c in cand:
                    if c.is_file():
                        add(c, it.get("name") or it.get("config") or c.stem,
                            idx_meta=it.get("meta") if isinstance(it.get("meta"), dict) else None)
                        break'''
assert old_call in s, "add call not found"
s = s.replace(old_call, new_call)

P.write_text(s)
print("patched OK")
