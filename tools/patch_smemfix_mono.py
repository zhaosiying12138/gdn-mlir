#!/usr/bin/env python3
"""Extend tools/gdn_smem_fix.py for multi-kernel (mono) modules: extract
ALL gpu.binary assemblies, apply both fixes per module, merge into ONE PTX
file (single header, union of extern decls, all entry bodies), assemble ONE
cubin with 3 entries + shared globals."""
import sys

PATH = "tools/gdn_smem_fix.py"
src = open(PATH).read()
if "extract_all" in src:
    print("already patched")
    sys.exit(0)

old = '''def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    src = os.path.join(d, "kernel.ptx")
    ptx = open(src).read()'''
new = '''# --------------------------------------------------------------------------
# fix 3 (G5-D mono): multi-entry modules
# --------------------------------------------------------------------------

def unescape_mlir_string(s):
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\\\" and i + 1 < n:
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
            mp = {"n": "\\n", "t": "\\t", "r": "\\r", '"': '"',
                  "\\\\": "\\\\", "'": "'"}
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
            if c == "\\\\":
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
    r"gpu\\.launch_func\\s+@[\\w:]*(?:[:@](?P<name>\\w+))[^\\n]*?args\\((?P<args>[^)]*)\\)",
    re.S)


def merge_ptx(mods):
    """mods: [(name, fixed ptx)] -> one PTX module (header of mod0, union of
    .extern decls, all entry bodies concatenated)."""
    def split(p):
        i = p.find(".visible .entry")
        return p[:i], p[i:]

    pre0, body0 = split(mods[0][1])
    parts = [pre0]
    seen = set()
    for ln in pre0.splitlines(True):
        s = ln.strip()
        if s.startswith(".extern"):
            seen.add(s)
    for _n, p in mods[1:]:
        pre, body = split(p)
        for ln in pre.splitlines(True):
            s = ln.strip()
            if s.startswith(".extern") and s not in seen:
                seen.add(s)
                parts.append(ln)
    parts.append(body0)
    for _n, p in mods[1:]:
        _pre, body = split(p)
        parts.append("\\n")
        parts.append(body)
    return "".join(parts)


def main_multi(d):
    """mono path: fix every module, merge, assemble one cubin."""
    mods = extract_all_ptx(os.path.join(d, "asmobj.mlir"))
    if len(mods) < 2:
        return False
    largs = parse_all_launch_args(os.path.join(d, "asmobj.mlir"))
    fixed = []
    for name, ptx in mods:
        out, decls, n_patch = fix_ptx(ptx)
        kname, values = None, largs.get(name, [])
        out, n_subst = fix_params(out, values)
        fixed.append((name, out))
        print(f"[smem_fix]   {name}: cvta={n_patch} shared={len(decls)} "
              f"params={n_subst}")
    merged = merge_ptx(fixed)
    dst_ptx = os.path.join(d, "kernel_smemfix.ptx")
    dst_cub = os.path.join(d, "kernel_smemfix.cubin")
    with open(dst_ptx, "w") as f:
        f.write(merged)
    ptxas = pick_ptxas()
    r = subprocess.run([ptxas, "-v", "-arch=sm_120a", dst_ptx, "-o", dst_cub],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"ptxas failed for merged {dst_ptx}")
    n_ent = merged.count(".visible .entry")
    print(f"[smem_fix] {os.path.basename(d)}: MERGED {len(fixed)} modules, "
          f"{n_ent} entries -> {os.path.basename(dst_cub)}")
    return True


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    if main_multi(d):
        return
    src = os.path.join(d, "kernel.ptx")
    ptx = open(src).read()'''
assert src.count(old) == 1
src = src.replace(old, new)
open(PATH, "w").write(src)
print("[patch_smemfix_mono] ok")
