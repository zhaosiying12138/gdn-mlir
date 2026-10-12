#!/usr/bin/env python3
"""Replace merge_ptx in the already-patched tools/gdn_smem_fix.py with the
symbol-renaming version (duplicate __gdn_smemN across merged entries
alias with wrong sizes -> OOB smem writes in the mono cubin)."""
import re

PATH = "tools/gdn_smem_fix.py"
src = open(PATH).read()
pat = re.compile(r"def merge_ptx\(mods\):.*?(?=def main_multi)", re.S)
assert pat.search(src), "merge_ptx not found"
new_fn = '''def merge_ptx(mods):
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
        return re.sub(r"__gdn_smem(\\d+)",
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
            if s.startswith(".extern") and s not in seen:
                seen.add(s)
                parts.append(ln)
    parts.append(rename_syms(body0, mods[0][0]))
    for n, p in mods[1:]:
        _pre, body = split(p)
        parts.append("\\n")
        parts.append(rename_syms(body, n))
    return "".join(parts)


'''
src = pat.sub(lambda _m: new_fn, src, count=1)
open(PATH, "w").write(src)
print("[patch_merge2] ok")
