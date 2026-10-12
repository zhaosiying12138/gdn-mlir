#!/usr/bin/env python3
"""Extract PTX from an MLIR gpu.binary op emitted with cubin-format=assembly.

The gpu-lower-to-nvvm-pipeline with cubin-format=assembly attaches the PTX
text as an `assembly = "..."` string attribute on #gpu.object. MLIR escapes
non-printable bytes as \\XX hex and the usual \\" \\\\ forms. This tool finds
the attribute and writes the unescaped text to the output file.

Usage: gdn_extract_ptx.py <in.mlir> <out.ptx>
"""

import sys


def unescape_mlir_string(s: str) -> str:
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt in "xX" and i + 3 < n:
                out.append(chr(int(s[i + 2:i + 4], 16)))
                i += 4
                continue
            if i + 2 < n and all(ch in "0123456789abcdefABCDEF" for ch in s[i + 1:i + 3]):
                # MLIR byte escapes: \XX (two hex digits)
                out.append(chr(int(s[i + 1:i + 3], 16)))
                i += 3
                continue
            mapping = {"n": "\n", "t": "\t", "r": "\r", '"': '"',
                       "\\": "\\", "'": "'"}
            if nxt in mapping:
                out.append(mapping[nxt])
                i += 2
                continue
            out.append(nxt)
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def extract(ptxlike: str) -> str:
    """Return the object whose text starts with a PTX header (.version)."""
    # find `assembly = "` occurrences
    best = None
    key = 'assembly = "'
    idx = 0
    while True:
        start = ptxlike.find(key, idx)
        if start < 0:
            break
        i = start + len(key)
        buf = []
        while i < len(ptxlike):
            c = ptxlike[i]
            if c == "\\":
                buf.append(ptxlike[i:i + 3])
                i += 3
                continue
            if c == '"':
                break
            buf.append(c)
            i += 1
        text = unescape_mlir_string("".join(buf))
        if ".version" in text and ".target" in text:
            best = text
            # prefer the longest (multi-object modules: take first with mma?)
            break
        idx = i + 1
    if best is None:
        raise SystemExit("error: no PTX assembly object found in input")
    return best


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    with open(sys.argv[1], "r", errors="replace") as f:
        content = f.read()
    ptx = extract(content)
    with open(sys.argv[2], "w") as f:
        f.write(ptx)
    print(f"extracted {len(ptx)} bytes of PTX -> {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
