#!/usr/bin/env python3
"""Patch tools/gdn_full_gen.py for G5-D round 2 (pipelined staged path).

Run in repo root AFTER copying _round2_emit.py.inc / _round2_stages.py.inc
into tools/.  Idempotent-ish: refuses to run twice (checks for marker).
"""
import sys

PATH = "tools/gdn_full_gen.py"
src = open(PATH).read()

if "G5-D round 2" in src:
    print("already patched")
    sys.exit(0)

# ---------------- P1/P2: ldm_a/ldm_b accept SSA k-offsets ----------------
pairs = [
    ("""    if not swap:
        row = addi(e, addi(e, e.c(mi * 16), f"%lh8_{name}"),
                   f"%lq7_{name}")
        col = addi(e, e.c(kb), f"%lt8_{name}")""",
     """    kbc = kb if isinstance(kb, str) else e.c(kb)
    if not swap:
        row = addi(e, addi(e, e.c(mi * 16), f"%lh8_{name}"),
                   f"%lq7_{name}")
        col = addi(e, kbc, f"%lt8_{name}")"""),
    ("""    else:
        row = addi(e, addi(e, e.c(kb), f"%lt8_{name}"), f"%lq7_{name}")
        col = addi(e, e.c(mi * 16), f"%lh8_{name}")""",
     """    else:
        row = addi(e, addi(e, kbc, f"%lt8_{name}"), f"%lq7_{name}")
        col = addi(e, e.c(mi * 16), f"%lh8_{name}")"""),
    ("""    if kt:
        row = addi(e, f"%nc_{name}_{ni}", f"%lq7_{name}")
        col = addi(e, e.c(kb), f"%lb8_{name}")""",
     """    kbc = kb if isinstance(kb, str) else e.c(kb)
    if kt:
        row = addi(e, f"%nc_{name}_{ni}", f"%lq7_{name}")
        col = addi(e, kbc, f"%lb8_{name}")"""),
    ("""    else:
        row = addi(e, addi(e, e.c(kb), f"%lb8_{name}"), f"%lq7_{name}")""",
     """    else:
        row = addi(e, addi(e, kbc, f"%lb8_{name}"), f"%lq7_{name}")"""),
]
for old, new in pairs:
    assert src.count(old) == 1, "anchor not found: %r..." % old[:60]
    src = src.replace(old, new)

# ---------------- P4: PIPE_TRANSFORM before ASYNC_TRANSFORM ----------------
anchor = 'ASYNC_TRANSFORM = """module attributes {transform.with_named_sequence}'
assert src.count(anchor) == 1
pipe_tmpl = '''PIPE_TRANSFORM = """module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %arg0
      : (!transform.any_op) -> !transform.op<"memref.alloc">
    transform.memref.multibuffer %allocs <factor = {mb}, skip_analysis>
      : (!transform.op<"memref.alloc">) -> !transform.any_op
    %func = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.create_async_groups %func bypass_l1 : (!transform.any_op) -> (!transform.any_op)
    %loops = transform.structured.match ops{["scf.for"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.pipeline_shared_memory_copies failures(suppress) %loops depth = {depth} peel_epilogue : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
"""


def pipe_transform(mb=2, depth=2):
    return PIPE_TRANSFORM.replace("{mb}", str(mb)).replace("{depth}",
                                                           str(depth))


'''
src = src.replace(anchor, pipe_tmpl + anchor)

# ---------------- P3: replace the three stage generators ----------------
start = '# ' + '-' * 75 + '\n# F1 "wy"\n# ' + '-' * 75 + '\n'
end = '# ' + '-' * 75 + '\n# fused flagship\n# ' + '-' * 75 + '\n'
i = src.index(start)
j = src.index(end)
inc = open("tools/_round2_emit.py.inc").read() + "\n\n" + \
    open("tools/_round2_stages.py.inc").read()
src = src[:i] + inc + "\n" + src[j:]

# ---------------- P5: main() rewiring ----------------
old_main = '''    if args.path == "fused":
        e, params, grid, nthr = gen_fused(shp, W, BV)
        kname = "gdn_fwd_fused"
    else:
        assert args.kernel, "--kernel f1|f2|f3 required for --path staged"
        fn = {"f1": gen_f1, "f2": gen_f2, "f3": gen_f3}[args.kernel]
        if args.kernel == "f1":
            e, params, grid, nthr = fn(shp, W, BV, stop_after=args.stop_after)
        else:
            e, params, grid, nthr = fn(shp, W, BV)
        kname = f"gdn_fwd_{args.kernel}"

    meta = (f"{args.path} W={W} BV={BV} copies={args.copies} "
            f"B={B} T={T} H={H} K={K} V={V} NC={NC}")
    mod = wrap_module(kname, params, hoist_constants(e.text()),
                      grid[0], grid[1], nthr, meta)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "payload.mlir"), "w") as f:
        f.write(mod)
    with open(os.path.join(args.out, "transform.mlir"), "w") as f:
        f.write(ASYNC_TRANSFORM if args.copies == "async"
                else IDENTITY_TRANSFORM)
    spec = {
        "kernel": kname, "path": args.path,
        "shape": {"B": B, "T": T, "H": H, "K": K, "V": V, "C": C,
                  "NC": NC, "Tp": Tp},
        "knobs": {"warps": W, "bv": BV, "copies": args.copies},
        "params": [{"name": n, "dtype": dt, "elems": int(el)}
                   for n, el, dt in params],
        "grid": [grid[0], grid[1], 1], "block": [nthr, 1, 1],
    }'''
new_main = '''    meta = (f"{args.path} W={W} BV={BV} bk={args.bk} mb={args.mb} "
            f"depth={args.depth} copies={args.copies} "
            f"B={B} T={T} H={H} K={K} V={V} NC={NC}")
    if args.path == "fused":
        e, params, grid, nthr = gen_fused(shp, W, BV)
        kname = "gdn_fwd_fused"
        mod = wrap_module(kname, params, hoist_constants(e.text()),
                          grid[0], grid[1], nthr, meta)
        transform_txt = (ASYNC_TRANSFORM if args.copies == "async"
                         else IDENTITY_TRANSFORM)
        spec = {
            "kernel": kname, "path": args.path,
            "shape": {"B": B, "T": T, "H": H, "K": K, "V": V, "C": C,
                      "NC": NC, "Tp": Tp},
            "knobs": {"warps": W, "bv": BV, "copies": args.copies},
            "params": [{"name": n, "dtype": dt, "elems": int(el)}
                       for n, el, dt in params],
            "grid": [grid[0], grid[1], 1], "block": [nthr, 1, 1],
        }
    elif args.path == "mono":
        # G5-D round 2: the staged trio in ONE cubin (shared globals, no
        # DtoD handoffs), every K-loop through the transform pipeline.
        kernels, params = gen_mono(shp, W, BV, bk=args.bk)
        kname = "gdn_fwd_mono"
        mod = wrap_module_mono(kernels, params, meta)
        transform_txt = pipe_transform(args.mb, args.depth)
        spec = {
            "kernel": kname, "path": args.path,
            "shape": {"B": B, "T": T, "H": H, "K": K, "V": V, "C": C,
                      "NC": NC, "Tp": Tp},
            "knobs": {"warps": W, "bv": BV, "bk": args.bk, "mb": args.mb,
                      "depth": args.depth},
            "params": [{"name": n, "dtype": dt, "elems": int(el)}
                       for n, el, dt in params],
            "kernels": [{"name": kn, "grid": [g[0], g[1], 1],
                         "block": [nt, 1, 1]}
                        for kn, _e, g, nt in kernels],
        }
    else:
        assert args.kernel, "--kernel f1|f2|f3 required for --path staged"
        fn = {"f1": gen_f1, "f2": gen_f2, "f3": gen_f3}[args.kernel]
        e, params, grid, nthr = fn(shp, W, BV, bk=args.bk)
        kname = f"gdn_fwd_{args.kernel}"
        mod = wrap_module(kname, params, hoist_constants(e.text()),
                          grid[0], grid[1], nthr, meta)
        transform_txt = pipe_transform(args.mb, args.depth)
        spec = {
            "kernel": kname, "path": args.path,
            "shape": {"B": B, "T": T, "H": H, "K": K, "V": V, "C": C,
                      "NC": NC, "Tp": Tp},
            "knobs": {"warps": W, "bv": BV, "bk": args.bk, "mb": args.mb,
                      "depth": args.depth},
            "params": [{"name": n, "dtype": dt, "elems": int(el)}
                       for n, el, dt in params],
            "grid": [grid[0], grid[1], 1], "block": [nthr, 1, 1],
        }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "payload.mlir"), "w") as f:
        f.write(mod)
    with open(os.path.join(args.out, "transform.mlir"), "w") as f:
        f.write(transform_txt)'''
assert src.count(old_main) == 1, "main() anchor"
src = src.replace(old_main, new_main)

# ---------------- P6: CLI knobs + mono choice ----------------
old_cli = '''    ap.add_argument("--path", choices=("staged", "fused"), required=True)
    ap.add_argument("--kernel", choices=("f1", "f2", "f3"))'''
new_cli = '''    ap.add_argument("--path", choices=("staged", "fused", "mono"),
                    required=True)
    ap.add_argument("--kernel", choices=("f1", "f2", "f3"))
    ap.add_argument("--bk", type=int, default=32,
                    help="K-loop step (pipelined tiles are [.,bk])")
    ap.add_argument("--mb", type=int, default=2,
                    help="multibuffer factor for pipeline tiles")
    ap.add_argument("--depth", type=int, default=2,
                    help="pipeline_shared_memory_copies depth")'''
assert src.count(old_cli) == 1, "cli anchor"
src = src.replace(old_cli, new_cli)

# validate(): mono path has no --kernel; guard the assert
old_val = '''    else:
        assert args.kernel, "--kernel f1|f2|f3 required for --path staged"'''
assert src.count(old_val) == 1

open(PATH, "w").write(src)
print("[patch_round2] gdn_full_gen.py patched OK")
