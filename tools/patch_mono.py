#!/usr/bin/env python3
"""Patch verify/bench/gen for the mono path (G5-D round 2)."""
import re

# ---- gen: --bv3 knob + mono per-kernel BV ----
P = "tools/gdn_full_gen.py"
src = open(P).read()
if "--bv3" not in src:
    old = '''    ap.add_argument("--depth", type=int, default=2,
                    help="pipeline_shared_memory_copies depth")'''
    new = old + '''
    ap.add_argument("--bv3", type=int, default=None,
                    help="BV for the f3 kernel in mono (default: --bv)")'''
    assert src.count(old) == 1
    src = src.replace(old, new)

    old = "        kernels, params = gen_mono(shp, W, BV, bk=args.bk)"
    new = "        kernels, params = gen_mono(shp, W, BV, args.bv3 or BV,\n                                bk=args.bk)"
    assert src.count(old) == 1
    src = src.replace(old, new)

    old = '''def gen_mono(shp, W, BV, bk=32):
    B, T, H, K, V, C, NC, Tp = shp
    VT = V // BV
    parts = []
    params = []

    def add_params(ps):
        for p in ps:
            if p not in params:
                params.append(p)

    for name, fn, grid in (("f1", gen_f1, (NC, B * H)),
                           ("f2", gen_f2, (B * H * VT, 1)),
                           ("f3", gen_f3, (NC * B * H, VT))):
        e, ps, _g, nthr = fn(shp, W, BV, bk=bk)
        add_params(ps)
        parts.append((f"gdn_fwd_{name}", e, grid, nthr))
    return parts, params'''
    new = '''def gen_mono(shp, W, BV2, BV3, bk=32):
    B, T, H, K, V, C, NC, Tp = shp
    VT2 = V // BV2
    VT3 = V // BV3
    parts = []
    params = []

    def add_params(ps):
        for p in ps:
            if p not in params:
                params.append(p)

    for name, fn, grid, bvk in (("f1", gen_f1, (NC, B * H), None),
                                ("f2", gen_f2, (B * H * VT2, 1), BV2),
                                ("f3", gen_f3, (NC * B * H, VT3), BV3)):
        e, ps, _g, nthr = fn(shp, W, bvk, bk=bk)
        add_params(ps)
        parts.append((f"gdn_fwd_{name}", e, grid, nthr))
    return parts, params'''
    assert src.count(old) == 1
    src = src.replace(old, new)
    open(P, "w").write(src)
    print("[patch_mono] gen ok")
else:
    print("[patch_mono] gen already")

# ---- verify: mono staged path ----
P = "tools/gdn_full_verify.py"
src = open(P).read()
if "mono" not in src:
    old = '''    # ---- staged path ----
    try:
        kf = {}
        for kk in ("f1", "f2", "f3"):
            d = rt.build(["--path", "staged", "--kernel", kk,
                          "--shape", f"{B},{T},{H},{K},{V}",
                          "--warps", str(warps), "--bv", str(bv)],
                         os.path.join(gen_root, f"{shape_tag}_{kk}"))
            kf[kk] = rt.FullKernel(d)
        for kk in ("f1", "f2", "f3"):
            rt.stage_inputs(kf[kk], inp, shape)
        kf["f1"].launch(stream)
        rt._ck(cu.cuStreamSynchronize(stream), "sync f1")
        rt.staged_handoff(kf["f1"], kf["f2"],
                          (("u", "u"), ("w", "w"), ("gc", "gc_out")))
        kf["f2"].launch(stream)
        rt._ck(cu.cuStreamSynchronize(stream), "sync f2")
        rt.staged_handoff(kf["f2"], kf["f3"],
                          (("h_all", "h_all"), ("v_new", "v_new"),
                           ("gc", "gc")))
        kf["f3"].launch(stream)
        rt._ck(cu.cuStreamSynchronize(stream), "sync f3")
        o = kf["f3"].readout("o", "bf16", B * Tp * H * V)
        ht = kf["f2"].readout("ht", "f32", B * H * K * V)
        compare("staged", o, ht)
        results["staged"]["intermediates"] = {
            "u_nonzero": int((kf["f1"].readout("u", "bf16",
                                               B * H * NC * C * V) != 0).sum()),
            "w_nonzero": int((kf["f1"].readout("w", "bf16",
                                               B * H * NC * C * K) != 0).sum()),
            "h_all_nonzero": int((kf["f2"].readout(
                "h_all", "bf16", B * H * NC * K * V) != 0).sum()),
        }
    except Exception as e:  # noqa: BLE001
        results["staged"] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}'''
    new = '''    # ---- staged path (G5-D round 2: mono cubin, shared globals, no DtoD) ----
    try:
        d = rt.build(["--path", "mono", "--shape", f"{B},{T},{H},{K},{V}",
                      "--warps", str(warps), "--bv", str(bv),
                      "--bv3", str(V)],
                     os.path.join(gen_root, f"{shape_tag}_mono"))
        km = rt.FullKernel(d)
        rt.stage_inputs(km, inp, shape)
        km.launch(stream, "gdn_fwd_f1")
        rt._ck(cu.cuStreamSynchronize(stream), "sync f1")
        km.launch(stream, "gdn_fwd_f2")
        rt._ck(cu.cuStreamSynchronize(stream), "sync f2")
        km.launch(stream, "gdn_fwd_f3")
        rt._ck(cu.cuStreamSynchronize(stream), "sync f3")
        o = km.readout("o", "bf16", B * Tp * H * V)
        ht = km.readout("ht", "f32", B * H * K * V)
        compare("staged", o, ht)
        results["staged"]["intermediates"] = {
            "u_nonzero": int((km.readout("u", "bf16",
                                         B * H * NC * C * V) != 0).sum()),
            "w_nonzero": int((km.readout("w", "bf16",
                                         B * H * NC * C * K) != 0).sum()),
            "h_all_nonzero": int((km.readout(
                "h_all", "bf16", B * H * NC * K * V) != 0).sum()),
        }
    except Exception as e:  # noqa: BLE001
        results["staged"] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}'''
    assert src.count(old) == 1
    src = src.replace(old, new)
    open(P, "w").write(src)
    print("[patch_mono] verify ok")
else:
    print("[patch_mono] verify already")

# ---- bench: mono staged path ----
P = "tools/gdn_full_bench.py"
src = open(P).read()
if "mono" not in src:
    old = '''    # ---- staged path ----
    kf = {}
    for kk in ("f1", "f2", "f3"):
        d = rt.build(["--path", "staged", "--kernel", kk,
                      "--shape", f"{B},{T},{H},{K},{V}",
                      "--warps", str(warps), "--bv", str(bv)],
                     os.path.join(gen_root, f"{name}_{kk}"))
        kf[kk] = rt.FullKernel(d)
        rt.stage_inputs(kf[kk], inp, shape)

    def staged_once():
        kf["f1"].launch(stream)
        rt._ck(cu.cuStreamSynchronize(stream), "s1")
        rt.staged_handoff(kf["f1"], kf["f2"],
                          (("u", "u"), ("w", "w"), ("gc", "gc_out")))
        kf["f2"].launch(stream)
        rt._ck(cu.cuStreamSynchronize(stream), "s2")
        rt.staged_handoff(kf["f2"], kf["f3"],
                          (("h_all", "h_all"), ("v_new", "v_new"),
                           ("gc", "gc")))
        kf["f3"].launch(stream)
        rt._ck(cu.cuStreamSynchronize(stream), "s3")'''
    new = '''    # ---- staged path (G5-D round 2: mono cubin, shared globals, no DtoD) ----
    km = rt.FullKernel(rt.build(
        ["--path", "mono", "--shape", f"{B},{T},{H},{K},{V}",
         "--warps", str(warps), "--bv", str(bv), "--bv3", str(V)],
        os.path.join(gen_root, f"{name}_mono")))
    rt.stage_inputs(km, inp, shape)

    def staged_once():
        km.launch(stream, "gdn_fwd_f1")
        km.launch(stream, "gdn_fwd_f2")
        km.launch(stream, "gdn_fwd_f3")
        rt._ck(cu.cuStreamSynchronize(stream), "sm")'''
    assert src.count(old) == 1
    src = src.replace(old, new)

    # per-stage timing in the bench worker
    old = '''    out["staged"] = timeit(staged_once)
    out["fused"] = timeit(fused_once)
    return out'''
    new = '''    out["staged"] = timeit(staged_once)
    for kk in ("f1", "f2", "f3"):
        out[kk] = timeit(lambda kk=kk: (km.launch(stream, f"gdn_fwd_{kk}"),
                                        rt._ck(cu.cuStreamSynchronize(stream),
                                               "sk")))
    out["fused"] = timeit(fused_once)
    return out'''
    assert src.count(old) == 1
    src = src.replace(old, new)
    open(P, "w").write(src)
    print("[patch_mono] bench ok")
else:
    print("[patch_mono] bench already")
