#!/usr/bin/env python3
"""g5i_t8192_check.py -- G5-I T8192 forward spot-check (GPU).

Three checks, same shared-recipe inputs (evidence/G5/refs/
bench_B1T8192H16_inputs.npz from tools/g5i_t8192_ref.py):
  1. vs the JAX fp64 ref (ref__o_f32/ref__ht_f32), bar 5e-3, when present;
  2. S-invariance: o/ht max-abs-diff between f2-segments 1 / 4 / 8;
  3. vs fla chunk fwd (bf16-scale agreement, bar 1e-2), when --fla given.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.expanduser("~/codebase/gdn-mlir")
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)

import numpy as np  # noqa: E402
import gdn_full_rt as rt  # noqa: E402

TAG = "bench_B1T8192H16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segs", default="1,4,8")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--bv", type=int, default=32)
    ap.add_argument("--fla", action="store_true")
    args = ap.parse_args()

    import cuda.bindings.driver as cu
    inp = dict(np.load(os.path.join(REPO, f"evidence/G5/refs/{TAG}_inputs.npz")))
    B, T, H, K = inp["q_bf16"].shape
    V = inp["v_bf16"].shape[-1]
    C = 64
    Tp = ((T + C - 1) // C) * C
    NC = Tp // C
    shape = (B, T, H, K, V, C, NC, Tp)

    rt._ck(cu.cuInit(0), "cuInit")
    e, dev = cu.cuDeviceGet(0); rt._ck((e,), "dev")
    e, ctx = cu.cuDevicePrimaryCtxRetain(dev); rt._ck((e,), "ctx")
    rt._ck(cu.cuCtxSetCurrent(ctx), "ctx")
    stream = cu.CUstream(0)

    ref_path = os.path.join(REPO, f"evidence/G5/refs/{TAG}_ref.npz")
    ref = dict(np.load(ref_path)) if os.path.exists(ref_path) else None

    outs = {}
    for s in [int(x) for x in args.segs.split(",")]:
        sfx = "" if s == 1 else f"_s{s}"
        d = os.path.join(REPO, "evidence/G5/gen_fwd", f"B1T8192_mono{sfx}")
        gen = ["--path", "mono", "--shape", f"{B},{T},{H},{K},{V}",
               "--warps", str(args.warps), "--bv", str(args.bv),
               "--bv3", str(V)]
        if s > 1:
            gen += ["--f2-segments", str(s)]
        km = rt.FullKernel(rt.build(gen, d))
        rt.stage_inputs(km, inp, shape)
        for kn in [k["name"] for k in km.spec.get("kernels", [])]:
            km.launch(stream, kn)
        rt._ck(cu.cuStreamSynchronize(stream), "sync")
        o = km.readout("o", "bf16", B * Tp * H * V).reshape(B, Tp, H, V)[:, :T]
        ht = km.readout("ht", "f32", B * H * K * V).reshape(B, H, K, V)
        outs[s] = (o, ht)
        line = f"S={s}: "
        if ref is not None:
            ro, rh = ref["ref__o_f64"], ref["ref__ht_f64"]
            do = float(np.abs(o.astype(np.float64) - ro).max())
            dh = float(np.abs(ht.astype(np.float64) - rh).max())
            line += f"o_vs_ref {do:.3e} ht_vs_ref {dh:.3e}  "
        print(line + f"|o|max {np.abs(o).max():.2f}", flush=True)

    segs = sorted(outs)
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a, b = outs[segs[i]], outs[segs[j]]
            print(f"S={segs[i]} vs S={segs[j]}: o "
                  f"{float(np.abs(a[0]-b[0]).max()):.3e}  ht "
                  f"{float(np.abs(a[1]-b[1]).max()):.3e}")

    if args.fla:
        os.environ.setdefault("FLA_FLASH_QLA", "0")
        import torch
        import gdn_bench_common as bc
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        case = bc.gen_case(B, T, H, H, K, V, seed=bc.SEED)
        io = bc.to_device_bench(case, device="cuda", dtype=bc.DTYPE_BENCH)
        with torch.no_grad():
            o_fla, ht_fla = chunk_gated_delta_rule(
                q=io["q"], k=io["k"], v=io["v"], g=io["g"], beta=io["beta"],
                scale=K ** -0.5, initial_state=io["h0"],
                output_final_state=True)
        o_f = o_fla.detach().float().cpu().numpy()
        ht_f = ht_fla.detach().float().cpu().numpy()
        for s in segs:
            o, ht = outs[s]
            print(f"S={s} vs fla: o {float(np.abs(o - o_f).max()):.3e}  "
                  f"ht {float(np.abs(ht - ht_f).max()):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
