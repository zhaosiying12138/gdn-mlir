#!/usr/bin/env python3
"""gdn_bwd_verify.py -- G5-C numeric verification of the multi-kernel GPU
backward (gdn_bwd_gen.py stages) against an fp64 numpy twin of the G1-B2 math
(gdn_l1_check.chunked_bwd_ref port, vectorized) and -- for the small shape --
the G1-A JAX vjp oracle (tools/gdn_ref_jax.gdn_chunk_vjp).

Chain: wy -> fwdscan -> caryA -> caryM -> bpre -> dhs -> dkb -> dqkwg -> wyb
(G5-J round5: the dhu serial walk is replaced by the [K,K]-carry restructure
-- caryA/caryM/bpre precompose the per-chunk affine scan operands
A_n = 2^gcl I - W^T Bhat, M_n = Qg^T - W^T P^T, B_n = M_n dO in PARALLEL;
dhs runs the reduced one-gemm-per-chunk carry chain; dv/dvp_all moved to
dqkwg, which already recomputed them per chunk).  Each stage is a zero-arg
cubin whose public memref.globals are staged via cuModuleGetGlobal (HtoD)
and read back (DtoH) after launch. Final grads are compared with the
fla-derived contract:
  dq/dk/dv <= 8e-3, dbeta/dg <= 2e-2, dh0 <= 8e-3  (vs fp64, bf16-fed kernel)

Inputs are bf16-quantized on the host (round-nearest) and the twin runs on
the SAME quantized values (widened to f64), so the diff measures kernel math
+ internal truncf, matching the fla grading methodology.

Usage (conda gdn):
  python tools/gdn_bwd_verify.py --gen-root evidence/G5/gen_bwd \
      --shape B,T,H,K,V [--chunk 64] [--jax] [--worker-dir DIR] [--stage DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

RCP_LN2 = 1.4426950408889634
STAGES = ["wy", "fwdscan", "caryA", "caryM", "bpre", "dhs", "dkb", "dqkwg",
          "wyb"]

# per-stage OUTPUT globals (the chaining contract; everything else is input)
OUT = {
    "wy": ["T_all", "u_all", "w_all", "kkt_all", "P_all"],
    # G5-T round14: gsc_all = the fwdscan prologue's per-chunk gc-cumsum
    # scratch (GDN_FSCAN_GSCG) -- written+read inside fwdscan only, listed
    # here so the staging loop treats it as owned (not host-staged); the
    # read loop skips it when the chain lacks it (knob-off arms).
    "fwdscan": ["h_all", "vnew_all", "gsc_all"],
    "caryA": ["A_all"],
    "caryM": ["M_all"],
    "bpre": ["B_all"],
    "dhs": ["dh_all", "dh0"],
    "dkb": ["dkB_sum", "rowB_c", "s5_c"],
    "dqkwg": ["dq", "dW_all", "dk_local", "dvp_all", "r3_c", "r4_c",
              "c4r_c", "dg_state"],
    "wyb": ["dk", "dbeta", "dg", "dv"],
}
INP = ["q", "k", "v", "beta", "g", "do", "h0", "dht"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def make_data(B, T, H, K, V, seed):
    """GDN-model inputs (gdn_l1_gen.make_data recipe) + seeded cotangents."""
    rng = np.random.default_rng(seed)

    def l2n(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    q = l2n(rng.standard_normal((B, T, H, K))).astype(np.float32)
    k = l2n(rng.standard_normal((B, T, H, K))).astype(np.float32)
    v = rng.standard_normal((B, T, H, V)).astype(np.float32)
    beta = (1 / (1 + np.exp(-rng.standard_normal((B, T, H))))).astype(np.float32)
    g = (-np.logaddexp(0.0, 0.5 * rng.standard_normal((B, T, H)))).astype(np.float32)
    h0 = (0.1 * rng.standard_normal((B, H, K, V))).astype(np.float32)
    do = rng.standard_normal((B, T, H, V)).astype(np.float32)
    dht = rng.standard_normal((B, H, K, V)).astype(np.float32)
    return dict(q=q, k=k, v=v, beta=beta, g=g, h0=h0, do=do, dht=dht)


def bf16_quant(x):
    """round-to-nearest-even bf16 via fp32 bit trick."""
    f = np.ascontiguousarray(x, dtype=np.float32)
    u = f.view(np.uint32)
    rounded = ((u >> 16) & 1) + 0x7FFF
    u = u + rounded
    return (u >> 16).astype(np.uint16)


def bf16_widen(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def layout(data, B, T, H, K, V, C):
    """host layout: chunk-major flattened [BHNC, C*W] (+ log2-domain g)."""
    NC = (T + C - 1) // C
    Tp = NC * C

    def pad_time(x):
        if x.shape[1] == Tp:
            return x
        pw = [(0, 0)] * x.ndim
        pw[1] = (0, Tp - T)
        return np.pad(x, pw)

    def ck(x, W, dt):
        x = pad_time(x)                       # [B, Tp, H, W] (or [..., H])
        if x.ndim == 4:
            x = np.ascontiguousarray(x.transpose(0, 2, 1, 3))  # [B, H, Tp, W]
            return x.reshape(B * H * NC, C * W).astype(dt)
        x = np.ascontiguousarray(x.transpose(0, 2, 1))          # [B, H, Tp]
        return x.reshape(B * H * NC, C).astype(dt)

    out = {
        "q": bf16_quant(ck(data["q"], K, np.float32)),
        "k": bf16_quant(ck(data["k"], K, np.float32)),
        "v": bf16_quant(ck(data["v"], V, np.float32)),
        "do": bf16_quant(ck(data["do"], V, np.float32)),
        "beta": ck(data["beta"], 1, np.float32).reshape(B * H * NC, C),
        "g_log2": (ck(data["g"], 1, np.float64) * RCP_LN2)
        .astype(np.float32).reshape(B * H * NC, C),
    }
    h0 = np.ascontiguousarray(data["h0"]).reshape(B * H, K * V)
    dht = np.ascontiguousarray(data["dht"]).reshape(B * H, K * V)
    out["h0"] = bf16_quant(h0)
    out["dht"] = bf16_quant(dht)
    out["_meta"] = {"B": B, "T": T, "H": H, "K": K, "V": V, "C": C,
                    "NC": NC, "BH": B * H, "Tp": Tp}
    return out


# ---------------------------------------------------------------------------
# fp64 twin (port of gdn_l1_check.chunked_bwd_ref; bf16-quantized inputs)
# ---------------------------------------------------------------------------

def bwd_twin(lay, with_intermediates=False):
    md = lay["_meta"]
    B, H, K, V, C = md["B"], md["H"], md["K"], md["V"], md["C"]
    NC, BH = md["NC"], md["BH"]

    def widen_chunks(u16, W):
        return bf16_widen(u16).reshape(BH, NC, C, W)

    q = widen_chunks(lay["q"], K)
    k = widen_chunks(lay["k"], K)
    v = widen_chunks(lay["v"], V)
    do = widen_chunks(lay["do"], V)
    beta = lay["beta"].astype(np.float64).reshape(BH, NC, C)
    g = (lay["g_log2"].astype(np.float64) / RCP_LN2).reshape(BH, NC, C)
    h0 = bf16_widen(lay["h0"]).reshape(BH, K, V)
    dht = bf16_widen(lay["dht"]).reshape(BH, K, V)
    scale = 1.0 / K ** 0.5

    dq = np.zeros((BH, NC, C, K))
    dk = np.zeros((BH, NC, C, K))
    dv = np.zeros((BH, NC, C, V))
    dbeta = np.zeros((BH, NC, C))
    dg = np.zeros((BH, NC, C))
    dh0 = np.zeros((BH, K, V))
    inter = {}
    m_inc = np.tril(np.ones((C, C)))
    m_str = np.tril(np.ones((C, C)), -1)

    for bh in range(BH):
        S = h0[bh].copy()
        h_all = np.zeros((NC, K, V))
        W_all = np.zeros((NC, C, K))
        U_all = np.zeros((NC, C, V))
        T_all = np.zeros((NC, C, C))
        vn_all = np.zeros((NC, C, V))
        for n in range(NC):
            gc = np.cumsum(g[bh, n])
            kkT = k[bh, n] @ k[bh, n].T
            A = np.tril(beta[bh, n][:, None] * kkT *
                        np.exp(gc[:, None] - gc[None, :]), -1)
            Tm = np.eye(C)
            for i in range(1, C):
                Tm[i, :i] = -(A[i, :i] @ Tm[:i, :i])
            U = Tm @ (beta[bh, n][:, None] * v[bh, n])
            W = Tm @ (beta[bh, n][:, None] * k[bh, n] * np.exp(gc)[:, None])
            vn = U - W @ S
            h_all[n] = S
            T_all[n], U_all[n], W_all[n], vn_all[n] = Tm, U, W, vn
            S = (S * np.exp(gc[-1])
                 + (k[bh, n] * np.exp(gc[-1] - gc)[:, None]).T @ vn)
        dH = dht[bh].copy()
        dh_all = np.zeros((NC, K, V))
        dvp_all = np.zeros((NC, C, V))
        dW_all = np.zeros((NC, C, K))
        for n in reversed(range(NC)):
            qc, kc, vc, bc = q[bh, n], k[bh, n], v[bh, n], beta[bh, n]
            gc = np.cumsum(g[bh, n])
            gcl = gc[-1]
            # G5-E round2 FIX: kkT must be THIS chunk's k k^T. It was only
            # computed in the forward loop above, so dbeta's ra term and
            # dg's AdA term silently used the STALE kkT of chunk NC-1
            # (~3.6e-2 / 2.4e-2 phantom errors vs the JAX oracle; the GPU
            # computes the per-chunk kkt correctly and was right).
            kkT = k[bh, n] @ k[bh, n].T
            Sn = h_all[n]
            D = np.exp(gc[:, None] - gc[None, :])
            Tm, U, W, vn = T_all[n], U_all[n], W_all[n], vn_all[n]
            P = np.tril((qc @ kc.T) * D, 0)
            Bhat = kc * np.exp(gcl - gc)[:, None]
            Qg = qc * np.exp(gc)[:, None]
            Kbg = bc[:, None] * kc * np.exp(gc)[:, None]
            Vb = bc[:, None] * vc
            dO = do[bh, n] * scale
            dPm = (dO @ vn.T) * D * m_inc
            dvp = P.T @ dO + Bhat @ dH
            dW = -(dvp @ Sn.T)
            G = dvp @ Vb.T + dW @ Kbg.T
            dAf = -(Tm.T @ (G * m_str) @ Tm.T) * D * m_str
            TW = Tm.T @ dW
            tv = Tm.T @ dvp
            dq1 = (dO * np.exp(gc)[:, None]) @ Sn.T
            dq[bh, n] = dq1 + dPm @ kc
            dk[bh, n] = ((dAf @ kc) * bc[:, None]
                         + dAf.T @ (kc * bc[:, None])
                         + TW * (bc * np.exp(gc))[:, None]
                         + (vn * np.exp(gcl - gc)[:, None]) @ dH.T
                         + dPm.T @ qc)
            dv[bh, n] = tv * bc[:, None]
            dbeta[bh, n] = ((dAf * kkT).sum(1)
                            + (TW * (kc * np.exp(gc)[:, None])).sum(1)
                            + (tv * vc).sum(1))
            AdA = dAf * kkT * bc[:, None]
            dgr = AdA.sum(1) - AdA.sum(0)
            dgr += (TW * Kbg).sum(1)
            dgr += (dq1 * qc).sum(1)
            PdP = (dO @ vn.T) * P
            dgr += PdP.sum(1) - PdP.sum(0)
            rowB = ((vn * np.exp(gcl - gc)[:, None]) @ dH.T * kc).sum(1)
            dg5a = np.exp(gcl) * (dH * Sn).sum()
            suf = np.cumsum(dgr[::-1])[::-1]
            pre = np.concatenate([[0.0], np.cumsum(rowB)[:-1]])
            dg[bh, n] = suf + pre + dg5a
            dH = np.exp(gcl) * dH + Qg.T @ dO - W.T @ dvp
            dh_all[n], dvp_all[n], dW_all[n] = dH.copy(), dvp, dW
        dh0[bh] = dH
        if with_intermediates and bh == 0:
            inter = {"h_all": h_all, "dh_all": dh_all, "T_all": T_all,
                     "vn_all": vn_all, "dvp_all": dvp_all, "dW_all": dW_all}
    res = {"dq": dq, "dk": dk, "dv": dv, "dbeta": dbeta, "dg": dg,
           "dh0": dh0, "_inter": inter}
    return res


# ---------------------------------------------------------------------------
# worker: run the whole chain on the GPU
# ---------------------------------------------------------------------------

def run_chain(gen_root, lay, timeout_sanity=120.0):
    import cuda.bindings.driver as cu

    man = json.load(open(os.path.join(gen_root, "bwd_manifest.json")))
    md = lay["_meta"]

    def ck(res, what):
        err = res[0] if isinstance(res, tuple) else res
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{what}: {err}")
        return res

    stream = cu.CUstream(0)
    ck(cu.cuInit(0), "init")
    e, dev = cu.cuDeviceGet(0)
    ck((e,), "dev")
    e, ctx = cu.cuDevicePrimaryCtxRetain(dev)
    ck((e,), "ctx")
    ck(cu.cuCtxSetCurrent(ctx), "setctx")

    host = {}       # name -> np array (raw bytes view) for staging/compare
    for nm in ("q", "k", "v", "do"):
        host[nm] = lay[nm]
    host["beta"] = lay["beta"].astype(np.float32)
    host["g"] = lay["g_log2"].astype(np.float32)
    host["h0"] = lay["h0"]
    host["dht"] = lay["dht"]

    times = {}
    for st in man["stages"]:
        name = st["name"]
        d = os.path.join(gen_root, st["dir"])
        raw = open(os.path.join(d, "kernel_smemfix.cubin"), "rb").read()
        e, mod = cu.cuModuleLoadData(raw)
        ck((e,), f"load {name}")
        ptx = open(os.path.join(d, "kernel_smemfix.ptx")).read()
        kname = re.search(r"\.visible \.entry (\w+)\(", ptx).group(1)
        e, fn = cu.cuModuleGetFunction(mod, kname.encode())
        ck((e,), f"fn {name}")
        gx, gy, _gz = st["grid"]
        bx = st["block"][0]
        # stage inputs
        for sym, ty in st["globals"].items():
            if sym in OUT[name]:
                continue
            if sym in st.get("zero_globals", []):
                continue    # r21b scratch (spin flags): memset pre-launch
            if sym in st.get("scratch_globals", []):
                continue    # r21b scratch (pex): write-before-read
            if sym not in host:
                raise RuntimeError(f"{name}: input global @{sym} not staged")
            e, ptr, sz = cu.cuModuleGetGlobal(mod, sym.encode())
            ck((e,), f"global {name}/{sym}")
            arr = host[sym]
            nbytes = arr.nbytes if arr.dtype != np.uint16 else arr.nbytes
            ck(cu.cuMemcpyHtoD(ptr, np.frombuffer(arr.tobytes(), dtype=np.uint8),
                               arr.nbytes), f"h2d {name}/{sym}")
        # r21b scratch contract: stream-ordered zero of the spin flags
        # (absolute per-launch targets) before the launch
        for sym in st.get("zero_globals", []):
            e, ptr, sz = cu.cuModuleGetGlobal(mod, sym.encode())
            ck((e,), f"zero global {name}/{sym}")
            ck(cu.cuMemsetD8Async(ptr, 0, sz, stream),
               f"zero {name}/{sym}")
        # launch + time
        ev0 = cu.cuEventCreate(0)[1]
        ev1 = cu.cuEventCreate(0)[1]
        cl = st.get("cluster")
        if cl:
            cfg = cu.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = gx, gy, 1
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = bx, 1, 1
            cfg.sharedMemBytes = 0
            cfg.hStream = stream
            cfg.numAttrs = 1
            a = cu.CUlaunchAttribute()
            a.id = (cu.CUlaunchAttributeID
                    .CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)
            a.value.clusterDim.x = cl[0]
            a.value.clusterDim.y = cl[1] if len(cl) > 1 else 1
            a.value.clusterDim.z = cl[2] if len(cl) > 2 else 1
            cfg.attrs = [a]
            ck(cu.cuLaunchKernelEx(cfg, fn, 0, 0), f"launchEx {name}")
        else:
            ck(cu.cuLaunchKernel(fn, gx, gy, 1, bx, 1, 1, 0, stream,
                                 None, 0), f"launch {name}")
        ck(cu.cuEventRecord(ev0, stream), "rec0")
        ck(cu.cuStreamSynchronize(stream), f"sync {name}")
        ck(cu.cuEventRecord(ev1, stream), "rec1")
        ck(cu.cuEventSynchronize(ev1), "evsync")
        err, ms = cu.cuEventElapsedTime(ev0, ev1)
        times[name] = round(ms, 4)
        # read outputs
        for sym in OUT[name]:
            if sym not in st["globals"]:
                continue    # r14: knob-off arms lack gsc_all
            ty = dict(st["globals"])[sym]
            rows, cols = (int(x) for x in
                          ty.split("<")[1].split("x")[0:2])
            dt = "f32" if ty.endswith("f32>") else "bf16"
            esz = 4 if dt == "f32" else 2
            e, ptr, _sz = cu.cuModuleGetGlobal(mod, sym.encode())
            ck((e,), f"global out {name}/{sym}")
            buf = np.zeros(rows * cols * esz, dtype=np.uint8)
            ck(cu.cuMemcpyDtoH(buf, ptr, rows * cols * esz),
               f"d2h {name}/{sym}")
            if dt == "bf16":
                host[sym] = buf.view(np.uint16).reshape(rows, cols)
            else:
                host[sym] = buf.view(np.float32).reshape(rows, cols)
    return host, times


def unchunk4(a, md, W):
    """[BH, NC, C, W] -> [B, T, H, W]"""
    B, H, NC, C, T = md["B"], md["H"], md["NC"], md["C"], md["T"]
    return (a.reshape(B, H, NC, C, W).transpose(0, 3, 1, 2, 4)
            .reshape(B, NC * C, H, W).transpose(0, 2, 1, 3)[:, :T, :, :])


def rechunk4(a, md, W):
    """[B, T, H, W] -> [BH, NC, C, W] (or [B,T,H] -> [BH, NC, C])"""
    B, H, NC, C, T = md["B"], md["H"], md["NC"], md["C"], md["T"]
    Tp = NC * C
    if a.ndim == 3:
        pw = [(0, 0)] * 3
        pw[1] = (0, Tp - T)
        a = np.pad(a, pw).transpose(0, 2, 1)
        return a.reshape(B * H, NC, C)
    pw = [(0, 0)] * 4
    pw[1] = (0, Tp - T)
    a = np.pad(a, pw).transpose(0, 2, 1, 3)
    return a.reshape(B, H, NC, C, W).reshape(B * H, NC, C, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", required=True)
    ap.add_argument("--shape", default="1,512,4,128,128")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jax", action="store_true",
                    help="cross-check the twin vs the JAX vjp oracle")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    B, T, H, K, V = (int(x) for x in args.shape.split(","))
    data = make_data(B, T, H, K, V, args.seed)
    lay = layout(data, B, T, H, K, V, args.chunk)
    md = lay["_meta"]

    t0 = time.time()
    ref = bwd_twin(lay, with_intermediates=True)
    print(f"[twin] fp64 backward done in {time.time() - t0:.1f}s")

    if args.jax:
        sys.path.insert(0, os.path.expanduser("~/codebase/gdn-mlir/tools"))
        import gdn_ref_jax as rj
        import jax.numpy as jnp
        jd = dict(q=jnp.asarray(data["q"], jnp.float64),
                  k=jnp.asarray(data["k"], jnp.float64),
                  v=jnp.asarray(data["v"], jnp.float64),
                  beta=jnp.asarray(data["beta"], jnp.float64),
                  g=jnp.asarray(data["g"], jnp.float64),
                  h0=jnp.asarray(data["h0"], jnp.float64))
        _o, _ht, jdq, jdk, jdv, jdb, jdg, jdh0 = rj.gdn_chunk_vjp(
            jd["q"], jd["k"], jd["v"], jd["beta"], jd["g"], h0=jd["h0"],
            do=jnp.asarray(data["do"], jnp.float64),
            dht=jnp.asarray(data["dht"], jnp.float64),
            chunk_size=args.chunk)
        print("[jax] twin-vs-vjp (fp64 vs UNquantized inputs; small residual"
              " from input quantization expected):")
        for nm, jg in (("dq", jdq), ("dk", jdk), ("dv", jdv), ("dbeta", jdb),
                       ("dg", jdg)):
            W = md["K"] if nm in ("dq", "dk") else md["V"]
            jj = rechunk4(np.asarray(jg), md, W)
            print(f"  {nm}: {np.abs(ref[nm] - jj).max():.3e}")
        jj0 = np.asarray(jdh0).transpose(0, 2, 1, 3).reshape(
            md["BH"], md["K"], md["V"])
        print(f"  dh0: {np.abs(ref['dh0'] - jj0).max():.3e}")

    host, times = run_chain(args.gen_root, lay)
    print("[chain] stage times (ms, single launch, incl. staging):", times)

    tol = {"dq": 8e-3, "dk": 8e-3, "dv": 8e-3, "dbeta": 2e-2, "dg": 2e-2,
           "dh0": 8e-3}
    report = {"shape": args.shape, "chunk": args.chunk, "seed": args.seed,
              "tolerances": tol, "stage_times_ms": times, "grads": {}}
    print(f"\n{'grad':6s} {'max_abs_diff':>14s} {'tol':>8s}  verdict")
    ok = True
    for nm in ("dq", "dk", "dv", "dbeta", "dg"):
        W = md["K"] if nm in ("dq", "dk") else md["V"]
        if nm in ("dbeta", "dg"):
            got = host[nm].reshape(md["BH"], md["NC"], md["C"]) \
                .astype(np.float64) \
                .reshape(md["B"], md["H"], md["NC"], md["C"]) \
                .transpose(0, 3, 1, 2).reshape(md["B"], md["NC"] * md["C"],
                                               md["H"]) \
                .transpose(0, 2, 1)[:, :md["T"], :]
            exp = ref[nm].reshape(md["B"], md["H"], md["NC"], md["C"]) \
                .transpose(0, 3, 1, 2).reshape(md["B"], md["NC"] * md["C"],
                                               md["H"]) \
                .transpose(0, 2, 1)[:, :md["T"], :]
        else:
            got = unchunk4(bf16_widen(host[nm]) if host[nm].dtype == np.uint16
                           else host[nm].reshape(md["BH"], md["NC"], md["C"], W)
                           .astype(np.float64), md, W)
            exp = unchunk4(ref[nm], md, W)
        d = np.abs(got - exp).max()
        verdict = "PASS" if d <= tol[nm] else "FAIL"
        ok = ok and (verdict == "PASS")
        print(f"{nm:6s} {d:14.3e} {tol[nm]:8.0e}  {verdict}")
        report["grads"][nm] = {"max_abs_diff": float(d), "tol": tol[nm],
                               "verdict": verdict}
    got0 = bf16_widen(host["dh0"]).reshape(md["B"], md["H"], md["K"], md["V"]) \
        if host["dh0"].dtype == np.uint16 else \
        host["dh0"].reshape(md["B"], md["H"], md["K"], md["V"]).astype(np.float64)
    # RC1: the fp64 twin emits dh0 flat as [BH, K, V]; B>1 cells need the
    # [B, H, K, V] split view (contiguous, value-identical reshape).
    d = np.abs(got0 - ref["dh0"].reshape(got0.shape)).max()
    verdict = "PASS" if d <= tol["dh0"] else "FAIL"
    ok = ok and (verdict == "PASS")
    print(f"{'dh0':6s} {d:14.3e} {tol['dh0']:8.0e}  {verdict}")
    report["grads"]["dh0"] = {"max_abs_diff": float(d), "tol": tol["dh0"],
                              "verdict": verdict}
    report["overall"] = "PASS" if ok else "FAIL"
    print(f"\nOVERALL: {report['overall']}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
