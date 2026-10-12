#!/usr/bin/env python3
"""gdn_l1_gen.py -- L1 (algorithm-structure layer) chunked GDN forward generator.

Emits a self-contained MLIR module (linalg + scf, memref-level) implementing the
CHUNKED gated-delta-rule forward pass, ready for CPU lowering with mlir-opt and
execution with mlir-runner (see scripts/run_l1.sh).  This generator IS the
output format of the future L1 expansion pass: it is fully parameterized by
shape/chunk and embeds all input data as memref.global constants.

Math conventions (ported from ~/codebase/fla/fla/ops/gated_delta_rule/naive.py,
`naive_chunk_gated_delta_rule`; cross-checked against G1-A's
evidence/G1/jax_spec_report.md):

  shapes : q,k [B,T,H,K] | v [B,T,H,V] | beta,g [B,T,H] | h0,ht [B,H,K,V]
           o [B,T,H,V]
  g      : per-step decay in NATURAL-log domain in the fla/G1-A contract
           (multiplier exp(g)).  The KERNEL works in the log2 domain like the
           fla Triton path: this generator embeds g_log2 = g * log2(e) =
           g * 1.4426950408889634 (RCP_LN2) and the MLIR computes pure
           math.exp2, i.e. exp2(g*RCP_LN2) == exp(g).
  gc     : INCLUSIVE within-chunk cumsum of g (chunk-local only)
  scale  : K**-0.5, applied to q; o is linear in q so it is folded into the
           final output scaling: o = (o_inter + o_intra) * scale
  A      : A[i,j] = beta_i * (k_i . k_j) * 2**(gc_i - gc_j)  for j < i
           (strictly lower triangular, 0 elsewhere);  A == -M with M the
           strictly-lower matrix of naive.py / the G1-A report
  T      : (I + A)^-1 == (I - M)^-1 by explicit FORWARD SUBSTITUTION:
           T[i,i] = 1 ;  T[i,j] = -sum_{j<=k<i} A[i,k] * T[k,j]
           (bit-identical row recurrence to naive.py's `attn` loop)
  u      : T @ (beta * v)                       [fla: k_cumsum  = attn @ v]
  w      : T @ (beta * k * 2**gc)               [fla: k_cumdecay]
  v_new  : u - w @ S
  o_intra: tril_inc((q @ k^T) * 2**(gc_i - gc_j)) @ v_new   (diagonal INCLUDED)
  o_inter: (q * 2**gc) @ S
  o      : (o_inter + o_intra) * scale
  S'     : S * 2**gc_last + (k * 2**(gc_last - gc))^T @ v_new

Tail chunk handling follows naive.py: T is zero-padded along time to a multiple
of C (beta=k=v=g=0 pad rows contribute nothing to A/u/S; padded o rows are
discarded by the consumer).

Usage:
  python3 tools/gdn_l1_gen.py [--shape B,T,H,K,V] [--chunk C] [--seed N]
                              [--emit fwd|bwd|both] [--npz ref.npz]
                              [--out out.mlir]

  --npz loads keys q,k,v,beta,g (NATURAL-log g, fla/G1-A convention), h0
  ([B,H,K,V]; zeros if absent), and optionally expected outputs o,ht which are
  embedded as @ref_o / @ref_ht globals and printed by @main.  With
  --emit bwd|both it additionally loads the cotangents do,dht and the
  reference grads dq,dk,dv,dbeta,dg,dh0 (@ref_* globals, printed by @main).
"""

import argparse
import sys

import numpy as np

RCP_LN2 = 1.4426950408889634  # log2(e): natural-log g -> log2-domain g


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def make_data(B, T, H, K, V, seed):
    """Deterministic seeded fp32 test data, GDN-model-style (mirrors G1-A
    tools/gdn_ref_jax.py::make_gdn_inputs): q,k l2-normalized; beta=sigmoid(x);
    g = -softplus(0.5x) NEGATIVE NATURAL-log decay; h0 [B,H,K,V]."""
    rng = np.random.default_rng(seed)

    def l2n(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    q = l2n(rng.standard_normal((B, T, H, K))).astype(np.float32)
    k = l2n(rng.standard_normal((B, T, H, K))).astype(np.float32)
    v = rng.standard_normal((B, T, H, V)).astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-rng.standard_normal((B, T, H))))).astype(np.float32)
    xg = 0.5 * rng.standard_normal((B, T, H))
    g = (-np.logaddexp(0.0, xg)).astype(np.float32)  # natural-log domain
    h0 = (0.1 * rng.standard_normal((B, H, K, V))).astype(np.float32)
    return {"q": q, "k": k, "v": v, "beta": beta, "g": g, "h0": h0}


def load_npz(path, bwd=False):
    """G1-A reference dump (natural-log g, h0/ht [B,H,K,V]). Tolerates a
    missing h0 (zeros) and missing expected outputs.

    bwd=False: returns inputs {q,k,v,beta,g,h0} + refs {o,ht} (fwd contract).
    bwd=True:  additionally loads the VJP cotangents do,dht into the data dict
               and the reference grads dq,dk,dv,dbeta,dg,dh0 into refs."""
    d = np.load(path)
    B, T, H, K = d["q"].shape
    V = d["v"].shape[-1]
    out = {n: d[n].astype(np.float32) for n in ("q", "k", "v", "beta", "g")}
    if "h0" in d and d["h0"] is not None:
        out["h0"] = d["h0"].astype(np.float32)
    else:
        out["h0"] = np.zeros((B, H, K, V), np.float32)
    if bwd:
        out["do"] = d["do"].astype(np.float32)
        out["dht"] = d["dht"].astype(np.float32)
        refs = {n: d[n].astype(np.float32)
                for n in ("dq", "dk", "dv", "dbeta", "dg", "dh0") if n in d}
    else:
        refs = {n: d[n].astype(np.float32) for n in ("o", "ht") if n in d}
    return out, refs


def make_bwd_data(data, seed):
    """Add seeded VJP cotangents do [B,T,H,V], dht [B,H,K,V] to a data dict
    (for seeded --emit bwd/both cases; leaves the fwd inputs untouched)."""
    rng = np.random.default_rng(seed + 977)
    B, T, H, K = data["q"].shape
    V = data["v"].shape[-1]
    data = dict(data)
    data["do"] = rng.standard_normal((B, T, H, V)).astype(np.float32)
    data["dht"] = rng.standard_normal((B, H, K, V)).astype(np.float32)
    return data


def pad_time(x, Tp):
    """Zero-pad axis 1 (time) up to Tp."""
    pad = Tp - x.shape[1]
    if pad == 0:
        return x
    pw = [(0, 0)] * x.ndim
    pw[1] = (0, pad)
    return np.pad(x, pw)


# ----------------------------------------------------------------------------
# MLIR text helpers
# ----------------------------------------------------------------------------

def fmt_f32(x):
    """MLIR float literal (this build requires an explicit '.' or exponent)."""
    s = "%.9g" % float(x)
    if "e" in s or "E" in s:
        m, _, e = s.partition("e")
        if "." not in m:
            m += ".0"
        return m + "e" + e
    if "." not in s:
        s += ".0"
    return s


def dense_literal(a, row_indent):
    """Nested dense<...> literal with one innermost row per line."""
    a = np.asarray(a)
    if a.ndim == 0:
        return fmt_f32(a)
    if a.ndim == 1:
        return "[" + ", ".join(fmt_f32(x) for x in a) + "]"
    ind = row_indent + "  "
    inner = (",\n" + ind).join(dense_literal(r, ind) for r in a)
    return "[\n" + ind + inner + "\n" + row_indent + "]"


class E:
    """Tiny line emitter with fresh SSA names and indentation."""

    def __init__(self):
        self.lines = []
        self.depth = 0
        self.n = 0

    def __call__(self, s):
        self.lines.append("  " * self.depth + s)

    def v(self, hint=""):
        self.n += 1
        return "%" + hint + str(self.n)

    def block(self, header):
        self(header + " {")
        self.depth += 1

    def end(self):
        self.depth -= 1
        self("}")

    def text(self):
        return "\n".join(self.lines) + "\n"


def mrt(shape, elem="f32", strided=None, dyn_off=False):
    """memref type string."""
    dims = "x".join(str(int(d)) for d in shape)
    if strided is not None:
        st = ", ".join(str(int(s)) for s in strided)
        off = "?" if dyn_off else "0"
        return f"memref<{dims}x{elem}, strided<[{st}], offset: {off}>>"
    return f"memref<{dims}x{elem}>"


# ----------------------------------------------------------------------------
# module emission
# ----------------------------------------------------------------------------

def emit_module(data, B, T, H, K, V, C, refs=None, meta="", emit="fwd"):
    """Emit the full .mlir module text. `data` arrays are UNPADDED (real T).

    emit="fwd"  : historical forward-only module (byte-identical output).
    emit="bwd"  : @gdn_bwd module printing the six gradients.
    emit="both" : @gdn_fwd + @gdn_bwd; @main prints o, ht then the gradients.
    For emit in {"bwd","both"} the data dict must contain the cotangents
    do [B,T,H,V] and dht [B,H,K,V]; refs may contain the expected grads
    dq,dk,dv,dbeta,dg,dh0 (embedded as @ref_* globals and printed)."""
    assert emit in ("fwd", "bwd", "both")
    assert C >= 2
    Tp = ((T + C - 1) // C) * C
    NC = Tp // C
    scale = 1.0 / np.sqrt(K)

    q, k = pad_time(data["q"], Tp), pad_time(data["k"], Tp)
    v = pad_time(data["v"], Tp)
    beta = pad_time(data["beta"], Tp)
    g = pad_time(data["g"], Tp)
    h0 = data["h0"]
    assert h0.shape == (B, H, K, V), f"h0 must be BxHxKxV, got {h0.shape}"
    # embed g in the kernel-internal log2 domain: exp2(g*RCP_LN2) == exp(g)
    g_log2 = (g.astype(np.float64) * RCP_LN2).astype(np.float32)
    do_bwd = emit in ("bwd", "both")
    if do_bwd:
        do = pad_time(data["do"], Tp)
        dht = data["dht"]
        assert dht.shape == (B, H, K, V), f"dht must be BxHxKxV, got {dht.shape}"

    e = E()
    e("// Generated by tools/gdn_l1_gen.py -- DO NOT EDIT")
    kind = {"fwd": "forward", "bwd": "backward", "both": "forward+backward"}[emit]
    e(f"// Chunked GDN {kind}, L1 (scf+linalg) "
      f"| B={B} T={T} T_pad={Tp} H={H} K={K} V={V} chunk={C} NC={NC} {meta}")
    e("// conventions: embedded g is NATURAL-log g scaled by RCP_LN2 (log2")
    e("// domain, like fla's Triton path); gc inclusive chunk cumsum; A strictly")
    e("// lower; T=(I+A)^-1 forward substitution; scale=K^-1/2 folded into o.")
    if do_bwd:
        e("// bwd: reverse chunk walk with state recompute; dA adjoint")
        e("//   dA = -strict_tril(T^T strict_tril(G) T^T), G = dvp (beta v)^T")
        e("//        + dW (beta k 2^gc)^T, dW = -(dvp S^T).")
    e("module {")
    e.depth += 1

    # ---- runtime helper declaration -------------------------------------
    e("func.func private @printMemrefF32(memref<*xf32>)")
    e("")

    # ---- input globals ---------------------------------------------------
    e('// ---- input data (zero time-padding to T_pad = %d) ----' % Tp)
    ins = [("q", q, (B, Tp, H, K)), ("k", k, (B, Tp, H, K)),
           ("v", v, (B, Tp, H, V)), ("beta", beta, (B, Tp, H)),
           ("g", g_log2, (B, Tp, H)), ("h0", h0, (B, H, K, V))]
    if do_bwd:
        ins += [("do", do, (B, Tp, H, V)), ("dht", dht, (B, H, K, V))]
    for name, arr, shape in ins:
        ty = mrt(shape)
        e(f'memref.global "private" constant @{name} : {ty} = dense<')
        lit = dense_literal(arr, "  " * e.depth)
        e.lines[-1] += lit + ">"
    e("")

    # ---- optional expected-output globals --------------------------------
    ref_fwd_shapes = {"o": (B, T, H, V), "ht": (B, H, K, V)}
    ref_bwd_shapes = {"dq": (B, T, H, K), "dk": (B, T, H, K), "dv": (B, T, H, V),
                      "dbeta": (B, T, H), "dg": (B, T, H), "dh0": (B, H, K, V)}
    if refs:
        e("// ---- expected outputs (from --npz), printed after kernel outputs ----")
        ref_keys = ([kx for kx in ("o", "ht") if kx in refs] if emit != "bwd" else []) \
            + [kx for kx in ("dq", "dk", "dv", "dbeta", "dg", "dh0") if kx in refs]
        for key in ref_keys:
            name = "ref_" + key
            shape = ref_fwd_shapes.get(key) or ref_bwd_shapes[key]
            e(f'memref.global "private" constant @{name} : {mrt(shape)} = dense<'
              + dense_literal(refs[key], "  " * e.depth) + ">")
        e("")

    # ---- kernels -----------------------------------------------------------
    tq4, tk4, tv4 = mrt((B, Tp, H, K)), mrt((B, Tp, H, K)), mrt((B, Tp, H, V))
    tb3, tg3 = mrt((B, Tp, H)), mrt((B, Tp, H))
    th0 = mrt((B, H, K, V))
    if emit in ("fwd", "both"):
        emit_kernel(e, B, Tp, H, K, V, C, NC, scale, tq4, tk4, tv4, tb3, tg3, th0)
    if do_bwd:
        e("")
        emit_bwd_kernel(e, B, Tp, H, K, V, C, NC, scale,
                        tq4, tk4, tv4, tb3, tg3, th0)

    # ---- main -------------------------------------------------------------
    e("")
    e("func.func @main() {")
    e.depth += 1
    e("%q = memref.get_global @q : " + tq4)
    e("%k = memref.get_global @k : " + tk4)
    e("%v = memref.get_global @v : " + tv4)
    e("%beta = memref.get_global @beta : " + tb3)
    e("%g = memref.get_global @g : " + tg3)
    e("%h0 = memref.get_global @h0 : " + th0)
    if emit in ("fwd", "both"):
        e(f"%o, %ht = func.call @gdn_fwd(%q, %k, %v, %beta, %g, %h0) : "
          f"({tq4}, {tk4}, {tv4}, {tb3}, {tg3}, {th0}) -> ({tv4}, {th0})")
        e("%o_ = memref.cast %o : " + tv4 + " to memref<*xf32>")
        e('call @printMemrefF32(%o_) : (memref<*xf32>) -> ()')
        e("%ht_ = memref.cast %ht : " + th0 + " to memref<*xf32>")
        e('call @printMemrefF32(%ht_) : (memref<*xf32>) -> ()')
    if do_bwd:
        e("%do = memref.get_global @do : " + tv4)
        e("%dht = memref.get_global @dht : " + th0)
        e(f"%dq, %dk, %dv, %dbeta, %dg, %dh0 = func.call "
          f"@gdn_bwd(%q, %k, %v, %beta, %g, %h0, %do, %dht) : "
          f"({tq4}, {tk4}, {tv4}, {tb3}, {tg3}, {th0}, {tv4}, {th0}) -> "
          f"({tq4}, {tk4}, {tv4}, {tb3}, {tg3}, {th0})")
        for nm, ty in (("%dq", tq4), ("%dk", tk4), ("%dv", tv4),
                       ("%dbeta", tb3), ("%dg", tg3), ("%dh0", th0)):
            e(f"{nm}_ = memref.cast {nm} : {ty} to memref<*xf32>")
            e(f'call @printMemrefF32({nm}_) : (memref<*xf32>) -> ()')
    if refs:
        for name in ("ref_o", "ref_ht", "ref_dq", "ref_dk", "ref_dv",
                     "ref_dbeta", "ref_dg", "ref_dh0"):
            key = name.split("_")[1]
            shapes = dict(ref_fwd_shapes)
            shapes.update(ref_bwd_shapes)
            if key in refs and not (emit == "bwd" and key in ("o", "ht")):
                ty = mrt(shapes[key])
                e(f"%{name} = memref.get_global @{name} : {ty}")
                e(f"%{name}_ = memref.cast %{name} : {ty} to memref<*xf32>")
                e(f'call @printMemrefF32(%{name}_) : (memref<*xf32>) -> ()')
    e("return")
    e.depth -= 1
    e("}")
    e.depth -= 1
    e("}")
    return e.text()


def emit_kernel(e, B, Tp, H, K, V, C, NC, scale, tq4, tk4, tv4, tb3, tg3, th0):
    o4 = tv4  # o has same shape/type class as v
    e("// Chunked GDN forward kernel. Returns (o, ht).")
    e(f"func.func @gdn_fwd(%q: {tq4}, %k: {tk4}, %v: {tv4}, %beta: {tb3}, "
      f"%g: {tg3}, %h0: {th0}) -> ({tv4}, {th0}) {{")
    e.depth += 1

    # constants
    e("%c0 = arith.constant 0 : index")
    e("%c1 = arith.constant 1 : index")
    e(f"%cB = arith.constant {B} : index")
    e(f"%cH = arith.constant {H} : index")
    e(f"%cC = arith.constant {C} : index")
    e(f"%cCm1 = arith.constant {C - 1} : index")
    e(f"%cNC = arith.constant {NC} : index")
    e("%f0 = arith.constant 0.0 : f32")
    e("%f1 = arith.constant 1.0 : f32")
    e(f"%fscale = arith.constant {fmt_f32(scale)} : f32")

    # outputs
    e(f"%o = memref.alloc() : {mrt((B, Tp, H, V))}")
    e(f"%ht = memref.alloc() : {mrt((B, H, K, V))}")

    # strided view types reused below
    t_qbh = mrt((Tp, K), strided=(H * K, 1), dyn_off=True)
    t_kbh = t_qbh
    t_vbh = mrt((Tp, V), strided=(H * V, 1), dyn_off=True)
    t_bh = mrt((Tp,), strided=(H,), dyn_off=True)
    t_qc = mrt((C, K), strided=(H * K, 1), dyn_off=True)
    t_kc = t_qc
    t_vc = mrt((C, V), strided=(H * V, 1), dyn_off=True)
    t_bc = mrt((C,), strided=(H,), dyn_off=True)
    t_obh = t_vbh
    t_oc = t_vc
    t_h0h = mrt((K, V), strided=(V, 1), dyn_off=True)
    t_C = mrt((C,))
    t_CK, t_CV, t_KV, t_CC = mrt((C, K)), mrt((C, V)), mrt((K, V)), mrt((C, C))

    e.block("scf.for %b = %c0 to %cB step %c1")
    e.block("scf.for %h = %c0 to %cH step %c1")

    # scratch buffers (per (b,h); reused across chunks)
    e("// scratch")
    e(f"%S = memref.alloc() : {t_KV}")
    e(f"%Abuf = memref.alloc() : {t_CC}")
    e(f"%Tbuf = memref.alloc() : {t_CC}")
    e(f"%kkt = memref.alloc() : {t_CC}")
    e(f"%qk = memref.alloc() : {t_CC}")
    e(f"%gc = memref.alloc() : {t_C}")
    e(f"%upre = memref.alloc() : {t_CV}")
    e(f"%ubuf = memref.alloc() : {t_CV}")
    e(f"%kb2 = memref.alloc() : {t_CK}")
    e(f"%wp = memref.alloc() : {t_CK}")
    e(f"%vp = memref.alloc() : {t_CV}")
    e(f"%vn = memref.alloc() : {t_CV}")
    e(f"%qg = memref.alloc() : {t_CK}")
    e(f"%kg = memref.alloc() : {t_CK}")
    e(f"%oib = memref.alloc() : {t_CV}")
    e(f"%oob = memref.alloc() : {t_CV}")
    e(f"%prod = memref.alloc() : {t_KV}")

    # state init: S = h0[b, h]
    e(f"%h0h = memref.subview %h0[%b, %h, 0, 0] [1, 1, {K}, {V}] [1, 1, 1, 1] : {th0} to {t_h0h}")
    e(f"linalg.copy ins(%h0h : {t_h0h}) outs(%S : {t_KV})")

    # (b,h) views
    e(f"%q_bh = memref.subview %q[%b, 0, %h, 0] [1, {Tp}, 1, {K}] [1, 1, 1, 1] : {tq4} to {t_qbh}")
    e(f"%k_bh = memref.subview %k[%b, 0, %h, 0] [1, {Tp}, 1, {K}] [1, 1, 1, 1] : {tk4} to {t_kbh}")
    e(f"%v_bh = memref.subview %v[%b, 0, %h, 0] [1, {Tp}, 1, {V}] [1, 1, 1, 1] : {tv4} to {t_vbh}")
    e(f"%beta_bh = memref.subview %beta[%b, 0, %h] [1, {Tp}, 1] [1, 1, 1] : {tb3} to {t_bh}")
    e(f"%g_bh = memref.subview %g[%b, 0, %h] [1, {Tp}, 1] [1, 1, 1] : {tg3} to {t_bh}")
    e(f"%o_bh = memref.subview %o[%b, 0, %h, 0] [1, {Tp}, 1, {V}] [1, 1, 1, 1] : {o4} to {t_obh}")

    # ---- chunk loop ----
    e.block("scf.for %n = %c0 to %cNC step %c1")
    e("%off = arith.muli %n, %cC : index")
    e(f"%q_c = memref.subview %q_bh[%off, 0] [{C}, {K}] [1, 1] : {t_qbh} to {t_qc}")
    e(f"%k_c = memref.subview %k_bh[%off, 0] [{C}, {K}] [1, 1] : {t_kbh} to {t_kc}")
    e(f"%v_c = memref.subview %v_bh[%off, 0] [{C}, {V}] [1, 1] : {t_vbh} to {t_vc}")
    e(f"%beta_c = memref.subview %beta_bh[%off] [{C}] [1] : {t_bh} to {t_bc}")
    e(f"%g_c = memref.subview %g_bh[%off] [{C}] [1] : {t_bh} to {t_bc}")
    e(f"%o_c = memref.subview %o_bh[%off, 0] [{C}, {V}] [1, 1] : {t_obh} to {t_oc}")

    # (a) inclusive chunk-local cumsum of g
    e("// (a) gc = inclusive cumsum(g)")
    e(f"%g0v = memref.load %g_c[%c0] : {t_bc}")
    e(f"memref.store %g0v, %gc[%c0] : {t_C}")
    e.block("scf.for %i = %c1 to %cC step %c1")
    e("%ip = arith.subi %i, %c1 : index")
    e(f"%prev = memref.load %gc[%ip] : {t_C}")
    e(f"%cur = memref.load %g_c[%i] : {t_bc}")
    e("%cs = arith.addf %prev, %cur : f32")
    e(f"memref.store %cs, %gc[%i] : {t_C}")
    e.end()
    e(f"%gcl = memref.load %gc[%cCm1] : {t_C}")

    # (b) kkt = k_c @ k_c^T ; A = strict-tril(beta_i * kkt * 2^(gc_i-gc_j))
    e("// (b) A = strict_tril(beta_i * (k k^T)_ij * 2^(gc_i-gc_j))")
    e(f"linalg.fill ins(%f0 : f32) outs(%kkt : {t_CC})")
    e("linalg.generic {")
    e("      indexing_maps = [")
    e("        affine_map<(i, j, r) -> (i, r)>,")
    e("        affine_map<(i, j, r) -> (j, r)>,")
    e("        affine_map<(i, j, r) -> (i, j)>")
    e("      ],")
    e('      iterator_types = ["parallel", "parallel", "reduction"]')
    e(f"    }} ins(%k_c, %k_c : {t_kc}, {t_kc}) outs(%kkt : {t_CC}) {{")
    e.depth += 1
    e("^bb0(%a: f32, %b2: f32, %acc: f32):")
    e("%p = arith.mulf %a, %b2 : f32")
    e("%s2 = arith.addf %acc, %p : f32")
    e("linalg.yield %s2 : f32")
    e.depth -= 1
    e("}")
    e("linalg.generic {")
    e("      indexing_maps = [")
    e("        affine_map<(i, j) -> (i, j)>,")
    e("        affine_map<(i, j) -> (i)>,")
    e("        affine_map<(i, j) -> (j)>,")
    e("        affine_map<(i, j) -> (i)>,")
    e("        affine_map<(i, j) -> (i, j)>")
    e("      ],")
    e('      iterator_types = ["parallel", "parallel"]')
    e(f"    }} ins(%kkt, %gc, %gc, %beta_c : {t_CC}, {t_C}, {t_C}, {t_bc}) outs(%Abuf : {t_CC}) {{")
    e.depth += 1
    e("^bb0(%kk: f32, %gi: f32, %gj: f32, %bt: f32, %out: f32):")
    e("%d0 = arith.subf %gi, %gj : f32")
    e("%e0 = math.exp2 %d0 : f32")
    e("%m0 = arith.mulf %kk, %bt : f32")
    e("%m1 = arith.mulf %m0, %e0 : f32")
    e("%i0 = linalg.index 0 : index")
    e("%j0 = linalg.index 1 : index")
    e("%lt = arith.cmpi slt, %j0, %i0 : index")
    e("%r0 = arith.select %lt, %m1, %f0 : f32")
    e("linalg.yield %r0 : f32")
    e.depth -= 1
    e("}")

    # (c) T = (I+A)^-1 forward substitution
    e("// (c) T = (I + A)^-1 via forward substitution")
    e(f"linalg.fill ins(%f0 : f32) outs(%Tbuf : {t_CC})")
    e.block("scf.for %i2 = %c0 to %cC step %c1")
    e(f"memref.store %f1, %Tbuf[%i2, %i2] : {t_CC}")
    e.block("scf.for %j2 = %c0 to %i2 step %c1")
    e(f"%sum = scf.for %k2 = %j2 to %i2 step %c1 iter_args(%acc2 = %f0) -> (f32) {{")
    e.depth += 1
    e(f"%aik = memref.load %Abuf[%i2, %k2] : {t_CC}")
    e(f"%tkj = memref.load %Tbuf[%k2, %j2] : {t_CC}")
    e("%pk = arith.mulf %aik, %tkj : f32")
    e("%ak = arith.addf %acc2, %pk : f32")
    e("scf.yield %ak : f32")
    e.depth -= 1
    e("}")
    e("%neg = arith.negf %sum : f32")
    e(f"memref.store %neg, %Tbuf[%i2, %j2] : {t_CC}")
    e.end()
    e.end()

    # (d) u = T @ (beta .* v) ; w = T @ (beta .* k .* 2.^gc)
    e("// (d) u = T @ (beta .* v), w = T @ (beta .* k .* 2.^gc)")
    e("linalg.generic {")
    e("      indexing_maps = [")
    e("        affine_map<(i, d) -> (i, d)>,")
    e("        affine_map<(i, d) -> (i)>,")
    e("        affine_map<(i, d) -> (i, d)>")
    e("      ],")
    e('      iterator_types = ["parallel", "parallel"]')
    e(f"    }} ins(%v_c, %beta_c : {t_vc}, {t_bc}) outs(%upre : {t_CV}) {{")
    e.depth += 1
    e("^bb0(%xv: f32, %bt1: f32, %out1: f32):")
    e("%r1 = arith.mulf %xv, %bt1 : f32")
    e("linalg.yield %r1 : f32")
    e.depth -= 1
    e("}")
    e(f"linalg.fill ins(%f0 : f32) outs(%ubuf : {t_CV})")
    e(f"linalg.matmul ins(%Tbuf, %upre : {t_CC}, {t_CV}) outs(%ubuf : {t_CV})")
    e("linalg.generic {")
    e("      indexing_maps = [")
    e("        affine_map<(i, d) -> (i, d)>,")
    e("        affine_map<(i, d) -> (i)>,")
    e("        affine_map<(i, d) -> (i)>,")
    e("        affine_map<(i, d) -> (i, d)>")
    e("      ],")
    e('      iterator_types = ["parallel", "parallel"]')
    e(f"    }} ins(%k_c, %beta_c, %gc : {t_kc}, {t_bc}, {t_C}) outs(%kb2 : {t_CK}) {{")
    e.depth += 1
    e("^bb0(%xk: f32, %bt2: f32, %gI: f32, %out2: f32):")
    e("%e2 = math.exp2 %gI : f32")
    e("%m3 = arith.mulf %xk, %bt2 : f32")
    e("%r2 = arith.mulf %m3, %e2 : f32")
    e("linalg.yield %r2 : f32")
    e.depth -= 1
    e("}")
    e(f"linalg.fill ins(%f0 : f32) outs(%wp : {t_CK})")
    e(f"linalg.matmul ins(%Tbuf, %kb2 : {t_CC}, {t_CK}) outs(%wp : {t_CK})")

    # (e) v_new = u - w @ S
    e("// (e) v_new = u - w @ S")
    e(f"linalg.fill ins(%f0 : f32) outs(%vp : {t_CV})")
    e(f"linalg.matmul ins(%wp, %S : {t_CK}, {t_KV}) outs(%vp : {t_CV})")
    e("linalg.generic {")
    e("      indexing_maps = [")
    e("        affine_map<(i, d) -> (i, d)>,")
    e("        affine_map<(i, d) -> (i, d)>,")
    e("        affine_map<(i, d) -> (i, d)>")
    e("      ],")
    e('      iterator_types = ["parallel", "parallel"]')
    e(f"    }} ins(%ubuf, %vp : {t_CV}, {t_CV}) outs(%vn : {t_CV}) {{")
    e.depth += 1
    e("^bb0(%xu: f32, %xv2: f32, %out3: f32):")
    e("%r3 = arith.subf %xu, %xv2 : f32")
    e("linalg.yield %r3 : f32")
    e.depth -= 1
    e("}")

    # o_inter = (q .* 2.^gc) @ S
    e("linalg.generic {")
    e("      indexing_maps = [")
    e("        affine_map<(i, d) -> (i, d)>,")
    e("        affine_map<(i, d) -> (i)>,")
    e("        affine_map<(i, d) -> (i, d)>")
    e("      ],")
    e('      iterator_types = ["parallel", "parallel"]')
    e(f"    }} ins(%q_c, %gc : {t_qc}, {t_C}) outs(%qg : {t_CK}) {{")
    e.depth += 1
    e("^bb0(%xq: f32, %gI2: f32, %out4: f32):")
    e("%e3 = math.exp2 %gI2 : f32")
    e("%r4 = arith.mulf %xq, %e3 : f32")
    e("linalg.yield %r4 : f32")
    e.depth -= 1
    e("}")
    e(f"linalg.fill ins(%f0 : f32) outs(%oob : {t_CV})")
    e(f"linalg.matmul ins(%qg, %S : {t_CK}, {t_KV}) outs(%oob : {t_CV})")

    # M = tril_inc((q k^T) .* 2^(gc_i-gc_j)) ; reuse Abuf; o_intra = M @ v_new
    e(f"linalg.fill ins(%f0 : f32) outs(%qk : {t_CC})")
    e("linalg.generic {")
    e("      indexing_maps = [")
    e("        affine_map<(i, j, r) -> (i, r)>,")
    e("        affine_map<(i, j, r) -> (j, r)>,")
    e("        affine_map<(i, j, r) -> (i, j)>")
    e("      ],")
    e('      iterator_types = ["parallel", "parallel", "reduction"]')
    e(f"    }} ins(%q_c, %k_c : {t_qc}, {t_kc}) outs(%qk : {t_CC}) {{")
    e.depth += 1
    e("^bb0(%a2: f32, %b3: f32, %acc3: f32):")
    e("%p2 = arith.mulf %a2, %b3 : f32")
    e("%s3 = arith.addf %acc3, %p2 : f32")
    e("linalg.yield %s3 : f32")
    e.depth -= 1
    e("}")
    e("linalg.generic {")
    e("      indexing_maps = [")
    e("        affine_map<(i, j) -> (i, j)>,")
    e("        affine_map<(i, j) -> (i)>,")
    e("        affine_map<(i, j) -> (j)>,")
    e("        affine_map<(i, j) -> (i, j)>")
    e("      ],")
    e('      iterator_types = ["parallel", "parallel"]')
    e(f"    }} ins(%qk, %gc, %gc : {t_CC}, {t_C}, {t_C}) outs(%Abuf : {t_CC}) {{")
    e.depth += 1
    e("^bb0(%qq: f32, %gi3: f32, %gj3: f32, %out5: f32):")
    e("%d3 = arith.subf %gi3, %gj3 : f32")
    e("%e4 = math.exp2 %d3 : f32")
    e("%m4 = arith.mulf %qq, %e4 : f32")
    e("%i3 = linalg.index 0 : index")
    e("%j3 = linalg.index 1 : index")
    e("%le = arith.cmpi sle, %j3, %i3 : index")
    e("%r5 = arith.select %le, %m4, %f0 : f32")
    e("linalg.yield %r5 : f32")
    e.depth -= 1
    e("}")
    e(f"linalg.fill ins(%f0 : f32) outs(%oib : {t_CV})")
    e(f"linalg.matmul ins(%Abuf, %vn : {t_CC}, {t_CV}) outs(%oib : {t_CV})")

    # o_c = (o_inter + o_intra) * scale
    e("linalg.generic {")
    e("      indexing_maps = [")
    e("        affine_map<(i, d) -> (i, d)>,")
    e("        affine_map<(i, d) -> (i, d)>,")
    e("        affine_map<(i, d) -> (i, d)>")
    e("      ],")
    e('      iterator_types = ["parallel", "parallel"]')
    e(f"    }} ins(%oob, %oib : {t_CV}, {t_CV}) outs(%o_c : {t_oc}) {{")
    e.depth += 1
    e("^bb0(%xa: f32, %xb: f32, %out6: f32):")
    e("%ad = arith.addf %xa, %xb : f32")
    e("%sc = arith.mulf %ad, %fscale : f32")
    e("linalg.yield %sc : f32")
    e.depth -= 1
    e("}")

    # S = S .* 2^gcl + (k .* 2.^(gcl-gc))^T @ v_new
    e("%dl = math.exp2 %gcl : f32")
    e("linalg.generic {")
    e("      indexing_maps = [")
    e("        affine_map<(i, d) -> (i, d)>,")
    e("        affine_map<(i, d) -> (i)>,")
    e("        affine_map<(i, d) -> (i, d)>")
    e("      ],")
    e('      iterator_types = ["parallel", "parallel"]')
    e(f"    }} ins(%k_c, %gc : {t_kc}, {t_C}) outs(%kg : {t_CK}) {{")
    e.depth += 1
    e("^bb0(%xk2: f32, %gI4: f32, %out7: f32):")
    e("%dg = arith.subf %gcl, %gI4 : f32")
    e("%eg = math.exp2 %dg : f32")
    e("%r6 = arith.mulf %xk2, %eg : f32")
    e("linalg.yield %r6 : f32")
    e.depth -= 1
    e("}")
    e(f"linalg.fill ins(%f0 : f32) outs(%prod : {t_KV})")
    e("linalg.generic {")
    e("      indexing_maps = [")
    e("        affine_map<(kk, vv, i) -> (i, kk)>,")
    e("        affine_map<(kk, vv, i) -> (i, vv)>,")
    e("        affine_map<(kk, vv, i) -> (kk, vv)>")
    e("      ],")
    e('      iterator_types = ["parallel", "parallel", "reduction"]')
    e(f"    }} ins(%kg, %vn : {t_CK}, {t_CV}) outs(%prod : {t_KV}) {{")
    e.depth += 1
    e("^bb0(%a4: f32, %b4: f32, %acc4: f32):")
    e("%p4 = arith.mulf %a4, %b4 : f32")
    e("%s4 = arith.addf %acc4, %p4 : f32")
    e("linalg.yield %s4 : f32")
    e.depth -= 1
    e("}")
    e("linalg.generic {")
    e("      indexing_maps = [")
    e("        affine_map<(a, b) -> (a, b)>,")
    e("        affine_map<(a, b) -> (a, b)>,")
    e("        affine_map<(a, b) -> (a, b)>")
    e("      ],")
    e('      iterator_types = ["parallel", "parallel"]')
    e(f"    }} ins(%S, %prod : {t_KV}, {t_KV}) outs(%S : {t_KV}) {{")
    e.depth += 1
    e("^bb0(%xs: f32, %xp: f32, %out8: f32):")
    e("%ms = arith.mulf %xs, %dl : f32")
    e("%rs = arith.addf %ms, %xp : f32")
    e("linalg.yield %rs : f32")
    e.depth -= 1
    e("}")
    e.end()  # chunk loop

    # ht[b, h] = S
    e(f"%hth = memref.subview %ht[%b, %h, 0, 0] [1, 1, {K}, {V}] [1, 1, 1, 1] : {mrt((B, H, K, V))} to {t_h0h}")
    e(f"linalg.copy ins(%S : {t_KV}) outs(%hth : {t_h0h})")
    for nm in ("%S", "%Abuf", "%Tbuf", "%kkt", "%qk", "%gc", "%upre", "%ubuf",
               "%kb2", "%wp", "%vp", "%vn", "%qg", "%kg", "%oib", "%oob", "%prod"):
        e(f"memref.dealloc {nm} : " + {
            "%S": t_KV, "%Abuf": t_CC, "%Tbuf": t_CC, "%kkt": t_CC, "%qk": t_CC,
            "%gc": t_C, "%upre": t_CV, "%ubuf": t_CV, "%kb2": t_CK, "%wp": t_CK,
            "%vp": t_CV, "%vn": t_CV, "%qg": t_CK, "%kg": t_CK, "%oib": t_CV,
            "%oob": t_CV, "%prod": t_KV}[nm])
    e.end()  # h loop
    e.end()  # b loop
    e(f"return %o, %ht : {tv4}, {th0}")
    e.depth -= 1
    e("}")


# ----------------------------------------------------------------------------
# backward emission (G1-B2)
# ----------------------------------------------------------------------------

def generic(e, maps, kinds, ins, outs, body):
    """Emit a linalg.generic in the same textual style as the fwd kernel.

    maps : affine-map bodies (without the affine_map<> wrapper), one per
           ins-then-outs operand, over the iterator vars.
    kinds: iterator types ("parallel"/"reduction"), one per iterator var.
    ins/outs: lists of (ssa, memref-type).
    body  : callable(ia: [str], oa: [str]) -> (lines: [str], result: str);
            ia/oa are the block-argument names for ins/outs.  Lines may use
            SSA values captured from the enclosing scope (same as the fwd
            kernel's %dl / %fscale uses)."""
    ivars = _iter_vars(maps)
    e("linalg.generic {")
    e("      indexing_maps = [")
    for m in maps:
        e(f"        affine_map<{m}>,")
    e.lines[-1] = e.lines[-1].rstrip(",")
    e("      ],")
    e('      iterator_types = [' + ", ".join(f'"{k}"' for k in kinds) + ']')
    insig = ", ".join(s for s, _ in ins) + " : " + ", ".join(t for _, t in ins)
    outsig = ", ".join(s for s, _ in outs) + " : " + ", ".join(t for _, t in outs)
    e(f"    }} ins({insig}) outs({outsig}) {{")
    e.depth += 1
    ia = [f"%a{n}" for n in range(len(ins))]
    oa = [f"%o{n}" for n in range(len(outs))]
    args = [f"{a}: f32" for a in ia] + [f"{a}: f32" for a in oa]
    e("^bb0(" + ", ".join(args) + "):")
    lines, res = body(ia, oa)
    for ln in lines:
        e(ln)
    e(f"linalg.yield {res} : f32")
    e.depth -= 1
    e("}")


def _iter_vars(maps):
    """Iterator variable names (the lhs of the first affine map)."""
    lhs = maps[0].split("->")[0]
    return [v.strip() for v in lhs.strip("()").split(",") if v.strip()]


def _fill(e, buf, ty, f0):
    e(f"linalg.fill ins({f0} : f32) outs({buf} : {ty})")


def _bwd_fwd_core(e, ctx):
    """Recompute the forward chunk intermediates (the L1 'recompute' knob):
    gc (inclusive chunk-local cumsum), kkt, A (strict-lower, decayed), T =
    (I+A)^-1 forward substitution, upre = beta*v, ubuf = u = T@upre,
    kb2 = beta*k*2^gc, wp = w = T@kb2.  Mirrors the @gdn_fwd code exactly."""
    (q_c, k_c, v_c, beta_c, g_c) = (ctx[n] for n in ("q_c", "k_c", "v_c", "beta_c", "g_c"))
    t_qc, t_kc, t_vc, t_bc = ctx["t_qc"], ctx["t_kc"], ctx["t_vc"], ctx["t_bc"]
    t_CC, t_C, t_CV, t_CK = ctx["t_CC"], ctx["t_C"], ctx["t_CV"], ctx["t_CK"]
    Abuf, Tbuf, kkt, gc, upre, ubuf, kb2, wp = (ctx[n] for n in
        ("Abuf", "Tbuf", "kkt", "gc", "upre", "ubuf", "kb2", "wp"))
    c0, c1, cC, cCm1, f0, f1 = (ctx[n] for n in ("c0", "c1", "cC", "cCm1", "f0", "f1"))

    # (a) gc = inclusive cumsum(g)
    e(f"%g0v = memref.load %g_c[{c0}] : {t_bc}")
    e(f"memref.store %g0v, {gc}[{c0}] : {t_C}")
    e.block(f"scf.for %i = {c1} to {cC} step {c1}")
    e(f"%ip = arith.subi %i, {c1} : index")
    e(f"%prev = memref.load {gc}[%ip] : {t_C}")
    e(f"%cur = memref.load %g_c[%i] : {t_bc}")
    e("%cs = arith.addf %prev, %cur : f32")
    e(f"memref.store %cs, {gc}[%i] : {t_C}")
    e.end()
    e(f"%gcl = memref.load {gc}[{cCm1}] : {t_C}")

    # (b) kkt = k k^T ; A = strict_tril(beta_i * kkt_ij * 2^(gc_i-gc_j))
    _fill(e, kkt, t_CC, f0)
    generic(e, ["(i, j, r) -> (i, r)", "(i, j, r) -> (j, r)", "(i, j, r) -> (i, j)"],
            ["parallel", "parallel", "reduction"],
            [(k_c, t_kc), (k_c, t_kc)], [(kkt, t_CC)],
            lambda a, o: (["%p = arith.mulf %a0, %a1 : f32",
                           "%s2 = arith.addf %o0, %p : f32"], "%s2"))
    generic(e, ["(i, j) -> (i, j)", "(i, j) -> (i)", "(i, j) -> (j)",
                "(i, j) -> (i)", "(i, j) -> (i, j)"],
            ["parallel", "parallel"],
            [(kkt, t_CC), (gc, t_C), (gc, t_C), (beta_c, t_bc)], [(Abuf, t_CC)],
            lambda a, o: (["%d0 = arith.subf %a1, %a2 : f32",
                           "%e0 = math.exp2 %d0 : f32",
                           "%m0 = arith.mulf %a0, %a3 : f32",
                           "%m1 = arith.mulf %m0, %e0 : f32",
                           "%i0 = linalg.index 0 : index",
                           "%j0 = linalg.index 1 : index",
                           "%lt = arith.cmpi slt, %j0, %i0 : index",
                           "%r0 = arith.select %lt, %m1, " + f0 + " : f32"], "%r0"))

    # (c) T = (I+A)^-1 forward substitution
    _fill(e, Tbuf, t_CC, f0)
    e.block(f"scf.for %i2 = {c0} to {cC} step {c1}")
    e(f"memref.store {f1}, {Tbuf}[%i2, %i2] : {t_CC}")
    e.block(f"scf.for %j2 = {c0} to %i2 step {c1}")
    e(f"%sum = scf.for %k2 = %j2 to %i2 step {c1} iter_args(%acc2 = {f0}) -> (f32) {{")
    e.depth += 1
    e(f"%aik = memref.load {Abuf}[%i2, %k2] : {t_CC}")
    e(f"%tkj = memref.load {Tbuf}[%k2, %j2] : {t_CC}")
    e("%pk = arith.mulf %aik, %tkj : f32")
    e("%ak = arith.addf %acc2, %pk : f32")
    e("scf.yield %ak : f32")
    e.depth -= 1
    e("}")
    e("%neg = arith.negf %sum : f32")
    e(f"memref.store %neg, {Tbuf}[%i2, %j2] : {t_CC}")
    e.end()
    e.end()

    # (d) upre = beta.*v ; u = T @ upre ; kb2 = beta.*k.*2^gc ; w = T @ kb2
    generic(e, ["(i, d) -> (i, d)", "(i, d) -> (i)", "(i, d) -> (i, d)"],
            ["parallel", "parallel"],
            [(v_c, t_vc), (beta_c, t_bc)], [(upre, t_CV)],
            lambda a, o: (["%r1 = arith.mulf %a0, %a1 : f32"], "%r1"))
    _fill(e, ubuf, t_CV, f0)
    e(f"linalg.matmul ins({Tbuf}, {upre} : {t_CC}, {t_CV}) outs({ubuf} : {t_CV})")
    generic(e, ["(i, d) -> (i, d)", "(i, d) -> (i)", "(i, d) -> (i)",
                "(i, d) -> (i, d)"],
            ["parallel", "parallel"],
            [(k_c, t_kc), (beta_c, t_bc), (gc, t_C)], [(kb2, t_CK)],
            lambda a, o: (["%e2 = math.exp2 %a2 : f32",
                           "%m3 = arith.mulf %a0, %a1 : f32",
                           "%r2 = arith.mulf %m3, %e2 : f32"], "%r2"))
    _fill(e, wp, t_CK, f0)
    e(f"linalg.matmul ins({Tbuf}, {kb2} : {t_CC}, {t_CK}) outs({wp} : {t_CK})")
    return "%gcl"


def emit_bwd_kernel(e, B, Tp, H, K, V, C, NC, scale,
                    tq4, tk4, tv4, tb3, tg3, th0):
    """@gdn_bwd(q,k,v,beta,g,h0,do,dht) -> (dq,dk,dv,dbeta,dg,dh0).

    Structure per (b,h): PASS 1 runs the chunked forward recompute storing
    only the per-chunk ENTRY states h_all[n] (recompute-vs-materialize knob);
    PASS 2 walks the chunks in REVERSE, recomputing gc/A/T/u/w per chunk,
    carrying the reverse state cotangent dH (seeded from dht):
      dvp  = P^T dO + Bhat dH                  (cotangent on v_new)
      dH'  = 2^gcl dH + (q.*2^gc)^T dO - w^T dvp   (dh0 = dH after chunk 0)
      dW   = -dvp S_n^T ;  G = dvp (beta v)^T + dW (beta k 2^gc)^T
      dA   = -strict_tril(T^T strict_tril(G) T^T)   (exact (I+A)^-1 adjoint)
      dAf  = dA .* 2^(gc_i-gc_j)                    (decay-free coords)
      dq   = (dO S^T).*2^gc + dPm k      dPm = (dO vn^T).*D.*tril_inc
      dk   = (dAf k).*beta + dAf^T(beta k) + (T^T dW).*(beta 2^gc)
             + (vn dH^T).*2^(gcl-gc) + dPm^T q
      dv   = (T^T dvp).*beta
      dbeta= rowsum(dAf.*(k k^T)) + rowsum((T^T dW).*(k 2^gc))
             + rowsum((T^T dvp).*v)
      dg_t = suffixsum_t[dA-chain rows-cols + W-chain + o_inter rows
                         + o_intra (rows-cols of (dO vn^T).*P)]
           + prefixsum_t[rowsum((vn dH^T .* 2^(gcl-gc)).*k)] + 2^gcl sum(dH.*S)
           (all decay factors are exp2(gc_log2) == e^(gc_nat), so the pieces
            are already natural-log-domain cotangents -- no RCP_LN2 rescale)
    """
    tdq4, tdk4, tdv4 = tq4, tk4, tv4
    tdb3, tdg3, tdh0 = tb3, tg3, th0
    e("// Chunked GDN backward kernel. Returns (dq, dk, dv, dbeta, dg, dh0).")
    e(f"func.func @gdn_bwd(%q: {tq4}, %k: {tk4}, %v: {tv4}, %beta: {tb3}, "
      f"%g: {tg3}, %h0: {th0}, %do: {tv4}, %dht: {th0}) -> "
      f"({tdq4}, {tdk4}, {tdv4}, {tdb3}, {tdg3}, {tdh0}) {{")
    e.depth += 1

    # constants
    e("%c0 = arith.constant 0 : index")
    e("%c1 = arith.constant 1 : index")
    e(f"%cB = arith.constant {B} : index")
    e(f"%cH = arith.constant {H} : index")
    e(f"%cC = arith.constant {C} : index")
    e(f"%cCm1 = arith.constant {C - 1} : index")
    e(f"%cNC = arith.constant {NC} : index")
    e("%f0 = arith.constant 0.0 : f32")
    e("%f1 = arith.constant 1.0 : f32")
    e(f"%fscale = arith.constant {fmt_f32(scale)} : f32")

    # outputs
    e(f"%dq = memref.alloc() : {mrt((B, Tp, H, K))}")
    e(f"%dk = memref.alloc() : {mrt((B, Tp, H, K))}")
    e(f"%dv = memref.alloc() : {mrt((B, Tp, H, V))}")
    e(f"%dbeta = memref.alloc() : {mrt((B, Tp, H))}")
    e(f"%dg = memref.alloc() : {mrt((B, Tp, H))}")
    e(f"%dh0o = memref.alloc() : {mrt((B, H, K, V))}")

    # strided view types
    t_qbh = mrt((Tp, K), strided=(H * K, 1), dyn_off=True)
    t_vbh = mrt((Tp, V), strided=(H * V, 1), dyn_off=True)
    t_bh = mrt((Tp,), strided=(H,), dyn_off=True)
    t_qc = mrt((C, K), strided=(H * K, 1), dyn_off=True)
    t_vc = mrt((C, V), strided=(H * V, 1), dyn_off=True)
    t_bc = mrt((C,), strided=(H,), dyn_off=True)
    t_h0h = mrt((K, V), strided=(V, 1), dyn_off=True)
    t_C = mrt((C,))
    t_C1 = mrt((1,))
    t_CK, t_CV, t_KV, t_CC = mrt((C, K)), mrt((C, V)), mrt((K, V)), mrt((C, C))
    t_HALL = mrt((NC, K, V))

    e.block("scf.for %b = %c0 to %cB step %c1")
    e.block("scf.for %h = %c0 to %cH step %c1")

    # scratch (per (b,h))
    e("// scratch")
    allocs = [
        ("S", t_KV), ("dH", t_KV), ("prod", t_KV), ("tK1", t_KV), ("tK2", t_KV),
        ("h_all", t_HALL), ("gc", t_C),
        ("kkt", t_CC), ("Abuf", t_CC), ("Tbuf", t_CC), ("dPr", t_CC),
        ("dPm", t_CC), ("Gb", t_CC), ("Xb", t_CC),
        ("kb2", t_CK), ("wp", t_CK), ("qg", t_CK), ("TW", t_CK),
        ("dkB", t_CK), ("dq1", t_CK), ("dkacc", t_CK), ("dW", t_CK),
        ("upre", t_CV), ("ubuf", t_CV), ("vp", t_CV), ("vn", t_CV),
        ("kg", t_CK), ("dO", t_CV), ("dOgc", t_CV), ("dvp", t_CV),
        ("tv", t_CV), ("vng", t_CV),
        ("rowB", t_C), ("ra", t_C), ("rb", t_C), ("rc", t_C), ("r3v", t_C),
        ("r4", t_C), ("c1r", t_C), ("c4r", t_C), ("tA", t_C), ("tB", t_C),
        ("dgr", t_C), ("dgp2", t_C), ("s5", t_C1),
    ]
    for nm, ty in allocs:
        e(f"%{nm} = memref.alloc() : {ty}")

    ctx = dict(t_qc=t_qc, t_kc=t_qc, t_vc=t_vc, t_bc=t_bc, t_CC=t_CC, t_C=t_C,
               t_CV=t_CV, t_CK=t_CK, t_KV=t_KV, c0="%c0", c1="%c1", cC="%cC",
               cCm1="%cCm1", f0="%f0", f1="%f1")
    for nm, _ in allocs:
        ctx[nm] = f"%{nm}"

    # state init: S = h0[b,h]; dH = dht[b,h]
    e(f"%h0h = memref.subview %h0[%b, %h, 0, 0] [1, 1, {K}, {V}] [1, 1, 1, 1] : {th0} to {t_h0h}")
    e(f"linalg.copy ins(%h0h : {t_h0h}) outs(%S : {t_KV})")
    e(f"%dhth = memref.subview %dht[%b, %h, 0, 0] [1, 1, {K}, {V}] [1, 1, 1, 1] : {th0} to {t_h0h}")
    e(f"linalg.copy ins(%dhth : {t_h0h}) outs(%dH : {t_KV})")

    # (b,h) views
    e(f"%q_bh = memref.subview %q[%b, 0, %h, 0] [1, {Tp}, 1, {K}] [1, 1, 1, 1] : {tq4} to {t_qbh}")
    e(f"%k_bh = memref.subview %k[%b, 0, %h, 0] [1, {Tp}, 1, {K}] [1, 1, 1, 1] : {tk4} to {t_qbh}")
    e(f"%v_bh = memref.subview %v[%b, 0, %h, 0] [1, {Tp}, 1, {V}] [1, 1, 1, 1] : {tv4} to {t_vbh}")
    e(f"%beta_bh = memref.subview %beta[%b, 0, %h] [1, {Tp}, 1] [1, 1, 1] : {tb3} to {t_bh}")
    e(f"%g_bh = memref.subview %g[%b, 0, %h] [1, {Tp}, 1] [1, 1, 1] : {tg3} to {t_bh}")
    e(f"%do_bh = memref.subview %do[%b, 0, %h, 0] [1, {Tp}, 1, {V}] [1, 1, 1, 1] : {tv4} to {t_vbh}")
    e(f"%dq_bh = memref.subview %dq[%b, 0, %h, 0] [1, {Tp}, 1, {K}] [1, 1, 1, 1] : {tdq4} to {t_qbh}")
    e(f"%dk_bh = memref.subview %dk[%b, 0, %h, 0] [1, {Tp}, 1, {K}] [1, 1, 1, 1] : {tdk4} to {t_qbh}")
    e(f"%dv_bh = memref.subview %dv[%b, 0, %h, 0] [1, {Tp}, 1, {V}] [1, 1, 1, 1] : {tdv4} to {t_vbh}")
    e(f"%dbeta_bh = memref.subview %dbeta[%b, 0, %h] [1, {Tp}, 1] [1, 1, 1] : {tdb3} to {t_bh}")
    e(f"%dg_bh = memref.subview %dg[%b, 0, %h] [1, {Tp}, 1] [1, 1, 1] : {tdg3} to {t_bh}")

    # ---- PASS 1: forward recompute, storing per-chunk entry states -------
    e("// ---- pass 1: forward recompute (store entry states h_all[n]) ----")
    e.block("scf.for %n = %c0 to %cNC step %c1")
    e("%off = arith.muli %n, %cC : index")
    e(f"%q_c = memref.subview %q_bh[%off, 0] [{C}, {K}] [1, 1] : {t_qbh} to {t_qc}")
    e(f"%k_c = memref.subview %k_bh[%off, 0] [{C}, {K}] [1, 1] : {t_qbh} to {t_qc}")
    e(f"%v_c = memref.subview %v_bh[%off, 0] [{C}, {V}] [1, 1] : {t_vbh} to {t_vc}")
    e(f"%beta_c = memref.subview %beta_bh[%off] [{C}] [1] : {t_bh} to {t_bc}")
    e(f"%g_c = memref.subview %g_bh[%off] [{C}] [1] : {t_bh} to {t_bc}")
    ctx.update(q_c="%q_c", k_c="%k_c", v_c="%v_c", beta_c="%beta_c", g_c="%g_c")
    e(f"%h_all_n = memref.subview %h_all[%n, 0, 0] [1, {K}, {V}] [1, 1, 1] : {t_HALL} to {t_h0h}")
    e(f"linalg.copy ins(%S : {t_KV}) outs(%h_all_n : {t_h0h})")
    gcl = _bwd_fwd_core(e, ctx)
    # v_new and the S update (fwd math, state in %S)
    _fill(e, "%vp", t_CV, "%f0")
    e(f"linalg.matmul ins(%wp, %S : {t_CK}, {t_KV}) outs(%vp : {t_CV})")
    generic(e, ["(i, d) -> (i, d)", "(i, d) -> (i, d)", "(i, d) -> (i, d)"],
            ["parallel", "parallel"],
            [("%ubuf", t_CV), ("%vp", t_CV)], [("%vn", t_CV)],
            lambda a, o: (["%r3 = arith.subf %a0, %a1 : f32"], "%r3"))
    e(f"%dl = math.exp2 {gcl} : f32")
    generic(e, ["(i, d) -> (i, d)", "(i, d) -> (i)", "(i, d) -> (i, d)"],
            ["parallel", "parallel"],
            [("%k_c", t_qc), ("%gc", t_C)], [("%kg", t_CK)],
            lambda a, o: ([f"%kgd = arith.subf {gcl}, %a1 : f32",
                           "%kge = math.exp2 %kgd : f32",
                           "%r6 = arith.mulf %a0, %kge : f32"], "%r6"))
    _fill(e, "%prod", t_KV, "%f0")
    generic(e, ["(kk, vv, i) -> (i, kk)", "(kk, vv, i) -> (i, vv)",
                "(kk, vv, i) -> (kk, vv)"],
            ["parallel", "parallel", "reduction"],
            [("%kg", t_CK), ("%vn", t_CV)], [("%prod", t_KV)],
            lambda a, o: (["%p4 = arith.mulf %a0, %a1 : f32",
                           "%s4 = arith.addf %o0, %p4 : f32"], "%s4"))
    generic(e, ["(a, b) -> (a, b)", "(a, b) -> (a, b)", "(a, b) -> (a, b)"],
            ["parallel", "parallel"],
            [("%S", t_KV), ("%prod", t_KV)], [("%S", t_KV)],
            lambda a, o: (["%ms = arith.mulf %a0, %dl : f32",
                           "%rs = arith.addf %ms, %a1 : f32"], "%rs"))
    e.end()  # pass 1 chunk loop

    # ---- PASS 2: reverse chunk walk ---------------------------------------
    e("// ---- pass 2: reverse chunk walk (dH seeded from dht) ----")
    e.block("scf.for %n2 = %c0 to %cNC step %c1")
    e("%rn = arith.subi %cNC, %c1 : index")
    e("%n = arith.subi %rn, %n2 : index")
    e("%off = arith.muli %n, %cC : index")
    e(f"%q_c = memref.subview %q_bh[%off, 0] [{C}, {K}] [1, 1] : {t_qbh} to {t_qc}")
    e(f"%k_c = memref.subview %k_bh[%off, 0] [{C}, {K}] [1, 1] : {t_qbh} to {t_qc}")
    e(f"%v_c = memref.subview %v_bh[%off, 0] [{C}, {V}] [1, 1] : {t_vbh} to {t_vc}")
    e(f"%beta_c = memref.subview %beta_bh[%off] [{C}] [1] : {t_bh} to {t_bc}")
    e(f"%g_c = memref.subview %g_bh[%off] [{C}] [1] : {t_bh} to {t_bc}")
    e(f"%do_c = memref.subview %do_bh[%off, 0] [{C}, {V}] [1, 1] : {t_vbh} to {t_vc}")
    e(f"%dq_c = memref.subview %dq_bh[%off, 0] [{C}, {K}] [1, 1] : {t_qbh} to {t_qc}")
    e(f"%dk_c = memref.subview %dk_bh[%off, 0] [{C}, {K}] [1, 1] : {t_qbh} to {t_qc}")
    e(f"%dv_c = memref.subview %dv_bh[%off, 0] [{C}, {V}] [1, 1] : {t_vbh} to {t_vc}")
    e(f"%dbeta_c = memref.subview %dbeta_bh[%off] [{C}] [1] : {t_bh} to {t_bc}")
    e(f"%dg_c = memref.subview %dg_bh[%off] [{C}] [1] : {t_bh} to {t_bc}")
    e(f"%Sn = memref.subview %h_all[%n, 0, 0] [1, {K}, {V}] [1, 1, 1] : {t_HALL} to {t_h0h}")
    ctx.update(q_c="%q_c", k_c="%k_c", v_c="%v_c", beta_c="%beta_c", g_c="%g_c")
    gcl = _bwd_fwd_core(e, ctx)
    e(f"%dl = math.exp2 {gcl} : f32")

    # v_new = u - w @ Sn
    _fill(e, "%vp", t_CV, "%f0")
    generic(e, ["(i, d, kk) -> (i, kk)", "(i, d, kk) -> (kk, d)",
                "(i, d, kk) -> (i, d)"],
            ["parallel", "parallel", "reduction"],
            [("%wp", t_CK), ("%Sn", t_h0h)], [("%vp", t_CV)],
            lambda a, o: (["%p5 = arith.mulf %a0, %a1 : f32",
                           "%s5p = arith.addf %o0, %p5 : f32"], "%s5p"))
    generic(e, ["(i, d) -> (i, d)", "(i, d) -> (i, d)", "(i, d) -> (i, d)"],
            ["parallel", "parallel"],
            [("%ubuf", t_CV), ("%vp", t_CV)], [("%vn", t_CV)],
            lambda a, o: (["%r3 = arith.subf %a0, %a1 : f32"], "%r3"))

    # dO = do*scale ; dOgc = dO .* 2^gc
    generic(e, ["(i, d) -> (i, d)", "(i, d) -> (i, d)"],
            ["parallel", "parallel"],
            [("%do_c", t_vc)], [("%dO", t_CV)],
            lambda a, o: (["%od = arith.mulf %a0, %fscale : f32"], "%od"))
    generic(e, ["(i, d) -> (i, d)", "(i, d) -> (i)", "(i, d) -> (i, d)"],
            ["parallel", "parallel"],
            [("%dO", t_CV), ("%gc", t_C)], [("%dOgc", t_CV)],
            lambda a, o: (["%eog = math.exp2 %a1 : f32",
                           "%rog = arith.mulf %a0, %eog : f32"], "%rog"))

    # P = tril_inc((q k^T) .* 2^(gc_i-gc_j)) into Abuf (A is dead after T)
    _fill(e, "%Abuf", t_CC, "%f0")
    generic(e, ["(i, j, r) -> (i, r)", "(i, j, r) -> (j, r)", "(i, j, r) -> (i, j)"],
            ["parallel", "parallel", "reduction"],
            [("%q_c", t_qc), ("%k_c", t_qc)], [("%Abuf", t_CC)],
            lambda a, o: (["%p2 = arith.mulf %a0, %a1 : f32",
                           "%s3 = arith.addf %o0, %p2 : f32"], "%s3"))
    generic(e, ["(i, j) -> (i, j)", "(i, j) -> (i)", "(i, j) -> (j)",
                "(i, j) -> (i, j)"],
            ["parallel", "parallel"],
            [("%Abuf", t_CC), ("%gc", t_C), ("%gc", t_C)], [("%Abuf", t_CC)],
            lambda a, o: (["%d3 = arith.subf %a1, %a2 : f32",
                           "%e4 = math.exp2 %d3 : f32",
                           "%m4 = arith.mulf %a0, %e4 : f32",
                           "%i3 = linalg.index 0 : index",
                           "%j3 = linalg.index 1 : index",
                           "%le = arith.cmpi sle, %j3, %i3 : index",
                           "%r5 = arith.select %le, %m4, %f0 : f32"], "%r5"))

    # dPr = dO vn^T ; dPm = dPr .* D .* tril_inc
    _fill(e, "%dPr", t_CC, "%f0")
    generic(e, ["(i, j, d) -> (i, d)", "(i, j, d) -> (j, d)", "(i, j, d) -> (i, j)"],
            ["parallel", "parallel", "reduction"],
            [("%dO", t_CV), ("%vn", t_CV)], [("%dPr", t_CC)],
            lambda a, o: (["%pd = arith.mulf %a0, %a1 : f32",
                           "%sd = arith.addf %o0, %pd : f32"], "%sd"))
    generic(e, ["(i, j) -> (i, j)", "(i, j) -> (i)", "(i, j) -> (j)",
                "(i, j) -> (i, j)"],
            ["parallel", "parallel"],
            [("%dPr", t_CC), ("%gc", t_C), ("%gc", t_C)], [("%dPm", t_CC)],
            lambda a, o: (["%dpd = arith.subf %a1, %a2 : f32",
                           "%dpe = math.exp2 %dpd : f32",
                           "%dpm = arith.mulf %a0, %dpe : f32",
                           "%di = linalg.index 0 : index",
                           "%dj = linalg.index 1 : index",
                           "%dle = arith.cmpi sle, %dj, %di : index",
                           "%dr5 = arith.select %dle, %dpm, %f0 : f32"], "%dr5"))

    # dvp = P^T dO + Bhat dH   (Bhat = k .* 2^(gcl-gc), folded inline)
    _fill(e, "%dvp", t_CV, "%f0")
    generic(e, ["(i, d, j) -> (j, i)", "(i, d, j) -> (j, d)", "(i, d, j) -> (i, d)"],
            ["parallel", "parallel", "reduction"],
            [("%Abuf", t_CC), ("%dO", t_CV)], [("%dvp", t_CV)],
            lambda a, o: (["%pt = arith.mulf %a0, %a1 : f32",
                           "%st = arith.addf %o0, %pt : f32"], "%st"))
    generic(e, ["(i, d, kk) -> (i, kk)", "(i, d, kk) -> (i)", "(i, d, kk) -> (kk, d)",
                "(i, d, kk) -> (i, d)"],
            ["parallel", "parallel", "reduction"],
            [("%k_c", t_qc), ("%gc", t_C), ("%dH", t_KV)], [("%dvp", t_CV)],
            lambda a, o: ([f"%bd = arith.subf {gcl}, %a1 : f32",
                           "%be = math.exp2 %bd : f32",
                           "%bm = arith.mulf %a0, %be : f32",
                           "%bp = arith.mulf %bm, %a2 : f32",
                           "%bs = arith.addf %o0, %bp : f32"], "%bs"))

    # dW = -(dvp Sn^T)
    _fill(e, "%dW", t_CK, "%f0")
    generic(e, ["(i, kk, d) -> (i, d)", "(i, kk, d) -> (kk, d)",
                "(i, kk, d) -> (i, kk)"],
            ["parallel", "parallel", "reduction"],
            [("%dvp", t_CV), ("%Sn", t_h0h)], [("%dW", t_CK)],
            lambda a, o: (["%wp2 = arith.mulf %a0, %a1 : f32",
                           "%ws = arith.subf %o0, %wp2 : f32"], "%ws"))

    # G = dvp (beta v)^T + dW (beta k 2^gc)^T ; mask strict-lower
    _fill(e, "%Gb", t_CC, "%f0")
    generic(e, ["(i, j, d) -> (i, d)", "(i, j, d) -> (j, d)", "(i, j, d) -> (i, j)"],
            ["parallel", "parallel", "reduction"],
            [("%dvp", t_CV), ("%upre", t_CV)], [("%Gb", t_CC)],
            lambda a, o: (["%gp1 = arith.mulf %a0, %a1 : f32",
                           "%gs1 = arith.addf %o0, %gp1 : f32"], "%gs1"))
    generic(e, ["(i, j, kk) -> (i, kk)", "(i, j, kk) -> (j, kk)",
                "(i, j, kk) -> (i, j)"],
            ["parallel", "parallel", "reduction"],
            [("%dW", t_CK), ("%kb2", t_CK)], [("%Gb", t_CC)],
            lambda a, o: (["%gp2 = arith.mulf %a0, %a1 : f32",
                           "%gs2 = arith.addf %o0, %gp2 : f32"], "%gs2"))
    generic(e, ["(i, j) -> (i, j)", "(i, j) -> (i, j)"],
            ["parallel", "parallel"],
            [("%Gb", t_CC)], [("%Gb", t_CC)],
            lambda a, o: (["%gi0 = linalg.index 0 : index",
                           "%gj0 = linalg.index 1 : index",
                           "%glt = arith.cmpi slt, %gj0, %gi0 : index",
                           "%gr0 = arith.select %glt, %a0, %f0 : f32"], "%gr0"))

    # X = G T^T ; Y = T^T X ; dAf = -strict_tril(Y .* D) into Gb (decay-free)
    _fill(e, "%Xb", t_CC, "%f0")
    generic(e, ["(i, j, m) -> (i, m)", "(i, j, m) -> (j, m)", "(i, j, m) -> (i, j)"],
            ["parallel", "parallel", "reduction"],
            [("%Gb", t_CC), ("%Tbuf", t_CC)], [("%Xb", t_CC)],
            lambda a, o: (["%xp = arith.mulf %a0, %a1 : f32",
                           "%xs = arith.addf %o0, %xp : f32"], "%xs"))
    _fill(e, "%Gb", t_CC, "%f0")
    generic(e, ["(i, j, m) -> (m, i)", "(i, j, m) -> (m, j)", "(i, j, m) -> (i, j)"],
            ["parallel", "parallel", "reduction"],
            [("%Tbuf", t_CC), ("%Xb", t_CC)], [("%Gb", t_CC)],
            lambda a, o: (["%yp = arith.mulf %a0, %a1 : f32",
                           "%ys = arith.addf %o0, %yp : f32"], "%ys"))
    generic(e, ["(i, j) -> (i, j)", "(i, j) -> (i)", "(i, j) -> (j)",
                "(i, j) -> (i, j)"],
            ["parallel", "parallel"],
            [("%Gb", t_CC), ("%gc", t_C), ("%gc", t_C)], [("%Gb", t_CC)],
            lambda a, o: (["%dd0 = arith.subf %a1, %a2 : f32",
                           "%de0 = math.exp2 %dd0 : f32",
                           "%dm0 = arith.mulf %a0, %de0 : f32",
                           "%dn0 = arith.negf %dm0 : f32",
                           "%di0 = linalg.index 0 : index",
                           "%dj0 = linalg.index 1 : index",
                           "%dlt = arith.cmpi slt, %dj0, %di0 : index",
                           "%dr0 = arith.select %dlt, %dn0, %f0 : f32"], "%dr0"))

    # TW = T^T dW ; tv = T^T dvp
    _fill(e, "%TW", t_CK, "%f0")
    generic(e, ["(i, kk, j) -> (j, i)", "(i, kk, j) -> (j, kk)",
                "(i, kk, j) -> (i, kk)"],
            ["parallel", "parallel", "reduction"],
            [("%Tbuf", t_CC), ("%dW", t_CK)], [("%TW", t_CK)],
            lambda a, o: (["%tp1 = arith.mulf %a0, %a1 : f32",
                           "%ts1 = arith.addf %o0, %tp1 : f32"], "%ts1"))
    _fill(e, "%tv", t_CV, "%f0")
    generic(e, ["(i, d, j) -> (j, i)", "(i, d, j) -> (j, d)", "(i, d, j) -> (i, d)"],
            ["parallel", "parallel", "reduction"],
            [("%Tbuf", t_CC), ("%dvp", t_CV)], [("%tv", t_CV)],
            lambda a, o: (["%tp2 = arith.mulf %a0, %a1 : f32",
                           "%ts2 = arith.addf %o0, %tp2 : f32"], "%ts2"))

    # dv = tv .* beta
    generic(e, ["(i, d) -> (i, d)", "(i, d) -> (i)", "(i, d) -> (i, d)"],
            ["parallel", "parallel"],
            [("%tv", t_CV), ("%beta_c", t_bc)], [("%dv_c", t_vc)],
            lambda a, o: (["%dvr = arith.mulf %a0, %a1 : f32"], "%dvr"))

    # dbeta = rowsum(dAf .* kkT) + rowsum(TW .* kb2) + rowsum(tv .* v)
    _fill(e, "%ra", t_C, "%f0")
    generic(e, ["(i, j) -> (i, j)", "(i, j) -> (i, j)", "(i, j) -> (i)"],
            ["parallel", "reduction"],
            [("%Gb", t_CC), ("%kkt", t_CC)], [("%ra", t_C)],
            lambda a, o: (["%ra1 = arith.mulf %a0, %a1 : f32",
                           "%ra2 = arith.addf %o0, %ra1 : f32"], "%ra2"))
    _fill(e, "%rb", t_C, "%f0")
    generic(e, ["(i, kk) -> (i, kk)", "(i, kk) -> (i, kk)", "(i, kk) -> (i)",
                "(i, kk) -> (i)"],
            ["parallel", "reduction"],
            [("%TW", t_CK), ("%k_c", t_qc), ("%gc", t_C)], [("%rb", t_C)],
            lambda a, o: (["%rb0 = math.exp2 %a2 : f32",
                           "%rb0k = arith.mulf %a0, %a1 : f32",
                           "%rb1 = arith.mulf %rb0k, %rb0 : f32",
                           "%rb2 = arith.addf %o0, %rb1 : f32"], "%rb2"))
    _fill(e, "%rc", t_C, "%f0")
    generic(e, ["(i, d) -> (i, d)", "(i, d) -> (i, d)", "(i, d) -> (i)"],
            ["parallel", "reduction"],
            [("%tv", t_CV), ("%v_c", t_vc)], [("%rc", t_C)],
            lambda a, o: (["%rc1 = arith.mulf %a0, %a1 : f32",
                           "%rc2 = arith.addf %o0, %rc1 : f32"], "%rc2"))
    generic(e, ["(i) -> (i)", "(i) -> (i)", "(i) -> (i)", "(i) -> (i)"],
            ["parallel"],
            [("%ra", t_C), ("%rb", t_C), ("%rc", t_C)], [("%dbeta_c", t_bc)],
            lambda a, o: (["%db1 = arith.addf %a0, %a1 : f32",
                           "%db2 = arith.addf %db1, %a2 : f32"], "%db2"))

    # qg = q .* 2^gc ; dq1 = dOgc Sn^T ; r3 = rowsum(dq1 .* q)
    generic(e, ["(i, kk) -> (i, kk)", "(i, kk) -> (i)", "(i, kk) -> (i, kk)"],
            ["parallel", "parallel"],
            [("%q_c", t_qc), ("%gc", t_C)], [("%qg", t_CK)],
            lambda a, o: (["%qg1 = math.exp2 %a1 : f32",
                           "%qg2 = arith.mulf %a0, %qg1 : f32"], "%qg2"))
    _fill(e, "%dq1", t_CK, "%f0")
    generic(e, ["(i, kk, d) -> (i, d)", "(i, kk, d) -> (kk, d)",
                "(i, kk, d) -> (i, kk)"],
            ["parallel", "parallel", "reduction"],
            [("%dOgc", t_CV), ("%Sn", t_h0h)], [("%dq1", t_CK)],
            lambda a, o: (["%dq1p = arith.mulf %a0, %a1 : f32",
                           "%dq1s = arith.addf %o0, %dq1p : f32"], "%dq1s"))
    _fill(e, "%r3v", t_C, "%f0")
    generic(e, ["(i, kk) -> (i, kk)", "(i, kk) -> (i, kk)", "(i, kk) -> (i)"],
            ["parallel", "reduction"],
            [("%dq1", t_CK), ("%q_c", t_qc)], [("%r3v", t_C)],
            lambda a, o: (["%r3p = arith.mulf %a0, %a1 : f32",
                           "%r3s = arith.addf %o0, %r3p : f32"], "%r3s"))
    # dq += dPm @ k ; store
    generic(e, ["(i, kk, j) -> (i, j)", "(i, kk, j) -> (j, kk)",
                "(i, kk, j) -> (i, kk)"],
            ["parallel", "parallel", "reduction"],
            [("%dPm", t_CC), ("%k_c", t_qc)], [("%dq1", t_CK)],
            lambda a, o: (["%dq2p = arith.mulf %a0, %a1 : f32",
                           "%dq2s = arith.addf %o0, %dq2p : f32"], "%dq2s"))
    e(f"linalg.copy ins(%dq1 : {t_CK}) outs(%dq_c : {t_qc})")

    # dkB = (vn .* 2^(gcl-gc)) dH^T ; rowB = rowsum(dkB .* k)
    generic(e, ["(i, d) -> (i, d)", "(i, d) -> (i)", "(i, d) -> (i, d)"],
            ["parallel", "parallel"],
            [("%vn", t_CV), ("%gc", t_C)], [("%vng", t_CV)],
            lambda a, o: ([f"%vg1 = arith.subf {gcl}, %a1 : f32",
                           "%vg2 = math.exp2 %vg1 : f32",
                           "%vg3 = arith.mulf %a0, %vg2 : f32"], "%vg3"))
    _fill(e, "%dkB", t_CK, "%f0")
    generic(e, ["(i, kk, d) -> (i, d)", "(i, kk, d) -> (kk, d)",
                "(i, kk, d) -> (i, kk)"],
            ["parallel", "parallel", "reduction"],
            [("%vng", t_CV), ("%dH", t_KV)], [("%dkB", t_CK)],
            lambda a, o: (["%kb1 = arith.mulf %a0, %a1 : f32",
                           "%kb2s = arith.addf %o0, %kb1 : f32"], "%kb2s"))
    _fill(e, "%rowB", t_C, "%f0")
    generic(e, ["(i, kk) -> (i, kk)", "(i, kk) -> (i, kk)", "(i, kk) -> (i)"],
            ["parallel", "reduction"],
            [("%dkB", t_CK), ("%k_c", t_qc)], [("%rowB", t_C)],
            lambda a, o: (["%rbw = arith.mulf %a0, %a1 : f32",
                           "%rbw2 = arith.addf %o0, %rbw : f32"], "%rbw2"))

    # dk = (dAf k).*beta + dAf^T (beta k) + TW.*(beta .* 2^gc) + dkB + dPm^T q
    _fill(e, "%dkacc", t_CK, "%f0")
    generic(e, ["(i, kk, j) -> (i, j)", "(i, kk, j) -> (i)", "(i, kk, j) -> (j, kk)",
                "(i, kk, j) -> (i, kk)"],
            ["parallel", "parallel", "reduction"],
            [("%Gb", t_CC), ("%beta_c", t_bc), ("%k_c", t_qc)], [("%dkacc", t_CK)],
            lambda a, o: (["%ka1 = arith.mulf %a0, %a1 : f32",
                           "%ka2 = arith.mulf %ka1, %a2 : f32",
                           "%ka3 = arith.addf %o0, %ka2 : f32"], "%ka3"))
    generic(e, ["(i, kk, j) -> (j, i)", "(i, kk, j) -> (j)", "(i, kk, j) -> (j, kk)",
                "(i, kk, j) -> (i, kk)"],
            ["parallel", "parallel", "reduction"],
            [("%Gb", t_CC), ("%beta_c", t_bc), ("%k_c", t_qc)], [("%dkacc", t_CK)],
            lambda a, o: (["%ka4 = arith.mulf %a0, %a1 : f32",
                           "%ka5 = arith.mulf %ka4, %a2 : f32",
                           "%ka6 = arith.addf %o0, %ka5 : f32"], "%ka6"))
    generic(e, ["(i, kk) -> (i, kk)", "(i, kk) -> (i, kk)", "(i, kk) -> (i)",
                "(i, kk) -> (i)", "(i, kk) -> (i, kk)"],
            ["parallel", "parallel"],
            [("%dkacc", t_CK), ("%TW", t_CK), ("%beta_c", t_bc), ("%gc", t_C)],
            [("%dkacc", t_CK)],
            lambda a, o: (["%ka7 = math.exp2 %a3 : f32",
                           "%ka8 = arith.mulf %a2, %ka7 : f32",
                           "%ka9 = arith.mulf %a1, %ka8 : f32",
                           "%kaa = arith.addf %a0, %ka9 : f32"], "%kaa"))
    generic(e, ["(i, kk) -> (i, kk)", "(i, kk) -> (i, kk)", "(i, kk) -> (i, kk)"],
            ["parallel", "parallel"],
            [("%dkacc", t_CK), ("%dkB", t_CK)], [("%dkacc", t_CK)],
            lambda a, o: (["%kab = arith.addf %a0, %a1 : f32"], "%kab"))
    generic(e, ["(i, kk, j) -> (j, i)", "(i, kk, j) -> (j, kk)",
                "(i, kk, j) -> (i, kk)"],
            ["parallel", "parallel", "reduction"],
            [("%dPm", t_CC), ("%q_c", t_qc)], [("%dkacc", t_CK)],
            lambda a, o: (["%kac = arith.mulf %a0, %a1 : f32",
                           "%kad = arith.addf %o0, %kac : f32"], "%kad"))
    e(f"linalg.copy ins(%dkacc : {t_CK}) outs(%dk_c : {t_qc})")

    # ---- dg ----
    # c1r = colsum(dAf .* kkT .* beta_row) ; r4/c4r = row/colsum(dPr .* P)
    _fill(e, "%c1r", t_C, "%f0")
    generic(e, ["(i, j) -> (i, j)", "(i, j) -> (i, j)", "(i, j) -> (i)",
                "(i, j) -> (j)"],
            ["parallel", "reduction"],
            [("%Gb", t_CC), ("%kkt", t_CC), ("%beta_c", t_bc)], [("%c1r", t_C)],
            lambda a, o: (["%cg1 = arith.mulf %a0, %a1 : f32",
                           "%cg2 = arith.mulf %cg1, %a2 : f32",
                           "%cg3 = arith.addf %o0, %cg2 : f32"], "%cg3"))
    _fill(e, "%r4", t_C, "%f0")
    generic(e, ["(i, j) -> (i, j)", "(i, j) -> (i, j)", "(i, j) -> (i)"],
            ["parallel", "reduction"],
            [("%dPr", t_CC), ("%Abuf", t_CC)], [("%r4", t_C)],
            lambda a, o: (["%r4p = arith.mulf %a0, %a1 : f32",
                           "%r4s = arith.addf %o0, %r4p : f32"], "%r4s"))
    _fill(e, "%c4r", t_C, "%f0")
    generic(e, ["(i, j) -> (i, j)", "(i, j) -> (i, j)", "(i, j) -> (j)"],
            ["parallel", "reduction"],
            [("%dPr", t_CC), ("%Abuf", t_CC)], [("%c4r", t_C)],
            lambda a, o: (["%c4p = arith.mulf %a0, %a1 : f32",
                           "%c4s = arith.addf %o0, %c4p : f32"], "%c4s"))
    # tA = beta.*ra - c1r + rb + r3 ; tB = r4 - c4r ; dgr = tA + tB
    # (dg's A-chain ROW term carries the beta_i row factor; dbeta's ra does not)
    generic(e, ["(i) -> (i)", "(i) -> (i)", "(i) -> (i)", "(i) -> (i)",
                "(i) -> (i)", "(i) -> (i)"],
            ["parallel"],
            [("%ra", t_C), ("%c1r", t_C), ("%rb", t_C), ("%r3v", t_C),
             ("%beta_c", t_bc)],
            [("%tA", t_C)],
            lambda a, o: (["%ta0 = arith.mulf %a0, %a4 : f32",
                           "%ta1 = arith.subf %ta0, %a1 : f32",
                           "%ta2 = arith.mulf %a2, %a4 : f32",
                           "%ta3 = arith.addf %ta1, %ta2 : f32",
                           "%ta4 = arith.addf %ta3, %a3 : f32"], "%ta4"))
    generic(e, ["(i) -> (i)", "(i) -> (i)", "(i) -> (i)"],
            ["parallel"],
            [("%r4", t_C), ("%c4r", t_C)], [("%tB", t_C)],
            lambda a, o: (["%tb1 = arith.subf %a0, %a1 : f32"], "%tb1"))
    generic(e, ["(i) -> (i)", "(i) -> (i)", "(i) -> (i)"],
            ["parallel"],
            [("%tA", t_C), ("%tB", t_C)], [("%dgr", t_C)],
            lambda a, o: (["%dgr1 = arith.addf %a0, %a1 : f32"], "%dgr1"))

    # s5 = 2^gcl * sum(dH .* Sn) ; dgp2 = prefix(rowB) + s5
    _fill(e, "%s5", t_C1, "%f0")
    generic(e, ["(kk, vv) -> (kk, vv)", "(kk, vv) -> (kk, vv)", "(kk, vv) -> (0)"],
            ["reduction", "reduction"],
            [("%dH", t_KV), ("%Sn", t_h0h)], [("%s5", t_C1)],
            lambda a, o: (["%s5p = arith.mulf %a0, %a1 : f32",
                           "%s5s = arith.addf %o0, %s5p : f32"], "%s5s"))
    e("%s5v = memref.load %s5[%c0] : " + t_C1)
    e("%s5x = arith.mulf %s5v, %dl : f32")
    e.block("scf.for %i6 = %c0 to %cC step %c1 iter_args(%pacc = %f0) -> (f32)")
    e("memref.store %pacc, %dgp2[%i6] : " + t_C)
    e("%rb_i = memref.load %rowB[%i6] : " + t_C)
    e("%pacc2 = arith.addf %pacc, %rb_i : f32")
    e("scf.yield %pacc2 : f32")
    e.end()
    generic(e, ["(i) -> (i)", "(i) -> (i)"],
            ["parallel"],
            [("%dgp2", t_C)], [("%dgp2", t_C)],
            lambda a, o: (["%dgpx = arith.addf %a0, %s5x : f32"], "%dgpx"))

    # dg = suffix(dgr) + dgp2  (the pieces are already natural-log cotangents:
    # every decay is exp2(gc_log2) == e^(gc_nat), so no RCP_LN2 rescale here)
    e.block("scf.for %i7 = %c0 to %cC step %c1 iter_args(%sacc = %f0) -> (f32)")
    e("%idx = arith.subi %cCm1, %i7 : index")
    e("%dr_i = memref.load %dgr[%idx] : " + t_C)
    e("%sacc2 = arith.addf %sacc, %dr_i : f32")
    e("%dp_i = memref.load %dgp2[%idx] : " + t_C)
    e("%dgt = arith.addf %sacc2, %dp_i : f32")
    e("memref.store %dgt, %dg_c[%idx] : " + t_bc)
    e("scf.yield %sacc2 : f32")
    e.end()

    # dH = dH .* 2^gcl + qg^T dO - w^T dvp
    _fill(e, "%tK1", t_KV, "%f0")
    generic(e, ["(kk, d, i) -> (i, kk)", "(kk, d, i) -> (i, d)",
                "(kk, d, i) -> (kk, d)"],
            ["parallel", "parallel", "reduction"],
            [("%qg", t_CK), ("%dO", t_CV)], [("%tK1", t_KV)],
            lambda a, o: (["%t1p = arith.mulf %a0, %a1 : f32",
                           "%t1s = arith.addf %o0, %t1p : f32"], "%t1s"))
    _fill(e, "%tK2", t_KV, "%f0")
    generic(e, ["(kk, d, i) -> (i, kk)", "(kk, d, i) -> (i, d)",
                "(kk, d, i) -> (kk, d)"],
            ["parallel", "parallel", "reduction"],
            [("%wp", t_CK), ("%dvp", t_CV)], [("%tK2", t_KV)],
            lambda a, o: (["%t2p = arith.mulf %a0, %a1 : f32",
                           "%t2s = arith.addf %o0, %t2p : f32"], "%t2s"))
    generic(e, ["(a, b) -> (a, b)", "(a, b) -> (a, b)", "(a, b) -> (a, b)",
                "(a, b) -> (a, b)"],
            ["parallel", "parallel"],
            [("%dH", t_KV), ("%tK1", t_KV), ("%tK2", t_KV)], [("%dH", t_KV)],
            lambda a, o: (["%dh1 = arith.mulf %a0, %dl : f32",
                           "%dh2 = arith.addf %dh1, %a1 : f32",
                           "%dh3 = arith.subf %dh2, %a2 : f32"], "%dh3"))
    e.end()  # pass 2 chunk loop

    # dh0[b,h] = dH
    e(f"%dh0h = memref.subview %dh0o[%b, %h, 0, 0] [1, 1, {K}, {V}] [1, 1, 1, 1] : {mrt((B, H, K, V))} to {t_h0h}")
    e(f"linalg.copy ins(%dH : {t_KV}) outs(%dh0h : {t_h0h})")
    for nm, ty in allocs:
        e(f"memref.dealloc %{nm} : {ty}")
    e.end()  # h loop
    e.end()  # b loop
    e(f"return %dq, %dk, %dv, %dbeta, %dg, %dh0o : "
      f"{tdq4}, {tdk4}, {tdv4}, {tdb3}, {tdg3}, {tdh0}")
    e.depth -= 1
    e("}")


# ----------------------------------------------------------------------------
# cli
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="1,128,2,16,16", help="B,T,H,K,V")
    ap.add_argument("--chunk", type=int, default=64, help="chunk size C")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit", default="fwd", choices=("fwd", "bwd", "both"),
                    help="emit forward kernel (default), backward kernel, or both")
    ap.add_argument("--npz", default=None, help="npz with q,k,v,beta,g,h0 (g log2-domain)")
    ap.add_argument("--out", default=None, help="output .mlir path (default stdout)")
    args = ap.parse_args()

    B, T, H, K, V = (int(x) for x in args.shape.split(","))
    refs = None
    do_bwd = args.emit in ("bwd", "both")
    if args.npz:
        data, refs = load_npz(args.npz, bwd=do_bwd)
        got = data["q"].shape
        assert got == (B, T, H, K), f"npz q shape {got} != --shape {(B, T, H, K)}"
        assert data["h0"].shape == (B, H, K, V), \
            f"npz h0 shape {data['h0'].shape} != {(B, H, K, V)}"
    else:
        data = make_data(B, T, H, K, V, args.seed)
        if do_bwd:
            data = make_bwd_data(data, args.seed)

    text = emit_module(data, B, T, H, K, V, args.chunk, refs=refs,
                       meta=f"seed={args.seed}", emit=args.emit)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
