#!/usr/bin/env python3
"""G5-J round5 probe (v2, bh0-only): which dbeta/dv term carries the
r5-vs-r4 movement.  The twin's _inter snapshot stores bh==0 ONLY."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/codebase/gdn-mlir/tools"))
import gdn_bwd_verify as gbv  # noqa: E402
from gdn_bwd_verify import make_data, layout, bwd_twin  # noqa: E402

B, T, H, K, V = 1, 512, 4, 128, 128
data = make_data(B, T, H, K, V, 0)
lay = layout(data, B, T, H, K, V, 64)
md = lay["_meta"]
BH, NC, C = md["BH"], md["NC"], md["C"]
ref = bwd_twin(lay, with_intermediates=True)

host5, _ = gbv.run_chain(
    os.path.expanduser("~/codebase/gdn-mlir/evidence/G5/scratch/r5_T512"),
    lay)
_R5 = dict(gbv.OUT)
gbv.OUT = {"wy": ["T_all", "u_all", "w_all", "kkt_all", "P_all"],
           "fwdscan": ["h_all", "vnew_all"],
           "dhu": ["dh_all", "dv", "dvp_all", "dh0"],
           "dkb": ["dkB_sum", "rowB_c", "s5_c"],
           "dqkwg": ["dq", "dW_all", "dk_local", "r3_c", "r4_c", "c4r_c",
                     "dg_state"],
           "wyb": ["dk", "dbeta", "dg"]}
host4, _ = gbv.run_chain(
    os.path.expanduser("~/codebase/gdn-mlir/evidence/G5/scratch/r4_final_T512"),
    lay)
gbv.OUT = _R5

wid = lambda u16, W_: gbv.bf16_widen(u16).reshape(BH, NC, C, W_)  # noqa
v = wid(lay["v"], V)
k = wid(lay["k"], K)
beta = lay["beta"].astype(np.float64).reshape(BH, NC, C)
gl2 = lay["g_log2"].astype(np.float64).reshape(BH, NC, C)
m_str = np.tril(np.ones((C, C)), -1)
D = np.exp2(gl2[:, :, :, None] - gl2[:, :, None, :])


def terms(dW, dvp, Tm):
    Kbg = beta[..., None] * k * np.exp2(gl2)[..., None]
    Vb = beta[..., None] * v
    G = np.einsum("bncv,bnjv->bncj", dvp, Vb) \
        + np.einsum("bnck,bnjk->bncj", dW, Kbg)
    TT = np.swapaxes(Tm, -1, -2)
    dAf = -(np.einsum("bnij,bnjk->bnik", TT,
                      np.einsum("bnjk,bnkl->bnjl", G * m_str, TT))) * D * m_str
    kkT = np.einsum("bnci,bnji->bncj", k, k)
    TW = np.einsum("bnij,bnjc->bnic", TT, dW)
    tv = np.einsum("bnij,bnjc->bnic", TT, dvp)
    return (dAf * kkT).sum(-1), (TW * (k * np.exp2(gl2)[..., None])).sum(-1), \
        (tv * v).sum(-1)


dWt = ref["_inter"]["dW_all"][None]
dvpt = ref["_inter"]["dvp_all"][None]
Tt = ref["_inter"]["T_all"][None]
tr = terms(dWt, dvpt, Tt)
print("twin self-consistency bh0:",
      np.abs((tr[0] + tr[1] + tr[2])[0] - ref["dbeta"][0]).max())
dvp4 = host4["dvp_all"].reshape(BH, NC, C, V).astype(np.float64)
for nm, host in (("r4", host4), ("r5", host5)):
    dW = host["dW_all"].reshape(BH, NC, C, K).astype(np.float64)
    dvp = host["dvp_all"].reshape(BH, NC, C, V).astype(np.float64)
    Tm = host["T_all"].reshape(BH, NC, C, C).astype(np.float64)
    t = terms(dW, dvp, Tm)
    print(f"[{nm}] bh0 term-vs-twin: "
          f"af {np.abs(t[0][0] - tr[0][0]).max():.3e}  "
          f"tw {np.abs(t[1][0] - tr[1][0]).max():.3e}  "
          f"tv {np.abs(t[2][0] - tr[2][0]).max():.3e}")
    print(f"[{nm}] bh0 dvp-vs-twin {np.abs(dvp[0] - dvpt[0]).max():.3e}"
          f"  dW {np.abs(dW[0] - dWt[0]).max():.3e}"
          f"  T {np.abs(Tm[0] - Tt[0]).max():.3e}"
          f"  dvp r5-vs-r4 {np.abs(dvp[0] - dvp4[0]).max():.3e}")
    dv = host["dv"].reshape(BH, NC, C, V).astype(np.float64)
    dv_own = np.einsum("bnij,bnjc->bnic", np.swapaxes(Tm, -1, -2), dvp) \
        * beta[..., None]
    print(f"[{nm}] bh0 dv-vs-own-recompute {np.abs(dv[0] - dv_own[0]).max():.3e}"
          f"   dv-vs-twin {np.abs(dv[0] - ref['dv'][0]).max():.3e}")
    print(f"[{nm}] bh0 dv OWN-recompute-vs-twin "
          f"{np.abs(dv_own[0] - ref['dv'][0]).max():.3e}")
