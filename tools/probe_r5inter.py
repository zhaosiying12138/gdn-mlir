#!/usr/bin/env python3
"""G5-J round5 probe: compare the r5 chain's stage intermediates vs the fp64
twin AND vs the round-4 dhu chain (dh_all/dvp_all/dW_all/dbeta sensitivity)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/codebase/gdn-mlir/tools"))
import gdn_bwd_verify as gbv  # noqa: E402
from gdn_bwd_verify import (make_data, layout, bwd_twin,  # noqa
                            bf16_widen)

shape = sys.argv[1] if len(sys.argv) > 1 else "1,512,4,128,128"
r5root = sys.argv[2] if len(sys.argv) > 2 else \
    "evidence/G5/scratch/r5_T512"
r4root = sys.argv[3] if len(sys.argv) > 3 else \
    "evidence/G5/scratch/r4_final_T512"
B, T, H, K, V = (int(x) for x in shape.split(","))
data = make_data(B, T, H, K, V, 0)
lay = layout(data, B, T, H, K, V, 64)
md = lay["_meta"]
ref = bwd_twin(lay, with_intermediates=True)

host5, _ = gbv.run_chain(os.path.expanduser(f"~/codebase/gdn-mlir/{r5root}"),
                         lay)
# run the round-4 chain under its OWN stage contract
_R5_OUT = dict(gbv.OUT)
gbv.OUT = {"wy": ["T_all", "u_all", "w_all", "kkt_all", "P_all"],
           "fwdscan": ["h_all", "vnew_all"],
           "dhu": ["dh_all", "dv", "dvp_all", "dh0"],
           "dkb": ["dkB_sum", "rowB_c", "s5_c"],
           "dqkwg": ["dq", "dW_all", "dk_local", "r3_c", "r4_c", "c4r_c",
                     "dg_state"],
           "wyb": ["dk", "dbeta", "dg"]}
host4, _ = gbv.run_chain(os.path.expanduser(f"~/codebase/gdn-mlir/{r4root}"),
                         lay)
gbv.OUT = _R5_OUT

BH, NC, C = md["BH"], md["NC"], md["C"]


def un2(a, W):
    return a.reshape(BH, NC, *(W,))


for nm, W in (("dh_all", K), ("dvp_all", V), ("dW_all", K), ("dv", V)):
    got5 = host5[nm].reshape(BH, NC, -1, W).astype(np.float64) \
        if nm in host5 else None
    got4 = host4[nm].reshape(BH, NC, -1, W).astype(np.float64) \
        if nm in host4 else None
    line = f"{nm:8s}"
    if nm in ref["_inter"]:
        line += f" r5-vs-twin {np.abs(got5[:, :NC] - ref['_inter'][nm]).max():.3e}"
    if got4 is not None:
        line += f" r5-vs-r4 {np.abs(got5 - got4).max():.3e}"
    print(line)
print("dh0 r5-vs-twin", np.abs(
    host5["dh0"].reshape(BH, K, V).astype(np.float64) - ref["dh0"]).max())
for nm in ("dq", "dk", "dbeta", "dg"):
    W = K if nm in ("dq", "dk") else 1
    g5 = host5[nm].reshape(BH, NC, -1).astype(np.float64) if W == 1 else \
        host5[nm].reshape(BH, NC, C, K).astype(np.float64)
    g4 = host4[nm].reshape(BH, NC, -1).astype(np.float64) if W == 1 else \
        host4[nm].reshape(BH, NC, C, K).astype(np.float64)
    r = ref[nm]
    print(f"{nm:8s} r5-vs-twin {np.abs(g5 - r).max():.3e}   "
          f"r5-vs-r4 {np.abs(g5 - g4).max():.3e}")
