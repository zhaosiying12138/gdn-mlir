#!/usr/bin/env python3
"""G5-H probe: compare dkb's rowB_c / s5_c / dkB_sum against a numpy
recomputation from the GPU's OWN upstream outputs (dh_all/vnew_all/h_all),
isolating the dg failure from the rest of the chain."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/codebase/gdn-mlir/tools"))
import gdn_bwd_verify as BV

RCP_LN2 = 1.4426950408889634

gen_root = sys.argv[1]
shape = sys.argv[2]
B, T, H, K, V = (int(x) for x in shape.split(","))
C = 64
data = BV.make_data(B, T, H, K, V, 0)
lay = BV.layout(data, B, T, H, K, V, C)
md = lay["_meta"]
BH, NC = md["BH"], md["NC"]

host, times = BV.run_chain(gen_root, lay)

# widen GPU outputs
k = BV.bf16_widen(lay["k"]).reshape(BH, NC, C, K)
g = (lay["g_log2"].astype(np.float64) / RCP_LN2).reshape(BH, NC, C)
vn = host["vnew_all"].reshape(BH, NC, C, V).astype(np.float64)
h_all = host["h_all"].reshape(BH, NC, K, V).astype(np.float64)
dh_all = host["dh_all"].reshape(BH, NC, K, V).astype(np.float64)  # PRE-update per GPU conv
dht = BV.bf16_widen(lay["dht"]).reshape(BH, K, V)

gpu_dkB = host["dkB_sum"].reshape(BH, NC, C, K).astype(np.float64)
gpu_rb = host["rowB_c"].reshape(BH, NC, C).astype(np.float64)
gpu_s5 = host["s5_c"].reshape(BH, NC).astype(np.float64)

for bh in range(min(BH, 2)):
    for n in range(NC):
        gc = np.cumsum(g[bh, n])
        gcl = gc[-1]
        dH = dh_all[bh, n]          # GPU convention: pre-update dH of chunk n
        vng = vn[bh, n] * np.exp(gcl - gc)[:, None]
        dkB = vng @ dH.T
        rb = (dkB * k[bh, n]).sum(1)
        rb_g = (gpu_dkB[bh, n] * k[bh, n]).sum(1)
        s5 = np.exp(gcl) * (dH * h_all[bh, n]).sum()
        if bh == 0 and n < 3:
            dd = np.abs(rb - gpu_rb[bh, n])
            worst = np.argmax(dd)
            print(f"bh{n} n={n}: |dkB|={np.abs(dkB).max():.4e} "
                  f"gpu={np.abs(gpu_dkB[bh,n]).max():.4e} "
                  f"d={np.abs(dkB-gpu_dkB[bh,n]).max():.3e} | "
                  f"rb-vs-mine d={dd.max():.3e} at row {worst} | "
                  f"rb-vs-GPUdkB d={np.abs(rb_g-gpu_rb[bh,n]).max():.3e} | "
                  f"s5 d={np.abs(s5-gpu_s5[bh,n]):.3e}")
d_rb_g = max(np.abs((gpu_dkB[bh, n] * k[bh, n]).sum(1)
                    - gpu_rb[bh, n]).max()
             for bh in range(BH) for n in range(NC))
print(f"rowB_c vs (dkB_sum*k).rowsum over ALL chunks: max d = {d_rb_g:.3e}")
print("stage times:", times)

# count bad rows + validate identity on this chain
bad = 0
rows_bad = []
for bh in range(BH):
    for n in range(NC):
        rb_g = (gpu_dkB[bh, n] * k[bh, n]).sum(1)
        dd = np.abs(rb_g - gpu_rb[bh, n])
        idx = np.where(dd > 1e-4 * max(1e-9, np.abs(rb_g).max()))[0]
        if len(idx):
            bad += 1
            rows_bad.extend(idx.tolist())
print("chunks with bad rows:", bad, "/", BH * NC,
      " distinct bad rows:", sorted(set(rows_bad))[:20])
