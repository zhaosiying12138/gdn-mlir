#!/usr/bin/env python3
"""G5-H A/B probe: run two chains on identical inputs, diff dhu's outputs
per chunk to localize the prefetch corruption."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/codebase/gdn-mlir/tools"))
import gdn_bwd_verify as BV

root_a, root_b, shape = sys.argv[1], sys.argv[2], sys.argv[3]
B, T, H, K, V = (int(x) for x in shape.split(","))
C = 64
data = BV.make_data(B, T, H, K, V, 0)
lay = BV.layout(data, B, T, H, K, V, C)
md = lay["_meta"]
BH, NC = md["BH"], md["NC"]

ha, _ = BV.run_chain(root_a, lay)
hb, _ = BV.run_chain(root_b, lay)

for nm in ("dh_all", "dvp_all", "dv"):
    a = ha[nm].reshape(BH, NC, -1).astype(np.float64)
    b = hb[nm].reshape(BH, NC, -1).astype(np.float64)
    d = np.abs(a - b).reshape(BH, NC, -1).max(axis=(0, 2))
    first = int(np.argmax(d > 1e-5)) if (d > 1e-5).any() else -1
    print(f"{nm}: max {d.max():.3e}  first bad chunk {first}  "
          f"(per-chunk max, first 8: {np.array2string(d[:8], precision=1)})")
# dv is [M, C*V] -> per-chunk via reshape(BH, NC, C*V); dh_all [M, K*V]
