#!/usr/bin/env python3
"""G5-H kslc dump probe: run the konly+DBG chain (dhu only matters), read
@g_kdbg = kslc [C, 2BK] at the moment slice-1's Bhat map reads it, and
compare both halves against every candidate k slice of chunk m."""
import os, re, sys
import numpy as np
import cuda.bindings.driver as cu

sys.path.insert(0, os.path.expanduser("~/codebase/gdn-mlir/tools"))
import gdn_bwd_verify as BV

ROOT = sys.argv[1]
B, T, H, K, V = (int(x) for x in sys.argv[2].split(","))
C = 64
BK = 32
data = BV.make_data(B, T, H, K, V, 0)
lay = BV.layout(data, B, T, H, K, V, C)
md = lay["_meta"]
BH, NC, M = md["BH"], md["NC"], md["BH"] * md["NC"]


def ck(r, w):
    err = r[0] if isinstance(r, tuple) else r
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{w}: {err}")
    return r


ck(cu.cuInit(0), "init")
dev = cu.cuDeviceGet(0)[1]
cu.cuCtxSetCurrent(cu.cuDevicePrimaryCtxRetain(dev)[1])
st = cu.CUstream(0)
host = {}
for nm in ("q", "k", "v", "do"):
    host[nm] = lay[nm]
host["beta"] = lay["beta"].astype(np.float32)
host["g"] = lay["g_log2"].astype(np.float32)
host["h0"] = lay["h0"]
host["dht"] = lay["dht"]
outs = {}
import json
man = json.load(open(f"{ROOT}/bwd_manifest.json"))
for stg in man["stages"]:
    d = os.path.join(ROOT, stg["dir"])
    if not os.path.exists(os.path.join(d, "kernel_smemfix.cubin")):
        continue
    mod = cu.cuModuleLoadData(
        open(os.path.join(d, "kernel_smemfix.cubin"), "rb").read())[1]
    ptx = open(os.path.join(d, "kernel_smemfix.ptx")).read()
    kname = re.search(r"\.visible \.entry (\w+)\(", ptx).group(1)
    fn = cu.cuModuleGetFunction(mod, kname.encode())[1]
    for sym, ty in stg["globals"].items():
        if sym in ("T_all", "u_all", "w_all", "kkt_all", "P_all", "h_all",
                   "vnew_all", "dh_all", "dv", "dvp_all", "dh0", "kdbg",
                   "dkB_sum", "rowB_c", "s5_c", "dq", "dW_all", "dk_local",
                   "r3_c", "r4_c", "c4r_c", "dg_state", "dk", "dbeta",
                   "dg"):
            continue
        if sym not in host:
            raise RuntimeError(f"unstaged {sym}")
        _, ptr, sz = cu.cuModuleGetGlobal(mod, sym.encode())
        arr = host[sym]
        cu.cuMemcpyHtoD(ptr, np.frombuffer(arr.tobytes(), dtype=np.uint8),
                        arr.nbytes)
    ck(cu.cuLaunchKernel(fn, stg["grid"][0], stg["grid"][1], 1,
                         stg["block"][0], 1, 1, 0, st, None, 0),
       f"launch {stg['name']}")
    ck(cu.cuStreamSynchronize(st), "sync")
    if stg["name"] == "dhu":
        _, ptr, _ = cu.cuModuleGetGlobal(mod, b"kdbg")
        buf = np.zeros(M * C * 2 * BK, dtype=np.uint16)
        cu.cuMemcpyDtoH(np.frombuffer(buf.tobytes(), dtype=np.uint8), ptr,
                        buf.nbytes)
        kdbg = buf.reshape(M, C * 2 * BK)
        break

k = BV.bf16_widen(lay["k"]).reshape(M, C, K)
print("kdbg nonzero cells:", int((kdbg != 0).sum()), "of", kdbg.size)
for m in (0, 1, 31, 32, 100):
    print(f"m={m} kdbg[:6]={kdbg[m, :6]} k[m,:6]={lay['k'][m, :6]}")
    got0 = kdbg[m, :C * BK].reshape(C, BK).astype(np.int64)
    got1 = kdbg[m, C * BK:].reshape(C, BK).astype(np.int64)
    best0 = max(range(4), key=lambda s: (got0 == lay["k"][m].reshape(C, K)[:, s*BK:(s+1)*BK].astype(np.int64)).sum())
    best1 = max(range(4), key=lambda s: (got1 == lay["k"][m].reshape(C, K)[:, s*BK:(s+1)*BK].astype(np.int64)).sum())
    m0 = (got0 == lay["k"][m].reshape(C, K)[:, best0*BK:(best0+1)*BK].astype(np.int64)).sum()
    m1 = (got1 == lay["k"][m].reshape(C, K)[:, best1*BK:(best1+1)*BK].astype(np.int64)).sum()
    print(f"m={m}: half0 best-match slice {best0} ({m0}/2048 cells), "
          f"half1 best-match slice {best1} ({m1}/2048 cells)")
