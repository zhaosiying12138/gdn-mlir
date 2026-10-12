#!/usr/bin/env python3
"""r9: launch ONE bwd-chain kernel N times (bench-style, zero globals) for ncu."""
import os, re, sys
import cuda.bindings.driver as cu

d = sys.argv[1]           # kernel config dir (has kernel_smemfix.cubin)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
gx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
bx = int(sys.argv[4]) if len(sys.argv) > 4 else 0
cu.cuInit(0)
dev = cu.cuDeviceGet(0)[1]
cu.cuCtxSetCurrent(cu.cuDevicePrimaryCtxRetain(dev)[1])
stream = cu.CUstream(0)
mod = cu.cuModuleLoadData(open(os.path.join(d, "kernel_smemfix.cubin"), "rb").read())[1]
ptx = open(os.path.join(d, "kernel_smemfix.ptx")).read()
kname = re.search(r"\.visible \.entry (\w+)\(", ptx).group(1)
fn = cu.cuModuleGetFunction(mod, kname.encode())[1]
if not gx:
    import json
    cfg = json.load(open(os.path.join(d, "config.json")))
    gx = cfg["grid"][0]; bx = cfg["threads"]
for _ in range(n):
    r = cu.cuLaunchKernel(fn, gx, 1, 1, bx, 1, 1, 0, stream, None, 0)
    err = r[0] if isinstance(r, tuple) else r
    assert err == cu.CUresult.CUDA_SUCCESS, err
cu.cuStreamSynchronize(stream)
print("done", kname, gx, bx)
