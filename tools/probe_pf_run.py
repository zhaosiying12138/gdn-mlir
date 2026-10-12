#!/usr/bin/env python3
"""G5-H micro-probe runner: stage A, launch pfp, read B, compare cols 32..64
of each 128-wide row (the half that went through the offset cp.async)."""
import os, re, sys
import numpy as np
import cuda.bindings.driver as cu

D = os.path.expanduser("~/codebase/gdn-mlir/evidence/G5/scratch/r4_pfprobe/pfp")


def ck(r, w):
    err = r[0] if isinstance(r, tuple) else r
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{w}: {err}")
    return r


ck(cu.cuInit(0), "init")
dev = cu.cuDeviceGet(0)[1]
cu.cuCtxSetCurrent(cu.cuDevicePrimaryCtxRetain(dev)[1])
mod = cu.cuModuleLoadData(open(f"{D}/kernel_smemfix.cubin", "rb").read())[1]
ptx = open(f"{D}/kernel_smemfix.ptx").read()
kname = re.search(r"\.visible \.entry (\w+)\(", ptx).group(1)
fn = cu.cuModuleGetFunction(mod, kname.encode())[1]

rng = np.random.default_rng(7)
A = (rng.standard_normal(8192) * 3).astype(np.float32)
A16 = GEN_none = None
u = np.ascontiguousarray(A, dtype=np.float32).view(np.uint32)
u = u + ((u >> 16) & 1) + 0x7FFF
A16 = (u >> 16).astype(np.uint16)
B16 = np.zeros(8192, dtype=np.uint16)

_, pa, _ = cu.cuModuleGetGlobal(mod, b"A")
_, pb, _ = cu.cuModuleGetGlobal(mod, b"B")
ck(cu.cuMemcpyHtoD(pa, np.frombuffer(A16.tobytes(), dtype=np.uint8),
                   A16.nbytes), "h2d A")
ck(cu.cuMemcpyHtoD(pb, np.frombuffer(B16.tobytes(), dtype=np.uint8),
                   B16.nbytes), "h2d B")
st = cu.CUstream(0)
ck(cu.cuLaunchKernel(fn, 1, 1, 1, 256, 1, 1, 0, st, None, 0), "launch")
ck(cu.cuStreamSynchronize(st), "sync")
out = np.zeros(8192, dtype=np.uint16)
ck(cu.cuMemcpyDtoH(np.frombuffer(out.tobytes(), dtype=np.uint8), pb,
                   out.nbytes), "d2h")

exp = A16.reshape(64, 128)[:, 32:64].ravel()
got = out.reshape(64, 128)[:, 32:64].ravel()
bad = (exp != got).sum()
print(f"[probe_pf] mismatched cols-32..64 cells: {bad}/2048 "
      f"(first bad idx {int(np.argmax(exp != got)) if bad else -1})")
if bad:
    i = int(np.argmax(exp != got))
    print(f"  exp[{i}:{i+4}] = {exp[i:i+4]}  got = {got[i:i+4]}")
    print(f"  exp row {i//32} col {i%32}; got-as-row pattern: "
          f"{(np.argmax(out.reshape(64,128)[:,32:64] != exp.reshape(64,32), axis=1)>0)[:8]}")
