#!/usr/bin/env python3
"""gdn_full_rt.py -- G5-B runtime for full-op GDN forward kernels.

Build (gdn_full_gen -> scripts/run_l2.sh -> tools/gdn_smem_fix.py) and
launch the zero-arg cubins (public memref.global buffers, the G3/G4
noargs pattern): inputs are uploaded / outputs read back through
cuModuleGetGlobal; the kernel itself launches with NO params.

Used by tools/gdn_full_verify.py (numerics) and tools/gdn_full_bench.py
(timing).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

REPO = os.path.expanduser("~/codebase/gdn-mlir")
TOOLS = os.path.join(REPO, "tools")

ENTRY_RE = re.compile(r"\.visible \.entry (\w+)\(")
NTID_RE = re.compile(r"\.(maxntid|reqntid) (\d+), (\d+), (\d+)")


def _cache_key():
    """R20-fwd: hash of everything that changes the emitted payload --
    every GDN_*/G5* env knob in scope + the generator/smem_fix sources.
    The stale-cache hole (a cached kernel_smemfix.cubin accepted under a
    DIFFERENT knob set or an edited generator) has bitten twice (r3,
    r19 -- the F1WY T2048 'verify' that was a stale gen_fwd hit)."""
    h = hashlib.sha256()
    for k in sorted(os.environ):
        if k.startswith("GDN_") or k.startswith("G5"):
            h.update(f"{k}={os.environ[k]}\n".encode())
    for f in ("gdn_full_gen.py", "gdn_smem_fix.py"):
        p = os.path.join(TOOLS, f)
        if os.path.exists(p):
            h.update(f"\n#{f}#".encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


def build(kernel_args, out_dir, force=False):
    """Generate + compile one kernel dir.  kernel_args: argv list for
    gdn_full_gen.py (without the script path and --out).

    R20-fwd cache keying: builds stamp full_spec.json with
    gdn_cache_key = _cache_key(); a later build under a different
    knob-set/generator redirects to <out_dir>_k<hash8> instead of
    returning (or clobbering) the cached dir.  LEGACY dirs without the
    stamp are trusted as-is (the gen_fwd ships keep their slots)."""
    key = _cache_key()
    spec_path = os.path.join(out_dir, "full_spec.json")
    if not force and os.path.exists(os.path.join(out_dir, "kernel_smemfix.cubin")) \
            and os.path.exists(spec_path):
        try:
            spec = json.load(open(spec_path))
        except Exception:  # noqa: BLE001
            spec = {}
        if spec.get("gdn_cache_key") in (None, key):
            return out_dir
        out_dir = f"{out_dir}_k{key[:8]}"      # knob-keyed variant slot
        spec_path = os.path.join(out_dir, "full_spec.json")
        if os.path.exists(os.path.join(out_dir, "kernel_smemfix.cubin")) \
                and os.path.exists(spec_path):
            return out_dir
    os.makedirs(out_dir, exist_ok=True)
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "gdn_full_gen.py")]
                       + kernel_args + ["--out", out_dir],
                       capture_output=True, text=True, cwd=REPO, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"gen failed: {r.stdout[-800:]}\n{r.stderr[-800:]}")
    b = subprocess.run(["bash", os.path.join(REPO, "scripts/run_l2.sh"), out_dir],
                       capture_output=True, text=True, cwd=REPO, timeout=3600)
    if b.returncode != 0 or "run_l2] done" not in b.stdout:
        raise RuntimeError(f"run_l2 failed: {b.stdout[-1500:]}\n{b.stderr[-800:]}")
    s = subprocess.run([sys.executable, os.path.join(TOOLS, "gdn_smem_fix.py"),
                        out_dir], capture_output=True, text=True, cwd=REPO,
                       timeout=900)
    if s.returncode != 0:
        raise RuntimeError(f"smem_fix failed: {s.stdout[-800:]}\n{s.stderr[-800:]}")
    try:
        spec = json.load(open(spec_path))
        spec["gdn_cache_key"] = key
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2)
    except Exception:  # noqa: BLE001
        pass
    return out_dir


def _ck(res, what):
    err = res[0] if isinstance(res, tuple) else res
    import cuda.bindings.driver as cu
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what}: {err}")
    return res


class FullKernel:
    """One loaded zero-arg cubin with global-buffer upload/download."""

    def __init__(self, dirp):
        import cuda.bindings.driver as cu
        self.cu = cu
        self.dir = dirp
        self.spec = json.load(open(os.path.join(dirp, "full_spec.json")))
        ptx = open(os.path.join(dirp, "kernel_smemfix.ptx")).read()
        self.name = ENTRY_RE.search(ptx).group(1)
        nt = NTID_RE.search(ptx)
        self.block = int(nt.group(2)) if nt else self.spec["block"][0]
        self.grid = self.spec.get("grid") or self.spec.get("kernels", [{}])[0].get("grid", [1, 1, 1])
        raw = open(os.path.join(dirp, "kernel_smemfix.cubin"), "rb").read()
        _ck(cu.cuModuleLoadData(raw), "cuModuleLoadData")
        e, mod = cu.cuModuleLoadData(raw)
        _ck((e,), "cuModuleLoadData")
        self.mod = mod
        e, fn = cu.cuModuleGetFunction(mod, self.name.encode())
        _ck((e,), "cuModuleGetFunction")
        self.fn = fn
        # G5-D round 2 (mono): one cubin, several entries, shared globals
        self.fns = {}
        for k in self.spec.get("kernels", []):
            nm = k["name"]
            e, f2 = cu.cuModuleGetFunction(mod, nm.encode())
            if e != cu.CUresult.CUDA_SUCCESS:
                e, f2 = cu.cuModuleGetFunction(mod, (nm + "_kernel").encode())
            _ck((e,), "cuModuleGetFunction " + nm)
            self.fns[nm] = (f2, k["grid"], k["block"])
        self.globals = {}
        for p in self.spec["params"]:
            e, ptr, _sz = cu.cuModuleGetGlobal(mod, p["name"].encode())
            if e == cu.CUresult.CUDA_SUCCESS:
                self.globals[p["name"]] = ptr
            # globals unused by a truncated (--stop-after) kernel are DCE'd;
            # leave them absent and let stage_inputs skip them.

    def upload(self, name, arr_f32, dtype):
        """arr: numpy array (f32 values); dtype 'bf16' rounds to bf16,
        'f32' uploads as-is (C-contiguous flat)."""
        import numpy as np
        cu = self.cu
        a = np.ascontiguousarray(arr_f32.reshape(-1))
        if dtype == "bf16":
            u32 = a.astype(np.float32).view(np.uint32)
            rounded = (u32.astype(np.uint64) + 0x7FFF + ((u32 >> 16) & 1)) >> 16
            buf = rounded.astype(np.uint16).tobytes()
        else:
            buf = a.astype(np.float32).tobytes()
        _ck(cu.cuMemcpyHtoD(self.globals[name],
                            np.frombuffer(buf, dtype=np.uint8), len(buf)),
            f"HtoD {name}")

    def zero(self, name):
        import numpy as np
        cu = self.cu
        p = [q for q in self.spec["params"] if q["name"] == name][0]
        nbytes = p["elems"] * (2 if p["dtype"] == "bf16" else 4)
        _ck(cu.cuMemsetD8(self.globals[name], 0, nbytes), f"zero {name}")

    def readout(self, name, dtype, elems):
        import numpy as np
        cu = self.cu
        if dtype == "bf16":
            buf = np.zeros(elems, dtype=np.uint16)
            _ck(cu.cuMemcpyDtoH(buf, self.globals[name], elems * 2),
                f"DtoH {name}")
            f32 = (buf.astype(np.uint32) << 16).view(np.float32)
            return f32.astype(np.float64)
        buf = np.zeros(elems, dtype=np.float32)
        _ck(cu.cuMemcpyDtoH(buf, self.globals[name], elems * 4),
            f"DtoH {name}")
        return buf.astype(np.float64)

    def launch(self, stream, name=None):
        cu = self.cu
        if name is None:
            _ck(cu.cuLaunchKernel(self.fn, self.grid[0], self.grid[1], 1,
                                  self.block, 1, 1, 0, stream, None, 0),
                f"cuLaunchKernel {self.name}")
            return
        fn, grid, block = self.fns[name]
        _ck(cu.cuLaunchKernel(fn, grid[0], grid[1], 1,
                              block[0], 1, 1, 0, stream, None, 0),
            f"cuLaunchKernel {name}")

    def copy_global_to(self, dst_kern, dst_name, src_name, elems, dtype):
        """Device-to-device copy of a global buffer into another module's
        global (the staged path's inter-kernel handoff: each kernel is a
        separate cubin with its own globals)."""
        import numpy as np
        cu = self.cu
        nbytes = elems * (2 if dtype == "bf16" else 4)
        _ck(cu.cuMemcpyDtoD(dst_kern.globals[dst_name],
                            self.globals[src_name], nbytes),
            f"DtoD {self.name}.{src_name}->{dst_kern.name}.{dst_name}")


def staged_handoff(k1, k2, pairs):
    """Copy buffers between modules: pairs = ((dst_name, src_name), ...)."""
    for dst_name, src_name in pairs:
        p = [q for q in k2.spec["params"] if q["name"] == dst_name][0]
        k1.copy_global_to(k2, dst_name, src_name, p["elems"], p["dtype"])


def pad_time(x, T, Tp):
    import numpy as np
    pad = Tp - x.shape[1]
    if pad == 0:
        return x
    pw = [(0, 0)] * x.ndim
    pw[1] = (0, pad)
    return np.pad(x, pw)


def stage_inputs(kern, inputs, shape, extra=None):
    """Upload the shared G5-A input dict {q_bf16,k_bf16,v_bf16,beta,g,h0}
    (numpy fp32 arrays, [B,T,H,*] layouts) into a kernel's globals,
    zero-padding time to Tp.  extra: name->array for intermediates?"""
    import numpy as np
    B, T, H, K, V, C, NC, Tp = shape
    have = {"q": pad_time(inputs["q_bf16"], T, Tp),
            "k": pad_time(inputs["k_bf16"], T, Tp),
            "v": pad_time(inputs["v_bf16"], T, Tp),
            "beta": pad_time(inputs["beta"], T, Tp),
            "g": pad_time(inputs["g"], T, Tp),
            "h0": inputs["h0"]}
    for p in kern.spec["params"]:
        n = p["name"]
        if n in have and n in kern.globals:
            kern.upload(n, have[n], p["dtype"])
        elif n in kern.globals:
            kern.zero(n)


def smi_snapshot():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10)
        u, m, c = r.stdout.strip().split("\n")[0].split(",")
        return {"util_pct": int(u.strip()), "mem_used_mib": int(m.strip()),
                "sm_clock_mhz": int(c.strip())}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
