#!/usr/bin/env python3
"""gdn_verify.py -- G4 numeric-correctness harness for the 20 GDN L2 configs.

Per config: regenerate (--driver noargs, fixed --seed) -> compile
(scripts/run_l2.sh + tools/gdn_smem_fix.py) -> launch on the GPU in an
isolated subprocess -> read back the FULL output -> compare element-wise
against an exact numpy fp64 reference twin of the payload's math computed
from the SAME embedded inputs (gdn_l2_gen.noargs_val(seed)).

Reference twins (mirror of the payload math in tools/gdn_l2_gen.py):
  kkt : C[r,c] = (c<=r) ? sum_k k[r,k]*k[c,k] * 2^(-0.0625*(r-c)) : 0
        (tril(k @ k^T) with exp2 decay; payload: mma f32 accum or K-tiled
        linalg f32 accum, then f32 decay-mul, truncf to io dtype)
  o   : O = (q @ h) + (A @ v)   (two chained GEMMs, same accumulation note)
  ref : 16x16 @ 16x8 f16 atom (in-tree sm80 rewrite / ldmatrix variant)

Thresholds (fla-derived contract, stage outputs graded at o/ht level):
  bf16/f16-io configs vs fp64 reference: max-abs-diff <= 5e-3
  f32-io configs:                         max-abs-diff <= 1e-3
(full-op fwd/bwd fla-tolerance validation happens in G5 with the complete
kernel; these stage-level bars are the G4 gate.)

tma=on config: the kernel moves data only through host-encoded CUtensorMaps
(its cubin has no @gdn_out global), so the worker stages the k input into a
device buffer, encodes descA/descB/descS exactly per the generator's
tensormap_spec.json via cuda-bindings cuTensorMapEncodeTiled, passes the
device map pointers through the three .u64 .ptr kernel params, and reads the
output back from the descS target buffer.

Per-config subprocess isolation with wall timeouts (a hanging kernel only
kills its own worker). Shared-GPU: opportunistic window, tiny buffers.

Usage (conda env gdn):
  python tools/gdn_verify.py [--gen-root evidence/G4/gen_verify]
                             [--out evidence/G4/verify_results.json]
                             [--seed 0] [--only NAME] [--timeout 120]
                             [--skip-build]  (reuse existing gen dirs)
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time

REPO = os.path.expanduser("~/codebase/gdn-mlir")
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)

import gdn_l2_gen  # noqa: E402  (payload math + noargs_val + builtin matrix)
import gdn_time    # noqa: E402  (asmobj/PTX param parsing, grid/out specs)

THRESH_BF16 = 5e-3   # bf16/f16-io configs (fla stage-output contract)
THRESH_F32 = 1e-3    # f32-io configs


def threshold_for(cfg):
    return THRESH_F32 if cfg["dtype"] == "f32" else THRESH_BF16


# ---------------------------------------------------------------------------
# exact numpy fp64 reference twins (same embedded inputs as the payload)
# ---------------------------------------------------------------------------

def _vals(rows, cols, seed):
    return [[gdn_l2_gen.noargs_val(r, c, seed) for c in range(cols)]
            for r in range(rows)]


def ref_output(cfg):
    """-> (fp64 ndarray of the expected @gdn_out, io dtype string)."""
    import numpy as np
    seed = int(cfg.get("seed", 0))
    stage = cfg["stage"]
    if stage == "kkt":
        K = cfg["dims"]["K"]
        k = np.array(_vals(64, K, seed), dtype=np.float64)
        S = k @ k.T
        d = np.arange(64)
        decay = np.exp2((d[:, None] - d[None, :]) * -0.0625)
        C = np.tril(S * decay)          # strict-tril incl. diagonal (sle c<=r)
        return C, cfg["dtype"]
    if stage == "o":
        K, V = cfg["dims"]["K"], cfg["dims"]["V"]
        q = np.array(_vals(64, K, seed), dtype=np.float64)
        h = np.array(_vals(K, V, seed), dtype=np.float64)
        a = np.array(_vals(64, K, seed), dtype=np.float64)
        v = np.array(_vals(K, V, seed), dtype=np.float64)
        # NOTE: q and a use the same value function -> identical matrices by
        # construction; that is what the payload embeds too (same globals
        # recipe), so the twin matches the kernel bit-for-bit in inputs.
        return q @ h + a @ v, cfg["dtype"]
    if stage in ("ref", "ref_ldmatrix"):
        lhs = np.array(_vals(16, 16, seed), dtype=np.float64)
        rhs = np.array(_vals(16, 8, seed), dtype=np.float64)
        return lhs @ rhs, "f32" if stage == "ref_ldmatrix" else "f16"
    raise ValueError(stage)


def io_bytes_to_f64(buf: bytes, dt: str):
    """Raw @gdn_out bytes -> fp64 array (bf16/f16 widened exactly)."""
    import numpy as np
    if dt == "bf16":
        u16 = np.frombuffer(buf, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.astype(np.float64)
    if dt == "f16":
        return np.frombuffer(buf, dtype=np.float16).astype(np.float64)
    if dt == "f32":
        return np.frombuffer(buf, dtype=np.float32).astype(np.float64)
    raise ValueError(dt)


def f64_to_io_bytes(arr, dt: str) -> bytes:
    """fp64 values (already io-exact) -> raw io-dtype bytes for HtoD."""
    import numpy as np
    a = np.ascontiguousarray(arr)
    if dt == "bf16":
        # values are exact multiples of 2^-7 -> bf16 = top half of f32 bits
        f = a.astype(np.float32)
        assert np.all(f.view(np.uint32) & 0xFFFF == 0), "not bf16-exact"
        return (f.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    if dt == "f16":
        f = a.astype(np.float16)
        assert np.all(f.astype(np.float64) == a), "not f16-exact"
        return f.tobytes()
    return a.astype(np.float32).tobytes()


# ---------------------------------------------------------------------------
# worker: launch ONE config on the GPU, read back, compare
# ---------------------------------------------------------------------------

def run_worker(dirp, timeout_note=""):
    import cuda.bindings.driver as cu
    import numpy as np

    def ck(res, what):
        err = res[0] if isinstance(res, tuple) else res
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{what}: {err}")
        return res

    cfg = json.load(open(os.path.join(dirp, "config.json")))
    seed = int(cfg.get("seed", 0))
    ptx = open(os.path.join(dirp, "kernel_smemfix.ptx")).read()
    kname = gdn_time.ENTRY_RE.search(ptx).group(1)
    nt = gdn_time.NTID_RE.search(ptx)
    block = int(nt.group(2)) if nt else 32 * cfg["warps"]
    gx, gy = gdn_time.grid_from_config(cfg)
    stream = cu.CUstream(0)
    ck(cu.cuInit(0), "cuInit")
    e, dev = cu.cuDeviceGet(0)
    ck((e,), "cuDeviceGet")
    e, ctx = cu.cuDevicePrimaryCtxRetain(dev)
    ck((e,), "cuDevicePrimaryCtxRetain")
    ck(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    res = {"name": cfg["name"], "grid": [gx, gy, 1], "block": [block, 1, 1],
           "kernel": kname, "seed": seed}

    raw = open(os.path.join(dirp, "kernel_smemfix.cubin"), "rb").read()
    e, mod = cu.cuModuleLoadData(raw)
    ck((e,), "cuModuleLoadData")
    e, fn = cu.cuModuleGetFunction(mod, kname.encode())
    ck((e,), "cuModuleGetFunction")

    # ---- kernel parameters (constants already baked in by gdn_smem_fix) --
    params = gdn_time.parse_ptx_params(dirp, kname)
    values = gdn_time.parse_launch_values(dirp)
    kparams = None
    keepalive = []
    tma = cfg.get("tma") == "on"
    dev_out = None
    if params:
        kparams = []
        ptr_pos = []
        for p in params:
            v = values[p["idx"]] if p["idx"] < len(values) else None
            if v is None or v["kind"] == "ptr":
                ptr_pos.append(len(kparams))
                kparams.append(0)
                keepalive.append(None)
                continue
            if v["kind"] in ("int", "bool"):
                if p["size"] == 1:
                    o = ctypes.c_uint8(v["value"] & 0xFF)
                elif p["size"] == 2:
                    o = ctypes.c_uint16(v["value"] & 0xFFFF)
                elif p["size"] == 4:
                    o = ctypes.c_uint32(v["value"] & 0xFFFFFFFF)
                else:
                    o = ctypes.c_uint64(v["value"])
            elif v["kind"] == "vec":
                vals = (v["value"] + [0.0] * 4)[:4]
                o = (ctypes.c_float * 4)(*vals)
            elif v["kind"] == "float":
                o = ctypes.c_float(v["value"])
            else:
                raise RuntimeError(f"unhandled param {p} = {v}")
            keepalive.append(o)
            kparams.append(ctypes.cast(ctypes.byref(o),
                                       ctypes.c_void_p).value)
        if kparams is not None and not tma:
            kparams = np.array(kparams, dtype=np.uint64)

    # ---- tma host wiring: encode CUtensorMaps per tensormap_spec.json ----
    if tma:
        spec = json.load(open(os.path.join(dirp, "tensormap_spec.json")))
        K = cfg["dims"]["K"]
        kt = cfg["dtype"]
        kb = f64_to_io_bytes(
            np.array(_vals(64, K, seed), dtype=np.float64), kt)
        e, d_k = cu.cuMemAlloc(len(kb))
        ck((e,), "cuMemAlloc(k)")
        ck(cu.cuMemcpyHtoD(d_k, np.frombuffer(kb, dtype=np.uint8),
                           len(kb)), "HtoD(k)")
        ob = f64_to_io_bytes(np.zeros((64, 64)), kt)
        e, dev_out = cu.cuMemAlloc(len(ob))
        ck((e,), "cuMemAlloc(out)")
        ck(cu.cuMemcpyHtoD(dev_out, np.frombuffer(ob, dtype=np.uint8),
                           len(ob)), "HtoD(out-zero)")

        def enc2(dname, desc, base):
            # cuda-bindings pythonic API (validated on this build): typed
            # list elements, int globalAddress, returns (CUresult, CUtensorMap)
            r = cu.cuTensorMapEncodeTiled(
                cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                desc["rank"], int(base),
                [cu.cuuint64_t(x) for x in desc["global_dim"]],
                [cu.cuuint64_t(x) for x in desc["global_strides_bytes"]],
                [cu.cuuint32_t(x) for x in desc["box_dim"]],
                [cu.cuuint32_t(x) for x in ([1] * desc["rank"])],
                cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
                cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
                cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
                cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
            err = r[0] if isinstance(r, tuple) else r
            if err != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"encode {dname}: {err}")
            return r[1] if isinstance(r, tuple) else r

        maps = {}
        for dname, desc in spec["descriptors"].items():
            base = d_k if dname in ("descA", "descB") else dev_out
            maps[dname] = enc2(dname, desc, base)

        # upload each map to device memory; each kernel param is a .u64 .ptr
        # whose VALUE is the device pointer to the 128-byte map
        for pos, dname in zip(ptr_pos, ("descA", "descB", "descS")):
            if dname not in maps:
                raise RuntimeError(f"ptr param #{pos} has no descriptor")
            raw_map = ctypes.string_at(maps[dname].getPtr(), 128)
            e, d_map = cu.cuMemAlloc(256)
            ck((e,), "cuMemAlloc(map)")
            ck(cu.cuMemcpyHtoD(d_map, np.frombuffer(raw_map, dtype=np.uint8),
                               128), "HtoD(map)")
            o = ctypes.c_uint64(int(d_map))
            keepalive[pos] = o
            kparams[pos] = ctypes.cast(ctypes.byref(o),
                                       ctypes.c_void_p).value
        if len(ptr_pos) != 3:
            raise RuntimeError(f"expected 3 tensormap ptrs, got {ptr_pos}")
        kparams = np.array(kparams, dtype=np.uint64)

    # ---- output global (non-tma configs) ----------------------------------
    gout = None
    out_bytes = None
    if not tma:
        orows, ocols, esz = gdn_time.out_spec(cfg)
        out_bytes = orows * ocols * esz
        e, gout, _gsz = cu.cuModuleGetGlobal(mod, b"gdn_out")
        ck((e,), "cuModuleGetGlobal(gdn_out)")

    def launch():
        ck(cu.cuLaunchKernel(fn, gx, gy, 1, block, 1, 1, 0, stream, kparams,
                             0), "cuLaunchKernel")

    def read_out():
        if tma:
            buf = np.zeros(64 * 64 * 2, dtype=np.uint8)
            ck(cu.cuMemcpyDtoH(buf, dev_out, 64 * 64 * 2), "DtoH(out)")
            return buf.tobytes()
        buf = np.zeros(out_bytes, dtype=np.uint8)
        ck(cu.cuMemcpyDtoH(buf, gout, out_bytes), "DtoH(out)")
        return buf.tobytes()

    def zero_out():
        if tma:
            ck(cu.cuMemsetD8(dev_out, 0, 64 * 64 * 2), "memset(out)")
        else:
            ck(cu.cuMemsetD8(gout, 0, out_bytes), "memset(gdn_out)")

    # ---- run + compare -----------------------------------------------------
    try:
        zero_out()
        launch()
        ck(cu.cuStreamSynchronize(stream), "sync after launch 1")
    except Exception as e:  # noqa: BLE001
        res["status"] = "LAUNCH_FAILED"
        res["error"] = str(e)
        return res
    got = io_bytes_to_f64(read_out(), cfg["dtype"] if cfg["stage"] != "ref_ldmatrix" else "f32")
    ref, refdt = ref_output(cfg)
    thr = threshold_for(cfg)
    diff = np.abs(got.reshape(ref.shape) - ref)
    res["max_abs_diff"] = float(diff.max())
    res["threshold"] = thr
    res["n_over_threshold"] = int((diff > thr).sum())
    res["out_total_elems"] = int(ref.size)
    res["nonzero_elems"] = int((got != 0).sum())
    res["ref_nonzero_elems"] = int((ref != 0).sum())

    # determinism: zero + relaunch must reproduce the exact same buffer
    zero_out()
    launch()
    ck(cu.cuStreamSynchronize(stream), "sync after launch 2")
    got2 = io_bytes_to_f64(read_out(), cfg["dtype"] if cfg["stage"] != "ref_ldmatrix" else "f32")
    res["deterministic"] = bool(np.array_equal(got, got2))

    res["status"] = ("PASS" if (res["n_over_threshold"] == 0
                                and res["deterministic"]) else "FAIL")
    return res


# ---------------------------------------------------------------------------
# parent: regenerate + compile + spawn workers
# ---------------------------------------------------------------------------

def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build_config(name, cfg_json, out_dir, seed):
    env = dict(os.environ)
    r = subprocess.run(
        [sys.executable, os.path.join(TOOLS, "gdn_l2_gen.py"),
         "--json", cfg_json, "--driver", "noargs", "--seed", str(seed),
         "--out", out_dir],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"gen failed: {r.stdout}\n{r.stderr}")
    b = subprocess.run(["bash", os.path.join(REPO, "scripts/run_l2.sh"),
                        out_dir], capture_output=True, text=True, cwd=REPO,
                       timeout=1800)
    if b.returncode != 0:
        raise RuntimeError(f"run_l2 failed: {b.stdout[-2000:]}\n"
                           f"{b.stderr[-2000:]}")
    s = subprocess.run([sys.executable, os.path.join(TOOLS,
                                                     "gdn_smem_fix.py"),
                        out_dir], capture_output=True, text=True, cwd=REPO,
                       timeout=600)
    if s.returncode != 0:
        raise RuntimeError(f"smem_fix failed: {s.stdout}\n{s.stderr}")
    return s.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", default=os.path.join(
        REPO, "evidence/G4/gen_verify"))
    ap.add_argument("--cfg-root", default=os.path.join(
        REPO, "evidence/G2/gen"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "evidence/G4/verify_results.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-dir", default=None)
    ap.add_argument("--skip-build", action="store_true",
                    help="reuse existing gen dirs (no regen/compile)")
    args = ap.parse_args()

    if args.worker:
        try:
            r = run_worker(args.worker_dir)
        except Exception as e:  # noqa: BLE001
            r = {"name": os.path.basename(args.worker_dir),
                 "status": "WORKER_CRASH",
                 "error": f"{type(e).__name__}: {e}"}
        print("RESULT:" + json.dumps(r))
        return 0

    names = [args.only] if args.only else [
        c["name"] for c in gdn_l2_gen.builtin_configs()]

    os.makedirs(args.gen_root, exist_ok=True)

    # generator determinism: same seed -> identical payload (twice), and
    # different seeds -> different data (checked on the first config)
    det = {"same_seed_identical": None, "seed_changes_data": None}
    probe = names[0]
    d1 = os.path.join(args.gen_root, "_det1")
    d2 = os.path.join(args.gen_root, "_det2")
    d3 = os.path.join(args.gen_root, "_det3")
    try:
        for d, seed in ((d1, args.seed), (d2, args.seed),
                        (d3, args.seed + 7)):
            subprocess.run([sys.executable, os.path.join(TOOLS,
                                                         "gdn_l2_gen.py"),
                            "--json", os.path.join(args.cfg_root, probe,
                                                   "config.json"),
                            "--driver", "noargs", "--seed", str(seed),
                            "--out", d], capture_output=True, text=True,
                           cwd=REPO, check=True)
        det["same_seed_identical"] = (
            sha256(os.path.join(d1, "payload.mlir"))
            == sha256(os.path.join(d2, "payload.mlir")))
        det["seed_changes_data"] = (
            sha256(os.path.join(d1, "payload.mlir"))
            != sha256(os.path.join(d3, "payload.mlir")))
        import shutil
        for d in (d1, d2, d3):
            shutil.rmtree(d, ignore_errors=True)
    except Exception as e:  # noqa: BLE001
        det["error"] = str(e)

    results = {}
    order = []
    for n in names:
        cfg_json = os.path.join(args.cfg_root, n, "config.json")
        out_dir = os.path.join(args.gen_root, n)
        entry = {"name": n}
        order.append(n)
        print(f"[gdn_verify] {n} ...", flush=True)
        t0 = time.time()
        if not args.skip_build:
            try:
                entry["build"] = build_config(n, cfg_json, out_dir, args.seed)
            except Exception as e:  # noqa: BLE001
                entry.update({"status": "BUILD_FAILED", "error": str(e)[-800:]})
                results[n] = entry
                print(f"    -> BUILD_FAILED: {entry['error'][-200:]}",
                      flush=True)
                continue
        try:
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--worker",
                 "--worker-dir", out_dir],
                capture_output=True, text=True, timeout=args.timeout,
                cwd=REPO)
            line = [l for l in p.stdout.splitlines()
                    if l.startswith("RESULT:")]
            r = (json.loads(line[-1][7:]) if line else
                 {"status": "WORKER_CRASH",
                  "error": ((p.stderr or p.stdout) or "")[-600:]})
        except subprocess.TimeoutExpired:
            r = {"status": "TIMEOUT",
                 "error": f"worker exceeded {args.timeout}s (kernel hang?)"}
        entry.update(r)
        entry["wall_s"] = round(time.time() - t0, 1)
        results[n] = entry
        print(f"    -> {entry.get('status')} max_diff="
              f"{entry.get('max_abs_diff')} nonzero="
              f"{entry.get('nonzero_elems')}/{entry.get('out_total_elems')}",
              flush=True)

    doc = {
        "description": "G4 numeric verification: 20 configs vs exact fp64 "
                       "reference twins on the GPU (same embedded inputs)",
        "thresholds": {"bf16/f16-io": THRESH_BF16, "f32-io": THRESH_F32},
        "contract": "stage outputs graded at o/ht level: 5e-3 bf16-io, "
                    "1e-3 f32-io (fla-derived; full-op validation in G5)",
        "seed": args.seed,
        "generator_determinism": det,
        "shared_gpu": True,
        "results_order": order,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    ok = sum(1 for r in results.values() if r.get("status") == "PASS")
    print(f"[gdn_verify] wrote {args.out}: {ok}/{len(results)} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
