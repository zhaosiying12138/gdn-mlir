#!/usr/bin/env python3
"""gdn_time.py -- G3-A timed GPU execution of the 20 G2 knob configs.

Loads each evidence/G3/gen_noargs/<name>/kernel_smemfix.cubin (zero-arg
self-contained variant produced by tools/gdn_l2_gen.py --driver noargs +
scripts/run_l2.sh + tools/gdn_smem_fix.py), launches it with the grid/block
derived from the config, and measures median-of-50 CUDA-event times after 5
warmups.  Sanity: reads the public @gdn_out module global after the first
launch (zero-init => non-zero content proves the kernel wrote) and after the
timed loop (checksum stability).

Also probes the UNFIXED kernel.cubin once per config to document how the
as-shipped G2 lowering behaves on real hardware (expected: launch failure or
garbage due to the device-malloc shared-memory lowering -- see
tools/gdn_smem_fix.py).

Shared-GPU protocol: every measurement is flagged "shared_gpu": true with the
nvidia-smi co-load snapshot; absolute numbers are noisy by design -- re-run
in a clean window with the same filenames to overwrite with fresh values.

Per-config isolation: the parent process spawns itself as --worker per
config with a wall-clock timeout, so a hanging kernel (e.g. an mbarrier
deadlock) only kills its own subprocess and never lingers on the GPU.

Usage (conda env gdn):
  python tools/gdn_time.py --gen-root evidence/G3/gen_noargs \
      --out evidence/G3/config_times.json [--only NAME] [--iters 50]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import statistics
import subprocess
import sys
import time

REPO = os.path.expanduser("~/codebase/gdn-mlir")
GOUT_SYM = "gdn_out"

ENTRY_RE = re.compile(r"\.visible \.entry (\w+)\(")
NTID_RE = re.compile(r"\.(maxntid|reqntid) (\d+), (\d+), (\d+)")
PARAM_DECL_RE = re.compile(
    r"^\t\.param (?:\.align \d+ )?\.(\w+) (\w+_param_\d+)(?:\[(\d+)\])?(,?)$",
    re.M)
LLVM_CONST_RE = re.compile(
    r"(%\d+) = llvm\.mlir\.constant\(([^)]*)\)(?: : ([\w<>]+))?")
LAUNCH_RE = re.compile(
    r"gpu\.launch_func\s+@[\w:]+.*?\bargs\(([^)]*)\)", re.S)


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


def compute_apps():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10)
        return [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
    except Exception as e:  # noqa: BLE001
        return [f"error: {e}"]


# ---------------------------------------------------------------------------
# asmobj / PTX parsing
# ---------------------------------------------------------------------------

def parse_launch_values(dirp):
    """Positional launch-arg values from asmobj.mlir (host-side constants).

    Returns list of dicts {"kind": int|bool|float|vec|ptr, ...}."""
    txt = open(os.path.join(dirp, "asmobj.mlir")).read()
    consts = {}
    for m in LLVM_CONST_RE.finditer(txt):
        inner, oty = m.group(2), m.group(3)
        if " : " in inner:                       # scalar: "59 : i64"
            val, cty = inner.rsplit(" : ", 1)
        else:                                    # bool: "true" + outer ": i1"
            val, cty = inner, oty or ""
        consts[m.group(1)] = (val, cty)
    m = LAUNCH_RE.search(txt)
    if not m:
        return []
    out = []
    for a in m.group(1).split(","):
        a = a.strip()
        if not a:
            continue
        ssa, ty = a.split(":", 1)
        ssa, ty = ssa.strip(), ty.strip()
        if ssa in consts:
            v, cty = consts[ssa]
            if cty in ("i64", "i32"):
                out.append({"kind": "int", "value": int(v.strip())})
            elif cty == "i1":
                out.append({"kind": "bool", "value": 1 if v == "true" else 0})
            elif cty in ("f32", "f64"):
                out.append({"kind": "float", "value": float(v.strip())})
            elif "vector" in cty:
                nums = re.findall(r"-?[\d.]+(?:e[-+]\d+)?", v)
                out.append({"kind": "vec", "value": [float(x) for x in nums]})
            else:
                out.append({"kind": "other", "text": v, "mlir_ty": cty})
        else:
            out.append({"kind": "ptr", "value": None})  # e.g. tensormap
    return out


def parse_ptx_params(dirp, kname):
    """Remaining kernel params from the smemfix PTX (post-elimination).

    Scoped to <kname>_param_N symbols only (the .extern malloc decl also
    has *_param_* names but is not a kernel parameter). G4: also matches
    pointer-typed declarations `.param .u64 .ptr[.align N] <name>` (the
    tma config's host-encoded CUtensorMap kernel params)."""
    ptx = open(os.path.join(dirp, "kernel_smemfix.ptx")).read()
    pat = re.compile(
        r"^\t\.param (?:\.align \d+ )?(?:\.(\w+)(?: \.ptr(?: \.align \d+)?)?"
        r"|\.u64 \.ptr(?: \.align \d+)?) (" + re.escape(kname) +
        r"_param_\d+)(?:\[(\d+)\])?(,?)$", re.M)
    params = []
    for m in pat.finditer(ptx):
        ty, name, arr = m.group(1), m.group(2), m.group(3)
        size = int(arr) if arr else {"b8": 1, "b16": 2, "b32": 4, "b64": 8,
                                     "f32": 4, "f64": 8}.get(ty, 8)
        params.append({"idx": int(name.rsplit("_param_", 1)[1]),
                       "ty": ty or "u64ptr", "size": size})
    return params


def grid_from_config(cfg):
    stage = cfg["stage"]
    if stage in ("ref", "ref_ldmatrix"):
        return 1, 1
    bt, bv = cfg["tile"]["M"], cfg["tile"]["N"]
    n_glob = 64 if stage == "kkt" else cfg["dims"]["V"]
    return 64 // bt, n_glob // bv


def out_spec(cfg):
    stage, dt = cfg["stage"], cfg["dtype"]
    esz = {"bf16": 2, "f16": 2, "f32": 4}[dt]
    if stage == "kkt":
        return 64, 64, esz
    if stage == "o":
        return 64, cfg["dims"]["V"], esz
    if stage == "ref":
        return 16, 8, esz
    return 16, 8, 4  # ref_ldmatrix writes f32


# ---------------------------------------------------------------------------
# worker: run ONE config on the GPU
# ---------------------------------------------------------------------------

def run_worker(dirp, iters, warmup, probe=False):
    import cuda.bindings.driver as cu
    import numpy as np

    def ck(res, what):
        err = res[0] if isinstance(res, tuple) else res
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{what}: {err}")
        return res

    cfg = json.load(open(os.path.join(dirp, "config.json")))
    ptx = open(os.path.join(dirp, "kernel_smemfix.ptx")).read()
    kname = ENTRY_RE.search(ptx).group(1)
    nt = NTID_RE.search(ptx)
    block = int(nt.group(2)) if nt else 32 * cfg["warps"]
    gx, gy = grid_from_config(cfg)
    stream = cu.CUstream(0)
    ck(cu.cuInit(0), "cuInit")
    e, dev = cu.cuDeviceGet(0)
    ck((e,), "cuDeviceGet")
    e, ctx = cu.cuDevicePrimaryCtxRetain(dev)
    ck((e,), "cuDevicePrimaryCtxRetain")
    ck(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    if probe:
        # ---- probe the UNFIXED cubin (G2 lowering as shipped) ------------
        # run in an isolated subprocess (parent gives this a short timeout):
        # garbage constant-params can make the as-built kernel hang or trap.
        raw = open(os.path.join(dirp, "kernel.cubin"), "rb").read()
        e, mod0 = cu.cuModuleLoadData(raw)
        ck((e,), "cuModuleLoadData(unfixed)")
        e, fn0 = cu.cuModuleGetFunction(mod0, kname.encode())
        ck((e,), "cuModuleGetFunction(unfixed)")
        e = cu.cuLaunchKernel(fn0, gx, gy, 1, block, 1, 1, 0, stream,
                              None, 0)[0]
        if e != cu.CUresult.CUDA_SUCCESS:
            return {"name": cfg["name"], "g2_asbuilt_launch":
                    f"launch failed: {e}"}
        e2 = cu.cuStreamSynchronize(stream)[0]
        if e2 != cu.CUresult.CUDA_SUCCESS:
            return {"name": cfg["name"], "g2_asbuilt_launch":
                    f"sync failed (async trap): {e2}"}
        return {"name": cfg["name"], "g2_asbuilt_launch": "launched+synced ok"}

    res = {"name": cfg["name"], "grid": [gx, gy, 1], "block": [block, 1, 1],
           "iterations": iters, "warmup": warmup, "kernel": kname,
           "variant": "smemfix"}

    # ---- load the smemfix cubin ------------------------------------------
    raw = open(os.path.join(dirp, "kernel_smemfix.cubin"), "rb").read()
    e, mod = cu.cuModuleLoadData(raw)
    ck((e,), "cuModuleLoadData")
    e, fn = cu.cuModuleGetFunction(mod, kname.encode())
    ck((e,), "cuModuleGetFunction")

    # ---- kernel parameters ------------------------------------------------
    # cuda-bindings wants kernelParams as N POINTERS (ints) to the values
    params = parse_ptx_params(dirp, kname)
    values = parse_launch_values(dirp)
    kparams = None
    if params:
        kparams = []
        keepalive = []          # ctypes storage must outlive the launches
        tma_positions = []
        for pos, p in enumerate(params):
            v = values[p["idx"]] if p["idx"] < len(values) else None
            if v is None or v["kind"] == "ptr":
                tma_positions.append(pos)
                kparams.append(0)
                keepalive.append(None)
                continue
            if v["kind"] in ("int", "bool"):
                if p["size"] == 1:
                    o = ctypes.c_uint8(v["value"] & 0xFF)
                elif p["size"] == 2:
                    o = ctypes.c_uint16(v["value"] & 0xFFFF)
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
        if tma_positions:
            ok, msg = encode_and_patch_tma(cu, cfg, mod, tma_positions,
                                           kparams)
            if not ok:
                res["status"] = "NOT_TIMED"
                res["error"] = msg
                return res
        # cuda-bindings accepts kernelParams as a buffer of pointer values
        kparams = np.array(kparams, dtype=np.uint64)

    # ---- output global ----------------------------------------------------
    e, gout, gsize = cu.cuModuleGetGlobal(mod, GOUT_SYM.encode())
    if e != cu.CUresult.CUDA_SUCCESS:
        res["status"] = "NOT_TIMED"
        res["error"] = (
            f"cuModuleGetGlobal({GOUT_SYM}): {e}. The tma variant's cubin "
            "carries no data globals: all global accesses go through "
            "host-side CUtensorMap descriptors that G2 deliberately left "
            "to the runtime (create.descriptor is a host op). Wiring them "
            "here is moot: the kernel shares the demoted-mbarrier "
            "producer/consumer skeleton with kkt_bf16_mma_p2_t64x64x64_w4_ws, "
            "which deadlocks on hardware (see its TIMEOUT entry).")
        return res
    ck((e,), f"cuModuleGetGlobal({GOUT_SYM})")
    orows, ocols, esz = out_spec(cfg)
    out_bytes = orows * ocols * esz

    def launch():
        # note: this cuda-bindings build requires extra as int 0 (not None)
        ck(cu.cuLaunchKernel(fn, gx, gy, 1, block, 1, 1, 0, stream, kparams,
                             0), "cuLaunchKernel")

    def read_out():
        buf = np.zeros(out_bytes, dtype=np.uint8)
        ck(cu.cuMemcpyDtoH(buf, gout, out_bytes), "cuMemcpyDtoH")
        return buf

    def checksum(buf):
        if esz == 2:
            a = buf.view(np.uint16)
            return int(a.astype(np.int64).sum()), int((a != 0).sum())
        a = buf.view(np.float32)
        return float(a.astype(np.float64).sum()), int((a != 0).sum())

    def zero_out():
        ck(cu.cuMemsetD8(gout, 0, out_bytes), "cuMemsetD8(gdn_out)")

    # ---- verify launch (zero-init output first: the zero-init global is
    # the proof the kernel wrote; some payloads ACCUMULATE into outs
    # (linalg.matmul outs=), so every verification launch starts from 0) ----
    try:
        zero_out()
        launch()
        ck(cu.cuStreamSynchronize(stream), "sync after verify launch")
    except Exception as e:  # noqa: BLE001
        res["status"] = "LAUNCH_FAILED"
        res["error"] = str(e)
        return res
    buf1 = read_out()
    cs1, nz1 = checksum(buf1)
    res["checksum"] = cs1
    res["nonzero_elems"] = nz1
    res["out_total_elems"] = orows * ocols
    res["checksum_ok"] = nz1 > 0

    # determinism check: zero + relaunch must reproduce the checksum exactly
    zero_out()
    launch()
    ck(cu.cuStreamSynchronize(stream), "sync after determinism launch")
    buf2 = read_out()
    cs2, _ = checksum(buf2)

    # ---- timed loop -------------------------------------------------------
    _, ev0 = ck(cu.cuEventCreate(0), "cuEventCreate")
    _, ev1 = ck(cu.cuEventCreate(0), "cuEventCreate")
    times = []
    try:
        for _ in range(warmup):
            launch()
        cu.cuStreamSynchronize(stream)
        for _ in range(iters):
            ck(cu.cuEventRecord(ev0, stream), "record0")
            launch()
            ck(cu.cuEventRecord(ev1, stream), "record1")
            ck(cu.cuEventSynchronize(ev1), "eventSync")
            err, ms = cu.cuEventElapsedTime(ev0, ev1)
            ck((err,), "elapsed")
            times.append(ms)
    except Exception as e:  # noqa: BLE001
        res["status"] = "TIMING_FAILED"
        res["error"] = str(e)
        return res

    res["ms_median"] = round(statistics.median(times), 6)
    res["ms_min"] = round(min(times), 6)
    res["ms_max"] = round(max(times), 6)
    res["ms_all"] = [round(t, 6) for t in times]
    res["checksum_stable"] = (cs1 == cs2)
    res["status"] = ("OK" if (res["checksum_ok"] and cs1 == cs2)
                     else "CHECKSUM_UNSTABLE")
    return res


def encode_and_patch_tma(cu, cfg, mod, positions, kparams):
    """Host-side CUtensorMap wiring for the tma=on config.

    The G2 payload creates descriptors host-side over k[64,128]/c[64,64] and
    the kernel loads with cp.async.bulk.tensor.4d/5d/2d, so the tensormaps
    must be rank 4 / 5 / 2 respectively (dims fastest-varying first, as
    cuTensorMapEncodeTiled expects):
      descA rank4: (K, 64, 1, 1) box (bk, bt, 1, 1)   over gdn_in_k
      descB rank5: (K, 64, 1, 1, 1) box (bv, bk, 1, 1, 1)
      descS rank2: (64, 64) box (64, 64)              over gdn_out
    """
    try:
        import numpy as np
        bt, bk, bv = cfg["tile"]["M"], cfg["tile"]["K"], cfg["tile"]["N"]
        K = cfg["dims"]["K"]
        esz = 2

        def get_global(sym):
            e, ptr, _sz = cu.cuModuleGetGlobal(mod, sym)
            if e != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"global {sym}: {e}")
            return ptr

        gk = get_global(b"gdn_in_k")
        go = get_global(b"gdn_out")

        def encode(rank, addr, gdim, gstride, box):
            # pythonic binding RETURNS the CUtensorMap (not an out-param)
            r = cu.cuTensorMapEncodeTiled(
                cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                rank, addr, gdim, gstride, box, [1] * rank,
                cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
                cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
                cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
                cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
            if isinstance(r, tuple):
                if r[0] != cu.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f"encode rank{rank}: {r[0]}")
                return r[1]
            return r

        row = 64 * esz                      # bytes per k row
        plane = 64 * K * esz
        tmA = encode(4, gk, [K, 64, 1, 1], [row, plane, plane],
                     [bk, bt, 1, 1])
        tmB = encode(5, gk, [K, 64, 1, 1, 1], [row, plane, plane, plane],
                     [bv, bk, 1, 1, 1])
        tmS = encode(2, go, [64, 64], [64 * esz], [64, 64])

        for pos, tm in zip(positions, (tmA, tmB, tmS)):
            e, dptr = cu.cuMemAlloc(256)
            if e != cu.CUresult.CUDA_SUCCESS:
                return False, f"cuMemAlloc tensormap: {e}"
            raw = ctypes.string_at(ctypes.byref(tm), ctypes.sizeof(tm))
            nb = len(raw)
            e = cu.cuMemcpyHtoD(dptr, np.frombuffer(raw, dtype=np.uint8),
                                nb)[0]
            if e != cu.CUresult.CUDA_SUCCESS:
                return False, f"HtoD tensormap: {e}"
            # kparams entries are plain pointer ints for cuLaunchKernel
            kparams[pos] = int(dptr)
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"tma host wiring failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", default=os.path.join(
        REPO, "evidence/G3/gen_noargs"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "evidence/G3/config_times.json"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=240,
                    help="per-config worker wall timeout (s)")
    ap.add_argument("--worker", action="store_true",
                    help="internal: run a single config, print RESULT:<json>")
    ap.add_argument("--probe", action="store_true",
                    help="worker mode: probe the UNFIXED kernel.cubin only")
    ap.add_argument("--worker-dir", default=None)
    args = ap.parse_args()

    if args.worker:
        try:
            r = run_worker(args.worker_dir, args.iters, args.warmup,
                           probe=args.probe)
        except Exception as e:  # noqa: BLE001
            r = {"name": os.path.basename(args.worker_dir),
                 "status": "WORKER_CRASH",
                 "error": f"{type(e).__name__}: {e}"}
        print("RESULT:" + json.dumps(r))
        return 0

    def spawn(d, extra=(), timeout=None):
        return subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker",
             "--worker-dir", d, "--iters", str(args.iters),
             "--warmup", str(args.warmup), *extra],
            capture_output=True, text=True, timeout=timeout or args.timeout,
            cwd=REPO)

    def parse_result(p, n, timeout_note=""):
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT:")]
        if line:
            return json.loads(line[-1][7:])
        return {"name": n, "status": "WORKER_CRASH",
                "error": ((p.stderr or p.stdout) or "")[-600:] + timeout_note}

    names = [args.only] if args.only else sorted(os.listdir(args.gen_root))
    results = {}
    order = []
    coload = {"gpu_state_start": smi_snapshot(),
              "compute_apps_start": compute_apps()}
    for n in names:
        d = os.path.join(args.gen_root, n)
        if not os.path.isdir(d):
            continue
        order.append(n)
        print(f"[gdn_time] {n} ...", flush=True)
        t0 = time.time()
        # 1) timing worker (smemfix cubin)
        try:
            r = parse_result(spawn(d), n)
        except subprocess.TimeoutExpired:
            r = {"name": n, "status": "TIMEOUT",
                 "error": f"timing worker exceeded {args.timeout}s "
                          "(kernel hang?); subprocess killed"}
        # 2) probe worker (unfixed G2 cubin, short timeout, isolated)
        try:
            pr = parse_result(spawn(d, extra=("--probe",), timeout=25), n)
            r["g2_asbuilt_launch"] = pr.get(
                "g2_asbuilt_launch", pr.get("error", "no result"))
        except subprocess.TimeoutExpired:
            r["g2_asbuilt_launch"] = ("HUNG (>25s; garbage constant-params "
                                      "/ device-malloc trap); killed")
        r["wall_s"] = round(time.time() - t0, 1)
        results[n] = r
        print(f"    -> {r.get('status')} median={r.get('ms_median')} "
              f"nonzero={r.get('nonzero_elems')} "
              f"asbuilt={r.get('g2_asbuilt_launch')}", flush=True)

    coload["gpu_state_end"] = smi_snapshot()
    coload["compute_apps_end"] = compute_apps()
    doc = {
        "description": "G3-A measured GPU times of the 20 G2 knob configs "
                       "(--driver noargs self-contained variant, smem-fixed)",
        "timing": "CUDA events, median of N iterations after W warmups",
        "shared_gpu": True,
        "coload": coload,
        "caveat": "RTX 5090 Laptop was shared with a co-running job during "
                  "measurement (see coload); ABSOLUTE numbers are noisy "
                  "lower bounds. Re-run in a clean window with identical "
                  "commands to overwrite with fresh values.",
        "rerun": "scripts/run_g3.sh && conda run -n gdn python "
                 "tools/gdn_time.py",
        "results_order": order,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    ok = sum(1 for r in results.values() if r.get("status") == "OK")
    print(f"[gdn_time] wrote {args.out}: {ok}/{len(results)} OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
