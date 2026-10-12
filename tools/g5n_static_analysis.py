#!/usr/bin/env python3
"""g5n_static_analysis.py -- G5-N round-7 static instruction analysis of the
landed bwd kernels (no GPU, no clean timing window needed).

For every kernel in the freshest landed bwd gen dir (evidence/G5/scratch/
r6_T2048, round-6 9-kernel chain) plus the fwd mono trio + fused fwd as
references, this tool computes from the SHIPPED artifacts only:

  * PTX opcode-family histograms (tools/harvest_flaptx.py FAMS, importable)
    on kernel_smemfix.ptx (the cubin the harness loads)
  * SASS opcode histograms (fresh nvdisasm of kernel_smemfix.cubin),
    including spill (LDL/STL), LDSM/LDGSTS and memory-width decodes
  * occupancy: regs (ptxas), smem/threads/grid (manifest) -> CTAs/SM,
    warps/SM against the project sm_120 model (gdn_costmodel constants)
  * per-chunk compute-intensity estimates: mma.sync sites (static PTX x
    block-fold for the NV-folded scan kernels), tensor FLOP, CUDA-core
    FLOP proxy (FFMA-class x threads), global-footprint bytes (manifest
    globals), estimated smem traffic bytes (SASS LDS/STS/LDSM widths)
  * distance-to-roofline: measured r6 per-stage medians vs floors from
    the G3 calibration with the CORRECTED rates (mma peak 115.5 TFLOP/s,
    the x32 warp-level fix; smem read 113.05 TB/s; DRAM 824.6 GB/s --
    all co-load LOWER BOUNDS, see evidence/G3/calibration.json +
    G6/search_cost.md)

Output: JSON (--out) + human tables on stdout. Deterministic, CPU-only.

Usage (conda gdn):
  python tools/g5n_static_analysis.py \
      --gen evidence/G5/scratch/r6_T2048 \
      --bench evidence/G5/bwd6_bench2048.json \
              evidence/G5/bwd6_bench2048_rerun.json \
      --out evidence/G5/scratch/g5n/static_analysis_r7.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import harvest_flaptx as HV  # noqa: E402  (FAMS / ptx_features)
import gdn_costmodel as cm   # noqa: E402  (sm_120 structural constants)

NVDISASM = "/usr/local/cuda/bin/nvdisasm"

# --- G3 calibration rates (corrected; co-load lower bounds) ---------------
CAL = json.load(open(os.path.join(REPO, "evidence/G3/calibration.json")))
M = CAL["measurements"]
# corrected mma peak: file flops field over-counts x32 (warp-level op);
# honest peak = instr/s / 32 * 4096 (evidence/G3/solver_report.md; the
# costmodel performs the same correction at load time)
MMA_PEAK = M["mma_bf16_m16n8k16"]["instr_per_s"] / 32.0 * 4096.0
SMEM_BW = M["smem_read_bw_gbps"]["gbps"] * 1e9        # B/s
DRAM_BW = M["gmem_read_dram"]["gbps"] * 1e9           # B/s
L2_BW = M["gmem_read_l2resident"]["gbps"] * 1e9       # B/s
BAR_NS = M["bar_sync_roundtrip"]["ns_per_bar"]

MMA_FLOP_PER_SITE = 2 * 16 * 8 * 16                    # m16n8k16

DTYPESZ = {"bf16": 2, "f32": 4, "f64": 8, "i32": 4, "i8": 1}


# ---------------------------------------------------------------------------
# artifact parsing
# ---------------------------------------------------------------------------

def split_ptx_entries(ptx: str) -> dict:
    """Split PTX text into per-.entry bodies ({entry_name: text})."""
    out, cur, name = {}, None, None
    for line in ptx.splitlines():
        m = re.match(r"\s*\.visible\s+\.entry\s+(\w+)", line)
        if m:
            if cur is not None:
                out[name] = "\n".join(cur)
            name, cur = m.group(1), []
        if cur is not None:
            cur.append(line)
    if cur is not None:
        out[name] = "\n".join(cur)
    return out


SASS_SEC = re.compile(r"^//--+\s*\.text\.([\w.]+)")
SASS_INSTR = re.compile(
    r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?U?PT?\d*\s+)?([A-Z][A-Z0-9._@!]*)")

WIDTHS = [(".128", 16), (".64", 8), (".32", 4), (".16", 2), (".8", 1)]

# PTX mma.sync shape -> FLOP per site (warp-level op)
MMA_SHAPE_FLOP = {
    "m16n8k16": 2 * 16 * 8 * 16,   # bf16
    "m16n8k8": 2 * 16 * 8 * 8,     # tf32
}


def ptx_mma_shapes(body: str) -> dict:
    """{shape: count} for mma.sync sites in a PTX body."""
    out = Counter()
    for m in re.findall(r"\bmma\.sync\.aligned\.(m\d+n\d+k\d+)\.", body):
        out[m] += 1
    return dict(out)


def instr_width(tok: str, default=4):
    for suf, w in WIDTHS:
        if suf in tok:
            return w
    return default


def disasm_sections(cubin: str) -> dict:
    """nvdisasm -> {section_name: [mnemonic tokens]}."""
    r = subprocess.run([NVDISASM, "-c", cubin], capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvdisasm failed on {cubin}: {r.stderr[:400]}")
    out, cur, name = {}, None, None
    for line in r.stdout.splitlines():
        m = SASS_SEC.match(line)
        if m:
            name = m.group(1)
            out.setdefault(name, [])
            continue
        if name is None:
            continue
        mi = SASS_INSTR.match(line)
        if mi:
            out[name].append(mi.group(1))
    return out


def sass_histogram(tokens) -> dict:
    base = Counter(t.split(".")[0] for t in tokens)
    fam = {
        "tensor": sum(base.get(k, 0) for k in ("HMMA", "QMMA", "OMMA")),
        "smem_ld": base.get("LDS", 0),
        "smem_st": base.get("STS", 0),
        "ldsm": base.get("LDSM", 0),
        "gmem_ld": base.get("LDG", 0),
        "gmem_st": base.get("STG", 0),
        "ldgsts": base.get("LDGSTS", 0),
        "spill_ld": base.get("LDL", 0),
        "spill_st": base.get("STL", 0),
        "bar": base.get("BAR", 0),
        "ffma": base.get("FFMA", 0),
        "hfma2": base.get("HFMA2", 0) + base.get("HFMA", 0),
        "fadd_fmul": base.get("FADD", 0) + base.get("FMUL", 0),
        "imad": base.get("IMAD", 0),
        "iadd3": base.get("IADD3", 0),
        "cvt": base.get("F2F", 0) + base.get("F2FP", 0) + base.get("I2F", 0)
               + base.get("S2F", 0) + base.get("F2I", 0)
               + base.get("F2S", 0) + base.get("PRMT", 0),
        "lop3": base.get("LOP3", 0),
        "shf": base.get("SHF", 0) + base.get("SHFL", 0),
        "mufu": base.get("MUFU", 0),
        "sel_mov": base.get("SEL", 0) + base.get("MOV", 0)
                   + base.get("MOVu32I", 0) + base.get("MOV32I", 0),
        "branch": sum(base.get(k, 0) for k in
                      ("BRA", "BSSY", "BSYNC", "BREAK", "BRX", "CALL",
                       "RET", "EXIT", "JMP", "LONGJMP")),
        "total": len(tokens),
    }
    # byte-width decodes for smem/global traffic estimates
    smem_ld_b = sum(instr_width(t) for t in tokens
                    if t.split(".")[0] == "LDS")
    smem_st_b = sum(instr_width(t) for t in tokens
                    if t.split(".")[0] == "STS")
    ldmatrix_b = sum(512 for t in tokens if t.split(".")[0] == "LDSM")
    fam["smem_ld_bytes_per_thread_pass"] = smem_ld_b
    fam["smem_st_bytes_per_thread_pass"] = smem_st_b
    fam["ldsm_bytes_per_warp_pass"] = 512 * base.get("LDSM", 0)
    return fam


def parse_ptxas_entries(txt: str) -> dict:
    """ptxas -v text -> {entry: metrics}."""
    out, cur = {}, None
    for line in txt.splitlines():
        m = re.search(r"entry function '(\w+)'", line)
        if m:
            cur = out.setdefault(m.group(1), {})
            continue
        if cur is None:
            continue
        for pat, key in ((r"Used (\d+) registers", "registers"),
                         (r"(\d+) bytes stack frame", "stack_frame"),
                         (r"(\d+) bytes spill stores", "spill_stores"),
                         (r"(\d+) bytes spill loads", "spill_loads")):
            mm = re.search(pat, line)
            if mm:
                cur[key] = int(mm.group(1))
    return out


def global_bytes(typestr: str) -> int:
    m = re.match(r"memref<([\dx]+)x(\w+)>", typestr)
    dims, dt = m.group(1).split("x"), m.group(2)
    n = 1
    for d in dims:
        n *= int(d)
    return n * DTYPESZ[dt]


# ---------------------------------------------------------------------------
# per-kernel analysis
# ---------------------------------------------------------------------------

def analyse_kernel(name, kdir, stage_meta, measured_ms, scan_fold):
    ptx = open(os.path.join(kdir, "kernel_smemfix.ptx")).read()
    ents = split_ptx_entries(ptx)
    ename, ebody = next(iter(ents.items())) if len(ents) == 1 \
        else (None, None)
    hist = HV.ptx_features(ebody) if ebody else {}
    shapes = ptx_mma_shapes(ebody) if ebody else {}
    sass = disasm_sections(os.path.join(kdir, "kernel_smemfix.cubin"))
    sname, stoks = next(iter(sass.items()))
    sass_f = sass_histogram(stoks)
    ptxas = parse_ptxas_entries(
        open(os.path.join(kdir, "ptxas_v.txt")).read())
    pt = next(iter(ptxas.values()))

    threads = stage_meta["block"][0]
    grid = stage_meta["grid"][0]
    smem = stage_meta["smem_bytes"]
    regs = pt["registers"]

    # occupancy (project sm_120 model)
    cta_regs = cm.REGS_PER_SM // (regs * threads)
    cta_smem = cm.SMEM_USABLE_B // max(1, smem) if smem else cm.CTA_SLOTS_PER_SM
    cta_sm = max(0, min(cta_regs, cta_smem, cm.CTA_SLOTS_PER_SM))
    warps_sm = cta_sm * threads // 32
    occ_pct = 100.0 * warps_sm / cm.WARP_SLOTS_PER_SM

    # per-chunk dynamic estimates: grid == B*H*NC -> static body == one
    # chunk per block; grid == B*H*NV (scan) -> NV blocks each walk every
    # chunk once, so per (bh, chunk) the body folds NV times.
    fold = scan_fold
    mma_sites = hist.get("mma.sync", 0)
    tensor_flop = sum(cnt * fold * MMA_SHAPE_FLOP.get(sh, 4096)
                      for sh, cnt in shapes.items())
    cuda_flop = (sass_f["ffma"] + sass_f["fadd_fmul"] + 2 * sass_f["hfma2"]) \
        * threads * fold
    smem_bytes = (sass_f["smem_ld_bytes_per_thread_pass"]
                  + sass_f["smem_st_bytes_per_thread_pass"]) * threads * fold
    smem_bytes += sass_f["ldsm_bytes_per_warp_pass"] * (threads // 32) * fold
    gbytes = sum(global_bytes(t) for t in stage_meta["globals"].values())

    # roofline floors per whole problem (NC chunks amortized)
    n_chunks = 512
    return {
        "name": name, "kernel": ename or sname,
        "grid": grid, "threads": threads, "smem": smem, "regs": regs,
        "spill_st": pt.get("spill_stores", 0),
        "spill_ld": pt.get("spill_loads", 0),
        "occupancy": {"ctas_per_sm": cta_sm, "warps_per_sm": warps_sm,
                      "pct_warp_slots": round(occ_pct, 1),
                      "limiting": ("smem" if cta_smem <= cta_regs
                                   else "regs")},
        "ptx": hist, "sass": sass_f, "mma_shapes": shapes,
        "f32_nocontract_note": (
            "PTX has zero fma.rn.f32 -- scalar f32 chains emit as separate "
            "add/sub/mul.rn.f32 (no FMA contraction on this path); counted "
            "as f32_scalar_ops="
            f"{len(re.findall(r'[.](add|sub|mul).rn.f32', ebody)) if ebody else 0}"),
        "ratios": {
            "ld.shared_per_mma": (round(hist["ld.shared"] / mma_sites, 2)
                                  if mma_sites else None),
            "st.shared_per_mma": (round(hist["st.shared"] / mma_sites, 2)
                                  if mma_sites else None),
            "ldmatrix_per_mma": (round(hist["ldmatrix"] / mma_sites, 2)
                                 if mma_sites else None),
            "ld.global_per_mma": (round(hist["ld.global"] / mma_sites, 2)
                                  if mma_sites else None),
            "fp32_ops_per_mma": (round(
                (sass_f["ffma"] + sass_f["fadd_fmul"])
                / (mma_sites * fold), 2) if mma_sites else None),
            "bar.sync_per_mma": (round(hist["bar.sync"] / mma_sites, 2)
                                 if mma_sites else None),
        },
        "per_chunk_est": {
            "mma_sites": mma_sites * fold,
            "tensor_mflop": round(tensor_flop / 1e6, 3),
            "cuda_core_mflop_est": round(cuda_flop / 1e6, 3),
            "smem_traffic_kib_est": round(smem_bytes / 1024.0, 1),
            "global_footprint_kib": round(gbytes / 32 / 1024.0, 1),
        },
        "measured_ms": measured_ms,
        "roofline": roofline(name, tensor_flop, cuda_flop, smem_bytes,
                             gbytes, measured_ms, n_chunks),
    }


def roofline(name, tensor_flop, cuda_flop, smem_bytes, gbytes, ms, n_chunks):
    if ms is None:
        return None
    T = ms * 1e-3
    tf = tensor_flop * n_chunks / T
    ccf = cuda_flop * n_chunks / T
    smb = smem_bytes * n_chunks / T
    gb = gbytes / T
    # floors: what the kernel WOULD take at the calibrated peaks
    floor_mma = tensor_flop * n_chunks / MMA_PEAK
    floor_cuda = ccf and cuda_flop * n_chunks / (
        MMA_PEAK * cm.CUDA_TO_TENSOR) or 0.0
    floor_smem = smem_bytes * n_chunks / SMEM_BW
    floor_dram = gbytes / DRAM_BW
    floor_l2 = gbytes / L2_BW
    fl = max(floor_mma, floor_cuda, floor_smem, floor_dram)
    binding = ("mma" if floor_mma == fl else
               "cuda" if floor_cuda == fl else
               "smem" if floor_smem == fl else "dram")
    return {
        "achieved_tensor_tflops": round(tf / 1e12, 4),
        "achieved_cudacore_tflops_est": round(ccf / 1e12, 4),
        "achieved_smem_tbps_est": round(smb / 1e12, 4),
        "achieved_dram_gbps": round(gb / 1e9, 1),
        "floor_ms": {k: round(v * 1e3, 4) for k, v in
                     (("mma", floor_mma), ("cuda", floor_cuda),
                      ("smem", floor_smem), ("dram", floor_dram),
                      ("l2_resident", floor_l2))},
        "binding": binding,
        "distance_x_measured_over_floor": round(ms / (fl * 1e3), 2),
        "tensor_peak_frac": round(tf / MMA_PEAK, 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="evidence/G5/scratch/r6_T2048")
    ap.add_argument("--bench", nargs="+", default=[
        "evidence/G5/bwd6_bench2048.json",
        "evidence/G5/bwd6_bench2048_rerun.json"])
    ap.add_argument("--fwd-mono",
                    default="evidence/G5/gen_fwd/B1T2048_mono")
    ap.add_argument("--fwd-fused",
                    default="evidence/G5/gen_fwd/B1T2048_fused")
    ap.add_argument("--out",
                    default="evidence/G5/scratch/g5n/"
                            "static_analysis_r7.json")
    args = ap.parse_args()

    man = json.load(open(os.path.join(REPO, args.gen,
                                      "bwd_manifest.json")))
    meas = {}
    for bp in args.bench:
        d = json.load(open(os.path.join(REPO, bp)))
        for k, v in d["ours_stage_ms_median"].items():
            meas.setdefault(k, []).append(v)
    meas = {k: sum(v) / len(v) for k, v in meas.items()}
    bh, nc = man["cfg"]["BH"], man["cfg"]["NC"]

    out = {"what": "G5-N round-7 static instruction analysis (CPU-only)",
           "gen_root": args.gen,
           "calibration": {"mma_peak_tflops": round(MMA_PEAK / 1e12, 2),
                           "corrected_x32_warp": True,
                           "smem_read_tbps": round(SMEM_BW / 1e12, 2),
                           "dram_gbps_coload_lb": round(DRAM_BW / 1e9, 1),
                           "l2_resident_gbps": round(L2_BW / 1e9, 1),
                           "bar_ns": BAR_NS},
           "kernels": {}, "fwd_reference": {}}

    for st in man["stages"]:
        nm = st["name"]
        grid = st["grid"][0]
        # scan kernels run grid = B*H*NV blocks, each walking every chunk:
        # per (bh, chunk) the static body folds NV times
        scan_fold = (grid // bh) if grid < bh * nc else 1
        out["kernels"][nm] = analyse_kernel(
            nm, os.path.join(REPO, args.gen, nm), st, meas.get(nm),
            scan_fold)
        r = out["kernels"][nm]
        print(f"[{nm:8s}] grid={grid:4d} thr={r['threads']} "
              f"regs={r['regs']} smem={r['smem']} "
              f"occ={r['occupancy']['pct_warp_slots']}% "
              f"({r['occupancy']['limiting']}) "
              f"mma/chunk={r['per_chunk_est']['mma_sites']} "
              f"binding={r['roofline']['binding'] if r['roofline'] else '-'} "
              f"dist={r['roofline']['distance_x_measured_over_floor']
                      if r['roofline'] else '-'}x")

    # fwd references (ratios only; single module, per-entry split)
    for tag, root in (("mono", args.fwd_mono), ("fused", args.fwd_fused)):
        kdir = os.path.join(REPO, root)
        if not os.path.isdir(kdir):
            continue
        ptx = open(os.path.join(kdir, "kernel_smemfix.ptx")).read()
        ents = split_ptx_entries(ptx)
        sass = disasm_sections(os.path.join(kdir,
                                            "kernel_smemfix.cubin"))
        ptxas = parse_ptxas_entries(
            open(os.path.join(kdir, "ptxas_v.txt")).read())
        meta = json.load(open(os.path.join(kdir, "meta.json")))
        mets = meta.get("metrics", {})
        for en, body in ents.items():
            short = en.replace("gdn_fwd_", "").replace("_kernel", "")
            sec = next((s for s in sass if s.endswith(en)), None)
            sf = sass_histogram(sass[sec]) if sec else {}
            ph = HV.ptx_features(body)
            mma = ph.get("mma.sync", 0)
            out["fwd_reference"][f"{tag}/{short}"] = {
                "ptx": ph, "sass": sf,
                "regs": (ptxas.get(en, {}).get("registers")
                         or mets.get("registers")),
                "smem": mets.get("smem_static") or mets.get("smem"),
                "ratios": {
                    "ld.shared_per_mma": round(ph["ld.shared"] / mma, 2)
                    if mma else None,
                    "st.shared_per_mma": round(ph["st.shared"] / mma, 2)
                    if mma else None,
                    "ldmatrix_per_mma": round(ph["ldmatrix"] / mma, 2)
                    if mma else None,
                    "ld.global_per_mma": round(ph["ld.global"] / mma, 2)
                    if mma else None,
                    "fp32_ops_per_mma": round(
                        (sf.get("ffma", 0) + sf.get("fadd_fmul", 0))
                        / mma, 2) if mma else None,
                    "bar.sync_per_mma": round(ph["bar.sync"] / mma, 2)
                    if mma else None,
                },
            }

    os.makedirs(os.path.dirname(os.path.join(REPO, args.out)), exist_ok=True)
    with open(os.path.join(REPO, args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[out] {args.out}")

    # compact ratio table
    print(f"\n{'kernel':10s} {'lds/mma':>8s} {'sts/mma':>8s} "
          f"{'ldsm/mma':>9s} {'ldg/mma':>8s} {'fp32/mma':>9s} "
          f"{'bar/mma':>8s}")
    rows = list(out["kernels"].items()) + list(
        out["fwd_reference"].items())
    for nm, r in rows:
        rt = r["ratios"]
        fmt = lambda v: f"{v:8.2f}" if isinstance(v, float) else f"{'--':>8s}"
        print(f"{nm:10s} {fmt(rt['ld.shared_per_mma'])} "
              f"{fmt(rt['st.shared_per_mma'])} "
              f"{fmt(rt['ldmatrix_per_mma']):>9s} "
              f"{fmt(rt['ld.global_per_mma'])} "
              f"{fmt(rt['fp32_ops_per_mma']):>9s} {fmt(rt['bar.sync_per_mma'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
