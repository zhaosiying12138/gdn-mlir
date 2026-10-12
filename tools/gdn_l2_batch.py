#!/usr/bin/env python3
"""G2-A batch driver: generate + compile the built-in knob matrix.

For every built-in config of tools/gdn_l2_gen.py:
  1. generate payload.mlir + transform.mlir + config.json into
     evidence/G2/gen/<name>/
  2. compile via scripts/run_l2.sh (mlir-opt -> PTX -> ptxas -v -> nvdisasm)
  3. collect metrics into evidence/G2/gen/INDEX.json

CPU-only: mlir-opt / ptxas / nvdisasm never touch the GPU.

Usage:
  gdn_l2_batch.py [--only NAME ...] [--gen-root DIR] [--jobs N]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))          # .../tools
REPO = os.path.dirname(HERE)                                # repo root
sys.path.insert(0, HERE)

import gdn_l2_gen  # noqa: E402


def run_config(name: str, cfg: dict, gen_root: str) -> dict:
    cdir = os.path.join(gen_root, name)
    os.makedirs(cdir, exist_ok=True)
    entry = {"name": name, "config": cfg, "path": cdir,
             "ptx_path": os.path.join(cdir, "kernel.ptx"),
             "status": "ok"}
    t0 = time.time()
    try:
        subprocess.run([sys.executable, os.path.join(HERE, "gdn_l2_gen.py"),
                        "--gen", name, "--out", cdir], check=True,
                       capture_output=True, text=True)
        r = subprocess.run(["bash", os.path.join(REPO, "scripts", "run_l2.sh"),
                            cdir], capture_output=True, text=True)
        if r.returncode != 0:
            entry["status"] = "compile-failed"
            entry["error"] = (r.stdout + r.stderr)[-2000:]
            return entry
        with open(os.path.join(cdir, "meta.json")) as f:
            meta = json.load(f)
        entry["meta"] = {
            "registers": meta.get("metrics", {}).get("registers"),
            "smem_static": meta.get("metrics", {}).get("smem_static"),
            "smem_dynamic": meta.get("metrics", {}).get("smem_dynamic", 0),
            "spill_stores": meta.get("metrics", {}).get("spill_stores"),
            "spill_loads": meta.get("metrics", {}).get("spill_loads"),
            "stack_frame": meta.get("metrics", {}).get("stack_frame"),
            "ptx_features": meta.get("ptx_features", {}),
            "sass_ops": meta.get("sass_ops", {}),
            "mma_signatures": meta.get("mma_signatures", []),
            "ex2_variants": meta.get("ex2_variants", []),
            "ldmatrix": meta.get("ldmatrix", 0),
            "ptx_version": meta.get("ptx_version"),
            "ptx_size_bytes": meta.get("ptx_size_bytes"),
            "sass_line_count": meta.get("sass_line_count"),
        }
        # G2-B merge compatibility fields
        entry["ptx"] = os.path.relpath(entry["ptx_path"], gen_root)
        m = entry["meta"]
        entry["regs"] = m["registers"]
        entry["smem"] = m["smem_static"]
        entry["spills"] = (m["spill_stores"] or 0) + (m["spill_loads"] or 0)
        entry["ptxas"] = m
    except Exception as e:  # noqa: BLE001
        entry["status"] = "failed"
        entry["error"] = str(e)[-2000:]
    entry["seconds"] = round(time.time() - t0, 1)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these built-in config names")
    ap.add_argument("--gen-root",
                    default=os.path.join(REPO, "evidence", "G2", "gen"))
    ap.add_argument("--jobs", type=int, default=2,
                    help="parallel compile jobs (small tools; keep <=4)")
    ap.add_argument("--merge", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="merge results into the existing INDEX.json "
                         "(keep entries from previous runs)")
    args = ap.parse_args()

    cfgs = {c["name"]: c for c in gdn_l2_gen.builtin_configs()}
    names = args.only if args.only else list(cfgs)
    os.makedirs(args.gen_root, exist_ok=True)

    entries = []
    if args.jobs <= 1:
        for n in names:
            e = run_config(n, cfgs[n], args.gen_root)
            print(f"[batch] {n}: {e['status']} ({e['seconds']}s)", flush=True)
            entries.append(e)
    else:
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
            futs = {ex.submit(run_config, n, cfgs[n], args.gen_root): n
                    for n in names}
            for fut in concurrent.futures.as_completed(futs):
                n = futs[fut]
                e = fut.result()
                print(f"[batch] {n}: {e['status']} ({e['seconds']}s)",
                      flush=True)
                entries.append(e)

    order = {n: i for i, n in enumerate(names)}
    entries.sort(key=lambda e: order.get(e["name"], 999))

    # --merge (default): keep entries of previous runs not in this run, so
    # `--only new-cfg` appends to INDEX.json instead of clobbering it (G2-C).
    if args.merge:
        out_path = os.path.join(args.gen_root, "INDEX.json")
        if os.path.exists(out_path):
            with open(out_path) as f:
                prev = json.load(f)
            run_names = {e["name"] for e in entries}
            kept = [c for c in prev.get("configs", [])
                    if c.get("name") not in run_names]
            entries = kept + entries
            entries.sort(key=lambda e: e.get("name", ""))

    ok = sum(1 for e in entries if e["status"] == "ok")
    index = {
        "description": "G2-A knob-configured kernel matrix (sm_120a PTX/SASS)",
        "generated_by": "tools/gdn_l2_batch.py + tools/gdn_l2_gen.py + "
                        "scripts/run_l2.sh",
        "target": "sm_120a (+ptx87)",
        "gpu_execution": False,
        "ok_count": ok,
        "total_count": len(entries),
        "configs": entries,
    }
    out = os.path.join(args.gen_root, "INDEX.json")
    with open(out, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[batch] INDEX.json -> {out} ({ok}/{len(entries)} ok)")

    # gate quick-check
    any_mma_bf16 = any(
        any("f32.bf16.bf16.f32" in s for s in e.get("meta", {})
            .get("mma_signatures", []))
        for e in entries if e["status"] == "ok")
    any_cpasync = any(e.get("meta", {}).get("ptx_features", {})
                      .get("cp.async", 0) > 0
                      for e in entries if e["status"] == "ok")
    print(f"[batch] gate-check: bf16-f32acc mma.sync={any_mma_bf16} "
          f"cp.async={any_cpasync}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
