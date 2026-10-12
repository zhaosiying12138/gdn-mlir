"""g7_flashqla_bwd_probe.py -- camera-ready TODO #2: does an SM120
FlashQLA BACKWARD number exist to compare against?  (GDN-MLIR G7/R26,
2026-10-10.)

G5-A (evidence/G5/baseline_report.md) recorded "SM120 ships a bf16
forward-only kernel" and benchmarked fwd only.  The local checkout
(shallow clone, v0.1.3, commit da06429 2026-09-30) DOES ship
blackwell_sm120/{fused_bwd,prepare_h,cp_bwd}.py and the public
chunk_gated_delta_rule_bwd is wired for sm120 (fused_gdr_bwd is not
None) -- so the "fwd-only" note must be re-verified by ATTEMPTING the
call, not by reading the release notes.

Probe: run the top-level flash_qla.chunk_gated_delta_rule under
autograd (the exact class of measurement G5 uses for the fla bwd
baseline) on the shared B1T2048H16K128V128 recipe, then
torch.autograd.grad((o*do).sum()+(ht*dht).sum(), inputs) -- timed.

Attempt ladder (each fully disclosed, mirroring the G5-A fwd harness):
  A. shipped defaults: auto_cp=True, no knobs;
  B. auto_cp=False dense (G5-A: auto_cp preprocess needs 110KB > 99KB
     opt-in on sm120);
  C. auto_cp=False + fwd block_DV forced to 64 (the G5-A disclosed shim
     for the fwd kernel's 112,640B > 101,376B smem under tilelang
     0.1.15) -- the bwd kernels themselves are never modified.
The sm120 aggregate_card_state import shim from the G5 harness is
applied (upstream export bug, disclosed).

Outcome classes: LAUNCHED (first-grad wall + 10 warm reps + finite
check) or FAILED (exception class + first 500 chars, per attempt).

GPU discipline: short burst, <4GB, opportunistic, co-job pid 3993998
never touched.  Output: evidence/G7/flashqla_bwd_probe.json
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser("~/codebase/gdn-mlir")
sys.path.insert(0, HERE)

import torch  # noqa: E402

import gdn_bench_common as C  # noqa: E402
import flash_qla  # noqa: E402
import tilelang  # noqa: E402
import flash_qla.ops.gated_delta_rule.chunk as _fq_chunk  # noqa: E402

# --- sm120 runtime shim (upstream export bug; same as G5-A harness) -------- #
SM120_SHIM = False
if not hasattr(_fq_chunk, "aggregate_card_state"):
    _fq_chunk.aggregate_card_state = None
    SM120_SHIM = True

_BDV_STATE = {"orig": None}


def _set_fwd_block_dv(bdv):
    """Force/clear the sm120 FWD kernel's block_DV (G5-A disclosed shim)."""
    import flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120.fused_fwd as ff
    if _BDV_STATE["orig"] is None:
        _BDV_STATE["orig"] = ff.tilelang_fused_chunk_gdr_fwd
    orig = _BDV_STATE["orig"]
    if bdv is None:
        ff.tilelang_fused_chunk_gdr_fwd = orig
        return

    def wrapped(*a, **kw):
        kw["block_DV"] = bdv
        return orig(*a, **kw)

    ff.tilelang_fused_chunk_gdr_fwd = wrapped


def smi_snapshot():
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10).stdout.strip()
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader", "-i", "0"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return {"gpu": gpu, "compute_apps": apps}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def main():
    assert torch.cuda.is_available()
    B, T, H, K, V = 1, 2048, 16, 128, 128
    scale = K ** -0.5
    case = C.gen_case(B=B, T=T, H=H, HV=H, K=K, V=V)
    io = C.to_device_bench(case, device="cuda", dtype=C.DTYPE_BENCH,
                           requires_grad=True)

    res = {
        "what": ("FlashQLA sm120 backward existence probe (camera-ready "
                 "TODO #2): attempt the public bwd under autograd"),
        "shape": {"B": B, "T": T, "H": H, "K": K, "V": V},
        "env": {
            "torch": torch.__version__,
            "tilelang": tilelang.__version__,
            "flash_qla": os.path.dirname(os.path.dirname(flash_qla.__file__)),
            "flash_qla_commit": "da06429 (v0.1.3, shallow clone 2026-10-09)",
            "gpu": torch.cuda.get_device_name(0),
            "capability": "sm_%d%d" % torch.cuda.get_device_capability(0),
            "sm120_aggregate_shim": SM120_SHIM,
        },
        "attempts": [],
        "smi_before": smi_snapshot(),
    }

    def run_attempt(name, auto_cp, force_bdv64):
        rec = {"attempt": name, "auto_cp": auto_cp,
               "fwd_block_dv_forced": 64 if force_bdv64 else None}
        _set_fwd_block_dv(64 if force_bdv64 else None)
        try:
            q = io["q"].detach().clone().requires_grad_(True)
            k = io["k"].detach().clone().requires_grad_(True)
            v = io["v"].detach().clone().requires_grad_(True)
            g = io["g"].detach().clone().requires_grad_(True)
            beta = io["beta"].detach().clone().requires_grad_(True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            o, ht = flash_qla.chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                initial_state=None, output_final_state=True,
                auto_cp=auto_cp)
            torch.cuda.synchronize()
            rec["fwd_s"] = round(time.perf_counter() - t0, 3)

            loss = (o * io["do"]).sum() + (ht * io["dht"]).sum()

            def grad():
                return torch.autograd.grad(
                    loss, (q, k, v, beta, g), retain_graph=True)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            grads = grad()
            torch.cuda.synchronize()
            rec["first_grad_s"] = round(time.perf_counter() - t0, 3)
            rec["grads_finite"] = all(
                bool(torch.isfinite(x.float()).all().item()) for x in grads)
            warm = []
            for _ in range(10):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                grad()
                torch.cuda.synchronize()
                warm.append(time.perf_counter() - t0)
            rec["warm_grad_ms"] = {
                "median": round(statistics.median(warm) * 1e3, 4),
                "min": round(min(warm) * 1e3, 4),
                "samples_ms": [round(x * 1e3, 4) for x in warm]}
            rec["status"] = "LAUNCHED"
        except Exception as e:
            rec["status"] = "FAILED"
            rec["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        finally:
            _set_fwd_block_dv(None)
        return rec

    res["attempts"].append(run_attempt("A_shipped_defaults", True, False))
    if res["attempts"][-1]["status"] != "LAUNCHED":
        res["attempts"].append(run_attempt("B_dense_no_autocp", False, False))
    if res["attempts"][-1]["status"] != "LAUNCHED":
        res["attempts"].append(run_attempt("C_dense_bdv64_fwdshim", False, True))

    res["smi_after"] = smi_snapshot()
    res["shared_gpu"] = True
    out = os.path.join(REPO, "evidence", "G7", "flashqla_bwd_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:4000])
    print(f"[g7-r26] wrote {out}")


if __name__ == "__main__":
    main()
