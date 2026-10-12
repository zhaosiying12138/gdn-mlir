#!/usr/bin/env python3
"""G6: measure fla's ON-DEVICE AUTOTUNE COST (cold vs warm), B1T2048 fwd+bwd.

Method (search_cost.md evidence, GDN-MLIR G6):
  * fresh TRITON_CACHE_DIR per phase => the FIRST call of
    chunk_gated_delta_rule pays full Triton compile + autotune-benchmark of
    every candidate config (FLA_CACHE_MODE default DISABLED: fla's disk
    config cache is NOT consulted; verified: fla/configs/<GPU> does not
    exist, FLA_CACHE_MODE unset -> always Triton autotune, cache.py:51).
  * the shared ~/.triton cache is never touched (own TRITON_CACHE_DIR), so
    the co-job is unaffected; nothing to restore afterwards.
  * Autotuner.run is wrapped to count, per kernel: configs in the (pruned)
    list, calls, and distinct tuned keys (self.cache size post-run).
  * cold = first call wall time (compile + benchmark all configs + 1 run);
    warm = median of 10 subsequent calls (per-call, sync'd).
  * bwd = torch.autograd.grad((o*do).sum()+(ht*dht).sum(), (q,k,v,beta,g,h0))
    exactly like tools/gdn_bench_flaptx.py (retain_graph=True).
GPU discipline: opportunistic window, <4GB, co-job pid 3411456 untouched;
nvidia-smi snapshots recorded before/after. Output: evidence/G6/fla_autotune_cost.json
"""
import os
import statistics
import subprocess
import sys
import time

# ---- env BEFORE torch/triton/fla imports ---------------------------------- #
REPO = os.path.expanduser("~/codebase/gdn-mlir")
os.environ["FLA_FLASH_QLA"] = "0"          # pure-Triton backend (G5 harness)
os.environ.setdefault("FLA_CACHE_MODE", "disabled")  # explicit default
FRESH_TC = "/tmp/g6_fla_triton_cache"
os.environ["TRITON_CACHE_DIR"] = FRESH_TC  # true-cold compile, co-job safe

sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, REPO)

import torch  # noqa: E402
import triton.runtime.autotuner as rt_autotuner  # noqa: E402


def smi_snapshot():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader", "-i", "0"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10).stdout.strip()
        return {"gpu": out, "compute_apps": apps}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


# ---- autotuner instrumentation -------------------------------------------- #
TUNERS = {}
_orig_run = rt_autotuner.Autotuner.run


def counted_run(self, *args, **kwargs):
    name = getattr(self.fn, "__name__", repr(self.fn))
    rec = TUNERS.setdefault(name, {"tuner": self, "n_calls": 0})
    rec["n_calls"] += 1
    n_cfg = len(self.configs)
    t0 = time.perf_counter()
    ret = _orig_run(self, *args, **kwargs)
    rec["last_call_s"] = round(time.perf_counter() - t0, 4)
    rec["n_configs_listed"] = n_cfg
    rec["n_configs_after_prune"] = len(self.configs)
    rec["n_tuned_keys"] = len(getattr(self, "cache", {}))
    return ret


rt_autotuner.Autotuner.run = counted_run


def tuner_stats():
    out = {}
    for name, rec in TUNERS.items():
        t = rec["tuner"]
        out[name] = {
            "n_calls": rec["n_calls"],
            "n_configs_listed_first_call": rec.get("n_configs_listed"),
            "n_configs_after_prune": len(t.configs),
            "n_tuned_keys": len(getattr(t, "cache", {})),
        }
    return out


def count_triton_cache():
    if not os.path.isdir(FRESH_TC):
        return 0
    return len([d for d in os.listdir(FRESH_TC)
                if os.path.isdir(os.path.join(FRESH_TC, d))])


def timed(fn, sync_first=True):
    if sync_first:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, r


def main():
    assert torch.cuda.is_available()
    dev = "cuda"
    # exact G5 shared recipe (tools/gdn_bench_common.py, fla test_chunk)
    import gdn_bench_common as C
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    B, T, H, K, V = 1, 2048, 16, 128, 128
    case = C.gen_case(B=B, T=T, H=H, HV=H, K=K, V=V)
    io = C.to_device_bench(case, device=dev, dtype=C.DTYPE_BENCH,
                           requires_grad=True)
    q, k, v, beta, g, h0 = (io["q"], io["k"], io["v"], io["beta"],
                            io["g"], io["h0"])
    scale = K ** -0.5

    res = {
        "what": "fla chunk_gated_delta_rule autotune cost (cold vs warm)",
        "shape": {"B": B, "T": T, "H": H, "K": K, "V": V},
        "env": {"FLA_FLASH_QLA": os.environ["FLA_FLASH_QLA"],
                "FLA_CACHE_MODE": os.environ["FLA_CACHE_MODE"],
                "TRITON_CACHE_DIR": FRESH_TC,
                "triton": __import__("triton").__version__,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0)},
        "smi_before": smi_snapshot(),
    }

    def fwd():
        o, ht = chunk_gated_delta_rule(q, k, v, g, beta=beta, scale=scale,
                                       initial_state=h0,
                                       output_final_state=True)
        return o, ht

    # ---------------- COLD fwd ---------------- #
    t_cold_fwd, (o, ht) = timed(fwd)
    res["cold_fwd_s"] = round(t_cold_fwd, 3)
    res["triton_cache_entries_after_cold_fwd"] = count_triton_cache()
    res["tuners_after_cold_fwd"] = tuner_stats()

    # ---------------- WARM fwd ---------------- #
    warm = []
    for _ in range(10):
        dt, _ = timed(fwd)
        warm.append(dt)
    res["warm_fwd_ms"] = {
        "median": round(statistics.median(warm) * 1e3, 4),
        "min": round(min(warm) * 1e3, 4),
        "samples_ms": [round(x * 1e3, 4) for x in warm],
    }

    # ---------------- COLD bwd ---------------- #
    loss = (o * io["do"]).sum() + (ht * io["dht"]).sum()

    def bwd():
        return torch.autograd.grad(loss, (q, k, v, beta, g, h0),
                                   retain_graph=True)

    t_cold_bwd, _ = timed(bwd)
    res["cold_bwd_s"] = round(t_cold_bwd, 3)
    res["triton_cache_entries_after_cold_bwd"] = count_triton_cache()
    res["tuners_after_cold_bwd"] = tuner_stats()

    # ---------------- WARM bwd ---------------- #
    warm = []
    for _ in range(10):
        dt, _ = timed(bwd)
        warm.append(dt)
    res["warm_bwd_ms"] = {
        "median": round(statistics.median(warm) * 1e3, 4),
        "min": round(min(warm) * 1e3, 4),
        "samples_ms": [round(x * 1e3, 4) for x in warm],
    }

    # ---------------- derived ---------------- #
    res["autotune_delta_fwd_s"] = round(
        t_cold_fwd - res["warm_fwd_ms"]["median"] / 1e3, 3)
    res["autotune_delta_bwd_s"] = round(
        t_cold_bwd - res["warm_bwd_ms"]["median"] / 1e3, 3)
    n_cfg_fwd = {k: v["n_configs_after_prune"]
                 for k, v in res["tuners_after_cold_fwd"].items()}
    n_cfg_bwd = {k: v["n_configs_after_prune"]
                 for k, v in res["tuners_after_cold_bwd"].items()
                 if k not in res["tuners_after_cold_fwd"]}
    res["configs_benchmarked_fwd"] = n_cfg_fwd
    res["configs_benchmarked_bwd_new"] = n_cfg_bwd
    res["n_configs_total_fwd"] = sum(n_cfg_fwd.values())
    res["n_configs_total_bwd_new"] = sum(n_cfg_bwd.values())
    res["n_configs_total"] = (res["n_configs_total_fwd"]
                              + res["n_configs_total_bwd_new"])
    res["smi_after"] = smi_snapshot()
    res["shared_gpu"] = True  # co-job pid 3411456 resident (protocol)

    out_path = os.path.join(REPO, "evidence", "G6", "fla_autotune_cost.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import json
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items()
                      if not k.startswith("samples")}, indent=1))
    print(f"[g6] wrote {out_path}")


if __name__ == "__main__":
    main()
