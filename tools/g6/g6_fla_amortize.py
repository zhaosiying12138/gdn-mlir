"""g6_fla_amortize.py -- camera-ready TODO #3: fla autotune AMORTIZATION
(GDN-MLIR G6/R26, 2026-10-10).

Question: the 50x search-cost claim (evidence/G6/search_cost.md,
fla_autotune_cost.json) is a COLD first-call comparison: fla pays
16.55 s fwd + 23.26 s bwd of Triton autotune search per (process, shape),
our Z3+model solver ~0.8 s.  How much of fla's cost is actually AMORTIZED
by fla's own disk caches on later processes?  This script measures it.

Mechanism (verified against fla @ ~/codebase/fla, triton 3.8.0):
  * fla kernels are decorated with `fla_cache_autotune`
    (fla/ops/utils/cache.py): CachedAutotuner.run first calls
    load_cached_config(kernel, key) which reads
    $FLA_CONFIG_DIR/{kernel}.json when FLA_CACHE_MODE != disabled.
    On a hit the config is pinned into self.cache and Triton's
    benchmark loop is skipped entirely (fla/ops/utils/cache.py::run /
    maybe_load_cached_config).
  * Triton 3.8 itself writes {kernel}.autotune.json result files into
    TRITON_CACHE_DIR after a real search (Autotuner.check_disk_cache,
    cache_results=True via fla's autotune_cache_kwargs,
    fla/utils/env.py:26 FLA_CACHE_RESULTS default '1'), and re-reads
    them on later processes (same tuning key + config list).
  * scripts/extract_triton_autotune_cache.py converts Triton's
    .autotune.json files into fla's {kernel}.json config files
    (autotune_entries keyed by md5 of the serialized tuning key).

Phases (driver below; every GPU phase is a FRESH python subprocess so
"first call" is a true process-cold measurement):
  P1 cold_populate : FLA_CACHE_MODE=disabled, FRESH TRITON_CACHE_DIR.
                     first fwd + first bwd = full search (the 39.8 s class
                     number, re-measured in this window); leaves
                     .autotune.json + cubins in $BASE/tc_pop.
  P2 extract       : extract_triton_autotune_cache.py -> $BASE/fla_configs
                     (no GPU work beyond name detection).
  P3 hit_coldtc    : FLA_CACHE_MODE=strict + FLA_CONFIG_DIR, FRESH
                     TRITON_CACHE_DIR.  first call = config lookup + JIT
                     compile of ONLY the chosen config (the
                     "configs shipped, new machine" scenario).
  P4 hit_warmtc    : same env, TRITON_CACHE_DIR == $BASE/tc_pop.
                     first call = config lookup + cubin load (the
                     "configs + compile cache warm" scenario).
  P5 native_hit    : FLA_CACHE_MODE=disabled, TRITON_CACHE_DIR == tc_pop.
                     Triton's own .autotune.json disk cache (what a user
                     with a persistent default ~/.triton gets with ZERO
                     fla-side configuration).

Each GPU phase records: first fwd/bwd wall, warm medians (10 reps),
per-kernel FLA lookup hits/misses, summed autotuner bench_time,
.autotune.json count in the phase triton cache, nvidia-smi snapshots.

GPU discipline: B1T2048 tiny tensors, short bursts, <4GB, opportunistic;
co-job (pid 3993998) never touched; shared ~/.triton never touched
(all caches under /tmp scratch).

Output: evidence/G6/fla_amortize_r26.json
"""
import json
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.expanduser("~/codebase/gdn-mlir")
FLA_REPO = os.path.expanduser("~/codebase/fla")
BASE = "/tmp/g6_r26_amortize"

# ---------------------------------------------------------------- phases --


def smi_snapshot():
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader", "-i", "0"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10).stdout.strip()
        return {"gpu": gpu, "compute_apps": apps}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def count_autotune_json(tc_dir):
    n = 0
    for root, _dirs, files in os.walk(tc_dir):
        n += sum(1 for f in files if f.endswith(".autotune.json"))
    return n


def run_phase(name):
    """The GPU workload, executed inside a fresh subprocess."""
    assert os.environ.get("G6_R26_PHASE") == name
    tc_dir = os.environ["TRITON_CACHE_DIR"]

    sys.path.insert(0, os.path.join(REPO, "tools"))
    sys.path.insert(0, REPO)

    import torch  # noqa: E402
    import triton.runtime.autotuner as rt_autotuner  # noqa: E402
    import fla.ops.utils.cache as fla_cache_mod  # noqa: E402

    # ---- instrumentation: FLA config lookups + autotuner bench time --- #
    FLA_LOOKUPS = []  # (kernel, hit_bool)

    _orig_load = fla_cache_mod.load_cached_config

    def counted_load(kernel_name, autotune_key=None):
        cfg = _orig_load(kernel_name, autotune_key)
        FLA_LOOKUPS.append((kernel_name, cfg is not None))
        return cfg

    fla_cache_mod.load_cached_config = counted_load

    BENCH_TIMES = {}
    _orig_run = rt_autotuner.Autotuner.run

    def timed_run(self, *args, **kwargs):
        t0 = time.perf_counter()
        ret = _orig_run(self, *args, **kwargs)
        k = getattr(self.fn, "__name__", repr(self.fn))
        rec = BENCH_TIMES.setdefault(k, {"n_runs": 0, "run_s": 0.0,
                                         "bench_s": 0.0})
        rec["n_runs"] += 1
        rec["run_s"] += time.perf_counter() - t0
        rec["bench_s"] += getattr(self, "bench_time", 0.0)
        return ret

    rt_autotuner.Autotuner.run = timed_run

    # ---- exact G5/G6 shared recipe ------------------------------------ #
    import gdn_bench_common as C
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    B, T, H, K, V = 1, 2048, 16, 128, 128
    case = C.gen_case(B=B, T=T, H=H, HV=H, K=K, V=V)
    io = C.to_device_bench(case, device="cuda", dtype=C.DTYPE_BENCH,
                           requires_grad=True)
    q, k, v, beta, g, h0 = (io["q"], io["k"], io["v"], io["beta"],
                            io["g"], io["h0"])
    scale = K ** -0.5

    res = {
        "phase": name,
        "env": {
            "FLA_CACHE_MODE": os.environ.get("FLA_CACHE_MODE", "<unset>"),
            "FLA_CONFIG_DIR": os.environ.get("FLA_CONFIG_DIR", "<unset>"),
            "TRITON_CACHE_DIR": tc_dir,
            "triton_cache_preexisting_autotune_json":
                count_autotune_json(tc_dir),
            "triton": __import__("triton").__version__,
            "torch": torch.__version__,
        },
        "smi_before": smi_snapshot(),
    }

    def fwd():
        o, ht = chunk_gated_delta_rule(q, k, v, g, beta=beta, scale=scale,
                                       initial_state=h0,
                                       output_final_state=True)
        return o, ht

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t0, r

    t0, (o, ht) = timed(fwd)
    res["first_fwd_s"] = round(t0, 3)
    res["autotuner_during_first_fwd"] = {
        k: {"n_runs": v["n_runs"], "run_s": round(v["run_s"], 3),
            "bench_s": round(v["bench_s"], 3)}
        for k, v in BENCH_TIMES.items()}
    res["fla_lookups_during_first_fwd"] = sorted(FLA_LOOKUPS)

    loss = (o * io["do"]).sum() + (ht * io["dht"]).sum()

    def bwd():
        return torch.autograd.grad(loss, (q, k, v, beta, g, h0),
                                   retain_graph=True)

    t0, _ = timed(bwd)
    res["first_bwd_s"] = round(t0, 3)
    res["fla_lookups_total"] = sorted(FLA_LOOKUPS)
    res["n_fla_lookup_hits"] = sum(1 for _, hit in FLA_LOOKUPS if hit)
    res["n_fla_lookup_misses"] = sum(1 for _, hit in FLA_LOOKUPS if not hit)

    warm_f = [timed(fwd)[0] for _ in range(10)]
    warm_b = [timed(bwd)[0] for _ in range(10)]
    res["warm_fwd_ms"] = {"median": round(statistics.median(warm_f) * 1e3, 4),
                          "min": round(min(warm_f) * 1e3, 4)}
    res["warm_bwd_ms"] = {"median": round(statistics.median(warm_b) * 1e3, 4),
                          "min": round(min(warm_b) * 1e3, 4)}

    res["autotune_json_in_tc_after"] = count_autotune_json(tc_dir)
    res["smi_after"] = smi_snapshot()
    out = os.path.join(BASE, f"phase_{name}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items()
                      if k in ("phase", "first_fwd_s", "first_bwd_s",
                               "warm_fwd_ms", "warm_bwd_ms",
                               "n_fla_lookup_hits", "n_fla_lookup_misses",
                               "autotune_json_in_tc_after")}, indent=1))
    print(f"[g6-r26] wrote {out}")


# ---------------------------------------------------------------- driver --


def spawn(name, extra_env):
    env = dict(os.environ)
    env.update(extra_env)
    env["G6_R26_PHASE"] = name
    r = subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--phase", name],
                       env=env, text=True, capture_output=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit(f"phase {name} failed rc={r.returncode}")


def main():
    os.makedirs(BASE, exist_ok=True)
    tc_pop = os.path.join(BASE, "tc_pop")
    tc_hit = os.path.join(BASE, "tc_hit")
    cfg_dir = os.path.join(BASE, "fla_configs")
    for d in (tc_pop, tc_hit):
        os.makedirs(d, exist_ok=True)   # fresh, empty

    common = {"FLA_FLASH_QLA": "0", "PYTHONUNBUFFERED": "1"}
    stamp = lambda: time.strftime("%H:%M:%S")  # noqa: E731

    print(f"[{stamp()}] P1 cold_populate (full search, fresh triton cache)")
    spawn("cold_populate", {**common, "FLA_CACHE_MODE": "disabled",
                            "TRITON_CACHE_DIR": tc_pop})

    print(f"[{stamp()}] P2 extract triton cache -> fla configs")
    env = dict(os.environ)
    env.update(common)
    env["FLA_CACHE_MODE"] = "disabled"
    r = subprocess.run(
        [sys.executable,
         os.path.join(FLA_REPO, "scripts",
                      "extract_triton_autotune_cache.py"),
         "--triton-cache-dir", tc_pop, "--output-dir", cfg_dir],
        env=env, text=True, capture_output=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit("extract failed")
    n_cfg = len([f for f in os.listdir(cfg_dir) if f.endswith(".json")])

    print(f"[{stamp()}] P3 hit_coldtc (strict FLA configs, fresh triton)")
    spawn("hit_coldtc", {**common, "FLA_CACHE_MODE": "strict",
                         "FLA_CONFIG_DIR": cfg_dir,
                         "TRITON_CACHE_DIR": tc_hit})

    print(f"[{stamp()}] P4 hit_warmtc (strict FLA configs, warm triton)")
    spawn("hit_warmtc", {**common, "FLA_CACHE_MODE": "strict",
                         "FLA_CONFIG_DIR": cfg_dir,
                         "TRITON_CACHE_DIR": tc_pop})

    print(f"[{stamp()}] P5 native_hit (fla cache disabled, warm triton)")
    spawn("native_hit", {**common, "FLA_CACHE_MODE": "disabled",
                         "TRITON_CACHE_DIR": tc_pop})

    # ------------------------------------------------------------ merge -- #
    phases = {}
    for name in ("cold_populate", "hit_coldtc", "hit_warmtc", "native_hit"):
        with open(os.path.join(BASE, f"phase_{name}.json")) as f:
            phases[name] = json.load(f)

    cold = phases["cold_populate"]
    cold_total = cold["first_fwd_s"] + cold["first_bwd_s"]
    merged = {
        "what": ("fla chunk_gated_delta_rule autotune AMORTIZATION "
                 "(camera-ready TODO #3): cold search vs disk-cache-hit "
                 "first call vs warm steady state, B1T2048H16K128V128"),
        "shape": {"B": 1, "T": 2048, "H": 16, "K": 128, "V": 128},
        "mechanism": {
            "fla_version": "local checkout ~/codebase/fla (fla_cache_autotune decorator on all GDN kernels)",
            "fla_cache_doc": "fla/fla/ops/utils/cache.py (FLA_CACHE_MODE strict/fuzzy/full; FLA_CONFIG_DIR overrides fla/configs/{gpu})",
            "extract_script": "fla/scripts/extract_triton_autotune_cache.py",
            "triton": "3.8.0 (Autotuner.check_disk_cache .autotune.json result cache, cache_results=1 default in fla/utils/env.py)",
        },
        "n_fla_config_files_extracted": n_cfg,
        "phases": phases,
        "derived": {},
        "protocol": ("each phase = fresh python process; first call wall "
                     "time incl. cuda sync; warm = median of 10; "
                     "opportunistic co-loaded window, co-job pid 3993998 "
                     "never touched; shared ~/.triton never touched"),
        "shared_gpu": True,
    }
    d = merged["derived"]
    for name in ("hit_coldtc", "hit_warmtc", "native_hit"):
        p = phases[name]
        tot = p["first_fwd_s"] + p["first_bwd_s"]
        d[f"{name}_first_call_total_s"] = round(tot, 3)
        d[f"{name}_amortized_vs_cold_x"] = round(cold_total / tot, 1) if tot else None
    d["cold_first_call_total_s"] = round(cold_total, 3)
    d["warm_steady_state_fwd_ms"] = cold["warm_fwd_ms"]["median"]
    d["warm_steady_state_bwd_ms"] = cold["warm_bwd_ms"]["median"]

    out_path = os.path.join(REPO, "evidence", "G6", "fla_amortize_r26.json")
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=1)
    print(json.dumps(d, indent=1))
    print(f"[g6-r26] wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--phase":
        run_phase(sys.argv[2])
    else:
        main()
