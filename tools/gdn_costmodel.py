#!/usr/bin/env python3
"""GDN calibrated analytical cost model (G3-B).

Analytical latency model for the GDN L2 knob-configured chunk-stage kernel
family (tools/gdn_l2_gen.py) on sm_120 (Blackwell consumer: 82 SMs,
Ampere-style mma.sync, 99KB opt-in smem per CTA, 64K 32-bit regs per SM).

This is the "calibrated analytical cost model" of docs/PLAN.md G3: every
rate is either (a) measured once by the G3-A microbenchmark window
(evidence/G3/calibration.json) or (b) a documented sm_120-class structural
constant. There are NO per-config fitted free constants. The only global
constants of the launch-overhead kind are T_LAUNCH_FALLBACK and T_MALLOC
(see calibration keys launch_ns / malloc_ns to override them).

Model forms (each iteration is evaluated by tools/gdn_solver.py --validate
and documented in evidence/G3/solver_report.md):

  v1  (contract form, roofline)
      T = T_launch + overlap(T_compute, T_gmem, T_smem) + T_bar + T_spill + T_ex2
      overlap = max(...) when the config is software-pipelined
      (pipeline>0 or warpspec producer/consumer: cp.async double buffering
      hides the copy behind compute), sum(...) when pipeline=0 (sync copies
      serialized behind gpu.barrier).
  v2  v1 + T_lsu + T_alloc
      T_lsu   = issue-bound term for the LSU instruction stream. The
                generated code issues UNIFORM-ADDRESS (non-distributed)
                global->shared copies: every thread executes every
                16B cp.async of the tile (verified in kernel.ptx: copy
                indices are block-uniform, no %tid in the address chains).
                Wire bytes are deduped by L1/L2 sectors, but the ISSUE cost
                is instructions x threads. Fragment ld.shared loads are
                CSE'd unique-per-iteration loads (verified: LDS static count
                = unique fragment loads, not 12x atoms).
      T_alloc = the memref.alloc lowering calls device malloc() for each
                tile buffer and cvta.to.shared's the result (verified in
                kernel.ptx: 2 mallocs per GEMM in the mma path, 3 allocs in
                the nomma path, +1 output staging buffer for tma=on).
                Cost = n_allocs * malloc_ns (single global constant).
  v3  v2 + occupancy/waves + overlap blend
      waves = ceil(total_ctas / (82 * ctas_per_sm)); ctas_per_sm from
      regs/smem/warp-slot limits; per-wave time = v2 per-CTA time.
      overlap becomes alpha*max + (1-alpha)*sum with alpha = 1.0 for
      pipeline>0 / warpspec, 0.0 for pipeline=0 (same as v1/v2 split,
      kept explicit for perturbation experiments).

Term definitions (each term is count / rate):

  T_mma   = mma_flops_per_cta / (mma_bf16_peak / 82)            [bf16 path]
          = mma_flops_per_cta / (mma_bf16_peak * TF32_RATIO / 82) [tf32 path]
            mma_flops_per_cta = 2*bt*bv*K*n_gemm (n_gemm=1 kkt, 2 o).
  T_ffma  = nomma path: MACs_per_cta * NOMMA_CYC_PER_MAC / clock. The
            convert-linalg-to-loops lowering emits an undistributed
            scalar triple loop (innermost body = 39 SASS inst / 3 scalar
            LDS per MAC, SASS-derived) -> 10 cycles/MAC structural
            constant. (Iteration 3 of the model; the original CUDA-core
            roofline form underpredicted the measured nomma configs by
            100-400x because the loop is issue/latency bound, not
            FLOP bound.)
  T_gmem  = gmem_bytes / (bw_effective / 82) with
            gmem_bytes = (bt*K + K*bv)*esz*n_gemm + bt*bv*esz
            (per CTA; equal to (M*K + K*N + M*N)*esz amplified by the
            tiling grid: A tiles re-read n_glob/bv times, B tiles re-read
            64/bt times). bw_effective = l2_bw when l2_resident (the
            timing harness loops the kernel; operands are KB-scale and
            L2-resident after the first run), else gmem_bw.
  T_smem  = fragment-load bytes / (smem_bw / 82):
            unique ld.shared per K-iter per thread
              = (bt/16)*(bk/ak)*a_ld + ((bv/warps)/8)*(bk/ak)*b_ld
            with a_ld/b_ld = 8/4 ops of esz bytes (bf16 m16n8k16) or
            2/1 ops (tf32 m16n8k4); x trips x n_gemm x threads x esz B.
  T_bar   = n_barriers_dyn * bar_sync_ns
            (mma path: 2*trips*n_gemm gpu.barrier; nomma: (trips+1)*n_gemm;
             warpspec: trips gpu.barrier + 2*trips mbarrier ops at the
             same bar_sync_ns estimate).
  T_spill = spill_bytes / (spill_bw) with spill_bw = l2_bw/82 * SPILL_BW_FRAC.
            spill ops are the ptxas LDL+STL counts from INDEX.json
            (compile-time measured; 0 when predicting un-compiled configs -
            the solve->compile->re-rank loopback covers that), converted to
            dynamic ops by x (trips+pipe)/(1+pipe) (undoes the peeled-body
            duplication of pipeline_shared_memory_copies), 4 B per op,
            x threads / CTA.
  T_ex2   = ex2_count / (ex2_rate / 82); ex2_count = bt*bv per CTA (kkt
            tril-decay epilogue; math.exp2 -> ex2.approx.f32).
  T_lsu   = warp-level LSU instructions / (LSU_IOPS_PER_CLK * clock):
            cp.async warp-ops = (bt+bv)*(bk/ch)*trips*n_gemm per copying
            warp x #copying warps (all warps classic, 1 producer warp for
            warpspec; ch = 8 bf16 / 4 f32); fragment ld.shared warp-ops =
            unique-loads-per-iter x trips x n_gemm x mma warps; epilogue
            st.global warp-ops = bt*bv/32 (tma=on: 0). The generated
            copies are uniform-address (every lane of a warp issues the
            same 16B), so wire bytes stay logical but ISSUE cost is per
            warp instruction.
  T_alloc = n_allocs * malloc_ns (0 for the G3-A static-smem-repaired
            cubins that the timing harness measures).

Determinism: pure int/float arithmetic, no RNG anywhere; Spearman uses
average ranks with a fixed tie rule. The Z3 driver (gdn_solver.py) sets a
fixed seed + rlimit.
"""

from __future__ import annotations

import json
import math
import os

# ---------------------------------------------------------------------------
# sm_120 hardware model (structural constants, documented)
# ---------------------------------------------------------------------------

SM_COUNT = 82
SMEM_TOTAL_B = 99 * 1024          # opt-in max smem per CTA (sm_120)
SMEM_RESERVED_B = 1024            # reserved (driver/system)
SMEM_USABLE_B = SMEM_TOTAL_B - SMEM_RESERVED_B   # 100352 B
REGS_PER_SM = 65536               # 64K 32-bit registers per SM
REGS_MAX_PER_THREAD = 255
WARP_SLOTS_PER_SM = 32            # max warps resident per SM
CTA_SLOTS_PER_SM = 16             # max CTAs resident per SM (sm_120 class)

# Throughput / ratio constants (sm_120 consumer class). All RATES come from
# calibration.json when present; these only derive secondary rates:
TENSOR_FLOP_PER_CLK_SM = 1024.0   # bf16 mma.sync w/ f32 acc, dense, per SM/clk
TF32_RATIO = 0.5                  # tf32 tensor rate vs bf16 tensor rate
CUDA_TO_TENSOR = 0.5              # fp32 CUDA-core FLOP vs bf16 tensor FLOP
BF16_NOMMA_OVERHEAD = 2.0         # bf16 CUDA-core math instruction-mix factor
LSU_IOPS_PER_CLK = 4.0            # LSU issue (inst/cycle/SM, incl. cp.async)
SPILL_BW_FRAC = 1.0               # spill (local mem) vs L2 per-SM share
# mma=off path: convert-linalg-to-loops emits a SCALAR triple loop whose
# innermost body is 39 SASS instructions (10x IADD.64 addressing + loop
# control + 3x LDS.U16 + FMA; measured on
# evidence/G3/gen_noargs/kkt_bf16_nomma_p2_t64x64x32_w4/sass.txt loop
# .L_x_17) -> ~39/4 issue pipes ~= 10 cycles per MAC. The loop is NOT
# distributed across threads by this lowering, and its LDS are
# latency-dominant, so 10 cyc/MAC is the documented structural constant
# (SASS-derived, not fitted to measured times).
NOMMA_CYC_PER_MAC = 10.0

# Fallback calibration (used ONLY when evidence/G3/calibration.json is
# missing keys; flagged "fallback" in outputs). Values are 5090-class
# datasheet estimates - NOT measurements; the gate verdict in the final
# report is computed with the real G3-A numbers.
FALLBACK = {
    "mma_bf16_peak_flops": 209.0e12,
    "gmem_bw": 1792e9,
    "l2_bw": 5500e9,
    "smem_bw": 20000e9,
    "ex2_rate": 3.3e12,
    "bar_sync_ns": 30.0,
    "launch_ns": 4000.0,
    "malloc_ns": 2000.0,
    "clock_hz": 2.49e9,
    "shared_gpu": None,
}

# ---------------------------------------------------------------------------
# calibration loading (tolerant to key naming / units)
# ---------------------------------------------------------------------------

_KEY_ALIASES = {
    "mma_bf16_peak_flops": ["mma_bf16_peak_flops", "mma_peak_flops",
                            "mma_bf16_peak", "mma_peak", "mma_bf16_tflops",
                            "mma_bf16_peak_gflops"],
    "gmem_bw": ["gmem_bw", "gmem_bandwidth", "gmem_gbps", "dram_bw",
                "gmem_bw_gbps", "gmem_bytes_s"],
    "l2_bw": ["l2_bw", "l2_bandwidth", "l2_gbps", "l2_bw_gbps",
              "l2_bytes_s"],
    "smem_bw": ["smem_bw", "smem_bandwidth", "smem_gbps", "smem_bw_gbps",
                "shared_bw", "smem_bytes_s"],
    "ex2_rate": ["ex2_rate", "ex2_elem_s", "ex2_elems", "ex2_throughput",
                 "ex2_gelem_s", "ex2_ops_s"],
    "bar_sync_ns": ["bar_sync_ns", "bar_ns", "barrier_ns", "bar_sync",
                    "barsync_ns"],
    "launch_ns": ["launch_ns", "launch_us", "launch_overhead_ns",
                  "launch_overhead_us", "kernel_launch_ns"],
    "malloc_ns": ["malloc_ns", "malloc_us", "device_malloc_ns",
                  "alloc_ns"],
    "clock_hz": ["clock_hz", "clock_ghz", "sm_clock_hz", "clock"],
    "shared_gpu": ["shared_gpu", "gpu_shared", "shared"],
}


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten(v, key))
            else:
                out[key] = v
    return out


def _canon(s):
    return "".join(c for c in str(s).lower() if c.isalnum())


def _find(flat, aliases):
    """Return (matched_key, value) or (None, None)."""
    canon = {_canon(k): (k, v) for k, v in flat.items()}
    for a in aliases:
        ca = _canon(a)
        if ca in canon:
            return canon[ca]
    for a in aliases:                      # substring fallback
        ca = _canon(a)
        for ck, (k, v) in canon.items():
            if ca and ca in ck:
                return k, v
    return None, None


def _norm_flops(v):
    v = float(v)
    if v < 1000:            # TFLOP/s
        return v * 1e12
    if v < 1e9:             # GFLOP/s
        return v * 1e9
    return v                # FLOP/s


def _norm_bw(v):
    v = float(v)
    if v < 1e6:             # GB/s
        return v * 1e9
    return v                # B/s


def _norm_rate(v):
    v = float(v)
    if v < 1e6:             # G/s
        return v * 1e9
    return v


def _norm_time(v, matched_key):
    """Normalize a time to ns. Units decided by the matched key's suffix
    when present ('us' -> microseconds, 'ns' -> nanoseconds); unitless
    values are ns unless they look like seconds (<1e-3)."""
    v = float(v)
    k = _canon(matched_key)
    if k.endswith("us") or k.endswith("useconds"):
        return v * 1e3
    if v < 1e-3:                        # seconds
        return v * 1e9
    return v                            # ns


def _norm_clock(v):
    v = float(v)
    if v < 100:             # GHz
        return v * 1e9
    if v < 1e7:             # MHz
        return v * 1e6
    if v < 1e10:            # kHz
        return v * 1e3
    return v                # Hz


class Calibration:
    """Normalized calibration table (SI units: B/s, FLOP/s, elem/s, ns).

    Parses the G3-A evidence/G3/calibration.json schema natively:
      device: {sm_count, clock_peak_khz, shared_per_sm_optin, regs_per_sm}
      mma_bf16_m16n8k16: {flops}            smem_read_bw_gbps: {gbps}
      gmem_read_l2resident: {gbps}          gmem_read_dram: {gbps}
      sfu_ex2_f32: {elem_per_s}             bar_sync_roundtrip: {ns_per_bar}
      cp_async_16b: {gbps}
    plus a generic alias search for other layouts. Falls back to the
    documented placeholder table when keys are missing (flagged).
    """

    def __init__(self, data=None, path=None):
        self.source = path or "<fallback>"
        self.fallback_used = set()
        self.m = dict(FALLBACK)
        self.m.setdefault("sm_count", SM_COUNT)
        self.m.setdefault("smem_usable_b", SMEM_USABLE_B)
        self.m.setdefault("regs_per_sm", REGS_PER_SM)
        self.m.setdefault("cpasync_bw", None)
        self._set = set()
        if data:
            self._parse_g3a(data)
        self._derive()

    def _parse_g3a(self, data):
        # G3-A nests everything under "measurements"
        root = data.get("measurements") if isinstance(
            data.get("measurements"), dict) else data

        def sec(name):
            d = root.get(name)
            return d if isinstance(d, dict) else {}

        def num(d, *keys):
            for k in keys:
                v = d.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return float(v)
            return None

        dev = sec("device")
        v = num(dev, "sm_count")
        if v:
            self.m["sm_count"] = int(v)
            self._set.add("sm_count")
        v = num(dev, "clock_peak_khz", "clock_khz")
        if v:
            self.m["clock_hz"] = v * 1e3
            self._set.add("clock_hz")
        v = num(dev, "shared_per_sm_optin", "sharedemperblockoptin")
        if v:
            self.m["smem_usable_b"] = int(v) - SMEM_RESERVED_B
            self._set.add("smem_usable_b")
        v = num(dev, "regs_per_sm", "regspermultiprocessor")
        if v:
            self.m["regs_per_sm"] = int(v)
            self._set.add("regs_per_sm")
        if dev.get("name"):
            self.m["device_name"] = dev["name"]
        # observed clocks during the measurement window (nvidia-smi coload
        # snapshots) - preferred over the peak attribute when present,
        # because T_lsu scales with the clock actually running
        coload = data.get("coload")
        obs = []
        if isinstance(coload, dict):
            for vv in coload.values():
                if isinstance(vv, dict):
                    c = vv.get("sm_clock_mhz")
                    if isinstance(c, (int, float)):
                        obs.append(float(c))
        if obs:
            obs.sort()
            self.m["clock_observed_hz"] = obs[len(obs) // 2] * 1e6
            if "clock_hz" not in self._set or \
                    self.m["clock_observed_hz"] > self.m["clock_hz"]:
                # laptop attribute can lag the boost clock actually used
                self.m["clock_hz"] = self.m["clock_observed_hz"]
                self._set.add("clock_hz")
                self.fallback_used.discard("clock_hz")
                self.fallback_used.discard("clock_hz(derived)")

        mma_sec = sec("mma_bf16_m16n8k16")
        mma_flops = num(mma_sec, "flops", "flops_per_s")
        mma_instr = num(mma_sec, "instr_per_s")
        if mma_flops and mma_instr:
            # G3-A's flops field multiplies per-THREAD instruction counts
            # by 4096 (m16n8k16 atom FLOPs), but mma.sync is a WARP-level
            # op: the honest peak = warp instructions/s x 4096 =
            # instr_per_s / 32 x 4096. Detect the discrepancy (a >2x gap
            # is physically impossible otherwise) and recompute.
            warp_flops = mma_instr / 32.0 * 4096.0
            if mma_flops > warp_flops * 1.5:
                self.m["mma_flops_field_asbuilt"] = mma_flops
                mma_flops = warp_flops
                self.m["mma_flops_corrected"] = True
        m = {
            "mma_bf16_peak_flops": mma_flops,
            "smem_bw": num(sec("smem_read_bw_gbps"), "gbps"),
            "l2_bw": num(sec("gmem_read_l2resident"), "gbps"),
            "gmem_bw": num(sec("gmem_read_dram"), "gbps"),
            "ex2_rate": num(sec("sfu_ex2_f32"), "elem_per_s"),
            "bar_sync_ns": num(sec("bar_sync_roundtrip"), "ns_per_bar"),
            "cpasync_bw": num(sec("cp_async_16b"), "gbps"),
        }
        for k, val in m.items():
            if val is None:
                continue
            if k.endswith("_bw") or k == "cpasync_bw":
                val = _norm_bw(val)
            self.m[k] = val
            self._set.add(k)
        if "shared_gpu" in data:
            self.m["shared_gpu"] = bool(data["shared_gpu"])

        # generic alias search for anything the schema did not provide
        flat = _flatten(data)
        for canon in ("mma_bf16_peak_flops", "gmem_bw", "l2_bw", "smem_bw",
                      "ex2_rate", "bar_sync_ns", "launch_ns", "malloc_ns",
                      "clock_hz"):
            if canon in self._set:
                continue
            mk, v = _find(flat, _KEY_ALIASES[canon])
            if v is None or isinstance(v, bool):
                continue
            try:
                if canon in ("bar_sync_ns", "launch_ns", "malloc_ns"):
                    self.m[canon] = _norm_time(v, mk)
                elif canon == "mma_bf16_peak_flops":
                    self.m[canon] = _norm_flops(v)
                elif canon == "clock_hz":
                    self.m[canon] = _norm_clock(v)
                else:
                    self.m[canon] = _norm_bw(v)
                self._set.add(canon)
            except (TypeError, ValueError):
                pass

    def _derive(self):
        for k in ("mma_bf16_peak_flops", "gmem_bw", "l2_bw", "smem_bw",
                  "ex2_rate", "bar_sync_ns", "launch_ns", "malloc_ns",
                  "clock_hz"):
            if k not in self._set:
                self.fallback_used.add(k)
        if self.m.get("cpasync_bw") is None:
            self.fallback_used.add("cpasync_bw")
        if self.m["clock_hz"] == FALLBACK["clock_hz"]:
            # derive SM clock from the measured tensor peak (documented):
            # clock = peak / (#SM * TENSOR_FLOP_PER_CLK_SM)
            self.m["clock_hz"] = self.m["mma_bf16_peak_flops"] / (
                self.m["sm_count"] * TENSOR_FLOP_PER_CLK_SM)
            if "clock_hz" in self.fallback_used:
                self.fallback_used.add("clock_hz(derived)")

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            return cls(data=json.load(f), path=path)

    # derived rates -----------------------------------------------------
    @property
    def sm_count(self):
        return self.m["sm_count"]

    def per_sm(self, key):
        return self.m[key] / self.m["sm_count"]

    @property
    def mma_tf32_peak(self):
        return self.m["mma_bf16_peak_flops"] * TF32_RATIO

    @property
    def cuda_fp32_peak(self):
        return self.m["mma_bf16_peak_flops"] * CUDA_TO_TENSOR

    def table(self):
        rows = {k: self.m.get(k) for k in
                ("mma_bf16_peak_flops", "gmem_bw", "l2_bw", "smem_bw",
                 "ex2_rate", "bar_sync_ns", "launch_ns", "malloc_ns",
                 "clock_hz", "sm_count", "smem_usable_b", "regs_per_sm",
                 "cpasync_bw", "device_name")}
        rows["mma_tf32_peak_derived"] = self.mma_tf32_peak
        rows["cuda_fp32_peak_derived"] = self.cuda_fp32_peak
        rows["shared_gpu"] = self.m["shared_gpu"]
        return rows

    def __repr__(self):
        return f"Calibration({json.dumps(self.table(), indent=1)})"


# ---------------------------------------------------------------------------
# analytic features (exact given the code structure of gdn_l2_gen.py)
# ---------------------------------------------------------------------------

ESZ = {"bf16": 2, "f32": 4, "f16": 2}
CHUNK = {"bf16": 8, "f32": 4, "f16": 8}      # 16B cp.async chunk elems
ATOM_K = {"bf16": 16, "tf32": 4, "f16": 16}


def analytic_features(cfg, n_chunks=1):
    """Dynamic counts / byte volumes per CTA for one knob config.

    Verified against evidence/G2/gen/INDEX.json static counts:
      kkt p0 t64x64x32 w4  mma=16/iter (=4*2*2), LDGSTS=512 (=64*4*2*...),
      LDS=80 (=CSE'd unique fragment loads), EX2=32 (=bt*bv/threads);
      o f32 p2 t64x64x32    mma=384 static = 3*(4*2*8*2) (peel x3), etc.
    """
    stage = cfg["stage"]
    dt = cfg["dtype"]
    if stage in ("ref", "ref_ldmatrix"):
        # anchors: single m16n8k16 f16 atom; ref_ldmatrix stages through
        # smem with ldmatrix loads (per-thread st.shared of fragments)
        ldm = stage == "ref_ldmatrix"
        return {
            "stage": stage, "esz": 2, "bt": 16, "bv": 8, "bk": 16,
            "K": 16, "n_glob": 8, "trips": 1, "n_gemm": 1, "warps": 1,
            "threads": 32, "mma": True, "pipe": 0, "ws": False, "tma": False,
            "ctas_per_chunk": 1, "n_chunks": n_chunks,
            "mma_flops_per_cta": 2 * 16 * 8 * 16,
            "ffma_flops_per_cta": 0,
            "atoms_per_warp": 1,
            "lds_ops_per_thread": 0,
            "cpasync_ops_per_cta": 0,
            "lsu_ops_per_cta": (16 + 4) if not ldm else (24 + 4),
            "gmem_bytes_per_cta": (16 * 16 + 16 * 8) * 2 + 16 * 8 * 2,
            "smem_bytes_per_cta": (16 * 16 + 16 * 8) * 2 * (32 if ldm else 1),
            "barriers_dyn": 1 if ldm else 0,
            "ex2_count_per_cta": 0,
            "n_allocs": 2 if ldm else 0,
            "smem_static_model": (16 * 16 + 16 * 8) * 2,
            "spill_dyn_ops_per_cta": 0,
        }

    bt, bv, bk = cfg["tile"]["M"], cfg["tile"]["N"], cfg["tile"]["K"]
    K = cfg["dims"]["K"]
    n_glob = 64 if stage == "kkt" else cfg["dims"]["V"]
    warps = cfg["warps"]
    ws = cfg.get("warpspec", "off") == "producer_consumer"
    tma = cfg.get("tma", "off") == "on"
    mma = bool(cfg["mma"])
    pipe = cfg["pipeline"]
    prec = cfg.get("prec", "bf16" if dt == "bf16" else "tf32")
    esz = ESZ[dt]
    ch = CHUNK[dt]
    ak = ATOM_K[prec if mma else "bf16"]
    n_gemm = 1 if stage == "kkt" else 2
    threads = 32 * (warps + (1 if ws else 0))
    trips = K // bk
    gx, gy = 64 // bt, n_glob // bv
    ctas_per_chunk = gx * gy

    flops = 2 * bt * bv * K * n_gemm
    gmem_bytes = (bt * K + K * bv) * esz * n_gemm + bt * bv * esz

    if mma:
        atoms_iter = (bt // 16) * ((bv // warps) // 8) * (bk // ak)
        atoms_per_warp = atoms_iter * trips * n_gemm
        a_ld, b_ld = (8, 4) if dt == "bf16" else (2, 1)
        lds_iter = (bt // 16) * (bk // ak) * a_ld \
            + ((bv // warps) // 8) * (bk // ak) * b_ld
        lds_per_thread = lds_iter * trips * n_gemm
        ffma = 0
        barriers = (trips if ws else 2 * trips * n_gemm)
        if ws:
            barriers += 2 * trips          # mbarrier try_wait + arrive
        n_allocs = 2 * n_gemm + (1 if tma else 0)
    else:
        atoms_per_warp = 0
        lds_per_thread = 8                 # transpose/epilogue stragglers
        ffma = flops
        barriers = (trips + 1) * n_gemm
        n_allocs = 3

    # cp.async: uniform-address copies issued in lockstep by every thread
    # of the copying warps (all warps in the classic path; the single
    # producer warp in the warpspec path); LSU cost is WARP instructions
    # (= per-thread count x #warps); tma=on replaces the stream with 2
    # bulk descriptors per K-iteration (negligible issue)
    if tma:
        cpasync_warp_ops = 2 * trips
        cpasync_per_cta = 2 * trips            # legacy field (ops)
    else:
        cpasync_per_thread_iter = (bt + bv) * (bk // ch)
        copy_warps = 1 if ws else warps
        cpasync_warp_ops = (cpasync_per_thread_iter * trips * n_gemm
                            * copy_warps)
        cpasync_per_cta = cpasync_warp_ops

    ex2_count = bt * bv if stage == "kkt" else 0
    # epilogue stores: one scalar st.global of esz bytes per output element
    # per thread (uniform fragment-lane addressing; INDEX st.global=32 for
    # kkt t64x64 = 4096/128 threads); tma=on stages through smem instead
    stg_ops = 0 if tma else bt * bv

    # static smem byte model (documented; n_gemm pairs of tile buffers,
    # multibuffer factor = max(pipe,1) except the single-buffered ws path;
    # + output staging buffer for tma; + mbarrier words for ws)
    if mma:
        pair = (bt * bk + bk * bv) * esz
        if ws:
            smem_static = pair + (64 * 64 * esz if tma else 0) + 8
        else:
            smem_static = pair * max(pipe, 1) * n_gemm
    else:
        smem_static = (bt * K + K * bv + bt * bv) * esz

    return {
        "stage": stage, "esz": esz, "bt": bt, "bv": bv, "bk": bk, "K": K,
        "n_glob": n_glob, "trips": trips, "n_gemm": n_gemm, "warps": warps,
        "threads": threads, "mma": mma, "pipe": pipe, "ws": ws, "tma": tma,
        "ctas_per_chunk": ctas_per_chunk, "n_chunks": n_chunks,
        "grid": [gx, gy],
        "mma_flops_per_cta": flops if mma else 0,
        "ffma_flops_per_cta": ffma,
        "atoms_per_warp": atoms_per_warp,
        "lds_ops_per_thread": lds_per_thread,
        "cpasync_ops_per_cta": cpasync_per_cta,
        "cpasync_warp_ops": cpasync_warp_ops,
        # fragment loads: warp instructions = per-thread count x mma warps
        "lds_warp_ops": lds_per_thread * (warps if mma else 0),
        # epilogue stores: bt*bv scalar stores / 32 lanes per warp inst
        "stg_warp_ops": stg_ops // 32 if stg_ops else 0,
        "stg_ops_per_cta": stg_ops,
        "gmem_bytes_per_cta": gmem_bytes,
        "smem_bytes_per_cta": ((lds_per_thread * threads * esz)
                               + (64 * 64 * esz if tma else 0)) if mma else 0,
        "barriers_dyn": barriers,
        "ex2_count_per_cta": ex2_count,
        "n_allocs": n_allocs,
        "smem_static_model": smem_static,
        "spill_dyn_ops_per_cta": 0,     # filled from INDEX when available
    }


def merge_index_features(feats, entry):
    """Overlay compile-time-measured features from a compile index entry
    (evidence/G2/gen INDEX.json or evidence/G3/gen_noargs/*/meta.json)."""
    if not entry:
        return feats, {}
    meta = entry.get("meta", {})
    mm = meta.get("metrics", meta)     # G3 flat metas nest under "metrics"
    pf = meta.get("ptx_features", {})
    so = meta.get("sass_ops", {})
    used = {}
    if entry.get("smemfix"):
        # the G3-A static-smem repair replaces the device malloc callseqs
        # with .shared declarations -> no alloc overhead for timed cubins
        feats["n_allocs"] = 0
        used["smemfix"] = True
    regs = mm.get("registers")
    if regs:
        feats["regs_measured"] = regs
        used["registers"] = regs
    spills = ((mm.get("spill_stores") or 0)
              + (mm.get("spill_loads") or 0))
    if spills == 0:
        spills = (so.get("LDL") or 0) + (so.get("STL") or 0)
    if spills:
        # undo peeled-body duplication: static ~= body*(1+pipe),
        # dynamic ~= body*(trips+pipe)
        pipe = max(feats["pipe"], 0)
        dyn = spills * (feats["trips"] + pipe) / (1 + pipe)
        feats["spill_dyn_ops_per_cta"] = dyn * feats["threads"]
        used["spill_static_ldl_stl"] = spills
    ex2 = pf.get("ex2.approx") or 0
    if ex2:
        feats["ex2_count_per_cta"] = ex2 * feats["threads"]
        used["ex2_static"] = ex2
    if pf.get("mma.sync"):
        used["mma_static"] = pf["mma.sync"]
    return feats, used


# ---------------------------------------------------------------------------
# prediction (model forms v1 / v2 / v3)
# ---------------------------------------------------------------------------

def _overlap(terms, piped, alpha=None):
    if alpha is None:
        alpha = 1.0 if piped else 0.0
    mx, sm = max(terms), sum(terms)
    return alpha * mx + (1.0 - alpha) * sm


def occupancy(feats, calib=None):
    """CTAs resident per SM from regs / smem / warp- and CTA-slot limits."""
    threads = feats["threads"]
    warps = threads // 32
    regs_per_sm = calib.m["regs_per_sm"] if calib else REGS_PER_SM
    smem_usable = calib.m["smem_usable_b"] if calib else SMEM_USABLE_B
    regs = feats.get("regs_measured", REGS_MAX_PER_THREAD)
    by_regs = regs_per_sm // (threads * regs)
    by_smem = max(1, smem_usable // max(1, feats["smem_static_model"]))
    by_warps = WARP_SLOTS_PER_SM // warps
    return max(1, min(by_regs, by_smem, by_warps, CTA_SLOTS_PER_SM))


def predict(cfg, calib, form="v2", n_chunks=1, index_entry=None,
            l2_resident=True, alpha=None):
    """Return (total_seconds, terms_dict). Forms: v1, v2, v3."""
    feats = analytic_features(cfg, n_chunks=n_chunks)
    feats, used_index = merge_index_features(feats, index_entry)

    esz = feats["esz"]
    terms = {}
    terms["T_launch"] = calib.m["launch_ns"] * 1e-9

    # compute ------------------------------------------------------------
    if feats["stage"] in ("ref", "ref_ldmatrix"):
        terms["T_mma"] = feats["mma_flops_per_cta"] / calib.per_sm(
            "mma_bf16_peak_flops")
        terms["T_ffma"] = 0.0
        terms["T_gmem"] = feats["gmem_bytes_per_cta"] / calib.per_sm("l2_bw")
        terms["T_smem"] = feats["smem_bytes_per_cta"] / calib.per_sm("smem_bw")
        terms["overlap"] = (terms["T_mma"] + terms["T_gmem"]
                            + terms["T_smem"])
        terms["T_bar"] = feats["barriers_dyn"] * calib.m["bar_sync_ns"] * 1e-9
        terms["T_spill"] = 0.0
        terms["T_ex2"] = 0.0
        terms["T_lsu"] = feats["lsu_ops_per_cta"] / (
            LSU_IOPS_PER_CLK * calib.m["clock_hz"])
        terms["T_alloc"] = feats["n_allocs"] * calib.m["malloc_ns"] * 1e-9
        total = (terms["T_launch"] + terms["overlap"] + terms["T_bar"]
                 + terms["T_lsu"] + terms["T_alloc"])
        return total, terms, feats, used_index

    if feats["mma"]:
        peak = (calib.m["mma_bf16_peak_flops"] if esz == 2
                else calib.mma_tf32_peak)
        terms["T_mma"] = feats["mma_flops_per_cta"] / (
            peak / calib.sm_count)
        terms["T_ffma"] = 0.0
    else:
        terms["T_mma"] = 0.0
        # scalar-loop model (see NOMMA_CYC_PER_MAC): the linalg->loops
        # body is issue/latency bound per MAC, not CUDA-core-roofline bound
        macs = feats["ffma_flops_per_cta"] / 2.0
        terms["T_ffma"] = macs * NOMMA_CYC_PER_MAC / calib.m["clock_hz"]

    bw_key = "l2_bw" if l2_resident else "gmem_bw"
    terms["T_gmem"] = feats["gmem_bytes_per_cta"] / calib.per_sm(bw_key)
    terms["T_smem"] = feats["smem_bytes_per_cta"] / calib.per_sm("smem_bw")
    piped = feats["pipe"] > 0 or feats["ws"]
    compute = [terms["T_mma"] + terms["T_ffma"], terms["T_gmem"],
               terms["T_smem"]]
    terms["overlap"] = _overlap(compute, piped, alpha)

    terms["T_bar"] = feats["barriers_dyn"] * calib.m["bar_sync_ns"] * 1e-9
    spill_bytes = feats["spill_dyn_ops_per_cta"] * 4.0
    terms["T_spill"] = spill_bytes / (calib.per_sm("l2_bw") * SPILL_BW_FRAC)
    terms["T_ex2"] = feats["ex2_count_per_cta"] / calib.per_sm("ex2_rate")

    if form == "v1":
        total = (terms["T_launch"] + terms["overlap"] + terms["T_bar"]
                 + terms["T_spill"] + terms["T_ex2"])
        return total, terms, feats, used_index

    # v2+: LSU issue (warp instructions) + device-alloc terms
    lsu_ops = (feats["cpasync_warp_ops"] + feats["lds_warp_ops"]
               + feats["stg_warp_ops"])
    terms["T_lsu"] = lsu_ops / (LSU_IOPS_PER_CLK * calib.m["clock_hz"])
    terms["T_alloc"] = feats["n_allocs"] * calib.m["malloc_ns"] * 1e-9
    if form == "v2":
        total = (terms["T_launch"] + terms["overlap"] + terms["T_bar"]
                 + terms["T_spill"] + terms["T_ex2"] + terms["T_lsu"]
                 + terms["T_alloc"])
        return total, terms, feats, used_index

    # v3: occupancy / waves scaling
    occ = occupancy(feats, calib)
    total_ctas = feats["ctas_per_chunk"] * feats["n_chunks"]
    slots = calib.sm_count * occ
    waves = max(1, math.ceil(total_ctas / slots))
    per_cta = (terms["overlap"] + terms["T_bar"] + terms["T_spill"]
               + terms["T_ex2"] + terms["T_lsu"] + terms["T_alloc"])
    feats["waves"] = waves            # counts stay out of terms (time only)
    feats["ctas_per_sm_model"] = occ
    total = terms["T_launch"] + waves * per_cta
    return total, terms, feats, used_index


# ---------------------------------------------------------------------------
# Spearman (no scipy)
# ---------------------------------------------------------------------------

def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    """Pearson correlation of average ranks (tie-safe)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def topk_overlap(xs, ys, k=5):
    """|top-k(x) intersect top-k(y)| / k (smaller-is-better lists)."""
    def top(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])[:k]
        return set(idx)
    return len(top(xs) & top(ys)) / k


# ---------------------------------------------------------------------------
# config_times loading (tolerant)
# ---------------------------------------------------------------------------

_MS_KEYS = ("ms_median", "median_ms", "ms", "median", "time_ms", "mean_ms",
            "t_ms", "min_ms", "med_ms")
_S_KEYS = ("median_s", "time_s", "seconds", "s")
_META_KEYS = ("unit", "note", "description", "gpu", "date", "shared",
              "shared_gpu", "method", "n_iters", "iters", "reps", "warmup")

# records skipped by status filtering on the last load_config_times call:
# [(name, status), ...]
LAST_TIMES_SKIPPED = []


def load_config_times(path):
    """Tolerant config-times loader. Accepts:
      {name: ms} | {name: {"median_ms"|"ms"|...: v}}
      {"median"|"ms"|"times"|...: {name: ms}}   (nested name->ms mapping)
      {"configs"|"results": [{name, median_ms|...}]}
       Unit resolution: explicit *_ms / *_s key names win; a top-level
    "unit" field ("ms"|"s") applies to unitless leaf numbers; default ms.
    Records carrying a "status" field are only accepted when status=="OK"
    (G3-A marks deadlocking / not-timed kernels TIMEOUT / NOT_TIMED).
    """
    with open(path) as f:
        d = json.load(f)
    out = {}
    skipped_status = []

    unit = "ms"
    if isinstance(d, dict):
        u = str(d.get("unit", "")).lower()
        if u in ("s", "sec", "seconds"):
            unit = "s"

    def val_ms(v, key_hint=""):
        v = float(v)
        kh = key_hint.lower()
        if kh.endswith("_s") or kh in ("time_s", "seconds", "s"):
            return v * 1e3
        if kh.endswith("_us") or kh in ("us", "microseconds"):
            return v * 1e-3
        return v * 1e3 if unit == "s" else v

    def eat(node, container_is_times=False):
        if isinstance(node, list):
            for r in node:
                if isinstance(r, dict) and "name" in r:
                    accept_record(r, r["name"])
        elif isinstance(node, dict):
            for k, v in node.items():
                if k in _META_KEYS:
                    continue
                if isinstance(v, (int, float)):
                    out[k] = val_ms(v, "times" if container_is_times else "")
                elif isinstance(v, dict):
                    rec = any(kk in v for kk in _MS_KEYS + _S_KEYS)
                    if rec:
                        accept_record(v, k)
                    elif k.lower() in ("median", "times", "ms", "median_ms",
                                       "time_ms", "results", "configs",
                                       "measurements"):
                        eat(v, container_is_times=True)
                    # else: unrelated metadata dict -> skip

    def accept_record(rec, name):
        st = str(rec.get("status", "OK")).upper()
        if st not in ("OK", "", "TIMED"):
            skipped_status.append((name, st))
            return
        for kk in _MS_KEYS + _S_KEYS:
            if isinstance(rec.get(kk), (int, float)):
                out[name] = val_ms(rec[kk], kk)
                return

    # explicit "results" mapping (the G3-A shape) first
    if isinstance(d, dict) and isinstance(d.get("results"), dict):
        for name, rec in d["results"].items():
            if isinstance(rec, dict):
                accept_record(rec, name)
        global LAST_TIMES_SKIPPED
        LAST_TIMES_SKIPPED = skipped_status
        return out
    eat(d)
    LAST_TIMES_SKIPPED = skipped_status
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="GDN analytical cost model")
    ap.add_argument("--calibration",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "evidence", "G3",
                        "calibration.json"))
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    c = Calibration.load(args.calibration)
    print(f"source: {c.source}")
    print("fallback keys used:", sorted(c.fallback_used) or "none")
    print(json.dumps(c.table(), indent=1))
