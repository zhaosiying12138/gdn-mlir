#!/usr/bin/env python3
"""GDN Z3 knob solver (G3-B): TileLang z3_scheduler mechanism ported to
CUDA/sm_120 - v1 form: SMT enumeration/ordering of the DISCRETE KNOB SPACE
(not instruction start-time scheduling yet).

Mechanism (docs/PLAN.md section 3):
  variables   : the knob vector - tile {M,N,K}, warps, pipeline depth,
                dtype/precision, mma on/off, warpspec, tma.
  constraints : resource capacities as Z3 assertions over an explicit smem
                byte model, the regs/SM budget, mma fragment alignment,
                structural legality (tile divisibility, pipeline depth vs
                K-loop trip count, warpspec/tma consistency), knob coupling
                (mma off => no separate precision knob).
  objective   : minimize the calibrated analytical cost model
                (tools/gdn_costmodel.py). The cost model is non-linear
                (max/blend, divisions), so v1 enumerates every legal knob
                assignment with Z3 (SAT + blocking clauses) and ranks
                outside the solver - same role as TileLang's
                estimate_latency-guided II binary search, one level up.
  determinism : z3 fixed seed + rlimit cap, fixed variable declaration
                order, fixed blocking-clause order, ranking sorted by
                (predicted_time, name). No RNG anywhere else.
  loopback    : solve -> generate/compile top-K (gdn_l2_gen.py +
                scripts/run_l2.sh) -> ptxas -v static re-check ->
                re-rank with measured compile-time features -> (window)
                timing of the survivors.

smem byte model (bytes, per CTA; documented):
    esz           = 2 (bf16) | 4 (f32)
    pair          = (bt*bk + bk*bv) * esz                  [A+B tile]
    mma, ws=off   : smem = pair * max(pipeline,1) * n_gemm  (multibuffer
                    factor = pipeline; o stage allocates one pair per GEMM)
    mma, ws=on    : smem = pair + 64*64*esz (tma output staging) + 8
                    (mbarrier)          [single-buffered producer/consumer]
    mma=off       : smem = (bt*K + K*bv + bt*bv) * esz
    constraint    : smem <= 99KB - 1KB reserved  (mode "strict", sm_120
                    opt-in) or smem <= 220KB (mode "generator", mirrors
                    tools/gdn_l2_gen.py validate()).

regs constraint: threads * 255 <= 65536 per SM, threads = 32*(warps + 1 if
warpspec). This forbids e.g. warps=8 + producer warp (288*255 > 64K).

Usage:
  gdn_solver.py --solve --stage kkt [--K 128] [--B 4 --T 8192 --H 16]
                [--mode strict|generator] [--form v1|v2|v3]
                [--top 5] [--out evidence/G3/solver_kkt.json]
  gdn_solver.py --validate [--out evidence/G3/validation.json]
  gdn_solver.py --selftest        (analytic features vs INDEX static counts)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import z3  # noqa: E402

import gdn_costmodel as cm  # noqa: E402

DEFAULT_SEED = 7                 # fixed (determinism)
DEFAULT_RLIMIT = 20_000_000      # cap solver effort (determinism)

TILE_DOMAIN = (32, 64, 128)      # G3 knob-space definition (see report:
# gdn_l2_gen also admits M=16 and 8-multiples below 32; excluded here per
# the G3 knob-space contract)
WARPS_DOMAIN = (2, 4, 8)
PIPE_DOMAIN = (0, 2, 3)


# ---------------------------------------------------------------------------
# Z3 knob space
# ---------------------------------------------------------------------------

class KnobSpace:
    """SMT encoding of the legal knob space for one stage + dims."""

    def __init__(self, stage, K, V=None, mode="strict",
                 tile_domain=TILE_DOMAIN, seed=DEFAULT_SEED,
                 rlimit=DEFAULT_RLIMIT):
        assert stage in ("kkt", "o")
        assert mode in ("strict", "generator")
        self.stage, self.K, self.V, self.mode = stage, K, V, mode
        self.n_glob = 64 if stage == "kkt" else V
        self.n_gemm = 1 if stage == "kkt" else 2

        s = z3.Solver()
        # z3 5.1.0: the solver seed parameter is `random_seed`
        s.set("random_seed", seed)
        s.set("rlimit", rlimit)

        # -- decision variables (fixed declaration order => determinism) --
        dtype = z3.Int("dtype")            # 0 = bf16, 1 = f32
        prec = z3.Int("prec")              # 0 = bf16, 1 = tf32
        bt = z3.Int("tile_M")
        bv = z3.Int("tile_N")
        bk = z3.Int("tile_K")
        warps = z3.Int("warps")
        pipe = z3.Int("pipeline")
        mma = z3.Bool("mma")
        ws = z3.Bool("warpspec")
        tma = z3.Bool("tma")
        self.v = dict(dtype=dtype, prec=prec, bt=bt, bv=bv, bk=bk,
                      warps=warps, pipe=pipe, mma=mma, ws=ws, tma=tma)

        m_dom = [t for t in tile_domain if 64 % t == 0]
        n_dom = [t for t in tile_domain
                 if self.n_glob % t == 0 and t >= 8]
        k_dom = [t for t in tile_domain if K % t == 0 and t >= 8]
        s.add(z3.Or([dtype == d for d in (0, 1)]))
        s.add(z3.Or([bt == t for t in m_dom]))
        s.add(z3.Or([bv == t for t in n_dom]))
        s.add(z3.Or([bk == t for t in k_dom]))
        s.add(z3.Or([warps == w for w in WARPS_DOMAIN]))
        s.add(z3.Or([pipe == p for p in PIPE_DOMAIN]))

        # -- knob coupling ------------------------------------------------
        # precision is a real knob only for dtype=f32+mma (tf32); bf16 mma
        # is bf16-only, f32 is tf32-only, and mma=off removes the knob.
        s.add(z3.Implies(dtype == 0, prec == 0))
        s.add(z3.Implies(dtype == 1, prec == 1))
        s.add(z3.Implies(z3.Not(mma), prec == dtype))

        # -- structural legality (mirrors gdn_l2_gen.validate) -------------
        s.add(z3.Implies(mma, bt % 16 == 0))
        s.add(z3.Implies(z3.And(mma, dtype == 0), bk % 16 == 0))
        s.add(z3.Implies(z3.And(mma, dtype == 1), bk % 4 == 0))
        s.add(z3.Implies(mma, bv % (8 * warps) == 0))
        # pipeline depth <= K-loop trip count
        s.add(z3.Implies(z3.And(mma, pipe > 0), K >= bk * pipe))
        s.add(z3.Implies(z3.And(z3.Not(mma), pipe > 0), K >= bk))
        # warpspec: kkt bf16 mma only; tma requires warpspec
        ws_ok = z3.And(stage == "kkt", dtype == 0, mma)
        s.add(z3.Implies(ws, ws_ok))
        s.add(z3.Implies(tma, ws))
        # canonical form: the warpspec payload uses an identity transform
        # script, so `pipeline` is not observable there - pin it to the
        # built-in convention (p2) to avoid enumerating duplicate code
        s.add(z3.Implies(ws, pipe == 2))

        # -- resource capacities -------------------------------------------
        esz = z3.If(dtype == 0, 2, 4)
        threads = 32 * (warps + z3.If(ws, 1, 0))
        # regs: <=255/thread is a ptxas given (all family members hit the
        # cap); the SM-level budget is threads*255 <= 65536
        s.add(threads * 255 <= 65536)
        pair = (bt * bk + bk * bv) * esz
        smem_mma = z3.If(ws,
                         pair + z3.If(tma, 64 * 64 * esz, 0) + 8,
                         pair * z3.If(pipe > 0, pipe, 1) * self.n_gemm)
        smem_nomma = (bt * K + K * bv + bt * bv) * esz
        smem = z3.If(mma, smem_mma, smem_nomma)
        cap = (cm.SMEM_USABLE_B if mode == "strict" else 220 * 1024)
        s.add(smem <= cap)
        self.smem_expr = smem

        # completion candidates for the documented z3 partial-model
        # artifact (G6/search_cost.md: a knob const the model OMITS reads
        # back as None through m[zv]; tools/gdn_solver.py enumerate read
        # used to crash on it).  Fixed order => deterministic completion.
        self.rlimit = rlimit
        self.domains = {
            "dtype": (0, 1), "prec": (0, 1), "bt": tuple(m_dom),
            "bv": tuple(n_dom), "bk": tuple(k_dom),
            "warps": WARPS_DOMAIN, "pipe": PIPE_DOMAIN,
            "mma": (False, True), "ws": (False, True), "tma": (False, True),
        }
        self.solver = s

    # -- enumeration -------------------------------------------------------
    def _complete(self, key, fixed):
        """Deterministic completion of a knob const the model omitted.

        First value from the declared domain (fixed order) that keeps the
        rest of the model assignment satisfiable -- check-and-assert, so an
        omitted knob can never inject an illegal value into the space.  The
        probe runs with the rlimit lifted: an rlimit abort inside the probe
        must not be mistaken for "candidate rejected".  At least one
        candidate is always legal (the model the solver returned was sat).
        """
        s = self.solver
        saved_rlimit = self.rlimit
        s.set("rlimit", 0)
        s.push()
        try:
            pin = [self.v[k] == (z3.BoolVal(v) if isinstance(v, bool)
                                 else z3.IntVal(v))
                   for k, v in fixed.items()]
            s.add(z3.And(pin))
            for cand in self.domains[key]:
                s.push()
                s.add(self.v[key] == (z3.BoolVal(cand)
                                      if isinstance(cand, bool)
                                      else z3.IntVal(cand)))
                ok = s.check() == z3.sat
                s.pop()
                if ok:
                    return cand
            raise RuntimeError(
                f"z3 partial model: no legal completion for knob {key!r} "
                f"given {fixed!r} (space {self.stage} K={self.K} "
                f"V={self.V}); this is a solver contract violation -- "
                f"please report with the full command line")
        finally:
            s.pop()
            s.set("rlimit", saved_rlimit)

    def enumerate(self, limit=100000):
        """All legal knob vectors (deterministic order).

        Robust against the G6 partial-model artifact: knob consts the
        model omits (m[zv] -> None; z3py get_interp, not an exception)
        are completed via _complete(); the blocking clause is built from
        the completed PYTHON values (never from m[zv] directly, which is
        None exactly in the artifact case).  self.enum_stats records how
        often the path fired; self.last_check records how enumeration
        ended ("unsat" = clean exhaustion; "unknown" = rlimit abort ->
        solve() retries with the cap lifted).
        """
        s = self.solver
        out = []
        stats = {"models": 0, "partial_models": 0,
                 "probe_completions": 0}
        while len(out) < limit:
            r = s.check()
            if r != z3.sat:
                self.last_check = str(r)
                break
            m = s.model()
            stats["models"] += 1
            vals, missing = {}, []
            for k, zv in self.v.items():
                val = m[zv]
                if val is None:
                    stats["partial_models"] += 1
                    missing.append(k)          # completed below
                    continue
                if z3.is_true(val):
                    vals[k] = True
                elif z3.is_false(val):
                    vals[k] = False
                else:
                    vals[k] = val.as_long()
            for k in missing:                  # declaration order
                vals[k] = self._complete(k, vals)
                stats["probe_completions"] += 1
            out.append(vals)
            block = [self.v[k] != (z3.BoolVal(v) if isinstance(v, bool)
                                   else z3.IntVal(v))
                     for k, v in vals.items()]
            s.add(z3.Or(block))
        self.enum_stats = stats
        return out

    def to_config(self, kv):
        cfg = {
            "stage": self.stage,
            "dtype": "bf16" if kv["dtype"] == 0 else "f32",
            "prec": "bf16" if kv["prec"] == 0 else "tf32",
            "dims": {"K": self.K, "V": self.V if self.stage == "o" else 64},
            "tile": {"M": kv["bt"], "N": kv["bv"], "K": kv["bk"]},
            "warps": kv["warps"],
            "mma": kv["mma"],
            "pipeline": kv["pipe"],
            "warpspec": "producer_consumer" if kv["ws"] else "off",
            "tma": "on" if kv["tma"] else "off",
        }
        cfg["name"] = config_name(cfg)
        return cfg


def config_name(cfg):
    st = cfg["stage"]
    n = (f"{st}_{cfg['dtype']}_{'mma' if cfg['mma'] else 'nomma'}"
         f"_p{cfg['pipeline']}_t{cfg['tile']['M']}x{cfg['tile']['N']}"
         f"x{cfg['tile']['K']}_w{cfg['warps']}")
    if cfg.get("warpspec") == "producer_consumer":
        n += "_ws"
        if cfg.get("tma") == "on":
            n += "_tma"
    if st == "o":
        n += f"_K{cfg['dims']['K']}V{cfg['dims']['V']}"
    return n


def knob_vector(cfg):
    return (cfg["dtype"], cfg.get("prec", "bf16" if cfg["dtype"] == "bf16"
                                  else "tf32"),
            cfg["tile"]["M"], cfg["tile"]["N"], cfg["tile"]["K"],
            cfg["warps"], cfg["mma"], cfg["pipeline"],
            cfg.get("warpspec", "off"), cfg.get("tma", "off"),
            cfg["dims"]["K"], cfg.get("dims", {}).get("V"))


# ---------------------------------------------------------------------------
# solve: enumerate + rank
# ---------------------------------------------------------------------------

def load_index(path=None):
    path = path or os.path.join(REPO, "evidence", "G2", "gen", "INDEX.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    return {c["name"]: c for c in d.get("configs", [])}


def load_g3_features(root=None):
    """evidence/G3/gen_noargs/<name>/meta.json - the G3-A rebuild of the 20
    configs that the timing harness actually measures (static-smem repaired
    cubins, fresh ptxas resources). Preferred over the G2 INDEX for
    validation because the timed cubins differ (regs/spills re-measured;
    the G2 malloc-based INDEX resources are stale for them)."""
    root = root or os.path.join(REPO, "evidence", "G3", "gen_noargs")
    out = {}
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        mp = os.path.join(root, name, "meta.json")
        if not os.path.isfile(mp):
            continue
        try:
            with open(mp) as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        cfg = {k: m[k] for k in ("stage", "dtype", "dims", "tile", "warps",
                                 "mma", "pipeline", "prec", "warpspec",
                                 "tma") if k in m}
        if "stage" not in cfg:
            continue
        cfg["name"] = name
        out[name] = {"name": name, "config": cfg, "meta": m,
                     "smemfix": True,
                     "path": os.path.join(root, name)}
    return out


def index_lookup(index, cfg):
    """Match an enumerated config to an INDEX entry by full knob vector."""
    target = knob_vector(cfg)
    for name, e in index.items():
        if e.get("config") and knob_vector(e["config"]) == target:
            return e
    return None


def solve(stage="kkt", K=128, V=None, B=1, T=64, H=1, mode="strict",
          form="v2", top=None, index=None, calib=None, limit=100000):
    t0 = time.time()
    space = KnobSpace(stage, K, V=V, mode=mode)
    combos = space.enumerate(limit=limit)
    rl_retry = False
    if getattr(space, "last_check", "unsat") == "unknown":
        # rlimit abort BEFORE the space was exhausted: silently accepting
        # it would return a truncated space as if complete (the G6 flake
        # class).  Retry once with the cap lifted -- these spaces are
        # small (<= 1e3 configs), the uncapped pass costs well under a
        # second, and the retry is deterministic.
        rl_retry = True
        space = KnobSpace(stage, K, V=V, mode=mode, rlimit=0)
        combos = space.enumerate(limit=limit)
        if getattr(space, "last_check", "unsat") == "unknown":
            raise RuntimeError(
                f"solver returned unknown even without the rlimit cap "
                f"({stage} K={K} V={V}); refusing to emit a truncated "
                f"space of {len(combos)} configs")
    t_enum = time.time() - t0
    calib = calib or cm.Calibration()

    n_chunks = max(1, B * H * ((T + 63) // 64))
    ranked = []
    for kv in combos:
        cfg = space.to_config(kv)
        entry = index_lookup(index, cfg) if index else None
        total, terms, feats, used = cm.predict(
            cfg, calib, form=form, n_chunks=n_chunks,
            index_entry=entry)
        ranked.append({
            "name": cfg["name"],
            "config": cfg,
            "predicted_ms": total * 1e3,
            "terms_ms": {k: v * 1e3 for k, v in terms.items()},
            "smem_model_bytes": feats["smem_static_model"],
            "has_index_features": bool(used),
        })
    ranked.sort(key=lambda r: (round(r["predicted_ms"], 12), r["name"]))
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    out = {
        "stage": stage, "dims": {"K": K, "V": V}, "shape": {"B": B, "T": T,
                                                            "H": H},
        "n_chunks": n_chunks,
        "constraints_mode": mode, "model_form": form,
        "space_size": len(combos),
        "enumeration_seconds": round(t_enum, 3),
        "z3": {"seed": DEFAULT_SEED, "rlimit": DEFAULT_RLIMIT,
               "last_check": getattr(space, "last_check", "sat-exhausted"),
               "rlimit_retry_uncapped": rl_retry,
               "enum_stats": getattr(space, "enum_stats", None)},
        "calibration_source": calib.source,
        "calibration_fallback_keys": sorted(calib.fallback_used),
        "ranking": ranked[:top] if top else ranked,
    }
    return out


# ---------------------------------------------------------------------------
# validate: Spearman gate vs measured times
# ---------------------------------------------------------------------------

def validate(index_path=None, times_path=None, calib_path=None,
             forms=("v1", "v2", "v3"), gate=0.7, verbose=True,
             g3_root=None, launch_from_anchor=True):
    index = load_index(index_path)
    g3feats = load_g3_features(g3_root)
    times_path = times_path or os.path.join(REPO, "evidence", "G3",
                                            "config_times.json")
    calib_path = calib_path or os.path.join(REPO, "evidence", "G3",
                                            "calibration.json")
    times = cm.load_config_times(times_path)
    calib = cm.Calibration.load(calib_path)

    # Model iteration "launch anchor": the calibration file does not
    # measure launch overhead; the ref_f16 anchor is a pure-launch kernel
    # (its non-launch terms are ~0.05us). Deriving the SINGLE GLOBAL
    # launch constant from it is the contract's allowed launch constant;
    # the anchor is additionally reported with-and-without in the report.
    launch_anchor = None
    if launch_from_anchor and "ref_f16_sm80_atom" in times:
        entry = g3feats.get("ref_f16_sm80_atom") or \
            index.get("ref_f16_sm80_atom")
        if entry:
            _, terms, _, _ = cm.predict(entry["config"], calib, form="v2",
                                        index_entry=entry)
            non_launch = sum(v for k, v in terms.items()
                             if k != "T_launch" and isinstance(v, float))
            launch_anchor = max(0.0, times["ref_f16_sm80_atom"] * 1e-3
                                - non_launch)
            calib.m["launch_ns"] = launch_anchor * 1e9
            calib.fallback_used.discard("launch_ns")

    def feat_entry(n):
        # prefer the G3-A rebuild (what the timing harness measures),
        # fall back to the G2 INDEX entry
        return g3feats.get(n) or index.get(n)

    names = [n for n in times if feat_entry(n)]
    missing_idx = [n for n in times if not feat_entry(n)]
    measured = [times[n] for n in names]

    per_config = []
    preds = {f: [] for f in forms}
    for n in names:
        entry = feat_entry(n)
        row = {"name": n, "stage": entry["config"]["stage"],
               "feature_source": ("g3_gen_noargs" if entry.get("smemfix")
                                  else "g2_index"),
               "measured_ms": times[n]}
        for f in forms:
            total, terms, feats, used = cm.predict(
                entry["config"], calib, form=f, n_chunks=1,
                index_entry=entry)
            row[f"pred_{f}_ms"] = total * 1e3
            preds[f].append(total * 1e3)
            if f == forms[-1]:
                row["terms_ms"] = {k: v * 1e3 for k, v in terms.items()}
        per_config.append(row)

    report = {"gate_threshold": gate, "n_measured": len(names),
              "n_missing_index_entry": missing_idx,
              "n_g3_rebuild_features": len(g3feats),
              "times_skipped_by_status": getattr(
                  cm, "LAST_TIMES_SKIPPED", []),
              "launch_anchor_s": launch_anchor,
              "calibration_source": calib.source,
              "calibration_table": calib.table(),
              "calibration_fallback_keys": sorted(calib.fallback_used),
              "shared_gpu": calib.m.get("shared_gpu"),
              "forms": {}, "per_config": per_config}
    for f in forms:
        rho = cm.spearman(preds[f], measured)
        stages = sorted({r["stage"] for r in per_config})
        per_stage = {}
        for st in stages:
            xs = [r[f"pred_{f}_ms"] for r in per_config if r["stage"] == st]
            ys = [r["measured_ms"] for r in per_config if r["stage"] == st]
            per_stage[st] = (cm.spearman(xs, ys) if len(xs) >= 3 else None)
        report["forms"][f] = {
            "spearman": rho,
            "spearman_per_stage": per_stage,
            "top5_overlap": cm.topk_overlap(preds[f], measured, 5),
        }
    best = max(forms, key=lambda f: report["forms"][f]["spearman"])
    report["best_form"] = best
    report["verdict"] = ("PASS" if report["forms"][best]["spearman"] >= gate
                         else "FAIL")

    if verbose:
        print(f"[validate] measured configs: {len(names)} "
              f"(missing INDEX entry: {len(missing_idx)}; "
              f"skipped by status: {getattr(cm, 'LAST_TIMES_SKIPPED', [])})")
        print(f"[validate] calibration: {calib.source} "
              f"(fallback keys: {sorted(calib.fallback_used) or 'none'})")
        print(f"[validate] shared_gpu flag: {calib.m.get('shared_gpu')}")
        for f in forms:
            r = report["forms"][f]
            ps = {k: (round(v, 3) if v is not None else None)
                  for k, v in r["spearman_per_stage"].items()}
            print(f"[validate] form {f}: Spearman={r['spearman']:.4f} "
                  f"top5_overlap={r['top5_overlap']:.2f} per-stage={ps}")
        print(f"[validate] best form: {best} -> "
              f"{report['forms'][best]['spearman']:.4f} "
              f"(gate {gate}) => {report['verdict']}")
    return report


# ---------------------------------------------------------------------------
# selftest: analytic feature formulas vs INDEX static PTX counts
# ---------------------------------------------------------------------------

def selftest(index_path=None, verbose=True):
    """Check the analytic per-iteration count formulas against the static
    PTX counts in INDEX.json (peel factor (1+pipeline) applied for the
    software-pipelined mma path, verified on the 18 GDN configs)."""
    index = load_index(index_path)
    rows = []
    for name, e in sorted(index.items()):
        cfg = e["config"]
        if cfg["stage"] in ("ref", "ref_ldmatrix"):
            continue
        feats = cm.analytic_features(cfg)
        pf = e.get("meta", {}).get("ptx_features", {})
        ws = cfg.get("warpspec") == "producer_consumer"
        tma = cfg.get("tma", "off") == "on"
        # peel factor: pipeline_shared_memory_copies duplicates the K-loop
        # body (1+pipeline)x in the PTX; the warpspec path keeps a plain
        # scf.for (identity transform script) -> no duplication
        peel = (1 + cfg["pipeline"]) if (cfg["mma"] and not ws) else 1

        atoms_iter = (feats["atoms_per_warp"]
                      // max(1, feats["trips"]) // feats["n_gemm"])
        exp_mma = atoms_iter * feats["n_gemm"] * peel

        ch = 8 if cfg["dtype"] == "bf16" else 4
        cp_iter = (feats["bt"] + feats["bv"]) * (feats["bk"] // ch)
        if tma:
            exp_cp = None    # tma.async.bulk replaces the cp.async stream
        else:
            exp_cp = cp_iter * feats["n_gemm"] * peel

        if cfg["mma"]:
            a_ld, b_ld = (8, 4) if cfg["dtype"] == "bf16" else (2, 1)
            ak = 16 if cfg["dtype"] == "bf16" else 4
            exp_lds = ((feats["bt"] // 16) * (feats["bk"] // ak) * a_ld
                       + ((feats["bv"] // cfg["warps"]) // 8)
                       * (feats["bk"] // ak) * b_ld)
            exp_lds = exp_lds * feats["n_gemm"] * (1 if ws else peel)
        else:
            exp_lds = pf.get("ld.shared") or 0   # not modeled (tiny)

        if ws:
            exp_bar = pf.get("bar.sync") or 0     # 1 bar + mbarrier ops
        elif cfg["mma"]:
            exp_bar = 2 * feats["n_gemm"] * peel
        else:
            exp_bar = 2 * feats["n_gemm"]

        rows.append({
            "name": name,
            "mma": (exp_mma, pf.get("mma.sync") or 0),
            "cpasync": (exp_cp, pf.get("cp.async") or 0),
            "lds": (exp_lds, pf.get("ld.shared") or 0),
            "bar": (exp_bar, pf.get("bar.sync") or 0),
        })
    if verbose:
        print(f"{'config':40s} {'mma model/index':>15s} "
              f"{'cp.async model/index':>21s} {'lds model/index':>15s} "
              f"{'bar model/index':>15s}")
        for r in rows:
            cp = r["cpasync"]
            cp_s = "tma" if cp[0] is None else f"{cp[0]:9d}"
            print(f"{r['name']:40s} {r['mma'][0]:6d}/{r['mma'][1]:<8d} "
                  f"{cp_s}/{cp[1]:<10d} "
                  f"{r['lds'][0]:6d}/{r['lds'][1]:<8d} "
                  f"{r['bar'][0]:5d}/{r['bar'][1]:<8d}")
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="GDN Z3 knob solver (G3-B)")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--stage", default="kkt", choices=("kkt", "o"))
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--V", type=int, default=None)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--mode", default="strict",
                    choices=("strict", "generator"))
    ap.add_argument("--form", default="v2", choices=("v1", "v2", "v3"))
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--index", default=None)
    ap.add_argument("--times", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--gate", type=float, default=0.7)
    args = ap.parse_args()

    if args.selftest:
        selftest(args.index)
        return 0
    if args.validate:
        rep = validate(index_path=args.index, times_path=args.times,
                       calib_path=args.calibration, gate=args.gate)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(rep, f, indent=1)
            print(f"[validate] report -> {args.out}")
        return 0 if rep["verdict"] == "PASS" else 1
    if args.solve:
        K = args.K if args.K is not None else (128 if args.stage == "kkt"
                                               else 64)
        V = args.V if args.V is not None else (64 if args.stage == "o"
                                               else None)
        out = solve(stage=args.stage, K=K, V=V, B=args.B, T=args.T, H=args.H,
                    mode=args.mode, form=args.form, top=args.top,
                    index=load_index(args.index), limit=args.limit)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
        print(f"[solve] stage={args.stage} mode={args.mode} form={args.form} "
              f"legal space={out['space_size']} "
              f"(enum {out['enumeration_seconds']}s)")
        for r in out["ranking"][:args.top or 10]:
            print(f"  #{r['rank']:3d} {r['name']:46s} "
                  f"{r['predicted_ms']*1e3:9.2f} us")
        if args.out:
            print(f"[solve] ranking -> {args.out}")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
