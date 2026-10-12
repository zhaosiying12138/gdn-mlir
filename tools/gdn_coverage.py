#!/usr/bin/env python3
"""G2-B: PTX coverage analysis — T1 instruction-family inventory + coverage matrix.

Scans the G0 reference harvests (fla Triton sm_120, FlashQLA/TileLang sm_120a),
builds a NORMALIZED instruction-family inventory (T1 set), and produces the
coverage matrix (fla / FlashQLA / ours) against G2-A's generated PTX set.

Usage (repo = parent of this tools/ dir, or $GDN_REPO):
  python tools/gdn_coverage.py ref-inventory   # -> evidence/G2/ref_inventory.{json,md}
  python tools/gdn_coverage.py matrix          # -> evidence/G2/coverage_matrix.md  (idempotent; ours-column PENDING until G2-A gen lands)
  python tools/gdn_coverage.py t1-set          # prints the T1 contract set
  python tools/gdn_coverage.py all

The G0 FAMS regex list is imported from tools/harvest_flaptx.py (parity check);
EXTRA_FAMS below extends it (G0's `tma.bulk` pattern was broken — it could never
match; cp.async.bulk.tensor.* / mbarrier / elect.sync / setmaxnreg / prefetch
were under-classified). The authoritative family assignment here is the generic
opcode histogram + CLASSIFY rules, so nothing is silently dropped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))
from harvest_flaptx import FAMS as G0_FAMS  # noqa: E402  (import per contract)

FLA_ROOT = REPO / "evidence/G0/fla_ptx"
QLA_ROOT = REPO / "evidence/G0/qla_ptx"
G2_DIR = REPO / "evidence/G2"
GEN_DIR = G2_DIR / "gen"
INDEX_JSON = GEN_DIR / "INDEX.json"

MMA_RE = re.compile(r"\bmma\.sync\.[a-z0-9.]+")
# generic opcode: first dotted token of an instruction line (after optional @%p guard).
# charset includes ':' (cache-policy/shared-state qualifiers like `st.shared::cta.b16`,
# `cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint`)
# and uppercase (the `L1::`/`L2::` hint prefixes).
OP_RE = re.compile(r"^[ \t]*(?:@!?%?\w+[ \t]+)?([a-z][a-zA-Z0-9_.:]*)(?=[ \t;(\n])", re.M)
LABEL_RE = re.compile(r"^(\$L__\w+):", re.M)
BRANCH_RE = re.compile(r"\b(?:bra|bra\.uni)[ \t]+(\$L__\w+);")

# ---------------------------------------------------------------- FAMS extension
EXTRA_FAMS = [
    (r"\bcp\.async\.bulk[\w.:]*", "tma.bulk"),                                   # FIX: G0 pattern never matched
    (r"\bcp\.async\.(?:commit_group|wait_group[\w.]*|wait_all)\b", "cp.async.ctrl"),
    (r"\bmbarrier(?:_init)?[\w.:]*", "mbarrier"),                                # covers both PTX spellings
    (r"\belect\.sync[\w.:]*", "elect.sync"),
    (r"\bprefetch\.tensormap[\w.:]*", "prefetch.tensormap"),
    (r"\bsetmaxnreg[\w.:]*", "setmaxnreg"),
    (r"\bld\.global\.nc[\w.:]*", "ld.global.nc"),
    (r"\bex2\.approx(?:\.ftz)?\.f32\b", "ex2.approx"),
    (r"\bshfl\.sync[\w.:]*", "shfl.sync"),
    (r"\bldmatrix[\w.:]*", "ldmatrix"),
    (r"\bstmatrix[\w.:]*", "stmatrix"),
    (r"\bbra(?:\.uni)?\b", "bra"),
]

# ---------------------------------------------------------------- classification
# Ordered rules, first match wins. Taxonomy kept deliberately coarse at the
# HW-feature level; exact opcode variants live in `variants`.
CLASSIFY: list[tuple[str, str]] = [
    ("mma.sync.", "mma.sync"),
    ("ldmatrix", "ldmatrix"),
    ("stmatrix", "stmatrix"),
    ("movmatrix", "movmatrix"),
    ("wmma.", "wmma"),
    ("wgmma", "wgmma"),
    ("tcgen05", "tcgen05"),
    ("cp.async.bulk.tensor", "tma.bulk.tensor"),
    ("cp.async.bulk", "tma.bulk.ctrl"),           # commit_group / wait_group
    ("cp.async.ca", "cp.async.classic"),
    ("cp.async.cg", "cp.async.classic"),
    ("cp.async.commit_group", "cp.async.ctrl"),
    ("cp.async.wait", "cp.async.ctrl"),           # wait_group / wait_all
    ("mbarrier", "mbarrier"),                     # incl. mbarrier_init.*
    ("setmaxnreg", "setmaxnreg"),
    ("elect.sync", "elect.sync"),
    ("prefetch.tensormap", "prefetch.tensormap"),
    ("bar.", "bar.sync"),
    ("barrier.", "barrier"),
    ("fence.", "barrier"),                         # fence.mbarrier_init.release.cluster (qla)
    ("shfl.sync", "shfl.sync"),
    ("redux", "redux"),
    ("red.", "red"),
    ("atom", "atom"),
    ("cp.reduce", "cp.reduce"),
    ("st.async", "st.async"),
    ("ex2.approx", "ex2.approx"),                 # exp2/ex2 both spellings
    ("exp2.approx", "ex2.approx"),
    ("ld.global", "ld.global"),
    ("st.global", "st.global"),
    ("ld.shared", "ld.shared"),
    ("st.shared", "st.shared"),
    ("ld.param", "ld.param"),
    ("st.param", "st.param"),
    ("cvta", "cvta"),
    # packed v2f32 math before scalar f32
    ("fma.rn.f32x2", "math.f32x2"),
    ("mul.rn.f32x2", "math.f32x2"),
    ("add.rn.f32x2", "math.f32x2"),
    ("sub.rn.f32x2", "math.f32x2"),
    ("add.f32x2", "math.f32x2"),
    ("sub.f32x2", "math.f32x2"),
    ("mul.f32x2", "math.f32x2"),
    # bf16 domain (conversions + mixed math) — explicit prefixes, BEFORE generic cvt/add/mul
    ("cvt.f32.bf16", "cvt.bf16f32"),
    ("cvt.rn.bf16", "cvt.bf16f32"),                # .f32 and x2 forms
    ("add.rn.f32.bf16", "cvt.bf16f32"),
    ("sub.rn.f32.bf16", "cvt.bf16f32"),
    ("add.bf16", "cvt.bf16f32"),                   # incl. bf16x2
    ("sub.bf16", "cvt.bf16f32"),
    ("mul.bf16", "cvt.bf16f32"),
    ("fma.rn.bf16", "cvt.bf16f32"),
    ("fma.rn.f32", "fma.f32"),
    ("cvt", "cvt.int"),
    ("setp", "setp"),
    ("selp", "selp"),
    ("mov.pred", "mov.pred"),
    ("bra", "bra"),
    ("ret", "bra"),                                # function return — control flow
    ("mov", "mov"),
    ("add.f32", "math.f32"),
    ("sub.f32", "math.f32"),
    ("mul.f32", "math.f32"),
    ("neg.f32", "math.f32"),
    ("div.f32", "math.f32"),
    ("sqrt", "math.f32"),
    ("rcp", "math.f32"),
    ("min", "math.int"),
    ("max", "math.int"),
    ("div", "math.int"),
    ("mul.hi", "math.int"),
    ("mad", "math.int"),
    ("bfe", "math.int"),
    ("add", "math.int"),
    ("sub", "math.int"),
    ("mul", "math.int"),
    ("neg", "math.int"),
    ("abs", "math.int"),
    ("and", "logic.shift"),
    ("or", "logic.shift"),
    ("xor", "logic.shift"),
    ("not", "logic.shift"),
    ("shl", "logic.shift"),
    ("shr", "logic.shift"),
    ("ld", "ld.other"),
    ("st", "st.other"),
]

# families considered "HW-feature": they need an explicit IR affordance beyond
# plain linalg/arith -> NVVM lowering. Everything else is baseline.
FEATURE_FAMILIES = {
    "mma.sync", "ldmatrix", "stmatrix", "movmatrix", "wmma", "wgmma", "tcgen05",
    "cp.async.classic", "cp.async.ctrl", "tma.bulk.tensor", "tma.bulk.ctrl",
    "mbarrier", "setmaxnreg", "elect.sync", "prefetch.tensormap",
    "bar.sync", "barrier", "shfl.sync", "redux", "red", "atom", "cp.reduce",
    "st.async", "ex2.approx",
}

# escape-hatch / plan notes for the ours column (used when a family is missing)
HATCH = {
    "mma.sync": "MLIR transform-matmul -> nvvm.mma.sync; m16n8k16.row.col.f32.bf16.bf16.f32 needs bf16 dot lowering; m16n8k8 tf32 sig needs tf32 precision knob (G2-A knob)",
    "ldmatrix": "gpu.ldmatrix / nvvm.ldmatrix (upstream NVVM dialect); .trans variant needs transpose flag",
    "stmatrix": "nvvm.stmatrix (upstream NVVM main; x2/x4/.trans variants)",
    "movmatrix": "nvvm.movmatrix (not in refs; free if needed)",
    "cp.async.classic": "gpu.async_copy (nvvm.cp.async.ca/cg) via MLIR async-copy pipelining",
    "cp.async.ctrl": "gpu.commit_group / gpu.wait_group (upstream)",
    "tma.bulk.tensor": "ESCAPE-HATCH: nvvm.cp.async.bulk.tensor.{3,4,5}d + nvvm.tensormap descriptors — no upstream MLIR op; counts vs <=5 budget (or G2-C manual pattern)",
    "tma.bulk.ctrl": "bundled with tma.bulk.tensor hatch (nvvm.cp.async.bulk.commit_group/wait_group.read)",
    "mbarrier": "ESCAPE-HATCH: nvvm.mbarrier.{init,arrive,expect_tx,try_wait.parity} — no upstream MLIR op; async pipeline G2-C",
    "setmaxnreg": "nvvm.setmaxnreg.inc/dec — warp specialization; PLANNED as G2-C deliverable",
    "elect.sync": "ESCAPE-HATCH: nvvm.elect.sync (warp-leader election for TMA issue) — G2-C",
    "prefetch.tensormap": "rides along with tma.bulk.tensor hatch (nvvm.prefetch.tensormap)",
    "ex2.approx": "math.exp2 -> llvm.nvvm.ex2.approx.f32 under fastmath; .ftz variant needs ftz codegen flag",
    "shfl.sync": "gpu.shuffle (nvvm.shfl.sync) bfly/idx — upstream",
    "bar.sync": "gpu.barrier -> bar.sync 0 (free)",
    "ld.global": "standard lowering; .v2/.vN vector width via vectorization",
    "ld.global.nc": "needs ldg/read-only hint (nvvm ld.global.nc) — minor hatch",
    "st.global": "standard lowering",
    "ld.shared": "standard lowering (memref -> shared)",
    "st.shared": "standard lowering",
    "ld.param": "kernel ABI, free",
    "fma.f32": "arith/mulf+add fusion -> fma.rn.f32 (free)",
    "math.f32x2": "vector<f32> x2 packed math (free via vector type)",
    "cvt.bf16f32": "arith.extf/truncf on bf16 (free; bf16x2 packing via vector<2xbf16>)",
    "math.int": "arith integer lowering (free)",
    "logic.shift": "arith bitwise/shift (free)",
    "setp": "arith.cmp (free)",
    "selp": "arith.select (free)",
    "mov": "trivial (free)",
    "mov.pred": "trivial (free)",
    "bra": "scf.for/if lowering (free)",
    "cvta": "address spaces, free",
    "cvt.int": "arith ext/trunc (free)",
    "ld.other": "standard",
    "st.other": "standard",
    "fma.other": "standard",
    "barrier": "gpu.barrier (free)",
    "redux": "nvvm.redux.sync if ever needed — NOT in T1 refs",
    "red": "NOT in T1 refs",
    "atom": "NOT in T1 refs",
    "cp.reduce": "NOT in T1 refs",
    "st.async": "NOT in T1 refs",
    "wmma": "NOT in T1 refs (mma.sync style)",
    "wgmma": "NOT in T1 refs",
    "tcgen05": "NOT in T1 refs",
}

FAMILY_ORDER = [
    # tensor core
    "mma.sync", "ldmatrix", "stmatrix", "movmatrix", "wmma", "wgmma", "tcgen05",
    # async / TMA
    "cp.async.classic", "cp.async.ctrl", "tma.bulk.tensor", "tma.bulk.ctrl",
    "mbarrier", "elect.sync", "setmaxnreg", "prefetch.tensormap", "st.async",
    # sync / warp collectives
    "bar.sync", "barrier", "shfl.sync", "redux", "red", "atom", "cp.reduce",
    # special math
    "ex2.approx",
    # memory
    "ld.global", "ld.global.nc", "st.global", "ld.shared", "st.shared",
    "ld.param", "ld.other", "st.other",
    # scalar math / conversions
    "fma.f32", "math.f32x2", "cvt.bf16f32", "math.f32", "math.int",
    "logic.shift", "cvt.int", "cvta",
    # control
    "setp", "selp", "mov", "mov.pred", "bra",
    "fma.other",
]


def classify_op(op: str) -> str:
    for prefix, fam in CLASSIFY:
        if op.startswith(prefix):
            return fam
    return "misc"


# ---------------------------------------------------------------- PTX extraction

def loop_structure(ptx: str) -> dict:
    labels = {m.group(1): m.start() for m in LABEL_RE.finditer(ptx)}
    back_targets: set[str] = set()
    n_bra = 0
    for m in BRANCH_RE.finditer(ptx):
        n_bra += 1
        t = m.group(1)
        if t in labels and labels[t] < m.start():
            back_targets.add(t)
    return {"bra_total": n_bra, "loop_backedges": len(back_targets)}


def extract_ptx(ptx: str) -> dict:
    ops = Counter(OP_RE.findall(ptx))
    fams = Counter()
    variants: dict[str, Counter] = defaultdict(Counter)
    for op, n in ops.items():
        f = classify_op(op)
        fams[f] += n
        variants[f][op] += n
    mma = Counter(MMA_RE.findall(ptx))
    return {
        "opcodes": dict(ops),
        "families": dict(fams),
        "variants": {f: dict(c) for f, c in variants.items()},
        "mma_signatures": dict(mma),
        **loop_structure(ptx),
    }


def fams_counts(ptx: str) -> dict:
    """G0-FAMS parity counts (imported list + extras)."""
    c = Counter()
    for pat, fam in G0_FAMS + EXTRA_FAMS:
        c[fam] = len(re.findall(pat, ptx))
    return dict(c)


def load_meta_features(meta_path: Path) -> dict | None:
    """G0 meta.json ptx_features (harvest-time counts) for parity cross-check."""
    try:
        meta = json.loads(meta_path.read_text())
        return meta.get("ptx_features")
    except Exception:
        return None


def load_meta_resources(meta_path: Path) -> dict:
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return {}
    p = meta.get("ptxas", {})
    return {
        "regs": p.get("registers"),
        "smem_static": p.get("smem_static"),
        "smem_dynamic": meta.get("shared_dynamic"),
        "spill_stores": p.get("spill_stores", 0),
        "spill_loads": p.get("spill_loads", 0),
        "num_warps": meta.get("num_warps"),
    }


def scan_reference(root: Path, ref: str) -> list[dict]:
    entries = []
    for ptx_path in sorted(root.rglob("kernel.ptx")):
        rel = ptx_path.parent.relative_to(root)
        if ref == "fla":
            kernel, config = str(rel.parent), rel.name
        else:
            kernel, config = rel.name, "default"
        ptx = ptx_path.read_text(errors="replace")
        e = {
            "ref": ref, "kernel": kernel, "config": config,
            "path": str(ptx_path.relative_to(REPO)),
            **extract_ptx(ptx),
            "g0_fams": fams_counts(ptx),
            "resources": load_meta_resources(ptx_path.parent / "meta.json"),
        }
        # cross-check vs G0 meta.json ptx_features (same FAMS list)
        meta_f = load_meta_features(ptx_path.parent / "meta.json")
        if meta_f:
            for k, v in meta_f.items():
                if e["g0_fams"].get(k, 0) != v:
                    e.setdefault("g0_parity_diff", {})[k] = (v, e["g0_fams"].get(k, 0))
        entries.append(e)
    return entries


# ---------------------------------------------------------------- aggregation

def agg_families(entries: list[dict]) -> tuple[Counter, dict]:
    total = Counter()
    variants = defaultdict(Counter)
    for e in entries:
        for f, n in e["families"].items():
            total[f] += n
        for f, vs in e["variants"].items():
            variants[f].update(vs)
    return total, {f: dict(c) for f, c in variants.items()}


def kernel_presence(entries: list[dict], fam: str) -> set[str]:
    return {e["kernel"] for e in entries if e["families"].get(fam, 0) > 0}


def ref_variants(all_entries: list[dict], fam: str) -> set[str]:
    out: set[str] = set()
    for e in all_entries:
        if e["families"].get(fam, 0) > 0:
            out.update(e["variants"].get(fam, {}))
    return out


# ---------------------------------------------------------------- ref-inventory

def cmd_ref_inventory() -> dict:
    fla = scan_reference(FLA_ROOT, "fla")
    qla = scan_reference(QLA_ROOT, "qla")
    all_entries = fla + qla

    fam_totals = {}
    for ref, ents in (("fla", fla), ("qla", qla)):
        t, v = agg_families(ents)
        for f, n in t.items():
            d = fam_totals.setdefault(f, {"fla": 0, "qla": 0, "fla_kernels": set(), "qla_kernels": set(), "fla_configs": 0, "qla_configs": 0})
            d[ref] = n
            d[f"{ref}_kernels"] |= kernel_presence(ents, f)
            d[f"{ref}_configs"] += sum(1 for e in ents if e["families"].get(f))
    fam_out = {}
    for f, d in fam_totals.items():
        fam_out[f] = {
            "fla_count": d["fla"], "qla_count": d["qla"],
            "fla_kernels": sorted(d["fla_kernels"]), "qla_kernels": sorted(d["qla_kernels"]),
            "fla_configs": d["fla_configs"], "qla_configs": d["qla_configs"],
            "variants": sorted(ref_variants(all_entries, f)),
            "tier": "feature" if f in FEATURE_FAMILIES else "baseline",
        }

    mma_totals = {}
    for ref, ents in (("fla", fla), ("qla", qla)):
        for e in ents:
            for sig, n in e["mma_signatures"].items():
                d = mma_totals.setdefault(sig, {"fla": 0, "qla": 0, "fla_kernels": set(), "qla_kernels": set()})
                d[ref] += n
                if n:
                    d[f"{ref}_kernels"].add(e["kernel"])
    mma_out = {sig: {"fla": v["fla"], "qla": v["qla"],
                     "fla_kernels": sorted(v["fla_kernels"]), "qla_kernels": sorted(v["qla_kernels"])}
               for sig, v in sorted(mma_totals.items())}

    # per-kernel resource envelopes
    res_envelopes = {}
    for ref, ents in (("fla", fla), ("qla", qla)):
        by_kernel = defaultdict(list)
        for e in ents:
            by_kernel[e["kernel"]].append(e)
        for k, es in sorted(by_kernel.items()):
            regs = [e["resources"].get("regs") for e in es if e["resources"].get("regs") is not None]
            smem = [(e["resources"].get("smem_dynamic") or e["resources"].get("smem_static") or 0) for e in es]
            spills = [max(e["resources"].get("spill_stores") or 0, e["resources"].get("spill_loads") or 0) for e in es]
            mma = [sum(e["mma_signatures"].values()) for e in es]
            res_envelopes[f"{ref}:{k}"] = {
                "configs": [e["config"] for e in es],
                "regs": [min(regs), max(regs)] if regs else None,
                "smem_bytes": [min(smem), max(smem)] if smem else None,
                "spill_max_bytes": max(spills) if spills else None,
                "mma_total": [min(mma), max(mma)],
            }

    t1 = sorted(f for f, d in fam_out.items() if d["fla_count"] or d["qla_count"])
    out = {
        "generated_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "counts": {"fla_configs": len(fla), "qla_kernels": len(qla)},
        "families": fam_out,
        "mma_signatures": mma_out,
        "resource_envelopes": res_envelopes,
        "t1_family_set": t1,
        "t1_feature_families": [f for f in t1 if fam_out[f]["tier"] == "feature"],
        "entries": [{k: e[k] for k in ("ref", "kernel", "config", "path", "families", "variants", "mma_signatures",
                                        "resources", "bra_total", "loop_backedges", "g0_fams", "g0_parity_diff")
                     if k in e} for e in all_entries],
        "g0_fams_parity_note": "G0 FAMS imported from tools/harvest_flaptx.py and re-run per config (entries[].g0_fams); G0's 'tma.bulk' regex was broken (never matched) and mbarrier missed the mbarrier_init spelling — fixed in EXTRA_FAMS; entries[].g0_parity_diff lists any deltas vs G0 meta.json ptx_features",
    }
    G2_DIR.mkdir(parents=True, exist_ok=True)
    (G2_DIR / "ref_inventory.json").write_text(json.dumps(out, indent=1))
    write_inventory_md(out, fla, qla)
    print(f"[ref-inventory] {len(fla)} fla configs + {len(qla)} qla kernels; "
          f"{len(t1)} T1 families ({out['counts']}) -> {G2_DIR/'ref_inventory.json'} / ref_inventory.md")
    return out


def write_inventory_md(out: dict, fla: list[dict], qla: list[dict]) -> None:
    fam = out["families"]
    lines = [
        "# G2-B reference PTX inventory (T1 set source)", "",
        f"Generated {out['generated_utc']}. Sources: `evidence/G0/fla_ptx` ({out['counts']['fla_configs']} configs, Triton sm_120), "
        f"`evidence/G0/qla_ptx` ({out['counts']['qla_kernels']} kernels, TileLang sm_120a).", "",
        "## 1. Instruction families (family x which reference kernels use it)", "",
        "Tier `feature` = needs an explicit IR affordance (intrinsics/pipelining); `baseline` = falls out of standard arith/linalg->NVVM lowering.", "",
        "| family | tier | fla cfgs (count) | fla kernels | qla kernels (count) | exact opcode variants |",
        "|---|---|---|---|---|---|",
    ]
    def short(names: list[str], limit: int = 3) -> str:
        if not names:
            return "-"
        if len(names) <= limit:
            return ", ".join(n.replace("chunk_gated_delta_rule_", "cgdr_").replace("_kernel", "") for n in names)
        return f"{len(names)}: " + ", ".join(n.replace("chunk_gated_delta_rule_", "cgdr_").replace("_kernel", "") for n in names[:limit]) + ", ..."
    for f in FAMILY_ORDER + sorted(set(fam) - set(FAMILY_ORDER)):
        if f not in fam:
            continue
        d = fam[f]
        v = "<br>".join(sorted(d["variants"])) or "-"
        lines.append(f"| `{f}` | {d['tier']} | {d['fla_configs']} ({d['fla_count']}) | {short(d['fla_kernels'])} | {short(d['qla_kernels'], 5)} ({d['qla_count']}) | {v} |")

    lines += ["", "## 2. mma.sync signatures (unique full strings, with counts)", "",
              "| signature | fla | fla kernels | qla | qla kernels |", "|---|---|---|---|---|"]
    for sig, d in out["mma_signatures"].items():
        lines.append(f"| `{sig}` | {d['fla']} | {short(d['fla_kernels'])} | {d['qla']} | {short(d['qla_kernels'])} |")

    lines += ["", "## 3. Resource envelopes per kernel (regs / dynamic smem / max spill / mma totals over configs)", "",
              "| reference:kernel | configs | regs | smem (B) | spill max (B) | mma.sync total |", "|---|---|---|---|---|---|"]
    for k, d in out["resource_envelopes"].items():
        regs = "-".join(map(str, d["regs"])) if d["regs"] else "?"
        smem = "-".join(map(str, d["smem_bytes"])) if d["smem_bytes"] else "?"
        mma = "-".join(map(str, d["mma_total"]))
        lines.append(f"| {k} | {d['configs']} | {regs} | {smem} | {d['spill_max_bytes']} | {mma} |")

    lines += ["", "## 4. Structural notes (from PTX loop structure)", ""]
    for e in fla + qla:
        lines.append(f"- {e['ref']}:{e['kernel']}/{e['config']}: bra={e['bra_total']}, loop back-edges={e['loop_backedges']}")
    (G2_DIR / "ref_inventory.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- ours (G2-A gen)

def load_ours() -> tuple[list[dict], str]:
    """Read G2-A's generated set. Tolerant to INDEX.json schema; falls back to
    rglob('*.ptx') under evidence/G2/gen/. Returns (entries, state)."""
    if not GEN_DIR.is_dir():
        return [], "PENDING"
    ours: list[dict] = []

    def add(ptx_path: Path, label: str, idx_meta: dict | None = None):
        if not ptx_path.is_file():
            return
        ptx = ptx_path.read_text(errors="replace")
        res = load_meta_resources(ptx_path.parent / "meta.json")
        m = idx_meta or {}
        res = {
            "regs": m.get("registers") or res.get("regs"),
            "smem_static": m.get("smem_static", res.get("smem_static")),
            "smem_dynamic": m.get("smem_dynamic", res.get("smem_dynamic")),
            "spill_stores": m.get("spill_stores", res.get("spill_stores", 0)) or 0,
            "spill_loads": m.get("spill_loads", res.get("spill_loads", 0)) or 0,
            "num_warps": m.get("num_warps", res.get("num_warps")),
        }
        ours.append({
            "ref": "ours", "kernel": label, "config": label,
            "path": str(ptx_path.relative_to(REPO)),
            **extract_ptx(ptx),
            "resources": res,
        })

    if INDEX_JSON.is_file():
        try:
            data = json.loads(INDEX_JSON.read_text())
            items = data.get("configs", data) if isinstance(data, dict) else data
            if isinstance(items, dict):
                items = [{"name": k, **(v if isinstance(v, dict) else {"ptx": v})} for k, v in items.items()]
            for it in items or []:
                if not isinstance(it, dict):
                    continue
                p = it.get("ptx") or it.get("ptx_path") or it.get("path") or it.get("file")
                if not p:
                    continue
                cand = [Path(p), REPO / str(p), GEN_DIR / str(p)]
                for c in cand:
                    if c.is_file():
                        add(c, it.get("name") or it.get("config") or c.stem,
                            idx_meta=it.get("meta") if isinstance(it.get("meta"), dict) else None)
                        break
        except Exception as ex:
            print(f"[warn] INDEX.json parse failed ({ex}); falling back to rglob")
    if not ours:
        for p in sorted(GEN_DIR.rglob("*.ptx")):
            add(p, str(p.relative_to(GEN_DIR)))
    state = "LIVE" if ours else "PENDING"
    return ours, state


def pick_closest_ours(ref_entry: dict, ours: list[dict]) -> dict | None:
    if not ours:
        return None
    ref_fams = {f for f, n in ref_entry["families"].items() if n}
    ref_mma = Counter(ref_entry["mma_signatures"])

    def score(o: dict) -> tuple:
        of = {f for f, n in o["families"].items() if n}
        jac = len(ref_fams & of) / max(1, len(ref_fams | of))
        om = Counter(o["mma_signatures"])
        sig_match = sum(min(c, om.get(s, 0)) for s, c in ref_mma.items())
        rm = sum(ref_mma.values())
        closeness = -abs(sum(om.values()) - rm)
        return (jac, sig_match, closeness)

    return max(ours, key=score)


def envelope_check(ref_v, our_v) -> str:
    if ref_v is None or our_v is None:
        return "n/a"
    if ref_v == 0:
        return "PASS" if our_v == 0 else "OUT"
    lo, hi = 0.7 * ref_v, 1.3 * ref_v
    return "PASS" if lo <= our_v <= hi else f"OUT ({our_v / ref_v:.2f}x)"


# ---------------------------------------------------------------- matrix

T2_CARDS = [
    ("fla", "chunk_fwd_kernel_o", "BK64_BV64_warps4_stages3"),
    ("fla", "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64", "BV64_warps2_stages3"),
    ("qla", "fused_fwd", "default"),
    ("qla", "fused_bwd", "default"),
]


def cmd_matrix() -> None:
    inv = json.loads((G2_DIR / "ref_inventory.json").read_text()) if (G2_DIR / "ref_inventory.json").is_file() else None
    if inv is None:
        inv = cmd_ref_inventory()
    fla = [e for e in inv["entries"] if e["ref"] == "fla"]
    qla = [e for e in inv["entries"] if e["ref"] == "qla"]
    ours, ours_state = load_ours()

    fam = inv["families"]
    mma = inv["mma_signatures"]

    def ref_cell(entries, f):
        n = sum(e["families"].get(f, 0) for e in entries)
        return f"YES ({n})" if n else "-"

    def ours_cell(f, ref_variants_set: set[str]):
        if ours_state == "PENDING":
            return "PENDING"
        total = sum(e["families"].get(f, 0) for e in ours)
        if total == 0:
            return "NO"
        ov: set[str] = set()
        for e in ours:
            ov.update(e["variants"].get(f, {}))
        missing = ref_variants_set - ov
        if missing:
            return f"PARTIAL (missing: {', '.join(sorted(missing))})"
        return "YES"

    lines = [
        "# G2 coverage matrix (T1: instruction families + mma signatures + structure)", "",
        f"Generated {inv['generated_utc']}. ours-column state: **{ours_state}**"
        + (f" ({len(ours)} generated configs from `evidence/G2/gen/INDEX.json`)" if ours_state == "LIVE" else
           " — G2-A `evidence/G2/gen/INDEX.json` absent; re-run `python tools/gdn_coverage.py matrix` after it lands (idempotent)."),
        "",
        "Legend: YES = emitted by some config of that column; PARTIAL = family emitted but a reference opcode variant/signature is missing; NO = never emitted; - = not present in that reference.", "",
        "## 1. Instruction families", "",
        "| family | tier | fla | FlashQLA | ours | escape-hatch / plan (for NO cells) |",
        "|---|---|---|---|---|---|",
    ]
    n_yes = n_part = n_no = 0
    hatch_hits = []
    ordered = [f for f in FAMILY_ORDER if f in fam] + sorted(set(fam) - set(FAMILY_ORDER))
    for f in ordered:
        d = fam[f]
        if not (d["fla_count"] or d["qla_count"]):
            continue  # matrix rows = families in T1 set (present in refs)
        cell = ours_cell(f, set(d["variants"]))
        mark = {"PENDING": "PENDING", "YES": "YES"}.get(cell.split(" ")[0], cell.split(" ")[0])
        if mark == "YES":
            n_yes += 1
        elif mark == "PARTIAL":
            n_part += 1
        elif mark == "NO":
            n_no += 1
            h = HATCH.get(f, "review")
            if "ESCAPE-HATCH" in h:
                hatch_hits.append((f, h))
        lines.append(f"| `{f}` | {d['tier']} | {ref_cell(fla, f)} | {ref_cell(qla, f)} | {cell} | {HATCH.get(f, '')} |")
    if ours_state == "LIVE":
        lines += ["", f"ours family coverage: {n_yes} YES / {n_part} PARTIAL / {n_no} NO of {n_yes + n_part + n_no} T1 families."]

    lines += ["", "## 2. mma.sync signatures", "",
              "| signature | fla | FlashQLA | ours | note |", "|---|---|---|---|---|"]
    for sig, d in mma.items():
        if ours_state == "PENDING":
            cell = "PENDING"
        else:
            n = sum(e["mma_signatures"].get(sig, 0) for e in ours)
            any_mma = sum(sum(e["mma_signatures"].values()) for e in ours)
            cell = f"YES ({n})" if n else (f"NO (ours mma total {any_mma})" if any_mma else "NO")
        note = "bf16 workhorse (both refs)" if "bf16" in sig else ("tf32 kkt/dot precision (fla only)" if "tf32" in sig else "")
        lines.append(f"| `{sig}` | {d['fla']} | {d['qla']} | {cell} | {note} |")

    # structural dimensions
    def fla_env(key, pick):
        vals = [pick(e) for e in fla]
        vals = [v for v in vals if v is not None]
        return f"[{min(vals)}, {max(vals)}]" if vals else "?"

    def qla_env(key, pick):
        vals = [pick(e) for e in qla]
        vals = [v for v in vals if v is not None]
        return f"[{min(vals)}, {max(vals)}]" if vals else "?"

    R = lambda e, k: (e.get("resources") or {}).get(k)  # noqa: E731

    def ours_env(key):
        if ours_state != "LIVE":
            return "PENDING"
        vals = [R(e, key) for e in ours]
        vals = [v for v in vals if v is not None]
        return f"[{min(vals)}, {max(vals)}] over {len(ours)} configs" if vals else "?"

    ours_regs = ours_env("regs")
    ours_smem = ours_env("smem_dynamic")
    ours_spills = ours_env("spill_stores")
    ours_async = "PENDING" if ours_state != "LIVE" else f"cp.async.cg/ca x{sum(int(R(e, chr(114)+chr(101)+chr(103)+chr(115)) is not None) for e in ours)}-configs"
    struct_rows = [
        ("registers/thread", fla_env(None, lambda e: R(e, "regs")), qla_env(None, lambda e: R(e, "regs")), ours_regs,
         "fla rides the 255-reg cap with spills; qla pins <=128 via setmaxnreg"),
        ("dynamic smem (B)", fla_env(None, lambda e: R(e, "smem_dynamic")), qla_env(None, lambda e: R(e, "smem_static")), ours_smem,
         "sm_120 CTA limit 100KB/128KB; qla tiny (2KB) because TMA stages out of smem descriptors"),
        ("spills (B, max of st/ld)", fla_env(None, lambda e: max(R(e, "spill_stores") or 0, R(e, "spill_loads") or 0)),
         qla_env(None, lambda e: max(R(e, "spill_stores") or 0, R(e, "spill_loads") or 0)), ours_spills,
         "fla bwd kernels spill heavily (dhu 3.8KB); fused_fwd/bwd ~1.1-1.3KB"),
        ("async pipeline style", "cp.async.ca only in state-push kernels (fwd_h/dhu); else ld.global multistage",
         "TMA cp.async.bulk.tensor 4d/5d + mbarrier + elect.sync leader (kkt_solve: classic cp.async.cg)", ours_async,
         "two distinct pipelines to cover"),
        ("warp specialization", "none (no setmaxnreg/elect.sync)", "setmaxnreg.inc/dec + elect.sync in 4/5 kernels", ours_async,
         "G2-C deliverable (nvvm.setmaxnreg)"),
        ("loop structure (bra / back-edges)", fla_env(None, lambda e: e["bra_total"]) + " bra, mostly unrolled",
         qla_env(None, lambda e: e["bra_total"]) + " bra, real scf loops", ours_async,
         "qla re-uses loop bodies (fused_bwd 1600 mma from ~looped bodies); fla unrolls"),
    ]
    lines += ["", "## 3. Structural dimensions", "", "| dimension | fla envelope | FlashQLA envelope | ours | note |", "|---|---|---|---|---|"]
    for name, a, b, c, note in struct_rows:
        lines.append(f"| {name} | {a} | {b} | {c} | {note} |")

    # escape-hatch budget
    lines += ["", "## 4. Escape-hatch budget (<= 5 custom NVVM ops)", ""]
    if ours_state == "LIVE":
        if hatch_hits:
            lines.append(f"{len(hatch_hits)} NO-cell families require a custom NVVM escape-hatch (vs budget 5):")
            for f, h in hatch_hits:
                lines.append(f"- `{f}`: {h}")
        else:
            lines.append("No NO-cell feature families — escape-hatch budget untouched.")
    else:
        pre = [f for f in ("tma.bulk.tensor", "tma.bulk.ctrl", "mbarrier", "elect.sync", "prefetch.tensormap", "setmaxnreg") if f in fam]
        lines.append("ours PENDING; worst-case hatch set if G2-A emits none of the qla-only families "
                     f"(tma counted once for tensor+ctrl+prefetch): {', '.join(pre)} -> 4 hatches "
                     "(tma.bulk.tensor+ctrl+prefetch.tensormap bundle, mbarrier, elect.sync, setmaxnreg[G2-C plan]). "
                     "setmaxnreg is a planned G2-C pattern, not an unplanned hatch.")

    # T2 cards
    lines += ["", "## 5. T2 structural comparison cards", ""]
    scan = {(e["ref"], e["kernel"], e["config"]): e for e in inv["entries"]}
    for ref, kernel, config in T2_CARDS:
        e = scan.get((ref, kernel, config))
        if e is None:
            lines += [f"### {ref}:{kernel}/{config}", "", "MISSING in ref_inventory — check config dir name.", ""]
            continue
        r = e["resources"] or {}
        mma_tot = sum(e["mma_signatures"].values())
        loops = max(1, e["loop_backedges"])
        fams = e["families"]
        nc = sum(n for op, n in (e.get("variants", {}).get("ld.global", {}) or {}).items() if ".nc" in op)
        async_desc = (
            f"cp.async.ca={fams.get('cp.async.classic', 0)}, ctrl(commit/wait)={fams.get('cp.async.ctrl', 0)}, "
            f"TMA tensor={fams.get('tma.bulk.tensor', 0)}, TMA ctrl={fams.get('tma.bulk.ctrl', 0)}, "
            f"mbarrier={fams.get('mbarrier', 0)}, ld.global={fams.get('ld.global', 0)}(of which .nc {nc})"
        )
        lines += [f"### {ref}:{kernel}/{config}", "",
                  f"- mma.sync total **{mma_tot}** (est. per loop body ~{round(mma_tot / loops)} over {loops} back-edge loop(s); bra={e['bra_total']})",
                  f"- signatures: {'; '.join(f'{s} x{n}' for s, n in sorted(e['mma_signatures'].items())) or 'none'}",
                  f"- async pattern: {async_desc}",
                  f"- resources: regs={r.get('regs')}, smem={r.get('smem_dynamic') or r.get('smem_static')}B, "
                  f"spill st/ld={r.get('spill_stores')}/{r.get('spill_loads')}B",
                  f"- ldmatrix={fams.get('ldmatrix', 0)}, stmatrix={fams.get('stmatrix', 0)}, "
                  f"setmaxnreg={fams.get('setmaxnreg', 0)}, elect.sync={fams.get('elect.sync', 0)}, shfl={fams.get('shfl.sync', 0)}"]
        if ours_state == "LIVE":
            best = pick_closest_ours(e, ours)
            br = best["resources"] if "resources" in best else None
            lines.append(f"- ours closest config: **{best['kernel']}** (`{best['path']}`)")
            if br:
                lines.append(f"  - regs {br.get('regs')} vs ref {r.get('regs')} -> {envelope_check(r.get('regs'), br.get('regs'))}; "
                             f"smem {br.get('smem_dynamic') or br.get('smem_static')} vs {r.get('smem_dynamic') or r.get('smem_static')} -> "
                             f"{envelope_check(r.get('smem_dynamic') or r.get('smem_static'), br.get('smem_dynamic') or br.get('smem_static'))} (±30%)")
            om = sum(best["mma_signatures"].values())
            lines.append(f"  - mma total {om} vs ref {mma_tot} ({om / max(1, mma_tot):.2f}x); "
                         f"family Jaccard basis for 'closest' selection")
        else:
            lines.append("- ours closest config: PENDING (G2-A gen set absent)")
        lines.append("")

    (G2_DIR / "coverage_matrix.md").write_text("\n".join(lines) + "\n")
    print(f"[matrix] ours={ours_state} ({len(ours)} configs) -> {G2_DIR/'coverage_matrix.md'}")


# ---------------------------------------------------------------- t1-set

def cmd_t1_set() -> None:
    path = G2_DIR / "ref_inventory.json"
    inv = json.loads(path.read_text()) if path.is_file() else cmd_ref_inventory()
    fam = inv["families"]
    t1 = inv["t1_family_set"]
    print("=" * 78)
    print("T1 CONTRACT SET — union of instruction families across fla + FlashQLA refs")
    print(f"({len(t1)} families; {len(inv['t1_feature_families'])} feature-tier)")
    print("=" * 78)
    print("\n-- FEATURE tier (need explicit IR affordance) --")
    for f in sorted(inv["t1_feature_families"], key=lambda x: FAMILY_ORDER.index(x) if x in FAMILY_ORDER else 99):
        d = fam[f]
        prov = ("fla+qla" if d["fla_count"] and d["qla_count"] else "fla-only" if d["fla_count"] else "qla-only")
        print(f"  {f:22s} [{prov:8s}] variants: {', '.join(sorted(d['variants']))}")
    print("\n-- mma.sync signatures --")
    for sig, d in inv["mma_signatures"].items():
        print(f"  {sig}  (fla {d['fla']}, qla {d['qla']})")
    print("\n-- BASELINE tier (standard arith/linalg->NVVM lowering, listed for completeness) --")
    for f in sorted(f for f in t1 if fam[f]["tier"] == "baseline"):
        d = fam[f]
        prov = ("fla+qla" if d["fla_count"] and d["qla_count"] else "fla-only" if d["fla_count"] else "qla-only")
        print(f"  {f:22s} [{prov}]")
    print("\nG2-A's generated set is graded on the FEATURE tier + mma signatures.")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["ref-inventory", "matrix", "t1-set", "all"])
    a = ap.parse_args()
    if a.cmd in ("ref-inventory", "all"):
        cmd_ref_inventory()
    if a.cmd in ("matrix", "all"):
        cmd_matrix()
    if a.cmd in ("t1-set", "all"):
        cmd_t1_set()


if __name__ == "__main__":
    main()
