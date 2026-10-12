#!/usr/bin/env python3
"""fla Triton kernel 参考收割器（G0-5，纯 CPU，无 GPU）。

用法:
  python tools/harvest_flaptx.py probe                 # 打印各 kernel 签名与 wrapper 源码（校对用）
  python tools/harvest_flaptx.py harvest               # AOT 编译全部 (kernel, config) → evidence/G0/fla_ptx/
  python tools/harvest_flaptx.py harvest --only fwd_h  # 只收割匹配名

产出（每个 kernel/config 目录）:
  kernel.ptx            PTX 源
  kernel.cubin          cubin
  ptxas_v.txt           ptxas -v 资源报告（regs/smem/spill）
  sass.txt              nvdisasm SASS
  meta.json             triton metadata（shared 等）+ 指令族直方图（PTX 级）
最后汇总 evidence/G0/fla_ptx/SUMMARY.md
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HOME = Path.home()
EVID = HOME / "codebase/gdn-mlir/evidence/G0/fla_ptx"
TARGET_ARCH = "sm_120a"

# ---------------------------------------------------------------- 环境探测

def find_tool(name: str) -> str | None:
    """定位 ptxas / nvdisasm / cuobjdump：优先 triton 自带，其次 CUDA toolkit。"""
    cand = []
    try:
        import triton
        tb = Path(triton.__file__).parent / "backends/nvidia/bin"
        cand += [tb / name, tb / (name + "-blackwell"), tb / (name + "_blackwell")]
    except Exception:
        pass
    cand += [Path("/usr/local/cuda/bin") / name, Path("/usr/local/cuda/bin") / name.replace("ptxas", "ptxas")]
    for c in cand:
        if c and c.is_file():
            return str(c)
    return shutil.which(name)


def import_kernels():
    """导入 GDN 路径全部 Triton kernel（fused + fallback + aux）。"""
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
    from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd_kernel, recompute_w_u_fwd_kernel
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )
    from fla.ops.common.chunk_o import (
        chunk_bwd_kernel_dqkwg,
        chunk_bwd_kernel_dv_local,
        chunk_fwd_kernel_o,
    )
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_kernel
    from fla.ops.utils.solve_tril import (
        merge_16x16_to_32x32_inverse_kernel,
        merge_16x16_to_64x64_inverse_kernel,
        solve_tril_16x16_kernel,
    )
    ks = {}
    for name, obj in [
        ("chunk_gated_delta_rule_fwd_kkt_solve_kernel", chunk_gated_delta_rule_fwd_kkt_solve_kernel),
        ("recompute_w_u_fwd_kernel", recompute_w_u_fwd_kernel),
        ("prepare_wy_repr_bwd_kernel", prepare_wy_repr_bwd_kernel),
        ("chunk_gated_delta_rule_fwd_kernel_h_blockdim64", chunk_gated_delta_rule_fwd_kernel_h_blockdim64),
        ("chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64", chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64),
        ("chunk_fwd_kernel_o", chunk_fwd_kernel_o),
        ("chunk_bwd_kernel_dqkwg", chunk_bwd_kernel_dqkwg),
        ("chunk_bwd_kernel_dv_local", chunk_bwd_kernel_dv_local),
        ("chunk_scaled_dot_kkt_fwd_kernel", chunk_scaled_dot_kkt_fwd_kernel),
        ("solve_tril_16x16_kernel", solve_tril_16x16_kernel),
        ("merge_16x16_to_32x32_inverse_kernel", merge_16x16_to_32x32_inverse_kernel),
        ("merge_16x16_to_64x64_inverse_kernel", merge_16x16_to_64x64_inverse_kernel),
    ]:
        ks[name] = obj
    return ks


def unwrap(obj):
    """CachedAutotuner/装饰链 → 原始 triton JITFunction（带 .params/.arg_names）。"""
    seen = 0
    while not hasattr(obj, "params") and seen < 6:
        nxt = getattr(obj, "fn", None) or getattr(obj, "kernel", None) or getattr(obj, "inner", None)
        if nxt is None:
            break
        obj, seen = nxt, seen + 1
    return obj


# ---------------------------------------------------------------- 配置空间
# 形状：主套件代表点 B=4,T=8192,H=HV=16,K=V=128（NT = T/64 = 128）
# 每 kernel 的小配置集：来自 fla autotune 空间的代表点（Blackwell 相关），
# G2 覆盖矩阵阶段再系统化扩展。

CONFIGS: dict[str, list[dict]] = {
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel": [
        dict(warps=2, stages=3), dict(warps=4, stages=3), dict(warps=4, stages=4),
    ],
    "recompute_w_u_fwd_kernel": [
        dict(warps=2, stages=3), dict(warps=4, stages=2),
    ],
    "prepare_wy_repr_bwd_kernel": [
        dict(warps=2, stages=4), dict(warps=4, stages=3),
    ],
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": [
        dict(BV=64, warps=2, stages=3), dict(BV=32, warps=2, stages=4),
    ],
    "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64": [
        dict(BV=64, warps=2, stages=3), dict(BV=32, warps=2, stages=4),
    ],
    "chunk_fwd_kernel_o": [
        dict(BK=64, BV=64, warps=4, stages=3),
        dict(BK=32, BV=32, warps=2, stages=3),
    ],
    "chunk_bwd_kernel_dqkwg": [
        dict(warps=4, stages=3), dict(warps=4, stages=2),
    ],
    "chunk_bwd_kernel_dv_local": [
        dict(warps=4, stages=3),
    ],
    "chunk_scaled_dot_kkt_fwd_kernel": [
        dict(BK=64, warps=4, stages=3), dict(BK=128, warps=8, stages=3),
    ],
    "solve_tril_16x16_kernel": [
        dict(warps=1, stages=4), dict(warps=4, stages=2),
    ],
    "merge_16x16_to_32x32_inverse_kernel": [dict(warps=4, stages=2)],
    "merge_16x16_to_64x64_inverse_kernel": [dict(warps=4, stages=2)],
}

# ---------------------------------------------------------------- 类型规则
# 非constexpr参数 → triton 类型字符串。按参数名模式推断（fla 命名约定），
# probe 模式打印全部参数名，未知名字收割时会报出来人工补表。

# 显式类型表（2026-10-09 probe + 包装器分配处核对：h/dh=k.new_empty→bf16；ht/dh0/dht fp32；do=o.dtype）
BF16_PTRS = {"k", "v", "w", "u", "o", "q", "beta", "dq", "dk", "dv", "dw", "du",
             "db", "dv2", "v_new", "h", "dh", "do"}
FP32_PTRS = {"A", "Ai", "g", "g_gamma", "gk", "dht", "dh0",
             "h0", "ht", "dg"}
I32_PTRS = {"cu_seqlens", "chunk_indices", "chunk_offsets"}

# 个别 kernel 的类型覆盖（运行时真实 dtype：A 以 k.dtype(bf16) 保存；wy 的 tl.dot(b_A,b_dw) 要求同 dtype；
# solve_tril 输入为 kkt 的 fp32 A、输出 output_dtype=k.dtype）
PER_KERNEL_TYPES: dict[str, dict[str, str]] = {
    "prepare_wy_repr_bwd_kernel": {"A": "*bf16"},
    "recompute_w_u_fwd_kernel": {"A": "*bf16"},
    "chunk_bwd_kernel_dv_local": {"A": "*bf16"},
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel": {"A": "*bf16"},
    "solve_tril_16x16_kernel": {"Ai": "*bf16"},
    "merge_16x16_to_32x32_inverse_kernel": {"A": "*bf16", "Ai": "*bf16"},
    "merge_16x16_to_64x64_inverse_kernel": {"A": "*bf16", "Ai": "*bf16"},
}

def infer_type(arg: str) -> str | None:
    if arg == "scale":
        return "fp32"
    if arg in ("T", "N", "NT", "NV", "NK"):
        return "i32"
    if arg in BF16_PTRS:
        return "*bf16"
    if arg in FP32_PTRS:
        return "*fp32"
    if arg in I32_PTRS:
        return "*i32"
    return None


# 形状 constexpr（主套件代表点）；fused kkt_solve 的 BC=16 等由 kernel 内部固定/需补
SHAPE_CONSTEXPR: dict[str, object] = dict(
    H=16, HV=16, K=128, V=128, B=4, BT=64, BK=64, BV=64, BC=16,
    USE_G=True, USE_GK=False, USE_G_GAMMA=False, USE_DW=True, USE_A=True,
    USE_INITIAL_STATE=True, STORE_FINAL_STATE=True, USE_FINAL_STATE_GRADIENT=True,
    SAVE_NEW_VALUE=True, STATE_V_FIRST=False, IS_VARLEN=False,
    USE_TMA=False, DOT_PRECISION="ieee",
)

def constexpr_for(kernel_name: str, cfg: dict) -> dict:
    c = dict(SHAPE_CONSTEXPR)
    c.update({k: v for k, v in cfg.items() if k not in ("warps", "stages")})
    return c

# ---------------------------------------------------------------- probe

def probe():
    ks = import_kernels()
    for name, k in ks.items():
        print("=" * 100)
        print("KERNEL:", name, "| wrapped:", type(k).__name__)
        raw = unwrap(k)
        print("  unwrapped:", type(raw).__name__)
        try:
            params = getattr(raw, "params", None)
            if params:
                for p in params:
                    print(f"  {p.name:24s} constexpr={bool(getattr(p, 'is_constexpr', False))}")
            else:
                print("  arg_names:", raw.arg_names)
        except Exception as e:
            print("  introspect fail:", e)
    print("=" * 100)
    print("wrapper 源码片段（dtype 依据），请人工核对 infer_type 表")

# ---------------------------------------------------------------- PTX 特征提取

FAMS = [
    (r"\bmma\.sync\.[\w.]+", "mma.sync"),
    (r"\bcp\.async\.[\w.]+", "cp.async"),
    (r"\bcm\.?[\w.]*\.cp\.async\.bulk[\w.]*", "tma.bulk"),
    (r"\bmbarrier[\w.]*", "mbarrier"),
    (r"\bexp2?\.approx[\w.]*", "exp.approx"),
    (r"\bld\.global[\w.]*", "ld.global"),
    (r"\bst\.global[\w.]*", "st.global"),
    (r"\bld\.shared[\w.]*", "ld.shared"),
    (r"\bst\.shared[\w.]*", "st.shared"),
    (r"\bbar\.sync", "bar.sync"),
    (r"\bbarrier\.", "barrier"),
    (r"\bshfl\.sync", "shfl.sync"),
    (r"\bredux[\w.]*", "redux"),
    (r"\bsetmaxnreg", "setmaxnreg"),
    (r"\bmovmatrix", "movmatrix"),
    (r"\bldmatrix", "ldmatrix"),
    (r"\bstmatrix", "stmatrix"),
    (r"\bwmma\.", "wmma"),
    (r"\bwgmma", "wgmma"),
    (r"\btcgen05", "tcgen05"),
    (r"\bfma\.rn\.f32", "fma.f32"),
]

def ptx_features(ptx: str) -> dict:
    c = Counter()
    for pat, fam in FAMS:
        c[fam] = len(re.findall(pat, ptx))
    # mma 形状细分
    for m in re.findall(r"mma\.sync\.(\w+)", ptx):
        c["mma.sync." + m.split(".")[0] if "." in m else "mma.sync.other"] += 0  # 占位：细节在 G2 展开
    return dict(c)

def parse_ptxas_v(txt: str) -> dict:
    out = {}
    m = re.search(r"Used (\d+) registers", txt); out["registers"] = int(m.group(1)) if m else None
    m = re.search(r"(\d+) bytes smem", txt); out["smem_static"] = int(m.group(1)) if m else None
    m = re.search(r"(\d+) bytes spill stores", txt); out["spill_stores"] = int(m.group(1)) if m else None
    m = re.search(r"(\d+) bytes spill loads", txt); out["spill_loads"] = int(m.group(1)) if m else None
    m = re.search(r"(\d+) bytes stack frame", txt); out["stack_frame"] = int(m.group(1)) if m else None
    return out

# ---------------------------------------------------------------- harvest

def compile_one(k, sig: dict, cexpr: dict, warps: int, stages: int):
    import triton
    from triton.compiler import ASTSource            # triton>=3.8
    from triton.backends.compiler import GPUTarget
    src = ASTSource(fn=k, signature=sig, constexprs=cexpr)
    tgt = GPUTarget("cuda", 120, 32)
    return triton.compile(src, target=tgt, options={"num_warps": warps, "num_stages": stages})

def harvest(only: str | None):
    import triton
    ks = import_kernels()
    ptxas = find_tool("ptxas-blackwell") or find_tool("ptxas")  # PTX ISA 9.3 仅 ptxas-blackwell 支持
    nvdisasm = find_tool("nvdisasm") or find_tool("cuobjdump")
    assert ptxas, "ptxas 不可得（triton backends 或 /usr/local/cuda）"
    EVID.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, k in ks.items():
        if only and only not in name:
            continue
        raw = unwrap(k)
        params = getattr(raw, "params", None)
        if not params:
            print(f"[skip] {name}: 无 params 自省")
            continue
        for cfg in CONFIGS.get(name, []):
            tag = "_".join(f"{k2}{v}" for k2, v in cfg.items())
            outdir = EVID / name / tag
            outdir.mkdir(parents=True, exist_ok=True)
            sig, cexpr = {}, {}
            unknown, missing_c = [], []
            for p in params:
                if getattr(p, "is_constexpr", False):
                    sig[p.name] = "constexpr"
                    v = SHAPE_CONSTEXPR.get(p.name)
                    if v is None:
                        missing_c.append(p.name)
                    cexpr[p.name] = v
                else:
                    t = infer_type(p.name)
                    if t is None:
                        unknown.append(p.name)
                    sig[p.name] = PER_KERNEL_TYPES.get(name, {}).get(p.name, t)
            # cfg 里的 tile 覆盖（BK/BV 等）
            for k2, v in cfg.items():
                if k2 in ("warps", "stages"):
                    continue
                cexpr[k2] = v
            if unknown or missing_c:
                print(f"[need-attrs] {name} 未知类型={unknown} 缺constexpr={missing_c}")
                continue
            warps, stages = cfg.get("warps", 4), cfg.get("stages", 3)
            try:
                ck = compile_one(raw, sig, cexpr, warps, stages)
            except Exception as e:
                print(f"[fail-compile] {name} {tag}: {type(e).__name__}: {str(e)[:200]}")
                (outdir / "ERROR.txt").write_text(repr(e))
                continue
            ptx = ck.asm["ptx"]
            (outdir / "kernel.ptx").write_text(ptx)
            if ck.asm.get("cubin"):
                (outdir / "kernel.cubin").write_bytes(ck.asm["cubin"])
            # ptxas -v 独立复跑（资源报告）
            r = subprocess.run([ptxas, "-v", f"--gpu-name={TARGET_ARCH}",
                                str(outdir / "kernel.ptx"), "-o", str(outdir / "kernel_v.cubin")],
                               capture_output=True, text=True)
            (outdir / "ptxas_v.txt").write_text(r.stderr)
            # SASS
            if nvdisasm and (outdir / "kernel.cubin").exists():
                s = subprocess.run([nvdisasm, "-c", str(outdir / "kernel.cubin")],
                                   capture_output=True, text=True)
                (outdir / "sass.txt").write_text(s.stdout or s.stderr)
            meta = {
                "kernel": name, "config": cfg,
                "shared_dynamic": getattr(ck.metadata, "shared", None),
                "num_warps": getattr(ck.metadata, "num_warps", warps),
                "ptxas": parse_ptxas_v(r.stderr),
                "ptx_features": ptx_features(ptx),
            }
            (outdir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
            rows.append(meta)
            print(f"[ok] {name} {tag}: regs={meta['ptxas']['registers']} smem={meta['ptxas']['smem_static']}")
    # 汇总
    if rows:
        import io
        buf = io.StringIO()
        buf.write("| kernel | config | regs | smem_dyn | spill_st | spill_ld | mma.sync | cp.async | exp.approx |\n")
        buf.write("|---|---|---|---|---|---|---|---|---|\n")
        for m in rows:
            p = m["ptxas"]; f = m["ptx_features"]
            buf.write(f"| {m['kernel']} | {m['config']} | {p['registers']} | {m.get('shared_dynamic')} | "
                      f"{p['spill_stores']} | {p['spill_loads']} | {f.get('mma.sync',0)} | "
                      f"{f.get('cp.async',0)} | {f.get('exp.approx',0)} |\n")
        (EVID / "SUMMARY.md").write_text(buf.getvalue())
        print(f"[done] {len(rows)} 条 → {EVID/'SUMMARY.md'}")


def summarize():
    """从已落盘的 meta.json 重建 SUMMARY（不重编译）。"""
    import io
    rows = []
    for m in sorted(EVID.rglob("meta.json")):
        rows.append(json.loads(m.read_text()))
    if rows:
        buf = io.StringIO()
        buf.write("| kernel | config | regs | smem_dyn | spill_st | spill_ld | mma.sync | cp.async | exp.approx |\n")
        buf.write("|---|---|---|---|---|---|---|---|---|\n")
        for m in rows:
            p = m["ptxas"]; f = m["ptx_features"]
            buf.write(f"| {m['kernel']} | {m['config']} | {p['registers']} | {m.get('shared_dynamic')} | "
                      f"{p['spill_stores']} | {p['spill_loads']} | {f.get('mma.sync',0)} | "
                      f"{f.get('cp.async',0)} | {f.get('exp.approx',0)} |\n")
        (EVID / "SUMMARY.md").write_text(buf.getvalue())
        print(f"[summarize] {len(rows)} 条 → {EVID/'SUMMARY.md'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "harvest", "summarize"])
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    (probe if a.mode == "probe" else (summarize if a.mode == "summarize" else lambda: harvest(a.only)))()
