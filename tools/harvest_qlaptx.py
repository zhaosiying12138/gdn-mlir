#!/usr/bin/env python3
"""FlashQLA (TileLang) sm_120 kernel 参考收割器（G0-6，纯 CPU）。

流程：工厂调用（镜像包装器实参）→ get_kernel_source() 出 CUDA 源 →
nvcc -ptx/-cubin -arch=sm_120a（-Xptxas -v 资源）→ nvdisasm SASS → 指令族直方图。
产出 evidence/G0/qla_ptx/<kernel>/<tag>/… + SUMMARY.md
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TILELANG_DEFAULT_TARGET", '{"kind":"cuda","arch":"sm_120a"}')

HOME = Path.home()
EVID = HOME / "codebase/gdn-mlir/evidence/G0/qla_ptx"
ARCH = "sm_120a"
sys.path.insert(0, str(Path(__file__).parent))  # 复用 fla 收割器的特征提取
from harvest_flaptx import FAMS, parse_ptxas_v, ptx_features  # noqa: E402

import torch  # noqa: E402


def _force_sm120a():
    """monkeypatch：所有 get_target_arch_and_code 调用点强制返回 120a。

    FlashQLA sm120 kernel 需要 setmaxnreg（arch-specific 'a' 后缀），
    而 tilelang 默认解析链会把 target 规范成不带 'a' 的 sm_120。
    只在本进程内 patch，不改 vendor 文件。
    """
    import importlib
    import tilelang.contrib.nvcc as _nvcc
    orig = _nvcc.get_target_arch_and_code

    def patched(target=None):
        arch, code = orig(target)
        return ("120a", code)

    _nvcc.get_target_arch_and_code = patched
    for name in ("tilelang.jit.adapter.libgen", "tilelang.cuda.backend",
                 "tilelang.jit.kernel"):
        try:
            m = importlib.import_module(name)
            if hasattr(m, "get_target_arch_and_code"):
                m.get_target_arch_and_code = patched
        except Exception:
            pass
    return orig


_force_sm120a()

bf16, fp32, i32 = torch.bfloat16, torch.float32, torch.int32

H, Hg, DK, DV, CHUNK = 16, 16, 128, 128, 64
SCALE = float(DK) ** -0.5

# 每条 = (内核名, 工厂导入路径, 工厂kwargs)；实参全部镜像自 FlashQLA 包装器
CALLS = [
    ("kkt_solve", "flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120.kkt_solve",
     dict(tilelang="tilelang_kkt_solve", args=dict(
         H=H, Hg=Hg, DK=DK, chunk_size=CHUNK, qkva_dtype=bf16, b_dtype=bf16,
         seqlen_dtype=i32, accum_dtype="float32", is_varlen=False))),
    ("prepare_h", "flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120.prepare_h",
     dict(tilelang="tilelang_prepare_h", args=dict(
         H=H, Hg=Hg, DK=DK, DV=DV, chunk_size=CHUNK, qkva_dtype=bf16, g_dtype=fp32,
         b_dtype=bf16, h0_dtype=fp32, ht_dtype=fp32, h_dtype=bf16, seqlen_dtype=i32,
         accum_dtype="float32", use_initial_state=True, store_final_state=True,
         store_h=True, is_varlen=False, is_cp=False, state_v_first=False,
         num_stages=2))),
    ("prepare_h_s3", "flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120.prepare_h",
     dict(tilelang="tilelang_prepare_h", args=dict(
         H=H, Hg=Hg, DK=DK, DV=DV, chunk_size=CHUNK, qkva_dtype=bf16, g_dtype=fp32,
         b_dtype=bf16, h0_dtype=fp32, ht_dtype=fp32, h_dtype=bf16, seqlen_dtype=i32,
         accum_dtype="float32", use_initial_state=True, store_final_state=True,
         store_h=True, is_varlen=False, is_cp=False, state_v_first=False,
         num_stages=3))),
    ("fused_fwd", "flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120.fused_fwd",
     dict(tilelang="tilelang_fused_chunk_gdr_fwd", args=dict(
         H=H, Hg=Hg, DK=DK, DV=DV, chunk_size=CHUNK, scale=SCALE, qkva_dtype=bf16,
         g_dtype=fp32, b_dtype=bf16, h0_dtype=fp32, ht_dtype=fp32, h_dtype=bf16,
         o_dtype=bf16, seqlen_dtype=i32, accum_dtype="float32",
         use_initial_state=True, store_final_state=True, store_h=True, store_o=True,
         is_varlen=False, is_cp=False, state_v_first=False, block_DV=128))),
    ("fused_bwd", "flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120.fused_bwd",
     dict(tilelang="tilelang_fused_chunk_gdr_bwd", args=dict(
         H=H, Hg=Hg, DK=DK, DV=DV, chunk_size=CHUNK, scale=SCALE, qkva_dtype=bf16,
         g_dtype=fp32, b_dtype=bf16, h_dtype=bf16, o_dtype=bf16, seqlen_dtype=i32,
         accum_dtype="float32", is_varlen=False, use_dht=False, state_v_first=False))),
]


def tilelang_includes() -> list[str]:
    """对齐 TileLang 内部 nvcc 的 -I（src + 3rdparty/cutlass/include 等）。"""
    import tilelang
    base = Path(tilelang.__file__).parent
    out: list[str] = []
    for p in [base / "src",
              base / "3rdparty",
              base / "3rdparty/cutlass/include",
              base / "include"]:
        if p.is_dir() and f"-I{p}" not in out:
            out.append(f"-I{p}")
    hits = list(base.glob("**/tl_templates"))
    for h in hits:
        if f"-I{h.parent}" not in out:
            out.append(f"-I{h.parent}")
    return out


def get_cu(kernel, outdir: Path) -> str:
    """TileLang JITKernel → CUDA 源（不 launch，不触碰 GPU）。"""
    for kwargs in ({"kernel_only": True}, {}):
        try:
            src = kernel.get_kernel_source(**kwargs)
            if src:
                return src
        except TypeError:
            continue
    raise RuntimeError("get_kernel_source 不可用")


def nvcc_build(cu_path: Path, outdir: Path, incs: list[str]):
    env = dict(os.environ)
    ptx, cubin = outdir / "kernel.ptx", outdir / "kernel.cubin"
    base = ["nvcc", "-O3", "--expt-relaxed-constexpr", "-arch=" + ARCH] + incs
    r1 = subprocess.run(base + ["-ptx", str(cu_path), "-o", str(ptx)],
                        capture_output=True, text=True, env=env, cwd=outdir)
    r2 = subprocess.run(base + ["-Xptxas", "-v", "-cubin", str(cu_path), "-o", str(cubin)],
                        capture_output=True, text=True, env=env, cwd=outdir)
    return ptx if r1.returncode == 0 else None, cubin if r2.returncode == 0 else None, r1.stderr + "\n" + r2.stderr


def main():
    import importlib
    EVID.mkdir(parents=True, exist_ok=True)
    incs = tilelang_includes()
    print("[qla] includes:", incs)
    rows = []
    for name, module_path, spec in CALLS:
        mod = importlib.import_module(module_path)
        factory = getattr(mod, spec["tilelang"])
        outdir = EVID / name
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            kern = factory(**spec["args"])
            cu = get_cu(kern, outdir)
            (outdir / "kernel.cu").write_text(cu)
        except Exception as e:
            print(f"[fail-factory] {name}: {type(e).__name__}: {str(e)[:300]}")
            (outdir / "ERROR.txt").write_text(repr(e))
            continue
        ptx, cubin, log = nvcc_build(outdir / "kernel.cu", outdir, incs)
        (outdir / "nvcc.log").write_text(log)
        if not ptx:
            print(f"[fail-nvcc] {name}: 见 {outdir/'nvcc.log'}")
            continue
        meta = {"kernel": name, "ptxas": parse_ptxas_v(log), "ptx_features": ptx_features(ptx.read_text())}
        nvdis = subprocess.run(["nvdisasm", "-c", str(cubin)], capture_output=True, text=True) if cubin else None
        if nvdis and nvdis.stdout:
            (outdir / "sass.txt").write_text(nvdis.stdout)
        (outdir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        rows.append(meta)
        p = meta["ptxas"]; f = meta["ptx_features"]
        print(f"[ok] {name}: regs={p['registers']} smem={p['smem_static']} "
              f"mma.sync={f.get('mma.sync',0)} cp.async={f.get('cp.async',0)} tma={f.get('tma.bulk',0)}")
    if rows:
        lines = ["| kernel | regs | smem | spill | mma.sync | cp.async | tma.bulk | setmaxnreg |",
                 "|---|---|---|---|---|---|---|---|"]
        for m in rows:
            p, f = m["ptxas"], m["ptx_features"]
            lines.append(f"| {m['kernel']} | {p['registers']} | {p['smem_static']} | "
                         f"{p['spill_stores']} | {f.get('mma.sync',0)} | {f.get('cp.async',0)} | "
                         f"{f.get('tma.bulk',0)} | {f.get('setmaxnreg',0)} |")
        (EVID / "SUMMARY.md").write_text("\n".join(lines) + "\n")
        print(f"[done] {len(rows)} 条 → {EVID/'SUMMARY.md'}")


if __name__ == "__main__":
    main()
