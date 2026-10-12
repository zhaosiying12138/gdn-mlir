#!/usr/bin/env python3
"""诊断 TileLang target 解析（为何 sm_120a 被降成 sm_120）。"""
import os
print("env TILELANG_DEFAULT_TARGET =", os.environ.get("TILELANG_DEFAULT_TARGET"))
import tilelang
import tvm
from tvm.target import Target as TVMTarget
from tilelang.contrib.nvcc import get_target_arch_and_code

tt = TVMTarget.current(allow_none=True)
print("tvm.current():", tt)
print("resolved arch:", get_target_arch_and_code(tt))

# 看 jit 装饰器的默认 target 解析
import inspect
import tilelang.jit as tj
src = inspect.getsource(tj)
for ln in src.splitlines():
    if "TILELANG_DEFAULT_TARGET" in ln or "def _default_target" in ln or "def parse_target" in ln:
        print("|", ln.strip())

# tilelang.Target 自身
try:
    t2 = tilelang.Target("cuda", arch="sm_120a")
    print("Target(cuda, arch=sm_120a):", t2, "attrs:", dict(getattr(t2, "attrs", {})))
    print("arch pair:", get_target_arch_and_code(t2))
except Exception as e:
    print("Target构造失败:", repr(e))
