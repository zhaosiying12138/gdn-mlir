#!/usr/bin/env python3
"""Patch tools/gdn_full_rt.py: FullKernel supports multi-kernel specs
(the round-2 mono module: 3 entries, shared globals, launch by name)."""
PATH = "tools/gdn_full_rt.py"
src = open(PATH).read()
if "mono" in src:
    print("already patched")
    raise SystemExit(0)

old = """        self.mod = mod
        e, fn = cu.cuModuleGetFunction(mod, self.name.encode())
        _ck((e,), "cuModuleGetFunction")
        self.fn = fn
        self.globals = {}"""
new = """        self.mod = mod
        e, fn = cu.cuModuleGetFunction(mod, self.name.encode())
        _ck((e,), "cuModuleGetFunction")
        self.fn = fn
        # G5-D round 2 (mono): one cubin, several entries, shared globals
        self.fns = {}
        for k in self.spec.get("kernels", []):
            e, f2 = cu.cuModuleGetFunction(mod, k["name"].encode())
            _ck((e,), "cuModuleGetFunction " + k["name"])
            self.fns[k["name"]] = (f2, k["grid"], k["block"])
        self.globals = {}"""
assert src.count(old) == 1
src = src.replace(old, new)

old = """    def launch(self, stream):
        cu = self.cu
        _ck(cu.cuLaunchKernel(self.fn, self.grid[0], self.grid[1], 1,
                              self.block, 1, 1, 0, stream, None, 0),
            f"cuLaunchKernel {self.name}")"""
new = """    def launch(self, stream, name=None):
        cu = self.cu
        if name is None:
            _ck(cu.cuLaunchKernel(self.fn, self.grid[0], self.grid[1], 1,
                                  self.block, 1, 1, 0, stream, None, 0),
                f"cuLaunchKernel {self.name}")
            return
        fn, grid, block = self.fns[name]
        _ck(cu.cuLaunchKernel(fn, grid[0], grid[1], 1,
                              block[0], 1, 1, 0, stream, None, 0),
            f"cuLaunchKernel {name}")"""
assert src.count(old) == 1
src = src.replace(old, new)
open(PATH, "w").write(src)
print("[patch_rt] ok")
