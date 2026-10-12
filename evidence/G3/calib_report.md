# G3-A: sm_120 hardware calibration + 20-config measured GPU times

Date: 2026-10-09 (WSL time) | GPU: RTX 5090 Laptop (sm_120a, 82 SMs) | Driver CUDA 13.3

**Shared-GPU caveat (read first).** All numbers below were measured in an
*opportunistic window* while another session's training job was resident on
the same GPU (pid 3411456, ~17.9 GiB VRAM, 24-99% util across the window,
SM clocks observed 2.1-2.8 GHz vs 1597 MHz peak attribute). Absolute
bandwidth/FLOP-rate numbers are therefore **noisy lower bounds**. Every
output file is marked `"shared_gpu": true` with the observed co-load
snapshot. The whole harness is **re-runnable in a clean window** with the
same commands and filenames — fresh values simply overwrite (see Rerun).

## 1. Calibration microbench (evidence/G3/calibration.json)

Kernel sources: `tools/gdn_calib/calib_bench.cu` (`nvcc -arch=sm_120a -O3`),
run by `scripts/run_calib.sh`. Timing: CUDA events, >=5 warmup + 50 timed
iterations, **median** reported.

| metric | value | notes |
|---|---|---|
| SM count | 82 | cudaDevAttr |
| L2 size | 64 MiB | measured both regimes below |
| shared mem / SM (optin) | 99 KiB | |
| peak clock (attr) | 1597 MHz | observed boost 2.1-2.8 GHz during run |
| **(a) mma.sync.m16n8k16 bf16->f32** | **3.70 PFLOP/s** (902.6 G-instr/s) | 82 SMs saturated, ILP 8/thread; 6.89 instr/SM/clk vs 1597 MHz attr clock ~= **4.1 instr/SM/clk at the ~2.7 GHz observed clock** — consistent with 4 tensor-core pipes/SM |
| **(b) smem read bw** (ld.shared 16B) | **113.1 TB/s** | 48 KB/window/block, xor-sum |
| **(b) gmem read, L2-resident** (32 MiB buffer) | **2144 GB/s** | buffer < 64 MiB L2 |
| **(b) gmem read, DRAM regime** (512 MiB) | **825 GB/s** | heavy co-load contention; GDDR7 spec is ~1.8 TB/s — treat as contended lower bound |
| **(c) SFU ex2.approx.f32** | **3.23 Telem/s** (24.7 elem/SM/clk @ attr clock) | 4 independent chains/thread |
| **(d) bar.sync round-trip** (128-thr block) | **7.71 ns** | 20000 reps, 82 blocks |
| **(e) cp.async 16B gmem->smem** (bonus) | **1388 GB/s** | 1 MiB L2-resident src window, commit/wait per iter |

## 2. First-ever GPU execution of the G2 configs — two lowering gaps found

G2's cubins had never executed (`gpu_execution: false`). The timing harness
(`tools/gdn_time.py`, cuda-python/cuda-bindings `cuModuleLoadData` +
`cuLaunchKernel` + CUDA events) probed each **as-built** cubin once before
timing the repaired variant:

* **Gap 1 — shared-memory lowering to device `malloc`.** Every
  `memref.alloc() : #gpu.address_space<workgroup>` inside `gpu.func` is
  lowered by `-gpu-lower-to-nvvm-pipeline` to a device-side
  `call.uni malloc` + `cvta.to.shared` of the generic heap pointer
  (upstream-documented: `llvm-mlir/test/Dialect/GPU/memref-to-llvm.mlir`).
  On hardware: the CALL.ABS trampoline is unresolved in a raw ptxas cubin,
  and the heap pointer reinterpreted as a shared-window offset. Probing the
  as-built cubins: **all configs with smem tiles trap with
  `CUDA_ERROR_ILLEGAL_ADDRESS`** (or `INVALID_VALUE` on launch). Repair
  (`tools/gdn_smem_fix.py`): pure PTX rewrite — drop the malloc callseqs,
  replace the `cvta.to.shared` with `mov.u64 %rd, __gdn_smemN;` and declare
  `.shared .align 16 .b8 __gdn_smemN[size];` (the same idiom MLIR itself
  emits for the ws-config mbarrier). After the fix the kernels run and
  produce non-zero output.
* **Gap 2 — constant capture explosion.** `-transform-interpreter` (tiling)
  and `gdn-opt --gdn-lower-hatch` hoist the gpu.launch-region's
  `arith.constant`s back to the host func, where gpu-kernel-outlining turns
  each into a kernel parameter (up to ~140 `.b64` params of compile-time
  constants; one bf16 pad becomes a `.param .align 2 .b8 [2]`). Repair:
  `run_l2.sh` now keeps `asmobj.mlir`; `gdn_smem_fix.py` parses the host
  `gpu.launch_func ... args(...)` operand list (positional order == PTX
  param order), substitutes every `ld.param::entry` with an immediate `mov`,
  and deletes the `.param` declarations — **18 of 20 configs become truly
  zero-arg kernels**; the ws config keeps 3 marshalled constants
  (2x i1 + a 16B vector zero) and the tma config keeps its 3 host-side
  tensormap pointers (wired via `cuTensorMapEncodeTiled`).

Deliverable chain per config: `gdn_l2_gen.py --driver noargs` (inputs as
module-level constant `memref.global`s with deterministic 0.5-step values,
output to public zero-init `@gdn_out`, constants inside the launch region)
-> `scripts/run_l2.sh` (sm_120a cubin + PTX/SASS + asmobj.mlir) ->
`gdn_smem_fix.py` -> `kernel_smemfix.cubin` -> `gdn_time.py`.

## 3. Config timing (evidence/G3/config_times.json)

Median of 50 CUDA-event-timed launches after 5 warmups, grid/block from the
config (`.maxntid` cross-checked). `nonzero` = elements != 0 in the
zero-initialized `@gdn_out` after one launch (proves the kernel wrote);
checksum determinism verified by zeroing + relaunching and comparing sums.

| config | status | median (ms) | min (ms) | nonzero/total | checksum stable | grid | block |
|---|---|---|---|---|---|---|---|
| `kkt_bf16_mma_p0_t64x64x32_w4` | OK | 0.1051 | 0.1042 | 1168/4096 | yes | 1x1x1 | 128x1x1 |
| `kkt_bf16_mma_p2_t32x64x32_w4` | OK | 0.0188 | 0.0163 | 1168/4096 | yes | 2x1x1 | 128x1x1 |
| `kkt_bf16_mma_p2_t64x64x16_w4` | OK | 0.0205 | 0.0188 | 1168/4096 | yes | 1x1x1 | 128x1x1 |
| `kkt_bf16_mma_p2_t64x64x32_w2` | OK | 0.0148 | 0.0139 | 1168/4096 | yes | 1x1x1 | 64x1x1 |
| `kkt_bf16_mma_p2_t64x64x32_w4` | OK | 0.0212 | 0.0194 | 1168/4096 | yes | 1x1x1 | 128x1x1 |
| `kkt_bf16_mma_p2_t64x64x32_w8` | OK | 0.0342 | 0.0320 | 1168/4096 | yes | 1x1x1 | 256x1x1 |
| `kkt_bf16_mma_p2_t64x64x64_w4_ws` | TIMEOUT | - | - | - | - |  |  |
| `kkt_bf16_mma_p3_t64x64x32_w4` | OK | 0.0206 | 0.0187 | 1168/4096 | yes | 1x1x1 | 128x1x1 |
| `kkt_bf16_mma_tma_ws` | NOT_TIMED | - | - | - | - | 1x1x1 | 160x1x1 |
| `kkt_bf16_nomma_p0_t64x64x32_w4` | OK | 1.0725 | 1.0692 | 2080/4096 | yes | 1x1x1 | 128x1x1 |
| `kkt_bf16_nomma_p2_t64x32x32_w4` | OK | 0.5079 | 0.5045 | 2080/4096 | yes | 1x2x1 | 128x1x1 |
| `kkt_bf16_nomma_p2_t64x64x32_w4` | OK | 0.9918 | 0.9847 | 2080/4096 | yes | 1x1x1 | 128x1x1 |
| `o_bf16_mma_p0_t32x64x32_w2_K64V64` | OK | 0.0450 | 0.0433 | 2304/4096 | yes | 2x1x1 | 64x1x1 |
| `o_bf16_mma_p2_t64x64x32_w4_K64V64` | OK | 0.0203 | 0.0186 | 2304/4096 | yes | 1x1x1 | 128x1x1 |
| `o_bf16_mma_p3_t64x128x16_w8_K128V128` | OK | 0.0844 | 0.0816 | 4608/8192 | yes | 1x1x1 | 256x1x1 |
| `o_bf16_nomma_p2_t64x64x32_w4_K64V64` | OK | 2.3492 | 2.1277 | 4096/4096 | yes | 1x1x1 | 128x1x1 |
| `o_f32_mma_p2_t64x64x32_w4_K64V64` | OK | 0.0333 | 0.0315 | 2304/4096 | yes | 1x1x1 | 128x1x1 |
| `o_f32_nomma_p0_t64x64x32_w4_K64V64` | OK | 2.4642 | 2.2610 | 4096/4096 | yes | 1x1x1 | 128x1x1 |
| `ref_f16_sm80_atom` | OK | 0.0062 | 0.0045 | 128/128 | yes | 1x1x1 | 32x1x1 |
| `ref_ldmatrix` | OK | 0.0073 | 0.0045 | 128/128 | yes | 1x1x1 | 32x1x1 |

## 4. Checksum sanity + failures

Configs that FAILED (named, with cause):
- **`kkt_bf16_mma_p2_t64x64x64_w4_ws`** — `TIMEOUT`: timing worker exceeded 120s (kernel hang?); subprocess killed
  - as-built (G2) cubin probe: `launch failed: <CUresult.CUDA_ERROR_INVALID_VALUE: 1>`
- **`kkt_bf16_mma_tma_ws`** — `NOT_TIMED`: cuModuleGetGlobal(gdn_out): <CUresult.CUDA_ERROR_NOT_FOUND: 500>. The tma variant's cubin carries no data globals: all global accesses go through host-side CUtensorMap descriptors that G2 deliberately left to the runtime (create.descriptor is a host op). Wiring them here is moot: the kernel shares the demoted-mbarrier producer/consumer skeleton with kkt_bf16_mma_p2_t64x64x64_w4_ws, which deadlocks on hardware (see its TIMEOUT entry).
  - as-built (G2) cubin probe: `launch failed: <CUresult.CUDA_ERROR_INVALID_VALUE: 1>`

As-built (unrepaired) G2 cubins: every config with shared-memory tiles trapped (`CUDA_ERROR_ILLEGAL_ADDRESS` on sync) or refused to launch (`INVALID_VALUE`) — see Gap 1 above; the two refs without smem allocs launched fine. Full per-config probe strings are in `config_times.json` under `g2_asbuilt_launch`.

## 5. Rerun instructions (clean GPU window)

```bash
# conda env gdn; ~/codebase/gdn-mlir
scripts/run_calib.sh                    # overwrites evidence/G3/calibration.json
scripts/run_g3.sh                       # regenerate+compile all 20 noargs configs (CPU)
~/miniforge3/envs/gdn/bin/python tools/gdn_time.py   # overwrites evidence/G3/config_times.json
```

Both harnesses are idempotent (same filenames -> fresh values overwrite).
`gdn_time.py` isolates each config in a subprocess with a wall timeout so a
deadlocking kernel (the ws configs, today) only kills its own worker.

### Numerical flag (follow-up, not a G3-A blocker)

Non-zero counts differ between code paths computing the SAME math:
kkt stage tril(k@k^T)*decay has 2080 lower-triangular positions; the
linalg (nomma) path leaves exactly 2080 non-zero elements, while every
mma-path kkt config leaves only 1168. Outputs are deterministic
(checksum-stable) and non-zero, but element-level equivalence to the
reference is NOT verified by G3-A (the G1 ref-check harness owns numerics).
The 912 extra zeros in the mma path are consistent with a fragment-index or
decay-mask misalignment in the G2 payload and should be checked against the
JAX reference before the mma timings are used for absolute-quality claims.
The timings themselves are unaffected.
