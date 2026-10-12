// calib_bench.cu -- G3-A one-time sm_120 (RTX 5090 Laptop, Blackwell)
// hardware calibration microbench.
//
// Measures (median of >=50 timed iterations after >=5 warmups, CUDA events):
//   a. mma.sync.m16n8k16.bf16.f32 tensor-core peak   (FLOP/s, instr/SM/clk)
//   b. shared-memory read bandwidth (ld.shared 16B)  (GB/s)
//      gmem DRAM read bandwidth + L2-resident read bandwidth (GB/s)
//   c. SFU ex2.approx.f32 throughput                 (elem/s)
//   d. bar.sync round-trip cost, 4-warp block        (ns)
//   e. cp.async 16B gmem->smem copy throughput       (GB/s) [bonus]
//
// Every kernel writes its accumulator through a never-taken branch to defeat
// dead-code elimination.  Numbers are self-relative ONLY if the GPU is
// otherwise idle; this harness is designed to run opportunistically next to a
// co-running job (see evidence/G3/calib_report.md shared-GPU caveat).
// Re-runnable: prints one JSON object to stdout; the wrapper merges metadata.
//
// Build: nvcc -arch=sm_120a -O3 calib_bench.cu -o calib_bench

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), \
          __FILE__, __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------- a. mma ---
// One mma.sync.m16n8k16 bf16 = 4096 FLOP. 8-deep ILP per thread.
__global__ void __launch_bounds__(128) k_mma(int reps, float* sink) {
  uint32_t a0 = 0x3f803f80u + threadIdx.x, a1 = a0 ^ 0x00ff00ffu;
  uint32_t a2 = a0 + 1, a3 = a1 + 1;
  uint32_t b0 = 0xbf80bf80u + blockIdx.x, b1 = b0 ^ 0x00010001u;
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  for (int it = 0; it < reps; ++it) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  if (c0 == 1e30f) sink[blockIdx.x] = c0 + c1 + c2 + c3;
}

// ------------------------------------------------------------ b. smem bw ---
__global__ void __launch_bounds__(256) k_smem(int iters, uint4* sink) {
  __shared__ uint4 buf[3072];  // 48 KB
  buf[threadIdx.x] = make_uint4(threadIdx.x, 1u, 2u, 3u);
  __syncthreads();
  uint4 x = make_uint4(0, 0, 0, 0);
  int tid = threadIdx.x;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      uint4 v = buf[(tid + (it * 8 + u) * 256) & 3071];
      x.x ^= v.x; x.y ^= v.y; x.z ^= v.z; x.w ^= v.w;
    }
  }
  if (x.x == 0xdeadbeefu) sink[blockIdx.x] = x;
}

// ------------------------------------------------------- b2. gmem reads ---
__global__ void __launch_bounds__(256) k_gread(const uint4* __restrict__ g,
                                               size_t n4, uint4* sink) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  uint4 x = make_uint4(0, 0, 0, 0);
  for (; i < n4; i += stride) {
    uint4 v = g[i];
    x.x ^= v.x; x.y ^= v.y; x.z ^= v.z; x.w ^= v.w;
  }
  if (x.x == 0xdeadbeefu) sink[blockIdx.x] = x;
}

// --------------------------------------------------------------- c. SFU ---
__global__ void __launch_bounds__(128) k_ex2(int reps, float* sink) {
  float x0 = threadIdx.x * 1e-3f, x1 = x0 + 0.1f, x2 = x0 + 0.2f,
        x3 = x0 + 0.3f;
  for (int it = 0; it < reps; ++it) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      asm volatile("ex2.approx.f32 %0, %1;" : "=f"(x0) : "f"(x0));
      asm volatile("ex2.approx.f32 %0, %1;" : "=f"(x1) : "f"(x1));
      asm volatile("ex2.approx.f32 %0, %1;" : "=f"(x2) : "f"(x2));
      asm volatile("ex2.approx.f32 %0, %1;" : "=f"(x3) : "f"(x3));
    }
  }
  if (x0 == 1e30f) sink[blockIdx.x] = x0 + x1 + x2 + x3;
}

// ------------------------------------------------------- d. bar.sync rt ---
__global__ void __launch_bounds__(128) k_bar(int reps, float* sink) {
  for (int it = 0; it < reps; ++it)
    asm volatile("bar.sync 0, 128;");
  if (threadIdx.x == 1000) sink[blockIdx.x] = 1.f;
}

// ------------------------------------------------ e. cp.async 16B [bonus] -
__global__ void __launch_bounds__(128) k_cpasync(int iters,
                                                 const uint4* __restrict__ g,
                                                 uint4* sink) {
  __shared__ uint4 buf[64][8];  // 8 KB staging
  int tid = threadIdx.x;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      size_t off = ((size_t)(it * 512 + u * 128 + tid)) & ((1u << 16) - 1);
      unsigned s = (unsigned)__cvta_generic_to_shared(&buf[tid & 63][u * 2]);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
                   "l"(g + off));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("bar.sync 0, 128;");
  }
  if (buf[0][0].x == 0xdeadbeefu) sink[blockIdx.x] = buf[0][0];
}

// ----------------------------------------------------------- timing util ---
struct Times {
  float median_ms, min_ms, max_ms;
};

static Times time_kernel(void (*launch)(void*), void* arg, int warmup,
                         int iters) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  std::vector<float> t;
  t.reserve(iters);
  for (int i = 0; i < warmup; ++i) launch(arg);
  CK(cudaDeviceSynchronize());
  for (int i = 0; i < iters; ++i) {
    CK(cudaEventRecord(a));
    launch(arg);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  CK(cudaEventDestroy(a));
  CK(cudaEventDestroy(b));
  return {t[t.size() / 2], t.front(), t.back()};
}

// launch trampolines (kernel<<<>>> inside a function pointer)
struct Args1 { int n; float* sink; int grid; };
static void l_mma(void* p) { Args1* a = (Args1*)p; k_mma<<<a->grid, 128>>>(a->n, a->sink); }
static void l_ex2(void* p) { Args1* a = (Args1*)p; k_ex2<<<a->grid, 128>>>(a->n, a->sink); }
static void l_bar(void* p) { Args1* a = (Args1*)p; k_bar<<<a->grid, 128>>>(a->n, a->sink); }

struct ArgsS { int n; uint4* sink; int grid; };
static void l_smem(void* p) { ArgsS* a = (ArgsS*)p; k_smem<<<a->grid, 256>>>(a->n, a->sink); }

struct ArgsG { const uint4* g; size_t n4; uint4* sink; int grid; };
static void l_gread(void* p) { ArgsG* a = (ArgsG*)p; k_gread<<<a->grid, 256>>>(a->g, a->n4, a->sink); }

struct ArgsC { int n; const uint4* g; uint4* sink; int grid; };
static void l_cpasync(void* p) { ArgsC* a = (ArgsC*)p; k_cpasync<<<a->grid, 128>>>(a->n, a->g, a->sink); }

int main() {
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  int l2 = 0, clk_khz = 0;
  CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
  CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
  const int nsm = prop.multiProcessorCount;
  const int grid8 = nsm * 8, grid1 = nsm;
  const double clk = clk_khz * 1e3;  // Hz (peak)

  float* fsink = nullptr;
  uint4* usink = nullptr;
  CK(cudaMalloc(&fsink, grid8 * sizeof(float)));
  CK(cudaMalloc(&usink, grid8 * sizeof(uint4)));

  // gmem buffers: L2-resident regime (<= 60% of L2) and DRAM regime (512 MB)
  size_t l2fit = std::min<size_t>(32u << 20, (size_t)(l2 * 0.6));
  size_t big = 512u << 20;
  uint4 *gfit = nullptr, *gbig = nullptr;
  CK(cudaMalloc(&gfit, l2fit));
  if (cudaMalloc(&gbig, big) != cudaSuccess) { big = 256u << 20; CK(cudaMalloc(&gbig, big)); }
  CK(cudaMemset(gfit, 1, l2fit));
  CK(cudaMemset(gbig, 1, big));

  printf("{\n");
  printf("  \"device\": {\"name\": \"%s\", \"sm_count\": %d, "
         "\"l2_bytes\": %d, \"clock_peak_khz\": %d, "
         "\"clock_peak_mhz\": %.0f, \"shared_per_sm_optin\": %d, "
         "\"regs_per_sm\": %d},\n",
         prop.name, nsm, l2, clk_khz, clk / 1e6,
         (int)prop.sharedMemPerBlockOptin, (int)prop.regsPerMultiprocessor);

  // ---- a. mma peak ----
  {
    Args1 a{2048, fsink, grid8};
    Times t = time_kernel(l_mma, &a, 5, 50);
    double instr = (double)grid8 * 128 * 2048 * 8;
    double s = t.median_ms * 1e-3;
    printf("  \"mma_bf16_m16n8k16\": {\"median_ms\": %.6f, \"min_ms\": %.6f, "
           "\"max_ms\": %.6f, \"instr_per_s\": %.4e, \"flops\": %.4e, "
           "\"instr_per_sm_clk_peaknorm\": %.2f, \"kernel_reps\": 2048, "
           "\"ilp\": 8, \"warmup\": 5, \"iters\": 50},\n",
           t.median_ms, t.min_ms, t.max_ms, instr / s, instr / s * 4096.0,
           instr / s / nsm / clk);
  }
  // ---- b1. smem read bw ----
  {
    ArgsS a{2000, usink, grid8};
    Times t = time_kernel(l_smem, &a, 5, 50);
    double by = (double)grid8 * 256 * 2000 * 8 * 16;
    double s = t.median_ms * 1e-3;
    printf("  \"smem_read_bw_gbps\": {\"median_ms\": %.6f, \"min_ms\": %.6f, "
           "\"max_ms\": %.6f, \"gbps\": %.2f, \"bytes_moved\": %.3e},\n",
           t.median_ms, t.min_ms, t.max_ms, by / s / 1e9, by);
  }
  // ---- b2. gmem L2-resident read bw ----
  {
    ArgsG a{gfit, l2fit / 16, usink, grid8};
    Times t = time_kernel(l_gread, &a, 5, 50);
    double s = t.median_ms * 1e-3;
    printf("  \"gmem_read_l2resident\": {\"median_ms\": %.6f, \"min_ms\": "
           "%.6f, \"max_ms\": %.6f, \"gbps\": %.2f, \"buffer_mb\": %.1f},\n",
           t.median_ms, t.min_ms, t.max_ms, l2fit / s / 1e9,
           l2fit / 1048576.0);
  }
  // ---- b3. gmem DRAM read bw ----
  {
    ArgsG a{gbig, big / 16, usink, grid8};
    Times t = time_kernel(l_gread, &a, 5, 50);
    double s = t.median_ms * 1e-3;
    printf("  \"gmem_read_dram\": {\"median_ms\": %.6f, \"min_ms\": %.6f, "
           "\"max_ms\": %.6f, \"gbps\": %.2f, \"buffer_mb\": %.0f},\n",
           t.median_ms, t.min_ms, t.max_ms, big / s / 1e9, big / 1048576.0);
  }
  // ---- c. SFU ex2 ----
  {
    Args1 a{4096, fsink, grid8};
    Times t = time_kernel(l_ex2, &a, 5, 50);
    double el = (double)grid8 * 128 * 4096 * 8 * 4;
    double s = t.median_ms * 1e-3;
    printf("  \"sfu_ex2_f32\": {\"median_ms\": %.6f, \"min_ms\": %.6f, "
           "\"max_ms\": %.6f, \"elem_per_s\": %.4e, \"elem_per_sm_clk\": %.3f},\n",
           t.median_ms, t.min_ms, t.max_ms, el / s, el / s / nsm / clk);
  }
  // ---- d. bar.sync round trip ----
  {
    Args1 a{20000, fsink, grid1};
    Times t = time_kernel(l_bar, &a, 5, 50);
    printf("  \"bar_sync_roundtrip\": {\"median_ms\": %.6f, \"min_ms\": %.6f, "
           "\"max_ms\": %.6f, \"ns_per_bar\": %.3f, \"reps\": 20000, "
           "\"block_threads\": 128},\n",
           t.median_ms, t.min_ms, t.max_ms, t.median_ms * 1e6 / 20000);
  }
  // ---- e. cp.async 16B ----
  {
    ArgsC a{4000, gfit, usink, grid8};
    Times t = time_kernel(l_cpasync, &a, 5, 50);
    double by = (double)grid8 * 128 * 4000 * 4 * 16;
    double s = t.median_ms * 1e-3;
    printf("  \"cp_async_16b\": {\"median_ms\": %.6f, \"min_ms\": %.6f, "
           "\"max_ms\": %.6f, \"gbps\": %.2f, \"src_window_mb\": 1.0}\n",
           t.median_ms, t.min_ms, t.max_ms, by / s / 1e9);
  }
  printf("}\n");
  return 0;
}
