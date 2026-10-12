#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tilelang_fused_chunk_gdr_bwd_kernel_kernel(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const bfloat16_t* __restrict__ b, bfloat16_t* __restrict__ db, float* __restrict__ dg, bfloat16_t* __restrict__ dk, __grid_constant__ const CUtensorMap dk_desc, const bfloat16_t* __restrict__ do_1, __grid_constant__ const CUtensorMap do_desc, bfloat16_t* __restrict__ dq, __grid_constant__ const CUtensorMap dq_desc, bfloat16_t* __restrict__ dv, __grid_constant__ const CUtensorMap dv_desc, const float* __restrict__ g, const bfloat16_t* __restrict__ h, __grid_constant__ const CUtensorMap h_desc, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const bfloat16_t* __restrict__ v, int batch_size, int num_chunks, int num_tokens);
extern "C" __global__ void __launch_bounds__(512, 1) tilelang_fused_chunk_gdr_bwd_kernel_kernel(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const bfloat16_t* __restrict__ b, bfloat16_t* __restrict__ db, float* __restrict__ dg, bfloat16_t* __restrict__ dk, __grid_constant__ const CUtensorMap dk_desc, const bfloat16_t* __restrict__ do_1, __grid_constant__ const CUtensorMap do_desc, bfloat16_t* __restrict__ dq, __grid_constant__ const CUtensorMap dq_desc, bfloat16_t* __restrict__ dv, __grid_constant__ const CUtensorMap dv_desc, const float* __restrict__ g, const bfloat16_t* __restrict__ h, __grid_constant__ const CUtensorMap h_desc, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const bfloat16_t* __restrict__ v, int batch_size, int num_chunks, int num_tokens) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* h_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* tmp_shared_2_1 = ((void*)((char*)buf_dyn_shmem + 16384));
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  void* do_shared = ((void*)((char*)buf_dyn_shmem + 57344));
  void* dqkv_shared = ((void*)((char*)buf_dyn_shmem + 73728));
  void* tmp_shared_2_2 = ((void*)((char*)buf_dyn_shmem + 90112));
  void* tmp_shared_2_3 = ((void*)((char*)buf_dyn_shmem + 106496));
  void* tmp_shared_4_1 = ((void*)((char*)buf_dyn_shmem + 122880));
  void* tmp_shared_1_1 = ((void*)((char*)buf_dyn_shmem + 139264));
  void* tmp_shared_1_2 = ((void*)((char*)buf_dyn_shmem + 147456));
  void* tmp_shared_1_3 = ((void*)((char*)buf_dyn_shmem + 155648));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 163840));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 164352));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 164864));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 165376));
  void* workspace_4 = ((void*)((char*)buf_dyn_shmem + 165888));
  void* workspace_5 = ((void*)((char*)buf_dyn_shmem + 166400));
  void* workspace_6 = ((void*)((char*)buf_dyn_shmem + 166912));
  __shared__ __align__(16) uint64_t bar_00_mem[1];
  auto bar_00 = reinterpret_cast<Barrier*>(bar_00_mem);
  __shared__ __align__(16) uint64_t bar_01_mem[1];
  auto bar_01 = reinterpret_cast<Barrier*>(bar_01_mem);
  __shared__ __align__(16) uint64_t bar_02_mem[1];
  auto bar_02 = reinterpret_cast<Barrier*>(bar_02_mem);
  __shared__ __align__(16) uint64_t bar_03_mem[1];
  auto bar_03 = reinterpret_cast<Barrier*>(bar_03_mem);
  __shared__ __align__(16) uint64_t bar_04_mem[1];
  auto bar_04 = reinterpret_cast<Barrier*>(bar_04_mem);
  __shared__ __align__(16) uint64_t bar_05_mem[1];
  auto bar_05 = reinterpret_cast<Barrier*>(bar_05_mem);
  __shared__ __align__(16) uint64_t bar_06_mem[1];
  auto bar_06 = reinterpret_cast<Barrier*>(bar_06_mem);
  __shared__ __align__(16) uint64_t bar_07_mem[1];
  auto bar_07 = reinterpret_cast<Barrier*>(bar_07_mem);
  __shared__ __align__(16) uint64_t bar_08_1_mem[1];
  auto bar_08_1 = reinterpret_cast<Barrier*>(bar_08_1_mem);
  __shared__ __align__(16) uint64_t bar_08_2_mem[1];
  auto bar_08_2 = reinterpret_cast<Barrier*>(bar_08_2_mem);
  __shared__ __align__(16) uint64_t bar_08_3_mem[1];
  auto bar_08_3 = reinterpret_cast<Barrier*>(bar_08_3_mem);
  __shared__ __align__(16) uint64_t bar_k_left_ready_mem[1];
  auto bar_k_left_ready = reinterpret_cast<Barrier*>(bar_k_left_ready_mem);
  __shared__ __align__(16) uint64_t bar_dhtmp_ready_mem[1];
  auto bar_dhtmp_ready = reinterpret_cast<Barrier*>(bar_dhtmp_ready_mem);
  __shared__ __align__(16) uint64_t bar_dh_right_ready_mem[1];
  auto bar_dh_right_ready = reinterpret_cast<Barrier*>(bar_dh_right_ready_mem);
  __shared__ __align__(16) uint64_t bar_s4_dot_a_mem[1];
  auto bar_s4_dot_a = reinterpret_cast<Barrier*>(bar_s4_dot_a_mem);
  __shared__ __align__(16) uint64_t bar_dg_k_init_mem[1];
  auto bar_dg_k_init = reinterpret_cast<Barrier*>(bar_dg_k_init_mem);
  __shared__ __align__(16) uint64_t bar_dg_k_rows_mem[1];
  auto bar_dg_k_rows = reinterpret_cast<Barrier*>(bar_dg_k_rows_mem);
  __shared__ __align__(16) uint64_t bar_dg_a_first_mem[1];
  auto bar_dg_a_first = reinterpret_cast<Barrier*>(bar_dg_a_first_mem);
  __shared__ __align__(16) uint64_t bar_09_mem[1];
  auto bar_09 = reinterpret_cast<Barrier*>(bar_09_mem);
  __shared__ __align__(16) uint64_t bar_10_mem[1];
  auto bar_10 = reinterpret_cast<Barrier*>(bar_10_mem);
  __shared__ __align__(16) uint64_t bar_11_mem[1];
  auto bar_11 = reinterpret_cast<Barrier*>(bar_11_mem);
  __shared__ __align__(16) uint64_t bar_12_mem[1];
  auto bar_12 = reinterpret_cast<Barrier*>(bar_12_mem);
  __shared__ __align__(16) uint64_t bar_13_mem[1];
  auto bar_13 = reinterpret_cast<Barrier*>(bar_13_mem);
  __shared__ __align__(16) uint64_t bar_14_mem[1];
  auto bar_14 = reinterpret_cast<Barrier*>(bar_14_mem);
  __shared__ __align__(16) uint64_t bar_15_mem[1];
  auto bar_15 = reinterpret_cast<Barrier*>(bar_15_mem);
  int batch_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int chunk_start_idx = 0;
  int num_iters = 0;
  __shared__ __align__(16) float g_shared[64];
  __shared__ __align__(16) float b_shared[64];
  float dh_fragment_0[64];
  float dh_fragment_1[64];
  __shared__ __align__(16) float g_exp_shared[64];
  __shared__ __align__(16) float g_rev_exp_shared[64];
  __shared__ __align__(16) float dg_shared[64];
  float dk_frag_l[32];
  float dk_frag_r[32];
  __shared__ __align__(16) float db_shared[64];
  float g_last_local_3[1];
  float reduce_fragment[2];
  float2 dg_last_local_3_pack[1];
  float dg_last_local_3[1];
  bfloat16_t dh_left_fragment[64];
  float dv_fragment[64];
  float odot_fragment_1[64];
  float dg_fragment_1[4];
  float2 dg_last_local_1_pack[1];
  float dg_last_local_1[1];
  float p_fragment[32];
  float mask_fragment[32];
  float a_fragment[32];
  float odot_fragment_2[64];
  float u_fragment[64];
  float da_fragment[32];
  float dp_fragment[32];
  float dq_frag_r[32];
  float dq_frag_l[32];
  float2 dg_fragment_2_pack[8];
  float dg_fragment_2[8];
  ushort lo_fragment[32];
  ushort hi_fragment[32];
  uint uint32_fragment[32];
  float dg_fragment_2b[4];
  float db_fragment[4];
  float db_fragment_clear[4];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(q_desc);
    tl::prefetch_tma_descriptor(dk_desc);
    tl::prefetch_tma_descriptor(dv_desc);
    tl::prefetch_tma_descriptor(dq_desc);
    tl::prefetch_tma_descriptor(h_desc);
    tl::prefetch_tma_descriptor(a_desc);
    tl::prefetch_tma_descriptor(do_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    bar_00[0].init(448);
    bar_01[0].init(384);
    bar_02[0].init(288);
    bar_03[0].init(256);
    bar_04[0].init(416);
    bar_05[0].init(288);
    bar_06[0].init(256);
    bar_07[0].init(256);
    bar_08_1[0].init(256);
    bar_08_2[0].init(384);
    bar_08_3[0].init(128);
    bar_k_left_ready[0].init(128);
    bar_dhtmp_ready[0].init(128);
    bar_dh_right_ready[0].init(128);
    bar_s4_dot_a[0].init(128);
    bar_dg_k_init[0].init(128);
    bar_dg_k_rows[0].init(128);
    bar_dg_a_first[0].init(128);
    bar_09[0].init(256);
    bar_10[0].init(288);
    bar_11[0].init(256);
    bar_12[0].init(128);
    bar_13[0].init(256);
    bar_14[0].init(256);
    bar_15[0].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = (((int)blockIdx.x) >> 4);
  seq_start_idx = 0;
  seq_end_idx = num_tokens;
  chunk_start_idx = 0;
  num_iters = (((seq_end_idx + 63) - seq_start_idx) >> 6);
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if (((((1 <= (chunk_start_idx + num_iters)) && ((chunk_start_idx + num_iters) <= num_chunks)) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
      condval = *(uint4*)(h + (((((((((int64_t)chunk_start_idx) * (int64_t)262144) + (((int64_t)num_iters) * (int64_t)262144)) + ((((int64_t)batch_idx) * ((int64_t)num_chunks)) * (int64_t)262144)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)16384)) + (((int64_t)i) * (int64_t)4096)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) - (int64_t)262144));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
    }
    *(uint4*)(((bfloat16_t*)h_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    if (((((num_iters * 64) + (i_1 * 32)) + (((int)threadIdx.x) >> 4)) + seq_start_idx) < (seq_end_idx + 64)) {
      bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_1;
      if (((((1 <= (((((i_1 * 32) + (((int)threadIdx.x) >> 4)) + seq_start_idx) >> 6) + num_iters)) && (((((num_iters * 64) + (i_1 * 32)) + (((int)threadIdx.x) >> 4)) + seq_start_idx) < (num_tokens + 64))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
        condval_1 = *(uint4*)(q + ((((((((((int64_t)num_iters) * (int64_t)131072) + (((int64_t)i_1) * (int64_t)65536)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)seq_start_idx) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)131072));
      } else {
        condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
      }
      *(uint4*)(((bfloat16_t*)tmp_shared_2_1) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_1 * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_1;
    } else {
      bfloat16_t broadcast_var_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      *(uint4*)(((bfloat16_t*)tmp_shared_2_1) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_1 * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2));
    }
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    if (((((num_iters * 64) + (i_2 * 32)) + (((int)threadIdx.x) >> 4)) + seq_start_idx) < (seq_end_idx + 64)) {
      bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_2;
      if (((((1 <= (((((i_2 * 32) + (((int)threadIdx.x) >> 4)) + seq_start_idx) >> 6) + num_iters)) && (((((num_iters * 64) + (i_2 * 32)) + (((int)threadIdx.x) >> 4)) + seq_start_idx) < (num_tokens + 64))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
        condval_2 = *(uint4*)(k + ((((((((((int64_t)num_iters) * (int64_t)131072) + (((int64_t)i_2) * (int64_t)65536)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)seq_start_idx) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)131072));
      } else {
        condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
      }
      *(uint4*)(((bfloat16_t*)k_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
    } else {
      bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      *(uint4*)(((bfloat16_t*)k_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
    }
  }
  if ((((num_iters * 64) + (((int)threadIdx.x) >> 3)) + seq_start_idx) < (seq_end_idx + 64)) {
    bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval_3;
    if (((((1 <= ((((((int)threadIdx.x) >> 3) + seq_start_idx) >> 6) + num_iters)) && ((((num_iters * 64) + (((int)threadIdx.x) >> 3)) + seq_start_idx) < (num_tokens + 64))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
      condval_3 = *(uint4*)(a + (((((((((int64_t)num_iters) * (int64_t)65536) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)1024)) + (((int64_t)seq_start_idx) * (int64_t)1024)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)65536));
    } else {
      condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
    }
    *(uint4*)(((bfloat16_t*)a_shared) + (((((((int)threadIdx.x) >> 3) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_3;
  } else {
    bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    *(uint4*)(((bfloat16_t*)a_shared) + (((((((int)threadIdx.x) >> 3) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    if (((((num_iters * 64) + (i_3 * 32)) + (((int)threadIdx.x) >> 4)) + seq_start_idx) < (seq_end_idx + 64)) {
      bfloat16_t broadcast_var_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_4;
      if (((((1 <= (((((i_3 * 32) + (((int)threadIdx.x) >> 4)) + seq_start_idx) >> 6) + num_iters)) && (((((num_iters * 64) + (i_3 * 32)) + (((int)threadIdx.x) >> 4)) + seq_start_idx) < (num_tokens + 64))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
        condval_4 = *(uint4*)(do_1 + ((((((((((int64_t)num_iters) * (int64_t)131072) + (((int64_t)i_3) * (int64_t)65536)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)seq_start_idx) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)131072));
      } else {
        condval_4 = make_uint4(__pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7));
      }
      *(uint4*)(((bfloat16_t*)do_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_4;
    } else {
      bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      *(uint4*)(((bfloat16_t*)do_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 2048)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
    }
  }
  if (((int)threadIdx.x) < 64) {
    if ((((num_iters * 64) + seq_start_idx) + ((int)threadIdx.x)) < (seq_end_idx + 64)) {
      float condval_5;
      if (((((1 <= (((seq_start_idx + ((int)threadIdx.x)) >> 6) + num_iters)) && ((((num_iters * 64) + seq_start_idx) + ((int)threadIdx.x)) < (num_tokens + 64))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
        condval_5 = g[((((((((int64_t)num_iters) * (int64_t)1024) + (((int64_t)seq_start_idx) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)1024)];
      } else {
        condval_5 = 0x0p+0f/*0.000000e+00*/;
      }
      g_shared[((int)threadIdx.x)] = condval_5;
    } else {
      float condval_6;
      if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
        condval_6 = g[((((((int64_t)seq_end_idx) * (int64_t)16) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)16)];
      } else {
        condval_6 = 0x0p+0f/*0.000000e+00*/;
      }
      g_shared[((int)threadIdx.x)] = condval_6;
    }
    if ((((num_iters * 64) + seq_start_idx) + ((int)threadIdx.x)) < (seq_end_idx + 64)) {
      bfloat16_t condval_7;
      if (((((1 <= (((seq_start_idx + ((int)threadIdx.x)) >> 6) + num_iters)) && ((((num_iters * 64) + seq_start_idx) + ((int)threadIdx.x)) < (num_tokens + 64))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
        condval_7 = b[((((((((int64_t)num_iters) * (int64_t)1024) + (((int64_t)seq_start_idx) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)1024)];
      } else {
        condval_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      }
      b_shared[((int)threadIdx.x)] = ((float)condval_7);
    } else {
      b_shared[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
    }
  }
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<160>();
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dh_fragment_0 + (i_4 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      float broadcast_var_10 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dh_fragment_1 + (i_5 * 4)) = make_float4(broadcast_var_10, broadcast_var_10, broadcast_var_10, broadcast_var_10);
    }
    bar_dhtmp_ready[0].arrive();
    for (int i_s = 0; i_s < num_iters; ++i_s) {
      int cur_idx = (((chunk_start_idx + num_iters) - i_s) - 1);
      bar_00[0].arrive();
      bar_00[0].wait((i_s & 1));
      tl::__sync_thread_partial(3, 128);
      if (((int)threadIdx.x) < 64) {
        g_exp_shared[((int)threadIdx.x)] = exp2f((g_shared[((int)threadIdx.x)] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
        g_rev_exp_shared[((int)threadIdx.x)] = exp2f(((g_shared[63] - g_shared[((int)threadIdx.x)]) * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
      }
      bar_01[0].arrive();
      bar_01[0].wait((i_s & 1));
      if (0 < i_s) {
        bar_k_left_ready[0].wait(((i_s + 1) & 1));
        tl::__sync_thread_partial(3, 128);
        #pragma unroll
        for (int i_6 = 0; i_6 < 8; ++i_6) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_4_1)[(((((((((int)threadIdx.x) >> 5) * 2048) + ((i_6 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_6 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)dh_fragment_1[(i_6 * 8)]), ((bfloat16_t)dh_fragment_1[((i_6 * 8) + 1)])), __pack_half2(((bfloat16_t)dh_fragment_1[((i_6 * 8) + 2)]), ((bfloat16_t)dh_fragment_1[((i_6 * 8) + 3)])), __pack_half2(((bfloat16_t)dh_fragment_1[((i_6 * 8) + 4)]), ((bfloat16_t)dh_fragment_1[((i_6 * 8) + 5)])), __pack_half2(((bfloat16_t)dh_fragment_1[((i_6 * 8) + 6)]), ((bfloat16_t)dh_fragment_1[((i_6 * 8) + 7)])));
        }
        tl::fence_proxy_async();
        bar_dh_right_ready[0].arrive();
      }
      tl::__sync_thread_partial(3, 128);
      g_last_local_3[0] = g_exp_shared[63];
      #pragma unroll
      for (int i_7 = 0; i_7 < 16; ++i_7) {
        float4 __1;
          float4 v_ = *(float4*)(dh_fragment_0 + (i_7 * 4));
          float4 v__1 = make_float4(g_last_local_3[0], g_last_local_3[0], g_last_local_3[0], g_last_local_3[0]);
          *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
          *(float2*)(&(__1.z)) = tl::mul2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
        *(float4*)(dh_fragment_0 + (i_7 * 4)) = __1;
        float4 __2;
          float4 v__2 = *(float4*)(dh_fragment_1 + (i_7 * 4));
          float4 v__3 = make_float4(g_last_local_3[0], g_last_local_3[0], g_last_local_3[0], g_last_local_3[0]);
          *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
          *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(v__2.z)), *(float2*)(&(v__3.z)));
        *(float4*)(dh_fragment_1 + (i_7 * 4)) = __2;
      }
      bar_04[0].arrive();
      bar_04[0].wait((i_s & 1));
      float broadcast_var_11 = 0x0p+0f/*0.000000e+00*/;
      *(float2*)(reduce_fragment + 0) = make_float2(broadcast_var_11, broadcast_var_11);
      #pragma unroll
      for (int i_8 = 0; i_8 < 32; ++i_8) {
        bfloat16_t h_shared_local_cast[2];
        *(uint1*)(h_shared_local_cast + 0) = *(uint1*)(((bfloat16_t*)h_shared) + (((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_8 >> 4) * 1024)) + ((i_8 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_8 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_8 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_8 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __3;
          float2 v__4 = *(float2*)(dh_fragment_1 + (i_8 * 2));
          float2 __4;
          uint1 v__5 = *(uint1*)(h_shared_local_cast + 0);
          ((float2*)(&__4))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__5))[0]);
          float2 v__6 = *(float2*)(reduce_fragment + 0);
          *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(v__4.x)), *(float2*)(&(__4.x)), *(float2*)(&(v__6.x)));
        *(float2*)(reduce_fragment + 0) = __3;
      }
      bar_08_2[0].arrive();
      bar_08_2[0].wait((i_s & 1));
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_9 = 0; i_9 < 8; ++i_9) {
        bfloat16_t broadcast_var_12 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_8;
        if (((((0 <= cur_idx) && (cur_idx < num_chunks)) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
          condval_8 = *(uint4*)(h + (((((((int64_t)cur_idx) * (int64_t)262144) + ((((int64_t)batch_idx) * ((int64_t)num_chunks)) * (int64_t)262144)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)16384)) + (((int64_t)i_9) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
        } else {
          condval_8 = make_uint4(__pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12));
        }
        *(uint4*)(((bfloat16_t*)h_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_9 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_8;
      }
      bar_08_3[0].arrive();
      bar_08_3[0].wait((i_s & 1));
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_10 = 0; i_10 < 32; ++i_10) {
        bfloat16_t h_shared_local_cast_1[2];
        *(uint1*)(h_shared_local_cast_1 + 0) = *(uint1*)(((bfloat16_t*)h_shared) + ((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_10 >> 3) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_10 & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_10 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_10 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __5;
          float2 v__7 = *(float2*)(dh_fragment_0 + ((((i_10 >> 4) * 32) + ((i_10 & 7) * 4)) + (((i_10 & 15) >> 3) * 2)));
          float2 __6;
          uint1 v__8 = *(uint1*)(h_shared_local_cast_1 + 0);
          ((float2*)(&__6))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__8))[0]);
          float2 v__9 = *(float2*)(reduce_fragment + 0);
          *(float2*)(&(__5.x)) = tl::fma2(*(float2*)(&(v__7.x)), *(float2*)(&(__6.x)), *(float2*)(&(v__9.x)));
        *(float2*)(reduce_fragment + 0) = __5;
      }
      bar_09[0].arrive();
      bar_09[0].wait((i_s & 1));
      bar_10[0].wait((i_s & 1));
      float broadcast_var_13 = 0x0p+0f/*0.000000e+00*/;
      dg_last_local_3_pack[0] = make_float2(broadcast_var_13, broadcast_var_13);
      dg_last_local_3_pack[0] = tl::add2(dg_last_local_3_pack[0], *(float2*)(reduce_fragment + 0));
      dg_last_local_3[0] = ((dg_last_local_3_pack[0]).x + (dg_last_local_3_pack[0]).y);
      tl::__sync_thread_partial(3, 128);
      dg_last_local_3[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(dg_last_local_3[0], (&(((float*)workspace_1)[0])));
      if (((int)threadIdx.x) == 0) {
        dg_shared[63] = (dg_shared[63] + dg_last_local_3[0]);
      }
      bar_11[0].arrive();
      bar_11[0].wait((i_s & 1));
      {
        bfloat16_t A_local[16];
        bfloat16_t B_local[32];
        for (int ki = 0; ki < 4; ++ki) {
          #pragma unroll
          for (int i_11 = 0; i_11 < 2; ++i_11) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_2)[((((((ki * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_11) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_11 * 8)])));
          }
          #pragma unroll
          for (int i_12 = 0; i_12 < 4; ++i_12) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_3)[(((((((((int)threadIdx.x) >> 6) * 4096) + (ki * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_12 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_12 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_12 * 8)])));
          }
          for (int i_13 = 0; i_13 < 2; ++i_13) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_0 + ((i_13 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_13 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_0 + (((i_13 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_13 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      {
        bfloat16_t A_local_1[16];
        bfloat16_t B_local_1[32];
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          #pragma unroll
          for (int i_14 = 0; i_14 < 2; ++i_14) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_2)[(((((((ki_1 * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(A_local_1[(i_14 * 8)])));
          }
          #pragma unroll
          for (int i_15 = 0; i_15 < 4; ++i_15) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_3)[(((((((((int)threadIdx.x) >> 6) * 4096) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_15 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_15 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_15 * 8)])));
          }
          for (int i_16 = 0; i_16 < 2; ++i_16) {
            for (int j_1 = 0; j_1 < 4; ++j_1) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_1 + ((i_16 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_16 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_1 + (((i_16 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_16 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
            }
          }
        }
      }
      bar_12[0].arrive();
      bar_12[0].wait((i_s & 1));
      bar_13[0].wait((i_s & 1));
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_17 = 0; i_17 < 8; ++i_17) {
        bfloat16_t do_shared_local_cast_3[8];
        float g_exp_shared_local_cast_4[8];
        bfloat16_t tmp_shared_2_3_local_cast_2[8];
        *(uint4*)(do_shared_local_cast_3 + 0) = *(uint4*)(((bfloat16_t*)do_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_17 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
        for (int vec_copy = 0; vec_copy < 2; ++vec_copy) {
          *(float4*)(g_exp_shared_local_cast_4 + (vec_copy * 4)) = make_float4(g_exp_shared[((i_17 * 8) + (((int)threadIdx.x) >> 4))], g_exp_shared[((i_17 * 8) + (((int)threadIdx.x) >> 4))], g_exp_shared[((i_17 * 8) + (((int)threadIdx.x) >> 4))], g_exp_shared[((i_17 * 8) + (((int)threadIdx.x) >> 4))]);
        }
        for (int vec = 0; vec < 2; ++vec) {
          float broadcast_var_14 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
          uint2 __7;
          float4 __8;
            float4 __9;
              float4 v__10 = make_float4(broadcast_var_14, broadcast_var_14, broadcast_var_14, broadcast_var_14);
              float4 __10;
              uint2 v__11 = *(uint2*)(do_shared_local_cast_3 + (vec * 4));
              ((float2*)(&__10))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__11))[0]);
              ((float2*)(&__10))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__11))[1]);
              *(float2*)(&(__9.x)) = tl::mul2(*(float2*)(&(v__10.x)), *(float2*)(&(__10.x)));
              *(float2*)(&(__9.z)) = tl::mul2(*(float2*)(&(v__10.z)), *(float2*)(&(__10.z)));
            float4 v__12 = *(float4*)(g_exp_shared_local_cast_4 + (vec * 4));
            *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(__9.x)), *(float2*)(&(v__12.x)));
            *(float2*)(&(__8.z)) = tl::mul2(*(float2*)(&(__9.z)), *(float2*)(&(v__12.z)));
          (reinterpret_cast<__nv_bfloat162*>(&__7))[0] = __float22bfloat162_rn(((float2*)(&__8))[0]);
          (reinterpret_cast<__nv_bfloat162*>(&__7))[1] = __float22bfloat162_rn(((float2*)(&__8))[1]);
          *(uint2*)(tmp_shared_2_3_local_cast_2 + (vec * 4)) = __7;
        }
        *(uint4*)(((bfloat16_t*)tmp_shared_2_3) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_17 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(tmp_shared_2_3_local_cast_2 + 0);
      }
      bar_14[0].arrive();
      bar_14[0].wait((i_s & 1));
      {
        bfloat16_t A_local_2[16];
        bfloat16_t B_local_2[32];
        tl::__sync_thread_partial(3, 128);
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          #pragma unroll
          for (int i_18 = 0; i_18 < 2; ++i_18) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_1)[((((((ki_2 * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_18) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_2[(i_18 * 8)])));
          }
          #pragma unroll
          for (int i_19 = 0; i_19 < 4; ++i_19) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_3)[(((((((((int)threadIdx.x) >> 6) * 4096) + (ki_2 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_19 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[(i_19 * 8)])));
          }
          for (int i_20 = 0; i_20 < 2; ++i_20) {
            for (int j_2 = 0; j_2 < 4; ++j_2) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_0 + ((i_20 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_20 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_0 + (((i_20 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_20 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
            }
          }
        }
      }
      {
        bfloat16_t A_local_3[16];
        bfloat16_t B_local_3[32];
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          #pragma unroll
          for (int i_21 = 0; i_21 < 2; ++i_21) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_1)[(((((((ki_3 * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_21) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(A_local_3[(i_21 * 8)])));
          }
          #pragma unroll
          for (int i_22 = 0; i_22 < 4; ++i_22) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_3)[(((((((((int)threadIdx.x) >> 6) * 4096) + (ki_3 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_22 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_22 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[(i_22 * 8)])));
          }
          for (int i_23 = 0; i_23 < 2; ++i_23) {
            for (int j_3 = 0; j_3 < 4; ++j_3) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_1 + ((i_23 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_23 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dh_fragment_1 + (((i_23 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_23 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
            }
          }
        }
      }
      bar_15[0].arrive();
      bar_15[0].wait((i_s & 1));
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_24 = 0; i_24 < 8; ++i_24) {
        tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_4_1)[(((((((((int)threadIdx.x) >> 5) * 2048) + ((i_24 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_24 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_24 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)dh_fragment_0[(i_24 * 8)]), ((bfloat16_t)dh_fragment_0[((i_24 * 8) + 1)])), __pack_half2(((bfloat16_t)dh_fragment_0[((i_24 * 8) + 2)]), ((bfloat16_t)dh_fragment_0[((i_24 * 8) + 3)])), __pack_half2(((bfloat16_t)dh_fragment_0[((i_24 * 8) + 4)]), ((bfloat16_t)dh_fragment_0[((i_24 * 8) + 5)])), __pack_half2(((bfloat16_t)dh_fragment_0[((i_24 * 8) + 6)]), ((bfloat16_t)dh_fragment_0[((i_24 * 8) + 7)])));
      }
      tl::fence_proxy_async();
      bar_dhtmp_ready[0].arrive();
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<168>();
      for (int i_s_1 = 0; i_s_1 < num_iters; ++i_s_1) {
        bar_00[0].arrive();
        bar_00[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial(4, 128);
        if (0 < i_s_1) {
          #pragma unroll
          for (int i_25 = 0; i_25 < 16; ++i_25) {
            bfloat16_t dqkv_shared_local_cast_5[2];
            uint1 __11;
            float2 v__13 = *(float2*)(dk_frag_l + (i_25 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__11))[0] = __float22bfloat162_rn(((float2*)(&v__13))[0]);
            *(uint1*)(dqkv_shared_local_cast_5 + 0) = __11;
            *(uint1*)(((bfloat16_t*)dqkv_shared) + ((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_25 >> 3) * 1024)) + ((i_25 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_25 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_25 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(dqkv_shared_local_cast_5 + 0);
          }
          #pragma unroll
          for (int i_26 = 0; i_26 < 16; ++i_26) {
            bfloat16_t dqkv_shared_local_cast_6[2];
            uint1 __12;
            float2 v__14 = *(float2*)(dk_frag_r + (i_26 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__12))[0] = __float22bfloat162_rn(((float2*)(&v__14))[0]);
            *(uint1*)(dqkv_shared_local_cast_6 + 0) = __12;
            *(uint1*)(((bfloat16_t*)dqkv_shared) + (((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_26 >> 3) * 1024)) + ((i_26 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_26 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_26 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096)) = *(uint1*)(dqkv_shared_local_cast_6 + 0);
          }
        }
        bar_01[0].arrive();
        bar_01[0].wait((i_s_1 & 1));
        bar_dhtmp_ready[0].wait((i_s_1 & 1));
        if (i_s_1 == 0) {
          #pragma unroll
          for (int i_27 = 0; i_27 < 8; ++i_27) {
            bfloat16_t broadcast_var_15 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((bfloat16_t*)tmp_shared_4_1) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_27 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_15, broadcast_var_15), __pack_nv_bfloat162(broadcast_var_15, broadcast_var_15), __pack_nv_bfloat162(broadcast_var_15, broadcast_var_15), __pack_nv_bfloat162(broadcast_var_15, broadcast_var_15));
          }
          tl::fence_proxy_async();
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_28 = 0; i_28 < 8; ++i_28) {
          *(uint4*)(dh_left_fragment + (i_28 * 8)) = *(uint4*)(((bfloat16_t*)tmp_shared_4_1) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_28 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
        }
        {
          bfloat16_t A_local_4[16];
          bfloat16_t B_local_4[32];
          #pragma unroll
          for (int i_29 = 0; i_29 < 16; ++i_29) {
            float broadcast_var_16 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(dv_fragment + (i_29 * 4)) = make_float4(broadcast_var_16, broadcast_var_16, broadcast_var_16, broadcast_var_16);
          }
          for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
            #pragma unroll
            for (int i_30 = 0; i_30 < 2; ++i_30) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_30 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_4[(i_30 * 8)])));
            }
            #pragma unroll
            for (int i_31 = 0; i_31 < 4; ++i_31) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_4_1)[((((((((((int)threadIdx.x) & 127) >> 6) * 4096) + (ki_4 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_31 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_31 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_4[(i_31 * 8)])));
            }
            for (int i_32 = 0; i_32 < 2; ++i_32) {
              for (int j_4 = 0; j_4 < 4; ++j_4) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + ((i_32 * 32) + (j_4 * 8))), reinterpret_cast<const unsigned*>(A_local_4 + (i_32 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + (j_4 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + (((i_32 * 32) + (j_4 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + (i_32 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + ((j_4 * 8) + 4)));
              }
            }
          }
        }
        if (0 < i_s_1) {
          bar_k_left_ready[0].arrive();
        }
        tl::__sync_thread_partial(4, 128);
        if (i_s_1 == 0) {
          #pragma unroll
          for (int i_33 = 0; i_33 < 8; ++i_33) {
            bfloat16_t broadcast_var_17 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((bfloat16_t*)tmp_shared_4_1) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_33 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_17, broadcast_var_17), __pack_nv_bfloat162(broadcast_var_17, broadcast_var_17), __pack_nv_bfloat162(broadcast_var_17, broadcast_var_17), __pack_nv_bfloat162(broadcast_var_17, broadcast_var_17));
          }
        } else {
          bar_dh_right_ready[0].wait(((i_s_1 + 1) & 1));
        }
        tl::fence_proxy_async();
        {
          bfloat16_t A_local_5[16];
          bfloat16_t B_local_5[32];
          tl::__sync_thread_partial(4, 128);
          for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
            #pragma unroll
            for (int i_34 = 0; i_34 < 2; ++i_34) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[(((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_34 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(A_local_5[(i_34 * 8)])));
            }
            #pragma unroll
            for (int i_35 = 0; i_35 < 4; ++i_35) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_4_1)[((((((((((int)threadIdx.x) & 127) >> 6) * 4096) + (ki_5 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_35 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_35 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_5[(i_35 * 8)])));
            }
            for (int i_36 = 0; i_36 < 2; ++i_36) {
              for (int j_5 = 0; j_5 < 4; ++j_5) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + ((i_36 * 32) + (j_5 * 8))), reinterpret_cast<const unsigned*>(A_local_5 + (i_36 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + (j_5 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + (((i_36 * 32) + (j_5 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + (i_36 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + ((j_5 * 8) + 4)));
              }
            }
          }
        }
        #pragma unroll
        for (int i_37 = 0; i_37 < 32; ++i_37) {
          float2 __13;
            float2 v__15 = *(float2*)(dv_fragment + (i_37 * 2));
            float2 v__16 = make_float2(g_rev_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_37 >> 4) * 16)) + ((i_37 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_rev_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_37 >> 4) * 16)) + ((i_37 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            *(float2*)(&(__13.x)) = tl::mul2(*(float2*)(&(v__15.x)), *(float2*)(&(v__16.x)));
          *(float2*)(dv_fragment + (i_37 * 2)) = __13;
        }
        bar_02[0].arrive();
        bar_02[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_6[16];
          bfloat16_t B_local_6[32];
          for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
            #pragma unroll
            for (int i_38 = 0; i_38 < 2; ++i_38) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_1_1)[((((((ki_6 * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_38) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_6[(i_38 * 8)])));
            }
            #pragma unroll
            for (int i_39 = 0; i_39 < 4; ++i_39) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)do_shared)[((((((((((int)threadIdx.x) & 127) >> 6) * 4096) + (ki_6 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_39 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_39 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_6[(i_39 * 8)])));
            }
            for (int i_40 = 0; i_40 < 2; ++i_40) {
              for (int j_6 = 0; j_6 < 4; ++j_6) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + ((i_40 * 32) + (j_6 * 8))), reinterpret_cast<const unsigned*>(A_local_6 + (i_40 * 8)), reinterpret_cast<const unsigned*>(B_local_6 + (j_6 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + (((i_40 * 32) + (j_6 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_6 + (i_40 * 8)), reinterpret_cast<const unsigned*>(B_local_6 + ((j_6 * 8) + 4)));
              }
            }
          }
        }
        bar_03[0].arrive();
        bar_03[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_41 = 0; i_41 < 8; ++i_41) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_2_1)[((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_41 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_41 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_41 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 8192)])), __pack_half2(((bfloat16_t)dv_fragment[(i_41 * 8)]), ((bfloat16_t)dv_fragment[((i_41 * 8) + 1)])), __pack_half2(((bfloat16_t)dv_fragment[((i_41 * 8) + 2)]), ((bfloat16_t)dv_fragment[((i_41 * 8) + 3)])), __pack_half2(((bfloat16_t)dv_fragment[((i_41 * 8) + 4)]), ((bfloat16_t)dv_fragment[((i_41 * 8) + 5)])), __pack_half2(((bfloat16_t)dv_fragment[((i_41 * 8) + 6)]), ((bfloat16_t)dv_fragment[((i_41 * 8) + 7)])));
        }
        bar_04[0].arrive();
        bar_04[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_7[16];
          bfloat16_t B_local_7[32];
          #pragma unroll
          for (int i_42 = 0; i_42 < 16; ++i_42) {
            float broadcast_var_18 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(dv_fragment + (i_42 * 4)) = make_float4(broadcast_var_18, broadcast_var_18, broadcast_var_18, broadcast_var_18);
          }
          tl::__sync_thread_partial(4, 128);
          for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
            #pragma unroll
            for (int i_43 = 0; i_43 < 2; ++i_43) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_1_2)[((((((ki_7 * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_43) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_7[(i_43 * 8)])));
            }
            #pragma unroll
            for (int i_44 = 0; i_44 < 4; ++i_44) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_1)[((((((((((int)threadIdx.x) & 127) >> 6) * 4096) + (ki_7 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_44 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_44 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_7[(i_44 * 8)])));
            }
            for (int i_45 = 0; i_45 < 2; ++i_45) {
              for (int j_7 = 0; j_7 < 4; ++j_7) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + ((i_45 * 32) + (j_7 * 8))), reinterpret_cast<const unsigned*>(A_local_7 + (i_45 * 8)), reinterpret_cast<const unsigned*>(B_local_7 + (j_7 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv_fragment + (((i_45 * 32) + (j_7 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_7 + (i_45 * 8)), reinterpret_cast<const unsigned*>(B_local_7 + ((j_7 * 8) + 4)));
              }
            }
          }
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_46 = 0; i_46 < 8; ++i_46) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)dqkv_shared)[((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_46 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_46 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_46 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 8192)])), __pack_half2(((bfloat16_t)dv_fragment[(i_46 * 8)]), ((bfloat16_t)dv_fragment[((i_46 * 8) + 1)])), __pack_half2(((bfloat16_t)dv_fragment[((i_46 * 8) + 2)]), ((bfloat16_t)dv_fragment[((i_46 * 8) + 3)])), __pack_half2(((bfloat16_t)dv_fragment[((i_46 * 8) + 4)]), ((bfloat16_t)dv_fragment[((i_46 * 8) + 5)])), __pack_half2(((bfloat16_t)dv_fragment[((i_46 * 8) + 6)]), ((bfloat16_t)dv_fragment[((i_46 * 8) + 7)])));
        }
        bar_05[0].arrive();
        bar_05[0].wait((i_s_1 & 1));
        #pragma unroll
        for (int i_47 = 0; i_47 < 32; ++i_47) {
          float broadcast_var_19 = -0x1p+0f/*-1.000000e+00*/;
          float2 __14;
            float2 __15;
              float2 v__17 = *(float2*)(dv_fragment + (i_47 * 2));
              float2 v__18 = make_float2(broadcast_var_19, broadcast_var_19);
              *(float2*)(&(__15.x)) = tl::mul2(*(float2*)(&(v__17.x)), *(float2*)(&(v__18.x)));
            float2 v__19 = make_float2(g_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_47 >> 4) * 16)) + ((i_47 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_47 >> 4) * 16)) + ((i_47 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            *(float2*)(&(__14.x)) = tl::mul2(*(float2*)(&(__15.x)), *(float2*)(&(v__19.x)));
          *(float2*)(dv_fragment + (i_47 * 2)) = __14;
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_48 = 0; i_48 < 32; ++i_48) {
          bfloat16_t tmp_shared_2_3_local_cast_7[2];
          *(uint1*)(tmp_shared_2_3_local_cast_7 + 0) = *(uint1*)(((bfloat16_t*)tmp_shared_2_3) + ((((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_48 >> 4) * 1024)) + ((i_48 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_48 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_48 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_48 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
          float2 __16;
          uint1 v__20 = *(uint1*)(tmp_shared_2_3_local_cast_7 + 0);
          ((float2*)(&__16))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__20))[0]);
          *(float2*)(odot_fragment_1 + (i_48 * 2)) = __16;
        }
        #pragma unroll
        for (int i_49 = 0; i_49 < 16; ++i_49) {
          float4 __17;
            float4 v__21 = *(float4*)(odot_fragment_1 + (i_49 * 4));
            float4 v__22 = *(float4*)(dv_fragment + (i_49 * 4));
            *(float2*)(&(__17.x)) = tl::mul2(*(float2*)(&(v__21.x)), *(float2*)(&(v__22.x)));
            *(float2*)(&(__17.z)) = tl::mul2(*(float2*)(&(v__21.z)), *(float2*)(&(v__22.z)));
          *(float4*)(odot_fragment_1 + (i_49 * 4)) = __17;
        }
        #pragma unroll
        for (int i_50 = 0; i_50 < 4; ++i_50) {
          dg_fragment_1[i_50] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv = 0; rv < 16; ++rv) {
            dg_fragment_1[i_50] = (dg_fragment_1[i_50] + odot_fragment_1[(((((i_50 >> 1) * 32) + ((rv & 7) * 4)) + ((i_50 & 1) * 2)) + (rv >> 3))]);
          }
          tl::__sync_thread_partial(4, 128);
          dg_fragment_1[i_50] = tl::AllReduce<tl::SumOp, 128, 64, 128, tl::NamedBarrier<128>>::run(dg_fragment_1[i_50], (&(((float*)workspace_2)[0])));
          dg_fragment_1[i_50] = tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<128>>::run(dg_fragment_1[i_50]);
        }
        tl::__sync_thread_partial(4, 128);
        if ((((((int)threadIdx.x) & 3) * 2) + (((int)threadIdx.x) >> 6)) == 2) {
          #pragma unroll
          for (int i_51 = 0; i_51 < 4; ++i_51) {
            dg_shared[(((((((int)threadIdx.x) & 63) >> 5) * 32) + (i_51 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = dg_fragment_1[i_51];
          }
        }
        bar_dg_k_init[0].arrive();
        bar_dg_k_init[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_52 = 0; i_52 < 8; ++i_52) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_2_3)[((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_52 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_52 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_52 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 8192)])), __pack_half2(((bfloat16_t)dv_fragment[(i_52 * 8)]), ((bfloat16_t)dv_fragment[((i_52 * 8) + 1)])), __pack_half2(((bfloat16_t)dv_fragment[((i_52 * 8) + 2)]), ((bfloat16_t)dv_fragment[((i_52 * 8) + 3)])), __pack_half2(((bfloat16_t)dv_fragment[((i_52 * 8) + 4)]), ((bfloat16_t)dv_fragment[((i_52 * 8) + 5)])), __pack_half2(((bfloat16_t)dv_fragment[((i_52 * 8) + 6)]), ((bfloat16_t)dv_fragment[((i_52 * 8) + 7)])));
        }
        bar_06[0].arrive();
        bar_06[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_53 = 0; i_53 < 32; ++i_53) {
          bfloat16_t k_shared_local_cast_8[2];
          *(uint1*)(k_shared_local_cast_8 + 0) = *(uint1*)(((bfloat16_t*)k_shared) + ((((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_53 >> 4) * 1024)) + ((i_53 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_53 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_53 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_53 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
          float2 __18;
          uint1 v__23 = *(uint1*)(k_shared_local_cast_8 + 0);
          ((float2*)(&__18))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__23))[0]);
          *(float2*)(odot_fragment_1 + (i_53 * 2)) = __18;
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_54 = 0; i_54 < 8; ++i_54) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_2_2)[((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_54 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_54 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_54 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 8192)])), __pack_half2(((bfloat16_t)odot_fragment_1[(i_54 * 8)]), ((bfloat16_t)odot_fragment_1[((i_54 * 8) + 1)])), __pack_half2(((bfloat16_t)odot_fragment_1[((i_54 * 8) + 2)]), ((bfloat16_t)odot_fragment_1[((i_54 * 8) + 3)])), __pack_half2(((bfloat16_t)odot_fragment_1[((i_54 * 8) + 4)]), ((bfloat16_t)odot_fragment_1[((i_54 * 8) + 5)])), __pack_half2(((bfloat16_t)odot_fragment_1[((i_54 * 8) + 6)]), ((bfloat16_t)odot_fragment_1[((i_54 * 8) + 7)])));
        }
        bar_07[0].arrive();
        bar_07[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_8[16];
          bfloat16_t B_local_8[16];
          #pragma unroll
          for (int i_55 = 0; i_55 < 8; ++i_55) {
            float broadcast_var_20 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(dk_frag_r + (i_55 * 4)) = make_float4(broadcast_var_20, broadcast_var_20, broadcast_var_20, broadcast_var_20);
          }
          for (int ki_8 = 0; ki_8 < 8; ++ki_8) {
            #pragma unroll
            for (int i_56 = 0; i_56 < 2; ++i_56) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_1)[((((((((ki_8 >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_56 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_8[(i_56 * 8)])));
            }
            #pragma unroll
            for (int i_57 = 0; i_57 < 2; ++i_57) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_4_1)[(((((((((ki_8 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_57 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_8[(i_57 * 8)])));
            }
            for (int i_58 = 0; i_58 < 2; ++i_58) {
              for (int j_8 = 0; j_8 < 2; ++j_8) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + ((i_58 * 16) + (j_8 * 8))), reinterpret_cast<const unsigned*>(A_local_8 + (i_58 * 8)), reinterpret_cast<const unsigned*>(B_local_8 + (j_8 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + (((i_58 * 16) + (j_8 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_8 + (i_58 * 8)), reinterpret_cast<const unsigned*>(B_local_8 + ((j_8 * 8) + 4)));
              }
            }
          }
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_59 = 0; i_59 < 8; ++i_59) {
          *(uint4*)(((bfloat16_t*)tmp_shared_4_1) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_59 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(dh_left_fragment + (i_59 * 8));
        }
        tl::fence_proxy_async();
        {
          bfloat16_t A_local_9[16];
          bfloat16_t B_local_9[16];
          #pragma unroll
          for (int i_60 = 0; i_60 < 8; ++i_60) {
            float broadcast_var_21 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(dk_frag_l + (i_60 * 4)) = make_float4(broadcast_var_21, broadcast_var_21, broadcast_var_21, broadcast_var_21);
          }
          tl::__sync_thread_partial(4, 128);
          for (int ki_9 = 0; ki_9 < 8; ++ki_9) {
            #pragma unroll
            for (int i_61 = 0; i_61 < 2; ++i_61) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_1)[((((((((ki_9 >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_61 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_9 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_9[(i_61 * 8)])));
            }
            #pragma unroll
            for (int i_62 = 0; i_62 < 2; ++i_62) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_4_1)[(((((((((ki_9 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_62 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_9 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_9[(i_62 * 8)])));
            }
            for (int i_63 = 0; i_63 < 2; ++i_63) {
              for (int j_9 = 0; j_9 < 2; ++j_9) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + ((i_63 * 16) + (j_9 * 8))), reinterpret_cast<const unsigned*>(A_local_9 + (i_63 * 8)), reinterpret_cast<const unsigned*>(B_local_9 + (j_9 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + (((i_63 * 16) + (j_9 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_9 + (i_63 * 8)), reinterpret_cast<const unsigned*>(B_local_9 + ((j_9 * 8) + 4)));
              }
            }
          }
        }
        #pragma unroll
        for (int i_64 = 0; i_64 < 16; ++i_64) {
          float2 __19;
            float2 v__24 = *(float2*)(dk_frag_r + (i_64 * 2));
            float2 v__25 = make_float2(g_rev_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_64 >> 3) * 16)) + ((i_64 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_rev_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_64 >> 3) * 16)) + ((i_64 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            *(float2*)(&(__19.x)) = tl::mul2(*(float2*)(&(v__24.x)), *(float2*)(&(v__25.x)));
          *(float2*)(dk_frag_r + (i_64 * 2)) = __19;
        }
        #pragma unroll
        for (int i_65 = 0; i_65 < 16; ++i_65) {
          float2 __20;
            float2 v__26 = *(float2*)(dk_frag_l + (i_65 * 2));
            float2 v__27 = make_float2(g_rev_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_65 >> 3) * 16)) + ((i_65 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_rev_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_65 >> 3) * 16)) + ((i_65 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            *(float2*)(&(__20.x)) = tl::mul2(*(float2*)(&(v__26.x)), *(float2*)(&(v__27.x)));
          *(float2*)(dk_frag_l + (i_65 * 2)) = __20;
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_66 = 0; i_66 < 16; ++i_66) {
          bfloat16_t dqkv_shared_local_cast_9[2];
          uint1 __21;
          float2 v__28 = *(float2*)(dk_frag_l + (i_66 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__21))[0] = __float22bfloat162_rn(((float2*)(&v__28))[0]);
          *(uint1*)(dqkv_shared_local_cast_9 + 0) = __21;
          *(uint1*)(((bfloat16_t*)dqkv_shared) + ((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_66 >> 3) * 1024)) + ((i_66 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_66 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_66 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(dqkv_shared_local_cast_9 + 0);
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_67 = 0; i_67 < 16; ++i_67) {
          bfloat16_t dqkv_shared_local_cast_10[2];
          uint1 __22;
          float2 v__29 = *(float2*)(dk_frag_r + (i_67 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__22))[0] = __float22bfloat162_rn(((float2*)(&v__29))[0]);
          *(uint1*)(dqkv_shared_local_cast_10 + 0) = __22;
          *(uint1*)(((bfloat16_t*)dqkv_shared) + (((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_67 >> 3) * 1024)) + ((i_67 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_67 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_67 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096)) = *(uint1*)(dqkv_shared_local_cast_10 + 0);
        }
        tl::fence_proxy_async();
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_68 = 0; i_68 < 32; ++i_68) {
          bfloat16_t dqkv_shared_local_cast_11[2];
          *(uint1*)(dqkv_shared_local_cast_11 + 0) = *(uint1*)(((bfloat16_t*)dqkv_shared) + ((((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_68 >> 4) * 1024)) + ((i_68 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_68 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_68 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_68 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
          bfloat16_t broadcast_var_22 = bfloat16_t(-0x1p+0f/*-1.000000e+00*/);
          float2 __23;
            float2 v__30 = *(float2*)(odot_fragment_1 + (i_68 * 2));
            float2 __24;
            uint1 __25;
              uint1 v__31 = *(uint1*)(dqkv_shared_local_cast_11 + 0);
              uint1 v__32 = make_uint1(__pack_nv_bfloat162(broadcast_var_22, broadcast_var_22));
              *(uint1*)(&(__25.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__31.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__32.x)))));
            ((float2*)(&__24))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&__25))[0]);
            *(float2*)(&(__23.x)) = tl::mul2(*(float2*)(&(v__30.x)), *(float2*)(&(__24.x)));
          *(float2*)(odot_fragment_1 + (i_68 * 2)) = __23;
        }
        #pragma unroll
        for (int i_69 = 0; i_69 < 4; ++i_69) {
          dg_fragment_1[i_69] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
            dg_fragment_1[i_69] = (dg_fragment_1[i_69] + odot_fragment_1[(((((i_69 >> 1) * 32) + ((rv_1 & 7) * 4)) + ((i_69 & 1) * 2)) + (rv_1 >> 3))]);
          }
          tl::__sync_thread_partial(4, 128);
          dg_fragment_1[i_69] = tl::AllReduce<tl::SumOp, 128, 64, 128, tl::NamedBarrier<128>>::run(dg_fragment_1[i_69], (&(((float*)workspace_4)[0])));
          dg_fragment_1[i_69] = tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<128>>::run(dg_fragment_1[i_69]);
        }
        tl::__sync_thread_partial(4, 128);
        if ((((((int)threadIdx.x) & 3) * 2) + (((int)threadIdx.x) >> 6)) == 2) {
          #pragma unroll
          for (int i_70 = 0; i_70 < 4; ++i_70) {
            dg_shared[(((((((int)threadIdx.x) & 63) >> 5) * 32) + (i_70 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = (dg_shared[(((((((int)threadIdx.x) & 63) >> 5) * 32) + (i_70 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] + dg_fragment_1[i_70]);
          }
        }
        float broadcast_var_23 = 0x0p+0f/*0.000000e+00*/;
        dg_last_local_1_pack[0] = make_float2(broadcast_var_23, broadcast_var_23);
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 2; ++rv_2) {
          dg_last_local_1_pack[0] = tl::add2(dg_last_local_1_pack[0], *(float2*)(dg_fragment_1 + (rv_2 * 2)));
        }
        dg_last_local_1[0] = ((dg_last_local_1_pack[0]).x + (dg_last_local_1_pack[0]).y);
        dg_last_local_1[0] = tl::AllReduce<tl::SumOp, 64, 32, 128, tl::NamedBarrier<128>>::run(dg_last_local_1[0], (&(((float*)workspace_3)[0])));
        dg_last_local_1[0] = tl::AllReduce<tl::SumOp, 32, 4, 128, tl::NamedBarrier<128>>::run(dg_last_local_1[0]);
        bar_dg_k_rows[0].arrive();
        bar_dg_k_rows[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial(4, 128);
        if (((int)threadIdx.x) == 128) {
          dg_shared[63] = (dg_shared[63] - dg_last_local_1[0]);
        }
        bar_08_1[0].arrive();
        bar_08_1[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_10[16];
          bfloat16_t B_local_10[16];
          for (int ki_10 = 0; ki_10 < 8; ++ki_10) {
            #pragma unroll
            for (int i_71 = 0; i_71 < 2; ++i_71) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_3)[((((((((ki_10 >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_71 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_10 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_10[(i_71 * 8)])));
            }
            #pragma unroll
            for (int i_72 = 0; i_72 < 2; ++i_72) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)h_shared)[(((((((((ki_10 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_72 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_10 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_10[(i_72 * 8)])));
            }
            for (int i_73 = 0; i_73 < 2; ++i_73) {
              for (int j_10 = 0; j_10 < 2; ++j_10) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + ((i_73 * 16) + (j_10 * 8))), reinterpret_cast<const unsigned*>(A_local_10 + (i_73 * 8)), reinterpret_cast<const unsigned*>(B_local_10 + (j_10 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + (((i_73 * 16) + (j_10 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_10 + (i_73 * 8)), reinterpret_cast<const unsigned*>(B_local_10 + ((j_10 * 8) + 4)));
              }
            }
          }
        }
        bar_08_2[0].arrive();
        bar_08_3[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_11[16];
          bfloat16_t B_local_11[16];
          for (int ki_11 = 0; ki_11 < 8; ++ki_11) {
            #pragma unroll
            for (int i_74 = 0; i_74 < 2; ++i_74) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_3)[((((((((ki_11 >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_74 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_11 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_11[(i_74 * 8)])));
            }
            #pragma unroll
            for (int i_75 = 0; i_75 < 2; ++i_75) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)h_shared)[(((((((((ki_11 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_75 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_11 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_11[(i_75 * 8)])));
            }
            for (int i_76 = 0; i_76 < 2; ++i_76) {
              for (int j_11 = 0; j_11 < 2; ++j_11) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + ((i_76 * 16) + (j_11 * 8))), reinterpret_cast<const unsigned*>(A_local_11 + (i_76 * 8)), reinterpret_cast<const unsigned*>(B_local_11 + (j_11 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + (((i_76 * 16) + (j_11 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_11 + (i_76 * 8)), reinterpret_cast<const unsigned*>(B_local_11 + ((j_11 * 8) + 4)));
              }
            }
          }
        }
        bar_10[0].arrive();
        bar_10[0].wait((i_s_1 & 1));
        bar_12[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_12[16];
          bfloat16_t B_local_12[16];
          for (int ki_12 = 0; ki_12 < 4; ++ki_12) {
            #pragma unroll
            for (int i_77 = 0; i_77 < 2; ++i_77) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_1_1)[((((((ki_12 * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_77) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_12[(i_77 * 8)])));
            }
            #pragma unroll
            for (int i_78 = 0; i_78 < 2; ++i_78) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_1)[(((((ki_12 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_78) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_12[(i_78 * 8)])));
            }
            for (int i_79 = 0; i_79 < 2; ++i_79) {
              for (int j_12 = 0; j_12 < 2; ++j_12) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + ((i_79 * 16) + (j_12 * 8))), reinterpret_cast<const unsigned*>(A_local_12 + (i_79 * 8)), reinterpret_cast<const unsigned*>(B_local_12 + (j_12 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + (((i_79 * 16) + (j_12 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_12 + (i_79 * 8)), reinterpret_cast<const unsigned*>(B_local_12 + ((j_12 * 8) + 4)));
              }
            }
          }
        }
        {
          bfloat16_t A_local_13[16];
          bfloat16_t B_local_13[16];
          for (int ki_13 = 0; ki_13 < 4; ++ki_13) {
            #pragma unroll
            for (int i_80 = 0; i_80 < 2; ++i_80) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_1_1)[((((((ki_13 * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_80) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_13[(i_80 * 8)])));
            }
            #pragma unroll
            for (int i_81 = 0; i_81 < 2; ++i_81) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_1)[((((((ki_13 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_81) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(B_local_13[(i_81 * 8)])));
            }
            for (int i_82 = 0; i_82 < 2; ++i_82) {
              for (int j_13 = 0; j_13 < 2; ++j_13) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + ((i_82 * 16) + (j_13 * 8))), reinterpret_cast<const unsigned*>(A_local_13 + (i_82 * 8)), reinterpret_cast<const unsigned*>(B_local_13 + (j_13 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + (((i_82 * 16) + (j_13 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_13 + (i_82 * 8)), reinterpret_cast<const unsigned*>(B_local_13 + ((j_13 * 8) + 4)));
              }
            }
          }
        }
        bar_13[0].arrive();
        bar_13[0].wait((i_s_1 & 1));
        bar_15[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_14[16];
          bfloat16_t B_local_14[16];
          for (int ki_14 = 0; ki_14 < 4; ++ki_14) {
            #pragma unroll
            for (int i_83 = 0; i_83 < 2; ++i_83) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_83 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_14 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_14 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_14[(i_83 * 8)])));
            }
            #pragma unroll
            for (int i_84 = 0; i_84 < 2; ++i_84) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_2)[(((((ki_14 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_84) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_14[(i_84 * 8)])));
            }
            for (int i_85 = 0; i_85 < 2; ++i_85) {
              for (int j_14 = 0; j_14 < 2; ++j_14) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + ((i_85 * 16) + (j_14 * 8))), reinterpret_cast<const unsigned*>(A_local_14 + (i_85 * 8)), reinterpret_cast<const unsigned*>(B_local_14 + (j_14 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_l + (((i_85 * 16) + (j_14 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_14 + (i_85 * 8)), reinterpret_cast<const unsigned*>(B_local_14 + ((j_14 * 8) + 4)));
              }
            }
          }
        }
        {
          bfloat16_t A_local_15[16];
          bfloat16_t B_local_15[16];
          for (int ki_15 = 0; ki_15 < 4; ++ki_15) {
            #pragma unroll
            for (int i_86 = 0; i_86 < 2; ++i_86) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_86 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_15 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_15 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_15[(i_86 * 8)])));
            }
            #pragma unroll
            for (int i_87 = 0; i_87 < 2; ++i_87) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_2)[((((((ki_15 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_87) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(B_local_15[(i_87 * 8)])));
            }
            for (int i_88 = 0; i_88 < 2; ++i_88) {
              for (int j_15 = 0; j_15 < 2; ++j_15) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + ((i_88 * 16) + (j_15 * 8))), reinterpret_cast<const unsigned*>(A_local_15 + (i_88 * 8)), reinterpret_cast<const unsigned*>(B_local_15 + (j_15 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk_frag_r + (((i_88 * 16) + (j_15 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_15 + (i_88 * 8)), reinterpret_cast<const unsigned*>(B_local_15 + ((j_15 * 8) + 4)));
              }
            }
          }
        }
      }
      #pragma unroll
      for (int i_89 = 0; i_89 < 16; ++i_89) {
        bfloat16_t dk_local_cast_12[2];
        if ((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_89 >> 3) * 16)) + ((i_89 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) < seq_end_idx) {
          uint1 __26;
          float2 v__33 = *(float2*)(dk_frag_l + (i_89 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__26))[0] = __float22bfloat162_rn(((float2*)(&v__33))[0]);
          *(uint1*)(dk_local_cast_12 + 0) = __26;
        }
        if ((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_89 >> 3) * 16)) + ((i_89 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) < seq_end_idx) {
          if (0 <= (((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_89 >> 3) * 16)) + ((i_89 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx)) {
            if ((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_89 >> 3) * 16)) + ((i_89 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) < num_tokens) {
              if (0 <= batch_idx) {
                if (batch_idx < batch_size) {
                  *(uint1*)(dk + (((((((((((((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)65536) + ((((int64_t)i_89) >> (int64_t)3) * (int64_t)32768)) + ((((int64_t)i_89) & (int64_t)1) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)2048)) + (((int64_t)seq_start_idx) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_89) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (int64_t)64)) = *(uint1*)(dk_local_cast_12 + 0);
                }
              }
            }
          }
        }
      }
      #pragma unroll
      for (int i_90 = 0; i_90 < 16; ++i_90) {
        bfloat16_t dk_local_cast_13[2];
        if ((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_90 >> 3) * 16)) + ((i_90 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) < seq_end_idx) {
          uint1 __27;
          float2 v__34 = *(float2*)(dk_frag_r + (i_90 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__27))[0] = __float22bfloat162_rn(((float2*)(&v__34))[0]);
          *(uint1*)(dk_local_cast_13 + 0) = __27;
        }
        if ((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_90 >> 3) * 16)) + ((i_90 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) < seq_end_idx) {
          if (0 <= (((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_90 >> 3) * 16)) + ((i_90 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx)) {
            if ((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_90 >> 3) * 16)) + ((i_90 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) < num_tokens) {
              if (0 <= batch_idx) {
                if (batch_idx < batch_size) {
                  *(uint1*)(dk + ((((((((((((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)65536) + ((((int64_t)i_90) >> (int64_t)3) * (int64_t)32768)) + ((((int64_t)i_90) & (int64_t)1) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)2048)) + (((int64_t)seq_start_idx) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_90) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(uint1*)(dk_local_cast_13 + 0);
                }
              }
            }
          }
        }
      }
    } else {
      if (((int)threadIdx.x) < 384) {
        tl::warpgroup_reg_alloc<160>();
        for (int i_s_2 = 0; i_s_2 < num_iters; ++i_s_2) {
          int cur_idx_1 = (((chunk_start_idx + num_iters) - i_s_2) - 1);
          bar_00[0].arrive();
          bar_00[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_16[16];
            bfloat16_t B_local_16[16];
            #pragma unroll
            for (int i_91 = 0; i_91 < 8; ++i_91) {
              float broadcast_var_24 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(p_fragment + (i_91 * 4)) = make_float4(broadcast_var_24, broadcast_var_24, broadcast_var_24, broadcast_var_24);
            }
            for (int ki_16 = 0; ki_16 < 8; ++ki_16) {
              #pragma unroll
              for (int i_92 = 0; i_92 < 2; ++i_92) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_1)[((((((((ki_16 >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_92 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_16 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_16[(i_92 * 8)])));
              }
              #pragma unroll
              for (int i_93 = 0; i_93 < 2; ++i_93) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[(((((((((ki_16 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_93 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_16 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_16[(i_93 * 8)])));
              }
              for (int i_94 = 0; i_94 < 2; ++i_94) {
                for (int j_16 = 0; j_16 < 2; ++j_16) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + ((i_94 * 16) + (j_16 * 8))), reinterpret_cast<const unsigned*>(A_local_16 + (i_94 * 8)), reinterpret_cast<const unsigned*>(B_local_16 + (j_16 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + (((i_94 * 16) + (j_16 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_16 + (i_94 * 8)), reinterpret_cast<const unsigned*>(B_local_16 + ((j_16 * 8) + 4)));
                }
              }
            }
          }
          bar_01[0].arrive();
          bar_01[0].wait((i_s_2 & 1));
          #pragma unroll
          for (int i_95 = 0; i_95 < 16; ++i_95) {
            float2 __28;
              float2 v__35 = make_float2(g_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_95 >> 3) * 16)) + ((i_95 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_95 >> 3) * 16)) + ((i_95 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
              float2 v__36 = *(float2*)(g_shared + (((((((int)threadIdx.x) >> 6) * 32) + (((i_95 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 128));
              *(float2*)(&(__28.x)) = tl::sub2(*(float2*)(&(v__35.x)), *(float2*)(&(v__36.x)));
            *(float2*)(mask_fragment + (i_95 * 2)) = __28;
          }
          #pragma unroll
          for (int i_96 = 0; i_96 < 32; ++i_96) {
            if ((((((((int)threadIdx.x) >> 6) * 32) + (((i_96 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_96 & 1)) <= (((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_96 >> 4) * 16)) + (((i_96 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 128)) {
              mask_fragment[i_96] = exp2f((mask_fragment[i_96] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
            } else {
              mask_fragment[i_96] = 0x0p+0f/*0.000000e+00*/;
            }
          }
          #pragma unroll
          for (int i_97 = 0; i_97 < 8; ++i_97) {
            float4 __29;
              float4 v__37 = *(float4*)(p_fragment + (i_97 * 4));
              float4 v__38 = *(float4*)(mask_fragment + (i_97 * 4));
              *(float2*)(&(__29.x)) = tl::mul2(*(float2*)(&(v__37.x)), *(float2*)(&(v__38.x)));
              *(float2*)(&(__29.z)) = tl::mul2(*(float2*)(&(v__37.z)), *(float2*)(&(v__38.z)));
            *(float4*)(p_fragment + (i_97 * 4)) = __29;
          }
          #pragma unroll
          for (int i_98 = 0; i_98 < 8; ++i_98) {
            float broadcast_var_25 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
            float4 __30;
              float4 v__39 = *(float4*)(p_fragment + (i_98 * 4));
              float4 v__40 = make_float4(broadcast_var_25, broadcast_var_25, broadcast_var_25, broadcast_var_25);
              *(float2*)(&(__30.x)) = tl::mul2(*(float2*)(&(v__39.x)), *(float2*)(&(v__40.x)));
              *(float2*)(&(__30.z)) = tl::mul2(*(float2*)(&(v__39.z)), *(float2*)(&(v__40.z)));
            *(float4*)(p_fragment + (i_98 * 4)) = __30;
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_99 = 0; i_99 < 4; ++i_99) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_1)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_99 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_99 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)p_fragment[(i_99 * 8)]), ((bfloat16_t)p_fragment[((i_99 * 8) + 1)])), __pack_half2(((bfloat16_t)p_fragment[((i_99 * 8) + 2)]), ((bfloat16_t)p_fragment[((i_99 * 8) + 3)])), __pack_half2(((bfloat16_t)p_fragment[((i_99 * 8) + 4)]), ((bfloat16_t)p_fragment[((i_99 * 8) + 5)])), __pack_half2(((bfloat16_t)p_fragment[((i_99 * 8) + 6)]), ((bfloat16_t)p_fragment[((i_99 * 8) + 7)])));
          }
          bar_02[0].arrive();
          bar_02[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_100 = 0; i_100 < 16; ++i_100) {
            bfloat16_t a_shared_local_cast_14[2];
            *(uint1*)(a_shared_local_cast_14 + 0) = *(uint1*)(((bfloat16_t*)a_shared) + ((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_100 >> 3) * 1024)) + ((i_100 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_100 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_100 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
            float2 __31;
            uint1 v__41 = *(uint1*)(a_shared_local_cast_14 + 0);
            ((float2*)(&__31))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__41))[0]);
            *(float2*)(a_fragment + (i_100 * 2)) = __31;
          }
          #pragma unroll
          for (int i_101 = 0; i_101 < 16; ++i_101) {
            float2 __32;
              float2 v__42 = *(float2*)(a_fragment + (i_101 * 2));
              float2 v__43 = *(float2*)(b_shared + (((((((int)threadIdx.x) >> 6) * 32) + (((i_101 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 128));
              *(float2*)(&(__32.x)) = tl::mul2(*(float2*)(&(v__42.x)), *(float2*)(&(v__43.x)));
            *(float2*)(a_fragment + (i_101 * 2)) = __32;
          }
          #pragma unroll
          for (int i_102 = 0; i_102 < 8; ++i_102) {
            float4 __33;
              float4 v__44 = *(float4*)(a_fragment + (i_102 * 4));
              float4 v__45 = *(float4*)(mask_fragment + (i_102 * 4));
              *(float2*)(&(__33.x)) = tl::mul2(*(float2*)(&(v__44.x)), *(float2*)(&(v__45.x)));
              *(float2*)(&(__33.z)) = tl::mul2(*(float2*)(&(v__44.z)), *(float2*)(&(v__45.z)));
            *(float4*)(a_fragment + (i_102 * 4)) = __33;
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_103 = 0; i_103 < 4; ++i_103) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_103 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_103 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)a_fragment[(i_103 * 8)]), ((bfloat16_t)a_fragment[((i_103 * 8) + 1)])), __pack_half2(((bfloat16_t)a_fragment[((i_103 * 8) + 2)]), ((bfloat16_t)a_fragment[((i_103 * 8) + 3)])), __pack_half2(((bfloat16_t)a_fragment[((i_103 * 8) + 4)]), ((bfloat16_t)a_fragment[((i_103 * 8) + 5)])), __pack_half2(((bfloat16_t)a_fragment[((i_103 * 8) + 6)]), ((bfloat16_t)a_fragment[((i_103 * 8) + 7)])));
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_104 = 0; i_104 < 8; ++i_104) {
            bfloat16_t tmp_shared_2_1_local_cast_15[8];
            *(uint4*)(tmp_shared_2_1_local_cast_15 + 0) = *(uint4*)(((bfloat16_t*)tmp_shared_2_1) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_104 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
            for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
              float4 __34;
              uint2 v__46 = *(uint2*)(tmp_shared_2_1_local_cast_15 + (vec_1 * 4));
              ((float2*)(&__34))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__46))[0]);
              ((float2*)(&__34))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__46))[1]);
              *(float4*)(odot_fragment_2 + ((i_104 * 8) + (vec_1 * 4))) = __34;
            }
          }
          bar_03[0].arrive();
          bar_03[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_17[16];
            bfloat16_t B_local_17[32];
            #pragma unroll
            for (int i_105 = 0; i_105 < 16; ++i_105) {
              float broadcast_var_26 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(u_fragment + (i_105 * 4)) = make_float4(broadcast_var_26, broadcast_var_26, broadcast_var_26, broadcast_var_26);
            }
            for (int ki_17 = 0; ki_17 < 4; ++ki_17) {
              #pragma unroll
              for (int i_106 = 0; i_106 < 2; ++i_106) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_106 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_17 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_17[(i_106 * 8)])));
              }
              #pragma unroll
              for (int i_107 = 0; i_107 < 4; ++i_107) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)h_shared)[((((((((((int)threadIdx.x) & 127) >> 6) * 4096) + (ki_17 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_107 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_107 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_17[(i_107 * 8)])));
              }
              for (int i_108 = 0; i_108 < 2; ++i_108) {
                for (int j_17 = 0; j_17 < 4; ++j_17) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + ((i_108 * 32) + (j_17 * 8))), reinterpret_cast<const unsigned*>(A_local_17 + (i_108 * 8)), reinterpret_cast<const unsigned*>(B_local_17 + (j_17 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + (((i_108 * 32) + (j_17 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_17 + (i_108 * 8)), reinterpret_cast<const unsigned*>(B_local_17 + ((j_17 * 8) + 4)));
                }
              }
            }
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_109 = 0; i_109 < 8; ++i_109) {
            bfloat16_t broadcast_var_27 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_9;
            if (((((0 <= cur_idx_1) && (cur_idx_1 < num_chunks)) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
              condval_9 = *(uint4*)(h + ((((((((int64_t)cur_idx_1) * (int64_t)262144) + ((((int64_t)batch_idx) * ((int64_t)num_chunks)) * (int64_t)262144)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)16384)) + (((int64_t)i_109) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)6144));
            } else {
              condval_9 = make_uint4(__pack_nv_bfloat162(broadcast_var_27, broadcast_var_27), __pack_nv_bfloat162(broadcast_var_27, broadcast_var_27), __pack_nv_bfloat162(broadcast_var_27, broadcast_var_27), __pack_nv_bfloat162(broadcast_var_27, broadcast_var_27));
            }
            *(uint4*)(((bfloat16_t*)h_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_109 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_9;
          }
          {
            bfloat16_t A_local_18[16];
            bfloat16_t B_local_18[32];
            tl::__sync_thread_partial(5, 128);
            for (int ki_18 = 0; ki_18 < 4; ++ki_18) {
              #pragma unroll
              for (int i_110 = 0; i_110 < 2; ++i_110) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[(((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_110 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_18 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_18 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(A_local_18[(i_110 * 8)])));
              }
              #pragma unroll
              for (int i_111 = 0; i_111 < 4; ++i_111) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)h_shared)[((((((((((int)threadIdx.x) & 127) >> 6) * 4096) + (ki_18 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_111 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_111 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_18[(i_111 * 8)])));
              }
              for (int i_112 = 0; i_112 < 2; ++i_112) {
                for (int j_18 = 0; j_18 < 4; ++j_18) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + ((i_112 * 32) + (j_18 * 8))), reinterpret_cast<const unsigned*>(A_local_18 + (i_112 * 8)), reinterpret_cast<const unsigned*>(B_local_18 + (j_18 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + (((i_112 * 32) + (j_18 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_18 + (i_112 * 8)), reinterpret_cast<const unsigned*>(B_local_18 + ((j_18 * 8) + 4)));
                }
              }
            }
          }
          bar_04[0].arrive();
          bar_04[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_113 = 0; i_113 < 8; ++i_113) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_2_3)[((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_113 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_113 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_113 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 16384)])), __pack_half2(((bfloat16_t)u_fragment[(i_113 * 8)]), ((bfloat16_t)u_fragment[((i_113 * 8) + 1)])), __pack_half2(((bfloat16_t)u_fragment[((i_113 * 8) + 2)]), ((bfloat16_t)u_fragment[((i_113 * 8) + 3)])), __pack_half2(((bfloat16_t)u_fragment[((i_113 * 8) + 4)]), ((bfloat16_t)u_fragment[((i_113 * 8) + 5)])), __pack_half2(((bfloat16_t)u_fragment[((i_113 * 8) + 6)]), ((bfloat16_t)u_fragment[((i_113 * 8) + 7)])));
          }
          #pragma unroll
          for (int i_114 = 0; i_114 < 32; ++i_114) {
            float2 __35;
              float2 v__47 = *(float2*)(u_fragment + (i_114 * 2));
              float2 v__48 = make_float2((g_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_114 >> 4) * 16)) + ((i_114 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/), (g_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_114 >> 4) * 16)) + ((i_114 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/));
              *(float2*)(&(__35.x)) = tl::mul2(*(float2*)(&(v__47.x)), *(float2*)(&(v__48.x)));
            *(float2*)(u_fragment + (i_114 * 2)) = __35;
          }
          #pragma unroll
          for (int i_115 = 0; i_115 < 32; ++i_115) {
            bfloat16_t v_local_cast_16[2];
            if ((((((((num_iters * 64) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + ((i_115 >> 4) * 16)) + ((i_115 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) - (i_s_2 * 64)) < (seq_end_idx + 64)) {
              bfloat16_t broadcast_var_28 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
              uint1 condval_10;
              if (((((1 <= ((((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_115 >> 4) * 16)) + ((i_115 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) >> 6) + num_iters) - i_s_2)) && ((((((((num_iters * 64) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + ((i_115 >> 4) * 16)) + ((i_115 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) - (i_s_2 * 64)) < (num_tokens + 64))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                condval_10 = *(uint1*)(v + (((((((((((((((int64_t)num_iters) * (int64_t)131072) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)65536)) + ((((int64_t)i_115) >> (int64_t)4) * (int64_t)32768)) + ((((int64_t)i_115) & (int64_t)1) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)2048)) + (((int64_t)seq_start_idx) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + (((((int64_t)i_115) & (int64_t)15) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (((int64_t)i_s_2) * (int64_t)131072)) - (int64_t)131328));
              } else {
                condval_10 = make_uint1(__pack_nv_bfloat162(broadcast_var_28, broadcast_var_28));
              }
              *(uint1*)(v_local_cast_16 + 0) = condval_10;
            }
            if ((((((((num_iters * 64) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + ((i_115 >> 4) * 16)) + ((i_115 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + seq_start_idx) - (i_s_2 * 64)) < (seq_end_idx + 64)) {
              float2 __36;
                float2 v__49 = *(float2*)(u_fragment + (i_115 * 2));
                float2 __37;
                uint1 v__50 = *(uint1*)(v_local_cast_16 + 0);
                ((float2*)(&__37))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__50))[0]);
                *(float2*)(&(__36.x)) = tl::add2(*(float2*)(&(v__49.x)), *(float2*)(&(__37.x)));
              *(float2*)(u_fragment + (i_115 * 2)) = __36;
            }
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_116 = 0; i_116 < 8; ++i_116) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_2_2)[((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_116 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_116 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_116 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 16384)])), __pack_half2(((bfloat16_t)u_fragment[(i_116 * 8)]), ((bfloat16_t)u_fragment[((i_116 * 8) + 1)])), __pack_half2(((bfloat16_t)u_fragment[((i_116 * 8) + 2)]), ((bfloat16_t)u_fragment[((i_116 * 8) + 3)])), __pack_half2(((bfloat16_t)u_fragment[((i_116 * 8) + 4)]), ((bfloat16_t)u_fragment[((i_116 * 8) + 5)])), __pack_half2(((bfloat16_t)u_fragment[((i_116 * 8) + 6)]), ((bfloat16_t)u_fragment[((i_116 * 8) + 7)])));
          }
          bar_05[0].arrive();
          bar_05[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_19[16];
            bfloat16_t B_local_19[16];
            #pragma unroll
            for (int i_117 = 0; i_117 < 8; ++i_117) {
              float broadcast_var_29 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(da_fragment + (i_117 * 4)) = make_float4(broadcast_var_29, broadcast_var_29, broadcast_var_29, broadcast_var_29);
            }
            tl::__sync_thread_partial(5, 128);
            for (int ki_19 = 0; ki_19 < 8; ++ki_19) {
              #pragma unroll
              for (int i_118 = 0; i_118 < 2; ++i_118) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_1)[((((((((ki_19 >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_118 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_19 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_19[(i_118 * 8)])));
              }
              #pragma unroll
              for (int i_119 = 0; i_119 < 2; ++i_119) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_2)[(((((((((ki_19 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_119 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_19 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_19[(i_119 * 8)])));
              }
              for (int i_120 = 0; i_120 < 2; ++i_120) {
                for (int j_19 = 0; j_19 < 2; ++j_19) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(da_fragment + ((i_120 * 16) + (j_19 * 8))), reinterpret_cast<const unsigned*>(A_local_19 + (i_120 * 8)), reinterpret_cast<const unsigned*>(B_local_19 + (j_19 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(da_fragment + (((i_120 * 16) + (j_19 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_19 + (i_120 * 8)), reinterpret_cast<const unsigned*>(B_local_19 + ((j_19 * 8) + 4)));
                }
              }
            }
          }
          {
            bfloat16_t A_local_20[16];
            bfloat16_t B_local_20[32];
            #pragma unroll
            for (int i_121 = 0; i_121 < 16; ++i_121) {
              float broadcast_var_30 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(u_fragment + (i_121 * 4)) = make_float4(broadcast_var_30, broadcast_var_30, broadcast_var_30, broadcast_var_30);
            }
            for (int ki_20 = 0; ki_20 < 4; ++ki_20) {
              #pragma unroll
              for (int i_122 = 0; i_122 < 2; ++i_122) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_122 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_20 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_20 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_20[(i_122 * 8)])));
              }
              #pragma unroll
              for (int i_123 = 0; i_123 < 4; ++i_123) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_2)[((((((((((int)threadIdx.x) & 127) >> 6) * 4096) + (ki_20 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_123 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_123 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_20[(i_123 * 8)])));
              }
              for (int i_124 = 0; i_124 < 2; ++i_124) {
                for (int j_20 = 0; j_20 < 4; ++j_20) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + ((i_124 * 32) + (j_20 * 8))), reinterpret_cast<const unsigned*>(A_local_20 + (i_124 * 8)), reinterpret_cast<const unsigned*>(B_local_20 + (j_20 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + (((i_124 * 32) + (j_20 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_20 + (i_124 * 8)), reinterpret_cast<const unsigned*>(B_local_20 + ((j_20 * 8) + 4)));
                }
              }
            }
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_125 = 0; i_125 < 8; ++i_125) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_2_1)[((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_125 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_125 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_125 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 16384)])), __pack_half2(((bfloat16_t)u_fragment[(i_125 * 8)]), ((bfloat16_t)u_fragment[((i_125 * 8) + 1)])), __pack_half2(((bfloat16_t)u_fragment[((i_125 * 8) + 2)]), ((bfloat16_t)u_fragment[((i_125 * 8) + 3)])), __pack_half2(((bfloat16_t)u_fragment[((i_125 * 8) + 4)]), ((bfloat16_t)u_fragment[((i_125 * 8) + 5)])), __pack_half2(((bfloat16_t)u_fragment[((i_125 * 8) + 6)]), ((bfloat16_t)u_fragment[((i_125 * 8) + 7)])));
          }
          bar_06[0].arrive();
          bar_06[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_21[16];
            bfloat16_t B_local_21[16];
            #pragma unroll
            for (int i_126 = 0; i_126 < 8; ++i_126) {
              float broadcast_var_31 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(dp_fragment + (i_126 * 4)) = make_float4(broadcast_var_31, broadcast_var_31, broadcast_var_31, broadcast_var_31);
            }
            tl::__sync_thread_partial(5, 128);
            for (int ki_21 = 0; ki_21 < 8; ++ki_21) {
              #pragma unroll
              for (int i_127 = 0; i_127 < 2; ++i_127) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)do_shared)[((((((((ki_21 >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_127 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_21 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_21[(i_127 * 8)])));
              }
              #pragma unroll
              for (int i_128 = 0; i_128 < 2; ++i_128) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_1)[(((((((((ki_21 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_128 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_21 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_21[(i_128 * 8)])));
              }
              for (int i_129 = 0; i_129 < 2; ++i_129) {
                for (int j_21 = 0; j_21 < 2; ++j_21) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dp_fragment + ((i_129 * 16) + (j_21 * 8))), reinterpret_cast<const unsigned*>(A_local_21 + (i_129 * 8)), reinterpret_cast<const unsigned*>(B_local_21 + (j_21 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dp_fragment + (((i_129 * 16) + (j_21 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_21 + (i_129 * 8)), reinterpret_cast<const unsigned*>(B_local_21 + ((j_21 * 8) + 4)));
                }
              }
            }
          }
          bar_07[0].arrive();
          bar_07[0].wait((i_s_2 & 1));
          #pragma unroll
          for (int i_130 = 0; i_130 < 8; ++i_130) {
            float4 __38;
              float4 v__51 = *(float4*)(da_fragment + (i_130 * 4));
              float4 v__52 = *(float4*)(mask_fragment + (i_130 * 4));
              *(float2*)(&(__38.x)) = tl::mul2(*(float2*)(&(v__51.x)), *(float2*)(&(v__52.x)));
              *(float2*)(&(__38.z)) = tl::mul2(*(float2*)(&(v__51.z)), *(float2*)(&(v__52.z)));
            *(float4*)(da_fragment + (i_130 * 4)) = __38;
          }
          #pragma unroll
          for (int i_131 = 0; i_131 < 16; ++i_131) {
            bfloat16_t tmp_shared_1_1_local_cast_17[2];
            *(uint1*)(tmp_shared_1_1_local_cast_17 + 0) = *(uint1*)(((bfloat16_t*)tmp_shared_1_1) + ((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_131 >> 3) * 1024)) + ((i_131 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_131 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_131 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
            float2 __39;
            uint1 v__53 = *(uint1*)(tmp_shared_1_1_local_cast_17 + 0);
            ((float2*)(&__39))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__53))[0]);
            *(float2*)(p_fragment + (i_131 * 2)) = __39;
          }
          #pragma unroll
          for (int i_132 = 0; i_132 < 8; ++i_132) {
            float4 __40;
              float4 v__54 = *(float4*)(p_fragment + (i_132 * 4));
              float4 v__55 = *(float4*)(dp_fragment + (i_132 * 4));
              *(float2*)(&(__40.x)) = tl::mul2(*(float2*)(&(v__54.x)), *(float2*)(&(v__55.x)));
              *(float2*)(&(__40.z)) = tl::mul2(*(float2*)(&(v__54.z)), *(float2*)(&(v__55.z)));
            *(float4*)(p_fragment + (i_132 * 4)) = __40;
          }
          #pragma unroll
          for (int i_133 = 0; i_133 < 8; ++i_133) {
            float4 __41;
              float4 v__56 = *(float4*)(dp_fragment + (i_133 * 4));
              float4 v__57 = *(float4*)(mask_fragment + (i_133 * 4));
              *(float2*)(&(__41.x)) = tl::mul2(*(float2*)(&(v__56.x)), *(float2*)(&(v__57.x)));
              *(float2*)(&(__41.z)) = tl::mul2(*(float2*)(&(v__56.z)), *(float2*)(&(v__57.z)));
            *(float4*)(dp_fragment + (i_133 * 4)) = __41;
          }
          #pragma unroll
          for (int i_134 = 0; i_134 < 8; ++i_134) {
            float broadcast_var_32 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
            float4 __42;
              float4 v__58 = *(float4*)(dp_fragment + (i_134 * 4));
              float4 v__59 = make_float4(broadcast_var_32, broadcast_var_32, broadcast_var_32, broadcast_var_32);
              *(float2*)(&(__42.x)) = tl::mul2(*(float2*)(&(v__58.x)), *(float2*)(&(v__59.x)));
              *(float2*)(&(__42.z)) = tl::mul2(*(float2*)(&(v__58.z)), *(float2*)(&(v__59.z)));
            *(float4*)(dp_fragment + (i_134 * 4)) = __42;
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_135 = 0; i_135 < 4; ++i_135) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_1)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_135 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_135 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)dp_fragment[(i_135 * 8)]), ((bfloat16_t)dp_fragment[((i_135 * 8) + 1)])), __pack_half2(((bfloat16_t)dp_fragment[((i_135 * 8) + 2)]), ((bfloat16_t)dp_fragment[((i_135 * 8) + 3)])), __pack_half2(((bfloat16_t)dp_fragment[((i_135 * 8) + 4)]), ((bfloat16_t)dp_fragment[((i_135 * 8) + 5)])), __pack_half2(((bfloat16_t)dp_fragment[((i_135 * 8) + 6)]), ((bfloat16_t)dp_fragment[((i_135 * 8) + 7)])));
          }
          bar_08_1[0].arrive();
          bar_08_1[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_22[16];
            bfloat16_t B_local_22[16];
            #pragma unroll
            for (int i_136 = 0; i_136 < 8; ++i_136) {
              float broadcast_var_33 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(dq_frag_r + (i_136 * 4)) = make_float4(broadcast_var_33, broadcast_var_33, broadcast_var_33, broadcast_var_33);
            }
            for (int ki_22 = 0; ki_22 < 8; ++ki_22) {
              #pragma unroll
              for (int i_137 = 0; i_137 < 2; ++i_137) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)do_shared)[((((((((ki_22 >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_137 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_22 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_22 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_22[(i_137 * 8)])));
              }
              #pragma unroll
              for (int i_138 = 0; i_138 < 2; ++i_138) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)h_shared)[(((((((((ki_22 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_138 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_22 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_22 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_22[(i_138 * 8)])));
              }
              for (int i_139 = 0; i_139 < 2; ++i_139) {
                for (int j_22 = 0; j_22 < 2; ++j_22) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_r + ((i_139 * 16) + (j_22 * 8))), reinterpret_cast<const unsigned*>(A_local_22 + (i_139 * 8)), reinterpret_cast<const unsigned*>(B_local_22 + (j_22 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_r + (((i_139 * 16) + (j_22 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_22 + (i_139 * 8)), reinterpret_cast<const unsigned*>(B_local_22 + ((j_22 * 8) + 4)));
                }
              }
            }
          }
          bar_08_2[0].arrive();
          bar_08_3[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_23[16];
            bfloat16_t B_local_23[16];
            #pragma unroll
            for (int i_140 = 0; i_140 < 8; ++i_140) {
              float broadcast_var_34 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(dq_frag_l + (i_140 * 4)) = make_float4(broadcast_var_34, broadcast_var_34, broadcast_var_34, broadcast_var_34);
            }
            for (int ki_23 = 0; ki_23 < 8; ++ki_23) {
              #pragma unroll
              for (int i_141 = 0; i_141 < 2; ++i_141) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)do_shared)[((((((((ki_23 >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_141 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_23 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_23 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_23[(i_141 * 8)])));
              }
              #pragma unroll
              for (int i_142 = 0; i_142 < 2; ++i_142) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)h_shared)[(((((((((ki_23 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_142 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_23 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_23 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_23[(i_142 * 8)])));
              }
              for (int i_143 = 0; i_143 < 2; ++i_143) {
                for (int j_23 = 0; j_23 < 2; ++j_23) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_l + ((i_143 * 16) + (j_23 * 8))), reinterpret_cast<const unsigned*>(A_local_23 + (i_143 * 8)), reinterpret_cast<const unsigned*>(B_local_23 + (j_23 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_l + (((i_143 * 16) + (j_23 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_23 + (i_143 * 8)), reinterpret_cast<const unsigned*>(B_local_23 + ((j_23 * 8) + 4)));
                }
              }
            }
          }
          bar_09[0].arrive();
          bar_09[0].wait((i_s_2 & 1));
          #pragma unroll
          for (int i_144 = 0; i_144 < 16; ++i_144) {
            float2 __43;
              float2 v__60 = *(float2*)(dq_frag_r + (i_144 * 2));
              float2 v__61 = make_float2(g_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_144 >> 3) * 16)) + ((i_144 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_144 >> 3) * 16)) + ((i_144 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
              *(float2*)(&(__43.x)) = tl::mul2(*(float2*)(&(v__60.x)), *(float2*)(&(v__61.x)));
            *(float2*)(dq_frag_r + (i_144 * 2)) = __43;
            float2 __44;
              float2 v__62 = *(float2*)(dq_frag_l + (i_144 * 2));
              float2 v__63 = make_float2(g_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_144 >> 3) * 16)) + ((i_144 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_144 >> 3) * 16)) + ((i_144 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
              *(float2*)(&(__44.x)) = tl::mul2(*(float2*)(&(v__62.x)), *(float2*)(&(v__63.x)));
            *(float2*)(dq_frag_l + (i_144 * 2)) = __44;
          }
          #pragma unroll
          for (int i_145 = 0; i_145 < 8; ++i_145) {
            float broadcast_var_35 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
            float4 __45;
              float4 v__64 = *(float4*)(dq_frag_r + (i_145 * 4));
              float4 v__65 = make_float4(broadcast_var_35, broadcast_var_35, broadcast_var_35, broadcast_var_35);
              *(float2*)(&(__45.x)) = tl::mul2(*(float2*)(&(v__64.x)), *(float2*)(&(v__65.x)));
              *(float2*)(&(__45.z)) = tl::mul2(*(float2*)(&(v__64.z)), *(float2*)(&(v__65.z)));
            *(float4*)(dq_frag_r + (i_145 * 4)) = __45;
            float broadcast_var_36 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
            float4 __46;
              float4 v__66 = *(float4*)(dq_frag_l + (i_145 * 4));
              float4 v__67 = make_float4(broadcast_var_36, broadcast_var_36, broadcast_var_36, broadcast_var_36);
              *(float2*)(&(__46.x)) = tl::mul2(*(float2*)(&(v__66.x)), *(float2*)(&(v__67.x)));
              *(float2*)(&(__46.z)) = tl::mul2(*(float2*)(&(v__66.z)), *(float2*)(&(v__67.z)));
            *(float4*)(dq_frag_l + (i_145 * 4)) = __46;
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_146 = 0; i_146 < 8; ++i_146) {
            bfloat16_t tmp_shared_2_1_local_cast_18[8];
            for (int vec_2 = 0; vec_2 < 2; ++vec_2) {
              uint2 __47;
              float4 v__68 = *(float4*)(odot_fragment_2 + ((i_146 * 8) + (vec_2 * 4)));
              (reinterpret_cast<__nv_bfloat162*>(&__47))[0] = __float22bfloat162_rn(((float2*)(&v__68))[0]);
              (reinterpret_cast<__nv_bfloat162*>(&__47))[1] = __float22bfloat162_rn(((float2*)(&v__68))[1]);
              *(uint2*)(tmp_shared_2_1_local_cast_18 + (vec_2 * 4)) = __47;
            }
            *(uint4*)(((bfloat16_t*)tmp_shared_2_1) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_146 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(tmp_shared_2_1_local_cast_18 + 0);
          }
          bar_10[0].arrive();
          bar_10[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_147 = 0; i_147 < 16; ++i_147) {
            bfloat16_t dqkv_shared_local_cast_19[2];
            uint1 __48;
            float2 v__69 = *(float2*)(dq_frag_l + (i_147 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__48))[0] = __float22bfloat162_rn(((float2*)(&v__69))[0]);
            *(uint1*)(dqkv_shared_local_cast_19 + 0) = __48;
            *(uint1*)(((bfloat16_t*)dqkv_shared) + ((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_147 >> 3) * 1024)) + ((i_147 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_147 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_147 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(dqkv_shared_local_cast_19 + 0);
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_148 = 0; i_148 < 16; ++i_148) {
            bfloat16_t dqkv_shared_local_cast_20[2];
            uint1 __49;
            float2 v__70 = *(float2*)(dq_frag_r + (i_148 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__49))[0] = __float22bfloat162_rn(((float2*)(&v__70))[0]);
            *(uint1*)(dqkv_shared_local_cast_20 + 0) = __49;
            *(uint1*)(((bfloat16_t*)dqkv_shared) + (((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_148 >> 3) * 1024)) + ((i_148 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_148 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_148 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096)) = *(uint1*)(dqkv_shared_local_cast_20 + 0);
          }
          bar_s4_dot_a[0].arrive();
          bar_s4_dot_a[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_149 = 0; i_149 < 8; ++i_149) {
            bfloat16_t dqkv_shared_local_cast_21[8];
            *(uint4*)(dqkv_shared_local_cast_21 + 0) = *(uint4*)(((bfloat16_t*)dqkv_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_149 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
            for (int vec_3 = 0; vec_3 < 2; ++vec_3) {
              float4 __50;
                float4 v__71 = *(float4*)(odot_fragment_2 + ((i_149 * 8) + (vec_3 * 4)));
                float4 __51;
                uint2 v__72 = *(uint2*)(dqkv_shared_local_cast_21 + (vec_3 * 4));
                ((float2*)(&__51))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__72))[0]);
                ((float2*)(&__51))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__72))[1]);
                *(float2*)(&(__50.x)) = tl::mul2(*(float2*)(&(v__71.x)), *(float2*)(&(__51.x)));
                *(float2*)(&(__50.z)) = tl::mul2(*(float2*)(&(v__71.z)), *(float2*)(&(__51.z)));
              *(float4*)(odot_fragment_2 + ((i_149 * 8) + (vec_3 * 4))) = __50;
            }
          }
          #pragma unroll
          for (int i_150 = 0; i_150 < 8; ++i_150) {
            float broadcast_var_37 = 0x0p+0f/*0.000000e+00*/;
            dg_fragment_2_pack[i_150] = make_float2(broadcast_var_37, broadcast_var_37);
            #pragma unroll
            for (int rv_3 = 0; rv_3 < 4; ++rv_3) {
              dg_fragment_2_pack[i_150] = tl::add2(dg_fragment_2_pack[i_150], *(float2*)(odot_fragment_2 + ((i_150 * 8) + (rv_3 * 2))));
            }
            dg_fragment_2[i_150] = ((dg_fragment_2_pack[i_150]).x + (dg_fragment_2_pack[i_150]).y);
            dg_fragment_2[i_150] = tl::AllReduce<tl::SumOp, 16, 1, 256, tl::NamedBarrier<128>>::run(dg_fragment_2[i_150]);
          }
          {
            bfloat16_t A_local_24[16];
            bfloat16_t B_local_24[16];
            for (int ki_24 = 0; ki_24 < 4; ++ki_24) {
              #pragma unroll
              for (int i_151 = 0; i_151 < 2; ++i_151) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_1_1)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_151 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_24 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_24 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_24[(i_151 * 8)])));
              }
              #pragma unroll
              for (int i_152 = 0; i_152 < 2; ++i_152) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_2)[(((((ki_24 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_152) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_24[(i_152 * 8)])));
              }
              for (int i_153 = 0; i_153 < 2; ++i_153) {
                for (int j_24 = 0; j_24 < 2; ++j_24) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_l + ((i_153 * 16) + (j_24 * 8))), reinterpret_cast<const unsigned*>(A_local_24 + (i_153 * 8)), reinterpret_cast<const unsigned*>(B_local_24 + (j_24 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_l + (((i_153 * 16) + (j_24 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_24 + (i_153 * 8)), reinterpret_cast<const unsigned*>(B_local_24 + ((j_24 * 8) + 4)));
                }
              }
            }
          }
          {
            bfloat16_t A_local_25[16];
            bfloat16_t B_local_25[16];
            for (int ki_25 = 0; ki_25 < 4; ++ki_25) {
              #pragma unroll
              for (int i_154 = 0; i_154 < 2; ++i_154) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_1_1)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_154 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_25 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_25 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_25[(i_154 * 8)])));
              }
              #pragma unroll
              for (int i_155 = 0; i_155 < 2; ++i_155) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_2_2)[((((((ki_25 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_155) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(B_local_25[(i_155 * 8)])));
              }
              for (int i_156 = 0; i_156 < 2; ++i_156) {
                for (int j_25 = 0; j_25 < 2; ++j_25) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_r + ((i_156 * 16) + (j_25 * 8))), reinterpret_cast<const unsigned*>(A_local_25 + (i_156 * 8)), reinterpret_cast<const unsigned*>(B_local_25 + (j_25 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_frag_r + (((i_156 * 16) + (j_25 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_25 + (i_156 * 8)), reinterpret_cast<const unsigned*>(B_local_25 + ((j_25 * 8) + 4)));
                }
              }
            }
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_157 = 0; i_157 < 16; ++i_157) {
            bfloat16_t dqkv_shared_local_cast_22[2];
            uint1 __52;
            float2 v__73 = *(float2*)(dq_frag_l + (i_157 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__52))[0] = __float22bfloat162_rn(((float2*)(&v__73))[0]);
            *(uint1*)(dqkv_shared_local_cast_22 + 0) = __52;
            *(uint1*)(((bfloat16_t*)dqkv_shared) + ((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_157 >> 3) * 1024)) + ((i_157 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_157 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_157 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(dqkv_shared_local_cast_22 + 0);
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_158 = 0; i_158 < 16; ++i_158) {
            bfloat16_t dqkv_shared_local_cast_23[2];
            uint1 __53;
            float2 v__74 = *(float2*)(dq_frag_r + (i_158 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__53))[0] = __float22bfloat162_rn(((float2*)(&v__74))[0]);
            *(uint1*)(dqkv_shared_local_cast_23 + 0) = __53;
            *(uint1*)(((bfloat16_t*)dqkv_shared) + (((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_158 >> 3) * 1024)) + ((i_158 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_158 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_158 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096)) = *(uint1*)(dqkv_shared_local_cast_23 + 0);
          }
          bar_11[0].arrive();
          bar_11[0].wait((i_s_2 & 1));
          #pragma unroll
          for (int i_159 = 0; i_159 < 16; ++i_159) {
            bfloat16_t a_shared_local_cast_24[2];
            *(uint1*)(a_shared_local_cast_24 + 0) = *(uint1*)(((bfloat16_t*)a_shared) + ((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_159 >> 3) * 1024)) + ((i_159 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_159 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_159 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
            float2 __54;
            uint1 v__75 = *(uint1*)(a_shared_local_cast_24 + 0);
            ((float2*)(&__54))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__75))[0]);
            *(float2*)(a_fragment + (i_159 * 2)) = __54;
          }
          #pragma unroll
          for (int i_160 = 0; i_160 < 8; ++i_160) {
            float4 __55;
              float4 v__76 = *(float4*)(a_fragment + (i_160 * 4));
              float4 v__77 = *(float4*)(da_fragment + (i_160 * 4));
              *(float2*)(&(__55.x)) = tl::mul2(*(float2*)(&(v__76.x)), *(float2*)(&(v__77.x)));
              *(float2*)(&(__55.z)) = tl::mul2(*(float2*)(&(v__76.z)), *(float2*)(&(v__77.z)));
            *(float4*)(a_fragment + (i_160 * 4)) = __55;
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_161 = 0; i_161 < 4; ++i_161) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_3)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_161 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_161 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)a_fragment[(i_161 * 8)]), ((bfloat16_t)a_fragment[((i_161 * 8) + 1)])), __pack_half2(((bfloat16_t)a_fragment[((i_161 * 8) + 2)]), ((bfloat16_t)a_fragment[((i_161 * 8) + 3)])), __pack_half2(((bfloat16_t)a_fragment[((i_161 * 8) + 4)]), ((bfloat16_t)a_fragment[((i_161 * 8) + 5)])), __pack_half2(((bfloat16_t)a_fragment[((i_161 * 8) + 6)]), ((bfloat16_t)a_fragment[((i_161 * 8) + 7)])));
          }
          #pragma unroll
          for (int i_162 = 0; i_162 < 16; ++i_162) {
            float2 __56;
              float2 v__78 = *(float2*)(a_fragment + (i_162 * 2));
              float2 v__79 = *(float2*)(b_shared + (((((((int)threadIdx.x) >> 6) * 32) + (((i_162 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 128));
              *(float2*)(&(__56.x)) = tl::mul2(*(float2*)(&(v__78.x)), *(float2*)(&(v__79.x)));
            *(float2*)(a_fragment + (i_162 * 2)) = __56;
          }
          #pragma unroll
          for (int i_163 = 0; i_163 < 8; ++i_163) {
            float4 __57;
              float4 v__80 = *(float4*)(a_fragment + (i_163 * 4));
              float4 v__81 = *(float4*)(p_fragment + (i_163 * 4));
              *(float2*)(&(__57.x)) = tl::add2(*(float2*)(&(v__80.x)), *(float2*)(&(v__81.x)));
              *(float2*)(&(__57.z)) = tl::add2(*(float2*)(&(v__80.z)), *(float2*)(&(v__81.z)));
            *(float4*)(a_fragment + (i_163 * 4)) = __57;
          }
          #pragma unroll
          for (int i_164 = 0; i_164 < 32; ++i_164) {
            float v__82 = a_fragment[i_164];
            uint x = (*(uint *)(&(v__82)));
            lo_fragment[i_164] = ((ushort)(x & (uint)65535));
            hi_fragment[i_164] = ((ushort)(x >> (uint)16));
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_165 = 0; i_165 < 16; ++i_165) {
            for (int j_t_vec = 0; j_t_vec < 2; ++j_t_vec) {
              ushort v__83 = hi_fragment[(((((i_165 >> 3) * 16) + ((i_165 & 3) * 4)) + (((i_165 & 7) >> 2) * 2)) + j_t_vec)];
              ((bfloat16_t*)tmp_shared_1_2)[((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_165 >> 2) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_165 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_165 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + j_t_vec)] = (*(bfloat16_t *)(&(v__83)));
            }
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_166 = 0; i_166 < 16; ++i_166) {
            for (int j_t_vec_1 = 0; j_t_vec_1 < 2; ++j_t_vec_1) {
              bfloat16_t v__84 = ((bfloat16_t*)tmp_shared_1_2)[((((((((((((int)threadIdx.x) >> 6) * 2048) + ((i_166 & 3) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (j_t_vec_1 * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + ((((i_166 >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_166 & 7) >> 2) + j_t_vec_1) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 8192)];
              hi_fragment[(((((i_166 >> 3) * 16) + ((i_166 & 3) * 4)) + (((i_166 & 7) >> 2) * 2)) + j_t_vec_1)] = (*(ushort *)(&(v__84)));
            }
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_167 = 0; i_167 < 16; ++i_167) {
            for (int j_t_vec_2 = 0; j_t_vec_2 < 2; ++j_t_vec_2) {
              ushort v__85 = lo_fragment[(((((i_167 >> 3) * 16) + ((i_167 & 3) * 4)) + (((i_167 & 7) >> 2) * 2)) + j_t_vec_2)];
              ((bfloat16_t*)tmp_shared_1_2)[((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_167 >> 2) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_167 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_167 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + j_t_vec_2)] = (*(bfloat16_t *)(&(v__85)));
            }
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_168 = 0; i_168 < 16; ++i_168) {
            for (int j_t_vec_3 = 0; j_t_vec_3 < 2; ++j_t_vec_3) {
              bfloat16_t v__86 = ((bfloat16_t*)tmp_shared_1_2)[((((((((((((int)threadIdx.x) >> 6) * 2048) + ((i_168 & 3) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (j_t_vec_3 * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + ((((i_168 >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_168 & 7) >> 2) + j_t_vec_3) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 8192)];
              lo_fragment[(((((i_168 >> 3) * 16) + ((i_168 & 3) * 4)) + (((i_168 & 7) >> 2) * 2)) + j_t_vec_3)] = (*(ushort *)(&(v__86)));
            }
          }
          #pragma unroll
          for (int i_169 = 0; i_169 < 32; ++i_169) {
            uint32_fragment[i_169] = ((uint)((hi_fragment[i_169] << (ushort)16) + lo_fragment[i_169]));
            uint v__87 = uint32_fragment[i_169];
            p_fragment[i_169] = (*(float *)(&(v__87)));
          }
          #pragma unroll
          for (int i_170 = 0; i_170 < 8; ++i_170) {
            float4 __58;
              float4 v__88 = *(float4*)(a_fragment + (i_170 * 4));
              float4 v__89 = *(float4*)(p_fragment + (i_170 * 4));
              *(float2*)(&(__58.x)) = tl::sub2(*(float2*)(&(v__88.x)), *(float2*)(&(v__89.x)));
              *(float2*)(&(__58.z)) = tl::sub2(*(float2*)(&(v__88.z)), *(float2*)(&(v__89.z)));
            *(float4*)(a_fragment + (i_170 * 4)) = __58;
          }
          #pragma unroll
          for (int i_171 = 0; i_171 < 4; ++i_171) {
            dg_fragment_2b[i_171] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_4 = 0; rv_4 < 8; ++rv_4) {
              dg_fragment_2b[i_171] = (dg_fragment_2b[i_171] + a_fragment[(((((i_171 >> 1) * 16) + ((rv_4 & 3) * 4)) + ((i_171 & 1) * 2)) + (rv_4 >> 2))]);
            }
            tl::__sync_thread_partial(5, 128);
            dg_fragment_2b[i_171] = tl::AllReduce<tl::SumOp, 128, 64, 256, tl::NamedBarrier<128>>::run(dg_fragment_2b[i_171], (&(((float*)workspace_5)[0])));
            dg_fragment_2b[i_171] = tl::AllReduce<tl::SumOp, 4, 1, 256, tl::NamedBarrier<128>>::run(dg_fragment_2b[i_171]);
          }
          tl::__sync_thread_partial(5, 128);
          if ((((int)threadIdx.x) % 16) == 0) {
            #pragma unroll
            for (int i_172 = 0; i_172 < 8; ++i_172) {
              dg_shared[(((i_172 * 8) + (((int)threadIdx.x) >> 4)) - 16)] = (dg_shared[(((i_172 * 8) + (((int)threadIdx.x) >> 4)) - 16)] + dg_fragment_2[i_172]);
            }
          }
          bar_dg_a_first[0].arrive();
          bar_dg_a_first[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial(5, 128);
          if ((((((int)threadIdx.x) & 3) * 2) + (((int)threadIdx.x) >> 6)) == 4) {
            #pragma unroll
            for (int i_173 = 0; i_173 < 4; ++i_173) {
              dg_shared[(((((((int)threadIdx.x) & 63) >> 5) * 32) + (i_173 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = (dg_shared[(((((((int)threadIdx.x) & 63) >> 5) * 32) + (i_173 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] + dg_fragment_2b[i_173]);
            }
          }
          #pragma unroll
          for (int i_174 = 0; i_174 < 32; ++i_174) {
            a_fragment[i_174] = ((float)((bfloat16_t*)tmp_shared_1_3)[((((((((((((int)threadIdx.x) >> 6) * 2048) + (((i_174 & 15) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + ((i_174 & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + ((((i_174 >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_174 & 3) >> 1) + (i_174 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 8192)]);
          }
          #pragma unroll
          for (int i_175 = 0; i_175 < 4; ++i_175) {
            db_fragment[i_175] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_5 = 0; rv_5 < 8; ++rv_5) {
              db_fragment[i_175] = (db_fragment[i_175] + a_fragment[(((((i_175 >> 1) * 16) + ((rv_5 & 3) * 4)) + ((i_175 & 1) * 2)) + (rv_5 >> 2))]);
            }
            tl::__sync_thread_partial(5, 128);
            db_fragment[i_175] = tl::AllReduce<tl::SumOp, 128, 64, 256, tl::NamedBarrier<128>>::run(db_fragment[i_175], (&(((float*)workspace_6)[0])));
            db_fragment[i_175] = tl::AllReduce<tl::SumOp, 4, 1, 256, tl::NamedBarrier<128>>::run(db_fragment[i_175]);
          }
          #pragma unroll
          for (int i_176 = 0; i_176 < 16; ++i_176) {
            float2 __59;
              float2 v__90 = *(float2*)(da_fragment + (i_176 * 2));
              float2 v__91 = *(float2*)(b_shared + (((((((int)threadIdx.x) >> 6) * 32) + (((i_176 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 128));
              *(float2*)(&(__59.x)) = tl::mul2(*(float2*)(&(v__90.x)), *(float2*)(&(v__91.x)));
            *(float2*)(da_fragment + (i_176 * 2)) = __59;
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_177 = 0; i_177 < 4; ++i_177) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_177 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_177 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)da_fragment[(i_177 * 8)]), ((bfloat16_t)da_fragment[((i_177 * 8) + 1)])), __pack_half2(((bfloat16_t)da_fragment[((i_177 * 8) + 2)]), ((bfloat16_t)da_fragment[((i_177 * 8) + 3)])), __pack_half2(((bfloat16_t)da_fragment[((i_177 * 8) + 4)]), ((bfloat16_t)da_fragment[((i_177 * 8) + 5)])), __pack_half2(((bfloat16_t)da_fragment[((i_177 * 8) + 6)]), ((bfloat16_t)da_fragment[((i_177 * 8) + 7)])));
          }
          bar_13[0].arrive();
          bar_13[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_26[16];
            bfloat16_t B_local_26[16];
            #pragma unroll
            for (int i_178 = 0; i_178 < 8; ++i_178) {
              float broadcast_var_38 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(da_fragment + (i_178 * 4)) = make_float4(broadcast_var_38, broadcast_var_38, broadcast_var_38, broadcast_var_38);
            }
            tl::__sync_thread_partial(5, 128);
            for (int ki_26 = 0; ki_26 < 4; ++ki_26) {
              #pragma unroll
              for (int i_179 = 0; i_179 < 2; ++i_179) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)a_shared)[((((((ki_26 * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_179) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_26[(i_179 * 8)])));
              }
              #pragma unroll
              for (int i_180 = 0; i_180 < 2; ++i_180) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)tmp_shared_1_2)[(((((ki_26 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_180) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_26[(i_180 * 8)])));
              }
              for (int i_181 = 0; i_181 < 2; ++i_181) {
                for (int j_26 = 0; j_26 < 2; ++j_26) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(da_fragment + ((i_181 * 16) + (j_26 * 8))), reinterpret_cast<const unsigned*>(A_local_26 + (i_181 * 8)), reinterpret_cast<const unsigned*>(B_local_26 + (j_26 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(da_fragment + (((i_181 * 16) + (j_26 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_26 + (i_181 * 8)), reinterpret_cast<const unsigned*>(B_local_26 + ((j_26 * 8) + 4)));
                }
              }
            }
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_182 = 0; i_182 < 4; ++i_182) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_182 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_182 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)da_fragment[(i_182 * 8)]), ((bfloat16_t)da_fragment[((i_182 * 8) + 1)])), __pack_half2(((bfloat16_t)da_fragment[((i_182 * 8) + 2)]), ((bfloat16_t)da_fragment[((i_182 * 8) + 3)])), __pack_half2(((bfloat16_t)da_fragment[((i_182 * 8) + 4)]), ((bfloat16_t)da_fragment[((i_182 * 8) + 5)])), __pack_half2(((bfloat16_t)da_fragment[((i_182 * 8) + 6)]), ((bfloat16_t)da_fragment[((i_182 * 8) + 7)])));
          }
          {
            bfloat16_t A_local_27[16];
            bfloat16_t B_local_27[16];
            #pragma unroll
            for (int i_183 = 0; i_183 < 8; ++i_183) {
              float broadcast_var_39 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(da_fragment + (i_183 * 4)) = make_float4(broadcast_var_39, broadcast_var_39, broadcast_var_39, broadcast_var_39);
            }
            tl::__sync_thread_partial(5, 128);
            for (int ki_27 = 0; ki_27 < 4; ++ki_27) {
              #pragma unroll
              for (int i_184 = 0; i_184 < 2; ++i_184) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_184 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_27 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_27 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_27[(i_184 * 8)])));
              }
              #pragma unroll
              for (int i_185 = 0; i_185 < 2; ++i_185) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_shared)[(((((((((((int)threadIdx.x) & 127) >> 6) * 2048) + (i_185 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_27 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_27 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_27[(i_185 * 8)])));
              }
              for (int i_186 = 0; i_186 < 2; ++i_186) {
                for (int j_27 = 0; j_27 < 2; ++j_27) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(da_fragment + ((i_186 * 16) + (j_27 * 8))), reinterpret_cast<const unsigned*>(A_local_27 + (i_186 * 8)), reinterpret_cast<const unsigned*>(B_local_27 + (j_27 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(da_fragment + (((i_186 * 16) + (j_27 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_27 + (i_186 * 8)), reinterpret_cast<const unsigned*>(B_local_27 + ((j_27 * 8) + 4)));
                }
              }
            }
          }
          {
            bfloat16_t A_local_28[16];
            bfloat16_t B_local_28[16];
            #pragma unroll
            for (int i_187 = 0; i_187 < 8; ++i_187) {
              float broadcast_var_40 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(a_fragment + (i_187 * 4)) = make_float4(broadcast_var_40, broadcast_var_40, broadcast_var_40, broadcast_var_40);
            }
            for (int ki_28 = 0; ki_28 < 8; ++ki_28) {
              #pragma unroll
              for (int i_188 = 0; i_188 < 2; ++i_188) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_2)[((((((((ki_28 >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_188 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_28 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_28 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_28[(i_188 * 8)])));
              }
              #pragma unroll
              for (int i_189 = 0; i_189 < 2; ++i_189) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)tmp_shared_2_2)[(((((((((ki_28 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_189 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_28 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_28 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_28[(i_189 * 8)])));
              }
              for (int i_190 = 0; i_190 < 2; ++i_190) {
                for (int j_28 = 0; j_28 < 2; ++j_28) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(a_fragment + ((i_190 * 16) + (j_28 * 8))), reinterpret_cast<const unsigned*>(A_local_28 + (i_190 * 8)), reinterpret_cast<const unsigned*>(B_local_28 + (j_28 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(a_fragment + (((i_190 * 16) + (j_28 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_28 + (i_190 * 8)), reinterpret_cast<const unsigned*>(B_local_28 + ((j_28 * 8) + 4)));
                }
              }
            }
          }
          bar_14[0].arrive();
          bar_14[0].wait((i_s_2 & 1));
          #pragma unroll
          for (int i_191 = 0; i_191 < 32; ++i_191) {
            if ((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_191 >> 4) * 16)) + (((i_191 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 128) <= (((((((int)threadIdx.x) >> 6) * 32) + (((i_191 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_191 & 1))) {
              da_fragment[i_191] = 0x0p+0f/*0.000000e+00*/;
            } else {
              da_fragment[i_191] = (da_fragment[i_191] * -0x1p+0f/*-1.000000e+00*/);
            }
          }
          #pragma unroll
          for (int i_192 = 0; i_192 < 8; ++i_192) {
            float4 __60;
              float4 v__92 = *(float4*)(a_fragment + (i_192 * 4));
              float4 v__93 = *(float4*)(da_fragment + (i_192 * 4));
              *(float2*)(&(__60.x)) = tl::mul2(*(float2*)(&(v__92.x)), *(float2*)(&(v__93.x)));
              *(float2*)(&(__60.z)) = tl::mul2(*(float2*)(&(v__92.z)), *(float2*)(&(v__93.z)));
            *(float4*)(a_fragment + (i_192 * 4)) = __60;
          }
          #pragma unroll
          for (int i_193 = 0; i_193 < 4; ++i_193) {
            db_fragment_clear[i_193] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_6 = 0; rv_6 < 8; ++rv_6) {
              db_fragment_clear[i_193] = (db_fragment_clear[i_193] + a_fragment[(((((i_193 >> 1) * 16) + ((rv_6 & 3) * 4)) + ((i_193 & 1) * 2)) + (rv_6 >> 2))]);
            }
            tl::__sync_thread_partial(5, 128);
            db_fragment_clear[i_193] = tl::AllReduce<tl::SumOp, 128, 64, 256, tl::NamedBarrier<128>>::run(db_fragment_clear[i_193], (&(((float*)workspace)[0])));
            db_fragment_clear[i_193] = tl::AllReduce<tl::SumOp, 4, 1, 256, tl::NamedBarrier<128>>::run(db_fragment_clear[i_193]);
            db_fragment[i_193] = (db_fragment[i_193] + db_fragment_clear[i_193]);
          }
          if ((((((int)threadIdx.x) & 3) * 2) + (((int)threadIdx.x) >> 6)) == 4) {
            #pragma unroll
            for (int i_194 = 0; i_194 < 4; ++i_194) {
              db_shared[(((((((int)threadIdx.x) & 63) >> 5) * 32) + (i_194 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = db_fragment[i_194];
            }
          }
          #pragma unroll
          for (int i_195 = 0; i_195 < 16; ++i_195) {
            float2 __61;
              float2 v__94 = *(float2*)(da_fragment + (i_195 * 2));
              float2 v__95 = make_float2(b_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_195 >> 3) * 16)) + ((i_195 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], b_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_195 >> 3) * 16)) + ((i_195 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
              *(float2*)(&(__61.x)) = tl::mul2(*(float2*)(&(v__94.x)), *(float2*)(&(v__95.x)));
            *(float2*)(da_fragment + (i_195 * 2)) = __61;
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_196 = 0; i_196 < 4; ++i_196) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_196 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_196 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)da_fragment[(i_196 * 8)]), ((bfloat16_t)da_fragment[((i_196 * 8) + 1)])), __pack_half2(((bfloat16_t)da_fragment[((i_196 * 8) + 2)]), ((bfloat16_t)da_fragment[((i_196 * 8) + 3)])), __pack_half2(((bfloat16_t)da_fragment[((i_196 * 8) + 4)]), ((bfloat16_t)da_fragment[((i_196 * 8) + 5)])), __pack_half2(((bfloat16_t)da_fragment[((i_196 * 8) + 6)]), ((bfloat16_t)da_fragment[((i_196 * 8) + 7)])));
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_197 = 0; i_197 < 32; ++i_197) {
            da_fragment[i_197] = (da_fragment[i_197] + ((float)((bfloat16_t*)tmp_shared_1_2)[((((((((((((int)threadIdx.x) >> 6) * 2048) + (((i_197 & 15) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + ((i_197 & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + ((((i_197 >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_197 & 3) >> 1) + (i_197 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 8192)]));
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_198 = 0; i_198 < 4; ++i_198) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)tmp_shared_1_2)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_198 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_198 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)da_fragment[(i_198 * 8)]), ((bfloat16_t)da_fragment[((i_198 * 8) + 1)])), __pack_half2(((bfloat16_t)da_fragment[((i_198 * 8) + 2)]), ((bfloat16_t)da_fragment[((i_198 * 8) + 3)])), __pack_half2(((bfloat16_t)da_fragment[((i_198 * 8) + 4)]), ((bfloat16_t)da_fragment[((i_198 * 8) + 5)])), __pack_half2(((bfloat16_t)da_fragment[((i_198 * 8) + 6)]), ((bfloat16_t)da_fragment[((i_198 * 8) + 7)])));
          }
          bar_15[0].arrive();
          bar_15[0].wait((i_s_2 & 1));
        }
      } else {
        tl::warpgroup_reg_dealloc<24>();
        if (((int)threadIdx.x) < 416) {
          for (int i_s_3 = 0; i_s_3 < (num_iters - 1); ++i_s_3) {
            int chunk_idx = ((num_iters - i_s_3) - 2);
            int left = ((chunk_idx * 64) + seq_start_idx);
            bar_00[0].arrive();
            bar_00[0].wait((i_s_3 & 1));
            bar_03[0].wait((i_s_3 & 1));
            #pragma unroll
            for (int i_199 = 0; i_199 < 2; ++i_199) {
              float condval_11;
              if (((((12 <= (((left + ((int)threadIdx.x)) >> 5) + i_199)) && ((((i_199 * 32) + left) + ((int)threadIdx.x)) < (num_tokens + 384))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                condval_11 = g[((((((((int64_t)i_199) * (int64_t)512) + (((int64_t)left) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)6144)];
              } else {
                condval_11 = 0x0p+0f/*0.000000e+00*/;
              }
              g_shared[(((i_199 * 32) + ((int)threadIdx.x)) - 384)] = condval_11;
            }
            bar_07[0].wait((i_s_3 & 1));
            if (tl::tl_shuffle_elect<32>()) {
              bar_00[0].expect_transaction(16384);
              tl::fence_proxy_async();
              #pragma unroll
              for (int i_200 = 0; i_200 < 2; ++i_200) {
                tl::tma_load(k_desc, bar_00[0], (&(((bfloat16_t*)k_shared)[(i_200 * 4096)])), (i_200 * 64), left, (((int)blockIdx.x) & 15), batch_idx);
              }
            }
            bar_15[0].wait((i_s_3 & 1));
            if (tl::tl_shuffle_elect<32>()) {
              bar_00[0].expect_transaction(16384);
              tl::fence_proxy_async();
              #pragma unroll
              for (int i_201 = 0; i_201 < 2; ++i_201) {
                tl::tma_load(q_desc, bar_00[0], (&(((bfloat16_t*)tmp_shared_2_1)[(i_201 * 4096)])), (i_201 * 64), left, (((int)blockIdx.x) & 15), batch_idx);
              }
            }
          }
          if (0 < num_iters) {
            bar_00[0].arrive();
          }
        } else {
          if (((int)threadIdx.x) < 448) {
            tl::fence_proxy_async();
            for (int i_s_4 = 0; i_s_4 < num_iters; ++i_s_4) {
              int left_1 = ((((num_iters * 64) + seq_start_idx) - (i_s_4 * 64)) - 64);
              bar_00[0].arrive();
              bar_00[0].wait((i_s_4 & 1));
              bar_01[0].wait((i_s_4 & 1));
              if (i_s_4 == 1) {
                if (0 <= batch_idx) {
                  #pragma unroll
                  for (int i_202 = 0; i_202 < 32; ++i_202) {
                    if (((((i_202 * 2) + (((int)threadIdx.x) >> 4)) + left_1) + 38) < seq_end_idx) {
                      if (-38 <= (((i_202 * 2) + (((int)threadIdx.x) >> 4)) + left_1)) {
                        if (((((i_202 * 2) + (((int)threadIdx.x) >> 4)) + left_1) + 38) < num_tokens) {
                          if (batch_idx < batch_size) {
                            *(uint4*)(dk + (((((((((int64_t)i_202) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)left_1) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)77824)) = *(uint4*)(((bfloat16_t*)dqkv_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + ((i_202 >> 2) * 512)) + (((((i_202 * 2) + (((int)threadIdx.x) >> 4)) + 6) & 7) * 64)) + ((((((((i_202 * 256) + (((int)threadIdx.x) * 8)) + 768) & 1023) >> 9) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_202 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if (1 < i_s_4) {
                  if (tl::tl_shuffle_elect<32>()) {
                    #pragma unroll
                    for (int i_203 = 0; i_203 < 2; ++i_203) {
                      tl::tma_store(dk_desc, (&(((bfloat16_t*)dqkv_shared)[(i_203 * 4096)])), (i_203 * 64), (left_1 + 64), (((int)blockIdx.x) & 15), batch_idx);
                    }
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0, true>();
                  }
                }
              }
              bar_04[0].arrive();
              bar_04[0].wait((i_s_4 & 1));
              bar_05[0].wait((i_s_4 & 1));
              if (i_s_4 == 0) {
                if (0 <= batch_idx) {
                  #pragma unroll
                  for (int i_204 = 0; i_204 < 32; ++i_204) {
                    if ((((i_204 * 2) + (((int)threadIdx.x) >> 4)) + left_1) < (seq_end_idx + 26)) {
                      if (13 <= ((((((int)threadIdx.x) >> 4) + left_1) >> 1) + i_204)) {
                        if ((((i_204 * 2) + (((int)threadIdx.x) >> 4)) + left_1) < (num_tokens + 26)) {
                          if (batch_idx < batch_size) {
                            *(uint4*)(dv + (((((((((int64_t)i_204) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)left_1) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)53248)) = *(uint4*)(((bfloat16_t*)dqkv_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + ((i_204 >> 2) * 512)) + (((((i_204 * 2) + (((int)threadIdx.x) >> 4)) + 6) & 7) * 64)) + ((((((((i_204 * 256) + (((int)threadIdx.x) * 8)) + 768) & 1023) >> 9) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_204 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if (tl::tl_shuffle_elect<32>()) {
                  #pragma unroll
                  for (int i_205 = 0; i_205 < 2; ++i_205) {
                    tl::tma_store(dv_desc, (&(((bfloat16_t*)dqkv_shared)[(i_205 * 4096)])), (i_205 * 64), left_1, (((int)blockIdx.x) & 15), batch_idx);
                  }
                  tl::tma_store_arrive();
                  tl::tma_store_wait<0, true>();
                }
              }
              bar_10[0].arrive();
              bar_10[0].wait((i_s_4 & 1));
              bar_11[0].wait((i_s_4 & 1));
              if (i_s_4 == 0) {
                if (0 <= batch_idx) {
                  #pragma unroll
                  for (int i_206 = 0; i_206 < 32; ++i_206) {
                    if ((((i_206 * 2) + (((int)threadIdx.x) >> 4)) + left_1) < (seq_end_idx + 26)) {
                      if (13 <= ((((((int)threadIdx.x) >> 4) + left_1) >> 1) + i_206)) {
                        if ((((i_206 * 2) + (((int)threadIdx.x) >> 4)) + left_1) < (num_tokens + 26)) {
                          if (batch_idx < batch_size) {
                            *(uint4*)(dq + (((((((((int64_t)i_206) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)left_1) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)53248)) = *(uint4*)(((bfloat16_t*)dqkv_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + ((i_206 >> 2) * 512)) + (((((i_206 * 2) + (((int)threadIdx.x) >> 4)) + 6) & 7) * 64)) + ((((((((i_206 * 256) + (((int)threadIdx.x) * 8)) + 768) & 1023) >> 9) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_206 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if (tl::tl_shuffle_elect<32>()) {
                  #pragma unroll
                  for (int i_207 = 0; i_207 < 2; ++i_207) {
                    tl::tma_store(dq_desc, (&(((bfloat16_t*)dqkv_shared)[(i_207 * 4096)])), (i_207 * 64), left_1, (((int)blockIdx.x) & 15), batch_idx);
                  }
                  tl::tma_store_arrive();
                  tl::tma_store_wait<0, true>();
                }
              }
            }
          } else {
            if (((int)threadIdx.x) < 480) {
              for (int i_s_5 = 0; i_s_5 < (num_iters - 1); ++i_s_5) {
                int chunk_idx_1 = ((num_iters - i_s_5) - 2);
                int left_2 = ((chunk_idx_1 * 64) + seq_start_idx);
                bar_02[0].arrive();
                bar_02[0].wait((i_s_5 & 1));
                bar_10[0].wait((i_s_5 & 1));
                if (tl::tl_shuffle_elect<32>()) {
                  bar_02[0].expect_transaction(16384);
                  tl::fence_proxy_async();
                  #pragma unroll
                  for (int i_208 = 0; i_208 < 2; ++i_208) {
                    tl::tma_load(h_desc, bar_02[0], (&(((bfloat16_t*)h_shared)[(i_208 * 4096)])), (i_208 * 64), 0, (((int)blockIdx.x) & 15), (chunk_start_idx + chunk_idx_1), batch_idx);
                  }
                }
                bar_14[0].wait((i_s_5 & 1));
                if (tl::tl_shuffle_elect<32>()) {
                  bar_02[0].expect_transaction(8192);
                  tl::fence_proxy_async();
                  tl::tma_load(a_desc, bar_02[0], (&(((bfloat16_t*)a_shared)[0])), 0, left_2, (((int)blockIdx.x) & 15), batch_idx);
                  bar_02[0].expect_transaction(16384);
                  #pragma unroll
                  for (int i_209 = 0; i_209 < 2; ++i_209) {
                    tl::tma_load(do_desc, bar_02[0], (&(((bfloat16_t*)do_shared)[(i_209 * 4096)])), (i_209 * 64), left_2, (((int)blockIdx.x) & 15), batch_idx);
                  }
                }
                bar_15[0].wait((i_s_5 & 1));
                #pragma unroll
                for (int i_210 = 0; i_210 < 2; ++i_210) {
                  bfloat16_t condval_12;
                  if (((((14 <= (((left_2 + ((int)threadIdx.x)) >> 5) + i_210)) && ((((i_210 * 32) + left_2) + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                    condval_12 = b[((((((((int64_t)i_210) * (int64_t)512) + (((int64_t)left_2) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)7168)];
                  } else {
                    condval_12 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  }
                  b_shared[(((i_210 * 32) + ((int)threadIdx.x)) - 448)] = ((float)condval_12);
                }
              }
              if (0 < num_iters) {
                bar_00[0].wait(((num_iters + 1) & 1));
              }
              if (0 < num_iters) {
                bar_02[0].arrive();
              }
            } else {
              for (int i_s_6 = 0; i_s_6 < num_iters; ++i_s_6) {
                int left_3 = ((((num_iters * 64) + seq_start_idx) - (i_s_6 * 64)) - 64);
                bar_05[0].arrive();
                bar_05[0].wait((i_s_6 & 1));
                bar_15[0].wait((i_s_6 & 1));
                if (i_s_6 == 0) {
                  if (0 <= batch_idx) {
                    #pragma unroll
                    for (int i_211 = 0; i_211 < 2; ++i_211) {
                      if ((((i_211 * 32) + left_3) + ((int)threadIdx.x)) < (seq_end_idx + 480)) {
                        if (15 <= (((left_3 + ((int)threadIdx.x)) >> 5) + i_211)) {
                          if ((((i_211 * 32) + left_3) + ((int)threadIdx.x)) < (num_tokens + 480)) {
                            if (batch_idx < batch_size) {
                              dg[((((((((int64_t)i_211) * (int64_t)512) + (((int64_t)left_3) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)7680)] = dg_shared[(((i_211 * 32) + ((int)threadIdx.x)) - 480)];
                            }
                          }
                        }
                      }
                    }
                  }
                  if (0 < ((seq_end_idx - seq_start_idx) & 63)) {
                    if (1 <= seq_end_idx) {
                      if (seq_end_idx <= num_tokens) {
                        if (0 <= batch_idx) {
                          if (batch_idx < batch_size) {
                            float condval_13;
                            if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                              condval_13 = dg[((((((int64_t)seq_end_idx) * (int64_t)16) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)16)];
                            } else {
                              condval_13 = 0x0p+0f/*0.000000e+00*/;
                            }
                            dg[((((((int64_t)seq_end_idx) * (int64_t)16) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)16)] = (condval_13 + dg_shared[63]);
                          }
                        }
                      }
                    }
                  }
                } else {
                  if (0 <= batch_idx) {
                    #pragma unroll
                    for (int i_212 = 0; i_212 < 2; ++i_212) {
                      if (15 <= (((left_3 + ((int)threadIdx.x)) >> 5) + i_212)) {
                        if ((((i_212 * 32) + left_3) + ((int)threadIdx.x)) < (num_tokens + 480)) {
                          if (batch_idx < batch_size) {
                            dg[((((((((int64_t)i_212) * (int64_t)512) + (((int64_t)left_3) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)7680)] = dg_shared[(((((int64_t)i_212) * (int64_t)32) + ((int64_t)((int)threadIdx.x))) - (int64_t)480)];
                          }
                        }
                      }
                    }
                  }
                }
                if (i_s_6 == 0) {
                  if (0 <= batch_idx) {
                    #pragma unroll
                    for (int i_213 = 0; i_213 < 2; ++i_213) {
                      if ((((i_213 * 32) + left_3) + ((int)threadIdx.x)) < (seq_end_idx + 480)) {
                        if (15 <= (((left_3 + ((int)threadIdx.x)) >> 5) + i_213)) {
                          if ((((i_213 * 32) + left_3) + ((int)threadIdx.x)) < (num_tokens + 480)) {
                            if (batch_idx < batch_size) {
                              db[((((((((int64_t)i_213) * (int64_t)512) + (((int64_t)left_3) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)7680)] = ((bfloat16_t)db_shared[(((i_213 * 32) + ((int)threadIdx.x)) - 480)]);
                            }
                          }
                        }
                      }
                    }
                  }
                } else {
                  if (0 <= batch_idx) {
                    #pragma unroll
                    for (int i_214 = 0; i_214 < 2; ++i_214) {
                      if (15 <= (((left_3 + ((int)threadIdx.x)) >> 5) + i_214)) {
                        if ((((i_214 * 32) + left_3) + ((int)threadIdx.x)) < (num_tokens + 480)) {
                          if (batch_idx < batch_size) {
                            db[((((((((int64_t)i_214) * (int64_t)512) + (((int64_t)left_3) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)7680)] = ((bfloat16_t)db_shared[(((((int64_t)i_214) * (int64_t)32) + ((int64_t)((int)threadIdx.x))) - (int64_t)480)]);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

