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

extern "C" __global__ void tilelang_fused_chunk_gdr_fwd_kernel_kernel(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const bfloat16_t* __restrict__ b, const float* __restrict__ g, const float* __restrict__ h0, __grid_constant__ const CUtensorMap h_desc, float* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ o, __grid_constant__ const CUtensorMap o_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens, int raw_batch_size);
extern "C" __global__ void __launch_bounds__(512, 1) tilelang_fused_chunk_gdr_fwd_kernel_kernel(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const bfloat16_t* __restrict__ b, const float* __restrict__ g, const float* __restrict__ h0, __grid_constant__ const CUtensorMap h_desc, float* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ o, __grid_constant__ const CUtensorMap o_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens, int raw_batch_size) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* h_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* q_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* v_shared = ((void*)((char*)buf_dyn_shmem + 98304));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 131072));
  void* o_shared = ((void*)((char*)buf_dyn_shmem + 147456));
  void* vd_shared = ((void*)((char*)buf_dyn_shmem + 163840));
  void* vn_shared = ((void*)((char*)buf_dyn_shmem + 180224));
  void* p_shared = ((void*)((char*)buf_dyn_shmem + 196608));
  __shared__ __align__(16) uint64_t data_is_ready_mem[2];
  auto data_is_ready = reinterpret_cast<Barrier*>(data_is_ready_mem);
  __shared__ __align__(16) uint64_t data_is_free_mem[2];
  auto data_is_free = reinterpret_cast<Barrier*>(data_is_free_mem);
  __shared__ __align__(16) uint64_t bar_o_mem[1];
  auto bar_o = reinterpret_cast<Barrier*>(bar_o_mem);
  __shared__ __align__(16) uint64_t bar_0_mem[1];
  auto bar_0 = reinterpret_cast<Barrier*>(bar_0_mem);
  __shared__ __align__(16) uint64_t bar_1_mem[1];
  auto bar_1 = reinterpret_cast<Barrier*>(bar_1_mem);
  __shared__ __align__(16) uint64_t _bar_2_mem[1];
  auto _bar_2 = reinterpret_cast<Barrier*>(_bar_2_mem);
  __shared__ __align__(16) uint64_t bar_3_mem[1];
  auto bar_3 = reinterpret_cast<Barrier*>(bar_3_mem);
  __shared__ __align__(16) uint64_t bar_4_mem[1];
  auto bar_4 = reinterpret_cast<Barrier*>(bar_4_mem);
  __shared__ __align__(16) uint64_t bar_5_mem[1];
  auto bar_5 = reinterpret_cast<Barrier*>(bar_5_mem);
  int batch_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int chunk_start_idx = 0;
  int raw_batch_idx = 0;
  int raw_seq_end_idx = 0;
  signed char need_store_final_state = (signed char)0;
  int num_iters = 0;
  int num_unmasked_iters = 0;
  float h_fragment[128];
  __shared__ __align__(16) float g_exp_shared[64];
  __shared__ __align__(16) float g_shared[128];
  __shared__ __align__(16) float b_shared[128];
  int seq_split_idx = 0;
  int chunk_split_idx = 0;
  float g_last_local[1];
  __shared__ __align__(16) float g_rev_exp_shared[64];
  float u_fragment[64];
  float v_fragment[64];
  float p_fragment[32];
  float g_fragment[32];
  float a_fragment[32];
  float o_fragment[64];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(q_desc);
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(v_desc);
    tl::prefetch_tma_descriptor(a_desc);
    tl::prefetch_tma_descriptor(o_desc);
    tl::prefetch_tma_descriptor(h_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    data_is_ready[0].init(96);
    data_is_ready[1].init(96);
    data_is_free[0].init(384);
    data_is_free[1].init(384);
    bar_o[0].init(128);
    bar_0[0].init(416);
    bar_1[0].init(256);
    _bar_2[0].init(128);
    bar_3[0].init(128);
    bar_4[0].init(128);
    bar_5[0].init(416);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = (((int)blockIdx.x) >> 4);
  seq_start_idx = 0;
  seq_end_idx = num_tokens;
  chunk_start_idx = 0;
  raw_batch_idx = (((int)blockIdx.x) >> 4);
  raw_seq_end_idx = seq_end_idx;
  need_store_final_state = ((signed char)((bool)1 & (raw_seq_end_idx == seq_end_idx)));
  num_iters = (((seq_end_idx + 63) - seq_start_idx) >> 6);
  num_unmasked_iters = ((seq_end_idx - seq_start_idx) >> 6);
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<128>();
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      *(float2*)(h_fragment + (i * 2)) = *(float2*)(h0 + ((((((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i) >> (int64_t)4) * (int64_t)2048)) + ((((int64_t)i) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + (((((int64_t)i) & (int64_t)15) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)));
    }
    for (int i_s = 0; i_s < num_iters; ++i_s) {
      data_is_ready[(i_s & 1)].wait(((i_s & 3) >> 1));
      bar_0[0].arrive();
      bar_0[0].wait((i_s & 1));
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)h_shared)[(((((((((int)threadIdx.x) >> 5) * 4096) + ((i_1 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)h_fragment[(i_1 * 8)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 1)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 2)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 3)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 4)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 5)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 6)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 7)])));
      }
      bar_1[0].arrive();
      bar_1[0].wait((i_s & 1));
      g_last_local[0] = g_exp_shared[63];
      #pragma unroll
      for (int i_2 = 0; i_2 < 32; ++i_2) {
        float4 __1;
          float4 v_ = *(float4*)(h_fragment + (i_2 * 4));
          float4 v__1 = make_float4(g_last_local[0], g_last_local[0], g_last_local[0], g_last_local[0]);
          *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
          *(float2*)(&(__1.z)) = tl::mul2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
        *(float4*)(h_fragment + (i_2 * 4)) = __1;
      }
      bar_5[0].arrive();
      bar_5[0].wait((i_s & 1));
      {
        bfloat16_t A_local[32];
        bfloat16_t B_local[32];
        for (int ki = 0; ki < 4; ++ki) {
          #pragma unroll
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)k_shared)[(((((((((i_s & 1) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (ki * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_3 * 8)])));
          }
          #pragma unroll
          for (int i_4 = 0; i_4 < 4; ++i_4) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)vn_shared)[(((((((((int)threadIdx.x) >> 6) * 4096) + (ki * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
          }
          for (int i_5 = 0; i_5 < 4; ++i_5) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + ((i_5 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + (((i_5 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      data_is_free[(i_s & 1)].arrive();
    }
    if ((bool)need_store_final_state) {
      if (0 <= raw_batch_idx) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 64; ++i_6) {
          if (raw_batch_idx < raw_batch_size) {
            *(float2*)(ht + (((((((((((int64_t)raw_batch_idx) * (int64_t)262144) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_6) >> (int64_t)4) * (int64_t)2048)) + ((((int64_t)i_6) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + (((((int64_t)i_6) & (int64_t)15) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(h_fragment + (i_6 * 2));
          }
        }
      }
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<128>();
      for (int i_s_1 = 0; i_s_1 < num_iters; ++i_s_1) {
        data_is_ready[(i_s_1 & 1)].wait(((i_s_1 & 3) >> 1));
        bar_0[0].arrive();
        bar_0[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial(3, 128);
        if (((int)threadIdx.x) < 192) {
          g_exp_shared[(((int)threadIdx.x) - 128)] = exp2f((g_shared[((((i_s_1 & 1) * 64) + ((int)threadIdx.x)) - 128)] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          float condval;
          if (((((i_s_1 * 64) + seq_start_idx) + ((int)threadIdx.x)) < (seq_end_idx + 128))) {
            condval = exp2f(((g_shared[(((i_s_1 & 1) * 64) + 63)] - g_shared[((((i_s_1 & 1) * 64) + ((int)threadIdx.x)) - 128)]) * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          } else {
            condval = 0x0p+0f/*0.000000e+00*/;
          }
          g_rev_exp_shared[(((int)threadIdx.x) - 128)] = condval;
        }
        bar_1[0].arrive();
        bar_1[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_1[16];
          bfloat16_t B_local_1[32];
          #pragma unroll
          for (int i_7 = 0; i_7 < 16; ++i_7) {
            float broadcast_var = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(u_fragment + (i_7 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
          }
          for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
            #pragma unroll
            for (int i_8 = 0; i_8 < 2; ++i_8) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[(((((((((i_s_1 & 1) * 8192) + ((ki_1 >> 2) * 4096)) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_8 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_1[(i_8 * 8)])));
            }
            #pragma unroll
            for (int i_9 = 0; i_9 < 4; ++i_9) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)h_shared)[((((((((((int)threadIdx.x) & 127) >> 6) * 8192) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_9 * 8)])));
            }
            for (int i_10 = 0; i_10 < 2; ++i_10) {
              for (int j_1 = 0; j_1 < 4; ++j_1) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + ((i_10 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_10 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + (((i_10 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_10 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
              }
            }
          }
        }
        tl::__sync_thread_partial(3, 128);
        #pragma unroll
        for (int i_11 = 0; i_11 < 32; ++i_11) {
          float2 __2;
            float2 v__2 = *(float2*)(u_fragment + (i_11 * 2));
            float2 v__3 = make_float2((g_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_11 >> 4) * 16)) + ((i_11 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/), (g_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_11 >> 4) * 16)) + ((i_11 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/));
            *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
          *(float2*)(u_fragment + (i_11 * 2)) = __2;
        }
        #pragma unroll
        for (int i_12 = 0; i_12 < 32; ++i_12) {
          bfloat16_t v_shared_local_cast[2];
          *(uint1*)(v_shared_local_cast + 0) = *(uint1*)(((bfloat16_t*)v_shared) + (((((((((((i_s_1 & 1) * 8192) + ((((int)threadIdx.x) >> 5) * 2048)) + ((i_12 >> 4) * 1024)) + ((i_12 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_12 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_12 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_12 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
          float2 __3;
            float2 v__4 = *(float2*)(u_fragment + (i_12 * 2));
            float2 __4;
            uint1 v__5 = *(uint1*)(v_shared_local_cast + 0);
            ((float2*)(&__4))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__5))[0]);
            *(float2*)(&(__3.x)) = tl::add2(*(float2*)(&(v__4.x)), *(float2*)(&(__4.x)));
          *(float2*)(u_fragment + (i_12 * 2)) = __3;
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_13 = 0; i_13 < 32; ++i_13) {
          bfloat16_t v_shared_local_cast_1[2];
          uint1 __5;
          float2 v__6 = *(float2*)(u_fragment + (i_13 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__6))[0]);
          *(uint1*)(v_shared_local_cast_1 + 0) = __5;
          *(uint1*)(((bfloat16_t*)v_shared) + (((((((((((i_s_1 & 1) * 8192) + ((((int)threadIdx.x) >> 5) * 2048)) + ((i_13 >> 4) * 1024)) + ((i_13 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_13 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_13 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192)) = *(uint1*)(v_shared_local_cast_1 + 0);
        }
        bar_3[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_2[16];
          bfloat16_t B_local_2[32];
          #pragma unroll
          for (int i_14 = 0; i_14 < 16; ++i_14) {
            float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(v_fragment + (i_14 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
          }
          tl::__sync_thread_partial(4, 128);
          for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
            #pragma unroll
            for (int i_15 = 0; i_15 < 2; ++i_15) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_shared)[((((((((i_s_1 & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_15 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_2[(i_15 * 8)])));
            }
            #pragma unroll
            for (int i_16 = 0; i_16 < 4; ++i_16) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)v_shared)[((((((((i_s_1 & 1) * 8192) + (((((int)threadIdx.x) & 127) >> 6) * 4096)) + (ki_2 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_16 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[(i_16 * 8)])));
            }
            for (int i_17 = 0; i_17 < 2; ++i_17) {
              for (int j_2 = 0; j_2 < 4; ++j_2) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(v_fragment + ((i_17 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_17 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(v_fragment + (((i_17 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_17 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
              }
            }
          }
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_18 = 0; i_18 < 8; ++i_18) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)vd_shared)[((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_18 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_18 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_18 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 8192)])), __pack_half2(((bfloat16_t)v_fragment[(i_18 * 8)]), ((bfloat16_t)v_fragment[((i_18 * 8) + 1)])), __pack_half2(((bfloat16_t)v_fragment[((i_18 * 8) + 2)]), ((bfloat16_t)v_fragment[((i_18 * 8) + 3)])), __pack_half2(((bfloat16_t)v_fragment[((i_18 * 8) + 4)]), ((bfloat16_t)v_fragment[((i_18 * 8) + 5)])), __pack_half2(((bfloat16_t)v_fragment[((i_18 * 8) + 6)]), ((bfloat16_t)v_fragment[((i_18 * 8) + 7)])));
        }
        bar_4[0].arrive();
        #pragma unroll
        for (int i_19 = 0; i_19 < 32; ++i_19) {
          float2 __6;
            float2 v__7 = *(float2*)(v_fragment + (i_19 * 2));
            float2 v__8 = make_float2(g_rev_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_19 >> 4) * 16)) + ((i_19 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_rev_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_19 >> 4) * 16)) + ((i_19 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)));
          *(float2*)(v_fragment + (i_19 * 2)) = __6;
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_20 = 0; i_20 < 8; ++i_20) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)vn_shared)[((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_20 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_20 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_20 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 8192)])), __pack_half2(((bfloat16_t)v_fragment[(i_20 * 8)]), ((bfloat16_t)v_fragment[((i_20 * 8) + 1)])), __pack_half2(((bfloat16_t)v_fragment[((i_20 * 8) + 2)]), ((bfloat16_t)v_fragment[((i_20 * 8) + 3)])), __pack_half2(((bfloat16_t)v_fragment[((i_20 * 8) + 4)]), ((bfloat16_t)v_fragment[((i_20 * 8) + 5)])), __pack_half2(((bfloat16_t)v_fragment[((i_20 * 8) + 6)]), ((bfloat16_t)v_fragment[((i_20 * 8) + 7)])));
        }
        bar_5[0].arrive();
        bar_5[0].wait((i_s_1 & 1));
        data_is_free[(i_s_1 & 1)].arrive();
      }
    } else {
      if (((int)threadIdx.x) < 384) {
        tl::warpgroup_reg_alloc<128>();
        for (int i_s_2 = 0; i_s_2 < num_iters; ++i_s_2) {
          data_is_ready[(i_s_2 & 1)].wait(((i_s_2 & 3) >> 1));
          bar_0[0].arrive();
          bar_0[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_3[16];
            bfloat16_t B_local_3[16];
            #pragma unroll
            for (int i_21 = 0; i_21 < 8; ++i_21) {
              float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(p_fragment + (i_21 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
            }
            for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
              #pragma unroll
              for (int i_22 = 0; i_22 < 2; ++i_22) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)q_shared)[(((((((((i_s_2 & 1) * 8192) + ((ki_3 >> 2) * 4096)) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_22 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_3[(i_22 * 8)])));
              }
              #pragma unroll
              for (int i_23 = 0; i_23 < 2; ++i_23) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[((((((((((i_s_2 & 1) * 8192) + ((ki_3 >> 2) * 4096)) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_23 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[(i_23 * 8)])));
              }
              for (int i_24 = 0; i_24 < 2; ++i_24) {
                for (int j_3 = 0; j_3 < 2; ++j_3) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + ((i_24 * 16) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_24 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + (((i_24 * 16) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_24 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
                }
              }
            }
          }
          #pragma unroll
          for (int i_25 = 0; i_25 < 16; ++i_25) {
            float2 __7;
              float2 v__9 = make_float2(g_shared[((((((i_s_2 & 1) * 64) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + ((i_25 >> 3) * 16)) + ((i_25 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_shared[((((((i_s_2 & 1) * 64) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + ((i_25 >> 3) * 16)) + ((i_25 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
              float2 v__10 = *(float2*)(g_shared + ((((((i_s_2 & 1) * 64) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_25 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 128));
              *(float2*)(&(__7.x)) = tl::sub2(*(float2*)(&(v__9.x)), *(float2*)(&(v__10.x)));
            *(float2*)(g_fragment + (i_25 * 2)) = __7;
          }
          #pragma unroll
          for (int i_26 = 0; i_26 < 32; ++i_26) {
            if ((((((((int)threadIdx.x) >> 6) * 32) + (((i_26 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_26 & 1)) <= (((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_26 >> 4) * 16)) + (((i_26 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 128)) {
              g_fragment[i_26] = exp2f((g_fragment[i_26] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
            } else {
              g_fragment[i_26] = 0x0p+0f/*0.000000e+00*/;
            }
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_27 = 0; i_27 < 16; ++i_27) {
            bfloat16_t a_shared_local_cast_2[2];
            *(uint1*)(a_shared_local_cast_2 + 0) = *(uint1*)(((bfloat16_t*)a_shared) + ((((((((((i_s_2 & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + ((i_27 >> 3) * 1024)) + ((i_27 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_27 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_27 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
            float2 __8;
            uint1 v__11 = *(uint1*)(a_shared_local_cast_2 + 0);
            ((float2*)(&__8))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__11))[0]);
            *(float2*)(a_fragment + (i_27 * 2)) = __8;
          }
          #pragma unroll
          for (int i_28 = 0; i_28 < 8; ++i_28) {
            float4 __9;
              float4 v__12 = *(float4*)(a_fragment + (i_28 * 4));
              float4 v__13 = *(float4*)(g_fragment + (i_28 * 4));
              *(float2*)(&(__9.x)) = tl::mul2(*(float2*)(&(v__12.x)), *(float2*)(&(v__13.x)));
              *(float2*)(&(__9.z)) = tl::mul2(*(float2*)(&(v__12.z)), *(float2*)(&(v__13.z)));
            *(float4*)(a_fragment + (i_28 * 4)) = __9;
          }
          #pragma unroll
          for (int i_29 = 0; i_29 < 16; ++i_29) {
            float2 __10;
              float2 v__14 = *(float2*)(a_fragment + (i_29 * 2));
              float2 v__15 = *(float2*)(b_shared + ((((((i_s_2 & 1) * 64) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_29 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 128));
              *(float2*)(&(__10.x)) = tl::mul2(*(float2*)(&(v__14.x)), *(float2*)(&(v__15.x)));
            *(float2*)(a_fragment + (i_29 * 2)) = __10;
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_30 = 0; i_30 < 16; ++i_30) {
            bfloat16_t a_shared_local_cast_3[2];
            uint1 __11;
            float2 v__16 = *(float2*)(a_fragment + (i_30 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__11))[0] = __float22bfloat162_rn(((float2*)(&v__16))[0]);
            *(uint1*)(a_shared_local_cast_3 + 0) = __11;
            *(uint1*)(((bfloat16_t*)a_shared) + ((((((((((i_s_2 & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + ((i_30 >> 3) * 1024)) + ((i_30 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_30 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_30 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(a_shared_local_cast_3 + 0);
          }
          bar_1[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_4[16];
            bfloat16_t B_local_4[32];
            #pragma unroll
            for (int i_31 = 0; i_31 < 16; ++i_31) {
              float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(o_fragment + (i_31 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
            }
            tl::__sync_thread_partial(5, 128);
            for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
              #pragma unroll
              for (int i_32 = 0; i_32 < 2; ++i_32) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)q_shared)[(((((((((i_s_2 & 1) * 8192) + ((ki_4 >> 2) * 4096)) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_32 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_4[(i_32 * 8)])));
              }
              #pragma unroll
              for (int i_33 = 0; i_33 < 4; ++i_33) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)h_shared)[((((((((((int)threadIdx.x) & 127) >> 6) * 8192) + (ki_4 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_33 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_33 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_4[(i_33 * 8)])));
              }
              for (int i_34 = 0; i_34 < 2; ++i_34) {
                for (int j_4 = 0; j_4 < 4; ++j_4) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + ((i_34 * 32) + (j_4 * 8))), reinterpret_cast<const unsigned*>(A_local_4 + (i_34 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + (j_4 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + (((i_34 * 32) + (j_4 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + (i_34 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + ((j_4 * 8) + 4)));
                }
              }
            }
          }
          #pragma unroll
          for (int i_35 = 0; i_35 < 8; ++i_35) {
            float broadcast_var_4 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
            float4 __12;
              float4 v__17 = *(float4*)(p_fragment + (i_35 * 4));
              float4 __13;
                float4 v__18 = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
                float4 v__19 = *(float4*)(g_fragment + (i_35 * 4));
                *(float2*)(&(__13.x)) = tl::mul2(*(float2*)(&(v__18.x)), *(float2*)(&(v__19.x)));
                *(float2*)(&(__13.z)) = tl::mul2(*(float2*)(&(v__18.z)), *(float2*)(&(v__19.z)));
              *(float2*)(&(__12.x)) = tl::mul2(*(float2*)(&(v__17.x)), *(float2*)(&(__13.x)));
              *(float2*)(&(__12.z)) = tl::mul2(*(float2*)(&(v__17.z)), *(float2*)(&(__13.z)));
            *(float4*)(p_fragment + (i_35 * 4)) = __12;
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_36 = 0; i_36 < 4; ++i_36) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)p_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_36 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_36 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)p_fragment[(i_36 * 8)]), ((bfloat16_t)p_fragment[((i_36 * 8) + 1)])), __pack_half2(((bfloat16_t)p_fragment[((i_36 * 8) + 2)]), ((bfloat16_t)p_fragment[((i_36 * 8) + 3)])), __pack_half2(((bfloat16_t)p_fragment[((i_36 * 8) + 4)]), ((bfloat16_t)p_fragment[((i_36 * 8) + 5)])), __pack_half2(((bfloat16_t)p_fragment[((i_36 * 8) + 6)]), ((bfloat16_t)p_fragment[((i_36 * 8) + 7)])));
          }
          bar_3[0].arrive();
          #pragma unroll
          for (int i_37 = 0; i_37 < 32; ++i_37) {
            float2 __14;
              float2 v__20 = *(float2*)(o_fragment + (i_37 * 2));
              float2 v__21 = make_float2((0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_37 >> 4) * 16)) + ((i_37 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]), (0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_37 >> 4) * 16)) + ((i_37 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
              *(float2*)(&(__14.x)) = tl::mul2(*(float2*)(&(v__20.x)), *(float2*)(&(v__21.x)));
            *(float2*)(o_fragment + (i_37 * 2)) = __14;
          }
          bar_4[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_5[16];
            bfloat16_t B_local_5[32];
            tl::__sync_thread_partial(5, 128);
            for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
              #pragma unroll
              for (int i_38 = 0; i_38 < 2; ++i_38) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)p_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_38 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_5[(i_38 * 8)])));
              }
              #pragma unroll
              for (int i_39 = 0; i_39 < 4; ++i_39) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)vd_shared)[((((((((((int)threadIdx.x) & 127) >> 6) * 4096) + (ki_5 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_39 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_39 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_5[(i_39 * 8)])));
              }
              for (int i_40 = 0; i_40 < 2; ++i_40) {
                for (int j_5 = 0; j_5 < 4; ++j_5) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + ((i_40 * 32) + (j_5 * 8))), reinterpret_cast<const unsigned*>(A_local_5 + (i_40 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + (j_5 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + (((i_40 * 32) + (j_5 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + (i_40 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + ((j_5 * 8) + 4)));
                }
              }
            }
          }
          bar_5[0].arrive();
          bar_5[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_41 = 0; i_41 < 8; ++i_41) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)o_shared)[((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_41 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_41 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_41 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 16384)])), __pack_half2(((bfloat16_t)o_fragment[(i_41 * 8)]), ((bfloat16_t)o_fragment[((i_41 * 8) + 1)])), __pack_half2(((bfloat16_t)o_fragment[((i_41 * 8) + 2)]), ((bfloat16_t)o_fragment[((i_41 * 8) + 3)])), __pack_half2(((bfloat16_t)o_fragment[((i_41 * 8) + 4)]), ((bfloat16_t)o_fragment[((i_41 * 8) + 5)])), __pack_half2(((bfloat16_t)o_fragment[((i_41 * 8) + 6)]), ((bfloat16_t)o_fragment[((i_41 * 8) + 7)])));
          }
          data_is_free[(i_s_2 & 1)].arrive();
        }
        bar_o[0].arrive();
      } else {
        tl::warpgroup_reg_dealloc<32>();
        if (((int)threadIdx.x) < 416) {
          tl::__sync_thread_partial(6, 32);
          for (int i_s_3 = 0; i_s_3 < num_iters; ++i_s_3) {
            data_is_free[(i_s_3 & 1)].wait((((i_s_3 >> 1) + 1) & 1));
            int left = ((i_s_3 * 64) + seq_start_idx);
            if ((left + 64) <= seq_end_idx) {
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_3 & 1)].expect_transaction(16384);
                tl::fence_proxy_async();
                #pragma unroll
                for (int i_42 = 0; i_42 < 2; ++i_42) {
                  tl::tma_load(q_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)q_shared)[(((i_s_3 & 1) * 8192) + (i_42 * 4096))])), (i_42 * 64), left, (((int)blockIdx.x) & 15), batch_idx);
                }
              }
            } else {
              tl::__sync_thread_partial(6, 32);
              #pragma unroll
              for (int i_43 = 0; i_43 < 32; ++i_43) {
                if ((((i_43 * 2) + (((int)threadIdx.x) >> 4)) + left) < (seq_end_idx + 24)) {
                  bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  uint4 condval_1;
                  if (((((12 <= ((((((int)threadIdx.x) >> 4) + left) >> 1) + i_43)) && ((((i_43 * 2) + (((int)threadIdx.x) >> 4)) + left) < (num_tokens + 24))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                    condval_1 = *(uint4*)(q + (((((((((int64_t)i_43) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)left) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)49152));
                  } else {
                    condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
                  }
                  *(uint4*)(((bfloat16_t*)q_shared) + (((((((((i_s_3 & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + ((((i_43 * 256) + (((int)threadIdx.x) * 8)) >> 10) * 512)) + ((((i_43 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_43 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_43 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 1536)) = condval_1;
                } else {
                  bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  *(uint4*)(((bfloat16_t*)q_shared) + (((((((((i_s_3 & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + ((((i_43 * 256) + (((int)threadIdx.x) * 8)) >> 10) * 512)) + ((((i_43 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_43 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_43 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 1536)) = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
                }
              }
            }
            tl::__sync_thread_partial(6, 32);
            if ((left + 64) <= seq_end_idx) {
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_3 & 1)].expect_transaction(16384);
                tl::fence_proxy_async();
                #pragma unroll
                for (int i_44 = 0; i_44 < 2; ++i_44) {
                  tl::tma_load(k_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)k_shared)[(((i_s_3 & 1) * 8192) + (i_44 * 4096))])), (i_44 * 64), left, (((int)blockIdx.x) & 15), batch_idx);
                }
              }
            } else {
              tl::__sync_thread_partial(6, 32);
              #pragma unroll
              for (int i_45 = 0; i_45 < 32; ++i_45) {
                if ((((i_45 * 2) + (((int)threadIdx.x) >> 4)) + left) < (seq_end_idx + 24)) {
                  bfloat16_t broadcast_var_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  uint4 condval_2;
                  if (((((12 <= ((((((int)threadIdx.x) >> 4) + left) >> 1) + i_45)) && ((((i_45 * 2) + (((int)threadIdx.x) >> 4)) + left) < (num_tokens + 24))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                    condval_2 = *(uint4*)(k + (((((((((int64_t)i_45) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)left) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)49152));
                  } else {
                    condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7));
                  }
                  *(uint4*)(((bfloat16_t*)k_shared) + (((((((((i_s_3 & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + ((((i_45 * 256) + (((int)threadIdx.x) * 8)) >> 10) * 512)) + ((((i_45 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_45 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_45 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 1536)) = condval_2;
                } else {
                  bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  *(uint4*)(((bfloat16_t*)k_shared) + (((((((((i_s_3 & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + ((((i_45 * 256) + (((int)threadIdx.x) * 8)) >> 10) * 512)) + ((((i_45 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_45 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_45 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 1536)) = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
                }
              }
            }
            data_is_ready[(i_s_3 & 1)].arrive();
          }
        } else {
          if (((int)threadIdx.x) < 448) {
            tl::__sync_thread_partial(7, 32);
            for (int i_s_4 = 0; i_s_4 < num_iters; ++i_s_4) {
              data_is_free[(i_s_4 & 1)].wait((((i_s_4 >> 1) + 1) & 1));
              int left_1 = ((i_s_4 * 64) + seq_start_idx);
              if ((left_1 + 64) <= seq_end_idx) {
                if (tl::tl_shuffle_elect<32>()) {
                  data_is_ready[(i_s_4 & 1)].expect_transaction(16384);
                  tl::fence_proxy_async();
                  #pragma unroll
                  for (int i_46 = 0; i_46 < 2; ++i_46) {
                    tl::tma_load(v_desc, data_is_ready[(i_s_4 & 1)], (&(((bfloat16_t*)v_shared)[(((i_s_4 & 1) * 8192) + (i_46 * 4096))])), (i_46 * 64), left_1, (((int)blockIdx.x) & 15), batch_idx);
                  }
                }
              } else {
                tl::__sync_thread_partial(7, 32);
                #pragma unroll
                for (int i_47 = 0; i_47 < 32; ++i_47) {
                  if ((((i_47 * 2) + (((int)threadIdx.x) >> 4)) + left_1) < (seq_end_idx + 26)) {
                    bfloat16_t broadcast_var_9 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    uint4 condval_3;
                    if (((((13 <= ((((((int)threadIdx.x) >> 4) + left_1) >> 1) + i_47)) && ((((i_47 * 2) + (((int)threadIdx.x) >> 4)) + left_1) < (num_tokens + 26))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                      condval_3 = *(uint4*)(v + (((((((((int64_t)i_47) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)left_1) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)53248));
                    } else {
                      condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9));
                    }
                    *(uint4*)(((bfloat16_t*)v_shared) + ((((((((i_s_4 & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + ((i_47 >> 2) * 512)) + (((((i_47 * 2) + (((int)threadIdx.x) >> 4)) + 6) & 7) * 64)) + ((((((((i_47 * 256) + (((int)threadIdx.x) * 8)) + 768) & 1023) >> 9) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_47 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_3;
                  } else {
                    bfloat16_t broadcast_var_10 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    *(uint4*)(((bfloat16_t*)v_shared) + ((((((((i_s_4 & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + ((i_47 >> 2) * 512)) + (((((i_47 * 2) + (((int)threadIdx.x) >> 4)) + 6) & 7) * 64)) + ((((((((i_47 * 256) + (((int)threadIdx.x) * 8)) + 768) & 1023) >> 9) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_47 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10));
                  }
                }
              }
              if ((left_1 + 64) <= seq_end_idx) {
                #pragma unroll
                for (int i_48 = 0; i_48 < 2; ++i_48) {
                  bfloat16_t condval_4;
                  if (((((13 <= (((left_1 + ((int)threadIdx.x)) >> 5) + i_48)) && ((((i_48 * 32) + left_1) + ((int)threadIdx.x)) < (num_tokens + 416))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                    condval_4 = b[((((((((int64_t)i_48) * (int64_t)512) + (((int64_t)left_1) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)6656)];
                  } else {
                    condval_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  }
                  b_shared[(((((i_s_4 & 1) * 64) + (i_48 * 32)) + ((int)threadIdx.x)) - 416)] = ((float)condval_4);
                }
              } else {
                #pragma unroll
                for (int i_49 = 0; i_49 < 2; ++i_49) {
                  if ((((i_49 * 32) + left_1) + ((int)threadIdx.x)) < (seq_end_idx + 416)) {
                    bfloat16_t condval_5;
                    if (((((13 <= (((left_1 + ((int)threadIdx.x)) >> 5) + i_49)) && ((((i_49 * 32) + left_1) + ((int)threadIdx.x)) < (num_tokens + 416))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                      condval_5 = b[((((((((int64_t)i_49) * (int64_t)512) + (((int64_t)left_1) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)6656)];
                    } else {
                      condval_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    }
                    b_shared[(((((i_s_4 & 1) * 64) + (i_49 * 32)) + ((int)threadIdx.x)) - 416)] = ((float)condval_5);
                  } else {
                    b_shared[(((((i_s_4 & 1) * 64) + (i_49 * 32)) + ((int)threadIdx.x)) - 416)] = 0x0p+0f/*0.000000e+00*/;
                  }
                }
              }
              data_is_ready[(i_s_4 & 1)].arrive();
            }
          } else {
            if (((int)threadIdx.x) < 480) {
              tl::__sync_thread_partial(8, 32);
              for (int i_s_5 = 0; i_s_5 < num_iters; ++i_s_5) {
                data_is_free[(i_s_5 & 1)].wait((((i_s_5 >> 1) + 1) & 1));
                int left_2 = ((i_s_5 * 64) + seq_start_idx);
                if ((left_2 + 64) <= seq_end_idx) {
                  if (tl::tl_shuffle_elect<32>()) {
                    data_is_ready[(i_s_5 & 1)].expect_transaction(8192);
                    tl::fence_proxy_async();
                    tl::tma_load(a_desc, data_is_ready[(i_s_5 & 1)], (&(((bfloat16_t*)a_shared)[((i_s_5 & 1) * 4096)])), 0, left_2, (((int)blockIdx.x) & 15), batch_idx);
                  }
                } else {
                  tl::__sync_thread_partial(8, 32);
                  #pragma unroll
                  for (int i_50 = 0; i_50 < 16; ++i_50) {
                    if ((((i_50 * 4) + (((int)threadIdx.x) >> 3)) + left_2) < (seq_end_idx + 56)) {
                      bfloat16_t broadcast_var_11 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      uint4 condval_6;
                      if (((((14 <= ((((((int)threadIdx.x) >> 3) + left_2) >> 2) + i_50)) && ((((i_50 * 4) + (((int)threadIdx.x) >> 3)) + left_2) < (num_tokens + 56))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                        condval_6 = *(uint4*)(a + (((((((((int64_t)i_50) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)1024)) + (((int64_t)left_2) * (int64_t)1024)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)57344));
                      } else {
                        condval_6 = make_uint4(__pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11));
                      }
                      *(uint4*)(((bfloat16_t*)a_shared) + ((((((((i_s_5 & 1) * 4096) + ((((i_50 * 256) + (((int)threadIdx.x) * 8)) >> 9) * 512)) + ((((i_50 * 4) + (((int)threadIdx.x) >> 3)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_50 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 3584)) = condval_6;
                    } else {
                      bfloat16_t broadcast_var_12 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      *(uint4*)(((bfloat16_t*)a_shared) + ((((((((i_s_5 & 1) * 4096) + ((((i_50 * 256) + (((int)threadIdx.x) * 8)) >> 9) * 512)) + ((((i_50 * 4) + (((int)threadIdx.x) >> 3)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_50 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 3584)) = make_uint4(__pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12));
                    }
                  }
                }
                if ((left_2 + 64) <= seq_end_idx) {
                  #pragma unroll
                  for (int i_51 = 0; i_51 < 2; ++i_51) {
                    float condval_7;
                    if (((((14 <= (((left_2 + ((int)threadIdx.x)) >> 5) + i_51)) && ((((i_51 * 32) + left_2) + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                      condval_7 = g[((((((((int64_t)i_51) * (int64_t)512) + (((int64_t)left_2) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)7168)];
                    } else {
                      condval_7 = 0x0p+0f/*0.000000e+00*/;
                    }
                    g_shared[(((((i_s_5 & 1) * 64) + (i_51 * 32)) + ((int)threadIdx.x)) - 448)] = condval_7;
                  }
                } else {
                  #pragma unroll
                  for (int i_52 = 0; i_52 < 2; ++i_52) {
                    if ((((i_52 * 32) + left_2) + ((int)threadIdx.x)) < (seq_end_idx + 448)) {
                      float condval_8;
                      if (((((14 <= (((left_2 + ((int)threadIdx.x)) >> 5) + i_52)) && ((((i_52 * 32) + left_2) + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                        condval_8 = g[((((((((int64_t)i_52) * (int64_t)512) + (((int64_t)left_2) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)7168)];
                      } else {
                        condval_8 = 0x0p+0f/*0.000000e+00*/;
                      }
                      g_shared[(((((i_s_5 & 1) * 64) + (i_52 * 32)) + ((int)threadIdx.x)) - 448)] = condval_8;
                    } else {
                      float condval_9;
                      if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                        condval_9 = g[((((((int64_t)seq_end_idx) * (int64_t)16) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)16)];
                      } else {
                        condval_9 = 0x0p+0f/*0.000000e+00*/;
                      }
                      g_shared[(((((i_s_5 & 1) * 64) + (i_52 * 32)) + ((int)threadIdx.x)) - 448)] = condval_9;
                    }
                  }
                }
                data_is_ready[(i_s_5 & 1)].arrive();
              }
            } else {
              for (int i_s_6 = 0; i_s_6 < num_unmasked_iters; ++i_s_6) {
                int right = ((i_s_6 * 64) + seq_start_idx);
                bar_0[0].arrive();
                bar_0[0].wait((i_s_6 & 1));
                if (0 < i_s_6) {
                  if (tl::tl_shuffle_elect<32>()) {
                    #pragma unroll
                    for (int i_53 = 0; i_53 < 2; ++i_53) {
                      tl::tma_store(o_desc, (&(((bfloat16_t*)o_shared)[(i_53 * 4096)])), (i_53 * 64), (right - 64), (((int)blockIdx.x) & 15), batch_idx);
                    }
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0, true>();
                  }
                }
                bar_5[0].arrive();
                bar_1[0].wait((i_s_6 & 1));
                if (tl::tl_shuffle_elect<32>()) {
                  #pragma unroll
                  for (int i_54 = 0; i_54 < 2; ++i_54) {
                    tl::tma_store(h_desc, (&(((bfloat16_t*)h_shared)[(i_54 * 8192)])), (i_54 * 64), 0, (((int)blockIdx.x) & 15), (chunk_start_idx + i_s_6), batch_idx);
                  }
                  tl::tma_store_arrive();
                  tl::tma_store_wait<0, true>();
                }
              }
              if (num_unmasked_iters < num_iters) {
                seq_split_idx = ((num_unmasked_iters * 64) + seq_start_idx);
              }
              if (num_unmasked_iters < num_iters) {
                chunk_split_idx = (chunk_start_idx + num_unmasked_iters);
              }
              if (num_unmasked_iters < num_iters) {
                int right_1 = seq_split_idx;
                bar_0[0].arrive();
                bar_0[0].wait((num_unmasked_iters & 1));
                if (0 < num_unmasked_iters) {
                  if (tl::tl_shuffle_elect<32>()) {
                    #pragma unroll
                    for (int i_55 = 0; i_55 < 2; ++i_55) {
                      tl::tma_store(o_desc, (&(((bfloat16_t*)o_shared)[(i_55 * 4096)])), (i_55 * 64), (right_1 - 64), (((int)blockIdx.x) & 15), batch_idx);
                    }
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0, true>();
                  }
                }
                bar_5[0].arrive();
                bar_1[0].wait((num_unmasked_iters & 1));
                if (tl::tl_shuffle_elect<32>()) {
                  #pragma unroll
                  for (int i_56 = 0; i_56 < 2; ++i_56) {
                    tl::tma_store(h_desc, (&(((bfloat16_t*)h_shared)[(i_56 * 8192)])), (i_56 * 64), 0, (((int)blockIdx.x) & 15), chunk_split_idx, batch_idx);
                  }
                  tl::tma_store_arrive();
                  tl::tma_store_wait<0, true>();
                }
              }
              seq_split_idx = (((num_iters * 64) + seq_start_idx) - 64);
              bar_o[0].wait(0);
              if (0 < num_iters) {
                if (0 <= batch_idx) {
                  #pragma unroll
                  for (int i_57 = 0; i_57 < 32; ++i_57) {
                    if ((((i_57 * 2) + (((int)threadIdx.x) >> 4)) + seq_split_idx) < (seq_end_idx + 30)) {
                      if (15 <= ((((((int)threadIdx.x) >> 4) + seq_split_idx) >> 1) + i_57)) {
                        if ((((i_57 * 2) + (((int)threadIdx.x) >> 4)) + seq_split_idx) < (num_tokens + 30)) {
                          if (batch_idx < batch_size) {
                            *(uint4*)(o + (((((((((int64_t)i_57) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)seq_split_idx) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)61440)) = *(uint4*)(((bfloat16_t*)o_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + ((i_57 >> 2) * 512)) + (((((i_57 * 2) + (((int)threadIdx.x) >> 4)) + 2) & 7) * 64)) + ((((((((i_57 * 256) + (((int)threadIdx.x) * 8)) + 256) & 1023) >> 9) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_57 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
                          }
                        }
                      }
                    }
                  }
                }
              }
              if ((((int)blockIdx.x) >> 4) == (batch_size - 1)) {
                if (0 <= batch_idx) {
                  for (int i_pad = 0; i_pad < (num_tokens - seq_end_idx); ++i_pad) {
                    if (0 <= (seq_end_idx + i_pad)) {
                      if (batch_idx < batch_size) {
                        bfloat16_t broadcast_var_13 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                        *(uint2*)(o + ((((((((int64_t)seq_end_idx) * (int64_t)2048) + (((int64_t)i_pad) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) - (int64_t)1920)) = make_uint2(__pack_nv_bfloat162(broadcast_var_13, broadcast_var_13), __pack_nv_bfloat162(broadcast_var_13, broadcast_var_13));
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

