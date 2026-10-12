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

extern "C" __global__ void tilelang_prepare_h_kernel_kernel(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const bfloat16_t* __restrict__ b, const float* __restrict__ g, const float* __restrict__ h0, __grid_constant__ const CUtensorMap h_desc, float* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, float* __restrict__ mt, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens);
extern "C" __global__ void __launch_bounds__(512, 1) tilelang_prepare_h_kernel_kernel(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const bfloat16_t* __restrict__ b, const float* __restrict__ g, const float* __restrict__ h0, __grid_constant__ const CUtensorMap h_desc, float* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, float* __restrict__ mt, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* v_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  void* h_shared = ((void*)((char*)buf_dyn_shmem + 98304));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 131072));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 155648));
  void* y_shared = ((void*)((char*)buf_dyn_shmem + 172032));
  void* z_shared_L = ((void*)((char*)buf_dyn_shmem + 188416));
  void* z_shared_R = ((void*)((char*)buf_dyn_shmem + 196608));
  __shared__ __align__(16) uint64_t data_is_ready_mem[3];
  auto data_is_ready = reinterpret_cast<Barrier*>(data_is_ready_mem);
  __shared__ __align__(16) uint64_t data_is_free_mem[3];
  auto data_is_free = reinterpret_cast<Barrier*>(data_is_free_mem);
  __shared__ __align__(16) uint64_t bar_0_mem[1];
  auto bar_0 = reinterpret_cast<Barrier*>(bar_0_mem);
  __shared__ __align__(16) uint64_t bar_1_mem[1];
  auto bar_1 = reinterpret_cast<Barrier*>(bar_1_mem);
  __shared__ __align__(16) uint64_t bar_2_mem[1];
  auto bar_2 = reinterpret_cast<Barrier*>(bar_2_mem);
  __shared__ __align__(16) uint64_t bar_3_mem[1];
  auto bar_3 = reinterpret_cast<Barrier*>(bar_3_mem);
  int batch_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int chunk_start_idx = 0;
  int num_iters = 0;
  signed char calc_mt = (signed char)0;
  float h_fragment[128];
  __shared__ __align__(16) float g_shared[192];
  float m_fragment_R[64];
  float g_prod_X[1];
  __shared__ __align__(16) float b_shared[192];
  float g_last_local_X[1];
  float m_fragment_L[64];
  float g_prod_Y[1];
  float g_last_local_Y[1];
  float g_last_local_S[1];
  float x_fragment[64];
  float z_fragment_R[32];
  __shared__ __align__(16) float g_rev_exp_shared[64];
  float y_fragment[64];
  float z_fragment_L[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(v_desc);
    tl::prefetch_tma_descriptor(a_desc);
    tl::prefetch_tma_descriptor(h_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    data_is_ready[0].init(96);
    data_is_ready[1].init(96);
    data_is_ready[2].init(96);
    data_is_free[0].init(384);
    data_is_free[1].init(384);
    data_is_free[2].init(384);
    bar_0[0].init(416);
    bar_1[0].init(256);
    bar_2[0].init(416);
    bar_3[0].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = (((int)blockIdx.x) >> 4);
  seq_start_idx = 0;
  seq_end_idx = num_tokens;
  chunk_start_idx = 0;
  num_iters = (((seq_end_idx + 63) - seq_start_idx) >> 6);
  calc_mt = (signed char)0;
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<168>();
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      *(float2*)(h_fragment + (i * 2)) = *(float2*)(h0 + ((((((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i) >> (int64_t)4) * (int64_t)2048)) + ((((int64_t)i) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + (((((int64_t)i) & (int64_t)15) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)));
    }
    for (int i_s = 0; i_s < num_iters; ++i_s) {
      data_is_ready[(i_s % 3)].wait(((i_s % 6) / 3));
      bar_0[0].arrive();
      bar_0[0].wait((i_s & 1));
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)h_shared)[(((((((((int)threadIdx.x) >> 5) * 4096) + ((i_1 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)h_fragment[(i_1 * 8)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 1)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 2)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 3)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 4)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 5)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 6)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 7)])));
      }
      tl::fence_proxy_async();
      bar_1[0].arrive();
      bar_1[0].wait((i_s & 1));
      g_last_local_S[0] = exp2f((g_shared[(((i_s % 3) * 64) + 63)] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
      #pragma unroll
      for (int i_2 = 0; i_2 < 32; ++i_2) {
        float4 __1;
          float4 v_ = *(float4*)(h_fragment + (i_2 * 4));
          float4 v__1 = make_float4(g_last_local_S[0], g_last_local_S[0], g_last_local_S[0], g_last_local_S[0]);
          *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
          *(float2*)(&(__1.z)) = tl::mul2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
        *(float4*)(h_fragment + (i_2 * 4)) = __1;
      }
      bar_2[0].arrive();
      bar_2[0].wait((i_s & 1));
      {
        bfloat16_t A_local[32];
        bfloat16_t B_local[32];
        for (int ki = 0; ki < 4; ++ki) {
          #pragma unroll
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)x_shared)[(((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (ki * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_3 * 8)])));
          }
          #pragma unroll
          for (int i_4 = 0; i_4 < 4; ++i_4) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)y_shared)[(((((((((int)threadIdx.x) >> 6) * 4096) + (ki * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
          }
          for (int i_5 = 0; i_5 < 4; ++i_5) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + ((i_5 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + (((i_5 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      bar_3[0].arrive();
      data_is_free[(i_s % 3)].arrive();
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 64; ++i_6) {
      *(float2*)(ht + ((((((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_6) >> (int64_t)4) * (int64_t)2048)) + ((((int64_t)i_6) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + (((((int64_t)i_6) & (int64_t)15) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(h_fragment + (i_6 * 2));
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<160>();
      if ((bool)calc_mt) {
        #pragma unroll
        for (int i_7 = 0; i_7 < 64; ++i_7) {
          if (((((((((int)threadIdx.x) & 63) >> 5) * 64) + ((i_7 >> 4) * 16)) + (((i_7 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == (((((((int)threadIdx.x) >> 6) * 32) + (((i_7 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1))) {
            m_fragment_R[i_7] = 0x1p+0f/*1.000000e+00*/;
          } else {
            m_fragment_R[i_7] = 0x0p+0f/*0.000000e+00*/;
          }
        }
      }
      if ((bool)calc_mt) {
        g_prod_X[0] = 0x0p+0f/*0.000000e+00*/;
      }
      for (int i_s_1 = 0; i_s_1 < num_iters; ++i_s_1) {
        data_is_ready[(i_s_1 % 3)].wait(((i_s_1 % 6) / 3));
        bar_0[0].arrive();
        bar_0[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_1[16];
          bfloat16_t B_local_1[32];
          #pragma unroll
          for (int i_8 = 0; i_8 < 16; ++i_8) {
            float broadcast_var = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(x_fragment + (i_8 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
          }
          for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
            #pragma unroll
            for (int i_9 = 0; i_9 < 2; ++i_9) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)a_shared)[((((((((i_s_1 % 3) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_9) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_1[(i_9 * 8)])));
            }
            #pragma unroll
            for (int i_10 = 0; i_10 < 4; ++i_10) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)k_shared)[((((((((i_s_1 % 3) * 8192) + (((((int)threadIdx.x) & 127) >> 6) * 4096)) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_10 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_10 * 8)])));
            }
            for (int i_11 = 0; i_11 < 2; ++i_11) {
              for (int j_1 = 0; j_1 < 4; ++j_1) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(x_fragment + ((i_11 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_11 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(x_fragment + (((i_11 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_11 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
              }
            }
          }
        }
        #pragma unroll
        for (int i_12 = 0; i_12 < 32; ++i_12) {
          float2 __2;
            float2 v__2 = *(float2*)(x_fragment + (i_12 * 2));
            float2 v__3 = make_float2((b_shared[((((((i_s_1 % 3) * 64) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + ((i_12 >> 4) * 16)) + ((i_12 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/), (b_shared[((((((i_s_1 % 3) * 64) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + ((i_12 >> 4) * 16)) + ((i_12 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/));
            *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
          *(float2*)(x_fragment + (i_12 * 2)) = __2;
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_13 = 0; i_13 < 8; ++i_13) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)x_shared)[((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_13 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 8192)])), __pack_half2(((bfloat16_t)x_fragment[(i_13 * 8)]), ((bfloat16_t)x_fragment[((i_13 * 8) + 1)])), __pack_half2(((bfloat16_t)x_fragment[((i_13 * 8) + 2)]), ((bfloat16_t)x_fragment[((i_13 * 8) + 3)])), __pack_half2(((bfloat16_t)x_fragment[((i_13 * 8) + 4)]), ((bfloat16_t)x_fragment[((i_13 * 8) + 5)])), __pack_half2(((bfloat16_t)x_fragment[((i_13 * 8) + 6)]), ((bfloat16_t)x_fragment[((i_13 * 8) + 7)])));
        }
        bar_2[0].arrive();
        if ((bool)calc_mt) {
          g_prod_X[0] = (g_prod_X[0] + g_shared[(((i_s_1 % 3) * 64) + 63)]);
        }
        if ((bool)calc_mt) {
          bar_2[0].wait((i_s_1 & 1));
        }
        if ((bool)calc_mt) {
          tl::__sync_thread_partial(4, 128);
          #pragma unroll
          for (int i_14 = 0; i_14 < 32; ++i_14) {
            bfloat16_t h_shared_local_cast[2];
            uint1 __3;
            float2 v__4 = *(float2*)(m_fragment_R + (i_14 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
            *(uint1*)(h_shared_local_cast + 0) = __3;
            *(uint1*)(((bfloat16_t*)h_shared) + (((((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_14 >> 3) * 1024)) + ((i_14 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_14 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_14 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8192)) = *(uint1*)(h_shared_local_cast + 0);
          }
        }
        if ((bool)calc_mt) {
          bar_3[0].wait((i_s_1 & 1));
        }
        tl::__sync_thread_partial(4, 128);
        if ((bool)calc_mt) {
          {
            bfloat16_t A_local_2[16];
            bfloat16_t B_local_2[16];
            #pragma unroll
            for (int i_15 = 0; i_15 < 8; ++i_15) {
              float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(z_fragment_R + (i_15 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
            }
            for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
              #pragma unroll
              for (int i_16 = 0; i_16 < 2; ++i_16) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[(((((((((i_s_1 % 3) * 8192) + ((ki_2 >> 2) * 4096)) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_16 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_2[(i_16 * 8)])));
              }
              #pragma unroll
              for (int i_17 = 0; i_17 < 2; ++i_17) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)h_shared)[((((((ki_2 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_17) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B_local_2[(i_17 * 8)])));
              }
              for (int i_18 = 0; i_18 < 2; ++i_18) {
                for (int j_2 = 0; j_2 < 2; ++j_2) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(z_fragment_R + ((i_18 * 16) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_18 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(z_fragment_R + (((i_18 * 16) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_18 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
                }
              }
            }
          }
        }
        if ((bool)calc_mt) {
          tl::__sync_thread_partial(4, 128);
          #pragma unroll
          for (int i_19 = 0; i_19 < 4; ++i_19) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)z_shared_R)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_19 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)z_fragment_R[(i_19 * 8)]), ((bfloat16_t)z_fragment_R[((i_19 * 8) + 1)])), __pack_half2(((bfloat16_t)z_fragment_R[((i_19 * 8) + 2)]), ((bfloat16_t)z_fragment_R[((i_19 * 8) + 3)])), __pack_half2(((bfloat16_t)z_fragment_R[((i_19 * 8) + 4)]), ((bfloat16_t)z_fragment_R[((i_19 * 8) + 5)])), __pack_half2(((bfloat16_t)z_fragment_R[((i_19 * 8) + 6)]), ((bfloat16_t)z_fragment_R[((i_19 * 8) + 7)])));
          }
        }
        tl::__sync_thread_partial(4, 128);
        if ((bool)calc_mt) {
          {
            bfloat16_t A_local_3[32];
            bfloat16_t B_local_3[16];
            for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
              #pragma unroll
              for (int i_20 = 0; i_20 < 4; ++i_20) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)x_shared)[(((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_20 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_20 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_3[(i_20 * 8)])));
              }
              #pragma unroll
              for (int i_21 = 0; i_21 < 2; ++i_21) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)z_shared_R)[(((((ki_3 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_21) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[(i_21 * 8)])));
              }
              for (int i_22 = 0; i_22 < 4; ++i_22) {
                for (int j_3 = 0; j_3 < 2; ++j_3) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(m_fragment_R + ((i_22 * 16) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_22 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(m_fragment_R + (((i_22 * 16) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_22 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
                }
              }
            }
          }
        }
        data_is_free[(i_s_1 % 3)].arrive();
      }
      if ((bool)calc_mt) {
        g_last_local_X[0] = exp2f((g_prod_X[0] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
        #pragma unroll
        for (int i_23 = 0; i_23 < 16; ++i_23) {
          float4 __4;
            float4 v__5 = *(float4*)(m_fragment_R + (i_23 * 4));
            float4 v__6 = make_float4(g_last_local_X[0], g_last_local_X[0], g_last_local_X[0], g_last_local_X[0]);
            *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
            *(float2*)(&(__4.z)) = tl::mul2(*(float2*)(&(v__5.z)), *(float2*)(&(v__6.z)));
          *(float4*)(m_fragment_R + (i_23 * 4)) = __4;
        }
        #pragma unroll
        for (int i_24 = 0; i_24 < 32; ++i_24) {
          *(float2*)(mt + ((((((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_24) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i_24) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_24) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(m_fragment_R + (i_24 * 2));
        }
      } else {
        #pragma unroll
        for (int i_25 = 0; i_25 < 16; ++i_25) {
          float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(m_fragment_R + (i_25 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
        }
        #pragma unroll
        for (int i_26 = 0; i_26 < 32; ++i_26) {
          *(float2*)(mt + ((((((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_26) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i_26) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_26) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(m_fragment_R + (i_26 * 2));
        }
      }
    } else {
      if (((int)threadIdx.x) < 384) {
        tl::warpgroup_reg_alloc<160>();
        if ((bool)calc_mt) {
          #pragma unroll
          for (int i_27 = 0; i_27 < 64; ++i_27) {
            if (((((((((int)threadIdx.x) & 63) >> 5) * 64) + ((i_27 >> 4) * 16)) + (((i_27 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == ((((((((int)threadIdx.x) >> 6) * 32) + (((i_27 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_27 & 1)) - 128)) {
              m_fragment_L[i_27] = 0x1p+0f/*1.000000e+00*/;
            } else {
              m_fragment_L[i_27] = 0x0p+0f/*0.000000e+00*/;
            }
          }
        }
        if ((bool)calc_mt) {
          g_prod_Y[0] = 0x0p+0f/*0.000000e+00*/;
        }
        for (int i_s_2 = 0; i_s_2 < num_iters; ++i_s_2) {
          data_is_ready[(i_s_2 % 3)].wait(((i_s_2 % 6) / 3));
          bar_0[0].arrive();
          bar_0[0].wait((i_s_2 & 1));
          g_last_local_Y[0] = g_shared[(((i_s_2 % 3) * 64) + 63)];
          tl::__sync_thread_partial(3, 128);
          if (((int)threadIdx.x) < 320) {
            g_rev_exp_shared[(((int)threadIdx.x) - 256)] = exp2f(((g_last_local_Y[0] - g_shared[((((i_s_2 % 3) * 64) + ((int)threadIdx.x)) - 256)]) * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          }
          g_last_local_Y[0] = exp2f((g_last_local_Y[0] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          bar_1[0].arrive();
          bar_1[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_4[16];
            bfloat16_t B_local_4[32];
            #pragma unroll
            for (int i_28 = 0; i_28 < 16; ++i_28) {
              float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(y_fragment + (i_28 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
            }
            for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
              #pragma unroll
              for (int i_29 = 0; i_29 < 2; ++i_29) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[(((((((((i_s_2 % 3) * 8192) + ((ki_4 >> 2) * 4096)) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_29 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_4[(i_29 * 8)])));
              }
              #pragma unroll
              for (int i_30 = 0; i_30 < 4; ++i_30) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)h_shared)[((((((((((int)threadIdx.x) & 127) >> 6) * 8192) + (ki_4 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_30 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_30 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_4[(i_30 * 8)])));
              }
              for (int i_31 = 0; i_31 < 2; ++i_31) {
                for (int j_4 = 0; j_4 < 4; ++j_4) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(y_fragment + ((i_31 * 32) + (j_4 * 8))), reinterpret_cast<const unsigned*>(A_local_4 + (i_31 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + (j_4 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(y_fragment + (((i_31 * 32) + (j_4 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + (i_31 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + ((j_4 * 8) + 4)));
                }
              }
            }
          }
          #pragma unroll
          for (int i_32 = 0; i_32 < 16; ++i_32) {
            float4 __5;
              float4 v__7 = *(float4*)(y_fragment + (i_32 * 4));
              float4 v__8 = make_float4(g_last_local_Y[0], g_last_local_Y[0], g_last_local_Y[0], g_last_local_Y[0]);
              *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)));
              *(float2*)(&(__5.z)) = tl::mul2(*(float2*)(&(v__7.z)), *(float2*)(&(v__8.z)));
            *(float4*)(y_fragment + (i_32 * 4)) = __5;
          }
          tl::__sync_thread_partial(3, 128);
          #pragma unroll
          for (int i_33 = 0; i_33 < 32; ++i_33) {
            bfloat16_t v_shared_local_cast_1[2];
            float g_rev_exp_shared_local_cast_2[2];
            *(uint1*)(v_shared_local_cast_1 + 0) = *(uint1*)(((bfloat16_t*)v_shared) + ((((((((((i_s_2 % 3) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + ((i_33 >> 4) * 2048)) + ((i_33 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_33 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 256));
            *(float2*)(g_rev_exp_shared_local_cast_2 + 0) = make_float2(g_rev_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_33 >> 4) * 16)) + ((i_33 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_rev_exp_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_33 >> 4) * 16)) + ((i_33 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            float2 __6;
              float2 v__9 = *(float2*)(y_fragment + (i_33 * 2));
              float2 __7;
                float2 __8;
                uint1 v__10 = *(uint1*)(v_shared_local_cast_1 + 0);
                ((float2*)(&__8))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__10))[0]);
                float2 v__11 = *(float2*)(g_rev_exp_shared_local_cast_2 + 0);
                *(float2*)(&(__7.x)) = tl::mul2(*(float2*)(&(__8.x)), *(float2*)(&(v__11.x)));
              *(float2*)(&(__6.x)) = tl::sub2(*(float2*)(&(v__9.x)), *(float2*)(&(__7.x)));
            *(float2*)(y_fragment + (i_33 * 2)) = __6;
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_34 = 0; i_34 < 8; ++i_34) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)y_shared)[((((((((((int)threadIdx.x) >> 5) * 2048) + ((i_34 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_34 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_34 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 16384)])), __pack_half2(((bfloat16_t)y_fragment[(i_34 * 8)]), ((bfloat16_t)y_fragment[((i_34 * 8) + 1)])), __pack_half2(((bfloat16_t)y_fragment[((i_34 * 8) + 2)]), ((bfloat16_t)y_fragment[((i_34 * 8) + 3)])), __pack_half2(((bfloat16_t)y_fragment[((i_34 * 8) + 4)]), ((bfloat16_t)y_fragment[((i_34 * 8) + 5)])), __pack_half2(((bfloat16_t)y_fragment[((i_34 * 8) + 6)]), ((bfloat16_t)y_fragment[((i_34 * 8) + 7)])));
          }
          bar_2[0].arrive();
          if ((bool)calc_mt) {
            g_prod_Y[0] = (g_prod_Y[0] + g_shared[(((i_s_2 % 3) * 64) + 63)]);
          }
          if ((bool)calc_mt) {
            bar_2[0].wait((i_s_2 & 1));
          }
          if ((bool)calc_mt) {
            tl::__sync_thread_partial(5, 128);
            #pragma unroll
            for (int i_35 = 0; i_35 < 32; ++i_35) {
              bfloat16_t h_shared_local_cast_3[2];
              uint1 __9;
              float2 v__12 = *(float2*)(m_fragment_L + (i_35 * 2));
              (reinterpret_cast<__nv_bfloat162*>(&__9))[0] = __float22bfloat162_rn(((float2*)(&v__12))[0]);
              *(uint1*)(h_shared_local_cast_3 + 0) = __9;
              *(uint1*)(((bfloat16_t*)h_shared) + ((((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_35 >> 3) * 1024)) + ((i_35 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_35 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_35 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(h_shared_local_cast_3 + 0);
            }
          }
          if ((bool)calc_mt) {
            bar_3[0].wait((i_s_2 & 1));
          }
          tl::__sync_thread_partial(5, 128);
          if ((bool)calc_mt) {
            {
              bfloat16_t A_local_5[16];
              bfloat16_t B_local_5[16];
              #pragma unroll
              for (int i_36 = 0; i_36 < 8; ++i_36) {
                float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
                *(float4*)(z_fragment_L + (i_36 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
              }
              for (int ki_5 = 0; ki_5 < 8; ++ki_5) {
                #pragma unroll
                for (int i_37 = 0; i_37 < 2; ++i_37) {
                  tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[(((((((((i_s_2 % 3) * 8192) + ((ki_5 >> 2) * 4096)) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_37 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_5 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_5[(i_37 * 8)])));
                }
                #pragma unroll
                for (int i_38 = 0; i_38 < 2; ++i_38) {
                  tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)h_shared)[(((((ki_5 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_38) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_5[(i_38 * 8)])));
                }
                for (int i_39 = 0; i_39 < 2; ++i_39) {
                  for (int j_5 = 0; j_5 < 2; ++j_5) {
                    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(z_fragment_L + ((i_39 * 16) + (j_5 * 8))), reinterpret_cast<const unsigned*>(A_local_5 + (i_39 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + (j_5 * 8)));
                    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(z_fragment_L + (((i_39 * 16) + (j_5 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + (i_39 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + ((j_5 * 8) + 4)));
                  }
                }
              }
            }
          }
          if ((bool)calc_mt) {
            tl::__sync_thread_partial(5, 128);
            #pragma unroll
            for (int i_40 = 0; i_40 < 4; ++i_40) {
              tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)z_shared_L)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_40 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_40 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)z_fragment_L[(i_40 * 8)]), ((bfloat16_t)z_fragment_L[((i_40 * 8) + 1)])), __pack_half2(((bfloat16_t)z_fragment_L[((i_40 * 8) + 2)]), ((bfloat16_t)z_fragment_L[((i_40 * 8) + 3)])), __pack_half2(((bfloat16_t)z_fragment_L[((i_40 * 8) + 4)]), ((bfloat16_t)z_fragment_L[((i_40 * 8) + 5)])), __pack_half2(((bfloat16_t)z_fragment_L[((i_40 * 8) + 6)]), ((bfloat16_t)z_fragment_L[((i_40 * 8) + 7)])));
            }
          }
          tl::__sync_thread_partial(5, 128);
          if ((bool)calc_mt) {
            {
              bfloat16_t A_local_6[32];
              bfloat16_t B_local_6[16];
              for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
                #pragma unroll
                for (int i_41 = 0; i_41 < 4; ++i_41) {
                  tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)x_shared)[(((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (ki_6 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_41 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_41 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_6[(i_41 * 8)])));
                }
                #pragma unroll
                for (int i_42 = 0; i_42 < 2; ++i_42) {
                  tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)z_shared_L)[(((((ki_6 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_42) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_6[(i_42 * 8)])));
                }
                for (int i_43 = 0; i_43 < 4; ++i_43) {
                  for (int j_6 = 0; j_6 < 2; ++j_6) {
                    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(m_fragment_L + ((i_43 * 16) + (j_6 * 8))), reinterpret_cast<const unsigned*>(A_local_6 + (i_43 * 8)), reinterpret_cast<const unsigned*>(B_local_6 + (j_6 * 8)));
                    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(m_fragment_L + (((i_43 * 16) + (j_6 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_6 + (i_43 * 8)), reinterpret_cast<const unsigned*>(B_local_6 + ((j_6 * 8) + 4)));
                  }
                }
              }
            }
          }
          data_is_free[(i_s_2 % 3)].arrive();
        }
        if ((bool)calc_mt) {
          g_last_local_Y[0] = exp2f((g_prod_Y[0] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          #pragma unroll
          for (int i_44 = 0; i_44 < 16; ++i_44) {
            float4 __10;
              float4 v__13 = *(float4*)(m_fragment_L + (i_44 * 4));
              float4 v__14 = make_float4(g_last_local_Y[0], g_last_local_Y[0], g_last_local_Y[0], g_last_local_Y[0]);
              *(float2*)(&(__10.x)) = tl::mul2(*(float2*)(&(v__13.x)), *(float2*)(&(v__14.x)));
              *(float2*)(&(__10.z)) = tl::mul2(*(float2*)(&(v__13.z)), *(float2*)(&(v__14.z)));
            *(float4*)(m_fragment_L + (i_44 * 4)) = __10;
          }
          #pragma unroll
          for (int i_45 = 0; i_45 < 32; ++i_45) {
            *(float2*)(mt + (((((((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_45) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i_45) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_45) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (int64_t)128)) = *(float2*)(m_fragment_L + (i_45 * 2));
          }
        } else {
          #pragma unroll
          for (int i_46 = 0; i_46 < 16; ++i_46) {
            float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(m_fragment_L + (i_46 * 4)) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
          }
          #pragma unroll
          for (int i_47 = 0; i_47 < 32; ++i_47) {
            *(float2*)(mt + (((((((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_47) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i_47) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_47) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (int64_t)128)) = *(float2*)(m_fragment_L + (i_47 * 2));
          }
        }
      } else {
        tl::warpgroup_reg_dealloc<24>();
        if (((int)threadIdx.x) < 416) {
          tl::__sync_thread_partial(6, 32);
          for (int i_s_3 = 0; i_s_3 < num_iters; ++i_s_3) {
            data_is_free[(i_s_3 % 3)].wait((((i_s_3 / 3) + 1) & 1));
            int left = ((i_s_3 * 64) + seq_start_idx);
            if ((left + 64) <= seq_end_idx) {
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_3 % 3)].expect_transaction(16384);
                tl::fence_proxy_async();
                #pragma unroll
                for (int i_48 = 0; i_48 < 2; ++i_48) {
                  tl::tma_load(k_desc, data_is_ready[(i_s_3 % 3)], (&(((bfloat16_t*)k_shared)[(((i_s_3 % 3) * 8192) + (i_48 * 4096))])), (i_48 * 64), left, (((int)blockIdx.x) & 15), batch_idx);
                }
              }
            } else {
              tl::__sync_thread_partial(6, 32);
              #pragma unroll
              for (int i_49 = 0; i_49 < 32; ++i_49) {
                if ((((i_49 * 2) + (((int)threadIdx.x) >> 4)) + left) < (seq_end_idx + 24)) {
                  bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  uint4 condval;
                  if (((((12 <= ((((((int)threadIdx.x) >> 4) + left) >> 1) + i_49)) && ((((i_49 * 2) + (((int)threadIdx.x) >> 4)) + left) < (num_tokens + 24))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                    condval = *(uint4*)(k + (((((((((int64_t)i_49) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)left) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)49152));
                  } else {
                    condval = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
                  }
                  *(uint4*)(((bfloat16_t*)k_shared) + (((((((((i_s_3 % 3) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + ((((i_49 * 256) + (((int)threadIdx.x) * 8)) >> 10) * 512)) + ((((i_49 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_49 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_49 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 1536)) = condval;
                } else {
                  bfloat16_t broadcast_var_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  *(uint4*)(((bfloat16_t*)k_shared) + (((((((((i_s_3 % 3) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + ((((i_49 * 256) + (((int)threadIdx.x) * 8)) >> 10) * 512)) + ((((i_49 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_49 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_49 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 1536)) = make_uint4(__pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7));
                }
              }
            }
            data_is_ready[(i_s_3 % 3)].arrive();
          }
        } else {
          if (((int)threadIdx.x) < 448) {
            tl::__sync_thread_partial(7, 32);
            for (int i_s_4 = 0; i_s_4 < num_iters; ++i_s_4) {
              data_is_free[(i_s_4 % 3)].wait((((i_s_4 / 3) + 1) & 1));
              int left_1 = ((i_s_4 * 64) + seq_start_idx);
              if ((left_1 + 64) <= seq_end_idx) {
                if (tl::tl_shuffle_elect<32>()) {
                  data_is_ready[(i_s_4 % 3)].expect_transaction(16384);
                  tl::fence_proxy_async();
                  tl::tma_load(v_desc, data_is_ready[(i_s_4 % 3)], (&(((bfloat16_t*)v_shared)[((i_s_4 % 3) * 8192)])), 0, left_1, (((int)blockIdx.x) & 15), batch_idx);
                }
              } else {
                #pragma unroll
                for (int i_50 = 0; i_50 < 32; ++i_50) {
                  if ((((i_50 * 2) + (((int)threadIdx.x) >> 4)) + left_1) < (seq_end_idx + 26)) {
                    bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    uint4 condval_1;
                    if (((((13 <= ((((((int)threadIdx.x) >> 4) + left_1) >> 1) + i_50)) && ((((i_50 * 2) + (((int)threadIdx.x) >> 4)) + left_1) < (num_tokens + 26))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                      condval_1 = *(uint4*)(v + (((((((((int64_t)i_50) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)left_1) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)53248));
                    } else {
                      condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
                    }
                    *(uint4*)(((bfloat16_t*)v_shared) + (((((i_s_4 % 3) * 8192) + (i_50 * 256)) + (((int)threadIdx.x) * 8)) - 3328)) = condval_1;
                  } else {
                    bfloat16_t broadcast_var_9 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    *(uint4*)(((bfloat16_t*)v_shared) + (((((i_s_4 % 3) * 8192) + (i_50 * 256)) + (((int)threadIdx.x) * 8)) - 3328)) = make_uint4(__pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9));
                  }
                }
              }
              tl::__sync_thread_partial(7, 32);
              if ((left_1 + 64) <= seq_end_idx) {
                if (tl::tl_shuffle_elect<32>()) {
                  data_is_ready[(i_s_4 % 3)].expect_transaction(8192);
                  tl::fence_proxy_async();
                  tl::tma_load(a_desc, data_is_ready[(i_s_4 % 3)], (&(((bfloat16_t*)a_shared)[((i_s_4 % 3) * 4096)])), 0, left_1, (((int)blockIdx.x) & 15), batch_idx);
                }
              } else {
                tl::__sync_thread_partial(7, 32);
                #pragma unroll
                for (int i_51 = 0; i_51 < 16; ++i_51) {
                  if ((((i_51 * 4) + (((int)threadIdx.x) >> 3)) + left_1) < (seq_end_idx + 52)) {
                    bfloat16_t broadcast_var_10 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    uint4 condval_2;
                    if (((((13 <= ((((((int)threadIdx.x) >> 3) + left_1) >> 2) + i_51)) && ((((i_51 * 4) + (((int)threadIdx.x) >> 3)) + left_1) < (num_tokens + 52))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                      condval_2 = *(uint4*)(a + (((((((((int64_t)i_51) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)1024)) + (((int64_t)left_1) * (int64_t)1024)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)53248));
                    } else {
                      condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10));
                    }
                    *(uint4*)(((bfloat16_t*)a_shared) + (((((((i_s_4 % 3) * 4096) + ((i_51 >> 1) * 512)) + (((((i_51 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_51 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
                  } else {
                    bfloat16_t broadcast_var_11 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    *(uint4*)(((bfloat16_t*)a_shared) + (((((((i_s_4 % 3) * 4096) + ((i_51 >> 1) * 512)) + (((((i_51 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_51 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11));
                  }
                }
              }
              data_is_ready[(i_s_4 % 3)].arrive();
            }
          } else {
            if (((int)threadIdx.x) < 480) {
              for (int i_s_5 = 0; i_s_5 < num_iters; ++i_s_5) {
                data_is_free[(i_s_5 % 3)].wait((((i_s_5 / 3) + 1) & 1));
                int left_2 = ((i_s_5 * 64) + seq_start_idx);
                if ((left_2 + 64) <= seq_end_idx) {
                  #pragma unroll
                  for (int i_52 = 0; i_52 < 2; ++i_52) {
                    float condval_3;
                    if (((((14 <= (((left_2 + ((int)threadIdx.x)) >> 5) + i_52)) && ((((i_52 * 32) + left_2) + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                      condval_3 = g[((((((((int64_t)i_52) * (int64_t)512) + (((int64_t)left_2) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)7168)];
                    } else {
                      condval_3 = 0x0p+0f/*0.000000e+00*/;
                    }
                    g_shared[(((((i_s_5 % 3) * 64) + (i_52 * 32)) + ((int)threadIdx.x)) - 448)] = condval_3;
                  }
                } else {
                  #pragma unroll
                  for (int i_53 = 0; i_53 < 2; ++i_53) {
                    if ((((i_53 * 32) + left_2) + ((int)threadIdx.x)) < (seq_end_idx + 448)) {
                      float condval_4;
                      if (((((14 <= (((left_2 + ((int)threadIdx.x)) >> 5) + i_53)) && ((((i_53 * 32) + left_2) + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                        condval_4 = g[((((((((int64_t)i_53) * (int64_t)512) + (((int64_t)left_2) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)7168)];
                      } else {
                        condval_4 = 0x0p+0f/*0.000000e+00*/;
                      }
                      g_shared[(((((i_s_5 % 3) * 64) + (i_53 * 32)) + ((int)threadIdx.x)) - 448)] = condval_4;
                    } else {
                      float condval_5;
                      if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                        condval_5 = g[((((((int64_t)seq_end_idx) * (int64_t)16) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)16)];
                      } else {
                        condval_5 = 0x0p+0f/*0.000000e+00*/;
                      }
                      g_shared[(((((i_s_5 % 3) * 64) + (i_53 * 32)) + ((int)threadIdx.x)) - 448)] = condval_5;
                    }
                  }
                }
                if ((left_2 + 64) <= seq_end_idx) {
                  #pragma unroll
                  for (int i_54 = 0; i_54 < 2; ++i_54) {
                    bfloat16_t condval_6;
                    if (((((14 <= (((left_2 + ((int)threadIdx.x)) >> 5) + i_54)) && ((((i_54 * 32) + left_2) + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                      condval_6 = b[((((((((int64_t)i_54) * (int64_t)512) + (((int64_t)left_2) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)7168)];
                    } else {
                      condval_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    }
                    b_shared[(((((i_s_5 % 3) * 64) + (i_54 * 32)) + ((int)threadIdx.x)) - 448)] = ((float)condval_6);
                  }
                } else {
                  #pragma unroll
                  for (int i_55 = 0; i_55 < 2; ++i_55) {
                    if ((((i_55 * 32) + left_2) + ((int)threadIdx.x)) < (seq_end_idx + 448)) {
                      bfloat16_t condval_7;
                      if (((((14 <= (((left_2 + ((int)threadIdx.x)) >> 5) + i_55)) && ((((i_55 * 32) + left_2) + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < batch_size))) {
                        condval_7 = b[((((((((int64_t)i_55) * (int64_t)512) + (((int64_t)left_2) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15)) - (int64_t)7168)];
                      } else {
                        condval_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      }
                      b_shared[(((((i_s_5 % 3) * 64) + (i_55 * 32)) + ((int)threadIdx.x)) - 448)] = ((float)condval_7);
                    } else {
                      b_shared[(((((i_s_5 % 3) * 64) + (i_55 * 32)) + ((int)threadIdx.x)) - 448)] = 0x0p+0f/*0.000000e+00*/;
                    }
                  }
                }
                data_is_ready[(i_s_5 % 3)].arrive();
              }
            } else {
              for (int i_s_6 = 0; i_s_6 < num_iters; ++i_s_6) {
                bar_0[0].arrive();
                bar_0[0].wait((i_s_6 & 1));
                bar_1[0].wait((i_s_6 & 1));
                if (tl::tl_shuffle_elect<32>()) {
                  #pragma unroll
                  for (int i_56 = 0; i_56 < 2; ++i_56) {
                    tl::tma_store(h_desc, (&(((bfloat16_t*)h_shared)[(i_56 * 8192)])), (i_56 * 64), 0, (((int)blockIdx.x) & 15), (chunk_start_idx + i_s_6), batch_idx);
                  }
                  tl::tma_store_arrive();
                  tl::tma_store_wait<0, true>();
                }
                bar_2[0].arrive();
              }
            }
          }
        }
      }
    }
  }
}

