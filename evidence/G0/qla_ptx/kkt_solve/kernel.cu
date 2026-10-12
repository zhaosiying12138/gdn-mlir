#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tilelang_kkt_solve_kernel_kernel(bfloat16_t* __restrict__ a, const bfloat16_t* __restrict__ b, const bfloat16_t* __restrict__ k, int data_batch_size, int num_chunks, int num_tokens);
extern "C" __global__ void __launch_bounds__(128, 1) tilelang_kkt_solve_kernel_kernel(bfloat16_t* __restrict__ a, const bfloat16_t* __restrict__ b, const bfloat16_t* __restrict__ k, int data_batch_size, int num_chunks, int num_tokens) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a16i_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* a16o_shared = ((void*)((char*)buf_dyn_shmem + 2176));
  void* a32_shared = ((void*)((char*)buf_dyn_shmem + 2176));
  int batch_idx = 0;
  int chunk_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  __shared__ __align__(16) float b_shared[64];
  float a32_fragment[32];
  float a16i_row[16];
  float a16i_sum[1];
  float a16o_fragment[2];
  int rmod = ((((int)blockIdx.x) >> 4) % data_batch_size);
  batch_idx = ((((0 <= data_batch_size) && (0 <= rmod)) || ((data_batch_size < 0) && (rmod <= 0))) ? rmod : (rmod + data_batch_size));
  int rmod_1 = ((((int)blockIdx.x) >> 4) % data_batch_size);
  int rdiv = ((((int)blockIdx.x) >> 4) / data_batch_size);
  chunk_idx = ((((0 <= data_batch_size) && (0 <= rmod_1)) || ((data_batch_size < 0) && (rmod_1 <= 0))) ? rdiv : (rdiv - 1));
  seq_start_idx = 0;
  seq_end_idx = num_tokens;
  int chunk_idx_1 = chunk_idx;
  int seq_start_idx_1 = seq_start_idx;
  int seq_end_idx_1 = seq_end_idx;
  if ((((chunk_idx_1 * 64) + seq_start_idx_1) + 64) <= seq_end_idx_1) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      int64_t rmod_2 = ((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size));
      int64_t rmod_3 = ((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size));
            int rmod_4 = ((((int)blockIdx.x) >> 4) % data_batch_size);
tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)k_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(k[(((((((((int64_t)chunk_idx_1) * (int64_t)131072) + (((int64_t)i) * (int64_t)16384)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)seq_start_idx_1) * (int64_t)2048)) + (((((((int64_t)0 <= ((int64_t)data_batch_size)) && ((int64_t)0 <= rmod_3)) || ((((int64_t)data_batch_size) < (int64_t)0) && (rmod_3 <= (int64_t)0))) ? rmod_3 : (rmod_3 + ((int64_t)data_batch_size))) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])), (((((((((chunk_idx_1 * 64) + (i * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) < num_tokens) && (0 <= ((((chunk_idx_1 * 64) + (i * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1))) && (((((0 <= data_batch_size) && (0 <= rmod_4)) || ((data_batch_size < 0) && (rmod_4 <= 0))) ? rmod_4 : (rmod_4 + data_batch_size)) < data_batch_size)) && (((((chunk_idx_1 * 64) + (i * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) < num_tokens)) && (0 <= ((((chunk_idx_1 * 64) + (i * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1))));
    }
    tl::cp_async_commit();
  } else {
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      if (((((chunk_idx_1 * 64) + (i_1 * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) < seq_end_idx_1) {
        bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        int rmod_5 = ((((int)blockIdx.x) >> 4) % data_batch_size);
        uint4 condval;
        if ((((0 <= ((((chunk_idx_1 * 64) + (i_1 * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1)) && (((((chunk_idx_1 * 64) + (i_1 * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) < num_tokens)) && (((((0 <= data_batch_size) && (0 <= rmod_5)) || ((data_batch_size < 0) && (rmod_5 <= 0))) ? rmod_5 : (rmod_5 + data_batch_size)) < data_batch_size))) {
          int64_t rmod_6 = ((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size));
          int64_t rmod_7 = ((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size));
          condval = *(uint4*)(k + (((((((((int64_t)chunk_idx_1) * (int64_t)131072) + (((int64_t)i_1) * (int64_t)16384)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)seq_start_idx_1) * (int64_t)2048)) + (((((((int64_t)0 <= ((int64_t)data_batch_size)) && ((int64_t)0 <= rmod_7)) || ((((int64_t)data_batch_size) < (int64_t)0) && (rmod_7 <= (int64_t)0))) ? rmod_7 : (rmod_7 + ((int64_t)data_batch_size))) * ((int64_t)num_tokens)) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
        }
        *(uint4*)(((bfloat16_t*)k_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
      } else {
        bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        *(uint4*)(((bfloat16_t*)k_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
      }
    }
  }
  if ((((chunk_idx_1 * 64) + seq_start_idx_1) + 64) <= seq_end_idx_1) {
    if (((int)threadIdx.x) < 64) {
      int rmod_8 = ((((int)blockIdx.x) >> 4) % data_batch_size);
      bfloat16_t condval_1;
      if ((((0 <= (((chunk_idx_1 * 64) + seq_start_idx_1) + ((int)threadIdx.x))) && ((((chunk_idx_1 * 64) + seq_start_idx_1) + ((int)threadIdx.x)) < num_tokens)) && (((((0 <= data_batch_size) && (0 <= rmod_8)) || ((data_batch_size < 0) && (rmod_8 <= 0))) ? rmod_8 : (rmod_8 + data_batch_size)) < data_batch_size))) {
        int64_t rmod_9 = ((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size));
        int64_t rmod_10 = ((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size));
        condval_1 = b[(((((((int64_t)chunk_idx_1) * (int64_t)1024) + (((int64_t)seq_start_idx_1) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + (((((((int64_t)0 <= ((int64_t)data_batch_size)) && ((int64_t)0 <= rmod_10)) || ((((int64_t)data_batch_size) < (int64_t)0) && (rmod_10 <= (int64_t)0))) ? rmod_10 : (rmod_10 + ((int64_t)data_batch_size))) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15))];
      } else {
        condval_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      }
      b_shared[((int)threadIdx.x)] = ((float)condval_1);
    }
  } else {
    if (((int)threadIdx.x) < 64) {
      if ((((chunk_idx_1 * 64) + seq_start_idx_1) + ((int)threadIdx.x)) < seq_end_idx_1) {
        int rmod_11 = ((((int)blockIdx.x) >> 4) % data_batch_size);
        bfloat16_t condval_2;
        if ((((0 <= (((chunk_idx_1 * 64) + seq_start_idx_1) + ((int)threadIdx.x))) && ((((chunk_idx_1 * 64) + seq_start_idx_1) + ((int)threadIdx.x)) < num_tokens)) && (((((0 <= data_batch_size) && (0 <= rmod_11)) || ((data_batch_size < 0) && (rmod_11 <= 0))) ? rmod_11 : (rmod_11 + data_batch_size)) < data_batch_size))) {
          int64_t rmod_12 = ((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size));
          int64_t rmod_13 = ((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size));
          condval_2 = b[(((((((int64_t)chunk_idx_1) * (int64_t)1024) + (((int64_t)seq_start_idx_1) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + (((((((int64_t)0 <= ((int64_t)data_batch_size)) && ((int64_t)0 <= rmod_13)) || ((((int64_t)data_batch_size) < (int64_t)0) && (rmod_13 <= (int64_t)0))) ? rmod_13 : (rmod_13 + ((int64_t)data_batch_size))) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15))];
        } else {
          condval_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        }
        b_shared[((int)threadIdx.x)] = ((float)condval_2);
      } else {
        b_shared[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
      }
    }
  }
  if ((((chunk_idx_1 * 64) + seq_start_idx_1) + 64) <= seq_end_idx_1) {
    tl::cp_async_wait<0>();
  }
  __syncthreads();
  {
    bfloat16_t A_local[16];
    bfloat16_t B_local[16];
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(a32_fragment + (i_2 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    for (int ki = 0; ki < 8; ++ki) {
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[((((((((ki >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_3 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_3 * 8)])));
      }
      #pragma unroll
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[(((((((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (i_4 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
      }
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(a32_fragment + ((i_5 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(a32_fragment + (((i_5 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 16; ++i_6) {
    float2 __1;
      float2 v_ = *(float2*)(a32_fragment + (i_6 * 2));
      float2 v__1 = make_float2(b_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_6 >> 3) * 16)) + ((i_6 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], b_shared[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_6 >> 3) * 16)) + ((i_6 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
      *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
    *(float2*)(a32_fragment + (i_6 * 2)) = __1;
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 32; ++i_7) {
    if (((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_7 >> 4) * 16)) + (((i_7 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < (((((((int)threadIdx.x) >> 6) * 32) + (((i_7 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1))) {
      a32_fragment[i_7] = 0x0p+0f/*0.000000e+00*/;
    } else {
      if (((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_7 >> 4) * 16)) + (((i_7 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == (((((((int)threadIdx.x) >> 6) * 32) + (((i_7 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1))) {
        a32_fragment[i_7] = 0x1p+0f/*1.000000e+00*/;
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_8 = 0; i_8 < 16; ++i_8) {
    if (((((((int)threadIdx.x) & 63) >> 5) * 2) + (i_8 >> 3)) == ((((((int)threadIdx.x) >> 6) * 2) + ((i_8 & 7) >> 2)) + 1)) {
      float broadcast_var_3 = -0x1p+0f/*-1.000000e+00*/;
      float2 __2;
        float2 v__2 = *(float2*)(a32_fragment + (i_8 * 2));
        float2 v__3 = make_float2(broadcast_var_3, broadcast_var_3);
        *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
      *(float2*)(((float*)a16o_shared) + (((((i_8 & 1) * 128) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + (((i_8 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __2;
    } else {
      if (((((((int)threadIdx.x) & 63) >> 5) * 2) + (i_8 >> 3)) == (((((int)threadIdx.x) >> 6) * 2) + ((i_8 & 7) >> 2))) {
        *(float2*)(((float*)a16i_shared) + ((((((((((int)threadIdx.x) & 63) >> 5) * 544) + ((i_8 >> 3) * 272)) + ((i_8 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + (((i_8 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(a32_fragment + (i_8 * 2));
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int k_s = 1; k_s < 16; ++k_s) {
    #pragma unroll
    for (int i_9 = 0; i_9 < 16; ++i_9) {
      if (i_9 < k_s) {
        a16i_row[i_9] = ((float*)a16i_shared)[(((((((int)threadIdx.x) & 31) >> 4) * 272) + (k_s * 16)) + i_9)];
      }
    }
    a16i_sum[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int k_r = 0; k_r < k_s; ++k_r) {
      a16i_sum[0] = (a16i_sum[0] - (((float*)a16i_shared)[(((((((int)threadIdx.x) & 31) >> 4) * 272) + (k_r * 16)) + (((int)threadIdx.x) & 15))] * a16i_row[k_r]));
    }
    __syncthreads();
    if ((((int)threadIdx.x) >> 5) == 0) {
      if ((((int)threadIdx.x) & 15) < k_s) {
        ((float*)a16i_shared)[(((((((int)threadIdx.x) & 31) >> 4) * 272) + (k_s * 16)) + (((int)threadIdx.x) & 15))] = a16i_sum[0];
      }
    }
  }
  float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(a16o_fragment + 0) = make_float2(broadcast_var_4, broadcast_var_4);
  __syncthreads();
  #pragma unroll
  for (int k_r_1 = 0; k_r_1 < 16; ++k_r_1) {
    float2 __3;
      float2 v__4 = make_float2(((float*)a16i_shared)[((((((int)threadIdx.x) >> 3) * 16) + k_r_1) + 272)], ((float*)a16i_shared)[((((((int)threadIdx.x) >> 3) * 16) + k_r_1) + 272)]);
      float2 v__5 = *(float2*)(((float*)a16o_shared) + ((k_r_1 * 16) + ((((int)threadIdx.x) & 7) * 2)));
      float2 v__6 = *(float2*)(a16o_fragment + 0);
      *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
    *(float2*)(a16o_fragment + 0) = __3;
  }
  __syncthreads();
  #pragma unroll
  for (int i_10 = 0; i_10 < 2; ++i_10) {
    ((float*)a16o_shared)[((((((int)threadIdx.x) & 7) * 32) + (i_10 * 16)) + (((int)threadIdx.x) >> 3))] = a16o_fragment[i_10];
  }
  float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(a16o_fragment + 0) = make_float2(broadcast_var_5, broadcast_var_5);
  __syncthreads();
  #pragma unroll
  for (int k_r_2 = 0; k_r_2 < 16; ++k_r_2) {
    float2 __4;
      float2 v__7 = make_float2(((float*)a16o_shared)[((k_r_2 * 16) + (((int)threadIdx.x) >> 3))], ((float*)a16o_shared)[((k_r_2 * 16) + (((int)threadIdx.x) >> 3))]);
      float2 v__8 = *(float2*)(((float*)a16i_shared) + ((k_r_2 * 16) + ((((int)threadIdx.x) & 7) * 2)));
      float2 v__9 = *(float2*)(a16o_fragment + 0);
      *(float2*)(&(__4.x)) = tl::fma2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)), *(float2*)(&(v__9.x)));
    *(float2*)(a16o_fragment + 0) = __4;
  }
  __syncthreads();
  *(float2*)(((float*)a16o_shared) + (((int)threadIdx.x) * 2)) = *(float2*)(a16o_fragment + 0);
  __syncthreads();
  *(float2*)(((float*)a32_shared) + (((((int)threadIdx.x) >> 3) * 64) + ((((int)threadIdx.x) & 7) * 2))) = *(float2*)(((float*)a16i_shared) + (((int)threadIdx.x) * 2));
  float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(((float*)a32_shared) + ((((((int)threadIdx.x) >> 3) * 64) + ((((int)threadIdx.x) & 7) * 2)) + 16)) = make_float2(broadcast_var_6, broadcast_var_6);
  *(float2*)(((float*)a32_shared) + ((((((int)threadIdx.x) >> 3) * 64) + ((((int)threadIdx.x) & 7) * 2)) + 1024)) = *(float2*)(a16o_fragment + 0);
  *(float2*)(((float*)a32_shared) + ((((((int)threadIdx.x) >> 3) * 64) + ((((int)threadIdx.x) & 7) * 2)) + 1040)) = *(float2*)(((float*)a16i_shared) + ((((int)threadIdx.x) * 2) + 272));
  __syncthreads();
  if ((((chunk_idx_1 * 64) + seq_start_idx_1) + 64) <= seq_end_idx_1) {
    #pragma unroll
    for (int i_11 = 0; i_11 < 8; ++i_11) {
      float a32_shared_local_cast_1[4];
      bfloat16_t a_local_cast[4];
      if ((((((chunk_idx_1 * 64) + (i_11 * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) < num_tokens) && (0 <= ((((chunk_idx_1 * 64) + (i_11 * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1))) {
        *(float4*)(a32_shared_local_cast_1 + 0) = *(float4*)(((float*)a32_shared) + ((i_11 * 512) + (((int)threadIdx.x) * 4)));
        uint2 __5;
        float4 v__10 = *(float4*)(a32_shared_local_cast_1 + 0);
        (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__10))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__5))[1] = __float22bfloat162_rn(((float2*)(&v__10))[1]);
        *(uint2*)(a_local_cast + 0) = __5;
        int64_t rmod_14 = ((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size));
        int64_t rmod_15 = ((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size));
        *(uint2*)(a + (((((((((int64_t)chunk_idx_1) * (int64_t)65536) + (((int64_t)i_11) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)seq_start_idx_1) * (int64_t)1024)) + (((((((int64_t)0 <= ((int64_t)data_batch_size)) && ((int64_t)0 <= rmod_15)) || ((((int64_t)data_batch_size) < (int64_t)0) && (rmod_15 <= (int64_t)0))) ? rmod_15 : (rmod_15 + ((int64_t)data_batch_size))) * ((int64_t)num_tokens)) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)4))) = *(uint2*)(a_local_cast + 0);
      }
    }
  } else {
    #pragma unroll
    for (int i_12 = 0; i_12 < 8; ++i_12) {
      float a32_shared_local_cast_3[4];
      bfloat16_t a_local_cast_2[4];
      if (((((chunk_idx_1 * 64) + (i_12 * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) < seq_end_idx_1) {
        *(float4*)(a32_shared_local_cast_3 + 0) = *(float4*)(((float*)a32_shared) + ((i_12 * 512) + (((int)threadIdx.x) * 4)));
        uint2 __6;
        float4 v__11 = *(float4*)(a32_shared_local_cast_3 + 0);
        (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&v__11))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__6))[1] = __float22bfloat162_rn(((float2*)(&v__11))[1]);
        *(uint2*)(a_local_cast_2 + 0) = __6;
        if (0 <= ((((chunk_idx_1 * 64) + (i_12 * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1)) {
          if (((((chunk_idx_1 * 64) + (i_12 * 8)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) < num_tokens) {
            int64_t rmod_16 = ((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size));
            int64_t rmod_17 = ((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size));
            *(uint2*)(a + (((((((((int64_t)chunk_idx_1) * (int64_t)65536) + (((int64_t)i_12) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)seq_start_idx_1) * (int64_t)1024)) + (((((((int64_t)0 <= ((int64_t)data_batch_size)) && ((int64_t)0 <= rmod_17)) || ((((int64_t)data_batch_size) < (int64_t)0) && (rmod_17 <= (int64_t)0))) ? rmod_17 : (rmod_17 + ((int64_t)data_batch_size))) * ((int64_t)num_tokens)) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)4))) = *(uint2*)(a_local_cast_2 + 0);
          }
        }
      }
    }
  }
}

