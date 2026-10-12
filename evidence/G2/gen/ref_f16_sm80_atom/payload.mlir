// GDN L2 'ref' anchor: single-atom f16 matmul lowered via the in-tree
// transform.nvgpu.rewrite_matmul_as_mma_sync (sm80 recipe retargeted sm_120a).
func.func @gdn_ref_kernel(%lhs: memref<16x16xf16>, %rhs: memref<16x8xf16>, %res: memref<16x8xf16>) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%bxx = %c32, %byy = %c1, %bzz = %c1) {
    linalg.matmul ins(%lhs, %rhs: memref<16x16xf16>, memref<16x8xf16>)
                 outs(%res: memref<16x8xf16>)
    gpu.terminator
  }
  return
}
