memref.global "private" constant @gdn_in_lhs : memref<16x16xf16> = dense<[
  [-3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0],
  [-0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5, 1.5, -3.0],
  [2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5],
  [-2.0, 0.0, 2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0],
  [0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0],
  [3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5],
  [-1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0, 3.0],
  [1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5, 1.5, -3.0, -1.0],
  [-2.5, -0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5, 1.5],
  [0.0, 2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5],
  [2.5, -2.0, 0.0, 2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0],
  [-1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5],
  [1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5],
  [-3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0],
  [-0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5, 1.5, -3.0],
  [2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5]
]>
memref.global "private" constant @gdn_in_rhs : memref<16x8xf16> = dense<[
  [-3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0],
  [-0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5],
  [2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0, 3.0],
  [-2.0, 0.0, 2.0, -2.5, -0.5, 1.5, -3.0, -1.0],
  [0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5, 1.5],
  [3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5],
  [-1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0],
  [1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5],
  [-2.5, -0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5],
  [0.0, 2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0],
  [2.5, -2.0, 0.0, 2.0, -2.5, -0.5, 1.5, -3.0],
  [-1.5, 0.5, 2.5, -2.0, 0.0, 2.0, -2.5, -0.5],
  [1.0, 3.0, -1.5, 0.5, 2.5, -2.0, 0.0, 2.0],
  [-3.0, -1.0, 1.0, 3.0, -1.5, 0.5, 2.5, -2.0],
  [-0.5, 1.5, -3.0, -1.0, 1.0, 3.0, -1.5, 0.5],
  [2.0, -2.5, -0.5, 1.5, -3.0, -1.0, 1.0, 3.0]
]>
memref.global @gdn_out : memref<16x8xf32> = dense<0.0>
// GDN L2 'ref_ldmatrix' anchor: single f16 m16n8k16 atom whose fragments
// are loaded with in-tree nvgpu.ldmatrix (x4 for A; x2 .trans for B) instead
// of the lane-indexed ld.shared.b16 loads the sm80 rewrite recipe emits.
func.func @gdn_ref_ldmatrix_kernel() {
  %hc1 = arith.constant 1 : index
  %hc32 = arith.constant 32 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %hc1, %gy = %hc1, %gz = %hc1)
             threads(%tx, %ty, %tz) in (%bxx = %hc32, %byy = %hc1, %bzz = %hc1) {
    %lhs = memref.get_global @gdn_in_lhs : memref<16x16xf16>
    %rhs = memref.get_global @gdn_in_rhs : memref<16x8xf16>
    %res = memref.get_global @gdn_out : memref<16x8xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    %c16 = arith.constant 16 : index
    %lane = gpu.thread_id x
    %g = arith.divsi %lane, %c4 : index
    %tig = arith.remsi %lane, %c4 : index
    %t2 = arith.muli %tig, %c2 : index
    %t2p1 = arith.addi %t2, %c1 : index
    %g8 = arith.addi %g, %c8 : index
    %padi = arith.constant 0.0 : f16
    %sa = memref.alloc() : memref<16x16xf16, #gpu.address_space<workgroup>>
    %sb = memref.alloc() : memref<16x16xf16, #gpu.address_space<workgroup>>
    %ok = arith.cmpi ult, %lane, %c16 : index
    scf.if %ok {
      // per-lane ld.global.b16 / st.shared.b16 staging (transfer_write to
      // workgroup does not lower through this pipeline - G2-A constraint 5)
      %a0 = memref.load %lhs[%lane, %c0] : memref<16x16xf16>
      memref.store %a0, %sa[%lane, %c0] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a1 = memref.load %lhs[%lane, %c1] : memref<16x16xf16>
      memref.store %a1, %sa[%lane, %c1] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a2 = memref.load %lhs[%lane, %c2] : memref<16x16xf16>
      memref.store %a2, %sa[%lane, %c2] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a3 = memref.load %lhs[%lane, %c3] : memref<16x16xf16>
      memref.store %a3, %sa[%lane, %c3] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a4 = memref.load %lhs[%lane, %c4] : memref<16x16xf16>
      memref.store %a4, %sa[%lane, %c4] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a5 = memref.load %lhs[%lane, %c5] : memref<16x16xf16>
      memref.store %a5, %sa[%lane, %c5] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a6 = memref.load %lhs[%lane, %c6] : memref<16x16xf16>
      memref.store %a6, %sa[%lane, %c6] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a7 = memref.load %lhs[%lane, %c7] : memref<16x16xf16>
      memref.store %a7, %sa[%lane, %c7] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a8 = memref.load %lhs[%lane, %c8] : memref<16x16xf16>
      memref.store %a8, %sa[%lane, %c8] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a9 = memref.load %lhs[%lane, %c9] : memref<16x16xf16>
      memref.store %a9, %sa[%lane, %c9] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a10 = memref.load %lhs[%lane, %c10] : memref<16x16xf16>
      memref.store %a10, %sa[%lane, %c10] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a11 = memref.load %lhs[%lane, %c11] : memref<16x16xf16>
      memref.store %a11, %sa[%lane, %c11] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a12 = memref.load %lhs[%lane, %c12] : memref<16x16xf16>
      memref.store %a12, %sa[%lane, %c12] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a13 = memref.load %lhs[%lane, %c13] : memref<16x16xf16>
      memref.store %a13, %sa[%lane, %c13] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a14 = memref.load %lhs[%lane, %c14] : memref<16x16xf16>
      memref.store %a14, %sa[%lane, %c14] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %a15 = memref.load %lhs[%lane, %c15] : memref<16x16xf16>
      memref.store %a15, %sa[%lane, %c15] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %b0 = memref.load %rhs[%lane, %c0] : memref<16x8xf16>
      memref.store %b0, %sb[%lane, %c0] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %b1 = memref.load %rhs[%lane, %c1] : memref<16x8xf16>
      memref.store %b1, %sb[%lane, %c1] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %b2 = memref.load %rhs[%lane, %c2] : memref<16x8xf16>
      memref.store %b2, %sb[%lane, %c2] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %b3 = memref.load %rhs[%lane, %c3] : memref<16x8xf16>
      memref.store %b3, %sb[%lane, %c3] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %b4 = memref.load %rhs[%lane, %c4] : memref<16x8xf16>
      memref.store %b4, %sb[%lane, %c4] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %b5 = memref.load %rhs[%lane, %c5] : memref<16x8xf16>
      memref.store %b5, %sb[%lane, %c5] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %b6 = memref.load %rhs[%lane, %c6] : memref<16x8xf16>
      memref.store %b6, %sb[%lane, %c6] : memref<16x16xf16, #gpu.address_space<workgroup>>
      %b7 = memref.load %rhs[%lane, %c7] : memref<16x8xf16>
      memref.store %b7, %sb[%lane, %c7] : memref<16x16xf16, #gpu.address_space<workgroup>>
    }
    gpu.barrier
    %af = nvgpu.ldmatrix %sa[%c0, %c0] numTiles = 4 transpose = false : memref<16x16xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %bf = nvgpu.ldmatrix %sb[%c0, %c0] numTiles = 2 transpose = true : memref<16x16xf16, #gpu.address_space<workgroup>> -> vector<2x2xf16>
    %zf = arith.constant 0.0 : f16
    %zacc = vector.broadcast %zf : f16 to vector<2x2xf16>
    %acc = nvgpu.mma.sync(%af, %bf, %zacc) mmaShape = [16, 8, 16] : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %e0 = vector.extract %acc[0, 0] : f16 from vector<2x2xf16>
    %e1 = vector.extract %acc[0, 1] : f16 from vector<2x2xf16>
    %e2 = vector.extract %acc[1, 0] : f16 from vector<2x2xf16>
    %e3 = vector.extract %acc[1, 1] : f16 from vector<2x2xf16>
    %r0 = arith.addi %g, %c0 : index
    %r1 = arith.addi %g8, %c0 : index
    %x0 = arith.extf %e0 : f16 to f32
    %x1 = arith.extf %e1 : f16 to f32
    %x2 = arith.extf %e2 : f16 to f32
    %x3 = arith.extf %e3 : f16 to f32
    memref.store %x0, %res[%r0, %t2] : memref<16x8xf32>
    memref.store %x1, %res[%r0, %t2p1] : memref<16x8xf32>
    memref.store %x2, %res[%r1, %t2] : memref<16x8xf32>
    memref.store %x3, %res[%r1, %t2p1] : memref<16x8xf32>
    gpu.terminator
  }
  return
}
