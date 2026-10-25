module {
  func.func @insert_keep_old(%arg0: memref<4xf32, strided<[?], offset: ?>>, %arg1: f32) -> (memref<4xf32>, memref<4xf32, strided<[?], offset: ?>>) {
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() alignment = 64 : memref<4xf32>
    memref.copy %arg0, %alloc : memref<4xf32, strided<[?], offset: ?>> to memref<4xf32>
    memref.store %arg1, %alloc[%c1] : memref<4xf32>
    %cast = memref.cast %alloc : memref<4xf32> to memref<4xf32, strided<[?], offset: ?>>
    return %alloc, %arg0 : memref<4xf32>, memref<4xf32, strided<[?], offset: ?>>
  }
}

