func.func @insert_keep_old(%x: tensor<4xf32>, %v: f32) -> (tensor<4xf32>, tensor<4xf32>) {
  %i = arith.constant 1 : index
  %y = tensor.insert %v into %x[%i] : tensor<4xf32>
  return %y, %x : tensor<4xf32>, tensor<4xf32>
}
