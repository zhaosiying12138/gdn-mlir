// gdn-opt L0 冒烟：parse + verify + gdn-to-chunk 改写（generic 形式）
func.func @smoke(%q: tensor<2x128x2x16xf32>,
                 %k: tensor<2x128x2x16xf32>,
                 %v: tensor<2x128x2x16xf32>,
                 %beta: tensor<2x128x2xf32>,
                 %g: tensor<2x128x2xf32>,
                 %h0: tensor<2x2x16x16xf32>)
    -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>) {
  %o, %ht = "gdn.fwd"(%q, %k, %v, %beta, %g, %h0)
      <{config = #gdn.config<64, true, false, true, true, true, false>}>
      : (tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>, tensor<2x128x2x16xf32>,
         tensor<2x128x2xf32>, tensor<2x128x2xf32>, tensor<2x2x16x16xf32>)
      -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
  return %o, %ht : tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>
}
