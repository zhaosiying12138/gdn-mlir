module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.rewrite_matmul_as_mma_sync %matmul
      : (!transform.any_op) -> ()
    transform.yield
  }
}
