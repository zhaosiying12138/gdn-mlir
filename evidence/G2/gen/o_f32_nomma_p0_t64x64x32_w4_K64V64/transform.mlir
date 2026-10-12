module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mms = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.tile_using_for %mms tile_sizes [16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %func2 = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.create_async_groups %func2 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
