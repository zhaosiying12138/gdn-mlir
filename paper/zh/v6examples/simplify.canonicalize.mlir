module {
  func.func @simplify(%arg0: i32) -> i32 {
    %c6_i32 = arith.constant 6 : i32
    %0 = arith.addi %arg0, %c6_i32 : i32
    %1 = arith.addi %arg0, %c6_i32 : i32
    %2 = arith.addi %0, %1 : i32
    return %2 : i32
  }
}

