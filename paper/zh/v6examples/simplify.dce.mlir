module {
  func.func @simplify(%arg0: i32) -> i32 {
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = arith.addi %c2_i32, %c4_i32 : i32
    %1 = arith.addi %arg0, %0 : i32
    %2 = arith.addi %arg0, %0 : i32
    %3 = arith.addi %1, %2 : i32
    return %3 : i32
  }
}

