func.func @simplify(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %k = arith.addi %c2, %c4 : i32
  %a = arith.addi %x, %k : i32
  %b = arith.addi %x, %k : i32
  %dead = arith.muli %a, %a : i32
  %r = arith.addi %a, %b : i32
  return %r : i32
}
