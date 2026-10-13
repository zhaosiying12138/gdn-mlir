## 十九、scf：循环携带状态与 iter_args

linalg 声明"算什么"，scf（structured control flow，结构化控制流方言）声明"按什么顺序算"。顺序在 GDN 里是语义契约：四步状态递推不可交换，前代换必须按行进行。把顺序写成 IR 的载体是 scf.for，而它的关键机制是 iter_args——循环携带变量（loop-carried variable，值随迭代轮次更新的量）在 SSA（静态单赋值）世界里的表达。命令式语言里"变量 x 在循环里被反复赋值"是隐式状态；scf 把它变成操作数与结果的显式配对，数据流在文本上闭合。T = (I+A)^(−1) 的前代换以三重循环的形态完整落在 IR 上。

### 1. scf.for 的解剖：init_args 与 yield 的配对

scf.for %i = 下界 to 上界 step 步长 { 循环体 }。不携带状态时它只做顺序控制；携带状态时写作 iter_args(%acc = %init) -> (类型)：循环体把 %acc 当块参数接收，读到的是上一轮的值；scf.yield %new 声明交给下一轮的值；循环结束后，最后一轮 yield 的值以 scf.for 结果的身份（这里取名 %sum）在循环外可用。最平实的携带状态经内存中转也能写——gc 的 chunk 内包含前缀和就是这样一段：

```
%g0v = memref.load %g_c[%c0] : memref<64xf32, strided<[2], offset: ?>>
memref.store %g0v, %gc[%c0] : memref<64xf32>
scf.for %i = %c1 to %cC step %c1 {
  %ip = arith.subi %i, %c1 : index
  %prev = memref.load %gc[%ip] : memref<64xf32>
  %cur = memref.load %g_c[%i] : memref<64xf32, strided<[2], offset: ?>>
  %cs = arith.addf %prev, %cur : f32
  memref.store %cs, %gc[%i] : memref<64xf32>
}
```
［evidence/G1/l1_bwd_chunked_T128_C64.mlir:2514-2522，逐字］

gc 是 chunk 门控的累积和：第 i 个位置读上一位置的结果、加上当前输入、写回。这里的循环携带状态存在 %gc 指向的内存里，跨迭代的数据流经 load/store 对可见。下一个展品则把携带状态留在值的世界里。

### 2. 归约循环：前代换的最内层

前代换求 T = (I+A)^(−1)，按定义 T[i,i]=1、T[i,j] = −Σ A[i,k]·T[k,j]（j≤k<i），逐行求解。它在 IR 里的完整形态是三重循环——i 选行、j 选列、k 做归约——最内层是教科书式的 iter_args 归约：

```
linalg.fill ins(%f0 : f32) outs(%Tbuf : memref<64x64xf32>)
scf.for %i2 = %c0 to %cC step %c1 {
  memref.store %f1, %Tbuf[%i2, %i2] : memref<64x64xf32>          // 单位对角
  scf.for %j2 = %c0 to %i2 step %c1 {
    %sum = scf.for %k2 = %j2 to %i2 step %c1 iter_args(%acc2 = %f0) -> (f32) {
      %aik = memref.load %Abuf[%i2, %k2] : memref<64x64xf32>
      %tkj = memref.load %Tbuf[%k2, %j2] : memref<64x64xf32>
      %pk = arith.mulf %aik, %tkj : f32
      %ak = arith.addf %acc2, %pk : f32
      scf.yield %ak : f32
    }
    %neg = arith.negf %sum : f32
    memref.store %neg, %Tbuf[%i2, %j2] : memref<64x64xf32>       // T[i,j] = −Σ A[i,k]·T[k,j]
  }
}
```
［evidence/G1/l1_bwd_chunked_T128_C64.mlir:2559-2573，逐字；fill 在 2559 行］

三层数据流各有要点。最内层：%acc2 以 %f0（常数 0.0）为第 0 轮值，每轮读 A[i,k] 与 T[k,j] 乘加出 %ak，yield 交下一轮；循环退出时 %sum 即完整的 Σ A[i,k]·T[k,j]，取负后写进 T[i,j]。中层：j2 的循环上界是 %i2——列只走到当前行之前；内层 k2 从 %j2 起到 %i2 止。两层上界互相依赖，三角形的迭代空间直接由 for 的 to 操作数表达，循环体里没有任何位置判断。最外层：先写单位对角，再算严格下三角区。依赖方向保证正确性：内层读的 T[k,j] 都在更早的行（k<i），那些行已在前几轮外层迭代中定稿——这正是"前代换"三个字的 IR 形态：按时间顺序逐行求解，无主元选取、无除法。

### 3. 为什么这个形态让"数学即 IR"

前代换、三角掩蔽、状态递推全部以基础 op 写出：memref.load、arith.mulf、scf.yield。全链没有库调用、没有不透明的黑盒 op，任何一步的数值行为都可以从文本直接读出。两个直接后果。其一，可执行：这份 IR 经第二十章的 11 pass 管线下降后在 CPU 上跑出数字，与 JAX fp64 规约对拍闭合——算法层自带一条独立于 GPU 的验证路径，下层任何环节可疑时都能回退到这里重新对照。其二，可搬运：同一份文本是第二十三章生成器的输出格式，未来 L1 展开 pass 的输出与之逐行同构，规格与实现不会漂移——生成器侧的发射序列与产物逐行对应［tools/gdn_l1_gen.py:433-473，发射序列与第十八章展品逐行同构；dialect/lib/GdnPasses.cpp:1-9，头注声明这一同构关系］。

顺带一个语义细节：反向链的第二遍前向重算里另有一处同构的前代换三重循环（2725-2738 行），两处形状相同、操作数不同——同一段发射代码参数化地服务两个调用点，这也是生成器形态的直接收益。

### 4. scf 到 cf 的下降

-convert-scf-to-cf 把结构化循环降为 cf（control flow）方言的显式分支：每个 scf.for 变成前置块、体块与出口块的组合，归纳变量与 iter_args 全部成为块参数，scf.yield 变成回边分支，循环出口汇合到后继块。这一步之后 IR 里只剩线性排列的基本块与跳转，结构信息（哪些跳转构成循环、循环携带什么）从 op 类型沉入块的拓扑。结构化形态在下降前的价值：scf.for 的上界、步长与携带变量都是操作数与类型系统里的显式事实，验证器保证单入口单出口，上层改写（第十八章的 linalg 展开、第二十一章的循环流水）可以依赖这些不变量做匹配；降成 cf 之后这类结构匹配就得靠模式识别块拓扑了。下降本身是机械的：L1 模块经全管线产出的 LLVM 方言文本 2088 行，可由 mlir-runner 直接 JIT 执行［scripts/run_l1.sh:26；evidence/G1/l1_harness_report.md:72-73］。

这次下降在整条管线里的位置：它排在 convert-linalg-to-loops 之后，所以 linalg 先展开成 scf.for、scf 再降成 cf——每个方言只在直接下层停留一次，两次下降合起来就是从声明式数学到可执行循环的完整路径。下一章讲这两个循环世界之下的内存世界：bufferize 如何把张量换成 memref。
