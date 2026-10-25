## 十、SCF 与 CF：顺序、归约和循环携带值

结构化控制流（structured control flow，SCF）把循环、条件和携带值保留为 op 与 region。控制流（control flow，CF）方言则提供基本块、块参数和显式分支。两者都能表达 SSA 数据流，但前者直接保留循环边界与条件结构，后者把这些信息分配到块拓扑中。

### 1. 前代换中的值携带与内存携带

循环携带值（loop-carried value）是某一轮产生、下一轮继续使用的值。`scf.for` 用 `iter_args(%acc = %init)` 声明它：首轮的 `%acc` 来自 `%init`，当前轮通过 `scf.yield` 交给下一轮，循环结果则是退出时携带的值。若循环一次也不执行，结果就是初始值。前代换求 `T=(I+A)⁻¹` 的真实 IR 同时展示了这种值归约和跨行内存依赖：

```mlir
linalg.fill ins(%f0 : f32) outs(%Tbuf : memref<64x64xf32>)
scf.for %i2 = %c0 to %cC step %c1 {
  memref.store %f1, %Tbuf[%i2, %i2] : memref<64x64xf32>
  scf.for %j2 = %c0 to %i2 step %c1 {
    %sum = scf.for %k2 = %j2 to %i2 step %c1
        iter_args(%acc2 = %f0) -> (f32) {
      %aik = memref.load %Abuf[%i2, %k2] : memref<64x64xf32>
      %tkj = memref.load %Tbuf[%k2, %j2] : memref<64x64xf32>
      %pk = arith.mulf %aik, %tkj : f32
      %ak = arith.addf %acc2, %pk : f32
      scf.yield %ak : f32
    }
    %neg = arith.negf %sum : f32
    memref.store %neg, %Tbuf[%i2, %j2] : memref<64x64xf32>
  }
}
```

[evidence/G1/l1_bwd_chunked_T128_C64.mlir:2559-2573]

最内层要读 `%acc2 → %ak → scf.yield → %sum` 这条链。`%acc2` 是当前轮的块参数，`%ak` 是加上 `A[i,k]×T[k,j]` 后的新值，`%sum` 是整个 `scf.for` 的 SSA 结果；`scf.yield` 结束当前迭代的数据流，并非从函数返回。外层 `j<i` 保证内层至少有一个 `k` 点，归约从 `%f0` 开始，最终按 `T[i,j]=-Σ A[i,k]T[k,j]` 写回。

取 `i2=3、j2=1` 可以手算这条链。内层依次处理 `k2=1、2`，初始 `%acc2=0`；第一轮加入 `A[3,1]T[1,1]`，第二轮把这个中间和作为 `%acc2`，再加入 `A[3,2]T[2,1]`。退出后的 `%sum` 只包含这两个有序乘积，取负后写 `T[3,1]`。其中 `T[1,1]` 是更早写下的单位对角，`T[2,1]` 来自前一行。IR 中没有单独的“更新 acc”指令，yield 的操作数与下一轮的块参数完成了这个概念上的更新。

若把 `iter_args` 整个删除，`%acc2` 和 `%sum` 的值通道也随之消失，只剩每次从内存读取或重新定义中间值的实现方式。携带值的类型与数量必须同时匹配初始化列表、循环结果类型和 yield 操作数列表；任何一处不一致都会在 op 验证阶段暴露。零次执行时直接采用初值的规则，也使空归约无须另写条件分支。

嵌套 region 让外层归纳变量 `%i2`、内层 `%j2` 直接出现在更深的循环体中。region 内新定义的 SSA 值受作用域约束，跨出循环需要 op 结果；内存内容则通过 load/store 留下效果。`%Tbuf` 的 memref 句柄始终是同一个 SSA 值，其指向的内容随外层逐行更新。读取 `T[k,j]` 时 `k<i`，所依赖的行已经在较早的外层迭代完成，所以前代换的行顺序属于正确性条件。

三个循环轴的依赖并不相同。外层 `i2` 不能把第 3 行移到第 2 行之前，因为第 3 行会读取第 2 行。固定 `i2` 后，不同 `j2` 写入不同的 `T[i,j]`，它们读取的是更早行以及本行输入 A，并未通过当前行的另一个 `j` 形成同类递推。最内层则通过 `%acc2` 明确串起浮点加法的值链。循环嵌套深度本身不回答可交换性，必须按读集、写集和携带值分别画依赖。

这条区分也限制常见改写。交换外层 `i2` 与中层 `j2` 会改变上界 `%i2` 的含义，还会破坏跨行可用性；重排最内层虽然保持实数求和项集合，却改变 f32 舍入次序。对固定行的不同列做并行化有另一种依赖依据，同时仍要处理外层状态与内存竞争。读 IR 时先找跨迭代路径，再讨论调度，比按 `scf.for` 的文本层级猜测安全。

### 2. 没有 iter_args 也可能携带状态

门控前缀和没有 `iter_args`，状态保存在 `%gc` 指向的内存中：

```mlir
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

[evidence/G1/l1_bwd_chunked_T128_C64.mlir:2514-2522]

读这段循环要沿地址追踪依赖：第 `i` 轮读取 `%gc[i-1]`，加上输入 `%g_c[i]`，再写 `%gc[i]`。句柄没有逐轮产生新版本，内存内容承担了携带作用。把它改写为标量 `iter_args` 也能表达同一数学递推，但还需要在每轮保存 `%gc[i]` 供后续矩阵计算使用。两种写法区分了 SSA 值依赖与内存读写依赖，优化器需要分别检查。

顺序还影响浮点结果。前代换的内层按 `k=j,j+1,…,i-1` 依次执行 `addf`，前缀和按 `i` 递增；`convert-scf-to-cf` 会保留这个求值顺序。f32 加法不满足结合律，交换归约顺序、树形归约或重新关联乘加可能改变舍入和溢出表现。数学等价的循环交换需要同时核对依赖方向与数值容差，不能只看实数公式。

### 3. for、if、while 与基本块下降

`scf.for` 表达带下界、上界和步长的计数循环，`iter_args` 与结果类型组成携带值合同。`scf.if` 用 `i1` 条件选择 then 或 else region，各分支用 `scf.yield` 产生统一的 op 结果；若只进行内存写等副作用，也可以没有结果。`scf.while` 适合退出条件需要在迭代中计算的循环，其 before region 计算条件并用 `scf.condition` 决定继续或退出，after region 执行循环体并用 `scf.yield` 更新携带值。三者都把控制结构留在 op 内，支配关系和 region 嵌套可直接检查。

带结果的 `scf.if` 让两个分支在同一结果类型上汇合，无须先分配临时内存再于分支外 load。`scf.while` 的 before region 可以先检查当前携带值，条件为假时不进入 after region；after region 产生的值经 yield 回到条件计算。阅读 while 时要把 `scf.condition` 的布尔操作数与其余携带操作数分开，前者选择控制方向，后者维持数据流。

`convert-scf-to-cf` 后，计数循环被拆成条件判断、循环体、回边与出口所在的基本块。归纳变量和 `iter_args` 成为块参数，`scf.yield` 变成带参数的 `cf.br`，条件成为 `cf.cond_br`。`scf.if` 同样下降成分支与汇合块，分支结果改由汇合块参数接收。此时仍有 SSA 和支配关系，但“哪些块共同构成一个 for”需要从拓扑恢复；循环上界、步长与结构化 region 的直观匹配已经消失。适于循环结构的分析和改写应在 SCF 层完成，LLVM 下译前的显式分支则留给 CF。

CF 块参数表示控制流边上的值传递。回到同一块时，参数可来自前一轮的 yield；从入口首次进入时，参数则来自初始化值。`cf.br ^body(%x)` 与函数调用不同，它不建立新的函数栈帧，也不隐式保存循环状态。`cf.cond_br` 只根据条件选择后继并传递各自的操作数，是否形成循环取决于 CFG 中是否存在回边。

这次下降主要换写控制结构，不会主动把两条 `addf` 改成另一种归约树。后续 canonicalization、循环变换或带快速数学许可的优化仍需分别审查。将数值漂移归因于 SCF 到 CF 之前，应先对照两边的分支、块参数和算术顺序，确认实际发生改写的阶段。

### 4. 【通用补充】iter_args 不建立异步可见性

下面的伪 IR 省略 async 运行时的完整类型，保留关键依赖。异步 region 写 `%buf`，顺序循环在没有等待 token 的情况下读取它：

```mlir
%token = async.execute {
  memref.store %one, %buf[%c0] : memref<1xf32>
  async.yield
}
%sum = scf.for %i = %c0 to %n step %c1
    iter_args(%acc = %zero) -> (f32) {
  %x = memref.load %buf[%c0] : memref<1xf32>
  %next = arith.addf %acc, %x : f32
  scf.yield %next : f32
}
// 读取前缺少 async.await %token 及目标要求的同步
```

`%acc` 只把浮点值从一轮交给下一轮，它没有表示异步写何时完成，也没有给 `%buf` 的 load/store 添加 happens-before 关系。循环可能重复读到旧值，或与异步写并发；把 token 加进 `iter_args` 同样只增加一个 SSA 携带值，不能替代等待。正确实现需要在读取前等待相应生产操作，并按目标内存模型配置 fence、barrier 或代理同步。

普通 memref load/store 只表达访问，不附送跨执行流的完成关系。原子操作可以为特定访问提供原子性，其顺序与可见范围仍要服从目标内存模型，不能自动修复这里缺失的等待。分析这类反例时应分别列出值依赖、内存依赖和同步依赖，三条链需要各自闭合。

内存句柄、视图和值结果在下降到这里后已经清晰可见。第十一章将检查 tensor 缓冲化怎样决定分配、拷贝和别名，以及这些 memref 操作怎样继续下降到 LLVM。
