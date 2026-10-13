## 十八、linalg：以控制流表达数学约束

算法层需要一种两头都成立的表示：对验证者，它要像数学一样可以逐行核对；对编译器，它要像程序一样可以逐层改写。linalg（linear algebra，MLIR 的结构化线性代数方言）正是为此存在的中间层。它的核心主张是：一个数学对象——矩阵乘、逐元素映射、带区域约束的复合运算——可以表达为一个结构化 op，迭代空间、下标映射与计算体三件东西声明在 op 的属性与区域里，op 之间保持普通的 SSA 数据流。以 GDN 反向链中 KKT 相关矩阵的构造为教材，linalg.generic 的每个部件可以逐一拆解；一个容易被忽视的用法是三角掩蔽以显式控制流的形式生活在计算体内，比较谓词的选择本身就是数学定义的一部分。

### 1. generic op 的五个部件

linalg.generic 是该方言的万能形式，named op（linalg.matmul、linalg.fill、linalg.copy 等）都可以看作它的预包装拼写。一个 generic 由五个部件构成：indexing_maps（下标映射表）、iterator_types（迭代空间性质表）、ins 与 outs（输入值与输出缓冲）、一个 ^bb0 计算块、以及块末的 linalg.yield。先看 A 矩阵构造的前置步骤——Gram 矩阵 kkt = k·kᵀ，64×64 输出的每个元素是两个 16 维键向量的内积：

```
linalg.fill ins(%f0 : f32) outs(%kkt : memref<64x64xf32>)
linalg.generic {
  indexing_maps = [
    affine_map<(i, j, r) -> (i, r)>,
    affine_map<(i, j, r) -> (j, r)>,
    affine_map<(i, j, r) -> (i, j)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%k_c, %k_c : memref<64x16xf32, strided<[32, 1], offset: ?>>, ...) outs(%kkt : memref<64x64xf32>) {
^bb0(%a0: f32, %a1: f32, %o0: f32):
  %p = arith.mulf %a0, %a1 : f32
  %s2 = arith.addf %o0, %p : f32
  linalg.yield %s2 : f32
}
```
［evidence/G1/l1_bwd_chunked_T128_C64.mlir:2524-2537，类型串有删节；fill 为归约清零，见第 4 小节］

iterator_types 声明三维迭代空间 (i, j, r) 每一维的性质：parallel 维的迭代点彼此独立，reduction 维对同一输出元素累加。indexing_maps 为 ins 与 outs 的每个操作数各配一条 affine_map（仿射映射），把迭代空间的一个点翻译成该操作数里的一个元素地址。第一条映射 (i,j,r)→(i,r) 取左操作数第 i 行第 r 列，第二条 (i,j,r)→(j,r) 取右操作数第 j 行第 r 列——两个操作数是同一块 k_c，两条映射读法不同，转置由此表达；第三条 (i,j,r)→(i,j) 定位输出。^bb0 块的参数按序对应每条映射扫出的标量：两个 f32 输入与一个 f32 输出；块体只有乘与加两条运算，linalg.yield 把结果交还给归约维的累加器。开头的 fill 也属于这个机制：归约的初值取 outs 缓冲的当前内容，先清零才能从 0 起累（第 4 小节展开这条纪律）。矩阵乘的全部数学——转置与累加——都被压进这三条映射与一个块体，没有离开 IR 的部分。

### 2. 广播：低维操作数进入高维迭代

indexing_maps 的右端维度数低于迭代空间时，同一元素会被多个迭代点读到，这是结构化 op 内建的广播语义。A 矩阵的构造 generic 用五条映射、四个输入，其中三条是降维读法：

```
linalg.generic {
  indexing_maps = [
    affine_map<(i, j) -> (i, j)>,
    affine_map<(i, j) -> (i)>,
    affine_map<(i, j) -> (j)>,
    affine_map<(i, j) -> (i)>,
    affine_map<(i, j) -> (i, j)>
  ],
  iterator_types = ["parallel", "parallel"]
} ins(%kkt, %gc, %gc, %beta_c : memref<64x64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32, strided<[2], offset: ?>>) outs(%Abuf : memref<64x64xf32>) {
^bb0(%a0: f32, %a1: f32, %a2: f32, %a3: f32, %o0: f32):
  %d0 = arith.subf %a1, %a2 : f32            // gc_i − gc_j
  %e0 = math.exp2 %d0 : f32                  // 2^(gc_i−gc_j)
  %m0 = arith.mulf %a0, %a3 : f32            // (k·kᵀ)_ij · β_i
  %m1 = arith.mulf %m0, %e0 : f32
  %i0 = linalg.index 0 : index
  %j0 = linalg.index 1 : index
  %lt = arith.cmpi slt, %j0, %i0 : index     // j < i：严格下三角
  %r0 = arith.select %lt, %m1, %f0 : f32     // 三角外归零
  linalg.yield %r0 : f32
}
```
［evidence/G1/l1_bwd_chunked_T128_C64.mlir:2538-2558，逐字（排版有折行）］

(i,j)→(i) 与 (i,j)→(j) 把一维的 gc（chunk 内累积门控）分别按行、按列广播进二维迭代：gc_i 减 gc_j 给出衰减因子 2^(gc_i−gc_j)，β_i 乘上 Gram 矩阵的对应元素。一个 op 同时吃矩阵、行向量、列向量，无需任何展开或复制。linalg.index 取当前迭代点坐标，是计算体感知位置的正规通道；掩蔽正是靠它实现。

### 3. 掩蔽即控制流：一谓词之差

掩蔽（mask）指按位置条件把部分输出元素置零。一种惯用做法是先物化一个 0/1 张量、再逐元素相乘；这份 IR 把条件直接写进计算体：cmpi slt 比较下标，select 归零。两个矩阵的掩蔽只差一个谓词——A 用 slt（j<i，严格下三角），含对角的衰减重乘矩阵用 sle（j≤i）：

```
%i3 = linalg.index 0 : index
%j3 = linalg.index 1 : index
%le = arith.cmpi sle, %j3, %i3 : index
%r5 = arith.select %le, %m4, %f0 : f32
linalg.yield %r5 : f32
```
［evidence/G1/l1_bwd_chunked_T128_C64.mlir:2848-2852，含对角掩蔽的构造块尾］

谓词即数学定义：A = strict_tril(β⊙(k·kᵀ)⊙2^(gc_i−gc_j)) 与含对角的版本在论文里差一个"严格"，在 IR 里差两个字母。不物化 mask 张量有两个理由。内存上，64×64 的掩码张量本身不大，但反向链里每个 chunk、每个矩阵都要构造一遍，写一遍读一遍的流量与生命周期管理是纯开销。可审计性上，条件与运算同居一个块内，验证者逐行读 IR 时无需追踪另一个张量的定义域与覆盖范围——三方交叉验证把 MLIR 输出对到 1e-6 量级时，这种直接可见性决定排查效率。掩蔽因此以控制流的形式存在于 IR 里：编译器看得见它，验证者也看得见它，下层任何改写都无法悄悄碰它。

### 4. 下降：convert-linalg-to-loops 做了什么

generic 是声明，机器要的是循环。-convert-linalg-to-loops 把每个结构化 op 的迭代空间实体化：parallel 维按声明顺序展开为外层 scf.for，reduction 维成为最内层带累加变量的归约循环，初值取 outs 缓冲的当前内容，块体内联为循环体，yield 的值经循环结果写回输出。下降是机械的，这带来一个有用的副产品：审查 named op 的语义可以借道这条下降。GDN 的 nomma 路径曾需要确认 linalg.matmul ins(bf16,bf16) outs(f32) 的累加精度，做法就是运行该 pass 后检查循环体——extf 提升操作数、f32 乘加、f32 写回，named op 的自动提升由此坐实［evidence/G4/GATE_REPORT.md:157-159］。在 CPU 管线里，这条下降紧随 bufferize 之后执行［scripts/run_l1.sh:25］。

### 5. 效果

用 linalg 表达算法层的收益由验证结果度量：整条 L1 反向链（含本章全部展品）经第二十章的 11 pass 管线在 CPU 上可执行，对 JAX fp64 规约前向约 6e-7、反向六梯度最差 6.4e-6，三方闭合。数学约束——下三角、衰减因子、行尺度——全部以 IR 可见的形式存在；任何下层改写破坏语义时，都能回到这份 CPU 可执行的文本重新对照。下一章讲循环本身：scf.for 的数据流如何承载前代换。
