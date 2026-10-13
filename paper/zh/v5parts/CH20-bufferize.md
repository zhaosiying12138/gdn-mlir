## 二十、bufferize：从张量世界到 memref 世界

MLIR 的上层方言以 tensor 为通货。tensor 是值语义的类型：一个 op 的输出是一整个不可变的值，函数边界传值，别名这个概念不存在——写第 i 行不会影响读到第 j 行的人。机器执行要的却是位置语义：一块实际内存、一组下标、就地读写。bufferize（缓冲化）就是从前者到后者的翻译工序。它的存在意义在于让上层只管数学、下层只管内存，两边的契约由一次系统性的改写衔接。

### 1. tensor 与 memref：两种世界观

tensor<2x128x2x16xf32> 只描述元素类型与形状；memref<64x64xf32, strided<[64, 1], offset: ?>> 还携带布局——步长（每维走一格跳多少元素）、偏移（起点距基地址多远）。L1 展品里的 memref.subview 产出的 strided<[32,1]>、strided<[2]> 类型正是布局信息跟着视图走：切 4 维张量的一块，剩余维的步长继承自源。值语义与位置语义的差别在改写时机上最要紧：tensor 上"先算 A 再算 B"与"穿插着算"只要数据流相同就等价，memref 上则牵动读写集与生存期。bufferize 是这两个世界之间唯一的一次大规模翻译。

### 2. one-shot-bufferize 做什么

one-shot-bufferize 以单趟遍历把 tensor 上的 op 翻译为 memref 上的等价 op：函数签名里的 tensor 参数换成 memref（由 bufferize-function-boundaries 选项开启），需要物化的中间张量自动插入 memref.alloc，发生破坏性写回而值仍需保持原样时自动插入 copy。得名于"一趟"：相对旧式逐方言、逐 op 的缓冲化 pass 链，它对整张张量数据流做一次性分析，据此插入最少的拷贝。

本仓库的 L1 管线里有一个如实的细节：生成器发出的模块本来就是 memref 级的（memref.subview、memref.load/store 直接出现，数据以 memref.global 常量内嵌），这条 pass 在当前 IR 上空转，保留它出于管线兼容——将来基于 tensor 的前端可以原样接入同一条命令行［evidence/G1/l1_harness_report.md:60-62；scripts/run_l1.sh:24］。换句话说，本编译器把"张量世界"留给了语义层（L0 的 gdn.fwd 吃 tensor），L1 直接在 memref 上书写算法，bufferize 在管线里的位置成为面向未来的插槽。

### 3. memref.alloc 的 SSA 语义：每 tile 一次分配

memref.alloc 在 IR 里是一个普通 op，结果是指向新分配缓冲的 SSA 值。SSA 体系只跟踪值，不跟踪内存何时可复用：两个 alloc 只要都活着就默认指向不相交的内存，通用改写无权把后来的分配折叠到先前分配的字节上——跨 alloc 的生存期装箱在 memref 层没有词汇。生成器因此为每个中间 tile 各发一条独立分配，前向 kernel 的分配序列开头八条：

```
e(f"%S = memref.alloc() : {t_KV}")
e(f"%Abuf = memref.alloc() : {t_CC}")
e(f"%Tbuf = memref.alloc() : {t_CC}")
e(f"%kkt = memref.alloc() : {t_CC}")
e(f"%qk = memref.alloc() : {t_CC}")
e(f"%gc = memref.alloc() : {t_C}")
e(f"%upre = memref.alloc() : {t_CV}")
e(f"%ubuf = memref.alloc() : {t_CV}")
```
［tools/gdn_l1_gen.py:380-387；其后 kb2/wp/vp/vn/qg/kg/oib/oob/prod 八条到 395 行，共十六个 tile］

这在 CPU 上无害——十六个小缓冲对堆不构成压力。到 GPU 上它成为真实的物理申报：workgroup 内的 memref.alloc 落到共享内存池，每条 alloc 都是一笔静态字节。当一个 kernel 的算法 tile 生存期拼接远超每 SM 驻留门槛、而任意时刻同时活跃的字节在门槛之下时，"生存期不重叠的 tile 共享同一批字节"这个物理事实在 memref 层无法表达——它只能推迟到 PTX 层，以生成器产出的布局表加地址改写施加（第三十五章的生存期别名正是做这件事，布局表的样例见 evidence/G5/gen_fwd/B4T2048_mono/smem_alias.json）。此处只需记住因果链：alloc 的 SSA 语义要求每 tile 一次分配，合法的字节共享必须后置到机器代码层。

### 4. 11 pass 管线：每条 pass 的职责

CPU 侧的完整下降管线是 scripts/run_l1.sh 的真实命令行：

```
"$OPT" "$IN" \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -expand-strided-metadata \
  -lower-affine \
  -convert-math-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-cf-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o "$OUT"
```
［scripts/run_l1.sh:23-35，逐字；$OPT 为构建目录下的 mlir-opt，19 行］

各条 pass 的职责一句话说清：

| pass | 职责 |
|---|---|
| one-shot-bufferize | tensor 到 memref 的整体缓冲化（本模块上空转，见第 2 小节） |
| convert-linalg-to-loops | 结构化 op 实体化为循环（第十八章第 4 小节） |
| convert-scf-to-cf | 结构化循环降为显式分支（第十九章第 4 小节） |
| expand-strided-metadata | 把 strided 布局拆成 size、stride、offset 基元 op |
| lower-affine | 仿射算术（affine.apply 等）降为普通 arith 运算 |
| convert-math-to-llvm | math.exp2 等超越函数降为 LLVM intrinsics |
| convert-arith-to-llvm | arith 的整数与浮点运算降为 LLVM 指令 |
| finalize-memref-to-llvm | memref 描述符、alloc 与 dealloc 降为 LLVM 调用 |
| convert-cf-to-llvm | cf 分支降为 LLVM 基本块与跳转 |
| convert-func-to-llvm | func 函数降为 LLVM 函数 |
| reconcile-unrealized-casts | 清理方言间翻译遗留的过渡转型 op |

排序有实质约束：expand-strided-metadata 与 lower-affine 对生成器发出的 strided 降秩 subview 是必需的，缺失即编译失败，配方取自树内集成测试 rank-reducing-subview.mlir；math.exp2 经 convert-math-to-llvm 变成 llvm.intr.exp2，由 JIT 对着共享运行库带进来的 libm 解析，无需先经 libm 方言中转［scripts/run_l1.sh:7-13 注释；evidence/G1/l1_harness_report.md:51-58］。

### 5. mlir-runner：JIT 执行

mlir-runner 把 LLVM 方言模块 JIT 编译为机器码并调用入口函数。执行需要三个共享运行库——runner_utils、c_runner_utils、float16_utils——它们提供 memref 打印原语（printMemrefF32）与 libm 链接：

```
"$RUNNER" "$OUT" --shared-libs="$LIBS" --entry-point-result=void
```
［scripts/run_l1.sh:37，逐字；$RUNNER 与 $LIBS 的三个 .so 清单在 20-21 行］

基础模块上的开销：mlir-opt 0.32 秒、mlir-runner 0.22 秒（JIT 加执行），下降后的 LLVM 方言文本 2088 行［evidence/G1/l1_harness_report.md:72-73］。打印出的 o 与 ht 由独立检查器对拍，误差落在打印精度地板约 1e-6 量级，七个用例全过——至此"数学即 IR"在 CPU 上闭环：第十八、十九章的掩蔽与前代换在普通处理器上跑出与 JAX fp64 规约一致的数字，算法层的正确性不再依赖任何 GPU 工具链。
