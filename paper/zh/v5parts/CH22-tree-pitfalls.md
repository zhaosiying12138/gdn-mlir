## 二十二、树内 pass 的反直觉行为实录

树内机制在真实构建中的实际行为与文档语义有出入：七条被撞上、以探针或源码定位、并已固化为对策的行为，每条按现象、根因、对策陈述，出处指向仓库证据文件。它们直接解释了本编译器若干设计选择的由来：异步 staging 为什么手动发射、staging 为什么线程跨步、某些循环为什么写成 scf.while。

### 1. create_async_groups：每组之后立即等待

现象：经 transform.nvgpu.create_async_groups 转换的 staging，PTX 里每组 cp.async 之后紧跟着 cp.async.wait_group 0，复制与计算结构性无法重叠；按此路线实现的 chunk 预取乒乓实测零收益（F2 计时 471 对 473 微秒）。根因在树内源码 NVGPU/Transforms/CreateAsyncGroups.cpp：组的创建与等待成对发射，等待不留给使用者摆放。对策：异步 staging 一律手动发射 nvgpu.device_async_copy——解释器不触碰这类 op，等待的位置归生成器决定，重叠由此成为可编排的自由度［evidence/G6/failure_longscroll.md:40-56；evidence/G5/fwd_report.md:330-335，源码路径在证据中记录］。

### 2. block-uniform staging：256 份重复搬运

现象：把"每条拷贝指令由全线程块共同发射"的 block-uniform staging 形态从阶段级微内核移植到全算子 K 循环后，staging 前向比移植前慢一倍——256 个线程各自把同一批 cp.async 发了一遍，搬运量 256 倍冗余［evidence/G6/failure_longscroll.md:11-22］。根因是规模效应：阶段级微内核只有 4 个 warp，全员发射的冗余可承受；全算子规模下同样的形态成为毒药。旋钮本身没有错，错在把它带出了适用规模——两级消融（阶段级与全算子级各测一遍）因此成为必需的验证结构。对策：staging 一律 thread-strided，每线程按 tid 跨步认领自己的 16 字节块——机器层发射器的 astage_tile 即按此纪律书写，地址按线程跨步加 16 字节步进、换行加 256［tools/gdn_bwd_gen.py:1687-1714］。

### 3. pipeline_shared_memory_copies failures(suppress)：把好循环改坏

现象：文档声明的逃生舱 failures(suppress) 本应让流水器容忍非拷贝循环，实际 -transform-interpreter 的输出在任何非拷贝 scf.for 上损坏——用一个 G2 payload 加哑循环的探针证实。对策分两层。其一，全部非 K 单元 pass 与 F2 的 chunk 循环改写为 scf.while：match 的 ops{["scf.for"]} 圈不到 while，流水器因此看不见它们。while 的书写有三条纪律——do 区域需要显式 ^bb0 头与全新参数名、条件必须 yield 全部参数、参数与结果类型一致。其二，同源的多缓冲问题：被多个循环使用的缓冲（S、vn、utile、gcs）触发 multibuffer 硬失败（找不到唯一候选循环），对策是 pin_alloc——一行不在任何循环里的哑 load 加 store，让该分配脱离圈定范围［evidence/G5/fwd_report.md:222-230；evidence/G6/failure_longscroll.md:24-38］。

### 4. LDGSTS 到 LDSM：异步拷贝后经 ldmatrix 读回的静默损坏

现象：用 device_async_copy（SASS 为 LDGSTS.E.BYPASS.128）staging 的 mma A tile，经普通 ld.shared 读回逐位正确（对全局源 dump 核对过），经 nvgpu.ldmatrix 读回错误——错的行块子集与构建调度相关（2-mma 构建错 32-63 行，3-mma 构建错 48-63 行），逐构建确定、对保真刷新不变；数值门以 dk 1.34、dv 2.18、dbeta 25.7 的超差抓到，而与该 tile 无关的 dq、dh0 保持原值。同一 tile 换 st.shared 写入，经完全相同的 ldmatrix、mma、epilogue 机件读回逐位正确（5.2e-8）。根因定位于本 sm_120a 工具链手动 cp.async 路径的静默失败模式。对策固化为发射纪律：raw cp.async 的字节只允许落在"读回者是标量 ld.shared"的 tile；一切喂 ldmatrix 的 tile 经 st.shared 逐元素写入，或经 laundering——一次恒等的 ld.shared 加 st.shared 重写，恰好覆盖 staging 单元［evidence/G5/bwd_report.md:945-965、2870-2877］。

### 5. linalg.matmul outs(f32) 自动提升，显式 generic 加 cast 反而断管线

现象：nomma 路径需要 bf16 操作数配 f32 累加。linalg.matmul ins(bf16,bf16) outs(f32) 在本树自动提升精度——下降出的循环体是 extf 提升操作数、f32 乘加、f32 写回（对 -convert-linalg-to-loops 的产物核对）；改写成自文档性更强的 linalg.generic 加显式 cast 块，却让 gpu-lower-to-nvvm-pipeline 在 workgroup memref 上报 "conversion of memref memory space ... failed"，二分定位到 generic op 本身。对策：保留 named op 拼写，累加器加宽交给其自动提升，payload 的拷贝与 staging 结构保持字节不变。教训的方向与直觉相反：在这条 NVVM 管线上，更"显式"的写法更不兼容［evidence/G4/GATE_REPORT.md:150-165］。

### 6. cubin-format=assembly：抽出 PTX 的唯一简路

现象：NVVM 管线默认产出 fatbin 形式的 gpu.binary，PTX 文本取不回来，而后续工序——PTX 后处理、ptxas-blackwell 资源报告、指令族覆盖统计——都以 PTX 文本为原料。对策链：给 -gpu-lower-to-nvvm-pipeline 附 cubin-format=assembly，PTX 以 #gpu.object 的 assembly 属性挂在 gpu.binary 上，工具解掉 MLIR 字符串的十六进制转义即得 kernel.ptx。备选路线都更差：cubin-format=ptx 被本构建直接拒绝（合法值为 offloading、llvm、assembly、isa、binary、bin、fatbinary、fatbin）；默认 fatbin 加 cuobjdump -ptx 可行，但要先从 .mlir 属性里抠出二进制，严格更繁。发射出的 PTX 标头为 .version 8.7 与 .target sm_120a，普通 ptxas 13.3 只认到 8.8，编译必须用 ptxas-blackwell［evidence/G2/gen_report.md:20-36；scripts/run_l2.sh:69-75、125-126；evidence/G0/GATE_REPORT.md，过程记录第 6 条］。

### 7. 解释器的额外嵌套破坏后续管线

现象：对只含裸顶层函数的 payload 运行 identity 工件（只 yield、无任何动作），-transform-interpreter 会把函数包进一层 module{}；这层额外嵌套使 gpu-lower-to-nvvm-pipeline 在 workgroup memref 上转换失败。对策：编译脚本按工件内容探测——工件里没有 transform.structured、nvgpu、memref 动作时整步跳过解释器，payload 原样进入下一 pass。同一探测逻辑顺带处理直通 op：payload 含 gdn.setmaxnreg、gdn.tma 等 op 时，先经 gdn-opt --gdn-lower-hatch 降为树内 op 再与工件合并（gdn-opt 能解析 payload 却不解析 transform 脚本，顺序由此固定）［scripts/run_l2.sh:38-59，注释与两个分支逐行可查］。

### 8. 七条的公共形态

树内机制在文档语义与实际行为之间留有缝隙：等待的位置、发射的冗余度、循环改写的边界条件、格式的接受域。这些缝隙无法从文档推知，只能以真实构建加探针测绘。对策在本编译器里全部沉淀为生成器纪律或管线分支。
