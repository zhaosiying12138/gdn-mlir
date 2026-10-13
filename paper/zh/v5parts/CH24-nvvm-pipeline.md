## 二十四、下译到 NVVM：gpu-lower-to-nvvm-pipeline 的旋钮

调度好的 IR 离硬件还有最后一层距离。L2 的 transform 工件解释完毕后，IR 里是 linalg、scf、nvgpu、nvvm 混合的 gpu.func；这些 op 仍然带着方言语义，GPU 无法直接装载。完成"方言 IR 到目标代码"这一步的，是 stock MLIR 的 gpu-lower-to-nvvm-pipeline。它把 GPU 方言的控制流与启动骨架、nvgpu 方言的机器 op、NVVM 方言的内在函数统一转换为 LLVM 方言的 IR，把 kernel 体从模块中勾勒（outlining）为 gpu.binary 目标对象，同时在主机侧留下 gpu.launch_func 调用点，最后交给 LLVM 的 NVPTX 后端产出汇编文本。本编译器在此处只做选择：chip 定哪一代硬件、features 声明哪一级指令集、format 决定产物以什么形态交出。三个旋钮拼成一行选项，位于整条命令链的最末端，也是 PTX 文本被抽出的位置——后处理层（第二十七章）的入口。冒烟整条管线的集成测试来自上游 sm_80（文件名沿用上游目录名，chip 重定向为 sm_120a 在 5090 上运行，见本章第 3 节）。

### 1. 完整命令链：从 payload 到 SASS

一条内核的编译由 scripts/run_l2.sh 驱动，分七步。先把 gdn 直通 op 下译为树内 op（gdn-opt 只解析 payload、不解析 transform 脚本，所以必须在合并之前做）：

```
if grep -qE 'gdn\.(setmaxnreg|elect|mbarrier|tma|bulk|mma)' payload.mlir; then
  "$GDN_OPT" payload.mlir --gdn-lower-hatch -o "$TMP/payload.gdn.mlir"
```
［scripts/run_l2.sh:41-43］

然后合并 payload 与 transform 工件、跑解释器；工件的调度动作（multibuffer、流水）在此刻固化进 IR。有一个例外写死在脚本里：恒等工件（不含任何 transform.structured/nvgpu/memref 动作）会整体跳过 -transform-interpreter——解释器会把裸顶层函数包进多余的 module 嵌套，而这层嵌套会让后续管线在 workgroup memref 上报 "conversion of memref memory space ... failed"（scripts/run_l2.sh:47-58 注释）。mma 关闭的路径还可能残留 linalg.matmul，补一遍 -convert-linalg-to-loops（63-64 行）。此后是本章的主句：

```
"$MLIR_OPT" "$TMP/sched2.mlir" -test-transform-dialect-erase-schedule \
  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a cubin-features=+ptx87 cubin-format=assembly" \
  -o "$TMP/asmobj.mlir"
```
［scripts/run_l2.sh:70-72，逐字］

erase-schedule 在前，把 transform 解释后残留的调度注记从 IR 上剥掉，让下译管线见到干净的函数体。产物 asmobj.mlir 里，PTX 以 gpu.binary 的 assembly 字符串属性存在，由 tools/gdn_extract_ptx.py 解转义写出 kernel.ptx（75 行；MLIR 把不可打印字节转义为反斜杠加两位十六进制，工具逐字符还原，tools/gdn_extract_ptx.py:5-8）。asmobj.mlir 本身也被留档——主机侧 gpu.launch_func 的实参清单是后处理第二项修复的输入（76-77 行注释）。最后 ptxas -v -arch=sm_120a 编译 cubin 并留下资源报告（81 行），nvdisasm -c 反汇编出 SASS（84 行），两者连同配置与特征直方图写进 meta.json（99-125 行）。

### 2. 三个旋钮的含义

cubin-chip=sm_120a 声明目标芯片。sm_120 是 Blackwell 消费级的计算能力编号；后缀 a 表示"架构加速"特性集——同代芯片有一个家族共有的指令子集，再加上仅本架构开放的加速指令（张量核新形态一类的架构条件指令）。不带后缀的 sm_120 只允许家族公共词汇，带 a 才解锁本架构专属词汇；ptxas 侧的 -arch=sm_120a 与之严格对应。本项目的硬件 RTX 5090 Laptop 报告即为 sm_120a，旋钮照抄设备事实。

cubin-features=+ptx87 向 NVVM 后端声明目标特性集合。后端把较新的 PTX 构造按 PTX ISA 版本分级门控，+ptx87 指定 8.7 层级的构造可用；sm_120a 的指令词汇恰好要求这一层级，缺省特性集会在部分下译路径上报不支持的构造。它与其余 +特性一样透传给 NVVM 的目标特性机制，不引入任何本项目私有拼写。

cubin-format=assembly 是三旋钮中承担架构决策的一个。该管线默认直接产出 cubin 二进制；assembly 改为产出 PTX 文本并挂在 gpu.binary 的 #gpu.object assembly 属性上。原因：本编译器在 PTX 定案之前还有一层后处理（第二十七章的四个变换全部发生在 NVVM 下译之后、ptxas 之前），必须以文本形态拿到 PTX；若走默认 cubin 路线，四个变换的介入点就不存在了。附带两个收益。其一，编译器版本可控——仓库统一选用 PTX ISA 9.3 能力的 ptxas-blackwell（scripts/run_l2.sh:34-36 的挑选顺序），把"哪个 ptxas 定案"从环境巧合变成受控输入。其二，asmobj.mlir 里主机侧 gpu.launch_func 的实参清单仍然在档，它是后处理中常量参数消除的输入——transform 解释与 hatch 下译会把 launch 区域内的常量提升回主机端，每个常量变成一个内核参数，最多可达约 140 个，而它们的值都能从这份清单恢复（tools/gdn_smem_fix.py:27-41 docstring）。

### 3. 管线冒烟：上游 sm_80 集成测试重定向 sm_120a

整条链（transform 解释 → nvgpu → nvvm → PTX → ptxas → cubin → 真机运行）在接入本项目内核之前，先用树内现成的集成测试冒烟：mlir/test/Integration/GPU/CUDA/TensorCore/sm80/transform-mma-sync-matmul-f16-f16-accum.mlir。scripts/smoke_sm80.sh 优先走 llvm-lit；装置不可用时转手动管线，把 chip 换成 sm_120a 后交给 mlir-runner 加载运行：

```
"$BUILD/bin/mlir-opt" "$T" \
  -transform-interpreter $ERASE \
  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a cubin-features=+ptx87" \
| "$BUILD/bin/mlir-runner" \
    --shared-libs="$BUILD/lib/libmlir_cuda_runtime.so,..." \
    --entry-point-result=void
```
［scripts/smoke_sm80.sh:37-43，节选（共享库清单删节）；$ERASE 为 erase-schedule 探测旗标，36-38 行］

sm_80 的测试能验证 sm_120a 管线，依据是同指令族：该测试从 transform dialect 驱动 nvgpu.mma.sync，经 NVVM 落成 mma.sync.aligned PTX，再由 ptxas 定案、真机装载比对结果——与本编译器 L3 内核走的是完全相同的下译路径；f16 输入、f32 累加的 mma.sync 指令族在两代芯片上拼写与片段布局一致，chip 旋钮只改变目标编码与架构条件指令的可用集。因此它冒烟的对象是工具链可用性：LLVM 构建、CUDA 运行时库、ptxas、装载执行这一整条机械。机械先于内核被证明可用，价值在调试分工——此后任何自产内核出问题，"管线本身坏了吗"可以从怀疑清单里一次划掉，注意力直接落在生成器或后处理；反之，升级 LLVM 主干后重跑一次冒烟，几秒钟内就能知道迁移是否破坏了下译链。冒烟通过后，全部 20 个配置、前向三 kernel 与反向 8-kernel 链在同一条机械上回归。

### 4. 效果

三个旋钮与七步命令链在全部 20 个 L2 配置、前向三 kernel 与反向 8-kernel 链上原样复用；每份证据目录（evidence/G4/gen_verify/*/ 等）都保存着同一条链的完整产物：payload.mlir、transform.mlir、asmobj.mlir、kernel.ptx、kernel_smemfix.ptx、ptxas 报告与 sass.txt。命令链本身零分支——分支（hatch 有无、工件恒等与否、linalg 残留与否）全部由对输入的显式检查承担并打印日志，保证同一目录重跑得到逐字节相同的产物，与第九章的确定性纪律衔接。
