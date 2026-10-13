## 二十一、transform dialect：调度作为可编程工件

传统编译器里，调度决策（怎么切块、哪级缓冲、流水多深）住在 pass 的 C++ 实现里，使用者能动的只有命令行旗标。transform dialect 把这件事倒过来：对 IR 施加什么改写，本身用 IR 表达。一个 transform 脚本引用 payload（被改写的普通模块）里的 op，对它们施加切块、多缓冲、软件流水等动作。调度由此从隐式知识变成显式工件——可 diff、可存档、可参数化生成。本编译器把每个 GPU 配置表达为 payload 加 transform 工件加旋钮值的三元组。

### 1. 工件与 payload：两种 IR 的分工

transform 脚本自身是一个 MLIR 模块，与 payload 各占一份文件、互不混写：

```
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    ...
    transform.yield
  }
}
```
［evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/transform.mlir:1-3、13，骨架］

module 级属性 transform.with_named_sequence 声明本模块携带命名序列；@__transform_main 是解释器的固定入口；%arg0 绑定 payload 的顶层 op，类型 !transform.any_op 表示"任意 op 的句柄"；transform.readonly 标记 payload 对外的接口只被读取。工件里其余每个 % 值都是句柄（handle）——transform 方言的一等值，类型标注它指向哪类 op，比如 !transform.op<"memref.alloc">。这份分离让"计算是什么"与"计算怎么铺"成为两个独立演化的文本。

### 2. 目标选择：transform.structured.match

工件的第一个动作几乎总是圈定目标。match 按 op 名在 payload 里收集实例，返回句柄：

```
%allocs = transform.structured.match ops{["memref.alloc"]} in %arg0
  : (!transform.any_op) -> !transform.op<"memref.alloc">
%func = transform.structured.match ops{["func.func"]} in %arg0
  : (!transform.any_op) -> !transform.any_op
%loops = transform.structured.match ops{["scf.for"]} in %arg0
  : (!transform.any_op) -> !transform.any_op
```
［evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/transform.mlir:3-4、7-8、10-11，三处 match 合排］

后续动作全部作用于句柄，于是"调度作用于谁"在工件文本里显式可见。一个实操细节：create_async_groups 会改写函数体，使指向旧循环的句柄失效，其后必须重新 match——仓库的工件因此在异步组建之后重新圈定 scf.for［evidence/G2/gen_report.md:121-123；evidence/G2/GATE_REPORT.md:33］。

### 3. 四个调度动作的语义

完整工件（kkt 阶段、bf16 加 mma、流水深度 2 的真实配置）串联三个动作：

```
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %arg0
      : (!transform.any_op) -> !transform.op<"memref.alloc">
    transform.memref.multibuffer %allocs <factor = 2, skip_analysis>
      : (!transform.op<"memref.alloc">) -> !transform.any_op
    %func = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.create_async_groups %func bypass_l1
      : (!transform.any_op) -> (!transform.any_op)
    %loops = transform.structured.match ops{["scf.for"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.pipeline_shared_memory_copies failures(propagate) %loops depth = 2 peel_epilogue
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
```
［evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/transform.mlir:1-15，逐字］

multibuffer <factor = 2, skip_analysis> 把句柄指向的每个 memref.alloc 改写为 factor 份副本轮转——factor=2 即双缓冲，一份数据在用时另一份在装；skip_analysis 跳过"哪个循环是热循环"的自动推断，按使用位置直接轮转。create_async_groups 把函数里全局内存到共享内存的拷贝打包成异步复制组；bypass_l1 选择 cp.async.cg 拼写（绕过一级缓存），省略该旗标则发射 cp.async.ca。pipeline_shared_memory_copies 对句柄循环做软件流水：depth = 2 即两段，peel_epilogue 把尾趟剥出循环以排空流水，failures(propagate) 声明无法流水的循环把错误上抛、不做静默跳过。

nomma 路径的工件展示第四个动作——切块：

```
%mms = transform.structured.match ops{["linalg.matmul"]} in %arg0
  : (!transform.any_op) -> !transform.any_op
%tiled, %loops:2 = transform.structured.tile_using_for %mms tile_sizes [16, 16]
  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
```
［evidence/G4/gen_verify/kkt_bf16_nomma_p0_t64x64x32_w4/transform.mlir:3-6，逐字］

tile_using_for 以 scf.for 为载体把命中的 matmul 切成 tile_sizes 的块，返回切后的 op 与外层循环两个句柄。2026-10 主干的拼写以树内测试为准：tile_using_for（旧文献里的 tile 已更名）、multibuffer 的尖括号参数表、pipeline 的 failures 子句，拼错即 verifier 拒绝［evidence/G2/GATE_REPORT.md:33；evidence/G2/gen_report.md:109-123］。

### 4. 十维旋钮到 IR 的逐维落点

生成器 tools/gdn_l2_gen.py 是旋钮值到 IR 文本的唯一通道：配置字典先过 validate() 核对取值域［tools/gdn_l2_gen.py:70-138］，再流入两组 f-string 模板——载荷模板发射 payload，工件模板发射 transform 脚本。十维的规范清单来自求解器的变量声明：dtype、prec、tile 三维、warps、pipeline、mma、warpspec、tma，进算术的用整数、纯开关用布尔［tools/gdn_solver.py:99-110；域常量 70-74］。生成器文档串对这组旋钮的可观测性有一条自我声明：每个旋钮都可验证地改变发射的 PTX［tools/gdn_l2_gen.py:15-28］。逐维落点如下表（payload 行号均出自第 5 节走查的目录 evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/）。

| 旋钮 | 取值域 | 落在 IR 的位置 | 例 |
|---|---|---|---|
| dtype | bf16、f32 | 载荷全链类型：函数参数与全局、共享 memref 的元素类型，拷贝向量宽度，mma fragment 向量形状 | bf16 发 memref<64x128xbf16> 与 vector<8xbf16>［gdn_l2_gen.py:156-157、425；payload.mlir:2、261］ |
| prec | bf16、tf32（仅 f32 加 mma 时独立） | mma 原子的 K 边与 nvgpu.mma.sync 的 tf32Enabled 属性 | bf16 发 mmaShape [16,8,16]，tf32 发 [16,8,4] 加 tf32Enabled［gdn_l2_gen.py:160-161、303-305］ |
| tile.M | 32、64、128 | gpu.launch 网格 x 维（64 行按 M 分块）、A tile 行数、mma 行组数 | M=64 时网格 x 维为 1［gdn_l2_gen.py:1011-1013；payload.mlir:69］ |
| tile.N | 32、64、128 | 网格 y 维、B tile 的 N 边、每 warp 分得的 N 份额 | N=64 时 B tile 两条边为 64 与 32［gdn_l2_gen.py:423-424；payload.mlir:215］ |
| tile.K | 32、64、128 | K 循环步长、smem tile 的 K 边、每趟展开的 mma 原子数 | 步长 32 即 tile.K［gdn_l2_gen.py:438-439；payload.mlir:256］ |
| warps | 2、4、8 | gpu.launch 线程 x 维（warp 数、每 warp 32 线）、每 warp N 份额、累加器个数与 warp 列偏移 | 4 warp 时线程 x 维 128、warp 列偏移步进 16［gdn_l2_gen.py:997、1013、1076；payload.mlir:70、85］ |
| pipeline | 0、2、3 | 工件模板参数：multibuffer 的 factor 与 pipeline_shared_memory_copies 的 depth；取 0 时工件降级为无 bypass_l1 的单缓冲 create_async_groups | factor = 2 与 depth = 2［gdn_l2_gen.py:1123-1124、1133-1135］ |
| mma | 真、假 | 载荷分支（mma fragment 载荷或 linalg.matmul 载荷）与工件分支（流水工件或 tile_using_for 工件） | 假路径发 tile_using_for tile_sizes [16,16]［gdn_l2_gen.py:1085-1090、1141-1148］ |
| warpspec | off、producer_consumer | 载荷分支 gen_payload_ws：warp 谓词、setmaxnreg 分档、mbarrier 环；工件退化为 identity | gdn.setmaxnreg dec 96 与 inc 224［gdn_l2_gen.py:565-568、585、624、1108-1112］ |
| tma | off、on（须 warpspec 开） | 4d、5d 共享 tile 类型，gdn.tma.async.load 与 store，host 侧 descriptor 与 tensormap_spec.json | descA 的 box 维界随 tile 走［gdn_l2_gen.py:552-556、589-594、1035-1041］ |

tile 三维的生成器合法域更宽（M 可到 16，N、K 取 8 的倍数即可），求解器按旋钮空间合同收窄到 32、64、128［tools/gdn_solver.py:70-74 的域常量注释；gdn_l2_gen.py:106-138］。两个发射点足以说明注入的含义。dtype 与 prec 在 fragment 构造处换成的就是类型串与属性串：

```python
av = "vector<4x2xbf16>" if dt == "bf16" else "vector<2x1xf32>"
bv_ = "vector<2x2xbf16>" if dt == "bf16" else "vector<1x1xf32>"
tf = " tf32Enabled" if dt == "f32" else ""
```
［tools/gdn_l2_gen.py:303-305，逐字］

这三个局部变量随后原样进入 mma.sync 的发射串，mmaShape 的 K 位取自 atom_k 的同维分支［tools/gdn_l2_gen.py:316-318］。warps 与 tile 则汇于发射几何的同一行 f-string：

```python
d_gx, d_gy, d_one, d_btx = (f"%c{64 // bt}", f"%c{n_glob // bv}",
                            "%c1",
                            f"%c{32 * (warps + 1) if ws else 32 * warps}")
```
［tools/gdn_l2_gen.py:1011-1013，逐字］

四个局部名再填进 gpu.launch 的 blocks 与 threads 槽位［tools/gdn_l2_gen.py:1043-1046］，落到 payload 即第 5 节将走查的 %bxx = %c128。表中全部落点有一个公共形态：旋钮值在生成期烧进 IR 文本，成为操作数类型、循环边界、launch 维度或工件模板参数，编译期之后无处可调。这是调度即 IR 最直白的证据——旋钮值即工件参数本身，改一个值等于重发一份文本，运行时没有任何等价入口。

### 5. 一个配置的端到端实例化走查

把十维落点放进一个真实目录。evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/ 的目录名即旋钮编码：kkt 阶段、bf16、mma 路径、流水深度 2、块 tile 64x64x32、4 个 warp；warpspec 与 tma 关闭时不入名，开启时以 ws、tma 段进入（对照同目录层的 kkt_bf16_mma_tma_ws）。经 ls 核实，目录内存放八个文件：payload.mlir、transform.mlir、config.json、kernel.ptx、kernel.cubin、ptxas_v.txt、sass.txt、meta.json——前三个由生成器写出，后五个由编译链产出。

走查从求解器一端开始。Z3 模型读出的旋钮向量经 to_config 翻译为配置字典，再经 config_name 拼成规范名：阶段、dtype、mma、流水、tile 三维、warps 依次入名，warpspec 开启时追加 ws 后缀，tma 再追加一段［tools/gdn_solver.py:261-275、278-290］。目录名、config.json 与 payload 首行注释因此是同一份字典的三个视图，任取其一都能还原全部旋钮值。

payload 里的旋钮值肉眼可读。首行注释把整份配置字典原样嵌进文件头：

```
// GDN L2 stage 'kkt': C = tril(k @ k^T) * exp2decay  knobs={"name":"kkt_bf16_mma_p2_t64x64x32_w4","stage":"kkt","dtype":"bf16",...,"warps":4,"mma":true,"pipeline":2,"prec":"bf16"}
```
［evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/payload.mlir:1，节选］

往下每一维各占其位：launch 的线程 x 维 %bxx = %c128 承载 warps=4（第 70 行）；共享 tile memref<64x32xbf16> 的两条边承载 tile.M 与 tile.K（第 214 行）；K 循环从 0 到 128、步长 32，携带 8 个 vector<2x2xf32> 累加器，步长承载 tile.K、累加器个数承载 warps 对 N 侧的切分（第 256 行）；nvgpu.mma.sync 的 mmaShape = [16, 8, 16] 与 bf16 操作数承载 dtype 与 prec（第 1310 行）。工件侧，factor = 2 与 depth = 2 分别落在第 5 行与第 12 行（第 3 节已全文引用这份工件）。config.json 存同一字典的机器可读形态，供脚本对账。

编译命令是单参数的：scripts/run_l2.sh <config-dir>，目录由 tools/gdn_l2_gen.py --gen 名称 --out 预先生成［scripts/run_l2.sh:4-6］。脚本合并两份 IR、过解释器与调度擦除（机制见第 8 节），末一步把 ptxas 的资源报告原样落盘：

```
"$PTXAS" -v -arch=sm_120a kernel.ptx -o kernel.cubin > ptxas_v.txt 2>&1
```
［scripts/run_l2.sh:81，逐字］

该配置的报告存档为：栈帧 3808 字节、溢出写 5708 字节、溢出读 6920 字节、255 寄存器、1 个 barrier［evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/ptxas_v.txt:4-5］。meta.json 再把报告与 PTX 直方图合并归档：mma.sync 48 条、cp.async 1542 条，SASS 侧 HMMA 48、LDGSTS 1536，PTX 文本 282907 字节。从 Z3 解出的旋钮向量到 cubin 与资源数字，全链只经过两份 IR 文件与一份脚本。复现这个配置等于重放这两个文件：配置的全部自由度已烧进两份文本，同一工具链下它们唯一决定全部产物，无需任何运行时搜索。

### 6. 参考实现作为旋钮空间坐标

覆盖工程把 fla 与 FlashQLA 的参考 kernel 放进同一坐标系：对每个参考 kernel 抓取 PTX 与 SASS 结构画像，在生成配置里按指令家族 Jaccard 距离找最近邻，再按寄存器与共享内存档位判分（容差约三成）。四张对照卡合排如下（最近配置列只写报告记录的配置名，未记录的维度不补写）：

| 参考 kernel | 我侧最近配置 | 参考侧结构要素 | 判分 |
|---|---|---|---|
| fla chunk_fwd_o（BK64_BV64_warps4_stages3） | ref_ldmatrix | 多级 ld.global 装载；255 寄存器、smem 24576 B、溢出 96/96 B；mma 96 条 | 寄存器 72 对 255，OUT（0.28 倍）；mma 1 对 96 |
| fla dhu（blockdim64、BV64、warps2、stages3） | kkt_bf16_mma_p2_t64x64x32_w2 | 经典 cp.async 流水；255 寄存器、smem 33288 B、溢出 3236/3788 B；mma 512 条，bf16 与 tf32 签名各半 | 寄存器 255 对 255，PASS；smem 无静态量，n/a；mma 96 对 512 |
| FlashQLA fused_fwd | kkt_bf16_mma_tma_ws | TMA 装载 31 条、mbarrier 74、elect.sync 18、setmaxnreg 4；128 寄存器封顶、smem 2048 B | 寄存器 255 对 128，OUT（1.99 倍）；smem 8 对 2048，OUT |
| FlashQLA fused_bwd | kkt_bf16_mma_tma_ws | TMA 41、mbarrier 194、elect.sync 21、setmaxnreg 4；循环体复用（bra 422）、mma 1600 条 | 寄存器 255 对 128，OUT（1.99 倍）；smem 8 对 2048，OUT |

［evidence/G2/coverage_matrix.md:71-114，四卡摘录；结构要素照抄卡片已记录项］

两极的对位结论由终局汇总给出。fla 极（多级 cp.async、255 寄存器、展开循环）由基础 mma 配置族承载：寄存器钉在 255 上限，栈帧覆盖 0 到 26.8 KB，pipeline 取 0、2、3 三档时 mma 条数为 16、48、64，LDGSTS 为 512、1536、2048。FlashQLA 极（warp 专化、TMA、128 寄存器档）由 kkt_bf16_mma_p2_t64x64x64_w4_ws 与 kkt_bf16_mma_tma_ws 承载：setmaxnreg dec96 与 inc224 把溢出从 22,844 B 压到 48 B，5 条 TMA 指令替换 1026 条经典拷贝，两者经 G4 修到硬件正确［evidence/G6/coverage_hatch_summary.md:63-77］。这套对位给出调度即 IR 的外部读法：一个参考实现落进本框架，坐标就是一组旋钮取值加一份工件，匹配与判分全部在配置名与归档产物上完成。复现一个参考实现，在这套语言里即选中一组旋钮取值。

### 7. 树内没有的调度原语

树内 transform 方言今天没有对应 op 的调度决策有三类，本仓把它们放在生成器与后处理层实现。

其一，warp 专化。工件里没有 warp 专化这个动作；生产者-消费者形态由载荷模板直接手写：nvvm.shfl.sync 广播 warp 编号，两条 arith.cmpi 谓词分出生产者与消费者分支，gdn.setmaxnreg dec 96 与 inc 224 分档寄存器预算，mbarrier 环手工布好［tools/gdn_l2_gen.py:565-568、585、624］。工件侧相应退化为 identity（只含 yield），无动作可施加［tools/gdn_l2_gen.py:1108-1112］；覆盖矩阵的结构维度表把 warp 专化记为参考侧 setmaxnreg 加 elect.sync、我侧的 G2-C 交付项［evidence/G2/coverage_matrix.md:62］。

其二，staging 纪律。全局到共享的搬运形态由载荷模板决定，transform 层无从表达每线程按线程号跨步认领 16 字节块这类纪律；全算子规模下全员发射的 block-uniform staging 造成 256 倍冗余搬运，教训与 thread-strided 对策完整记录在第二十二章第 2 条。

其三，寄存器上限的逐入口注入。多入口 cubin 需要逐 kernel 的 .maxnreg 指令，注入发生在 PTX 文本层：工具按环境变量表对入口名做子串匹配，命中后紧随 .maxntid 追加指令［tools/gdn_smem_fix.py:683-712］，前后对照与双遍编译的对账陷阱见第二十八章第 3 节。

三类决策的表达介质各异——载荷 op、模板纪律、PTX 后处理——但开关都落在同一套旋钮机制里。warpspec 与 tma 是求解器声明表里的两个布尔变量，与 dtype、tile 同列，耦合约束照常施加（tma 蕴含 warpspec，warpspec 形态下 pipeline 固定为 2）［tools/gdn_solver.py:99-110、70-74］；寄存器预算表是环境旋钮，构建缓存的键把每个 GDN 前缀环境变量连同生成器源码一并哈希，换一组环境值即换一个缓存槽，旧产物不会被新旋钮组合误用［tools/gdn_full_rt.py:28-40］。调度即 IR 的准确边界由此可述：能以 transform op 表达的调度（切块、多缓冲、异步组、软件流水）住在工件里；其余调度决策以载荷文本与环境旋钮表达。全部决策共用一个旋钮空间、一个求解域与一份证据命名法，介质差异只反映树内方言的覆盖现状，旋钮模型本身贯通始终。

### 8. 执行模型：解释器与调度擦除

工件与 payload 的合体是物理的：脚本把两份文件首尾相接成 combined.mlir，再交给解释器——

```
cat "$TMP/payload.gdn.mlir" transform.mlir > "$TMP/combined.mlir"
"$MLIR_OPT" "$TMP/combined.mlir" -transform-interpreter -o "$TMP/sched.mlir"
```
［scripts/run_l2.sh:54-55，逐字；合并前 gdn 直通 op 先经 gdn-opt --gdn-lower-hatch 降为树内 op，38-46 行］

-transform-interpreter 进入合并后的模块，找到 @__transform_main，把每条 transform op 解释为对 payload 的实际改写；-test-transform-dialect-erase-schedule 随后把 transform 方言的序列从产物里删除，输出只剩被改写后的 payload，与手写等效 IR 无异：

```
"$MLIR_OPT" "$TMP/sched2.mlir" -test-transform-dialect-erase-schedule \
  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a cubin-features=+ptx87 cubin-format=assembly" \
  -o "$TMP/asmobj.mlir"
```
［scripts/run_l2.sh:70-72，逐字；$MLIR_OPT 在 18 行定义］

两处防御值得记录。其一，identity 工件（只含 yield、payload 直接发射 op 的配置）整步跳过解释器——解释器会把裸顶层函数包进一层 module{}，这层额外嵌套让后续 NVVM 管线在 workgroup memref 上转换失败，运行脚本按工件内容探测后绕行［scripts/run_l2.sh:48-58；该行为的完整分析见第二十二章第 7 条］。其二，nomma 路径若 linalg.matmul 在调度后幸存（工件没匹配到或刻意不切），解释器之后补一条 -convert-linalg-to-loops 兜底［scripts/run_l2.sh:61-67］。

### 9. 调度即 IR 的意义

工件是纯文本：同一 payload 换一份工件即换一套调度，两套调度的差别可以逐行 diff。工件由生成器按旋钮值模板化发射——multibuffer 的 factor 与 pipeline 的 depth 正是求解器解出的流水旋钮，注入点即模板参数［tools/gdn_l2_gen.py:1121-1135］。复现一个配置等于复现一份工件加一组旋钮值：调度知识可存档、可在 CI 里断言、可在两家参考实现之间对位比较。第二十二章将展示这套机制的真实代价——树内调度动作在文档语义与实际行为之间存在缝隙，工件作者的对策全部以证据记录在案。
