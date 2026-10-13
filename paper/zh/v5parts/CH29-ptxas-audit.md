## 二十九、ptxas 与 SASS 审计

cubin 交给 GPU 之前，编译器还能做最后一类工作：审计。ptxas（NVIDIA 的 PTX 汇编器）每次编译都打印一份资源报告，nvdisasm 能把 cubin 反汇编成 SASS（硬件真实执行的指令序），compute-sanitizer 能在真机上拦截越界与竞态。这三样东西构成本编译器的资源审计方法学：调度器的成本模型从静态计数取数（第十一章），配置门从资源包络复核（第三十六章塔总览表），缺陷定位从报告与反汇编差分取证。

### 1. ptxas -v：资源报告的读法

ptxas 加 -v 编译时输出逐入口资源报告，字段含义固定。一份真实输出（G4 时代 kkt 配置，scripts/run_l2.sh 第 81 行把 stdout/stderr 落盘为 ptxas_v.txt）：

```
ptxas-blackwell info    : Compiling entry function 'gdn_kkt_kernel_kernel' for 'sm_120a'
ptxas-blackwell info    : Function properties for gdn_kkt_kernel_kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas-blackwell info    : Used 96 registers, used 1 barriers
```
［evidence/G4/gen_verify/kkt_bf16_mma_p0_t64x64x32_w4/ptxas_v.txt，节选（gmem 与编译耗时两行略）］

四个字段各自裁决一件事：Used N registers 定占用上界（每 SM 寄存器文件 65,536 字节，线程数乘每线程寄存器数是驻留瓶颈之一）；bytes smem 定共享内存池（改写后的第二次编译才报此字段，见下节样例）；spill stores/loads 定溢出（寄存器不够时数据被压进本地内存，访存代价放大，是发射器过宽的直接信号）；stack frame 定栈帧。这些数字的消费方是编译期自己的两道关卡：配置门把 20 个旋钮配置的资源读数与包络比对（第三十六章塔总览表的"ptxas -v 资源包络"行），成本模型的溢出项在 ptxas 报告缺失时回退到 SASS 计数（第三节）。字段解析有现成函数 parse_ptxas_v（tools/harvest_flaptx.py:246），run_l2.sh 把解析结果连同 PTX 特征直方图、mma 签名清单、SASS 指令计数一并写进 meta.json——每个配置目录自带一份机器可读的资源档案。

### 2. 双遍陷阱：168 寄存器倒退案例

本编译器每个 kernel 被同一 ptxas 编译两次：run_l2.sh 的第一遍（kernel.ptx 到 kernel.cubin，报告落 ptxas_v.txt），与 gdn_smem_fix.py 改写后的第二遍（kernel_smemfix.ptx 到出货 cubin）。两遍的 PTX 文本不同（第二十八章的变换全部生效），ptxas 的调度自由度不同，寄存器分配可以合法地不同——所以 ptxas_v.txt 的数字对出货 cubin 没有对账效力。这个陷阱有一个完整案例。一次双 CTA 构建按 ptxas_v.txt 记录了 128 寄存器并据此判定驻留合法；实测该 kernel 性能倒退到 0.78 倍，ncu 显示 Block Limit Registers = 1——出货 cubin 实际占用 168 寄存器，2×256×168 超过 65,536，每 SM 只剩 8 个 warp。复盘确认：第一遍报告 128，第二遍的改写后 PTX 给了 ptxas 不同的调度空间，落到 168 寄存器、零溢出。修复是 PTX 源内注入 .maxnreg 128（第二十八章第三节），注入后的第二遍重新对齐：

```
ptxas-blackwell info    : Used 174 registers, used 1 barriers, 47616 bytes smem
ptxas-blackwell info    : Used 128 registers, used 1 barriers, 88 bytes cumulative stack size, 47616 bytes smem
ptxas-blackwell info    : Function properties for gdn_dqkwg_kernel_kernel
    88 bytes stack frame, 84 bytes spill stores, 120 bytes spill loads
```
［evidence/G5/scratch/r19b_final_T8192/dqkwg/ptxas_smemfix_v.txt：同一文件里改写态与 maxnreg 态两次编译的对照，节选；案例全文 evidence/G5/bwd_report.md:2574-2596］

审计守则由此定型：资源对账一律读第二遍的报告（ptxas_smemfix_v.txt 或 maxnreg 注入后重编译的输出）；文档里记录的第一遍数字只作过程参考。该案例同时复核了 ptxas-blackwell 忽略 --maxrregcount 旗标、遵循源内 .maxnreg 指令这条工具链事实。

### 3. nvdisasm 与静态指令计数

nvdisasm -c 把 cubin 反汇编为 SASS，run_l2.sh 第 84 行落盘 sass.txt。静态计数是为成本模型供数的关键用途：对 SASS 文本数指令，得到每 kernel 的 HMMA（张量核）、LDGSTS（cp.async）、LDS/STS（共享读写）、LDL/STL（本地内存，即溢出访问）、BAR.SYNC、LDSM（ldmatrix）、EX2 计数，写进 meta.json 的 sass_ops 字典（scripts/run_l2.sh:107-111）；同一函数还从 PTX 侧提取 mma 签名清单与 ex2 变体清单（120-122 行），直接供第三十八章 T1 覆盖矩阵作输入。计数的验证口径：18 个配置的静态 PTX 计数与生成器预测 18/18 精确一致（mma.sync、ld.shared、bar.sync 全对，cp.async 偏差不超过 1.6%，第十一章）。成本模型消费端的一个细节：溢出项的计数在 ptxas 报告缺失时回退到 SASS 的 LDL 加 STL 静态计数，并除掉流水 peel 造成的函数体复制，换算成动态次数再除以带宽速率（tools/gdn_costmodel.py:616-625、711-712）。SASS 侧的计数因此直接进了排序公式——审计产物与调度器之间有一条真实数据流。

### 4. compute-sanitizer：越界与竞态的定位器

数值验证对越界类缺陷常常无感（读到的可能是合法但陈旧的字节），compute-sanitizer 在真机上拦截原始访问。两个本仓真实案例。其一，TMA 对齐缺陷：静态 .shared 以 .align 16 声明时 ptxas 把 tile 摆到 16B 而非 128B 边界，运行以 CUDA_ERROR_MISALIGNED_ADDRESS 陷入，sanitizer 把故障点定位到 UTMALDG.4D 这一条装载指令（evidence/G4/GATE_REPORT.md:139-142）——从"哪条指令"反推"对齐要求"，第二十八章的 .align 128 由此立规。其二，重名 .shared 合并缺陷：ptxas 跨入口静默合并重名符号后，mono 的 f3 kernel 经 f1 的更小符号越界写共享内存，sanitizer 报出 0xa800 处的 shared write（evidence/G5/fwd_report.md:238）——第二十七章合并时的逐 kernel 改名由此立规。racecheck 工具同样在册：跨 warp 扫描原语的同步缺陷即由它配合逐位一致性协议定位（evidence/G5/scratch/r17f/racecheck_f1ks.py）。

### 5. SASS 差分：一条指令定案一个缺陷

审计的最后手法是差分：把通过版本与失败版本的 cubin 各自反汇编、逐指令对比。当两份 SASS 的差异小到可枚举时，缺陷的触发条件就被锁定。本仓最干净的一次：前向 F1 kernel 在强制单 CTA 驻留时全部正确、在双 CTA 驻留时数据损坏，两个版本的 SASS 差恰好一条指令：

```
MOV R4, 0x2490   // 通过版：Tg 的 malloc 尺寸 9,360
MOV R4, 0x4410   // 失败版：同一尺寸 17,424
```
［evidence/G5/fwd_report.md，双 CTA 损坏根因一节，逐字；两版代码体完全相同］

一条指令之差证明代码路径完全一致、触发条件纯属驻留状态。配合 tile 转储把损坏单元对齐到 zero_up 循环漏写的格子：该清零循环以 32×8 的跨步走过 128 线程的块，每 tile 有 768 个格子从未被清零；单 CTA 时这些格子恰好读到全零（共享内存初始态，逐格实测 393,216 个全为零），双 CTA 共驻时读到同 SM 另一 CTA 的活跃数据。修复是把清零跨步改回线程数（GDN_ZFIX），双 CTA 下十次重复逐位与单 CTA 参考一致。方法学的教益：SASS 差分把"数值错误"翻译成"驻留依赖"，tile 转储再把"驻留依赖"翻译成"具体循环的具体跨步"——两级证据把一个只在共驻时显形的缺陷变成一条可指认的代码行。这个案例也解释了为什么逐位一致性协议（第三十六章）在此类缺陷上不可或缺：损坏值多在 1e-14 量级，数值容差永远抓不住它们。
