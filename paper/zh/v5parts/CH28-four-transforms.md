## 二十八、四个变换的前后对照

证据树里保存着六十余对 kernel.ptx 与 kernel_smemfix.ptx 成对文件，以下引用全部取自这些成对证据。四个变换合起来覆盖两个正确性问题（静态共享内存、生存期别名）、一个驻留问题（寄存器上限）、一个发射效率问题（bank 几何）。

### 1. 静态共享内存修复：device malloc 到 .shared 符号

问题：stock 管线把 workgroup 空间的 memref.alloc 下译为设备端堆分配。 cubin 因此带着对 malloc 桩的未解析调用（裸 ptxas 产物不链接设备运行时），即便解析成功，堆指针又被重解释为共享内存窗口偏移，cp.async 与 ld.shared 触及的共享地址全部越界——携带共享 tile 的 cubin 在真机上全部非法地址错误，通过率为零。这是上游文档化的行为（llvm-mlir 的 memref-to-llvm.mlir 测试明文示范该下译形态），修复只能在管线之外施加。变换前的真实 PTX：

```
{ // callseq 0, 0
.param .b64 param0;
.param .b64 retval0;
st.param::func.b64 [param0], 8192;
call.uni (retval0), malloc, (param0);
ld.param::func.b64 %rd220, [retval0];
} // callseq 0
...
cvta.to.shared.u64       %rd8,  %rd220;      // 普通往返
cvta.to.shared::cluster.u64 %rd196, %rd221;  // TMA 的 cluster 形态
```
［evidence/G4/gen_verify/kkt_bf16_mma_tma_ws/kernel.ptx:98-104、188、389-390（制表对齐有规整）］

变换后，malloc 序列整体删除、入口前插入静态声明、两种 cvta 都改为对符号直接取址：

```
.shared .align 128 .b8 __gdn_smem0[8192];   // 声明插在 .visible .entry 前
.shared .align 128 .b8 __gdn_smem1[8192];
...
mov.u64 %rd8,   __gdn_smem0;                  // cvta.to.shared → mov 取址
mov.u64 %rd196, __gdn_smem1;                  // ::cluster 形态同样改写
```
［evidence/G4/gen_verify/kkt_bf16_mma_tma_ws/kernel_smemfix.ptx:17-19、371-372］

三个工程细节决定成败。其一，声明必须 .align 128：TMA 装载在 sm_120a 上即使 swizzle 关闭也要求 128B 对齐，早期 .align 16 让 ptxas 把 tile 摆到 16B 而非 128B 边界、首次装载即 CUDA_ERROR_MISALIGNED_ADDRESS（tools/gdn_smem_fix.py:199-204 注释记录这次定位）。其二，cvta.to.shared::cluster 变体必须一并改写，否则 TMA 目的读已删除 malloc 定义的寄存器、设备直接陷入错误态——CVTA_RE 的 (?:::\w+)? 可选组即为此设（73-80 行）。其三，识别按 callseq 边界配对开闭编号，尺寸与结果寄存器各由一个窄正则取出，匹配不完整就原样保留该序列——从不猜测。每个 malloc 结果对应一个静态符号、其全部 cvta 转换共享同一符号（157-171 行）。修复后通过率从零到全量。

### 2. 生存期别名：独占 malloc 到竞技场基址加偏移

问题：算法 tile 的生存期拼接远超双 CTA 驻留的 50.2–50.7 KB 门槛，任意时刻活跃字节却在门槛之下；"生存期不重叠的 tile 共享字节"这一物理事实无法在 IR 层表达（SSA 合法性要求每 tile 一次分配），只能在 PTX 层施加。生成器为每个 kernel 产出布局表 smem_alias.json：identify 按 malloc 尺寸识别 tile 名，arenas 声明竞技场基 tile 及其字节数，alias 把每个被折叠 tile 映射到基 tile 加字节偏移。前向 F1 的布局表原文（vt 与 k2 两个 tile 分别折叠进 Abf 与 Tt 两个竞技场）：

```
"gdn_fwd_f1_kernel": {
 "identify": { "9216": "Abf", "9504": "Tt", "8704": "vt", "8976": "k2" },
 "arenas":   { "Abf": 9216, "Tt": 9504 },
 "alias":    { "vt": ["Abf", 0], "k2": ["Tt", 0] }
}
```
［evidence/G5/gen_fwd/B4T2048_mono/smem_alias.json，前向 F1 布局表原文］

变换后，被别名 tile 的地址重定向到竞技场符号加字节偏移，独占声明消失：

```
mov.u64 %rd17, __gdn_smem_gdn_fwd_f1_kernel_4;   // vt → Abf 竞技场
add.u64 %rd17, %rd17, 0;
mov.u64 %rd18, __gdn_smem_gdn_fwd_f1_kernel_5;   // k2 → Tt 竞技场
add.u64 %rd18, %rd18, 0;
```
［evidence/G5/gen_fwd/B4T2048_mono/kernel.ptx:92-101、248-249 → kernel_smemfix.ptx:188-191；替换模板在 tools/gdn_smem_fix.py:177-184］

识别核心是"malloc 结果寄存器按唯一尺寸反查 tile 名"：同一尺寸出现多次即整体弃用别名（114-137 行）。之所以按尺寸识别，是因为 LLVM 会重排 malloc 调用序列的先后，顺序不可依赖，尺寸是唯一稳定的锚。安全边界有三处显式回退：尺寸匹配到零个 malloc（该 tile 被优化掉，跳过）、匹配多个（弃用别名）、布局表引用了未识别的 tile（弃用）——弃用后的形态与第 1 节的独占声明完全一致，正确性不受影响，损失的只有字节数。设计立场：别名属于纯增益变换，失败模式必须是"不省"，绝不允许"算错"。效果即图 6 的三个池压缩（dqkwg 100,864 降到 47,616B、caryA 82,176 降到 28,592B、wyb 100,096 降到 45,056B）与三个 kernel 的双 CTA 驻留（dqkwg 实测每 SM warp 数 7.59 升到 14.53）。

### 3. 寄存器上限注入：.maxnreg

问题：装着三个 kernel 的单一 cubin 需要逐 kernel 的寄存器预算，而 ptxas-blackwell 忽略 --maxrregcount 旗标、遵循 PTX 源内的 .maxnreg 指令；旗标本身也是逐文件的，对 mono cubin 无差别限制没有意义——f2 想要 128、f3 想要 168 时，一个旗标给不出两个数。注入因此落在 PTX 文本上：工具按环境变量表 GDN_MAXREG_K='f3:168,f2:128' 对入口名做子串匹配，匹配后紧随 .maxntid 追加指令，未匹配任何入口则显式告警（tools/gdn_smem_fix.py:683-712）。真实前后对照：

```
// 变换前                                    // 变换后
.visible .entry gdn_dqkwg_kernel_kernel()    .visible .entry gdn_dqkwg_kernel_kernel()
.maxntid 256, 1, 1                           .maxntid 256, 1, 1
                                                  .maxnreg 128
                                              {
```
［evidence/G5/scratch/r19b_final_T8192/dqkwg/kernel.ptx:37-38 → kernel_smemfix.ptx:49-52］

工程细节有二。注入发生在 mono 合并之后，逐入口追加在 .maxntid 旁，环境变量表因此对合并后的全部入口一次生效。对账必须读第二次编译的报告：ptxas 报告文件只对应第一次编译，最终 cubin 出自改写后的第二次编译，两次的寄存器分配可能不同——第二十九章展开这个双遍陷阱的完整案例（168 寄存器倒退）。注入后占用从 15.02% 升到 29.17%、每 SM warp 数从 8.00 升到 14.23，代价是换出的少量溢出（dqkwg 注入后 84B spill stores、120B spill loads，第二十九章样例可见）。

### 4. WST 宽存融合：2×16 位到 1×32 位

问题：smem 到 global 的逐 cell 16 位搬运，WST 旋钮下应配对成 32 位读写，指令与循环迭代减半。变换前后同一循环、同一基址寄存器，四条指令变两条：

```
// 变换前（kernel.ptx:3328-3335）             // 变换后（kernel_smemfix.ptx:3270-3272）
$L__BB0_94:                                  $L__BB0_94:
ld.shared.b16 %r1653, [%rd543];              ld.shared.b32 %r9000, [%rd543];
ld.shared.b16 %r1654, [%rd543+2];            st.global.b32 [%rd544], %r9000;
st.global.b16 [%rd544+2], %r1654;
st.global.b16 [%rd544],   %r1653;
```
［evidence/G5/gen_fwd/B4T2048_mono/kernel.ptx → kernel_smemfix.ptx，该目录共融合 7 处］

字节精确等价由构造保证：32 位寄存器恰好装下两个相邻 16 位 cell，写向相同地址。配对按地址配而非文本顺序——+2 偏移的 store 常被 LLVM 调度在前，工具按"共享地址相邻且全局偏移差恰为 2"识别（正则 470-473 行、偏移校验与顺序约束 483-547 行）。新寄存器从 %r9000 起编号：该版本 ptxas 拒绝区间声明，工具把新寄存器写成显式逗号清单（549-556 行）。WST 负责融合，W32P（第三十四章的 lane 重映射）负责让配对在 bank 几何上合法——两层各管一半，缺一则该存储通路的收益不存在。

### 5. 合起来看

四个变换共享同一批基础设施：第 1、2 节吃 callseq 正则与 cvta 状态表，第 3 节吃 mono 合并后的入口流，第 4 节是独立的行级配对器；全部挂在同一个键控缓存与同一份双遍报告之下。每个变换的验收口径一致——PTX 逐行可 diff、产物 cubin 跨进程逐位复现、失败路径要么回退要么响亮。
