## 二十二、NVVM 之后的改写：静态共享内存、别名、寄存器与宽存

`tools/gdn_smem_fix.py` 位于 NVVM 已产出 PTX、ptxas 尚未生成出货 cubin 的缝隙中。它是 Python 驱动的 PTX 文本后处理器，不进入 MLIR PassManager，也不重写 SSA IR。运行器先调用生成器与 `scripts/run_l2.sh`，再调用该工具；最终装载的是改写后重新汇编的 `kernel_smemfix.cubin`[tools/gdn_full_rt.py:72-85,107-124]。四个变换各自利用这一层才完整可见的信息：malloc 调用序列、最终分配尺寸、逐入口文本和已调度的窄存指令。

这里的正则不是通用 PTX 解析器。它们只接受生成器与 LLVM 当前产出的窄语法，匹配锚定行首、指令名和寄存器形式，未知的合法写法应当保持不变。第二次 ptxas 再承担完整语法检查，真机验证负责语义检查。窄识别、保守回退和权威汇编器组合起来，避免用一个不完整的解析器去猜测所有 PTX。

### 1. device malloc 变成带 128B 对齐的 .shared

stock 下译把 workgroup memref 分配表示成设备端 malloc，再把返回指针转换到 shared 窗口。裸 ptxas 不链接设备运行时，堆指针也不具备共享 tile 所需的地址语义。真实输入中，一次 8192B 分配和两种地址转换同时存在：

```ptx
{ // callseq 0, 0
.param .b64 param0;
.param .b64 retval0;
st.param::func.b64 [param0], 8192;
call.uni (retval0), malloc, (param0);
ld.param::func.b64 %rd220, [retval0];
} // callseq 0
// ...
cvta.to.shared.u64 %rd8, %rd220;
cvta.to.shared::cluster.u64 %rd196, %rd221;
```

[evidence/G4/gen_verify/kkt_bf16_mma_tma_ws/kernel.ptx:98-104,188,389，中间省略]

改写后，调用序列消失，入口前出现静态符号，普通 shared 与 cluster 变体都改成符号取址：

```ptx
.shared .align 128 .b8 __gdn_smem0[8192];
.shared .align 128 .b8 __gdn_smem1[8192];
// ...
mov.u64 %rd8, __gdn_smem0;
mov.u64 %rd196, __gdn_smem1;
```

[evidence/G4/gen_verify/kkt_bf16_mma_tma_ws/kernel_smemfix.ptx:17-18,170,371，中间省略]

前后对照应追踪 `%rd220/%rd221` 的 use-def 链，确认没有遗留对已删除 malloc 返回寄存器的读取。`::cluster` 是 TMA 目的地址使用的形式；漏改它会让张量装载写入无效共享地址。`.align 128` 也有硬件含义：sm_120a 的 `cp.async.bulk.tensor` 目的即使关闭 swizzle 仍要求 128B 对齐，较早的 16B 对齐曾触发 `CUDA_ERROR_MISALIGNED_ADDRESS`[evidence/G4/GATE_REPORT.md:131-142]。

工具先配对 callseq 的开闭编号，只有同时识别 malloc、尺寸和返回寄存器才删除；随后按源寄存器共享同一个静态符号：

```python
if "malloc" not in body:
    return m.group(0)
sm = SIZE_RE.search(body)
rm = RET_RE.search(body)
if not sm or not rm:
    return m.group(0)
malloc_reg_size[rm.group(1)] = int(sm.group(1))
return ""
# ...
lines = [f".shared .align 128 .b8 {s}[{n}];" for s, n in decls]
```

[tools/gdn_smem_fix.py:101-110,199-204]

这里能判断文本改写的保守边界：识别不完整就保留原文；`CVTA_RE` 中的可选 `(?:::\w+)?` 同时覆盖普通和 cluster 形式[tools/gdn_smem_fix.py:73-80]。静态声明沿用 MLIR 其他路径已经使用的 PTX 习惯写法，合法性再由第二次 ptxas 检查。

声明中的字节数之和还不是最终资源账。ptxas 会把不满足对齐粒度的声明向上取整，F1KS 的 46,144B 声明合计因此对应 46,352B 报告值[tools/gdn_full_gen.py:4403-4409]。审计时应同时保留两处数字：前者用于核对生成器的 arena 布局，后者用于判断实际驻留。只用声明总和可能误判一个贴着双 CTA 门槛的配置。

### 2. 生存期别名把独立分配映射到竞技场

共享内存峰值按同一时刻仍存活的字节计算，IR 中先后存在的独立 tile 可以复用同一物理区间。SSA 值唯一性约束的是值及其定义关系，并不禁止底层内存别名。项目在 IR 中保留清晰的独立 `memref.alloc`，到 PTX 层依据已审计生存期实施复用；这是一项布局选择，与“MLIR 原理禁止别名”无关。

生成器随产物写出 `smem_alias.json`。`identify` 用 malloc 字节数反查 tile，`arenas` 声明基区大小，`alias` 给出基 tile 与字节偏移。JSON 只把生成器已经完成的生存期分析传给后处理器；尺寸相同本身不能证明可复用，区间不越界与最后读、首次写之间的同步仍由生成器负责。改写器承担可靠识别和机械映射，不会从一张尺寸表臆造生存期事实。F1 的布局如下：

```json
"gdn_fwd_f1_kernel": {
 "identify": {"9216": "Abf", "9504": "Tt",
              "8704": "vt", "8976": "k2"},
 "arenas": {"Abf": 9216, "Tt": 9504},
 "alias": {"vt": ["Abf", 0], "k2": ["Tt", 0]}
}
```

[evidence/G5/gen_fwd/B4T2048_mono/smem_alias.json:3-23，按单行压缩展示]

F1 的 vt、k2 在求解结束后才使用，Abf、Tt 的对应字节此时已经死亡。原 PTX 分别转换四个独立 malloc 结果；改写后，后两者落到前两者基址：

```ptx
// before
cvta.to.shared.u64 %rd17, %rd113;  // vt, 8704B
cvta.to.shared.u64 %rd18, %rd114;  // k2, 8976B
// after
mov.u64 %rd17, __gdn_smem_gdn_fwd_f1_kernel_4;
add.u64 %rd17, %rd17, 0;
mov.u64 %rd18, __gdn_smem_gdn_fwd_f1_kernel_5;
add.u64 %rd18, %rd18, 0;
```

[evidence/G5/gen_fwd/B4T2048_mono/kernel.ptx:89-102,248-249；evidence/G5/gen_fwd/B4T2048_mono/kernel_smemfix.ptx:181-191]

`add 0` 看似多余，却保留统一模板；非零偏移时同一形式直接表达子区间。尺寸用于识别，因为 LLVM 可以调整 malloc 调用顺序。唯一性检查决定整个布局能否采用：

```python
if len(regs) != 1:
    print(f"[smem_fix] WARN alias: size {sz_s} for '{tile}' "
          f"matches {len(regs)} mallocs -- aliasing disabled")
    ok = False
    break
# ...
if not ok:
    arena_of, alias_of, reg_of_tile, tile_of_reg = {}, {}, {}, {}
```

[tools/gdn_smem_fix.py:132-136,150-151]

零匹配表示分配已消失，无需重定向；多匹配或引用未识别 tile 会弃用别名，退回每个 malloc 一个独立声明。失败损失空间收益，不改变已验证的寻址语义。这也是后处理层适合承载该变换的原因：最终尺寸已经稳定，失败时仍保留完整原始分配。

### 3. .maxnreg 给出逐入口的静态预算

单模块多入口（mono）cubin 内不同 kernel 可以有不同寄存器预算，文件级 `--maxrregcount` 无法同时表达 `f2:128` 与 `f3:168`。`GDN_MAXREG_K` 按入口名子串匹配并插入 PTX directive：

```ptx
// before
.visible .entry gdn_dqkwg_kernel_kernel()
.maxntid 256, 1, 1
{
// after
.visible .entry gdn_dqkwg_kernel_kernel()
.maxntid 256, 1, 1
    .maxnreg 128
{
```

[evidence/G5/scratch/r19b_final_T8192/dqkwg/kernel.ptx:37-39；evidence/G5/scratch/r19b_final_T8192/dqkwg/kernel_smemfix.ptx:49-52]

当前注入器的入口匹配与追加逻辑为：

```python
m = re.match(r"\s*\.visible\s+\.entry\s+([\w.$]+)", ln)
if m:
    name = m.group(1)
    key = next((k for k in want if k in name), None)
    if key:
        out.append(f"\t.maxnreg {want[key]}")
        applied[name] = want[key]
```

[tools/gdn_smem_fix.py:699-707]

上面的 dqkwg 文本来自注入审计产物，directive 位于目标 entry 的资源声明区。`.maxnreg` 约束 ptxas 对该 entry 的静态寄存器分配；`gdn.setmaxnreg` 则下降为 `nvvm.setmaxnreg`，用于 warp 专化执行期间按 increase/decrease 方向调整寄存器配额[dialect/lib/GdnHatch.cpp:29-39]。前者影响编译后的入口资源账，后者是设备代码中的动态寄存器管理指令，二者不能互相替代。

注入器逐项记录实际命中的 entry，请求数与命中数不符就打印警告[tools/gdn_smem_fix.py:683-712]。真正对账还要读改写后那次 ptxas 的报告：文本变化会改变调度与分配，第一遍未修复 PTX 的寄存器数没有资格代表出货 cubin。第二十三章的 128/168 案例正说明这条边界。

更低的上限也不自动带来更优机器。寄存器收紧可能提高驻留，同时把活跃值压到本地内存；是否值得，取决于增加的驻留能否覆盖溢出流量及其延迟。因而 `.maxnreg` 的验收至少要并列记录寄存器数、spill、驻留 CTA 和端到端时间，不能只检查 directive 已出现在文本里。

### 4. WST 把四条 16 位访存合成两条 32 位访存

宽存变换 WST（wide-store）寻找相邻的 shared 16 位装载及其 global 16 位存储。下面同一循环中，LLVM 把高地址 store 调到了低地址 store 前面，配对若只按文本顺序就会漏掉：

```ptx
// before
$L__BB0_94:
ld.shared.b16 %r1653, [%rd543];
ld.shared.b16 %r1654, [%rd543+2];
st.global.b16 [%rd544+2], %r1654;
st.global.b16 [%rd544], %r1653;
// after
$L__BB0_94:
ld.shared.b32 %r9000, [%rd543];
st.global.b32 [%rd544], %r9000;
```

[evidence/G5/gen_fwd/B4T2048_mono/kernel.ptx:3328-3332；evidence/G5/gen_fwd/B4T2048_mono/kernel_smemfix.ptx:3270-3272]

配对器按地址关系工作：shared 偏移必须相差 2B，global 必须同基址且也相差 2B，每个源寄存器只能有一次使用，装载必须先于各自 store。满足这些条件后，小端 32 位寄存器装入的两个 16 位单元与原来的字节序列一致，写回地址也一致。未满足任一条件的站点保留原指令，所以候选缺失只是失去融合机会。

新寄存器从 `%r9000` 起分配，并用显式逗号清单声明：

```python
decl = "\t.reg .b32 " + ", ".join("%%r%d" % i
                                  for i in range(9000, new_reg[0])) + ";"
```

[tools/gdn_smem_fix.py:554-562]

对应真实产物为 `.reg .b32 %r9000, %r9001, %r9002, %r9003;`[evidence/G5/gen_fwd/B4T2048_mono/kernel_smemfix.ptx:60]。这里不用区间或计数声明，因为当前 ptxas 对这种高编号形式有限制。核心替换与删除旧指令的逻辑位于同一份配对器[tools/gdn_smem_fix.py:518-553]；mono 路径还受 `GDN_WST` 和 `GDN_WST_SCOPE` 双重限定，只有选定模块执行[tools/gdn_smem_fix.py:773-777]。

四条变两条直接减少指令站点，却不承诺时间减半。宽存还要求配对地址满足对齐，实际收益受 LSU 发射、shared bank 几何和全局扇区利用率影响。配对正确性由地址与字节图像证明，性能则交给同形状交错配对测量；WST 与后续的 lane 几何调整各自有不同验收问题。

### 5. 缓存、mono 合并与失败路径共同保护产物

后处理产物会被缓存，缓存键必须覆盖能够改变 PTX 的输入。当前键纳入所有 `GDN_`、`G5` 前缀环境变量以及生成器、改写器自身源码：

```python
for k in sorted(os.environ):
    if k.startswith("GDN_") or k.startswith("G5"):
        h.update(f"{k}={os.environ[k]}\n".encode())
for f in ("gdn_full_gen.py", "gdn_smem_fix.py"):
    p = os.path.join(TOOLS, f)
    if os.path.exists(p):
        h.update(f"\n#{f}#".encode())
        with open(p, "rb") as fh:
            h.update(fh.read())
return h.hexdigest()[:16]
```

[tools/gdn_full_rt.py:28-44]

键随 `full_spec.json` 落盘。旋钮或源码变化时，构建转向 `<out_dir>_k<hash8>`，旧目录保留以供取证[tools/gdn_full_rt.py:47-70]。没有缓存键标记的旧目录按兼容策略继续受信，这条例外也属于缓存合同，不能省略。

mono 合并还有独立的符号问题：ptxas 会跨 entry 合并同名 `.shared`，三个模块各自的 `__gdn_smem0` 若尺寸不同，就可能让一个 kernel 经另一个 kernel 较小的共享区越界。合并器在拼接前加入模块标签：

```python
def rename_syms(text, tag):
    return re.sub(r"__gdn_smem(\d+)",
                  lambda m: "__gdn_smem_%s_%s" % (tag, m.group(1)),
                  text)
```

[tools/gdn_smem_fix.py:715-729]

这一道改名与四变换共同接受第二次 ptxas 检查。整个工具的失败纪律很具体：别名识别不唯一就回退，WST 条件不全就保留窄指令，找不到 `.visible .entry` 就拒绝插入声明，最终汇编失败则打印 stdout/stderr 并非零退出[tools/gdn_smem_fix.py:199-208,800-804,843-847]。数值路径不做重结合，WST 要求字节图像相同，别名要求生存期与偏移已经证明；不能保证时就保留独立、可汇编的形态。后处理因此既改变机器产物，也始终把“无法识别”与“允许改写”分开。
