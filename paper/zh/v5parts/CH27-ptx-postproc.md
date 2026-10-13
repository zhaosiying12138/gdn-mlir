## 二十七、PTX 文本后处理的架构与原则

NVVM 下译结束后、ptxas 定案之前，本编译器插入一层自己的东西：一个 856 行的 Python 文本改写器 gdn_smem_fix.py，承载四个变换（下一章逐个对照）。为什么允许自己存在于两个黑盒编译器之间？判据先于答案：逐个检查四个问题能在哪一层被表达。设备 malloc 是 stock 管线的既定行为（上游测试 memref-to-llvm.mlir 明文示范），改它等于改上游的下译策略；生存期别名与 SSA 合法性冲突——IR 层每 tile 必须各有一份分配，"这些 tile 共享字节"没有合法的表达位置；寄存器上限在 PTX 源内指令里才有逐入口语义，IR 层不存在对应物；16 位存储的配对只在 LLVM 调度完成后的指令序里可见，调度前无从谈起。四个修复点的共同位置由此确定：全部在 NVVM 下译之后才成为可观察的 PTX 事实。修复若想进上游 MLIR，每条都要走完整的提案周期；本项目选在 PTX 文本层修复，换来即时可控与可审计。原则先行：复刻 MLIR 已有的习惯用法、每步改写要么逐位相同要么回退到正确形态、失败必须响亮。

### 1. 定位：修在 PTX 层的三条纪律

第一条纪律是复刻。静态共享内存的修复（第二十八章第一节）复刻 MLIR 自身为 warp 专化配置已经发射的 .shared 声明习惯用法——修复后的 PTX 与树内某条路径的既有写法同构，不发明新语法。这把"改写器写出的 PTX 是否合法"的问题转化为"MLIR 已经在发同样的东西"，由树内先例背书。第二条纪律是位级等价或回退。生存期别名（第二节）作为纯增益变换，其失败模式被设计为"不省字节、保持每 tile 独立声明"，绝不允许"算错"；识别不唯一时整体弃用别名。第三条纪律是失败响亮：找不到 .visible .entry 标记时直接 raise SystemExit 拒绝改写（tools/gdn_smem_fix.py:205-207），ptxas 第二次编译失败同样打印完整输出后退出——静默通过不可接受，因为这一层的产物就是出货 cubin 的直接前身。

### 2. 工具解剖：正则、状态与模板

工具对 PTX 的理解由四个正则承担，全部锚定在行首制表符上以避免误匹配操作数文本：

```
CALLSEQ = re.compile(
    r"\t\{ // callseq (\d+), 0\n"          # malloc 调用序列的开界
    r"((?:\t[^\n]*\n)+?)"
    r"\t\} // callseq \1\n")               # 闭界（编号与开界一致）
SIZE_RE = re.compile(r"st\.param::func\.b64[ \t]+\[param0\],[ \t]+(\d+);")
RET_RE  = re.compile(r"ld\.param::func\.b64[ \t]+(%r[d]\d+),[ \t]+\[retval0\];")
CVTA_RE = re.compile(
    r"\tcvta\.to\.shared(?:::\w+)?\.u64[ \t]+(%r[d]\d+),[ \t]+(%r[d]\d+);")
```
［tools/gdn_smem_fix.py:73-80，逐字；工具契约全文在 1-49 行 docstring］

这套设计的要害是状态在闭包间传递而文本只被改写一次：CALLSEQ.sub 的回调 eat_callseq 删除每个 malloc 调用序列，同时把"结果寄存器 → 字节数"记进 malloc_reg_size 表（SIZE_RE 取尺寸、RET_RE 取返回寄存器，101-110 行）；随后 CVTA_RE.sub 的回调 patch_cvta 查这张表，把每条 cvta.to.shared 替换为对 .shared 符号的 mov 取址模板，符号名与声明在 decls 列表里同步生成（157-195 行）。(?:::\w+)? 这一可选组匹配 cvta.to.shared::cluster 变体——TMA 的目的地址走该形态，漏掉它意味着 TMA 目的读已删除的 malloc 定义寄存器、设备直接陷入错误态（注释 171-176 行明记这个教训）。

主流程按产物形态分两条（组织结构见 755-811 与 813-856 行的 main_multi/main）。单 kernel 目录的路径：fix_ptx 改写共享内存与别名 → 常量参数消除（fix_params，解析 asmobj.mlir 里 gpu.launch_func 的实参清单，把每个编译期常数参数的 ld.param 替换为立即数 mov 并删除死参数声明，280-333 行；非恒定参数如 TMA 描述符原样保留为真参数）→ 写 kernel_smemfix.ptx → 以同一 ptxas 编译 kernel_smemfix.cubin → 打印报告。mono 目录（多个 kernel 装一个 cubin）先逐模块改写，再 merge_ptx 合并、apply_maxnreg_k 注入、一次编译。报告逐 kernel 打印 cvta 替换数、共享声明清单、参数替换数与融合数——审计的第一手材料，也是第三十六章配置门"编译期静态复核"的数据来源之一。

### 3. 键控缓存：旋钮加源码的指纹

改写器的产物会进缓存目录，而缓存的最大敌人是陈旧：换一组旋钮环境变量、或改了一行生成器代码，旧 cubin 就成了假命中。工具链的防线是把缓存键定义成"一切能改变产物的输入"的哈希：

```
def _cache_key():
    h = hashlib.sha256()
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
［tools/gdn_full_rt.py:28-45，逐字；装配逻辑 52-64 行：build 时把键写进 full_spec.json，命中检查发现键不匹配即整体重定向到 `<out_dir>_k<hash8>` 变体槽，旧目录原样保留］

键的输入域是全部 GDN_ 与 G5 前缀的环境旋钮加上生成器与改写器两个源文件本身的字节。源码入键的含义直白：任何对工具的手改都会使全部缓存自动失效，不需要任何人记得去清缓存；旋钮入键的含义同样直白：同一目录名之下，不同旋钮组合是不同的内核，理应占不同的槽位。docstring 记录了这条防线成立之前同一目录缓存被错误采信的两次事故（一次换了旋钮、一次改了生成器），作为设计动机的实证。变体槽机制同时保住了取证完整性——旧目录从不被覆盖，任何一次历史测量的 cubin 都还能找到。

### 4. mono 合并：重名 .shared 的防御

前向最终形态是三个 kernel 装进单一 cubin，这要求把三份 PTX 合成一个模块。合并本身的骨架朴素——取首模块头部、并集 .extern 声明、拼接全部入口体（tools/gdn_smem_fix.py:715-752）——但有一个硬件工具链级的坑必须主动防御：ptxas 跨入口静默合并重名 .shared 符号。三个 kernel 各自带 __gdn_smem0，尺寸不同；ptxas 把它们并成一个，mono 的 f3 kernel 就经 f1 的更小符号越界写共享内存（第四十章发现 6）。防御是逐 kernel 改名：合并前把每个模块的 __gdn_smemN 重写为带 kernel 标签的 `__gdn_smem_<tag>_N`：

```
def rename_syms(text, tag):
    return re.sub(r"__gdn_smem(\d+)",
                  lambda m: "__gdn_smem_%s_%s" % (tag, m.group(1)),
                  text)
```
［tools/gdn_smem_fix.py:726-729，逐字；注释 715-720 行记录越界写事故］

这一层是全编译器里最接近"胶水"的部分，但它的每个组件都按编译器部件的标准建造：契约写进 docstring、匹配锚定行首、模板逐字可查、缓存键覆盖全部输入、合并防御已知工具链缺陷。
