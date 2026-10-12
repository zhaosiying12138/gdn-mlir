# 4 PTX 后处理层：`gdn_smem_fix.py` 与构建链的修复/变换全集

## 4.0 章首：为什么在 MLIR 流水线之后还需要一个 PTX 后处理层

前两章（第 2、3 章）分别解剖了 L1 算法 IR 与 MMA 发射机器；本章下移一层，解剖整条构建链的**最后一个编译器**——一个 856 行的纯文本 PTX 改写器 `tools/gdn_smem_fix.py`，连同它上游的 `scripts/run_l2.sh`（134 行编译驱动）与下游的 `tools/gdn_full_rt.py`（运行时/缓存键控）。这一层的存在不是工程惯性，而是被 stock MLIR 流水线（`-gpu-lower-to-nvvm-pipeline`，最终 fork 的 2026 年 10 月 LLVM 主干）的**四个真实缺口**逼出来的：前两个缺口让"未经后处理的 cubin 在真实硬件上根本无法正确运行"，后两个缺口让"后处理后的 cubin 无法达到驻留（occupancy）与发射效率的目标"。四个缺口全部以 GPU 探针取证或源码级验证钉死，无一来自推测：

**缺口一：workgroup 分配下译为设备端 `malloc`（正确性缺口，致命）。** payload 中每一条作用于 `#gpu.address_space<workgroup>` 的 `memref.alloc()`，在被 `-gpu-lower-to-nvvm-pipeline` 下译后都不是静态共享内存符号，而是一段**设备端堆分配调用序列**：`call.uni (retval0), malloc, (param0)` 取回通用地址空间的堆指针，再经 `cvta.to.shared` 把该通用指针转换成共享窗口偏移。这是上游已文档化的行为（`llvm-mlir/test/Dialect/GPU/memref-to-llvm.mlir` 即按此形态生成期望输出）。它在真实硬件上有两条独立的死亡路径：

1. 裸 ptxas 产物不链接设备运行时（cudadevrt），cubin 中残留一条指向 `malloc` 蹦床函数的未解析 `CALL.ABS`——模块加载或内核启动直接失败（`CUDA_ERROR_INVALID_VALUE`）；
2. 即便 `malloc` 得以解析，返回的堆指针被**按共享内存窗口偏移重新解释**，后续 `cp.async` / `ld.shared` 落在共享地址空间之外——同步点上报 `CUDA_ERROR_ILLEGAL_ADDRESS`，或静默读出垃圾。

首次硬件执行（evidence/G3/calib_report.md 第 2 节）的取证结论是：**所有携带共享内存 tile 的 G2 cubin，在未修复形态下全部陷阱**（`CUDA_ERROR_ILLEGAL_ADDRESS` 或启动期 `INVALID_VALUE`）；只有两条不含 workgroup 分配的参考内核（sm80 原子、ldmatrix 锚）能正常启动。缺口一因此不是"性能不达标"，而是"零通过率"。

**缺口二：常数提升导致内核参数爆炸（正确性/效率混合缺口）。** `-transform-interpreter`（tiling 调度）与 `gdn-opt --gdn-lower-hatch`（本仓方言 op 的下译通道）都会把 gpu.launch 区域内部的 `arith.constant` **重新提升回宿主函数**；随后 gpu-kernel-outlining 把其中每一个都变成一条内核参数——单个内核最多膨胀出约 140 条 `.b64` 的编译期常量参数，甚至一个 bf16 填充值都会变成 `.param .align 2 .b8 [2]`。这些参数的值在编译期完全已知，运行时逐条经 `ld.param::entry` 读入既浪费发射带宽，又让"零参数内核 + 公共全局缓冲区"的 noargs 启动契约（第 5 章的输入输出 staging 设计）无从谈起。

**缺口三：寄存器帽不可经 ptxas 命令行施加（驻留缺口）。** 2 CTA/SM 驻留要求 `2 × 256 线程 × 128 寄存器 = 65,536` 恰好占满寄存器堆，即每内核寄存器数必须**逐内核**钉在 128。但本链选定的 `ptxas-blackwell`（Triton 捆绑、PTX ISA 9.3 能力，为 `sm_120a` 与 `cp.async.bulk.tensor` 所需）**无视 `--maxrregcount` 命令行旗标**——实测在帽值 128 下仍然编出 168 寄存器；而 PTX 源内的 `.maxnreg` 指令是**逐 entry** 生效且被尊重的（§4.3）。寄存器帽因此必须由 PTX 后处理层注入。

**缺口四：bank 几何约束编译器不可见（发射效率缺口）。** 生成器把 smem→global 的搬运发射为逐 cell 的 `ld.shared.b16`/`st.global.b16` 对（本工具链中不存在 128 位全局存储通路：`vector.load/store` 能 lower 但被逐元素拆成 b16 op，见第 5 章取证）。把相邻 b16 对融合成 b32 宽存取可以减半 LSU 指令与循环迭代数，但融合合法性取决于**共享内存 bank 几何**——tile 行距 80B（20 字）时，一个 warp 的 32 次 b32 共享读横跨两行，lane 28–31 与 lane 0–3 落进同一组 bank，2 way 冲突反而净慢 4.3 μs。编译器既不知道行距的语义，也不会为 bank 几何重排 lane→cell 映射；这条边界只能在 PTX 层（融合规则）与生成器层（lane 重映射）联手关闭（§4.4）。

四个缺口合起来界定了这一层的职责：**在 MLIR 语义完整下译之后、ptxas 定案之前，以纯文本改写修复正确性（缺口一、二）、施加驻留约束（缺口三）、收割发射效率（缺口四），并保证每一步改写的字节级可验证性。**

### 4.0.1 构建链全景（最终形态）

理解本章所有变换的宿主，是下面这条两遍 ptxas 的构建链。以一个配置目录 `<dir>`（含 `payload.mlir`、`transform.mlir`、`config.json`，由 `gdn_full_gen.py` / `gdn_bwd_gen.py` / `gdn_l2_gen.py` 产出，可选伴随 `smem_alias.json`）为单位：

```
生成器 (payload.mlir + transform.mlir [+ smem_alias.json])
  │
  ▼  scripts/run_l2.sh
[gdn.* hatch op 存在时] gdn-opt --gdn-lower-hatch        # 方言 op → 树内 nvgpu/nvvm
[transform 脚本含 structured/nvgpu/memref 时] mlir-opt -transform-interpreter
[linalg 残存时] mlir-opt -convert-linalg-to-loops
mlir-opt -gpu-lower-to-nvvm-pipeline=
        "cubin-chip=sm_120a cubin-features=+ptx87 cubin-format=assembly"
  │        └─ cubin-format=assembly → gpu.binary 内嵌 PTX 文本属性
  ▼
asmobj.mlir ──(gdn_extract_ptx.py)──► kernel.ptx
  │                                      │
  │                                      ▼  第一遍 ptxas（-v -arch=sm_120a）
  │                                  kernel.cubin + ptxas_v.txt (+ sass.txt, meta.json)
  ▼  tools/gdn_smem_fix.py（本章主角）
extract_all_ptx(asmobj.mlir)  或  直读 kernel.ptx
  ├─ Fix 1  静态 smem 修复（callseq 删除 + cvta 重写 + .shared 声明，§4.1）
  ├─ Fix 1+ 生存期竞技场别名（smem_alias.json 消费，§4.2）
  ├─ Fix 2  常数参数消除（标量 + 聚合常数字节替换，§4.6）
  ├─ merge_ptx 多模块合并（≥2 个 gpu.binary 时，§4.5）
  ├─ widen_stores / shuffle_pair_stores（GDN_WST / GDN_WSP，§4.4）
  └─ apply_maxnreg_k（GDN_MAXREG_K 逐 entry .maxnreg 注入，§4.3）
  │
  ▼  第二遍 ptxas（-v -arch=sm_120a）
kernel_smemfix.ptx + kernel_smemfix.cubin   ←—— 计时/验证装载的最终产物
```

要点先定义后用：

- **第一遍 ptxas 与第二遍 ptxas 的产物不同**。`kernel.cubin`/`ptxas_v.txt` 来自 `run_l2.sh` 对未修复 `kernel.ptx` 的编译；真正装载执行的 `kernel_smemfix.cubin` 来自 `gdn_smem_fix.py` 的第二遍编译。两遍给 ptxas 的输入不同，调度自由度不同，**资源报告不可互换**——这一"双遍教训"本身是 §4.3 的主角。
- **ptxas 选型**（`pick_ptxas()`，`tools/gdn_smem_fix.py:64`；`scripts/run_l2.sh:27`）：优先 Triton 环境捆绑的 `ptxas-blackwell`，其次捆绑 `ptxas`，最后回退 `/usr/local/cuda/bin/ptxas`。两条链保持一致，避免不同 ptxas 版本的 ISA/调度差异污染 A/B。
- **幂等性**：`gdn_smem_fix.py` 覆盖式重写 `<dir>/kernel_smemfix.{ptx,cubin}`；既不需要修复的内核原样拷贝，保证计时harness 永远有可装载的目标（模块 docstring 明文契约）。
- **单内核与多内核双入口**：`main()` 先尝试 `main_multi()`——当 `asmobj.mlir` 含 ≥2 个 `gpu.binary` 汇编属性（mono 三元组，§4.5）时走"逐模块修复 + 合并 + 单 cubin"路径；否则退回单内核路径，直读 `kernel.ptx` 并消费平铺结构的 `smem_alias.json`。

以下每节按统一四件套展开：【动机 → 源码片段（路径）→ PTX 前后对比 → 效果数字】。

## 4.1 静态共享内存修复（Fix 1 主体）：device `malloc` → 静态 `.shared` 符号

### 4.1.1 动机与修复思路

缺口一（§4.0）的修复思路是**复刻 MLIR 自身为静态共享内存已经使用的习惯用法**：ws 配置的 mbarrier 下译本来就会发射静态 `.shared` 符号——`run_l2.sh` 的上游流水线对 `gpu.barrier` 相关 lowering 正是这么做的。修复因此不是发明，而是把"每个 workgroup `memref.alloc` 一段设备堆分配"统一改写为"每个分配结果一个静态 `.shared` 符号"：

1. 删除每一段 malloc 调用序列（callseq）；
2. 把引用该分配结果的每一条 `cvta.to.shared` 换成 `mov.u64 %rd, __gdn_smemN;`；
3. 在 entry 前声明 `.shared .align 128 .b8 __gdn_smemN[<malloc 尺寸>];`。

分配尺寸与结果寄存器的配对信息就在被删除的 callseq 自身之内，无需读取任何外部文件——这保证了 Fix 1 对**所有**内核无差别生效（不需要生成器配合）。

### 4.1.2 源码片段

三个正则锚点加两个改写函数构成全部机制（`tools/gdn_smem_fix.py:73-209`）：

```python
CALLSEQ = re.compile(
    r"\t\{ // callseq (\d+), 0\n"
    r"((?:\t[^\n]*\n)+?)"
    r"\t\} // callseq \1\n")  # closing brace omits the ", 0" suffix
SIZE_RE = re.compile(r"st\.param::func\.b64[ \t]+\[param0\],[ \t]+(\d+);")
RET_RE  = re.compile(r"ld\.param::func\.b64[ \t]+(%r[d]\d+),[ \t]+\[retval0\];")
CVTA_RE = re.compile(
    r"\tcvta\.to\.shared(?:::\w+)?\.u64[ \t]+(%r[d]\d+),[ \t]+(%r[d]\d+);")
```

`eat_callseq`（`gdn_smem_fix.py:101`）对每段 callseq 判定：体中不含 `malloc` 的调用序列原样保留（该正则也覆盖非 malloc 的设备调用）；含 malloc 的则从 `st.param::func.b64 [param0], <N>` 提取尺寸、从 `ld.param::func.b64 %rdX, [retval0]` 提取结果寄存器，登记 `malloc_reg_size[%rdX] = N` 后**整段删除**。`patch_cvta`（`gdn_smem_fix.py:166`）对每条 cvta 判定：源寄存器不在 `malloc_reg_size` 中的（非分配结果，例如 mbarrier 符号的转换）原样保留；在其中的，为该分配结果**首次**铸造符号 `__gdn_smem{len(decls)}` 并登记尺寸，随后统一替换为 `mov.u64`。同一分配结果的**每一条** cvta 转换共享同一符号——这个"每结果一符号"的要点是 §4.1.4 的坑的直接对策。最后把全部声明插在 `.visible .entry` 标记之前（找不到该标记则拒绝改写，防止把符号声明插进指令区）。

### 4.1.3 PTX 前后对比

修复前（`evidence/G4/gen_verify/kkt_bf16_mma_p2_t64x64x32_w4/kernel.ptx:35-42`，上游流水线的原始输出）：

```ptx
	{ // callseq 0, 0
	.param .b64 	param0;
	.param .b64 	retval0;
	st.param::func.b64 	[param0], 8192;
	call.uni (retval0), malloc, (param0);
	ld.param::func.b64 	%rd24, [retval0];
	} // callseq 0
	cvta.to.shared.u64 	%rd5, %rd24;
```

修复后（同目录 `kernel_smemfix.ptx`；声明区第 18–19 行与指令区第 37–38 行）：

```ptx
	.shared .align 128 .b8 __gdn_smem0[8192];
	.shared .align 128 .b8 __gdn_smem1[8192];
	...
.visible .entry gdn_kkt_kernel_kernel()
	...
	mov.u64 	%rd5, __gdn_smem0;
	mov.u64 	%rd6, __gdn_smem1;
```

八条字节的堆分配序列与指针转换消失，取而代之的是一次立即数装载。对 TMA 配置（`kkt_bf16_mma_tma_ws`），`cp.async.bulk.tensor` 目的地址的转换是**集群味**的（`kernel.ptx:389-390`）：

```ptx
	cvta.to.shared::cluster.u64 	%rd196, %rd221;
	cvta.to.shared::cluster.u64 	%rd197, %rd220;
```

修复后同样落到静态符号（`kernel_smemfix.ptx:371-372`）：

```ptx
	mov.u64 	%rd196, __gdn_smem1;
	mov.u64 	%rd197, __gdn_smem0;
```

### 4.1.4 `cvta.to.shared::cluster` 也必须改写：G4 的第二个正确性坑

初版 `CVTA_RE` 只匹配裸 `cvta.to.shared.u64`。G4 硬件验证暴露了后果：TMA 配置的两个 `cp.async.bulk.tensor` 目的地址下译为 `cvta.to.shared::cluster.u64`，且**转换的是同一个 malloc 结果**；callseq 删除后，这两条指令读取的是**从未定义的寄存器**——垃圾共享地址直达 `UTMALDG` 的目的操作数，GPU 上报 `CUDA_ERROR_ASSERT`。修复即正则中的 `(?:::\w+)?` 可选组：裸味与 `::cluster` 味一并匹配，且因符号按"分配结果"而非"指令"铸造，同一结果的两种转换自动共享同一符号（`patch_cvta` 注释原文记录了这条取证："leaving those unpatched made them read registers whose defining malloc callseq was already deleted => garbage smem destination => device fault"，`gdn_smem_fix.py:170-176`）。

### 4.1.5 128B 对齐规则：UTMALDG 的硬约束

声明的对齐从初版的 `.align 16`（忠实复刻 mbarrier 习惯用法）提高到 `.align 128`，依据是 G4 的第三个硬件取证：`cp.async.bulk.tensor`（SASS `UTMALDG`）的共享内存目的地址在 `sm_120a` 上**即使 swizzle=none 也要求 128 字节对齐**。`.align 16` 时 ptxas 可以把某块 tile 放在 16B 对齐但非 128B 对齐的偏移上，首次 TMA 装载即故障 `CUDA_ERROR_MISALIGNED_ADDRESS`——compute-sanitizer 把故障点精确钉在 `UTMALDG.4D`（`evidence/G4/GATE_REPORT.md` §2.3）。代码内注释（`gdn_smem_fix.py:199-203`）将此规则固化：

```python
    if decls:
        # .align 128: cp.async.bulk.tensor (UTMALDG) smem destinations must
        # be 128-byte aligned even with swizzle=none on sm_120a (G4: with
        # .align 16 the tiles can land at 16B-but-not-128B offsets and the
        # first TMA load faults with CUDA_ERROR_MISALIGNED_ADDRESS).
        lines = [f".shared .align 128 .b8 {s}[{n}];" for s, n in decls]
```

该规则后被推广为纪律：任何未来经 memref.alloc → device malloc → smem_fix 改写而来的 TMA 目的 tile 都自动获得 128B 对齐；手写 PTX 必须保持之（G4 报告 §5 明文）。

### 4.1.6 效果数字

- **通过率从 0 到全量**：G3 首跑中所有携带 smem tile 的未修复 cubin 全部陷阱；修复后全部内核运行并产出非零输出（`evidence/G3/calib_report.md` §2 Gap 1：18 个可计时配置 OK，两个 ws/tma 配置由 G4 进一步修复后完成）。
- 与 Fix 2（§4.6）合并后，**18/20 配置成为真正的零参数内核**；ws 配置保留 3 条封送常数（2× i1 + 一个 16B 向量零），tma 配置保留 3 条宿主侧 tensormap 指针参数（经 `cuTensorMapEncodeTiled` 装配）。
- 修复是纯字节级重写：不触碰任何计算指令，G4 的 20/20 逐配置硬件验证（含确定性重跑）在其之上完成。

## 4.2 生存期竞技场别名（arena aliasing，Fix 1 扩展）

### 4.2.1 动机：驻留壁垒与"字节记账"

`sm_120a`（RTX 5090 Laptop，82 SM）的共享内存经济：每 SM 102,400B、每块 optin 上限 101,376B、校准实测 optin 共享 99KiB（`evidence/G3/calib_report.md` §1）。**2 CTA/SM 驻留**因此要求每内核静态共享池 ≤ 约 50.2–50.7KB（反向链记账门槛 50,176B、正向链 50,688B），同时寄存器 ≤ 128（2×256×128 = 65,536）。第 6、7 章反复出现的困境是：算法需要的 tile 生存期拼接起来远超单块上限，但**任意时刻真正活跃的字节**却在门槛之下。

MLIR 层面无法直接表达"多块 tile 复用同一片字节"：每块 tile 一个 `memref.alloc` 是 SSA 合法性的要求（第 3 章的 staging 原语族都按此发射）。但共享内存的物理事实不关心 SSA——**生存期不重叠的 tile 可以共享字节**。arena aliasing 就是把这一定理施加在 PTX 层：MLIR 程序保持每 tile 一次分配（生成器语义完整、可验证性不变），后处理把被别名 tile 的 malloc 删除、其 cvta 改写为"竞技场基符号 + 字节偏移"。生成器侧的生存期审计方法学（峰值生存期审计、逐相位活跃表、三条硬规则）由第 7 章展开；本节只讲机制端——`gdn_smem_fix.py` 如何消费规范、如何对抗 LLVM 的三项"不配合"，以及防御性回退立场。

**定义**：一个**竞技场（arena）**是一块以某个宿主 tile 名义声明、按竞技场总尺寸（而非宿主 tile 自身尺寸）声明的静态 `.shared` 符号；**别名（alias）**规范描述"tile T 的字节落在宿主 H 的竞技场内偏移 off 夕"。

### 4.2.2 规范格式：`smem_alias.json`

单内核路径（`main()`，`gdn_smem_fix.py:819-824`）直接读取配置目录下的 `smem_alias.json`；多内核路径（`main_multi()`，`gdn_smem_fix.py:764-767`）读取其 `"modules"` 子映射按模块名分发。三个键（`fix_ptx` docstring 与 `gdn_smem_fix.py:114-151`）：

```json
{
 "identify": {"16384": "accCC", "9216": "pthi", "9504": "ptlo", ...},
 "arenas":   {"accCC": 34816, "vnh": 10240},
 "alias":    {"pthi": ["accCC", 0], "bdt": ["accCC", 9216], "akb": ["accCC", 18944], ...}
}
```

真实样本取自最终 dqkwg 2-CTA 链（`evidence/G5/scratch/r15w_2cta_T2048_dqkwg/dqkwg/smem_alias.json`）：17 个 identify 条目、2 个竞技场（accCC 34,816B、vnh 10,240B）、16 个别名条目——包括 bhh@13,824、bhl@16,384（+1/+3 行垫，见 §4.2.3）、akb@18,944、accCK@24,064。该规范的作者在生成器（`gdn_bwd_gen.py`/`gdn_full_gen.py`），本工具是纯消费者。

### 4.2.3 按 malloc 尺寸识别：LLVM 重排的对策与"识别垫"

机制端第一问题是**把规范里的 tile 名对应到 PTX 里的具体 malloc**。不能用顺序：**LLVM 会重排 malloc callseq，顺序不是稳定键**（`fix_ptx` docstring 明文；这是实测行为而非猜测）。对策是**按唯一分配尺寸识别**——每个 tile 的 malloc 尺寸在生成器侧被垫成全内核唯一，于是尺寸即身份：

- ldmatrix 消费的 tile：垫**从不寻址的额外行**——行距必须保持 16B 的倍数（ldmatrix 行地址 16B 对齐是硬约束，G4/G5 双重取证；奇数列垫非法，垫必须按 8 列步进，正向链的一组尺寸 11,264/12,288/13,312/14,336/1,024/1,280/18,432 即此规则的产物）；
- 仅通用访存的 f32 tile：+4 列垫。

对应代码（`gdn_smem_fix.py:115-151`）：按尺寸建 `size_regs` 倒排表；每个 identify 条目要求**恰好一个**同尺寸 malloc。任何失配走防御性回退（§4.2.6）。

### 4.2.4 cvta → 竞技场基 + 偏移的重写

识别成功后，`patch_cvta` 对别名 tile 的每条 cvta 发射两条指令：基符号装载加常量偏移（`gdn_smem_fix.py:178-184`）：

```python
        tile = tile_of_reg.get(src)
        if tile is not None and tile in alias_of:
            base, off = alias_of[tile]
            sym = base_sym(base)
            n_patch += 1
            n_alias += 1
            return (f"\tmov.u64 \t{dst}, {sym};\n"
                    f"\tadd.u64 \t{dst}, {dst}, {off};")
```

`base_sym`（`gdn_smem_fix.py:157-164`）为竞技场宿主铸造**按竞技场尺寸**声明的符号（`arena_of[base_tile]`）；宿主自己的 cvta 也落在同一符号上（偏移 0 的别名等价于宿主自身）。

最终 dqkwg 链的真实 PTX（`evidence/G5/scratch/r15w_2cta_T2048_dqkwg/dqkwg/kernel_smemfix.ptx`）。改写后的别名地址（akb@accCC+18,944 与 accCK@accCC+24,064）：

```ptx
	mov.u64 	%rd24, __gdn_smem9;
	add.u64 	%rd24, %rd24, 18944;
	...
	mov.u64 	%rd12, __gdn_smem9;
	add.u64 	%rd12, %rd12, 24064;
```

声明区（同文件 37–48 行）展示最终池形态——10 个 256B 标量块 + 两个竞技场，合计 **47,616B**：

```ptx
	.shared .align 128 .b8 __gdn_smem0[256];
	...
	.shared .align 128 .b8 __gdn_smem9[34816];
	.shared .align 128 .b8 __gdn_smem10[256];
	.shared .align 128 .b8 __gdn_smem11[10240];
```

### 4.2.5 LLVM 的三项"不配合"与逐项对策

机制端的全部趣味在于：规范描述的是 MLIR 里的分配，而改写发生在 LLVM 已经优化过的 PTX 上。三项实测的"不配合"各有专门对策，且全部**响亮而不沉默**（每项都有打印诊断）：

**（一）malloc 重排。** 上节已述：顺序不可作键，唯一尺寸作键。

**（二）未用宿主被 DCE。** 一个"宿主"分配若在 MLIR 语义里没有任何使用（它的存在只是为了给竞技场提供名义），LLVM 会把它**整体删除**——尺寸键找不到任何 malloc。对策不是发明使用，而是**规定宿主必须是首个使用（first-USE）tile**：宿主自己必须有真实数据流动。规范侧的这条纪律在 bwd 报告中作为工具链教训 #1 记录（"an arena HOST alloc with no uses is DCE'd by LLVM and the size key finds 0 mallocs -- arena bases must be first-USE tiles"）。

**（三）store-to-load 前传吞噬别名 tile。** 更隐蔽：当一条链的 epilogue 把 tile A 的内容经共享内存移交给同一 warp 同 lane 映射的下一条链时，LLVM 的 store-to-load 前传会把这次交接**直接前传进寄存器**——于是该 tile 的 malloc 在 PTX 里根本不存在。wyb 的 t2 对（dk_t2 停靠对）即此情形。早期行为是硬 WARN 并**整体禁用别名**，后果是一条 121KB 静态共享的 ptxas 失败；现在的行为（`gdn_smem_fix.py:121-131`）是把该 tile 从别名映射中摘除并继续（数值上无损——前传本身保持了值与舍入链）：

```python
            if len(regs) == 0:
                # G5-T round13: the malloc was optimized away ENTIRELY
                # (e.g. LLVM store-to-load-forwarded a same-lane epilogue
                # handoff into registers -- the wyb t2 pair).  There is no
                # cvta to patch and no bytes to assign: drop the tile from
                # the alias map and continue.
                print(f"[smem_fix] NOTE alias: size {sz_s} for '{tile}' "
                      f"matches 0 mallocs (optimized away) -- skipped")
```

机制端还从验证门学到两条**别名合法性纪律**（生成器侧规范必须遵守，第 7 章展开细节）：其一，**分裂 tile 不得别名其自身源 tile**（dqkwg 的 accCC 自别名 bug 被验证门捕获）；其二，别名区不得压在多缓冲 drain 尚未完成的字节之下（正向 kt 竞技场案例：solve tile 压在偏移 0 处使每个 CTA 的 u 行 0–31 损坏——流水化 kkt 循环在循环后 drain 期仍写 kt 的多缓冲，10,240B 之下必须假想有晚写；`bisect_shift` 探针二分定罪后把 solve tile 上移到 kt 真实字节之上，逐位精确）。

### 4.2.6 防御性回退立场

任何识别失配（某尺寸匹配多个 malloc、规范引用了未识别 tile、别名映射为空）都会**丢弃整个规范**并回退到"每 malloc 一声明"的经典行为——`gdn_smem_fix.py:132-151` 的 WARN 路径。回退结果**正确但未重叠**：内核照常运行，只是不省字节。设计立场是：别名是纯增益变换，其失败模式必须是"不赚"，绝不许是"算错"。全部数值等价性由此主张并由验证门背书（见效果节）。

### 4.2.7 效果数字

arena aliasing 是整条反向链从 6.0ms 走到 2.2ms 的核心杠杆之一，关键落点：

| 内核 | 池变化 | 寄存器 | 驻留 | 阶段加速（同窗配对） | 证据 |
|---|---|---|---|---|---|
| dqkwg（r9 首落，两竞技场 accCC 17,408 / vnh 10,240） | 99,328→48,640B | 132→96 | 1→2 CTA | 1.772→1.075ms（1.65×） | bwd_report r9 |
| dqkwg（最终四 tile 别名 bhh/bhl/akb/accCK 入 arena1） | **100,864→47,616B**（低于 50,176 门槛 3,560B） | 128（.maxnreg，§4.3） | 2 CTA | 0.632→0.547ms @T2048（1.16×）；2.323→1.844 @T8192（1.26×） | bwd_report r15 |
| caryA（竞技场"溶解式"瘦身：四个共活 tile 退役为三个 bf16 mma tile + 别名） | **82,176→28,592B** | 255→76 | 1→2 CTA | 0.309→0.0618ms（5.00×）@T2048；2.78× @T512 | bwd_report r11 |
| wyb（tv 3-mma 入 tT/accCK 竞技场，22 malloc→17 声明，其一按竞技场尺寸 18,432） | 98,048→100,096（+T 分裂对真实足迹）；再瘦身 100,096→**45,056B** | 226→128（176B/228B 溢出保留） | 1→2 CTA | 0.817→0.3395ms（2.41×） | bwd_report r11/r13 |
| fwd f1（F1AL 移植，13 malloc→5 声明） | 101,376→89,600B | — | 1 CTA（算术上 2-CTA 不可达） | **诚实负结果**：f1 +4.7μs，旋钮默认关 | fwd_report r15 |
| fwd f3（F3KS：ot 压入 P 的死竞技场，P 的 malloc 以 ot 的 17,408B 声明、按尺寸别名于 +0） | 82,176→47,360B | — | 2 CTA | **诚实负结果**：f3 +96.1μs @T8192（带宽-bound 内核不受益于波减半） | fwd_report r20 |

dqkwg r9 的 ncu 前后对照给出机制证据：理论/实际占用 16.67%/15.81%→33.33%/30.28%，实际 warp/SM 7.59→14.53，调度器活跃 warp 1.98→3.69，无资格周期 88.98%→79.56%。r15 的 2-CTA 落地版本：占用 15.02%→29.17% @T2048、14.23 warp/SM @T8192（此前 8.00）。所有落点数值位级不变或按章内验证门裁决（第 8 章）；两条负结果（f1/f3 正向）同样按测量关闭——机制本身可复用，是"问题被算术/测量关闭"而非"机制失效"。

## 4.3 `.maxnreg` 注入：ptxas-blackwell 的旗标盲区与双遍教训

### 4.3.1 动机

2 CTA/SM 的寄存器算术（2×256×128=65,536）要求把逐内核寄存器数钉在 128。工具侧本来备有探针旋钮 `GDN_MAXREG`：给第二遍 ptxas 追加 `--maxrregcount`（`gdn_smem_fix.py:797-799/839-842`）。实测结论（bwd 报告 r15 原文）："**ptxas-blackwell IGNORES --maxrregcount**（GDN_MAXREG re-verified useless: 168 regs at cap 128），but the PTX **.maxnreg directive placed inside the entry body (next to .maxntid) IS honored**"。命令行帽是"发了但没人听"，PTX 指令帽才有效——帽的施加点因此必须移入 PTX 文本。

### 4.3.2 源码片段与 PTX 前后

`apply_maxnreg_k`（`tools/gdn_smem_fix.py:683-712`）读取 `GDN_MAXREG_K`（形如 `f3:168,f2:128`），对每个 `.visible .entry NAME` 行按**子串匹配**注入：

```python
    for ln in lines:
        out.append(ln)
        m = re.match(r"\s*\.visible\s+\.entry\s+([\w.$]+)", ln)
        if m:
            name = m.group(1)
            key = next((k for k in want if k in name), None)
            if key:
                out.append(f"\t.maxnreg {want[key]}")
                applied[name] = want[key]
```

逐 entry 的键控是必须的：`--maxrregcount` 是**逐文件**旗标，对携带三个内核（f1/f2/f3 各有不同寄存器预算）的 mono cubin 无差别帽没有意义；`.maxnreg` 是**逐内核**指令（docstring 原文：单内核链另有构建脚本侧的 `GDN_MAXNREG_ENTRY=dqkwg:128` 同型注入）。注入位置紧随 entry 头（与 `.maxntid` 并排）。真实产物（`evidence/G5/scratch/r15w_2cta_T2048_dqkwg/dqkwg/kernel_smemfix.ptx.maxnreg.ptx:49-51`；该目录为后续复测所重建，记录值 128，此处样本为 96 类再探针）：

```ptx
.visible .entry gdn_dqkwg_kernel_kernel()
.maxntid 256, 1, 1
	.maxnreg 96
{
```

注入后不匹配（wanted ≠ applied）时打印 WARN，防止键拼错静默失效。

### 4.3.3 双遍教训：`ptxas_v.txt` 只是第一遍

这条教训值得单独成节，因为它是"构建链说谎"的典型案例。r14 的构建脚本从 `ptxas_v.txt` 抄录寄存器数并记档"128 regs / 44B spill"。但 `ptxas_v.txt` 是 **run_l2.sh 第一遍** ptxas（未修复 `kernel.ptx` → `kernel.cubin`）的报告；实际装载的 `kernel_smemfix.cubin` 出自**第二遍**（对 smem_fix 改写后 PTX 的编译），改写后的 PTX 给了 ptxas 不同的调度自由度，2-CTA 机身在第二遍编出 **168 寄存器（0 溢出）**。第一个 A/B 因此"莫名"倒退 0.78×（0.811ms）：ncu 显示 Block Limit Registers = 1（2×256×168 > 65,536）——名义 16-warp 的形式实际跑 8 warp/SM，严格劣于 r14。两条工具链教训（bwd 报告 r15 原文要点）：

1. **做数的寄存器数是 smem_fix 那一遍的**；`ptxas_v.txt` 不得用于对它下结论（r14 记档的 128 恰是第一遍的数；r14 已装载的 cubin 经复跑第二遍验证确为 128/0 溢出）。
2. 帽必须经 `.maxnreg` 施加；注入后 128 寄存器 / 32B 溢出（barrier 瘦身后 64B）。

r16 起此教训**结构化**进构建链：`r16y_build.sh` 把第二遍的 `-v` 报告记入 `ptxas_smemfix_v.txt`，寄存器数在 bench 之前就对账。其对决策的直接影响有实例：dqkwg P11R 重构的 P11R=4 臂（自然 222 寄存器）在帽 128 下需 232B 栈 / 276B 溢出存 / 632B 溢出取，仅 1.026×；P11R=2（自然 174）96B 栈 / 96/148B 溢出，1.040× 胜出——两个数字都在 bench 前由双遍纪律暴露。

### 4.3.4 效果数字

- dqkwg：注入 `.maxnreg 128` 后 ncu Block Limit Registers = 2、Block Limit Shared Mem = 2——**2 CTA/SM 双门槛同时达成**；占用 15.02%→29.17% @T2048，14.23 warp/SM @T8192（8.00 起），无资格周期 89%→83%，调度器活跃 warp 1.97→3.63，SM 吞吐 18%→30%、MEM 19%→36%（bwd r15）。
- 阶段收益（与 §4.2 的池瘦身叠加）：dqkwg 0.632→0.547ms @T2048（1.16×）、2.323→1.844 @T8192（1.26×，2048 CTA 填满加倍后的 SM 槽，waves 6.24→3.12）。
- 边界纪律：`.maxnreg` 注入是 2-CTA 臂的需求——T512 类小网格的 v3 臂 cubin 与记录逐字节一致，构建时用 `GDN_MAXNREG_ENTRY=nomatch:0` 空过。
- 负面对照：f3 曾以 `GDN_MAXREG_K="f3:168"` 探针——240 寄存器本可 2 CTA（240×128×2≤65,536）但 82,176B×2 > 100KB 共享不允许，寄存器帽无法移动占用；工具照 landing 供后续探针（fwd r14"non-lever, corrected from the brief"）。这是"帽只在寄存器是短板时才是杠杆"的实测注脚。

## 4.4 宽存取四元组融合（GDN_WST）与 W32P 互补半区重映射（GDN_W32P）

### 4.4.1 动机：cell pass 的 LSU 账本

本工具链不存在 128 位全局存储通路（第 5 章取证：`vector.load/store` 与 `vector<8xbf16>→vector<4xi32>` 位拆变体都能 lower，但被逐元素拆成 b16 op）。生成器因此把 smem→global 搬运发射为**逐 cell 的 b16 单元**。删除探针（timing-only 构建，不装载）把账本钉死（fwd r15）：f2 的两条 cell pass（h_all、v_new）占 T8192 297μs 链中的 **35.4μs + 17.9μs**（合计约 53μs 上限）；f1 的 u/w pass 每 CTA 还要长一倍（C×V + C×K 个 cell）。WST 攻击指令数；W32P 解决 WST 在 80B 行距 tile 上被 bank 冲突挡住的那一半。

### 4.4.2 WST：四元组融合的源码与规则

生成器侧先配对：`copy_s2g_vec` 的配对 cell 模式按 `(2i, 2i+1)` 发射相邻 cell 对（`cols >= 64` 且 `rows*cols % (2*nthr) == 0`，否则回落单 cell 体）。PTX 侧 `widen_stores()`（`tools/gdn_smem_fix.py:482-563`，`GDN_WST` 启用、`GDN_WST_SCOPE` 按 entry 名圈定）把每个**无条件四元组**

```
ld.shared.b16 [B1+o]; ld.shared.b16 [B1+o+2]; st.global.b16 [B2+g]; st.global.b16 [B2+g+2]
```

熔合为

```
ld.shared.b32 [B1+o]; st.global.b32 [B2+g]
```

合法条件（源码逐一检查，否则该位点不动）：两个 b16 源寄存器**均单次使用**；两条 store 的全局地址相邻（`ob == oa+2`）；**两种 store 顺序都接受**——LLVM 常把 +2 的 store 排在前面；顺序健全性要求每条 load 先于自己的 store、低偏移 load 先于高偏移 load，而 **store 顺序不受约束**。字节精确性由构造保证：熔合后的 32 位寄存器恰是两个相邻 16 位 cell（小端），`st.global.b32` 把相同字节写进相同地址（生成器的偶对 cell 映射保证 4B 对齐）。铸造的新 b32 寄存器从 9000 起编号（"远高于 LLVM 的编号"），并以**显式逗号清单**声明——本 ptxas 拒绝 from-to 区间式 `.reg` 声明，计数式无法点名高编号（`gdn_smem_fix.py:554-562`）。

### 4.4.3 PTX 前后对比（真实产物）

熔合前（`evidence/G5/scratch/r15x/wst2_T8192_w4bv32/kernel.ptx:3668-3672`——注意 +2 的 store 被 LLVM 排在前面）：

```ptx
$L__BB0_93:
	ld.shared.b16 	%r1848, [%rd587];
	ld.shared.b16 	%r1849, [%rd587+2];
	st.global.b16 	[%rd588+2], %r1849;
	st.global.b16 	[%rd588], %r1848;
```

熔合后（同目录 `kernel_smemfix.ptx:3610-3614`；新寄存器声明在第 62 行 `.reg .b32 %r9000, %r9001;`）：

```ptx
$L__BB0_93:
	ld.shared.b32 	%r9000, [%rd587];
	st.global.b32 	[%rd588], %r9000;
	add.s64 	%rd589, %rd589, 128;
	add.s64 	%rd588, %rd588, 512;
```

四条 LSU 指令对两条；循环迭代数同步减半（步长 128B 共享 / 512B 全局）。

### 4.4.4 80B 行距的 bank 冲突律与 `cols >= 64` 规则

WST 的配对在 f1/f3（列数 128）上直接合法：**每 warp 的 32 次 b32 共享读保持在一行之内**。f2 的 BV=32 tile 不行——这就是 r16 bank 冲突律（fwd r16 负结果，r24 复述）：BV=32 tile 行宽 (BV+8)=40 元素 = **80B = 20 字**行距；平铺配对映射下一个 warp 横跨**两相邻行**，20 字的行距把 lane 28–31 的字地址搬进 lane 0–3 所在的 bank 组——**2 way 冲突**，单 cell 映射本没有。实测配对（仅映射、不熔合）在 f2 上 **+4.3μs（47/60 对更慢）**。WST 的作用域规则因此定为 `cols >= 64`；f2 保留原 pass。任何未来的 f2 存储通路工作必须先解决这个几何（r16 原文把它列为文档化约束）。

### 4.4.5 W32P：`80 mod 32 = 16` 互补半区律

r24 关闭了这条 frontier：路线 (a)——BV=64 重铺——在纸面上死亡（F2US 之下池 115,200B > 101,376B optin，且恰是 r14 在 w8/bv64 实测 ptxas 失败的那个数；退 F2US 去凑 96,768B 则同时 surrender r14 实测的 −104.5μs 与减半 f2 网格）。路线 (b)——**逐行 warp 重映射**——落地为生成器旋钮 `GDN_W32P`（`gdn_full_gen.py` 的 `copy_s2g_vec` 第三模式；作用条件 `cols==32 and rows%8==0 and nthr==128`，即仅 f2 的 S/vn tile；f1 的 pass 列数 = V/K = 128、f3 的 = 128，本就 WST 配对）。

**互补半区律**：保持**单 cell 冲突无关的共享读**原样（同地址、同 bank 图样——这正是杀死 tile 级配对的东西，一点不动），把配对搬到 **lane 之间**：对 pair-step `i`（cell 计数器），

```
row     = 8*(i>>7) + warp + 4*(lane>=16)
pair col = 2*(lane%16)
```

即 lane 0–15 读行 `8*round + warp`、lane 16–31 读行 `r+4`。字地址分别落在 `[20r, 20r+16)` 与 `[20r+80, 20r+96)`；**80 mod 32 = 16**，故两个半 warp 恰好命中 **32 个 bank 的两个互补 16-bank 半区**——对每个 `r` 零冲突。映射是双射：(round j, warp w, 半区 h) → 行 `8j+w+4h` 覆盖行 `8j..8j+7`，pair 由 `lane%16` 索引。纯下标算术（lane/warp 常量复用 `%lane`/`%warp`，CSE 出循环），池、tile、LDSM 垫、占用一概不变。两条 b16 load 保持 `[base]`/`[base+2]`、两条 store 保持 `[g]`/`[g+2]`——**结构模式不变**，所以 `widen_stores` 原样把它们熔合：f2 的 smem_fix wst 计数从 0 变 2（f2 的两条存储 pass 循环体与 f1/f3 的 `cols>=64` pass 完全同型地熔成 `ld.shared.b32`/`st.global.b32`）。

这就是"W32P 四元组融合"的完整含义：**W32P 负责让四元组在 80B 行距上合法（bank 无冲突），WST 负责把它熔合**；两层各管一半，缺一则 f2 的存储通路收益不存在。

### 4.4.6 姊妹变换 WSP：跨 lane shuffle 配对（保留在工具中）

同一条 f2 frontier 的另一条技术路线是 `shuffle_pair_stores()`（`tools/gdn_smem_fix.py:578-680`，`GDN_WSP`/`GDN_WSP_SCOPE`，在 `widen_stores` **之后**运行，只处理 WST 熔不掉的单 cell）：保留 HEAD 的单 cell 共享读（本就无冲突），跨 lane 配对——lane 2j 持 pair 的低半、lane 2j+1 持高半；一条 `shfl.sync.bfly.b32 delta=1` 让每个 lane 拿到伙伴的位，偶 lane 用 `prmt` 打包 own|partner<<16 发一条 `st.global.b32`，奇 lane 不存。合法性门（每处检查，否则跳过）：load 与 store 同基本块（中间无标签）；外层 while 循环 `STEP % 32 == 0` 且 `(BOUND+STEP) % STEP == 0`（warp 一致迭代，`shfl.sync` 合法——计数器从 `%tid` 起）；store 字节偏移为 0（pair 基址）。W32P 胜出后 WSP 作为已验证的备用路线保留在工具里。

### 4.4.7 效果数字

| 变换 | 对象 | 收益（同窗配对 A/B） | 数值等价 | 证据 |
|---|---|---|---|---|
| WST | f1（u/w pass） | **−11.8/−11.4μs** @T8192（两场，1–5/60 对更慢）、**−10.6μs** @B4T2048（3/60）；staged e2e −13.4/−9.0/−7.0μs | 位级相同（T8192 门 o/ht 0.000e+00；验证 4/4 记录位数） | fwd r15 |
| WST | f3（o pass） | ≈0（−0.4μs，界内） | 同上 | fwd r15/r19 校正 |
| 仅配对不熔合 | f2 | **+4.3μs**（负结果，bank 律的实测注脚） | — | fwd r16 |
| W32P（+WST 熔合） | f2（S/vn pass） | **−17.9…−22.6μs** @T8192（7 场）、−9.7…−25.6μs @B4、−2.0…−3.8μs @T2048；staged e2e −9.1…−12.0μs @T8192；慢对率 1–9/80（B4/T8192） | 位级相同（对 purity 构建逐字节比较；fp64 门记录位数） | fwd r24 |
| W32P 记录迁移 | 全链 | R25 存档 n=30：三形状 formals 全面上移——T8192 **1.349×**、B4T2048 **1.221×**、T2048 **2.125×** | fwd_verify 4/4、det N=5 | fwd r25 |

WST 收益与探针账本自洽：f1 按 0.176μs/MB 计，f2 的 100.6MB pass 写给出约 17μs 上限——W32P 实测 −18…−23μs @T8192 恰好收满。T8192 链上另有一项已定性的 **f3 布局税**（+11.8…+12.6μs，模块布局抽签、f2 的 +300B PTX 触发，f3/f1 的 PTX 段与 purity 逐字节相同）——净链在每一形状、每一场仍为正；R25 的存档重跑重掷了布局。

## 4.5 多模块合并（mono cubin）：重名 `.shared` 符号的静默统一坑

### 4.5.1 动机

正向链的 staged 形态最初是三个独立 cubin（f1/f2/f3），内核边界靠 DtoD 拷贝桥接——B1T2048 下约 0.1ms 纯搬运。mono 形态把三者放进**一个 cubin、共享公共全局缓冲区**，内核边界退化为流序同步。这要求 `gdn_smem_fix.py` 从"修一个模块"扩展为"提取 + 修复 + 合并多个模块"。

### 4.5.2 常量回收：`asmobj.mlir` 汇编属性的反转义

多模块路径的输入不是 `kernel.ptx`（那只含第一个模块），而是 `asmobj.mlir` 本身：`extract_all_ptx()`（`tools/gdn_smem_fix.py:415-447`）扫描每个 `gpu.binary @NAME` 的 `assembly = "..."` 字符串属性，用 `unescape_mlir_string()`（`gdn_smem_fix.py:385-412`）反转义 MLIR 字符串编码——`\xNN`/`\NN` 十六进制对、`\n`/`\t`/`\r`/`\"`/`\\`/`\'` 转义——还原 PTX 文本，并以 `.version` 与 `.target` 的存在性做完整性校验。这段"从 MLIR 工件回收 PTX 常量"的代码与 `run_l2.sh` 保留 `asmobj.mlir` 的决定（G3-A 起）构成同一枚硬币的两面：asmobj.mlir 既是常数参数的值源（§4.6），也是多模块 PTX 的正源。配套的 `parse_all_launch_args()`（`gdn_smem_fix.py:450-473`）对每个 `gpu.launch_func` 分别解析常数参数值。

### 4.5.3 坑：ptxas 对重名 `.shared` 符号的静默统一

合并的最初实现朴素拼接三个已修复模块——每个模块的 Fix 1 都铸造 `__gdn_smem0`、`__gdn_smem1`……GPU 上的后果（fwd 报告 G5-D bug #4 原文要点）：**ptxas 静默统一跨 entry 的重名 `.shared` 符号**——f1/f2/f3 的 `__gdn_smem0` 被当作同一块内存，但三者尺寸不匹配，f3 经由 f1 那个更小的 `__gdn_smem0` **越界写**（compute-sanitizer 定位：shared write at 0xa800）。没有编译错误、没有链接诊断——完全静默，只有数值验证能听见。

### 4.5.4 修复：逐内核改名与合并算法

`merge_ptx()`（`tools/gdn_smem_fix.py:715-752`）的三步：模块 0 的文件头保留；后续模块的 `.shared` 声明收集进来并全部改名，`.extern` 声明取并集去重；所有 entry 体顺序拼接。改名规则 `__gdn_smem<N>` → `__gdn_smem_<模块标签>_<N>`，**声明与引用一体替换**：

```python
    def rename_syms(text, tag):
        return re.sub(r"__gdn_smem(\d+)",
                      lambda m: "__gdn_smem_%s_%s" % (tag, m.group(1)),
                      text)
```

真实产物（`evidence/G5/gen_fwd/bench_B1T2048H16_mono/kernel_smemfix.ptx:23-28`）：

```ptx
	.shared .align 128 .b8 __gdn_smem_gdn_fwd_f1_kernel_0[256];
	.shared .align 128 .b8 __gdn_smem_gdn_fwd_f1_kernel_1[256];
	.shared .align 128 .b8 __gdn_smem_gdn_fwd_f1_kernel_2[256];
	.shared .align 128 .b8 __gdn_smem_gdn_fwd_f1_kernel_3[6144];
	.shared .align 128 .b8 __gdn_smem_gdn_fwd_f1_kernel_4[9216];
	.shared .align 128 .b8 __gdn_smem_gdn_fwd_f1_kernel_5[9504];
	...
```

`main_multi()`（`gdn_smem_fix.py:755-810`）的完整流水：提取全部模块 → 逐模块 `fix_ptx`（含 `"modules"` 映射分发的别名规范——r15 起正向链也具备竞技场能力）→ `fix_params` → WST → WSP（在 WST 之后，只碰熔剩的单）→ `merge_ptx` → `apply_maxnreg_k` → 第二遍 ptxas。逐模块打印统计行（`cvta= shared= params= wst= wsp=`）——W32P 一节引用的"f2 wst 计数 0→2"即出自此行。

### 4.5.5 效果数字

- mono 三元组成为正向 staged/fused 全部形态的载体；DtoD 桥接拷贝（约 0.1ms @B1T2048）消失（fwd G5-D）。
- 一个 cubin 多 entry、公共全局缓冲区：`gdn_full_rt.py` 的 `FullKernel` 按 `full_spec.json` 的 kernels 表逐 entry 取 `cuModuleGetFunction`，全局参数表逐个 `cuModuleGetGlobal`（`gdn_full_rt.py:125-147`）。
- 逐 entry `.maxnreg`（§4.3）与逐模块 WST/WSP 作用域都在合并后、编译前的单一 PTX 上施加——合并是它们的宿主。

## 4.6 常数参数消除（Fix 2）：标量、聚合常数字节映像与签名收尾

### 4.6.1 动机与值源

缺口二（§4.0）的机理复述：tiling 与 hatch 下译把 launch 区域常数提升回宿主，outlining 把它们变成内核参数——最多约 140 条 `.b64`。修复的值源是 `run_l2.sh` 从 G3-A 起保留的 `asmobj.mlir`：宿主侧 `gpu.launch_func ... args(%c : i64, ...)` 的操作数表仍在，**位置顺序 == PTX 参数顺序**，于是每个常数参数的值可回收。非常数参数（tma 配置的宿主侧 tensormap 描述符指针）保留为真实参数。

### 4.6.2 标量替换

`parse_launch_args()`（`tools/gdn_smem_fix.py:280-312`）把 launch 操作数 SSA 名映射到 `llvm.mlir.constant` 值；`_const_bits()`（`gdn_smem_fix.py:263-277`）把值字面量转 PTX 立即数（整型直书；i1 → 0/1；f32/f64 → `0f%08X`/`0d%016X` 十六进制位型；bf16 取 f32 编码高 16 位）。`fix_params()`（`gdn_smem_fix.py:333-378`）把每条

```ptx
	ld.param::entry.b64 	%rd269, [gdn_o_kernel_kernel_param_129];
```

替换为

```ptx
	mov.b64 	%rd269, 63;
```

（真实前后对照：`evidence/G4/gen_verify/o_f32_nomma_p0_t64x64x32_w4_K64V64/kernel.ptx:159` 与 `kernel_smemfix.ptx:31`），并删除已无用的 `.param` 声明。

### 4.6.3 聚合常数的字节替换

G5-G 轮次发现该提升通道会连**向量常数**一起提升：payload 一旦携带 hatch op，`gdn-opt` 的 rewriter 把 launch 区域常数提升为宿主实参，**包括 bf16-mma 累加器零**——

```mlir
%zero = llvm.mlir.constant(dense<0.0> : vector<2x2xf32>) : !llvm.array<2 x vector<2xf32>>
```

这类参数在 PTX 里以多条**带偏移**的 `ld.param` 装载（真实样本，ws 配置 `kernel_smemfix.ptx:237-238`——该配置保留的 3 条封送常数之一，16B 向量零）：

```ptx
	ld.param::entry.b64 	%rd3141, [gdn_kkt_kernel_kernel_param_136+8];
	ld.param::entry.b64 	%rd3140, [gdn_kkt_kernel_kernel_param_136];
```

`LLVM_AGG_CONST_RE`（`gdn_smem_fix.py:223-226`）识别 `dense<...> : vector<d1 x d2 x T>` 形态，`_agg_bytes()`（`gdn_smem_fix.py:231-260`）把稠密字面量（标量广播或嵌套列表）按元素类型打包成**小端字节映像**；`fix_params` 的 `sub_ld` 对带偏移的装载按 `off:off+width` 切片字节映像、经 `_bytes_imm` 转成立即数 `mov`。不做此替换的后果是零参数启动直接 `CUDA_ERROR_INVALID_VALUE`（bwd 报告 G5-G 工具链注记原文）。

### 4.6.4 签名收尾

参数删除后的两处收尾（`fix_params` 尾部，`gdn_smem_fix.py:372-377`）：最后一个残留声明的尾逗号摘除；参数表被清空时把 `name(\n\n)` 折叠为 `name()`——**本 ptxas 拒绝括号内的空行**。这是"改写器必须活在真实汇编器的文法约束下"的又一处微观注脚。

### 4.6.5 效果数字

- 20 个 G3/G4 配置中 **18 个成为真正的零参数内核**（noargs 契约：输入经公共 `memref.global` 缓冲上载/读回，内核裸启动）；ws 配置保留 3 条封送常数（2× i1 + 16B 向量零）；tma 配置保留 3 条 tensormap 指针（`cuTensorMapEncodeTiled` 宿主编码后以设备指针传入）。
- 单内核 `main()` 的统计行（`gdn_smem_fix.py:848-852`）逐构建对账：`malloc_callseqs / cvta_patched / shared_decls / static_smem_total / const_params_substituted / dynamic_params_kept`——其中 `static_smem_total` 正是 §4.2 各池数字的直接来源。

## 4.7 `cache_key` 哈希键控：陈旧缓存坑的两次咬史与根治

### 4.7.1 两次咬史

后处理层与运行时（`tools/gdn_full_rt.py` 的 `build()`）共同构成一条**带缓存的构建链**：目标目录已有 `kernel_smemfix.cubin` 且有 `full_spec.json` 时直接复用。这个短路在两处真实地咬人：

1. **第一次（正向 r3/r4）**：`evidence/G5/gen_fwd/B1T8192_mono` 槽位里躺着一个**陈旧的 w8/bv64 cubin**，被第 4 轮的 bench/裁决运行当作新构型取用——构型错配直到下一轮才被发现（fwd 报告 r5 节："a STALE w8/bv64 cubin served to round-4 bench/adjudication runs through the build-cache hole"；fused 槽位同样持有陈旧 r2 构建 `b2549100`）。所有涉事数字作废重测。
2. **第二次（正向 r19）**：F1WY 变体的 T2048 "verify" 实际是一次**陈旧 gen_fwd 缓存命中**——它把一个真 UB 掩盖成了"通过"：宽度 16 的 `gpu.shuffle` 下译为 `shfl.sync ... 0xFFFF` 成员掩码，lane 16..31 执行它是未定义行为；T2048 的缓存假阳性让该缺陷漏网，是 T8192 fp64 门（o 3.6e-1 FAIL）抓住的。教训被写进复现规约：变体验证永远用新鲜 `--gen-root`/scratch 目录。

两次事故的共同根因：缓存键为"目录名"，而**目录名不编码旋钮集与生成器/改写器源状态**。

### 4.7.2 根治：哈希键控

`_cache_key()`（`tools/gdn_full_rt.py:28-46`）：

```python
def _cache_key():
    """R20-fwd: hash of everything that changes the emitted payload --
    every GDN_*/G5* env knob in scope + the generator/smem_fix sources.
    The stale-cache hole (a cached kernel_smemfix.cubin accepted under a
    DIFFERENT knob set or an edited generator) has bitten twice (r3,
    r19 -- the F1WY T2048 'verify' that was a stale gen_fwd hit)."""
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

键 = sha256(**作用域内全部 `GDN_*`/`G5*` 环境旋钮的有序键值** + **`gdn_full_gen.py` 与 `gdn_smem_fix.py` 两份源文件的字节**)。`build()` 的缓存协议（`gdn_full_rt.py:48-110`）：构建成功后把 `gdn_cache_key` 写入 `full_spec.json`；后续构建若键不匹配，**重定向到 `<out_dir>_k<hash8>` 变体槽**而不是返回（或覆写）缓存目录；无戳的遗留目录按原样信任（gen_fwd 记录船位保持不动，不强制重建）。三态测试：同键命中、异键重定向（新鲜 `_k` 目录）、遗留信任。

### 4.7.3 效果与协议地位

- 该洞自 R20 关闭后未再发生；生成器每次编辑后 gen_fwd 槽位的戳**按设计陈旧**（r23 编辑后键值 `780ab7d2`，r24 的编辑再次移动它）——`rt.build` 重定向、永不陈旧服务（fwd r24 §4 原文）。
- 键控同时覆盖了"旋钮集差异"与"源码编辑"两类漂移，这正是两次事故的两类肇因（第一次是构型/旋钮错配，第二次掩蔽 UB 的验证也属旋钮错配类）。
- 与第 9 章的测量规约互为表里：cubin md5 复核（重建 vs 记录）+ cache_key 键控共同保证"测的是想测的字节"。

## 4.8 小结：变换一览

| # | 变换 | 开关/入口 | 作用层 | 一句话效果 | 详节 |
|---|---|---|---|---|---|
| 1 | 静态 smem 修复 | 默认（单/多模块） | callseq + cvta + 声明 | smem tile 内核通过率 0→全量；`::cluster` 同规则 | §4.1 |
| 2 | 128B 对齐 | 默认 | `.shared` 声明 | UTMALDG 目的地址合法 | §4.1.5 |
| 3 | 竞技场别名 | `smem_alias.json`（`"modules"` 分发） | malloc 删除 + cvta→基+偏移 | dqkwg 100,864→47,616B、caryA 82,176→28,592B、wyb 100,096→45,056B ⇒ 2 CTA/SM | §4.2 |
| 4 | `.maxnreg` 注入 | `GDN_MAXREG_K` / `GDN_MAXNREG_ENTRY` | entry 体指令 | ptxas-blackwell 旗标盲区的唯一有效帽；dqkwg 1.16–1.26× | §4.3 |
| 5 | WST 四元组融合 | `GDN_WST`（`GDN_WST_SCOPE`） | b16 四元组→b32 对 | f1 −11~13μs，LSU 指令与迭代减半，字节精确 | §4.4 |
| 6 | W32P 行重映射 | `GDN_W32P`（生成器侧） | lane→row 映射 | 80 mod 32=16 互补半区零冲突；f2 −18~26μs @T8192；R25 三形状 formals 上移 | §4.4.5 |
| 7 | WSP shuffle 配对 | `GDN_WSP` | 跨 lane b32 打包 | f2 frontier 备用路线，保留在工具 | §4.4.6 |
| 8 | mono 合并 | ≥2 个 `gpu.binary` 自动 | 多 PTX → 单 cubin | DtoD 桥接（~0.1ms）消失；重名 `.shared` 逐内核改名防静默统一 | §4.5 |
| 9 | 常数参数消除 | 默认 | `ld.param`→`mov` | 18/20 零参数内核；聚合常数字节映像替换 | §4.6 |
| 10 | cache_key 键控 | `gdn_full_rt.build()` | 构建缓存 | 陈旧缓存两次咬史（r3/r19）根治：异键重定向 | §4.7 |

这十个变换共享同一条工程伦理：**每一步都是可审计的字节级重写**——要么位级相同（WST/W32P/别名/常数消除的全部落点都以此立据），要么回退到正确而不省的形态（别名的防御性回退），且每一步的诊断打印（`[smem_fix]` 前缀的 NOTE/WARN/统计行）保证失败响亮。后处理层因此不是对 MLIR 的绕过，而是把 stock 流水线留在 PTX 门口的四笔账——两笔正确性、一笔驻留、一笔发射效率——逐一付清的最短路径。
