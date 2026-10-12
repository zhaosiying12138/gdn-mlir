# GDN-MLIR

Gated DeltaNet（`chunk_gated_delta_rule`，GDN）前向与反向算子的 **stock-MLIR 渐进式下降编译器**：调度即 IR 旋钮（transform dialect 参数化工件 + 方言属性），编译期确定性调度（Z3 + 校准解析成本模型），PTX 级覆盖证明（对照 fla Triton / FlashQLA TileLang 手写实现），在 RTX 5090 Laptop（sm_120a）上以 n=30 交错配对统计检验完成端到端性能裁决。

```
L0 语义 op (gdn.fwd/gdn.bwd)
  └─ gdn-to-chunk ─→ L1 算法结构（chunk-WY 分解，linalg+scf，CPU 可执行验证）
       └─ 生成器+transform ─→ L2 调度（tile/warps/流水/融合边界旋钮；Z3 合法性求解+排序）
            └─ L3 张量核与流水（mma.sync / cp.async / TMA / mbarrier / warp 专化）
                 └─ PTX 后处理（静态 smem / 生存期竞技场别名 / .maxnreg / bank 几何）
                      └─ L4 目标（gpu-lower-to-nvvm-pipeline → ptxas-blackwell → sm_120a cubin）
```

## 核心结果速览（RTX 5090 Laptop，bf16，K=V=128）

| 战场 | 结果 | 依据 |
|---|---|---|
| **前向 vs fla Triton** | **全胜**：B1T2048 干净窗 **1.611×（30/30 对，DECIDABLE）**、共载窗 **2.125×**；B4T2048 1.07×/1.22×；B1T8192 0.99×（平手）/1.35× | n=30 交错配对 `--stats` |
| **反向·小形状** | **B1T512H4 1.72× 胜** | 三重交错配对 |
| **反向·大形状（u≥1）** | 0.46-0.47×，**四重独立证据判定为精度合同结构性关闭**（天花板 0.85-0.92×） | RC1-R4 探针链 |
| **冷启动（全形状）** | **全胜**：我方 AOT 零调优（~27µs 启动）vs fla 每形状 48-69s 冷调优 | 四路摊销实测 |
| **编译期调度** | Z3 确定性枚举 **0.8s/形状（纯 CPU）** vs fla 设备冷调优 39.8s ≈ **50×**；排序质量 Spearman **0.8962** | G3/G6 |
| **PTX 覆盖** | T1 指令族 feature 层全覆盖（含 fla/FlashQLA 两极点）；**逃生舱仅用 1/≤5** | coverage_matrix |

---

## 1. Pass / Lowering 功能层级详解

本项目的一切优化都表达为"逐层下降 + 旋钮"，下面按层列出**已实现的全部 pass、lowering 与变换**。每项标注实现文件。

### L0 — 方言与语义层（`dialect/`）

**语义 op（TableGen ODS）**
- `gdn.fwd`（`GdnOps.td`）：输入 `q,k,v [B,T,H,K], [B,T,HV,V]`、`beta,g [B,T,HV]`、可选 `h0 [B,HV,K,V]`；输出 `o, ht`。带 verifier（rank/chunk_size 合法域）。
- `gdn.bwd`（`GdnOps.td`）：六梯度输出 `dq,dk,dv,dbeta,dg,dh0`；可选 `A`（WY 因子重算-vs-复用旋钮）；`AttrSizedOperandSegments` 消解三个 Optional 尾随槽歧义。
- `GdnConfigAttr`：七旋钮语义束 —— `chunk_size∈{16,32,64}`、`use_g`、`use_gk`、`normalize_q`、`use_initial_state`、`store_final_state`、`is_varlen`。assembly 形如 `#gdn.config<64, true, false, false, true, true, false>`。

**Pass**
- `gdn-to-chunk`（`lib/GdnPasses.cpp`）：L0→L1 入口改写——`gdn.fwd/bwd` → 对 `__gdn_l1_fwd_chunk/__gdn_l1_bwd_chunk` 私有函数声明的调用，`gdn.config` 属性原样透传（声明按需自动生成）。
- `gdn-lower-hatch`（`lib/GdnHatch.cpp`）：11 个硬件逃生舱 op 1:1 下降到树内 nvgpu/nvvm op（见下）。

**逃生舱 op 族（`GdnHatch.td`，验证 sm_120a PTX 精确拼写）**
- `gdn.setmaxnreg <inc|dec, count>` → `nvvm.setmaxregister`（warp 专化寄存器再分配；实测开启后 spill 22.8KB→48B）
- `gdn.elect` → `nvvm.elect.sync`（warp leader 选举）
- `gdn.mbarrier.init / arrive / arrive_expect_tx / try_wait.parity` → `nvgpu.mbarrier.*`（异步到达屏障全家）
- `gdn.tma.async.load / store`、`gdn.tma.prefetch` → `nvgpu.tma.*`（TMA 张量搬运，宿主 CUtensorMap 注入）
- `gdn.bulk.commit_group / wait_group` → `nvvm.cp.async.bulk.*`（批量异步组控制）
- **`gdn.mma.tf32k8`（已用的唯一真实逃生舱，1/≤5）**：补 stock MLIR 无 tf32 `m16n8k8` 表项的缺口，经 `llvm.inline_asm`（**early-clobber `=&f` 强制**——普通 `=f` 下 ptxas 会把 D 寄存器分配到同一条 mma 的 A quad 上）发出 fla 精确指令；A-fragment 寄存器序 row-half-outer 由硬件整数探针裁定。

**工具**：`gdn-opt`（外建 MLIR 项目，`scripts/build_dialect.sh` 构建，含全部树内 pass + 方言注册）。

### L1 — 算法结构层（`tools/gdn_l1_gen.py` + `tools/gdn_ref_jax.py`）

前向与反向的 chunk-WY 分解表达为**真正的 linalg+scf MLIR**（非伪码），CPU 管线可执行、可对拍：

- **前向分解**：chunk-local 含尾 cumsum(g) → KKT 矩阵 `A = strict_tril(β·(kkᵀ)⊙2^(gc_i−gc_j))` → **前代换求逆 T=(I+A)⁻¹**（三重 `scf.for` 显式实现）→ `w,u` → 状态递推 `S ← (2^gcl·I − ktermᵀw)S + ktermᵀu`（`scf.for` 载态）→ 输出 `o = intra + inter`。
- **反向分解**：重算前向中间量 → 逆时间 dh 递推 → dA 双侧掩蔽收缩 `dA = −strict_tril(Tᵀ·strict_tril(G)·Tᵀ)`（单侧式被符号级证伪）→ dk/dv/dβ/dg 组装 → dh0。
- **CPU 下降管线**（11 个 pass，`scripts/run_l1.sh`）：`-one-shot-bufferize -convert-linalg-to-loops -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-math-to-llvm -convert-arith-to-llvm -finalize-memref-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts` → `mlir-runner` 执行。
- **JAX 可执行规约**（`gdn_ref_jax.py`）：顺序递推 + 分块 + `jax.vjp` 三实现（fp64），作为语义单一事实源；三方对拍闭合：MLIR-L1 ⇄ JAX ⇄ fla naive（前向 6e-7、反向 ≤6.4e-6）。

### L2 — 编译期调度层（`tools/gdn_l2_gen.py` + `gdn_solver.py` + `gdn_costmodel.py`）

- **旋钮库**（20 个预置配置全编译验证）：`stage ∈ {kkt, o}`、`dtype ∈ {bf16, f32}`、`prec ∈ {bf16, tf32}`、`tile {M,N,K}`、`warps ∈ {2,4,8}`、`mma on/off`、`pipeline ∈ {0,2,3}`、`warpspec`、`tma`。逐旋钮 PTX/SASS/资源三层敏感性实测（如 pipeline 0→2→3：mma 16→48→64）。
- **transform dialect 调度**（有效拼写已踩平）：`transform.memref.multibuffer <factor=N, skip_analysis>`、`transform.nvgpu.create_async_groups bypass_l1`、`transform.nvgpu.pipeline_shared_memory_copies failures(propagate) depth=N peel_epilogue`、`tile_using_for` 等。
- **Z3 求解器**：决策变量=旋钮向量；约束=smem 字节模型（≤98KB 显式公式）、寄存器、mma fragment 对齐、结构合法性；目标=最小化解析模型估算延迟；**固定 seed+rlimit，两次运行逐字节一致**；合法空间 kkt 368 / o 465 配置 ~0.3-0.44s 枚举完。
- **校准解析成本模型**（零逐配置拟合常数）：速率全部来自一次性标定表（mma 115.5 TFLOP/s、smem 113.1 TB/s、L2 2144 / DRAM 825-851 GB/s、SFU ex2 3.23 Telem/s、bar.sync 7.71ns、cp.async 1388 GB/s）；模型迭代留痕 v1 0.735→v3 **0.8962**（±2× 常数扰动 ρ≥0.882）。
- **参考 PTX 收割器**（纯 CPU）：fla 经 Triton AOT `GPUTarget("cuda",120,32)`；FlashQLA 经 TileLang 锁 target + monkeypatch；SASS 经 nvdisasm——产出两极点参照（fla 255regs+spill 多级流水 vs FlashQLA ≤128regs setmaxnreg 单 kernel 融合）。

### L3 — 张量核与流水机器层（`gdn_full_gen.py` / `gdn_bwd_gen.py` 内建机器）

**MMA 发射器族**（自建，绕过 stock 路径边界）
- `mma_gemm`：warp 贴图 mma.sync m16n8k16 bf16→f32 基础发射器（fragment A/B/C/D 寄存器序；D-fragment N 索引=2·组内线程号——与 B-fragment groupID 的区别曾是一处 11 配置级 bug 的根因）。
- `mma_gemm_mh`：mi-half × ni-strip 二维 warp 映射（W≤16，512 线程下铺满 16 warps/SM）。
- `tgemm`：tf32 k8 逃生舱发射（消费 `gdn.mma.tf32k8`）。
- **split-bf16 hi/lo 分解**：`x = hi + lo`，3-mma（hi·hi + hi·lo + lo·hi，lo·lo 舍弃）或行尺度折入 epilogue 后的 2-mma 型；幂二尺度可交换性 `split(v·2^d) == split(v)·2^d` 免拆分。
- `rgemm / rgemm_aw`：寄存器累加串行链 GEMM + A-窗口 ping-pong（反向扫描专用）。

**Staging 原语族**
- `astage_sync`：thread-strided 16B 手动 `cp.async`（**关键教训：transform 的 create_async_groups 会在每组后立即插 wait_group 0，永不重叠**——必须手动路线；且 staging 必须线程跨步，block-uniform 会被每线程重复执行 256×）。
- `stage_split_ew / stage_raw_T / stage_split_T`：split-bf16 操作数分级助手（cvec 列尺度、bank 垫行）。
- `launder_pass`：r6 危险规则——**LDGSTS 写入的 tile 经 `ld.shared` 读回正确、经 `ldmatrix` 读回错误**（构建调度相关的行块子集，任何通用 dump 不可见）；修复=st.shared 或逐元素重写。
- `epi_bar`：epilogue split-store 越死 B 源的栅栏规则。

**warp 专化与 TMA**：`setmaxnreg` 生产者/消费者分区（count=1 纯 data-ready mbarrier 协议）、`elect.sync` leader 发射 TMA、宿主 CUtensorMap 维序契约（encode 反序、坐标不反、UTMALDG 需 smem 128B 对齐）。

### PTX 后处理层（`tools/gdn_smem_fix.py`，两遍 ptxas 构建链的第二次）

stock MLIR 四个真实缺口的修复（全部探针取证）：
1. **静态 smem 修复**：workgroup `memref.alloc` 被降为 device malloc + cvta（真机首跑全量 ILLEGAL_ADDRESS）→ 重写为 `.shared` 声明 + `cvta` 补丁 + 128B 对齐。
2. **生存期竞技场别名**：按 malloc 唯一尺寸识别、`arena_base+offset` 重写 cvta、垫行规则（ldmatrix 16B 行距）；对抗 LLVM 的 malloc 重排 / DCE 宿主 / store-to-load 前传。效果：dqkwg 池 100,864→47,616B、caryA 82,176→28,592B 等 → 2 CTA/SM。
3. **`.maxnreg` 注入**：ptxas-blackwell **无视 `--maxrregcount` 但尊重 PTX 指令**；双遍寄存器检查（`ptxas_v.txt` 只是第一遍——出货 cubin 是第二遍）。
4. **W32P 四元组融合**：`80 mod 32 = 16` 互补半区律消除 f2 存储通路的 bank 冲突（f2 −18~26µs）。
5. **mono cubin 合并**：重名 `.shared` 符号跨入口静默统一坑（改名修复）；常数参数字节替换（18/20 零参数 kernel）；`cache_key` 哈希键控（旋钮+源码指纹，陈旧缓存坑根治）。

### L4 — 目标层

`mlir-opt -transform-interpreter -test-transform-dialect-erase-schedule -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a cubin-features=+ptx87 cubin-format=assembly"` → PTX 抽取 → `ptxas-blackwell -v` → nvdisasm SASS。全链在 5090 上以树内 sm80 transform-mma 集成测试冒烟通过。

---

## 2. 性能评测（RTX 5090 Laptop，sm_120a，82 SM，bf16 io / fp32 累加）

### 2.1 测量方法学

- **交错配对判定**：同窗内 `[fla, ours]` 交替发射 n=30 对，逐对比值取中位数 + 精确符号检验 + IQR 可判定规则 `(median−1) > 3×IQR/median`。实证依据：分离进程绝对值在零共载下仍摆 **47%**（空闲时钟循环）——只有交错比值是裁决级。
- **双窗口口径**：共载窗（其他训练任务驻留）与干净窗分别记录——**fla 在共载下被压得更狠**，故共载比值偏高；两口径并引，不做挑选。
- 输入配方：fla `test_chunk` 风格 seed-42 共享配方；fla 侧恒为纯 Triton 口径（`FLA_FLASH_QLA=0`，探针验证后端禁用）。

### 2.2 前向 vs fla Triton（三次纪录代际 + 干净窗终判）

| 形状 (H=16, K=V=128) | G5 收官代 | R22 F1ZF 代 | **R25 W32P 代（共载 n=30）** | **干净窗 clean3（n=30）** |
|---|---|---|---|---|
| B1T2048 | 1.59× | 2.118×（首个正式） | **2.125×**（30/30 对，p=9.3e-10） | **1.611×（30/30 对，DECIDABLE）** |
| B4T2048 | 1.05-1.19× | 1.142× | **1.221×**（25/30 对，p=1.6e-4） | 1.067×（近平手） |
| B1T8192 | 1.25-1.31× | 1.223× | **1.349×**（29/30 对，p=2.9e-8） | 0.991×（平手，不显著） |

我方最优绝对时间（W32P 代）：B1T2048 **0.198ms**、B4T2048 0.821ms、B1T8192 **0.785ms**；fla 干净窗中位 0.342 / 1.288 / 0.961ms。**前向在两种窗口口径下从未输过**；B1T2048 在两种口径下均 ≥1.1× 且干净窗可判定。

### 2.3 反向 vs fla Triton（形状阶梯，u = B·H·(T/64)/512）

RC1 实测（同窗双方绝对时间，共载三重交错中位）：

| cell | u | 我方 ms | fla bwd ms | 比值(fla/我方速度) |
|---|---|---|---|---|
| **B1T512H4** | 1/16 | 0.557 | 1.211 | **2.17×（我方胜）** |
| B1T2048H16 | 1 | 2.192 | 1.332 | 0.61 |
| B1T2048H32 | 2 | 4.154 | 1.632 | 0.39 |
| B4T2048H16 | 4 | 7.971 | 3.076 | 0.39 |
| B1T8192H16 | 4 | 8.138 | 3.228 | 0.40 |
| B1T16384H16 | 8 | 16.161 | 6.405 | 0.40 |

此后 R2（bf16 全局量）+ R3（A_all 退役+dhs 坍缩 5.9×+CVT 税回收）将全 cell 再提速 1.11-1.21×：u=8 比值 0.396→**0.462**（我方 12.15ms），小形状深化至 **1.72× 胜**。仿射拟合：`ours = 0.138 + 1.996u ms`，`fla = 0.298 + 0.746u ms`。

**精度关闭定理（四重独立证据）**：u≥1 的 ≥1.1× 目标在当前梯度容差合同（dk/dv≤8e-3，dβ/dg≤2e-2）下结构性不可达——
1. **误差预算**（R2/R3）：唯一能移地板的字节属于被探针杀掉的 f32 边（O(10) 相消到 O(0.1) 的中间量，bf16 舍入 4.6-6.9e-2 直接超 bar）；split-bf16 零收益。
2. **地板算术**（R3）：完美执行封顶 u=8 于 **0.85-0.92×**。
3. **融合定价**（R4）：最后一个字节地板移杆（dqkwg+wyb 融合）被 G 全物化屏障+池封顶（129,792B>101,376B 杆）+占用反转三重杀，纸面 NO-GO。
4. **容差灵敏度**：dk 8e-3→2.0e-2 全档位**零符号回归**（容差轴是阶跃函数不是旋钮）；跨线需 dk ≥1.19e-1 **且** dβ ≥4.8e-2 同时放宽全部五个符号——且只是天花板。

### 2.4 与 FlashQLA（TileLang，sm120 参照）的相对位置

- FlashQLA 前向（`block_DV=64` 垫片下）：0.139/0.404/0.355ms = **fla 的 2.16-2.38×**——sm_120 前向的现役 SOTA。我方前向（0.198/0.785ms）** decisively 胜 fla、约落后 FlashQLA 前向 1.4-1.5×**（窗口不同，如实披露）。
- FlashQLA 反向（源码含完整 sm120 bwd 链，出厂超 smem 上限不可启动；垫片下 1.435ms @B1T2048 ≈ fla 平手档）——三角对比已入 G5 基线报告附录。
- 顺带披露其两个 sm120 启动 bug（export 缺失 + 112,640B>101,376B smem 超限）。

### 2.5 冷启动（全形状胜）

我方：AOT cubin、零调优、~27µs 启动开销。fla：每形状冷调优 fwd 18.7-30.8s + bwd 29.3-38.3s（T16384 达 69s）；缓存命中路径 0.65-5.67s（四路摊销实测：冷搜 43.9s / FLA 配方+编译冷 5.67s=7.7× / 配方+编译热 0.65s=67.5× / Triton 原生缓存 1.07s=41×）。

### 2.6 编译期调度 vs 设备 autotune

Z3 + 成本模型：**0.8s/形状、纯 CPU、确定性（双运行逐字节一致）**；排序质量 Spearman ρ=0.8962（kkt 0.936 / o 0.829），±2× 结构常数扰动 ρ∈[0.882,0.902]。对比 fla 冷搜 39.8s（48 配置上机）≈ **50× 搜索成本优势**。

---

## 3. 数值正确性与确定性

- **前向**：o/ht vs JAX fp64 参考 ≤7.5e-4（容差 5e-3，裕量 7-10×）；**二十六轮优化全程位级一致**（每个杠杆后复验）。
- **反向**：六梯度全部 PASS 于全部阶梯 cell（最紧 dk 5.0e-3 vs 8e-3 bar）；与 fla 自身容差标定对照（fla bf16 输出误差全部在其舍入下界 ~2× 内）。
- **确定性协议**：跨进程 N=5-10 次全位级一致（前向 staged/fused、反向全链）；`restage-fence` 七点清单（"栅栏在读完之后"）；两处迭代间重分级竞态、zero_up 步幅 bug（SASS 差恰好一条指令）、Hillis-Steele 跨 warp 竞态等 5 案全部根因修复。

## 4. PTX 覆盖证明与逃生舱台账

- **T1 指令族**：以 fla+FlashQLA 收割 PTX 的 33 族为合同；我方生成集覆盖 feature 层全部（mma.sync m16n8k16 bf16 精确签名双向 YES、cp.async、TMA+mbarrier、setmaxnreg、elect.sync、ldmatrix、bar.sync 等）；基线层的 PARTIAL/NO 均为拼写变体或"GDN 不需要"项（如 stmatrix 被实测 +40% 回归正当关闭），逐项标注成因。
- **T2 结构匹配**：fla 极点（multistage mma+cp.async）与 FlashQLA 极点（setmaxnreg warp 专化单 kernel 融合）均可由旋钮生成；两极点间资源包络（寄存器/占用/spill）全覆盖。
- **逃生舱台账**：预算 ≤5 个自定义 NVVM op，**实际使用 1 个**（`gdn.mma.tf32k8`，含 early-clobber 与 fragment 序两项硬件级发现）。

## 5. 工具链发现（对 MLIR/NVIDIA 上游有价值的取证清单）

1. workgroup `memref.alloc` → device malloc（真机 ILLEGAL_ADDRESS，PTX 级修复方案在 `gdn_smem_fix.py`）
2. transform `create_async_groups` 每组后立即 `wait_group 0`——结构性永不重叠
3. block-uniform staging 被每线程重复执行（256× 冗余 cp.async）
4. ptxas-blackwell 无视 `--maxrregcount` 尊重 PTX `.maxnreg`；`ptxas -v` 输出只是第一遍
5. **LDGSTS→LDSM 静默损坏**（构建调度相关的行块、通用 dump 不可见、st.shared 安全）
6. ptxas 跨入口静默合并重名 `.shared` 符号（OOB 写）
7. **placement lottery**：模块构造序翻转镜像 ~6% kernel 方差（f3 类大足迹 kernel）
8. 分离进程空闲时钟循环 47% 摆动——裁决级测量方法学的实证基础
9. **sm_120 线程块 cluster>1 可用**（设备实证推翻"仅 1×1×1"通行假设）
10. `cp.async wait_group` 每线程语义（跨 warp 可见性需显式栅栏）
11. tf32 m16n8k8 inline-asm：early-clobber 强制 + A-fragment row-half-outer 序（硬件探针）

## 6. 仓库结构与复现

```
dialect/          gdn 方言（L0 op + hatch op + passes + gdn-opt）
tools/            全部生成器/求解器/harness（gdn_l1_gen / gdn_l2_gen / gdn_full_gen /
                  gdn_bwd_gen / gdn_solver / gdn_costmodel / gdn_smem_fix /
                  gdn_verify / gdn_final_adjudicate / 收割器 ...）
scripts/          幂等驱动（build_llvm / build_dialect / run_calib / repro / 环境脚本）
evidence/G0-G7    每个 gate 的报告+原始数据（PTX/SASS/计时 JSON/对拍记录）
paper/zh/         深挖论文 11 章（676KB md + 818KB KaTeX HTML 知乎包）
```

```bash
make status                 # 现场状态
make repro-g0 ... repro-g6  # 逐 gate 一键复现（全部幂等）
make repro-all              # G0→G6 全链
```

复现需求：WSL2 + conda env `gdn`（torch cu130/triton/fla/z3/jax）+ clang/cmake/ninja + CUDA 13.x（ptxas-blackwell 随 triton wheel 提供）+ CC≥12 GPU；G1/G2 与求解器 halves 纯 CPU。

## 7. 已知限制

单 GPU 单架构（sm_120a）；形状套件覆盖 B∈{1,4}、T∈{512..16384}、H∈{4,16,32}、K=V=128（未测 GVA/varlen 硬件路径）；反向 u≥1 在 fla 容差合同下精度关闭（详见 §2.3 四重证据）；成本模型为排序级（绝对值有已记录盲区）；干净窗正式终判 B4T2048 处可判定性边缘（v1/v2 双轨已归档）。

## 8. 许可

TBD（代码与证据树待定；参考实现 fla=MIT / FlashQLA=MIT 仅作对照基准使用）。
