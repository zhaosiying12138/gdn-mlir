## 十五、调度空间的合法域：Z3 变量、耦合与资源约束

一个旋钮向量只有在生成器能够发射、IR 能够满足结构前提、资源能够装入目标硬件时才具有可执行意义。`KnobSpace` 把这些条件写成 SMT（模理论可满足性）断言：Z3 负责判断变量指派是否同时满足全部条件，生成器与 ptxas 随后在各自层级复核。断言集定义了求解器眼中的合法域，它是否与真实生成器一致，需要持续对账。

### 1. 十维变量、域与声明序

```python
# -- decision variables (fixed declaration order => determinism) --
dtype = z3.Int("dtype")            # 0 = bf16, 1 = f32
prec = z3.Int("prec")              # 0 = bf16, 1 = tf32
bt = z3.Int("tile_M")
bv = z3.Int("tile_N")
bk = z3.Int("tile_K")
warps = z3.Int("warps")
pipe = z3.Int("pipeline")
mma = z3.Bool("mma")
ws = z3.Bool("warpspec")
tma = z3.Bool("tma")
self.v = dict(dtype=dtype, prec=prec, bt=bt, bv=bv, bk=bk,
              warps=warps, pipe=pipe, mma=mma, ws=ws, tma=tma)
```
[tools/gdn_solver.py:98-110]

`z3.Int` 创建整数符号，`z3.Bool` 创建布尔符号；此时尚未有任何赋值。`dtype` 和 `prec` 用整数编码，因为元素宽度等算术需要引用它们；三个纯开关用布尔量。`self.v` 的插入顺序固定为 dtype、prec、M、N、K、warps、pipeline、mma、warpspec、tma，模型读取、缺失值补全和输出翻译都沿这份登记表进行。

基域为 tile {32, 64, 128}、warps {2, 4, 8}、pipeline {0, 2, 3} [tools/gdn_solver.py:70-74]。M 还要整除 64，N 要整除 kkt 的 64 或 o 阶段的 V，K tile 要整除全局 K；过滤后的列表用 `z3.Or` 写成成员约束 [tools/gdn_solver.py:112-121]。布尔变量的域由类型给出，`prec` 则由下述耦合约束固定到 0 或 1。生成器还接受 M=16 和更小的 8 倍数 tile，求解器主动采用更窄的 G3 研究域，两种“可生成”和“纳入搜索”的口径不能混用。

这个窄域决定完备性命题的边界：对基域穷尽可能覆盖研究合同中的每一点，却没有覆盖生成器语法允许的全部配置。若以后加入 M=16 或新的 pipeline 值，需要同步扩展 Python 域元组、Z3 成员断言和补全候选，再重新记录空间数量。只改其中一处会造成“求解器看不到”或“补全能产生而主约束拒绝”的分裂。

### 2. 耦合、对齐与流水前提

`z3.Implies(a, b)` 表示前件成立时后件必须成立；前件为假时，该断言不再限制其他变量。`z3.If(c, x, y)` 则构造条件表达式，根据符号条件在两个值之间选择，常用于资源公式。它与 Python 的即时分支不同，选择会保留在待求解公式里。

```python
s.add(z3.Implies(dtype == 0, prec == 0))
s.add(z3.Implies(dtype == 1, prec == 1))
s.add(z3.Implies(z3.Not(mma), prec == dtype))
s.add(z3.Implies(mma, bt % 16 == 0))
s.add(z3.Implies(z3.And(mma, dtype == 0), bk % 16 == 0))
s.add(z3.Implies(z3.And(mma, dtype == 1), bk % 4 == 0))
s.add(z3.Implies(mma, bv % (8 * warps) == 0))
s.add(z3.Implies(z3.And(mma, pipe > 0), K >= bk * pipe))
s.add(z3.Implies(z3.And(z3.Not(mma), pipe > 0), K >= bk))
ws_ok = z3.And(stage == "kkt", dtype == 0, mma)
s.add(z3.Implies(ws, ws_ok))
s.add(z3.Implies(tma, ws))
s.add(z3.Implies(ws, pipe == 2))
```
[tools/gdn_solver.py:126-145，省略注释]

前三条把精度与数据类型锁在一起：bf16 对应 prec=bf16，f32 对应 tf32；前两条已经完全决定 prec，no-MMA 的第三条重复同一规范映射，不再增加自由度。MMA 对齐对应真实原子：M 高为 16，bf16 的 K 原子为 16，tf32 为 4；N 按 8×warps 分配，保证每个 warp 至少得到完整的八列份额 [tools/gdn_l2_gen.py:296-318]。

流水约束限制深度不得超过可用 K 趟数。warp 专化只开放给 kkt 的 bf16 MMA；TMA 蕴含 warp 专化，因为描述符、生产者和 mbarrier 协议都位于该模板。warpspec 使用 identity transform，`pipeline` 在那里不改变工件，固定为 2 只是规范形，避免枚举出文本不同而调度等价的重复点 [tools/gdn_solver.py:138-145；tools/gdn_l2_gen.py:1108-1112]。

用真实内建配置 `kkt_bf16_mma_p3_t64x64x32_w4` 代入可以看到断言怎样协同：K=128，bt=64、bv=64、bk=32、warps=4、pipe=3 [tools/gdn_l2_gen.py:1185-1189]。三个 tile 都整除全局维；M 和 bf16 K 对齐成立，64 能被 8×4 整除；三阶流水需要至少 96 个 K 元素，实际有 128 个。warpspec 与 TMA 关闭，相关蕴涵不额外限制该点。结构检查通过后，普通 MMA 的单版本 A/B 缓冲为 8192 B，三个版本合计 24576 B；128 线程按 255 寄存器估计占 32640 个，也低于 SM 预算。这个手算展示的是每条约束的可解释性，最终仍要生成并检查真实产物。

### 3. 寄存器与三种共享内存账

资源公式用 `If` 按 dtype、warp 专化、TMA 和流水开关选择字节数。下面保留当前求解器原式，其中 no-MMA 一行必须与生成器实际分配对账。

```python
esz = z3.If(dtype == 0, 2, 4)
threads = 32 * (warps + z3.If(ws, 1, 0))
s.add(threads * 255 <= 65536)
pair = (bt * bk + bk * bv) * esz
smem_mma = z3.If(ws,
                 pair + z3.If(tma, 64 * 64 * esz, 0) + 8,
                 pair * z3.If(pipe > 0, pipe, 1) * self.n_gemm)
smem_nomma = (bt * K + K * bv + bt * bv) * esz
smem = z3.If(mma, smem_mma, smem_nomma)
cap = (cm.SMEM_USABLE_B if mode == "strict" else 220 * 1024)
s.add(smem <= cap)
```
[tools/gdn_solver.py:148-160]

寄存器账按每线程 255 个上限估计，warp 专化增加一个生产者 warp。每 SM 的 65,536 个寄存器是结构预算，因此八 warp 再加生产者会得到 288 线程并越界；普通八 warp 的 256 线程则只剩极小的分配余量。255 是该算子族的保守模型假设，实际每个入口仍以 ptxas 报告为准 [tools/gdn_solver.py:149-152；evidence/G3/solver_report.md:45-56]。

共享内存有三种形态。普通 MMA 的 `pair` 是 A、B 两个 tile，流水深度决定版本数，pipeline=0 按单缓冲计，o 阶段再乘两个 GEMM。warp 专化采用单份输入缓冲，TMA 增加 64×64 输出 staging，另计 8 字节 mbarrier。no-MMA 让两个操作数的完整 K 维驻留，并给累加 tile 记账。strict 容量为 100,352 B，即 99 KiB opt-in 减 1 KiB 保留；generator 模式的 220 KiB 只镜像生成器校验上限，不能替代 strict 硬件预算 [tools/gdn_costmodel.py:112-114；tools/gdn_solver.py:157-160]。

### 4. 从断言追到 IR 与失败现场

| 约束 | 对应 IR 位置 | 违反后的可观察失败 |
|---|---|---|
| 类型与精度耦合 | MMA fragment 类型、形状和精度属性 | 生成器抛出配置错误；手改 IR 可能触发 verifier 或下降失败 |
| 全局维与 tile 整除 | launch 网格、块偏移、共享 tile 类型 | 生成器拒绝残块；绕过检查会造成覆盖缺口或越界风险 |
| MMA 的 M/N/K 对齐 | fragment 下标与 `nvgpu.mma.sync` | fragment 份额不完整，校验、验证或 lowering 失败 |
| warp 数与 N 分配 | 线程几何和每 warp 的 N 偏移 | 某些 warp 没有完整原子份额，生成器拒绝配置 |
| 流水深度与 K 趟数 | multibuffer、异步组和 K 循环 | 生成器拒绝深度越界；强行改写会使流水前提失效 |
| TMA 与 warp 专化耦合 | 描述符、生产者分支、mbarrier | 生成器拒绝缺少协议载体的组合 |
| 寄存器预算 | launch 线程数和 ptxas 分配 | 模型排除超预算点；实际超额会影响驻留或装载 |
| 共享内存预算 | workgroup 分配及其缓冲版本 | 生成器容量报错，或后续汇编、装载与驻留核算失败 |

这张表提供双向审计路径：由约束能找到它保护的 IR，由 IR 反推字节数或对齐条件也能检查断言。生成器的 `ValueError`、MLIR verifier、ptxas 和 CUDA 模块装载是不同关卡，任何一个通过都不能替代其余关卡。

### 5. no-MMA 公式的一致性问题

当前生成器已经明确把 no-MMA 累加器改为 f32：

```python
esz = 2 if dt == "bf16" else 4
smem = (bt * bk + bk * bv) * esz
if cfg["mma"] and cfg["pipeline"] > 0:
    smem *= cfg["pipeline"]
if cfg["mma"] is False:
    # G4: G2 operand staging + f32 accumulator (see gen_payload_linalg)
    smem = (bt * k_glob + k_glob * bv) * esz + bt * bv * 4
if smem > 220 * 1024:
    raise ValueError(f"smem budget exceeded: {smem}B")
```
[tools/gdn_l2_gen.py:130-138]

`gen_payload_linalg()` 生成的 `ac_t` 形状为 bt×bv、元素类型为 f32，说明常数 4 来自实际 IR，而非额外保守余量 [tools/gdn_l2_gen.py:768-799]。求解器第 157 行与成本模型第 567 行仍使用 `(bt*K + K*bv + bt*bv)*esz`，让累加器随输入 dtype 取 2 或 4 字节 [tools/gdn_costmodel.py:557-568]。两处模型一致地保留了旧式，却与当前生成器不一致。

对账应从生成物的分配清单开始，逐个记录元素类型、形状、分配次数、缓冲版本和生命周期，再把这些量映射到公式的一项。solver 与 costmodel 使用相同旧公式只能证明两处彼此一致，不能构成对生成器的独立验证。还应注意 workgroup 分配在后续 PTX 修复中可能改变表现形式，所以静态公式、生成器预估和 ptxas 报告要按同一语义层比较。一个可重复的检查是挑选位于容量边界附近的点，分别手算三份结果，再用实际分配类型判定哪一项发生漂移。

以一个真实合法的边界配置为例：o 阶段、bf16、K=192、V=128、tile.M=64、tile.N=128、tile.K=64、mma=off。旧估算为 `(64*192 + 192*128 + 64*128)*2 = 90,112 B`，低于 strict 100,352 B；按当前生成器的 f32 accumulator，应为 `(64*192 + 192*128)*2 + 64*128*4 = 106,496 B`，超过 strict 6,144 B。生成器的 220 KiB 上限接受该点，strict 求解域却应把它排除。这个例子同时通过了 tile.M=64 整除 64、N=128 整除 V=128、K=64 整除 192 的结构检查，避免拿一个生成器本来就不接受的角点讨论模型误差。

因此，G3 报告中的 368、465、204 分别只能作为 kkt K=128、o K=128/V=128、o K=64/V=64 在归档模型下的空间记录 [evidence/G3/solver_report.md:72-83]。`unsat` 能证明给定断言已被穷尽，不能证明断言与现实一致。当前没有重算后的修正数量，也没有已修复的实现可供引用；正确的审计顺序是先统一生成器、solver、costmodel 的元素宽度与分配生命周期，再重新枚举、编译并复核资源报告。
