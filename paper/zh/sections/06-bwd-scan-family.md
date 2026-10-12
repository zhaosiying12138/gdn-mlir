# 6 反向 kernel 体系：扫描族

> 覆盖对象：`tools/gdn_bwd_gen.py`（8687 行）中的扫描类 kernel——前向状态重算扫描 `fwdscan`，反向携带族 `caryA` / `caryM` / `bpre` / `dhs`，以及 pre-update 部分和的并行重算 kernel `dkb`；配套的 CPU 数学验证脚本 `tools/gdn_carry_validate.py`、`tools/gdn_fwdscan_ksplit.py`；全部性能与数值数字取自 `evidence/G5/bwd_report.md` 及其引用的 json/npz 证据链。本章按**最终系统**呈现：只描述装机（ship）形态与其设计依据，历史演化仅以"结构前/后对比"的形式出现；测量口径均为同窗口配对交错 A/B（CUDA event、30 轮取中位数），数值门槛为 fla 派生契约 dq/dk/dv ≤ 8e-3、dbeta/dg ≤ 2e-2、dh0 ≤ 8e-3，确定性契约为新起进程位级一致（fresh-process bit-identical，N=5–10）。

## 6.0 章首：反向为何是"重算前向 + 两条方向相反的扫描"，以及九 kernel 拓扑

### 6.0.1 反向计算的两遍结构

分块 GDN（chunked GDN，块长 C=64）的反向传播在数学上是严格的两遍算法（numpy 孪生 `tools/gdn_l1_check.py::chunked_bwd_ref`，与 JAX vjp 在 fp64 下对到 1.4e-14）：

**第一遍——重算前向。** 反向梯度需要前向的全部中间量（每块的 WY 表示与入场状态），而这些量在前向推理结束后并不保留，必须在反向 kernel 内重算：

- 每块的代数重算：块内门控累积 `gc = cumsum(g)`（inclusive，log2 域）、核积 `kkT = k k^T`、WY 下三角系统 `A = strict_tril(beta .* kkT .* 2^(gc_i−gc_j))`、块伴随 `(I+A)^{-1}` 的分块求解、`u = T(beta v)`、`w = T(beta k 2^gc)`、`P = tril((q k^T) .* 2^(gc_i−gc_j))`——这些是纯逐块代数，天然并行，由 **wy** kernel（grid B·H·NC）完成；
- 跨块状态递推：`v_new = u − w S`、`S' = 2^gcl S + (k 2^(gcl−gc))^T v_new`（`gcl` 为块末累积门），同时把每块**入场状态** `h_all[n] = S` 与 `v_new` 落盘供反向消费——这是一条**顺时间方向的串行扫描**，由 **fwdscan** kernel（grid B·H·NV，NV = V/BV 个 v-切片各自独立行走）完成。

**第二遍——反向（逆时间）扫描与逐块局部项。** 以终端余切 `dht` 为种子逆时间行走：

```
dvp_n = P_n^T dO_n + Bhat_n X_n
X_{n-1} = 2^{gcl_n} X_n + Qg_n^T dO_n − W_n^T dvp_n        （X_n = 到达块 n 的 dH）
```

其中 `dO = do/sqrt(K)`，`Bhat = k .* 2^(gcl−gc)`、`Qg = q .* 2^gc`。这条逆时间扫描在最终系统中被重构为**仿射携带形式**（§6.1）：算子预复合（**caryA** / **caryM** / **bpre**，grid 均为 B·H·NC，纯并行）+ 携带链本体（**dhs**，grid B·H·NV）。行走沿途还需要三个消费 X_n（pre-update 值）的部分和——dkB、rowB、s5——它们被剥离为独立的逐块并行 kernel **dkb**（grid B·H·NC）。

因此，反向链整体是"**重算前向 + 两条方向相反的扫描**"的架构：九个 kernel 中只有两个含跨块串行递推——fwdscan（顺向重算）与 dhs（逆向余切携带）——其余七个全部是逐块（或逐 v-切片） embarrassingly parallel 的代数 kernel。两条扫描脊就是反向链的延迟骨架：所有性能工作的目标，要么把工作从串行链上剥离（携带重构、dkb 拆分），要么压缩扫描步本身的延迟（rgemm 的 ILP 重构、A 窗口、跨块预取），要么打开并行度（BV 切片、2-CTA 驻留）。而所有这些手段都受同一条铁律约束：**扫描状态 S 与 X 必须保持 f32 值类**——把递推状态本身 bf16 化会在 32/128 块上复合粗化 h_all/vnew/dh_all，因此并行度的每一次扩张都必须是"纯列并行"或"已证明的重结合类"。

### 6.0.2 九 kernel 拓扑（最终装机形态）

链的发射见 `tools/gdn_bwd_gen.py::main()`，默认 `--kernels wy,fwdscan,caryA,caryM,bpre,dhs,dkb,dqkwg,wyb`，以扁平化 `[M, C·W]` 全局量（M = BH·NC，m = bh·NC + n）串联，零参 cubin、harness 经 `bwd_manifest.json` 逐级搬运：

| # | kernel | grid | 职责 | 串行性 |
|---|--------|------|------|--------|
| 1 | wy | B·H·NC | 逐块 WY：gc 累积、kkT、A、(I+A)^{-1} 分块解、u/w | 并行 |
| 2 | **fwdscan** | B·H·NV | **顺向状态扫描**：h_all、vnew_all | **串行（NC 步/CTA）** |
| 3 | **caryA** | B·H·NC | A_all[m] = 2^gcl I − W^T (k·2^(gcl−gc))，[K,K] | 并行 |
| 4 | **caryM** | B·H·NC | M_all[m] = Qg^T − W^T P^T，[K,C] | 并行 |
| 5 | **bpre** | B·H·NC | B_all[m] = M_all[m] @ (do·K^{−1/2})，[K,V] | 并行 |
| 6 | **dhs** | B·H·NV | **逆向携带链**：X_{n-1} = A X_n + B；dh_all、dh0 | **串行（NC 步/CTA）** |
| 7 | **dkb** | B·H·NC | dkB_sum / rowB_c / s5_c（消费 pre-update dH） | 并行 |
| 8 | dqkwg | B·H·NC | dq、dW、dk_local、r3/r4/c4r、dg_state | 并行 |
| 9 | wyb | B·H·NC | dAf 修正链、最终 dk/dbeta/dg | 并行 |

本章覆盖第 2–7 号（扫描族）；wy / dqkwg / wyb 属逐块代数族，见他章。旧的单一反向行走 kernel `dhu`（把 dvp 计算、dH 更新、dkB/rowB/s5 全部串在一条逆时间链里，约 8 个 GEMM 级段落/块）保留在文件中仅作 A/B 基准（`--kernels` 可选），装机链不再使用；它 2.17 ms 的阶段耗时与"每块三趟 GEMM 级部分和挂在延迟关键路径上"的结构缺陷，正是携带重构与 dkb 拆分的动机。

### 6.0.3 全章通用的平台常数与判据

先定义后用，全章共享：

- **形状记号**：B、T、H、K=128、V=128、C=64（块长）、NC=T/C、BH=B·H、M=BH·NC；门 g 为自然对数域预乘 1/log2 得到的 **log2 域** f32，`gc` 为块内 inclusive 累积（f32 串行链），`gcl = gc[C−1]`。
- **sm_120a 静态共享内存上限 101376 B**（ptxas 实测；98304 B 接受、131072 B 拒绝）。2 CTA/SM 的静态池门槛约 50.2–50.7 KB（dqkwg 按 50176 B、fwdscan/dkb 按 50688 B 口径）；256 线程下 2×256×R ≤ 65536 给出寄存器门槛 R ≤ 128。
- **r6 危险规则**：`ldmatrix` 的操作数 tile 必须 `st.shared` 写入或逐元素 staging——**绝不**允许原始 LDGSTS（cp.async）落地后直接喂 LDSM（本工具链上静默损坏）；原始 cp.async 落地的 tile 只允许标量 `ld.shared` 消费。**r9 规则 3**：每个 `wait_group` 之后必须 `gpu.barrier` 再跨 warp 读。**r10 复用栅栏律**：缓冲区下次重 staging 前，其上一轮读者必须已被一道 trailing barrier 收口（复用距离 ≥ 2 次迭代即天然满足）。
- **split-bf16 三链接**：`hi = bf16(x)`，`lo = bf16(x − hi)`，则 `x ≈ hi + lo` 携带约 16 位尾数；`hi·hi + hi·lo + lo·hi` 三条 mma（弃 lo·lo ≈ 2^-32）恢复 ~f32 质量乘积（残差 ~2^-17），是 tf32（双操作数 2^-11 → 2^-17 类）在数值敏感点的替代路径（`_split_bf16`，tools/gdn_bwd_gen.py）。
- **cp.async 手工发射**：`astage_tile`（16 B 线程跨步突发，token 由调用方组队）+ `agroup` + `await_all`——不用 `nvgpu.device_async_create_groups`/transform 的 async-copy 组（其 lowering 在每组后自动插 `wait_group 0`，会把预取与它本应覆盖的计算串行化）。

---

## 6.1 仿射携带数学：反向余切扫描的精确重写（教学式推导）

本节是 caryA/caryM/bpre/dhs 四个 kernel 的共同数学基础。结论先行：**逆时间余切扫描精确地等于一个逐块仿射线性递推**

$$X_{n-1} = A_n X_n + B_n,\qquad X_{NC-1} = \mathrm{dht},\qquad A_n = 2^{gcl_n} I_K - W_n^T \hat{B}_n,\quad B_n = M_n\, dO_n,\ \ M_n = Qg_n^T - W_n^T P_n^T,$$

其中 $X_n$ 恰是**到达块 n 的 dH**（即 dh_all[m] 的 pre-update 契约值），$W = w_{all}$（wy 输出，f32）、$\hat B = k \cdot 2^{gcl-gc_{row}}$（Bhat）、$Qg = q\cdot 2^{gc_{row}}$、$P = P_{all}$、$dO = do/\sqrt K$。这不是近似，而是顺序更新的**代数重结合**。

### 6.1.1 推导

从顺序形式出发（每块两式，见 `gen_dhu` 与 numpy 孪生）：

$$\mathrm{dvp}_n = P_n^T\,dO_n + \hat B_n X_n \tag{1}$$
$$X_{n-1} = 2^{gcl_n} X_n + Qg_n^T\, dO_n - W_n^T\, \mathrm{dvp}_n \tag{2}$$

(1) 描述余切在块内先经过 $P^T$（token 间路径）与 $\hat B$（状态路径）汇入 dvp；(2) 描述状态余切的衰减与两路注入。将 (1) 代入 (2)：

$$X_{n-1} = 2^{gcl_n} X_n + Qg_n^T dO_n - W_n^T\left(P_n^T dO_n + \hat B_n X_n\right)$$

按 $X_n$ 与 $dO_n$ 收集：

$$X_{n-1} = \underbrace{\left(2^{gcl_n} I_K - W_n^T \hat B_n\right)}_{A_n\ \in\ \mathbb R^{K\times K}} X_n + \underbrace{\left(Qg_n^T - W_n^T P_n^T\right)}_{M_n\ \in\ \mathbb R^{K\times C}}\; dO_n .$$

即 $B_n = M_n dO_n \in \mathbb R^{K\times V}$。**关键结构性观察**：

1. **$A_n$ 与 v-切片无关（vt-INDEPENDENT）**。$A_n$ 只依赖 $W$（f32）、$k$（bf16 全局）、$g$——不含任何 v 或 do 分量。于是一个 $[K,K]$ 算子每（bh, 块）只需复合一次，即服务全部 NV 个 v-切片的行走。
2. **$M_n$ 同样 vt 无关**，只有最后一步 $B_n = M_n dO_n$ 按 v 列切分（$[K,C]\times[C,V]$ → 每 BV 列一片）。
3. **边界与产出**：种子 $X_{NC-1} = dht$；每步之前 $X_n$ 恰为 dh_all[m] 要存的 pre-update 值（dkb/dqkwg 的消费契约）；走完最后一步（n=0）得 $dh_0 = X_{-1}$。这与顺序实现的存取逐位同义——重写不改变任何被消费量的语义。

CPU 验证脚本 `tools/gdn_carry_validate.py` 的对照对象是**真顺序形式**（保留 dvp 中间量，即 (1)(2) 原式，fp64），而非复合式自身，避免"用自己验自己"：

| 检查（fp64，两个门形状 T512/T2048） | 结果 |
|---|---|
| 复合式 vs 真顺序式（含 dh0） | max abs **4.9e-17 / 8.3e-17**——在 \|X\|~130 尺度上为 **2–3 ulp** |
| 二级段分解（每段零种子局部走 + 段复合算子 + 串行边界链 + 后缀积修正），S ∈ {2,4,8,16} | 同为 2–3 ulp |
| fp32 回放（GPU 操作数精度：bf16 展宽的 q/k/do、f32 W、f32 gc 累积） | 重结合底噪 **1.2e-8**（门分布） |
| 近单位化门压力测试（\|\|A\|\|≈8.8） | fp32 底噪 2.4e-6 |
| 与真实 `bwd_twin` 锚点（dh0，bh0） | 1.0e-10 / 7.6e-10（锚点差来自 gc 的 f32/log2 建模，非提取错误） |

GPU 上链内实测：dh_all 对顺序行走值的再现误差 **2.2e-8**——恰落在 CPU 预测的重结合底噪上。

### 6.1.2 超收缩性：ρ(A) ≤ 2.3e-18

对基准门分布（g = −softplus(randn)，log2 域），逐块算子的谱半径与范数为（fp64 全谱测量）：

$$\rho(A_n) \le 2.3\times 10^{-18},\qquad \lVert A_n\rVert \le 2.2\times 10^{-17}.$$

机理：$gcl$ 强负 ⇒ $2^{gcl}$ 极小，而 $W = T(\beta k 2^{gc})$ 的行尺度含 $2^{gc}$、$\hat B$ 的行尺度含 $2^{gcl-gc}$，乘积整体被 $2^{gcl}$ 压制——块间耦合**一步之内消亡**（$\lVert A\rVert^2 \sim 10^{-34}$，远低于 f32 eps 2^-24 ≈ 6e-8）。两点推论：

- **理论上**，在该分布上做深度截断的完全并行扫描在 fp32 内就是精确的（边界修正只传一步）；
- **工程上不这么做**：弱门分布（近单位化门）下 $\lVert A\rVert$ 可达 O(10)，截断即错误。系统**完整执行精确的一级递推**——正确性不依赖门分布，超收缩性只是"为什么 dhs 的串行链在数值上如此温顺（误差不随 NC 放大）"的解释，以及"段切分等更强并行化在此分布上不会赢走数量级"的预兆（§6.6.1）。

### 6.1.3 重构的工程含义

顺序形式把每块的全部操作数工作（P/Bhat/Qg/W 的装配与三个 GEMM）钉在串行依赖链上；仿射形式把它拆成三层：

1. **并行预复合层**（caryA/caryM/bpre，grid B·H·NC）：每块独立造 $A_n$、$M_n$、$B_n$，存入全局 `A_all[M,K·K]` / `M_all[M,K·C]` / `B_all[M,K·V]`（均 f32）；
2. **串行链层**（dhs，grid B·H·NV）：每块只剩**一个** GEMM——`Xa = B-切片; Xa += A_all[m] @ X; X = Xa`——外加两次 K×BV 的全局搬运；
3. **余量归位**：顺序形式沿途算出的 dvp_all 本就被 dqkwg 逐块恒等地重算（同 GEMM、同划分、同顺序、读同一 dh_all 值），dv 移到 wyb 的 `tv .* beta` 尾部——携带重构后这些输出各有唯一生产者，验证链闭合（此前 dhu 的"清洁 dvp_all"曾掩盖 dqkwg 侧一个潜在的 bf16-Bhat 污染：dhu 退役后 dbeta 从 3.0e-4 跳到 1.33e-2，二分定位到 dqkwg 自 r2 起把 Bhat 暂存进 bf16 tileCKb——修为闲置的 f32 tileCVb；该 bug 之所以长期隐形，是因为验证 harness 每次启动前重 staging 输入全局量，wyb 总是读到 dhu 的干净 dvp_all。重构本身充当了揭发者）。

测得总账（T2048 同窗口配对）：扫描复合体（旧 dhu + dkb = 3.13 ms）→（caryA+caryM+bpre+dhs+dkb = 2.01 ms）= **1.56×**；其中 dhu 2.17 ms → 0.41+0.46+0.29+0.59 = 1.76 ms，dkb 0.96 → 0.25 ms。以下各节逐 kernel 展开。

---

## 6.2 fwdscan：重算前向的顺向状态扫描

### 6.2.1 动机（算子层面）

反向需要每块入场状态 `h_all[n]` 与块内新值 `v_new[n]`（dk 的 Bhat 项、dv、dW 等全都消费它们），它们只能由顺向扫描重算。kernel 每个 (bh, vt) 配一个 CTA，行走：

```
每块 n：h_all[n] = S（存入场态）
        acc = w_n @ S          （K 收缩，BK=64 切片链）
        vn  = u_n − acc        （→ vnew_all[n]）
        S  *= 2^gcl            （逐元素衰减）
        kg  = k_n .* 2^(gcl−gc_i)   （[C,K] f32 行缩放）
        S  += kg^T @ vn        （C 收缩）
```

即 $S' = MS + b$、$M = 2^{gcl}I - kg^T w$、$b = kg^T u$——但 M 从不显式装配，三因子行走就是 kernel。它与前向 F2 kernel 跑**同一递推**（前向侧 83 µs），却在改造前耗时 916 µs（11×）：ncu 剖面显示这是纯延迟核——**88.3% 无合格周期、L1TEX 记分板占 17.0 cyc/inst 停顿平均的 43.9%、每调度器 2 warp（64 CTA、1/SM、192 寄存器、0.39 波、DRAM 10.7%）**，没有任何资源饱和。定位出的四个杀手：每块一条 64 次依赖全局载入的**串行 gc 累积**、`astage_sync` 式"发射即排空"的**零重叠 staging**、每块 9 道 barrier、每块 5 趟独立的 stage/emap/store 扫。f32 载荷（h_all+vnew 写 50 MB + w/u/k 读 84 MB @T2048，约 800 GB/s 有效带宽下 ~170 µs 流量地板）是诚实的 2×，不是 11× 的来源。

### 6.2.2 源码片段（gen_fwdscan_v3：装机主体）

chunk 循环核心（`tools/gdn_bwd_gen.py::gen_fwdscan_v3`，节选）：

```python
# group A: w[n, ks=0] -> wpp col half 0（由 h_all 存储扫覆盖）
astage_tile(e, G["w_all"], wkty, "%m", K, 0, wpp, 0, C, BK, "f32", toksA)
gA = agroup(e, toksA)
# h_all 入场态存储（逐元素）……
await_all(e, gA); e.l("gpu.barrier")
gemm(e, acc, wpp, S, C, BV, BK, b_off=(0, 0), tag="fwdscan.wS")   # gemm-a 片 0
toksB = []   # group B：w 片 1 + k 全量 bf16 + u，飞在 gemm-a 下面
astage_tile(..., wpp, BK, ...); astage_tile(..., kpp, ...); astage_tile(..., utl, ...)
gB = agroup(e, toksB); await_all(e, gB); e.l("gpu.barrier")
gemm(e, acc, wpp, S, C, BV, BK, a_off=(0, BK), b_off=(BK, 0), acc=True, ...)  # 片 1 链入
# vn = u − acc：一趟写 vn tile + vnew_all 全局；S *= 2^gcl（无 barrier）；
# kg 扫：kpp .* expc → 原地覆写已死的 wpp；ONE 全 K tgemm：S += kg^T @ vn
gemm(e, S, wpp, vn, K, BV, C, at=True, acc=True, tag="fwdscan.kg")
```

两条 tf32 m16n8k8 递推 GEMM（站点 `fwdscan.wS` / `fwdscan.kg`）经逐点二分验证收编（h_all 扰动 8.0e-4 rel、vnew 5.5e-5 rel，最终梯度位移 ≤ 3.0e-4，被既有最大误差单元吸收）；`expc[i] = 2^(gcl−gc_i)` 每块只算一次（原为每 kg 元素一次 MUFU）；vn/衰减/kg 三趟互不重叠的扫**背靠背免 barrier**，每块 barrier 从 9 降到 5。

**NC 免疫混合 prologue（GDN_FSCAN_GSCG）**——本章标题所指的机制。v3 的原设计把全部 NC 条块内 gc 累积行停进一块 `[NC, C]` f32 smem tile；池随 NC = T/64 线性增长，判据即发射期一行算术：

```python
_base = (K*BV*4 + C*2*BK*4 + C*K*2 + 3*(C*BV*4) + C*4)      # BV=32 时 90368 B
GSCG_GLOBAL = _base + NC*C*4 > 101376                        # gsc tile 是否还塞得下
```

- **塞得下**（T2048：90368 + 8192 = 98560 B；T512 类似）：smem 臂**逐字节保持 v3**——payload、cubin 全部与基线相同（md5 e680e1d94e47），**零回归由构造保证**，这就是"位级零回退"：混合开关以池不等式为唯一判据，旗舰形状不付任何代价。
- **塞不下**（T8192：90368 + 32768 = 123136 B > 101376 B 静态上限——原设计在该形状**根本不可编译**，辅助 v2 链实测 11.42 ms）：累积行改落新全局 `@gsc_all [M, C]` f32。数值等价的关键是**同一串行序 f32 链改由 `scf.for` iter_args 携带**（寄存器驻留、无读回：同样的加法按同样顺序发生 ⇒ gc 值位级一致）：

```python
e.l(f"%gtfr = scf.for %n = 0 to {NC} step 1 "
    f"iter_args(%gtok = {gpfl}) -> (!nvgpu.device.async.token) {{")
# 块顶：await 上一块预取的 gsc 行 + barrier；行 n 的读法：gr[%grp, %xe]
# 块中：把行 n+1 以单元素 4B cp.async.ca 预取进 gr 的另一行（ping-pong），token 随 yield 回传
```

消费侧是 2 行 `[2, C]` ping-pong（gr，512 B），L2 读藏在上一块的计算下面；每线程无条件发一个 4 B 拷贝（`tid % C` 覆盖，重复写字节相同——良性自重复类，**非 bypassL1**：4 B 拷贝必须降为 `.ca`，`.cg` 仅 16 B）。NV 个同 bh 的兄弟 CTA 以相同值写同一批字节——良性（各 CTA 只在自家 barrier 之后读自家行）。**S 保持 f32 [K, BV] 递推态不动**（两臂皆然）：切分它会粗化 h_all/vnew。池在 T8192 降到 90880 B（90368 + 512）。

### 6.2.3 结构前后对比

| 维度 | v2（回退臂） | v3 + GSCG 混合（装机） |
|---|---|---|
| gc 累积 | 每块一次 64 次依赖全局载入的串行链，**钉在每块关键路径上** | 全部 NC 条链提升为 prologue（线程 t < NC 串行走块 t，同序 f32 ⇒ 位级同值）；smem 臂整块 tile、超限臂 `@gsc_all` + iter_args 链 + 2 行 ping-pong |
| w/k/u staging | `astage_sync`（发射即排空，零重叠） | 组 A（w 片 0，藏在 h_all 存储下）+ 组 B（w 片 1 + k + u，飞在 gemm-a 下）；消费者仅标量 ld.shared（r6） |
| GEMM | 标量 sgemm（每 MAC ~5–6 条指令、2 次 smem 载入） | 两条 tf32 tgemm；gemm-b 合一为单次全 K 调用（原两片调用的 acc 链式重结合——已 6/6 验证） |
| exp2 | 每 kg 元素一次 MUFU | 每块一次 expc[C] 扫 + 逐元素乘 |
| barrier/块 | 9 | 5（vn/衰减/kg 三趟免 barrier 背靠背） |
| smem | 65792 B | 98560 B（T2048）/ 90880 B（T8192 全局臂）；202/118 寄存器，0 spill |

未随 v3 落地并记录为否定的两项：**跨块 w 预取（v3b）**——裸 `wait_group N` 的 lowering 确实可用，solo 实测 0.358 ms 更快，但预取目的地（唯一 wpp tile）在拷贝落地时**不是死 tile**：与同块 kg 扫写、gemm-b 读竞态（verify：dk 31、dg 2.1 FAIL），正确版需第三块 [C,K] 缓冲，99 KB 预算放不下——撤销并记录；**bf16 mma + ldmatrix 操作数 tile**（F2 的精确 GEMM 形态，可砍 ~10× MIO 压力）会改 h_all/vnew 值类，tf32 路径已在既有值类上通过，该风险未花费。

### 6.2.4 效果

- fwdscan **0.916 → 0.374 ms @T2048（2.45×）**、0.203 → 0.097（2.11×）@T512（同窗口配对）；
- T8192 链从"不可编译"变为可建：fwdscan 1.487 ms，bwd e2e **8.670/8.611 ms，较 v2 回退的 11.42 ms 为 1.32×**；6/6 通过（dq 2.078e-4/38×、dk 1.881e-3/4.3×、dv 1.050e-5/762×、dbeta 1.654e-3/12×、dg 1.145e-3/17×、dh0 1.965e-7/40700×）；
- 全局臂在 T2048 强制开启实测 **0.96×（0.374→0.388）**：prologue 的散布 4 B 全局写约 15 µs，cp.async 预取未能收回（读延迟本已被藏住）——这一实测回退正是混合门"只在 smem tile 放不下时才切全局臂"的设计依据；
- 确定性：新起进程 N=5 位级一致（含全局臂）。装机链 fwdscan 最终耗时 0.376 ms @T2048 / ~1.50 ms @T8192。

---

## 6.3 caryA / caryM / bpre：携带算子的逐块预复合

三个 kernel 都是 grid B·H·NC 的纯并行 kernel，分别预复合 §6.1 中的 $A_n$（[K,K]）、$M_n$（[K,C]）、$B_n$（[K,V]）。它们的共同设计语法是：**split-bf16 mma 化 + tile 消解（diet）+ epilogue 全局直存**。

### 6.3.1 caryA：A_all = 2^gcl I − W^T (k·2^(gcl−gc))

**动机（算子层面）。** $A = 2^{gcl}I - W^T \hat B$ 是 [K,K] 乘积 $W^T[K,C] \times \hat B[C,K]$ 加对角。tf32 化（早期形态）只拿到 1.39× 并卡死在 255 寄存器 / 82176 B smem = 1 CTA/SM。瓶颈不是 mma 缺席而是**四块共活 tile 的池**：原始 k 暂存 `kst[C,K]` bf16、Bhat 复合 `bhb[C,K]` f32、W 列块 `wb[C,BK2]` f32、输出块 `ab[BK2,BK2]` f32——按 (i0, j0) 双重块循环迭代时全部同时活着，**不重构就没有合法的 lifetime 别名**。

**源码片段（mma 臂，gen_caryA）**：

```python
seg = k.alloc(e, "seg", C, 1, "f32")                 # 2^(gcl - gc_c)
for i0 in range(0, K, BK2):                          # A 的行块（W^T 的行）
    stage_split_T(e, ahi, alo, G["w_all"], wkty, m, K, i0, BK2, C, cvec=seg)
    for j0 in range(0, K, BK2):                      # A 的列块（k^T 的行）
        stage_raw_T(e, bt, G["k"], kty, m, K, j0, BK2, C)
        def epi_ca(mi, ni, ei, row, col, acc):       # -neg + 对角 + 全局直存
            x = mma_extract(e, acc, ei // 2, ei % 2, "cax"); nx = negf(x)
            if i0 == j0:   # 对角块：row==col 处 += 2^gcl（select 融入 epilogue）
                dz = select(cmpi eq row col, 2^gcl, 0.0); nx = addf(nx, dz)
            gstore(e, nx, G["A_all"], aty, [m, row*K + col + i0*K + j0])
        accs = mma_gemm_mh(..., ahi, bt)                      # hi 链接
        accs = mma_gemm_mh(..., alo, bt, accs=accs, epilogue=epi_ca)  # lo 链接
        e.l("gpu.barrier")                            # r10 复用栅栏
```

**Bhat 因子折入 A splits 的幂二交换恒等式。** 旧路径先在 smem 里把 k 逐元素乘 `2^(gcl−gc_c)` 造成 Bhat（f32 tile），再当 GEMM 的 B 操作数。新路径把这个**列因子折进 split 本身**：`stage_split_T` 的 `cvec=` 参数在截断前对源列乘 `seg[c] = 2^(gcl−gc_c)`。其精确性是一条恒等式：对任意整数 d，

$$\mathrm{split}(v\cdot 2^d) = \mathrm{split}(v)\cdot 2^d$$

——2 的幂与 bf16 截断**可交换**（只移指数域，不碰尾数舍入），也与 hi/lo 残差可交换（残差同步移指数）。因此乘积保持"B 侧已缩放"的旧值，bhb 整块退役，`bt` 直接是**原始 k^T 的 raw bf16**（全局本就是 bf16——**精确**，无舍入）。对角 `2^gcl` 与取负折进 mma epilogue 的 select/addf；A_all 全局直存（bpre playbook，退役 store_rows 扫及其 barrier）。

**结构前后对比。**

| | 标量路径 | split-bf16 mma 路径（装机） |
|---|---|---|
| 共活 tile | kst [C,K] bf16 16384 + bhb [C,K] f32 32768 + wb [C,BK2] f32 16384 + ab [BK2,BK2] f32 16384（净和 81920 B，含垫报告口径 **82176 B**） | ahi/alo（W^T splits，[BK2, C+8] bf16 ×2 = 9216+9504，i0 一次 staging 供两个 j0 复用）+ bt [BK2+1, C+8] 9360（raw k^T）+ gc 256 + seg 256 = **28592 B** |
| 寄存器 | 255 | **76**（0 spill） |
| 驻留 | 1 CTA/SM | **2 CTAs/SM** |
| 数值 | f32 sgemm | 重结合实测：A_all 8.39M/8.39M 元素不同，max **5.6e-22 绝对**（2.8e-4 rel——被替换的 tf32 截断类；A 本身超收缩、值 ~1e-18 量级），全部最终梯度位不变 |

**效果。** caryA **0.3093 → 0.0618 ms @T2048（5.00×）**、2.78× @T512。5× 的结构原因即"**arena 消解式移植**"：不是给旧 tile 结构做别名（不重构就不合法），而是把 bpre 式 split-bf16 移植过来后四块共活 tile **整体退役**，池压掉 65%、寄存器压掉 70%，2 CTA/SM 的驻留增益与 mma 吞吐叠加在已 1.39× 的 tf32 底座上再拿 3.6×。附带否定：**w-窗口轮转**（GDN_CARYA_WF，i0 边界的 16 KB 同步 w_all 读改原始 cp.async 窗口跨整个 j0 循环飞行，`_split_T_smem` 从 smem 重分裂——位级一致）实测 **0.96×**：54 µs 的 kernel 里那个 i0 边界停顿早已摊薄，16 KB 窗口的 smem 流量 + await/barrier 边界更贵——默认关、旋钮留档。

### 6.3.2 caryM：M_all = Qg^T − W^T P^T

**动机。** $M$ 是 $[K,C]$ 乘积 $W^T[K,C]\times P^T[C,C]$（注意 P 侧转置）减 $Qg^T$。tf32 单站点形态 160 寄存器 / 64256 B。

**源码要点（gen_caryM，mma 臂）。** 3 链接 k16 切片 split mma，**双操作数都直接从全局切片**——"扁平 col0 算术即切片"：

```python
for i0 in range(0, K, BK2):                       # M 的 K 行块
    # mb 初始化扫（f32 精确保持 Qg^T 项）：mb[a,c] = q[c, i0+a] * 2^gc_c
    for kb0 in range(0, C, 16):                   # C 方向 k16 切片
        stage_split_T(e, whs, wls, G["w_all"], wkty, m, K, i0 + kb0*K, BK2, 16)
        stage_split_ew(e, phs, pls, G["P_all"], pty, m, C, kb0, C, 16)
        accs = mma_gemm(..., whs, phs)            # hi·hi
        accs = mma_gemm(..., whs, pls, accs=accs) # hi·lo
        accs = mma_gemm(..., wls, phs, accs=accs, epilogue=_cm_epi if 末片 else None)
        e.l("gpu.barrier")                        # r10 复用栅栏（下一片 restage 前）
```

A 侧（$W^T$ 的 [BK2,16] 切片）经**转置 staging** 的 `col0 = i0 + kb0·K` 取出（W 按 [C, C·K] 扁平行主序，col0 算术直接落在 k16 列片上）；B 侧（P 的自然取向切片）`col0 = kb0`。末片 epilogue 做"读-减-存"：`mb[row,col] −= acc`，把 f32 初始化值**精确保持**（Qg^T 项不经过 mma）。tf32 的双操作数 2^-11 在此换为单操作数分裂的 2^-17 类。

**结构前后对比。** 旧 tf32 形态：`astage_sync` 把整块 P 停进 `pt[C,C]` f32（16384 B）+ 每 i0 停 `wb[C,BK2]` f32（16384 B）+ `mb`，单站点 tgemm [BK2,C,C] at=bt=neg=acc 一次成型。新形态：pt/wb 全部退役，双操作数改从全局按 k16 切片直接分裂（whs/wls [BK2,24] + phs/pls [C,24]，四个小 split tile 共 ~6.5 KB），24 条 Kred=16 mma 链接按 i0 分组进寄存器累加器；池 64256 → **45504 B**、寄存器 160 → 76、0 spill、2 CTAs/SM。

**效果。** caryM **0.1647 → 0.0814 ms（1.96× 阶段 / 2.02× solo @T2048；1.41× @T512）**。下游 dh0 富余随之收紧（2.761e-6 → 2.169e-7，36900× 富余）——M_all 更净直接改善携带链种子质量。

### 6.3.3 bpre：B_all = M_all @ (do·K^{−1/2})

**动机。** $B = M\,dO$ 是 $[K,C]\times[C,V]$；注意顺序形式是逐元素缩放 dO，复合形式把 $K^{-1/2}$ **改为求和后乘一次**（全程只一次 f32 舍入）。

**源码要点（gen_bpre，mma 臂）。** A = M 行块的 hi/lo 融合分裂（f32 源，~2^-16 乘积——B_all 喂超收缩携带链，当时 dh0 富余约 4000×）；B = **原始 dO^T raw bf16**（全局即 bf16，`stage_raw_T` 转置暂存 [V, C+8]——精确无需分裂）；scale 在 epilogue（`epi_b`：`x * (1/sqrt(K))` 后 `gstore` 直存 B_all）。寄存器累加器跨整个 K=C 归约链共享；[BK2, V] f32 bb 暂存 tile 与其 scale 扫退役，smem **49 KB → 37 KB**（按 smem 获 2 CTAs/SM 而非 1）。W=8 无 warp 守卫。

**结构前后对比。** 旧标量形态：`stage_rows` 停 `dob[C,V]` bf16 + 每 i0 一块 `bb[BK2,V]` f32 输出暂存，`aggemm`（胞格主序、A 从全局串行读）求积后一趟独立 scale 扫再 `store_rows`。新形态：dob 换成一次转置 raw dO^T（dotT），bb 与 scale 扫退役（scale 折入 mma epilogue、B_all 全局直存），aggemm 换两条 mma 链接（bmh/bml）——**发射期指令数与 smem 同时下降**，且 A 载荷从"胞格主序串行 LDG"变为 ldmatrix。

**效果与一个被确定性关卡揪出的竞态。** 该 kernel 的 K 循环 mma **跨迭代 restage 竞态**（`stage_split_ew` 在慢 warp 的 ldmatrix 读仍在飞行时覆写 ahi/alo——r9 类漏网点）曾造成 1–3 ulp 的 B_all 翻转：被交换的操作数元素恰是近零的 LO 分量，acc 扰动极小，却沿 dh_all → dkB_sum 传成**一个 dk 元素的 bf16 舍入翻转**——即运行间 1.719e-3 ↔ 4.719e-3 的 dk 富余摆动（修复前 14/1000 新起模块发射复现；在每次迭代末补一道 `gpu.barrier` 复用栅栏后 **0/1000**）。这是"末位 ulp 竞态 × bf16 量化台阶"放大路径的标本，也是扫描族所有 mma 化 kernel 统一执行 r10 复用栅栏律的直接原因。

---

## 6.4 dhs：反向携带链本体

### 6.4.1 动机（算子层面）

携带链是反向图上唯一的逆时间串行依赖：$X_{n-1} = A_n X_n + B_n$。预复合之后，每步只剩：

```
dh_all[m] ← X          （pre-update 契约：先存后更新）
Xa = B_all[m] 的 v-切片 （[K, BV] f32）
Xa += A_all[m] @ X     （[K,K]×[K,BV]——每块唯一 GEMM）
X  = Xa
```

循环结束存 dh0。没有 P/T/k/q/w/beta 暂存、没有累积、没有 dvp/dv epilogue——smem 只需 X + Xa + 窗口。设计问题只剩两个：GEMM 怎么读 A（64 KB 的 [K,K] f32 staging tile 毫无意义），以及每步的 B-切片搬运与 A 读取怎么藏进计算。

### 6.4.2 rgemm：寄存器累加器 + kk 外层归约（每块单 GEMM 的串行链）

初版 GEMM（`aggemm`，胞格主序）让 dhs 停在 0.83 ms：每线程一个输出胞格、kk 内层一次全局 A 载入接一次 FMA——1 层深的依赖链把 LDG 延迟完全串行化。`rgemm` 反转结构：

- 划分：输出胞格 $e = i\cdot N + j$ 归线程 $t$（$e \in \{t + c\cdot NT\}$，CPC = M·N/NT 个/线程）；N = BV 整除 NT ⇒ **同线程全部 CPC 个胞格共享 j = t mod BV**——每个 kk **一次** b（smem）载入服务所有累加器；
- kk 外层 `scf.for` 的 iter_args 携带 CPC 个 f32 寄存器累加器；每 kk 做 CPC 次**相互独立**的 A 全局载入（`A[i_c, kk]`，行各异，warp 内广播）+ CPC 个 FMA——**ILP = CPC**（16）对抗 1 层依赖；
- `acc_buf` 参数以 smem 里的 B-切片作加性初值（"acc=True 但初值来自另一缓冲"）。

仅此一项 dhs **0.83 → 0.59 ms**。`rgemm_aw` 是其窗口变体（下述），胞格划分与 kk 升序 FMA 链**逐位相同**。

### 6.4.3 A-窗口 ping-pong（rgemm_aw，GDN_DHS_AW）

**动机。** rgemm 消除了依赖链，但每块仍有 **2048 次 warp 广播式全局载入/线程**（K×K 的 A 逐 kk 行读）——长记分板链使 dhs 仍是 #2 阶段。整块 [K,K] f32 staging 要 64 KB；答案是把 A 当流：**滚动 [K, BKW=32] f32 smem 窗口双缓冲（AW[0]/AW[1]，各 16 KB），cp.async 预取超前一个窗口**（`astage_tile` 16 B 突发，24 次/块）。

```python
for w in range(NW):                                  # NW = K // BKW
    if w + 1 < NW:                                   # 超前一个窗口发射
        astage_tile(e, G["A_all"], aty, "%m", K, (w+1)*BKW, AW[(w+1) % 2], 0, K, BKW, "f32", toks)
        gw = agroup(e, toks)
    rgemm_aw(e, Xa, AW[w % 2], w * BKW, X, K, BV, BKW, acc_buf=Xa)
    e.l("gpu.barrier")                               # r10：本窗口读毕，才允许 2 次迭代后的重staging
    if w + 1 < NW:
        await_all(e, gw); e.l("gpu.barrier")         # r9 规则 3
```

危险规则合规：窗口只喂 rgemm 的**标量 ld.shared**（r6 合法口味，绝不 ldmatrix）；窗口读以 trailing barrier 收口后其缓冲的下一次重 staging 至少在 2 次迭代之后（r10）。

**值等价证明（AW=0 对照）。** 同一种子输入、同 caryA cubin（A_all/B_all/dht 位级相同）下建 AW=0 与 AW=1 两条链：**dh_all 与 dh0 逐位相同**。窗口是纯延迟改动——窗口值是同批全局元素的原始 cp.async 字节拷贝，kk 升序 FMA 链未动。这一"证明而非假设"的对照是链内所有纯 staging 改动的范式。

**跨块预取（GDN_DHS_XP）。** 块 c 的**最后一个**窗口（w = NW−1）顺带发射**块 c+1 的窗口 0** 进 AW[0]（此刻空闲：w 自己的读在 AW[1]；AW[0] 的上轮读者已被 w−1 的 trailing barrier 收口——与块内轮转同一的 1 次迭代净空），cp.async 组令牌**搭 `%ci` 的 scf.for iter_args**（`!nvgpu.device.async.token`——本 MLIR 构建注册的助记符，'async.token' 是语法错误）进入下一迭代 prologue 的 await；末块发一次**无害自重复**（自己的窗口 0 进 AW[0]，无人读）保证携带令牌恒有效，循环出口排空（CTA 退出时无在飞 cp.async）。每块 prologue 的暴露等待从"仅被 B 扫覆盖"变为"被整整一块的工作覆盖"。`GDN_DHS_XP=0` 逐字节再生基线 payload——干净的二分臂。

**FPIPE：B-切片置顶。** B-切片（Xa，[K,BV] f32）原本是 dh_all 存储***之后***的同步 `stage_rows`；改为**块顶**发射的 cp.async 组（`GDN_BFPIPE` 名单含 "dhs"），由 dh_all 存储扫 + 窗口 0 await 覆盖。Xa 此刻已死（上一块的拷贝扫读过它、有 trailing barrier 收口）且只喂 rgemm 的标量读（r6）。字节拷贝 + 同值同算 ⇒ 位级一致。

### 6.4.4 BV=16：买并行度的池瘦身

BV=32 时 X+Xa+2×AW = 65536 B = 1 CTA/SM，而 grid BH·NV = 64 CTA 欠填充设备，串行 NC 行走是关键路径——单纯"瘦身到 2 CTA/SM"不改变任何事（grid 不变）。**有效瘦身是减半 BV**：X+Xa 32768 → 16384，池 49152 ≤ 50176，grid 64 → 128（2/SM），每 CTA 串行工作减半；A-窗口流量翻倍但每条 A 行现被 8 个 v-兄弟读——**留在 L2**。**值精确**：V 维是纯列并行——k 窗口行走、B 扫、每输出元素的 FMA 链全部不变（各形状位级一致已验）。`GDN_DHS_BV` 旋钮，=32 逐字节恢复。

### 6.4.5 结构对比与效果

| 维度 | rgemm 初版 | + A 窗口 | + 跨块预取 + FPIPE + BV=16（装机） |
|---|---|---|---|
| A 来源 | 全局 warp 广播载入（2048 次/线程/块） | [K,32] f32 双缓冲窗口，超前 1 窗口 | 同左 + 块边界令牌续飞 |
| B-切片 | 块内同步 stage_rows | 同左 | 块顶 cp.async，被存储扫覆盖 |
| smem | 32768 B（X+Xa） | 65536 B | **49152 B**（BV=16） |
| grid | 64 CTA | 64 | **128 CTA = 2/SM** |
| dhs @T2048 | 0.83 → 0.59（rgemm ILP） | 0.592 → 0.320（**1.85×**） | XP 1.03× → 0.312；BV=16 **1.101×** → 0.282；FPIPE 1.028× → 0.272 |
| dhs @T8192 / @T512(H4) | — | 1.83× | BV=16 1.073× → 1.086，FPIPE 1.034× → 1.047 / BV=16 **1.61×** → 0.041，FPIPE 1.065× → 0.038 |

否定记录：**BKW=16**（8 KB 窗口、48 KB 池、理论 2-CTA）实测 **0.91×**——窗口数翻倍使每窗口 wait+barrier+发射开销翻倍、每次预取下的 rgemm 覆盖减半，而 kernel 是内存吞吐约束（ncu 48.9%），池瘦身买不到驻留（grid 128 已 2 波）；**BV=8 过冲**（grid 256 ≈ 2.9 波 @2/SM + 再翻倍的 A 窗口流量）@T8192 实测 **0.80×**——甜点恰在 BV=16。XP 的收益也小于预估（1.03× vs 预估 0.05–0.1 ms 级）：B 扫重叠本已藏住了大部分 prologue await。装机 dhs 最终 **0.272 ms @T2048 / 1.047 @T8192 / 0.038 @T512(H4)**。

---

## 6.5 dkb：pre-update 部分和的并行重算

### 6.5.1 动机：为什么"拆"在数学上是无损的

逆时间行走沿途要为每块 n 产三个消费 pre-update dH（即 X_n）的部分和（dk 的 Bhat 项、dg 的前缀项与状态衰减常数）：

- `dkB[i,·] = (vn .* 2^(gcl−gc_i}) @ dH^T`（[C,K]，进 dk 的 `(vn·2^(gcl−gc)) @ dH^T` 项）；
- `rowB[i] = Σ_j dkB[i,j]·k[i,j]`（dg 的 Bhat 前缀项 `pre`）；
- `s5 = 2^gcl · Σ_{i∈K, j∈BV} dH[i,j]·h_all[i,j]`（dg5a 状态衰减常数）。

关键时序事实：**顺序行走在更新 dH 之前就把 pre-update 值存进了 dh_all[m]**——所以这三个量是 dh_all 的**逐块纯函数**，可以在行走结束后由独立 kernel 完全并行重算（每块一个 CTA），而不必串在延迟关键路径上。旧 dhu 把它们（vng staging + 2 个 K 切片 GEMM + rowsum + 256 部分和 s5 折叠）钉在行走内，每块多出三趟 GEMM 级段落与 ~17.5 KB smem。剥离后 dkb 还**顺带折掉 NV 归约**：dqkwg 原先每输出胞格 4 次全局载入的逐 vt 折叠，改为 dkb 在此走 v-切片并发射**块求和全局** `dkB_sum[M, C·K]` / `rowB_c[M, C]` / `s5_c[M, 1]`（各仅一个写者）——dqkwg 每胞格只读一个值。数值上刻意贴近原折叠序：per-vt GEMM 同划分、rowB 按 vt 主序、s5 同 256 部分/线程分布；唯一重结合是 dkB_sum 的 vt 链（(((v0+v1)+v2)+v3 对旧 ((((a+v0)+v1)+v2)+v3）——f32 末位 ulp 效应，远低于 4.5× dk 富余。

### 6.5.2 split-bf16 3-mma 与数值选择

dkB 站点保持标量会令 dkb 指令吞吐受限（每 MAC ~5–6 条指令）；tf32 化被第三轮逐点二分**否决**（dH 操作数 ~O(10)，tf32 dkB 使 dk 1.04e-2，**超 8e-3 杠**）。split-bf16 3-mma 通过：

```python
_split_bf16(e, vh, vl, vngb, C, BV)     # A = vng = vn .* 2^(gcl-gc) 的 hi/lo
_split_bf16(e, hh, hl, cur, K, BV)      # B = dH 切片的 hi/lo（cur = 本 vt 的 dHv）
for vt in range(NV):                    # vt 折叠进 mma 累加器（寄存器）
    maccs = mma_gemm(e, f"dkBhh{vt}", C, K, BV, 8, vh, hh, accs=...)
    maccs = mma_gemm(e, f"dkBhl{vt}", C, K, BV, 8, vh, hl, accs=maccs)
    maccs = mma_gemm(e, f"dkBlh{vt}", C, K, BV, 8, vl, hh, accs=maccs)
```

hi·hi + hi·lo + lo·hi（弃 lo·lo ~2^-32），~2^-16 乘积误差对 dk **不可见**（1.730e-3 vs 标量 1.731e-3）。split tile 带 +8 bf16 列（16 B/行）行垫——G5-F 的 LDSM 教训：8 行 ldmatrix 落在 8 个不同 bank 组。dHv 的 staging 用线程跨步 16 B 异步突发（逐元素 scf.for 每元素 ~7 条索引算术；dkb 是指令吞吐约束而非延迟约束）；**FPIPE 化**把 dHv 的 wait 移过 vngb emap——vnew 全局读 + gc 缩放这趟等体量全局扫成为 DMA 的覆盖（tile 不相交、dHv 只喂标量读）。

rowB 的一个被探针逮住的 bug（记录为设计教训）：第一版对**运行中累加器**逐 (vt, ks) 做 rowsum——`rowsum(v0·k) + rowsum((v0+v1)·k) + … = 4v0+3v1+2v2+v3`，~10 量级 dg 错误；修正为对**最终求和 tile 一次性** rowsum（同数学值、不同 f32 结合——rowB 喂 dg 前缀项有 ~50× 富余）。

### 6.5.3 s5：串行折 → 线程轴树折（GDN_DKB_S5T）

BV=16 使 NV 翻到 8 后，"线程 0 串行走完每 vt 全部 128 个 s5p 部分和"成为可见串行尾巴（NV×128 = 1024 次依赖 LDS+FADD/CTA）。S5T 重排布局：部分和落 `s5p2[threads, NV+1]`（每 vt 一列，**同一逐线程部分和折叠写**——分布与值恒同），线程轴以**跨步折半屏障树**归约（t < s：`s5p2[t][*] += s5p2[t+s][*]`，每级一道 barrier），随后线程 0 折 NV 列并**一次性**乘 `2^gcl`（逐 vt 的 exp2 对和式分配律）。s5v tile 退役；池 32288 → 36352 B（仍 2 CTAs/SM）、寄存器 100 → 112、0 spill。值类变化范围实测恰为预测锥：位级差异仅在 **s5_c + dg_state**（s5_c 的唯一消费者是 dqkwg 的 dg5a 项 → dg，杠 2e-2 对当前 3.274e-4 峰值，~60× 富余），33/35 数组位级一致，6/6 表格每一位等于既有记录。

### 6.5.4 2-CTA 转换：accCK 退役 + BV=16（GDN_DKB_2CTA）

**动机的精确化。** dkb 以 ~90 KB 池跑 1 CTA/SM（dHv [128,32] f32 + accCK [64,128] f32 + hh/hl [128,40] bf16 对——都是 K=128 尺寸的 tile）。纯别名审计结论：把 vh/vl 别名进 accCK 字节（lifetime 不相交——accCK 唯一写者是最终 maccs 存储扫）只能降到 ~79 KB，**仍是 1 CTA/SM，零收益**。真正的转换是**退役**而非别名：

1. **maccs 直停 dkB_sum 全局**（r14 park_epi 模式：同一寄存器值、同一 `[m, row·K+col]` 胞格，替代 accCK 中转 + store_rows 扫 + 2 道 barrier；**位级一致**——字节搬家）；rowB 的 rowsum 改 `gcol` 读 L2 热的 dkB_sum 全局（r9 dqkwg 的 P 退役模式；本 CTA 刚写、单 CTA 行——同值回读）。accCK 的 32768 B 退役；
2. **BV=16**：dHv/vngb f32 与 vh/vl/hh/hl split 对减半（[·, 24]，48 B 行跨距：12i mod 32 给 8 个不同 bank 组——+16 B 垫性质保持）。

池 89616 → **32288 B ≤ 50176**、寄存器 126 → **100**（Kred=16 链接减半活 fragment 集）——2×256×100 = 51200 寄存器容 65536 regfile 有余。**栅格门**（r9 派发规则）：`BH·NC ≥ 164` 才装机 2-CTA 臂——小网格（T512：M=32 CTA）在任何驻留度下都欠填充，保持 r17 形态 payload **逐字节不变**（未门控的 BV=16 臂在 T512 实测 **0.86×**：NV=8 的 s5 串行折 + 每 vt barrier 花约 4.5 µs 而无驻留可买——诚实否定，门即由此来）。

### 6.5.5 效果与两条否定

分臂实测（同窗口配对）：

| 臂 | 池 | 寄存器 | dkb @T2048 | dkb @T8192 |
|---|---|---|---|---|
| 基线（accCK + BV=32） | 89616 B（1/SM） | 126 / 0 spill | 0.2334–0.2371 | 0.8060–0.8136 |
| 仅 park（位级一致） | 56848 B（仍 1/SM） | 126 | 0.2376（0.99×） | — |
| + BV=16（装机，栅格门） | **32288 B（2/SM）** | **100** | **0.1907（1.22×）** | **0.6290（1.28×）** |
| + S5T 树折（装机） | 36352 B | 112 | **0.183（1.05–1.07×）** | **0.607（1.04×）** |

（dkb 拆分本身连同初版 split-bf16：0.96 → 0.25 ms，3.8×。）两条否定及其机制定论：

- **dHv ping-pong（GDN_DKB_DHP）**：静态奇偶 dHvA/dHvB，下一 vt 的突发在本 vt emap 后发射、飞在分裂 + mma 链 + s5 折之下（令牌绝不穿越 mma 内部；`wait_group 1` 保留最新组在飞；末 vt 全排空；r10 以跨 vt trailing barrier 的 2 次迭代距离满足）。寄存器过杆（127 pass1 / 110 smemfix，2-CTA 存活）、**位级一致**，但实测 **0.97×/0.99×**——r17 的 FPIPE emap **已经**覆盖了 dHv DMA，循环顶 await 排空的是近乎付清的组，而额外的发射/等待机械 + 8 KB 池更贵。**真正的串行结构是跨 vt 的 mma 累加器链（每 vt 3 条依赖链接、承值折叠序）+ 每 vt 一道 s5p2 列 barrier——那是值类重结合（vt 折是纯和），不是 staging 玩法。**
- **vt 折重结合（GDN_DKB_VTF）**：按上一定论建**偶/奇双累加器组**（偶 vt 链 maccs0、奇 vt 链 maccs1，park 扫时每 fragment 胞格做**一次固定序加法** even+odd）。取偶奇而非 vt 对半，是因为 warp 按序发射——对半的两条独立链在程序序上仍先后发射，偶奇才能链接级交错、把发射级关键路径减半（+32 活 fragment 寄存器两种取法相同）。寄存器 126（0 spill，2-CTA 存活；ptxas 交错复用只 +14）、池不变、6/6 与位差范围全按预测锥（dkB_sum → rowB_c → dk_local → dk → dg_state → dg）。实测 **0.99×/1.00×**——低于 1.05× 杆。根因：**跨 vt 链在 2 CTA/SM 下本就不是绑定的串行结构**（16 个驻留 warp 已在 warp 调度器层面重叠两条 vt 链）——这是"每 warp/CTA 工作不变 ⇒ 调度器 ILP 已在 ⇒ 无收益"定律的又一次独立确证。

装机 dkb 最终 **0.183 ms @T2048 / 0.607 @T8192 / 0.028 @T512(H4，门控保持旧形态)**。

---

## 6.6 扫描族的三个诚实否定：设计空间的边界测绘

以下三项不是失败罗列：每项都是**数学先行验证、机械随后落地、性能诚实测量**的完整闭环，共同给出扫描族设计空间的三条边界——"更强的并行化在哪里不再有收益"。负结果旋钮全部保留在树中（默认 0，关闭时逐字节再生装机 payload），供未来形状域变更时重新评估。

### 6.6.1 边界一：段切分（2-level segment scan）——数学精确、经济为负

二级分解（CPU 已实现并验证，`gdn_carry_validate.py::seg_walk`）：段 s 覆盖块 [s·G, (s+1)·G)；每段零种子局部走 Y、复合段算子 (A_seg = ∏A，B_seg = 局部底部输出)；串行边界链 $T_s = A_{seg,s+1} T_{s+1} + B_{seg,s+1}$（S 条，微小）；修正 $X_n = P_n T_{top(n)} + Y_n$（$P_n = A_{n+1}\cdots A_{hi-1}$ 后缀积）。fp64 下对每个 S ∈ {2,4,8,16} 都与真顺序式同在 **2–3 ulp**。

**不装机的决定是测量出的经济学**：段复合需要 [K,K]×[K,K] 矩阵积（每个 2M MAC）——标量实现即成新瓶颈；tf32/bf16 复合重开精度/混沌问题（f32 级 dh 扰动会重采样 mma 操作数量化箱——dbeta/dv 的混沌类）；再加 3 个 kernel 的发射开销，对冲一个 ~0.3 ms 的剩余天花板。而 dhs 经 rgemm + 窗口 + BV=16 后已至 0.28 ms：**并行深度的收益上限被串行步的延迟瘦身提前吃掉**。这是"测量负经济学，非数学失败"的边界：在超收缩分布下（§6.1.2）段修正一步即死，更强并行深度天然无物可赢。

### 6.6.2 边界二：fwdscan K-split 的 warp-group 形态（GDN_FKS_WG / v4）——数学 GO、性能 0.98–0.99×

**孪生数学（tools/gdn_fwdscan_ksplit.py）。** fwdscan 的行走 $S' = 2^{gcl}S + kg^T(u - wS)$ 对 **K 行对半** $R_0/R_1$ 精确可分：衰减逐元素（行分裂零耦合）；$kg^T vn$ 的输出行只需自家 kg 列（行分裂，vn 共享）；唯一跨 K 耦合是 $wS = \sum_h w[:,R_h]@S[R_h]$——交换 [C,BV] f32 部分和并**固定序加一次**。CPU 孪生（对 kernel 精确 k 序的严格左折模拟）判定 **GO**：fp64 恒等（h_all **逐位相等**、vnew 5.7e-17/5.4e-17 @NC=32/128）；f32 纯重结合 vnew ≤ 3.08e-8/2.89e-8 rel、**h_all 完全不动**（vn 的 ulp 差经 kg^T vn 注入 S，落在 S 自身舍入之下被舍掉）；误差放大 1.00×。

**GPU 端口决策。** 双 CTA 旗标形式的正确性类恰是 F1KS 险族（跨波边界的对等自旋死锁、跨发射旗标陈旧、普通数据载入的 PTX 内存模型栅栏）——设计期即否决，改**单 256 线程 CTA 内的 warp 组分裂**（warp 0–3 持 $R_0$、4–7 持 $R_1$）：交换走 smem（两个 acc tile 同 CTA 可见，`gpu.barrier` 即交接栅栏），各组并行走自己的 staging/gemm，资源包络与双 CTA 形态相同（256 线程 / 49152 rf 寄存器 / 每 SM 一链），零新同步面、无条件确定性。

**数学门（链上实测）。** h_all **位级一致（0/8388608 元素）**；vnew maxabs 2.384e-7（2 ulp）、rel 5.95e-8——孪生预测的同末位 ulp 类、按分布缩放；下游锥恰为预测（vnew_all + dkb + dqkwg + caryA 各数组，25/35 位级一致）；6/6 两位数全等；关闭旋钮双形状逐字节一致；det N=5 位级一致。

**性能定论。** 两个结构臂（融合 per-group staging 组 / 恢复 v3 A-B 覆盖节奏）配对 ×3：**0.98–0.99×**。根因（机制级）：warp 组重划分**不改变每 warp 的工作量**——m16n8 tile 粒度迫使每个 [C,BV] 半 GEMM 取 4-warp 铺瓦，每 warp 每块仍走 4 条原子 tile 链（v3：2 链 × 2 片；v4：4 链 × 1 半），而 8 个驻留 warp **本已**交错 v3 那些形式上依赖的片链（设计想要的 ILP 在 warp 调度器层面已在）；设计真正需要的驻留门（2 CTA/SM = 16 warp）仍结构性关闭：池 98560 B 对 50688 B 杆（[C,K] w/kg tile 与复制的 vn/acc 不随 K-split 减半；即便 BV=16 也 78080 B）。

### 6.6.3 边界三：fwdscan K-split 的 cluster 双 CTA 形态（GDN_FKS2 / ks2）——机械全证、性能 0.88–0.89×、族闭合

**可行性先行（推翻一个先例记录）。** sm_120 簇 >1 可用：`cuLaunchKernelEx` + `CLUSTER_DIMENSION` 接受 2/4/8 x 维（本 RTX 5090 Laptop、CC 12.0），`%cluster_nctarank` 读取核证明硬件真以簇模式执行（普通发射到处读 1/0；2x1x1 读 2 且成对 {0,1}；4x1x1 读 4 且组内 0–3）。先前"仅 1x1x1"的记录对本设备/驱动不成立——探针代码在树。

**设计。** 每 (bh, vt) 行走配一对 256 线程 CTA（`%ctaid.x` 奇偶定 rank r，持 K 行 $R_r$）：每块 p_r = w[:,R_r]@S[R_r] 停入 `@pex_all[walk, r]` + 对 `@flagx[walk, r]` 的 RELEASE 原子；对等旗标 ACQUIRE 自旋；`vn = u − (p_0 + p_1)` 双部分和**固定序**从全局读（两 CTA 同值）；vnew 双写同字节（良性类）；`S[R_r] *= 2^gcl`；`S[R_r] += kg[:,R_r]^T @ vn`（每 CTA 恰一个 BK 折叠）。三险全闭：(1) 前向进度——簇属性使对共驻成为**驱动保证**（核以 %ctaid.x 奇偶映射、不读簇 SRS，普通发射下只要对恰好共驻仍值正确）；(2) 跨发射陈旧——旗标目标绝对化，harness 经 manifest 的 `zero_globals` 契约在**每次发射前流序 memset**（`pex_all` 为 `scratch_globals`，写前读免 staging）；(3) 栅栏——探针验证 `memref.atomic_rmw addf` 降为 `atom.acq_rel.SYS.global.add.f32`（强于所需 .gpu 域）。

**资源与数学门。** 池 S[K/2,BV] 8192 + wpp[C,K/2] 16384（w 半 cp.async 暂存、kg 原地覆写）+ kpp 8192 + vn 8192 + acc 8192 + gr 512 + expc 256 = **49920 B ≤ 50688**；**89 寄存器**（杆 128）——**首个同时过双杆的 fwdscan 形态**：真 2 CTA/SM、16 warp/SM、T2048 全部 128 CTA 单波驻留。数学：6/6 双形状、每一位数等于记录表；**h_all 位级一致**（2 ulp 的 vnew 重结合被 S 更新乘积的 tf32 舍入吸收——vnew 18607/4194304 胞格移动）；det N=5 新起进程位级一致（旗标/自旋交换确定且免死锁）。

**性能定论与分解。** 配对 ×3：**0.88× @T2048 / 0.89× @T8192**。分解测量：v3 核在 grid 128 上挂 2x1x1 簇属性耗时 722.8 µs == 2×361.4——**簇属性本身免费**；ks2 的差值是 **+57 µs 的交换成本**（park 存 + SYS 域原子往返 + vn 扫的 3× 全局读）。而每 CTA 减半的工作**没买到任何可测收益**：若每块关键路径真是 K 并行 GEMM 链，ks2 应落在 ≈0.5×361+57 ≈ 237 µs，实测 424.9。**结论：扫描的串行结构是每块的 staging/barrier/交换延迟，不是 GEMM 宽度**——与 v4 从 warp 组侧得到的同一定律。

**族闭合。** v4（warp 组）、ks2（双 CTA 簇）加此项分解，fwdscan 的 K-split/驻留族以**三个独立 GPU 数据点**闭合；簇发射 harness 契约（cluster / zero_globals / scratch_globals 三个 manifest 字段）留树，供未来任何跨 CTA 设计直接使用。

### 6.6.4 三条边界的合并陈述

把三个否定并置，扫描族的收益边界呈一条一致的定律：

> **当每 warp / 每 CTA 的工作量不变时，调度器层面的 ILP 已经在场；串行扫描的剩余延迟由 staging、barrier 与（若有）交换的延迟构成，而非算术宽度。** 因此合法且有效的手段只有三类——把工作移出串行链（携带预复合、dkb 拆分）、压缩每步的暴露延迟（rgemm ILP、A 窗口、跨块预取）、以及纯列并行地增开 CTA（BV 切片）；任何"把同一工作铺到更多执行体"的形态（warp 组对半、双 CTA 对半、段深度）在本形状域内为中性或负收益。

---

## 6.7 小结：扫描族的最终形态

装机链（T2048 / T8192 / T512(H4)，同窗口配对阶段耗时，ms）：

| kernel | 职责一句话 | 关键机制 | T2048 | T8192 | T512(H4) |
|---|---|---|---|---|---|
| fwdscan | 顺向重算 h_all/vnew_all | prologue 累积提升 + 组 A/B ping-pong + tf32 双 GEMM + GSCG 混合 NC 免疫 | 0.376 | ~1.50 | 0.097 |
| caryA | A_all [K,K] | split-bf16 双链接 + Bhat 幂二折入 split + epilogue 直存；2 CTA/SM | 0.054 | — | 0.016 |
| caryM | M_all [K,C] | 双操作数 k16 切片 3 链接 + f32 精确初始化读-减 | 0.083 | — | — |
| bpre | B_all [K,V] | split×raw + 尾部单次缩放 + r10 复用栅栏 | 0.051 | — | — |
| dhs | 逆向携带链 | rgemm 寄存器累加 + A 滚动窗口 + iter_args 跨块令牌 + BV=16 | 0.272 | 1.047 | 0.038 |
| dkb | dkB/rowB/s5 并行重算 | split-bf16 3-mma vt 折叠 + accCK 退役 park + s5 树折；2 CTA/SM（栅格门） | 0.183 | 0.607 | 0.028 |

扫描族六 kernel 合计 ≈ 1.02 ms @T2048（占 bwd e2e ≈2.15–2.20 ms 的 ~46%）；对照重构前的旧形态（单一 dhu 2.17 ms + dkb 前身 0.96 ms，合计 3.13 ms，且 T8192 上 fwdscan 不可编译），族级累计约 **3.1×**（跨窗口量级参考）并把 T8192 从"不可编译"带入可建可验。数值与确定性：三形状 6/6 全过（最窄富余 dk 4.3× @T8192），全部阶段输出新起进程位级一致；两个纯延迟改动（dhs 窗口、dkb park）以位级等价证明收尾，三个值类改动（caryA/caryM 分裂、dkb BV=16/S5T）以预测锥位差审计收尾。三条否定边界（段切分、warp 组、簇双 CTA）与一条合并定律（§6.6.4）把"还能更快吗"的问题从猜测变成测绘。

> 证据索引：`tools/gdn_bwd_gen.py`（gen_fwdscan_v2/v3/v4/ks2、gen_caryA/caryM/bpre、gen_dhs、gen_dkb、rgemm/rgemm_aw、aggemm、astage_tile/agroup/await_all、_split_bf16 系列、main）；`tools/gdn_carry_validate.py`、`tools/gdn_fwdscan_ksplit.py`；`evidence/G5/bwd_report.md` 及 `evidence/G5/` 下 bwd*_verify/bench json、scratch 探针与配对日志；`tools/gdn_l1_check.py::chunked_bwd_ref`（两遍结构的 numpy 孪生）。
