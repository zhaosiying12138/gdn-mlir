# 2. L1 算法 IR 与数学：chunk-WY 分解的完整 MLIR 表达

> 本章覆盖范围：L0 语义规约（`docs/L0_SEMANTICS.md`）之下、GPU 调度层（L2 及以下各层）之上的**算法结构层 L1**。逐对象展开 chunk-WY 分解的全部数学（前向与反向），并给出每一个数学对象在真实 MLIR 中的对应物：生成器源码（`tools/gdn_l1_gen.py`，1428 行）、生成的 IR 产物样本（`evidence/G1/l1_chunked_T128_C64.mlir`、`evidence/G1/l1_bwd_chunked_T128_C64.mlir`）、CPU 下降与执行管线（`scripts/run_l1.sh`），以及全部数值证据（`evidence/G1/jax_spec_report.md`、`evidence/G1/l1_harness_report.md`、`evidence/G1/l1_check_output.txt`、`evidence/G1/l1_bwd_check_output.txt`、`evidence/G1/GATE_REPORT.md`）。本章所有数字均取自上述证据文件；正文不重复推导证据文件未支撑的结论。

## 2.0 章首：L1 的存在意义

编译一个算子之前，必须先让算子本身**可定义、可执行、可对拍**。GDN-MLIR 的分层栈中，L0 方言（`gdn.fwd` / `gdn.bwd` 两个 op 加 `#gdn.config<...>` 旋钮属性）只声明"意图"——它规定了输入输出形状与若干语义开关，但不包含任何算法结构；GPU 层（L2 及以下）关心 tile、流水与 warp 几何，但对"状态递推究竟等价于哪个三角线性方程组"保持沉默。夹在中间的 **L1 是算法结构层**：它把 Gated DeltaNet 的 chunked WY 分解——也就是 fla 生态中手写 Triton kernel 所实现的那套算法（`chunk_local_cumsum` → `prepare_wy_repr` 求 $(I+M)^{-1}$ → `chunk_o` 输出分解 → `chunk_delta_h` 状态更新；反向则经 `prepare_wy_repr_bwd`）——**原样表达为原生（stock）MLIR**：memref 级的 `linalg` + `scf`，不引入任何自定义方言操作，不 fork 编译器。因此 L1 与 FLA 的算法结构**同构**：每一个数学对象（含尾 cumsum、KKT Gram 矩阵、前代换求逆、WY 系数 $u/w$、状态递推、输出 intra/inter 分解，以及反向的全部六条梯度链）在 IR 里都有一个显式的、可指认的对应物。

L1 的三个不可替代的作用：

1. **可被 CPU 执行验证**。L1 模块是自包含的：全部输入以 `memref.global` 常数嵌入，经 stock `mlir-opt` 下降、`mlir-runner` 在 CPU 上 JIT 执行，通过 `printMemrefF32` 把计算结果打印到 stdout，与 fp64 JAX 规约（`tools/gdn_ref_jax.py`）和 fla naive torch oracle 三方对拍（`evidence/G1/GATE_REPORT.md`：前向 7/7 PASS、反向 3 npz × 6 梯度 PASS）。这使得"算法正确性"与"GPU 调度正确性"解耦：任何下层（L2–L4）的改动若破坏语义，可以回退到 L1 口径重新仲裁。
2. **与 FLA 同构的算法结构层，即"语义单一事实源"的物化形式**。`docs/L0_SEMANTICS.md` 记录的语义唯一事实源链为：`tools/gdn_ref_jax.py`（JAX fp64，44/44 自洽）⇄ fla naive（torch，1e-6/1e-7 量级相符）⇄ MLIR L1（`tools/gdn_l1_gen.py`，6e-7 量级相符）。L1 是这条链的终点：数学上它必须与 JAX 规约逐式等价，工程上它是第一个"真 IR"实现。
3. **它是未来方言展开 pass 的输出格式规约**。`docs/L0_SEMANTICS.md` 明确：`gdn-to-chunk` pass 的完整 chunk-AST 展开落地后，"输出与 `tools/gdn_l1_gen.py` 生成的 L1 文本**逐行同构**（前代换解三角、scf.for 载态已验证）"。也就是说，生成器不是一次性脚手架，而是 L1 展开 pass 的**参考输出定义**——生成器 IS 未来 pass 的输出格式（`tools/gdn_l1_gen.py:6-8` 的原话："This generator IS the output format of the future L1 expansion pass"）。

本章的写法遵循一个固定模式（"四件套"）：对每个 pattern，先给**动机**（算子层面为什么需要这个对象，配例子），再给**真实源码片段**（路径+行号标注，来自 `tools/` 下只读文件），然后给 **IR 对比**（数学/参考实现 vs 生成的 MLIR 产物），最后给**效果数字**（`evidence/G1/` 的对拍结果）。数学公式一律先定义后使用。

## 2.1 算子契约与记号

### 2.1.1 形状与输入输出契约

GDN 算子（fla op contract，`tools/gdn_ref_jax.py:11-19`、`evidence/G1/jax_spec_report.md` §3）：

| 张量 | 形状 | 语义 |
|---|---|---|
| $q$ | $[B,T,H,K]$ | 查询（模型侧已 L2 归一化） |
| $k$ | $[B,T,H,K]$ | 键（模型侧已 L2 归一化） |
| $v$ | $[B,T,H,V]$ | 值 |
| $\beta$ | $[B,T,H]$ | delta-rule 强度（**原始值**，sigmoid 由上游模型做） |
| $g$ | $[B,T,H]$ | 逐 token 对数衰减，**自然对数域** |
| $h_0$ | $[B,H,K,V]$ | 可选初始状态 |
| 返回 $o$ | $[B,T,H,V]$ | 输出 |
| 返回 $h_t$ | $[B,H,K,V]$ | 终态 |

$\beta$ 与 $g$ 都是逐 $(b,t,\text{head})$ 标量——算子内部从不混合头（H=HV in GDN）。测试数据生成遵循 GDN 模型风格（`tools/gdn_ref_jax.py::make_gdn_inputs`、`tools/gdn_l1_gen.py::make_data`）：$q,k$ 逐行 L2 归一化、$\beta=\mathrm{sigmoid}(x)$、$g=-\mathrm{softplus}(0.5x)$（负的自然对数域衰减）、$h_0\sim 0.1\,\mathcal N$。

### 2.1.2 逐 token 递推（语义骨架，先定义后用）

L1 的一切数学都从下列逐步递推出发（`tools/gdn_ref_jax.py:31-35`，顺序细节是 fla 契约的一部分，"do NOT fix these"）：

$$
h \;\leftarrow\; e^{g_t}\, h \qquad(\text{衰减在 delta 更新}\textbf{之前})
$$

$$
b_v \;\leftarrow\; \beta_t\,\bigl(v_t - k_t\, h\bigr), \qquad h \;\leftarrow\; h + k_t^{\!\top}\!\otimes\, b_v
$$

$$
o_t \;\leftarrow\; q_t\, h
$$

其中 $k_t\,h$ 是 $[K]\times[K,V]$ 的矩阵-向量作用（$\sum_j k_t[j]\,h[j,:]$），$k_t^{\top}\otimes b_v$ 是 $[K,V]$ 外积。**顺序不可交换**：若把 delta 更新放在衰减之前，状态会多乘一次 $e^{g_t}$；若把 $\beta$ 乘在 $v_t$ 而非残差 $(v_t-k_t h)$ 上，则与 fla 的 $M$ 矩阵行向 $\beta$ 约定冲突（见 §2.2.5）。这条递推的可执行定义是 `gdn_seq`（`jax.lax.scan` 移植，代码见 §2.5.1）。

### 2.1.3 记号总表与命名对照

本章统一采用**生成器约定**（`tools/gdn_l1_gen.py:14-40`）。注意 `tools/gdn_ref_jax.py` 对 $u/w$ 使用**相反的字母命名**（该文件 $w:=T(\beta v)\in\mathbb R^{C\times V}$、$u:=T(\beta k e^{cd})\in\mathbb R^{C\times K}$），数学完全一致，读代码时须对照下表：

| 记号（本章/生成器） | 定义 | `gdn_ref_jax.py` 记号 | fla 记号（生成器 docstring 标注） | 生成器缓冲区 |
|---|---|---|---|---|
| $C$ / $\mathrm{NC}$ | chunk 长度 / chunk 数 $=\lceil T/C\rceil$ | `BT` / `n` | — | 编译期常数 |
| $\mathrm{gc}$ | chunk 内 $g$ 的**含尾**（inclusive）累加和 | `decay` / `cd` | `chunk_local_cumsum` | `%gc` |
| $\mathrm{gcl}$ | $\mathrm{gc}$ 的末元素 $\mathrm{gc}_{C-1}$ | `d_last` | — | `%gcl`（标量） |
| $D$ | 衰减差矩阵 $D_{ij}=2^{\mathrm{gc}_i-\mathrm{gc}_j}$ | `L_mask`（tril 后） | — | 内联（每次现算） |
| $A$ | $A_{ij}=\beta_i (k_i\!\cdot\! k_j)\,2^{\mathrm{gc}_i-\mathrm{gc}_j}$，$j<i$，严格下三角 | $-M$（`attn` 取负前） | $-M$ | `%Abuf` |
| $M$ | fla 记号：$M_{ij}=-\beta_i (k_i\!\cdot\! k_j)\,e^{cd_i-cd_j}$ | `attn` 初值 | $M$ | — |
| $T$ | $(I+A)^{-1}=(I-M)^{-1}$，单位下三角 | `attn` 终值 | `attn` | `%Tbuf` |
| $u$ | $T(\beta\odot v)\in\mathbb R^{C\times V}$ | `w` | `k_cumsum` | `%ubuf`（前级 `%upre`） |
| $w$ | $T(\beta\odot k\odot 2^{\mathrm{gc}})\in\mathbb R^{C\times K}$ | `u` | `k_cumdecay` | `%wp`（前级 `%kb2`） |
| $v_{\text{new}}$ | $u-wS\in\mathbb R^{C\times V}$ | `v_new = w - u@S`（注意字母互换） | — | `%vn` |
| $P$ | $\mathrm{tril}_{\mathrm{incl}}\bigl((qk^{\top})\odot D\bigr)$，**对角保留** | `a_i` | — | 前向复用 `%Abuf` |
| $k_{\mathrm{term}}$ | $k\odot 2^{\mathrm{gcl}-\mathrm{gc}}\in\mathbb R^{C\times K}$ | `k_term` | — | `%kg` |
| $\hat B$ | $k\odot 2^{\mathrm{gcl}-\mathrm{gc}}$（反向 Bhat 链，与 $k_{\mathrm{term}}$ 同式） | `Bhat` | — | 内联进缩并 |
| $S$ | 状态 $[K,V]$ | `S` | `S`/`h` | `%S` / 反向 `%Sn`（`h_all[n]`） |

其中 $\odot$ 为逐元素乘（行向量按行广播），$\mathrm{tril}_{\mathrm{incl}}$/$\mathrm{strict\_tril}$ 分别为含对角/严格下三角掩蔽。

## 2.2 前向数学：从递推到 chunk

### 2.2.1 $q$ 的 L2 归一化约定与 $\mathrm{scale}=K^{-1/2}$

**动机（算子层面）**。GDN 模型在调用算子前对 $q,k$ 逐行 L2 归一化（把每行变成单位向量），这是**模型侧**职责；算子内部**不做**归一化（`docs/L0_SEMANTICS.md` 第 3 条：算子内部不做 q/k 的 L2 归一；config.normalize_q 仅当上游需要时打开，fla 语义默认关闭）。$\mathrm{scale}$ 约定为

$$
\mathrm{scale} = K^{-1/2}
$$

（`naive.py` 的 `1 / (q.shape[-1] ** 0.5)`，不是其他幂次），乘在（归一化后的）$q$ 上。归一化 + scale 的组合效果是：$q$ 行的内积被约束在 $[-1,1]\cdot K^{-1/2}$ 量级，使 $P$ 矩阵与输出 $o$ 的数值范围受控。证据文件给出的实例：G1-A 的 dump 中 $\max|{\cdot}|\approx 8.2$，fp32 存储舍入约 $1.4\times 10^{-8}$（`evidence/G1/jax_spec_report.md` §2）。

**scale 的折叠位置**。$o$ 关于 $q$ 线性，规约把 scale 预乘进 $q$（`gdn_seq`：`q = q * scale` 在递推之前；`gdn_chunk` 同）；L1 kernel 则把它折叠到**最后的输出存储**：

$$
o = (\,o_{\text{inter}} + o_{\text{intra}}\,)\cdot \mathrm{scale}
$$

（`tools/gdn_l1_gen.py:22-23`、生成的 IR 中 `%fscale = arith.constant 0.25 : f32` 当 $K=16$）。反向对称地在叶子处折叠 $\mathrm{dO}=\mathrm{do}\cdot\mathrm{scale}$（§2.3.2）。两种折叠位置数学等价，但 kernel 侧省掉一次 $[C,K]$ 逐元素乘。

**源码片段**（`tools/gdn_l1_gen.py:66-82`，数据侧归一化约定与生成器嵌入约定）：

```python
def make_data(B, T, H, K, V, seed):
    """Deterministic seeded fp32 test data, GDN-model-style ...:
    q,k l2-normalized; beta=sigmoid(x); g = -softplus(0.5x) NEGATIVE
    NATURAL-log decay; h0 [B,H,K,V]."""
    rng = np.random.default_rng(seed)
    def l2n(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    q = l2n(rng.standard_normal((B, T, H, K))).astype(np.float32)
    k = l2n(rng.standard_normal((B, T, H, K))).astype(np.float32)
    ...
    g = (-np.logaddexp(0.0, xg)).astype(np.float32)  # natural-log domain
```

**效果数字**。h0 路径被真实行使：G1-A 两个仅 $h_0$ 不同的 dump（`ref_jax_T128_C64.npz` vs `ref_jax_T128_C64_h0.npz`）的 $o$ 相差 $2.9\times 10^{-2}$，而 MLIR 对两者的对拍差同为 $6.407\times10^{-7}$（`evidence/G1/l1_check_output.txt` jax_C64 / jax_C64_h0 两行）。

### 2.2.2 $g$ 的数域约定：自然对数域输入、$\log_2$ 域执行、$\mathrm{RCP\_LN2}$

**动机（算子层面）**。fla 契约里 $g$ 是**自然对数域**逐 token 衰减，乘子为 $e^{g}$（e 底）。GPU/CPU kernel 惯用 $\exp_2$（硬件 exp2 指令/一次乘法尾数操作），因此 fla 的 Triton 路径在 kernel 前把 $g$ 预乘 $\mathrm{RCP\_LN2}=\log_2 e=1.4426950408889634$（`fla/ops/gated_delta_rule/chunk.py` 的 `chunk_local_cumsum(..., scale=RCP_LN2)`）。L1 生成器做同样的事（`tools/gdn_l1_gen.py:59,223-224`）：

$$
g_{\log_2} = g\cdot \log_2 e, \qquad \exp_2(g_{\log_2}) = e^{g}
$$

```python
RCP_LN2 = 1.4426950408889634  # log2(e): natural-log g -> log2-domain g
...
# embed g in the kernel-internal log2 domain: exp2(g*RCP_LN2) == exp(g)
g_log2 = (g.astype(np.float64) * RCP_LN2).astype(np.float32)
```

**IR 对比**。生成的模块头部把该约定写进注释（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:3-5`）：

```mlir
// conventions: embedded g is NATURAL-log g scaled by RCP_LN2 (log2
// domain, like fla's Triton path); gc inclusive chunk cumsum; A strictly
// lower; T=(I+A)^-1 forward substitution; scale=K^-1/2 folded into o.
```

IR 内所有衰减因子均为 `math.exp2`。这一约定在反向有一个非平凡推论——**dg 链不做 RCP_LN2 重标度**——是 G1-B2 的关键发现之一（§2.3.10）。

**效果数字**。若把数域搞错（例如 kernel 直接 $\exp_2(g)$ 而不预乘），128 步累计衰减将偏差 $2^{(1-\log_2 e)\sum g}$ 量级（指数级放大）；对拍链上任何数域错误都会让 `evidence/G1/l1_check_output.txt` 的 $10^{-7}$ 级差直接爆掉，7/7 PASS 是该约定的执行级证据。

### 2.2.3 状态递推的仿射性：chunk 化的数学引擎

**动机（算子层面）**。逐步递推是一个长度 $T$ 的串行扫描——GPU 上不可并行。chunk-WY 分解的核心观察是：**一个 chunk 内 $C$ 步递推对状态 $S$ 是一个仿射映射**，可以显式写成"一个 $[K,K]$ 线性部分 + 一个 $[K,V]$ 偏置部分"，代价是解一个 $C\times C$ 单位下三角方程组。这是从"扫描"到"矩阵乘"的转换枢纽，也是 L1 与 fla kernel 同构的原因。

**推导**（先定义后用；推导在自然对数域进行，kernel 以 $\exp_2$ 实现等价形式）。设 chunk 内时间下标 $i=0,\dots,C-1$，含尾累加 $\mathrm{gc}_i=\sum_{t\le i}g_t$。对逐步递推去递归化：第 $i$ 步读到的（衰减后）状态是

$$
h_i^{\text{dec}} = e^{g_i}S_{i-1} = e^{\mathrm{gc}_i}\,S_{\text{entry}} \;+\; \sum_{j<i}\bigl(k_j\, e^{\mathrm{gc}_i-\mathrm{gc}_j}\bigr)^{\!\top} b_j,
$$

其中 $b_j$ 是第 $j$ 步的 delta 修正量 $b_j=\beta_j\bigl(v_j-k_j h_j^{\text{dec}}\bigr)$。代入并按 $i$ 收集：

$$
b_i + \sum_{j<i} \underbrace{\beta_i\,(k_i\!\cdot\! k_j)\,e^{\mathrm{gc}_i-\mathrm{gc}_j}}_{A_{ij}} \, b_j
\;=\; \beta_i v_i - \beta_i\bigl(k_i\, e^{\mathrm{gc}_i}\bigr) S_{\text{entry}} .
$$

即 chunk 内全部 $b$ 向量满足一个单位下三角线性方程组 $(I+A)\,b = \beta\odot v - (\beta\odot k\odot e^{\mathrm{gc}})\,S_{\text{entry}}$，其中 $A$ 严格下三角（§2.2.5 定义）。于是

$$
v_{\text{new}} := b = T(\beta\odot v) - T(\beta\odot k\odot e^{\mathrm{gc}})\,S_{\text{entry}} = u - w\,S_{\text{entry}},
$$

$$
u := T(\beta\odot v)\in\mathbb R^{C\times V}, \qquad w := T(\beta\odot k\odot e^{\mathrm{gc}})\in\mathbb R^{C\times K}, \qquad T:=(I+A)^{-1}.
$$

chunk 出口状态（把 $\sum_j(k_j e^{\mathrm{gc}_{C-1}-\mathrm{gc}_j})^{\top}b_j$ 写成 $k_{\mathrm{term}}^{\top}v_{\text{new}}$，$k_{\mathrm{term}}:=k\odot e^{\mathrm{gcl}-\mathrm{gc}}$）：

$$
S_{\text{exit}} \;=\; 2^{\mathrm{gcl}}\,S_{\text{entry}} + k_{\mathrm{term}}^{\top}\bigl(u - w\,S_{\text{entry}}\bigr)
\;=\; \underbrace{\bigl(2^{\mathrm{gcl}}\,I_K - k_{\mathrm{term}}^{\top}w\bigr)}_{\Phi\;\in\;\mathbb R^{K\times K}}\,S_{\text{entry}} \;+\; \underbrace{k_{\mathrm{term}}^{\top}u}_{b\;\in\;\mathbb R^{K\times V}} .
$$

这正是**仿射递推** $S' = \Phi S + b$：线性部分 $\Phi=2^{\mathrm{gcl}}I-k_{\mathrm{term}}^{\top}w$ 是"单位阵减一个秩 $\le\min(K,C)$ 修正"，偏置部分 $b=k_{\mathrm{term}}^{\top}u$。三个直接推论：

- **输出可分解**：$o_i = q_i h_i^{\text{after}}$ 关于 $S_{\text{entry}}$ 仿射，故 $o = o_{\text{inter}}(S) + o_{\text{intra}}$（§2.2.9）；
- **跨 chunk 仍是同一仿射递推**：整个序列是 $\mathrm{NC}$ 次仿射复合（L1 顺序执行该复合；跨 chunk 并行扫描是后续层的话题）；
- **反向结构确定**：仿射映射的伴随给出 $\mathrm{dH}$ 反向递推（§2.3.3）。

**IR 对比**。生成器不显式物化 $\Phi$，而是按上式右端原样执行：`%dl = math.exp2 %gcl`（$2^{\mathrm{gcl}}$）、`%kg`（$k_{\mathrm{term}}$）、`%wp`（$w$）、`%ubuf`（$u$），状态更新见 §2.2.8 的 IR。numpy 孪生（`tools/gdn_l1_check.py::chunked_ref:86-87`）逐式对应：

```python
S = (S * np.exp(gc[-1])
     + (kc * np.exp(gc[-1] - gc)[:, None]).T @ vn).astype(f32)
```

**效果数字**。chunk 化推导本身的正确性由"chunk-vs-seq"列独立验证（不涉及 MLIR）：numpy chunked 参考对 sequential 移植的最大差 $1.192\times10^{-7}$–$5.662\times10^{-7}$（fp32，`evidence/G1/l1_check_output.txt` 末列）；fp64 下 JAX 规约内部 chunk-vs-seq 达 $5.6\times10^{-16}$（$o$）/$1.2\times10^{-15}$（$h_t$）（`evidence/G1/jax_spec_report.md` §1）。

### 2.2.4 chunk-local 含尾 cumsum $\mathrm{gc}$

**动机（算子层面）**。chunk 内所有衰减因子都以"累计衰减差"出现：$D_{ij}=2^{\mathrm{gc}_i-\mathrm{gc}_j}$（A 矩阵、P 矩阵）、$2^{\mathrm{gc}}$（$w$ 的 k 行、$o_{\text{inter}}$ 的 q 行）、$2^{\mathrm{gcl}}$（状态线性部分）、$2^{\mathrm{gcl}-\mathrm{gc}}$（$k_{\mathrm{term}}$）。$\mathrm{gc}$ 必须是**仅在 chunk 内部的含尾（inclusive）累加**：

$$
\mathrm{gc}_i = \sum_{t=0}^{i} g_t \quad (i=0,\dots,C-1), \qquad \mathrm{gc} \text{ 不跨 chunk}
$$

跨 chunk 的全局 cumsum 是经典错误：它会让 $D_{ij}$ 带上 chunk 前历史，使 $A$、$P$ 与状态更新整体错位（`evidence/G1/jax_spec_report.md` §3："Global (cross-chunk) cumsum would be wrong"）。跨 chunk 衰减只通过两个显式因子进入：状态上的 $2^{\mathrm{gcl}}$ 与 k 行上的 $2^{\mathrm{gcl}-\mathrm{gc}}$——即每个 chunk 把"整段历史衰减"折叠成对入口状态的一次乘法。

**源码片段**（`tools/gdn_l1_gen.py:420-431`；序列依赖决定了这是标量 `scf.for` 而非 `linalg.generic`）：

```python
# (a) inclusive chunk-local cumsum of g
e(f"%g0v = memref.load %g_c[%c0] : {t_bc}")
e(f"memref.store %g0v, %gc[%c0] : {t_C}")
e.block("scf.for %i = %c1 to %cC step %c1")
e("%ip = arith.subi %i, %c1 : index")
e(f"%prev = memref.load %gc[%ip] : {t_C}")
e(f"%cur = memref.load %g_c[%i] : {t_bc}")
e("%cs = arith.addf %prev, %cur : f32")
e(f"memref.store %cs, %gc[%i] : {t_C}")
e.end()
e(f"%gcl = memref.load %gc[%cCm1] : {t_C}")
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2514-2523`，逐 token 载入跨步 $2$ 的 strided 视图——见 §2.4.3——累加进稠密 scratch `%gc`）：

```mlir
%g0v = memref.load %g_c[%c0] : memref<64xf32, strided<[2], offset: ?>>
memref.store %g0v, %gc[%c0] : memref<64xf32>
scf.for %i = %c1 to %cC step %c1 {
  %ip = arith.subi %i, %c1 : index
  %prev = memref.load %gc[%ip] : memref<64xf32>
  %cur = memref.load %g_c[%i] : memref<64xf32, strided<[2], offset: ?>>
  %cs = arith.addf %prev, %cur : f32
  memref.store %cs, %gc[%i] : memref<64xf32>
}
%gcl = memref.load %gc[%cCm1] : memref<64xf32>
```

**效果数字**。含尾正确性与 chunk 边界一起由尾块用例检验：`tail_t96`（$T{=}96=64{+}32$，一个尾 chunk 补 32 个零行）$o$ 差 $5.662\times10^{-7}$、`tail_b2`（$B{=}2,T{=}100=64{+}36$）$3.099\times10^{-6}$（`evidence/G1/l1_check_output.txt`）——补零行的 $\beta=k=g=0$ 使其成为 A/u/S 的**精确无操作行**（§2.3.11），任何 cumsum 边界错误都会在这些用例上显形。

### 2.2.5 KKT Gram 矩阵与 $A$ 的构造

**动机（算子层面）**。§2.2.3 推导中的耦合矩阵

$$
A_{ij} = \beta_i\,\bigl(k_i\!\cdot\! k_j\bigr)\,2^{\mathrm{gc}_i-\mathrm{gc}_j} \quad (j<i), \qquad A_{ij}=0 \;(j\ge i)
$$

由三因子复合：Gram 矩阵 $\mathrm{kkt}=kk^{\top}$（生成器缓冲区名 `%kkt`，即"KKT"之名的直接来源）、行向 $\beta$（**不对称**：$\beta_i$ 只乘行 $i$，$\beta$ 对 $v$ 则是全局缩放——这是 fla 契约第 4 条，`docs/L0_SEMANTICS.md`）、衰减差 $D_{ij}$。生成器记号 $A=-M$（$M$ 为 naive.py/G1-A 报告的严格下三角矩阵 $M_{ij}=-\beta_i(k_i\cdot k_j)e^{cd_i-cd_j}$），从而 $I+A=I-M$，两种记号给出同一个 $T$。例：$C=3$、$k$ 已 L2 归一化时 $|k_i\cdot k_j|\le 1$，$A_{ij}$ 的量级由 $\beta_i\in(0,1)$ 与 $2^{\mathrm{gc}_i-\mathrm{gc}_j}\le 1$（$g\le 0$ 使 $\mathrm{gc}$ 单调不增，故 $D_{ij}\le 1$）共同压制——这是 delta rule 数值稳定的结构性来源。

**源码片段**（`tools/gdn_l1_gen.py:433-473`；两步：缩并 + 掩蔽逐元素）：

```python
# (b) kkt = k_c @ k_c^T ; A = strict-tril(beta_i * kkt * 2^(gc_i-gc_j))
linalg.fill -> %kkt
linalg.generic  indexing_maps = [(i,j,r)->(i,r), (i,j,r)->(j,r), (i,j,r)->(i,j)]
                iterator_types = [parallel, parallel, reduction]
                ins(%k_c, %k_c) outs(%kkt):  %p = a*b ; acc+=%p
linalg.generic  indexing_maps = [(i,j)->(i,j), (i,j)->(i), (i,j)->(j), (i,j)->(i), (i,j)->(i,j)]
                ins(%kkt, %gc, %gc, %beta_c) outs(%Abuf):
                  %d0 = gc_i - gc_j ; %e0 = math.exp2 %d0
                  %m0 = kkt_ij * beta_i ; %m1 = %m0 * %e0
                  %i0/%j0 = linalg.index 0/1
                  %lt = arith.cmpi slt %j0, %i0
                  %r0 = arith.select %lt, %m1, %f0
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2524-2558`，第二个 generic 逐字）：

```mlir
linalg.fill ins(%f0 : f32) outs(%kkt : memref<64x64xf32>)
linalg.generic {
  indexing_maps = [ affine_map<(i, j, r) -> (i, r)>, affine_map<(i, j, r) -> (j, r)>,
                    affine_map<(i, j, r) -> (i, j)> ],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%k_c, %k_c : ...) outs(%kkt : memref<64x64xf32>) {
^bb0(%a0: f32, %a1: f32, %o0: f32):
  %p = arith.mulf %a0, %a1 : f32
  %s2 = arith.addf %o0, %p : f32
  linalg.yield %s2 : f32
}
linalg.generic {
  indexing_maps = [ affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>,
                    affine_map<(i, j) -> (j)>, affine_map<(i, j) -> (i)>,
                    affine_map<(i, j) -> (i, j)> ],
  iterator_types = ["parallel", "parallel"]
} ins(%kkt, %gc, %gc, %beta_c : ...) outs(%Abuf : memref<64x64xf32>) {
^bb0(%a0: f32, %a1: f32, %a2: f32, %a3: f32, %o0: f32):
  %d0 = arith.subf %a1, %a2 : f32
  %e0 = math.exp2 %d0 : f32
  %m0 = arith.mulf %a0, %a3 : f32
  %m1 = arith.mulf %m0, %e0 : f32
  %i0 = linalg.index 0 : index
  %j0 = linalg.index 1 : index
  %lt = arith.cmpi slt, %j0, %i0 : index
  %r0 = arith.select %lt, %m1, %f0 : f32
  linalg.yield %r0 : f32
}
```

掩蔽的 IR 手法值得注意：三角掩码不是数据（没有物化 mask 矩阵），而是**控制计算**——`linalg.index` 取回迭代坐标，`arith.cmpi slt` + `arith.select` 在寄存器级完成 $\mathbb 1[j<i]$ 乘法。这一个手法在本章反复出现（A、P、dPm、G、dAf 的全部三角掩蔽）。

**效果数字**。$A$ 的正确性（连同 $T$）被整条链覆盖：MLIR vs 嵌入 numpy chunked 参考（同一 $A$ 构造）$5.066\times10^{-7}$–$6.407\times10^{-7}$（`evidence/G1/l1_check_output.txt` diff_o 列），numpy chunked vs sequential（独立算法路径）$1.192\times10^{-7}$–$5.662\times10^{-7}$（末列）。

### 2.2.6 前代换求逆 $T=(I+A)^{-1}$：MLIR 三重循环

**动机（算子层面）**。$u=T(\beta v)$、$w=T(\beta k 2^{\mathrm{gc}})$ 都需要 $(I+A)^{-1}$ 作用到右端。$A$ 严格下三角 $\Rightarrow$ $I+A$ 单位下三角 $\Rightarrow$ 求逆/求解是**前代换（forward substitution）**，$O(C^3/6)$ 次标量乘加，无主元选取、无除法（对角元恒 1）。L1 的选择是把这个求解**完全显式地**写成三重 `scf.for`——不调用任何求解库、不引入黑盒 op——因为 L1 的职责是把算法结构钉死为可审计的 IR；行递推、Neumann 级数、分块求解等变体留给后续层作为可 A/B 的旋钮（`evidence/G1/l1_harness_report.md` "Known limitations"：`solve variant` 旋钮）。

**数学定义**（与 naive.py 的 attn 行递推逐式对应）：

$$
T_{i,i}=1;\qquad T_{i,j} = -\sum_{k=j}^{i-1} A_{i,k}\,T_{k,j} \quad (j<i)
$$

（`tools/gdn_l1_gen.py:27-29`：与 naive.py 的 `attn` 循环是相同的行递推；naive.py 的等价形式是就地行更新 $\text{attn}[i,:i] \mathrel{+}= \sum_k \text{attn}[i,k]\,\text{attn}[k,:i]$ 再加单位阵，见 §2.5.1 代码。）

**源码片段**（`tools/gdn_l1_gen.py:475-493`，生成器发射逻辑；注意最内层是携带 `iter_args` 累加器的 `scf.for`——MLIR 的"归约循环"原语）：

```python
# (c) T = (I+A)^-1 forward substitution
e(f"linalg.fill ins(%f0 : f32) outs(%Tbuf : {t_CC})")
e.block("scf.for %i2 = %c0 to %cC step %c1")            # 行 i
e(f"memref.store %f1, %Tbuf[%i2, %i2] : {t_CC}")         # T[i,i]=1
e.block("scf.for %j2 = %c0 to %i2 step %c1")             # 列 j<i
e(f"%sum = scf.for %k2 = %j2 to %i2 step %c1 "
  f"iter_args(%acc2 = %f0) -> (f32) {{")                 # 归约 k∈[j,i)
e(f"%aik = memref.load %Abuf[%i2, %k2] : {t_CC}")
e(f"%tkj = memref.load %Tbuf[%k2, %j2] : {t_CC}")
e("%pk = arith.mulf %aik, %tkj : f32")
e("%ak = arith.addf %acc2, %pk : f32")
e("scf.yield %ak : f32")
...
e("%neg = arith.negf %sum : f32")
e(f"memref.store %neg, %Tbuf[%i2, %j2] : {t_CC}")
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2559-2573`，真实生成文本，前向样本 `l1_chunked_T128_C64.mlir:1960` 起同构）：

```mlir
linalg.fill ins(%f0 : f32) outs(%Tbuf : memref<64x64xf32>)
scf.for %i2 = %c0 to %cC step %c1 {
  memref.store %f1, %Tbuf[%i2, %i2] : memref<64x64xf32>
  scf.for %j2 = %c0 to %i2 step %c1 {
    %sum = scf.for %k2 = %j2 to %i2 step %c1 iter_args(%acc2 = %f0) -> (f32) {
      %aik = memref.load %Abuf[%i2, %k2] : memref<64x64xf32>
      %tkj = memref.load %Tbuf[%k2, %j2] : memref<64x64xf32>
      %pk = arith.mulf %aik, %tkj : f32
      %ak = arith.addf %acc2, %pk : f32
      scf.yield %ak : f32
    }
    %neg = arith.negf %sum : f32
    memref.store %neg, %Tbuf[%i2, %j2] : memref<64x64xf32>
  }
}
```

**前后对比**（同一数学的三个实现形态）：

| 形态 | 位置 | 形式 |
|---|---|---|
| JAX 规约（行递推，向量化） | `tools/gdn_ref_jax.py:196-200` | `for i in range(1,c): attn = attn.at[..., i, :i].add(jnp.sum(attn[..., i, :i, None] * attn[..., :i, :i], axis=-2))`；随后 `attn += I` |
| numpy 孪生（标量三重循环） | `tools/gdn_l1_check.py:71-78` | `for i: Tm[i,i]=1; for j<i: s=0; for kk in range(j,i): s += A[i,kk]*Tm[kk,j]; Tm[i,j]=-s` |
| L1 MLIR（三重 `scf.for`） | 上文 IR | 最内层归约以 `iter_args` 携带累加器，逐元素 `memref.load/store` |

三者是同一行递推：JAX 形式按行向量化（一次 `add` 完成整行），MLIR 形式把内积降为标量循环——后者牺牲向量换来两点：(i) 完全显式、可逐行审计；(ii) 对任意 $C$ 参数化生成（生成器只依赖编译期常数 `%cC`），不依赖库。

**效果数字**。前代换路径连同 $A$ 构造覆盖于全部 7 个前向用例（`evidence/G1/l1_check_output.txt`，gate $<10^{-3}$，实测最大 $3.099\times10^{-6}$）；`evidence/G1/l1_harness_report.md` 记录该求解为 $O(C^3/6)$ 标量迭代（$C{=}64$ 时约 43k 次/chunk），base 模块 `mlir-opt` 0.32 s / `mlir-runner` 0.22 s 完成 JIT 与执行。

### 2.2.7 $u/w$ 的计算与 $v_{\text{new}}$

**动机（算子层面）**。$u,w$ 是 WY 表示的两块"系数"：$u$ 携带值信息（$T$ 作用于 $\beta\odot v$），$w$ 携带键信息（$T$ 作用于 $\beta\odot k\odot 2^{\mathrm{gc}}$，fla 名 `k_cumdecay`）；二者与入口状态组合出 $v_{\text{new}}$——chunk 内全部 delta 修正量的解。定义（生成器约定）：

$$
\mathrm{upre} = \beta\odot v, \qquad u = T\,\mathrm{upre}, \qquad \mathrm{kb2} = \beta\odot k\odot 2^{\mathrm{gc}}, \qquad w = T\,\mathrm{kb2}, \qquad v_{\text{new}} = u - w\,S .
$$

**源码片段**（`tools/gdn_l1_gen.py:495-550`）：两个逐元素 generic（`%upre`、`%kb2`）+ 两个命名缩并 + 一个残差 generic：

```python
e(f"linalg.fill ins(%f0 : f32) outs(%ubuf : {t_CV})")
e(f"linalg.matmul ins(%Tbuf, %upre : {t_CC}, {t_CV}) outs(%ubuf : {t_CV})")
...
e(f"linalg.fill ins(%f0 : f32) outs(%wp : {t_CK})")
e(f"linalg.matmul ins(%Tbuf, %kb2 : {t_CC}, {t_CK}) outs(%wp : {t_CK})")
# (e) v_new = u - w @ S
e(f"linalg.fill ins(%f0 : f32) outs(%vp : {t_CV})")
e(f"linalg.matmul ins(%wp, %S : {t_CK}, {t_KV}) outs(%vp : {t_CV})")
generic(... ins(%ubuf, %vp) outs(%vn): %r3 = subf %a0, %a1 ...)
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2586-2604,2605-2618`）：

```mlir
linalg.fill ins(%f0 : f32) outs(%ubuf : memref<64x16xf32>)
linalg.matmul ins(%Tbuf, %upre : memref<64x64xf32>, memref<64x16xf32>) outs(%ubuf : memref<64x16xf32>)
...
linalg.fill ins(%f0 : f32) outs(%wp : memref<64x16xf32>)
linalg.matmul ins(%Tbuf, %kb2 : memref<64x64xf32>, memref<64x16xf32>) outs(%wp : memref<64x16xf32>)
linalg.fill ins(%f0 : f32) outs(%vp : memref<64x16xf32>)
linalg.matmul ins(%wp, %S : memref<64x16xf32>, memref<16x16xf32>) outs(%vp : memref<64x16xf32>)
linalg.generic { ... } ins(%ubuf, %vp : ...) outs(%vn : memref<64x16xf32>) {
^bb0(%a0: f32, %a1: f32, %o0: f32):
  %r3 = arith.subf %a0, %a1 : f32
  linalg.yield %r3 : f32
}
```

注意两个生成器代码模式：(i) 命名 op `linalg.matmul` 是**累加语义**（写入 outs 前 accumulate），因此每个缩并缓冲先 `linalg.fill` 清零——这是生成器的固定"fill-before-accumulate"模式；(ii) 反向 pass 复用同一组缓冲名（`%upre`/`%kb2` 在 G 链中再次出现，§2.3.4）。

**效果数字**。`evidence/G1/l1_harness_report.md` 前向表全部 7 行 PASS；其中对 G1-A dump 的三行（jax_C64_h0/jax_C64/jax_C32）同时验证了 $u/w$ 路径与 JAX 的字母互换约定（jax 的 `w`=$u$、`u`=$w$）没有接错——接错将产生 $O(1)$ 级错误而非 $10^{-7}$。

### 2.2.8 状态更新与 `h_all` 递推

**动机（算子层面）**。chunk 出口的仿射状态更新（§2.2.3）：

$$
S \;\leftarrow\; 2^{\mathrm{gcl}}\odot S \;+\; \bigl(k\odot 2^{\mathrm{gcl}-\mathrm{gc}}\bigr)^{\!\top} v_{\text{new}} \;=\; 2^{\mathrm{gcl}} S + k_{\mathrm{term}}^{\top}v_{\text{new}},
$$

右侧的 $S$ 是**更新前**的入口状态（对应"衰减先于 delta 更新"的逐步顺序）；$k_{\mathrm{term}}^{\top}v_{\text{new}}$ 是 $[K,C]\times[C,V]$ 缩并。前向 kernel 中 $S$ 是跨 chunk 循环携带的活跃状态；反向 pass 1 则把**每个 chunk 的入口状态**存入 `h_all[n]`（$[\mathrm{NC},K,V]$ scratch，`tools/gdn_l1_gen.py:892` `t_HALL = mrt((NC, K, V))`）——`h_all` 递推即"沿 chunk 前进时逐片保存 $S_{\text{entry}}(n)$"。

**源码片段**（`tools/gdn_l1_gen.py:629-677` 前向状态更新；转置不物化，折进 indexing_maps）：

```python
# S = S .* 2^gcl + (k .* 2.^(gcl-gc))^T @ v_new
e("%dl = math.exp2 %gcl : f32")
generic(... ins(%k_c, %gc) outs(%kg):      # kg = k_term
        %dg = gcl - gc_i ; %eg = math.exp2 %dg ; %r6 = k*eg)
e(f"linalg.fill ins(%f0 : f32) outs(%prod : {t_KV})")
generic(  maps = ["(kk, vv, i) -> (i, kk)", "(kk, vv, i) -> (i, vv)",
                 "(kk, vv, i) -> (kk, vv)"],
         kinds = ["parallel", "parallel", "reduction"],
         ins(%kg, %vn) outs(%prod):  acc += kg[i,kk]*vn[i,vv])
generic(... ins(%S, %prod) outs(%S):  %ms = S*dl ; %rs = ms + prod)
```

注意缩并的转置实现：`(kk,vv,i) -> (i,kk)` 把输入的第一维（chunk 行）映射为归约维——`k_term`^T @ `v_new` 无需显式转置缓冲。终态写出：`%hth = memref.subview %ht[%b, %h, ...]` + `linalg.copy ins(%S) outs(%hth)`（`tools/gdn_l1_gen.py:680-682`）。

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:951-953,961`，反向 pass 1 的 h_all 保存与状态推进）：

```mlir
%h_all_n = memref.subview %h_all[%n, 0, 0] [1, 16, 16] [1, 1, 1] : memref<2x16x16xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
linalg.copy ins(%S : memref<16x16xf32>) outs(%h_all_n : memref<16x16xf32, strided<[16, 1], offset: ?>>)
...（重算 gc/A/T/u/w，推进 S）...
%dl = math.exp2 %gcl : f32
```

**效果数字**。状态路径由 $h_t$ 列独立验证：全部用例 $\max|h_t-\text{ref}|$ 为 $5.513\times10^{-7}$–$3.099\times10^{-6}$（`evidence/G1/l1_check_output.txt`）；$h_0$ 是否真实传播亦被 h0/no-h0 dump 对比证实（$o$ 差 $2.9\times10^{-2}$；$h_t$ 在两 dump 中相同属预期——128 步衰减把 $h_0$ 贡献压到 fp32 分辨率之下，`evidence/G1/l1_harness_report.md` 明确记录"expected, not a bug"）。

### 2.2.9 输出的 intra/inter 分解

**动机（算子层面）**。$o_i$ 读的是第 $i$ 步 delta 更新**之后**的状态，拆开即两项：

$$
o_i = \underbrace{\bigl(q_i\,2^{\mathrm{gc}_i}\bigr) S_{\text{entry}}}_{o_{\text{inter},i}\;(\text{跨 chunk 记忆})} \;+\; \sum_{j\le i}\underbrace{\bigl(q_i\!\cdot\! k_j\bigr)\,2^{\mathrm{gc}_i-\mathrm{gc}_j}}_{P_{ij}}\, v_{\text{new},j} \;=\; o_{\text{inter},i} + \bigl(P\,v_{\text{new}}\bigr)_i ,
$$

$$
P = \mathrm{tril}_{\text{incl}}\bigl((q\,k^{\top})\odot D\bigr) \quad(\textbf{对角保留，只掩严格上三角}), \qquad o = (o_{\text{inter}} + o_{\text{intra}})\cdot\mathrm{scale}.
$$

两处易错的对偶必须强调：$A$ 是**严格**下三角（chunk 内因果耦合只存在于 $j<i$），而 $P$ **含对角**（第 $i$ 步输出看到自己的更新 $b_i$）；$o_{\text{inter}}$ 与 $w$/`k_term` 共享 $2^{\mathrm{gc}}$ 型行因子，但作用对象是 $q$ 行。$\mathrm{scale}$ 折叠在最终存储（§2.2.1）。

**源码片段**（`tools/gdn_l1_gen.py:552-627`：`%qg = q⊙2^gc` → `oob = qg@S`；`%qk=qk^T` 缩并 → 复用 `%Abuf` 存 $P$（此时 A 已死）→ `oib = P@vn`；最终 `o_c = (oob+oib)*scale` 存入 strided 输出视图）：

```python
# o_inter = (q .* 2.^gc) @ S
generic(... ins(%q_c, %gc) outs(%qg):  %e3 = exp2(gc_i) ; %r4 = q*e3)
linalg.matmul ins(%qg, %S) outs(%oob)
# M = tril_inc((q k^T) .* 2^(gc_i-gc_j)) ; reuse Abuf; o_intra = M @ v_new
...  # 缩并 qk^T，再 exp2 差 + sle 掩蔽（注意 sle = 含对角，与 A 的 slt 相对）
linalg.matmul ins(%Abuf, %vn) outs(%oib)
# o_c = (o_inter + o_intra) * scale
generic(... ins(%oob, %oib) outs(%o_c):  %ad = addf ; %sc = ad * %fscale)
```

**IR 对比**。掩蔽方向的前后对照是本 pattern 的关键细节：A 用 `arith.cmpi slt`（$j<i$，严格），P 用 `arith.cmpi sle`（$j\le i$，含对角）——同一 `linalg.index` 手法、一个比较谓词之差（`tools/gdn_l1_gen.py:469` vs `:604`）。

**效果数字**。含对角正确性有独立证据：若误用严格掩蔽，对角项 $q_i\!\cdot\! k_i\,b_i$（量级 $O(\beta)$）丢失，误差远超 $10^{-3}$ gate；实测 jax_C64 两行 $6.407\times10^{-7}$（`evidence/G1/l1_check_output.txt`）。JAX 规约同式对照（`tools/gdn_ref_jax.py:220-226`：`a_i` 用 `triu_strict` 掩蔽即保留对角）。

### 2.2.10 前向小结：一个 chunk 的完整数据流

把 §2.2.4–§2.2.9 串起来，每 chunk（每 $(b,h)$）的 L1 数据流为：

$$
g_c \xrightarrow{\text{cumsum}} \mathrm{gc},\,\mathrm{gcl} \;\rightarrow\; \mathrm{kkt}=kk^{\top} \;\rightarrow\; A=\mathrm{strict\_tril}(\beta\,\mathrm{kkt}\odot D) \;\rightarrow\; T=(I+A)^{-1} \;\rightarrow\; \{u=T(\beta v),\; w=T(\beta k 2^{\mathrm{gc}})\}
$$

$$
\rightarrow\; v_{\text{new}}=u-wS \;\rightarrow\; \begin{cases} o = \bigl((q\odot 2^{\mathrm{gc}})S + P\,v_{\text{new}}\bigr)\cdot\mathrm{scale} \\[2pt] S \leftarrow 2^{\mathrm{gcl}}S + k_{\mathrm{term}}^{\top}v_{\text{new}} \end{cases}
$$

前向 kernel 使用 17 个 scratch 缓冲（`tools/gdn_l1_gen.py:379-396` 的 allocs：`S/Abuf/Tbuf/kkt/qk/gc/upre/ubuf/kb2/wp/vp/vn/qg/kg/oib/oob/prod`），循环结构为 `b → h → chunk` 三层 `scf.for`（`tools/gdn_l1_gen.py:375-411`；`evidence/G1/l1_chunked_T128_C64.mlir:1877-1905` 的真实 IR 同构）。代表性前向模块（B1 T128 H2 K16 V16 C64，seed 0）共 2167 行（`evidence/G1/l1_harness_report.md` Deliverables 表）。
## 2.3 反向数学：重算、dH 递推与 dA 双侧掩蔽公式

反向 kernel `@gdn_bwd(q,k,v,beta,g,h0,do,dht) -> (dq,dk,dv,dbeta,dg,dh0)` 以同一 memref+linalg+scf 风格生成（`tools/gdn_l1_gen.py:828-1384`），代码量约为前向的 1.6 倍（3540 行 vs 2167 行，同为 B1 T128 C64 seed 0，`evidence/G1/l1_harness_report.md`）。本节按"结构 → 链条 → 组装"展开，全部公式取自生成器 docstring（`tools/gdn_l1_gen.py:830-852`）、`evidence/G1/l1_harness_report.md` 的 Backward (G1-B2) 一节、以及 numpy 孪生 `tools/gdn_l1_check.py::chunked_bwd_ref`。

### 2.3.1 重算（recompute）策略与两遍结构

**动机（算子层面）**。反向需要每个 chunk 的前向中间量（$\mathrm{gc},A,T,u,w,v_{\text{new}},P$）与入口状态 $S_n$。两种极端是"全部重算"（算两次前向，零额外存储）与"全部物化"（pass 1 存所有中间量）。L1 选择**重算**并把存储压到最小：**只存每 chunk 的入口状态** `h_all[n]`（$[\mathrm{NC},K,V]$）。

**结构**（`tools/gdn_l1_gen.py:941-984`；`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2504,2662` 的两段注释逐字对应）：

- **Pass 1（前向重算）**：精确执行前向 chunk 数学，但**不计算 $o$**，只把每个 chunk 的入口状态 `linalg.copy` 进 `%h_all[%n]`；中间量（gc/A/T/u/w）即弃。
- **Pass 2（反向 chunk 行走）**：`scf.for %n2 = 0..NC`，以**正步长 + 下标重映射**实现逆序：

```mlir
// ---- pass 2: reverse chunk walk (dH seeded from dht) ----
scf.for %n2 = %c0 to %cNC step %c1 {
  %rn = arith.subi %cNC, %c1 : index
  %n = arith.subi %rn, %n2 : index          // n = NC-1-n2
  %off = arith.muli %n, %cC : index
  ...
  %Sn = memref.subview %h_all[%n, 0, 0] [1, 16, 16] ...   // 入口状态
```

（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2662-2678`。）负步长 `scf.for` 被刻意规避（`evidence/G1/l1_harness_report.md`：逆向 chunk 循环、dg 的 suffix/prefix 循环全部用正步长 + 索引重映射）。每个 chunk 在 pass 2 内**重算** gc/A/T/u/w（`_bwd_fwd_core`，`tools/gdn_l1_gen.py:746-825`，与前向逐式相同），再读 $S_n$=`h_all[n]` 组出 $v_{\text{new}}=u-wS_n$。

**重算旋钮**。`evidence/G1/l1_harness_report.md` "Notes for G2"：反向 kernel 每 $(b,h)$ 分配约 37 个 scratch（前向 17；生成器 allocs 清单在 `tools/gdn_l1_gen.py:899-912` 逐项列出），且 pass 2 重算 gc/A/T/u/w——"materialize 变体（pass 1 存 u,w,P）是 L2 显然的存储/计算折中 A/B 旋钮"。

**效果数字**。重算路径的全梯度对拍：3 npz × 6 梯度全部 PASS（gate $<2\times10^{-4}$ vs fp32 参照 / $<3\times10^{-4}$ vs fp64 参照，实测最差 $6.4\times10^{-6}$，`evidence/G1/l1_bwd_check_output.txt`）；孪生自身对 JAX vjp 的 fp64 差 $\le 1.4\times10^{-14}$。

### 2.3.2 叶子折叠 $\mathrm{dO}$ 与 $v_{\text{new}}$ 的总余切 $\mathrm{dvp}$

**动机**。$o=(o_{\text{inter}}+o_{\text{intra}})\cdot\mathrm{scale}$，故余切在叶子处折叠一次即可（与前向 scale 折叠互为镜像）：

$$
\mathrm{dO} = \mathrm{do}\cdot\mathrm{scale} \quad\in\mathbb R^{C\times V}.
$$

$v_{\text{new}}$ 从两条链收到余切：输出链（经 $P$）与状态链（经 $k_{\mathrm{term}}$，即 Bhat 链）：

$$
\mathrm{dvp} = P^{\top}\mathrm{dO} + \hat B\,\mathrm{dH}, \qquad \hat B := k\odot 2^{\mathrm{gcl}-\mathrm{gc}}\in\mathbb R^{C\times K}.
$$

**源码片段**（`tools/gdn_l1_gen.py:1017-1026`：dO/dOgc；`:1028-1045`：P（含对角掩蔽 `sle`，复用 `%Abuf`——此时 A 已死）；`:1066-1081`：dvp 两缩并，$\hat B$ **内联**进第二个缩并（`%bd = gcl - gc_i; %be = exp2; %bm = k*be; %bp = bm*dH_kv; acc+=`）——Bhat 不物化缓冲）：

```python
# dvp = P^T dO + Bhat dH   (Bhat = k .* 2^(gcl-gc), folded inline)
_fill(e, "%dvp", t_CV, "%f0")
generic(  maps=["(i, d, j) -> (j, i)", "(i, d, j) -> (j, d)", "(i, d, j) -> (i, d)"],
         ins=("%Abuf", "%dO"), outs=("%dvp",))   # P^T dO
generic(  maps=["(i, d, kk) -> (i, kk)", "(i, d, kk) -> (i)", "(i, d, kk) -> (kk, d)",
                "(i, d, kk) -> (i, d)"],
         ins=("%k_c", "%gc", "%dH"), outs=("%dvp",))  # + (k*2^(gcl-gc)) dH
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2901-2917`，Bhat 内联缩并逐字）：

```mlir
linalg.generic {
  indexing_maps = [ affine_map<(i, d, kk) -> (i, kk)>, affine_map<(i, d, kk) -> (i)>,
                    affine_map<(i, d, kk) -> (kk, d)>, affine_map<(i, d, kk) -> (i, d)> ],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%k_c, %gc, %dH : ...) outs(%dvp : memref<64x16xf32>) {
^bb0(%a0: f32, %a1: f32, %a2: f32, %o0: f32):
  %bd = arith.subf %gcl, %a1 : f32
  %be = math.exp2 %bd : f32
  %bm = arith.mulf %a0, %be : f32
  %bp = arith.mulf %bm, %a2 : f32
  %bs = arith.addf %o0, %bp : f32
  linalg.yield %bs : f32
}
```

转置（$P^{\top}$）再次以 indexing_maps 表达（`(i,d,j)->(j,i)`），不物化转置缓冲。

### 2.3.3 $\mathrm{dH}$ 反向递推

**动机**。$S_{\text{exit}} = \Phi S_{\text{entry}} + b$（§2.2.3 的仿射形式）直接给出状态余切的反向递推：$\mathrm{dH}$ 是"离开该 chunk 的状态"的余切，从 $\mathrm{dht}$ 播种，逐 chunk（逆序）更新：

$$
\mathrm{dH}' \;=\; 2^{\mathrm{gcl}}\,\mathrm{dH} \;+\; \bigl(q\odot 2^{\mathrm{gc}}\bigr)^{\!\top}\mathrm{dO} \;-\; w^{\top}\mathrm{dvp},
$$

chunk 0 处理完后 $\mathrm{dh0}[b,h]=\mathrm{dH}$。三项分别对应仿射递推的三条活跃路径：衰减（线性部分的 $2^{\mathrm{gcl}}I$）、$o_{\text{inter}}$ 链（$q\odot2^{\mathrm{gc}}$ 作用在 $S$ 上）、$v_{\text{new}}$ 链（$w$ 作用在 $S$ 上，经 $\mathrm{dvp}$ 汇流）。

**源码片段**（`tools/gdn_l1_gen.py:1350-1371`；`%qg` 已在 dq 链算出（§2.3.9），此处复用）：

```python
# dH = dH .* 2^gcl + qg^T dO - w^T dvp
_fill(e, "%tK1", t_KV, "%f0")
generic(maps=["(kk, d, i) -> (i, kk)", "(kk, d, i) -> (i, d)", "(kk, d, i) -> (kk, d)"],
        ins=("%qg", "%dO"), outs=("%tK1",))          # qg^T dO
_fill(e, "%tK2", t_KV, "%f0")
generic(maps=["(kk, d, i) -> (i, kk)", "(kk, d, i) -> (i, d)", "(kk, d, i) -> (kk, d)"],
        ins=("%wp", "%dvp"), outs=("%tK2",))         # w^T dvp
generic(ins=("%dH", "%tK1", "%tK2"), outs=("%dH",)):
    %dh1 = dH*dl ; %dh2 = dh1 + tK1 ; %dh3 = dh2 - tK2
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:3448` 起的三 generic；`%dl = math.exp2 %gcl` 在 chunk 头部一次算出）。

**效果数字**。$\mathrm{dh0}$ 列的对拍差为全梯度中最小之一：$4.5\times10^{-7}$（npz 三用例，fp32/fp64 双口径，`evidence/G1/l1_bwd_check_output.txt`）——单 $[K,V]$ 缓冲的递推路径无 chunk 内重组，误差不叠加。

### 2.3.4 $\mathrm{dW}$ 与 $G$：$T$ 的余切汇聚

**动机**。$u,w$ 都线性依赖 $T$，两条链的余切在 $T$ 处汇聚。$w$ 进入 $v_{\text{new}}=u-wS_n$（经 $\mathrm{dvp}$），$u$ 不依赖 $S_n$ 故其对 $T$ 的贡献经 $\mathrm{dvp}$ 的 $u$ 部分同样汇入；组合结果：

$$
\mathrm{dW} = -\,\mathrm{dvp}\,S_n^{\top}\;\in\mathbb R^{C\times K}, \qquad
G \;=\; \mathrm{dvp}\,(\beta v)^{\top} \;+\; \mathrm{dW}\,(\beta k\,2^{\mathrm{gc}})^{\top} \;=\; \mathrm{dvp}\,\mathrm{upre}^{\top} + \mathrm{dW}\,\mathrm{kb2}^{\top},
$$

$G=\partial L/\partial T\in\mathbb R^{C\times C}$。注意生成器**复用前向缓冲** `%upre`（$=\beta v$）与 `%kb2`（$=\beta k 2^{\mathrm{gc}}$）——它们在 `_bwd_fwd_core` 重算时已经就位（`tools/gdn_l1_gen.py:1092-1104`），$G$ 的两个缩并因此零额外预处理。

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2918-2972`：`%ws = arith.subf %o0, %wp2`（dW 的负号折进减法缩并）、两个累加缩并进 `%Gb`、随后**独立的一个掩蔽 generic** 对 $G$ 取严格下三角：`%glt = cmpi slt %gj0, %gi0 ; %gr0 = select %glt, %a0, %f0`）。这一步"先掩蔽"不是可读性选择，而是正确性要求——下一小节推导。

### 2.3.5 dA 双侧掩蔽公式：推导与单侧形式的失效

**动机**。$T=(I+A)^{-1}$ 把 $G$ 变回对 $A$（即对 $k,\beta,g$ 的入口）的余切，是反向最微妙的一步。矩阵求逆的微分与伴随：

$$
\mathrm{d}T = -\,T\,(\mathrm{d}A)\,T .
$$

对损失 $L$（本规约中 $L=\sum \mathrm{do}\odot o+\sum \mathrm{dht}\odot h_t$，`evidence/G1/jax_spec_report.md` §2）：

$$
\mathrm{d}L = \langle G,\mathrm{d}T\rangle = \bigl\langle -T^{\top}G\,T^{\top},\,\mathrm{d}A\bigr\rangle .
$$

**为什么必须先掩蔽 $G$**。$A$ 严格下三角 $\Rightarrow$ $\mathrm{d}A$ 严格下三角 $\Rightarrow$ $\mathrm{d}T=-T\,\mathrm{d}A\,T$ 严格下三角（$T$ 单位下三角：$(T\,\mathrm{d}A)$ 的 $(i,j)$ 元 $=\sum_{k>j}T_{ik}\mathrm{d}A_{kj}$，当 $j\ge i$ 时 $k>j\ge i$ 使 $T_{ik}=0$；严格下三角乘下三角仍严格下三角）。于是 $\langle G,\mathrm{d}T\rangle$ 只依赖 $G$ 的严格下三角部分。但若**不先**把 $G$ 换成 $\mathrm{strict\_tril}(G)$ 就做乘积，$T^{\top}GT^{\top}$ 的严格下三角部分会通过 $T$ 的**单位对角**把 $G$ 的对角/上三角分量"泄漏"进结果（乘积项 $\sum_{k,l}T_{ki}\,G_{kl}\,T_{jl}$ 中 $k,l$ 不受 $j<i$ 约束）。因此正确形式是**双侧掩蔽**：

$$
\boxed{\;\mathrm{d}A \;=\; -\,\mathrm{strict\_tril}\Bigl(T^{\top}\cdot \mathrm{strict\_tril}(G)\cdot T^{\top}\Bigr)\;}
$$

**为什么单侧形式错**。任何只在一侧乘 $T^{\top}$（或对 $\mathrm{d}A$ 做对角补正）的近似都等价于丢掉一个方向的伴随混合：左乘 $T^{\top}$ 混合 $G$ 的**列**方向、右乘 $T^{\top}$ 混合**行**方向，而 $T$ 的严格下三角元素恰是 $-A$ 的 Neumann 累积（$T=I-A+A^2-\cdots$），两个方向的混合都携带 $A$ 的一阶信息。符号级验证（`evidence/G1/l1_harness_report.md` "The dA correction" 一节原话）：任务简报中的单侧变体 `dA.T + T^T (dA - diag dA)` 对 $C\ge 3$ 不复现 JAX vjp——$C{=}3$ 时该式给出 $2g_{21}+c\,g_{31}$，而真值是 $g_{21}+c\,g_{31}+c\,g_{32}a$（其中 $a$ 为 $A$ 的元素、$c$ 为相应衰减因子）：单侧形式既**重复计数**直达项（$2g_{21}$）又**丢失**交叉项（$c\,g_{32}a$，只可能由双侧乘积产生）。数值确认：双侧掩蔽式在 fp64 numpy 孪生上对三个 npz dump 的全部六个梯度 $\le 1.4\times10^{-14}$；且该式与 fla `prepare_wy_repr_bwd` 的 `dot(A, dA, A)` 等价——fla 从**转置存储布局**装载 $T$ 逆，其表达式读作 $T^{\top}GT^{\top}$，与上式一致（`evidence/G1/l1_harness_report.md` "Deviations" 第 4 条）。

**源码片段**（`tools/gdn_l1_gen.py:1113-1137`：三次缩并/掩蔽——$X=G_bT^{\top}$、$Y=T^{\top}X$、终步把衰减、负号、外层掩蔽合并进**一个** generic）：

```python
# X = G T^T ; Y = T^T X ; dAf = -strict_tril(Y .* D) into Gb (decay-free)
generic(maps=["(i,j,m)->(i,m)", "(i,j,m)->(j,m)", "(i,j,m)->(i,j)"],
        ins=("%Gb", "%Tbuf"), outs=("%Xb",))            # X = G_str T^T
generic(maps=["(i,j,m)->(m,i)", "(i,j,m)->(m,j)", "(i,j,m)->(i,j)"],
        ins=("%Tbuf", "%Xb"), outs=("%Gb",))            # Y = T^T X  (Gb 复用)
generic(ins=("%Gb", "%gc", "%gc"), outs=("%Gb",)):
    %dd0 = gc_i - gc_j ; %de0 = math.exp2 %dd0
    %dm0 = Y * de0 ; %dn0 = arith.negf %dm0
    %dlt = cmpi slt j, i ; %dr0 = select %dlt, %dn0, %f0
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2973-3020`，终步逐字）：

```mlir
linalg.fill ins(%f0 : f32) outs(%Xb : memref<64x64xf32>)
linalg.generic { ... affine_map<(i,j,m)->(i,m)>, <(i,j,m)->(j,m)>, <(i,j,m)->(i,j)> ... }
  ins(%Gb, %Tbuf : ...) outs(%Xb : memref<64x64xf32>) { ^bb0: %xp/%xs ... }   // X
linalg.fill ins(%f0 : f32) outs(%Gb : memref<64x64xf32>)
linalg.generic { ... affine_map<(i,j,m)->(m,i)>, <(i,j,m)->(m,j)>, <(i,j,m)->(i,j)> ... }
  ins(%Tbuf, %Xb : ...) outs(%Gb : memref<64x64xf32>) { ^bb0: %yp/%ys ... }   // Y
linalg.generic { ... <(i,j)->(i,j)>, <(i,j)->(i)>, <(i,j)->(j)>, <(i,j)->(i,j)> ... }
  ins(%Gb, %gc, %gc : ...) outs(%Gb : memref<64x64xf32>) {
^bb0(%a0: f32, %a1: f32, %a2: f32, %o0: f32):
  %dd0 = arith.subf %a1, %a2 : f32
  %de0 = math.exp2 %dd0 : f32
  %dm0 = arith.mulf %a0, %de0 : f32
  %dn0 = arith.negf %dm0 : f32
  %di0 = linalg.index 0 : index
  %dj0 = linalg.index 1 : index
  %dlt = arith.cmpi slt, %dj0, %di0 : index
  %dr0 = arith.select %dlt, %dn0, %f0 : f32
  linalg.yield %dr0 : f32
}
```

成本注记：dA 修正 = 每 chunk 两个 $C\times C\times C$ 缩并 + 掩蔽（`evidence/G1/GATE_REPORT.md` 数学成果第 3 条），分块/SIMD 化归 L2。

### 2.3.6 $\mathrm{dAf}$：衰减坐标

**动机**。$A$ 的定义自带一个衰减因子：$A=\mathrm{strict\_tril}\bigl(\beta\,(\mathrm{kk}^{\top})\odot D\bigr)$。对 $k$ 链求导时链式法则会**再**引入一个 $D$（$\partial A_{ij}/\partial(\text{raw }kk^{\top}_{ij})=\beta_i D_{ij}$）。把两个 $D$ 合并，定义**衰减自由坐标**下的余切（fla 约定）：

$$
\mathrm{dAf} \;=\; \mathrm{d}A \odot D \;=\; -\,\mathrm{strict\_tril}\Bigl(\bigl(T^{\top}\mathrm{strict\_tril}(G)T^{\top}\bigr)\odot D\Bigr),
$$

此后 $k$ 链（dk/dbeta/dg 的 A-链）中**不再出现任何 $\exp_2$ 因子**——§2.3.5 终步 generic 中的 `%de0 = math.exp2 %dd0` 正是这最后一次（也是该链唯一一次）衰减作用。numpy 孪生同式：`dAf = -(Tm.T @ (G * m_str) @ Tm.T) * D * m_str`（`tools/gdn_l1_check.py:197`）。

### 2.3.7 $\mathrm{dv}$ 与 $\mathrm{d\beta}$

$$
\mathrm{dv} = \bigl(T^{\top}\mathrm{dvp}\bigr)\odot\beta \qquad\in\mathbb R^{C\times V}
$$

（$v$ 只经 $\mathrm{upre}=\beta v$ 进入 $u=T\,\mathrm{upre}$；`tools/gdn_l1_gen.py:1139-1158`：`%tv = T^T dvp` 缩并，`%dvr = tv * beta` 存入 `%dv_c`。）

$$
\mathrm{d\beta}_i = \underbrace{\sum_j \mathrm{dAf}_{ij}\,\mathrm{kkt}_{ij}}_{\text{A-链}} \;+\; \underbrace{\sum_k \mathrm{TW}_{ik}\,k_{ik}\,2^{\mathrm{gc}_i}}_{\text{W-链，}\textbf{无}\,\beta} \;+\; \underbrace{\sum_d \mathrm{tv}_{id}\,v_{id}}_{\text{v-链}}, \qquad \mathrm{TW}:=T^{\top}\mathrm{dW}.
$$

三个行和由三个带归约维的 generic 算出（`%ra/%rb/%rc`），再用一个一维 generic 相加（`tools/gdn_l1_gen.py:1160-1186`）。**易错点**（`evidence/G1/l1_harness_report.md` 梯度公式处明确标注 "NO beta here"）：dbeta 的 W-链行和是 $k\,2^{\mathrm{gc}}$（**不带**行因子 $\beta$），而 dg 的同源 W-链（下节）**带** $\beta$——"同一 reduction 带不带行因子，极易顺手合并出错"。

### 2.3.8 $\mathrm{dk}$：五项组装

$$
\mathrm{dk} \;=\; \underbrace{(\mathrm{dAf}\,k)\odot\beta}_{\text{A-链（行）}} \;+\; \underbrace{\mathrm{dAf}^{\top}(\beta\,k)}_{\text{A-链（列）}} \;+\; \underbrace{\mathrm{TW}\odot(\beta\,2^{\mathrm{gc}})}_{\text{W-链}} \;+\; \underbrace{\bigl(v_{\text{new}}\odot 2^{\mathrm{gcl}-\mathrm{gc}}\bigr)\mathrm{dH}^{\top}}_{\hat B\text{-链（状态）}} \;+\; \underbrace{\mathrm{dPm}^{\top}q}_{o_{\text{intra}}\text{-链}}
$$

其中

$$
\mathrm{dPm} = \bigl(\mathrm{dO}\,v_{\text{new}}^{\top}\bigr)\odot D\odot\mathrm{tril}_{\text{incl}} \qquad (\text{P 的余切，含衰减含对角}).
$$

**源码**：`tools/gdn_l1_gen.py:1047-1064`（dPr=dO vn^T 原始缩并；dPm=掩蔽+衰减 generic）、`:1216-1235`（`%vng = vn⊙2^{gcl-gc}`、`%dkB = vng dH^T`、`%rowB = rowsum(dkB⊙k)`——后两者同时服务 dg）、`:1237-1272`（五项按序累加进 `%dkacc`：两个 $C\times C\times K$ 缩并（行/列两个方向）、一个 elementwise（TW 链，`%ka7 = exp2(gc); %ka8 = beta*ka7; %ka9 = TW*ka8`）、一个 elementwise 加（dkB）、一个 $C\times C\times K$ 缩并（dPm^T q），最后 `linalg.copy` 存入 strided 输出视图）。

**效果数字**。dk 是六梯度中对拍差最大的列（npz 三用例 $5.6$–$5.8\times10^{-6}$，`evidence/G1/l1_bwd_check_output.txt`）——五条链的误差叠加，但仍在 printMemrefF32 六位有效数字地板（`evidence/G1/l1_harness_report.md`）。

### 2.3.9 $\mathrm{dq}$

$$
\mathrm{dq} = \underbrace{\bigl(\mathrm{dO}\,S_n^{\top}\bigr)\odot 2^{\mathrm{gc}}}_{o_{\text{inter}}\text{-链}} \;+\; \mathrm{dPm}\;k \qquad\in\mathbb R^{C\times K}
$$

（`tools/gdn_l1_gen.py:1188-1214`：`%qg = q⊙2^gc`（服务 dH 链复用）、`%dq1 = dOgc Sn^T`（其中 `%dOgc = dO⊙2^gc` 先行算出）、`%r3v = rowsum(dq1⊙q)`（服务 dg，见下节）、`dq1 += dPm @ k`、`linalg.copy` 存储。）

### 2.3.10 $\mathrm{dg}$ 的组装：G1-B2 四项发现的集中地

**动机**。$g$ 以**指数**进入每一处衰减因子，因此 $\mathrm{dg}$ 是所有梯度中结构最繁的：chunk 内它是一个"行和 − 列和"的后缀和结构（衰减差的求导差分），chunk 间还叠加 Bhat 链的严格前缀和与一个 chunk 常数。生成器把它组装为（`tools/gdn_l1_gen.py:847-849` docstring、`evidence/G1/l1_harness_report.md` 梯度公式）：

$$
\mathrm{dg}_t \;=\; \underbrace{\mathrm{suffixsum}_t\bigl[\,\mathrm{rows}_t-\mathrm{cols}_t\bigr]_{\mathrm{dAf}\,\odot\,\beta_i\,\mathrm{kkt}} \;+\; \mathrm{rows}_t\bigl(\mathrm{TW}\odot(\beta k 2^{\mathrm{gc}})\bigr) \;+\; \mathrm{rows}_t\bigl((\mathrm{dO}S_n^{\top})\odot(q\,2^{\mathrm{gc}})\bigr) \;+\; \bigl[\,\mathrm{rows}_t-\mathrm{cols}_t\bigr]_{(\mathrm{dO}\,v_{\text{new}}^{\top})\odot P}\,\bigr]}_{\text{四片行/列向量 } t_A+t_B \;=\; \mathrm{dgr}}
$$

$$
+\; \underbrace{\sum_{c<t}\mathrm{rowB}_c}_{\text{严格前缀}} \;+\; \underbrace{2^{\mathrm{gcl}}\sum_{k,v}\mathrm{dH}\odot S_n}_{\text{chunk 常数}}, \qquad \mathrm{rowB}_t := \mathrm{rows}_t\Bigl(\bigl(v_{\text{new}}\odot 2^{\mathrm{gcl}-\mathrm{gc}}\bigr)\mathrm{dH}^{\top}\odot k\Bigr).
$$

对应生成器缓冲：`%ra/%rb/%rc/%r3v/%r4/%c1r/%c4r`（行/列和）→ `%tA/%tB` → `%dgr`；`%rowB` → 前缀循环 `%dgp2`；`%s5`（$[\,1\,]$ memref 的全归约）→ `%s5x = s5*2^{gcl}` 广播加；最后后缀循环写入 `%dg_c`。

**发现一（o_intra 项必须配 raw 缩并）**。第四片是 $\bigl[\mathrm{rows}-\mathrm{cols}\bigr]_{(\mathrm{dO}v_{\text{new}}^{\top})\odot P}$——用**原始** $\mathrm{dPr}=\mathrm{dO}v_{\text{new}}^{\top}$ 与 $P$ 配对，**不是** $\mathrm{dPm}\odot P$：dPm 已含衰减 $D$，再乘 $P$（$P$ 自带 $D$）会**双计衰减**。代价实测 $0.1$–$0.5$ abs 误差（`evidence/G1/l1_harness_report.md` 两个 pitfalls 之首，"dg 的 o_intra g-项配 RAW (dO vn^T) 与 P——NOT dPm * P"）。IR 佐证：`%r4/%c4r` 的输入是 `%dPr` 与 `%Abuf`（=P），而 dq/dk 链用的是 `%dPm`（`tools/gdn_l1_gen.py:1284-1295` vs `:1208-1213`）。该项等价于 fla 的 `rowsum(dq_loc q) - rowsum(dk_loc k)` 以 $v=v_{\text{new}}$ 的 q/k-路线。

**发现二（dg 无 RCP_LN2 重标度）**。kernel 内嵌的 $g$ 是 $\log_2$ 域（$g\cdot\log_2 e$，§2.2.2），于是**每一个**衰减因子 $\exp_2(\mathrm{gc}_{\log_2})\equiv e^{\mathrm{gc}_{\text{nat}}}$ 天然就是自然域乘子——所有 dg 分片已是自然对数域余切，**最后再乘 $\log_2 e$ 是 bug**（dg 整体被放大 $1.4427$ 倍）。生成器在该处留有注释（`tools/gdn_l1_gen.py:1338-1339`）：

```python
# dg = suffix(dgr) + dgp2  (the pieces are already natural-log cotangents:
# every decay is exp2(gc_log2) == e^(gc_nat), so no RCP_LN2 rescale here)
```

这是"数域约定"从约定条款变成可执行判据的直接例子：`evidence/G1/GATE_REPORT.md` 将其列为数学成果第 2 条。

**发现三（$\beta$ 行因子的有/无）**。dg 的 A-链行项带 $\beta_i$（`%tA` 组装中 `%ta0 = ra*beta` 与 `%ta2 = rb*beta`，列项 `%c1r = colsum(dAf⊙kkt⊙beta_row)` 减去——注释"dg's A-chain ROW term carries the beta_i row factor; dbeta's ra does not"，`tools/gdn_l1_gen.py:1296-1308`），而 dbeta 的同源行和 `%ra` **不带** $\beta$；W-链同构对照见 §2.3.7。

**发现四（Bhat 链 = 严格前缀）**。fla 把 $\mathrm{rowB}$ 之和加到 chunk 末行再做一次全局逆 cumsum，代数上等于 $-\mathrm{suffix}(\mathrm{rowB})+\sum\mathrm{rowB}=$ 严格前缀 $\sum_{c<t}\mathrm{rowB}_c$；L1 直接发射前缀（一个携带累加器的 `scf.for`，**先存后加**实现排他前缀），见 IR。两侧的实现差异在 fp64 孪生对 JAX vjp $\le 1.4\times10^{-14}$ 下验证等价（`evidence/G1/l1_harness_report.md` "Deviations" 第 1、2 条）。

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:3380-3419`——s5 全归约、前缀循环、后缀循环三段；后缀循环用 `idx = C-1-i7` 正步长逆序，且累加发生在读 $\mathrm{dgr}[\mathrm{idx}]$ 之后写入之前——**含尾**后缀和）：

```mlir
scf.for %i6 = %c0 to %cC step %c1 iter_args(%pacc = %f0) -> (f32) {
  memref.store %pacc, %dgp2[%i6] : memref<64xf32>      // 先存：排他前缀
  %rb_i = memref.load %rowB[%i6] : memref<64xf32>
  %pacc2 = arith.addf %pacc, %rb_i : f32
  scf.yield %pacc2 : f32
}
linalg.generic { ... } ins(%dgp2) outs(%dgp2) {          // + s5x 广播常数
^bb0(%a0: f32, %o0: f32):
  %dgpx = arith.addf %a0, %s5x : f32
  linalg.yield %dgpx : f32
}
scf.for %i7 = %c0 to %cC step %c1 iter_args(%sacc = %f0) -> (f32) {
  %idx = arith.subi %cCm1, %i7 : index                   // 正步长逆序
  %dr_i = memref.load %dgr[%idx] : memref<64xf32>
  %sacc2 = arith.addf %sacc, %dr_i : f32                 // 含尾后缀累加
  %dp_i = memref.load %dgp2[%idx] : memref<64xf32>
  %dgt = arith.addf %sacc2, %dp_i : f32
  memref.store %dgt, %dg_c[%idx] : memref<64xf32, strided<[2], offset: ?>>
  scf.yield %sacc2 : f32
}
```

**效果数字**。dg 列对拍：jax_C64/jax_C64_h0 $6.0\times10^{-6}$、jax_C32 $4.3\times10^{-6}$（fp32/fp64 双口径，`evidence/G1/l1_bwd_check_output.txt`）；seeded 尾块用例最差 $6.6\times10^{-6}$（chunk32）——发现一/二若犯任何一个，误差是 $0.1$–$1.4$ 量级，比 gate 大三个数量级以上。

### 2.3.11 尾块语义与正步长逆序

尾块（$T$ 非 $C$ 整数倍）在反向同样以零填充闭合：补零行 $\beta=k=g=\mathrm{do}=0$ 对 A/u/P/dvp/G/dA 及全部 suffix/prefix/rowsum 贡献**精确零**，真实行精确、补零行读零（`evidence/G1/l1_harness_report.md`；由 `tail_t96`（最差梯度 $\mathrm{dbeta}$ $3.6\times10^{-5}$）与 `tail_b2`（$\mathrm{dk}$ $8.1\times10^{-6}$）两个 seeded 用例 PASS 验证，`evidence/G1/l1_bwd_check_output.txt`）。逆序统一用正步长 + 下标重映射（chunk 级 `n = NC-1-n2`、行级 `idx = C-1-i7`）——`evidence/G1/l1_harness_report.md` 明确记录"负步长被刻意规避"。

## 2.4 生成器架构与 CPU 执行管线

### 2.4.1 生成器总体结构

`tools/gdn_l1_gen.py`（1428 行）是参数化的 MLIR 文本生成器：`--shape B,T,H,K,V --chunk C --seed N --emit fwd|bwd|both --npz <ref.npz> --out <f.mlir>`（默认形状 `1,128,2,16,16`、$C{=}64$；约束 `assert C >= 2`）。`--emit fwd` 的输出与反向功能加入前的版本**逐字节一致**（`evidence/G1/l1_harness_report.md` G1-B2 更新注记："the fwd output is byte-identical"）。时间维先零填充 $T_p=\lceil T/C\rceil\cdot C$、$\mathrm{NC}=T_p/C$（`tools/gdn_l1_gen.py:213-214`）。生成模块的解剖：

```
module {
  func.func private @printMemrefF32(memref<*xf32>)         // 运行时外部函数
  memref.global "private" constant @q : memref<BxTpxHxKxf32> = dense<...>
  ...  @k @v @beta @g(log2 域) @h0   [bwd: @do @dht]        // 输入常数嵌入
  [memref.global ... @ref_o @ref_ht @ref_dq ... ]          // 可选期望输出(--npz)
  func.func @gdn_fwd(...) -> (o, ht)                       // emit fwd|both
  func.func @gdn_bwd(...) -> (dq, dk, dv, dbeta, dg, dh0)  // emit bwd|both
  func.func @main() { get_global×N; func.call; memref.cast; call @printMemrefF32×N }
}
```

代表性产物：前向 2167 行（`evidence/G1/l1_chunked_T128_C64.mlir`）、反向 3540 行（`evidence/G1/l1_bwd_chunked_T128_C64.mlir`），其中绝大多数行数是嵌入的 dense 常数（反向模块的 kernel 代码从 2424 行起）。

### 2.4.2 代码模式一：`memref.global` 常数嵌入

**动机**。让模块**自包含**：不依赖外部数据加载通道，`mlir-runner` 直接执行；同时把"对拍输入"固化进 IR 文本，任何一次运行都可从 `.mlir` 文件完整复现。`--npz` 还会把期望输出嵌入为 `@ref_*` global 并由 `@main` 打印（`tools/gdn_l1_gen.py:264-277`），使产物自带"题目+答案"。

**源码片段**（`tools/gdn_l1_gen.py:136-158`——两个字面量工具；`:252-262` global 发射）：

```python
def fmt_f32(x):
    """MLIR float literal (this build requires an explicit '.' or exponent)."""
    s = "%.9g" % float(x)
    ...  # 补小数点/指数格式
def dense_literal(a, row_indent):
    """Nested dense<...> literal with one innermost row per line."""
    ...
    inner = (",\n" + ind).join(dense_literal(r, ind) for r in a)
```

```python
for name, arr, shape in ins:
    ty = mrt(shape)
    e(f'memref.global "private" constant @{name} : {ty} = dense<')
    lit = dense_literal(arr, "  " * e.depth)
    e.lines[-1] += lit + ">"
```

**IR 产物**（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:12-16`）：

```mlir
// ---- input data (zero time-padding to T_pad = 128) ----
memref.global "private" constant @q : memref<1x128x2x16xf32> = dense<[
  [
    [
      [0.0342215411, -0.0359566025, 0.174311697, 0.0285519548, -0.145799711, 0.0984197706, 0.354925722, 0.257778674, -0.191544265, -0.344425321, -0.169644266, 0.0112481993, -0.63283217, -0.0595512167, -0.339114904, -0.199310198],
      ...
```

两条工具链适配发现记录在 `evidence/G1/l1_harness_report.md` "Pass-spelling findings"：本构建（2026 年 10 月主线的 LLVM/MLIR 树）的解析器对字面量严格——dense 浮点列表必须有十进制点或指数（`1` 报错、`1.0` 合法，故有 `fmt_f32`），且**拒绝**扁平化（flattened）dense 列表，`dense_literal` 因此发射**完全嵌套**的字面量。

### 2.4.3 代码模式二：strided rank-reduced subview 视图族

**动机**。kernel 按 $(b,h)$、按 chunk 切片操作 4-D/3-D global，但算法对象是 $[C,K]$/$[C,V]$/$[C]$ 的 2-D/1-D 矩阵。生成器用**两级 `memref.subview`**（先 $(b,h)$ 再 chunk 偏移）直接在嵌入布局上取 strided 视图——**零打包拷贝**：

```mlir
%q_bh = memref.subview %q[%b, 0, %h, 0] [1, 128, 1, 16] [1, 1, 1, 1] : ... to memref<128x16xf32, strided<[32, 1], offset: ?>>
%q_c  = memref.subview %q_bh[%off, 0] [64, 16] [1, 1] : ... to memref<64x16xf32, strided<[32, 1], offset: ?>>
%beta_c = memref.subview %beta_bh[%off] [64] [1] : ... to memref<64xf32, strided<[2], offset: ?>>
```

（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:2507-2511`。$q$ 行步长 $32=H\cdot K$、$\beta$ 行步长 $2=H$——即"跨步 2 读 beta"这类布局直接由 strided 类型表达。）两个必须遵守的语义（`evidence/G1/l1_harness_report.md`）：`memref.subview` 的步长列表是**相对乘子**（非绝对步长），连续 chunk 选择写 `[1,1,1,1]`；rank-reducing subview（`[1,Tp,1,K]→[Tp,K]`）需要下降管线中 `-expand-strided-metadata -lower-affine` 才能落到 LLVM（配方取自 in-tree 测试 `mlir/test/Integration/Dialect/Linalg/CPU/rank-reducing-subview.mlir`）。类型字符串由 `mrt()` 辅助函数拼装（`tools/gdn_l1_gen.py:188-195`）。

### 2.4.4 代码模式三：`scf.for` 载态 + linalg 命名/泛型算子

**动机**。L1 需要显式的跨迭代状态（前向 $S$、反向 $\mathrm{dH}$ 与 `h_all`），同时需要矩阵级算子表达算法结构。设计取舍（`evidence/G1/l1_harness_report.md` "L1 structure" 设计选择）选了 **memref 就地变异**而非 tensor SSA 状态携带（`tensor.insert_slice`/yield）：不需要 bufferization 分析、编译干净——"语义优先于形式"；tensor 状态携带形式留给 L2。

三个子模式：

- **载态循环**：`scf.for %b → %h → %n(chunk)`；状态 $S$/dH 是每 $(b,h)$ 的 `memref.alloc` scratch，循环体内 load/store 就地更新（初始化 `linalg.copy` 自 $h_0[b,h]$ 的 rank-reduced subview，终态 copy 到 $h_t[b,h]$）。scratch 清单：前向 17 项（`tools/gdn_l1_gen.py:379-396`）、反向约 37 项（`:899-912`；含 `h_all`、dA 链的 `Gb/Xb/dPr/dPm`、dg 链的 8 个 $[C]$ 行向量等）。
- **fill-before-accumulate**：`linalg.matmul`/缩并型 `linalg.generic` 是累加语义，所有 outs 先 `linalg.fill`（`_fill` 辅助，`tools/gdn_l1_gen.py:742-743`）。
- **`generic()` 发射器**（`tools/gdn_l1_gen.py:701-733`）：把 indexing_maps / iterator_types / 块体 lambda 统一成一次调用，反向 kernel 的全部 40 余个 `linalg.generic` 由它发射；affine map 顺便承担转置（`(kk,vv,i)->(i,kk)` 型）与广播（`(kk,vv)->(0)` 型，s5 全归约）。

### 2.4.5 代码模式四：`printMemrefF32` 驱动

**动机**。CPU 侧唯一观测通道。模块声明外部运行时函数并把结果 cast 后打印（`tools/gdn_l1_gen.py:247,300-317`）：

```mlir
func.func private @printMemrefF32(memref<*xf32>)
...
%dq_ = memref.cast %dq : memref<1x128x2x16xf32> to memref<*xf32>
call @printMemrefF32(%dq_) : (memref<*xf32>) -> ()
```

（`evidence/G1/l1_bwd_chunked_T128_C64.mlir:3516-3539` 的 `@main` 依次打印六个梯度；fwd/both 模式先打印 $o,h_t$。）实现来自 `libmlir_c_runner_utils`（`evidence/G1/l1_harness_report.md`：仅需 private 声明 + `memref.cast`）。**精度边界**：printMemrefF32 打印约 6 位有效数字，$10^{-6}$ 以下的差不叮分辨——因此全部实测差（前向 $10^{-7}$ 级、反向 $10^{-6}$–$10^{-5}$ 级）都被理解为**打印地板**而非真实计算误差（`evidence/G1/l1_harness_report.md` Known limitations 第 1 条）。检查器 `parse_memrefs`（`tools/gdn_l1_check.py:235-261`）用括号配对 + `ast.literal_eval` 解析 `data = [...]` 文本块。

### 2.4.6 CPU 下降与执行管线：完整命令行与每步作用

**完整管线**（`scripts/run_l1.sh` 原文，与 `evidence/G1/l1_harness_report.md` "Working pipeline" 一致）：

```bash
BUILD=~/codebase/llvm-mlir/build

"$BUILD/bin/mlir-opt" in.mlir \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -expand-strided-metadata \
  -lower-affine \
  -convert-math-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-cf-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o out.mlir

"$BUILD/bin/mlir-runner" out.mlir \
  --shared-libs="$BUILD/lib/libmlir_runner_utils.so,$BUILD/lib/libmlir_c_runner_utils.so,$BUILD/lib/libmlir_float16_utils.so" \
  --entry-point-result=void
```

**每步作用**：

| pass | 作用 | 备注 |
|---|---|---|
| `-one-shot-bufferize="bufferize-function-boundaries"` | tensor→memref 缓冲化 | 对本 IR 是 **no-op**（已是 memref 级），为与 tensor 前端管线兼容而保留 |
| `-convert-linalg-to-loops` | `linalg.generic/matmul/fill/copy` → `scf.for` 循环 | 全部缩并/逐元素算子的标量化 |
| `-convert-scf-to-cf` | `scf.for` → `cf.br/cond_br` 控制流图 | |
| `-expand-strided-metadata` | strided memref 类型 → 显式 offset/size/stride 操作 | **必需**：rank-reducing strided subview 落 LLVM 的前提（§2.4.3） |
| `-lower-affine` | `affine.apply` 等 → arith | 同上必需 |
| `-convert-math-to-llvm` | `math.exp2` → `llvm.intr.exp2` | JIT 对共享运行时库带入的 libm 求解；**无需** `-convert-math-to-libm` |
| `-convert-arith-to-llvm` | arith → llvm 内联汇编/dialect | |
| `-finalize-memref-to-llvm` | memref 描述符、alloc/dealloc、`printMemrefF32`/`memref.cast` 的 C 运行时桥接 | |
| `-convert-cf-to-llvm` / `-convert-func-to-llvm` | 控制流/函数 → LLVM dialect | |
| `-reconcile-unrealized-casts` | 清理各转换间残留的 unrealized_conversion_cast | |
| `mlir-runner`（三个共享库） | JIT 编译 + 执行 `@main` | runner_utils/c_runner_utils（printMemrefF32）/float16_utils |

**效果数字**：base 模块 `mlir-opt` 0.32 s、`mlir-runner` 0.22 s（JIT+运行）；下降后的 LLVM-dialect 输出 2088 行（`evidence/G1/l1_harness_report.md`）。

## 2.5 JAX 可执行规约：语义单一事实源

### 2.5.1 三实现：`gdn_seq` / `gdn_chunk` / vjp 包装

`tools/gdn_ref_jax.py`（272 行，CPU-only、fp64 `jax_enable_x64`、可 jit、可微）从 fla oracle（`~/codebase/fla/fla/ops/gated_delta_rule/naive.py`，只读）逐行移植，构成整个工程的**语义单一事实源**：L0 方言的语义描述、L1 生成器的每一条数学、以及后续层的数值验证，全部以它为对拍基准。

**(a) 顺序规约 `gdn_seq`**——逐步递推的 `lax.scan` 移植（`tools/gdn_ref_jax.py:117-131`，注意四步的精确顺序：衰减在 delta 之前、$\beta$ 乘残差、输出读更新后状态）：

```python
def step(h, inputs):
    q_t, k_t, v_t, beta_t, g_t = inputs
    h = h * jnp.exp(g_t)[..., None, None]                    # 衰减先
    b_v = v_t - jnp.einsum("bhk,bhkv->bhv", k_t, h)          # 残差
    b_v = b_v * beta_t[..., None]                            # β 乘残差
    h = h + k_t[..., :, None] * b_v[..., None, :]            # delta 更新
    o_t = jnp.einsum("bhk,bhkv->bhv", q_t, h)                # 更新后读出
    return h, o_t
ht, o = jax.lax.scan(step, h0, (q_, k_, v_, beta_, g_))
```

**(b) chunked WY 规约 `gdn_chunk`**——`naive_chunk_gated_delta_rule` 的移植（`tools/gdn_ref_jax.py:139-242`），关键三段：含尾 chunk-local cumsum 与衰减掩码

```python
decay = jnp.cumsum(g.reshape(B, H, n, c), axis=-1)          # inclusive
diff = decay[..., :, None] - decay[..., None, :]
L_mask = jnp.tril(jnp.exp(jnp.tril(diff)))
```

$(I+M)^{-1}$ 的**行递推**（与 L1 前代换逐式对应，§2.2.6 对比表）：

```python
attn = -((k_beta @ k.swapaxes(-1, -2)) * L_mask)            # M，严格下三角
for i in range(1, c):
    attn = attn.at[..., i, :i].add(
        jnp.sum(attn[..., i, :i, None] * attn[..., :i, :i], axis=-2))
attn = attn + jnp.eye(c, dtype=q.dtype)                     # T = (I+M)^-1
w = jnp.matmul(attn, vc)                                    # = 本章 u
u = jnp.matmul(attn, k_beta_c * e_decay[..., None])         # = 本章 w
```

chunk 主循环（$v_{\text{new}}/o$ 分解/状态更新，`tools/gdn_ref_jax.py:210-232`）。

**(c) vjp 包装**——两个规约共用 `_vjp_common`（`tools/gdn_ref_jax.py:248-272`），对闭包 $f$ 取 `jax.vjp`，以 $(\mathrm{do},\mathrm{dht})$ 为余切反解六个梯度：

```python
def _vjp_common(fn, q, k, v, beta, g, h0, do, dht, **kw):
    h0 = _zeros_h0(q, v, h0)      # h0 缺省时对 zeros 微分
    def f(q_, k_, v_, beta_, g_, h0_):
        return fn(q_, k_, v_, beta_, g_, h0_=h0_, **kw)
    (o, ht), vjp_fn = jax.vjp(f, q, k, v, beta, g, h0)
    dq, dk, dv, dbeta, dg, dh0 = vjp_fn((do, dht))
    return o, ht, dq, dk, dv, dbeta, dg, dh0
```

**"忠实移植"纪律**（`tools/gdn_ref_jax.py:21-30`，"do NOT fix these"）：$\mathrm{scale}=K^{-0.5}$ 预乘进 $q$；$\beta$ 原始值；$g$ 自然对数域（并显式警告 MLIR 侧若用 $\exp_2$ 必须乘 $\mathrm{RCP\_LN2}$）；递推顺序固定。与 torch oracle 的两处**非数学性**差异被声明且保留：oracle 内部硬 cast fp32（规约保持调用方 dtype=fp64）、oracle 以 `output_final_state` 门控终态（规约恒返回 $(o,h_t)$）。

### 2.5.2 为什么它是"单一事实源"：44/44 自洽验证

`tools/gdn_ref_check.py`（jax 0.11.2 CPU，2026-10-08，$B{=}2,H{=}2,K{=}V{=}16$，fp64，GDN 风格输入）给出九类检查、44/44 PASS（`evidence/G1/jax_spec_report.md` §1）：

| 检查（最差组合） | 达到 | 容限 |
|---|---|---|
| chunked vs sequential，$o$（T=96/128，C=32/64，h0 有/无，8 组合） | $5.6\times10^{-16}$ | $10^{-10}$ |
| chunked vs sequential，$h_t$（同 8 组合） | $1.2\times10^{-15}$ | $10^{-10}$ |
| VJP 梯度 chunked vs sequential（六梯度，8 组合） | $1.2\times10^{-14}$ | $10^{-8}$ |
| 有限差分 vs VJP（seq，6 随机坐标） | $2.3\times10^{-9}$ | $10^{-5}$ |
| 有限差分 vs VJP（chunk，6 随机坐标） | $9.0\times10^{-10}$ | $10^{-5}$ |
| jit 一致性 seq / chunk | $0$ / $1.1\times10^{-16}$ | $10^{-12}$ |
| fla `naive_recurrent` vs jax seq（fp32 rel） | $1.2\times10^{-7}$ | $10^{-4}$ |
| fla `naive_chunk` vs jax chunk（fp32 rel） | $2.3\times10^{-6}$ | $10^{-4}$ |

这组数字完成三件事：**内部一致性**（chunk-vs-seq 是两个独立算法路径的互证，$10^{-16}$ 级即 fp64 舍入地板；有限差分独立验证 vjp 的微分手工推导无误）；**可执行性**（jit 编译不改变语义）；**外部锚定**（对 fla oracle 的 fp32 口径相符——fla naive 内部硬 cast fp32，故 fp64 对比不可能，容差放宽到 $10^{-4}$）。T=96 与 C=64 的组合专门行使零填充尾块（96→128 补 32 行）。

### 2.5.3 参考工件与三方对拍数字

规约以 `tools/gdn_ref_dump.py` 落出三个 npz 工件（`evidence/G1/`，均 $B{=}2,T{=}128,H{=}2,K{=}V{=}16$，fp64 计算、seed `PRNGKey(20261008)`；主数组 fp32 + 同名 `*_fp64` 副本 + `config` JSON，共 33 键/文件）：

| 文件 | chunk | h0 | chunk-vs-seq (fp64) |
|---|---|---|---|
| `ref_jax_T128_C64.npz` | 64 | 零 | $8.9\times10^{-16}$ |
| `ref_jax_T128_C32.npz` | 32 | 零 | $4.4\times10^{-16}$ |
| `ref_jax_T128_C64_h0.npz` | 64 | 种子化（$\max 0.31$） | $8.9\times10^{-16}$ |

三文件共享同一套 $q,k,v,\beta,g,\mathrm{do},\mathrm{dht}$（仅 $h_0$ 不同），使 MLIR-vs-JAX 差跨 chunk 尺寸直接可比。梯度是 $L=\sum\mathrm{do}\odot o+\sum\mathrm{dht}\odot h_t$ 的 VJP。

**三方对拍闭环**（`evidence/G1/GATE_REPORT.md` 判据 5 的链）：

$$
\underbrace{\text{MLIR L1}}_{\text{CPU fp32 执行}} \xrightarrow{\;\substack{\text{fwd }6.4\times10^{-7}\\ \text{bwd }\le 6.4\times10^{-6}}\;} \underbrace{\text{JAX 规约}}_{\text{CPU fp64}} \xrightarrow{\;\substack{1.2\times10^{-7}\,(\text{recurrent})\\ 2.3\times10^{-6}\,(\text{chunk})}\;} \underbrace{\text{fla naive torch}}_{\text{oracle}}
$$

外加第四极——numpy 孪生（`tools/gdn_l1_check.py` 的 `chunked_ref/seq_ref/chunked_bwd_ref`）：fp64 下孪生对 JAX vjp $\le 1.4\times10^{-14}$（三 npz 全六梯度），fp32 下孪生作为 MLIR 的直接参照把"算法差"与"执行差"分离。前向终表（`evidence/G1/l1_check_output.txt`，gate $10^{-3}$）：

| 用例 | 形状 | $\max\|o-\mathrm{ref}\|$ | $\max|h_t-\mathrm{ref}|$ | numpy chunk-vs-seq |
|---|---|---|---|---|
| base | B1 T128 C64 | $6.184\times10^{-7}$ | $8.643\times10^{-7}$ | $2.384\times10^{-7}$ |
| chunk32 | B1 T128 C32 | $5.513\times10^{-7}$ | $7.153\times10^{-7}$ | $1.192\times10^{-7}$ |
| tail_t96 | B1 T96 C64（尾 64+32） | $5.662\times10^{-7}$ | $6.258\times10^{-7}$ | $2.384\times10^{-7}$ |
| tail_b2 | B2 T100 C64（尾 64+36） | $5.960\times10^{-7}$ | $3.099\times10^{-6}$ | $3.874\times10^{-7}$ |
| jax_C64_h0 | B2 T128 C64，G1-A npz，种子 h0 | $6.407\times10^{-7}$ | $8.643\times10^{-7}$ | $5.662\times10^{-7}$ |
| jax_C64 | B2 T128 C64，npz，h0=0 | $6.407\times10^{-7}$ | $8.643\times10^{-7}$ | $5.662\times10^{-7}$ |
| jax_C32 | B2 T128 C32，npz，h0=0 | $5.066\times10^{-7}$ | $5.513\times10^{-7}$ | $5.066\times10^{-7}$ |

反向终表（`evidence/G1/l1_bwd_check_output.txt`；npz 单元格为 fp32/fp64 双口径；gate $2\times10^{-4}$/$3\times10^{-4}$ + 有限性检查，全部 finite=True）：

| 用例 | 形状 | dq | dk | dv | dbeta | dg | dh0 |
|---|---|---|---|---|---|---|---|
| jax_C64_h0 | B2 T128 C64 种子 h0 | $4.5/4$ | $5.8/6$ | $5.0/5$ | $6.4/7$ | $6.0/6$ | $0.45/0.5$（$\times10^{-6}$） |
| jax_C64 | B2 T128 C64 h0=0 | $4.5/4$ | $5.8/6$ | $5.0/5$ | $6.4/7$ | $6.0/6$ | $0.45/0.5$ |
| jax_C32 | B2 T128 C32 h0=0 | $4.5/4$ | $5.6/6$ | $5.0/5$ | $5.5/5$ | $4.3/4$ | $0.45/0.5$ |
| base（孪生） | B1 T128 C64 | $3.2$ | $6.0$ | $4.9$ | $5.0$ | $1.6$ | $0.058$（$\times10^{-6}$） |
| chunk32（孪生） | B1 T128 C32 | $2.3$ | $5.0$ | $2.1$ | $2.4$ | $6.6$ | $0.49$ |
| tail_t96（孪生） | B1 T96 C64 | $4.9$ | $4.8$ | $0.95$ | $36$ | $0.77$ | $0.37$ |
| tail_b2（孪生） | B2 T100 C64 | $3.2$ | $8.1$ | $4.9$ | $4.4$ | $3.0$ | $0.49$ |

### 2.5.4 检查器：`tools/gdn_l1_check.py`

检查器把规约、生成器、runner 串成单一入口（前向 `python tools/gdn_l1_check.py`；反向 `--bwd` / `--bwd-only-npz`）：生成 → `scripts/run_l1.sh` 执行 → `parse_memrefs` 解析 → 对照两极（嵌入 numpy 参照 / G1-A npz 的 JAX 输出）。四个 seeded 前向用例刻意覆盖尾块与多 batch；反向 gate 的双容差（fp32 主数组 $2\times10^{-4}$ 或 fp64 副本 $3\times10^{-4}$）直接承认打印地板的存在。npz 文件缺席时容错跳过（跨工作流解耦）。

## 2.6 L1 的边界与向 L2 的接口

`evidence/G1/l1_harness_report.md` "Known limitations" 与 "Notes for G2" 划定 L1 的诚实边界，每条同时是 L2 的接口（旋钮）：

1. **观测地板**：printMemrefF32 约 6 位有效数字，$10^{-6}$ 以下不可分辨——真实 fp32 计算误差"很可能小于实测 $\sim6\times10^{-7}$"。
2. **无 tensor SSA 历史**：状态经 memref 就地变异，无函数式历史——对 L1 无碍，对自动微分有碍；L3 的 WY 反向按独立 kernel 处理（本章 §2.3 即其数学）。
3. **scratch 逐 $(b,h)$ alloc/dealloc**：融合/持久缓冲调度归 L2。
4. **前代换 $O(C^3/6)$ 标量迭代**：刻意显式；行递推/Neumann/分块变体是 `solve variant` 旋钮，重新生成即可 A/B。
5. **dA 修正 = 每 chunk 两个 $C^3$ 缩并 + 掩蔽**：blocking/SIMD 归 L2。
6. **fp32、B/H 循环不 tile、无向量化**：linalg-to-loops 发射标量循环；vector dialect/SIMD 归 L2。
7. **重算-物化旋钮**：pass 1 存 u/w/P 的 materialize 变体是显式列出的存储/计算折中。
8. **与 fla kernel 级分解的等价-差异清单**（`evidence/G1/l1_harness_report.md` "Deviations"）：同一数学、不同分布——Bhat 链前缀直发（≡ fla 加末行+全局逆 cumsum）、dg 行组装显式四片（≡ fla 的 dq_loc/dk_loc 路线）、dH 寄存器式携带（fla 物化 dh[n]）、fla 的转置 A 存储怪癖被识别但未复制。

最后回到章首的定位：`docs/L0_SEMANTICS.md` 记录 `gdn-to-chunk` pass 当前的入口调用式占位与未来"chunk-AST 展开"，其验收口径即本章的全部内容——展开 pass 的输出必须与 `tools/gdn_l1_gen.py` 生成的 L1 文本**逐行同构**（前代换解三角、`scf.for` 载态两个代表性结构已先行验证）。换言之，L1 层的交付物有二：一份**可执行的算法定义**（本章 §2.2–§2.3 的全部数学 + IR），与一份**生成格式的规范**（本章 §2.4 的代码模式）；二者共同构成 Schedule-as-IR 论证中"分层可验证 IR"的第一块可对拍的基石。

---

### 本章引用文件清单

| 文件 | 角色 |
|---|---|
| `tools/gdn_l1_gen.py`（1428 行） | L1 生成器（前向 + 反向），本章主剖析对象 |
| `tools/gdn_ref_jax.py`（272 行） | JAX fp64 可执行规约（seq/chunk/vjp） |
| `tools/gdn_l1_check.py`（548 行） | 数值自检驱动（numpy 孪生 + memref 解析 + 双 gate） |
| `scripts/run_l1.sh` | CPU 下降 + 执行 harness |
| `docs/L0_SEMANTICS.md` | L0 语义规约（8 条约定裁定） |
| `evidence/G1/jax_spec_report.md` | G1-A：JAX 规约结果与约定（44/44、npz 工件、fla 契约） |
| `evidence/G1/l1_harness_report.md` | G1-B：L1 harness、管线发现、数学约定、G1-B2 反向全量 |
| `evidence/G1/l1_chunked_T128_C64.mlir`（2167 行） | 前向 IR 产物样本 |
| `evidence/G1/l1_bwd_chunked_T128_C64.mlir`（3540 行） | 反向 IR 产物样本（本章 IR 引用的主要来源） |
| `evidence/G1/l1_check_output.txt` / `l1_bwd_check_output.txt` | 前向 7/7、反向 7 用例对拍原始输出 |
| `evidence/G1/ref_jax_T128_{C64,C32,C64_h0}.npz` | G1-A 参考工件（33 键/文件） |
| `evidence/G1/GATE_REPORT.md` | G1 gate 判据与三方对拍链总结 |
