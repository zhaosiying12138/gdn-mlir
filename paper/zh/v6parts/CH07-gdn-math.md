## 七、GDN 的数学结构：从逐 Token 递推到 Chunk-WY

Gated DeltaNet 的每个 token 都先衰减状态，再用当前值与预测值之差修正状态。chunk（分块）化要保持这四步的先后顺序，同时把块内 token 的依赖整理成可批量计算的矩阵表达式。三角系统来自修正项之间的因果关系：第 i 个修正能读取更早 token 的写回，却不能读取当前 token 尚未完成的写回。

### 1. 四步递推与自然对数域

固定一个 batch 与 head，令状态 S 为 K×V 矩阵，k、q 为 K 维行向量，v、b 为 V 维行向量。省略可线性外提的输出缩放后，逐 token 语义为：

```text
D_t       = e^(g_t),                         g_t <= 0
S_decay_t = D_t * S_(t-1)                    # 先衰减
b_t       = beta_t * (v_t - k_t S_decay_t)   # 再作 delta 修正
S_t       = S_decay_t + k_t^T b_t            # 随后外积写回
o_t       = q_t S_t                          # 最后读取输出
```

[tools/gdn_ref_jax.py:97-133；输入生成约定见 64-83 行]

`beta` 是上游已经处理好的原始强度，q、k 的 L2 归一化也由调用方完成。规约的测试输入用 sigmoid 生成 beta，使其位于 (0,1)，并用负 softplus 生成 g[tools/gdn_ref_jax.py:69-83]；算子本身不重复施加这些激活。JAX 规约按同一顺序执行：

```python
h = h * jnp.exp(g_t)[..., None, None]
b_v = v_t - jnp.einsum("bhk,bhkv->bhv", k_t, h)
b_v = b_v * beta_t[..., None]
h = h + k_t[..., :, None] * b_v[..., None, :]
o_t = jnp.einsum("bhk,bhkv->bhv", q_t, h)
```

[tools/gdn_ref_jax.py:117-129]

【通用补充】顺序错误在一个标量例中就会显形。取 S=3、k=q=1、v=5、beta=1、g=-ln2，正确顺序先得到 S_decay=3/2，再有 b=5-3/2=7/2，最终状态为 5。若先按未衰减状态计算修正并写回，再对整体衰减，会得到 5/2。两种结果的差异来自 delta 读取的状态不同，与浮点舍入无关。

外部语义中的 g 始终是自然对数，衰减写成 e^g。L1 内核沿用 fla 的 log2 实现域，先在生成器中计算 g2 = g·log2(e)，再让 MLIR 执行 `math.exp2(g2)`；恒等式为 exp2(g·log2(e)) = e^g。

```python
RCP_LN2 = 1.4426950408889634  # log2(e)
# ……输入在时间维补零后……
g_log2 = (g.astype(np.float64) * RCP_LN2).astype(np.float32)
```

[tools/gdn_l1_gen.py:59,217-225]

因此，外部推导中的 e^g 与内部 `math.exp2` 是同一数学语义在两种计算域中的表达。内部累积量也应理解为 g2 的累积，求差后再执行 exp2；反向 dg 已处于自然对数余切域，无需在末端再乘一次 log2(e)[tools/gdn_l1_gen.py:1338-1348]。实数恒等式不消除浮点舍入：g2 的 f32 转换、累积求和以及 exp2 求值仍会引入误差，所以不能期待 L1 与 JAX 逐位相同。

### 2. C 个修正方程联立后出现严格下三角

取一个长度为 C 的 chunk，入口状态记为 S。定义块内包含当前 token 的累计门控 gc_i = g_0+…+g_i，i 从 0 到 C-1。第 i 步衰减后的状态可以展开成入口历史与更早写回两项：

```text
S_decay_i = e^(gc_i) S
            + sum_(j<i) e^(gc_i-gc_j) k_j^T b_j
```

指数 gc_i-gc_j 恰好收集第 j 步写回之后到第 i 步读取之前的各次衰减。把它代入 b_i = beta_i(v_i-k_i S_decay_i)，得到：

```text
b_i + sum_(j<i) A_ij b_j = R_i - Z_i S

A_ij = beta_i (k_i·k_j) e^(gc_i-gc_j),   j < i
R_i  = beta_i v_i
Z_i  = beta_i e^(gc_i) k_i
```

将 C 个方程按行堆叠，B、R 是 C×V 矩阵，Z 是 C×K 矩阵，A 是 C×C 矩阵，便有 `(I+A)B = R-ZS`。项目所称的 KKT 内积先形成 k_i·k_j，再经 beta 行因子、衰减和严格下三角掩蔽得到 A。严格性来自 j<i：对角项对应当前写回，而当前写回在修正计算之后发生，所以对角必须为零；上三角对应未来 token，也必须为零。

按 1 起始下标展开前三行，因果结构会更直观：

```text
b_1 = R_1 - Z_1 S
b_2 = R_2 - Z_2 S - A_21 b_1
b_3 = R_3 - Z_3 S - A_31 b_1 - A_32 b_2
```

第二行只含 b_1，第三行只含 b_1、b_2。继续代入当然可以得到只含 R-ZS 的显式表达式，但前代换保留这种逐行读取已解未知量的形式，避免先构造稠密逆矩阵。块内矩阵乘由此获得了与逐 token 扫描相同的依赖含义。

I+A 的对角全为 1，是单位下三角矩阵。它始终可逆，求解不需要选主元，也没有除以透视元的步骤。g≤0 使 j<i 时的衰减因子不超过 1；调用方的归一化限制单个内积的规模，这些条件与三角可解性共同描述算法结构，但不能替代具体输入上的数值验证。

### 3. 前代换产生 WY 中间量

令 T=(I+A)^-1。逐元素展开 `(I+A)T=I`，得到前代换（forward substitution）的精确递推：

```text
T_ii = 1
T_ij = -sum_(k=j..i-1) A_ik T_kj,        j < i
```

严格下三角 A 还是幂零矩阵：A^C=0。于是逆也可以写成有限多项式：

```text
T = I - A + A^2 - A^3 + ... + (-1)^(C-1) A^(C-1)
```

级数在有限项后精确结束，没有截断近似。A 表示一步依赖，A^2 表示跨两步的依赖，更高次项继续收集更长路径；前代换按另一计算顺序得到同一个 T。C=3 时有 T=I-A+A^2，第三行第一列会出现 A_32·A_21-A_31，这正是后文反向反例中交叉项的来源。实际 L1 选择显式三重循环前代换，工作量约为 O(C^3/6) 次标量迭代，并未发射这条矩阵多项式[evidence/G1/l1_harness_report.md:160-172]。

生成器把这三层循环直接发射为 `scf.for`，内层只读取已经算出的 T_kj：

```text
for i in range(C):
    T[i, i] = 1
    for j in range(i):
        acc = 0
        for k in range(j, i):
            acc += A[i, k] * T[k, j]
        T[i, j] = -acc
```

[tools/gdn_l1_gen.py:475-493，中间累加体有省略；A 的 j<i 掩蔽见 451-473 行]

采用 L1 生成器的命名，WY 表示（WY representation）把联立解组织成值侧和键侧中间量：

```text
u     = T R = T(beta ⊙ v)                 # C×V
w     = T Z = T(beta ⊙ k ⊙ e^gc)          # C×K
v_new = B = u - w S                        # C×V
```

```python
e(f"linalg.matmul ins(%Tbuf, %upre : {t_CC}, {t_CV}) outs(%ubuf : {t_CV})")
# ……生成 kb2 = beta .* k .* exp2(gc_log2)……
e(f"linalg.matmul ins(%Tbuf, %kb2 : {t_CC}, {t_CK}) outs(%wp : {t_CK})")
# ……vp = w @ S……
e("%r3 = arith.subf %xu, %xv2 : f32")
```

[tools/gdn_l1_gen.py:495-550，省略处已标出]

符号需要固定两处映射。JAX 参考代码的局部变量 `w_jax` 等于此处的 u，`u_jax` 等于此处的 w；比较公式时应按定义换名，不能按字母猜测[tools/gdn_ref_jax.py:202-225]。无下标的 u 专指 C×V 中间量；若发射代码把一次内存读取命名为 `u_load`，它只表示从 u[i,d] 取出的标量负载。推导中用 `u_elem` 称呼这类临时标量，避免把 u 与 u_load 误列成两个不同的 WY 矩阵。

先记 gcl 为当前 chunk 末行的累计门控，并令 K_hat_i = k_i·e^(gcl-gc_i)。矩阵维数提供了另一道校验。w 是 C×K，S 是 K×V，所以 wS 与 u 都是 C×V，能够相减得到 v_new；K_hat^T 是 K×C，因此 K_hat^T w 是 K×K，正好左乘入口状态。阅读 JAX 的 w-u 命名时也可先做这项维数检查，再套用上段映射，避免把值侧中间量与键侧中间量混用。

【通用补充】用 C=2、K=V=1 做一次完整手算。取 q_0=q_1=k_0=k_1=1，beta_0=beta_1=1/2，v_0=2、v_1=4，g_0=g_1=0，入口 S=2，缩放因子为 1。此时：

```text
R = [1, 2]^T             Z = [1/2, 1/2]^T
A = [[0, 0], [1/2, 0]]  T = [[1, 0], [-1/2, 1]]
u = [1, 3/2]^T           w = [1/2, 1/4]^T
v_new = u - w*S = [0, 1]^T
```

逐 token 计算给出 b_0=(1/2)(2-2)=0，b_1=(1/2)(4-2)=1，与 v_new 完全一致。状态也有 S'=2+[1,1]v_new=3；本例的输出掩蔽矩阵取 P=[[1,0],[1,1]]，给出 o=[2,2]^T+P·v_new=[2,3]^T。这个例子显示 u 只是尚未扣除入口状态影响的 WY 中间量，v_new 才对应逐 token 的 delta 修正。

### 4. 状态递推与输出共享同一个 v_new

令 gcl=gc_(C-1)。由上面已经定义的 K_hat_i = k_i·e^(gcl-gc_i)，从最后一个 token 的状态反推整块，跨 chunk 递推为：

```text
S' = e^gcl S + K_hat^T v_new
   = (e^gcl I - K_hat^T w) S + K_hat^T u
```

第二行显式展示 chunk 对入口状态的仿射映射：线性部分只乘一次入口 S，常量部分由本块输入决定。等式右侧的 S 都是本 chunk 的入口状态。尾块采用零填充，padding 行的 beta、k、v、g 均为零，额外输出由消费方裁掉[tools/gdn_ref_jax.py:152-180,228-238]。

这种组织方式改变了状态访问的粒度。逐 token 形式每一步都要读取并写回 K×V 状态；chunk 形式以入口 S 和出口 S' 为跨块接口，块内更多计算转移到 C×C 三角系统、C×K 与 C×V 中间量以及矩阵乘。收益取决于 C、K、V 和实现方式：较大的 C 能减少跨块状态往返，也会增大三角求解与中间存储的工作量。当前 CPU L1 保留显式标量前代换来固定语义，后续调度可以选择不同的求解和平行策略，不能从矩阵形式直接推出无条件加速。

输出在写回后读取，所以其块内掩蔽包含对角。将第 i 步完成写回后的状态展开：

```text
S_i = e^(gc_i) S + sum_(j<=i) e^(gc_i-gc_j) k_j^T b_j
o_i = q_i S_i
```

第一项直接形成 o_inter，求和项则形成 o_intra。下标 j≤i 保留了当前 token 刚完成的外积写回，这与修正方程中的 j<i 相差恰好一个对角位置。定义：

```text
P_ij = (q_i·k_j) e^(gc_i-gc_j),          j <= i
o_inter = (q ⊙ e^gc) S
o_intra = P v_new
o       = (o_inter + o_intra) * K^(-1/2)
```

A 使用严格下三角，P 使用含对角下三角，两者对应递推中的不同时间点。q 只线性进入输出，因此 JAX 在入口处缩放 q，L1 则在最终写出时统一乘 K^(-1/2)[tools/gdn_l1_gen.py:552-627]。第八章将看到 u、w、v_new、S、o_inter 与 o_intra 如何一一落到 memref 缓冲和 CPU 指令。

### 5. 反向的 dA 需要预掩蔽与双侧收缩

设标量损失为 L，G=∂L/∂T。由 T=(I+A)^-1 的微分：

```text
dT = -T dA T
dL = <G, dT> = <-T^T G T^T, dA>
```

内积恒等式中的两个 T^T 分别来自微分两侧的 T；缺少任意一侧，都会丢掉由复合依赖路径产生的梯度。直接整理得到候选式后，还要处理变量支撑集：A 与 dT 都只在严格下三角有自由度。数学上，dT 的严格下三角性已使其他 G 分量正交消失；项目用预掩蔽把这一事实显式写出，再完成左右收缩和最终支撑集投影：

```text
G_m = strict_tril(G)
dA  = -strict_tril(T^T G_m T^T)
dAf = dA ⊙ e^(gc_i-gc_j)
```

```python
dW = -(dvp @ Sn.T)
G = dvp @ Vb.T + dW @ Kbg.T
# ……m_str 为严格下三角掩码，D 为衰减矩阵……
dAf = -(Tm.T @ (G * m_str) @ Tm.T) * D * m_str
```

[tools/gdn_l1_check.py:192-198；Vb、Kbg 与 D 的定义见 184-190 行；MLIR 发射对应 tools/gdn_l1_gen.py:1083-1137]

【通用补充】C=3 的最小数值反例能显示左侧 T^T 所携带的交叉项。取 A_21=2、A_32=3、A_31=0，其余严格下三角项为零，则 T_31=A_32·A_21-A_31=6。令 G_31=1，其余 G 项为零：

```text
完整双侧式：dA_21 = 3,  dA_31 = -1,  dA_32 = 2
只做 -strict_tril(G T^T)：
             dA_21 = 0,  dA_31 = -1,  dA_32 = 2
```

T_31 中的乘积 A_32·A_21 表示跨两个 token 的依赖路径；A_21 改变时，T_31 的变化率是 A_32=3，所以完整梯度必须含 dA_21=3。单侧收缩只保留直达项，恰好漏掉这条路径。C=2 没有这种两段链，C=3 因而成为最小反例。项目证据也记录了单侧形式在 C≥3 时与 JAX VJP 不符，双侧掩蔽式在 fp64 孪生中通过核对[evidence/G1/l1_harness_report.md:220-240]。
