# G1 Gate 报告（方言+语义+CPU 三方对拍）—— ✅ 全判据达成（2026-10-09 03:30, commit d12c12e）

## Gate 判据核验（docs/PLAN.md G1）

| # | 判据 | 状态 | 证据 |
|---|---|---|---|
| 1 | L0 方言（gdn.fwd/gdn.bwd + GdnConfigAttr 旋钮 + verifier + gdn-to-chunk + gdn-opt） | ✅ 5555704 | dialect/，冒烟全绿（roundtrip/rewrite/attr 透传/私有声明生成） |
| 2 | JAX 可执行规约（seq+chunk+vjp，fp64） | ✅ 9597f4d | gdn_ref_check.py **44/44**；jax_spec_report.md |
| 3 | MLIR L1 前向（linalg+scf 真 IR，CPU 执行） | ✅ 4d4c283 | gdn_l1_check.py **7/7**，vs JAX dumps ~6e-7 |
| 4 | MLIR L1 反向（全梯度） | ✅ d12c12e | **3 npz × 6 grads PASS**（≤6.4e-6）；l1_bwd_check_output.txt |
| 5 | 三方语义一致（MLIR ⇄ JAX ⇄ fla naive，前向+反向） | ✅ | 链：L1↔JAX(6e-6) + JAX↔fla(1e-6/2e-6) |

## 关键数字

- **前向**：MLIR L1 vs JAX fp64 dumps ~6e-7；JAX chunked vs sequential（fp64）o 5.6e-16/ht 1.2e-15/grads 1.2e-14；JAX vs fla naive 1.2e-7 / 2.3e-6（fp32 口径）。
- **反向**（fp32 执行 / fp64 参照）：

| case | dq | dk | dv | dbeta | dg | dh0 |
|---|---|---|---|---|---|---|
| T128/C64/h0 | 4.5e-6 | 5.8e-6 | 5.0e-6 | 6.4e-6 | 6.0e-6 | 4.5e-7 |
| T128/C64 | 4.5e-6 | 5.8e-6 | 5.0e-6 | 6.4e-6 | 6.0e-6 | 4.5e-7 |
| T128/C32 | 4.5e-6 | 5.6e-6 | 5.0e-6 | 5.5e-6 | 4.3e-6 | 4.5e-7 |

（全部在 printMemrefF32 六位有效数字地板；numpy 孪生对 JAX fp64 ≤1.4e-14；`--emit fwd` 输出与改动前逐字节一致）

## 数学成果（G2 的 L1 基础，已入 l1_harness_report.md）

1. **dA 精确公式**：`dA = −stricttril(Tᵀ·stricttril(G)·Tᵀ)`，`dAf = dA ⊙ e^(gc_i−gc_j)`，其中
   `G = dvp(βv)ᵀ + dW(βk e^{gc})ᵀ`，`dW = −dvp·S_nᵀ`。G 必须**先掩蔽再两侧乘 Tᵀ**；
   单侧修正 `dA·T + Tᵀ(dA−diag)` 对 C≥3 不成立（符号级验证），双侧掩蔽式与 fla
   `prepare_wy_repr_bwd`（计入其转置 A 存储）等价。
2. dg 的 o_intra 项必须配 raw `(dO·vnᵀ)`（双计 decay 代价 0.35 abs）；
   dg 链**不做** RCP_LN2 重标度（exp2(gc_log2)≡e^{gc_nat}，天然同域）；
   dbeta/dg W-链 = 同一 reduction 带/不带行因子 β；
   Bhat 链 dg 项 = 严格前缀和（≡ fla 的"加末行+逆 cumsum"）。
3. 结构：pass1 重算前向只存入口态 h_all[n]；pass2 正步长逆走 chunk（n=NC−1−n2）；
   bwd 每 (b,h) 约 37 个 scratch（fwd 17）→ materialize-u/w/P 变体是 L2 旋钮；
   dA 修正 = 每 chunk 两个 C×C×C 缩并 + 掩蔽（blocking/SIMD 化归 L2）。

## 约定裁定（docs/L0_SEMANTICS.md，8 条）

g=自然对数域（kernel 内乘 log2(e) 用 exp2）；scale=K^{-1/2}；算子内不做 L2 归一；
beta 原始值、行向进入 `M[i,j]=−β_i(k_i·k_j)e^{cd_i−cd_j}`；T=(I+M)^{-1} 前代换；
含尾 cumsum；尾块补零；`h←e^{g_t}h` 先于 delta 更新；fla naive 两函数 beta/g 位置互换。

## 工程沉淀（LLVM Oct-2026 适配 10 条 + B2 数学 5 条）

见 RECOVERY.md 与 l1_harness_report.md；全部为 CAKE 式规则库条目。

## 结论

**G1 达成**。语义层（L0 方言 + JAX 规约 + L1 全梯度 MLIR 表达 + 三方对拍）闭合，
约定唯一化完成。进入 **G2：旋钮→PTX 覆盖矩阵**（L2 transform 旋钮库 + T1 指令族
覆盖 + T2 结构匹配 fla/FlashQLA 两极点）。
