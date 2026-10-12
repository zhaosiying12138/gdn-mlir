# L0 语义规约（gdn.fwd / gdn.bwd）—— 2026-10-09 按 G1-A 实测约定修订

> 语义唯一事实源链（已全线打通）：`tools/gdn_ref_jax.py`（JAX fp64，44/44 自洽）
> ⇄ fla naive（torch，1e-6/1e-7 相符）⇄ MLIR L1（`tools/gdn_l1_gen.py`，6e-7 相符）。
> 数值约定权威记录：`evidence/G1/jax_spec_report.md`。

## 关键约定（G1-A 实测裁定，勿凭直觉改）

1. **g 是自然对数域**：decay = exp(g)。fla Triton 路径在 kernel 前把 g 乘
   RCP_LN2=log2(e) 再用 exp2 —— MLIR kernel 同样预乘后用 `math.exp2`。
2. **scale = K^{-1/2}**（不是 K^{-0.2}），乘在（归一化后的）q 上。
3. **算子内部不做 q/k 的 L2 归一**（模型侧负责）；config.normalize_q 仅当上游
   需要时打开（fla 语义默认关闭）。
4. **beta 为原始值**（sigmoid 由上游做）；进入 chunk 系统的方式不对称：
   `M[i,j] = -beta_i (k_i·k_j) e^{cd_i-cd_j}`（行向 beta）。
5. **逆的约定**：`T = (I+M)^{-1}`，M 严格下三角；等价于对 `A = +beta_i(k_i·k_j)2^{gc_i-gc_j}`
   （L1 生成器记法，= -M）做前代换求 `(I+A)^{-1}`。chunk 内 g 累计为**含尾**
   cumsum；intra-chunk 注意力保留对角；`S ← e^{cd_last}·S + (k⊙e^{cd_last-cd_i})^T(w-u@S)`；
   尾块补零（beta=0/g=0 行为无操作）。
6. **顺序细节**：`h ← e^{g_t}·h` 在 delta 更新**之前**；beta 乘在 `(v_t - k_t@h)` 上。
7. fla 陷阱：`naive_chunk_gated_delta_rule(q,k,v,g,beta,...)` 与
   `naive_recurrent_gated_delta_rule(q,k,v,beta,g,...)` 的 beta/g 位置互换。
8. h0/ht 形状 [B,H,K,V]（fp32）；生产 kernel 中间态 h/dh 为 bf16（G0 收割裁定），
   数值验证层（G1）用 f32/f64。

## 输入输出（gdn.fwd / gdn.bwd）

见 `dialect/include/Gdn/GdnOps.td` 的 op 描述（generic 语法示例：
`dialect/test/smoke_l0.mlir`；config 属性：`#gdn.config<64, true, false, true, true, true, false>`）。

## `gdn.config` 旋钮

| 参数 | 含义 | G1 对拍取值 |
|---|---|---|
| chunk_size | L1 分块 {16,32,64} | 32/64 已对拍 |
| use_g / use_gk | 标量门（ln 域）/逐通道门 | true / false |
| normalize_q | 算子内 L2 归一（fla 默认不做） | false |
| use_initial_state / store_final_state | h0 / ht | true/true（含 h0=0 用例） |
| is_varlen | varlen | false（G4 扩展） |

## L0→L1（`gdn-to-chunk` pass）

当前：入口调用式占位（自动生成 `func.private @__gdn_l1_fwd_chunk/__gdn_l1_bwd_chunk`
声明，`gdn.config` 属性原样透传）。完整 chunk-AST 展开随 G2 落地，输出与
`tools/gdn_l1_gen.py` 生成的 L1 文本逐行同构（前代换解三角、scf.for 载态已验证）。
