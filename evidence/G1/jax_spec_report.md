# G1-A: JAX executable spec of Gated DeltaNet — results & conventions

Workstream A of milestone G1 (GDN-MLIR). Everything CPU-only, fp64 (`jax_enable_x64`).

- Spec: `tools/gdn_ref_jax.py`
  - `gdn_seq(q,k,v,beta,g, scale=None, h0=None)` — sequential per-token recurrence (`lax.scan`), port of `naive_recurrent_gated_delta_rule`
  - `gdn_chunk(q,k,v,beta,g, scale=None, h0=None, chunk_size=64, output_per_chunk_states=False)` — chunked WY forward, port of `naive_chunk_gated_delta_rule`; returns `(o, ht)` or `(o, ht, states[B,H,n,K,V])` (state after each chunk)
  - `gdn_seq_vjp` / `gdn_chunk_vjp` — return `(o, ht, dq, dk, dv, dbeta, dg, dh0)` via `jax.vjp` with cotangents `(do, dht)`
- Oracle (ground truth, read-only): `~/codebase/fla/fla/ops/gated_delta_rule/naive.py`
- Checker: `tools/gdn_ref_check.py` (exit 0 iff all PASS); Dumps: `tools/gdn_ref_dump.py`

## 1. Results (tools/gdn_ref_check.py, jax 0.11.2 CPU, 2026-10-08)

Shapes B=2, H=HV=2, K=V=16, fp64, seeded (`PRNGKey(12345)`); inputs GDN-style
(q,k l2-normalized, beta=sigmoid(x), g=-softplus(0.5x)). 44/44 PASS.

| Check (worst over combos) | achieved | tol | result |
|---|---|---|---|
| chunked vs sequential, o (T=96/128, C=32/64, h0 y/n) | 5.6e-16 | 1e-10 | PASS |
| chunked vs sequential, ht (same 8 combos) | 1.2e-15 | 1e-10 | PASS |
| VJP grads chunked vs sequential (dq,dk,dv,dbeta,dg,dh0; 8 combos) | 1.2e-14 | 1e-8 | PASS |
| finite-diff vs VJP, 6 random coords (seq, T=96 C=64 h0) | 2.3e-09 | 1e-5 | PASS |
| finite-diff vs VJP, 6 random coords (chunk, T=96 C=64 h0) | 9.0e-10 | 1e-5 | PASS |
| jit parity seq (T=96) | 0.0 | 1e-12 | PASS |
| jit parity chunk (T=96 C=64) | 1.1e-16 | 1e-12 | PASS |
| torch fla `naive_recurrent` vs jax seq, o+ht (fp32, rel) | 1.2e-07 | 1e-4 | PASS |
| torch fla `naive_chunk` vs jax chunk, o+ht (fp32, rel) | 2.3e-06 | 1e-4 | PASS |

T=96 with C=64 exercises the zero-padded tail chunk (96→128, 32 pad rows);
T=96/C=32, T=128/C=32, T=128/C=64 are exact multiples. The torch cross-check is
fp32 because fla's naive.py hard-casts all inputs/states to float32 internally
(fp64 torch comparison is impossible without modifying the read-only oracle).

## 2. Reference artifacts (consumed by workstream B)

`evidence/G1/` — all B=2, T=128, H=2, K=16, V=16, fp64 compute, seed
`PRNGKey(20261008)`; primary arrays fp32, identical fp64 copies under
`<name>_fp64`, plus a `config` JSON key:

| file | chunk | h0 | chunk-vs-seq (fp64) |
|---|---|---|---|
| `ref_jax_T128_C64.npz` | 64 | zeros | 8.9e-16 |
| `ref_jax_T128_C32.npz` | 32 | zeros | 4.4e-16 |
| `ref_jax_T128_C64_h0.npz` | 64 | seeded (max 0.31) | 8.9e-16 |

npz keys per file (33): inputs `q,k,v,beta,g,h0,do,dht` (do/dht = seeded VJP
cotangents), outputs `o,ht`, grads `dq,dk,dv,dbeta,dg,dh0` — each also as
`*_fp64`. Shapes: q/k [2,128,2,16], v/o [2,128,2,16], beta/g [2,128,2],
h0/ht/dh0 [2,2,16,16]. Grads are of `L = sum(do*o) + sum(dht*ht)`.
The three files share identical q,k,v,beta,g,do,dht (only h0 differs), so
MLIR-vs-JAX diffs are directly comparable across chunk sizes. fp32-cast
rounding of the stored values is ~1.4e-8 abs (max|value| ~ 8.2).

## 3. Conventions the MLIR side MUST replicate

Documented from `fla/fla/ops/gated_delta_rule/naive.py` (do not "fix" these;
fidelity to fla is the contract):

- **Shapes**: q,k [B,T,H,K]; v [B,T,H,V]; beta [B,T,H]; g [B,T,H]; h [B,H,K,V];
  o [B,T,H,V]. beta/g are per (b,t,head) — one head axis (H=HV in GDN; the op
  itself never mixes heads).
- **scale**: default `None -> K**-0.5` (naive.py: `1 / (q.shape[-1] ** 0.5)`),
  folded into q *before* the recurrence. NOTE: the op does NOT L2-normalize q —
  the GDN model l2-normalizes q (and k) before calling. Our dumps contain
  already-normalized q,k.
- **beta convention**: raw value used as-is (model applies sigmoid before the
  op). Multiplies the correction: `b_v = beta * (v - k @ h)`; in the chunked
  path it scales the k rows of the `(I+M)^-1` system asymmetrically
  (`M ∝ beta_i * k_i·k_j`, row-wise beta) and scales v globally (`v*beta`).
- **g convention — the big one**: g is a **natural-log** per-token decay;
  multiplier is `exp(g)` (e-base). The fla *Triton* path pre-scales
  `g *= RCP_LN2 = log2(e) = 1.4426950408889634`
  (`fla/ops/gated_delta_rule/chunk.py`: `chunk_local_cumsum(..., scale=RCP_LN2)`)
  so kernels can call `exp2`. An MLIR kernel using `exp2` must apply
  `exp2(g * RCP_LN2)`, NOT `exp2(g)`. In the *sequential* op g is applied
  step-wise: `h <- exp(g_t) * h` BEFORE the delta update (order matters).
- **Chunk-local g cumsum**: the chunked variant uses an *inclusive* cumsum of g
  within each chunk only (`cd = cumsum(g_chunk)`); inter-chunk decay enters via
  the `exp(cd_last)` factor on S and `exp(cd_last - cd_i)` on the k rows of the
  state update. Global (cross-chunk) cumsum would be wrong.
- **(I+M)^-1, not (I-A)**: strictly-lower `M[i,j] = -beta_i (k_i·k_j)
  exp(cd_i - cd_j)` (i>j); naive.py builds `T = (I+M)^-1` by forward row
  substitution (`attn[i,:i] += Σ_j attn[i,j]·attn[j,:i]`, rows in increasing i).
  `w = T @ (beta*v)`, `u = T @ (beta*k*exp(cd))`.
- **Intra-chunk output**: `o_i = (q_i*exp(cd)) @ S + tril_incl_diag(q_i k_i^T ⊙
  exp(cd_i-cd_j)) @ (w - u@S)` — note the qk attention here KEEPS the diagonal
  and applies the same `exp(cd_i-cd_j)` factors; only strictly-upper is masked.
- **State update per chunk**: `S <- exp(cd_last)*S + (k_i ⊙
  exp(cd_last-cd_i))^T @ (w - u@S)` (S on the right side is the pre-update S).
- **Tail padding**: non-multiple T is zero-padded to a chunk multiple (q,k,v,
  beta,g all zero rows — beta=0/g=0 make pad rows no-ops); output rows `>=T`
  are sliced off. Chunk count `n = ceil(T/C)`.
- **Quirks to be aware of** (kept as-is):
  - fla's `naive_chunk_gated_delta_rule` signature order is `(q,k,v,g,beta,...)`
    while `naive_recurrent_gated_delta_rule` is `(q,k,v,beta,g,...)` — easy to
    transpose beta/g by accident when calling the torch oracle.
  - naive.py hard-casts everything to fp32 internally and gates the final
    state behind `output_final_state`; our spec keeps caller dtype (fp64) and
    always returns `(o, ht)`.

## 4. Repro

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate gdn
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= \
       XLA_FLAGS=--xla_force_host_platform_device_count=1
cd ~/codebase/gdn-mlir
python tools/gdn_ref_check.py   # 44/44 PASS
python tools/gdn_ref_dump.py    # rewrites evidence/G1/ref_jax_*.npz
```
