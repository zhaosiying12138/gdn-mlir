# G1-B: L1 chunked-GDN as real MLIR (scf+linalg) on CPU — harness & results

Workstream B of milestone G1 (GDN-MLIR). The CHUNKED gated-delta-rule forward
algorithm now runs as REAL MLIR: a parameterized Python generator emits a
self-contained `linalg` + `scf` (memref-level) module, which is lowered with
`mlir-opt` and executed with `mlir-runner` on CPU, and is verified numerically
against an embedded numpy reference plus the G1-A JAX dumps. CPU only, no GPU,
no ninja, single-process tools throughout.

**G1-B2 update**: the generator now also emits the BACKWARD pass
(`--emit bwd|fwd|both`, default `fwd` — the fwd output is byte-identical to
G1-B1's), verified against the JAX vjp gradients in the G1-A dumps (see the
`Backward (G1-B2)` section at the bottom).

## Deliverables

| artifact | what it is |
|---|---|
| `tools/gdn_l1_gen.py` | generator: emits the whole .mlir module (data embedded as `memref.global` constants, kernels `@gdn_fwd` and/or `@gdn_bwd`, `@main` printing `o`, `ht` and/or `dq,dk,dv,dbeta,dg,dh0` via `printMemrefF32`). Flags: `--shape B,T,H,K,V --chunk C --seed N --emit fwd\|bwd\|both --npz <ref.npz> --out <f.mlir>` (stdout default). This IS the output format of the future L1 expansion pass. |
| `scripts/run_l1.sh` | lowering + execution harness: `run_l1.sh <in.mlir> [out.mlir]`. |
| `tools/gdn_l1_check.py` | numeric self-check driver (numpy chunked + sequential fwd references, numpy bwd twin, memref parser; fwd gate < 1e-3; `--bwd` gate < 2e-4). |
| `evidence/G1/l1_chunked_T128_C64.mlir` | representative generated fwd module (B1 T128 H2 K16 V16 C64, seed 0, 2167 lines). |
| `evidence/G1/l1_check_output.txt` | captured fwd checker output (ALL PASS). |
| `evidence/G1/l1_bwd_chunked_T128_C64.mlir` | representative generated bwd module (B1 T128 H2 K16 V16 C64, seed 0, 3540 lines). |
| `evidence/G1/l1_bwd_check_output.txt` | captured bwd checker output (ALL PASS). |

## Working pipeline (exact commands)

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

Pass-spelling findings (this tree is Oct 2026 main):

- The task-suggested pipeline is missing two passes that are REQUIRED for the
  strided rank-reduced subviews the generator emits: `-expand-strided-metadata`
  and `-lower-affine` (recipe copied from in-tree
  `mlir/test/Integration/Dialect/Linalg/CPU/rank-reducing-subview.mlir`).
- `math.exp2` lowers via `-convert-math-to-llvm` to `llvm.intr.exp2`, which the
  JIT resolves against libm pulled in through the shared runtime libs — no
  `-convert-math-to-libm` needed (verified against
  `mlir/test/Conversion/MathToLLVM/math-to-llvm.mlir`).
- `-one-shot-bufferize="bufferize-function-boundaries"` is a no-op on this IR
  (it is already memref-level) but runs cleanly and is kept for pipeline
  compatibility with tensor-based frontends later.
- `printMemrefF32` needs only `func.func private @printMemrefF32(memref<*xf32>)`
  + `memref.cast` to `memref<*xf32>` (from libmlir_c_runner_utils).
- This build's parser is strict about literals: dense float lists require a
  decimal point / exponent (`1` is an error, `1.0` is fine) and flat (flattened)
  dense element lists are REJECTED for shaped types — the generator emits fully
  nested literals.
- `memref.subview` stride lists are RELATIVE multipliers of the source strides
  (not absolute strides): `[1,1,1,1]` for contiguous chunk selection.

Runtime: mlir-opt 0.32 s, mlir-runner 0.22 s (JIT + run) for the base module;
lowered LLVM-dialect output 2088 lines.

## Numeric results (tools/gdn_l1_check.py, 2026-10-08)

Gate: max abs diff < 1e-3 for both `o` and final state `ht` (fp32 CPU).
All diffs sit at the printMemrefF32 printing-precision floor (~1e-6).

| case | shape | max abs diff o | max abs diff ht | numpy chunk-vs-seq | verdict |
|---|---|---|---|---|---|
| base | B1 T128 H2 K16 V16 C64 | 6.184e-07 | 8.643e-07 | 2.384e-07 | PASS |
| chunk32 | B1 T128 H2 K16 V16 C32 | 5.513e-07 | 7.153e-07 | 1.192e-07 | PASS |
| tail_t96 | B1 T96 H2 K16 V16 C64 (tail 64+32) | 5.662e-07 | 6.258e-07 | 2.384e-07 | PASS |
| tail_b2 | B2 T100 H2 K16 V16 C64 (tail 64+36) | 5.960e-07 | 3.099e-06 | 3.874e-07 | PASS |
| jax_C64_h0 | B2 T128 C64, G1-A npz, seeded h0 | 6.407e-07 | 8.643e-07 | 5.662e-07 | PASS |
| jax_C64 | B2 T128 C64, G1-A npz, h0=0 | 6.407e-07 | 8.643e-07 | 5.662e-07 | PASS |
| jax_C32 | B2 T128 C32, G1-A npz, h0=0 | 5.066e-07 | 5.513e-07 | 5.066e-07 | PASS |

- Seeded cases compare MLIR vs the embedded numpy chunked reference; the
  "chunk-vs-seq" column is the numpy chunked-vs-sequential cross-check (port of
  `naive_recurrent_gated_delta_rule`), which validates the chunked derivation
  itself before any MLIR is involved.
- `jax_*` cases embed the G1-A dumps' inputs (`evidence/G1/ref_jax_T128_*.npz`,
  fp64 compute, fp32 cast) and compare the MLIR output against the dump's own
  `o`/`ht` — i.e. G1-B matches G1-A end to end. (`o` differs by 2.9e-2 between
  the h0/no-h0 dumps, so the h0 path is genuinely exercised; `ht` is identical
  in both only because exp(sum g) over 128 steps decays the h0 contribution
  below fp32 resolution — expected, not a bug.)

## Math conventions (match G1-A's jax_spec_report.md / fla naive.py)

Oracle: `~/codebase/fla/fla/ops/gated_delta_rule/naive.py`
(`naive_chunk_gated_delta_rule`). Everything below is implemented per chunk of
size C, per (b, h), with state S [K,V]:

- shapes: q,k [B,T,H,K]; v [B,T,H,V]; beta,g [B,T,H]; **h0/ht [B,H,K,V]**;
  o [B,T,H,V].
- g is a NATURAL-log per-token decay in the fla/G1-A contract. The L1 KERNEL
  works log2-domain like fla's Triton path: the generator embeds
  `g_log2 = g * log2(e) = g * 1.4426950408889634` and all MLIR decay factors
  are `math.exp2` of that — `exp2(g*RCP_LN2) == exp(g)` exactly.
- gc = INCLUSIVE chunk-local cumsum of g (never cross-chunk).
- scale = K^-1/2 belongs to q; q enters only the (linear) output terms, so the
  kernel folds it into the final store: `o = (o_inter + o_intra) * scale`.
- A (the task's sign convention) = strictly-lower
  `A[i,j] = beta_i * (k_i . k_j) * 2^(gc_i - gc_j)` (j<i). This is `-M` with M
  the strictly-lower matrix of naive.py / the G1-A report.
- T = (I + A)^-1 == (I - M)^-1 by EXPLICIT FORWARD SUBSTITUTION (the
  `solve variant=forward-substitution` knob): triple `scf.for` (rows i, cols
  j<i, reduction k in [j,i)): `T[i,i]=1; T[i,j] = -sum_k A[i,k]*T[k,j]`.
  Algebraically identical to naive.py's row-recurrence `attn` loop
  (`attn[i,:i] += sum attn[i,k]*attn[k,:i]` computes (I-M)^-1); verified by
  the chunk-vs-seq column above (and by G1-A's torch cross-checks).
- u = T @ (beta .* v)  [fla `k_cumsum`];  w = T @ (beta .* k .* 2^gc)
  [fla `k_cumdecay`];  v_new = u - w @ S.
- o_intra = tril_INCL_diag((q @ k^T) .* 2^(gc_i - gc_j)) @ v_new — the qk
  attention keeps the diagonal (only strictly-upper masked).
- o_inter = (q .* 2^gc) @ S;  o = (o_inter + o_intra) * scale.
- S <- 2^gc_last * S + (k .* 2^(gc_last - gc))^T @ v_new (pre-update S on the
  right side; decay applied before the delta update in the sequential view).
- Tail chunks: naive.py zero-padding along T to NC*C (beta=k=v=g=0 pad rows are
  exact no-ops in A/u/S; o pad rows discarded by the consumer).

## L1 structure of the generated kernel (design choices)

`func.func @gdn_fwd(q,k,v,beta,g: <BxTpx...>, h0: <BxHxKxV>) -> (o, ht)`:

- `scf.for` over b, h, then over chunk n; state S is a per-(b,h)
  `memref.alloc` [K,V] mutated in place (init via `linalg.copy` from an
  h0[b,h] rank-reduced subview; final state copied to ht[b,h]). The memref +
  in-place-mutation variant was chosen over tensor.insert_slice/yield because
  it needs no bufferization analysis and compiles cleanly — semantics first at
  L1; the tensor state-carrying form is a welcome L2 change.
- Chunk operands are strided rank-reduced subviews
  (`memref.subview` of `memref.subview`): q_c/k_c [C,K] stride [H*K,1],
  v_c/o_c [C,V] stride [H*V,1], beta_c/g_c [C] stride [H], so the kernel
  indexes straight into the embedded global layouts (no packing copies).
- Steps inside a chunk: (a) `scf.for` cumsum gc; (b) k k^T as a
  `linalg.generic` contraction + `linalg.generic` elementwise with
  `linalg.index`-based strict-tril mask and `math.exp2`; (c) forward
  substitution as explicit triple scf.for; (d) `linalg.matmul`s for u, w;
  (e) elementwise generics + matmuls for v_new, o_inter/o_intra, the scaled
  store into o's subview, and the S update
  (`S <- S*2^gc_last + (k .* 2^(gc_last-gc))^T @ v_new` with the transpose
  realized as a generic contraction over chunk rows).
- All matmul/contraction outs buffers are zero-filled first (linalg named ops
  accumulate into outs).

## Known limitations (L1)

- printMemrefF32 prints ~6 significant digits, so diffs below ~1e-6 are not
  resolvable through stdout; the true fp32 compute error vs the fp32 numpy
  reference is likely smaller than the measured ~6e-7.
- Memref copy semantics: state is updated in place via load/store through
  linalg; there is no tensor-level SSA history (fine at L1; matters for autodiff
  L3, where the WY backward from G1-A's VJP must be its own kernel anyway).
- Scratch buffers are malloc/free'd per (b,h) iteration (memref.alloc inside
  the b/h loops); a fused/persistent-buffer schedule is L2 territory.
- The forward-substitution solve is O(C^3/6) scalar loop iterations —
  deliberately explicit; the `solve variant` knob (e.g. row-recurrence form,
  Neumann series, blocked) can be A/B-tested by regenerating.
- fp32 only (bfloat16 comes later); B and H are looped, not tiled; no
  vectorization (linalg-to-loops emits scalar loops — vector dialect / SIMD is
  L2).

## Integration notes

- To consume G1-A dumps directly:
  `python tools/gdn_l1_gen.py --shape 2,128,2,16,16 --chunk 64 --npz evidence/G1/ref_jax_T128_C64_h0.npz --out x.mlir && scripts/run_l1.sh x.mlir`
  — embeds inputs AND expected o/ht globals (@ref_o/@ref_ht, also printed).
- The checker (`tools/gdn_l1_check.py`) auto-detects the G1-A npz files and
  adds them as cross-workstream cases; absent files are tolerated.
- Backward: `python tools/gdn_l1_gen.py --shape 2,128,2,16,16 --chunk 64
  --emit bwd --npz evidence/G1/ref_jax_T128_C64_h0.npz --out x.mlir &&
  scripts/run_l1.sh x.mlir` — embeds inputs AND cotangents (do/dht) plus the
  six expected-grad globals (@ref_dq..@ref_dh0, also printed).
  `python tools/gdn_l1_check.py --bwd` runs the G1-B2 gate.
- Environment: python/numpy from conda env `gdn`
  (`source ~/miniforge3/etc/profile.d/conda.sh && conda activate gdn`).

# Backward (G1-B2)

`tools/gdn_l1_gen.py --emit bwd|both` adds `func.func @gdn_bwd(q,k,v,beta,g,
h0,do,dht) -> (dq,dk,dv,dbeta,dg,dh0)` in the same memref+linalg+scf style
(~1.6x the fwd code size; all data embedded as `memref.global` dense, do/dht
zero time-padded like the fwd inputs). `--emit both` emits both kernels and
`@main` prints `o, ht` then the six grads.

## Structure (per b,h; the L1 "recompute-vs-materialize" knob = recompute)

- **Pass 1 (forward recompute)**: exact fwd chunk math, storing ONLY the
  per-chunk ENTRY states `h_all[n]` ([NC,K,V] scratch per (b,h)); o is not
  computed. Intermediates (gc, A, T, u, w) are discarded.
- **Pass 2 (reverse chunk walk)**: `scf.for %n2 = 0..NC` with
  `n = NC-1-n2` (positive step; avoids negative-step scf.for), carrying the
  reverse state cotangent `dH` seeded from `dht`. Per chunk it RECOMPUTES
  gc/A/T/u/w (same code as pass 1), reads `S_n = h_all[n]`, forms
  `vn = u - w S_n`, and:
  - `dO = do * scale` (scale folded at the leaf, mirror of the fwd o-store)
  - `dvp = P^T dO + Bhat dH` (total cotangent on v_new; Bhat = k e^(gcl-gc))
  - `dW = -(dvp S_n^T)`, `G = dvp (beta v)^T + dW (beta k e^gc)^T` (cotangent
    on T), then the dA adjoint below
  - input grads (see formulas below), then
    `dH <- e^gcl dH + (q e^gc)^T dO - w^T dvp`; after chunk 0, `dh0[b,h] = dH`.
- Tail chunks: zero-padded rows (beta=k=g=do=0) contribute exact zeros to
  A/u/P/dvp/G/dA and to every suffix/prefix/rowsum, so real rows are exact and
  pad rows read 0 (checked by the tail_t96/tail_b2 cases).

## The dA correction (T = (I+A)^-1 adjoint) — exact formula

With G = dL/dT, `dT = -T dA T` (inverse differential). Since T is unit lower
triangular and dA is strictly lower, dT is strictly lower, so only the
strict-lower part of G contributes — masking G FIRST is REQUIRED (diag/upper
G terms would otherwise leak into the strict-lower of the product through the
unit diagonal of T). The adjoint is:

```
dA = -strict_tril( T^T . strict_tril(G) . T^T )          [decayed coords]
dAf = dA .* e^(gc_i - gc_j)                               [decay-free coords]
```

`dAf` (fla's convention) moves the decay out of the k-chain:
`dk += (dAf @ k).*beta + dAf^T (beta k)` etc. `dAf` is computed by two
contractions (X = G_m T^T; Y = T^T X) plus one elementwise mask+negate.
Derivation cross-checked numerically: the task brief's one-sided variant
`dA.T + T^T (dA - diag dA)` does NOT reproduce the JAX vjp for C >= 3 (e.g.
C=3 gives `2 g21 + c g31` where the truth is `g21 + c g31 + c g32 a`); the
double-sided masked form above is exact (fp64 twin: all six grads <= 1.4e-14
vs the JAX vjp on all three npz dumps).

## Gradient formulas (per chunk; S = entry state, dH = exit cotangent)

```
dO   = do * scale
dq   = (dO S^T) .* e^gc + dPm @ k          dPm = (dO vn^T) .* e^(gc_i-gc_j) .* tril_inc
dk   = (dAf k) .* beta + dAf^T (beta k) + (T^T dW) .* (beta e^gc)
       + (vn dH^T) .* e^(gcl-gc) + dPm^T q
dv   = (T^T dvp) .* beta
dbeta= rows(dAf .* (k k^T)) + rows((T^T dW) .* (k e^gc))     <-- NO beta here
       + rows((T^T dvp) .* v)
dg_t = suffixsum_t[ (rows-cols)(dAf .* beta_i (k k^T))       <-- beta_i ON rows only
                    + rows((T^T dW) .* (beta k e^gc))
                    + rows((dO S^T) .* (q e^gc))
                    + (rows-cols)((dO vn^T) .* P) ]
     + sum_{c<t} rows((vn dH^T .* e^(gcl-gc)) .* k)          (strict prefix)
     + e^gcl sum(dH .* S)                                    (constant per chunk)
```

Two pitfalls found the hard way (both cost ~0.1-0.5 abs error on dg/dbeta):

1. The o_intra g-term pairs the RAW `(dO vn^T)` with `P` — NOT `dPm * P`
   (dPm already carries the decay e^(gc_i-gc_j); multiplying by P
   double-counts it). This equals fla's `rowsum(dq_loc q) - rowsum(dk_loc k)`
   q/k-route with `v = v_new`.
2. NO RCP_LN2 rescale on dg: every decay factor in the kernel is
   `exp2(gc_log2) == e^(gc_nat)`, so all pieces are already natural-log
   cotangents; a final `* log2(e)` is a bug (dg comes out scaled by 1.4427).

Also note dbeta's W-chain row is `k e^gc` (no beta) while dg's W-chain row is
`beta k e^gc` — the same reduction with and without the row factor, easy to
merge by accident.

## Numeric results (tools/gdn_l1_check.py --bwd, 2026-10-08)

Gate: max abs diff < 2e-4 vs the dump's fp32 grads (or < 3e-4 vs `*_fp64`),
plus finite checks. npz cells show `fp32/fp64` max abs diffs. All diffs sit at
the printMemrefF32 printing-precision floor (~6 significant digits).

| case | shape | dq | dk | dv | dbeta | dg | dh0 | verdict |
|---|---|---|---|---|---|---|---|---|
| jax_C64_h0 | B2 T128 C64, seeded h0 | 4.5e-6/4e-6 | 5.8e-6/6e-6 | 5.0e-6/5e-6 | 6.4e-6/7e-6 | 6.0e-6/6e-6 | 4.5e-7/5e-7 | PASS |
| jax_C64 | B2 T128 C64, h0=0 | 4.5e-6/4e-6 | 5.8e-6/6e-6 | 5.0e-6/5e-6 | 6.4e-6/7e-6 | 6.0e-6/6e-6 | 4.5e-7/5e-7 | PASS |
| jax_C32 | B2 T128 C32, h0=0 | 4.5e-6/4e-6 | 5.6e-6/6e-6 | 5.0e-6/5e-6 | 5.5e-6/5e-6 | 4.3e-6/4e-6 | 4.5e-7/5e-7 | PASS |

Seeded cases vs the fp32 numpy bwd twin (validates the tail-chunk path; the
twin itself is validated against the JAX vjp to <= 1.4e-14 in fp64):

| case | shape | worst grad diff |
|---|---|---|
| base | B1 T128 C64 | 6.0e-6 (dk) |
| chunk32 | B1 T128 C32 | 6.6e-6 (dg) |
| tail_t96 | B1 T96 C64 (tail 64+32) | 3.6e-5 (dbeta) |
| tail_b2 | B2 T100 C64 (tail 64+36) | 8.1e-6 (dk) |

All runs finite (no NaN/Inf); `python tools/gdn_l1_check.py` (fwd gate) still
ALL PASS after the changes and `--emit fwd` output is byte-identical to
G1-B1's generator (diff-clean).

## Deviations from fla's kernel-level decomposition

Same math, different distribution (fla = `fla/ops/common/{chunk_o,chunk_delta_h}.py`
+ `fla/ops/gated_delta_rule/wy_fast.py::prepare_wy_repr_bwd`):

1. **Bhat-chain dg term**: fla adds `sum(rowB)` to the chunk's LAST row and
   then applies one global reverse-cumsum, which algebraically equals
   `-suffix(rowB) + sum(rowB)` = the strict prefix `sum_{c<t} rowB[c]` used
   here. We emit the prefix directly (one scf.for with a running accumulator).
2. **dg per-row assembly**: fla computes `rowsum(dq_loc.*q) - rowsum(dk_loc.*k)`
   from the LOCAL (pre-W/A-chain) dq/dk and reverse-cumsums everything at the
   end; we assemble the same four row/col-sums explicitly (tA/tB vectors +
   one suffix loop). Verified equivalent to 1e-14 in the fp64 twin.
3. **dh storage**: fla stores `dh[n]` = gradient wrt the state LEAVING chunk n
   and reads it from the dqkwg kernel; our reverse recurrence carries `dH` as
   a register-like [K,V] buffer (no dh materialization, only `h_all` entry
   states from pass 1).
4. **fla quirk noted but not copied**: `prepare_wy_repr_bwd` loads the T
   inverse from a transposed storage layout, so its `dot(A, dA, A)` reads as
   `T^T G T^T` there — matching our formula exactly once the layout is
   accounted for.
5. Everything else is the same decomposition: dvp, dW, G, the masked
   double-sided dA correction, dAf decay-free coordinates, the dh reverse
   recurrence, and the recompute-w/u strategy (fla `recompute_w_u_fwd`).

## Notes for G2 (and known limitations)

- The bwd kernel allocates ~37 scratch buffers per (b,h) (vs 17 in fwd) and
  recomputes gc/A/T/u/w in pass 2; the materialize variant (store u,w,P per
  chunk from pass 1) is the obvious memory/compute trade to A/B in L2.
- The dA correction is two CxCxC contractions + mask per chunk; blocked/simd
  forms are L2.
- printMemrefF32 remains the only observation channel (stdout, ~6 significant
  digits); the fp32 twin-vs-MLIR agreement (~1e-6 to 4e-5) confirms the true
  fp32 compute error is at the print floor.
- The suffix/prefix dg loops and the reverse chunk loop all use positive
  scf.for steps with index remapping (`n = NC-1-n2`, `idx = C-1-i`) — negative
  steps were avoided deliberately.
