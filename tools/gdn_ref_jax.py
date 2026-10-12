"""
GDN-MLIR G1-A -- JAX executable specification of the Gated DeltaNet operator
(gated delta rule), ported line by line from the fla oracle:

    ~/codebase/fla/fla/ops/gated_delta_rule/naive.py
        naive_recurrent_gated_delta_rule  ->  gdn_seq
        naive_chunk_gated_delta_rule      ->  gdn_chunk

CPU-only, fp64 (jax_enable_x64), jittable and differentiable (jax.vjp).

Shapes (fla op contract):
    q      [B, T, H, K]    queries
    k      [B, T, H, K]    keys
    v      [B, T, H, V]    values
    beta   [B, T, H]       delta-rule strength (RAW: the GDN model applies
                           sigmoid before calling; the op uses it as-is)
    g      [B, T, H]       per-token log-decay, NATURAL-log space
    h0     [B, H, K, V]    optional initial state
    returns (o [B, T, H, V], ht [B, H, K, V])

Conventions ported faithfully from naive.py (do NOT "fix" these):
  * scale: default None -> K ** -0.5 (naive.py: 1 / (q.shape[-1] ** 0.5)),
    folded into q up front.  q is NOT L2-normalized inside the op; the GDN
    model l2-normalizes q (and k) before calling the op.
  * beta is used raw (no sigmoid inside); it multiplies (v_t - k_t @ h).
  * g decays the state multiplicatively as exp(g_t) -- e-base, NOT 2**g_t.
    (The fla Triton path pre-scales g by RCP_LN2 = log2(e) -- see
    fla/ops/gated_delta_rule/chunk.py chunk_local_cumsum(..., scale=RCP_LN2)
    -- only so the kernels can use exp2; an MLIR implementation that uses
    exp2 must compute exp2(g * 1.4426950408889634).)
  * sequential per step, in this exact order (decay BEFORE the delta update):
        h   <- exp(g_t) * h
        bv  <- beta_t * (v_t - k_t @ h)
        h   <- h + k_t (x) bv
        o_t <- q_t @ h
  * chunked: inclusive cumsum of g WITHIN each chunk; strictly-lower
    M[i,j] = -beta_i * (k_i . k_j) * exp(cd_i - cd_j); T = (I + M)^(-1)
    built by the same row-substitution recurrence naive.py uses; then
    w = T @ (beta*v), u = T @ (beta*k*exp(cd)), per chunk:
        v_prime = u @ S
        v_new   = w - v_prime
        o_i     = (q_i * exp(cd)) @ S  +  tril(q_i k_i^T * exp(cd_i-cd_j)) @ v_new
        S       <- exp(cd_last) * S + (k_i * exp(cd_last - cd_i))^T @ v_new
    Padding of the tail chunk uses zeros (beta=0, g=0 rows are no-ops).

Intentional, non-mathematical differences vs the torch oracle:
  * The oracle hard-casts all inputs to float32 internally; we keep the
    caller's dtype (the spec is run in fp64).
  * The final state is always returned (oracle gates it behind
    output_final_state).
"""

from __future__ import annotations

import functools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


# --------------------------------------------------------------------------- #
# shared input generation (GDN-model-style): q,k l2-normalized, beta=sigmoid,
# g = -softplus(0.5*x) (negative log-decays), optional seeded h0, seeded
# cotangents do/dht for VJP / finite-difference checks.
# --------------------------------------------------------------------------- #
def make_gdn_inputs(key, B, T, H, K, V, with_h0=False, dtype=jnp.float64):
    kq, kk, kv, kb, kg, kh, kdo, kdht = jax.random.split(key, 8)
    q = jax.random.normal(kq, (B, T, H, K), dtype=dtype)
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)  # caller-side l2norm
    k = jax.random.normal(kk, (B, T, H, K), dtype=dtype)
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)  # caller-side l2norm
    v = jax.random.normal(kv, (B, T, H, V), dtype=dtype)
    beta = jax.nn.sigmoid(jax.random.normal(kb, (B, T, H), dtype=dtype))
    g = -jax.nn.softplus(0.5 * jax.random.normal(kg, (B, T, H), dtype=dtype))
    h0 = None
    if with_h0:
        h0 = 0.1 * jax.random.normal(kh, (B, H, K, V), dtype=dtype)
    do = jax.random.normal(kdo, (B, T, H, V), dtype=dtype)
    dht = jax.random.normal(kdht, (B, H, K, V), dtype=dtype)
    return dict(q=q, k=k, v=v, beta=beta, g=g, h0=h0, do=do, dht=dht)


def _zeros_h0(q, v, h0):
    if h0 is not None:
        return h0
    B, _, H, K = q.shape
    V = v.shape[-1]
    return jnp.zeros((B, H, K, V), dtype=q.dtype)


# --------------------------------------------------------------------------- #
# (a) sequential spec: port of naive_recurrent_gated_delta_rule
# --------------------------------------------------------------------------- #
def gdn_seq(q, k, v, beta, g, scale=None, h0=None):
    """Sequential (lax.scan over T) gated delta rule.

    q,k: [B,T,H,K], v: [B,T,H,V], beta,g: [B,T,H], h0: [B,H,K,V] or None.
    Returns (o [B,T,H,V], ht [B,H,K,V]).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5  # naive.py: 1 / (q.shape[-1] ** 0.5)
    h0 = _zeros_h0(q, v, h0)
    q = q * scale

    # [B,T,H,*] -> [T,B,H,*] for lax.scan over the leading axis
    q_ = jnp.transpose(q, (1, 0, 2, 3))
    k_ = jnp.transpose(k, (1, 0, 2, 3))
    v_ = jnp.transpose(v, (1, 0, 2, 3))
    beta_ = jnp.transpose(beta, (1, 0, 2))
    g_ = jnp.transpose(g, (1, 0, 2))

    def step(h, inputs):
        q_t, k_t, v_t, beta_t, g_t = inputs
        # h = h.clone() * g_i.exp()[..., None, None]
        h = h * jnp.exp(g_t)[..., None, None]
        # b_v = v_i - (h * k_i[..., None]).sum(-2)   ==  v_i - k_i @ h
        b_v = v_t - jnp.einsum("bhk,bhkv->bhv", k_t, h)
        # b_v = b_v * beta_i[..., None]
        b_v = b_v * beta_t[..., None]
        # h = h + k_i.unsqueeze(-1) * b_v.unsqueeze(-2)
        h = h + k_t[..., :, None] * b_v[..., None, :]
        # o_i = einsum('bhd,bhdm->bhm', q_i, h)
        o_t = jnp.einsum("bhk,bhkv->bhv", q_t, h)
        return h, o_t

    ht, o = jax.lax.scan(step, h0, (q_, k_, v_, beta_, g_))
    o = jnp.transpose(o, (1, 0, 2, 3))  # [T,B,H,V] -> [B,T,H,V]
    return o, ht


# --------------------------------------------------------------------------- #
# (b) chunked WY-representation spec: port of naive_chunk_gated_delta_rule
# --------------------------------------------------------------------------- #
def gdn_chunk(q, k, v, beta, g, scale=None, h0=None, chunk_size=64,
              output_per_chunk_states=False):
    """Chunked WY-representation forward (port of naive_chunk_gated_delta_rule).

    Returns (o [B,T,H,V], ht [B,H,K,V]) or, with output_per_chunk_states=True,
    (o, ht, states [B,H,n,K,V]) where states[:,:,i] is the state AFTER chunk i.
    """
    BT = int(chunk_size)
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    # naive.py transposes to [B,H,T,*] then pads the tail chunk with zeros
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    beta = jnp.transpose(beta, (0, 2, 1))
    g = jnp.transpose(g, (0, 2, 1))

    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_len)))
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_len)))

    L = T + pad_len
    n = L // BT
    c = BT

    q = q * scale
    v = v * beta[..., None]  # beta-scaled values
    k_beta = k * beta[..., None]  # beta-scaled keys

    # chunkify: [B,H,L,*] -> [B,H,n,c,*]; decay = within-chunk inclusive cumsum
    qc = q.reshape(B, H, n, c, K)
    kc = k.reshape(B, H, n, c, K)
    vc = v.reshape(B, H, n, c, V)
    k_beta_c = k_beta.reshape(B, H, n, c, K)
    decay = jnp.cumsum(g.reshape(B, H, n, c), axis=-1)  # inclusive
    e_decay = jnp.exp(decay)  # [B,H,n,c]

    # L_mask[i,j] = exp(cd_i - cd_j) for i >= j else 0
    diff = decay[..., :, None] - decay[..., None, :]
    L_mask = jnp.tril(jnp.exp(jnp.tril(diff)))

    # attn = -((k_beta @ k^T) * L_mask), diagonal and above zeroed:
    # M[i,j] = -beta_i (k_i . k_j) exp(cd_i - cd_j)  (strictly lower)
    kk = jnp.matmul(k_beta_c, jnp.swapaxes(kc, -1, -2))  # [B,H,n,c,c]
    attn = -(kk * L_mask)
    triu_incl_diag = jnp.triu(jnp.ones((c, c), dtype=bool), 0)
    attn = jnp.where(triu_incl_diag, 0.0, attn)

    # T = (I + M)^(-1): the exact row-substitution recurrence from naive.py
    # (attn rows are updated in increasing i, reading already-updated rows j<i)
    for i in range(1, c):
        attn = attn.at[..., i, :i].add(
            jnp.sum(attn[..., i, :i, None] * attn[..., :i, :i], axis=-2)
        )
    attn = attn + jnp.eye(c, dtype=q.dtype)

    # w = T @ (beta*v);  u = T @ (beta*k*exp(cd))
    w = jnp.matmul(attn, vc)  # [B,H,n,c,V]
    u = jnp.matmul(attn, k_beta_c * e_decay[..., None])  # [B,H,n,c,K]

    S = h0 if h0 is not None else jnp.zeros((B, H, K, V), dtype=q.dtype)
    triu_strict = jnp.triu(jnp.ones((c, c), dtype=bool), 1)  # keep diagonal
    o_chunks = []
    states = []
    for i in range(n):
        q_i = qc[:, :, i]
        k_i = kc[:, :, i]
        w_i = w[:, :, i]
        u_i = u[:, :, i]
        cd = decay[:, :, i]  # [B,H,c]
        e_cd = e_decay[:, :, i]
        d_last = cd[..., -1]  # [B,H]

        # attn_i[i,j] = (q_i . k_j) * exp(cd_i - cd_j) for i >= j (diagonal kept)
        a_i = jnp.matmul(q_i, jnp.swapaxes(k_i, -1, -2)) * L_mask[:, :, i]
        a_i = jnp.where(triu_strict, 0.0, a_i)

        v_prime = jnp.matmul(u_i, S)  # [B,H,c,V]
        v_new = w_i - v_prime
        o_inter = jnp.matmul(q_i * e_cd[..., None], S)
        o_i = o_inter + jnp.matmul(a_i, v_new)

        # S <- exp(cd_last) * S + (k_i * exp(cd_last - cd_i))^T @ v_new
        k_term = k_i * jnp.exp(d_last[..., None] - cd)[..., None]  # [B,H,c,K]
        S = S * jnp.exp(d_last)[..., None, None] + jnp.matmul(
            jnp.swapaxes(k_term, -1, -2), v_new
        )
        o_chunks.append(o_i)
        states.append(S)

    o = jnp.stack(o_chunks, axis=2)  # [B,H,n,c,V]
    o = o.reshape(B, H, n * c, V)[:, :, :T]  # unpad
    o = jnp.transpose(o, (0, 2, 1, 3))  # [B,H,T,V] -> [B,T,H,V]

    if output_per_chunk_states:
        return o, S, jnp.stack(states, axis=2)  # [B,H,n,K,V]
    return o, S


# --------------------------------------------------------------------------- #
# (d) VJP wrappers: (o, ht, dq, dk, dv, dbeta, dg, dh0)
# --------------------------------------------------------------------------- #
def _vjp_common(fn, q, k, v, beta, g, h0, do, dht, **kw):
    h0 = _zeros_h0(q, v, h0)  # differentiate w.r.t. zeros when h0 is absent

    def f(q_, k_, v_, beta_, g_, h0_):
        return fn(q_, k_, v_, beta_, g_, h0=h0_, **kw)

    (o, ht), vjp_fn = jax.vjp(f, q, k, v, beta, g, h0)
    if do is None:
        do = jnp.ones_like(o)
    if dht is None:
        dht = jnp.zeros_like(ht)
    dq, dk, dv, dbeta, dg, dh0 = vjp_fn((do, dht))
    return o, ht, dq, dk, dv, dbeta, dg, dh0


def gdn_seq_vjp(q, k, v, beta, g, scale=None, h0=None, do=None, dht=None):
    """VJP of the sequential spec w.r.t. (q,k,v,beta,g,h0), cotangents (do,dht)."""
    return _vjp_common(gdn_seq, q, k, v, beta, g, h0, do, dht, scale=scale)


def gdn_chunk_vjp(q, k, v, beta, g, scale=None, h0=None, chunk_size=64,
                  do=None, dht=None):
    """VJP of the chunked spec w.r.t. (q,k,v,beta,g,h0), cotangents (do,dht)."""
    return _vjp_common(gdn_chunk, q, k, v, beta, g, h0, do, dht,
                       scale=scale, chunk_size=int(chunk_size))
