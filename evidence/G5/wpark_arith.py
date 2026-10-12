#!/usr/bin/env python3
"""r24b: the W-park pool/traffic arithmetic (CPU-only, exact integers).

All numbers derived from the shipped dqkwg v4 geometry (C=64 K=128 V=128
BK=32 BV=32, P11R=2) and the measured calibrations:
  pool 47,616 B (r15) | 2-CTA bar 50,176 B | DRAM 824.6 GB/s,
  L2-resident 2,144 GB/s (static-r7, co-loaded lower bounds).
Prints the decision table used in bwd_wpark_decision_memo.md.
"""

C, K, V, BK, BV = 64, 128, 128, 32, 32
KS = K // BK          # 4 ks blocks
VS = V // BV          # 4 vs blocks
POOL = 47_616
BAR2CTA = 50_176
DRAM = 824.6e9        # B/s (co-loaded)
L2 = 2144e9           # B/s (co-loaded, resident)
DQKWG = {2048: 514.4e-6, 8192: 1728.9e-6}     # r22 medians (s)
E2E = {2048: 2.16e-3, 8192: 7.68e-3}          # r18-final chain (s)
M = {2048: 16 * 32, 8192: 16 * 128}           # BH*NC CTAs

tile_ck = C * BK                    # elems per [C,BK] park tile
park_f32 = KS * tile_ck * 4         # dw-only 4-set f32 park
park_bf16 = KS * tile_ck * 2

print("== pool")
print(f"ship pool                 {POOL} B")
print(f"2-CTA bar                 {BAR2CTA} B   headroom {BAR2CTA-POOL} B")
print(f"smem park dw-only f32     {park_f32} B  -> pool {POOL+park_f32}"
      f"  ({'FIT' if POOL+park_f32 <= BAR2CTA else 'DEAD'} vs 2-CTA bar)")
print(f"smem park dw-only bf16    {park_bf16} B -> pool {POOL+park_bf16}"
      f"  ({'FIT' if POOL+park_bf16 <= BAR2CTA else 'DEAD'} vs 2-CTA bar)")
print(f"smem park dw+dq1 f32      {2*park_f32} B -> DEAD")
print(f"accCK single-window reuse {tile_ck*4} B alone > headroom"
      f" {BAR2CTA-POOL} B -> cannot host even ONE park set")

sweep = C * BV * 2 + C * BV * 4     # akb bf16 read + dvp f32 read
sweep_sm = C * BV * 2 + 2 * C * BV * 2   # akb smem + vnh/vnl smem
print("\n== staging sweeps (per CTA per chunk)")
print(f"vs-only sweep global reads {sweep} B (akb {C*BV*2} + dvp {C*BV*4})")
print(f"vs-only sweep smem writes  {sweep_sm} B")
print(f"P11R=2 sweeps {2*VS}  P11R=4 sweeps {VS}")
saved_g = (2 * VS - VS) * sweep
saved_sm = (2 * VS - VS) * sweep_sm
print(f"saved by the full kill    {saved_g} B global reads,"
      f" {saved_sm} B smem writes")

print("\n== global park traffic (park-resume, per CTA per chunk)")
p_f32 = KS * (3 * tile_ck * 4 + 3 * tile_ck * 4)   # 3W+3R of [C,BK]f32
p_bf16 = p_f32 // 2
print(f"f32 park   +{p_f32} B   net vs saved {p_f32-saved_g:+} B")
print(f"bf16 park  +{p_bf16} B   net vs saved {p_bf16-saved_g:+} B")

print("\n== time pricing (per shape)")
p11r12 = {2048: (544.5 - 523.6), 8192: (1823.3 - 1755.6)}  # us, r16
for T in (2048, 8192):
    add = M[T] * (p_f32 - saved_g)
    addb = M[T] * (p_bf16 - saved_g)
    t_l2, t_dr = add / L2, add / DRAM
    tb_l2, tb_dr = addb / L2, addb / DRAM
    ceil_us = p11r12[T] / 2          # half the sweep kill, park FREE
    print(f"T{T}: f32 park net traffic +{add/1e6:.1f} MB"
          f" -> +{t_l2*1e6:.0f}us (L2) / +{t_dr*1e6:.0f}us (DRAM)"
          f" on dqkwg {DQKWG[T]*1e6:.0f}us"
          f" = {DQKWG[T]/(DQKWG[T]+t_l2):.3f}x/{DQKWG[T]/(DQKWG[T]+t_dr):.3f}x")
    print(f"     bf16 park net +{addb/1e6:.1f} MB"
          f" -> +{tb_l2*1e6:.0f}us (L2) / +{tb_dr*1e6:.0f}us (DRAM)"
          f" = {DQKWG[T]/(DQKWG[T]+tb_l2):.3f}x/{DQKWG[T]/(DQKWG[T]+tb_dr):.3f}x")
    print(f"     FREE-park ceiling: P11R 2->4 ~= half the r16 1->2 gain"
          f" = -{ceil_us:.0f}us -> dqkwg {DQKWG[T]*1e6-ceil_us:.0f}us"
          f" ({DQKWG[T]/(DQKWG[T]-ceil_us/1e6):.3f}x),"
          f" e2e {E2E[T]*1e3:.2f}->{E2E[T]*1e3-ceil_us/1e3:.2f}ms"
          f" ({(E2E[T]-ceil_us/1e6)/E2E[T]:.4f}x)")
