# G6: the failure longscroll — every honest negative, five rounds, with numbers

One section per recorded negative, project-wide. Each entry: what was
tried, the number, the lesson, the file. These are load-bearing paper
material: the wins (`knob_ablation.md`) are only interpretable against
this ledger of what did NOT work. GPU context everywhere: shared RTX 5090
Laptop, co-job resident, medians flagged `shared_gpu` in the source jsons.

---

## 1. G2/G5-D: block-uniform staging = 256× redundant copies (the r2 trap)

**Tried:** porting the G2-style generator-unrolled block-uniform staging
(every copy instruction executed by every thread) to the full-op K-loops.
**Number:** measured **2× SLOWER** than round 1 before the fix (staged fwd
regressed vs r1; 256× redundant cp.async issue on 256 threads).
**Lesson:** the G2 stage kernels are 4-warp micro-kernels where
block-uniform issue is affordable; at full-op scale the staging MUST be
thread-strided (each thread copies its own 16B chunks, transfer index
tid/iv-dependent). This interaction (knob correct at stage scale, toxic at
full-op scale) is exactly why the paper needs the two-level ablation.
**File:** `evidence/G5/fwd_report.md` round-2 §1.1.

## 2. G2/G5-D: `pipeline_shared_memory_copies` failures(suppress) is broken

**Tried:** letting the pipeliner tolerate non-copy loops via the
`failures(suppress)` clause (the documented upstream escape).
**Number:** the `-transform-interpreter` output dies on ANY non-copy
`scf.for` (probe-verified on a G2 payload + dummy loop).
**Lesson:** all non-K cell passes and the F2 chunk loop are `scf.while`
(which the transform does not match); with its own quirks (explicit ^bb0
header, condition yields ALL args). Related transform landmines recorded
alongside: multibuffer() hard-fails on buffers used in several loops (a
one-line dummy load+store pins them out); `create_async_groups`
invalidates handles (re-match loops after it); vector.transfer on workgroup
memrefs does not lower (reads AND writes).
**Files:** `evidence/G5/fwd_report.md` round-2 bugs §1–3; `evidence/G2/gen_report.md`
constraint 3–5.

## 3. G5-F r3: chunk ping-pong double-buffer = zero gain (and the wait_group-0 discovery)

**Tried:** the brief's planned lever 1 — manual `nvgpu.device_async_copy`
prefetch of chunk n+1 (w/k/gc into parity-indexed halves), gemm K-steps
fully unrolled, parity folded into index arithmetic.
**Number:** F2 471→473 µs. **Exactly nothing.**
**Lesson (two):** (a) the chunk chain was not load-latency-bound — the real
constraint was 8-way LDSM bank conflicts (see §4's sibling win: 473→249 µs
from +8-col padding alone); (b) machine constraint, source-verified in this
LLVM tree (`NVGPU/Transforms/CreateAsyncGroups.cpp`): every
`create_async_groups`-converted staging is IMMEDIATELY followed by
`cp.async.wait_group 0` — the transform route can structurally never
overlap a prefetch with compute; manual `device_async_copy` ops pass
through untouched and the wait placement is ours. The ping-pong structure
was kept (correct + no cost) and later carried the padding fix.
**File:** `evidence/G5/fwd_report.md` round-3 §1a.

## 4. G5-I r4: the segment-split scan — built, exact, refuted by Amdahl

**Tried:** the 2-pass segment split (`--f2-segments S`): pass A zero-seed
segment walks, pass C true-entry re-runs; math proven exact (matmul-free
handoff valid because every factor M_n carries the full chunk decay:
||M_seg|| ≤ 3.6e-13 at L=1, 4.3e-52 at L=4; T8192 o/ht BIT-IDENTICAL
across S=1/4/8).
**Number:** every shape SLOWER — B1T2048 staged 0.448 (S=1) vs 0.478 (S=4);
B1T8192 1.298 vs 1.541/1.507 (S=4/8) = **16–33% loss**. The Amdahl probe:
f2c alone at S=8 (16-chunk chains, 2 CTAs/SM) = 0.529 ms vs serial 0.640
→ chain length + co-residency together are worth **17% of F2**; the other
83% is per-chunk throughput, all of which the split duplicates.
**Lesson:** after r3, F2 is per-chunk-THROUGHPUT-bound, not
chain-latency-bound; no scan restructure can win at these shapes — the
~250 µs projection had modeled a latency-bound kernel that no longer
existed. (Validity boundary of the matmul-free split also documented:
near-unity gates (`--stress-g 0.02`) grow ||M_seg|| to 0.07–0.45 and would
breach the 5e-3 gate; the M-composition fallback was designed, not built.)
**File:** `evidence/G5/fwd_report.md` round-4 (§1, §4); `evidence/G5/g5i_split_sim*.txt`.

## 5. G5-K r5: F2 barrier 2→1/chunk = +19.7 µs (the L1 carveout cost)

**Tried:** reduce the shipped pp-F2 body's 2 barriers/chunk to 1 (vn
row-ping-pong, v_new pass moved to next iteration's top; sd B-reads proven
warp-local).
**Number:** **+19.7 µs** at B1T8192, 55/60 interleaved pairs SLOWER; the
variant costs +5 KB smem (84.5→89.6 KB) which shrinks the L1 carveout the
u-loads/store passes rely on — the saved `bar.sync` (marginal cost 7.7 ns,
`evidence/G3/calibration.json`) is worth ~nothing against it.
**Lesson:** smem is not free — near the 99 KB cap each KB has an occupancy/
carveout price; barrier-count levers must be A/B'd against the smem they
spend. Code kept behind `G5K_F2BAR=1`. (Same round's audit: the "~8
barriers/chunk" premise was stale — the shipped kernel had 2; the r2-style
body's 8 describes a kernel that is not the flagship.)
**File:** `evidence/G5/fwd_report.md` round-5 §0, §2; `evidence/G5/scratch/g5k_ab_*.json`.

## 6. G5-G/G5-H: tf32 inside dhu — the double negative (occupancy cliff + non-composing margins)

**Tried (twice):** tf32-mma the 3 numerics-clean dhu sites
(dvp_pdO/dh_qg/dh_w), fully unrolled in the serial reverse chunk walk.
**Numbers:** round 3: ptxas 128 regs/188 B spill → 255 regs/116–128 B
spill, dhu 3.69→4.67 ms. Round 4 retest after the dkb split freed the
register budget: compiles spill-free at 246 regs BUT (a) 246 regs halves
occupancy (2→1 CTAs/SM), dhu 2.70→3.05 ms; (b) the round-3 per-site-clean
margins do NOT compose — combined, dk 1.09e-2 > the 8e-3 bar (dh_w's tf32
error rides the dH carry across all 32 reverse chunks and dvp_pdO feeds
dvp → dW → dk).
**Lesson:** per-site numeric bisects are necessary but not sufficient —
error COMPOSES through carried state; and register pressure near an
unrolled serial walk has an occupancy cliff, so "clean margin" ≠ "ship".
dhu stayed scalar until the round-5 restructure retired it.
**File:** `evidence/G5/bwd_report.md` round-3 + round-4 "What did NOT work".

## 7. G5-J r5: tf32 on dvp_bh = rebuild chaos (measured, then the site is pinned scalar)

**Tried:** tf32-mma the dvp recompute's Bhat@dH term.
**Number:** dbeta 1.5e-5 → 2.9e-3 → 5.8e-3 across three numerically
equivalent builds — CHAOTIC, because any f32-level dh perturbation
resamples the mma's operand quantization bins.
**Lesson:** some sites are not merely margin-tight but unstable under
operand quantization; a single-build bisect cannot qualify them. dvp_bh
stays scalar; split-bf16 (deterministic 2^-16 product error) is the
documented path to mma speed at scalar accuracy.
**File:** `evidence/G5/bwd_report.md` round-5 (the latent-bug section).

## 8. G5-J r5: tree/segment scan for dhs — negative economics

**Tried (designed + CPU-validated, not shipped):** the 2-level scan for
the reverse cotangent chain (segment walks + composed boundary
correction); machinery implemented and verified on CPU (2–3 ulp fp64 at
every S∈{2,4,8,16}, and the operators are hyper-contractive:
ρ(A_n) ≤ 2.3e-18, ||A_n|| ≤ 2.2e-17).
**Number:** the 1-level chain is already latency-light (dhs 0.59 ms after
rgemm); the segment version needs [K,K]×[K,K] composes (2M MACs each,
scalar = the new bottleneck; tf32/bf16 composes re-open §6/§7's precision
questions) + 3 more kernel launches, against a remaining ~0.3 ms ceiling.
**Lesson:** "more parallelism" must clear its own overhead against a
measured serial floor — measured negative economics, not a math failure.
(The exact 1-level affine form IS the shipped win:
X_{n-1}=(2^gcl·I−WᵀB̂)X_n+(Qgᵀ−WᵀPᵀ)dO — dhu retired.)
**File:** `evidence/G5/bwd_report.md` round-5 ("What did NOT happen",
"math verdict" table).

## 9. G5-H r4: the async zero-delivery mystery (OPEN)

**Tried:** k/q/w slice ping-pong prefetch inside the real dhu walk — async
bursts into column-offset halves of staging tiles, drained at read points.
**Number:** ks/qslc read back ALL-ZERO at the slice-1 read point (dvp loses
the whole B̂@dH term; dh loses qgᵀdO) — while the IDENTICAL bursts verify
bit-exact in an isolated probe kernel (`scratch/r4_pfprobe`, src/dst
offsets {0,32}) and every full-tile burst in the same chain is
bit-identical. Deterministic; invariant to wait placement (issue-top vs
issue-end) and to which of k/qw is enabled.
**Lesson:** an interaction between the walk's live-buffer set and the
cp.async lowering in this build; root cause NOT found — gates left in the
generator (`GDN_PF_K`/`GDN_PF_QW`, default off), probes committed. Status:
open. The successor idea (async B ping-pong in the new dhs) carries the
same risk profile.
**File:** `evidence/G5/bwd_report.md` round-4 "What did NOT work" §2.

## 10. G5-K r5: B4T2048 window-variance disclosure (an ours-side reading that was NOT ours-side)

**Observed:** final adjudication B4T2048 read 1.052× (run 1) vs 1.189×
(run 2) — a swing that looks like an ours regression.
**Number:** ours medians stable 0.980–1.004 ms across both windows (r4:
0.987); the swing is fla's MEDIAN variance (1.056 vs 1.165 ms; fla min
stable 0.865–0.875), while the paired A/B shows ours −148 µs at this
shape.
**Lesson:** single-window ratio-of-medians against a noisy opponent
overstates certainty; the interleaved per-pair criterion + paired A/B + a
stable-min cross-check are what make a speedup claim honest. (All windows
flagged in-file; co-job resident throughout.)
**File:** `evidence/G5/fwd_report.md` round-5 §5; `final_adjudicate_g5k_run{1,2}.json`.

## 11. G5-C→E: the "bf16 amplification" misdiagnosis and the grading-twin bug (epistemology)

**What happened:** round-1 bwd failed dk/dbeta/dg (dk 6.6e-1…9.2e-1) and
the round-1 report attributed it to bf16 operand amplification through the
dA CxCxC chain — and started an fp32-promotion campaign on that theory.
Round-2 instrumentation found FOUR independent defects, none of them the
hypothesized operand floor: (1) dqkwg dk_local epilogue scaled the whole
accumulator by beta·2^gc (only TW carries it; deterministic ~0.52 dk
error); (2) dhu's rb accumulator zeroed outside the reverse walk
(rowB_part accumulated across chunks); (3) dg_state's prefix loop yielded
the last element, not the running sum; (4) **the fp64 grading twin itself
was wrong** — `kkT` computed only in the forward loop, the reverse loop's
dbeta/dg terms used the STALE kkT of chunk NC−1; against the JAX oracle on
the same quantized inputs |jax−twin| = 3.6e-2/2.4e-2 (exactly the round-1
"failures") while |jax−GPU| = 1.5e-5/6.9e-6.
**Number:** after the four fixes: 6/6 PASS both shapes, worst grad ~180×
headroom (dq 5.2e-7 … dh0 1.1e-8 at T512).
**Lesson (the paper's epistemology paragraph):** a wrong reference oracle
manufactures exactly the error signature you least suspect; the fp32 work
was still necessary (it made the assembly bugs visible under bf16 noise)
but the ATTRIBUTION was wrong — grade the grader, bisect with TWO
independent references, and never promote precision before the assembly is
proven (the round-1 report's "input-quantization floor" claim was itself
partly this bug).
**File:** `evidence/G5/bwd_report.md` round-2 ("What the failures actually were").

## 12. G5-J r5: the latent bf16-B̂ bug masked by the harness's H2D re-staging

**What happened:** dqkwg's dvp recompute had staged B̂ into a **bf16 tile
since round 2** (2e-3 relative error, visible at the seed chunk where
|B̂dh| ~ O(1)). It stayed invisible for three rounds because the verify
harness re-stages every input global H2D before each launch: wyb always
received **dhu's** clean dvp_all, and dqkwg's poisoned overwrite never
reached a verifier — the r3/r4 verify tables graded the clean path while
the BENCH runs used the poisoned one. Surfaced only when round-5 retired
dhu and dqkwg became dvp_all's only producer (first verify: dbeta 3.0e-4
→ 1.33e-2). Fix: the bhat tile is the f32 tCVb.
**Lesson:** a harness that re-initializes intermediates between stages can
mask producer/consumer contracts; the chain must be verified END-TO-END
through the producers the bench actually runs. (Corollary recorded: the
round-1 "twin-vs-vjp spread attributed to input quantization" was partly
§11's bug — two lessons about trusting intermediates, one root cause
class.)
**File:** `evidence/G5/bwd_report.md` round-5 (the latent-bug section).

---

## Additional recorded negatives (one-liners, files as above)

* **In-epilogue global stores** (fwd r3 §1d): F2 249→380 µs (+40%) —
  fragment-lane 2B@4B-stride stores vs coalesced cell passes. Reverted.
* **F1 A_m STS pad** (fwd r5 §2): −0.8 µs, 28/60 pairs — 557K STS conflicts
  are real but off the critical path. `G5K_F1PAD=1`.
* **128-bit global store path** (fwd r5 §1): does not exist in this
  toolchain — `vector.store` lowers to element-split `st.global.b16` x8;
  the scalar 64B/warp stores are already the best this pipeline emits.
* **BV=64 dhu retile** (bwd r3): needs ~150 KB+ smem vs the 101376 B cap —
  resource wall, not a knob.
* **warpspec at stage level** (G3): hardware mbarrier deadlock before the
  G4 protocol fix; ranking stayed provisional until it ran.
* **nomma (linalg→loops) lowering** (G3 §6.1): 50–100× off the mma path —
  the one structural dead end the cost model caught before any GPU time
  was spent on it.
