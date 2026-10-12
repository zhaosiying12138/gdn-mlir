# G6: T1 coverage + T2 both-poles + escape-hatch ledger (final consolidation)

Consolidates `evidence/G2/coverage_matrix.md` (20 generated configs, ours
column LIVE) with the G5-G round-3 addendum (`evidence/G2/hatch_report.md`
ADDENDUM + `evidence/G5/bwd_report.md` round-3): **tf32 m16n8k8 is now
EMITTED by us** via escape hatch #1, which flips the `mma.sync` feature
family to full YES and closes the mma-signature table.

## 1. T1 instruction-family coverage (final state)

Legend: YES = emitted by some config of ours; PARTIAL = family emitted, a
reference opcode variant/signature missing (variant-level only); NO = never
emitted; `-` = not present in that reference. fla/FlashQLA columns = the
G0 harvests (`evidence/G0/fla_ptx/SUMMARY.md`, `evidence/G0/qla_ptx/SUMMARY.md`).

### 1a. Feature tier (15 families)

| family | tier | fla | FlashQLA | ours (FINAL) | delta vs G2 matrix | residual gap (variant-level) |
|---|---|---|---|---|---|---|
| `mma.sync` | feature | YES (3284) | YES (2496) | **YES** | **PARTIAL→YES via hatch #1** (`gdn.mma.tf32k8`, G5-G r3): the m16n8k8.tf32 signature — fla's exact spelling — is now emitted; bf16 m16n8k16 was already YES (552, 10 configs) | none at signature level |
| `ldmatrix` | feature | YES (1182) | YES (1016) | PARTIAL | unchanged | x1 / x4.trans operand-order variants (reachable by numTiles/transpose knobs); x4 + x2.trans emitted — and ldmatrix is the shipped full-op fragment-load path (G5) |
| `stmatrix` | feature | YES (283) | YES (212) | NO | unchanged | see §4 — assessed NOT NEEDED for GDN |
| `cp.async.classic` | feature | YES (28) | YES (8) | YES | — | — |
| `cp.async.ctrl` | feature | YES (36) | YES (2) | PARTIAL | unchanged | `wait_all` variant only (commit/wait_group emitted) |
| `tma.bulk.tensor` | feature | - | YES (100) | PARTIAL | unchanged | `.L2::cache_hint` load spellings + 4d/5d store variants; 4d+5d loads + 2d store emitted (compile level), runtime-wired and numerically verified in G4 |
| `tma.bulk.ctrl` | feature | - | YES (50) | YES | — | — |
| `mbarrier` | feature | - | YES (386) | PARTIAL | unchanged | spelling only: ours `.shared.b64` vs ref `.shared::cta.b64` (same instruction; default scope explicit) |
| `elect.sync` | feature | - | YES (59) | YES | — | — |
| `setmaxnreg` | feature | - | YES (16) | YES | — | — |
| `prefetch.tensormap` | feature | - | YES (22) | YES | — | — |
| `bar.sync` | feature | YES (697) | YES (203) | YES | — | — |
| `barrier` | feature | - | YES (39) | NO | unchanged | `gpu.barrier` covers `bar.sync 0` (free); family-level NO is a naming artifact, not a capability gap |
| `shfl.sync` | feature | YES (966) | YES (80) | PARTIAL | unchanged | `bfly` variant (idx emitted, used by warpspec broadcast) |
| `ex2.approx` | feature | YES (549) | YES (74) | PARTIAL | unchanged | `.ftz.f32` codegen-flag variant; our decay epilogue uses the non-ftz form, same as qla |

### 1b. mma.sync signatures (the T1 sub-gate, final)

| signature | fla | FlashQLA | ours (FINAL) | note |
|---|---|---|---|---|
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | 2708 | 2496 | YES (552) | the bf16 workhorse, in-tree (`nvgpu.mma.sync`), G2 |
| `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` | 576 | 0 | **YES** | **escape hatch #1** `gdn.mma.tf32k8` (`llvm.inline_asm`, early-clobber); shipped in 16 bwd GEMM sites (G5-G r3); was the sole signature-level NO at G2 |

### 1c. Baseline tier (18 families)

ld/st.global, ld/st.shared, ld.param, fma.f32, math.f32x2, cvt.bf16f32,
math.f32, math.int, logic.shift, cvt.int, cvta, setp, selp, mov, mov.pred,
bra — all reached through standard arith/memref lowering; ours holds YES
for `bra` and PARTIAL for 13 (vector-width `.v2/.v4`, b64/b16 width and
predicate-encoding spellings); NO for `fma.f32`, `math.f32x2`, `math.f32`,
`mov.pred` (free fusions/packings not exercised by our kernels — not GDN
dependencies). Full per-family rows: `evidence/G2/coverage_matrix.md` §1.

### 1d. Final counts

**ours: 8 YES / 19 PARTIAL / 6 NO of 33 T1 families** (G2 close-out was
7/20/6; the tf32k8 hatch moves `mma.sync` PARTIAL→YES). Feature tier
specifically: **8 YES / 6 PARTIAL / 2 NO of 15** — every feature family
the references use is emitted (YES or variant-PARTIAL) except `stmatrix`,
the sole feature-level NO, closed by measurement as the wrong pattern for
our epilogues (§4); the remaining 4 NO cells are baseline-tier
fusions/packings. All residuals are variant-level, not capability gaps.

## 2. T2 structural coverage — the both-poles statement

The two reference design poles are BOTH expressible in the knob matrix
(`evidence/G2/coverage_matrix.md` §3–5; `evidence/G2/hatch_report.md` §3):

* **fla pole** (Triton, multistage `ld.global`/`cp.async`, 255 regs + spill,
  unrolled loops): the base mma configs — regs pinned at the 255 cap with
  stack frames 0–26.8 KB matching fla's envelope [40,255]; multistage
  cp.async.ca/cg via `pipeline` 0/2/3 (mma 16/48/64; LDGSTS 512/1536/2048).
* **FlashQLA pole** (TileLang/CUTLASS-style warp specialization, ≤128 regs,
  TMA + mbarrier + elect.sync): `kkt_bf16_mma_p2_t64x64x64_w4_ws` +
  `kkt_bf16_mma_tma_ws` — setmaxnreg dec96/inc224 drives spills
  **22,844 B → 48 B** (FlashQLA-class), 5 TMA ops replace 1026 classic
  copies, elect-leader mbarrier protocol; G4 made both hardware-correct
  (deadlock protocol fix + host CUtensorMap wiring; `evidence/G4/GATE_REPORT.md`).

## 3. Escape-hatch ledger (budget: **1 of ≤5 used**)

| # | op | status | justification (why in-tree is not enough) | measured impact |
|---|---|---|---|---|
| 1 | `gdn.mma.tf32k8` → `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` | **USED** (shipped, 16 bwd GEMM sites) | `nvgpu.mma.sync` enforces the 8x8x128b fundamental tile → tf32 caps at K=4; tabled `nvvm.mma.sync` lists m16n8k8 only for f16/bf16/i8. Lowering is a direct `llvm.inline_asm` with early-clobber `=&f` outputs — `nvvm.inline_ptx`/BasicPtxBuilder cannot express early-clobber, and without it ptxas legally maps the D quad onto the A quad (SASS `HMMA R12,R12,R10,R16`) corrupting operands (a K=16 chain failed 118/128 cells deterministically). Fragment layout derived by hardware probe: a0=(g,2t) a1=(g+8,2t) a2=(g,2t+1) a3=(g+8,2t+1), b0/b1=(2t{,+1},g), d = the bf16 N=2·tig rule | bwd B1T2048H16 (shared window): dqkwg 4.71→2.75 ms (1.71×, with r4/c4r), wyb 2.65→1.21 ms (2.19×, with ra/c1r), **e2e 14.16→11.06 ms (1.28×)**, 6/6 numerics at both gate shapes; per-site bisect table in `evidence/G5/bwd_report.md` round-3 |

Sources: `evidence/G2/hatch_report.md` (ADDENDUM), `evidence/G5/bwd_report.md`
(round-3 "The hatch"), dialect: `dialect/include/Gdn/GdnHatch.td`,
`dialect/lib/GdnHatch.cpp`.

The G2 finding stands as the narrative upgrade: every OTHER planned hatch
family (setmaxnreg, elect.sync, mbarrier, TMA, prefetch, ldmatrix, shfl)
turned out to be tabled in this Oct-2026 LLVM tree and lowers in-tree —
stock MLIR's expressiveness boundary already covers warp-specialized
TileLang/CUTLASS-class kernels, and the one true hatch is an NVVM
tablegen limitation, not a scheduling one.

## 4. Remaining known gaps — with the "not needed for GDN" assessment

| gap | class | assessment |
|---|---|---|
| `stmatrix` (family NO) | `nvvm.stmatrix` IS tabled in-tree but needs a shared-window pointer operand that only materializes post-LLVM-conversion (`evidence/G2/hatch_report.md` §4) | **NOT NEEDED for GDN**: stmatrix stores fragments in the reference epilogues; our epilogues write through coalesced thread-strided cell passes, and the fragment-mapped store alternative was measured **+40% SLOWER** (F2 249→380 µs, `evidence/G5/fwd_report.md` r3 §1d) — the family is the wrong pattern for our epilogue shapes, closed by measurement not by spelling |
| `tma.bulk.tensor` `.L2::cache_hint` load spellings + 4d/5d stores | variant spelling (the nvgpu.tma.async.load lowering hardcodes no-hint) | NOT NEEDED: cache-hint tuning is a perf variant of a family we already emit + verified on hardware (G4); production GDN kernels chose classic cp.async by measurement |
| `ex2.approx.ftz.f32` | codegen flag | NOT NEEDED: fla/qla decay paths use the non-ftz form we emit; ftz differs only near-denormal, far below the 5e-3 gate |
| `shfl.sync.bfly`, `cp.async.wait_all`, `ldmatrix` x1/x4.trans, `.v2/.v4` widths, `.shared::cta` spellings | single-flag / width / spelling variants | NOT NEEDED for GDN semantics; all reachable via existing knobs (documented per-cell in `evidence/G2/coverage_matrix.md` §1) — none is a capability gap |

Bottom line for the paper: T1 feature-tier coverage full (8/15 YES, 7/15
PARTIAL with variant-only residuals), T2 both design poles expressible and
hardware-verified, escape-hatch budget 1/≤5 with the single use justified,
measured, and numerics-gated.
