# WSL 崩溃恢复协议

## 恢复三步

1. `cd ~/codebase/gdn-mlir && make status`（各 stage 状态 + 资源余量）
2. 查 `logs/watchdog.log` 尾部，确认崩溃前是否有 STOP/TERM 记录
3. 重跑"进行中"的 stage 驱动命令（全部幂等：worktree/已配置 build/ninja 增量/pip 已装自动跳过）；G7 之后每个 gate 有官方复现入口 `make repro-g0 .. repro-g6`（见 EVIDENCE_INDEX.md）

## 阶段状态表（每完成一项就地更新并 commit）

| Stage | 命令 | 状态 | 证据 |
|---|---|---|---|
| G0-2 仓骨架 | — | **done** | scripts/ + docs/PLAN.md |
| G0-3a env-gdn phase A | `make env-a` | **done** | py3.12.15 + pybind11/nanobind2.15/numpy |
| G0-3b env-gdn phase B | `make env-b` | **done** | torch 2.14.1+cu130 / triton 3.8.0 / fla(e) / z3 5.1.0 / jax 0.11.2 |
| G0-4 LLVM/MLIR build | `make llvm` | **done**（01:30，1.9GB） | logs/build_llvm.log；含 SM80 测试开关与 runner 工具库 |
| G0-4b sm80 冒烟（GPU 共享） | `bash scripts/smoke_sm80_manual.sh` | **done**（teardown 瑕疵备注） | evidence/G0/sm80_smoke.txt |
| G0-5 fla PTX 收割 | `make harvest-fla` | **done 22/22** | evidence/G0/fla_ptx/SUMMARY.md |
| G0-6 FlashQLA PTX 收割 | `make harvest-qla` | **done 5/5**（tilelang 0.1.15） | evidence/G0/qla_ptx/SUMMARY.md |
| G0 gate 汇总 | — | **done ✅ G0 全判据达成** | evidence/G0/GATE_REPORT.md |
| G1-A JAX 规约 | tools/gdn_ref_check.py | **done 44/44** | evidence/G1/jax_spec_report.md |
| G1-B MLIR L1 fwd | tools/gdn_l1_check.py | **done 7/7** (vs JAX 6e-7) | evidence/G1/l1_harness_report.md |
| G1-C gdn 方言 L0 | scripts/build_dialect.sh | **done** 冒烟绿 | dialect/ + commit 5555704 |
| G1-B2 MLIR L1 bwd | tools/gdn_l1_check.py --bwd | **done 3npz×6grads** | evidence/G1/l1_bwd_check_output.txt |
| G1 gate 汇总 | — | **done ✅ G1 全判据达成** | evidence/G1/GATE_REPORT.md |
| G2-A 旋钮库+PTX矩阵 | tools/gdn_l2_batch.py | **done 20/20** | evidence/G2/gen/INDEX.json + gen_report.md |
| G2-B 覆盖矩阵分析 | tools/gdn_coverage.py | **done**（feature 层全覆盖） | evidence/G2/coverage_matrix.md |
| G2-C warp-spec+hatch | GdnHatch.td + ws/tma 配置 | **done** hatch 0/5 | evidence/G2/hatch_report.md |
| G2 gate 汇总 | — | **done ✅ G2 达成** | evidence/G2/GATE_REPORT.md |
| G3-A 标定+20配置计时 | scripts/run_calib.sh, tools/gdn_time.py | **done** 18/20（shared-GPU 口径） | evidence/G3/calibration.json+config_times.json |
| G3-B Z3+成本模型 | tools/gdn_solver.py --validate | **done ρ=0.8962** | evidence/G3/solver_report.md |
| G3 gate 汇总 | — | **done ✅ G3 达成** | evidence/G3/GATE_REPORT.md |
| G4 GPU 正确性 | tools/gdn_verify.py | **done ✅ 20/20 双seed PASS** | evidence/G4/GATE_REPORT.md |
| G5-A 基线+参考 | gdn_bench_flaptx.py 等 | **done** | evidence/G5/baseline_report.md + fla_baseline.json |
| G5-B 前向全算子 GPU | gdn_full_gen.py（5 轮迭代） | **done 数值 4/4 + 性能过线（1.59/1.05-1.19/1.25-1.31× 交错口径）** | evidence/G5/fwd_report.md |
| G5-C 反向全算子 GPU | gdn_bwd_gen.py（5 轮迭代） | **done 数值 6/6 双形状；性能诚实未达 0.126×（7.2× 轨迹+全账目）** | evidence/G5/bwd_report.md |
| G5 gate 终判 | gdn_final_adjudicate.py | **done ✅ fwd 3/3 ≥1.1×；bwd 诚实开放战线；G5 关闭** | evidence/G5/GATE_REPORT_FINAL.md |
| G6 证据链综合 | tools/g6/*（消融/搜索成本/覆盖账本/失败长卷） | **done ✅（≈50× 搜索成本、hatch 1/≤5、12 条失败条目）** | evidence/G6/GATE_REPORT.md |
| G7-A 论文 | paper/ | **doing**（骨架已落：outline/abstract/contributions/figures plan） | paper/ |
| G7-B 开源+复现 | scripts/repro.sh + Makefile repro-* | **doing**（本工作流：per-gate 复现目标、README OSS 首页、EVIDENCE_INDEX、干净窗口终判 hook） | EVIDENCE_INDEX.md + scripts/adjudicate_when_clean.sh |
| G7-A 论文骨架 | paper/ 四件套 | **done**（118 证据引用+10 条 TODO） | paper/outline.md |
| G7-B 复现+开源 | Makefile repro-* + README + 索引 + 钩子 | **done** | EVIDENCE_INDEX.md |
| G7 gate | — | **done ✅ G0-G7 全收官** | evidence/G7/GATE_REPORT.md |

## 当前下一步（体制变化战役收官后，2026-10-12）

1. **bwd：体制变化战役（RC1→R4）已收官**——用户推翻 D2 后四轮完成：
   装机杠杆 `GDN_BF16G`（9 符号 bf16 全局量）/`GDN_ARET`（caryA 退役 +
   dhs 移位拷贝坍缩）/`GDN_TGEMM_AH`（wS CVT 税 hoist），最终链 ==
   r3-final bytes（8 kernel，R4 健康认证 48/48 cubin + det N=3 位级）。
   战绩：小形状 B1T512H4 **1.723×** 胜、冷启动全形状胜、u≥1 全阶梯
   0.411–0.488× 且被**三重独立证据**（预算/地板算术/融合 NO-GO）在
   精度层面关闭（u=8 天花板 0.85–0.92×，误差合同的数字）。正式收官
   记录：`evidence/G5/CAMPAIGN_CLOSEOUT.md`——**用户确认待定**（接受
   精度关闭结论 / 放宽容差合同（重开 1.33–1.49× 算术）/ 重定指令范围）。
   重开仅限新体制（本阶梯外形状、学习门数据）或容差合同变更。
2. **fwd：R25 正式记录不变**（W32P cubins e7c8eb4c/39f094c3/be5b2377，
   共载 n=30 交错：T2048 2.125× / B4 1.2211× / T8192 1.3493×；干净窗
   clean3 抽查并存双引：T2048 1.611× DECIDABLE、B4 1.067× / T8192
   0.991× parity 类）。战役未动任何 fwd 旋钮；fwd 战线在当前几何关闭，
   无排期轮次。
3. **G7-B 遗留**：License 选择（README 内 TBD 标记）、引用条目填充；如需精简仓体积按
   README"Repository size"节执行（先量后删，勿盲删）。
4. **干净窗口终判**：clean3 **已执行**（2026-10-11，经
   `scripts/adjudicate_when_clean.sh`，记录
   `evidence/G5/final_adjudicate_clean3.json`，见上条双引口径）——
   hook 使命完成，无待办。
5. **论文**：深度解剖长文已扩至**11 章**——`paper/zh/sections/11-regime-change-campaign.md`
   （体制变化战役章：标度律/误差预算/退役与坍缩/hoist/融合定价/精度关闭定理），
   装配文件 `paper/zh/gdn-mlir-deep-dive.{md,html}` 已再生成（目录同步）；
   战役收官记录见 `evidence/G5/CAMPAIGN_CLOSEOUT.md`。

## 干净窗口终判 hook（G7-B 交付，已执行）

G5 前向终判原为**交错口径**（同窗 ours/fla 交替，对共载负载稳健）；正式的
**干净窗口**复判由 `scripts/adjudicate_when_clean.sh` 承担：

- 轮询 `nvidia-smi --query-compute-apps`，连续 3 次采样（默认 120s 间隔，
  `ADJ_POLL_SEC`/`ADJ_CLEAN_RUNS` 可调）无任何常驻 compute 进程即视为干净；
- 干净后**先在 `~/codebase/GPU-COORDINATION.md` 登记协调行**（协议第 2 条），
  再运行一次 `tools/gdn_final_adjudicate.py --out evidence/G5/final_adjudicate_clean.json`，
  结束后把协调行改为 done/failed；全程**绝不触碰对方进程**；
- `--check` 干跑：只报告当前窗口是否干净（`make adjudicate-check`）；
- 锁 `/tmp/gdn_adjudicate_clean.lock`（tmpfs，崩溃自清）；日志
  `logs/adjudicate_clean.log`。

**clean3 已于 2026-10-11 执行完毕**（干净窗、W32P ship、n=30 stats：
B1T2048 1.611× DECIDABLE / T8192 0.991× parity-not-significant /
B4T2048 1.067× parity；`evidence/G5/final_adjudicate_clean3.json`），
与 r25 共载正式记录按"双窗双引"并存。hook 无待办。

## 注意

- /tmp 里的东西（信号量/暂停标记/watchdog pid）崩溃即消失——这是设计：它们全部可再生。
- 若 `ninja` 被 watchdog TERM 过：直接重跑 `make llvm`，增量续编。
- 若 conda env 损坏：删 `~/miniforge3/envs/gdn`（或 gdn-tl）重跑 `make env-a && make env-b`（gdn-tl 由 harvest_qla.sh 自愈重建）。
- 复现输出落 `evidence/repro/<gate>/`（已 gitignore）；`run_calib.sh`/`run_g3.sh`/`gdn_time.py`/`gdn_verify.py`/`g6_fla_autotune_cost.py` 按设计就地覆写其 canonical evidence 文件——落地数字在 git 历史中可恢复（`git checkout -- evidence/G3` 等）。
- 仓体积：evidence/ 工作树 ~698MB（G5/refs 438MB + G5/scratch 96MB + 各 gen 目录 ~135MB 可再生）；对策与度量见 README"Repository size & evidence policy"与 EVIDENCE_INDEX.md 尾注。**未删除任何文件。**
