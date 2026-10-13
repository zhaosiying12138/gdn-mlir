## 四十一、仓库与复现

一本宣称"每个数字有出处"的手册，最后必须回答：读者如何亲手重跑这些数字。术语先行：门（gate），指一个自带通过判据、成本上限与产物落盘的自包含验证单元；幂等，指重复执行同一命令得到同一产物、已完成的部分自动跳过。

### 1. 仓库结构

```
dialect/          gdn 方言（L0 op + GdnConfigAttr + 12 个直通 op + passes + gdn-opt）
tools/            全部生成器与 harness（gdn_l1_gen / gdn_ref_jax / gdn_l2_gen /
                  gdn_solver / gdn_costmodel / gdn_full_gen / gdn_bwd_gen /
                  gdn_smem_fix / gdn_verify / gdn_final_adjudicate / rc1_price）
scripts/          幂等驱动（build_llvm / build_dialect / run_calib / repro / resource_guard）
evidence/G0-G7    每个门的报告 + 原始数据（PTX/SASS/计时 JSON/交叉验证记录/覆盖矩阵）
paper/zh/         深挖论文与本文
```

一个组织原则贯穿全仓：工具产出证据，门消费工具，证据目录只追加不修改。读者从任何一章的出处标注出发，都能在 evidence 下找到对应的原始文件；每个门目录内有报告（人读）与 JSON/PTX/SASS（机读）两层。

### 2. 七道复现门

| 门 | 内容 | 设备与成本 |
|---|---|---|
| G0 | 双环境、LLVM 构建、管线冒烟（上游 sm_80 测试重定向 sm_120a 在 5090 运行）、fla 与 FlashQLA PTX 收割 | 纯 CPU（冷启约 2–2.5 小时，GPU 仅 1 分钟） |
| G1 | 语义三方闭合：JAX 规约自检 44 项、MLIR L1 前反向、方言构建冒烟 | 纯 CPU，约 15 分钟 |
| G2 | 旋钮矩阵：20 配置生成编译加 PTX 覆盖分析 | 纯 CPU，约 25 分钟 |
| G3 | 标定加排序门槛：sm_120 标定表、20 配置上机计时、Spearman 门 | 两步 GPU，约 45 分钟 |
| G4 | 逐配置正确性：20 配置对 fp64 孪生参考验证，双随机种子 | GPU，约 1 小时 |
| G5 | 全算子验证加性能终判：前反向数值、交错终判 | GPU，约 1 至 1.5 小时 |
| G6 | 搜索成本：Z3 计时加 fla autotune 成本测量 | CPU 加 GPU，约 10 分钟 |

［Makefile:30-52；驱动 scripts/repro.sh，make repro-list 列出门与成本］入口约定：make repro-g1 到 repro-g6 逐门执行，make repro-all 顺序全跑、首败即停并输出汇总表。门的顺序即依赖顺序——G2 的覆盖矩阵吃 G0 的收割物，G5 的终判吃 G4 验证过的 cubin。全部七门约 4–6 小时，其中纯 CPU 部分约 3 小时。

门的读法示例，说明"出处可追"如何落地：第二章前向表里 B1T8192 的 1.349 倍，追到 evidence/G5/final_adjudicate_r25f_w32p.json 的 results.B1T8192（speedup_ratio_of_medians 1.3493 即表中之数；stats 块另有 30 对、逐对比值中位 1.3307、四分位距 0.2363、符号检验 p 值 0.0），该 JSON 同时带着窗口前后的 nvidia-smi 快照与双侧原始样本；第三十八章的三态表追到 evidence/G6/coverage_hatch_summary.md，其每格又指回 G2 覆盖矩阵与 G0 收割物。证据树因此是可回溯的链，任何一章的数字都能一路走到原始字节。

### 3. 环境需求

WSL2、conda 环境 gdn、CUDA 13.x、计算能力 12 以上的 NVIDIA GPU。语义与求解两大部分（G1、G2 的核心）完全纯 CPU，不需要 GPU——这是分层设计的直接红利：第三十六章的塔基与求解器的合法性枚举在笔记本上即可复验。GPU 部分对标 RTX 5090 Laptop（sm_120a，24GB）；标定表（第十章）随设备重跑自动更新，排序门槛与验证结论不绑定具体卡型号。GPU 使用另有协调约定：正确性短跑可与他人共享窗口（显存低于 2GB），计时测量只在登记过的窗口进行——这是第三十七章双窗口口径的执行侧对应物。

### 4. 资源纪律

一句话版本：内存预算按实测空闲动态封顶，全局同一时刻只允许一个重构建，一切长任务幂等可断点。展开为机制：并行度不取机器核数，取空闲内存除以每作业 2.5GB 并夹在 4 到 12 之间；watchdog 每 30 秒采样，空闲低于 6G 冻结构建、回升到 8G 恢复、跌破 3G 终止（ninja 增量续编无损）；重构建槽位用信号量全局互斥；长任务分 stage 加内容哈希戳，已完成即跳过，恢复等于重跑驱动脚本。核心函数五行的样子：

```
rg_jobs() {
  local a j
  a=$(rg_avail_gb)
  j=$(( a * 10 / 25 ))     # 每 job 预算 2.5GB
  (( j < 4 ))  && j=4
  (( j > 12 )) && j=12     # 铁律上限
  echo "$j"
}
```
［scripts/resource_guard.sh:17-26；完整铁律见 CONTRIBUTING.md，watchdog 与槽位实现同文件 28-60 行］

纪律的来历是两次真实代价：本机 WSL 曾因内存耗尽整机崩溃两次，/tmp 随崩溃清空。因此守则还规定持久物只落盘到仓库内（ext4），gate 产物即时提交——内存中不留唯一副本。这套纪律使七道门在 47GB 内存、与训练会话共享的机器上全程可跑，也是复现承诺的物质基础：复现脚本本身先复现了自己的存活。
