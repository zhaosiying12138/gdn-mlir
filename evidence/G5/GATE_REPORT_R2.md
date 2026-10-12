# G5 Gate 报告（决战计时）—— 第 2 轮/N（2026-10-09，commits 412dc96/9a1683f）

## 本轮战果总表

| 方向 | 第1轮 | 第2轮 | vs fla | 提升 |
|---|---|---|---|---|
| fwd staged B1T2048 | 1.597ms | **0.663ms** | 0.48×（fla 0.316ms） | 2.4× |
| fwd staged B1T8192 | 5.398ms | **2.450ms** | 0.36× | 2.2× |
| fwd staged B4T2048 | 4.476ms | **1.636ms** | 0.55× | 2.7× |
| bwd e2e B1T2048 | 58.9ms | **14.2ms** | 0.073×（fla 隐含 1.03ms） | 4.2× |

**数值**：fwd 全程绿（每杠杆后复验，o/ht ≤7.5e-4）；**bwd 本轮 6/6 全 PASS**（B1T512H4 与 B1T2048H16 双形状，最差梯度 4.5e-5，裕量 ~180×）——**反向数值债清零**。

## 反向数值翻案（本轮最重要发现）

第 1 轮"bf16 底噪经 dA 链放大 64×"的诊断**被证伪**：仪表化逐中间量对拍发现 4 个确定性 bug——①dqkwg 的 dk_local epilogue 多乘 β·2^gc；②dhu 的 rb 累加器逆走未清零；③dg_state 前缀 scf.for yield 了末元素而非累加和；④**评分孪生自身用了陈旧 kkT**（第 NC-1 chunk 的）——所谓"残余超差"其实是孪生的误差（|jax−孪生|=3.6e-2 恰好等于原"失败值"，而 |jax−GPU|=1.5e-5）。fp32 中间量+Kahan 仍必要（让 bug 可见并买下裕量）。教训入库：**怀疑 kernel 前先审评分器**。

## 本轮交付的杠杆与发现

**fwd**：①全部 K 循环走 G2 transform 流水（multibuffer-2+async groups+pipeline depth-2）+ **thread-strided staging——关键发现：G2 式 block-uniform 展开staging 会被每个线程重复执行（256× 冗余 cp.async，修前比第 1 轮还慢 2×）**；②`failures(suppress)` 在本 LLVM 构建损坏（任何非 copy 的 scf.for 杀掉 transform 输出）→ 非 K 循环全改 scf.while + pin_alloc；③16×16 分块 solve（对角 16 步代入+非角 mma，F1 429→270µs）；④标量 pass 数学消除 3 个（β/2^gc/kterm 折叠）；⑤F3 融合 qk+oi 单循环；⑥**mono cubin**（3 kernel 共享全局，砍 DtoD ~0.1ms）——顺带发现 **ptxas 跨入口静默合并重名 .shared 符号导致 OOB**（已按 kernel 改名，新 PTX merge 工具入库）。
**bwd**：kkt+P warp-tile mma 化（SASS 128 HMMA/96 LDSM）；kkt_all/P_all 提升为全局量（杀掉 dhu/dqkwg/wyb 的重复重建）；线程分区 staging（原 128× 冗余发射）；s5/rowB 归约并行化；标量段 threads=256。

## 剩余差距（ncu 定位，第 3 轮任务单）

- **fwd F2 = 51-60% 时间**：串行 32-chunk 链 MIO 队列满阻塞（43% issue 间隔）→ chunk 级双缓冲（G4 mbarrier 协议）与/或 segment-split 扫描（状态交接 = 前缀 decay 积，数学允许）；F1 1-CTA/SM（89KB smem）；F3 DRAM-bound 31% 占用（255 regs）。
- **bwd 残余 14ms = f32 操作数标量 GEMM**（dW/dvp/dq1/TW/dk 尾部）——bf16 mma 会重新毒化（实测 4e-4 的 dW 噪声→5e-5 dAf）→ 杠杆=tf32 mma（m16n8k8，G2 记录的 NVVM 表缺口正是它——inline-PTX hatch 候选）、split-bf16 3-mma 仿真、dhu BV=64 re-tile。

## 判据状态

| # | 判据 | 状态 |
|---|---|---|
| 1-2 | 基线/FlashQLA 横比 | ✅（第1轮） |
| 3 | fwd 数值 | ✅（全程保持） |
| 4 | bwd 数值 | ✅ **6/6（本轮翻案达成）** |
| 5 | e2e ≥1.1× | ❌ 第2轮（fwd 0.36-0.55×、bwd 0.073×）——差距全部 ncu 定位、杠杆在第 3 轮任务单 |
| 6 | 单 kernel 旗舰 | 建成+数值过；性能随 staged 共同迭代 |

第 3 轮：F2 chunk 双缓冲/segment-split（fwd 最大杠杆）→ bwd tf32-mma/dhu re-tile → 干净窗口交错终判。
