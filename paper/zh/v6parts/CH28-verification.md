## 二十八、验证塔与容差合同：从 FileCheck 到跨进程逐位一致

一段 IR 通过解析、一个 kernel 数值达标、一次计时更快，分别只排除了部分故障。验证需要把结构、资源、数值、写集、时序和性能拆成不同判据，并记录每把尺子的盲区。容差回答误差是否越界，逐位一致回答相同执行是否产生相同字节，交错终判回答性能差能否与测量窗口区分；三者不能互相代签。

### 1. 四层验证各查一种可归因性

结构测试与数值测试从入口就分开：L0 冒烟只固定 op、config 和调用边界，具体的 FileCheck/lit 写法已在第三章给出；它们不能证明六梯度数值或共享内存时序。四层验证的组织如下：

| 层 | 被测对象 | 主要尺子 | 核心结论 |
|---|---|---|---|
| L1 语义 | CPU memref 级 L1 | JAX fp64、fla naive、MLIR L1 三方交叉 | 数学定义与执行规约闭合 |
| L2 静态 | 20 个旋钮配置及最终 PTX/SASS | FileCheck 类结构断言、ptxas 资源、SASS 指令 | 配置、指令与资源申报可审计 |
| L3 阶段 | 逐配置 GPU kernel | 逐位相同的输入、算法同构的 fp64 twin | 20/20 在种子 0 与 7 下分别通过 |
| L4 全链 | 完整算子、写集、重复运行与计时 | 六梯度、结构不变量、逐位一致、交错终判 | kernel 间合同与发布结论闭合 |

L1 的三方距离落在各自量程地板：MLIR L1 对 JAX 前向约 6e-7，反向六梯度最差 6.4e-6；JAX 前向对 fla naive recurrent/chunk 的 fp32 相对误差分别为 1.2e-7 和 2.3e-6。JAX 自身还用顺序、chunk 与 VJP 对照约束规约。三条独立边同时很小，比单一“与参考一致”更能排除参考和实现共同写错的可能。

L2 必须读最终发布 cubin 对应的 ptxas 报告，因为后处理后的第二次编译可能改变寄存器与 spill。SASS 能确认 mma、LDSM、LDGSTS 等指令确已落地，仍不能由指令存在推出操作数映射正确。L3 的 20/20 双种子则把数据置换纳入验证，降低只对一组嵌入输入调参的风险。[evidence/G4/GATE_REPORT.md:8-17,49-56]

### 2. fp64 twin、写集与量化台阶互相定位

阶段参考遵守两条纪律。第一，GPU 与 numpy twin 接收同一份已量化输入字节，输入侧差不混入结果；第二，twin 采用相同 chunk、缩放次序和算法形态，以 fp64 重算，测到的差主要归因于 kernel 内部舍入与实现缺陷。若换成另一种算法参考，算法重结合差和机器误差会混在一起，定位能力随之下降。

结构不变量进一步检查“该写的格子是否都写了”。kkt 的 64×64 严格下三角连同对角应有 2080 个单元，一次 mma 路径只覆盖 1168 个：

```text
frag_epilogue stored the accumulator's even elements at column
ncol + %g -- the lane's B-fragment N index -- instead of the
accumulator/D-fragment N index 2*threadID_in_group.
Result: two lanes wrote conflicting values to the same columns while
other columns were never written (kkt: 1168/2080 cells covered;
o: 2304/4096).
After the fix every mma config writes all 2080 kkt tril cells / all
o cells and matches the fp64 twin at 9.77e-4.
```

[evidence/G4/GATE_REPORT.md:62-75，原文节选]

读这份记录应先数写集，再看 fragment 布局。B fragment 的 lane 列索引被误用于 D fragment，导致两条 lane 写同列、其他列缺失。数值容差也会失败，2080 的精确计数却把故障定位到 epilogue 列映射；FileCheck 即使成功匹配 `mma.sync`，也看不见这个写集漏洞。

修复后的 9.77e-4 还有一项量化对账：9.765625e-4 等于 2^-10。对处于 [0.25,0.5) 区间的 bf16 数，相邻量化点间隔为 2^-9，四舍五入的最大半格误差正是 2^-10。精确模拟 f32 累加与 bf16 输出舍入得到 9.765625e-4，GPU 实测的小数记录 9.77e-4 与其一致。[evidence/G4/GATE_REPORT.md:72-75] 这把最大残差归因到输出量化台阶；它并不单独证明所有中间操作无误，所以仍需 fp64 twin、完整写集和双种子共同裁决。

### 3. 容差合同以基线距离标定，以本方裕度放行

端到端合同按输出张量分档，使用最大绝对误差：

| 输出 | 放行门限 |
|---|---|
| o、ht | 5e-3 |
| dq、dk、dv、dh0 | 8e-3 |
| dβ、dg | 2e-2 |

门限参照 fla 自身对 JAX fp64 的距离。dβ、dg 若沿用 8e-3，会比基线若干标定残差还紧，失去“同级准确”的含义；2e-2 与这些输出的基线距离处于同一量级。合同约束本方产物，并不保证每个 fla 标定点都自动落入该门限。

容差文件保留了每个形状的误差、参考幅值与 bf16 输出地板。大形状 dg 的一条记录是：

```json
"dg": {
  "fla_dtype": "torch.float32",
  "max_abs_err": 0.02581325795185463,
  "ref_max_abs": 3.7026142737449854,
  "rel_err": 0.0069716303247937235,
  "bf16_out_floor": 0.005954688964605381
}
```

[evidence/G5/refs/tolerances.json:218-223]

读点时比较 max_abs_err 与 bf16_out_floor，而非只看 rel_err。对当前 JSON 的 24 个输出记录，直接用 `max_abs_err / bf16_out_floor` 逐项计算，得到 0.612–4.335 倍，中位数 2.072 倍（本文据 `evidence/G5/refs/tolerances.json` 复算）。残差总体处于输出量化尺度附近；最小项低于 1，说明 `bf16_out_floor` 是尺度估计而非逐元素严格下界。这里统计的是 fla 对 fp64 参考的误差与估计输出地板之比，不能把其他报告的筛选或四舍五入口径混作同一组数字。

本方还要报告门限除以实测最大误差的裕度。当前 R3/R4 八 kernel 的六形状记录中，最紧的一项为 dk 的 1.588 倍：

| 输出 | 对应最大误差 | 最紧裕度 |
|---|---:|---:|
| o | 6.486e-4 | 约 7.7 倍 |
| ht | 7.493e-4 | 约 6.7 倍 |
| dq | 7.5585e-4 | 约 10.58 倍 |
| dk | 5.0362e-3 | 约 1.588 倍 |
| dv | 5.6902e-4 | 约 14.06 倍 |
| dβ | 5.1743e-3 | 约 3.865 倍 |
| dg | 3.5887e-3 | 约 5.573 倍 |
| dh0 | 2.2018e-4 | 约 36.33 倍 |

[evidence/G5/scratch/r3_logs/verify_*.json，六形状当前记录；前向数字沿用正式验证记录]

dk 是最紧的反向输出，改动其操作数精度、累加顺序或携带输入时应提高审查优先级。dv、dh0 的大裕度也不能兑换成随意舍入，因为误差可能经其他梯度间接传播。裕度是特定形状与输入分布下的观测，不是对未来配置的证明。

### 4. 跨进程逐位与交错终判封住另外两类漏洞

低 ulp 竞态可以稳定地躲在宽容差内。终判协议固定同一 cubin、输入与发射几何，在全新 CUDA 上下文和模块加载下跨进程重复 5 次，要求各阶段输出逐位一致。验证史记录了五个真实竞态，均在数值容差内，多数只有 1–3 ulp；bpre 的单 kernel 新鲜模块实验曾在 1000 次中出现 14 次双稳态，修复后为 0。wyb 的另一处竞态在同进程十次循环中未复现，说明进程内重复不足以覆盖模块轮换打开的时序窗口。[evidence/G5/bwd_report.md:1516-1586]

补屏障约付出 0.3% 的 T2048 端到端代价，T512 上处于窗口噪声内。[evidence/G5/bwd_report.md:1675-1685] 这笔性能成本换来对陈旧共享读、重 staging 与 fragment 时序的可见性。逐位一致仍只证明已测执行没有漂移，不能替代不同输入下的数值验证。

性能终判采用同一进程、同一窗口、逐位相同输入的本方与基线交错成对计时。逐对比值的中位数与中位数之比是两个估计量，报告时必须分别命名。符号检验直接用胜负对数计算精确二项尾概率：

```python
def sign_test_p(ratios, ref=1.0):
    wins = sum(1 for r in ratios if r > ref)
    losses = sum(1 for r in ratios if r < ref)
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    p_lt = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return round(min(1.0, 2.0 * p_lt), 6)
```

[tools/gdn_final_adjudicate.py:80-94，省略函数 docstring]

读实现可见，平局的比值被丢弃，n 只计胜负对，k 取较少一侧，再计算双侧精确尾概率。正式协议使用 30 对并辅以效应量相对四分位距的可判性规则；记录字段 p 经四舍五入到六位小数后为 0，通常意味着未舍入 p < 5e-7，不表示数学概率为零。交错设计让共载和时钟漂移尽量同时作用于一对样本，绝对毫秒数仍需与窗口信息一起阅读。

### 5. 故障矩阵规定测试的取舍

| 故障 | 能发现的测试 | 不能发现或难以定界的测试 |
|---|---|---|
| D fragment 列映射错误 | 非零写集计数、fp64 twin | 只匹配 mma 文本的 FileCheck、资源报告 |
| 1–3 ulp 重 staging 竞态 | 跨进程逐位一致、针对性竞态隔离 | 常规数值容差、单次编译检查 |
| 生存期别名发生读写重叠 | 全阶段逐位对照、完整输出与 sanitizer | arena JSON 存在性检查、ptxas 编译成功 |
| 寄存器或 spill 导致驻留回退 | 最终 ptxas、SASS、占用画像 | 数值 oracle、纯 IR 结构断言 |
| 仿射重结合或缩放次序错误 | 真顺序 fp64 对照、六梯度端到端门 | FileCheck、指令族覆盖、占用测量 |
| 共载或时钟造成性能假象 | 同窗交错配对与可判性规则 | 分离进程绝对时延、数值正确性测试 |

FileCheck 与数值裁决器（oracle）的分工由此清晰：前者固定产物中应当出现或消失的结构，后者检查执行结果与数学参考的距离。资源报告补充机器预算，写集检查补充覆盖，逐位协议补充运行时序，交错计时补充性能证据。只有各层结论指向同一个可复现 cubin，验证塔才把“代码看似正确”收敛为可归档的发布依据。
