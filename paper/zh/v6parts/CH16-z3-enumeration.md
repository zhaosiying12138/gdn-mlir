## 十六、穷尽而不失控：阻塞枚举、防御与确定性

有限旋钮域可以逐点列出，难点在于证明没有漏点，同时妥善处理求解器的部分模型和资源中断。`KnobSpace.enumerate()` 把每次可满足判定转化为一个完整向量，再用阻塞子句永久排除它；外部成本模型只消费这份清单。合法性、排名与终止凭证因此可以分别检查。

### 1. check 的三态与阻塞子句

`Solver.check()` 返回 `sat`、`unsat` 或 `unknown`。`sat` 表示当前断言集存在满足指派，随后才能调用 `model()` 取见证；`unsat` 表示当前断言集已经证明无解；`unknown` 表示本次判定没有得到结论，常见诱因包括资源限额或非线性算术。第三种结果不能当作空间为空。`sat` 也不要求解唯一，模型只是至少一个见证；这里的“模型”是变量指派，与用于预测时间的成本模型属于两种对象。

```python
while len(out) < limit:
    r = s.check()
    if r != z3.sat:
        self.last_check = str(r)
        break
    m = s.model()
    # ……按声明序读取，并补全 m[zv] 为 None 的维度……
    out.append(vals)
    block = [self.v[k] != (z3.BoolVal(v) if isinstance(v, bool)
                           else z3.IntVal(v))
             for k, v in vals.items()]
    s.add(z3.Or(block))
```
[tools/gdn_solver.py:230-257，省略模型读取与统计]

阻塞子句（blocking clause）对刚得到的完整向量逐维写“不等于”，再用 `Or` 连接。下一个模型只要有一维不同就可满足它，完全相同的向量则被排除。每轮新增一条子句，旧子句始终保留，所以不会重复输出。布尔值和整数分别用 `BoolVal`、`IntVal` 构造，避免把 Python 类型直接混进符号表达式。

【通用补充】假设只有 tile {32, 64} 与一个 mma 布尔量，空间共四点。得到 `(32, true)` 后，阻塞条件允许 `(32, false)`、`(64, false)`、`(64, true)`，恰好排除原向量。若误用 `And` 连接所有“不等于”，只有所有维都不同的点才能通过，会错误丢掉只改一维的合法点。四轮阻塞之后的 `unsat` 才对应这四点已经穷尽。

当所有合法向量都被阻塞后，新增的 `check()` 返回 `unsat`，`last_check` 留下终止证据。这证明的是“当前变量域、当前约束和全部阻塞子句下已无新点”。若循环因 `limit` 停下，或最后一次判定为 `unknown`，都没有这份完备性。即便正常得到 `unsat`，它也只覆盖模型定义的空间；no-MMA 字节公式若与生成器漂移，穷尽仍旧成立，现实合法域却需要重新对账。

### 2. partial model 的确定性补全

Z3 的模型可以省略对可满足见证不必要的常量解释，此时 `m[zv]` 返回 `None`。直接调用 `as_long()` 会把一个求解协议问题推迟成类型异常。当前实现先读取所有已给出的值，再按声明序处理缺失键；每补全一个值，就把它加入后续补全所要固定的指派。

```python
s = self.solver
saved_rlimit = self.rlimit
s.set("rlimit", 0)
s.push()
try:
    pin = [self.v[k] == (z3.BoolVal(v) if isinstance(v, bool)
                         else z3.IntVal(v))
           for k, v in fixed.items()]
    s.add(z3.And(pin))
    for cand in self.domains[key]:
        s.push()
        s.add(self.v[key] == (z3.BoolVal(cand)
                              if isinstance(cand, bool)
                              else z3.IntVal(cand)))
        ok = s.check() == z3.sat
        s.pop()
        if ok:
            return cand
    raise RuntimeError(
        f"z3 partial model: no legal completion for knob {key!r} "
        f"given {fixed!r} (space {self.stage} K={self.K} "
        f"V={self.V}); this is a solver contract violation -- "
        f"please report with the full command line")
finally:
    s.pop()
    s.set("rlimit", saved_rlimit)
```
[tools/gdn_solver.py:187-212]

外层 `push` 固定其余已读值，内层 `push/pop` 逐候选增加一个等式。第一个仍返回 `sat` 的域值就是补全结果，合法性由原约束集直接判定，无需在 Python 里复制一套规则。域元组顺序固定，多个合法补全时也会选同一个。补全发生在本轮阻塞子句加入之前，原来的 sat 模型保证至少存在一个候选；若全部失败，异常会携带旋钮、形状和固定指派 [tools/gdn_solver.py:177-186]。

探测前把 `rlimit` 置零很关键：若候选检查因限额返回 `unknown`，简单比较 `== sat` 会把它误当成非法候选。`finally` 恢复求解器状态和原限额，避免临时固定值泄漏到后续枚举。输出中的 `enum_stats` 记录模型数、缺失解释和补全探测次数，使这条防御路径可被审计 [tools/gdn_solver.py:228-229、250-252]。

### 3. unknown 重试与拒绝截断

主枚举遇到 `unknown` 时，已经产出的列表只是前缀。`solve()` 不会继续给它排名，而是从头重建无资源上限的空间并重试一次。

```python
if getattr(space, "last_check", "unsat") == "unknown":
    rl_retry = True
    space = KnobSpace(stage, K, V=V, mode=mode, rlimit=0)
    combos = space.enumerate(limit=limit)
    if getattr(space, "last_check", "unsat") == "unknown":
        raise RuntimeError(
            f"solver returned unknown even without the rlimit cap "
            f"({stage} K={K} V={V}); refusing to emit a truncated "
            f"space of {len(combos)} configs")
```
[tools/gdn_solver.py:360-373，省略注释]

`rlimit` 是求解器内部工作量计数，默认 20,000,000，与墙钟超时相比更适合重放 [tools/gdn_solver.py:67-68、93-96]。无上限重试仍返回 `unknown` 时，异常明确拒绝输出截断空间，并带上阶段、形状和已得配置数。`rlimit_retry_uncapped` 随 JSON 落盘，复核者能看到结果是否走过重试 [tools/gdn_solver.py:396-406]。这项防御保护清单完整性，不保证每个模型点都能编译或数值正确，后两者仍由生成、汇编和设备验证负责。

### 4. 非线性目标放在求解器外排序

项目没有把这项目标直接交给 `Optimize`。成本模型含最大值、混合、除法以及占用相关分岔，合法域公式也含变量乘积；把它们一并塞入优化搜索，会增加非线性算术与 `unknown` 的处理负担。当前空间只有数百个归档点，逐点预测更便宜，也能保留每项 `terms_ms` 供审计。这个选择不表示 `Optimize` 不能处理任何非线性问题，而是利用小空间把判定完备性与目标求值解耦 [tools/gdn_solver.py:14-22]。

```python
ranked.sort(key=lambda r: (round(r["predicted_ms"], 12), r["name"]))
for i, r in enumerate(ranked):
    r["rank"] = i + 1
```
[tools/gdn_solver.py:393-395]

外部排序键先取预测毫秒并舍入到十二位小数，再按规范配置名排序。第一键决定预测优劣，第二键为相同预测值提供稳定次序；排名不再依赖 Z3 返回模型的先后。每个点保留配置、预测时间、成本分解和 smem 模型值，因而可以独立重算排序 [tools/gdn_solver.py:379-409]。

### 5. 确定性、时间数字与结论范围

求解器设置 `random_seed=7`，声明序、域顺序、缺失键处理顺序和阻塞子句叠加顺序固定；终排序又用 `(predicted_time, name)` 的具体实现消除模型顺序影响 [tools/gdn_solver.py:67-68、93-110、254-257]。G3 对 368 项排名的名称顺序和预测时间做了逐字节一致性复核，G6 重跑复现了 368/465 空间及首选 [evidence/G3/solver_report.md:60-64；evidence/G6/search_cost.md:24-28]。逐字节比较应指向排名有效载荷；`enumeration_seconds` 是墙钟观测，跨环境比较整份 JSON 时需单独处理。

可操作的复现检查要固定阶段、K/V、形状、标定输入、模型形式和 Z3 版本，连续求解两次，再比较配置名序列、预测值的序列化字节、成本分解及结束态。若只比较 Top-1，排序中部漂移会被漏掉；若只比较枚举序列，又会漏掉外部排序的不确定性。完备枚举配合全序排序还带来一个可检验性质：求解器内部给出模型的顺序即使变化，最终排名仍应相同；若清单被截断，这个性质失去前提。

时间数字也与结论层次绑定。G3 记录 kkt K=128 的 368 点枚举 0.29 秒，o K=128/V=128 的 465 点为 0.33 秒，o K=64/V=64 的 204 点为 0.15 秒 [evidence/G3/solver_report.md:72-83]。G6 在同一环境重计前两个空间，枚举分别为 0.307 和 0.379 秒，含全清单成本模型排序的完整求解分别为 0.378 和 0.444 秒；kkt 与 o 两次 stage-kernel 求解合计约 0.8 秒 CPU [evidence/G6/search_cost.md:17-40]。

0.8 秒与 fla 前向 16.55 秒、反向 23.26 秒、合计 39.81 秒设备首搜的对比，比较的是两份搜索/排序墙钟账本：本方对 stage-kernel 候选做离线枚举和排序，fla 对自己的候选逐个编译并上机计时。它们不是同一候选集合，也不是同形状受控对拍；约 50 倍只是这些记录的搜索成本比值，不是完整算子的运行加速比。它不包含一次性标定，也不代表完整算子无需 GPU 验证。完整算子的发布配置另外使用实测 A/B，阶段模型在 18 个测点上的 Spearman 0.8962 也只衡量该测点集的排序质量 [evidence/G6/search_cost.md:103-111；evidence/G3/GATE_REPORT.md:9-10]。此外，368/465/204 是 G3 模型下的归档数量；生成器与 no-MMA 资源公式对账完成前，不能把这些数量提升为当前所有路径无条件完备的合法空间。

### 6. 【练习】给 limit 饱和增加显式防线

现有限额只控制 `while len(out) < limit`，它本身不是饱和检测。请设计一个结束态 `limit-saturated`：达到上限时再判定一次，区分“恰好枚举完”“仍有未枚举点”和“无法判定”；只有最后的 `unsat` 才允许进入完整空间排名，后两种情况分别拒绝输出或进入既定无上限重试。测试至少覆盖 limit 为 0、1、空间大小减一、空间大小和空间大小加一，并检查异常信息及 JSON 结束态。这个练习描述待实现的防护，不代表现有项目已经具备它。
