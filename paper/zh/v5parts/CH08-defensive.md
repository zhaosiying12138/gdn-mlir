## 八、防御性求解：补全与拒绝

求解器的输出直接喂给编译与计时，一个非法值或一份残缺清单都会污染整条证据链——下游的排序、Top-K、实例化与裁决全部建立在它之上。防御性求解因此把求解器库自身的产物当作不可信输入：求解模型可能缺维（partial model 现象），判定可能被资源限额中断（unknown 截断）。两套防御——逐候选探测的补全，与解除限额后的重试加拒绝——背后是同一条工程立场：宁可拒绝产出，绝不出具残缺空间。

术语先立。partial model 指缺少部分变量指派的求解模型；见证（witness）指足以证明可满足性的那份部分指派；补全（completion）指为缺席变量确定取值的过程。

### 1. partial model 现象与成因

z3 返回的求解模型只包含它认为需要指派的变量。当某个变量的取值被其余指派唯一决定——比如 prec 在 dtype 与 mma 的组合下只剩一个合法值（第六章的耦合约束）——求解器可以不把它写进模型。z3py 的读回接口此时返回 None：一个空值，一个不抛异常的空值（docstring 特意注明这是 z3py 的 get_interp 行为，218-219 行）。该行为符合 z3 的契约：模型是可满足性的见证，只要它指派的变量足以验证 sat 即可，没有义务为每个声明过的变量给出取值。现象在读回代码里的形态很朴素：m[zv] 不抛错、不返回哨兵字符串，安静地给 None——若读回处直接 as_long()，崩溃点是下游某行类型错误，与病因隔着一层。源码注释把这段历史记录在案：G6 搜索成本测量中发现 enumerate 在 None 上崩溃（163-166 行引文档记录）。防御的前提是承认现象会再来：补全逻辑是读回协议的一部分，每次读模型都要走。

### 2. 补全算法：逐候选 SAT 探测（继承展品）

```
pin = [self.v[k] == z3.IntVal(v) for k, v in fixed.items()]
s.add(z3.And(pin))
for cand in self.domains[key]:
    s.push()
    s.add(self.v[key] == z3.IntVal(cand))
    ok = s.check() == z3.sat          # 第一个保持可满足的候选即补全值
    s.pop()
    if ok: return cand
```
［tools/gdn_solver.py:192-204（继承展品）；外层先解除 rlimit，避免探测被限额中断误判］

算法四步：把已读回的值全部钉住（pin 断言，And 连接）；按域的固定顺序逐个候选试探；每次试探 push 保存求解器状态、断言该变量等于候选、check 判定、pop 撤销；第一个保持整体可满足的候选即补全值。

这个构造的安全性来自判定对象本身：候选被接受的条件是"钉住的指派加该候选在全部约束下可满足"，而判定的约束集正是定义合法性的那一份——非法候选在 check 处即被否决，补全不可能向清单注入任何非法值。换一个说法，合法性无需在补全处重复实现，它被委托给同一个求解器的同一条 check。确定性有双重来源：域元组顺序固定，返回第一个合法候选；枚举遍历缺失键也按声明序进行（第五章）。存在性同样有保证：求解器返回的模型本身是 sat 的，缺席变量在其余值钉住后至少还有一个合法取值（函数 docstring 记录了这条论证，184-185 行）；万一全部候选被否决，RuntimeError 把它当作求解器契约违例上报，异常文本带完整的阶段、K、V 与钉住值（205-209 行），绝不静默返回。

### 3. 探测前解除限额与 push/pop 卫生

补全的探测运行在限额解除的求解器上：进入 _complete 先把 rlimit 置零，退出时在 finally 里恢复原值（187-189、210-212 行）。原因在探测的判定语义：探测的 check 若被 rlimit 中断，返回的是 unknown；代码把返回值与 z3.sat 比较，unknown 落在"非 sat"一侧，会被误读成"候选被否决"——一个其实合法的候选被限额误杀，补全可能跳到后面的候选甚至误报无解。解除限额让探测的每次判定都是完备判定，"非 sat"才严格等于"该候选确实非法"。

push/pop 的配对同样严格：外层一个 push 包住全部 pin 断言，每个候选一个 push/pop 对，finally 保证异常路径也恢复现场。补全结束后求解器回到枚举离开时的状态——约束集与限额都复原，阻塞子句照常在其上叠加。还有一个次序细节：补全发生在本轮阻塞子句加入之前，所以被钉住的当前解此刻仍满足全部约束，"至少一个候选合法"的存在性论证才成立。

### 4. unknown 截断：重试一次，然后拒绝（新展品）

枚举的另一个出口是 unknown。rlimit 耗尽时 check 返回 unknown，循环以 last_check 记为 unknown 结束——此时清单只是空间的截断，绝不能当作完整空间输出。solve 的处置：

```
if getattr(space, "last_check", "unsat") == "unknown":
    # rlimit abort BEFORE the space was exhausted: silently accepting
    # it would return a truncated space as if complete (the G6 flake
    # class).  Retry once with the cap lifted -- these spaces are
    # small (<= 1e3 configs), the uncapped pass costs well under a
    # second, and the retry is deterministic.
    rl_retry = True
    space = KnobSpace(stage, K, V=V, mode=mode, rlimit=0)
    combos = space.enumerate(limit=limit)
    if getattr(space, "last_check", "unsat") == "unknown":
        raise RuntimeError(
            f"solver returned unknown even without the rlimit cap "
            f"({stage} K={K} V={V}); refusing to emit a truncated "
            f"space of {len(combos)} configs")
```
［tools/gdn_solver.py:360-373，逐字（注释并入）］

三级处置。正常路径以 unsat 结束，空间干净穷尽。unknown 则整体重建：用 rlimit 为零的参数新建一个 KnobSpace 从头枚举——空间至多千级配置，无上限一遍远低于一秒（源码注释的原话：well under a second），且重试本身的确定性有保障（同种子同声明序，第九章）。重试仍 unknown 则抛异常，异常文本明写 refusing to emit a truncated space——拒绝产出。重试的代价核算也写在注释里：这些空间极小，无上限一遍对 0.8 秒的新形状调度预算（第四章）没有实质威胁——防御没有以牺牲编译期调度的速度承诺为代价。

工程立场在此一句话可以说尽：残缺空间里的排序没有意义。截断可能恰好截掉最优配置，其上的排名、Top-K、后续实例化与计时全部建立在一个看不见的洞上；把这样的清单交出去，失败要到几小时后的裁决环节才显形，且无法定位到"清单本来就残缺"这个根因。对照第七章的两态凭证：unsat 是穷尽的证明，unknown 什么都不证明——防御层做的事就是把这两态的差别从"实现细节"提升为"输出协议"。拒绝把一个不可见的失败变成一个立即的、带完整上下文的失败。防御路径的触发本身也被记账：enum_stats 记录模型数、partial model 数、补全次数三个计数（228-229 行），随输出 JSON 的 z3 字段落盘（403-406 行）——补全是否触发、触发多少次、重试是否发生（rlimit_retry_uncapped），复核者逐项可查。
