## 六、约束工程

约束是领域知识进入求解器的唯一通道。每条约束对应一条可以独立陈述的合法性事实——旋钮之间的语义依赖、工件拼写的对齐要求、硬件资源的容量上限。约束工程的目标是让这份知识可审计：断言列表与文档一一对应，任何一条可以被单独质疑、删除、验证。KnobSpace 构造器里的约束按耦合、结构、资源三类展开，前两类决定"拼写上存在的东西是否语义成立"，第三类决定"语义成立的东西是否装得进硬件"。

术语先立。断言（assertion）指加入求解器的 z3 表达式；Implies 是布尔蕴涵（前件成立则后件必须成立）；z3.If 是条件项（按布尔条件在两个表达式里选一个的符号写法）。

### 1. 耦合约束：旋钮间的语义依赖

某些旋钮只在另一些旋钮的取值下才有独立含义。精度旋钮 prec 是典型：tf32 只对 f32 数据配 mma 的组合存在——

```
# precision is a real knob only for dtype=f32+mma (tf32); bf16 mma
# is bf16-only, f32 is tf32-only, and mma=off removes the knob.
s.add(z3.Implies(dtype == 0, prec == 0))
s.add(z3.Implies(dtype == 1, prec == 1))
s.add(z3.Implies(z3.Not(mma), prec == dtype))
```
［tools/gdn_solver.py:123-128，逐字（注释并入）］

三条断言的净效果：prec 拥有独立自由度当且仅当 dtype 为 f32 且 mma 为真；其余组合下 prec 被固定，枚举不会产出"bf16 数据配 tf32 精度"这类拼写上存在、语义上无意义的配置。warp 专化与 TMA 的耦合同理（139-145 行）：ws 只对 kkt 阶段的 bf16 加 mma 开放——注意这里的阶段条件是编码期就坍缩的 Python 布尔（stage 是构造器参数，构建 kkt 空间时该项直接为真，构建 o 空间时整条蕴涵的前件永假）；tma 蕴含 ws；ws 形态下 pipe 被固定为 2——warpspec 载荷用 identity 工件（第二十一章），流水在该形态下不可观测，固定到内建约定是为了避免枚举出行为相同、名字不同的重复配置。耦合约束消灭的是冗余坐标：它们缩小的是编码的膨胀，语义空间本身没有变化。

### 2. 结构合法性约束（继承展品）

结构约束守住工件拼写的硬性要求——mma fragment 的对齐与流水深度的上限：

```
s.add(z3.Implies(mma, bt % 16 == 0))                     # M 对齐
s.add(z3.Implies(z3.And(mma, dtype == 0), bk % 16 == 0)) # bf16 K 对齐
s.add(z3.Implies(z3.And(mma, dtype == 1), bk % 4 == 0))  # tf32 K 对齐
s.add(z3.Implies(mma, bv % (8 * warps) == 0))            # N 对齐
s.add(z3.Implies(z3.And(mma, pipe > 0), K >= bk * pipe)) # 流水深度 ≤ K 趟数
```
［tools/gdn_solver.py:131-137（继承展品）；紧随的 137 行给 nomma 路径单设 K >= bk］

五条断言对应五个物理事实：M 维按 16 对齐，源自 mma fragment 的 M 高度；K 维按数据宽度打包对齐，bf16 每 16 个元素一组、tf32 每 4 个一组，来自 mma 指令的操作数布局；N 维按 8 乘 warps 对齐，每个 warp 至少分得一个完整的 N 侧 fragment 份额；流水深度乘 tile K 不能超过归约长度，multibuffer 的每份缓冲都要有对应的装载趟次；nomma 路径若开流水，同样要求 K 装得下一次 tile。注意这些断言全部带条件前件——Implies(mma, …)——因为 fragment 布局只对 mma 路径存在；前件让约束在无关分支上自动失效，mma 与 nomma 两条路径在同一约束集里互不挤占。这些约束把第二十一章工件拼写的合法性（factor 超过趟数、fragment 尺寸错位）从"编译器报错时才发现"提前到求解阶段——被拒绝的组合根本不进入实例化，错误信息从 ptxas 的回溯变成一行可指认的断言。

### 3. 资源约束：寄存器预算与 smem 字节模型

资源约束把硬件容量写成不等式。寄存器一条，先看展品：

```
esz = z3.If(dtype == 0, 2, 4)
threads = 32 * (warps + z3.If(ws, 1, 0))
# regs: <=255/thread is a ptxas given (all family members hit the
# cap); the SM-level budget is threads*255 <= 65536
s.add(threads * 255 <= 65536)
```
［tools/gdn_solver.py:148-152，逐字（注释并入）；esz 供下文 smem 模型复用］

不等式的三个成分各有出处。每线程 255 的上限来自 ptxas 的分配事实——源码注释记录该算子家族的 kernel 全员顶格，保守按上限记账。每 SM 65,536 个 32 位寄存器是 sm_120 的结构常数（gdn_costmodel.py 的 REGS_PER_SM）。线程数按 32 乘 warp 数计算，ws 为真时多计一支——warp 专化把一个 warp 让渡给生产者角色，驻留与寄存器预算照付。这条断言的直接效果写在模块头部：warps=8 加生产者 warp 即 288 线程，乘 255 超出 64K 预算，该组合在求解阶段即被剔除，无需等编译失败（docstring 40-41 行）。

共享内存按字节模型记账（继承展品）：

```
pair = (bt * bk + bk * bv) * esz                    # tile 对字节
smem_mma = z3.If(ws,
                 pair + z3.If(tma, 64*64*esz, 0) + 8,  # ws：单缓冲+TMA staging+mbarrier
                 pair * z3.If(pipe > 0, pipe, 1) * n_gemm)  # 流水=多缓冲版本数
smem_nomma = (bt * K + K * bv + bt * bv) * esz        # nomma：整 K 驻留
smem = z3.If(mma, smem_mma, smem_nomma)
s.add(smem <= cap)                                   # strict 100,352B / generator 220KB
```
［tools/gdn_solver.py:153-161（继承展品）；n_gemm：kkt 阶段 1 个 GEMM、o 阶段 2 个（90-91 行）］

三种形态各有推导。其一，mma 无 warp 专化：共享内存放 A 与 B 两个 tile（pair 字节，esz 随 dtype 取 2 或 4）；流水的每个版本各要一套缓冲（multibuffer factor 等于 pipe，pipe 为 0 视作 1）；o 阶段两个 GEMM 各配一对 tile——三项相乘即该形态的字节数。其二，ws 形态：生产者与消费者单缓冲，pair 一份即够；TMA 开启时加 64 乘 64 乘 esz 的输出 staging，再加 8 字节的 mbarrier 固定开销。其三，nomma：标量路径没有 tile 流水，K 维整段驻留，三个 tile 的并集照全尺寸记账。容量分两档：strict 档 100,352 字节——sm_120 每块 99KB 的 opt-in 上限减 1KB 驱动保留（gdn_costmodel.py 的 SMEM_TOTAL_B 减 SMEM_RESERVED_B，112-114 行）——守住真机可装载；generator 档 220KB，镜像生成器自身 validate() 的上限，用于与生成器口径对账。

### 4. 写成 If 与 Implies 的教学意义

先把三类约束的清单摆在一起，便于对账：

表 23-1：KnobSpace 构造器的断言清单
| 类别 | 条数 | 守护的对象 | 行号 |
|---|---|---|---|
| 域成员 | 6 | 各变量只取域内值（第五章） | 116-121 |
| 耦合 | 6 | prec 对 dtype 与 mma 的从属；ws、tma、pipe 之间的从属 | 126-128、140-145 |
| 结构 | 6 | fragment 对齐、流水深度上限 | 131-137 |
| 资源 | 2 | 寄存器预算、共享内存容量 | 152、160 |

二十条断言定义了第七章要穷尽的全部合法性。它们只用三个 z3py 构造写成：Implies（蕴涵）、And 与 Or（合取、析取）、If（条件项）。共同点是符号表达式：在没有具体取值时也成立，求解器对整张断言集做一次全局判定。这正是与 Python if-else 的分野——if-else 在编码期对具体值执行分支，而写约束时旋钮尚无值，分支无从执行；z3.If 把分岔留进公式，求解器在搜索中对每个候选指派分别求值。教学上的收益有三条。约束即断言：每条 s.add 可被单独验证——临时加一条假设再 check，空间的空与非空就是对该假设的裁决。约束即文档：断言列表与字节模型的文字推导逐行对齐，审阅者对照阅读即可。约束即测试：删除任一条重新枚举，合法空间尺寸的变化就是该条约束的独立贡献量——这张清单同时也是一张可逐条做差分的实验设计表。
