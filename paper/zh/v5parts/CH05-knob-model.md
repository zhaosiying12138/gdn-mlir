## 五、旋钮空间的建模

求解器读到的世界由建模决定。建模回答两个问题：哪些自由度交给求解器（决策变量），每个自由度允许取哪些值（域）。这份建模的质量决定下游一切——约束引用变量、枚举读取模型、补全遍历域、阻塞子句按声明序构造，全部挂在这份十维声明上。

术语先立。决策变量（decision variable）指求解器自由指派的符号量；z3.Int 声明整数符号常量、z3.Bool 声明布尔符号常量，求解之前它们没有值，只是约束里的名字。域（domain）指一个变量的全部候选取值集合。

### 1. 十维决策变量与固定声明序

七个整数变量加三个布尔变量，按一个写死的顺序声明：

```
# -- decision variables (fixed declaration order => determinism) --
dtype = z3.Int("dtype")            # 0 = bf16, 1 = f32
prec = z3.Int("prec")              # 0 = bf16, 1 = tf32
bt = z3.Int("tile_M")
bv = z3.Int("tile_N")
bk = z3.Int("tile_K")
warps = z3.Int("warps")
pipe = z3.Int("pipeline")
mma = z3.Bool("mma")
ws = z3.Bool("warpspec")
tma = z3.Bool("tma")
self.v = dict(dtype=dtype, prec=prec, bt=bt, bv=bv, bk=bk,
              warps=warps, pipe=pipe, mma=mma, ws=ws, tma=tma)
```
［tools/gdn_solver.py:98-110，逐字］

整数承载"从几个候选里选一个数"的维度：dtype 与 prec 各两个候选，tile 三维、warp 数、流水深度从各自域里选。布尔承载开关：mma、warp 专化、TMA。整数与布尔的分工本身是一个编码决策。dtype 只有 0 与 1 两个候选，表面上看布尔也装得下；用整数的理由有两条。其一，它要进入算术：共享内存字节模型按 dtype 选元素宽度（第六章的 esz 项），条件写法是 If(dtype == 0, 2, 4)，整数比较是最直接的载体。其二，统一的整数编码让多数读回路径走同一条 as_long 转换。布尔用在纯开关上，代价是三处代码都要做类型分派——阻塞子句对布尔维用 BoolVal、对整数维用 IntVal（254-256 行），域登记要分别写 (False, True) 与整数元组。编码的原则可以提炼成一句：进算术的用整数，纯开关的用布尔，两类的分派集中在一处处理。

self.v 这份字典是变量的唯一登记处，声明序即字典序——注释把意图写明：固定声明序服务于确定性。这个顺序被三处消费：枚举循环按它读取模型（for k, zv in self.v.items()，第七章）、补全按它遍历缺失键（第八章）、阻塞子句按它逐项构造。声明序因此是确定性控制面的第一根支柱（第九章）。

### 2. 域：从全局维与硬件宽度推导

域不硬写在求解断言里，先经三份模块级常量给出基域：

```
TILE_DOMAIN = (32, 64, 128)      # G3 knob-space definition (see report:
# gdn_l2_gen also admits M=16 and 8-multiples below 32; excluded here per
# the G3 knob-space contract)
WARPS_DOMAIN = (2, 4, 8)
PIPE_DOMAIN = (0, 2, 3)
```
［tools/gdn_solver.py:70-74，逐字（注释并入）］

tile 三档候选来自切块的常用粒度；warp 数三档对应 64/128/256 线程的发射几何；流水深度 0、2、三档覆盖关闭与两级软件流水。深度 0 的语义是关闭流水：字节模型里缓冲版本数写作 If(pipe > 0, pipe, 1)（第六章），0 与 1 折叠为同一份单缓冲。基域再按当前阶段与形状过滤——tile 必须整除对应的全局维，否则切块铺不满边界：

```
m_dom = [t for t in tile_domain if 64 % t == 0]
n_dom = [t for t in tile_domain
         if self.n_glob % t == 0 and t >= 8]
k_dom = [t for t in tile_domain if K % t == 0 and t >= 8]
s.add(z3.Or([dtype == d for d in (0, 1)]))
s.add(z3.Or([bt == t for t in m_dom]))
s.add(z3.Or([bv == t for t in n_dom]))
s.add(z3.Or([bk == t for t in k_dom]))
s.add(z3.Or([warps == w for w in WARPS_DOMAIN]))
s.add(z3.Or([pipe == p for p in PIPE_DOMAIN]))
```
［tools/gdn_solver.py:112-121，逐字；n_glob 与 n_gemm 的阶段分岔在 90-91 行（kkt 阶段 64、o 阶段 V）］

域的约束来源有三。其一，整除全局维：M 维的整除对象是常量 64，即 chunk 尺寸；N 维整除 n_glob（kkt 阶段 64、o 阶段 V）；K 维整除归约长度 K。tile 不整除全局维时边界产生残块，生成器不处理残块，因此直接从域里剔除。其二，宽度下限：N、K 两维加了 t >= 8 的下限，低于 8 的取值不构成有效的 tile 划分。其三，合同裁剪：生成器 gdn_l2_gen 本身允许 M=16 与 32 以下的 8 倍数，G3 旋钮空间合同显式把它们排除，把研究空间钉在报告口径上——注释里专门留了这段交代，防止后人把"求解器域"误读为"生成器能力"。

域登记两次。第一次作为约束：z3.Or 的成员断言告诉求解器每个变量只许取域内值。第二次作为 Python 数据：self.domains 字典把十个变量的候选存成元组（bt、bv、bk 三项直接复用 m_dom/n_dom/k_dom 的过滤结果，168-173 行），供补全算法按序探测。同一份过滤结果供两个消费方，约束与数据不会漂移。

### 3. 决策变量是旋钮：解即工件参数

决策变量选在旋钮层，带来的简化贯穿全链。枚举产出的向量经 to_config 翻译为配置字典（261-275 行），再经 config_name 拼成规范名（278-289 行）——kkt_bf16_mma_p2_t64x64x32_w4 这样一段字符串。规范名既是证据文件里的键，也是第三十八章 T2 结构匹配的检索键。向量到工件的映射是直接的：pipe 同时供 transform.memref.multibuffer 的 factor 与 pipeline_shared_memory_copies 的 depth 取参（第二十一章），tile 三维给 tile_sizes，warps 给发射几何，dtype、prec、mma、ws、tma 选择生成器路径与载荷形态。

这条设计的深层含义：求解器的解无需经过任何指令级翻译即可执行。变量是旋钮时，一个满足全部约束的指派本身就是 transform 工件的参数表；合法性已在求解层断言（第六章），实例化后的 ptxas 静态复核（第二十九章）作为第二道对账存在。求解器因此不包含任何指令调度逻辑——它解出的每个值都直接是生成器的一个参数。对照 TileLang 的做法（第四章）：变量落在指令时槽时，求解器之后还需要一套从时槽到代码的展开机制；变量落在旋钮层时，展开机制由既有的工件模板承担，求解器与生成器之间只剩一份字典翻译。

翻译是解与工件之间的全部黏合：

```
def to_config(self, kv):
    cfg = {
        "stage": self.stage,
        "dtype": "bf16" if kv["dtype"] == 0 else "f32",
        "prec": "bf16" if kv["prec"] == 0 else "tf32",
        "dims": {"K": self.K, "V": self.V if self.stage == "o" else 64},
        "tile": {"M": kv["bt"], "N": kv["bv"], "K": kv["bk"]},
        "warps": kv["warps"],
        "mma": kv["mma"],
        "pipeline": kv["pipe"],
        "warpspec": "producer_consumer" if kv["ws"] else "off",
        "tma": "on" if kv["tma"] else "off",
    }
    cfg["name"] = config_name(cfg)
    return cfg
```
［tools/gdn_solver.py:261-275，逐字］

翻译是逐维一一对应的：整数码转回字符串枚举（0 转 bf16、1 转 f32），布尔转回开关语义，dims 按阶段补齐形状参数。名字由 config_name 按固定模板拼出（278-289 行）——阶段、类型、mma 路径、流水深度、tile 三维、warp 数，warp 专化与 TMA 作后缀。同一向量永远拼出同一名字，排名文件、证据目录与第十七章 T2 结构卡因此共享同一套键；回读方向同理，knob_vector 把配置字典还原为向量（292-298 行），供枚举结果与生成器 INDEX 对账。解、配置、名字、工件四层之间的映射全部是纯函数式的字典翻译——这正是"决策变量是旋钮"在工程上的兑现：没有隐藏状态，没有需要解释的中间表示。
