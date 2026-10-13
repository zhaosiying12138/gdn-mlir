## 二十三、生成器即发射器：用 Python 写 IR 的工程学

到本章为止，L1 与 L2 的全部 IR 都出自 Python 生成器：第九到十一章的算法文本来自 gdn_l1_gen.py，第二十一章的调度工件来自 gdn_l2_gen.py。为什么用逐行发射文本的方式生产 IR，而不写成 C++ pass？因为生成器在这里承担双重角色：它今天是产物——输出的 .mlir 文件直接进编译管线；它同时是规格——未来 L1 展开 pass 的输出格式。两头的合同写在两处头注里。生成器侧原文："This generator IS the output format of the future L1 expansion pass"（本生成器即未来 L1 展开 pass 的输出格式），并声明按形状与 chunk 参数化、全部输入数据内嵌为 memref.global 常量［tools/gdn_l1_gen.py:4-8，docstring］。方言侧对应陈述："届时本 pass 发出的 IR 与 tools/gdn_l1_gen.py 生成的 L1 文本逐行同构（生成器即本 pass 的可执行规格）"［dialect/lib/GdnPasses.cpp:1-9，文件头注］。规格即输出，两者没有漂移的空间。

### 1. 为什么用 Python 逐行发射

第一是迭代速度：改一行模板、重跑生成器、diff 产物，整个回路以秒计；C++ pass 的每次修改要重编方言、重链工具，回路以分钟计。当算法层还在演化（chunk 形态、掩蔽谓词、前代换结构都要反复试错）时，回路长度直接决定探索面积。第二是可审查性：产物是普通文本文件，进证据目录、进版本控制，review 与 CI 对它的处理和对源码完全一致。第三是零编译依赖：生成器只依赖 numpy，语义验证（第十八章的三方对拍）在纯 CPU 环境即可完整复现。C++ pass 并未缺席——L0 到 L1 的入口改写是真正的 pass（第十五章）——但算法本体的展开留在生成器里，直到形态稳定才冻结为 pass 的规格。

### 2. 发射器骨架：E 类

生成器的核心是一个二十四行的发射器类，管三件事：行缓冲、深度缩进、全新 SSA 名。

```
class E:
    """Tiny line emitter with fresh SSA names and indentation."""

    def __init__(self):
        self.lines = []
        self.depth = 0
        self.n = 0

    def __call__(self, s):
        self.lines.append("  " * self.depth + s)

    def v(self, hint=""):
        self.n += 1
        return "%" + hint + str(self.n)

    def block(self, header):
        self(header + " {")
        self.depth += 1

    def end(self):
        self.depth -= 1
        self("}")

    def text(self):
        return "\n".join(self.lines) + "\n"
```
［tools/gdn_l1_gen.py:161-184，逐字］

调用约定：实例可调用，e("一行 IR") 追加一行并按 depth 前缀缩进；e.block("scf.for %i2 = ...") 开域并加一层深度，e.end() 收域；e.v("sum") 发一个未占用的 SSA 名（%sum 加序号），名字冲突在构造上不可能。嵌套结构的缩进由调用序列唯一决定，不存在手写对齐这种可漂移的东西。linalg.generic 的双层花括号（属性体与块体）无法用 block/end 表达，就用显式的 e.depth 加一减一夹住块体——第十八章展品里那两行深度操作正是它。

### 3. f-string 模板与旋钮注入

形状、chunk、类型串等旋钮经函数参数进入 f-string 模板。A 矩阵 generic 的发射序列与产物逐行同构：

```
e("linalg.generic {")
e("      indexing_maps = [")
e("        affine_map<(i, j) -> (i, j)>,")
e("        affine_map<(i, j) -> (i)>,")
e("        affine_map<(i, j) -> (j)>,")
e("        affine_map<(i, j) -> (i)>,")
e("        affine_map<(i, j) -> (i, j)>")
e("      ],")
e('      iterator_types = ["parallel", "parallel"]')
e(f"    }} ins(%kkt, %gc, %gc, %beta_c : {t_CC}, {t_C}, {t_C}, {t_bc}) outs(%Abuf : {t_CC}) {{")
e.depth += 1
e("^bb0(%kk: f32, %gi: f32, %gj: f32, %bt: f32, %out: f32):")
e("%d0 = arith.subf %gi, %gj : f32")
e("%e0 = math.exp2 %d0 : f32")
e("%m0 = arith.mulf %kk, %bt : f32")
e("%m1 = arith.mulf %m0, %e0 : f32")
e("%i0 = linalg.index 0 : index")
e("%j0 = linalg.index 1 : index")
e("%lt = arith.cmpi slt, %j0, %i0 : index")
e("%r0 = arith.select %lt, %m1, %f0 : f32")
e("linalg.yield %r0 : f32")
e.depth -= 1
e("}")
```
［tools/gdn_l1_gen.py:451-473，逐字；阶段注释在 433-434 行］

模板行与 IR 行一一对应，读发射代码等于读产物。更高一层是 generic() 帮手：把下标映射、迭代类型、操作数表、块体回调参数化，反向链的十几个 generic 共用同一帮手，块体里还能引用外围作用域的 SSA 值［tools/gdn_l1_gen.py:701-732，帮手签名与文档］。L2 侧同构：gen_transform 按旋钮字典发射工件，multibuffer 的 factor 与 pipeline 的 depth 直接来自求解器解——旋钮注入点就是模板参数，解空间的每个点对应一份可读文本［tools/gdn_l2_gen.py:1121-1135］。

### 4. 确定性：同种子两遍，哈希相同

生成器不含无序来源：数据由 numpy 的 default_rng(seed) 产生，发射顺序由代码序唯一决定，哈希随机化（PYTHONHASHSEED 一类）不参与输出路径。因此同种子两次生成的 MLIR 哈希相同；全部旋钮关断时产物与默认配置逐字节等价，核对口径为 27/27 份 payload 与 18/18 个 cubin。这支撑全书的可复现主张：排名与验证引用的都是可存档的文件，两次完整编译逐字节一致。落盘侧的配套是哈希键控：构建把全部 GDN 环境旋钮加生成器与改写器源码做成 gdn_cache_key = sha256(...)，写进 full_spec.json；后来的构建若旋钮集或生成器版本不同，重定向到目录名后缀加哈希前八位的新目录，旧缓存既不误用也不被覆盖。同旋钮命中缓存、异旋钮重定向、遗留目录信任，三种行为都测试过——这套键控堵住过"新验证读到旧缓存 cubin"的真实漏洞［evidence/G5/fwd_report.md:2292-2298、2785-2787］。

### 5. 生成物三件套与落盘

每个配置目录的生成物是三件套：payload.mlir（计算本体）、transform.mlir（调度工件）、config.json（旋钮值）——L2 生成器逐配置落盘这组文件［evidence/G2/gen_report.md:4-5，Deliverables 一节］。编译链在其上继续产出 kernel.ptx、kernel.cubin、ptxas_v.txt、sass.txt 与 meta.json（资源指标加指令直方图），文件清单与用途注释在脚本头部［scripts/run_l2.sh:8-14］。三件套的分离是第二十一章"调度即 IR"的物理形式：payload 不动、换一份工件即得新调度；哈希键控保证任何一份缓存都能对回它的旋钮环境。生成器、工件、缓存三者合起来，把"编译器怎么调度"从一次性的构建事件变成可归档、可复查、可再生的数据资产。
