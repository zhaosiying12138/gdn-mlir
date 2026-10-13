## 十三、用 TableGen 定义领域方言

编译器的一切改写都作用于 IR，而 IR 里的每个 op 都需要一个定义：它叫什么名字、吃哪些操作数、吐哪些结果、满足什么约束。MLIR 允许用纯 C++ 写这些定义，但一个稍具规模的方言要手写成千上万行访问器、构造器与解析样板。TableGen 与其 ODS（Operation Definition Specification，操作定义规约）把这件事变成声明式描述：方言作者在 .td 文件里写下操作数与结果的类型约束和名字，mlir-tblgen 工具在构建期生成等价的 C++ 样板。gdn 方言只包含两个语义 op（gdn.fwd 与 gdn.bwd）、一个配置属性与若干硬件直通 op，体量恰好够作教学样本。

### 1. 方言骨架：先声明方言本身

定义 op 之前要先定义方言。方言是命名空间与注册单位的载体：名字决定 IR 里的前缀（gdn.fwd 的 gdn），cppNamespace 决定生成 C++ 类的归属（::gdn），dependentDialects 声明本方言的 op 在下降中会触碰的方言，注册时由框架一并加载。

```
def Gdn_Dialect : Dialect {
  let name = "gdn";
  let summary = "Gated DeltaNet (gated delta rule) operator dialect";
  let cppNamespace = "::gdn";
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = [
    "tensor::TensorDialect",
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "func::FuncDialect",
    "mlir::NVVM::NVVMDialect",
    "nvgpu::NVGPUDialect",
    "gpu::GPUDialect",
    "arith::ArithDialect",
  ];
}
```
［dialect/include/Gdn/GdnDialect.td:13-36，节选］

其一，dependentDialects 列出的八个方言正是 L1 展开将要使用的词汇表（linalg、scf、arith）加上机器层的词汇表（nvgpu、nvvm、gpu），声明在这里意味着"加载 gdn 方言时连带准备好它下降所需的一切"。其二，useDefaultAttributePrinterParser 让方言属性（下一章的 GdnConfigAttr）的解析与打印走方言统一入口，省掉手写 hook。

### 2. ODS 核心：dag 语法与操作数声明

op 定义的主体是两段 dag（有向无环图记法，TableGen 用圆括号加冒号表示"约束 : 名字"的列表）。GdnOps.td 为两个语义 op 定义了一个公共基类，然后各自展开：

```
class GdnOp<string mnemonic, list<Trait> traits = []>
    : Op<Gdn_Dialect, mnemonic, traits>;

def GdnFwdOp : GdnOp<"fwd", [Pure]> {
  let summary = "GDN forward (semantic entry)";
  let arguments = (ins
    AnyRankedTensor:$q, AnyRankedTensor:$k, AnyRankedTensor:$v,
    AnyRankedTensor:$beta, AnyRankedTensor:$g,
    Optional<AnyRankedTensor>:$h0,
    GdnConfigAttr:$config
  );
  let results = (outs AnyRankedTensor:$o, AnyRankedTensor:$ht);
  let hasVerifier = 1;
}
```
［dialect/include/Gdn/GdnOps.td:7-34，节选］

逐项拆解这段声明的语法要素。let summary 是文档字符串，会进入生成代码与 --help 输出；let description 可承载更长的约定说明——GdnFwdOp 的描述块写下了六个操作数与两个结果的形状契约：q 与 k 是 B×T×H×K 的四维张量，v 是 B×T×HV×V，beta 与 g 是 B×T×HV 的三维张量（beta 为写入强度，g 为对数域衰减、取值不超过零），h0 是可选的 B×HV×K×V 的 f32 张量，输出 o 形同 v、ht 形同 h0，且缩放因子 K 的负二分之一次方作用于归一化后的 q［dialect/include/Gdn/GdnOps.td:16-25］。把形状契约写进 op 描述的好处是文档与定义同源：IR 的读者用 op 的 --help 或源码注释即可读到完整签名，无需另查手册。

let arguments 的值以 (ins …) 开始，ins 是"输入侧"标记，其后每个元素形如 约束:$名字。AnyRankedTensor 是类型约束，要求该操作数为任意秩的张量；更细的形状与元素类型检查交给 verifier（第 4 小节），这里刻意放宽，使 L0 声明对 B·T·H 各维保持通用。`Optional<AnyRankedTensor>`:$h0 声明一个可选操作数：生成的访问器 op.getH0() 返回可空的 Value，调用方以 if (op.getH0()) 判断有无——第十五章的 pass 正是靠这个判断决定调用参数表的长度。Variadic 形式（同一约束匹配零个或多个连续操作数）在 gdn 方言的硬件直通 op 里也有实例：gdn.tma.async.load 的坐标参数写作 `Variadic<Index>`:$coordinates，接受一至五维的下标［dialect/include/Gdn/GdnHatch.td:102-115］。let results 的 (outs …) 与 ins 同构，声明两个张量结果 o 与 ht。traits 列表里的 Pure 来自 SideEffectInterfaces，声明该 op 无副作用，从而合法参与消除公共子表达式等变换。

### 3. AttrSizedOperandSegments：消解尾随可选槽的歧义

gdn.bwd 的操作数表比前向长得多，也正是它引出了 ODS 的一个经典难题：

```
def GdnBwdOp : GdnOp<"bwd", [Pure, AttrSizedOperandSegments]> {
  let summary = "GDN backward (semantic entry)";
  let arguments = (ins
    AnyRankedTensor:$q, AnyRankedTensor:$k, AnyRankedTensor:$v,
    AnyRankedTensor:$beta, AnyRankedTensor:$g,
    Optional<AnyRankedTensor>:$h0,
    Optional<AnyRankedTensor>:$A,
    AnyRankedTensor:$dout,
    Optional<AnyRankedTensor>:$dht,
    GdnConfigAttr:$config
  );
  let results = (outs AnyRankedTensor:$dq, AnyRankedTensor:$dk, AnyRankedTensor:$dv,
                        AnyRankedTensor:$dbeta, AnyRankedTensor:$dg, AnyRankedTensor:$dh0);
  let hasVerifier = 1;
}
```
［dialect/include/Gdn/GdnOps.td:40-59，节选］

问题出在三个 Optional 槽：h0、A 与 dht。当解析器读到一串具体操作数时，仅凭个数无法断定每个槽各自吃掉了哪几个——假设收到七个张量，可能是"h0 在、A 在、dht 缺"，也可能是"h0 在、A 缺、dht 在"；前两个可选槽相邻且类型约束相同，任何按位置的切分都站不住。AttrSizedOperandSegments 这个 trait 解决歧义：它要求 op 携带一个 operand_segment_sizes 整数数组属性，逐槽记录该槽捕获的操作数个数，访问器按段切分操作数表。打印成文本时该属性随 generic 形式一起输出，解析时据此还原槽边界；C++ 侧 getH0()、getA()、getDht() 的返回值因此无歧义。trait 同时让第十五章的下降 pass 能按"槽有无"而非"位置"组装 L1 调用参数。

### 4. verifier：用 C++ hook 检查合法域

ODS 声明的是结构性约束；更深的合法域检查交给 verifier。let hasVerifier = 1 会为生成的 op 类声明一个虚函数钩子：

```
LogicalResult gdn::GdnFwdOp::verify() {
  if (failed(verifyConfig(getConfig(), *this)))
    return failure();
  auto q = mlir::dyn_cast<RankedTensorType>(getQ().getType());
  auto v = mlir::dyn_cast<RankedTensorType>(getV().getType());
  auto beta = mlir::dyn_cast<RankedTensorType>(getBeta().getType());
  if (!q || q.getRank() != 4)
    return emitError() << "q must be rank-4 [B,T,H,K]";
  if (!v || v.getRank() != 4)
    return emitError() << "v must be rank-4 [B,T,H,V]";
  if (!beta || beta.getRank() != 3)
    return emitError() << "beta must be rank-3 [B,T,HV]";
  return success();
}
```
［dialect/lib/GdnDialect.cpp:50-63；verifyConfig 在 43-48 行，配合 Gdn.h:32 的 kAllowedChunkSizes = {16, 32, 64}］

写法上有三条纪律。第一，检查按"结构可达、语义合法"分层：类型是否张量由 ODS 约束兜底，秩与具体维度含义（q 与 v 是秩 4，beta 是秩 3）由这里补齐，chunk_size 落在 {16,32,64} 由 verifyConfig 检查。第二，报错用 op 自身的 emitError()，错误信息携带出错的 op 定位，测试可以按文本断言。第三，verifier 在每次 parse 之后、每个 pass 之后由框架自动调用，属于 L0 的验证合同——第四节表 4 中 L0 行的"verifier 合法域"即指此处；数值正确性则由 JAX fp64 规约在三方交叉中把守，两层各司其职。

同一套钩子在方言的其余 op 上有更多实例，写法完全一致。gdn.bwd 的 verify 目前只复核配置合法域（dialect/lib/GdnDialect.cpp:65-69），其操作数形状与前向同族、复用前向的检查思路。硬件直通 op 则各守各的硬件合同：gdn.setmaxnreg 的 count 必须是 8 的倍数且落在 24 至 256 之间，gdn.tma.async.load 与 gdn.tma.async.store 的坐标维数必须在一到五之间——两条检查分别写在 GdnHatch.cpp:246-252 与 254-266。这些数字全部来自 PTX 指令集对相应指令的约束，verifier 把它们挡在 IR 入口，使非法配置在进入下降管线之前就报错定位，而非在 PTX 汇编期才暴露难解的工具链错误。

### 5. assembly 形式：generic 打印与 #gdn.config<...>

op 有两种文本形式。未定义 let assemblyFormat 时只有 generic 形式：操作数表、属性字典、类型签名全部显式写出。gdn 方言的两个语义 op 刻意保持 generic 形式，冒烟测试即以此书写：

```
%o, %ht = "gdn.fwd"(%q, %k, %v, %beta, %g, %h0)
    <{config = #gdn.config<64, true, false, true, true, true, false>}>
    : (tensor<2x128x2x16xf32>, ..., tensor<2x2x16x16xf32>)
      -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
```
［dialect/test/smoke_l0.mlir:9-13，类型串有删节］

属性值 #gdn.config<64, true, false, true, true, true, false> 的拼写由 GdnConfigAttr 自己的 assemblyFormat 决定（下一章展品）。相比之下，硬件直通 op 多用自定义 assemblyFormat 换取可读性，例如 gdn.setmaxnreg 写作 "enum($direction) $count attr-dict"，gdn.bulk.wait_group 用 ($read^)? 的可选组语法容纳 UnitAttr［dialect/include/Gdn/GdnHatch.td:42、158］。两种形式等价：pretty 形式方便人读，generic 形式保证任何 op 都可无损往返，工具链默认用 pretty 打印、用 generic 做核对。

### 6. 从 TableGen 到 C++：mlir-tblgen 生成什么

.td 文件本身只是源。构建系统对每个 .td 调用 mlir-tblgen，产出 .inc 文件再被 C++ 包含：

```
set(LLVM_TARGET_DEFINITIONS Gdn/GdnOps.td)
mlir_tablegen(GdnOps.h.inc -gen-op-decls)
mlir_tablegen(GdnOps.cpp.inc -gen-op-defs)
mlir_tablegen(GdnHatchEnums.h.inc -gen-enum-decls)
mlir_tablegen(GdnHatchEnums.cpp.inc -gen-enum-defs)
```
［dialect/include/CMakeLists.txt:7-15，节选；完整清单（含 GdnDialect 与 GdnAttrDefs 段、GdnIncGen 公共目标）见第十六章第 2 节的全文展品］

-gen-op-decls 生成 op 类的声明：每个命名操作数与结果的访问器（getQ、getConfig）、builder、verifier 钩子声明，以及 AttrSizedOperandSegments 所需的段切分逻辑；-gen-op-defs 生成对应实现。-gen-enum-decls 为 GdnRegDir 这类枚举属性生成 C++ 枚举与字符串往返。这些 .inc 汇入公共目标 GdnIncGen，方言库将其列为依赖，保证生成先于编译（第十六章展开工程细节）。手工验证生成物是否如预期，可直接在构建目录上运行 mlir-tblgen -gen-op-decls 查看 stdout——这也是调试 ODS 写法的常规手段。效果归结为一句话：方言作者写下的约 60 行 op 声明，换出的是带全量访问器、类型检查与文本往返能力的 C++ 类，而所有后续 pass（第十五章）只需面向这些类编程。
