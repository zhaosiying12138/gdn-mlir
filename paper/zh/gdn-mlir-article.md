GDN-MLIR：Gated DeltaNet 算子的 MLIR 渐进式下降编译器
——从语义 op 到 sm_120a cubin：编译期调度、PTX 级覆盖证明与交错配对性能裁决
（本文全部代码与 IR 展品取自仓库真实文件，出处随展品标注）

本文描述 GDN-MLIR：Gated DeltaNet（chunk_gated_delta_rule，线性注意力家族中带门控 delta rule 的算子）前向与反向算子的编译器。编译器接受一个只声明语义的方言 op，经 L0（语义）、L1（算法结构）、L2（调度）、L3（张量核与流水）、L4（sm_120a 目标代码）五层渐进式下降，产出可直接装载的 cubin。实现全部建立在 stock MLIR（2026-10 LLVM 主干）之上，引入的自定义 NVVM op 为 1 个（预算上限 5 个）。

一、这是什么

这个仓库的交付物是一个编译器。所有前向与反向内核都是同一条 L0→L4 渐进式下降管线的产物；fla（Triton）与 FlashQLA（TileLang）两家手写实现的设计点，在编译器中被表达为调度层旋钮空间中的两个极端配置——复现一个参考实现的工作从重写内核变成调参。内核速度用于检验方法论的有效性。四个可检验的主张：

1. 表达力：旋钮空间在 PTX 指令族与结构两层覆盖两家手写参考；mma 指令签名层对两家双向一致；为此只需 1 个自定义 NVVM op（预算 ≤5）。
2. 调度：编译期 Z3 求解加一次性硬件标定的解析成本模型，每新形状调度 0.8 秒纯 CPU，两次完整编译逐字节一致，排序与实测的 Spearman 秩相关 0.8962。
3. 性能：前向三个基准形状在两类负载窗口下全面超过 fla Triton（共载窗口 2.125 / 1.221 / 1.349 倍）；反向小形状 1.72 倍。
4. 边界：反向大形状 0.41–0.49 倍，四类独立分析证明该值由梯度容差合同决定——在 fla 的容差下，理想执行的上限为 0.85–0.92 倍。

对标的哲学是 Google 的 XLA/TPU 栈：调度决策全部在编译期做出、编译产物不携带运行时搜索、每层 IR 独立可验证。四支柱对位：

表 1：与 XLA/TPU 四支柱的对位

| XLA/TPU 支柱 | 本编译器对应物 |
|---|---|
| 语义单一事实源（HLO + autodiff） | gdn 方言 L0 op + JAX fp64 可执行规约（三方交叉验证闭合） |
| 编译期确定性调度 | Z3 合法性求解 + 校准解析成本模型（固定种子，两次完整编译逐字节一致） |
| 分层可验证 IR | 每层一套独立验证：L1 三方交叉 → L2 静态资源复核 → L3 fp64 参考对照 → L4 统计裁决 |
| 布局系统 | 共享内存/寄存器布局旋钮族 + 生存期别名 |

二、证据速览

环境：RTX 5090 Laptop（Blackwell sm_120a，82 SM，24GB），WSL2，CUDA 13.x，bf16 输入 / fp32 累加，K=V=128。形状记号 B·T·H（批·序列长·头数）。基线为 fla 的纯 Triton 路径（FLA_FLASH_QLA=0）。

测量口径（全文适用）：所有性能对比采用交错终判（interleaved adjudication）——在同一 GPU 窗口内，将本方与基线置于同一进程、同一位相同的输入上交替成对计时，每形状 30 对；点估计取逐对比值的中位数，显著性由精确符号检验给出。共载（co-load）指测量窗口内 GPU 上常驻其它计算任务；空闲窗口指无其它计算任务。两类窗口分别报告。

1. 前向对 fla Triton

表 2：前向对 fla Triton（每形状 30 对交替计时）

| 形状 | 共载窗口 | 空闲窗口 | 绝对时间（本方 / fla，共载中位） |
|---|---|---|---|
| B1·T2048·H16 | 2.125×（30/30 胜，p<1e-9） | 1.611×（30/30 胜，p<1e-9） | 0.187 / 0.396 ms |
| B4·T2048·H16 | 1.221×（25/30 胜，p≈3.2e-4） | 1.067×（30/30 胜） | 0.821 / 1.003 ms |
| B1·T8192·H16 | 1.349×（29/30 胜，p≈5.8e-8） | 0.991×（持平，p=0.043） | 0.785 / 1.059 ms |

六个窗口的全部测量中，本方在逐对比值中位数意义上均未低于基线。共载窗口的比值偏高具有明确机制：共载把双方都压离理想时钟，fla 的 kernel 体积更大、受压更深；空闲窗口将双方放回理想状态，B1·T8192 相应回到持平档。

图 1（数据）：前向加速比柱状对比（共载/空闲两窗口）——B1·T2048：2.125× / 1.611×；B4·T2048：1.221× / 1.067×；B1·T8192：1.349× / 0.991×；虚线为 1.0 基准，数据对应表 2。

2. 反向对 fla Triton

反向以形状负载参数 u = B·H·(T/64)/512（总 chunk 数除以 512）刻画，u 越大跨 chunk 的状态扫描链越长。小形状 B1·T512·H4（u=1/16）本方 1.72 倍；u ≥ 1 的各形状为 0.411–0.488 倍，u=8 记录 0.462。时间对 u 的仿射拟合为：本方 0.138 + 1.996u 毫秒，fla 0.298 + 0.746u 毫秒——斜率差由精度合同强制的 fp32 中间量字节解释，完整论证见第九节。

图 2（数据）：反向阶梯折线（时间对 u）——本方：0.557 / 2.192 / 4.154 / 7.971 / 8.138 / 16.161 ms（u = 1/16、1、2、4、4、8）；fla：1.211 / 1.332 / 1.632 / 3.076 / 3.228 / 6.405 ms。终态空闲窗口复测（u=8）：本方 12.15 ms、fla 5.62 ms，比值 0.462。

3. 冷启动与部署形态

表 3：部署形态对比

| | 本方 | fla Triton |
|---|---|---|
| 部署形态 | AOT cubin，无运行时调优 | 运行时 autotune |
| 新进程冷启动 | 约 27 µs | 每形状前向 18.7–30.8 s + 反向 29.3–38.3 s |
| 缓存命中后 | 同上 | 0.65–5.67 s |
| 新形状调度决策 | 0.8 s 纯 CPU（GPU 时间为零） | 39.8 s 设备时间（48 个配置上机） |

50 倍为首搜口径；fla 磁盘缓存摊销后为 7.7–67 倍，两个口径并列报告。

4. 与 FlashQLA（TileLang）的相对位置

FlashQLA 前向是 sm_120 上现役最快的实现（0.139 / 0.404 / 0.355 ms，为 fla 的 2.16–2.38 倍）；本方前向（0.187 / 0.821 / 0.785 ms）明确快于 fla、落后 FlashQLA 前向约 1.4–1.5 倍（窗口口径不同，如实披露）。FlashQLA 的 sm_120 反向链在当前发布状态下无法启动：缺少 export，单 kernel 共享内存 112,640B 超出 101,376B 上限；加垫片修复后为 1.435 ms（B1·T2048），与 fla 相当。

5. 一句话覆盖结论

T1 指令族终态 33 族：8 YES / 19 PARTIAL / 6 NO，全部 PARTIAL 的残差属于指令变体而非能力缺口；mma 指令签名层对两家参考双向一致；T2 结构匹配：fla 极点与 FlashQLA 极点的每个结构要素都在旋钮空间内可生成。逃生舱预算 ≤5，实际支出 1（gdn.mma.tf32k8）。展开见第八节。

三、问题：GDN 内核为什么值得一个编译器

1. 算子

Gated DeltaNet 对每个 token 维护状态矩阵并按固定顺序递推：

h ← 2^g · h                （衰减在先，g ≤ 0）
b ← β ⊙ (v − k·h)          （delta 修正项）
h ← h + kᵀ ⊗ b             （外积写回）
o ← q · h

四步顺序不可交换，属于语义契约。按扫描直接实现意味着每个 token 读写一遍 K×V 状态，带宽不可承受。fla 一系的 chunk 化将递推改写为矩阵乘：一个 chunk 内 C 步递推对状态构成一个仿射映射，这是从扫描到矩阵乘的转换枢纽。chunk 化后的核心是三个对象。

第一，KKT 矩阵与前代换。chunk 内每个 token 的修正项 b_i 依赖当时的状态，而状态又依赖同一 chunk 内更早 token 的修正——把这 C 个相互依赖的方程联立求解，恰好得到一个下三角线性系统（每个 token 只依赖更早的 token）。该系统就是 KKT 矩阵 A = strict_tril( β ⊙ (k·kᵀ) ⊙ 2^(gc_i−gc_j) )：严格下三角，I+A 为单位下三角，其逆 T = (I+A)^(−1) 由前代换（按时间顺序逐行求解，无主元选取、无除法）求得。数值稳定性有三个结构性来源：键向量经 L2 归一化使内积不超过 1、β ∈ (0,1)、衰减因子单调不超过 1。

第二，跨 chunk 状态递推 S′ = (2^gcl·I − k̃ᵀw)·S + k̃ᵀu：衰减只通过状态上的 2^gcl 与键行上的 2^(gcl−gc) 两个显式因子进入，每个 chunk 将整段历史衰减折叠为对入口状态的一次乘法。第三，输出分解 o = o_intra + o_inter。反向传播中最微妙的一步是 dA 的收缩：单侧近似重复计数直达项并丢失交叉项（C=3 反例在符号层面将其否定），正确形式为双侧掩蔽收缩 dA = −strict_tril( Tᵀ · strict_tril(G) · Tᵀ )，与 fla 的 prepare_wy_repr_bwd 等价。

2. 现状与本编译器的回答

这个算子目前由手写内核服务：fla 基于 Triton，运行时 autotune，每形状首次搜索需要 39.8 秒设备时间；FlashQLA 基于 TileLang，手写 warp 专化调度。TileLang 自身的 z3 调度器只存在于 Ascend 后端，CUDA 侧退化为运行时搜索。手写路线有两个结构性代价：调度知识不可组合，每个新形状与新算子需要重新手写；调度决策推迟到运行时，产生冷启动与部署开销。

本编译器把调度从内核源码中抽出，变成编译期可求解的问题：调度以 IR 表达（transform dialect 参数化工件与方言属性承载全部旋钮），合法性由 Z3 判定，优劣由校准解析成本模型排序，正确性由分层验证把关。参考实现由此成为旋钮空间中的坐标。

四、编译器总览：L0→L4，每层回答一个问题

每一层固定一类决策、把另一类决策留给下层作为旋钮，并由一套独立验证把关；下层改动破坏语义时，可回退上层口径重新对照。表 4 给出各层职责，以下六小节逐层陈述最终实现；每层附真实的 IR 前后对照与 pass 核心代码。

表 4：分层职责与验证

| 层 | 固定的决策 | 留给下层的旋钮 | 验证 |
|---|---|---|---|
| L0 语义 | 数值约定与算子形态 | chunk_size、use_g/use_gk、初始/终态、varlen | verifier 合法域 |
| L1 算法结构 | chunk-WY 分解的 IR 形态 | solve 变体、中间量物化 vs 重算 | JAX fp64 规约三方交叉 |
| L2 调度 | 合法旋钮空间的枚举与排序 | 十维旋钮取值（tile/warps/流水/融合边界） | ptxas 静态资源复核 + 排序质量门槛 |
| L3 机器 | 数值到指令的映射、数据通路与存储几何 | mma 精度位点、staging 形态、bank 布局 | fp64 参考逐配置对照 |
| L4 目标与 PTX 后处理 | 物理资源事实（别名、寄存器上限、对齐） | — | 位级等价与跨进程逐位复现 |

1. L0 语义层：这个算子是什么

L0 只声明意图，展开选择全部留给下层。数值约定（衰减在先、β 作用于修正项、L2 归一化位置）固定于方言 verifier 与生成器；GDN 的数值约定密集，任何一条弄错都会产生形状正确而数值整体错误的静默偏移，因此语义只有一份。gdn.fwd 接收 q、k、v、β、g 与可选初始状态 h0，输出 o 与终态 ht；gdn.bwd 输出六个梯度 dq、dk、dv、dβ、dg、dh0。GdnConfigAttr 捆绑七个语义旋钮（chunk_size、use_g、use_gk、normalize_q、use_initial_state、store_final_state、is_varlen），它们改变算子的数学形态，与 L2 的调度旋钮分属两个层面。L0 的真实 IR（玩具形状）：

%o, %ht = "gdn.fwd"(%q, %k, %v, %beta, %g, %h0)
    <{config = #gdn.config<64, true, false, true, true, true, false>}>
    : (...) -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
［dialect/test/smoke_l0.mlir］

变换一：gdn-to-chunk（L0→L1 入口改写）。该 pass 把语义 op 替换为对 L1 私有函数的调用，config 属性原样透传。pass 的核心是一个标准的 MLIR OpRewritePattern：

struct FwdToLowering : OpRewritePattern<gdn::GdnFwdOp> {
  LogicalResult matchAndRewrite(gdn::GdnFwdOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> args{op.getQ(), op.getK(), op.getV(),
                             op.getBeta(), op.getG()};
    if (op.getH0()) args.push_back(op.getH0());
    auto fnTy = rewriter.getFunctionType(ValueRange(args).getTypes(),
                                         op.getResultTypes());
    if (auto module = op->getParentOfType<ModuleOp>())
      ensureL1Entry(module, rewriter, kL1FwdSym, fnTy);   // 按需生成声明
    auto call = rewriter.create<func::CallOp>(loc, rewriter.getStringAttr(kL1FwdSym),
                                              op.getResultTypes(), args);
    call->setAttr("gdn.config", op.getConfig());         // 语义属性透传
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};
［dialect/lib/GdnPasses.cpp:38-59；ensureL1Entry 先查符号表、缺失才在模块头生成 func.func private 声明，26-36 行］

说明：改写不展开任何算法，只建立 L0 与 L1 的调用边界——这一薄层让语义层可以被单独 verify、dump 与回归。对 smoke_l0.mlir 实际运行 gdn-opt --gdn-to-chunk 的真实输出：

module {
  func.func private @__gdn_l1_fwd_chunk(tensor<2x128x2x16xf32>, ..., tensor<2x2x16x16xf32>)
      -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
  func.func @smoke(%arg0: ..., %arg5: tensor<2x2x16x16xf32>) -> (...) {
    %0:2 = call @__gdn_l1_fwd_chunk(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
        {gdn.config = #gdn.config<64, true, false, true, true, true, false>}
        : (...) -> (tensor<2x128x2x16xf32>, tensor<2x2x16x16xf32>)
    return %0#0, %0#1 : ...
  }
}
［build-dialect/bin/gdn-opt --gdn-to-chunk 的真实 stdout，类型串有删节标记］

变换二：gdn-lower-hatch（硬件直通 op 的 1:1 下降）。12 个硬件入口 op 集中登记、一一映射到树内 op，下降实现是纯粹的枚举翻译。以 gdn.setmaxnreg 为例：

struct SetmaxnregLowering : OpRewritePattern<GdnSetmaxnregOp> {
  LogicalResult matchAndRewrite(GdnSetmaxnregOp op,
                                PatternRewriter &rewriter) const override {
    auto action = op.getDirection() == GdnRegDir::inc
                      ? NVVM::SetMaxRegisterAction::increase
                      : NVVM::SetMaxRegisterAction::decrease;
    NVVM::SetMaxRegisterOp::create(rewriter, op.getLoc(),
                                   (uint32_t)op.getCount(), action);
    rewriter.eraseOp(op);
    return success();
  }
};
［dialect/lib/GdnHatch.cpp:29-41；tma.async.load 的下降同文件 101-110 行——补一个零偏移谓词后直接替换为树内 nvgpu::TmaAsyncLoadOp，无 PTX 内联］

说明：这两个下降的共同形态是"薄"——只翻译记号，不生成调度，因此 11 个直通 op 的预算支出为零；全仓唯一需要 PTX 内联实现的是 gdn.mma.tf32k8（第八节第四小节）。

2. L1 算法结构层：它等价于哪条程序

chunk-WY 分解在 L1 表达为真实的 linalg 加 scf MLIR，在 CPU 上经 11 个树内 pass 下降后由 mlir-runner 执行。变换三：三角掩蔽写成控制流。A 矩阵的构造在单个 linalg.generic 内完成——用 linalg.index 取行列下标、cmpi slt 判严格下三角、select 归零，全程不物化 mask 张量：

linalg.generic {
  indexing_maps = [ (i,j)->(i,j), (i,j)->(i), (i,j)->(j), (i,j)->(i), (i,j)->(i,j) ],
  iterator_types = ["parallel", "parallel"]
} ins(%kkt, %gc, %gc, %beta_c : ...) outs(%Abuf : memref<64x64xf32>) {
^bb0(%a0: f32, %a1: f32, %a2: f32, %a3: f32, %o0: f32):
  %d0 = arith.subf %a1, %a2 : f32            // gc_i − gc_j
  %e0 = math.exp2 %d0 : f32                  // 2^(gc_i−gc_j)
  %m0 = arith.mulf %a0, %a3 : f32            // (k·kᵀ)_ij · β_i
  %m1 = arith.mulf %m0, %e0 : f32
  %i0 = linalg.index 0 : index
  %j0 = linalg.index 1 : index
  %lt = arith.cmpi slt, %j0, %i0 : index     // j < i：严格下三角
  %r0 = arith.select %lt, %m1, %f0 : f32     // 三角外归零
  linalg.yield %r0 : f32
}
［evidence/G1/l1_bwd_chunked_T128_C64.mlir:2538-2558；前置的 kkt = k·kᵀ Gram generic 在 2524-2537 行］

说明：换一个比较谓词即可得到 P 的构造（sle 含对角）——同一手法只差一个谓词。掩蔽因此是 IR 里的显式控制流，编译器与验证者都能逐行看到。

变换四：前代换写成三重循环。T = (I+A)^(−1) 的求逆同样直接落在 IR 上，最内层是携带累加变量的归约循环：

linalg.fill ins(%f0 : f32) outs(%Tbuf : memref<64x64xf32>)
scf.for %i2 = %c0 to %cC step %c1 {
  memref.store %f1, %Tbuf[%i2, %i2] : memref<64x64xf32>          // 单位对角
  scf.for %j2 = %c0 to %i2 step %c1 {
    %sum = scf.for %k2 = %j2 to %i2 step %c1 iter_args(%acc2 = %f0) -> (f32) {
      %aik = memref.load %Abuf[%i2, %k2] : memref<64x64xf32>
      %tkj = memref.load %Tbuf[%k2, %j2] : memref<64x64xf32>
      %ak  = arith.addf %acc2, arith.mulf %aik, %tkj : f32
      scf.yield %ak : f32
    }
    %neg = arith.negf %sum : f32
    memref.store %neg, %Tbuf[%i2, %j2] : memref<64x64xf32>       // T[i,j] = −Σ A[i,k]·T[k,j]
  }
}
［evidence/G1/l1_bwd_chunked_T128_C64.mlir:2559-2573，逐字］

说明：算法直接写在 IR 中，没有库调用、没有黑盒 op——"数学即 IR"是这一层的验证基石：任何下层改动破坏语义，都能回退到这份 CPU 可执行的 IR 重新对照。生成这段 IR 的 Python 侧发射序列与产物逐行同构（生成器即未来 L1 展开 pass 的可执行规格，两者一一对应；发射代码见 tools/gdn_l1_gen.py:451-473，阶段注释 433-434 行）。

验证：三方交叉闭合。JAX fp64 可执行规约（顺序递推、分块、jax.vjp 三实现互证）是语义单一事实源。JAX 自检 44/44 项：分块与顺序两条独立路径在 fp64 下相差 5.6e-16（舍入量级），VJP 梯度 1.2e-14，有限差分独立验证 2.3e-9——评分参考自身先被验证。MLIR L1 对 JAX：前向 7/7 用例约 6e-7（恰为 memref 打印的 6 位有效数字地板），反向六梯度最差 6.4e-6（验证门 2e-4）。JAX 对 fla naive：前向 1.2e-7、反向 2.3e-6（fla naive 内部固定 fp32，对照经 fp64 JAX 中转取 fp32 口径）。三方闭合。

3. L2 调度层：这条程序在硬件上怎么铺

调度即 IR：每份内核是一个参数化 transform 工件加一组旋钮取值的实例化。变换五：transform 工件实例化。下面是一份真实工件（kkt 阶段、bf16+mma、流水深度 2 配置）的完整 transform 脚本：

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %arg0
    transform.memref.multibuffer %allocs <factor = 2, skip_analysis>
    %func = transform.structured.match ops{["func.func"]} in %arg0
    transform.nvgpu.create_async_groups %func bypass_l1
    %loops = transform.structured.match ops{["scf.for"]} in %arg0
    transform.nvgpu.pipeline_shared_memory_copies failures(propagate)
        %loops depth = 2 peel_epilogue
    transform.yield
  }
}
［evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/transform.mlir:1-15；发射模板在 tools/gdn_l2_gen.py:1121-1135（factor 与 depth 即 Z3 解出的 pipeline 旋钮）］

说明：这份工件先对共享内存 tile 做双缓冲（multibuffer factor=2），再建异步复制组，最后对循环做深度 2 的软件流水——三个动作全部以 transform dialect 表达，旋钮值（2）由求解器注入。工件驱动的价值在于：调度是可 diff 的文本，复现一个配置等于复现一份工件。旋钮空间十维（dtype、mma 精度、tile M/N/K、warps、流水深度、mma 开关、warp 专化、TMA），合法性由 Z3 枚举、排序由成本模型给出，两者实现见第六节。

两个写法由实测排除：树内 create_async_groups 会在每组后立即插入 wait_group 0，复制结构性无法与计算重叠，异步 staging 须手动发射（第五节的做法）；staging 地址须线程跨步，block-uniform 形态会被每线程重复执行 256 次。

4. L3 机器层：数据怎么流进张量核

机器层的机制按功能分五组：数值到指令的映射（split-bf16 与 mma 发射器族）、算法级改写（扫描的仿射携带形式）、数据通路（异步 staging、乒乓预取、warp 专化与 TMA）、存储几何（bank 冲突、垫行与 32 位宽存）、驻留资源（生存期别名与寄存器上限）。每组机制的含义、代码与实测效果在第五节按内核族完整展开；这里给出贯穿全部五组的两个基础发射原语。

变换六：mma 发射与累加器串接。张量核乘法的最小发射单位是树内 nvgpu.mma.sync；生成器把它按 warp 贴图铺开，跨 k 段用 SSA 链接同一累加器：

prev = accs[(mi, ni)]
for kb in range(k0, k0 + Kred, 16):
    o = f"%mr_{name}_{mi}_{ni}_{kb}"
    e.l(f"{o} = nvgpu.mma.sync(%fa_{name}_{mi}_{kb}, "
        f"%fb_{name}_{ni}_{kb}, {prev}) mmaShape = [16, 8, 16] : "
        f"(vector<4x2xbf16>, vector<2x2xbf16>, "
        f"vector<2x2xf32>) -> vector<2x2xf32>")
    prev = o
［tools/gdn_bwd_gen.py:1886-1892，mma_gemm 内层］

说明：prev → o 的 SSA 数据流就是"同累加器累积"的机器级形态——第五节的 split-bf16 三链（hi·hi + hi·lo + lo·hi）正是靠 accs=maccs 参数把三次调用串进同一组累加器。落到 PTX 后，这条链呈现为连续三条 mma 共享前驱的 D 寄存器组（第五节第二小节有真实 PTX 对照）。

变换七：手动异步 staging。异步复制绕过树内 transform 手动发射，每个线程按跨步下标认领一个 16 字节块：

el = 8 if dt == "bf16" else 4            # 每块 8 个 bf16 = 16B
nch = Cc // el;  tot = R * nch
for kk in range(tot // nthr):
    p = tid if kk == 0 else tid + kk * nthr          # 线程跨步认领
    r  = p // nch;  cc = p % nch                     # 行 / 块列
    gcol = r * cm + cc * el                          # 全局列地址
    t = nvgpu.device_async_copy src[m, gcol], dst[r, dcol], el bypassL1
    toks.append(t)                                   # token 交调用方分组等待
［tools/gdn_bwd_gen.py:1687-1714，astage_tile 节选；astage_sync 封装在 1728-1740 行：发射 → create_group → wait 0 → barrier］

对应真实 PTX（kkt 配置，注意地址按线程跨步 +16/+32/+48、换行 +256）：

cp.async.ca.shared.global [%rd5],  [%rd549], 16;
cp.async.ca.shared.global [%rd14], [%rd549+16], 16;
cp.async.ca.shared.global [%rd15], [%rd549+32], 16;
cp.async.ca.shared.global [%rd16], [%rd549+48], 16;
cp.async.ca.shared.global [%rd17], [%rd549+256], 16;
［evidence/G4/gen_verify/kkt_bf16_mma_p0_t64x64x32_w4/kernel.ptx:606-610；bypassL1 对应 .cg 变体见 kkt_..._p2 同名文件 59 行］

说明：手动发射的原因在上一小节已述——树内 pass 的每组 wait_group 0 破坏重叠。配套的数据完整性规则（cp.async 写入的 tile 经 ld.shared 读回正确、经 ldmatrix 读回错误）与 warp 专化/TMA 的契约在第五节随内核族展开。

5. PTX 后处理层：stock 流水线遗留的四个问题

MLIR 语义完整下译之后、ptxas 定案之前，stock 流水线遗留四个真实问题——两个影响正确性（静态共享内存、生存期别名的实现载体）、一个影响驻留（寄存器上限）、一个影响发射效率（bank 几何）。856 行文本改写器 gdn_smem_fix.py 承载全部四个变换，每个变换都有真实的 PTX 前后对照，完整展开在第七节。这一层的工程原则：每步改写要么逐位相同、要么回退到正确形态、失败时显式报告；cache_key 取全部旋钮环境变量加源码指纹的哈希，键不匹配时缓存自动失效。

6. L4 目标层

命令链：gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a cubin-features=+ptx87 cubin-format=assembly" 产出 PTX，经后处理改写，再由 ptxas-blackwell -v 编译为 cubin，SASS 由 nvdisasm 审计。全链在 5090 上以树内 sm_80 transform-mma 集成测试冒烟通过。

五、内核族与制胜机制

本节按最终系统状态描述全部内核机制，文中 kernel 代号均为仓库内真实标识符。

1. 前向：三 kernel 分阶段、单一 cubin

前向最终形态为三个 kernel 装进单一 cubin（kernel 边界即流序同步，跨 kernel 的设备内拷贝约 0.1 毫秒每形状消失）：F1 做 token 并行的 WY 预计算（gc 扫描、A、T、u、w），F2 是全算子唯一的跨 chunk 串行链（状态常驻共享内存），F3 从中间量重算输出。单 launch 的 persistent 融合版本实现完整、数值全过，但 WY 预计算从 2048 个 CTA 的并行塌缩进 64-CTA 串行链，端到端 4.53 对 0.785 毫秒（T8192）；融合版本保留为融合边界旋钮的极限端设计记录，正式判定由三 kernel 方案承担。正式配置（三个判定形状同套）：mono, warps=4/bv=32 加环境旋钮 GDN_F1KS GDN_F2US GDN_WST GDN_ZFIX GDN_SCANDB GDN_F1ZF GDN_W32P。

关键机制（每个带机制与测量）：

1. F1KS——前向第一个每 SM 双 CTA kernel。F1 的停顿 24.3% 来自访存等待（long_scoreboard，DRAM 利用率仅 56.8%），属延迟受限，驻留翻倍直接转化为延迟下降：bf16 求解压缩共享内存加 B 操作数 K 分裂加生存期别名三步把申报池压到 46,352B ≤ 49,152B（双 CTA 门槛），F1 在 T8192 减少 53.5 微秒（278.7→225.2）。配套的 ZFIX/SCANDB 修复（清零步长、跨 warp 扫描原语的同步）保证双 CTA 配置逐位正确。

2. F1ZF——清零折进空闲线程。phase-0 的 8.5 微秒清零与 64 线程串行代入期间全程空闲的线程 64–127 并发执行；两者写集不相交，逐位一致；三形状各减 6.7 至 7.9 微秒，约为理论删除上限 8.5 微秒的 91%。

3. F2 加 16B 垫行——前向单笔最大杠杆。被 ldmatrix 读取的 tile 行距恰为 128B 整数倍时，8 行落入同一 4-bank 组形成 8 路冲突；每 tile 增加 8 列 bf16（16 字节）使行距错开，F2 从 473 降到 249 微秒，F1/F3 的全部 tile 同样受益。

4. F2US——u 段重回预取管线。2 字节装载配 4 字节跨距把 4KB 放大成约 8KB 内存扇区，且位于串行链上；u 段进入组 A 乒乓缓冲后 F2 减少 104.5 微秒（60 对测量零对变慢），T8192 的判定从 1.021 提升到 1.101 倍。

5. WST 加 W32P——无冲突的 32 位合并存储。F2 的 BV=32 tile 行距 80 字节。WST 把相邻 cell 配对成 32 位读写、指令与循环迭代减半；直接配对会让两个半 warp 的字地址落入同一 bank 组（实测反慢 4.3 微秒）。变换八：W32P 的 lane 到行重映射在生成器层解决——把 lane 0–15 映射到行 r、lane 16–31 映射到行 r+4：

hi16 = cmpi uge %lane, 16                     # 是否高半 warp
hi4  = select hi16, 4, 0                      # 行槽 +4
hcol = %warp + hi4                            # 半 warp 行槽 0..7
c0   = (%lane % 16) * 2                       # 列基：2*(lane%16)

def body3(i):                                 # 每线程逐元素循环
    k   = i / nthr                            # nthr=128，即 i>>7
    row = k * 8 + hcol                        # row = 8*(i>>7)+warp+4*(lane>=16)
［tools/gdn_full_gen.py:1038-1047，GDN_W32P 分支，CSE 会把 lane 常量提出循环］

说明：行距 80 字节即 20 个字，两个半 warp 的字地址分别落在 20r 至 20r+15 与 20r+80 至 20r+95；由 80 mod 32 = 16，两个区间恰好命中 32 个 bank 的两个互补 16-bank 半区，冲突清零。整个变换是纯下标算术，不影响共享内存池与占用；F2 在七个场景减少 17.9 至 22.6 微秒，与带宽核算一致（0.176 µs/MB × 100.6 MB ≈ 17 微秒上限）。WST 的 PTX 前后对照见第七节第四小节。

时间去向：F1 在 T8192 约 227 微秒的逐段删除探针测量（删除一个段、测整 kernel 时间差）给出 uw 段 116.5、kkt 59.5、solve 41.3、gc 扫描 17.6、phase-0 8.4 微秒；各段之和超过总时延，因删除收益含重叠成分，相对排序可靠——临界路径在 uw 段，上文五个杠杆也集中于此。

图 3（数据）：F1 逐段成本横条（T8192，微秒）——uw 段 116.5；kkt 59.5；solve 41.3；gc 扫描 17.6；phase-0 8.4。

2. 反向：8-kernel 链

终态反向链为 8 个 kernel：wy → fwdscan → caryM → bpre → dhs → dkb → dqkwg → wyb。wy 重算 WY 因子（六个梯度均穿过其写入的张量）；扫描族五个 kernel 执行两条方向相反的状态扫描；计算族两个 kernel 组装梯度——dqkwg 是全链最大的 kernel（dq、dk_local、dW、dvp_all），wyb 执行 dA 双侧掩蔽收缩、产出最终 dk、dβ、dg、dv。整链从基线 58.88 毫秒优化到 2.15 毫秒；数值侧同步改善，wy 链的 dk 误差从 1.691e-3 收紧到 1.284e-3，dβ、dg、dh0 的容差裕度扩大 23、61、36900 倍。

机制一：扫描的仿射携带改写。反向 dh 递推每 chunk 原需 8 段 GEMM 的串行链。代数重结合（非近似，fp64 对照 2–3 ulp）将其改写为携带形式 Xₙ₋₁ = Aₙ·Xₙ + Bₙ；复合满足结合律且每个块只与其前驱交互，于是拆成三部分：携带算子的并行预复合（caryM、bpre——相邻块的 Aₙ 先乘）、每步仅 1 个 GEMM 的携带链（dhs）、部分和的独立重算（dkb）。两个数学性质保证正确与可并行：算子超收缩（谱范数不超过 2.3e-18）使复合误差不随链长放大；Aₙ 与 v 切片无关，一次复合服务全部列切片。扫描族整体从 3.13 毫秒降到约 1.02 毫秒。

机制二：split-bf16 hi/lo mma。反向链的多数 GEMM 有一侧操作数必须是 fp32 中间量——这些中间量存在大数相消，直接以 bf16 存储会在读取时破坏精度（第九节的误差预算给出完整分析）。张量核对这类乘积提供三种精度：bf16（7 位尾数，截断指数域）、tf32（10 位尾数）、fp32；tf32 在 dkb 位点使 dk 输出误差达到 1.04e-2，超出 8e-3 容差。最终方案把 fp32 值拆成两个 bf16（一处 GEMM 调用位置下文简称位点）：hi = bf16(x)，lo = bf16(x − hi)，hi+lo 在 fp32 中精确可加（8+8 位尾数装得进 24 位）；乘积展开为 hi·hi + hi·lo + lo·hi 三条 bf16 mma 以 fp32 累加器链接，lo·lo（约 2⁻³²）舍弃。变换九：hi/lo 拆分的发射是两遍扫描——先 truncf 出主部，再对残差 truncf：

# 第一遍：hi = bf16(v)
%sh = tid + j*nthr;  i, j = %sh/Cc, %sh%Cc
%v  = load src[i, j]              # 可选乘精确 2 的幂列尺度 cvec
%sht = arith.truncf %v : f32 to bf16
store %sht, hi[i, j]
gpu.barrier                       # hi 全部落盘后才能算残差
# 第二遍：lo = bf16(v − hi)
%h  = load hi[i, j]
%r  = arith.subf %v, %h : f32
%slt = arith.truncf %r : f32 to bf16
store %slt, lo[i, j]
［tools/gdn_bwd_gen.py:1398-1422，_split_bf16，两遍间的 gpu.barrier 在 1410 行］

说明：拆分必须两遍——残差依赖同地址的 hi 已写入，屏障保证整块一致性。三链的调用点（dkb 位点）把 vnew 与 h 各拆一次，然后三次 mma_gemm 串进同一累加器：

_split_bf16(e, vh, vl, vngb, C, BV)      # vnew → hi/lo
_split_bf16(e, hh, hl, cur, K, BV)       # h     → hi/lo
maccs = mma_gemm(e, "dkBhh", C, K, BV, 8, vh.name, vh.ty, hh.name, hh.ty, accs=maccs)
maccs = mma_gemm(e, "dkBhl", C, K, BV, 8, vh.name, vh.ty, hl.name, hh.ty, accs=maccs)
maccs = mma_gemm(e, "dkBlh", C, K, BV, 8, vl.name, vl.ty, hh.name, hh.ty, accs=maccs)
［tools/gdn_bwd_gen.py:4246-4258，dkB 站点 SPLIT 分支］

落到真实 PTX，三链呈现为连续三条同型 bf16 mma——A 片段 {%r280-283} 为 hi、{%r284-287} 为 lo，B 片段 {%r332,333} 为 hi、{%r334,335} 为 lo，恰好构成 hi·hi、hi·lo、lo·hi；三条的 C 操作数 {%r300-303}/{%r304-307}/{%r308-311} 正是上一 k 段对应三条的 D，同一组累加器寄存器跨 k 传递：

mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {%r336, %r337, %r338, %r339}, {%r280, %r281, %r282, %r283},
    {%r332, %r333}, {%r300, %r301, %r302, %r303};        // hi·hi
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {%r340, %r341, %r342, %r343}, {%r280, %r281, %r282, %r283},
    {%r334, %r335}, {%r304, %r305, %r306, %r307};        // hi·lo
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {%r344, %r345, %r346, %r347}, {%r284, %r285, %r286, %r287},
    {%r332, %r333}, {%r308, %r309, %r310, %r311};        // lo·hi
［evidence/G5/scratch/r21b_ks2_T8192/dkb/kernel.ptx:336-355，dkb 即 split-bf16 位点内核；寄存器组对应关系为逐条核对］

图 4（数据）：dkb 位点 dk 输出误差对比（对 fp64 参考，对数刻度）——tf32 单精度路径 1.04e-2（超出容差线 8e-3，被否决）；split-bf16 路径 1.73e-3（容差之内）。split-bf16 乘积残差约 2⁻¹⁷、比 tf32 低约 30 倍，dkb 内核同时从 0.96 提速到 0.25 毫秒。

两条代数性质各省一条 mma。其一，行尺度折入 epilogue：缩放因子只依赖输出行时，缩放完全移出缩并在 epilogue 做一次标量乘，dvp_bh 位点由 3-mma 降为 2-mma。其二，2 的幂尺度可交换性 split(v·2^d) = split(v)·2^d 精确成立（生成器 docstring 有证明，tools/gdn_bwd_gen.py:941-949），对应的转置融合拆分把列尺度直接乘在装载值上：

v = load src[m, col(j)]                 # 全局装载
if cvec: v = v * ldsc(cvec, s)          # 精确 2 的幂列尺度，可进 split
%puh = arith.truncf v : f32 to bf16     # hi
store %puh, hi[j, s]
%puw = arith.extf %puh : bf16 to f32
%pul = arith.truncf (v − %puw) : f32 to bf16   # lo = bf16(v − hi)
store %pul, lo[j, s]
［tools/gdn_bwd_gen.py:956-969，stage_split_T 循环体；2-mma 用法见 7652-7658 行注释：A 侧 raw bf16 不拆、行尺度折 epilogue］

说明：caryA 的预乘 tile 由此退化为原样拷贝——共享内存从 82,176B 降到 28,592B，寄存器从 255 降到 76，该 kernel 提速 5.00 倍。这是改结合方式而不改任何值的机器级表达。

机制三：生存期别名与寄存器上限。每 SM 同时驻留两个线程块要求每 kernel 静态共享内存池不超过 50.2–50.7 KB；算法 tile 的生存期拼接远超上限，任意时刻活跃字节却在门槛之下。别名变换的完整机制、布局表与 PTX 前后对照在第七节第二小节展开。效果：dqkwg 池 100,864 降到 47,616B、caryA 82,176 降到 28,592B、wyb 100,096 降到 45,056B，三个 kernel 全部达成双 CTA 驻留（dqkwg 实测每 SM warp 数 7.59 升到 14.53）；配套的 .maxnreg 注入把寄存器逐 kernel 定在 128，占用从 15.02% 升到 29.17%。两个名字相近的机制各司其职：.maxnreg 是 PTX 后处理注入的逐 kernel 寄存器上限；gdn.setmaxnreg 是 warp 专化下生产者与消费者 warp 之间的寄存器再分配（实测某 kernel 溢出从 22,844B 降到 48B）。

图 5（数据）：反向三大 kernel 静态共享内存池（别名前 → 别名后，门槛 50.2 KB）——dqkwg 100.9 → 47.6 KB；caryA 82.2 → 28.6 KB；wyb 100.1 → 45.1 KB。

3. 两个负结果与其物理

失败按机制记录、按测量关闭。占用提升是杠杆还是空转，取决于 kernel 受限于延迟还是带宽：F1 停顿的 24.3% 来自访存等待（DRAM 利用率仅 56.8%），属延迟受限，驻留翻倍带来 53.5 微秒收益；F3 每次加载约 120 MB，属带宽受限，同样的驻留手段实测反慢 96.1 微秒——强制 F3 单 CTA 为 115.6 微秒，其中 K 循环重排本身约 110 微秒，驻留仅贡献约 20 微秒。串行扫描同理：将扫描工作按 K 维拆到多 warp 或多 CTA（含 sm_120 线程块簇；实测簇属性本身零开销，跨 CTA 数据交换成本 57 微秒），三个数据点均在 0.88–0.99 之间——每 warp 工作量不变时指令级并行已饱和，剩余延迟由数据搬运与同步构成。

六、编译期调度：Z3 与校准成本模型

XLA 在 TPU 上的哲学是调度决策全部在编译期做出、编译产物不携带运行时搜索；把它移植到原生 MLIR、消费级 Blackwell、单算子族上，必须回答两问：凭什么排序（成本模型的可靠性），凭什么相信生成的代码（覆盖与验证，第八、十节）。

1. 求解器：约束即代码

决策变量是十维旋钮向量（固定声明序）。约束直接写成 Z3 表达式，与文档一一对应。资源容量约束（共享内存字节模型）：

pair = (bt * bk + bk * bv) * esz                    # tile 对字节
smem_mma = z3.If(ws,
                 pair + z3.If(tma, 64*64*esz, 0) + 8,  # ws：单缓冲+TMA staging+mbarrier
                 pair * z3.If(pipe > 0, pipe, 1) * n_gemm)  # 流水=多缓冲版本数
smem_nomma = (bt * K + K * bv + bt * bv) * esz        # nomma：整 K 维驻留
smem = z3.If(mma, smem_nomma)
s.add(smem <= cap)                                   # strict 100,352B / generator 220KB
［tools/gdn_solver.py:153-161；寄存器约束在 149-152 行：threads*255 ≤ 65536］

结构合法性约束（mma fragment 对齐与流水深度上限）：

s.add(z3.Implies(mma, bt % 16 == 0))                     # M 对齐
s.add(z3.Implies(z3.And(mma, dtype == 0), bk % 16 == 0)) # bf16 K 对齐
s.add(z3.Implies(z3.And(mma, dtype == 1), bk % 4 == 0))  # tf32 K 对齐
s.add(z3.Implies(mma, bv % (8 * warps) == 0))            # N 对齐
s.add(z3.Implies(z3.And(mma, pipe > 0), K >= bk * pipe)) # 流水深度 ≤ K 趟数
［tools/gdn_solver.py:131-137］

说明：这些约束把第四节的 transform 工件拼写合法性（multibuffer factor 不能超过 K 循环趟数、fragment 尺寸对齐）从"编译器报错时才发现"提前到求解阶段。枚举驱动采用 SAT 加阻塞子句做全空间穷尽：

while len(out) < limit:
    r = s.check()
    if r != z3.sat:
        self.last_check = str(r)      # "unsat" = 空间干净穷尽
        break
    m = s.model()
    # ……读取/补全各旋钮值……
    out.append(vals)
    block = [self.v[k] != z3.IntVal(v) for k, v in vals.items()]
    s.add(z3.Or(block))               # 阻塞子句：下一解必须不同
［tools/gdn_solver.py:230-235、253-257，中间为模型读取与补全］

kkt 阶段 368 个合法配置 0.29 秒枚举完毕，o 阶段 465 个配置 0.33 秒；空间穷尽性由 last_check=unsat 背书，固定种子与固定声明序保证两次运行逐字节一致。两处防御性设计保证解的合法性与完备性。其一，求解器对被其余指派唯一决定的变量可能不返回取值（partial model 伪影），补全逻辑先钉住其余指派、再按域内顺序逐候选 SAT 探测：

pin = [self.v[k] == z3.IntVal(v) for k, v in fixed.items()]
s.add(z3.And(pin))
for cand in self.domains[key]:
    s.push()
    s.add(self.v[key] == z3.IntVal(cand))
    ok = s.check() == z3.sat          # 第一个保持可满足的候选即补全值
    s.pop()
    if ok: return cand
［tools/gdn_solver.py:192-204；外层先解除 rlimit，避免探测被限额中断误判］

其二，枚举若以 unknown（资源上限中断）而非 unsat 结束，绝不能把截断空间当完整解输出——先无上限重试一次，仍失败即抛异常拒绝产出（tools/gdn_solver.py:360-373，异常文本明写 refusing to emit a truncated space）。残缺空间里的排序没有意义。

2. 成本模型：零逐配置拟合

成本模型不含任何对实测配置时间调参的自由常数：每个速率来自一次性标定或为文档化的 sm_120 结构常数；模型每项为"计数除以速率"。张量核主项：

if feats["mma"]:
    peak = calib.m["mma_bf16_peak_flops"] if esz == 2 else calib.mma_tf32_peak
    terms["T_mma"] = feats["mma_flops_per_cta"] / (peak / calib.sm_count)
else:  # 标量环：每 MAC 按发射/延迟受限建模，速率=SASS 推导的 10 cyc/MAC
    macs = feats["ffma_flops_per_cta"] / 2.0
    terms["T_ffma"] = macs * NOMMA_CYC_PER_MAC / calib.m["clock_hz"]
［tools/gdn_costmodel.py:689-700；分子为封闭公式 2·bt·bv·K·n_gemm，已对静态 PTX 计数自检］

计数端甚至可以直接取编译期资源报告——溢出项的计数来自 ptxas 的 spill 报告（缺失时回退 SASS 的 LDL+STL 静态计数，并除掉流水 peel 造成的函数体复制换算成动态次数），速率端用标定表的 L2 带宽按 SM 均摊：

spills = mm.get("spill_stores", 0) + mm.get("spill_loads", 0)
dyn    = spills * (trips + pipe) / (1 + pipe)     # 静态计数 → 动态次数
terms["T_spill"] = dyn * threads * 4.0 / (calib.per_sm("l2_bw") * SPILL_BW_FRAC)
［tools/gdn_costmodel.py:616-625、711-712，节选］

一次性标定表由六个微基准 kernel 构成，分钟级、显存低于 2GB，属单次沉没成本：

表 5：sm_120 一次性标定表

| 量 | 标定值 |
|---|---|
| mma bf16 峰值 | 115.5 TFLOP/s（tf32 按 2:1 折算 57.8） |
| 共享内存读带宽 | 113.1 TB/s |
| L2 / DRAM 带宽 | 2144.3 / 824.6 GB/s（空闲窗口 851.4） |
| SFU ex2 吞吐 | 3.23 Telem/s |
| bar.sync | 7.707 ns |
| cp.async | 1388 GB/s |
| kernel 发射基准 | 6.073 µs |

分子的指令计数在编译期精确：18 个配置的静态 PTX 自检中，mma.sync、ld.shared、bar.sync 计数 18/18 精确，cp.async 偏差不超过 1.6%；误差只剩分母。模型输出用于排序，绝对值有已记录的偏差（mma 路径时间低估 3–5 倍，均匀缩放，秩不变）。排序质量：与 18 个实测配置的 Spearman 秩相关 0.8962（门槛 0.7；kkt 阶段 0.936，o 阶段 0.829，top-5 配置重叠 0.8）；全部结构常数扰动正负两倍，秩相关仍在 0.882–0.902 之间。

3. 搜索成本对比

图 6（数据）：新形状调度/调优时间对比，对数刻度——本方冷启动发射 27 µs；本方新形状调度 0.8 s（纯 CPU）；fla 缓存摊销后 0.65–5.67 s；fla 首次搜索 39.8 s（48 个 Triton 配置逐个上机）。

本方每新形状 0.8 秒纯 CPU，对 fla 首搜 39.8 秒设备时间约 50 倍；fla 磁盘缓存摊销后 7.7–67 倍，两个口径并列报告。部署形态的差别更大：本方产出 AOT cubin，新进程冷启动约 27 微秒；fla 每个形状在新进程内重新 autotune，缓存命中亦需 0.65–5.67 秒。

七、PTX 后处理：四个变换的前后对照

856 行文本改写器 gdn_smem_fix.py 承载四个变换。以下每个变换给出真实 PTX 的前后对照（证据树保存了六十余对 kernel.ptx / kernel_smemfix.ptx 成对文件）与工具核心代码。

1. 静态共享内存修复（device malloc → .shared 符号）

stock 管线把 workgroup 的 memref.alloc 下译为设备端堆分配，携带共享内存 tile 的 cubin 在真机上全部非法地址错误，通过率为零。变换前的真实 PTX（malloc 调用序列，尺寸 8192）：

{ // callseq 0, 0
.param .b64 param0;
.param .b64 retval0;
st.param::func.b64 [param0], 8192;
call.uni (retval0), malloc, (param0);
ld.param::func.b64 %rd220, [retval0];
} // callseq 0
...
cvta.to.shared.u64       %rd8,  %rd220;      // 普通往返
cvta.to.shared::cluster.u64 %rd196, %rd221;  // TMA 的 cluster 形态
［evidence/G4/gen_verify/kkt_bf16_mma_tma_ws/kernel.ptx:98-104、389-390］

变换后：malloc 序列整体删除，入口前插入静态声明，两种 cvta 都改写为对符号的直接取址：

.shared .align 128 .b8 __gdn_smem0[8192];   // 声明插在 .visible .entry 前
.shared .align 128 .b8 __gdn_smem1[8192];
...
mov.u64 %rd8,   __gdn_smem0;                  // cvta.to.shared → mov 取址
mov.u64 %rd196, __gdn_smem1;                  // ::cluster 形态同样改写
［evidence/G4/gen_verify/kkt_bf16_mma_tma_ws/kernel_smemfix.ptx:17-19、371-372］

说明：修复复刻 MLIR 自身为静态共享内存已经使用的写法。两个硬件细节决定成败：声明必须 .align 128——TMA 装载在 sm_120a 上即使不启用 swizzle 也要求 128B 对齐；cvta.to.shared::cluster 变体必须一并改写，否则读已删除 malloc 定义的寄存器、设备直接陷入错误态。工具按 callseq 边界与 cvta 正则匹配（tools/gdn_smem_fix.py:73-80），替换模板即上面的 mov 加 .shared 声明（195、204 行）。修复后通过率从零到全量。

2. 生存期别名（独占 malloc → 竞技场基址加偏移）

生成器为每个 kernel 产出一份布局表，声明哪些 tile 与哪个基 tile 共享字节；工具按 malloc 的唯一尺寸识别 tile 并重写地址。布局表原文（前向 F1：vt 与 k2 两个 tile 分别折叠进 Abf 与 Tt 两个竞技场）：

"gdn_fwd_f1_kernel": {
 "identify": { "9216": "Abf", "9504": "Tt", "8704": "vt", "8976": "k2" },
 "arenas":   { "Abf": 9216, "Tt": 9504 },
 "alias":    { "vt": ["Abf", 0], "k2": ["Tt", 0] }
}
［evidence/G5/gen_fwd/B4T2048_mono/smem_alias.json］

变换前：vt 与 k2 各自 malloc 8704/8976 字节并独占声明。变换后的真实 PTX（两个 tile 的地址被重定向到竞技场符号，独占声明消失——整份文件不再有 8704/8976 的声明）：

mov.u64 %rd17, __gdn_smem_gdn_fwd_f1_kernel_4;   // vt → Abf 竞技场
add.u64 %rd17, %rd17, 0;
mov.u64 %rd18, __gdn_smem_gdn_fwd_f1_kernel_5;   // k2 → Tt 竞技场
add.u64 %rd18, %rd18, 0;
［evidence/G5/gen_fwd/B4T2048_mono/kernel.ptx:92-101、248-249 → kernel_smemfix.ptx:189-192］

识别核心：malloc 结果寄存器按尺寸反查 tile 名，同一尺寸出现多次即整体弃用别名（安全回退）：

for sz_s, tile in alias.get("identify", {}).items():
    regs = size_regs.get(int(sz_s), [])
    if len(regs) != 1:  ok = False      # 尺寸不唯一 → 别名禁用
    reg_of_tile[regs[0]] = tile
［tools/gdn_smem_fix.py:114-137（识别）+ 177-184（替换模板 mov+add）］

说明：SSA 合法性要求每 tile 一次分配，"生存期不重叠的 tile 共享字节"这一物理事实只能在 PTX 层施加；设计立场是别名属于纯增益变换，失败模式必须是"不省"，绝不允许"算错"。效果即图 5 的三个池压缩与双 CTA 驻留。

3. 寄存器上限注入（.maxnreg）

ptxas-blackwell 忽略 --maxrregcount 旗标、遵循 PTX 源内 .maxnreg 指令；且旗标是逐文件的，对装着三个 kernel 的单一 cubin 无差别限制没有意义，须逐入口注入。真实前后对照（dqkwg kernel，上限 128）：

// 变换前                                    // 变换后
.visible .entry gdn_dqkwg_kernel_kernel()    .visible .entry gdn_dqkwg_kernel_kernel()
.maxntid 256, 1, 1                           .maxntid 256, 1, 1
                                                  .maxnreg 128
                                              {
［evidence/G5/scratch/r19b_final_T8192/dqkwg/kernel.ptx:37-38 → kernel_smemfix.ptx:49-52］

工具按环境变量表 GDN_MAXREG_K='f3:168,f2:128' 对入口名做子串匹配、未匹配显式告警（tools/gdn_smem_fix.py:683-712，逐行读入、entry 行后追加 .maxnreg N）。工程细节：ptxas 报告文件只对应第一次编译，最终 cubin 出自改写后的第二次编译，两次的寄存器分配可能不同，资源对账以第二次为准。注入后占用从 15.02% 升到 29.17%、每 SM warp 数从 8.00 升到 14.23。

4. WST 宽存融合（2×16 位 → 1×32 位）

smem 到 global 的逐 cell 16 位搬运在 WST 下配对融合。真实前后对照（同一循环、同一基址寄存器，四条指令变两条；注意 +2 偏移的 store 被 LLVM 调度在前，工具按地址配对而非文本顺序）：

// 变换前（kernel.ptx:3328-3335）             // 变换后（kernel_smemfix.ptx:3270-3272）
$L__BB0_94:                                  $L__BB0_94:
ld.shared.b16 %r1653, [%rd543];              ld.shared.b32 %r9000, [%rd543];
ld.shared.b16 %r1654, [%rd543+2];            st.global.b32 [%rd544], %r9000;
st.global.b16 [%rd544+2], %r1654;
st.global.b16 [%rd544],   %r1653;
［evidence/G5/gen_fwd/B4T2048_mono/kernel.ptx → kernel_smemfix.ptx，本目录共融合 7 处］

说明：32 位寄存器恰好装下两个相邻 16 位 cell，字节精确等价由构造保证。新寄存器从 %r9000 起编号（该版本 ptxas 拒绝区间声明，须写成显式逗号清单）；配对条件是共享地址相邻且全局偏移差恰为 2（tools/gdn_smem_fix.py:476-479 正则、528 行偏移校验、544-547 替换模板）。WST 负责融合，W32P（第五节变换八）负责让配对合法——两层各管一半，缺一则该存储通路的收益不存在。

八、覆盖证明与逃生舱预算

1. 参考 PTX 的纯 CPU 收割与两个极点

两家的 PTX 与 SASS 全部离线获取：fla 经 Triton AOT 以 GPUTarget("cuda",120,32) 目标编译（自带 ptxas -v 资源日志）；FlashQLA 经 TileLang 锁定 target 后导出 kernel 源；SASS 经 nvdisasm 反汇编。两家构成调度谱系的两个极点：fla 寄存器 40–255 顶格、大量溢出、多级流水、无 warp 专化、循环全展开；FlashQLA 寄存器 88–128 由 setmaxnreg 固定、TMA 加 mbarrier 加 elect.sync 的单 kernel 融合、真循环。FlashKDA 作为第三风格点由 L2 旋钮吸收（网格映射旋钮显式包含 token 并行的 WY 预段）。

2. T1：指令族覆盖

图 7（数据）：T1 指令族三态判定堆叠条——全部 33 族：8 YES / 19 PARTIAL / 6 NO；feature 层 15 族：8 YES / 6 PARTIAL / 1 NO。YES 指已发射且变体齐备；PARTIAL 指已发射但缺个别变体；NO 指从未发射。全部 PARTIAL 的残差属于指令变体（操作码拼写、向量宽度、旗标），能力层无缺口；唯一的 feature 级 NO（stmatrix）在实测 40% 回归后按测量结论关闭。

表 6：mma 指令签名层对比（条数为收割/生成 PTX 中的实际计数）

| mma 指令签名 | fla | FlashQLA | 本方 |
|---|---|---|---|
| bf16 m16n8k16 | 2708 条 | 2496 条 | 552 条，YES |
| tf32 m16n8k8 | 576 条 | 0 条 | YES（经唯一逃生舱，fla 精确拼写） |

本方是三者中唯一同时发射两个家族 mma 签名的生成方。T1 的直接读法：fla 能发出的指令族，本方旋钮空间都能发出；FlashQLA 的 warp 专化指令族（setmaxnreg、elect、mbarrier、TMA）同样在格内，其中 setmaxnreg 的实机收益（溢出 22,844B 降到 48B）与 ldmatrix 家族（读侧 8 路冲突的消除）都在最终配置中启用。

3. T2：结构匹配

四张结构卡（fla 的两个 kernel 加 FlashQLA 的两个 fused kernel），每张按指令族 Jaccard 相似度、mma 签名匹配与指令总数接近度从本方 20 个配置中选最近邻，再对寄存器、共享内存、溢出三项资源以正负 30% 包络判分。结论：fla 极点（多级流水加 cp.async 加 255 寄存器顶格）与 FlashQLA 极点（warp 专化加 TMA 加单 kernel 融合）的每个结构要素都在旋钮空间内可生成。T2 度量风格的可达性，规模差异由资源包络单独判分：FlashQLA fused_bwd 单 kernel 1600 条 mma，与本方单 chunk kernel 相差 50 倍，该卡资源判定为 OUT，结构要素全部在格内。

4. 唯一的逃生舱：gdn.mma.tf32k8

必要性：NVVM 表列的 mma.sync m16n8k8 只有 f16/bf16/i8；nvgpu.mma.sync 的 verifier 强制 8×8×128b 基本片，tf32 封顶 K=4——本方 stock 路径的 tf32 只能发出 m16n8k4 形态（真实 PTX 存证：evidence/G4/gen_verify/o_f32_mma_p2_.../kernel.ptx:6229，mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32），而反向的 fp32 GEMM 需要 fla 收割到的 m16n8k8 精确拼写。表内不可达属于 NVVM TableGen 的限制，与调度能力无关。工具链缺陷：经 inline asm 发射时，普通输出约束下 ptxas 将 D 四元组分配到同一条 mma 的 A 四元组上（读写重叠损坏操作数；整型探针 118/128 单元确定性失败），early-clobber 约束禁止重叠后消失——该缺陷对任何在 mma 上使用 inline asm 的实现成立。硬件探针：A-fragment 寄存器序为 row-half 外层，与 bf16 的 k-八元组配对直觉相反，由 16×8×8 整型探针在真机确认（tf32 精确表示整型，D = A·B + C 逐位成立；N(0,1) 输入偏差 4.2e-3 证实真 tf32 语义）。克制：tf32 按 24 个 fp32 GEMM 位点逐点数值验证，16 个放行（wy.u 位点误差 1.06e-2 超容差被否决；dhu 位点性能负收益整体回退），放行位点贡献所在阶段端到端 14.16 降到 11.06 毫秒。预算 5 个、支出 1 个，支出的那一个带完整的必要性论证、缺陷定位与硬件探针。

九、反向的精度上界

反向小形状 1.72 倍、前向全形状超过 fla、冷启动全形状占优；u ≥ 1 的大形状为 0.41–0.49 倍。对该结果的结论是：

0.41–0.49 由误差容差合同决定，与实现质量无关。在 fla 的梯度容差（dq/dk/dv ≤ 8e-3、dβ/dg ≤ 2e-2）下，u ≥ 1 形状的 1.1 倍目标结构性不可达——理想执行的上限为 0.85–0.92 倍。

这里的"理想执行"按屋顶线模型（roofline，以标定带宽为唯一约束的执行时间下界）取 100% 利用率。该结论先于优化得出：CPU 字节定价器在 GPU 侧运行之前预测比值随 u 单调递减（边际每单位 u 2.67 倍、u ≥ 1 的所有形状都低于 1、小形状 2.17 倍是唯一超过基线的形状；终态优化将其深化到 1.72 倍），随后的阶梯实测精确验证。形状负载对性能比值的走向可以在编译期预测。

图 8（数据）：u=8 的流量下界算术横条（毫秒）——fla Triton（分母）5.62 → 1.00×；本方理想执行（100% 屋顶线）6.1–6.6 → 上限 0.85–0.92×；本方实测 12.15 → 0.462×。实测距下界尚有约 2 倍执行余量，执行余量全部兑现亦只能到 0.92 倍。

四类独立分析各自排除一条逃逸方向，松动任何一条其余仍成立。

1. 误差预算排除"把中间量降到 bf16"。反向链有 16 个 fp32 中间量候选，逐个做降精度测试：10 个可降、6 个被否决（合计 470 KiB 每兆行）。否决的原因值得展开：这些中间量存在大数相消——某量的真实量级为 O(0.1)，由若干 O(10) 的项相减得到；bf16 只有 7 位尾数，把 O(10) 的项各自舍入就引入 4.6e-2 至 6.9e-2 的绝对误差，恰好砸在被相消后的小值上，超出容差 2.4 至 11 倍。split-bf16 能提升乘积精度但节省不了存储流量（hi 与 lo 都要存，第五节变换九的两遍发射正对应这笔账）；dq、dk、dv 是被评分的输出，合同锁定 fp32。

2. 流量下界算术排除"把执行做到理想"。优化终态的字节流量下界约 0.75 毫秒每单位 u。屋顶线理想执行下 u=8 需 6.1–6.6 毫秒，对 fla 的 5.62 毫秒为 0.85–0.92 倍；即使整段删除三个最大 kernel（非真实选项）亦只到 0.88 倍。

3. 融合定价排除"删除 kernel 间数据搬运"。最后两个大 kernel（dqkwg 与 wyb）的融合被三重否定：可行最优共享内存池 97,024B，仅完美融合需求的 82%（第七节的别名机制在这里也无能为力——受限于全物化屏障下真正同时活跃的 tile）；单 CTA 占用反转实测 1.155–1.26 倍；u=8 净时间差最好、中位、最差为 +0.62、−0.11、−0.83 毫秒。即使不可实现的完美融合亦被字节算术限制在约 0.54 倍。结论 NO-GO。

4. 容差扫描排除"放宽容差合同"。dk 容差从 8e-3 放宽至 2e-2（2.5 倍）分四档测试，全部档位下没有任何被否决的中间量因此获得降精度资格（被否决符号的误差增量比最宽档还紧 6 倍）；首个解锁需放宽 8.2 倍；达到 1.1 倍需 dk 放宽约 15 倍且 dβ 放宽 2.4 倍两条轴同时进行，且全部解锁亦只将上界移至 1.08–1.15。容差的作用是阶跃的：达到特定档位之前，放宽不解锁任何符号。

四条合起来将 u ≥ 1 的 1.1 倍目标在合同层面封死；可改的输入（梯度容差、门控衰减设计、带宽标定）没有一项在编译器手里。全部核算可离线复算（仓库内三个定价脚本 tools/rc1_price.py、tools/r4_price.py、tools/tol_sweep.py），后两类分析的结论在不修改任何内核代码的情况下得出。

十、验证与测量

1. 四层验证与容差合同

表 7：分层验证

| 层 | 被测对象 | 尺子 | 结果 |
|---|---|---|---|
| 语义 | MLIR L1（CPU 执行） | JAX fp64 规约 ⇄ fla naive | 前向约 6e-7 / 反向最差 6.4e-6，三方闭合 |
| 配置 | 20 个旋钮配置 | ptxas -v 资源包络 + SASS 指令组合 | 编译期静态复核 |
| 阶段 | 逐配置 GPU kernel | fp64 numpy 参考（逐位相同的 bf16 输入） | 20/20，双随机种子 |
| 全算子 | 端到端 kernel 链 | JAX fp64 / fp64 参考 + 结构不变量 | 全部元素在容差内，每次变更后复验 |

fp64 参考实现的两条纪律：与被测 kernel 接收逐位相同的输入（主机端精确 bf16 量化），且算法同构——差值只度量 kernel 数学与内部截断。两个结构性检查各自抓到过真实缺陷：非零元计数精确一致（曾抓到 mma 路径只写 1168/2080 个下三角元）；实测 mma 最大偏差 9.77e-4 恰等于管线的量化预测 9.765625e-4，证明残差是输出量化而非计算误差。

容差定在"fla 自身对 fp64 参考的距离"量级：o 与 ht 不超过 5e-3，dq、dk、dv、dh0 不超过 8e-3，dβ 与 dg 不超过 2e-2。fla 标定（evidence/G5/refs/tolerances.json）显示 fla 的残差系统性落在其输出 dtype 的 bf16 舍入下界的约 2 倍（1.36–4.33×，中位 1.9×），即 fla 的残差基本就是 bf16 舍入，该组容差覆盖"与 fla 同级准确"的可达域。本方裕度：前向约 7 倍门限余量（o 6.486e-4 / ht 7.493e-4），反向最紧的 dk 1.6–15.6 倍，dv 736–848 倍，dh0 达 3.7e4–5.7e4 倍。

2. 为什么绝对时延不可信

采用交错终判的依据是两条实测事实：分离进程的绝对时延在零负载下仍随时钟状态漂移（同一 cubin 相隔数分钟的两个进程可相差 47%）；同进程内模块加载顺序改变全局内存布局，使访存最重 kernel 出现约 6% 的系统性偏差，加载顺序互换偏差精确反号。因此加速比的分母一律取同窗交错测得的基线，绝对毫秒数只作工程参考。

3. 确定性

编译端：Z3 固定声明序、种子 7、资源上限 2×10⁷，两次完整编译逐字节一致；生成器同种子两次生成的 MLIR 哈希相同；全部旋钮关断时产物与默认配置逐字节等价（27/27 payload、18/18 cubin）。排名因此是可存档、可 diff、CI 可断言的证据文件。

运行端：同一 cubin、同一输入、同一发射几何下输出逐位一致，跨进程（全新 CUDA 上下文与模块加载）验证 5 次全过方为通过。该协议抓到过五个真实竞态，全部在数值容差之内、多数仅 1–3 ulp——数值验证对这类缺陷无感，逐位一致性是唯一能抓到它们的尺子；其中两个曾被更高的驻留占用掩盖。逐位可复现的代价约 0.3%（T2048）。

十一、工具链发现

以下发现对 MLIR 与 NVIDIA 工具链具有独立价值，每条均有探针或 SASS 级证据：

1. workgroup 的 memref.alloc 被 stock 管线下译为设备端 malloc，真机全部非法地址错误；修复方案在 gdn_smem_fix.py（第七节第一小节）。
2. transform 的 create_async_groups 在每组后立即插入 wait_group 0，异步复制结构性无法重叠（第四节第三小节）。
3. block-uniform 的 staging 被每线程重复执行（256 次冗余复制）。
4. ptxas-blackwell 忽略 --maxrregcount、遵循 PTX 源内 .maxnreg；ptxas -v 报告只对应第一次编译，资源对账以第二次为准（第七节第三小节）。
5. cp.async 到 ldmatrix 的静默数据损坏：经 ld.shared 读回正确、经 ldmatrix 读回错误，仅影响与构建调度相关的行块子集；st.shared 安全。
6. ptxas 跨入口静默合并重名 .shared 符号，产生越界写。
7. 模块加载顺序效应：加载顺序决定 .global 布局，访存最重 kernel 出现约 6% 系统偏差，顺序互换偏差精确反号。
8. 分离进程的绝对时延在零负载下仍随时钟状态漂移（实测同一 cubin 可相差 47%）。
9. sm_120 线程块簇大于 1 可用（设备验证，推翻只有 1×1×1 的通行假设）；簇属性本身零开销，跨 CTA 数据交换的 57 微秒成本决定其收益。
10. cp.async 的 wait_group 为每线程语义，跨 warp 可见性需要显式栅栏。
11. tf32 m16n8k8 的 inline-asm 发射：early-clobber 为正确性必要条件；A-fragment 寄存器序为 row-half-outer，与 bf16 直觉相反（第八节第四小节）。

十二、仓库结构与复现

dialect/          gdn 方言（L0 op + GdnConfigAttr + 12 个直通 op + passes + gdn-opt）
tools/            全部生成器与 harness（gdn_l1_gen / gdn_ref_jax / gdn_l2_gen / gdn_solver /
                  gdn_costmodel / gdn_full_gen / gdn_bwd_gen / gdn_smem_fix / gdn_verify /
                  gdn_final_adjudicate / rc1_price / r4_price / tol_sweep）
scripts/          幂等驱动（build_llvm / build_dialect / run_calib / repro / 空闲窗口轮询）
evidence/G0-G7    每个 gate 的报告 + 原始数据（PTX/SASS/计时 JSON/交叉验证记录/覆盖矩阵）
paper/zh/         深挖论文 11 章 + 本文

复现入口：make repro-g1（语义三方闭合，纯 CPU，约 15 分钟）、make repro-g2（旋钮矩阵加覆盖分析，纯 CPU，约 25 分钟）、make repro-g3（标定加排序门槛，两步 GPU，约 45 分钟）、make repro-g4（逐配置正确性，GPU，约 1 小时）、make repro-g5（全算子验证加交错终判，GPU，约 1 至 1.5 小时）、make repro-g6（搜索成本，约 10 分钟）、make repro-all（顺序执行，首败即停）。环境需求：WSL2、conda 环境 gdn、CUDA 13.x、计算能力 12 以上 GPU；语义与求解部分完全纯 CPU。

十三、相关工作

表 8：相关工作定位

| 系统 | 栈 | 调度方式 | 与本编译器的关系 |
|---|---|---|---|
| fla | Triton | 运行时 autotune（首搜 39.8 s/形状） | 性能基线；T1/T2 覆盖合同一方 |
| FlashQLA | TileLang | 手写 warp 专化调度 | 覆盖合同另一方；sm_120 前向现役最快（本方落后 1.4–1.5 倍） |
| TileLang | DSL | z3 调度器（仅 Ascend 后端） | 编译期求解机制的来源；其 CUDA 侧空白即移植点 |
| XLA/TPU | 编译器 | 编译期确定性调度 | 哲学对标的四支柱 |
| CAKE | agent 写 IR | 自改进闭环 | 不同路线：本编译器以约束求解替代 agent 迭代 |
| FlexLA / Hexcute / TileLoom | 各系 DSL | 调度不由 IR 承载 | 相关工作空白：无"调度即 IR 的 stock-MLIR 渐进下降 + SMT 编译期调度 + PTX 覆盖验证" |

血统交代：Z3 编译期求解的机制来自 TileLang 的 Ascend 实践，本项目将其上移一层——决策变量从指令时槽改为旋钮向量，此移植在 CUDA 侧无先例；验证塔的组织借鉴 XLA 的分层可验证哲学；2026-10 LLVM 主干已原生支持本编译器以薄封装过渡的大部分直通 op 词汇。

十四、论文与证据

深挖论文（11 章，中文，676 KB Markdown 加 818 KB KaTeX HTML）逐模式展开全部机制，每项含动机、源码、IR 前后与效果，位于 paper/zh/gdn-mlir-deep-dive.*。证据索引 EVIDENCE_INDEX.md（gate 级）、CAMPAIGN_CLOSEOUT.md（精度上界收官）与 RECOVERY.md（恢复协议）。参考实现 fla 与 FlashQLA 均为 MIT 许可，在本项目中仅作对照基准使用，其 PTX 收割物不含其源码。
