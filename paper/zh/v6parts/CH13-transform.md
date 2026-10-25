## 十三、Transform Dialect：用 IR 编写调度

一份 `linalg.matmul` 或 `scf.for` 描述计算结构，切块大小、缓冲版本数和流水深度决定这些结构怎样执行。Transform Dialect（变换方言）把后一类决定也写成 MLIR：变换脚本持有指向计算 IR 的句柄，按顺序调用改写动作。调度因此成为可解析、可比较、可随配置存档的文本，同时仍由上游变换实现执行。

### 1. payload、transform 与句柄

被改写的计算称为载荷 IR（payload IR），本目录中是 `payload.mlir`；描述改写过程的变换 IR（transform IR）位于 `transform.mlir`。两者可以在同一编译输入中出现，却承担不同角色。payload 含 `func.func`、`gpu.launch`、`memref.alloc` 等计算操作；transform 含匹配、切块、流水等控制操作。普通的 SSA 值表示数据，变换句柄（handle）表示一组 payload 操作或值与该组对象之间的映射。

`!transform.any_op` 是任意操作句柄，一个句柄可以关联多个操作；`!transform.op<"memref.alloc">` 进一步限定对象类型。`transform.structured.match` 在指定作用域内按操作名收集对象并产生新句柄。匹配不到对象通常得到空句柄，所以后续还应检查实际改写结果，不能把脚本通过解析等同于调度已经生效。

模块属性 `transform.with_named_sequence` 声明命名序列，`transform.named_sequence @__transform_main` 给出解释器入口。入口参数的 `transform.readonly` 约束该根句柄的消费方式；它允许序列通过子句柄修改 payload，并不把整次执行变成只读分析。根句柄 `%arg0` 因而可以反复用于 `match`。

### 2. 十五行工件的动作顺序

下面的真实工件对应 kkt、bf16、MMA、两级流水、64×64×32 tile 和四 warp 配置，全文只有十五行。

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %arg0
      : (!transform.any_op) -> !transform.op<"memref.alloc">
    transform.memref.multibuffer %allocs <factor = 2, skip_analysis>
      : (!transform.op<"memref.alloc">) -> !transform.any_op
    %func = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.create_async_groups %func bypass_l1 : (!transform.any_op) -> (!transform.any_op)
    %loops = transform.structured.match ops{["scf.for"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.pipeline_shared_memory_copies failures(propagate) %loops depth = 2 peel_epilogue : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```
[evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/transform.mlir:1-15]

第 3–6 行先找到两个共享内存分配，再执行多缓冲（multibuffer）。该变换把分配扩大为原来的两倍并修正使用者，使相邻循环迭代使用不同版本，消除临时缓冲造成的迭代间依赖。`skip_analysis` 跳过跨迭代无依赖证明，正确性责任随之交给工件作者；这份 payload 使用 `vector.transfer_write`，当前分析所接受的覆盖形式无法证明它，项目记录也明确要求这个选项 [evidence/G2/gen_report.md:121-123]。

以这份配置手算缓冲账：每个 64×32 bf16 tile 占 4096 B，A、B 各一份时为 8192 B；factor=2 后，两个分配合计 16384 B。这个乘法是审查调度时的静态预期，解释后的 IR 应呈现扩容后的分配和按迭代切换的使用者。它还不能代替机器资源报告，尤其该 G2 归档的 `smem_static` 没有可解析数值，需结合后续 PTX 修复链和实际装载结果判断。

第 7–9 行匹配函数并建立异步复制组。`create_async_groups` 把函数内可识别的全局到共享内存 `vector.transfer` 改成异步复制，在连续复制后建立组并插入等待。`bypass_l1` 对满足 16 字节条件的传输设置绕过 L1 的属性；这里的 `vector<8xbf16>` 恰好是 16 字节，后续可看到 `cp.async.cg`。省略该选项时走 `cp.async.ca`。异步指令的存在并不自动形成重叠，等待位置与循环流水共同决定重叠窗口。

第 10–12 行重新匹配 `scf.for`，再做共享内存复制的软件流水。`depth = 2` 与双缓冲配套，让下一阶段的装载可以和当前阶段的计算交错；`peel_epilogue` 将排空阶段剥成循环后的尾声，避免依赖循环体内的谓词化支持。上游把足够缓冲和深度内无依赖列为调用者责任，这正解释了工件中 multibuffer 必须先于 pipeliner。

【通用补充】两级流水可以用三次迭代理解：序言先装载第 0 块；稳态在第 i 块计算期间装载第 i+1 块；最后一块没有后继装载，只剩计算与等待。`peel_epilogue` 把最后这一步放到主循环外。depth 描述同时在飞的阶段数，既不是线程数，也不等同于循环展开次数。若只有单缓冲，提前装载会覆盖仍在使用的数据，所以缓冲版本和流水深度必须联合审查。

这里还体现了句柄消费（handle consumption）。修改 payload 的 transform op 通常消费输入句柄；被消费句柄及其指向子树的关联句柄可能失效，继续使用类似使用悬空指针。`create_async_groups` 明确消费 `%func` 并返回新函数句柄，函数内部旧循环句柄也可能失效。工件没有沿用任何旧循环句柄，而是在第 10 行重新 `match`，从当前 payload 建立新映射 [../llvm-mlir/mlir/docs/Dialects/Transform.md:272-324；../llvm-mlir/mlir/include/mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.td:43-58]。`%arg0` 只读保留，使这次重匹配有稳定的作用域入口。若下一个动作仍以整个函数为目标，可以使用 `create_async_groups` 返回的新函数句柄；这里要寻找另一类操作，所以从根作用域重新匹配更直接。句柄是否可继续使用取决于操作声明的消费效果和失效规则，不能仅凭 SSA 名称仍然存在作判断。

{{FIG12}}

### 3. 切块不等于 GPU 线程分发

no-MMA 路径使用另一个动作：以 `scf.for` 为载体做结构化切块。

```mlir
%mms = transform.structured.match ops{["linalg.matmul"]} in %arg0
  : (!transform.any_op) -> !transform.any_op
%tiled, %loops:2 = transform.structured.tile_using_for %mms tile_sizes [16, 16]
  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
```
[evidence/G4/gen_verify/kkt_bf16_nomma_p0_t64x64x32_w4/transform.mlir:3-5]

`tile_using_for` 把二维迭代域分成 16×16 块，返回切块后的操作和两个外层循环句柄。三个结果可以继续交给向量化、循环变换或其他选择动作。它改变迭代结构，没有在这些结果上指定 `gpu.thread_id`、warp 或 lane 的映射。

线程几何在 payload 中独立给出：

```mlir
gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
           threads(%tx, %ty, %tz) in (%bxx = %c128, %byy = %c1, %bzz = %c1) {
```
[evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/payload.mlir:69-70]

这两行只规定网格与 CTA 的线程数。MMA 路径随后由生成器显式计算 warp、lane 和 fragment 下标 [evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/payload.mlir:75-85]。no-MMA 路径若没有进一步分发，`-convert-linalg-to-loops` 留下的仍是标量循环；G3 的 SASS 审计记录了未分发的三层循环及其高开销 [evidence/G3/solver_report.md:108-118]。因此，检查 tiling 是否生效要看新循环和 tiled op，检查 GPU 分发则要另看线程下标、映射操作及最终 SASS。

### 4. 解释器、调度擦除与 identity 绕行

运行脚本把 payload 与 transform 合并，`-transform-interpreter` 从 `@__transform_main` 开始顺序执行动作；之后的 `-test-transform-dialect-erase-schedule` 删除变换序列，使后续 NVVM 管线只看到改写后的 payload。

```bash
cat "$TMP/payload.gdn.mlir" transform.mlir > "$TMP/combined.mlir"
"$MLIR_OPT" "$TMP/combined.mlir" -transform-interpreter -o "$TMP/sched.mlir"
# ……中间省略 identity 分支与 linalg 兜底……
"$MLIR_OPT" "$TMP/sched2.mlir" -test-transform-dialect-erase-schedule \
  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120a cubin-features=+ptx87 cubin-format=assembly" \
  -o "$TMP/asmobj.mlir"
```
[scripts/run_l2.sh:53-55、61-72，两段合排]

可检查的产出分三层：`sched.mlir` 应出现多缓冲、异步组和流水循环；`asmobj.mlir` 应只含计算与 GPU 对象，不再含命名序列；最终 `meta.json` 可用指令计数复核。真实解释结果把分配从单个 64×32 tile 扩成了两个版本：

```mlir
%alloc = memref.alloc() : memref<2x64x32xbf16, #gpu.address_space<workgroup>>
%alloc_0 = memref.alloc() : memref<2x64x32xbf16, #gpu.address_space<workgroup>>
// 后续仍有 device_async_copy：tile 的数据通路也被保留下来
%186 = nvgpu.device_async_copy %arg0[%12, %182], %subview_3[%c0, %c0], 8 bypassL1 : ...
```

[paper/zh/v6examples/kkt.sched.mlir:214-215,267-268；真实 transform-interpreter 输出]

这两行直接验证 `factor=2` 已作用于 A/B 两个分配，而不是只改变了 transform 文本。`device_async_copy` 则验证异步复制动作进入了 payload。全文生成的 `kkt.sched.mlir` 还保留 K 循环与尾声结构，可用它继续做修改 depth、tile 或 match 目标的差异实验。该配置的 PTX 含 48 条 `mma.sync` 与 1542 条 `cp.async`，SASS 含 48 条 HMMA 与 1536 条 LDGSTS，说明调度确实到达机器产物 [evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/meta.json:26-29、51-62]。

identity schedule 只含 `transform.yield`，常用于 warp 专化等直接发射目标 op 的 payload。当前脚本检测不到 structured、nvgpu 或 memref 变换时，会跳过解释器并直接复制 payload [scripts/run_l2.sh:48-59]。这是针对当前版本的绕行：该解释器会把裸顶层函数再包进 `module {}`，额外嵌套导致后续管线在 workgroup memref 上转换失败。这个行为带工具链版本条件，不应提升为 Transform Dialect 的普遍语义；更换版本后应先用同一 identity 工件复测，再决定是否保留绕行。

### 5. propagate 与 suppress 的失败边界

kkt 工件的 `failures(propagate)` 表示：循环边界不是静态值、找不到可流水的共享内存装载等前提失败会向上传播，解释器以错误结束。这对当前模板很合适，因为静默保留原循环会让配置名中的 `p2` 与实际产物脱节。

生产前向模板采用过另一种策略：

```mlir
%loops = transform.structured.match ops{["scf.for"]} in %arg0
  : (!transform.any_op) -> !transform.any_op
transform.nvgpu.pipeline_shared_memory_copies failures(suppress) %loops depth = {depth} peel_epilogue : (!transform.any_op) -> !transform.any_op
```
[tools/gdn_full_gen.py:591-593]

`failures(suppress)` 把可恢复的前提失败压下，结果句柄继续关联原循环，后续序列可以执行；它不会把不可恢复错误或后续转换失败变成成功。当前树还有一个更窄的实测限制：非复制型 `scf.for` 即使在该模式下也可能使整个解释过程失败，所以生产生成器把普通 cell 循环写成 `scf.while`，只让目标 K 循环保持 `scf.for` [tools/gdn_full_gen.py:1129-1135]。选择失败模式后仍需核对哪些循环真正被流水，不能把 suppress 当作普遍兼容开关。

