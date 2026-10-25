## 二十、机器 Op 的合同：MMA、ldmatrix、异步复制、mbarrier 与 TMA

机器 op 同时受两类合同约束。MLIR verifier 检查类型、形状、属性与部分合法域，硬件还要求寄存器片段坐标、每 lane 地址、跨线程完成协议和内存对齐。一个 `vector<2x2xf32>` 可以类型正确而列映射错误，一个 mbarrier 可以操作数合法而相位永远无法完成。阅读生成器时，SSA 定义链只解决值从哪里来；还要继续追踪这个值在 32 条 lane 上分别代表什么。

### 1. MMA fragment 的类型与坐标必须一起读

fragment（片段）是分布式 tile 在单 lane 寄存器中的部分。对 m16n8k16，令 lane 为 L，groupID `g=L/4`，组内线号 `tig=L%4`。A、B 与累加器 D 虽同属一次 `nvgpu.mma.sync`，每 lane 持有的元素数和坐标并不相同：

```mlir
%g = arith.divsi %lane, %c4 : index
%tig = arith.remsi %lane, %c4 : index
%t2 = arith.muli %tig, %c2 : index
%rowA0 = arith.addi %g, %c0 : index
%rowAh0 = arith.addi %g8, %c0 : index
%ncolg0 = arith.addi %ncol0, %g : index
%ncolt0 = arith.addi %ncol0, %t2 : index
%ncolgp10 = arith.addi %ncol0, %t2p1 : index
// ……A/B fragment 装载……
%g0f0_0_0r = nvgpu.mma.sync(%g0f0_0_0a_b7, %g0f0_0_0b_b3, %ia0_0)
  mmaShape = [16, 8, 16] :
  (vector<4x2xbf16>, vector<2x2xbf16>, vector<2x2xf32>)
  -> vector<2x2xf32>
```

[evidence/G4/gen_verify/kkt_bf16_mma_p2_t64x64x32_w4/payload.mlir:151-153,290-301,1385，合并节选并缩排]

类型先给出寄存器账：A 的 `vector<4x2xbf16>` 是每 lane 8 个 bf16，B 的 `vector<2x2xbf16>` 是 4 个，D 的 `vector<2x2xf32>` 是 4 个 fp32；16×8 输出的 128 个元素除以 32 lane 正好为 4。`prev` 到结果的 SSA 链承担跨 K 段累加。坐标再给出语义：A 行使用 `g` 与 `g+8`，K 坐标按 `2*tig` 及相邻元素、8 列半区展开；B 的 N 坐标使用 groupID `g`；D 的四个元素位于行 `g/g+8`、列 `2*tig` 与 `2*tig+1`。

取 L=5 手算，g=1、tig=1。该 lane 的 A 行落在 1 与 9，B 负责 N=1 的列片段，D 却负责输出列 2 与 3、行 1 与 9。若只看“每 lane 四个输出”而忽略坐标，很容易把 B 的 N=1 错用于 D 的列起点。向量中的 `[0,0]、[0,1]、[1,0]、[1,1]` 也不是任意排列：前两元素对应行 g 的相邻两列，后两元素对应行 g+8 的相同两列。写 epilogue 时，extract 下标、行值与列值必须作为一组核对。

G3 的 1168/2080 缺陷就出在 B 与 D 的列坐标混用。旧 epilogue 把 D 的偶数元素写到 B 所用的 `ncol+g`，导致 lane 间冲突写与未覆盖列，kkt 仅覆盖 1168 个严格下三角元素，o 也只有 2304/4096 个非零。当前代码用 `%ncolt` 和 `%ncolgp1` 分别写 D 的偶、奇列；G4 修复后 kkt 精确覆盖 2080 个参考位置，o 全覆盖，MMA 配置对 fp64 参考的最大差为 bf16 输出量化量级。[tools/gdn_l2_gen.py:273-282,338-358][evidence/G4/GATE_REPORT.md:53-75]

同一类型检查也解释当前生成器的 tf32 路径选择：它使用 m16n8k4，A/B fragment 改为 `vector<2x1xf32>` 与 `vector<1x1xf32>`，D 仍为 fp32。这里的 K=4 是该生成器路径当前的选择，不应被理解为上游上限；NVGPU 的 shapeK 只是基本片大小，kTile 可以组合多个基本片，当前上游已能表达 TF32 m16n8k8。[tools/gdn_l2_gen.py:296-318；../llvm-mlir/mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp:218-264]

### 2. ldmatrix 的地址是每 lane 输入

`nvgpu.ldmatrix` 让 warp 从共享内存聚拢 8×8 的 16 位矩阵片。每条 lane 提供一个 16B 行地址；x4 的四个矩阵片按 lane 分组取得地址，树内下降不会把统一基址自动扩散成这些行。当前参考核显式计算象限坐标：

```mlir
%lr8 = arith.remsi %lane, %c8 : index
%ld8 = arith.divsi %lane, %c8 : index
%ld8m2 = arith.remsi %ld8, %c2 : index
%arowoff = arith.muli %ld8m2, %c8 : index
%ld16 = arith.divsi %lane, %c16 : index
%acol = arith.muli %ld16, %c8 : index
%arow = arith.addi %lr8, %arowoff : index
%af = nvgpu.ldmatrix %sa[%arow, %acol] numTiles = 4 transpose = false : ...
%brow = arith.remsi %lane, %c16 : index
%bf = nvgpu.ldmatrix %sb[%brow, %c0] numTiles = 2 transpose = true : ...
```

[tools/gdn_l2_gen.py:720-738；两条 op 的完整 memref/vector 类型省略]

A 的地址满足行 `L%8 + 8*((L/8)%2)`、列 `8*(L/16)`，对应 TL、BL、TR、BR 四个象限；B 的 x2 trans 路径用 `L%16` 选择 K 主序行。读 PTX 时应看到每条 lane 的地址计算先于 `ldmatrix...shared`，且上游共享 tile 已写入并完成同步。统一 `%sa[%c0,%c0]` 也能通过类型检查，却让所有 lane 指向同一行；G4 曾观察到 16 行输出相同、最大差 0.032。采用上述地址后，参考锚与 fp64 结果完全一致。[evidence/G4/GATE_REPORT.md:167-177][evidence/G4/gen_verify/ref_ldmatrix/kernel_smemfix.ptx:83-96]

这里的一个地址对应一整行 16B，并非该 lane 最终持有的一个元素。硬件完成跨 lane 聚合后，x4 返回寄存器的次序还与四个矩阵片一一对应；地址象限顺序或 transpose 属性改变时，fragment 解释也要同步改变。verifier 不会替读者证明每 lane 地址落在有效行内，也不会证明此前的 `st.shared` 已被其他 warp 看见，因此需要把地址公式、屏障和最终 MMA 结果一起审计。

### 3. 异步复制的 token 只追踪完成，跨 warp 可见仍需同步

`nvgpu.device_async_copy` 发起全局到共享复制并返回 token；`device_async_create_group` 把本线程的一组 token 组成完成组；`device_async_wait ... numGroups = 0` 等到本线程没有未完成的旧组，非零 N 则允许保留较新的 N 组。等待具有每线程语义，其他 warp 的复制是否完成、共享写入是否对读者可见，需要另行建立同步关系。

```python
def agroup(e, toks):
    g = e.v("agr")
    e.l(f"{g} = nvgpu.device_async_create_group {', '.join(toks)}")
    return g

def await_all(e, g):
    e.l(f"nvgpu.device_async_wait {g} numGroups = 0")

# astage_tile 先按线程跨步发射 16B copy 并收集 token
g = agroup(e, toks)
await_all(e, g)
e.l("gpu.barrier")
```

[tools/gdn_bwd_gen.py:1662-1674,1711-1740，函数与调用体合并节选]

这段同步封装中，token 列表必须覆盖所有本线程发射的 copy，`gpu.barrier` 再把各线程的完成等待连接成 CTA 级发布点。手工按线程跨步分配 16B 块还避免地址流被所有 warp 重复杂射；`nthr` 必须与真实发射线程数一致，否则部分块无人复制。需要预取与计算重叠时，wait 应推迟到首次消费之前；树内某些 `create_async_groups` 路径会在组后立即 wait 0，手工发射的价值正在于由调用者决定这个同步点。[tools/gdn_bwd_gen.py:1664-1685]

双缓冲可以据此理解：第 i+1 组复制已经发起后，消费第 i 个缓冲区前使用 `numGroups = 1`，允许较新的组保持未完成，同时等较旧组完成。这个数值描述尚未完成的组数，既不表示字节数，也不表示等待多少线程。若消费者跨 warp，仍要在旧组完成与读取之间设置 CTA 或 mbarrier 可见性协议；只把 0 改成 1，无法自动得到正确的流水。

### 4. mbarrier 是相位协议，elect 与寄存器调整也有执行范围

当前 warp 专化路径把 mbarrier 用作纯数据就绪屏障。初始化计数为 1，生产者 warp 选出的 leader 是唯一到达者，消费者只等待相位；TMA 开启时，到达者还用 `expect_tx` 预告搬运字节，两个 TMA 完成的 transaction credit 与这一次 arrive 共同完成相位。

```mlir
%bar = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>>
scf.if %isz { gdn.mbarrier.init %bar, %c1 : ... }
gpu.barrier
// 生产者 warp
 gdn.setmaxnreg dec 96
 %el = gdn.elect -> i1
 scf.if %el {
   gdn.mbarrier.arrive.expect_tx %bar, %c16384 : ...
   gdn.tma.async.load %descA[%k0, %c0, %c0, %c0], %bar to %sa : ...
   gdn.tma.async.load %descB[%k0, %c0, %c0, %c0, %c0], %bar to %sb : ...
 }
// 消费者 warps
 gdn.setmaxnreg inc 224
 gdn.mbarrier.try_wait.parity %bar, %parity, %cticks : ...
// ……MMA 消费数据……
gpu.barrier
%parn = arith.xori %parity, %true : i1
```

[evidence/G4/gen_verify/kkt_bf16_mma_tma_ws/payload.mlir:313,370-396,1199-1200，控制流缩排与重复类型省略]

这里有两组独立同步。init 后的 `gpu.barrier` 保证屏障初始化先于首次 arrive/wait 对各 warp 可见；循环尾的 `gpu.barrier` 保证消费者读完单缓冲后，生产者才能进入下一次覆盖。`parity` 每轮异或翻转，使 try_wait 等待正确相位。16384B 恰为两个 64×64 bf16 tile 的总字节数，expect_tx 若与描述符实际搬运量不符，数据就绪条件同样失真。非 TMA 路径则先 wait 本生产 warp 的 cp.async group，再由 elected leader 做普通 arrive。[tools/gdn_l2_gen.py:544-547,585-631]

协议可以按相位逐项审计，数据就绪和缓冲复用分别对应最后两行：

| 步骤 | 状态与动作 | 证明的责任 |
|---|---|---|
| 初始化 | `init count=1`，随后 CTA barrier | 屏障对象先于任何 arrive/wait 对所有 warp 可见 |
| 生产 | elected leader 做 `arrive.expect_tx(16384)`，发出两次 TMA | 到达次数与预告字节数正确，且只有一个到达者 |
| 完成 | 两个 TMA 的 transaction credit 与 arrive 共同完成当前相位 | 数据实际到齐，而非只有指令已发射 |
| 消费 | 消费者 `try_wait.parity` 通过后读取 tile | 等待的是正确相位，消费者不需要再 arrive |
| 复用 | 消费结束后的 CTA barrier，parity 异或翻转 | 下一次覆盖发生在全部读者结束之后，形成下一轮相位 |

这张表也解释了为什么计数为 1：相位完成所依赖的 arrive 只能来自在等待之前就能执行的参与者。若还要求正在 try_wait 的消费者事后 arrive，相位完成便会依赖一个被该相位阻塞的动作。数据就绪与缓冲复用也要分开证明；前者由 mbarrier 相位完成，后者由循环尾全 CTA barrier 防止过早覆盖。增加 arrive 次数或延长 try_wait ticks 都无法修补这两个条件中的缺失项。

`gdn.elect` 下降为 `NVVM::ElectSyncOp`，在活跃 warp 中选出一条 lane，适合唯一的 TMA 发射或 mbarrier arrive；它不承担 CTA 级互斥。`setmaxnreg dec 96` 与 `inc 224` 分别声明生产者和消费者的寄存器上限，应按 warp 一致执行，并结合角色数与占用预算检查。方言 verifier 对数值只要求 8 的倍数等局部条件，无法证明分支是否 warp 一致，也无法证明整个相位协议无死锁。[dialect/lib/GdnHatch.cpp:29-49,243-250]

### 5. TMA 描述符采用 CUDA 维序，装载坐标保持该顺序

张量内存加速器（TMA，Tensor Memory Accelerator）依据描述符搬运多维盒形区域。MLIR memref 的最内维位于维度表末尾，CUDA TensorMap 的 c0 则对应最内维。主机辅助函数因此在编码时反转 globalDim、globalStrides 与 boxDim：

```cpp
globalDim[i] = static_cast<uint64_t>(descriptor->sizes[Rank - i - 1]);
globalStrides[i] = static_cast<uint64_t>(
    descriptor->strides[Rank - i - 2] * elementSizeInBytes[tensorDataType]);
// ……
boxDim[r] = static_cast<uint32_t>(inputBoxDims[tensorRank - r - 1]);
```

[llvm-mlir/mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp:505-520,568-570，循环节选]

这段代码回答描述符怎样编码。设备装载坐标沿另一路径直接透传进 `cp.async.bulk.tensor` 的 c0..cN，不再反转，所以 payload 已按 CUDA 序写 `%k0` 在第一位。G4 的 descA 是 rank-4，`global_dim=[128,64,1,1]`、`box_dim=[64,64,1,1]`、坐标为 `[%k0,0,0,0]`；descB 同理以 K 为最内维。共享 B tile 也排成 `[bv,bk]` 的 K 连续形式，才能由行主序输入连续填充。[evidence/G4/gen_verify/kkt_bf16_mma_tma_ws/tensormap_spec.json:10-57][tools/gdn_l2_gen.py:518-531]

描述符类型与坐标正确仍遗漏单位和目的地址合同。globalDim 与 boxDim 以元素计数，globalStrides 以字节计数，K 坐标 `%k0` 也是元素偏移；把 64 个 bf16 的 128B 错填成坐标 128，会把装载位置放大两倍。编码后的 CUtensorMap 由主机上传并传设备指针，G4 装置采用 256B 对齐的描述符分配。共享内存目的 tile 另有 128B 对齐要求，即使 `swizzle=none` 也适用，两种对齐不要混为同一对象。[evidence/G4/GATE_REPORT.md:106-142]

当前 PTX 修复器为静态 tile 统一发射 `.shared .align 128 .b8 symbol[size]`，并插在 `.visible .entry` 之前。[tools/gdn_smem_fix.py:199-208] G4 记录过 16B 对齐导致 `CUDA_ERROR_MISALIGNED_ADDRESS`，sanitizer 定位到 UTMALDG；128B 修复后 TMA 配置完成并通过数值检查。主机还要按 spec 编码 CUtensorMap、上传并把设备指针作为 kernel 参数传入，cubin 中存在 TMA 指令并不能替代这份运行时接线。[evidence/G4/GATE_REPORT.md:106-148]

最终审计应把 verifier、PTX/SASS 和全输出验证连起来。fragment 列坐标、ldmatrix 每 lane 行地址、异步组的跨 warp 可见性、mbarrier 到达计数与缓冲复用、TMA 字节数和维序，都超出单纯 IR 类型合法域。当前修复版的证据是 G4 在 seed 0 与 seed 7 下均为 20/20 通过，重发射逐位确定；这些结果验证的是已执行配置及其协议，仍应随生成器改动重跑。[evidence/G4/GATE_REPORT.md:8-17,49-56]
