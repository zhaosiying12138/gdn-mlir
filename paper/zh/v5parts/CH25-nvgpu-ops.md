## 二十五、nvgpu 与 nvvm op 地图：verifier 规则与陷阱

L3 生成器直接把机器层 op 写进 IR：nvgpu.mma.sync、nvgpu.ldmatrix、nvgpu.device_async_copy、mbarrier 族、TMA 族、elect。这些 op 各自带两层契约——MLIR verifier 强制的类型与形状规则，和硬件强制的寄存器布局、地址几何、完成语义。契约弄错一层，结果通常是形状正确而数值整体错误，或者干脆挂死。以下用法均摘自生成器真实发射代码。先定义两个贯穿全文的词：fragment（片段）指一张分布式 tile 在单线程寄存器里的视图——32 个 lane 各持几个元素，拼起来才是完整的 mma 操作数；verifier 指 MLIR 在 IR 检查阶段强制的合法域，越界即编译失败。

### 1. nvgpu.mma.sync：8×8×128b 基本片与片段类型

nvgpu.mma.sync 的 verifier 强制最小基本片为 8 行 × 8 列 × K 维 128 比特。由此直接推出各 dtype 的 K 封顶：128 比特装 bf16（每元素 16 比特）恰为 8 对，即 mmaShape [16,8,16]；装 tf32（每元素 32 比特）只有 4 个，tf32 在此路线上封顶 K=4。本仓 bf16 主力发射（全族 kernel 同款）：

```
e.l(f"{o} = nvgpu.mma.sync(%fa_{name}_{mi}_{kb}, "
    f"%fb_{name}_{ni}_{kb}, {prev}) mmaShape = [16, 8, 16] : "
    f"(vector<4x2xbf16>, vector<2x2xbf16>, "
    f"vector<2x2xf32>) -> vector<2x2xf32>")
```
［tools/gdn_bwd_gen.py:1886-1892，mma_gemm 内层］

三个类型串的含义：A 片段 vector<4x2xbf16> 是每 lane 8 个 bf16（4 对 k 相邻元素 × 2 行半），B 片段 vector<2x2xbf16> 每 lane 4 个，累加器 vector<2x2xf32> 每 lane 4 个 fp32——16×8 输出共 128 元素除以 32 lane 恰为 4；prev→o 的 SSA 链就是同累加器跨 k 段累积。tf32 的 K=4 封顶正是第二十六章逃生舱的必要性之一：反向需要的 m16n8k8 拼写在此路线结构上不可达。

### 2. nvgpu.ldmatrix：行广播陷阱与象限地址

ldmatrix 让 32 个 lane 各交一个 16 字节行的共享内存地址，硬件聚拢成 8×8×16b 的片段。陷阱：树内下译把这一个地址寄存器原样透传、不做任何行扩散——若 IR 里写 uniform 基址（全部 lane 同一 [%c0,%c0]），32 个 lane 就寻址同一行，mma 读到行广播的 A：GPU 输出 16 行全部相同、对参考最大偏差 0.032（evidence/G4/GATE_REPORT.md 2.5 节）。修复是生成器为每 lane 显式发射象限地址——m16n8k16 的 A 片段四个象限按 lane L → 行 L%8 + 8*((L/8)%2)、列 8*(L/16) 分布，B 转置行按 L%16：

```
# ldmatrix per-lane quadrant indices (G4 anchor semantics)
%lq7 = arith.remsi %mL, 8 : index        # lane 在 8 行内的行号
%lq8 = arith.divsi %mL, 8 : index
%lqh = arith.remsi %lq8, 2 : index       # 行半选择（上/下 8 行）
%lqt = arith.divsi %mL, 16 : index       # 列块选择（左/右 8 列）
%lh8 = arith.muli %lqh, 8 : index
%lt8 = arith.muli %lqt, 8 : index
```
［tools/gdn_bwd_gen.py:1846-1852；两条 nvgpu.ldmatrix 的发射在 1872-1885 行］

修复后锚点配置对参考逐位为零。本仓另一条与 ldmatrix 共生的规则（r6）：凡喂 ldmatrix 的 tile 必须经 st.shared 写入——cp.async 落盘的 tile 经 ld.shared 读回正确、经 ldmatrix 读回却静默损坏（第四十章发现 5）；生成器为跨过该缺陷备有恒等重写 launder_pass 把 tile 先 st.shared 一遍（tools/gdn_bwd_gen.py:1743-1745 注释）。

### 3. nvgpu.device_async_copy：token、组与每线程语义

device_async_copy 发起一条全局到共享的异步拷贝并返回一个 token；token 列表交给 device_async_create_group 成组；device_async_wait %g numGroups = N 等到未完成组不超过 N。两个契约要点。其一，wait_group 完成是每线程语义——本线程的 wait 只保证本线程发起的拷贝就绪，跨 warp 可见性必须补显式 gpu.barrier（第四十章发现 10）。其二，树内 transform 的 create_async_groups 会在每组后立即插 wait_group 0，异步复制结构性无法与计算重叠（第十三章），所以本仓全部手动发射。同步落盘的封装：

```
def agroup(e, toks):
    e.l(f"{g} = nvgpu.device_async_create_group {', '.join(toks)}")
def await_all(e, g):
    e.l(f"nvgpu.device_async_wait {g} numGroups = 0")
def astage_sync(...):
    toks = []; astage_tile(...)   # 线程跨步认领 16B 块
    g = agroup(e, toks); await_all(e, g)
    e.l("gpu.barrier")            # 跨 warp 可见性
```
［tools/gdn_bwd_gen.py:1718-1741，节选；astage_tile 线程跨步认领在 1687-1714 行］

落到 PTX 即第一章引过的 cp.async.ca.shared.global 突发串（地址按线程跨步 +16/+32/+48）。该模式在反向链替换同 dtype 标量搬运后，dqkwg 从 2.53 降到 1.55 毫秒。

### 4. nvgpu.mbarrier：计数合同与纯数据就绪屏障

mbarrier（内存屏障）族：create 建组、init 设到达计数、arrive 记一次到达、arrive.expect_tx 预告传输字节数、try_wait.parity 按相位轮转等待。陷阱在计数合同：初版协议把单一 mbarrier 的 init 计数设为 32×warps，生产者选首 lane 到达一次、再让每个消费者线程在 mma 后各到达一次——相位完成需要 1+32×warps 次到达，而消费者此刻正阻塞在等待该相位的 try_wait.parity 里，等待的门数包含着自己，第 0 次迭代即挂死（evidence/G4/GATE_REPORT.md 2.2 节，G3 记录 TIMEOUT）。修复后的协议把 mbarrier 定义为纯数据就绪屏障：

```
%bar = nvgpu.mbarrier.create -> <memorySpace = ...workgroup>
scf.if %isz { gdn.mbarrier.init %bar, %c1 : ... }   # init 计数 1
gpu.barrier                                          # init 可见性
  gdn.setmaxnreg dec 96                              # 生产者 warp
  %el = gdn.elect -> i1
  gdn.mbarrier.arrive.expect_tx %bar, %c{txbytes}    # 唯一到达者
  gdn.tma.async.load %descA[%k0, 0, 0, 0], %bar to %sa
  gdn.setmaxnreg inc 224                             # 消费者 warp
  gdn.mbarrier.try_wait.parity %bar, %parity, %cticks  # 只等待，不到达
```
［tools/gdn_l2_gen.py:564、571-574、585-593、624-625，同一函数内节选；相位异或翻转在 631 行］

要点：init 计数 1、生产者选首 lane 是唯一到达者（TMA 开启时 expect_tx 预告的字节数与两次装载的 complete_tx 抵扣共同完成相位）；消费者只等待；init 之后立刻 gpu.barrier 保证全 warp 可见；单缓冲复用安全由循环尾的 gpu.barrier 承担。该协议 20/20 配置真机全过、逐位确定。

### 5. TMA：encode 维度反序、坐标不反、128B 对齐

TMA（张量内存加速器）用描述符描述全局张量，硬件按盒（box）整块搬运。三份契约缺一不可。其一，编码维度反序：树内主机辅助函数 mgpuTensorMapEncodeTiledMemref 把 memref 维序逆序成 CUDA 维序填 globalDim/globalStrides/boxDim。其二，坐标不反：装载坐标原样透传进 cp.async.bulk.tensor 的 c0..cN、c0 为最内维，所以 payload 里坐标必须直接按 CUDA 序写（K 偏移在最前）。这两条都是读上游运行时源码（CudaRuntimeWrappers.cpp）逆推得到的——文档没有把"编码反序、坐标不反"的组合写成一句合同，读上游实现是唯一定约途径；结论被固化为生成器侧的 tensormap_spec.json，由验证器按 spec 用 cuTensorMapEncodeTiled 编码并传设备指针（evidence/G4/GATE_REPORT.md 2.3 节）。其三，共享内存目的 tile 必须 128B 对齐，即使 swizzle 关闭——曾以 .align 16 声明，ptxas 把 tile 摆到 16B 而非 128B 边界，首次装载即 CUDA_ERROR_MISALIGNED_ADDRESS，由 compute-sanitizer 定位在 UTMALDG.4D。描述符类型与装载：

```
!nvgpu.tensormap.descriptor<tensor = memref<1x1x64x64xbf16, 3>,
    swizzle = none, l2promo = none, oob = zero, interleave = none>
gdn.tma.async.load %descA[%k0, %c0, %c0, %c0], %bar to %sa : ...
```
［tools/gdn_l2_gen.py:463-466、591-594；spec 样本 evidence/G4/gen_verify/kkt_bf16_mma_tma_ws/tensormap_spec.json——descA rank-4、box [64,64,1,1]、坐标 "K offset first"］

附带推论：TMA 以最内维连续填充共享内存，B tile 必须排成 k-行（K 连续）才能从行主序全局装出。

### 6. elect.sync 与下降薄层

gdn.elect 在 warp 的活跃线程里选出一条 lane 返回布尔，专用于"生产者 warp 只需一条线程发 TMA 或到达屏障"的场合（上两节引文中的 %el 即它）。下降是纯记号翻译：gdn.elect 替换为 NVVM::ElectSyncOp（dialect/lib/GdnHatch.cpp:43-51）；mbarrier 族与 TMA 族同样一一映射到树内 nvgpu op（53-130 行），零预算支出。全套 warp 专化词汇（setmaxnreg 配对 dec 96/inc 224）由此保持"语义在 gdn 方言、机制在树内"的分层。
