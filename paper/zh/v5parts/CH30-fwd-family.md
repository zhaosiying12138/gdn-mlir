## 三十、前向内核族：三 kernel 分工与五个杠杆

前向算子一次调用要完成三件性质迥异的工作：chunk 内的 WY 预计算（纯并行的代数）、跨 chunk 的状态扫描（全算子唯一的串行依赖）、从中间量重算输出（带宽受限的读回）。三件事对并行度、驻留、数据通路的要求互相冲突，任何单一 kernel 形态都要为其中一件牺牲另一件。最终发运的解法是三个 kernel 各领一件事、装进单一 cubin；投入产出经过实测裁决的五个杠杆随后按机制展开。术语先定义：CTA（cooperative thread array，一个线程块）、grid（CTA 网格）、驻留（每 SM 同时容纳的 CTA 数）、NC（序列切成的 chunk 数，块长 64）。所有数字取自三个判定形状上的同窗交错配对测量。

### 1. 三 kernel 分工与 mono cubin 的流序同步

| kernel | grid | 职责 | 并行性 |
|---|---|---|---|
| F1（wy） | (NC, B·H) | gc 扫描、A、T、u、w 的 WY 预计算 | chunk 并行，每 CTA 一个 chunk-head 对 |
| F2（scan） | (B·H·VT, 1) | 跨 chunk 状态递推，状态常驻共享内存，产出逐 chunk 状态序列、更新后的 v 与终态 | 全算子唯一串行链 |
| F3（o） | (NC·B·H, VT) | 从逐 chunk 状态序列与更新后的 v 重算输出 o | chunk 并行，带宽受限 |

发射载体是 mono：生成器把三个零实参 kernel 与一套公共全局缓冲封进一个 cubin，F1 的 gc 输出在装配时直接改名为 gc、三个 kernel 共享同一全局。关键性质写在装配函数的契约里：

```
def wrap_module_mono(kernels, params, meta):
    """One module, one set of public globals, three zero-arg kernels.
    Kernel boundaries are the stream-order syncs; the staged path then runs
    without any device-to-device handoff copies."""
```

［tools/gdn_full_gen.py:3925-3929］

kernel 边界即流序同步：同一 stream 上 F2 天然等 F1 完成、F3 等待 F2，无需任何显式事件或信号量。分立 cubin 时代的设备到设备交接拷贝（每形状约 0.1 毫秒）随之消失。正式配置三个判定形状同套：mono、warps=4、bv=32，加环境旋钮 GDN_F1KS、GDN_F2US、GDN_WST、GDN_ZFIX、GDN_SCANDB、GDN_F1ZF、GDN_W32P（后两个旋钮的机制在第三十四章）。

### 2. fused 旗舰的诚实定位

生成器另有 --path fused：单 persistent kernel，grid B·H·VT，chunk 循环进核，状态 S 常驻共享内存，每 chunk 依次八个 mma 位点。它实现完整、数值全过，端到端却是 4.53 对 0.785 毫秒（T8192），T2048 上为 1.14 对 0.21 毫秒。差距的根源在并行度：WY 预计算在三重奏里按 (NC, B·H) 全并行（T8192 上 2048 个 CTA），在 fused 里塌缩进 64-CTA 的串行 chunk 链——结构性差异靠移植 tile 修复不了。fused 保留为融合边界旋钮的极限端设计记录，正式判定由三 kernel 方案承担。配套审计同时记录：三重奏的全部全局到共享 staging 已是异步拷贝，仅存的同步全局读没有任何可重叠窗口——fused 内部那类异步流水机制在三重奏身上没有落点。

### 3. F1KS：双 CTA 驻留与延迟受限的证据

驻留翻倍是否转化为性能，取决于 kernel 的受限模式。F1 的实测画像：停顿的 24.3% 来自访存等待（long_scoreboard），DRAM 利用率仅 56.8%，占用 16.24%（每 SM 8 个 warp）——延迟受限，多一个驻留 CTA 能直接填充等待周期。三步把申报共享内存池压过双 CTA 门槛：bf16 求解压缩（去掉 f32 的 A 镜像 tile）、uw 段 B 操作数 K 分裂（半尺寸 tile 两趟复用）、生存期别名（半 tile 落在求解之后的死字节上，机制在第三十五章）。申报池 46,352B，恰在 49,152B 门槛之下。K 分裂的等价性有明确构造：分裂点落在四个 m16n8k16 块之间，每输出元素的 f32 加法序列与原四块链位级相同［tools/gdn_full_gen.py:2307-2316 注释］。发运 payload 里两趟预取组的形态：

```
%acp   = nvgpu.device_async_copy %v[%ix_223], %vt[%idv_5, %im_31], 8 bypassL1
    : memref<16777216xbf16> to memref<32x136xbf16, #gpu.address_space<workgroup>>
...  (8 条：v 半块 ×4 + k 半块 ×4，组 token 由调用方在 mma 前统一等待)
```

［evidence/G5/gen_fwd/B1T8192_mono/payload.mlir:2255］

配套的 ZFIX 与 SCANDB 修复（清零步长全覆盖、扫描双缓冲）保证双 CTA 共驻下逐位正确——两个被单 CTA 驻留掩盖的缺陷由此关闭。效果：F1 在 T8192 减少 53.5 微秒（278.7 到 225.2）。

### 4. F1ZF：清零折进空闲线程

删除探针给 phase-0 清零定价约 8.5 微秒：它为转置 T tile 的上三角写零，串行排在 64 线程的串行代入之前——同一时刻线程 64 至 127 全程空闲。杠杆是把清零折进这段空闲：

```
def zero_up(i):
    r = divsi(e, i, cC)                    # 行
    c = remsi(e, i, cC)                    # 列
    gt = e.v("svgt")
    e.l(f"{gt} = arith.cmpi sgt, {r}, {c} : index")   # 块级谓词（折叠后）
    e.l(f"scf.if {gt} {{")
    e.l(f"  memref.store {zbf}, {Tb}[{r}, {c}] : {Tb_ty}")
    e.l(f"  memref.store {zbf}, {Tg}[{r}, {c}] : {Tg_ty}")
    e.l("}")
```

［tools/gdn_full_gen.py:1367-1376；折叠臂的谓词与步长改写在 1341-1352 的注释契约中］

位一致性论证写在生成器注释里：折叠臂只写代入永不触碰的单元（串行链写全部同块单元，含其负零上角），写集不相交，T 的每个位与原构建确定性相同；一条 barrier 取代原来的两条。三形状各减 6.7 至 7.9 微秒，约为理论删除上限 8.5 微秒的 91%——折叠是重叠，删除是废除，两者之差就是剩余的 9%。

### 5. F2 垫行：读侧 bank 冲突的修复

被 ldmatrix（矩阵片段装载指令）读取的 tile 行距恰为 128 字节整数倍时，一条装载的 8 行落入同一 4-bank 组，形成 8 路冲突。修复是每个被读 tile 加 8 列 bf16 垫（16 字节）错开行距：F2 从 473 降到 249 微秒，前向单笔最大杠杆；F1 与 F3 的全部同类 tile 同样受益。bank 组织原理、冲突几何与垫的完整代数在第三十四章展开。

### 6. F2US：u 段重回乒乓预取管线

u 段每 chunk 每 CTA 4KB，此前由 fragment 映射直接从全局读：2 字节装载配 4 字节跨距，每个 32 字节扇区只交付一半字节，4KB 被放大成约 8KB 的内存扇区流量；每一发装载还都坐在串行链上，L2 延迟直接暴露。F2US 把 u 重新 staging 进组 A 乒乓缓冲：行乒乓 tile ut 为 2C 行乘 BV+8 列，随 w、k、gc 一起在 chunk 顶部发出、chunk 末收割［tools/gdn_full_gen.py:2694-2700 注释；池 84,480 到 94,720B，仍单 CTA］。发运 payload 中 epilogue 的读法已从全局变成带垫共享 tile：

```
%vnu = memref.load %ut[%ix_311, %nct_vp_0] : memref<128x40xbf16, ...>   ; F2US
%vnf = arith.extf %vnu : bf16 to f32
%vnd = arith.subf %vnf, %evn : f32            ; u − (w@S)
```

［evidence/G5/gen_fwd/B1T8192_mono/payload.mlir:4157 起，节选］

字节与算术顺序不变，位级一致。效果：F2 在 T8192 减少 104.5 微秒（60 对测量零对变慢），该形状的判定比值从 1.021 提升到 1.101。资源律同时立规：更宽的 w8/bv64 配置下 ut 翻倍使池达到 115,200B、超过 101,376B 上限，ptxas 拒绝——F2US 因此是按形状启用的环境旋钮，作无条件默认。

### 7. 时间去向：逐段删除探针

五个杠杆为何这样排序，由逐段删除探针回答：构建一个删掉某段输出的变体，测整 kernel 的时间差，得到该段的表观成本。F1 在 T8192 约 227 微秒基线上的读数：uw 段 116.5（51.2%）、kkt 59.5、solve 41.3、gc 扫描 17.6、phase-0 8.4 微秒。各段之和超过总时延，因为每段的删除收益含与其它段重叠的成分；相对排序可靠——临界路径在 uw 段，本章五个杠杆的作用点也集中于此。探针同时给后续工作立界：kkt 段的 staging 流实测折合 826GB/s，超过同窗 D2D 拷贝标定的 765GB/s 器件聚合上限，该段以算法最小流量顶在带宽屋顶线上，作为性能之门关闭。

{{FIG4}}
