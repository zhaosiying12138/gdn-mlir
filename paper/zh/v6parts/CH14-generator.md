## 十四、从十个参数到三份产物：生成器、解释器与表达边界

`kkt_bf16_mma_p2_t64x64x32_w4` 同时是配置名、证据目录名和 payload 首行的一部分。它背后的决策向量有十维：`dtype`、`prec`、tile 的 M/N/K、`warps`、`pipeline`、`mma`、`warpspec`、`tma`。`stage` 与全局 K/V 给定问题，目录名和 `driver` 服务于产物组织，均不增加这十维自由度。生成器的职责是把这份向量变成同源的 `payload.mlir`、`transform.mlir` 与 `config.json`，再由解释器和离线工具链生成机器产物。

### 1. 先校验，再发射

校验函数把跨参数条件集中在发射之前。下面一段守护 MMA fragment 与 K 循环；更早的代码还检查类型、warp 数、warp 专化和 TMA 的组合，并为缺省的 `prec`、`warpspec`、`tma` 填值。

```python
if cfg["mma"]:
    if bt % 16:
        raise ValueError("mma=on requires tile.M multiple of 16")
    if dt == "bf16" and bk % 16:
        raise ValueError("mma=on bf16 needs tile.K multiple of 16")
    if dt == "f32" and bk % 4:
        raise ValueError("mma=on f32(tf32) needs tile.K multiple of 4")
    if bv % (warps * 8):
        raise ValueError("mma=on requires tile.N multiple of 8*warps")
    if cfg["pipeline"] > 0 and k_glob // bk < cfg["pipeline"]:
        raise ValueError("pipeline depth exceeds K-loop trip count")
```
[tools/gdn_l2_gen.py:119-129]

这些检查对应具体 IR 前提：M/N/K 要铺满 fragment，每个 warp 要取得完整 N 侧份额，流水版本数不能超过 K 迭代数。内建配置经 `C()` 调用 `validate()`，JSON 配置在 `main()` 中也先校验 [tools/gdn_l2_gen.py:1174-1180、1321-1327]。校验发生在本次目录写入之前，不会先写出半份 payload 再报错；已有目录里的旧产物仍需单独管理。

发射侧由 `gen_payload()` 与 `gen_transform()` 分担。前者依据 `mma` 和 `warpspec` 选择直接 MMA fragment、linalg 或生产者—消费者模板，并把 tile、类型和 launch 几何写入文本 [tools/gdn_l2_gen.py:908-919、1075-1090]；后者只携带当前 transform 能表达的调度动作 [tools/gdn_l2_gen.py:1103-1167]。`Buf.l()` 累积行，`Buf.text()` 加末尾换行，输出次序完全由调用次序决定 [tools/gdn_l2_gen.py:145-153]。

```python
with open(os.path.join(out, "payload.mlir"), "w") as f:
    f.write(gen_payload(cfg))
with open(os.path.join(out, "transform.mlir"), "w") as f:
    f.write(gen_transform(cfg))
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
if cfg.get("tma") == "on":
    with open(os.path.join(out, "tensormap_spec.json"), "w") as f:
        json.dump(tensormap_spec(cfg), f, indent=2)
```
[tools/gdn_l2_gen.py:1337-1345]

三次写入共享同一个已校验字典，避免分别重建参数造成漂移。TMA 配置额外产生 `tensormap_spec.json`，因为设备端描述符还需要主机编码合同；这不改变“三份基础产物”的划分。

从求解器进入时，`to_config()` 先把整数编码还原为类型字符串，把三个布尔量还原为开关，再按阶段补齐 K/V；`config_name()` 用同一字典生成规范名。warp 专化追加 `_ws`，TMA 再追加 `_tma`，o 阶段名还携带 K/V，防止不同形状落到同一证据目录 [tools/gdn_solver.py:261-289]。因此目录名是派生索引，配置字典才是生成输入。手工改名不会改动 IR，手工只改 payload 注释也不会改动配置；审查时应以字典、发射文本和实际机器产物三方一致为准。

### 2. 一个配置的端到端实例化

真实 payload 把配置同时保留在注释、类型、常量与操作结构里。下面合排关键行，省略常量表和展开的地址计算。

```mlir
// GDN L2 stage 'kkt': C = tril(k @ k^T) * exp2decay  knobs={"name":"kkt_bf16_mma_p2_t64x64x32_w4","stage":"kkt","dtype":"bf16","dims":{"K":128,"V":64},"tile":{"M":64,"N":64,"K":32},"warps":4,"mma":true,"pipeline":2,"prec":"bf16"}
func.func @gdn_kkt_kernel(%kg: memref<64x128xbf16>, %cg: memref<64x64xbf16>) {
  // ……
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%bxx = %c128, %byy = %c1, %bzz = %c1) {
    // ……
    %sa0 = memref.alloc() : memref<64x32xbf16, #gpu.address_space<workgroup>>
    %sb0 = memref.alloc() : memref<64x32xbf16, #gpu.address_space<workgroup>>
```
[evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/payload.mlir:1-2、69-70、214-215，合排]

`dtype=bf16` 进入输入、输出和共享 tile 的元素类型；tile.M=64 与 tile.K=32 进入 `%sa0`，tile.N=64 与 tile.K=32 进入 `%sb0`。网格为 1×1，因为全局 64×64 输出恰好由一个 tile 覆盖；128 个线程等于四 warp。K=128 的全局归约在第 256 行以 32 为步长进入 `scf.for`，MMA 原子的 `mmaShape = [16, 8, 16]` 和 bf16/f32 fragment 类型可在第 1310 行核对 [evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/payload.mlir:256、1310]。这些位置让读者能从产物反推参数，而不必相信目录名。

```json
{
  "name": "kkt_bf16_mma_p2_t64x64x32_w4",
  "stage": "kkt",
  "dtype": "bf16",
  "dims": {
    "K": 128,
    "V": 64
  },
  "tile": {
    "M": 64,
    "N": 64,
    "K": 32
  },
  "warps": 4,
  "mma": true,
  "pipeline": 2,
  "prec": "bf16"
}
```
[evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/config.json:1-18]

`config.json` 是机器可读视图，payload 注释便于人工追踪，二者应能对账。该 G2 归档没有 `warpspec` 与 `tma` 键，这是旧归档的字段形态；当前生成器默认二者为 `off`，求解器的 `to_config()` 则显式写出 `producer_consumer/off` 与 `on/off` [tools/gdn_l2_gen.py:93-103；tools/gdn_solver.py:261-275]。审查旧目录时应记录这个差异，不能虚构归档中不存在的键。

十维进入三份基础产物的路径可以直接沿分支追踪。`dtype/prec` 改 payload 类型和 MMA 原子；三个 tile 改 launch、memref 与循环，其中 M、N 还经过上限为 16 的模板计算决定 no-MMA 工件切块；`warps` 改线程数和 warp 的 N 偏移；`pipeline` 改工件的 multibuffer factor 与流水 depth；`mma` 同时选择 payload 模板和 transform 分支；`warpspec/tma` 选择直接操作、描述符与 identity 工件。所有值都进入配置字典，只有当前表达路径需要的字段会改变 transform 文本。

一个有效的实例化检查应当同时查看差分与不变量。把普通 MMA 配置的 pipeline 从 2 改为 3，config 数值、工件 factor/depth、缓冲版本和 PTX 复制组织应一起改变，输入数学语义与函数参数则保持不变。把 `warpspec` 打开后，payload 会大幅改写而 transform 变成 identity，这仍是配置生效。反过来，若一个旋钮只改了目录名，三份产物和 PTX 均无差异，就应立即检查它是否在当前分支被固定、被默认值覆盖，或根本没有接入发射路径。

### 3. 从两份 IR 到 cubin 与资源账

`scripts/run_l2.sh <config-dir>` 先合并并解释两份 IR，再擦除调度、下降 NVVM、提取 `kernel.ptx`。第 81 行用 `ptxas -v -arch=sm_120a kernel.ptx -o kernel.cubin` 同时生成二进制和资源报告，第 84 行生成 `sass.txt`，第 93–127 行把配置、ptxas 指标、PTX/SASS 计数写入 `meta.json` [scripts/run_l2.sh:69-84、93-127]。当前脚本还保存 `asmobj.mlir`，便于检查 launch 参数 [scripts/run_l2.sh:75-78]。

```text
"metrics": {
  "registers": 255,
  "smem_static": null,
  "spill_stores": 5708,
  "spill_loads": 6920,
  "stack_frame": 3808
},
"ptx_features": { "mma.sync": 48, "cp.async": 1542 },
"sass_ops": { "HMMA": 48, "LDGSTS": 1536 },
"ptx_size_bytes": 282907
```
[evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/meta.json:18-29、51-62，字段合排]

原始报告还给出 3808 字节栈帧、5708 字节溢出写、6920 字节溢出读、255 寄存器和一个 barrier [evidence/G2/gen/kkt_bf16_mma_p2_t64x64x32_w4/ptxas_v.txt:1-6]。这些数字是 ptxas 对实际 PTX 的分配结果，可用来复核求解器的资源假设；`smem_static=null` 也提醒读者，这份 G2 lowering 没留下可解析的静态共享内存数字，不能把空字段当作零字节。`cubin` 证明汇编成功，PTX/SASS 计数证明调度特征存在，数值正确性仍需独立验证。

这里存在三种不同口径的容量检查：生成器 `validate()` 用 220 KiB 挡下明显超出演射边界的配置，Z3 strict 用 100,352 B 建模可装载预算，ptxas 与 CUDA 装载面对最终二进制 [tools/gdn_l2_gen.py:130-138；tools/gdn_costmodel.py:112-114]。前两者都是模型或工程门槛，最后一层才可能包含对齐、静态与动态分配及驱动细节。容量对账要比较同一生命周期和同一份分配清单，不能只比较两个总数是否接近。

### 4. 三个参数平面按消费位置划分

第一类是上述 L2 config：它描述 stage-kernel 的十维空间，可由 Z3 枚举，并随三份基础产物归档。第二类是生产生成器的 `GDN_*` 环境参数，例如 `GDN_F1WY` 选择 warp 并行求解，`GDN_SCANDB` 选择扫描双缓冲 [tools/gdn_full_gen.py:1091-1101、1150-1159]。它们控制完整算子的模板分支、驻留和手工 staging，不自动进入 L2 的十维求解域。

第三类在 PTX 与汇编阶段消费。`GDN_MAXREG_K` 按入口名子串给多入口 PTX 分别插入 `.maxnreg`，`GDN_MAXREG` 则给整个文件追加 ptxas 的 `--maxrregcount` [tools/gdn_smem_fix.py:683-712、789-800]。此时 transform 调度早已擦除，逐 kernel 的 PTX 指令无法由前面的 transform op 表达。相同的环境变量前缀不代表同一平面，判断依据是参数在哪个工具、哪个表示层被读取。

完整算子构建把全部 `GDN_`/`G5` 环境值以及生成器、PTX 修改器源码纳入缓存键，避免不同参数平面误用旧 cubin [tools/gdn_full_rt.py:28-44]。这个键控连接了三类参数，却没有把它们合并成一个已经由 Z3 完备描述的搜索空间。

复现记录也应按平面拆开：L2 实验保存 config 与两份 IR；生产模板实验保存环境变量子集及生成器版本；PTX 实验还要保存修改后的 PTX、汇编器选项和逐入口报告。只记录其中一层，可能得到相同的 MLIR 却得到不同的 cubin，也可能得到相同的 PTX 却使用了不同的主机描述符。缓存键能防止误用旧产物，完整证据仍需能解释每一次文本变化来自哪个平面。

### 5. 生成器与 transform 的表达边界

warp 专化在当前 L2 中由 payload 直接写成 `scf.if` 生产者—消费者分支，含 warp 编号广播、`gdn.setmaxnreg dec 96/inc 224`、mbarrier 和 elect；对应 `gen_transform()` 只发 identity 序列 [tools/gdn_l2_gen.py:565-568、584-625、1108-1112]。所以 warp 专化是十维配置的一部分，其结构实现位于 payload 模板，不能据此声称所有调度都由 transform op 表达。

手工 staging 也依赖模板纪律。生产生成器用线程跨步的 `scf.while` 组织 cell 搬运，并刻意让非 K 循环避开当前 pipeliner 的匹配范围 [tools/gdn_full_gen.py:1129-1147]。线程到地址的分配、bank 几何和跨 kernel 缓冲复用需要显式下标与资源设计。PTX `.maxnreg` 更位于 MLIR 之后。三层介质各自保留可检查文本：payload/transform、生产模板及其环境配置、PTX 与 ptxas 报告。

求解范围同样需要分层。Z3 和成本模型排序的是 kkt、o 等 stage-kernel 配置，单个目录的 cubin 只实例化一个阶段。完整算子的发布选择还受多 kernel 交互、主机协议和实际负载影响，G5 使用了成对实测 A/B；G6 明确把这部分与离线排名成本分开记账 [evidence/G6/search_cost.md:93-111]。三份基础产物给出可复现的实例化入口，ptxas 给出静态复核，完整算子的采用决策仍要由设备测量闭合。
