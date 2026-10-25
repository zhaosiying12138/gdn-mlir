## 二十三、出货前的审计：ptxas、SASS、Sanitizer 与差分定位

cubin 装载前的最后一份静态资源账来自 ptxas，硬件实际执行的指令序可由 nvdisasm 反汇编为 SASS，越界与竞态则交给 compute-sanitizer 在真机上观察。三类证据回答的问题不同：资源账估计驻留与溢出，SASS 检查指令选择和控制流，动态工具捕获具体访问。只有把它们指向同一个产物，审计才成立。

### 1. ptxas -v 的字段要落到资源含义

`ptxas -v` 按 entry 报资源。一个 kkt 配置的真实输出为：

```text
ptxas-blackwell info : Compiling entry function 'gdn_kkt_kernel_kernel' for 'sm_120a'
ptxas-blackwell info : Function properties for gdn_kkt_kernel_kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas-blackwell info : Used 96 registers, used 1 barriers
```

[evidence/G4/gen_verify/kkt_bf16_mma_p0_t64x64x32_w4/ptxas_v.txt:2-5]

`Used 96 registers` 是每线程静态寄存器数。结合 CTA 线程数与每 SM 65,536 个 32 位寄存器，可计算寄存器允许的驻留 CTA 数；实际驻留还受共享内存、线程数和硬件粒度共同约束，项目审计所用的同一边界见[evidence/G5/bwd_report.md:2574-2599]。`bytes smem` 是每 CTA 静态共享内存，适合与双 CTA 门槛对账。`stack frame` 指向本地栈需求，`spill stores/loads` 按字节报告寄存器溢出到本地内存的规模，非零时需要继续查访问频率和所在热路径。`used 1 barriers` 是 barrier 资源占用，不等于动态执行了一次 `bar.sync`。

这份样例的 96 寄存器、1 个 barrier、三类零字节只描述该 entry。`gmem` 描述模块全局存储，也不能当作每线程开销。读报告时先把单位、作用域和 entry 对齐，再谈 occupancy，能避免把几个同名“大小”混成一笔账。

### 2. 双遍报告只认出货那一遍

第一遍由 `run_l2.sh` 对未修复 `kernel.ptx` 执行，报告写入 `ptxas_v.txt`，随后 `kernel.cubin` 生成第一版 SASS[scripts/run_l2.sh:74-84]。第二遍由 `gdn_smem_fix.py` 对 `kernel_smemfix.ptx` 再调用同一个 ptxas，产生实际装载的 cubin[tools/gdn_smem_fix.py:790-804,833-847]。共享声明、别名、`.maxnreg` 和宽存都在两遍之间改变文本，ptxas 因而可以合法地选择另一种调度与寄存器分配。

这也意味着 `meta.json` 中由第一遍自动采集的 registers 不能无条件贴到第二版 cubin 上。审计目录需要显式保存第二遍输出，并让它与 `kernel_smemfix.cubin` 的哈希同槽归档；若又注入 cap 或改动 PTX，还要保存新的重汇编报告。文件名相近不构成证据链，产物、报告和配置键必须一一对应。

归档中有一个代价直接的案例：第一遍报 128 寄存器，第二遍出货 cubin 却为 168、零溢出。256 线程下，双 CTA 需要 2×256×168 个寄存器，超过 65,536，ncu 因寄存器只允许单 CTA，每 SM 仅 8 个 warp；阶段性能变为预期的 0.78 倍。插入 `.maxnreg 128` 后，2×256×128 恰好装满寄存器文件，双 CTA 恢复，实测 achieved occupancy 从 15.02% 升至 29.17%[evidence/G5/bwd_report.md:2574-2600]。

同一资源报告文件还保留了另一组两次汇编对照：

```text
Used 174 registers, used 1 barriers, 47616 bytes smem
Used 128 registers, used 1 barriers, 88 bytes cumulative stack size, 47616 bytes smem
88 bytes stack frame, 84 bytes spill stores, 120 bytes spill loads
```

[evidence/G5/scratch/r19b_final_T8192/dqkwg/ptxas_smemfix_v.txt:5,10-11，按字段节选]

174→128 来自后续上限注入，同时出现 88B 栈帧与 84/120B 溢出。它能说明 cap 的代价必须和收益一起读；128/168 案例则说明第二遍自然分配也可能高于第一遍。两组数字方向不同，结论一致：报告必须绑定具体 PTX 和 cubin，出货对账读取第二遍及其后续重汇编结果。

### 3. SASS 计数检查指令形态，不冒充动态时间

`nvdisasm -c` 把 cubin 反汇编成 SASS。`run_l2.sh` 对文本中的 HMMA、LDGSTS、LDS、STS、LDL、STL、BAR.SYNC、LDSM、EX2 做静态计数，同时保存 PTX 的精确 MMA 签名[scripts/run_l2.sh:83-84,107-120]。同一份 kkt 配置记录为：

```json
"ptx_features": {"mma.sync": 16, "cp.async": 514},
"sass_ops": {"HMMA": 16, "LDGSTS": 512,
             "LDS": 83, "LDL": 0, "STL": 0,
             "BAR.SYNC": 2, "EX2": 30}
```

[evidence/G4/gen_verify/kkt_bf16_mma_p0_t64x64x32_w4/meta.json:30-33,55-64，保留相关字段]

HMMA 是张量核矩阵乘加，LDGSTS 对应全局到共享的异步搬运路径。HMMA=16 与 PTX 的 16 个 MMA 站点相符，LDGSTS=512 则说明该配置的异步复制已落实到预期指令族；LDL/STL 为零与这份第一遍报告的零溢出相互印证。TMA 配置可作为反例：它有批量张量搬运，却记录 LDGSTS=0，因为实际走的是 UTMALDG 等其他指令[evidence/G4/gen_verify/kkt_bf16_mma_tma_ws/meta.json:30-34,55-64]。

这些计数是静态指令站点数。循环次数、warp 数、谓词执行率和访存延迟都未包含，不能直接把 HMMA 或 LDGSTS 数换算成毫秒。它们适合验证“目标指令是否出现”“意外溢出是否存在”“两版代码差在哪”，动态成本仍需计时或剖析器补齐。

做版本差分时还要区分寄存器重编号与真正指令变化。可先看 opcode、立即数和控制流标签，再追踪操作数依赖；若直接按整行文本排序，大量合法重编号会淹没关键差异。反向地，差异很少也不能立即推出语义相近：一条地址立即数、屏障或谓词就足以改变整个 kernel 的访问集合。

### 4. Sanitizer 给出访问现场，racecheck 给出同步假设

compute-sanitizer 的 memcheck 用于越界、非法地址和对齐错误，racecheck 检查共享内存访问竞态。G4 的 TMA 故障被定位到 `UTMALDG.4D`，由此确认静态 tile 的 128B 对齐要求[evidence/G4/GATE_REPORT.md:131-142]；mono 曾跨 entry 合并同名 `.shared`，sanitizer 报出 0xa800 的 shared write，指向 F3 经较小符号越界写[evidence/G5/fwd_report.md:236-239]。两类报告都把抽象的数值损坏还原成具体地址与指令。

竞态检查需要同样的证据纪律。项目已有可直接置于工具下运行的 F1 脚本，其注释明确给出 `compute-sanitizer --tool racecheck python <this>` 的用法[evidence/G5/scratch/r17f/racecheck_f1ks.py:1-7]。真实日志列出 6 组读写 hazard，第一组为 kernel+0x420 的读与 +0x490 的写，汇总为 6 errors[evidence/G5/scratch/r17f/racecheck_f1ks.log:1-21]。这些 offset 可与 SASS 地址对照，进一步还原到扫描步的跨 warp 读写。

hazard 数不等于错误输出单元数，日志跑完也不证明数值正确。动态工具会改变调度，某些只在双 CTA 共驻出现的缺陷反而可能被隐藏；报告也记录了 sanitizer 串行化后缺陷不显形的情况[evidence/G5/fwd_report.md:1671-1676]。因此 sanitizer 用于提出或证伪具体访问假设，最终仍需位级输出、重复运行和驻留对照闭合。

### 5. 单条 MOV 差分把数值故障缩到清零步幅

F1 在强制单 CTA 时正确、双 CTA 时损坏。两版 SASS 的唯一差异是：

```sass
MOV R4, 0x2490   // 通过版：Tg 尺寸 9,360B
MOV R4, 0x4410   // 失败版：Tg 尺寸 17,424B
```

[evidence/G5/fwd_report.md:1812-1815]

计算指令路径相同，差别落在共享内存尺寸及随之而来的驻留状态。单条 MOV 本身还没有指出清零错误，它提供的是高价值收缩：无需再审查两套不同的数学实现，应转而测量共驻如何改变未初始化字节。接着转储 Tb/Tg，坏单元精确落在 `zero_up` 漏写的位置：清零循环以 32×8=256 为步幅，却只有 128 个线程，导致每 tile 768 个应清零单元从未写入。单 CTA 的 393,216/393,216 个受查单元恰好读到零，双 CTA 时同一地址含活跃旧数据[evidence/G5/fwd_report.md:1798-1824]。当前生成器把修复后的步幅选择写得很直白：

```python
# 旧路径 32*8；GDN_ZFIX 使用 nthr，覆盖全部 cell。
_zfix_stride = nthr if os.environ.get("GDN_ZFIX") else 32 * 8
if not _zf:
    if not _np0:
        emit_cell_while(e, C * C, _zfix_stride, zero_up)
```

[tools/gdn_full_gen.py:1376-1385，注释压缩]

差分先排除一套不同的数学代码，tile 转储再指出具体写集漏洞，最后以步幅改为线程数、双 CTA 下十次逐位一致收尾[evidence/G5/fwd_report.md:1874-1885]。很多错误值只有 1e-14 量级，容差比较会把它们视为通过；位级差分在这里承担的是正确性检查，而非更苛刻的性能偏好。

### 6. 【通用补充】mlir-reduce 与 pass 二分

自动缩例适合缺陷已能稳定复现、但 IR 很大的情况。首先写一个 interestingness 脚本：只有原始故障签名仍存在时返回 0，其他失败、环境错误和“文件无法解析”都返回非 0。随后对故障发生层的输入运行：

```bash
mlir-reduce input.mlir --test=interesting.sh -o reduced.mlir
```

缩例过程中要保存原始 IR、工具版本和完整 pipeline，并反复确认故障没有变成另一种错误。若问题只在某个 pass 后出现，先对 pipeline 前缀做二分：用前 N 个 pass 分别产生 good/bad 输出，找到首个使输出由好变坏的 pass，再对该 pass 的输入缩例。二分时固定输入、选项、线程数与随机种子，good 与 bad 的判定也要使用同一判定器（oracle）；否则首个坏前缀可能只是环境波动的位置。崩溃问题可先保存 crash reproducer；数值问题则要把参考比较嵌入 interestingness，不能只检查编译器是否退出成功。

本仓现有 `scripts/`、`tools/` 没有把 `mlir-reduce` 或自动 pass bisect 接入这条调试链；项目实际使用的是生成器删除探针、手工配置对照、SASS 差分和真机验证。删除探针也明确标注 timing-only，输出会损坏，不可用于数值验收[tools/gdn_full_gen.py:1842-1864]。自动缩例是补充手段，不能替代这里的位级与机器证据。
