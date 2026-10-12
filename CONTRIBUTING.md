# 资源纪律铁律（最高优先级，永久生效）

背景：本机 WSL 曾因内存 OOM 整机崩溃两次（47GB RAM 与训练会话共享；FSDP2 级任务内存运行时爬升，启动余量不可信；/tmp 是 tmpfs 随崩溃清空）。**任何构建/实验/agent 任务不得例外：**

1. **线程预算按实测空闲内存动态封顶**：任何 ninja/make/并行编译启动前 `free -g`；jobs = clamp(空闲GB ÷ 2.5, 4, 12)；绝不用满 24 核。实现见 `scripts/resource_guard.sh`（`rg_jobs`）。
2. **watchdog 强制开启**（`rg_watchdog_start`，每 30s 采样）：空闲 <6G 冻结（SIGSTOP）己方构建，≥8G 恢复（SIGCONT），<3G 终止（SIGTERM，ninja 可续编无损恢复）。
3. **全局同一时刻只允许一个重构建**：`rg_gate_heavy` 信号量；多 agent 并行时 CPU 预算由主会话统一分配，agent 不得各自开满。
4. **实验进程显式限线程**：torch/OMP/MKL `num_threads` 必须显式设置；Z3 单线程+rlimit；Triton par_compile/批量 nvcc 限池大小。
5. **一切幂等可断点**：长任务分 stage + 内容哈希戳（已完成→跳过）；恢复 = 重跑驱动脚本。
6. **持久物只落盘不落 /tmp**：产物/证据一律在本仓（ext4）；gate 产物即时 git commit；内存中不留唯一副本。
7. **崩溃恢复**：WSL 重启后 `cd ~/codebase/gdn-mlir && make status` → 按 `RECOVERY.md` 续跑在进行的 stage。
8. **GPU 纪律并行生效**：遵守 `~/codebase/GPU-COORDINATION.md`——启动 GPU 任务前查 `nvidia-smi --query-compute-apps`，登记窗口，**绝不 kill 对方进程**；正确性短跑可共享（<2GB 显存），计时只在空闲窗口。
