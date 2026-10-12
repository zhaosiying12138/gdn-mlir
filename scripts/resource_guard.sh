#!/usr/bin/env bash
# GDN-MLIR 资源守卫 —— 铁律 §0.5 的机械执行（所有构建/实验必须经过这里）
# 用法:
#   source scripts/resource_guard.sh
#   rg_gate_heavy            # 申请全局唯一重构建槽位（trap 自动释放）
#   J=$(rg_jobs)             # 按当前空闲内存计算并行度（clamp(空闲GB/2.5, 4, 12)）
#   rg_watchdog_start $PID   # 对 PID 施加 <6G STOP / >=8G CONT / <3G TERM 保护
set -u

RG_ROOT="$HOME/codebase/gdn-mlir"
RG_LOGDIR="$RG_ROOT/logs"; mkdir -p "$RG_LOGDIR"
RG_SEM="/tmp/gdnmlir_heavy.lock"   # tmp 有意：崩溃即自动清空，天然可再生

rg_avail_gb() { free -g | awk '/^Mem:/{print $7}'; }

rg_jobs() {
  local a j
  a=$(rg_avail_gb)
  j=$(( a * 10 / 25 ))     # 每 job 预算 2.5GB
  (( j < 4 ))  && j=4
  (( j > 12 )) && j=12     # 铁律上限
  echo "$j"
}

rg_gate_heavy() {
  while ! mkdir "$RG_SEM" 2>/dev/null; do
    echo "[rg] $(date +%T) 重构建槽位占用中，等 60s..."
    sleep 60
  done
  trap 'rmdir "$RG_SEM" 2>/dev/null' EXIT
  echo "[rg] $(date +%T) 获得重构建槽位"
}

rg_watchdog_start() {
  local pid="${1:?need pid}"
  (
    while kill -0 "$pid" 2>/dev/null; do
      sleep 30
      a=$(free -g | awk '/^Mem:/{print $7}')
      st=$(awk '/^State:/{print $2}' "/proc/$pid/status" 2>/dev/null || echo X)
      echo "$(date +%F_%T) pid=$pid avail=${a}G state=$st" >> "$RG_LOGDIR/watchdog.log"
      if (( a < 6 )) && [ "$st" != "T" ]; then
        kill -STOP "$pid" 2>/dev/null && \
          echo "$(date +%F_%T) *** STOP $pid (avail<6G)" >> "$RG_LOGDIR/watchdog.log"
      elif (( a >= 8 )) && [ "$st" = "T" ]; then
        kill -CONT "$pid" 2>/dev/null && \
          echo "$(date +%F_%T) *** CONT $pid (avail>=8G)" >> "$RG_LOGDIR/watchdog.log"
      fi
      if (( a < 3 )); then
        echo "$(date +%F_%T) *** TERM $pid (avail<3G，ninja 增量续编无损)" >> "$RG_LOGDIR/watchdog.log"
        kill -TERM "$pid" 2>/dev/null
        exit 9
      fi
    done
  ) &
  echo "[rg] watchdog 已启动 pid=$pid (watcher=$!)"
}
