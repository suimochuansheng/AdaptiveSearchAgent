#!/usr/bin/env bash
# 启动会话式 Agent 的 Celery Worker（阶段 C1/C2）。
#
# 用法：
#   ./scripts/run_worker.sh
#   ENABLE_ASYNC_TASK=true WORKER_POOL=prefork ./scripts/run_worker.sh
#
# 进程池说明（WORKER_POOL，默认 solo）：
#   - solo：单进程，兼容性最好。本机（WSL2 + 沙箱）用 prefork 会因 billiard
#     创建信号量报 PermissionError，所以默认用 solo。
#   - prefork：多进程，吞吐高，普通 Linux 服务器上可用；需要配 WORKER_CONCURRENCY。
#
# 没有 Worker 时：/tasks/conversation 仍会返回 202（入队成功），
# 但任务会一直停在 queued——"有队列没人干"。
set -euo pipefail

cd "$(dirname "$0")/.."
eval "$(poetry env activate)"

POOL="${WORKER_POOL:-solo}"
CONCURRENCY="${WORKER_CONCURRENCY:-1}"

if [ "$POOL" = "solo" ]; then
  exec celery -A src.tasks_conversation.celery_app worker \
    --loglevel=info \
    --pool=solo
else
  exec celery -A src.tasks_conversation.celery_app worker \
    --loglevel=info \
    --pool="$POOL" \
    --concurrency="$CONCURRENCY"
fi
