#!/usr/bin/env bash
# 启动会话式 Agent 的 Celery Worker（阶段 C1）。
#
# 用法：
#   ./scripts/run_worker.sh
#   ENABLE_ASYNC_TASK=true ./scripts/run_worker.sh   # 后端进程也要开同一个开关
#
# 没有 Worker 时：/tasks/conversation 仍会返回 202（任务入队成功），
# 但任务会一直停在 queued——这是"有队列没人干"的典型现象。
set -euo pipefail

cd "$(dirname "$0")/.."
eval "$(poetry env activate)"

exec celery -A src.tasks_conversation.celery_app worker \
  --loglevel=info \
  --concurrency="${WORKER_CONCURRENCY:-1}"
