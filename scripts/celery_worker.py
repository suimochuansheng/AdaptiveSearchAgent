#!/usr/bin/env python3
"""Celery Worker 启动脚本。

用法：
    poetry run python scripts/celery_worker.py

通过此脚本启动 Celery Worker 进程，消费 src.tasks 中定义的异步任务。
"""

from __future__ import annotations

from src.tasks import celery_app

if __name__ == "__main__":
    celery_app.worker_main(argv=["worker", "--loglevel=info", "--concurrency=2"])
