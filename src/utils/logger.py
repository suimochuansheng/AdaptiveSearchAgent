# src/utils/logger.py
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from pythonjsonlogger import jsonlogger

# 全局 logger 实例
# 目标：配置 JSON 格式的 logging，并提供 log_node 装饰器，自动记录节点耗时、节点名、task_id。
_logger = None


def get_logger():
    global _logger
    if _logger is None:
        _logger = logging.getLogger("adaptive_agent")
        _logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(node_name)s %(duration_ms)s %(task_id)s %(iteration)s"
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
    return _logger


def log_node(node_name: str):
    """装饰器：记录节点执行耗时和状态"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(state: dict[str, Any], *args, **kwargs):
            logger = get_logger()
            start = time.perf_counter()
            task_id = state.get("task_id", "unknown")
            iteration = state.get("iteration", 0)
            try:
                result = await func(state, *args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    "Node finished",
                    extra={
                        "node_name": node_name,
                        "duration_ms": round(duration_ms, 2),
                        "task_id": task_id,
                        "iteration": iteration,
                    },
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    f"Node failed: {e}",
                    extra={
                        "node_name": node_name,
                        "duration_ms": round(duration_ms, 2),
                        "task_id": task_id,
                        "iteration": iteration,
                        "error": str(e),
                    },
                )
                raise

        return wrapper

    return decorator
