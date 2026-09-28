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
    """
    获取单例日志实例（全局唯一），使用 JSON 格式输出日志
    日志包含自定义字段：node_name、duration_ms、task_id、iteration
    """
    global _logger  # 声明使用全局变量 _logger

    # 如果全局日志实例未初始化，才进行初始化（单例模式）
    if _logger is None:
        # 创建名为 "adaptive_agent" 的日志器
        _logger = logging.getLogger("adaptive_agent")
        # 设置日志最低输出级别为 INFO
        _logger.setLevel(logging.INFO)

        # 创建控制台日志处理器（输出到终端）
        handler = logging.StreamHandler()

        # 定义 JSON 日志格式，包含标准字段 + 自定义扩展字段
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(node_name)s %(duration_ms)s %(task_id)s %(iteration)s"
        )

        # 给处理器设置 JSON 格式
        handler.setFormatter(formatter)
        # 把处理器添加到日志器
        _logger.addHandler(handler)

    # 返回已经初始化好的全局日志实例
    return _logger


def log_node(node_name: str):
    """
    节点执行日志装饰器（高阶装饰器）
    作用：自动记录异步节点函数的执行耗时、成功/失败状态、任务ID、迭代次数等信息
    :param node_name: 当前节点名称，用于日志区分不同节点
    """

    def decorator(func: Callable):
        """
        真正的装饰器函数，接收被装饰的目标函数
        :param func: 被装饰的异步节点执行函数
        """

        # 保留原函数的元信息（名称、文档字符串等），避免装饰器破坏函数特性
        @wraps(func)
        async def wrapper(state: dict[str, Any], *args, **kwargs):
            """
            包装函数：在原函数执行前后增加日志逻辑
            :param state: 状态字典，包含 task_id / iteration 等上下文信息
            :param args: 额外位置参数
            :param kwargs: 额外关键字参数
            """
            # 获取全局单例日志实例
            logger = get_logger()

            # 记录函数开始执行的高精度时间（用于计算耗时）
            start = time.perf_counter()

            # 从状态字典中获取任务ID，不存在则默认 unknown
            task_id = state.get("task_id", "unknown")
            # 从状态字典中获取迭代次数，不存在则默认 0
            iteration = state.get("iteration", 0)

            try:
                # 执行被装饰的原异步函数
                result = await func(state, *args, **kwargs)

                # 计算执行耗时，转换为毫秒并保留 2 位小数
                duration_ms = (time.perf_counter() - start) * 1000

                # 打印节点执行成功日志，携带自定义扩展字段
                logger.info(
                    "Node finished",
                    extra={
                        "node_name": node_name,  # 节点名称
                        "duration_ms": round(duration_ms, 2),  # 执行耗时(ms)
                        "task_id": task_id,  # 任务ID
                        "iteration": iteration,  # 迭代次数
                    },
                )

                # 返回原函数的执行结果
                return result

            except Exception as e:
                # 捕获原函数执行过程中的所有异常
                # 计算失败耗时
                duration_ms = (time.perf_counter() - start) * 1000

                # 打印节点执行失败日志，包含错误信息
                logger.error(
                    f"Node failed: {e}",
                    extra={
                        "node_name": node_name,
                        "duration_ms": round(duration_ms, 2),
                        "task_id": task_id,
                        "iteration": iteration,
                        "error": str(e),  # 异常信息
                    },
                )

                # 抛出原始异常，不中断程序的异常处理流程
                raise

        # 返回包装后的异步函数
        return wrapper

    # 返回装饰器
    return decorator
