"""
AdaptiveSearchAgent 的 FastAPI 入口模块

提供生产级流式 API，支持：
- 多会话隔离（thread_id）
- 分布式锁（Redis）防止并发冲突
- 任务状态管理（PostgreSQL）
- Human-in-the-Loop 中断恢复
- SSE 流式输出
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import BaseModel
from redis import asyncio as aioredis

from config import settings

# 导入公共图工厂和状态
from src.checkpointer import close_checkpointer, init_checkpointer
from src.graph_factory import get_graph

if TYPE_CHECKING:
    from src.state import AgentState

# ---------- 日志配置 ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 用于跟踪后台任务，以便在 shutdown 时优雅取消
cleanup_task = None

# ---------- Redis 客户端（用于分布式锁）----------
redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)


# ---------- Pydantic 请求模型（自动校验请求体）----------
class ChatRequest(BaseModel):
    thread_id: str  # 会话唯一标识
    message: str | None = None  # 首次请求时传入的用户消息
    resume_value: Any | None = None  # 恢复时传入的用户审批结果（如 "yes"/"no"）


# ---------- PostgreSQL 任务状态表管理 ----------
async def ensure_task_states_table():
    """确保 task_states 表存在，用于记录每个 thread_id 的执行状态"""
    from src.checkpointer import _global_pool

    assert _global_pool is not None, "数据库连接池未初始化"
    async with _global_pool.connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS task_states (
                thread_id TEXT PRIMARY KEY,
                state TEXT NOT NULL DEFAULT 'idle',
                interrupt_time TIMESTAMP
            )
        """)


async def get_task_state(thread_id: str) -> str:
    """查询任务状态：idle, running, interrupted, completed, failed"""
    from src.checkpointer import _global_pool

    assert _global_pool is not None, "数据库连接池未初始化"
    async with _global_pool.connection() as conn:
        result = await conn.execute(
            "SELECT state FROM task_states WHERE thread_id = %s", (thread_id,)
        )
        row = await result.fetchone()
        # cast(str, ...) 告诉 mypy “相信我，这个值一定是 str 类型”
    return cast(str, row["state"]) if row else "idle"


async def set_task_state(thread_id: str, state: str, interrupt_time=None):
    """更新任务状态，可选记录中断时间（用于超时清理）"""
    from src.checkpointer import _global_pool

    assert _global_pool is not None, "数据库连接池未初始化"
    async with _global_pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO task_states (thread_id, state, interrupt_time)
            VALUES (%s, %s, %s)
            ON CONFLICT (thread_id) DO UPDATE
            SET state = %s, interrupt_time = %s
        """,
            (thread_id, state, interrupt_time, state, interrupt_time),
        )


# ---------- 后台清理任务：清理超时中断的会话 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------- 启动阶段 ----------
    # 1. 初始化数据库连接和任务表
    await init_checkpointer()
    await ensure_task_states_table()

    # 2. 启动后台清理任务
    async def clean_interrupted_tasks():
        while True:
            await asyncio.sleep(60)  # 每分钟执行一次
            from src.checkpointer import _global_pool

            if _global_pool is None:
                continue
            async with _global_pool.connection() as conn:
                result = await conn.execute("""
                    SELECT thread_id FROM task_states
                    WHERE state = 'interrupted'
                    AND interrupt_time < NOW() - INTERVAL '10 minutes'
                """)
                rows = await result.fetchall()
            for row in rows:
                thread_id = row["thread_id"]
                await set_task_state(thread_id, "failed")
                # 释放可能残留的 Redis 锁
                lock = redis_client.lock(f"agent_lock:{thread_id}")
                if await lock.locked():
                    await lock.release()
                logger.info(f"清理超时中断会话: {thread_id}")

    global cleanup_task
    cleanup_task = asyncio.create_task(clean_interrupted_tasks())
    logger.info("后台清理任务已启动")
    logger.info("FastAPI 启动完成")

    yield  #  FastAPI 应用启动时从开头执行到这里，关闭时从这里继续执行到末尾

    # ---------- 关闭阶段 ----------
    # 1. 取消后台清理任务
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            logger.info("后台清理任务已取消")

    # 2. 关闭数据库和 Redis 连接
    await close_checkpointer()
    await redis_client.close()
    logger.info("FastAPI 关闭")


app = FastAPI(lifespan=lifespan)

# ---------- 统一 Agent 执行器（带分布式锁和中断处理）----------
# api_main.py 中修正后的执行器


# ruff: noqa: C901    先保证功能正确，后续再重构这个函数以降低复杂度
async def execute_agent(thread_id: str, user_input: str | None = None, resume_value: Any = None):
    """
    执行 LangGraph 图，产出 SSE 事件流。
    - 首次请求：使用 user_input 构建完整 AgentState。
    - 恢复请求：使用 Command(resume=...) 恢复被中断的图。
    产出事件类型：status（中间状态）、token（流式文本）、final（最终报告）、kpi、interrupt、error
    """
    import time as time_module

    from src.utils.llm_utils import get_model_switch_count, reset_model_switch_count

    # 【首次请求】重置模型切换计数器，避免历史会话数据干扰
    if resume_value is None:
        reset_model_switch_count()

    # 配置 LangGraph 会话ID + 使用的模型--我们有模型切换
    config = {"configurable": {"thread_id": thread_id, "llm_provider": "ollama"}}

    # ====================== 核心：分布式锁 ======================
    # 为每个 thread_id 创建 Redis 分布式锁，防止同一会话并发执行
    lock = redis_client.lock(f"agent_lock:{thread_id}", timeout=600)
    # 非阻塞获取锁：获取不到直接返回，不等待
    acquired = await lock.acquire(blocking=False)

    # 锁获取失败 → 检查状态：只有 interrupted 状态允许恢复，否则直接报错
    if not acquired:
        status = await get_task_state(thread_id)
        if status != "interrupted":
            raise HTTPException(429, "该会话正忙，请稍后再试")

    # 记录任务开始时间，用于计算耗时
    start_time = time_module.perf_counter()
    final_state = None  # 保存图执行的最终状态

    try:
        # 获取编译好的 LangGraph 执行图
        graph = await get_graph()

        # ====================== 分支1：恢复中断的任务 ======================
        if resume_value is not None:
            # 从 checkpoint 中恢复会话，继续执行
            async for event in graph.astream(
                Command(resume=resume_value),  # 恢复指令
                config=config,
                stream_mode="values",  # 以状态值模式流式返回
            ):
                final_state = event  # 更新最新状态

                # 有最终报告 → 推送 final 事件
                if final_state.get("final_report"):
                    yield {"type": "final", "content": final_state["final_report"]}

                # 有置信度等中间状态 → 推送 status 事件
                if "confidence_score" in final_state:
                    yield {
                        "type": "status",
                        "confidence": final_state["confidence_score"],
                        "iteration": final_state.get("iteration", 0),
                        "missing_info": final_state.get("missing_info", ""),
                    }

        # ====================== 分支2：首次执行新任务 ======================
        else:
            # 首次执行必须有 user_input
            assert user_input is not None, "首次执行必须提供 message"
            # 初始化 Agent 状态（全新会话）
            initial_state: AgentState = {
                "user_query": user_input,
                "plan": [],
                "search_results": [],
                "confidence_score": 0.0,
                "missing_info": "",
                "retry_keywords": [],
                "iteration": 0,
                "final_report": "",
                "human_approved": False,
                "task_id": thread_id,
                "thread_id": thread_id,
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "current_llm": "ollama",
                "pending_keywords": [],
                "_batch_keywords": [],
            }

            # 流式执行图
            async for event in graph.astream(initial_state, config=config, stream_mode="values"):
                final_state = event

                # 返回最终报告
                if final_state.get("final_report"):
                    yield {"type": "final", "content": final_state["final_report"]}

                # 返回中间执行状态
                if "confidence_score" in final_state:
                    yield {
                        "type": "status",
                        "confidence": final_state["confidence_score"],
                        "iteration": final_state.get("iteration", 0),
                        "missing_info": final_state.get("missing_info", ""),
                    }

        # ====================== 任务正常完成：统计KPI ======================
        elapsed = time_module.perf_counter() - start_time
        if final_state:
            kpi_data = {
                "total_tokens": final_state.get("total_tokens", 0),
                "input_tokens": final_state.get("input_tokens", 0),
                "output_tokens": final_state.get("output_tokens", 0),
                "current_llm": final_state.get("current_llm", "ollama"),
                "model_switches": get_model_switch_count(),  # 模型切换次数
                "thread_id": thread_id,
                "elapsed_seconds": elapsed,  # 执行耗时
                "human_approved": final_state.get("human_approved"),
                "confidence_score": final_state.get("confidence_score", 0.0),
            }
            # 向前端返回KPI数据
            yield {"type": "kpi", "data": kpi_data}

        # 更新任务状态：已完成
        await set_task_state(thread_id, "completed")
        # 释放 Redis 锁（必须执行，否则锁会一直占用直到超时）
        await lock.release()

    # ====================== 异常处理 ======================
    # 图被主动中断 → 标记状态为 interrupted，不释放锁（留给恢复任务用）
    except GraphInterrupt:
        await set_task_state(thread_id, "interrupted", interrupt_time=time_module.perf_counter())
        raise  # 抛出给上层SSE处理

    # 其他异常 → 标记失败，强制释放锁
    except Exception:
        await set_task_state(thread_id, "failed")
        await lock.release()
        raise


# ---------- SSE 流式端点（核心）----------
@app.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest):
    """
    流式对话端点。
    - 首次请求：传入 message 和 thread_id，开始执行图。
    - 恢复请求：传入 resume_value 和 thread_id，恢复被中断的图。
    返回 Server-Sent Events (SSE) 流，事件类型：token, interrupt, error
    """
    thread_id = body.thread_id
    user_input = body.message
    resume_value = body.resume_value

    # 检查当前状态：如果会话正在运行或被中断且未携带恢复值，拒绝
    status = await get_task_state(thread_id)
    if status in ("running", "interrupted") and resume_value is None:
        raise HTTPException(429, "会话正在执行或被中断，请等待或提供 resume_value 恢复")

    async def event_generator():
        try:
            async for event in execute_agent(thread_id, user_input, resume_value):
                # 客户端断开连接时停止生成
                if await request.is_disconnected():
                    break
                yield f"data: {json.dumps(event)}\n\n"
        except GraphInterrupt as e:
            # 捕获中断异常，从中提取中断数据（例如审批问题）
            interrupt_data = e.args[0] if e.args else {"question": "需要您的批准"}
            interrupt_event = {
                "type": "interrupt",
                "question": interrupt_data.get("question", "请确认"),
                "data": interrupt_data,
            }
            yield f"data: {json.dumps(interrupt_event)}\n\n"
        except Exception as e:
            logger.exception("Agent 执行错误")
            error_event = {
                "type": "error",
                "message": f"服务异常：{str(e)}",
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------- 状态查询端点 ----------
@app.get("/status/{thread_id}")
async def get_status(thread_id: str):
    """查询会话当前状态（用于前端重连时判断）"""
    try:
        state = await get_task_state(thread_id)
        return {"status": state}
    except Exception:
        logger.exception("获取状态失败")
        raise HTTPException(500, "无法获取会话状态") from None


# ---------- 健康检查 ----------
@app.get("/health")
async def health_check():
    """简单健康检查，确认服务存活"""
    return {"status": "ok", "message": "FastAPI + LangGraph 集成运行中"}
