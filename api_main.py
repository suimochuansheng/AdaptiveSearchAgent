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

from pathlib import Path

from dotenv import load_dotenv

# 将 .env 注入 os.environ（langfuse SDK 从环境变量读取配置）
_ENV_FILE = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)

import asyncio
import json
import logging

logger = logging.getLogger(__name__)
from contextlib import asynccontextmanager
from datetime import UTC
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import BaseModel
from redis import asyncio as aioredis

from config import settings

# Sentry 崩溃自动捕获（仅当配置了 DSN 时启用）
import sentry_sdk

if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        traces_sample_rate=1.0,
        environment=settings.ENV,
    )
    logger.info("✅ Sentry 已启用 (environment=%s)", settings.ENV)

# Langfuse 可观测性（仅当启用且密钥非空时初始化）
if settings.LANGFUSE_ENABLED and settings.LANGFUSE_PUBLIC_KEY:
    from langfuse.langchain import CallbackHandler

    langfuse_handler = CallbackHandler()
    logger.info("✅ Langfuse 追踪已启用 (host=%s)", settings.LANGFUSE_HOST)
else:
    langfuse_handler = None
    logger.warning("⚠️ Langfuse 未启用 (检查 LANGFUSE_ENABLED 和 LANGFUSE_PUBLIC_KEY)")

# 导入公共图工厂和状态
from src import checkpointer  # 模块级引用，供测试 mock 及新增端点使用
from src.checkpointer import close_checkpointer, init_checkpointer
from src.graph_factory import get_graph

if TYPE_CHECKING:
    from src.state import AgentState

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
    from src import checkpointer

    assert checkpointer._global_pool is not None, "数据库连接池未初始化"
    async with checkpointer._global_pool.connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS task_states (
                thread_id TEXT PRIMARY KEY,
                state TEXT NOT NULL DEFAULT 'idle',
                interrupt_time TIMESTAMP
            )
        """)


async def get_task_state(thread_id: str) -> str:
    """查询任务状态：idle, running, interrupted, completed, failed"""
    from src import checkpointer

    # assert 强制检查
    assert checkpointer._global_pool is not None, "数据库连接池未初始化"
    async with checkpointer._global_pool.connection() as conn:
        # 查这个task_states表看这个thread_id的状态是什么，如果没有记录就默认idle
        result = await conn.execute(
            "SELECT state FROM task_states WHERE thread_id = %s", (thread_id,)
        )
        # fetchone() = 只拿第一条记录，返回一个 dict（psycopg 的 dict_row row_factory），如果没有记录则返回 None
        row = await result.fetchone()
    # cast(str, ...) 告诉 mypy “相信我，这个值一定是 str 类型”
    # 返回对应session_id的状态，如果没有记录就返回默认状态 "idle"
    return cast("str", row["state"]) if row else "idle"


async def set_task_state(thread_id: str, state: str, interrupt_time=None):
    """更新任务状态，可选记录中断时间（用于超时清理）"""
    from src import checkpointer

    assert checkpointer._global_pool is not None, "数据库连接池未初始化"
    async with checkpointer._global_pool.connection() as conn:
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
            from src import checkpointer

            if checkpointer._global_pool is None:
                continue
            async with checkpointer._global_pool.connection() as conn:
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

# ---------- CORS 中间件（允许前端 file:// 或 localhost 跨域访问）----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段允许所有来源；生产环境应限定域名
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, OPTIONS 等全部放开
    allow_headers=["*"],  # Content-Type, Authorization 等全部放开
)

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
    from datetime import datetime

    from src.utils.llm_utils import get_model_switch_count, reset_model_switch_count

    # 【首次请求】重置模型切换计数器，避免历史会话数据干扰
    if resume_value is None:
        reset_model_switch_count()

    # 配置 LangGraph 会话ID + 使用的模型 + Langfuse 回调
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id, "llm_provider": "ollama"},
    }
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    # ====================== 核心：分布式锁 ======================
    # 为每个 thread_id 创建 Redis 分布式锁，防止同一会话并发执行
    lock = redis_client.lock(f"agent_lock:{thread_id}", timeout=600)
    # 非阻塞获取锁：获取不到直接返回，不等待
    acquired = await lock.acquire(blocking=False)

    # 锁获取失败 → 分情况处理
    if not acquired:
        status = await get_task_state(thread_id)
        if status == "interrupted":
            # 恢复请求：允许通过（锁由上次中断请求持有，会在 except 中释放）
            pass
        elif status in ("completed", "failed", "idle"):
            # 僵尸锁：任务已完成但 Redis 锁残留（服务重启/持久化导致）
            logger.warning("检测到僵尸锁(thread_id=%s, status=%s)，强制清理", thread_id, status)
            await redis_client.delete(f"agent_lock:{thread_id}")
            acquired = await lock.acquire(blocking=False)
            if not acquired:
                raise HTTPException(429, "该会话正忙，清理僵尸锁后仍无法获取，请稍后再试")
        else:
            # running 状态 → 真正在执行中
            raise HTTPException(429, "该会话正忙，请稍后再试")

    # 记录任务开始时间，用于计算耗时
    start_time = time_module.perf_counter()
    final_state = None  # 保存图执行的最终状态

    # 去重跟踪变量：仅在指标变化时才推送 SSE 事件，避免重复输出
    _last_conf: float = -1.0  # 上次已发送的置信度（-1 确保首次必定发送）
    _last_iter: int = -1  # 上次已发送的迭代轮次
    _last_result_count: int = 0  # 上次已发送的搜索结果数量

    try:
        # 标记任务为运行中
        await set_task_state(thread_id, "running")
        # 获取编译好的 LangGraph 执行图
        graph = await get_graph()

        # ====================== 分支1：恢复中断的任务 ======================
        if resume_value is not None:
            # 从 checkpoint 中恢复会话，继续执行
            async for event in graph.astream(
                Command(
                    resume=resume_value
                ),  # 恢复指令-从 human_approval 节点的 interrupt() 处恢复
                config=config,
                stream_mode="values",  # 以状态值模式流式返回
            ):
                final_state = event  # 更新最新状态

                # ── 中间状态：仅在置信度/轮次变化时推送，避免重复 ──
                cur_conf = final_state.get("confidence_score", 0.0)
                cur_iter = final_state.get("iteration", 0)
                if cur_conf != _last_conf or cur_iter != _last_iter:
                    _last_conf = cur_conf
                    _last_iter = cur_iter
                    yield {
                        "type": "status",
                        "confidence": cur_conf,
                        "iteration": cur_iter,
                        "missing_info": final_state.get("missing_info", ""),
                    }

                # ── 搜索结果：仅在有新增时推送摘要 ──
                results = final_state.get("search_results", [])
                cur_count = len(results)
                if cur_count > _last_result_count:
                    new_results = results[_last_result_count:]
                    _last_result_count = cur_count
                    logger.info(
                        "📤 [resume] 发送 search_result 事件: count=%d, new=%d",
                        cur_count,
                        len(new_results),
                    )
                    for r in new_results:
                        kw = r.get("keyword", "")
                        ct = r.get("content", "")
                        logger.info(
                            "  → keyword=%s content_len=%d",
                            kw,
                            len(ct),
                        )
                        yield {
                            "type": "search_result",
                            "keyword": kw,
                            "content": ct[:300],
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

                # ── 中间状态：仅在置信度/轮次变化时推送，避免重复 ──
                cur_conf = final_state.get("confidence_score", 0.0)
                cur_iter = final_state.get("iteration", 0)
                if cur_conf != _last_conf or cur_iter != _last_iter:
                    _last_conf = cur_conf
                    _last_iter = cur_iter
                    yield {
                        "type": "status",
                        "confidence": cur_conf,
                        "iteration": cur_iter,
                        "missing_info": final_state.get("missing_info", ""),
                    }

                # ── 搜索结果：仅在有新增时推送摘要 ──
                results = final_state.get("search_results", [])
                cur_count = len(results)
                if cur_count > _last_result_count:
                    new_results = results[_last_result_count:]
                    _last_result_count = cur_count
                    logger.info(
                        "📤 发送 search_result 事件: count=%d, new=%d",
                        cur_count,
                        len(new_results),
                    )
                    for r in new_results:
                        kw = r.get("keyword", "")
                        ct = r.get("content", "")
                        logger.info(
                            "  → keyword=%s content_len=%d",
                            kw,
                            len(ct),
                        )
                        yield {
                            "type": "search_result",
                            "keyword": kw,
                            "content": ct[:300],
                        }

        # ====================== 中断检测 ==============================
        # astream(stream_mode="values") 遇到 interrupt() 时不抛异常，
        # 而是把中断状态作为最后一个 event 正常返回。此处主动检测并触发。
        if final_state and "__interrupt__" in final_state:
            interrupt_payload = final_state["__interrupt__"]
            # __interrupt__ 是 tuple[Interrupt]，提取 .value
            if hasattr(interrupt_payload, "__iter__"):
                interrupt_payload = interrupt_payload[0]  # type: ignore[index]
            interrupt_value = getattr(interrupt_payload, "value", interrupt_payload)
            logger.info(
                "检测到中断状态(confidence=%.2f)，触发 GraphInterrupt",
                final_state.get("confidence_score", 0.0),
            )
            await set_task_state(thread_id, "interrupted", interrupt_time=datetime.now(UTC))
            raise GraphInterrupt(interrupt_value)

        # ====================== 任务正常完成：先 KPI，再最终报告 ======================
        elapsed = time_module.perf_counter() - start_time
        if final_state:
            # 诊断日志：检查 final_report 是否存在
            report = final_state.get("final_report", "")
            logger.info(
                "Agent 执行完成: elapsed=%.1fs, confidence=%.2f, final_report_len=%d, search_results=%d",
                elapsed,
                final_state.get("confidence_score", 0.0),
                len(report),
                len(final_state.get("search_results", [])),
            )
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
            # KPI 必须在 final 之前发送，确保前端在停止读取前收到
            yield {"type": "kpi", "data": kpi_data}

            # 最终报告最后发送（前端收到后即结束流）
            if report:
                yield {"type": "final", "content": report}

        # 更新任务状态：已完成
        await set_task_state(thread_id, "completed")
        # 释放 Redis 锁（仅当本请求获取了锁时）
        if acquired:
            await lock.release()

    # ====================== 异常处理 ======================
    # 客户端主动断开 → 释放锁并静默退出
    except GeneratorExit:
        if acquired:
            await lock.release()
        raise

    # 图被主动中断 → 释放锁让恢复请求可以重新获取
    except GraphInterrupt:
        await set_task_state(thread_id, "interrupted", interrupt_time=datetime.now(UTC))
        if acquired:
            await lock.release()
        raise  # 抛出给上层SSE处理

    # 其他异常 → 标记失败，强制释放锁
    except Exception:
        await set_task_state(thread_id, "failed")
        if acquired:
            await lock.release()
        raise


# ---------- SSE 流式端点（核心）----------
# 注册POST请求接口：前端通过 /chat/stream 地址调用该函数进行流式对话
@app.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest):
    """
    流式对话端点。
    - 首次请求：传入 message 和 thread_id，开始执行图。
    - 恢复请求：传入 resume_value 和 thread_id，恢复被中断的图。
    返回 Server-Sent Events (SSE) 流，事件类型：token, interrupt, error

    args:
    - request: FastAPI 内置的功能，专门用来检测前端 / 客户端是否断开连接
    - body: ChatRequest 对象，包含 thread_id, message, resume_value
    """
    # 从请求体JSON中提取三个核心参数
    thread_id = body.thread_id  # 会话唯一ID，用于区分不同用户/对话
    user_input = body.message  # 用户输入的问题内容（首次请求必填）
    resume_value = body.resume_value  # 恢复中断会话的指令（yes/no，仅恢复时传）

    # ===================== 会话状态校验 =====================
    # 获取当前thread_id对应的任务状态：idle(空闲) / running(执行中) / interrupted(已中断)
    status = await get_task_state(thread_id)

    # 校验规则（自上而下优先级）：
    # 1. idle/completed/failed → 直接放行
    # 2. interrupted + 无 resume_value → 拒绝（必须走恢复流程）
    # 3. running + 无 resume_value → 检查 Redis 锁，僵尸则清理，否则拒绝
    if status == "idle":
        pass  # 正常放行
    elif status == "interrupted" and resume_value is None:
        raise HTTPException(429, "会话已中断，请提供 resume_value 恢复或刷新页面")
    elif status == "running" and resume_value is None:
        # 用户发了新消息 → 检查 Redis 锁是否残留（服务器重启后锁可能已消失）
        lock_key = f"agent_lock:{thread_id}"
        lock_exists = await redis_client.exists(lock_key)
        if not lock_exists:
            # 僵尸 running 状态：Redis 锁已消失但 task_states 未清理 → 自动修复
            logger.warning("检测到僵尸 running 状态(thread_id=%s)，自动清理", thread_id)
            await set_task_state(thread_id, "idle")
        else:
            raise HTTPException(429, "该会话正在执行中，请等待完成后再发送新消息")

    # ===================== SSE事件生成器（核心流式逻辑） =====================
    # 定义异步生成器：持续向后端推送流式数据（打字机效果/中断/错误）
    async def event_generator():
        try:
            # 调用核心Agent执行函数，异步迭代获取每一步的事件流
            async for event in execute_agent(thread_id, user_input, resume_value):
                # 监听客户端连接状态：如果用户关闭页面/断开连接，立即停止推送
                if await request.is_disconnected():
                    break
                evt_type = event.get("type", "?")
                logger.info("📡 SSE 推送: type=%s", evt_type)
                # yield是流式推送，按照SSE协议格式返回数据：data: JSON字符串\n\n
                yield f"data: {json.dumps(event)}\n\n"

        # ===================== 捕获客户端主动断开 =====================
        except GeneratorExit:
            # 前端关闭连接或刷新页面时触发，直接清理退出
            # 【严禁在此处 yield — 会抛出 RuntimeError】
            logger.info("SSE 连接被客户端关闭 (GeneratorExit), thread_id=%s", thread_id)
            return

        # ===================== 捕获Agent执行中断（需要用户审批） =====================
        except GraphInterrupt as e:
            # 从中断异常中提取数据（通常包含需要用户确认的问题）
            interrupt_data = e.args[0] if e.args else {"question": "需要您的批准"}

            # 构造标准中断事件，返回给前端
            interrupt_event = {
                "type": "interrupt",  # 事件类型：中断
                "question": interrupt_data.get("question", "请确认"),  # 给用户的提示
                "data": interrupt_data,  # 中断的完整上下文数据
            }
            # 推送给前端
            yield f"data: {json.dumps(interrupt_event)}\n\n"

        # ===================== 捕获其他所有异常 =====================
        except Exception as e:
            # 后台打印完整错误日志，方便排查问题
            logger.exception("Agent 执行错误")

            # 构造标准错误事件，返回给前端展示
            error_event = {
                "type": "error",  # 事件类型：错误
                "message": f"服务异常：{str(e)}",  # 错误提示信息
            }
            # 推送给前端
            yield f"data: {json.dumps(error_event)}\n\n"

    # ===================== 返回SSE流式响应 =====================
    # FastAPI将异步生成器包装成流式响应，媒体类型为 text/event-stream（SSE标准）
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


# ===== 新增端点：异步任务状态查询（ENABLE_ASYNC_TASK 控制启用） =====


@app.get("/api/task/{thread_id}/status")
async def get_task_status(thread_id: str):
    """查询异步任务的执行状态。

    复用 task_states 表（由 ensure_task_states_table 创建），
    返回 thread_id 对应的执行状态和最近更新时间。

    Args:
        thread_id: 会话/任务唯一标识。

    Returns:
        {"thread_id": str, "state": str, "updated_at": str | None}
        或 404 {"detail": "Task not found"}。
    """
    assert checkpointer._global_pool is not None, "数据库连接池未初始化"

    async with checkpointer._global_pool.connection() as conn:
        result = await conn.execute(
            "SELECT state, interrupt_time FROM task_states WHERE thread_id = %s",
            (thread_id,),
        )
        row = await result.fetchone()

    if row is None:
        raise HTTPException(404, "Task not found")

    return {
        "thread_id": thread_id,
        "state": row["state"],
        "updated_at": row["interrupt_time"].isoformat() if row["interrupt_time"] else None,
    }


@app.get("/api/task/{thread_id}/result")
async def get_task_result(thread_id: str):
    """查询异步任务的最终执行结果。

    仅当任务状态为 completed 时返回 final_report；
    否则返回 202 引导前端轮询 /api/task/{thread_id}/status。

    Args:
        thread_id: 会话/任务唯一标识。

    Returns:
        completed → {"thread_id": str, "final_report": str, "total_tokens": int}
        未完成   → 202 {"message": str}
        未找到   → 404 {"detail": str}
    """
    # 1. 先查 task_states 确认状态
    assert checkpointer._global_pool is not None, "数据库连接池未初始化"

    async with checkpointer._global_pool.connection() as conn:
        result = await conn.execute(
            "SELECT state FROM task_states WHERE thread_id = %s",
            (thread_id,),
        )
        row = await result.fetchone()

    if row is None:
        raise HTTPException(404, "Task not found")

    if row["state"] != "completed":
        return JSONResponse(
            status_code=202,
            content={
                "message": (
                    "Task is still processing, " f"please check /api/task/{thread_id}/status"
                ),
            },
        )

    # 2. 从 LangGraph Checkpointer 捞取最终状态
    graph = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    final_state = await graph.aget_state(config)

    if final_state is None or not final_state.values:
        raise HTTPException(500, "已完成的任务无法读取最终状态，请联系管理员")

    values = final_state.values
    return {
        "thread_id": thread_id,
        "final_report": values.get("final_report", ""),
        "total_tokens": values.get("total_tokens", 0),
    }


# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_main:app",
        host=settings.SERVICE_HOST,
        port=settings.SERVICE_PORT,
        reload=True,
    )
