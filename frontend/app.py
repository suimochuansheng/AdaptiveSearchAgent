"""
Chainlit 前端入口 — 自适应搜索助手。

职责：
- 用户身份识别（AskUserMessage）
- 会话中断恢复（行内 AskActionMessage，弱感知）
- SSE 流式消费（6 种事件类型）
- 多次中断审批循环（Human-in-the-Loop）
"""

import json
import logging
import os
from contextlib import suppress
from typing import Any

import chainlit as cl
import httpx
from dotenv import load_dotenv

# ---------------------------测试用的私有化函数允许外部调用---------------------------
__all__ = ["_stream_agent_with_interrupt"]

# ── 环境 & 配置 ──────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 开发开关 ────────────────────────────────────────────────
DEVELOPER_MODE = True  # True=跳过名字输入+锁定 thread_id / False=正式多用户模式

# ── 会话键名常量 ─────────────────────────────────────────────
SESSION_THREAD_ID = "thread_id"
SESSION_USER_NAME = "user_name"
SESSION_PENDING_INTERRUPT = "pending_interrupt"  # 标记有未处理的中断


# ═══════════════════════════════════════════════════════════════
# SSE 事件处理
# ═══════════════════════════════════════════════════════════════


async def _handle_sse_event(event_type: str, data: dict, msg: cl.Message) -> bool:
    """处理单个 SSE 事件。返回 True = 流结束（应 break）。"""

    if event_type == "status":
        conf = data.get("confidence", 0.0)
        it = data.get("iteration", 0)
        missing = data.get("missing_info", "")
        parts = [f"\n🔄 **第 {it} 轮评估** — 置信度 {conf:.0%}"]
        if missing:
            parts.append(f"  ⚠️ 缺失信息：{missing}")
        await msg.stream_token("\n".join(parts))

    elif event_type == "search_result":
        kw = data.get("keyword", "")
        content = data.get("content", "")
        if kw and content:
            await msg.stream_token(f"\n🔍 **「{kw}」** → {content}\n")
        else:
            logger.warning("search_result 跳过显示: kw=%r content_empty=%s", kw, not content)

    elif event_type == "final":
        if data.get("content"):
            await msg.stream_token(data["content"])
        return True  # 正常结束

    elif event_type == "kpi":
        kpi = data.get("data", {})
        await msg.stream_token(
            f"\n---\n📊 **执行统计**\n"
            f"- 模型：{kpi.get('current_llm', '?')}\n"
            f"- 输入 Token：{kpi.get('input_tokens', 0):,}\n"
            f"- 输出 Token：{kpi.get('output_tokens', 0):,}\n"
            f"- 总 Token：{kpi.get('total_tokens', 0):,}\n"
            f"- 置信度：{kpi.get('confidence_score', 0):.0%}\n"
            f"- 模型切换：{kpi.get('model_switches', 0)} 次\n"
            f"- 耗时：{kpi.get('elapsed_seconds', 0):.1f}s\n"
        )

    elif event_type == "error":
        await msg.stream_token(f"\n❌ **错误**：{data.get('message', '未知错误')}")
        return True

    return False


# ═══════════════════════════════════════════════════════════════
# 流式请求（中断感知版本）
# ═══════════════════════════════════════════════════════════════


async def _stream_agent_with_interrupt(
    *,
    thread_id: str,
    message: str | None,
    resume_value: str | None,
    msg: cl.Message,
    client: httpx.AsyncClient,
) -> str | None:
    """向 /chat/stream 发起流式请求，遇 interrupt 事件时弹出审批卡片。

    Returns:
        None  — 流正常结束或错误结束
        "yes" — 用户点击批准（需外层 while 循环继续恢复）
        "no"  — 用户点击拒绝（需外层 while 循环结束会话）
    """

    # ── 构造请求体 ────────────────────────────────────────
    payload: dict[str, Any] = {"thread_id": thread_id}
    if resume_value is not None:
        payload["resume_value"] = resume_value
        payload["message"] = None
    else:
        payload["message"] = message or ""

    async with client.stream("POST", f"{FASTAPI_BASE_URL}/chat/stream", json=payload) as response:
        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            try:
                event_data = json.loads(line[6:])
            except json.JSONDecodeError:
                logger.warning("无效 JSON 行: %s", line)
                continue

            event_type = event_data.get("type", "")

            # 中断事件：弹出行内审批卡片
            if event_type == "interrupt":
                question = event_data.get("question", "需要您的批准")
                cl.user_session.set(SESSION_PENDING_INTERRUPT, True)

                wait_msg = cl.Message(content=f"⏳ 等待审批：{question}")
                await wait_msg.send()

                res = await cl.AskActionMessage(
                    content=f"**{question}**",
                    actions=[
                        cl.Action(
                            name="approve",
                            value="yes",
                            label="✅ 批准继续",
                            payload={"action": "approve"},
                        ),
                        cl.Action(
                            name="reject", value="no", label="❌ 放弃", payload={"action": "reject"}
                        ),
                    ],
                ).send()

                choice: str = str(getattr(res, "value", "no")) if res else "no"
                wait_msg.content = f"{'✅ 已批准' if choice == 'yes' else '❌ 已放弃'}，继续执行..."
                await wait_msg.update()
                cl.user_session.set(SESSION_PENDING_INTERRUPT, False)
                return choice

            # 普通事件：委托给通用处理器
            should_stop = await _handle_sse_event(event_type, event_data, msg)
            if should_stop:
                await msg.update()
                return None

    await msg.update()
    return None


# ═══════════════════════════════════════════════════════════════
# Chainlit 生命周期 — on_chat_start
# ═══════════════════════════════════════════════════════════════


async def _check_backend_status(thread_id: str) -> dict:
    """静默查询后端会话状态。返回 {"status": "idle"|...} 或 {"error": str}。"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{FASTAPI_BASE_URL}/status/{thread_id}")
        if resp.status_code == 200:
            data: dict[str, Any] = resp.json()
            return data
        return {"status": "idle", "error": f"HTTP {resp.status_code}"}
    except Exception as exc:
        logger.warning("后端状态查询失败: %s", exc)
        return {"status": "idle", "error": str(exc)}


async def _send_welcome(user_name: str) -> None:
    """发送欢迎消息。"""
    await cl.Message(
        content=(
            f"👋 欢迎，**{user_name}**！\n\n"
            "我是自适应搜索助手，基于 LangGraph 构建。\n"
            "输入你的问题，我会自动搜索、评估并生成报告。"
        )
    ).send()


async def _handle_interrupted_session(thread_id: str) -> None:
    """处理中断会话：行内卡片询问用户意图。"""

    res = await cl.AskActionMessage(
        content=(
            "🔔 **检测到上一次研讨尚未完成。**\n\n"
            "可能是网络中断或你上次关闭了页面。你想怎么处理？"
        ),
        actions=[
            cl.Action(
                name="resume_interrupted",
                value="resume",
                label="✅ 恢复上一次研讨",
                description="从中断处继续执行",
                payload={"action": "resume"},
            ),
            cl.Action(
                name="new_interrupted",
                value="new",
                label="🆕 开启全新研讨",
                description="放弃上一次未完成的任务",
                payload={"action": "new"},
            ),
        ],
        timeout=120,
    ).send()

    # Chainlit 2.x 返回类型可能是 Action / dict / str，多路径探测
    raw = repr(res)
    logger.info("中断卡片返回值: type=%s repr=%s", type(res).__name__, raw)

    # 尝试多种取值路径
    choice: str | None = None
    if res is None:
        choice = None
    elif isinstance(res, str):
        choice = res
    elif isinstance(res, dict):
        choice = res.get("value") or res.get("name")
    elif hasattr(res, "value"):
        choice = getattr(res, "value", None)

    logger.info("解析后的 choice=%s", choice)

    if choice is None:
        await _close_old_session(thread_id)
        await cl.Message(content="⏰ 操作超时，已自动开启全新研讨。").send()
    elif choice in ("resume", "resume_interrupted"):
        await _resume_interrupted_session(thread_id)
    else:
        await _close_old_session(thread_id)
        await cl.Message(content="✅ 已关闭上次研讨，开始新的对话。").send()


async def _close_old_session(thread_id: str) -> None:
    """通知后端关闭中断的旧会话。"""
    with suppress(Exception):
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                f"{FASTAPI_BASE_URL}/chat/stream",
                json={"thread_id": thread_id, "resume_value": "no", "message": None},
            )
    logger.info("已关闭中断会话: thread_id=%s", thread_id)


async def _resume_interrupted_session(thread_id: str) -> None:
    """恢复被中断的会话：发送 resume_value=yes 并流式输出后续结果。"""

    status_msg = cl.Message(content="⏳ 正在恢复上一次研讨，请稍候...")
    await status_msg.send()

    msg = cl.Message(content="")
    await msg.send()

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        try:
            await _stream_agent_with_interrupt(
                thread_id=thread_id,
                message=None,
                resume_value="yes",
                msg=msg,
                client=client,
            )
            status_msg.content = "✅ 上一次研讨已恢复完成。"
            await status_msg.update()

        except httpx.HTTPStatusError as exc:
            msg.content = f"❌ 后端错误 (HTTP {exc.response.status_code})"
            await msg.update()
            logger.exception("恢复中断会话时后端报错")
        except httpx.TimeoutException:
            msg.content = "❌ 恢复请求超时，请重试"
            await msg.update()
            logger.exception("恢复中断会话超时")
        except Exception as exc:
            msg.content = f"❌ 恢复失败：{exc}"
            await msg.update()
            logger.exception("恢复中断会话时发生未知错误")


@cl.on_chat_start
async def on_chat_start() -> None:
    """会话启动：身份识别 → 静默查后端状态 → 仅 interrupted 时出行内卡片。"""

    # ── 第一步：确定用户身份和 thread_id ────────────────────
    if DEVELOPER_MODE:
        user_name = "developer"
        thread_id = "developer_workspace"  # 锁定，刷新 100% 命中同一会话
    else:
        name_res = await cl.AskUserMessage(
            content="🍵 研讨室一号，请问阁下尊姓大名？", timeout=60
        ).send()
        user_name = (
            name_res["output"].strip()
            if name_res and name_res.get("output", "").strip()
            else "访客"
        )
        thread_id = f"{user_name}-{cl.context.session.id[:8]}"

    cl.user_session.set(SESSION_USER_NAME, user_name)
    cl.user_session.set(SESSION_THREAD_ID, thread_id)
    cl.user_session.set(SESSION_PENDING_INTERRUPT, False)

    logger.info(
        "会话启动: user=%s, thread_id=%s, dev_mode=%s", user_name, thread_id, DEVELOPER_MODE
    )

    # ── 第二步：静默检查后端状态 ────────────────────────────
    status_data = await _check_backend_status(thread_id)
    status = status_data.get("status", "idle")
    logger.info("会话状态: thread_id=%s, status=%s", thread_id, status)

    # ── 第三步：根据状态分流 ─────────────────────────────────
    if status == "interrupted":
        await _handle_interrupted_session(thread_id)
    else:
        # idle / completed / failed / running(僵尸) → 正常欢迎
        await _send_welcome(user_name)


# ═══════════════════════════════════════════════════════════════
# 用户消息入口（含多次中断循环）
# ═══════════════════════════════════════════════════════════════


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """用户发送消息时触发。支持首次搜索 + 多次中断审批循环。"""

    thread_id: str = cl.user_session.get(SESSION_THREAD_ID)
    if not thread_id:
        await cl.Message(content="❌ 会话未初始化，请刷新页面。").send()
        return

    msg = cl.Message(content="")
    await msg.send()

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        try:
            # 第一次请求
            result = await _stream_agent_with_interrupt(
                thread_id=thread_id,
                message=message.content,
                resume_value=None,
                msg=msg,
                client=client,
            )

            # 循环处理多次中断（Agent 可能在多轮搜索中都触发审批）
            while result in ("yes", "no"):
                result = await _stream_agent_with_interrupt(
                    thread_id=thread_id,
                    message=None,
                    resume_value=result,
                    msg=msg,
                    client=client,
                )

        except httpx.HTTPStatusError as exc:
            msg.content = f"❌ 后端错误 (HTTP {exc.response.status_code})"
            await msg.update()
            logger.exception("on_message 后端错误")
        except httpx.TimeoutException:
            msg.content = "❌ 请求超时，请重试"
            await msg.update()
            logger.exception("on_message 超时")
        except Exception as exc:
            msg.content = f"❌ 服务异常：{exc}"
            await msg.update()
            logger.exception("on_message 未知错误")


# ═══════════════════════════════════════════════════════════════
# 用户主动停止
# ═══════════════════════════════════════════════════════════════


@cl.on_stop
async def on_stop() -> None:
    thread_id = cl.user_session.get(SESSION_THREAD_ID)
    logger.info("用户主动停止: thread_id=%s", thread_id)
