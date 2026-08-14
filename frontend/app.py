"""
Chainlit 前端入口 — 自适应搜索助手（全自动模式）。

职责：
- 用户身份识别（AskUserMessage）
- SSE 流式消费 — 思考过程 → cl.Step / 最终报告 → cl.Message
"""

import json
import logging
import os

import chainlit as cl
import httpx
from dotenv import load_dotenv

# ── 环境 & 配置 ──────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 开发开关 ────────────────────────────────────────────────
DEVELOPER_MODE = True

# ── 会话键名常量 ─────────────────────────────────────────────
SESSION_THREAD_ID = "thread_id"
SESSION_USER_NAME = "user_name"

# ── 步骤名称 ─────────────────────────────────────────────────
THINKING_STEP_NAME = "思考过程"


# ═══════════════════════════════════════════════════════════════
# SSE 事件处理 — 分两路：思考过程 → Step / 最终输出 → Message
# ═══════════════════════════════════════════════════════════════


def _format_kpi(data: dict) -> str:
    """格式化 KPI 统计文本。"""
    kpi = data.get("data", {})
    return (
        f"\n---\n📊 **执行统计**\n"
        f"- 模型：{kpi.get('current_llm', '?')}\n"
        f"- 输入 Token：{kpi.get('input_tokens', 0):,}\n"
        f"- 输出 Token：{kpi.get('output_tokens', 0):,}\n"
        f"- 总 Token：{kpi.get('total_tokens', 0):,}\n"
        f"- 置信度：{kpi.get('confidence_score', 0):.0%}\n"
        f"- 模型切换：{kpi.get('model_switches', 0)} 次\n"
        f"- 耗时：{kpi.get('elapsed_seconds', 0):.1f}s\n"
    )


async def _handle_thinking_event(
    event_type: str,
    data: dict,
    step: cl.Step,
) -> None:
    """将中间状态/搜索/KPI 事件流式写入可折叠 Step。"""
    if event_type == "status":
        conf = data.get("confidence", 0.0)
        it = data.get("iteration", 0)
        missing = data.get("missing_info", "")
        text = f"\n🔄 **第 {it} 轮评估** — 置信度 {conf:.0%}"
        if missing:
            text += f"\n  ⚠️ 缺失信息：{missing}"
        await step.stream_token(text + "\n")

    elif event_type == "search_result":
        kw = data.get("keyword", "")
        content = data.get("content", "")
        if kw and content:
            await step.stream_token(f"\n🔍 **「{kw}」** → {content}\n")

    elif event_type == "kpi":
        await step.stream_token(_format_kpi(data))


async def _handle_output_event(
    event_type: str,
    data: dict,
    msg: cl.Message,
) -> bool:
    """将最终报告/错误事件流式写入主消息。返回 True = 流结束。"""
    if event_type == "final":
        if data.get("content"):
            await msg.stream_token(data["content"])
        return True

    elif event_type == "error":
        await msg.stream_token(f"\n❌ **错误**：{data.get('message', '未知错误')}")
        return True

    return False


# ═══════════════════════════════════════════════════════════════
# 流式请求（全自动，无中断审批）
# ═══════════════════════════════════════════════════════════════


async def _stream_agent(
    *,
    thread_id: str,
    message: str,
    thinking_step: cl.Step,
    output_msg: cl.Message,
    client: httpx.AsyncClient,
) -> None:
    """向 /chat/stream 发起流式请求。

    中间过程（status/search_result/kpi）→ thinking_step.stream_token()
    最终报告（final/error）→ output_msg.stream_token()
    """
    payload = {"thread_id": thread_id, "message": message}

    async with client.stream(
        "POST",
        f"{FASTAPI_BASE_URL}/chat/stream",
        json=payload,
    ) as response:
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

            # 分流：思考事件 → Step / 输出事件 → Message
            if event_type in ("status", "search_result", "kpi"):
                await _handle_thinking_event(event_type, event_data, thinking_step)
            else:
                should_stop = await _handle_output_event(
                    event_type,
                    event_data,
                    output_msg,
                )
                if should_stop:
                    await output_msg.update()
                    return

    await output_msg.update()


# ═══════════════════════════════════════════════════════════════
# Chainlit 生命周期 — on_chat_start
# ═══════════════════════════════════════════════════════════════


async def _send_welcome(user_name: str) -> None:
    await cl.Message(
        content=(
            f"👋 欢迎，**{user_name}**！\n\n"
            "我是自适应搜索助手，基于 LangGraph 构建。\n"
            "输入你的问题，我会自动搜索、评估并生成报告。"
        )
    ).send()


@cl.on_chat_start
async def on_chat_start() -> None:
    """会话启动：身份识别 → 欢迎。"""
    if DEVELOPER_MODE:
        user_name = "developer"
        thread_id = "developer_workspace"
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

    logger.info(
        "会话启动: user=%s, thread_id=%s, dev_mode=%s",
        user_name,
        thread_id,
        DEVELOPER_MODE,
    )
    await _send_welcome(user_name)


# ═══════════════════════════════════════════════════════════════
# 用户消息入口（全自动：Step 思考框 → 流式报告）
# ═══════════════════════════════════════════════════════════════


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """用户发送消息时触发。

    1. 创建 cl.Step("思考过程") — 收集中间日志
    2. 创建 cl.Message — 接收最终报告
    3. 图自动运行到底，无需人工审批
    """
    thread_id: str = cl.user_session.get(SESSION_THREAD_ID)
    if not thread_id:
        await cl.Message(content="❌ 会话未初始化，请刷新页面。").send()
        return

    async with cl.Step(name=THINKING_STEP_NAME, type="run") as thinking_step:
        output_msg = cl.Message(content="")
        await output_msg.send()

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
        ) as client:
            try:
                await _stream_agent(
                    thread_id=thread_id,
                    message=message.content,
                    thinking_step=thinking_step,
                    output_msg=output_msg,
                    client=client,
                )
            except httpx.HTTPStatusError as exc:
                output_msg.content = f"❌ 后端错误 (HTTP {exc.response.status_code})"
                await output_msg.update()
                logger.exception("on_message 后端错误")
            except httpx.TimeoutException:
                output_msg.content = "❌ 请求超时，请重试"
                await output_msg.update()
                logger.exception("on_message 超时")
            except Exception as exc:
                output_msg.content = f"❌ 服务异常：{exc}"
                await output_msg.update()
                logger.exception("on_message 未知错误")


# ═══════════════════════════════════════════════════════════════
# 用户主动停止
# ═══════════════════════════════════════════════════════════════


@cl.on_stop
async def on_stop() -> None:
    thread_id = cl.user_session.get(SESSION_THREAD_ID)
    logger.info("用户主动停止: thread_id=%s", thread_id)
