import json
import logging
import os
from contextlib import suppress

import chainlit as cl
import httpx
from dotenv import load_dotenv

load_dotenv()  # 加载 frontend/.env
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
CHAINLIT_PORT = int(os.getenv("CHAINLIT_PORT", 8001))

# 配置日志（可自定义）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 公共工具函数
# =============================================================================


async def _handle_sse_event(
    event_type: str,
    data: dict,
    msg: cl.Message,
) -> bool:
    """处理单个 SSE 事件，返回 True 表示流已结束（应 break/return）。

    抽取为公共函数，避免 on_message 和 resume_agent 中重复代码。
    """
    if event_type == "status":
        # 中间状态：显示置信度变化和迭代进度
        conf = data.get("confidence", 0.0)
        it = data.get("iteration", 0)
        missing = data.get("missing_info", "")
        lines = [f"\n\n🔄 **第 {it} 轮评估** — 置信度 {conf:.0%}"]
        if missing:
            lines.append(f"  ⚠️ 缺失信息：{missing}")
        await msg.stream_token("\n".join(lines))

    elif event_type == "final":
        if data.get("content"):
            await msg.stream_token(data["content"])
        return True  # 正常结束

    elif event_type == "kpi":
        kpi = data.get("data", {})
        kpi_text = (
            f"\n\n---\n📊 **执行统计**\n"
            f"- 模型：{kpi.get('current_llm', '?')}\n"
            f"- 输入 Token：{kpi.get('input_tokens', 0):,}\n"
            f"- 输出 Token：{kpi.get('output_tokens', 0):,}\n"
            f"- 总 Token：{kpi.get('total_tokens', 0):,}\n"
            f"- 置信度：{kpi.get('confidence_score', 0):.0%}\n"
            f"- 模型切换：{kpi.get('model_switches', 0)} 次\n"
            f"- 耗时：{kpi.get('elapsed_seconds', 0):.1f}s\n"
        )
        await msg.stream_token(kpi_text)

    elif event_type == "error":
        await msg.stream_token(f"\n\n❌ **错误**：{data.get('message', '未知错误')}")
        return True  # 异常结束

    return False  # 继续读下一个事件


async def _stream_agent(
    *,
    thread_id: str,
    message: str | None,
    resume_value: str | None,
    msg: cl.Message,
    client: httpx.AsyncClient,
) -> None:
    """向 /chat/stream 发起流式请求，并将 SSE 事件逐条写入 msg。

    统一了首次执行和恢复执行两种场景的 HTTP 流消费逻辑。
    """
    payload: dict = {"thread_id": thread_id}
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
                logger.warning(f"Invalid JSON line: {line}")
                continue

            event_type = event_data.get("type", "")
            should_stop = await _handle_sse_event(event_type, event_data, msg)
            if should_stop:
                break

    await msg.update()


# =============================================================================
# Chainlit 生命周期
# =============================================================================


@cl.on_chat_start
async def start():
    """会话开始时：初始化 thread_id，检查是否有中断的会话。

    三种路径：
    - idle / 无记录  → 正常开始，等用户发消息
    - interrupted    → 弹窗让用户选择"恢复"还是"新对话"
    """
    thread_id = cl.context.session.id
    cl.user_session.set("thread_id", thread_id)

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{FASTAPI_BASE_URL}/status/{thread_id}")
        except Exception as e:
            logger.error(f"Check status failed: {e}")
            await cl.Message(content="⚠️ 无法连接后端服务，请确认后端已启动").send()
            return

    if resp.status_code != 200:
        await cl.Message(content="✅ 欢迎使用 AdaptiveSearchAgent！请输入你的问题。").send()
        return

    status = resp.json().get("status", "idle")
    if status != "interrupted":
        await cl.Message(content="✅ 欢迎使用 AdaptiveSearchAgent！请输入你的问题。").send()
        return

    # ── 状态为 interrupted ────────────────────────────────────
    actions = [
        cl.Action(name="resume", value="resume", label="✅ 恢复上一次对话"),
        cl.Action(name="new", value="new", label="🆕 开始新对话"),
    ]
    res = await cl.AskActionMessage(
        content="🔔 检测到有未完成的会话（等待审批中），是否继续？",
        actions=actions,
    ).send()

    choice = res.get("value")

    if choice == "resume":
        # ── 路径 A：立即恢复，不需要等用户额外发消息 ───────────
        status_msg = cl.Message(content="⏳ 正在恢复之前的对话，请稍候...")
        await status_msg.send()

        msg = cl.Message(content="")
        await msg.send()

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            try:
                await _stream_agent(
                    thread_id=thread_id,
                    message=None,
                    resume_value="yes",
                    msg=msg,
                    client=client,
                )
            except httpx.HTTPStatusError as e:
                await msg.update(content=f"❌ HTTP {e.response.status_code}")
            except httpx.TimeoutException:
                await msg.update(content="❌ 请求超时，请重试")
            except Exception as e:
                logger.exception("Unexpected error during resume")
                await msg.update(content=f"❌ 恢复失败：{str(e)}")

    else:
        # ── 路径 B：用户选择新对话，发 resume_value="no" 关闭旧会话 ──
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as close_client:
            with suppress(Exception):
                await close_client.post(
                    f"{FASTAPI_BASE_URL}/chat/stream",
                    json={"thread_id": thread_id, "resume_value": "no", "message": None},
                )
        await cl.Message(content="✅ 已关闭上次会话，请输入新的问题开始对话。").send()


# =============================================================================
# 中断审批 —— 在 SSE 流中处理
# =============================================================================

# 说明：审批交互不在此文件中作为独立函数暴露，而是在 _handle_sse_event
# 的调用方（当前为 on_message 的 _stream_agent 调用）中，
# on_message 在收到 interrupt 事件后，由业务层处理 AskActionMessage 弹窗，
# 然后调用 _stream_agent(resume_value="yes"/"no") 恢复。
#
# 因此 on_message 需要能感知 interrupt 事件 —— 当前设计中
# _handle_sse_event 不处理 interrupt 类型，
# 需要由 _stream_agent 调用方自行处理。

# 为了支持中断审批，我们对 _stream_agent 做一点改造：
# 让它返回"是否因 interrupt 而暂停"，由调用方决定后续动作。


async def _stream_agent_with_interrupt(
    *,
    thread_id: str,
    message: str | None,
    resume_value: str | None,
    msg: cl.Message,
    client: httpx.AsyncClient,
) -> str | None:
    """增强版 _stream_agent，返回值表示流结束原因。

    Returns:
        None          — 正常结束（final / kpi 后 break）
        "yes" / "no"  — 用户选择了审批结果（需要调用方发起恢复请求）
        "error"       — 发生错误
    """
    payload: dict = {"thread_id": thread_id}
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
                logger.warning(f"Invalid JSON line: {line}")
                continue

            event_type = event_data.get("type", "")

            # ── interrupt 特殊处理：弹窗获取用户选择 ──
            if event_type == "interrupt":
                question = event_data.get("question", "需要您的批准")
                # 显示等待状态
                wait_msg = cl.Message(content=f"⏳ **等待审批**：{question}")
                await wait_msg.send()

                actions = [
                    cl.Action(name="approve", value="yes", label="✅ 批准"),  # type: ignore[call-arg]
                    cl.Action(name="reject", value="no", label="❌ 拒绝"),  # type: ignore[call-arg]
                ]
                res = await cl.AskActionMessage(content=f"**{question}**", actions=actions).send()
                choice: str = str(res.get("value", "no")) if res else "no"
                await wait_msg.update(  # type: ignore[call-arg]
                    content=f"✅ 已选择：{'批准' if choice == 'yes' else '拒绝'}，继续执行..."
                )
                return choice  # 返回给调用方，调用方负责发起恢复请求

            # ── 其余事件：统一处理 ──
            should_stop = await _handle_sse_event(event_type, event_data, msg)
            if should_stop:
                await msg.update()
                return None  # 正常结束

    await msg.update()
    return None


@cl.on_message
async def on_message(message: cl.Message):
    """处理用户消息 —— 支持首次搜索 + 中断审批循环。

    流程：
    1. 首次执行：发送用户消息 → 图执行 → 可能遇到 interrupt
    2. 遇到 interrupt → 弹窗审批 → 自动发起恢复请求
    3. 恢复执行 → 可能再次 interrupt（理论上不会，但支持）
    4. 直到 final/kpi/error 结束
    """
    thread_id: str = cl.user_session.get("thread_id")  # type: ignore[assignment]

    msg = cl.Message(content="")
    await msg.send()

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        try:
            # ── 首次执行 ──
            result = await _stream_agent_with_interrupt(
                thread_id=thread_id,
                message=message.content,
                resume_value=None,
                msg=msg,
                client=client,
            )

            # ── 循环处理中断（用户可能多次审批） ──
            while result in ("yes", "no"):
                result = await _stream_agent_with_interrupt(
                    thread_id=thread_id,
                    message=None,
                    resume_value=result,
                    msg=msg,
                    client=client,
                )

        except httpx.HTTPStatusError as e:
            await msg.update(content=f"❌ HTTP {e.response.status_code}: {e.response.text}")  # type: ignore[call-arg]
        except httpx.TimeoutException:
            await msg.update(content="❌ 请求超时，请重试")  # type: ignore[call-arg]
        except Exception as e:
            logger.exception("Unexpected error in on_message")
            await msg.update(content=f"❌ 内部错误：{str(e)}")  # type: ignore[call-arg]


@cl.on_stop
async def on_stop():
    """用户主动停止会话时，可以通知后端（可选）"""
    thread_id = cl.user_session.get("thread_id")
    logger.info(f"Session {thread_id} stopped by user")
