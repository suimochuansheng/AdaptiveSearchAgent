"""
Chainlit 前端入口 — 自适应搜索助手（全自动模式）。

职责：
- 用户身份识别（AskUserMessage）
- SSE 流式消费 — 思考过程 → cl.Step / 最终报告 → cl.Message
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

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
SESSION_MODE = "chat_mode"

# ── 对话模式（默认 deep_research，旧流程零变化）────────────────
MODE_DEEP_RESEARCH = "deep_research"
MODE_CONVERSATION = "conversation"
DEFAULT_CHAT_MODE = MODE_DEEP_RESEARCH
CONVERSATION_ENDPOINT = "/chat/conversation"
APPROVAL_TIMEOUT_SECONDS = 120
APPROVAL_ACTION_APPROVE = "approve"
APPROVAL_ACTION_REJECT = "reject"

# ── 步骤名称 ─────────────────────────────────────────────────
THINKING_STEP_NAME = "思考过程"

# ── 流式输出分块大小（字符）—— 打字机效果的每段长度 ────────────
_STREAM_CHUNK_SIZE = 60

# ── RAG 文档上传（rag-ingest-gateway 子服务）────────────────────
RAG_GATEWAY_BASE_URL = os.getenv("RAG_GATEWAY_BASE_URL", "http://localhost:8100")
ALLOWED_UPLOAD_EXTS = {".pdf", ".docx", ".md", ".txt"}
SESSION_UPLOAD_HISTORY = "upload_history"


def _format_file_size(num_bytes: int) -> str:
    """字节数 → 人类可读大小。"""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / 1024 / 1024:.1f} MB"


async def _ingest_document(
    file_name: str,
    content: bytes,
    mime_type: str,
    kb_id: str = "default",
) -> dict:
    """POST /api/v1/ingest 提交文档，返回 {task_id, status}。"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        resp = await client.post(
            f"{RAG_GATEWAY_BASE_URL}/api/v1/ingest",
            files={"file": (file_name, content, mime_type or "application/octet-stream")},
            data={"kb_id": kb_id},
        )
        resp.raise_for_status()
        return resp.json()


async def _query_task_status(task_id: str) -> dict:
    """GET /api/v1/tasks/{task_id} 查询摄入任务状态。"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
        resp = await client.get(f"{RAG_GATEWAY_BASE_URL}/api/v1/tasks/{task_id}")
        if resp.status_code == 404:
            return {"task_id": task_id, "status": "NOT_FOUND"}
        resp.raise_for_status()
        return resp.json()


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
        content = data.get("content", "")
        logger.info(f"📦 开始分块输出，总长度: {len(content)}")
        if content:
            # 后端把整篇报告放在一个 final 事件里；切成小块逐段 stream，
            # Chainlit 才会逐段渲染（一次 stream_token 全文 = 一次性 append）
            for i in range(0, len(content), _STREAM_CHUNK_SIZE):
                await msg.stream_token(content[i : i + _STREAM_CHUNK_SIZE])
                # 打字机效果：每段输出后稍作停顿，避免一次性输出过快
                await asyncio.sleep(0.1)
        else:
            # 空报告：stream_token 不会触发消息创建，显式发送空消息兜底
            await msg.send()
        return True

    elif event_type == "error":
        await msg.stream_token(f"\n❌ **错误**：{data.get('message', '未知错误')}")
        return True

    return False


# ═══════════════════════════════════════════════════════════════
# 流式请求（全自动，无中断审批）
# ═══════════════════════════════════════════════════════════════


async def _stream_agent(  # noqa: C901
    *,
    thread_id: str,
    message: str,
    client: httpx.AsyncClient,
    thinking_step: cl.Step | None = None,
    resume_value: str | None = None,
    output_msg: cl.Message | None = None,
    suppress_thinking: bool = False,
    lazy_create_step: bool = False,
) -> tuple[dict | None, cl.Message | None]:
    """向 /chat/stream 发起流式请求。

    中间过程（status/search_result/kpi）→ 惰性创建「思考过程」Step 并流式写入；
    最终报告（final/error）→ output_msg.stream_token()；
    中断（interrupt）→ 返回 (中断数据, output_msg)，由调用方处理。

    suppress_thinking=True 时跳过所有思考类事件（用于“取消”路径：
    不渲染思考过程、不显示 KPI，仅接收最终取消报告）。
    lazy_create_step=True 时在首个思考事件到达时自动创建并发送 Step，
    使 Step.send() 与 Message.stream_start() 在同一事件循环中紧密执行。
    """
    payload: dict = {"thread_id": thread_id}
    if resume_value is not None:
        payload["resume_value"] = resume_value
    else:
        payload["message"] = message

    step = thinking_step
    # 仅当 Step 由本函数惰性创建时，才由本函数负责 update；
    # 调用方传入的 Step 由调用方负责 update，避免同一个 Step 被 update 两次。
    owns_step = thinking_step is None

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

            # 中断事件：停止读取，返回中断数据交给上层做人在环处理
            if event_type == "interrupt":
                if step is not None and owns_step:
                    await step.update()
                if output_msg is not None:
                    await output_msg.update()
                return event_data.get("data", {}), output_msg

            # 思考类事件：仅在需要时惰性创建 Step 并写入
            if event_type in ("status", "search_result", "kpi"):
                if suppress_thinking:
                    continue
                # 兜底：evaluator 运行前（iteration == 0）的 status 是初始快照占位，跳过
                if event_type == "status" and event_data.get("iteration", 0) == 0:
                    continue

                # 关键修改：当 lazy_create_step=True 时，在第一个思考事件到达时创建并发送 Step
                if step is None and lazy_create_step:
                    step = cl.Step(name=THINKING_STEP_NAME, type="run")
                    await step.send()
                    owns_step = True  # 惰性创建的 Step 由本函数管理

                if step is None:
                    # 没有 Step 可写入，跳过思考事件
                    continue

                await _handle_thinking_event(event_type, event_data, step)
                continue

            # 输出类事件（final/error）
            if output_msg is None:
                # 不预先 send 空消息：首个 stream_token 会触发 stream_start 创建消息，
                # 确保 Message 的创建时刻晚于「思考过程」Step
                output_msg = cl.Message(content="")
            should_stop = await _handle_output_event(
                event_type,
                event_data,
                output_msg,
            )
            if should_stop:
                # 惰性创建的 Step 由本函数负责收尾 update
                if step is not None and owns_step:
                    await step.update()
                return None, output_msg

    # 流正常结束但未收到 final/error（例如仅 kpi 事件）时的收尾
    if step is not None and owns_step:
        await step.update()
    if output_msg is not None:
        await output_msg.update()
    return None, output_msg


async def _ask_missing_info(interrupt_data: dict) -> str | None:
    """人在环：展示缺失信息提示，向用户询问补充信息。

    Args:
        interrupt_data: 后端 interrupt 事件的 data 字段（need_more_info payload）。

    Returns:
        用户输入的补充信息；未输入/超时则返回 None。
    """
    missing_info = interrupt_data.get("missing_info", "")
    summary = interrupt_data.get("summary", [])

    await cl.Message(
        content=(
            "🛑 **系统已尽力搜索，但信息仍不充分，请补充以下缺失信息以便定向搜索**\n\n"
            f"**缺失信息**：{missing_info}"
        )
    ).send()

    # 已搜到的内容摘要（可折叠展示）
    if summary:
        lines = []
        for item in summary:
            title = item.get("title", "")
            source = item.get("source", "")
            preview = item.get("content_preview", "")
            lines.append(f"- **{title}**（来源：{source}）\n  {preview}")
        async with cl.Step(name="已搜到的内容摘要", type="tool") as step:
            step.output = "\n\n".join(lines)

    # 输入框 + 发送按钮
    res = await cl.AskUserMessage(
        content="请在下方输入补充信息，我将据此进行定向搜索：",
        timeout=300,
    ).send()

    if res and res.get("output", "").strip():
        return res["output"].strip()
    return None


async def _ask_cost_confirm(interrupt_data: dict) -> str | None:
    """人在环：高成本搜索前置确认（cost_confirm）。

    展示预计搜索成本，向用户提供「确认继续 / 取消」两个按钮。
    确认 → 返回 "confirmed"；取消 → 返回 "cancelled"；超时/无响应 → None。
    """
    keyword_count = interrupt_data.get("keyword_count", 0)
    estimated_cost = interrupt_data.get("estimated_cost", 0.0)
    message = interrupt_data.get("message", "")

    content = message or (
        f"本次需要搜索 {keyword_count} 个关键词，预计消耗约 {estimated_cost} 元，是否继续？"
    )
    await cl.Message(content=f"🛑 **高成本搜索确认**\n\n{content}").send()

    res = await cl.AskActionMessage(
        content="请确认是否继续本次搜索：",
        actions=[
            cl.Action(name="confirm", payload={"value": "confirmed"}, label="✅ 确认继续"),
            cl.Action(name="cancel", payload={"value": "cancelled"}, label="❌ 取消"),
        ],
        timeout=300,
    ).send()

    if not res:
        return None

    # Chainlit 2.x 返回 AskActionResponse：
    # {"name": ..., "payload": {...}, "label": ..., "tooltip": ..., "forId": ..., "id": ...}
    # 业务值在 payload.value 中，而非顶层 value。
    payload = res.get("payload") or {}
    value = payload.get("value")

    # 兼容旧版 Chainlit（1.x 曾把 value 直接放在顶层）
    if value is None:
        value = res.get("value")

    return str(value).strip() if value else None


# ═══════════════════════════════════════════════════════════════
# 对话模式（/chat/conversation）— 模式开关 / 流式渲染 / 审批卡片
# ═══════════════════════════════════════════════════════════════


def _current_mode() -> str:
    """返回当前对话模式；未设置或取值非法时一律回落到 deep_research。"""
    try:
        mode = cl.user_session.get(SESSION_MODE)
    except Exception:  # noqa: BLE001 —— 脱离 Chainlit 运行时（脚本/单测）时按默认模式
        mode = None
    return mode if mode in (MODE_DEEP_RESEARCH, MODE_CONVERSATION) else DEFAULT_CHAT_MODE


def _select_widget_cls() -> type:
    """取 Chainlit 的 Select 组件类（2.x 在 cl.input_widget，旧版在 cl.Select）。"""
    try:
        return cl.input_widget.Select  # type: ignore[attr-defined]
    except (AttributeError, KeyError):
        return cl.Select  # type: ignore[attr-defined]


def _make_chat_settings() -> cl.ChatSettings:
    """构造模式开关（默认 deep_research）。"""
    return cl.ChatSettings(
        inputs=[
            _select_widget_cls()(
                id=SESSION_MODE,
                label="对话模式",
                values=[MODE_DEEP_RESEARCH, MODE_CONVERSATION],
                initial_value=DEFAULT_CHAT_MODE,
                description=(
                    "deep_research = 深度调研（默认，/chat/stream）；"
                    "conversation = 会话式 Agent（/chat/conversation，含审批卡片）"
                ),
            )
        ]
    )


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    """用户切换模式时写入会话；默认值始终是 deep_research，旧流程不受影响。"""
    mode = str(settings.get(SESSION_MODE) or DEFAULT_CHAT_MODE)
    if mode not in (MODE_DEEP_RESEARCH, MODE_CONVERSATION):
        mode = DEFAULT_CHAT_MODE
    cl.user_session.set(SESSION_MODE, mode)
    logger.info("对话模式切换: %s", mode)


def _tool_names(event: dict) -> list[str]:
    """从 tool_call 帧中取出工具名列表（容错：结构异常时返回空列表）。"""
    names: list[str] = []
    tools = event.get("tools")
    if isinstance(tools, list):
        for item in tools:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
    if not names and event.get("tool"):
        names.append(str(event["tool"]))
    return names


def _format_error_bubble(message: str) -> str:
    """错误气泡文本（❌ 前缀，与旧端点 /chat/stream 的呈现保持一致）。"""
    return f"❌ **执行出错**\n\n{message or '未知错误'}"


async def _write_error(message: str, output_msg: cl.Message | None) -> cl.Message:
    """把可读错误写进主消息；没有主消息时新建并发送一个错误气泡。"""
    text = _format_error_bubble(message)
    if output_msg is None:
        bubble = cl.Message(content=text)
        await bubble.send()
        return bubble
    existing = getattr(output_msg, "content", "") or ""
    output_msg.content = f"{existing}\n\n{text}" if existing else text
    await output_msg.update()
    return output_msg


async def _consume_conversation_frames(  # noqa: C901
    response: httpx.Response,
    output_msg: cl.Message | None,
    step: cl.Step | None,
) -> tuple[dict | None, cl.Message | None, cl.Step | None]:
    """消费会话端点 SSE 帧（token/tool_call/ping/error/final/interrupt）。

    - token：惰性创建主消息并 stream_token（增量渲染）
    - tool_call：写入「思考过程」Step（惰性创建）
    - ping：心跳，直接忽略（不打断渲染、不刷屏）
    - error：可读错误气泡，结束本次流
    - final：去重收尾 —— 流过 token 只 update()；一个 token 都没有才写 content
    - interrupt：返回 data 交给审批卡片
    - 未知 type：静默忽略，绝不抛 KeyError

    Returns:
        (中断数据或 None, 主消息, 思考 Step)
    """
    owns_step = step is None
    streamed = False

    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
        except (json.JSONDecodeError, TypeError):
            logger.warning("会话流无效 JSON 行: %s", line)
            continue
        if not isinstance(event, dict):
            continue

        event_type = event.get("type", "")

        if event_type == "token":
            content = event.get("content")
            if not isinstance(content, str) or not content:
                continue
            if output_msg is None:
                # 不预先 send 空消息：首个 stream_token 会触发消息创建
                output_msg = cl.Message(content="")
            await output_msg.stream_token(content)
            streamed = True
            continue

        if event_type == "tool_call":
            names = _tool_names(event)
            if not names:
                continue
            if step is None:
                step = cl.Step(name=THINKING_STEP_NAME, type="run")
                await step.send()
                owns_step = True
            await step.stream_token(f"\n🔧 调用工具: {', '.join(names)}\n")
            continue

        if event_type == "ping":
            continue

        if event_type == "interrupt":
            if step is not None and owns_step:
                await step.update()
            if output_msg is not None:
                await output_msg.update()
            data = event.get("data")
            return (data if isinstance(data, dict) else {}), output_msg, step

        if event_type == "final":
            answer = event.get("content")
            answer = answer if isinstance(answer, str) else ""
            if output_msg is None:
                # 一个 token 都没收到（模型直接返回整段）：才把 content 写进消息
                if answer:
                    output_msg = cl.Message(content=answer)
                    await output_msg.send()
            else:
                # 已经流过 token：只收尾，绝不重复输出整段回答
                if not streamed and answer:
                    output_msg.content = answer
                await output_msg.update()
            if step is not None and owns_step:
                await step.update()
            return None, output_msg, step

        if event_type == "error":
            output_msg = await _write_error(str(event.get("message") or "未知错误"), output_msg)
            if step is not None and owns_step:
                await step.update()
            return None, output_msg, step

        # 未知事件类型：静默忽略
        logger.debug("忽略未知会话事件: %s", event_type)

    if step is not None and owns_step:
        await step.update()
    if output_msg is not None:
        await output_msg.update()
    return None, output_msg, step


async def _stream_conversation(
    *,
    thread_id: str,
    message: str,
    client: httpx.AsyncClient,
    user_id: str | None = None,
    output_msg: cl.Message | None = None,
    step: cl.Step | None = None,
) -> tuple[dict | None, cl.Message | None, cl.Step | None]:
    """POST /chat/conversation，渲染五类 SSE 帧。

    Returns:
        (中断数据或 None, 主消息, 思考 Step)；中断数据非空时由调用方弹审批卡片。
    """
    payload = {"thread_id": thread_id, "message": message, "user_id": user_id or thread_id}
    async with client.stream(
        "POST",
        f"{FASTAPI_BASE_URL}{CONVERSATION_ENDPOINT}",
        json=payload,
    ) as response:
        response.raise_for_status()
        return await _consume_conversation_frames(response, output_msg, step)


def _format_money(value: object) -> str:
    """金额格式化（容错：非数字原样展示）。"""
    try:
        return f"{float(value):.2f} 元"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return f"{value} 元" if value not in (None, "") else "未知"


def _format_approval_card(interrupt_data: dict) -> str:
    """审批卡片正文：工具名 / 参数摘要 / 预估成本 / 剩余预算 / 到期时间。"""
    return (
        "### 🛡️ 审批请求\n\n"
        f"- **工具名**：`{interrupt_data.get('tool') or '未知工具'}`\n"
        f"- **参数摘要**：`{interrupt_data.get('args_digest') or '（无）'}`\n"
        f"- **预估成本**：{_format_money(interrupt_data.get('cost_estimate', 0))}\n"
        f"- **剩余预算**：{_format_money(interrupt_data.get('budget_remaining', 0))}\n"
        f"- **到期时间**：{interrupt_data.get('expires_at') or '未知'}\n"
    )


async def _ask_approval(interrupt_data: dict) -> str | None:
    """展示审批卡片并等待用户点击。

    复用 `_ask_cost_confirm` 的交互范式（AskActionMessage + payload.value 2.x/1.x 兼容）。
    Returns:
        "approve" / "reject"；超时或无响应返回 None（调用方按 reject 处理）。
        **不支持「记住选择」**，避免隐式放行。
    """
    await cl.Message(content=_format_approval_card(interrupt_data)).send()

    res = await cl.AskActionMessage(
        content=f"是否授权本次执行？（{APPROVAL_TIMEOUT_SECONDS} 秒未响应将自动拒绝）",
        actions=[
            cl.Action(
                name="approve",
                payload={"value": APPROVAL_ACTION_APPROVE},
                label="✅ 同意执行",
            ),
            cl.Action(
                name="reject",
                payload={"value": APPROVAL_ACTION_REJECT},
                label="❌ 拒绝",
            ),
        ],
        timeout=APPROVAL_TIMEOUT_SECONDS,
    ).send()

    if not res:
        return None

    payload = res.get("payload") or {}
    value = payload.get("value")
    # 兼容旧版 Chainlit（1.x 曾把 value 直接放在顶层）
    if value is None:
        value = res.get("value")

    return str(value).strip() if value else None


async def _decide_approval(interrupt_data: dict, decisions: dict[str, str | None]) -> str | None:
    """同一 approval_id 只问一次；重复帧复用上次决定，不再发卡片。"""
    approval_id = str(interrupt_data.get("approval_id") or "")
    if approval_id and approval_id in decisions:
        return decisions[approval_id]
    decision = await _ask_approval(interrupt_data)
    if approval_id:
        decisions[approval_id] = decision
    return decision


def _build_resume_body(interrupt_data: dict, decision: str) -> dict:
    """构造恢复请求体：三字段（approval_id / decision / request_id）。

    - `approval_id`：服务端签发的一次性凭证（32 hex），前端只回传，不生成；
    - `request_id`：每次都用新的 uuid4（客户端幂等键，服务端 24h 去重，防重放）；
    - 2026-11-11 裁决（OI-TDM-003）：删除 `token` 字段，只保留三字段。
      契约见 `docs/contracts/resume-http.md`。
    """
    return {
        "approval_id": str(interrupt_data.get("approval_id") or ""),
        "decision": decision,
        "request_id": str(uuid.uuid4()),
    }


RESUME_DISABLED_MESSAGE = (
    "⚠️ **审批功能未启用（等待 T9-2）**\n\n"
    "后端尚未提供审批恢复接口，本次审批不会执行，界面可继续使用。"
)


async def _resume_conversation(
    *,
    thread_id: str,
    interrupt_data: dict,
    decision: str,
    client: httpx.AsyncClient,
    output_msg: cl.Message | None = None,
    step: cl.Step | None = None,
) -> tuple[str, dict | None, cl.Message | None, cl.Step | None]:
    """POST /chat/conversation/{thread_id}/resume，并按 200 的 SSE 继续渲染。

    Returns:
        (状态, 下一次中断数据, 主消息, 思考 Step)。
        状态取值：resumed / unavailable / conflict / expired；
        404/405（后端路由未实现）只提示、不抛异常、不卡界面。
    """
    body = _build_resume_body(interrupt_data, decision)
    url = f"{FASTAPI_BASE_URL}{CONVERSATION_ENDPOINT}/{thread_id}/resume"

    async with client.stream("POST", url, json=body) as response:
        status = response.status_code
        if status in (404, 405):
            await cl.Message(content=RESUME_DISABLED_MESSAGE).send()
            return "unavailable", None, output_msg, step
        if status == 409:
            await cl.Message(content="⚠️ 该审批已被处理或重复提交（409），请重新发起。").send()
            return "conflict", None, output_msg, step
        if status == 410:
            await cl.Message(content="⏰ 审批已过期（410），请重新发起。").send()
            return "expired", None, output_msg, step

        response.raise_for_status()
        next_interrupt, output_msg, step = await _consume_conversation_frames(
            response, output_msg, step
        )
        return "resumed", next_interrupt, output_msg, step


def _register_approval(interrupt_data: dict, shown: set[str]) -> bool:
    """登记已展示的 approval_id：首次返回 True；重复帧记日志后返回 False。

    幂等护栏：同一 approval_id 只允许渲染一张卡片，重复帧（服务端重放/重连）
    直接忽略，既不重复弹卡，也不重复调用 resume。
    """
    approval_id = str(interrupt_data.get("approval_id") or "")
    if not approval_id:
        # 没有 approval_id 的帧无法去重，按首次处理
        return True
    if approval_id in shown:
        logger.info("忽略重复审批帧: approval_id=%s", approval_id)
        return False
    shown.add(approval_id)
    return True


async def _on_conversation_message(message: cl.Message) -> None:  # noqa: C901 - 审批状态机
    """conversation 模式的消息入口：流式渲染 → 审批卡片 → 恢复（若后端支持）。"""
    thread_id = cl.user_session.get(SESSION_THREAD_ID)
    if not thread_id:
        await cl.Message(content="❌ 会话未初始化，请刷新页面。").send()
        return

    user_id = cl.user_session.get(SESSION_USER_NAME) or None
    # 契约要求 user_id 非空：会话里没有用户名时也回退成 thread_id，
    # 保证真实请求体里 user_id 永远不是 null。
    user_id = str(user_id or thread_id)
    output_msg: cl.Message | None = None
    step: cl.Step | None = None

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        try:
            interrupt_data, output_msg, step = await _stream_conversation(
                thread_id=thread_id,
                message=message.content or "",
                client=client,
                user_id=user_id,
                output_msg=output_msg,
                step=step,
            )

            # 审批循环：同一轮里可能连续出现多个需要审批的调用。
            # 同一 approval_id 只问一次：重复帧（服务端重放/重连）复用上次决定，不再发卡片。
            decisions: dict[str, str | None] = {}
            # 幂等护栏：记下已展示过的 approval_id。重复帧直接忽略，
            # 既不重复渲染卡片，也不重复调用 resume（避免同一次执行被放行两次）。
            shown_approval_ids: set[str] = set()
            while interrupt_data:
                if not _register_approval(interrupt_data, shown_approval_ids):
                    break
                decision = await _decide_approval(interrupt_data, decisions)

                if decision == APPROVAL_ACTION_APPROVE:
                    state, interrupt_data, output_msg, step = await _resume_conversation(
                        thread_id=thread_id,
                        interrupt_data=interrupt_data,
                        decision=APPROVAL_ACTION_APPROVE,
                        client=client,
                        output_msg=output_msg,
                        step=step,
                    )
                    if state != "resumed":
                        break
                elif decision == APPROVAL_ACTION_REJECT:
                    await cl.Message(content="🙅 已拒绝执行，本次调用未被发起。").send()
                    # 拒绝也必须走 resume：把 decision 送回后端消费凭据并让图按
                    # reject 分支继续；否则会话停在 interrupted，下一条消息 409。
                    state, interrupt_data, output_msg, step = await _resume_conversation(
                        thread_id=thread_id,
                        interrupt_data=interrupt_data,
                        decision=APPROVAL_ACTION_REJECT,
                        client=client,
                        output_msg=output_msg,
                        step=step,
                    )
                    if state != "resumed":
                        break
                else:
                    # 超时：按 reject 处理，同样走 resume 消费凭据，避免卡在 interrupted
                    await cl.Message(
                        content=(
                            f"⏰ 审批超时（{APPROVAL_TIMEOUT_SECONDS} 秒），"
                            "已自动拒绝，本次调用未被发起。"
                        )
                    ).send()
                    state, interrupt_data, output_msg, step = await _resume_conversation(
                        thread_id=thread_id,
                        interrupt_data=interrupt_data,
                        decision=APPROVAL_ACTION_REJECT,
                        client=client,
                        output_msg=output_msg,
                        step=step,
                    )
                    if state != "resumed":
                        break

        except httpx.HTTPStatusError as exc:
            await _write_error(f"后端错误 (HTTP {exc.response.status_code})", output_msg)
            logger.exception("conversation 模式后端错误")
        except httpx.TimeoutException:
            await _write_error("请求超时，请重试", output_msg)
            logger.exception("conversation 模式超时")
        except Exception as exc:  # noqa: BLE001
            await _write_error(f"服务异常：{exc}", output_msg)
            logger.exception("conversation 模式未知错误")


# ═══════════════════════════════════════════════════════════════
# RAG 文档上传（前端拖拽上传 → 后端异步处理 → 状态查询）
# ═══════════════════════════════════════════════════════════════


async def _handle_uploaded_files(file_elements: list) -> None:
    """处理聊天输入框中拖拽/附带的文件，逐个提交到 RAG 网关。

    说明：Chainlit 2.x 没有 on_file_upload 装饰器，拖拽上传的文件会作为
    cl.File 元素附着在用户消息的 message.elements 中，这里统一读取并提交。
    """
    history: list = cl.user_session.get(SESSION_UPLOAD_HISTORY) or []

    for file_el in file_elements:
        file_name = getattr(file_el, "name", "") or "unknown"
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in ALLOWED_UPLOAD_EXTS:
            await cl.Message(
                content=f"❌ 不支持的文件类型：{file_name}（仅支持 PDF / DOCX / MD / TXT）",
            ).send()
            continue

        # 读取文件字节：优先本地路径，其次内存 content
        if getattr(file_el, "path", None):
            try:
                content = Path(file_el.path).read_bytes()
            except Exception as exc:  # noqa: BLE001
                await cl.Message(content=f"❌ 读取文件失败：{file_name} — {exc}").send()
                continue
        elif getattr(file_el, "content", None):
            raw = file_el.content
            content = raw if isinstance(raw, bytes) else str(raw).encode("utf-8", errors="ignore")
        else:
            await cl.Message(content=f"❌ 无法获取文件内容：{file_name}").send()
            continue

        mime = getattr(file_el, "mime", None) or "application/octet-stream"
        await cl.Message(
            content=f"📤 正在上传 **{file_name}**（{_format_file_size(len(content))}）...",
        ).send()

        try:
            payload = await _ingest_document(file_name, content, mime)
        except Exception as exc:  # noqa: BLE001
            logger.exception("文档上传失败")
            await cl.Message(content=f"❌ 上传失败：{file_name} — {exc}").send()
            continue

        task_id = payload.get("task_id", "")
        status = payload.get("status", "processing")
        history.append(
            {
                "file_name": file_name,
                "task_id": task_id,
                "status": status,
                "time": datetime.now().strftime("%H:%M:%S"),
            }
        )
        cl.user_session.set(SESSION_UPLOAD_HISTORY, history)

        await cl.Message(
            content=(
                f"✅ 文档 **{file_name}** 已提交处理\n\n- task_id: `{task_id}`\n- 状态: `{status}`"
            ),
            actions=[
                cl.Action(name="check_task", payload={"task_id": task_id}, label="🔍 查询状态"),
            ],
        ).send()


@cl.action_callback("check_task")
async def on_check_task(action: cl.Action) -> None:
    """点击「查询状态」按钮时回调，查询 RAG 网关任务状态。"""
    task_id = (action.payload or {}).get("task_id", "")
    if not task_id:
        await cl.Message(content="❌ 无效的 task_id").send()
        return

    try:
        task = await _query_task_status(task_id)
    except Exception as exc:  # noqa: BLE001
        await cl.Message(content=f"❌ 查询失败：{exc}").send()
        return

    status = task.get("status", "UNKNOWN")
    icon = {"COMPLETED": "✅", "FAILED": "❌", "RUNNING": "⏳", "PENDING": "🕓"}.get(status, "❓")
    lines = [
        f"{icon} 任务状态：**{status}**",
        f"- 文件：{task.get('source_file', '')}",
        f"- task_id：`{task_id}`",
    ]
    if task.get("created_at"):
        lines.append(f"- 创建时间：{task['created_at']}")
    if task.get("updated_at"):
        lines.append(f"- 更新时间：{task['updated_at']}")
    if task.get("error_message"):
        lines.append(f"- 错误信息：{task['error_message']}")
    await cl.Message(content="\n".join(lines)).send()


async def _show_upload_history() -> None:
    """展示本次会话内提交过的文档任务（前端本地记录 + 逐条查询状态）。

    历史记录功能需后端配合：RAG 网关当前只有 GET /api/v1/tasks/{task_id}，
    没有 GET /api/v1/tasks 列表端点，因此这里基于会话内记录的 task_id
    逐条查询。若需跨会话/持久化历史，建议后端增加
    GET /api/v1/tasks?kb_id=... 分页接口。
    """
    history: list = cl.user_session.get(SESSION_UPLOAD_HISTORY) or []
    if not history:
        await cl.Message(content="📂 本次会话暂无文档导入记录。").send()
        return

    rows = []
    for item in history:
        task_id = item.get("task_id", "")
        file_name = item.get("file_name", "")
        uploaded = item.get("time", "")
        status = item.get("status", "processing")
        try:
            task = await _query_task_status(task_id)
            status = task.get("status", status)
        except Exception:  # noqa: BLE001
            pass
        rows.append(f"| {file_name} | {status} | {uploaded} | `{task_id}` |")

    header = "| 文件名 | 状态 | 提交时间 | task_id |\n|---|---|---|---|"
    await cl.Message(content="📂 **本次会话导入记录**\n\n" + header + "\n" + "\n".join(rows)).send()


# ═══════════════════════════════════════════════════════════════
# Chainlit 生命周期 — on_chat_start
# ═══════════════════════════════════════════════════════════════


async def _send_welcome(user_name: str) -> None:
    await cl.Message(
        content=(
            f"👋 欢迎，**{user_name}**！\n\n"
            "我是自适应搜索助手，基于 LangGraph 构建。\n"
            "输入你的问题，我会自动搜索、评估并生成报告。\n\n"
            "---\n\n"
            "📁 **知识库管理**\n\n"
            "点击下方链接打开独立上传页面（支持 PDF / DOCX / MD / TXT）：\n\n"
            f"[📁 打开知识库管理]({RAG_GATEWAY_BASE_URL}/static/upload.html)"
        )
    ).send()


@cl.on_chat_start
async def on_chat_start() -> None:
    """会话启动：身份识别 → 欢迎。"""
    if DEVELOPER_MODE:
        user_name = "developer"
        thread_id = f"test_{uuid.uuid4().hex[:8]}"
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
    # 模式开关：默认 deep_research（旧流程），用户可在设置面板切到 conversation
    cl.user_session.set(SESSION_MODE, DEFAULT_CHAT_MODE)
    await _make_chat_settings().send()

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
async def on_message(message: cl.Message) -> None:  # noqa: C901
    """用户发送消息时触发。

    0. 若消息附带文件（cl.File 元素，拖拽/附件上传）→ 提交 RAG 网关摄入；
       发送 /history → 展示本次会话导入记录
    1. 首轮执行：「思考过程」Step 在首个思考事件到达时由 _stream_agent 惰性创建
    2. 最终报告消息在首个输出事件到达时惰性创建（Step 下方）
    3. 若触发中断 → 按类型分流：
       - cost_confirm：高成本搜索确认（确认继续 / 取消）
       - need_more_info：缺失信息补全询问
    4. 第二阶段（确认继续）传入 lazy_create_step=True，由 _stream_agent 在首个
       思考事件到达时创建并发送 Step，使 Step.send() 与 Message.stream_start()
       紧密执行；取消路径 suppress_thinking=True，不渲染思考过程/KPI

    模式分流：conversation 模式在函数最前面整体交给 _on_conversation_message，
    旧 deep_research 流程一行不改。
    """
    if _current_mode() == MODE_CONVERSATION:
        await _on_conversation_message(message)
        return

    thread_id: str = cl.user_session.get(SESSION_THREAD_ID)
    if not thread_id:
        await cl.Message(content="❌ 会话未初始化，请刷新页面。").send()
        return

    # ── RAG 文档上传已迁移至独立页面：左侧入口 → /static/upload.html ──
    # 原输入框文件分流逻辑保留备用（如需恢复，取消以下注释即可）：
    # file_elements = [el for el in (message.elements or []) if isinstance(el, cl.File)]
    # if file_elements:
    #     await _handle_uploaded_files(file_elements)
    #     return

    # ── 会话内导入记录（历史记录功能需后端配合：RAG 网关暂无
    #    GET /api/v1/tasks 列表端点，这里基于会话内 task_id 逐条查询）──
    if (message.content or "").strip() == "/history":
        await _show_upload_history()
        return

    # 1. 不预先创建 Step；_stream_agent 会在首个真实思考事件到达时惰性创建，
    #    避免取消等场景出现空的「思考过程」折叠块
    output_msg: cl.Message | None = None

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
    ) as client:
        try:
            # 第一阶段：初始搜索 + 评估
            interrupt_data, output_msg = await _stream_agent(
                thread_id=thread_id,
                message=message.content,
                client=client,
                output_msg=output_msg,
                lazy_create_step=True,
            )

            # 人在环：按 interrupt 类型分流（缺失信息补全 / 高成本搜索确认）
            if interrupt_data:
                interrupt_type = interrupt_data.get("type", "")
                if interrupt_type == "cost_confirm":
                    feedback = await _ask_cost_confirm(interrupt_data)
                    if feedback is None:
                        # 超时未响应视为取消，发送 cancelled 让图优雅结束
                        feedback = "cancelled"
                else:
                    feedback = await _ask_missing_info(interrupt_data)

                if feedback and feedback != "cancelled":
                    # 不预先创建 Step，让 _stream_agent 在收到第一个思考事件时惰性创建
                    # 这样 Step.send() 与 Message.stream_start() 在同一个事件循环中执行
                    _, output_msg = await _stream_agent(
                        thread_id=thread_id,
                        message="",
                        client=client,
                        thinking_step=None,  # 不传入 Step
                        resume_value=feedback,
                        output_msg=output_msg,
                        lazy_create_step=True,  # 让 _stream_agent 自动创建和管理 Step
                    )
                elif feedback == "cancelled":
                    # 取消：仍恢复图以获取 writer 的取消报告，但不渲染思考过程/KPI
                    _, output_msg = await _stream_agent(
                        thread_id=thread_id,
                        message="",
                        client=client,
                        resume_value=feedback,
                        output_msg=output_msg,
                        suppress_thinking=True,
                    )
                else:
                    await cl.Message(content="已取消补全，会话保持中断，可稍后继续。").send()

        except httpx.HTTPStatusError as exc:
            if output_msg is None:
                output_msg = cl.Message(content="")
                await output_msg.send()
            output_msg.content = f"❌ 后端错误 (HTTP {exc.response.status_code})"
            await output_msg.update()
            logger.exception("on_message 后端错误")
        except httpx.TimeoutException:
            if output_msg is None:
                output_msg = cl.Message(content="")
                await output_msg.send()
            output_msg.content = "❌ 请求超时，请重试"
            await output_msg.update()
            logger.exception("on_message 超时")
        except Exception as exc:
            if output_msg is None:
                output_msg = cl.Message(content="")
                await output_msg.send()
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
