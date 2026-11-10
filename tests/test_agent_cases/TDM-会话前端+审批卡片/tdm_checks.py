#!/usr/bin/env python
"""TDM 会话前端 + 审批卡片 —— 测试窗口独立验收脚本（G9 / T-demo）。

用法（仓库根目录）：
  poetry run python "tests/test_agent_cases/TDM-会话前端+审批卡片/tdm_checks.py"

退出码：
  0 = 全部检查通过
  1 = 至少一项失败（A/B 必过项失败即判 T-demo 不通过）

存在原因：
  开发窗口自带的 10 条自检只证明"开发者自己写的行为"，其中两条还把缺陷写成了期望
  （user_id=None、resume.token=""）。本脚本由测试窗口独立重写：用进程内假 Chainlit
  对象 + 假 SSE 流，逐条对照车道包第 2/3 节冻结契约与验收卡 A/B/C，不依赖真模型、
  不依赖后端端口、不出网。

台账口径：TDM-001~010 为开发窗口自检行（保持原样，不改他人行）；TDM-011~020 为
测试窗口独立验收行（本脚本 [PASS]/[FAIL] 明细即其证据）。

判定口径：
  - A1~A8 为卡面必过项；B1~B3 为不变量；C1~C3 只记 open_issue，不单独打回。
  - B1 按卡面字面规则执行：config.py / pyproject.toml 只要工作树有 diff 即 FAIL；
    证据里同时给出归因（本工作树改动的来源），供主计划窗口裁决。

编码：UTF-8（台账 CSV 另用 GB18030，本脚本不写台账）。
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import frontend.app as _frontend_app  # noqa: E402

# 本脚本大量 monkeypatch frontend.app 的模块级符号（_ask_approval、MODE_* 等），
# 这些符号随 TDM 前端改动引入；用例提交时前端改动尚未入库，pre-commit 会先 stash
# 未暂存文件再跑 mypy，静态检查会按 HEAD 版本的 frontend/app.py 误报 attr-defined。
# 因此这里按 Any 使用，属性真实存在与否交由脚本运行期断言负责。
app: Any = _frontend_app

# ═══════════════════════════════════════════════════════════════
# 假对象：Chainlit Message / Step / AskActionMessage / httpx
# ═══════════════════════════════════════════════════════════════

SENT_MESSAGES: list[FakeMessage] = []


class FakeMessage:
    """记录 send/update/stream_token，用于断言"用户最终看到什么"。"""

    def __init__(self, content: str = "", **kwargs: Any) -> None:
        self.content = content
        self.sent = False
        self.updates = 0
        self.tokens: list[str] = []
        SENT_MESSAGES.append(self)

    async def send(self) -> None:
        self.sent = True

    async def update(self) -> None:
        self.updates += 1

    async def stream_token(self, text: str) -> None:
        self.tokens.append(text)
        self.content += text


class FakeStep:
    """记录思考轨迹；tool_call 必须落在 stream_token 里才叫"可见"。"""

    def __init__(self, name: str = "", type: str = "run", **kwargs: Any) -> None:
        self.name = name
        self.type = type
        self.output = ""
        self.sent = False
        self.updates = 0
        self.tokens: list[str] = []

    async def send(self) -> None:
        self.sent = True

    async def update(self) -> None:
        self.updates += 1

    async def stream_token(self, text: str) -> None:
        self.tokens.append(text)
        self.output += text


class FakeAskActionMessage:
    """按队列返回点击结果；队列空 = 超时（返回 None）。记录 actions/timeout。"""

    queue: list[dict | None] = []
    last_instance: FakeAskActionMessage | None = None

    def __init__(self, content: str = "", actions: list | None = None, timeout: int = 0) -> None:
        self.content = content
        self.actions = actions or []
        self.timeout = timeout
        FakeAskActionMessage.last_instance = self

    async def send(self) -> dict | None:
        return FakeAskActionMessage.queue.pop(0) if FakeAskActionMessage.queue else None


class FakeResponse:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = app.httpx.Request("POST", "http://fake/resume")
            response = app.httpx.Response(self.status_code, request=request)
            raise app.httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=response
            )

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


class FakeClient:
    """直接传给 _stream_conversation 的最小 httpx.AsyncClient 替身。"""

    def __init__(self, responses: list[FakeResponse] | None = None) -> None:
        self.calls: list[dict] = []
        self.responses = list(responses or [])

    def stream(self, method: str, url: str, json: dict | None = None) -> FakeResponse:
        self.calls.append({"method": method, "url": url, "json": json})
        return self.responses.pop(0)


class FakeAsyncClient(FakeClient):
    """满足 `async with httpx.AsyncClient(...) as client:` 的替身。"""

    responses: list[FakeResponse] = []
    instances: list[FakeAsyncClient] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(FakeAsyncClient.responses)
        FakeAsyncClient.instances.append(self)

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


class FakeUserSession:
    def __init__(self, values: dict | None = None) -> None:
        self.values = dict(values or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value


# 原对象快照：每个用例开始都还原，避免用例间串味。
_ORIG_ASK_APPROVAL = app._ask_approval
_ORIG_ON_CONVERSATION = app._on_conversation_message
_ORIG_ASYNC_CLIENT = app.httpx.AsyncClient
_ORIG_TIMEOUT = app.APPROVAL_TIMEOUT_SECONDS


def _sse(*events: dict) -> list[str]:
    return [f"data: {json.dumps(event, ensure_ascii=False)}" for event in events]


INTERRUPT_DATA = {
    "approval_id": "apr_demo_001",
    "tool_call_id": "call_demo_001",
    "tool": "deep_research",
    "args_digest": "query=量子计算&max=5",
    "cost_estimate": 0.1,
    "budget_remaining": 2.0,
    "permission": "dangerous",
    "expires_at": "2026-11-10T20:00:00+08:00",
}


def _reset() -> None:
    SENT_MESSAGES.clear()
    FakeAskActionMessage.queue.clear()
    FakeAskActionMessage.last_instance = None
    FakeAsyncClient.responses = []
    FakeAsyncClient.instances = []
    app.cl.Message = FakeMessage
    app.cl.Step = FakeStep
    app.cl.AskActionMessage = FakeAskActionMessage
    app.cl.user_session = FakeUserSession()
    app._ask_approval = _ORIG_ASK_APPROVAL
    app._on_conversation_message = _ORIG_ON_CONVERSATION
    app.httpx.AsyncClient = _ORIG_ASYNC_CLIENT
    app.APPROVAL_TIMEOUT_SECONDS = _ORIG_TIMEOUT


def _run_drive(message: str = "需要深度调研") -> None:
    asyncio.run(app._on_conversation_message(FakeMessage(message)))


async def _consume(lines: list[str]) -> tuple[Any, Any, Any]:
    client = FakeClient([FakeResponse(lines)])
    return await app._stream_conversation(thread_id="t-x", message="hi", client=client)


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return (proc.stdout or "").strip()


# ═══════════════════════════════════════════════════════════════
# A1：模式开关 + 分流 + 请求体
# ═══════════════════════════════════════════════════════════════


def check_a1a_mode_switch() -> str:
    """默认 deep_research；切到 conversation 后 on_message 必须走新链路。"""
    _reset()
    assert app._current_mode() == app.MODE_DEEP_RESEARCH, "空会话默认模式不是 deep_research"
    app.cl.user_session = FakeUserSession({app.SESSION_MODE: "bogus"})
    assert app._current_mode() == app.MODE_DEEP_RESEARCH, "非法模式未回落 deep_research"

    settings = app._make_chat_settings().settings()
    assert settings[app.SESSION_MODE] == app.MODE_DEEP_RESEARCH, (
        "ChatSettings 初值不是 deep_research"
    )

    called: list[str] = []

    async def _fake_conv(message: Any) -> None:
        called.append(message.content)

    app._on_conversation_message = _fake_conv
    try:
        app.cl.user_session = FakeUserSession({app.SESSION_MODE: app.MODE_CONVERSATION})
        asyncio.run(app.on_message(FakeMessage("hi")))
    finally:
        app._on_conversation_message = _ORIG_ON_CONVERSATION

    assert called == ["hi"], (
        f"conversation 模式未分流到 _on_conversation_message（called={called}）"
    )
    return "默认/非法值均深调研；ChatSettings 初值正确；切 conversation 后 on_message 分流成功"


def check_a1b_request_body() -> str:
    """请求发往 /chat/conversation；user_id 缺省时必须回退 thread_id。"""
    _reset()
    app.cl.user_session = FakeUserSession({app.SESSION_THREAD_ID: "tid-1"})
    FakeAsyncClient.responses = [FakeResponse(_sse({"type": "final", "content": "ok"}))]
    app.httpx.AsyncClient = FakeAsyncClient  # type: ignore[assignment]
    _run_drive("hi")

    client = FakeAsyncClient.instances[-1]
    call = client.calls[0]
    assert call["url"].endswith("/chat/conversation"), f"端点错误: {call['url']}"
    body = call["json"]
    assert body.get("thread_id") == "tid-1", f"thread_id 错误: {body!r}"
    assert body.get("message") == "hi", f"message 错误: {body!r}"
    assert body.get("user_id") == "tid-1", (
        f"user_id 缺省时未回退 thread_id（实发 {body.get('user_id')!r}）"
    )
    return f"POST /chat/conversation body={body!r}"


# ═══════════════════════════════════════════════════════════════
# A2~A4：SSE 帧渲染
# ═══════════════════════════════════════════════════════════════


def check_a1c_explicit_user_id() -> str:
    """A1b 正控：显式 user_id 必须原样发送，不能被回退逻辑覆盖。"""
    _reset()
    app.cl.user_session = FakeUserSession(
        {app.SESSION_THREAD_ID: "tid-1c", app.SESSION_USER_NAME: "developer"}
    )
    FakeAsyncClient.responses = [FakeResponse(_sse({"type": "final", "content": "ok"}))]
    app.httpx.AsyncClient = FakeAsyncClient  # type: ignore[assignment]
    _run_drive("hi")
    body = FakeAsyncClient.instances[-1].calls[0]["json"]
    assert body.get("user_id") == "developer", f"显式 user_id 被回退覆盖: {body.get('user_id')!r}"

    # 直接调用路径同样保留显式值
    _reset()
    client = FakeClient([FakeResponse(_sse({"type": "final", "content": "ok"}))])
    asyncio.run(
        app._stream_conversation(
            thread_id="tid-1c", message="hi", client=client, user_id="u-explicit"
        )
    )
    explicit_body = client.calls[0]["json"]
    assert explicit_body.get("user_id") == "u-explicit", (
        f"直传 user_id 被覆盖: {explicit_body.get('user_id')!r}"
    )
    return "显式 user_id='developer' 原样发送；直传 'u-explicit' 也未被覆盖"


def check_a2_token_final_dedupe() -> str:
    """3 个 token + 同内容 final：最终消息 == 3 段拼接，不是两遍。"""
    _reset()
    lines = _sse(
        {"type": "token", "content": "你"},
        {"type": "token", "content": "好"},
        {"type": "token", "content": "！"},
        {"type": "final", "content": "你好！"},
    )
    _, output_msg, _ = asyncio.run(_consume(lines))
    assert output_msg is not None, "3 个 token 后没有主消息"
    assert output_msg.tokens == ["你", "好", "！"], f"token 段数不对: {output_msg.tokens!r}"
    assert output_msg.content == "你好！", f"最终内容不是拼接结果: {output_msg.content!r}"
    assert output_msg.content.count("你好！") == 1, f"final 被二次追加: {output_msg.content!r}"
    assert output_msg.updates >= 1, "final 没有 trigger update() 收尾"
    return "3 token 流式；final 仅 update，content=='你好！' 且只出现 1 次"


def check_a2b_final_only() -> str:
    """一个 token 都没收到时，final 才允许把 content 写进消息（且只写一次）。"""
    _reset()
    _, output_msg, _ = asyncio.run(_consume(_sse({"type": "final", "content": "完整回答"})))
    assert output_msg is not None, "零 token 时 final 没有输出消息"
    assert output_msg.content == "完整回答", f"final 兜底内容错误: {output_msg.content!r}"
    assert output_msg.tokens == [], "final 兜底路径不应再走 stream_token"
    assert output_msg.sent is True, "兜底消息未发送"
    return "零 token 时 final 写入 '完整回答'，无 stream_token 追加"


def check_a3a_tool_call_and_ping() -> str:
    """tool_call 有可见轨迹；仅 ping 时不产生任何用户可见输出。"""
    _reset()
    # 1) ping 单独一帧：不建消息、不发消息
    _, output_msg, _ = asyncio.run(_consume(_sse({"type": "ping"})))
    assert output_msg is None, "ping 创建了主消息"
    assert SENT_MESSAGES == [], f"ping 产生了用户可见输出: {[m.content for m in SENT_MESSAGES]!r}"

    # 2) tool_call 必须写进思考 Step
    lines = _sse(
        {"type": "tool_call", "tools": [{"name": "search_knowledge", "args": {}}]},
        {"type": "token", "content": "答"},
        {"type": "final", "content": "答"},
    )
    _, _, step = asyncio.run(_consume(lines))
    assert step is not None, "tool_call 没有创建思考 Step"
    assert "search_knowledge" in step.output, f"Step 里看不到工具名: {step.output!r}"
    assert step.sent is True, "Step 未 send，用户看不到轨迹"
    return f"ping 无输出；tool_call 轨迹={step.output.strip()!r}"


def check_a3b_error_readable() -> str:
    """error 帧显示为可读错误，不抛异常。"""
    _reset()
    lines = _sse({"type": "token", "content": "半句"}, {"type": "error", "message": "模型调用超时"})
    _, output_msg, _ = asyncio.run(_consume(lines))
    assert output_msg is not None, "error 没有输出消息"
    assert "❌" in output_msg.content, f"错误没有可读标记: {output_msg.content!r}"
    assert "模型调用超时" in output_msg.content, f"错误原文丢失: {output_msg.content!r}"
    return f"错误气泡={output_msg.content!r}"


def check_a4_unknown_type() -> str:
    """未知 type 被忽略且不抛异常，后续帧照常渲染。"""
    _reset()
    lines = _sse(
        {"type": "whatever", "content": "垃圾"},
        {"type": "token", "content": "正常"},
        {"type": "final", "content": "正常"},
    )
    _, output_msg, _ = asyncio.run(_consume(lines))
    assert output_msg is not None and output_msg.content == "正常", (
        f"未知帧影响了正常渲染: {output_msg and output_msg.content!r}"
    )
    assert "垃圾" not in output_msg.content, "未知帧内容泄漏到用户可见输出"
    return "whatever 帧被静默忽略；正常 token/final 渲染不受影响"


# ═══════════════════════════════════════════════════════════════
# A5~A8：审批卡片与恢复
# ═══════════════════════════════════════════════════════════════


def check_a5_approval_card() -> str:
    """卡片五要素 + 两个按钮，且不支持"记住选择"。"""
    _reset()
    FakeAskActionMessage.queue = [{"payload": {"value": app.APPROVAL_ACTION_APPROVE}}]
    decision = asyncio.run(app._ask_approval(INTERRUPT_DATA))
    assert decision == app.APPROVAL_ACTION_APPROVE, f"payload.value 取值失败: {decision!r}"

    card = next((m for m in SENT_MESSAGES if "审批请求" in m.content), None)
    assert card is not None, "没有发出审批卡片消息"
    for field in ("工具名", "参数摘要", "预估成本", "剩余预算", "到期时间"):
        assert field in card.content, f"卡片缺字段: {field}"
    assert "deep_research" in card.content and "0.10 元" in card.content, "工具名/成本未渲染"
    assert "2.00 元" in card.content and "2026-11-10T20:00:00+08:00" in card.content, (
        "预算/到期未渲染"
    )

    ask = FakeAskActionMessage.last_instance
    assert ask is not None, "没有弹 AskActionMessage"
    assert ask.timeout == 120, f"审批超时不是 120s: {ask.timeout}"
    labels = " ".join(str(a.label) for a in ask.actions)
    assert len(ask.actions) == 2, f"按钮数不是 2: {labels}"
    assert "同意执行" in labels and "拒绝" in labels, f"按钮文案不对: {labels}"
    return "工具名/参数摘要/成本 0.10/预算 2.00/到期时间齐全；两按钮；timeout=120"


def check_a6_approve_resume() -> str:
    """同意 → 发 resume；body 含 approval_id/decision/request_id(uuid4)/token。"""
    _reset()
    app.cl.user_session = FakeUserSession(
        {app.SESSION_THREAD_ID: "tid-6", app.SESSION_USER_NAME: "developer"}
    )
    FakeAsyncClient.responses = [
        FakeResponse(_sse({"type": "interrupt", "data": INTERRUPT_DATA})),
        FakeResponse(_sse({"type": "final", "content": "继续完成"})),
    ]
    app.httpx.AsyncClient = FakeAsyncClient  # type: ignore[assignment]
    FakeAskActionMessage.queue = [{"payload": {"value": app.APPROVAL_ACTION_APPROVE}}]
    _run_drive()

    client = FakeAsyncClient.instances[-1]
    assert len(client.calls) == 2, f"HTTP 调用数不对: {client.calls!r}"
    resume = client.calls[1]
    assert resume["url"].endswith("/chat/conversation/tid-6/resume"), (
        f"resume URL 错误: {resume['url']}"
    )
    body = resume["json"]
    assert set(body) == {"approval_id", "decision", "request_id", "token"}, f"字段不全: {body!r}"
    assert body["approval_id"] == "apr_demo_001", f"approval_id 错误: {body!r}"
    assert body["decision"] == "approve", f"decision 错误: {body!r}"
    assert uuid.UUID(body["request_id"]).version == 4, f"request_id 不是 uuid4: {body!r}"
    assert "token" in body, "缺 token 字段"
    assert isinstance(body["token"], str), f"token 不是 str: {body['token']!r}"
    return (
        "POST /chat/conversation/tid-6/resume；四字段齐全；request_id="
        f"{body['request_id']}（v4）；token={body['token']!r}（契约无来源，见 OI-TDM-003）"
    )


def check_a6b_reject_no_resume() -> str:
    """拒绝 → 绝不发 resume，只提示已拒绝。"""
    _reset()
    app.cl.user_session = FakeUserSession(
        {app.SESSION_THREAD_ID: "tid-6b", app.SESSION_USER_NAME: "developer"}
    )
    FakeAsyncClient.responses = [FakeResponse(_sse({"type": "interrupt", "data": INTERRUPT_DATA}))]
    app.httpx.AsyncClient = FakeAsyncClient  # type: ignore[assignment]
    FakeAskActionMessage.queue = [{"payload": {"value": app.APPROVAL_ACTION_REJECT}}]
    _run_drive()

    client = FakeAsyncClient.instances[-1]
    assert len(client.calls) == 1, f"拒绝后仍发出额外请求: {client.calls!r}"
    assert all("/resume" not in c["url"] for c in client.calls), "拒绝后调用了 resume"
    assert any("已拒绝执行" in m.content for m in SENT_MESSAGES), "拒绝后没有可读提示"
    return "仅 1 次 /chat/conversation；无 /resume；提示'已拒绝执行'"


def check_a7_resume_disabled() -> str:
    """404/405 → 显示"审批功能未启用"，不崩、不卡、结束流程。"""
    _reset()
    app.cl.user_session = FakeUserSession(
        {app.SESSION_THREAD_ID: "tid-7", app.SESSION_USER_NAME: "developer"}
    )
    FakeAsyncClient.responses = [
        FakeResponse(_sse({"type": "interrupt", "data": INTERRUPT_DATA})),
        FakeResponse([], status_code=404),
    ]
    app.httpx.AsyncClient = FakeAsyncClient  # type: ignore[assignment]
    FakeAskActionMessage.queue = [{"payload": {"value": app.APPROVAL_ACTION_APPROVE}}]
    _run_drive()
    assert any("审批功能未启用（等待 T9-2）" in m.content for m in SENT_MESSAGES), (
        "404 后没有 审批功能未启用 提示"
    )

    # 405 同样处理
    SENT_MESSAGES.clear()
    FakeAsyncClient.responses = [
        FakeResponse(_sse({"type": "interrupt", "data": INTERRUPT_DATA})),
        FakeResponse([], status_code=405),
    ]
    FakeAskActionMessage.queue = [{"payload": {"value": app.APPROVAL_ACTION_APPROVE}}]
    _run_drive()
    assert any("审批功能未启用（等待 T9-2）" in m.content for m in SENT_MESSAGES), (
        "405 后没有 审批功能未启用 提示"
    )
    return "404 与 405 均显示'审批功能未启用（等待 T9-2）'，无异常、流程正常结束"


def check_a8_timeout_auto_reject() -> str:
    """把超时压到 2s：自动按 reject 处理、有提示、不调用 resume。"""
    _reset()
    app.APPROVAL_TIMEOUT_SECONDS = 2
    app.cl.user_session = FakeUserSession({app.SESSION_THREAD_ID: "tid-8"})
    FakeAsyncClient.responses = [FakeResponse(_sse({"type": "interrupt", "data": INTERRUPT_DATA}))]
    app.httpx.AsyncClient = FakeAsyncClient  # type: ignore[assignment]
    FakeAskActionMessage.queue = []  # 模拟无响应
    _run_drive()

    ask = FakeAskActionMessage.last_instance
    assert ask is not None and ask.timeout == 2, f"超时未压到 2s: {ask and ask.timeout}"
    client = FakeAsyncClient.instances[-1]
    assert len(client.calls) == 1, f"超时后仍调用了后端: {client.calls!r}"
    assert all("/resume" not in c["url"] for c in client.calls), "超时后调用了 resume"
    blob = " ".join(m.content for m in SENT_MESSAGES)
    assert "审批超时" in blob and "自动拒绝" in blob, f"超时提示缺失: {blob!r}"
    return "AskAction timeout=2；无 resume；提示含'审批超时'与'自动拒绝'"


# ═══════════════════════════════════════════════════════════════
# B：不变量
# ═══════════════════════════════════════════════════════════════


def check_b1a_src_clean() -> str:
    """src/ 与 api_main.py 零改动。"""
    dirty = _git("status", "--porcelain", "--", "src/", "api_main.py")
    assert dirty == "", f"src/ 或 api_main.py 有改动:\n{dirty}"
    return "git status src/ api_main.py 为空"


def check_b1b_forbidden_clean() -> str:
    """config.py / pyproject.toml 无 diff（卡面字面门禁）。"""
    dirty = _git("status", "--porcelain", "--", "config.py", "pyproject.toml")
    if dirty:
        cfg = _git("diff", "--", "config.py")
        pyp = _git("diff", "--", "pyproject.toml")
        attribution = []
        if "EVAL_JUDGE_MODEL" in cfg:
            attribution.append("config.py=EVAL_JUDGE_MODEL(G1 评测)")
        if "import-linter" in pyp:
            attribution.append("pyproject.toml=import-linter(T1-1 G2)")
        attr = "；".join(attribution) or "未识别"
        raise AssertionError(
            f"禁改文件有 diff（卡面门禁 FAIL）: {dirty.replace(chr(10), ' | ')}；"
            f"内容归因: {attr}；G9 自身越界=0，待主窗口清树/裁决"
        )
    return "config.py / pyproject.toml 无 diff"


def _top_level_defs(src: str) -> dict[str, str]:
    """顶层函数/类的 AST 摘要，用于判断「旧定义是否被改写」。"""
    tree = ast.parse(src)
    return {
        node.name: ast.dump(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


# 模式分流必须改写的入口，其余旧定义一律不允许变动
_ALLOWED_CHANGED_DEFS = {"on_chat_start", "on_message"}


def check_b2_old_flow_untouched() -> str:
    """旧流程零改动：除模式分流入口外，frontend/app.py 既有定义在 AST 层面未变。

    口径说明：
    - squash 合入后拿不到单独的「修复提交」，只能拿最后一次改到该文件的提交；
    - 行级 numstat 会被格式化误伤（相邻 f-string 拆行也算删除），故改比 AST；
    - 合入必然改写 on_chat_start / on_message 来做模式分流，这两个入口列入允许改写集合。
    """
    rev = _git("log", "-1", "--format=%H", "--", "frontend/app.py")
    assert rev, "frontend/app.py 没有提交记录，无法判断"
    old_src = _git("show", f"{rev}^:frontend/app.py")
    assert old_src, f"取不到 {rev[:7]}^:frontend/app.py，无法判断"
    new_src = (REPO_ROOT / "frontend/app.py").read_text(encoding="utf-8")
    old_defs = _top_level_defs(old_src)
    new_defs = _top_level_defs(new_src)
    assert old_defs, "旧版本里没有可对比的顶层定义"
    missing = sorted(set(old_defs) - set(new_defs))
    assert not missing, f"旧定义被删除: {missing}"
    changed = sorted(
        name
        for name, dump in old_defs.items()
        if new_defs.get(name) != dump and name not in _ALLOWED_CHANGED_DEFS
    )
    assert not changed, f"旧定义被改写: {changed}"
    return (
        f"frontend/app.py 旧 {len(old_defs)} 个顶层定义 AST 未变"
        f"（允许改写: {', '.join(sorted(_ALLOWED_CHANGED_DEFS))}；提交 {rev[:7]}）"
    )


# ═══════════════════════════════════════════════════════════════
# C：边界（记 open_issue，不单独打回）
# ═══════════════════════════════════════════════════════════════


def check_c1_truncated_json() -> str:
    """截断/非法 JSON 行告警后继续，不崩。"""
    _reset()
    logger = logging.getLogger("frontend.app")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        lines = [
            'data: {"type":"tok',  # 截断
            "data: not-json",  # 完全非法
            "data: [1,2,3]",  # 合法 JSON 但不是对象
            'data: {"type":"token","content":"好"}',
            'data: {"type":"final","content":"好"}',
        ]
        _, output_msg, _ = asyncio.run(_consume(lines))
    finally:
        logger.removeHandler(handler)

    assert output_msg is not None and output_msg.content == "好", "坏行之后正常帧没渲染"
    warns = [r for r in records if "无效 JSON" in r.getMessage()]
    assert len(warns) >= 2, f"坏行没有告警（只捕获 {len(warns)} 条）"
    return "2 条坏行告警；后续 token/final 正常渲染为 '好'"


def check_c2_same_thread_two_rounds() -> str:
    """连续两轮复用同一 thread_id，不新建会话。"""
    _reset()
    app.cl.user_session = FakeUserSession({app.SESSION_THREAD_ID: "tid-c2"})
    FakeAsyncClient.responses = [
        FakeResponse(_sse({"type": "final", "content": "第一轮"})),
        FakeResponse(_sse({"type": "final", "content": "第二轮"})),
    ]
    app.httpx.AsyncClient = FakeAsyncClient  # type: ignore[assignment]
    _run_drive("第一轮")
    _run_drive("第二轮")

    threads = [c["json"]["thread_id"] for inst in FakeAsyncClient.instances for c in inst.calls]
    assert threads == ["tid-c2", "tid-c2"], f"两轮 thread_id 不一致: {threads!r}"
    assert app.cl.user_session.get(app.SESSION_THREAD_ID) == "tid-c2", "会话 thread_id 被改写"
    return "两轮均 POST thread_id=tid-c2，会话键未被重建"


def check_c3_duplicate_card() -> str:
    """同一 approval_id 重复出现时只应显示一次卡片。"""
    _reset()
    app.cl.user_session = FakeUserSession({app.SESSION_THREAD_ID: "tid-c3"})
    dup = dict(INTERRUPT_DATA, approval_id="apr_dup_001")
    FakeAsyncClient.responses = [
        FakeResponse(_sse({"type": "interrupt", "data": dup})),
        FakeResponse(_sse({"type": "interrupt", "data": dup})),  # 同一 approval_id 再弹
        FakeResponse(_sse({"type": "final", "content": "done"})),
    ]
    app.httpx.AsyncClient = FakeAsyncClient  # type: ignore[assignment]
    FakeAskActionMessage.queue = [
        {"payload": {"value": app.APPROVAL_ACTION_APPROVE}},
        {"payload": {"value": app.APPROVAL_ACTION_APPROVE}},
    ]
    _run_drive()

    cards = [m for m in SENT_MESSAGES if "审批请求" in m.content]
    assert len(cards) == 1, (
        f"同 approval_id=apr_dup_001 显示了 {len(cards)} 张卡片（未按 approval_id 去重）"
    )
    return "同 approval_id 只显示 1 张卡片"


def _approval_cards() -> list[Any]:
    return [m for m in SENT_MESSAGES if "审批请求" in m.content]


def _all_calls() -> list[dict]:
    return [c for inst in FakeAsyncClient.instances for c in inst.calls]


def _resume_calls() -> list[dict]:
    return [c for c in _all_calls() if "/resume" in c["url"]]


def check_c3b_distinct_approvals() -> str:
    """C3 正控：不同 approval_id 必须各弹一张卡，防"全局只显示一张"的过度去重。"""
    _reset()
    app.cl.user_session = FakeUserSession({app.SESSION_THREAD_ID: "tid-c3b"})
    first = dict(INTERRUPT_DATA, approval_id="apr_a", tool="deep_research")
    second = dict(INTERRUPT_DATA, approval_id="apr_b", tool="calculator")
    FakeAsyncClient.responses = [
        FakeResponse(_sse({"type": "interrupt", "data": first})),
        FakeResponse(_sse({"type": "interrupt", "data": second})),
        FakeResponse(_sse({"type": "final", "content": "done"})),
    ]
    app.httpx.AsyncClient = FakeAsyncClient  # type: ignore[assignment]
    FakeAskActionMessage.queue = [
        {"payload": {"value": app.APPROVAL_ACTION_APPROVE}},
        {"payload": {"value": app.APPROVAL_ACTION_APPROVE}},
    ]
    _run_drive()

    cards = _approval_cards()
    assert len(cards) == 2, f"两个不同 approval_id 只显示了 {len(cards)} 张卡片（过度去重）"
    assert "deep_research" in cards[0].content and "calculator" in cards[1].content, (
        "卡片与 approval_id 的对应关系错乱"
    )
    bodies = [c["json"] for c in _resume_calls()]
    assert [b["approval_id"] for b in bodies] == [
        "apr_a",
        "apr_b",
    ], f"resume 的 approval_id 顺序错误: {[b['approval_id'] for b in bodies]!r}"
    return "apr_a 与 apr_b 各 1 张卡（共 2 张），两次 resume 分别对应"


def check_c3c_duplicate_reuse() -> str:
    """C3 重复帧：同 approval_id 不再弹卡，也不再 resume；reject/超时不重复提示。"""
    # 1) approve：同 approval_id 第二次出现 -> 1 张卡、1 次 resume（重复帧被忽略）
    _reset()
    app.cl.user_session = FakeUserSession({app.SESSION_THREAD_ID: "tid-c3c"})
    dup = dict(INTERRUPT_DATA, approval_id="apr_reuse")
    FakeAsyncClient.responses = [
        FakeResponse(_sse({"type": "interrupt", "data": dup})),
        FakeResponse(_sse({"type": "interrupt", "data": dup})),
        FakeResponse(_sse({"type": "final", "content": "done"})),
    ]
    app.httpx.AsyncClient = FakeAsyncClient  # type: ignore[assignment]
    FakeAskActionMessage.queue = [{"payload": {"value": app.APPROVAL_ACTION_APPROVE}}]
    _run_drive()

    cards = _approval_cards()
    resumes = _resume_calls()
    assert len(cards) == 1, f"同 approval_id 显示了 {len(cards)} 张卡片"
    assert len(resumes) == 1, f"重复帧不应再次 resume（resume 次数={len(resumes)}）"
    assert all(c["json"]["decision"] == "approve" for c in resumes), (
        "resume 的 decision 不是 approve"
    )

    # 2) reject：重复帧不得第二次提示，也不得调用 resume
    _reset()
    app.cl.user_session = FakeUserSession({app.SESSION_THREAD_ID: "tid-c3c-r"})
    FakeAsyncClient.responses = [
        FakeResponse(_sse({"type": "interrupt", "data": dup})),
        FakeResponse(_sse({"type": "interrupt", "data": dup})),
    ]
    app.httpx.AsyncClient = FakeAsyncClient  # type: ignore[assignment]
    FakeAskActionMessage.queue = [{"payload": {"value": app.APPROVAL_ACTION_REJECT}}]
    _run_drive()
    reject_msgs = [m for m in SENT_MESSAGES if "已拒绝执行" in m.content]
    assert len(_approval_cards()) == 1, "reject 路径重复弹卡"
    assert len(reject_msgs) == 1, f"reject 提示重复了 {len(reject_msgs)} 次"
    assert _resume_calls() == [], "reject 路径调用了 resume"

    # 3) 超时：同样只提示一次、不调用 resume
    _reset()
    app.APPROVAL_TIMEOUT_SECONDS = 2
    app.cl.user_session = FakeUserSession({app.SESSION_THREAD_ID: "tid-c3c-t"})
    FakeAsyncClient.responses = [
        FakeResponse(_sse({"type": "interrupt", "data": dup})),
        FakeResponse(_sse({"type": "interrupt", "data": dup})),
    ]
    app.httpx.AsyncClient = FakeAsyncClient  # type: ignore[assignment]
    FakeAskActionMessage.queue = []  # 无响应
    _run_drive()
    timeout_msgs = [m for m in SENT_MESSAGES if "审批超时" in m.content]
    assert len(_approval_cards()) == 1, "超时路径重复弹卡"
    assert len(timeout_msgs) == 1, f"超时提示重复了 {len(timeout_msgs)} 次"
    assert _resume_calls() == [], "超时路径调用了 resume"
    return "同 approval_id：1 卡 + 1 次 resume；重复帧被忽略；reject/超时各 1 次提示、0 resume"


CHECKS: list[tuple[str, str, Any]] = [
    ("TDM-A1a", "模式开关默认 deep_research + conversation 分流", check_a1a_mode_switch),
    ("TDM-A1b", "conversation 请求体 user_id 回退", check_a1b_request_body),
    ("TDM-A1c", "显式 user_id 不被回退覆盖", check_a1c_explicit_user_id),
    ("TDM-A2", "3 token + 同内容 final 去重", check_a2_token_final_dedupe),
    ("TDM-A2b", "final 零 token 兜底", check_a2b_final_only),
    ("TDM-A3a", "tool_call 可见 / ping 静默", check_a3a_tool_call_and_ping),
    ("TDM-A3b", "error 帧可读", check_a3b_error_readable),
    ("TDM-A4", "未知 type 忽略", check_a4_unknown_type),
    ("TDM-A5", "审批卡片五要素 + 两按钮", check_a5_approval_card),
    ("TDM-A6", "同意 -> resume body", check_a6_approve_resume),
    ("TDM-A6b", "拒绝 -> 不发 resume", check_a6b_reject_no_resume),
    ("TDM-A7", "resume 404/405 未启用不崩", check_a7_resume_disabled),
    ("TDM-A8", "2s 超时自动拒绝", check_a8_timeout_auto_reject),
    ("TDM-B1a", "src/api_main.py 零改动", check_b1a_src_clean),
    ("TDM-B1b", "config.py/pyproject.toml 零改动", check_b1b_forbidden_clean),
    ("TDM-B2", "旧流程纯新增", check_b2_old_flow_untouched),
    ("TDM-C1", "截断/非法 JSON 告警不崩", check_c1_truncated_json),
    ("TDM-C2", "同 thread_id 连轮", check_c2_same_thread_two_rounds),
    ("TDM-C3", "同 approval_id 卡片去重", check_c3_duplicate_card),
    ("TDM-C3b", "不同 approval_id 各一张卡", check_c3b_distinct_approvals),
    ("TDM-C3c", "重复帧忽略 / reject 超时不重复", check_c3c_duplicate_reuse),
]


def main() -> int:
    passed = 0
    failed = 0
    for case_id, title, func in CHECKS:
        try:
            detail = func()
        except Exception as exc:  # noqa: BLE001 —— 单条失败不影响其余用例
            failed += 1
            print(f"[FAIL] {case_id} {title} — {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"[PASS] {case_id} {title} — {detail}")
    print(f"RESULT: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
