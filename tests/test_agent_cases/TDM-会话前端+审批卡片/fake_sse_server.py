#!/usr/bin/env python
"""TDM 手测用假 SSE 后端 —— 只用于给前端注入 interrupt 审批帧。

背景：真实 `/chat/conversation` 目前只吐 token/tool_call/ping/error/final，
还没有审批中断；而 T9-2 的 resume 路由也尚未落地。为了手测审批卡片，
这里起一个**独立**的假后端（默认 8002），前端用 `FASTAPI_BASE_URL` 指过来即可。

不触碰 `api_main.py` / `src/`，也不影响真实后端的任何行为。

运行：
  poetry run python "tests/test_agent_cases/TDM-会话前端+审批卡片/fake_sse_server.py"
  # 或
  poetry run uvicorn "fake_sse_server:app" --app-dir "tests/test_agent_cases/TDM-会话前端+审批卡片" --port 8002

会话端点行为（按 message 关键字区分场景）：
  - message 含「深度调研」或「审批」→ token → tool_call → interrupt（弹审批卡片）
  - 其它                                → token 流 + final（验证去重）

审批恢复：**故意不实现** `/resume`，FastAPI 会天然返回 404，
正好用来验收「审批功能未启用（等待 T9-2）」提示。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

app = FastAPI(title="TDM fake SSE backend")

_NORMAL_TOKENS = ["你好", "！我是会话助手，", "可以帮你搜索资料、", "做深度调研。"]
_INTERRUPT_TOKENS = ["好的，", "我先做一次深度调研，"]


def _frame(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _approval_payload() -> dict:
    expires_at = (datetime.now().astimezone() + timedelta(minutes=10)).isoformat(timespec="seconds")
    return {
        "approval_id": "apr_fake_001",
        "tool_call_id": "call_fake_001",
        "tool": "deep_research",
        "args_digest": "query=量子计算最新进展&max_results=5",
        "cost_estimate": 0.1,
        "budget_remaining": 2.0,
        "permission": "dangerous",
        "expires_at": expires_at,
    }


async def _normal_stream() -> AsyncIterator[str]:
    for token in _NORMAL_TOKENS:
        yield _frame({"type": "token", "content": token})
        await asyncio.sleep(0.12)
    yield _frame({"type": "ping"})
    # final 故意与 token 拼出的全文完全相同：前端若重复输出，肉眼立即可见
    yield _frame({"type": "final", "content": "".join(_NORMAL_TOKENS)})


async def _interrupt_stream() -> AsyncIterator[str]:
    for token in _INTERRUPT_TOKENS:
        yield _frame({"type": "token", "content": token})
        await asyncio.sleep(0.12)
    yield _frame(
        {
            "type": "tool_call",
            "tools": [{"name": "deep_research", "args": {"query": "量子计算最新进展"}}],
        }
    )
    await asyncio.sleep(0.12)
    yield _frame({"type": "interrupt", "data": _approval_payload()})


@app.post("/chat/conversation")
async def chat_conversation(request: Request) -> StreamingResponse:
    body = await request.json()
    message = str(body.get("message") or "")
    if "深度调研" in message or "审批" in message:
        return StreamingResponse(_interrupt_stream(), media_type="text/event-stream")
    return StreamingResponse(_normal_stream(), media_type="text/event-stream")


@app.post("/chat/stream")
async def chat_stream(request: Request) -> StreamingResponse:
    """假实例上也保留旧端点，切回 deep_research 时不至于直接 404。"""

    async def _stream() -> AsyncIterator[str]:
        yield _frame({"type": "status", "confidence": 0.5, "iteration": 1})
        yield _frame({"type": "final", "content": "（假后端）旧流程 final 报告"})
        yield _frame({"type": "kpi", "data": {"elapsed_seconds": 0.1}})

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "tdm-fake-sse"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
