"""Langfuse 连接冒烟测试（Smoke Test）。

用途：验证当前 `.env` 配置能否成功向 Langfuse 平台上传一条 Trace 与 Generation。

运行方式（任选其一）：
    1. 直接运行： python tests/test_langfuse_connection.py
    2. pytest：    pytest tests/test_langfuse_connection.py -s

前置条件：
    - `.env` 中 LANGFUSE_ENABLED=true
    - LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 非空
    - 已安装 langfuse（>=4.x，见 pyproject.toml；注意 requirements.txt 目前缺失该依赖）

说明：
    本脚本使用 langfuse v4 的 OpenTelemetry 风格 API（`start_as_current_observation`）。
    v4 已移除旧的 `langfuse.trace(...).generation(...)` 调用方式，请勿沿用 v2/v3 写法。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# 将项目根目录 .env 注入 os.environ（langfuse SDK 从环境变量读取配置）
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


def _check_prerequisites() -> tuple[bool, str]:
    """校验 Langfuse 配置是否就绪，返回 (是否就绪, 说明)。"""
    enabled = os.getenv("LANGFUSE_ENABLED", "").lower() in ("true", "1", "yes", "on")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    host = os.getenv("LANGFUSE_HOST", os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"))

    if not enabled:
        return False, "LANGFUSE_ENABLED 未开启，跳过冒烟测试"
    if not public_key or not secret_key:
        return False, "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 为空，跳过冒烟测试"
    return True, host


def run_smoke_test() -> int:
    """执行冒烟测试，返回进程退出码（0=成功，1=失败）。"""
    from langfuse import Langfuse  # 延迟导入，便于在未安装时给出清晰提示

    ok, detail = _check_prerequisites()
    if not ok:
        print(f"[SKIP] {detail}")
        return 0  # 未配置不视为失败，避免 CI 误报

    print(f"[INFO] Langfuse host = {detail}")

    client = Langfuse()

    # 1) 认证检查：验证密钥 + 网络连通性
    print("[1/3] 正在执行 auth_check() ...")
    if not client.auth_check():
        print("[FAIL] 认证失败：请检查 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST")
        return 1
    print("[OK] 认证通过")

    # 2) 上传一条 Trace + 一条 Generation
    print("[2/3] 正在上传 Trace + Generation ...")
    trace_id: str | None = None
    try:
        with client.start_as_current_observation(
            as_type="span",
            name="smoke-test-trace",
            input={"purpose": "langfuse connectivity smoke test"},
            metadata={"source": "tests/test_langfuse_connection.py"},
        ):
            trace_id = client.get_current_trace_id()
            with client.start_as_current_observation(
                as_type="generation",
                name="smoke-test-generation",
                model="smoke-test-model",
                input="ping",
                output="pong",
                usage_details={"input": 4, "output": 4, "total": 8},
            ):
                pass
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 上传失败：{exc!r}")
        return 1

    # 3) 刷新缓冲，确保异步上报完成
    client.flush()
    print("[3/3] flush 完成")

    url = client.get_trace_url(trace_id=trace_id)
    print(f"[OK] 冒烟测试通过。trace_id = {trace_id}")
    if url:
        print(f"[INFO] 查看 Trace：{url}")
    return 0


@pytest.mark.e2e
def test_langfuse_connection() -> None:
    """pytest 入口：未配置时跳过，配置了就绪时执行真实连通性测试。"""
    ok, detail = _check_prerequisites()
    if not ok:
        pytest.skip(detail)
    assert run_smoke_test() == 0


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())
