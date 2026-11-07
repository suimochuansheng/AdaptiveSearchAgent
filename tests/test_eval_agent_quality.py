"""Agent 端到端评测脚本的纯函数测试（阶段 B4）。

不连网络、不调模型：只测 SSE 解析、judge 输出解析、汇总与对比。
"""

import json

from scripts.eval_agent_quality import (
    build_judge_prompt,
    compare_reports,
    parse_judge_score,
    parse_sse_frames,
    summarize,
)


def _sse(event: dict) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False)


def test_parse_sse_collects_final_and_tools() -> None:
    lines = [
        _sse({"type": "token", "content": "1.7"}),
        _sse({"type": "tool_call", "tools": [{"name": "calculator", "args": {}}]}),
        _sse({"type": "final", "content": "1.77"}),
    ]

    parsed = parse_sse_frames(lines)

    assert parsed["answer"] == "1.77"
    assert parsed["tools"] == ["calculator"]
    assert parsed["errors"] == []


def test_parse_sse_records_errors() -> None:
    parsed = parse_sse_frames([_sse({"type": "error", "message": "boom"})])

    assert parsed["errors"] == ["boom"]
    assert parsed["answer"] == ""


def test_parse_sse_ignores_malformed_lines() -> None:
    parsed = parse_sse_frames(
        ["data: {不是 JSON", "event: ping", _sse({"type": "final", "content": "ok"})]
    )

    assert parsed["answer"] == "ok"


def test_build_judge_prompt_contains_all_parts() -> None:
    prompt = build_judge_prompt("问题X", "参考答案Y", "系统回答Z")

    assert "问题X" in prompt and "参考答案Y" in prompt and "系统回答Z" in prompt


def test_parse_judge_score_plain_and_fenced() -> None:
    assert parse_judge_score('{"score": 8, "reason": "ok"}')[0] == 8.0
    assert parse_judge_score('```json\n{"score": 6}\n```')[0] == 6.0


def test_parse_judge_score_falls_back_to_regex() -> None:
    score, _ = parse_judge_score('评价如下：{"score": 7, "reason": "基本正确"} 完毕')

    assert score == 7.0


def test_parse_judge_score_clamps_and_rejects() -> None:
    assert parse_judge_score('{"score": 99}')[0] == 10.0
    assert parse_judge_score("完全无法解析") == (None, "")


def test_summarize_counts_failures() -> None:
    summary = summarize(
        [
            {"id": "q1", "score": 8},
            {"id": "q2", "score": 6},
            {"id": "q3", "score": None},
        ]
    )

    assert summary["total"] == 3
    assert summary["scored"] == 2
    assert summary["failed"] == 1
    assert summary["mean_score"] == 7.0


def test_compare_reports_computes_deltas() -> None:
    baseline = {
        "summary": {"mean_score": 7.0},
        "results": [
            {"id": "q1", "score": 8},
            {"id": "q2", "score": 6},
        ],
    }
    current = {
        "summary": {"mean_score": 7.5},
        "results": [
            {"id": "q1", "score": 8},
            {"id": "q2", "score": 7},
        ],
    }

    diff = compare_reports(baseline, current)

    assert diff["mean_delta"] == 0.5
    assert diff["improved"] == 1
    assert diff["regressed"] == 0
    assert {row["id"]: row["delta"] for row in diff["rows"]} == {"q1": 0.0, "q2": 1.0}


def test_compare_uses_only_common_questions() -> None:
    """题集增删后，均值只按两边都有的题算——避免"加了新题就不可比"。"""
    baseline = {
        "summary": {"mean_score": 8.0},
        "results": [{"id": "q1", "score": 8}, {"id": "q2", "score": 8}],
    }
    current = {
        "summary": {"mean_score": 7.0},
        "results": [{"id": "q1", "score": 8}, {"id": "q3", "score": 10}],
    }

    diff = compare_reports(baseline, current)

    assert diff["common"] == 1
    assert diff["only_baseline"] == ["q2"]
    assert diff["only_current"] == ["q3"]
    assert diff["mean_delta"] == 0.0  # q1 两边都是 8，新题 q3 不参与
