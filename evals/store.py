"""评测结果落库：`evals/results/<run_id>/` 下的 meta.json + per_case.json。

目录约定（不进 git，见 .gitignore）：

    evals/results/<run_id>/
        meta.json       # run_id / 时间 / config / 模型 / 裁判 / 样本 / commit / 开关 / schema 指纹
        per_case.json   # 每题：case_id / question / answer_digest / score / latency_ms / tokens / ...
        report.md       # 由 evals.report 生成

本模块只做文件 IO；门禁判定在 evals.gate（纯函数）。测试/冒烟可用
`EVAL_RESULTS_DIR` 或直接改 `store.RESULTS_DIR` 指向临时目录。
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals.gate import Run

_PROJ = Path(__file__).resolve().parents[1]

#: 结果根目录；冒烟测试通过改这个变量隔离产物。
RESULTS_DIR = Path(os.environ.get("EVAL_RESULTS_DIR") or (_PROJ / "evals" / "results"))

#: meta.json 必含字段（验收要求：时间 / 配置 / 模型 / 裁判 / 样本 / commit / 开关 / schema 指纹）。
META_REQUIRED = (
    "run_id",
    "created_at",
    "config_id",
    "model",
    "judge_model",
    "n_cases",
    "git_commit",
    "flags",
    "schema_sha256",
)

#: per_case 每题必含字段。
CASE_REQUIRED = (
    "case_id",
    "question",
    "answer_digest",
    "score",
    "latency_ms",
    "tokens",
    "tool_calls",
    "error",
)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class StoreError(ValueError):
    """结果库读写失败：路径不合法、缺文件、缺字段、JSON 坏了。"""


def new_run_id() -> str:
    """时间戳 + 短随机哈希：同秒多次运行也不互撞。"""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def git_commit() -> str:
    """当前代码版本（短 commit）；取不到时返回 unknown，不让落库失败。"""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_PROJ,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _run_dir(run_id: str) -> Path:
    if not _RUN_ID_RE.match(run_id or ""):
        raise StoreError(f"run_id 不合法：{run_id!r}（只允许字母、数字、点、下划线、连字符）")
    return RESULTS_DIR / run_id


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _validate_meta(meta: Mapping[str, Any]) -> None:
    missing = [key for key in META_REQUIRED if key not in meta]
    if missing:
        raise StoreError(f"meta 缺字段：{', '.join(missing)}")
    if not isinstance(meta["n_cases"], int) or meta["n_cases"] < 0:
        raise StoreError(f"meta.n_cases 必须是非负整数，实际 {meta['n_cases']!r}")
    if not isinstance(meta["flags"], Mapping):
        raise StoreError(f"meta.flags 必须是对象，实际 {type(meta['flags']).__name__}")
    for key in ("run_id", "created_at", "config_id", "model", "git_commit", "schema_sha256"):
        if not str(meta[key] or "").strip():
            raise StoreError(f"meta.{key} 不能为空")


def _validate_cases(per_case: list[dict[str, Any]]) -> None:
    if not isinstance(per_case, list):
        raise StoreError("per_case 必须是数组")
    for index, case in enumerate(per_case, start=1):
        if not isinstance(case, Mapping):
            raise StoreError(f"per_case 第 {index} 条不是对象")
        missing = [key for key in CASE_REQUIRED if key not in case]
        if missing:
            raise StoreError(f"per_case 第 {index} 条缺字段：{', '.join(missing)}")


def save_run(run_id: str, meta: Mapping[str, Any], per_case: list[dict[str, Any]]) -> Path:
    """写一次 run 的 meta.json + per_case.json，返回 run 目录。"""
    _validate_meta(meta)
    _validate_cases(per_case)
    if meta["run_id"] != run_id:
        raise StoreError(f"meta.run_id={meta['run_id']!r} 与参数 run_id={run_id!r} 不一致")
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "meta.json", dict(meta))
    _write_json(run_dir / "per_case.json", list(per_case))
    return run_dir


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StoreError(f"找不到 {path}") from None
    except json.JSONDecodeError as exc:
        raise StoreError(f"{path} 不是合法 JSON：{exc}") from None
    except OSError as exc:
        raise StoreError(f"读 {path} 失败：{exc}") from None


def load_run(run_id: str) -> Run:
    """按 run_id 读一次 run；目录/文件缺失或结构不对时给可读错误。"""
    run_dir = _run_dir(run_id)
    if not run_dir.is_dir():
        raise StoreError(f"找不到 run {run_id!r}（目录 {run_dir}）")
    meta = _read_json(run_dir / "meta.json")
    per_case = _read_json(run_dir / "per_case.json")
    if not isinstance(meta, Mapping):
        raise StoreError(f"{run_dir / 'meta.json'} 必须是对象")
    if not isinstance(per_case, list):
        raise StoreError(f"{run_dir / 'per_case.json'} 必须是数组")
    return Run(run_id=run_id, meta=meta, per_case=tuple(per_case))


def load_run_path(path: str | Path) -> Run:
    """从 run 目录或 meta.json 路径读一次 run。"""
    target = Path(path).expanduser()
    if target.is_file() and target.name == "meta.json":
        target = target.parent
    if not target.is_dir():
        raise StoreError(f"路径不存在或不是 run 目录：{target}")
    run_id = target.name
    meta = _read_json(target / "meta.json")
    per_case = _read_json(target / "per_case.json")
    if not isinstance(meta, Mapping) or not isinstance(per_case, list):
        raise StoreError(f"{target} 不是合法的 run 目录")
    return Run(run_id=run_id, meta=meta, per_case=tuple(per_case))


def resolve_run_ref(ref: str) -> Run:
    """解析 `<run_id|path|last>`；last = 最近一次 run（与 compare 无关）。"""
    text = str(ref).strip()
    if not text:
        raise StoreError("run 引用为空")
    if text == "last":
        latest = latest_run()
        if latest is None:
            raise StoreError("结果库里还没有任何 run")
        return latest
    candidate = Path(text).expanduser()
    if candidate.exists():
        return load_run_path(candidate)
    return load_run(text)


def _created_key(run: Run) -> str:
    return str(run.meta.get("created_at") or run.run_id)


def list_runs() -> list[Run]:
    """列出全部 run，按时间从旧到新；坏目录跳过并在 stderr 提示。"""
    if not RESULTS_DIR.is_dir():
        return []
    runs: list[Run] = []
    for entry in RESULTS_DIR.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if not (entry / "meta.json").is_file():
            # 允许 results 下放子目录（如 CI 的 ci-smoke），不当成坏 run 报警
            continue
        try:
            runs.append(load_run(entry.name))
        except StoreError as exc:
            print(f"warn: 跳过结果目录 {entry.name}：{exc}", file=sys.stderr)
    runs.sort(key=_created_key)
    return runs


def latest_run() -> Run | None:
    """最近一次 run；没有则 None。"""
    runs = list_runs()
    return runs[-1] if runs else None
