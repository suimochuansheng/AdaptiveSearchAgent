"""评测报告输出路径（阶段 B4 后：报告不再写进仓库）。

为什么单独抽出来：
- 文档已移出仓库（避免误提交），报告也应跟着写到外部文档目录；
- 两个评测脚本 + 一个 shell 脚本共用同一套默认值，避免各写各的；
- 允许 EVAL_REPORT_DIR 环境变量覆盖（CI / 临时目录用）。
"""

from __future__ import annotations

import os
from pathlib import Path

from config import settings

# src/utils/report_paths.py → 仓库根
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REPORT_DIR = REPO_ROOT.parent / "文档" / "项目" / "AdaptiveSearchAgent" / "评测报告"


def eval_report_dir() -> Path:
    """返回评测报告目录：环境变量 > 配置项 > 默认外部目录。"""
    override = os.getenv("EVAL_REPORT_DIR") or getattr(settings, "EVAL_REPORT_DIR", "")
    if override:
        return Path(override).expanduser()
    return DEFAULT_REPORT_DIR
