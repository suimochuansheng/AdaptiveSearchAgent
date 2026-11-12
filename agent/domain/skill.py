"""技能领域模型（T6-4）：`SkillSpec` 是扫描结果，不承载技能正文。

纯数据、零第三方依赖：本模块只 import 标准库，不得 import langchain / langgraph
（`pyproject.toml` 的 `core-no-framework` contract 会拦）。技能正文在
`agent/adapters/skills/loader.py` 里按需读取，领域层只描述“有哪些技能”。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - dataclass 字段类型需运行时可见，便于契约自检


@dataclass(frozen=True)
class SkillSpec:
    """一条技能索引：name / 用途 / 适用场景 / 文件路径 / 正文 token 估算。

    `path` 只能来自 `loader.scan()` 的白名单结果；业务代码不得用用户输入的
    `name` 拼路径（防目录穿越）。
    """

    name: str
    description: str
    when_to_use: str
    path: Path
    size_tokens: int
