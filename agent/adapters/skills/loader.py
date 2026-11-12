"""Skills 加载器（T6-4）：扫描 `SKILL.md`、按 mtime 热加载、按白名单读正文。

设计取舍（见交接单）：**渐进披露**。系统提示里最多放一行索引，技能正文由
`load_skill(name)` 按需读取；不把全部技能正文拼进稳定区，避免吃掉上下文预算。

安全边界：`load(name)` 只接受 `[a-z0-9-]`，并且只从 `scan()` 产出的 `SkillSpec.path`
读取；**绝不用用户输入的 name 拼路径**（防 `../` 目录穿越）。

损坏处理：缺 frontmatter / YAML 解析失败 / 缺字段 / name 不合法 -> 跳过该文件 +
一条 warning，其余技能照常返回，不崩。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from agent.domain.skill import SkillSpec
from config import settings
from src.context_manager import estimate_tokens
from src.utils.logger import get_logger

logger = get_logger()

#: 技能名白名单：小写字母、数字、短横线（不含 `/`、`.`、空格）
_NAME_RE = re.compile(r"^[a-z0-9-]+$")

#: 仓库根：`agent/adapters/skills/loader.py` -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]

#: 截断标记（正文超 `SKILL_MAX_BODY_TOKENS` 时追加）
TRUNCATED_MARK = "\n\n…（技能正文已截断）"

#: mtime 缓存：path -> (st_mtime_ns, st_size, spec 或 None)
#: 存 None 表示“上次已判定为坏文件”，文件没变就不重复 warning。
_CACHE: dict[str, tuple[int, int, SkillSpec | None]] = {}


# ── 对外：目录 / 扫描 / 读取 ──────────────────────────────────────────


def skills_root() -> Path:
    """返回技能根目录；`SKILLS_DIR` 为相对路径时按仓库根解析。"""
    root = Path(settings.SKILLS_DIR).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root


def scan() -> list[SkillSpec]:
    """扫描 `SKILLS_DIR/*/SKILL.md`，返回按 name 排序的技能索引。

    同一路径 mtime / size 未变时复用上次解析结果；新建 / 修改 / 删除文件后
    下一次 `scan()` 自动生效，无需重启进程。
    """
    root = skills_root()
    if not root.is_dir():
        _CACHE.clear()
        return []

    found: dict[str, SkillSpec] = {}
    alive: set[str] = set()
    for path in sorted(root.glob("*/SKILL.md")):
        key = str(path)
        alive.add(key)
        try:
            stat = path.stat()
        except OSError as exc:
            logger.warning("跳过技能文件 %s：无法读取文件状态（%s）", path, exc)
            continue

        cached = _CACHE.get(key)
        if cached is not None and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            spec = cached[2]
        else:
            spec = _parse_file(path)
            _CACHE[key] = (stat.st_mtime_ns, stat.st_size, spec)

        if spec is None:
            continue
        kept = found.get(spec.name)
        if kept is not None:
            logger.warning("技能重名，跳过 %s：name=%s 已被 %s 占用", path, spec.name, kept.path)
            continue
        found[spec.name] = spec

    # 删掉已不存在的文件缓存，避免改名 / 删除后仍命中旧记录
    for key in list(_CACHE):
        if key not in alive:
            del _CACHE[key]

    return sorted(found.values(), key=lambda item: item.name)


def load(name: str) -> str:
    """按 name 读取技能正文（不含 frontmatter）；未知 / 非法名字返回可读错误。"""
    raw = str(name or "").strip()
    specs = scan()

    if not _NAME_RE.fullmatch(raw):
        return (
            f"已拒绝：技能名不合法 {raw!r}。只允许小写字母、数字和短横线（[a-z0-9-]）；"
            f"不会按路径读取。可用技能：{_names_hint(specs)}。"
        )

    spec = next((item for item in specs if item.name == raw), None)
    if spec is None:
        return f"未知技能：{raw}。可用技能：{_names_hint(specs)}。请先调用 list_skills。"

    try:
        text = spec.path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("读取技能 %s 失败：%s", spec.name, exc)
        return f"读取技能 {spec.name} 失败：{exc}"

    split = _split_frontmatter(text)
    body = split[1] if split is not None else text
    return _truncate(body.strip(), settings.SKILL_MAX_BODY_TOKENS)


# ── 内部：解析 / 截断 / 文案 ──────────────────────────────────────────


def _parse_file(path: Path) -> SkillSpec | None:
    """解析单个 SKILL.md；任何损坏都只 warning + 返回 None。"""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("跳过技能文件 %s：读取失败（%s）", path, exc)
        return None

    split = _split_frontmatter(text)
    if split is None:
        logger.warning("跳过技能文件 %s：缺少 YAML frontmatter（首尾 --- 包裹）", path)
        return None

    meta_text, body = split
    try:
        meta: Any = yaml.safe_load(meta_text)
    except yaml.YAMLError as exc:
        logger.warning("跳过技能文件 %s：frontmatter 不是合法 YAML（%s）", path, exc)
        return None
    if not isinstance(meta, dict):
        logger.warning("跳过技能文件 %s：frontmatter 必须是 key: value 映射", path)
        return None

    name = str(meta.get("name") or "").strip()
    description = str(meta.get("description") or "").strip()
    when_to_use = str(meta.get("when_to_use") or "").strip()
    if not _NAME_RE.fullmatch(name):
        logger.warning("跳过技能文件 %s：name=%r 不合法（只允许 [a-z0-9-]）", path, name)
        return None
    if not description or not when_to_use:
        logger.warning("跳过技能文件 %s：缺少 description 或 when_to_use", path)
        return None

    return SkillSpec(
        name=name,
        description=description,
        when_to_use=when_to_use,
        path=path,
        size_tokens=estimate_tokens(body),
    )


def _split_frontmatter(text: str) -> tuple[str, str] | None:
    """切出 `---` 包裹的 frontmatter 与正文；没有合法首尾分隔符返回 None。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            meta = "\n".join(lines[1:index]).strip()
            body = "\n".join(lines[index + 1 :]).strip()
            return meta, body
    return None


def _truncate(text: str, max_tokens: int) -> str:
    """把正文截到 `max_tokens` 以内；超限时保留前缀并追加截断标记。"""
    limit = int(max_tokens)
    if limit <= 0:
        return ""
    if estimate_tokens(text) <= limit:
        return text

    budget = max(0, limit - estimate_tokens(TRUNCATED_MARK))
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= budget:
            low = mid
        else:
            high = mid - 1
    return text[:low].rstrip() + TRUNCATED_MARK


def _names_hint(specs: list[SkillSpec]) -> str:
    return "、".join(item.name for item in specs) or "（暂无可用技能）"
