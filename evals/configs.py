"""评测配置矩阵（A / B / C）。

只定义"要注入后端的开关"，不碰 `config.py` 的默认值：默认值始终是生产口径
（全关），评测时按 config 显式覆盖。

- A：全关，复现改造前 raw 行为；
- B：治理（工具策略 + 追踪），审批 / Skills 仍关；
- C：在 B 上再加 HITL。C 同时打开 `EVAL_AUTO_APPROVE=1`——与 docs/baseline.md §8
  C 组同口径，避免 20 题里偶发一个需审批工具把评测挂到超时；该开关只在
  `ENV=dev` 下生效，生产 `ENV=prod` 会被忽略。

用法：
    from evals.configs import get_configs

    for config in get_configs(["A", "B"]):
        env = dict(config.env)  # 直接传给后端子进程
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    """一个评测配置：id、名字、注入后端的 env、人读说明。"""

    id: str
    name: str
    env: dict[str, str]
    description: str


def _flags(
    *,
    tool_policy: bool,
    trace: bool,
    hitl: bool,
    skills: bool = False,
    auto_approve: bool = False,
) -> dict[str, str]:
    """把布尔开关转成后端可读的 `0/1` 环境变量。"""
    env = {
        "ENABLE_TOOL_POLICY": "1" if tool_policy else "0",
        "ENABLE_TRACE": "1" if trace else "0",
        "ENABLE_HITL": "1" if hitl else "0",
        "ENABLE_SKILLS": "1" if skills else "0",
    }
    if auto_approve:
        env["EVAL_AUTO_APPROVE"] = "1"
    return env


CONFIGS: dict[str, EvalConfig] = {
    "A": EvalConfig(
        id="A",
        name="全关",
        env=_flags(tool_policy=False, trace=False, hitl=False),
        description="四能力全关，复现改造前 raw 行为；与 docs/baseline.md §7 可直接横比",
    ),
    "B": EvalConfig(
        id="B",
        name="治理",
        env=_flags(tool_policy=True, trace=True, hitl=False),
        description="工具级权限预算 + 链路追踪；审批与 Skills 关",
    ),
    "C": EvalConfig(
        id="C",
        name="治理+审批",
        env=_flags(tool_policy=True, trace=True, hitl=True, auto_approve=True),
        description="B 上再加 HITL；评测专用自动放行（与 §8 C 组同口径）",
    ),
}


def get_config(config_id: str) -> EvalConfig:
    """按 id 取配置；不存在时抛出带可选值的 ValueError。"""
    key = config_id.strip().upper()
    try:
        return CONFIGS[key]
    except KeyError:
        raise ValueError(f"未知配置 {config_id!r}；可选：{', '.join(CONFIGS)}") from None


def get_configs(config_ids: list[str] | tuple[str, ...]) -> list[EvalConfig]:
    """按给定顺序取多个配置；空列表报错，避免"什么都没跑"却像成功。"""
    if not config_ids:
        raise ValueError("至少需要一个配置，例如 --configs A")
    return [get_config(config_id) for config_id in config_ids]


def parse_config_ids(raw: str) -> list[str]:
    """解析 `--configs A,B,C`；忽略空白与空段。"""
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    if not ids:
        raise ValueError("--configs 不能为空，例如 --configs A,B,C")
    return ids
