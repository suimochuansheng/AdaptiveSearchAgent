#!/usr/bin/env python
"""T6-0 冒烟：打印 `TOOLS` 的 OpenAI function schema 指纹。

用法（任意 cwd 均可，建议仓库根目录）：

    poetry run python scripts/smoke_tools_schema.py          # 人读摘要
    poetry run python scripts/smoke_tools_schema.py --json   # 完整 schema JSON 到 stdout

基线（2026-11-11 冻结，T6-0）：

    names : search_knowledge, search_tavily, calculator, deep_research, read_offload
    sha256: c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99
    bytes : 2558

哈希口径与开工基线命令一致：对
`json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()` 取 sha256。
`--json` 输出同构的完整 schema（缩进 + 键排序），供人工 diff；它不参与哈希计算。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # 让脚本在任意 cwd 下都能 import src
    sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description="打印 TOOLS 的 schema 指纹（T6-0）")
    parser.add_argument(
        "--json",
        action="store_true",
        help="把完整 schema 以 JSON 打到 stdout（供人工比对，不参与哈希）",
    )
    args = parser.parse_args()

    from langchain_core.utils.function_calling import convert_to_openai_tool

    from src.conversation_graph import TOOLS

    schemas = [convert_to_openai_tool(tool) for tool in TOOLS]
    canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()

    if args.json:
        print(json.dumps(schemas, sort_keys=True, ensure_ascii=False, indent=2))
        return 0

    names = [schema["function"]["name"] for schema in schemas]
    print(f"names : {', '.join(names)}")
    print(f"count : {len(schemas)}")
    print(f"sha256: {hashlib.sha256(canonical).hexdigest()}")
    print(f"bytes : {len(canonical)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
