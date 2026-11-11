"""可观测性适配层（T7-0）。

- `tracer.py`：`SpanHandle` / `Tracer` 端口 / `JsonlTracer` / `NoopTracer`；
- `langchain_handler.py`：LangChain 回调 -> span（llm / tool）；
- `turn.py`：一轮的根 span 与汇总。

只在 `ENABLE_TRACE` 打开时被 api_main 实例化；关闭时本包不产生任何 IO。
"""
