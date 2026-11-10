import json
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam

MODEL = "qwen35-9b-official"  # ← 在函数外定义

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

tools: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

questions = [
    "北京今天天气怎么样？",
    "上海呢？",
    "帮我查下广州的天气",
    "深圳冷不冷？",
    "杭州下雨吗？",
    "成都天气如何",
    "明天去武汉要带伞吗？",
    "南京现在多少度",
    "重庆天气",
    "天津怎么样",
]

ok, bad_json, no_call = 0, 0, 0
for q in questions:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": q}],
        tools=tools,
        temperature=0,
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        call: Any = msg.tool_calls[0]
        try:
            args = json.loads(call.function.arguments)
            ok += 1
            print(f"[OK]   {q} -> {args}")
        except json.JSONDecodeError:
            bad_json += 1
            print(f"[BAD]  {q} -> {call.function.arguments!r}")
    else:
        no_call += 1
        print(f"[NONE] {q} -> {msg.content!r}")

print(f"\n总计 {len(questions)} 次：合法 tool_calls={ok}, JSON 坏={bad_json}, 未调用={no_call}")
print(f"工具调用成功率 = {ok / len(questions) * 100:.1f}%")
