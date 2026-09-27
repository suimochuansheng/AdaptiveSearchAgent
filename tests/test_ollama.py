#!/usr/bin/env python3
import urllib.error
import urllib.request

import pytest

url = "http://localhost:11434/api/tags"

try:
    with urllib.request.urlopen(url, timeout=5) as response:
        print(f"✅ Ollama 服务正常运行 (HTTP {response.status})")
except urllib.error.URLError as e:
    print(f"❌ 连接失败: {e.reason}")
    pytest.skip(f"Ollama 不可用: {e.reason}", allow_module_level=True)
