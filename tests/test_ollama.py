#!/usr/bin/env python3
import sys
import urllib.request

url = "http://localhost:11434/api/tags"

try:
    with urllib.request.urlopen(url, timeout=5) as response:
        print(f"✅ Ollama 服务正常运行 (HTTP {response.status})")
except urllib.error.URLError as e:
    print(f"❌ 连接失败: {e.reason}")
    sys.exit(1)
