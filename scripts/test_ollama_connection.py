#!/usr/bin/env python3
"""
Ollama 连接测试脚本

用于检测 WSL 环境是否能正常访问 Windows 宿主机的 Ollama 服务。
同时验证 Embedding 模型是否可用。

用法：
    poetry run python scripts/test_ollama_connection.py

依赖：
    - httpx (已安装)
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx

from config import settings
from src.utils.embedding import EMBEDDING_MODEL

# ============================================================
# 颜色输出（终端友好）
# ============================================================


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


def print_ok(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")


def print_err(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")


def print_warn(msg: str):
    print(f"{Colors.YELLOW}⚠️ {msg}{Colors.RESET}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️ {msg}{Colors.RESET}")


# ============================================================
# 测试函数
# ============================================================


async def test_ollama_connection():  # noqa: C901
    """测试 Ollama 连接"""
    print_info("=" * 60)
    print_info("Ollama 连接测试")
    print_info("=" * 60)

    # 1. 检查配置中的 Ollama 地址
    base_url = settings.ollama_base_url
    print_info(f"📡 配置的 Ollama 地址: {base_url}")

    # 尝试从环境变量中读取 OLLAMA_HOST（用于 Windows 宿主机测试）
    env_host = os.environ.get("OLLAMA_HOST", "")
    if env_host:
        print_info(f"📡 OLLAMA_HOST 环境变量: {env_host}")

    # 2. 检查 WSL 是否能访问 Windows 宿主机
    print_info("\n🔍 检测 WSL 网络环境...")

    # 尝试解析 host.docker.internal
    import socket

    try:
        host_ip = socket.gethostbyname("host.docker.internal")
        print_info(f"✅ host.docker.internal 解析为: {host_ip}")
    except socket.gaierror:
        print_warn("host.docker.internal 无法解析 (Docker 可能未运行)")

    # 尝试获取 Windows 宿主机的实际 IP (通过 resolv.conf)
    try:
        resolv_path = "/etc/resolv.conf"
        if os.path.exists(resolv_path):
            with open(resolv_path) as f:
                for line in f:
                    if line.startswith("nameserver"):
                        windows_ip = line.split()[1]
                        print_info(f"✅ Windows 宿主机 IP (从 resolv.conf): {windows_ip}")
                        break
    except Exception:
        pass

    # 3. 测试 Ollama API
    print_info("\n🔍 测试 Ollama API 连通性...")

    async with httpx.AsyncClient(timeout=5.0) as client:
        # 测试 /api/tags (获取模型列表)
        try:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            model_names = [m.get("name", "unknown") for m in models]
            print_ok(
                f"✅ Ollama 连接成功，已安装模型: {', '.join(model_names) if model_names else '(无)'}"
            )
        except httpx.ConnectError:
            print_err(f"❌ 无法连接到 Ollama ({base_url})")
            print_info("   请检查: ")
            print_info("   1. Windows 端 Ollama 是否已启动 (系统托盘是否有图标)")
            print_info("   2. Ollama 是否允许外部访问 (OLLAMA_HOST=0.0.0.0:11434)")
            print_info("   3. Windows 防火墙是否开放了 11434 端口")
            print_info("   4. 在 WSL 中执行: curl {base_url}/api/tags")
            return False
        except httpx.TimeoutException:
            print_err(f"❌ Ollama 连接超时 ({base_url})")
            print_info("   请检查 Windows 防火墙或 Ollama 服务状态")
            return False
        except Exception as e:
            print_err(f"❌ Ollama 连接失败: {e}")
            return False

    # 4. 测试 Embedding 模型
    print_info("\n🔍 测试 Embedding 模型...")

    embedding_model = EMBEDDING_MODEL
    print_info(f"📡 使用的 Embedding 模型: {embedding_model}")

    # 检查模型是否已安装（Ollama 返回的模型名带 tag，如 nomic-embed-text:latest，需去掉 tag 再匹配）
    installed_models = [m.split(":")[0] for m in model_names]
    if embedding_model not in installed_models:
        print_warn(f"⚠️ 模型 '{embedding_model}' 未安装")
        print_info(f"   执行: ollama pull {embedding_model}")
        return False

    # 测试 Embedding API
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base_url}/api/embeddings",
                json={
                    "model": embedding_model,
                    "prompt": "测试文本，用于验证 Embedding 是否正常工作",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding", [])
            if embedding and len(embedding) > 0:
                print_ok(f"✅ Embedding 模型正常工作 (向量维度: {len(embedding)})")
            else:
                print_err("❌ Embedding 返回空向量")
                return False
    except Exception as e:
        print_err(f"❌ Embedding API 调用失败: {e}")
        return False

    # 5. 全部通过
    print_info("\n" + "=" * 60)
    print_ok("🎉 所有 Ollama 连接测试通过！")
    print_info("=" * 60)
    return True


# ============================================================
# 入口
# ============================================================


def main():
    success = asyncio.run(test_ollama_connection())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
