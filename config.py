"""AdaptiveSearchAgent 配置模块。

从项目根目录 .env 文件加载环境变量，提供类型安全的全局配置访问。
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """项目全局配置，自动从 .env 文件和环境变量中加载。

    所有字段名自动映射为大写环境变量（如 ollama_model → OLLAMA_MODEL）。
    """

    # Tavily 搜索 API 密钥
    tavily_api_key: str = ""

    # DeepSeek API 备援密钥
    deepseek_api_key: str = ""

    # Ollama 本地模型名称（默认使用 qwen2.5）
    ollama_model: str = "qwen2.5"

    # Ollama 本地服务地址
    ollama_base_url: str = "http://localhost:11434"

    # OpenAI 兼容 API 地址（备援使用）
    openai_base_url: str = ""

    # OpenAI API 密钥（备援使用）
    openai_api_key: str = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# 全局单例，项目各处从此导入
settings = Settings()
