"""AdaptiveSearchAgent 配置模块。

从项目根目录 .env 文件加载环境变量，提供类型安全的全局配置访问。
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """项目全局配置，自动从 .env 文件和环境变量中加载。

    所有字段名自动映射为大写环境变量（如 ollama_model → OLLAMA_MODEL）。
    """

    # Tavily 搜索 API 密钥
    TAVILY_API_KEY: str = ""

    # DeepSeek API 备援密钥
    DEEPSEEK_API_KEY: str = ""

    # Ollama本地模型名称
    OLLAMA_MODEL_NAME: str = ""

    # Ollama 本地服务地址
    OLLAMA_BASE_URL: str = ""

    # OpenAI 兼容 API 地址（备援使用）
    OPENAI_BASE_URL: str = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# 全局单例，项目各处从此导入
settings = Settings()
