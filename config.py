"""AdaptiveSearchAgent 配置模块。

从项目根目录 .env 文件加载环境变量，提供类型安全的全局配置访问。
"""

from enum import StrEnum

from pydantic_settings import BaseSettings


class LLMProvider(StrEnum):
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"


class Settings(BaseSettings):
    """项目全局配置，自动从 .env 文件和环境变量中加载。

    pydantic-settings 默认 case_sensitive=False，因此 .env 中的大写环境变量
    （如 TAVILY_API_KEY）会自动映射到对应的小写字段名。
    """

    # Tavily 搜索 API 密钥
    tavily_api_key: str = ""

    # DeepSeek API 备援密钥
    deepseek_api_key: str = ""

    # DeepSeek 本地模型名称
    DEEPSEEK_MODEL_NAME: str = ""
    # Ollama 本地模型名称
    ollama_model_name: str = ""

    # Ollama 本地服务地址
    ollama_base_url: str = ""

    # OpenAI 兼容 API 地址（备援使用）
    openai_base_url: str = ""

    # 结果可信度阈值（0~1）
    confidence_threshold: float = 0.8
    # 枚举字段，控制 planner 和 evaluator 使用的模型提供商
    llm_provider: LLMProvider = LLMProvider.OLLAMA  # 默认本地

    # 最大迭代次数
    max_iterations: int = 3

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# 全局单例，项目各处从此导入
settings = Settings()
