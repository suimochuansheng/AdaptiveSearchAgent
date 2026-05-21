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

    # 并发控制
    max_concurrent_searches: int = 2  # 同时执行的搜索任务数（Send 并发数）
    max_concurrent_llm_calls: int = 2  # 同时执行的 LLM 调用数（Semaphore）
    use_parallel_search: bool = True  # 是否启用并行搜索（用于性能对比）

    # 双模型备援
    llm_max_retries: int = 2  # 主模型失败后的重试次数（超过后切换到备援模型）
    llm_fallback_enabled: bool = True  # 是否启用备援切换

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }
    langchain_tracing_v2: bool | None = False  # 或者 str，按实际需要

    # PostgreSQL 数据库连接配置（如果需要持久化）
    postgres_user: str = "postgres"
    postgres_password: str = ""
    postgres_db: str = "langgraph_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    # 连接池配置
    postgres_pool_min_size: int = 5
    postgres_pool_max_size: int = 20

    class Config:
        env_file = ".env"
        extra = "ignore"


# 全局单例，项目各处从此导入
settings = Settings()
