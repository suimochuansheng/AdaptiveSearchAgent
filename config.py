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

    # 千问模型API Key
    ALIYUN_QWEN_MODEL_NAME: str = "qwen3.8-max"
    ALIYUN_QWEN_API_KEY: str = ""
    ALIYUN_QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # qwen 3.8-max模型 新增控制思考深度的参数
    ALIYUN_QWEN_MODEL_REASONING_EFFORT: str = "medium"
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
        "extra": "ignore",  # 忽略 .env 中未定义字段
    }
    langchain_tracing_v2: bool | None = False  # 或者 str，按实际需要

    # Redis 连接配置（分布式锁、缓存）
    REDIS_URL: str = ""
    REDIS_PASSWORD: str = ""

    # PostgreSQL 数据库连接配置（Checkpointer 持久化 + 任务状态管理）
    postgres_user: str = ""
    postgres_password: str = ""
    postgres_db: str = ""
    postgres_host: str = ""
    postgres_port: int = 5432

    # 新增字段，与 .env 中的变量名一致
    fastapi_base_url: str = "http://localhost:8000"  # 后端基础 URL
    chainlit_port: int = 8001  # Chainlit 前端端口

    # FastAPI / Uvicorn 服务配置
    SERVICE_HOST: str = "0.0.0.0"
    SERVICE_PORT: int = 8000

    # RAG 摄入子服务地址
    INGEST_SERVICE_URL: str = "http://localhost:8100"

    # 上下文长度保护
    MAX_SEARCH_RESULTS: int = 5  # 单次检索最多保留条数
    MAX_CONTEXT_TOKENS: int = 4000  # writer 上下文 token 上限

    # Langfuse 可观测性
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_ENABLED: bool = False

    # DEBUG 模式开关，控制日志输出和调试信息
    DEBUG: bool = False

    # ===== 新功能开关（默认关闭，零破坏性） =====

    # 动态工具路由：根据用户画像/上下文动态选择工具集
    ENABLE_DYNAMIC_TOOLS: bool = False

    # 长期记忆：持久化用户偏好与历史话题
    ENABLE_MEMORY: bool = False

    # 异步任务：超时请求转为后台异步任务处理
    ENABLE_ASYNC_TASK: bool = False

    # 异步任务触发阈值（秒），超过此阈值的请求将转换为后台任务
    ASYNC_TASK_THRESHOLD_SECONDS: int = 120


# 全局单例，项目各处从此导入
settings = Settings()
