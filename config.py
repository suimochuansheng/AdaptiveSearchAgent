"""AdaptiveSearchAgent 配置模块。

从项目根目录 .env 文件加载环境变量，提供类型安全的全局配置访问。
"""

from enum import StrEnum
from typing import Literal

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

    # 会话式 Agent 的上下文窗口（tokens）。
    # 不显式设置时窗口由 Ollama 的模型默认值决定，阶段 B 的预算分层就失去锚点；
    # 这里固定 8192（3080 10G 上 qwen 7B INT4 + KV cache 可承载）。
    ollama_num_ctx: int = 8192

    # ── 阶段 B：上下文管理预算（tokens）──
    # 关闭后 build_context 短路成"system + 全量历史"，行为回退到阶段 A
    context_management_enabled: bool = True
    # 最近若干轮保留原文的预算
    context_history_budget: int = 3000
    # 更早轮次压缩成摘要的预算
    context_summary_budget: int = 800
    # 单条工具消息进 prompt 的预算（超出走段落重要度压缩）
    tool_output_token_budget: int = 1200
    # 工具输出超过该 token 数就落盘，prompt 只留压缩版 + 指针（Offload）。
    # 默认等于 prompt 预算：凡是"压缩"就是有损，有损就把原件留在磁盘上，
    # 模型需要细节时还能按指针回读；调大它则退化为"只对超大结果落盘"。
    offload_token_threshold: int = 1200
    offload_enabled: bool = True
    # 回读卸载文件时的预算：要比 prompt 里的压缩版更大，否则"回读"等于又压一遍
    offload_read_token_budget: int = 3000

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
    max_concurrent_searches: int = 5  # 同时执行的搜索任务数（Send 并发数）
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
    MAX_CONTEXT_TOKENS: int = 8000  # writer 上下文 token 上限

    # Langfuse 可观测性
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_ENABLED: bool = False

    # Sentry 崩溃自动捕获
    SENTRY_DSN: str = ""  # 为空则不启用
    ENV: str = "dev"  # 环境标识 (dev/staging/prod)

    # DEBUG 模式开关，控制日志输出和调试信息
    DEBUG: bool = False

    # ===== 新功能开关（默认关闭，零破坏性） =====

    # 动态工具路由：根据用户画像/上下文动态选择工具集
    ENABLE_DYNAMIC_TOOLS: bool = False

    # 长期记忆：持久化用户偏好与历史话题
    ENABLE_MEMORY: bool = False

    # 异步任务：超时请求转为后台异步任务处理
    ENABLE_ASYNC_TASK: bool = False
    # 异步任务结果在 Redis 的存活时间（秒），默认 7 天
    ASYNC_TASK_RESULT_TTL: int = 604800
    # 异步任务僵尸判定：running 超过这么久没更新就认为 Worker 没了（秒）
    ASYNC_STALE_SECONDS: int = 1800
    # 僵尸任务最多自动重跑几次（超过就置 failed，避免无限循环）
    ASYNC_MAX_ATTEMPTS: int = 2
    # 终态任务行保留天数（结果在 Redis 另有 TTL）
    ASYNC_PURGE_DAYS: int = 7
    # 维护循环间隔（秒）：回收僵尸 + 清理终态行
    ASYNC_MAINTENANCE_INTERVAL_SECONDS: int = 300
    # RAG 子服务熔断冷却时间（秒）：连接级失败后这段时间内直接跳过检索
    RAG_COOLDOWN_SECONDS: int = 60
    # deep_research 的执行后端：
    #   workflow = 真跑旧 workflow 图（默认，行为与现在一致）
    #   stub     = 假图（开发期用：秒回、不依赖 RAG/Ollama 子服务）
    # 因为 deep_research 已经是普通工具，换后端不影响外层会话循环。
    DEEP_RESEARCH_BACKEND: Literal["workflow", "stub"] = "workflow"
    # ── 阶段 C3：跨会话画像 ──
    # 关掉后不读画像、不触发抽取，行为与阶段 B 完全一致
    MEMORY_ENABLED: bool = False
    # 画像注入稳定区的 token 预算
    MEMORY_PROFILE_TOKEN_BUDGET: int = 200
    # 每 N 轮触发一次异步抽取（写侧，C3b 接入）
    MEMORY_EXTRACT_EVERY_TURNS: int = 10

    # 异步任务触发阈值（秒），超过此阈值的请求将转换为后台任务
    ASYNC_TASK_THRESHOLD_SECONDS: int = 120


# 全局单例，项目各处从此导入
settings = Settings()
