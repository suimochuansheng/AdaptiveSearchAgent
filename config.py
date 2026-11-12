"""AdaptiveSearchAgent 配置模块。

从项目根目录 .env 文件加载环境变量，提供类型安全的全局配置访问。
"""

from enum import StrEnum
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings

from src.utils.logger import get_logger

logger = get_logger()


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
    # 评测裁判模型（Ragas 三指标 + Agent judge 共用）。
    # 留空时取 ALIYUN_QWEN_MODEL_NAME，保证"裁判是谁"只有一个事实来源；
    # 需要临时换裁判时用环境变量 EVAL_JUDGE_MODEL=... ，不要改 .env。
    EVAL_JUDGE_MODEL: str = ""

    @model_validator(mode="after")
    def _fill_eval_judge_model(self) -> "Settings":
        if not self.EVAL_JUDGE_MODEL:
            self.EVAL_JUDGE_MODEL = self.ALIYUN_QWEN_MODEL_NAME
        return self

    @model_validator(mode="after")
    def _clamp_subagent_output_budget(self) -> "Settings":
        """把子代理摘要上限夹到主会话工具输出预算之下。

        原因：摘要进主会话会过 `shrink_tool_message`，超预算会被二次压缩，
        `…（已截断）` 标记与首尾保留都会丢；这里在启动时一次性告警并夹紧生效值。
        """
        raw = int(self.SUBAGENT_MAX_OUTPUT_TOKENS)
        budget = int(self.tool_output_token_budget)
        effective = raw
        if raw >= budget:
            # 留 100 token 余量：主会话还会在摘要之外拼工具消息模板/指针。
            effective = max(1, budget - 100)
            logger.warning(
                "SUBAGENT_MAX_OUTPUT_TOKENS=%s 不小于 tool_output_token_budget=%s；"
                "摘要进主会话会被 shrink_tool_message 二次压缩（截断标记会丢），"
                "已将生效值夹紧为 %s",
                raw,
                budget,
                effective,
            )
        self.SUBAGENT_EFFECTIVE_MAX_OUTPUT_TOKENS = effective
        return self

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

    # 工具级权限与预算（T6-3）：默认关闭；关时执行器零包装，代码路径与改造前一致
    ENABLE_TOOL_POLICY: bool = False
    # 链路追踪（T7-0）：默认关闭；关时零文件 IO、零 callbacks，SSE 帧逐帧一致
    ENABLE_TRACE: bool = False
    # 工具级人工审批（G7/T9-2）：默认关闭；关时执行器仍走 G5 的“需审批但未启用”拒绝分支，
    # 审批路由即使被调用也返回 404。开启后危险工具在工具函数内 interrupt() 挂起。
    ENABLE_HITL: bool = False
    # 审批凭据有效期（秒）：契约冻结 10 分钟；验收时临时调 1 秒验证 410
    APPROVAL_TTL_SECONDS: int = 600
    # ⚠️ 仅评测/演示用，生产禁用：策略判定"需审批"时直接放行（不调 interrupt、不签凭据）。
    # 默认 False；仅在 G11 基线评测（C 组）显式打开。ENV=prod 时即使设为 True 也忽略并 warning。
    EVAL_AUTO_APPROVE: bool = False
    # 审批墓碑保留时长（秒）：28 小时内区分“已消费（409）/ 已过期（410）/ 从未存在（404）”
    APPROVAL_TOMBSTONE_TTL_SECONDS: int = 86400
    # MCP 工具热插拔（T9-3）：默认关闭；关时零子进程、零 mcp 导入，注册表恒为 5 个内置工具。
    # 开启后启动阶段连 stdio MCP Server，把远端工具登记进注册表并重建会话图。
    ENABLE_MCP: bool = False
    # stdio 子进程命令与参数（参数按 shell 规则拆分，支持 "--flag value" 这种写法）
    MCP_SERVER_CMD: str = "python"
    MCP_SERVER_ARGS: str = "scripts/mcp_server.py"
    # 握手 / tools/list / tools/call 的单次超时（秒）
    MCP_TIMEOUT_SECONDS: float = 10.0
    # MCP 工具单次输出上限（token，进 ToolSpec 供预算使用）
    MCP_MAX_OUTPUT_TOKENS: int = 2000
    # 子代理工具（T9-4）：默认关闭；关时注册表仍是 5 个内置工具、schema 哈希不变。
    # 开启后启动阶段把 subagent_task 登记进注册表并重建会话图（与 MCP 同一套做法）。
    ENABLE_SUBAGENT: bool = False
    # 子代理单次任务最多跑几轮（agent ⇄ tools 循环；每轮 = 模型 + 工具两步）
    SUBAGENT_MAX_TURNS: int = 8
    # 子代理回传主会话的摘要上限（token）：超了保留首尾、中间标“已截断”。
    # 摘要进主会话会过 shrink_tool_message，超预算会被二次压缩，截断标记会丢；
    # 所以默认 1000 必须小于 tool_output_token_budget（1200），并在启动时校验。
    SUBAGENT_MAX_OUTPUT_TOKENS: int = 1000
    # 生效值：_clamp_subagent_output_budget 按主会话工具输出预算夹紧后的值。
    # 业务代码只读它，不要直接读上面的原始配置。
    SUBAGENT_EFFECTIVE_MAX_OUTPUT_TOKENS: int = 1000
    # 子代理单次任务的总超时（秒）；超时返回可读错误，不抛栈
    SUBAGENT_TIMEOUT_SECONDS: int = 300
    # span 落盘目录（每 span 一行 JSON：<TRACE_DIR>/<YYYY-MM-DD>.jsonl）；logs/ 已被 gitignore
    TRACE_DIR: str = "logs/trace"
    # 单轮工具调用上限（达到后本轮后续调用直接拒绝）
    TOOL_MAX_CALLS_PER_TURN: int = 8
    # 会话累计工具费用上限（元）；v1 计价 = 每次调用 1 个 unit
    TOOL_SESSION_COST_LIMIT: float = 2.0
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
    # RAG 检索 HTTP 超时（秒）：BGE 精排单次约 10~11s，原 10s 会稳定超时
    RAG_TIMEOUT_SECONDS: float = 30.0
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
    # 抽取时最多看最近多少条消息（控制成本）
    MEMORY_EXTRACT_MAX_MESSAGES: int = 20
    # 评测报告输出目录（空 = 默认 ../文档/项目/AdaptiveSearchAgent/评测报告）
    EVAL_REPORT_DIR: str = ""

    # 异步任务触发阈值（秒），超过此阈值的请求将转换为后台任务
    ASYNC_TASK_THRESHOLD_SECONDS: int = 120

    # ── 新功能开关：Skills 加载（T6-4）──
    # 默认关闭；关时注册表仍是 5 个内置工具、schema 哈希不变。
    # 开启后启动阶段登记 list_skills / load_skill 并重建会话图（与 MCP / 子代理同一套做法）。
    ENABLE_SKILLS: bool = False
    # 技能包根目录（相对路径按仓库根解析）；每个子目录一个 SKILL.md
    SKILLS_DIR: str = "skills"
    # list_skills 最多返回多少条索引（控制工具输出，不把全部技能塞进上下文）
    SKILLS_INDEX_MAX: int = 30
    # 单个技能正文的 token 上限；超出截断并加标记
    SKILL_MAX_BODY_TOKENS: int = 2000

    # ── 新功能开关：工具输出注入防护（T6-5）──
    # 默认关闭；关时 shrink_tool_message 逐字节不变、系统提示不追加。
    # 打开后所有工具输出（含 MCP / 子代理）与 load_skill 正文统一用 <untrusted> 包裹。
    ENABLE_TOOL_OUTPUT_GUARD: bool = False


# 全局单例，项目各处从此导入
settings = Settings()
