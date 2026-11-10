"""端口（Port）占位。

T1-1 只建骨架：不定义 Protocol、不写任何签名；
端口签名与契约测试在 T1-2 / T-arch 落地。
"""

# 端口 -> 适配器 清单（只列名字，不写签名）：
#
#   AgentRuntime  -> adapters/runtime
#   ModelClient   -> adapters/models
#   SessionStore  -> adapters/stores
#   ArtifactStore -> adapters/stores
#   TaskStore     -> adapters/tasks
#   MemoryStore   -> adapters/stores
#   EventBus      -> adapters/events
#
# 非端口适配器（由组合根直接装配，不经过端口）：
#
#   adapters/locks  -> 分布式锁薄封装
#   adapters/api    -> 入站 HTTP / SSE 路由
#   adapters/tools  -> 工具实现与注册表
