"""指标统计组件（异步安全版）。

用于记录 AI 模型的运行状态、JSON 解析命中率、Token 消耗以及搜索迭代次数。
所有文件 I/O 操作（mkdir / open / json.dump）均通过 asyncio.to_thread()
委托给线程池执行，不会阻塞 AsyncIO 事件循环，兼容 FastAPI / langgraph dev 等异步服务器。
"""

import asyncio
import json
from pathlib import Path


class Metrics:
    """指标统计组件。

    所有会触发磁盘写入的公开方法均为 async，内部通过 asyncio.to_thread()
    将同步文件 I/O 抛给线程池，确保事件循环不被阻塞。
    """

    def __init__(self, data_path: Path = Path("data/metrics.json")):
        """初始化指标统计器（同步，仅在模块导入时执行一次）。"""
        self._data_path = data_path

        # --- 核心指标定义 ---
        self.parse_total = 0  # JSON 解析总调用次数（分母）
        self.parse_success = 0  # 直接解析成功的次数 (Strategy 1)
        self.parse_fallback = 0  # 通过正则回退机制解析成功的次数 (Strategy 2/3)
        self.parse_deepseek_fallback = 0  # 通过调用 DeepSeek 重试成功的次数
        self.total_tokens = 0  # 累计消耗的 Token 总量
        self.iterations_per_query: list[int] = []  # 每次查询的迭代次数
        self.search_rounds = 0  # 累计搜索轮次
        self.parallel_speedup = 0.0  # 并行加速比

        # 初始化时从本地文件加载已有数据（同步，导入阶段无事件循环）
        self._load_sync()

    @property
    def data_path(self) -> Path:
        """获取当前指标文件存储路径。"""
        return self._data_path

    # ── 同步文件 I/O（仅供 to_thread 或初始化调用） ─────────────────

    def _load_sync(self) -> None:
        """从本地 JSON 文件加载统计数据（同步，仅初始化时调用）。"""
        if not self._data_path.exists():
            return
        try:
            with open(self._data_path, encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if k.startswith("_") or k == "data_path":
                    continue
                if hasattr(self, k):
                    setattr(self, k, v)
        except (json.JSONDecodeError, OSError) as e:
            print(f"警告: 无法加载指标文件 {self._data_path}: {e}")

    def _save_sync(self) -> None:
        """将当前指标写入磁盘（同步，供 asyncio.to_thread 调用）。"""
        # 确保父目录存在（mkdir -p 语义）
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        to_save = {
            "parse_total": self.parse_total,
            "parse_success": self.parse_success,
            "parse_fallback": self.parse_fallback,
            "parse_deepseek_fallback": self.parse_deepseek_fallback,
            "total_tokens": self.total_tokens,
            "iterations_per_query": self.iterations_per_query,
            "search_rounds": self.search_rounds,
        }
        with open(self._data_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2, default=str)

    # ── 异步持久化（公开方法） ─────────────────────────────────────

    async def save(self) -> None:
        """将当前指标异步写入磁盘（非阻塞）。"""
        # asyncio.to_thread：把同步函数 _save_sync 丢到线程池执行，
        # 主线程的事件循环可以继续处理其他协程，不会卡住服务器。
        await asyncio.to_thread(self._save_sync)

    async def record_parse(
        self,
        success: bool,
        fallback: bool = False,
        deepseek_fallback: bool = False,
    ) -> None:
        """记录一次 JSON 解析的结果（异步）。"""
        self.parse_total += 1
        if success:
            self.parse_success += 1
        if fallback:
            self.parse_fallback += 1
        if deepseek_fallback:
            self.parse_deepseek_fallback += 1
        # 每次记录后异步持久化，防止程序崩溃丢失数据
        await self.save()

    async def record_iterations(self, iterations: int) -> None:
        """记录单次请求的迭代深度（异步）。"""
        self.iterations_per_query.append(iterations)
        await self.save()

    async def record_parallel_speedup(self, speedup: float) -> None:
        """记录并行加速比（异步）。"""
        self.parallel_speedup = speedup
        await self.save()

    # ── 纯计算（同步，无 I/O） ─────────────────────────────────────

    def get_parse_success_rate(self) -> float:
        """计算 JSON 解析的总成功率（包含所有回退策略）。

        Returns:
            float: 0.0 到 1.0 之间的成功率。
        """
        if self.parse_total > 0:
            return self.parse_success / self.parse_total
        return 0.0


# ── 全局单例 ─────────────────────────────────────────────────────────
# 整个应用通过导入这个 metrics 实例来共享统计数据
metrics = Metrics()
