import json
from pathlib import Path


class Metrics:
    """
    指标统计组件。
    用于记录 AI 模型的运行状态、JSON 解析命中率、Token 消耗以及搜索迭代次数。
    支持数据自动持久化到本地 JSON 文件。
    """

    def __init__(self, data_path: Path = Path("data/metrics.json")):
        """
        初始化指标统计器。

        Args:
            data_path: 统计数据存储的本地路径，默认为 'data/metrics.json'
        """
        self._data_path = data_path  # 存储路径，作为私有属性防止被误序列化

        # --- 核心指标定义 ---
        self.parse_success = 0  # 直接解析成功的次数 (Strategy 1)
        self.parse_fallback = 0  # 通过正则回退机制解析成功的次数 (Strategy 2/3)
        self.parse_deepseek_fallback = 0  # 通过调用更强的模型（如 DeepSeek）重试成功的次数
        self.total_tokens = 0  # 累计消耗的 Token 总量
        self.iterations_per_query: list[int] = []  # 记录每次查询经历的迭代次数（反映任务复杂度）
        self.search_rounds = 0  # 累计搜索轮次

        # 初始化时从本地文件加载已有数据
        self._load()

    @property
    def data_path(self) -> Path:
        """获取当前指标文件存储路径"""
        return self._data_path

    def _load(self):
        """
        从本地 JSON 文件加载统计数据。
        会自动过滤掉私有字段及只读属性，确保内部状态正确恢复。
        """
        if self._data_path.exists():
            try:
                with open(self._data_path, encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    # 跳过以 _ 开头的私有字段，跳过 data_path 属性
                    if k.startswith("_") or k == "data_path":
                        continue
                    # 仅当对象拥有该属性时才进行赋值
                    if hasattr(self, k):
                        setattr(self, k, v)
            except (json.JSONDecodeError, Exception) as e:
                print(f"警告: 无法加载指标文件 {self._data_path}: {e}")

    def save(self):
        """
        将当前的统计指标保存到本地 JSON 文件。
        保存过程中会自动创建必要的文件夹目录。
        """
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        # 显式定义需要保存的字段，避免保存不必要的内部状态
        to_save = {
            "parse_success": self.parse_success,
            "parse_fallback": self.parse_fallback,
            "parse_deepseek_fallback": self.parse_deepseek_fallback,
            "total_tokens": self.total_tokens,
            "iterations_per_query": self.iterations_per_query,
            "search_rounds": self.search_rounds,
        }
        with open(self._data_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2, default=str)

    def record_parse(self, success: bool, fallback: bool = False, deepseek_fallback: bool = False):
        """
        记录一次 JSON 解析的结果。

        Args:
            success: 是否解析成功
            fallback: 是否是通过本地正则回退机制成功的
            deepseek_fallback: 是否是通过调用外部模型修复成功的
        """
        if success:
            self.parse_success += 1
        if fallback:
            self.parse_fallback += 1
        if deepseek_fallback:
            self.parse_deepseek_fallback += 1

        # 每次记录后立即持久化，防止程序崩溃导致数据丢失
        self.save()

    def record_iterations(self, iterations: int):
        """
        记录单次请求的迭代深度（如 ReAct 模式下的思考轮次）。

        Args:
            iterations: 迭代次数
        """
        self.iterations_per_query.append(iterations)
        self.save()

    def get_parse_success_rate(self) -> float:
        """
        计算 JSON 解析的总成功率（包含所有回退策略）。

        Returns:
            float: 0.0 到 1.0 之间的成功率
        """
        # 注意：这里的逻辑假设 success 是总计数，如果 success 仅指第一层成功，
        # 则 total 计算方式可能需要调整为所有层级之和
        total = self.parse_success + self.parse_fallback + self.parse_deepseek_fallback
        return self.parse_success / total if total > 0 else 0.0


# --- 全局单例 ---
# 整个应用通过导入这个 metrics 实例来共享统计数据
metrics = Metrics()
