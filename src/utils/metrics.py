# 目标：收集关键指标（解析成功率、重试次数、总 token 等），并提供保存到 data/metrics.json 的方法。

import json
from pathlib import Path


class Metrics:
    def __init__(self, data_path: Path = Path("data/metrics.json")):
        self.data_path = data_path
        self.parse_success = 0
        self.parse_fallback = 0  # 使用正则回退的次数
        self.parse_deepseek_fallback = 0  # 使用 DeepSeek 最终回退的次数
        self.total_tokens = 0
        self.iterations_per_query: list[int] = []  # 每次查询的迭代次数
        self.search_rounds = 0
        self._load()

    def _load(self):
        if self.data_path.exists():
            with open(self.data_path) as f:
                data = json.load(f)
                for k, v in data.items():
                    if hasattr(self, k):
                        setattr(self, k, v)

    def save(self):
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    def record_parse(self, success: bool, fallback: bool = False, deepseek_fallback: bool = False):
        if success:
            self.parse_success += 1
        if fallback:
            self.parse_fallback += 1
        if deepseek_fallback:
            self.parse_deepseek_fallback += 1
        self.save()

    def record_iterations(self, iterations: int):
        self.iterations_per_query.append(iterations)
        self.save()

    def get_parse_success_rate(self) -> float:
        total = self.parse_success + self.parse_fallback + self.parse_deepseek_fallback
        return self.parse_success / total if total > 0 else 0.0


# 全局单例
metrics = Metrics()
