"""config 模块挡板测试 — 确保所有字段可访问，无 AttributeError。"""

from config import settings


def test_all_fields_accessible() -> None:
    """访问每个字段不应抛出异常。"""
    fields = [
        "tavily_api_key",
        "deepseek_api_key",
        "ollama_model_name",
        "ollama_base_url",
        "openai_base_url",
        "confidence_threshold",
        "max_iterations",
    ]
    for field in fields:
        getattr(settings, field)


def test_confidence_threshold_default() -> None:
    """默认可信度阈值为 0.8。"""
    assert settings.confidence_threshold == 0.8


def test_max_iterations_default() -> None:
    """默认最大迭代次数为 3。"""
    assert settings.max_iterations == 3
