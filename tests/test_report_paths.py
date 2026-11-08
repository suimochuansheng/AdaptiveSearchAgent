"""评测报告路径测试：报告必须写到仓库外，且可被环境变量覆盖。"""

from pathlib import Path

import pytest

from config import settings
from src.utils.report_paths import DEFAULT_REPORT_DIR, REPO_ROOT, eval_report_dir


def test_default_report_dir_is_outside_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL_REPORT_DIR", raising=False)
    monkeypatch.setattr(settings, "EVAL_REPORT_DIR", "")

    report_dir = eval_report_dir()

    assert report_dir == DEFAULT_REPORT_DIR
    assert "评测报告" in str(report_dir)
    assert "AdaptiveSearchAgent" in str(report_dir)
    # 报告目录必须在仓库之外（避免再次误提交）
    assert REPO_ROOT not in report_dir.parents


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))

    assert eval_report_dir() == tmp_path
