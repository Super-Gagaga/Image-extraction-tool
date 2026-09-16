"""日志配置的单元测试。"""

from pathlib import Path

from image_extraction_tool.logging_config import configure_logging


def test_configure_logging_creates_log_file(tmp_path: Path) -> None:
    """验证配置日志时会创建日志文件并返回其路径。"""
    log_file = configure_logging(tmp_path / "logs")

    assert log_file == tmp_path / "logs" / "application.log"
    assert log_file.exists()
