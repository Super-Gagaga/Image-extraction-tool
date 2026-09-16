"""桌面应用的日志配置。"""

from __future__ import annotations

import logging
import os
from pathlib import Path


APP_NAME = "ImageExtractionTool"


def default_log_directory() -> Path:
    """返回当前用户的日志目录（仅计算路径，不创建目录）。"""
    local_app_data = os.environ.get("LOCALAPPDATA")
    # Windows 使用 %LOCALAPPDATA%，其他平台退回类 Unix 的用户数据目录
    base = Path(local_app_data) if local_app_data else Path.home() / ".local" / "share"
    return base / APP_NAME / "logs"


def configure_logging(log_directory: Path | None = None) -> Path:
    """只配置一次文件日志，并返回当前生效的日志文件路径。"""
    directory = log_directory or default_log_directory()
    directory.mkdir(parents=True, exist_ok=True)
    log_file = directory / "application.log"

    root_logger = logging.getLogger()
    # 幂等保护：同一日志文件已挂载处理器时不再重复添加，避免日志被写多份
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == log_file.resolve()
        for handler in root_logger.handlers
    ):
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

    return log_file
