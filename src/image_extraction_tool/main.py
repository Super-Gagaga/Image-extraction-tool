"""桌面应用启动引导。"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication, QMessageBox

from image_extraction_tool.logging_config import configure_logging
from image_extraction_tool.ui.main_window import MainWindow


LOGGER = logging.getLogger(__name__)


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """创建或复用进程内唯一的 Qt 应用实例。

    测试会多次调用本函数，因此已存在实例时直接返回，避免重复构造 QApplication。
    """
    existing = QApplication.instance()
    if existing is not None:
        return existing

    application = QApplication(list(argv) if argv is not None else sys.argv)
    application.setApplicationName("智能抠图工具")
    application.setOrganizationName("ImageExtractionTool")
    return application


def main(argv: Sequence[str] | None = None) -> int:
    """启动图形界面，并兜底处理启动阶段的异常。

    返回进程退出码：正常结束为 Qt 事件循环的返回值，启动失败为 1。
    """
    configure_logging()
    application: QApplication | None = None
    try:
        application = create_application(argv)
        window = MainWindow()
        window.show()
        return application.exec()
    except Exception as exc:
        # 记录完整堆栈便于排查，同时用对话框告知用户启动失败
        LOGGER.exception("应用启动失败")
        if application is not None:
            QMessageBox.critical(None, "启动失败", f"应用无法启动：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
