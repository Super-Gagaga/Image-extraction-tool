"""在 Qt 工作线程中执行全尺寸 PNG 合成与编码。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from image_extraction_tool.domain.document import ImageDocument
from image_extraction_tool.errors import ExportError
from image_extraction_tool.infrastructure.exporter import export_png


class ExportWorker(QObject):
    """一次性导出任务；完成和失败均通过信号返回 UI 线程。"""

    # 成功时携带导出后的路径，失败时携带面向用户的错误文案
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, document: ImageDocument, destination: Path) -> None:
        super().__init__()
        self._document = document
        self._destination = destination

    @Slot()
    def run(self) -> None:
        # 本槽在工作线程中执行，结果只通过信号回传，避免跨线程直接操作界面
        try:
            result_path = export_png(self._document, self._destination)
        except ExportError as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result_path)
