"""应用主窗口。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox

from image_extraction_tool.domain.document import ImageDocument
from image_extraction_tool.errors import ImageLoadError
from image_extraction_tool.infrastructure.image_io import SUPPORTED_EXTENSIONS, load_image
from image_extraction_tool.ui.canvas_view import CanvasView


class MainWindow(QMainWindow):
    """顶层窗口；仅在功能真正实现后才挂接对应控件。"""

    def __init__(self) -> None:
        super().__init__()
        self.document: ImageDocument | None = None
        self.setObjectName("mainWindow")
        self.setWindowTitle("智能抠图工具")
        self.resize(1100, 720)
        self.setAcceptDrops(True)

        self.canvas = CanvasView(self)
        self.setCentralWidget(self.canvas)
        self._size_label = QLabel("未加载图片")
        self._zoom_label = QLabel("—")
        # 尺寸与缩放常驻状态栏右侧，加载提示显示在左侧
        self.statusBar().addPermanentWidget(self._size_label)
        self.statusBar().addPermanentWidget(self._zoom_label)
        self.statusBar().showMessage("可打开或拖入 PNG、JPG、JPEG、WEBP 图片")

        self._create_actions()

    def _create_actions(self) -> None:
        """创建工具栏动作并绑定快捷键。"""
        toolbar = self.addToolBar("文件和视图")
        toolbar.setObjectName("mainToolbar")

        open_action = QAction("打开图片", self)
        open_action.setObjectName("openImageAction")
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_image_dialog)
        toolbar.addAction(open_action)

        toolbar.addSeparator()
        self.fit_action = QAction("适应窗口", self)
        self.fit_action.setObjectName("fitImageAction")
        self.fit_action.setShortcut(QKeySequence("0"))
        # 未加载图片前禁用视图类动作
        self.fit_action.setEnabled(False)
        self.fit_action.triggered.connect(self.canvas.fit_image)
        toolbar.addAction(self.fit_action)

        self.actual_size_action = QAction("100%", self)
        self.actual_size_action.setObjectName("actualSizeAction")
        self.actual_size_action.setShortcut(QKeySequence("1"))
        self.actual_size_action.setEnabled(False)
        self.actual_size_action.triggered.connect(self.canvas.show_actual_size)
        toolbar.addAction(self.actual_size_action)

        self.canvas.zoom_changed.connect(self._update_zoom)

    def open_image_dialog(self) -> None:
        """弹出文件选择框并加载用户选中的图片。"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            "",
            "图片 (*.png *.jpg *.jpeg *.webp)",
        )
        if path:
            self.load_image_path(Path(path))

    def load_image_path(self, path: str | Path) -> bool:
        """加载单张图片；失败时保留当前文档不变。

        返回是否加载成功，供拖放事件决定是否接受该操作。
        """
        try:
            new_document = load_image(path)
        except ImageLoadError as exc:
            QMessageBox.warning(self, "无法打开图片", str(exc))
            return False

        self.document = new_document
        self.canvas.set_image(new_document.original_rgba)
        width, height = new_document.size
        self._size_label.setText(f"{width} × {height} px")
        self.fit_action.setEnabled(True)
        self.actual_size_action.setEnabled(True)
        self.statusBar().showMessage(f"已加载 {Path(path).name}", 5000)
        return True

    def _update_zoom(self, scale: float) -> None:
        """把画布缩放倍数格式化为百分比显示。"""
        self._zoom_label.setText(f"{scale * 100:.0f}%")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """仅接受单个、扩展名受支持的本地文件拖动。"""
        urls = event.mimeData().urls()
        if len(urls) == 1 and Path(urls[0].toLocalFile()).suffix.lower() in SUPPORTED_EXTENSIONS:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """松开拖放时加载图片，失败则不接受该操作。"""
        urls = event.mimeData().urls()
        if len(urls) == 1 and self.load_image_path(urls[0].toLocalFile()):
            event.acceptProposedAction()
        else:
            event.ignore()
