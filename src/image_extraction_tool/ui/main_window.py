"""应用主窗口。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox, QSpinBox

from image_extraction_tool.application.export_task import ExportWorker
from image_extraction_tool.domain.brush import BrushSettings, paint_mask_segment
from image_extraction_tool.domain.compositor import compose_result
from image_extraction_tool.domain.document import ImageDocument, ToolType
from image_extraction_tool.errors import ImageLoadError
from image_extraction_tool.infrastructure.image_io import SUPPORTED_EXTENSIONS, load_image
from image_extraction_tool.ui.canvas_view import CanvasView


class MainWindow(QMainWindow):
    """顶层窗口；仅在功能真正实现后才挂接对应控件。"""

    # 供测试与外部集成等待导出结果；成功携带路径，失败携带错误文案
    export_finished = Signal(object)
    export_failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.document: ImageDocument | None = None
        # 导出任务运行期间非空，用于防止重复启动并在关闭窗口时等待线程收尾
        self._export_thread: QThread | None = None
        self._export_worker: ExportWorker | None = None
        self.setObjectName("mainWindow")
        self.setWindowTitle("智能抠图工具")
        self.resize(1100, 720)
        self.setAcceptDrops(True)

        self.canvas = CanvasView(self)
        self.setCentralWidget(self.canvas)
        self._size_label = QLabel("未加载图片")
        self._zoom_label = QLabel("—")
        self._tool_label = QLabel("工具：无")
        # 尺寸、缩放与当前工具常驻状态栏右侧，加载提示显示在左侧
        self.statusBar().addPermanentWidget(self._size_label)
        self.statusBar().addPermanentWidget(self._zoom_label)
        self.statusBar().addPermanentWidget(self._tool_label)
        self.statusBar().showMessage("可打开或拖入 PNG、JPG、JPEG、WEBP 图片")

        self._create_actions()

    def _create_actions(self) -> None:
        """创建工具栏动作、画笔控件并绑定快捷键与信号。"""
        toolbar = self.addToolBar("文件和视图")
        toolbar.setObjectName("mainToolbar")

        self.open_action = QAction("打开图片", self)
        self.open_action.setObjectName("openImageAction")
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_image_dialog)
        toolbar.addAction(self.open_action)

        self.export_action = QAction("导出 PNG", self)
        self.export_action.setObjectName("exportPngAction")
        self.export_action.setShortcut(QKeySequence.StandardKey.Save)
        # 未加载文档时禁用，导出过程中也会临时禁用
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(self.export_dialog)
        toolbar.addAction(self.export_action)

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

        toolbar.addSeparator()
        # 两个画笔动作放在同一个互斥组里，由 Qt 保证“同时只有一个生效”
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.erase_action = QAction("抠除画笔", self, checkable=True)
        self.erase_action.setObjectName("eraseBrushAction")
        # 用 data 携带工具枚举，避免再维护一份“动作 → 工具”的映射
        self.erase_action.setData(ToolType.ERASE_BRUSH)
        self.restore_action = QAction("恢复画笔", self, checkable=True)
        self.restore_action.setObjectName("restoreBrushAction")
        self.restore_action.setData(ToolType.RESTORE_BRUSH)
        for action in (self.erase_action, self.restore_action):
            action.setEnabled(False)
            self.tool_group.addAction(action)
            toolbar.addAction(action)
        self.tool_group.triggered.connect(self._tool_selected)

        # 画笔参数控件；取值范围与 BrushSettings 的校验保持一致
        toolbar.addWidget(QLabel("大小"))
        self.brush_size = QSpinBox()
        self.brush_size.setObjectName("brushSizeSpinBox")
        self.brush_size.setRange(1, 200)
        self.brush_size.setValue(20)
        self.brush_size.setSuffix(" px")
        self.brush_size.setEnabled(False)
        toolbar.addWidget(self.brush_size)

        toolbar.addWidget(QLabel("硬度"))
        self.brush_hardness = QSpinBox()
        self.brush_hardness.setObjectName("brushHardnessSpinBox")
        self.brush_hardness.setRange(0, 100)
        self.brush_hardness.setValue(100)
        self.brush_hardness.setSuffix("%")
        self.brush_hardness.setEnabled(False)
        toolbar.addWidget(self.brush_hardness)

        toolbar.addWidget(QLabel("不透明度"))
        self.brush_opacity = QSpinBox()
        self.brush_opacity.setObjectName("brushOpacitySpinBox")
        self.brush_opacity.setRange(1, 100)
        self.brush_opacity.setValue(100)
        self.brush_opacity.setSuffix("%")
        self.brush_opacity.setEnabled(False)
        toolbar.addWidget(self.brush_opacity)

        self.clear_brush_action = QAction("清空画笔修改", self)
        self.clear_brush_action.setObjectName("clearBrushAction")
        self.clear_brush_action.setEnabled(False)
        self.clear_brush_action.triggered.connect(self.clear_brush_edits)
        toolbar.addAction(self.clear_brush_action)

        self.canvas.zoom_changed.connect(self._update_zoom)
        self.canvas.brush_segment.connect(self._paint_segment)

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
        # 画布显示的是合成结果而非原图，保证预览与最终导出一致
        self.canvas.set_image(compose_result(new_document))
        width, height = new_document.size
        self._size_label.setText(f"{width} × {height} px")
        self.fit_action.setEnabled(True)
        self.actual_size_action.setEnabled(True)
        self.export_action.setEnabled(True)
        self._set_editing_enabled(True)
        # 载入后默认选中抠除画笔，用户可以立即在图上涂抹
        self.erase_action.setChecked(True)
        self._set_active_tool(ToolType.ERASE_BRUSH)
        self.statusBar().showMessage(f"已加载 {Path(path).name}", 5000)
        return True

    def _set_editing_enabled(self, enabled: bool) -> None:
        """统一开关全部画笔相关控件（导出期间会临时关闭）。"""
        for control in (
            self.erase_action,
            self.restore_action,
            self.clear_brush_action,
            self.brush_size,
            self.brush_hardness,
            self.brush_opacity,
        ):
            control.setEnabled(enabled)

    def _tool_selected(self, action: QAction) -> None:
        """互斥组选中变化时切换当前工具。"""
        self._set_active_tool(action.data())

    def _set_active_tool(self, tool: ToolType) -> None:
        """同步当前工具到画布与状态栏；非画笔类工具一律视为“无”。"""
        self.canvas.set_active_tool(tool if tool in (ToolType.ERASE_BRUSH, ToolType.RESTORE_BRUSH) else None)
        names = {ToolType.ERASE_BRUSH: "抠除画笔", ToolType.RESTORE_BRUSH: "恢复画笔"}
        self._tool_label.setText(f"工具：{names.get(tool, '无')}")

    def _brush_settings(self) -> BrushSettings:
        """把界面上的百分比控件换算为画笔内核使用的 0～1 参数。"""
        return BrushSettings(
            size_px=self.brush_size.value(),
            hardness=self.brush_hardness.value() / 100.0,
            opacity=self.brush_opacity.value() / 100.0,
        )

    def _paint_segment(self, start, end) -> None:
        """把画布发出的一段笔迹写入当前工具的蒙版，并刷新预览。"""
        if self.document is None:
            return
        # 画布只负责报坐标，具体落到哪张蒙版由这里按当前选中工具决定
        tool = self.erase_action.data() if self.erase_action.isChecked() else self.restore_action.data()
        if tool not in (ToolType.ERASE_BRUSH, ToolType.RESTORE_BRUSH):
            return
        target = self.document.erase_mask if tool is ToolType.ERASE_BRUSH else self.document.restore_mask
        affected = paint_mask_segment(
            target,
            (start.x(), start.y()),
            (end.x(), end.y()),
            self._brush_settings(),
        )
        if affected is not None:
            # 只有真正改动了像素才重新合成，避免拖到图片外时做无谓的重绘
            self.document.mark_edited()
            self.canvas.update_image(compose_result(self.document))

    def clear_brush_edits(self) -> None:
        """清空抠除与恢复画笔，并刷新预览。"""
        if self.document is None:
            return
        self.document.clear_brush_masks()
        self.canvas.update_image(compose_result(self.document))
        self.statusBar().showMessage("已清空抠除与恢复画笔修改", 3000)

    def export_dialog(self) -> None:
        """弹出保存对话框，并以其结果启动后台导出。"""
        if self.document is None:
            return
        # 默认文件名沿用源文件名，方便与原图对应
        default_name = "抠图结果.png"
        if self.document.source_path is not None:
            default_name = f"{self.document.source_path.stem}-抠图.png"
        path, _ = QFileDialog.getSaveFileName(self, "导出透明 PNG", default_name, "PNG 图片 (*.png)")
        if path:
            self.start_export(Path(path))

    def start_export(self, destination: Path) -> bool:
        """启动非阻塞全尺寸导出；已有任务运行时拒绝重复启动。"""
        if self.document is None or self._export_thread is not None:
            return False
        # 导出期间冻结编辑入口，避免合成读到的蒙版被同时修改
        self._set_editing_enabled(False)
        self.export_action.setEnabled(False)
        self.canvas.set_active_tool(None)
        self.statusBar().showMessage(f"正在导出 {destination.name}…")

        # worker 移入子线程后只能通过信号交互：成功/失败负责回传结果、
        # 退出事件循环并销毁对象，线程结束再恢复界面状态。
        thread = QThread(self)
        worker = ExportWorker(self.document, destination)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._export_succeeded)
        worker.failed.connect(self._export_failed)
        worker.succeeded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.succeeded.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._export_thread_finished)
        self._export_thread = thread
        self._export_worker = worker
        thread.start()
        return True

    def _export_succeeded(self, path: Path) -> None:
        """在 UI 线程展示导出结果。"""
        self.statusBar().showMessage(f"已导出 {path}", 5000)
        self.export_finished.emit(path)

    def _export_failed(self, message: str) -> None:
        """在 UI 线程提示导出失败。"""
        QMessageBox.warning(self, "导出失败", message)
        self.export_failed.emit(message)

    def _export_thread_finished(self) -> None:
        """线程收尾后清理引用，并按当前文档恢复编辑入口。"""
        self._export_thread = None
        self._export_worker = None
        has_document = self.document is not None
        self._set_editing_enabled(has_document)
        self.export_action.setEnabled(has_document)
        # 恢复导出前选中的画笔工具
        if self.erase_action.isChecked():
            self._set_active_tool(ToolType.ERASE_BRUSH)
        elif self.restore_action.isChecked():
            self._set_active_tool(ToolType.RESTORE_BRUSH)

    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭窗口前等待导出线程结束，避免进程退出时线程仍在使用文档。"""
        if self._export_thread is not None:
            self._export_thread.quit()
            self._export_thread.wait()
        super().closeEvent(event)

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
