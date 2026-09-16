"""应用主窗口。"""

from __future__ import annotations

from collections.abc import Callable
from math import floor
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, QThread, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox, QProgressBar, QSpinBox

from image_extraction_tool.application.command_stack import (
    ApplyAiMaskCommand,
    BrushStrokeRecorder,
    ColorStateCommand,
    CommandStack,
    CompoundCommand,
    HistoryCommand,
    MaskPatchCommand,
)
from image_extraction_tool.application.export_task import ExportWorker
from image_extraction_tool.application.segmentation_task import SegmentationWorker
from image_extraction_tool.domain.brush import BrushSettings, brush_segment_bounds, paint_mask_segment
from image_extraction_tool.domain.color_selection import SelectedColor, build_color_mask, representative_colors
from image_extraction_tool.domain.compositor import compose_result
from image_extraction_tool.domain.document import ImageDocument, ToolType
from image_extraction_tool.domain.segmentation import CancelToken, Segmenter
from image_extraction_tool.errors import ImageLoadError, ModelMissingError
from image_extraction_tool.infrastructure.image_io import SUPPORTED_EXTENSIONS, load_image
from image_extraction_tool.infrastructure.model_store import ModelStore
from image_extraction_tool.infrastructure.segmenters import OnnxU2NetSegmenter
from image_extraction_tool.ui.canvas_view import CanvasView
from image_extraction_tool.ui.color_panel import ColorPanel


class MainWindow(QMainWindow):
    """顶层窗口；仅在功能真正实现后才挂接对应控件。"""

    # 供测试与外部集成等待导出结果；成功携带路径，失败携带错误文案
    export_finished = Signal(object)
    export_failed = Signal(str)
    segmentation_finished = Signal(object)
    segmentation_cancelled = Signal()
    segmentation_failed = Signal(str)

    def __init__(
        self,
        *,
        model_store: ModelStore | None = None,
        segmenter_factory: Callable[[Path], Segmenter] | None = None,
    ) -> None:
        super().__init__()
        self.document: ImageDocument | None = None
        # 导出任务运行期间非空，用于防止重复启动并在关闭窗口时等待线程收尾
        self._export_thread: QThread | None = None
        self._export_worker: ExportWorker | None = None
        self._segmentation_thread: QThread | None = None
        self._segmentation_worker: SegmentationWorker | None = None
        self._segmentation_token: CancelToken | None = None
        self._segmentation_document: ImageDocument | None = None
        self._model_store = model_store or ModelStore()
        self._segmenter_factory = segmenter_factory or (lambda path: OnnxU2NetSegmenter(path))
        self._active_tool = ToolType.NONE
        self.history = CommandStack(max_commands=100, max_bytes=256 * 1024 * 1024)
        self._stroke_recorder: BrushStrokeRecorder | None = None
        self.setObjectName("mainWindow")
        self.setWindowTitle("智能抠图工具")
        self.resize(1100, 720)
        self.setAcceptDrops(True)

        self.canvas = CanvasView(self)
        self.setCentralWidget(self.canvas)
        self.color_panel = ColorPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.color_panel)
        self._size_label = QLabel("未加载图片")
        self._zoom_label = QLabel("—")
        self._tool_label = QLabel("工具：无")
        self._model_label = QLabel("模型：未运行")
        self._model_progress = QProgressBar()
        self._model_progress.setObjectName("segmentationProgressBar")
        self._model_progress.setRange(0, 100)
        self._model_progress.setFixedWidth(150)
        self._model_progress.hide()
        # 尺寸、缩放与当前工具常驻状态栏右侧，加载提示显示在左侧
        self.statusBar().addPermanentWidget(self._size_label)
        self.statusBar().addPermanentWidget(self._zoom_label)
        self.statusBar().addPermanentWidget(self._tool_label)
        self.statusBar().addPermanentWidget(self._model_label)
        self.statusBar().addPermanentWidget(self._model_progress)
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

        self.smart_cut_action = QAction("智能抠图", self)
        self.smart_cut_action.setObjectName("smartCutAction")
        self.smart_cut_action.setEnabled(False)
        self.smart_cut_action.triggered.connect(self.start_segmentation)
        toolbar.addAction(self.smart_cut_action)

        self.cancel_segmentation_action = QAction("取消智能抠图", self)
        self.cancel_segmentation_action.setObjectName("cancelSegmentationAction")
        self.cancel_segmentation_action.setEnabled(False)
        self.cancel_segmentation_action.triggered.connect(self.cancel_segmentation)
        toolbar.addAction(self.cancel_segmentation_action)

        self.select_model_action = QAction("选择模型", self)
        self.select_model_action.setObjectName("selectModelAction")
        self.select_model_action.triggered.connect(self.select_model_dialog)
        toolbar.addAction(self.select_model_action)

        self.undo_action = QAction("撤销", self)
        self.undo_action.setObjectName("undoAction")
        # Windows 明确使用 Ctrl+Z / Ctrl+Y；显式序列也便于真实键盘事件测试，
        # 避免部分 Qt/PySide 版本把 StandardKey 枚举转换成空序列。
        self.undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self.undo)
        toolbar.addAction(self.undo_action)

        self.redo_action = QAction("重做", self)
        self.redo_action.setObjectName("redoAction")
        self.redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self.redo)
        toolbar.addAction(self.redo_action)

        self.cancel_action = QAction(self)
        self.cancel_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        self.cancel_action.triggered.connect(self.canvas.cancel_current_operation)
        self.addAction(self.cancel_action)

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
        # 取色和两种画笔放在同一个互斥组里，由 Qt 保证“同时只有一个生效”
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.color_picker_action = QAction("选取色块", self, checkable=True)
        self.color_picker_action.setObjectName("colorPickerAction")
        self.color_picker_action.setData(ToolType.COLOR_PICKER)
        self.erase_action = QAction("抠除画笔", self, checkable=True)
        self.erase_action.setObjectName("eraseBrushAction")
        # 用 data 携带工具枚举，避免再维护一份“动作 → 工具”的映射
        self.erase_action.setData(ToolType.ERASE_BRUSH)
        self.restore_action = QAction("恢复画笔", self, checkable=True)
        self.restore_action.setObjectName("restoreBrushAction")
        self.restore_action.setData(ToolType.RESTORE_BRUSH)
        for action in (self.color_picker_action, self.erase_action, self.restore_action):
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
        self.canvas.brush_started.connect(self._begin_brush_stroke)
        self.canvas.brush_segment.connect(self._paint_segment)
        self.canvas.brush_finished.connect(self._finish_brush_stroke)
        self.canvas.brush_cancelled.connect(self._cancel_brush_stroke)
        self.canvas.color_point_selected.connect(self._select_color_point)
        self.canvas.color_region_selected.connect(self._select_color_region)
        self.canvas.hover_position_changed.connect(self._update_hover_color)
        self.color_panel.tolerance_changed.connect(self._change_color_tolerance)
        self.color_panel.remove_requested.connect(self.remove_selected_color)
        self.color_panel.clear_requested.connect(self.clear_selected_colors)

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
        if self._segmentation_thread is not None:
            return False
        try:
            new_document = load_image(path)
        except ImageLoadError as exc:
            QMessageBox.warning(self, "无法打开图片", str(exc))
            return False

        self.document = new_document
        self._stroke_recorder = None
        self.history.clear()
        self._update_history_actions()
        # 画布显示的是合成结果而非原图，保证预览与最终导出一致
        self.canvas.set_image(compose_result(new_document))
        width, height = new_document.size
        self._size_label.setText(f"{width} × {height} px")
        self.fit_action.setEnabled(True)
        self.actual_size_action.setEnabled(True)
        self.export_action.setEnabled(True)
        self.smart_cut_action.setEnabled(True)
        self.color_panel.set_tolerance(new_document.color_tolerance)
        self.color_panel.set_colors(new_document.selected_colors)
        self._set_editing_enabled(True)
        # 载入后默认选中抠除画笔，用户可以立即在图上涂抹
        self.erase_action.setChecked(True)
        self._set_active_tool(ToolType.ERASE_BRUSH)
        self.statusBar().showMessage(f"已加载 {Path(path).name}", 5000)
        return True

    def _set_editing_enabled(self, enabled: bool) -> None:
        """统一开关全部画笔相关控件（导出期间会临时关闭）。"""
        for control in (
            self.color_picker_action,
            self.erase_action,
            self.restore_action,
            self.clear_brush_action,
            self.brush_size,
            self.brush_hardness,
            self.brush_opacity,
        ):
            control.setEnabled(enabled)
        self.color_panel.set_document_enabled(enabled)

    def _tool_selected(self, action: QAction) -> None:
        """互斥组选中变化时切换当前工具。"""
        self._set_active_tool(action.data())

    def _set_active_tool(self, tool: ToolType) -> None:
        """同步当前互斥工具到画布与状态栏。"""
        self._active_tool = tool
        active = tool if tool in (ToolType.COLOR_PICKER, ToolType.ERASE_BRUSH, ToolType.RESTORE_BRUSH) else None
        self.canvas.set_active_tool(active)
        names = {
            ToolType.COLOR_PICKER: "选取色块",
            ToolType.ERASE_BRUSH: "抠除画笔",
            ToolType.RESTORE_BRUSH: "恢复画笔",
        }
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
        tool = self._active_tool
        if tool not in (ToolType.ERASE_BRUSH, ToolType.RESTORE_BRUSH):
            return
        target = self.document.erase_mask if tool is ToolType.ERASE_BRUSH else self.document.restore_mask
        if self._stroke_recorder is None:
            self._begin_brush_stroke()
        bounds = brush_segment_bounds(
            self.document.size,
            (start.x(), start.y()),
            (end.x(), end.y()),
            self.brush_size.value(),
        )
        if bounds is not None and self._stroke_recorder is not None:
            self._stroke_recorder.include_before(bounds)
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

    def _begin_brush_stroke(self) -> None:
        """为当前画笔创建受影响矩形记录器，不复制整张蒙版。"""
        if self.document is None:
            return
        if self._active_tool is ToolType.ERASE_BRUSH:
            self._stroke_recorder = BrushStrokeRecorder(self.document, "erase_mask", "抠除笔画")
        elif self._active_tool is ToolType.RESTORE_BRUSH:
            self._stroke_recorder = BrushStrokeRecorder(self.document, "restore_mask", "恢复笔画")

    def _finish_brush_stroke(self) -> None:
        if self._stroke_recorder is None:
            return
        command = self._stroke_recorder.finish()
        self._stroke_recorder = None
        if command is not None:
            self._push_history(command)

    def _cancel_brush_stroke(self) -> None:
        if self._stroke_recorder is None:
            return
        restored = self._stroke_recorder.cancel()
        self._stroke_recorder = None
        if restored and self.document is not None:
            self.canvas.update_image(compose_result(self.document))

    def _select_color_point(self, position: QPointF) -> None:
        """从不可变原图采集单点颜色并重建颜色蒙版。"""
        if self.document is None or self._active_tool is not ToolType.COLOR_PICKER:
            return
        x, y = floor(position.x()), floor(position.y())
        if not (0 <= x < self.document.size[0] and 0 <= y < self.document.size[1]):
            return
        red, green, blue, _ = self.document.original_pixel(x, y)
        self._add_selected_colors([SelectedColor((red, green, blue))])

    def _select_color_region(self, region: QRectF) -> None:
        """从原图框选区域提取多种代表色并加入可见列表。"""
        if self.document is None or self._active_tool is not ToolType.COLOR_PICKER:
            return
        colors = representative_colors(
            self.document.original_rgba,
            (region.left(), region.top(), region.right(), region.bottom()),
        )
        self._add_selected_colors(colors)

    def _add_selected_colors(self, colors: list[SelectedColor]) -> None:
        if self.document is None:
            return
        before_colors = tuple(self.document.selected_colors)
        before_tolerance = self.document.color_tolerance
        if not self.document.add_selected_colors(colors):
            return
        self._rebuild_color_mask()
        self._push_history(
            ColorStateCommand(
                self.document,
                before_colors,
                before_tolerance,
                tuple(self.document.selected_colors),
                self.document.color_tolerance,
                "添加背景色",
            )
        )

    def _rebuild_color_mask(self) -> None:
        """始终从不可变原图和当前参数重新计算颜色蒙版。"""
        if self.document is None:
            return
        self.document.color_mask = build_color_mask(
            self.document.original_rgba,
            self.document.selected_colors,
            self.document.color_tolerance,
        )
        self.document.mark_edited()
        self.color_panel.set_colors(self.document.selected_colors)
        self.canvas.update_image(compose_result(self.document))

    def _change_color_tolerance(self, tolerance: int) -> None:
        if self.document is None or tolerance == self.document.color_tolerance:
            return
        before_colors = tuple(self.document.selected_colors)
        before_tolerance = self.document.color_tolerance
        self.document.color_tolerance = tolerance
        self._rebuild_color_mask()
        self._push_history(
            ColorStateCommand(
                self.document,
                before_colors,
                before_tolerance,
                tuple(self.document.selected_colors),
                tolerance,
                "修改颜色容差",
            )
        )

    def remove_selected_color(self, index: int) -> None:
        """移除指定列表项并由剩余颜色重建蒙版。"""
        if self.document is None or not 0 <= index < len(self.document.selected_colors):
            return
        before_colors = tuple(self.document.selected_colors)
        before_tolerance = self.document.color_tolerance
        del self.document.selected_colors[index]
        self._rebuild_color_mask()
        self._push_history(
            ColorStateCommand(
                self.document,
                before_colors,
                before_tolerance,
                tuple(self.document.selected_colors),
                self.document.color_tolerance,
                "移除背景色",
            )
        )

    def clear_selected_colors(self) -> None:
        """只清空颜色选择，保留抠除与恢复画笔。"""
        if self.document is None or not self.document.selected_colors:
            return
        before_colors = tuple(self.document.selected_colors)
        before_tolerance = self.document.color_tolerance
        self.document.clear_color_selection()
        self.color_panel.set_colors([])
        self.canvas.update_image(compose_result(self.document))
        self.statusBar().showMessage("已清空颜色选择", 3000)
        self._push_history(
            ColorStateCommand(
                self.document,
                before_colors,
                before_tolerance,
                (),
                self.document.color_tolerance,
                "清空颜色选择",
            )
        )

    def _update_hover_color(self, position: QPointF | None) -> None:
        """从不可变原图显示指针下的色块、RGB 与 HEX。"""
        if self.document is None or position is None:
            self.color_panel.set_hover_color(None)
            return
        x, y = floor(position.x()), floor(position.y())
        if not (0 <= x < self.document.size[0] and 0 <= y < self.document.size[1]):
            self.color_panel.set_hover_color(None)
            return
        red, green, blue, _ = self.document.original_pixel(x, y)
        self.color_panel.set_hover_color((red, green, blue))

    def clear_brush_edits(self) -> None:
        """清空抠除与恢复画笔，并刷新预览。"""
        if self.document is None:
            return
        before = {
            "erase_mask": self.document.erase_mask.copy(),
            "restore_mask": self.document.restore_mask.copy(),
        }
        self.document.clear_brush_masks()
        self.canvas.update_image(compose_result(self.document))
        self.statusBar().showMessage("已清空抠除与恢复画笔修改", 3000)
        commands: list[MaskPatchCommand] = []
        for mask_name in ("erase_mask", "restore_mask"):
            # 清空只会改变原本非零的区域，历史数据也只保存该区域，
            # 避免大图上的稀疏画笔因一次清空而退化为整图历史副本。
            changed_bbox = before[mask_name].getbbox()
            if changed_bbox is not None:
                after = getattr(self.document, mask_name)
                commands.append(
                    MaskPatchCommand.from_images(
                        self.document,
                        mask_name,
                        changed_bbox,
                        before[mask_name].crop(changed_bbox),
                        after.crop(changed_bbox),
                        description="清空画笔修改",
                    )
                )
        if commands:
            self._push_history(CompoundCommand(tuple(commands), "清空画笔修改"))

    def apply_ai_mask(self, new_mask: Image.Image) -> None:
        """原子应用一个已成功生成的 AI 蒙版，并把它写入编辑历史。

        阶段 5 的后台推理只需在成功信号中调用本入口；尺寸或模式校验失败
        会在写入文档和历史之前抛错，当前编辑状态保持不变。
        """
        if self.document is None:
            raise RuntimeError("尚未加载图片")
        command = ApplyAiMaskCommand.create(self.document, new_mask)
        command.redo()
        self._push_history(command)
        self._sync_document_to_ui()

    def select_model_dialog(self) -> None:
        """选择本地 ONNX 模型；不会复制或上传模型文件。"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择本地抠图模型",
            str(self._model_store.model_path.parent),
            "ONNX 模型 (*.onnx)",
        )
        if not path:
            return
        try:
            self._model_store.select(path)
        except ModelMissingError as exc:
            QMessageBox.warning(self, "模型不可用", str(exc))
            return
        self._model_label.setText(f"模型：{Path(path).name}")
        self.statusBar().showMessage("已选择本地模型", 3000)

    def start_segmentation(self) -> bool:
        """在独立线程中加载模型并对不可变原图副本执行智能抠图。"""
        if self.document is None or self._segmentation_thread is not None or self._export_thread is not None:
            return False
        try:
            model_path = self._model_store.require_model()
        except ModelMissingError as exc:
            QMessageBox.warning(self, "缺少智能抠图模型", str(exc))
            self._model_label.setText("模型：文件缺失")
            return False

        token = CancelToken()
        segmenter = self._segmenter_factory(model_path)
        worker = SegmentationWorker(segmenter, self.document.original_rgba, token)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._segmentation_progress)
        worker.succeeded.connect(self._segmentation_succeeded)
        worker.cancelled.connect(self._segmentation_was_cancelled)
        worker.failed.connect(self._segmentation_failed)
        for signal in (worker.succeeded, worker.cancelled, worker.failed):
            signal.connect(thread.quit)
            signal.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._segmentation_thread_finished)

        self._segmentation_thread = thread
        self._segmentation_worker = worker
        self._segmentation_token = token
        self._segmentation_document = self.document
        self._set_segmentation_busy(True)
        thread.start()
        return True

    def cancel_segmentation(self) -> None:
        """请求协作式取消；工作线程不会再把结果提交给当前文档。"""
        if self._segmentation_token is None:
            return
        self._segmentation_token.cancel()
        self.cancel_segmentation_action.setEnabled(False)
        self._model_label.setText("模型：正在取消…")

    def _segmentation_progress(self, value: int, message: str) -> None:
        self._model_progress.setValue(min(max(value, 0), 100))
        self._model_label.setText(f"模型：{message}")

    def _segmentation_succeeded(self, mask: Image.Image) -> None:
        token = self._segmentation_token
        if token is not None and token.is_cancelled:
            self._segmentation_was_cancelled()
            return
        if self.document is not self._segmentation_document:
            self._segmentation_failed("图片已发生变化，智能抠图结果未应用")
            return
        try:
            self.apply_ai_mask(mask)
        except (RuntimeError, ValueError) as exc:
            self._segmentation_failed(str(exc))
            return
        self._model_label.setText("模型：完成")
        self.segmentation_finished.emit(mask)

    def _segmentation_was_cancelled(self) -> None:
        self._model_label.setText("模型：已取消")
        self.segmentation_cancelled.emit()

    def _segmentation_failed(self, message: str) -> None:
        self._model_label.setText("模型：失败")
        QMessageBox.warning(self, "智能抠图失败", message)
        self.segmentation_failed.emit(message)

    def _set_segmentation_busy(self, busy: bool) -> None:
        has_document = self.document is not None
        self.open_action.setEnabled(not busy)
        self.select_model_action.setEnabled(not busy)
        self.smart_cut_action.setEnabled(has_document and not busy)
        self.cancel_segmentation_action.setEnabled(busy)
        self.export_action.setEnabled(has_document and not busy)
        self._set_editing_enabled(has_document and not busy)
        self._model_progress.setVisible(busy)
        if busy:
            self.undo_action.setEnabled(False)
            self.redo_action.setEnabled(False)
            self._model_progress.setValue(0)
            self._set_active_tool(ToolType.NONE)
        else:
            self._update_history_actions()
            self._restore_checked_tool()

    def _restore_checked_tool(self) -> None:
        if self.color_picker_action.isChecked():
            self._set_active_tool(ToolType.COLOR_PICKER)
        elif self.erase_action.isChecked():
            self._set_active_tool(ToolType.ERASE_BRUSH)
        elif self.restore_action.isChecked():
            self._set_active_tool(ToolType.RESTORE_BRUSH)

    def _segmentation_thread_finished(self) -> None:
        self._segmentation_thread = None
        self._segmentation_worker = None
        self._segmentation_token = None
        self._segmentation_document = None
        self._set_segmentation_busy(False)

    def _push_history(self, command: HistoryCommand) -> None:
        """记录已成功应用的命令并同步动作状态。"""
        self.history.push_applied(command)
        self._update_history_actions()

    def undo(self) -> None:
        """撤销一个完整编辑操作并刷新全部派生 UI。"""
        if self._stroke_recorder is not None:
            # 编辑中的笔画尚未进入历史；撤销键先取消它，不能越过它继续
            # 撤销上一条已完成命令。
            self.canvas.cancel_current_operation()
            self._update_history_actions()
            return
        self.canvas.cancel_current_operation()
        if self.history.undo():
            self._sync_document_to_ui()
            self.statusBar().showMessage("已撤销", 2000)
        self._update_history_actions()

    def redo(self) -> None:
        """重做一个完整编辑操作并刷新全部派生 UI。"""
        self.canvas.cancel_current_operation()
        if self.history.redo():
            self._sync_document_to_ui()
            self.statusBar().showMessage("已重做", 2000)
        self._update_history_actions()

    def _sync_document_to_ui(self) -> None:
        if self.document is None:
            return
        self.color_panel.set_tolerance(self.document.color_tolerance)
        self.color_panel.set_colors(self.document.selected_colors)
        self.canvas.update_image(compose_result(self.document))

    def _update_history_actions(self) -> None:
        enabled = self.document is not None and self._export_thread is None and self._segmentation_thread is None
        self.undo_action.setEnabled(enabled and self.history.can_undo)
        self.redo_action.setEnabled(enabled and self.history.can_redo)
        undo_name = self.history.undo_description
        redo_name = self.history.redo_description
        self.undo_action.setText(f"撤销 {undo_name}" if undo_name else "撤销")
        self.redo_action.setText(f"重做 {redo_name}" if redo_name else "重做")

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
        if self.document is None or self._export_thread is not None or self._segmentation_thread is not None:
            return False
        # 导出期间冻结编辑入口，避免合成读到的蒙版被同时修改
        self._set_editing_enabled(False)
        self.open_action.setEnabled(False)
        self.smart_cut_action.setEnabled(False)
        self.select_model_action.setEnabled(False)
        self.export_action.setEnabled(False)
        self.undo_action.setEnabled(False)
        self.redo_action.setEnabled(False)
        self._set_active_tool(ToolType.NONE)
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
        self.open_action.setEnabled(True)
        self.select_model_action.setEnabled(True)
        self.smart_cut_action.setEnabled(has_document)
        self._set_editing_enabled(has_document)
        self.export_action.setEnabled(has_document)
        self._update_history_actions()
        # 恢复导出前选中的画笔工具
        self._restore_checked_tool()

    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭窗口前等待导出线程结束，避免进程退出时线程仍在使用文档。"""
        if self._export_thread is not None:
            self._export_thread.quit()
            self._export_thread.wait()
        if self._segmentation_token is not None:
            self._segmentation_token.cancel()
        if self._segmentation_thread is not None:
            self._segmentation_thread.quit()
            self._segmentation_thread.wait()
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
