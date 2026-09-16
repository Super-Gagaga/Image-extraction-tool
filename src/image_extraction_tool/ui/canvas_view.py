"""支持缩放、平移与画笔交互的图片画布。"""

from __future__ import annotations

from PIL import Image
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QRubberBand

from image_extraction_tool.domain.document import ToolType


# 缩放上下限，避免极端缩放下出现数值问题
MIN_SCALE = 0.02
MAX_SCALE = 64.0


def pil_to_qimage(image: Image.Image) -> QImage:
    """由 RGBA 的 Pillow 图像生成独占数据的 QImage 副本。"""
    rgba = image if image.mode == "RGBA" else image.convert("RGBA")
    raw = rgba.tobytes("raw", "RGBA")
    # QImage 直接引用 raw 缓冲区，必须 copy() 成自己持有数据，否则 raw 被回收后会悬空
    return QImage(
        raw,
        rgba.width,
        rgba.height,
        rgba.width * 4,
        QImage.Format.Format_RGBA8888,
    ).copy()


class CanvasView(QGraphicsView):
    """场景坐标即原图像素坐标的图片视图。"""

    zoom_changed = Signal(float)
    # 画笔事件用 object 传递 QPointF：(起点, 终点) 表示一段笔迹
    brush_segment = Signal(object, object)
    brush_started = Signal()
    brush_finished = Signal()
    brush_cancelled = Signal()
    color_point_selected = Signal(object)
    color_region_selected = Signal(object)
    hover_position_changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("canvasView")
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._image_item = QGraphicsPixmapItem()
        # 使用最近邻显示：放大时每个画布色块都对应一个原图像素，
        # 不在相邻像素之间生成导出结果中不存在的平滑颜色。
        self._image_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self._scene.addItem(self._image_item)
        self._has_image = False
        self._panning = False
        self._pan_origin = QPoint()
        # 左键工具（None 表示左键不参与编辑）；绘画状态由按下/移动/松开依次维护
        self._active_tool = None
        self._brushing = False
        self._brush_last_position: QPointF | None = None
        self._selecting_color = False
        self._selection_origin = QPoint()
        self._selection_band = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())

        self.setBackgroundBrush(self._checkerboard_brush())
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        # 缩放锚点交给 wheelEvent 手动计算，保持指针下的像素不动
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    @staticmethod
    def _checkerboard_brush() -> QBrush:
        """构造透明区域使用的棋盘格背景笔刷。"""
        tile = QPixmap(24, 24)
        tile.fill(QColor("#f0f0f0"))
        painter = QPainter(tile)
        # 24×24 的贴图里画出对角两个 12×12 深色方块，平铺后即为棋盘格
        painter.fillRect(0, 0, 12, 12, QColor("#d0d0d0"))
        painter.fillRect(12, 12, 12, 12, QColor("#d0d0d0"))
        painter.end()
        return QBrush(tile)

    @property
    def has_image(self) -> bool:
        return self._has_image

    @property
    def zoom_factor(self) -> float:
        # 视图变换矩阵的 m11 分量即水平缩放系数
        return self.transform().m11()

    def set_image(self, image: Image.Image) -> None:
        """载入图片并按窗口大小适应显示。"""
        pixmap = QPixmap.fromImage(pil_to_qimage(image))
        self._image_item.setPixmap(pixmap)
        # 场景矩形与图片像素尺寸一致，保证场景坐标 == 图片坐标
        self._scene.setSceneRect(0.0, 0.0, float(pixmap.width()), float(pixmap.height()))
        self._has_image = True
        self.fit_image()

    def update_image(self, image: Image.Image) -> None:
        """替换同尺寸预览缓存，并保留当前缩放和平移。

        与 set_image 的区别是不重置视图变换，因此画笔编辑时可以边画边刷新预览。
        """
        pixmap = QPixmap.fromImage(pil_to_qimage(image))
        self._image_item.setPixmap(pixmap)
        self._scene.setSceneRect(0.0, 0.0, float(pixmap.width()), float(pixmap.height()))

    def set_active_tool(self, tool: ToolType | None) -> None:
        """设置互斥左键工具，并取消尚未结束的旧工具操作。"""
        self.cancel_current_operation()
        self._active_tool = tool

    def cancel_current_operation(self) -> None:
        """取消尚未完成的笔画或颜色框选。"""
        was_brushing = self._brushing
        self._brushing = False
        self._brush_last_position = None
        self._selecting_color = False
        self._selection_band.hide()
        if was_brushing:
            self.brush_cancelled.emit()

    def _editable_scene_position(self, event: QMouseEvent) -> QPointF | None:
        """返回事件对应的原图坐标；落在图片之外时返回 None。

        图片外的左键拖动不参与绘画，直接交回基类处理。
        """
        position = self.mapToScene(event.position().toPoint())
        # boundingRect 与场景矩形一致，即图片像素范围
        bounds = self._image_item.boundingRect()
        if 0.0 <= position.x() < bounds.width() and 0.0 <= position.y() < bounds.height():
            return position
        return None

    def fit_image(self) -> None:
        """缩放图片直到完整适应窗口，保持宽高比。"""
        if not self._has_image:
            return
        self.resetTransform()
        self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._emit_zoom()

    def show_actual_size(self) -> None:
        """切换到 100% 显示，并尽量保持当前视图中心不变。"""
        if not self._has_image:
            return
        center = self.mapToScene(self.viewport().rect().center())
        self.resetTransform()
        self.centerOn(center)
        self._emit_zoom()

    def image_position(self, viewport_position: QPoint | QPointF) -> QPointF:
        """把视口坐标映射为原图坐标。"""
        return self.mapToScene(viewport_position.toPoint() if isinstance(viewport_position, QPointF) else viewport_position)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """以鼠标指针为锚点进行滚轮缩放。"""
        if not self._has_image or event.angleDelta().y() == 0:
            event.ignore()
            return
        # 记录缩放前后指针所指的场景点，用差值回补平移量实现“像素跟手”
        old_scene_position = self.mapToScene(event.position().toPoint())
        requested = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        current = self.zoom_factor
        target = min(max(current * requested, MIN_SCALE), MAX_SCALE)
        if target == current:
            # 已到达缩放上下限，接受事件但不改变视图
            event.accept()
            return
        self.scale(target / current, target / current)
        new_scene_position = self.mapToScene(event.position().toPoint())
        delta = new_scene_position - old_scene_position
        self.translate(delta.x(), delta.y())
        self._emit_zoom()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """右键按下开始拖动平移；左键在图片内按下则开始一段笔迹。"""
        if event.button() == Qt.MouseButton.RightButton and self._has_image:
            self._panning = True
            self._pan_origin = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._has_image and self._active_tool is ToolType.COLOR_PICKER:
            position = self._editable_scene_position(event)
            if position is not None:
                self._selecting_color = True
                self._selection_origin = event.position().toPoint()
                self._selection_band.setGeometry(QRect(self._selection_origin, self._selection_origin))
                self._selection_band.show()
                event.accept()
                return
        if event.button() == Qt.MouseButton.LeftButton and self._has_image and self._active_tool in (
            ToolType.ERASE_BRUSH,
            ToolType.RESTORE_BRUSH,
        ):
            position = self._editable_scene_position(event)
            if position is not None:
                self._brushing = True
                self._brush_last_position = position
                self.brush_started.emit()
                # 起点与终点相同即单点笔迹，使单击也能落笔
                self.brush_segment.emit(position, position)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """平移时按拖动增量滚动滚动条；绘画时按上一位置到当前位置发出笔迹。"""
        hover_position = self._editable_scene_position(event)
        self.hover_position_changed.emit(hover_position)
        if self._panning:
            current = event.position().toPoint()
            delta = current - self._pan_origin
            self._pan_origin = current
            # 反向滚动使画面跟随鼠标移动方向
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self._selecting_color:
            selection = QRect(self._selection_origin, event.position().toPoint()).normalized()
            self._selection_band.setGeometry(selection)
            event.accept()
            return
        if self._brushing:
            position = self._editable_scene_position(event)
            if position is None:
                # 离开图片即断开当前连续段；再次进入时从新位置起笔，
                # 避免沿图片外的未知鼠标轨迹错误补出一条跨图直线。
                self._brush_last_position = None
            elif self._brush_last_position is None:
                self.brush_segment.emit(position, position)
                self._brush_last_position = position
            else:
                self.brush_segment.emit(self._brush_last_position, position)
                self._brush_last_position = position
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """右键松开结束平移；左键松开补齐最后一段笔迹并结束本次绘制。"""
        if event.button() == Qt.MouseButton.RightButton and self._panning:
            self._panning = False
            self.viewport().unsetCursor()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._selecting_color:
            release_position = event.position().toPoint()
            viewport_rect = QRect(self._selection_origin, release_position).normalized()
            self._selection_band.hide()
            self._selecting_color = False
            if viewport_rect.width() <= 3 and viewport_rect.height() <= 3:
                self.color_point_selected.emit(self.mapToScene(self._selection_origin))
            else:
                scene_start = self.mapToScene(self._selection_origin)
                scene_end = self.mapToScene(release_position)
                region = QRectF(scene_start, scene_end).normalized().intersected(self._image_item.boundingRect())
                if region.width() > 0 and region.height() > 0:
                    self.color_region_selected.emit(region)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._brushing:
            # 松开前再画一次，覆盖快速拖动时最后一段来不及处理的距离
            position = self._editable_scene_position(event)
            if position is not None:
                segment_start = self._brush_last_position or position
                self.brush_segment.emit(segment_start, position)
            self._brushing = False
            self._brush_last_position = None
            self.brush_finished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self.hover_position_changed.emit(None)
        super().leaveEvent(event)

    def _emit_zoom(self) -> None:
        """把当前缩放倍数广播给状态栏等订阅者。"""
        self.zoom_changed.emit(self.zoom_factor)
