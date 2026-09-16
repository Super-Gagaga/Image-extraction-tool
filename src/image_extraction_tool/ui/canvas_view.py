"""支持缩放与平移的图片画布。"""

from __future__ import annotations

from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("canvasView")
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._image_item = QGraphicsPixmapItem()
        # 缩放时使用平滑插值，避免像素化锯齿
        self._image_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._image_item)
        self._has_image = False
        self._panning = False
        self._pan_origin = QPoint()

        self.setBackgroundBrush(self._checkerboard_brush())
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        # 缩放锚点交给 wheelEvent 手动计算，保持指针下的像素不动
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

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
        """右键按下开始拖动平移。"""
        if event.button() == Qt.MouseButton.RightButton and self._has_image:
            self._panning = True
            self._pan_origin = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """平移过程中按拖动增量滚动滚动条。"""
        if self._panning:
            current = event.position().toPoint()
            delta = current - self._pan_origin
            self._pan_origin = current
            # 反向滚动使画面跟随鼠标移动方向
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """右键松开结束拖动平移并恢复光标。"""
        if event.button() == Qt.MouseButton.RightButton and self._panning:
            self._panning = False
            self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _emit_zoom(self) -> None:
        """把当前缩放倍数广播给状态栏等订阅者。"""
        self.zoom_changed.emit(self.zoom_factor)
