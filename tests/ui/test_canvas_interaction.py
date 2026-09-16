"""画布交互（平移、缩放、坐标映射）的界面测试。"""

from PIL import Image
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QPainter, QWheelEvent
from PySide6.QtWidgets import QApplication

from image_extraction_tool.ui.canvas_view import CanvasView


def test_right_drag_pans_without_changing_source_image(qtbot) -> None:
    """验证右键拖动只改变滚动位置，不会修改源图像素。"""
    source = Image.new("RGBA", (1000, 800), (20, 40, 60, 255))
    before = source.tobytes()
    view = CanvasView()
    view.resize(320, 240)
    qtbot.addWidget(view)
    view.show()
    view.set_image(source)
    view.show_actual_size()

    start = QPoint(160, 120)
    end = QPoint(90, 70)
    old_h = view.horizontalScrollBar().value()
    old_v = view.verticalScrollBar().value()
    qtbot.mousePress(view.viewport(), Qt.MouseButton.RightButton, pos=start)
    qtbot.mouseMove(view.viewport(), end)
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.RightButton, pos=end)

    assert (view.horizontalScrollBar().value(), view.verticalScrollBar().value()) != (old_h, old_v)
    assert source.tobytes() == before


def test_actual_size_maps_viewport_distance_to_same_image_distance(qtbot) -> None:
    """验证 100% 显示时视口 1 像素距离对应原图 1 像素。"""
    view = CanvasView()
    view.resize(400, 300)
    qtbot.addWidget(view)
    view.show()
    view.set_image(Image.new("RGBA", (800, 600), "white"))
    view.show_actual_size()

    first = view.image_position(QPoint(100, 100))
    second = view.image_position(QPoint(101, 100))

    assert second.x() - first.x() == pytest.approx(1.0)


def test_wheel_zoom_keeps_image_point_under_pointer(qtbot) -> None:
    """验证滚轮缩放后指针下方的图像点保持不动，且缩放倍数为 1.15。"""
    view = CanvasView()
    view.resize(400, 300)
    qtbot.addWidget(view)
    view.show()
    view.set_image(Image.new("RGBA", (800, 600), "white"))
    view.show_actual_size()
    pointer = QPoint(130, 110)
    before = view.image_position(pointer)
    event = QWheelEvent(
        QPointF(pointer),
        QPointF(view.viewport().mapToGlobal(pointer)),
        QPoint(),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )

    QApplication.sendEvent(view.viewport(), event)
    after = view.image_position(pointer)

    assert view.zoom_factor == pytest.approx(1.15)
    assert after.x() == pytest.approx(before.x(), abs=0.01)
    assert after.y() == pytest.approx(before.y(), abs=0.01)


def test_canvas_uses_original_pixel_nearest_neighbor_rendering(qtbot) -> None:
    """画布缩放不得在原图像素之间生成平滑插值颜色。"""
    view = CanvasView()
    qtbot.addWidget(view)

    assert view._image_item.transformationMode() == Qt.TransformationMode.FastTransformation
    assert not view.renderHints() & QPainter.RenderHint.SmoothPixmapTransform
