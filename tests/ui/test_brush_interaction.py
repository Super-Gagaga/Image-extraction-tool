"""手动画笔、工具互斥和后台导出的真实 Qt 事件测试。"""

from pathlib import Path
from math import floor

import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from image_extraction_tool.domain.compositor import compose_alpha
from image_extraction_tool.ui.canvas_view import MIN_CURSOR_RADIUS_PX
from image_extraction_tool.ui.main_window import MainWindow


def _loaded_window(qtbot, tmp_path: Path, size: tuple[int, int] = (160, 120)) -> MainWindow:
    """构造一个已显示并加载好测试图片的主窗口。"""
    source = tmp_path / "source.png"
    Image.new("RGBA", size, (80, 120, 160, 255)).save(source)
    window = MainWindow()
    window.resize(700, 500)
    qtbot.addWidget(window)
    window.show()
    assert window.load_image_path(source)
    return window


@pytest.mark.parametrize("scale", [0.5, 1.0, 4.0])
def test_one_pixel_brush_is_accurate_at_multiple_zoom_levels(qtbot, tmp_path: Path, scale: float) -> None:
    """验证 1px 画笔在多个缩放级别下都只命中唯一的原图像素。"""
    window = _loaded_window(qtbot, tmp_path)
    window.brush_size.setValue(1)
    window.canvas.resetTransform()
    window.canvas.scale(scale, scale)
    window.canvas.centerOn(50.5, 40.5)
    # 由目标像素反推视口坐标，再按该坐标点击，模拟用户对准某像素落笔
    viewport_point = window.canvas.mapFromScene(QPointF(50.5, 40.5))
    mapped = window.canvas.image_position(viewport_point)
    expected_pixel = (floor(mapped.x()), floor(mapped.y()))

    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=viewport_point)

    assert window.document is not None
    assert window.document.erase_mask.getpixel(expected_pixel) == 255
    assert sum(
        window.document.erase_mask.getpixel((x, y)) > 0
        for y in range(window.document.size[1])
        for x in range(window.document.size[0])
    ) == 1


def test_fast_drag_is_interpolated_and_tool_modes_are_exclusive(qtbot, tmp_path: Path) -> None:
    """验证快速拖动被插值为连续笔迹，且两个画笔工具互斥。"""
    window = _loaded_window(qtbot, tmp_path)
    window.brush_size.setValue(1)
    window.canvas.show_actual_size()
    start = window.canvas.mapFromScene(QPointF(20.5, 50.5))
    end = window.canvas.mapFromScene(QPointF(120.5, 50.5))
    mapped_start = window.canvas.image_position(start)
    mapped_end = window.canvas.image_position(end)

    # 只发一次 move 事件模拟快速拖动，检验画笔自身完成插值
    qtbot.mousePress(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(window.canvas.viewport(), pos=end)
    qtbot.mouseRelease(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)

    assert window.document is not None
    start_x = floor(mapped_start.x())
    end_x = floor(mapped_end.x())
    row = floor(mapped_start.y())
    assert floor(mapped_end.y()) == row
    assert all(window.document.erase_mask.getpixel((x, row)) == 255 for x in range(start_x, end_x + 1))
    # 切换到恢复画笔后抠除画笔应自动取消选中
    window.restore_action.trigger()
    assert window.restore_action.isChecked()
    assert not window.erase_action.isChecked()
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)
    assert compose_alpha(window.document).getpixel((end_x, row)) == 255


def _pixel_at_scene(window: MainWindow, scene_position: QPointF) -> tuple[int, int]:
    """返回点击该原图位置时实际命中的像素坐标。

    ``mapFromScene`` 会把坐标取整回 QPoint，直接按输入坐标取整会差一个像素，
    因此这里按与画布一致的路径反推真实落笔像素。
    """
    viewport_point = window.canvas.mapFromScene(scene_position)
    mapped = window.canvas.image_position(viewport_point)
    return floor(mapped.x()), floor(mapped.y())


def test_erase_works_again_after_restore_brush(qtbot, tmp_path: Path) -> None:
    """先恢复、再抠除同一区域，抠除必须仍然生效（回归：恢复层曾永久压住抠除）。"""
    window = _loaded_window(qtbot, tmp_path)
    window.canvas.show_actual_size()
    window.brush_size.setValue(1)
    scene_position = QPointF(40.5, 30.5)
    point = window.canvas.mapFromScene(scene_position)
    pixel = _pixel_at_scene(window, scene_position)
    assert window.document is not None

    # 1) 抠除一个像素，预览变透明
    window.erase_action.trigger()
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)
    assert window.document.erase_mask.getpixel(pixel) == 255
    assert compose_alpha(window.document).getpixel(pixel) == 0

    # 2) 用恢复画笔在同一像素上恢复为不透明
    window.restore_action.trigger()
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)
    assert compose_alpha(window.document).getpixel(pixel) == 255
    # 恢复必须清除该像素上的抠除，两张蒙版不允许同时命中
    assert window.document.erase_mask.getpixel(pixel) == 0

    # 3) 再次抠除同一像素，预览必须重新变透明
    window.erase_action.trigger()
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)
    assert window.document.restore_mask.getpixel(pixel) == 0
    assert compose_alpha(window.document).getpixel(pixel) == 0


def test_alternating_brush_strokes_undo_as_single_steps(qtbot, tmp_path: Path) -> None:
    """交替使用两支画笔后逐步撤销，每次都应回到上一支画笔的结果。"""
    window = _loaded_window(qtbot, tmp_path)
    window.canvas.show_actual_size()
    window.brush_size.setValue(1)
    scene_position = QPointF(40.5, 30.5)
    point = window.canvas.mapFromScene(scene_position)
    pixel = _pixel_at_scene(window, scene_position)
    assert window.document is not None

    window.erase_action.trigger()
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)
    window.restore_action.trigger()
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)

    # 撤销恢复：像素回到被抠除状态
    window.undo()
    assert compose_alpha(window.document).getpixel(pixel) == 0
    assert window.document.restore_mask.getpixel(pixel) == 0
    assert window.document.erase_mask.getpixel(pixel) == 255

    # 继续撤销抠除：像素回到原图不透明状态
    window.undo()
    assert compose_alpha(window.document).getpixel(pixel) == 255
    assert window.document.erase_mask.getpixel(pixel) == 0

    # 重做两步后应精确回到恢复后的状态
    window.redo()
    window.redo()
    assert window.document.erase_mask.getpixel(pixel) == 0
    assert window.document.restore_mask.getpixel(pixel) == 255
    assert compose_alpha(window.document).getpixel(pixel) == 255


def test_right_drag_never_paints_and_clear_restores_preview(qtbot, tmp_path: Path) -> None:
    """验证右键拖动只平移不落笔，且清空画笔能恢复原始预览。"""
    window = _loaded_window(qtbot, tmp_path, (1000, 800))
    window.canvas.show_actual_size()
    start = window.canvas.mapFromScene(QPointF(500.5, 400.5))
    end = start + window.canvas.mapFromScene(QPointF(50.0, 30.0)) - window.canvas.mapFromScene(QPointF(0.0, 0.0))

    qtbot.mousePress(window.canvas.viewport(), Qt.MouseButton.RightButton, pos=start)
    qtbot.mouseMove(window.canvas.viewport(), pos=end)
    qtbot.mouseRelease(window.canvas.viewport(), Qt.MouseButton.RightButton, pos=end)

    assert window.document is not None
    assert window.document.erase_mask.getextrema() == (0, 0)
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    assert window.document.erase_mask.getextrema()[1] == 255
    window.clear_brush_action.trigger()
    assert window.document.erase_mask.getextrema() == (0, 0)
    assert compose_alpha(window.document).getextrema() == (255, 255)


def test_brush_reentry_does_not_bridge_across_image(qtbot, tmp_path: Path) -> None:
    """笔画移出图片再进入时，应在新位置重新起笔而不是跨图连线。"""
    window = _loaded_window(qtbot, tmp_path)
    window.brush_size.setValue(1)
    window.canvas.resetTransform()
    window.canvas.centerOn(80.0, 60.0)
    start = window.canvas.mapFromScene(QPointF(10.5, 60.5))
    outside = window.canvas.mapFromScene(QPointF(80.0, -20.0))
    reentry = window.canvas.mapFromScene(QPointF(150.5, 60.5))
    mapped_start = window.canvas.image_position(start)
    mapped_reentry = window.canvas.image_position(reentry)
    start_pixel = (floor(mapped_start.x()), floor(mapped_start.y()))
    reentry_pixel = (floor(mapped_reentry.x()), floor(mapped_reentry.y()))
    bridge_pixel = (
        floor((mapped_start.x() + mapped_reentry.x()) / 2.0),
        floor((mapped_start.y() + mapped_reentry.y()) / 2.0),
    )

    qtbot.mousePress(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(window.canvas.viewport(), pos=outside)
    qtbot.mouseMove(window.canvas.viewport(), pos=reentry)
    qtbot.mouseRelease(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=reentry)

    assert window.document is not None
    assert window.document.erase_mask.getpixel(start_pixel) == 255
    assert window.document.erase_mask.getpixel(reentry_pixel) == 255
    assert window.document.erase_mask.getpixel(bridge_pixel) == 0


def test_export_runs_in_worker_and_matches_preview(qtbot, tmp_path: Path) -> None:
    """验证导出在后台线程完成，且文件内容与预览合成一致。"""
    window = _loaded_window(qtbot, tmp_path)
    window.brush_size.setValue(1)
    point = window.canvas.mapFromScene(QPointF(30.5, 25.5))
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)
    output = tmp_path / "exported.png"

    # 等待导出完成信号，确认整个流程异步返回而非阻塞界面
    with qtbot.waitSignal(window.export_finished, timeout=5000):
        assert window.start_export(output)

    assert window._export_thread is not None
    qtbot.waitUntil(lambda: window._export_thread is None, timeout=5000)
    assert window.document is not None
    with Image.open(output) as exported:
        assert exported.mode == "RGBA"
        assert exported.size == window.document.size
        assert exported.getchannel("A").tobytes() == compose_alpha(window.document).tobytes()


def _hover_at_scene(window: MainWindow, scene_position: QPointF) -> None:
    """把指针移到指定原图位置并投递一次真实的鼠标移动事件。

    ``QTest.mouseMove`` 不产生带鼠标跟踪的移动事件，而 QGraphicsView 只在
    收到底层鼠标移动事件时才更新画笔光标，因此这里直接投递 ``QMouseEvent``。
    """
    viewport = window.canvas.viewport()
    position = QPointF(window.canvas.mapFromScene(scene_position))
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        position,
        position,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(viewport, event)


@pytest.mark.parametrize("action_name", ["erase_action", "restore_action"])
def test_brush_tools_show_size_cursor_following_pointer(qtbot, tmp_path: Path, action_name: str) -> None:
    """两种画笔在图片内移动时显示作用范围光标，且半径等于画笔半径。"""
    window = _loaded_window(qtbot, tmp_path)
    window.brush_size.setValue(40)
    window.canvas.show_actual_size()
    getattr(window, action_name).trigger()

    _hover_at_scene(window, QPointF(50.5, 40.5))
    assert window.canvas.brush_cursor_visible
    assert window.canvas.brush_cursor_radius == pytest.approx(20.0)

    # 移出图片范围后不显示，避免在空白处误导用户以为仍可落笔
    _hover_at_scene(window, QPointF(-20.0, 40.5))
    assert not window.canvas.brush_cursor_visible


def test_brush_cursor_tracks_size_changes_and_hides_for_other_tools(qtbot, tmp_path: Path) -> None:
    """调整画笔尺寸立即改变光标半径；切换到取色工具后光标隐藏。"""
    window = _loaded_window(qtbot, tmp_path)
    window.canvas.show_actual_size()
    window.erase_action.trigger()
    _hover_at_scene(window, QPointF(50.5, 40.5))

    window.brush_size.setValue(10)
    assert window.canvas.brush_cursor_radius == pytest.approx(5.0)
    window.brush_size.setValue(80)
    assert window.canvas.brush_cursor_radius == pytest.approx(40.0)

    window.color_picker_action.trigger()
    assert not window.canvas.brush_cursor_visible

    # 切回画笔后无需移动鼠标，也应立即恢复显示
    window.erase_action.trigger()
    assert window.canvas.brush_cursor_visible

    # 指针离开画布后圆环消失，且不会用最后位置被尺寸变化重新唤起
    QApplication.sendEvent(window.canvas.viewport(), QEvent(QEvent.Type.Leave))
    assert not window.canvas.brush_cursor_visible
    window.brush_size.setValue(30)
    assert not window.canvas.brush_cursor_visible


def test_tiny_brush_cursor_keeps_visible_minimum_radius(qtbot, tmp_path: Path) -> None:
    """1px 画笔在小倍率下放大到最小可见半径，放大到高倍率后回归真实尺寸。"""
    window = _loaded_window(qtbot, tmp_path)
    window.brush_size.setValue(1)
    window.erase_action.trigger()
    window.canvas.show_actual_size()

    # 100% 下 0.5 原图像素不足半个屏幕像素，显示半径被抬到最小可见半径
    _hover_at_scene(window, QPointF(50.5, 40.5))
    assert window.canvas.brush_cursor_radius == pytest.approx(MIN_CURSOR_RADIUS_PX)

    # 800% 下真实半径已大于最小可见半径，圆环精确等于 0.5 原图像素
    window.canvas.resetTransform()
    window.canvas.scale(8.0, 8.0)
    _hover_at_scene(window, QPointF(50.5, 40.5))
    assert window.canvas.brush_cursor_radius == pytest.approx(0.5)


def test_brush_cursor_hidden_during_right_button_pan(qtbot, tmp_path: Path) -> None:
    """右键平移期间画笔不生效，作用范围光标应暂时隐藏。"""
    window = _loaded_window(qtbot, tmp_path)
    window.erase_action.trigger()
    point = window.canvas.mapFromScene(QPointF(50.5, 40.5))
    _hover_at_scene(window, QPointF(50.5, 40.5))
    assert window.canvas.brush_cursor_visible

    qtbot.mousePress(window.canvas.viewport(), Qt.MouseButton.RightButton, pos=point)
    assert not window.canvas.brush_cursor_visible

    qtbot.mouseRelease(window.canvas.viewport(), Qt.MouseButton.RightButton, pos=point)
    assert window.canvas.brush_cursor_visible
