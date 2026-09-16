"""手动画笔、工具互斥和后台导出的真实 Qt 事件测试。"""

from pathlib import Path
from math import floor

import pytest
from PIL import Image
from PySide6.QtCore import QPointF, Qt

from image_extraction_tool.domain.compositor import compose_alpha
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
