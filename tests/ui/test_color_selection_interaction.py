"""取色、框选多色、悬停信息和独立清空的真实 Qt 事件测试。"""

from pathlib import Path

from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from image_extraction_tool.domain.document import ToolType
from image_extraction_tool.ui.main_window import MainWindow


def _color_window(qtbot, tmp_path: Path, image: Image.Image) -> MainWindow:
    source = tmp_path / "colors.png"
    image.save(source)
    window = MainWindow()
    window.resize(980, 620)
    qtbot.addWidget(window)
    window.show()
    assert window.load_image_path(source)
    window.canvas.show_actual_size()
    return window


def _viewport_point(window: MainWindow, x: float, y: float):
    return window.canvas.mapFromScene(QPointF(x, y))


def test_point_pick_hover_and_brush_mode_are_separate(qtbot, tmp_path: Path) -> None:
    image = Image.new("RGBA", (120, 80), (10, 20, 30, 255))
    for x in range(60, 120):
        for y in range(80):
            image.putpixel((x, y), (200, 100, 50, 255))
    window = _color_window(qtbot, tmp_path, image)
    window.color_panel.tolerance_spin.setValue(0)
    window.color_picker_action.trigger()
    first = _viewport_point(window, 20.5, 20.5)
    second = _viewport_point(window, 90.5, 20.5)

    # 右键仍只负责平移，不能新增颜色。
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.RightButton, pos=first)
    assert window.document is not None
    assert window.document.selected_colors == []

    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=first)
    assert [selected.rgb for selected in window.document.selected_colors] == [(10, 20, 30)]
    assert window.document.color_mask.getpixel((20, 20)) == 0
    assert window.document.color_mask.getpixel((90, 20)) == 255

    # 无按键 hover 在 Windows 的 headless Qt 后端不会稳定更新系统光标，
    # 直接向 viewport 投递同等的真实 QMouseEvent，以验证画布事件链本身。
    hover_event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(second),
        QPointF(window.canvas.viewport().mapToGlobal(second)),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(window.canvas.viewport(), hover_event)
    assert window.color_panel.hover_swatch.property("rgb") == (200, 100, 50)
    assert "RGB(200, 100, 50)" in window.color_panel.hover_text.text()
    assert "#C86432" in window.color_panel.hover_text.text()

    window.erase_action.trigger()
    count_before = len(window.document.selected_colors)
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=second)
    assert window._active_tool is ToolType.ERASE_BRUSH
    assert len(window.document.selected_colors) == count_before
    assert window.document.erase_mask.getpixel((90, 20)) > 0


def test_drag_region_adds_deduplicated_multiple_colors(qtbot, tmp_path: Path) -> None:
    image = Image.new("RGBA", (160, 100), (255, 0, 0, 255))
    for x in range(40, 80):
        for y in range(100):
            image.putpixel((x, y), (0, 255, 0, 255))
    for x in range(80, 120):
        for y in range(100):
            image.putpixel((x, y), (0, 0, 255, 255))
    for x in range(120, 160):
        for y in range(100):
            image.putpixel((x, y), (255, 0, 0, 255))
    window = _color_window(qtbot, tmp_path, image)
    window.color_picker_action.trigger()
    start = _viewport_point(window, 1.5, 1.5)
    end = _viewport_point(window, 158.5, 98.5)

    qtbot.mousePress(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(window.canvas.viewport(), pos=end)
    qtbot.mouseRelease(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)

    assert window.document is not None
    values = {selected.rgb for selected in window.document.selected_colors}
    assert values == {(255, 0, 0), (0, 255, 0), (0, 0, 255)}
    assert window.color_panel.color_list.count() == 3
    assert window.document.color_mask.getextrema() == (0, 0)


def test_tolerance_rebuild_and_independent_clear_operations(qtbot, tmp_path: Path) -> None:
    image = Image.new("RGBA", (120, 80), (100, 100, 100, 255))
    for x in range(60, 120):
        for y in range(80):
            image.putpixel((x, y), (110, 100, 100, 255))
    window = _color_window(qtbot, tmp_path, image)
    window.color_panel.tolerance_spin.setValue(0)
    window.color_picker_action.trigger()
    exact = _viewport_point(window, 20.5, 20.5)
    nearby = _viewport_point(window, 90.5, 20.5)
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=exact)

    assert window.document is not None
    assert window.document.color_mask.getpixel((90, 20)) == 255
    window.color_panel.tolerance_spin.setValue(10)
    assert window.document.color_mask.getpixel((90, 20)) == 0
    window.color_panel.tolerance_spin.setValue(0)
    assert window.document.color_mask.getpixel((90, 20)) == 255

    window.erase_action.trigger()
    window.brush_size.setValue(1)
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=nearby)
    erase_before = window.document.erase_mask.tobytes()
    qtbot.mouseClick(window.color_panel.clear_button, Qt.MouseButton.LeftButton)
    assert window.document.selected_colors == []
    assert window.document.color_mask.getextrema() == (255, 255)
    assert window.document.erase_mask.tobytes() == erase_before

    # 重新选色后清空画笔，颜色列表与颜色蒙版必须保持不变。
    window.color_picker_action.trigger()
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=exact)
    color_mask_before = window.document.color_mask.tobytes()
    window.clear_brush_action.trigger()
    assert len(window.document.selected_colors) == 1
    assert window.document.color_mask.tobytes() == color_mask_before
    assert window.document.erase_mask.getextrema() == (0, 0)


def test_single_color_can_be_removed_from_visible_list(qtbot, tmp_path: Path) -> None:
    image = Image.new("RGBA", (120, 80), (255, 0, 0, 255))
    for x in range(60, 120):
        for y in range(80):
            image.putpixel((x, y), (0, 255, 0, 255))
    window = _color_window(qtbot, tmp_path, image)
    window.color_panel.tolerance_spin.setValue(0)
    window.color_picker_action.trigger()
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=_viewport_point(window, 20.5, 20.5))
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=_viewport_point(window, 90.5, 20.5))

    assert window.document is not None
    assert window.color_panel.color_list.count() == 2
    window.color_panel.color_list.setCurrentRow(0)
    removed_rgb = window.color_panel.color_list.currentItem().data(Qt.ItemDataRole.UserRole)
    qtbot.mouseClick(window.color_panel.remove_button, Qt.MouseButton.LeftButton)

    assert window.color_panel.color_list.count() == 1
    assert removed_rgb not in {selected.rgb for selected in window.document.selected_colors}
    removed_x = 20 if removed_rgb == (255, 0, 0) else 90
    assert window.document.color_mask.getpixel((removed_x, 20)) == 255
