"""撤销、重做与编辑历史的真实 Qt 交互测试。"""

from math import floor
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QToolButton

from image_extraction_tool.domain.compositor import compose_alpha
from image_extraction_tool.ui.main_window import MainWindow


def _history_window(qtbot, tmp_path: Path) -> MainWindow:
    image = Image.new("RGBA", (160, 100), (20, 40, 60, 180))
    for x in range(80, 160):
        for y in range(100):
            image.putpixel((x, y), (200, 100, 50, 220))
    source = tmp_path / "history.png"
    image.save(source)
    window = MainWindow()
    window.resize(900, 600)
    qtbot.addWidget(window)
    window.show()
    assert window.load_image_path(source)
    window.canvas.show_actual_size()
    return window


def _point(window: MainWindow, x: float, y: float):
    return window.canvas.mapFromScene(QPointF(x, y))


def _preview_alpha(window: MainWindow, x: int, y: int) -> int:
    image = window.canvas._image_item.pixmap().toImage()
    return image.pixelColor(x, y).alpha()


def _click_action(qtbot, window: MainWindow, action) -> None:
    """通过工具栏按钮发送真实鼠标事件，不直接调用业务槽。"""
    button = next(
        child
        for child in window.findChildren(QToolButton)
        if child.defaultAction() is action
    )
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


def test_fast_stroke_is_one_command_and_shortcuts_restore_preview(qtbot, tmp_path: Path) -> None:
    window = _history_window(qtbot, tmp_path)
    window.brush_size.setValue(1)
    start = _point(window, 20.5, 30.5)
    middle = _point(window, 60.5, 30.5)
    end = _point(window, 120.5, 30.5)
    mapped = window.canvas.image_position(middle)
    pixel = floor(mapped.x()), floor(mapped.y())

    qtbot.mousePress(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(window.canvas.viewport(), pos=middle)
    qtbot.mouseMove(window.canvas.viewport(), pos=end)
    qtbot.mouseRelease(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)

    assert window.document is not None
    edited_mask = window.document.erase_mask.tobytes()
    assert window.history.undo_count == 1
    assert compose_alpha(window.document).getpixel(pixel) == 0
    assert _preview_alpha(window, *pixel) == 0

    _click_action(qtbot, window, window.undo_action)
    assert window.document.erase_mask.getbbox() is None
    assert compose_alpha(window.document).getpixel(pixel) == 180
    assert _preview_alpha(window, *pixel) == 180
    assert window.history.redo_count == 1

    _click_action(qtbot, window, window.redo_action)
    assert window.document.erase_mask.tobytes() == edited_mask
    assert compose_alpha(window.document).getpixel(pixel) == 0
    assert _preview_alpha(window, *pixel) == 0


def test_new_brush_after_undo_clears_redo_and_clear_is_reversible(qtbot, tmp_path: Path) -> None:
    window = _history_window(qtbot, tmp_path)
    window.brush_size.setValue(1)
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=_point(window, 20.5, 20.5))
    first_mask = window.document.erase_mask.tobytes()
    window.undo_action.trigger()
    assert window.history.can_redo

    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=_point(window, 40.5, 20.5))
    assert not window.history.can_redo
    assert window.document.erase_mask.tobytes() != first_mask

    before_clear = window.document.erase_mask.tobytes()
    window.clear_brush_action.trigger()
    assert window.document.erase_mask.getbbox() is None
    window.undo_action.trigger()
    assert window.document.erase_mask.tobytes() == before_clear
    window.redo_action.trigger()
    assert window.document.erase_mask.getbbox() is None


def test_color_add_tolerance_remove_and_clear_are_undoable(qtbot, tmp_path: Path) -> None:
    window = _history_window(qtbot, tmp_path)
    window.color_panel.tolerance_spin.setValue(0)
    window.color_picker_action.trigger()
    left = _point(window, 20.5, 20.5)
    right = _point(window, 120.5, 20.5)

    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=left)
    assert len(window.document.selected_colors) == 1
    assert window.document.color_mask.getpixel((20, 20)) == 0
    window.undo_action.trigger()
    assert window.document.selected_colors == []
    assert window.document.color_mask.getextrema() == (255, 255)
    window.redo_action.trigger()
    assert len(window.document.selected_colors) == 1

    window.color_panel.tolerance_spin.setValue(255)
    assert window.document.color_tolerance == 255
    assert window.document.color_mask.getpixel((120, 20)) == 0
    window.undo_action.trigger()
    assert window.document.color_tolerance == 0
    assert window.document.color_mask.getpixel((120, 20)) == 255

    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=right)
    assert len(window.document.selected_colors) == 2
    window.color_panel.color_list.setCurrentRow(0)
    window.color_panel.remove_button.click()
    assert len(window.document.selected_colors) == 1
    window.undo_action.trigger()
    assert len(window.document.selected_colors) == 2

    window.color_panel.clear_button.click()
    assert window.document.selected_colors == []
    window.undo_action.trigger()
    assert len(window.document.selected_colors) == 2
    assert window.color_panel.color_list.count() == 2


def test_ai_result_is_committed_atomically_and_undoable(qtbot, tmp_path: Path) -> None:
    window = _history_window(qtbot, tmp_path)
    assert window.document is not None

    with pytest.raises(ValueError, match="AI 蒙版"):
        window.apply_ai_mask(Image.new("L", (1, 1), 0))
    assert window.history.undo_count == 0
    assert window.document.ai_mask.getextrema() == (255, 255)

    window.apply_ai_mask(Image.new("L", window.document.size, 64))
    assert window.history.undo_count == 1
    assert window.document.ai_mask.getextrema() == (64, 64)
    assert _preview_alpha(window, 20, 20) == 64

    _click_action(qtbot, window, window.undo_action)
    assert window.document.ai_mask.getextrema() == (255, 255)
    assert _preview_alpha(window, 20, 20) == 180
    _click_action(qtbot, window, window.redo_action)
    assert window.document.ai_mask.getextrema() == (64, 64)
    assert _preview_alpha(window, 20, 20) == 64
