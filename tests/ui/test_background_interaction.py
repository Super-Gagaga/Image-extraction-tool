"""选择背景、清除背景与导出对话框状态测试。"""

from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QFileDialog, QMessageBox

from image_extraction_tool.domain.compositor import compose_result
from image_extraction_tool.infrastructure.model_store import ModelStore
from image_extraction_tool.ui.main_window import MainWindow


def _window(qtbot, tmp_path: Path) -> MainWindow:
    source = tmp_path / "source.png"
    Image.new("RGBA", (4, 3), (10, 20, 30, 255)).save(source)
    window = MainWindow(model_store=ModelStore(tmp_path / "missing.onnx"))
    qtbot.addWidget(window)
    assert window.load_image_path(source)
    # 制造一个透明像素，便于观察背景是否被合成到下方
    assert window.document is not None
    window.document.ai_mask.putpixel((0, 0), 0)
    window.canvas.update_image(compose_result(window.document))
    return window


def test_background_actions_are_disabled_without_document(qtbot, tmp_path: Path) -> None:
    window = MainWindow(model_store=ModelStore(tmp_path / "missing.onnx"))
    qtbot.addWidget(window)

    assert not window.choose_background_action.isEnabled()
    assert not window.clear_background_action.isEnabled()


def test_select_and_clear_background_updates_preview_and_actions(qtbot, tmp_path: Path) -> None:
    window = _window(qtbot, tmp_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (8, 6), (200, 100, 50)).save(background)

    assert window.choose_background_action.isEnabled()
    assert window.background_button.isEnabled()
    assert not window.clear_background_action.isEnabled()
    assert window.set_background_path(background)

    assert window.document is not None
    assert window.document.background is not None
    assert window.document.background.source_path == background
    assert window.clear_background_action.isEnabled()
    assert compose_result(window.document).getpixel((0, 0)) == (200, 100, 50, 255)

    window.clear_background()

    assert window.document.background is None
    assert not window.clear_background_action.isEnabled()
    assert compose_result(window.document).getpixel((0, 0))[3] == 0


def test_invalid_background_keeps_state_and_reports_error(qtbot, tmp_path: Path, monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, message: messages.append(message))
    window = _window(qtbot, tmp_path)
    good = tmp_path / "good.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(good)
    assert window.set_background_path(good)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")

    assert not window.set_background_path(broken)

    assert window.document is not None
    assert window.document.background is not None
    assert window.document.background.source_path == good
    assert messages and "损坏" in messages[0]


def test_background_dialog_uses_selected_file(qtbot, tmp_path: Path, monkeypatch) -> None:
    window = _window(qtbot, tmp_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(background)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(background), "图片"))

    window.select_background_dialog()

    assert window.document is not None
    assert window.document.background is not None
    assert window.document.background.source_path == background


def test_export_dialog_mentions_background_once_selected(qtbot, tmp_path: Path, monkeypatch) -> None:
    window = _window(qtbot, tmp_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(background)
    assert window.set_background_path(background)

    captured: dict[str, str] = {}

    def fake_save(parent, title, default, filter_text):
        captured.update(title=title, default=default)
        return "", ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake_save)

    window.export_dialog()

    assert captured["title"] == "导出带背景 PNG"
    assert captured["default"] == "source-背景合成.png"


def test_export_dialog_keeps_transparent_naming_without_background(qtbot, tmp_path: Path, monkeypatch) -> None:
    window = _window(qtbot, tmp_path)
    captured: dict[str, str] = {}

    def fake_save(parent, title, default, filter_text):
        captured.update(title=title, default=default)
        return "", ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake_save)

    window.export_dialog()

    assert captured["title"] == "导出透明 PNG"
    assert captured["default"] == "source-抠图.png"


def test_loading_new_image_clears_background(qtbot, tmp_path: Path) -> None:
    window = _window(qtbot, tmp_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(background)
    assert window.set_background_path(background)
    assert window.clear_background_action.isEnabled()

    next_source = tmp_path / "next.png"
    Image.new("RGBA", (2, 2), (7, 8, 9, 255)).save(next_source)
    assert window.load_image_path(next_source)

    assert window.document is not None
    assert window.document.background is None
    assert not window.clear_background_action.isEnabled()


def test_background_actions_follow_editing_enabled_state(qtbot, tmp_path: Path) -> None:
    """导出/推理期间必须冻结背景选择，避免导出读到半改状态的背景。"""
    window = _window(qtbot, tmp_path)

    window._set_editing_enabled(False)
    assert not window.background_button.isEnabled()
    assert not window.choose_background_action.isEnabled()
    assert not window.clear_background_action.isEnabled()

    window._set_editing_enabled(True)
    assert window.background_button.isEnabled()
    assert window.choose_background_action.isEnabled()
