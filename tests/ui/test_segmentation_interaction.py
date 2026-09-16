"""智能抠图成功、取消与失败恢复的真实 Qt 事件测试。"""

from pathlib import Path
from threading import Event, get_ident
from time import sleep

from PIL import Image
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QMessageBox, QToolButton

from image_extraction_tool.domain.segmentation import CancelToken
from image_extraction_tool.errors import ModelLoadError
from image_extraction_tool.infrastructure.model_store import ModelStore
from image_extraction_tool.ui.main_window import MainWindow


class _SuccessSegmenter:
    name = "test-success"

    def __init__(self) -> None:
        self.thread_id: int | None = None

    def load(self) -> None:
        self.thread_id = get_ident()

    def predict(self, image, progress=None, cancel_token=None):
        assert cancel_token is not None
        if progress:
            progress(50)
        return Image.new("L", image.size, 0)


class _CancellableSegmenter:
    name = "test-cancellable"

    def __init__(self) -> None:
        self.started = Event()

    def load(self) -> None:
        pass

    def predict(self, image, progress=None, cancel_token: CancelToken | None = None):
        assert cancel_token is not None
        self.started.set()
        while not cancel_token.is_cancelled:
            sleep(0.005)
        cancel_token.raise_if_cancelled()
        raise AssertionError("unreachable")


class _FailingSegmenter:
    name = "test-failure"

    def load(self) -> None:
        raise ModelLoadError("测试模型损坏")

    def predict(self, image, progress=None, cancel_token=None):
        raise AssertionError("加载失败后不得推理")


def _window(qtbot, tmp_path: Path, segmenter) -> MainWindow:
    model = tmp_path / "test.onnx"
    model.write_bytes(b"mock")
    source = tmp_path / "source.png"
    Image.new("RGBA", (120, 80), (30, 60, 90, 180)).save(source)
    window = MainWindow(
        model_store=ModelStore(model),
        segmenter_factory=lambda _path: segmenter,
    )
    window.resize(980, 620)
    qtbot.addWidget(window)
    window.show()
    assert window.load_image_path(source)
    window.canvas.show_actual_size()
    return window


def _click_action(qtbot, window: MainWindow, action) -> None:
    button = next(child for child in window.findChildren(QToolButton) if child.defaultAction() is action)
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


def test_success_runs_off_ui_thread_applies_history_and_allows_manual_repair(qtbot, tmp_path: Path) -> None:
    segmenter = _SuccessSegmenter()
    ui_thread_id = get_ident()
    window = _window(qtbot, tmp_path, segmenter)

    with qtbot.waitSignal(window.segmentation_finished, timeout=3000):
        _click_action(qtbot, window, window.smart_cut_action)
    qtbot.waitUntil(lambda: window._segmentation_thread is None, timeout=3000)

    assert segmenter.thread_id is not None and segmenter.thread_id != ui_thread_id
    assert window.document is not None
    assert window.document.ai_mask.getextrema() == (0, 0)
    assert window.history.undo_count == 1
    assert window.smart_cut_action.isEnabled()

    window.undo_action.trigger()
    assert window.document.ai_mask.getextrema() == (255, 255)
    window.redo_action.trigger()
    assert window.document.ai_mask.getextrema() == (0, 0)

    window.restore_action.trigger()
    point = window.canvas.mapFromScene(40.5, 30.5)
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)
    assert window.document.restore_mask.getbbox() is not None


def test_cancel_keeps_ui_responsive_and_does_not_modify_document(qtbot, tmp_path: Path) -> None:
    segmenter = _CancellableSegmenter()
    window = _window(qtbot, tmp_path, segmenter)
    assert window.document is not None
    before = window.document.ai_mask.tobytes()

    _click_action(qtbot, window, window.smart_cut_action)
    qtbot.waitUntil(segmenter.started.is_set, timeout=3000)
    event_loop_tick: list[bool] = []
    QTimer.singleShot(0, lambda: event_loop_tick.append(True))
    qtbot.waitUntil(lambda: bool(event_loop_tick), timeout=1000)

    with qtbot.waitSignal(window.segmentation_cancelled, timeout=3000):
        _click_action(qtbot, window, window.cancel_segmentation_action)
    qtbot.waitUntil(lambda: window._segmentation_thread is None, timeout=3000)

    assert window.document.ai_mask.tobytes() == before
    assert window.history.undo_count == 0
    assert window.erase_action.isEnabled()
    assert not window.cancel_segmentation_action.isEnabled()
    assert not window._model_progress.isVisible()


def test_load_failure_restores_manual_tools_and_preserves_document(qtbot, tmp_path: Path, monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, message: messages.append(message))
    window = _window(qtbot, tmp_path, _FailingSegmenter())
    assert window.document is not None
    before = window.document.ai_mask.tobytes()

    with qtbot.waitSignal(window.segmentation_failed, timeout=3000):
        _click_action(qtbot, window, window.smart_cut_action)
    qtbot.waitUntil(lambda: window._segmentation_thread is None, timeout=3000)

    assert messages == ["测试模型损坏"]
    assert window.document.ai_mask.tobytes() == before
    assert window.history.undo_count == 0
    assert window.erase_action.isEnabled()
    point = window.canvas.mapFromScene(20.5, 20.5)
    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)
    assert window.document.erase_mask.getbbox() is not None


def test_missing_model_does_not_enter_busy_state(qtbot, tmp_path: Path, monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, message: messages.append(message))
    missing = tmp_path / "missing.onnx"
    source = tmp_path / "source.png"
    Image.new("RGBA", (20, 10), "white").save(source)
    window = MainWindow(model_store=ModelStore(missing))
    qtbot.addWidget(window)
    window.show()
    assert window.load_image_path(source)

    _click_action(qtbot, window, window.smart_cut_action)

    assert len(messages) == 1
    assert "未找到本地模型" in messages[0]
    assert window._segmentation_thread is None
    assert window.erase_action.isEnabled()
    assert window.smart_cut_action.isEnabled()
