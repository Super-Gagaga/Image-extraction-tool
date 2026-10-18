"""默认分段器工厂：按模型文件名选择动态输入尺寸。"""

from pathlib import Path

from image_extraction_tool.infrastructure.model_store import ModelStore
from image_extraction_tool.ui import main_window
from image_extraction_tool.ui.main_window import MainWindow


class _RecorderSegmenter:
    name = "recorder"

    def __init__(self, path: Path, *, dynamic_input_size: tuple[int, int] = (320, 320)) -> None:
        self.path = path
        self.dynamic_input_size = dynamic_input_size

    def load(self) -> None:
        pass

    def predict(self, image, progress=None, cancel_token=None):
        raise AssertionError("本测试不执行推理")


def test_default_factory_uses_1024_for_birefnet_and_320_for_unknown(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("IMAGE_EXTRACTION_INPUT_SIZE", raising=False)
    monkeypatch.setattr(main_window, "OnnxU2NetSegmenter", _RecorderSegmenter)
    model = tmp_path / "birefnet-general.onnx"
    model.write_bytes(b"mock")
    window = MainWindow(model_store=ModelStore(model))
    qtbot.addWidget(window)

    segmenter = window._segmenter_factory(model)
    assert isinstance(segmenter, _RecorderSegmenter)
    assert segmenter.dynamic_input_size == (1024, 1024)

    custom = tmp_path / "custom-model.onnx"
    custom.write_bytes(b"mock")
    assert window._segmenter_factory(custom).dynamic_input_size == (320, 320)
