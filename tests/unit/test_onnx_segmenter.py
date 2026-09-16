"""ONNX 智能抠图适配器的预处理、输出恢复和缓存测试。"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_extraction_tool.domain.segmentation import CancelToken
from image_extraction_tool.errors import InferenceError, ModelLoadError, ModelMissingError, SegmentationCancelled
from image_extraction_tool.infrastructure.model_store import (
    MODEL_ENVIRONMENT_VARIABLE,
    ModelStore,
    default_model_path,
)
from image_extraction_tool.infrastructure.segmenters.onnx_u2net import OnnxU2NetSegmenter


class _Input:
    def __init__(self, shape) -> None:
        self.name = "input"
        self.shape = shape


class _Session:
    def __init__(self, output: np.ndarray, shape=(1, 3, 2, 2)) -> None:
        self.output = output
        self.input = _Input(shape)
        self.feed: dict[str, np.ndarray] | None = None

    def get_inputs(self):
        return [self.input]

    def run(self, _output_names, feed):
        self.feed = feed
        return [self.output]


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    OnnxU2NetSegmenter.clear_session_cache()


def test_preprocess_and_output_restore_keep_size_mode_and_direction(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "u2net.onnx"
    model.write_bytes(b"test model placeholder for mocked runtime")
    output = np.array([[[[0.0, 1.0], [0.25, 0.75]]]], dtype=np.float32)
    session = _Session(output)
    monkeypatch.setattr(
        "image_extraction_tool.infrastructure.segmenters.onnx_u2net.ort.InferenceSession",
        lambda *_args, **_kwargs: session,
    )
    source = Image.new("RGBA", (4, 2), (100, 120, 140, 200))
    progress: list[int] = []

    mask = OnnxU2NetSegmenter(model).predict(source, progress.append)

    assert mask.mode == "L"
    assert mask.size == source.size
    assert mask.getpixel((0, 0)) == 0
    assert mask.getpixel((3, 0)) == 255
    assert mask.getpixel((0, 1)) == 64
    assert mask.getpixel((3, 1)) == 191
    assert progress == sorted(progress)
    assert progress[-1] == 100
    assert session.feed is not None
    tensor = session.feed["input"]
    assert tensor.shape == (1, 3, 2, 2)
    assert tensor.dtype == np.float32


def test_dynamic_input_and_session_are_reused(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "dynamic.onnx"
    model.write_bytes(b"mock")
    created: list[_Session] = []

    def make_session(*_args, **_kwargs):
        session = _Session(np.ones((1, 1, 5, 7), dtype=np.float32), (1, 3, "height", "width"))
        created.append(session)
        return session

    monkeypatch.setattr(
        "image_extraction_tool.infrastructure.segmenters.onnx_u2net.ort.InferenceSession",
        make_session,
    )
    first = OnnxU2NetSegmenter(model, dynamic_input_size=(7, 5))
    second = OnnxU2NetSegmenter(model, dynamic_input_size=(7, 5))

    first.predict(Image.new("RGB", (9, 4), "white"))
    second.predict(Image.new("RGB", (9, 4), "white"))

    assert len(created) == 1
    assert created[0].feed is not None
    assert created[0].feed["input"].shape == (1, 3, 5, 7)


def test_cancel_and_invalid_output_never_return_partial_mask(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"mock")
    session = _Session(np.zeros((1, 2, 2, 2), dtype=np.float32))
    monkeypatch.setattr(
        "image_extraction_tool.infrastructure.segmenters.onnx_u2net.ort.InferenceSession",
        lambda *_args, **_kwargs: session,
    )
    segmenter = OnnxU2NetSegmenter(model)
    token = CancelToken()
    token.cancel()

    with pytest.raises(SegmentationCancelled):
        segmenter.predict(Image.new("RGB", (2, 2)), cancel_token=token)
    with pytest.raises(InferenceError, match="输出形状"):
        segmenter.predict(Image.new("RGB", (2, 2)))


def test_model_store_reports_missing_model_without_changing_selection(tmp_path: Path) -> None:
    missing = tmp_path / "missing.onnx"
    store = ModelStore(missing)

    with pytest.raises(ModelMissingError, match="未找到本地模型"):
        store.require_model()
    assert store.model_path == missing
    with pytest.raises(ModelMissingError, match=".onnx"):
        store.select(tmp_path / "model.bin")
    assert store.model_path == missing


def test_real_onnx_runtime_error_is_translated_to_domain_error(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.onnx"
    invalid.write_bytes(b"not an onnx protobuf")

    with pytest.raises(ModelLoadError, match="无法加载 ONNX 模型"):
        OnnxU2NetSegmenter(invalid).load()


def test_environment_model_path_and_session_cache_limit(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / "configured.onnx"
    monkeypatch.setenv(MODEL_ENVIRONMENT_VARIABLE, str(configured))
    assert default_model_path() == configured

    created: list[_Session] = []

    def make_session(*_args, **_kwargs):
        session = _Session(np.ones((1, 1, 2, 2), dtype=np.float32))
        created.append(session)
        return session

    monkeypatch.setattr(
        "image_extraction_tool.infrastructure.segmenters.onnx_u2net.ort.InferenceSession",
        make_session,
    )
    for index in range(3):
        model = tmp_path / f"model-{index}.onnx"
        model.write_bytes(bytes([index]))
        OnnxU2NetSegmenter(model).load()

    assert len(created) == 3
    assert len(OnnxU2NetSegmenter._session_cache) == 2
