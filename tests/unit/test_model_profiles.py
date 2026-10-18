"""动态输入尺寸预设与解析测试。"""

from pathlib import Path

import pytest

from image_extraction_tool.infrastructure.model_profiles import (
    DEFAULT_DYNAMIC_INPUT_SIZE,
    INPUT_SIZE_ENVIRONMENT_VARIABLE,
    parse_input_size,
    resolve_dynamic_input_size,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1024", (1024, 1024)),
        ("1024x768", (1024, 768)),
        ("640×480", (640, 480)),
        (" 512 ", (512, 512)),
    ],
)
def test_parse_input_size_accepts_common_forms(value: str, expected: tuple[int, int]) -> None:
    assert parse_input_size(value) == expected


@pytest.mark.parametrize("value", ["", "abc", "1024x", "0x512", "1x2x3", "-1"])
def test_parse_input_size_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_input_size(value)


def test_known_models_default_to_1024_and_others_keep_320(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(INPUT_SIZE_ENVIRONMENT_VARIABLE, raising=False)

    assert resolve_dynamic_input_size(tmp_path / "BiRefNet-general.onnx") == (1024, 1024)
    assert resolve_dynamic_input_size(tmp_path / "rmbg-1.4.onnx") == (1024, 1024)
    assert resolve_dynamic_input_size(tmp_path / "ISNet.onnx") == (1024, 1024)
    assert resolve_dynamic_input_size(tmp_path / "u2net.onnx") == DEFAULT_DYNAMIC_INPUT_SIZE


def test_environment_variable_overrides_filename_preset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(INPUT_SIZE_ENVIRONMENT_VARIABLE, "640x360")
    assert resolve_dynamic_input_size(tmp_path / "birefnet-general.onnx") == (640, 360)

    monkeypatch.setenv(INPUT_SIZE_ENVIRONMENT_VARIABLE, "768")
    assert resolve_dynamic_input_size(tmp_path / "u2net.onnx") == (768, 768)


def test_invalid_environment_variable_raises_with_readable_message(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(INPUT_SIZE_ENVIRONMENT_VARIABLE, "oops")
    with pytest.raises(ValueError, match="输入尺寸"):
        resolve_dynamic_input_size(tmp_path / "u2net.onnx")
