"""图片读取的集成测试。"""

from pathlib import Path

import pytest
from PIL import Image

from image_extraction_tool.errors import ImageLoadError
from image_extraction_tool.infrastructure.image_io import load_image


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg", ".webp"])
def test_load_supported_image_as_rgba(tmp_path: Path, suffix: str) -> None:
    """验证各受支持格式都能读入并统一为 RGBA。"""
    path = tmp_path / f"sample{suffix}"
    Image.new("RGB", (7, 5), (20, 40, 60)).save(path)

    document = load_image(path)

    assert document.original_rgba.mode == "RGBA"
    assert document.size == (7, 5)
    assert document.source_path == path


def test_invalid_image_does_not_decode(tmp_path: Path) -> None:
    """验证损坏文件会抛出带“损坏”提示的 ImageLoadError。"""
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")

    with pytest.raises(ImageLoadError, match="损坏"):
        load_image(path)


def test_load_preserves_rgba_pixels_and_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "rgba.png"
    source = Image.new("RGBA", (2, 1))
    source.putdata([(1, 2, 3, 0), (250, 240, 230, 127)])
    source.save(path)

    document = load_image(path)

    assert document.size == (2, 1)
    assert list(document.original_rgba.getdata()) == [(1, 2, 3, 0), (250, 240, 230, 127)]


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    """验证不支持的扩展名会抛出带“仅支持”提示的 ImageLoadError。"""
    with pytest.raises(ImageLoadError, match="仅支持"):
        load_image(tmp_path / "sample.bmp")
