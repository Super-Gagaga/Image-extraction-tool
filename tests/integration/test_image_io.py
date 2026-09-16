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


def test_format_is_detected_from_file_header_not_extension(tmp_path: Path) -> None:
    path = tmp_path / "image.bin"
    Image.new("RGB", (3, 2), (20, 40, 60)).save(path, format="PNG")

    document = load_image(path)

    assert document.size == (3, 2)
    assert document.original_pixel(1, 1) == (20, 40, 60, 255)
    assert len(document.original_bytes()) == 3 * 2 * 4


def test_exif_orientation_is_applied_without_resampling(tmp_path: Path) -> None:
    path = tmp_path / "oriented.jpg"
    source = Image.new("RGB", (2, 1))
    source.putdata([(255, 0, 0), (0, 255, 0)])
    exif = Image.Exif()
    exif[274] = 6
    source.save(path, format="JPEG", exif=exif)

    document = load_image(path)

    assert document.size == (1, 2)
    assert document.original_rgba.mode == "RGBA"


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
    """验证文件头识别到不支持的 BMP 时会拒绝。"""
    path = tmp_path / "sample.bmp"
    Image.new("RGB", (2, 2), "white").save(path, format="BMP")

    with pytest.raises(ImageLoadError, match="仅支持"):
        load_image(path)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ImageLoadError, match="找不到图片"):
        load_image(tmp_path / "missing.png")
