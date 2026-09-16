"""全尺寸透明 PNG 导出集成测试。"""

from pathlib import Path

from PIL import Image
import pytest

from image_extraction_tool.domain.compositor import compose_result
from image_extraction_tool.domain.color_selection import SelectedColor, build_color_mask
from image_extraction_tool.domain.document import ImageDocument
from image_extraction_tool.infrastructure.exporter import export_png
from image_extraction_tool.errors import ExportError


def test_export_png_matches_full_resolution_preview(tmp_path: Path) -> None:
    """验证导出结果为全分辨率 RGBA，且像素与预览合成完全一致。"""
    document = ImageDocument(Image.new("RGBA", (13, 9), (40, 80, 120, 200)))
    document.erase_mask.putpixel((2, 3), 255)
    document.restore_mask.putpixel((2, 3), 100)
    expected = compose_result(document)

    # 传入不带扩展名的路径，验证导出会补全为 .png
    output = export_png(document, tmp_path / "result")

    with Image.open(output) as exported:
        exported.load()
        assert exported.mode == "RGBA"
        assert exported.size == (13, 9)
        assert exported.tobytes() == expected.tobytes()


def test_export_failure_is_reported_and_leaves_no_temporary_png(tmp_path: Path) -> None:
    """验证导出失败会抛出 ExportError，且不残留临时文件。"""
    document = ImageDocument(Image.new("RGBA", (2, 2), "white"))
    # 用一个普通文件占位，使其下的目录无法创建，从而触发写入失败
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("occupied", encoding="utf-8")

    with pytest.raises(ExportError, match="无法导出"):
        export_png(document, blocking_file / "result.png")

    assert list(tmp_path.glob(".*.png")) == []


def test_export_includes_color_selection_mask(tmp_path: Path) -> None:
    """验证颜色抠除结果经过统一合成管线进入最终 PNG。"""
    source = Image.new("RGBA", (2, 1))
    source.putdata([(20, 40, 60, 255), (200, 210, 220, 255)])
    document = ImageDocument(source)
    selected = SelectedColor((20, 40, 60))
    document.add_selected_colors([selected])
    document.color_mask = build_color_mask(document.original_rgba, document.selected_colors, tolerance=0)

    output = export_png(document, tmp_path / "color-result.png")

    with Image.open(output) as exported:
        assert [exported.getpixel((x, 0))[3] for x in range(2)] == [0, 255]


def test_export_uses_binary_union_at_original_resolution(tmp_path: Path) -> None:
    """验证导出按原图像素合并颜色和画笔区域，并保留其余 RGBA。"""
    source = Image.new("RGBA", (3, 2))
    source.putdata(
        [
            (1, 2, 3, 40),
            (4, 5, 6, 80),
            (7, 8, 9, 120),
            (10, 11, 12, 160),
            (13, 14, 15, 200),
            (16, 17, 18, 240),
        ]
    )
    document = ImageDocument(source)
    document.color_mask.putpixel((0, 0), 0)
    document.erase_mask.putpixel((1, 1), 12)

    output = export_png(document, tmp_path / "binary-union.png")

    with Image.open(output) as exported:
        exported.load()
        assert exported.mode == "RGBA"
        assert exported.size == source.size
        assert exported.getpixel((0, 0)) == (1, 2, 3, 0)
        assert exported.getpixel((1, 1)) == (13, 14, 15, 0)
        assert exported.getpixel((2, 0)) == source.getpixel((2, 0))
        assert exported.getpixel((2, 1)) == source.getpixel((2, 1))


def test_export_round_trip_preserves_unedited_rgba_pixels(tmp_path: Path) -> None:
    source = Image.new("RGBA", (3, 1))
    source.putdata([(10, 20, 30, 17), (40, 50, 60, 128), (70, 80, 90, 255)])
    document = ImageDocument(source)
    document.erase_mask.putpixel((1, 0), 255)

    output = export_png(document, tmp_path / "round-trip.png")

    with Image.open(output) as exported:
        exported.load()
        assert list(exported.getdata()) == [
            (10, 20, 30, 17),
            (40, 50, 60, 0),
            (70, 80, 90, 255),
        ]
    assert document.original_rgba.tobytes() == source.tobytes()
