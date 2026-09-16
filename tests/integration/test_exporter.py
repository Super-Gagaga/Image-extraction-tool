"""全尺寸透明 PNG 导出集成测试。"""

from pathlib import Path

from PIL import Image
import pytest

from image_extraction_tool.domain.compositor import compose_result
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
