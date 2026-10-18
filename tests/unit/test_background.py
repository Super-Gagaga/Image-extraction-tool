"""背景适配与合成测试。"""

from pathlib import Path

from PIL import Image

from image_extraction_tool.domain.background import fit_background
from image_extraction_tool.domain.compositor import compose_result
from image_extraction_tool.domain.document import ImageDocument


def test_fit_background_covers_target_size_with_rgba() -> None:
    """验证不同尺寸的背景会被覆盖填充并统一为 RGBA。"""
    fitted = fit_background(Image.new("RGB", (3, 1), (12, 34, 56)), (5, 4))

    assert fitted.size == (5, 4)
    assert fitted.mode == "RGBA"
    assert fitted.getpixel((0, 0)) == (12, 34, 56, 255)
    assert fitted.getpixel((4, 3)) == (12, 34, 56, 255)


def test_fit_background_center_crops_instead_of_stretching() -> None:
    """宽高比不一致时居中裁剪，而不是把左右两半压进画布。"""
    source = Image.new("RGB", (100, 50), (255, 0, 0))
    source.paste((0, 0, 255), (50, 0, 100, 50))

    fitted = fit_background(source, (50, 50))

    assert fitted.getpixel((0, 25))[0] > 200
    assert fitted.getpixel((49, 25))[2] > 200


def test_fit_background_returns_independent_copy_at_same_size() -> None:
    source = Image.new("RGBA", (2, 2), (1, 2, 3, 4))

    fitted = fit_background(source, (2, 2))

    assert fitted is not source
    assert fitted.tobytes() == source.tobytes()


def test_compose_result_flattens_transparency_onto_background() -> None:
    """透明像素显示背景，不透明像素保留前景。"""
    document = ImageDocument(Image.new("RGBA", (2, 1), (10, 20, 30, 255)))
    document.erase_mask.putpixel((0, 0), 255)
    document.set_background(Image.new("RGB", (2, 1), (200, 100, 50)))

    result = compose_result(document)

    assert result.mode == "RGBA"
    assert result.getpixel((0, 0)) == (200, 100, 50, 255)
    assert result.getpixel((1, 0)) == (10, 20, 30, 255)


def test_translucent_background_keeps_its_alpha() -> None:
    """半透明背景不会被强行压成不透明。"""
    document = ImageDocument(Image.new("RGBA", (1, 1), (10, 20, 30, 255)))
    document.erase_mask.putpixel((0, 0), 255)
    document.set_background(Image.new("RGBA", (1, 1), (200, 100, 50, 128)))

    result = compose_result(document)

    assert result.getpixel((0, 0)) == (200, 100, 50, 128)


def test_clear_background_restores_transparent_result() -> None:
    document = ImageDocument(Image.new("RGBA", (1, 1), (10, 20, 30, 255)))
    document.erase_mask.putpixel((0, 0), 255)
    document.set_background(Image.new("RGB", (1, 1), (200, 100, 50)))

    document.clear_background()

    assert document.background is None
    assert compose_result(document).getpixel((0, 0))[3] == 0


def test_set_background_fits_to_document_and_keeps_source_path() -> None:
    document = ImageDocument(Image.new("RGBA", (2, 3), (10, 20, 30, 255)))

    document.set_background(Image.new("RGB", (10, 10), (1, 2, 3)), Path("background.png"))

    assert document.background is not None
    assert document.background.image.size == (2, 3)
    assert document.background.source_path == Path("background.png")
    assert document.revision == 1
