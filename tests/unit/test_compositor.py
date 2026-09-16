"""独立蒙版合成优先级测试。"""

from PIL import Image

from image_extraction_tool.domain.compositor import compose_alpha, compose_result
from image_extraction_tool.domain.document import ImageDocument


def test_erase_removes_and_restore_has_final_priority() -> None:
    """验证抠除会移除像素，而恢复层优先级最高。"""
    document = ImageDocument(Image.new("RGBA", (3, 1), (10, 20, 30, 255)))
    document.ai_mask.putpixel((1, 0), 0)
    document.color_mask.putpixel((2, 0), 0)
    document.erase_mask.putpixel((0, 0), 255)
    document.restore_mask.putpixel((0, 0), 255)
    document.restore_mask.putpixel((1, 0), 255)

    alpha = compose_alpha(document)
    # 第 0 位被抠除但又被恢复（恢复优先），第 1 位由 AI 抠除后恢复，第 2 位被颜色抠除
    assert [alpha.getpixel((x, 0)) for x in range(3)] == [255, 255, 0]


def test_composition_preserves_original_rgb_and_caps_original_alpha() -> None:
    """验证合成只改 Alpha，且恢复不会超过原图自身的不透明度。"""
    document = ImageDocument(Image.new("RGBA", (1, 1), (11, 22, 33, 80)))
    document.restore_mask.putpixel((0, 0), 255)

    assert compose_result(document).getpixel((0, 0)) == (11, 22, 33, 80)


def test_clear_brush_masks_does_not_change_ai_or_color_masks() -> None:
    """验证清空画笔只影响抠除与恢复蒙版，算法蒙版保持不变。"""
    document = ImageDocument(Image.new("RGBA", (2, 2), "white"))
    document.ai_mask.putpixel((0, 0), 12)
    document.color_mask.putpixel((0, 0), 34)
    document.erase_mask.putpixel((0, 0), 255)
    document.restore_mask.putpixel((1, 1), 255)

    document.clear_brush_masks()

    assert document.ai_mask.getpixel((0, 0)) == 12
    assert document.color_mask.getpixel((0, 0)) == 34
    assert document.erase_mask.getextrema() == (0, 0)
    assert document.restore_mask.getextrema() == (0, 0)
