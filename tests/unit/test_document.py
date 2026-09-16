"""图像文档模型的单元测试。"""

from PIL import Image

from image_extraction_tool.domain.document import ImageDocument


def test_document_source_is_defensively_copied_and_masks_match_source() -> None:
    """验证原始像素为防御性拷贝，且四张蒙版与原图尺寸一致。"""
    source = Image.new("RGBA", (3, 2), (10, 20, 30, 40))
    document = ImageDocument(source)
    # 外部修改传入对象与取出的副本，都不应影响文档内部像素
    source.putpixel((0, 0), (1, 2, 3, 4))
    exposed = document.original_rgba
    exposed.putpixel((0, 0), (5, 6, 7, 8))

    assert document.original_rgba.getpixel((0, 0)) == (10, 20, 30, 40)
    for mask in (document.ai_mask, document.color_mask, document.erase_mask, document.restore_mask):
        assert mask.mode == "L"
        assert mask.size == (3, 2)
