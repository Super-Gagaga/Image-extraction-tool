"""由不可变原图与独立蒙版统一生成 Alpha 和结果图。"""

from __future__ import annotations

from PIL import Image, ImageChops

from image_extraction_tool.domain.document import ImageDocument


def compose_alpha(document: ImageDocument) -> Image.Image:
    """按 AI、颜色、抠除、恢复的优先级生成全尺寸 Alpha。

    各蒙版语义统一为“255 表示保留”，因此叠加保留范围只需逐层取较暗值；
    恢复层最后用取较亮值叠加，从而取得最高优先级。
    """
    original_alpha = document.original_rgba.getchannel("A")
    # 前三层依次求交集：AI 抠图 → 颜色抠除 → 手动抠除（抠除蒙版需先取反）
    base = ImageChops.darker(original_alpha, document.ai_mask)
    base = ImageChops.darker(base, document.color_mask)
    removed = ImageChops.darker(base, ImageChops.invert(document.erase_mask))
    # 恢复优先于算法和抠除，但不创造原图中不存在的不透明度。
    restorable = ImageChops.darker(original_alpha, document.restore_mask)
    return ImageChops.lighter(removed, restorable)


def compose_result(document: ImageDocument) -> Image.Image:
    """每次都从不可变原图和当前全尺寸蒙版重新合成 RGBA 结果。"""
    result = document.original_rgba
    result.putalpha(compose_alpha(document))
    return result
