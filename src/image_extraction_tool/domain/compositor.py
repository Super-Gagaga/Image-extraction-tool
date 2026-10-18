"""由不可变原图与独立蒙版统一生成 Alpha 和结果图。"""

from __future__ import annotations

from PIL import Image, ImageChops

from image_extraction_tool.domain.background import flatten_on_background
from image_extraction_tool.domain.document import ImageDocument


def compose_alpha(document: ImageDocument) -> Image.Image:
    """在原图分辨率上生成用于预览与导出的最终 Alpha。

    颜色匹配区域或抠除画笔触及的像素直接变为完全透明；恢复画笔触及的
    像素恢复为原图自身 Alpha。未触及像素保留原图 Alpha。AI 蒙版仍作为
    后续智能抠图阶段的基础 Alpha，当前默认全白，不改变手工流程。

    两支画笔在落笔时会互相清除对方的蒙版，因此同一像素最多只留在一张画笔
    蒙版上，恢复层不会把后来的抠除笔迹永久盖住。这里的恢复优先只是兜底：
    万一两张蒙版仍出现重叠（例如调用方直接改蒙版），恢复依然取最终结果。
    """
    original_alpha = document.original_rgba.getchannel("A")
    # AI 蒙版可以保留灰度边缘；当前尚未运行 AI 时为全白。
    base = ImageChops.darker(original_alpha, document.ai_mask)

    # color_mask 的 0 表示颜色命中；erase_mask 任意非零表示画笔触及。
    color_removed = document.color_mask.point(lambda value: 255 if value < 255 else 0)
    brush_removed = document.erase_mask.point(lambda value: 255 if value > 0 else 0)
    removed_region = ImageChops.lighter(color_removed, brush_removed)
    transparent = Image.new("L", document.size, 0)
    removed = Image.composite(transparent, base, removed_region)

    # 恢复画笔任意非零都恢复原图透明度，而不是强制变为不透明。
    restored_region = document.restore_mask.point(lambda value: 255 if value > 0 else 0)
    return Image.composite(original_alpha, removed, restored_region)


def compose_result(document: ImageDocument) -> Image.Image:
    """每次都从不可变原图、当前全尺寸蒙版和背景重新合成结果。

    没有背景时返回保留透明通道的 RGBA；设置背景后把前景叠加到背景上，
    预览与导出共用本函数，保证两者像素一致。
    """
    result = document.original_rgba
    result.putalpha(compose_alpha(document))
    background = document.background
    if background is None:
        return result
    return flatten_on_background(result, background.image)
