"""背景图片的适配与合成规则。

背景图与抠图结果分离保存：文档只持有已适配到原图尺寸的背景快照，
预览和导出每次都通过同一套合成函数重新生成结果，保证两者一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


@dataclass(frozen=True, slots=True)
class Background:
    """已适配到文档尺寸的背景图；``source_path`` 仅用于界面提示。"""

    image: Image.Image
    source_path: Path | None = None


def fit_background(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """把背景图按“覆盖并居中裁剪”适配到文档尺寸。

    宽高比不一致时裁掉超出画布的部分，保证整幅画布都被背景填满且不拉伸。
    返回独立的 RGBA 副本，不修改调用方传入的图片。
    """
    if size[0] < 1 or size[1] < 1:
        raise ValueError("背景尺寸必须为正数")
    rgba = image if image.mode == "RGBA" else image.convert("RGBA")
    if rgba.size == size:
        return rgba.copy()
    return ImageOps.fit(
        rgba,
        size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def flatten_on_background(result: Image.Image, background: Image.Image) -> Image.Image:
    """把带 Alpha 的结果叠加到背景上，返回同尺寸 RGBA。

    背景自身若带透明度会保留在结果 Alpha 中；常见的不透明背景会得到
    Alpha=255 的成品图。
    """
    if background.size != result.size:
        raise ValueError("背景尺寸必须与结果尺寸一致")
    background_rgba = background if background.mode == "RGBA" else background.convert("RGBA")
    result_rgba = result if result.mode == "RGBA" else result.convert("RGBA")
    return Image.alpha_composite(background_rgba, result_rgba)
