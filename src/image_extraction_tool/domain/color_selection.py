"""颜色选择、区域代表色和全尺寸颜色蒙版算法。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor

import numpy as np
from PIL import Image


RGBColor = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class SelectedColor:
    """一个从不可变原图中选出的 RGB 背景色。"""

    rgb: RGBColor

    def __post_init__(self) -> None:
        if len(self.rgb) != 3 or any(not 0 <= channel <= 255 for channel in self.rgb):
            raise ValueError("RGB 通道必须在 0 到 255 之间")

    @property
    def hex_value(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)


def rgb_distance_squared(first: RGBColor, second: RGBColor) -> int:
    """返回两种颜色的平方欧氏距离，避免边界判断时引入开方误差。"""
    return sum((left - right) ** 2 for left, right in zip(first, second, strict=True))


def build_color_mask(
    original_rgba: Image.Image,
    selected_colors: Sequence[SelectedColor],
    tolerance: int,
    *,
    chunk_rows: int = 512,
) -> Image.Image:
    """从原图重建颜色保留蒙版；匹配任一已选颜色的像素被设为 0。

    使用分块计算控制大图的临时数组内存。每次调用都只读取原图，不依赖旧蒙版，
    因此反复调整容差不会累积透明化损伤。
    """
    if original_rgba.mode != "RGBA":
        raise ValueError("original_rgba 必须使用 RGBA 模式")
    if not 0 <= tolerance <= 255:
        raise ValueError("tolerance 必须在 0 到 255 之间")
    if chunk_rows < 1:
        raise ValueError("chunk_rows 必须为正数")
    if not selected_colors:
        return Image.new("L", original_rgba.size, 255)

    width, height = original_rgba.size
    result = np.full((height, width), 255, dtype=np.uint8)
    threshold_squared = tolerance * tolerance
    for top in range(0, height, chunk_rows):
        bottom = min(height, top + chunk_rows)
        rgb = np.asarray(original_rgba.crop((0, top, width, bottom)), dtype=np.uint8)[..., :3]
        rgb_signed = rgb.astype(np.int16)
        matched = np.zeros((bottom - top, width), dtype=bool)
        for selected in selected_colors:
            difference = rgb_signed - np.asarray(selected.rgb, dtype=np.int16)
            squared = np.square(difference, dtype=np.int32)
            matched |= np.sum(squared, axis=2) <= threshold_squared
        result_chunk = result[top:bottom]
        result_chunk[matched] = 0
    return Image.fromarray(result, mode="L")


def representative_colors(
    original_rgba: Image.Image,
    box: tuple[float, float, float, float],
    *,
    max_colors: int = 8,
) -> list[SelectedColor]:
    """从框选区域提取按像素数量排序且去重的代表色。

    大区域先等比缩小到 256×256 再做中位切分量化，避免框选整张大图时产生
    无界内存和计算开销；完全透明像素不参与颜色统计。
    """
    if original_rgba.mode != "RGBA":
        raise ValueError("original_rgba 必须使用 RGBA 模式")
    if max_colors < 1:
        raise ValueError("max_colors 必须为正数")

    left = max(0, floor(min(box[0], box[2])))
    top = max(0, floor(min(box[1], box[3])))
    right = min(original_rgba.width, ceil(max(box[0], box[2])))
    bottom = min(original_rgba.height, ceil(max(box[1], box[3])))
    if left >= right or top >= bottom:
        return []

    region = original_rgba.crop((left, top, right, bottom))
    # 最近邻缩小不会在两块纯色的边界凭空制造混合色。
    region.thumbnail((256, 256), Image.Resampling.NEAREST)
    rgba = np.asarray(region, dtype=np.uint8)
    opaque_rgb = rgba[..., :3][rgba[..., 3] > 0]
    if opaque_rgb.size == 0:
        return []

    samples = Image.fromarray(opaque_rgb.reshape((1, -1, 3)), mode="RGB")
    quantized = samples.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
    counts = sorted(quantized.getcolors(maxcolors=max_colors) or [], reverse=True)
    palette = quantized.getpalette() or []
    colors: list[SelectedColor] = []
    seen: set[RGBColor] = set()
    for _, palette_index in counts:
        offset = palette_index * 3
        rgb = tuple(palette[offset : offset + 3])
        if len(rgb) == 3 and rgb not in seen:
            typed_rgb: RGBColor = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            seen.add(typed_rgb)
            colors.append(SelectedColor(typed_rgb))
    return colors
