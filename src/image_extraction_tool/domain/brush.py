"""以原图像素为单位的画笔内核。"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, hypot

import numpy as np
from PIL import Image


# 坐标一律使用原图像素单位；BoundingBox 为左闭右开的 (left, top, right, bottom)
Point = tuple[float, float]
BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class BrushSettings:
    """画笔参数；尺寸始终表示原图像素，不随视图缩放变化。"""

    size_px: int = 20
    hardness: float = 1.0
    opacity: float = 1.0

    def __post_init__(self) -> None:
        # 在构造期就拒绝越界参数，避免把非法值带进逐点绘制流程
        if not 1 <= self.size_px <= 200:
            raise ValueError("size_px 必须在 1 到 200 之间")
        if not 0.0 <= self.hardness <= 1.0:
            raise ValueError("hardness 必须在 0 到 1 之间")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity 必须在 0 到 1 之间")


def interpolated_points(start: Point, end: Point, size_px: int) -> list[Point]:
    """按不超过画笔直径四分之一的步长补齐相邻采样点。

    鼠标事件是离散的，快速拖动时相邻两次事件可能相距很远，直接按点绘制会留下
    断续圆斑；这里在两点之间线性插值，使笔迹保持连续。
    """
    distance = hypot(end[0] - start[0], end[1] - start[1])
    # 步长上限取直径的四分之一：既保证连续性，又不会产生过多采样点。
    # 下限 0.25px 用于避免原地点击时步长为 0 而除零。
    max_step = max(size_px / 4.0, 0.25)
    steps = max(1, ceil(distance / max_step))
    return [
        (
            start[0] + (end[0] - start[0]) * index / steps,
            start[1] + (end[1] - start[1]) * index / steps,
        )
        for index in range(steps + 1)
    ]


def brush_segment_bounds(
    image_size: tuple[int, int],
    start: Point,
    end: Point,
    size_px: int,
) -> BoundingBox | None:
    """返回画段可能修改的、裁剪到原图范围内的矩形。"""
    width, height = image_size
    radius = 0.5 if size_px == 1 else size_px / 2.0
    left = max(0, floor(min(start[0], end[0]) - radius))
    top = max(0, floor(min(start[1], end[1]) - radius))
    right = min(width, ceil(max(start[0], end[0]) + radius))
    bottom = min(height, ceil(max(start[1], end[1]) + radius))
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def paint_mask_segment(
    mask: Image.Image,
    start: Point,
    end: Point,
    settings: BrushSettings,
) -> BoundingBox | None:
    """把一段连续笔迹写入 ``L`` 蒙版并返回实际影响的矩形。

    画笔强度使用最大值合并，避免同一笔迹因采样密度不同而产生深浅条纹。
    1px 画笔走专用路径，保证每个采样位置只写入一个原图像素。
    返回 None 表示没有任何像素被修改（笔迹完全落在蒙版之外或透明度为 0）。
    """
    if mask.mode != "L":
        raise ValueError("画笔蒙版必须使用 L 模式")
    if settings.opacity <= 0:
        return None

    points = interpolated_points(start, end, settings.size_px)
    width, height = mask.size
    if settings.size_px == 1:
        # 1px 画笔若走通用圆形路径，浮点到整数取整会让相邻像素被同时覆盖，
        # 因此这里直接 floor 到单个像素，保证“每个采样位置只影响一个像素”。
        pixels = mask.load()
        value = round(255 * settings.opacity)
        touched: list[tuple[int, int]] = []
        for point_x, point_y in points:
            pixel_x = floor(point_x)
            pixel_y = floor(point_y)
            # 只处理落在蒙版内的采样点，越界部分直接跳过
            if 0 <= pixel_x < width and 0 <= pixel_y < height:
                pixels[pixel_x, pixel_y] = max(pixels[pixel_x, pixel_y], value)
                touched.append((pixel_x, pixel_y))
        if not touched:
            return None
        return (
            min(point[0] for point in touched),
            min(point[1] for point in touched),
            max(point[0] for point in touched) + 1,
            max(point[1] for point in touched) + 1,
        )

    radius = settings.size_px / 2.0
    # 先按所有采样点外扩一个半径算出包围盒，并裁剪到蒙版范围内；
    # 后续只在这块小区域上做数组运算，避免整图遍历的开销。
    left = max(0, floor(min(point[0] for point in points) - radius))
    top = max(0, floor(min(point[1] for point in points) - radius))
    right = min(width, ceil(max(point[0] for point in points) + radius))
    bottom = min(height, ceil(max(point[1] for point in points) + radius))
    if left >= right or top >= bottom:
        return None

    # 取出裁剪区域，并用 +0.5 偏移把网格坐标对齐到像素中心
    region = np.array(mask.crop((left, top, right, bottom)), dtype=np.uint8)
    pixel_x = np.arange(left, right, dtype=np.float32) + 0.5
    pixel_y = np.arange(top, bottom, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(pixel_x, pixel_y)
    # 硬度定义“完全实心”的范围：硬度为 1 时整个半径内都是实心
    hard_radius = radius * settings.hardness

    for center_x, center_y in points:
        distance = np.hypot(grid_x - center_x, grid_y - center_y)
        if settings.hardness >= 1.0:
            # 全硬笔刷无需羽化，直接用圆形布尔掩码
            coverage = (distance <= radius).astype(np.float32)
        else:
            # 从实心半径到最大半径之间线性衰减；羽化宽度为 0 时用极小值兜底防除零
            feather_width = max(radius - hard_radius, np.finfo(np.float32).eps)
            coverage = np.clip((radius - distance) / feather_width, 0.0, 1.0)
        strength = np.rint(coverage * settings.opacity * 255.0).astype(np.uint8)
        # 按最大值合并：同一像素被反复覆盖时取最深的一次，避免叠加出更深的斑块
        np.maximum(region, strength, out=region)

    # 只把裁剪区域贴回原蒙版，其余像素保持原样
    mask.paste(Image.fromarray(region, mode="L"), (left, top))
    return left, top, right, bottom
