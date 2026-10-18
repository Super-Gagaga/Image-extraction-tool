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
    返回 None 表示没有任何像素被修改（笔迹完全落在蒙版之外或透明度为 0）。
    """
    coverage = segment_coverage(mask.size, start, end, settings)
    if coverage is None:
        return None
    return paint_coverage(mask, coverage)


@dataclass(frozen=True, slots=True, eq=False)
class SegmentCoverage:
    """一段笔迹栅格化后的强度贴片，已裁剪到蒙版范围内。

    ``eq=False`` 是刻意的：``strength`` 是 numpy 数组，自动生成的 ``__eq__``
    和 ``__hash__`` 会在逐元素比较时抛错，而这里只需要按身份传递贴片。
    """

    left: int
    top: int
    strength: np.ndarray

    @property
    def bbox(self) -> BoundingBox:
        """贴片覆盖的矩形。"""
        height, width = self.strength.shape
        return self.left, self.top, self.left + width, self.top + height

    @property
    def painted_bbox(self) -> BoundingBox | None:
        """贴片中真正带强度（非零）的子矩形；全为 0 时返回 None。"""
        rows = np.flatnonzero(self.strength.any(axis=1))
        columns = np.flatnonzero(self.strength.any(axis=0))
        if rows.size == 0 or columns.size == 0:
            return None
        return (
            self.left + int(columns[0]),
            self.top + int(rows[0]),
            self.left + int(columns[-1]) + 1,
            self.top + int(rows[-1]) + 1,
        )


def segment_coverage(
    image_size: tuple[int, int],
    start: Point,
    end: Point,
    settings: BrushSettings,
) -> SegmentCoverage | None:
    """把一段笔迹栅格化为强度贴片，不接触任何蒙版。

    写入一张蒙版和从另一张蒙版清除同一批像素都基于这份贴片，因此互斥的
    两张画笔蒙版在像素级完全对齐，不会出现一侧多涂一个像素的错位。
    """
    if settings.opacity <= 0:
        return None

    points = interpolated_points(start, end, settings.size_px)
    width, height = image_size
    value = round(255 * settings.opacity)

    if settings.size_px == 1:
        # 1px 画笔若走通用圆形路径，浮点到整数取整会让相邻像素被同时覆盖，
        # 因此这里直接 floor 到单个像素，保证“每个采样位置只影响一个像素”。
        pixel_points = {
            (floor(point_x), floor(point_y))
            for point_x, point_y in points
            if 0 <= floor(point_x) < width and 0 <= floor(point_y) < height
        }
        if not pixel_points:
            return None
        left = min(point[0] for point in pixel_points)
        top = min(point[1] for point in pixel_points)
        right = max(point[0] for point in pixel_points) + 1
        bottom = max(point[1] for point in pixel_points) + 1
        strength = np.zeros((bottom - top, right - left), dtype=np.uint8)
        for point_x, point_y in pixel_points:
            strength[point_y - top, point_x - left] = value
        return SegmentCoverage(left, top, strength)

    radius = settings.size_px / 2.0
    # 先按所有采样点外扩一个半径算出包围盒，并裁剪到蒙版范围内；
    # 后续只在这块小区域上做数组运算，避免整图遍历的开销。
    left = max(0, floor(min(point[0] for point in points) - radius))
    top = max(0, floor(min(point[1] for point in points) - radius))
    right = min(width, ceil(max(point[0] for point in points) + radius))
    bottom = min(height, ceil(max(point[1] for point in points) + radius))
    if left >= right or top >= bottom:
        return None

    # 用 +0.5 偏移把网格坐标对齐到像素中心
    pixel_x = np.arange(left, right, dtype=np.float32) + 0.5
    pixel_y = np.arange(top, bottom, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(pixel_x, pixel_y)
    # 硬度定义“完全实心”的范围：硬度为 1 时整个半径内都是实心
    hard_radius = radius * settings.hardness
    # 从实心半径到最大半径之间线性衰减；羽化宽度为 0 时用极小值兜底防除零
    feather_width = max(radius - hard_radius, np.finfo(np.float32).eps)

    strength = np.zeros((bottom - top, right - left), dtype=np.uint8)
    for center_x, center_y in points:
        distance = np.hypot(grid_x - center_x, grid_y - center_y)
        if settings.hardness >= 1.0:
            # 全硬笔刷无需羽化，直接用圆形布尔掩码
            coverage = (distance <= radius).astype(np.float32)
        else:
            coverage = np.clip((radius - distance) / feather_width, 0.0, 1.0)
        # 按最大值合并：同一像素被反复覆盖时取最深的一次，避免叠加出更深的斑块
        np.maximum(
            strength,
            np.rint(coverage * settings.opacity * 255.0).astype(np.uint8),
            out=strength,
        )
    return SegmentCoverage(left, top, strength)


def paint_coverage(mask: Image.Image, coverage: SegmentCoverage) -> BoundingBox | None:
    """把强度贴片按最大值合并写入 ``L`` 蒙版，返回实际写入的矩形。"""
    if mask.mode != "L":
        raise ValueError("画笔蒙版必须使用 L 模式")
    if coverage.bbox[0] < 0 or coverage.bbox[1] < 0:
        raise ValueError("强度贴片必须位于蒙版坐标系内")
    if coverage.painted_bbox is None:
        return None
    region = np.array(mask.crop(coverage.bbox), dtype=np.uint8)
    if region.shape != coverage.strength.shape:
        raise ValueError("强度贴片尺寸超出蒙版范围")
    np.maximum(region, coverage.strength, out=region)
    mask.paste(Image.fromarray(region, mode="L"), (coverage.left, coverage.top))
    return coverage.painted_bbox


def clear_coverage(mask: Image.Image, coverage: SegmentCoverage) -> BoundingBox | None:
    """清除 ``L`` 蒙版上被同一段笔迹覆盖的像素，返回被清除的矩形。

    返回 None 表示贴片没有触及任何像素，或这些像素本来就是 0，蒙版未被改动。
    """
    if mask.mode != "L":
        raise ValueError("画笔蒙版必须使用 L 模式")
    if coverage.painted_bbox is None:
        return None
    region = np.array(mask.crop(coverage.bbox), dtype=np.uint8)
    if region.shape != coverage.strength.shape:
        raise ValueError("强度贴片尺寸超出蒙版范围")
    touched = coverage.strength > 0
    if not np.any(region[touched]):
        return None
    region[touched] = 0
    mask.paste(Image.fromarray(region, mode="L"), (coverage.left, coverage.top))
    return coverage.painted_bbox


def paint_mask_segment_exclusive(
    mask: Image.Image,
    opposite_mask: Image.Image,
    start: Point,
    end: Point,
    settings: BrushSettings,
) -> BoundingBox | None:
    """把笔迹写入 ``mask``，同时清除 ``opposite_mask`` 上的同一批像素。

    抠除与恢复必须互斥：合成时恢复层压过抠除层，若不清除对方蒙版，早先一次
    恢复就会永久盖住之后落下的抠除笔迹，用户无法再抠除同一区域。清除范围与
    写入范围取自同一份强度贴片，因此两侧不会错位。返回实际写入的矩形。
    """
    if mask.size != opposite_mask.size:
        raise ValueError("两张画笔蒙版必须尺寸一致")
    coverage = segment_coverage(mask.size, start, end, settings)
    if coverage is None:
        return None
    written = paint_coverage(mask, coverage)
    if written is not None:
        clear_coverage(opposite_mask, coverage)
    return written
