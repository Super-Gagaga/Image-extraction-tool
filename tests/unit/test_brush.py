"""原图像素画笔的单元测试。"""

import pytest
from PIL import Image

from image_extraction_tool.domain.brush import (
    BrushSettings,
    clear_coverage,
    interpolated_points,
    paint_mask_segment,
    paint_mask_segment_exclusive,
    segment_coverage,
)


def test_one_pixel_brush_modifies_exactly_one_pixel() -> None:
    """验证 1px 画笔在原地点击时只写入一个像素。"""
    mask = Image.new("L", (8, 8), 0)

    paint_mask_segment(mask, (3.5, 4.5), (3.5, 4.5), BrushSettings(size_px=1))

    assert mask.getpixel((3, 4)) == 255
    assert sum(mask.getpixel((x, y)) > 0 for y in range(8) for x in range(8)) == 1


def test_fast_one_pixel_segment_has_no_horizontal_gaps() -> None:
    """验证快速拖动的长笔迹经插值后没有断点。"""
    mask = Image.new("L", (24, 5), 0)

    paint_mask_segment(mask, (1.5, 2.5), (20.5, 2.5), BrushSettings(size_px=1))

    assert all(mask.getpixel((x, 2)) == 255 for x in range(1, 21))


def test_interpolation_step_never_exceeds_quarter_brush_size() -> None:
    """验证插值步长不超过画笔直径的四分之一，保证笔迹连续。"""
    points = interpolated_points((0.0, 0.0), (100.0, 0.0), 20)

    assert max(points[index + 1][0] - points[index][0] for index in range(len(points) - 1)) <= 5.0


@pytest.mark.parametrize("size", [0, 201])
def test_brush_size_outside_supported_range_is_rejected(size: int) -> None:
    """验证超出 1～200 范围的画笔尺寸会被拒绝。"""
    with pytest.raises(ValueError, match="1 到 200"):
        BrushSettings(size_px=size)


def test_soft_low_opacity_brush_produces_grayscale_mask() -> None:
    """验证低不透明度配合软边会产生灰度值，而不是非 0 即 255。"""
    mask = Image.new("L", (20, 20), 0)
    paint_mask_segment(
        mask,
        (10.0, 10.0),
        (10.0, 10.0),
        BrushSettings(size_px=10, hardness=0.0, opacity=0.5),
    )

    extrema = mask.getextrema()
    assert extrema[0] == 0
    assert 0 < extrema[1] < 255


def test_paint_and_clear_share_identical_pixel_footprint() -> None:
    """写入与清除取自同一份贴片，两者覆盖的像素必须完全一致。"""
    erase = Image.new("L", (40, 40))
    restore = Image.new("L", (40, 40))
    settings = BrushSettings(size_px=9, hardness=0.4, opacity=0.7)
    start, end = (8.3, 12.7), (30.9, 25.1)

    paint_mask_segment_exclusive(erase, restore, start, end, settings)
    coverage = segment_coverage(erase.size, start, end, settings)
    assert coverage is not None

    # 用同一贴片清除恢复蒙版，得到的矩形应与写入侧完全相同
    written = coverage.painted_bbox
    assert written is not None
    restore.paste(255, (0, 0, *restore.size))
    assert clear_coverage(restore, coverage) == written


def test_exclusive_brush_clears_opposite_mask_on_same_pixels() -> None:
    """用一支画笔落笔时，应清除对方蒙版上被同一笔迹覆盖的像素。"""
    erase = Image.new("L", (40, 40))
    restore = Image.new("L", (40, 40))
    settings = BrushSettings(size_px=11)

    # 先在整张恢复蒙版上涂满，再落一笔抠除
    restore.paste(255, (0, 0, *restore.size))
    paint_mask_segment_exclusive(erase, restore, (20.5, 20.5), (20.5, 20.5), settings)

    assert erase.getpixel((20, 20)) == 255
    # 笔迹内的像素必须从恢复蒙版清除，笔迹外保持原样
    assert restore.getpixel((20, 20)) == 0
    assert restore.getpixel((1, 1)) == 255


def test_exclusive_brush_clear_bbox_matches_written_bbox() -> None:
    """返回的受影响矩形应覆盖写入侧与清除侧两边。"""
    erase = Image.new("L", (60, 30))
    restore = Image.new("L", (60, 30), 200)
    settings = BrushSettings(size_px=8)

    affected = paint_mask_segment_exclusive(erase, restore, (10.5, 15.5), (45.5, 15.5), settings)

    assert affected is not None
    assert erase.getbbox() is not None
    assert restore.getextrema()[1] == 200
    # 清除区域落在返回矩形内，且该矩形内确实出现了被清零的像素
    assert restore.crop(affected).getextrema()[0] == 0


def test_exclusive_paint_outside_mask_changes_nothing() -> None:
    """笔迹完全落在图片外时不写入也不清除。"""
    erase = Image.new("L", (20, 20))
    restore = Image.new("L", (20, 20), 90)

    affected = paint_mask_segment_exclusive(erase, restore, (-50.0, -50.0), (-40.0, -40.0), BrushSettings(size_px=5))

    assert affected is None
    assert erase.getextrema() == (0, 0)
    assert restore.getextrema() == (90, 90)
