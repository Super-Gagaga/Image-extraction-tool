"""原图像素画笔的单元测试。"""

import pytest
from PIL import Image

from image_extraction_tool.domain.brush import BrushSettings, interpolated_points, paint_mask_segment


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
