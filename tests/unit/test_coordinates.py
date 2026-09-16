"""画布坐标到图片坐标换算的单元测试。"""

import pytest

from image_extraction_tool.domain.coordinates import canvas_to_image


def test_canvas_to_image_applies_offset_scale_and_preserves_fractions() -> None:
    """验证换算同时应用偏移与缩放，并保留小数精度。"""
    assert canvas_to_image(
        35.0,
        45.0,
        scale=2.0,
        offset_x=5.0,
        offset_y=9.0,
        image_size=(100, 80),
    ) == pytest.approx((15.0, 18.0))


def test_canvas_to_image_clips_to_image_bounds() -> None:
    """验证越界坐标被夹取到图片范围内。"""
    assert canvas_to_image(
        -100.0,
        900.0,
        scale=0.5,
        offset_x=10.0,
        offset_y=20.0,
        image_size=(12, 8),
    ) == (0.0, 7.0)


def test_canvas_to_image_rejects_invalid_scale() -> None:
    """验证非正缩放系数会抛出 ValueError。"""
    with pytest.raises(ValueError, match="scale"):
        canvas_to_image(0, 0, scale=0, offset_x=0, offset_y=0, image_size=(1, 1))
