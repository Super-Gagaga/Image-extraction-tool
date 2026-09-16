"""与 Qt 无关的坐标换算辅助函数。"""

from __future__ import annotations


def canvas_to_image(
    canvas_x: float,
    canvas_y: float,
    *,
    scale: float,
    offset_x: float,
    offset_y: float,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    """把画布坐标映射为裁剪到图片范围内的浮点图片坐标。

    画布坐标 = 图片坐标 × scale + offset，因此反向换算为先减偏移再除以缩放。
    """
    if scale <= 0:
        # 保留 "scale" 字样，便于调用方与测试按关键字识别该错误
        raise ValueError("scale 必须为正数")
    width, height = image_size
    if width < 1 or height < 1:
        raise ValueError("image_size must be positive")
    image_x = (canvas_x - offset_x) / scale
    image_y = (canvas_y - offset_y) / scale
    # 夹取到 [0, 尺寸-1]，越界时贴到最近的有效像素
    return (
        min(max(image_x, 0.0), float(width - 1)),
        min(max(image_y, 0.0), float(height - 1)),
    )
