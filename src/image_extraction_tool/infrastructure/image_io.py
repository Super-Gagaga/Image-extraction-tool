"""使用 Pillow 按文件头解码图片并统一为原始 RGBA 像素。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from image_extraction_tool.domain.document import ImageDocument
from image_extraction_tool.errors import ImageLoadError


# 扩展名仅用于文件对话框和拖放预筛选；真正格式由 Pillow 的文件头识别。
SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
SUPPORTED_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})


def load_image(path: str | Path) -> ImageDocument:
    """按文件头解码图片，校正 EXIF 方向并统一转换为 RGBA。

    不依赖扩展名，不缩放或重采样。返回对象可通过 ``size``、
    ``original_rgba.getpixel((x, y))`` 和
    ``original_rgba.tobytes("raw", "RGBA")`` 读取尺寸、像素和按行排列的
    RGBA 字节数据。
    """
    source_path = Path(path)

    try:
        with Image.open(source_path) as opened:
            # 先强制读完数据，避免延迟解码在退出 with 后才暴露错误
            opened.load()
            if opened.format not in SUPPORTED_FORMATS:
                raise ImageLoadError("仅支持 PNG、JPG、JPEG 和 WEBP 图片。")
            # exif_transpose 处理手机拍摄图片的旋转信息，再统一成 RGBA
            normalized = ImageOps.exif_transpose(opened).convert("RGBA")
    except FileNotFoundError as exc:
        raise ImageLoadError(f"找不到图片：{source_path}") from exc
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"无法读取图片，文件可能已损坏：{source_path.name}") from exc

    return ImageDocument(normalized, source_path=source_path)
