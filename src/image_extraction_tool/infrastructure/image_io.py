"""带格式校验与异常处理的图片解码。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from image_extraction_tool.domain.document import ImageDocument
from image_extraction_tool.errors import ImageLoadError


# 允许导入的图片扩展名（统一小写比较）
SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def load_image(path: str | Path) -> ImageDocument:
    """解码受支持的图片，按 EXIF 校正方向并统一转换为 RGBA。"""
    source_path = Path(path)
    if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError("仅支持 PNG、JPG、JPEG 和 WEBP 图片。")

    try:
        with Image.open(source_path) as opened:
            # 先强制读完数据，避免延迟解码在退出 with 后才暴露错误
            opened.load()
            # exif_transpose 处理手机拍摄图片的旋转信息，再统一成 RGBA
            normalized = ImageOps.exif_transpose(opened).convert("RGBA")
    except FileNotFoundError as exc:
        raise ImageLoadError(f"找不到图片：{source_path}") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"无法读取图片，文件可能已损坏：{source_path.name}") from exc

    return ImageDocument(normalized, source_path=source_path)
