"""透明 PNG 的全尺寸、安全写入实现。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from image_extraction_tool.domain.compositor import compose_result
from image_extraction_tool.domain.document import ImageDocument
from image_extraction_tool.errors import ExportError


def export_png(document: ImageDocument, destination: str | Path) -> Path:
    """从原图和全尺寸蒙版重新合成，并以原子替换方式写出 RGBA PNG。"""
    path = Path(destination)
    # 后缀不是 .png 时自动补全，避免用户漏写扩展名而保存出无后缀文件
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 先在同一目录写临时文件再替换：同目录保证 os.replace 为原子操作，
        # 写入中途失败也不会留下半截的目标文件。
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}-",
            suffix=".png",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        compose_result(document).save(temporary_path, format="PNG")
        os.replace(temporary_path, path)
    except (OSError, ValueError) as exc:
        # 失败时清理临时文件，避免在目标目录残留隐藏文件
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ExportError(f"无法导出 PNG：{path}") from exc
    return path
