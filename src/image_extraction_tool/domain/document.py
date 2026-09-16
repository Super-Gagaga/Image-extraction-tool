"""核心文档与视图状态模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from PIL import Image


class ToolType(Enum):
    """互斥的编辑工具类型。"""

    NONE = auto()
    COLOR_PICKER = auto()
    ERASE_BRUSH = auto()
    RESTORE_BRUSH = auto()


class OverlayMode(Enum):
    """画布的展示模式。"""

    RESULT = auto()
    ORIGINAL = auto()
    MASK = auto()


@dataclass(slots=True)
class ViewState:
    """缩放、平移等导航状态，与图像编辑状态相互独立。"""

    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    active_tool: ToolType = ToolType.NONE
    overlay_mode: OverlayMode = OverlayMode.RESULT


class ImageDocument:
    """一张图片及其全分辨率蒙版。

    原始图像以私有属性保存，调用方拿到的是副本，避免预览或编辑代码
    意外改写文档的原始像素。四张蒙版均为与原图等大的 8 位灰度图：
    ai_mask/color_mask 用 255 表示保留、erase_mask/restore_mask 用 255 表示要处理。
    """

    def __init__(self, original_rgba: Image.Image, source_path: Path | None = None) -> None:
        # 统一按 RGBA 处理，保证后续蒙版与导出逻辑不依赖原始色彩模式
        if original_rgba.mode != "RGBA":
            raise ValueError("original_rgba must use RGBA mode")
        if original_rgba.width < 1 or original_rgba.height < 1:
            raise ValueError("original_rgba must not be empty")

        self.source_path = source_path
        self._original_rgba = original_rgba.copy()
        self.ai_mask = Image.new("L", self._original_rgba.size, 255)
        self.color_mask = Image.new("L", self._original_rgba.size, 255)
        self.erase_mask = Image.new("L", self._original_rgba.size, 0)
        self.restore_mask = Image.new("L", self._original_rgba.size, 0)
        # 每次修改蒙版后自增，供 UI 判断是否需要重绘缓存
        self.revision = 0

    @property
    def original_rgba(self) -> Image.Image:
        """返回原始像素的防御性副本（原始像素视为不可变）。"""
        return self._original_rgba.copy()

    @property
    def size(self) -> tuple[int, int]:
        return self._original_rgba.size
