"""已知 ONNX 抠图模型的推理尺寸预设。

ONNX 图里固定了 H/W 的模型会直接采用声明尺寸；只有动态 H/W 的模型才需要
外部指定推理尺寸。U²-Net 类模型通常在 320×320 训练，而 BiRefNet、RMBG-1.4
等 1024×1024 模型如果被导出成动态尺寸，默认按 320×320 推理会明显损失质量，
因此这里按已知模型文件名给出合理默认值，并允许用户通过环境变量覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DYNAMIC_INPUT_SIZE = (320, 320)
INPUT_SIZE_ENVIRONMENT_VARIABLE = "IMAGE_EXTRACTION_INPUT_SIZE"

# 文件名片段 -> 动态尺寸模型的默认推理尺寸。固定尺寸模型不会用到这些值。
_KNOWN_MODEL_INPUT_SIZES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("birefnet", (1024, 1024)),
    ("rmbg", (1024, 1024)),
    ("isnet", (1024, 1024)),
)


def parse_input_size(value: str) -> tuple[int, int]:
    """把 ``1024``、``1024x1024`` 或 ``1024×1024`` 解析为 ``(宽, 高)``。"""
    normalized = value.strip().casefold().replace("×", "x")
    if not normalized:
        raise ValueError("输入尺寸不能为空")
    parts = normalized.split("x")
    if len(parts) == 1:
        parts = [parts[0], parts[0]]
    if len(parts) != 2:
        raise ValueError(f"无法识别的输入尺寸：{value}")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"无法识别的输入尺寸：{value}") from exc
    if width < 1 or height < 1:
        raise ValueError(f"输入尺寸必须为正整数：{value}")
    return width, height


def resolve_dynamic_input_size(model_path: str | Path) -> tuple[int, int]:
    """解析动态尺寸模型应使用的推理尺寸。

    优先级：``IMAGE_EXTRACTION_INPUT_SIZE`` 环境变量 > 已知模型文件名 > 320×320。
    """
    configured = os.environ.get(INPUT_SIZE_ENVIRONMENT_VARIABLE)
    if configured:
        return parse_input_size(configured)
    filename = Path(model_path).name.casefold()
    for keyword, size in _KNOWN_MODEL_INPUT_SIZES:
        if keyword in filename:
            return size
    return DEFAULT_DYNAMIC_INPUT_SIZE
