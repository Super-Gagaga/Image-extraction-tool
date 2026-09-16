"""本地 ONNX 模型路径的解析与校验。"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from image_extraction_tool.errors import ModelMissingError


MODEL_ENVIRONMENT_VARIABLE = "IMAGE_EXTRACTION_MODEL"
DEFAULT_MODEL_FILENAME = "u2net.onnx"


def default_model_path() -> Path:
    """返回开发目录或打包程序旁的默认模型位置。"""
    configured = os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
    if configured:
        return Path(configured)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "models" / DEFAULT_MODEL_FILENAME
    repository_root = Path(__file__).resolve().parents[3]
    repository_models = repository_root / "models"
    if repository_models.is_dir():
        return repository_models / DEFAULT_MODEL_FILENAME
    return Path.cwd() / "models" / DEFAULT_MODEL_FILENAME


class ModelStore:
    """保存当前选中的模型路径；模型权重始终位于仓库之外或 models 目录。"""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model_path = Path(model_path) if model_path is not None else default_model_path()

    @property
    def model_path(self) -> Path:
        return self._model_path

    def select(self, model_path: str | Path) -> None:
        candidate = Path(model_path)
        if candidate.suffix.lower() != ".onnx":
            raise ModelMissingError("请选择扩展名为 .onnx 的本地模型文件")
        self._model_path = candidate

    def require_model(self) -> Path:
        path = self._model_path.expanduser()
        if path.suffix.lower() != ".onnx":
            raise ModelMissingError("智能抠图模型必须是 .onnx 文件")
        if not path.is_file():
            raise ModelMissingError(
                f"未找到本地模型：{path}\n请将 u2net.onnx 放入 models 目录，或点击“选择模型”。"
            )
        return path.resolve()
