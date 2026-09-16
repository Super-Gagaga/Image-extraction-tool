"""使用 ONNX Runtime 执行 U²-Net 类前景分割模型。"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import (
    DeviceReset,
    EPFail,
    EngineError,
    Fail,
    InvalidArgument,
    InvalidGraph,
    InvalidProtobuf,
    ModelLoadCanceled,
    ModelRequiresCompilation,
    NoModel,
    NoSuchFile,
    NotFound,
    NotImplemented as OrtNotImplemented,
    RuntimeException,
)
from PIL import Image

from image_extraction_tool.domain.segmentation import CancelToken, ProgressCallback
from image_extraction_tool.errors import InferenceError, ModelLoadError, ModelMissingError


ORT_ERRORS = (
    DeviceReset,
    EPFail,
    EngineError,
    Fail,
    InvalidArgument,
    InvalidGraph,
    InvalidProtobuf,
    ModelLoadCanceled,
    ModelRequiresCompilation,
    NoModel,
    NoSuchFile,
    NotFound,
    OrtNotImplemented,
    RuntimeException,
)


class OnnxU2NetSegmenter:
    """适配单输入、NCHW RGB 的 U²-Net 及兼容显著性分割模型。

    固定尺寸模型直接采用声明的高宽，动态尺寸模型默认使用 320×320。
    会话按模型绝对路径和执行提供程序缓存，同一进程不会重复加载。
    """

    _session_cache: OrderedDict[
        tuple[Path, int, int, tuple[str, ...]], ort.InferenceSession
    ] = OrderedDict()
    _cache_lock = Lock()
    _max_cached_sessions = 2

    def __init__(
        self,
        model_path: str | Path,
        *,
        dynamic_input_size: tuple[int, int] = (320, 320),
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        if dynamic_input_size[0] < 1 or dynamic_input_size[1] < 1:
            raise ValueError("dynamic_input_size 必须为正数")
        self.model_path = Path(model_path)
        self.dynamic_input_size = dynamic_input_size
        self.providers = tuple(providers)
        self._session: ort.InferenceSession | None = None

    @property
    def name(self) -> str:
        return "U²-Net (ONNX Runtime)"

    @classmethod
    def clear_session_cache(cls) -> None:
        """释放进程缓存的模型会话，主要用于测试和显式模型切换。"""
        with cls._cache_lock:
            cls._session_cache.clear()

    def load(self) -> None:
        path = self.model_path.expanduser()
        if not path.is_file():
            raise ModelMissingError(f"未找到本地模型：{path}")
        resolved = path.resolve()
        stat = resolved.stat()
        key = (resolved, stat.st_mtime_ns, stat.st_size, self.providers)
        with self._cache_lock:
            cached = self._session_cache.get(key)
            if cached is not None:
                self._session_cache.move_to_end(key)
                self._session = cached
                return
            try:
                session = ort.InferenceSession(str(resolved), providers=list(self.providers))
            except (OSError, ValueError, RuntimeError, *ORT_ERRORS) as exc:
                raise ModelLoadError(f"无法加载 ONNX 模型：{exc}") from exc
            self._validate_input(session)
            # 同一路径被替换后丢弃旧文件对应的缓存，再把全局缓存限制为两项。
            for existing_key in tuple(self._session_cache):
                if existing_key[0] == resolved:
                    del self._session_cache[existing_key]
            self._session_cache[key] = session
            while len(self._session_cache) > self._max_cached_sessions:
                self._session_cache.popitem(last=False)
            self._session = session

    def predict(
        self,
        image: Image.Image,
        progress: ProgressCallback | None = None,
        cancel_token: CancelToken | None = None,
    ) -> Image.Image:
        token = cancel_token or CancelToken()
        token.raise_if_cancelled()
        self._report(progress, 5)
        if self._session is None:
            self.load()
        token.raise_if_cancelled()
        self._report(progress, 20)

        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA")
        original_size = image.size
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        input_width, input_height = self._input_size()
        resized = cv2.resize(rgb, (input_width, input_height), interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)) / np.asarray(
            (0.229, 0.224, 0.225), dtype=np.float32
        )
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
        token.raise_if_cancelled()
        self._report(progress, 35)

        assert self._session is not None
        input_name = self._session.get_inputs()[0].name
        try:
            outputs = self._session.run(None, {input_name: tensor})
        except (ValueError, RuntimeError, *ORT_ERRORS) as exc:
            raise InferenceError(f"ONNX 推理失败：{exc}") from exc
        token.raise_if_cancelled()
        self._report(progress, 85)

        if not outputs:
            raise InferenceError("ONNX 模型没有返回输出")
        probability = self._extract_probability(np.asarray(outputs[0]))
        restored = cv2.resize(probability, original_size, interpolation=cv2.INTER_LINEAR)
        mask = Image.fromarray(np.rint(np.clip(restored, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
        token.raise_if_cancelled()
        self._report(progress, 100)
        return mask

    @staticmethod
    def _report(callback: ProgressCallback | None, value: int) -> None:
        if callback is not None:
            callback(value)

    @staticmethod
    def _validate_input(session: ort.InferenceSession) -> None:
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ModelLoadError("仅支持单输入 ONNX 分割模型")
        shape = inputs[0].shape
        if len(shape) != 4:
            raise ModelLoadError("模型输入必须是 NCHW 四维张量")
        channels = shape[1]
        if isinstance(channels, int) and channels != 3:
            raise ModelLoadError("模型输入必须包含 3 个 RGB 通道")

    def _input_size(self) -> tuple[int, int]:
        assert self._session is not None
        shape = self._session.get_inputs()[0].shape
        height = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else self.dynamic_input_size[1]
        width = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else self.dynamic_input_size[0]
        return width, height

    @staticmethod
    def _extract_probability(output: np.ndarray) -> np.ndarray:
        array = np.asarray(output, dtype=np.float32)
        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 2 or array.size == 0:
            raise InferenceError(f"不支持的模型输出形状：{output.shape}")
        if not np.isfinite(array).all():
            raise InferenceError("模型输出包含 NaN 或无穷值")
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum < 0.0 or maximum > 255.0:
            array = 1.0 / (1.0 + np.exp(-np.clip(array, -80.0, 80.0)))
        elif maximum > 1.0:
            array = array / 255.0
        return np.clip(array, 0.0, 1.0)
