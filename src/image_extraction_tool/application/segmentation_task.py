"""智能抠图工作线程对象。"""

from __future__ import annotations

import logging

from PIL import Image
from PySide6.QtCore import QObject, Signal, Slot

from image_extraction_tool.domain.segmentation import CancelToken, Segmenter
from image_extraction_tool.errors import SegmentationCancelled, SegmenterError


LOGGER = logging.getLogger(__name__)


class SegmentationWorker(QObject):
    """在所属 QThread 中加载模型并执行推理，不直接修改文档或控件。"""

    progress = Signal(int, str)
    succeeded = Signal(object)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, segmenter: Segmenter, image: Image.Image, cancel_token: CancelToken) -> None:
        super().__init__()
        self._segmenter = segmenter
        self._image = image.copy()
        self._cancel_token = cancel_token

    @Slot()
    def run(self) -> None:
        try:
            self._cancel_token.raise_if_cancelled()
            self.progress.emit(0, "正在加载本地模型…")
            self._segmenter.load()
            self._cancel_token.raise_if_cancelled()
            self.progress.emit(10, "正在分析图片…")
            mask = self._segmenter.predict(
                self._image,
                progress=lambda value: self.progress.emit(
                    min(max(10 + round(value * 0.9), 10), 100),
                    "正在分析图片…",
                ),
                cancel_token=self._cancel_token,
            )
            self._cancel_token.raise_if_cancelled()
            if mask.mode != "L" or mask.size != self._image.size:
                raise ValueError("分割器返回的蒙版模式或尺寸不正确")
        except SegmentationCancelled:
            self.cancelled.emit()
        except SegmenterError as exc:
            LOGGER.warning("智能抠图失败：%s", exc)
            self.failed.emit(str(exc))
        except Exception as exc:
            LOGGER.exception("智能抠图发生未预期错误")
            self.failed.emit(f"智能抠图失败：{exc}")
        else:
            self.progress.emit(100, "智能抠图完成")
            self.succeeded.emit(mask)
