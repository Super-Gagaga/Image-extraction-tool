"""可替换的本地智能抠图接口与线程安全取消令牌。"""

from __future__ import annotations

from collections.abc import Callable
from threading import Event
from typing import Protocol

from PIL import Image

from image_extraction_tool.errors import SegmentationCancelled


ProgressCallback = Callable[[int], None]


class CancelToken:
    """可跨线程设置并查询的协作式取消令牌。"""

    def __init__(self) -> None:
        self._event = Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise SegmentationCancelled("智能抠图已取消")


class Segmenter(Protocol):
    """智能抠图实现必须遵循的、与具体模型无关的接口。"""

    @property
    def name(self) -> str: ...

    def load(self) -> None: ...

    def predict(
        self,
        image: Image.Image,
        progress: ProgressCallback | None = None,
        cancel_token: CancelToken | None = None,
    ) -> Image.Image: ...
