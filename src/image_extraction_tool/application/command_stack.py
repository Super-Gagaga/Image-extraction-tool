"""受数量和内存约束的编辑命令栈。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
import zlib

from PIL import Image

from image_extraction_tool.domain.color_selection import SelectedColor, build_color_mask
from image_extraction_tool.domain.document import ImageDocument


BoundingBox = tuple[int, int, int, int]


class HistoryCommand(Protocol):
    """可逆编辑命令的最小接口。"""

    description: str

    @property
    def memory_cost(self) -> int: ...

    def undo(self) -> None: ...

    def redo(self) -> None: ...


class CommandStack:
    """保存已执行命令，限制历史步数和压缩数据总量。"""

    def __init__(self, *, max_commands: int = 100, max_bytes: int = 256 * 1024 * 1024) -> None:
        if max_commands < 50:
            raise ValueError("max_commands 不能小于 50")
        if max_bytes < 1:
            raise ValueError("max_bytes 必须为正数")
        self.max_commands = max_commands
        self.max_bytes = max_bytes
        self._undo: list[HistoryCommand] = []
        self._redo: list[HistoryCommand] = []
        self._memory_bytes = 0

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_count(self) -> int:
        return len(self._undo)

    @property
    def redo_count(self) -> int:
        return len(self._redo)

    @property
    def memory_bytes(self) -> int:
        return self._memory_bytes

    @property
    def undo_description(self) -> str | None:
        return self._undo[-1].description if self._undo else None

    @property
    def redo_description(self) -> str | None:
        return self._redo[-1].description if self._redo else None

    def push_applied(self, command: HistoryCommand) -> None:
        """记录一个已经成功应用的原子命令，并清空重做分支。"""
        self._memory_bytes -= sum(item.memory_cost for item in self._redo)
        self._redo.clear()
        self._undo.append(command)
        self._memory_bytes += command.memory_cost
        self._trim()

    def undo(self) -> bool:
        if not self._undo:
            return False
        # 先执行、成功后再移动栈指针；命令抛错时历史仍保持原状，
        # 不会出现“界面没恢复但撤销步骤已经丢失”的半完成状态。
        command = self._undo[-1]
        command.undo()
        self._undo.pop()
        self._redo.append(command)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        command = self._redo[-1]
        command.redo()
        self._redo.pop()
        self._undo.append(command)
        return True

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._memory_bytes = 0

    def _trim(self) -> None:
        # 始终保留最新命令，即使单个命令本身超过软内存上限。
        while len(self._undo) > 1 and (
            len(self._undo) + len(self._redo) > self.max_commands
            or self._memory_bytes > self.max_bytes
        ):
            removed = self._undo.pop(0)
            self._memory_bytes -= removed.memory_cost


@dataclass(slots=True)
class MaskPatchCommand:
    """只保存一张蒙版受影响矩形的压缩前后数据。"""

    document: ImageDocument
    mask_name: str
    bbox: BoundingBox
    _before: bytes
    _after: bytes
    description: str = "画笔操作"

    @classmethod
    def from_images(
        cls,
        document: ImageDocument,
        mask_name: str,
        bbox: BoundingBox,
        before: Image.Image,
        after: Image.Image,
        *,
        description: str = "画笔操作",
    ) -> "MaskPatchCommand":
        size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        if before.mode != "L" or after.mode != "L" or before.size != size or after.size != size:
            raise ValueError("蒙版差异图必须是与 bbox 一致的 L 图")
        return cls(
            document=document,
            mask_name=mask_name,
            bbox=bbox,
            _before=zlib.compress(before.tobytes()),
            _after=zlib.compress(after.tobytes()),
            description=description,
        )

    @property
    def memory_cost(self) -> int:
        return len(self._before) + len(self._after) + 96

    @property
    def patch_size(self) -> tuple[int, int]:
        return self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]

    def undo(self) -> None:
        self._apply(self._before)

    def redo(self) -> None:
        self._apply(self._after)

    def _apply(self, compressed: bytes) -> None:
        target = getattr(self.document, self.mask_name)
        patch = Image.frombytes("L", self.patch_size, zlib.decompress(compressed))
        target.paste(patch, self.bbox[:2])
        self.document.mark_edited()


@dataclass(slots=True)
class _MaskSnapshot:
    """单张蒙版在一次笔画期间的初始状态，矩形随笔画范围扩大。"""

    mask_name: str
    bbox: BoundingBox | None = None
    image: Image.Image | None = None

    def include(self, target: Image.Image, bbox: BoundingBox) -> None:
        """记下 ``bbox`` 内的初始像素；矩形扩大时保留已记录的部分。"""
        if self.bbox is None:
            self.bbox = bbox
            self.image = target.crop(bbox)
            return
        union = (
            min(self.bbox[0], bbox[0]),
            min(self.bbox[1], bbox[1]),
            max(self.bbox[2], bbox[2]),
            max(self.bbox[3], bbox[3]),
        )
        if union == self.bbox:
            return
        expanded = target.crop(union)
        assert self.image is not None
        expanded.paste(self.image, (self.bbox[0] - union[0], self.bbox[1] - union[1]))
        self.bbox = union
        self.image = expanded

    def restore(self, target: Image.Image) -> bool:
        """把记录的初始像素写回蒙版；尚未记录过任何区域时返回 False。"""
        if self.bbox is None or self.image is None:
            return False
        target.paste(self.image, self.bbox[:2])
        return True

    def command(self, document: ImageDocument, description: str) -> MaskPatchCommand | None:
        """生成这层蒙版的前后差异命令；像素没有变化时返回 None。"""
        if self.bbox is None or self.image is None:
            return None
        after = getattr(document, self.mask_name).crop(self.bbox)
        if after.tobytes() == self.image.tobytes():
            return None
        return MaskPatchCommand.from_images(
            document,
            self.mask_name,
            self.bbox,
            self.image,
            after,
            description=description,
        )


class BrushStrokeRecorder:
    """在一整条笔画期间累计初始矩形，释放时生成单个命令。

    一次画笔只往一张蒙版写入强度，但会清除对方蒙版上的同一批像素，两张蒙版
    都可能变化，因此这里为每张受影响的蒙版各记录一份前状态，并在结束时可把
    它们组合成同一个历史步骤。
    """

    def __init__(
        self,
        document: ImageDocument,
        mask_name: str,
        description: str,
        *,
        opposite_mask_name: str | None = None,
    ) -> None:
        self.document = document
        self.description = description
        # 第一项是落笔写入的蒙版，其余是需要同步清除的对方蒙版
        mask_names = (mask_name, opposite_mask_name) if opposite_mask_name else (mask_name,)
        self._snapshots = [_MaskSnapshot(name) for name in mask_names]

    @property
    def mask_name(self) -> str:
        """落笔写入的蒙版名。"""
        return self._snapshots[0].mask_name

    @property
    def bbox(self) -> BoundingBox | None:
        """已记录的累积矩形，尚未落笔时为 None。"""
        return self._snapshots[0].bbox

    def include_before(self, bbox: BoundingBox) -> None:
        """在当前画段写入前扩展各张蒙版的初始状态快照。"""
        for snapshot in self._snapshots:
            snapshot.include(getattr(self.document, snapshot.mask_name), bbox)

    def finish(self) -> HistoryCommand | None:
        commands = tuple(
            command
            for snapshot in self._snapshots
            if (command := snapshot.command(self.document, self.description)) is not None
        )
        if not commands:
            return None
        if len(commands) == 1:
            return commands[0]
        return CompoundCommand(commands, self.description)

    def cancel(self) -> bool:
        """恢复笔画前的全部蒙版状态，返回是否有蒙版被还原。"""
        restored = False
        for snapshot in self._snapshots:
            restored |= snapshot.restore(getattr(self.document, snapshot.mask_name))
        if restored:
            self.document.mark_edited()
        return restored


@dataclass(slots=True)
class ColorStateCommand:
    """通过重建颜色蒙版恢复颜色列表和容差，不保存整图副本。"""

    document: ImageDocument
    before_colors: tuple[SelectedColor, ...]
    before_tolerance: int
    after_colors: tuple[SelectedColor, ...]
    after_tolerance: int
    description: str

    @property
    def memory_cost(self) -> int:
        return 128 + 12 * (len(self.before_colors) + len(self.after_colors))

    def undo(self) -> None:
        self._apply(self.before_colors, self.before_tolerance)

    def redo(self) -> None:
        self._apply(self.after_colors, self.after_tolerance)

    def _apply(self, colors: Sequence[SelectedColor], tolerance: int) -> None:
        self.document.selected_colors = list(colors)
        self.document.color_tolerance = tolerance
        self.document.color_mask = build_color_mask(
            self.document.original_rgba,
            self.document.selected_colors,
            tolerance,
        )
        self.document.mark_edited()


@dataclass(slots=True)
class CompoundCommand:
    """把多张蒙版的同步变化组合成一个历史步骤。"""

    commands: tuple[HistoryCommand, ...]
    description: str

    @property
    def memory_cost(self) -> int:
        return 64 + sum(command.memory_cost for command in self.commands)

    def undo(self) -> None:
        for command in reversed(self.commands):
            command.undo()

    def redo(self) -> None:
        for command in self.commands:
            command.redo()


class ApplyAiMaskCommand(MaskPatchCommand):
    """AI 任务成功后可原子应用的全尺寸蒙版命令。"""

    @classmethod
    def create(cls, document: ImageDocument, new_mask: Image.Image) -> "ApplyAiMaskCommand":
        if new_mask.mode != "L" or new_mask.size != document.size:
            raise ValueError("AI 蒙版必须是与原图等大的 L 图")
        bbox = (0, 0, *document.size)
        base = MaskPatchCommand.from_images(
            document,
            "ai_mask",
            bbox,
            document.ai_mask.copy(),
            new_mask.copy(),
            description="应用智能蒙版",
        )
        return cls(base.document, base.mask_name, base.bbox, base._before, base._after, base.description)
