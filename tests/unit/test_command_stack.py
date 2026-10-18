"""阶段 4 编辑历史内核测试。"""

from dataclasses import dataclass

import pytest
from PIL import Image

from image_extraction_tool.application.command_stack import (
    ApplyAiMaskCommand,
    BrushStrokeRecorder,
    CommandStack,
    MaskPatchCommand,
)
from image_extraction_tool.domain.brush import (
    BrushSettings,
    brush_segment_bounds,
    paint_mask_segment,
    paint_mask_segment_exclusive,
)
from image_extraction_tool.domain.document import ImageDocument


@dataclass
class _CounterCommand:
    state: list[int]
    delta: int = 1
    description: str = "计数"
    memory_cost: int = 1
    fail_undo: bool = False

    def undo(self) -> None:
        if self.fail_undo:
            raise RuntimeError("undo failed")
        self.state[0] -= self.delta

    def redo(self) -> None:
        self.state[0] += self.delta


def _apply_counter(stack: CommandStack, state: list[int], **kwargs) -> None:
    command = _CounterCommand(state, **kwargs)
    command.redo()
    stack.push_applied(command)


def test_stack_supports_at_least_fifty_steps_and_new_edit_clears_redo() -> None:
    stack = CommandStack(max_commands=60, max_bytes=1024)
    state = [0]
    for _ in range(60):
        _apply_counter(stack, state)

    assert state == [60]
    assert stack.undo_count == 60
    for _ in range(50):
        assert stack.undo()
    assert state == [10]
    assert stack.redo_count == 50

    _apply_counter(stack, state, delta=5)
    assert state == [15]
    assert stack.redo_count == 0
    assert not stack.can_redo


def test_stack_limits_memory_and_keeps_history_when_undo_fails() -> None:
    stack = CommandStack(max_commands=100, max_bytes=10)
    state = [0]
    for _ in range(4):
        _apply_counter(stack, state, memory_cost=4)
    assert stack.memory_bytes <= 10
    assert stack.undo_count == 2

    failing = _CounterCommand(state, memory_cost=1, fail_undo=True)
    stack.push_applied(failing)
    with pytest.raises(RuntimeError, match="undo failed"):
        stack.undo()
    assert stack.undo_description == "计数"
    assert stack.undo_count == 3
    assert stack.redo_count == 0


def test_mask_patch_only_stores_and_restores_affected_rectangle() -> None:
    document = ImageDocument(Image.new("RGBA", (100, 80), "white"))
    bbox = (20, 10, 24, 13)
    before = document.erase_mask.crop(bbox)
    document.erase_mask.paste(255, bbox)
    after = document.erase_mask.crop(bbox)
    command = MaskPatchCommand.from_images(document, "erase_mask", bbox, before, after)

    assert command.patch_size == (4, 3)
    command.undo()
    assert document.erase_mask.getbbox() is None
    command.redo()
    assert document.erase_mask.getbbox() == bbox


def test_whole_brush_stroke_is_one_small_patch_on_large_document() -> None:
    document = ImageDocument(Image.new("RGBA", (4096, 2160), "white"))
    recorder = BrushStrokeRecorder(document, "erase_mask", "抠除笔画")
    settings = BrushSettings(size_px=9)
    segments = [((100.5, 100.5), (140.5, 100.5)), ((140.5, 100.5), (180.5, 110.5))]
    for start, end in segments:
        bounds = brush_segment_bounds(document.size, start, end, settings.size_px)
        assert bounds is not None
        recorder.include_before(bounds)
        paint_mask_segment(document.erase_mask, start, end, settings)

    edited = document.erase_mask.tobytes()
    command = recorder.finish()
    assert command is not None
    assert command.patch_size[0] < 100
    assert command.patch_size[1] < 30
    assert command.memory_cost < document.size[0] * document.size[1] // 100

    stack = CommandStack()
    stack.push_applied(command)
    assert stack.undo_count == 1
    stack.undo()
    assert document.erase_mask.getbbox() is None
    stack.redo()
    assert document.erase_mask.tobytes() == edited


def test_ai_mask_is_validated_before_atomic_history_commit() -> None:
    document = ImageDocument(Image.new("RGBA", (12, 8), "white"))
    stack = CommandStack()
    invalid = Image.new("L", (11, 8), 0)

    with pytest.raises(ValueError, match="AI 蒙版"):
        ApplyAiMaskCommand.create(document, invalid)
    assert stack.undo_count == 0
    assert document.ai_mask.getextrema() == (255, 255)

    command = ApplyAiMaskCommand.create(document, Image.new("L", document.size, 64))
    command.redo()
    stack.push_applied(command)
    assert document.ai_mask.getextrema() == (64, 64)
    assert stack.undo()
    assert document.ai_mask.getextrema() == (255, 255)
    assert stack.redo()
    assert document.ai_mask.getextrema() == (64, 64)


def test_two_mask_recorder_undoes_and_cancels_both_masks_together() -> None:
    """一次互斥笔画会同时改动两张蒙版，撤销和取消都必须一并还原。"""
    document = ImageDocument(Image.new("RGBA", (60, 40), "white"))
    # 预置一层恢复，模拟“先恢复、再抠除”的场景
    document.restore_mask.paste(255, (20, 10, 40, 30))
    settings = BrushSettings(size_px=6)
    start, end = (30.5, 20.5), (34.5, 20.5)

    recorder = BrushStrokeRecorder(
        document,
        "erase_mask",
        "抠除笔画",
        opposite_mask_name="restore_mask",
    )
    bounds = brush_segment_bounds(document.size, start, end, settings.size_px)
    assert bounds is not None
    recorder.include_before(bounds)
    paint_mask_segment_exclusive(document.erase_mask, document.restore_mask, start, end, settings)

    before_erase = document.erase_mask.tobytes()
    before_restore = document.restore_mask.tobytes()
    assert document.erase_mask.getbbox() is not None
    # 抠除命中的像素必须已经从恢复蒙版清除
    assert document.restore_mask.getpixel((32, 20)) == 0

    # 一次笔画只产生一个历史步骤，撤销时两张蒙版同时还原
    command = recorder.finish()
    assert command is not None
    stack = CommandStack()
    stack.push_applied(command)
    assert stack.undo_count == 1
    stack.undo()
    assert document.erase_mask.getextrema() == (0, 0)
    assert document.restore_mask.getpixel((32, 20)) == 255
    stack.redo()
    assert document.erase_mask.tobytes() == before_erase
    assert document.restore_mask.tobytes() == before_restore


def test_two_mask_recorder_cancel_restores_both_masks() -> None:
    """取消未完成的笔画时，写入侧与清除侧都要回到落笔前状态。"""
    document = ImageDocument(Image.new("RGBA", (40, 30), "white"))
    settings = BrushSettings(size_px=5)
    start, end = (20.5, 15.5), (22.5, 15.5)

    recorder = BrushStrokeRecorder(
        document,
        "restore_mask",
        "恢复笔画",
        opposite_mask_name="erase_mask",
    )
    bounds = brush_segment_bounds(document.size, start, end, settings.size_px)
    assert bounds is not None
    recorder.include_before(bounds)
    paint_mask_segment_exclusive(document.restore_mask, document.erase_mask, start, end, settings)
    assert document.restore_mask.getbbox() is not None

    assert recorder.cancel()
    assert document.restore_mask.getextrema() == (0, 0)
    assert document.erase_mask.getextrema() == (0, 0)
