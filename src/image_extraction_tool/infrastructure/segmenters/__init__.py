"""可替换的本地抠图算法适配器。"""

from image_extraction_tool.infrastructure.segmenters.onnx_u2net import OnnxU2NetSegmenter

__all__ = ["OnnxU2NetSegmenter"]
