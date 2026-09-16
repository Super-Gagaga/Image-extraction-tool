"""应用级异常层次结构。"""


class ImageExtractionError(Exception):
    """所有可预期的、面向用户的异常基类。"""


class StartupError(ImageExtractionError):
    """桌面应用无法完成启动时抛出。"""


class ImageLoadError(ImageExtractionError):
    """图片无法解码为受支持的文档时抛出。"""


class ExportError(ImageExtractionError):
    """透明 PNG 无法安全写入目标路径时抛出。"""


class SegmenterError(ImageExtractionError):
    """本地智能抠图服务发生可恢复错误。"""


class ModelMissingError(SegmenterError):
    """配置的本地模型文件不存在或不可读。"""


class ModelLoadError(SegmenterError):
    """ONNX Runtime 无法创建模型会话。"""


class InferenceError(SegmenterError):
    """模型输入、推理或输出无法转换为有效蒙版。"""


class SegmentationCancelled(SegmenterError):
    """用户取消了尚未提交的智能抠图任务。"""
