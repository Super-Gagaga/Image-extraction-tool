"""应用级异常层次结构。"""


class ImageExtractionError(Exception):
    """所有可预期的、面向用户的异常基类。"""


class StartupError(ImageExtractionError):
    """桌面应用无法完成启动时抛出。"""


class ImageLoadError(ImageExtractionError):
    """图片无法解码为受支持的文档时抛出。"""


class ExportError(ImageExtractionError):
    """透明 PNG 无法安全写入目标路径时抛出。"""
