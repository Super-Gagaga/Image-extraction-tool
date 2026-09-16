# 本地智能抠图工具

一款面向 Windows 10/11 的离线桌面抠图工具。项目按 UI、应用、领域和基础设施四层组织，原始图片与编辑蒙版相互独立。

当前已支持 PNG、JPG、JPEG、WEBP 导入、透明棋盘格画布、滚轮指针中心缩放、右键拖动平移、适应窗口与 100% 查看，1～200 原图像素的抠除/恢复画笔，以及单点取色、区域多色提取、颜色容差、独立清空、撤销/重做、本地 ONNX 智能抠图和后台透明 PNG 导出。

图片读取由 Pillow 完成：`Image.open()` 根据文件头选择 JPEG、PNG 或 WEBP 解码器，随后使用 `ImageOps.exif_transpose()` 校正 EXIF 方向，并通过 `.convert("RGBA")` 统一为四通道 8 位像素。读取过程保持原始分辨率，不缩放、不重采样；没有 Alpha 的图片会得到 `A=255`。

```python
from image_extraction_tool.infrastructure.image_io import load_image

document = load_image("input.png")
width, height = document.size
pixel = document.original_rgba.getpixel((10, 20))  # (R, G, B, A)
rgba_bytes = document.original_rgba.tobytes("raw", "RGBA")
# 也可以使用 document.original_pixel(10, 20) 和 document.original_bytes()
```

`rgba_bytes` 按行排列，每个像素连续占 4 个字节；文件不存在、格式不支持或图片损坏时，`load_image()` 会抛出 `ImageLoadError`。

画布使用原图像素的最近邻显示。导出时在原图分辨率上合并颜色匹配区域和抠除画笔区域，命中像素写为完全透明；其余像素保留原始 RGB 与 Alpha，并保存为 RGBA PNG。

智能抠图默认读取 `models/u2net.onnx`，也可在界面选择其他兼容模型或通过 `IMAGE_EXTRACTION_MODEL` 环境变量指定。模型格式和校验方式见 [模型说明](models/README.md)。模型加载和推理都在工作线程中执行，可显示进度并取消。

## 开发环境

- Python 3.11～3.14（64 位）
- Windows 10/11

建议在虚拟环境中安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

启动应用：

```powershell
image-extraction-tool
```

也可以直接运行模块：

```powershell
python -m image_extraction_tool
```

运行测试：

```powershell
python -m pytest
```

应用日志默认保存在 `%LOCALAPPDATA%\ImageExtractionTool\logs\application.log`。技术设计与阶段验收标准见 [技术设计文档](docs/抠图工具技术设计文档.md)。
