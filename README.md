# 本地智能抠图工具

一款面向 Windows 10/11 的离线桌面抠图工具。项目按 UI、应用、领域和基础设施四层组织，原始图片与编辑蒙版相互独立。

当前已支持 PNG、JPG、JPEG、WEBP 导入、透明棋盘格画布、滚轮指针中心缩放、右键拖动平移、适应窗口与 100% 查看，以及 1～200 原图像素的抠除/恢复画笔、硬度、不透明度、独立清空和后台透明 PNG 导出。

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
