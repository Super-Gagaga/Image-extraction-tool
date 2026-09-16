# 本地智能抠图工具

一款面向 Windows 10/11 的离线桌面抠图工具。项目按 UI、应用、领域和基础设施四层组织，原始图片与编辑蒙版相互独立。

当前已支持 PNG、JPG、JPEG、WEBP 导入、透明棋盘格画布、滚轮指针中心缩放、右键拖动平移、适应窗口与 100% 查看，1～200 原图像素的抠除/恢复画笔，以及单点取色、区域多色提取、颜色容差、独立清空、撤销/重做和后台透明 PNG 导出。

画布使用原图像素的最近邻显示。导出时在原图分辨率上合并颜色匹配区域和抠除画笔区域，命中像素写为完全透明；其余像素保留原始 RGB 与 Alpha，并保存为 RGBA PNG。

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
