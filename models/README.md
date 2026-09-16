# 本地模型

模型权重不提交到仓库，也不会由程序自动上传或下载。

## 默认模型

当前适配器支持 U²-Net 及兼容的单输入 ONNX 显著性分割模型：

- 输入为 `NCHW` 四维 RGB 张量，通道数为 3；
- 固定输入尺寸直接读取模型声明；动态尺寸默认按 320×320 推理；
- 第一项输出应为单通道二维蒙版，可使用 `[1, 1, H, W]`、`[1, H, W]` 或 `[H, W]`；
- 输出可为 0～1 概率、0～255 灰度或未经过 Sigmoid 的 logits。

把模型命名为 `u2net.onnx` 并放在此目录，即可点击“智能抠图”。也可以在界面点击“选择模型”，或设置环境变量：

```powershell
$env:IMAGE_EXTRACTION_MODEL = "D:\Models\u2net.onnx"
image-extraction-tool
```

模型由用户自行取得。使用前应核对模型来源、许可证和 SHA-256；项目不代为分发第三方权重。Windows 可使用以下命令核验下载页面公布的摘要：

```powershell
Get-FileHash .\models\u2net.onnx -Algorithm SHA256
```

模型缺失、格式不兼容或加载失败时，程序会保留当前文档，颜色工具和手动画笔仍可继续使用。
