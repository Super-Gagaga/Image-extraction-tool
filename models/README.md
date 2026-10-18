# 本地模型

模型权重不提交到仓库，也不会由程序自动上传或下载。

## 支持的模型格式

当前适配器支持 U²-Net 及兼容的单输入 ONNX 显著性分割模型：

- 输入为 `NCHW` 四维 RGB 张量，通道数为 3；
- 固定输入尺寸直接读取模型声明；动态尺寸默认按 320×320 推理（见下节）；
- 第一项输出应为单通道二维蒙版，可使用 `[1, 1, H, W]`、`[1, H, W]` 或 `[H, W]`；
- 输出可为 0～1 概率、0～255 灰度或未经过 Sigmoid 的 logits。

## 动态输入尺寸

BiRefNet、RMBG-1.4 等模型通常在 1024×1024 上训练和推理。如果下载到的 ONNX 把
H/W 声明为动态维度，程序默认会按 320×320 推理，效果会明显变差。处理方式：

1. 优先使用固定 1024×1024 的导出（Hugging Face 上的 `onnx/model.onnx` 通常是固定尺寸）；
2. 文件名包含 `birefnet`、`rmbg`、`isnet` 时，程序自动按 1024×1024 推理；
3. 其他动态尺寸模型用环境变量指定：

```powershell
$env:IMAGE_EXTRACTION_INPUT_SIZE = "1024x1024"
```

## 推荐模型

| 模型 | 权重体积 | 推理尺寸 | 特点 | 许可证 |
| --- | --- | --- | --- | --- |
| U²-Net（默认 `u2net.onnx`） | 标准权重约 176 MB；本目录当前为约 25 MB 轻量导出 | 320×320 | 基线，最快 | Apache-2.0 |
| RMBG-1.4 | 约 168 MB（int8 约 42 MB） | 1024×1024 | 人像/商品效果好，CPU 快 | 仅非商用 |
| BiRefNet general | 约 927 MB | 1024×1024 | 发丝/透明边缘最好 | MIT |
| BiRefNet lite | 约 214 MB | 1024×1024 | 质量接近 general，体积小、CPU 略快 | MIT |

下载地址（Hugging Face；国内可先设置 `$env:HF_ENDPOINT = "https://hf-mirror.com"`）：

- BiRefNet：`onnx-community/BiRefNet-ONNX` 的 `onnx/model.onnx`
- BiRefNet lite：`onnx-community/BiRefNet_lite-ONNX` 的 `onnx/model.onnx`
- RMBG-1.4：`briaai/RMBG-1.4` 的 `onnx/model.onnx` 或 `onnx/model_quantized.onnx`

把文件放到本目录并改成易识别的名字，例如 `birefnet-general.onnx`，然后在界面点击“选择模型”。
也可以设置 `IMAGE_EXTRACTION_MODEL` 让程序启动时直接使用该模型：

```powershell
$env:IMAGE_EXTRACTION_MODEL = "D:\Models\birefnet-general.onnx"
image-extraction-tool
```

`models/*` 已被 `.gitignore` 忽略，权重不会误提交。

> rembg 是 Python 封装库而不是模型：默认内置的 U²-Net 与本项目已支持的模型同架构，而且会
> 自动下载权重到 `~/.u2net`，与本项目“模型由用户自行提供”的策略不一致。需要它某个权重时，
> 直接下载对应 `.onnx` 即可。

## 校验与自检

使用前应核对模型来源、许可证和 SHA-256：

```powershell
Get-FileHash .\models\birefnet-general.onnx -Algorithm SHA256
```

项目自带自检脚本，会打印输入输出规格、动态/固定尺寸、加载与推理耗时，并验证输出是单通道蒙版：

```powershell
python scripts\check_model.py models\birefnet-general.onnx
```

已知 BiRefNet 权重的 SHA-256（来自 Hugging Face LFS，可按模型卡更新）：

- `BiRefNet-ONNX/onnx/model.onnx`：`58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c`
- `BiRefNet_lite-ONNX/onnx/model.onnx`：`5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333`

模型缺失、格式不兼容或加载失败时，程序会保留当前文档，颜色工具和手动画笔仍可继续使用。
