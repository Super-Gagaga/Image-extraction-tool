"""ONNX 抠图模型自检：打印输入输出规格、加载/推理耗时并输出蒙版预览。

用法（在仓库根目录）：

    python scripts/check_model.py models/birefnet-general.onnx
    python scripts/check_model.py models/birefnet-general.onnx --input-size 1024x1024 --save check-preview

动态尺寸模型默认按 320×320 推理；BiRefNet、RMBG 等文件名会自动使用 1024×1024，
也可以用 ``--input-size`` 或环境变量 ``IMAGE_EXTRACTION_INPUT_SIZE`` 覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from image_extraction_tool.infrastructure.model_profiles import (
    DEFAULT_DYNAMIC_INPUT_SIZE,
    INPUT_SIZE_ENVIRONMENT_VARIABLE,
    parse_input_size,
    resolve_dynamic_input_size,
)
from image_extraction_tool.infrastructure.segmenters import OnnxU2NetSegmenter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_image() -> Image.Image:
    """生成一张带主体和背景的合成图，只用于验证推理链路，不用于评估画质。"""
    width, height = 720, 480
    image = Image.new("RGB", (width, height), (36, 42, 58))
    draw = ImageDraw.Draw(image)
    for y in range(0, height, 6):
        shade = 36 + (y * 60 // height)
        draw.rectangle([0, y, width, y + 2], fill=(shade, shade + 6, shade + 20))
    draw.ellipse([180, 70, 540, 430], fill=(212, 126, 86))
    draw.ellipse([280, 170, 440, 330], fill=(246, 224, 206))
    draw.rectangle([330, 30, 390, 90], fill=(246, 224, 206))
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查本地 ONNX 抠图模型是否可被当前适配器使用。")
    parser.add_argument("model", nargs="?", default="models/u2net.onnx", help="ONNX 模型路径（默认 models/u2net.onnx）")
    parser.add_argument("--input-size", help="动态尺寸模型的推理尺寸，例如 1024 或 1024x1024")
    parser.add_argument("--image", help="可选的真实测试图片；缺省使用合成图")
    parser.add_argument("--save", metavar="PREFIX", help="保存蒙版与透明合成图，例如 check-preview")
    parser.add_argument("--no-hash", action="store_true", help="跳过 SHA-256 计算")
    args = parser.parse_args(argv)

    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        print(f"错误：未找到模型文件 {model_path}")
        return 1

    print(f"模型：{model_path}")
    print(f"大小：{model_path.stat().st_size / 1024 / 1024:.1f} MB")
    if not args.no_hash:
        print(f"SHA-256：{_sha256(model_path)}")

    if args.input_size:
        try:
            dynamic_size = parse_input_size(args.input_size)
        except ValueError as exc:
            print(f"错误：{exc}")
            return 1
    else:
        dynamic_size = resolve_dynamic_input_size(model_path)

    segmenter = OnnxU2NetSegmenter(model_path, dynamic_input_size=dynamic_size)
    try:
        started = time.perf_counter()
        segmenter.load()
        load_seconds = time.perf_counter() - started
        info = segmenter.describe()
    except Exception as exc:  # 自检脚本需要把各类加载失败都转成可读信息
        print(f"错误：模型加载失败：{exc}")
        return 1

    outputs = info["outputs"]
    print(f"输入：{info['input_name']} {info['input_shape']}")
    for output in outputs:
        print(f"输出：{output['name']} {output['shape']}")
    if len(outputs) != 1:
        print("提示：模型有多个输出，适配器只读取第一个；请确认第一个输出是最终蒙版。")
    width, height = info["input_size"]
    if info["dynamic"]:
        print(f"尺寸类型：动态 -> 按 {width}×{height} 推理")
        if not args.input_size and dynamic_size == DEFAULT_DYNAMIC_INPUT_SIZE:
            print(
                "提示：未识别的动态尺寸模型按 320×320 推理；若这是 1024×1024 模型，"
                f"请用 --input-size 1024 或设置 {INPUT_SIZE_ENVIRONMENT_VARIABLE}=1024。"
            )
    else:
        print(f"尺寸类型：固定 -> 按 {width}×{height} 推理")
    print(f"加载耗时：{load_seconds:.2f} s")

    if args.image:
        try:
            image = Image.open(args.image)
            image.load()
        except (OSError, ValueError) as exc:
            print(f"错误：无法读取测试图片：{exc}")
            return 1
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA")
    else:
        image = _sample_image()

    try:
        started = time.perf_counter()
        mask = segmenter.predict(image, progress=lambda value: print(f"\r推理进度：{value}%", end="", flush=True))
        inference_seconds = time.perf_counter() - started
    except Exception as exc:  # 同上，自检脚本负责汇总错误
        print(f"\n错误：推理失败：{exc}")
        return 1
    print()

    array = np.asarray(mask, dtype=np.uint8)
    foreground = float((array > 127).mean() * 100)
    print(f"推理耗时：{inference_seconds:.2f} s（原图 {image.width}×{image.height}）")
    print(f"蒙版：{mask.mode} {mask.size}，范围 {mask.getextrema()}，平均 {array.mean():.1f}，前景占比 {foreground:.1f}%")

    if args.save:
        prefix = Path(args.save)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        mask_path = prefix.with_name(prefix.name + "-mask.png")
        rgba_path = prefix.with_name(prefix.name + ".png")
        mask.save(mask_path)
        rgba = image.convert("RGB")
        rgba.putalpha(mask)
        rgba.save(rgba_path)
        print(f"已保存：{mask_path}、{rgba_path}")

    print("自检通过：模型可加载，输出为单通道蒙版。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
