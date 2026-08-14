from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def depth_visualization(depth: np.ndarray, max_depth_m: float = 10.0) -> Image.Image:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    result = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        upper = min(max_depth_m, float(np.percentile(depth[valid], 99)))
        result[valid] = (np.clip(depth[valid] / max(upper, 1e-6), 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(result, mode="L")


def contact_sheet(bev: Image.Image, robot_images: Iterable[Image.Image], title: str = "") -> Image.Image:
    robots = [image.convert("RGB") for image in robot_images]
    thumb_w = 320
    thumb_h = 320
    bev_thumb = bev.convert("RGB")
    bev_thumb.thumbnail((thumb_w, thumb_h))
    robots = [image.resize((thumb_w, thumb_h)) for image in robots]
    width = max(thumb_w * max(1, len(robots)), bev_thumb.width)
    title_h = 28
    height = title_h + bev_thumb.height + thumb_h
    sheet = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 7), title, fill="white", font=ImageFont.load_default())
    sheet.paste(bev_thumb, ((width - bev_thumb.width) // 2, title_h))
    y = title_h + bev_thumb.height
    for index, image in enumerate(robots):
        sheet.paste(image, (index * thumb_w, y))
    return sheet


def before_after_contact_sheet(before_dir: Path, after_dir: Path, instruction: str) -> Image.Image:
    before = contact_sheet(
        Image.open(before_dir / "bev/annotated.png"),
        [Image.open(before_dir / f"robots/robot_{i:02d}/rgb.png") for i in range(1, 4)],
        "BEFORE",
    )
    after = contact_sheet(
        Image.open(after_dir / "bev/annotated.png"),
        [Image.open(after_dir / f"robots/robot_{i:02d}/rgb.png") for i in range(1, 4)],
        "AFTER",
    )
    width = max(before.width, after.width)
    result = Image.new("RGB", (width, before.height + after.height + 32), (16, 16, 16))
    result.paste(before, (0, 0))
    result.paste(after, (0, before.height))
    ImageDraw.Draw(result).text((8, before.height + after.height + 8), instruction, fill="white")
    return result
