from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

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


def _fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    source=image.convert("RGB")
    scale=min(width/source.width,height/source.height)
    size=(max(1,int(round(source.width*scale))),max(1,int(round(source.height*scale))))
    return source.resize(size)


def contact_sheet(
    bev: Image.Image,
    robot_images: Iterable[Image.Image],
    title: str = "",
    diagnostic_images: Iterable[Image.Image] = (),
) -> Image.Image:
    """Compose an HD review sheet with legible, equal-width BEV diagnostics."""
    robots=[image.convert("RGB").resize((1024,1024)) for image in robot_images]
    diagnostics=[image.convert("RGB") for image in diagnostic_images]
    width=1024*max(1,len(robots))
    top_panels=[bev.convert("RGB"),*diagnostics]
    cell_width=width//len(top_panels)
    top_height=min(1024,cell_width)
    title_h=32
    height=title_h+top_height+1024
    sheet=Image.new("RGB",(width,height),(24,24,24))
    draw=ImageDraw.Draw(sheet)
    draw.text((8,8),title,fill="white",font=ImageFont.load_default())
    for index,panel in enumerate(top_panels):
        fitted=_fit_panel(panel,cell_width,top_height)
        x=index*cell_width+(cell_width-fitted.width)//2
        y=title_h+(top_height-fitted.height)//2
        sheet.paste(fitted,(x,y))
    for index,image in enumerate(robots):
        sheet.paste(image,(index*1024,title_h+top_height))
    return sheet


def height_visualization(height: np.ndarray) -> Image.Image:
    values=np.asarray(height,dtype=np.float32)
    valid=np.isfinite(values)
    result=np.zeros((*values.shape,3),dtype=np.uint8)
    if valid.any():
        low=float(np.percentile(values[valid],2))
        high=float(np.percentile(values[valid],98))
        safe=np.where(valid,values,low)
        normalized=np.clip((safe-low)/max(high-low,1e-6),0.,1.)
        result[...,0]=(normalized*255).astype(np.uint8)
        result[...,1]=((1.-np.abs(normalized-.5)*2.)*255).astype(np.uint8)
        result[...,2]=((1.-normalized)*255).astype(np.uint8)
        result[~valid]=0
    return Image.fromarray(result)


def _binary_mask(values: np.ndarray, name: str) -> np.ndarray:
    array=np.asarray(values,dtype=np.uint8)
    if array.ndim!=2:
        raise ValueError(f"{name} must be a 2-D array")
    unique=set(map(int,np.unique(array)))
    if not unique.issubset({0,1}):
        raise ValueError(f"{name} must be binary, got {sorted(unique)}")
    return array


def occupancy_visualization(
    occupancy: np.ndarray, region_mask: Optional[np.ndarray] = None
) -> Image.Image:
    """Visualize binary occupancy; green is navigable inside the region."""
    values=_binary_mask(occupancy,"occupancy")
    result=np.zeros((*values.shape,3),dtype=np.uint8)
    result[values==1]=(210,210,210)
    if region_mask is not None:
        mask=_binary_mask(region_mask,"region_mask")
        if mask.shape!=values.shape:
            raise ValueError("occupancy and region_mask shapes differ")
        result[(values==1)&(mask==1)]=(35,190,85)
    return Image.fromarray(result)


def region_overlay_visualization(
    rgb: np.ndarray, region_mask: np.ndarray, occupancy: Optional[np.ndarray] = None
) -> Image.Image:
    """Overlay the authored HSSD region and its navigable support on RGB."""
    base=np.asarray(rgb,dtype=np.uint8)
    mask=_binary_mask(region_mask,"region_mask")
    if base.shape!=(*mask.shape,3):
        raise ValueError("RGB and region_mask shapes differ")
    result=base.astype(np.float32)
    tint=np.zeros_like(result); tint[...,0]=25; tint[...,1]=210; tint[...,2]=235
    alpha=(mask>0)[...,None]*.18
    result=result*(1.-alpha)+tint*alpha
    padded=np.pad(mask,1,constant_values=0)
    interior=(
        padded[1:-1,1:-1]&padded[:-2,1:-1]&padded[2:,1:-1]
        &padded[1:-1,:-2]&padded[1:-1,2:]
    )
    boundary=(mask>0)&~interior.astype(bool)
    result[boundary]=(0,255,255)
    if occupancy is not None:
        nav=_binary_mask(occupancy,"occupancy")
        if nav.shape!=mask.shape:
            raise ValueError("occupancy and region_mask shapes differ")
        support=(nav==1)&(mask==1)
        result[support]=result[support]*.82+np.array([35,190,85],np.float32)*.18
    return Image.fromarray(np.clip(result,0,255).astype(np.uint8))


def labeled_panel(image: Image.Image, label: str) -> Image.Image:
    result=image.convert("RGB").copy()
    draw=ImageDraw.Draw(result)
    height=18
    draw.rectangle((0,0,result.width-1,height),fill=(0,0,0))
    draw.text((4,4),str(label),fill=(255,255,255),font=ImageFont.load_default())
    return result


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
