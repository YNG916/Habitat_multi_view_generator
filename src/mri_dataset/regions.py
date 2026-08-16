"""Pure geometry helpers for official HSSD semantic region polygons."""
from __future__ import annotations
import math
from typing import Iterable, Optional, Sequence
import numpy as np
from PIL import Image, ImageDraw

def polygon_area_xz(poly_loop: Sequence[Sequence[float]]) -> float:
    polygon=np.asarray(poly_loop,dtype=np.float64)
    if polygon.ndim!=2 or polygon.shape[0]<3 or polygon.shape[1]!=3: return 0.0
    x,z=polygon[:,0],polygon[:,2]
    return float(abs(np.dot(x,np.roll(z,-1))-np.dot(z,np.roll(x,-1)))*.5)

def point_in_polygon_xz(point,poly_loop,boundary_tolerance_m:float=1e-5)->bool:
    """Membership in an authored HSSD poly_loop, including its boundary."""
    polygon=np.asarray(poly_loop,dtype=np.float64); query=np.asarray(point,dtype=np.float64)
    if polygon.ndim!=2 or polygon.shape[0]<3 or polygon.shape[1]!=3: return False
    px,pz=float(query[0]),float(query[2]); inside=False
    for index in range(len(polygon)):
        ax,az=polygon[index,(0,2)]; bx,bz=polygon[(index+1)%len(polygon),(0,2)]
        dx,dz=bx-ax,bz-az; length2=dx*dx+dz*dz
        if length2>0:
            t=max(0.,min(1.,((px-ax)*dx+(pz-az)*dz)/length2))
            if math.hypot(px-(ax+t*dx),pz-(az+t*dz))<=boundary_tolerance_m: return True
        if (az>pz)!=(bz>pz) and px<ax+(pz-az)*dx/(bz-az): inside=not inside
    return inside

def unique_region_for_point(point,regions:Iterable,floor_tolerance_m:float)->Optional[object]:
    query=np.asarray(point,dtype=np.float64)
    matches=[r for r in regions if abs(float(query[1])-float(r.representative_floor_y))<=float(floor_tolerance_m) and point_in_polygon_xz(query,r.semantic_polygon_world)]
    return matches[0] if len(matches)==1 else None

def region_mask(mapping,poly_loop)->np.ndarray:
    """Rasterize an authored HSSD polygon in the exact BEV pixel frame."""
    image=Image.new("L",(mapping.width,mapping.height),0)
    vertices=[
        mapping.world_to_bev(float(point[0]),float(point[2]))
        for point in poly_loop
    ]
    ImageDraw.Draw(image).polygon(vertices,fill=1)
    return np.asarray(image,dtype=np.uint8)
