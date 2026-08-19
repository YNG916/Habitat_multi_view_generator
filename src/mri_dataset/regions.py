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


def controlled_object_region_membership(
    position_world,
    floor_y: float,
    floor_spec,
    region_spec,
    pathfinder,
    floor_surface_y,
    floor_tolerance_m: float,
    navmesh_projection_tolerance_m: float = 1e-3,
) -> dict:
    """Validate object XZ on the state's floor, NavMesh and authored region."""
    reasons = []
    position = np.asarray(position_world, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        return {"passed": False, "reasons": ["non-finite object position"]}
    projected = np.array([position[0], float(floor_y), position[2]])
    snapped = np.asarray(pathfinder.snap_point(projected), dtype=np.float64)
    if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
        return {"passed": False, "reasons": ["no finite NavMesh floor projection"]}
    projection_error = float(np.linalg.norm(snapped[[0, 2]] - projected[[0, 2]]))
    if projection_error > float(navmesh_projection_tolerance_m):
        reasons.append(
            f"NavMesh XZ projection error {projection_error:.6f} m exceeds "
            f"{float(navmesh_projection_tolerance_m):.6f} m"
        )
    if not pathfinder.is_navigable(snapped):
        reasons.append("floor projection is not navigable")
    try:
        island_id = int(pathfinder.get_island(snapped))
    except Exception:
        island_id = None
        reasons.append("NavMesh island lookup failed")
    if island_id is not None and island_id not in set(region_spec.allowed_island_ids):
        reasons.append(f"NavMesh island {island_id} is outside selected region islands")
    try:
        physical_floor_y = float(floor_surface_y(snapped))
    except Exception as exc:
        physical_floor_y = None
        reasons.append(f"physical floor lookup failed: {exc}")
    if physical_floor_y is not None and (
        not math.isfinite(physical_floor_y)
        or abs(physical_floor_y - float(floor_y)) > float(floor_tolerance_m)
    ):
        reasons.append("floor projection is outside the selected physical floor")
    semantic_query = np.array([snapped[0], float(floor_y), snapped[2]])
    assigned = unique_region_for_point(
        semantic_query, floor_spec.regions, floor_tolerance_m
    )
    assigned_region_id = getattr(assigned, "region_id", None)
    if assigned_region_id != region_spec.region_id:
        reasons.append(
            "semantic region mismatch: "
            f"expected {region_spec.region_id}, got {assigned_region_id or 'none/ambiguous'}"
        )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "projected_floor_point": snapped.tolist(),
        "projection_error_xz_m": projection_error,
        "physical_floor_y": physical_floor_y,
        "island_id": island_id,
        "assigned_region_id": assigned_region_id,
    }

def region_mask(mapping,poly_loop)->np.ndarray:
    """Rasterize an authored HSSD polygon in the exact BEV pixel frame."""
    image=Image.new("L",(mapping.width,mapping.height),0)
    vertices=[
        mapping.world_to_bev(float(point[0]),float(point[2]))
        for point in poly_loop
    ]
    ImageDraw.Draw(image).polygon(vertices,fill=1)
    return np.asarray(image,dtype=np.uint8)
