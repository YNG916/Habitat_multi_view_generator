from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from .world_state import ObjectState


def is_decomposed_canonical_id(canonical_id: str) -> bool:
    return "/decomposed/" in f"/{str(canonical_id).replace(chr(92),'/')}"


def canonical_template_index(template_handles, dataset_root) -> Dict[str, List[str]]:
    from pathlib import Path
    root=Path(dataset_root).resolve()
    result: Dict[str, List[str]]={}
    for handle in template_handles:
        try:
            canonical=Path(handle).resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        result.setdefault(canonical,[]).append(handle)
    return result


def resolve_canonical_templates(template_handles, dataset_root, canonical_ids) -> Dict[str, str]:
    index=canonical_template_index(template_handles,dataset_root)
    result={}
    for canonical in canonical_ids:
        matches=index.get(str(canonical),[])
        if len(matches)!=1:
            raise KeyError(
                f"Canonical HSSD asset {canonical} resolved to {len(matches)} handles"
            )
        result[str(canonical)]=matches[0]
    return result


def handles_by_suffix(template_manager, suffixes: Iterable[str]) -> Dict[str, str]:
    handles = template_manager.get_template_handles()
    resolved = {}
    for suffix in suffixes:
        matches = [handle for handle in handles if handle.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one rigid template ending in {suffix!r}, found {matches}")
        resolved[suffix] = matches[0]
    return resolved


def list_template_records(template_manager) -> List[dict]:
    records = []
    for handle in template_manager.get_template_handles():
        template = template_manager.get_template_by_handle(handle)
        records.append(
            {
                "handle": handle,
                "render_asset": str(template.render_asset_handle),
                "collision_asset": str(template.collision_asset_handle),
                "semantic_id": int(template.semantic_id),
                "scale": [float(x) for x in template.scale],
            }
        )
    return records


def aabb_dict(aabb, transform=None) -> dict:
    local_minimum = np.asarray(aabb.min, dtype=np.float64)
    local_maximum = np.asarray(aabb.max, dtype=np.float64)
    corners = np.array(
        [[x, y, z] for x in (local_minimum[0], local_maximum[0])
         for y in (local_minimum[1], local_maximum[1])
         for z in (local_minimum[2], local_maximum[2])],
        dtype=np.float64,
    )
    if transform is not None:
        from .coordinates import transform_points
        corners = transform_points(np.asarray(transform, dtype=np.float64), corners)
    minimum = corners.min(axis=0)
    maximum = corners.max(axis=0)
    return {
        "min_world": minimum.tolist(),
        "max_world": maximum.tolist(),
        "dimensions_m": (maximum - minimum).tolist(),
    }


def controlled_object_collision_free(obj: ObjectState, state, scene_bounds, margin_m: float = 0.05) -> bool:
    if not obj.active:
        return True
    bbox = obj.bbox
    if not bbox:
        return False
    low = np.asarray(bbox["min_world"])
    high = np.asarray(bbox["max_world"])
    scene_low, scene_high = map(np.asarray, scene_bounds)
    if np.any(low < scene_low - margin_m) or np.any(high > scene_high + margin_m):
        return False
    for robot in state.robots:
        delta = np.asarray(obj.position_world)[[0, 2]] - np.asarray(robot.base_position_world)[[0, 2]]
        if np.linalg.norm(delta) < 0.35:
            return False
    for other in state.objects:
        if other.instance_id == obj.instance_id or not other.active or not other.bbox:
            continue
        other_low = np.asarray(other.bbox["min_world"])
        other_high = np.asarray(other.bbox["max_world"])
        overlap = np.all(high > other_low + margin_m) and np.all(other_high > low + margin_m)
        if overlap:
            return False
    return True
