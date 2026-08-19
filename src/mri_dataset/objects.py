from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List

import numpy as np

from .world_state import ObjectState


def inspect_approved_object_registry(
    registry: dict,
    semantic_category_ids=None,
) -> dict:
    """Validate approved-pool structure without requiring HSSD or Habitat."""
    errors = []
    records = []
    approved = registry.get("approved_assets")
    if registry.get("schema_version") != "2.0.0":
        errors.append("registry schema_version must be 2.0.0")
    if not isinstance(approved, dict):
        return {"passed": False, "records": [], "errors": errors + ["approved_assets must be an object"]}

    seen = {}
    allowed_categories = set(semantic_category_ids or {})
    for raw_category in sorted(approved):
        category = str(raw_category)
        category_records = approved[raw_category]
        if allowed_categories and category not in allowed_categories:
            errors.append(f"{category}: missing from semantic_category_ids")
        if not isinstance(category_records, list) or not category_records:
            errors.append(f"{category}: approved pool must be a non-empty list")
            continue
        category_semantic_ids = set()
        for index, raw_record in enumerate(category_records):
            prefix = f"{category}[{index}]"
            item = {"category": category, "index": index, "errors": []}
            if not isinstance(raw_record, dict):
                item["errors"].append("record must be an object")
                records.append(item)
                errors.append(f"{prefix}: record must be an object")
                continue
            canonical = raw_record.get("canonical_id")
            item["canonical_id"] = canonical
            safe = isinstance(canonical, str) and bool(canonical) and "\\" not in canonical
            if safe:
                pure = PurePosixPath(canonical)
                safe = (
                    not pure.is_absolute()
                    and canonical == pure.as_posix()
                    and all(part not in ("", ".", "..") for part in pure.parts)
                    and canonical.endswith(".object_config.json")
                )
            item["canonical_id_safe"] = bool(safe)
            item["decomposed"] = bool(
                isinstance(canonical,str)
                and is_decomposed_canonical_id(canonical)
            )
            if not safe:
                item["errors"].append("canonical_id is not a safe normalized relative object config path")
            if safe and item["decomposed"]:
                item["errors"].append("decomposed assets are forbidden")
            if safe:
                if canonical in seen:
                    item["errors"].append(f"duplicate canonical_id (first seen at {seen[canonical]})")
                else:
                    seen[canonical] = prefix

            fingerprint = raw_record.get("asset_fingerprint")
            item["asset_fingerprint_expected"] = fingerprint
            if not (
                isinstance(fingerprint, str)
                and len(fingerprint) == 64
                and all(char in "0123456789abcdef" for char in fingerprint)
            ):
                item["errors"].append("asset_fingerprint must be a lowercase SHA-256 digest")

            semantic_id = raw_record.get("semantic_id")
            try:
                semantic_id = int(semantic_id)
                if semantic_id <= 0:
                    raise ValueError
                category_semantic_ids.add(semantic_id)
                item["semantic_id_expected"] = semantic_id
            except (TypeError, ValueError):
                item["errors"].append("semantic_id must be a positive integer")

            extent = raw_record.get("extent_xyz_m")
            extent_ok = (
                isinstance(extent, list)
                and len(extent) == 3
                and all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in extent)
            )
            if not extent_ok:
                item["errors"].append("extent_xyz_m must contain three finite positive values")
            else:
                item["extent_xyz_m_expected"] = [float(value) for value in extent]

            item["passed"] = not item["errors"]
            records.append(item)
            errors.extend(f"{prefix}: {message}" for message in item["errors"])
        if len(category_semantic_ids) > 1:
            errors.append(f"{category}: approved records have inconsistent semantic_id values")

    return {"passed": not errors, "records": records, "errors": errors}


def inspect_approved_object_sources(registry: dict, dataset_root) -> dict:
    """Validate current HSSD files/hashes; called once by integration preflight."""
    root = Path(dataset_root).resolve()
    pure = inspect_approved_object_registry(registry)
    records = []
    errors = list(pure["errors"])
    for pure_record in pure["records"]:
        item = dict(pure_record)
        item["errors"] = list(pure_record["errors"])
        canonical = item.get("canonical_id")
        source_path = None
        if item.get("canonical_id_safe"):
            source_path = (root / canonical).resolve()
            try:
                source_path.relative_to(root)
            except ValueError:
                item["errors"].append(
                    "canonical_id resolves outside the HSSD dataset root"
                )
                source_path = None
        item["source_path"] = str(source_path) if source_path is not None else None
        item["source_exists"] = bool(source_path and source_path.is_file())
        if source_path is not None and not source_path.is_file():
            item["errors"].append("source object config does not exist")
        if source_path is not None and source_path.is_file():
            actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            item["asset_fingerprint_actual"] = actual_hash
            item["hash_ok"] = bool(
                actual_hash == item.get("asset_fingerprint_expected")
            )
            if not item["hash_ok"]:
                item["errors"].append("asset fingerprint mismatch")
            try:
                source_config = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                item["errors"].append(f"cannot parse source object config: {exc}")
            else:
                actual_semantic = source_config.get("semantic_id")
                item["semantic_id_source"] = actual_semantic
                if actual_semantic != item.get("semantic_id_expected"):
                    item["errors"].append(
                        "source semantic_id does not match registry"
                    )
                item["source_up"] = source_config.get("up")
                item["source_front"] = source_config.get("front")
                up = np.asarray(source_config.get("up", []), dtype=np.float64)
                front = np.asarray(source_config.get("front", []), dtype=np.float64)
                orientation_ok = (
                    up.shape == (3,)
                    and front.shape == (3,)
                    and np.all(np.isfinite(up))
                    and np.all(np.isfinite(front))
                    and np.linalg.norm(up) > 0
                    and np.linalg.norm(front) > 0
                    and float(np.dot(up / np.linalg.norm(up), [0, 1, 0])) > 0.999
                    and abs(float(np.dot(up, front))) < 1e-6
                )
                item["identity_orientation_metadata_ok"] = bool(orientation_ok)
                if not orientation_ok:
                    item["errors"].append(
                        "source up/front metadata is inconsistent with identity upright orientation"
                    )
        item.setdefault("hash_ok", False)
        item["passed"] = not item["errors"]
        records.append(item)
        prefix = f"{item.get('category')}[{item.get('index')}]"
        source_only_errors = item["errors"][len(pure_record["errors"]):]
        errors.extend(f"{prefix}: {message}" for message in source_only_errors)
    return {"passed": not errors, "records": records, "errors": errors}


def normalize_canonical_asset_identifier(value: str) -> str:
    """Return a canonical HSSD-relative object-config ID or raise ValueError."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("asset_identifier is not a normalized canonical path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or value != pure.as_posix()
        or any(part in ("", ".", "..") for part in pure.parts)
        or not value.endswith(".object_config.json")
    ):
        raise ValueError("asset_identifier is not a normalized canonical path")
    return value


def validate_controlled_object_identity(
    obj: ObjectState,
    controlled_object_pools,
    semantic_category_ids=None,
) -> List[str]:
    """Pure per-state approved category/asset membership validation."""
    errors = []
    category = str(obj.category)
    pools = controlled_object_pools or {}
    if category not in pools:
        errors.append(f"unknown controlled-object category {category!r}")
    if semantic_category_ids is not None:
        semantic_value = semantic_category_ids.get(category)
        if not isinstance(semantic_value, int) or semantic_value <= 0:
            errors.append(f"category {category!r} has no valid semantic category ID")
    try:
        canonical = normalize_canonical_asset_identifier(obj.asset_identifier)
    except ValueError as exc:
        errors.append(str(exc))
        return errors
    categories = sorted(
        pool_category
        for pool_category, identifiers in pools.items()
        if canonical in identifiers
    )
    if category in pools and canonical not in pools[category]:
        errors.append(
            f"asset_identifier is not approved for category {category!r}: {canonical}"
        )
    if not categories:
        errors.append(f"unknown approved asset_identifier: {canonical}")
    elif categories != [category]:
        errors.append(
            f"asset_identifier category ownership mismatch: {canonical} -> {categories}"
        )
    return errors


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


def resolve_canonical_template(handle_index, canonical_id: str) -> str:
    """Resolve one canonical identifier by exact lookup, never by suffix."""
    canonical = normalize_canonical_asset_identifier(canonical_id)
    matches = list(handle_index.get(canonical, []))
    if len(matches) != 1:
        raise KeyError(
            f"Canonical HSSD asset {canonical} resolved to {len(matches)} handles"
        )
    return matches[0]


def resolve_canonical_templates(template_handles, dataset_root, canonical_ids) -> Dict[str, str]:
    index=canonical_template_index(template_handles,dataset_root)
    return {
        normalize_canonical_asset_identifier(canonical): resolve_canonical_template(
            index, canonical
        )
        for canonical in canonical_ids
    }


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
