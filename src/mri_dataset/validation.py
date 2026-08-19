from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .bev import BevMapping
from .coordinates import forward_from_quaternion, yaw_to_quaternion_xyzw
from .serialization import load_numeric
from .protocol import benchmark_visible_observers
from .objects import validate_controlled_object_identity
from .scene_registry import sha256_file
from .state_io import read_json


class ValidationError(RuntimeError):
    pass


BEV_REGISTRATION_MAX_ERROR_PIXELS = 3.0
BEV_REGISTRATION_MIN_TOLERANCE_M = 0.02
DEFAULT_HEIGHT_VALIDATION_MAX_ERROR_M = 0.02
DEFAULT_SUPPORT_CONTACT_TOLERANCE_M = 0.005


def instance_centroid_registration(pixels_rc, u, v, mapping):
    """Compare an OBJECT_ID mask centroid with its known world-space origin."""
    pixels = np.asarray(pixels_rc, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1:] != (2,) or not len(pixels):
        raise ValueError("Instance mask must contain at least one [row, col] pixel")
    centroid_v, centroid_u = np.mean(pixels, axis=0)
    delta_u = float(centroid_u - float(u))
    delta_v = float(centroid_v - float(v))
    error_m = float(
        np.hypot(
            delta_u * mapping.meters_per_pixel_x,
            delta_v * mapping.meters_per_pixel_z,
        )
    )
    tolerance_m = max(
        BEV_REGISTRATION_MIN_TOLERANCE_M,
        BEV_REGISTRATION_MAX_ERROR_PIXELS
        * max(mapping.meters_per_pixel_x, mapping.meters_per_pixel_z),
    )
    return {
        "centroid_uv": [float(centroid_u), float(centroid_v)],
        "error_pixels": float(np.hypot(delta_u, delta_v)),
        "error_m": error_m,
        "tolerance_m": float(tolerance_m),
    }


def _matrix(item, key):
    result = np.asarray(item[key], dtype=np.float64)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValidationError(f"Invalid transform {key}")
    return result


def validate_state_dir(
    state_dir: Path,
    pathfinder=None,
    tolerance: float = 1e-5,
    minimum_separation_m: float = 0.0,
    height_max_error_m: float = DEFAULT_HEIGHT_VALIDATION_MAX_ERROR_M,
    support_contact_tolerance_m: float = DEFAULT_SUPPORT_CONTACT_TOLERANCE_M,
    expected_camera_forward_offset_m: Optional[float] = None,
) -> List[str]:
    state_dir = Path(state_dir)
    errors: List[str] = []
    try:
        metadata = read_json(state_dir / "state.json")
    except Exception as exc:
        return [f"cannot read state.json: {exc}"]
    bev = metadata["bev"]
    mapping = BevMapping(
        bev["x_min"], bev["x_max"], bev["z_min"], bev["z_max"], bev["width"], bev["height"]
    )
    height_validation = bev.get("height_depth_validation", {})
    if not height_validation.get("available"):
        errors.append("BEV height validation is unavailable")
    else:
        max_error = height_validation.get("max_abs_error_m")
        if max_error is None or not math.isfinite(float(max_error)):
            errors.append("BEV height validation has no finite maximum error")
        elif float(max_error) > height_max_error_m:
            errors.append(
                f"BEV height ray error {float(max_error):.6f} m exceeds "
                f"{height_max_error_m:.6f} m"
            )
    ground_support = metadata.get("geometry_validation", {}).get(
        "robot_ground_support", {}
    )
    if set(ground_support) != {
        robot["robot_id"] for robot in metadata["robots"]
    }:
        errors.append("robot ground-support reports are missing or incomplete")
    else:
        for robot_id, report in ground_support.items():
            if abs(float(report.get(
                "support_gap_m", math.inf
            ))) > support_contact_tolerance_m:
                errors.append(f"{robot_id}: visual proxy is not grounded")
            if abs(float(report.get(
                "proxy_origin_offset_from_base_m", math.inf
            ))) > 1e-4:
                errors.append(f"{robot_id}: proxy origin is not aligned with base")
    for robot in metadata["robots"]:
        robot_id = robot["robot_id"]
        base = np.asarray(robot["base_position_world"], dtype=np.float64)
        camera = np.asarray(robot["camera_position_world"], dtype=np.float64)
        if not math.isclose(camera[1] - base[1], robot["camera_height_m"], abs_tol=tolerance):
            errors.append(f"{robot_id}: camera height mismatch")
        camera_offset = float(robot.get("camera_forward_offset_m", 0.0))
        if expected_camera_forward_offset_m is not None and not math.isclose(
            camera_offset, expected_camera_forward_offset_m, abs_tol=tolerance
        ):
            errors.append(f"{robot_id}: camera forward offset/config mismatch")
        forward = np.asarray(robot["forward_world"], dtype=np.float64)
        expected_camera = (
            base
            + np.array([0.0, float(robot["camera_height_m"]), 0.0])
            + forward * camera_offset
        )
        if not np.allclose(camera, expected_camera, atol=tolerance):
            errors.append(f"{robot_id}: camera mount position mismatch")
        proxy_match = re.search(
            r"_h(\d{3})\.object_config\.json$",
            robot.get("proxy_asset_handle", ""),
        )
        if proxy_match is not None:
            proxy_height = int(proxy_match.group(1)) / 100.0
            if not math.isclose(
                proxy_height, float(robot["camera_height_m"]), abs_tol=1e-9
            ):
                errors.append(f"{robot_id}: proxy mast/camera height mismatch")
        robot_world = _matrix(robot, "T_world_from_robot")
        habitat_world = _matrix(robot, "T_world_from_camera_habitat")
        habitat_inverse = _matrix(robot, "T_camera_habitat_from_world")
        cv_world = _matrix(robot, "T_world_from_camera_cv")
        cv_inverse = _matrix(robot, "T_camera_cv_from_world")
        if not np.allclose(habitat_inverse @ habitat_world, np.eye(4), atol=tolerance):
            errors.append(f"{robot_id}: Habitat camera transforms are not inverses")
        if not np.allclose(cv_inverse @ cv_world, np.eye(4), atol=tolerance):
            errors.append(f"{robot_id}: OpenCV camera transforms are not inverses")
        if not np.allclose(robot_world[:3, 3], base, atol=tolerance):
            errors.append(f"{robot_id}: robot transform translation mismatch")
        if not np.allclose(habitat_world[:3, 3], camera, atol=tolerance):
            errors.append(f"{robot_id}: camera transform translation mismatch")
        expected_forward = forward_from_quaternion(yaw_to_quaternion_xyzw(robot["yaw_rad"]))
        if not np.allclose(expected_forward, robot["forward_world"], atol=tolerance):
            errors.append(f"{robot_id}: stored forward vector mismatch")
        cv_forward_world = cv_world[:3, :3] @ np.array([0.0, 0.0, 1.0])
        if not np.allclose(cv_forward_world, expected_forward, atol=tolerance):
            errors.append(f"{robot_id}: OpenCV forward basis mismatch")
        u, v = mapping.world_to_bev(base[0], base[2])
        if not (-1 <= u <= mapping.width and -1 <= v <= mapping.height):
            errors.append(f"{robot_id}: projected BEV point outside map")
        if pathfinder is not None and not pathfinder.is_navigable(base):
            errors.append(f"{robot_id}: base is not navigable")
        intrinsics = robot["camera_intrinsics"]
        expected_shape = (intrinsics["height"], intrinsics["width"])
        files = robot["files"]
        try:
            rgb = np.asarray(Image.open(state_dir / files["rgb"]))
            depth = load_numeric(state_dir / files["depth"])
            if rgb.shape[:2] != expected_shape or rgb.dtype != np.uint8:
                errors.append(f"{robot_id}: invalid RGB shape/dtype {rgb.shape}/{rgb.dtype}")
            if depth.shape != expected_shape or depth.dtype != np.float32:
                errors.append(f"{robot_id}: invalid depth shape/dtype {depth.shape}/{depth.dtype}")
            if np.isnan(depth).any():
                errors.append(f"{robot_id}: depth contains NaN")
            if "instance" in files:
                instance = load_numeric(state_dir / files["instance"])
                if instance.shape != expected_shape:
                    errors.append(f"{robot_id}: invalid instance shape")
            if "semantic" in files:
                semantic = load_numeric(state_dir / files["semantic"])
                if semantic.shape != expected_shape or semantic.dtype != np.uint16:
                    errors.append(
                        f"{robot_id}: invalid semantic shape/dtype "
                        f"{semantic.shape}/{semantic.dtype}"
                    )
        except Exception as exc:
            errors.append(f"{robot_id}: referenced observation failed: {exc}")
    for i, first in enumerate(metadata["robots"]):
        for second in metadata["robots"][i + 1 :]:
            distance = np.linalg.norm(
                np.asarray(first["base_position_world"])[[0, 2]]
                - np.asarray(second["base_position_world"])[[0, 2]]
            )
            if distance < minimum_separation_m:
                errors.append(f"robot separation {distance:.3f} m is below {minimum_separation_m:.3f} m")
    bev_arrays={}
    for key,expected_dtype in (
        ("height",np.float32),("occupancy",np.uint8),("region_mask",np.uint8),
        ("navigable_region_mask",np.uint8),
    ):
        try:
            array=load_numeric(state_dir/bev["files"][key])
            bev_arrays[key]=array
            if array.shape!=(mapping.height,mapping.width) or array.dtype!=expected_dtype:
                errors.append(f"BEV {key}: invalid shape/dtype {array.shape}/{array.dtype}")
        except Exception as exc:
            errors.append(f"BEV {key}: {exc}")
    region_mask_array=bev_arrays.get("region_mask")
    occupancy_array=bev_arrays.get("occupancy")
    navigable_region_array=bev_arrays.get("navigable_region_mask")
    if occupancy_array is not None:
        values=set(map(int,np.unique(occupancy_array)))
        if not values.issubset({0,1}) or 1 not in values:
            errors.append("BEV occupancy must be a nonempty binary mask")
    if navigable_region_array is not None:
        values=set(map(int,np.unique(navigable_region_array)))
        if not values.issubset({0,1}) or 1 not in values:
            errors.append("BEV navigable_region_mask must be a nonempty binary mask")
        if occupancy_array is not None and region_mask_array is not None:
            expected=((occupancy_array==1)&(region_mask_array==1)).astype(np.uint8)
            if not np.array_equal(navigable_region_array,expected):
                errors.append("BEV navigable_region_mask is inconsistent")
    if region_mask_array is not None:
        values=set(map(int,np.unique(region_mask_array)))
        if not values.issubset({0,1}) or 1 not in values:
            errors.append("BEV region_mask must be a nonempty binary mask")
        for robot in metadata["robots"]:
            u,v=mapping.world_to_bev(
                robot["base_position_world"][0],robot["base_position_world"][2]
            )
            row=int(round(v));col=int(round(u))
            if not (0<=row<mapping.height and 0<=col<mapping.width):
                errors.append(f"{robot['robot_id']}: region-mask projection outside BEV")
            elif int(region_mask_array[row,col])!=1:
                errors.append(f"{robot['robot_id']}: base is outside registered region_mask")
            if occupancy_array is not None and 0<=row<mapping.height and 0<=col<mapping.width and int(occupancy_array[row,col])==0:
                errors.append(f"{robot['robot_id']}: base is outside registered occupancy")
    if metadata.get("bev_scope")!="semantic_region" or bev.get("bev_scope")!="semantic_region":
        errors.append("state/BEV scope is not semantic_region")
    if not metadata.get("region_id") or metadata.get("region_id")!=bev.get("region_id"):
        errors.append("state/BEV region_id mismatch")
    if "semantic" in bev["files"]:
        try:
            bev_semantic = load_numeric(state_dir / bev["files"]["semantic"])
            if (
                bev_semantic.shape != (mapping.height, mapping.width)
                or bev_semantic.dtype != np.uint16
            ):
                errors.append("BEV semantic: invalid shape/dtype")
            if bev.get("semantic_encoding") != "controlled_entity_category_id":
                errors.append("BEV semantic encoding is not declared")
            allowed = {0, *map(int, bev.get("semantic_category_ids", {}).values())}
            unexpected = set(map(int, np.unique(bev_semantic))) - allowed
            if unexpected:
                errors.append(f"BEV semantic: unknown category IDs {sorted(unexpected)}")
        except Exception as exc:
            errors.append(f"BEV semantic: {exc}")
    if "instance" in bev["files"]:
        try:
            bev_instance = load_numeric(state_dir / bev["files"]["instance"])
            if bev_instance.shape != (mapping.height, mapping.width):
                errors.append("BEV instance: invalid shape")
            if bev.get("instance_id_encoding") != "Habitat SemanticSensorTarget.OBJECT_ID":
                errors.append("BEV instance channel is not declared as Habitat OBJECT_ID")
            entity_object_ids = bev.get("entity_object_ids", {})
            for robot in metadata["robots"]:
                u, v = mapping.world_to_bev(
                    robot["base_position_world"][0],
                    robot["base_position_world"][2],
                )
                object_id = entity_object_ids.get(robot["robot_id"])
                if object_id is None:
                    errors.append(f"{robot['robot_id']}: missing runtime OBJECT_ID mapping")
                    continue
                pixels = np.argwhere(bev_instance == int(object_id))
                if not len(pixels):
                    errors.append(
                        f"{robot['robot_id']}: robot proxy is absent from BEV OBJECT_ID mask"
                    )
                    continue
                registration = instance_centroid_registration(pixels, u, v, mapping)
                if registration["error_m"] > registration["tolerance_m"]:
                    errors.append(
                        f"{robot['robot_id']}: BEV proxy centroid is misregistered by "
                        f"{registration['error_m']:.3f} m / "
                        f"{registration['error_pixels']:.2f} px "
                        f"(limit {registration['tolerance_m']:.3f} m)"
                    )
        except Exception as exc:
            errors.append(f"BEV instance: {exc}")
    objects_path = state_dir / metadata["objects_path"]
    try:
        objects = read_json(objects_path)
        object_categories=[obj.get("category") for obj in objects]
        if len(object_categories)!=len(set(object_categories)):
            errors.append("WorldState has duplicate controlled-object categories")
        for obj in objects:
            if obj["active"] and obj.get("bbox"):
                low = np.asarray(obj["bbox"]["min_world"], dtype=np.float64)
                high = np.asarray(obj["bbox"]["max_world"], dtype=np.float64)
                if (
                    low.shape != (3,)
                    or high.shape != (3,)
                    or not np.all(np.isfinite(low))
                    or not np.all(np.isfinite(high))
                    or np.any(low > high)
                ):
                    errors.append(f"{obj['instance_id']}: invalid world-space bbox")
    except Exception as exc:
        errors.append(f"objects metadata: {exc}")
    return errors


def validate_dataset_manifest(root: Path, config=None) -> List[str]:
    root = Path(root)
    errors = []
    try:
        dataset = read_json(root / "dataset.json")
    except Exception as exc:
        return [f"cannot read dataset.json: {exc}"]
    disk_states = {
        str(path.parent.relative_to(root))
        for path in root.glob("scenes/*/floors/*/regions/*/states/*/state.json")
    }
    disk_edits = {
        str(path.relative_to(root))
        for path in root.glob("interventions/*/*/*/edit_*.json")
    }
    indexed_states = set(dataset.get("states", []))
    indexed_edits = set(dataset.get("interventions", []))
    if indexed_states != disk_states:
        errors.append("dataset.json state index does not match published state directories")
    if indexed_edits != disk_edits:
        errors.append("dataset.json intervention index does not match published edits")
    if config is not None:
        fingerprint = dataset.get("generation_fingerprint")
        if fingerprint and fingerprint != config.generation_fingerprint():
            errors.append("generation config fingerprint mismatch")
        object_preflight_path = root / "approved_object_preflight_report.json"
        if not object_preflight_path.is_file():
            errors.append("missing approved-object Habitat preflight report")
        elif not read_json(object_preflight_path).get("passed"):
            errors.append("approved-object Habitat preflight did not pass")
        else:
            object_preflight = read_json(object_preflight_path)
            if object_preflight.get("generation_fingerprint") != config.generation_fingerprint():
                errors.append("approved-object preflight generation fingerprint mismatch")
            if object_preflight.get("controlled_object_registry_sha256") != sha256_file(
                config.controlled_object_registry_path
            ):
                errors.append("approved-object preflight registry fingerprint mismatch")
        if config.run_multilevel_calibration_preflight:
            calibration_path = root / "calibration_report.json"
            if not calibration_path.exists():
                errors.append("missing multi-height orthographic calibration report")
            else:
                calibration = read_json(calibration_path)
                if not calibration.get("passed"):
                    errors.append("multi-height orthographic calibration did not pass")

    split_state_sets = []
    split_after_sets = []
    split_edit_sets = []
    for split in ("train", "val", "test"):
        path = root / f"splits/{split}.json"
        if not path.exists():
            errors.append(f"missing split manifest: {path.relative_to(root)}")
            continue
        payload = read_json(path)
        split_state_sets.append(set(payload.get("states", [])))
        split_after_sets.append(set(payload.get("after_states", [])))
        split_edit_sets.append(set(payload.get("interventions", [])))
        if config is not None:
            expected_scenes = set(config.scene_splits[split])
            actual_scenes = set(payload.get("scenes", []))
            if not actual_scenes.issubset(expected_scenes):
                errors.append(f"{split} contains an HSSD scene assigned elsewhere")
        for regime, regime_payload in payload.get("level2_by_regime", {}).items():
            regime_path = root / f"splits/level2_{split}_{regime}.json"
            if not regime_path.exists():
                errors.append(f"missing regime manifest: {regime_path.relative_to(root)}")
                continue
            standalone = read_json(regime_path)
            if set(standalone.get("interventions", [])) != set(
                regime_payload.get("interventions", [])
            ):
                errors.append(f"{split}/{regime} standalone manifest disagrees with split")

    for name, collections in (
        ("factual states", split_state_sets),
        ("after states", split_after_sets),
        ("interventions", split_edit_sets),
    ):
        for index, first in enumerate(collections):
            for second in collections[index + 1 :]:
                if first & second:
                    errors.append(f"{name} leak across train/val/test")
    if split_state_sets and set().union(*split_state_sets) != set(
        dataset.get("factual_states", [])
    ):
        errors.append("Level-1 split union does not equal factual_states")
    if split_after_sets and set().union(*split_after_sets) != set(
        dataset.get("intervention_derived_states", [])
    ):
        errors.append("Level-2 after-state split union does not equal derived states")
    if split_edit_sets and set().union(*split_edit_sets) != indexed_edits:
        errors.append("Level-2 split union does not equal interventions")
    return errors


def _regime_parameter_value(intervention) -> tuple:
    parameters = intervention.parameters
    if intervention.type == "robot_translate":
        return "robot_translate_m", abs(float(parameters["forward_m"]))
    if intervention.type == "robot_rotate":
        return "robot_rotate_deg", float(math.degrees(parameters["delta_yaw_rad"]))
    if intervention.type == "object_place_relative":
        return "object_place_relative_m", abs(float(parameters["distance_m"]))
    if intervention.type == "object_translate":
        displacement = np.asarray(parameters["displacement_m"], dtype=np.float64)
        return "object_translate_m", float(np.linalg.norm(displacement[[0, 2]]))
    return None, None


def validate_dataset(root: Path, config=None) -> Dict[str, object]:
    root = Path(root)
    state_paths = sorted(root.glob("scenes/*/floors/*/regions/*/states/*/state.json"))
    report: Dict[str, object] = {"root": str(root), "states_checked": len(state_paths), "errors": {}}
    manifest_errors = validate_dataset_manifest(root, config)
    if manifest_errors:
        report["errors"]["_manifest"] = manifest_errors
    active_backend = None
    active_target = None

    def backend_for_target(scene_id:str,floor_id:str,region_id:str):
        nonlocal active_backend, active_target
        if config is None:
            return None
        target=(scene_id,floor_id,region_id)
        if active_target != target:
            if active_backend is not None:
                active_backend.close()
            from .habitat_backend import HabitatBackend
            scene = config.registry().scene(scene_id)
            floor=scene.floor(floor_id)
            active_backend=HabitatBackend(config,scene,floor,floor.region(region_id))
            active_target = target
        return active_backend

    try:
        for state_json in state_paths:
            metadata = read_json(state_json)
            scene_id = metadata["scene_id"]
            floor_id=metadata["floor_id"]
            region_id=metadata["region_id"]
            if metadata.get("dataset_source") != "hssd":
                report["errors"][str(state_json.parent.relative_to(root))] = [
                    "state dataset_source is not hssd"
                ]
                continue
            backend=backend_for_target(scene_id,floor_id,region_id)
            pathfinder = backend.sim.pathfinder if backend is not None else None
            errors = validate_state_dir(
                state_json.parent,
                pathfinder,
                minimum_separation_m=config.min_inter_robot_distance_m if config else 0.0,
                height_max_error_m=(
                    config.height_validation_max_error_m
                    if config else DEFAULT_HEIGHT_VALIDATION_MAX_ERROR_M
                ),
                support_contact_tolerance_m=(
                    config.support_contact_tolerance_m
                    if config else DEFAULT_SUPPORT_CONTACT_TOLERANCE_M
                ),
                expected_camera_forward_offset_m=(
                    config.robot_camera_forward_offset_m
                    if config else None
                ),
            )
            if config is not None:
                from .state_io import load_world_state
                world_state = load_world_state(state_json.parent)
                backend=backend_for_target(scene_id,floor_id,region_id)
                if world_state.region_id!=backend.region_id: errors.append("WorldState region_id does not match registry")
                for robot in world_state.robots:
                    if not backend.point_in_region(robot.base_position_world): errors.append(f"{robot.robot_id}: outside selected semantic region")
                active_objects = [obj for obj in world_state.objects if obj.active]
                identity_by_object = {
                    obj.instance_id: validate_controlled_object_identity(
                        obj,
                        config.controlled_object_pools,
                        config.semantic_category_ids,
                    )
                    for obj in active_objects
                }
                runtime_asset_identity_valid = not any(identity_by_object.values())
                membership_by_object = {}
                for obj in active_objects:
                    identity_errors = identity_by_object[obj.instance_id]
                    if identity_errors:
                        errors.extend(
                            f"CRITICAL {obj.instance_id}: {message}"
                            for message in identity_errors
                        )
                    membership = backend.object_region_membership(
                        obj.position_world, world_state.floor_y
                    )
                    membership_by_object[obj.instance_id] = membership
                    if not membership["passed"]:
                        errors.append(
                            f"CRITICAL {obj.instance_id}: outside selected semantic region: "
                            + "; ".join(membership["reasons"])
                        )
                runtime_geometry_valid = (
                    runtime_asset_identity_valid
                    and all(item["passed"] for item in membership_by_object.values())
                )
                if runtime_geometry_valid:
                    for obj in active_objects:
                        collision = backend.object_collision_report(
                            world_state, obj.instance_id
                        )
                        if not collision["collision_free"]:
                            errors.append(
                                f"{obj.instance_id}: static-scene collision "
                                f"{collision['rejected_contacts']}"
                            )
                        physical_floor_y = membership_by_object[
                            obj.instance_id
                        ].get("physical_floor_y")
                        if physical_floor_y is None:
                            continue
                        if not obj.bbox:
                            continue
                        bottom = float(obj.bbox["min_world"][1])
                        support_error = abs(bottom - physical_floor_y)
                        if support_error > max(
                            0.01, float(config.support_contact_tolerance_m)
                        ):
                            errors.append(
                                f"{obj.instance_id}: physical support error "
                                f"{support_error:.4f} m"
                            )
                    for robot in world_state.robots:
                        collision = backend.entity_collision_report(
                            world_state, robot.robot_id
                        )
                        if not collision["collision_free"]:
                            errors.append(
                                f"{robot.robot_id}: proxy collision "
                                f"{collision['rejected_contacts']}"
                            )
            if config is not None:
                if runtime_geometry_valid:
                    for robot in world_state.robots:
                        support = backend.robot_support_report(
                            world_state, robot.robot_id
                        )
                        if abs(float(support["support_gap_m"])) > float(
                            config.support_contact_tolerance_m
                        ):
                            errors.append(
                                f"{robot.robot_id}: physical support gap "
                                f"{support['support_gap_m']:.6f} m"
                            )
                        if abs(float(
                            support["proxy_origin_offset_from_base_m"]
                        )) > 1e-4:
                            errors.append(
                                f"{robot.robot_id}: proxy/base origin offset "
                                f"{support['proxy_origin_offset_from_base_m']:.6f} m"
                            )
            if errors:
                report["errors"][str(state_json.parent.relative_to(root))] = errors
        for edit_path in sorted(root.glob("interventions/*/*/*/edit_*.json")):
            edit = read_json(edit_path)
            relative_edit = str(edit_path.relative_to(root))
            edit_errors = []
            try:
                from .interventions import (
                    Intervention,
                    apply_intervention,
                    canonical_instruction,
                    validate_robot_translation,
                )
                from .state_io import load_world_state

                before = load_world_state(root / edit["before_state_path"])
                after = load_world_state(root / edit["after_state_path"])
                intervention = Intervention.from_dict(edit["structured_intervention"])
                expected_instruction = canonical_instruction(intervention, before)
                if config is not None:
                    expected_split = config.scene_split(edit["scene_id"])
                    regime = edit.get("benchmark_regime")
                    if edit.get("split") != expected_split:
                        edit_errors.append("edit split does not match scene-level split")
                    if regime not in config.level2_regimes_by_split[expected_split]:
                        edit_errors.append("edit regime is not allowed for its split")
                    else:
                        domain_key, value = _regime_parameter_value(intervention)
                        if domain_key is not None and not any(
                            math.isclose(value, float(candidate), abs_tol=1e-8)
                            for candidate in config.intervention_regimes[regime][domain_key]
                        ):
                            edit_errors.append(
                                f"intervention value {value} is outside {regime}/{domain_key}"
                            )
                    if int(edit.get("target_visible_observers_before", -1)) < int(
                        config.min_target_visible_observers
                    ):
                        edit_errors.append(
                            "intervention target does not meet benchmark visibility gate"
                        )
                if edit.get("instruction") != expected_instruction:
                    edit_errors.append(
                        "canonical instruction is not equivalent to structured intervention"
                    )

                before_target=(
                    before.robot(intervention.target_id)
                    if intervention.target_id.startswith("robot_")
                    else before.object(intervention.target_id)
                )
                after_target=(
                    after.robot(intervention.target_id)
                    if intervention.target_id.startswith("robot_")
                    else after.object(intervention.target_id)
                )
                minimum=int(config.min_target_visible_observers) if config is not None else 1
                before_views=benchmark_visible_observers(before_target)
                after_views=benchmark_visible_observers(after_target)
                if before_views<minimum:
                    edit_errors.append("intervention target is not visible before the edit")
                if intervention.type=="object_remove":
                    if after_target.active or after_views!=0:
                        edit_errors.append("removed object is active or visible after the edit")
                elif after_views<minimum:
                    edit_errors.append("intervention target is not visible after the edit")
                declared=edit.get("observable_edit",{})
                if declared.get("before_visible_views")!=before_views or declared.get("after_visible_views")!=after_views:
                    edit_errors.append("observable_edit counts do not match state visibility")

                if intervention.type == "robot_translate":
                    if config is not None:
                        backend = backend_for_target(edit["scene_id"],edit["floor_id"],edit["region_id"])
                        validate_robot_translation(
                            before,
                            after,
                            intervention,
                            backend.sim.pathfinder,
                            config.floor_tolerance_m,
                            config.min_inter_robot_distance_m,
                            config.controlled_object_min_separation_m,
                        )
                        robot = after.robot(intervention.target_id)
                        nav_target = np.asarray(
                            backend.sim.pathfinder.snap_point(
                                robot.base_position_world
                            ),
                            dtype=np.float64,
                        )
                        physical_floor_y = backend.floor_surface_y(nav_target)
                        if abs(
                            float(robot.base_position_world[1]) - physical_floor_y
                        ) > 0.01:
                            edit_errors.append(
                                "robot target Y is not supported by the physical floor"
                            )
                    else:
                        old = np.asarray(
                            before.robot(intervention.target_id).base_position_world
                        )
                        new = np.asarray(
                            after.robot(intervention.target_id).base_position_world
                        )
                        actual = float(np.linalg.norm((new - old)[[0, 2]]))
                        expected = float(intervention.parameters["forward_m"])
                        if not math.isclose(actual, expected, abs_tol=1e-6):
                            edit_errors.append(
                                f"horizontal robot translation is {actual}, "
                                f"requested {expected}"
                            )
                elif intervention.type == "object_translate":
                    requested = np.asarray(
                        intervention.parameters["displacement_m"],
                        dtype=np.float64,
                    )
                    if requested.shape != (3,) or abs(float(requested[1])) > 1e-9:
                        edit_errors.append(
                            "floor-supported object_translate requires Y displacement 0"
                        )
                    expected_after = apply_intervention(before, intervention)
                    expected_position = np.asarray(
                        expected_after.object(intervention.target_id).position_world
                    )
                    actual_position = np.asarray(
                        after.object(intervention.target_id).position_world
                    )
                    if not np.allclose(
                        actual_position[[0, 2]],
                        expected_position[[0, 2]],
                        atol=1e-6,
                    ):
                        edit_errors.append("object horizontal displacement mismatch")
            except (KeyError, RuntimeError, ValueError) as exc:
                edit_errors.append(str(exc))
            if edit_errors:
                report["errors"].setdefault(relative_edit, []).extend(edit_errors)
    finally:
        if active_backend is not None:
            active_backend.close()
    report["passed"] = not report["errors"] and bool(state_paths)
    return report
