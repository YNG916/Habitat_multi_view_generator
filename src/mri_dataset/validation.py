from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .bev import BevMapping, ROBOT_COLORS
from .coordinates import forward_from_quaternion, yaw_to_quaternion_xyzw
from .state_io import read_json


class ValidationError(RuntimeError):
    pass


ROBOT_PROXY_REGISTRATION_TOLERANCE_M = 0.35
DEFAULT_HEIGHT_VALIDATION_MAX_ERROR_M = 0.02


def nearest_instance_distance_m(pixels_rc, u, v, mapping):
    pixels = np.asarray(pixels_rc, dtype=np.float64)
    if not len(pixels):
        return None
    dx = (pixels[:, 1] - float(u)) * mapping.meters_per_pixel_x
    dz = (pixels[:, 0] - float(v)) * mapping.meters_per_pixel_z
    return float(np.min(np.hypot(dx, dz)))


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
    for robot in metadata["robots"]:
        robot_id = robot["robot_id"]
        base = np.asarray(robot["base_position_world"], dtype=np.float64)
        camera = np.asarray(robot["camera_position_world"], dtype=np.float64)
        if not math.isclose(camera[1] - base[1], robot["camera_height_m"], abs_tol=tolerance):
            errors.append(f"{robot_id}: camera height mismatch")
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
            depth = np.load(state_dir / files["depth"])
            if rgb.shape[:2] != expected_shape or rgb.dtype != np.uint8:
                errors.append(f"{robot_id}: invalid RGB shape/dtype {rgb.shape}/{rgb.dtype}")
            if depth.shape != expected_shape or depth.dtype != np.float32:
                errors.append(f"{robot_id}: invalid depth shape/dtype {depth.shape}/{depth.dtype}")
            if np.isnan(depth).any():
                errors.append(f"{robot_id}: depth contains NaN")
            if "instance" in files:
                instance = np.load(state_dir / files["instance"])
                if instance.shape != expected_shape:
                    errors.append(f"{robot_id}: invalid instance shape")
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
    for key, expected_dtype in [("height", np.float32), ("occupancy", np.uint8)]:
        try:
            array = np.load(state_dir / bev["files"][key])
            if array.shape != (mapping.height, mapping.width) or array.dtype != expected_dtype:
                errors.append(f"BEV {key}: invalid shape/dtype {array.shape}/{array.dtype}")
        except Exception as exc:
            errors.append(f"BEV {key}: {exc}")
    if "instance" in bev["files"]:
        try:
            bev_instance = np.load(state_dir / bev["files"]["instance"])
            annotated = np.asarray(Image.open(state_dir / bev["files"]["annotated"]).convert("RGB"))
            if bev_instance.shape != (mapping.height, mapping.width):
                errors.append("BEV instance: invalid shape")
            if bev.get("instance_id_encoding") != "Habitat SemanticSensorTarget.OBJECT_ID":
                errors.append("BEV instance channel is not declared as Habitat OBJECT_ID")
            entity_object_ids = bev.get("entity_object_ids", {})
            for index, robot in enumerate(metadata["robots"]):
                u, v = mapping.world_to_bev(robot["base_position_world"][0], robot["base_position_world"][2])
                object_id = entity_object_ids.get(robot["robot_id"])
                if object_id is None:
                    errors.append(f"{robot['robot_id']}: missing runtime OBJECT_ID mapping")
                    continue
                pixels = np.argwhere(bev_instance == int(object_id))
                if len(pixels):
                    nearest_m = nearest_instance_distance_m(pixels, u, v, mapping)
                    if nearest_m > ROBOT_PROXY_REGISTRATION_TOLERANCE_M:
                        errors.append(
                            f"{robot['robot_id']}: BEV proxy is misregistered by {nearest_m:.3f} m"
                        )
                else:
                    col, row = int(round(u)), int(round(v))
                    patch = annotated[max(0, row-3):row+4, max(0, col-3):col+4]
                    expected = np.asarray(ROBOT_COLORS[index % len(ROBOT_COLORS)])
                    if not np.any(np.all(patch == expected, axis=-1)):
                        errors.append(f"{robot['robot_id']}: neither proxy nor annotation is registered to BEV")
        except Exception as exc:
            errors.append(f"BEV instance: {exc}")
    objects_path = state_dir / metadata["objects_path"]
    try:
        objects = read_json(objects_path)
        for obj in objects:
            if obj["active"] and obj.get("bbox"):
                bottom = float(obj["bbox"]["min_world"][1])
                if abs(bottom - metadata["floor_y"]) > 0.08:
                    errors.append(f"{obj['instance_id']}: unsupported/floor intersection ({bottom:.3f})")
    except Exception as exc:
        errors.append(f"objects metadata: {exc}")
    return errors


def validate_dataset(root: Path, config=None) -> Dict[str, object]:
    root = Path(root)
    state_paths = sorted(root.glob("scenes/*/states/*/state.json"))
    report: Dict[str, object] = {"root": str(root), "states_checked": len(state_paths), "errors": {}}
    backends = {}
    try:
        if config is not None:
            from .habitat_backend import HabitatBackend
            for path in state_paths:
                scene_id = path.parents[2].name
                if scene_id not in backends:
                    backends[scene_id] = HabitatBackend(config, scene_id)
        for state_json in state_paths:
            scene_id = state_json.parents[2].name
            pathfinder = backends[scene_id].sim.pathfinder if scene_id in backends else None
            errors = validate_state_dir(
                state_json.parent,
                pathfinder,
                minimum_separation_m=config.min_inter_robot_distance_m if config else 0.0,
                height_max_error_m=(
                    config.height_validation_max_error_m
                    if config else DEFAULT_HEIGHT_VALIDATION_MAX_ERROR_M
                ),
            )
            if config is not None:
                from .state_io import load_world_state
                world_state = load_world_state(state_json.parent)
                for obj in world_state.objects:
                    if not obj.active:
                        continue
                    collision = backends[scene_id].object_collision_report(
                        world_state, obj.instance_id
                    )
                    if not collision["collision_free"]:
                        errors.append(
                            f"{obj.instance_id}: static-scene collision "
                            f"{collision['rejected_contacts']}"
                        )
            if errors:
                report["errors"][str(state_json.parent.relative_to(root))] = errors
        for edit_path in sorted(root.glob("interventions/*/edit_*.json")):
            edit = read_json(edit_path)
            if edit["structured_intervention"]["type"] == "robot_translate":
                before = read_json(root / edit["before_state_path"] / "state.json")
                after = read_json(root / edit["after_state_path"] / "state.json")
                target = edit["structured_intervention"]["target_id"]
                b = next(r for r in before["robots"] if r["robot_id"] == target)
                a = next(r for r in after["robots"] if r["robot_id"] == target)
                actual = float(np.linalg.norm(np.asarray(a["base_position_world"]) - np.asarray(b["base_position_world"])))
                expected = abs(float(edit["structured_intervention"]["forward_m"]))
                if not math.isclose(actual, expected, abs_tol=1e-6):
                    report["errors"][str(edit_path.relative_to(root))] = [
                        f"robot translation is {actual}, requested {expected}"
                    ]
            elif edit["structured_intervention"]["type"] == "object_translate":
                before_objects = read_json(root / edit["before_state_path"] / "objects.json")
                after_objects = read_json(root / edit["after_state_path"] / "objects.json")
                target = edit["structured_intervention"]["target_id"]
                b = next(o for o in before_objects if o["instance_id"] == target)
                a = next(o for o in after_objects if o["instance_id"] == target)
                actual = np.asarray(a["position_world"]) - np.asarray(b["position_world"])
                expected = np.asarray(edit["structured_intervention"]["displacement_m"])
                if not np.allclose(actual, expected, atol=1e-6):
                    report["errors"][str(edit_path.relative_to(root))] = ["object displacement mismatch"]
    finally:
        for backend in backends.values():
            backend.close()
    report["passed"] = not report["errors"] and bool(state_paths)
    return report
