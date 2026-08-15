from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .bev import BevMapping
from .coordinates import forward_from_quaternion, yaw_to_quaternion_xyzw
from .state_io import read_json


class ValidationError(RuntimeError):
    pass


BEV_REGISTRATION_MAX_ERROR_PIXELS = 3.0
BEV_REGISTRATION_MIN_TOLERANCE_M = 0.02
DEFAULT_HEIGHT_VALIDATION_MAX_ERROR_M = 0.02


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
                backend = backends[scene_id]
                for obj in world_state.objects:
                    if not obj.active:
                        continue
                    collision = backend.object_collision_report(
                        world_state, obj.instance_id
                    )
                    if not collision["collision_free"]:
                        errors.append(
                            f"{obj.instance_id}: static-scene collision "
                            f"{collision['rejected_contacts']}"
                        )
                    floor_point = np.asarray(
                        backend.sim.pathfinder.snap_point(obj.position_world),
                        dtype=np.float64,
                    )
                    physical_floor_y = backend.floor_surface_y(floor_point)
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
            if errors:
                report["errors"][str(state_json.parent.relative_to(root))] = errors
        for edit_path in sorted(root.glob("interventions/*/edit_*.json")):
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
                if edit.get("instruction") != expected_instruction:
                    edit_errors.append(
                        "canonical instruction is not equivalent to structured intervention"
                    )

                if intervention.type == "robot_translate":
                    if config is not None:
                        backend = backends[edit["scene_id"]]
                        validate_robot_translation(
                            before,
                            after,
                            intervention,
                            backend.sim.pathfinder,
                            config.floor_tolerance_m,
                            config.min_inter_robot_distance_m,
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
        for backend in backends.values():
            backend.close()
    report["passed"] = not report["errors"] and bool(state_paths)
    return report
