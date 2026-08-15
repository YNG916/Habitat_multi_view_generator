from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .coordinates import forward_from_quaternion, normalize_angle, yaw_to_quaternion_xyzw
from .world_state import WorldState


@dataclass
class Intervention:
    type: str
    target_id: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "target_id": self.target_id, **self.parameters}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Intervention":
        required = {"type", "target_id"}
        if not required.issubset(data):
            raise ValueError(f"Intervention is missing {sorted(required - set(data))}")
        return cls(data["type"], data["target_id"], {k: v for k, v in data.items() if k not in required})


def _meters(value: float) -> str:
    return f"{value:g} meter{'s' if not math.isclose(abs(value), 1.0) else ''}"


def _object_translation_instruction(obj_category: str, parameters: Dict[str, Any]) -> str:
    displacement = np.asarray(parameters["displacement_m"], dtype=np.float64)
    if displacement.shape != (3,) or not np.all(np.isfinite(displacement)):
        raise ValueError("displacement_m must contain three finite components")
    frame = parameters.get("reference_frame", "world")
    nonzero = np.flatnonzero(np.abs(displacement) > 1e-12)
    if len(nonzero) == 1:
        axis_index = int(nonzero[0])
        sign = "+" if displacement[axis_index] > 0 else "-"
        axis = "XYZ"[axis_index]
        magnitude = abs(float(displacement[axis_index]))
        if frame == "world":
            direction = f"the world {sign}{axis} direction"
        elif frame == "object_local":
            direction = f"the {obj_category}'s local {sign}{axis} direction"
        elif frame == "reference_robot":
            reference = int(str(parameters["reference_id"]).split("_")[-1])
            suffix = ""
            if axis == "Z":
                suffix = " (forward)" if sign == "-" else " (backward)"
            elif axis == "X":
                suffix = " (right)" if sign == "+" else " (left)"
            direction = f"Robot {reference}'s local {sign}{axis}{suffix} direction"
        else:
            raise ValueError(f"Unsupported object translation frame: {frame}")
        return f"Move the {obj_category} {_meters(magnitude)} along {direction}."

    vector = ", ".join(f"{float(value):g}" for value in displacement)
    if frame == "world":
        frame_description = "the world XYZ coordinate frame"
    elif frame == "object_local":
        frame_description = f"the {obj_category}'s local XYZ coordinate frame"
    elif frame == "reference_robot":
        reference = int(str(parameters["reference_id"]).split("_")[-1])
        frame_description = f"Robot {reference}'s local XYZ coordinate frame"
    else:
        raise ValueError(f"Unsupported object translation frame: {frame}")
    return (
        f"Move the {obj_category} by displacement [{vector}] meters in "
        f"{frame_description}."
    )


def canonical_instruction(edit: Intervention, state: WorldState) -> str:
    number = int(edit.target_id.split("_")[-1]) if edit.target_id.startswith("robot_") else None
    p = edit.parameters
    if edit.type == "robot_translate":
        return f"Robot {number} moves forward by {_meters(float(p['forward_m']))}."
    if edit.type == "robot_rotate":
        degrees = abs(math.degrees(float(p["delta_yaw_rad"])))
        # Positive Habitat yaw rotates heading left in the XZ top-down convention.
        direction = "left" if float(p["delta_yaw_rad"]) > 0 else "right"
        return f"Robot {number} turns {direction} by {degrees:g} degrees."
    obj = state.object(edit.target_id)
    if edit.type == "object_remove":
        return f"Remove the {obj.category}."
    if edit.type == "object_place_relative":
        reference = int(str(p["reference_id"]).split("_")[-1])
        return (
            f"Move the {obj.category} {_meters(float(p['distance_m']))} "
            f"in front of Robot {reference}."
        )
    if edit.type == "object_translate":
        return _object_translation_instruction(obj.category, p)
    raise ValueError(f"No canonical instruction for {edit.type}")


def apply_intervention(before: WorldState, edit: Intervention, after_state_id: Optional[str] = None) -> WorldState:
    after = copy.deepcopy(before)
    after.parent_state_id = before.state_id
    after.state_id = after_state_id or f"{before.state_id}_after"
    after.intervention = edit.to_dict()
    p = edit.parameters
    if edit.type == "robot_translate":
        if p.get("reference_frame", "target_local") != "target_local":
            raise ValueError("robot_translate v0.1 requires reference_frame=target_local")
        robot = after.robot(edit.target_id)
        forward = forward_from_quaternion(yaw_to_quaternion_xyzw(robot.yaw_rad))
        position = np.asarray(robot.base_position_world, dtype=np.float64)
        position += float(p["forward_m"]) * forward
        robot.base_position_world = position.tolist()
        robot.synchronize_camera()
    elif edit.type == "robot_rotate":
        robot = after.robot(edit.target_id)
        robot.yaw_rad = normalize_angle(robot.yaw_rad + float(p["delta_yaw_rad"]))
        robot.synchronize_camera()
    elif edit.type == "object_remove":
        after.object(edit.target_id).active = False
    elif edit.type == "object_place_relative":
        if p.get("relation", "front") != "front":
            raise ValueError("Only relation=front is currently supported")
        obj = after.object(edit.target_id)
        robot = after.robot(str(p["reference_id"]))
        forward = forward_from_quaternion(yaw_to_quaternion_xyzw(robot.yaw_rad))
        old = np.asarray(obj.position_world, dtype=np.float64)
        target = np.asarray(robot.base_position_world, dtype=np.float64)
        target += float(p["distance_m"]) * forward
        # The Habitat backend resolves the target XZ's physical support Y
        # before rendering; keep the old Y only as an intermediate pure-state value.
        target[1] = obj.position_world[1]
        obj.position_world = target.tolist()
        _translate_bbox(obj, target - old)
    elif edit.type == "object_translate":
        obj = after.object(edit.target_id)
        displacement = np.asarray(p["displacement_m"], dtype=np.float64)
        frame = p.get("reference_frame", "world")
        if displacement.shape != (3,):
            raise ValueError("displacement_m must have three components")
        if frame == "object_local":
            from .coordinates import rotate_vector
            displacement = rotate_vector(obj.quaternion_world_xyzw, displacement)
        elif frame == "reference_robot":
            from .coordinates import rotate_vector
            displacement = rotate_vector(
                yaw_to_quaternion_xyzw(after.robot(str(p["reference_id"])).yaw_rad), displacement
            )
        elif frame != "world":
            raise ValueError(f"Unsupported object translation frame: {frame}")
        obj.position_world = (np.asarray(obj.position_world) + displacement).tolist()
        _translate_bbox(obj, displacement)
    else:
        raise ValueError(f"Unknown intervention type: {edit.type}")
    return after


def _translate_bbox(obj, displacement: np.ndarray) -> None:
    if not obj.bbox:
        return
    for key in ("min_world", "max_world"):
        obj.bbox[key] = (np.asarray(obj.bbox[key], dtype=np.float64) + displacement).tolist()


def validate_robot_translation(
    before: WorldState,
    after: WorldState,
    edit: Intervention,
    pathfinder,
    floor_tolerance_m: float,
    min_separation_m: float,
    endpoint_tolerance_m: float = 1e-3,
) -> None:
    robot_before = before.robot(edit.target_id)
    robot_after = after.robot(edit.target_id)
    old = np.asarray(robot_before.base_position_world, dtype=np.float64)
    new = np.asarray(robot_after.base_position_world, dtype=np.float64)
    requested = float(edit.parameters["forward_m"])
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("robot_translate forward_m must be a positive finite distance")

    expected_delta = requested * forward_from_quaternion(
        yaw_to_quaternion_xyzw(robot_before.yaw_rad)
    )
    actual_delta = new - old
    if not np.allclose(
        actual_delta[[0, 2]], expected_delta[[0, 2]], rtol=0.0, atol=1e-6
    ):
        raise ValueError("Robot translation is not the requested target-local forward vector")
    actual_horizontal = float(np.linalg.norm(actual_delta[[0, 2]]))
    if not math.isclose(actual_horizontal, requested, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Horizontal metric edit changed from {requested} m to "
            f"{actual_horizontal} m"
        )

    nav_start = np.asarray(pathfinder.snap_point(old), dtype=np.float64)
    nav_target = np.asarray(pathfinder.snap_point(new), dtype=np.float64)
    if not np.all(np.isfinite(nav_start)) or not np.all(np.isfinite(nav_target)):
        raise ValueError("Robot translation start or target is outside the NavMesh")
    if (
        np.linalg.norm(nav_target[[0, 2]] - new[[0, 2]]) > endpoint_tolerance_m
        or not pathfinder.is_navigable(nav_target)
    ):
        raise ValueError("Exact robot target is not navigable; intervention rejected")

    requested_nav_target = np.array(
        [new[0], nav_start[1], new[2]], dtype=np.float64
    )
    stepped = np.asarray(
        pathfinder.try_step_no_sliding(nav_start, requested_nav_target),
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(stepped))
        or np.linalg.norm(stepped[[0, 2]] - new[[0, 2]]) > endpoint_tolerance_m
    ):
        raise ValueError("Exact forward path is blocked on the NavMesh")
    if abs(float(nav_target[1] - nav_start[1])) > floor_tolerance_m:
        raise ValueError("Robot intervention changed floor")

    for other in after.robots:
        if other.robot_id == edit.target_id:
            continue
        separation = np.linalg.norm(
            new[[0, 2]] - np.asarray(other.base_position_world, dtype=np.float64)[[0, 2]]
        )
        if separation < min_separation_m:
            raise ValueError("Robot intervention violates minimum separation")
