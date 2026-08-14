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


def canonical_instruction(edit: Intervention, state: WorldState) -> str:
    number = int(edit.target_id.split("_")[-1]) if edit.target_id.startswith("robot_") else None
    p = edit.parameters
    if edit.type == "robot_translate":
        return f"Robot {number} moves forward by {float(p['forward_m']):g} meter{'s' if float(p['forward_m']) != 1 else ''}."
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
        return f"Move the {obj.category} {float(p['distance_m']):g} meter{'s' if float(p['distance_m']) != 1 else ''} in front of Robot {reference}."
    if edit.type == "object_translate":
        return f"Move the {obj.category} by the specified metric displacement."
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


def validate_robot_translation(before: WorldState, after: WorldState, edit: Intervention, pathfinder, floor_tolerance_m: float, min_separation_m: float) -> None:
    robot_before = before.robot(edit.target_id)
    robot_after = after.robot(edit.target_id)
    old = np.asarray(robot_before.base_position_world)
    new = np.asarray(robot_after.base_position_world)
    requested = abs(float(edit.parameters["forward_m"]))
    actual = float(np.linalg.norm(new - old))
    if not math.isclose(actual, requested, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Metric edit changed from {requested} m to {actual} m")
    if not pathfinder.is_navigable(new):
        raise ValueError("Exact robot target is not navigable; intervention rejected")
    if abs(float(new[1] - old[1])) > floor_tolerance_m:
        raise ValueError("Robot intervention changed floor")
    for other in after.robots:
        if other.robot_id == edit.target_id:
            continue
        separation = np.linalg.norm(new[[0, 2]] - np.asarray(other.base_position_world)[[0, 2]])
        if separation < min_separation_m:
            raise ValueError("Robot intervention violates minimum separation")
