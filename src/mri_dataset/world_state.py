from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

from .coordinates import (
    camera_intrinsics,
    camera_transforms,
    forward_from_quaternion,
    transform_matrix,
    yaw_to_quaternion_xyzw,
)


COORDINATE_CONVENTION = {
    "handedness": "right-handed",
    "world_up": "+Y",
    "ground_plane": "XZ",
    "robot_forward_at_yaw_zero": "-Z",
    "camera_local_forward": "-Z",
    "yaw_axis": "+Y",
    "yaw_positive": "right-hand-rule",
    "bev_up": "-Z",
    "bev_right": "+X",
    "distance_unit": "meter",
    "angle_unit": "radian",
    "quaternion_order": "xyzw",
    "opencv_camera_axes": {"x": "right", "y": "down", "z": "forward"},
}


def _lists(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _lists(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_lists(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


@dataclass
class CameraState:
    position_world: List[float]
    quaternion_world_xyzw: List[float]
    intrinsics: Dict[str, Any]

    @classmethod
    def create(
        cls,
        position_world: List[float],
        quaternion_world_xyzw: List[float],
        width: int,
        height: int,
        hfov_deg: float,
        near: float,
        far: float,
    ) -> "CameraState":
        return cls(
            list(map(float, position_world)),
            list(map(float, quaternion_world_xyzw)),
            camera_intrinsics(width, height, hfov_deg, near, far),
        )

    def geometry_dict(self) -> Dict[str, Any]:
        result = {
            "camera_position_world": self.position_world,
            "camera_quaternion_world_xyzw": self.quaternion_world_xyzw,
            "camera_intrinsics": self.intrinsics,
        }
        result.update(camera_transforms(self.position_world, self.quaternion_world_xyzw))
        return _lists(result)


@dataclass
class RobotState:
    robot_id: str
    base_position_world: List[float]
    yaw_rad: float
    camera_height_m: float
    camera: CameraState
    proxy_asset_handle: str = ""
    proxy_semantic_id: int = 0
    visibility: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        robot_id: str,
        base_position_world: List[float],
        yaw_rad: float,
        camera_height_m: float,
        width: int,
        height: int,
        hfov_deg: float,
        near: float,
        far: float,
        proxy_asset_handle: str = "",
        proxy_semantic_id: int = 0,
    ) -> "RobotState":
        base = np.asarray(base_position_world, dtype=np.float64)
        quaternion = yaw_to_quaternion_xyzw(yaw_rad)
        camera_position = base + np.array([0.0, camera_height_m, 0.0])
        camera = CameraState.create(
            camera_position.tolist(), quaternion.tolist(), width, height, hfov_deg, near, far
        )
        return cls(
            robot_id=robot_id,
            base_position_world=base.tolist(),
            yaw_rad=float(yaw_rad),
            camera_height_m=float(camera_height_m),
            camera=camera,
            proxy_asset_handle=proxy_asset_handle,
            proxy_semantic_id=int(proxy_semantic_id),
        )

    def synchronize_camera(self) -> None:
        base = np.asarray(self.base_position_world, dtype=np.float64)
        quaternion = yaw_to_quaternion_xyzw(self.yaw_rad)
        self.camera.position_world = (base + [0.0, self.camera_height_m, 0.0]).tolist()
        self.camera.quaternion_world_xyzw = quaternion.tolist()

    def metadata(self, paths: Dict[str, str]) -> Dict[str, Any]:
        quaternion = yaw_to_quaternion_xyzw(self.yaw_rad)
        data = {
            "robot_id": self.robot_id,
            "base_position_world": self.base_position_world,
            "yaw_rad": float(self.yaw_rad),
            "quaternion_world_xyzw": quaternion,
            "forward_world": forward_from_quaternion(quaternion),
            "camera_height_m": float(self.camera_height_m),
            "T_world_from_robot": transform_matrix(self.base_position_world, quaternion),
            "proxy_asset_handle": self.proxy_asset_handle,
            "proxy_semantic_id": self.proxy_semantic_id,
            "visibility": self.visibility,
            "files": paths,
        }
        data.update(self.camera.geometry_dict())
        return _lists(data)


@dataclass
class ObjectState:
    instance_id: str
    category: str
    asset_handle: str
    position_world: List[float]
    quaternion_world_xyzw: List[float]
    active: bool = True
    movable: bool = True
    semantic_id: int = 0
    bbox: Dict[str, Any] = field(default_factory=dict)
    visibility: Dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> Dict[str, Any]:
        result = asdict(self)
        result["T_world_from_object"] = transform_matrix(
            self.position_world, self.quaternion_world_xyzw
        )
        return _lists(result)


@dataclass
class WorldState:
    schema_version: str
    state_id: str
    scene_id: str
    floor_y: float
    random_seed: int
    robots: List[RobotState]
    objects: List[ObjectState] = field(default_factory=list)
    bev: Dict[str, Any] = field(default_factory=dict)
    overlap: Dict[str, Any] = field(default_factory=dict)
    parent_state_id: str = ""
    intervention: Dict[str, Any] = field(default_factory=dict)
    geometry_validation: Dict[str, Any] = field(default_factory=dict)

    def robot(self, robot_id: str) -> RobotState:
        for robot in self.robots:
            if robot.robot_id == robot_id:
                return robot
        raise KeyError(f"Unknown robot: {robot_id}")

    def object(self, instance_id: str) -> ObjectState:
        for obj in self.objects:
            if obj.instance_id == instance_id:
                return obj
        raise KeyError(f"Unknown object: {instance_id}")

    def metadata(self, robot_paths: Dict[str, Dict[str, str]], objects_path: str) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_id": self.state_id,
            "scene_id": self.scene_id,
            "floor_y": float(self.floor_y),
            "random_seed": int(self.random_seed),
            "coordinate_convention": COORDINATE_CONVENTION,
            "bev": _lists(self.bev),
            "robots": [r.metadata(robot_paths[r.robot_id]) for r in self.robots],
            "objects_path": objects_path,
            "controlled_object_count": len(self.objects),
            "difficulty_overlap": _lists(self.overlap),
            "parent_state_id": self.parent_state_id or None,
            "intervention": self.intervention or None,
            "geometry_validation": _lists(self.geometry_validation),
        }
