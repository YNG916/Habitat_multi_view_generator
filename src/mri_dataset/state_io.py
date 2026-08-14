from __future__ import annotations

import json
from pathlib import Path

from .world_state import CameraState, ObjectState, RobotState, WorldState


def read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_world_state(state_dir: Path) -> WorldState:
    state_dir = Path(state_dir)
    metadata = read_json(state_dir / "state.json")
    objects = [ObjectState(**item_without_transform(item)) for item in read_json(state_dir / metadata["objects_path"])]
    robots = []
    for item in metadata["robots"]:
        camera = CameraState(
            position_world=item["camera_position_world"],
            quaternion_world_xyzw=item["camera_quaternion_world_xyzw"],
            intrinsics=item["camera_intrinsics"],
        )
        robots.append(
            RobotState(
                robot_id=item["robot_id"],
                base_position_world=item["base_position_world"],
                yaw_rad=item["yaw_rad"],
                camera_height_m=item["camera_height_m"],
                camera=camera,
                proxy_asset_handle=item.get("proxy_asset_handle", ""),
                proxy_semantic_id=item.get("proxy_semantic_id", 0),
                visibility=item.get("visibility", {}),
            )
        )
    return WorldState(
        schema_version=metadata["schema_version"], state_id=metadata["state_id"],
        scene_id=metadata["scene_id"], floor_y=metadata["floor_y"],
        random_seed=metadata["random_seed"], robots=robots, objects=objects,
        bev=metadata.get("bev", {}), overlap=metadata.get("difficulty_overlap", {}),
        parent_state_id=metadata.get("parent_state_id") or "",
        intervention=metadata.get("intervention") or {},
        geometry_validation=metadata.get("geometry_validation") or {},
    )


def item_without_transform(item: dict) -> dict:
    result = dict(item)
    result.pop("T_world_from_object", None)
    return result
