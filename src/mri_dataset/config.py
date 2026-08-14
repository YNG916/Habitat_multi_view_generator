from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CollectorConfig:
    scene_dataset_config: str = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    scenes: List[str] = field(default_factory=lambda: ["apt_1"])
    navmesh_root: str = "data/replica_cad/navmeshes"
    output_root: str = "outputs/mri_dataset"
    gpu_device_id: int = 0
    num_robots: int = 3
    width: int = 2048
    height: int = 2048
    hfov_deg: float = 90.0
    near: float = 0.05
    far: float = 20.0
    camera_height_min_m: float = 0.4
    camera_height_max_m: float = 1.4
    min_obstacle_distance_m: float = 0.55
    min_inter_robot_distance_m: float = 0.9
    local_sampling_radius_m: float = 3.5
    floor_tolerance_m: float = 0.25
    heading_mode: str = "mixed"
    shared_heading_jitter_deg: float = 20.0
    bev_meters_per_pixel: float = 0.00625
    bev_camera_height_m: float = 2.2
    bev_near: float = 0.02
    bev_far: float = 10.0
    enable_instance: bool = True
    enable_robot_proxies: bool = True
    robot_proxy_configs: List[str] = field(
        default_factory=lambda: [
            "assets/robot_proxies/robot_red.object_config.json",
            "assets/robot_proxies/robot_green.object_config.json",
            "assets/robot_proxies/robot_blue.object_config.json",
        ]
    )
    controlled_object_whitelist: Dict[str, str] = field(
        default_factory=lambda: {
            "cup": "frl_apartment_cup_01.object_config.json",
            "bowl": "frl_apartment_bowl_01.object_config.json",
            "book": "frl_apartment_book_01.object_config.json",
        }
    )
    controlled_objects_per_state: int = 1
    controlled_object_min_separation_m: float = 0.65
    num_states: int = 1
    num_edits_per_state: int = 1
    random_seed: int = 123
    scene_overrides: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {"apt_1": {"bev_camera_height_m": 2.2}}
    )

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()

    @property
    def dataset_config_path(self) -> Path:
        return self.resolve(self.scene_dataset_config)

    @property
    def output_path(self) -> Path:
        return self.resolve(self.output_root)

    def navmesh_path(self, scene_id: str) -> Path:
        return self.resolve(str(Path(self.navmesh_root) / f"{scene_id}.navmesh"))

    def scene_value(self, scene_id: str, key: str) -> Any:
        return self.scene_overrides.get(scene_id, {}).get(key, getattr(self, key))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: Optional[str] = None, **overrides: Any) -> CollectorConfig:
    data: Dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    data.update({key: value for key, value in overrides.items() if value is not None})
    known = CollectorConfig.__dataclass_fields__
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"Unknown collector config fields: {unknown}")
    return CollectorConfig(**data)
