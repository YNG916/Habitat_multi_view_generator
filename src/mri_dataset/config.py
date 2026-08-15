from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CollectorConfig:
    dataset_version: str = "1.0.0"
    protocol_version: str = "mri-formal-v1"
    scene_dataset_config: str = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    scenes: List[str] = field(default_factory=lambda: ["apt_1"])
    navmesh_root: str = "data/replica_cad/navmeshes"
    output_root: str = "outputs/mri_dataset"
    gpu_device_id: int = 0
    num_robots: int = 3
    # Formal dataset defaults. Use collector_pilot.json for quick validation.
    width: int = 2048
    height: int = 2048
    hfov_deg: float = 90.0
    near: float = 0.05
    far: float = 20.0
    pinhole_validation_min_samples: int = 3
    pinhole_validation_median_error_m: float = 0.03
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
    bev_ceiling_clearance_m: float = 0.15
    bev_near: float = 0.02
    bev_far: float = 10.0
    height_validation_samples: int = 24
    height_validation_min_samples: int = 8
    height_validation_max_error_m: float = 0.02
    collision_penetration_tolerance_m: float = 0.002
    support_contact_tolerance_m: float = 0.005
    max_state_sampling_attempts: int = 100
    enable_instance: bool = True
    enable_semantic: bool = True
    semantic_category_ids: Dict[str, int] = field(
        default_factory=lambda: {
            "robot": 1, "cup": 10, "bowl": 11, "book": 12
        }
    )
    benchmark_visibility_min_pixels: int = 20
    min_target_visible_observers: int = 1
    require_visible_robot_target: bool = False
    require_visible_object_target: bool = False
    enable_robot_proxies: bool = True
    robot_proxy_configs: List[str] = field(
        default_factory=lambda: [
            "assets/robot_proxies/robot_red.object_config.json",
            "assets/robot_proxies/robot_green.object_config.json",
            "assets/robot_proxies/robot_blue.object_config.json",
        ]
    )
    robot_proxy_height_variants: Dict[str, List[str]] = field(default_factory=dict)
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
    num_states_by_split: Dict[str, int] = field(default_factory=dict)
    num_edits_per_state: int = 1
    random_seed: int = 123
    resume: bool = True
    save_visualizations: bool = True
    compress_numeric_arrays: bool = False
    run_multilevel_calibration_preflight: bool = True
    max_intervention_sampling_attempts: int = 80
    scene_splits: Dict[str, List[str]] = field(
        default_factory=lambda: {"train": ["apt_1"], "val": [], "test": []}
    )
    level2_regimes_by_split: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "train": ["id"],
            "val": ["id", "ood"],
            "test": ["id", "ood"],
        }
    )
    intervention_type_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "robot_translate": 0.25,
            "robot_rotate": 0.20,
            "object_translate": 0.25,
            "object_place_relative": 0.20,
            "object_remove": 0.10,
        }
    )
    intervention_regimes: Dict[str, Dict[str, List[float]]] = field(
        default_factory=lambda: {
            "id": {
                "robot_translate_m": [0.5, 1.0],
                "robot_rotate_deg": [-60.0, -30.0, 30.0, 60.0],
                "object_translate_m": [0.5, 1.0],
                "object_place_relative_m": [0.5, 1.0],
            },
            "ood": {
                "robot_translate_m": [0.75, 1.5],
                "robot_rotate_deg": [-90.0, -45.0, 45.0, 90.0],
                "object_translate_m": [0.75, 1.5],
                "object_place_relative_m": [0.75, 1.5],
            },
        }
    )
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

    def scene_split(self, scene_id: str) -> str:
        matches = [
            split for split, scenes in self.scene_splits.items() if scene_id in scenes
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Scene {scene_id!r} must belong to exactly one split, got {matches}"
            )
        return matches[0]

    def states_for_scene(self, scene_id: str) -> int:
        split = self.scene_split(scene_id)
        return int(self.num_states_by_split.get(split, self.num_states))

    def generation_fingerprint(self) -> str:
        """Fingerprint fields that change generated sample semantics or bytes."""
        data = self.to_dict()
        for key in (
            "output_root",
            "num_states",
            "num_states_by_split",
            "num_edits_per_state",
            "resume",
        ):
            data.pop(key, None)
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        if self.dataset_version != "1.0.0":
            raise ValueError("This generator currently writes dataset_version=1.0.0")
        if len(self.scenes) != len(set(self.scenes)) or not self.scenes:
            raise ValueError("scenes must be a non-empty list without duplicates")
        assigned = [scene for scenes in self.scene_splits.values() for scene in scenes]
        if len(assigned) != len(set(assigned)):
            raise ValueError("A scene appears in more than one scene_splits partition")
        if set(assigned) != set(self.scenes):
            raise ValueError("scene_splits must assign every configured scene exactly once")
        if set(self.scene_splits) != {"train", "val", "test"}:
            raise ValueError("scene_splits must contain train, val, and test")
        for scene in self.scenes:
            if "bev_camera_height_m" not in self.scene_overrides.get(scene, {}):
                raise ValueError(
                    f"Scene {scene} needs an explicit bev_camera_height_m override"
                )
        if self.width < 2 or self.height < 2 or self.bev_meters_per_pixel <= 0:
            raise ValueError("Image sizes and BEV meters-per-pixel must be positive")
        if not self.enable_instance:
            raise ValueError("Formal protocol requires OBJECT_ID instance observations")
        required_semantics = {"robot", *self.controlled_object_whitelist}
        if self.enable_semantic and not required_semantics.issubset(
            self.semantic_category_ids
        ):
            raise ValueError("semantic_category_ids is missing a controlled category")
        semantic_ids = list(map(int, self.semantic_category_ids.values()))
        if any(value <= 0 for value in semantic_ids) or len(semantic_ids) != len(
            set(semantic_ids)
        ):
            raise ValueError("semantic category IDs must be unique positive integers")
        for height_key, variants in self.robot_proxy_height_variants.items():
            height = float(height_key)
            if not self.camera_height_min_m <= height <= self.camera_height_max_m:
                raise ValueError(f"Robot proxy height {height} is outside camera range")
            if len(variants) < self.num_robots:
                raise ValueError(
                    f"Robot proxy height {height} has fewer variants than robots"
                )
        if self.num_states < 0 or self.num_edits_per_state < 0:
            raise ValueError("num_states and num_edits_per_state cannot be negative")
        if set(self.num_states_by_split) - set(self.scene_splits):
            raise ValueError("num_states_by_split contains an unknown split")
        if any(int(value) < 0 for value in self.num_states_by_split.values()):
            raise ValueError("num_states_by_split values cannot be negative")
        if self.pinhole_validation_min_samples < 1:
            raise ValueError("pinhole_validation_min_samples must be at least 1")
        if self.pinhole_validation_median_error_m <= 0:
            raise ValueError("pinhole_validation_median_error_m must be positive")
        if not 1 <= self.height_validation_min_samples <= self.height_validation_samples:
            raise ValueError(
                "height_validation_min_samples must be between 1 and height_validation_samples"
            )
        if self.max_state_sampling_attempts < 1:
            raise ValueError("max_state_sampling_attempts must be at least 1")
        if self.max_intervention_sampling_attempts < 1:
            raise ValueError("max_intervention_sampling_attempts must be at least 1")
        if self.benchmark_visibility_min_pixels < 1:
            raise ValueError("benchmark_visibility_min_pixels must be at least 1")
        if self.min_target_visible_observers < 0:
            raise ValueError("min_target_visible_observers cannot be negative")
        supported = {
            "robot_translate",
            "robot_rotate",
            "object_translate",
            "object_place_relative",
            "object_remove",
        }
        if not self.intervention_type_weights:
            raise ValueError("intervention_type_weights cannot be empty")
        if not set(self.intervention_type_weights).issubset(supported):
            raise ValueError("intervention_type_weights contains an unsupported type")
        if any(float(value) <= 0 for value in self.intervention_type_weights.values()):
            raise ValueError("All intervention type weights must be positive")
        required_regime_keys = {
            "robot_translate_m",
            "robot_rotate_deg",
            "object_translate_m",
            "object_place_relative_m",
        }
        for regime, specification in self.intervention_regimes.items():
            if set(specification) != required_regime_keys:
                raise ValueError(f"Regime {regime!r} has incomplete parameter domains")
            if any(not values for values in specification.values()):
                raise ValueError(f"Regime {regime!r} contains an empty parameter domain")
        for split, regimes in self.level2_regimes_by_split.items():
            if split not in self.scene_splits:
                raise ValueError(f"Unknown Level-2 split {split!r}")
            unknown = set(regimes) - set(self.intervention_regimes)
            if unknown:
                raise ValueError(f"Unknown regimes for {split}: {sorted(unknown)}")

        id_spec = self.intervention_regimes.get("id")
        ood_spec = self.intervention_regimes.get("ood")
        if id_spec and ood_spec:
            for key in required_regime_keys:
                if set(map(float, id_spec[key])) & set(map(float, ood_spec[key])):
                    raise ValueError(f"ID and OOD domains overlap for {key}")


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
    config = CollectorConfig(**data)
    config.validate()
    return config
