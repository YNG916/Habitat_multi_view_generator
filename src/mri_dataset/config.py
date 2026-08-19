from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_CORRECTNESS_REVISION = "unique-object-categories-and-zero-runtime-margin-v1"


def _protocol_file_sha256(path: Path, ignored_json_keys=()):
    """Hash protocol content while excluding review-only JSON metadata."""
    if not path.is_file():
        return "missing"
    if not ignored_json_keys:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    ignored = set(ignored_json_keys)

    def strip(value):
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if key not in ignored}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    payload = json.dumps(
        strip(json.loads(path.read_text(encoding="utf-8"))),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass
class CollectorConfig:
    """Configuration for the HSSD-only formal collector."""

    dataset_version: str = "3.1.0"
    protocol_version: str = "mri-hssd-region-formal-v2.1"
    dataset_source: str = "hssd"
    scene_dataset_config: str = "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"
    official_scene_splits: str = "data/scene_datasets/hssd-hab/scene_splits.yaml"
    scene_registry: str = "data/hssd_processed/scene_registry.json"
    split_manifest: str = "data/hssd_processed/split_manifest.json"
    navmesh_cache_root: str = "data/hssd_processed/navmeshes"
    controlled_object_registry: str = "configs/hssd_controlled_objects.json"
    hssd_preprocess_overrides_path: str = "configs/hssd_preprocess_overrides.json"
    require_preprocessed_registry: bool = True
    internal_val_fraction: float = .10
    split_seed: int = 20260317
    max_scenes_per_split: Dict[str, int] = field(default_factory=dict)
    scenes: List[str] = field(default_factory=list)
    scene_splits: Dict[str, List[str]] = field(
        default_factory=lambda: {"train": [], "val": [], "test": []}
    )

    output_root: str = "outputs/mri_hssd"
    gpu_device_id: int = 0
    num_robots: int = 3
    width: int = 2048
    height: int = 2048
    hfov_deg: float = 90.
    near: float = .05
    far: float = 20.
    pinhole_validation_min_samples: int = 3
    pinhole_validation_median_error_m: float = .03
    robot_body_diameter_m: float = .46
    robot_body_height_m: float = .107
    camera_height_min_m: float = .15
    camera_height_max_m: float = .15
    robot_camera_forward_offset_m: float = .235
    navmesh_agent_radius_m: float = .28
    navmesh_agent_height_m: float = .20
    navmesh_agent_max_climb_m: float = .05
    navmesh_agent_max_slope_deg: float = 20.
    min_obstacle_distance_m: float = .55
    min_inter_robot_distance_m: float = .9
    local_sampling_radius_m: float = 3.5
    floor_tolerance_m: float = .25
    heading_mode: str = "mixed"
    shared_heading_jitter_deg: float = 20.

    bev_meters_per_pixel: float = .00625
    bev_preferred_camera_height_m: float = 2.2
    bev_ceiling_clearance_m: float = .15
    bev_near: float = .02
    bev_far: float = 10.
    bev_context_margin_m: float = .75  # floor-global debug only
    region_context_margin_m: float = .75
    region_min_navigable_area_m2: float = 3.5
    region_min_extent_m: float = 1.5
    region_max_extent_m: float = 15.
    region_min_sample_count: int = 24
    region_sampling_trials: int = 24
    region_min_sampling_success_rate: float = .5
    preprocess_region_mask_min_fraction: float = .10
    bev_ceiling_ray_samples: int = 64
    bev_ceiling_ray_max_distance_m: float = 8.
    bev_min_ceiling_height_m: float = 1.6
    bev_open_scene_margin_m: float = .25
    floor_samples_per_island: int = 1024
    floor_min_samples_per_island: int = 64
    floor_min_island_area_m2: float = 2.
    floor_min_navigable_area_m2: float = 12.
    floor_group_tolerance_m: float = .30
    floor_max_vertical_span_m: float = .45
    floor_ambiguity_min_vertical_separation_m: float = .75
    floor_ambiguity_max_xz_overlap_ratio: float = .25
    preprocess_bev_width: int = 256
    preprocess_bev_min_finite_fraction: float = .02
    preprocess_bev_min_rgb_std: float = 1.

    height_validation_samples: int = 24
    height_validation_min_samples: int = 8
    height_validation_max_error_m: float = .03
    collision_penetration_tolerance_m: float = .002
    support_contact_tolerance_m: float = .005
    robot_floor_collision_tolerance_m: float = .012
    max_state_sampling_attempts: int = 100
    enable_instance: bool = True
    enable_semantic: bool = True
    semantic_category_ids: Dict[str, int] = field(default_factory=lambda: {
        "robot": 1, "cup": 10, "bowl": 11, "book": 12, "bottle": 13,
        "box": 14, "bag": 15, "basket": 16, "can": 17, "shoe": 18,
        "toy": 19,
    })
    benchmark_visibility_min_pixels: int = 32
    benchmark_visibility_min_fraction: float = .0001
    min_target_visible_observers: int = 1
    require_visible_robot_target: bool = False
    require_visible_object_target: bool = False

    enable_robot_proxies: bool = True
    robot_proxy_configs: List[str] = field(default_factory=lambda: [
        "assets/robot_proxies/robot_red.object_config.json",
        "assets/robot_proxies/robot_green.object_config.json",
        "assets/robot_proxies/robot_blue.object_config.json",
    ])
    robot_proxy_height_variants: Dict[str, List[str]] = field(default_factory=dict)
    controlled_object_pools: Dict[str, List[str]] = field(default_factory=dict)
    controlled_object_asset_hashes: Dict[str, str] = field(default_factory=dict)
    controlled_objects_min_per_state: int = 2
    controlled_objects_max_per_state: int = 4
    controlled_object_min_separation_m: float = .65
    num_states: int = 1
    states_per_region: int = 1
    max_states_per_scene: int = 0
    num_states_by_split: Dict[str, int] = field(default_factory=dict)
    num_edits_per_state: int = 1
    random_seed: int = 123
    resume: bool = True
    save_visualizations: bool = True
    compress_numeric_arrays: bool = True
    run_multilevel_calibration_preflight: bool = True
    max_intervention_sampling_attempts: int = 80
    level2_regimes_by_split: Dict[str, List[str]] = field(default_factory=lambda: {
        "train": ["id"], "val": ["id", "ood"], "test": ["id", "ood"],
    })
    intervention_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "robot_translate": .25, "robot_rotate": .20, "object_translate": .25,
        "object_place_relative": .20, "object_remove": .10,
    })
    intervention_regimes: Dict[str, Dict[str, List[float]]] = field(
        default_factory=lambda: {
            "id": {
                "robot_translate_m": [.5, 1.], "robot_rotate_deg": [-60., -30., 30., 60.],
                "object_translate_m": [.5, 1.], "object_place_relative_m": [.5, 1.],
            },
            "ood": {
                "robot_translate_m": [.75, 1.5], "robot_rotate_deg": [-90., -45., 45., 90.],
                "object_translate_m": [.75, 1.5], "object_place_relative_m": [.75, 1.5],
            },
        }
    )

    @property
    def repo_root(self): return REPO_ROOT
    def resolve(self, value):
        path = Path(value)
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()
    dataset_config_path = property(lambda s: s.resolve(s.scene_dataset_config))
    official_scene_splits_path = property(lambda s: s.resolve(s.official_scene_splits))
    scene_registry_path = property(lambda s: s.resolve(s.scene_registry))
    split_manifest_path = property(lambda s: s.resolve(s.split_manifest))
    navmesh_cache_root_path = property(lambda s: s.resolve(s.navmesh_cache_root))
    controlled_object_registry_path = property(lambda s: s.resolve(s.controlled_object_registry))
    hssd_preprocess_overrides_file = property(lambda s: s.resolve(s.hssd_preprocess_overrides_path))
    output_path = property(lambda s: s.resolve(s.output_root))

    def navmesh_cache_path(self, scene_id):
        return self.navmesh_cache_root_path / f"{scene_id}.navmesh"

    def hssd_preprocess_overrides(self):
        path = self.hssd_preprocess_overrides_file
        if not path.exists(): return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict): raise ValueError("HSSD overrides must be an object")
        return data

    def registry(self):
        from .scene_registry import SceneRegistry
        return SceneRegistry.load(self.scene_registry_path)

    def scene_split(self, scene_id):
        matches = [split for split, values in self.scene_splits.items() if scene_id in values]
        if len(matches) != 1:
            raise ValueError(f"HSSD scene {scene_id!r} must have exactly one split: {matches}")
        return matches[0]

    def states_for_scene(self, scene_id):
        """Legacy alias; formal quotas are defined per semantic region."""
        return self.states_for_region(scene_id)

    def states_for_region(self, scene_id):
        return int(
            self.num_states_by_split.get(
                self.scene_split(scene_id), self.states_per_region
            )
        )

    def state_targets(self, specs, num_states_override=None):
        """Apply per-region quota and an optional deterministic per-scene cap."""
        used={}
        result=[]
        for scene,floor,region in specs:
            target=(
                int(num_states_override)
                if num_states_override is not None
                else self.states_for_region(scene.scene_id)
            )
            cap=int(self.max_states_per_scene)
            if cap>0:
                target=min(target,max(0,cap-used.get(scene.scene_id,0)))
            used[scene.scene_id]=used.get(scene.scene_id,0)+target
            if target>0:
                result.append((scene,floor,region,target))
        return result

    def collection_specs(self, scene_id=None, floor_id=None, region_id=None):
        allowed,result=set(self.scenes),[]
        for scene in self.registry().scenes:
            if not scene.eligible or scene.scene_id not in allowed: continue
            if scene_id is not None and scene.scene_id!=scene_id: continue
            for floor in scene.eligible_floors:
                if floor_id is not None and floor.floor_id!=floor_id: continue
                for region in floor.eligible_regions:
                    if region_id is None or region.region_id==region_id:
                        result.append((scene,floor,region))
        if scene_id is not None and not result:
            raise ValueError(f"No eligible HSSD region matches {scene_id}/{floor_id or '*'}/{region_id or '*'}")
        order={"train":0,"val":1,"test":2}
        return sorted(result,key=lambda item:(order[self.scene_split(item[0].scene_id)],item[0].scene_id,item[1].floor_id,item[2].region_id))

    def to_dict(self): return asdict(self)

    def robot_proxy_asset_fingerprints(self):
        configured = list(self.robot_proxy_configs)
        for variants in self.robot_proxy_height_variants.values(): configured.extend(variants)
        result = {}
        for directory in sorted({self.resolve(path).parent for path in configured}):
            if not directory.is_dir():
                result[str(directory)] = "missing"
                continue
            for asset in sorted(path for path in directory.iterdir() if path.is_file()):
                try: key = str(asset.relative_to(REPO_ROOT))
                except ValueError: key = str(asset)
                result[key] = hashlib.sha256(asset.read_bytes()).hexdigest()
        return result

    def generation_fingerprint(self):
        data = self.to_dict()
        for key in ("output_root", "num_states", "num_states_by_split", "num_edits_per_state", "resume"):
            data.pop(key, None)
        data["_generator_correctness_revision"] = GENERATOR_CORRECTNESS_REVISION
        data["_robot_proxy_assets_sha256"] = self.robot_proxy_asset_fingerprints()
        for label, path, ignored_keys in (
            (
                "scene_registry",
                self.scene_registry_path,
                ("preview_path", "preview_panels"),
            ),
            ("split_manifest", self.split_manifest_path, ()),
            ("controlled_objects", self.controlled_object_registry_path, ()),
        ):
            data[f"_{label}_sha256"] = _protocol_file_sha256(path, ignored_keys)
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def validate(self):
        if self.dataset_version != "3.1.0" or self.dataset_source != "hssd":
            raise ValueError("Formal generation is HSSD-only dataset version 3.1.0")
        lowered = str(self.dataset_config_path).lower()
        if self.dataset_config_path.name != "hssd-hab.scene_dataset_config.json":
            raise ValueError("Use standard hssd-hab.scene_dataset_config.json")
        if any(token in lowered for token in ("replica", "uncluttered", "articulated")):
            raise ValueError("ReplicaCAD and HSSD variants are forbidden")
        if set(self.scene_splits) != {"train", "val", "test"}:
            raise ValueError("scene_splits needs train/val/test")
        assigned = sum(self.scene_splits.values(), [])
        if len(assigned) != len(set(assigned)): raise ValueError("HSSD scene split leakage")
        if self.require_preprocessed_registry:
            required = {
                "HSSD config": self.dataset_config_path, "official splits": self.official_scene_splits_path,
                "scene registry": self.scene_registry_path, "split manifest": self.split_manifest_path,
                "object registry": self.controlled_object_registry_path,
            }
            for label, path in required.items():
                if not path.is_file(): raise FileNotFoundError(f"Missing {label}: {path}")
            if not self.scenes or set(assigned) != set(self.scenes):
                raise ValueError("Manifest must assign every configured HSSD scene")
            eligible = {scene.scene_id for scene in self.registry().scenes if scene.eligible}
            if not set(self.scenes).issubset(eligible): raise ValueError("Ineligible HSSD scene configured")
        if min(self.width, self.height) < 2 or self.bev_meters_per_pixel <= 0:
            raise ValueError("Invalid image/BEV resolution")
        if self.camera_height_min_m != self.camera_height_max_m:
            raise ValueError("Formal camera height must be fixed")
        if self.camera_height_min_m <= self.robot_body_height_m:
            raise ValueError("Camera must be above the robot body")
        if self.navmesh_agent_radius_m < self.robot_body_diameter_m / 2:
            raise ValueError("NavMesh radius smaller than robot")
        if self.navmesh_agent_height_m < self.camera_height_max_m + .03:
            raise ValueError("NavMesh height does not clear camera")
        front = self.robot_body_diameter_m / 2
        if not front <= self.robot_camera_forward_offset_m <= front + .05:
            raise ValueError("Camera not at robot front")
        if self.min_obstacle_distance_m < front + .10: raise ValueError("Unsafe obstacle clearance")
        if self.min_inter_robot_distance_m < self.robot_body_diameter_m + .10:
            raise ValueError("Unsafe robot clearance")
        if not self.enable_instance: raise ValueError("OBJECT_ID is mandatory")
        if self.require_preprocessed_registry and self.controlled_objects_max_per_state > 0 and not self.controlled_object_pools:
            raise ValueError("HSSD approved object registry has no curated asset pools")
        if any(not assets for assets in self.controlled_object_pools.values()):
            raise ValueError("Every object category needs approved assets")
        if not 0 <= self.controlled_objects_min_per_state <= self.controlled_objects_max_per_state:
            raise ValueError("Invalid controlled object count range")
        available_categories=sum(bool(assets) for assets in self.controlled_object_pools.values())
        if available_categories and self.controlled_objects_max_per_state > available_categories:
            raise ValueError(
                "controlled_objects_max_per_state exceeds the number of approved categories"
            )
        if self.enable_semantic and not {"robot", *self.controlled_object_pools}.issubset(self.semantic_category_ids):
            raise ValueError("Missing semantic category")
        ids = list(map(int, self.semantic_category_ids.values()))
        if any(value <= 0 for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("Semantic IDs must be unique positive values")
        if min(
            self.num_states,self.states_per_region,self.max_states_per_scene,
            self.num_edits_per_state,
        )<0:
            raise ValueError("Negative counts")
        if set(self.num_states_by_split) - set(self.scene_splits): raise ValueError("Unknown count split")
        if self.heading_mode not in {"mixed","random","shared_focus"}: raise ValueError("Unknown heading_mode")
        if not 0 <= self.benchmark_visibility_min_fraction <= 1: raise ValueError("Invalid visibility fraction")
        if not 0 < self.region_min_navigable_area_m2 or self.region_min_extent_m <= 0 or self.region_max_extent_m <= self.region_min_extent_m:
            raise ValueError("Invalid region thresholds")
        if not 1 <= self.height_validation_min_samples <= self.height_validation_samples:
            raise ValueError("Invalid height validation samples")
        supported = {"robot_translate", "robot_rotate", "object_translate", "object_place_relative", "object_remove"}
        if not self.intervention_type_weights or not set(self.intervention_type_weights).issubset(supported):
            raise ValueError("Unsupported interventions")
        keys = {"robot_translate_m", "robot_rotate_deg", "object_translate_m", "object_place_relative_m"}
        for name, spec in self.intervention_regimes.items():
            if set(spec) != keys or any(not values for values in spec.values()):
                raise ValueError(f"Incomplete regime {name}")
        for split, regimes in self.level2_regimes_by_split.items():
            if split not in self.scene_splits or set(regimes) - set(self.intervention_regimes):
                raise ValueError(f"Invalid Level-2 regimes for {split}")
        if "id" in self.intervention_regimes and "ood" in self.intervention_regimes:
            for key in keys:
                if set(map(float, self.intervention_regimes["id"][key])) & set(map(float, self.intervention_regimes["ood"][key])):
                    raise ValueError(f"ID/OOD overlap for {key}")


def _materialize_hssd_sources(config):
    path=config.controlled_object_registry_path
    if config.require_preprocessed_registry and path.is_file():
        data=json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version")=="2.0.0":
            from .objects import inspect_approved_object_registry
            registry_report=inspect_approved_object_registry(
                data,config.dataset_config_path.parent,config.semantic_category_ids
            )
            if not registry_report["passed"]:
                details="; ".join(registry_report["errors"][:8])
                raise ValueError(f"Invalid approved HSSD object registry: {details}")
            approved=data.get("approved_assets",{})
            pools={}; hashes={}
            for category,records in approved.items():
                if not isinstance(records,list): raise ValueError("Approved pools must be lists")
                pools[str(category)]=[]
                for record in records:
                    canonical=str(record["canonical_id"])
                    if canonical.startswith("/") or ".." in Path(canonical).parts: raise ValueError("Unsafe canonical object ID")
                    pools[str(category)].append(canonical); hashes[canonical]=str(record["asset_fingerprint"])
            config.controlled_object_pools=pools; config.controlled_object_asset_hashes=hashes
        elif config.require_preprocessed_registry:
            raise ValueError("Legacy one-handle object registry is unsupported")
    if not config.require_preprocessed_registry:return
    if not config.split_manifest_path.is_file() or not config.scene_registry_path.is_file():return
    from .scene_registry import load_split_manifest
    manifest=load_split_manifest(config.split_manifest_path)
    eligible={scene.scene_id for scene in config.registry().scenes if scene.eligible}
    splits={}
    for split in ("train","val","test"):
        values=[scene for scene in manifest["scene_splits"][split] if scene in eligible]
        maximum=config.max_scenes_per_split.get(split)
        if maximum is not None:
            if int(maximum)<0:raise ValueError("Negative max_scenes_per_split")
            values=values[:int(maximum)]
        splits[split]=values
    config.scene_splits=splits
    config.scenes=sum((splits[name] for name in ("train","val","test")),[])


def load_config(path: Optional[str] = None, **overrides: Any) -> CollectorConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    data.update({key: value for key, value in overrides.items() if value is not None})
    unknown = sorted(set(data) - set(CollectorConfig.__dataclass_fields__))
    if unknown: raise ValueError(f"Unknown collector config fields: {unknown}")
    config = CollectorConfig(**data)
    _materialize_hssd_sources(config)
    config.validate()
    return config
