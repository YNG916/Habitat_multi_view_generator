"""Persistent HSSD scene/floor preprocessing registry and split manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


REGISTRY_SCHEMA_VERSION = "1.0.0"
SPLIT_SCHEMA_VERSION = "1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fraction(seed: int, scene_id: str) -> float:
    payload = f"{int(seed)}|{scene_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64)


@dataclass(frozen=True)
class FloorSpec:
    floor_id: str
    representative_floor_y: float
    allowed_island_ids: List[int]
    navigable_area_m2: float
    navigable_bounds_world: List[List[float]]
    visual_bev_bounds_world: List[List[float]]
    bev_camera_height_m: float
    eligible: bool
    rejection_reasons: List[str] = field(default_factory=list)
    preprocessing_validation: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "FloorSpec":
        return cls(
            floor_id=str(data["floor_id"]),
            representative_floor_y=float(data["representative_floor_y"]),
            allowed_island_ids=[int(value) for value in data["allowed_island_ids"]],
            navigable_area_m2=float(data["navigable_area_m2"]),
            navigable_bounds_world=[
                [float(value) for value in bound]
                for bound in data["navigable_bounds_world"]
            ],
            visual_bev_bounds_world=[
                [float(value) for value in bound]
                for bound in data["visual_bev_bounds_world"]
            ],
            bev_camera_height_m=float(data["bev_camera_height_m"]),
            eligible=bool(data["eligible"]),
            rejection_reasons=list(map(str, data.get("rejection_reasons", []))),
            preprocessing_validation=dict(data.get("preprocessing_validation", {})),
        )

    def validate(self) -> None:
        if not self.floor_id:
            raise ValueError("floor_id cannot be empty")
        if not self.allowed_island_ids or len(self.allowed_island_ids) != len(
            set(self.allowed_island_ids)
        ):
            raise ValueError(f"{self.floor_id}: allowed islands must be unique/non-empty")
        if self.navigable_area_m2 <= 0:
            raise ValueError(f"{self.floor_id}: navigable area must be positive")
        if self.bev_camera_height_m <= 0:
            raise ValueError(f"{self.floor_id}: BEV camera height must be positive")
        for name, bounds in (
            ("navigable", self.navigable_bounds_world),
            ("visual", self.visual_bev_bounds_world),
        ):
            if len(bounds) != 2 or any(len(bound) != 3 for bound in bounds):
                raise ValueError(f"{self.floor_id}: invalid {name} bounds")
            if any(low >= high for low, high in zip(bounds[0], bounds[1])):
                raise ValueError(f"{self.floor_id}: unordered {name} bounds")
        if self.eligible and self.rejection_reasons:
            raise ValueError(f"{self.floor_id}: eligible floor has rejection reasons")


@dataclass(frozen=True)
class SceneSpec:
    dataset_source: str
    scene_id: str
    official_split: str
    eligible: bool
    rejection_reasons: List[str]
    cached_navmesh_path: str
    navmesh_sha256: str
    navmesh_settings: Dict[str, object]
    navmesh_settings_fingerprint: str
    rendered_scene_aabb: List[List[float]]
    floors: List[FloorSpec]
    preprocessing_validation: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "SceneSpec":
        return cls(
            dataset_source=str(data["dataset_source"]),
            scene_id=str(data["scene_id"]),
            official_split=str(data["official_split"]),
            eligible=bool(data["eligible"]),
            rejection_reasons=list(map(str, data.get("rejection_reasons", []))),
            cached_navmesh_path=str(data["cached_navmesh_path"]),
            navmesh_sha256=str(data.get("navmesh_sha256", "")),
            navmesh_settings=dict(data["navmesh_settings"]),
            navmesh_settings_fingerprint=str(
                data["navmesh_settings_fingerprint"]
            ),
            rendered_scene_aabb=[
                [float(value) for value in bound]
                for bound in data["rendered_scene_aabb"]
            ],
            floors=[FloorSpec.from_dict(item) for item in data.get("floors", [])],
            preprocessing_validation=dict(data.get("preprocessing_validation", {})),
        )

    def floor(self, floor_id: str, require_eligible: bool = True) -> FloorSpec:
        matches = [floor for floor in self.floors if floor.floor_id == floor_id]
        if len(matches) != 1:
            raise KeyError(f"Scene {self.scene_id!r} has no unique floor {floor_id!r}")
        result = matches[0]
        if require_eligible and not result.eligible:
            raise ValueError(
                f"Scene/floor {self.scene_id}/{floor_id} is ineligible: "
                f"{result.rejection_reasons}"
            )
        return result

    @property
    def eligible_floors(self) -> List[FloorSpec]:
        return [floor for floor in self.floors if floor.eligible]

    def validate(self) -> None:
        if self.dataset_source != "hssd":
            raise ValueError(f"{self.scene_id}: registry source must be hssd")
        if self.official_split not in {"train", "val"}:
            raise ValueError(f"{self.scene_id}: invalid official HSSD split")
        if len(self.rendered_scene_aabb) != 2 or any(
            len(bound) != 3 for bound in self.rendered_scene_aabb
        ):
            raise ValueError(f"{self.scene_id}: invalid rendered AABB")
        floor_ids = [floor.floor_id for floor in self.floors]
        if len(floor_ids) != len(set(floor_ids)):
            raise ValueError(f"{self.scene_id}: duplicate floor IDs")
        for floor in self.floors:
            floor.validate()
        if self.eligible != bool(self.eligible_floors):
            raise ValueError(
                f"{self.scene_id}: scene eligibility must match eligible floors"
            )
        if self.eligible and self.rejection_reasons:
            raise ValueError(f"{self.scene_id}: eligible scene has rejection reasons")


@dataclass
class SceneRegistry:
    dataset_source: str
    dataset_config_path: str
    official_splits_path: str
    scenes: List[SceneSpec]
    schema_version: str = REGISTRY_SCHEMA_VERSION
    preprocessing_config: Dict[str, object] = field(default_factory=dict)
    statistics: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "SceneRegistry":
        registry = cls(
            schema_version=str(data.get("schema_version", REGISTRY_SCHEMA_VERSION)),
            dataset_source=str(data["dataset_source"]),
            dataset_config_path=str(data["dataset_config_path"]),
            official_splits_path=str(data["official_splits_path"]),
            preprocessing_config=dict(data.get("preprocessing_config", {})),
            scenes=[SceneSpec.from_dict(item) for item in data.get("scenes", [])],
            statistics=dict(data.get("statistics", {})),
        )
        registry.validate()
        return registry

    @classmethod
    def load(cls, path: Path) -> "SceneRegistry":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        from .serialization import write_json

        self.statistics = self.compute_statistics()
        self.validate()
        write_json(Path(path), self.to_dict())

    def validate(self) -> None:
        if self.schema_version != REGISTRY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported scene registry schema {self.schema_version}")
        if self.dataset_source != "hssd":
            raise ValueError("Scene registry must be HSSD-only")
        scene_ids = [scene.scene_id for scene in self.scenes]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("Scene registry contains duplicate scene IDs")
        for scene in self.scenes:
            scene.validate()

    def scene(self, scene_id: str, require_eligible: bool = True) -> SceneSpec:
        matches = [scene for scene in self.scenes if scene.scene_id == scene_id]
        if len(matches) != 1:
            raise KeyError(f"Scene registry has no unique scene {scene_id!r}")
        result = matches[0]
        if require_eligible and not result.eligible:
            raise ValueError(
                f"Scene {scene_id!r} is ineligible: {result.rejection_reasons}"
            )
        return result

    def compute_statistics(self) -> dict:
        eligible_scenes = [scene for scene in self.scenes if scene.eligible]
        all_floors = [floor for scene in self.scenes for floor in scene.floors]
        eligible_floors = [floor for floor in all_floors if floor.eligible]
        rejection_counts: Dict[str, int] = {}
        for scene in self.scenes:
            for reason in scene.rejection_reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            for floor in scene.floors:
                for reason in floor.rejection_reasons:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        return {
            "scenes_total": len(self.scenes),
            "scenes_eligible": len(eligible_scenes),
            "scenes_rejected": len(self.scenes) - len(eligible_scenes),
            "floors_total": len(all_floors),
            "floors_eligible": len(eligible_floors),
            "floors_rejected": len(all_floors) - len(eligible_floors),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        }


def load_official_hssd_splits(path: Path) -> Dict[str, List[str]]:
    """Parse HSSD's deliberately simple train/val YAML without PyYAML."""
    result: Dict[str, List[str]] = {"train": [], "val": []}
    current: Optional[str] = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":") and not line.startswith("-"):
            current = line[:-1].strip()
            if current not in result:
                raise ValueError(f"Unsupported official HSSD split key {current!r}")
            continue
        if line.startswith("-") and current is not None:
            value = line[1:].strip().strip("'\"")
            if not value:
                raise ValueError("Empty HSSD scene ID in scene_splits.yaml")
            result[current].append(value)
            continue
        raise ValueError(f"Unsupported scene_splits.yaml line: {raw!r}")
    if not result["train"] or not result["val"]:
        raise ValueError("Official HSSD train/val splits must both be non-empty")
    flattened = result["train"] + result["val"]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Official HSSD train/val splits overlap or contain duplicates")
    return result


def build_hssd_split_manifest(
    official_splits: Dict[str, Sequence[str]],
    seed: int,
    internal_val_fraction: float = 0.10,
    available_scene_ids: Optional[Iterable[str]] = None,
) -> dict:
    if not 0.0 < float(internal_val_fraction) < 1.0:
        raise ValueError("internal_val_fraction must be strictly between 0 and 1")
    available = set(available_scene_ids) if available_scene_ids is not None else None
    official_train = sorted(
        scene for scene in official_splits["train"]
        if available is None or scene in available
    )
    official_val = sorted(
        scene for scene in official_splits["val"]
        if available is None or scene in available
    )
    if not official_train or not official_val:
        raise ValueError("Available HSSD data must include official train and val scenes")
    ranked_train = sorted(
        official_train, key=lambda scene: (stable_fraction(seed, scene), scene)
    )
    validation_count = max(1, int(round(len(ranked_train) * internal_val_fraction)))
    validation_count = min(validation_count, len(ranked_train) - 1)
    internal_val = sorted(ranked_train[:validation_count])
    formal_train = sorted(ranked_train[validation_count:])
    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "dataset_source": "hssd",
        "split_unit": "hssd_scene_id",
        "seed": int(seed),
        "internal_val_fraction": float(internal_val_fraction),
        "official_counts": {
            "train": len(official_train),
            "val": len(official_val),
        },
        "scene_splits": {
            "train": formal_train,
            "val": internal_val,
            "test": official_val,
        },
    }
    validate_hssd_split_manifest(manifest, official_splits)
    return manifest


def validate_hssd_split_manifest(
    manifest: dict, official_splits: Optional[Dict[str, Sequence[str]]] = None
) -> None:
    if manifest.get("dataset_source") != "hssd":
        raise ValueError("Split manifest must be HSSD-only")
    if manifest.get("split_unit") != "hssd_scene_id":
        raise ValueError("HSSD split unit must be the scene ID")
    splits = manifest.get("scene_splits", {})
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("HSSD manifest needs train, val, and test")
    flattened = [scene for values in splits.values() for scene in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("An HSSD scene leaks across formal splits")
    if official_splits is not None:
        official_train = set(official_splits["train"])
        official_val = set(official_splits["val"])
        if not set(splits["train"] + splits["val"]).issubset(official_train):
            raise ValueError("Formal train/internal-val contains non-train HSSD scenes")
        if not set(splits["test"]).issubset(official_val):
            raise ValueError("Formal test contains a non-val HSSD scene")


def load_split_manifest(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    validate_hssd_split_manifest(result)
    return result
