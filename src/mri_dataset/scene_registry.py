"""Persistent HSSD scene/floor/region preprocessing registry and split manifests."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REGISTRY_SCHEMA_VERSION="2.1.0"
SPLIT_SCHEMA_VERSION="1.0.0"

def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def stable_fraction(seed:int,scene_id:str)->float:
    value=int.from_bytes(hashlib.sha256(f"{int(seed)}|{scene_id}".encode()).digest()[:8],"big")
    return value/float(2**64)

@dataclass(frozen=True)
class RegionSpec:
    region_id:str
    region_category:str
    floor_id:str
    representative_floor_y:float
    semantic_polygon_world:List[List[float]]
    allowed_island_ids:List[int]
    navigable_area_m2:float
    navigable_bounds_world:List[List[float]]
    visual_bev_bounds_world:List[List[float]]
    bev_camera_height_m:float
    eligible:bool
    rejection_reasons:List[str]=field(default_factory=list)
    preprocessing_validation:Dict[str,object]=field(default_factory=dict)

    @classmethod
    def from_dict(cls,data):
        return cls(
            region_id=str(data["region_id"]),region_category=str(data.get("region_category","unknown")),
            floor_id=str(data["floor_id"]),representative_floor_y=float(data["representative_floor_y"]),
            semantic_polygon_world=[[float(v) for v in point] for point in data["semantic_polygon_world"]],
            allowed_island_ids=[int(v) for v in data["allowed_island_ids"]],
            navigable_area_m2=float(data["navigable_area_m2"]),
            navigable_bounds_world=[[float(v) for v in point] for point in data["navigable_bounds_world"]],
            visual_bev_bounds_world=[[float(v) for v in point] for point in data["visual_bev_bounds_world"]],
            bev_camera_height_m=float(data["bev_camera_height_m"]),eligible=bool(data["eligible"]),
            rejection_reasons=list(map(str,data.get("rejection_reasons",[]))),
            preprocessing_validation=dict(data.get("preprocessing_validation",{})),
        )
    def validate(self):
        from .regions import polygon_area_xz
        if not self.region_id or not self.floor_id: raise ValueError("Region/floor IDs cannot be empty")
        if polygon_area_xz(self.semantic_polygon_world)<=0: raise ValueError(f"{self.region_id}: malformed polygon")
        polygon=np.asarray(self.semantic_polygon_world,dtype=float)
        low,high=self.visual_bev_bounds_world
        polygon_clipped=(
            np.any(polygon[:,0]<low[0]-1e-5) or np.any(polygon[:,0]>high[0]+1e-5)
            or np.any(polygon[:,2]<low[2]-1e-5) or np.any(polygon[:,2]>high[2]+1e-5)
        )
        if self.eligible and polygon_clipped:
            raise ValueError(f"{self.region_id}: eligible semantic polygon is clipped by BEV bounds")
        if not self.allowed_island_ids or len(self.allowed_island_ids)!=len(set(self.allowed_island_ids)): raise ValueError(f"{self.region_id}: invalid islands")
        if self.navigable_area_m2<=0 or self.bev_camera_height_m<=0: raise ValueError(f"{self.region_id}: invalid area/height")
        for bounds in (self.navigable_bounds_world,self.visual_bev_bounds_world):
            if len(bounds)!=2 or any(len(b)!=3 for b in bounds) or any(a>=b for a,b in zip(bounds[0],bounds[1])): raise ValueError(f"{self.region_id}: invalid bounds")
        if self.eligible and self.rejection_reasons: raise ValueError(f"{self.region_id}: eligible with rejections")

@dataclass(frozen=True)
class FloorSpec:
    floor_id:str
    representative_floor_y:float
    allowed_island_ids:List[int]
    navigable_area_m2:float
    navigable_bounds_world:List[List[float]]
    visual_bev_bounds_world:List[List[float]]
    bev_camera_height_m:float
    eligible:bool
    regions:List[RegionSpec]=field(default_factory=list)
    rejection_reasons:List[str]=field(default_factory=list)
    preprocessing_validation:Dict[str,object]=field(default_factory=dict)
    @classmethod
    def from_dict(cls,data):
        return cls(
            floor_id=str(data["floor_id"]),representative_floor_y=float(data["representative_floor_y"]),
            allowed_island_ids=[int(v) for v in data["allowed_island_ids"]],
            navigable_area_m2=float(data["navigable_area_m2"]),
            navigable_bounds_world=[[float(v) for v in b] for b in data["navigable_bounds_world"]],
            visual_bev_bounds_world=[[float(v) for v in b] for b in data["visual_bev_bounds_world"]],
            bev_camera_height_m=float(data["bev_camera_height_m"]),eligible=bool(data["eligible"]),
            regions=[RegionSpec.from_dict(item) for item in data.get("regions",[])],
            rejection_reasons=list(map(str,data.get("rejection_reasons",[]))),
            preprocessing_validation=dict(data.get("preprocessing_validation",{})),
        )
    def validate(self):
        if not self.floor_id or not self.allowed_island_ids: raise ValueError("Invalid floor")
        if len(self.allowed_island_ids)!=len(set(self.allowed_island_ids)) or self.navigable_area_m2<=0 or self.bev_camera_height_m<=0: raise ValueError(f"{self.floor_id}: invalid floor geometry")
        ids=[r.region_id for r in self.regions]
        if len(ids)!=len(set(ids)): raise ValueError(f"{self.floor_id}: duplicate region IDs")
        for region in self.regions:
            region.validate()
            if region.floor_id!=self.floor_id:
                raise ValueError(f"{region.region_id}: floor mismatch")
            if abs(region.representative_floor_y-self.representative_floor_y)>1e-4:
                raise ValueError(f"{region.region_id}: representative floor Y mismatch")
        if self.eligible!=bool(self.eligible_regions): raise ValueError(f"{self.floor_id}: eligibility must match regions")
        if self.eligible and self.rejection_reasons: raise ValueError(f"{self.floor_id}: eligible with rejections")
    def region(self,region_id,require_eligible=True):
        matches=[r for r in self.regions if r.region_id==region_id]
        if len(matches)!=1: raise KeyError(f"{self.floor_id}: no unique region {region_id}")
        if require_eligible and not matches[0].eligible: raise ValueError(f"Ineligible region {region_id}")
        return matches[0]
    @property
    def eligible_regions(self): return [r for r in self.regions if r.eligible]

@dataclass(frozen=True)
class SceneSpec:
    dataset_source:str
    scene_id:str
    official_split:str
    eligible:bool
    rejection_reasons:List[str]
    cached_navmesh_path:str
    navmesh_sha256:str
    navmesh_settings:Dict[str,object]
    navmesh_settings_fingerprint:str
    preprocessing_fingerprint:str
    semantic_regions_sha256:str
    rendered_scene_aabb:List[List[float]]
    floors:List[FloorSpec]
    preprocessing_validation:Dict[str,object]=field(default_factory=dict)
    @classmethod
    def from_dict(cls,data):
        return cls(
            dataset_source=str(data["dataset_source"]),scene_id=str(data["scene_id"]),
            official_split=str(data["official_split"]),eligible=bool(data["eligible"]),
            rejection_reasons=list(map(str,data.get("rejection_reasons",[]))),
            cached_navmesh_path=str(data["cached_navmesh_path"]),navmesh_sha256=str(data.get("navmesh_sha256","")),
            navmesh_settings=dict(data["navmesh_settings"]),navmesh_settings_fingerprint=str(data["navmesh_settings_fingerprint"]),
            preprocessing_fingerprint=str(data.get("preprocessing_fingerprint","")),
            semantic_regions_sha256=str(data.get("semantic_regions_sha256","")),
            rendered_scene_aabb=[[float(v) for v in b] for b in data["rendered_scene_aabb"]],
            floors=[FloorSpec.from_dict(item) for item in data.get("floors",[])],
            preprocessing_validation=dict(data.get("preprocessing_validation",{})),
        )
    def floor(self,floor_id,require_eligible=True):
        matches=[f for f in self.floors if f.floor_id==floor_id]
        if len(matches)!=1: raise KeyError(f"{self.scene_id}: no unique floor {floor_id}")
        if require_eligible and not matches[0].eligible: raise ValueError(f"Ineligible floor {floor_id}")
        return matches[0]
    @property
    def eligible_floors(self): return [f for f in self.floors if f.eligible]
    @property
    def eligible_regions(self): return [r for f in self.eligible_floors for r in f.eligible_regions]
    def validate(self):
        if self.dataset_source!="hssd" or self.official_split not in {"train","val"}: raise ValueError(f"{self.scene_id}: invalid HSSD scene")
        ids=[f.floor_id for f in self.floors]
        if len(ids)!=len(set(ids)): raise ValueError(f"{self.scene_id}: duplicate floors")
        for floor in self.floors: floor.validate()
        if self.eligible and (
            not self.preprocessing_fingerprint or not self.semantic_regions_sha256
        ):
            raise ValueError(f"{self.scene_id}: missing preprocessing/source fingerprint")
        if self.eligible!=bool(self.eligible_floors): raise ValueError(f"{self.scene_id}: eligibility mismatch")
        if self.eligible and self.rejection_reasons: raise ValueError(f"{self.scene_id}: eligible with rejections")

@dataclass
class SceneRegistry:
    dataset_source:str
    dataset_config_path:str
    official_splits_path:str
    scenes:List[SceneSpec]
    schema_version:str=REGISTRY_SCHEMA_VERSION
    preprocessing_config:Dict[str,object]=field(default_factory=dict)
    statistics:Dict[str,object]=field(default_factory=dict)
    @classmethod
    def from_dict(cls,data):
        registry=cls(schema_version=str(data.get("schema_version","")),dataset_source=str(data["dataset_source"]),
            dataset_config_path=str(data["dataset_config_path"]),official_splits_path=str(data["official_splits_path"]),
            preprocessing_config=dict(data.get("preprocessing_config",{})),
            scenes=[SceneSpec.from_dict(item) for item in data.get("scenes",[])],statistics=dict(data.get("statistics",{})))
        registry.validate(); return registry
    @classmethod
    def load(cls,path):
        return cls.from_dict(json.loads(Path(path).read_text()))
    def to_dict(self): return asdict(self)
    def save(self,path):
        from .serialization import write_json
        self.statistics=self.compute_statistics(); self.validate(); write_json(Path(path),self.to_dict())
    def validate(self):
        if self.schema_version!=REGISTRY_SCHEMA_VERSION: raise ValueError(f"Unsupported scene registry schema {self.schema_version}; rerun HSSD region preprocessing")
        if self.dataset_source!="hssd": raise ValueError("Registry must be HSSD")
        ids=[s.scene_id for s in self.scenes]
        if len(ids)!=len(set(ids)): raise ValueError("Duplicate scene IDs")
        for scene in self.scenes: scene.validate()
    def scene(self,scene_id,require_eligible=True):
        matches=[s for s in self.scenes if s.scene_id==scene_id]
        if len(matches)!=1: raise KeyError(f"No unique scene {scene_id}")
        if require_eligible and not matches[0].eligible: raise ValueError(f"Ineligible scene {scene_id}")
        return matches[0]
    def compute_statistics(self):
        scenes=[s for s in self.scenes if s.eligible]; floors=[f for s in self.scenes for f in s.floors]
        ef=[f for f in floors if f.eligible]; regions=[r for f in floors for r in f.regions]; er=[r for r in regions if r.eligible]
        reasons={}
        for scene in self.scenes:
            for reason in scene.rejection_reasons: reasons[reason]=reasons.get(reason,0)+1
            for floor in scene.floors:
                for reason in floor.rejection_reasons: reasons[reason]=reasons.get(reason,0)+1
                for region in floor.regions:
                    for reason in region.rejection_reasons: reasons[reason]=reasons.get(reason,0)+1
        return {"scenes_total":len(self.scenes),"scenes_eligible":len(scenes),"scenes_rejected":len(self.scenes)-len(scenes),
            "floors_total":len(floors),"floors_eligible":len(ef),"floors_rejected":len(floors)-len(ef),
            "regions_total":len(regions),"regions_eligible":len(er),"regions_rejected":len(regions)-len(er),
            "rejection_reason_counts":dict(sorted(reasons.items()))}

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
