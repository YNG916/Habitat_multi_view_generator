from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np

from .bev import compute_fov_overlap
from .objects import controlled_object_collision_free
from .sampling import build_robot_states, sample_robot_positions, sample_yaws
from .protocol import protocol_descriptor, stable_seed
from .serialization import (
    save_rendered_state,
    state_directory_complete,
    write_json,
)
from .world_state import WorldState


def make_world_state(backend, config, state_id: str, seed: int, deterministic_debug: bool = False) -> WorldState:
    rng = np.random.default_rng(seed)
    backend.sim.pathfinder.seed(int(seed))
    positions = sample_robot_positions(
        backend.sim.pathfinder, rng, config.num_robots,
        config.min_obstacle_distance_m, config.min_inter_robot_distance_m,
        config.local_sampling_radius_m, config.floor_tolerance_m,
        allowed_island_ids=backend.region_spec.allowed_island_ids,
        representative_floor_y=backend.region_spec.representative_floor_y,
        region_spec=backend.region_spec,
    )
    yaws = sample_yaws(
        positions, rng, config.heading_mode, config.shared_heading_jitter_deg
    )
    heights = None
    surface_ys = np.asarray(
        [backend.floor_surface_y(point) for point in positions], dtype=np.float64
    )
    if float(np.ptp(surface_ys)) > config.floor_tolerance_m:
        raise ValueError("Robot samples resolve to inconsistent physical floor elevations")
    for point, surface_y in zip(positions, surface_ys):
        point[1] = surface_y
    floor_y = float(np.median(surface_ys))
    robots = build_robot_states(positions, yaws, rng, config, deterministic_heights=heights)
    state = WorldState(
        schema_version="0.4.0", state_id=state_id, scene_id=backend.scene_id,
        floor_y=floor_y, random_seed=seed, robots=robots,
        dataset_source="hssd",floor_id=backend.floor_id,region_id=backend.region_id,
        region_category=backend.region_category,bev_scope="semantic_region",
        region_context_margin_m=float(config.region_context_margin_m),
    )
    state.objects = sample_controlled_objects(backend, state, config, rng)
    state.overlap = compute_fov_overlap(backend.mapping, robots, config.hfov_deg)
    validate_sampled_state(backend, state, config)
    return state


def sample_object_choices(pools, count, rng):
    """Sample distinct categories, then one approved variant within each category."""
    count=int(count)
    if count<0:
        raise ValueError("Controlled-object count cannot be negative")
    categories=sorted(category for category,assets in pools.items() if assets)
    if count>len(categories):
        raise ValueError(
            f"Requested {count} controlled objects from only "
            f"{len(categories)} non-empty categories"
        )
    if count==0:
        return []
    indices=np.atleast_1d(rng.choice(len(categories),size=count,replace=False))
    result=[]
    for category_index in indices:
        category=categories[int(category_index)]
        assets=list(pools[category])
        result.append((category,assets[int(rng.integers(len(assets)))]))
    return result


def sample_controlled_objects(backend,state,config,rng)->list:
    maximum=int(config.controlled_objects_max_per_state); minimum=int(config.controlled_objects_min_per_state)
    if maximum<=0:return []
    target=int(rng.integers(minimum,maximum+1))
    choices=sample_object_choices(backend.controlled_handles,target,rng)
    anchor=np.asarray(state.robots[0].base_position_world)
    island=int(backend.sim.pathfinder.get_island(anchor))
    if island not in backend.region_spec.allowed_island_ids: raise ValueError("Object anchor outside selected region islands")
    result=[]
    for index,(category,asset) in enumerate(choices,start=1):
        for _ in range(400):
            point=np.asarray(backend.sim.pathfinder.get_random_navigable_point(100,island),dtype=np.float64)
            if not np.all(np.isfinite(point)) or not backend.point_in_region(point):continue
            if np.linalg.norm(point[[0,2]]-anchor[[0,2]])>min(2.5,config.local_sampling_radius_m):continue
            object_floor_y=backend.floor_surface_y(point)
            if abs(object_floor_y-state.floor_y)>config.floor_tolerance_m:continue
            robot_distances=[np.linalg.norm(point[[0,2]]-np.asarray(robot.base_position_world)[[0,2]]) for robot in state.robots]
            object_distances=[np.linalg.norm(point[[0,2]]-np.asarray(obj.position_world)[[0,2]]) for obj in result]
            if min(robot_distances)<config.controlled_object_min_separation_m:continue
            if object_distances and min(object_distances)<config.controlled_object_min_separation_m:continue
            candidate=backend.create_object_state(category,asset,point[0],point[2],object_floor_y,index)
            state.objects=result+[candidate]
            if controlled_object_collision_free(candidate,state,backend.render_bev_bounds) and backend.object_collision_free(state,candidate.instance_id):
                result.append(candidate);break
        else:raise RuntimeError(f"Could not place controlled object {index} in region")
    return result


def validate_sampled_state(backend, state, config) -> None:
    categories=[obj.category for obj in state.objects]
    if len(categories)!=len(set(categories)):
        raise ValueError("A WorldState contains duplicate controlled-object categories")
    pathfinder = backend.sim.pathfinder
    for robot in state.robots:
        point = np.asarray(robot.base_position_world)
        if not backend.point_in_region(point):
            raise ValueError(f"{robot.robot_id} is outside selected semantic region")
        if not pathfinder.is_navigable(point):
            raise ValueError(f"{robot.robot_id} is not navigable: {point}")
        if abs(point[1] - state.floor_y) > config.floor_tolerance_m:
            raise ValueError(f"{robot.robot_id} is on a different floor")
    for i, first in enumerate(state.robots):
        for second in state.robots[i + 1 :]:
            distance = np.linalg.norm(
                np.asarray(first.base_position_world)[[0, 2]]
                - np.asarray(second.base_position_world)[[0, 2]]
            )
            if distance < config.min_inter_robot_distance_m:
                raise ValueError(f"Robots violate separation: {distance:.3f} m")

    for robot in state.robots:
        collision = backend.entity_collision_report(state, robot.robot_id)
        if not collision["collision_free"]:
            raise ValueError(
                f"{robot.robot_id} proxy collision "
                f"{collision['rejected_contacts']}"
            )



def initialize_dataset_root(root: Path, config) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(exist_ok=True)
    (root / "scenes").mkdir(exist_ok=True)
    (root / "interventions").mkdir(exist_ok=True)
    dataset_path = root / "dataset.json"
    fingerprint = config.generation_fingerprint()
    if dataset_path.exists():
        with dataset_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        previous = existing.get("generation_fingerprint")
        if previous is None:
            raise ValueError(
                "Legacy output root has no formal generation fingerprint; "
                "use a new output_root for MRI Dataset v1"
            )
        if previous != fingerprint:
            raise ValueError(
                "Output root was created with a different generation config; "
                "use a new output_root instead of mixing dataset protocols"
            )
        return
    write_json(
        dataset_path,
        {
            "schema_version":config.dataset_version,
            "dataset_source": "hssd",
            "dataset_version": config.dataset_version,
            "generator": "mri_dataset",
            "generation_fingerprint": fingerprint,
            "config": config.to_dict(),
            "protocol": protocol_descriptor(config),
            "split_unit": "hssd_scene_id",
            "scene_splits": config.scene_splits,
            "scene_registry": config.scene_registry,
            "split_manifest": config.split_manifest,
            "states": [],
            "interventions": [],
        },
    )
    write_json(
        root / "categories.json",
        {
            "controlled_objects":[
                {"category":category,"canonical_hssd_asset_ids":assets}
                for category,assets in sorted(config.controlled_object_pools.items())
            ],
            "robot_proxies": ["red", "green", "blue"],
            "semantic_category_ids": config.semantic_category_ids,
            "semantic_scope": (
                "controlled entities only; native HSSD scene geometry is label 0"
            ),
        },
    )


def update_dataset_index(root: Path, require_referenced_after_states: bool = True) -> None:
    root = Path(root)
    dataset_path = root / "dataset.json"
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    state_paths = sorted(root.glob("scenes/*/floors/*/regions/*/states/*/state.json"))
    edit_paths = sorted(root.glob("interventions/*/*/*/edit_*.json"))

    configured_splits = dataset.get("scene_splits", {})
    scene_to_split = {
        scene: split
        for split, scenes in configured_splits.items()
        for scene in scenes
    }

    factual_states = []
    derived_states = []
    buckets = {
        split: {
            "scenes": set(),
            "states": [],
            "after_states": [],
            "interventions": [],
            "level2_by_regime": {},
        }
        for split in ("train", "val", "test")
    }

    for path in state_paths:
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        relative = str(path.parent.relative_to(root))
        scene_id = metadata.get("scene_id", path.parents[2].name)
        split = scene_to_split.get(scene_id, "train" if not scene_to_split else None)
        if split not in buckets:
            raise ValueError(f"Scene {scene_id!r} is not assigned to a dataset split")
        buckets[split]["scenes"].add(scene_id)
        origin = metadata.get("state_origin")
        if origin is None:
            origin = "intervention_derived" if metadata.get("parent_state_id") else "factual"
        if origin == "factual":
            factual_states.append(relative)
            buckets[split]["states"].append(relative)
        elif origin == "intervention_derived":
            derived_states.append(relative)
            buckets[split]["after_states"].append(relative)
        else:
            raise ValueError(f"Unknown state_origin {origin!r} in {path}")

    referenced_after_states = set()
    for path in edit_paths:
        with path.open("r", encoding="utf-8") as handle:
            edit = json.load(handle)
        relative = str(path.relative_to(root))
        scene_id = edit["scene_id"]
        split = edit.get("split") or scene_to_split.get(
            scene_id, "train" if not scene_to_split else None
        )
        if split not in buckets or split != scene_to_split.get(scene_id, split):
            raise ValueError(f"Intervention {path} has an invalid split assignment")
        regime = edit.get("benchmark_regime", "id")
        after_path = edit.get("after_state_path")
        before_path = edit.get("before_state_path")
        if after_path:
            referenced_after_states.add(after_path)
        buckets[split]["interventions"].append(relative)
        regime_bucket = buckets[split]["level2_by_regime"].setdefault(
            regime, {"after_states": [], "interventions": [], "pairs": []}
        )
        regime_bucket["interventions"].append(relative)
        if after_path:
            regime_bucket["after_states"].append(after_path)
        if before_path and after_path:
            regime_bucket["pairs"].append(
                {
                    "edit": relative,
                    "before_state": before_path,
                    "after_state": after_path,
                }
            )

    orphan_after = sorted(set(derived_states) - referenced_after_states)
    if orphan_after and require_referenced_after_states:
        raise ValueError(
            "Intervention-derived states are not referenced by an edit: "
            f"{orphan_after[:5]}"
        )

    dataset["states"] = factual_states + derived_states
    dataset["factual_states"] = factual_states
    dataset["intervention_derived_states"] = derived_states
    dataset["interventions"] = [str(path.relative_to(root)) for path in edit_paths]
    dataset["counts"] = {
        "factual_states": len(factual_states),
        "intervention_derived_states": len(derived_states),
        "interventions": len(edit_paths),
    }
    write_json(dataset_path, dataset)

    for split, payload in buckets.items():
        serializable = {
            "schema_version": "1.0.0",
            "split": split,
            "scenes": sorted(payload["scenes"]),
            "states": sorted(payload["states"]),
            "after_states": sorted(payload["after_states"]),
            "interventions": sorted(payload["interventions"]),
            "level2_by_regime": {},
        }
        for regime, regime_payload in sorted(payload["level2_by_regime"].items()):
            serializable["level2_by_regime"][regime] = {
                key: sorted(value, key=lambda item: json.dumps(item, sort_keys=True))
                for key, value in regime_payload.items()
            }
        write_json(root / f"splits/{split}.json", serializable)
        write_json(
            root / f"splits/level1_{split}.json",
            {
                "schema_version": "1.0.0",
                "split": split,
                "scenes": serializable["scenes"],
                "states": serializable["states"],
            },
        )
        regimes = set(dataset.get("protocol", {}).get(
            "intervention_regimes", {}
        )) | set(serializable["level2_by_regime"])
        for regime in sorted(regimes):
            regime_payload = serializable["level2_by_regime"].get(
                regime, {"after_states": [], "interventions": [], "pairs": []}
            )
            write_json(
                root / f"splits/level2_{split}_{regime}.json",
                {
                    "schema_version": "1.0.0",
                    "split": split,
                    "benchmark_regime": regime,
                    **regime_payload,
                },
            )


def collect_level1(
    backend,
    config,
    root: Path,
    num_states: int,
    deterministic_debug: bool = False,
) -> List[Path]:
    initialize_dataset_root(root, config)
    scene_dir = root / "scenes" / backend.scene_id
    floor_dir=scene_dir/"floors"/backend.floor_id
    region_dir=floor_dir/"regions"/backend.region_id
    (region_dir/"states").mkdir(parents=True,exist_ok=True)
    write_json(
        scene_dir / "scene.json",
        {
            "dataset_source": "hssd",
            "scene_id": backend.scene_id,
            "split": config.scene_split(backend.scene_id),
            "official_hssd_split": backend.scene_spec.official_split,
            "rendered_scene_aabb": backend.scene_spec.rendered_scene_aabb,
        },
    )
    write_json(
        floor_dir / "floor.json",
        {
            "dataset_source": "hssd",
            "scene_id": backend.scene_id,
            "floor_id": backend.floor_id,
            "split": config.scene_split(backend.scene_id),
            "cached_navmesh": backend.scene_spec.cached_navmesh_path,
            "navmesh_sha256": backend.scene_spec.navmesh_sha256,
            "navmesh_settings": backend.scene_spec.navmesh_settings,
            "allowed_island_ids": backend.floor_spec.allowed_island_ids,
            "representative_floor_y": backend.floor_spec.representative_floor_y,
            "navigable_area_m2": backend.floor_spec.navigable_area_m2,
            "navmesh_bounds_world": [
                backend.navmesh_bounds[0].tolist(), backend.navmesh_bounds[1].tolist()
            ],
            "render_bev_bounds_world": [
                backend.render_bev_bounds[0].tolist(), backend.render_bev_bounds[1].tolist()
            ],
            "bev_camera_height_m": backend.floor_spec.bev_camera_height_m,
        },
    )
    write_json(region_dir/"region.json",{
        "dataset_source":"hssd","scene_id":backend.scene_id,"floor_id":backend.floor_id,
        "region_id":backend.region_id,"region_category":backend.region_category,
        "split":config.scene_split(backend.scene_id),"semantic_polygon_world":backend.region_spec.semantic_polygon_world,
        "allowed_island_ids":backend.region_spec.allowed_island_ids,
        "navigable_area_m2":backend.region_spec.navigable_area_m2,
        "render_bev_bounds_world":[backend.render_bev_bounds[0].tolist(),backend.render_bev_bounds[1].tolist()],
        "bev_camera_height_m":backend.region_spec.bev_camera_height_m,
        "bev_scope":"semantic_region","region_context_margin_m":config.region_context_margin_m})
    saved = []
    skipped = []
    failures = Counter()
    max_attempts = int(config.max_state_sampling_attempts)
    for index in range(1, num_states + 1):
        state_id = f"state_{index:06d}"
        state_dir=region_dir/"states"/state_id
        if state_dir.exists():
            if config.resume and state_directory_complete(state_dir):
                skipped.append(state_dir)
                continue
            raise FileExistsError(
                f"Existing state is incomplete or resume is disabled: {state_dir}"
            )
        use_debug_pose = deterministic_debug and index == 1
        last_error = None
        for attempt in range(1 if use_debug_pose else max_attempts):
            seed = stable_seed(
                config.random_seed,
                config.protocol_version,
                backend.scene_id,
                backend.floor_id,
                backend.region_id,
                state_id,
                "level1",
                attempt,
            )
            try:
                state = make_world_state(
                    backend,
                    config,
                    state_id,
                    seed,
                    deterministic_debug=use_debug_pose,
                )
                saved.append(save_rendered_state(backend, state, state_dir))
                break
            except (ValueError, RuntimeError) as exc:
                last_error = exc
                failures[f"{type(exc).__name__}: {exc}"] += 1
        else:
            raise RuntimeError(
                f"Could not sample collision-free {state_id} after "
                f"{max_attempts} attempts: {last_error}"
            ) from last_error
    write_json(
        region_dir/"level1_collection_status.json",
        {
            "scene_id": backend.scene_id,
            "floor_id":backend.floor_id,
            "region_id":backend.region_id,
            "region_category":backend.region_category,
            "target_states": int(num_states),
            "new_states": len(saved),
            "resumed_states": len(skipped),
            "rejections": sum(failures.values()),
            "rejection_reasons": dict(failures.most_common()),
            "complete": len(saved) + len(skipped) == int(num_states),
        },
    )
    # An interrupted Level-2 slot may already have an atomic after-state but
    # not its edit JSON. Level-2 recovery will reconstruct that record.
    update_dataset_index(root,require_referenced_after_states=False)
    return saved
