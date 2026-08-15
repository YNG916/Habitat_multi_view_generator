from __future__ import annotations

import json
import re
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


APT1_DEBUG_POSITIONS = np.array(
    [
        [-0.558784008026123, 0.11937291920185089, 2.502380847930908],
        [0.25762394070625305, 0.11937291920185089, 3.131887912750244],
        [3.1363348960876465, 0.11937291920185089, 3.8737664222717285],
    ],
    dtype=np.float64,
)
APT1_DEBUG_YAWS = [0.0, np.pi / 2.0, -np.pi / 2.0]
APT1_DEBUG_HEIGHTS = [0.6, 0.9, 1.2]


def make_world_state(backend, config, state_id: str, seed: int, deterministic_debug: bool = False) -> WorldState:
    rng = np.random.default_rng(seed)
    backend.sim.pathfinder.seed(int(seed))
    if deterministic_debug:
        if backend.scene_id != "apt_1" or config.num_robots != 3:
            raise ValueError("The preserved deterministic debug pose is defined for apt_1 with 3 robots")
        positions = [point.copy() for point in APT1_DEBUG_POSITIONS]
        yaws = APT1_DEBUG_YAWS
        heights = APT1_DEBUG_HEIGHTS
    else:
        positions = sample_robot_positions(
            backend.sim.pathfinder, rng, config.num_robots,
            config.min_obstacle_distance_m, config.min_inter_robot_distance_m,
            config.local_sampling_radius_m, config.floor_tolerance_m,
        )
        yaws = sample_yaws(positions, rng, config.heading_mode, config.shared_heading_jitter_deg)
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
        schema_version="0.1.0", state_id=state_id, scene_id=backend.scene_id,
        floor_y=floor_y, random_seed=seed, robots=robots,
    )
    state.objects = sample_controlled_objects(backend, state, config, rng)
    state.overlap = compute_fov_overlap(backend.mapping, robots, config.hfov_deg)
    validate_sampled_state(backend, state, config)
    return state


def sample_controlled_objects(backend, state, config, rng) -> list:
    if config.controlled_objects_per_state <= 0:
        return []
    categories = sorted(backend.controlled_handles)
    anchor = np.asarray(state.robots[0].base_position_world)
    island = int(backend.sim.pathfinder.get_island(anchor))
    result = []
    for index in range(1, config.controlled_objects_per_state + 1):
        category = categories[int(rng.integers(0, len(categories)))]
        for _ in range(400):
            point = np.asarray(
                backend.sim.pathfinder.get_random_navigable_point_near(
                    anchor, min(2.5, config.local_sampling_radius_m), 100, island
                ),
                dtype=np.float64,
            )
            if not np.all(np.isfinite(point)):
                continue
            object_floor_y = backend.floor_surface_y(point)
            if abs(object_floor_y - state.floor_y) > config.floor_tolerance_m:
                continue
            robot_distances = [
                np.linalg.norm(point[[0, 2]] - np.asarray(robot.base_position_world)[[0, 2]])
                for robot in state.robots
            ]
            object_distances = [
                np.linalg.norm(point[[0, 2]] - np.asarray(obj.position_world)[[0, 2]])
                for obj in result
            ]
            if min(robot_distances) < config.controlled_object_min_separation_m:
                continue
            if object_distances and min(object_distances) < config.controlled_object_min_separation_m:
                continue
            candidate = backend.create_object_state(category, point[0], point[2], object_floor_y, index)
            state.objects = result + [candidate]
            if (
                controlled_object_collision_free(candidate, state, backend.render_bev_bounds)
                and backend.object_collision_free(state, candidate.instance_id)
            ):
                result.append(candidate)
                break
        else:
            raise RuntimeError(f"Could not place controlled object {index} collision-free")
    return result


def validate_sampled_state(backend, state, config) -> None:
    pathfinder = backend.sim.pathfinder
    for robot in state.robots:
        point = np.asarray(robot.base_position_world)
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



def layout_family(scene_id: str, explicit=None) -> str:
    """Return rendered-stage identity, grouping furniture rearrangements."""
    explicit = explicit or {}
    if explicit.get(scene_id):
        return explicit[scene_id]
    if re.match(r"^apt_\d+$", scene_id):
        return "frl_apartment_stage"
    match = re.match(r"^(v3_sc\d+)_staging_\d+$", scene_id)
    return match.group(1) if match else scene_id


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
            "schema_version": "1.0.0",
            "dataset_version": config.dataset_version,
            "generator": "mri_dataset",
            "generation_fingerprint": fingerprint,
            "config": config.to_dict(),
            "protocol": protocol_descriptor(config),
            "split_unit": "replicacad_macro_furniture_layout_family",
            "scene_splits": config.scene_splits,
            "states": [],
            "interventions": [],
        },
    )
    write_json(
        root / "categories.json",
        {
            "controlled_objects": [
                {"category": category, "template_suffix": suffix}
                for category, suffix in config.controlled_object_whitelist.items()
            ],
            "robot_proxies": ["red", "green", "blue"],
            "semantic_category_ids": config.semantic_category_ids,
            "semantic_scope": (
                "controlled entities only; ReplicaCAD stage/furniture is label 0"
            ),
        },
    )


def update_dataset_index(root: Path) -> None:
    root = Path(root)
    dataset_path = root / "dataset.json"
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    state_paths = sorted(root.glob("scenes/*/states/*/state.json"))
    edit_paths = sorted(root.glob("interventions/*/edit_*.json"))

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
            "layout_families": set(),
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
        family = layout_family(
            scene_id, dataset.get("config", {}).get("scene_layout_families", {})
        )
        buckets[split]["layout_families"].add(family)
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
    if orphan_after:
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
            "layout_families": sorted(payload["layout_families"]),
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
                "layout_families": serializable["layout_families"],
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
    (scene_dir / "states").mkdir(parents=True, exist_ok=True)
    write_json(
        scene_dir / "scene.json",
        {
            "scene_id": backend.scene_id,
            "layout_family": config.layout_family(backend.scene_id),
            "split": config.scene_split(backend.scene_id),
            "navmesh": str(Path(config.navmesh_root) / f"{backend.scene_id}.navmesh"),
            "bounds_world": [
                backend.render_bev_bounds[0].tolist(),
                backend.render_bev_bounds[1].tolist(),
            ],
            "navmesh_bounds_world": [
                backend.navmesh_bounds[0].tolist(),
                backend.navmesh_bounds[1].tolist(),
            ],
            "render_bev_bounds_world": [
                backend.render_bev_bounds[0].tolist(),
                backend.render_bev_bounds[1].tolist(),
            ],
        },
    )
    saved = []
    skipped = []
    failures = Counter()
    max_attempts = int(config.max_state_sampling_attempts)
    for index in range(1, num_states + 1):
        state_id = f"state_{index:06d}"
        state_dir = scene_dir / "states" / state_id
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
        scene_dir / "level1_collection_status.json",
        {
            "scene_id": backend.scene_id,
            "target_states": int(num_states),
            "new_states": len(saved),
            "resumed_states": len(skipped),
            "rejections": sum(failures.values()),
            "rejection_reasons": dict(failures.most_common()),
            "complete": len(saved) + len(skipped) == int(num_states),
        },
    )
    update_dataset_index(root)
    return saved
