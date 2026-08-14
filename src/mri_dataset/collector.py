from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from .bev import compute_fov_overlap
from .objects import controlled_object_collision_free
from .sampling import build_robot_states, sample_robot_positions, sample_yaws
from .serialization import save_rendered_state, write_json
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
    robots = build_robot_states(positions, yaws, rng, config, deterministic_heights=heights)
    state = WorldState(
        schema_version="0.1.0", state_id=state_id, scene_id=backend.scene_id,
        floor_y=float(positions[0][1]), random_seed=seed, robots=robots,
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
            if not np.all(np.isfinite(point)) or abs(point[1] - state.floor_y) > config.floor_tolerance_m:
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
            candidate = backend.create_object_state(category, point[0], point[2], state.floor_y, index)
            state.objects = result + [candidate]
            if controlled_object_collision_free(candidate, state, backend.scene_bounds):
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


def layout_family(scene_id: str) -> str:
    # Preserve apt_N identity while grouping common rearrangement suffixes.
    match = re.match(r"^(apt_\d+)(?:[_-](?:rearrange|variant|v)\w*)?$", scene_id)
    return match.group(1) if match else scene_id


def initialize_dataset_root(root: Path, config) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(exist_ok=True)
    (root / "scenes").mkdir(exist_ok=True)
    (root / "interventions").mkdir(exist_ok=True)
    write_json(
        root / "dataset.json",
        {
            "schema_version": "0.1.0", "generator": "mri_dataset",
            "config": config.to_dict(), "split_unit": "apartment_layout_family",
            "states": [], "interventions": [],
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
        },
    )


def update_dataset_index(root: Path) -> None:
    dataset_path = root / "dataset.json"
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    state_paths = sorted(root.glob("scenes/*/states/*/state.json"))
    edit_paths = sorted(root.glob("interventions/*/edit_*.json"))
    dataset["states"] = [str(path.parent.relative_to(root)) for path in state_paths]
    dataset["interventions"] = [str(path.relative_to(root)) for path in edit_paths]
    write_json(dataset_path, dataset)
    families = {}
    for path in state_paths:
        scene_id = path.parents[2].name
        families.setdefault(layout_family(scene_id), []).append(str(path.parent.relative_to(root)))
    # A one-scene pilot is train-only. Future family lists can be explicitly
    # assigned without ever splitting images or variants of one family.
    train = sorted(item for states in families.values() for item in states)
    write_json(root / "splits/train.json", {"layout_families": sorted(families), "states": train})
    write_json(root / "splits/val.json", {"layout_families": [], "states": []})
    write_json(root / "splits/test.json", {"layout_families": [], "states": []})


def collect_level1(backend, config, root: Path, num_states: int, deterministic_debug: bool = False) -> List[Path]:
    if not (root / "dataset.json").exists():
        initialize_dataset_root(root, config)
    scene_dir = root / "scenes" / backend.scene_id
    (scene_dir / "states").mkdir(parents=True, exist_ok=True)
    write_json(
        scene_dir / "scene.json",
        {
            "scene_id": backend.scene_id,
            "layout_family": layout_family(backend.scene_id),
            "navmesh": str(Path(config.navmesh_root) / f"{backend.scene_id}.navmesh"),
            "bounds_world": [backend.scene_bounds[0].tolist(), backend.scene_bounds[1].tolist()],
        },
    )
    saved = []
    for index in range(1, num_states + 1):
        state_id = f"state_{index:06d}"
        state_dir = scene_dir / "states" / state_id
        state = make_world_state(
            backend, config, state_id, config.random_seed + index - 1,
            deterministic_debug=deterministic_debug and index == 1,
        )
        saved.append(save_rendered_state(backend, state, state_dir))
    update_dataset_index(root)
    return saved
