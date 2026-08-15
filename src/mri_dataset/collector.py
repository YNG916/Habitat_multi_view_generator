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

    factual_states = []
    derived_states = []
    factual_by_family = {}
    derived_by_family = {}
    for path in state_paths:
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        relative = str(path.parent.relative_to(root))
        origin = metadata.get("state_origin")
        if origin is None:
            origin = "intervention_derived" if metadata.get("parent_state_id") else "factual"
        family = layout_family(metadata.get("scene_id", path.parents[2].name))
        if origin == "factual":
            factual_states.append(relative)
            factual_by_family.setdefault(family, []).append(relative)
        elif origin == "intervention_derived":
            derived_states.append(relative)
            derived_by_family.setdefault(family, []).append(relative)
        else:
            raise ValueError(f"Unknown state_origin {origin!r} in {path}")

    interventions_by_family = {}
    for path in edit_paths:
        with path.open("r", encoding="utf-8") as handle:
            edit = json.load(handle)
        family = layout_family(edit["scene_id"])
        interventions_by_family.setdefault(family, []).append(
            str(path.relative_to(root))
        )

    dataset["states"] = factual_states + derived_states
    dataset["factual_states"] = factual_states
    dataset["intervention_derived_states"] = derived_states
    dataset["interventions"] = [str(path.relative_to(root)) for path in edit_paths]
    write_json(dataset_path, dataset)

    # A one-scene pilot is train-only. Keep task inputs and intervention targets
    # in separate fields so a Level-2 after-state is never silently consumed as
    # a factual Level-1 training state.
    families = sorted(set(factual_by_family) | set(derived_by_family))
    train = {
        "layout_families": families,
        "states": sorted(
            item for family in families for item in factual_by_family.get(family, [])
        ),
        "after_states": sorted(
            item for family in families for item in derived_by_family.get(family, [])
        ),
        "interventions": sorted(
            item
            for family in families
            for item in interventions_by_family.get(family, [])
        ),
    }
    write_json(root / "splits/train.json", train)
    empty_split = {
        "layout_families": [],
        "states": [],
        "after_states": [],
        "interventions": [],
    }
    write_json(root / "splits/val.json", empty_split)
    write_json(root / "splits/test.json", empty_split)


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
            "bounds_world": [backend.render_bev_bounds[0].tolist(), backend.render_bev_bounds[1].tolist()],
            "navmesh_bounds_world": [
                backend.navmesh_bounds[0].tolist(), backend.navmesh_bounds[1].tolist()
            ],
            "render_bev_bounds_world": [
                backend.render_bev_bounds[0].tolist(), backend.render_bev_bounds[1].tolist()
            ],
        },
    )
    saved = []
    max_attempts = int(config.max_state_sampling_attempts)
    if max_attempts < 1:
        raise ValueError("max_state_sampling_attempts must be at least 1")
    for index in range(1, num_states + 1):
        state_id = f"state_{index:06d}"
        state_dir = scene_dir / "states" / state_id
        use_debug_pose = deterministic_debug and index == 1
        last_error = None
        for attempt in range(1 if use_debug_pose else max_attempts):
            seed = (
                config.random_seed + index - 1
                if attempt == 0
                else config.random_seed + index - 1 + attempt * 1_000_003
            )
            try:
                state = make_world_state(
                    backend,
                    config,
                    state_id,
                    seed,
                    deterministic_debug=use_debug_pose,
                )
                break
            except (ValueError, RuntimeError) as exc:
                last_error = exc
        else:
            raise RuntimeError(
                f"Could not sample collision-free {state_id} after "
                f"{max_attempts} attempts: {last_error}"
            ) from last_error
        saved.append(save_rendered_state(backend, state, state_dir))
    update_dataset_index(root)
    return saved
