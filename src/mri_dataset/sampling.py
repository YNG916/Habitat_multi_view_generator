from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from .coordinates import normalize_angle
from .world_state import RobotState


def _valid_position(pathfinder, point: np.ndarray, floor_y: float, min_obstacle: float, floor_tolerance: float) -> bool:
    return bool(
        np.all(np.isfinite(point))
        and pathfinder.is_navigable(point)
        and abs(float(point[1]) - floor_y) <= floor_tolerance
        and pathfinder.distance_to_closest_obstacle(point, 2.0) >= min_obstacle
    )


def sample_robot_positions(
    pathfinder,
    rng: np.random.Generator,
    num_robots: int,
    min_obstacle_distance_m: float,
    min_inter_robot_distance_m: float,
    local_sampling_radius_m: float,
    floor_tolerance_m: float,
    allowed_island_ids=None,
    representative_floor_y=None,
    max_tries: int = 1000,
) -> List[np.ndarray]:
    pathfinder.seed(int(rng.integers(0, 2**31 - 1)))
    allowed = (
        sorted(map(int, allowed_island_ids))
        if allowed_island_ids is not None
        else list(range(int(pathfinder.num_islands)))
    )
    if not allowed:
        raise ValueError("No allowed NavMesh islands for the selected floor")
    areas = np.asarray([pathfinder.island_area(index) for index in allowed], dtype=np.float64)
    probabilities = areas / areas.sum()
    anchor = None
    for _ in range(max_tries):
        island_id = allowed[int(rng.choice(len(allowed), p=probabilities))]
        candidate = np.asarray(
            pathfinder.get_random_navigable_point(100, island_id), dtype=np.float64
        )
        if not np.all(np.isfinite(candidate)):
            continue
        floor_y = (
            float(candidate[1])
            if representative_floor_y is None
            else float(representative_floor_y)
        )
        if (
            int(pathfinder.get_island(candidate)) in allowed
            and _valid_position(
                pathfinder, candidate, floor_y,
                min_obstacle_distance_m, floor_tolerance_m,
            )
        ):
            anchor = candidate
            break
    if anchor is None:
        raise RuntimeError("Could not sample a valid anchor robot position")
    island = int(pathfinder.get_island(anchor))
    positions = [anchor]
    for robot_index in range(1, num_robots):
        for _ in range(max_tries):
            point = np.asarray(
                pathfinder.get_random_navigable_point(100, island),
                dtype=np.float64,
            )
            if (
                not np.all(np.isfinite(point))
                or np.linalg.norm(point[[0, 2]] - anchor[[0, 2]])
                > local_sampling_radius_m
            ):
                continue
            if not _valid_position(
                pathfinder, point, float(anchor[1]),
                min_obstacle_distance_m, floor_tolerance_m,
            ):
                continue
            if int(pathfinder.get_island(point)) != island:
                continue
            distances = [np.linalg.norm(point[[0, 2]] - old[[0, 2]]) for old in positions]
            if min(distances) >= min_inter_robot_distance_m:
                positions.append(point)
                break
        else:
            raise RuntimeError(f"Failed clustered sampling for robot {robot_index + 1}")
    return positions


def sample_yaws(
    positions: Sequence[np.ndarray],
    rng: np.random.Generator,
    mode: str,
    shared_heading_jitter_deg: float,
) -> List[float]:
    selected_mode = mode
    if mode == "mixed":
        selected_mode = "shared_region" if rng.random() < 0.65 else "random"
    if selected_mode == "random":
        return [float(rng.uniform(-math.pi, math.pi)) for _ in positions]
    if selected_mode != "shared_region":
        raise ValueError(f"Unknown heading mode: {mode}")
    target = np.mean(np.asarray(positions)[:, [0, 2]], axis=0)
    target += rng.normal(0.0, 0.4, size=2)
    jitter = math.radians(shared_heading_jitter_deg)
    result = []
    for point in positions:
        dx = float(target[0] - point[0])
        dz = float(target[1] - point[2])
        # Forward(yaw) is [-sin(yaw), 0, -cos(yaw)].
        yaw = math.atan2(-dx, -dz) + float(rng.uniform(-jitter, jitter))
        result.append(normalize_angle(yaw))
    return result


def build_robot_states(positions, yaws, rng, config, deterministic_heights=None) -> List[RobotState]:
    height_variants = {
        float(key): value for key, value in config.robot_proxy_height_variants.items()
    }
    if deterministic_heights is not None:
        heights = list(map(float, deterministic_heights))
    elif height_variants:
        available = np.asarray(sorted(height_variants), dtype=np.float64)
        heights = available[
            rng.integers(0, len(available), size=len(positions))
        ].tolist()
    else:
        heights = rng.uniform(
            config.camera_height_min_m,
            config.camera_height_max_m,
            len(positions),
        )
    robots = []
    for index, (position, yaw, height) in enumerate(zip(positions, yaws, heights), start=1):
        if height_variants:
            matching = [
                candidate
                for candidate in height_variants
                if math.isclose(candidate, float(height), abs_tol=1e-9)
            ]
            if not matching:
                raise ValueError(
                    f"Camera height {height} has no embodiment-matched proxy mesh"
                )
            proxy = height_variants[matching[0]][index - 1]
        else:
            proxy = (
                config.robot_proxy_configs[index - 1]
                if index <= len(config.robot_proxy_configs)
                else ""
            )
        robots.append(
            RobotState.create(
                f"robot_{index:02d}", position.tolist(), yaw, float(height),
                config.width, config.height, config.hfov_deg, config.near, config.far,
                camera_forward_offset_m=config.robot_camera_forward_offset_m,
                proxy_asset_handle=proxy, proxy_semantic_id=1000 + index,
            )
        )
    return robots
