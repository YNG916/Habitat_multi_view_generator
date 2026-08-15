"""Offline HSSD scene preprocessing for robot-specific floor-local collection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .bev import habitat_orthographic_depth_to_metric
from .protocol import stable_seed
from .scene_registry import (
    FloorSpec,
    SceneRegistry,
    SceneSpec,
    build_hssd_split_manifest,
    load_official_hssd_splits,
    sha256_file,
)
from .serialization import write_json


def navmesh_settings_dict(config) -> dict:
    return {
        "agent_radius_m": float(config.navmesh_agent_radius_m),
        "agent_height_m": float(config.navmesh_agent_height_m),
        "agent_max_climb_m": float(config.navmesh_agent_max_climb_m),
        "agent_max_slope_deg": float(config.navmesh_agent_max_slope_deg),
        "include_static_objects": True,
    }


def navmesh_settings_fingerprint(settings: dict) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def habitat_navmesh_settings(config):
    import habitat_sim

    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = float(config.navmesh_agent_radius_m)
    settings.agent_height = float(config.navmesh_agent_height_m)
    settings.agent_max_climb = float(config.navmesh_agent_max_climb_m)
    settings.agent_max_slope = float(config.navmesh_agent_max_slope_deg)
    settings.include_static_objects = True
    return settings


def discover_installed_hssd_scenes(dataset_config_path: Path) -> List[str]:
    dataset_root = Path(dataset_config_path).resolve().parent
    scene_root = dataset_root / "scenes"
    if not scene_root.is_dir():
        return []
    suffix = ".scene_instance.json"
    return sorted(
        path.name[: -len(suffix)]
        for path in scene_root.glob(f"*{suffix}")
        if path.is_file()
    )


def _create_simulator(config, scene_id: str, agent_config=None):
    import habitat_sim

    simulator = habitat_sim.SimulatorConfiguration()
    simulator.scene_dataset_config_file = str(config.dataset_config_path)
    simulator.scene_id = str(scene_id)
    simulator.enable_physics = True
    simulator.gpu_device_id = int(config.gpu_device_id)
    if agent_config is None:
        agent_config = habitat_sim.agent.AgentConfiguration()
        agent_config.sensor_specifications = []
    return habitat_sim.Simulator(
        habitat_sim.Configuration(simulator, [agent_config])
    )


def _sample_island_points(
    pathfinder,
    island_id: int,
    count: int,
    seed: int,
) -> np.ndarray:
    pathfinder.seed(int(seed))
    points = []
    for _ in range(int(count)):
        point = np.asarray(
            pathfinder.get_random_navigable_point(100, int(island_id)),
            dtype=np.float64,
        )
        if np.all(np.isfinite(point)) and int(pathfinder.get_island(point)) == int(
            island_id
        ):
            points.append(point)
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _group_islands_by_floor(
    island_records: Sequence[dict], tolerance_m: float
) -> List[List[dict]]:
    groups: List[List[dict]] = []
    for record in sorted(island_records, key=lambda item: item["median_y"]):
        matching = None
        for group in groups:
            weighted_y = sum(
                item["median_y"] * item["area_m2"] for item in group
            ) / sum(item["area_m2"] for item in group)
            if abs(record["median_y"] - weighted_y) <= float(tolerance_m):
                matching = group
                break
        if matching is None:
            groups.append([record])
        else:
            matching.append(record)
    return groups


def _bounds_from_points(points: np.ndarray) -> List[List[float]]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    for axis in range(3):
        if maximum[axis] - minimum[axis] < 1e-3:
            minimum[axis] -= 5e-4
            maximum[axis] += 5e-4
    return [minimum.tolist(), maximum.tolist()]


def _visual_bounds(
    navigable_bounds: Sequence[Sequence[float]],
    scene_aabb: Sequence[Sequence[float]],
    margin_m: float,
) -> List[List[float]]:
    nav_low, nav_high = map(np.asarray, navigable_bounds)
    scene_low, scene_high = map(np.asarray, scene_aabb)
    low = scene_low.copy()
    high = scene_high.copy()
    low[0] = max(scene_low[0], nav_low[0] - float(margin_m))
    low[2] = max(scene_low[2], nav_low[2] - float(margin_m))
    high[0] = min(scene_high[0], nav_high[0] + float(margin_m))
    high[2] = min(scene_high[2], nav_high[2] + float(margin_m))
    if high[0] - low[0] < 0.5 or high[2] - low[2] < 0.5:
        raise ValueError("floor visual BEV bounds are degenerate")
    return [low.tolist(), high.tolist()]


def _xz_overlap_ratio(first: FloorSpec, second: FloorSpec) -> float:
    a_low, a_high = map(np.asarray, first.visual_bev_bounds_world)
    b_low, b_high = map(np.asarray, second.visual_bev_bounds_world)
    width = max(0.0, min(a_high[0], b_high[0]) - max(a_low[0], b_low[0]))
    depth = max(0.0, min(a_high[2], b_high[2]) - max(a_low[2], b_low[2]))
    intersection = width * depth
    a_area = (a_high[0] - a_low[0]) * (a_high[2] - a_low[2])
    b_area = (b_high[0] - b_low[0]) * (b_high[2] - b_low[2])
    return float(intersection / max(1e-9, min(a_area, b_area)))


def _estimate_bev_camera_height(
    sim,
    points: np.ndarray,
    floor_y: float,
    scene_aabb: Sequence[Sequence[float]],
    config,
    override: Optional[float] = None,
) -> Tuple[float, dict]:
    if override is not None:
        chosen = float(override)
        return chosen, {
            "method": "explicit_hssd_preprocess_override",
            "chosen_height_m": chosen,
        }

    import habitat_sim

    candidates = []
    sampled = points[
        np.linspace(
            0,
            max(0, len(points) - 1),
            min(int(config.bev_ceiling_ray_samples), len(points)),
            dtype=np.int64,
        )
    ]
    minimum_ceiling = float(config.bev_min_ceiling_height_m)
    maximum_ray = float(config.bev_ceiling_ray_max_distance_m)
    for point in sampled:
        origin = np.asarray(point, dtype=np.float64).copy()
        origin[1] = float(floor_y) + 0.05
        ray = habitat_sim.geo.Ray(
            origin.astype(np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        result = sim.cast_ray(ray, max_distance=maximum_ray, buffer_distance=0.0)
        for hit in result.hits:
            clearance = 0.05 + float(hit.ray_distance)
            if clearance >= minimum_ceiling:
                candidates.append(clearance)
                break

    preferred = float(config.bev_preferred_camera_height_m)
    clearance = float(config.bev_ceiling_clearance_m)
    scene_top = float(scene_aabb[1][1]) - float(floor_y)
    if candidates:
        ceiling_estimate = float(np.percentile(candidates, 20))
        chosen = min(preferred, ceiling_estimate - clearance)
        method = "upward_bullet_ceiling_p20"
    else:
        ceiling_estimate = None
        chosen = max(preferred, scene_top + float(config.bev_open_scene_margin_m))
        method = "open_scene_above_rendered_contents"
    maximum = float(config.bev_far) - 0.25
    chosen = min(chosen, maximum)
    report = {
        "method": method,
        "rays_sampled": len(sampled),
        "valid_ceiling_hits": len(candidates),
        "ceiling_clearance_samples_m": candidates,
        "ceiling_estimate_m": ceiling_estimate,
        "scene_top_above_floor_m": scene_top,
        "preferred_height_m": preferred,
        "chosen_height_m": chosen,
    }
    if chosen <= max(0.5, float(config.camera_height_max_m) + 0.1):
        raise ValueError(
            f"no safe BEV camera height: chosen {chosen:.3f} m for floor {floor_y:.3f}"
        )
    return float(chosen), report


def _bev_sanity_check(
    config,
    scene_id: str,
    floor: FloorSpec,
    preview_path: Optional[Path] = None,
) -> dict:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis

    bounds = floor.visual_bev_bounds_world
    x_extent = float(bounds[1][0] - bounds[0][0])
    z_extent = float(bounds[1][2] - bounds[0][2])
    width = int(config.preprocess_bev_width)
    height = max(32, int(round(width * z_extent / x_extent)))
    specs = []
    for uuid, sensor_type in (
        ("preprocess_rgb", habitat_sim.SensorType.COLOR),
        ("preprocess_depth", habitat_sim.SensorType.DEPTH),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        spec.resolution = [height, width]
        spec.position = [0.0, 0.0, 0.0]
        spec.orientation = [0.0, 0.0, 0.0]
        spec.near = float(config.bev_near)
        spec.far = float(config.bev_far)
        spec.ortho_scale = 1.0 / x_extent
        specs.append(spec)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = specs
    sim = _create_simulator(config, scene_id, agent)
    try:
        state = habitat_sim.AgentState()
        state.position = np.array(
            [
                0.5 * (bounds[0][0] + bounds[1][0]),
                floor.representative_floor_y + floor.bev_camera_height_m,
                0.5 * (bounds[0][2] + bounds[1][2]),
            ],
            dtype=np.float32,
        )
        state.rotation = quat_from_angle_axis(
            -math.pi / 2.0, np.array([1.0, 0.0, 0.0])
        )
        sim.get_agent(0).set_state(state, infer_sensor_states=True)
        observations = sim.get_sensor_observations(agent_ids=[0])[0]
        rgb = np.asarray(observations["preprocess_rgb"])[..., :3].astype(np.uint8)
        raw_depth = np.asarray(
            observations["preprocess_depth"], dtype=np.float32
        )
        metric = habitat_orthographic_depth_to_metric(
            raw_depth, config.bev_near, config.bev_far
        )
        finite_fraction = float(np.isfinite(metric).mean())
        rgb_std = float(rgb.astype(np.float32).std())
        passed = bool(
            finite_fraction >= float(config.preprocess_bev_min_finite_fraction)
            and rgb_std >= float(config.preprocess_bev_min_rgb_std)
        )
        if preview_path is not None:
            Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(preview_path)
        return {
            "passed": passed,
            "resolution_hw": [height, width],
            "finite_metric_depth_fraction": finite_fraction,
            "rgb_std": rgb_std,
            "preview_path": str(preview_path) if preview_path else None,
        }
    finally:
        sim.close()


def _relative_to_repo(config, path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(config.repo_root))
    except ValueError:
        return str(path)


def preprocess_hssd_scene(
    config,
    scene_id: str,
    official_split: str,
    preview_root: Optional[Path] = None,
) -> SceneSpec:
    settings_payload = navmesh_settings_dict(config)
    settings_fingerprint = navmesh_settings_fingerprint(settings_payload)
    navmesh_path = config.navmesh_cache_path(scene_id)
    planned_path = _relative_to_repo(config, navmesh_path)
    scene_file = (
        config.dataset_config_path.parent
        / "scenes"
        / f"{scene_id}.scene_instance.json"
    )
    if not scene_file.is_file():
        return SceneSpec(
            dataset_source="hssd",
            scene_id=scene_id,
            official_split=official_split,
            eligible=False,
            rejection_reasons=["missing_scene_instance"],
            cached_navmesh_path=planned_path,
            navmesh_sha256="",
            navmesh_settings=settings_payload,
            navmesh_settings_fingerprint=settings_fingerprint,
            rendered_scene_aabb=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            floors=[],
            preprocessing_validation={"scene_file": str(scene_file)},
        )

    sim = None
    try:
        sim = _create_simulator(config, scene_id)
        aabb = sim.scene_aabb
        scene_aabb = [
            np.asarray(aabb.min, dtype=np.float64).tolist(),
            np.asarray(aabb.max, dtype=np.float64).tolist(),
        ]
        settings = habitat_navmesh_settings(config)
        if not sim.recompute_navmesh(sim.pathfinder, settings):
            raise RuntimeError("Habitat recompute_navmesh returned false")
        if not sim.pathfinder.is_loaded or sim.pathfinder.num_islands < 1:
            raise RuntimeError("recomputed NavMesh is empty")
        navmesh_path.parent.mkdir(parents=True, exist_ok=True)
        if not sim.pathfinder.save_nav_mesh(str(navmesh_path)):
            raise RuntimeError("Habitat failed to save robot-specific NavMesh")

        island_records = []
        rejected_islands = []
        for island_id in range(int(sim.pathfinder.num_islands)):
            area = float(sim.pathfinder.island_area(island_id))
            if area < float(config.floor_min_island_area_m2):
                rejected_islands.append(
                    {"island_id": island_id, "reason": "island_too_small", "area_m2": area}
                )
                continue
            points = _sample_island_points(
                sim.pathfinder,
                island_id,
                int(config.floor_samples_per_island),
                stable_seed(config.random_seed, "hssd_preprocess", scene_id, island_id),
            )
            if len(points) < int(config.floor_min_samples_per_island):
                rejected_islands.append(
                    {
                        "island_id": island_id,
                        "reason": "insufficient_island_samples",
                        "sample_count": len(points),
                    }
                )
                continue
            y05, median_y, y95 = np.percentile(points[:, 1], [5, 50, 95])
            vertical_span = float(y95 - y05)
            if vertical_span > float(config.floor_max_vertical_span_m):
                rejected_islands.append(
                    {
                        "island_id": island_id,
                        "reason": "connected_multilevel_island",
                        "vertical_span_m": vertical_span,
                    }
                )
                continue
            island_records.append(
                {
                    "island_id": island_id,
                    "area_m2": area,
                    "median_y": float(median_y),
                    "vertical_span_m": vertical_span,
                    "points": points,
                }
            )

        groups = _group_islands_by_floor(
            island_records, float(config.floor_group_tolerance_m)
        )
        floor_specs = []
        scene_override = config.hssd_preprocess_overrides().get(scene_id, {})
        for floor_index, group in enumerate(groups):
            floor_id = f"floor_{floor_index:02d}"
            points = np.concatenate([item["points"] for item in group], axis=0)
            area = float(sum(item["area_m2"] for item in group))
            representative_y = float(
                sum(item["median_y"] * item["area_m2"] for item in group)
                / max(area, 1e-9)
            )
            nav_bounds = _bounds_from_points(points)
            visual_bounds = _visual_bounds(
                nav_bounds, scene_aabb, float(config.bev_context_margin_m)
            )
            reasons = []
            if area < float(config.floor_min_navigable_area_m2):
                reasons.append("floor_navigable_area_too_small")
            override = scene_override.get(floor_id, scene_override)
            camera_override = (
                override.get("bev_camera_height_m")
                if isinstance(override, dict)
                else None
            )
            try:
                bev_height, height_report = _estimate_bev_camera_height(
                    sim,
                    points,
                    representative_y,
                    scene_aabb,
                    config,
                    camera_override,
                )
            except (RuntimeError, ValueError) as exc:
                bev_height = float(config.bev_preferred_camera_height_m)
                height_report = {"passed": False, "error": str(exc)}
                reasons.append("unsafe_bev_camera_height")
            floor_specs.append(
                FloorSpec(
                    floor_id=floor_id,
                    representative_floor_y=representative_y,
                    allowed_island_ids=sorted(item["island_id"] for item in group),
                    navigable_area_m2=area,
                    navigable_bounds_world=nav_bounds,
                    visual_bev_bounds_world=visual_bounds,
                    bev_camera_height_m=bev_height,
                    eligible=not reasons,
                    rejection_reasons=reasons,
                    preprocessing_validation={
                        "islands": [
                            {
                                key: value
                                for key, value in item.items()
                                if key != "points"
                            }
                            for item in group
                        ],
                        "sample_count": len(points),
                        "bev_camera_height": height_report,
                    },
                )
            )
    except Exception as exc:
        if sim is not None:
            sim.close()
        return SceneSpec(
            dataset_source="hssd",
            scene_id=scene_id,
            official_split=official_split,
            eligible=False,
            rejection_reasons=[f"preprocess_failure:{type(exc).__name__}:{exc}"],
            cached_navmesh_path=planned_path,
            navmesh_sha256="",
            navmesh_settings=settings_payload,
            navmesh_settings_fingerprint=settings_fingerprint,
            rendered_scene_aabb=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            floors=[],
            preprocessing_validation={"scene_file": str(scene_file)},
        )
    finally:
        if sim is not None:
            sim.close()

    ambiguous = set()
    for index, first in enumerate(floor_specs):
        for second in floor_specs[index + 1 :]:
            vertical = abs(
                first.representative_floor_y - second.representative_floor_y
            )
            overlap = _xz_overlap_ratio(first, second)
            if (
                vertical >= float(config.floor_ambiguity_min_vertical_separation_m)
                and overlap >= float(config.floor_ambiguity_max_xz_overlap_ratio)
            ):
                ambiguous.update([first.floor_id, second.floor_id])
    if ambiguous:
        floor_specs = [
            replace(
                floor,
                eligible=False,
                rejection_reasons=[
                    *floor.rejection_reasons,
                    "ambiguous_overlapping_multifloor_geometry",
                ],
            )
            if floor.floor_id in ambiguous
            else floor
            for floor in floor_specs
        ]

    checked_floors = []
    for floor in floor_specs:
        if not floor.eligible:
            checked_floors.append(floor)
            continue
        preview = (
            Path(preview_root) / f"{scene_id}_{floor.floor_id}_bev.png"
            if preview_root is not None
            else None
        )
        try:
            sanity = _bev_sanity_check(config, scene_id, floor, preview)
            reasons = [] if sanity["passed"] else ["low_resolution_bev_sanity_failed"]
        except Exception as exc:
            sanity = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
            reasons = ["low_resolution_bev_sanity_failed"]
        validation = dict(floor.preprocessing_validation)
        validation["low_resolution_bev"] = sanity
        checked_floors.append(
            replace(
                floor,
                eligible=not reasons,
                rejection_reasons=[*floor.rejection_reasons, *reasons],
                preprocessing_validation=validation,
            )
        )

    scene_eligible = any(floor.eligible for floor in checked_floors)
    scene_reasons = [] if scene_eligible else ["no_eligible_floor"]
    return SceneSpec(
        dataset_source="hssd",
        scene_id=scene_id,
        official_split=official_split,
        eligible=scene_eligible,
        rejection_reasons=scene_reasons,
        cached_navmesh_path=planned_path,
        navmesh_sha256=sha256_file(navmesh_path) if navmesh_path.is_file() else "",
        navmesh_settings=settings_payload,
        navmesh_settings_fingerprint=settings_fingerprint,
        rendered_scene_aabb=scene_aabb,
        floors=checked_floors,
        preprocessing_validation={
            "scene_file": str(scene_file),
            "navmesh_num_islands": len(island_records) + len(rejected_islands),
            "rejected_islands": rejected_islands,
        },
    )


def preprocess_hssd(
    config,
    scene_ids: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    preview_root: Optional[Path] = None,
) -> SceneRegistry:
    if config.dataset_source != "hssd":
        raise ValueError("HSSD preprocessing refuses a non-HSSD config")
    if not config.dataset_config_path.is_file():
        raise FileNotFoundError(config.dataset_config_path)
    if not config.official_scene_splits_path.is_file():
        raise FileNotFoundError(config.official_scene_splits_path)

    official = load_official_hssd_splits(config.official_scene_splits_path)
    installed = discover_installed_hssd_scenes(config.dataset_config_path)
    official_lookup = {
        scene: split for split, scenes in official.items() for scene in scenes
    }
    selected = sorted(
        set(scene_ids) if scene_ids is not None else set(official_lookup) & set(installed)
    )
    unknown = sorted(set(selected) - set(official_lookup))
    if unknown:
        raise ValueError(f"Requested scenes are absent from official HSSD splits: {unknown}")
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError("No installed official HSSD scenes were selected")

    existing = {}
    if config.scene_registry_path.is_file():
        try:
            existing = {
                scene.scene_id: scene
                for scene in SceneRegistry.load(config.scene_registry_path).scenes
            }
        except (OSError, ValueError):
            existing = {}
    expected_fingerprint = navmesh_settings_fingerprint(
        navmesh_settings_dict(config)
    )
    scenes = []
    for index, scene_id in enumerate(selected, start=1):
        cached = existing.get(scene_id)
        cached_path = (
            config.resolve(cached.cached_navmesh_path) if cached is not None else None
        )
        reusable = bool(
            cached is not None
            and cached.official_split == official_lookup[scene_id]
            and cached.navmesh_settings_fingerprint == expected_fingerprint
            and cached_path is not None
            and cached_path.is_file()
            and cached.navmesh_sha256 == sha256_file(cached_path)
        )
        action = "reusing" if reusable else "preprocessing"
        print(
            f"[{index}/{len(selected)}] {action} HSSD scene {scene_id}",
            flush=True,
        )
        scene = (
            cached
            if reusable
            else preprocess_hssd_scene(
                config, scene_id, official_lookup[scene_id], preview_root
            )
        )
        scenes.append(scene)
        partial = SceneRegistry(
            dataset_source="hssd",
            dataset_config_path=_relative_to_repo(config, config.dataset_config_path),
            official_splits_path=_relative_to_repo(config, config.official_scene_splits_path),
            preprocessing_config={
                **navmesh_settings_dict(config),
                "floor_group_tolerance_m": float(config.floor_group_tolerance_m),
                "floor_max_vertical_span_m": float(config.floor_max_vertical_span_m),
                "bev_context_margin_m": float(config.bev_context_margin_m),
            },
            scenes=scenes,
        )
        partial.save(config.scene_registry_path)

    manifest = build_hssd_split_manifest(
        official,
        int(config.split_seed),
        float(config.internal_val_fraction),
        available_scene_ids=installed,
    )
    manifest["official_splits_sha256"] = sha256_file(
        config.official_scene_splits_path
    )
    write_json(config.split_manifest_path, manifest)
    registry = SceneRegistry.load(config.scene_registry_path)
    return registry
