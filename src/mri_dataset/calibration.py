from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .bev import habitat_orthographic_depth_to_metric
from .config import REPO_ROOT
from .habitat_backend import HabitatBackend
from .objects import handles_by_suffix


CALIBRATION_HEIGHTS_M = (0.2, 0.5, 1.0, 1.5)
CALIBRATION_X_M = (-1.5, -0.5, 0.5, 1.5)


def validate_multilevel_orthographic_depth(config) -> dict:
    """Run a real Habitat render/Bullet regression at four known heights."""
    scene_id = "empty_stage"
    calibration_config = replace(
        config,
        scenes=[scene_id],
        scene_splits={"train": [scene_id], "val": [], "test": []},
        scene_overrides={scene_id: {"bev_camera_height_m": 2.2}},
        bev_meters_per_pixel=max(float(config.bev_meters_per_pixel), 0.01),
    )
    backend = HabitatBackend(calibration_config, scene_id)
    try:
        asset_dir = REPO_ROOT / "assets/calibration"
        manager = backend.sim.get_object_template_manager()
        loaded = manager.load_configs(str(asset_dir))
        if not loaded:
            raise RuntimeError("Could not load multi-height calibration object")
        suffix = "multilevel_slabs.object_config.json"
        handle = handles_by_suffix(manager, [suffix])[suffix]
        rigid = backend.sim.get_rigid_object_manager().add_object_by_template_handle(
            handle
        )
        if rigid is None:
            raise RuntimeError("Could not instantiate multi-height calibration object")
        rigid.motion_type = backend.habitat_sim.physics.MotionType.KINEMATIC
        # Habitat centers an imported mesh around its aggregate AABB. Restore
        # the authored y=0 slab bottoms to the physical stage floor.
        rigid.translation = np.array(
            [0.0, -float(rigid.collision_shape_aabb.min[1]), 0.0], dtype=np.float32
        )

        camera_y = 2.2
        center_x = 0.5 * (backend.mapping.x_min + backend.mapping.x_max)
        center_z = 0.5 * (backend.mapping.z_min + backend.mapping.z_max)
        state = backend.habitat_sim.AgentState()
        state.position = np.array([center_x, camera_y, center_z], dtype=np.float32)
        from habitat_sim.utils.common import quat_from_angle_axis
        state.rotation = quat_from_angle_axis(
            -math.pi / 2.0, np.array([1.0, 0.0, 0.0])
        )
        backend.sim.get_agent(config.num_robots).set_state(
            state, infer_sensor_states=True
        )
        observations = backend.sim.get_sensor_observations(
            agent_ids=[config.num_robots]
        )[config.num_robots]
        raw_depth = np.asarray(observations["bev_depth"], dtype=np.float32)
        metric_depth = habitat_orthographic_depth_to_metric(
            raw_depth, config.bev_near, config.bev_far
        )
        height_map = camera_y - metric_depth

        records = []
        for x, expected_height in zip(CALIBRATION_X_M, CALIBRATION_HEIGHTS_M):
            u, v = backend.mapping.world_to_bev(x, 0.0)
            row = int(round(v))
            col = int(round(u))
            rendered_height = float(height_map[row, col])
            ray = backend.habitat_sim.geo.Ray(
                np.array([x, camera_y, 0.0], dtype=np.float32),
                np.array([0.0, -1.0, 0.0], dtype=np.float32),
            )
            hit = backend.sim.cast_ray(
                ray, max_distance=float(config.bev_far), buffer_distance=0.0
            )
            if not hit.has_hits():
                raise RuntimeError(f"Calibration slab at x={x} produced no Bullet hit")
            bullet_height = camera_y - float(hit.hits[0].ray_distance)
            records.append(
                {
                    "expected_height_m": expected_height,
                    "pixel_rc": [row, col],
                    "rendered_height_m": rendered_height,
                    "bullet_height_m": bullet_height,
                    "render_error_m": abs(rendered_height - expected_height),
                    "bullet_error_m": abs(bullet_height - expected_height),
                    "render_bullet_error_m": abs(rendered_height - bullet_height),
                }
            )
        maximum_error = max(
            max(record["render_error_m"], record["bullet_error_m"])
            for record in records
        )
        tolerance = float(config.height_validation_max_error_m)
        return {
            "available": True,
            "passed": maximum_error <= tolerance,
            "scene_id": scene_id,
            "asset": "assets/calibration/multilevel_slabs.object_config.json",
            "heights_m": list(CALIBRATION_HEIGHTS_M),
            "max_abs_error_m": maximum_error,
            "tolerance_m": tolerance,
            "records": records,
        }
    finally:
        backend.close()
