from __future__ import annotations

import math
import numpy as np

from .bev import BevMapping, habitat_orthographic_depth_to_metric
from .config import REPO_ROOT
from .objects import handles_by_suffix

CALIBRATION_HEIGHTS_M = (.2, .5, 1., 1.5)
CALIBRATION_X_M = (-1.5, -.5, .5, 1.5)


def validate_multilevel_orthographic_depth(config) -> dict:
    """Dataset-independent Habitat-Sim v0.3.3 orthographic-depth regression."""
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis

    mapping = BevMapping.from_bounds(
        np.array([-2., 0., -1.]), np.array([2., 2., 1.]), .01
    )
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = "calibration_depth"
    spec.sensor_type = habitat_sim.SensorType.DEPTH
    spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    spec.resolution = [mapping.height, mapping.width]
    spec.position = [0., 0., 0.]
    spec.orientation = [0., 0., 0.]
    spec.near = float(config.bev_near)
    spec.far = float(config.bev_far)
    spec.ortho_scale = 1. / (mapping.x_max - mapping.x_min)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [spec]
    simulator = habitat_sim.SimulatorConfiguration()
    simulator.scene_id = "NONE"
    simulator.enable_physics = True
    simulator.gpu_device_id = int(config.gpu_device_id)
    sim = habitat_sim.Simulator(habitat_sim.Configuration(simulator, [agent]))
    try:
        asset_dir = REPO_ROOT / "assets/calibration"
        manager = sim.get_object_template_manager()
        if not manager.load_configs(str(asset_dir)):
            raise RuntimeError("Could not load calibration object")
        handle = handles_by_suffix(
            manager, ["multilevel_slabs.object_config.json"]
        )["multilevel_slabs.object_config.json"]
        rigid = sim.get_rigid_object_manager().add_object_by_template_handle(handle)
        if rigid is None:
            raise RuntimeError("Could not instantiate calibration object")
        rigid.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        rigid.translation = np.array(
            [0., -float(rigid.collision_shape_aabb.min[1]), 0.], dtype=np.float32
        )

        camera_y = 2.2
        state = habitat_sim.AgentState()
        state.position = np.array([0., camera_y, 0.], dtype=np.float32)
        state.rotation = quat_from_angle_axis(
            -math.pi / 2., np.array([1., 0., 0.])
        )
        sim.get_agent(0).set_state(state, infer_sensor_states=True)
        raw = np.asarray(
            sim.get_sensor_observations(agent_ids=[0])[0]["calibration_depth"],
            dtype=np.float32,
        )
        metric = habitat_orthographic_depth_to_metric(
            raw, config.bev_near, config.bev_far
        )
        height_map = camera_y - metric
        records = []
        for x, expected in zip(CALIBRATION_X_M, CALIBRATION_HEIGHTS_M):
            u, v = mapping.world_to_bev(x, 0.)
            row, col = int(round(v)), int(round(u))
            rendered = float(height_map[row, col])
            ray = habitat_sim.geo.Ray(
                np.array([x, camera_y, 0.], dtype=np.float32),
                np.array([0., -1., 0.], dtype=np.float32),
            )
            hit = sim.cast_ray(
                ray, max_distance=float(config.bev_far), buffer_distance=0.
            )
            if not hit.has_hits():
                raise RuntimeError(f"No Bullet hit at calibration x={x}")
            bullet = camera_y - float(hit.hits[0].ray_distance)
            records.append({
                "expected_height_m": expected,
                "pixel_rc": [row, col],
                "rendered_height_m": rendered,
                "bullet_height_m": bullet,
                "render_error_m": abs(rendered - expected),
                "bullet_error_m": abs(bullet - expected),
                "render_bullet_error_m": abs(rendered - bullet),
            })
        maximum = max(
            max(item["render_error_m"], item["bullet_error_m"])
            for item in records
        )
        tolerance = float(config.height_validation_max_error_m)
        return {
            "available": True,
            "passed": maximum <= tolerance,
            "scene_id": "NONE",
            "dataset_independent": True,
            "asset": "assets/calibration/multilevel_slabs.object_config.json",
            "heights_m": list(CALIBRATION_HEIGHTS_M),
            "max_abs_error_m": maximum,
            "tolerance_m": tolerance,
            "records": records,
        }
    finally:
        sim.close()
