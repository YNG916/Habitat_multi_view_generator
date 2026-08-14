from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .bev import BevMapping, habitat_orthographic_depth_to_metric, occupancy_from_pathfinder
from .coordinates import forward_from_quaternion, transform_matrix, yaw_to_quaternion_xyzw
from .objects import aabb_dict, handles_by_suffix
from .world_state import ObjectState, WorldState


class HabitatBackend:
    """The only layer allowed to mutate Habitat state.

    Every call to ``apply_world_state`` reconstructs all controlled entities from
    an immutable-ish WorldState snapshot. Scene-owned ReplicaCAD furniture is
    never removed or serialized as a controlled object.
    """

    def __init__(self, config, scene_id: str):
        import habitat_sim

        self.habitat_sim = habitat_sim
        self.config = config
        self.scene_id = scene_id
        self.navmesh_path = config.navmesh_path(scene_id)
        if not config.dataset_config_path.exists():
            raise FileNotFoundError(f"Scene dataset config not found: {config.dataset_config_path}")
        if not self.navmesh_path.exists():
            raise FileNotFoundError(f"Explicit navmesh not found for {scene_id}: {self.navmesh_path}")

        preflight = habitat_sim.PathFinder()
        if not preflight.load_nav_mesh(str(self.navmesh_path)):
            raise RuntimeError(f"Could not load navmesh: {self.navmesh_path}")
        bounds = preflight.get_bounds()
        self.scene_bounds = (
            np.asarray(bounds[0], dtype=np.float64),
            np.asarray(bounds[1], dtype=np.float64),
        )
        self.mapping = BevMapping.from_bounds(
            self.scene_bounds[0], self.scene_bounds[1], config.bev_meters_per_pixel
        )
        self.sim = self._create_simulator()
        if not self.sim.pathfinder.load_nav_mesh(str(self.navmesh_path)):
            self.close()
            raise RuntimeError(f"Habitat loaded {scene_id} but explicit navmesh load failed")
        self.spawned_object_ids: List[int] = []
        self.render_ids: Dict[str, Tuple[int, int]] = {}
        self._load_proxy_templates()
        self.controlled_handles = self._resolve_controlled_handles()

    def _sensor(self, uuid, sensor_type, subtype, resolution, near, far):
        spec = self.habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = subtype
        spec.resolution = list(resolution)
        spec.position = [0.0, 0.0, 0.0]
        spec.orientation = [0.0, 0.0, 0.0]
        spec.near = float(near)
        spec.far = float(far)
        return spec

    def _create_simulator(self):
        hs = self.habitat_sim
        agent_configs = []
        for index in range(1, self.config.num_robots + 1):
            sensors = []
            rgb = self._sensor(
                f"robot_{index:02d}_rgb", hs.SensorType.COLOR, hs.SensorSubType.PINHOLE,
                [self.config.height, self.config.width], self.config.near, self.config.far,
            )
            rgb.hfov = self.config.hfov_deg
            depth = self._sensor(
                f"robot_{index:02d}_depth", hs.SensorType.DEPTH, hs.SensorSubType.PINHOLE,
                [self.config.height, self.config.width], self.config.near, self.config.far,
            )
            depth.hfov = self.config.hfov_deg
            sensors.extend([rgb, depth])
            if self.config.enable_instance:
                instance = self._sensor(
                    f"robot_{index:02d}_instance", hs.SensorType.SEMANTIC, hs.SensorSubType.PINHOLE,
                    [self.config.height, self.config.width], self.config.near, self.config.far,
                )
                instance.hfov = self.config.hfov_deg
                sensors.append(instance)
            agent = hs.agent.AgentConfiguration()
            agent.sensor_specifications = sensors
            agent_configs.append(agent)

        bev_sensors = []
        for uuid, sensor_type in [("bev_rgb", hs.SensorType.COLOR), ("bev_depth", hs.SensorType.DEPTH)]:
            spec = self._sensor(
                uuid, sensor_type, hs.SensorSubType.ORTHOGRAPHIC,
                [self.mapping.height, self.mapping.width], self.config.bev_near, self.config.bev_far,
            )
            # v0.3.3 defines horizontal world extent as 1 / ortho_scale.
            spec.ortho_scale = 1.0 / (self.mapping.x_max - self.mapping.x_min)
            bev_sensors.append(spec)
        if self.config.enable_instance:
            spec = self._sensor(
                "bev_instance", hs.SensorType.SEMANTIC, hs.SensorSubType.ORTHOGRAPHIC,
                [self.mapping.height, self.mapping.width], self.config.bev_near, self.config.bev_far,
            )
            spec.ortho_scale = 1.0 / (self.mapping.x_max - self.mapping.x_min)
            bev_sensors.append(spec)
        bev_agent = hs.agent.AgentConfiguration()
        bev_agent.sensor_specifications = bev_sensors
        agent_configs.append(bev_agent)

        simulator = hs.SimulatorConfiguration()
        simulator.scene_dataset_config_file = str(self.config.dataset_config_path)
        simulator.scene_id = self.scene_id
        simulator.enable_physics = True
        simulator.gpu_device_id = int(self.config.gpu_device_id)
        return hs.Simulator(hs.Configuration(simulator, agent_configs))

    def _load_proxy_templates(self) -> None:
        manager = self.sim.get_object_template_manager()
        directories = {str(self.config.resolve(path).parent) for path in self.config.robot_proxy_configs}
        for directory in directories:
            loaded = manager.load_configs(directory)
            if not loaded and self.config.enable_robot_proxies:
                raise RuntimeError(f"No robot proxy configs loaded from {directory}")

    def _resolve_controlled_handles(self) -> Dict[str, str]:
        manager = self.sim.get_object_template_manager()
        suffixes = list(self.config.controlled_object_whitelist.values())
        resolved_suffix = handles_by_suffix(manager, suffixes)
        return {
            category: resolved_suffix[suffix]
            for category, suffix in self.config.controlled_object_whitelist.items()
        }

    def resolve_proxy_handle(self, config_path: str) -> str:
        suffix = Path(config_path).name
        return handles_by_suffix(self.sim.get_object_template_manager(), [suffix])[suffix]

    def resolve_runtime_handle(self, handle: str) -> str:
        manager = self.sim.get_object_template_manager()
        if manager.get_library_has_handle(handle):
            return handle
        suffix = Path(handle).name
        return handles_by_suffix(manager, [suffix])[suffix]

    @staticmethod
    def _agent_quaternion(quaternion_xyzw):
        from habitat_sim.utils.common import quat_from_coeffs
        return quat_from_coeffs(np.asarray(quaternion_xyzw, dtype=np.float64))

    @staticmethod
    def _rigid_quaternion(quaternion_xyzw):
        import magnum as mn
        x, y, z, w = map(float, quaternion_xyzw)
        return mn.Quaternion(mn.Vector3(x, y, z), w)

    def _clear_spawned(self) -> None:
        manager = self.sim.get_rigid_object_manager()
        for object_id in self.spawned_object_ids:
            if manager.get_object_by_id(object_id) is not None:
                manager.remove_object_by_id(object_id)
        self.spawned_object_ids.clear()
        self.render_ids.clear()

    def _spawn(self, handle: str, position, quaternion, semantic_id: int, entity_id: str):
        manager = self.sim.get_rigid_object_manager()
        handle = self.resolve_runtime_handle(handle)
        obj = manager.add_object_by_template_handle(handle)
        if obj is None:
            raise RuntimeError(f"Habitat failed to instantiate rigid template {handle}")
        obj.motion_type = self.habitat_sim.physics.MotionType.KINEMATIC
        obj.translation = np.asarray(position, dtype=np.float32)
        obj.rotation = self._rigid_quaternion(quaternion)
        obj.semantic_id = int(semantic_id)
        self.spawned_object_ids.append(int(obj.object_id))
        self.render_ids[entity_id] = (int(obj.object_id), int(semantic_id))
        return obj

    def apply_world_state(self, state: WorldState) -> None:
        self._clear_spawned()
        for index, robot in enumerate(state.robots):
            robot.synchronize_camera()
            agent_state = self.habitat_sim.AgentState()
            agent_state.position = np.asarray(robot.camera.position_world, dtype=np.float32)
            agent_state.rotation = self._agent_quaternion(robot.camera.quaternion_world_xyzw)
            self.sim.get_agent(index).set_state(agent_state, infer_sensor_states=True)
            if self.config.enable_robot_proxies:
                handle = self.resolve_proxy_handle(robot.proxy_asset_handle)
                self._spawn(
                    handle, robot.base_position_world, yaw_to_quaternion_xyzw(robot.yaw_rad),
                    robot.proxy_semantic_id, robot.robot_id,
                )
        for obj_state in state.objects:
            if obj_state.active:
                self._spawn(
                    obj_state.asset_handle, obj_state.position_world, obj_state.quaternion_world_xyzw,
                    obj_state.semantic_id, obj_state.instance_id,
                )

        center_x = 0.5 * (self.mapping.x_min + self.mapping.x_max)
        center_z = 0.5 * (self.mapping.z_min + self.mapping.z_max)
        bev_height = float(self.config.scene_value(self.scene_id, "bev_camera_height_m"))
        camera_y = float(state.floor_y + bev_height)
        bev_state = self.habitat_sim.AgentState()
        bev_state.position = np.array([center_x, camera_y, center_z], dtype=np.float32)
        from habitat_sim.utils.common import quat_from_angle_axis
        bev_state.rotation = quat_from_angle_axis(-math.pi / 2.0, np.array([1.0, 0.0, 0.0]))
        self.sim.get_agent(self.config.num_robots).set_state(bev_state, infer_sensor_states=True)

    def create_object_state(self, category: str, x: float, z: float, floor_y: float, instance_index: int) -> ObjectState:
        if category not in self.controlled_handles:
            raise KeyError(f"Controlled category not in curated whitelist: {category}")
        handle = self.controlled_handles[category]
        manager = self.sim.get_rigid_object_manager()
        obj = manager.add_object_by_template_handle(handle)
        if obj is None:
            raise RuntimeError(f"Could not instantiate controlled object template {handle}")
        obj.motion_type = self.habitat_sim.physics.MotionType.KINEMATIC
        local_min_y = float(obj.collision_shape_aabb.min[1])
        position = np.array([x, floor_y - local_min_y, z], dtype=np.float64)
        obj.translation = position.astype(np.float32)
        bbox = aabb_dict(obj.collision_shape_aabb, transform_matrix(position, [0.0, 0.0, 0.0, 1.0]))
        manager.remove_object_by_id(obj.object_id)
        return ObjectState(
            instance_id=f"object_{instance_index:03d}", category=category,
            asset_handle=self.config.controlled_object_whitelist[category],
            position_world=position.tolist(), quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
            active=True, movable=True, semantic_id=2000 + instance_index, bbox=bbox,
        )

    def refresh_object_bboxes(self, state: WorldState) -> None:
        # Must be called after apply_world_state; query authoritative Habitat AABBs.
        manager = self.sim.get_rigid_object_manager()
        for obj_state in state.objects:
            if not obj_state.active:
                continue
            renderer_id = self.render_ids[obj_state.instance_id][0]
            rigid = manager.get_object_by_id(renderer_id)
            obj_state.bbox = aabb_dict(
                rigid.collision_shape_aabb,
                transform_matrix(obj_state.position_world, obj_state.quaternion_world_xyzw),
            )

    def render(self, state: WorldState) -> dict:
        self.apply_world_state(state)
        self.refresh_object_bboxes(state)
        agent_ids = list(range(self.config.num_robots + 1))
        observations = self.sim.get_sensor_observations(agent_ids=agent_ids)
        robots = {}
        for index, robot in enumerate(state.robots, start=1):
            observation = observations[index - 1]
            rgb = np.asarray(observation[f"robot_{index:02d}_rgb"])[..., :3].astype(np.uint8)
            depth = np.asarray(observation[f"robot_{index:02d}_depth"], dtype=np.float32)
            instance = None
            if self.config.enable_instance:
                instance = np.asarray(observation[f"robot_{index:02d}_instance"], dtype=np.int32)
            robots[robot.robot_id] = {"rgb": rgb, "depth": depth, "instance": instance}
        bev_obs = observations[self.config.num_robots]
        bev_rgb = np.asarray(bev_obs["bev_rgb"])[..., :3].astype(np.uint8)
        bev_depth = np.asarray(bev_obs["bev_depth"], dtype=np.float32)
        bev_instance = (
            np.asarray(bev_obs["bev_instance"], dtype=np.int32)
            if self.config.enable_instance else None
        )
        camera_height = float(self.config.scene_value(self.scene_id, "bev_camera_height_m"))
        metric_bev_depth = habitat_orthographic_depth_to_metric(
            bev_depth, self.config.bev_near, self.config.bev_far
        )
        height = (camera_height - metric_bev_depth).astype(np.float32)
        occupancy = occupancy_from_pathfinder(self.sim.pathfinder, self.mapping, state.floor_y)
        self._populate_visibility(state, robots)
        return {
            "robots": robots,
            "bev_rgb": bev_rgb,
            "bev_depth": bev_depth,
            "bev_metric_depth": metric_bev_depth,
            "bev_instance": bev_instance,
            "height": height,
            "occupancy": occupancy,
        }

    def _populate_visibility(self, state: WorldState, robot_outputs: dict) -> None:
        entities = {r.robot_id: r for r in state.robots}
        entities.update({obj.instance_id: obj for obj in state.objects})
        for entity_id, entity in entities.items():
            entity.visibility = {}
            render_ids = self.render_ids.get(entity_id, ())
            for observer_id, output in robot_outputs.items():
                instance = output["instance"]
                count = 0
                if instance is not None and render_ids:
                    count = int(np.isin(instance, render_ids).sum())
                entity.visibility[observer_id] = {
                    "visible": bool(count > 0),
                    "visible_pixel_count": count,
                    "method": "semantic_sensor_rigid_id" if instance is not None else "unavailable",
                }

    def validate_pinhole_depth_centers(self, state: WorldState, robot_outputs: dict) -> dict:
        records = []
        for robot in state.robots:
            depth = robot_outputs[robot.robot_id]["depth"]
            row, col = depth.shape[0] // 2, depth.shape[1] // 2
            saved_depth = float(depth[row, col])
            direction = forward_from_quaternion(robot.camera.quaternion_world_xyzw)
            ray = self.habitat_sim.geo.Ray(
                np.asarray(robot.camera.position_world, dtype=np.float32),
                np.asarray(direction, dtype=np.float32),
            )
            result = self.sim.cast_ray(ray, max_distance=float(self.config.far), buffer_distance=0.0)
            ray_distance = float(result.hits[0].ray_distance) if result.has_hits() else None
            error = abs(saved_depth - ray_distance) if ray_distance is not None and saved_depth > 0 else None
            records.append({
                "robot_id": robot.robot_id, "center_pixel_rc": [row, col],
                "saved_depth_m": saved_depth, "physics_ray_distance_m": ray_distance,
                "absolute_error_m": error,
            })
        finite_errors = [record["absolute_error_m"] for record in records if record["absolute_error_m"] is not None]
        return {
            "depth_convention": "pinhole camera Z-depth; center pixel equals forward-ray distance",
            "records": records,
            "max_absolute_error_m": max(finite_errors) if finite_errors else None,
        }

    def validate_height_rays(self, state: WorldState, height_map: np.ndarray, samples: int = 12, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        valid = np.argwhere(np.isfinite(height_map))
        if len(valid) == 0:
            return {"available": False, "reason": "no finite orthographic depth"}
        selected = valid[rng.choice(len(valid), size=min(samples, len(valid)), replace=False)]
        camera_y = state.floor_y + float(self.config.scene_value(self.scene_id, "bev_camera_height_m"))
        errors = []
        for row, col in selected:
            x, z = self.mapping.bev_to_world(float(col), float(row))
            ray = self.habitat_sim.geo.Ray(
                np.array([x, camera_y, z], dtype=np.float32), np.array([0.0, -1.0, 0.0], dtype=np.float32)
            )
            result = self.sim.cast_ray(ray, max_distance=float(self.config.bev_far), buffer_distance=0.0)
            if not result.has_hits():
                continue
            ray_height = camera_y - float(result.hits[0].ray_distance) - state.floor_y
            errors.append(abs(ray_height - float(height_map[row, col])))
        return {
            "available": bool(errors), "sample_count": len(errors),
            "mean_abs_error_m": float(np.mean(errors)) if errors else None,
            "max_abs_error_m": float(np.max(errors)) if errors else None,
            "note": "render mesh depth compared with physics collision-mesh ray hits",
        }

    def close(self) -> None:
        if getattr(self, "sim", None) is not None:
            self.sim.close()
            self.sim = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
