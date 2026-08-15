from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .bev import BevMapping, habitat_orthographic_depth_to_metric, occupancy_from_pathfinder
from .coordinates import (
    camera_transforms,
    forward_from_quaternion,
    transform_matrix,
    yaw_to_quaternion_xyzw,
)
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
        self.navmesh_bounds = (
            np.asarray(bounds[0], dtype=np.float64),
            np.asarray(bounds[1], dtype=np.float64),
        )
        # Probe the complete rendered scene before fixing the orthographic
        # sensor extent. Navigability and visual coverage are different bounds.
        self.render_bev_bounds = self.navmesh_bounds
        self.scene_bounds = self.render_bev_bounds
        self.mapping = BevMapping.from_bounds(
            self.render_bev_bounds[0], self.render_bev_bounds[1],
            config.bev_meters_per_pixel,
        )
        self._runtime_dataset_config_path = self._filtered_dataset_config()
        self.sim = None
        probe = None
        try:
            probe = self._create_simulator(bounds_probe=True)
            scene_aabb = probe.scene_aabb
            self.render_bev_bounds = (
                np.asarray(scene_aabb.min, dtype=np.float64),
                np.asarray(scene_aabb.max, dtype=np.float64),
            )
            self.scene_bounds = self.render_bev_bounds
            self.mapping = BevMapping.from_bounds(
                self.render_bev_bounds[0], self.render_bev_bounds[1],
                config.bev_meters_per_pixel,
            )
            probe.close()
            probe = None
            self.sim = self._create_simulator()
        except Exception:
            if probe is not None:
                probe.close()
            self._remove_runtime_dataset_config()
            raise
        if not self.sim.pathfinder.load_nav_mesh(str(self.navmesh_path)):
            self.close()
            raise RuntimeError(f"Habitat loaded {scene_id} but explicit navmesh load failed")
        self.spawned_object_ids: List[int] = []
        self.render_ids: Dict[str, Tuple[int, int]] = {}
        self._occupancy_cache: Dict[float, np.ndarray] = {}
        self._load_proxy_templates()
        self._validate_proxy_dimensions()
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

    def _filtered_dataset_config(self) -> Path:
        """Drop references to optional local resources that are not installed."""
        source = self.config.dataset_config_path
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        base = source.parent
        navmeshes = data.get("navmesh_instances", {})
        data["navmesh_instances"] = {
            key: value
            for key, value in navmeshes.items()
            if (base / value).exists()
        }
        urdf_paths = data.get("articulated_objects", {}).get("paths", {}).get(".urdf", [])
        data["articulated_objects"]["paths"][".urdf"] = [
            value for value in urdf_paths if "hab_fetch_1.0" not in value
        ]
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".scene_dataset_config.json",
            prefix=".mri-",
            dir=base,
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(data, handle)
            return Path(handle.name)

    def _remove_runtime_dataset_config(self) -> None:
        path = getattr(self, "_runtime_dataset_config_path", None)
        if path is not None:
            path.unlink(missing_ok=True)
            self._runtime_dataset_config_path = None

    def _create_simulator(self, bounds_probe: bool = False):
        hs = self.habitat_sim
        if bounds_probe:
            agent = hs.agent.AgentConfiguration()
            agent.sensor_specifications = []
            simulator = hs.SimulatorConfiguration()
            simulator.scene_dataset_config_file = str(self._runtime_dataset_config_path)
            simulator.scene_id = self.scene_id
            simulator.enable_physics = True
            simulator.gpu_device_id = int(self.config.gpu_device_id)
            return hs.Simulator(hs.Configuration(simulator, [agent]))

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
                instance.semantic_target = type(instance.semantic_target).OBJECT_ID
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
            spec.semantic_target = type(spec.semantic_target).OBJECT_ID
            bev_sensors.append(spec)
        bev_agent = hs.agent.AgentConfiguration()
        bev_agent.sensor_specifications = bev_sensors
        agent_configs.append(bev_agent)

        simulator = hs.SimulatorConfiguration()
        simulator.scene_dataset_config_file = str(self._runtime_dataset_config_path)
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

    def _validate_proxy_dimensions(self) -> None:
        """Reject a stale proxy export before collecting any samples."""
        if not self.config.enable_robot_proxies:
            return
        manager = self.sim.get_rigid_object_manager()
        expected_diameter = float(self.config.robot_body_diameter_m)
        expected_height = float(self.config.robot_body_height_m)
        for config_path in self.config.robot_proxy_configs:
            handle = self.resolve_proxy_handle(config_path)
            rigid = manager.add_object_by_template_handle(handle)
            if rigid is None:
                raise RuntimeError(f"Could not instantiate robot proxy {handle}")
            try:
                bounds = rigid.root_scene_node.cumulative_bb
                minimum = np.asarray(bounds.min, dtype=np.float64)
                extent = np.asarray(bounds.max, dtype=np.float64) - minimum
                diameter = max(float(extent[0]), float(extent[2]))
                height = float(extent[1])
                if abs(float(minimum[1])) > 1e-4:
                    raise ValueError(
                        f"Robot proxy {handle} does not touch local Y=0: "
                        f"min_y={minimum[1]:.6f} m"
                    )
                if abs(diameter - expected_diameter) > 0.002:
                    raise ValueError(
                        f"Robot proxy {handle} diameter is {diameter:.6f} m, "
                        f"expected {expected_diameter:.6f} m; rerun "
                        "scripts/prepare_robot_proxy_assets.py"
                    )
                if abs(height - expected_height) > 0.002:
                    raise ValueError(
                        f"Robot proxy {handle} height is {height:.6f} m, "
                        f"expected {expected_height:.6f} m"
                    )
            finally:
                manager.remove_object_by_id(rigid.object_id)

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

    def _spawn(
        self,
        handle: str,
        position,
        quaternion,
        semantic_id: int,
        entity_id: str,
        support_floor_y: Optional[float] = None,
        require_floor_origin: bool = False,
    ):
        manager = self.sim.get_rigid_object_manager()
        handle = self.resolve_runtime_handle(handle)
        obj = manager.add_object_by_template_handle(handle)
        if obj is None:
            raise RuntimeError(f"Habitat failed to instantiate rigid template {handle}")
        obj.motion_type = self.habitat_sim.physics.MotionType.KINEMATIC
        translation = np.asarray(position, dtype=np.float64).copy()
        if support_floor_y is not None:
            collision_min_y = float(obj.collision_shape_aabb.min[1])
            visual_min_y = float(obj.root_scene_node.cumulative_bb.min[1])
            if require_floor_origin and abs(visual_min_y) > 1e-4:
                manager.remove_object_by_id(obj.object_id)
                raise ValueError(
                    f"{entity_id} visual proxy local ground is {visual_min_y:.6f} m; "
                    "robot assets/COM must touch Y=0 so mesh and camera agree"
                )
            translation[1] = (
                float(support_floor_y)
                if require_floor_origin
                else float(support_floor_y) - collision_min_y
            )
        obj.rotation = self._rigid_quaternion(quaternion)
        obj.semantic_id = int(semantic_id)
        obj.translation = translation.astype(np.float32)
        self.spawned_object_ids.append(int(obj.object_id))
        self.render_ids[entity_id] = (int(obj.object_id), int(semantic_id))
        return obj

    def bev_camera_height_above_floor(self, floor_y: float) -> float:
        scene_override = self.config.scene_overrides.get(self.scene_id, {})
        if "bev_camera_height_m" not in scene_override:
            raise ValueError(
                f"{self.scene_id} requires an explicit scene_overrides.bev_camera_height_m"
            )
        requested = float(scene_override["bev_camera_height_m"])
        clearance = float(self.config.scene_value(self.scene_id, "bev_ceiling_clearance_m"))
        maximum = float(self.render_bev_bounds[1][1] - floor_y - clearance)
        if not float(self.config.bev_near) < requested < maximum:
            raise ValueError(
                f"BEV camera height {requested:.3f} m is unsafe for {self.scene_id}; "
                f"it must be below the scene ceiling estimate with {clearance:.3f} m clearance "
                f"(maximum {maximum:.3f} m). Add a scene_overrides entry."
            )
        return requested

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
                    support_floor_y=robot.base_position_world[1],
                    require_floor_origin=True,
                )
        for obj_state in state.objects:
            if obj_state.active:
                self._spawn(
                    obj_state.asset_handle, obj_state.position_world, obj_state.quaternion_world_xyzw,
                    obj_state.semantic_id, obj_state.instance_id,
                )

        center_x = 0.5 * (self.mapping.x_min + self.mapping.x_max)
        center_z = 0.5 * (self.mapping.z_min + self.mapping.z_max)
        bev_height = self.bev_camera_height_above_floor(state.floor_y)
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

    def floor_surface_y(self, position_world) -> float:
        """Resolve physical floor Y below a same-floor NavMesh sample."""
        point = np.asarray(position_world, dtype=np.float64)
        origin = point.copy()
        origin[1] += 0.35
        ray = self.habitat_sim.geo.Ray(
            origin.astype(np.float32), np.array([0.0, -1.0, 0.0], dtype=np.float32)
        )
        result = self.sim.cast_ray(ray, max_distance=1.5, buffer_distance=0.0)
        for hit in result.hits:
            if int(getattr(hit, "object_id", -1)) in self.spawned_object_ids:
                continue
            return float(origin[1] - hit.ray_distance)
        raise RuntimeError(f"Could not resolve physical floor below {point.tolist()}")

    def support_object_on_floor(self, obj_state: ObjectState, floor_y: float) -> None:
        """Set object Y from its collision AABB instead of preserving stale Y."""
        manager = self.sim.get_rigid_object_manager()
        handle = self.resolve_runtime_handle(obj_state.asset_handle)
        rigid = manager.add_object_by_template_handle(handle)
        if rigid is None:
            raise RuntimeError(f"Could not instantiate controlled object template {handle}")
        try:
            local_aabb = rigid.collision_shape_aabb
            position = np.asarray(obj_state.position_world, dtype=np.float64)
            position[1] = float(floor_y) - float(local_aabb.min[1])
            obj_state.position_world = position.tolist()
            obj_state.bbox = aabb_dict(
                local_aabb,
                transform_matrix(position, obj_state.quaternion_world_xyzw),
            )
        finally:
            manager.remove_object_by_id(rigid.object_id)

    def robot_support_report(self, state: WorldState, target_id: str) -> dict:
        """Measure rendered proxy ground contact and base/camera alignment."""
        robot = state.robot(target_id)
        self.apply_world_state(state)
        rigid_id = self.render_ids[target_id][0]
        rigid = self.sim.get_rigid_object_manager().get_object_by_id(rigid_id)
        if rigid is None:
            raise RuntimeError(f"Spawned robot proxy disappeared: {target_id}")
        collision_min_y = float(rigid.collision_shape_aabb.min[1])
        visual_min_y = float(rigid.root_scene_node.cumulative_bb.min[1])
        proxy_origin_y = float(rigid.translation[1])
        physical_floor_y = self.floor_surface_y(robot.base_position_world)
        visual_bottom_y = proxy_origin_y + visual_min_y
        return {
            "robot_id": target_id,
            "proxy_local_min_y_m": visual_min_y,
            "collision_local_min_y_m": collision_min_y,
            "proxy_origin_world_y_m": proxy_origin_y,
            "physical_floor_world_y_m": physical_floor_y,
            "visual_bottom_world_y_m": visual_bottom_y,
            "support_gap_m": visual_bottom_y - physical_floor_y,
            "proxy_origin_offset_from_base_m": (
                proxy_origin_y - float(robot.base_position_world[1])
            ),
        }

    def entity_collision_report(self, state: WorldState, target_id: str) -> dict:
        """Use Bullet contacts to reject entity penetration.

        OBJECT_ID segmentation and collision validation share the same spawned
        rigid entity. The query entity is temporarily dynamic because Bullet
        does not report kinematic-vs-kinematic overlaps; simulation time is
        never advanced.
        """
        self.apply_world_state(state)
        self.refresh_object_bboxes(state)
        if target_id not in self.render_ids:
            raise KeyError(f"Controlled entity is not active/spawned: {target_id}")
        rigid_id = self.render_ids[target_id][0]
        rigid = self.sim.get_rigid_object_manager().get_object_by_id(rigid_id)
        if rigid is None:
            raise RuntimeError(f"Spawned rigid entity disappeared: {target_id}")

        try:
            robot = state.robot(target_id)
            quaternion = yaw_to_quaternion_xyzw(robot.yaw_rad)
            is_robot = True
        except KeyError:
            obj = state.object(target_id)
            quaternion = obj.quaternion_world_xyzw
            is_robot = False
        position = np.asarray(rigid.translation, dtype=np.float64)
        bbox = aabb_dict(
            rigid.collision_shape_aabb,
            transform_matrix(position, quaternion),
        )
        bbox_bottom = float(bbox["min_world"][1])

        rigid.motion_type = self.habitat_sim.physics.MotionType.DYNAMIC
        rigid.linear_velocity = np.zeros(3, dtype=np.float32)
        rigid.angular_velocity = np.zeros(3, dtype=np.float32)
        penetration_tolerance = float(self.config.collision_penetration_tolerance_m)
        support_tolerance = float(self.config.support_contact_tolerance_m)
        floor_contact_tolerance = (
            float(self.config.robot_floor_collision_tolerance_m)
            if is_robot else support_tolerance
        )
        self.sim.perform_discrete_collision_detection()
        rejected = []
        contacts_checked = 0
        for contact in self.sim.get_physics_contact_points():
            if not contact.is_active:
                continue
            if rigid_id not in (contact.object_id_a, contact.object_id_b):
                continue
            contacts_checked += 1
            distance = float(contact.contact_distance)
            if distance >= -penetration_tolerance:
                continue
            target_is_a = contact.object_id_a == rigid_id
            other_id = int(contact.object_id_b if target_is_a else contact.object_id_a)
            target_position = (
                contact.position_on_a_in_ws if target_is_a else contact.position_on_b_in_ws
            )
            normal = np.asarray(contact.contact_normal_on_b_in_ws, dtype=np.float64)
            is_floor_support = (
                other_id == int(self.habitat_sim.stage_id)
                and abs(float(normal[1])) >= 0.9
                and abs(float(target_position[1]) - bbox_bottom) <= 0.03
                and distance >= -floor_contact_tolerance
            )
            if not is_floor_support:
                rejected.append({
                    "other_object_id": other_id,
                    "penetration_m": -distance,
                    "target_contact_world": [float(x) for x in target_position],
                })
        return {
            "collision_free": not rejected,
            "target_entity_id": target_id,
            "target_object_id": int(rigid_id),
            "contacts_checked": contacts_checked,
            "rejected_contacts": rejected,
        }

    def entity_collision_free(self, state: WorldState, target_id: str) -> bool:
        return bool(self.entity_collision_report(state, target_id)["collision_free"])

    def object_collision_report(self, state: WorldState, target_id: str) -> dict:
        state.object(target_id)
        return self.entity_collision_report(state, target_id)

    def object_collision_free(self, state: WorldState, target_id: str) -> bool:
        return bool(self.object_collision_report(state, target_id)["collision_free"])

    def _semantic_from_instance(self, instance: Optional[np.ndarray], state: WorldState):
        if instance is None or not self.config.enable_semantic:
            return None
        semantic = np.zeros(instance.shape, dtype=np.uint16)
        for robot in state.robots:
            runtime = self.render_ids.get(robot.robot_id)
            if runtime is not None:
                semantic[instance == runtime[0]] = int(
                    self.config.semantic_category_ids["robot"]
                )
        for obj in state.objects:
            if not obj.active:
                continue
            runtime = self.render_ids.get(obj.instance_id)
            if runtime is not None:
                semantic[instance == runtime[0]] = int(
                    self.config.semantic_category_ids[obj.category]
                )
        return semantic

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
            robots[robot.robot_id] = {
                "rgb": rgb,
                "depth": depth,
                "instance": instance,
                "semantic": self._semantic_from_instance(instance, state),
            }
        bev_obs = observations[self.config.num_robots]
        bev_rgb = np.asarray(bev_obs["bev_rgb"])[..., :3].astype(np.uint8)
        bev_depth = np.asarray(bev_obs["bev_depth"], dtype=np.float32)
        bev_instance = (
            np.asarray(bev_obs["bev_instance"], dtype=np.int32)
            if self.config.enable_instance else None
        )
        camera_height = self.bev_camera_height_above_floor(state.floor_y)
        metric_bev_depth = habitat_orthographic_depth_to_metric(
            bev_depth, self.config.bev_near, self.config.bev_far
        )
        height = (camera_height - metric_bev_depth).astype(np.float32)
        floor_key = round(float(state.floor_y), 3)
        if floor_key not in self._occupancy_cache:
            self._occupancy_cache[floor_key] = occupancy_from_pathfinder(
                self.sim.pathfinder,
                self.mapping,
                state.floor_y,
                navmesh_bounds=self.navmesh_bounds,
            )
        occupancy = self._occupancy_cache[floor_key]
        self._populate_visibility(state, robots)
        return {
            "robots": robots,
            "bev_rgb": bev_rgb,
            "bev_depth": bev_depth,
            "bev_metric_depth": metric_bev_depth,
            "bev_instance": bev_instance,
            "bev_semantic": self._semantic_from_instance(bev_instance, state),
            "entity_object_ids": {key: value[0] for key, value in self.render_ids.items()},
            "height": height,
            "occupancy": occupancy,
        }

    def _populate_visibility(self, state: WorldState, robot_outputs: dict) -> None:
        entities = {r.robot_id: r for r in state.robots}
        entities.update({obj.instance_id: obj for obj in state.objects})
        for entity_id, entity in entities.items():
            entity.visibility = {}
            render_object_id = self.render_ids.get(entity_id, (None, None))[0]
            for observer_id, output in robot_outputs.items():
                instance = output["instance"]
                count = 0
                if instance is not None and render_object_id is not None:
                    count = int((instance == render_object_id).sum())
                geometrically_visible = bool(count > 0)
                benchmark_visible = bool(
                    count >= int(self.config.benchmark_visibility_min_pixels)
                )
                entity.visibility[observer_id] = {
                    "visible": geometrically_visible,
                    "geometrically_visible": geometrically_visible,
                    "benchmark_visible": benchmark_visible,
                    "benchmark_min_pixels": int(
                        self.config.benchmark_visibility_min_pixels
                    ),
                    "visible_pixel_count": count,
                    "method": "object_id_sensor" if instance is not None else "unavailable",
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

    def validate_pinhole_depth_grid(self, state: WorldState, robot_outputs: dict) -> dict:
        """Validate off-center Z-depth and pixel-center convention with Bullet rays."""
        records = []
        fractions = (0.25, 0.5, 0.75)
        for robot in state.robots:
            depth = robot_outputs[robot.robot_id]["depth"]
            instance = robot_outputs[robot.robot_id].get("instance")
            intrinsics = robot.camera.intrinsics
            world_from_cv = camera_transforms(
                robot.camera.position_world,
                robot.camera.quaternion_world_xyzw,
            )["T_world_from_camera_cv"]
            for row_fraction in fractions:
                for col_fraction in fractions:
                    row = min(depth.shape[0] - 1, int(row_fraction * depth.shape[0]))
                    col = min(depth.shape[1] - 1, int(col_fraction * depth.shape[1]))
                    direction_cv = np.array(
                        [
                            (col - float(intrinsics["cx"])) / float(intrinsics["fx"]),
                            (row - float(intrinsics["cy"])) / float(intrinsics["fy"]),
                            1.0,
                        ],
                        dtype=np.float64,
                    )
                    ray_scale = float(np.linalg.norm(direction_cv))
                    direction_world = world_from_cv[:3, :3] @ (
                        direction_cv / ray_scale
                    )
                    ray = self.habitat_sim.geo.Ray(
                        np.asarray(robot.camera.position_world, dtype=np.float32),
                        direction_world.astype(np.float32),
                    )
                    result = self.sim.cast_ray(
                        ray,
                        max_distance=float(self.config.far),
                        buffer_distance=0.0,
                    )
                    if not result.has_hits():
                        continue
                    hit = result.hits[0]
                    # Static stage geometry is the only stable render/collision
                    # oracle across ReplicaCAD's optional furniture proxies.
                    if int(hit.object_id) != int(self.habitat_sim.stage_id):
                        continue
                    if (
                        instance is not None
                        and int(instance[row, col]) != int(self.habitat_sim.stage_id)
                    ):
                        continue
                    expected_z = float(hit.ray_distance) / ray_scale
                    saved_z = float(depth[row, col])
                    if not math.isfinite(saved_z) or saved_z <= 0.0:
                        continue
                    records.append(
                        {
                            "robot_id": robot.robot_id,
                            "pixel_rc": [int(row), int(col)],
                            "saved_z_depth_m": saved_z,
                            "bullet_stage_z_depth_m": expected_z,
                            "absolute_error_m": abs(saved_z - expected_z),
                        }
                    )
        errors = [record["absolute_error_m"] for record in records]
        return {
            "available": bool(records),
            "sample_count": len(records),
            "pixel_coordinate_convention": (
                "integer pixel centers with cx=width/2, cy=height/2"
            ),
            "validation_scope": "3x3 off-center grid; first Bullet hit is static stage",
            "mean_abs_error_m": float(np.mean(errors)) if errors else None,
            "median_abs_error_m": float(np.median(errors)) if errors else None,
            "p90_abs_error_m": (
                float(np.percentile(errors, 90)) if errors else None
            ),
            "max_abs_error_m": float(np.max(errors)) if errors else None,
            "records": records,
        }

    def validate_height_rays(self, state: WorldState, height_map: np.ndarray, samples: int = 12, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        finite = np.isfinite(height_map)
        floor_band_m = max(0.03, float(self.config.height_validation_max_error_m))
        valid = np.argwhere(finite & (np.abs(height_map) <= floor_band_m))
        if len(valid) == 0:
            return {
                "available": False,
                "reason": "no rendered pixels match the resolved physical floor",
            }
        selected = valid[rng.choice(len(valid), size=min(samples, len(valid)), replace=False)]
        camera_y = state.floor_y + self.bev_camera_height_above_floor(state.floor_y)
        errors = []
        records = []
        skipped_nonfloor_collision = 0
        for row, col in selected:
            x, z = self.mapping.bev_to_world(float(col), float(row))
            ray = self.habitat_sim.geo.Ray(
                np.array([x, camera_y, z], dtype=np.float32), np.array([0.0, -1.0, 0.0], dtype=np.float32)
            )
            result = self.sim.cast_ray(ray, max_distance=float(self.config.bev_far), buffer_distance=0.0)
            if not result.has_hits():
                continue
            ray_distance = float(result.hits[0].ray_distance)
            ray_height = camera_y - ray_distance - state.floor_y
            if abs(ray_height) > floor_band_m:
                skipped_nonfloor_collision += 1
                continue
            saved_height = float(height_map[row, col])
            error = abs(ray_height - saved_height)
            errors.append(error)
            records.append({
                "pixel_rc": [int(row), int(col)],
                "world_xz": [float(x), float(z)],
                "saved_height_m": saved_height,
                "render_depth_m": float(camera_y - state.floor_y - saved_height),
                "physics_ray_distance_m": ray_distance,
                "absolute_error_m": error,
            })
        return {
            "available": bool(errors), "sample_count": len(errors),
            "skipped_nonfloor_collision_rays": skipped_nonfloor_collision,
            "mean_abs_error_m": float(np.mean(errors)) if errors else None,
            "max_abs_error_m": float(np.max(errors)) if errors else None,
            "records": records,
            "validation_scope": f"physical-floor pixels within {floor_band_m:.3f} m",
            "note": "floor render depth compared with Bullet downward-ray distance",
        }

    def close(self) -> None:
        if getattr(self, "sim", None) is not None:
            self.sim.close()
            self.sim = None
        self._remove_runtime_dataset_config()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
