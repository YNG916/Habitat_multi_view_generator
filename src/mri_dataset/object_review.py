from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .objects import canonical_template_index, inspect_approved_object_registry
from .serialization import write_json


REVIEW_RESOLUTION = 384
VISUAL_FLOOR_OFFSET_LIMIT_M = 0.01
EXTENT_TOLERANCE_M = 1e-4


def review_sensor_specs(habitat_sim):
    specs = []
    for uuid, sensor_type in (
        ("candidate_rgb", habitat_sim.SensorType.COLOR),
        ("candidate_instance", habitat_sim.SensorType.SEMANTIC),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [REVIEW_RESOLUTION, REVIEW_RESOLUTION]
        spec.hfov = 55.0
        spec.near = 0.02
        spec.far = 10.0
        spec.position = [0.0, 0.0, 0.0]
        spec.orientation = [0.0, 0.0, 0.0]
        if uuid == "candidate_instance":
            spec.semantic_target = type(spec.semantic_target).OBJECT_ID
        specs.append(spec)
    return specs


def find_review_anchor(sim, scene_id, config):
    navmesh = config.navmesh_cache_path(scene_id)
    if navmesh.is_file() and not sim.pathfinder.load_nav_mesh(str(navmesh)):
        raise RuntimeError(f"Could not load object-review NavMesh: {navmesh}")
    if not sim.pathfinder.is_loaded:
        raise FileNotFoundError(f"Object-review NavMesh is missing: {navmesh}")
    best = None
    best_clearance = -1.0
    for _ in range(2000):
        point = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float64)
        if not np.all(np.isfinite(point)):
            continue
        clearance = float(sim.pathfinder.distance_to_closest_obstacle(point, 3.0))
        if clearance > best_clearance:
            best, best_clearance = point, clearance
        if clearance >= 1.6:
            return point
    if best is None:
        raise RuntimeError("No navigable object-review anchor")
    return best


def physical_floor_y(sim, habitat_sim, point):
    point = np.asarray(point, dtype=np.float64)
    origin = point.copy()
    origin[1] += 0.35
    ray = habitat_sim.geo.Ray(
        origin.astype(np.float32),
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
    )
    result = sim.cast_ray(ray, max_distance=1.5, buffer_distance=0.0)
    if not result.hits:
        raise RuntimeError(f"Could not resolve physical floor below {point.tolist()}")
    return float(origin[1] - result.hits[0].ray_distance)


def set_review_lighting(sim, habitat_sim):
    import magnum as mn

    lights = [
        habitat_sim.gfx.LightInfo(
            mn.Vector4(1.5, 2.0, 2.0, 1.0),
            mn.Color3(2.5, 2.5, 2.5),
            habitat_sim.gfx.LightPositionModel.Global,
        ),
        habitat_sim.gfx.LightInfo(
            mn.Vector4(-1.5, 1.0, 1.0, 1.0),
            mn.Color3(1.5, 1.5, 1.5),
            habitat_sim.gfx.LightPositionModel.Global,
        ),
        habitat_sim.gfx.LightInfo(
            mn.Vector4(0.0, 2.5, -1.0, 1.0),
            mn.Color3(1.0, 1.0, 1.0),
            habitat_sim.gfx.LightPositionModel.Global,
        ),
    ]
    sim.set_light_setup(lights, "candidate_lights")


def render_supported_object(
    sim,
    handle,
    extent,
    anchor,
    habitat_sim,
    floor_y=None,
):
    """Render identity-oriented rigid geometry using formal collision-AABB support."""
    import magnum as mn

    manager = sim.get_rigid_object_manager()
    obj = manager.add_object_by_template_handle(handle)
    if obj is None:
        raise RuntimeError(f"Cannot instantiate {handle}")
    try:
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        # HSSD templates inherit Habitat's 4 cm default rigid-object margin.
        # It expands the collision AABB and would lift every small object.
        obj.margin = 0.0
        obj.rotation = mn.Quaternion(mn.Vector3(0.0, 0.0, 0.0), 1.0)
        obj.set_light_setup("candidate_lights")
        collision = obj.collision_shape_aabb
        visual = obj.root_scene_node.cumulative_bb
        collision_min = np.asarray(collision.min, dtype=np.float64)
        collision_max = np.asarray(collision.max, dtype=np.float64)
        visual_min = np.asarray(visual.min, dtype=np.float64)
        visual_max = np.asarray(visual.max, dtype=np.float64)
        support_y = float(anchor[1] if floor_y is None else floor_y)
        position = np.asarray(anchor, dtype=np.float64).copy()
        position[1] = support_y - float(collision_min[1])
        obj.translation = position.astype(np.float32)

        center_y = float(position[1] + (visual_min[1] + visual_max[1]) * 0.5)
        extent = np.asarray(extent, dtype=np.float64)
        horizontal = max(float(extent[0]), float(extent[2]))
        distance = max(0.45, horizontal * 2.2, float(extent[1]) * 2.0)
        views = (
            ((0.0, distance), 0.0),
            ((0.0, -distance), math.pi),
            ((distance, 0.0), math.pi / 2),
            ((-distance, 0.0), -math.pi / 2),
        )
        best_image = None
        best_pixels = -1
        view_pixels = []
        for (dx, dz), yaw in views:
            camera = habitat_sim.AgentState()
            camera.position = np.asarray(
                [anchor[0] + dx, center_y, anchor[2] + dz],
                np.float32,
            )
            camera.rotation = habitat_sim.utils.common.quat_from_angle_axis(
                yaw, np.array([0.0, 1.0, 0.0])
            )
            sim.get_agent(0).set_state(camera, infer_sensor_states=True)
            observations = sim.get_sensor_observations(agent_ids=[0])[0]
            instance = np.asarray(observations["candidate_instance"])
            pixels = int((instance == obj.object_id).sum())
            view_pixels.append(pixels)
            if pixels > best_pixels:
                best_pixels = pixels
                rgb = np.asarray(observations["candidate_rgb"])[..., :3].astype(np.uint8)
                locations = np.argwhere(instance == obj.object_id)
                if len(locations):
                    low = locations.min(axis=0)
                    high = locations.max(axis=0) + 1
                    pad = max(8, int(0.15 * max(*(high - low))))
                    r0 = max(0, int(low[0]) - pad)
                    r1 = min(rgb.shape[0], int(high[0]) + pad)
                    c0 = max(0, int(low[1]) - pad)
                    c1 = min(rgb.shape[1], int(high[1]) + pad)
                    best_image = Image.fromarray(rgb[r0:r1, c0:c1]).resize(
                        (REVIEW_RESOLUTION, REVIEW_RESOLUTION)
                    )
        if best_pixels <= 0 or best_image is None:
            raise RuntimeError(f"Object has zero OBJECT_ID pixels: {handle}")
        geometry = {
            "object_id": int(obj.object_id),
            "position_world": position.tolist(),
            "collision_aabb_local": {
                "min": collision_min.tolist(),
                "max": collision_max.tolist(),
                "extent": (collision_max - collision_min).tolist(),
            },
            "visual_aabb_local": {
                "min": visual_min.tolist(),
                "max": visual_max.tolist(),
                "extent": (visual_max - visual_min).tolist(),
            },
            "physical_floor_y": support_y,
            "collision_support_error_m": float(
                position[1] + collision_min[1] - support_y
            ),
            "visual_floor_offset_m": float(
                position[1] + visual_min[1] - support_y
            ),
            "object_id_pixels_by_view": view_pixels,
            "best_object_id_pixels": best_pixels,
        }
        return best_image, best_pixels, geometry
    finally:
        manager.remove_object_by_id(obj.object_id)


def object_contact_sheet(category, records, images, path):
    cell_w, cell_h, columns = REVIEW_RESOLUTION, 430, 4
    rows = max(1, math.ceil(len(images) / columns))
    sheet = Image.new(
        "RGB", (columns * cell_w, rows * cell_h), (20, 20, 20)
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (record, image) in enumerate(zip(records, images)):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        sheet.paste(image.resize((cell_w, REVIEW_RESOLUTION)), (x, y))
        extent = record["extent_xyz_m"]
        label = (
            f"{index:02d} {category}  "
            f"{extent[0]:.2f}x{extent[1]:.2f}x{extent[2]:.2f}m"
        )
        draw.text((x + 5, y + 389), label, fill="white", font=font)
        draw.text(
            (x + 5, y + 405),
            Path(record["canonical_id"]).stem[:48],
            fill=(190, 220, 255),
            font=font,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _create_review_simulator(config, scene_id):
    import habitat_sim

    simulator = habitat_sim.SimulatorConfiguration()
    simulator.scene_dataset_config_file = str(config.dataset_config_path)
    simulator.scene_id = scene_id
    simulator.enable_physics = True
    simulator.gpu_device_id = int(config.gpu_device_id)
    simulator.override_scene_light_defaults = True
    simulator.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = review_sensor_specs(habitat_sim)
    sim = habitat_sim.Simulator(habitat_sim.Configuration(simulator, [agent]))
    set_review_lighting(sim, habitat_sim)
    return sim, habitat_sim


def run_approved_object_preflight(
    config,
    report_path=None,
    contact_sheet_root=None,
    raise_on_error=False,
):
    """Instantiate and render every approved HSSD asset before formal collection."""
    registry = json.loads(
        config.controlled_object_registry_path.read_text(encoding="utf-8")
    )
    dataset_root = config.dataset_config_path.resolve().parent
    pure = inspect_approved_object_registry(
        registry, dataset_root, config.semantic_category_ids
    )
    report_path = Path(
        report_path
        or config.repo_root / "outputs/hssd_object_preflight.json"
    )
    contact_sheet_root = Path(
        contact_sheet_root
        or config.repo_root / "outputs/hssd_object_review"
    )
    report = {
        "schema_version": "1.0.0",
        "dataset_source": "hssd",
        "registry_path": str(config.controlled_object_registry_path),
        "scene_dataset_config": str(config.dataset_config_path),
        "registry_validation_passed": bool(pure["passed"]),
        "scene_id": None,
        "physical_floor_y": None,
        "records": [],
        "contact_sheets": {},
        "errors": list(pure["errors"]),
    }
    if not pure["passed"]:
        report["passed"] = False
        write_json(report_path, report)
        if raise_on_error:
            raise RuntimeError(
                "Approved-object registry preflight failed: "
                + "; ".join(report["errors"][:8])
            )
        return report

    specs = config.collection_specs()
    if not specs:
        report["errors"].append("No eligible HSSD scene is available for object preflight")
        report["passed"] = False
        write_json(report_path, report)
        if raise_on_error:
            raise RuntimeError(report["errors"][-1])
        return report
    scene_id = specs[0][0].scene_id
    report["scene_id"] = scene_id
    images_by_category = {}
    records_by_category = {}
    try:
        sim, habitat_sim = _create_review_simulator(config, scene_id)
    except Exception as exc:
        report["errors"].append(f"could not create Habitat review simulator: {exc}")
        report["asset_count"] = 0
        report["category_counts"] = {}
        report["passed"] = False
        write_json(report_path, report)
        if raise_on_error:
            raise RuntimeError(report["errors"][-1]) from exc
        return report
    try:
        anchor = find_review_anchor(sim, scene_id, config)
        floor_y = physical_floor_y(sim, habitat_sim, anchor)
        report["review_anchor_world"] = anchor.tolist()
        report["physical_floor_y"] = floor_y
        manager = sim.get_object_template_manager()
        handle_index = canonical_template_index(
            manager.get_template_handles(), dataset_root
        )
        source_records = {
            (str(category), str(record["canonical_id"])): record
            for category, values in registry["approved_assets"].items()
            for record in values
        }
        for pure_record in pure["records"]:
            category = pure_record["category"]
            canonical = pure_record["canonical_id"]
            source = source_records[(category, canonical)]
            record = dict(pure_record)
            record["runtime_handle_match_count"] = len(
                handle_index.get(canonical, [])
            )
            record["runtime_handle"] = None
            record["resolved_handle"] = None
            record["template_semantic_id"] = None
            record["semantic_id_actual"] = None
            record["collision_asset"] = None
            record["instantiate_ok"] = False
            record["collider_ok"] = False
            record["render_object_id_pixels"] = 0
            record["dynamic_errors"] = []
            image = Image.new(
                "RGB",
                (REVIEW_RESOLUTION, REVIEW_RESOLUTION),
                (80, 0, 0),
            )
            matches = handle_index.get(canonical, [])
            if len(matches) != 1:
                record["dynamic_errors"].append(
                    f"canonical template resolved to {len(matches)} runtime handles"
                )
            else:
                handle = matches[0]
                record["runtime_handle"] = handle
                record["resolved_handle"] = handle
                attributes = manager.get_template_by_handle(handle)
                record["template_semantic_id"] = int(attributes.semantic_id)
                record["semantic_id_actual"] = int(attributes.semantic_id)
                record["collision_asset"] = str(attributes.collision_asset_handle)
                if int(attributes.semantic_id) != int(source["semantic_id"]):
                    record["dynamic_errors"].append(
                        "runtime template semantic_id does not match registry"
                    )
                try:
                    image, pixels, geometry = render_supported_object(
                        sim,
                        handle,
                        source["extent_xyz_m"],
                        anchor,
                        habitat_sim,
                        floor_y=floor_y,
                    )
                except Exception as exc:
                    record["dynamic_errors"].append(
                        f"instantiation/render failed: {exc}"
                    )
                else:
                    record["instantiate_ok"] = True
                    record["render_object_id_pixels"] = int(pixels)
                    record.update(geometry)
                    visual_extent = np.asarray(
                        geometry["visual_aabb_local"]["extent"],
                        dtype=np.float64,
                    )
                    expected_extent = np.asarray(
                        source["extent_xyz_m"], dtype=np.float64
                    )
                    record["extent_max_abs_error_m"] = float(
                        np.max(np.abs(visual_extent - expected_extent))
                    )
                    collision_extent = np.asarray(
                        geometry["collision_aabb_local"]["extent"],
                        dtype=np.float64,
                    )
                    record["collision_shape_usable"] = bool(
                        np.all(np.isfinite(collision_extent))
                        and np.all(collision_extent > 0)
                    )
                    record["collider_ok"] = record["collision_shape_usable"]
                    record["extent_xyz_m"] = visual_extent.tolist()
                    if record["extent_max_abs_error_m"] > EXTENT_TOLERANCE_M:
                        record["dynamic_errors"].append(
                            "visual AABB extent does not match approved registry"
                        )
                    if not record["collision_shape_usable"]:
                        record["dynamic_errors"].append(
                            "collision AABB is missing or degenerate"
                        )
                    if abs(geometry["collision_support_error_m"]) > 1e-5:
                        record["dynamic_errors"].append(
                            "collision shape is not supported on the physical floor"
                        )
                    if abs(geometry["visual_floor_offset_m"]) > VISUAL_FLOOR_OFFSET_LIMIT_M:
                        record["dynamic_errors"].append(
                            "visual geometry is visibly floating above or penetrating the floor"
                        )
                    if pixels <= 0:
                        record["dynamic_errors"].append(
                            "OBJECT_ID rendering contains no object pixels"
                        )
            record["passed"] = bool(
                pure_record["passed"] and not record["dynamic_errors"]
            )
            record["validation_passed"] = record["passed"]
            report["records"].append(record)
            records_by_category.setdefault(category, []).append(source)
            images_by_category.setdefault(category, []).append(image)
            report["errors"].extend(
                f"{category}/{canonical}: {message}"
                for message in record["dynamic_errors"]
            )

        for category in sorted(records_by_category):
            sheet_path = contact_sheet_root / f"{category}.png"
            object_contact_sheet(
                category,
                records_by_category[category],
                images_by_category[category],
                sheet_path,
            )
            report["contact_sheets"][category] = str(sheet_path)
    except Exception as exc:
        report["errors"].append(f"approved-object preflight execution failed: {exc}")
    finally:
        sim.close()

    report["asset_count"] = len(report["records"])
    report["category_counts"] = {
        category: len(values)
        for category, values in sorted(records_by_category.items())
    }
    report["passed"] = not report["errors"] and all(
        record["passed"] for record in report["records"]
    )
    write_json(report_path, report)
    if raise_on_error and not report["passed"]:
        raise RuntimeError(
            "Approved-object Habitat preflight failed: "
            + "; ".join(report["errors"][:8])
        )
    return report
