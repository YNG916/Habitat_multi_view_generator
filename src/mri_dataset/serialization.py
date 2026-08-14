from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .bev import annotate_bev, compute_fov_overlap
from .visualization import contact_sheet, depth_visualization


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")


def save_rendered_state(backend, state, state_dir: Path) -> Path:
    """Render and atomically publish one complete state directory."""
    state_dir = Path(state_dir)
    state_dir.parent.mkdir(parents=True, exist_ok=True)
    if state_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing state: {state_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{state_dir.name}.tmp-", dir=str(state_dir.parent)))
    try:
        (temporary / "bev").mkdir()
        (temporary / "robots").mkdir()
        outputs = backend.render(state)
        mapping = backend.mapping
        clean_bev = Image.fromarray(outputs["bev_rgb"])
        clean_bev.save(temporary / "bev/rgb.png")
        annotated = annotate_bev(outputs["bev_rgb"], mapping, state.robots, backend.config.hfov_deg)
        annotated.save(temporary / "bev/annotated.png")
        np.save(temporary / "bev/height.npy", outputs["height"].astype(np.float32))
        np.save(temporary / "bev/occupancy.npy", outputs["occupancy"].astype(np.uint8))
        if outputs["bev_instance"] is not None:
            np.save(temporary / "bev/instance.npy", outputs["bev_instance"].astype(np.int32))

        robot_paths = {}
        robot_images = []
        for robot in state.robots:
            directory = temporary / "robots" / robot.robot_id
            directory.mkdir()
            output = outputs["robots"][robot.robot_id]
            Image.fromarray(output["rgb"]).save(directory / "rgb.png")
            np.save(directory / "depth.npy", output["depth"].astype(np.float32))
            depth_visualization(output["depth"]).save(directory / "depth_vis.png")
            paths = {
                "rgb": f"robots/{robot.robot_id}/rgb.png",
                "depth": f"robots/{robot.robot_id}/depth.npy",
                "depth_visualization": f"robots/{robot.robot_id}/depth_vis.png",
            }
            if output["instance"] is not None:
                np.save(directory / "instance.npy", output["instance"].astype(np.int32))
                paths["instance"] = f"robots/{robot.robot_id}/instance.npy"
            robot_paths[robot.robot_id] = paths
            robot_images.append(Image.fromarray(output["rgb"]))

        state.overlap = compute_fov_overlap(mapping, state.robots, backend.config.hfov_deg)
        camera_height = backend.bev_camera_height_above_floor(state.floor_y)
        height_validation = backend.validate_height_rays(
            state,
            outputs["height"],
            samples=backend.config.height_validation_samples,
            seed=state.random_seed,
        )
        max_height_error = height_validation.get("max_abs_error_m")
        if not height_validation.get("available") or max_height_error is None:
            raise RuntimeError("BEV height validation produced no comparable render/physics rays")
        if max_height_error > backend.config.height_validation_max_error_m:
            raise RuntimeError(
                f"BEV height validation failed: {max_height_error:.6f} m exceeds "
                f"{backend.config.height_validation_max_error_m:.6f} m"
            )
        state.bev = {
            **mapping.metadata(),
            "files": {
                "rgb": "bev/rgb.png", "annotated": "bev/annotated.png",
                "height": "bev/height.npy", "occupancy": "bev/occupancy.npy",
            },
            "camera_position_world": [
                0.5 * (mapping.x_min + mapping.x_max), state.floor_y + camera_height,
                0.5 * (mapping.z_min + mapping.z_max),
            ],
            "camera_quaternion_world_xyzw": [-2 ** -0.5, 0.0, 0.0, 2 ** -0.5],
            "orthographic_scale": 1.0 / (mapping.x_max - mapping.x_min),
            "orthographic_extent_x_m": mapping.x_max - mapping.x_min,
            "orthographic_extent_z_m": mapping.z_max - mapping.z_min,
            "camera_height_above_floor_m": camera_height,
            "render_resolution_hw": [mapping.height, mapping.width],
            "bounds_source": "rendered_scene_aabb",
            "navmesh_bounds_world": [
                backend.navmesh_bounds[0].tolist(), backend.navmesh_bounds[1].tolist()
            ],
            "rgb_representation": "interior top-down cutaway rendered below the ceiling",
            "ceiling_clearance_m": float(backend.config.bev_ceiling_clearance_m),
            "height_definition": "surface_world_y - floor_y",
            "height_depth_conversion": "v0.3.3 orthographic generic-unprojection output is linearized with near/far, then height = camera_height_above_floor - metric_depth",
            "height_depth_validation": height_validation,
        }
        if outputs["bev_instance"] is not None:
            state.bev["files"]["instance"] = "bev/instance.npy"
            state.bev["instance_id_encoding"] = "Habitat SemanticSensorTarget.OBJECT_ID"
            state.bev["entity_object_ids"] = outputs["entity_object_ids"]
        state.geometry_validation = {
            "pinhole_depth_center_rays": backend.validate_pinhole_depth_centers(state, outputs["robots"]),
        }
        write_json(temporary / "objects.json", [obj.metadata() for obj in state.objects])
        write_json(temporary / "state.json", state.metadata(robot_paths, "objects.json"))
        contact_sheet(annotated, robot_images, f"{state.state_id} | {state.overlap['category']}").save(
            temporary / "contact_sheet.png"
        )
        os.replace(temporary, state_dir)
        return state_dir
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
