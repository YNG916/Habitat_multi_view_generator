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
    """Atomically publish JSON so interruption cannot leave a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def save_numeric(path: Path, array: np.ndarray, compress: bool) -> str:
    path = Path(path)
    if compress:
        output = path.with_suffix(".npz")
        np.savez_compressed(output, data=array)
    else:
        output = path.with_suffix(".npy")
        np.save(output, array)
    return output.name


def load_numeric(path: Path) -> np.ndarray:
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if loaded.files != ["data"]:
                raise ValueError(f"Compressed array {path} must contain only key 'data'")
            return np.asarray(loaded["data"])
        finally:
            loaded.close()
    return np.asarray(loaded)


def state_directory_complete(state_dir: Path) -> bool:
    """A published directory is complete when every metadata reference exists."""
    state_dir = Path(state_dir)
    metadata_path = state_dir / "state.json"
    if not metadata_path.is_file():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        references = [metadata["objects_path"], *metadata["bev"]["files"].values()]
        for robot in metadata["robots"]:
            references.extend(robot["files"].values())
        return all((state_dir / reference).is_file() for reference in references)
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


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
        if not state.parent_state_id:
            minimum = int(backend.config.min_target_visible_observers)
            visible_robots = [
                robot
                for robot in state.robots
                if sum(
                    bool(record.get("benchmark_visible", False))
                    for record in robot.visibility.values()
                )
                >= minimum
            ]
            visible_objects = [
                obj
                for obj in state.objects
                if obj.active
                and sum(
                    bool(record.get("benchmark_visible", False))
                    for record in obj.visibility.values()
                )
                >= minimum
            ]
            if backend.config.require_visible_robot_target and not visible_robots:
                raise RuntimeError(
                    "Factual state has no benchmark-visible robot intervention target"
                )
            if backend.config.require_visible_object_target and not visible_objects:
                raise RuntimeError(
                    "Factual state has no benchmark-visible object intervention target"
                )
        clean_bev = Image.fromarray(outputs["bev_rgb"])
        clean_bev.save(temporary / "bev/rgb.png")
        annotated = annotate_bev(outputs["bev_rgb"], mapping, state.robots, backend.config.hfov_deg)
        if backend.config.save_visualizations:
            annotated.save(temporary / "bev/annotated.png")
        height_file = save_numeric(
            temporary / "bev/height",
            outputs["height"].astype(np.float32),
            backend.config.compress_numeric_arrays,
        )
        occupancy_file = save_numeric(
            temporary / "bev/occupancy",
            outputs["occupancy"].astype(np.uint8),
            backend.config.compress_numeric_arrays,
        )
        if outputs["bev_semantic"] is not None:
            bev_semantic_file = save_numeric(
                temporary / "bev/semantic",
                outputs["bev_semantic"].astype(np.uint16),
                backend.config.compress_numeric_arrays,
            )
        if outputs["bev_instance"] is not None:
            bev_instance_file = save_numeric(
                temporary / "bev/instance",
                outputs["bev_instance"].astype(np.int32),
                backend.config.compress_numeric_arrays,
            )

        robot_paths = {}
        robot_images = []
        for robot in state.robots:
            directory = temporary / "robots" / robot.robot_id
            directory.mkdir()
            output = outputs["robots"][robot.robot_id]
            Image.fromarray(output["rgb"]).save(directory / "rgb.png")
            depth_file = save_numeric(
                directory / "depth",
                output["depth"].astype(np.float32),
                backend.config.compress_numeric_arrays,
            )
            paths = {
                "rgb": f"robots/{robot.robot_id}/rgb.png",
                "depth": f"robots/{robot.robot_id}/{depth_file}",
            }
            if backend.config.save_visualizations:
                depth_visualization(output["depth"]).save(directory / "depth_vis.png")
                paths["depth_visualization"] = (
                    f"robots/{robot.robot_id}/depth_vis.png"
                )
            if output["semantic"] is not None:
                semantic_file = save_numeric(
                    directory / "semantic",
                    output["semantic"].astype(np.uint16),
                    backend.config.compress_numeric_arrays,
                )
                paths["semantic"] = f"robots/{robot.robot_id}/{semantic_file}"
            if output["instance"] is not None:
                instance_file = save_numeric(
                    directory / "instance",
                    output["instance"].astype(np.int32),
                    backend.config.compress_numeric_arrays,
                )
                paths["instance"] = f"robots/{robot.robot_id}/{instance_file}"
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
        valid_height_rays = int(height_validation.get("sample_count", 0))
        if not height_validation.get("available") or max_height_error is None:
            raise RuntimeError("BEV height validation produced no comparable render/physics rays")
        if valid_height_rays < backend.config.height_validation_min_samples:
            raise RuntimeError(
                f"BEV height validation returned only {valid_height_rays} valid rays; "
                f"requires {backend.config.height_validation_min_samples}"
            )
        if max_height_error > backend.config.height_validation_max_error_m:
            raise RuntimeError(
                f"BEV height validation failed: {max_height_error:.6f} m exceeds "
                f"{backend.config.height_validation_max_error_m:.6f} m"
            )
        state.bev = {
            **mapping.metadata(),
            "files": {
                "rgb": "bev/rgb.png",
                "height": f"bev/{height_file}",
                "occupancy": f"bev/{occupancy_file}",
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
        if backend.config.save_visualizations:
            state.bev["files"]["annotated"] = "bev/annotated.png"
        if outputs["bev_semantic"] is not None:
            state.bev["files"]["semantic"] = f"bev/{bev_semantic_file}"
            state.bev["semantic_encoding"] = "controlled_entity_category_id"
            state.bev["semantic_scope"] = (
                "robots and generator-controlled movable objects; 0 is unlabeled scene"
            )
            state.bev["semantic_category_ids"] = (
                backend.config.semantic_category_ids
            )
        if outputs["bev_instance"] is not None:
            state.bev["files"]["instance"] = f"bev/{bev_instance_file}"
            state.bev["instance_id_encoding"] = "Habitat SemanticSensorTarget.OBJECT_ID"
            state.bev["entity_object_ids"] = outputs["entity_object_ids"]
        pinhole_validation = backend.validate_pinhole_depth_grid(
            state, outputs["robots"]
        )
        pinhole_samples = int(pinhole_validation.get("sample_count", 0))
        pinhole_median_error = pinhole_validation.get("median_abs_error_m")
        if pinhole_samples < backend.config.pinhole_validation_min_samples:
            raise RuntimeError(
                f"Pinhole calibration returned only {pinhole_samples} static-stage rays; "
                f"requires {backend.config.pinhole_validation_min_samples}"
            )
        if (
            pinhole_median_error is None
            or pinhole_median_error
            > backend.config.pinhole_validation_median_error_m
        ):
            raise RuntimeError(
                "Pinhole off-center median calibration error "
                f"{pinhole_median_error} exceeds "
                f"{backend.config.pinhole_validation_median_error_m:.6f} m"
            )
        state.geometry_validation = {
            "pinhole_depth_center_rays": backend.validate_pinhole_depth_centers(
                state, outputs["robots"]
            ),
            "pinhole_depth_offcenter_grid": pinhole_validation,
        }
        write_json(temporary / "objects.json", [obj.metadata() for obj in state.objects])
        write_json(temporary / "state.json", state.metadata(robot_paths, "objects.json"))
        if backend.config.save_visualizations:
            contact_sheet(
                annotated,
                robot_images,
                f"{state.state_id} | {state.overlap['category']}",
            ).save(temporary / "contact_sheet.png")
        os.replace(temporary, state_dir)
        return state_dir
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
