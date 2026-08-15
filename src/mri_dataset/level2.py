from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .collector import update_dataset_index
from .interventions import Intervention, apply_intervention, canonical_instruction, validate_robot_translation
from .objects import controlled_object_collision_free
from .serialization import save_rendered_state, write_json
from .state_io import load_world_state
from .visualization import before_after_contact_sheet


def visibility_transition(before, after, target_id: str) -> dict:
    try:
        before_entity, after_entity = before.robot(target_id), after.robot(target_id)
    except KeyError:
        before_entity, after_entity = before.object(target_id), after.object(target_id)
    return {
        robot.robot_id: {
            "before_visible": bool(before_entity.visibility.get(robot.robot_id, {}).get("visible", False)),
            "after_visible": bool(after_entity.visibility.get(robot.robot_id, {}).get("visible", False)),
            "before_benchmark_visible": bool(
                before_entity.visibility.get(robot.robot_id, {}).get(
                    "benchmark_visible", False
                )
            ),
            "after_benchmark_visible": bool(
                after_entity.visibility.get(robot.robot_id, {}).get(
                    "benchmark_visible", False
                )
            ),
            "before_visible_pixel_count": before_entity.visibility.get(robot.robot_id, {}).get("visible_pixel_count"),
            "after_visible_pixel_count": after_entity.visibility.get(robot.robot_id, {}).get("visible_pixel_count"),
        }
        for robot in before.robots
    }


def validate_robot_edit(backend, before, state, edit: Intervention) -> None:
    if edit.type == "robot_translate":
        validate_robot_translation(
            before,
            state,
            edit,
            backend.sim.pathfinder,
            backend.config.floor_tolerance_m,
            backend.config.min_inter_robot_distance_m,
        )
        robot = state.robot(edit.target_id)
        nav_target = np.asarray(
            backend.sim.pathfinder.snap_point(robot.base_position_world),
            dtype=np.float64,
        )
        physical_floor_y = backend.floor_surface_y(nav_target)
        original_floor_y = float(
            before.robot(edit.target_id).base_position_world[1]
        )
        if (
            abs(physical_floor_y - original_floor_y)
            > backend.config.floor_tolerance_m
        ):
            raise ValueError("Robot target is outside the same physical floor")
        robot.base_position_world[1] = float(physical_floor_y)
        robot.synchronize_camera()
    elif edit.type != "robot_rotate":
        raise ValueError(f"Unsupported robot edit validation: {edit.type}")

    collision = backend.entity_collision_report(state, edit.target_id)
    if not collision["collision_free"]:
        raise ValueError(
            "Robot edit penetrates scene geometry or another controlled entity: "
            f"{collision['rejected_contacts']}"
        )

def validate_object_edit(backend, state, target_id: str) -> None:
    obj = state.object(target_id)
    if not obj.active:
        return
    query = np.asarray([obj.position_world[0], state.floor_y, obj.position_world[2]])
    floor_point = np.asarray(backend.sim.pathfinder.snap_point(query), dtype=np.float64)
    if (
        not np.all(np.isfinite(floor_point))
        or np.linalg.norm(
            floor_point[[0, 2]]
            - np.asarray(obj.position_world, dtype=np.float64)[[0, 2]]
        )
        > 1e-3
        or not backend.sim.pathfinder.is_navigable(floor_point)
    ):
        raise ValueError("Controlled object target is outside the navigable interior")
    physical_floor_y = backend.floor_surface_y(floor_point)
    if abs(physical_floor_y - state.floor_y) > backend.config.floor_tolerance_m:
        raise ValueError("Controlled object target is outside the same physical floor")
    edit_type = state.intervention.get("type")
    if edit_type == "object_translate":
        requested = np.asarray(state.intervention["displacement_m"], dtype=np.float64)
        if requested.shape != (3,) or abs(float(requested[1])) > 1e-9:
            raise ValueError(
                "Floor-supported object_translate requires displacement_m Y = 0"
            )
    if edit_type in {"object_translate", "object_place_relative"}:
        backend.support_object_on_floor(obj, physical_floor_y)
    if not controlled_object_collision_free(obj, state, backend.render_bev_bounds):
        raise ValueError("Controlled object edit overlaps another controlled entity or leaves visual bounds")
    collision = backend.object_collision_report(state, target_id)
    if not collision["collision_free"]:
        raise ValueError(
            "Controlled object edit penetrates static scene geometry: "
            f"{collision['rejected_contacts']}"
        )


def collect_level2(backend, config, root: Path, num_edits: int, edit_type: str = "robot_translate") -> List[Path]:
    root = Path(root)
    before_dirs = sorted(root.glob(f"scenes/{backend.scene_id}/states/state_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if not before_dirs:
        raise RuntimeError("No Level 1 states found. Run collect_level1.py first.")
    intervention_dir = root / "interventions" / backend.scene_id
    intervention_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    edit_index = len(list(intervention_dir.glob("edit_*.json"))) + 1
    for before_dir in before_dirs:
        if len(saved) >= num_edits:
            break
        before = load_world_state(before_dir)
        if edit_type == "robot_translate":
            edit = Intervention("robot_translate", "robot_02", {"reference_frame": "target_local", "forward_m": 1.0})
        elif edit_type == "robot_rotate":
            edit = Intervention("robot_rotate", "robot_02", {"delta_yaw_rad": float(np.pi / 3.0)})
        elif edit_type == "object_remove":
            if not before.objects:
                continue
            edit = Intervention("object_remove", before.objects[0].instance_id, {})
        elif edit_type == "object_place_relative":
            if not before.objects:
                continue
            edit = Intervention("object_place_relative", before.objects[0].instance_id, {"reference_id": "robot_01", "relation": "front", "distance_m": 1.0})
        elif edit_type == "object_translate":
            if not before.objects:
                continue
            edit = Intervention("object_translate", before.objects[0].instance_id, {"reference_frame": "world", "displacement_m": [0.5, 0.0, 0.0]})
        else:
            raise ValueError(f"Unsupported edit type: {edit_type}")
        after = apply_intervention(before, edit, f"state_after_{edit_index:06d}")
        try:
            if edit.type.startswith("robot_"):
                validate_robot_edit(backend, before, after, edit)
            elif edit.type.startswith("object_"):
                validate_object_edit(backend, after, edit.target_id)
        except ValueError:
            continue  # Reject exact invalid targets; never snap or shorten them.
        after_dir = root / "scenes" / backend.scene_id / "states" / after.state_id
        save_rendered_state(backend, after, after_dir)
        instruction = canonical_instruction(edit, before)
        edit_path = intervention_dir / f"edit_{edit_index:06d}.json"
        write_json(edit_path, {
            "schema_version": "0.1.0", "edit_id": f"edit_{edit_index:06d}", "scene_id": backend.scene_id,
            "before_state_id": before.state_id, "before_state_path": str(before_dir.relative_to(root)),
            "after_state_id": after.state_id, "after_state_path": str(after_dir.relative_to(root)),
            "structured_intervention": edit.to_dict(), "instruction": instruction,
            "instruction_paraphrases": [], "mllm_generated": False, "human_verified": False,
            "visibility_transition": visibility_transition(before, after, edit.target_id),
        })
        before_after_contact_sheet(before_dir, after_dir, instruction).save(intervention_dir / f"edit_{edit_index:06d}_contact_sheet.png")
        saved.append(edit_path)
        edit_index += 1
    if len(saved) < num_edits:
        raise RuntimeError(f"Only {len(saved)}/{num_edits} exact {edit_type} interventions were valid; rejected without snapping")
    update_dataset_index(root)
    return saved
