from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .collector import initialize_dataset_root, update_dataset_index
from .interventions import Intervention, apply_intervention, canonical_instruction, validate_robot_translation
from .objects import controlled_object_collision_free
from .protocol import (
    benchmark_visible_observers,
    intervention_key,
    sample_intervention,
    stable_seed,
)
from .serialization import (
    save_rendered_state,
    state_directory_complete,
    write_json,
)
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
            backend.config.controlled_object_min_separation_m,
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
        if not backend.point_in_region(nav_target):
            raise ValueError("Robot intervention crosses semantic region")
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
    membership = backend.object_region_membership(obj.position_world, state.floor_y)
    if not membership["passed"]:
        raise ValueError(
            "Controlled object intervention leaves selected region: "
            + "; ".join(membership["reasons"])
        )
    physical_floor_y = float(membership["physical_floor_y"])
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


def _validate_observable_transition(before,after,edit:Intervention,minimum:int)->None:
    before_entity=(
        before.robot(edit.target_id) if edit.target_id.startswith("robot_")
        else before.object(edit.target_id)
    )
    after_entity=(
        after.robot(edit.target_id) if edit.target_id.startswith("robot_")
        else after.object(edit.target_id)
    )
    before_views=benchmark_visible_observers(before_entity)
    after_views=benchmark_visible_observers(after_entity)
    if before_views<int(minimum):
        raise ValueError("Intervention target is not benchmark-visible before the edit")
    if edit.type=="object_remove":
        if after_entity.active or after_views!=0:
            raise ValueError("Removed object must be inactive and invisible after the edit")
    elif after_views<int(minimum):
        raise ValueError("Intervention target is not benchmark-visible after the edit")


def _validate_edit(backend,before,after,edit:Intervention)->None:
    if before.region_id!=backend.region_id or after.region_id!=backend.region_id:
        raise ValueError("Level-2 before/after region mismatch")
    if before.bev and after.bev and before.bev.get("bounds_world")!=after.bev.get("bounds_world"):
        raise ValueError("Level-2 before/after BEV frame mismatch")
    if edit.type.startswith("robot_"):
        validate_robot_edit(backend, before, after, edit)
    elif edit.type.startswith("object_"):
        validate_object_edit(backend, after, edit.target_id)
    else:
        raise ValueError(f"Unsupported intervention type: {edit.type}")


def _edit_identifiers(before_state_id: str, regime: str, slot: int) -> tuple:
    suffix = before_state_id.removeprefix("state_")
    edit_id = f"edit_{suffix}_{slot:03d}_{regime}"
    after_state_id = f"state_after_{suffix}_{slot:03d}_{regime}"
    return edit_id, after_state_id


def _sampling_candidate(
    before,
    config,
    regime: str,
    slot: int,
    attempt: int,
    requested_type: Optional[str],
):
    seed = stable_seed(
        config.random_seed,
        config.protocol_version,
        before.scene_id,
        before.floor_id,
        before.region_id,
        before.state_id,
        regime,
        slot,
        attempt,
    )
    edit = sample_intervention(before, config, regime, seed, requested_type)
    return edit, seed


def _write_edit_record(
    path: Path,
    root: Path,
    before_dir: Path,
    after_dir: Path,
    before,
    after,
    edit: Intervention,
    config,
    regime: str,
    slot: int,
    sampling_seed: int,
    sampling_attempt: int,
    recovered: bool = False,
) -> None:
    split = config.scene_split(before.scene_id)
    write_json(
        path,
        {
            "schema_version": "1.0.0",
            "edit_id": path.stem,
            "scene_id": before.scene_id,
            "dataset_source": "hssd",
            "floor_id":before.floor_id,
            "region_id":before.region_id,
            "region_category":before.region_category,
            "bev_scope":"semantic_region",
            "split": split,
            "benchmark_regime": regime,
            "protocol_version": config.protocol_version,
            "before_state_id": before.state_id,
            "before_state_path": str(before_dir.relative_to(root)),
            "after_state_id": after.state_id,
            "after_state_path": str(after_dir.relative_to(root)),
            "structured_intervention": edit.to_dict(),
            "instruction": canonical_instruction(edit, before),
            "instruction_paraphrases": [],
            "mllm_generated": False,
            "human_verified": False,
            "sampling": {
                "slot": int(slot),
                "attempt": int(sampling_attempt),
                "seed": int(sampling_seed),
                "recovered_after_interruption": bool(recovered),
            },
            "target_visible_observers_before": benchmark_visible_observers(
                before.robot(edit.target_id)
                if edit.target_id.startswith("robot_")
                else before.object(edit.target_id)
            ),
            "visibility_transition":visibility_transition(before,after,edit.target_id),
            "observable_edit":{
                "criterion":"visible_before_then_absent_after" if edit.type=="object_remove" else "visible_before_and_after",
                "before_visible_views":benchmark_visible_observers(before.robot(edit.target_id) if edit.target_id.startswith("robot_") else before.object(edit.target_id)),
                "after_visible_views":benchmark_visible_observers(after.robot(edit.target_id) if edit.target_id.startswith("robot_") else after.object(edit.target_id)),
            },
            "bev_frame_identical":before.bev.get("bounds_world")==after.bev.get("bounds_world"),
        },
    )


def _recover_edit_record(
    edit_path: Path,
    root: Path,
    before_dir: Path,
    after_dir: Path,
    before,
    config,
    regime: str,
    slot: int,
    requested_type: Optional[str],
) -> Intervention:
    if not state_directory_complete(after_dir):
        raise RuntimeError(f"Interrupted after-state is incomplete: {after_dir}")
    after = load_world_state(after_dir)
    stored_key = intervention_key(Intervention.from_dict(after.intervention))
    for attempt in range(int(config.max_intervention_sampling_attempts)):
        try:
            candidate, seed = _sampling_candidate(
                before, config, regime, slot, attempt, requested_type
            )
        except ValueError:
            continue
        if intervention_key(candidate) == stored_key:
            _write_edit_record(
                edit_path,
                root,
                before_dir,
                after_dir,
                before,
                after,
                candidate,
                config,
                regime,
                slot,
                seed,
                attempt,
                recovered=True,
            )
            return candidate
    raise RuntimeError(
        f"Could not reconstruct sampling metadata for interrupted state {after_dir}"
    )


def collect_level2(
    backend,
    config,
    root: Path,
    num_edits_per_state: Optional[int] = None,
    edit_type: Optional[str] = None,
    regimes: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Generate deterministic Level-2 pairs for each factual input state.

    ``num_edits_per_state`` is applied independently to every requested regime.
    Train defaults to ID only; validation/test default to both ID and OOD.
    Existing complete slots are skipped when ``config.resume`` is enabled.
    """
    root = Path(root)
    initialize_dataset_root(root, config)
    before_dirs = sorted(
        root.glob(
            f"scenes/{backend.scene_id}/floors/{backend.floor_id}/regions/{backend.region_id}/states/"
            "state_[0-9][0-9][0-9][0-9][0-9][0-9]"
        )
    )
    if not before_dirs:
        raise RuntimeError("No Level 1 states found. Run collect_level1.py first.")
    split = config.scene_split(backend.scene_id)
    allowed_regimes = list(config.level2_regimes_by_split.get(split, []))
    selected_regimes = list(regimes) if regimes is not None else allowed_regimes
    if not set(selected_regimes).issubset(allowed_regimes):
        raise ValueError(
            f"Regimes {selected_regimes} are not allowed for split {split}; "
            f"expected a subset of {allowed_regimes}"
        )
    slots = (
        int(config.num_edits_per_state)
        if num_edits_per_state is None
        else int(num_edits_per_state)
    )
    if slots < 0:
        raise ValueError("num_edits_per_state cannot be negative")

    intervention_dir=root/"interventions"/backend.scene_id/backend.floor_id/backend.region_id
    intervention_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    resumed = 0
    recovered = 0
    failures = Counter()
    for before_dir in before_dirs:
        before = load_world_state(before_dir)
        if before.floor_id!=backend.floor_id or before.region_id!=backend.region_id or before.dataset_source!="hssd":
            raise ValueError("Level-2 source state does not match region backend")
        for regime in selected_regimes:
            accepted_keys = set()
            for slot in range(1, slots + 1):
                edit_id, after_state_id = _edit_identifiers(
                    before.state_id, regime, slot
                )
                edit_path = intervention_dir / f"{edit_id}.json"
                after_dir = (
                    root
                    / "scenes"
                    / backend.scene_id
                    / "floors"
                    / backend.floor_id
                    / "regions"
                    / backend.region_id
                    / "states"
                    / after_state_id
                )
                if edit_path.exists() and after_dir.exists():
                    if not config.resume or not state_directory_complete(after_dir):
                        raise RuntimeError(
                            f"Cannot resume inconsistent Level-2 slot {edit_id}"
                        )
                    resumed += 1
                    with edit_path.open("r", encoding="utf-8") as handle:
                        import json
                        edit_record=json.load(handle)
                    existing_edit=Intervention.from_dict(
                        edit_record["structured_intervention"]
                    )
                    accepted_keys.add(intervention_key(existing_edit))
                    contact_path=intervention_dir/f"{edit_id}_contact_sheet.png"
                    if config.save_visualizations and not contact_path.is_file():
                        before_after_contact_sheet(
                            before_dir,after_dir,edit_record["instruction"]
                        ).save(contact_path)
                    continue
                if edit_path.exists() and not after_dir.exists():
                    raise RuntimeError(
                        f"Edit metadata exists but after-state is missing: {edit_path}"
                    )
                if after_dir.exists():
                    if not config.resume:
                        raise FileExistsError(after_dir)
                    recovered_edit = _recover_edit_record(
                        edit_path,
                        root,
                        before_dir,
                        after_dir,
                        before,
                        config,
                        regime,
                        slot,
                        edit_type,
                    )
                    accepted_keys.add(intervention_key(recovered_edit))
                    recovered += 1
                    saved.append(edit_path)
                    continue

                last_error = None
                for attempt in range(int(config.max_intervention_sampling_attempts)):
                    try:
                        edit, seed = _sampling_candidate(
                            before, config, regime, slot, attempt, edit_type
                        )
                        key = intervention_key(edit)
                        if key in accepted_keys:
                            raise ValueError(
                                "Duplicate intervention candidate within state/regime"
                            )
                        after = apply_intervention(before, edit, after_state_id)
                        _validate_edit(backend, before, after, edit)
                        save_rendered_state(
                            backend,after,after_dir,
                            post_render_validator=lambda rendered,_outputs: _validate_observable_transition(
                                before,rendered,edit,config.min_target_visible_observers
                            ),
                        )
                        _write_edit_record(
                            edit_path,
                            root,
                            before_dir,
                            after_dir,
                            before,
                            after,
                            edit,
                            config,
                            regime,
                            slot,
                            seed,
                            attempt,
                        )
                        if config.save_visualizations:
                            instruction = canonical_instruction(edit, before)
                            before_after_contact_sheet(
                                before_dir, after_dir, instruction
                            ).save(intervention_dir / f"{edit_id}_contact_sheet.png")
                        accepted_keys.add(key)
                        saved.append(edit_path)
                        break
                    except (KeyError, RuntimeError, ValueError) as exc:
                        last_error = exc
                        message = f"{type(exc).__name__}: {exc}"
                        failures[message[:500]] += 1
                else:
                    raise RuntimeError(
                        f"Could not generate valid {regime} slot {slot} for "
                        f"{before.state_id} after "
                        f"{config.max_intervention_sampling_attempts} attempts: "
                        f"{last_error}"
                    ) from last_error

    write_json(
        intervention_dir / "level2_collection_status.json",
        {
            "scene_id": backend.scene_id,
            "split":split,
            "region_id":backend.region_id,
            "region_category":backend.region_category,
            "regimes": selected_regimes,
            "factual_states": len(before_dirs),
            "target_edits_per_state_per_regime": slots,
            "new_edits": len(saved),
            "resumed_edits": resumed,
            "recovered_edits": recovered,
            "rejections": sum(failures.values()),
            "rejection_reasons": dict(failures.most_common()),
            "complete": len(saved) + resumed
            == len(before_dirs) * len(selected_regimes) * slots,
        },
    )
    update_dataset_index(root)
    return saved
