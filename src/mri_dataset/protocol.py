from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable, Optional

import numpy as np

from .interventions import Intervention


SUPPORTED_INTERVENTIONS = (
    "robot_translate",
    "robot_rotate",
    "object_translate",
    "object_place_relative",
    "object_remove",
)


def stable_seed(base_seed: int, *parts: object) -> int:
    """Return a process-independent uint32 seed for a logical sample slot."""
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def intervention_key(edit: Intervention) -> str:
    return json.dumps(edit.to_dict(), sort_keys=True, separators=(",", ":"))


def benchmark_visible_observers(entity) -> int:
    return sum(
        bool(record.get("benchmark_visible", False))
        for record in entity.visibility.values()
    )


def _eligible(items: Iterable, minimum_observers: int) -> list:
    return [
        item
        for item in items
        if benchmark_visible_observers(item) >= minimum_observers
    ]


def _choose(rng: np.random.Generator, values):
    values = list(values)
    if not values:
        raise ValueError("Cannot sample from an empty domain")
    return values[int(rng.integers(0, len(values)))]


def _available_types(state, config) -> dict:
    minimum = int(config.min_target_visible_observers)
    robots = _eligible(state.robots, minimum)
    objects = _eligible((obj for obj in state.objects if obj.active and obj.movable), minimum)
    return {
        "robot_translate": robots,
        "robot_rotate": robots,
        "object_translate": objects,
        "object_place_relative": objects if state.robots else [],
        "object_remove": objects,
    }


def sample_intervention(
    state,
    config,
    regime: str,
    seed: int,
    requested_type: Optional[str] = None,
) -> Intervention:
    """Sample one deterministic, information-complete intervention candidate.

    Geometry feasibility is deliberately checked by the Habitat backend after
    this pure protocol sampler returns. Rejected candidates are resampled with a
    different stable attempt seed and are never snapped or shortened.
    """
    if regime not in config.intervention_regimes:
        raise ValueError(f"Unknown intervention regime: {regime}")
    if requested_type is not None and requested_type not in SUPPORTED_INTERVENTIONS:
        raise ValueError(f"Unsupported intervention type: {requested_type}")

    rng = np.random.default_rng(int(seed))
    available = _available_types(state, config)
    if requested_type is None:
        types = [kind for kind in config.intervention_type_weights if available[kind]]
        if not types:
            raise ValueError("No benchmark-visible intervention target is available")
        weights = np.asarray(
            [float(config.intervention_type_weights[kind]) for kind in types],
            dtype=np.float64,
        )
        weights /= weights.sum()
        edit_type = str(rng.choice(types, p=weights))
    else:
        edit_type = requested_type
        if not available[edit_type]:
            raise ValueError(
                f"No benchmark-visible target is available for {edit_type}"
            )

    target = _choose(rng, available[edit_type])
    specification = config.intervention_regimes[regime]
    if edit_type == "robot_translate":
        distance = float(_choose(rng, specification["robot_translate_m"]))
        return Intervention(
            edit_type,
            target.robot_id,
            {"reference_frame": "target_local", "forward_m": distance},
        )
    if edit_type == "robot_rotate":
        degrees = float(_choose(rng, specification["robot_rotate_deg"]))
        return Intervention(
            edit_type,
            target.robot_id,
            {"delta_yaw_rad": float(math.radians(degrees))},
        )
    if edit_type == "object_remove":
        return Intervention(edit_type, target.instance_id, {})
    if edit_type == "object_place_relative":
        reference = _choose(rng, state.robots)
        distance = float(_choose(rng, specification["object_place_relative_m"]))
        return Intervention(
            edit_type,
            target.instance_id,
            {
                "reference_id": reference.robot_id,
                "reference_frame": "reference_robot",
                "relation": "front",
                "distance_m": distance,
            },
        )

    distance = float(_choose(rng, specification["object_translate_m"]))
    axis = int(_choose(rng, (0, 2)))
    sign = float(_choose(rng, (-1.0, 1.0)))
    displacement = [0.0, 0.0, 0.0]
    displacement[axis] = sign * distance
    if bool(rng.integers(0, 2)):
        return Intervention(
            edit_type,
            target.instance_id,
            {"reference_frame": "world", "displacement_m": displacement},
        )
    reference = _choose(rng, state.robots)
    return Intervention(
        edit_type,
        target.instance_id,
        {
            "reference_frame": "reference_robot",
            "reference_id": reference.robot_id,
            "displacement_m": displacement,
        },
    )


def protocol_descriptor(config) -> dict:
    return {
        "protocol_version": config.protocol_version,
        "scene_splits": config.scene_splits,
        "level2_regimes_by_split": config.level2_regimes_by_split,
        "intervention_type_weights": config.intervention_type_weights,
        "intervention_regimes": config.intervention_regimes,
        "minimum_target_visible_observers": config.min_target_visible_observers,
        "benchmark_visibility_min_pixels":config.benchmark_visibility_min_pixels,
        "benchmark_visibility_min_fraction":config.benchmark_visibility_min_fraction,
        "bev_scope":"semantic_region",
        "region_context_margin_m":config.region_context_margin_m,
        "controlled_object_pools":config.controlled_object_pools,
        "controlled_object_sampling":"categories_without_replacement_then_variant_within_category",
        "controlled_object_runtime_collision_margin_m":0.0,
        "seed_derivation":"sha256(base_seed|scene|floor|region|state|regime|slot|attempt)[:32-bit]",
    }
