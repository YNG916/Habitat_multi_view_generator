# Multi-Robot World-Imagination Dataset Generator

This repository contains a Habitat-Sim 0.3.3 pipeline whose source of truth is a serializable `WorldState`. Both Level 1 states and Level 2 counterfactual after-states pass through the same Habitat renderer. Habitat agents are invisible camera rigs; separately spawned colored kinematic proxies make robots visible to one another.

The frozen geometry convention is right-handed Habitat world coordinates (`+Y` up, ground `X-Z`, zero-yaw forward `-Z`). BEV right is `+X` and BEV up is `-Z`. Metadata stores xyzw quaternions, Habitat and OpenCV camera transforms, calibrated pinhole intrinsics, actual BEV pixel scale, and relative file paths.

The default production preset renders robot views at `2048 x 2048` and
targets `0.00625 m/pixel` for BEV. Use `configs/collector_pilot.json`
(`768 x 768`, `0.0125 m/pixel`) for quick correctness tests. Visual BEV
bounds come from the rendered scene AABB; NavMesh bounds are stored separately
and occupancy is computed once per scene/floor with Habitat's top-down API.

Habitat-Sim 0.3.3 pinhole depth is metric, but its generic unprojection pass
does not produce linear distance for an orthographic projection. BEV depth is
therefore orthographically linearized before computing
`camera_height_above_floor - metric_depth`; a real Habitat/Bullet integration
test guards this behavior and every state is rejected if ray errors exceed 2 cm.
Instance arrays explicitly target
`SemanticSensorTarget.OBJECT_ID`; per-state runtime object-ID mappings are
stored in metadata. Level-2 object edits are checked against Bullet contacts,
including static ReplicaCAD furniture and walls.

Robot proxies are lightweight 32-sided robot-vacuum meshes with a layered shell,
rubber bumper, drive wheels, side brush, lidar turret, status button, and
front/rear sensor windows. Regenerate these self-contained assets with
`python scripts/generate_robot_proxy_meshes.py`.

Run from this directory with the existing `habitat` environment:

```bash
conda run -n habitat python -m unittest discover -s tests -v
conda run -n habitat python scripts/debug_world_state.py --scene apt_1 --seed 123
conda run -n habitat python scripts/collect_level1.py --config configs/collector.json --num-states 10
conda run -n habitat python scripts/collect_level2.py --config configs/collector.json --root outputs/mri_dataset --type robot_translate
conda run -n habitat python scripts/validate_dataset.py --root outputs/mri_dataset
conda run -n habitat python scripts/make_contact_sheet.py --root outputs/mri_dataset --num-samples 10
```

For a smaller smoke test, replace `configs/collector.json` with
`configs/collector_pilot.json`; its default output is
`outputs/mri_dataset_pilot`.

`scripts/list_object_templates.py` reports installed handles; collection uses only the whitelist in `configs/collector.json`. Missing traditional ReplicaCAD semantic annotations do not block RGB-D, BEV, or state generation. The `instance.npy` channel stores Habitat rigid-object IDs, with the exact per-state entity mapping recorded in `state.json`.
