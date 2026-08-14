#!/usr/bin/env python3
import argparse
import json

import _bootstrap  # noqa: F401
from mri_dataset.config import load_config
from mri_dataset.habitat_backend import HabitatBackend


def main():
    parser = argparse.ArgumentParser(description="Inspect resolved scene/navmesh geometry")
    parser.add_argument("--config", default="configs/collector.json")
    parser.add_argument("--scene", default="apt_1")
    args = parser.parse_args()
    config = load_config(args.config)
    with HabitatBackend(config, args.scene) as backend:
        report = {
            "scene_id": args.scene, "dataset_config": str(config.dataset_config_path),
            "navmesh": str(backend.navmesh_path),
            "navmesh_bounds": [
                backend.navmesh_bounds[0].tolist(), backend.navmesh_bounds[1].tolist()
            ],
            "render_bev_bounds": [
                backend.render_bev_bounds[0].tolist(), backend.render_bev_bounds[1].tolist()
            ],
            "num_islands": backend.sim.pathfinder.num_islands,
            "bev": backend.mapping.metadata(), "controlled_handles": backend.controlled_handles,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
