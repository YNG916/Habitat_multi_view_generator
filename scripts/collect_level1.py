#!/usr/bin/env python3
import argparse
import _bootstrap  # noqa: F401
from mri_dataset.collector import collect_level1
from mri_dataset.config import load_config
from mri_dataset.habitat_backend import HabitatBackend


def main():
    parser = argparse.ArgumentParser(description="Collect HSSD Level-1 multi-robot states")
    parser.add_argument("--config", default="configs/collector_hssd_smoke.json")
    parser.add_argument("--num-states", type=int)
    parser.add_argument("--scene")
    parser.add_argument("--floor")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config = load_config(args.config, output_root=args.output_root)
    for scene, floor in config.collection_specs(args.scene, args.floor):
        with HabitatBackend(config, scene, floor) as backend:
            target = args.num_states if args.num_states is not None else config.states_for_scene(scene.scene_id)
            paths = collect_level1(backend, config, config.output_path, target)
        print(f"{scene.scene_id}/{floor.floor_id}: {len(paths)} new Level-1 states")


if __name__ == "__main__":
    main()
