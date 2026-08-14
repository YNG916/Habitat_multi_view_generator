#!/usr/bin/env python3
import argparse

import _bootstrap  # noqa: F401
from mri_dataset.collector import collect_level1
from mri_dataset.config import load_config
from mri_dataset.habitat_backend import HabitatBackend


def main():
    parser = argparse.ArgumentParser(description="Collect Level 1 multi-robot RGB-D world states")
    parser.add_argument("--config", default="configs/collector.json")
    parser.add_argument("--num-states", type=int)
    parser.add_argument("--scene")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config = load_config(args.config, num_states=args.num_states, output_root=args.output_root)
    scenes = [args.scene] if args.scene else config.scenes
    for scene in scenes:
        with HabitatBackend(config, scene) as backend:
            paths = collect_level1(backend, config, config.output_path, config.num_states)
        print(f"Generated {len(paths)} Level 1 states for {scene} under {config.output_path}")


if __name__ == "__main__":
    main()
