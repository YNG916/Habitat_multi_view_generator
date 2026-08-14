#!/usr/bin/env python3
import argparse

import _bootstrap  # noqa: F401
from mri_dataset.config import load_config
from mri_dataset.habitat_backend import HabitatBackend
from mri_dataset.level2 import collect_level2


def main():
    parser = argparse.ArgumentParser(description="Generate Level 2 pairs via shared WorldState edits")
    parser.add_argument("--config", default="configs/collector.json")
    parser.add_argument("--root")
    parser.add_argument("--scene", default="apt_1")
    parser.add_argument("--num-edits-per-state", type=int, default=1)
    parser.add_argument("--type", default="robot_translate", choices=["robot_translate", "robot_rotate", "object_translate", "object_place_relative", "object_remove"])
    args = parser.parse_args()
    config = load_config(args.config, output_root=args.root)
    with HabitatBackend(config, args.scene) as backend:
        paths = collect_level2(backend, config, config.output_path, args.num_edits_per_state, args.type)
    print(f"Generated {len(paths)} Level 2 edits under {config.output_path}")


if __name__ == "__main__":
    main()
