#!/usr/bin/env python3
import argparse
import _bootstrap  # noqa: F401
from mri_dataset.collector import collect_level1
from mri_dataset.config import load_config
from mri_dataset.habitat_backend import HabitatBackend


def main():
    parser = argparse.ArgumentParser(description="Generate one reproducible HSSD debug state")
    parser.add_argument("--config", default="configs/collector_hssd_smoke.json")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--floor", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-root", default="outputs/hssd_debug")
    args = parser.parse_args()
    config = load_config(args.config, random_seed=args.seed, output_root=args.output_root)
    scene = config.registry().scene(args.scene)
    floor = scene.floor(args.floor)
    with HabitatBackend(config, scene, floor) as backend:
        outputs = collect_level1(backend, config, config.output_path, 1)
    print(f"Generated reproducible HSSD state: {outputs[0]}")


if __name__ == "__main__":
    main()
