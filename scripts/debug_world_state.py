#!/usr/bin/env python3
import argparse

import _bootstrap  # noqa: F401
from mri_dataset.collector import collect_level1
from mri_dataset.config import load_config
from mri_dataset.habitat_backend import HabitatBackend


def main():
    parser = argparse.ArgumentParser(description="Generate the preserved deterministic apt_1 debug state")
    parser.add_argument("--config", default="configs/collector.json")
    parser.add_argument("--scene", default="apt_1")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-root", default="outputs/debug_dataset")
    args = parser.parse_args()
    config = load_config(args.config, scenes=[args.scene], random_seed=args.seed, output_root=args.output_root)
    with HabitatBackend(config, args.scene) as backend:
        outputs = collect_level1(backend, config, config.output_path, 1, deterministic_debug=True)
    print(f"Generated deterministic state: {outputs[0]}")


if __name__ == "__main__":
    main()
