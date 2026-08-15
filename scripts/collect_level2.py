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
    parser.add_argument(
        "--scene", help="One scene; default collects every configured scene"
    )
    parser.add_argument(
        "--num-edits-per-state",
        "--num-edits",
        dest="num_edits_per_state",
        type=int,
        help="Valid edits per factual state and per requested regime",
    )
    parser.add_argument(
        "--type",
        default="mixed",
        choices=[
            "mixed",
            "robot_translate",
            "robot_rotate",
            "object_translate",
            "object_place_relative",
            "object_remove",
        ],
    )
    parser.add_argument(
        "--regimes",
        help="Comma-separated subset of the split protocol, e.g. id,ood",
    )
    args = parser.parse_args()
    config = load_config(args.config, output_root=args.root)
    scenes = [args.scene] if args.scene else config.scenes
    regimes = args.regimes.split(",") if args.regimes else None
    total = 0
    for scene in scenes:
        with HabitatBackend(config, scene) as backend:
            paths = collect_level2(
                backend,
                config,
                config.output_path,
                args.num_edits_per_state,
                None if args.type == "mixed" else args.type,
                regimes,
            )
        total += len(paths)
        print(f"Generated {len(paths)} new Level 2 edits for {scene}")
    print(f"Generated {total} new Level 2 edits under {config.output_path}")


if __name__ == "__main__":
    main()
