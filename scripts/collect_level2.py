#!/usr/bin/env python3
import argparse
import _bootstrap  # noqa: F401
from mri_dataset.config import load_config
from mri_dataset.habitat_backend import HabitatBackend
from mri_dataset.object_review import run_approved_object_preflight
from mri_dataset.level2 import collect_level2


def main():
    parser = argparse.ArgumentParser(description="Collect HSSD Level-2 intervention pairs")
    parser.add_argument("--config", default="configs/collector_hssd_smoke.json")
    parser.add_argument("--root")
    parser.add_argument("--scene")
    parser.add_argument("--floor")
    parser.add_argument("--region")
    parser.add_argument("--num-edits-per-state", "--num-edits", dest="num_edits", type=int)
    parser.add_argument("--type", default="mixed", choices=[
        "mixed","robot_translate","robot_rotate","object_translate",
        "object_place_relative","object_remove",
    ])
    parser.add_argument("--regimes")
    args = parser.parse_args()
    config = load_config(args.config, output_root=args.root)
    run_approved_object_preflight(config,raise_on_error=True)
    regimes = args.regimes.split(",") if args.regimes else None
    total = 0
    for scene,floor,region in config.collection_specs(args.scene,args.floor,args.region):
        with HabitatBackend(config,scene,floor,region) as backend:
            paths = collect_level2(
                backend, config, config.output_path, args.num_edits,
                None if args.type == "mixed" else args.type, regimes,
            )
        total += len(paths)
        print(f"{scene.scene_id}/{floor.floor_id}/{region.region_id}: {len(paths)} new Level-2 edits")
    print(f"Total new edits: {total}")


if __name__ == "__main__":
    main()
