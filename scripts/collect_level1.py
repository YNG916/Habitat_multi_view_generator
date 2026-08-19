#!/usr/bin/env python3
import argparse
import _bootstrap  # noqa: F401
from mri_dataset.collector import collect_level1, initialize_dataset_root
from mri_dataset.config import load_config
from mri_dataset.habitat_backend import HabitatBackend
from mri_dataset.object_review import run_and_publish_approved_object_preflight


def main():
    parser = argparse.ArgumentParser(description="Collect HSSD Level-1 multi-robot states")
    parser.add_argument("--config", default="configs/collector_hssd_smoke.json")
    parser.add_argument("--num-states", type=int)
    parser.add_argument("--scene")
    parser.add_argument("--floor")
    parser.add_argument("--region")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config = load_config(args.config, output_root=args.output_root)
    initialize_dataset_root(config.output_path,config)
    run_and_publish_approved_object_preflight(
        config,config.output_path,raise_on_error=True
    )
    specs=config.collection_specs(args.scene,args.floor,args.region)
    for scene,floor,region,target in config.state_targets(specs,args.num_states):
        with HabitatBackend(config,scene,floor,region) as backend:
            paths = collect_level1(backend, config, config.output_path, target)
        print(f"{scene.scene_id}/{floor.floor_id}/{region.region_id}: {len(paths)} new Level-1 states")


if __name__ == "__main__":
    main()
