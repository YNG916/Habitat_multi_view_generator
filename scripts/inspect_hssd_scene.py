#!/usr/bin/env python3
import argparse
import json
import _bootstrap  # noqa: F401
from mri_dataset.config import load_config
from mri_dataset.habitat_backend import HabitatBackend


def main():
    parser = argparse.ArgumentParser(description="Inspect one registered HSSD scene/floor")
    parser.add_argument("--config", default="configs/collector_hssd_smoke.json")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--floor")
    parser.add_argument("--region")
    args = parser.parse_args()
    config = load_config(args.config)
    scene = config.registry().scene(args.scene)
    floors = [scene.floor(args.floor)] if args.floor else scene.eligible_floors
    report=[]
    for floor in floors:
        regions=[floor.region(args.region)] if args.region else floor.eligible_regions
        for region in regions:
            with HabitatBackend(config,scene,floor,region) as backend:
                report.append({"dataset_source":"hssd","scene_id":scene.scene_id,
                    "official_split":scene.official_split,"formal_split":config.scene_split(scene.scene_id),
                    "floor_id":floor.floor_id,"region_id":region.region_id,"region_category":region.region_category,
                    "floor_y":region.representative_floor_y,"allowed_island_ids":region.allowed_island_ids,
                    "navigable_area_m2":region.navigable_area_m2,"cached_navmesh":str(backend.navmesh_path),
                    "bev_camera_height_m":region.bev_camera_height_m,"bev":backend.mapping.metadata(),
                    "controlled_handles":backend.controlled_handles})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
