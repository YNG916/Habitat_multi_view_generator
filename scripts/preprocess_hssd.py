#!/usr/bin/env python3
import argparse
from pathlib import Path
import _bootstrap  # noqa: F401
from mri_dataset.config import load_config
from mri_dataset.hssd_preprocess import preprocess_hssd


def main():
    parser = argparse.ArgumentParser(
        description="Precompute robot-specific NavMeshes and scene/floor registry for standard HSSD"
    )
    parser.add_argument("--config", default="configs/collector_hssd.json")
    parser.add_argument("--scene", action="append", dest="scenes")
    parser.add_argument("--scene-file")
    parser.add_argument("--registry-path")
    parser.add_argument("--split-manifest-path")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--preview-root", default="outputs/hssd_eligible_region_previews"
    )
    args = parser.parse_args()
    config = load_config(
        args.config,
        require_preprocessed_registry=False,
        scene_registry=args.registry_path,
        split_manifest=args.split_manifest_path,
    )
    scenes = args.scenes
    if args.scene_file:
        file_scenes = [
            line.strip()
            for line in Path(args.scene_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        scenes = [*(scenes or []), *file_scenes]
    registry = preprocess_hssd(
        config, scene_ids=scenes, limit=args.limit,
        preview_root=Path(args.preview_root),
    )
    print(registry.statistics)


if __name__ == "__main__":
    main()
