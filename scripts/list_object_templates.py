#!/usr/bin/env python3
import argparse
import json

import _bootstrap  # noqa: F401
from mri_dataset.config import load_config
from mri_dataset.habitat_backend import HabitatBackend
from mri_dataset.objects import list_template_records


def main():
    parser = argparse.ArgumentParser(description="List actual installed ReplicaCAD rigid templates")
    parser.add_argument("--config", default="configs/collector.json")
    parser.add_argument("--scene", default="apt_1")
    parser.add_argument("--contains", default="")
    args = parser.parse_args()
    config = load_config(args.config)
    with HabitatBackend(config, args.scene) as backend:
        records = list_template_records(backend.sim.get_object_template_manager())
    if args.contains:
        records = [record for record in records if args.contains.lower() in record["handle"].lower()]
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
