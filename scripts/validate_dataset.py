#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from mri_dataset.config import CollectorConfig, load_config
from mri_dataset.validation import validate_dataset


def main():
    parser = argparse.ArgumentParser(description="Validate saved dataset geometry and files")
    parser.add_argument("--root", required=True)
    parser.add_argument("--config")
    parser.add_argument("--no-simulator", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = None
    if not args.no_simulator:
        if args.config:
            config = load_config(args.config, output_root=str(root))
        else:
            with (root / "dataset.json").open("r", encoding="utf-8") as handle:
                data = json.load(handle)["config"]
            data["output_root"] = str(root)
            config = CollectorConfig(**data)
    report = validate_dataset(root, config)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
