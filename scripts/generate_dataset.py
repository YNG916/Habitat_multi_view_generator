#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
from mri_dataset.calibration import validate_multilevel_orthographic_depth
from mri_dataset.collector import collect_level1, initialize_dataset_root, update_dataset_index
from mri_dataset.config import load_config
from mri_dataset.habitat_backend import HabitatBackend
from mri_dataset.level2 import collect_level2
from mri_dataset.serialization import write_json
from mri_dataset.validation import validate_dataset


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the formal multi-scene MRI Dataset v1"
    )
    parser.add_argument("--config", default="configs/collector.json")
    parser.add_argument("--output-root")
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help="Collect only this configured scene; may be repeated",
    )
    parser.add_argument("--num-states", type=int, help="Override target per selected scene")
    parser.add_argument("--num-edits-per-state", type=int)
    parser.add_argument(
        "--stage",
        choices=["all", "level1", "level2", "validate"],
        default="all",
    )
    parser.add_argument(
        "--regimes", help="Comma-separated protocol subset, such as id or id,ood"
    )
    parser.add_argument(
        "--validation",
        choices=["full", "metadata", "none"],
        default="full",
        help="Final validation strength; full replays Bullet checks",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(
        args.config,
        output_root=args.output_root,
        resume=False if args.no_resume else None,
    )
    scenes = args.scenes or config.scenes
    unknown = sorted(set(scenes) - set(config.scenes))
    if unknown:
        raise ValueError(f"Scenes are not present in the frozen config: {unknown}")
    regimes = args.regimes.split(",") if args.regimes else None
    root = config.output_path
    initialize_dataset_root(root, config)
    report = {
        "schema_version": "1.0.0",
        "protocol_version": config.protocol_version,
        "generation_fingerprint": config.generation_fingerprint(),
        "started_at_utc": utc_now(),
        "selected_scenes": scenes,
        "stage": args.stage,
        "new_level1_states": {},
        "new_level2_edits": {},
    }
    write_json(root / "generation_report.json", report)
    if config.run_multilevel_calibration_preflight:
        calibration = validate_multilevel_orthographic_depth(config)
        write_json(root / "calibration_report.json", calibration)
        report["multilevel_orthographic_calibration"] = {
            "passed": bool(calibration["passed"]),
            "max_abs_error_m": calibration["max_abs_error_m"],
            "heights_m": calibration["heights_m"],
        }
        write_json(root / "generation_report.json", report)
        if not calibration["passed"]:
            raise RuntimeError("Multi-height orthographic calibration failed")

    run_level1 = args.stage in {"all", "level1"}
    run_level2 = args.stage in {"all", "level2"}
    if run_level1 or run_level2:
        for scene in scenes:
            with HabitatBackend(config, scene) as backend:
                if run_level1:
                    target = (
                        args.num_states
                        if args.num_states is not None
                        else config.states_for_scene(scene)
                    )
                    level1_paths = collect_level1(
                        backend, config, root, target
                    )
                    report["new_level1_states"][scene] = len(level1_paths)
                    write_json(root / "generation_report.json", report)
                if run_level2:
                    level2_paths = collect_level2(
                        backend,
                        config,
                        root,
                        args.num_edits_per_state,
                        regimes=regimes,
                    )
                    report["new_level2_edits"][scene] = len(level2_paths)
                    write_json(root / "generation_report.json", report)

    update_dataset_index(root)
    validation_mode = args.validation
    if args.stage == "validate" and validation_mode == "none":
        validation_mode = "full"
    if validation_mode != "none":
        validation = validate_dataset(
            root, config if validation_mode == "full" else None
        )
        write_json(root / "validation_report.json", validation)
        report["validation"] = {
            "mode": validation_mode,
            "passed": bool(validation["passed"]),
            "states_checked": validation["states_checked"],
            "error_groups": len(validation["errors"]),
        }
    report["finished_at_utc"] = utc_now()
    report["complete"] = bool(
        validation_mode == "none" or report["validation"]["passed"]
    )
    write_json(root / "generation_report.json", report)
    print(f"Dataset root: {root}")
    print(f"Generation complete: {report['complete']}")
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
