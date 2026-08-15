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


def utc_now(): return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description="Generate the HSSD-only MRI Dataset v2")
    parser.add_argument("--config", default="configs/collector_hssd.json")
    parser.add_argument("--output-root")
    parser.add_argument("--scene", action="append", dest="scenes")
    parser.add_argument("--floor", help="Use with a single --scene")
    parser.add_argument("--num-states", type=int)
    parser.add_argument("--num-edits-per-state", type=int)
    parser.add_argument("--stage", choices=["all","level1","level2","validate"], default="all")
    parser.add_argument("--regimes")
    parser.add_argument("--validation", choices=["full","metadata","none"], default="full")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_config(
        args.config, output_root=args.output_root,
        resume=False if args.no_resume else None,
    )
    if args.floor and (not args.scenes or len(args.scenes) != 1):
        raise ValueError("--floor requires exactly one --scene")
    selected = []
    if args.scenes:
        for scene_id in args.scenes:
            selected.extend(config.collection_specs(scene_id, args.floor))
    else:
        selected = config.collection_specs()
    regimes = args.regimes.split(",") if args.regimes else None
    root = config.output_path
    initialize_dataset_root(root, config)
    report = {
        "schema_version":"2.0.0","dataset_source":"hssd",
        "protocol_version":config.protocol_version,
        "generation_fingerprint":config.generation_fingerprint(),
        "started_at_utc":utc_now(),
        "selected_scene_floors":[f"{s.scene_id}/{f.floor_id}" for s,f in selected],
        "stage":args.stage,"new_level1_states":{},"new_level2_edits":{},
    }
    write_json(root/"generation_report.json",report)
    if config.run_multilevel_calibration_preflight:
        calibration=validate_multilevel_orthographic_depth(config)
        write_json(root/"calibration_report.json",calibration)
        report["orthographic_calibration"]={
            "passed":bool(calibration["passed"]),
            "max_abs_error_m":calibration["max_abs_error_m"],
            "dataset_independent":True,
        }
        write_json(root/"generation_report.json",report)
        if not calibration["passed"]: raise RuntimeError("Orthographic calibration failed")
    run_l1=args.stage in {"all","level1"}
    run_l2=args.stage in {"all","level2"}
    if run_l1 or run_l2:
        for scene,floor in selected:
            key=f"{scene.scene_id}/{floor.floor_id}"
            with HabitatBackend(config,scene,floor) as backend:
                if run_l1:
                    target=args.num_states if args.num_states is not None else config.states_for_scene(scene.scene_id)
                    paths=collect_level1(backend,config,root,target)
                    report["new_level1_states"][key]=len(paths)
                    write_json(root/"generation_report.json",report)
                if run_l2:
                    paths=collect_level2(
                        backend,config,root,args.num_edits_per_state,regimes=regimes
                    )
                    report["new_level2_edits"][key]=len(paths)
                    write_json(root/"generation_report.json",report)
    update_dataset_index(root)
    mode=args.validation
    if args.stage=="validate" and mode=="none": mode="full"
    if mode!="none":
        validation=validate_dataset(root,config if mode=="full" else None)
        write_json(root/"validation_report.json",validation)
        report["validation"]={
            "mode":mode,"passed":bool(validation["passed"]),
            "states_checked":validation["states_checked"],
            "error_groups":len(validation["errors"]),
        }
    report["finished_at_utc"]=utc_now()
    report["complete"]=bool(mode=="none" or report["validation"]["passed"])
    write_json(root/"generation_report.json",report)
    print(f"Dataset root: {root}")
    print(f"Generation complete: {report['complete']}")
    if not report["complete"]: raise SystemExit(2)


if __name__=="__main__": main()
