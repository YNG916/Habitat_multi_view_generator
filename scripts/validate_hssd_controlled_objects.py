#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from mri_dataset.config import load_config
from mri_dataset.object_review import run_approved_object_preflight


def main():
    parser=argparse.ArgumentParser(
        description="Validate, instantiate, support, and render every approved HSSD object"
    )
    parser.add_argument("--config",default="configs/collector_hssd.json")
    parser.add_argument(
        "--report",default="outputs/hssd_object_preflight.json"
    )
    parser.add_argument(
        "--contact-sheet-root",
        default="outputs/hssd_object_review",
    )
    args=parser.parse_args()
    config=load_config(args.config)
    report=run_approved_object_preflight(
        config,
        report_path=args.report,
        contact_sheet_root=args.contact_sheet_root,
    )
    print(json.dumps({
        "passed":report["passed"],
        "asset_count":report.get("asset_count",0),
        "category_counts":report.get("category_counts",{}),
        "errors":report["errors"],
        "report":args.report,
        "contact_sheet_root":args.contact_sheet_root,
    },indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
