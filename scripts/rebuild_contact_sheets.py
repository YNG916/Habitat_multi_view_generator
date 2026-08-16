#!/usr/bin/env python3
"""Rebuild visual review sheets from published arrays/images without rerendering."""
import argparse,json
from pathlib import Path
from PIL import Image
import _bootstrap  # noqa:F401
from mri_dataset.visualization import contact_sheet,before_after_contact_sheet


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    args=parser.parse_args();root=Path(args.root).resolve();states=0;edits=0
    for state_json in sorted(root.glob("scenes/*/floors/*/regions/*/states/*/state.json")):
        state_dir=state_json.parent;metadata=json.loads(state_json.read_text())
        files=metadata["bev"]["files"]
        diagnostics=[
            Image.open(state_dir/files[key])
            for key in ("region_mask_visualization","occupancy_visualization","height_visualization")
            if key in files
        ]
        robots=[Image.open(state_dir/robot["files"]["rgb"]) for robot in metadata["robots"]]
        contact_sheet(
            Image.open(state_dir/files["annotated"]),robots,
            f"{metadata['state_id']} | {metadata['difficulty_overlap']['category']}",
            diagnostic_images=diagnostics,
        ).save(state_dir/"contact_sheet.png")
        states+=1
    for edit_path in sorted(root.glob("interventions/*/*/*/edit_*.json")):
        data=json.loads(edit_path.read_text())
        before_after_contact_sheet(
            root/data["before_state_path"],root/data["after_state_path"],data["instruction"]
        ).save(edit_path.with_name(f"{edit_path.stem}_contact_sheet.png"))
        edits+=1
    print({"state_contact_sheets":states,"intervention_contact_sheets":edits})

if __name__=="__main__":main()
