#!/usr/bin/env python3
"""Write reproducible registry, visibility, and storage statistics."""
from __future__ import annotations
import argparse,json,math
from collections import Counter
from pathlib import Path
import numpy as np
import _bootstrap  # noqa:F401
from mri_dataset.config import load_config
from mri_dataset.serialization import write_json


def histogram(values,bins):
    counts,_=np.histogram(values,bins=bins)
    return {f"[{bins[i]},{bins[i+1]})":int(counts[i]) for i in range(len(counts))}

def quantiles(values):
    return {key:float(value) for key,value in zip(("min","p25","median","p75","max"),np.percentile(values,[0,25,50,75,100]))}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/collector_hssd.json")
    parser.add_argument("--pilot-root",default="outputs/mri_hssd_region_smoke")
    parser.add_argument("--old-registry",default="data/hssd_processed/scene_registry.json")
    parser.add_argument("--candidates",default="data/hssd_processed/object_candidates.json")
    parser.add_argument("--output",default="outputs/hssd_region_protocol_report.json")
    args=parser.parse_args();config=load_config(args.config)
    registry=config.registry();eligible_scenes=[s for s in registry.scenes if s.eligible]
    floors=[f for s in registry.scenes for f in s.floors];regions=[r for f in floors for r in f.regions]
    eligible=[r for r in regions if r.eligible]
    areas=[r.navigable_area_m2 for r in eligible]
    metric=[]
    for r in eligible:
        low,high=r.visual_bev_bounds_world;x=high[0]-low[0];z=high[2]-low[2];metric.append((x,z))
    resolution={}
    for label,mpp in (("formal",config.bev_meters_per_pixel),("practical_1024",.0125)):
        sizes=[]
        for x,z in metric:
            width=max(2,int(math.ceil(x/mpp))+1);height=max(2,int(round(width*z/x)));sizes.append((width,height))
        resolution[label]={"meters_per_pixel":mpp,"width_pixels":quantiles([x[0] for x in sizes]),"height_pixels":quantiles([x[1] for x in sizes])}
    report={
        "registry":str(config.scene_registry_path),"statistics":registry.statistics,
        "eligible_regions_per_scene":quantiles([len(s.eligible_regions) for s in eligible_scenes]),
        "region_category_histogram":dict(sorted(Counter(r.region_category for r in eligible).items())),
        "region_navigable_area_m2":quantiles(areas),
        "region_area_histogram_m2":histogram(areas,[0,3.5,5,10,20,50,100,200,500,float("inf")]),
        "bev_metric_width_m":quantiles([x for x,_ in metric]),
        "bev_metric_height_m":quantiles([z for _,z in metric]),
        "bev_pixel_statistics":resolution,
    }
    old_path=Path(args.old_registry)
    if old_path.is_file():
        old=json.loads(old_path.read_text());ambiguous={
            scene["scene_id"] for scene in old.get("scenes",[])
            if any("ambiguous_overlapping_multifloor_geometry" in f.get("rejection_reasons",[]) for f in scene.get("floors",[]))
        }
        current={s.scene_id for s in eligible_scenes}
        report["previous_ambiguous_multifloor"]={"scenes":len(ambiguous),"recovered_as_region_eligible":len(ambiguous&current),"recovered_scene_ids":sorted(ambiguous&current)}
    approved=json.loads(config.controlled_object_registry_path.read_text()).get("approved_assets",{})
    report["approved_objects"]={"categories":len(approved),"assets_total":sum(map(len,approved.values())),"assets_per_category":{key:len(value) for key,value in approved.items()}}
    candidate_path=Path(args.candidates)
    if candidate_path.is_file():
        candidates=json.loads(candidate_path.read_text())
        preview_pixels=[
            item["preview_object_id_pixels"]
            for values in candidates.get("candidates",{}).values()
            for item in values if "preview_object_id_pixels" in item
        ]
        report["object_candidates"]={
            "counts":{key:len(value) for key,value in candidates.get("candidates",{}).items()},
            "decomposed_excluded":candidates.get("decomposed_records_excluded"),
            "preview_object_id_pixels":quantiles(preview_pixels) if preview_pixels else None,
            "contact_sheets":candidates.get("contact_sheets",{}),
        }
    pilot=Path(args.pilot_root)
    if pilot.is_dir():
        states=sorted(pilot.glob("scenes/*/floors/*/regions/*/states/*/state.json"));pixels=[];fractions=[];views=[]
        for state_path in states:
            data=json.loads(state_path.read_text());objects=json.loads((state_path.parent/data["objects_path"]).read_text())
            for entity in [*data["robots"],*objects]:
                visible=0
                for item in entity.get("visibility",{}).values():
                    pixels.append(item.get("visible_pixel_count",0));fractions.append(item.get("visible_image_fraction",0.));visible+=bool(item.get("benchmark_visible"))
                views.append(visible)
        edits=[json.loads(path.read_text()) for path in pilot.glob("interventions/*/*/*/edit_*.json")]
        bytes_total=sum(path.stat().st_size for path in pilot.rglob("*") if path.is_file())
        state_sizes=[sum(path.stat().st_size for path in state.parent.rglob("*") if path.is_file()) for state in states]
        report["pilot"]={
            "root":str(pilot),"states":len(states),"factual_states":sum(not json.loads(p.read_text()).get("parent_state_id") for p in states),
            "interventions":len(edits),"validation":json.loads((pilot/"validation_report.json").read_text()) if (pilot/"validation_report.json").is_file() else None,
            "visibility_pixels":quantiles(pixels),"visibility_image_fraction":quantiles(fractions),
            "benchmark_visible_views_histogram":dict(sorted(Counter(map(int,views)).items())),
            "observable_edits_valid":sum(
                e["observable_edit"]["before_visible_views"]>0
                and (
                    e["observable_edit"]["after_visible_views"]==0
                    if e["structured_intervention"]["type"]=="object_remove"
                    else e["observable_edit"]["after_visible_views"]>0
                )
                for e in edits
            ),
            "identical_bev_frames":sum(bool(e.get("bev_frame_identical")) for e in edits),
            "bytes_total":bytes_total,"state_bytes":quantiles(state_sizes),
        }
        targets=config.state_targets(config.collection_specs());factual=sum(item[3] for item in targets)
        derived=sum(item[3]*len(config.level2_regimes_by_split[config.scene_split(item[0].scene_id)])*config.num_edits_per_state for item in targets)
        pilot_config=json.loads((pilot/"dataset.json").read_text())["config"]
        grouped=[]
        for state in states:
            directory=state.parent
            robot_bytes=sum(path.stat().st_size for path in (directory/"robots").rglob("*") if path.is_file())
            bev_bytes=sum(path.stat().st_size for path in (directory/"bev").rglob("*") if path.is_file())
            fixed_bytes=sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
            grouped.append((robot_bytes,bev_bytes,fixed_bytes))
        grouped=np.asarray(grouped,dtype=np.float64)
        total_states=factual+derived
        def scaled_projection(width,mpp):
            robot_factor=(float(width)/float(pilot_config["width"]))**2
            bev_factor=(float(pilot_config["bev_meters_per_pixel"])/float(mpp))**2
            estimates=grouped[:,0]*robot_factor+grouped[:,1]*bev_factor+grouped[:,2]
            return {"robot_width_height":int(width),"bev_meters_per_pixel":float(mpp),
                "mean_bytes_per_state":int(estimates.mean()),
                "estimated_state_storage_bytes":int(total_states*estimates.mean())}
        report["projection"]={
            "planned_factual_states":factual,"planned_after_states":derived,
            "formal_2048":scaled_projection(config.width,config.bev_meters_per_pixel),
            "practical_1024":scaled_projection(1024,.0125),
            "estimate_note":"Pilot files are split into robot rasters, BEV rasters, and fixed metadata/contact sheets; raster groups are scaled by pixel area. Excludes preprocessing previews and small index overhead.",
        }
    write_json(Path(args.output),report);print(json.dumps(report,indent=2))

if __name__=="__main__":main()
