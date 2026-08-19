#!/usr/bin/env python3
"""Inventory and render whole-object HSSD candidates; never auto-approve."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import _bootstrap  # noqa:F401
from mri_dataset.config import load_config
from mri_dataset.hssd_preprocess import discover_installed_hssd_scenes
from mri_dataset.objects import is_decomposed_canonical_id
from mri_dataset.serialization import write_json



from mri_dataset.object_review import (
    find_review_anchor as _preview_anchor,
    object_contact_sheet as _contact_sheet,
    render_supported_object as _render_candidate,
    review_sensor_specs as _sensors,
    set_review_lighting,
)

def main():
    parser=argparse.ArgumentParser(description="Build review-only whole-object HSSD candidate pools")
    parser.add_argument("--config",default="configs/collector_hssd.json")
    parser.add_argument("--output",default="data/hssd_processed/object_candidates.json")
    parser.add_argument("--contact-sheet-root",default="outputs/hssd_object_candidates")
    parser.add_argument("--categories",default="bag,basket,book,bottle,bowl,box,can,cup,shoe,toy")
    parser.add_argument("--limit-per-category",type=int,default=20)
    parser.add_argument("--include-decomposed",action="store_true")
    parser.add_argument("--no-contact-sheets",action="store_true")
    args=parser.parse_args();config=load_config(args.config,require_preprocessed_registry=False)
    categories=[value.strip() for value in args.categories.split(",") if value.strip()]
    root=config.dataset_config_path.resolve().parent
    lexicon=json.loads((root/"semantics/hssd-hab_semantic_lexicon.json").read_text())
    name_to_id={item["name"]:int(item["id"]) for item in lexicon["classes"]}
    missing=sorted(set(categories)-set(name_to_id))
    if missing:raise ValueError(f"Categories absent from real HSSD lexicon: {missing}")
    scenes=discover_installed_hssd_scenes(config.dataset_config_path)
    if not scenes:raise RuntimeError("No installed HSSD scene")
    import habitat_sim
    cfg=habitat_sim.SimulatorConfiguration();cfg.scene_dataset_config_file=str(config.dataset_config_path)
    cfg.scene_id=scenes[0];cfg.enable_physics=True;cfg.gpu_device_id=int(config.gpu_device_id)
    cfg.override_scene_light_defaults=True
    cfg.scene_light_setup=habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    agent=habitat_sim.agent.AgentConfiguration();agent.sensor_specifications=_sensors(habitat_sim)
    sim=habitat_sim.Simulator(habitat_sim.Configuration(cfg,[agent]))
    set_review_lighting(sim,habitat_sim)
    try:
        templates=sim.get_object_template_manager();rigid=sim.get_rigid_object_manager()
        targets={name_to_id[name]:name for name in categories}
        records={name:[] for name in categories};excluded=0
        for handle in sorted(templates.get_template_handles()):
            path=Path(handle).resolve()
            try:canonical=path.relative_to(root).as_posix()
            except ValueError:continue
            if is_decomposed_canonical_id(canonical) and not args.include_decomposed:
                excluded+=1;continue
            attributes=templates.get_template_by_handle(handle);semantic_id=int(attributes.semantic_id)
            if semantic_id not in targets:continue
            obj=rigid.add_object_by_template_handle(handle)
            if obj is None:continue
            try:
                bounds=obj.root_scene_node.cumulative_bb
                extent=np.asarray(bounds.max,dtype=np.float64)-np.asarray(bounds.min,dtype=np.float64)
                horizontal=max(float(extent[0]),float(extent[2]));height=float(extent[1])
                if not np.all(np.isfinite(extent)) or not (.025<=horizontal<=.60 and .01<=height<=.60):continue
                collision=str(attributes.collision_asset_handle);collision_path=Path(collision)
                collider=bool(collision) and (collision=="NONE" or collision_path.is_file() or (root/"objects"/collision_path).is_file())
                score=abs(horizontal-.18)+.5*abs(height-.16)+(0 if collider else 10)
                records[targets[semantic_id]].append({
                    "canonical_id":canonical,"runtime_handle":handle,"source_path":canonical,
                    "semantic_id":semantic_id,"extent_xyz_m":[float(v) for v in extent],
                    "collider_available":collider,"render_asset":str(attributes.render_asset_handle),
                    "collision_asset":collision,"physical_score":score,
                    "asset_fingerprint":hashlib.sha256(path.read_bytes()).hexdigest(),
                })
            finally:rigid.remove_object_by_id(obj.object_id)
        for values in records.values():values.sort(key=lambda item:(item["physical_score"],item["canonical_id"]))
        records={key:value[:args.limit_per_category] for key,value in records.items()}
        anchor=_preview_anchor(sim,scenes[0],config)
        sheets={}
        if not args.no_contact_sheets:
            sheet_root=Path(args.contact_sheet_root)
            for category,values in records.items():
                if not values:continue
                images=[]
                for item in values:
                    image,pixels,_geometry=_render_candidate(
                        sim,item["runtime_handle"],item["extent_xyz_m"],anchor,habitat_sim
                    )
                    item["preview_object_id_pixels"]=pixels
                    images.append(image)
                path=sheet_root/f"{category}.png";_contact_sheet(category,values,images,path)
                sheets[category]=str(path)
        write_json(Path(args.output),{
            "schema_version":"2.0.0","dataset_source":"hssd",
            "approval_status":"candidates_only",
            "decomposed_excluded_by_default":not args.include_decomposed,
            "decomposed_records_excluded":excluded,
            "selection_policy":{"whole_object_only":not args.include_decomposed,
                "horizontal_extent_m":[.025,.60],"height_m":[.01,.60],
                "candidate_limit_per_category":args.limit_per_category},
            "contact_sheets":sheets,"candidates":records,
        })
        print(json.dumps({key:len(value) for key,value in records.items()},indent=2))
    finally:sim.close()

if __name__=="__main__":main()
