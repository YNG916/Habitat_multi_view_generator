#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import _bootstrap  # noqa: F401
from mri_dataset.config import load_config
from mri_dataset.hssd_preprocess import discover_installed_hssd_scenes
from mri_dataset.serialization import write_json


def main():
    parser=argparse.ArgumentParser(
        description="Select exact real HSSD rigid-template handles for controlled objects"
    )
    parser.add_argument("--config",default="configs/collector_hssd.json")
    parser.add_argument("--output",default="configs/hssd_controlled_objects.json")
    parser.add_argument("--categories",default="cup,bowl,book,bottle,box")
    args=parser.parse_args()
    config=load_config(args.config,require_preprocessed_registry=False)
    categories=[value.strip() for value in args.categories.split(",") if value.strip()]
    lexicon_path=config.dataset_config_path.parent/"semantics/hssd-hab_semantic_lexicon.json"
    lexicon=json.loads(lexicon_path.read_text(encoding="utf-8"))
    name_to_id={item["name"]:int(item["id"]) for item in lexicon["classes"]}
    missing=sorted(set(categories)-set(name_to_id))
    if missing: raise ValueError(f"Categories absent from HSSD lexicon: {missing}")
    scenes=discover_installed_hssd_scenes(config.dataset_config_path)
    if not scenes: raise RuntimeError("No standard HSSD scenes installed")
    import habitat_sim
    sim_cfg=habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file=str(config.dataset_config_path)
    sim_cfg.scene_id=scenes[0]
    sim_cfg.enable_physics=True
    sim_cfg.gpu_device_id=int(config.gpu_device_id)
    agent=habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications=[]
    sim=habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg,[agent]))
    try:
        template_manager=sim.get_object_template_manager()
        rigid_manager=sim.get_rigid_object_manager()
        target_ids={name_to_id[name]:name for name in categories}
        records={name:[] for name in categories}
        for handle in sorted(template_manager.get_template_handles()):
            attributes=template_manager.get_template_by_handle(handle)
            semantic_id=int(attributes.semantic_id)
            if semantic_id not in target_ids: continue
            rigid=rigid_manager.add_object_by_template_handle(handle)
            if rigid is None: continue
            try:
                bounds=rigid.root_scene_node.cumulative_bb
                low=np.asarray(bounds.min,dtype=np.float64)
                high=np.asarray(bounds.max,dtype=np.float64)
                extent=high-low
                horizontal=max(float(extent[0]),float(extent[2]))
                height=float(extent[1])
                if not np.all(np.isfinite(extent)): continue
                if not (.025<=horizontal<=.50 and .01<=height<=.50): continue
                score=abs(horizontal-.14)+.5*abs(height-.12)
                records[target_ids[semantic_id]].append({
                    "handle":handle,"semantic_id":semantic_id,
                    "extent_xyz_m":extent.tolist(),"selection_score":score,
                })
            finally:
                rigid_manager.remove_object_by_id(rigid.object_id)
        selected={}
        for category in categories:
            records[category].sort(key=lambda item:(item["selection_score"],item["handle"]))
            if not records[category]:
                raise RuntimeError(f"No physically suitable real HSSD template for {category}")
            selected[category]=records[category][0]["handle"]
            records[category]=records[category][:20]
        write_json(Path(args.output),{
            "schema_version":"1.0.0","dataset_source":"hssd",
            "source_dataset_config":config.scene_dataset_config,
            "selection_policy":{
                "real_hssd_templates_only":True,
                "horizontal_extent_m":[.025,.50],"height_m":[.01,.50],
                "target_horizontal_extent_m":.14,"target_height_m":.12,
            },
            "selected_handles":selected,"candidates":records,
        })
        print(json.dumps(selected,indent=2))
    finally:
        sim.close()


if __name__=="__main__": main()
