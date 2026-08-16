#!/usr/bin/env python3
"""Inventory and render whole-object HSSD candidates; never auto-approve."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import _bootstrap  # noqa:F401
from mri_dataset.config import load_config
from mri_dataset.hssd_preprocess import discover_installed_hssd_scenes
from mri_dataset.objects import is_decomposed_canonical_id
from mri_dataset.serialization import write_json


def _sensors(habitat_sim):
    specs=[]
    for uuid,sensor_type in (
        ("candidate_rgb",habitat_sim.SensorType.COLOR),
        ("candidate_instance",habitat_sim.SensorType.SEMANTIC),
    ):
        spec=habitat_sim.CameraSensorSpec();spec.uuid=uuid
        spec.sensor_type=sensor_type
        spec.sensor_subtype=habitat_sim.SensorSubType.PINHOLE
        spec.resolution=[384,384];spec.hfov=55.;spec.near=.02;spec.far=10.
        spec.position=[0.,0.,0.];spec.orientation=[0.,0.,0.]
        if uuid=="candidate_instance":
            spec.semantic_target=type(spec.semantic_target).OBJECT_ID
        specs.append(spec)
    return specs


def _preview_anchor(sim,scene_id,config):
    navmesh=config.navmesh_cache_path(scene_id)
    if navmesh.is_file() and not sim.pathfinder.load_nav_mesh(str(navmesh)):
        raise RuntimeError(f"Could not load preview NavMesh: {navmesh}")
    best=None;best_clearance=-1.
    for _ in range(2000):
        point=np.asarray(sim.pathfinder.get_random_navigable_point(),dtype=np.float64)
        if not np.all(np.isfinite(point)):continue
        clearance=float(sim.pathfinder.distance_to_closest_obstacle(point,3.))
        if clearance>best_clearance:best,best_clearance=point,clearance
        if clearance>=1.6:return point
    if best is None:raise RuntimeError("No preview NavMesh point")
    return best


def _render_candidate(sim,handle,extent,anchor,habitat_sim):
    manager=sim.get_rigid_object_manager();obj=manager.add_object_by_template_handle(handle)
    if obj is None:raise RuntimeError(f"Cannot render {handle}")
    try:
        obj.motion_type=habitat_sim.physics.MotionType.KINEMATIC
        obj.set_light_setup("candidate_lights")
        collision=obj.collision_shape_aabb
        position=np.asarray(anchor,dtype=np.float64).copy()
        position[1]=float(anchor[1])-float(collision.min[1])
        obj.translation=position.astype(np.float32)
        visual=obj.root_scene_node.cumulative_bb
        center_y=float(position[1]+(float(visual.min[1])+float(visual.max[1]))*.5)
        horizontal=max(float(extent[0]),float(extent[2]))
        distance=max(.45,horizontal*2.2,float(extent[1])*2.0)
        views=(
            ((0.,distance),0.),((0.,-distance),math.pi),
            ((distance,0.),math.pi/2),((-distance,0.),-math.pi/2),
        )
        best_image=None;best_pixels=-1
        for (dx,dz),yaw in views:
            camera=habitat_sim.AgentState()
            camera.position=np.asarray([anchor[0]+dx,center_y,anchor[2]+dz],np.float32)
            camera.rotation=habitat_sim.utils.common.quat_from_angle_axis(yaw,np.array([0.,1.,0.]))
            sim.get_agent(0).set_state(camera,infer_sensor_states=True)
            observations=sim.get_sensor_observations(agent_ids=[0])[0]
            instance=np.asarray(observations["candidate_instance"])
            pixels=int((instance==obj.object_id).sum())
            if pixels>best_pixels:
                best_pixels=pixels
                rgb=np.asarray(observations["candidate_rgb"])[...,:3].astype(np.uint8)
                locations=np.argwhere(instance==obj.object_id)
                if len(locations):
                    low=locations.min(axis=0);high=locations.max(axis=0)+1
                    pad=max(8,int(.15*max(*(high-low))))
                    r0=max(0,int(low[0])-pad);r1=min(rgb.shape[0],int(high[0])+pad)
                    c0=max(0,int(low[1])-pad);c1=min(rgb.shape[1],int(high[1])+pad)
                    best_image=Image.fromarray(rgb[r0:r1,c0:c1]).resize((384,384))
        if best_pixels<=0:raise RuntimeError(f"Candidate has zero OBJECT_ID pixels: {handle}")
        return best_image,best_pixels
    finally:manager.remove_object_by_id(obj.object_id)


def _contact_sheet(category,records,images,path):
    cell_w,cell_h,columns=384,430,4
    rows=max(1,math.ceil(len(images)/columns))
    sheet=Image.new("RGB",(columns*cell_w,rows*cell_h),(20,20,20));draw=ImageDraw.Draw(sheet);font=ImageFont.load_default()
    for index,(record,image) in enumerate(zip(records,images)):
        x=(index%columns)*cell_w;y=(index//columns)*cell_h
        sheet.paste(image.resize((cell_w,384)),(x,y))
        label=f"{index:02d} {category}  {record['extent_xyz_m'][0]:.2f}x{record['extent_xyz_m'][1]:.2f}x{record['extent_xyz_m'][2]:.2f}m"
        draw.text((x+5,y+389),label,fill="white",font=font)
        draw.text((x+5,y+405),Path(record["canonical_id"]).stem[:48],fill=(190,220,255),font=font)
    path.parent.mkdir(parents=True,exist_ok=True);sheet.save(path)


def main():
    parser=argparse.ArgumentParser(description="Build review-only whole-object HSSD candidate pools")
    parser.add_argument("--config",default="configs/collector_hssd.json")
    parser.add_argument("--output",default="data/hssd_processed/object_candidates.json")
    parser.add_argument("--contact-sheet-root",default="outputs/hssd_controlled_object_candidate_contact_sheets")
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
    import magnum as mn
    lights=[
        habitat_sim.gfx.LightInfo(mn.Vector4(1.5,2.0,2.0,1.0),mn.Color3(2.5,2.5,2.5),habitat_sim.gfx.LightPositionModel.Global),
        habitat_sim.gfx.LightInfo(mn.Vector4(-1.5,1.0,1.0,1.0),mn.Color3(1.5,1.5,1.5),habitat_sim.gfx.LightPositionModel.Global),
        habitat_sim.gfx.LightInfo(mn.Vector4(0.0,2.5,-1.0,1.0),mn.Color3(1.0,1.0,1.0),habitat_sim.gfx.LightPositionModel.Global),
    ]
    sim.set_light_setup(lights,"candidate_lights")
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
                    image,pixels=_render_candidate(
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
