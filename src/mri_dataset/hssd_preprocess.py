"""Offline standard-HSSD preprocessing for semantic-region-local collection."""
from __future__ import annotations
import hashlib,json,math,re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable,Optional,Sequence
import numpy as np
from PIL import Image
from .bev import habitat_orthographic_depth_to_metric, occupancy_from_pathfinder
from .protocol import stable_seed
from .regions import point_in_polygon_xz,polygon_area_xz,region_mask
from .scene_registry import FloorSpec,RegionSpec,SceneRegistry,SceneSpec,build_hssd_split_manifest,load_official_hssd_splits,sha256_file
from .serialization import write_json
from .visualization import (
    height_visualization, labeled_panel, occupancy_visualization,
    region_overlay_visualization,
)

PREPROCESSING_SCHEMA_VERSION="2.1.0-region-v7"
_PREPROCESS_FIELDS=(
"floor_samples_per_island","floor_min_samples_per_island","floor_min_island_area_m2",
"floor_min_navigable_area_m2","floor_group_tolerance_m","floor_max_vertical_span_m",
"floor_tolerance_m","region_context_margin_m","region_min_navigable_area_m2",
"region_min_extent_m","region_max_extent_m","region_min_sample_count",
"region_sampling_trials","region_min_sampling_success_rate",
"bev_preferred_camera_height_m","bev_ceiling_clearance_m","bev_ceiling_ray_samples",
"bev_ceiling_ray_max_distance_m","bev_min_ceiling_height_m","bev_open_scene_margin_m",
"bev_near","bev_far","preprocess_bev_width","preprocess_bev_min_finite_fraction",
"preprocess_bev_min_rgb_std","preprocess_region_mask_min_fraction",
"min_obstacle_distance_m","min_inter_robot_distance_m","local_sampling_radius_m","num_robots",
)

def navmesh_settings_dict(config):
    return {"agent_radius_m":float(config.navmesh_agent_radius_m),"agent_height_m":float(config.navmesh_agent_height_m),
    "agent_max_climb_m":float(config.navmesh_agent_max_climb_m),"agent_max_slope_deg":float(config.navmesh_agent_max_slope_deg),"include_static_objects":True}

def _fingerprint(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def navmesh_settings_fingerprint(settings): return _fingerprint(settings)

def preprocessing_config_dict(config):
    return {"schema":PREPROCESSING_SCHEMA_VERSION,**{name:getattr(config,name) for name in _PREPROCESS_FIELDS}}

def preprocessing_fingerprint(config,scene_file,semantic_file):
    scene_id=Path(scene_file).name.removesuffix(".scene_instance.json")
    scene_override=config.hssd_preprocess_overrides().get(scene_id,{})
    return _fingerprint({"algorithm":preprocessing_config_dict(config),
        "scene_instance_sha256":sha256_file(scene_file),
        "semantic_regions_sha256":sha256_file(semantic_file),
        "scene_override":scene_override})

def habitat_navmesh_settings(config):
    import habitat_sim
    settings=habitat_sim.NavMeshSettings(); settings.set_defaults()
    settings.agent_radius=float(config.navmesh_agent_radius_m); settings.agent_height=float(config.navmesh_agent_height_m)
    settings.agent_max_climb=float(config.navmesh_agent_max_climb_m); settings.agent_max_slope=float(config.navmesh_agent_max_slope_deg)
    settings.include_static_objects=True; return settings

def discover_installed_hssd_scenes(dataset_config_path):
    root=Path(dataset_config_path).resolve().parent/"scenes"; suffix=".scene_instance.json"
    return sorted(path.name[:-len(suffix)] for path in root.glob(f"*{suffix}") if path.is_file()) if root.is_dir() else []

def semantic_regions_path(config,scene_id):
    return config.dataset_config_path.resolve().parent/"semantics"/"scenes"/f"{scene_id}.semantic_config.json"

def load_hssd_region_annotations(path):
    data=json.loads(Path(path).read_text())
    records=data.get("region_annotations",[])
    if not isinstance(records,list) or not records: raise ValueError(f"No HSSD region_annotations in {path}")
    return records

def _create_simulator(config,scene_id,agent_config=None):
    import habitat_sim
    cfg=habitat_sim.SimulatorConfiguration(); cfg.scene_dataset_config_file=str(config.dataset_config_path)
    cfg.scene_id=str(scene_id); cfg.enable_physics=True; cfg.gpu_device_id=int(config.gpu_device_id)
    if agent_config is None:
        agent_config=habitat_sim.agent.AgentConfiguration(); agent_config.sensor_specifications=[]
    agents=list(agent_config) if isinstance(agent_config,(list,tuple)) else [agent_config]
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg,agents))

def _sample_island_points(pathfinder,island_id,count,seed):
    pathfinder.seed(int(seed)); points=[]
    for _ in range(int(count)):
        point=np.asarray(pathfinder.get_random_navigable_point(100,int(island_id)),dtype=np.float64)
        if np.all(np.isfinite(point)) and int(pathfinder.get_island(point))==int(island_id): points.append(point)
    return np.asarray(points,dtype=np.float64) if points else np.empty((0,3),dtype=np.float64)

def _group_islands_by_floor(records,tolerance_m):
    groups=[]
    for record in sorted(records,key=lambda item:item["median_y"]):
        matching=None
        for group in groups:
            y=sum(item["median_y"]*item["area_m2"] for item in group)/sum(item["area_m2"] for item in group)
            if abs(record["median_y"]-y)<=float(tolerance_m): matching=group; break
        (groups.append([record]) if matching is None else matching.append(record))
    return groups

def _bounds_from_points(points):
    low=points.min(axis=0); high=points.max(axis=0)
    for axis in range(3):
        if high[axis]-low[axis]<1e-3: low[axis]-=5e-4; high[axis]+=5e-4
    return [low.tolist(),high.tolist()]

def _visual_bounds(nav_bounds,scene_aabb,margin,semantic_polygon=None):
    """Bounds around all authored region vertices and navigable support."""
    nav_low,nav_high=(np.asarray(value,dtype=np.float64).copy() for value in nav_bounds)
    if semantic_polygon is not None:
        polygon=np.asarray(semantic_polygon,dtype=np.float64)
        if polygon.ndim!=2 or polygon.shape[0]<3 or polygon.shape[1]!=3 or not np.isfinite(polygon).all():
            raise ValueError("malformed semantic polygon bounds")
        nav_low[[0,2]]=np.minimum(nav_low[[0,2]],polygon[:,[0,2]].min(axis=0))
        nav_high[[0,2]]=np.maximum(nav_high[[0,2]],polygon[:,[0,2]].max(axis=0))
    low,high=(np.asarray(value,dtype=np.float64).copy() for value in scene_aabb)
    low[0]=max(low[0],nav_low[0]-margin); low[2]=max(low[2],nav_low[2]-margin)
    high[0]=min(high[0],nav_high[0]+margin); high[2]=min(high[2],nav_high[2]+margin)
    if high[0]-low[0]<.5 or high[2]-low[2]<.5: raise ValueError("degenerate visual bounds")
    return [low.tolist(),high.tolist()]

def _estimate_bev_camera_height(sim,points,floor_y,scene_aabb,config,override=None):
    if override is not None: return float(override),{"method":"override","chosen_height_m":float(override)}
    import habitat_sim
    candidates=[]; count=min(int(config.bev_ceiling_ray_samples),len(points))
    sampled=points[np.linspace(0,max(0,len(points)-1),count,dtype=np.int64)]
    for point in sampled:
        origin=np.asarray(point,dtype=np.float64); origin[1]=floor_y+.05
        result=sim.cast_ray(habitat_sim.geo.Ray(origin.astype(np.float32),np.array([0,1,0],np.float32)),
            max_distance=float(config.bev_ceiling_ray_max_distance_m),buffer_distance=0.)
        for hit in result.hits:
            clearance=.05+float(hit.ray_distance)
            if clearance>=float(config.bev_min_ceiling_height_m): candidates.append(clearance); break
    preferred=float(config.bev_preferred_camera_height_m)
    if candidates:
        ceiling=float(np.percentile(candidates,20)); chosen=min(preferred,ceiling-float(config.bev_ceiling_clearance_m)); method="region_upward_ray_p20"
    else:
        ceiling=None; chosen=max(preferred,float(scene_aabb[1][1])-floor_y+float(config.bev_open_scene_margin_m)); method="open_region"
    chosen=min(chosen,float(config.bev_far)-.25)
    if chosen<=max(.5,float(config.camera_height_max_m)+.1): raise ValueError("No safe region BEV height")
    return chosen,{"method":method,"rays_sampled":len(sampled),"valid_ceiling_hits":len(candidates),"ceiling_estimate_m":ceiling,"chosen_height_m":chosen}

def _sampling_success(pathfinder,points,config,seed):
    points=np.asarray([
        point for point in points
        if pathfinder.distance_to_closest_obstacle(point,2.0)
        >=float(config.min_obstacle_distance_m)
    ],dtype=np.float64)
    if len(points)<config.num_robots: return 0.
    rng=np.random.default_rng(seed); successes=0; trials=int(config.region_sampling_trials)
    for _ in range(trials):
        anchor=points[int(rng.integers(len(points)))]; selected=[anchor]
        candidates=points[np.linalg.norm(points[:,[0,2]]-anchor[[0,2]],axis=1)<=float(config.local_sampling_radius_m)]
        rng.shuffle(candidates)
        for point in candidates:
            if all(np.linalg.norm(point[[0,2]]-old[[0,2]])>=float(config.min_inter_robot_distance_m) for old in selected):
                selected.append(point)
                if len(selected)==config.num_robots: break
        successes+=int(len(selected)==config.num_robots)
    return successes/max(1,trials)

def _region_id(index,name):
    slug=re.sub(r"[^a-z0-9]+","_",str(name).lower()).strip("_") or "unknown"
    return f"region_{index:03d}_{slug}"

def _region_specs(sim,config,scene_id,floor_id,floor_y,group,annotations,scene_aabb,override):
    island_points={item["island_id"]:item["points"] for item in group}
    relevant=[(index,a) for index,a in enumerate(annotations) if abs(float(a.get("floor_height",0))-floor_y)<=float(config.floor_tolerance_m)]
    assignments={index:[] for index,_ in relevant}; ambiguous=0; unassigned=0
    for island_id,points in island_points.items():
        for point in points:
            matches=[index for index,a in relevant if point_in_polygon_xz(point,a.get("poly_loop",[]))]
            if len(matches)==1: assignments[matches[0]].append((island_id,point))
            elif len(matches)>1: ambiguous+=1
            else: unassigned+=1
    result=[]
    for index,annotation in relevant:
        values=assignments[index]; reasons=[]; polygon=annotation.get("poly_loop",[])
        if polygon_area_xz(polygon)<=0: reasons.append("malformed_semantic_region")
        if len(values)<int(config.region_min_sample_count): reasons.append("insufficient_region_nav_samples")
        if not values: continue
        points=np.asarray([value[1] for value in values],dtype=np.float64)
        polygon_array=np.asarray(polygon,dtype=np.float64)
        polygon_extent_x=float(np.ptp(polygon_array[:,0]))
        polygon_extent_z=float(np.ptp(polygon_array[:,2]))
        counts={island_id:sum(1 for value in values if value[0]==island_id) for island_id in island_points}
        area=sum(float(next(item["area_m2"] for item in group if item["island_id"]==island_id))*count/max(1,len(island_points[island_id])) for island_id,count in counts.items())
        nav_bounds=_bounds_from_points(points); extent_x=nav_bounds[1][0]-nav_bounds[0][0]; extent_z=nav_bounds[1][2]-nav_bounds[0][2]
        if area<float(config.region_min_navigable_area_m2): reasons.append("region_navigable_area_too_small")
        if min(extent_x,extent_z)<float(config.region_min_extent_m): reasons.append("region_useful_extent_too_small")
        if max(extent_x,extent_z,polygon_extent_x,polygon_extent_z)>float(config.region_max_extent_m): reasons.append("region_extent_too_large")
        success=_sampling_success(
            sim.pathfinder,points,config,
            stable_seed(config.random_seed,"region_success",scene_id,index),
        )
        if success<float(config.region_min_sampling_success_rate): reasons.append("region_three_robot_sampling_unreliable")
        visual=_visual_bounds(
            nav_bounds,scene_aabb,float(config.region_context_margin_m),polygon
        )
        tolerance=1e-5
        polygon_contained=bool(np.all(
            (polygon_array[:,0]>=visual[0][0]-tolerance)
            &(polygon_array[:,0]<=visual[1][0]+tolerance)
            &(polygon_array[:,2]>=visual[0][2]-tolerance)
            &(polygon_array[:,2]<=visual[1][2]+tolerance)
        ))
        if not polygon_contained:
            reasons.append("semantic_polygon_outside_rendered_scene_bounds")
        region_id=_region_id(index,annotation.get("name",annotation.get("label",index)))
        camera_override=override.get(region_id,{}).get("bev_camera_height_m") if isinstance(override.get(region_id,{}),dict) else None
        try: height,height_report=_estimate_bev_camera_height(sim,points,floor_y,scene_aabb,config,camera_override)
        except ValueError as exc: height=float(config.bev_preferred_camera_height_m); height_report={"error":str(exc)}; reasons.append("unsafe_region_bev_camera_height")
        result.append(RegionSpec(region_id=region_id,region_category=str(annotation.get("label",annotation.get("name","unknown"))),
            floor_id=floor_id,representative_floor_y=floor_y,semantic_polygon_world=[[float(v) for v in p] for p in polygon],
            allowed_island_ids=sorted(k for k,v in counts.items() if v),navigable_area_m2=float(area),
            navigable_bounds_world=nav_bounds,visual_bev_bounds_world=visual,bev_camera_height_m=height,
            eligible=not reasons,rejection_reasons=reasons,preprocessing_validation={
            "semantic_name":annotation.get("name"),"polygon_area_m2":polygon_area_xz(polygon),
            "semantic_polygon_extent_xz_m":[polygon_extent_x,polygon_extent_z],
            "semantic_polygon_fully_contained_in_bev":polygon_contained,
            "navigable_extent_xz_m":[extent_x,extent_z],
            "sample_count":len(points),
            "navigable_samples_inside_visual_bev_fraction":float(np.mean(
                (points[:,0]>=visual[0][0])&(points[:,0]<=visual[1][0])
                &(points[:,2]>=visual[0][2])&(points[:,2]<=visual[1][2])
            )),
            "floor_height_p05_p50_p95_m":[float(v) for v in np.percentile(points[:,1],[5,50,95])],
            "floor_height_span_p05_p95_m":float(np.percentile(points[:,1],95)-np.percentile(points[:,1],5)),
            "estimated_robot_sampling_success_rate":success,"bev_camera_height":height_report}))
    return result,{"ambiguous_point_count":ambiguous,"unassigned_point_count":unassigned,"candidate_region_count":len(relevant)}

def _bev_sanity_checks(config,scene_id,regions,preview_paths=None):
    """Render every eligible region with one scene load and one agent each."""
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
    regions=list(regions)
    preview_paths=list(preview_paths or [None]*len(regions))
    if len(preview_paths)!=len(regions):
        raise ValueError("Region/preview count mismatch")
    agents=[]; geometries=[]
    for region in regions:
        bounds=region.visual_bev_bounds_world
        extent_x=bounds[1][0]-bounds[0][0]
        extent_z=bounds[1][2]-bounds[0][2]
        width=int(config.preprocess_bev_width)
        height=max(32,int(round(width*extent_z/extent_x)))
        specs=[]
        for uuid,sensor_type in (
            ("rgb",habitat_sim.SensorType.COLOR),
            ("depth",habitat_sim.SensorType.DEPTH),
        ):
            spec=habitat_sim.CameraSensorSpec()
            spec.uuid=uuid; spec.sensor_type=sensor_type
            spec.sensor_subtype=habitat_sim.SensorSubType.ORTHOGRAPHIC
            spec.resolution=[height,width]
            spec.near=float(config.bev_near); spec.far=float(config.bev_far)
            spec.ortho_scale=1./extent_x
            specs.append(spec)
        agent=habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications=specs
        agents.append(agent)
        geometries.append((bounds,extent_x,extent_z,width,height))
    if not agents:
        return []
    sim=_create_simulator(config,scene_id,agents)
    try:
        navmesh_path=config.navmesh_cache_path(scene_id)
        if not sim.pathfinder.load_nav_mesh(str(navmesh_path)):
            raise RuntimeError(f"Could not load cached NavMesh for sanity: {navmesh_path}")
        for index,(region,geometry) in enumerate(zip(regions,geometries)):
            bounds=geometry[0]
            state=habitat_sim.AgentState()
            state.position=np.array([
                (bounds[0][0]+bounds[1][0])*.5,
                region.representative_floor_y+region.bev_camera_height_m,
                (bounds[0][2]+bounds[1][2])*.5,
            ],np.float32)
            state.rotation=quat_from_angle_axis(
                -math.pi/2,np.array([1.,0.,0.])
            )
            sim.get_agent(index).set_state(state,infer_sensor_states=True)
        observations=sim.get_sensor_observations(
            agent_ids=list(range(len(regions)))
        )
        reports=[]
        from .bev import BevMapping
        for index,(region,preview_path,geometry) in enumerate(zip(
            regions,preview_paths,geometries
        )):
            obs=observations[index]
            bounds,extent_x,extent_z,width,height=geometry
            rgb=np.asarray(obs["rgb"])[...,:3].astype(np.uint8)
            metric=habitat_orthographic_depth_to_metric(
                np.asarray(obs["depth"],np.float32),config.bev_near,config.bev_far
            )
            mapping=BevMapping(
                float(bounds[0][0]),float(bounds[1][0]),
                float(bounds[0][2]),float(bounds[1][2]),width,height,
            )
            mask=region_mask(mapping,region.semantic_polygon_world)
            occupancy=occupancy_from_pathfinder(
                sim.pathfinder,mapping,region.representative_floor_y,
                navmesh_bounds=sim.pathfinder.get_bounds(),
                allowed_island_ids=region.allowed_island_ids,
                vertical_tolerance_m=float(config.floor_tolerance_m),
            )
            occupancy_values=set(map(int,np.unique(occupancy)))
            occupancy_binary=occupancy_values.issubset({0,1}) and 1 in occupancy_values
            navigable=occupancy==1
            selected_nav=navigable&(mask>0)
            coverage=float(mask.mean())
            finite=float(np.isfinite(metric).mean())
            std=float(rgb.astype(np.float32).std())
            passed=(
                finite>=float(config.preprocess_bev_min_finite_fraction)
                and std>=float(config.preprocess_bev_min_rgb_std)
                and coverage>=float(config.preprocess_region_mask_min_fraction)
                and occupancy_binary and bool(selected_nav.any())
            )
            if preview_path:
                Path(preview_path).parent.mkdir(parents=True,exist_ok=True)
                height_map=(region.bev_camera_height_m-metric).astype(np.float32)
                occupancy_vis=occupancy_visualization(occupancy,mask)
                overlay=region_overlay_visualization(rgb,mask,occupancy)
                panels=[
                    labeled_panel(Image.fromarray(rgb),"RGB"),
                    labeled_panel(overlay,"RGB + OFFICIAL HSSD REGION"),
                    labeled_panel(Image.fromarray(mask*255,mode="L"),"SEMANTIC REGION MASK"),
                    labeled_panel(occupancy_vis,"NAVMESH: GREEN = IN REGION"),
                    labeled_panel(height_visualization(height_map),"METRIC HEIGHT"),
                ]
                preview=np.concatenate([np.asarray(panel) for panel in panels],axis=1)
                Image.fromarray(preview).save(preview_path)
            reports.append({
                "passed":passed,"resolution_hw":[height,width],
                "metric_extent_xz_m":[extent_x,extent_z],
                "finite_metric_depth_fraction":finite,"rgb_std":std,
                "region_mask_coverage":coverage,
                "occupancy_binary":occupancy_binary,
                "occupancy_values":sorted(occupancy_values),
                "occupancy_navigable_pixel_fraction":float(navigable.mean()),
                "selected_region_occupancy_fraction":float(
                    selected_nav.sum()/max(1,navigable.sum())
                ),
                "region_mask_navigable_fraction":float(
                    selected_nav.sum()/max(1,(mask>0).sum())
                ),
                "preview_panels":["rgb","rgb_region_overlay","region_mask","occupancy","height"],
                "preview_path":str(preview_path) if preview_path else None,
            })
        return reports
    finally:
        sim.close()


def _relative(config,path):
    try:return str(Path(path).resolve().relative_to(config.repo_root))
    except ValueError:return str(Path(path).resolve())

def _failed_scene(config,scene_id,split,reason,nav_path,settings,nav_fp,pre_fp="",sem_hash=""):
    return SceneSpec("hssd",scene_id,split,False,[reason],_relative(config,nav_path),"",settings,nav_fp,pre_fp,sem_hash,[[0,0,0],[1,1,1]],[],{})

def preprocess_hssd_scene(config,scene_id,official_split,preview_root=None):
    settings=navmesh_settings_dict(config); nav_fp=navmesh_settings_fingerprint(settings); nav_path=config.navmesh_cache_path(scene_id)
    root=config.dataset_config_path.resolve().parent; scene_file=root/"scenes"/f"{scene_id}.scene_instance.json"; sem_file=semantic_regions_path(config,scene_id)
    if not scene_file.is_file(): return _failed_scene(config,scene_id,official_split,"missing_scene_instance",nav_path,settings,nav_fp)
    if not sem_file.is_file(): return _failed_scene(config,scene_id,official_split,"missing_semantic_regions",nav_path,settings,nav_fp)
    sem_hash=sha256_file(sem_file); pre_fp=preprocessing_fingerprint(config,scene_file,sem_file); sim=None
    scene_override=config.hssd_preprocess_overrides().get(scene_id,{})
    skip_reason=scene_override.get("skip_scene_reason") if isinstance(scene_override,dict) else None
    if skip_reason:
        return _failed_scene(
            config,scene_id,official_split,f"manual_preprocess_rejection:{skip_reason}",
            nav_path,settings,nav_fp,pre_fp,sem_hash,
        )
    try:
        annotations=load_hssd_region_annotations(sem_file); sim=_create_simulator(config,scene_id); aabb=sim.scene_aabb
        scene_aabb=[np.asarray(aabb.min,dtype=float).tolist(),np.asarray(aabb.max,dtype=float).tolist()]
        if not sim.recompute_navmesh(sim.pathfinder,habitat_navmesh_settings(config)): raise RuntimeError("recompute_navmesh false")
        nav_path.parent.mkdir(parents=True,exist_ok=True)
        if not sim.pathfinder.save_nav_mesh(str(nav_path)): raise RuntimeError("save_nav_mesh false")
        records=[]; rejected=[]
        for island_id in range(int(sim.pathfinder.num_islands)):
            area=float(sim.pathfinder.island_area(island_id))
            if area<float(config.floor_min_island_area_m2): rejected.append({"island_id":island_id,"reason":"island_too_small","area_m2":area}); continue
            points=_sample_island_points(sim.pathfinder,island_id,int(config.floor_samples_per_island),stable_seed(config.random_seed,"preprocess",scene_id,island_id))
            if len(points)<int(config.floor_min_samples_per_island): rejected.append({"island_id":island_id,"reason":"insufficient_samples"}); continue
            y05,median,y95=np.percentile(points[:,1],[5,50,95])
            if y95-y05>float(config.floor_max_vertical_span_m): rejected.append({"island_id":island_id,"reason":"connected_multilevel_island"}); continue
            records.append({"island_id":island_id,"area_m2":area,"median_y":float(median),"vertical_span_m":float(y95-y05),"points":points})
        floors=[]
        for floor_index,group in enumerate(_group_islands_by_floor(records,config.floor_group_tolerance_m)):
            floor_id=f"floor_{floor_index:02d}"; points=np.concatenate([item["points"] for item in group]); area=sum(item["area_m2"] for item in group)
            floor_y=sum(item["median_y"]*item["area_m2"] for item in group)/area
            nav_bounds=_bounds_from_points(points); floor_visual=_visual_bounds(nav_bounds,scene_aabb,float(config.bev_context_margin_m))
            regions,assignment=_region_specs(sim,config,scene_id,floor_id,floor_y,group,annotations,scene_aabb,scene_override.get(floor_id,scene_override) if isinstance(scene_override,dict) else {})
            eligible_regions=[region for region in regions if region.eligible]
            previews=[
                Path(preview_root)/f"{scene_id}_{floor_id}_{region.region_id}.png"
                if preview_root else None
                for region in eligible_regions
            ]
            try:
                sanity_reports=_bev_sanity_checks(
                    config,scene_id,eligible_regions,previews
                )
            except Exception as exc:
                sanity_reports=[
                    {"passed":False,"error":f"{type(exc).__name__}:{exc}"}
                    for _ in eligible_regions
                ]
            sanity_by_id={
                region.region_id:report
                for region,report in zip(eligible_regions,sanity_reports)
            }
            checked=[]
            for region in regions:
                if not region.eligible:
                    checked.append(region)
                    continue
                sanity=sanity_by_id[region.region_id]
                reasons=[] if sanity["passed"] else ["region_bev_sanity_failed"]
                validation=dict(region.preprocessing_validation)
                validation["low_resolution_bev"]=sanity
                checked.append(RegionSpec(**{
                    **asdict(region),"eligible":not reasons,
                    "rejection_reasons":reasons,
                    "preprocessing_validation":validation,
                }))
            floor_reasons=[] if any(r.eligible for r in checked) else ["no_eligible_region"]
            floors.append(FloorSpec(floor_id,float(floor_y),sorted(item["island_id"] for item in group),float(area),nav_bounds,floor_visual,
                float(config.bev_preferred_camera_height_m),not floor_reasons,checked,floor_reasons,{"region_assignment":assignment}))
        eligible=any(f.eligible for f in floors)
        return SceneSpec("hssd",scene_id,official_split,eligible,[] if eligible else ["no_eligible_floor"],_relative(config,nav_path),
            sha256_file(nav_path),settings,nav_fp,pre_fp,sem_hash,scene_aabb,floors,
            {"scene_file":_relative(config,scene_file),"semantic_regions_file":_relative(config,sem_file),"semantic_region_count":len(annotations),"rejected_islands":rejected})
    except Exception as exc:
        return _failed_scene(config,scene_id,official_split,f"preprocess_failure:{type(exc).__name__}:{exc}",nav_path,settings,nav_fp,pre_fp,sem_hash)
    finally:
        if sim is not None: sim.close()


def merge_scene_registry(existing, updated_scenes, selected_scene_ids, template):
    """Deterministically replace selected entries while preserving all others."""
    selected = set(map(str, selected_scene_ids))
    updates = list(updated_scenes)
    update_ids = [scene.scene_id for scene in updates]
    if len(update_ids) != len(set(update_ids)):
        raise ValueError("Subset preprocessing produced duplicate scene IDs")
    if set(update_ids) != selected:
        raise ValueError("Subset preprocessing did not produce every selected scene")
    preserved = {}
    if existing is not None:
        if existing.schema_version != template.schema_version:
            raise ValueError("Existing SceneRegistry schema is incompatible")
        if existing.dataset_source != template.dataset_source:
            raise ValueError("Existing SceneRegistry dataset source is incompatible")
        if existing.dataset_config_path != template.dataset_config_path:
            raise ValueError("Existing SceneRegistry HSSD root/config is incompatible")
        if existing.official_splits_path != template.official_splits_path:
            raise ValueError("Existing SceneRegistry official split source is incompatible")
        if existing.preprocessing_config != template.preprocessing_config:
            raise ValueError(
                "Existing SceneRegistry preprocessing fingerprint/config is stale; "
                "run an explicit full rebuild"
            )
        preserved = {
            scene.scene_id: scene
            for scene in existing.scenes
            if scene.scene_id not in selected
        }
    for scene in updates:
        preserved[scene.scene_id] = scene
    template.scenes = [preserved[scene_id] for scene_id in sorted(preserved)]
    template.statistics = template.compute_statistics()
    template.validate()
    return template

def preprocess_hssd(config,scene_ids:Optional[Iterable[str]]=None,limit=None,preview_root=None):
    official=load_official_hssd_splits(config.official_scene_splits_path); installed=discover_installed_hssd_scenes(config.dataset_config_path)
    lookup={scene:split for split,values in official.items() for scene in values}
    selected=sorted(set(scene_ids) if scene_ids is not None else set(lookup)&set(installed))
    if limit is not None:selected=selected[:int(limit)]
    if not selected or set(selected)-set(lookup): raise ValueError("No valid official HSSD scenes selected")
    subset_update = scene_ids is not None or limit is not None
    existing_registry = None
    existing={}
    if config.scene_registry_path.is_file():
        try:
            existing_registry = SceneRegistry.load(config.scene_registry_path)
        except (OSError,ValueError):
            if subset_update:
                raise
        if existing_registry is not None:
            existing={s.scene_id:s for s in existing_registry.scenes}
    nav_fp=navmesh_settings_fingerprint(navmesh_settings_dict(config)); scenes=[]
    for index,scene_id in enumerate(selected,1):
        scene_file=config.dataset_config_path.resolve().parent/"scenes"/f"{scene_id}.scene_instance.json"; sem_file=semantic_regions_path(config,scene_id)
        expected_pre=preprocessing_fingerprint(config,scene_file,sem_file) if scene_file.is_file() and sem_file.is_file() else ""
        cached=existing.get(scene_id); cached_path=config.resolve(cached.cached_navmesh_path) if cached else None
        reusable=bool(cached and cached.official_split==lookup[scene_id] and cached.navmesh_settings_fingerprint==nav_fp and
            cached.preprocessing_fingerprint==expected_pre and cached.semantic_regions_sha256==sha256_file(sem_file) and
            cached_path and cached_path.is_file() and cached.navmesh_sha256==sha256_file(cached_path))
        print(f"[{index}/{len(selected)}] {'reusing' if reusable else 'preprocessing'} HSSD regions {scene_id}",flush=True)
        scenes.append(cached if reusable else preprocess_hssd_scene(config,scene_id,lookup[scene_id],preview_root))
    template = SceneRegistry(
        "hssd",
        _relative(config,config.dataset_config_path),
        _relative(config,config.official_scene_splits_path),
        [],
        preprocessing_config=preprocessing_config_dict(config),
    )
    registry = (
        merge_scene_registry(existing_registry, scenes, selected, template)
        if subset_update
        else merge_scene_registry(None, scenes, selected, template)
    )
    registry.save(config.scene_registry_path)
    manifest=build_hssd_split_manifest(official,int(config.split_seed),float(config.internal_val_fraction),available_scene_ids=installed)
    manifest["official_splits_sha256"]=sha256_file(config.official_scene_splits_path); write_json(config.split_manifest_path,manifest)
    return SceneRegistry.load(config.scene_registry_path)
