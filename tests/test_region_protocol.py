import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mri_dataset.bev import BevMapping
from mri_dataset.collector import sample_controlled_objects, sample_object_choices
from mri_dataset.config import load_config
from mri_dataset.hssd_preprocess import _visual_bounds, preprocessing_fingerprint
from mri_dataset.interventions import Intervention, apply_intervention
from mri_dataset.level2 import _validate_edit, _validate_observable_transition
from mri_dataset.objects import (
    is_decomposed_canonical_id, resolve_canonical_templates,
)
from mri_dataset.regions import (
    point_in_polygon_xz, region_mask, unique_region_for_point,
)
from mri_dataset.sampling import sample_robot_positions
from mri_dataset.scene_registry import FloorSpec, RegionSpec
from mri_dataset.world_state import ObjectState, RobotState, WorldState


def region(region_id, polygon, floor_id="floor_00", eligible=True):
    points=np.asarray(polygon,dtype=float)
    low=points.min(axis=0); high=points.max(axis=0)
    low[1]-=.001; high[1]+=.001
    return RegionSpec(
        region_id=region_id,region_category="room",floor_id=floor_id,
        representative_floor_y=0.,semantic_polygon_world=points.tolist(),
        allowed_island_ids=[0],navigable_area_m2=8.,
        navigable_bounds_world=[low.tolist(),high.tolist()],
        visual_bev_bounds_world=[[low[0]-.5,-1.,low[2]-.5],[high[0]+.5,3.,high[2]+.5]],
        bev_camera_height_m=2.2,eligible=eligible,
        rejection_reasons=[] if eligible else ["test"],
    )


class RegionProtocolTests(unittest.TestCase):
    def setUp(self):
        self.left=region("left",[[-2,0,-1],[0,0,-1],[0,0,1],[-2,0,1]])
        self.right=region("right",[[0,0,-1],[2,0,-1],[2,0,1],[0,0,1]])

    def test_semantic_region_point_assignment_is_unique(self):
        self.assertIs(unique_region_for_point([-1,0,0],[self.left,self.right],.1),self.left)
        self.assertIs(unique_region_for_point([1,0,0],[self.left,self.right],.1),self.right)
        self.assertIsNone(unique_region_for_point([0,0,0],[self.left,self.right],.1))

    def test_euclidean_close_points_can_be_cross_region(self):
        a=np.array([-.01,0,0]); b=np.array([.01,0,0])
        self.assertLess(np.linalg.norm(a-b),.03)
        self.assertTrue(point_in_polygon_xz(a,self.left.semantic_polygon_world))
        self.assertFalse(point_in_polygon_xz(a,self.right.semantic_polygon_world))
        self.assertTrue(point_in_polygon_xz(b,self.right.semantic_polygon_world))

    def test_duplicate_region_ids_and_floor_mismatch_are_rejected(self):
        def floor(regions):
            return FloorSpec("floor_00",0.,[0],20.,[[-2,-.001,-1],[2,.001,1]],
                [[-2.5,-1,-1.5],[2.5,3,1.5]],2.2,True,regions)
        with self.assertRaisesRegex(ValueError,"duplicate region"):
            floor([self.left,deepcopy(self.left)]).validate()
        wrong=region("wrong",[[-1,0,-1],[1,0,-1],[1,0,1],[-1,0,1]],"floor_01")
        with self.assertRaisesRegex(ValueError,"floor mismatch"):
            floor([wrong]).validate()
        wrong_y=region("wrong_y",[[-1,0,-1],[1,0,-1],[1,0,1],[-1,0,1]])
        object.__setattr__(wrong_y,"representative_floor_y",.2)
        with self.assertRaisesRegex(ValueError,"floor Y mismatch"):
            floor([wrong_y]).validate()

        clipped_rejected=region(
            "clipped_rejected",[[-1,0,-1],[1,0,-1],[1,0,1],[-1,0,1]],
            eligible=False,
        )
        object.__setattr__(clipped_rejected,"visual_bev_bounds_world",[[-.5,-1,-.5],[.5,3,.5]])
        clipped_rejected.validate()
        clipped_eligible=deepcopy(clipped_rejected)
        object.__setattr__(clipped_eligible,"eligible",True)
        object.__setattr__(clipped_eligible,"rejection_reasons",[])
        with self.assertRaisesRegex(ValueError,"eligible semantic polygon is clipped"):
            clipped_eligible.validate()

    def test_region_local_bounds_and_registered_mask(self):
        bounds=_visual_bounds([[-1,0,-2],[2,.1,3]],[[-10,-2,-10],[10,4,10]],.75)
        self.assertEqual(bounds[0][0],-1.75); self.assertEqual(bounds[1][2],3.75)
        authored=[[-4,0,-3],[3,0,-3],[3,0,4],[-4,0,4]]
        expanded=_visual_bounds(
            [[-1,0,-2],[2,.1,3]],[[-10,-2,-10],[10,4,10]],.75,authored
        )
        self.assertEqual(expanded[0][0],-4.75)
        self.assertEqual(expanded[1][2],4.75)
        mapping=BevMapping(-2.,2.,-2.,2.,101,101)
        polygon=[[-1,0,-1],[1,0,-1],[1,0,1],[-1,0,1]]
        mask=region_mask(mapping,polygon)
        self.assertEqual(mask.shape,(101,101))
        self.assertEqual(mask[50,50],1); self.assertEqual(mask[0,0],0)
        self.assertAlmostEqual(mask.mean(),.25,delta=.03)

    def test_regions_of_one_scene_cannot_cross_splits(self):
        from mri_dataset.scene_registry import build_hssd_split_manifest
        manifest=build_hssd_split_manifest(
            {"train":["scene_a","scene_b"],"val":["scene_c"]},17,.5
        )
        scene_to_split={
            scene:split for split,scenes in manifest["scene_splits"].items()
            for scene in scenes
        }
        inherited=[scene_to_split["scene_a"] for _ in ("region_1","region_2","region_3")]
        self.assertEqual(len(set(inherited)),1)

    def test_preprocessing_fingerprint_changes_with_region_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            scene=Path(directory)/"scene.json"; semantic=Path(directory)/"semantic.json"
            scene.write_text("scene"); semantic.write_text("semantic")
            first=load_config(require_preprocessed_registry=False)
            second=load_config(require_preprocessed_registry=False,region_context_margin_m=.95)
            third=load_config(require_preprocessed_registry=False,region_min_navigable_area_m2=9.)
            a=preprocessing_fingerprint(first,scene,semantic)
            self.assertNotEqual(a,preprocessing_fingerprint(second,scene,semantic))
            self.assertNotEqual(a,preprocessing_fingerprint(third,scene,semantic))


class FakePathfinder:
    def __init__(self,points): self.points=[np.asarray(p,dtype=float) for p in points]; self.index=0
    def seed(self,value): pass
    @property
    def num_islands(self): return 1
    def island_area(self,index): return 10.
    def get_random_navigable_point(self,*args):
        point=self.points[self.index%len(self.points)]; self.index+=1; return point.copy()
    def get_island(self,point): return 0
    def is_navigable(self,point): return True
    def distance_to_closest_obstacle(self,point,max_distance): return 2.


class RegionSamplingTests(unittest.TestCase):
    def test_three_robot_sampling_stays_in_one_region(self):
        selected=region("selected",[[0,0,0],[3,0,0],[3,0,3],[0,0,3]])
        pathfinder=FakePathfinder([[-1,0,1],[.4,0,.4],[1.5,0,.4],[.4,0,1.5]])
        points=sample_robot_positions(
            pathfinder,np.random.default_rng(7),3,.1,.8,3.5,.1,
            allowed_island_ids=[0],representative_floor_y=0.,region_spec=selected,
        )
        self.assertEqual(len(points),3)
        self.assertTrue(all(point_in_polygon_xz(p,selected.semantic_polygon_world) for p in points))

    def test_controlled_object_rejects_outside_region_point(self):
        pathfinder=FakePathfinder([[-1,0,0],[1,0,0]])
        robot=RobotState.create("robot_01",[0,0,0],0,.15,8,8,90,.05,20)
        state=WorldState("0.4.0","state","scene",0.,1,[robot],region_id="selected")
        selected=SimpleNamespace(allowed_island_ids=[0])
        class Backend:
            controlled_handles={"box":["asset_a"]}; region_spec=selected
            sim=SimpleNamespace(pathfinder=pathfinder)
            render_bev_bounds=(np.array([-3,-1,-3]),np.array([3,2,3]))
            def point_in_region(self,point): return point[0]>=0
            def floor_surface_y(self,point): return 0.
            def object_collision_free(self,state,target): return True
            def create_object_state(self,category,asset,x,z,floor_y,index):
                return ObjectState(f"object_{index:03d}",category,asset,[x,.1,z],[0,0,0,1],
                    asset_identifier="objects/a.object_config.json",bbox={"min_world":[x-.1,0,z-.1],"max_world":[x+.1,.2,z+.1]})
        config=SimpleNamespace(
            controlled_objects_min_per_state=1,controlled_objects_max_per_state=1,
            local_sampling_radius_m=3.5,floor_tolerance_m=.1,
            controlled_object_min_separation_m=.65,
        )
        result=sample_controlled_objects(Backend(),state,config,np.random.default_rng(2))
        self.assertEqual(len(result),1); self.assertGreaterEqual(result[0].position_world[0],0)


class ObjectPoolTests(unittest.TestCase):
    def test_deterministic_multi_asset_category_and_variant_sampling(self):
        pools={"box":["a","b"],"toy":["c","d","e"]}
        first=sample_object_choices(pools,20,np.random.default_rng(123))
        second=sample_object_choices(pools,20,np.random.default_rng(123))
        self.assertEqual(first,second)
        self.assertGreater(len(set(first)),2)

    def test_canonical_resolution_is_exact_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); one=root/"objects/a.object_config.json"
            one.parent.mkdir(); one.write_text("{}"); outside=Path(directory).parent/"outside.json"
            resolved=resolve_canonical_templates([str(one),str(outside)],root,["objects/a.object_config.json"])
            self.assertEqual(resolved["objects/a.object_config.json"],str(one))
            with self.assertRaises(KeyError):
                resolve_canonical_templates([],root,["objects/a.object_config.json"])
            with self.assertRaises(KeyError):
                resolve_canonical_templates([str(one),str(one)],root,["objects/a.object_config.json"])

    def test_decomposed_assets_are_identified_portably(self):
        self.assertTrue(is_decomposed_canonical_id("objects/decomposed/x.object_config.json"))
        self.assertFalse(is_decomposed_canonical_id("objects/x.object_config.json"))


class RegionLevel2Tests(unittest.TestCase):
    def test_apply_intervention_preserves_region_and_bev_frame(self):
        robot=RobotState.create("robot_01",[0,0,0],0,.15,8,8,90,.05,20)
        obj=ObjectState("object_001","box","asset",[1,.1,0],[0,0,0,1])
        before=WorldState("0.4.0","before","scene",0.,1,[robot],[obj],
            region_id="room",bev={"bounds_world":[[-1,0,-1],[2,2,2]]})
        edit=Intervention("object_remove","object_001",{})
        after=apply_intervention(before,edit,"after")
        self.assertEqual(after.region_id,before.region_id)
        self.assertEqual(after.bev["bounds_world"],before.bev["bounds_world"])

    def test_observable_transition_distinguishes_remove_from_motion(self):
        robot=RobotState.create("robot_01",[0,0,0],0,.15,8,8,90,.05,20)
        obj=ObjectState("object_001","box","asset",[1,.1,0],[0,0,0,1])
        obj.visibility={"robot_01":{"benchmark_visible":True}}
        before=WorldState("0.4.0","before","scene",0.,1,[robot],[obj],region_id="room")
        removed=apply_intervention(before,Intervention("object_remove","object_001",{}),"after")
        removed.object("object_001").visibility={"robot_01":{"benchmark_visible":False}}
        _validate_observable_transition(
            before,removed,Intervention("object_remove","object_001",{}),1
        )
        moved=deepcopy(before)
        moved.object("object_001").visibility={"robot_01":{"benchmark_visible":False}}
        with self.assertRaisesRegex(ValueError,"not benchmark-visible after"):
            _validate_observable_transition(
                before,moved,Intervention("object_translate","object_001",{}),1
            )

    def test_region_or_bev_frame_mismatch_is_rejected(self):
        before=SimpleNamespace(region_id="room",bev={"bounds_world":[1]})
        after=SimpleNamespace(region_id="other",bev={"bounds_world":[1]})
        backend=SimpleNamespace(region_id="room")
        edit=Intervention("object_remove","object_001",{})
        with self.assertRaisesRegex(ValueError,"region mismatch"):
            _validate_edit(backend,before,after,edit)
        after.region_id="room"; after.bev={"bounds_world":[2]}
        with self.assertRaisesRegex(ValueError,"BEV frame mismatch"):
            _validate_edit(backend,before,after,edit)


if __name__=="__main__": unittest.main()
