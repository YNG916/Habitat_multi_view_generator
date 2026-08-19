import json
import unittest
from pathlib import Path
import numpy as np

try:
    import habitat_sim  # noqa: F401
    HABITAT_AVAILABLE=True
except ImportError:
    HABITAT_AVAILABLE=False

from mri_dataset.calibration import validate_multilevel_orthographic_depth
from mri_dataset.collector import make_world_state
from mri_dataset.config import REPO_ROOT, load_config
from mri_dataset.habitat_backend import HabitatBackend


class HabitatDatasetIndependentTests(unittest.TestCase):
    @unittest.skipUnless(HABITAT_AVAILABLE,"Habitat-Sim unavailable")
    def test_multilevel_orthographic_depth(self):
        config=load_config(require_preprocessed_registry=False)
        report=validate_multilevel_orthographic_depth(config)
        self.assertTrue(report["passed"],report)
        self.assertTrue(report["dataset_independent"])
        self.assertLessEqual(report["max_abs_error_m"],config.height_validation_max_error_m)


def hssd_ready():
    registry=REPO_ROOT/"data/hssd_processed/scene_registry_smoke.json"
    objects=REPO_ROOT/"configs/hssd_controlled_objects.json"
    if not (HABITAT_AVAILABLE and registry.is_file() and objects.is_file()):
        return False
    try:
        return bool(json.loads(objects.read_text()).get("approved_assets"))
    except Exception:
        return False


@unittest.skipUnless(hssd_ready(),"Preprocessed standard HSSD assets unavailable")
class HSSDIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=load_config(
            str(REPO_ROOT/"configs/collector_hssd_smoke.json"),
            width=96,height=96,bev_meters_per_pixel=.08,
            save_visualizations=False,height_validation_samples=32,
        )
        cls.scene,cls.floor,cls.region=cls.config.collection_specs()[0]
        cls.backend=HabitatBackend(cls.config,cls.scene,cls.floor,cls.region)

    @classmethod
    def tearDownClass(cls): cls.backend.close()

    def test_real_hssd_render_semantic_and_grounding(self):
        state=make_world_state(self.backend,self.config,"hssd_integration",123)
        self.assertEqual(state.dataset_source,"hssd")
        self.assertEqual(state.floor_id,self.floor.floor_id)
        self.assertEqual(state.region_id,self.region.region_id)
        outputs=self.backend.render(state)
        self.assertEqual(outputs["bev_semantic"].dtype,np.uint16)
        self.assertGreater(np.isfinite(outputs["bev_metric_depth"]).sum(),0)
        self.assertEqual(set(map(int,np.unique(outputs["occupancy"]))),{0,1})
        expected=((outputs["occupancy"]==1)&(outputs["region_mask"]==1))
        self.assertGreater(int(expected.sum()),0)
        for robot in state.robots:
            support=self.backend.robot_support_report(state,robot.robot_id)
            self.assertAlmostEqual(support["support_gap_m"],0.,places=4)
            self.assertAlmostEqual(support["proxy_origin_offset_from_base_m"],0.,places=5)
        self.backend.apply_world_state(state)
        manager=self.backend.sim.get_rigid_object_manager()
        for obj in state.objects:
            rigid_id=self.backend.render_ids[obj.instance_id][0]
            rigid=manager.get_object_by_id(rigid_id)
            floor_y=self.backend.floor_surface_y(obj.position_world)
            visual_bottom=(
                float(rigid.translation[1])
                +float(rigid.root_scene_node.cumulative_bb.min[1])
            )
            self.assertAlmostEqual(float(rigid.margin),0.0,places=7)
            self.assertLessEqual(
                abs(visual_bottom-floor_y),
                0.01,
                "approved collider/visual support offset exceeds 1 cm",
            )

    def test_floor_local_sampling_and_cached_navmesh(self):
        state=make_world_state(self.backend,self.config,"hssd_floor",456)
        allowed=set(self.region.allowed_island_ids)
        for robot in state.robots:
            island=int(self.backend.sim.pathfinder.get_island(robot.base_position_world))
            self.assertIn(island,allowed)
            self.assertAlmostEqual(
                robot.camera.position_world[1]-robot.base_position_world[1],
                .15,places=7,
            )
        self.assertTrue(self.backend.navmesh_path.is_file())
        self.assertTrue(self.scene.navmesh_settings["include_static_objects"])


if __name__=="__main__": unittest.main()
