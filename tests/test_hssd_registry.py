import tempfile
import unittest
from pathlib import Path

from mri_dataset.hssd_preprocess import navmesh_settings_dict
from mri_dataset.config import load_config
from mri_dataset.scene_registry import (
    FloorSpec, RegionSpec, SceneRegistry, SceneSpec, build_hssd_split_manifest,
    load_official_hssd_splits, validate_hssd_split_manifest,
)


class HSSDSplitTests(unittest.TestCase):
    def test_official_mapping_and_scene_level_isolation(self):
        official={"train":["a","b","c","d"],"val":["e","f"]}
        manifest=build_hssd_split_manifest(official,17,.25)
        validate_hssd_split_manifest(manifest,official)
        self.assertEqual(set(manifest["scene_splits"]["test"]),{"e","f"})
        self.assertFalse(
            set(manifest["scene_splits"]["train"]) &
            set(manifest["scene_splits"]["val"])
        )

    def test_installed_official_split_file_is_valid(self):
        path=Path(__file__).resolve().parents[1]/"data/scene_datasets/hssd-hab/scene_splits.yaml"
        if not path.is_file(): self.skipTest("HSSD split file unavailable")
        splits=load_official_hssd_splits(path)
        self.assertGreater(len(splits["train"]),1)
        self.assertGreater(len(splits["val"]),1)
        self.assertFalse(set(splits["train"]) & set(splits["val"]))


class HSSDRegistryTests(unittest.TestCase):
    def test_floor_registry_round_trip(self):
        config=load_config(require_preprocessed_registry=False)
        region=RegionSpec(
            region_id="region_000_kitchen",region_category="kitchen",
            floor_id="floor_00",representative_floor_y=0.,
            semantic_polygon_world=[[0.,0.,0.],[4.,0.,0.],[4.,0.,5.],[0.,0.,5.]],
            allowed_island_ids=[1],navigable_area_m2=20.,
            navigable_bounds_world=[[0.,-.001,0.],[4.,.001,5.]],
            visual_bev_bounds_world=[[-.5,-1.,-.5],[4.5,3.,5.5]],
            bev_camera_height_m=2.2,eligible=True,
        )
        floor=FloorSpec(
            floor_id="floor_00",representative_floor_y=0.,
            allowed_island_ids=[1],navigable_area_m2=20.,
            navigable_bounds_world=[[0.,-.001,0.],[4.,.001,5.]],
            visual_bev_bounds_world=[[-.5,-1.,-.5],[4.5,3.,5.5]],
            bev_camera_height_m=2.2,eligible=True,regions=[region],
        )
        settings=navmesh_settings_dict(config)
        scene=SceneSpec(
            dataset_source="hssd",scene_id="scene_a",official_split="train",
            eligible=True,rejection_reasons=[],
            cached_navmesh_path="navmeshes/scene_a.navmesh",navmesh_sha256="abc",
            navmesh_settings=settings,navmesh_settings_fingerprint="nav-fingerprint",
            preprocessing_fingerprint="preprocess-fingerprint",
            semantic_regions_sha256="semantic-fingerprint",
            rendered_scene_aabb=[[-1.,-1.,-1.],[6.,4.,6.]],floors=[floor],
        )
        registry=SceneRegistry(
            "hssd","hssd-hab.scene_dataset_config.json","scene_splits.yaml",[scene]
        )
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"registry.json"
            registry.save(path)
            loaded=SceneRegistry.load(path)
        loaded_floor=loaded.scene("scene_a").floor("floor_00")
        self.assertEqual(loaded_floor.allowed_island_ids,[1])
        self.assertEqual(loaded_floor.region("region_000_kitchen").region_category,"kitchen")
        self.assertTrue(settings["include_static_objects"])
        self.assertGreaterEqual(settings["agent_radius_m"],config.robot_body_diameter_m/2)


if __name__=="__main__": unittest.main()
