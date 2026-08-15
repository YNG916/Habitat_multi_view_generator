import tempfile
import unittest
from pathlib import Path

from mri_dataset.hssd_preprocess import navmesh_settings_dict
from mri_dataset.config import load_config
from mri_dataset.scene_registry import (
    FloorSpec, SceneRegistry, SceneSpec, build_hssd_split_manifest,
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
        floor=FloorSpec(
            "floor_00",0.,[1],20.,
            [[0.,-.001,0.],[4.,.001,5.]],
            [[-.5,-1.,-.5],[4.5,3.,5.5]],
            2.2,True,
        )
        settings=navmesh_settings_dict(config)
        scene=SceneSpec(
            "hssd","scene_a","train",True,[],"navmeshes/scene_a.navmesh",
            "abc",settings,"fingerprint",
            [[-1.,-1.,-1.],[6.,4.,6.]],[floor],
        )
        registry=SceneRegistry(
            "hssd","hssd-hab.scene_dataset_config.json","scene_splits.yaml",[scene]
        )
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"registry.json"
            registry.save(path)
            loaded=SceneRegistry.load(path)
        self.assertEqual(loaded.scene("scene_a").floor("floor_00").allowed_island_ids,[1])
        self.assertTrue(settings["include_static_objects"])
        self.assertGreaterEqual(settings["agent_radius_m"],config.robot_body_diameter_m/2)


if __name__=="__main__": unittest.main()
