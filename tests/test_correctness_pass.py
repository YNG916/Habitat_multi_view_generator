import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mri_dataset.collector import write_collection_metadata
from mri_dataset.habitat_backend import HabitatBackend
from mri_dataset.hssd_preprocess import merge_scene_registry
from mri_dataset.object_review import (
    DATASET_PREFLIGHT_REPORT,
    run_and_publish_approved_object_preflight,
)
from mri_dataset.objects import validate_controlled_object_identity
from mri_dataset.regions import controlled_object_region_membership
from mri_dataset.scene_registry import FloorSpec, RegionSpec, SceneRegistry, SceneSpec
from mri_dataset.serialization import write_json
from mri_dataset.world_state import ObjectState


def object_state(category="box", identifier="objects/a.object_config.json"):
    return ObjectState(
        "object_001",
        category,
        "/historical/root/runtime-handle",
        [0.0, 0.2, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        asset_identifier=identifier,
    )


def rejected_scene(scene_id, marker):
    return SceneSpec(
        "hssd", scene_id, "train", False, [marker],
        f"nav/{scene_id}.navmesh", "", {}, "", "", "",
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], [],
    )


def registry(scenes):
    return SceneRegistry(
        "hssd", "dataset.json", "splits.yaml", list(scenes),
        preprocessing_config={"fingerprint": "same"},
    )


class PortableAssetTests(unittest.TestCase):
    def test_runtime_resolution_uses_identifier_not_historical_handle(self):
        canonical = "objects/a.object_config.json"
        backend = HabitatBackend.__new__(HabitatBackend)
        backend.config = SimpleNamespace(
            controlled_object_pools={"box": [canonical]},
            semantic_category_ids={"box": 14},
        )
        backend._runtime_handles_by_identifier = {canonical: "/new/root/runtime"}
        backend.resolve_runtime_handle = lambda handle: handle
        self.assertEqual(
            backend.runtime_handle_for_object_state(object_state()),
            "/new/root/runtime",
        )

    def test_approved_pool_membership_and_normalization(self):
        pools = {
            "box": ["objects/a.object_config.json"],
            "bag": ["objects/b.object_config.json"],
        }
        semantic = {"box": 14, "bag": 15}
        self.assertEqual(
            validate_controlled_object_identity(object_state(), pools, semantic), []
        )
        wrong = validate_controlled_object_identity(
            object_state("box", "objects/b.object_config.json"), pools, semantic
        )
        self.assertTrue(any("not approved" in item for item in wrong))
        unknown_asset = validate_controlled_object_identity(
            object_state("box", "objects/missing.object_config.json"), pools, semantic
        )
        self.assertTrue(any("unknown approved" in item for item in unknown_asset))
        unknown_category = validate_controlled_object_identity(
            object_state("cup"), pools, semantic
        )
        self.assertTrue(any("unknown controlled-object category" in item for item in unknown_category))
        noncanonical = validate_controlled_object_identity(
            object_state("box", "objects/../a.object_config.json"), pools, semantic
        )
        self.assertTrue(any("normalized canonical" in item for item in noncanonical))


class RegistrySubsetUpdateTests(unittest.TestCase):
    def test_replace_one_preserves_other_scenes(self):
        existing = registry([
            rejected_scene("A", "old-A"), rejected_scene("B", "old-B"),
            rejected_scene("C", "old-C"),
        ])
        merged = merge_scene_registry(
            existing, [rejected_scene("B", "new-B")], ["B"], registry([])
        )
        self.assertEqual([scene.scene_id for scene in merged.scenes], ["A", "B", "C"])
        self.assertEqual(merged.scene("B", require_eligible=False).rejection_reasons, ["new-B"])
        self.assertEqual(merged.scene("A", require_eligible=False).rejection_reasons, ["old-A"])

    def test_new_scene_is_added_and_repeat_is_deterministic(self):
        existing = registry([
            rejected_scene("A", "A"), rejected_scene("B", "B"),
            rejected_scene("C", "C"),
        ])
        first = merge_scene_registry(
            existing, [rejected_scene("D", "D")], ["D"], registry([])
        )
        second = merge_scene_registry(
            first, [rejected_scene("D", "D")], ["D"], registry([])
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual([scene.scene_id for scene in first.scenes], ["A", "B", "C", "D"])

    def test_duplicate_updates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            merge_scene_registry(
                registry([]),
                [rejected_scene("D", "one"), rejected_scene("D", "two")],
                ["D"],
                registry([]),
            )

    def test_incompatible_existing_fingerprint_is_rejected(self):
        existing = registry([rejected_scene("A", "A")])
        template = registry([])
        template.preprocessing_config = {"fingerprint": "changed"}
        with self.assertRaisesRegex(ValueError, "stale"):
            merge_scene_registry(
                existing, [rejected_scene("A", "new")], ["A"], template
            )


class RegionMembershipTests(unittest.TestCase):
    class Pathfinder:
        def snap_point(self, point): return np.asarray(point, dtype=np.float64)
        def is_navigable(self, point): return True
        def get_island(self, point): return 0

    @staticmethod
    def region(region_id, x0, x1):
        return RegionSpec(
            region_id, "room", "floor_00", 0.0,
            [[x0, 0.0, -1.0], [x1, 0.0, -1.0], [x1, 0.0, 1.0], [x0, 0.0, 1.0]],
            [0], 2.0, [[x0, -0.1, -1.0], [x1, 0.1, 1.0]],
            [[x0, -0.1, -1.0], [x1, 2.0, 1.0]], 1.5, True,
        )

    def setUp(self):
        self.left = self.region("left", -1.0, 0.0)
        self.right = self.region("right", 0.001, 1.0)
        self.floor = SimpleNamespace(regions=[self.left, self.right])
        self.pathfinder = self.Pathfinder()

    def check(self, position):
        return controlled_object_region_membership(
            position, 0.0, self.floor, self.left, self.pathfinder,
            lambda point: 0.0, 0.25,
        )

    def test_same_region_and_elevated_origin_pass(self):
        self.assertTrue(self.check([-0.5, 0.0, 0.0])["passed"])
        self.assertTrue(self.check([-0.5, 1.7, 0.0])["passed"])

    def test_other_region_and_close_across_wall_fail(self):
        self.assertFalse(self.check([0.8, 0.2, 0.0])["passed"])
        close = self.check([0.002, 0.2, 0.0])
        self.assertFalse(close["passed"])
        self.assertTrue(any("semantic region mismatch" in item for item in close["reasons"]))


class MetadataHierarchyTests(unittest.TestCase):
    def test_floor_metadata_is_invariant_across_regions(self):
        left = RegionMembershipTests.region("left", -1.0, 0.0)
        right = RegionMembershipTests.region("right", 0.001, 1.0)
        floor = FloorSpec(
            "floor_00", 0.0, [0], 4.0,
            [[-1.0, -0.1, -1.0], [1.0, 0.1, 1.0]],
            [[-1.0, -0.1, -1.0], [1.0, 2.0, 1.0]],
            1.5, True, [left, right], preprocessing_validation={"floor": True},
        )
        scene = SimpleNamespace(
            official_split="train", rendered_scene_aabb=[[-1, -1, -1], [1, 2, 1]],
            cached_navmesh_path="navmesh", navmesh_sha256="hash", navmesh_settings={"x": 1},
        )
        config = SimpleNamespace(
            scene_split=lambda scene_id: "train", region_context_margin_m=0.75
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def backend(region):
                return SimpleNamespace(
                    scene_id="scene", floor_id="floor_00", region_id=region.region_id,
                    region_category=region.region_category, scene_spec=scene,
                    floor_spec=floor, region_spec=region,
                )
            write_collection_metadata(backend(left), config, root)
            floor_path = root / "scenes/scene/floors/floor_00/floor.json"
            first = floor_path.read_bytes()
            write_collection_metadata(backend(right), config, root)
            self.assertEqual(first, floor_path.read_bytes())
            left_meta = json.loads((floor_path.parent / "regions/left/region.json").read_text())
            right_meta = json.loads((floor_path.parent / "regions/right/region.json").read_text())
            self.assertNotEqual(left_meta["navigable_bounds_world"], right_meta["navigable_bounds_world"])
            floor_meta = json.loads(floor_path.read_text())
            self.assertNotIn("region_id", floor_meta)
            self.assertNotIn("render_bev_bounds_world", floor_meta)


class PreflightPublicationTests(unittest.TestCase):
    def test_dataset_root_contract_and_all_entry_points_use_helper(self):
        report = {"passed": True, "schema_version": "1.0.0"}
        def fake(config, report_path=None, **kwargs):
            write_json(report_path, report)
            return report
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("mri_dataset.object_review.run_approved_object_preflight", side_effect=fake):
                result = run_and_publish_approved_object_preflight(object(), root)
            self.assertEqual(result, report)
            self.assertEqual(
                json.loads((root / DATASET_PREFLIGHT_REPORT).read_text()), report
            )
        repo = Path(__file__).resolve().parents[1]
        for script in ("generate_dataset.py", "collect_level1.py", "collect_level2.py"):
            source = (repo / "scripts" / script).read_text()
            self.assertIn("run_and_publish_approved_object_preflight", source)


if __name__ == "__main__":
    unittest.main()
