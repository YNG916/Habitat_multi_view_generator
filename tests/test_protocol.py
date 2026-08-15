import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mri_dataset.config import load_config
from mri_dataset.protocol import intervention_key, sample_intervention, stable_seed
from mri_dataset.serialization import load_numeric, save_numeric
from mri_dataset.world_state import ObjectState, RobotState, WorldState


def make_state(visible=True):
    robot = RobotState.create(
        "robot_01", [0.0, 0.0, 0.0], 0.0, 0.8, 64, 64, 90, 0.05, 20
    )
    robot.visibility = {
        "robot_02": {"benchmark_visible": visible, "visible_pixel_count": 100}
    }
    obj = ObjectState(
        "object_001",
        "book",
        "book.object_config.json",
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        visibility={
            "robot_01": {"benchmark_visible": visible, "visible_pixel_count": 100}
        },
    )
    return WorldState("0.1.0", "state_000001", "102815859", 0.0, 7, [robot], [obj])


class FormalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(require_preprocessed_registry=False)
        self.state = make_state()

    def test_stable_seed_is_repeatable_and_slot_sensitive(self):
        first = stable_seed(123, "102815859", "state_000001", "id", 1, 0)
        self.assertEqual(first, stable_seed(123, "102815859", "state_000001", "id", 1, 0))
        self.assertNotEqual(first, stable_seed(123, "102815859", "state_000001", "id", 2, 0))

    def test_every_intervention_type_is_sampled_from_frozen_domain(self):
        for edit_type in self.config.intervention_type_weights:
            edit = sample_intervention(
                self.state, self.config, "id", 12345, requested_type=edit_type
            )
            repeated = sample_intervention(
                self.state, self.config, "id", 12345, requested_type=edit_type
            )
            self.assertEqual(intervention_key(edit), intervention_key(repeated))
            self.assertEqual(edit.type, edit_type)

    def test_id_and_ood_numeric_domains_are_disjoint(self):
        for key, id_values in self.config.intervention_regimes["id"].items():
            self.assertFalse(
                set(map(float, id_values))
                & set(map(float, self.config.intervention_regimes["ood"][key]))
            )

    def test_visibility_gate_rejects_unobservable_targets(self):
        with self.assertRaisesRegex(ValueError, "No benchmark-visible.*target"):
            sample_intervention(make_state(False), self.config, "id", 2)

    def test_operational_counts_do_not_change_generation_fingerprint(self):
        changed = load_config(num_states=99, resume=False, require_preprocessed_registry=False)
        self.assertEqual(
            self.config.generation_fingerprint(), changed.generation_fingerprint()
        )


    def test_formal_config_rejects_nonstandard_hssd_variants(self):
        with self.assertRaisesRegex(ValueError, "standard"):
            load_config(
                require_preprocessed_registry=False,
                scene_dataset_config="data/hssd-hab-uncluttered.scene_dataset_config.json",
            )

    def test_scene_id_split_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "leak"):
            load_config(
                require_preprocessed_registry=False,
                scenes=["102815859"],
                scene_splits={
                    "train": ["102815859"],
                    "val": ["102815859"],
                    "test": [],
                },
            )


class FinishedRobotAssetTests(unittest.TestCase):
    asset_dir = Path(__file__).resolve().parents[1] / "assets" / "robot_proxies"

    def test_agents_share_one_grounded_finished_topology(self):
        geometry_payloads = []
        for color in ("red", "green", "blue"):
            obj_path = self.asset_dir / f"robot_{color}.obj"
            lines = obj_path.read_text(encoding="utf-8").splitlines()
            geometry = [
                line for line in lines
                if line.startswith(("v ", "vt ", "vn ", "f "))
            ]
            geometry_payloads.append(geometry)
            y_values = [
                float(line.split()[2]) for line in lines if line.startswith("v ")
            ]
            vertices = np.asarray([
                [float(value) for value in line.split()[1:4]]
                for line in lines if line.startswith("v ")
            ])
            self.assertAlmostEqual(min(y_values), 0.0, places=8)
            dimensions = np.ptp(vertices, axis=0)
            self.assertAlmostEqual(max(dimensions[0], dimensions[2]), 0.46, places=5)
            self.assertAlmostEqual(dimensions[1], 0.107, delta=0.002)
            config = json.loads(
                (self.asset_dir / f"robot_{color}.object_config.json").read_text()
            )
            self.assertFalse(config["use_bounding_box_for_collision"])
            self.assertEqual(config["COM"], [0.0, 0.0, 0.0])
        self.assertEqual(geometry_payloads[0], geometry_payloads[1])
        self.assertEqual(geometry_payloads[1], geometry_payloads[2])

    def test_agent_albedos_are_distinct_and_source_is_attributed(self):
        albedos = [
            (self.asset_dir / f"robot_{color}_albedo.png").read_bytes()
            for color in ("red", "green", "blue")
        ]
        self.assertEqual(len(set(albedos)), 3)
        attribution = (self.asset_dir / "ATTRIBUTION.md").read_text()
        self.assertIn("Moryak", attribution)
        self.assertIn("CC BY 4.0", attribution)
        self.assertTrue(
            (self.asset_dir / "source" / "robot_vacuum_original.fbx").is_file()
        )

    def test_formal_embodiment_uses_fixed_camera_height(self):
        config = load_config(require_preprocessed_registry=False)
        self.assertAlmostEqual(config.robot_body_diameter_m, 0.46, places=8)
        self.assertAlmostEqual(config.robot_body_height_m, 0.107, places=8)
        self.assertAlmostEqual(config.camera_height_min_m, 0.15, places=8)
        self.assertAlmostEqual(config.camera_height_max_m, 0.15, places=8)
        self.assertEqual(config.robot_proxy_height_variants, {})
        self.assertAlmostEqual(
            config.robot_camera_forward_offset_m, 0.235, places=8
        )


class NumericStorageTests(unittest.TestCase):
    def test_compressed_and_uncompressed_arrays_round_trip(self):
        array = np.arange(24, dtype=np.float32).reshape(4, 6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compressed = save_numeric(root / "compressed", array, True)
            plain = save_numeric(root / "plain", array, False)
            np.testing.assert_array_equal(load_numeric(root / compressed), array)
            np.testing.assert_array_equal(load_numeric(root / plain), array)


if __name__ == "__main__":
    unittest.main()
