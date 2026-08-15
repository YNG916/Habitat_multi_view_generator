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
    return WorldState("0.1.0", "state_000001", "apt_1", 0.0, 7, [robot], [obj])


class FormalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.state = make_state()

    def test_stable_seed_is_repeatable_and_slot_sensitive(self):
        first = stable_seed(123, "apt_1", "state_000001", "id", 1, 0)
        self.assertEqual(first, stable_seed(123, "apt_1", "state_000001", "id", 1, 0))
        self.assertNotEqual(first, stable_seed(123, "apt_1", "state_000001", "id", 2, 0))

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
        changed = load_config(num_states=99, resume=False)
        self.assertEqual(
            self.config.generation_fingerprint(), changed.generation_fingerprint()
        )


    def test_replica_scene_ids_are_grouped_by_real_stage_family(self):
        self.assertEqual(
            self.config.layout_family("apt_0"), "frl_apartment_stage"
        )
        self.assertEqual(
            self.config.layout_family("apt_5"), "frl_apartment_stage"
        )
        self.assertEqual(
            self.config.layout_family("v3_sc2_staging_00"), "v3_sc2"
        )
        self.assertEqual(
            self.config.layout_family("v3_sc2_staging_19"), "v3_sc2"
        )

    def test_same_stage_family_cannot_cross_splits(self):
        with self.assertRaisesRegex(ValueError, "layout family appears"):
            load_config(
                scenes=["apt_0", "apt_5"],
                scene_splits={"train": ["apt_0"], "val": ["apt_5"], "test": []},
                scene_overrides={
                    "apt_0": {"bev_camera_height_m": 2.2},
                    "apt_5": {"bev_camera_height_m": 2.2},
                },
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
