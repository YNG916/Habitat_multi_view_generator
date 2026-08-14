import unittest

import numpy as np

from mri_dataset.interventions import Intervention, apply_intervention
from mri_dataset.world_state import ObjectState, RobotState, WorldState


class ObjectInterventionTests(unittest.TestCase):
    def setUp(self):
        robot = RobotState.create("robot_01", [0.0, 0.2, 0.0], 0.0, 0.8, 64, 64, 90, 0.05, 20)
        obj = ObjectState(
            "object_001", "book", "frl_apartment_book_01.object_config.json",
            [0.5, 0.3, 0.5], [0.0, 0.0, 0.0, 1.0], bbox={
                "min_world": [0.4, 0.2, 0.4], "max_world": [0.6, 0.4, 0.6],
                "dimensions_m": [0.2, 0.2, 0.2],
            },
        )
        self.state = WorldState("0.1.0", "before", "test", 0.2, 1, [robot], [obj])

    def test_world_translation_is_exact_and_moves_bbox(self):
        edit = Intervention("object_translate", "object_001", {"reference_frame": "world", "displacement_m": [0.5, 0.0, -0.25]})
        after = apply_intervention(self.state, edit)
        np.testing.assert_allclose(after.object("object_001").position_world, [1.0, 0.3, 0.25])
        np.testing.assert_allclose(after.object("object_001").bbox["min_world"], [0.9, 0.2, 0.15])

    def test_relative_front_uses_xz_relation_and_preserves_support_y(self):
        edit = Intervention("object_place_relative", "object_001", {"reference_id": "robot_01", "relation": "front", "distance_m": 1.0})
        after = apply_intervention(self.state, edit)
        np.testing.assert_allclose(after.object("object_001").position_world, [0.0, 0.3, -1.0])
        self.assertAlmostEqual(after.object("object_001").bbox["min_world"][1], 0.2)

    def test_removal_does_not_mutate_before_state(self):
        after = apply_intervention(self.state, Intervention("object_remove", "object_001", {}))
        self.assertFalse(after.object("object_001").active)
        self.assertTrue(self.state.object("object_001").active)


if __name__ == "__main__":
    unittest.main()
