import math
import unittest

import numpy as np

from mri_dataset.interventions import Intervention, apply_intervention
from mri_dataset.world_state import RobotState, WorldState


def robot(robot_id, position, yaw):
    return RobotState.create(robot_id, position, yaw, 0.8, 64, 64, 90, 0.05, 20)


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.state = WorldState("0.1.0", "before", "test", 0.25, 1, [
            robot("robot_01", [0, 0.25, 0], 0),
            robot("robot_02", [2, 0.25, 2], math.pi / 2),
            robot("robot_03", [-2, 0.25, 2], 0),
        ])

    def test_exact_local_forward_translation_and_camera_sync(self):
        edit = Intervention("robot_translate", "robot_02", {"reference_frame": "target_local", "forward_m": 1.0})
        after = apply_intervention(self.state, edit)
        np.testing.assert_allclose(after.robot("robot_02").base_position_world, [1, 0.25, 2], atol=1e-12)
        np.testing.assert_allclose(after.robot("robot_02").camera.position_world, [1, 1.05, 2], atol=1e-12)
        np.testing.assert_allclose(self.state.robot("robot_02").base_position_world, [2, 0.25, 2], atol=1e-12)

    def test_rotation_does_not_translate(self):
        after = apply_intervention(self.state, Intervention("robot_rotate", "robot_02", {"delta_yaw_rad": math.pi / 3}))
        np.testing.assert_allclose(after.robot("robot_02").base_position_world, [2, 0.25, 2])
        self.assertAlmostEqual(after.robot("robot_02").yaw_rad, 5 * math.pi / 6)


if __name__ == "__main__":
    unittest.main()
