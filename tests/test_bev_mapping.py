import math
import unittest

import numpy as np

from mri_dataset.bev import BevMapping, habitat_orthographic_depth_to_metric
from mri_dataset.coordinates import forward_from_quaternion, yaw_to_quaternion_xyzw


class BevMappingTests(unittest.TestCase):
    def setUp(self):
        self.mapping = BevMapping.from_bounds([-2.0, 0.0, -4.0], [5.0, 3.0, 8.0], 0.07)

    def test_round_trip(self):
        for x, z in [(-2, -4), (5, 8), (1.2, -0.7)]:
            u, v = self.mapping.world_to_bev(x, z)
            actual = self.mapping.bev_to_world(u, v)
            np.testing.assert_allclose(actual, [x, z], atol=1e-12)

    def test_bev_arrow_orientation(self):
        expected_pixel_directions = {
            0.0: [0.0, -1.0], math.pi / 2: [-1.0, 0.0],
            -math.pi / 2: [1.0, 0.0], math.pi: [0.0, 1.0],
        }
        center = self.mapping.world_to_bev(1.0, 1.0)
        for yaw, expected in expected_pixel_directions.items():
            forward = forward_from_quaternion(yaw_to_quaternion_xyzw(yaw))
            endpoint = self.mapping.world_to_bev(1.0 + forward[0], 1.0 + forward[2])
            direction = np.asarray(endpoint) - center
            direction /= np.linalg.norm(direction)
            np.testing.assert_allclose(direction, expected, atol=0.01)


    def test_habitat_033_orthographic_depth_conversion(self):
        near, far = 0.02, 10.0
        distances = np.array([0.1, 0.7, 1.0, 2.3], dtype=np.float64)
        depth_buffer = (distances - near) / (far - near)
        coefficient_a = 0.5 * (-2.0 / (far - near) - 1.0)
        coefficient_b = 0.5 * (-(far + near) / (far - near))
        habitat_values = coefficient_b / (depth_buffer + coefficient_a)
        np.testing.assert_allclose(
            habitat_orthographic_depth_to_metric(habitat_values, near, far), distances, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
