import math
import unittest

import numpy as np

from mri_dataset.coordinates import forward_from_quaternion, yaw_to_quaternion_xyzw


class CoordinateConventionTests(unittest.TestCase):
    def test_known_yaw_forward_vectors(self):
        expected = {
            0.0: [0.0, 0.0, -1.0],
            math.pi / 2: [-1.0, 0.0, 0.0],
            -math.pi / 2: [1.0, 0.0, 0.0],
            math.pi: [0.0, 0.0, 1.0],
        }
        for yaw, vector in expected.items():
            with self.subTest(yaw=yaw):
                np.testing.assert_allclose(forward_from_quaternion(yaw_to_quaternion_xyzw(yaw)), vector, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
