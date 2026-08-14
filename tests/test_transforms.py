import unittest

import numpy as np

from mri_dataset.coordinates import camera_transforms, transform_points, yaw_to_quaternion_xyzw


class TransformTests(unittest.TestCase):
    def test_camera_inverse_pairs(self):
        transforms = camera_transforms([1.0, 2.0, 3.0], yaw_to_quaternion_xyzw(0.7))
        np.testing.assert_allclose(transforms["T_camera_habitat_from_world"] @ transforms["T_world_from_camera_habitat"], np.eye(4), atol=1e-12)
        np.testing.assert_allclose(transforms["T_camera_cv_from_world"] @ transforms["T_world_from_camera_cv"], np.eye(4), atol=1e-12)

    def test_zero_yaw_opencv_forward_maps_to_negative_world_z(self):
        transform = camera_transforms([0.0, 0.0, 0.0], yaw_to_quaternion_xyzw(0.0))["T_world_from_camera_cv"]
        point = transform_points(transform, np.array([[0.0, 0.0, 1.0]]))[0]
        np.testing.assert_allclose(point, [0.0, 0.0, -1.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
