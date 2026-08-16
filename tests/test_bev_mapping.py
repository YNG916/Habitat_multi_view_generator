import math
import unittest

import numpy as np

from mri_dataset.bev import (
    BevMapping,
    habitat_orthographic_depth_to_metric,
    occupancy_from_pathfinder,
)
from mri_dataset.visualization import occupancy_visualization
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


    def test_orthographic_conversion_masks_invalid_values(self):
        actual = habitat_orthographic_depth_to_metric(
            np.array([0.0, np.nan], dtype=np.float32), 0.02, 10.0
        )
        self.assertTrue(np.isnan(actual).all())
        with self.assertRaises(ValueError):
            habitat_orthographic_depth_to_metric(np.ones(1), 1.0, 1.0)

    def test_orthographic_conversion_accepts_both_sides_of_singularity(self):
        near, far = 0.02, 10.0
        metric = np.array([1.0, 2.0, 5.0, 8.0], dtype=np.float64)
        depth_buffer = (metric - near) / (far - near)
        p22 = -2.0 / (far - near)
        p32 = -(far + near) / (far - near)
        coefficient_a = 0.5 * (p22 - 1.0)
        coefficient_b = 0.5 * p32
        pseudo_depth = coefficient_b / (depth_buffer + coefficient_a)

        self.assertGreater(pseudo_depth[2], 0.0)
        self.assertLess(pseudo_depth[3], 0.0)
        actual = habitat_orthographic_depth_to_metric(pseudo_depth, near, far)
        np.testing.assert_allclose(actual, metric, atol=1e-5)

        outside_metric = np.array([-1.0, 12.0])
        outside_buffer = (outside_metric - near) / (far - near)
        outside_raw = coefficient_b / (outside_buffer + coefficient_a)
        self.assertTrue(
            np.isnan(habitat_orthographic_depth_to_metric(outside_raw, near, far)).all()
        )


    def test_navmesh_occupancy_is_registered_into_larger_visual_bounds(self):
        class FakePathfinder:
            @staticmethod
            def get_bounds():
                return np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 2.0])

            @staticmethod
            def get_topdown_view(_meters_per_pixel, _floor_y, _eps):
                return np.array([[1, 0], [0, 1]], dtype=bool)

        mapping = BevMapping(-1.0, 3.0, -1.0, 3.0, 5, 5)
        occupancy = occupancy_from_pathfinder(
            FakePathfinder(), mapping, 0.0, FakePathfinder.get_bounds()
        )
        self.assertEqual(occupancy[1, 1], 1)
        self.assertEqual(occupancy[1, 2], 0)
        self.assertEqual(occupancy[2, 2], 1)
        self.assertEqual(int(occupancy[0].sum()), 0)
        self.assertEqual(int(occupancy[:, 0].sum()), 0)
        self.assertEqual(set(map(int,np.unique(occupancy))),{0,1})

    def test_island_map_minus_one_is_filtered_before_uint8_conversion(self):
        class FakePathfinder:
            eps=None
            @staticmethod
            def get_bounds():
                return np.array([0.,0.,0.]),np.array([2.,1.,2.])
            @classmethod
            def get_topdown_island_view(cls,_meters_per_pixel,_floor_y,eps):
                cls.eps=eps
                return np.array([[-1,4],[2,4]],dtype=np.int32)
        mapping=BevMapping(0.,1.999,0.,1.999,2,2)
        occupancy=occupancy_from_pathfinder(
            FakePathfinder(),mapping,0.,allowed_island_ids=[4],
            vertical_tolerance_m=.25,
        )
        np.testing.assert_array_equal(occupancy,[[0,1],[0,1]])
        self.assertEqual(FakePathfinder.eps,.25)
        self.assertEqual(occupancy.dtype,np.uint8)

    def test_occupancy_visualization_rejects_nonbinary_ground_truth(self):
        with self.assertRaisesRegex(ValueError,"must be binary"):
            occupancy_visualization(np.array([[0,255]],dtype=np.uint8))
        image=np.asarray(occupancy_visualization(
            np.array([[0,1],[1,1]],dtype=np.uint8),
            np.array([[0,0],[1,1]],dtype=np.uint8),
        ))
        np.testing.assert_array_equal(image[0,0],[0,0,0])
        np.testing.assert_array_equal(image[0,1],[210,210,210])
        np.testing.assert_array_equal(image[1,0],[35,190,85])


if __name__ == "__main__":
    unittest.main()
