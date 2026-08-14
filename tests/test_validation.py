import unittest

import numpy as np

from mri_dataset.bev import BevMapping
from mri_dataset.validation import nearest_instance_distance_m


class ValidationMetricTests(unittest.TestCase):
    def test_instance_registration_distance_is_resolution_independent(self):
        coarse = BevMapping(0.0, 10.0, 0.0, 10.0, 201, 201)
        fine = BevMapping(0.0, 10.0, 0.0, 10.0, 1601, 1601)
        coarse_distance = nearest_instance_distance_m(
            np.array([[100, 105]]), 100, 100, coarse
        )
        fine_distance = nearest_instance_distance_m(
            np.array([[800, 840]]), 800, 800, fine
        )
        self.assertAlmostEqual(coarse_distance, 0.25)
        self.assertAlmostEqual(fine_distance, 0.25)


if __name__ == "__main__":
    unittest.main()
