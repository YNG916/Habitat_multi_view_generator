import unittest

import numpy as np

from mri_dataset.bev import BevMapping
from mri_dataset.validation import instance_centroid_registration


class ValidationMetricTests(unittest.TestCase):
    def test_instance_centroid_registration_is_metric_and_resolution_aware(self):
        coarse = BevMapping(0.0, 10.0, 0.0, 10.0, 201, 201)
        fine = BevMapping(0.0, 10.0, 0.0, 10.0, 1601, 1601)
        coarse_result = instance_centroid_registration(
            np.array([[99, 104], [101, 106]]), 100, 100, coarse
        )
        fine_result = instance_centroid_registration(
            np.array([[799, 839], [801, 841]]), 800, 800, fine
        )
        self.assertAlmostEqual(coarse_result["error_m"], 0.25)
        self.assertAlmostEqual(fine_result["error_m"], 0.25)
        self.assertAlmostEqual(coarse_result["tolerance_m"], 0.15)
        self.assertAlmostEqual(fine_result["tolerance_m"], 0.02)
        self.assertGreater(
            coarse_result["error_m"], coarse_result["tolerance_m"]
        )
        self.assertGreater(fine_result["error_m"], fine_result["tolerance_m"])


if __name__ == "__main__":
    unittest.main()
