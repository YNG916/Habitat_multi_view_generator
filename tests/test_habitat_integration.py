import unittest
from pathlib import Path

import numpy as np

try:
    import habitat_sim  # noqa: F401
    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False

from mri_dataset.collector import make_world_state
from mri_dataset.config import REPO_ROOT, load_config
from mri_dataset.habitat_backend import HabitatBackend


ASSETS_AVAILABLE = (
    (REPO_ROOT / "data/replica_cad/replicaCAD.scene_dataset_config.json").exists()
    and (REPO_ROOT / "data/replica_cad/navmeshes/apt_1.navmesh").exists()
)


@unittest.skipUnless(HABITAT_AVAILABLE and ASSETS_AVAILABLE, "Habitat/ReplicaCAD unavailable")
class HabitatCorrectnessIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            str(REPO_ROOT / "configs/collector.json"),
            width=96,
            height=96,
            bev_meters_per_pixel=0.08,
            height_validation_samples=32,
        )
        cls.backend = HabitatBackend(cls.config, "apt_1")

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()

    def test_real_orthographic_depth_height_and_object_ids(self):
        state = make_world_state(
            self.backend, self.config, "integration_depth", 123, deterministic_debug=True
        )
        outputs = self.backend.render(state)
        valid = np.isfinite(outputs["bev_metric_depth"])
        self.assertGreater(int(valid.sum()), 0)
        # v0.3.3's generic unprojection is not linear metric depth for an
        # orthographic projection; the integration ray check below is the
        # non-circular correctness oracle.
        self.assertGreater(
            float(np.max(np.abs(
                outputs["bev_metric_depth"][valid] - outputs["bev_depth"][valid]
            ))),
            0.1,
        )

        validation = self.backend.validate_height_rays(
            state, outputs["height"], samples=32, seed=123
        )
        self.assertTrue(validation["available"], validation)
        self.assertLessEqual(
            validation["max_abs_error_m"],
            self.config.height_validation_max_error_m,
            validation,
        )

        instance = outputs["bev_instance"]
        self.assertIsNotNone(instance)
        visible_robot_ids = [
            outputs["entity_object_ids"][robot.robot_id]
            for robot in state.robots
            if np.any(instance == outputs["entity_object_ids"][robot.robot_id])
        ]
        self.assertTrue(visible_robot_ids, "BEV OBJECT_ID sensor saw no robot proxy")

    def test_bullet_rejects_controlled_object_overlapping_robot(self):
        state = make_world_state(
            self.backend, self.config, "integration_collision", 124
        )
        obj = state.objects[0]
        robot = state.robots[0]
        obj.position_world[0] = robot.base_position_world[0]
        obj.position_world[2] = robot.base_position_world[2]
        self.backend.support_object_on_floor(obj, state.floor_y)
        report = self.backend.object_collision_report(state, obj.instance_id)
        self.assertFalse(report["collision_free"], report)
        self.assertTrue(report["rejected_contacts"], report)


if __name__ == "__main__":
    unittest.main()
