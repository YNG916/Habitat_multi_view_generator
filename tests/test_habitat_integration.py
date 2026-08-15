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

from mri_dataset.interventions import Intervention, apply_intervention
from mri_dataset.level2 import validate_object_edit, validate_robot_edit

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
            self.backend, self.config, "integration_collision", 124,
            deterministic_debug=True,
        )
        obj = state.objects[0]
        robot = state.robots[0]
        obj.position_world[0] = robot.base_position_world[0]
        obj.position_world[2] = robot.base_position_world[2]
        self.backend.support_object_on_floor(obj, state.floor_y)
        report = self.backend.object_collision_report(state, obj.instance_id)
        self.assertFalse(report["collision_free"], report)
        self.assertTrue(report["rejected_contacts"], report)

        robot_report = self.backend.entity_collision_report(state, robot.robot_id)
        self.assertFalse(robot_report["collision_free"], robot_report)
        current_object_id = self.backend.render_ids[obj.instance_id][0]
        self.assertTrue(
            any(
                item["other_object_id"] == current_object_id
                for item in robot_report["rejected_contacts"]
            ),
            robot_report,
        )

    def test_robot_translate_validates_path_floor_and_bullet_target(self):
        state = make_world_state(
            self.backend, self.config, "integration_robot_move", 155
        )
        edit = Intervention(
            "robot_translate",
            "robot_02",
            {"reference_frame": "target_local", "forward_m": 1.0},
        )
        after = apply_intervention(state, edit)
        validate_robot_edit(self.backend, state, after, edit)
        old = np.asarray(state.robot("robot_02").base_position_world)
        new = np.asarray(after.robot("robot_02").base_position_world)
        self.assertAlmostEqual(float(np.linalg.norm((new - old)[[0, 2]])), 1.0)
        nav_target = self.backend.sim.pathfinder.snap_point(new)
        self.assertAlmostEqual(
            float(new[1]), self.backend.floor_surface_y(nav_target), places=5
        )

    def test_object_translate_resolves_target_floor_support(self):
        state = make_world_state(
            self.backend, self.config, "integration_object_move", 125
        )
        target_id = state.objects[0].instance_id
        edit = Intervention(
            "object_translate",
            target_id,
            {"reference_frame": "world", "displacement_m": [0.5, 0.0, 0.0]},
        )
        after = apply_intervention(state, edit)
        validate_object_edit(self.backend, after, target_id)
        before_xz = np.asarray(state.object(target_id).position_world)[[0, 2]]
        after_xz = np.asarray(after.object(target_id).position_world)[[0, 2]]
        np.testing.assert_allclose(after_xz - before_xz, [0.5, 0.0], atol=1e-6)
        nav_target = self.backend.sim.pathfinder.snap_point(
            after.object(target_id).position_world
        )
        physical_floor_y = self.backend.floor_surface_y(nav_target)
        self.assertAlmostEqual(
            after.object(target_id).bbox["min_world"][1],
            physical_floor_y,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
