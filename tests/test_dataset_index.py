import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mri_dataset.collector import update_dataset_index
from mri_dataset.habitat_backend import HabitatBackend
from mri_dataset.serialization import write_json
from mri_dataset.world_state import RobotState, WorldState


class DatasetIndexTests(unittest.TestCase):
    def test_factual_and_intervention_states_are_split_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "splits").mkdir()
            write_json(root / "dataset.json", {"states": [], "interventions": []})

            factual = root / "scenes/apt_1/states/state_000001"
            derived = root / "scenes/apt_1/states/state_after_000001"
            factual.mkdir(parents=True)
            derived.mkdir(parents=True)
            write_json(
                factual / "state.json",
                {
                    "scene_id": "apt_1",
                    "state_origin": "factual",
                    "parent_state_id": None,
                },
            )
            write_json(
                derived / "state.json",
                {
                    "scene_id": "apt_1",
                    "state_origin": "intervention_derived",
                    "parent_state_id": "state_000001",
                },
            )
            edit_path = root / "interventions/apt_1/edit_000001.json"
            edit_path.parent.mkdir(parents=True)
            write_json(edit_path, {"scene_id": "apt_1"})

            update_dataset_index(root)

            with (root / "dataset.json").open(encoding="utf-8") as handle:
                dataset = json.load(handle)
            with (root / "splits/train.json").open(encoding="utf-8") as handle:
                train = json.load(handle)
            self.assertEqual(dataset["factual_states"], ["scenes/apt_1/states/state_000001"])
            self.assertEqual(
                dataset["intervention_derived_states"],
                ["scenes/apt_1/states/state_after_000001"],
            )
            self.assertEqual(train["states"], dataset["factual_states"])
            self.assertEqual(train["after_states"], dataset["intervention_derived_states"])
            self.assertEqual(
                train["interventions"], ["interventions/apt_1/edit_000001.json"]
            )


class VisibilityLabelTests(unittest.TestCase):
    def test_geometric_and_benchmark_visibility_are_distinct(self):
        backend = object.__new__(HabitatBackend)
        backend.config = SimpleNamespace(benchmark_visibility_min_pixels=20)
        backend.render_ids = {"robot_01": (7, 1001)}
        robot = RobotState.create(
            "robot_01", [0.0, 0.0, 0.0], 0.0, 0.8, 8, 8, 90, 0.05, 20
        )
        state = WorldState("0.1.0", "state", "scene", 0.0, 1, [robot])
        instance = np.zeros((5, 5), dtype=np.int32)
        instance.flat[:19] = 7

        backend._populate_visibility(
            state, {"robot_01": {"instance": instance}}
        )
        label = robot.visibility["robot_01"]
        self.assertTrue(label["geometrically_visible"])
        self.assertFalse(label["benchmark_visible"])
        self.assertEqual(label["visible_pixel_count"], 19)

        instance.flat[19] = 7
        backend._populate_visibility(
            state, {"robot_01": {"instance": instance}}
        )
        self.assertTrue(robot.visibility["robot_01"]["benchmark_visible"])


if __name__ == "__main__":
    unittest.main()

