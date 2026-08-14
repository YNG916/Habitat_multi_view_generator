import os
import json
import numpy as np
from PIL import Image

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis


DATASET = "data/replica_cad/replicaCAD.scene_dataset_config.json"
SCENE = "apt_1"
NAVMESH = "data/replica_cad/navmeshes/apt_1.navmesh"

OUT_DIR = "debug_level1_three_robots"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_ROBOTS = 3
WIDTH = 512
HEIGHT = 512
HFOV = 90.0


# ============================================================
# Sensor
# ============================================================

def make_sensor(uuid, sensor_type):
    spec = habitat_sim.CameraSensorSpec()

    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    spec.resolution = [HEIGHT, WIDTH]

    # IMPORTANT:
    # Sensor is placed at agent origin.
    # We directly place agent at camera world position later.
    spec.position = [0.0, 0.0, 0.0]

    spec.orientation = [0.0, 0.0, 0.0]

    spec.hfov = HFOV
    spec.near = 0.05
    spec.far = 20.0

    return spec


def make_agent_config(robot_idx):
    rgb = make_sensor(
        f"robot_{robot_idx:02d}_rgb",
        habitat_sim.SensorType.COLOR,
    )

    depth = make_sensor(
        f"robot_{robot_idx:02d}_depth",
        habitat_sim.SensorType.DEPTH,
    )

    cfg = habitat_sim.agent.AgentConfiguration()

    cfg.sensor_specifications = [
        rgb,
        depth,
    ]

    return cfg


# ============================================================
# Simulator
# ============================================================

sim_cfg = habitat_sim.SimulatorConfiguration()

sim_cfg.scene_dataset_config_file = DATASET
sim_cfg.scene_id = SCENE
sim_cfg.enable_physics = True
sim_cfg.gpu_device_id = 0

agent_cfgs = [
    make_agent_config(i + 1)
    for i in range(NUM_ROBOTS)
]

cfg = habitat_sim.Configuration(
    sim_cfg,
    agent_cfgs,
)

sim = habitat_sim.Simulator(cfg)


# ============================================================
# NavMesh
# ============================================================

assert sim.pathfinder.load_nav_mesh(NAVMESH)
assert sim.pathfinder.is_loaded

print("NavMesh loaded.")


# ============================================================
# Sampling
# ============================================================

def sample_good_position(
    sim,
    min_obstacle_dist=0.6,
    max_tries=200,
):
    for _ in range(max_tries):
        p = sim.pathfinder.get_random_navigable_point()

        if not sim.pathfinder.is_navigable(p):
            continue

        dist = sim.pathfinder.distance_to_closest_obstacle(
            p,
            max_search_radius=2.0,
        )

        if dist >= min_obstacle_dist:
            return np.array(p, dtype=np.float32)

    raise RuntimeError(
        "Could not sample a position sufficiently far from obstacles."
    )


def sample_three_positions(
    sim,
    min_robot_dist=1.0,
):
    positions = []

    for robot_id in range(NUM_ROBOTS):
        for _ in range(200):
            p = sample_good_position(sim)

            valid = True

            for old_p in positions:
                xz_dist = np.linalg.norm(
                    p[[0, 2]] - old_p[[0, 2]]
                )

                if xz_dist < min_robot_dist:
                    valid = False
                    break

            if valid:
                positions.append(p)
                break
        else:
            raise RuntimeError(
                f"Failed to sample position for Robot {robot_id + 1}"
            )

    return positions


positions = sample_three_positions(sim)


# ============================================================
# Robot configurations
# ============================================================

rng = np.random.default_rng(123)

camera_heights = [
    0.60,
    0.90,
    1.20,
]

yaws = [
    0.0,
    np.pi / 2.0,
    -np.pi / 2.0,
]

robots = []


# ============================================================
# Set camera poses
# ============================================================

for i in range(NUM_ROBOTS):

    robot_id = i + 1

    base_position = positions[i]

    camera_position = base_position.copy()
    camera_position[1] += camera_heights[i]

    yaw = yaws[i]

    q = quat_from_angle_axis(
        yaw,
        np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        ),
    )

    state = habitat_sim.AgentState()

    # Here agent == camera rig
    state.position = camera_position
    state.rotation = q

    agent = sim.get_agent(i)

    agent.set_state(
        state,
        infer_sensor_states=True,
    )

    actual = agent.get_state()

    print()
    print(f"Robot {robot_id}")
    print("  base position:", base_position)
    print("  camera position:", actual.position)
    print("  yaw rad:", yaw)
    print("  yaw deg:", np.degrees(yaw))

    robots.append(
        {
            "robot_id": f"robot_{robot_id:02d}",

            "base_position_world": [
                float(x)
                for x in base_position
            ],

            "camera_position_world": [
                float(x)
                for x in actual.position
            ],

            "camera_height_m":
                float(camera_heights[i]),

            "yaw_rad":
                float(yaw),

            "yaw_deg":
                float(np.degrees(yaw)),
        }
    )


# ============================================================
# Render all robot agents
# ============================================================

obs = sim.get_sensor_observations(
    agent_ids=[0, 1, 2]
)

for i in range(NUM_ROBOTS):

    robot_id = i + 1

    rgb_key = f"robot_{robot_id:02d}_rgb"
    depth_key = f"robot_{robot_id:02d}_depth"

    robot_obs = obs[i]

    rgb = np.asarray(
        robot_obs[rgb_key]
    )

    depth = np.asarray(
        robot_obs[depth_key],
        dtype=np.float32,
    )

    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    # RGB
    Image.fromarray(
        rgb.astype(np.uint8)
    ).save(
        os.path.join(
            OUT_DIR,
            f"robot_{robot_id:02d}_rgb.png",
        )
    )

    # raw depth
    np.save(
        os.path.join(
            OUT_DIR,
            f"robot_{robot_id:02d}_depth.npy",
        ),
        depth,
    )

    # depth visualization
    valid = (
        np.isfinite(depth)
        & (depth > 0)
    )

    depth_vis = np.zeros(
        depth.shape,
        dtype=np.uint8,
    )

    if valid.any():

        max_vis_depth = min(
            10.0,
            float(
                np.percentile(
                    depth[valid],
                    99
                )
            ),
        )

        depth_normalized = np.clip(
            depth / max_vis_depth,
            0.0,
            1.0,
        )

        depth_vis = (
            depth_normalized * 255
        ).astype(np.uint8)

    Image.fromarray(
        depth_vis
    ).save(
        os.path.join(
            OUT_DIR,
            f"robot_{robot_id:02d}_depth_vis.png",
        )
    )


# ============================================================
# Save metadata
# ============================================================

metadata = {
    "scene_id": SCENE,

    "coordinate_convention": {
        "world_up": "+Y",
        "camera_forward": "-Z",
    },

    "robots": robots,
}

with open(
    os.path.join(
        OUT_DIR,
        "state.json"
    ),
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


print()
print("Saved outputs to:", OUT_DIR)

sim.close()