import os
import numpy as np
from PIL import Image

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis


DATASET = "data/replica_cad/replicaCAD.scene_dataset_config.json"
SCENE = "apt_1"
NAVMESH = "data/replica_cad/navmeshes/apt_1.navmesh"

OUT_DIR = "debug_single_camera"
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------
# 1. Simulator config
# --------------------------------------------------

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_dataset_config_file = DATASET
sim_cfg.scene_id = SCENE
sim_cfg.enable_physics = True
sim_cfg.gpu_device_id = 0


# --------------------------------------------------
# 2. RGB sensor
# --------------------------------------------------

rgb_spec = habitat_sim.CameraSensorSpec()
rgb_spec.uuid = "rgb"
rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
rgb_spec.resolution = [512, 512]

# Camera is 1 meter above agent base
rgb_spec.position = [0.0, 1.0, 0.0]

rgb_spec.hfov = 90.0
rgb_spec.near = 0.05
rgb_spec.far = 20.0


# --------------------------------------------------
# 3. Depth sensor
# --------------------------------------------------

depth_spec = habitat_sim.CameraSensorSpec()
depth_spec.uuid = "depth"
depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
depth_spec.resolution = [512, 512]
depth_spec.position = [0.0, 1.0, 0.0]
depth_spec.hfov = 90.0
depth_spec.near = 0.05
depth_spec.far = 20.0


# --------------------------------------------------
# 4. Agent
# --------------------------------------------------

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [
    rgb_spec,
    depth_spec,
]


cfg = habitat_sim.Configuration(
    sim_cfg,
    [agent_cfg],
)

sim = habitat_sim.Simulator(cfg)


# --------------------------------------------------
# 5. Load navmesh
# --------------------------------------------------

assert sim.pathfinder.load_nav_mesh(NAVMESH)
assert sim.pathfinder.is_loaded

print("Navmesh loaded.")


# --------------------------------------------------
# 6. Initialize agent
# --------------------------------------------------

agent = sim.initialize_agent(0)

base_position = np.array(
    sim.pathfinder.get_random_navigable_point(),
    dtype=np.float32,
)

yaw = np.random.uniform(-np.pi, np.pi)

state = habitat_sim.AgentState()

state.position = base_position

state.rotation = quat_from_angle_axis(
    yaw,
    np.array([0.0, 1.0, 0.0]),
)

agent.set_state(
    state,
    infer_sensor_states=True,
)

print("Base position:", base_position)
print("Yaw [rad]:", yaw)
print("Yaw [deg]:", np.degrees(yaw))


# --------------------------------------------------
# 7. Inspect actual state
# --------------------------------------------------

actual_state = agent.get_state()

print("Actual agent position:", actual_state.position)
print("Actual agent rotation:", actual_state.rotation)

for sensor_id, sensor_state in actual_state.sensor_states.items():
    print(
        f"{sensor_id}:",
        "position =", sensor_state.position,
        "rotation =", sensor_state.rotation,
    )


# --------------------------------------------------
# 8. Render
# --------------------------------------------------

obs = sim.get_sensor_observations()

rgb = np.asarray(obs["rgb"])
depth = np.asarray(obs["depth"], dtype=np.float32)

if rgb.shape[-1] == 4:
    rgb = rgb[..., :3]


# --------------------------------------------------
# 9. Save RGB
# --------------------------------------------------

Image.fromarray(rgb.astype(np.uint8)).save(
    os.path.join(OUT_DIR, "rgb.png")
)


# --------------------------------------------------
# 10. Save raw depth
# --------------------------------------------------

np.save(
    os.path.join(OUT_DIR, "depth.npy"),
    depth,
)


# --------------------------------------------------
# 11. Save depth visualization
# --------------------------------------------------

valid = np.isfinite(depth) & (depth > 0)

depth_vis = np.zeros_like(depth, dtype=np.uint8)

if valid.any():
    max_vis_depth = min(
        10.0,
        float(np.percentile(depth[valid], 99)),
    )

    normalized = np.clip(
        depth / max_vis_depth,
        0.0,
        1.0,
    )

    depth_vis = (
        normalized * 255.0
    ).astype(np.uint8)

Image.fromarray(depth_vis).save(
    os.path.join(OUT_DIR, "depth_vis.png")
)


# --------------------------------------------------
# 12. Statistics
# --------------------------------------------------

print("RGB shape:", rgb.shape)
print("Depth shape:", depth.shape)

if valid.any():
    print("Depth min:", float(depth[valid].min()))
    print("Depth max:", float(depth[valid].max()))
    print("Depth mean:", float(depth[valid].mean()))

print(f"Saved outputs to: {OUT_DIR}")

sim.close()