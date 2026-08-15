from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .coordinates import forward_from_quaternion, yaw_to_quaternion_xyzw


ROBOT_COLORS = [(230, 55, 55), (45, 190, 90), (50, 105, 235)]


def habitat_orthographic_depth_to_metric(
    depth: np.ndarray, near: float, far: float
) -> np.ndarray:
    """Linearize Habitat-Sim 0.3.3 orthographic DEPTH observations.

    Habitat does run its generic depth-unprojection pass before Python readback.
    For a pinhole projection that output is metric Z-depth. With the v0.3.3
    orthographic projection matrix, however, the same generic pass yields a
    reciprocal pseudo-depth. This inverse mapping is verified against real
    downward orthographic observations and Bullet rays in the integration test.
    """
    raw = np.asarray(depth, dtype=np.float64)
    if not 0.0 < near < far:
        raise ValueError("Orthographic near/far planes must satisfy 0 < near < far")
    p22 = -2.0 / (far - near)
    p32 = -(far + near) / (far - near)
    coefficient_a = 0.5 * (p22 - 1.0)
    coefficient_b = 0.5 * p32
    result = np.full(raw.shape, np.nan, dtype=np.float64)
    candidate = np.isfinite(raw) & (raw != 0.0)
    depth_buffer = np.full(raw.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        depth_buffer[candidate] = coefficient_b / raw[candidate] - coefficient_a
    metric = near + depth_buffer * (far - near)
    epsilon = 1e-6
    valid = (
        candidate
        & np.isfinite(depth_buffer)
        & (depth_buffer >= -epsilon)
        & (depth_buffer <= 1.0 + epsilon)
        & (metric >= near - epsilon)
        & (metric <= far + epsilon)
    )
    result[valid] = metric[valid]
    return result.astype(np.float32)


@dataclass(frozen=True)
class BevMapping:
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    width: int
    height: int

    @classmethod
    def from_bounds(
        cls, minimum: Iterable[float], maximum: Iterable[float], requested_mpp: float
    ) -> "BevMapping":
        low = np.asarray(minimum, dtype=np.float64)
        high = np.asarray(maximum, dtype=np.float64)
        x_extent = high[0] - low[0]
        z_extent = high[2] - low[2]
        width = max(2, int(math.ceil(x_extent / requested_mpp)) + 1)
        # Habitat orthographic projection preserves sensor aspect; derive Z
        # pixels from actual X/Z extents to keep RGB and BEV registered.
        height = max(2, int(round(width * z_extent / x_extent)))
        return cls(float(low[0]), float(high[0]), float(low[2]), float(high[2]), width, height)

    @property
    def meters_per_pixel_x(self) -> float:
        return (self.x_max - self.x_min) / (self.width - 1)

    @property
    def meters_per_pixel_z(self) -> float:
        return (self.z_max - self.z_min) / (self.height - 1)

    def world_to_bev(self, x: float, z: float) -> Tuple[float, float]:
        u = (float(x) - self.x_min) / (self.x_max - self.x_min) * (self.width - 1)
        v = (float(z) - self.z_min) / (self.z_max - self.z_min) * (self.height - 1)
        return u, v

    def bev_to_world(self, u: float, v: float) -> Tuple[float, float]:
        x = self.x_min + float(u) / (self.width - 1) * (self.x_max - self.x_min)
        z = self.z_min + float(v) / (self.height - 1) * (self.z_max - self.z_min)
        return x, z

    def metadata(self) -> dict:
        result = asdict(self)
        result["meters_per_pixel_x"] = self.meters_per_pixel_x
        result["meters_per_pixel_z"] = self.meters_per_pixel_z
        result["pixel_convention"] = "u right is +X; v down is +Z; pixel centers registered to bounds"
        return result


def occupancy_from_pathfinder(
    pathfinder,
    mapping: BevMapping,
    floor_y: float,
    navmesh_bounds=None,
    allowed_island_ids=None,
) -> np.ndarray:
    """Rasterize NavMesh in C++ and register it to the visual BEV.

    Habitat rows increase with world +Z and columns with +X. The normalized
    lookup also supports visual bounds that extend beyond the NavMesh.
    """
    if navmesh_bounds is None:
        navmesh_bounds = pathfinder.get_bounds()
    nav_low = np.asarray(navmesh_bounds[0], dtype=np.float64)
    nav_high = np.asarray(navmesh_bounds[1], dtype=np.float64)
    native_mpp = min(mapping.meters_per_pixel_x, mapping.meters_per_pixel_z)
    if allowed_island_ids is None:
        native = np.asarray(
            pathfinder.get_topdown_view(float(native_mpp), float(floor_y)),
            dtype=np.uint8,
        )
    else:
        island_views = [
            np.asarray(
                pathfinder.get_topdown_island_view(
                    float(native_mpp), float(floor_y), int(island_id)
                ),
                dtype=np.uint8,
            )
            for island_id in allowed_island_ids
        ]
        if not island_views:
            raise ValueError("Floor-local occupancy requires at least one island")
        native = np.maximum.reduce(island_views)
    occupancy = np.zeros((mapping.height, mapping.width), dtype=np.uint8)
    if native.ndim != 2 or 0 in native.shape:
        return occupancy

    xs = np.linspace(mapping.x_min, mapping.x_max, mapping.width)
    zs = np.linspace(mapping.z_min, mapping.z_max, mapping.height)
    x_extent = float(nav_high[0] - nav_low[0])
    z_extent = float(nav_high[2] - nav_low[2])
    cols = np.floor((xs - nav_low[0]) / x_extent * native.shape[1]).astype(np.int64)
    rows = np.floor((zs - nav_low[2]) / z_extent * native.shape[0]).astype(np.int64)
    valid_cols = (cols >= 0) & (cols < native.shape[1])
    valid_rows = (rows >= 0) & (rows < native.shape[0])
    occupancy[np.ix_(valid_rows, valid_cols)] = native[np.ix_(rows[valid_rows], cols[valid_cols])]
    return occupancy


def annotate_bev(clean_rgb: np.ndarray, mapping: BevMapping, robots, hfov_deg: float) -> Image.Image:
    image = Image.fromarray(np.asarray(clean_rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    wedge_length_m = min(2.0, 0.2 * max(mapping.x_max - mapping.x_min, mapping.z_max - mapping.z_min))
    for index, robot in enumerate(robots):
        color = ROBOT_COLORS[index % len(ROBOT_COLORS)]
        u, v = mapping.world_to_bev(robot.base_position_world[0], robot.base_position_world[2])
        forward = forward_from_quaternion(yaw_to_quaternion_xyzw(robot.yaw_rad))
        left_yaw = robot.yaw_rad + math.radians(hfov_deg / 2.0)
        right_yaw = robot.yaw_rad - math.radians(hfov_deg / 2.0)
        left = forward_from_quaternion(yaw_to_quaternion_xyzw(left_yaw))
        right = forward_from_quaternion(yaw_to_quaternion_xyzw(right_yaw))
        lu, lv = mapping.world_to_bev(
            robot.base_position_world[0] + wedge_length_m * left[0],
            robot.base_position_world[2] + wedge_length_m * left[2],
        )
        ru, rv = mapping.world_to_bev(
            robot.base_position_world[0] + wedge_length_m * right[0],
            robot.base_position_world[2] + wedge_length_m * right[2],
        )
        fu, fv = mapping.world_to_bev(
            robot.base_position_world[0] + 0.8 * forward[0],
            robot.base_position_world[2] + 0.8 * forward[2],
        )
        draw.polygon([(u, v), (lu, lv), (ru, rv)], fill=(*color, 55), outline=(*color, 210))
        draw.line([(u, v), (fu, fv)], fill=(*color, 255), width=3)
        radius = 6
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(*color, 255), outline=(0, 0, 0, 255), width=1)
        label = f"{robot.robot_id.replace('robot_', 'R')} h={robot.camera_height_m:.2f}m"
        draw.text((u + 8, v - 14), label, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255), font=font)
    return image


def compute_fov_overlap(mapping: BevMapping, robots, hfov_deg: float, max_range_m: float = 5.0) -> dict:
    yy, xx = np.indices((mapping.height, mapping.width))
    world_x = mapping.x_min + xx * mapping.meters_per_pixel_x
    world_z = mapping.z_min + yy * mapping.meters_per_pixel_z
    masks = []
    half_fov = math.radians(hfov_deg) / 2.0
    for robot in robots:
        dx = world_x - robot.base_position_world[0]
        dz = world_z - robot.base_position_world[2]
        distance = np.hypot(dx, dz)
        forward = forward_from_quaternion(yaw_to_quaternion_xyzw(robot.yaw_rad))
        cosine = (dx * forward[0] + dz * forward[2]) / np.maximum(distance, 1e-9)
        masks.append((distance <= max_range_m) & (cosine >= math.cos(half_fov)))
    common = np.logical_and.reduce(masks)
    union = np.logical_or.reduce(masks)
    common_iou = float(common.sum() / max(1, union.sum()))
    pairwise = {}
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            inter = np.logical_and(masks[i], masks[j]).sum()
            combined = np.logical_or(masks[i], masks[j]).sum()
            pairwise[f"robot_{i+1:02d}_robot_{j+1:02d}"] = float(inter / max(1, combined))
    if common_iou >= 0.12:
        difficulty = "easy"
    elif max(pairwise.values(), default=0.0) >= 0.08:
        difficulty = "medium"
    else:
        difficulty = "hard"
    return {
        "method": "metric_bev_fov_wedges",
        "category_definition": "geometric FOV overlap without occlusion",
        "common_iou": common_iou,
        "pairwise_iou": pairwise,
        "category": difficulty,
    }
