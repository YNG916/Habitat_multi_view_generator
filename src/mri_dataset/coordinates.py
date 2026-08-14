from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np


HABITAT_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def normalize_angle(yaw_rad: float) -> float:
    return float((yaw_rad + math.pi) % (2.0 * math.pi) - math.pi)


def yaw_to_quaternion_xyzw(yaw_rad: float) -> np.ndarray:
    half = 0.5 * float(yaw_rad)
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=np.float64)


def normalize_quaternion_xyzw(quaternion: Iterable[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if q.shape != (4,) or norm == 0.0 or not np.isfinite(norm):
        raise ValueError(f"Invalid xyzw quaternion: {q}")
    return q / norm


def quaternion_to_matrix(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(quaternion_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotate_vector(quaternion_xyzw: Iterable[float], vector: Iterable[float]) -> np.ndarray:
    return quaternion_to_matrix(quaternion_xyzw) @ np.asarray(vector, dtype=np.float64)


def forward_from_quaternion(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    forward = rotate_vector(quaternion_xyzw, [0.0, 0.0, -1.0])
    forward[np.abs(forward) < 1e-12] = 0.0
    return forward


def transform_matrix(position: Iterable[float], quaternion_xyzw: Iterable[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_matrix(quaternion_xyzw)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def invert_transform(transform: Iterable[Iterable[float]]) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("Expected a 4x4 rigid transform")
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ matrix[:3, 3]
    return inverse


def camera_transforms(
    camera_position_world: Iterable[float], quaternion_world_xyzw: Iterable[float]
) -> Dict[str, np.ndarray]:
    world_from_habitat = transform_matrix(camera_position_world, quaternion_world_xyzw)
    habitat_from_world = invert_transform(world_from_habitat)
    world_from_cv = world_from_habitat @ HABITAT_FROM_CV
    cv_from_world = invert_transform(world_from_cv)
    return {
        "T_world_from_camera_habitat": world_from_habitat,
        "T_camera_habitat_from_world": habitat_from_world,
        "T_world_from_camera_cv": world_from_cv,
        "T_camera_cv_from_world": cv_from_world,
    }


def camera_intrinsics(
    width: int,
    height: int,
    hfov_deg: float,
    near: float,
    far: float,
) -> Dict[str, object]:
    if width <= 0 or height <= 0 or not 0.0 < hfov_deg < 180.0:
        raise ValueError("Invalid camera dimensions or horizontal FOV")
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    # Habitat pixels are square; vertical FOV follows from the image aspect ratio.
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return {
        "width": int(width),
        "height": int(height),
        "hfov_deg": float(hfov_deg),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "near": float(near),
        "far": float(far),
        "K": matrix.tolist(),
    }


def backproject_depth(depth: np.ndarray, intrinsics: Dict[str, object]) -> np.ndarray:
    """Backproject Habitat pinhole Z-depth to OpenCV camera coordinates."""
    depth = np.asarray(depth, dtype=np.float64)
    rows, cols = np.indices(depth.shape)
    z = depth
    x = (cols - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (rows - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    return np.stack([x, y, z], axis=-1)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_world_points(
    points_world: np.ndarray,
    camera_cv_from_world: np.ndarray,
    intrinsics: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    camera = transform_points(camera_cv_from_world, points_world)
    z = camera[..., 2]
    pixels = np.empty(camera.shape[:-1] + (2,), dtype=np.float64)
    pixels[..., 0] = float(intrinsics["fx"]) * camera[..., 0] / z + float(intrinsics["cx"])
    pixels[..., 1] = float(intrinsics["fy"]) * camera[..., 1] / z + float(intrinsics["cy"])
    return pixels, z


def reproject_between_cameras(
    depth_source: np.ndarray,
    source_intrinsics: Dict[str, object],
    world_from_source_cv: np.ndarray,
    target_cv_from_world: np.ndarray,
    target_intrinsics: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    source_points = backproject_depth(depth_source, source_intrinsics)
    world_points = transform_points(world_from_source_cv, source_points)
    return project_world_points(world_points, target_cv_from_world, target_intrinsics)
