#!/usr/bin/env python3
"""Prepare a licensed, finished robot-vacuum mesh for Habitat-Sim.

This script does not procedurally assemble robot geometry. It only imports the
authored FBX, applies one rigid coordinate transform, normalizes scale/origin,
and writes three material-color variants that share identical topology.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import magnum.trade as trade


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "assets" / "robot_proxies"
SOURCE_DIR = ASSET_DIR / "source"
SOURCE_FBX = SOURCE_DIR / "robot_vacuum_original.fbx"
SOURCE_ALBEDO = SOURCE_DIR / "robot_vacuum_original_albedo.png"
# A 0.46 m footprint keeps the robot visually useful in room-scale RGB/BEV
# observations while remaining small enough for ordinary indoor passages.
TARGET_DIAMETER_M = 0.46
AGENT_COLORS = {
    "red": (235, 52, 50),
    "green": (45, 205, 92),
    "blue": (45, 105, 235),
}


def load_finished_mesh():
    manager = trade.ImporterManager()
    importer = manager.load_and_instantiate("AnySceneImporter")
    if importer is None:
        raise RuntimeError("Magnum AnySceneImporter is unavailable")
    importer.open_file(str(SOURCE_FBX))
    if not importer.is_opened or importer.mesh_count != 1:
        raise RuntimeError(
            f"Expected one mesh in {SOURCE_FBX}, got {importer.mesh_count}"
        )

    mesh = importer.mesh(0)
    if mesh.primitive.name != "TRIANGLES" or not mesh.is_indexed:
        raise RuntimeError("Finished robot source must be an indexed triangle mesh")

    positions = np.asarray(
        mesh.attribute(trade.MeshAttribute.POSITION), dtype=np.float64
    ).copy()
    normals = np.asarray(
        mesh.attribute(trade.MeshAttribute.NORMAL), dtype=np.float64
    ).copy()
    texcoords = np.asarray(
        mesh.attribute(trade.MeshAttribute.TEXTURE_COORDINATES), dtype=np.float64
    ).copy()
    indices = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3).copy()

    scene = importer.scene(importer.default_scene)
    transforms = np.asarray(
        scene.field(trade.SceneField.TRANSFORMATION), dtype=np.float64
    ).reshape(-1, 4, 4)
    if len(transforms) != 1:
        raise RuntimeError(f"Expected one scene transform, got {len(transforms)}")
    transform = transforms[0]
    homogeneous = np.concatenate(
        [positions, np.ones((len(positions), 1), dtype=np.float64)], axis=1
    )
    positions = (homogeneous @ transform.T)[:, :3]

    linear = transform[:3, :3]
    normals = normals @ np.linalg.inv(linear)
    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_lengths, 1e-12)

    horizontal_extent = max(
        float(np.ptp(positions[:, 0])),
        float(np.ptp(positions[:, 2])),
    )
    if horizontal_extent <= 0:
        raise RuntimeError("Finished robot mesh has an invalid horizontal extent")
    scale = TARGET_DIAMETER_M / horizontal_extent
    positions *= scale
    positions[:, 0] -= 0.5 * (
        float(positions[:, 0].min()) + float(positions[:, 0].max())
    )
    positions[:, 2] -= 0.5 * (
        float(positions[:, 2].min()) + float(positions[:, 2].max())
    )
    positions[:, 1] -= float(positions[:, 1].min())

    if abs(float(positions[:, 1].min())) > 1e-9:
        raise RuntimeError("Normalized robot mesh does not touch local Y=0")
    if float(positions[:, 1].max()) > 0.15:
        raise RuntimeError("Finished robot height is implausible after normalization")
    return positions, normals, texcoords, indices


def write_tinted_albedo(color_name: str, color_rgb) -> Path:
    source = np.asarray(Image.open(SOURCE_ALBEDO).convert("RGB"), dtype=np.float32)
    intensity = source.mean(axis=2, keepdims=True) / 255.0
    target = np.asarray(color_rgb, dtype=np.float32).reshape(1, 1, 3)
    tinted = intensity * target
    dark = intensity[:, :, 0] < 0.16
    tinted[dark] = source[dark]
    output = ASSET_DIR / f"robot_{color_name}_albedo.png"
    Image.fromarray(np.clip(np.rint(tinted), 0, 255).astype(np.uint8)).save(
        output, optimize=True
    )
    return output


def write_obj(path: Path, positions, normals, texcoords, indices) -> None:
    stem = path.stem
    lines = [
        "# Finished robot vacuum mesh; normalized from the attributed source FBX.",
        f"mtllib {stem}.mtl",
        "o finished_robot_vacuum",
    ]
    lines.extend(
        f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in positions
    )
    lines.extend(f"vt {u:.9f} {v:.9f}" for u, v in texcoords)
    lines.extend(f"vn {x:.9f} {y:.9f} {z:.9f}" for x, y, z in normals)
    lines.extend(["usemtl finished_robot_material", "s 1"])
    for triangle in indices:
        refs = [f"{int(index) + 1}/{int(index) + 1}/{int(index) + 1}" for index in triangle]
        lines.append("f " + " ".join(refs))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_material(color_name: str) -> None:
    stem = f"robot_{color_name}"
    (ASSET_DIR / f"{stem}.mtl").write_text(
        "\n".join([
            "newmtl finished_robot_material",
            "Ka 0.080000 0.080000 0.080000",
            "Kd 1.000000 1.000000 1.000000",
            "Ks 0.180000 0.180000 0.180000",
            "Ns 72.000000",
            "illum 2",
            f"map_Kd {stem}_albedo.png",
            "",
        ]),
        encoding="utf-8",
    )


def write_object_config(color_name: str, semantic_id: int) -> None:
    stem = f"robot_{color_name}"
    payload = {
        "render_asset": f"{stem}.obj",
        "collision_asset": f"{stem}.obj",
        "use_bounding_box_for_collision": False,
        "margin": 0.001,
        "mass": 4.0,
        "COM": [0.0, 0.0, 0.0],
        "join_collision_meshes": True,
        "semantic_id": semantic_id,
    }
    (ASSET_DIR / f"{stem}.object_config.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def remove_legacy_height_variants() -> None:
    for pattern in ("robot_*_h*.obj", "robot_*_h*.object_config.json"):
        for path in ASSET_DIR.glob(pattern):
            path.unlink()


def main() -> None:
    for source in (SOURCE_FBX, SOURCE_ALBEDO):
        if not source.is_file():
            raise FileNotFoundError(source)
    positions, normals, texcoords, indices = load_finished_mesh()
    remove_legacy_height_variants()
    for semantic_id, (color_name, color_rgb) in enumerate(
        AGENT_COLORS.items(), start=1001
    ):
        write_tinted_albedo(color_name, color_rgb)
        write_obj(
            ASSET_DIR / f"robot_{color_name}.obj",
            positions,
            normals,
            texcoords,
            indices,
        )
        write_material(color_name)
        write_object_config(color_name, semantic_id)

    dimensions = np.ptp(positions, axis=0)
    source_sha = hashlib.sha256(SOURCE_FBX.read_bytes()).hexdigest()
    print(
        "Prepared finished robot vacuum: "
        f"vertices={len(positions)}, triangles={len(indices)}, "
        f"dimensions_xyz_m={dimensions.tolist()}, min_y={positions[:, 1].min():.9f}, "
        f"source_sha256={source_sha}"
    )


if __name__ == "__main__":
    main()
