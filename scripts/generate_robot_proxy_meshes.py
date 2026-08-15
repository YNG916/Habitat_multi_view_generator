#!/usr/bin/env python3
"""Generate lightweight, self-contained robot-vacuum OBJ/MTL assets."""

from __future__ import annotations

import json
import math
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "robot_proxies"
PALETTES = {
    "red": {
        "body": (0.42, 0.045, 0.035),
        "deck": (0.72, 0.08, 0.06),
    },
    "green": {
        "body": (0.035, 0.34, 0.08),
        "deck": (0.06, 0.58, 0.14),
    },
    "blue": {
        "body": (0.035, 0.12, 0.42),
        "deck": (0.05, 0.20, 0.70),
    },
}


class Mesh:
    def __init__(self):
        self.vertices = []
        self.faces = []

    def vertex(self, x, y, z):
        self.vertices.append((float(x), float(y), float(z)))
        return len(self.vertices)

    def face(self, material, indices):
        self.faces.append((material, tuple(indices)))

    def cylinder(
        self,
        radius,
        y_bottom,
        y_top,
        segments,
        material,
        center_x=0.0,
        center_z=0.0,
        top_material=None,
        bottom_material=None,
        caps=True,
    ):
        bottom = []
        top = []
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            x = center_x + radius * math.sin(angle)
            z = center_z - radius * math.cos(angle)
            bottom.append(self.vertex(x, y_bottom, z))
            top.append(self.vertex(x, y_top, z))
        for index in range(segments):
            nxt = (index + 1) % segments
            self.face(material, (bottom[index], bottom[nxt], top[nxt], top[index]))
        if caps:
            self.face(bottom_material or material, bottom)
            self.face(top_material or material, reversed(top))

    def box(self, x_min, x_max, y_min, y_max, z_min, z_max, material):
        vertices = [
            self.vertex(x_min, y_min, z_min),
            self.vertex(x_max, y_min, z_min),
            self.vertex(x_max, y_min, z_max),
            self.vertex(x_min, y_min, z_max),
            self.vertex(x_min, y_max, z_min),
            self.vertex(x_max, y_max, z_min),
            self.vertex(x_max, y_max, z_max),
            self.vertex(x_min, y_max, z_max),
        ]
        for face in (
            (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6),
            (3, 0, 4, 7), (0, 3, 2, 1), (4, 5, 6, 7),
        ):
            self.face(material, (vertices[i] for i in face))

    def rotated_plate(self, center_x, center_z, length, width, y_min, y_max, angle, material):
        forward = (math.cos(angle) * length * 0.5, math.sin(angle) * length * 0.5)
        lateral = (-math.sin(angle) * width * 0.5, math.cos(angle) * width * 0.5)
        corners = [
            (center_x - forward[0] - lateral[0], center_z - forward[1] - lateral[1]),
            (center_x + forward[0] - lateral[0], center_z + forward[1] - lateral[1]),
            (center_x + forward[0] + lateral[0], center_z + forward[1] + lateral[1]),
            (center_x - forward[0] + lateral[0], center_z - forward[1] + lateral[1]),
        ]
        bottom = [self.vertex(x, y_min, z) for x, z in corners]
        top = [self.vertex(x, y_max, z) for x, z in corners]
        for index in range(4):
            nxt = (index + 1) % 4
            self.face(material, (bottom[index], bottom[nxt], top[nxt], top[index]))
        self.face(material, bottom)
        self.face(material, reversed(top))

    def obj_text(self, material_file):
        lines = [f"mtllib {material_file}", "o low_poly_robot_vacuum"]
        lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in self.vertices)
        current_material = None
        for material, indices in self.faces:
            if material != current_material:
                lines.append(f"usemtl {material}")
                current_material = material
            lines.append("f " + " ".join(str(index) for index in indices))
        return "\n".join(lines) + "\n"


def build_mesh(camera_height_m=None):
    mesh = Mesh()
    mesh.cylinder(0.282, 0.035, 0.108, 32, "body", top_material="body", bottom_material="trim")
    mesh.cylinder(0.255, 0.108, 0.119, 32, "deck", top_material="deck")
    mesh.cylinder(0.292, 0.055, 0.101, 32, "trim", caps=False)
    mesh.cylinder(0.052, 0.119, 0.174, 20, "sensor", 0.065, 0.035, top_material="sensor")
    mesh.cylinder(0.018, 0.119, 0.128, 16, "status", -0.075, -0.020, top_material="status")

    if camera_height_m is not None:
        camera_height_m = float(camera_height_m)
        if camera_height_m < 0.30:
            raise ValueError("Camera mast height must clear the robot-vacuum body")
        # Thin telescoping mast and compact camera head. The optical center is
        # exactly camera_height_m, matching RobotState camera metadata.
        mesh.cylinder(
            0.018, 0.119, camera_height_m - 0.035, 12, "trim",
            center_x=0.0, center_z=0.035,
        )
        mesh.box(
            -0.052, 0.052,
            camera_height_m - 0.035, camera_height_m + 0.035,
            -0.012, 0.082,
            "sensor",
        )
        mesh.box(
            -0.018, 0.018,
            camera_height_m - 0.012, camera_height_m + 0.012,
            -0.020, -0.013,
            "status",
        )

    # Two drive wheels and small front/rear sensor windows.
    mesh.box(-0.298, -0.260, 0.012, 0.076, -0.090, 0.090, "wheel")
    mesh.box(0.260, 0.298, 0.012, 0.076, -0.090, 0.090, "wheel")
    mesh.box(-0.067, 0.067, 0.068, 0.101, -0.298, -0.291, "sensor")
    mesh.box(-0.040, 0.040, 0.045, 0.068, 0.284, 0.294, "status")

    # Three low-profile side-brush arms near the front-right corner.
    brush_center = (0.195, -0.176)
    for angle_deg in (15.0, 135.0, 255.0):
        mesh.rotated_plate(
            brush_center[0], brush_center[1], 0.135, 0.010,
            0.018, 0.024, math.radians(angle_deg), "brush"
        )
    return mesh


def material_text(body, deck):
    def rgb(values):
        return " ".join(f"{value:.4f}" for value in values)

    return f"""newmtl body
Ka {rgb(tuple(value * 0.12 for value in body))}
Kd {rgb(body)}
Ks 0.2200 0.2200 0.2200
Ns 52
illum 2

newmtl deck
Ka {rgb(tuple(value * 0.14 for value in deck))}
Kd {rgb(deck)}
Ks 0.4200 0.4200 0.4200
Ns 96
illum 2

newmtl trim
Ka 0.0060 0.0070 0.0080
Kd 0.0280 0.0320 0.0360
Ks 0.1200 0.1200 0.1200
Ns 28
illum 2

newmtl sensor
Ka 0.0100 0.0120 0.0150
Kd 0.0400 0.0550 0.0700
Ks 0.5200 0.5200 0.5200
Ns 128
illum 2

newmtl wheel
Ka 0.0040 0.0040 0.0040
Kd 0.0180 0.0180 0.0200
Ks 0.0400 0.0400 0.0400
Ns 12
illum 2

newmtl brush
Ka 0.0100 0.0100 0.0100
Kd 0.0750 0.0750 0.0800
Ks 0.0500 0.0500 0.0500
Ns 16
illum 2

newmtl status
Ka 0.0200 0.0700 0.0750
Kd 0.1200 0.7800 0.8500
Ks 0.6000 0.6000 0.6000
Ns 128
illum 2
"""


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    camera_heights = [None, 0.4, 0.6, 0.9, 1.2, 1.4]
    generated = 0
    for index, (name, palette) in enumerate(PALETTES.items(), start=1):
        material_name = f"robot_{name}.mtl"
        (ASSET_DIR / material_name).write_text(
            material_text(palette["body"], palette["deck"]), encoding="utf-8"
        )
        for camera_height_m in camera_heights:
            suffix = "" if camera_height_m is None else f"_h{round(100 * camera_height_m):03d}"
            stem = f"robot_{name}{suffix}"
            mesh = build_mesh(camera_height_m)
            (ASSET_DIR / f"{stem}.obj").write_text(
                mesh.obj_text(material_name), encoding="utf-8"
            )
            config = {
                "render_asset": f"{stem}.obj",
                "use_bounding_box_for_collision": True,
                "mass": 4.0,
                "COM": [0.0, 0.08, 0.0],
                "join_collision_meshes": True,
                "semantic_id": 1000 + index,
            }
            (ASSET_DIR / f"{stem}.object_config.json").write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8"
            )
            generated += 1
    print(f"Generated {generated} robot-vacuum proxies in {ASSET_DIR}")


if __name__ == "__main__":
    main()
