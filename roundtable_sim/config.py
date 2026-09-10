"""Scenario configuration: guests, layout dimensions, drink station.

Everything the round-table scenario needs to know about the *world* lives here so
that the scene builder, the controller and the renderer agree on one source of
truth.  Distances are metres, angles are radians unless the name says ``_deg``.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

PEPSI = "Pepsi"
DIET = "Diet Pepsi"


@dataclass
class Guest:
    name: str
    order: str  # PEPSI or DIET
    shirt: str  # rgb hex
    skin: str  # rgb hex
    hair: str  # rgb hex


DEFAULT_GUESTS: List[Guest] = [
    Guest("Alice", PEPSI, "#c0392b", "#f1c27d", "#3b2417"),
    Guest("Ben", DIET, "#2980b9", "#8d5524", "#1c1c1c"),
    Guest("Chloe", DIET, "#27ae60", "#ffdbac", "#b5651d"),
    Guest("Dev", PEPSI, "#8e44ad", "#c68642", "#0f0f0f"),
    Guest("Elena", PEPSI, "#f39c12", "#e0ac69", "#5a3825"),
    Guest("Felix", DIET, "#16a085", "#f1c27d", "#d6a55b"),
    Guest("Grace", PEPSI, "#d35400", "#8d5524", "#1c1c1c"),
    Guest("Hugo", DIET, "#2c3e50", "#ffdbac", "#4a4a4a"),
    Guest("Isla", PEPSI, "#e84393", "#e0ac69", "#7a1f1f"),
    Guest("Jamal", DIET, "#f1c40f", "#6b4423", "#0f0f0f"),
]


@dataclass
class Layout:
    # Round table
    table_radius: float = 1.5
    table_height: float = 0.74  # top surface
    table_thickness: float = 0.05
    # Guests / chairs
    n_guests: int = 10
    chair_radius: float = 1.95  # chair (and guest) centre distance from table centre
    seat_height: float = 0.45
    first_guest_angle: float = math.radians(288.0)  # guest 0, others go counter-clockwise
    # Coaster where the can is placed: to the guest's right on the table edge
    coaster_radius: float = 1.36
    coaster_angle_offset: float = math.radians(9.0)
    coaster_size: float = 0.05
    # Where the robot stands to talk to / serve a guest (in the gap to the guest's right)
    serve_radius: float = 1.62
    serve_angle_offset: float = math.radians(18.0)
    # "Ring road" the robot walks on around the chairs
    ring_radius: float = 2.75
    # Drink station (a bar-height side table) south of the round table
    station_angle: float = math.radians(270.0)
    station_center_dist: float = 3.35
    station_half_x: float = 0.75  # tangential half width
    station_half_y: float = 0.22  # radial half depth
    station_height: float = 0.85
    station_can_row_offset: float = 0.10  # can row: this far from the station centre towards the table
    robot_station_standoff: float = 0.42  # base -> can horizontal distance when grasping
    # Cans
    can_radius: float = 0.033
    can_half_height: float = 0.061
    can_mass: float = 0.38
    n_cans_per_type: int = 6
    # Robot
    pelvis_height: float = 0.74
    walk_speed: float = 0.55
    turn_speed: float = 0.7  # in-place turns (rad/s, average); peak is 1.5x with the smoothstep profile
    # Simulation
    timestep: float = 0.004
    record_fps: int = 30

    def guest_angle(self, i: int) -> float:
        return self.first_guest_angle + i * 2.0 * math.pi / self.n_guests

    def polar(self, r: float, ang: float, z: float = 0.0):
        return (r * math.cos(ang), r * math.sin(ang), z)

    def coaster_pos(self, i: int):
        ang = self.guest_angle(i) + self.coaster_angle_offset
        return self.polar(self.coaster_radius, ang, self.table_height)

    def serve_pose(self, i: int):
        """(x, y, yaw) where the robot stands to serve guest i, facing the coaster."""
        ang = self.guest_angle(i) + self.serve_angle_offset
        x, y, _ = self.polar(self.serve_radius, ang)
        cx, cy, _ = self.coaster_pos(i)
        yaw = math.atan2(cy - y, cx - x)
        return (x, y, yaw)

    def ask_pose(self, i: int):
        """Where the robot stands to ask guest i: same spot, but facing the guest."""
        ang = self.guest_angle(i) + self.serve_angle_offset
        x, y, _ = self.polar(self.serve_radius, ang)
        gx, gy, _ = self.polar(self.chair_radius, self.guest_angle(i))
        yaw = math.atan2(gy - y, gx - x)
        return (x, y, yaw)

    def station_center(self):
        return self.polar(self.station_center_dist, self.station_angle, self.station_height)

    def station_frame(self):
        """Unit vectors: tangential (along the station's long side) and inward (towards table)."""
        a = self.station_angle
        inward = (-math.cos(a), -math.sin(a))
        tang = (-math.sin(a), math.cos(a))
        return tang, inward

    def can_positions(self):
        """Initial can poses on the station: list of (x, y, z, kind)."""
        cx, cy, top = self.station_center()
        tang, inward = self.station_frame()
        # one row, Pepsi on the tangential-negative half, Diet on the positive half
        n = self.n_cans_per_type
        spacing = 0.11
        out = []
        for k in range(2 * n):
            kind = PEPSI if k < n else DIET
            offset = (k - (2 * n - 1) / 2.0) * spacing
            # rows sit slightly behind the front edge (front edge = centre + inward*half_y)
            bx = cx + tang[0] * offset + inward[0] * self.station_can_row_offset
            by = cy + tang[1] * offset + inward[1] * self.station_can_row_offset
            out.append((bx, by, top + self.can_half_height + 0.002, kind))
        return out

    def station_serve_pose(self, can_xy):
        """Robot base pose to grasp a can standing at can_xy with the right hand.

        The robot faces the station (away from the table). The right hand does a
        side grasp with the palm facing the robot's midline, so the can should sit
        a little to the right of the robot's centre line.
        """
        tang, inward = self.station_frame()
        yaw = math.atan2(-inward[1], -inward[0])  # facing away from the table
        right = (math.sin(yaw), -math.cos(yaw))
        lateral = 0.06  # can sits 6 cm to the right of the robot's centre line
        # The robot stands between the table and the station, i.e. on the inward side
        # of the can, and the can is offset to the robot's right.
        x = can_xy[0] + inward[0] * self.robot_station_standoff - right[0] * lateral
        y = can_xy[1] + inward[1] * self.robot_station_standoff - right[1] * lateral
        return (x, y, yaw)

    def robot_start_pose(self):
        cx, cy, _ = self.station_center()
        tang, inward = self.station_frame()
        x = cx + inward[0] * 0.65 + tang[0] * 1.1
        y = cy + inward[1] * 0.65 + tang[1] * 1.1
        yaw = math.atan2(inward[1], inward[0])
        return (x, y, yaw)


@dataclass
class ScenarioConfig:
    layout: Layout = field(default_factory=Layout)
    guests: List[Guest] = field(default_factory=lambda: list(DEFAULT_GUESTS))
    seed: int = 7

    @staticmethod
    def load(path: str | Path | None) -> "ScenarioConfig":
        cfg = ScenarioConfig()
        if path is None:
            return cfg
        data = json.loads(Path(path).read_text())
        if "layout" in data:
            for k, v in data["layout"].items():
                if not hasattr(cfg.layout, k):
                    raise KeyError(f"unknown layout key {k!r}")
                setattr(cfg.layout, k, v)
        if "guests" in data:
            cfg.guests = [Guest(**g) for g in data["guests"]]
            cfg.layout.n_guests = len(cfg.guests)
        cfg.seed = data.get("seed", cfg.seed)
        return cfg

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"layout": asdict(self.layout), "guests": [asdict(g) for g in self.guests], "seed": self.seed},
            indent=2))
