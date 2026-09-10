"""Round-table drink service scenario: the G1 asks each guest, fetches the can, serves it.

Runs the MuJoCo simulation headless and records the trajectory (all qpos at
``record_fps``) plus per-frame metadata (phase, dialog, camera focus) so that
rendering can happen afterwards, in parallel, with any camera setup.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

import mujoco
import numpy as np

from config import ScenarioConfig
from g1_kinematics import (ARM_ASK, ARM_STAND, ARM_WALK, NQ_ROBOT, NV_ROBOT, G1Puppet,
                           mat_from_quat, quat_from_mat, rot_z, smoothstep, wrap_angle)

Pose2D = Tuple[float, float, float]


# --------------------------------------------------------------------------------------
# base motion primitives (precomputed, so the gait can look ahead)
# --------------------------------------------------------------------------------------
class BaseTrajectory:
    """Sampled (x, y, yaw) trajectory starting at t0, held constant after t_end."""

    def __init__(self, t0: float, ts: np.ndarray, poses: np.ndarray):
        self.t0 = t0
        self.ts = ts  # relative times, increasing
        self.poses = poses  # N x 3 (x, y, yaw) with yaw unwrapped
        self.t_end = t0 + float(ts[-1])

    def pose(self, t: float) -> Pose2D:
        tr = min(max(t - self.t0, 0.0), float(self.ts[-1]))
        x = float(np.interp(tr, self.ts, self.poses[:, 0]))
        y = float(np.interp(tr, self.ts, self.poses[:, 1]))
        yaw = float(np.interp(tr, self.ts, self.poses[:, 2]))
        return x, y, wrap_angle(yaw)


def _chaikin(points: np.ndarray, passes: int = 3) -> np.ndarray:
    pts = points
    for _ in range(passes):
        if len(pts) < 3:
            break
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            out.append(0.75 * a + 0.25 * b)
            out.append(0.25 * a + 0.75 * b)
        out.append(pts[-1])
        pts = np.array(out)
    return pts


def _resample(points: np.ndarray, ds: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-6:
        return points[:1]
    n = max(int(total / ds) + 1, 2)
    ss = np.linspace(0.0, total, n)
    return np.stack([np.interp(ss, s, points[:, i]) for i in range(2)], axis=1)


def make_turn(t0: float, pose: Pose2D, yaw_target: float, turn_speed: float) -> BaseTrajectory:
    x, y, yaw = pose
    d = wrap_angle(yaw_target - yaw)
    T = max(abs(d) / turn_speed + 0.25, 0.3)
    ts = np.linspace(0.0, T, max(int(T * 100), 2))
    yaws = yaw + d * np.array([smoothstep(v / T) for v in ts])
    poses = np.stack([np.full_like(ts, x), np.full_like(ts, y), yaws], axis=1)
    return BaseTrajectory(t0, ts, poses)


def make_walk(t0: float, pose: Pose2D, waypoints: List[Tuple[float, float]], speed: float,
              accel: float = 0.7, max_yaw_rate: float = 0.8) -> BaseTrajectory:
    """Walk along a polyline (corners rounded); heading = path tangent.

    The speed profile is limited by acceleration *and* by curvature (v <= yaw_rate / kappa),
    so the robot slows down in corners instead of spinning while translating, which is
    what a footstep gait can actually follow.
    """
    pts = np.array([[pose[0], pose[1]]] + [list(w) for w in waypoints], float)
    keep = [0] + [i for i in range(1, len(pts)) if np.linalg.norm(pts[i] - pts[i - 1]) > 1e-4]
    pts = pts[keep]
    if len(pts) < 2:
        return make_turn(t0, pose, pose[2], 1.0)
    ds = 0.02
    path = _resample(_chaikin(pts, 4), ds)
    if len(path) < 3:
        return make_turn(t0, pose, pose[2], 1.0)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    tang = np.unwrap(np.arctan2(np.gradient(path[:, 1]), np.gradient(path[:, 0])))
    # curvature -> speed limit, smoothed a little so the limit is not spiky
    kappa = np.abs(np.gradient(tang, s, edge_order=1))
    kappa = np.convolve(kappa, np.ones(7) / 7.0, mode="same")
    v_lim = np.minimum(speed, max_yaw_rate / np.maximum(kappa, 1e-6))
    v_lim = np.maximum(v_lim, 0.12)
    # forward/backward passes for the acceleration limit (v^2 = v0^2 + 2 a ds)
    v = v_lim.copy()
    v[0] = 0.0
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * accel * (s[i] - s[i - 1])))
    v[-1] = 0.0
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * accel * (s[i + 1] - s[i])))
    # time along the path
    t_path = np.zeros_like(s)
    for i in range(1, len(s)):
        vm = max(0.5 * (v[i] + v[i - 1]), 0.05)
        t_path[i] = t_path[i - 1] + (s[i] - s[i - 1]) / vm
    T = float(t_path[-1])
    ts = np.linspace(0.0, T, max(int(T * 100), 2))
    ss = np.interp(ts, t_path, s)
    xs = np.interp(ss, s, path[:, 0])
    ys = np.interp(ss, s, path[:, 1])
    tang_s = np.interp(ss, s, tang)
    yaw = np.empty_like(ts)
    yaw[0] = pose[2]
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        err = wrap_angle(tang_s[i] - yaw[i - 1])
        yaw[i] = yaw[i - 1] + np.clip(err, -1.2 * dt, 1.2 * dt)
    poses = np.stack([xs, ys, yaw], axis=1)
    return BaseTrajectory(t0, ts, poses)


# --------------------------------------------------------------------------------------
# scenario
# --------------------------------------------------------------------------------------
@dataclass
class FrameMeta:
    t: float
    phase: str
    guest: int
    speaker: str
    text: str
    served: int
    focus: Tuple[float, float, float]
    carrying: str


class RoundTableScenario:
    def __init__(self, cfg: ScenarioConfig, model: mujoco.MjModel, max_guests: Optional[int] = None):
        self.cfg = cfg
        self.L = cfg.layout
        self.max_guests = len(cfg.guests) if max_guests is None else min(max_guests, len(cfg.guests))
        self.m = model
        self.d = mujoco.MjData(model)
        self.dt = model.opt.timestep
        self.t = 0.0
        self.puppet = G1Puppet(model, self.L.pelvis_height, self.L.robot_start_pose())
        self.traj: Optional[BaseTrajectory] = None
        self.hold_pose: Pose2D = self.L.robot_start_pose()
        # cans
        self.can_info = []  # (body name, kind, qpos adr, geom id)
        for k, (_, _, _, kind) in enumerate(self.L.can_positions()):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"can_{k}_free")
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"can_{k}_geom")
            self.can_info.append((f"can_{k}", kind, int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]), g))
        self.can_taken = [False] * len(self.can_info)
        self.attached: Optional[dict] = None
        # narrative state
        self.phase = "idle"
        self.guest = -1
        self.dialog = ("", "", -1.0)  # speaker, text, until
        self.served: List[int] = []
        self.focus = (0.0, 0.0, 0.8)
        self.events: List[dict] = []
        # recording
        self.record_dt = 1.0 / self.L.record_fps
        self.next_record = 0.0
        self.frames_q: List[np.ndarray] = []
        self.frames_meta: List[FrameMeta] = []
        # gravity compensation: the puppet must not be accelerated by gravity between overwrites
        pelvis = self.puppet.robot.pelvis
        for b in range(model.nbody):
            p = b
            while p != 0:
                if p == pelvis:
                    model.body_gravcomp[b] = 1.0
                    break
                p = model.body_parentid[p]
        mujoco.mj_forward(model, self.d)

    # -- helpers ------------------------------------------------------------------------
    def base_at(self, t: float) -> Pose2D:
        if self.traj is None:
            return self.hold_pose
        return self.traj.pose(t)

    def base_now(self) -> Pose2D:
        return self.base_at(self.t)

    def log(self, kind: str, **kw) -> None:
        self.events.append({"t": round(self.t, 3), "kind": kind, **kw})

    def say(self, speaker: str, text: str, duration: float) -> Generator:
        self.dialog = (speaker, text, self.t + duration)
        self.log("say", speaker=speaker, text=text)
        yield from self.wait(duration)

    def wait(self, duration: float) -> Generator:
        t_end = self.t + duration
        while self.t < t_end:
            yield

    def wait_until(self, cond: Callable[[], bool], timeout: float = 10.0) -> Generator:
        t_end = self.t + timeout
        while not cond() and self.t < t_end:
            yield

    # -- base motion --------------------------------------------------------------------
    def turn_to(self, yaw: float) -> Generator:
        self.traj = make_turn(self.t, self.base_now(), yaw, self.L.turn_speed)
        while self.t < self.traj.t_end:
            yield
        self.hold_pose = self.traj.pose(self.traj.t_end)
        self.traj = None

    def walk(self, waypoints: List[Tuple[float, float]], final_yaw: Optional[float] = None) -> Generator:
        pose = self.base_now()
        first = np.array(waypoints[0]) - np.array(pose[:2])
        if np.linalg.norm(first) > 0.05:
            heading = math.atan2(first[1], first[0])
            if abs(wrap_angle(heading - pose[2])) > math.radians(35):
                yield from self.turn_to(heading)
                pose = self.base_now()
        self.puppet.arm_swing = 0.22
        self.traj = make_walk(self.t, pose, waypoints, self.L.walk_speed)
        while self.t < self.traj.t_end:
            yield
        self.hold_pose = self.traj.pose(self.traj.t_end)
        self.traj = None
        self.puppet.arm_swing = 0.0
        if final_yaw is not None:
            yield from self.turn_to(final_yaw)
        # let the feet settle
        yield from self.wait_until(lambda: self.puppet.gait.settled(self.base_now()), timeout=2.0)

    def ring_route(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Waypoints from start to goal via the ring road around the chairs."""
        L = self.L
        R = L.ring_radius
        a0 = math.atan2(start_xy[1], start_xy[0])
        a1 = math.atan2(goal_xy[1], goal_xy[0])
        r0 = math.hypot(*start_xy)
        r1 = math.hypot(*goal_xy)
        da = wrap_angle(a1 - a0)
        pts: List[Tuple[float, float]] = []
        # Radial "spokes" only when we are well inside/outside the ring; when we are
        # already close to it, blend the radius along the first part of the arc instead
        # of taking a sharp corner.
        near0 = abs(r0 - R) < 0.25
        near1 = abs(r1 - R) < 0.25
        if not near0:
            pts.append((R * math.cos(a0), R * math.sin(a0)))
        n = max(int(abs(da) / math.radians(6)), 1)
        blend = min(math.radians(35), abs(da) * 0.5) if abs(da) > 1e-6 else 0.0
        for i in range(1, n + 1):
            a = a0 + da * i / n
            r = R
            if near0 and blend > 0:
                k = min(abs(a - a0) / blend, 1.0)
                r = r0 + (R - r0) * smoothstep(k)
            if near1 and blend > 0:
                k = min(abs(a1 - a) / blend, 1.0)
                r = r + (r1 - R) * (1.0 - smoothstep(k)) if near0 else r1 + (R - r1) * smoothstep(k)
            pts.append((r * math.cos(a), r * math.sin(a)))
        if not near1 or math.hypot(pts[-1][0] - goal_xy[0], pts[-1][1] - goal_xy[1]) > 0.02:
            pts.append(goal_xy)
        return [p for p in pts if math.hypot(p[0] - start_xy[0], p[1] - start_xy[1]) > 0.06] or [goal_xy]

    def go_to(self, goal: Pose2D) -> Generator:
        self.phase = "walk"
        pose = self.base_now()
        pts = self.ring_route((pose[0], pose[1]), (goal[0], goal[1]))
        yield from self.walk(pts, final_yaw=goal[2])

    # -- arm helpers ----------------------------------------------------------------------
    def arm_move(self, target_pos, target_rot, duration: float) -> Generator:
        arm = self.puppet.arms["right"]
        arm.move_to(self.puppet.q, np.asarray(target_pos, float), target_rot, duration, self.t)
        while not arm.cart_done(self.t):
            yield
        # an unreachable target shows up as a large residual; record it so check_run.py can fail
        if arm.last_ik_err > 0.02:
            self.log("ik_warn", phase=self.phase, err=round(float(arm.last_ik_err), 4),
                     target=[round(float(v), 3) for v in target_pos])

    def arm_hold(self) -> None:
        arm = self.puppet.arms["right"]
        arm.set_pose(self.puppet.q[self.puppet.robot.qadr(arm.joints)], rate=3.0)

    def hand_close(self) -> Generator:
        self.puppet.hands["right"].close_on_can()
        yield from self.wait_until(lambda: self.puppet.hands["right"].settled(self.puppet.q), timeout=1.5)

    def hand_open(self) -> Generator:
        self.puppet.hands["right"].open()
        yield from self.wait_until(lambda: self.puppet.hands["right"].settled(self.puppet.q), timeout=1.5)

    # -- cans -------------------------------------------------------------------------------
    def can_pos(self, k: int) -> np.ndarray:
        adr = self.can_info[k][2]
        return self.d.qpos[adr:adr + 3].copy()

    def pick_can_index(self, kind: str) -> int:
        """Nearest untaken can of the requested kind to the robot's current position."""
        bx, by, _ = self.base_now()
        best, best_d = -1, 1e9
        for k, (name, ck, adr, _, _) in enumerate(self.can_info):
            if ck != kind or self.can_taken[k]:
                continue
            p = self.can_pos(k)
            dd = math.hypot(p[0] - bx, p[1] - by)
            if dd < best_d:
                best, best_d = k, dd
        if best < 0:
            raise RuntimeError(f"out of {kind}")
        return best

    def attach_can(self, k: int) -> None:
        name, kind, adr, vadr, gid = self.can_info[k]
        hp, hR = self.puppet.hand_frame("right")
        cp = self.d.qpos[adr:adr + 3].copy()
        cq = self.d.qpos[adr + 3:adr + 7].copy()
        rel_pos = hR.T @ (cp - hp)
        rel_rot = hR.T @ mat_from_quat(cq)
        self.attached = {"k": k, "adr": adr, "vadr": vadr, "gid": gid, "rel_pos": rel_pos, "rel_rot": rel_rot}
        self.m.geom_contype[gid] = 0
        self.m.geom_conaffinity[gid] = 0
        self.can_taken[k] = True
        self.log("grasp", can=name, drink=kind)

    def detach_can(self) -> None:
        a = self.attached
        if a is None:
            return
        self.m.geom_contype[a["gid"]] = 1
        self.m.geom_conaffinity[a["gid"]] = 1
        self.d.qvel[a["vadr"]:a["vadr"] + 6] = 0.0
        self.log("release", can=self.can_info[a["k"]][0])
        self.attached = None

    def _sync_attached(self) -> None:
        a = self.attached
        if a is None:
            return
        hp, hR = self.puppet.hand_frame("right")
        adr = a["adr"]
        self.d.qpos[adr:adr + 3] = hp + hR @ a["rel_pos"]
        self.d.qpos[adr + 3:adr + 7] = quat_from_mat(hR @ a["rel_rot"])
        self.d.qvel[a["vadr"]:a["vadr"] + 6] = 0.0

    # -- the story ---------------------------------------------------------------------------
    def script(self) -> Generator:
        L, cfg = self.L, self.cfg
        for s in ("left", "right"):
            self.puppet.arms[s].set_pose(ARM_WALK[s], rate=1.5)
        self.phase = "intro"
        self.focus = (0.0, 0.0, 0.8)
        yield from self.say("robot", "Good evening everyone! I'll be taking your drink orders.", 3.0)

        for i, g in enumerate(cfg.guests[: self.max_guests]):
            self.guest = i
            gx, gy, _ = L.polar(L.chair_radius, L.guest_angle(i))
            # 1. walk to the guest and ask
            self.focus = (gx, gy, 1.0)
            yield from self.go_to(L.ask_pose(i))
            self.phase = "ask"
            self.puppet.arms["right"].set_pose(ARM_ASK["right"], rate=2.5)
            self.log("ask", guest=i, name=g.name)
            yield from self.say("robot", f"Hi {g.name}! Would you like a Pepsi or a Diet Pepsi?", 2.6)
            yield from self.say("guest", f"A {g.order}, please!", 2.2)
            yield from self.say("robot", f"One {g.order}, coming right up.", 1.8)
            self.puppet.arms["right"].set_pose(ARM_WALK["right"], rate=2.5)
            yield from self.wait(0.5)

            # 2. fetch the can from the station
            k = self.pick_can_index(g.order)
            cpos = self.can_pos(k)
            self.focus = tuple(cpos)
            yield from self.go_to(L.station_serve_pose((cpos[0], cpos[1])))
            self.phase = "grasp"
            bx, by, byaw = self.base_now()
            Rg = rot_z(byaw)
            fwd = np.array([math.cos(byaw), math.sin(byaw), 0.0])
            cpos = self.can_pos(k)
            self.focus = tuple(cpos)
            yield from self.arm_move(cpos - 0.12 * fwd + np.array([0, 0, 0.02]), Rg, 1.3)
            yield from self.arm_move(cpos, Rg, 0.8)
            yield from self.hand_close()
            self.attach_can(k)
            yield from self.wait(0.3)
            yield from self.arm_move(cpos + np.array([0, 0, 0.12]), Rg, 0.7)
            # carry pose: can held in front of the chest (elbow flexed), clear of the table top
            carry = np.array([bx, by, 0.0]) + Rg @ np.array([0.27, -0.15, 0.93])
            yield from self.arm_move(carry, Rg, 1.0)
            self.arm_hold()
            yield from self.wait(0.2)

            # 3. bring it to the guest's coaster
            cx, cy, cz = L.coaster_pos(i)
            self.focus = (cx, cy, cz + 0.06)
            yield from self.go_to(L.serve_pose(i))
            self.phase = "place"
            bx, by, byaw = self.base_now()
            Rg = rot_z(byaw)
            fwd = np.array([math.cos(byaw), math.sin(byaw), 0.0])
            target = np.array([cx, cy, cz + L.can_half_height + 0.006 + 0.003])
            yield from self.arm_move(target + np.array([0, 0, 0.10]), Rg, 1.3)
            yield from self.arm_move(target, Rg, 0.8)
            yield from self.hand_open()
            self.detach_can()
            self.served.append(i)
            self.log("served", guest=i, name=g.name, order=g.order)
            yield from self.wait(0.25)
            yield from self.arm_move(target + np.array([0, 0, 0.08]) - 0.10 * fwd, Rg, 0.8)
            yield from self.say("robot", f"Here you go, {g.name}. Enjoy!", 1.6)
            yield from self.say("guest", "Thank you!", 1.2)
            self.puppet.arms["right"].set_pose(ARM_WALK["right"], rate=2.0)
            yield from self.wait(0.6)

        # 4. wrap up: step back to the ring and take a bow
        self.guest = -1
        self.phase = "done"
        pose = self.base_now()
        a = math.atan2(pose[1], pose[0])
        yield from self.walk([(L.ring_radius * math.cos(a), L.ring_radius * math.sin(a))],
                             final_yaw=math.atan2(-pose[1], -pose[0]))
        self.focus = (0.0, 0.0, 0.8)
        for s in ("left", "right"):
            self.puppet.arms[s].set_pose(ARM_STAND[s], rate=1.5)
        yield from self.say("robot", "Everyone is served. Cheers!", 3.0)
        self.log("done")
        yield from self.wait(2.0)

    # -- main loop -------------------------------------------------------------------------------
    def step_once(self) -> None:
        """One control + physics tick after the script has updated its targets."""
        self.puppet.set_base(self.base_now())
        self.puppet.update(self.t, self.dt, self.base_at)
        self.d.qpos[:NQ_ROBOT] = self.puppet.q
        self.d.qvel[:NV_ROBOT] = 0.0
        self._sync_attached()
        mujoco.mj_step(self.m, self.d)
        self.t += self.dt
        if self.t + 1e-9 >= self.next_record:
            self._record()
            self.next_record += self.record_dt

    def run(self, verbose: bool = True) -> None:
        gen = self.script()
        t_wall = time.time()
        done = False
        last_print = 0.0
        while not done:
            try:
                next(gen)
            except StopIteration:
                done = True
            self.step_once()
            if verbose and self.t - last_print > 20.0:
                last_print = self.t
                print(f"[sim] t={self.t:6.1f}s phase={self.phase:6s} guest={self.guest:2d} served={len(self.served)} "
                      f"wall={time.time() - t_wall:5.1f}s", flush=True)
        if verbose:
            print(f"[sim] finished: {self.t:.1f}s of simulation, {len(self.frames_q)} frames, "
                  f"{time.time() - t_wall:.1f}s wall", flush=True)

    def _record(self) -> None:
        speaker, text, until = self.dialog
        if self.t > until:
            speaker, text = "", ""
        carrying = ""
        if self.attached is not None:
            carrying = self.can_info[self.attached["k"]][1]
        self.frames_q.append(self.d.qpos.copy())
        self.frames_meta.append(FrameMeta(round(self.t, 4), self.phase, self.guest, speaker, text,
                                          len(self.served), tuple(float(v) for v in self.focus), carrying))

    def save(self, out_dir: Path) -> Tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        traj_path = out_dir / "trajectory.npz"
        meta_path = out_dir / "meta.json"
        np.savez_compressed(traj_path, qpos=np.array(self.frames_q, dtype=np.float32),
                            t=np.array([f.t for f in self.frames_meta], dtype=np.float64))
        meta = {
            "fps": self.L.record_fps,
            "frames": [f.__dict__ for f in self.frames_meta],
            "events": self.events,
            "guests": [{"name": g.name, "order": g.order, "shirt": g.shirt} for g in self.cfg.guests[: self.max_guests]],
            "layout": {k: v for k, v in self.L.__dict__.items()},
        }
        meta_path.write_text(json.dumps(meta))
        return traj_path, meta_path
