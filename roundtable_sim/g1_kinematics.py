"""Kinematic control of the Unitree G1 in MuJoCo: IK, footstep gait, hand poses.

The robot is *puppeteered*: every tick the controller writes the full joint
configuration (floating base + 43 joints) and MuJoCo only simulates the cans.
That trades physical locomotion realism for a deterministic, CPU-friendly demo
that never falls over.  The pieces are deliberately independent so the base
trajectory could later be produced by an RL policy instead:

* :class:`G1Model`      - joint/body bookkeeping and forward kinematics on a scratch MjData
* :func:`solve_ik`      - damped-least-squares IK with joint weights and a rest-pose nullspace
* :class:`FootstepGait` - plants/swings feet so they follow a moving base without sliding
* :class:`ArmController`- joint-space poses or Cartesian (IK) targets, smoothly blended
* :class:`HandController`- open/closed finger poses for the BrainCo Revo2 hands
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np

SIDES = ("left", "right")
LEG_JOINTS = {s: [f"{s}_hip_pitch_joint", f"{s}_hip_roll_joint", f"{s}_hip_yaw_joint",
                  f"{s}_knee_joint", f"{s}_ankle_pitch_joint", f"{s}_ankle_roll_joint"] for s in SIDES}
ARM_JOINTS = {s: [f"{s}_shoulder_pitch_joint", f"{s}_shoulder_roll_joint", f"{s}_shoulder_yaw_joint",
                  f"{s}_elbow_joint", f"{s}_wrist_roll_joint", f"{s}_wrist_pitch_joint", f"{s}_wrist_yaw_joint"]
              for s in SIDES}
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
# BrainCo Revo2: 3 thumb joints + proximal/distal on index, middle, ring, pinky (11 per hand;
# the real hand drives them with 6 motors, the distal joints being coupled to the proximal ones)
HAND_FINGERS = ("index", "middle", "ring", "pinky")
HAND_JOINTS = {s: [f"{s}_thumb_metacarpal_joint", f"{s}_thumb_proximal_joint", f"{s}_thumb_distal_joint"]
               + [f"{s}_{f}_{seg}_joint" for f in HAND_FINGERS for seg in ("proximal", "distal")] for s in SIDES}
FOOT_BODY = {s: f"{s}_ankle_roll_link" for s in SIDES}
HAND_BODY = {s: f"{s}_wrist_yaw_link" for s in SIDES}

# Menagerie "stand" keyframe, arms only (shoulder pitch/roll/yaw, elbow, wrist r/p/y)
ARM_STAND = {"left": [0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0], "right": [0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0]}
# relaxed arms for walking
ARM_WALK = {"left": [0.25, 0.18, 0.0, 0.65, 0.0, 0.0, 0.0], "right": [0.25, -0.18, 0.0, 0.65, 0.0, 0.0, 0.0]}
# "which one would you like?" presenting gesture (right arm), palm up-ish
ARM_ASK = {"right": [-0.55, -0.35, 0.35, 1.25, 0.9, 0.2, 0.0]}
# "here you go" after setting a can down: upper arm vertical, forearm raised ~45 deg, open hand
# beside the chest (an open-palm presenting gesture that also keeps the hand above the table)
ARM_TUCK = {"right": [0.1, -0.35, 0.0, -0.6, 0.0, 0.0, 0.0]}
# hand joint order: thumb metacarpal (opposition), thumb proximal, thumb distal, then
# proximal/distal for index, middle, ring, pinky.  All joints close towards the palm for
# positive values on both hands (the left model is mirrored), so the poses are side-agnostic.
HAND_OPEN = {s: [0.0] * 11 for s in SIDES}
# Power grasp around a 66 mm can lying against the palm (tuned numerically on the model):
# the four fingers curl over the can's far side (pads ~2 mm off the surface, knuckles clear),
# the fully opposed thumb presses the near-inner side, so pads and thumb face each other.
# The Revo2's fingers are about as long as the can is wide, so this is the same partial wrap
# the real hand does on a 330 ml can, not a fully enclosed fist.
_CLOSED_CAN = [1.48, 0.00, 0.20,  # thumb: metacarpal (opposition), proximal, distal
               0.55, 0.35,        # index proximal, distal
               0.70, 0.30,        # middle
               0.65, 0.35,        # ring
               0.50, 0.30]        # pinky
HAND_CLOSED_CAN = {s: list(_CLOSED_CAN) for s in SIDES}
# where a grasped can's centre sits in the wrist_yaw_link (hand) frame: over the distal half of
# the palm, one can radius plus 1 mm off the palm surface (+y right / -y left), centred on the
# fingers along the can's axis
GRASP_OFFSET = {"right": np.array([0.120, 0.054, 0.0]), "left": np.array([0.120, -0.054, 0.0])}
FOOT_SITE_HEIGHT = 0.033  # ankle_roll_link origin above the sole
NQ_ROBOT = 7 + 29 + 2 * 11  # floating base, 29 body joints, two 11-joint hands
NV_ROBOT = NQ_ROBOT - 1


# --------------------------------------------------------------------------------------
# small math helpers
# --------------------------------------------------------------------------------------
def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_from_mat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).reshape(9))
    return q


def mat_from_quat(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def quat_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def rotvec_from_mats(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Rotation vector that takes R_current to R_target (world frame)."""
    q = quat_from_mat(R_target @ R_current.T)
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    d = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if d > 0.9995:
        out = q0 + s * (q1 - q0)
        return out / np.linalg.norm(out)
    th = math.acos(d)
    return (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


def smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# --------------------------------------------------------------------------------------
# model bookkeeping / FK
# --------------------------------------------------------------------------------------
class G1Model:
    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.d = mujoco.MjData(model)  # scratch data for IK / FK
        self.jq: Dict[str, int] = {}
        self.jv: Dict[str, int] = {}
        self.jrange: Dict[str, Tuple[float, float]] = {}
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.jq[name] = int(model.jnt_qposadr[j])
            self.jv[name] = int(model.jnt_dofadr[j])
            self.jrange[name] = (float(model.jnt_range[j][0]), float(model.jnt_range[j][1]))
        self.body = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(model.nbody)}
        self.pelvis = self.body["pelvis"]
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        # the robot owns the first NQ_ROBOT entries of qpos, the cans come after it
        nq_robot = 0
        for j in range(model.njnt):
            if self._is_robot_body(int(model.jnt_bodyid[j])):
                width = 7 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
                nq_robot = max(nq_robot, int(model.jnt_qposadr[j]) + width)
        if nq_robot != NQ_ROBOT:
            raise RuntimeError(f"robot qpos layout mismatch: model has {nq_robot} robot qpos entries, expected {NQ_ROBOT}")

    def _is_robot_body(self, b: int) -> bool:
        while b != 0:
            if b == self.pelvis:
                return True
            b = int(self.m.body_parentid[b])
        return False

    # -- accessors --------------------------------------------------------------------
    def qadr(self, names: Sequence[str]) -> np.ndarray:
        return np.array([self.jq[n] for n in names], dtype=int)

    def vadr(self, names: Sequence[str]) -> np.ndarray:
        return np.array([self.jv[n] for n in names], dtype=int)

    def ranges(self, names: Sequence[str]) -> np.ndarray:
        return np.array([self.jrange[n] for n in names])

    def fk(self, q_robot: np.ndarray) -> None:
        """Forward kinematics of the robot part of qpos on the scratch data."""
        self.d.qpos[:NQ_ROBOT] = q_robot
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def body_pose(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        b = self.body[name]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def point_jacobian(self, body_name: str, point_world: np.ndarray):
        mujoco.mj_jac(self.m, self.d, self._jacp, self._jacr, point_world, self.body[body_name])
        return self._jacp, self._jacr


@dataclass
class IKTask:
    body: str
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_pos: Optional[np.ndarray] = None
    target_rot: Optional[np.ndarray] = None  # 3x3
    pos_weight: float = 1.0
    rot_weight: float = 0.5


def solve_ik(robot: G1Model, q: np.ndarray, joints: Sequence[str], tasks: Sequence[IKTask],
             q_rest: Optional[np.ndarray] = None, joint_weights: Optional[np.ndarray] = None,
             iters: int = 6, damping: float = 5e-3, rest_gain: float = 0.3, max_step: float = 0.25,
             tol: float = 1e-4, ranges: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Damped least squares IK on a subset of joints.

    ``q`` is the full robot qpos (modified in place and returned).  Joint weights
    < 1 make a joint "expensive" (used only when needed), which is how the waist
    is kept mostly still while the arm reaches.  ``ranges`` (n x 2) overrides the
    model's joint limits, e.g. to keep the elbow away from the straight-arm singularity.
    """
    qadr = robot.qadr(joints)
    vadr = robot.vadr(joints)
    rng = robot.ranges(joints) if ranges is None else np.asarray(ranges, float)
    n = len(joints)
    w = np.ones(n) if joint_weights is None else np.asarray(joint_weights, float)
    err_norm = 0.0
    for _ in range(iters):
        robot.fk(q)
        rows, errs = [], []
        for t in tasks:
            pos, R = robot.body_pose(t.body)
            p = pos + R @ t.offset
            jacp, jacr = robot.point_jacobian(t.body, p)
            if t.target_pos is not None:
                rows.append(t.pos_weight * jacp[:, vadr])
                errs.append(t.pos_weight * (t.target_pos - p))
            if t.target_rot is not None:
                rows.append(t.rot_weight * jacr[:, vadr])
                errs.append(t.rot_weight * rotvec_from_mats(t.target_rot, R))
        J = np.vstack(rows)
        e = np.concatenate(errs)
        err_norm = float(np.linalg.norm(e))
        if err_norm < tol:
            break
        Jw = J * w[None, :]
        JJt = Jw @ Jw.T + (damping ** 2) * np.eye(J.shape[0])
        Jpinv = Jw.T @ np.linalg.solve(JJt, np.eye(J.shape[0]))
        dq_w = Jpinv @ e
        if q_rest is not None:
            N = np.eye(n) - Jpinv @ Jw
            dq_w += N @ (rest_gain * (q_rest - q[qadr]) / w)
        dq = w * dq_w
        m = np.max(np.abs(dq))
        if m > max_step:
            dq *= max_step / m
        q[qadr] = np.clip(q[qadr] + dq, rng[:, 0], rng[:, 1])
    return q, err_norm


# --------------------------------------------------------------------------------------
# gait
# --------------------------------------------------------------------------------------
@dataclass
class FootState:
    x: float
    y: float
    yaw: float


class FootstepGait:
    """Keeps the feet planted while the (scripted) base moves; steps when needed.

    Footfall targets are the nominal foot pose relative to where the base *will*
    be shortly after touchdown, so the swing foot lands ahead of the body and
    leaves behind it, like a real walk, without any sliding.
    """
    T_SWING = 0.34
    T_DS = 0.10  # minimum double support
    LEAD = 0.26  # after touchdown: time at which the stance foot is centred under the base
    STEP_HEIGHT = 0.065
    STEP_THRESH_POS = 0.025
    STEP_THRESH_YAW = 0.12
    URGENT_POS = 0.20
    URGENT_YAW = 0.55
    FOOT_OFFSET = {"left": np.array([0.0, 0.117]), "right": np.array([0.0, -0.117])}

    def __init__(self, base_xy_yaw: Tuple[float, float, float]):
        self.feet: Dict[str, FootState] = {}
        for s in SIDES:
            x, y, yaw = self.nominal(s, base_xy_yaw)
            self.feet[s] = FootState(x, y, yaw)
        self.swing: Optional[dict] = None
        self.t_last_td = -1.0
        self.last_swing_side = "right"
        self.phase = 0.0  # advances by pi per swing (left swing: 0..pi, right swing: pi..2pi)
        self.swing_progress = 0.0
        self.swing_side: Optional[str] = None

    @classmethod
    def nominal(cls, side: str, base: Tuple[float, float, float]) -> Tuple[float, float, float]:
        bx, by, byaw = base
        ox, oy = cls.FOOT_OFFSET[side]
        c, s = math.cos(byaw), math.sin(byaw)
        return bx + c * ox - s * oy, by + s * ox + c * oy, byaw

    def _error(self, side: str, base: Tuple[float, float, float]) -> Tuple[float, float]:
        nx, ny, nyaw = self.nominal(side, base)
        f = self.feet[side]
        return math.hypot(nx - f.x, ny - f.y), abs(wrap_angle(nyaw - f.yaw))

    def update(self, t: float, base_now: Tuple[float, float, float],
               base_at: Callable[[float], Tuple[float, float, float]]) -> Dict[str, Tuple[np.ndarray, float]]:
        """Returns per-foot (site position xyz, yaw)."""
        if self.swing is not None:
            sw = self.swing
            s = min(max((t - sw["t0"]) / self.T_SWING, 0.0), 1.0)
            self.swing_progress = s
            self.phase = (0.0 if sw["side"] == "left" else math.pi) + math.pi * s
            if s >= 1.0:
                tx, ty, tyaw = sw["target"]
                self.feet[sw["side"]] = FootState(tx, ty, tyaw)
                self.swing = None
                self.swing_side = None
                self.t_last_td = t
        # a foot that has fallen far behind the base must step *now* (skip double support)
        urgent = self.swing is None and any(
            self._error(side, base_now)[0] > self.URGENT_POS or self._error(side, base_now)[1] > self.URGENT_YAW
            for side in SIDES)
        if self.swing is None and ((t - self.t_last_td) >= self.T_DS or urgent):
            fut = base_at(t + self.T_SWING + self.LEAD)
            need = []
            for side in SIDES:
                ep, ey = self._error(side, fut)
                if ep > self.STEP_THRESH_POS or ey > self.STEP_THRESH_YAW:
                    need.append((side, ep + 0.3 * ey))
            if need:
                other = "left" if self.last_swing_side == "right" else "right"
                side = other if any(n[0] == other for n in need) else max(need, key=lambda n: n[1])[0]
                f = self.feet[side]
                self.swing = {"side": side, "t0": t, "start": (f.x, f.y, f.yaw), "target": self.nominal(side, fut)}
                self.last_swing_side = side
                self.swing_side = side
                self.swing_progress = 0.0

        out: Dict[str, Tuple[np.ndarray, float]] = {}
        for side in SIDES:
            f = self.feet[side]
            if self.swing is not None and self.swing["side"] == side:
                s = min(max((t - self.swing["t0"]) / self.T_SWING, 0.0), 1.0)
                k = smoothstep(s)
                sx, sy, syaw = self.swing["start"]
                tx, ty, tyaw = self.swing["target"]
                x = sx + (tx - sx) * k
                y = sy + (ty - sy) * k
                yaw = syaw + wrap_angle(tyaw - syaw) * k
                z = FOOT_SITE_HEIGHT + self.STEP_HEIGHT * math.sin(math.pi * s)
                out[side] = (np.array([x, y, z]), yaw)
            else:
                out[side] = (np.array([f.x, f.y, FOOT_SITE_HEIGHT]), f.yaw)
        return out

    def stance_side(self) -> Optional[str]:
        if self.swing is None:
            return None
        return "left" if self.swing["side"] == "right" else "right"

    def settled(self, base_now: Tuple[float, float, float]) -> bool:
        if self.swing is not None:
            return False
        return all(self._error(s, base_now)[0] < 0.03 and self._error(s, base_now)[1] < 0.15 for s in SIDES)


# --------------------------------------------------------------------------------------
# arm & hand controllers
# --------------------------------------------------------------------------------------
class ArmController:
    """One arm: either tracks a joint-space pose (rate limited) or an IK target.

    Cartesian moves interpolate from the hand pose at the start of the move to the
    goal (position lerp + quaternion slerp) so the hand travels along a straight
    line with an S-curve speed profile.
    """

    def __init__(self, robot: G1Model, side: str, use_waist: bool):
        self.robot = robot
        self.side = side
        self.joints = list(ARM_JOINTS[side])
        self.ik_joints = self.joints + (["waist_pitch_joint", "waist_yaw_joint"] if use_waist else [])
        self.weights = np.array([1.0] * len(self.joints) + ([0.2, 0.2] if use_waist else []))
        self.mode = "pose"
        self.pose_target = np.array(ARM_STAND[side], float)
        self.pose_rate = 2.0  # rad/s
        self.cart: Optional[dict] = None
        self.rest = np.array(ARM_STAND[side] + ([0.0, 0.0] if use_waist else []), float)
        self.last_ik_err = 0.0
        self.ik_rate = 5.0  # rad/s cap on IK-driven joint motion (no pops if a solution flips)
        # IK joint limits.  In the G1 model the elbow is ~90 deg bent at q=0, *positive* q extends
        # it and the arm is straight (singular; the hyper-extended branch lies beyond) at ~1.5 rad.
        # Cap extension at 1.1 rad (still 98% of the reach) and allow the model's full flexion so
        # targets close to the chest stay reachable.  The waist only leans/turns a little.
        self.ik_ranges = robot.ranges(self.ik_joints).copy()
        self.ik_ranges[3] = [max(self.ik_ranges[3, 0], -0.95), min(self.ik_ranges[3, 1], 1.1)]
        if use_waist:
            self.ik_ranges[7] = [-0.12, 0.45]  # waist pitch: slight lean back .. lean forward
            self.ik_ranges[8] = [-0.5, 0.5]    # waist yaw

    # joint-space
    def set_pose(self, pose: Sequence[float], rate: float = 2.0) -> None:
        self.mode = "pose"
        self.pose_target = np.array(pose, float)
        self.pose_rate = rate
        self.cart = None

    # Cartesian
    def move_to(self, q: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray, duration: float, t_now: float,
                offset: Optional[np.ndarray] = None) -> None:
        off = GRASP_OFFSET[self.side] if offset is None else offset
        self.robot.fk(q)
        pos, R = self.robot.body_pose(HAND_BODY[self.side])
        start_pos = pos + R @ off
        self.cart = {
            "t0": t_now, "T": max(duration, 1e-3), "offset": off,
            "p0": start_pos, "p1": np.asarray(target_pos, float),
            "q0": quat_from_mat(R), "q1": quat_from_mat(target_rot),
        }
        self.mode = "ik"

    def cart_done(self, t_now: float) -> bool:
        return self.cart is None or t_now >= self.cart["t0"] + self.cart["T"] + 0.05

    def update(self, q: np.ndarray, t_now: float, dt: float, pose_offset: Optional[np.ndarray] = None) -> None:
        qadr = self.robot.qadr(self.joints)
        if self.mode == "pose":
            target = self.pose_target + (0.0 if pose_offset is None else pose_offset)
            cur = q[qadr]
            step = np.clip(target - cur, -self.pose_rate * dt, self.pose_rate * dt)
            q[qadr] = cur + step
            # let the waist relax back too
            for wj in ("waist_pitch_joint", "waist_yaw_joint"):
                a = self.robot.jq[wj]
                q[a] += np.clip(-q[a], -1.0 * dt, 1.0 * dt)
        else:
            c = self.cart
            s = smoothstep((t_now - c["t0"]) / c["T"])
            p = c["p0"] + (c["p1"] - c["p0"]) * s
            R = mat_from_quat(slerp(c["q0"], c["q1"], s))
            task = IKTask(HAND_BODY[self.side], c["offset"], p, R, pos_weight=1.0, rot_weight=0.4)
            ik_adr = self.robot.qadr(self.ik_joints)
            q_prev = q[ik_adr].copy()
            _, self.last_ik_err = solve_ik(self.robot, q, self.ik_joints, [task], q_rest=self.rest,
                                           joint_weights=self.weights, iters=5, rest_gain=0.15,
                                           ranges=self.ik_ranges)
            q[ik_adr] = q_prev + np.clip(q[ik_adr] - q_prev, -self.ik_rate * dt, self.ik_rate * dt)


class HandController:
    def __init__(self, robot: G1Model, side: str):
        self.robot = robot
        self.side = side
        self.qadr = robot.qadr(HAND_JOINTS[side])
        self.target = np.array(HAND_OPEN[side], float)
        self.rate = 4.0

    def open(self) -> None:
        self.target = np.array(HAND_OPEN[self.side], float)

    def close_on_can(self) -> None:
        self.target = np.array(HAND_CLOSED_CAN[self.side], float)

    def update(self, q: np.ndarray, dt: float) -> None:
        cur = q[self.qadr]
        q[self.qadr] = cur + np.clip(self.target - cur, -self.rate * dt, self.rate * dt)

    def settled(self, q: np.ndarray) -> bool:
        return bool(np.max(np.abs(q[self.qadr] - self.target)) < 0.02)


# --------------------------------------------------------------------------------------
# whole-body puppet
# --------------------------------------------------------------------------------------
class G1Puppet:
    """Combines base trajectory, gait, arms and hands into one qpos vector per tick."""

    # hip pitch joint relative to the pelvis origin, and the longest hip->ankle distance we allow
    HIP_OFFSET = {"left": np.array([0.0, 0.064, -0.103]), "right": np.array([0.0, -0.064, -0.103])}
    MAX_LEG_REACH = 0.615

    def __init__(self, model: mujoco.MjModel, pelvis_height: float, start_pose: Tuple[float, float, float]):
        self.robot = G1Model(model)
        self.pelvis_height = pelvis_height
        self.q = np.zeros(NQ_ROBOT)
        self.q[3] = 1.0
        for side in SIDES:
            self.q[self.robot.qadr(ARM_JOINTS[side])] = ARM_STAND[side]
        # slightly bent knees so the leg IK starts away from the straight-leg singularity
        for side in SIDES:
            self.q[self.robot.jq[f"{side}_hip_pitch_joint"]] = -0.25
            self.q[self.robot.jq[f"{side}_knee_joint"]] = 0.5
            self.q[self.robot.jq[f"{side}_ankle_pitch_joint"]] = -0.25
        self.base = start_pose  # (x, y, yaw)
        self.gait = FootstepGait(start_pose)
        self.arms = {"left": ArmController(self.robot, "left", use_waist=False),
                     "right": ArmController(self.robot, "right", use_waist=True)}
        self.hands = {s: HandController(self.robot, s) for s in SIDES}
        self.leg_rest = {s: np.array([-0.25, 0.0, 0.0, 0.5, -0.25, 0.0]) for s in SIDES}
        # IK joint limits for the legs: keep the knee at least slightly bent and the hip
        # pitch/roll/yaw in the range a walking human uses (rules out the backwards branch)
        self.leg_ranges = self.robot.ranges(LEG_JOINTS["left"]).copy()
        self.leg_ranges[0] = [-1.6, 1.2]   # hip pitch
        self.leg_ranges[1] = [-0.5, 0.5]   # hip roll (mirrored range is symmetric enough here)
        self.leg_ranges[2] = [-1.2, 1.2]   # hip yaw
        self.leg_ranges[3, 0] = 0.05       # knee
        self.arm_swing = 0.0  # amplitude of walking arm swing (rad)
        self.swing_right = True  # False while carrying a can: keep the drink steady
        self.walking = False

    # -- base -------------------------------------------------------------------------
    def set_base(self, xy_yaw: Tuple[float, float, float]) -> None:
        self.base = xy_yaw

    def base_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y, yaw = self.base
        return np.array([x, y, 0.0]), rot_z(yaw)

    # -- tick -------------------------------------------------------------------------
    def update(self, t: float, dt: float, base_at: Callable[[float], Tuple[float, float, float]]) -> None:
        q = self.q
        x, y, yaw = self.base
        feet = self.gait.update(t, self.base, base_at)

        # pelvis: small bob + sway towards the stance foot during swing
        stance = self.gait.stance_side()
        sway = 0.0
        bob = 0.0
        if stance is not None:
            s = self.gait.swing_progress
            sway = (0.015 if stance == "left" else -0.015) * math.sin(math.pi * s)
            bob = -0.01 * math.sin(math.pi * s)
        c, sn = math.cos(yaw), math.sin(yaw)
        q[0] = x - sn * sway
        q[1] = y + c * sway
        q[2] = self.pelvis_height + bob
        q[3:7] = quat_yaw(yaw)

        # legs via IK on foot bodies
        pelvis_pos = q[0:3]
        R_base = rot_z(yaw)
        for side in SIDES:
            pos, fyaw = feet[side]
            # keep the target inside the leg's reach (hip pitch joint -> ankle), otherwise the
            # damped IK ends up in a bent-backwards branch it cannot leave
            hip = pelvis_pos + R_base @ (self.HIP_OFFSET[side])
            vec = pos - hip
            dist = np.linalg.norm(vec)
            if dist > self.MAX_LEG_REACH:
                pos = hip + vec * (self.MAX_LEG_REACH / dist)
            task = IKTask(FOOT_BODY[side], np.zeros(3), pos, rot_z(fyaw), pos_weight=1.0, rot_weight=0.5)
            qadr = self.robot.qadr(LEG_JOINTS[side])
            _, err = solve_ik(self.robot, q, LEG_JOINTS[side], [task], q_rest=self.leg_rest[side],
                              iters=4, damping=8e-3, rest_gain=0.05, max_step=0.3, ranges=self.leg_ranges)
            knee = q[self.robot.jq[f"{side}_knee_joint"]]
            if err > 0.03 or knee < 0.06:
                # bad branch / stuck at the straight-leg limit: restart from the rest pose
                q[qadr] = self.leg_rest[side]
                solve_ik(self.robot, q, LEG_JOINTS[side], [task], q_rest=self.leg_rest[side],
                         iters=12, damping=8e-3, rest_gain=0.05, max_step=0.3, ranges=self.leg_ranges)

        # arms
        ph = self.gait.phase
        swing_l = self.arm_swing * math.sin(ph)     # left arm forward while right leg swings
        swing_r = -self.arm_swing * math.sin(ph) if self.swing_right else 0.0
        self.arms["left"].update(q, t, dt, pose_offset=np.array([swing_l, 0, 0, 0, 0, 0, 0]))
        self.arms["right"].update(q, t, dt, pose_offset=np.array([swing_r, 0, 0, 0, 0, 0, 0]))
        for side in SIDES:
            self.hands[side].update(q, dt)

    # -- helpers used by the scenario -------------------------------------------------
    def hand_frame(self, side: str) -> Tuple[np.ndarray, np.ndarray]:
        self.robot.fk(self.q)
        return self.robot.body_pose(HAND_BODY[side])

    def grasp_point(self, side: str) -> np.ndarray:
        pos, R = self.hand_frame(side)
        return pos + R @ GRASP_OFFSET[side]
