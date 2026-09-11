"""Unitree G1 29-DoF with BrainCo Revo 2 hands: stand, walk to a table, reach, grasp and lift the demo object
(default: a 12 oz Pepsi can; 20 oz / 500 mL bottles selectable).

Pipeline (Isaac Sim 5.0 / Isaac Lab 2.2, CPU physics):
  1. Merged URDF (sim/build_g1_revo2_urdf.py) -> USD via the Isaac URDF importer (cached under work/g1-runtime/usd).
  2. Unitree's whole-body actuator gains + standing pose; Unitree's simulation-only locomotion policy
     (unitree_sim_isaaclab/assets/model/policy.onnx: 910-d history obs -> 12 leg targets @ 50 Hz).
  3. Walk forward on a velocity command until the bottle is within arm reach, stop, settle.
  4. Differential IK on the right arm brings the pre-shaped hand next to the bottle (pocket geometry from
     revo2_hand_grasp.py: bottle surface ~34 mm in front of the knuckle axes, axis 40 mm toward the fingertips).
  5. Close (rigid-linkage finger coupling), lift 10 cm, hold, report.

Phases can be run separately (--phase stand|walk|grasp|all) and the arm/grasp part also works with --fixed-base.
Everything here is a simulation development tool; the locomotion weights are Unitree's sim-only release.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pepsidemo_sim as pd  # noqa: E402

PROJECT = pd.PROJECT_ROOT
URDF = PROJECT / "work/g1-runtime/urdf/g1_29dof_revo2.urdf"
MOUNTS = PROJECT / "work/g1-runtime/urdf/g1_29dof_revo2.json"
POLICY = PROJECT / "work/unitree_sim_isaaclab/assets/model/policy.onnx"
REVO_URDF = PROJECT / "work/brainco-description/revo2_system/urdf/revo2_{side}.urdf"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--phase", choices=["load", "ikcheck", "stand", "walk", "grasp", "all"], default="all")
parser.add_argument("--fixed-base", action="store_true", help="pin the pelvis (no locomotion policy); arm + grasp only")
parser.add_argument("--hand", choices=["right", "left"], default="right")
parser.add_argument("--bottle", choices=["pepsi-12oz-can", "pepsi-20oz", "pepsi-500ml"], default="pepsi-12oz-can",
                    help="object to grasp (assets/bottles/ presets; the demo serves 12 oz cans)")
parser.add_argument("--table-x", type=float, default=1.6, help="m ahead of the robot start (floating base)")
parser.add_argument("--bottle-y", type=float, default=-0.12, help="m lateral bottle offset on the table (negative = robot's right)")
parser.add_argument("--table-height", type=float, default=None, help="table top height (m); default: pelvis start height - 0.02")
parser.add_argument("--walk-speed", type=float, default=0.5, help="m/s; Unitree's sim policy stands still below ~0.5 m/s")
parser.add_argument("--reach-x", type=float, default=0.35, help="desired bottle distance ahead of the pelvis after stopping")
parser.add_argument("--stop-lead", type=float, default=0.20, help="command zero velocity this far before reach-x (stopping distance)")
parser.add_argument("--gap", type=float, default=0.034)
parser.add_argument("--distal-offset", type=float, default=0.040)
parser.add_argument("--grasp-height", type=float, default=None,
                    help="palm centre above the object base (m); default per object: 0.10 on bottles, 0.06 on the can (mid-body, ~its CoM)")
parser.add_argument("--lift", type=float, default=0.10)
parser.add_argument("--holder", type=float, default=0.0, help="height (m) of a rigid insert around the object base (basket insert, sized from the mesh + 3 mm); 0 = none")
parser.add_argument("--finger-effort", type=float, default=1.5, help="finger drive torque limit Nm (the real hand is current-limited; 0.3-0.5 ~ gentle stall)")
parser.add_argument("--press", type=float, default=0.0, help="m: approach target pushes the palm this far *into* the bottle surface before closing; released on lift")
parser.add_argument("--arm-time-scale", type=float, default=1.0, help="multiply arm motion durations (slower = less disturbance to the balance policy)")
parser.add_argument("--collider", choices=["convex_hull", "convex_decomposition"], default="convex_decomposition")
parser.add_argument("--reconvert", action="store_true")
parser.add_argument("--merge-fixed", action="store_true", help="merge fixed joints on import (fewer bodies; fingertip/touch links fold into the distal links)")
parser.add_argument("--snapshot", action="store_true")
parser.add_argument("--video", action="store_true")
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--video-size", type=str, default="1280x720")
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--max-seconds", type=float, default=40.0)
parser.add_argument("--dt", type=float, default=0.005, help="physics step; control runs at 50 Hz regardless")
parser.add_argument("--solver-iters", type=int, default=16, help="articulation position solver iterations (velocity = 1/8 of this, min 1)")
parser.add_argument("--ik-gain", type=float, default=0.3, help="fraction of the IK Newton step applied to the commanded target per 50 Hz tick")
parser.add_argument("--arm-kp-scale", type=float, default=1.0, help="multiply the grasping arm's PD stiffness/damping from the approach on (diagnostic for arm compliance)")
parser.add_argument("--arm-gains", choices=["grasp", "teleop", "arm_sdk", "sim"], default="grasp",
                    help="PD gains on the grasping arm once it leaves the standing pose (raise phase on), all sent over rt/arm_sdk on the real "
                         "robot (docs/robot/README.md, arm stiffness study). 'grasp' = uniform kp 120 / kd 3, the profile that holds the "
                         "bottle through the lift in sim and our proposal for the real arm. 'teleop' = what Unitree's xr_teleoperate and "
                         "BrainCo's stack send this G1: shoulders+elbow 300/3, wrists 40/1.5 (fails the palm press in sim). 'arm_sdk' = "
                         "Unitree's minimal arm7 example, --arm-sdk-kp/--arm-sdk-kd on all 7 joints (60/1.5 default; too soft). "
                         "'sim' = Unitree's whole-body RL sim gains (shoulders 100/2, elbow 50/2, wrists 40/2; last night's runs)")
parser.add_argument("--arm-sdk-kp", type=float, default=60.0)
parser.add_argument("--arm-sdk-kd", type=float, default=1.5)
parser.add_argument("--wrist-kp", type=float, default=None, help="override the wrist kp in the teleop/arm_sdk profiles (sim study: the wrist is the joint the palm press loads)")
parser.add_argument("--wrist-kd", type=float, default=None)
parser.add_argument("--policy", type=str, default=str(POLICY), help="ONNX locomotion policy path")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu", rendering_mode="performance")
args = parser.parse_args()
if args.video:
    args.snapshot = True
if args.snapshot:
    args.enable_cameras = True
CAM_W, CAM_H = (int(v) for v in args.video_size.lower().split("x"))
pd.ensure_dirs()
pd.select_experience(args)

launcher = AppLauncher(args, renderer="RayTracedLighting", anti_aliasing=0, multi_gpu=False)
app = launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg, UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms, matrix_from_quat, quat_error_magnitude, quat_from_matrix, quat_inv, quat_slerp,
    subtract_frame_transforms,
)

if args.snapshot:
    from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

side = args.hand
IK_GAIN = args.ik_gain
log_dir = pd.LOG_DIR
ts = time.strftime("%Y%m%d-%H%M%S")

# ----------------------------------------------------------------------------------------------------------------------
# Unitree whole-body joint conventions (from unitree_sim_isaaclab/action_provider/action_provider_wh_dds.py)
# ----------------------------------------------------------------------------------------------------------------------
LEG_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
OLD_ACTION_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
    "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
SIDE_ARM = [j for j in ARM_JOINTS if j.startswith(side)]
HAND_ACTIVE = {  # SDK order thumb, thumb_aux, index, middle, ring, pinky
    "thumb": f"{side}_thumb_proximal_joint", "thumb_aux": f"{side}_thumb_metacarpal_joint",
    "index": f"{side}_index_proximal_joint", "middle": f"{side}_middle_proximal_joint",
    "ring": f"{side}_ring_proximal_joint", "pinky": f"{side}_pinky_proximal_joint",
}
DISTAL = {
    f"{side}_index_distal_joint": (f"{side}_index_proximal_joint", 1.155),
    f"{side}_middle_distal_joint": (f"{side}_middle_proximal_joint", 1.155),
    f"{side}_ring_distal_joint": (f"{side}_ring_proximal_joint", 1.155),
    f"{side}_pinky_distal_joint": (f"{side}_pinky_proximal_joint", 1.155),
    f"{side}_thumb_distal_joint": (f"{side}_thumb_proximal_joint", 1.0),
}
UPPER = {"index": 1.41, "middle": 1.41, "ring": 1.41, "pinky": 1.41, "thumb": 1.03, "thumb_aux": 1.57}


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def palm_frame_in_hand_base():
    """Palm centre (mean knuckle origin) and triad (fingers f, index->pinky w, palm normal n) in {side}_hand_base_link."""
    import xml.etree.ElementTree as ET
    root = ET.parse(str(REVO_URDF).format(side=side)).getroot()
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
        joints[j.get("name")] = (j.find("parent").get("link"), j.find("child").get("link"), xyz, rpy_to_mat(*rpy))
    c2j = {v[1]: k for k, v in joints.items()}

    axes = {}
    for j in root.findall("joint"):
        ax = j.find("axis")
        axes[j.get("name")] = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "0 0 1").split()])

    def pose(link):
        R, p = np.eye(3), np.zeros(3)
        chain = []
        while link in c2j and link != f"{side}_hand_base_link":
            chain.append(joints[c2j[link]])
            link = joints[c2j[link]][0]
        for _, _, xyz, Rj in reversed(chain):
            p = p + R @ xyz
            R = R @ Rj
        return R, p

    knuckles = {f: pose(f"{side}_{f}_proximal_link")[1] for f in ("index", "middle", "ring", "pinky")}
    R_idx, p_idx = pose(f"{side}_index_proximal_link")
    _, p_tip = pose(f"{side}_index_tip_link")
    f = p_tip - p_idx
    f /= np.linalg.norm(f)
    w = knuckles["pinky"] - knuckles["index"]
    w -= np.dot(w, f) * f
    w /= np.linalg.norm(w)
    n = np.cross(R_idx @ axes[f"{side}_index_proximal_joint"], f)
    n -= np.dot(n, f) * f
    n -= np.dot(n, w) * w
    n /= np.linalg.norm(n)
    if np.dot(np.cross(f, w), n) < 0:
        w = -w
    return np.mean(list(knuckles.values()), axis=0), f, w, n


# ----------------------------------------------------------------------------------------------------------------------
def convert_robot():
    if not URDF.is_file():
        raise SystemExit(f"Missing merged URDF {URDF}; run sim/build_g1_revo2_urdf.py first")
    base_tag = "fixed" if args.fixed_base else "floating"
    cfg = UrdfConverterCfg(
        asset_path=str(URDF), usd_dir=str(pd.USD_CACHE / "g1_revo2"),
        usd_file_name=f"g1_29dof_revo2_{base_tag}_{args.collider}{'_merged' if args.merge_fixed else ''}.usd",
        force_usd_conversion=args.reconvert, make_instanceable=False, fix_base=args.fixed_base,
        merge_fixed_joints=args.merge_fixed, link_density=0.0, collider_type=args.collider, self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force", target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=5.0),
        ),
    )
    t0 = time.time()
    usd = UrdfConverter(cfg).usd_path
    print(f"G1_USD: {usd} ({time.time() - t0:.1f}s)", flush=True)
    return usd


def robot_cfg(usd_path):
    hand_expr = [".*_thumb_.*_joint", ".*_index_.*_joint", ".*_middle_.*_joint", ".*_ring_.*_joint", ".*_pinky_.*_joint"]
    return ArticulationCfg(
        prim_path="/World/G1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=True, linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=args.solver_iters,
                solver_velocity_iteration_count=max(1, args.solver_iters // 8),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.80 if not args.fixed_base else 1.0),
            joint_pos={
                ".*_hip_pitch_joint": -0.20, ".*_knee_joint": 0.42, ".*_ankle_pitch_joint": -0.23,
                ".*_elbow_joint": 0.87, "left_shoulder_roll_joint": 0.18, "left_shoulder_pitch_joint": 0.35,
                "right_shoulder_roll_joint": -0.18, "right_shoulder_pitch_joint": 0.35,
                ".*_thumb_.*_joint": 0.0, ".*_index_.*_joint": 0.0, ".*_middle_.*_joint": 0.0,
                ".*_ring_.*_joint": 0.0, ".*_pinky_.*_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.90,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", ".*waist.*"],
                effort_limit_sim={".*_hip_yaw_joint": 88.0, ".*_hip_roll_joint": 139.0, ".*_hip_pitch_joint": 88.0,
                                  ".*_knee_joint": 139.0, ".*waist_yaw_joint": 88.0, ".*waist_roll_joint": 35.0, ".*waist_pitch_joint": 35.0},
                velocity_limit_sim={".*_hip_yaw_joint": 32.0, ".*_hip_roll_joint": 20.0, ".*_hip_pitch_joint": 32.0,
                                    ".*_knee_joint": 20.0, ".*waist_yaw_joint": 32.0, ".*waist_roll_joint": 30.0, ".*waist_pitch_joint": 30.0},
                stiffness={".*_hip_yaw_joint": 150.0, ".*_hip_roll_joint": 150.0, ".*_hip_pitch_joint": 200.0, ".*_knee_joint": 200.0, ".*waist.*": 200.0},
                damping={".*_hip_yaw_joint": 5.0, ".*_hip_roll_joint": 5.0, ".*_hip_pitch_joint": 5.0, ".*_knee_joint": 5.0, ".*waist.*": 5.0},
                armature=0.01,
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit_sim=35.0, velocity_limit_sim=30.0, stiffness=20.0, damping=2.0, armature=0.01,
            ),
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
                effort_limit_sim=25.0, velocity_limit_sim=37.0, stiffness=100.0, damping=2.0, armature=0.01,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_yaw_joint", ".*_elbow_joint"],
                effort_limit_sim=25.0, velocity_limit_sim=37.0, stiffness=50.0, damping=2.0, armature=0.01,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=[".*_wrist_.*"],
                effort_limit_sim={".*_wrist_yaw_joint": 5.0, ".*_wrist_roll_joint": 25.0, ".*_wrist_pitch_joint": 5.0},
                velocity_limit_sim={".*_wrist_yaw_joint": 22.0, ".*_wrist_roll_joint": 37.0, ".*_wrist_pitch_joint": 22.0},
                stiffness=40.0, damping=2.0, armature=0.01,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=hand_expr, effort_limit_sim=args.finger_effort, velocity_limit_sim=2.3,  # Revo 2: full close <= 0.65 s
                stiffness=6.0, damping=0.3, armature=0.001,
            ),
        },
    )


class LocoPolicy:
    """Unitree's sim-only G1 locomotion policy: 10-step history of 91-d obs -> 12 leg joint targets (50 Hz)."""

    def __init__(self, robot, device):
        import onnxruntime as ort

        self.robot = robot
        self.device = device
        self.sess = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        names = robot.joint_names
        idx = {n: i for i, n in enumerate(names)}
        self.leg_ids = [idx[n] for n in LEG_JOINTS]
        self.waist_ids = [idx[n] for n in WAIST_JOINTS]
        self.arm_ids = [idx[n] for n in ARM_JOINTS]
        self.obs_ids = self.leg_ids + self.arm_ids
        self.old_ids = [idx[n] for n in OLD_ACTION_JOINTS]
        self.pos_in_old = {n: OLD_ACTION_JOINTS.index(n) for n in OLD_ACTION_JOINTS}
        self.default_pos = robot.data.default_joint_pos[0].clone()
        self.default_vel = robot.data.default_joint_vel[0].clone()
        self.hist = deque(maxlen=10)
        self.last_action29 = torch.zeros(29, device=device)
        self.action_scale = 0.25
        self.arm_targets = self.default_pos[self.arm_ids].clone()

    def reset(self):
        self.hist.clear()
        self.last_action29.zero_()

    def obs(self, command):
        d = self.robot.data
        q = d.joint_pos[0]
        qd = d.joint_vel[0]
        cur = torch.cat([
            d.root_ang_vel_b[0], d.projected_gravity_b[0], torch.tensor(command, device=self.device, dtype=torch.float32),
            (q[self.obs_ids] - self.default_pos[self.obs_ids]), (qd[self.obs_ids] - self.default_vel[self.obs_ids]),
            self.last_action29,
        ])
        if not self.hist:
            for _ in range(10):
                self.hist.append(cur)
        else:
            self.hist.append(cur)
        return torch.clip(torch.cat(list(self.hist)), -100.0, 100.0)

    def step(self, command, joint_targets):
        """Fill leg + waist entries of `joint_targets` (full joint vector) and return it."""
        o = self.obs(command).cpu().numpy().astype(np.float32)[None, :]
        a = torch.tensor(self.sess.run(None, {self.in_name: o})[0][0], device=self.device)
        # bookkeeping of the 29-d "last action" the policy was trained with: raw leg actions, absolute waist/arm targets
        for k, n in enumerate(LEG_JOINTS):
            self.last_action29[self.pos_in_old[n]] = a[k]
        for k, n in enumerate(WAIST_JOINTS):
            self.last_action29[self.pos_in_old[n]] = self.default_pos[self.waist_ids[k]]
        for k, n in enumerate(ARM_JOINTS):
            self.last_action29[self.pos_in_old[n]] = self.arm_targets[k]
        joint_targets[self.leg_ids] = torch.clip(a, -100, 100) * self.action_scale + self.default_pos[self.leg_ids]
        joint_targets[self.waist_ids] = self.default_pos[self.waist_ids]
        joint_targets[self.arm_ids] = self.arm_targets
        return joint_targets


def bottle_asset():
    """USD of the demo object + (radius at grasp height, max radius, height, mass); resolves the per-object grasp height."""
    import make_bottle_mesh as mbm
    import trimesh

    obj = mbm.OUT_DIR / f"{args.bottle}.obj"
    if not obj.is_file():
        mbm.build(args.bottle, **mbm.PRESETS[args.bottle])
    if args.grasp_height is None:
        args.grasp_height = mbm.default_grasp_height(args.bottle)
    mesh = trimesh.load(str(obj), force="mesh")
    sec = mesh.section(plane_origin=[0, 0, args.grasp_height], plane_normal=[0, 0, 1])
    pts = np.asarray(sec.vertices)
    r = float(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2).max())
    r_max = float(np.sqrt(mesh.vertices[:, 0] ** 2 + mesh.vertices[:, 1] ** 2).max())
    print(f"G1_OBJECT {args.bottle}: height {1000*(mesh.bounds[1][2]-mesh.bounds[0][2]):.0f} mm, radius {1000*r:.1f} mm at grasp height "
          f"{1000*args.grasp_height:.0f} mm (max {1000*r_max:.1f} mm), mass {mbm.PRESETS[args.bottle]['mass']:.3f} kg", flush=True)
    usd = MeshConverter(MeshConverterCfg(
        asset_path=str(obj), usd_dir=str(pd.USD_CACHE / "bottles"), usd_file_name=f"{args.bottle}.usd",
        force_usd_conversion=False, make_instanceable=False, collision_approximation="convexDecomposition",
        mass_props=sim_utils.MassPropertiesCfg(mass=mbm.PRESETS[args.bottle]["mass"]),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=2, max_depenetration_velocity=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
    )).usd_path
    return usd, r, r_max, float(mesh.bounds[1][2] - mesh.bounds[0][2]), mbm.PRESETS[args.bottle]["mass"]


def main():
    dt = args.dt
    decim = max(1, int(round(0.02 / dt)))  # policy / IK at 50 Hz
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=dt, render_interval=decim))
    ground = sim_utils.GroundPlaneCfg(color=(0.16, 0.17, 0.19))
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95))
    light.func("/World/Light", light)

    usd = convert_robot()
    robot = Articulation(robot_cfg(usd))

    # table + bottle
    pelvis_z0 = 0.80 if not args.fixed_base else 1.0
    table_x = args.table_x if not args.fixed_base else args.reach_x
    table_top = args.table_height if args.table_height is not None else pelvis_z0 - 0.02
    bottle_usd, r_grasp, r_max, bottle_h, bottle_mass = bottle_asset()
    table_cfg = sim_utils.CuboidCfg(
        size=(0.4, 0.9, 0.04), collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.38, 0.30)),
    )
    table_cfg.func("/World/TableTop", table_cfg, translation=(table_x, 0.0, table_top - 0.02))
    for i, (dx, dy) in enumerate(((-0.17, -0.42), (-0.17, 0.42), (0.17, -0.42), (0.17, 0.42))):
        leg = sim_utils.CuboidCfg(size=(0.04, 0.04, table_top - 0.04), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)))
        leg.func(f"/World/TableLeg{i}", leg, translation=(table_x + dx, dy, (table_top - 0.04) / 2))
    bottle_pos0 = np.array([table_x, args.bottle_y, table_top])
    if args.holder > 0.0:
        # basket insert: four static walls hugging the object body (3 mm clearance), open at the top
        rb = r_max + 0.003
        wall_t = 0.01
        for k, (dx, dy, sx, sy) in enumerate(((rb + wall_t / 2, 0.0, wall_t, 2 * rb + 2 * wall_t), (-(rb + wall_t / 2), 0.0, wall_t, 2 * rb + 2 * wall_t),
                                             (0.0, rb + wall_t / 2, 2 * rb, wall_t), (0.0, -(rb + wall_t / 2), 2 * rb, wall_t))):
            wall = sim_utils.CuboidCfg(size=(sx, sy, args.holder), collision_props=sim_utils.CollisionPropertiesCfg(),
                                       visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.22)))
            wall.func(f"/World/Holder{k}", wall, translation=(table_x + dx, args.bottle_y + dy, table_top + args.holder / 2))
    bottle = RigidObject(RigidObjectCfg(
        prim_path="/World/Bottle", init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(bottle_pos0.tolist())),
        spawn=sim_utils.UsdFileCfg(
            usd_path=bottle_usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=2),
            mass_props=sim_utils.MassPropertiesCfg(mass=bottle_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.25, 0.75), roughness=0.35),
        ),
    ))
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.9, friction_combine_mode="average", restitution=0.0)
    mat.func("/World/Looks/BottleMaterial", mat)
    sim_utils.bind_physics_material("/World/Bottle", "/World/Looks/BottleMaterial")

    camera = None
    if args.snapshot:
        camera = Camera(CameraCfg(
            prim_path="/World/Cam", update_period=0, height=CAM_H, width=CAM_W, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=2.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)),
        ))

    sim.set_camera_view((2.5, -2.5, 1.6), (0.8, 0.0, 0.8))
    sim.reset()

    jn = robot.joint_names
    idx = {n: i for i, n in enumerate(jn)}
    print(f"G1_JOINTS ({len(jn)}): {jn}", flush=True)
    print(f"G1_BODIES ({len(robot.body_names)}): {robot.body_names}", flush=True)
    print(f"G1_FIXED_BASE={robot.is_fixed_base} mass={float(robot.data.default_mass.sum()):.2f} kg", flush=True)
    missing = [n for n in LEG_JOINTS + WAIST_JOINTS + ARM_JOINTS + list(HAND_ACTIVE.values()) if n not in idx]
    if missing:
        raise RuntimeError(f"joints missing from the articulation: {missing}")
    hand_body = f"{side}_hand_base_link"
    if hand_body not in robot.body_names:
        raise RuntimeError(f"{hand_body} not found among bodies (merge_fixed_joints collapsed it?)")
    hand_body_idx = robot.body_names.index(hand_body)
    for grp in ("right_index_proximal_joint", "right_index_distal_joint", "right_thumb_metacarpal_joint", "right_shoulder_pitch_joint", "right_wrist_yaw_joint", "left_knee_joint"):
        ji = idx[grp]
        print(f"G1_GAINS {grp:30s} kp={float(robot.data.joint_stiffness[0, ji]):7.1f} kd={float(robot.data.joint_damping[0, ji]):6.2f} "
              f"effort_lim={float(robot.data.joint_effort_limits[0, ji]):7.1f} vel_lim={float(robot.data.joint_velocity_limits[0, ji]):6.1f} armature={float(robot.data.joint_armature[0, ji]):.3f}", flush=True)

    mounts = json.loads(MOUNTS.read_text())["mounts"][side]
    palm_c_h, f_h, w_h, n_h = palm_frame_in_hand_base()
    palm_h = torch.tensor(palm_c_h, device=sim.device, dtype=torch.float32)
    R_mount = torch.tensor(rpy_to_mat(*mounts["rpy"]), device=sim.device, dtype=torch.float32)
    # desired hand-base orientation in the pelvis frame: true finger direction -> +x, palm normal -> toward the midline,
    # index above pinky. Built from the URDF palm triad (the hand-base axes are ~18 deg off the palm normal).
    sgn_n = 1.0 if side == "right" else -1.0
    B_h = np.stack([f_h, w_h, n_h], axis=1)
    f_t, n_t = np.array([1.0, 0.0, 0.0]), np.array([0.0, sgn_n, 0.0])
    w_t = np.cross(n_t, f_t)  # right-handed completion (f x w = n)
    T_ = np.stack([f_t, w_t, n_t], axis=1)
    R_des_np = T_ @ np.linalg.inv(B_h)
    u_, _, vt_ = np.linalg.svd(R_des_np)
    R_des_np = u_ @ vt_
    if (R_des_np @ (np.zeros(3)))[2] > 1e9:  # placeholder to keep structure simple
        pass
    print(f"G1_HAND_TRIAD f={np.round(f_h,3).tolist()} w={np.round(w_h,3).tolist()} n={np.round(n_h,3).tolist()} -> desired hand R maps f->{np.round(R_des_np @ f_h,2).tolist()} n->{np.round(R_des_np @ n_h,2).tolist()} w->{np.round(R_des_np @ w_h,2).tolist()}", flush=True)

    # controllers
    targets = robot.data.default_joint_pos.clone()  # (1, n)
    loco = None if args.fixed_base else LocoPolicy(robot, sim.device)
    arm_ids = [idx[n] for n in SIDE_ARM]
    arm_ids_t = torch.tensor(arm_ids, device=sim.device)
    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.05}),
        num_envs=1, device=sim.device,
    )
    jacobi_body = hand_body_idx - 1 if robot.is_fixed_base else hand_body_idx
    jacobi_joints = arm_ids if robot.is_fixed_base else [i + 6 for i in arm_ids]

    def hand_pose_b():
        """hand base link pose in the pelvis frame."""
        pos_w = robot.data.body_pos_w[0, hand_body_idx]
        quat_w = robot.data.body_quat_w[0, hand_body_idx]
        return subtract_frame_transforms(robot.data.root_pos_w[0:1], robot.data.root_quat_w[0:1], pos_w[None], quat_w[None])

    def jacobian_b():
        jac = robot.root_physx_view.get_jacobians()[:, jacobi_body, :, jacobi_joints]
        if not robot.is_fixed_base:
            Rb = matrix_from_quat(quat_inv(robot.data.root_quat_w))
            jac = jac.clone()
            jac[:, :3, :] = torch.bmm(Rb, jac[:, :3, :])
            jac[:, 3:, :] = torch.bmm(Rb, jac[:, 3:, :])
        return jac

    def palm_target_to_hand_pose(palm_pos_b, R_hand_b):
        """Given desired palm-centre position (pelvis frame) and hand-base rotation, return hand base pos/quat."""
        pos = palm_pos_b - R_hand_b @ palm_h
        quat = quat_from_matrix(R_hand_b[None])[0]
        return pos, quat

    R_hand_des = torch.tensor(R_des_np, device=sim.device, dtype=torch.float32)

    # hand joint targets
    n_j = len(jn)

    def hand_targets(fingers, thumb, aux):
        q = torch.zeros(n_j, device=sim.device)
        for k, name in HAND_ACTIVE.items():
            frac = {"thumb": thumb, "thumb_aux": aux}.get(k, fingers)
            q[idx[name]] = frac * UPPER[k]
        return q

    q_hand_open = hand_targets(0.0, 0.0, 0.0)
    q_hand_pre = hand_targets(0.0, 0.0, 0.9)
    q_hand_close = hand_targets(0.9, 0.75, 0.9)
    hand_ids = [idx[n] for n in HAND_ACTIVE.values()] + [idx[n] for n in DISTAL if n in idx]

    hand_prev = {"q": q_hand_open, "q0": q_hand_open, "t0": 0.0, "T": 0.7}

    def set_hand(q_cmd_hand, T=0.7):
        hand_prev["q0"] = hand_prev["q"].clone() if hand_prev["q"] is not None else q_hand_open
        # start the ramp from the current *targets* of the active joints
        cur = q_hand_open.clone()
        for name in HAND_ACTIVE.values():
            cur[idx[name]] = targets[0, idx[name]]
        hand_prev["q0"] = cur
        hand_prev["q"] = q_cmd_hand
        hand_prev["t0"] = t
        hand_prev["T"] = T

    def apply_hand(_unused=None):
        a = min(1.0, (t - hand_prev["t0"]) / max(hand_prev["T"], 1e-3))
        a = 0.5 - 0.5 * math.cos(math.pi * a)
        q_cmd_hand = hand_prev["q0"] + a * (hand_prev["q"] - hand_prev["q0"])
        qa = robot.data.joint_pos[0]
        for name in HAND_ACTIVE.values():
            targets[0, idx[name]] = q_cmd_hand[idx[name]]
        for dname, (pname, mult) in DISTAL.items():
            if dname in idx:
                targets[0, idx[dname]] = min(mult * float(qa[idx[pname]]), UPPER["index"] * 1.155 if "thumb" not in dname else UPPER["thumb"])

    # recording
    rec = None
    if args.video:
        import subprocess
        import imageio_ffmpeg
        from PIL import Image, ImageDraw, ImageFont

        mp4 = log_dir / f"g1-{ts}{('-' + args.tag) if args.tag else ''}.mp4"
        ffm = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.Popen([ffm, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CAM_W}x{CAM_H}",
                                 "-r", str(args.video_fps), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                                 "-pix_fmt", "yuv420p", str(mp4)], stdin=subprocess.PIPE)
        font = ImageFont.load_default()

        def rec(text):
            for _ in range(2):
                sim.render()
            camera.update(dt)
            rgb = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
            img = Image.fromarray(np.ascontiguousarray(rgb))
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, CAM_W, 22], fill=(0, 0, 0))
            d.text((6, 5), text, fill=(255, 255, 255), font=font)
            proc.stdin.write(img.tobytes())

        def rec_close():
            proc.stdin.close()
            proc.wait()
            gif = mp4.with_suffix(".gif")
            subprocess.run([ffm, "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                            "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse", str(gif)], check=False)
            print(f"G1_VIDEO: {mp4} gif: {gif}", flush=True)
    capture_every = max(1, int(round(1.0 / (args.video_fps * dt))))

    def snapshot(label):
        if camera is None:
            return
        for _ in range(4):
            sim.render()
        camera.update(dt)
        from PIL import Image
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
        p = log_dir / f"g1-{ts}-{label}.png"
        Image.fromarray(np.ascontiguousarray(rgb)).save(p)
        print(f"G1_SNAPSHOT {label}: {p}", flush=True)

    def place_camera(mode):
        if camera is None:
            return
        pel = robot.data.root_pos_w[0].cpu().numpy()
        if mode == "wide":
            eye = pel + np.array([2.2, -2.4, 1.0])
            look = pel + np.array([0.6, 0.0, -0.1])
        else:  # close-up on the bottle from the robot's right-front
            look = bottle.data.root_pos_w[0].cpu().numpy() + np.array([0.0, 0.0, 0.12])
            eye = look + np.array([0.55, -0.75, 0.45])
        camera.set_world_poses_from_view(torch.tensor([eye.tolist()], device=sim.device, dtype=torch.float32),
                                         torch.tensor([look.tolist()], device=sim.device, dtype=torch.float32))

    # ------------------------------------------------------------------------------------------------------------------
    # state machine
    # ------------------------------------------------------------------------------------------------------------------
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.set_joint_position_target(targets)
    robot.write_data_to_sim()
    robot.reset()
    bottle.reset()
    if loco:
        loco.reset()
    place_camera("wide")
    snapshot("start")

    phase = "stand"
    if args.phase == "load":
        # collision geometry audit: how many collision prims each hand-related body carries
        import omni.usd
        from pxr import Usd, UsdPhysics
        stage = omni.usd.get_context().get_stage()
        coll = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/G1"), Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                p = prim
                while p and not p.HasAPI(UsdPhysics.RigidBodyAPI):
                    p = p.GetParent()
                key = p.GetName() if p else "?"
                coll[key] = coll.get(key, 0) + 1
        for k in sorted(coll):
            if any(s in k for s in (f"{side}_index", f"{side}_thumb", f"{side}_pinky", f"{side}_hand_base", f"{side}_wrist_yaw", "pelvis")):
                print(f"G1_COLLISION {k}: {coll[k]} collision prim(s)", flush=True)
        missing = [b for b in robot.body_names if (f"{side}_index" in b or f"{side}_thumb" in b) and b not in coll]
        print(f"G1_COLLISION bodies without collision: {missing}", flush=True)
        n_all = sum(coll.values())
        print(f"G1_COLLISION total collision prims under /World/G1: {n_all}; bodies with collision: {sorted(coll)[:12]} ...", flush=True)
        # dump the subtree of one finger link and of the palm
        for name in (f"{side}_index_distal_link", f"{side}_hand_base_link", "left_ankle_roll_link"):
            for prim in stage.Traverse():
                if prim.GetName() == name and str(prim.GetPath()).startswith("/World/G1"):
                    kids = [(c.GetName(), c.GetTypeName(), c.HasAPI(UsdPhysics.CollisionAPI), c.IsInstanceProxy()) for c in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)) if c != prim]
                    print(f"G1_SUBTREE {prim.GetPath()}: {kids[:12]}", flush=True)
                    break
        print("G1_LOAD_OK", flush=True)
        return
    if args.phase == "ikcheck":
        # finite-difference check of the analytic Jacobian used by the IK (fixed base recommended)
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        for _ in range(4):
            sim.step(render=False)
            robot.update(dt)
        q0 = robot.data.joint_pos.clone()
        p0, qq0 = hand_pose_b()
        # frame consistency: hand base orientation relative to the wrist yaw link should equal the URDF mount rotation
        wrist_idx = robot.body_names.index(f"{side}_wrist_yaw_link")
        Rw = matrix_from_quat(robot.data.body_quat_w[0:1, wrist_idx])[0]
        Rh = matrix_from_quat(robot.data.body_quat_w[0:1, hand_body_idx])[0]
        R_rel = Rw.T @ Rh
        print(f"IKCHECK wrist->hand rotation from sim:\n{np.round(R_rel.cpu().numpy(), 3)}\nURDF mount rotation:\n{np.round(R_mount.cpu().numpy(), 3)}", flush=True)
        knuckles = [robot.body_names.index(f"{side}_{f}_proximal_link") for f in ("index", "middle", "ring", "pinky")]
        pk = robot.data.body_pos_w[0, knuckles].mean(0)
        pc = robot.data.body_pos_w[0, hand_body_idx] + Rh @ palm_h
        print(f"IKCHECK palm centre from knuckle bodies {np.round(pk.cpu().numpy(),4)} vs hand_base+palm_h {np.round(pc.cpu().numpy(),4)}", flush=True)
        J = jacobian_b()[0].cpu().numpy()
        print(f"IKCHECK jacobian shape {J.shape}; hand body idx {hand_body_idx} jacobi body {jacobi_body}; arm ids {arm_ids}", flush=True)
        # closed-loop IK sanity: (a) +5 cm in x keeping orientation, (b) rotate to the mount orientation in place
        ik_pos = DifferentialIKController(DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls",
                                                                      ik_params={"lambda_val": 0.05}), num_envs=1, device=sim.device)
        for stage, (dpos, use_mount, pos_only) in enumerate(((torch.tensor([0.05, 0.0, 0.0], device=sim.device), False, True),
                                                             (torch.tensor([0.05, 0.0, 0.0], device=sim.device), False, False),
                                                             (torch.tensor([0.0, 0.0, 0.0], device=sim.device), True, False))):
            p_s, q_s = hand_pose_b()
            pos_t = p_s[0] + dpos
            quat_t = quat_from_matrix(R_hand_des[None])[0] if use_mount else q_s[0]
            for i in range(150):  # 3 s at 50 Hz
                p_hb, q_hb = hand_pose_b()
                q_arm = robot.data.joint_pos[0:1, arm_ids_t]
                if pos_only:
                    ik_pos.set_command(pos_t[None], ee_quat=q_hb)
                    q_des = ik_pos.compute(p_hb, q_hb, jacobian_b()[:, :3, :], q_arm)
                else:
                    ik.set_command(torch.cat([pos_t, quat_t])[None])
                    q_des = ik.compute(p_hb, q_hb, jacobian_b(), q_arm)
                dq = torch.clip(IK_GAIN * (q_des - q_arm), -0.05, 0.05)
                tg = targets.clone()
                tg[0, arm_ids_t] = targets[0, arm_ids_t] + dq[0]
                targets[0, arm_ids_t] = tg[0, arm_ids_t]
                robot.set_joint_position_target(tg)
                robot.write_data_to_sim()
                for _ in range(4):
                    sim.step(render=False)
                    robot.update(dt)
                if i % 25 == 0 or i == 149:
                    perr = float(torch.linalg.norm(pos_t - p_hb[0]))
                    rerr = float(quat_error_magnitude(quat_t[None], q_hb)[0])
                    qm = np.round(robot.data.joint_pos[0, arm_ids_t].cpu().numpy(), 2).tolist()
                    qt = np.round(targets[0, arm_ids_t].cpu().numpy(), 2).tolist()
                    dqn = np.round((q_des - q_arm)[0].cpu().numpy(), 3).tolist()
                    print(f"IKTEST stage {stage} i={i:3d} pos err {perr*1000:6.1f} mm rot err {math.degrees(rerr):6.1f} deg | q_meas={qm} q_target={qt} newton_dq={dqn}", flush=True)
        for k, jid in enumerate(arm_ids):
            q = q0.clone()
            q[0, jid] += 0.05
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            p1, qq1 = hand_pose_b()
            dp = (p1[0] - p0[0]).cpu().numpy()
            pred = J[:3, k] * 0.05
            R0 = matrix_from_quat(qq0)[0]; R1 = matrix_from_quat(qq1)[0]
            dR = (R1 @ R0.T).cpu().numpy()
            ang = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))
            axis = np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]]) / (2*math.sin(ang)) if ang > 1e-6 else np.zeros(3)
            print(f"  {jn[jid]:28s} dpos actual {np.round(dp*1000,1)} mm  jacobian {np.round(pred*1000,1)} mm | drot actual {np.round(axis*ang,3)} rad  jacobian {np.round(J[3:, k]*0.05,3)} rad", flush=True)
            robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        return
    if args.fixed_base or args.phase == "grasp":
        phase = "settle"
    command = [0.0, 0.0, 0.0, 0.8]
    t = 0.0
    step = 0
    phase_t0 = 0.0
    hand_cmd = q_hand_open
    palm_start = None
    palm_goal = None
    quat_start = None  # hand-base orientation at the start of the current move (pelvis frame), slerped to R_hand_des
    quat_des_final = quat_from_matrix(R_hand_des[None])[0]
    move_T = 1.0
    result = {"phase_log": [], "fixed_base": args.fixed_base, "bottle": args.bottle, "hand": side}
    lift_ref_z = None
    grasp_t0 = None

    def log_phase(name):
        nonlocal phase, phase_t0
        print(f"G1_PHASE {name} at t={t:.2f}s pelvis=({float(robot.data.root_pos_w[0,0]):.2f},{float(robot.data.root_pos_w[0,1]):.2f},{float(robot.data.root_pos_w[0,2]):.2f})", flush=True)
        result["phase_log"].append({"phase": name, "t": t})
        phase, phase_t0 = name, t

    def bottle_rel_b():
        """bottle base position in the pelvis frame."""
        bp = bottle.data.root_pos_w[0:1]
        bq = bottle.data.root_quat_w[0:1]
        p, _ = subtract_frame_transforms(robot.data.root_pos_w[0:1], robot.data.root_quat_w[0:1], bp, bq)
        return p[0]

    def grasp_palm_target_b(gap, lift=0.0):
        """palm-centre target in the pelvis frame for the pocket geometry (approach from the robot's right)."""
        b = bottle_rel_b()
        sgn = 1.0 if side == "right" else -1.0  # palm normal points toward the midline
        return torch.tensor([float(b[0]) - args.distal_offset, float(b[1]) - sgn * (r_grasp + gap),
                             float(b[2]) + args.grasp_height + lift], device=sim.device)

    ee_err_log = []
    held = None
    while app.is_running() and t < args.max_seconds:
        control_tick = step % decim == 0
        if control_tick:
            # ---------------- phase logic (50 Hz) ----------------
            pel = robot.data.root_pos_w[0]
            if not args.fixed_base and float(pel[2]) < 0.45:
                print(f"G1_FALLEN at t={t:.2f}s (pelvis z={float(pel[2]):.2f})", flush=True)
                result["fallen"] = t
                break
            if phase == "stand":
                command = [0.0, 0.0, 0.0, 0.8]
                if t - phase_t0 > 2.0:
                    if args.phase == "stand":
                        break
                    log_phase("walk")
            elif phase == "walk":
                dist = float(bottle.data.root_pos_w[0, 0]) - float(pel[0])
                y_line = float(bottle.data.root_pos_w[0, 1]) - args.bottle_y
                q = robot.data.root_quat_w[0]
                yaw = math.atan2(2.0 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])), 1.0 - 2.0 * (float(q[2]) ** 2 + float(q[3]) ** 2))
                vy = max(-0.25, min(0.25, 1.2 * (y_line - float(pel[1]))))
                wz = max(-0.4, min(0.4, -1.5 * yaw))
                command = [args.walk_speed, vy, wz, 0.8]
                if dist <= args.reach_x + args.stop_lead:
                    command = [0.0, 0.0, 0.0, 0.8]
                    log_phase("settle")
            elif phase == "settle":
                command = [0.0, 0.0, 0.0, 0.8]
                if t - phase_t0 > 1.5:
                    dist = float(bottle.data.root_pos_w[0, 0]) - float(pel[0])
                    print(f"G1_STOPPED bottle {dist:.2f} m ahead, {float(bottle.data.root_pos_w[0,1]) - float(pel[1]):+.2f} m lateral", flush=True)
                    result["stop_distance_m"] = dist
                    if args.phase == "walk":
                        break
                    place_camera("close")
                    set_hand(q_hand_pre, 0.5)  # thumb into opposition before the approach
                    palm_start = hand_pose_b()[0][0] + torch.tensor([0.0, 0.0, 0.0], device=sim.device)
                    # current palm centre (pelvis frame)
                    p_hb, q_hb = hand_pose_b()
                    palm_start = p_hb[0] + matrix_from_quat(q_hb)[0] @ palm_h
                    # 1) raise the hand above the table plane where it is (arms hang ~17 cm below the table top)
                    up = grasp_palm_target_b(gap=args.gap + 0.08)
                    palm_goal = torch.stack([palm_start[0], palm_start[1] - 0.05, up[2] + 0.05])
                    quat_start = q_hb[0].clone()
                    move_T = 1.5 * args.arm_time_scale
                    if args.arm_gains != "sim":
                        # the real arm runs on whatever kp/kd we put in rt/arm_sdk (measured on the robot, docs/robot/):
                        # Unitree's teleop uses 300/3 on shoulder+elbow and 40/1.5 on the wrists; the minimal example 60/1.5
                        if args.arm_gains == "teleop":
                            kp_l = [40.0 if "wrist" in n else 300.0 for n in SIDE_ARM]
                            kd_l = [1.5 if "wrist" in n else 3.0 for n in SIDE_ARM]
                        elif args.arm_gains == "grasp":
                            kp_l = [120.0] * len(SIDE_ARM)
                            kd_l = [3.0] * len(SIDE_ARM)
                        else:
                            kp_l = [args.arm_sdk_kp] * len(SIDE_ARM)
                            kd_l = [args.arm_sdk_kd] * len(SIDE_ARM)
                        if args.wrist_kp is not None:
                            kp_l = [args.wrist_kp if "wrist" in n else k for n, k in zip(SIDE_ARM, kp_l)]
                        if args.wrist_kd is not None:
                            kd_l = [args.wrist_kd if "wrist" in n else k for n, k in zip(SIDE_ARM, kd_l)]
                        kp = torch.tensor([kp_l], device=sim.device)
                        kd = torch.tensor([kd_l], device=sim.device)
                        robot.write_joint_stiffness_to_sim(kp, joint_ids=arm_ids)
                        robot.write_joint_damping_to_sim(kd, joint_ids=arm_ids)
                        print(f"G1_ARM_GAINS {args.arm_gains}: kp={kp_l} kd={kd_l} on {side} arm {SIDE_ARM}", flush=True)
                    log_phase("raise")
            elif phase == "raise":
                if t - phase_t0 > move_T + 0.5:
                    # 2) out to the pre-grasp pose beside the bottle (8 cm off the palm normal), above table height
                    palm_start = palm_goal
                    palm_goal = grasp_palm_target_b(gap=args.gap + 0.08)
                    move_T = 2.0 * args.arm_time_scale
                    log_phase("pregrasp")
            elif phase == "pregrasp":
                if t - phase_t0 > move_T + 0.8:
                    palm_start = palm_goal
                    palm_goal = grasp_palm_target_b(gap=args.gap - args.press)  # palm-contact approach
                    move_T = 1.5 * args.arm_time_scale
                    if args.arm_kp_scale != 1.0:
                        kp = robot.data.joint_stiffness[0:1, arm_ids_t] * args.arm_kp_scale
                        kd = robot.data.joint_damping[0:1, arm_ids_t] * args.arm_kp_scale
                        robot.write_joint_stiffness_to_sim(kp, joint_ids=arm_ids)
                        robot.write_joint_damping_to_sim(kd, joint_ids=arm_ids)
                        print(f"G1_ARM_GAINS scaled x{args.arm_kp_scale}: kp={np.round(kp[0].cpu().numpy(),1).tolist()}", flush=True)
                    log_phase("approach")
            elif phase == "approach":
                if t - phase_t0 > move_T + 0.6:
                    set_hand(q_hand_close, 0.7)
                    grasp_t0 = t
                    palm_start = palm_goal  # hold the grasp pose while the fingers close
                    log_phase("close")
            elif phase == "close":
                if t - phase_t0 > 1.4:
                    lift_ref_z = float(bottle.data.root_pos_w[0, 2])
                    palm_start = palm_goal
                    sgn = 1.0 if side == "right" else -1.0
                    # lift; ease the palm press off by (press + 3 mm) so the bottle is not dragged along the insert wall
                    palm_goal = palm_goal + torch.tensor([0.0, -sgn * (args.press + 0.003), args.lift], device=sim.device)
                    move_T = 1.2
                    log_phase("lift")
            elif phase == "lift":
                if t - phase_t0 > move_T + 3.0:
                    bz = float(bottle.data.root_pos_w[0, 2])
                    held = bool(bz - lift_ref_z > 0.6 * args.lift)
                    result["result"] = {"held": held, "bottle_rise_m": bz - lift_ref_z, "lift_cmd_m": args.lift,
                                        "final_arm_targets": targets[0, arm_ids].cpu().tolist(),
                                        "hand_joint_pos": robot.data.joint_pos[0, hand_ids].cpu().tolist()}
                    print(f"G1_GRASP_RESULT held={held} bottle_rise={100*(bz-lift_ref_z):.1f}cm of {100*args.lift:.0f}cm commanded", flush=True)
                    palm_start = palm_goal
                    log_phase("done")
            elif phase == "done":
                if t - phase_t0 > 1.0:
                    break

            # ---------------- arm IK toward the interpolated palm goal ----------------
            if palm_goal is not None:
                a = min(1.0, (t - phase_t0) / move_T)
                a = 0.5 - 0.5 * math.cos(math.pi * a)
                palm_des = palm_start + a * (palm_goal - palm_start)
                if quat_start is not None and phase == "raise":
                    quat_des = quat_slerp(quat_start, quat_des_final, a)
                else:
                    quat_des = quat_des_final
                R_des = matrix_from_quat(quat_des[None])[0]
                pos_des = palm_des - R_des @ palm_h
                p_hb, q_hb = hand_pose_b()
                ik.set_command(torch.cat([pos_des, quat_des])[None])
                q_arm = robot.data.joint_pos[0:1, arm_ids_t]
                q_des = ik.compute(p_hb, q_hb, jacobian_b(), q_arm)
                # integrate the *commanded* target with the IK step (the PD arm sags under gravity, so re-basing on the
                # measured angles would leave a permanent error); limit per-tick motion and clamp to soft limits
                dq = torch.clip(IK_GAIN * (q_des - q_arm), -0.05, 0.05)  # fractional Newton step: the PD arm lags the target
                q_new = targets[0:1, arm_ids_t] + dq
                q_new = torch.minimum(torch.maximum(q_new, q_arm - 0.35), q_arm + 0.35)  # anti-windup vs. blocked joints
                lim = robot.data.soft_joint_pos_limits[0, arm_ids_t]
                q_new = torch.minimum(torch.maximum(q_new, lim[:, 0]), lim[:, 1])
                targets[0, arm_ids_t] = q_new[0]
                if loco:
                    loco.arm_targets = targets[0, loco.arm_ids].clone()
                err = float(torch.linalg.norm(palm_des - (p_hb[0] + matrix_from_quat(q_hb)[0] @ palm_h)))
                rot_err = float(quat_error_magnitude(quat_des[None], q_hb)[0])
                if step % (decim * 25) == 0:
                    qa_ = q_new[0].cpu().numpy(); lo_ = lim[:, 0].cpu().numpy(); hi_ = lim[:, 1].cpu().numpy()
                    sat = [SIDE_ARM[k].replace(f"{side}_", "").replace("_joint", "") + ("↑" if qa_[k] >= hi_[k] - 1e-3 else "↓")
                           for k in range(len(SIDE_ARM)) if qa_[k] >= hi_[k] - 1e-3 or qa_[k] <= lo_[k] + 1e-3]
                    Rh_ = matrix_from_quat(q_hb)[0]
                    ex_ = np.round((Rh_ @ torch.tensor([1.0, 0, 0], device=sim.device)).cpu().numpy(), 2)  # hand +x (palm normal)
                    ez_ = np.round((Rh_ @ torch.tensor([0, 0, 1.0], device=sim.device)).cpu().numpy(), 2)  # hand +z (fingers)
                    b_ = np.round(bottle_rel_b().cpu().numpy(), 3); pc_ = np.round((p_hb[0] + Rh_ @ palm_h).cpu().numpy(), 3)
                    print(f"  t={t:5.2f}s {phase:9s} palm err {err*1000:5.1f} mm rot err {math.degrees(rot_err):5.1f} deg | palm_normal(pelvis)={ex_.tolist()} fingers={ez_.tolist()} palm_c={pc_.tolist()} bottle_base={b_.tolist()} limits={sat}", flush=True)

            if phase in ("approach", "close") and step % (decim * 10) == 0:
                names_ = [f"{side}_index_proximal_link", f"{side}_pinky_proximal_link", f"{side}_index_distal_link", f"{side}_thumb_distal_link"]
                ids_ = [robot.body_names.index(n_) for n_ in names_ if n_ in robot.body_names]
                pos_ = robot.data.body_pos_w[0, ids_].cpu().numpy()
                bb_ = bottle.data.root_pos_w[0].cpu().numpy()
                print(f"  DBG t={t:5.2f} {phase} knuckle_index={np.round(pos_[0],3).tolist()} knuckle_pinky={np.round(pos_[1],3).tolist()} index_distal={np.round(pos_[2],3).tolist()} thumb_distal={np.round(pos_[3],3).tolist()} bottle_base={np.round(bb_,3).tolist()} q_index={float(robot.data.joint_pos[0, idx[HAND_ACTIVE['index']]]):.2f} q_thumb={float(robot.data.joint_pos[0, idx[HAND_ACTIVE['thumb']]]):.2f}", flush=True)
            # ---------------- legs (policy) or fixed ----------------
            if loco is not None:
                targets[0] = loco.step(command, targets[0])
        apply_hand()  # every physics step: distal joints track 1.155 x the measured proximal (rigid linkage)
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()

        sim.step(render=False)
        robot.update(dt)
        bottle.update(dt)
        step += 1
        t = step * dt
        if rec is not None and step % capture_every == 0:
            if phase in ("stand", "walk"):
                place_camera("wide")
            rec(f"G1 + Revo 2 {side} | {args.bottle} | t={t:5.2f}s {phase} | cmd vx={command[0]:.2f}")
        if step % (decim * 50) == 0 and phase in ("stand", "walk"):
            pel = robot.data.root_pos_w[0]
            legv = float(robot.data.joint_vel[0, loco.leg_ids].abs().mean()) if loco else 0.0
            print(f"  t={t:5.2f}s {phase:6s} pelvis x={float(pel[0]):.2f} z={float(pel[2]):.2f} vx={float(robot.data.root_lin_vel_b[0,0]):+.2f} cmd={command[:3]} leg|qd|={legv:.2f}", flush=True)
        if not torch.isfinite(robot.data.joint_pos).all():
            raise RuntimeError("non-finite joint state")

    snapshot("end")
    if rec is not None:
        for _ in range(args.video_fps):
            rec(f"G1 + Revo 2 | result: {'HELD' if held else ('not held' if held is not None else phase)}")
        rec_close()
    out = log_dir / f"g1-{ts}{('-' + args.tag) if args.tag else ''}.json"
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"G1_REPORT: {out}", flush=True)


if __name__ == "__main__":
    import threading

    rc = 1
    try:
        main()
        rc = 0
    except BaseException as exc:
        import traceback

        print(f"G1_ABORT: {exc!r}", flush=True)
        traceback.print_exc()
    finally:
        timer = threading.Timer(20.0, lambda: os._exit(3))
        timer.daemon = True
        timer.start()
        try:
            app.close()
        finally:
            sys.stdout.flush()
            os._exit(rc)
