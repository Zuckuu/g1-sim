"""Standalone BrainCo Revo 2 hand + bottle-proxy grasp test in Isaac Sim / Isaac Lab.

What it does
  1. Converts the BrainCo Revo 2 URDF (from Unitree's xr_teleoperate assets) to USD with the Isaac URDF importer.
  2. Spawns the hand fixed in space, palm facing +X, index finger up, fingers pointing -Y when open.
  3. Stands a bottle (dimensioned PET bottle mesh, or a plain cylinder) on a kinematic table in front of the palm.
  4. Ramps the six active joints closed (distal joints follow the URDF mimic ratios).
  5. Lowers the table 15 cm and checks whether the bottle stays in the hand for several seconds.
  6. Prints REVO2_GRASP_RESULT and writes a JSON report (+ optional PNG snapshots) under work/g1-runtime/logs.

This is a physics/geometry sandbox for the hand-bottle interaction only. No G1 body, no DDS, no headset.
Bottle presets use public dimensions (500 mL: 231x66 mm 0.525 kg; 20 oz: 222x72.8 mm 0.64 kg) until the real
demo bottle is measured or scanned (--bottle-mesh takes any OBJ/STL with its origin at the base centre, +Z up).
"""

import argparse
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pepsidemo_sim as pd  # noqa: E402

project_root = pd.PROJECT_ROOT
usd_out_dir = pd.USD_CACHE / "brainco_hand"
log_dir = pd.LOG_DIR
MODELS = {
    # official BrainCo description repo (URDF with separate collision meshes, and a ready USD)
    "official-urdf": project_root / "work/brainco-description/revo2_system/urdf/revo2_{side}.urdf",
    "official-usd": project_root / "work/brainco-description/revo2_system/usd/revo2_{side}.usd",
    # URDF shipped with Unitree's xr_teleoperate (older export, extra base rotation, visual meshes only)
    "xr-urdf": project_root / "work/xr_teleoperate/assets/brainco_hand/brainco_{side}.urdf",
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--hand", choices=["left", "right"], default="right")
parser.add_argument("--grasp-mode", choices=["side", "top"], default="side",
                    help="side: hand vertical beside a standing bottle (palm normal horizontal); top: palm facing down over the bottle, fingers hook under the shoulder/neck")
parser.add_argument("--model", choices=list(MODELS), default="official-urdf")
parser.add_argument("--model-urdf", type=str, default="", help="explicit hand URDF path (overrides --model)")
parser.add_argument("--raw-urdf", action="store_true", help="import the URDF as-is. Default folds the empty fingertip frames and 2 g touch pads into the distal links; otherwise the Isaac importer gives each empty tip link a fake 1.0 kg mass (5 kg per hand) which made early grasps look far better than physics allows")
parser.add_argument("--bottle", choices=["pepsi-500ml", "pepsi-20oz", "cylinder"], default="pepsi-500ml",
                    help="bottle model: generated PET bottle presets, or the plain cylinder proxy")
parser.add_argument("--bottle-mesh", type=str, default="", help="OBJ/STL of a real bottle (metres, origin at base centre, +Z up); overrides --bottle")
parser.add_argument("--grasp-height", type=float, default=0.10, help="m above the bottle base where the palm centre sits (mesh bottles); cylinder default = half height")
parser.add_argument("--bottle-radius", type=float, default=0.0325, help="cylinder proxy only: m; 0.0325 ~ 500 mL PET body, 0.0365 ~ 20 oz")
parser.add_argument("--bottle-height", type=float, default=0.22, help="cylinder proxy only: m")
parser.add_argument("--bottle-mass", type=float, default=None, help="kg; default: preset mass (500 mL 0.525, 20 oz 0.64, cylinder 0.55)")
parser.add_argument("--bottle-friction", type=float, default=0.9, help="static/dynamic friction of the bottle material")
parser.add_argument("--gap", type=float, default=0.028, help="m from the knuckle axis line to the bottle surface along the palm normal; the palm face itself is ~0.023 m out")
parser.add_argument("--distal-offset", type=float, default=0.035, help="m from knuckle line toward fingertips (bottle axis)")
parser.add_argument("--vertical-offset", type=float, default=0.0, help="m; shift the bottle up (+) or down (-) relative to the nominal grasp height")
parser.add_argument("--close-fingers", type=float, default=0.9, help="fraction of finger flexion range to command")
parser.add_argument("--close-thumb", type=float, default=0.75, help="fraction of thumb flexion range to command")
parser.add_argument("--thumb-aux", type=float, default=0.9, help="fraction of thumb rotation (opposition) range to command")
parser.add_argument("--stiffness", type=float, default=6.0, help="finger drive stiffness Nm/rad")
parser.add_argument("--finger-effort", type=float, default=1.5, help="finger drive torque limit Nm (real hand: current-limited)")
parser.add_argument("--thumb-lead", type=float, default=0.0, help="s: thumb flexes this long before the fingers start closing")
parser.add_argument("--damping", type=float, default=0.3, help="finger drive damping Nm/(rad/s)")
parser.add_argument("--lower-by", type=float, default=0.15, help="m to lower the support table after the grasp")
parser.add_argument("--close-seconds", type=float, default=0.7, help="duration of the finger closing ramp (real hand: full range in <= 0.65 s)")
parser.add_argument("--hold-seconds", type=float, default=3.0)
parser.add_argument("--max-steps", type=int, default=0, help="0 = keep running after the test until the window closes")
parser.add_argument("--snapshot", action="store_true", help="save PNG snapshots (forces camera rendering)")
parser.add_argument("--reconvert", action="store_true", help="force URDF -> USD reconversion")
parser.add_argument("--merge-fixed", action="store_true", help="merge fixed joints on import (as the full-robot import does)")
parser.add_argument("--dt", type=float, default=1.0/240.0, help="physics step")
parser.add_argument("--static-table", action="store_true", help="A/B: static collider table at the first trial position instead of a kinematic body (no drop test)")
parser.add_argument("--collider", choices=["convex_hull", "convex_decomposition"], default="convex_decomposition",
                    help="collision approximation for the hand meshes (decomposition keeps the palm pocket concave)")
parser.add_argument("--tag", type=str, default="", help="label stored in the JSON report")
parser.add_argument("--sweep-gap", type=str, default="", help="comma list of gaps (m) to try in one process")
parser.add_argument("--sweep-distal", type=str, default="", help="comma list of distal offsets (m)")
parser.add_argument("--sweep-vertical", type=str, default="", help="comma list of vertical offsets (m)")
parser.add_argument("--snapshot-all", action="store_true", help="snapshot every sweep trial, not just the best")
parser.add_argument("--video", action="store_true", help="record an MP4 (+ small GIF) of the trial / of the best sweep trial")
parser.add_argument("--video-all", action="store_true", help="record every sweep trial")
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--video-size", type=str, default="960x540", help="camera resolution WxH for video/snapshots")
parser.add_argument("--log-joints", action="store_true", help="write a per-step CSV (joints, bottle pose) for each trial")
parser.add_argument("--coupling", choices=["linkage", "independent"], default="linkage",
                    help="distal joint law: 'linkage' tracks 1.155 x the measured proximal angle (rigid 4-bar coupling of the real "
                         "Revo 2), 'independent' drives the distal to 1.155 x the proximal *target* (lets fingertips claw around an object)")

# Isaac Lab launcher args (--headless, --device, ...). The experience file is resolved by Isaac Lab from its own
# apps/ directory (headless / rendering variants); the plain isaacsim experience lacks Lab's render presets.
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
# PhysX GPU kernels do not load on this laptop's RTX 5060 / driver 580 combination ("Could not find CUDA module"),
# so physics runs on the CPU by default; a single hand + bottle is cheap. Override with --device cuda:0 to retry GPU.
parser.set_defaults(device="cpu", rendering_mode="performance")
args = parser.parse_args()
if args.video:
    args.snapshot = True
if args.snapshot:
    args.enable_cameras = True
CAM_W, CAM_H = (int(v) for v in args.video_size.lower().split("x"))
pd.ensure_dirs()
pd.select_experience(args)

# ----------------------------------------------------------------------------------------------------------------------
# URDF forward kinematics at q=0 (pure numpy) to find the palm frame before the simulator starts.
# ----------------------------------------------------------------------------------------------------------------------


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def parse_urdf(path):
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        ax = j.find("axis")
        axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "0 0 1").split()])
        lim = j.find("limit")
        joints[j.get("name")] = {
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": xyz,
            "R": rpy_to_mat(*rpy),
            "axis": axis,
            "lower": float(lim.get("lower", 0)) if lim is not None else 0.0,
            "upper": float(lim.get("upper", 0)) if lim is not None else 0.0,
        }
    child_to_joint = {jd["child"]: name for name, jd in joints.items()}
    links = [l.get("name") for l in root.findall("link")]
    return joints, child_to_joint, links


def link_pose_at_zero(link, joints, child_to_joint):
    """Return (R, p) of `link` in the URDF root link frame at q=0."""
    R = np.eye(3)
    p = np.zeros(3)
    chain = []
    while link in child_to_joint:
        j = joints[child_to_joint[link]]
        chain.append(j)
        link = j["parent"]
    for j in reversed(chain):
        p = p + R @ j["xyz"]
        R = R @ j["R"]
    return R, p


def mat_to_quat_wxyz(R):
    m = R
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


side = args.hand
model_path = Path(args.model_urdf).resolve() if args.model_urdf else Path(str(MODELS[args.model]).format(side=side))
if model_path.suffix == ".urdf" and not args.raw_urdf and not args.model_urdf:
    # fold fingertip / touch-pad links into the distal links (see --raw-urdf) using the builder's helper
    import build_g1_revo2_urdf as bld
    _root = ET.parse(model_path).getroot()
    bld.abspath_meshes(_root, model_path.parent)
    bld.fold_fixed_children(_root, side)
    _folded = pd.RUNTIME / "urdf" / f"{model_path.stem}_{args.model}_folded.urdf"
    _folded.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(_root).write(_folded, encoding="unicode", xml_declaration=True)
    fk_urdf_path = model_path  # frames (incl. fingertips) come from the original
    model_path = _folded
else:
    fk_urdf_path = model_path
if not model_path.is_file():
    raise SystemExit(f"Missing model file: {model_path}")
# The palm frame is always derived from a URDF (for the USD model we use the matching official URDF).
urdf_path = fk_urdf_path if (fk_urdf_path.suffix == ".urdf" and not args.model_urdf) else Path(str(MODELS["official-urdf"]).format(side=side))
if not urdf_path.is_file():
    raise SystemExit(f"Missing URDF for kinematics: {urdf_path}")
joints, child_to_joint, link_names = parse_urdf(urdf_path)
# Link naming differs in case between the two URDFs ("_link" vs "_Link").
def L(stem):
    for cand in (f"{side}_{stem}_link", f"{side}_{stem}_Link", f"{side}_{stem}"):
        if cand in link_names:
            return cand
    raise KeyError(stem)

J = {
    "index": f"{side}_index_proximal_joint", "middle": f"{side}_middle_proximal_joint",
    "ring": f"{side}_ring_proximal_joint", "pinky": f"{side}_pinky_proximal_joint",
    "thumb_aux": f"{side}_thumb_metacarpal_joint", "thumb": f"{side}_thumb_proximal_joint",
}
DISTAL = {
    f"{side}_index_distal_joint": (J["index"], 1.155), f"{side}_middle_distal_joint": (J["middle"], 1.155),
    f"{side}_ring_distal_joint": (J["ring"], 1.155), f"{side}_pinky_distal_joint": (J["pinky"], 1.155),
    f"{side}_thumb_distal_joint": (J["thumb"], 1.0),
}
for name in list(J.values()) + list(DISTAL):
    if name not in joints:
        raise SystemExit(f"URDF joint not found: {name}")

# Knuckle (proximal joint) origins and the index finger tip at q=0, in the URDF root (base_link) frame.
knuckles = {}
for f in ("index", "middle", "ring", "pinky"):
    _, p_child = link_pose_at_zero(joints[J[f]]["child"], joints, child_to_joint)
    knuckles[f] = p_child
R_idx, p_idx = link_pose_at_zero(joints[J["index"]]["child"], joints, child_to_joint)
_, p_idx_tip = link_pose_at_zero(L("index_tip"), joints, child_to_joint)
_, p_thumb_tip = link_pose_at_zero(L("thumb_tip"), joints, child_to_joint)
palm_center_b = np.mean([knuckles[f] for f in knuckles], axis=0)
f_dir = p_idx_tip - p_idx
f_dir /= np.linalg.norm(f_dir)  # open-finger direction
w_raw = knuckles["pinky"] - knuckles["index"]  # across the palm, index -> pinky
w_dir = w_raw - np.dot(w_raw, f_dir) * f_dir
w_dir /= np.linalg.norm(w_dir)
# Flexion axis of the index proximal joint in base frame; +q flexes toward the palm-side, tip moves along axis x f.
axis_b = R_idx @ joints[J["index"]]["axis"]
n_dir = np.cross(axis_b, f_dir)
n_dir -= np.dot(n_dir, f_dir) * f_dir
n_dir -= np.dot(n_dir, w_dir) * w_dir
n_dir /= np.linalg.norm(n_dir)
if np.dot(np.cross(f_dir, w_dir), n_dir) < 0:
    # keep a right-handed (f, w, n) triad; flip w (it only defines up/down of the hand)
    w_dir = -w_dir
B = np.stack([f_dir, w_dir, n_dir], axis=1)  # columns: base-frame directions of f, w, n
# Target world directions: palm normal n -> +X (toward the bottle), w -> -Z (index up), f = w x n -> -Y.
if args.grasp_mode == "side":
    n_t = np.array([1.0, 0.0, 0.0])   # palm faces +X toward the bottle
    w_t = np.array([0.0, 0.0, -1.0])  # index above pinky
else:
    n_t = np.array([0.0, 0.0, -1.0])  # palm faces down onto the bottle top
    w_t = np.array([1.0, 0.0, 0.0])   # index -> pinky along +X (finger plane is the Y-Z plane)
f_t = np.cross(w_t, n_t)
T = np.stack([f_t, w_t, n_t], axis=1)
R_world_base = T @ B.T
hand_quat = mat_to_quat_wxyz(R_world_base)

# ----------------------------------------------------------------------------------------------------------------------
# Launch the simulator
# ----------------------------------------------------------------------------------------------------------------------
launcher = AppLauncher(
    args, width=960, height=640, window_width=1280, window_height=800,
    renderer="RayTracedLighting", anti_aliasing=0, multi_gpu=False,
)
app = launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg, UrdfConverter, UrdfConverterCfg  # noqa: E402

camera = None
if args.snapshot:
    from isaaclab.sensors import Camera, CameraCfg  # noqa: E402


def convert_urdf():
    if model_path.suffix != ".urdf":
        return str(model_path)  # official USD, use as-is
    usd_out_dir.mkdir(parents=True, exist_ok=True)
    cfg = UrdfConverterCfg(
        asset_path=str(model_path),
        usd_dir=str(usd_out_dir),
        usd_file_name=f"revo2_{side}_{model_path.stem}_{args.collider}{'_merged' if args.merge_fixed else ''}.usd",
        force_usd_conversion=args.reconvert,
        make_instanceable=False,
        fix_base=True,
        merge_fixed_joints=args.merge_fixed,
        link_density=0.0,
        collider_type=args.collider,
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force", target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=args.stiffness, damping=args.damping),
        ),
    )
    converter = UrdfConverter(cfg)
    return converter.usd_path


def resolve_bottle():
    """Return dict(kind, usd/obj paths, radius at grasp height, height, mass, anchor offset origin->base)."""
    if args.bottle_mesh:
        obj = Path(args.bottle_mesh).resolve()
        name = obj.stem
        mass = args.bottle_mass if args.bottle_mass is not None else 0.55
    elif args.bottle == "cylinder":
        h = args.bottle_height
        return dict(kind="cylinder", name="cylinder", radius=args.bottle_radius, height=h,
                    mass=args.bottle_mass if args.bottle_mass is not None else 0.55,
                    grasp_height=h / 2.0 if args.grasp_height == 0.10 else args.grasp_height, anchor=h / 2.0, obj=None)
    else:
        import make_bottle_mesh as mbm
        obj = mbm.OUT_DIR / f"{args.bottle}.obj"
        if not obj.is_file():
            mbm.build(args.bottle, **mbm.PRESETS[args.bottle])
        name = args.bottle
        mass = args.bottle_mass if args.bottle_mass is not None else mbm.PRESETS[args.bottle]["mass"]
    import trimesh
    mesh = trimesh.load(str(obj), force="mesh")
    zmin = float(mesh.bounds[0][2])
    height = float(mesh.bounds[1][2] - zmin)
    gh = args.grasp_height
    # radius of the bottle at the grasp height: slice the mesh with a horizontal plane
    section = mesh.section(plane_origin=[0.0, 0.0, zmin + gh], plane_normal=[0.0, 0.0, 1.0])
    if section is not None and len(section.vertices) > 0:
        pts = np.asarray(section.vertices)
    else:
        pts = mesh.vertices
    radius = float(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2).max())
    return dict(kind="mesh", name=name, radius=radius, height=height, mass=mass, grasp_height=gh, anchor=-zmin, obj=str(obj))


def parse_sweep(text, default):
    if not text:
        return [default]
    return [float(v) for v in text.split(",")]


def main():
    t0 = time.time()
    usd_path = convert_urdf()
    print(f"REVO2_USD: {usd_path} ({time.time() - t0:.1f}s)", flush=True)

    dt = args.dt
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=dt, render_interval=4))

    # World
    ground = sim_utils.GroundPlaneCfg(color=(0.15, 0.16, 0.18))
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95))
    light.func("/World/Light", light)

    # Hand placement: palm center (mean of the four knuckle axes) at a fixed world point.
    palm_world = np.array([0.0, 0.0, 1.0])
    base_pos = palm_world - R_world_base @ palm_center_b
    hand = Articulation(ArticulationCfg(
        prim_path="/World/Hand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=2,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(base_pos.tolist()), rot=tuple(hand_quat.tolist())),
        actuators={
            "active": ImplicitActuatorCfg(
                joint_names_expr=[".*_proximal_joint", ".*_metacarpal_joint"],
                stiffness=args.stiffness, damping=args.damping, effort_limit_sim=args.finger_effort,
            ),
            "distal": ImplicitActuatorCfg(
                joint_names_expr=[".*_distal_joint"],
                stiffness=args.stiffness, damping=args.damping, effort_limit_sim=args.finger_effort,
            ),
        },
    ))

    # Bottle standing on a kinematic table in front of the palm. Poses are (re)set per trial.
    B = resolve_bottle()
    r, h = B["radius"], B["height"]
    print(f"REVO2_BOTTLE {B['name']} ({B['kind']}): height {h*1000:.0f} mm, radius at grasp height "
          f"{B['grasp_height']*1000:.0f} mm = {r*1000:.1f} mm, mass {B['mass']:.3f} kg", flush=True)
    table_thickness = 0.04
    if args.static_table:
        g0 = parse_sweep(args.sweep_gap, args.gap)[0]; d0 = parse_sweep(args.sweep_distal, args.distal_offset)[0]
        v0 = parse_sweep(args.sweep_vertical, args.vertical_offset)[0]
        axis0 = palm_world[:2] + n_t[:2] * (r + g0) + f_t[:2] * d0
        top0 = palm_world[2] - B["grasp_height"] + v0
        st = sim_utils.CuboidCfg(size=(0.6, 0.6, table_thickness), collision_props=sim_utils.CollisionPropertiesCfg(),
                                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.38, 0.30)))
        st.func("/World/StaticTable", st, translation=(float(axis0[0]) + 0.1, float(axis0[1]), top0 - table_thickness / 2))
    table = RigidObject(RigidObjectCfg(
        prim_path="/World/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.5)),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, table_thickness),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.38, 0.30)),
        ),
    ))
    bottle_rigid = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16, solver_velocity_iteration_count=2, max_depenetration_velocity=1.0,
    )
    bottle_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)
    bottle_mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=args.bottle_friction, dynamic_friction=args.bottle_friction,
        friction_combine_mode="average", restitution=0.0,
    )
    if B["kind"] == "cylinder":
        bottle_spawn = sim_utils.CylinderCfg(
            radius=r, height=h, rigid_props=bottle_rigid, mass_props=sim_utils.MassPropertiesCfg(mass=B["mass"]),
            collision_props=bottle_coll, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.25, 0.75)),
            physics_material=bottle_mat,
        )
    else:
        bottle_usd = MeshConverter(MeshConverterCfg(
            asset_path=B["obj"], usd_dir=str(pd.USD_CACHE / "bottles"), usd_file_name=f"{B['name']}.usd",
            force_usd_conversion=args.reconvert, make_instanceable=False, collision_approximation="convexDecomposition",
            mass_props=sim_utils.MassPropertiesCfg(mass=B["mass"]), rigid_props=bottle_rigid, collision_props=bottle_coll,
        )).usd_path
        print(f"REVO2_BOTTLE_USD: {bottle_usd}", flush=True)
        bottle_spawn = sim_utils.UsdFileCfg(
            usd_path=bottle_usd, rigid_props=bottle_rigid, mass_props=sim_utils.MassPropertiesCfg(mass=B["mass"]),
            collision_props=bottle_coll,
            # (OmniGlass renders invisible in RTX real-time mode; keep an opaque colour and film from behind the hand)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.25, 0.75), roughness=0.35),
        )
    bottle = RigidObject(RigidObjectCfg(
        prim_path="/World/Bottle", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 1.0)), spawn=bottle_spawn,
    ))
    if B["kind"] != "cylinder":
        # UsdFileCfg has no physics_material field: create the material and bind it to the whole bottle prim
        bottle_mat.func("/World/Looks/BottleMaterial", bottle_mat)
        sim_utils.bind_physics_material("/World/Bottle", "/World/Looks/BottleMaterial")

    global camera
    if args.snapshot:
        camera = Camera(CameraCfg(
            prim_path="/World/Cam", update_period=0, height=CAM_H, width=CAM_W, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, focus_distance=1.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)),
        ))

    sim.set_camera_view((0.55, -0.55, 1.25), (0.05, 0.0, 1.0))
    sim.reset()

    if not hand.is_fixed_base:
        raise RuntimeError("Expected a fixed-base hand articulation")
    jn = hand.joint_names
    print(f"REVO2_JOINTS ({len(jn)}): {jn}", flush=True)
    print(f"REVO2_BODIES ({len(hand.body_names)}): {hand.body_names}", flush=True)
    print("REVO2_MASSES " + ", ".join(f"{n}={float(m):.4f}" for n, m in zip(hand.body_names, hand.data.default_mass[0])), flush=True)
    lim = hand.data.joint_pos_limits[0].cpu().numpy()
    for i, name in enumerate(jn):
        print(f"  {name:38s} limits [{lim[i,0]:+.3f}, {lim[i,1]:+.3f}] rad", flush=True)
    idx = {name: i for i, name in enumerate(jn)}
    n_j = len(jn)
    lim_t = hand.data.joint_pos_limits[0]

    def targets(fingers, thumb, aux):
        q = torch.zeros((1, n_j), device=sim.device)
        for f in ("index", "middle", "ring", "pinky"):
            q[0, idx[J[f]]] = fingers * joints[J[f]]["upper"]
        q[0, idx[J["thumb"]]] = thumb * joints[J["thumb"]]["upper"]
        q[0, idx[J["thumb_aux"]]] = aux * joints[J["thumb_aux"]]["upper"]
        for dname, (pname, mult) in DISTAL.items():
            if dname in idx:
                q[0, idx[dname]] = min(mult * float(q[0, idx[pname]]), joints[dname]["upper"])
        return torch.minimum(torch.maximum(q, lim_t[:, 0]), lim_t[:, 1])

    q_open = targets(0.0, 0.0, 0.0)
    q_preshape = targets(0.0, 0.0, args.thumb_aux)  # thumb rotated into opposition, fingers still open
    q_close = targets(args.close_fingers, args.close_thumb, args.thumb_aux)

    ts = time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "hand": side, "model": args.model, "usd": str(usd_path), "tag": args.tag,
        "bottle": {"name": B["name"], "kind": B["kind"], "radius_at_grasp": r, "height": h, "mass": B["mass"],
                   "grasp_height": B["grasp_height"], "friction": args.bottle_friction, "mesh": B["obj"]},
        "targets": {"close_fingers": args.close_fingers, "close_thumb": args.close_thumb, "thumb_aux": args.thumb_aux},
        "drive": {"stiffness": args.stiffness, "damping": args.damping, "close_seconds": args.close_seconds,
                  "finger_effort": args.finger_effort, "thumb_lead": args.thumb_lead},
        "joint_names": jn, "q_close_cmd": q_close[0].cpu().tolist(), "palm_center_world": palm_world.tolist(),
        "frame_note": "side: palm normal -> +X, index->pinky -> -Z, fingers -> -Y | top: palm normal -> -Z, index->pinky -> +X", "grasp_mode": args.grasp_mode,
    }
    snaps = []

    def snapshot(label):
        if camera is None:
            return
        for _ in range(6):
            sim.render()
        camera.update(dt)
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
        from PIL import Image
        path = log_dir / f"revo2-{side}-{ts}-{label}.png"
        Image.fromarray(rgb[..., :3].astype(np.uint8)).save(path)
        snaps.append(str(path))
        print(f"REVO2_SNAPSHOT {label}: {path}", flush=True)

    class Recorder:
        """Streams rendered frames to ffmpeg (imageio-ffmpeg static binary) and writes MP4 + a small GIF."""

        def __init__(self, label):
            import subprocess
            import imageio_ffmpeg
            from PIL import ImageFont
            self.mp4 = log_dir / f"revo2-{side}-{ts}-{label}.mp4"
            self.gif = log_dir / f"revo2-{side}-{ts}-{label}.gif"
            self.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            self.proc = subprocess.Popen(
                [self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CAM_W}x{CAM_H}",
                 "-r", str(args.video_fps), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                 str(self.mp4)], stdin=subprocess.PIPE,
            )
            self.font = ImageFont.load_default()
            self.n = 0

        def frame(self, text):
            from PIL import Image, ImageDraw
            for _ in range(2):
                sim.render()
            camera.update(dt)
            rgb = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
            img = Image.fromarray(np.ascontiguousarray(rgb))
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, CAM_W, 22], fill=(0, 0, 0))
            d.text((6, 5), text, fill=(255, 255, 255), font=self.font)
            self.proc.stdin.write(img.tobytes())
            self.n += 1

        def close(self):
            import subprocess
            self.proc.stdin.close()
            self.proc.wait()
            subprocess.run(
                [self.ffmpeg, "-y", "-loglevel", "error", "-i", str(self.mp4), "-vf",
                 "fps=12,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse", str(self.gif)],
                check=False,
            )
            print(f"REVO2_VIDEO: {self.mp4} ({self.n} frames) gif: {self.gif}", flush=True)
            return str(self.mp4), str(self.gif)

    def set_pose(obj, pos):
        pose = torch.zeros((1, 7), device=sim.device)
        pose[0, :3] = torch.tensor(pos, device=sim.device, dtype=torch.float32)
        pose[0, 3] = 1.0
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))

    def run_trial(gap, distal, vertical, label, want_snaps, want_video=False):
        # palm centre sits at grasp_height above the bottle base (plus vertical offset); the bottle axis is gap+radius
        # in front of the knuckle line and `distal` toward the fingertips
        if args.grasp_mode == "side":
            axis_xy = palm_world[:2] + n_t[:2] * (r + gap) + f_t[:2] * distal
            table_top_z = palm_world[2] - B["grasp_height"] + vertical
        else:
            # top grasp: bottle axis `distal` toward the fingertips (in the finger plane), cap top `gap` below the knuckle line
            axis_xy = palm_world[:2] + f_t[:2] * distal + w_t[:2] * vertical
            table_top_z = palm_world[2] - gap - h
        bottle_center = np.array([axis_xy[0], axis_xy[1], table_top_z + B["anchor"]])  # object origin pose
        table_pos = np.array([bottle_center[0] + 0.1, bottle_center[1], table_top_z - table_thickness / 2])
        # reset scene state: hand pre-shaped (thumb rotated into opposition, fingers open) with the bottle parked
        # 0.3 m away, then the bottle+table are brought into the grasp pocket. This mimics the arm approaching a
        # standing bottle with an already pre-shaped hand, which a fixed-base hand cannot do by moving itself.
        far = np.array([0.0, 0.0, -0.5])
        hand.write_joint_state_to_sim(q_preshape, torch.zeros_like(q_preshape))
        hand.set_joint_position_target(q_preshape)
        hand.write_data_to_sim()
        set_pose(table, table_pos + far if not args.static_table else table_pos + np.array([0.0, 0.0, -1.5]))
        set_pose(bottle, bottle_center + far)
        hand.reset(); bottle.reset(); table.reset()
        if camera is not None:
            look = [float(bottle_center[0]), float(bottle_center[1]), float(table_top_z + h / 2.0)]
            # over-the-shoulder view: behind the hand (-X), toward the fingertips (-Y), elevated; the bottle is glass
            look[2] = float(table_top_z + B["grasp_height"])
            if args.grasp_mode == "top":
                look[2] = float(table_top_z + h - 0.03)
                eye = torch.tensor([[look[0] + 0.32, look[1] - 0.30, look[2] + 0.12]], device=sim.device, dtype=torch.float32)
            else:
                eye = torch.tensor([[look[0] - 0.30, look[1] - 0.24, look[2] + 0.24]], device=sim.device, dtype=torch.float32)
            target = torch.tensor([look], device=sim.device, dtype=torch.float32)
            camera.set_world_poses_from_view(eye, target)

        # timeline (s)
        t_settle = 0.3                      # hand settles in the pre-shape with the bottle parked away
        t_pre_end = t_settle + 0.3          # bottle+table teleported into the pocket at t_settle, settle
        t_close_end = t_pre_end + args.thumb_lead + args.close_seconds  # thumb (lead) then fingers close
        t_hold_end = t_close_end + 0.5
        t_lower_end = t_hold_end + 0.6      # table drops away
        t_test_end = t_lower_end + args.hold_seconds
        table_pose0 = table.data.root_pose_w.clone()
        rec = Recorder(label) if (want_video and camera is not None) else None
        capture_every = max(1, int(round(1.0 / (args.video_fps * dt))))
        csv_rows = []
        z_ref = None
        min_z = None
        max_tilt = 0.0
        closed_info = None
        step = 0
        t = 0.0
        done_snap = set()
        while t < t_test_end + dt:
            if t < t_pre_end:
                q_cmd = q_preshape
                if abs(t - t_settle) < dt / 2:  # bring the bottle and its table into the grasp pocket
                    if not args.static_table:
                        set_pose(table, table_pos)
                    set_pose(bottle, bottle_center)
                    table_pose0[0, :3] = torch.tensor(table_pos, device=sim.device, dtype=torch.float32)
            elif t < t_close_end:
                # thumb flexion ramps first (lead), fingers follow; both cosine ramps
                a_th = min(1.0, (t - t_pre_end) / max(args.close_seconds, 1e-3))
                a_fi = min(1.0, max(0.0, (t - t_pre_end - args.thumb_lead) / max(args.close_seconds, 1e-3)))
                a_th = 0.5 - 0.5 * math.cos(math.pi * a_th)
                a_fi = 0.5 - 0.5 * math.cos(math.pi * a_fi)
                q_cmd = q_preshape + a_fi * (q_close - q_preshape)
                th = idx[J["thumb"]]
                q_cmd[0, th] = q_preshape[0, th] + a_th * (q_close[0, th] - q_preshape[0, th])
            else:
                q_cmd = q_close
            if args.coupling == "linkage":
                # rigid 4-bar coupling: the distal joint can only be where the proximal joint puts it
                q_cmd = q_cmd.clone()
                qa = hand.data.joint_pos[0]
                for dname, (pname, mult) in DISTAL.items():
                    if dname in idx:
                        q_cmd[0, idx[dname]] = min(mult * float(qa[idx[pname]]), joints[dname]["upper"])
            hand.set_joint_position_target(q_cmd)
            hand.write_data_to_sim()
            if t_hold_end <= t < t_lower_end and not args.static_table:
                a = 0.5 - 0.5 * math.cos(math.pi * (t - t_hold_end) / (t_lower_end - t_hold_end))
                pose = table_pose0.clone()
                pose[0, 2] -= a * args.lower_by
                table.write_root_pose_to_sim(pose)
            sim.step(render=False)  # physics only; rendering happens in snapshot()
            hand.update(dt); bottle.update(dt); table.update(dt)
            step += 1
            t = step * dt
            bp = bottle.data.root_pos_w[0]
            w_, x_, y_, z_ = (float(v) for v in bottle.data.root_quat_w[0])
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x_ * x_ + y_ * y_)))))
            if not torch.isfinite(hand.data.joint_pos).all() or not torch.isfinite(bp).all():
                raise RuntimeError("Non-finite physics state")
            phase = ("pre-shape" if t < t_settle else "bottle in pocket" if t < t_pre_end else "closing" if t < t_close_end
                     else "holding" if t < t_hold_end else "table dropping" if t < t_lower_end else "hold test")
            if rec is not None and step % capture_every == 0:
                rec.frame(f"Revo 2 {side} | {B['name']} | gap {gap*1000:.0f} mm fwd {distal*1000:.0f} mm | t={t:4.2f}s {phase}")
            if args.log_joints:
                csv_rows.append([f"{t:.4f}", phase] + [f"{v:.4f}" for v in hand.data.joint_pos[0].cpu().tolist()]
                                + [f"{float(v):.4f}" for v in bp] + [f"{tilt:.2f}"])
            if want_snaps and "open" not in done_snap and t >= t_settle - 2 * dt:
                done_snap.add("open"); snapshot(f"{label}-open")
            if closed_info is None and t >= t_hold_end - 2 * dt:
                z_ref = float(bp[2])
                qp = hand.data.joint_pos[0]
                closed_info = {
                    "target_minus_actual": {name: round(float(q_close[0, idx[name]] - qp[idx[name]]), 3) for name in J.values()},
                    "bottle_pos": bp.cpu().tolist(), "bottle_tilt_deg": tilt,
                }
                if want_snaps:
                    snapshot(f"{label}-closed")
            if t_lower_end <= t <= t_test_end:
                z = float(bp[2])
                min_z = z if min_z is None else min(min_z, z)
                max_tilt = max(max_tilt, tilt)
        drop = z_ref - min_z
        errs = closed_info["target_minus_actual"]
        n_blocked = sum(1 for k, v in errs.items() if "thumb_metacarpal" not in k and v > 0.12)
        stayed = bool(drop < 0.03 and max_tilt < 25.0)
        if stayed and n_blocked >= 2:
            status = "held"
        elif stayed:
            status = "stuck-no-finger-contact"  # bottle did not fall but no finger was stopped by it: interpenetration
        else:
            status = "dropped"
        held = status == "held"
        media = {}
        if rec is not None:
            for _ in range(int(0.5 * args.video_fps)):  # linger half a second on the final state
                rec.frame(f"Revo 2 {side} | {B['name']} | result: {status} (sag {drop*100:.1f} cm, tilt {max_tilt:.0f} deg)")
            media["mp4"], media["gif"] = rec.close()
        if args.log_joints:
            csv_path = log_dir / f"revo2-{side}-{ts}-{label}-joints.csv"
            with open(csv_path, "w") as fh:
                fh.write(",".join(["t", "phase"] + jn + ["bottle_x", "bottle_y", "bottle_z", "tilt_deg"]) + "\n")
                fh.write("\n".join(",".join(row) for row in csv_rows) + "\n")
            media["joints_csv"] = str(csv_path)
            print(f"REVO2_JOINTS_CSV: {csv_path}", flush=True)
        if want_snaps:
            snapshot(f"{label}-after-lower")
        blocked = {k.replace(f"{side}_", "").replace("_joint", ""): v for k, v in closed_info["target_minus_actual"].items()}
        print(
            f"REVO2_TRIAL {label}: {status} drop={drop*100:.1f}cm tilt_max={max_tilt:.0f}deg blocked_fingers={n_blocked} "
            f"gap={gap*1000:.0f}mm distal={distal*1000:.0f}mm vert={vertical*1000:+.0f}mm bottle={B['name']} blocked(rad)={blocked}",
            flush=True,
        )
        return {
            "label": label, "gap": gap, "distal_offset": distal, "vertical_offset": vertical,
            "bottle_center_init": bottle_center.tolist(), "held": held, "status": status, "n_blocked": n_blocked,
            "drop_m": drop, "max_tilt_deg": max_tilt,
            "closed_state": closed_info, "final_joint_pos": hand.data.joint_pos[0].cpu().tolist(), "media": media,
            "coupling": args.coupling,
        }

    gaps = parse_sweep(args.sweep_gap, args.gap)
    distals = parse_sweep(args.sweep_distal, args.distal_offset)
    verticals = parse_sweep(args.sweep_vertical, args.vertical_offset)
    trials = []
    combos = [(g, d, v) for g in gaps for d in distals for v in verticals]
    for i, (g, d, v) in enumerate(combos):
        label = f"t{i:02d}"
        want = args.snapshot and (len(combos) == 1 or args.snapshot_all)
        want_video = args.video and (len(combos) == 1 or args.video_all)
        trials.append(run_trial(g, d, v, label, want, want_video))
    held = [tr for tr in trials if tr["held"]]
    # snapshot/record the best held trial (least sag and tilt) if we did not already
    if (args.snapshot or args.video) and not (args.snapshot_all or args.video_all) and len(combos) > 1 and held:
        best = min(held, key=lambda tr: 10.0 * tr["drop_m"] + tr["max_tilt_deg"] / 100.0)  # sag matters more than tilt
        trials.append(run_trial(best["gap"], best["distal_offset"], best["vertical_offset"], "best", True, args.video))
    common["coupling"] = args.coupling
    summary = {**common, "trials": trials, "snapshots": snaps,
               "n_held": len(held), "n_trials": len(trials)}
    out = log_dir / f"revo2-{side}-{ts}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"REVO2_GRASP_RESULT held {len(held)}/{len(trials)} trials; report: {out}", flush=True)
    for tr in held:
        print(f"  HELD gap={tr['gap']*1000:.0f}mm distal={tr['distal_offset']*1000:.0f}mm vert={tr['vertical_offset']*1000:+.0f}mm drop={tr['drop_m']*100:.1f}cm", flush=True)

    if args.max_steps == 0 and not args.headless:
        print("Test finished; window stays open (close it to exit).", flush=True)
        while app.is_running():
            hand.set_joint_position_target(q_close)
            hand.write_data_to_sim()
            sim.step()
            hand.update(dt); bottle.update(dt); table.update(dt)


if __name__ == "__main__":
    import threading

    def _watchdog():
        print("REVO2_WATCHDOG: shutdown took too long, forcing exit", flush=True)
        os._exit(3)

    rc = 1
    try:
        main()
        rc = 0
    except BaseException as exc:  # make SystemExit/KeyboardInterrupt visible in logs too
        import traceback

        print(f"REVO2_ABORT: {exc!r}", flush=True)
        traceback.print_exc()
    finally:
        timer = threading.Timer(20.0, _watchdog)
        timer.daemon = True
        timer.start()
        try:
            app.close()
        finally:
            # Kit occasionally hangs on shutdown after GPU errors and keeps VRAM allocated; never linger.
            sys.stdout.flush()
            os._exit(rc)
