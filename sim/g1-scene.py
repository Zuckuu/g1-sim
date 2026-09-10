"""Reduced fixed-base G1 + Dex3 scene, using Unitree's official articulation config.

This checks asset loading, joint state, and physics. No DDS, headset, or grasp policy.
"""
import argparse
import builtins
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pepsidemo_sim as pd  # noqa: E402

task_root = pd.PROJECT_ROOT
unitree_root = task_root / "work/unitree_sim_isaaclab"
os.environ["PROJECT_ROOT"] = str(unitree_root)
sys.path.insert(0, str(unitree_root))
asset_path = unitree_root / "assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
if not asset_path.is_file():
    raise SystemExit(f"Required Unitree asset is not installed: {asset_path}")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--max-steps", type=int, default=0, help="0 keeps the scene open until closed")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
pd.ensure_dirs()
pd.select_experience(args)  # trimmed Isaac Lab experience (see pepsidemo_sim.py)
launcher = AppLauncher(
    args, width=640, height=480, window_width=960, window_height=640,
    renderer="RayTracedLighting", anti_aliasing=0, multi_gpu=False,
)
app = launcher.app
# note: do NOT set builtins.ISAAC_LAUNCHED_FROM_TERMINAL = True here; it makes Isaac Sim skip
# creating the PhysicsContext and sensors/physics then fail during sim.reset().

# Simulator-dependent imports must occur after application startup.
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from robots.unitree import G129_CFG_WITH_DEX3_BASE_FIX


def static_box(path, size, position, color):
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    cfg.func(path, cfg, translation=position)


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(device=args.device, dt=0.005, render_interval=4)
    )
    sim.set_camera_view((2.0, -2.0, 1.65), (0.25, 0.0, 0.85))
    static_box("/World/Floor", (5.0, 5.0, 0.1), (0.0, 0.0, -0.05), (0.12, 0.14, 0.17))
    static_box("/World/TableTop", (0.8, 0.8, 0.06), (0.6, 0.0, 0.67), (0.45, 0.38, 0.30))
    for index, (x, y) in enumerate(((0.25, -0.35), (0.25, 0.35), (0.95, -0.35), (0.95, 0.35))):
        static_box(f"/World/TableLeg{index}", (0.045, 0.045, 0.64), (x, y, 0.32), (0.3, 0.3, 0.3))
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
    light.func("/World/Light", light)

    robot_cfg = G129_CFG_WITH_DEX3_BASE_FIX.copy()
    robot_cfg.prim_path = "/World/G1"
    robot = Articulation(robot_cfg)
    # Same dimensions/mass/friction as Unitree's stock cylinder, not a Pepsi bottle.
    cylinder = RigidObject(RigidObjectCfg(
        prim_path="/World/Cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, -0.20, 0.88)),
        spawn=sim_utils.CylinderCfg(
            radius=0.018, height=0.35,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.8)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5, dynamic_friction=1.5,
                friction_combine_mode="max", restitution=0.0,
            ),
        ),
    ))
    sim.reset()
    if not robot.is_initialized or not cylinder.is_initialized:
        raise RuntimeError("Robot or cylinder physics failed to initialize")
    if not robot.is_fixed_base:
        raise RuntimeError("Expected the fixed-base Unitree articulation")
    hands = [name for name in robot.joint_names if "hand" in name]
    if len(hands) != 14:
        raise RuntimeError(f"Expected 14 Dex3 finger joints, found {hands}")
    targets = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(targets, robot.data.default_joint_vel.clone())
    robot.reset()
    cylinder.reset()
    print(f"G1_SCENE_READY: fixed base; {robot.num_joints} joints; 14 Dex3 finger joints.", flush=True)
    print("Holding the initial joint targets. This scene does not yet perform a grasp. Close the window to exit.", flush=True)
    report_dir = task_root / "work/g1-runtime/logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "g1-joints.json").write_text(json.dumps(robot.joint_names, indent=2) + "\n")
    step = 0
    while app.is_running():
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        cylinder.update(sim.get_physics_dt())
        step += 1
        if not torch.isfinite(robot.data.joint_pos).all() or not torch.isfinite(cylinder.data.root_pos_w).all():
            raise RuntimeError("Non-finite physics state detected")
        if step == 200:
            z = float(cylinder.data.root_pos_w[0, 2])
            if not 0.84 < z < 0.91:
                raise RuntimeError(f"Cylinder did not remain upright on the table: center height {z:.3f} m")
            print(f"G1_PHYSICS_OK: 200 steps; finite joint state; cylinder center height {z:.3f} m.", flush=True)
        if args.max_steps > 0 and step >= args.max_steps:
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
