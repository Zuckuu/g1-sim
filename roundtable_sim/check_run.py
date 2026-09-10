"""Sanity checks for a recorded run (trajectory.npz + meta.json).

    python check_run.py out/run

Fails (exit code 1) if the puppet ever "breaks": joint jumps between consecutive
30 Hz frames, arm moves whose IK target was not reached, a contorted waist or an
upper arm swung behind the back, knees bent the wrong way, over-extended legs,
feet under the floor, or cans that did not end up upright on their coaster.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import ScenarioConfig  # noqa: E402
from fetch_assets import G1_DIR  # noqa: E402
from g1_kinematics import ARM_JOINTS, LEG_JOINTS, NQ_ROBOT  # noqa: E402
from render import Run  # noqa: E402


def check(run_dir: Path, verbose: bool = True) -> bool:
    run = Run.load(run_dir)
    m = mujoco.MjModel.from_xml_path(str(G1_DIR / "roundtable_scene.xml"))
    d = mujoco.MjData(m)
    L = ScenarioConfig().layout
    ok = True

    def fail(msg: str) -> None:
        nonlocal ok
        ok = False
        print(f"[check] FAIL: {msg}")

    # 1. joint continuity (robot joints only, skip the free base)
    q = run.qpos[:, 7:NQ_ROBOT]
    dq = np.abs(np.diff(q, axis=0))
    jumps = np.argwhere(dq > 0.35)
    if len(jumps):
        frames = sorted(set(int(j[0]) for j in jumps))
        fail(f"{len(frames)} frames with joint jumps > 0.35 rad/frame, first at t={run.t[frames[0]]:.2f}s")

    # 1b. arm IK residuals recorded by the scenario (unreachable targets make the IK wander off)
    warns = [e for e in run.events if e["kind"] == "ik_warn"]
    if warns:
        fail(f"{len(warns)} arm moves ended with a large IK residual, first at t={warns[0]['t']:.2f}s "
             f"({warns[0]['phase']}, err {warns[0]['err']})")

    # 1c. upper body never contorted: waist stays near upright, upper arm never swings far back
    def qcol(name: str) -> np.ndarray:
        return run.qpos[:, m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]]
    wy, wp = qcol("waist_yaw_joint"), qcol("waist_pitch_joint")
    if np.abs(wy).max() > 0.6 or wp.min() < -0.3 or wp.max() > 0.5:
        i = int(np.argmax(np.abs(wy) + np.abs(wp)))
        fail(f"waist contorted (yaw {wy[i]:+.2f}, pitch {wp[i]:+.2f} rad at t={run.t[i]:.2f}s)")
    for s in ("left", "right"):
        sp = qcol(f"{s}_shoulder_pitch_joint")
        if sp.max() > 1.2:
            fail(f"{s} upper arm swung behind the back ({sp.max():.2f} rad at t={run.t[int(sp.argmax())]:.2f}s)")

    # 2. knees, leg reach, foot height
    knee = {s: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_knee_joint")] for s in ("left", "right")}
    for s, adr in knee.items():
        if run.qpos[:, adr].min() < -0.05:
            fail(f"{s} knee below -0.05 rad (min {run.qpos[:, adr].min():.3f})")
    feet = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")}
    hips = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hip_pitch_link") for s in ("left", "right")}
    min_z, max_reach = 1e9, 0.0
    for i in range(0, len(run.t), 2):
        d.qpos[:] = run.qpos[i]
        mujoco.mj_kinematics(m, d)
        for s in ("left", "right"):
            min_z = min(min_z, d.xpos[feet[s]][2])
            max_reach = max(max_reach, float(np.linalg.norm(d.xpos[feet[s]] - d.xpos[hips[s]])))
    if min_z < 0.02:
        fail(f"foot site below 2 cm (min {min_z:.3f} m; sole is 3.3 cm below the site)")
    if max_reach > 0.64:
        fail(f"leg over-extended: hip->ankle {max_reach:.3f} m")

    # 3. every served guest has an upright can on their coaster at the end
    d.qpos[:] = run.qpos[-1]
    mujoco.mj_kinematics(m, d)
    served = [e for e in run.events if e["kind"] == "served"]
    can_adrs = []
    for k in range(12):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"can_{k}_free")
        can_adrs.append(int(m.jnt_qposadr[j]))
    for e in served:
        cx, cy, cz = L.coaster_pos(e["guest"])
        best = min(can_adrs, key=lambda a: math.hypot(run.qpos[-1][a] - cx, run.qpos[-1][a + 1] - cy))
        p = run.qpos[-1][best:best + 3]
        w, x, y, z = run.qpos[-1][best + 3:best + 7]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
        dist = math.hypot(p[0] - cx, p[1] - cy)
        if dist > 0.03 or tilt > 3.0 or abs(p[2] - (cz + 0.006 + L.can_half_height)) > 0.01:
            fail(f"guest {e['guest']} ({e['name']}): can off coaster (dist {dist:.3f} m, tilt {tilt:.1f} deg, z {p[2]:.3f})")
    if len(served) != len(run.guests):
        fail(f"served {len(served)} of {len(run.guests)} guests")

    if verbose:
        arm = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS["right"]]
        leg = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in LEG_JOINTS["right"]]
        print(f"[check] {len(run.t)} frames, {run.t[-1]:.1f} s; served {len(served)}/{len(run.guests)}; "
              f"max joint step {dq.max():.3f} rad/frame (right arm {np.abs(np.diff(run.qpos[:, arm], axis=0)).max():.3f}, "
              f"right leg {np.abs(np.diff(run.qpos[:, leg], axis=0)).max():.3f}); min foot z {min_z:.3f}; "
              f"max hip->ankle {max_reach:.3f} m")
        print("[check] OK" if ok else "[check] problems found")
    return ok


if __name__ == "__main__":
    sys.exit(0 if check(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "out" / "run")) else 1)
