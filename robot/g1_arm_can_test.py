#!/usr/bin/env python3
"""Staged, recorded arm + hand test on the real G1: raise the arm, put the palm on a 12 oz can, grasp, lift.

The arm is driven over Unitree's `rt/arm_sdk` (the motion controller keeps the legs; the blend weight in
motor_cmd[29].q is ramped 0 -> 1 while we command the measured pose, so nothing jumps). The palm target geometry is
the one that held the can in the simulator (docs/robot/README.md, "Placement rule"): palm face 3 mm off the can,
palm centre 45 mm above the can base (live wrap; fingers on the body, not the rim), can axis 15 mm toward the fingertips from the knuckle line, arrive 3 cm high
and descend, thumb up until the palm is at the can, then oppose + ramp-close with contact freeze (the bench recipe,
run through robot/revo2_hand_test.py), then lift.

Kinematics: pinocchio on the robot's own URDF (~/unitree/g1_description/g1_29dof_rev_1_0.urdf) plus the palm frame
of the Revo 2 on the wrist yaw link, derived from sim/build_g1_revo2_urdf.py with the (still unmeasured) 12 mm
adapter. Targets are given in the pelvis "level" frame (pelvis origin, roll/pitch removed with the pelvis IMU).
Free-space palm moves match the sim: IK is solved once, then the seven arm joints cosine-interpolate (the live DLS
servo from the measured pose is what made the raise shimmy). The dryrun is the plan: it solves every waypoint
(raise = one curl from the hang pose into a hand-at-chest ready pose in front of the table edge, then short vias at
--clearance above the LOOK table, then pregrasp) and checks every joint interpolant against the table slab and a box
model of the torso/head/hips; the live stages replay exactly those joints (a live re-solve once picked the IK basin
that folds the arm into the chest). park() retraces the executed waypoints backwards. Contact moves use the same
spline and only differ at the end (press-lead clamp + settle). `--offline --stage dryrun --can-x .. --table-x ..`
runs the planner on a laptop from a recorded arm pose (no robot, no DDS).

Stages. Start at the top of this list and only move down after the previous one matched:
  check      read-only snapshot: DDS rates, controller kp, loco FSM, palm pose, hands, foreign publishers
  fsm        read-only watch: print loco FSM + controller gains every time they change (work the remote; Ctrl-C)
  look       read-only: D435i depth + RGB -> table plane + 12 oz can in the pelvis level frame (no motion)
  step       take over, move ONE joint --delta rad and back (--joint wrist_yaw, --delta 0.05). Proves arm_sdk
             is live in the current FSM without any other motion. Repeat with --reps. Then try the next joint.
  handshake  step of wrist_yaw +0.06 rad (the first live arm_sdk proof)
  dryrun     plan the whole can sequence kinematically from the measured pose (no motion)
  raise      take over, move to the planned ready pose (hand at the chest, before the table edge), hold, park
  all        take over -> raise -> pregrasp -> approach -> descend -> grasp -> lift -> lower -> release -> retreat
             -> park; `--until <stage>` stops after that stage (holding, with the prompt) and reverses safely

Prompt between stages: Enter continues; `n dx dy dz` moves the can estimate by cm (re-plans the following stages);
`j dx dy dz` jogs the palm by cm right now; `s` grabs a camera still; `p` parks and exits; `x` freezes the arm.

Safety: refuses to publish while the controller is in zero torque / damping (arm_sdk has no effect there), refuses
if anyone else publishes on rt/arm_sdk, hard-locked to one arm (--arm; right needs --allow-right). Joint speed is
capped, the command never leads the measured joint by more than --windup rad (a blocked joint cannot wind up
torque), joints stay inside URDF limits, and sustained over-torque, a stalled approach, a stale state stream, a
foreign publisher, a palm below the table, or a free-space move that ends >40 mm off FREEZE the arm at its measured
pose (weight stays 1, PD holds). From a freeze the operator parks
(slow return to the start pose, weight -> 0) or releases (weight -> 0 where it is). The remote's damping button
(L2+B) overrides everything at any time - keep it in hand.

Run on the Jetson (python 3.8 env with unitree_sdk2py + pinocchio 3.2):
  PY=~/miniforge3/envs/g1brainco/bin/python
  $PY g1_arm_can_test.py --stage check --rpc
  $PY g1_arm_can_test.py --stage fsm                     # work the remote: L2+B, L2+UP, R1+X; confirm each line
  $PY g1_arm_can_test.py --stage step                     # wrist_yaw +0.05 rad and back (after standing)
  $PY g1_arm_can_test.py --stage step --joint wrist_pitch --delta 0.05
  $PY g1_arm_can_test.py --stage handshake                # wrist_yaw +0.06 (same as the first step, named)
  $PY g1_arm_can_test.py --stage look                     # table + can from the D435i; prints --can-x/y/z
  $PY g1_arm_can_test.py --stage dryrun --look            # look, then kinematic plan with the discovered can
  $PY g1_arm_can_test.py --stage dryrun --table-height 0.74
  $PY g1_arm_can_test.py --stage all --table-height 0.74 --until pregrasp
  $PY g1_arm_can_test.py --stage all --table-height 0.74
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------------------------------------------------
G1_URDF = os.path.expanduser("~/unitree/g1_description/g1_29dof_rev_1_0.urdf")
JOINT_NAMES = [  # motor index order == pinocchio q order for this URDF (verified on the robot)
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
ARM_IDX = {"left": list(range(15, 22)), "right": list(range(22, 29))}
WAIST_IDX = [12, 13, 14]
UPPER_IDX = ARM_IDX["left"] + ARM_IDX["right"] + WAIST_IDX  # the 17 joints rt/arm_sdk blends
WEIGHT_SLOT = 29           # motor_cmd[29].q = arm_sdk blend weight
SIGNATURE_SLOT = 34        # motor_cmd[34].reserve stamps our messages (unused slot)
SIGNATURE = 0x5150
EFFORT = {i: 25.0 for i in range(15, 29)}
for _i in (20, 21, 27, 28):
    EFFORT[_i] = 5.0       # wrist pitch / yaw
# Palm frame in {side}_wrist_yaw_link: centre = mean knuckle origin, axes f (fingers) / w / n (palm normal, out of
# the palm face) as a right-handed triad. From sim/build_g1_revo2_urdf.py mount (rubber-hand point 0.0415, +/-0.003
# plus 12 mm adapter) and g1_walk_grasp.palm_frame_in_hand_base(). Palm face is 0.023 m out along n.
PALM = {
    "left": dict(p=[0.13007, 0.03118, -0.01005], f=[1.0, 0.0, 0.0], w=[0.0, 0.19215, 0.98137], n=[0.0, -0.98137, 0.19215]),
    "right": dict(p=[0.13007, -0.03117, -0.01006], f=[1.0, 0.0, 0.0], w=[0.0, 0.19215, -0.98137], n=[0.0, 0.98137, 0.19215]),
}
# d435_link is ROS camera_link (x viewing, y left, z up). RealSense deprojects in optical (x right, y down, z fwd).
R_D435_LINK_FROM_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
SOLE_PTS = [[-0.05, 0.025, -0.03], [-0.05, -0.025, -0.03], [0.12, 0.03, -0.03], [0.12, -0.03, -0.03]]  # ankle_roll frame
SOLE_R = 0.005
CAN_R, CAN_H = 0.0331, 0.1224   # 12 oz can (assets/bottles/pepsi-12oz-can.json)
FINGERS = ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]
FINGERPRINT = {"thumb": 0.21, "index": 0.20, "middle": 0.28, "ring": 0.25, "pinky": 0.22}  # live wrap on the can body
STAGES_ALL = ["raise", "pregrasp", "approach", "descend", "grasp", "lift", "lower", "release", "retreat", "park"]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--stage", choices=["check", "fsm", "look", "step", "handshake", "dryrun", "raise", "all", "recover"], default="check")
parser.add_argument("--resume-plan", default=None, help="recover: JSON of the run that left the arm out (its dryrun plan + takeover pose are retraced)")
parser.add_argument("--until", choices=STAGES_ALL, default=None, help="all: stop after this stage (hold + prompt), then reverse out")
parser.add_argument("--joint", choices=["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"],
                    default="wrist_yaw", help="step: which arm joint to move (everything else is held)")
parser.add_argument("--delta", type=float, default=0.05, help="step: radians to add, then reverse (0.05 rad = 2.9 deg)")
parser.add_argument("--reps", type=int, default=1, help="step: how many times to go out and back")
parser.add_argument("--watch-seconds", type=float, default=0.0, help="fsm: stop after this many seconds (0 = until Ctrl-C)")
parser.add_argument("--arm", choices=["left", "right"], default="left")
parser.add_argument("--allow-right", action="store_true", help="required to drive the right arm/hand (other tests run there)")
parser.add_argument("--iface", default="eth0")
parser.add_argument("--domain", type=int, default=0)
parser.add_argument("--urdf", default=G1_URDF)
# can placement, pelvis level frame (x ahead, y left, z up from the pelvis origin)
parser.add_argument("--can-x", type=float, default=None, help="can axis ahead of the pelvis; default: D435i --look")
parser.add_argument("--can-y", type=float, default=None, help="can axis lateral; default: D435i --look (else +/-0.12 toward the working side)")
parser.add_argument("--can-z", type=float, default=None, help="can BASE height relative to the pelvis origin; default from --table-height")
parser.add_argument("--table-height", type=float, default=None, help="floor -> table top (m); combined with the pelvis height estimate (or --pelvis-height)")
parser.add_argument("--pelvis-height", type=float, default=None, help="floor -> pelvis origin (m); default: estimated from the leg kinematics (feet on the floor)")
# grasp geometry (sim defaults that held)
parser.add_argument("--gap", type=float, default=0.026, help="knuckle line to can surface along the palm normal (m); palm face is 0.023 out -> 3 mm clearance")
parser.add_argument("--press", type=float, default=0.0, help="approach this far past the surface (compliant press through the arm PD). 0.005 + press-lead 0.03 pushed the can over")
parser.add_argument("--distal-offset", type=float, default=0.015, help="can axis this far toward the fingertips from the knuckle line (m)")
parser.add_argument("--palm-x-trim", type=float, default=-0.025, help="calibration: shift every palm target along level x (m); -0.025 = the hand was an inch too far forward at the can")
parser.add_argument("--palm-y-trim", type=float, default=0.0, help="calibration: shift every palm target along level y (m); for the left hand +0.01 stops 1 cm further from the can")
parser.add_argument("--grasp-height", type=float, default=0.045, help="palm centre above the can base (m); 0.045 wraps the body, 0.055 caught the rim")
parser.add_argument("--approach-rise", type=float, default=0.03)
parser.add_argument("--lift", type=float, default=0.14)
parser.add_argument("--pregrasp-gap", type=float, default=0.08, help="pre-grasp standoff added to --gap (m)")
# arm controller
parser.add_argument("--kp", type=float, default=120.0, help="arm PD stiffness sent in rt/arm_sdk (sim: 60 collapsed under the press, 120 held; Unitree teleop runs 300)")
parser.add_argument("--kd", type=float, default=3.0)
parser.add_argument("--other-kp", type=float, default=60.0, help="PD on the other arm (held at its takeover pose; Unitree example gains)")
parser.add_argument("--other-kd", type=float, default=1.5)
parser.add_argument("--waist-kp", type=float, default=120.0, help="PD on the 3 waist joints. At 60 the extended arm's ~11 Nm folded the waist 10 deg forward")
parser.add_argument("--waist-kd", type=float, default=3.0)
parser.add_argument("--waist-upright", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="waist yaw/roll/pitch target (rad); a slow bounded integral holds the torso there")
parser.add_argument("--vmax", type=float, default=0.35, help="rad/s cap on every arm joint command (Unitree example 0.5, August program 0.75). Overridden by --speed-rung unless --vmax is also passed.")
parser.add_argument("--windup", type=float, default=0.12, help="rad: the command may lead the measured joint by at most this (kp*windup = max static torque)")
parser.add_argument("--press-lead", type=float, default=0.012, help="rad: extra command lead allowed once a contact move (approach/descend/lower/jog) has "
                    "finished interpolating; kp*press-lead = the palm press torque (120*0.03 = 3.6 Nm, ~7 N at the palm)")
parser.add_argument("--ik-gain", type=float, default=0.5, help="unused for palm moves (joint spline); kept for a Cartesian servo if we re-enable one")
parser.add_argument("--ik-lambda", type=float, default=0.05)
parser.add_argument("--time-scale", type=float, default=1.0, help="multiply every motion duration. Overridden by --speed-rung unless --time-scale is also passed.")
parser.add_argument("--speed-rung", type=int, default=None, choices=[1, 2, 3, 4, 5, 6, 7],
                    help="arm-motion increment (same planned joints): "
                         "1=1.8x/0.25, 2=1.5x/0.35, 3=1.2x/0.45, 4=1.0x/0.50 Unitree example, "
                         "5=0.85x/0.60, 6=0.70x/0.70, 7=0.55x/0.75 August vmax. "
                         "Pregrasp/approach/descend/lower stay on --fine-time-scale/--fine-vmax.")
parser.add_argument("--fine-time-scale", type=float, default=1.5,
                    help="floor on time-scale for pregrasp/approach/descend/lower (rung 2; never faster than this even at higher rungs)")
parser.add_argument("--fine-vmax", type=float, default=0.35,
                    help="cap on vmax for those near-can moves (rad/s)")
parser.add_argument("--no-early-arrive", action="store_true",
                    help="free-space waypoints always dwell the full 1.5 s after the spline (default: done once the joints are within "
                         "0.02 rad and the palm within 8 mm, >= 0.2 s after the spline; pregrasp always dwells)")
parser.add_argument("--weight-seconds", type=float, default=3.0, help="arm_sdk weight ramp 0->1 (and back)")
parser.add_argument("--pos-tol", type=float, default=0.008, help="m: motion counts as arrived below this palm error")
parser.add_argument("--rot-tol", type=float, default=0.10, help="rad")
parser.add_argument("--hold", type=float, default=5.0, help="s to hold the can up before lowering")
# guards
parser.add_argument("--torque-frac", type=float, default=0.85, help="freeze if |tau_est| exceeds this fraction of the URDF effort limit for --torque-seconds")
parser.add_argument("--torque-seconds", type=float, default=0.3)
parser.add_argument("--stall-seconds", type=float, default=3.0, help="freeze if a motion makes no progress toward its goal for this long")
parser.add_argument("--max-seconds", type=float, default=900.0, help="session watchdog (freeze, then park)")
parser.add_argument("--max-tilt", type=float, default=0.30, help="rad: freeze if the pelvis rolls/pitches beyond this")
# hand
parser.add_argument("--hand-tool", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "revo2_hand_test.py"))
parser.add_argument("--no-hand", action="store_true", help="skip the hand stages (arm placement rehearsal)")
parser.add_argument("--ramp-rate", type=float, default=0.8)
parser.add_argument("--squeeze", type=float, default=0.10)
parser.add_argument("--stall-threshold", type=float, default=0.07)
parser.add_argument("--aux-target", type=float, default=1.0)
# misc
parser.add_argument("--rpc", action="store_true", help="check: also query motion_switcher.CheckMode and robot_state.ServiceList (read-only RPC)")
parser.add_argument("--stop-arm-example", action="store_true", help="switch Unitree's g1_arm_example service off first (it owns rt/arm_sdk when an action plays)")
parser.add_argument("--auto", action="store_true", help="no operator prompts (stages follow each other after --auto-pause s)")
parser.add_argument("--auto-pause", type=float, default=0.0, help="s to wait at each old operator prompt in --auto (0 = no pause; 1 was for a live abort window)")
parser.add_argument("--camera", default="http://127.0.0.1:8080/rgb.mjpg", help="MJPEG stream to grab a still from at each stage ('' = off)")
parser.add_argument("--clearance", type=float, default=0.08,
                    help="m above the TABLE (not the can top) for the horizontal via on a long reach; capped at 12 cm so the shoulder stays in the reaching pose")
parser.add_argument("--look", action="store_true", help="run D435i table/can discovery before dryrun/motion and fill --can-x/y/z")
parser.add_argument("--look-json", default="/tmp/g1-look-latest.json", help="where --stage look writes the estimate")
parser.add_argument("--look-xmax", type=float, default=0.95, help="m: LOOK depth ROI ahead (g1_fetch passes ~2.6 for the approach)")
parser.add_argument("--look-ymax", type=float, default=0.55, help="m: LOOK depth ROI to each side")
parser.add_argument("--vision", default="http://127.0.0.1:8080", help="g1_vision_stream.py base URL (look reads /color.jpg /depth.f32 /calib.json); '' = open the D435i directly")
parser.add_argument("--keep", action="store_true", help="handshake/raise: do not park at the end (leave the arm where it is, weight 1)")
parser.add_argument("--table-x", type=float, default=None, help="near edge of the table (level x, m); default from --look")
parser.add_argument("--offline", action="store_true", help="no robot: dryrun from --offline-q/--offline-rpy with the given can (kinematics only)")
parser.add_argument("--offline-q", default="0.191,0.148,0.023,1.203,-0.028,-0.014,0.124", help="offline: measured arm q (7, rad)")
parser.add_argument("--offline-rpy", default="0.2,1.6", help="offline: pelvis roll,pitch (deg)")
parser.add_argument("--label", default="")
parser.add_argument("--out", default="")
args = parser.parse_args()

# Same planned joints; only the spline clock and the per-tick joint cap change.
SPEED_RUNGS = {
    1: (1.8, 0.25), 2: (1.5, 0.35), 3: (1.2, 0.45), 4: (1.0, 0.50),
    5: (0.85, 0.60), 6: (0.70, 0.70), 7: (0.55, 0.75),
}
if args.speed_rung is not None:
    ts, vm = SPEED_RUNGS[args.speed_rung]
    if "--time-scale" not in sys.argv:
        args.time_scale = ts
    if "--vmax" not in sys.argv:
        args.vmax = vm

if args.arm == "right" and not args.allow_right:
    sys.exit("refusing to drive the right arm without --allow-right")
SIDE = args.arm
SGN = 1.0 if SIDE == "right" else -1.0        # palm normal points toward the midline: +y for the right hand
ARM = ARM_IDX[SIDE]
OTHER = ARM_IDX["left" if SIDE == "right" else "right"] + WAIST_IDX
if args.can_y is None and not args.look:
    args.can_y = -0.12 if SIDE == "right" else 0.12
DT = 0.02
T0 = time.monotonic()
FINE_LABELS = ("pregrasp", "approach", "descend", "lower", "jog")


def clock_for(label, contact=False):
    """Transit follows --speed-rung. Near-can palm moves never go faster than the last proven fine clock."""
    ts, vm = float(args.time_scale), float(args.vmax)
    lab = (label or "").split()[0].lower()
    fine = bool(contact) or lab in FINE_LABELS
    if fine:
        ts = max(ts, float(args.fine_time_scale))
        vm = min(vm, float(args.fine_vmax))
    return ts, vm, fine


def now():
    return time.monotonic() - T0


EVENTS = []


def log(msg, **kw):
    EVENTS.append(dict(t=round(now(), 3), msg=msg, **kw))
    print("[%7.2f] %s" % (now(), msg), flush=True)


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def smoothstep(a):
    a = min(1.0, max(0.0, a))
    return 0.5 - 0.5 * math.cos(math.pi * a)


# ----------------------------------------------------------------------------------------------------------------------
# kinematics (pinocchio)
# ----------------------------------------------------------------------------------------------------------------------
import pinocchio as pin  # noqa: E402


class Kin:
    """pinocchio model of the body + an operational frame for the Revo 2 palm. One shared `data`, so every call
    takes the lock (the 50 Hz thread and the operator thread both use it)."""

    def __init__(self, urdf, side):
        self.model = pin.buildModelFromUrdf(urdf)
        names = list(self.model.names)[1:]
        assert [n.replace("_joint", "") for n in names] == JOINT_NAMES, "URDF joint order differs from the motor index order"
        wrist = "%s_wrist_yaw_link" % side
        fid_w = self.model.getFrameId(wrist)
        fr = self.model.frames[fid_w]
        pf = PALM[side]
        R_pw = np.stack([np.array(pf["f"]), np.array(pf["w"]), np.array(pf["n"])], axis=1)  # palm axes in wrist coords
        u, _, vt = np.linalg.svd(R_pw)
        R_pw = u @ vt
        placement = fr.placement * pin.SE3(R_pw, np.array(pf["p"]))
        self.fid = self.model.addFrame(pin.Frame("%s_palm" % side, fr.parentJoint, fid_w, placement, pin.FrameType.OP_FRAME))
        self.fid_wrist = fid_w
        self.fid_elbow = self.model.getFrameId("%s_elbow_link" % side)
        self.fid_sroll = self.model.getFrameId("%s_shoulder_roll_link" % side)
        self.fid_sole = [self.model.getFrameId("left_ankle_roll_link"), self.model.getFrameId("right_ankle_roll_link")]
        self.data = self.model.createData()
        self.lo = np.array(self.model.lowerPositionLimit)
        self.hi = np.array(self.model.upperPositionLimit)
        self.lock = threading.Lock()

    def fk(self, q29):
        q = np.asarray(q29, dtype=float)
        with self.lock:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            M = self.data.oMf[self.fid]
            return M.translation.copy(), M.rotation.copy()

    def fk_frame(self, q29, name):
        q = np.asarray(q29, dtype=float)
        with self.lock:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            M = self.data.oMf[self.model.getFrameId(name)]
            return M.translation.copy(), M.rotation.copy()

    def fk_arm(self, q29):
        """one FK: shoulder-roll origin, elbow, wrist, palm p, palm R of the working arm (pelvis frame)."""
        q = np.asarray(q29, dtype=float)
        with self.lock:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            M = self.data.oMf[self.fid]
            return (self.data.oMf[self.fid_sroll].translation.copy(), self.data.oMf[self.fid_elbow].translation.copy(),
                    self.data.oMf[self.fid_wrist].translation.copy(), M.translation.copy(), M.rotation.copy())

    def jac(self, q29):
        q = np.asarray(q29, dtype=float)
        with self.lock:
            J = pin.computeFrameJacobian(self.model, self.data, q, self.fid, pin.LOCAL_WORLD_ALIGNED)
            return np.array(J[:, ARM])

    def fk_jac(self, q29):
        """palm p, R and the 6x7 arm Jacobian from ONE kinematics pass (computeFrameJacobian runs the forward
        kinematics; the palm placement is read from it). Same numbers as fk() + jac(), half the pinocchio work -
        the dryrun's IK sweep is ~40k of these on the Jetson."""
        q = np.asarray(q29, dtype=float)
        with self.lock:
            J = pin.computeFrameJacobian(self.model, self.data, q, self.fid, pin.LOCAL_WORLD_ALIGNED)
            M = pin.updateFramePlacement(self.model, self.data, self.fid)
            return M.translation.copy(), M.rotation.copy(), np.array(J[:, ARM])

    def sole_min_z(self, q29, R_level):
        """lowest foot sphere point, in the level frame (pelvis origin)."""
        q = np.asarray(q29, dtype=float)
        zs = []
        with self.lock:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            for fid in self.fid_sole:
                M = self.data.oMf[fid]
                for p in SOLE_PTS:
                    zs.append((R_level @ (M.rotation @ np.array(p) + M.translation))[2] - SOLE_R)
        return min(zs)

    def ik(self, q29, p_goal, R_goal, iters=400, q_bias=None, k_ns=0.4):
        """kinematic IK (no robot involved) to a palm pose in the pelvis frame; returns q29, pos err, rot err.
        q_bias (7): preferred arm joints pulled in through the 1-D null space (the elbow swivel) - the palm pose is
        unchanged, the elbow swings. Used to keep the upper arm off the chest when reaching across the midline."""
        q = np.asarray(q29, dtype=float).copy()
        I7 = np.eye(7)
        lam_I6 = (args.ik_lambda ** 2) * np.eye(6)
        lo, hi = self.lo[ARM] + 0.03, self.hi[ARM] - 0.03
        qb = None if q_bias is None else np.asarray(q_bias, dtype=float)
        for _ in range(iters):
            p, R, J = self.fk_jac(q)
            e = np.concatenate([p_goal - p, pin.log3(R_goal @ R.T)])
            converged = np.linalg.norm(e[:3]) < 0.001 and np.linalg.norm(e[3:]) < 0.01
            if converged and qb is None:
                break
            JJt = J @ J.T + lam_I6
            dq = np.clip(0.5 * (J.T @ np.linalg.solve(JJt, e)), -0.05, 0.05)
            if qb is not None:
                N = I7 - J.T @ np.linalg.solve(JJt, J)
                dq_ns = np.clip(N @ (k_ns * (qb - q[ARM])), -0.03, 0.03)
                if converged and np.max(np.abs(dq_ns)) < 1e-4:
                    break
                dq = dq + dq_ns
            q[ARM] = np.minimum(np.maximum(q[ARM] + dq, lo), hi)
        p, R = self.fk(q)
        return q, float(np.linalg.norm(p_goal - p)), float(np.linalg.norm(pin.log3(R_goal @ R.T)))


def reach_arm_guesses(q_body):
    """Forward-reach arm guesses (not the overhead raise basin). Retreat just hit a midline can from
    q≈[-1.3, -0.75, …] at 0.2 mm while the raise-chain IK reported 257 mm for the same xy."""
    templates = [
        (-0.70, 0.45, -0.30, 0.25, -0.50, 0.30, 0.30),
        (-1.20, -0.40, 0.80, 0.20, 0.50, 0.00, 0.00),
        (-0.90, 0.20, -0.50, 0.50, -0.30, 0.20, 0.50),
        (0.20, 0.30, 0.00, 0.80, 0.00, 0.00, 0.00),
        (-0.45, 0.75, -0.55, -0.55, -0.85, 0.45, 0.90),
        # elbow-out / abducted: left arm across the midline (live approach that cleared the chest)
        (-0.73, 0.35, -0.67, -0.36, -0.71, 0.83, 1.14),
        (-0.95, 0.65, -0.45, 0.05, -0.55, 0.50, 0.70),
    ]
    out = []
    flip = 1.0 if SIDE == "left" else -1.0
    for pitch, roll, yaw, elbow, wr, wp, wy in templates:
        q = np.asarray(q_body, dtype=float).copy()
        q[ARM] = np.array([pitch, flip * roll, flip * yaw, elbow, flip * wr, wp, flip * wy], dtype=float)
        out.append(q)
    return out


def ik_best(seeds, p_goal, R_goal, iters=200):
    """Try several IK seeds; keep the lowest position error. DLS from an overhead pose is a different homotopy."""
    best = None
    for qs in seeds:
        q, ep, er = KIN.ik(np.asarray(qs, dtype=float).copy(), p_goal, R_goal, iters=iters)
        if best is None or ep < best[1] - 1e-4 or (abs(ep - best[1]) < 1e-3 and er < best[2]):
            best = (q, ep, er)
        if ep < 0.005 and er < 0.05:
            break
    return best


def ik_seeds_from(q_now):
    return [np.asarray(q_now, dtype=float).copy()] + reach_arm_guesses(q_now)


def shoulder_abduction(q_arm):
    """Positive = working shoulder rolled away from the torso."""
    return -SGN * float(np.asarray(q_arm, dtype=float)[1])


def can_on_far_side():
    """Can is on the other arm's side of the midline (left arm to -y, right arm to +y)."""
    if CAN.get("y") is None:
        return False
    return (SGN * float(CAN["y"])) > 0.0


def ik_near(seeds, p_goal, R_goal, q_attract, max_ep=0.025, iters=200, q_bias=None, k_ns=0.4, min_abduction=0.0):
    """Among IK solutions that hit the palm, pick the one closest in joint space to q_attract.
    Stops DLS from switching back to the overhead homotopy on a via that is still near the hip.
    min_abduction: among accurate hits, prefer an abducted shoulder (across-midline). Accuracy still wins."""
    qa = np.asarray(q_attract, dtype=float)[ARM]
    best = None
    fallback = None
    for qs in seeds:
        q, ep, er = KIN.ik(np.asarray(qs, dtype=float).copy(), p_goal, R_goal, iters=iters, q_bias=q_bias, k_ns=k_ns)
        dq = float(np.max(np.abs(q[ARM] - qa)))
        if fallback is None or ep < fallback[1]:
            fallback = (q, ep, er, dq)
        if ep <= max_ep and er < 0.18:
            hug = 0.0
            if min_abduction > 0.0 and shoulder_abduction(q[ARM]) < min_abduction:
                hug = min_abduction - shoulder_abduction(q[ARM])
            cost = (20.0 * ep + hug + 0.1 * dq) if min_abduction > 0.0 else dq
            if best is None or cost < best[3]:
                best = (q, ep, er, cost)
    pick = best if best is not None else fallback
    return pick[0], pick[1], pick[2]


def is_reaching_arm(q_arm):
    """Overhead raise is shoulder_pitch ≈ +1.6; a table reach is negative / near zero."""
    return float(np.asarray(q_arm, dtype=float)[0]) < 0.35


def arm_limit_margin(q_arm):
    lo = np.asarray(q_arm, dtype=float) - KIN.lo[ARM]
    hi = KIN.hi[ARM] - np.asarray(q_arm, dtype=float)
    return float(np.min(np.minimum(lo, hi)))


def palm_level_at_q(q, R_pL):
    p, _ = KIN.fk(q)
    return R_pL.T @ p


def table_edge_x():
    """Near edge of the table (level x). From D435i when --look ran; else a conservative stand-off."""
    if CAN.get("table_x") is not None:
        return float(CAN["table_x"])
    if CAN.get("x") is not None:
        return max(0.14, float(CAN["x"]) - 0.16)
    return 0.22


def ready_palm_L(p_now_L):
    """Palm pose in the free space between the robot and the table, at LOOK table + --clearance."""
    x = min(max(0.10, table_edge_x() - 0.08), table_edge_x() - 0.05)
    y = float(p_now_L[1])
    if CAN.get("y") is not None:
        y = 0.7 * y + 0.3 * float(CAN["y"])
    z = float(p_now_L[2]) + 0.20
    if CAN.get("z") is not None:
        z = float(CAN["z"]) + float(args.clearance)
    return np.array([x, y, z])


# Self-collision proxy (level frame, pelvis origin). Boxes hold the arm *centreline*, so they include the arm/hand
# radius. From the URDF: shoulders y=+/-0.10 z=0.29, upper arm hangs at y=0.147, D435 at z=0.47, hips y=+/-0.12.
# The fold that hit the robot tonight had shoulder_roll -0.79 with the arm hanging: elbow at y=0.13, into the torso.
# Raw mesh extents in the pelvis frame (STL bounding boxes): the arm points below carry their own radius and a point
# violates a box when its Euclidean distance to the box is under that radius (rounded box, so a cylinder passing a
# box corner is judged correctly - a plain inflated box rejected the forward reach by 1 mm at the chest corner).
BODY_BOXES = [  # (name, x_min, x_max, |y|_max, z_min, z_max)
    ("torso", -0.07, 0.08, 0.108, 0.035, 0.36),
    ("head", -0.07, 0.07, 0.078, 0.37, 0.575),
    ("hips", -0.08, 0.10, 0.165, -0.48, -0.07),
    ("pelvis", -0.05, 0.05, 0.064, -0.15, 0.035),
]
# Sphere radii on the arm centreline. Upper-arm 35 mm matches the mesh; a 6 mm slack on that sphere
# only is for wraps past midline that graze the front chest corner in the proxy (live: 1-5 mm).
ARM_R = {"upper arm": 0.035, "elbow": 0.034, "forearm": 0.03, "wrist": 0.03, "palm": 0.025, "fingertips": 0.02,
         "palm face": 0.015, "hand back": 0.015}


def _box_dist(p, b):
    dx = max(b[1] - p[0], 0.0, p[0] - b[2])
    dy = max(abs(p[1]) - b[3], 0.0)
    dz = max(b[4] - p[2], 0.0, p[2] - b[5])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _arm_points_L(sr, el, wr, pp, R, R_pL):
    L = R_pL.T
    el_L, wr_L, pp_L = L @ el, L @ wr, L @ pp
    f, n = L @ R[:, 0], L @ R[:, 2]
    return [("upper arm", L @ (sr + 0.6 * (el - sr))), ("elbow", el_L), ("forearm", el_L + 0.5 * (wr_L - el_L)),
            ("wrist", wr_L), ("palm", pp_L), ("fingertips", pp_L + 0.08 * f),
            ("palm face", pp_L + 0.035 * n), ("hand back", pp_L - 0.035 * n)]


def _body_violation(sr, el, wr, pp, R, R_pL):
    for name, p in _arm_points_L(sr, el, wr, pp, R, R_pL):
        r = ARM_R[name]
        # 6 mm slack on the upper-arm sphere: a wrap past midline grazes the front-left chest
        # corner in this proxy (1-5 mm) while the real tube misses it. Still refuses a hang-fold into the ribs.
        slack = 0.006 if name == "upper arm" else 0.0
        for b in BODY_BOXES:
            d = _box_dist(p, b)
            if d + slack < r:
                return "%s %.0f mm into the %s (level %s)" % (name, (r - d) * 1000, b[0], np.round(p, 3).tolist())
    return None


def body_clear(q29, R_pL):
    """(ok, why): the working arm's upper arm / forearm / hand stays off the torso, head, hips and pelvis."""
    sr, el, wr, pp, R = KIN.fk_arm(q29)
    why = _body_violation(sr, el, wr, pp, R, R_pL)
    return why is None, why


def path_check(q0_arm, q1_arm, q_body, R_pL, n=None, table=True, from_below=False):
    """Sample the joint-linear interpolant q0 -> q1 (one sample per <=0.05 rad of the largest joint move). (ok, why).
    Table: palm >= table+4 cm and fingertips >= table+2.5 cm whenever they are past the table's near edge.
    from_below: recovery from a pose already under those lines - the floors become 'never lower than at q0 - 5 mm'.
    Body: BODY_BOXES on every sample (the endpoint included)."""
    q = np.asarray(q_body, dtype=float).copy()
    q0 = np.asarray(q0_arm, dtype=float)
    q1 = np.asarray(q1_arm, dtype=float)
    if n is None:
        n = int(max(21, math.ceil(float(np.max(np.abs(q1 - q0))) / 0.05) + 1))
    edge = table_edge_x() - 0.03
    zt = None if CAN.get("z") is None else float(CAN["z"])
    z_palm_min = None if zt is None else zt + 0.04
    z_tip_min = None if zt is None else zt + 0.025
    if from_below and zt is not None:
        q[ARM] = q0
        _sr, _el, _wr, pp0, R0 = KIN.fk_arm(q)
        pL0 = R_pL.T @ pp0
        tip0 = pL0 + 0.08 * (R_pL.T @ R0[:, 0])
        z_palm_min = min(z_palm_min, float(pL0[2]) - 0.005)
        z_tip_min = min(z_tip_min, float(tip0[2]) - 0.005)
    for i in range(n):
        a = i / float(n - 1)
        q[ARM] = q0 + (q1 - q0) * a
        sr, el, wr, pp, R = KIN.fk_arm(q)
        why = _body_violation(sr, el, wr, pp, R, R_pL)
        if why is not None:
            return False, "sample %d/%d: %s" % (i, n - 1, why)
        if table and zt is not None:
            pL = R_pL.T @ pp
            tip = pL + 0.08 * (R_pL.T @ R[:, 0])
            if pL[0] > edge and pL[2] < z_palm_min:
                return False, "palm dips to z_L %.3f at x=%.2f (table %.3f)" % (pL[2], pL[0], zt)
            if tip[0] > edge and tip[2] < z_tip_min:
                return False, "fingertips dip to z_L %.3f at x=%.2f (table %.3f)" % (tip[2], tip[0], zt)
    return True, None


def path_table_margin(q0_arm, q1_arm, q_body, R_pL, n=None):
    """Closest approach (m) of the palm / fingertips to the table slab {x >= edge, z <= top} along the interpolant.
    None without a table. 0 = touching/inside."""
    if CAN.get("z") is None:
        return None
    q = np.asarray(q_body, dtype=float).copy()
    q0 = np.asarray(q0_arm, dtype=float)
    q1 = np.asarray(q1_arm, dtype=float)
    if n is None:
        n = int(max(21, math.ceil(float(np.max(np.abs(q1 - q0))) / 0.05) + 1))
    edge, zt = table_edge_x(), float(CAN["z"])
    best = float("inf")
    for i in range(n):
        q[ARM] = q0 + (q1 - q0) * (i / float(n - 1))
        p, R = KIN.fk(q)
        pL = R_pL.T @ p
        for pt, r in ((pL, 0.025), (pL + 0.08 * (R_pL.T @ R[:, 0]), 0.015)):
            d = math.sqrt(max(0.0, edge - pt[0]) ** 2 + max(0.0, pt[2] - zt) ** 2) - r
            best = min(best, max(0.0, d))
    return best


def spline_clears_table(q0_arm, q1_arm, q_body, R_pL):
    """compat: (ok, why) - table + body check of the joint interpolant."""
    return path_check(q0_arm, q1_arm, q_body, R_pL)


def find_hip_reach_q(q_body, R_pL, R_des, q_attract):
    """Best ready config (see find_hip_reach_candidates) or (None, None, None)."""
    c = find_hip_reach_candidates(q_body, R_pL, R_des, q_attract)
    if not c:
        return None, None, None
    return c[0][1], c[0][2], c[0][3]


def find_hip_reach_candidates(q_body, R_pL, R_des, q_attract):
    """Ready configs with the palm in front of the table edge (LOOK table_x), reachable from q_body by one checked
    joint move. Sorted best first: [(score, q29, ep, palm_level), ...]. The caller (dryrun) takes the first one
    from which the via chain to pregrasp also verifies - a local score alone flipped between LOOK jitters."""
    q_body = np.asarray(q_body, dtype=float)
    p_now = palm_level_at_q(q_body, R_pL)
    z_hi = float(palm_clearance_z() if palm_clearance_z() is not None else 0.25)
    edge = table_edge_x()
    x_ready = min(max(0.10, edge - 0.08), edge - 0.05)
    y = float(p_now[1])
    y2 = 0.7 * y + 0.3 * float(CAN["y"]) if CAN.get("y") is not None else y
    xs = sorted(set([round(v, 3) for v in (0.10, 0.14, 0.18, x_ready, edge - 0.06) if 0.08 <= v <= edge - 0.04]))
    targets = []
    for x in xs:
        for yy in (y, y2, 0.5 * (y + y2)):
            targets.append(np.array([x, yy, z_hi]))
            if CAN.get("z") is not None:
                targets.append(np.array([x, yy, float(CAN["z"]) + 0.05]))
    seeds = reach_arm_guesses(q_body) + [q_attract, q_body]
    out = []
    rejects = {}

    def rej(k):
        rejects[k] = rejects.get(k, 0) + 1

    for tgt in targets:
        q, ep, er = ik_near(seeds, R_pL @ tgt, R_des, q_attract, max_ep=0.02)
        if args.offline:
            log("    ready-cand %s -> %.1f mm / %.0f deg, q %s" % (np.round(tgt, 3).tolist(), ep * 1000, math.degrees(er), np.round(q[ARM], 2).tolist()))
        # no shoulder-pitch filter here: the palm at chest height in front of the shoulder is held with the upper arm
        # hanging slightly back and the elbow fully flexed (pitch +0.5..+1.4). Whether that pose and the way into it are
        # safe is decided by path_check (table + body), not by a basin heuristic.
        if ep > 0.015 or er > 0.18:
            rej("ik")
            continue
        pL = palm_level_at_q(q, R_pL)
        if float(pL[0]) > edge - 0.04:
            rej("over-edge")
            continue
        lo_m = q[ARM] - KIN.lo[ARM]
        hi_m = KIN.hi[ARM] - q[ARM]
        n_tight = int(np.sum(np.minimum(lo_m, hi_m) < 0.08))
        roll_out = -SGN * float(q[ARM][1])          # shoulder abduction (positive = arm away from the torso)
        if n_tight > 1 or roll_out > 1.6 or roll_out < 0.05:
            rej("joint limits (%d tight, roll %.2f)" % (n_tight, roll_out))   # only the fully flexed elbow may sit on a stop
            continue
        ok, why = path_check(q_body[ARM], q[ARM], q_body, R_pL)
        if not ok:
            rej(why.split(":")[-1].strip()[:40])
            continue
        dq = float(np.max(np.abs(q[ARM] - np.asarray(q_attract)[ARM])))
        margin = arm_limit_margin(q[ARM])
        tm = path_table_margin(q_body[ARM], q[ARM], q_body, R_pL)
        tm = 0.10 if tm is None else tm
        score = (ep + 0.02 * dq + 0.2 * max(0.0, z_hi - float(pL[2])) + 0.3 * max(0.0, 0.15 - margin)
                 + 0.10 * max(0.0, 0.25 - roll_out) + 0.10 * max(0.0, roll_out - 1.1) + 1.0 * max(0.0, 0.06 - tm))
        if any(float(np.max(np.abs(o[1][ARM] - q[ARM]))) < 0.05 for o in out):
            continue                                  # duplicate config from a neighbouring target
        out.append((score, q, ep, pL))
    out.sort(key=lambda c: c[0])
    if not out:
        log("REACH-FOLD: no ready config in x %.2f..%.2f passed (%s)" % (
            xs[0] if xs else 0.0, xs[-1] if xs else 0.0, ", ".join("%s x%d" % kv for kv in sorted(rejects.items(), key=lambda kv: -kv[1]))))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# DDS: lowstate in, arm_sdk out (50 Hz thread)
# ----------------------------------------------------------------------------------------------------------------------
if not args.offline:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: E402
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_  # noqa: E402
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402
    from unitree_sdk2py.utils.crc import CRC  # noqa: E402

LOCK = threading.Lock()
STATE = {"q": None, "dq": None, "tau": None, "rpy": None, "mode_machine": None, "t": None, "n": 0}
LOWCMD = {"kp": None, "kd": None, "n": 0, "t": None, "skip": 0}
HAND = {"q": None, "t": None, "n": 0}
HAND_LAST_RC = 0
FOREIGN = []
ROWS = []          # 50 Hz: [t, q_meas(7), q_cmd(7), tau(7), palm_meas(3), palm_des(3), err_mm, rot_err_deg, weight]
SUMMARY = {"stages": {}}
STILLS = []


def on_lowstate(m):
    ms = m.motor_state
    with LOCK:
        STATE["q"] = np.array([ms[i].q for i in range(29)])
        STATE["dq"] = np.array([ms[i].dq for i in range(29)])
        STATE["tau"] = np.array([ms[i].tau_est for i in range(29)])
        STATE["rpy"] = np.array(list(m.imu_state.rpy)[:3])
        STATE["mode_machine"] = int(m.mode_machine)
        STATE["t"] = now()
        STATE["n"] += 1


def on_lowcmd(m):
    # 1 kHz LowCmd_ in pure Python will starve RPC; keep ~50 Hz for the kp/kd gate
    with LOCK:
        LOWCMD["skip"] = LOWCMD.get("skip", 0) + 1
        if LOWCMD["skip"] % 20 != 1:
            return
        LOWCMD["kp"] = np.array([m.motor_cmd[i].kp for i in range(29)])
        LOWCMD["kd"] = np.array([m.motor_cmd[i].kd for i in range(29)])
        LOWCMD["n"] += 1
        LOWCMD["t"] = now()


def on_hand(m):
    st = m.states
    if len(st) < 6:
        return
    with LOCK:
        HAND["q"] = [float(st[i].q) for i in range(6)]
        HAND["t"] = now()
        HAND["n"] += 1


def on_arm_sdk(m):
    if int(m.motor_cmd[SIGNATURE_SLOT].reserve) == SIGNATURE:
        return
    with LOCK:
        FOREIGN.append(dict(t=round(now(), 3), weight=float(m.motor_cmd[WEIGHT_SLOT].q),
                            kp=[float(m.motor_cmd[i].kp) for i in UPPER_IDX]))


def state():
    with LOCK:
        if STATE["q"] is None:
            return None
        return dict(q=STATE["q"].copy(), dq=STATE["dq"].copy(), tau=STATE["tau"].copy(), rpy=STATE["rpy"].copy(),
                    mode_machine=STATE["mode_machine"], t=STATE["t"])


if args.offline:
    if args.stage not in ("dryrun", "check"):
        sys.exit("--offline only supports --stage dryrun/check")
    _q29 = np.zeros(29)
    _q29[ARM] = np.array([float(v) for v in args.offline_q.split(",")])
    _rp = [math.radians(float(v)) for v in args.offline_rpy.split(",")]
    with LOCK:
        STATE.update(q=_q29, dq=np.zeros(29), tau=np.zeros(29), rpy=np.array([_rp[0], _rp[1], 0.0]), mode_machine=5, t=0.0, n=1)
    sub_lowcmd = None
    crc = None
else:
    ChannelFactoryInitialize(args.domain, args.iface)
    sub_state = ChannelSubscriber("rt/lowstate", LowState_)
    sub_state.Init(on_lowstate, 0)
    sub_armsdk = ChannelSubscriber("rt/arm_sdk", LowCmd_)
    sub_armsdk.Init(on_arm_sdk, 0)
    sub_hand = ChannelSubscriber("rt/brainco/%s/state" % SIDE, MotorStates_)
    sub_hand.Init(on_hand, 0)
    sub_lowcmd = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub_lowcmd.Init(on_lowcmd, 0)
    crc = CRC()
pub = None  # created only when a motion stage starts

KIN = Kin(args.urdf, SIDE)


class ArmSdk(threading.Thread):
    """50 Hz publisher. Owns the command vector; the main thread hands it motions."""

    def __init__(self):
        super().__init__(daemon=True)
        self.cmd = None            # (29,) targets; only UPPER_IDX are sent
        self.kp = np.zeros(29)
        self.kd = np.zeros(29)
        self.weight = 0.0
        self.motion = None         # dict(kind='joint'|'weight', ...); palm goals become a joint spline
        self.frozen = None
        self.running = True
        self.q_nom = None
        self.R_pL = np.eye(3)      # pelvis <- level frame rotation at the last stage start
        self.last_err = None
        self.over_since = {}
        self.progress = {"best": None, "t": None}
        self.q_attract = None
        self.q_safe = None         # last arm command while the palm was above the table
        self.allow_dip = False     # park-lift may start from a pose already below the freeze line
        self.q_home = None         # park target when it is not the takeover pose (recover: the original hang pose)
        self.ierr = np.zeros(7)    # gravity-sag compensation: bounded integral of the joint error, learned while holding
        self.mode_machine = 0

    def take_over(self, st):
        self.mode_machine = st["mode_machine"]
        self.cmd = st["q"].copy()
        for i in ARM:
            self.kp[i], self.kd[i] = args.kp, args.kd
        for i in OTHER:
            self.kp[i], self.kd[i] = args.other_kp, args.other_kd
        for i in WAIST_IDX:
            self.kp[i], self.kd[i] = args.waist_kp, args.waist_kd
        self.q_nom = st["q"][ARM].copy()
        self.q_start = st["q"].copy()
        self.waist_target = np.array(args.waist_upright, dtype=float)
        self.waist_ierr = np.zeros(3)

    def hold_waist(self, st):
        """torso upright: command = target + bounded integral of the waist error, rate-limited (<= 0.05 rad/s), so the
        torso straightens over a few seconds instead of snapping. kp 120 * 0.15 rad = 18 Nm of extra hold."""
        if st is None or self.cmd is None:
            return
        qw = st["q"][WAIST_IDX]
        self.waist_ierr = np.clip(self.waist_ierr + 0.6 * (self.waist_target - qw) * DT, -0.15, 0.15)
        want = self.waist_target + self.waist_ierr
        step = np.clip(want - self.cmd[WAIST_IDX], -0.05 * DT, 0.05 * DT)
        self.cmd[WAIST_IDX] = self.cmd[WAIST_IDX] + step

    def set_level(self, st):
        r, p = float(st["rpy"][0]), float(st["rpy"][1])
        self.R_pL = rpy_to_mat(r, p, 0.0).T

    def palm_now(self, st):
        p, R = KIN.fk(st["q"])
        return p, R

    def start_palm_motion(self, p_goal_L, R_goal, T, label, contact=False, q_plan=None):
        """Plan a joint-space cosine spline to a palm pose (same recipe as sim/g1_walk_grasp.py).
        IK runs once from the current command; the 50 Hz loop never servos. contact=True: after the spline
        the command may lead its a=1 value by at most --press-lead, and the move is done as soon as the palm
        stops making progress - the arm never winds up against the can.
        q_plan: use these (dryrun-verified) arm joints as the spline end instead of solving IK live - a live
        re-solve picked a different IK basin than the dryrun tonight and folded the arm into the torso."""
        st = state()
        p0, R0 = self.palm_now(st)
        p1 = self.R_pL @ np.asarray(p_goal_L)
        R1 = self.R_pL @ R_goal if R_goal is not None else R0
        q_seed = st["q"].copy()
        # cmd is already q_nom + ierr (sag hold). Spline in nominal joints so tick's
        # q_des = lerp(q0, q1) + ierr equals cmd at a=0; using cmd as q0 double-counted
        # ierr and popped the palm a few cm up at every waypoint.
        q0 = self.cmd[ARM].copy() - self.ierr
        q_seed[ARM] = q0
        if q_plan is not None:
            q_est = q_seed.copy()
            q_est[ARM] = np.asarray(q_plan, dtype=float)[ARM] if len(q_plan) == 29 else np.asarray(q_plan, dtype=float)
            p_chk, R_chk = KIN.fk(q_est)          # planned arm joints on the CURRENT waist / legs
            drift = float(np.linalg.norm(p1 - p_chk))
            if drift > 0.15:
                log("MOVE %s: the planned joints put the palm %.0f mm from the planned level goal - stale plan? refusing" % (label, drift * 1000))
                return False
            # The standing controller leans the waist as the arm goes out (5 cm at the palm by via-hover): re-solve from
            # the planned joints for the level-frame goal with the live posture. Same IK basin, small correction only.
            picked = None
            g_L = np.asarray(p_goal_L, dtype=float)
            for dz in (0.0, -0.02, -0.04, 0.03):
                if dz != 0.0 and CAN.get("z") is not None and g_L[2] + dz < float(CAN["z"]) + 0.045:
                    continue
                g_p = self.R_pL @ (g_L + np.array([0.0, 0.0, dz]))
                q_corr, ep_c, er_c = KIN.ik(q_est.copy(), g_p, R1, iters=120)
                dq_c = float(np.max(np.abs(q_corr[ARM] - q_est[ARM])))
                if ep_c < 0.01 and er_c < 0.15 and dq_c < 0.6:
                    picked = (dz, q_corr, ep_c, er_c, dq_c, g_p)
                    break
            if picked is not None:
                dz, q_corr, ep_c, er_c, dq_c, g_p = picked
                log("MOVE %s: posture correction %.0f -> %.1f mm%s (dq %.2f rad; waist %s, rpy %s deg)" % (
                    label, drift * 1000, ep_c * 1000, "" if dz == 0.0 else " with the waypoint %+.0f cm in z" % (dz * 100), dq_c,
                    np.round(st["q"][WAIST_IDX], 3).tolist(), np.degrees(st["rpy"]).round(1).tolist()))
                q_est, ep, er, p1 = q_corr, ep_c, er_c, g_p
            else:
                log("MOVE %s: posture correction rejected (%.1f mm / %.1f deg / dq %.2f) - following the planned joints, %.0f mm off the level goal" % (
                    label, ep_c * 1000, math.degrees(er_c), dq_c, drift * 1000))
                p1, R1 = p_chk, R_chk
                ep = er = 0.0
        else:
            attract = self.q_attract if self.q_attract is not None else q_seed
            seeds = ik_seeds_from(q_seed)
            pL_goal = np.asarray(p_goal_L, dtype=float)
            hip = float(pL_goal[0]) <= table_edge_x()
            lifting = hip and CAN.get("z") is not None and not is_reaching_arm(q0)
            if lifting:
                attract = q_seed
            elif hip and CAN.get("z") is not None:
                seeds = reach_arm_guesses(q_seed) + seeds
            q_est, ep, er = ik_near(seeds, p1, R1, attract)
        q1 = np.minimum(np.maximum(q_est[ARM], KIN.lo[ARM] + 0.03), KIN.hi[ARM] - 0.03)
        if ep > 0.04 or er > 0.20:
            log("MOVE %s: IK rest %.1f mm / %.1f deg - refusing this pose (would drive the arm into a bad config)" % (
                label, ep * 1000, math.degrees(er)))
            self.freeze("IK cannot reach %s (%.0f mm / %.1f deg)" % (label, ep * 1000, math.degrees(er)))
            return False
        if not contact and ep > 0.025:
            log("MOVE %s: IK rest %.1f mm - too weak for a free-space table move, not starting" % (label, ep * 1000))
            return False
        ok_path, why = path_check(q0, q1, q_seed, self.R_pL, table=not contact, from_below=self.allow_dip)
        if not ok_path:
            log("MOVE %s: refusing joint spline (%s) dq %.2f rad" % (label, why, float(np.max(np.abs(q1 - q0)))))
            return False
        self.q_nom = q0.copy()
        dq_max = float(np.max(np.abs(q1 - q0)))
        ts, vm, fine = clock_for(label, contact)
        T_eff = max(0.3, T * ts, 1.57 * dq_max / (0.8 * vm))
        self.progress = {"best": None, "t": now()}
        # free-space waypoints may finish as soon as the joints are on the plan (see tick); pregrasp keeps the full
        # dwell so the sag integral has settled right before the contact moves
        early = (not contact) and (not args.no_early_arrive) and label != "pregrasp"
        self.motion = dict(kind="joint", q0=q0, q1=q1, T=T_eff, t0=now(), label=label, done=False,
                           contact=contact, cmd_a1=None, p0=p0, p1=p1, R1=R1, early=early, vmax=vm)
        log("MOVE %s: joint spline palm %s -> %s (pelvis), %.1fs%s, dq_max %.2f rad, %s %.1f mm / %.1f deg%s | q1 %s" % (
            label, np.round(p0, 3).tolist(), np.round(p1, 3).tolist(), T_eff,
            " (stretched for vmax %.2f)" % vm if T_eff > T * ts + 1e-6 else "",
            dq_max, "planned joints, FK rest" if q_plan is not None else "IK rest", ep * 1000, math.degrees(er),
            (", contact/fine" if fine else ""), np.round(q1, 2).tolist()))
        if ep > 0.02 or er > 0.15:
            log("MOVE %s: weak IK (%.1f mm / %.1f deg) - check the goal / can estimate" % (label, ep * 1000, math.degrees(er)))
        return True

    def start_joint_motion(self, q_goal_arm, T, label):
        ts, vm, _fine = clock_for(label, False)
        self.motion = dict(kind="joint", q0=self.cmd[ARM].copy(), q1=np.asarray(q_goal_arm, dtype=float),
                           T=max(0.3, T * ts), t0=now(), label=label, done=False, vmax=vm)
        log("MOVE %s: joints %s -> %s, %.1fs" % (label, np.round(self.motion["q0"], 3).tolist(), np.round(self.motion["q1"], 3).tolist(), self.motion["T"]))

    def start_weight(self, w_goal, T, label):
        self.motion = dict(kind="weight", w0=self.weight, w1=w_goal, T=max(0.2, T), t0=now(), label=label, done=False)
        log("WEIGHT %s: %.2f -> %.2f over %.1fs" % (label, self.weight, w_goal, T))

    def wait(self, timeout=None):
        """block until the current motion is done (or frozen)."""
        while True:
            m = self.motion               # the 50 Hz thread may null this on a freeze - never re-read it mid-check
            if m is None or m.get("done") or self.frozen is not None:
                break
            time.sleep(0.02)
            if timeout is not None and now() - m["t0"] > timeout:
                return False
        return self.frozen is None

    def unfreeze(self, why):
        """operator-driven recovery: clear the freeze and the torque timers (else the same guard re-fires at once)."""
        log("UNFREEZE (%s): was frozen for %s" % (why, self.frozen))
        with LOCK:
            self.frozen = None
        self.over_since = {}
        self.progress = {"best": None, "t": now()}
        self.ierr = np.zeros(7)       # the next command starts from the measured pose; re-learn the sag from there

    def freeze(self, why):
        if self.frozen:
            return
        st = state()
        with LOCK:
            self.frozen = why
        table_hit = any(s in why for s in ("dipped", "below the table", "through the table", "hitting the table"))
        if table_hit and st is not None and CAN.get("z") is not None:
            p_f, _ = KIN.fk(st["q"])
            if float((self.R_pL.T @ p_f)[2]) > float(CAN["z"]) + 0.005:
                table_hit = False        # guard fired above the table top: hold where we are, no command jump
        if st is not None and self.cmd is not None:
            if table_hit and self.q_safe is not None:
                self.cmd[ARM] = self.q_safe.copy()
                log("FREEZE: %s -> restoring last table-clear command (not holding the impact pose), weight %.2f" % (why, self.weight))
            else:
                self.cmd[ARM] = st["q"][ARM]   # hold where the arm physically is
                log("FREEZE: %s -> holding the measured pose (weight %.2f). Operator: park or release." % (why, self.weight))
        else:
            log("FREEZE: %s (weight %.2f)" % (why, self.weight))
        self.motion = None

    # ---------------- 50 Hz ----------------
    def run(self):
        global pub
        next_t = time.monotonic()
        while self.running:
            next_t += DT
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001
                log("tick exception %r" % (e,))
                self.freeze("exception in control tick: %r" % (e,))
            time.sleep(max(0.0, next_t - time.monotonic()))

    def tick(self):
        if self.cmd is None or pub is None:
            return
        st = state()
        t = now()
        self.guards(st, t)
        if self.weight > 0.5 and self.frozen is None:
            self.hold_waist(st)
        m = self.motion
        palm_des = None
        err = rot_err = None
        if m is not None and not m.get("done") and self.frozen is None:
            a = smoothstep((t - m["t0"]) / m["T"])
            if m["kind"] == "weight":
                self.weight = m["w0"] + (m["w1"] - m["w0"]) * a
                if a >= 1.0:
                    m["done"] = True
            elif m["kind"] == "joint":
                q_des = m["q0"] + (m["q1"] - m["q0"]) * a
                palm_planned = m.get("p1") is not None
                if palm_planned and not m.get("contact") and st is not None and a >= 1.0:
                    # kp 120 sags ~0.03 rad/joint with the arm out (4 cm at the palm; via-mid3 was frozen as "blocked"
                    # for exactly that). The plan is the joints: while holding a waypoint, slowly pull them onto it.
                    # Bounded (0.06 rad = 7 Nm) and carried into the next move as feed-forward
                    # (next spline starts at cmd-ierr so this is not applied twice).
                    self.ierr = np.clip(self.ierr + 1.5 * (m["q1"] - st["q"][ARM]) * DT, -0.06, 0.06)
                if palm_planned:
                    q_des = q_des + self.ierr       # feed-forward for contact moves too (no integration there)
                vmax = float(m.get("vmax", args.vmax))
                step = np.clip(q_des - self.cmd[ARM], -vmax * DT, vmax * DT)
                q_new = self.cmd[ARM] + step
                if st is not None:
                    # every motion is windup-limited: a blocked joint can never be asked for more than kp*windup.
                    # (the fold that hit the torso tonight was a pure joint move with no clamp: -26 Nm before the guard)
                    qm = st["q"][ARM]
                    q_new = np.minimum(np.maximum(q_new, qm - args.windup), qm + args.windup)
                    if palm_planned and a >= 1.0 and m.get("contact"):
                        if m["cmd_a1"] is None:
                            m["cmd_a1"] = self.cmd[ARM].copy()
                        q_new = np.minimum(np.maximum(q_new, m["cmd_a1"] - args.press_lead), m["cmd_a1"] + args.press_lead)
                    q_new = np.minimum(np.maximum(q_new, KIN.lo[ARM] + 0.03), KIN.hi[ARM] - 0.03)
                self.cmd[ARM] = q_new
                caught = float(np.max(np.abs(q_des - self.cmd[ARM]))) < 1e-4
                if not palm_planned and st is not None and a >= 1.0 and not caught and t - m["t0"] > m["T"] + 2.0:
                    lag = float(np.max(np.abs(q_des - st["q"][ARM])))
                    self.freeze("joint move %s blocked: %.2f rad short of the target %.1fs after its end" % (m["label"], lag, t - m["t0"] - m["T"]))
                    return
                if st is not None and not m.get("contact") and CAN.get("z") is not None:
                    p_guard, _ = KIN.fk(st["q"])
                    p_Lv = self.R_pL.T @ p_guard
                    if float(p_Lv[0]) <= table_edge_x() - 0.03 or float(p_Lv[2]) >= float(CAN["z"]) + 0.05:
                        self.q_safe = self.cmd[ARM].copy()
                    elif not self.allow_dip and float(p_Lv[0]) > table_edge_x() and float(p_Lv[2]) < float(CAN["z"]) + 0.02:
                        self.freeze("palm z_L %.3f is below the table %.3f at x=%.2f (hitting the table?)" % (
                            p_Lv[2], CAN["z"], p_Lv[0]))
                if palm_planned and st is not None and self.frozen is None:
                    p_cur, R_cur = KIN.fk(st["q"])
                    q_fk = st["q"].copy()
                    q_fk[ARM] = q_des
                    palm_des, _ = KIN.fk(q_fk)
                    err = float(np.linalg.norm(m["p1"] - p_cur))
                    rot_err = float(np.linalg.norm(pin.log3(m["R1"] @ R_cur.T)))
                    self.last_err = (err, rot_err)
                    if (not m.get("contact") and p_cur is not None):
                        p_Lv = self.R_pL.T @ p_cur
                        if CAN.get("z") is None or float(p_Lv[0]) <= table_edge_x() - 0.03 or float(p_Lv[2]) >= float(CAN["z"]) + 0.05:
                            self.q_safe = self.cmd[ARM].copy()
                        if CAN.get("z") is not None and p_Lv[0] > table_edge_x() and abs(p_Lv[1]) < 0.45 and p_Lv[2] < CAN["z"] - 0.02:
                            self.freeze("palm z_L %.3f is below the table %.3f at x=%.2f (hitting the table?)" % (
                                p_Lv[2], CAN["z"], p_Lv[0]))
                        elif m.get("p0") is not None and m.get("p1") is not None:
                            z0 = float((self.R_pL.T @ m["p0"])[2])
                            z1 = float((self.R_pL.T @ m["p1"])[2])
                            z_floor = min(z0, z1) - 0.08
                            if CAN.get("z") is not None:
                                z_floor = max(z_floor, float(CAN["z"]) + 0.03)
                            if self.allow_dip:
                                z_floor = min(z_floor, z0 - 0.03)
                            if p_Lv[0] > table_edge_x() - 0.02 and p_Lv[2] < z_floor:
                                self.freeze("palm dipped to z_L %.3f on %s (floor %.3f) - interpolant through the table?" % (
                                    p_Lv[2], m["label"], z_floor))
                    if a >= 1.0 and self.frozen is None:
                        pr = self.progress
                        if pr["best"] is None or err < pr["best"] - 0.001:
                            pr["best"], pr["t"] = err, t
                        budget_out = bool(m.get("contact") and m.get("cmd_a1") is not None and
                                          np.any(np.abs(q_new - m["cmd_a1"]) >= args.press_lead - 1e-6))
                        if not m.get("contact"):
                            dwell = t - m["t0"] - m["T"]
                            jerr = np.abs(m["q1"] - st["q"][ARM])
                            if dwell > 1.5:
                                if float(jerr.max()) > 0.15 or err > 0.08:
                                    self.freeze("%s blocked: joint %s is %.2f rad off its target, palm %.0f mm off" % (
                                        m["label"], JOINT_NAMES[ARM[int(jerr.argmax())]], float(jerr.max()), err * 1000))
                                else:
                                    m["done"] = True
                                    log("ARRIVED %s: palm err %.1f mm, rot %.1f deg, palm %s | joint err max %.3f rad, sag comp %s" % (
                                        m["label"], err * 1000, math.degrees(rot_err), np.round(p_cur, 3).tolist(), float(jerr.max()),
                                        np.round(self.ierr, 3).tolist()))
                            elif m.get("early") and dwell >= 0.2 and float(jerr.max()) < 0.02 and err < 0.008:
                                # the joints are on the plan already (the 1.5 s dwell was for the sag integral, which is
                                # carried into the next move anyway); cycle 2 spent 1.5 s at every one of 14 waypoints
                                m["done"] = True
                                log("ARRIVED %s (early, %.1fs after the spline): palm err %.1f mm, rot %.1f deg, palm %s | joint err max %.3f rad, sag comp %s" % (
                                    m["label"], dwell, err * 1000, math.degrees(rot_err), np.round(p_cur, 3).tolist(), float(jerr.max()),
                                    np.round(self.ierr, 3).tolist()))
                        elif m.get("contact") and err < args.pos_tol and rot_err < args.rot_tol:
                            m["done"] = True
                            log("ARRIVED %s: palm err %.1f mm, rot %.1f deg, palm %s" % (
                                m["label"], err * 1000, math.degrees(rot_err), np.round(p_cur, 3).tolist()))
                        elif m.get("contact") and (budget_out or t - pr["t"] > 1.0):
                            m["done"] = True
                            log("SETTLED %s: palm err %.1f mm, rot %.1f deg - %s; holding (press lead <= %.3f rad)" % (
                                m["label"], err * 1000, math.degrees(rot_err),
                                "press budget used" if budget_out else "no more progress", args.press_lead))
                        elif m.get("contact") and t - m["t0"] > m["T"] + 6.0:
                            m["done"] = True
                            log("TIMEOUT %s: palm err %.1f mm, rot %.1f deg (holding here)" % (
                                m["label"], err * 1000, math.degrees(rot_err)))
                        elif m.get("contact") and err > args.pos_tol * 2 and t - pr["t"] > args.stall_seconds:
                            self.freeze("no progress toward %s for %.1fs (palm err %.1f mm) - blocked?" % (
                                m["label"], t - pr["t"], err * 1000))
                elif a >= 1.0 and caught:
                    m["done"] = True
        self.publish(st, t, palm_des, err, rot_err)

    def guards(self, st, t):
        if self.frozen is not None:
            return
        if st is None or t - st["t"] > 0.3:
            self.freeze("rt/lowstate stale (%.2fs)" % (t - (st["t"] if st else 0.0)))
            return
        if abs(st["rpy"][0]) > args.max_tilt or abs(st["rpy"][1]) > args.max_tilt:
            self.freeze("pelvis tilt %.2f/%.2f rad" % (st["rpy"][0], st["rpy"][1]))
            return
        for i in ARM:
            frac = abs(st["tau"][i]) / EFFORT[i]
            if frac > args.torque_frac:
                if i not in self.over_since:
                    self.over_since[i] = t
                elif t - self.over_since[i] > args.torque_seconds:
                    self.freeze("over-torque %s %.1f Nm for %.2fs" % (JOINT_NAMES[i], st["tau"][i], t - self.over_since[i]))
                    return
            else:
                self.over_since.pop(i, None)
        with LOCK:
            nf = len(FOREIGN)
        if nf:
            self.freeze("another publisher on rt/arm_sdk (%d msgs, first %s)" % (nf, FOREIGN[0]))
            return
        if t > args.max_seconds:
            self.freeze("session watchdog %.0fs" % args.max_seconds)

    def publish(self, st, t, palm_des, err, rot_err):
        msg = unitree_hg_msg_dds__LowCmd_()
        msg.mode_pr = 0
        msg.mode_machine = self.mode_machine
        for i in UPPER_IDX:  # same fields Unitree's example and the August program write (mode left at 0)
            mc = msg.motor_cmd[i]
            mc.q = float(self.cmd[i])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(self.kp[i])
            mc.kd = float(self.kd[i])
        msg.motor_cmd[WEIGHT_SLOT].q = float(min(1.0, max(0.0, self.weight)))
        msg.motor_cmd[SIGNATURE_SLOT].reserve = SIGNATURE
        msg.crc = crc.Crc(msg)
        pub.Write(msg)
        if st is not None:
            p_cur, _ = KIN.fk(st["q"]) if palm_des is not None else (np.full(3, np.nan), None)
            row = [round(t, 3)] + [round(float(x), 4) for x in st["q"][ARM]] + [round(float(x), 4) for x in self.cmd[ARM]] + \
                  [round(float(x), 2) for x in st["tau"][ARM]] + [round(float(x), 4) for x in p_cur] + \
                  ([round(float(x), 4) for x in palm_des] if palm_des is not None else [None] * 3) + \
                  [round(err * 1000, 1) if err is not None else None, round(math.degrees(rot_err), 1) if rot_err is not None else None, round(self.weight, 3)]
            with LOCK:
                ROWS.append(row)


ARMSDK = ArmSdk()


# ----------------------------------------------------------------------------------------------------------------------
# helpers: targets, prompts, camera, hand tool, saving
# ----------------------------------------------------------------------------------------------------------------------
CAN = {"x": args.can_x, "y": args.can_y, "z": args.can_z, "table_x": None}      # can base, level frame
R_PALM_DES = None


def palm_rotation_des():
    """palm triad target in the level frame: fingers +x, palm normal toward the midline, index above pinky."""
    f_t = np.array([1.0, 0.0, 0.0])
    n_t = np.array([0.0, SGN, 0.0])
    w_t = np.cross(n_t, f_t)
    return np.stack([f_t, w_t, n_t], axis=1)


def palm_target(gap, dz=0.0):
    """palm-centre target (level frame) for the pocket geometry: approach from the working side."""
    return np.array([CAN["x"] - args.distal_offset + args.palm_x_trim, CAN["y"] - SGN * (CAN_R + gap) + args.palm_y_trim,
                     CAN["z"] + args.grasp_height + dz])


def stage_goals():
    g = {}
    # Pregrasp stays at least 55 mm up so the via interpolant clears the 4 cm table floor.
    # Descend/lift/lower use --grasp-height (default 45 mm = wrap on the can body).
    pre_z = max(float(args.grasp_height), 0.055)
    g["pregrasp"] = palm_target(args.gap + args.pregrasp_gap, dz=pre_z - float(args.grasp_height))
    g["approach"] = palm_target(args.gap - args.press, args.approach_rise)
    g["descend"] = palm_target(args.gap - args.press)
    g["lift"] = palm_target(args.gap - args.press) + np.array([0.0, -SGN * (args.press + 0.003), args.lift])
    g["lower"] = palm_target(args.gap - args.press) + np.array([0.0, -SGN * (args.press + 0.003), 0.0])
    g["retreat"] = palm_target(args.gap + args.pregrasp_gap, args.approach_rise)
    return g


def raise_goal(p_palm_now_L):
    """Lift the palm at its current xy up to the LOOK table + --clearance. Same xy so hang-homotopy IK can do it;
    a high palm at the hip is an overhead pose and is how we used to smash the counter."""
    z = float(p_palm_now_L[2]) + 0.20
    if CAN.get("z") is not None:
        z = float(CAN["z"]) + float(args.clearance)
    return np.array([float(p_palm_now_L[0]), float(p_palm_now_L[1]), z])


def grab_still(tag):
    if not args.camera or args.auto:
        return None
    import urllib.request
    path = out_path()[:-5] + "-%s.jpg" % tag
    try:
        with urllib.request.urlopen(args.camera, timeout=3.0) as r:
            buf = b""
            t_end = time.monotonic() + 3.0
            while time.monotonic() < t_end:
                chunk = r.read(4096)
                if not chunk:
                    break
                buf += chunk
                a = buf.find(b"\xff\xd8")
                b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
                if a >= 0 and b > a:
                    with open(path, "wb") as f:
                        f.write(buf[a:b + 2])
                    STILLS.append(dict(t=round(now(), 3), tag=tag, path=path))
                    log("STILL %s -> %s (%d bytes)" % (tag, path, b + 2 - a))
                    return path
    except Exception as e:  # noqa: BLE001
        log("still %s failed: %r" % (tag, e))
    return None


_OUT = {"path": None}


def out_path():
    if _OUT["path"] is None:
        _OUT["path"] = args.out or "/tmp/g1-arm-%s-%s-%s.json" % (SIDE, args.stage, datetime.now().strftime("%Y%m%d-%H%M%S"))
    return _OUT["path"]


def save():
    with LOCK:
        rows = list(ROWS)
        foreign = list(FOREIGN)
    doc = dict(arm=SIDE, stage=args.stage, started=datetime.now().isoformat(timespec="seconds"), args=vars(args), can=CAN,
               palm_frame=PALM[SIDE], frozen=ARMSDK.frozen, events=EVENTS, foreign_arm_sdk=foreign, stills=STILLS, summary=SUMMARY,
               columns=["t"] + ["q_" + JOINT_NAMES[i] for i in ARM] + ["cmd_" + JOINT_NAMES[i] for i in ARM] + ["tau_" + JOINT_NAMES[i] for i in ARM]
               + ["palm_x", "palm_y", "palm_z", "des_x", "des_y", "des_z", "goal_err_mm", "goal_rot_deg", "weight"],
               rows=rows)
    with open(out_path(), "w") as f:
        json.dump(doc, f, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
    print("G1_ARM_TEST_OUT %s rows=%d stills=%d" % (out_path(), len(rows), len(STILLS)), flush=True)


def ask(prompt):
    """operator prompt; returns the stripped line ('' = continue). --auto waits --auto-pause and continues."""
    if args.auto:
        time.sleep(args.auto_pause)
        return ""
    if not sys.stdin.isatty():
        sys.exit("motion stages need an operator at the keyboard (or --auto)")
    try:
        return input(prompt).strip()
    except EOFError:
        log("prompt stdin closed (SSH drop?) -> park")
        return "p"
    except KeyboardInterrupt:
        print("", flush=True)
        log("prompt interrupted -> park")
        return "p"


def handle_prompt_cmd(line):
    """returns 'go' | 'park' | 'again' (stay in the prompt loop)."""
    parts = line.split()
    if not parts or parts[0] in ("go", "y", "yes", "ok"):
        return "go"
    if parts[0] == "p":
        return "park"
    if parts[0] == "x":
        ARMSDK.freeze("operator")
        return "park"
    if parts[0] == "s":
        grab_still("prompt")
        return "again"
    if parts[0] in ("n", "j") and len(parts) == 4:
        d = np.array([float(v) for v in parts[1:]]) / 100.0
        if parts[0] == "n":
            CAN["x"] += d[0]
            CAN["y"] += d[1]
            CAN["z"] += d[2]
            log("CAN estimate moved by %s cm -> %s" % (np.round(d * 100, 1).tolist(), {k: round(v, 3) for k, v in CAN.items()}))
        else:
            st = state()
            p, R = KIN.fk(st["q"])
            p_L = ARMSDK.R_pL.T @ p
            ARMSDK.start_palm_motion(p_L + d, ARMSDK.R_pL.T @ R if R_PALM_DES is None else R_PALM_DES, 1.0 + 8.0 * float(np.linalg.norm(d)), "jog", contact=True)
            ARMSDK.wait()
            grab_still("jog")
        return "again"
    print("   Enter=continue | n dx dy dz (cm, move the can estimate) | j dx dy dz (cm, jog the palm) | s still | p park | x freeze")
    return "again"


def prompt_loop(text):
    while True:
        if ARMSDK.frozen:
            return "park"
        line = ask(text)
        if ARMSDK.frozen:       # a guard fired while the operator was typing
            return "park"
        r = handle_prompt_cmd(line)
        if r != "again":
            return r


def run_hand(stage, extra):
    """run revo2_hand_test.py for one hand stage; returns its JSON doc (or None)."""
    global HAND_LAST_RC
    HAND_LAST_RC = 0
    if args.no_hand:
        log("HAND %s skipped (--no-hand)" % stage)
        return None
    out = out_path()[:-5] + "-hand-%s.json" % stage
    cmd = [sys.executable, args.hand_tool, "--hand", SIDE, "--stage", stage, "--iface", args.iface, "--mode", "ramp", "--out", out] + extra
    if SIDE == "right":
        cmd.append("--allow-right")
    log("HAND %s: %s" % (stage, " ".join(cmd[1:])))
    HAND_LAST_RC = subprocess.call(cmd)
    log("HAND %s exit %d" % (stage, HAND_LAST_RC))
    try:
        with open(out) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def hand_q_now():
    with LOCK:
        q = HAND["q"]
    return None if q is None else [float(v) for v in q]


def hand_is_open(q=None):
    """Closers (not thumb_aux) near 0. Opposed-and-open is thumb_aux=1 with the fingers out."""
    q = hand_q_now() if q is None else q
    if q is None:
        return True
    closers = [q[0], q[2], q[3], q[4], q[5]]
    return max(closers) <= 0.15 and q[1] <= 0.20


def ensure_hand_open(why):
    """Revo holds its last command across processes. Open at cycle start so oppose/close are not refused."""
    if args.no_hand:
        return True
    q = hand_q_now()
    if q is not None and not hand_is_open(q):
        log("HAND not open (q %s) - releasing before %s" % ([round(v, 2) for v in q], why))
        run_hand("release", [])
    q = hand_q_now()
    return q is None or hand_is_open(q)


def grasp_verdict(doc):
    if doc is None:
        return "no hand recording", False
    c = doc.get("summary", {}).get("close", {}).get("contact", {})
    if not c:
        return "no contact data", False
    parts = []
    ok = True
    for k, ref in FINGERPRINT.items():
        v = c.get(k)
        if v is None:
            parts.append("%s=?" % k)
            ok = False
            continue
        act = v["act"]
        miss = "no contact" in v.get("why", "")
        parts.append("%s=%.2f%s" % (k, act, "!" if miss or abs(act - ref) > 0.08 else ""))
        if miss or act > 0.45:
            ok = False
    return " ".join(parts) + (" -> can in hand (bench fingerprint)" if ok else " -> NOT the can fingerprint"), ok


# ----------------------------------------------------------------------------------------------------------------------
# read-only check
# ----------------------------------------------------------------------------------------------------------------------
def controller_state():
    with LOCK:
        kp = LOWCMD["kp"]
        n = LOWCMD["n"]
    if kp is None:
        return "no rt/lowcmd (controller not publishing)", False
    legs, arms, waist = kp[0:12], kp[15:29], kp[12:15]
    if np.max(kp[:29]) <= 0.0:
        return "ZERO TORQUE / damping (rt/lowcmd kp all 0, %d msgs): arm_sdk has no effect here" % n, False
    return ("controller active: kp legs %.0f-%.0f, waist %.0f-%.0f, arms %.0f-%.0f (%d msgs)" %
            (legs.min(), legs.max(), waist.min(), waist.max(), arms.min(), arms.max(), n)), True


def do_check(seconds=2.5, verbose=True):
    if args.offline:
        st = state()
        R_pL = rpy_to_mat(float(st["rpy"][0]), float(st["rpy"][1]), 0.0).T
        p, R = KIN.fk(st["q"])
        info = dict(offline=True, controller_ok=False, arm_q=st["q"][ARM].round(3).tolist(), palm_level=(R_pL.T @ p).round(3).tolist(),
                    pelvis_height_est=args.pelvis_height or 0.79, fsm=None)
        log("OFFLINE: arm q %s | palm (level) %s | pelvis height %.3f (assumed)" % (info["arm_q"], info["palm_level"], info["pelvis_height_est"]))
        SUMMARY["check"] = info
        return info
    time.sleep(seconds)
    st = state()
    with LOCK:
        n_state, n_hand, hand_q, n_foreign = STATE["n"], HAND["n"], HAND["q"], len(FOREIGN)
    if st is None:
        log("no rt/lowstate in %.1f s - wrong interface (--iface %s)?" % (seconds, args.iface))
        save()
        os._exit(1)
    ctrl_txt, ctrl_ok = controller_state()
    R_pL = rpy_to_mat(float(st["rpy"][0]), float(st["rpy"][1]), 0.0).T
    p, R = KIN.fk(st["q"])
    p_L = R_pL.T @ p
    n_L = R_pL.T @ R[:, 2]
    f_L = R_pL.T @ R[:, 0]
    sole = KIN.sole_min_z(st["q"], R_pL.T)
    pelvis_h = -sole
    info = dict(lowstate_hz=n_state / seconds, mode_machine=st["mode_machine"], imu_rpy_deg=np.degrees(st["rpy"]).round(1).tolist(),
                controller=ctrl_txt, controller_ok=ctrl_ok, arm_q=st["q"][ARM].round(3).tolist(), arm_tau=st["tau"][ARM].round(2).tolist(),
                palm_level=p_L.round(3).tolist(), palm_normal_level=n_L.round(2).tolist(), fingers_level=f_L.round(2).tolist(),
                pelvis_height_est=round(pelvis_h, 3), hand_hz=n_hand / seconds, hand_q=hand_q, foreign_arm_sdk=n_foreign)
    f = loco_fsm()
    info["fsm"] = f
    SUMMARY["check"] = info
    if verbose:
        log("lowstate %.0f Hz, mode_machine %d, IMU rpy %s deg" % (info["lowstate_hz"], st["mode_machine"], info["imu_rpy_deg"]))
        log(fsm_text(f))
        log(ctrl_txt)
        log("%s arm q %s | tau %s Nm" % (SIDE, info["arm_q"], info["arm_tau"]))
        log("palm centre (level frame) %s m | palm normal %s | fingers %s" % (info["palm_level"], info["palm_normal_level"], info["fingers_level"]))
        log("pelvis height estimate %.3f m (valid only with both feet flat on the floor)" % pelvis_h)
        log("%s hand: %.0f Hz, q %s" % (SIDE, info["hand_hz"], None if hand_q is None else [round(v, 2) for v in hand_q]))
        log("rt/arm_sdk foreign publishers: %d msgs" % n_foreign)
    return info


# ----------------------------------------------------------------------------------------------------------------------
# look: D435i table plane + can in the pelvis level frame (read-only, no arm_sdk)
# ----------------------------------------------------------------------------------------------------------------------
def capture_rgbd(n_frames=5):
    """Aligned RGB-D. Prefer the dashboard snapshot so we do not steal the D435i."""
    if args.vision:
        try:
            return capture_rgbd_stream(args.vision)
        except Exception as e:  # noqa: BLE001
            log("LOOK: vision stream %s unavailable (%s); opening D435i directly" % (args.vision, e))
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(10):
            pipe.wait_for_frames(5000)
        depths, color, scale = [], None, 0.001
        for _ in range(n_frames):
            frames = align.process(pipe.wait_for_frames(5000))
            df, cf = frames.get_depth_frame(), frames.get_color_frame()
            if not df or not cf:
                continue
            depths.append(np.asanyarray(df.get_data()).astype(np.float32))
            color = np.asanyarray(cf.get_data())
            scale = float(df.get_units())
        if not depths or color is None:
            raise RuntimeError("D435i opened but produced no aligned frames")
        depth = np.median(np.stack(depths, 0), axis=0) * scale
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return color, depth, dict(fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy, w=intr.width, h=intr.height)
    finally:
        pipe.stop()


def capture_rgbd_stream(base, tries=8):
    import urllib.request
    import cv2
    last_err = None
    for _ in range(tries):
        try:
            calib = json.loads(urllib.request.urlopen(base.rstrip("/") + "/calib.json", timeout=2).read())
            if not calib.get("fx"):
                raise RuntimeError("no intrinsics yet")
            color_b = urllib.request.urlopen(base.rstrip("/") + "/color.jpg", timeout=2).read()
            color = cv2.imdecode(np.frombuffer(color_b, np.uint8), cv2.IMREAD_COLOR)
            raw = urllib.request.urlopen(base.rstrip("/") + "/depth.f32", timeout=2).read()
            h, w = int(calib["h"]), int(calib["w"])
            if color is None or len(raw) < 4 * h * w:
                raise RuntimeError("incomplete snap (color %s depth %d bytes)" % (None if color is None else color.shape, len(raw)))
            depth = np.frombuffer(raw, dtype="<f4")[: h * w].reshape(h, w).copy()
            log("LOOK: RGB-D from %s (%dx%d, aligned to %s)" % (base, w, h, calib.get("aligned_to", "?")))
            return color, depth, calib
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.25)
    raise RuntimeError(last_err)


def _ransac_table(pts, n_iter=100, thresh=0.012):
    """Near-horizontal plane in the level frame. Returns (n, d, inlier_mask, z_median) with n·x = d, n_z > 0."""
    n_pts = pts.shape[0]
    if n_pts < 80:
        return None
    rng = np.random.RandomState(0)
    best_n, best_nvec, best_d = 0, None, None
    for _ in range(n_iter):
        i = rng.choice(n_pts, 3, replace=False)
        v = np.cross(pts[i[1]] - pts[i[0]], pts[i[2]] - pts[i[0]])
        ln = float(np.linalg.norm(v))
        if ln < 1e-8:
            continue
        nvec = v / ln
        if abs(nvec[2]) < 0.92:
            continue
        if nvec[2] < 0:
            nvec = -nvec
        d = float(nvec @ pts[i[0]])
        n_in = int(np.sum(np.abs(pts @ nvec - d) < thresh))
        if n_in > best_n:
            best_n, best_nvec, best_d = n_in, nvec, d
    if best_nvec is None or best_n < 80:
        return None
    mask = np.abs(pts @ best_nvec - best_d) < thresh
    z = float(np.median(pts[mask, 2]))
    return best_nvec, best_d, mask, z


def _cluster_xy(pts, uv, pitch=0.015, min_n=35):
    if pts.shape[0] < min_n:
        return []
    origin = pts[:, :2].min(0)
    ij = np.floor((pts[:, :2] - origin) / pitch).astype(int)
    cells = {}
    for k, (i, j) in enumerate(ij):
        cells.setdefault((int(i), int(j)), []).append(k)
    seen, clusters = set(), []
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    for seed in cells:
        if seed in seen:
            continue
        stack, idxs = [seed], []
        seen.add(seed)
        while stack:
            c = stack.pop()
            idxs.extend(cells[c])
            ci, cj = c
            for di, dj in neigh:
                n = (ci + di, cj + dj)
                if n in cells and n not in seen:
                    seen.add(n)
                    stack.append(n)
        if len(idxs) >= min_n:
            idx = np.array(idxs, dtype=int)
            clusters.append((pts[idx], uv[idx]))
    return clusters


def _project_level(p_L, t_cam, R_cam, R_pL, K):
    p_pelvis = R_pL @ np.asarray(p_L, dtype=float)
    p_link = R_cam.T @ (p_pelvis - t_cam)
    p_opt = R_D435_LINK_FROM_OPT.T @ p_link
    if p_opt[2] <= 0.05:
        return None
    u = K["fx"] * p_opt[0] / p_opt[2] + K["cx"]
    v = K["fy"] * p_opt[1] / p_opt[2] + K["cy"]
    return int(round(u)), int(round(v))


def _ray_level(u, v, K, t_cam, R_cam, R_pL):
    p_opt = np.array([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], 1.0])
    d_L = R_pL.T @ (R_cam @ (R_D435_LINK_FROM_OPT @ p_opt))
    o_L = R_pL.T @ t_cam
    return o_L, d_L


def _hit_plane(o, d, nvec, d_plane):
    den = float(nvec @ d)
    if abs(den) < 1e-8:
        return None
    s = (d_plane - float(nvec @ o)) / den
    if s < 0.05:
        return None
    return o + s * d


def _fill_bw_holes(bw):
    """Fill enclosed 0-regions (shiny cans punch holes in the table UV mask). Border-connected zeros stay empty (counter front)."""
    import cv2
    h, w = bw.shape[:2]
    flood = bw.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(0, w, 6):
        if flood[0, x] == 0:
            cv2.floodFill(flood, mask, (x, 0), 64)
        if flood[h - 1, x] == 0:
            cv2.floodFill(flood, mask, (x, h - 1), 64)
    for y in range(0, h, 6):
        if flood[y, 0] == 0:
            cv2.floodFill(flood, mask, (0, y), 64)
        if flood[y, w - 1] == 0:
            cv2.floodFill(flood, mask, (w - 1, y), 64)
    filled = bw.copy()
    filled[(bw == 0) & (flood == 0)] = 255
    return filled


def _blobs_from_mask(obj, blue, tag, h, w, K, t_cam, R_cam, R_pL, nvec, d_plane, lo, hi):
    """Score can-like contours on one binary mask. Logs area rejects so a merged blob is visible."""
    import cv2
    scored, dbg = [], []
    n_pass_geom = 0
    cnts, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        rec = dict(area=int(area), bbox=(int(x), int(y), int(bw), int(bh)),
                   aspect=round(bw / float(bh), 2) if bh else 0, src=tag, why=None)
        if area < 300 or bh < 16 or bw < 8:
            rec["why"] = "tiny"
            if area >= 200:
                dbg.append(rec)
            continue
        if area > 50000:
            rec["why"] = "area"
            dbg.append(rec)
            continue
        aspect = rec["aspect"]
        u_c, v_c = x + 0.5 * bw, y + 0.38 * bh
        o, d = _ray_level(u_c, v_c, K, t_cam, R_cam, R_pL)
        hit = _hit_plane(o, d, nvec, d_plane + 0.5 * CAN_H * float(nvec[2]))
        why = None
        if (y + 0.5 * bh) > 0.62 * h:
            why = "low-in-image"
        elif not (0.25 <= aspect <= 1.25):
            why = "aspect"
        elif hit is None:
            why = "no-plane"
        elif not (lo[0] - 0.05 <= hit[0] <= hi[0] + 0.05 and lo[1] - 0.05 <= hit[1] <= hi[1] + 0.05):
            why = "off-table"
            rec["xy"] = hit[:2].round(3).tolist()
        else:
            p_pelvis = R_pL @ hit
            p_opt = R_D435_LINK_FROM_OPT.T @ (R_cam.T @ (p_pelvis - t_cam))
            z_cam = float(p_opt[2])
            diam = bw * z_cam / K["fx"]
            height = bh * z_cam / K["fy"]
            clipped = x <= 2 or y <= 2 or x + bw >= w - 3 or y + bh >= h - 3
            sil = []
            roi = obj[y:y + bh, x:x + bw] > 0
            rows = roi[: max(3, int(0.55 * bh))]
            for row in rows:
                idx = np.flatnonzero(row)
                if idx.size >= 3:
                    sil.append(int(idx[-1] - idx[0] + 1))
            if sil:
                diam = float(np.median(sil)) * z_cam / K["fx"]
            rec.update(xy=hit[:2].round(3).tolist(), diam_mm=round(diam * 1000), height_mm=round(height * 1000),
                       z_cam=round(z_cam, 2), clipped=clipped)
            if z_cam < 0.2:
                why = "zcam"
            elif not (0.035 <= diam <= 0.140 and 0.065 <= height <= 0.300):
                why = "size"
            else:
                n_pass_geom += 1
                roi_blue = blue[y:y + bh, x:x + bw]
                bf = float(np.mean(roi_blue > 0))
                kind = "pepsi-blue-12oz" if bf >= 0.22 or tag == "blue" else "pepsi-silver-12oz"
                if not (0.040 <= diam <= 0.100):
                    kind = "can-like"
                size_s = math.exp(-((diam - 0.066) / 0.028) ** 2)
                ident_s = 1.25 if "blue" in kind else (1.0 if "silver" in kind else 0.6)
                front_s = 1.0 if 0.12 <= hit[0] <= 0.75 else 0.35
                clip_s = 0.7 if clipped else 1.0
                score = size_s * ident_s * front_s * clip_s
                rec.update(ident=kind, score=round(score, 3), blue_frac=round(bf, 2))
                scored.append(dict(score=score, xy=hit[:2].copy(), radius=diam / 2.0,
                                   height=CAN_H if "12oz" in kind else height,
                                   blue_frac=bf, n=int(area), ident=kind,
                                   bbox=(int(x), int(y), int(bw), int(bh)), kind=kind, clipped=clipped, src=tag))
        rec["why"] = why
        dbg.append(rec)
    return scored, dbg, len(cnts), n_pass_geom


def _cans_from_color(color, K, t_cam, R_cam, R_pL, nvec, d_plane, tab_xy, prefer_y=None, table_uv=None):
    """Shiny cans often have no depth. Find Pepsi-blue / silver blobs and drop them onto the table plane."""
    import cv2
    h, w = color.shape[:2]
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(color)
    blue = ((b > 70) & (b > g + 12) & (b > r + 20)).astype(np.uint8) * 255
    blue = cv2.bitwise_or(blue, cv2.inRange(hsv, (80, 35, 35), (140, 255, 255)))
    kernel = np.ones((7, 7), np.uint8)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    diff_obj = None
    table_img = None
    if table_uv is not None and len(table_uv) > 30:
        uu = np.clip(table_uv[::3, 0], 0, w - 1)
        vv = np.clip(table_uv[::3, 1], 0, h - 1)
        med = np.median(color[vv, uu].astype(np.float32), axis=0)
        diff = np.linalg.norm(color.astype(np.float32) - med, axis=2)
        not_table = (diff > 32).astype(np.uint8) * 255
        table_img = np.zeros((h, w), np.uint8)
        table_img[np.clip(table_uv[:, 1], 0, h - 1), np.clip(table_uv[:, 0], 0, w - 1)] = 255
        table_img = cv2.dilate(table_img, np.ones((15, 15), np.uint8))
        table_img = cv2.morphologyEx(table_img, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        table_img = _fill_bw_holes(table_img)
        # open hard so a person-shadow / glare bridge cannot swallow the can (seen live: 1 contour, area>40k, 0 blobs logged)
        not_table = cv2.bitwise_and(not_table, table_img)
        not_table = cv2.morphologyEx(not_table, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        not_table = cv2.morphologyEx(not_table, cv2.MORPH_CLOSE, kernel)
        not_table = cv2.bitwise_and(not_table, cv2.bitwise_not(cv2.dilate(blue, np.ones((9, 9), np.uint8))))
        diff_obj = not_table
        try:
            cv2.imwrite("/tmp/g1-look-mask.jpg", np.hstack([
                cv2.cvtColor(blue, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(not_table, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(cv2.bitwise_or(blue, not_table), cv2.COLOR_GRAY2BGR)]))
        except Exception:
            pass
    lo, hi = np.percentile(tab_xy, [4, 96], axis=0)
    scored, dbg, n_cnt, n_pass_geom = [], [], 0, 0
    for tag, mask in (("blue", blue), ("diff", diff_obj)):
        if mask is None:
            continue
        s, d, n, g = _blobs_from_mask(mask, blue, tag, h, w, K, t_cam, R_cam, R_pL, nvec, d_plane, lo, hi)
        scored.extend(s)
        dbg.extend(d)
        n_cnt += n
        n_pass_geom += g
    dbg.sort(key=lambda r: -r["area"])
    for rec in dbg[:10]:
        extra = " ".join("%s=%s" % (k, rec[k]) for k in ("xy", "diam_mm", "height_mm", "ident", "score") if k in rec)
        log("  blob %s area %s aspect %s why=%s %s" % (
            rec.get("src", "?"), rec.get("area"), rec.get("aspect"), rec.get("why"), extra))
    # NMS: same can from blue + diff
    scored.sort(key=lambda c: -c["score"])
    uniq = []
    for c in scored:
        if any(float(np.linalg.norm(c["xy"] - k["xy"])) < 0.08 for k in uniq):
            continue
        uniq.append(c)
    log("LOOK: object contours %d, geom-ok %d, kept %d" % (n_cnt, n_pass_geom, len(uniq)))
    return uniq


def _pregrasp_ik_mm(xy, z, q, R_pL):
    """IK residual (mm) for a side-grasp pregrasp at this can. Does not mutate CAN permanently."""
    saved = dict(CAN)
    try:
        CAN["x"], CAN["y"], CAN["z"] = float(xy[0]), float(xy[1]), float(z)
        goal_L = palm_target(args.gap + args.pregrasp_gap)
        R_des = R_pL @ palm_rotation_des()
        _, ep, _er = ik_best(ik_seeds_from(q), R_pL @ goal_L, R_des)
        return ep * 1000.0
    finally:
        CAN.update(saved)


def _pick_can(scored, z_table, st, R_pL):
    """Pick a 12 oz can in view. Rank by whether this arm can IK a pregrasp, then by blob score.
    No hardcoded x/y — the camera says where the can is; dryrun still refuses an unreachable pose."""
    pool = [c for c in scored if c.get("ident") != "can-like" and c.get("score", 0) >= 0.08]
    if not pool:
        pool = [c for c in scored if c.get("score", 0) >= 0.15]
    if not pool:
        return None
    q = st["q"].copy()
    ranked = []
    for c in pool:
        mm = _pregrasp_ik_mm(c["xy"], z_table, q, R_pL)
        log("  reach %s xy=(%.3f, %.3f)  pregrasp IK %.0f mm" % (c["ident"], c["xy"][0], c["xy"][1], mm))
        ranked.append((mm, -float(c["score"]), c))
    ranked.sort()
    return ranked[0][2]


def do_look(info):
    """Grab D435i RGB-D, fit a table plane, find a 12 oz can standing on it. No motion."""
    st = state()
    if st is None:
        log("LOOK: no rt/lowstate")
        return dict(ok=False, why="no lowstate")
    try:
        t_cam, R_cam = KIN.fk_frame(st["q"], "d435_link")
    except Exception as e:  # noqa: BLE001
        log("LOOK: d435_link missing from URDF (%r)" % (e,))
        return dict(ok=False, why="no d435_link")
    R_pL = rpy_to_mat(float(st["rpy"][0]), float(st["rpy"][1]), 0.0).T
    view_link = R_cam[:, 0]
    view_L = R_pL.T @ view_link
    origin_L = R_pL.T @ t_cam
    log("LOOK: D435i origin (level) %s m, +x_link (view) %s" % (np.round(origin_L, 3).tolist(), np.round(view_L, 2).tolist()))
    try:
        log("LOOK: capturing aligned RGB-D (640x480, ~0.5 s)...")
        color, depth, K = capture_rgbd()
    except Exception as e:  # noqa: BLE001
        log("LOOK: RealSense capture failed: %r" % (e,))
        return dict(ok=False, why="realsense: %r" % (e,))
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(0, w, 2), np.arange(0, h, 2))
    z = depth[vs, us]
    valid = np.isfinite(z) & (z > 0.25) & (z < max(2.2, args.look_xmax + 0.6))
    us, vs, z = us[valid], vs[valid], z[valid]
    x_opt = (us - K["cx"]) * z / K["fx"]
    y_opt = (vs - K["cy"]) * z / K["fy"]
    p_opt = np.stack([x_opt, y_opt, z], axis=1)
    p_link = (R_D435_LINK_FROM_OPT @ p_opt.T).T
    p_pelvis = (R_cam @ p_link.T).T + t_cam
    p_L = (R_pL.T @ p_pelvis.T).T
    uv = np.stack([us, vs], axis=1).astype(int)
    roi = (p_L[:, 0] > 0.08) & (p_L[:, 0] < args.look_xmax) & (np.abs(p_L[:, 1]) < args.look_ymax) & (p_L[:, 2] > -0.50) & (p_L[:, 2] < 0.25)
    p_roi, uv_roi = p_L[roi], uv[roi]
    log("LOOK: %d depth pts, %d in front-of-robot ROI" % (p_L.shape[0], p_roi.shape[0]))
    fitted = _ransac_table(p_roi)
    if fitted is None:
        log("LOOK: no horizontal table plane in front of the robot")
        return dict(ok=False, why="no table", n_roi=int(p_roi.shape[0]), camera_origin_level=origin_L.round(3).tolist())
    nvec, d_plane, table_mask, z_table = fitted
    tilt_deg = math.degrees(math.acos(min(1.0, abs(nvec[2]))))
    ph = args.pelvis_height if args.pelvis_height is not None else info.get("pelvis_height_est")
    table_floor = None if ph is None else ph + z_table
    below_mm = z_table * 1000
    tab_xy = p_roi[table_mask, :2]
    table_x_min = None
    edge_yaw_deg = None
    edge_bins = 0
    table_y_pct = None
    if len(tab_xy):
        front = tab_xy[tab_xy[:, 0] > 0.02]
        if len(front):
            table_x_min = float(np.percentile(front[:, 0], 8))
        # lateral extent of the visible table top (10/50/90 % of y): the search turns toward where the table continues
        table_y_pct = [round(float(v), 3) for v in np.percentile(tab_xy[:, 1], (10, 50, 90))]
        # front-edge orientation: nearest table x per 4 cm y-bin, line x = a + b*y (robust refit), yaw = atan(b).
        # The robot squares up to the table by turning -yaw (CCW positive): a CCW-rotated robot sees the edge farther
        # away on its left (b > 0). Used by g1_fetch.py before strafing (a 20 deg drift lost the can from view tonight).
        ys_e, xs_e = [], []
        for yb in np.arange(-0.45, 0.45, 0.04):
            sel = (front[:, 1] >= yb) & (front[:, 1] < yb + 0.04)
            if int(np.sum(sel)) >= 25:
                ys_e.append(yb + 0.02)
                xs_e.append(float(np.percentile(front[sel, 0], 5)))
        if len(ys_e) >= 5:
            A = np.stack([np.ones(len(ys_e)), np.array(ys_e)], axis=1)
            xs_a = np.array(xs_e)
            coef = np.linalg.lstsq(A, xs_a, rcond=None)[0]
            keep = np.abs(xs_a - A @ coef) < 0.03            # drop bins behind the can / at a corner
            if int(keep.sum()) >= 5:
                coef = np.linalg.lstsq(A[keep], xs_a[keep], rcond=None)[0]
                edge_yaw_deg = math.degrees(math.atan(float(coef[1])))
                edge_bins = int(keep.sum())
    log("LOOK: table z_level=%+.3f m (%.0f mm %s pelvis), tilt %.1f deg, %d inliers%s%s%s" % (
        z_table, abs(below_mm), "above" if z_table > 0 else "below", tilt_deg, int(np.sum(table_mask)),
        "" if table_floor is None else ", floor->top %.3f m" % table_floor,
        "" if table_x_min is None else ", front x=%.3f m" % table_x_min,
        "" if edge_yaw_deg is None else ", edge yaw %+.1f deg (%d bins; robot squares up by turning %+.1f deg)" % (edge_yaw_deg, edge_bins, -edge_yaw_deg)))
    scored = []
    try:
        scored = _cans_from_color(color, K, t_cam, R_cam, R_pL, nvec, d_plane, tab_xy, 0.0,
                                  table_uv=uv_roi[table_mask])
        log("LOOK: color found %d can candidate(s)" % len(scored))
        for c in scored[:4]:
            log("  cand %s score %.2f  xy (%.3f, %.3f)  diam %.0f mm  h %.0f mm" % (
                c["ident"], c["score"], c["xy"][0], c["xy"][1], 2000 * c["radius"], 1000 * c["height"]))
    except Exception as e:  # noqa: BLE001
        log("LOOK: color can finder failed (%r)" % (e,))
    can = _pick_can(scored, z_table, st, R_pL)
    out = dict(ok=False, table_z_level=round(z_table, 4), table_tilt_deg=round(tilt_deg, 2),
               table_floor_m=None if table_floor is None else round(table_floor, 4),
               table_x_min=None if table_x_min is None else round(table_x_min, 3),
               table_edge_yaw_deg=None if edge_yaw_deg is None else round(edge_yaw_deg, 1), table_edge_bins=edge_bins,
               table_y_pct=table_y_pct,
               table_inliers=int(np.sum(table_mask)), n_clusters=len(scored),
               camera_origin_level=origin_L.round(3).tolist(), view_level=view_L.round(3).tolist(),
               pelvis_height=ph, candidates=[{k: (v.round(3).tolist() if hasattr(v, "round") else v)
                                               for k, v in c.items() if k not in ("uv", "bbox")} for c in scored[:5]])
    if can is None:
        log("LOOK: table found, no 12 oz can in the D435i view")
        out["why"] = "no can"
        # search cue for g1_fetch: a blue, can-like blob that was rejected (clipped at the image edge, too small,
        # low score) still says which way to strafe. The can at the right edge of the frame read 38 mm / score 0.149.
        hints = [c for c in scored if c.get("blue_frac", 0) >= 0.4 or c.get("clipped")]
        if hints:
            hbest = max(hints, key=lambda c: c.get("score", 0))
            out["hint_xy"] = [round(float(hbest["xy"][0]), 3), round(float(hbest["xy"][1]), 3)]
            out["hint_why"] = "%s %.0f mm, score %.2f%s" % (hbest.get("ident"), 2000 * hbest["radius"], hbest.get("score", 0), ", clipped" if hbest.get("clipped") else "")
            log("LOOK hint: blue blob at x=%.3f y=%+.3f (%s) - not a confirmed can" % (out["hint_xy"][0], out["hint_xy"][1], out["hint_why"]))
    else:
        can_x, can_y = float(can["xy"][0]), float(can["xy"][1])
        can_z = z_table  # can BASE = table top
        out.update(ok=True, can_x=round(can_x, 4), can_y=round(can_y, 4), can_z=round(can_z, 4),
                   diameter_mm=round(2 * can["radius"] * 1000, 1), height_mm=round(can["height"] * 1000, 1),
                   blue_frac=round(can["blue_frac"], 2), ident=can["ident"], n_pts=can["n"], score=round(can["score"], 3))
        log("LOOK: %s at (level) x=%.3f y=%.3f z=%.3f  diameter %.0f mm  height %.0f mm  blue %.0f%%  score %.2f" % (
            can["ident"], can_x, can_y, can_z, out["diameter_mm"], out["height_mm"], 100 * can["blue_frac"], can["score"]))
        log("LOOK flags: --can-x %.3f --can-y %.3f --can-z %.3f%s" % (
            can_x, can_y, can_z, "" if table_floor is None else "   (or --table-height %.3f)" % table_floor))
    try:
        import cv2
        vis = color.copy()
        if table_mask.any():
            uu, vv = uv_roi[table_mask][::4, 0], uv_roi[table_mask][::4, 1]
            vis[np.clip(vv, 0, h - 1), np.clip(uu, 0, w - 1)] = (40, 180, 40)
        for c in scored[:6]:
            if not c.get("bbox"):
                continue
            x, y, bw, bh = c["bbox"]
            picked = can is not None and c is can
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 220, 255) if picked else (180, 180, 0), 2 if picked else 1)
        if can is not None:
            pix = _project_level([out["can_x"], out["can_y"], z_table + 0.5 * can["height"]], t_cam, R_cam, R_pL, K)
            if pix is not None:
                cv2.circle(vis, pix, max(8, int(can["radius"] * 900)), (0, 220, 255), 2)
                cv2.putText(vis, "%s %.0fmm" % (can["ident"], out["height_mm"]), (pix[0] + 8, pix[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "table z_L=%+.3f" % z_table, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 180, 40), 1, cv2.LINE_AA)
        jpg = (args.look_json[:-5] if args.look_json.endswith(".json") else args.look_json) + ".jpg"
        cv2.imwrite(jpg, vis)
        out["jpeg"] = jpg
        log("LOOK overlay: %s" % jpg)
    except Exception as e:  # noqa: BLE001
        log("LOOK: overlay skipped (%r)" % (e,))
    p_palm, _ = KIN.fk(st["q"])
    p_palm_L = R_pL.T @ p_palm
    out["palm_level"] = p_palm_L.round(3).tolist()
    if can is not None:
        out["palm_to_can_m"] = round(float(np.linalg.norm(p_palm_L - np.array([out["can_x"], out["can_y"], z_table + 0.055]))), 3)
    try:
        with open(args.look_json, "w") as f:
            json.dump(out, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
        log("LOOK json: %s" % args.look_json)
    except Exception as e:  # noqa: BLE001
        log("LOOK: could not write %s (%r)" % (args.look_json, e))
    return out


def apply_look(look):
    """Copy a successful look into CAN (overwrites --can-x/y/z)."""
    if not look or not look.get("ok"):
        return False
    CAN["x"] = float(look["can_x"])
    CAN["y"] = float(look["can_y"])
    CAN["z"] = float(look["can_z"])
    if args.table_x is not None:
        CAN["table_x"] = float(args.table_x)
    elif look.get("table_x_min") is not None:
        CAN["table_x"] = float(look["table_x_min"])
    args.can_x, args.can_y, args.can_z = CAN["x"], CAN["y"], CAN["z"]
    log("LOOK applied -> can (level) x=%.3f y=%.3f z=%.3f%s" % (
        CAN["x"], CAN["y"], CAN["z"],
        "" if CAN.get("table_x") is None else "  table front x=%.3f" % CAN["table_x"]))
    return True


def rpc_queries():
    out = {}
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        msc = MotionSwitcherClient()
        msc.SetTimeout(2.0)
        msc.Init()
        code, data = msc.CheckMode()
        out["motion_switcher.CheckMode"] = {"code": code, "data": data}
    except Exception as e:  # noqa: BLE001
        out["motion_switcher.CheckMode"] = {"error": repr(e)}
    try:
        rsc = RobotStateClient()
        rsc.SetTimeout(2.0)
        rsc.Init()
        code, data = rsc.ServiceList()
        try:
            data = json.loads(data) if isinstance(data, str) else data
        except Exception:  # noqa: BLE001
            pass
        if isinstance(data, dict) and "list" in data:
            data = {s.get("name"): s.get("status") for s in data["list"]}
        elif isinstance(data, list):
            data = {s.get("name"): s.get("status") for s in data if isinstance(s, dict)}
        out["robot_state.ServiceList"] = {"code": code, "data": data}
    except Exception as e:  # noqa: BLE001
        out["robot_state.ServiceList"] = {"error": repr(e)}
    return out


try:
    from unitree_sdk2py.rpc.client import Client as _RpcClient

    class RobotStateClient(_RpcClient):
        def __init__(self):
            super().__init__("robot_state", False)

        def Init(self):
            self._SetApiVerson("1.0.0.1")
            self._RegistApi(1001, 0)  # ServiceSwitch
            self._RegistApi(1002, 0)
            self._RegistApi(1003, 0)  # ServiceList

        def ServiceList(self):
            return self._Call(1003, "{}")

        def ServiceSwitch(self, name, on):
            return self._Call(1001, json.dumps({"name": name, "switch": 1 if on else 0}))

    class LocoQuery(_RpcClient):
        """read-only side of Unitree's loco ("sport") service: the FSM the remote drives. Never calls a Set api."""

        def __init__(self):
            super().__init__("sport", False)

        def Init(self):
            self._SetApiVerson("1.0.0.0")
            for api in (7001, 7002, 7003, 7004, 7005):
                self._RegistApi(api, 0)

        def _get(self, api):
            code, data = self._Call(api, "{}")
            parsed = data
            if isinstance(data, str):
                try:
                    parsed = json.loads(data)
                except Exception:  # noqa: BLE001
                    parsed = data
            if isinstance(parsed, dict) and "data" in parsed:
                parsed = parsed["data"]
            if code != 0:
                if api in (7001, 7002):
                    t = now()
                    if t - _LOCO.get("err_t", -10) > 8.0:
                        log("loco Get api %d code %s (3104 = timeout under DDS load)" % (api, code))
                        _LOCO["err_t"] = t
                return None
            return parsed

        def fsm(self):
            fid = self._get(7001)
            out = dict(fsm_id=fid, fsm_mode=None, balance_mode=None, stand_height=None)
            if fid is None:
                return out          # the service is not answering (AI standing 802): do not pay a second timeout
            out["fsm_mode"] = self._get(7002)
            # balance / stand height are rejected (code 7301) in ZeroTorque and time out in the AI FSMs (802 answers
            # 7001 only): ask for them only when the mode query answered
            if fid != 0 and out["fsm_mode"] is not None:
                out["balance_mode"] = self._get(7003)
                out["stand_height"] = self._get(7005)
            return out
except Exception:  # noqa: BLE001
    RobotStateClient = None
    LocoQuery = None

FSM_NAMES = {0: "ZeroTorque", 1: "Damp", 2: "Squat", 3: "Sit", 4: "StandUp (locked standing, L2+UP)", 200: "Start (main operation control, R1+X)",
             702: "Lie2StandUp", 706: "Squat2StandUp"}
BALANCE_NAMES = {0: "regular (arm_sdk + Move allowed)", 1: "continuous gait / running (arm_sdk blocks Move)"}


def fsm_text(f):
    if not f:
        return "loco FSM: no answer"
    fid = f.get("fsm_id")
    try:
        fid_i = int(fid)
    except (TypeError, ValueError):
        fid_i = None
    bal = f.get("balance_mode")
    try:
        bal_i = int(bal) if bal is not None else None
    except (TypeError, ValueError):
        bal_i = None
    return "loco FSM id %s (%s) mode %s | balance %s (%s) | stand height %s" % (
        fid, FSM_NAMES.get(fid_i, "?"), f.get("fsm_mode"),
        bal if bal is not None else "n/a", BALANCE_NAMES.get(bal_i, "n/a in this mode"),
        f.get("stand_height") if f.get("stand_height") is not None else "n/a")


_LOCO = {"c": None, "dead": False}


def loco_fsm(retry=False):
    """One query per call, 1 s timeout. In the AI standing FSM (802) the loco service does not answer at all
    (code 3104): the old create-retry-query pattern cost ~7.5 s at check and again at takeover. After the first
    no-answer of a session later calls return None at once; --stage fsm polls with retry=True."""
    if LocoQuery is None:
        return None
    if _LOCO["dead"] and not retry:
        return None
    try:
        if _LOCO["c"] is None:
            c = LocoQuery()
            c.SetTimeout(1.0)
            c.Init()
            _LOCO["c"] = c
        f = _LOCO["c"].fsm()
        _LOCO["dead"] = f.get("fsm_id") is None
        return f
    except Exception as e:  # noqa: BLE001
        log("loco query failed: %r" % (e,))
        _LOCO["c"] = None
        return None


def gains_text():
    with LOCK:
        kp = None if LOWCMD["kp"] is None else LOWCMD["kp"].copy()
        kd = None if LOWCMD["kd"] is None else LOWCMD["kd"].copy()
    if kp is None:
        return "rt/lowcmd: none"
    grp = lambda a, b: "%.0f-%.0f" % (kp[a:b].min(), kp[a:b].max())  # noqa: E731
    grpd = lambda a, b: "%.1f-%.1f" % (kd[a:b].min(), kd[a:b].max())  # noqa: E731
    return "controller kp legs %s waist %s arms %s | kd legs %s waist %s arms %s" % (grp(0, 12), grp(12, 15), grp(15, 29), grpd(0, 12), grpd(12, 15), grpd(15, 29))


def do_fsm_watch():
    """read-only: print the loco FSM, the controller gains and the arm pose every time something changes, until Ctrl-C.
    Lets the operator confirm each remote step (L2+B, L2+UP, R1+X) does what we expect before any arm_sdk message."""
    log("FSM WATCH (read-only, no arm_sdk). Work the remote; every change is printed. Ctrl-C to stop.")
    log("  expected: ZeroTorque (id 0, kp=0) -> L2+B Damp (id 1) -> L2+UP StandUp (id 4, legs have kp) -> R1+X Start (id 200, arm_sdk can blend)")
    last = None
    t_last_line = 0.0
    t_end = now() + args.watch_seconds if args.watch_seconds > 0 else None
    while t_end is None or now() < t_end:
        f = loco_fsm(retry=True)
        st = state()
        with LOCK:
            kp = None if LOWCMD["kp"] is None else LOWCMD["kp"].copy()
            kd = None if LOWCMD["kd"] is None else LOWCMD["kd"].copy()
            n_lowcmd = LOWCMD["n"]
        ctrl_txt, _ = controller_state()
        key = (None if f is None else (f.get("fsm_id"), f.get("fsm_mode"), f.get("balance_mode")),
               None if kp is None else tuple(np.round(kp[:29], 0).tolist()))
        if key != last or now() - t_last_line > 10.0:
            last = key
            t_last_line = now()
            log(fsm_text(f))
            log("  %s" % gains_text())
            if st is not None:
                p, R = KIN.fk(st["q"])
                log("  IMU rpy %s deg | %s arm q %s | waist %s | palm (pelvis) %s" % (
                    np.degrees(st["rpy"]).round(1).tolist(), SIDE, st["q"][ARM].round(3).tolist(), st["q"][WAIST_IDX].round(3).tolist(), np.round(p, 3).tolist()))
                if kp is not None and kp[:29].max() > 0:
                    log("  controller kp per joint: legs %s | waist %s | L arm %s | R arm %s" % (
                        np.round(kp[0:12], 0).tolist(), np.round(kp[12:15], 0).tolist(), np.round(kp[15:22], 0).tolist(), np.round(kp[22:29], 0).tolist()))
                    if kd is not None:
                        log("  controller kd per joint: legs %s | waist %s | L arm %s | R arm %s" % (
                            np.round(kd[0:12], 1).tolist(), np.round(kd[12:15], 1).tolist(), np.round(kd[15:22], 1).tolist(), np.round(kd[22:29], 1).tolist()))
            SUMMARY.setdefault("fsm_watch", []).append(dict(t=round(now(), 2), fsm=f, controller=ctrl_txt, lowcmd_msgs=n_lowcmd,
                                                            kp=None if kp is None else np.round(kp[:29], 1).tolist(),
                                                            kd=None if kd is None else np.round(kd[:29], 2).tolist(),
                                                            arm_q=None if st is None else st["q"][ARM].round(4).tolist(),
                                                            all_q=None if st is None else st["q"].round(4).tolist()))
        time.sleep(1.0)
    log("FSM WATCH ended after %.0f s (%d snapshots)" % (args.watch_seconds, len(SUMMARY.get("fsm_watch", []))))


# ----------------------------------------------------------------------------------------------------------------------
# dry run: kinematic rehearsal of the whole sequence from the measured pose
# ----------------------------------------------------------------------------------------------------------------------
def resolve_can_z(info):
    if CAN["z"] is not None:
        return
    if args.table_height is None:
        log("can height unknown: give --table-height (floor -> table top) or --can-z (can base relative to the pelvis origin)")
        save()
        os._exit(1)
    ph = args.pelvis_height if args.pelvis_height is not None else info["pelvis_height_est"]
    CAN["z"] = args.table_height - ph
    log("can base z = table %.3f - pelvis %.3f = %+.3f m (level frame)%s" % (args.table_height, ph, CAN["z"],
        "" if args.pelvis_height is not None else "  [pelvis height from leg FK - feet must be flat on the floor]"))


# The dryrun is THE plan. Every free-space waypoint's joints are stored here and the live stages replay them
# (start_palm_motion(q_plan=...)); nothing re-solves IK on the robot. park() retraces the executed steps backwards.
PLAN = {"mode": None, "order": [], "steps": {}, "can": None, "done": []}
STAGE_OF_STEP = {"raise": "raise", "fold": "raise", "pregrasp": "pregrasp", "approach": "approach", "descend": "descend",
                 "lift": "lift", "lower": "lower", "retreat": "retreat"}


def plan_stage_of(step):
    return "pregrasp" if step.startswith("via") else STAGE_OF_STEP.get(step, step)


def plan_steps_for(stage):
    return [s for s in PLAN["order"] if plan_stage_of(s) == stage]


def plan_is_current():
    """False once the operator moved the can estimate (n dx dy dz) - the stored joints no longer match the goals."""
    c = PLAN.get("can")
    return c is not None and all(CAN.get(k) is not None and abs(float(CAN[k]) - float(c[k])) < 0.0015 for k in ("x", "y", "z"))


def do_dryrun(info):
    st = state()
    R_pL = rpy_to_mat(float(st["rpy"][0]), float(st["rpy"][1]), 0.0).T
    R_des = R_pL @ palm_rotation_des()
    q0 = st["q"].copy()
    p0, _ = KIN.fk(q0)
    p0_L = R_pL.T @ p0
    goals = stage_goals()
    stop = args.until or "retreat"
    seq_req = ["raise", "pregrasp", "approach", "descend", "lift", "lower", "retreat"]
    if stop == "grasp":
        stop = "descend"
    if stop not in seq_req:
        stop = "retreat"
    required = set(seq_req[: seq_req.index(stop) + 1])
    ok_now, why_now = body_clear(q0, R_pL)
    if not ok_now:
        log("DRYRUN: the measured pose already violates the body model (%s) - boxes too tight? not planning from here" % why_now)
    if args.offline:
        sr, el, wr, pp, R = KIN.fk_arm(q0)
        log("OFFLINE arm points now: %s" % "; ".join("%s %s" % (n, np.round(p, 3).tolist()) for n, p in _arm_points_L(sr, el, wr, pp, R, R_pL)))
    q_pg, ep_pg, _ = ik_near(reach_arm_guesses(q0) + ik_seeds_from(q0), R_pL @ goals["pregrasp"], R_des, reach_arm_guesses(q0)[0])
    seq = []            # (step name, palm goal (level), planned q29 or None)
    mode = None
    vias_pre = None
    # 1) preferred: one checked joint move from the measured pose straight into a ready pose in the free space in
    #    front of the table edge (no overhead lift). Candidates are tried best-first and the first one whose whole via
    #    chain to pregrasp verifies is the plan.
    q_ready = p_ready = None
    if ep_pg < 0.02 and ok_now:
        cands = find_hip_reach_candidates(q0, R_pL, R_des, q_pg)
        for k, (score, q_c, ep_c, p_c) in enumerate(cands[:8]):
            geom = plan_table_vias(p_c, goals["pregrasp"])
            vias_c = plan_table_vias(p_c, goals["pregrasp"], q_seed=q_c, R_pL=R_pL, R_des=R_des, quiet=True)
            q_last = vias_c[-1][2] if vias_c else q_c
            q_p, ep_p, er_p = ik_near(ik_seeds_from(q_last) + reach_arm_guesses(q_last) + [q_pg], R_pL @ goals["pregrasp"], R_des,
                                      q_last if is_reaching_arm(q_last[ARM]) else q_pg)
            ok_p, why_p = path_check(q_last[ARM], q_p[ARM], q_last, R_pL)
            full = len(vias_c) == len(geom) and ep_p < 0.01 and er_p < 0.12 and ok_p
            log("READY cand %d: palm %s q %s | vias %d/%d, pregrasp %.1f mm%s -> %s" % (
                k + 1, np.round(p_c, 3).tolist(), np.round(q_c[ARM], 2).tolist(), len(vias_c), len(geom), ep_p * 1000,
                "" if ok_p else " (%s)" % why_p, "OK" if full else "no"))
            if full:
                q_ready, p_ready, vias_pre = q_c, p_c, vias_c
                break
    if q_ready is not None:
        mode = "direct"
        seq.append(("raise", p_ready, q_ready))
        q_via, via_from = q_ready, p_ready
        log("DRYRUN: RAISE = one move from the measured pose to the ready pose %s (arm swings out and forward, palm %.0f cm before the table edge x=%.2f)" % (
            np.round(p_ready, 3).tolist(), (table_edge_x() - p_ready[0]) * 100, table_edge_x()))
    else:
        rg = raise_goal(p0_L)
        q_raise, ep_r, er_r = ik_near(ik_seeds_from(q0), R_pL @ rg, R_des, q0)
        seq.append(("raise", rg, q_raise if ep_r < 0.02 else None))
        q_via, via_from = q_raise, rg
        mode = "lift"
        if ep_r < 0.03 and not is_reaching_arm(q_raise[ARM]) and ep_pg < 0.02:
            q_hip, _, p_hip = find_hip_reach_q(q_raise, R_pL, R_des, q_pg)
            if q_hip is not None:
                seq.append(("fold", p_hip, q_hip))
                q_via, via_from = q_hip, p_hip
                mode = "lift+fold"
                log("DRYRUN: no direct reaching move; RAISE = lift at the current xy, then FOLD to reaching at %s" % np.round(p_hip, 3).tolist())
            else:
                log("DRYRUN: lift stays in the hang basin; no reaching fold from that lifted pose")
    geom = plan_table_vias(via_from, goals["pregrasp"])
    vias = vias_pre if vias_pre is not None else plan_table_vias(via_from, goals["pregrasp"], q_seed=q_via, R_pL=R_pL, R_des=R_des)
    if geom and not vias:
        log("DRYRUN: no reachable table-clearing via (a single interpolant would cut through the table)")
    seq += vias
    seq += [("pregrasp", goals["pregrasp"], None), ("approach", goals["approach"], None), ("descend", goals["descend"], None),
            ("lift", goals["lift"], None), ("lower", goals["lower"], None), ("retreat", goals["retreat"], None)]
    log("DRYRUN %s arm, can base (level) x=%.3f y=%.3f z=%.3f, table edge x=%.2f, palm now %s  (required through %s)" % (
        SIDE, CAN["x"], CAN["y"], CAN["z"], table_edge_x(), np.round(p0_L, 3).tolist(), stop))
    report = []
    ok_all = ok_now and not (geom and not vias)
    q_prev = q0.copy()
    dur = {"raise": 3.0, "fold": 3.0, "pregrasp": 3.0, "approach": 2.5, "descend": 1.5, "lift": 2.0, "lower": 2.0, "retreat": 2.0}
    PLAN.update(mode=mode, order=[], steps={}, can={k: CAN.get(k) for k in ("x", "y", "z")}, done=[])
    for name, goal_L, q_hint in seq:
        goal = R_pL @ goal_L
        if q_hint is not None:
            q = np.asarray(q_hint, dtype=float).copy()
            p_chk, R_chk = KIN.fk(q)
            ep = float(np.linalg.norm(goal - p_chk))
            er = float(np.linalg.norm(pin.log3(R_des @ R_chk.T)))
        elif name == "raise":
            q, ep, er = ik_near(ik_seeds_from(q_prev), goal, R_des, q_prev)
        else:
            # short moves at the can: stay in the basin we arrived in (attract = previous waypoint); the first table
            # pose after a non-reaching waypoint is attracted to the clean forward-reach solution instead.
            # Null-space bias toward shoulder abduction: reaching across to a midline can otherwise drags the upper
            # arm along the chest (the body model flagged approach/descend by 1-13 mm).
            attract = q_prev if is_reaching_arm(q_prev[ARM]) else q_pg
            seeds = ik_seeds_from(q_prev) + reach_arm_guesses(q_prev) + [q_pg]
            far = can_on_far_side() or (CAN.get("y") is not None and abs(float(CAN["y"])) < 0.05)
            min_abd = 0.20 if far else 0.12
            q, ep, er = ik_near(seeds, goal, R_des, attract, min_abduction=min_abd)
            for roll_bias in (0.35, 0.55, 0.75, 0.95, 1.15):
                ok_b, _ = body_clear(q, R_pL)
                ok_pb, _ = path_check(q_prev[ARM], q[ARM], q_prev, R_pL, table=name not in ("approach", "descend", "lower"))
                if ok_b and ok_pb and shoulder_abduction(q[ARM]) >= min_abd:
                    break
                q_bias = np.asarray(attract, dtype=float)[ARM].copy()
                q_bias[1] = -SGN * roll_bias
                q_b, ep_b, er_b = ik_near(seeds + [q], goal, R_des, attract, q_bias=q_bias, k_ns=0.85, min_abduction=min_abd)
                if ep_b < 0.01 and er_b < 0.12 and ep_b <= ep + 0.003:
                    q, ep, er = q_b, ep_b, er_b
        dq = q[ARM] - q_prev[ARM]
        ok_path, why = path_check(q_prev[ARM], q[ARM], q_prev, R_pL, table=name not in ("approach", "descend", "lower"))
        if not ok_path:
            log("  %-9s interpolant refused: %s" % (name, why))
        margin_lo = q[ARM] - KIN.lo[ARM]
        margin_hi = KIN.hi[ARM] - q[ARM]
        tight = [JOINT_NAMES[ARM[k]].replace(SIDE + "_", "") + ("v" if margin_lo[k] < 0.08 else "^") for k in range(7) if min(margin_lo[k], margin_hi[k]) < 0.08]
        ok = ep < 0.01 and er < 0.12 and ok_path
        if name in required or name == "fold" or name.startswith("via"):
            ok_all &= ok
        T = dur.get(name, 2.5) * args.time_scale
        vpk = float(np.max(np.abs(dq))) / T * 1.57  # smoothstep peak velocity factor pi/2
        tm = path_table_margin(q_prev[ARM], q[ARM], q_prev, R_pL) if name not in ("approach", "descend", "lower") else None
        report.append(dict(stage=name, goal_level=np.round(goal_L, 3).tolist(), err_mm=round(ep * 1000, 1), rot_deg=round(math.degrees(er), 1),
                           q=np.round(q[ARM], 3).tolist(), dq_max=round(float(np.max(np.abs(dq))), 3), v_peak=round(vpk, 2), limits=tight, ok=ok,
                           path_why=why, table_margin_mm=None if tm is None else round(tm * 1000)))
        log("  %-9s goal %s -> err %5.1f mm / %4.1f deg | max joint move %.2f rad, peak %.2f rad/s%s | q %s%s%s" % (
            name, np.round(goal_L, 3).tolist(), ep * 1000, math.degrees(er), float(np.max(np.abs(dq))), vpk,
            " (>vmax %.2f -> slower)" % args.vmax if vpk > args.vmax else "", np.round(q[ARM], 2).tolist(),
            ("  near limits: " + ",".join(tight)) if tight else "",
            "" if tm is None else "  | table margin %.0f mm" % (tm * 1000)))
        PLAN["order"].append(name)
        PLAN["steps"][name] = dict(q=q.copy(), goal_L=np.asarray(goal_L, dtype=float).copy(), ok=ok, T=dur.get(name, 2.5),
                                   table_margin=tm)
        q_prev = q.copy()
    SUMMARY["dryrun"] = dict(can=dict(CAN), pelvis_height=args.pelvis_height or info["pelvis_height_est"], stages=report, reachable=bool(ok_all),
                             mode=mode, table_x=table_edge_x())
    if ok_all:
        log("DRYRUN OK (%s): every interpolant clears the table and the body; the live stages replay exactly these joints" % mode)
    else:
        bad = [r["stage"] for r in report if not r["ok"] and (r["stage"] in required or r["stage"] == "fold" or r["stage"].startswith("via"))]
        log("DRYRUN NOT OK (%s): %s failed (table z_L=%s)." % (
            mode, ", ".join(bad) if bad else "path", "?" if CAN.get("z") is None else "%.3f" % CAN["z"]))
    return ok_all


# ----------------------------------------------------------------------------------------------------------------------
# motion stages
# ----------------------------------------------------------------------------------------------------------------------
def finish_arm():
    """End the session: park if frozen, but never drop weight while the palm is in the table."""
    if ARMSDK.cmd is None or ARMSDK.weight <= 0.0 or args.keep:
        return
    st = state()
    p_L = None
    if st is not None:
        ARMSDK.set_level(st)
        p, _ = KIN.fk(st["q"])
        p_L = ARMSDK.R_pL.T @ p
    if p_L is not None and palm_in_table(p_L):
        log("exit: palm still in the table (x=%.2f z_L=%.3f) - keeping weight 1. L2+B if it is wedged." % (p_L[0], p_L[2]))
        return
    if ARMSDK.frozen:
        log("ending in a freeze (%s): PARK" % ARMSDK.frozen)
        park()
        return
    release_weight()


def begin_motion_session(info):
    """preconditions, then take over the arm with a weight ramp while commanding the measured pose."""
    global pub
    if not info["controller_ok"]:
        log("not arming: %s. Operator: remote on -> L2+B (damping) -> L2+UP (locked standing) [-> R1+X main control], then re-run." % info["controller"])
        save()
        os._exit(1)
    if info["foreign_arm_sdk"]:
        log("not arming: someone else publishes on rt/arm_sdk (g1_arm_example action? another test?) -> --stop-arm-example / stop it first")
        save()
        os._exit(1)
    if not args.no_hand and args.stage == "all" and info["hand_q"] is None:
        log("not arming: no rt/brainco/%s/state - `sudo systemctl restart brainco_hand.service` (boot race), or --no-hand" % SIDE)
        save()
        os._exit(1)
    if args.stop_arm_example:
        if RobotStateClient is None:
            log("--stop-arm-example: RPC client unavailable")
        else:
            rsc = RobotStateClient()
            rsc.SetTimeout(3.0)
            rsc.Init()
            code, data = rsc.ServiceSwitch("g1_arm_example", False)
            log("ServiceSwitch g1_arm_example off -> code %s %s (restore: ServiceSwitch on, or reboot)" % (code, data))
            time.sleep(0.5)
    st = state()
    ARMSDK.take_over(st)
    ARMSDK.set_level(st)
    pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
    pub.Init()
    if args.stage == "recover":
        # the previous session died with weight 1 and the controller kept applying its last arm_sdk command; a 0 -> 1
        # ramp here would first hand the arm to the controller's own target (through whatever is in the way).
        ARMSDK.weight = 1.0
        ARMSDK.start()
        time.sleep(0.1)
        log("TAKEOVER (recover): publishing the measured pose at weight 1 immediately (arm kp %.0f/kd %.1f) | %s" % (
            args.kp, args.kd, fsm_text(loco_fsm())))
        time.sleep(0.5)
    else:
        ARMSDK.start()
        time.sleep(0.1)
        log("TAKEOVER: commanding the measured pose (arm kp %.0f/kd %.1f, other arm %.0f/%.1f, waist %.0f/%.1f -> upright %s), weight 0 -> 1 over %.1fs | %s" %
            (args.kp, args.kd, args.other_kp, args.other_kd, args.waist_kp, args.waist_kd, args.waist_upright, args.weight_seconds, fsm_text(loco_fsm())))
        ARMSDK.start_weight(1.0, args.weight_seconds, "takeover")
        ARMSDK.wait()
    st2 = state()
    drift = st2["q"][ARM] - st["q"][ARM]
    log("TAKEOVER done: arm moved %s rad during the ramp (should be ~0)" % np.round(drift, 3).tolist())
    SUMMARY["takeover"] = dict(q_start=st["q"][ARM].round(4).tolist(), drift=drift.round(4).tolist())
    grab_still("takeover")


def release_weight():
    ARMSDK.motion = None
    ARMSDK.frozen = None
    ARMSDK.start_weight(0.0, args.weight_seconds, "release")
    while ARMSDK.motion is not None and not ARMSDK.motion.get("done"):
        time.sleep(0.02)
    time.sleep(0.3)
    log("RELEASED: weight 0, the motion controller owns the arms again")


def palm_in_table(p_L):
    return CAN.get("z") is not None and float(p_L[0]) > table_edge_x() and float(p_L[2]) < float(CAN["z"]) + 0.04


def park():
    """slow return, weight -> 0 at the end. With an executed plan: retrace its waypoints backwards (every interpolant
    was verified forwards), then the joint-space home. Without one: get off the table first, retract, then home.
    Never releases weight while the palm is still in the counter."""
    if ARMSDK.cmd is None:
        return
    if ARMSDK.frozen:
        ARMSDK.unfreeze("park")
    ARMSDK.allow_dip = True
    st = state()
    ARMSDK.set_level(st)
    ARMSDK.cmd[ARM] = st["q"][ARM]
    done = [s for s in PLAN.get("done", []) if s in PLAN["steps"]]
    # Only the transit waypoints are a way home: raise/fold and the table vias (clearance height, verified interpolants).
    # pregrasp/approach/descend/lift/lower are the excursion at the can - cycle 3's park retraced lower -> lift ->
    # descend -> approach first (30 s, palm back beside the released can) because the nearest executed waypoint was
    # retreat and everything before it in execution order was replayed.
    transit = [s for s in done if plan_stage_of(s) == "raise" or s.startswith("via")]
    if transit:
        p_now, _ = KIN.fk(state()["q"])
        p_now_L = ARMSDK.R_pL.T @ p_now
        dists = [float(np.linalg.norm(p_now_L - PLAN["steps"][n]["goal_L"])) for n in transit]
        k = int(np.argmin(dists))
        todo = transit[:k] if dists[k] < 0.06 else transit[:k + 1]   # at waypoint k -> go to k-1, ...; between -> nearest first
        log("PARK: palm %s is %.0f mm from %s; retracing %s -> home" % (
            np.round(p_now_L, 3).tolist(), dists[k] * 1000, transit[k], " -> ".join(reversed(todo)) if todo else "(nothing)"))
        for name in reversed(todo):
            s = PLAN["steps"][name]
            ok_step = _move_palm_once("park-" + name, s["goal_L"], max(1.5, s.get("T", 2.5)), R_PALM_DES, False, q_plan=s["q"])
            if not ok_step and not ARMSDK.frozen:
                # refused (path would brush the table from here): lift 6 cm at the current xy, then try once more
                p_c, _ = KIN.fk(state()["q"])
                p_cL = ARMSDK.R_pL.T @ p_c
                log("PARK: %s refused from here - lifting 6 cm at the current xy first, then retrying" % name)
                ARMSDK.q_attract = None
                if _move_palm_once("park-lift", np.array([p_cL[0], p_cL[1], p_cL[2] + 0.06]), 2.5, None, False) and not ARMSDK.frozen:
                    ok_step = _move_palm_once("park-" + name, s["goal_L"], max(1.5, s.get("T", 2.5)), R_PALM_DES, False, q_plan=s["q"])
            if not ok_step:
                if ARMSDK.frozen:
                    log("PARK: retrace to %s froze (%s) - holding with weight 1. Enter at the prompt retries, L2+B if wedged." % (name, ARMSDK.frozen))
                else:
                    log("PARK: retrace to %s refused - holding here with weight 1." % name)
                ARMSDK.allow_dip = False
                return
        PLAN["done"] = []
        st = state()
        ARMSDK.cmd[ARM] = st["q"][ARM]
    elif ARMSDK.q_safe is not None:
        log("PARK: pulling back to last table-clear pose")
        ARMSDK.start_joint_motion(ARMSDK.q_safe, max(2.5, args.time_scale * 2.0), "park-safe")
        ARMSDK.wait(timeout=12.0)
        if ARMSDK.frozen:
            ARMSDK.unfreeze("park-safe")
        st = state()
        ARMSDK.cmd[ARM] = st["q"][ARM]
    st = state()
    p, _ = KIN.fk(st["q"])
    p_L = ARMSDK.R_pL.T @ p
    zc = palm_clearance_z()
    if zc is None:
        zc = float(p_L[2]) + 0.08
    if p_L[0] > table_edge_x():
        z_up = max(float(zc), float(p_L[2]) + 0.06)
        log("PARK: lifting off the table first (palm x=%.2f z=%.2f -> %.2f)" % (p_L[0], p_L[2], z_up))
        _move_palm_once("park-lift", np.array([p_L[0], p_L[1], z_up]), 2.5, None, False)
        ARMSDK.frozen = None
        st = state()
        p, _ = KIN.fk(st["q"])
        p_L = ARMSDK.R_pL.T @ p
        if palm_in_table(p_L):
            log("PARK: palm still in the table (z_L %.3f) - holding weight 1, not retracting. L2+B if it is wedged." % p_L[2])
            ARMSDK.allow_dip = False
            return
        p0, _ = KIN.fk(ARMSDK.q_start)
        p0_L = ARMSDK.R_pL.T @ p0
        _move_palm_once("park-retract", np.array([p0_L[0], p0_L[1], max(float(zc), float(p_L[2]))]), 3.5, None, False)
        ARMSDK.frozen = None
        st = state()
        ARMSDK.cmd[ARM] = st["q"][ARM]
        p, _ = KIN.fk(st["q"])
        p_L = ARMSDK.R_pL.T @ p
        if palm_in_table(p_L):
            log("PARK: retract left the palm in the table (z_L %.3f) - holding, not releasing." % p_L[2])
            ARMSDK.allow_dip = False
            return
    ARMSDK.allow_dip = False
    q_goal = (ARMSDK.q_home if ARMSDK.q_home is not None else ARMSDK.q_start)[ARM]
    dist = float(np.max(np.abs(q_goal - st["q"][ARM])))
    ok_path, why = path_check(st["q"][ARM], q_goal, st["q"], ARMSDK.R_pL)
    if not ok_path:
        log("PARK: home interpolant refused (%s) - holding here, not releasing." % why)
        return
    ARMSDK.start_joint_motion(q_goal, max(2.0, dist / (0.6 * args.vmax)), "park")
    ARMSDK.wait(timeout=dist / (0.5 * args.vmax) + 8.0)
    st = state()
    log("PARK: residual %s rad" % np.round(q_goal - st["q"][ARM], 3).tolist())
    p, _ = KIN.fk(st["q"])
    p_L = ARMSDK.R_pL.T @ p
    if palm_in_table(p_L):
        log("PARK: still in the table after home (z_L %.3f) - not releasing weight" % p_L[2])
        return
    release_weight()


def palm_clearance_z():
    """Palm z (level) for table-clearing vias: LOOK table + --clearance."""
    if CAN.get("z") is None:
        return None
    return float(CAN["z"]) + float(args.clearance)


def plan_table_vias(p_from_L, p_to_L, q_seed=None, R_pL=None, R_des=None, quiet=False):
    """short xy steps at table-clearance height: [(label, palm_level, q29 or None), ...]. With q_seed/R_pL/R_des each
    via is solved (attracted to the previous one) and its interpolant checked; unreachable vias are dropped."""
    _log = (lambda *a, **k: None) if quiet else log
    p_from_L = np.asarray(p_from_L, dtype=float)
    p_to_L = np.asarray(p_to_L, dtype=float)
    zc = palm_clearance_z()
    horiz = float(np.linalg.norm(p_to_L[:2] - p_from_L[:2]))
    if zc is None or horiz <= 0.12:
        return []
    z_transit = float(zc)
    z_lo = float(CAN["z"]) + 0.03
    z_hi = z_transit + 0.03
    vias = []
    n = max(1, int(math.ceil(horiz / 0.08)))
    for i in range(1, n + 1):
        a = i / float(n)
        xy = (1.0 - a) * p_from_L[:2] + a * p_to_L[:2]
        label = "via-hover" if i == n else "via-mid%d" % i
        vias.append((label, np.array([xy[0], xy[1], z_transit]), None))
    if vias and float(np.linalg.norm(vias[-1][1] - p_to_L)) < 0.02:
        vias.pop()
    if q_seed is None or R_pL is None or R_des is None:
        return vias
    kept, q = [], q_seed.copy()
    z_used = z_transit
    for label, g, _ in vias:
        g_try = None
        q2 = ep = er = None
        for z in (z_transit, z_used, float(CAN["z"] + 0.08), float(CAN["z"] + 0.065), float(CAN["z"] + 0.05)):
            cand = np.array([g[0], g[1], float(np.clip(z, z_lo, z_hi))])
            q2, ep, er = ik_near(reach_arm_guesses(q) + ik_seeds_from(q), R_pL @ cand, R_des, q)
            if ep > 0.01 or er > 0.12:
                continue
            ok_path, why = path_check(q[ARM], q2[ARM], q, R_pL)
            dq = float(np.max(np.abs(q2[ARM] - q[ARM])))
            if not ok_path:
                _log("VIA skip %s z=%.3f: %s" % (label, cand[2], why))
                continue
            if dq > 1.6:
                _log("VIA skip %s z=%.3f: joint jump %.2f rad" % (label, cand[2], dq))
                continue
            g_try, z_used = cand, float(cand[2])
            break
        if g_try is not None:
            kept.append((label, g_try, q2.copy()))
            q = q2
        else:
            _log("VIA skip %s at %s (no table-clear reaching IK from the previous via)" % (label, np.round(g, 3).tolist()))
    return kept


def _move_palm_once(label, goal_L, T, R_goal, contact, q_plan=None):
    if ARMSDK.frozen and not ARMSDK.allow_dip:
        return False
    ARMSDK.set_level(state())
    started = ARMSDK.start_palm_motion(np.asarray(goal_L, dtype=float), R_PALM_DES if R_goal is None else R_goal, T, label,
                                       contact=contact, q_plan=q_plan)
    if not started:
        log("%s: not started" % label.upper())
        return False
    ok = ARMSDK.wait()
    if ok and ARMSDK.frozen is None and label in PLAN["steps"] and q_plan is not None:
        PLAN["done"].append(label)
    st = state()
    p, R = KIN.fk(st["q"])
    p_L = ARMSDK.R_pL.T @ p
    SUMMARY["stages"][label] = dict(goal_level=np.round(goal_L, 4).tolist(), palm_level=np.round(p_L, 4).tolist(),
                                    err_mm=round(float(np.linalg.norm(p_L - np.asarray(goal_L))) * 1000, 1),
                                    q=st["q"][ARM].round(4).tolist(), tau=st["tau"][ARM].round(2).tolist(), frozen=ARMSDK.frozen)
    log("%s: palm (level) %s vs goal %s -> %.1f mm | tau %s" % (label.upper(), np.round(p_L, 3).tolist(), np.round(goal_L, 3).tolist(),
                                                              SUMMARY["stages"][label]["err_mm"], SUMMARY["stages"][label]["tau"]))
    grab_still(label)
    return ok and ARMSDK.frozen is None


def _move_joint_once(label, q_arm, T):
    if ARMSDK.frozen and not ARMSDK.allow_dip:
        return False
    st = state()
    q_body = st["q"].copy()
    if ARMSDK.cmd is not None:
        q_body[ARM] = ARMSDK.cmd[ARM]
    ok_path, why = path_check(q_body[ARM], q_arm, q_body, ARMSDK.R_pL)
    if not ok_path:
        log("%s: joint path refused (%s) - not started" % (label.upper(), why))
        return False
    dq = float(np.max(np.abs(np.asarray(q_arm) - q_body[ARM])))
    ARMSDK.start_joint_motion(q_arm, max(T, 1.57 * dq / (0.8 * args.vmax) / max(args.time_scale, 1e-6)), label)
    ok = ARMSDK.wait()
    st = state()
    p, _ = KIN.fk(st["q"])
    p_L = ARMSDK.R_pL.T @ p
    log("%s: q %s palm (level) %s | tau %s" % (label.upper(), np.round(st["q"][ARM], 3).tolist(),
                                               np.round(p_L, 3).tolist(), np.round(st["tau"][ARM], 2).tolist()))
    grab_still(label)
    return ok and ARMSDK.frozen is None


def replay_plan_stage(stage, T, R_goal):
    """Run the dryrun's waypoints for one stage with their stored joints. Each interpolant is re-checked from the
    arm's actual command first (table + body); a refusal stops the stage with the arm holding where it is."""
    steps = plan_steps_for(stage)
    if not steps:
        log("PLAN %s: no planned steps" % stage)
        return False
    n = len(steps)
    for i, name in enumerate(steps):
        s = PLAN["steps"][name]
        if not s["ok"]:
            log("PLAN %s: step %s was not OK in the dryrun - stopping before it" % (stage, name))
            return False
        p_now, _ = KIN.fk(state()["q"])
        p_now_L = ARMSDK.R_pL.T @ p_now
        log("PLAN %s: %s -> palm (level) %s%s" % (
            stage, name, np.round(s["goal_L"], 3).tolist(),
            "" if CAN.get("z") is None else "  (%.0f cm above the table, %.0f cm from the current palm)" % (
                (s["goal_L"][2] - CAN["z"]) * 100, float(np.linalg.norm(s["goal_L"] - p_now_L)) * 100)))
        # Intermediate 8 cm vias were stuck at 2.5 s (then * time-scale) because s["T"] sat in the max()
        # and the 0.6 factor never won. Use the planned step duration; shorten hops that are not the last in the stage.
        Ti = float(s.get("T", T))
        if i + 1 < n and name.startswith("via"):
            Ti *= 0.6
        Ti = max(1.2, Ti)
        if not _move_palm_once(name, s["goal_L"], Ti, R_goal, False, q_plan=s["q"]):
            return False
        if ARMSDK.frozen:
            return False
    return True


def move_palm(label, goal_L, T, R_goal=None, contact=False):
    """Free-space: replay the dryrun plan for this stage when it is still valid (same can estimate); otherwise fall
    back to live planning: fold to a reaching config, step across the table in short vias, then to the goal."""
    goal_L = np.asarray(goal_L, dtype=float).copy()
    if not contact and ARMSDK.frozen is None and label in ("raise", "pregrasp", "lift", "retreat") and plan_steps_for(label):
        if plan_is_current():
            return replay_plan_stage(label, T, R_goal)
        log("PLAN %s: can estimate moved since the dryrun (%s -> %s) - planning this stage live" % (
            label, {k: round(float(v), 3) for k, v in (PLAN.get("can") or {}).items() if v is not None},
            {k: round(float(CAN[k]), 3) for k in ("x", "y", "z") if CAN.get(k) is not None}))
    if contact and plan_is_current() and label in PLAN["steps"] and PLAN["steps"][label]["ok"]:
        # contact move on the planned joints (posture-corrected live); the press/settle logic is unchanged
        ARMSDK.q_attract = PLAN["steps"][label]["q"]
        return _move_palm_once(label, goal_L, T, R_goal, contact, q_plan=PLAN["steps"][label]["q"])
    if not contact and ARMSDK.frozen is None:
        st = state()
        p, _ = KIN.fk(st["q"])
        p_L = ARMSDK.R_pL.T @ p
        R_use = R_PALM_DES if R_goal is None else R_goal
        R_des = ARMSDK.R_pL @ R_use
        q_seed = st["q"].copy()
        if ARMSDK.cmd is not None:
            q_seed[ARM] = ARMSDK.cmd[ARM]
        q_goal, ep_g, _ = ik_near(reach_arm_guesses(q_seed) + ik_seeds_from(q_seed), ARMSDK.R_pL @ goal_L, R_des,
                                  q_seed if is_reaching_arm(q_seed[ARM]) else reach_arm_guesses(q_seed)[0])
        ARMSDK.q_attract = q_goal
        horiz = float(np.linalg.norm(goal_L[:2] - p_L[:2]))
        vias = []
        if ep_g < 0.02 and horiz > 0.12 and CAN.get("z") is not None:
            if not is_reaching_arm(q_seed[ARM]):
                q_hip, _eph, p_hip = find_hip_reach_q(q_seed, ARMSDK.R_pL, R_des, q_goal)
                if q_hip is None:
                    log("VIA %s: cannot fold to a hip reaching config - not crossing the table" % label)
                    return False
                log("VIA %s: joint-fold to reaching at hip palm %s (dq %.2f rad)" % (
                    label, np.round(p_hip, 3).tolist(), float(np.max(np.abs(q_hip[ARM] - q_seed[ARM])))))
                if not _move_joint_once(label + "-fold", q_hip[ARM], max(3.0, T)):
                    return False
                st = state()
                p, _ = KIN.fk(st["q"])
                p_L = ARMSDK.R_pL.T @ p
                q_seed = st["q"].copy()
                if ARMSDK.cmd is not None:
                    q_seed[ARM] = ARMSDK.cmd[ARM]
            vias = plan_table_vias(p_L, goal_L, q_seed=q_seed, R_pL=ARMSDK.R_pL, R_des=R_des)
        n = max(1, len(vias))
        for i, (vlabel, vg) in enumerate(vias):
            log("VIA %s: %s (level) %s  (%.0f cm above the table)" % (
                label, vlabel, np.round(vg, 3).tolist(), (vg[2] - CAN["z"]) * 100))
            if not _move_palm_once(vlabel, vg, max(1.2, T * (0.6 if i + 1 < n else 1.0)), R_goal, False):
                return False
            if ARMSDK.frozen:
                return False
        st = state()
        p, _ = KIN.fk(st["q"])
        p_L = ARMSDK.R_pL.T @ p
        q_now = st["q"].copy()
        if ARMSDK.cmd is not None:
            q_now[ARM] = ARMSDK.cmd[ARM]
        attract = ARMSDK.q_attract if ARMSDK.q_attract is not None else q_now
        qg, epg, _erg = ik_near(ik_seeds_from(q_now) + reach_arm_guesses(q_now), ARMSDK.R_pL @ goal_L, R_des, attract)
        ok_path, why = path_check(q_now[ARM], qg[ARM], q_now, ARMSDK.R_pL)
        if not ok_path:
            log("MOVE %s: remaining spline refused (%s) - stopping above the table" % (label, why))
            return False
        if epg > 0.025:
            log("MOVE %s: remaining goal IK %.1f mm - stopping above the table" % (label, epg * 1000))
            return False
    return _move_palm_once(label, goal_L, T, R_goal, contact)


ARM_JOINT_SHORT = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


def stage_step(joint, delta, reps, settle=0.8):
    """the smallest possible arm test: one joint, +delta then back, `reps` times, everything else held. Reports how far
    the joint actually moved, the residual after returning, what the other six did, and the peak torque. The first
    call (handshake) proves rt/arm_sdk is live in the robot's current FSM state."""
    k = ARM_JOINT_SHORT.index(joint)
    jidx = ARM[k]
    lim_lo, lim_hi = KIN.lo[jidx] + 0.05, KIN.hi[jidx] - 0.05
    st0 = state()
    q0 = ARMSDK.cmd[ARM].copy()
    if not (lim_lo <= q0[k] + delta <= lim_hi):
        log("STEP refused: %s %+.3f would leave [%.2f, %.2f]" % (joint, q0[k] + delta, lim_lo, lim_hi))
        return False
    results = []
    f0 = loco_fsm()
    log("STEP %s %s: %+.3f rad x%d, everything else held | before: %s" % (SIDE, joint, delta, reps, fsm_text(f0)))
    for r in range(reps):
        q1 = q0.copy()
        q1[k] += delta
        T = max(1.0, abs(delta) / (0.5 * args.vmax))
        tau_peak = 0.0
        ARMSDK.start_joint_motion(q1, T, "step %d/%d %s %+.3f" % (r + 1, reps, joint, delta))
        t_end = now() + T + settle
        while now() < t_end and not ARMSDK.frozen:
            s = state()
            tau_peak = max(tau_peak, float(np.max(np.abs(s["tau"][ARM]))))
            time.sleep(0.02)
        s1 = state()
        moved = float(s1["q"][jidx] - st0["q"][jidx])
        others = np.round(np.delete(s1["q"][ARM] - st0["q"][ARM], k), 3).tolist()
        ARMSDK.start_joint_motion(q0, T, "step %d/%d back" % (r + 1, reps))
        t_end = now() + T + settle
        while now() < t_end and not ARMSDK.frozen:
            s = state()
            tau_peak = max(tau_peak, float(np.max(np.abs(s["tau"][ARM]))))
            time.sleep(0.02)
        s2 = state()
        back = float(s2["q"][jidx] - st0["q"][jidx])
        followed = abs(moved - delta) < max(0.02, 0.35 * abs(delta))
        results.append(dict(rep=r + 1, commanded=delta, moved=round(moved, 4), residual=round(back, 4), others_moved=others,
                            tau_peak_Nm=round(tau_peak, 2), followed=followed))
        log("STEP %d/%d: %s moved %+.3f rad of %+.3f commanded (%.0f%%), back to %+.3f | other joints %s | peak |tau| %.1f Nm -> %s" % (
            r + 1, reps, joint, moved, delta, 100.0 * moved / delta if delta else 0.0, back, others, tau_peak,
            "FOLLOWED" if followed else "DID NOT FOLLOW"))
        if ARMSDK.frozen:
            break
    f1 = loco_fsm()
    live = all(x["followed"] for x in results) and bool(results)
    SUMMARY["step"] = dict(joint=joint, delta=delta, reps=reps, results=results, live=live, fsm_before=f0, fsm_after=f1)
    log("STEP result: arm_sdk %s in this state | after: %s" % (
        "LIVE - joint follows the command" if live else "NOT LIVE - joint did not follow (FSM does not blend arm_sdk here, or the weight is ignored)", fsm_text(f1)))
    return live


def stage_handshake():
    return stage_step("wrist_yaw", 0.06, 1)


def stage_grasp():
    """thumb across, ramp close with contact freeze (the bench recipe), verdict from the contact map."""
    if args.no_hand:
        log("GRASP skipped (--no-hand)")
        return True
    ensure_hand_open("grasp")
    oppose_extra = ["--aux-target", "%.2f" % args.aux_target, "--aux-seconds", "1.2"]
    close_extra = ["--ramp-rate", "%.2f" % args.ramp_rate, "--speed", "1.0", "--stall-threshold", "%.2f" % args.stall_threshold,
                    "--squeeze", "%.2f" % args.squeeze, "--hold", "0", "--keep", "--aux-target", "%.2f" % args.aux_target]
    doc_o = run_hand("oppose", oppose_extra)
    if HAND_LAST_RC != 0:
        log("HAND oppose refused - releasing and retrying once")
        ensure_hand_open("oppose retry")
        doc_o = run_hand("oppose", oppose_extra)
    time.sleep(0.3)
    doc_c = run_hand("close", close_extra)
    if HAND_LAST_RC != 0:
        log("HAND close refused - releasing and retrying once")
        ensure_hand_open("close retry")
        doc_o = run_hand("oppose", oppose_extra)
        time.sleep(0.3)
        doc_c = run_hand("close", close_extra)
    verdict, ok = grasp_verdict(doc_c)
    SUMMARY["grasp"] = dict(verdict=verdict, ok=ok, oppose_out=None if doc_o is None else doc_o.get("args", {}).get("out"),
                            close_summary=None if doc_c is None else doc_c.get("summary"))
    log("GRASP contact map: %s" % verdict)
    grab_still("grasp")
    return ok and HAND_LAST_RC == 0


def stage_release_hand():
    if args.no_hand:
        return
    run_hand("release", [])
    grab_still("release")


def run_all(info):
    global R_PALM_DES
    R_PALM_DES = palm_rotation_des()
    order = STAGES_ALL
    until = args.until or "park"
    stop_idx = order.index(until)
    st = state()
    p0, R0 = KIN.fk(st["q"])
    goals = stage_goals()
    T = {"raise": 3.0, "pregrasp": 3.0, "approach": 2.5, "descend": 1.5, "lift": 2.0, "lower": 2.0, "retreat": 2.0}
    reached = []      # stages completed, for the reverse-out
    grasped = False
    ensure_hand_open("cycle start")
    for k, name in enumerate(order):
        if k > stop_idx:
            break
        if ARMSDK.frozen:
            break
        if name == "raise":
            steps = plan_steps_for("raise")
            if steps and plan_is_current():
                g = PLAN["steps"][steps[-1]]["goal_L"]
                if PLAN["mode"] == "direct":
                    what = ("one move: the %s arm swings OUT to the side and forward into the ready pose, palm %s "
                            "(%.0f cm above the table, %.0f cm before its edge). Keep the %s side clear" % (
                                SIDE, np.round(g, 2).tolist(), (g[2] - CAN["z"]) * 100, (table_edge_x() - g[0]) * 100, SIDE))
                else:
                    what = "lift at the current xy, then fold to the ready pose %s (%s)" % (np.round(g, 2).tolist(), " -> ".join(steps))
                r = prompt_loop("> RAISE = %s. Enter to go: " % what)
            else:
                r = prompt_loop("> RAISE the %s arm to the ready pose (palm %.0f cm above the can top, %.0f cm out). Enter to go: " %
                                (SIDE, (raise_goal(ARMSDK.R_pL.T @ p0)[2] - CAN["z"] - CAN_H) * 100, 5))
            if r == "park":
                break
            if not move_palm("raise", raise_goal(ARMSDK.R_pL.T @ p0), T["raise"], R_goal=R_PALM_DES):
                break
            if ARMSDK.frozen:
                break
        elif name == "pregrasp":
            r = prompt_loop("> PREGRASP: step out over the table (%.0f cm above the top), then down beside the can (%.0f cm off). Enter to go: " % (
                args.clearance * 100, (args.gap + args.pregrasp_gap - 0.023) * 100))
            if r == "park":
                break
            if not move_palm("pregrasp", stage_goals()["pregrasp"], T["pregrasp"]):
                break
        elif name == "approach":
            r = prompt_loop("> APPROACH: check the camera - palm face should be %.0f cm from the can, fingers forward, thumb up. Fix with n/j, Enter to move in (+%.0f cm high): " %
                            ((args.gap + args.pregrasp_gap - 0.023) * 100, args.approach_rise * 100))
            if r == "park":
                break
            move_palm("approach", stage_goals()["approach"], T["approach"], contact=True)
        elif name == "descend":
            r = prompt_loop("> DESCEND %.0f cm onto the grasp height (palm should touch the can with a gentle press). Enter: " % (args.approach_rise * 100))
            if r == "park":
                break
            move_palm("descend", stage_goals()["descend"], T["descend"], contact=True)
        elif name == "grasp":
            r = prompt_loop("> GRASP: thumb across, then ramp-close with contact freeze. Palm on the can? Enter: ")
            if r == "park":
                break
            grasped = stage_grasp()
            if not grasped:
                r = prompt_loop("> contact map is not the can fingerprint. Enter to LIFT anyway, or 'p' to release + park: ")
                if r == "park":
                    stage_release_hand()
                    grasped = False
                    break
        elif name == "lift":
            r = prompt_loop("> LIFT %.0f cm and hold %.0f s. Enter: " % (args.lift * 100, args.hold))
            if r == "park":
                break
            move_palm("lift", stage_goals()["lift"], T["lift"])
            t_hold = now()
            while now() - t_hold < args.hold and not ARMSDK.frozen:
                time.sleep(0.2)
            with LOCK:
                hq = HAND["q"]
            log("HELD %.0f s: hand q %s" % (args.hold, None if hq is None else [round(v, 2) for v in hq]))
            grab_still("held")
        elif name == "lower":
            r = prompt_loop("> LOWER the can back onto the table. Enter: ")
            if r == "park":
                break
            move_palm("lower", stage_goals()["lower"], T["lower"], contact=True)
        elif name == "release":
            r = prompt_loop("> RELEASE the hand (fingers, then thumb). Enter: ")
            if r == "park":
                break
            stage_release_hand()
            grasped = False
        elif name == "retreat":
            r = prompt_loop("> RETREAT: back out to the pre-grasp standoff. Enter: ")
            if r == "park":
                break
            move_palm("retreat", stage_goals()["retreat"], T["retreat"])
        elif name == "park":
            r = prompt_loop("> PARK: return to the start pose and hand the arm back. Enter: ")
            park()
            return
        reached.append(name)
    # ---- stopped early (--until, prompt 'p', or a freeze): reverse out safely ----
    if ARMSDK.frozen:
        r = prompt_loop("> FROZEN (%s). Enter = PARK (slow return, weight 0) | 'r' releases weight where it is: " % ARMSDK.frozen)
        park()
        return
    if args.until == "pregrasp" and "pregrasp" in reached and not ARMSDK.frozen:
        prompt_loop("> at pregrasp. Palm beside the can, above the table? Enter to PARK: ")
    if grasped and args.until in ("grasp", "lift"):
        prompt_loop("> holding. Enter to LOWER (if lifted), RELEASE the hand and back out: ")
        if "lift" in reached:
            move_palm("lower", stage_goals()["lower"], T["lower"], contact=True)
        stage_release_hand()
    elif "descend" in reached or "approach" in reached:
        prompt_loop("> at the can. Enter to back out: ")
    if any(s in reached for s in ("approach", "descend", "lift", "lower")):
        move_palm("retreat", stage_goals()["retreat"], T["retreat"])
    park()


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
def on_sigint(sig, frame):
    if ARMSDK.cmd is None:
        save()
        os._exit(130)
    ARMSDK.freeze("SIGINT")


signal.signal(signal.SIGINT, on_sigint)
if args.look and args.can_z is None:
    can_txt = "can from D435i --look"
else:
    can_txt = "can (level) x=%s y=%s z=%s" % (
        "?" if args.can_x is None else "%.2f" % args.can_x,
        "?" if args.can_y is None else "%.2f" % args.can_y,
        "?" if args.can_z is None else "%.3f" % args.can_z)
log("G1 ARM+CAN test: %s arm, stage %s, %s, kp/kd %.0f/%.1f, vmax %.2f rad/s, time-scale %.2f%s, fine %.2fx/%.2f (pregrasp/approach/descend/lower), windup %.2f rad%s" %
    (SIDE, args.stage, can_txt, args.kp, args.kd, args.vmax, args.time_scale,
     "" if args.speed_rung is None else " (speed-rung %d; next is %s)" % (
         args.speed_rung, "done" if args.speed_rung >= max(SPEED_RUNGS) else "%d=%.2fx/%.2f" % (
             args.speed_rung + 1, SPEED_RUNGS[args.speed_rung + 1][0], SPEED_RUNGS[args.speed_rung + 1][1])),
     args.fine_time_scale, args.fine_vmax, args.windup, (" | " + args.label) if args.label else ""))

info = do_check()
if args.rpc or args.stage == "check":
    rq = rpc_queries()
    SUMMARY["rpc"] = rq
    cm = rq.get("motion_switcher.CheckMode", {})
    log("motion_switcher.CheckMode -> %s" % (cm.get("data") if "data" in cm else cm))
    sl = rq.get("robot_state.ServiceList", {})
    d = sl.get("data")
    if isinstance(d, dict):
        log("services running: %s" % sorted(k for k, v in d.items() if v == 0))
    else:
        log("robot_state.ServiceList -> %s" % (sl,))

if args.stage == "check":
    save()
    os._exit(0)

if args.stage == "fsm":
    try:
        do_fsm_watch()
    finally:
        save()
    os._exit(0)

if sub_lowcmd is not None:
    sub_lowcmd.Close()   # 1 kHz LowCmd_ deserialisation is expensive; only needed for the gate / fsm watch
if args.table_x is not None:
    CAN["table_x"] = float(args.table_x)

need_look = args.stage == "look" or args.look or (
    args.stage in ("dryrun", "raise", "all") and (CAN["z"] is None or CAN["x"] is None) and args.table_height is None)
if need_look:
    look = do_look(info)
    SUMMARY["look"] = look
    if look.get("ok"):
        apply_look(look)
    if args.stage == "look":
        save()
        os._exit(0 if look.get("ok") else 3)
    if not look.get("ok"):
        log("look failed (%s) - not moving. Put a 12 oz can in the head camera view, or pass --can-x/--can-y/--can-z." % look.get("why"))
        save()
        os._exit(3)

if args.stage in ("step", "handshake"):
    joint = "wrist_yaw" if args.stage == "handshake" else args.joint
    delta = 0.06 if args.stage == "handshake" else args.delta
    reps = 1 if args.stage == "handshake" else args.reps
    r = ask("> TAKEOVER the %s arm (weight ramp, hold measured pose), then %s %+0.3f rad x%d and back. "
            "Robot must be standing (not zero-torque). Remote damping in hand. Enter to go, p to abort: " % (SIDE, joint, delta, reps))
    if r in ("p", "x"):
        log("aborted before takeover")
        save()
        os._exit(0)
    begin_motion_session(info)
    try:
        stage_step(joint, delta, reps)
        if not args.keep:
            release_weight()
    finally:
        if ARMSDK.cmd is not None and ARMSDK.weight > 0.0 and not args.keep:
            if ARMSDK.frozen:
                log("ending in a freeze (%s): PARK" % ARMSDK.frozen)
                park()
            else:
                release_weight()
        ARMSDK.running = False
        time.sleep(0.1)
        save()
    os._exit(0)

if args.stage == "recover":
    if not args.resume_plan:
        sys.exit("--stage recover needs --resume-plan <json of the run that left the arm out>")
    with open(args.resume_plan) as f:
        prev = json.load(f)
    pc = prev.get("can") or {}
    for k in ("x", "y", "z", "table_x"):
        if pc.get(k) is not None:
            CAN[k] = float(pc[k])
    dr = (prev.get("summary") or {}).get("dryrun") or {}
    st0 = state()
    PLAN.update(mode=dr.get("mode"), order=[], steps={}, can={k: CAN.get(k) for k in ("x", "y", "z")}, done=[])
    for s in dr.get("stages", []):
        q29 = st0["q"].copy()
        q29[ARM] = np.array(s["q"], dtype=float)
        PLAN["order"].append(s["stage"])
        PLAN["steps"][s["stage"]] = dict(q=q29, goal_L=np.array(s["goal_level"], dtype=float), ok=bool(s.get("ok")), T=2.5)
    arrived = [e["msg"].split("ARRIVED ")[1].split(":")[0] for e in prev.get("events", []) if e["msg"].startswith("ARRIVED ")]
    PLAN["done"] = [n for n in PLAN["order"] if n in arrived]
    tk = (prev.get("summary") or {}).get("takeover") or {}
    q_home = None
    if tk.get("q_start"):
        q_home = st0["q"].copy()
        q_home[ARM] = np.array(tk["q_start"], dtype=float)
    R_PALM_DES = palm_rotation_des()
    log("RECOVER from %s: can %s | plan %s | arrived %s | home q %s" % (
        args.resume_plan, {k: round(float(v), 3) for k, v in CAN.items() if v is not None}, PLAN["order"], PLAN["done"],
        None if q_home is None else np.round(q_home[ARM], 3).tolist()))
    p_now, _ = KIN.fk(st0["q"])
    R_pL0 = rpy_to_mat(float(st0["rpy"][0]), float(st0["rpy"][1]), 0.0).T
    log("RECOVER: palm now (level) %s, table z %.3f edge x %.2f" % (np.round(R_pL0.T @ p_now, 3).tolist(), CAN["z"], table_edge_x()))
    begin_motion_session(info)
    ARMSDK.q_home = q_home
    t_w = now()
    while now() - t_w < 15.0 and ARMSDK.frozen is None:
        st = state()
        werr = float(np.max(np.abs(st["q"][WAIST_IDX] - ARMSDK.waist_target)))
        p, _ = KIN.fk(st["q"])
        if int((now() - t_w) * 2) % 4 == 0:
            log("RECOVER: straightening the torso: waist %s (err %.3f rad), palm (level) %s" % (
                np.round(st["q"][WAIST_IDX], 3).tolist(), werr, np.round(ARMSDK.R_pL.T @ p, 3).tolist()))
        if werr < 0.03:
            break
        time.sleep(0.5)
    try:
        park()
    except Exception as e:  # noqa: BLE001
        log("recover aborted: %r" % (e,))
        traceback.print_exc()
    if ARMSDK.weight > 0.0:
        log("RECOVER: still holding weight %.2f - staying alive so the arm keeps its command. Ctrl-C to exit (the controller keeps the last command)." % ARMSDK.weight)
        while ARMSDK.weight > 0.0:
            time.sleep(2.0)
            st = state()
            p, _ = KIN.fk(st["q"])
            log("holding: palm (level) %s tau %s" % (np.round(ARMSDK.R_pL.T @ p, 3).tolist(), np.round(st["tau"][ARM], 1).tolist()))
    ARMSDK.running = False
    time.sleep(0.1)
    save()
    os._exit(0)

resolve_can_z(info)
reach_ok = do_dryrun(info)
if args.stage == "dryrun":
    save()
    os._exit(0 if reach_ok else 3)
if not reach_ok and args.stage == "all":
    log("dry run failed - not moving.")
    save()
    os._exit(3)

begin_motion_session(info)
try:
    if args.stage == "raise":
        R_PALM_DES = palm_rotation_des()
        st = state()
        p0, _ = KIN.fk(st["q"])
        move_palm("raise", raise_goal(ARMSDK.R_pL.T @ p0), 3.0, R_goal=R_PALM_DES)
        prompt_loop("> holding the ready pose. Enter to PARK: ")
        if not args.keep:
            park()
    else:
        run_all(info)
except Exception as e:  # noqa: BLE001
    log("stage aborted: %r" % (e,))
    traceback.print_exc()
    try:
        finish_arm()
    except Exception:  # noqa: BLE001
        pass
finally:
    try:
        finish_arm()
    except Exception:  # noqa: BLE001
        pass
    ARMSDK.running = False
    time.sleep(0.1)
    st = state()
    if st is not None:
        log("DONE: %s arm q %s" % (SIDE, st["q"][ARM].round(3).tolist()))
    save()
    os._exit(0)
