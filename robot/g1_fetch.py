#!/usr/bin/env python3
"""G1 fetch orchestrator: LOOK for the can, pick the arm, walk/strafe until the can is in that arm's workspace, grasp.

The proven grasp stays in g1_arm_can_test.py (run here as a subprocess); this file only adds the decision policy and
the locomotion. Everything is in the pelvis level frame: x forward, y to the robot's LEFT (Unitree convention for
SetVelocity too: vx forward, vy left, vyaw counter-clockwise).

Workspace policy (offline planner map, robot/g1_arm_can_test.py --offline over a grid, 2026-09-14, table 0.16 m
above the pelvis):
  * each arm reaches cans on ITS side of the midline and ~8 cm across it (left arm y=-0.029 is a planner
    chest-hug unless the elbow swivels out; the arm itself can wrap past midline);
  * depth past the table edge: 0.08..0.22 m with the edge 0.30 m ahead, up to 0.26 m with the edge at 0.26 m,
    only to 0.20 m with the edge at 0.34 m. Absolute limit: can ~0.52-0.54 m from the pelvis;
  * so: arm = left if can_y >= -0.08 else right; walk only if the can is outside the arm's band with margin; the walk puts
    the table edge at clamp(0.44 - depth, 0.26, 0.34) and the can at y = +/-0.07 on the arm's side.

Stages (each one is the live test of the next building block; nothing walks without --allow-walk):
  check           read-only: FSM id, odometry, battery, arm_sdk traffic, remote
  policy          read-only: LOOK, arm choice, the walk it WOULD do
  walk-handshake  SetVelocity(0,0,0) + StopMove: proves the loco API accepts commands in this FSM (no motion)
  walk-test       one small strafe/step (--test-vx/--test-vy/--test-seconds), displacement measured by odometry
  reposition      LOOK -> walk -> settle -> LOOK until the can is in band (<= --max-attempts), no arm motion
  fetch           reposition, then the grasp (g1_arm_can_test.py --stage all --look --arm <chosen>)
"""
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--stage", default="policy", choices=["check", "policy", "lidar", "walk-handshake", "walk-test", "reposition", "fetch"])
parser.add_argument("--iface", default="eth0")
parser.add_argument("--domain", type=int, default=0)
parser.add_argument("--arm", default="auto", choices=["auto", "left", "right"], help="auto: left if the can is left of midline or within 8 cm to the right, else right")
parser.add_argument("--allow-right", action="store_true", help="the policy may pick the right arm (else it uses the left and walks)")
parser.add_argument("--allow-walk", action="store_true", help="required for any SetVelocity (walk-handshake/walk-test/reposition/fetch)")
parser.add_argument("--dry", action="store_true", help="print every loco command instead of sending it")
parser.add_argument("--auto", action="store_true", help="no operator prompt before a walk")
# workspace policy
parser.add_argument("--y-target", type=float, default=0.07, help="m: where the can should sit laterally on the arm's side after a walk")
parser.add_argument("--x-target", type=float, default=0.44, help="m: preferred can distance ahead of the pelvis after a walk")
parser.add_argument("--edge-min", type=float, default=0.26, help="m: never put the table edge closer than this")
parser.add_argument("--edge-max", type=float, default=0.34)
parser.add_argument("--depth-max", type=float, default=0.26, help="m: cans deeper than this past the edge are unreachable (no leaning)")
parser.add_argument("--band-y-min", type=float, default=-0.08, help="m: can y on the arm's side must be >= this for 'no walk needed' (negative = that many metres across midline)")
parser.add_argument("--band-y-max", type=float, default=0.22)
parser.add_argument("--band-depth", default="0.08,0.20", help="m: depth past the edge accepted without walking")
parser.add_argument("--band-edge", default="0.25,0.36", help="m: table edge distance accepted without walking")
# walking envelope
parser.add_argument("--walk-speed", type=float, default=0.50, help="m/s for free-form walk-test displacements (FSM 802 only steps at >= 0.5)")
parser.add_argument("--speed-cap", type=float, default=0.50, help="m/s: hard refusal above this (the calibrated quanta use exactly 0.5)")
parser.add_argument("--walk-min", type=float, default=0.04, help="m: displacements below this are not walked")
parser.add_argument("--walk-max", type=float, default=0.75, help="m: per-command cap on each axis for free-form walk-test displacements (quanta are exact)")
parser.add_argument("--walk-seconds-max", type=float, default=2.5)
parser.add_argument("--settle-seconds", type=float, default=2.5, help="s of |v| < 0.03 m/s after StopMove before LOOK")
parser.add_argument("--continuous", action="store_true", help="re-send the velocity at 10 Hz for the whole duration (teleop style)")
parser.add_argument("--max-attempts", type=int, default=6, help="max locomotion commands per reposition")
parser.add_argument("--yaw-tol-deg", type=float, default=8.0, help="square up to the table when its edge is tilted more than this in view")
parser.add_argument("--search-turns", type=int, default=4, help="max 16-degree search turns when no can is in view")
parser.add_argument("--test-vx", type=float, default=0.0)
parser.add_argument("--test-vy", type=float, default=0.15)
parser.add_argument("--test-omega", type=float, default=0.0, help="rad/s yaw rate for walk-test (CCW positive)")
parser.add_argument("--test-seconds", type=float, default=1.0)
parser.add_argument("--omega-cap", type=float, default=0.8, help="rad/s: hard refusal above this (the calibrated turn uses 0.8)")
parser.add_argument("--max-tilt-deg", type=float, default=6.0)
parser.add_argument("--min-soc", type=float, default=30.0, help="%%: refuse to walk below this battery level")
parser.add_argument("--fsm-ok", default="802,200", help="loco FSM ids in which walking is allowed")
# tools
parser.add_argument("--tool", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "g1_arm_can_test.py"))
parser.add_argument("--lidar-tool", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "g1_lidar_look.py"))
parser.add_argument("--lidar-seconds", type=float, default=1.5)
parser.add_argument("--no-lidar", action="store_true", help="camera-only search")
parser.add_argument("--look-xmax", type=float, default=2.6, help="m: camera LOOK depth ROI ahead during the approach")
parser.add_argument("--look-ymax", type=float, default=1.2)
parser.add_argument("--approach-range", type=float, default=1.10, help="m: step forward toward a LiDAR/camera target while its table edge is farther than this")
parser.add_argument("--grasp-args", default="--until lift --speed-rung 7 --auto --auto-pause 0",
                    help="extra flags for the grasp run (fetch stage) when the LEFT arm is chosen. speed-rung 7 = 0.55x/0.75 transit; near-can moves stay at 1.5x/0.35")
parser.add_argument("--grasp-args-right", default="--until lift --speed-rung 7 --auto --auto-pause 0 --fluid --press 0.01 --palm-y-trim 0.02",
                    help="extra flags when the RIGHT arm is chosen (live-gated 16 Sep: right needs fluid transit, press 0.01 and palm-y-trim 0.02 for fingerprint)")
parser.add_argument("--out", default=None)
args = parser.parse_args()

T0 = time.monotonic()
EVENTS = []
OUT = args.out or "/tmp/g1-fetch-%s-%s.json" % (args.stage, datetime.now().strftime("%Y%m%d-%H%M%S"))
SUMMARY = {"stage": args.stage, "args": vars(args), "looks": [], "walks": [], "policy": None}


def now():
    return time.monotonic() - T0


def log(msg):
    line = "[%7.2f] %s" % (now(), msg)
    print(line, flush=True)
    EVENTS.append(dict(t=round(now(), 3), msg=msg))


def save():
    SUMMARY["events"] = EVENTS
    with open(OUT, "w") as f:
        json.dump(SUMMARY, f, indent=1, default=str)
    print("G1_FETCH_OUT %s" % OUT, flush=True)


def die(msg, code=1):
    log("STOP: " + msg)
    save()
    os._exit(code)


# ----------------------------------------------------------------------------------------------------------------------
# DDS: odometry (with IMU), battery, arm_sdk traffic - subscribers only. The loco RPC client is created on demand.
# ----------------------------------------------------------------------------------------------------------------------
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, WirelessController_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_, LowCmd_  # noqa: E402

ChannelFactoryInitialize(args.domain, args.iface)
LOCK = threading.Lock()
ODOM = {"t": None, "n": 0, "p": None, "v": None, "rpy": None, "yaw_speed": None, "mode": None}
BMS = {"soc": None, "t": None}
ARMSDK = {"n": 0, "t": None}
REMOTE = {"n": 0, "t": None, "keys": None}


def on_odom(m):
    with LOCK:
        ODOM.update(t=time.monotonic(), p=[float(v) for v in m.position], v=[float(v) for v in m.velocity],
                    rpy=[float(v) for v in m.imu_state.rpy], yaw_speed=float(m.yaw_speed), mode=int(m.mode))
        ODOM["n"] += 1


def on_bms(m):
    soc = m.soc
    if isinstance(soc, (list, tuple)):
        nz = [x for x in soc if x]
        soc = nz[0] if nz else 0
    with LOCK:
        BMS.update(soc=float(soc), t=time.monotonic())


def on_armsdk(m):
    with LOCK:
        ARMSDK["n"] += 1
        ARMSDK["t"] = time.monotonic()


def on_remote(m):
    with LOCK:
        REMOTE["n"] += 1
        REMOTE["t"] = time.monotonic()
        REMOTE["keys"] = int(m.keys)


_subs = []
for topic, typ, cb in (("rt/odommodestate", SportModeState_, on_odom), ("rt/lf/bmsstate", BmsState_, on_bms),
                       ("rt/arm_sdk", LowCmd_, on_armsdk), ("rt/wirelesscontroller", WirelessController_, on_remote)):
    s = ChannelSubscriber(topic, typ)
    s.Init(cb, 0)
    _subs.append(s)


def odom():
    with LOCK:
        return None if ODOM["p"] is None else dict(ODOM)


def body_frame_delta(p0, yaw0, p1):
    """odom-frame displacement -> (forward, left) in the body frame at the start of the move."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    c, s = math.cos(yaw0), math.sin(yaw0)
    return c * dx + s * dy, -s * dx + c * dy


# ----------------------------------------------------------------------------------------------------------------------
# loco RPC (GET for checks, SET only through walk())
# ----------------------------------------------------------------------------------------------------------------------
_LOCO = {"c": None}


def loco():
    if _LOCO["c"] is None:
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        c = LocoClient()
        c.SetTimeout(1.0)
        c.Init()
        _LOCO["c"] = c
    return _LOCO["c"]


def set_velocity(vx, vy, omega, duration):
    """api 7105 exactly as LocoClient.SetVelocity sends it, but returning (code, data) (the SDK wrapper drops data;
    its StopMove() is SetVelocity(0,0,0) with duration 1)."""
    return loco()._Call(7105, json.dumps({"velocity": [float(vx), float(vy), float(omega)], "duration": float(duration)}))


def stop_move():
    return set_velocity(0.0, 0.0, 0.0, 1.0)


def loco_get(api):
    c = loco()
    code, data = c._Call(api, "{}")
    if code != 0:
        return None
    try:
        d = json.loads(data)
        return d.get("data", d) if isinstance(d, dict) else d
    except Exception:  # noqa: BLE001
        return data


def fsm_id():
    for _ in range(3):
        f = loco_get(7001)
        if f is not None:
            try:
                return int(f)
            except (TypeError, ValueError):
                return f
        time.sleep(0.2)
    return None


# ----------------------------------------------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------------------------------------------
def robot_check(verbose=True):
    time.sleep(2.0)
    o = odom()
    with LOCK:
        n_odom, n_arm, t_arm, soc, n_rem, keys = ODOM["n"], ARMSDK["n"], ARMSDK["t"], BMS["soc"], REMOTE["n"], REMOTE["keys"]
    f = fsm_id()
    arm_recent = t_arm is not None and time.monotonic() - t_arm < 3.0     # an arm program published within 3 s
    info = dict(fsm_id=f, odom_hz=n_odom / 2.0, odom=o, battery_soc=soc, arm_sdk_msgs=n_arm if arm_recent else 0,
                remote_hz=n_rem / 2.0, remote_keys=keys)
    if verbose:
        log("loco FSM id %s | odom %.0f Hz %s | battery %s%% | rt/arm_sdk %s | remote %.0f Hz" % (
            f, info["odom_hz"], "n/a" if o is None else "p=%s rpy=%s deg v=%s" % (
                [round(v, 3) for v in o["p"]], [round(math.degrees(v), 1) for v in o["rpy"]], [round(v, 3) for v in o["v"]]),
            "?" if soc is None else "%.0f" % soc, "ACTIVE (%d msgs)" % n_arm if arm_recent else "quiet", info["remote_hz"]))
    SUMMARY["check"] = info
    return info


def walk_preconditions(info, need_flag=True):
    """Everything that must hold before ANY SetVelocity. Returns a list of problems (empty = go)."""
    bad = []
    if need_flag and not args.allow_walk:
        bad.append("--allow-walk not given")
    ok_ids = [int(v) for v in args.fsm_ok.split(",") if v.strip()]
    if info.get("fsm_id") not in ok_ids:
        bad.append("loco FSM id %s not in %s (robot must be standing under the motion controller)" % (info.get("fsm_id"), ok_ids))
    o = info.get("odom")
    if o is None or info.get("odom_hz", 0) < 20:
        bad.append("odometry missing/slow (%.0f Hz)" % info.get("odom_hz", 0))
    else:
        if max(abs(o["rpy"][0]), abs(o["rpy"][1])) > math.radians(args.max_tilt_deg):
            bad.append("pelvis tilt %.1f/%.1f deg" % (math.degrees(o["rpy"][0]), math.degrees(o["rpy"][1])))
        if max(abs(v) for v in o["v"]) > 0.05:
            bad.append("robot already moving (v=%s)" % [round(v, 3) for v in o["v"]])
    if info.get("arm_sdk_msgs", 0) > 0:
        bad.append("rt/arm_sdk is active (%d msgs) - an arm program is running; never walk with the arm out" % info["arm_sdk_msgs"])
    if info.get("battery_soc") is not None and info["battery_soc"] < args.min_soc:
        bad.append("battery %.0f%% < %.0f%%" % (info["battery_soc"], args.min_soc))
    return bad


# ----------------------------------------------------------------------------------------------------------------------
# LOOK via the arm tool (read-only stage)
# ----------------------------------------------------------------------------------------------------------------------
def lidar_look(tag):
    """Long-range finder: robot/g1_lidar_look.py (read-only). Returns its JSON (ok = a can-like object on a table)."""
    lj = "/tmp/g1-fetch-lidar-%s.json" % tag
    cmd = [sys.executable, args.lidar_tool, "--iface", args.iface, "--seconds", "%.1f" % args.lidar_seconds, "--out", lj]
    log("LIDAR (%s): %s" % (tag, " ".join(cmd[1:])))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    for line in p.stdout.splitlines():
        if "LIDAR:" in line and ("can-like" in line or "patch" in line or "no " in line):
            log("  " + line.strip()[:170])
    try:
        with open(lj) as f:
            d = json.load(f)
    except Exception as e:  # noqa: BLE001
        log("LIDAR failed: rc %d (%r)" % (p.returncode, e))
        return None
    d["tag"] = tag
    SUMMARY.setdefault("lidar", []).append(d)
    return d


def look(tag):
    py = sys.executable
    lj = "/tmp/g1-fetch-look-%s.json" % tag
    cmd = [py, args.tool, "--stage", "look", "--iface", args.iface, "--look-json", lj, "--out", "/tmp/g1-fetch-look-%s-run.json" % tag,
           "--look-xmax", "%.2f" % args.look_xmax, "--look-ymax", "%.2f" % args.look_ymax]
    log("LOOK (%s): %s" % (tag, " ".join(cmd[1:])))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    for line in p.stdout.splitlines():
        if "LOOK:" in line and ("pepsi" in line or "table z" in line or "no " in line):
            log("  " + line.strip()[:160])
    try:
        with open(lj) as f:
            d = json.load(f)
    except Exception as e:  # noqa: BLE001
        log("LOOK failed: rc %d, no json (%r); tail: %s" % (p.returncode, e, p.stdout.strip().splitlines()[-1:] if p.stdout.strip() else p.stderr[-300:]))
        return None
    d["tag"] = tag
    SUMMARY["looks"].append(d)
    if not d.get("ok"):
        log("LOOK (%s): %s" % (tag, d.get("why")))
        return d
    log("LOOK (%s): can x=%.3f y=%.3f z=%.3f | table edge x=%s | depth past edge %.3f" % (
        tag, d["can_x"], d["can_y"], d["can_z"], d.get("table_x_min"), d["can_x"] - float(d["table_x_min"] or d["can_x"])))
    return d


# ----------------------------------------------------------------------------------------------------------------------
# policy
# ----------------------------------------------------------------------------------------------------------------------
LEFT_CROSS_Y = 0.08  # left arm takes cans this far to the robot's right of midline; beyond that, right arm


def choose_arm(can_y):
    """Left for its side plus LEFT_CROSS_Y across midline; else right (or left + walk if right is not allowed)."""
    if args.arm != "auto":
        return args.arm, "forced by --arm"
    if can_y >= -LEFT_CROSS_Y:
        return "left", "can y=%+.3f is in the left-arm band (to %.0f mm across midline)" % (can_y, LEFT_CROSS_Y * 1000)
    if not args.allow_right:
        return "left", "can is %.0f mm to the robot's right (past left-arm %.0f mm band), but --allow-right not given: left arm + walk" % (
            -can_y * 1000, LEFT_CROSS_Y * 1000)
    return "right", "can y=%+.3f is past the left-arm midline band -> right arm" % can_y


def plan_walk(lk, arm):
    """Given a LOOK (real or dead-reckoned) and the arm: is the can in band? If not, the body-frame displacement
    (forward dx, left dy) that puts it in the sweet spot, and the yaw the robot should turn to face the table squarely.
    Returns dict(in_band, unreachable, why, dx, dy, yaw_deg, edge, edge_after, can_after)."""
    sgn = 1.0 if arm == "left" else -1.0          # the arm's side of the midline
    x, y = float(lk["can_x"]), float(lk["can_y"])
    edge = lk.get("table_x_min")
    if edge is None:
        return dict(in_band=False, why="LOOK gave no table edge", dx=0.0, dy=0.0, yaw_deg=0.0, unreachable=True)
    edge = float(edge)
    depth = x - edge
    d_lo, d_hi = [float(v) for v in args.band_depth.split(",")]
    e_lo, e_hi = [float(v) for v in args.band_edge.split(",")]
    y_side = sgn * y                               # positive = on the arm's side
    eyaw = lk.get("table_edge_yaw_deg")
    yaw_deg = 0.0
    # square up only when close: the far-range edge estimate (camera at 2 m, LiDAR slit) is too noisy to steer on
    if eyaw is not None and edge <= args.approach_range and int(lk.get("table_edge_bins") or 5) >= 5 and abs(float(eyaw)) > args.yaw_tol_deg:
        yaw_deg = -float(eyaw)                     # turn CCW by this to square up (see do_look in the arm tool)
    if depth > args.depth_max + 0.005 and edge <= args.approach_range:
        return dict(in_band=False, unreachable=True, dx=0.0, dy=0.0, yaw_deg=yaw_deg, edge=edge, can=[x, y],
                    why="can is %.0f mm past the table edge; the arm reaches %.0f mm at most (edge %.2f m ahead) - move the can" % (
                        depth * 1000, args.depth_max * 1000, args.edge_min))
    reasons = []
    if yaw_deg:
        reasons.append("table edge tilted %+.1f deg in view (turn %+.1f deg)" % (float(eyaw), yaw_deg))
    if not (args.band_y_min <= y_side <= args.band_y_max):
        reasons.append("can %s of the %s hand's band (y_side %+.3f, band %.2f..%.2f)" % (
            "inboard" if y_side < args.band_y_min else "outboard", arm, y_side, args.band_y_min, args.band_y_max))
    if not (d_lo <= depth <= d_hi):
        reasons.append("depth past edge %.3f outside %.2f..%.2f" % (depth, d_lo, d_hi))
    if not (e_lo <= edge <= e_hi):
        reasons.append("table edge %.3f ahead outside %.2f..%.2f" % (edge, e_lo, e_hi))
    if not reasons:
        return dict(in_band=True, unreachable=False, dx=0.0, dy=0.0, yaw_deg=0.0, edge=edge, can=[x, y],
                    why="can in the %s arm's band (y_side %+.3f, depth %.3f, edge %.3f, edge yaw %s)" % (
                        arm, y_side, depth, edge, "n/a" if eyaw is None else "%+.1f deg" % float(eyaw)))
    # target stance: edge at clamp(x_target - depth, edge_min, edge_max), can at sgn * y_target
    edge_after = min(max(args.x_target - depth, args.edge_min), args.edge_max)
    dx = edge - edge_after                          # walk forward by dx (negative = back)
    dy = y - sgn * args.y_target                    # walk left by dy (negative = right): can_y' = y - dy
    if abs(dx) < args.walk_min:
        dx = 0.0
    if abs(dy) < args.walk_min:
        dy = 0.0
    return dict(in_band=False, unreachable=False, dx=dx, dy=dy, yaw_deg=yaw_deg, edge=edge, edge_after=edge - dx, can=[x, y],
                can_after=[x - dx, y - dy], why="; ".join(reasons))


# Step quanta measured on the robot 2026-09-14 (FSM 802, AI standing): the policy only steps for commands of
# >= 0.5 m/s held >= 0.6-1.0 s; almost all of the travel happens after StopMove as the step completes. Below that
# (0.3 m/s, 0.5 rad/s, 0.4 s) the body leans and re-centres. Left steps drift ~6 cm forward (toward the table).
QUANTA = {
    "left_small": dict(vx=0.0, vy=0.5, omega=0.0, T=0.6, move=0.20, drift_fwd=0.06),
    "left": dict(vx=0.0, vy=0.5, omega=0.0, T=1.0, move=0.30, drift_fwd=0.06),
    "right": dict(vx=0.0, vy=-0.5, omega=0.0, T=1.0, move=0.22, drift_fwd=0.02),   # 0.6 s aborted the step (twist)
    "fwd": dict(vx=0.5, vy=0.0, omega=0.0, T=1.0, move=0.18, drift_fwd=0.0),       # measured +181 mm (+3 deg yaw)
    "fwd_long": dict(vx=0.5, vy=0.0, omega=0.0, T=1.5, move=0.48, drift_fwd=0.0),  # measured +477 mm (peak 0.51 m/s)
    "back": dict(vx=-0.5, vy=0.0, omega=0.0, T=1.0, move=0.18, drift_fwd=0.0),     # UNCALIBRATED: assumed = fwd
    "turn_ccw": dict(vx=0.0, vy=0.0, omega=0.8, T=1.0, move_deg=16.0),
    "turn_cw": dict(vx=0.0, vy=0.0, omega=-0.8, T=1.0, move_deg=16.0),
}


def pick_step(plan):
    """One calibrated command toward the plan, highest priority first: get off the table edge (back), square up
    (turn), lateral, depth. None when every residual is below half a quantum."""
    edge = plan.get("edge")
    dx, dy, yaw = plan["dx"], plan["dy"], plan["yaw_deg"]
    if edge is not None and edge < args.edge_min:
        return "back", "table edge %.2f m is closer than --edge-min %.2f" % (edge, args.edge_min)
    if edge is not None and edge > args.approach_range and plan.get("can") is not None:
        # far target (LiDAR or long-range camera): face it, then walk straight at it; lateral fine-tuning waits
        # until the table is within --approach-range where the camera geometry is good
        bearing = math.degrees(math.atan2(plan["can"][1], plan["can"][0]))
        if abs(bearing) > 10.0:
            return ("turn_ccw" if bearing > 0 else "turn_cw"), "approach: can at bearing %+.0f deg, %.2f m - face it" % (bearing, math.hypot(*plan["can"]))
        q = "fwd_long" if edge - QUANTA["fwd_long"]["move"] >= 0.55 else "fwd"
        return q, "approach: table edge %.2f m ahead (> %.2f), can at %.2f m" % (edge, args.approach_range, math.hypot(*plan["can"]))
    if abs(yaw) > args.yaw_tol_deg:
        return ("turn_ccw" if yaw > 0 else "turn_cw"), "square up %+.1f deg" % yaw
    if dy >= 0.10:
        q = "left" if dy >= 0.26 else "left_small"
        if edge is not None and edge - QUANTA[q]["drift_fwd"] < args.edge_min:
            return "back", "a left step drifts %.0f cm toward the table (edge %.2f m)" % (QUANTA[q]["drift_fwd"] * 100, edge)
        return q, "can %.0f cm too far right for the target" % (dy * 100)
    if dy <= -0.11:
        return "right", "can %.0f cm too far left for the target" % (-dy * 100)
    if dx <= -0.12:
        return "back", "table edge %.2f m: step back" % edge
    if dx >= 0.09 and edge is not None and edge - QUANTA["fwd"]["move"] >= args.edge_min:
        return "fwd", "table edge %.2f m: step forward" % edge
    return None, "residuals below one step (dx %+.0f, dy %+.0f mm, yaw %+.1f deg)" % (dx * 1000, dy * 1000, yaw)


def near_band(plan):
    """Out of band only by less than a step? Then let the grasp tool's dryrun be the judge (it refuses safely)."""
    if plan.get("unreachable") or plan.get("edge") is None:
        return False
    return abs(plan["dx"]) < 0.09 and -0.11 < plan["dy"] < 0.10 and abs(plan["yaw_deg"]) <= args.yaw_tol_deg and plan["edge"] <= args.approach_range


def do_step(name, label):
    q = QUANTA[name]
    dx, dy = q["vx"] * q["T"], q["vy"] * q["T"]
    args.continuous = True
    return walk(dx, dy, label, T=q["T"], omega=q["omega"], exact=True)


SEARCH = {"turns": 0, "dir": None}


def search_step(lk, ld=None):
    """The camera sees no confirmed can: pick the next search move from THIS image/scan only.
      1. a rejected blue blob (clipped at the frame edge, too small) -> strafe toward it;
      2. table top in view -> strafe along it toward where it continues (10/50/90 % lateral extent);
      3. LiDAR sees the counter front (>= 1 m) -> face it / walk to it (the camera picks the top up within ~0.6 m);
      4. nothing -> sweep turns.
    Returns (step, why) or (None, why) once --search-turns is spent."""
    if SEARCH["turns"] >= args.search_turns:
        return None, "no can after %d search moves" % SEARCH["turns"]
    SEARCH["turns"] += 1
    if lk is not None and lk.get("hint_xy") is not None:
        hx, hy = lk["hint_xy"]
        if hy < -0.05:
            return "right", "blue blob at the frame edge (x=%.2f y=%+.2f, %s): strafe right" % (hx, hy, lk.get("hint_why"))
        if hy > 0.05:
            return "left_small", "blue blob at the frame edge (x=%.2f y=%+.2f, %s): strafe left" % (hx, hy, lk.get("hint_why"))
    ypct = (lk or {}).get("table_y_pct")
    if lk is not None and lk.get("table_x_min") is not None and ypct:
        lo, med, hi = ypct
        edge = float(lk["table_x_min"])
        if edge < args.edge_min:
            return "back", "table edge %.2f m too close to strafe along it" % edge
        if SEARCH["dir"] is None:
            SEARCH["dir"] = "left_small" if med >= 0.0 else "right"          # table continues on that side
        return SEARCH["dir"], "table top spans y %+.2f..%+.2f (median %+.2f) but no can on it: strafe %s" % (
            lo, hi, med, "left" if SEARCH["dir"] == "left_small" else "right")
    if ld is not None and ld.get("counter_dist") is not None:
        cy = float(ld.get("counter_yaw_deg") or 0.0)
        dist = float(ld["counter_dist"])
        if abs(cy) > args.yaw_tol_deg:
            return ("turn_cw" if cy > 0 else "turn_ccw"), "LiDAR: counter front %.2f m ahead tilted %+.1f deg - face it" % (dist, cy)
        q = "fwd_long" if dist - QUANTA["fwd_long"]["move"] >= 0.55 else "fwd"
        return q, "LiDAR: counter front %.2f m ahead, no can in the camera yet - walk in" % dist
    if SEARCH.get("blind_fwd", 0) < 3 and SEARCH.get("last_counter_dist") is not None and 0.45 < SEARCH["last_counter_dist"] < 1.3:
        # between the LiDAR's 1 m crop and the camera's ~0.6 m table-top range neither sensor sees the counter that
        # the previous scan had straight ahead: one small step, then look again
        SEARCH["blind_fwd"] = SEARCH.get("blind_fwd", 0) + 1
        return "fwd", "blind zone (LiDAR < 1 m, camera > 0.6 m): counter was %.2f m ahead one step ago - step in" % SEARCH["last_counter_dist"]
    SEARCH["dir"] = SEARCH["dir"] if SEARCH["dir"] in ("turn_ccw", "turn_cw") else "turn_ccw"
    return SEARCH["dir"], "nothing in view: sweep %s" % ("left" if SEARCH["dir"] == "turn_ccw" else "right")


# ----------------------------------------------------------------------------------------------------------------------
# walking
# ----------------------------------------------------------------------------------------------------------------------
def prompt(text):
    if args.auto or args.dry:
        return ""
    try:
        return input(text).strip()
    except (EOFError, KeyboardInterrupt):
        return "p"


def walk(dx, dy, label, T=None, omega=0.0, exact=False):
    """One SetVelocity with the body-frame displacement (forward dx, left dy) and/or a yaw rate omega (rad/s, CCW
    positive), then StopMove, then settle. Odometry measures what actually happened. Returns dict or None on refusal.
    T: force the command duration; otherwise dist / --walk-speed, at least 0.8 s. exact: a calibrated quantum -
    dx/dy/T are sent as given (no per-command clamp), still under the hard caps."""
    if not exact:
        dx = max(-args.walk_max, min(args.walk_max, dx))
        dy = max(-args.walk_max, min(args.walk_max, dy))
    dist = max(abs(dx), abs(dy))
    if dist < args.walk_min and abs(omega) < 1e-6:
        log("WALK %s: displacement %.0f/%.0f mm below --walk-min, not walking" % (label, dx * 1000, dy * 1000))
        return dict(dx=dx, dy=dy, skipped=True)
    if T is None:
        T = min(args.walk_seconds_max, max(0.8, dist / args.walk_speed))
    vx, vy = dx / T, dy / T
    if max(abs(vx), abs(vy)) > args.speed_cap + 1e-9:
        log("WALK %s refused: %.2f m/s exceeds the --speed-cap %.2f m/s" % (label, max(abs(vx), abs(vy)), args.speed_cap))
        return None
    if abs(omega) > args.omega_cap + 1e-9:
        log("WALK %s refused: %.2f rad/s exceeds the --omega-cap %.2f rad/s" % (label, abs(omega), args.omega_cap))
        return None
    info = robot_check(verbose=False)
    bad = walk_preconditions(info)
    if bad:
        log("WALK %s refused: %s" % (label, "; ".join(bad)))
        return None
    o0 = odom()
    log("WALK %s: forward %+.0f mm, left %+.0f mm, yaw rate %+.2f rad/s -> SetVelocity(vx=%+.3f, vy=%+.3f, w=%+.2f) for %.2f s, then StopMove%s" % (
        label, dx * 1000, dy * 1000, omega, vx, vy, omega, T, " [DRY]" if args.dry else ""))
    if prompt("> WALK: robot will step %+.0f mm forward / %+.0f mm left / turn %+.0f deg nominal. Remote in hand. Enter to go, p to abort: " % (
            dx * 1000, dy * 1000, math.degrees(omega * T))) in ("p", "x"):
        log("WALK %s aborted by the operator" % label)
        return None
    rec = dict(label=label, dx=dx, dy=dy, omega=omega, vx=vx, vy=vy, T=T, dry=args.dry, p0=o0["p"], yaw0=o0["rpy"][2])
    if not args.dry:
        t_send = time.monotonic()
        code, data = set_velocity(vx, vy, omega, T)
        rec["set_code"] = code
        log("  SetVelocity -> code %s %s (%.0f ms)" % (code, data if code else "", (time.monotonic() - t_send) * 1000))
        if code != 0:
            log("WALK %s: SetVelocity rejected (code %s) - not walking" % (label, code))
            SUMMARY["walks"].append(rec)
            return None
        # watch it: peak speed, tilt guard. --continuous re-sends the velocity at 10 Hz (0.5 s duration each) the way
        # Unitree's teleop does; a single 1 s command in FSM 802 only produced a 16 mm weight shift.
        t_end = time.monotonic() + T + (0.0 if args.continuous else 0.3)
        peak = 0.0
        n_resend = 0
        t_start = time.monotonic()
        t_next = t_start + 0.1
        t_sample = t_start
        profile = []
        while time.monotonic() < t_end:
            o = odom()
            if o is not None:
                sp = math.hypot(o["v"][0], o["v"][1])
                peak = max(peak, sp)
                if time.monotonic() >= t_sample:
                    f_, l_ = body_frame_delta(o0["p"], o0["rpy"][2], o["p"])
                    profile.append((round(time.monotonic() - t_start, 2), round(sp, 3), round(f_, 3), round(l_, 3)))
                    t_sample += 0.25
                if max(abs(o["rpy"][0]), abs(o["rpy"][1])) > math.radians(args.max_tilt_deg + 4):
                    log("WALK %s: tilt %.1f deg during the move -> StopMove" % (label, math.degrees(max(abs(o["rpy"][0]), abs(o["rpy"][1])))))
                    break
            if args.continuous and time.monotonic() >= t_next:
                set_velocity(vx, vy, omega, 0.5)
                n_resend += 1
                t_next += 0.1
            time.sleep(0.02)
        if n_resend:
            rec["resent"] = n_resend
        rec["profile"] = profile
        log("  speed profile (t, |v|, fwd, left): %s" % " ".join("%.2f:%.3f/%+.3f/%+.3f" % p for p in profile))
        code_s, _ = stop_move()
        rec["stop_code"] = code_s
        log("  StopMove -> code %s | peak odom speed %.3f m/s" % (code_s, peak))
        rec["peak_speed"] = peak
    # settle: |v| < 0.03 for settle_seconds (max 8 s)
    t_quiet = None
    t_max = time.monotonic() + 8.0
    while time.monotonic() < t_max:
        o = odom()
        if o is not None and math.hypot(o["v"][0], o["v"][1]) < 0.03 and abs(o["yaw_speed"]) < 0.05:
            if t_quiet is None:
                t_quiet = time.monotonic()
            elif time.monotonic() - t_quiet >= args.settle_seconds:
                break
        else:
            t_quiet = None
        time.sleep(0.05)
    o1 = odom()
    fwd, left = body_frame_delta(o0["p"], o0["rpy"][2], o1["p"])
    dyaw = math.degrees(o1["rpy"][2] - o0["rpy"][2])
    rec.update(p1=o1["p"], moved_forward=fwd, moved_left=left, dyaw_deg=dyaw)
    log("WALK %s done: odometry says forward %+.0f mm, left %+.0f mm, yaw %+.1f deg (commanded %+.0f / %+.0f mm)" % (
        label, fwd * 1000, left * 1000, dyaw, dx * 1000, dy * 1000))
    if not args.dry and math.hypot(dx, dy) >= args.walk_min:
        ratio = math.hypot(fwd, left) / max(1e-6, math.hypot(dx, dy))
        if ratio < 0.3:
            log("WALK %s: moved only %.0f%% of the command - the controller may not step at this speed (raise --walk-speed?)" % (label, ratio * 100))
        elif ratio > 2.0:
            log("WALK %s: moved %.0f%% of the command - overshoot; stopping the reposition loop" % (label, ratio * 100))
            rec["overshoot"] = True
    SUMMARY["walks"].append(rec)
    return rec


# ----------------------------------------------------------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------------------------------------------------------
def stage_policy(lk=None, allow_walk_loop=False):
    """LOOK, choose the arm, decide the next step. With allow_walk_loop: one calibrated command per LOOK until the can
    is in band (<= --max-attempts commands). The grasp only ever follows a REAL LOOK in band."""
    attempt = 0
    steps = 0
    while True:
        attempt += 1
        if lk is None:
            lk = look("a%d" % attempt)
        ld = None
        if (lk is None or not lk.get("ok")) and not args.no_lidar and (lk is None or lk.get("table_x_min") is None):
            # camera sees neither can nor table top: the LiDAR (>= 1 m, crops closer) finds the counter FRONT FACE
            # (distance, yaw, height) and, when the points allow, a can-sized cluster above it
            ld = lidar_look("a%d" % attempt)
            if ld is not None and ld.get("counter_dist") is not None:
                SEARCH["last_counter_dist"] = float(ld["counter_dist"])
                log("LIDAR: counter front %.2f m ahead, yaw %+.1f deg, top %.2f m above the floor, %s wide%s" % (
                    ld["counter_dist"], ld.get("counter_yaw_deg") or 0.0, ld.get("counter_height_m") or 0.0, ld.get("counter_width_txt") or "?",
                    "" if not ld.get("ok") else "; can-like cluster at x=%.2f y=%+.2f" % (ld["can_x"], ld["can_y"])))
            if ld is not None and ld.get("ok"):
                lk = dict(ok=True, source="lidar", tag=ld["tag"], can_x=ld["can_x"], can_y=ld["can_y"], can_z=ld["can_z"],
                          table_x_min=ld["table_x_min"], table_edge_yaw_deg=ld.get("table_edge_yaw_deg"), table_edge_bins=99)
        if lk is not None and lk.get("table_x_min") is not None and lk.get("source") != "lidar":
            SEARCH["last_counter_dist"] = None       # the camera has the table: no blind-zone stepping needed
        if lk is None or not lk.get("ok"):
            # nothing to grasp in this image: search and LOOK again - no memory of earlier images
            step, why_step = search_step(lk, ld)
            log("SEARCH: %s%s" % (why_step, "" if step else " -> giving up"))
            SUMMARY["policy"] = dict(attempt=attempt, search=why_step, look=None if lk is None else lk.get("tag"))
            if step is None:
                die("can not found (%s)" % why_step, 3)
            if not allow_walk_loop:
                return None, lk
            if steps >= args.max_attempts:
                die("still searching after %d commands" % steps, 5)
            rec = do_step(step, "search-%d-%s" % (attempt, step))
            steps += 1
            if rec is None:
                die("walk refused/aborted", 6)
            if args.dry:
                log("SEARCH [DRY]: the turn was not sent; stopping")
                return None, lk
            lk = None
            continue
        arm, why_arm = choose_arm(float(lk["can_y"]))
        plan = plan_walk(lk, arm)
        SUMMARY["policy"] = dict(attempt=attempt, arm=arm, why_arm=why_arm, plan=plan, look=lk.get("tag"))
        log("POLICY: %s | %s" % (why_arm, plan["why"]))
        if plan.get("unreachable"):
            die("unreachable: %s" % plan["why"], 4)
        if plan["in_band"]:
            if lk.get("source") == "lidar":
                lk = None           # never hand a LiDAR estimate to the grasp: confirm with the camera first
                continue
            log("POLICY: no walk needed -> grasp with the %s arm" % arm)
            return arm, lk
        step, why_step = pick_step(plan)
        log("POLICY: %s -> %s (dx %+.0f mm, dy %+.0f mm, yaw %+.1f deg)" % (why_step, step or "no step", plan["dx"] * 1000, plan["dy"] * 1000, plan["yaw_deg"]))
        if step is None and near_band(plan) and lk.get("source") != "lidar":
            log("POLICY: within one step of the band -> let the grasp tool's dryrun decide (it refuses safely)")
            return arm, lk
        if not allow_walk_loop:
            return arm, lk
        if step is None:
            die("out of band but every residual is below one step (%s)" % plan["why"], 5)
        if steps >= args.max_attempts:
            die("still out of band after %d commands" % steps, 5)
        rec = do_step(step, "reposition-%d-%s" % (attempt, step))
        steps += 1
        if rec is None:
            die("walk refused/aborted", 6)
        if args.dry:
            log("POLICY [DRY]: the command was not sent; stopping after the first plan")
            return arm, lk
        lk = None       # LOOK again from the new stance


def grasp_cmd_for(arm):
    """Grasp subprocess command for the chosen arm, with that arm's calibrated flags. Right keeps tonight's
    recipe (fluid + press + y-trim); left keeps the long-proven plain rung-7 flags."""
    cmd = [sys.executable, args.tool, "--stage", "all", "--look", "--arm", arm, "--iface", args.iface]
    cmd += (args.grasp_args_right if arm == "right" else args.grasp_args).split()
    if arm == "right":
        if not args.allow_right:
            die("policy chose the right arm without --allow-right (should not happen)")
        cmd.append("--allow-right")
    cmd += ["--out", OUT[:-5] + "-grasp.json"]
    return cmd


def stage_fetch():
    arm, lk = stage_policy(allow_walk_loop=True)
    cmd = grasp_cmd_for(arm)
    log("GRASP: %s" % " ".join(cmd[1:]))
    if args.dry:
        log("GRASP skipped [DRY]")
        return
    save()
    rc = subprocess.call(cmd)
    log("GRASP exit %d" % rc)
    SUMMARY["grasp_rc"] = rc
    if rc == 3 and not SUMMARY.get("grasp_retry"):
        # the tool's dryrun refused (exit 3, nothing moved): one corrective step from a fresh LOOK, then try again
        SUMMARY["grasp_retry"] = True
        log("GRASP: dryrun refused - one more reposition pass, then retry")
        SEARCH["turns"] = 0
        arm2, lk2 = stage_policy(allow_walk_loop=True)
        cmd = grasp_cmd_for(arm2)  # rebuild: the retry may switch arms, and each arm has its own flags
        log("GRASP retry: %s" % " ".join(cmd[1:]))
        save()
        rc = subprocess.call(cmd)
        log("GRASP retry exit %d" % rc)
        SUMMARY["grasp_rc"] = rc


try:
    log("G1 FETCH stage %s | arm %s%s | walk %s%s" % (
        args.stage, args.arm, " (+right allowed)" if args.allow_right else "", "ALLOWED" if args.allow_walk else "disabled",
        " [DRY]" if args.dry else ""))
    info = robot_check()
    if args.stage == "check":
        bad = walk_preconditions(info, need_flag=False)
        log("walk preconditions: %s" % ("OK" if not bad else "; ".join(bad)))
    elif args.stage == "policy":
        stage_policy(allow_walk_loop=False)
    elif args.stage == "lidar":
        ld = lidar_look("solo")
        if ld is None:
            die("lidar look failed")
        log("LIDAR stage: %s" % ("can-like object at x=%.2f y=%+.2f, table edge %.2f m" % (ld["can_x"], ld["can_y"], ld["table_x_min"]) if ld.get("ok") else ld.get("why")))
    elif args.stage == "walk-handshake":
        bad = walk_preconditions(info)
        if bad:
            die("; ".join(bad))
        if prompt("> HANDSHAKE: SetVelocity(0,0,0, 0.5 s) then StopMove - no motion expected. Remote in hand. Enter: ") in ("p", "x"):
            die("aborted")
        if args.dry:
            log("HANDSHAKE [DRY]: would call SetVelocity(0,0,0,0.5) and StopMove")
        else:
            o0 = odom()
            code, data = set_velocity(0.0, 0.0, 0.0, 0.5)
            log("SetVelocity(0,0,0,0.5) -> code %s %s" % (code, data if code else ""))
            time.sleep(0.8)
            code_s, data_s = stop_move()
            log("StopMove -> code %s %s" % (code_s, data_s if code_s else ""))
            time.sleep(1.0)
            o1 = odom()
            fwd, left = body_frame_delta(o0["p"], o0["rpy"][2], o1["p"])
            log("HANDSHAKE: odometry moved %+.0f/%+.0f mm (expect ~0). %s" % (
                fwd * 1000, left * 1000, "loco API accepts velocity commands in this FSM" if code == 0 else "REJECTED - walking is not available in FSM %s" % info.get("fsm_id")))
            SUMMARY["handshake"] = dict(set_code=code, stop_code=code_s, moved=[fwd, left])
    elif args.stage == "walk-test":
        rec = walk(args.test_vx * args.test_seconds, args.test_vy * args.test_seconds, "test", T=args.test_seconds, omega=args.test_omega)
        if rec is None:
            die("walk-test refused")
    elif args.stage == "reposition":
        stage_policy(allow_walk_loop=True)
    elif args.stage == "fetch":
        stage_fetch()
finally:
    save()
