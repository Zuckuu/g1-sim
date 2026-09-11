#!/usr/bin/env python3
"""Read-only snapshot of a live Unitree G1 (+ BrainCo Revo 2 hands) over DDS.

Runs ON the robot's Jetson (or any machine on the 192.168.123.0/24 robot LAN) and records a few seconds of
everything the robot publishes that matters for the simulator: joint positions, the PD gains the motion
controller is actually sending, IMU, battery, hand finger positions/currents, arm_sdk activity. It never
publishes a command. With --rpc it additionally asks two standard read-only queries (current motion mode,
service list).

    # on the Jetson (python 3.8 env that has unitree_sdk2py). Two passes: cyclonedds deserializes in pure Python,
    # and one 500 Hz topic is about all one Jetson core keeps up with.
    PY=~/miniforge3/envs/g1brainco/bin/python
    $PY g1_snapshot.py --group body --iface eth0 --seconds 4 --out /tmp/g1-body.json --rpc   # lowstate, hands, battery, IMU
    $PY g1_snapshot.py --group cmd  --iface eth0 --seconds 4 --out /tmp/g1-cmd.json          # lowcmd gains, arm_sdk

Output: one JSON file per pass with per-topic rates, the last message of each topic, and statistics of the joint
and finger time series, plus a human summary on stdout. Python 3.8 compatible (the robot ships 3.8). A watchdog
force-exits the process a few seconds after the capture window, so it cannot hang on DDS teardown.
"""
import argparse
import dataclasses
import json
import math
import os
import platform
import socket
import statistics
import sys
import threading
import time
from typing import Any, Dict, List, Optional

# Unitree G1 29-DoF motor order (LowState_.motor_state index -> joint). Slots 29..34 are unused on the 29-DoF.
G1_29_JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
ARM_SDK_WEIGHT_SLOT = 29  # LowCmd_.motor_cmd[29].q carries the arm_sdk blend weight on the 29-DoF
BRAINCO_FINGERS = ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]  # brainco_hand_service order
CAPTURING = threading.Event()  # handlers do nothing once cleared (we cannot Close() readers safely)


def jsonable(x: Any) -> Any:
    if dataclasses.is_dataclass(x):
        return {f.name: jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (bytes, bytearray)):
        return list(x)
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, (int, str, bool)) or x is None:
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    return str(x)


def series_stats(rows: List[List[float]], names: List[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    if not rows:
        return out
    n = min(len(names), len(rows[0]))
    for j in range(n):
        col = [r[j] for r in rows]
        out[names[j]] = {
            "mean": statistics.fmean(col),
            "std": statistics.pstdev(col) if len(col) > 1 else 0.0,
            "min": min(col),
            "max": max(col),
        }
    return out


class TopicTap:
    """Subscribe to one topic, keep the last message, count arrivals, optionally keep a time series."""

    def __init__(self, name: str, msg_type, extract=None, max_rows: int = 5000):
        from unitree_sdk2py.core.channel import ChannelSubscriber

        self.name = name
        self.count = 0
        self.first_t = None  # type: Optional[float]
        self.last_t = None  # type: Optional[float]
        self.last = None
        self.rows = []  # type: List[Any]
        self.extract = extract
        self.max_rows = max_rows
        self.lock = threading.RLock()  # report() calls rate_hz() while holding it
        self.sub = ChannelSubscriber(name, msg_type)
        self.sub.Init(self._on_msg, 0)  # queueLen 0: handler runs on the DDS listener, no extra thread per topic

    def _on_msg(self, msg):
        if not CAPTURING.is_set():
            return
        now = time.monotonic()
        with self.lock:
            self.count += 1
            if self.first_t is None:
                self.first_t = now
            self.last_t = now
            self.last = msg
            if self.extract is not None and len(self.rows) < self.max_rows:
                try:
                    self.rows.append(self.extract(msg))
                except Exception:  # never let a parse error kill the tap
                    pass

    def rate_hz(self) -> Optional[float]:
        with self.lock:
            if self.count < 2 or self.first_t is None or self.last_t is None or self.last_t <= self.first_t:
                return None
            return (self.count - 1) / (self.last_t - self.first_t)

    def report(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "count": self.count,
                "rate_hz": self.rate_hz(),
                "last": jsonable(self.last) if self.last is not None else None,
            }


def build_taps(group: str) -> Dict[str, TopicTap]:
    """group 'body': rt/lowstate + hands + battery + IMU + remote. group 'cmd': rt/lowcmd + rt/arm_sdk.
    'all' subscribes to everything (only if the machine can deserialize ~1000 big msgs/s in Python)."""
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (BmsState_, HandState_, IMUState_, LowCmd_, LowState_,
                                                        MainBoardState_)
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import Error_, MotorStates_, WirelessController_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ as GoSportModeState_

    def ls_row(m):
        ms = m.motor_state
        return {
            "q": [ms[i].q for i in range(35)],
            "dq": [ms[i].dq for i in range(35)],
            "tau": [ms[i].tau_est for i in range(35)],
            "temp0": [ms[i].temperature[0] for i in range(35)],
            "temp1": [ms[i].temperature[1] for i in range(35)],
            "rpy": list(m.imu_state.rpy),
            "gyro": list(m.imu_state.gyroscope),
            "acc": list(m.imu_state.accelerometer),
            "tick": m.tick,
        }

    def lc_row(m):
        mc = m.motor_cmd
        return {
            "q": [mc[i].q for i in range(35)],
            "kp": [mc[i].kp for i in range(35)],
            "kd": [mc[i].kd for i in range(35)],
            "tau": [mc[i].tau for i in range(35)],
            "mode": [mc[i].mode for i in range(35)],
        }

    def hand_row(m):
        st = m.states
        n = min(6, len(st))
        return {"q": [st[i].q for i in range(n)], "dq": [st[i].dq for i in range(n)],
                "cur_A": [st[i].tau_est for i in range(n)]}

    body = {
        "rt/lowstate": lambda: TopicTap("rt/lowstate", LowState_, ls_row),
        "rt/lf/bmsstate": lambda: TopicTap("rt/lf/bmsstate", BmsState_),
        "rt/lf/mainboardstate": lambda: TopicTap("rt/lf/mainboardstate", MainBoardState_),
        "rt/odommodestate": lambda: TopicTap("rt/odommodestate", GoSportModeState_),
        "rt/secondary_imu": lambda: TopicTap("rt/secondary_imu", IMUState_),
        "rt/brainco/left/state": lambda: TopicTap("rt/brainco/left/state", MotorStates_, hand_row),
        "rt/brainco/right/state": lambda: TopicTap("rt/brainco/right/state", MotorStates_, hand_row),
        "rt/dex3/right/state": lambda: TopicTap("rt/dex3/right/state", HandState_),
        "rt/wirelesscontroller": lambda: TopicTap("rt/wirelesscontroller", WirelessController_),
        "rt/lf/emergency_stop": lambda: TopicTap("rt/lf/emergency_stop", Error_),
    }
    cmd = {
        "rt/lowcmd": lambda: TopicTap("rt/lowcmd", LowCmd_, lc_row, max_rows=2000),
        "rt/arm_sdk": lambda: TopicTap("rt/arm_sdk", LowCmd_,
                                       lambda m: {"weight": m.motor_cmd[ARM_SDK_WEIGHT_SLOT].q,
                                                  "kp": [m.motor_cmd[i].kp for i in range(15, 29)],
                                                  "kd": [m.motor_cmd[i].kd for i in range(15, 29)]},
                                       max_rows=1000),
        "rt/user_lowcmd": lambda: TopicTap("rt/user_lowcmd", LowCmd_),
    }
    wanted = {"body": body, "cmd": cmd, "all": {**body, **cmd}}[group]
    return {name: make() for name, make in wanted.items()}


def rpc_queries(timeout_s: float = 2.0) -> Dict[str, Any]:
    """Two standard read-only queries. They do send a DDS request, so they sit behind --rpc."""
    out = {}  # type: Dict[str, Any]
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        msc = MotionSwitcherClient()
        msc.SetTimeout(timeout_s)
        msc.Init()
        code, data = msc.CheckMode()
        out["motion_switcher.CheckMode"] = {"code": code, "data": data}
    except Exception as e:  # noqa: BLE001
        out["motion_switcher.CheckMode"] = {"error": repr(e)}
    try:
        from unitree_sdk2py.rpc.client import Client

        class RobotStateClient(Client):
            def __init__(self):
                super().__init__("robot_state", False)

            def Init(self):
                self._SetApiVerson("1.0.0.1")
                self._RegistApi(1001, 0)  # ServiceSwitch (not used)
                self._RegistApi(1002, 0)  # SetReportFreq (not used)
                self._RegistApi(1003, 0)  # ServiceList

            def ServiceList(self):
                return self._Call(1003, "{}")

        rsc = RobotStateClient()
        rsc.SetTimeout(timeout_s)
        rsc.Init()
        code, data = rsc.ServiceList()
        try:
            data = json.loads(data) if isinstance(data, str) else data
        except Exception:
            pass
        out["robot_state.ServiceList"] = {"code": code, "data": data}
    except Exception as e:  # noqa: BLE001
        out["robot_state.ServiceList"] = {"error": repr(e)}
    return out


class _NoTap:
    """Stand-in for topics not subscribed in this pass."""
    lock = threading.RLock()
    rows = []  # type: List[Any]
    last = None
    count = 0

    @staticmethod
    def rate_hz():
        return None


def summarize(taps: Dict[str, TopicTap], seconds: float) -> Dict[str, Any]:
    def tap(name):
        return taps.get(name, _NoTap)

    ls = tap("rt/lowstate")
    lc = tap("rt/lowcmd")
    summary = {}  # type: Dict[str, Any]

    with ls.lock:
        rows = list(ls.rows)
        last = ls.last
    if last is not None and rows:
        q = [r["q"] for r in rows]
        dq = [r["dq"] for r in rows]
        tau = [r["tau"] for r in rows]
        temp0 = [r["temp0"] for r in rows]
        names = G1_29_JOINTS + ["slot%d" % i for i in range(29, 35)]
        active = [i for i in range(35) if last.motor_state[i].mode != 0 or abs(last.motor_state[i].q) > 1e-6
                  or last.motor_state[i].temperature[0] != 0]
        summary["lowstate"] = {
            "rate_hz": ls.rate_hz(),
            "mode_machine": last.mode_machine,
            "mode_pr": last.mode_pr,
            "tick": last.tick,
            "active_motor_slots": active,
            "n_active_motors": len(active),
            "imu_rpy_deg": [math.degrees(v) for v in last.imu_state.rpy],
            "imu_quaternion_wxyz": list(last.imu_state.quaternion),
            "imu_gyro_rad_s": list(last.imu_state.gyroscope),
            "imu_accel_m_s2": list(last.imu_state.accelerometer),
            "imu_temperature_C": last.imu_state.temperature,
            "q_rad": series_stats(q, names),
            "dq_rad_s": series_stats(dq, names),
            "tau_est_Nm": series_stats(tau, names),
            "motor_temperature_C": {names[i]: last.motor_state[i].temperature[0] for i in range(35)},
            "motor_mode": {names[i]: last.motor_state[i].mode for i in range(35)},
            "motor_vol_V": {names[i]: last.motor_state[i].vol for i in range(35)},
            "motor_motorstate_flags": {names[i]: last.motor_state[i].motorstate for i in range(35)},
        }
        ticks = [r["tick"] for r in rows]
        if len(ticks) > 2:
            dt_ms = [(b - a) for a, b in zip(ticks[:-1], ticks[1:]) if b > a]
            if dt_ms:
                summary["lowstate"]["tick_delta_ms"] = {"median": statistics.median(dt_ms), "max": max(dt_ms)}

    with lc.lock:
        lrows = list(lc.rows)
        llast = lc.last
    if llast is not None:
        names = G1_29_JOINTS + ["slot%d" % i for i in range(29, 35)]
        kp = {names[i]: llast.motor_cmd[i].kp for i in range(35)}
        kd = {names[i]: llast.motor_cmd[i].kd for i in range(35)}
        summary["lowcmd"] = {
            "rate_hz": lc.rate_hz(),
            "note": "PD gains the motion controller is streaming to the motors right now (mode-dependent).",
            "mode_machine": llast.mode_machine,
            "mode_pr": llast.mode_pr,
            "kp": kp,
            "kd": kd,
            "mode": {names[i]: llast.motor_cmd[i].mode for i in range(35)},
            "q_target_rad": series_stats([r["q"] for r in lrows], names) if lrows else {},
            "tau_ff_Nm": series_stats([r["tau"] for r in lrows], names) if lrows else {},
        }

    arm = tap("rt/arm_sdk")
    if "rt/arm_sdk" in taps:
        with arm.lock:
            arows = list(arm.rows)
        summary["arm_sdk"] = {
            "rate_hz": arm.rate_hz(),
            "count": arm.count,
            "weight_last": arows[-1]["weight"] if arows else None,
            "weight_max": max(r["weight"] for r in arows) if arows else None,
            "kp_arm_last": arows[-1]["kp"] if arows else None,
            "kd_arm_last": arows[-1]["kd"] if arows else None,
            "note": "Non-zero rate = something (usually Unitree's built-in arm service) is publishing arm_sdk. "
                    "weight>0 means it is actually steering the arms.",
        }
        summary["user_lowcmd"] = {"count": tap("rt/user_lowcmd").count, "rate_hz": tap("rt/user_lowcmd").rate_hz()}

    for side in ("left", "right"):
        htap = tap("rt/brainco/%s/state" % side)
        if "rt/brainco/%s/state" % side not in taps:
            continue
        with htap.lock:
            hrows = list(htap.rows)
            hlast = htap.last
        if hlast is None:
            summary["brainco_%s" % side] = {"present": False}
            continue
        summary["brainco_%s" % side] = {
            "present": True,
            "rate_hz": htap.rate_hz(),
            "q_norm_0open_1closed": series_stats([r["q"] for r in hrows], BRAINCO_FINGERS),
            "dq_norm": series_stats([r["dq"] for r in hrows], BRAINCO_FINGERS),
            "current_A": series_stats([r["cur_A"] for r in hrows], BRAINCO_FINGERS),
        }

    bms = tap("rt/lf/bmsstate").last
    if bms is not None:
        def scalar(v):
            if isinstance(v, (list, tuple)):
                nz = [x for x in v if x]
                return nz[0] if len(nz) == 1 else (list(v) if nz else 0)
            return v

        cells = [v for v in (list(bms.cell_vol) if isinstance(bms.cell_vol, (list, tuple)) else []) if v]
        voltage = scalar(bms.bmsvoltage)
        current = scalar(bms.current)
        summary["battery"] = {
            "soc_pct": scalar(bms.soc), "soh_pct": scalar(bms.soh),
            "voltage_V": voltage / 1000.0 if isinstance(voltage, (int, float)) and voltage > 100 else voltage,
            "current_A": current / 1000.0 if isinstance(current, (int, float)) and abs(current) > 100 else current,
            "pack_from_cells_V": sum(cells) / 1000.0 if cells else None,
            "cycles": scalar(bms.cycle), "bmsstate": scalar(bms.bmsstate),
            "cell_count": len(cells),
            "cell_min_V": min(cells) / 1000.0 if cells else None, "cell_max_V": max(cells) / 1000.0 if cells else None,
            "temperatures": jsonable(bms.temperature),
            "raw": jsonable(bms),
            "raw_units_note": "bmsvoltage/current/cell_vol are raw; /1000 applied when they look like mV/mA",
        }
    mb = tap("rt/lf/mainboardstate").last
    if mb is not None:
        summary["mainboard"] = jsonable(mb)

    for side in ("left", "right"):
        d3 = tap("rt/dex3/%s/state" % side)
        if d3.last is not None:
            ms = d3.last.motor_state
            summary["dex3_%s" % side] = {
                "rate_hz": d3.rate_hz(),
                "note": "Dex3 topics exist even without Dex3 hands; all-zero q/power means no Dex3 attached.",
                "q": [m.q for m in ms][:7], "power_v": d3.last.power_v, "power_a": d3.last.power_a,
                "error": d3.last.error,
            }
    wc = tap("rt/wirelesscontroller").last
    if wc is not None:
        summary["remote"] = {"rate_hz": tap("rt/wirelesscontroller").rate_hz(), "keys": wc.keys,
                             "lx_ly_rx_ry": [wc.lx, wc.ly, wc.rx, wc.ry]}
    odo = tap("rt/odommodestate").last
    if odo is not None:
        summary["odom"] = {"rate_hz": tap("rt/odommodestate").rate_hz(), "mode": odo.mode,
                           "position_m": list(odo.position), "velocity_m_s": list(odo.velocity),
                           "body_height_m": odo.body_height, "yaw_speed": odo.yaw_speed}
    summary["capture_seconds"] = seconds
    return summary


def print_summary(s: Dict[str, Any]) -> None:
    def fmt(vals, w=7, p=3):
        return " ".join(("%" + str(w) + "." + str(p) + "f") % v for v in vals)

    ls = s.get("lowstate")
    if "lowstate" in s or "brainco_right" in s or "battery" in s:
        print("\n=== G1 body (rt/lowstate) ===")
    if not ls:
        pass
    else:
        print("  rate %.0f Hz  mode_machine=%s mode_pr=%s  active motors=%d %s" % (
            ls["rate_hz"] or 0, ls["mode_machine"], ls["mode_pr"], ls["n_active_motors"],
            "" if ls["n_active_motors"] == 29 else "(NOT 29!)"))
        print("  IMU rpy deg: %s   accel: %s" % (fmt(ls["imu_rpy_deg"], 7, 2), fmt(ls["imu_accel_m_s2"], 6, 2)))
        print("  %-22s %8s %8s %8s %8s %6s" % ("joint", "q_mean", "q_std", "tau_mean", "dq_std", "temp"))
        for name in G1_29_JOINTS:
            q = ls["q_rad"].get(name)
            if not q:
                continue
            print("  %-22s %8.4f %8.5f %8.3f %8.4f %6.0f" % (
                name, q["mean"], q["std"], ls["tau_est_Nm"][name]["mean"], ls["dq_rad_s"][name]["std"],
                ls["motor_temperature_C"][name]))
        if "tick_delta_ms" in ls:
            print("  tick delta ms: median %.1f max %.1f" % (ls["tick_delta_ms"]["median"], ls["tick_delta_ms"]["max"]))
    lc = s.get("lowcmd")
    if lc:
        print("\n=== Motion controller PD gains (rt/lowcmd) ===")
        print("  rate %.0f Hz  mode_machine=%s mode_pr=%s" % (lc["rate_hz"] or 0, lc["mode_machine"], lc["mode_pr"]))
        print("  %-22s %7s %7s %5s %9s %9s" % ("joint", "kp", "kd", "mode", "q_tgt", "tau_ff"))
        for name in G1_29_JOINTS:
            qt = lc["q_target_rad"].get(name, {}).get("mean", float("nan"))
            tf = lc["tau_ff_Nm"].get(name, {}).get("mean", float("nan"))
            print("  %-22s %7.1f %7.2f %5d %9.4f %9.3f" % (name, lc["kp"][name], lc["kd"][name], lc["mode"][name], qt, tf))
    a = s.get("arm_sdk")
    if a is not None:
        print("\n=== rt/arm_sdk ===")
        print("  msgs seen: %s (%.1f Hz), weight last=%s max=%s; rt/user_lowcmd msgs: %s" % (
            a.get("count"), a.get("rate_hz") or 0, a.get("weight_last"), a.get("weight_max"),
            s.get("user_lowcmd", {}).get("count")))
        if a.get("kp_arm_last"):
            print("  arm_sdk kp (L arm 7, R arm 7): %s" % fmt(a["kp_arm_last"], 6, 1))
            print("  arm_sdk kd (L arm 7, R arm 7): %s" % fmt(a["kd_arm_last"], 6, 2))
    for side in ("left", "right"):
        if "brainco_%s" % side not in s:
            continue
        h = s.get("brainco_%s" % side, {})
        print("\n=== BrainCo %s hand (rt/brainco/%s/state) ===" % (side, side))
        if not h.get("present"):
            print("  NOT PRESENT (no messages)")
            continue
        print("  rate %.0f Hz" % (h["rate_hz"] or 0))
        print("  %-10s %8s %8s %8s" % ("finger", "q_mean", "q_std", "cur_A"))
        for f in BRAINCO_FINGERS:
            q = h["q_norm_0open_1closed"].get(f)
            if q:
                print("  %-10s %8.3f %8.4f %8.3f" % (f, q["mean"], q["std"], h["current_A"][f]["mean"]))
    b = s.get("battery")
    if b:
        print("\n=== Battery ===\n  SOC %s%%  SOH %s%%  V=%s  I=%s  pack(from cells)=%s V  cycles %s  cells %s (%s-%s V)  temps %s" % (
            b["soc_pct"], b["soh_pct"], b["voltage_V"], b["current_A"], b["pack_from_cells_V"], b["cycles"],
            b["cell_count"], b["cell_min_V"], b["cell_max_V"], b["temperatures"]))
    for k in ("motion_switcher.CheckMode", "robot_state.ServiceList"):
        if k in s.get("rpc", {}):
            print("\n=== %s ===\n  %s" % (k, json.dumps(s["rpc"][k])[:1500]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="eth0", help="network interface on the robot LAN (default eth0)")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--group", choices=["body", "cmd", "all"], default="body")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out", default="g1-snapshot.json")
    ap.add_argument("--rpc", action="store_true", help="also query motion mode + service list (read-only RPCs)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # Hard watchdog: DDS teardown in the Python SDK can hang; never let this tool outlive its capture by much.
    def _watchdog():
        sys.stderr.write("\n[g1_snapshot] watchdog: forcing exit\n")
        sys.stderr.flush()
        os._exit(3)
    wd = threading.Timer(args.seconds + 25.0, _watchdog)
    wd.daemon = True
    wd.start()

    t_start = time.monotonic()

    def stage(msg):
        sys.stderr.write("[g1_snapshot %6.2fs] %s\n" % (time.monotonic() - t_start, msg))
        sys.stderr.flush()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(args.domain, args.iface)
    stage("dds factory up on %s" % args.iface)

    CAPTURING.set()
    taps = build_taps(args.group)
    stage("subscribed to %d topics" % len(taps))
    t0 = time.time()
    time.sleep(args.seconds)
    CAPTURING.clear()
    elapsed = time.time() - t0
    stage("capture done: " + ", ".join("%s=%d" % (n.split("/")[-2] + "/" + n.split("/")[-1], t.count)
                                       for n, t in taps.items()))

    report = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
                 "iface": args.iface, "domain": args.domain, "group": args.group},
        "capture_seconds": elapsed,
        "topics": {},
    }
    for name, tp in taps.items():
        report["topics"][name] = tp.report()
        stage("serialized %s" % name)
    report["summary"] = summarize(taps, elapsed)
    stage("summary computed")

    def write():
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(jsonable(report), f, indent=1)

    write()  # data is on disk before any RPC can misbehave
    stage("wrote %s" % args.out)
    if args.rpc:
        report["summary"]["rpc"] = rpc_queries()
        write()
        stage("rpc done")
    if not args.quiet:
        print_summary(report["summary"])
        print("\nwrote %s (%.1f kB)" % (args.out, os.path.getsize(args.out) / 1024.0))
    # No subscriber Close(): tearing readers down while the listener fires is what hangs. Just leave.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
