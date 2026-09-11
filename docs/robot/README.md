# Real G1 — values pulled from the robot (2026-09-10, first contact)

Everything here was read from the live robot over DDS with `robot/g1_snapshot.py` (subscribe-only) plus two
standard read-only RPC queries, and from files already on its Jetson. No motion command was sent. The robot was
freshly booted (11 min uptime), upright, motors in **zero-torque** (every joint torque ≈ 0 while the IMU reads
level, so it was hanging on its stand, not standing on its feet). Remote controller off (no `rt/wirelesscontroller`).

Raw captures: `snapshot-2026-09-10-body-zerotorque.json` (rt/lowstate, hands, battery, IMU, services) and
`snapshot-2026-09-10-cmd-zerotorque.json` (rt/lowcmd gains, rt/arm_sdk activity). Re-run any time:

```bash
# on the Jetson (ssh g1); two passes because cyclonedds deserializes in pure Python and one 1 kHz topic is a core
PY=~/miniforge3/envs/g1brainco/bin/python
scp robot/g1_snapshot.py g1:/tmp/ && ssh g1 "$PY /tmp/g1_snapshot.py --group body --seconds 5 --out /tmp/g1-body.json --rpc; \
                                           $PY /tmp/g1_snapshot.py --group cmd  --seconds 4 --out /tmp/g1-cmd.json"
```

## Identity

| Item | Value |
| --- | --- |
| Model | G1 29-DoF (29 active motor slots of 35; `mode_machine = 5`, `mode_pr = 0` i.e. pitch/roll ankle & waist convention) |
| Jetson | Orin NX class, JetPack R35.3.1 (Ubuntu 20.04, Python 3.8 system), 15 GB RAM, **1.9 TB NVMe**, uptime resets to 1970 clock until NTP |
| Network | eth0 192.168.123.164 (robot LAN, DDS domain 0), wlan0 192.168.1.229 (`Public_Wifi`), motion PC at .161 |
| Unitree modules | master_service_pc4 1.0.0.2, unitree_patch_pc4 1.0.0.2, video_hub_pc4 1.0.1.1, ota_pipe 1.1.0.3 |
| Battery | 13S pack, 50.5 V (cells 3.883–3.888 V), **SOC 69 %, SOH 91 %, 44 cycles**, −2.15 A idle (≈110 W with Jetson), 24–28 °C |
| Motor temps | 31–43 °C at rest (motors had been powered) |
| Right hand | **BrainCo Revo 2, serial `BCXTR2265J2500018`** (XT = Touch variant), hardware_type 6, sku_type 1, firmware **1.0.9.U**, Modbus slave 127 on `/dev/ttyUSB1` @ 460800 |
| Left hand | **BrainCo Revo 2, serial `BCXTL2265J2500018`** (Touch left, sku MEDIUM_LEFT), firmware 1.0.9.U, Modbus slave 126 on `/dev/ttyUSB2` @ 460800. Was **not detected at boot**: the service scans each port once, the left hand had not answered Modbus yet (green LED = 24 V only), and there is no retry. `sudo systemctl restart brainco_hand.service` found both hands in 0.5 s; `rt/brainco/left/state` now live at ~65 Hz. |
| USB-485 | FTDI FT4232H quad UART (ttyUSB0–3), Realtek hubs. No Intel RealSense on the bus → no head camera visible. `lidar_driver` stopped, no `rt/utlidar/*` topics. |

## What is running on the robot (services, `robot_state.ServiceList`)

Running (status 0): `ai_sport` (locomotion controller), `motion_switcher` (mode = **`ai`**, form 0), **`g1_arm_example`**
(Unitree's built-in arm-action service — it owns `rt/arm_sdk` when an action plays; must be stopped before we drive
the arms, exactly as BrainCo's README warns), `dex3_service_l/r` (publish empty Dex3 topics, harmless),
`audio_player_service`, `chat_go`, `vui_service` (onboard TTS/ASR: `rt/api/voice`, `rt/api/gpt`, `rt/audio_msg`),
`state_estimator`, `battery_guard`, `emergency_stop`, `basic_service`, `webrtc_*`, `unistore`, `bashrunner`.
Stopped (status 1): `auto_test_arm`, `auto_test_low`, `lidar_driver`, `ota_box`, `unitree_slam`.

Plus, under systemd as user `unitree`: **`brainco_hand.service`** → `~/brainco_hand_service/bin/brainco_hand_server`
(Unitree's serial→DDS bridge, builds from github.com/unitreerobotics/brainco_hand_service). It publishes
`rt/brainco/right/state` and listens on `rt/brainco/right/cmd` (`unitree_go::MotorStates_/MotorCmds_`, 6 entries:
**[thumb, thumb_aux, index, middle, ring, pinky]**, `q` = position normalized 0 open … 1 closed, `dq` = speed 0…1,
`tau_est` = motor current in A). Measured loop rate **59–71 Hz** (target 100 Hz; a serial write+read per cycle).

## Live rates and the robot's resting pose (zero torque, hanging)

| Topic | Type | Rate | Notes |
| --- | --- | --- | --- |
| `rt/lowstate` | hg LowState_ | **1 kHz** (tick Δ = 1 ms; Python kept 870–900 Hz) | 35 motor slots, IMU, remote bytes |
| `rt/lowcmd` | hg LowCmd_ | 995 Hz | controller → motors. In zero-torque: `mode=1, kp=kd=tau=q=0` on all 29 |
| `rt/secondary_imu` | hg IMUState_ | ~830 Hz | torso IMU |
| `rt/lf/bmsstate`, `rt/lf/mainboardstate` | | ~15 Hz | |
| `rt/odommodestate` | go SportModeState_ | ~35 Hz | state-estimator odometry |
| `rt/brainco/right/state` | go MotorStates_ | 59–71 Hz | all fingers 0.000 (open), currents ±0.05 A noise |
| `rt/arm_sdk`, `rt/user_lowcmd` | hg LowCmd_ | 0 (idle) | publishers exist (`g1_arm_example`) but silent until an action runs |
| `rt/wirelesscontroller` | | 0 | remote off |

IMU rpy = (−0.46°, −0.49°, 0.49°), accel z = 9.82 m/s². Resting joint angles (rad) with the arms hanging limp:
waist yaw −0.41; legs hip 0.07–0.08, knee 0.12–0.13, ankle pitch −0.21/−0.23; left arm sh-pitch 0.00, sh-roll −0.04,
**sh-yaw 1.55**, elbow 1.21, wr-roll 0.42, wr-pitch 0.12, wr-yaw −0.33; right arm sh-pitch 0.10, sh-roll 0.06,
**sh-yaw −0.90**, elbow 1.33, **wr-roll −1.06**, wr-pitch 0.12, wr-yaw −0.13. Lesson (the August on-robot program learned
it too): a limp arm is nowhere near the URDF zero, so any arm controller must **start from the measured pose** and
ramp the `arm_sdk` weight 0→1, never jump to a canned pose.

## Files on the robot that matter to us

| Path (on Jetson) | What |
| --- | --- |
| `~/unitree/g1_description/g1_29dof_rev_1_0.urdf` | md5 `f6a38a3a…` — **identical** to the URDF our merged sim model is built from |
| `~/xr_teleoperate/assets/brainco_hand/brainco_{left,right}.urdf`, `brainco.yml` | Unitree's official Revo 2 URDF + retargeting config (copied to `robot/assets/from-robot/`) |
| `~/brainco_hand_service/` | the serial↔DDS bridge (see above) and `test/interactive_hand.cpp` (1,646 lines, Aug 18–20): an arm+hand pose program — arm_sdk kp **60** / kd **1.5**, 50 Hz, max 0.75 rad/s, waits for lowstate, captures HOME from the measured pose, ramps weight |
| `~/unitree-g1-brainco-hand/` | BrainCo's ROS 2 integration (Pinocchio IK, EE frame **0.20 m along +x from the wrist joint**, `smach_config.yaml` still says `robot_dof: 23` — wrong for this robot) |
| `~/unitree_sdk2_latest` (2026-08-17), `~/unitree_sdk2_python` (2025-08-07) | SDKs; python env `~/miniforge3/envs/g1brainco` (3.8) has `unitree_sdk2py` + cyclonedds |
| `~/xr_teleoperate` (2025-10-14) | has `--ee brainco` (`robot_hand_brainco.py` → `rt/brainco/*`) |
| aliases in `~/.bashrc` | `runserver`, `runleft/runright` (interactive_hand), `sdk2test` (arm7 example), `sdk2diag` |

## Real values vs. our simulator

| Quantity | Real robot | Our sim (before) | Action |
| --- | --- | --- | --- |
| Body URDF | `g1_29dof_rev_1_0.urdf` md5 f6a38a3a | same file | none |
| Arm PD gains when we drive the arm (`rt/arm_sdk`) | Unitree's minimal example + August program: **kp 60 / kd 1.5** on all 7. Unitree's own teleop (`xr_teleoperate/robot_arm.py`) and BrainCo's stack: **shoulders+elbow kp 300 / kd 3, wrists kp 40 / kd 1.5** | shoulders 100/2, sh-yaw+elbow 50/2, wrists 40/2 (Unitree's RL sim gains) | `g1_walk_grasp.py --arm-gains {teleop,arm_sdk,sim}`; see the gain study below |
| Arm command rate / speed cap | 50 Hz (`control_dt` 0.02), 0.5–0.75 rad/s | 50 Hz IK tick | none; keep the 0.75 rad/s cap in mind for the real controller |
| Wrist effort limits | URDF: pitch/yaw 5 N·m, roll 25 N·m | same | none |
| Hand command interface | normalized 0…1 per motor, 6 motors, ~60 Hz, current feedback in A | joint targets in rad on 11 joints | map with the URDF ranges below; sim already couples distal = 1.155 × proximal |
| Hand joint ranges (Unitree URDF) | thumb rotation 0–**1.5184**, thumb flex 0–**1.0472** (×2 links), finger proximal 0–**1.4661**, distal 0–1.693 | BrainCo URDF: 1.57 / 1.03 / 1.41 / 1.63 | ≤0.06 rad apart; use Unitree's as the 0…1 scale on this robot |
| Hand mass (Unitree URDF) | base 0.238 kg, thumb 0.07, each finger 0.019 → ≈0.39 kg | BrainCo URDF masses (folded) | compare in the merged model |
| Wrist→hand adapter offset | **still unmeasured** (our 12 mm vs Zack's 8 mm are guesses; Unitree's Dex3 flange is at x = 0.0415, y = ∓0.003 on `*_wrist_yaw_link`) | 12 mm | measure with calipers on the mounted right hand |
| Loco controller gains (standing) | not visible yet: zero-torque streams kp = kd = 0 | Unitree sim gains (legs 150–200 / 5) | re-run `--group cmd` once the operator puts it in damping → stand |
| Onboard speech | `vui_service` + `rt/api/voice` (`AudioClient.TtsMaker`, `SetVolume`, `LedControl`) and `rt/api/gpt` / `rt/audio_msg` ASR text | none | candidate for the "Pepsi or Diet?" prompt without extra hardware |

### Arm stiffness study (sim, fixed base, 20 oz bottle in the 60 mm insert, same approach every run)

The first thing the real gains changed. Same command line, only the grasping arm's PD gains differ from the raise
phase on (`sim/run.sh g1_walk_grasp.py --headless --phase grasp --fixed-base --holder 0.06 --finger-effort 0.4 --gap 0.026
--press 0.003 --distal-offset 0.010 --reach-x 0.28 --lift 0.14 --arm-time-scale 1.6 --arm-gains ...`):

| Arm gains (kp / kd) | Palm error at contact → close | Result |
| --- | --- | --- |
| `sim` — shoulders 100/2, elbow 50/2, wrists 40/2 (last night's runs) | ~0.2 mm → held pose | wrap + partial lift, slips after ~1.5 s (full-chain video) |
| `arm_sdk` 60 / 1.5 on all 7 (Unitree's minimal example) | 17 mm → **62 mm**, 17° | pushed back by the palm press, elbow at its limit, **no lift** (`sim-grasp-fixedbase-armsdk-kp60-failed.json`) |
| `grasp` (= `arm_sdk` 120 / 3 on all 7; now the default) | 17 mm → **0.7 mm**, 0.1° | **held, bottle rose 13.3 cm of 14 cm** (`sim-grasp-fixedbase-armsdk-kp120-held.json`) — first full hold-through-lift in sim |
| `teleop` — shoulders+elbow 300/3, wrists 40/1.5 (what Unitree's teleop sends this robot) | 18 mm → **73 mm**, 24° | palm never reaches the bottle (drifts −x/+z instead of +y), **no lift** (`sim-grasp-fixedbase-teleop-gains-failed.json`) |
| `teleop --wrist-kp 120 --wrist-kd 3` — 300/3 shoulders, 120/3 wrists | 17 mm → **72 mm**, 24° | same deviation as above, **no lift** (`sim-grasp-fixedbase-teleop-stiffwrist-failed.json`) |

Reading: the palm-press grasp needs a stiff arm — the compliance that made last night's lift slip was the arm, not
the fingers — and the one profile that held is **uniform 120/3**. Too soft (60) and the press pushes the arm back;
with 300 on the shoulders the approach path itself deviates before contact, wrist stiffness making no difference,
which points at an interaction between the very stiff joints and our IK integrator/anti-windup (tuned at kp 50–100)
rather than at contact physics. Open item: re-tune `--ik-gain` / the windup clamp for kp 300, or simply run the real
arm at 120/3. On the real robot the gains are ours to set in every `rt/arm_sdk` message and Unitree's own teleop
already runs 300/3 on the big joints, so 120/3 is well inside what this hardware is driven at.

### Normalized ↔ radians for the sim (Unitree URDF scale)

`q_norm = q_rad / upper` per **driven** joint: thumb_metacarpal (thumb_aux) /1.5184, thumb_proximal (thumb) /1.0472,
{index,middle,ring,pinky}_proximal /1.4661. Distal joints follow the linkage (sim: 1.155 × proximal; URDF upper 1.693).
BrainCo's serial order is [thumb, thumb_aux, index, middle, ring, pinky] — note **thumb (flex) first, thumb_aux
(rotation) second** in the DDS array, the opposite of the URDF chain order.

## Real can grasps, both hands (2026-09-10, 22:10–22:25)

Hand-only, robot zero-torque on the stand, operator holding a 12 oz can against the palm (palm vertical, fingers
horizontal across the lower body, thumb starting "up"). Tool: `robot/revo2_hand_test.py` (recordings in
`hand-tests/`, ~95 Hz state). Protocol that held the can on both hands:

1. **OPPOSE** `thumb_aux` 0 → 1.0 in 0.2 steps; each step tracks exactly, ~0.2 s per step at speed 0.6.
2. **CLOSE** thumb + 4 fingers advance 0.10 per step (0.8 s dwell); a finger whose actual lags its command by
   ≥ 0.07 has hit the can and is frozen; stop when all five have stalled.
3. **HOLD** the final command (stall + 0.10). **RELEASE** fingers first, then `thumb_aux` back.

| | thumb | thumb_aux | index | middle | ring | pinky |
|---|---|---|---|---|---|---|
| right, stall position | 0.20 | 1.00 | 0.18 | 0.28 | 0.25 | 0.20 |
| left, stall position | 0.18 | 1.00 | 0.19 | 0.29 | 0.28 | 0.19 |
| hold command (both) | 0.30 | 1.00 | 0.30 | 0.40 | 0.40 | 0.30 |
| left hold current, mean | 0 mA | 200 mA (at its stop) | 0 mA | 0 mA | −1 mA | 1 mA |
| left hold drift over 10 s | 0.000 | −0.028 | 0.000 | 0.000 | 0.000 | 0.000 |

What this tells us:
* Contact comes early: the fingers are only 18–29 % closed (≈ 16–25° at the proximal joints) when they meet a 66 mm
  can staged against the palm. The grasp is palm + nearly straight fingers + opposed thumb, not a deep wrap.
* The closers hold with **zero current**: the Revo 2 finger drives are non-backdrivable, so the squeeze applied at
  stall is kept for free. Grip force is set by how far past the stall we command (0.10 here) and the firmware's
  stall/current protection; we cannot read it from the state.
* Motor transients: every step start/brake shows 0.4–1.0 A (braking to −2 A) for 20–80 ms; steady hold currents are
  < 50 mA on fingers, 200 mA on `thumb_aux` at either end stop. Any over-current guard must be time-filtered
  (`--max-current 1.2 --over-current-seconds 0.25`); a single-sample 0.8 A guard falsely aborted the first left run.
* A 0.10 step takes ~120 ms at speed 0.6, so open → stalled on the can is ~0.4 s of motion; the 0.8 s dwells were
  for observation. The 0.2 s bridge/serial latency is not a factor at demo speeds.
* Identical numbers left/right: one recipe serves both hands (Pepsi in one, Diet Pepsi in the other).

### Smooth closes (22:30–22:45, left hand, 7 more runs)

`--mode ramp`: every closer's command ramps continuously at 50 Hz; the moment a finger stops tracking (cmd − act ≥
threshold, or no motion for 40 ms) it is frozen at actual + squeeze. No pauses. Plot:
`hand-tests/revo2-left-smooth-comparison.png`; recordings `revo2-left-smooth{1..5}.json`, `revo2-left-offset{1,2}.json`.

| run | ramp /s | speed | squeeze | thumb lag | close → all stopped | contact thumb / index / middle / ring / pinky | hold current (mA) | drift 6 s |
|---|---|---|---|---|---|---|---|---|
| step protocol | 0.10 steps | 0.6 | +0.10 | – | 3.26 s | 0.18 / 0.19 / 0.29 / 0.28 / 0.19 | 0–1 | 0.000 |
| smooth-1 | 0.4 | 0.6 | +0.10 | 0 | **0.87 s** | 0.20 / 0.21 / 0.29 / 0.28 / 0.21 | 0–26 | 0.002 |
| smooth-2 | 0.8 | 1.0 | +0.10 | 0 | **0.47 s** | 0.21 / 0.21 / 0.29 / 0.27 / 0.19 | 6–23 | 0.002 |
| smooth-3 | 0.8 | 1.0 | +0.10 | 0.3 s | 0.54 s | 0.06 / 0.29 / 0.34 / 0.33 / 0.28 | 12–21 | 0.005 |
| smooth-4 | 0.8 | 1.0 | +0.20 | 0 | 0.45 s | 0.21 / 0.21 / 0.28 / 0.26 / 0.19 | 9–25 | 0.003 |
| smooth-5 | 1.2 | 1.0 | +0.10 ramped 0.3 s | 0 | 0.64 s (0.32 to contact) | 0.21 / 0.21 / 0.28 / 0.25 / 0.20 | 0–1 | 0.000 |
| offset-1 (can ~2 cm off palm) | 0.8 | 1.0 | +0.10 | 0 | 0.45 s | 0.21 / 0.21 / 0.28 / 0.26 / 0.21 | 8–22 | 0.004 |
| offset-2 (can ~3–4 cm off palm) | 0.8 | 1.0 | +0.10 | 0 | 0.45 s | 0.21 / 0.21 / 0.27 / 0.25 / 0.20 | 9–21 | 0.002 |

Findings:
* **The contact map is a fingerprint.** With the can against the palm every run stops at thumb 0.20 ± 0.01, index 0.21,
  middle 0.28 ± 0.01, ring 0.26 ± 0.02, pinky 0.20 ± 0.01 — 8 of 8 runs, both protocols, both speeds. In the demo this
  is a free grasp check: fingers stopping near these values = can in hand; fingers running past ~0.45 = missed.
* **Tracking latency ≈ 55 ms** (bridge 100 Hz + Modbus + finger controller): the actual lags the command by 0.02 at
  0.4/s, 0.04 at 0.8/s, 0.10 at 1.2/s. Contact threshold must be ≥ 0.06 × rate + 0.03; 0.07 is safe up to 0.8/s,
  1.2/s needed 0.10 and ran at the edge. A 1.2/s close reaches the can in 0.25–0.32 s, 0.8/s in 0.34–0.45 s.
* **Thumb lag 0.3 s changes the grasp**: fingers wrap deeper (0.28–0.34) and the thumb meets the can at 0.06 — the
  fingers push the can into the thumb root before the thumb flexes. Cradle rather than pinch; both hold.
* **Squeeze +0.20 vs +0.10 is invisible in the state** (hold currents 9–25 mA either way; non-backdrivable drives).
  Only a pull test ranks them.
* **Offset runs ended in the same contact map** — the closing fingers drag a loosely held can back against the palm,
  so a 2–4 cm standoff at the basket is recovered by the hand itself (to be confirmed with the operator's account).
* Transients are unchanged by ramping: single-sample braking spikes of 1.3–2.4 A at each finger's stop in every
  mode; steady hold ≤ 26 mA.

**Recommended demo recipe (both hands):** pre-shape `thumb_aux` → 1.0 during the approach (1.2 s ramp); close with
`--mode ramp --ramp-rate 0.8 --speed 1.0 --stall-threshold 0.07 --squeeze 0.10` (optionally `--squeeze-seconds 0.2`,
`--thumb-lag 0.3` for the cradle); verify contact map; release with a 0.8 s finger ramp, thumb_aux last.

## Safe next steps on the real robot (in order)

1. **Measure the adapter** (calipers): wrist flange face → Revo 2 base flange; also confirm fingers-along-forearm,
   palm-to-midline, thumb-up orientation. Feeds `sim/build_g1_revo2_urdf.py --adapter-offset`.
2. ~~Left hand~~ — done: it was a boot-order race in `brainco_hand_service`, not hardware. After any power cycle,
   if only one hand publishes, `sudo systemctl restart brainco_hand.service`. (A retry loop or `ExecStartPre=sleep 10`
   in the unit would remove the manual step.) `robot/probe_hands.cpp` is the read-only per-port / per-slave-ID probe
   that found it.
3. **Right hand open/close cycles** (hand only, robot stays zero-torque on the stand): publish `rt/brainco/right/cmd`
   with speed 1.0, log `rt/brainco/right/state` → real close time, per-finger stall current on nothing and on the
   20 oz bottle held into the palm by hand. That calibrates `--finger-effort` and the stall angle in the sim.
4. Operator: remote on → damping (L2+B) → stand; re-run `g1_snapshot.py --group cmd` → real standing pose and the
   controller's leg/waist/arm kp/kd, which replace the guessed sim gains.
5. Before any `rt/arm_sdk` use: `RobotStateClient.ServiceSwitch("g1_arm_example", 0)` (or BrainCo's launch does it),
   start from the measured pose, ramp weight 0→1 over ≥2 s, cap 0.5 rad/s, e-stop in hand.
