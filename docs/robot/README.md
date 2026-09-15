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
| USB-485 | FTDI FT4232H quad UART (ttyUSB0–3), Realtek hubs. USB Wi-Fi (`0bda:a85b`). **No camera on the bus.** |
| Cameras (2026-09-10 23:28) | **No feed, nothing to capture.** `/dev/video*` empty; `rs-enumerate-devices` "No device detected"; CSI Argus "No cameras available"; DDS `rt/frontvideostream` silent (0 msgs / 2 s). Unitree `videohub_pc4` is running and *would* publish H.264 from `/dev/video4` @ 1920×1080 if that node existed. A dashboard already waiting on `:8080` (`/tmp/g1-vision-stream.py`, started from 192.168.1.242) reports the same: `{"device": false, "source": "none"}`. Motion PC `.161` pings, no camera TCP ports. Probe: `robot/g1_camera_probe.py` → `docs/robot/snapshot-2026-09-10-camera.json`. |

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

### The same flow in the simulator (22:40–23:05) — first held can

`sim/g1_walk_grasp.py --finger-mode stall` reproduces the bench flow on the whole robot: normalized finger targets
ramp at 0.8/s, each finger freezes at contact + 0.10 when it stops tracking, stiff hold (`docs/robot/
sim-grasp-fixedbase-can-stall-held.json`, video `hand-tests/sim-can-stall-held.mp4`). Getting the arm to the can
took four fixes, each one visible in the video of the previous failure:

| attempt | what happened | fix |
|---|---|---|
| can-stall-1 | approach knocked the free-standing can over (1 mm clearance, IK settling) | more clearance, or an insert |
| can-stall-2 | with a 40 mm insert the fingers gripped the can's **top** (contacts 0.42–0.53); squeeze pivoted it over the lip | palm centre 55 mm, not 60 (bench: hand rested on its pinky edge ≈ 45 mm) |
| can-stall-3/4 | at 45 mm the fingers swept along the table and wedged under the can's base chime | approach 3 cm high, then **descend** onto the grasp height |
| can-stall-5/6 | the **pre-opposed thumb** sticks ~8 cm out in front of the palm and toppled the can while the palm was still 6–9 cm away; with 11 mm clearance the upper fingers dragged the top of the can in first and it slipped | approach with the thumb **up**, rotate it across only when the palm is at the can; 3 mm clearance |
| **can-stall-7** | **held, 14.2 cm of 14** — contacts thumb 0.09, index 0.25, pinky 0.29, middle 0.40, ring 0.40 (deeper than the bench: the sim palm is ~5 mm further off the can) | now the script defaults |

Placement rule that came out of it: palm face 3 mm off the can, palm centre 55 mm above the base, can axis 15 mm
toward the fingertips from the knuckle line, arrive 3 cm high and descend, thumb up until arrival. The bench
protocol (oppose first, then place the can) works for a hand-held can but not for an arm arriving from the side.

**How accurately must the arm place the palm?** (sim, fixed base, one axis varied from the recipe at a time)

| axis | tried | result |
|---|---|---|
| palm clearance (approach depth) | 3 mm · 8 mm · 13 mm | held · held · **dropped** (pinky never reached the can; top dragged in first) |
| along the fingers (distal offset) | +15 mm · +35 mm · −5 mm | held · held (12.7 cm) · **dropped** (can at the palm heel, thumb never touched) |
| height (palm centre above base) | 40 · 55 · 70 mm | held · held · held |
| hand | right · left | held · held (left rose 25 cm for a 14 cm command — check the left-arm lift target) |

| palm press (target inside the can surface) | 5 mm · 10 mm | held · held — and the contact map moves onto the bench values (thumb 0.19–0.21, pinky 0.20–0.23, index 0.28) |

So: ±15 mm in height and roughly −0/+20 mm along the fingers are free; the approach depth is the tight one. With a
5 mm compliant press as the nominal (now the default: the arm's kp 120 PD turns it into a gentle touch) the working
window is about −5 mm (deeper) to +13 mm (short), i.e. the arm must hit the depth to roughly ±9 mm. The contact map
covers the rest: if the pinky/ring run past ~0.45 without stopping, the palm was short — open, step 10 mm deeper,
close again.

## Live arm + can test (no walking)

**Checkpoint 2026-09-14:** two consecutive left-arm cycles on the standing G1 (FSM id 802, AI standing) found the 12 oz Pepsi from the head D435i, placed the palm, grasped (contact map matched the bench fingerprint), lifted ~14 cm, held 5 s, lowered, released, and retraced home. Cycle 1 needed a 2.5 cm `--palm-x-trim` (fingers an inch too far forward) and was one press too deep into the can; cycle 2 with `--press 0` / `--press-lead 0.012` and that trim as default did not knock the can. Neither cycle walks. Default arm is **left**; right needs `--allow-right`. Keep L2+B in hand.

Tool: `robot/g1_arm_can_test.py`. Copy to `/tmp` on the Jetson (`~/miniforge3/envs/g1brainco/bin/python`). After a reboot also copy and start `robot/g1_vision_stream.py --port 8080` (aligned RGB-D on `:8080`; LOOK reads it). If `left hand: 0 Hz`, `sudo systemctl restart brainco_hand.service`.

```bash
PY=~/miniforge3/envs/g1brainco/bin/python
scp -o IPQoS=none robot/g1_arm_can_test.py robot/g1_vision_stream.py robot/revo2_hand_test.py g1-wifi:/tmp/
# vision stream (once per boot):
ssh g1-wifi "cd /tmp && setsid nohup $PY /tmp/g1_vision_stream.py --port 8080 > /tmp/vision.log 2>&1 < /dev/null &"
# full cycle (LOOK fills can x/y/z and table front):
ssh -t g1-wifi "$PY /tmp/g1_arm_can_test.py --stage all --look --until lift --time-scale 1.8 --vmax 0.25"
# laptop-only planner (no robot): --offline --stage dryrun --urdf <g1_29dof_rev_1_0.urdf> --can-x … --table-x …
```

`--until lift` is the proven path. `--until descend --no-hand` is the placement-only rehearsal. `--stage recover --resume-plan <json>` retraces a stranded run (takeover at weight 1, no 0→1 ramp). `--stage check` / `fsm` / `step` still send no / one-joint motion.

The dryrun **is** the live plan: every waypoint's joints are stored and replayed (live re-solve once folded the arm into the chest). Interpolants are checked against the LOOK table slab and a mesh-derived torso/head/hips box model. The standing controller's waist is held upright (kp 120 + bounded integral); a second bounded integral on the arm joints cancels gravity sag (~4 cm at the palm at kp 120). Cross-midline reaches use the elbow-swivel null space so the upper arm stays off the chest.

Left-hand workspace on this counter (table ~0.95 m, edge x≈0.32): can within ~0.50 m and not more than ~2 cm to the robot's **right**. A can at (0.516, −0.036) is refused without moving; the same can at (0.442, −0.002) grasped. Right-of-centre cans need the right arm (not yet run live).

`--stage step` **refuses** to publish while FSM id is 0 / kp is 0. First check on an earlier boot (2026-09-11 01:10): FSM id 0 ZeroTorque; camera later: D435i on the Jetson USB bus, stream on `http://127.0.0.1:8080`.

## Fetch: arm choice + walking to the can (`robot/g1_fetch.py`)

The grasp above assumes the can is already inside one arm's workspace. `g1_fetch.py` wraps it: LOOK, pick the arm,
step/strafe until the can is in that arm's band, then run the grasp as a subprocess. Locomotion is Unitree's loco
API (`SetVelocity`, api 7105 on service `sport`) — the same call `xr_teleoperate --motion` uses while the robot is
in control mode in the **ai** motion-switcher mode with arms on `arm_sdk`, i.e. our exact configuration. From a
lightweight process the service answers GETs in ~85 ms (FSM id **802**, mode 0); the 3104 timeouts seen from the arm
tool were that process's own 1 kHz DDS load. Odometry: `rt/odommodestate` at 52 Hz (position, velocity, IMU rpy),
zero drift standing still. Battery: `rt/lf/bmsstate` soc.

**Workspace map** (offline planner, `--offline` dryruns over a grid, can 0.16 m above the pelvis; `docs/robot/reach_map_{0.26,0.30,0.34}.json`, key = table edge distance):
each arm reaches cans on its own side of the midline down to ~2.5 cm across it (live: left arm y = −0.006 grasped,
y = −0.025 refused) and out to ≥ 0.25 m on its own side; depth past the table edge 0.08–0.22 m with the edge 0.30 m
ahead (up to 0.26 m with the edge at 0.26 m, only 0.20 m at 0.34 m). Absolute limit ≈ 0.52–0.54 m from the pelvis.

**Policy:** arm = left if can y ≥ 0 else right (`--allow-right` needed, else left + walk). No walk if the can is on the
arm's side (y_side 0–0.22), 0.08–0.20 m past the edge, edge 0.25–0.36 m ahead. Farther away: LiDAR finds the counter
**front face** (the driver crops < 1.0 m and glossy tops return little), walk in with calibrated 0.5 m/s quanta, then
the D435i takes over within ~0.6 m. A clipped blue blob at the frame edge is a search cue, not a can estimate; the
next LOOK has to confirm. ≤ 6 locomotion commands; deeper than 0.26 m past the edge → "unreachable, move the can".
Refuses if FSM ∉ {802, 200}, odometry < 20 Hz, tilt > 6°, already moving, `rt/arm_sdk` active in the last 3 s,
battery < 30 %, or without `--allow-walk`. Every command is StopMove + settle, then a fresh LOOK.

**Checkpoint 2026-09-14 evening:** FSM 802 only **steps** at ≥ 0.5 m/s held ≥ 0.6–1.0 s (≤ 0.3 m/s is a lean).
Calibrated: left 0.20/0.30 m, right ~0.22 m, forward 0.18 m (1.0 s) / 0.48 m (1.5 s), turn 16°. One `reposition`
strafed right onto a can at the frame edge; one `--stage fetch --allow-walk --allow-right --auto` then grasped,
lifted, held, released, and parked (fingerprint match, 156 s grasp cycle). Right-arm grasp not yet live. LiDAR
walk-in from 1–2 m as a single `fetch` is the next test.

```bash
PY=~/miniforge3/envs/g1brainco/bin/python
scp -o IPQoS=none robot/g1_fetch.py robot/g1_lidar_look.py robot/g1_arm_can_test.py robot/revo2_hand_test.py g1-wifi:/tmp/
# lidar_driver: ServiceSwitch on (once per boot if it is stopped)
ssh -t g1-wifi "$PY /tmp/g1_fetch.py --stage check"
ssh -t g1-wifi "$PY /tmp/g1_fetch.py --stage lidar"                    # counter front face, no motion
ssh -t g1-wifi "$PY /tmp/g1_fetch.py --stage walk-handshake --allow-walk --auto"   # SetVelocity(0,0,0)
ssh -t g1-wifi "$PY /tmp/g1_fetch.py --stage walk-test --allow-walk --auto --continuous --test-vy 0.50 --test-seconds 1.0"
ssh -t g1-wifi "$PY /tmp/g1_fetch.py --stage reposition --allow-walk --allow-right --auto"   # LOOK -> step -> LOOK, no arm
ssh -t g1-wifi "$PY /tmp/g1_fetch.py --stage fetch --allow-walk --allow-right --auto"        # proven left-arm path
# right arm: rehearse placement first
ssh -t g1-wifi "$PY /tmp/g1_arm_can_test.py --arm right --allow-right --stage all --look --until pregrasp --no-hand --time-scale 1.8 --vmax 0.25"
```

## Safe next steps on the real robot (in order)

1. **Measure the adapter** (calipers): wrist flange face → Revo 2 base flange; also confirm fingers-along-forearm,
   palm-to-midline, thumb-up orientation. Feeds `sim/build_g1_revo2_urdf.py --adapter-offset`.
2. ~~Left hand~~ — done: it was a boot-order race in `brainco_hand_service`, not hardware. After any power cycle,
   if only one hand publishes, `sudo systemctl restart brainco_hand.service`. (A retry loop or `ExecStartPre=sleep 10`
   in the unit would remove the manual step.) `robot/probe_hands.cpp` is the read-only per-port / per-slave-ID probe
   that found it.
3. **FSM watch then a one-joint step** (`g1_arm_can_test.py --stage fsm`, then `--stage step`) before any raise/grasp.
   Operator: remote on → L2+B (damping) → L2+UP (locked standing) → R1+X (main control). Confirm each id in `fsm`.
4. Before any `rt/arm_sdk` use: `RobotStateClient.ServiceSwitch("g1_arm_example", 0)` (or `--stop-arm-example`),
   start from the measured pose, ramp weight 0→1 over ≥2 s, cap 0.35 rad/s, e-stop in hand.
