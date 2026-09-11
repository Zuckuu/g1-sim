# Two-week plan — Revo 2 edition

Demo date: ~2026-09-23 (two weeks from 2026-09-09; exact date TBD). Audience: PepsiCo executives.
Goal of the demo: a supervised autonomous loop — follow → stop → ask "Pepsi or Diet?" → grasp the right bottle
from a carrier's basket → hand it over → repeat until "Pepsi delivered". Goal of the meeting: a paid R&D pilot.

## Ground truth we work from

* Robot: Unitree G1 (29-DoF assumed) with Jetson pack; **not yet in hand, arrival date unknown**.
* Hands: two BrainCo Revo 2, **physically here**. Variant (Basic/Pro/Touch), power adapter and USB-RS485 adapter
  availability: unconfirmed (see open questions).
* Compute: RTX 5060 Laptop (8 GB). Isaac Sim 5.0 + Isaac Lab 2.2 run here with CPU physics. Fine for hand and
  arm-scale scenes, not for RL at scale.
* Sim assets available now: Unitree G1 29-DoF USDs (fixed-base and whole-body, Dex1/Dex3/Inspire variants),
  BrainCo official Revo 2 URDF/MJCF/USD, Unitree G1 URDF+meshes, BrainCo's G1 integration stack (ROS 2, dual
  RS-485, Pinocchio arm IK), Unitree xr_teleoperate with `--ee brainco`.

## Critical path (what actually decides the demo)

1. **Real hand holds the real bottle** — bench test, this week, needs only power + USB-RS485 (`hand/`).
   Gate: 10/10 closes on a full 500 mL bottle in a clamp, no slip for 10 s, at `normal` force or below.
2. **G1 arm brings the pre-shaped hand to a bottle in a basket** — sim first (fixed-base G1 + Revo 2 model),
   then real. Gate in sim: scripted approach + grasp + lift for a bottle at 5 basket positions, ≥9/10.
3. **Handover** — release only when the guest has the bottle (motor current drop / touch pressure / brief pull
   detection), never on a timer. Gate: 10/10 handovers to a person on the bench (hand in a clamp is enough).
4. **Speech loop** — ASR + intent ("pepsi"/"diet"/"stop"), TTS prompt, timeouts and re-asks. Off the critical
   path technically, on it for the show. Gate: works with room noise and three different voices.
5. **Following** — one designated leader with a visual marker, bounded speed, stop on loss, stop when leader
   stops. Only after 1–3 pass. Fallback: stationary serving station, carriers come to the robot.

## Calendar

| Days | Track | Evidence required |
| --- | --- | --- |
| 1–2 | Bench: probe motor mapping, grasp cycles on the real bottle, log currents/pressures | `hand/logs/*.json`, video |
| 1–3 | Sim: Revo 2 grasp pocket tuned (done: first held grasps), 20 oz variant, friction sensitivity | `revo2-*.json` sweeps |
| 2–4 | Sim: combined G1+Revo 2 model (URDF merge → USD), fixed-base arm reach to basket positions with Pinocchio IK from BrainCo's stack | reach map, joint limits respected |
| 3–5 | Handover signal on the bench (current drop / pressure), release policy | 10/10 bench handovers |
| 4–6 | Speech loop as a standalone service; basket marker detection (AprilTag) with a webcam | end-to-end table-top mock without robot |
| on arrival | G1: hands mounted (wrist adapter, dual RS-485 to the Unitree USB-485 board, power from top port 2), BrainCo arm/hand services up, arm_sdk verified | hands respond via `rt/brainco/*/cmd`, arm follows IK |
| arrival+2 | Stationary serving loop on the real robot: ask → grasp from fixed basket → handover | 8/10 |
| arrival+4 | Following with a marked leader; stop/loss behaviour | 10 laps, zero collisions |
| last 3 days | Freeze. Rehearse with real carriers, bottles, room spacing, noise. Operator with e-stop drilled | 3 clean full runs |

If the robot arrives with fewer than 5 days left, the demo is the **stationary** version with a rehearsed
script; following is dropped, not the grasp.

## Status after night 1 (2026-09-10, 01:30)

Done in sim: full G1 + Revo 2 model; Unitree's locomotion policy walks it; arm IK reaches a palm target to 0.2 mm;
grasp state machine with videos and logs. Not done: a grasp that survives the lift. Key correction: the hand-only
"held" results were an artifact of 1 kg fingertips from the URDF importer; with real finger masses the side grasp on
a free-standing bottle fails for a physical reason (bottle tips at ~2 N; fingers push away from the palm before they
hook). Next in sim, in order: (1) palm-contact approach (advance until the bottle moves or a palm contact sensor
fires), (2) thumb-first close with current-limited fingers (`--finger-effort 0.4`), (3) lift only when all four
fingers report stall, (4) basket-insert geometry in the scene since that is the demo. On the bench, the same three
questions decide everything, which is why the power adapter matters this week.

## Status after first contact with the real robot (2026-09-10, 21:30)

The G1 is here, on its stand, freshly booted, zero-torque, SSH from this laptop works (`ssh g1`). Read-only capture
with `robot/g1_snapshot.py` (details and every number in `docs/robot/README.md`):

* It is a 29-DoF G1 (`mode_machine 5`), motion mode `ai`, battery 69 % / SOH 91 % / 44 cycles, `rt/lowstate` at 1 kHz.
* The **right Revo 2 (Touch, serial BCXTR2265J2500018, fw 1.0.9.U) is mounted and live** on `rt/brainco/right/state` at
  ~60 Hz through Unitree's `brainco_hand_service`. The **left hand is not detected** on any RS-485 port.
* Someone already worked on this robot in August (`brainco_hand_service/test/interactive_hand.cpp`, arm_sdk poses,
  aliases). Their arm gains are Unitree's `rt/arm_sdk` example gains, **kp 60 / kd 1.5**.
* Sim consequence, tested tonight: with those real arm gains the palm-press approach collapses (palm error 17 → 60 mm
  on contact, elbow at its limit, no lift) where the stiffer RL sim gains lifted the bottle out. Arm stiffness under
  arm_sdk is now the first hardware parameter to establish; `g1_walk_grasp.py --arm-gains arm_sdk --arm-sdk-kp N`
  runs the sim at any candidate value.
* Gates that moved: the Revo 2 bench test no longer needs a power adapter (the robot powers the hand); it becomes a
  hand-only DDS test on the mounted hands, robot left in zero-torque on the stand. Both hands are online (the left
  was a boot-order race in `brainco_hand_service`, fixed by a service restart).
* **12 oz can, first sim results (21:50):** the bottle recipe does not transfer. Whole robot, fixed base, `grasp`
  gains, can in a 40 mm insert: the 3 mm palm press shoves the 0.38 kg can over the insert wall (`sim-grasp-fixedbase-can-1`);
  without the press, grasp heights 45–60 mm and finger effort 0.3–0.35 lift it 1.7–2.2 cm and drop it (`can-a`, `-c`);
  a 50 mm insert puts the fingers into the wall (`can-b`). Hand-only top grasp (palm on the lid, fingers down the
  neck), 9 placements, effort 0.4: 0/9. The can is short, light and smooth: the partial finger wrap pushes it out of
  the hand instead of into the palm. Next: (1) the real right hand on a real can — five minutes on the robot tells us
  the true friction and stall currents the sim is guessing at; (2) in sim, close-first-then-press (fingers drag the can
  into the palm before the arm loads it), thumb opposition lower on the can, and a cup-shaped insert the can can pivot
  in; (3) top grasp with fingertips hooked under the seam rather than on the neck.
* **Real hands hold the can (22:10–22:45).** Right first, then reproduced on the left with `robot/revo2_hand_test.py`:
  thumb across (`thumb_aux` 1.0), fingers stop on a palm-staged can at thumb 0.20 / index 0.21 / middle 0.28 /
  ring 0.26 / pinky 0.20 of their range, hold at +0.10 with zero current (non-backdrivable drives), zero drift.
  Smooth continuous-ramp closes reach the can in 0.35–0.45 s at 0.8/s with the same contact map in 8/8 runs; the map
  doubles as a grasp check. Details and the demo recipe in `docs/robot/README.md`. **Sim consequence:** the grasp to
  model is palm contact + opposed thumb + fingers at 20–30 % flexion with a stiff position hold, not a deep torque-
  limited wrap; the can must be brought against the palm by the arm (or dragged in by the fingers) before closing.
* **Sim holds the can (23:00).** `g1_walk_grasp.py --finger-mode stall` (now default) ports the bench flow; with the
  arm placement fixed (thumb up on approach, arrive 3 cm high and descend, palm 3 mm off the can, palm centre 55 mm)
  the fixed-base G1 lifts the can 14 cm on both hands. Placement tolerance measured: height ±15 mm and +20 mm along
  the fingers are free; approach depth must be within ~10 mm — that is the number the arm controller has to hit, and
  the finger contact map tells us when it did not. Next: the same with the balance policy running (floating base),
  the can in a basket at chest height instead of a table, and the release/hand-over.

## Simulation tracks (this laptop)

* `sim/revo2_hand_grasp.py`: hand-only grasp physics. Use it to fix the approach pose (bottle centre ≈ 35 mm past
  the knuckle line, surface on the palm), finger/thumb close fractions, and to test 500 mL vs 20 oz. Next:
  friction sweep (0.5–1.2), bottle mass 0.4–0.7 kg, thumb-aux sweep, left hand.
* Combined robot model: Unitree `g1_29dof_rev_1_0.urdf` with the `*_rubber_hand` links replaced by the Revo 2
  URDFs at the wrist adapter transform (BrainCo uses an end-effector target 50 mm along +x from the wrist yaw
  joint; the exact adapter offset must be measured from the physical adapter). Import via Isaac Lab
  `UrdfConverter` as one articulation. This is what the arm approach and later teleop/data collection run on.
* Teleop (Quest 2): only after the combined model exists. `xr_teleoperate --ee brainco --sim` expects Unitree's
  sim DDS topics; the sim side for BrainCo hands has to be added (there is no `brainco` task in
  `unitree_sim_isaaclab`). Worth it only if we decide to collect demonstrations; scripted grasping does not need it.

## Decisions taken

* No new dexterous grasp policy training on the critical path. Scripted approach + IK + closed-loop close, with
  learning only for a specific failure we can name.
* One hand (right) and one bottle size first. Baskets get rigid inserts and one designated pick slot.
* Physics on CPU; the RTX 5060 is for rendering only.

## Answered (2026-09-09 evening)

1. Hands are **Revo 2 Pro and Touch** (12–64 V). Touch gives 9 pressure points per finger → handover signal.
2. **No power adapter yet.** The Pro/Touch box has the power cable; BrainCo's adapter is 24 V with an XT30 plug, so any
   24 V bench supply + XT30 pigtail works (confirm polarity on the cable). Also needed: a USB-RS485 dongle (BrainCo's
   dual-port kit, or a generic CH340/FTDI one on the supplied 485 cable). Bench test blocked until both are in hand.
3. ~~Demo bottle: standard US 20 oz Pepsi~~ **Superseded 2026-09-10 21:45: the demo serves 12 oz cans** (Pepsi blue,
   Diet Pepsi silver; 122 mm tall, 66.2 mm body, 54 mm top seam, ~0.38 kg full). Modelled as `pepsi-12oz-can`
   (`assets/bottles/`), now the default object in both sim scripts, grasp height 60 mm (mid-body). Zack's
   `roundtable_sim` already used exactly this can. Consequences: fits the 100 mm opening easily and is 40 % lighter
   than the bottle, but it is short, smooth aluminium, and tips or pops out of a shallow insert when pushed.

## Still open

4. Robot arrival estimate and whether it ships with the BrainCo wrist adapters already fitted.
5. Room: floor type, distance between guests, expected noise (affects following and speech gates).
