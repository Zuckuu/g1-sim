# PepsiDemo — Unitree G1 + BrainCo Revo 2 bottle-serving demo

Target: in ~2 weeks, a G1 humanoid with BrainCo Revo 2 dexterous hands follows two basket carriers through a
room of PepsiCo executives, asks each guest "Pepsi or Diet Pepsi?", picks the right bottle out of the basket
and hands it over, until someone says "Pepsi delivered". Robot arrival date unknown; the two Revo 2 hands are in
hand now; everything else starts in simulation on this laptop (RTX 5060 Laptop, 8 GB VRAM).

Hardware pivot (2026-09-09): the hands are **BrainCo Revo 2**, not Unitree Dex3. Dex3 datasets, joint mappings
and Unitree's Dex3 sim tasks do not transfer. Revo 2 facts that drive the design:

| Revo 2 | Value | Consequence |
| --- | --- | --- |
| Actuation | 6 motors / 6 active joints / 11 DoF (thumb flex + thumb rotation, one flex motor per finger, coupled distal joints) | Power grasp, not fine in-hand manipulation |
| Grip | ≥50 N power grasp, ≥15 N pinch, ≥20 kg static load | A full 500 mL/20 oz bottle (0.55–0.62 kg) is well within limits |
| Opening | 100 mm thumb-to-index max | 500 mL (~65 mm) and 20 oz (~73 mm) bottles fit; a 2 L (~110 mm) does not |
| Speed | open/close ≤0.65 s, 0.1° repeatability | Fast enough to grasp and release in a conversation-paced loop |
| Interface | RS-485 (Modbus RTU), CAN FD, EtherCAT; 12–28 V (Basic) / 12–64 V (Pro, Touch) | On the G1: dual RS-485 via Unitree's USB-485 board; on the bench: USB-RS485 adapter |
| Touch variant | pressure / friction / proximity per fingertip | If ours are `XT*` (Touch), we get a direct "bottle is in the hand" signal |

Sources: [Revo 2 parameters](https://www.brainco-hz.com/docs/revolimb-hand/en/revo2/parameters.html),
[product manual](https://brainco-common-public.oss-cn-hangzhou.aliyuncs.com/web-config/docs-sdk/Revo-2-Product-Manual-V2.1EN.pdf).

## Layout

```
README.md            this file: status, how to run, findings
docs/plan.md         the two-week plan, gates and open questions (Revo 2 edition)
docs/g1-tonight.md   earlier runbook from the Dex3 era (kept for the Unitree sim/XR notes)
docs/g1-sources.json pinned upstream revisions and dataset metadata
sim/                 Isaac Sim 5.0 / Isaac Lab 2.2 scripts (run with sim/run.sh <script> [args])
  revo2_hand_grasp.py   fixed Revo 2 hand + bottle grasp test: placement sweeps, videos, joint logs, side/top modes
  build_g1_revo2_urdf.py merges Unitree G1 29-DoF URDF + BrainCo Revo 2 URDFs into one robot (mount frames from FK)
  g1_walk_grasp.py      whole robot: Unitree locomotion policy (walk), arm IK, grasp/lift state machine, videos
  make_bottle_mesh.py   dimensioned PET bottle meshes (500 mL, 20 oz)
  g1-scene.py           fixed-base G1 29-DoF (Unitree Dex3 USD, body stand-in) load/physics check
  pepsidemo_sim.py      shared helpers (trimmed Isaac Lab experience files, paths)
  start-isaac-smoke.sh  the original environment bootstrap (already executed; kept as the install recipe)
hand/                physical Revo 2 bench test over USB-RS485 (hand/README.md, revo2_bench.py)
work/                (git-ignored, ~22 GB) runtime + vendored sources:
  g1-runtime/isaac50    Python 3.11 venv: isaacsim 5.0.0, isaaclab 0.44.9 (editable), torch 2.7.0+cu128
  g1-runtime/downloads  unitree-assets.zip (1.3 GB, sha256-verified) ; g1-runtime/usd  converted USDs ; g1-runtime/logs
  hand-venv             Python 3.12 venv with bc-stark-sdk 2.0.5 for the real hand
  IsaacLab              v2.2.0 sparse checkout (source/isaaclab + apps)
  unitree_sim_isaaclab  Unitree's Isaac Lab sim (assets/robots extracted: G1 29-DoF dex1/dex3/inspire, fixed and whole-body)
  xr_teleoperate        Unitree XR teleop (has --ee brainco support and the BrainCo retargeting config)
  unitree_ros           robots/g1_description (G1 URDF + meshes, for a combined G1+Revo2 model)
  brainco-description   BrainCo official Revo 2 URDF/MJCF/USD + meshes (revo2_system)
  brainco-hand-sdk      BrainCo official Python SDK examples (revo2/)
  unitree-g1-brainco-hand  BrainCo's G1 integration tutorial/repo (ROS 2, dual RS-485, arm IK)
```

## Running the simulation

The Isaac stack lives in `work/g1-runtime/isaac50`; `sim/run.sh` wraps it and logs to `work/g1-runtime/logs`.
Needs a shell with GPU + display access (the normal desktop terminal works).

```bash
# Revo 2 hand closes on the 500 mL Pepsi bottle model; sweep placements in one process, snapshot the best one
# (--rendering_mode balanced gives clean images; the default "performance" is noisy but lighter on VRAM)
sim/run.sh revo2_hand_grasp.py --headless --snapshot --rendering_mode balanced \
    --sweep-gap 0.028,0.032,0.036 --sweep-distal 0.030,0.035,0.040 --sweep-vertical=-0.02,0,0.02

# same, watch it live (window stays open after the test)
sim/run.sh revo2_hand_grasp.py --gap 0.032 --distal-offset 0.040

# 20 oz bottle, plain cylinder proxy, or a scan of the real bottle (OBJ/STL, metres, origin at base centre, +Z up)
sim/run.sh revo2_hand_grasp.py --headless --bottle pepsi-20oz --sweep-gap 0.030,0.034,0.038
sim/run.sh revo2_hand_grasp.py --headless --bottle cylinder --bottle-radius 0.0365 --bottle-mass 0.62
sim/run.sh revo2_hand_grasp.py --headless --bottle-mesh /path/to/real_bottle_scan.obj --bottle-mass 0.53

# regenerate / customise the bottle meshes (dimensions, waist, mass)
work/g1-runtime/isaac50/bin/python sim/make_bottle_mesh.py pepsi-500ml
work/g1-runtime/isaac50/bin/python sim/make_bottle_mesh.py custom --height 0.225 --diameter 0.066 --waist 0.9 --mass 0.53 --name measured

# G1 body stand-in (Unitree Dex3 USD) loads and simulates
sim/run.sh g1-scene.py --headless --max-steps 300
```

Platform notes learned the hard way (all handled in the scripts):

* PhysX GPU kernels do not load on this RTX 5060 / driver 580 machine (`Could not find CUDA module ...`), so
  physics runs on the CPU (`--device cpu` default). Rendering still uses the GPU.
* Isaac Lab's stock experience files require the `isaaclab_tasks/rl/mimic/assets` extensions, which the sparse
  checkout does not have; `pepsidemo_sim.py` generates trimmed copies next to them.
* Do not set `builtins.ISAAC_LAUNCHED_FROM_TERMINAL`; it disables physics-context creation.
* Kit can hang on shutdown and keep ~3 GB VRAM; the scripts force-exit after a 20 s watchdog. If a run dies,
  check `nvidia-smi` for stale `isaac50/bin/python` processes before starting another.

## Findings so far (simulation, 2026-09-09)

* BrainCo's official `revo2_right.urdf` imports cleanly (11 revolute joints: 6 active + 5 coupled distal,
  limits 0–1.41 rad fingers, 0–1.57 rad thumb rotation, 0–1.03 rad thumb flex). Convex-decomposition colliders
  are required: the palm's convex hull swallows the grasp pocket and produces false "holds" by interpenetration.
* Hand geometry (from the collision mesh, hand base frame): palm face is 20 mm in front of the base-frame origin
  and ~23 mm in front of the knuckle axes; proximal phalanx 32 mm, distal ~43 mm, total finger reach ~70 mm.
* Placement sweeps with a 65 mm / 0.55 kg bottle proxy (thumb pre-shaped into opposition, then the bottle
  enters the pocket, then fingers + thumb close in 0.7 s, then the table drops 15 cm and we hold 3 s):
  27 placements → 8 held. The robust block is **bottle centre 35 mm past the knuckle line toward the fingertips,
  bottle surface 30–34 mm in front of the knuckle axes (resting on the palm face)** — 6/6 held there across
  ±20 mm vertical error, sag ≤0.5 cm, tilt 10–17°. Surface at 26 mm (pressed into the palm) fails every time;
  centres at 25 mm or 45 mm along the fingers mostly fail (pushed out / beyond the curl). Tolerance along the
  finger direction is therefore only about ±5 mm, along the palm normal about ±2 mm, vertically ≥±20 mm: the
  arm approach must control depth and forward offset tightly, height is forgiving. Reports:
  `docs/revo2-grasp-sweep-500ml.json` (27 trials), `docs/revo2-grasp-best-500ml.json` (the held case below).

| pre-shaped, bottle entering | closed | table dropped 15 cm, 3 s later |
| --- | --- | --- |
| ![open](docs/img/revo2-grasp-open.png) | ![closed](docs/img/revo2-grasp-closed.png) | ![held](docs/img/revo2-grasp-after-lower.png) |

* **Bottle model.** `sim/make_bottle_mesh.py` builds a revolved PET bottle to the public dimensions
  (500 mL: 231 mm × 66 mm, 0.525 kg full, 60.7 mm grip waist; 20 oz: 222 mm × 72.8 mm, 0.64 kg, 65.5 mm waist),
  with petaloid base ring, waist, shoulder, 28 mm neck and cap; profiles in `docs/bottle-*-profile.json`. It is
  dimensionally the real bottle, not its trademarked styling. A phone photogrammetry scan of the actual demo bottle
  can replace it via `--bottle-mesh`. The grasp is taken 100 mm above the base (label/waist zone).
* **500 mL bottle mesh, 9 placements → 6 held** (`docs/revo2-grasp-sweep-pepsi500-mesh.json`). Best: axis 40 mm
  past the knuckle line, waist surface ~35 mm in front of the knuckle axes: 0 cm sag, 14° tilt. The held block is
  the same as for the cylinder but shifted ~5 mm toward the fingertips because the waist is 2.6 mm narrower than the
  proxy was. (Gap labels in that JSON are 2.6 mm smaller than the true surface gap; fixed in the script since.)

| 500 mL bottle model, closed | table dropped, 3 s later |
| --- | --- |
| ![closed](docs/img/revo2-pepsi500-closed.png) | ![held](docs/img/revo2-pepsi500-after-lower.png) |

* **20 oz bottle mesh (72.8 mm, 0.64 kg), 9 placements → 6 held** (`docs/revo2-grasp-sweep-pepsi20oz-mesh.json`):
  held for surface gaps 30–34 mm at 35–40 mm forward, plus two edge cases; fails at 38 mm gap with ≥40 mm forward
  (beyond the curl). So both demo-candidate bottles are graspable with the same approach pose to within a few mm.
  ![20oz held](docs/img/revo2-pepsi20oz-after-lower.png)

### Finger coupling correction and the 20 oz failure analysis (`--coupling linkage`, now the default)

The runs above drove each distal joint to its own target, which let fingertips claw around the bottle while the
proximal joint was blocked at 0 rad (the joint logs showed index/middle/ring stopping within 0.05 rad of open).
The real Revo 2 finger is a rigid 4-bar linkage: the distal angle is 1.155 × the proximal angle, full stop. With
that coupling enforced (distal target tracks the *measured* proximal), the 20 oz sweep gives **4/9 held**
(`work/g1-runtime/logs/revo2-right-20260909-230417.json`), the fingers now wrap at ~0.5 rad proximal flexion and
the bottle tilts only 6–9° instead of 15°. What the per-step logs (`*-joints.csv`) show about the failures:

| placement (surface gap / forward) | what happens | outcome |
| --- | --- | --- |
| 30–34 mm / 35–40 mm | fingers touch at ~0.4 rad, drag the bottle 18 mm away from the palm and 24 mm toward the thumb, the thumb (pushed to its open limit) stops it, tilt settles at 8° | **held**, 0 cm sag |
| 38 mm / 40 mm (4 mm too far from the palm) | fingers touch lower on the bottle, drag it 25 mm out and 42 mm toward the thumb, then close underneath it; the bottle rides up 37 mm on the closing fingers and tips over the thumb before the table even drops | dropped |
| 30 mm / 45 mm (5 mm too far toward the fingertips) | fingers sweep the bottle 57 mm along the palm toward the wrist; it ends pinched against the thumb at 26° tilt and falls when the support goes | dropped |
| 38 mm / 45 mm | fingers close completely without touching; the bottle is outside the reach of the curl and is knocked over by the fingertips | dropped |

Servo tuning does not rescue bad placement: a stiffer, faster close with a stronger thumb (20 Nm/rad, thumb 90 %)
drops to 2/9 — it flicks the bottle out; a slower, softer close (4 Nm/rad, 1.2 s) stays at 4/9 with a slightly
different held set. Conclusions for the robot: (1) the approach must end with the bottle **against the palm**
(surface 7–11 mm in front of the palm face; ±3 mm) and the bottle axis **35–40 mm toward the fingertips** (±3–5 mm)
— i.e. approach until palm contact, not to an open-loop pose; (2) close with the current-limited, compliant mode
the hand offers rather than a hard position slam; (3) the thumb is the backstop and reaches its joint limit under
load, so thumb-side placement error is the dangerous direction. Height error of ±20 mm is harmless.

Videos of one held and three failed placements (over-the-shoulder view, 30 fps): `docs/video/`.

### CORRECTION (2026-09-10, 01:00): the hand-only "held" results above were an import artifact

BrainCo's URDF has empty fingertip frames (`*_tip_link`, no inertial). The Isaac URDF importer gives such links
PhysX's default **1.0 kg** mass — five fake kilograms per hand, one on each fingertip. Those sluggish, heavy fingers
pressed the bottle gently and every "held" case above depended on it. With the fingertips folded into the distal
links (real ~10 g fingers; `sim/build_g1_revo2_urdf.py::fold_fixed_children`, now also the default in
`revo2_hand_grasp.py`), the same placements go **0/18** for the side grasp on a free-standing 20 oz bottle and
**0/9** for a first top-down attempt. The physics is simple: a free-standing bottle tips at ~2 N of side push at
10 cm height (m·g·r ≈ 0.2 N·m), fingers curling toward the palm normal push the bottle *away from the palm* before
they can hook it, and a finger at 0.6–1.5 N·m delivers 10–30 N at the tip. Sim-only lesson so far, but it is the
right physics and it changes the grasp design:

* the approach has to end with the **palm already pressing the bottle** (or the bottle constrained by the basket
  insert), so that the finger reaction has nowhere to go;
* the fingers must be **current-limited** so they stall on the object instead of shoving it (the real Revo 2 does
  this; `--finger-effort 0.4` reproduces the stall in sim, fingers stop at ~0.6 rad on the bottle);
* the thumb belongs on the far side of the bottle before the fingers move.

## G1 + Revo 2 whole-robot pipeline (`sim/g1_walk_grasp.py`)

![model](docs/img/g1-revo2-model.png)

* **Model**: `sim/build_g1_revo2_urdf.py` merges Unitree's `g1_29dof_rev_1_0.urdf` with BrainCo's `revo2_{left,right}.urdf`
  (fingers along the forearm, palms toward the midline, thumbs up; 12 mm adapter offset — to be measured on the real
  adapter). 51 joints (29 + 2×11), 54 bodies, 33.8 kg. Imported with fixed joints merged for the body but the hand's
  tip/touch links folded by us (the importer's own merge corrupts finger geometry, and un-merged massless links make
  the 34 kg articulation unstable).
* **Walking**: Unitree's simulation-only locomotion policy (`unitree_sim_isaaclab/assets/model/policy.onnx`, 910-d
  history obs → 12 leg targets at 50 Hz, Unitree's whole-body gains and standing pose) runs on this model unchanged:
  stable stand for 40 s, walks at 0.5–0.65 m/s, stops on command. Commands below ~0.5 m/s are a standing deadband.
  Video `docs/video/g1-revo2-stand-walk.mp4`.
  ![walk](docs/img/g1-walk-frames.png)
* **Arm**: differential IK (damped least squares) on the 7 right-arm joints to a palm-centre target, orientation from
  the true palm triad (the hand-base axes are 18° off the palm normal), integrating on the commanded target with a
  0.3 gain (the PD arm sags under gravity; re-basing on measured angles leaves a permanent 40 mm error), anti-windup,
  raise → out → in motion plan (the hands hang 17 cm below the table top). Reaches the grasp pose to **0.2 mm / 0.1°**.
  Jacobian and frames verified by finite differences (`--phase ikcheck`).
* **Full chain video** (`docs/video/g1-revo2-walk-reach-grasp-full-chain.mp4`, 24 s): stand → walk 1.2 m with lateral/heading
  correction → stop 0.38 m from the bottle → raise → pre-grasp → palm-contact approach (bottle in a basket insert) →
  current-limited close (fingers stall on the bottle at ~0.33 rad) → lift. The bottle comes out of the insert in the hand
  and is carried up for ~1.5 s, then slips out and falls back to the table (bottle axis 10 mm past the knuckle line,
  `--press 0.003 --finger-effort 0.4 --holder 0.06 --arm-time-scale 1.6`). Report `docs/g1-full-chain.json`.
  ![full chain](docs/img/g1-full-chain-frames.png)
  ![grasp](docs/img/g1-full-chain-grasp-zoom.png)
* **Grasp + lift**: partial (above) — the wrap is real but the hold does not survive the lift yet; see the correction above. Attempts on record: free-standing bottle (pushed over),
  bottle in a 60 mm insert with default fingers (reaction pushes the compliant arm 20 mm back; Unitree's wrist effort
  limit is 5 N·m), insert + gentle fingers (fingers stall correctly on the bottle, partial wrap, slips on lift).
  Videos in `docs/video/g1-revo2-reach-grasp-attempt-*.mp4`.
  ![grasp attempt](docs/img/g1-grasp-attempt-frames.png)

```bash
sim/run.sh g1_walk_grasp.py --headless --video --phase walk                       # stand + walk to the table
sim/run.sh g1_walk_grasp.py --headless --video --phase grasp --fixed-base --holder 0.06 --finger-effort 0.4
sim/run.sh g1_walk_grasp.py --headless --video --phase all --holder 0.06 --finger-effort 0.4   # full chain
sim/run.sh g1_walk_grasp.py --headless --phase ikcheck --fixed-base                # jacobian / frame diagnostics
```

These are development metrics on a proxy object with guessed friction; they say nothing yet about the real hand.
The bench test in `hand/` is what turns them into evidence.
