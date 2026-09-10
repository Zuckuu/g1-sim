# Round-table drink service — G1 in MuJoCo (CPU only)

A Unitree G1 (29 DoF + Dex3 three-finger hands) serves ten guests seated around a
round table. It walks up to each guest, asks *"Pepsi or Diet Pepsi?"*, walks to a
drink station, grabs the requested can with its right hand, carries it back and
sets it down on the coaster next to that guest — ten times — then says goodbye.

This runs **headless on a CPU-only machine** (no Isaac Sim, no NVIDIA GPU needed)
and writes MP4 videos + screenshots, so you can review the behaviour without
running anything heavy on your own PC. It also runs interactively on a laptop.

| | |
|---|---|
| Physics / renderer | [MuJoCo](https://mujoco.org) 3.x, software OpenGL (EGL/OSMesa/GLFW) |
| Robot model | `unitree_g1/g1_with_hands.xml` from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (BSD-3, downloaded on first run) |
| Control | kinematic "puppet": scripted base path, footstep gait + leg IK, arm IK, finger poses |
| Cans | real rigid bodies — they are simulated, stand on the table, and fall if dropped |

## Quick start (Linux / macOS / Windows, no GPU)

```bash
cd roundtable_sim
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py --out out/run                        # ~1 min sim + ~1 h of CPU rendering for all videos
```

Faster iterations:

```bash
python run.py --out out/quick --guests 2 --videos firstguest   # one short video
python run.py --out out/run --skip-sim --videos overview        # re-render an existing recording
python run.py --out out/run --videos none                       # simulation + screenshots only
python check_run.py out/run                                     # sanity-check a recording (see below)
python run.py --view                                            # live MuJoCo viewer (needs a display)
```

`check_run.py` replays a recording and fails if the puppet ever "breaks": joint jumps
between frames, knees bent the wrong way, over-extended legs, feet under the floor, or a
can that did not end up upright on its coaster. Run it after changing the layout or the
controller before spending an hour on rendering.

On a machine without a display set `MUJOCO_GL=egl` (default here) or `MUJOCO_GL=osmesa`.
`ffmpeg` must be on the PATH for video encoding (`imageio-ffmpeg` ships one).

### Outputs (`--out` directory)

| File | What |
|---|---|
| `g1_roundtable_service_2x.mp4` | director's cut of the entire run at 2x speed: follow cam, "ask" cam, over-the-shoulder grasp/place close-ups, overview picture-in-picture, dialog bubbles, HUD |
| `g1_roundtable_first_guest_1x.mp4` | the first guest end-to-end in real time |
| `g1_roundtable_overview_4x.mp4` | fixed overview camera, 4x time-lapse of the whole service |
| `screenshots/*.png` | scene overview, annotated top-down layout, asking, grasp close-up, hand detail, placing, final state |
| `trajectory.npz`, `meta.json` | the recorded run (qpos at 30 Hz + phase/dialog metadata) — re-render with any camera without re-simulating |
| `scenario_used.json` | the guests/layout used; copy, edit and pass with `--config` |

## Changing the scenario

Everything is data driven from `config.py`:

* **Guests** — names, orders (`Pepsi` / `Diet Pepsi`), shirt/skin/hair colours. Edit
  `DEFAULT_GUESTS` or pass a JSON with `--config` (see `scenario_used.json` for the format).
  The table always seats `len(guests)`; `--guests N` serves only the first N of them.
* **Layout** — table radius/height, chair radius, coaster position (to the guest's right),
  where the robot stands to ask/serve, the "ring road" radius it walks on, the drink
  station position/height, can size, walking/turning speed, pelvis height, physics timestep.
* **Dialogue and choreography** — `scenario.py::RoundTableScenario.script()` is a plain
  generator; each `yield from` is one action (`go_to`, `say`, `arm_move`, `hand_close`, …).

## How it works

```
config.py        guests + layout numbers (single source of truth)
scene.py         builds the MJCF: room, table, chairs, stylised guests, name cards, coasters,
                 station, 12 cans (free bodies), cameras, generated label textures; includes the G1
g1_kinematics.py G1Model (FK/jacobians), solve_ik (damped least squares + nullspace),
                 FootstepGait (plants/swings feet under a moving base), ArmController,
                 HandController, G1Puppet (assembles one qpos per tick)
scenario.py      the story: navigation via the ring road, ask/fetch/grasp/carry/place per guest,
                 can attach/detach, trajectory + event recording
render.py        offline renderer: Director (camera per phase), Pillow overlays, parallel chunked
                 encoding, screenshots, annotated top-down
run.py           CLI
fetch_assets.py  sparse-clones the Menagerie G1 model into assets/ (gitignored, ~36 MB)
```

The robot is **puppeteered**: every 4 ms tick the controller writes the full joint
configuration (floating base + 43 joints) and MuJoCo only simulates the cans. That
trades physically-simulated balance for a deterministic demo that never falls over
and runs 10x faster than real time on 4 CPU cores. Concretely:

* **Locomotion** — the base follows a precomputed path (corners rounded, speed limited by
  acceleration *and* path curvature so it slows down in corners, heading = path tangent).
  A footstep planner keeps both feet planted and swings one foot at a time to where its
  nominal position *will* be shortly after touchdown, so the feet never slide. Each foot pose
  is solved with 6-DoF leg IK (targets clamped to the leg's reach, knee never straight, and a
  restart from the rest pose if a solution goes bad). Turning in place steps as well.
* **Reaching** — 7-DoF arm IK plus waist pitch/yaw (down-weighted so the torso only leans
  when the arm alone can't reach). Cartesian moves interpolate the hand pose from where it is
  to the goal (S-curve), so the hand travels in straight lines.
* **Grasp** — the Dex3 fingers close to a pose tuned around a 66 mm can (index/middle wrap the
  front, thumb closes the side). Because the finger geometry is only visual here, the can is
  attached to the hand frame once the fingers are closed and released again on the coaster
  (3 mm drop, then physics takes over). Hand/robot geoms have collisions disabled.
* **Dialogue** — text bubbles in the video, driven by `say()` events in the script.

## Relation to the Isaac Sim setup in this repo

The Isaac Lab task in the top-level README needs an RTX GPU and cannot run on a CPU-only
cloud box, which is why this scenario was built in MuJoCo. The scenario logic
(`config.py`, the `script()` generator, the footstep/IK approach) is simulator agnostic,
and the joint names match Unitree's URDF/USD (`left_hip_pitch_joint`, `right_wrist_yaw_joint`,
…), so the same choreography can be ported to the Isaac Lab environment later — e.g. by
replacing `G1Puppet` with the RL locomotion policy plus the existing scripted grasp.

## Known limitations / next steps

* Walking and balance are scripted, not physically simulated (no RL policy yet). The gait is
  a brisk, slightly crouched walk (pelvis at 0.74 m) that keeps the legs away from singularities;
  in-place turns are deliberately slow (0.7 rad/s) so they read as steps rather than a spin,
  especially in the 4x time-lapse.
* The grasp is a visual/kinematic attach, not friction-based; the finger meshes overlap the
  can by a few millimetres in the closed pose.
* Guests are stylised primitive-geometry people; the scene is a simple room.
* Software rendering is slow (~0.3 s per 720p frame with shadows); use `--no-shadows`
  or `--videos overview` for quick looks.
