# G1 bottle grasping: tonight's starting point (Dex3 era — superseded)

> Written before we learned the hands are BrainCo Revo 2, not Dex3. The Unitree sim/XR notes below still apply;
> the Dex3 dataset and joint-mapping parts do not. Current status and plan: `../README.md`, `plan.md`.

Working target: G1 29-DoF body with Dex3-1 hands. Nick believes his headset is a Meta Quest 2. Local compute is an RTX 5060 Laptop with 8 GB VRAM and a working host driver. Robot arrival is unknown. All initial work is simulation-only.

Quest 2 compatibility is provisional: it supports the WebXR hand-tracking route, but the inspected Unitree device list explicitly names Quest 3/3S. Verify both hands in the [Immersive Web hand-tracking sample](https://immersive-web.github.io/webxr-samples/immersive-hands.html) using the headset's own browser, then test Unitree's XR bridge. A successful browser sample does not establish full bridge compatibility. [NVIDIA's WebXR client documentation](https://docs.omniverse.nvidia.com/xr/omniverse-spatial-docs/latest/clients/meta/03-connect.html) explicitly includes Quest 2 in its hand-tracking configuration guidance.

Tonight's target is a recorded, replayable, teleoperated grasp in simulation. A first bottle-proxy grasp is the stretch target after the stock cylinder example works. Neither is an autonomous policy or evidence of physical reliability.

## Reusable work found

1. **Official bottle demonstrations:** [Unitree G1_Dex3_PickBottle_Dataset](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_PickBottle_Dataset). The actual `meta/info.json` reports 202 episodes, 176,774 frames at 30 Hz, 28-dimensional state/action vectors, and two 640×480 camera streams. The dataset card declares Apache-2.0. This is demonstration data, not a pretrained policy. Joint order includes 14 arm and 14 hand entries; left and right finger ordering differs. Actual metadata is LeRobot v3.0; the README still embeds a v2.1 example. Use the actual metadata and episode index, not the README's old file paths. Camera placement, bottle geometry, and action interpretation require inspection before reuse.
2. **Official simulation and XR control:** [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) and [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate). Source inspected locally. The task `Isaac-PickPlace-Cylinder-G129-Dex3-Joint` fixes the robot base. The XR stack supports simulation control and recording; its device list names Quest 3 and Quest 3S. Confirm Nick's exact headset against the [XR device guide](https://github.com/unitreerobotics/xr_teleoperate/wiki/XR_Device).
3. **Physical bottle-grasp evidence:** [A Rapid Deployment Pipeline for Autonomous Humanoid Grasping](https://linqi-ye.github.io/docs/g1grasp.pdf) reports G1 + Dex3 drink-bottle grasps at five table positions using perception, IK, and staged motions. Its bottle is 22 cm high and 6 cm in diameter. A downloadable implementation was not located in the checked paper/author page. Treat it as engineering reference, not an available controller.
4. **Downloadable checkpoint candidate:** [SII-Linzy G1 SONIC bottle checkpoint](https://huggingface.co/SII-Linzy/groot-g1-sonic-grab-bottle-deploy-checkpoint-10000). Model card and file listing inspected, weights not downloaded. The author reports 5/10 single-bottle simulation successes and no real deployment. It emits SONIC motion tokens plus hand joints, so it is not a direct replacement for the stock joint-control task. Keep as a secondary research lead.

## Sequence for tonight

### 1. Establish usable compute

Observed in this session: Ubuntu 24.04, roughly 32 GB RAM, 247 GB free disk, and an RTX 5060 Laptop GPU identified through `/proc/driver/nvidia`. `nvidia-smi` fails here and NVIDIA device nodes are not visible. A kernel driver is present. This does not establish that the host driver is broken; session device access may be the issue. Run `nvidia-smi` in the normal host terminal to distinguish those cases before changing drivers.

Nick subsequently verified that the normal host terminal runs `nvidia-smi` successfully: RTX 5060 Laptop, 8,151 MiB VRAM, driver 580.159.03. The GPU access failure is specific to this execution environment; no driver repair is indicated. Eight GB is below the published Isaac Sim 5.0 minimum. Start with an empty simulator smoke test, then attempt a reduced single-G1 scene if that passes.

No working Isaac runtime was established. The system Python package check and checked project paths did not identify an existing installation. `start-isaac-smoke.sh` prepares an isolated Python 3.11 / PyTorch 2.7.0 cu128 / Isaac Sim 5.0.0 environment under this task's `work/g1-runtime` directory, tests CUDA computation, then opens an empty viewport with `isaac-smoke.py`. Run it from the normal Ubuntu desktop terminal. It does not yet install Isaac Lab, Unitree assets, or the XR environment. Its shell syntax and Python syntax were checked; installation and graphics execution remain untested here. [Isaac Lab dependency reference](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/isaaclab_pip_installation.html).

Run simulation on an RTX workstation, with the Jetson reserved for subsequent robot deployment. For reference, [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) list 32 GB RAM and 16 GB VRAM minimum, and exclude GPUs without RT cores such as A100/H100. A smaller laptop GPU may handle reduced scenes, but must be smoke-tested. Choose a matched Isaac Sim / Isaac Lab / CUDA / PyTorch combination after confirming the compute host; the Unitree README specifies Isaac Sim 5.x for RTX 50-series GPUs. Do not run its automatic installer blindly: it uses system package installation and does not pin Isaac Lab for every version.

### 2. Reproduce the unmodified cylinder scene

Complete source checkouts, initialize their submodules, download the required Unitree USD assets, and install the supported simulation and teleoperation environments. Source checkout alone does not include these dependencies.

The upstream simulation launch command, **after installation**, from the simulation repository is:

```bash
python sim_main.py --device cpu --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --enable_dex3_dds --robot_type g129
```

`--device cpu` is the upstream example's physics/tensor choice; it does not remove Isaac's graphics requirements. Verify the robot, object, camera service, reset behavior, and joint state publication before connecting XR.

### 3. Connect the Quest and record one episode

Use the upstream certificate/WebXR setup. Quest and the teleoperation host need network connectivity; establish HTTPS trust, then enter the XR session in the headset browser. The simulator's camera service must be reachable by the teleoperation process. On a single machine, the example below uses its loopback address; remote compute needs an explicitly reachable host address and corresponding network configuration.

From `xr_teleoperate/teleop`, **after installation**:

```bash
python teleop_hand_and_arm.py \
  --input-mode=hand --arm=G1_29 --ee=dex3 \
  --img-server-ip=127.0.0.1 --sim --record
```

The inspected code selects DDS domain 1 for simulation; physical mode selects domain 0. Keep `--sim`. Use the same simulation domain on both sides. Start control with `r`; toggle episode recording with `s`. Check upstream instructions for the exact XR connection steps and camera configuration. Align the virtual hands before starting control.

Minimum milestone: move the simulated arm and fingers, grasp/lift/place the stock cylinder, save an episode, and replay it. Record failure attempts too, with separate outcome labels. Manually verify that replay includes the object and hand motion expected; joint playback alone is not proof of a successful physical grasp.

### 4. Replace the object and measure the grasp

The inspected stock scene uses a cylinder with radius 0.018 m, height 0.35 m, mass 0.4 kg, static/dynamic friction 1.5, and friction combine mode `max`. It is not a representative Pepsi bottle. First preserve that baseline; then add a separate bottle-proxy task.

Measure the intended bottle's diameter at the grasp, height, and filled mass when available. Until then, mark every proxy dimension and material value as an assumption. Keep its bottom on the support surface when changing its height. Verify contact material combination on both the fingers and object; changing only object friction may not reduce the effective friction under `max`.

Initial proposed evaluation: lift the bottle 10 cm clear of its starting support, hold for 5 seconds without support contact or dropping, then place it down. Record bottle/hand poses, joint targets/states, outcome, seed, geometry, friction, and mass. Use simulation truth for scoring and initial debugging, then explicitly replace privileged pose inputs with perception when building autonomy. Test nearby starting positions and multiple friction values. These are development criteria, not a real-world safety qualification.

The stock termination named `success` is an out-of-workspace reset check, not a bottle lift-and-hold success detector. Implement a separate task-specific metric rather than reporting that flag as grasp success.

### 5. Use existing data to accelerate autonomy

Inspect representative official bottle episodes, including video, alongside their hand and arm trajectories. Reuse grasp candidates only after verifying joint names, units, reference frames, camera setup, and starting pose. Do not assume the LeRobot dataset can be fed directly into Unitree's JSON replay path. Collect initial matched sim demonstrations with Quest before choosing scripted grasp execution or imitation learning. A pretrained network is optional for the first repeatable grasp.

## Preparation completed

- Sparse source checkouts inspected at `work/unitree_sim_isaaclab` and `work/xr_teleoperate` in this task workspace.
- Upstream launch flags, hand/body target, recording controls, DDS domain selection, object configuration, and reset semantics checked in source.
- Official dataset metadata and candidate checkpoint model card downloaded to `work/research`.
- Source revisions saved in the adjacent `g1-sources.json`.

Simulation, headset connection, replay, and any grasp have **not** been run or validated in this session. Next dependency is a usable GPU host and the exact Quest model.
