# g1-sim

Isaac Sim pick-and-place for a **Unitree G1** with **BrainCo Revo2** (or Dex3) hands. Click the Pepsi-sized can, click the table, and the scripted controller tries to pick and place it.

Based on [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab). This repo adds the BrainCo Revo2 task, a click-to-pick demo, and a Windows launch script.

> **CPU-only alternative:** [`roundtable_sim/`](roundtable_sim/README.md) contains a MuJoCo version of
> the round-table drink-service scenario (G1 asks 10 seated guests "Pepsi or Diet Pepsi?", fetches
> the can and places it on their coaster). It runs headless without a GPU and renders videos and
> screenshots, so it can run in the cloud instead of on your PC.

## Requirements

- Windows 10/11
- **NVIDIA GPU** (this machine uses an RTX 3060 12 GB — keep only **one** Isaac window open)
- **Python 3.11** (3.12 will break Isaac Lab)
- Isaac Sim **5.1** + Isaac Lab (installed into a venv, not system Python)
- Disk space for Isaac Sim, Isaac Lab, and the Unitree USD assets

Do **not** run `isaaclab.bat` if the project path contains a space (for example `C:\Users\Zack Le\...`). That script can install packages into the wrong Python. Always use the venv interpreter:

`.\env_isaaclab\Scripts\python.exe`

## One-time setup

From the project root:

```powershell
cd C:\path\to\g1-sim

# 1. Create the venv with Python 3.11
py -3.11 -m venv env_isaaclab
.\env_isaaclab\Scripts\Activate.ps1

# 2. Isaac Sim 5.1
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 3. PyTorch (CUDA 12.8 wheels — match your driver)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 4. Isaac Lab (clone next to this repo, then install)
git clone https://github.com/isaac-sim/IsaacLab.git
# Use Isaac Lab's own installer from a path WITHOUT a space if possible.
# Then:
.\env_isaaclab\Scripts\python.exe -m pip install -e .\IsaacLab

# 5. Unitree SDK + sim extras
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
.\env_isaaclab\Scripts\python.exe -m pip install -e .\unitree_sdk2_python
.\env_isaaclab\Scripts\python.exe -m pip install -r .\unitree_sim_isaaclab\requirements.txt
```

### Download USD assets

The robot / table meshes are not in git. From `unitree_sim_isaaclab`:

```powershell
cd unitree_sim_isaaclab
# Git LFS must be installed
.\fetch_assets.sh
```

You need `unitree_sim_isaaclab\assets\` after this step. The BrainCo G1 USD used here is:

`unitree_sim_isaaclab\assets\robots\g1-29dof-brainco-base-fix-usd\g1_29dof_with_brainco_base_fix.usd`

If that folder is missing, rebuild it with `unitree_sim_isaaclab\tools\convert_g1_brainco.py` (uses the combined G1 + Revo2 URDF).

## Run the sim

Always from the **project root**. Accept the Isaac EULA via the script (it sets `OMNI_KIT_ACCEPT_EULA`).

```powershell
cd C:\path\to\g1-sim

# BrainCo Revo2 hands — click the can, then the table
.\run_g1_hands.ps1 -BrainCo -Demo

# Dex3 three-finger hands
.\run_g1_hands.ps1 -Dex3 -Demo
```

First launch can take several minutes while Kit pulls extensions.

Equivalent manual command (BrainCo demo):

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:ACCEPT_EULA = "Y"
$env:PYTHONIOENCODING = "utf-8"
& ".\env_isaaclab\Scripts\python.exe" `
  ".\unitree_sim_isaaclab\sim_main.py" `
  --device cpu `
  --task Isaac-PickPlace-Cylinder-G129-BrainCo-Joint `
  --robot_type g129 `
  --enable_cameras `
  --action_source scripted
```

`--enable_cameras` is required (the scene spawns cameras). `--device cpu` is the physics device used here; rendering still uses the NVIDIA GPU.

### If `.\run_g1_hands.ps1` is not found

Your shell is probably still inside `unitree_sim_isaaclab`. `cd` back to the project root, or run the copy in that folder (it forwards to the parent).

## Click-to-pick controls

Use the Isaac window that shows the robot. Close extra Viewport windows if clicks miss.

| Input | Action |
|---|---|
| Click once | Lock the can |
| Click the table | Start pick and place |
| **R** | Reset the can (and the robot root). Do not use Isaac Stop/Play to reset. |

The robot does not move until the table click. After you change Python (the scripted grasp, tasks, etc.), **quit Isaac and launch again** — the process does not reload those files.

## Tasks

| Task | Hands |
|---|---|
| `Isaac-PickPlace-Cylinder-G129-BrainCo-Joint` | BrainCo Revo2 (this demo) |
| `Isaac-PickPlace-Cylinder-G129-Dex3-Joint` | Dex3 |

## DDS / real robot

The sim publishes the same DDS topics as a real G1.

- Use DDS **domain 1** for this sim (the launch path already does).
- **Domain 0 is the physical robot.** Do not run the sim on domain 0 if a real G1 is on the network.

## Notes for this machine

- VRAM is tight on a 12 GB 3060. Only one Isaac instance at a time. Kill the old window before relaunching.
- Expect well below 100 Hz on the control loop. That is normal.
- After the window opens: **PerspectiveCamera → Cameras → PerspectiveCamera** if you are not looking at the robot.

## License

`unitree_sim_isaaclab` is Apache-2.0 (Unitree Robotics). See `unitree_sim_isaaclab/LICENSE`.
