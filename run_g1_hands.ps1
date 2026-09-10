# Launch Unitree G1 pick-place in Isaac Sim.
# Usage:
#   .\run_g1_hands.ps1 -Dex3
#   .\run_g1_hands.ps1 -Dex3 -Demo    # click the can, then click where to place it
#   .\run_g1_hands.ps1 -BrainCo -Demo # BrainCo Revo2 hands
# These scenes spawn cameras, so --enable_cameras is required.

param(
    [switch]$Dex3,
    [switch]$BrainCo,
    [switch]$Demo
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root "env_isaaclab\Scripts\python.exe"
$sim = Join-Path $root "unitree_sim_isaaclab"

$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:ACCEPT_EULA = "Y"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

if ($BrainCo) {
    $task = "Isaac-PickPlace-Cylinder-G129-BrainCo-Joint"
    $hand = $null
} elseif ($Dex3) {
    $task = "Isaac-PickPlace-Cylinder-G129-Dex3-Joint"
    $hand = "--enable_dex3_dds"
} else {
    $task = "Isaac-PickPlace-Cylinder-G129-Dex1-Joint"
    $hand = "--enable_dex1_dds"
}

$argsList = @(
    "sim_main.py",
    "--device", "cpu",
    "--task", $task,
    "--robot_type", "g129"
)
if ($hand) {
    $argsList += $hand
}
$argsList += "--enable_cameras"
if ($Demo) {
    $argsList += @("--action_source", "scripted")
}

Write-Host "Starting $task"
Push-Location $sim
try {
    & $py @argsList
} finally {
    Pop-Location
}
