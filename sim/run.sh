#!/usr/bin/env bash
# Run any script in sim/ inside the project's Isaac Sim 5.0 / Isaac Lab 2.2 environment.
#   sim/run.sh revo2_hand_grasp.py --headless --snapshot
#   sim/run.sh g1-scene.py --device cpu
# Needs GPU + display access (normal desktop terminal, or a shell that inherits DISPLAY).
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
runtime_root="$project_root/work/g1-runtime"
py="$runtime_root/isaac50/bin/python"
if [[ $# -lt 1 ]]; then
    printf 'usage: %s <script-in-sim/> [args...]\n' "$0" >&2
    exit 2
fi
script="$1"; shift
[[ "$script" == */* ]] || script="$script_dir/$script"
mkdir -p "$runtime_root/logs"
run_log="$runtime_root/logs/$(basename "${script%.*}")-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$run_log") 2>&1
trap 'printf "\nStopped on line %s. Log: %s\n" "$LINENO" "$run_log" >&2' ERR
if ! nvidia-smi >/dev/null 2>&1; then
    printf 'No GPU access in this shell. Run from the normal Ubuntu desktop terminal.\n' >&2
    exit 1
fi
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH PYTHONHOME
cd "$project_root"
printf 'log: %s\n' "$run_log"
"$py" "$script" "$@"
printf '\nDone. Log: %s\n' "$run_log"
