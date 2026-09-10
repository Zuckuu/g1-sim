#!/usr/bin/env bash
# Run from the normal Ubuntu desktop terminal after workspace setup completes.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_root="$(cd -- "$script_dir/.." && pwd)"
runtime_root="$task_root/work/g1-runtime"
py="$runtime_root/isaac50/bin/python"
mkdir -p "$runtime_root/logs"
run_log="$runtime_root/logs/g1-scene-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$run_log") 2>&1
trap 'printf "\nStopped on line %s. Log: %s\n" "$LINENO" "$run_log" >&2' ERR
if ! nvidia-smi; then
    printf 'Run this in the normal Ubuntu desktop terminal with GPU access.\n' >&2
    exit 1
fi
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME
cd "$task_root"
"$py" "$script_dir/g1-scene.py" --device cpu "$@"
printf '\nG1 scene exited. Log: %s\n' "$run_log"
