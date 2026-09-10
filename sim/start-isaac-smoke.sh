#!/usr/bin/env bash
# Host-terminal bootstrap for an isolated Isaac Sim 5.0 smoke test.
# Does not install Isaac Lab, Unitree, XR, system packages, or GPU drivers.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_root="$(cd -- "$script_dir/.." && pwd)"
runtime_root="$task_root/work/g1-runtime"
mkdir -p "$runtime_root/logs"
run_log="$runtime_root/logs/smoke-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$run_log") 2>&1
trap 'printf "\nStopped on line %s. Log: %s\n" "$LINENO" "$run_log" >&2' ERR

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi; then
    printf 'GPU access is unavailable here. Run this script in the normal Ubuntu desktop terminal.\n' >&2
    exit 1
fi
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    printf 'No desktop display found. Run this from your Ubuntu desktop terminal.\n' >&2
    exit 1
fi
if command -v uv >/dev/null; then
    uv_bin="$(command -v uv)"
elif [[ -x /home/nicholas/.local/bin/uv ]]; then
    uv_bin=/home/nicholas/.local/bin/uv
else
    printf 'uv is missing. Stop here and report this message; no system changes were made.\n' >&2
    exit 1
fi

export UV_CACHE_DIR="$runtime_root/cache/uv"
export UV_PYTHON_INSTALL_DIR="$runtime_root/python"
export PIP_CACHE_DIR="$runtime_root/cache/pip"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
env_dir="$runtime_root/isaac50"

printf '\nInstalling a dedicated Python environment at %s\n' "$env_dir"
printf 'The Isaac packages are large. First installation and shader compilation can take considerable time.\n'
if [[ ! -x "$env_dir/bin/python" ]]; then
    "$uv_bin" venv --python 3.11 --seed "$env_dir"
fi
py="$env_dir/bin/python"
"$py" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
"$py" -m pip install --upgrade pip
"$py" -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

"$py" - <<'PY'
import torch
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'
assert torch.version.cuda == '12.8', f'Unexpected CUDA build: {torch.version.cuda}'
x = torch.ones((32, 32), device='cuda')
y = x @ x
torch.cuda.synchronize()
assert float(y[0, 0]) == 32.0
print('CUDA computation passed:', torch.cuda.get_device_name(0), torch.__version__)
PY

"$py" -m pip install 'isaacsim[all,extscache]==5.0.0' \
    --extra-index-url https://pypi.nvidia.com
"$py" -m pip freeze > "$runtime_root/logs/isaac50-packages.txt"

printf '\nOpening a small empty Isaac viewport. First startup may take over ten minutes.\n'
printf 'Isaac may present its license prompt in this terminal.\n'
printf 'The window will close automatically after 120 update frames. This does not test G1 grasping.\n'
"$py" "$script_dir/isaac-smoke.py"
printf '\nSmoke test completed. Log: %s\nNext: install Isaac Lab and load the reduced G1 scene.\n' "$run_log"
