"""Shared helpers for the PepsiDemo Isaac Sim / Isaac Lab scripts.

Import this *before* creating the AppLauncher and call `select_experience(args)`.

Why this exists: the Isaac Lab checkout under work/IsaacLab is sparse (only `source/isaaclab`), but Isaac Lab's
stock experience (.kit) files also list `isaaclab_assets`, `isaaclab_tasks`, `isaaclab_mimic` and `isaaclab_rl`
as required extensions. Kit then looks for them in the remote registry and aborts ("untrusted extension").
We generate trimmed copies of the stock kit files (same settings, only the `isaaclab` extension) next to the
originals so the `${app}`-relative paths inside them keep working.
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORK = PROJECT_ROOT / "work"
ISAACLAB_APPS = WORK / "IsaacLab" / "apps"
RUNTIME = WORK / "g1-runtime"
LOG_DIR = RUNTIME / "logs"
USD_CACHE = RUNTIME / "usd"

_DROP = re.compile(r'^"isaaclab_(assets|tasks|mimic|rl)"\s*=')
_PREFIX = "pepsidemo."


def _trimmed_kit(stock_name: str) -> Path:
    src = ISAACLAB_APPS / stock_name
    dst = ISAACLAB_APPS / (_PREFIX + stock_name)
    if not src.is_file():
        raise FileNotFoundError(src)
    text = src.read_text()
    lines = []
    for line in text.splitlines():
        if _DROP.match(line.strip()):
            continue
        # dependencies on other stock app files must point at their trimmed twins
        line = re.sub(r'^"(isaaclab\.python[.\w]*)"\s*=', lambda m: f'"{_PREFIX}{m.group(1)}" =', line)
        lines.append(line)
    out = "\n".join(lines) + "\n"
    if not dst.is_file() or dst.read_text() != out:
        dst.write_text(out)
    return dst


def select_experience(args) -> str:
    """Return the path of a trimmed experience file matching Isaac Lab's own headless/camera selection."""
    for name in ("isaaclab.python.headless.kit", "isaaclab.python.kit"):
        _trimmed_kit(name)
    headless = bool(getattr(args, "headless", False))
    cameras = bool(getattr(args, "enable_cameras", False))
    if cameras:
        name = "isaaclab.python.headless.rendering.kit" if headless else "isaaclab.python.rendering.kit"
    else:
        name = "isaaclab.python.headless.kit" if headless else "isaaclab.python.kit"
    path = _trimmed_kit(name)
    args.experience = str(path)
    return str(path)


def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    USD_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
