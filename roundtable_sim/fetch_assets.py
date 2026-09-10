"""Download the robot models: the Unitree G1 body and the BrainCo Revo2 hands.

* Unitree G1 MJCF from MuJoCo Menagerie (BSD-3, Unitree Robotics / Google DeepMind,
  ~36 MB of STL meshes).  It ships with Dex3 hands; ``scene.py`` swaps those for the Revo2.
* BrainCo Revo2 dexterous hand MJCF + meshes from BrainCo's public robot-description
  repository (``BrainCoTech/brainco-description``, ``revo2_system/``).

Neither is committed to git. Run this once (or let ``run.py`` call it automatically on
first use)::

    python fetch_assets.py

The files end up in ``roundtable_sim/assets/unitree_g1/`` and
``roundtable_sim/assets/brainco_revo2/``.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

G1_DIR = ASSETS_DIR / "unitree_g1"
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
# Pin to a known-good revision so the joint layout the controller relies on never
# changes underneath us.  Bump deliberately.
MENAGERIE_REF = "main"
G1_REQUIRED = ("g1_with_hands.xml", "assets/right_wrist_yaw_link.STL", "LICENSE")

REVO2_DIR = ASSETS_DIR / "brainco_revo2"
BRAINCO_URL = "https://github.com/BrainCoTech/brainco-description.git"
BRAINCO_REF = "main"
REVO2_REQUIRED = ("mjcf/revo2_right.xml", "mjcf/revo2_left.xml",
                  "meshes/hands/visual/right/right_base_link.STL",
                  "meshes/hands/visual/left/left_base_link.STL")


def _present(root: Path, required) -> bool:
    return all((root / f).is_file() for f in required)


def _sparse_clone(url: str, ref: str, subdir: str, dest: Path) -> None:
    """Clone only ``subdir`` of a git repo into ``dest`` (blobless, depth 1)."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ASSETS_DIR / f"_{dest.name}_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    print(f"[fetch_assets] cloning {url} (sparse: {subdir}) ...", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "--branch", ref, url, str(tmp)],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp), "sparse-checkout", "set", subdir], check=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(tmp / subdir), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)


def fetch(force: bool = False) -> Path:
    """Unitree G1 from MuJoCo Menagerie -> ``assets/unitree_g1``."""
    if _present(G1_DIR, G1_REQUIRED) and not force:
        return G1_DIR
    _sparse_clone(MENAGERIE_URL, MENAGERIE_REF, "unitree_g1", G1_DIR)
    # The preview PNGs are 1.7 MB each and unused.
    for png in G1_DIR.glob("*.png"):
        png.unlink()
    if not _present(G1_DIR, G1_REQUIRED):
        raise RuntimeError(f"Unitree G1 assets incomplete in {G1_DIR}")
    print(f"[fetch_assets] done -> {G1_DIR}", flush=True)
    return G1_DIR


def fetch_revo2(force: bool = False) -> Path:
    """BrainCo Revo2 hands (MJCF + meshes) -> ``assets/brainco_revo2``."""
    if _present(REVO2_DIR, REVO2_REQUIRED) and not force:
        return REVO2_DIR
    _sparse_clone(BRAINCO_URL, BRAINCO_REF, "revo2_system", REVO2_DIR)
    if not _present(REVO2_DIR, REVO2_REQUIRED):
        raise RuntimeError(f"BrainCo Revo2 assets incomplete in {REVO2_DIR}")
    print(f"[fetch_assets] done -> {REVO2_DIR}", flush=True)
    return REVO2_DIR


def fetch_all(force: bool = False) -> tuple[Path, Path]:
    return fetch(force), fetch_revo2(force)


if __name__ == "__main__":
    fetch_all(force="--force" in sys.argv)
