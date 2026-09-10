"""Download the Unitree G1 MJCF model (with Dex3 hands) from MuJoCo Menagerie.

The model is BSD-3 licensed (Unitree Robotics / Google DeepMind) and is ~36 MB of
STL meshes, so it is not committed to git. Run this once (or let ``run.py`` call it
automatically on first use)::

    python fetch_assets.py

The files end up in ``roundtable_sim/assets/unitree_g1/``.
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
REQUIRED_FILES = ("g1_with_hands.xml", "assets/right_hand_palm_link.STL", "LICENSE")


def assets_present() -> bool:
    return all((G1_DIR / f).is_file() for f in REQUIRED_FILES)


def fetch(force: bool = False) -> Path:
    """Sparse-clone only the ``unitree_g1`` folder of MuJoCo Menagerie."""
    if assets_present() and not force:
        return G1_DIR
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ASSETS_DIR / "_menagerie_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    print(f"[fetch_assets] cloning {MENAGERIE_URL} (sparse: unitree_g1) ...", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         "--branch", MENAGERIE_REF, MENAGERIE_URL, str(tmp)],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp), "sparse-checkout", "set", "unitree_g1"], check=True)
    if G1_DIR.exists():
        shutil.rmtree(G1_DIR)
    shutil.move(str(tmp / "unitree_g1"), str(G1_DIR))
    shutil.rmtree(tmp, ignore_errors=True)
    # The preview PNGs are 1.7 MB each and unused.
    for png in G1_DIR.glob("*.png"):
        png.unlink()
    if not assets_present():
        raise RuntimeError(f"Unitree G1 assets incomplete in {G1_DIR}")
    print(f"[fetch_assets] done -> {G1_DIR}", flush=True)
    return G1_DIR


if __name__ == "__main__":
    fetch(force="--force" in sys.argv)
