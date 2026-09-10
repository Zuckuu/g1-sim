"""Generate dimensionally faithful PET soda-bottle meshes (OBJ) for the grasp simulation.

The bottle is a revolved profile: petaloid-style base ring, straight lower body, contoured grip waist, label
panel, shoulder, 28 mm neck finish, cap. The trademarked Pepsi surface styling (swirl, embossing) is not
reproduced — only the dimensions that matter for a power grasp: diameter along the height, height, mass.

Presets (public dimensions, +/- a few mm between bottling runs; replace with your own measurements):
  pepsi-500ml   231 mm tall, 66 mm max diameter, ~0.525 kg full (US 16.9 fl oz / EU 500 ml)
  pepsi-20oz    222 mm tall, 72.8 mm max diameter, ~0.64 kg full (US 20 fl oz vending spec)

Usage:
  python sim/make_bottle_mesh.py pepsi-500ml            # -> assets/bottles/pepsi-500ml.obj + .json
  python sim/make_bottle_mesh.py pepsi-20oz --waist 0.90
  python sim/make_bottle_mesh.py custom --height 0.225 --diameter 0.066 --mass 0.53 --name mybottle

Any existing OBJ/STL (e.g. a phone photogrammetry scan of the real bottle) can be used directly by the grasp
script with --bottle-mesh <file>; this generator is for when no scan exists yet.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import trimesh

OUT_DIR = Path(__file__).resolve().parent.parent / "assets/bottles"

PRESETS = {
    # height, max diameter, full mass, waist ratio (grip diameter / max diameter), waist centre (fraction of height)
    "pepsi-500ml": dict(height=0.231, diameter=0.066, mass=0.525, waist=0.92, waist_center=0.45),
    "pepsi-20oz": dict(height=0.2224, diameter=0.0728, mass=0.64, waist=0.90, waist_center=0.42),
}


def profile(height, diameter, waist, waist_center, neck_d=0.0274, cap_d=0.0305, cap_h=0.0145):
    """Return (r, z) polyline of the outer contour from the base centre to the cap centre."""
    R = diameter / 2.0
    H = height
    body_top = H - cap_h - 0.012 - 0.045  # top of the label panel, where the shoulder starts
    neck_top = H - cap_h
    wc = waist_center * H
    ww = 0.045  # waist half-height
    pts = [
        (0.0, 0.0),
        (0.40 * R, 0.0),           # flat base contact ring (petaloid feet sit on roughly this radius)
        (0.80 * R, 0.003),
        (0.97 * R, 0.012),
        (R, 0.030),                # full diameter reached
        (R, wc - ww - 0.010),
        (waist * R, wc - ww * 0.5),  # grip waist
        (waist * R, wc + ww * 0.5),
        (R, wc + ww + 0.010),
        (R, body_top),             # label panel
        (0.93 * R, body_top + 0.012),
        (0.72 * R, body_top + 0.026),
        (0.50 * R, body_top + 0.036),
        (neck_d / 2.0 + 0.002, body_top + 0.045),  # shoulder meets neck
        (neck_d / 2.0, neck_top - 0.002),
        (cap_d / 2.0, neck_top),   # cap
        (cap_d / 2.0, H - 0.0015),
        (cap_d / 2.0 - 0.0015, H),
        (0.0, H),
    ]
    return np.array(pts, dtype=float)


def radius_at(prof, z):
    r, zz = prof[:, 0], prof[:, 1]
    order = np.argsort(zz)
    return float(np.interp(z, zz[order], r[order]))


def build(name, height, diameter, mass, waist, waist_center, sections=96):
    prof = profile(height, diameter, waist, waist_center)
    # trimesh revolve expects x = radius, y = height; points on the axis close the surface
    mesh = trimesh.creation.revolve(prof, sections=sections)
    mesh.merge_vertices()
    mesh.fix_normals()
    if not mesh.is_watertight:
        raise RuntimeError("revolved bottle is not watertight")
    if mesh.volume < 0:
        mesh.invert()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    obj = OUT_DIR / f"{name}.obj"
    mesh.export(obj)
    meta = {
        "name": name, "obj": str(obj), "height": height, "max_diameter": diameter, "mass_full_kg": mass,
        "waist_ratio": waist, "waist_center_frac": waist_center, "volume_m3": float(mesh.volume),
        "centroid_z": float(mesh.centroid[2]), "faces": int(len(mesh.faces)), "origin": "base centre, +Z up",
        "radius_at_height": {f"{z:.3f}": radius_at(prof, z) for z in np.arange(0.02, height - 0.01, 0.01)},
        "profile_rz": prof.tolist(),
        "note": "dimensions are public approximations of the real bottle; styling is not reproduced",
    }
    (OUT_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return obj, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=list(PRESETS) + ["custom"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--height", type=float)
    ap.add_argument("--diameter", type=float)
    ap.add_argument("--mass", type=float)
    ap.add_argument("--waist", type=float, help="grip diameter / max diameter, e.g. 0.9")
    ap.add_argument("--waist-center", type=float, help="fraction of height where the grip waist is centred")
    a = ap.parse_args()
    p = dict(PRESETS.get(a.preset, PRESETS["pepsi-500ml"]))
    for k in ("height", "diameter", "mass", "waist", "waist_center"):
        v = getattr(a, k)
        if v is not None:
            p[k] = v
    name = a.name or a.preset
    obj, meta = build(name, **p)
    grasp_z = 0.10
    print(f"{obj}\n  height {p['height']*1000:.0f} mm, max dia {p['diameter']*1000:.1f} mm, waist dia "
          f"{p['waist']*p['diameter']*1000:.1f} mm, mass {p['mass']} kg, volume {meta['volume_m3']*1e6:.0f} cm3, "
          f"faces {meta['faces']}\n  radius at {grasp_z*1000:.0f} mm height: {radius_at(np.array(meta['profile_rz']), grasp_z)*1000:.1f} mm")


if __name__ == "__main__":
    main()
