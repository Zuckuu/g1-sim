"""Generate dimensionally faithful soda-container meshes (OBJ) for the grasp simulation.

Two kinds of revolved profile:
  bottle  petaloid-style base ring, straight lower body, contoured grip waist, label panel, shoulder, 28 mm neck
          finish, cap (PET bottles)
  can     standard necked-in aluminium beverage can: stand ring, bottom chime, straight body, neck taper, seamed
          top chime with a recessed lid
The trademarked Pepsi surface styling is not reproduced — only the dimensions that matter for a power grasp:
diameter along the height, height, mass.

Presets (public dimensions, +/- a few mm between production runs; replace with your own measurements):
  pepsi-12oz-can  122.4 mm tall, 66.2 mm body, 54 mm top seam, ~0.38 kg full (US 12 fl oz / 355 mL; the demo object)
  pepsi-500ml     231 mm tall, 66 mm max diameter, ~0.525 kg full (US 16.9 fl oz / EU 500 ml)
  pepsi-20oz      222 mm tall, 72.8 mm max diameter, ~0.64 kg full (US 20 fl oz vending spec)

Usage:
  python sim/make_bottle_mesh.py pepsi-12oz-can         # -> assets/bottles/pepsi-12oz-can.obj + .json
  python sim/make_bottle_mesh.py pepsi-20oz --waist 0.90
  python sim/make_bottle_mesh.py custom --height 0.225 --diameter 0.066 --mass 0.53 --name mybottle
  python sim/make_bottle_mesh.py custom --kind can --height 0.1224 --diameter 0.0662 --mass 0.37 --name diet-can

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
    # 12 fl oz / 355 mL aluminium can ("211 x 413" body, 202 end): 0.355 L x 1.04 kg/L (regular) + 14.7 g can = 0.38 kg;
    # Diet Pepsi is ~0.37 kg. waist/waist_center are unused for cans (kind="can").
    "pepsi-12oz-can": dict(height=0.1224, diameter=0.0662, mass=0.38, waist=1.0, waist_center=0.5, kind="can"),
}
# where the palm centre should sit above the base when nothing else is specified: label zone on a bottle,
# middle of the body on a can (also roughly its centre of mass)
DEFAULT_GRASP_HEIGHT = {"bottle": 0.10, "can": 0.06}


def kind_of(preset_name: str) -> str:
    return PRESETS.get(preset_name, {}).get("kind", "bottle")


def default_grasp_height(preset_name: str) -> float:
    return DEFAULT_GRASP_HEIGHT[kind_of(preset_name)]


def can_profile(height, diameter, seam_d=0.0540, lid_recess=0.0045, stand_ring_d=0.0490, neck_d=0.0570):
    """(r, z) outer contour of a necked-in beverage can, base centre to lid centre.

    Body cylinder from the bottom chime to ~82 % of the height, then the neck tapers to the seam diameter; the top
    chime is the double seam; the lid sits recessed inside it. The bottom stands on a ring (the dome inside the
    ring is left flat: irrelevant to grasping, keeps the mesh simple and the base stable).
    """
    R = diameter / 2.0
    H = height
    body_top = 0.82 * H              # start of the neck taper (~100 mm on a 122 mm can)
    neck_top = H - 0.008             # top of the taper / bottom of the seam
    pts = [
        (0.0, 0.0),
        (stand_ring_d / 2.0, 0.0),           # stand ring
        (0.93 * R, 0.0035),                  # bottom chime
        (R, 0.011),                          # full body diameter
        (R, body_top),                       # straight body
        (neck_d / 2.0, body_top + 0.6 * (neck_top - body_top)),  # neck taper
        (seam_d / 2.0, neck_top),
        (seam_d / 2.0, H),                   # seam (top chime)
        (seam_d / 2.0 - 0.0035, H),          # seam width
        (seam_d / 2.0 - 0.0035, H - lid_recess),  # recessed lid
        (0.0, H - lid_recess),
    ]
    return np.array(pts, dtype=float)


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


def build(name, height, diameter, mass, waist, waist_center, sections=96, kind="bottle"):
    prof = can_profile(height, diameter) if kind == "can" else profile(height, diameter, waist, waist_center)
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
        "name": name, "kind": kind, "obj": str(obj), "height": height, "max_diameter": diameter, "mass_full_kg": mass,
        "default_grasp_height": DEFAULT_GRASP_HEIGHT[kind],
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
    ap.add_argument("--waist", type=float, help="bottles: grip diameter / max diameter, e.g. 0.9")
    ap.add_argument("--waist-center", type=float, help="bottles: fraction of height where the grip waist is centred")
    ap.add_argument("--kind", choices=["bottle", "can"], default=None, help="profile family (default: the preset's, or bottle)")
    a = ap.parse_args()
    p = dict(PRESETS.get(a.preset, PRESETS["pepsi-500ml"]))
    for k in ("height", "diameter", "mass", "waist", "waist_center", "kind"):
        v = getattr(a, k)
        if v is not None:
            p[k] = v
    p.setdefault("kind", "bottle")
    name = a.name or a.preset
    obj, meta = build(name, **p)
    grasp_z = DEFAULT_GRASP_HEIGHT[p["kind"]]
    print(f"{obj}\n  height {p['height']*1000:.0f} mm, max dia {p['diameter']*1000:.1f} mm, waist dia "
          f"{p['waist']*p['diameter']*1000:.1f} mm, mass {p['mass']} kg, volume {meta['volume_m3']*1e6:.0f} cm3, "
          f"faces {meta['faces']}\n  radius at {grasp_z*1000:.0f} mm height: {radius_at(np.array(meta['profile_rz']), grasp_z)*1000:.1f} mm")


if __name__ == "__main__":
    main()
