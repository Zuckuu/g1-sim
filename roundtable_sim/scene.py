"""Builds the MJCF world: round table, 10 seated guests, drink station, cans, cameras.

The Unitree G1 (with Dex3 hands) is pulled in via ``<include>`` from the Menagerie
model that ``fetch_assets.py`` downloads.  Everything else is generated from
``config.Layout`` so the controller can compute poses from the same numbers.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import DIET, PEPSI, Guest, Layout, ScenarioConfig
from fetch_assets import G1_DIR, fetch

GEN_DIR_NAME = "gen"  # textures + generated xml live inside the g1 folder so relative paths resolve
SCENE_XML_NAME = "roundtable_scene.xml"
G1_NOKEY_NAME = "g1_with_hands_nokey.xml"

PEPSI_BLUE = "#0e4c9a"
PEPSI_RED = "#e32934"
DIET_SILVER = "#d9dee3"
CAN_TOP = "#b9bec4"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def hex_to_rgba(h: str, a: float = 1.0) -> str:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return f"{r:.3f} {g:.3f} {b:.3f} {a:.3f}"


def quat_z(yaw: float) -> str:
    return f"{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}"


def fmt(v: Iterable[float]) -> str:
    return " ".join(f"{x:.5f}" for x in v)


def camera_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> str:
    """MuJoCo cameras look down their -z axis with +y up: derive xyaxes from a look-at."""
    pos = np.asarray(pos, float)
    fwd = np.asarray(target, float) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    return fmt(right) + " " + fmt(cam_up)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------------------
# textures
# --------------------------------------------------------------------------------------
def _can_label(path: Path, kind: str) -> None:
    w, h = 1024, 256
    if kind == PEPSI:
        bg, fg, band = PEPSI_BLUE, "#ffffff", PEPSI_RED
        text = "PEPSI"
    else:
        bg, fg, band = DIET_SILVER, PEPSI_BLUE, PEPSI_RED
        text = "DIET PEPSI"
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    # two label repeats around the circumference so the text is readable from most angles
    for k in range(2):
        cx = w // 8 + k * (w // 2)
        d.ellipse([cx - 80, h // 2 - 80, cx + 80, h // 2 + 80], fill="#ffffff")
        d.pieslice([cx - 80, h // 2 - 80, cx + 80, h // 2 + 80], start=180, end=360, fill=band)
        d.pieslice([cx - 80, h // 2 - 80, cx + 80, h // 2 + 80], start=0, end=180, fill=PEPSI_BLUE)
        d.rectangle([cx - 80, h // 2 - 14, cx + 80, h // 2 + 14], fill="#ffffff")
        fsize = 120 if kind == PEPSI else 64
        f = _font(fsize)
        d.text((cx + 105, h // 2 - fsize * 0.62), text, fill=fg, font=f)
    d.rectangle([0, 0, w, 14], fill=CAN_TOP)
    d.rectangle([0, h - 14, w, h], fill=CAN_TOP)
    img.save(path)


def _name_card(path: Path, name: str, order: str, color: str) -> None:
    # Cube textures must be square; the box face is ~2:1 so the image is drawn 2:1
    # and squeezed to a square (the face stretches it back).
    w, h = 512, 256
    img = Image.new("RGB", (w, h), "#fbf7ee")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 28], fill=color)
    f = _font(96)
    tw = d.textlength(name, font=f)
    d.text(((w - tw) / 2, 50), name, fill="#222222", font=f)
    f2 = _font(40)
    sub = f"wants: {order}"
    tw2 = d.textlength(sub, font=f2)
    d.text(((w - tw2) / 2, 175), sub, fill=PEPSI_BLUE if order == PEPSI else "#555555", font=f2)
    _save_face_texture(img, path)


def _save_face_texture(img: Image.Image, path: Path) -> None:
    """MuJoCo maps a single-image cube texture onto each box face with the image's
    x axis along the face's z axis, so rotate by 90 deg and squeeze to a square."""
    img.rotate(90, expand=True).resize((512, 512), Image.LANCZOS).save(path)


def _sign(path: Path, text: str, bg: str, fg: str) -> None:
    w, h = 1024, 256  # 4:1 face, squeezed to a square (see _save_face_texture)
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    f = _font(120)
    tw = d.textlength(text, font=f)
    d.text(((w - tw) / 2, 55), text, fill=fg, font=f)
    _save_face_texture(img, path)


def make_textures(gen_dir: Path, cfg: ScenarioConfig) -> None:
    gen_dir.mkdir(parents=True, exist_ok=True)
    _can_label(gen_dir / "label_pepsi.png", PEPSI)
    _can_label(gen_dir / "label_diet.png", DIET)
    _sign(gen_dir / "sign_drinks.png", "DRINKS", "#1b1b1b", "#f5f5f5")
    for i, g in enumerate(cfg.guests):
        _name_card(gen_dir / f"card_{i}.png", g.name, g.order, g.shirt)


# --------------------------------------------------------------------------------------
# world pieces
# --------------------------------------------------------------------------------------
def _chair_and_guest(i: int, g: Guest, L: Layout) -> str:
    """Chair + stylised seated person in a local frame: x towards the table, y left, z up."""
    ang = L.guest_angle(i)
    pos = L.polar(L.chair_radius, ang)
    yaw = ang + math.pi  # face the table centre
    shirt, skin, hair = hex_to_rgba(g.shirt), hex_to_rgba(g.skin), hex_to_rgba(g.hair)
    wood = hex_to_rgba("#4a3524")
    cushion = hex_to_rgba("#6d5a4b")
    shoe = hex_to_rgba("#222222")
    trousers = hex_to_rgba("#2f3542")
    sh = L.seat_height
    parts = [f'<body name="guest_{i}" pos="{fmt(pos)}" quat="{quat_z(yaw)}">']
    # chair
    parts.append(f'  <geom type="box" size="0.23 0.23 0.025" pos="0 0 {sh - 0.025:.3f}" rgba="{cushion}" class="furniture"/>')
    parts.append(f'  <geom type="box" size="0.02 0.23 0.24" pos="-0.22 0 {sh + 0.24:.3f}" rgba="{wood}" class="furniture"/>')
    for sx in (-0.2, 0.2):
        for sy in (-0.2, 0.2):
            parts.append(f'  <geom type="cylinder" size="0.018 {(sh - 0.05) / 2:.3f}" pos="{sx} {sy} {(sh - 0.05) / 2:.3f}" rgba="{wood}" class="furniture"/>')
    # legs (seated)
    for sy in (-0.09, 0.09):
        parts.append(f'  <geom type="capsule" fromto="0.0 {sy} {sh + 0.07:.3f} 0.40 {sy} {sh + 0.07:.3f}" size="0.07" rgba="{trousers}" class="person"/>')
        parts.append(f'  <geom type="capsule" fromto="0.40 {sy} {sh + 0.05:.3f} 0.42 {sy} 0.07" size="0.055" rgba="{trousers}" class="person"/>')
        parts.append(f'  <geom type="box" size="0.12 0.05 0.03" pos="0.50 {sy} 0.03" rgba="{shoe}" class="person"/>')
    # torso, shoulders, neck, head, hair
    parts.append(f'  <geom type="box" size="0.11 0.18 0.27" pos="-0.03 0 {sh + 0.33:.3f}" rgba="{shirt}" class="person"/>')
    for sy in (-0.19, 0.19):
        parts.append(f'  <geom type="sphere" size="0.065" pos="-0.03 {sy} {sh + 0.55:.3f}" rgba="{shirt}" class="person"/>')
    parts.append(f'  <geom type="cylinder" size="0.045 0.035" pos="-0.03 0 {sh + 0.63:.3f}" rgba="{skin}" class="person"/>')
    parts.append(f'  <geom type="sphere" size="0.105" pos="-0.02 0 {sh + 0.75:.3f}" rgba="{skin}" class="person"/>')
    parts.append(f'  <geom type="sphere" size="0.108" pos="-0.045 0 {sh + 0.78:.3f}" rgba="{hair}" class="person"/>')
    # arms resting on the table
    for sy in (-1, 1):
        parts.append(f'  <geom type="capsule" fromto="-0.03 {0.21 * sy:.3f} {sh + 0.53:.3f} 0.18 {0.24 * sy:.3f} {L.table_height + 0.04:.3f}" size="0.045" rgba="{shirt}" class="person"/>')
        parts.append(f'  <geom type="capsule" fromto="0.18 {0.24 * sy:.3f} {L.table_height + 0.04:.3f} 0.50 {0.17 * sy:.3f} {L.table_height + 0.04:.3f}" size="0.04" rgba="{skin}" class="person"/>')
    parts.append("</body>")
    return "\n".join(parts)


def _coaster_and_card(i: int, L: Layout) -> str:
    cx, cy, cz = L.coaster_pos(i)
    ang = L.guest_angle(i)
    card_pos = L.polar(L.coaster_radius - 0.16, ang, L.table_height + 0.032)
    # the card faces the guest (outwards); texture is on every face so both sides read
    yaw = ang
    return "\n".join([
        f'<geom name="coaster_{i}" type="cylinder" size="{L.coaster_size} 0.003" pos="{cx:.4f} {cy:.4f} {cz + 0.003:.4f}" '
        f'rgba="0.15 0.15 0.17 1" class="furniture"/>',
        f'<body name="card_{i}" pos="{fmt(card_pos)}" quat="{quat_z(yaw)}">',
        f'  <geom type="box" size="0.003 0.065 0.032" material="card_{i}" class="decor"/>',
        "</body>",
    ])


def _station(L: Layout) -> str:
    cx, cy, top = L.station_center()
    yaw = L.station_angle + math.pi / 2  # long side tangential
    hx, hy = L.station_half_x, L.station_half_y
    frame = hex_to_rgba("#2b2b2b")
    parts = [f'<body name="station" pos="{cx:.4f} {cy:.4f} 0" quat="{quat_z(yaw)}">']
    parts.append(f'  <geom name="station_top" type="box" size="{hx} {hy} 0.02" pos="0 0 {top - 0.02:.3f}" rgba="0.93 0.93 0.95 1" class="furniture"/>')
    parts.append(f'  <geom type="box" size="{hx - 0.05:.3f} {hy - 0.03:.3f} 0.015" pos="0 0 {top * 0.45:.3f}" rgba="{frame}" class="furniture"/>')
    for sx in (-hx + 0.04, hx - 0.04):
        for sy in (-hy + 0.04, hy - 0.04):
            parts.append(f'  <geom type="box" size="0.02 0.02 {(top - 0.04) / 2:.3f}" pos="{sx:.3f} {sy:.3f} {(top - 0.04) / 2:.3f}" rgba="{frame}" class="furniture"/>')
    # "DRINKS" sign on the side facing the table (local +y is the inward direction)
    parts.append(f'  <geom type="box" size="0.30 0.004 0.075" pos="0 {hy + 0.004:.3f} {top - 0.13:.3f}" material="sign_drinks" class="decor"/>')
    parts.append("</body>")
    return "\n".join(parts)


def _cans(L: Layout) -> Tuple[str, List[str]]:
    parts, names = [], []
    r, hh = L.can_radius, L.can_half_height
    for k, (x, y, z, kind) in enumerate(L.can_positions()):
        name = f"can_{k}"
        names.append(name)
        mat = "label_pepsi" if kind == PEPSI else "label_diet"
        parts.append(
            f'<body name="{name}" pos="{x:.4f} {y:.4f} {z:.4f}">\n'
            f'  <freejoint name="{name}_free"/>\n'
            f'  <geom name="{name}_geom" type="cylinder" size="{r} {hh}" mass="{L.can_mass}" material="{mat}" class="can"/>\n'
            f'  <geom type="cylinder" size="{r * 0.9:.4f} 0.003" pos="0 0 {hh - 0.001:.4f}" rgba="0.72 0.74 0.77 1" class="decor" mass="0"/>\n'
            f'</body>')
    return "\n".join(parts), names


def _table(L: Layout) -> str:
    R, H, T = L.table_radius, L.table_height, L.table_thickness
    return "\n".join([
        f'<geom name="table_top" type="cylinder" size="{R} {T / 2:.3f}" pos="0 0 {H - T / 2:.4f}" material="tablecloth" class="furniture"/>',
        f'<geom type="cylinder" size="{R + 0.01:.3f} 0.012" pos="0 0 {H - T - 0.012:.4f}" rgba="0.96 0.93 0.86 1" class="furniture"/>',
        f'<geom type="cylinder" size="0.22 {(H - T) / 2:.3f}" pos="0 0 {(H - T) / 2:.4f}" rgba="0.25 0.22 0.2 1" class="furniture"/>',
        f'<geom type="cylinder" size="0.65 0.02" pos="0 0 0.02" rgba="0.25 0.22 0.2 1" class="furniture"/>',
        # centrepiece
        f'<geom type="cylinder" size="0.09 0.06" pos="0 0 {H + 0.06:.3f}" rgba="0.55 0.65 0.35 1" class="decor"/>',
        f'<geom type="sphere" size="0.16" pos="0 0 {H + 0.24:.3f}" rgba="0.36 0.55 0.28 1" class="decor"/>',
        # rug
        f'<geom type="cylinder" size="{L.ring_radius + 0.55:.3f} 0.004" pos="0 0 0.004" rgba="0.42 0.16 0.16 1" class="decor"/>',
    ])


ROOM_X = 7.0  # half width of the dining room
ROOM_Y_MIN, ROOM_Y_MAX = -9.5, 6.0
ROOM_H = 3.4


def _room() -> str:
    wall = "0.82 0.84 0.86 1"
    base = "0.35 0.30 0.28 1"
    ym, yM, h = ROOM_Y_MIN, ROOM_Y_MAX, ROOM_H
    yc, yh = (ym + yM) / 2, (yM - ym) / 2
    parts = []
    for x in (-ROOM_X, ROOM_X):
        parts.append(f'<geom type="box" size="0.05 {yh:.3f} {h / 2:.3f}" pos="{x:.3f} {yc:.3f} {h / 2:.3f}" rgba="{wall}" class="decor"/>')
        parts.append(f'<geom type="box" size="0.06 {yh:.3f} 0.08" pos="{x:.3f} {yc:.3f} 0.08" rgba="{base}" class="decor"/>')
    for y in (ym, yM):
        parts.append(f'<geom type="box" size="{ROOM_X:.3f} 0.05 {h / 2:.3f}" pos="0 {y:.3f} {h / 2:.3f}" rgba="{wall}" class="decor"/>')
        parts.append(f'<geom type="box" size="{ROOM_X:.3f} 0.06 0.08" pos="0 {y:.3f} 0.08" rgba="{base}" class="decor"/>')
    # a few "paintings" on the back wall so the follow camera has some parallax cues
    for x, col in ((-3.5, "0.55 0.25 0.20 1"), (0.0, "0.20 0.35 0.55 1"), (3.5, "0.25 0.45 0.30 1")):
        parts.append(f'<geom type="box" size="0.7 0.02 0.5" pos="{x} {yM - 0.07:.3f} 1.9" rgba="0.15 0.12 0.10 1" class="decor"/>')
        parts.append(f'<geom type="box" size="0.62 0.03 0.42" pos="{x} {yM - 0.07:.3f} 1.9" rgba="{col}" class="decor"/>')
    return "\n".join(parts)


def _cameras(L: Layout) -> str:
    sx, sy, _ = L.station_center()
    ov_pos = (0.0, -7.6, 5.2)
    ov_target = (0.0, -0.6, 0.6)
    td_pos = (0.0, -0.9, 9.5)
    side_pos = (7.5, -2.5, 3.2)
    return "\n".join([
        f'<camera name="overview" pos="{fmt(ov_pos)}" xyaxes="{camera_xyaxes(ov_pos, ov_target)}" fovy="48"/>',
        f'<camera name="topdown" pos="{fmt(td_pos)}" xyaxes="1 0 0 0 1 0" fovy="55"/>',
        f'<camera name="side" pos="{fmt(side_pos)}" xyaxes="{camera_xyaxes(side_pos, (0, -0.5, 0.7))}" fovy="45"/>',
    ])


# --------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------
LOWPOLY_DIR_NAME = "assets_lowpoly"
LOWPOLY_REDUCTION = 0.7  # drop 70% of the triangles; barely visible at video resolution


def _decimate_meshes(g1_dir: Path) -> bool:
    """Write decimated copies of the robot STL meshes (software rendering is geometry bound).

    Returns True if the low-poly set is available. Falls back to the originals when
    the optional decimation dependencies are missing.
    """
    src_dir = g1_dir / "assets"
    dst_dir = g1_dir / LOWPOLY_DIR_NAME
    stls = sorted(src_dir.glob("*.STL"))
    if dst_dir.is_dir() and all((dst_dir / s.name).is_file() for s in stls):
        return True
    try:
        import fast_simplification  # type: ignore
        import trimesh  # type: ignore
    except ImportError:
        print("[scene] trimesh/fast-simplification not installed; rendering full-resolution meshes", flush=True)
        return False
    dst_dir.mkdir(exist_ok=True)
    n_in = n_out = 0
    for s in stls:
        mesh = trimesh.load(str(s), force="mesh")
        n_in += len(mesh.faces)
        if len(mesh.faces) < 400:
            mesh.export(str(dst_dir / s.name))
            n_out += len(mesh.faces)
            continue
        v, f = fast_simplification.simplify(mesh.vertices, mesh.faces, target_reduction=LOWPOLY_REDUCTION)
        trimesh.Trimesh(v, f, process=False).export(str(dst_dir / s.name))
        n_out += len(f)
    print(f"[scene] decimated robot meshes: {n_in} -> {n_out} triangles", flush=True)
    return True


def _strip_keyframes(src: Path, dst: Path, lowpoly: bool) -> None:
    xml = src.read_text()
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
    if lowpoly:
        xml = xml.replace('meshdir="assets"', f'meshdir="{LOWPOLY_DIR_NAME}"', 1)
    dst.write_text(xml)


def build_scene_xml(cfg: ScenarioConfig, g1_dir: Path, lowpoly: bool = True) -> str:
    L = cfg.layout
    gen = g1_dir / GEN_DIR_NAME
    make_textures(gen, cfg)
    _strip_keyframes(g1_dir / "g1_with_hands.xml", g1_dir / G1_NOKEY_NAME, lowpoly and _decimate_meshes(g1_dir))
    cans_xml, _ = _cans(L)

    materials = [
        f'<texture name="label_pepsi" type="2d" file="{GEN_DIR_NAME}/label_pepsi.png"/>',
        f'<material name="label_pepsi" texture="label_pepsi" specular="0.6" shininess="0.6" reflectance="0.05"/>',
        f'<texture name="label_diet" type="2d" file="{GEN_DIR_NAME}/label_diet.png"/>',
        f'<material name="label_diet" texture="label_diet" specular="0.6" shininess="0.6" reflectance="0.05"/>',
        f'<texture name="sign_drinks" type="cube" file="{GEN_DIR_NAME}/sign_drinks.png"/>',
        f'<material name="sign_drinks" texture="sign_drinks"/>',
        '<texture name="tablecloth_tex" type="2d" builtin="checker" rgb1="0.95 0.92 0.85" rgb2="0.90 0.86 0.78" width="256" height="256"/>',
        '<material name="tablecloth" texture="tablecloth_tex" texrepeat="12 12" texuniform="true" reflectance="0.05"/>',
        '<texture name="floor_tex" type="2d" builtin="checker" mark="edge" rgb1="0.56 0.47 0.37" rgb2="0.52 0.43 0.34" markrgb="0.40 0.32 0.25" width="512" height="512"/>',
        '<material name="floor" texture="floor_tex" texrepeat="10 10" texuniform="true" reflectance="0.12"/>',
        '<texture name="sky" type="skybox" builtin="gradient" rgb1="0.75 0.82 0.92" rgb2="0.35 0.42 0.55" width="512" height="3072"/>',
    ]
    for i in range(L.n_guests):
        materials.append(f'<texture name="card_{i}" type="cube" file="{GEN_DIR_NAME}/card_{i}.png"/>')
        materials.append(f'<material name="card_{i}" texture="card_{i}"/>')

    guests = "\n".join(_chair_and_guest(i, g, L) for i, g in enumerate(cfg.guests))
    coasters = "\n".join(_coaster_and_card(i, L) for i in range(L.n_guests))

    xml = f"""<mujoco model="g1_roundtable_pepsi_service">
  <include file="{G1_NOKEY_NAME}"/>

  <option timestep="{L.timestep}" integrator="implicitfast"/>

  <visual>
    <global offwidth="1920" offheight="1080" azimuth="140" elevation="-20"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight diffuse="0.45 0.45 0.45" ambient="0.35 0.35 0.35" specular="0.3 0.3 0.3"/>
    <map znear="0.02" zfar="60" shadowclip="2" shadowscale="0.8"/>
    <rgba haze="0.75 0.82 0.92 1"/>
  </visual>

  <default>
    <default class="furniture">
      <geom contype="1" conaffinity="1" friction="0.9 0.005 0.0001" group="0"/>
    </default>
    <default class="person">
      <geom contype="0" conaffinity="0" group="0"/>
    </default>
    <default class="decor">
      <geom contype="0" conaffinity="0" group="0"/>
    </default>
    <default class="can">
      <geom contype="1" conaffinity="1" friction="0.8 0.005 0.0001" condim="4" solref="0.01 1" group="0"/>
    </default>
  </default>

  <asset>
    {chr(10).join('    ' + m for m in materials)}
  </asset>

  <statistic center="0 -1.5 1.0" extent="12"/>

  <worldbody>
    <light name="sun" pos="1.5 -2.5 6.5" dir="-0.2 0.35 -1" diffuse="0.85 0.85 0.82" specular="0.25 0.25 0.25" directional="true" castshadow="true"/>
    <light name="fill_a" pos="-5 4 4" dir="0.7 -0.55 -0.6" diffuse="0.28 0.28 0.32" specular="0.05 0.05 0.05" castshadow="false"/>
    <light name="fill_b" pos="5 3 4" dir="-0.7 -0.45 -0.6" diffuse="0.22 0.22 0.25" specular="0.05 0.05 0.05" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.1" material="floor" class="furniture"/>

{_room()}

{_cameras(L)}

{_table(L)}

{guests}

{coasters}

{_station(L)}

{cans_xml}
  </worldbody>
</mujoco>
"""
    return xml


def write_scene(cfg: ScenarioConfig, lowpoly: bool = True) -> Path:
    g1_dir = fetch()
    xml_path = g1_dir / SCENE_XML_NAME
    xml_path.write_text(build_scene_xml(cfg, g1_dir, lowpoly=lowpoly))
    return xml_path


def load_model(cfg: ScenarioConfig, lowpoly: bool = True):
    """Compile the scene. Returns (model, xml_path).

    The robot is driven kinematically (its joint positions are written every step),
    so its geoms are made non-colliding: contacts between a kinematic body and the
    furniture would only inject meaningless constraint forces.
    """
    import mujoco

    xml_path = write_scene(cfg, lowpoly=lowpoly)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    robot_bodies = set()
    for b in range(model.nbody):
        p = b
        while p != 0:
            if p == pelvis:
                robot_bodies.add(b)
                break
            p = model.body_parentid[p]
    for g in range(model.ngeom):
        if model.geom_bodyid[g] in robot_bodies:
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0
    return model, xml_path


if __name__ == "__main__":
    import mujoco

    cfg = ScenarioConfig()
    m, p = load_model(cfg)
    print(f"wrote {p}: nq={m.nq} nbody={m.nbody} ngeom={m.ngeom} ncam={m.ncam}")
