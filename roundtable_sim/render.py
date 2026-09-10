"""Offline renderer for a recorded round-table run: MP4 videos and PNG screenshots.

Rendering is decoupled from simulation: ``scenario.py`` records qpos at 30 Hz plus
per-frame metadata; this module replays it through MuJoCo's offscreen renderer
with a small "director" that picks cameras from the current phase, draws dialog
bubbles / HUD with Pillow and encodes with ffmpeg.  Frames are split into chunks
rendered by parallel worker processes (software OpenGL is CPU bound).
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# make the worker processes deterministic about the GL backend
os.environ.setdefault("MUJOCO_GL", "egl")

WIDTH, HEIGHT = 1280, 720
PIP_SCALE = 0.27
PEPSI_BLUE = (14, 76, 154)
DIET_SILVER = (200, 205, 212)
ROBOT_BUBBLE = (24, 28, 36)
GUEST_BUBBLE = (250, 247, 238)


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
@dataclass
class Run:
    qpos: np.ndarray  # N x nq
    t: np.ndarray
    frames: List[dict]
    events: List[dict]
    guests: List[dict]
    layout: dict
    fps: int

    @staticmethod
    def load(run_dir: Path) -> "Run":
        z = np.load(run_dir / "trajectory.npz")
        meta = json.loads((run_dir / "meta.json").read_text())
        return Run(z["qpos"], z["t"], meta["frames"], meta["events"], meta["guests"], meta["layout"], meta["fps"])

    def base_pose(self, i: int) -> Tuple[float, float, float]:
        q = self.qpos[i]
        w, x, y, z = q[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(q[0]), float(q[1]), yaw

    def guest_xy(self, i: int) -> Tuple[float, float]:
        L = self.layout
        ang = L["first_guest_angle"] + i * 2 * math.pi / L["n_guests"]
        return L["chair_radius"] * math.cos(ang), L["chair_radius"] * math.sin(ang)


# --------------------------------------------------------------------------------------
# director: camera per frame
# --------------------------------------------------------------------------------------
@dataclass
class CamParams:
    fixed: Optional[str]  # name of a fixed camera, or None for a free camera
    lookat: Tuple[float, float, float] = (0.0, 0.0, 0.8)
    distance: float = 3.0
    azimuth: float = 90.0
    elevation: float = -20.0


def _ang_lerp(a: float, b: float, k: float) -> float:
    d = (b - a + 180.0) % 360.0 - 180.0
    return a + d * k


class Director:
    """Turns phase/focus metadata into smoothed free-camera parameters (hard cuts on mode change)."""

    def __init__(self, run: Run, style: str = "director"):
        self.run = run
        self.style = style
        self.prev_mode: Optional[str] = None
        self.state: Optional[CamParams] = None
        self.tau = 0.45  # seconds, for look-at / distance / elevation
        self.tau_az = 1.3  # slower azimuth so the camera does not whip around when the robot turns in place

    def target(self, i: int) -> Tuple[str, CamParams]:
        f = self.run.frames[i]
        phase = f["phase"]
        x, y, yaw = self.run.base_pose(i)
        yaw_deg = math.degrees(yaw)
        if self.style == "overview":
            return "overview", CamParams("overview")
        if phase in ("intro", "done"):
            return "overview", CamParams("overview")
        if phase == "walk":
            return "follow", CamParams(None, (x, y, 0.8), 3.1, yaw_deg + 28.0, -17.0)
        if phase == "ask":
            g = f["guest"]
            gx, gy = self.run.guest_xy(g)
            mid = ((x + gx) / 2, (y + gy) / 2, 1.0)
            az = math.degrees(math.atan2(gy, gx))  # look outward from the table towards the guest
            return "ask", CamParams(None, mid, 3.0, az + 12.0, -20.0)
        if phase == "grasp":  # from the far side of the station, facing the robot: hand + can unobstructed
            fx, fy, fz = f["focus"]
            return phase, CamParams(None, (fx, fy, fz + 0.03), 0.95, yaw_deg + 150.0, -32.0)
        if phase == "place":  # from above the table centre looking out: coaster, hand, robot and guest all visible
            fx, fy, fz = f["focus"]
            az_out = math.degrees(math.atan2(fy, fx))
            return phase, CamParams(None, (fx, fy, fz + 0.02), 1.45, az_out + 8.0, -33.0)
        if phase == "farewell":  # 3/4 view from the robot's right: gesture in profile, can and guest in frame
            return phase, CamParams(None, (x + 0.15 * math.cos(yaw), y + 0.15 * math.sin(yaw), 0.95), 2.0,
                                    yaw_deg + 110.0, -20.0)
        return "follow", CamParams(None, (x, y, 0.8), 3.1, yaw_deg + 28.0, -17.0)

    def step(self, i: int, dt: float) -> CamParams:
        mode, tgt = self.target(i)
        if tgt.fixed is not None or self.state is None or mode != self.prev_mode:
            self.state = tgt
            self.prev_mode = mode
            return tgt
        k = 1.0 - math.exp(-dt / self.tau)
        k_az = 1.0 - math.exp(-dt / self.tau_az)
        s = self.state
        la = tuple(s.lookat[j] + (tgt.lookat[j] - s.lookat[j]) * k for j in range(3))
        self.state = CamParams(None, la, s.distance + (tgt.distance - s.distance) * k,
                               _ang_lerp(s.azimuth, tgt.azimuth, k_az), s.elevation + (tgt.elevation - s.elevation) * k)
        self.prev_mode = mode
        return self.state


def plan_cameras(run: Run, frame_ids: Sequence[int], style: str) -> List[CamParams]:
    d = Director(run, style)
    out = []
    prev_t = None
    for i in frame_ids:
        t = float(run.t[i])
        dt = 1.0 / run.fps if prev_t is None else max(t - prev_t, 1e-3)
        prev_t = t
        out.append(d.step(i, dt))
    return out


# --------------------------------------------------------------------------------------
# overlays
# --------------------------------------------------------------------------------------
def _font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


PHASE_TEXT = {
    "intro": "greeting the table",
    "walk": "walking",
    "ask": "taking the order",
    "grasp": "picking up the can",
    "place": "placing the can on the coaster",
    "farewell": "drink served",
    "done": "all guests served",
}


def _wrap(draw, text: str, font, max_w: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_overlays(img: np.ndarray, run: Run, i: int, pip: Optional[np.ndarray], show_title: bool,
                  fonts: dict) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(img).convert("RGBA")
    W, H = im.size
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = run.frames[i]
    g = f["guest"]
    guest = run.guests[g] if 0 <= g < len(run.guests) else None

    # ---- HUD (top-left)
    pad = 14
    hud_lines = []
    if guest is not None:
        hud_lines.append((f"Guest {g + 1}/{len(run.guests)}: {guest['name']}  ·  wants {guest['order']}", fonts["hud_b"]))
    phase = f["phase"]
    ptxt = PHASE_TEXT.get(phase, phase)
    if phase == "walk" and guest is not None:
        if f["carrying"]:
            ptxt = f"carrying the {f['carrying']} to {guest['name']}'s seat"
        elif _after_order(run, i):
            ptxt = "walking to the drink station"
        else:
            ptxt = f"walking over to {guest['name']}"
    elif phase == "grasp" and guest is not None:
        ptxt = f"picking up a {guest['order']}"
    hud_lines.append((f"G1: {ptxt}", fonts["hud"]))
    hud_lines.append((f"sim time {f['t']:6.1f} s   ·   served {f['served']}/{len(run.guests)}", fonts["hud_s"]))
    wmax = max(d.textlength(t, font=ft) for t, ft in hud_lines)
    hh = sum(ft.size + 6 for _, ft in hud_lines) + 2 * pad - 6
    d.rounded_rectangle([pad, pad, pad + wmax + 2 * pad, pad + hh], radius=10, fill=(15, 18, 24, 175))
    yy = pad + pad - 2
    for t, ft in hud_lines:
        d.text((2 * pad, yy), t, font=ft, fill=(240, 240, 240, 255))
        yy += ft.size + 6

    # ---- progress chips (bottom-left)
    cx, cy = pad, H - pad - 34
    served_ids = [e["guest"] for e in run.events if e["kind"] == "served" and e["t"] <= f["t"] + 1e-6]
    for k, gg in enumerate(run.guests):
        ft = fonts["chip"]
        tw = d.textlength(gg["name"], font=ft) + 18
        done = k in served_ids
        active = (k == g)
        is_pepsi = gg["order"] == "Pepsi"
        col = PEPSI_BLUE if is_pepsi else DIET_SILVER
        fill = (*col, 240) if done else (30, 34, 42, 170)
        outline = (255, 255, 255, 255) if active else (255, 255, 255, 60)
        d.rounded_rectangle([cx, cy, cx + tw, cy + 30], radius=8, fill=fill, outline=outline, width=2)
        if done:
            tcol = (255, 255, 255, 255) if is_pepsi else (20, 40, 90, 255)
        else:
            tcol = (255, 255, 255, 255) if active else (200, 200, 200, 255)
        d.text((cx + 9, cy + 6), gg["name"], font=ft, fill=tcol)
        cx += tw + 8

    # ---- dialog bubble
    if f["speaker"]:
        robot = f["speaker"] == "robot"
        name = "G1" if robot else (guest["name"] if guest else "Guest")
        ft, ft_n = fonts["dialog"], fonts["hud_b"]
        max_w = int(W * 0.46)
        lines = _wrap(d, f["text"], ft, max_w - 2 * pad)
        bw = max([d.textlength(l, font=ft) for l in lines] + [d.textlength(name, font=ft_n)]) + 2 * pad
        bh = len(lines) * (ft.size + 6) + ft_n.size + 2 * pad + 4
        bx = pad if robot else W - pad - bw
        by = H - pad - 34 - 24 - bh
        fill = (*ROBOT_BUBBLE, 225) if robot else (*GUEST_BUBBLE, 235)
        tcol = (245, 245, 245, 255) if robot else (30, 30, 30, 255)
        ncol = (120, 190, 255, 255) if robot else (200, 60, 60, 255)
        d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=14, fill=fill)
        # little tail
        tail_x = bx + 40 if robot else bx + bw - 40
        d.polygon([(tail_x - 12, by + bh), (tail_x + 12, by + bh), (tail_x, by + bh + 16)], fill=fill)
        d.text((bx + pad, by + pad - 2), name, font=ft_n, fill=ncol)
        yy = by + pad + ft_n.size + 4
        for l in lines:
            d.text((bx + pad, yy), l, font=ft, fill=tcol)
            yy += ft.size + 6

    # ---- title card
    if show_title:
        ft = fonts["title"]
        title = "Unitree G1 · round-table drink service"
        sub = "MuJoCo simulation · scripted whole-body control · 10 guests, Pepsi or Diet Pepsi"
        tw = d.textlength(title, font=ft)
        sw = d.textlength(sub, font=fonts["hud"])
        bw = max(tw, sw) + 60
        bx = (W - bw) / 2
        d.rounded_rectangle([bx, H * 0.38, bx + bw, H * 0.38 + ft.size + fonts["hud"].size + 50], radius=16,
                            fill=(15, 18, 24, 200))
        d.text(((W - tw) / 2, H * 0.38 + 18), title, font=ft, fill=(255, 255, 255, 255))
        d.text(((W - sw) / 2, H * 0.38 + 30 + ft.size), sub, font=fonts["hud"], fill=(200, 210, 230, 255))

    im = Image.alpha_composite(im, layer)

    # ---- picture in picture
    if pip is not None:
        pim = Image.fromarray(pip).convert("RGBA")
        pw, ph = int(W * PIP_SCALE), int(W * PIP_SCALE * pip.shape[0] / pip.shape[1])
        pim = pim.resize((pw, ph), Image.BILINEAR)
        px, py = W - pad - pw, pad  # top-right, clear of the guest bubble
        frame = Image.new("RGBA", (pw + 6, ph + 6), (255, 255, 255, 220))
        im.paste(frame, (px - 3, py - 3), frame)
        im.paste(pim, (px, py))
        dd = ImageDraw.Draw(im)
        dd.rounded_rectangle([px + 6, py + 6, px + 96, py + 26], radius=6, fill=(15, 18, 24, 190))
        dd.text((px + 12, py + 8), "overview", font=fonts["chip"], fill=(255, 255, 255, 255))
    return np.asarray(im.convert("RGB"))


def _after_order(run: Run, i: int) -> bool:
    """True if the current guest has already been asked (so a walk is towards the station)."""
    f = run.frames[i]
    g = f["guest"]
    asked = [e for e in run.events if e["kind"] == "ask" and e.get("guest") == g and e["t"] <= f["t"]]
    return bool(asked)


def make_fonts() -> dict:
    return {"hud": _font(20), "hud_b": _font(22), "hud_s": _font(16), "chip": _font(15),
            "dialog": _font(24), "title": _font(40)}


def annotate_topdown(img: np.ndarray, run: Run) -> np.ndarray:
    """Label guests, station and robot on a frame from the fixed 'topdown' camera.

    The topdown camera is axis aligned (pos (0,-0.9,9.5), looking straight down,
    fovy 55, image up = +y), so the projection is a simple scale about its centre.
    """
    from PIL import Image, ImageDraw

    im = Image.fromarray(img).convert("RGBA")
    W, H = im.size
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cam_xy, cam_z, fovy = (0.0, -0.9), 9.5, 55.0
    L = run.layout

    def proj(x, y, z):
        f = (H / 2) / math.tan(math.radians(fovy / 2)) / (cam_z - z)
        return W / 2 + (x - cam_xy[0]) * f, H / 2 - (y - cam_xy[1]) * f

    ft, ft_s, ft_t = _font(18), _font(15), _font(26)
    for i, g in enumerate(run.guests):
        gx, gy = run.guest_xy(i)
        ang = math.atan2(gy, gx)
        lx, ly = proj((L["chair_radius"] + 0.55) * math.cos(ang), (L["chair_radius"] + 0.55) * math.sin(ang), 0.5)
        label = f"{i + 1}. {g['name']}"
        sub = g["order"]
        tw = max(d.textlength(label, font=ft), d.textlength(sub, font=ft_s)) + 14
        col = PEPSI_BLUE if g["order"] == "Pepsi" else (90, 96, 104)
        d.rounded_rectangle([lx - tw / 2, ly - 22, lx + tw / 2, ly + 22], radius=8, fill=(*col, 225))
        d.text((lx - d.textlength(label, font=ft) / 2, ly - 20), label, font=ft, fill=(255, 255, 255, 255))
        d.text((lx - d.textlength(sub, font=ft_s) / 2, ly + 2), sub, font=ft_s, fill=(235, 235, 235, 255))
    # station
    sa = L["station_angle"]
    sx, sy = L["station_center_dist"] * math.cos(sa), L["station_center_dist"] * math.sin(sa)
    px, py = proj(sx, sy, 0.85)
    st = "drink station: 6 Pepsi | 6 Diet Pepsi"
    stw = d.textlength(st, font=ft) + 20
    d.rounded_rectangle([px - stw / 2, py + 40, px + stw / 2, py + 68], radius=8, fill=(20, 20, 20, 220))
    d.text((px - stw / 2 + 10, py + 44), st, font=ft, fill=(255, 255, 255, 255))
    # robot
    x, y, yaw = run.base_pose(0)
    rx, ry = proj(x, y, 0.7)
    d.ellipse([rx - 12, ry - 12, rx + 12, ry + 12], outline=(255, 80, 80, 255), width=3)
    d.text((rx + 16, ry - 10), "G1 start", font=ft, fill=(200, 30, 30, 255))
    # legend / title
    title = f"Top view: {len(run.guests)} guests, coaster to each guest's right, G1 walks the ring road"
    tw = d.textlength(title, font=ft_t)
    d.rounded_rectangle([14, 14, 30 + tw, 58], radius=10, fill=(15, 18, 24, 200))
    d.text((22, 20), title, font=ft_t, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(im, layer).convert("RGB"))


# --------------------------------------------------------------------------------------
# rendering workers
# --------------------------------------------------------------------------------------
class SceneRenderer:
    def __init__(self, xml_path: str, width: int = WIDTH, height: int = HEIGHT, shadows: bool = True):
        import mujoco
        self.mujoco = mujoco
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.shadows = shadows

    def render(self, qpos: np.ndarray, cam: CamParams) -> np.ndarray:
        mj = self.mujoco
        self.d.qpos[:] = qpos
        mj.mj_forward(self.m, self.d)
        if cam.fixed is not None:
            self.cam.type = mj.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_CAMERA, cam.fixed)
        else:
            self.cam.type = mj.mjtCamera.mjCAMERA_FREE
            self.cam.lookat[:] = cam.lookat
            self.cam.distance = cam.distance
            self.cam.azimuth = cam.azimuth
            self.cam.elevation = cam.elevation
        self.r.update_scene(self.d, camera=self.cam)
        self.r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 1 if self.shadows else 0
        return self.r.render()

    def close(self):
        self.r.close()


def _worker_render_chunk(args) -> str:
    (xml_path, run_dir, frame_ids, cams, pip_cams, out_path, fps, title_frames, overlays, width, height,
     shadows) = args
    import imageio.v2 as imageio

    run = Run.load(Path(run_dir))
    main = SceneRenderer(xml_path, width, height, shadows=shadows)
    pip_r = SceneRenderer(xml_path, 640, 360, shadows=False) if pip_cams is not None else None
    fonts = make_fonts() if overlays else None
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                                ffmpeg_log_level="error", output_params=["-crf", "23", "-preset", "medium"])
    try:
        for k, i in enumerate(frame_ids):
            img = main.render(run.qpos[i], cams[k])
            pip = pip_r.render(run.qpos[i], pip_cams[k]) if pip_r is not None else None
            if overlays:
                img = draw_overlays(img, run, i, pip, show_title=(i in title_frames), fonts=fonts)
            writer.append_data(img)
    finally:
        writer.close()
        main.close()
        if pip_r is not None:
            pip_r.close()
    return out_path


def render_video(run_dir: Path, xml_path: Path, out_path: Path, frame_ids: Sequence[int], style: str = "director",
                 pip: bool = True, overlays: bool = True, workers: int = 4, fps: int = 30, title_seconds: float = 3.0,
                 width: int = WIDTH, height: int = HEIGHT, shadows: bool = True) -> Path:
    run = Run.load(run_dir)
    frame_ids = list(frame_ids)
    cams = plan_cameras(run, frame_ids, style)
    pip_cams = [CamParams("overview")] * len(frame_ids) if (pip and style != "overview") else None
    title_frames = set(frame_ids[: int(title_seconds * fps)]) if title_seconds > 0 else set()
    n = len(frame_ids)
    workers = max(1, min(workers, n))
    chunk = int(math.ceil(n / workers))
    tmp_dir = out_path.parent / f".chunks_{out_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for w in range(workers):
        sl = slice(w * chunk, min((w + 1) * chunk, n))
        if sl.start >= n:
            break
        jobs.append((str(xml_path), str(run_dir), frame_ids[sl], cams[sl],
                     None if pip_cams is None else pip_cams[sl], str(tmp_dir / f"chunk_{w:02d}.mp4"), fps,
                     title_frames, overlays, width, height, shadows))
    t0 = time.time()
    print(f"[render] {out_path.name}: {n} frames on {len(jobs)} workers ...", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(jobs)) as pool:
        chunks = pool.map(_worker_render_chunk, jobs)
    lst = tmp_dir / "list.txt"
    lst.write_text("".join(f"file '{Path(c).resolve()}'\n" for c in chunks))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(out_path)], check=True)
    for c in chunks:
        os.remove(c)
    lst.unlink()
    tmp_dir.rmdir()
    print(f"[render] wrote {out_path} ({n} frames, {time.time() - t0:.0f}s)", flush=True)
    return out_path


def render_frame(run: Run, xml_path: Path, i: int, cam: CamParams, overlays: bool = True, pip: bool = False,
                 width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    r = SceneRenderer(str(xml_path), width, height)
    img = r.render(run.qpos[i], cam)
    pip_img = None
    if pip:
        pr = SceneRenderer(str(xml_path), 640, 360, shadows=False)
        pip_img = pr.render(run.qpos[i], CamParams("overview"))
        pr.close()
    r.close()
    if overlays:
        img = draw_overlays(img, run, i, pip_img, show_title=False, fonts=make_fonts())
    return img


def frame_at_time(run: Run, t: float) -> int:
    return int(np.clip(np.searchsorted(run.t, t), 0, len(run.t) - 1))
