"""Run the G1 round-table drink-service simulation and render videos + screenshots.

Headless (CPU only, no GPU needed)::

    python run.py --out out/run1                # full 10-guest run, all videos
    python run.py --out out/quick --guests 2    # shorter smoke test
    python run.py --out out/run1 --skip-sim     # re-render an existing recording

Interactive on a desktop machine (needs a display; light enough for a laptop GPU)::

    python run.py --view

Outputs (in --out):
    trajectory.npz / meta.json          recorded run (replayable)
    screenshots/*.png                   key moments
    g1_roundtable_service_2x.mp4        director cut of the whole run, 2x speed, PiP overview
    g1_roundtable_first_guest_1x.mp4    first guest end-to-end in real time
    g1_roundtable_overview_4x.mp4       fixed overview camera time-lapse
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

os.environ.setdefault("MUJOCO_GL", "egl")


def simulate(cfg, out_dir: Path, max_guests, live_view: bool = False):
    from scenario import RoundTableScenario
    from scene import load_model

    model, xml_path = load_model(cfg)
    sc = RoundTableScenario(cfg, model, max_guests=max_guests)
    if live_view:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, sc.d) as viewer:
            gen = sc.script()
            t_wall0 = time.time()
            while viewer.is_running():
                try:
                    next(gen)
                except StopIteration:
                    break
                sc.step_once()
                viewer.sync()
                lag = sc.t - (time.time() - t_wall0)
                if lag > 0:
                    time.sleep(min(lag, 0.05))
        return sc, xml_path
    sc.run()
    sc.save(out_dir)
    return sc, xml_path


def screenshots(run_dir: Path, xml_path: Path, shots_dir: Path) -> None:
    from PIL import Image

    from render import CamParams, Director, Run, annotate_topdown, frame_at_time, render_frame

    run = Run.load(run_dir)
    shots_dir.mkdir(parents=True, exist_ok=True)
    d = Director(run)
    ev = {}
    for e in run.events:
        ev.setdefault(e["kind"], []).append(e)

    def save(name: str, i: int, cam: CamParams, overlays: bool = True, pip: bool = False, annotate=None):
        img = render_frame(run, xml_path, i, cam, overlays=overlays, pip=pip)
        if annotate is not None:
            img = annotate(img)
        Image.fromarray(img).save(shots_dir / name)
        print(f"[shots] {name}", flush=True)

    i0 = frame_at_time(run, 0.5)
    save("01_scene_overview.png", i0, CamParams("overview"), overlays=False)
    save("02_layout_topdown_annotated.png", i0, CamParams("topdown"), overlays=False,
         annotate=lambda img: annotate_topdown(img, run))
    if ev.get("ask"):
        i = frame_at_time(run, ev["ask"][0]["t"] + 1.6)
        save("03_asking_first_guest.png", i, d.target(i)[1], pip=True)
    if ev.get("grasp"):
        i = frame_at_time(run, ev["grasp"][0]["t"] - 0.05)
        save("04_grasping_can_closeup.png", i, d.target(i)[1], pip=True)
        # a tighter look at the hand, seen from the far side of the station
        _, cam = d.target(i)
        x, y, yaw = run.base_pose(i)
        save("05_grasp_hand_detail.png", i, CamParams(None, cam.lookat, 0.85, math.degrees(yaw) + 150, -22),
             overlays=False)
    if ev.get("release"):
        i = frame_at_time(run, ev["release"][0]["t"] + 0.6)
        save("06_placing_can_on_coaster.png", i, d.target(i)[1], pip=True)
        i = frame_at_time(run, ev["release"][0]["t"] + 2.4)
        save("06b_here_you_go.png", i, d.target(i)[1], pip=True)
    if ev.get("done"):
        i = frame_at_time(run, ev["done"][0]["t"] + 1.0)
        save("07_all_guests_served_overview.png", i, CamParams("overview"))
        save("08_all_guests_served_topdown.png", i, CamParams("topdown"), overlays=False)


def videos(run_dir: Path, xml_path: Path, which: list, workers: int, shadows: bool) -> None:
    from render import Run, frame_at_time, render_video

    run = Run.load(run_dir)
    n = len(run.frames)
    for name in which:
        if name == "director":
            render_video(run_dir, xml_path, run_dir / "g1_roundtable_service_2x.mp4", range(0, n, 2),
                         style="director", pip=True, workers=workers, shadows=shadows)
        elif name == "firstguest":
            served = [e for e in run.events if e["kind"] == "served"]
            t_end = served[0]["t"] + 5.0 if served else run.t[-1]
            render_video(run_dir, xml_path, run_dir / "g1_roundtable_first_guest_1x.mp4",
                         range(0, frame_at_time(run, t_end) + 1), style="director", pip=True, workers=workers,
                         shadows=shadows)
        elif name == "overview":
            render_video(run_dir, xml_path, run_dir / "g1_roundtable_overview_4x.mp4", range(0, n, 4),
                         style="overview", pip=False, workers=workers, title_seconds=2.0, shadows=shadows)
        else:
            raise SystemExit(f"unknown video '{name}' (use director, firstguest, overview or none)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "out" / "run"), help="output directory")
    ap.add_argument("--config", default=None, help="scenario json (guests, layout); default built-in 10 guests")
    ap.add_argument("--guests", type=int, default=None, help="only serve the first N guests (smoke tests)")
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--videos", default="director,firstguest,overview",
                    help="comma list of: director, firstguest, overview, none")
    ap.add_argument("--no-shadows", action="store_true", help="faster rendering")
    ap.add_argument("--skip-sim", action="store_true", help="reuse trajectory.npz in --out")
    ap.add_argument("--skip-shots", action="store_true")
    ap.add_argument("--view", action="store_true", help="interactive MuJoCo viewer instead of recording")
    args = ap.parse_args()

    from config import ScenarioConfig

    cfg = ScenarioConfig.load(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(out_dir / "scenario_used.json")

    if args.view:
        simulate(cfg, out_dir, args.guests, live_view=True)
        return
    if not args.skip_sim:
        _, xml_path = simulate(cfg, out_dir, args.guests)
    else:
        from scene import write_scene
        xml_path = write_scene(cfg)
    if not args.skip_shots:
        screenshots(out_dir, xml_path, out_dir / "screenshots")
    which = [v.strip() for v in args.videos.split(",") if v.strip() and v.strip() != "none"]
    if which:
        videos(out_dir, xml_path, which, args.workers, shadows=not args.no_shadows)


if __name__ == "__main__":
    main()
