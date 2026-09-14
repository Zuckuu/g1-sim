"""G1 vision dashboard: RGB + depth MJPEG, plus a metric snapshot for look/IK.

Run on the Jetson with the g1brainco env:
    ~/miniforge3/envs/g1brainco/bin/python g1_vision_stream.py [--port 8080]

Source priority: RealSense D435i (RGB+depth, depth aligned to color) -> /dev/video0
(RGB only) -> waiting page. Hot-plug friendly.

Serves:
    /            dashboard (RGB + depth colormap)
    /rgb.mjpg    MJPEG color
    /depth.mjpg  MJPEG depth colormap
    /status      JSON health
    /color.jpg   latest color frame
    /depth.f32   latest aligned depth, little-endian float32 meters, HxW
    /calib.json  color intrinsics (aligned depth uses the same)

Read-only. Safe to leave running. `g1_arm_can_test.py --stage look` reads /color.jpg
+/depth.f32+/calib.json so it does not steal the device.
"""
import argparse
import glob
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content=width=device-width,initial-scale=1>
<title>G1 vision</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;margin:0;padding:12px}
h1{font-size:18px;margin:0 0 8px}#meta{color:#9ab;font-size:13px;margin-bottom:8px}
.row{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0}figcaption{font-size:13px;color:#9ab}
img{background:#000;max-width:46vw;border:1px solid #333}</style></head><body>
<h1>G1 vision</h1><div id=meta>connecting…</div>
<div class=row><figure><img id=rgb src=/rgb.mjpg><figcaption>color</figcaption></figure>
<figure><img id=depth src=/depth.mjpg><figcaption>depth 0.3–3 m (aligned)</figcaption></figure></div>
<script>setInterval(async()=>{try{const r=await fetch('/status');const s=await r.json();
document.getElementById('meta').textContent=s.device?
(s.source+' '+s.fps.toFixed(1)+' fps '+s.w+'x'+s.h+' | '+s.ts):
('NO CAMERA ('+s.source+') — plug into Jetson USB…');}catch(e){}},2000);</script>
</body></html>"""

WAIT_PAGE = """<!doctype html><html><head><meta charset=utf-8><meta http-equiv=refresh content=5>
<title>G1 vision</title></head><body style="background:#111;color:#eee;font-family:sans-serif">
<h1>No camera detected</h1><p>Plug a webcam / RealSense into a Jetson USB port —
streaming starts on its own.</p><p>Retrying…</p></body></html>"""


def no_depth_slate():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "NO DEPTH SOURCE", (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120, 120, 120), 2)
    ok, jpg = cv2.imencode(".jpg", img)
    return jpg.tobytes() if ok else None


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.rgb = None
        self.depth = None
        self.color_bgr = None
        self.depth_m = None
        self.calib = None
        self.device = False
        self.source = "none"
        self.frames = 0
        self.t0 = time.time()
        self.color_fn = -1
        self.depth_fn = -1

    def fps(self):
        dt = time.time() - self.t0
        return self.frames / dt if dt > 1 else 0.0

    def mark_live(self, source):
        with self.lock:
            self.device = True
            self.source = source
            self.frames = 0
            self.t0 = time.time()


def depth_colormap(depth_m):
    d = np.clip(depth_m, 0.3, 3.0)
    norm = ((d - 0.3) / 2.7 * 255).astype(np.uint8)
    norm[~(np.isfinite(d)) | (d <= 0.3001)] = 0
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def encode(jpg_quality, *imgs):
    out = []
    for img in imgs:
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        out.append(jpg.tobytes() if ok else None)
    return out


def run_realsense(state):
    import pyrealsense2 as rs

    # One pipeline. Color 6 / depth 15 is the USB2 ceiling on this Jetson
    # (firmware rejects 10/15). Align depth onto color so /depth.f32 is metric
    # in the color camera frame (d435_link / look).
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    cintr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    calib = dict(fx=float(cintr.fx), fy=float(cintr.fy), cx=float(cintr.ppx), cy=float(cintr.ppy),
                 w=int(cintr.width), h=int(cintr.height), aligned_to="color")
    with state.lock:
        state.calib = calib
    try:
        state.mark_live("realsense")
        print("vision: realsense streaming (color6+depth15, aligned to color)", flush=True)
        last_color_fn = -1
        while True:
            frames = align.process(pipe.wait_for_frames(5000))
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if cf is not None and cf.frame_number != last_color_fn:
                last_color_fn = cf.frame_number
                bgr = np.asanyarray(cf.get_data())
                jpg = encode(70, bgr)[0]
                if jpg:
                    with state.lock:
                        state.rgb = jpg
                        state.color_bgr = bgr
                        state.color_fn = last_color_fn
                        state.frames += 1
            if df is not None:
                depth = np.asanyarray(df.get_data()).astype(np.float32) * float(df.get_units())
                jpg = encode(70, depth_colormap(depth))[0]
                if jpg:
                    with state.lock:
                        state.depth = jpg
                        state.depth_m = depth
                        state.depth_fn = df.frame_number
                        state.frames += 1
    finally:
        try:
            pipe.stop()
        except Exception:
            pass


def run_webcam(state):
    devs = sorted(glob.glob("/dev/video*"))
    if not devs:
        raise RuntimeError("no /dev/video*")
    cap = cv2.VideoCapture(devs[0], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    if not cap.isOpened():
        raise RuntimeError("%s would not open" % devs[0])
    try:
        slate = no_depth_slate()
        state.mark_live("webcam:%s" % devs[0])
        print("vision: webcam %s streaming (no depth)" % devs[0], flush=True)
        with state.lock:
            state.depth = slate
            state.depth_m = None
            state.calib = dict(fx=0, fy=0, cx=0, cy=0, w=640, h=480, aligned_to="none")
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("%s read failed (unplugged?)" % devs[0])
            jpg = encode(70, frame)[0]
            if jpg:
                with state.lock:
                    state.rgb = jpg
                    state.color_bgr = frame
                    state.frames += 1
    finally:
        cap.release()


def capture_loop(state):
    while True:
        try:
            try:
                run_realsense(state)
            except Exception as rs_err:
                print("vision: realsense unavailable (%s), trying webcam" % rs_err, flush=True)
                run_webcam(state)
        except Exception as e:
            with state.lock:
                state.device = False
            print("vision: %s (retry in 5s)" % e, flush=True)
            time.sleep(5)


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _mjpg(self, kind):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while True:
                with state.lock:
                    frame = state.rgb if kind == "rgb" else state.depth
                    ok = state.device and frame is not None
                if not ok:
                    time.sleep(0.5)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.066)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            with state.lock:
                dev = state.device
            if path == "/":
                self._send(PAGE if dev else WAIT_PAGE, "text/html")
            elif path == "/status":
                with state.lock:
                    self._send(json.dumps({
                        "device": state.device, "source": state.source,
                        "fps": state.fps(), "w": 640, "h": 480,
                        "color_fn": state.color_fn, "depth_fn": state.depth_fn,
                        "has_depth_m": state.depth_m is not None,
                        "ts": time.strftime("%H:%M:%S"),
                    }), "application/json")
            elif path == "/calib.json":
                with state.lock:
                    calib = state.calib
                if not calib:
                    self.send_response(503)
                    self.end_headers()
                    return
                self._send(json.dumps(calib), "application/json")
            elif path == "/color.jpg":
                with state.lock:
                    jpg = state.rgb
                if not jpg:
                    self.send_response(503)
                    self.end_headers()
                    return
                self._send(jpg, "image/jpeg")
            elif path == "/depth.f32":
                with state.lock:
                    depth = None if state.depth_m is None else state.depth_m.astype("<f4", copy=False)
                if depth is None:
                    self.send_response(503)
                    self.end_headers()
                    return
                self._send(np.ascontiguousarray(depth).tobytes(), "application/octet-stream")
            elif path in ("/rgb.mjpg", "/depth.mjpg") and dev:
                try:
                    self._mjpg("rgb" if path == "/rgb.mjpg" else "depth")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(404)
                self.end_headers()

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    state = State()
    threading.Thread(target=capture_loop, args=(state,), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(state))
    print("vision: serving on :%d" % args.port, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
