#!/usr/bin/env python3
"""Read-only G1 camera inventory. Does not move the robot.

Checks USB, /dev/video*, Intel RealSense, CSI/Argus, and DDS rt/frontvideostream.
If any source produces a frame, writes a JPEG. Safe to run while other vision
daemons (e.g. /tmp/g1-vision-stream.py) are waiting for a device.

Run on the Jetson:
  ~/miniforge3/envs/g1brainco/bin/python g1_camera_probe.py --out /tmp/g1-camera.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--iface", default="eth0")
parser.add_argument("--domain", type=int, default=0)
parser.add_argument("--seconds", type=float, default=2.0)
parser.add_argument("--out", default="/tmp/g1-camera.json")
args = parser.parse_args()

rep = {"started": datetime.now().isoformat(timespec="seconds"), "host": os.uname()[1], "sources": {}}


def run(cmd, timeout=4):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, text=True)
        return p.returncode, p.stdout
    except Exception as e:
        return 1, str(e)


def save_jpg(path, bgr):
    try:
        import cv2
        cv2.imwrite(path, bgr)
        return path
    except Exception as e:
        return "write failed: %s" % e


# USB
code, out = run(["lsusb"])
rep["usb"] = [ln.strip() for ln in out.splitlines() if ln.strip()]
rep["usb_videoish"] = [ln for ln in rep["usb"] if any(k in ln.lower() for k in ("intel", "realsense", "uvc", "camera", "webcam", "orbbec"))]

# V4L
code, out = run(["bash", "-lc", "ls -l /dev/video* /dev/media* 2>/dev/null; echo '---'; v4l2-ctl --list-devices 2>/dev/null"])
rep["v4l"] = out.strip()
videos = [p for p in ["/dev/video%d" % i for i in range(16)] if os.path.exists(p)]
rep["video_nodes"] = videos

# RealSense
try:
    import pyrealsense2 as rs
    ctx = rs.context()
    devs = list(ctx.devices)
    rep["realsense"] = [{"name": d.get_info(rs.camera_info.name), "serial": d.get_info(rs.camera_info.serial_number)} for d in devs]
except Exception as e:
    code, out = run(["rs-enumerate-devices"])
    rep["realsense"] = {"error": str(e), "enumerate": out.strip()[:500]}

# CSI / Argus
code, out = run(["gst-launch-1.0", "-q", "nvarguscamerasrc", "sensor-id=0", "num-buffers=1", "!", "fakesink"], timeout=6)
rep["argus"] = out.strip()[-400:]

# OpenCV grab of any /dev/video*
frames = []
if videos:
    try:
        import cv2
        for node in videos:
            cap = cv2.VideoCapture(node)
            ok, img = cap.read()
            cap.release()
            rec = {"node": node, "ok": bool(ok), "shape": list(img.shape) if ok else None}
            if ok:
                rec["jpeg"] = save_jpg(args.out.replace(".json", "-%s.jpg" % os.path.basename(node)), img)
            frames.append(rec)
    except Exception as e:
        rep["opencv_error"] = str(e)
rep["opencv_grabs"] = frames

# DDS front video (Unitree videohub_pc4 publishes this when /dev/video4 exists)
dds = {"topic": "rt/frontvideostream", "count": 0, "first": None}
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import Go2FrontVideoData_
    ChannelFactoryInitialize(args.domain, args.iface)
    n = {"c": 0, "msg": None}

    def on(msg):
        n["c"] += 1
        if n["msg"] is None:
            n["msg"] = {
                "time_frame": int(msg.time_frame),
                "len720": len(msg.video720p),
                "len360": len(msg.video360p),
                "len180": len(msg.video180p),
            }

    sub = ChannelSubscriber("rt/frontvideostream", Go2FrontVideoData_)
    sub.Init(on, 0)
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        time.sleep(0.05)
    dds["count"] = n["c"]
    dds["first"] = n["msg"]
    dds["rate_hz"] = n["c"] / args.seconds if args.seconds else 0
except Exception as e:
    dds["error"] = str(e)
rep["dds_frontvideostream"] = dds

rep["conclusion"] = (
    "no camera hardware on this Jetson"
    if not videos and not (isinstance(rep.get("realsense"), list) and rep["realsense"]) and dds["count"] == 0
    else "camera present"
)
print(json.dumps({k: rep[k] for k in ("started", "video_nodes", "realsense", "dds_frontvideostream", "conclusion")}, indent=2))
with open(args.out, "w") as f:
    json.dump(rep, f, indent=2)
print("G1_CAMERA_OUT", args.out)
os._exit(0)
