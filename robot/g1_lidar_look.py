#!/usr/bin/env python3
"""G1 LiDAR look (read-only): find a table top and can-sized objects on it with the head Livox Mid-360.

Accumulates ~1 s of rt/utlidar/cloud_livox_mid360, moves the points into the pelvis LEVEL frame (URDF mount pose
mid360_link via pinocchio FK on the live joint state, IMU roll/pitch removed; x forward, y left, z up, origin at the
pelvis), then:
  * floor plane (sanity: must sit at -pelvis height, ~-0.79 m - the sensor is mounted upside down in the URDF),
  * horizontal planes at table height (z_level in --table-z), the biggest one is the table: extent, nearest edge
    distance, edge line yaw in the body frame (same convention as the camera LOOK: the robot squares up by turning -yaw),
  * clusters standing on the table top (3 cm grid connected components) with can-like size (height 0.07-0.17 m,
    footprint 0.03-0.13 m) -> candidates sorted by distance.
Output: log lines + JSON (--out). No robot command of any kind.
"""
import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--iface", default="eth0")
parser.add_argument("--seconds", type=float, default=1.0, help="cloud accumulation time")
parser.add_argument("--urdf", default=os.path.expanduser("~/unitree/g1_description/g1_29dof_rev_1_0.urdf"))
parser.add_argument("--range", type=float, default=4.0, help="m: ignore points farther than this (xy)")
parser.add_argument("--table-z", default="-0.20,0.30", help="level-frame z band for table tops (pelvis ~0.79 m above the floor; the sensor's own height ring starts ~+0.32)")
parser.add_argument("--min-table-pts", type=int, default=150)
parser.add_argument("--out", default="/tmp/g1-lidar-look.json")
parser.add_argument("--dump", default=None, help="also save the level-frame points (npz) for offline analysis")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

T0 = time.monotonic()


def log(msg):
    print("[%7.2f] %s" % (time.monotonic() - T0, msg), flush=True)


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


# ---------------------------------------------------------------------------------------------------------- DDS in
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_  # noqa: E402

ChannelFactoryInitialize(0, args.iface)
LOCK = threading.Lock()
FRAMES = []
LOW = {"q": None, "rpy": None, "n": 0}
PT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "<f4"), ("ring", "<u2"), ("t", "<f4")])


def on_cloud(m):
    with LOCK:
        if len(FRAMES) < 60:
            FRAMES.append((bytes(m.data), int(m.point_step), int(m.width) * int(m.height), str(m.header.frame_id)))


def on_low(m):
    with LOCK:
        if LOW["n"] % 10 == 0:      # 1 kHz topic: sample it
            ms = m.motor_state
            LOW["q"] = np.array([ms[i].q for i in range(29)])
            LOW["rpy"] = np.array(list(m.imu_state.rpy)[:3])
        LOW["n"] += 1


sub_c = ChannelSubscriber("rt/utlidar/cloud_livox_mid360", PointCloud2_)
sub_c.Init(on_cloud, 0)
sub_l = ChannelSubscriber("rt/lowstate", LowState_)
sub_l.Init(on_low, 0)
time.sleep(max(0.6, args.seconds))
with LOCK:
    frames = list(FRAMES)
    q, rpy = LOW["q"], LOW["rpy"]
if not frames:
    log("no LiDAR frames on rt/utlidar/cloud_livox_mid360 (lidar_driver running?)")
    sys.exit(2)
if q is None:
    log("no rt/lowstate")
    sys.exit(2)

pts = []
for data, step, n, frame in frames:
    if step != PT_DTYPE.itemsize:
        log("unexpected point_step %d (expected %d)" % (step, PT_DTYPE.itemsize))
        sys.exit(2)
    a = np.frombuffer(data, dtype=PT_DTYPE, count=n)
    pts.append(np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64))
P_s = np.concatenate(pts, axis=0)
P_s = P_s[np.isfinite(P_s).all(axis=1)]
P_s = P_s[np.linalg.norm(P_s, axis=1) > 0.05]
log("LIDAR: %d frames, %d points, frame '%s'" % (len(frames), P_s.shape[0], frames[0][3]))

# ------------------------------------------------------------------------------------------------ sensor -> level frame
import pinocchio as pin  # noqa: E402

model = pin.buildModelFromUrdf(args.urdf)
data = model.createData()
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
M = data.oMf[model.getFrameId("mid360_link")]
R_m, t_m = M.rotation, M.translation
R_level = rpy_to_mat(float(rpy[0]), float(rpy[1]), 0.0)           # pelvis -> level (yaw kept)
P_p = (R_m @ P_s.T).T + t_m
P = (R_level @ P_p.T).T
sensor_L = R_level @ t_m
log("LIDAR: sensor at (level) %s m; IMU roll/pitch %.1f/%.1f deg" % (np.round(sensor_L, 3).tolist(), math.degrees(rpy[0]), math.degrees(rpy[1])))
rxy = np.hypot(P[:, 0], P[:, 1])
P = P[(rxy < args.range) & (rxy > 0.15)]
rxy = np.hypot(P[:, 0], P[:, 1])
out = dict(ok=False, n_points=int(P.shape[0]), sensor_level=np.round(sensor_L, 3).tolist(), frames=len(frames))
if args.dump:
    np.savez_compressed(args.dump, P=P)
    log("LIDAR: level-frame points saved to %s" % args.dump)

# floor sanity
low = P[(P[:, 2] < -0.45) & (P[:, 2] > -1.2)]
if low.shape[0] > 200:
    h, e = np.histogram(low[:, 2], bins=np.arange(-1.2, -0.45, 0.01))
    zf = float(e[int(np.argmax(h))] + 0.005)
    out["floor_z_level"] = round(zf, 3)
    log("LIDAR: floor plane at z_level %.3f (%d pts) -> pelvis height %.2f m%s" % (
        zf, int(h.max()), -zf, "" if 0.65 < -zf < 0.95 else "  ** unexpected: check the mount pose / frame **"))
else:
    log("LIDAR: no floor points below the pelvis (%d) - mount pose or frame convention wrong?" % low.shape[0])

# Measured 2026-09-14 on this G1: the driver publishes nothing closer than ~1.0 m (range histogram starts at 1.0),
# the forward sector has ~3x fewer points than the sides, and a glossy counter TOP returns almost nothing at the
# 9-13 deg grazing angles we get from 1-2 m. What the sensor sees well is the counter's vertical FRONT FACE (floor
# to counter height) - so that is the primary target: distance, yaw and height of the counter from range.
out["sensor_note"] = "points < 1.0 m are cropped by the driver; glossy table tops return little - the front face is the target"
zf = out.get("floor_z_level", -0.78)
az_all = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
F = P[(P[:, 2] > zf + 0.15) & (P[:, 2] < 0.40) & (np.abs(az_all) < 50) & (rxy > 0.9) & (rxy < 3.5)]
counter = None
if F.shape[0] >= 200:
    # vertical-face cells: 5 cm xy cells with >= 15 points spanning >= 0.25 m in z
    cell = 0.05
    fx0, fy0 = float(F[:, 0].min()), float(F[:, 1].min())
    ci = ((F[:, 0] - fx0) / cell).astype(int)
    cj = ((F[:, 1] - fy0) / cell).astype(int)
    key = ci * 100000 + cj
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    zmin = np.full(len(uniq), np.inf)
    zmax = np.full(len(uniq), -np.inf)
    np.minimum.at(zmin, inv, F[:, 2])
    np.maximum.at(zmax, inv, F[:, 2])
    good = (cnt >= 15) & ((zmax - zmin) >= 0.25)
    cells_xy = np.stack([fx0 + (uniq[good] // 100000) * cell + cell / 2, fy0 + (uniq[good] % 100000) * cell + cell / 2], axis=1)
    if cells_xy.shape[0] >= 6:
        # RANSAC line through the face cells (x = a + b*y); support = cells within 4 cm; prefer the nearest well-supported line
        rng = np.random.RandomState(1)
        best = None
        n_c = cells_xy.shape[0]
        for _ in range(300):
            i, j = rng.choice(n_c, 2, replace=False)
            dy_ = cells_xy[j, 1] - cells_xy[i, 1]
            if abs(dy_) < 0.08:
                continue
            b = (cells_xy[j, 0] - cells_xy[i, 0]) / dy_
            if abs(b) > 1.0:          # more than 45 deg off square: not the counter we are facing
                continue
            a = cells_xy[i, 0] - b * cells_xy[i, 1]
            res = np.abs(cells_xy[:, 0] - (a + b * cells_xy[:, 1])) / math.sqrt(1 + b * b)
            sup = int((res < 0.04).sum())
            dist = a / math.sqrt(1 + b * b)
            score = sup - 4.0 * max(0.0, dist - 1.0)      # a nearer face with a few cells less still wins
            if sup >= 6 and (best is None or score > best[0]):
                best = (score, a, b, sup)
        if best is not None:
            _, a, b, sup = best
            m_line = np.abs(cells_xy[:, 0] - (a + b * cells_xy[:, 1])) / math.sqrt(1 + b * b) < 0.04
            A = np.stack([np.ones(int(m_line.sum())), cells_xy[m_line, 1]], axis=1)
            a, b = np.linalg.lstsq(A, cells_xy[m_line, 0], rcond=None)[0]
            nrm = math.sqrt(1 + b * b)
            dist = float(a / nrm)
            yaw = math.degrees(math.atan(float(b)))
            y_lo, y_hi = float(cells_xy[m_line, 1].min()), float(cells_xy[m_line, 1].max())
            d_all = (P[:, 0] - (a + b * P[:, 1])) / nrm                 # signed distance behind the face line
            on_face = (np.abs(d_all) < 0.06) & (P[:, 1] > y_lo - 0.05) & (P[:, 1] < y_hi + 0.05) & (P[:, 2] > zf + 0.1) & (P[:, 2] < 0.45)
            top_z = float(np.percentile(P[on_face, 2], 99)) if int(on_face.sum()) > 30 else None
            counter = dict(counter_dist=round(dist, 3), counter_yaw_deg=round(yaw, 1), counter_cells=int(m_line.sum()),
                           counter_y_extent=[round(y_lo, 2), round(y_hi, 2)], counter_width_txt="%.2f m" % (y_hi - y_lo))
            if top_z is not None:
                above = (d_all > 0.0) & (d_all < 0.35) & (P[:, 1] > y_lo) & (P[:, 1] < y_hi) & (P[:, 2] > top_z + 0.06) & (P[:, 2] < top_z + 0.40)
                n_above = int(above.sum())
                counter.update(counter_top_z=round(top_z, 3), counter_height_m=round(top_z - zf, 2), n_above_top=n_above,
                               counter_like=bool(0.65 <= top_z - zf <= 1.15 and n_above < 0.15 * int(on_face.sum())))
                # can hint: clusters 3-20 cm above the top, 2-60 cm behind the face, within its width
                Cz = P[(d_all > 0.02) & (d_all < 0.60) & (P[:, 1] > y_lo) & (P[:, 1] < y_hi) & (P[:, 2] > top_z + 0.03) & (P[:, 2] < top_z + 0.20)]
                hint = None
                if Cz.shape[0] >= 6:
                    gi_ = ((Cz[:, 0] - Cz[:, 0].min()) / 0.04).astype(int)
                    gj_ = ((Cz[:, 1] - Cz[:, 1].min()) / 0.04).astype(int)
                    k_ = gi_ * 100000 + gj_
                    u_, inv_, c_ = np.unique(k_, return_inverse=True, return_counts=True)
                    kbest = int(np.argmax(c_))
                    if c_[kbest] >= 6:
                        Cc = Cz[inv_ == kbest]
                        ext_ = [float(Cc[:, 0].max() - Cc[:, 0].min()), float(Cc[:, 1].max() - Cc[:, 1].min())]
                        if max(ext_) <= 0.15:
                            hint = dict(x=round(float(Cc[:, 0].mean()), 3), y=round(float(Cc[:, 1].mean()), 3), n=int(c_[kbest]),
                                        height=round(float(Cc[:, 2].max() - top_z), 3))
                counter["can_hint"] = hint
            out.update(counter)
            log("LIDAR: counter front face %.2f m ahead, yaw %+.1f deg (square up by turning %+.1f), %s wide (y %+.2f..%+.2f), top %s%s%s" % (
                dist, yaw, -yaw, counter["counter_width_txt"], y_lo, y_hi,
                "?" if top_z is None else "%.2f m above the floor" % (top_z - zf),
                "" if top_z is None else (", free space above -> counter-like" if counter.get("counter_like") else ", %d pts above it -> wall?" % counter.get("n_above_top", 0)),
                "" if not counter.get("can_hint") else "; can-like cluster at x=%.2f y=%+.2f (%d pts, h %.0f mm)" % (
                    counter["can_hint"]["x"], counter["can_hint"]["y"], counter["can_hint"]["n"], counter["can_hint"]["height"] * 1000)))
            if counter.get("can_hint") and counter.get("counter_like"):
                out.update(ok=True, can_x=counter["can_hint"]["x"], can_y=counter["can_hint"]["y"], can_z=top_z,
                           table_x_min=round(dist, 3), table_edge_yaw_deg=round(yaw, 1), n_cans=1, source="counter-front")
if counter is None:
    log("LIDAR: no vertical counter/table front within +-50 deg, 0.9-3.5 m")

# table planes: z histogram in the table band, but each candidate must be a COMPACT horizontal patch (connected
# 5 cm cells), not the ring of wall points at the sensor's own height that a plain histogram picks up
z_lo, z_hi = [float(v) for v in args.table_z.split(",")]
band = P[(P[:, 2] > z_lo) & (P[:, 2] < z_hi)]
tables = []


def largest_patch(T, cell=0.05, min_cells=6):
    """largest connected component of T's xy occupancy; returns (mask over T, n_cells)."""
    if T.shape[0] == 0:
        return None, 0
    ox_, oy_ = float(T[:, 0].min()), float(T[:, 1].min())
    gi_ = ((T[:, 0] - ox_) / cell).astype(int)
    gj_ = ((T[:, 1] - oy_) / cell).astype(int)
    key_ = gi_ * 100000 + gj_
    uniq_ = set(int(k) for k in np.unique(key_))
    seen = {}
    best = None
    for k in uniq_:
        if k in seen:
            continue
        stack = [k]
        seen[k] = True
        comp_ = [k]
        while stack:
            kk = stack.pop()
            i0, j0 = divmod(kk, 100000)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (i0 + di) * 100000 + (j0 + dj)
                    if nb in uniq_ and nb not in seen:
                        seen[nb] = True
                        stack.append(nb)
                        comp_.append(nb)
        if best is None or len(comp_) > len(best):
            best = comp_
    if best is None or len(best) < min_cells:
        return None, 0
    bs = set(best)
    mask = np.array([int(k) in bs for k in key_])
    return mask, len(best)


if band.shape[0] >= args.min_table_pts:
    h, e = np.histogram(band[:, 2], bins=np.arange(z_lo, z_hi + 0.01, 0.01))
    order = np.argsort(h)[::-1]
    used = np.zeros(len(h), dtype=bool)
    for k in order:
        if h[k] < args.min_table_pts // 3 or used[k]:
            continue
        zt = float(e[k] + 0.005)
        sel = np.abs(band[:, 2] - zt) < 0.02
        zt = float(np.median(band[sel, 2]))
        sel = np.abs(band[:, 2] - zt) < 0.02
        used[max(0, k - 3):k + 4] = True
        T_all = band[sel]
        mask, n_cells = largest_patch(T_all)
        if mask is None:
            continue
        T = T_all[mask]
        if T.shape[0] < args.min_table_pts // 3:
            continue
        # a table top owns its footprint: few points 4-30 cm above/below it over the same xy cells. A wall, shelf or
        # person sliced at this height has as many points in the neighbouring bands (three such "patches" at
        # different heights over one footprint showed up to the robot's left-rear tonight).
        ox_, oy_ = float(T[:, 0].min()) - 0.05, float(T[:, 1].min()) - 0.05
        gi_ = ((T[:, 0] - ox_) / 0.05).astype(int)
        gj_ = ((T[:, 1] - oy_) / 0.05).astype(int)
        occ_ = np.zeros((gi_.max() + 2, gj_.max() + 2), dtype=bool)
        occ_[gi_, gj_] = True
        dz = P[:, 2] - zt
        nb = P[(np.abs(dz) > 0.04) & (np.abs(dz) < 0.30)]
        ni = ((nb[:, 0] - ox_) / 0.05).astype(int)
        nj = ((nb[:, 1] - oy_) / 0.05).astype(int)
        okn = (ni >= 0) & (nj >= 0) & (ni < occ_.shape[0]) & (nj < occ_.shape[1])
        okn[okn] = occ_[ni[okn], nj[okn]]
        n_other = int(okn.sum())
        ratio = T.shape[0] / max(1.0, float(n_other))
        if args.verbose:
            log("  patch z %+.3f: %d pts in %d cells (of %d in the band); %d pts 4-30 cm above/below over the footprint -> ratio %.2f%s" % (
                zt, T.shape[0], n_cells, T_all.shape[0], n_other, ratio, "" if ratio >= 0.8 else "  (rejected: not a table top)"))
        if ratio < 0.8:
            continue
        # footprint and near edge (in xy)
        r = np.hypot(T[:, 0], T[:, 1])
        bearing = np.degrees(np.arctan2(T[:, 1], T[:, 0]))
        i_near = int(np.argmin(r))
        # front edge samples: nearest point per 3-degree bearing bin, then a line fit x = a + b*y in the body frame
        ys_e, xs_e = [], []
        for b0 in np.arange(-90, 90, 3.0):
            m = (bearing >= b0) & (bearing < b0 + 3.0)
            if int(m.sum()) >= 8:
                j = np.argmin(r[m])
                xs_e.append(float(T[m][j, 0]))
                ys_e.append(float(T[m][j, 1]))
        edge_yaw = None
        edge_dist = None
        if len(xs_e) >= 4:
            A = np.stack([np.ones(len(ys_e)), np.array(ys_e)], axis=1)
            xs_a = np.array(xs_e)
            coef = np.linalg.lstsq(A, xs_a, rcond=None)[0]
            keep = np.abs(xs_a - A @ coef) < 0.04
            if int(keep.sum()) >= 4:
                coef = np.linalg.lstsq(A[keep], xs_a[keep], rcond=None)[0]
            edge_yaw = math.degrees(math.atan(float(coef[1])))
            edge_dist = float(coef[0]) / math.sqrt(1.0 + float(coef[1]) ** 2)     # perpendicular distance to the edge line
        tables.append(dict(z_level=round(zt, 3), n=int(T.shape[0]), cells=n_cells, x_min=round(float(T[:, 0].min()), 3), x_max=round(float(T[:, 0].max()), 3),
                           y_min=round(float(T[:, 1].min()), 3), y_max=round(float(T[:, 1].max()), 3),
                           centroid=[round(float(T[:, 0].mean()), 3), round(float(T[:, 1].mean()), 3)],
                           nearest=[round(float(T[i_near, 0]), 3), round(float(T[i_near, 1]), 3)], nearest_range=round(float(r[i_near]), 3),
                           edge_yaw_deg=None if edge_yaw is None else round(edge_yaw, 1),
                           edge_dist=None if edge_dist is None else round(edge_dist, 3), edge_bins=len(xs_e), _pts=T))
        if len(tables) >= 3:
            break
tables.sort(key=lambda t: -t["cells"])
for t in tables:
    log("LIDAR: patch z_level %+.3f (%.2f m above the floor): %d pts / %d cells, x %.2f..%.2f, y %+.2f..%+.2f, nearest %s at %.2f m, edge yaw %s, edge %s m" % (
        t["z_level"], t["z_level"] - out.get("floor_z_level", -0.79), t["n"], t["cells"], t["x_min"], t["x_max"], t["y_min"], t["y_max"], t["nearest"], t["nearest_range"],
        "n/a" if t["edge_yaw_deg"] is None else "%+.1f deg" % t["edge_yaw_deg"], "n/a" if t["edge_dist"] is None else "%.2f" % t["edge_dist"]))
if not tables:
    log("LIDAR: no horizontal plane in z_level %.2f..%.2f (%d pts in band)" % (z_lo, z_hi, band.shape[0]))
    out["why"] = "no table"
    json.dump(out, open(args.out, "w"), indent=1)
    sys.exit(3)
table = tables[0]
T = table.pop("_pts")
for t in tables[1:]:
    t.pop("_pts", None)
out["table"] = table
out["other_planes"] = tables[1:]

# objects on the table: above the plane, over the footprint (5 cm occupancy of table points, 1-cell dilation)
zt = table["z_level"]
cell = 0.05
ox, oy = float(T[:, 0].min()) - cell, float(T[:, 1].min()) - cell
gi = ((T[:, 0] - ox) / cell).astype(int)
gj = ((T[:, 1] - oy) / cell).astype(int)
occ = np.zeros((gi.max() + 3, gj.max() + 3), dtype=bool)
occ[gi, gj] = True
occ_d = occ.copy()
occ_d[1:, :] |= occ[:-1, :]
occ_d[:-1, :] |= occ[1:, :]
occ_d[:, 1:] |= occ[:, :-1]
occ_d[:, :-1] |= occ[:, 1:]
above = P[(P[:, 2] > zt + 0.025) & (P[:, 2] < zt + 0.30)]
ai = ((above[:, 0] - ox) / cell).astype(int)
aj = ((above[:, 1] - oy) / cell).astype(int)
inside = (ai >= 0) & (aj >= 0) & (ai < occ_d.shape[0]) & (aj < occ_d.shape[1])
inside[inside] = occ_d[ai[inside], aj[inside]]
O = above[inside]
log("LIDAR: %d points above the table top (over its footprint)" % O.shape[0])
clusters = []
if O.shape[0] >= 4:
    c2 = 0.03
    ci = ((O[:, 0] - ox) / c2).astype(int)
    cj = ((O[:, 1] - oy) / c2).astype(int)
    key = ci * 100000 + cj
    uniq, inv = np.unique(key, return_inverse=True)
    cells = {int(k): [] for k in uniq}
    for idx, k in enumerate(inv):
        cells[int(uniq[k])].append(idx)
    # connected components over occupied 3 cm cells (8-neighbourhood)
    label = {}
    comp = []
    for k in cells:
        if k in label:
            continue
        stack = [k]
        label[k] = len(comp)
        members = []
        while stack:
            kk = stack.pop()
            members.extend(cells[kk])
            i0, j0 = divmod(kk, 100000)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (i0 + di) * 100000 + (j0 + dj)
                    if nb in cells and nb not in label:
                        label[nb] = label[k]
                        stack.append(nb)
        comp.append(members)
    for members in comp:
        C = O[members]
        if C.shape[0] < 4:
            continue
        ext = [float(C[:, 0].max() - C[:, 0].min()), float(C[:, 1].max() - C[:, 1].min())]
        height = float(C[:, 2].max() - zt)
        cx, cy = float(C[:, 0].mean()), float(C[:, 1].mean())
        can_like = 0.07 <= height <= 0.17 and max(ext) <= 0.13 and min(ext) <= 0.10
        clusters.append(dict(x=round(cx, 3), y=round(cy, 3), n=int(C.shape[0]), height=round(height, 3), extent=[round(v, 3) for v in ext],
                             range=round(math.hypot(cx, cy), 3), bearing_deg=round(math.degrees(math.atan2(cy, cx)), 1), can_like=bool(can_like)))
clusters.sort(key=lambda c: (not c["can_like"], c["range"]))
out["clusters"] = clusters
cans = [c for c in clusters if c["can_like"]]
for c in clusters[:8]:
    log("  %s x=%.3f y=%+.3f  range %.2f m bearing %+.0f deg  h %.0f mm  footprint %.0fx%.0f mm  %d pts" % (
        "CAN?" if c["can_like"] else "obj ", c["x"], c["y"], c["range"], c["bearing_deg"], c["height"] * 1000, c["extent"][0] * 1000, c["extent"][1] * 1000, c["n"]))
if cans:
    out.update(ok=True, can_x=cans[0]["x"], can_y=cans[0]["y"], can_z=zt, table_x_min=table["edge_dist"] if table["edge_dist"] is not None else table["nearest_range"],
               table_edge_yaw_deg=table["edge_yaw_deg"], n_cans=len(cans))
    log("LIDAR: %d can-like object(s); nearest at x=%.3f y=%+.3f (%.2f m, bearing %+.0f deg), table edge %.2f m ahead, edge yaw %s" % (
        len(cans), cans[0]["x"], cans[0]["y"], cans[0]["range"], cans[0]["bearing_deg"], out["table_x_min"],
        "n/a" if table["edge_yaw_deg"] is None else "%+.1f deg" % table["edge_yaw_deg"]))
else:
    out["why"] = "table but no can-like object"
    log("LIDAR: table found, no can-like object on it")
json.dump(out, open(args.out, "w"), indent=1)
print("G1_LIDAR_LOOK_OUT %s" % args.out, flush=True)
