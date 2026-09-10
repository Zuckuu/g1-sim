"""Merge Unitree's G1 29-DoF URDF with BrainCo's Revo 2 hand URDFs into one robot description.

  work/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf   (body, arms, rubber-hand stubs)
+ work/brainco-description/revo2_system/urdf/revo2_{left,right}.urdf
= work/g1-runtime/urdf/g1_29dof_revo2.urdf   (29 body joints + 2 x 11 finger joints, absolute mesh paths)

Mounting convention (until the physical BrainCo wrist adapter is measured): the hand's flange face sits
`--adapter-offset` metres beyond Unitree's rubber-hand mount point on the wrist yaw link, fingers along the
forearm (+x of the wrist link), palm facing the body midline, thumb up. Both hands' frames are derived from
the hand URDFs themselves (finger direction, curl direction, index/pinky positions), not hard-coded.

Run with the project's Isaac venv (numpy only):
  work/g1-runtime/isaac50/bin/python sim/build_g1_revo2_urdf.py [--adapter-offset 0.012]
"""

import argparse
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
G1_URDF = PROJECT / "work/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
REVO_DIR = PROJECT / "work/brainco-description/revo2_system"
OUT_DIR = PROJECT / "work/g1-runtime/urdf"


# ----------------------------------------------------------------------------------------------------------------------
# small kinematics helpers (same math as revo2_hand_grasp.py)
# ----------------------------------------------------------------------------------------------------------------------
def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def mat_to_rpy(R):
    """Inverse of rpy_to_mat (URDF fixed-axis roll-pitch-yaw)."""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    p = math.asin(sy)
    if abs(math.cos(p)) > 1e-8:
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        r = 0.0
        y = math.atan2(-R[0, 1], R[1, 1])
    return r, p, y


def parse_joints(root):
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        ax = j.find("axis")
        axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "0 0 1").split()])
        joints[j.get("name")] = dict(parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                                     xyz=xyz, R=rpy_to_mat(*rpy), axis=axis, type=j.get("type"))
    return joints


def link_pose(link, joints, child_to_joint, stop_link):
    R = np.eye(3)
    p = np.zeros(3)
    chain = []
    while link in child_to_joint and link != stop_link:
        j = joints[child_to_joint[link]]
        chain.append(j)
        link = j["parent"]
    for j in reversed(chain):
        p = p + R @ j["xyz"]
        R = R @ j["R"]
    return R, p


def hand_frame(side, root):
    """Finger direction f, palm normal n (curl direction), and index/pinky knuckle positions in {side}_hand_base_link."""
    joints = parse_joints(root)
    c2j = {jd["child"]: n for n, jd in joints.items()}
    base = f"{side}_hand_base_link"
    R_idx, p_idx = link_pose(f"{side}_index_proximal_link", joints, c2j, base)
    _, p_tip = link_pose(f"{side}_index_tip_link", joints, c2j, base)
    _, p_pinky = link_pose(f"{side}_pinky_proximal_link", joints, c2j, base)
    f = p_tip - p_idx
    f /= np.linalg.norm(f)
    axis = R_idx @ joints[f"{side}_index_proximal_joint"]["axis"]
    n = np.cross(axis, f)
    n -= np.dot(n, f) * f
    n /= np.linalg.norm(n)
    return f, n, p_idx, p_pinky


def mount_rotation(side, f_h, n_h, p_idx, p_pinky):
    """Rotation R (wrist <- hand) with fingers along +x, palm toward the midline, index above pinky."""
    f_w = np.array([1.0, 0.0, 0.0])
    n_w = np.array([0.0, 1.0, 0.0]) if side == "right" else np.array([0.0, -1.0, 0.0])
    # complete both triads right-handedly and solve R * B_h = B_w
    B_h = np.stack([f_h, n_h, np.cross(f_h, n_h)], axis=1)
    B_w = np.stack([f_w, n_w, np.cross(f_w, n_w)], axis=1)
    R = B_w @ np.linalg.inv(B_h)
    up = (R @ p_idx)[2] - (R @ p_pinky)[2]
    if up < 0:  # thumb would point down: flip the palm normal sign is not allowed, so rotate 180 deg about the fingers
        R = rpy_to_mat(math.pi, 0.0, 0.0) @ R
        up = (R @ p_idx)[2] - (R @ p_pinky)[2]
    assert up > 0, "index knuckle must end up above the pinky knuckle"
    # re-orthonormalise
    u, _, vt = np.linalg.svd(R)
    return u @ vt


def compose(xyz1, rpy1, xyz2, rpy2):
    """T1 * T2 for URDF origins -> (xyz, rpy)."""
    R1, R2 = rpy_to_mat(*rpy1), rpy_to_mat(*rpy2)
    xyz = np.asarray(xyz1) + R1 @ np.asarray(xyz2)
    return xyz, mat_to_rpy(R1 @ R2)


def _origin(el):
    o = el.find("origin")
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return xyz, rpy


def fold_fixed_children(root, side):
    """Merge the hand's fixed-joint children (empty fingertip frames, 2 g touch pads) into their parent links.

    Keeps the collision/visual geometry (re-expressed in the parent frame) and sums the masses. This avoids massless
    bodies in the full-robot articulation and the importer's own fixed-joint merge, which corrupts finger geometry.
    """
    links = {l.get("name"): l for l in root.findall("link")}
    for j in list(root.findall("joint")):
        if j.get("type") != "fixed" or j.get("name") == f"{side}_hand_base_joint":
            continue
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        if child == f"{side}_hand_base_link" or parent == "world":
            continue
        P, C = links[parent], links[child]
        jxyz, jrpy = _origin(j)
        for tag in ("visual", "collision"):
            for g in C.findall(tag):
                gg = copy.deepcopy(g)
                gxyz, grpy = _origin(g)
                nxyz, nrpy = compose(jxyz, jrpy, gxyz, grpy)
                o = gg.find("origin")
                if o is None:
                    o = ET.SubElement(gg, "origin")
                o.set("xyz", " ".join(f"{v:.6f}" for v in nxyz))
                o.set("rpy", " ".join(f"{v:.6f}" for v in nrpy))
                P.append(gg)
        ci = C.find("inertial")
        pi = P.find("inertial")
        if ci is not None and pi is not None:
            m_c = float(ci.find("mass").get("value"))
            pm = pi.find("mass")
            pm.set("value", f"{float(pm.get('value')) + m_c:.6f}")
        root.remove(j)
        root.remove(C)
        del links[child]


def abspath_meshes(root, base_dir):
    for el in root.iter("mesh"):
        fn = el.get("filename", "")
        if fn.startswith("package://"):
            fn = fn.split("package://", 1)[1].split("/", 1)[1]
        p = (base_dir / fn).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        el.set("filename", str(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-offset", type=float, default=0.012, help="m from Unitree's rubber-hand mount point to the Revo 2 flange face")
    ap.add_argument("--out", default=str(OUT_DIR / "g1_29dof_revo2.urdf"))
    ap.add_argument("--hands", choices=["both", "right", "left"], default="both")
    a = ap.parse_args()

    g1 = ET.parse(G1_URDF).getroot()
    g1.set("name", "g1_29dof_revo2")
    abspath_meshes(g1, G1_URDF.parent)
    # remove the rubber hand stubs
    for side in ("left", "right"):
        for j in list(g1.findall("joint")):
            if j.get("name") == f"{side}_hand_palm_joint":
                g1.remove(j)
        for l in list(g1.findall("link")):
            if l.get("name") == f"{side}_rubber_hand":
                g1.remove(l)
    # mount point from the removed stub (kept as a constant here: xyz 0.0415 +/-0.003 0 on the wrist yaw link)
    mount_xyz = {"left": np.array([0.0415, 0.003, 0.0]), "right": np.array([0.0415, -0.003, 0.0])}

    report = {}
    sides = ("left", "right") if a.hands == "both" else (a.hands,)
    for side in sides:
        hand_path = REVO_DIR / "urdf" / f"revo2_{side}.urdf"
        hand = ET.parse(hand_path).getroot()
        abspath_meshes(hand, hand_path.parent)
        f_h, n_h, p_idx, p_pinky = hand_frame(side, hand)  # frames from the original tree (tips are needed here)
        fold_fixed_children(hand, side)
        R = mount_rotation(side, f_h, n_h, p_idx, p_pinky)
        rpy = mat_to_rpy(R)
        xyz = mount_xyz[side] + np.array([a.adapter_offset, 0.0, 0.0])
        # copy links/joints except the world link and the fixed world joint
        for l in hand.findall("link"):
            if l.get("name") == "world":
                continue
            g1.append(copy.deepcopy(l))
        for j in hand.findall("joint"):
            if j.get("name") == f"{side}_hand_base_joint":
                continue
            jj = copy.deepcopy(j)
            g1.append(jj)
        # materials (if any) with unique names
        for m in hand.findall("material"):
            mm = copy.deepcopy(m)
            mm.set("name", f"{side}_revo2_{m.get('name')}")
            g1.append(mm)
        # dont_collapse keeps the hand base as its own body when the importer merges fixed joints (IK target frame)
        mount = ET.SubElement(g1, "joint", name=f"{side}_hand_mount_joint", type="fixed", dont_collapse="true")
        ET.SubElement(mount, "origin", xyz=" ".join(f"{v:.6f}" for v in xyz), rpy=" ".join(f"{v:.6f}" for v in rpy))
        ET.SubElement(mount, "parent", link=f"{side}_wrist_yaw_link")
        ET.SubElement(mount, "child", link=f"{side}_hand_base_link")
        report[side] = dict(xyz=xyz.tolist(), rpy=list(rpy), fingers_dir_hand=f_h.tolist(), palm_normal_hand=n_h.tolist(),
                            fingers_dir_wrist=(R @ f_h).tolist(), palm_normal_wrist=(R @ n_h).tolist(),
                            knuckle_index_wrist=(xyz + R @ p_idx).tolist(), knuckle_pinky_wrist=(xyz + R @ p_pinky).tolist())

    # drop material definitions that clash (BrainCo uses unnamed materials inline, Unitree defines few) - keep as is
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(g1, space="  ")
    ET.ElementTree(g1).write(out, encoding="unicode", xml_declaration=True)
    n_joints = sum(1 for j in g1.findall("joint") if j.get("type") in ("revolute", "continuous", "prismatic"))
    n_links = len(g1.findall("link"))
    print(f"wrote {out}: {n_links} links, {n_joints} movable joints")
    for side, r in report.items():
        print(f"  {side}: mount xyz={np.round(r['xyz'], 4).tolist()} rpy={np.round(r['rpy'], 4).tolist()} | fingers->wrist {np.round(r['fingers_dir_wrist'], 3).tolist()} palm normal->wrist {np.round(r['palm_normal_wrist'], 3).tolist()} | index knuckle z {r['knuckle_index_wrist'][2]:+.4f} pinky z {r['knuckle_pinky_wrist'][2]:+.4f}")
    (out.with_suffix(".json")).write_text(__import__("json").dumps(dict(adapter_offset=a.adapter_offset, mounts=report), indent=2) + "\n")


if __name__ == "__main__":
    main()
