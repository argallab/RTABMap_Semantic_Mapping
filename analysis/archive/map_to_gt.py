"""Transform from an exporter map to the ground-truth frame.

initial_pose (/ground_truth/initial_pose, last message before the recording
starts) is the ground-truth pose of the chair at the FIRST graph node, so

    T_gt<-map = initial_pose (heading-corrected) * inverse(first node pose in map)

Exporter runs with poses_final.csv (WC_MAP_ORIGIN=first_node, the default) put
the first node at (0, 0, 0°), so T_gt<-map is initial_pose itself.  Older runs
and WC_MAP_ORIGIN=odom keep RTAB-Map's frame (origin = where odometry started)
and need the inverse(first node) term, which this script applies either way.

The ceiling camera's heading reads ~4 deg off (fit per recording: -3.97 to
-4.70 deg); pass --heading-offset-deg.  For reference the script also prints
the best SE(2) fit of the whole trajectory to ground truth and the median
position error of both.  Anchoring on one pose leaves ~11-17 cm median error
(a 1-2 deg heading error at the start swings the far end of the map); the
whole-trajectory fit gives 3.5-5 cm.

Writes analysis/results/map_to_gt_<run>.txt (run folders are root-owned) with both transforms and a static_transform_publisher
line (parent = ground-truth frame, child = map).

Usage: python3 map_to_gt.py 10-12-01.db tf_10-12-01 ../output/<run> [--heading-offset-deg -4.7] [--gt-frame ground_truth]
"""
import argparse
import os

import numpy as np

import common as C
import gt as G


def se2(x, y, th):
    T = np.eye(3)
    T[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    T[:2, 2] = [x, y]
    return T


def xyth(T):
    return T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db"); ap.add_argument("bag"); ap.add_argument("run")
    ap.add_argument("--heading-offset-deg", type=float, default=-4.1)
    ap.add_argument("--gt-frame", default="ground_truth")
    a = ap.parse_args()

    con = C.connect(a.db)
    nodes = C.load_nodes(con)
    P = None
    for st in ("final", "manhattan", "icp_round_2", "icp_round_1", "tag_GTSAM", "rtabmap_graph"):
        f = os.path.join(a.run, f"poses_{st}.csv")
        if os.path.exists(f):
            P, stage = C.load_pose_csv(f), st
            break
    if P is None:
        raise SystemExit("no poses_*.csv in the run folder")
    ids = sorted(i for i in P if i >= 1)
    t = np.array([nodes[i][0] for i in ids])
    b = G.load_bag(os.path.join(C.DB_DIR, a.bag))
    ip = b["initial_pose"]
    before = [k for k in range(len(ip)) if ip[k, 0] <= t[0] + 1.0]
    if not before:
        raise SystemExit("no initial_pose message before the recording start")
    r = ip[before[-1]]
    i0 = ids[0]
    N0 = se2(P[i0][0, 3], P[i0][1, 3], np.arctan2(P[i0][1, 0], P[i0][0, 0]))
    T = se2(r[1], r[2], r[3] + np.radians(a.heading_offset_deg)) @ np.linalg.inv(N0)

    # check against ground truth along the whole drive
    q = G.interp_pose(b["gt"], t)
    ok = np.isfinite(q).all(1)
    src = np.array([P[i][:2, 3] for i in ids])[ok]
    dst = q[ok, :2]
    R, tt = G.fit_se2(src, dst)
    Tf = np.eye(3); Tf[:2, :2] = R; Tf[:2, 2] = tt
    err = lambda M: np.hypot(*((M[:2, :2] @ src.T).T + M[:2, 2] - dst).T) * 1000

    x, y, th = xyth(T)
    xf, yf, thf = xyth(Tf)
    e_ip, e_fit = err(T), err(Tf)
    lines = [
        f"run {os.path.basename(os.path.normpath(a.run))}  poses {stage}  first node {i0}",
        f"initial_pose msg {before[-1]} at {r[0] - t[0]:+.1f} s: x {r[1]:.4f} y {r[2]:.4f} yaw {np.degrees(r[3]):.3f} deg"
        f"  (heading offset {a.heading_offset_deg:+.2f} deg applied)",
        f"T_gt<-map from initial_pose : x {x:.4f}  y {y:.4f}  yaw {np.degrees(th):.3f} deg"
        f"   position error vs GT median {np.median(e_ip):.0f} mm, p90 {np.percentile(e_ip, 90):.0f} mm",
        f"T_gt<-map best fit (all nodes): x {xf:.4f}  y {yf:.4f}  yaw {np.degrees(thf):.3f} deg"
        f"   position error vs GT median {np.median(e_fit):.0f} mm, p90 {np.percentile(e_fit, 90):.0f} mm",
        f"static TF (initial_pose): ros2 run tf2_ros static_transform_publisher --x {x:.4f} --y {y:.4f} --yaw {th:.5f} "
        f"--frame-id {a.gt_frame} --child-frame-id map",
    ]
    print("\n".join(lines))
    out = os.path.join(C.ROOT, "analysis", "results", f"map_to_gt_{os.path.basename(os.path.normpath(a.run))}.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
