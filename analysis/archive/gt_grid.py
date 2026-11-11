"""Occupancy grid of a recording built from GROUND-TRUTH poses (gaps filled with odometry).

Every node is placed at its /ground_truth/odom pose (heading corrected by
--heading-offset-deg); where ground truth drops out (gap > 0.1 s), the pose is
chained from the nearest node with ground truth using the wheel odometry
between them.  The scans are the exporter's own cleaned points (from a run's
cloud PCD), and the grid is built with the Python replica of the exporter
(grid_sim: percentile height veto, keep rules, CC filter) plus wall thinning
with automatic ledge detection (thin_walls), so the only difference from that
run's map is the poses.  The result is expressed in the run's map frame (SE(2)
fit of GT to the run's poses), so it lines up with the run's grid.

  python3 gt_grid.py R01_02.db ../output/<run> [--heading-offset-deg 3.71] [--scan-rot-deg 0]

Writes analysis/results/gridsim/gtmap_<run>_h<offset>_r<rot>.pgm and a labelled
comparison png (run's map | GT-pose map).
"""
import argparse
import os

import cv2
import numpy as np

import common as C
import gt as G
import grid_sim as GS
import thin_walls as TW


def se2(x, y, th):
    T = np.eye(4); c, s = np.cos(th), np.sin(th)
    T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = [x, y]
    return T


def labelled(img, title):
    v = cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    H, W = v.shape[:2]; M = 30
    o = cv2.copyMakeBorder(v, M, 24, M, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for k, x in enumerate(range(0, W, 100)):
        cv2.line(o, (x + M, M), (x + M, H + M), (200, 160, 60), 1); cv2.putText(o, chr(65 + k), (x + M + 44, 21), 0, 0.6, (200, 80, 0), 2)
    for k, y in enumerate(range(0, H, 100)):
        cv2.line(o, (M, y + M), (W + M, y + M), (200, 160, 60), 1); cv2.putText(o, str(k + 1), (8, y + M + 56), 0, 0.6, (200, 80, 0), 2)
    cv2.putText(o, title, (M, H + M + 18), 0, 0.55, (0, 0, 200), 2)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db"); ap.add_argument("run")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--heading-offset-deg", type=float, default=None, help="GT yaw minus heading (default: gt_heading_offset_deg from the run's map yaml)")
    ap.add_argument("--scan-rot-deg", type=float, default=0.0, help="extra rotation of every scan about base_link (camera yaw test)")
    a = ap.parse_args()
    bag = a.bag or "tf_" + a.db.replace(".db", "")
    run = a.run; ts = os.path.basename(os.path.normpath(run))
    hoff = a.heading_offset_deg
    if hoff is None:
        import re
        m = re.search(r"gt_heading_offset_deg:\s*([-0-9.]+)", open(os.path.splitext(C.grid_pgm(run))[0] + ".yaml").read())
        hoff = float(m.group(1)) if m else 0.0

    scans, origins, stamps, align, _ = GS.load_run(a.db, run)
    ids = sorted(scans)
    Pf = None
    for st in ("final", "manhattan", "tag_GTSAM", "rtabmap_graph"):
        f = os.path.join(run, f"poses_{st}.csv")
        if os.path.exists(f):
            Pf = C.load_pose_csv(f); break
    con = C.connect(a.db); nodes = C.load_nodes(con)
    odo = {i: nodes[i][2] for i in ids}
    q = G.interp_pose(G.load_bag(os.path.join(C.DB_DIR, bag))["gt"], np.array([stamps[i] for i in ids]))
    has = {i: np.isfinite(r).all() for i, r in zip(ids, q)}
    gtT = {i: se2(r[0], r[1], r[2] - np.radians(hoff)) for i, r in zip(ids, q) if has[i]}
    # fill dropouts: chain odometry from the nearest node with GT (previous, else next)
    filled = 0
    gt_ids = [i for i in ids if has[i]]
    for k, i in enumerate(ids):
        if has[i]:
            continue
        prev = [j for j in gt_ids if j < i]
        j = prev[-1] if prev else next(j for j in gt_ids if j > i)
        gtT[i] = gtT[j] @ np.linalg.inv(odo[j]) @ odo[i]
        filled += 1
    # into the run's map frame
    R, t = G.fit_se2(np.array([gtT[i][:2, 3] for i in gt_ids]), np.array([Pf[i][:2, 3] for i in gt_ids]))
    A = np.eye(4); A[:2, :2] = R; A[:2, 3] = t
    c, s = np.cos(np.radians(a.scan_rot_deg)), np.sin(np.radians(a.scan_rot_deg))
    Rz = np.eye(4); Rz[:2, :2] = [[c, -s], [s, c]]
    new_scans, new_origins = {}, {}
    for i in ids:
        Pinv = np.linalg.inv(Pf[i])
        base = scans[i].astype(np.float64) @ Pinv[:3, :3].T + Pinv[:3, 3]
        o_base = Pinv @ np.r_[origins[i], Pf[i][2, 3], 1.0]
        T = A @ gtT[i] @ Rz
        new_scans[i] = (base @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        new_origins[i] = (T @ o_base)[:2]
    print(f"{a.db}: {len(ids)} nodes, ground truth on {len(gt_ids)}, {filled} filled with odometry; heading offset {hoff:.2f} deg, scan rotation {a.scan_rot_deg:+.1f} deg")
    img, feat, aux = GS.build_grid(new_scans, new_origins, stamps, 0.0, "pct", log=lambda *x: None, want_aux=True)
    lm = TW.detect_ledge(img, aux)[0]
    out, _ = TW.thin(img, feat, ledge_mask=lm)
    tag = f"gtmap_{ts}_h{hoff:.2f}_r{a.scan_rot_deg:+.1f}"
    d = os.path.join(C.ROOT, "analysis", "results", "gridsim")
    cv2.imwrite(os.path.join(d, tag + ".pgm"), out)
    ref = cv2.imread(C.grid_pgm(run), cv2.IMREAD_GRAYSCALE)
    l1, l2 = labelled(ref, f"run {ts} (its own poses)"), labelled(out, f"ground-truth poses (offset {hoff:.2f}, scan rot {a.scan_rot_deg:+.1f})")
    w = max(l1.shape[1], l2.shape[1])
    pad = lambda x: cv2.copyMakeBorder(x, 0, 0, 0, w - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.imwrite(os.path.join(d, tag + "_compare.png"), np.vstack([pad(l1), np.full((8, w, 3), 128, np.uint8), pad(l2)]))
    print(f"  occupied {(out == 0).sum() * GS.RES**2:.2f} m2 (run {(ref == 0).sum() * GS.RES**2:.2f})   wrote analysis/results/gridsim/{tag}.pgm and _compare.png")


if __name__ == "__main__":
    main()
