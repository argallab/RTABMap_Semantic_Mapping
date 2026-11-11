"""Build a map from ground-truth (ceiling camera) poses and compare it with a
pipeline/live map of the same recording.

Each node's cleaned scan is placed with the GT pose interpolated at the node
stamp (+ --delta), GT heading corrected by --heading-offset-deg.  Nodes without
GT (gaps > 0.1 s) are skipped.  The map is a hit-count projection of the wall
band (z 0.10-1.00 m) at 2.5 cm: occupied where >= --min-hits points land.
This is NOT the exporter's ray-traced grid (no free space); it shows where
walls and objects really are.

Outputs in output/gt_map_<db>/:
  gt_map.pgm/.yaml      GT-pose map (black = occupied, white = observed nothing / unknown)
  compare.png           overlay with the comparison poses (live graph by default),
                        SE(2)-aligned to GT: black = both, blue = GT only, red = other only

Usage: python3 gt_map.py 10-08-01.db tf_10-08-01 [--heading-offset-deg -4.12] [--compare poses.csv]
"""
import argparse
import os

import cv2
import numpy as np

import common as C
import gt as G

RES = 0.025


def project(scans, poses, R=None, t=None):
    pts = []
    for i in sorted(scans):
        T = poses.get(i)
        if T is None:
            continue
        p = scans[i]
        p = p[(p[:, 2] > 0.10) & (p[:, 2] < 1.00)]
        w = p.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        xy = w[:, :2]
        if R is not None:
            xy = xy @ R.T + t
        pts.append(xy)
    return np.concatenate(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db"); ap.add_argument("bag")
    ap.add_argument("--heading-offset-deg", type=float, default=-4.1)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--min-hits", type=int, default=5)
    ap.add_argument("--compare", help="pose CSV to compare (default: live graph in Admin)")
    ap.add_argument("--xscale", type=float, default=1.0)
    ap.add_argument("--yscale", type=float, default=1.0)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    con = C.connect(a.db)
    nodes = C.load_nodes(con)
    stamps = {i: v[0] for i, v in nodes.items()}
    live = C.load_rtabmap_opt_poses(con)
    other = C.load_pose_csv(a.compare) if a.compare else live
    g = G.load_bag(os.path.join(C.DB_DIR, a.bag))["gt"]
    hoff = np.radians(a.heading_offset_deg)
    scans = C.clean_scans(a.db, live.keys())
    if a.xscale != 1.0 or a.yscale != 1.0:
        scans = {i: (p * np.array([a.xscale, a.yscale, 1.0], np.float32)) for i, p in scans.items()}

    # GT poses per node
    ids = sorted(scans)
    q = G.interp_pose(g, np.array([stamps[i] for i in ids]) + a.delta)
    gtp = {}
    for i, r in zip(ids, q):
        if np.isfinite(r).all():
            T = np.eye(4)
            c, s = np.cos(r[2] + hoff), np.sin(r[2] + hoff)
            T[:2, :2] = [[c, -s], [s, c]]; T[0, 3], T[1, 3] = r[0], r[1]
            gtp[i] = T

    # align comparison trajectory to GT (SE2 fit on node positions)
    common_ids = [i for i in gtp if i in other]
    src = np.array([[other[i][0, 3], other[i][1, 3]] for i in common_ids])
    dst = np.array([[gtp[i][0, 3], gtp[i][1, 3]] for i in common_ids])
    R, t = G.fit_se2(src, dst)

    P_gt = project({i: scans[i] for i in gtp}, gtp)
    P_ot = project({i: scans[i] for i in common_ids}, other, R, t)
    lo = np.minimum(P_gt.min(0), P_ot.min(0)) - 0.5
    hi = np.maximum(P_gt.max(0), P_ot.max(0)) + 0.5
    W, H = (np.ceil((hi - lo) / RES).astype(int) + 1)

    def counts(P):
        c = np.zeros((H, W), np.int32)
        ix = ((P[:, 0] - lo[0]) / RES).astype(int); iy = ((P[:, 1] - lo[1]) / RES).astype(int)
        np.add.at(c, (iy, ix), 1)
        return np.flipud(c)  # image row 0 = max y (map convention)

    cg, co = counts(P_gt), counts(P_ot)
    og, oo = cg >= a.min_hits, co >= a.min_hits

    out = a.outdir or f"{C.ROOT}/output/gt_map_{a.db.replace('.db', '')}"
    os.makedirs(out, exist_ok=True)
    pgm = np.full((H, W), 205, np.uint8)
    pgm[cg > 0] = 254
    pgm[og] = 0
    cv2.imwrite(f"{out}/gt_map.pgm", pgm)
    with open(f"{out}/gt_map.yaml", "w") as f:
        f.write(f"image: gt_map.pgm\nmode: trinary\nresolution: {RES}\norigin: [{lo[0]:.3f}, {lo[1]:.3f}, 0]\n"
                f"negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    vis = np.full((H, W, 3), 255, np.uint8)
    vis[og & ~oo] = (230, 120, 0)     # blue: GT only
    vis[oo & ~og] = (0, 0, 220)       # red: comparison only
    vis[og & oo] = (0, 0, 0)          # black: both
    cv2.imwrite(f"{out}/compare.png", cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(f"{out}/gt_map.png", pgm)
    print(f"GT nodes {len(gtp)}/{len(ids)}  compare nodes {len(common_ids)}  grid {W}x{H} @ {RES} m")
    print(f"occupied cells: GT {og.sum()}  compare {oo.sum()}  both {(og & oo).sum()}  "
          f"GT-only {(og & ~oo).sum()}  compare-only {(oo & ~og).sum()}")
    print(f"wrote {out}/gt_map.pgm/.yaml/.png and compare.png")


if __name__ == "__main__":
    main()
