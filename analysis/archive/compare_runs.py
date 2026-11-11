"""Compare two exporter runs of the same recording against ground truth.

Outputs (printed + analysis/results/compare_<db>_<runA>_vs_<runB>/):
  * ATE / RPE / heading error per pipeline stage for both runs (GT heading-corrected)
  * wall offsets vs GT: each run's final-pose scans vs GT-pose scans, per 20 cm
    straight wall piece (median / p90 / max mm) + arrow image
  * grids.png: the two exporter grids side by side (as saved, no re-alignment)

Usage: python3 compare_runs.py 10-07-03.db tf_10-07-03 <outA> <outB> [--heading-offset-deg -3.97] [--labels base yawfix]
"""
import argparse
import os
from collections import defaultdict

import cv2
import numpy as np

import common as C
import gt as G

STAGES = ["rtabmap_graph", "tag_GTSAM", "icp_round_1", "icp_round_2", "manhattan", "final"]


def wall_offsets(scans, ids, P, gtp, R, t):
    def pts(PP, align):
        out = []
        for i in ids:
            p = scans[i][::3]
            p = p[(p[:, 2] > 0.1) & (p[:, 2] < 1.0)]
            w = (p @ PP[i][:3, :3].T + PP[i][:3, 3])[:, :2]
            out.append(w @ R.T + t if align else w)
        return np.concatenate(out)
    Pg, Pl = pts(gtp, False), pts(P, True)
    dg, dl = defaultdict(list), defaultdict(list)
    for k, p in zip(map(tuple, np.floor(Pg / 0.2).astype(int)), Pg): dg[k].append(p)
    for k, p in zip(map(tuple, np.floor(Pl / 0.2).astype(int)), Pl): dl[k].append(p)
    offs = []
    for k in set(dg) & set(dl):
        A, B = np.array(dg[k]), np.array(dl[k])
        if len(A) < 40 or len(B) < 40: continue
        Q = np.vstack([A, B]); m = Q.mean(0); ev, evec = np.linalg.eigh(np.cov((Q - m).T))
        if ev[0] > 0.15 * ev[1]: continue
        n = evec[:, 0]
        offs.append(abs(np.median((B - m) @ n) - np.median((A - m) @ n)) * 1000)
    return np.array(offs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db"); ap.add_argument("bag"); ap.add_argument("outA"); ap.add_argument("outB")
    ap.add_argument("--heading-offset-deg", type=float, default=-4.0)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    a = ap.parse_args()
    con = C.connect(a.db)
    nodes = C.load_nodes(con); stamps = {i: v[0] for i, v in nodes.items()}
    live = C.load_rtabmap_opt_poses(con)
    g = G.load_bag(os.path.join(C.DB_DIR, a.bag))["gt"].copy(); g[:, 3] += np.radians(a.heading_offset_deg)
    dst = os.path.join(C.ROOT, "analysis", "results",
                       f"compare_{a.db.replace('.db','')}_{os.path.basename(a.outA)}_vs_{os.path.basename(a.outB)}")
    os.makedirs(dst, exist_ok=True)

    print(f"\n=== {a.db}: {a.labels[0]} = {os.path.basename(a.outA)}   {a.labels[1]} = {os.path.basename(a.outB)} ===")
    print(f"  {'stage':14s} {'run':8s} | ATE rmse  med   p90 (mm) | RPE1m RPE5m (mm) | heading err med/p90 (deg)")
    final = {}
    for st in STAGES:
        for lab, out in zip(a.labels, (a.outA, a.outB)):
            f = os.path.join(out, f"poses_{st}.csv")
            if not os.path.exists(f): continue
            P = C.load_pose_csv(f)
            e = G.evaluate(P, stamps, g)["fit"]
            print(f"  {st:14s} {lab:8s} | {e['ate_rmse_mm']:7.1f} {e['ate_median_mm']:5.1f} {e['ate_p90_mm']:6.1f} |"
                  f" {e['rpe_1m_trans_median_mm']:5.1f} {e['rpe_5m_trans_median_mm']:5.1f} | {e['yaw_err_median_deg']:.2f} / {e['yaw_err_p90_deg']:.2f}")
            final[(lab, st)] = P

    # wall offsets vs GT at the graph stage and the final stage
    scans = C.clean_scans(a.db, live.keys())
    ids0 = sorted(scans)
    q = G.interp_pose(g, np.array([stamps[i] for i in ids0]))
    gtp = {}
    for i, r in zip(ids0, q):
        if np.isfinite(r).all():
            T = np.eye(4); c, s = np.cos(r[2]), np.sin(r[2]); T[:2, :2] = [[c, -s], [s, c]]; T[0, 3], T[1, 3] = r[0], r[1]
            gtp[i] = T
    print("\n  wall position vs GT walls (|offset| per 20 cm straight wall piece, mm):")
    for st in ("rtabmap_graph", "icp_round_2", "manhattan"):
        for lab in a.labels:
            P = final.get((lab, st))
            if P is None: continue
            ids = [i for i in gtp if i in P]
            R, t = G.fit_se2(np.array([[P[i][0, 3], P[i][1, 3]] for i in ids]), np.array([[gtp[i][0, 3], gtp[i][1, 3]] for i in ids]))
            o = wall_offsets(scans, ids, P, gtp, R, t)
            print(f"    {st:14s} {lab:8s}: median {np.median(o):5.1f}  p90 {np.percentile(o, 90):5.1f}  max {o.max():5.1f}  (n={len(o)})")

    # grids side by side
    ims = []
    for lab, out in zip(a.labels, (a.outA, a.outB)):
        ts = os.path.basename(os.path.normpath(out))
        im = cv2.imread(C.grid_pgm(out), cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.putText(im, lab, (10, 25), 0, 0.8, (0, 0, 200), 2)
        ims.append(im)
    if len(ims) == 2:
        h = max(i.shape[0] for i in ims)
        ims = [cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 10, cv2.BORDER_CONSTANT, value=(128, 128, 128)) for i in ims]
        cv2.imwrite(os.path.join(dst, "grids.png"), np.hstack(ims))
        print(f"\n  grids side by side -> {os.path.join(dst, 'grids.png')}")


if __name__ == "__main__":
    main()
