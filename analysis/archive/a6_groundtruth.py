"""Analysis 6 (ground truth available): separate pose error from sensor/timing error.

  1. Map from REFERENCE poses: each node's cleaned scan placed with the ceiling-
     camera pose interpolated at node_stamp + delta (no registration, no
     odometry).  Sweep delta: the minimum of the wall spread is the scan
     latency measured against an independent clock; the spread at the minimum
     is the sensor/calibration floor.
  2. Same wall metric for odometry, the live graph and any exporter pose dumps,
     next to their ATE/RPE, so the wall metric can be checked against true
     pose error.

GT heading is corrected by the measured gt_base_link heading offset (straight-
driving crab angle, see results/a6_heading_offset_*.log); lever arm ignored
(fitted ~(23, 13) mm).

Usage: python3 a6_groundtruth.py 10-07-01.db tf_10-07-01 [--poses label=csv ...] [--heading-offset-deg -4.08]
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import common as C
import gt as G

CHEAP = C.WallCfg(target_pts=750_000, queries=5_000)
W = {}


def gt_poses(ids, stamps, g, delta, hoff):
    ts = np.array([stamps[i] for i in ids])
    p = G.interp_pose(g, ts + delta)
    out = {}
    for i, q in zip(ids, p):
        if not np.isfinite(q).all():
            continue
        T = np.eye(4)
        c, s = np.cos(q[2] + hoff), np.sin(q[2] + hoff)
        T[:2, :2] = [[c, -s], [s, c]]
        T[0, 3], T[1, 3] = q[0], q[1]
        out[i] = T
    return out


def near(nid, p):
    r = np.hypot(p[:, 0], p[:, 1])
    return np.where(r < 1.3, 0.5, np.nan)


def wall(poses, full):
    res = C.wall_metric(W["scans"], poses, C.WallCfg() if full else CHEAP,
                        bins={"near": (near, np.array([0.0, 1.0]))})
    nb = res["near"][0]
    return dict(n=res["n"], ref=res["ref_median_mm"], local=res["local_median_mm"], glob=res["global_median_mm"],
                glob_p90=res["global_p90_mm"], near_scan=nb["scan_median_mm"], near_ref=nb["ref_median_mm"],
                near_glob=nb["global_median_mm"])


def eval_delta(args):
    delta, full = args
    P = gt_poses(W["ids"], W["stamps"], W["g"], delta, W["hoff"])
    return dict(delta=delta, full=full, nodes=len(P), **wall(P, full))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("bag")
    ap.add_argument("--poses", nargs="*", default=[])
    ap.add_argument("--heading-offset-deg", type=float, default=-4.08)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    con = C.connect(a.db)
    nodes = C.load_nodes(con)
    stamps = {i: v[0] for i, v in nodes.items()}
    live = C.load_rtabmap_opt_poses(con)
    bag = os.path.join(C.DB_DIR, a.bag)
    g = G.load_bag(bag)["gt"]
    hoff = np.radians(a.heading_offset_deg)
    scans = C.clean_scans(a.db, live.keys())
    ids = sorted(i for i in scans)
    W.update(scans=scans, ids=ids, stamps=stamps, g=g, hoff=hoff)
    tag = a.db.replace(".db", "")
    out = {}

    # 1. delta sweep on the GT-built map
    deltas = [round(x, 2) for x in np.arange(-0.30, 0.1001, 0.05)]
    with Pool(a.procs) as pool:
        rows = pool.map(eval_delta, [(d, False) for d in deltas], chunksize=1)
    best = min(rows, key=lambda r: r["glob"])
    with Pool(2) as pool:
        full = pool.map(eval_delta, [(0.0, True), (best["delta"], True)])
    print(f"\n=== map from GROUND-TRUTH poses, {a.db}  (GT heading offset {a.heading_offset_deg:+.2f} deg) ===")
    print("  delta(s)  global  near_global   ref  near_ref  near_scan   n   (cheap cfg)")
    for r in rows:
        print("  {delta:+5.2f}   {glob:6.1f}  {near_glob:8.1f}  {ref:6.1f}  {near_ref:6.1f}  {near_scan:6.1f}  {n}".format(**r))
    print("  full cfg:")
    for r in full:
        print("  {delta:+5.2f}   {glob:6.1f}  {near_glob:8.1f}  {ref:6.1f}  {near_ref:6.1f}  {near_scan:6.1f}  {n}   p90 {glob_p90:.1f}".format(**r))
    out["gt_delta_sweep"] = {"rows": rows, "full": full}

    # 2. wall metric + ATE/RPE per pose source
    gcorr = g.copy()
    gcorr[:, 3] += hoff
    srcs = {"odom": {i: nodes[i][2] for i in live}, "live_graph": live}
    for spec in a.poses:
        k, p = spec.split("=", 1)
        srcs[k] = C.load_pose_csv(p)
    srcs[f"GT@{best['delta']:+.2f}s"] = gt_poses(ids, stamps, g, best["delta"], hoff)
    print(f"\n=== pose sources: wall metric (full cfg) vs ATE/RPE (GT heading-corrected, SE2 fit) ===")
    print("  source              global  near_glob   ref   |  ATE rmse  median   p90  | RPE1m  RPE5m  yaw-err")
    out["sources"] = {}
    for k, P in srcs.items():
        w = wall(P, True)
        if k.startswith("GT@"):
            e = None
            print(f"  {k:18s}  {w['glob']:6.1f}  {w['near_glob']:8.1f}  {w['ref']:5.1f}  |  (reference)")
        else:
            e = G.evaluate(P, stamps, gcorr)["fit"]
            print(f"  {k:18s}  {w['glob']:6.1f}  {w['near_glob']:8.1f}  {w['ref']:5.1f}  |  {e['ate_rmse_mm']:7.1f} {e['ate_median_mm']:7.1f} {e['ate_p90_mm']:6.1f}"
                  f"  | {e['rpe_1m_trans_median_mm']:5.1f} {e['rpe_5m_trans_median_mm']:6.1f}  {e['yaw_err_median_deg']:5.2f}")
        out["sources"][k] = {"wall": w, "ate_rpe": e}
    os.makedirs(f"{C.ROOT}/analysis/results", exist_ok=True)
    with open(f"{C.ROOT}/analysis/results/a6_{tag}.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    C.log_trial("a6_groundtruth.py", a.db, ",".join(srcs), {"bag": a.bag, "heading_offset_deg": a.heading_offset_deg,
                "deltas": deltas, "cheap": C.asdict(CHEAP)}, out, time.time() - t0,
                notes="analysis 6: GT-built map delta sweep; wall metric vs ATE")


if __name__ == "__main__":
    main()
