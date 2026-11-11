"""Score pipeline variants (tools/tag_holdout dumps, exporter dumps) against
ground truth: ATE / RPE per stage, GT heading-corrected, SE(2)-fit and anchored
at the first node.  Wall metric (global, near-global) is reported alongside but
is only meaningful for small refinements (it is blind to errors > ~20 cm).

Usage: python3 a6_score_variants.py 10-07-01.db tf_10-07-01 holdout/10-07-01 [--walls]
"""
import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import common as C
import gt as G

STAGES = ["rtabmap_graph", "tag_GTSAM", "icp_round_1", "icp_round_2"]
W = {}


def walls(args):
    label, path = args
    P = C.load_pose_csv(path) if isinstance(path, str) else path
    r = C.wall_metric(W["scans"], P)
    return label, r["global_median_mm"], r["ref_median_mm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db"); ap.add_argument("bag"); ap.add_argument("dir")
    ap.add_argument("--heading-offset-deg", type=float, default=-4.08)
    ap.add_argument("--walls", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    con = C.connect(a.db)
    nodes = C.load_nodes(con)
    stamps = {i: v[0] for i, v in nodes.items()}
    live = C.load_rtabmap_opt_poses(con)
    bag = G.load_bag(os.path.join(C.DB_DIR, a.bag))
    g = bag["gt"].copy(); g[:, 3] += np.radians(a.heading_offset_deg)
    start = G.interp_pose(g, np.array([stamps[min(live)]]))[0]

    rows = [("odom", "-", {i: nodes[i][2] for i in live}), ("live graph", "-", live)]
    for pre in sorted({p[:-len("_heldout.txt")] for p in glob.glob(f"{a.dir}/*_heldout.txt")}):
        for st in STAGES:
            f = f"{pre}_{st}.csv"
            if os.path.exists(f):
                rows.append((os.path.basename(pre), st, f))

    import re
    def toff(name):
        # shifted variants (..._s010 / _s015) represent poses at stamp - 0.10 / 0.15 s
        m = re.search(r"_s0(\d\d)$", name)
        return -int(m.group(1)) / 100.0 if m else 0.0

    res = []
    for name, st, src in rows:
        P = C.load_pose_csv(src) if isinstance(src, str) else src
        e = G.evaluate(P, stamps, g, time_offset=toff(name), anchor=start)
        res.append(dict(variant=name, stage=st, fit=e["fit"], anchored=e["anchored"], src=src if isinstance(src, str) else None))
    if a.walls:
        W["scans"] = C.clean_scans(a.db, live.keys())
        jobs = [(f"{r['variant']}|{r['stage']}", r["src"] if r["src"] else (rows[i][2])) for i, r in enumerate(res)]
        with Pool(4) as pool:
            wl = dict((k, (gl, rf)) for k, gl, rf in pool.map(walls, jobs, chunksize=1))
        for r in res:
            r["wall_global"], r["wall_ref"] = wl[f"{r['variant']}|{r['stage']}"]

    print(f"\n=== {a.db} vs ground truth (heading offset {a.heading_offset_deg:+.2f} deg) ===")
    print(f"  {'variant':10s} {'stage':14s} | ATE fit rmse/med/p90 (mm) | anchored rmse | RPE1m RPE5m (mm) | yaw err | walls glob/ref")
    for r in res:
        f, an = r["fit"], r["anchored"]
        wtxt = f"{r['wall_global']:5.1f}/{r['wall_ref']:4.1f}" if "wall_global" in r else ""
        print(f"  {r['variant']:10s} {r['stage']:14s} | {f['ate_rmse_mm']:7.1f} {f['ate_median_mm']:6.1f} {f['ate_p90_mm']:6.1f} |"
              f" {an['ate_rmse_mm']:9.1f}   | {f['rpe_1m_trans_median_mm']:5.1f} {f['rpe_5m_trans_median_mm']:6.1f}    |"
              f" {f['yaw_err_median_deg']:5.2f}   | {wtxt}")
    os.makedirs(f"{C.ROOT}/analysis/results", exist_ok=True)
    tag = a.db.replace(".db", "")
    with open(f"{C.ROOT}/analysis/results/a6_variants_{tag}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    C.log_trial("a6_score_variants.py", a.db, a.dir, {"bag": a.bag, "heading_offset_deg": a.heading_offset_deg},
                [{k: v for k, v in r.items() if k != "src"} for r in res], time.time() - t0,
                notes="ATE/RPE of pipeline variants vs ceiling-camera GT")


if __name__ == "__main__":
    main()
