"""Analysis 1: wall spread binned by point range.

Question: is the ~24 mm global plateau a sensor noise floor or pose error?

For each wall patch (classified exactly as in the C++ metric), the spread of
the patch's points across the wall is recomputed using only points whose
horizontal range from their own scan origin falls in a bin.  Windows:
  scan   : same node only           -> single-scan sensor spread (floor)
  ref    : |dID| <= 30  (~1 m path) -> + short-term pose jitter
  local  : |dID| <= 150 (~6 m path)
  global : all visits               -> + misalignment between visits
If near-range global ~= near-range scan/ref, the poses are already as good as
the near-range sensor can show, and the plateau is noise.

Pose sources: odom (Node.pose), graph (Admin.opt_poses = exporter's
"rtabmap graph"), or any exporter pose CSV via --poses label=path.

Usage:  python3 a1_range_bins.py 09-29-2.db [--poses icp2=../output/<ts>/poses_icp_round_2.csv ...]
"""
import argparse
import json
import time

import numpy as np

import common as C

COARSE = [0.30, 1.30, 1.90, 2.50]
FINE = [0.30, 0.70, 1.00, 1.30, 1.60, 1.90, 2.20, 2.50]


def rng(nid, p):
    return np.hypot(p[:, 0], p[:, 1])


def fmt_bins(rows):
    lines = ["  range bin      scan    ref   local  global   (patches: scan/ref/local/global)"]
    for r in rows:
        lines.append("  {lo:4.2f}-{hi:4.2f} m  {scan_median_mm:6.1f} {ref_median_mm:6.1f} {local_median_mm:6.1f} "
                     "{global_median_mm:6.1f}   ({scan_n}/{ref_n}/{local_n}/{global_n})".format(**r))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--poses", nargs="*", default=[], help="label=path to exporter pose CSV")
    ap.add_argument("--sources", default="odom,graph")
    a = ap.parse_args()

    con = C.connect(a.db)
    graph = C.load_rtabmap_opt_poses(con)
    sources = {}
    if "odom" in a.sources:
        sources["odom"] = C.load_odom_poses(con, graph.keys())
    if "graph" in a.sources:
        sources["graph"] = graph
    for spec in a.poses:
        label, path = spec.split("=", 1)
        sources[label] = C.load_pose_csv(path)

    scans = C.clean_scans(a.db, graph.keys())
    bins = {"range_coarse": (rng, np.array(COARSE)), "range_fine": (rng, np.array(FINE))}
    for label, poses in sources.items():
        t0 = time.time()
        res = C.wall_metric(scans, poses, bins=bins)
        dt = time.time() - t0
        print(f"\n=== {a.db}  poses={label} ===")
        print("  all points: ref {ref_median_mm:.1f} / local {local_median_mm:.1f} / global "
              "{global_median_mm:.1f} mm median  (n={n})".format(**res))
        print(fmt_bins(res["range_coarse"]))
        print(fmt_bins(res["range_fine"]))
        C.log_trial("a1_range_bins.py", a.db, label,
                    {"wall": C.asdict(C.WallCfg()), "clean": C.asdict(C.CleanCfg()),
                     "coarse": COARSE, "fine": FINE},
                    res, dt, notes="analysis 1: range-binned wall spread")
        with open(f"{C.ROOT}/analysis/results/a1_{a.db.replace('.db','')}_{label}.json", "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    import os
    os.makedirs(f"{C.ROOT}/analysis/results", exist_ok=True)
    main()
