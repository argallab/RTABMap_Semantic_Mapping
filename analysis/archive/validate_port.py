"""Gate: the Python port must reproduce the C++ exporter's numbers before any
new analysis built on it is trusted.

C++ reference values (09-29-2.db, output/2026-09-30_12-11-47/metrics.txt):
  scan clean: scans=2629 dropped=23  pts raw=20341832 crop=18789973 kept=18702094
  walls rtabmap graph: ref 19.4 / local 20.6 / global 26.4 mm median
                       p90 34.1 / 36.8 / 47.5  (n=9780)
"""
import sys
import time

import numpy as np

import common as C

DB = sys.argv[1] if len(sys.argv) > 1 else "09-29-2.db"


def main():
    t0 = time.time()
    con = C.connect(DB)
    graph = C.load_rtabmap_opt_poses(con)
    print(f"[{DB}] Admin.opt_poses robot nodes: {len(graph)}")

    scans = C.clean_scans(DB, graph.keys())
    n_pts = sum(len(p) for p in scans.values())
    print(f"cleaned scans={len(scans)}  kept pts={n_pts}")

    wt = C.wall_metric(scans, graph)
    print("walls rtabmap graph (python): ref {ref_median_mm:.1f} / local {local_median_mm:.1f} / "
          "global {global_median_mm:.1f} mm median   p90 {ref_p90_mm:.1f} / {local_p90_mm:.1f} / "
          "{global_p90_mm:.1f}   (n={n})".format(**wt))

    odom = C.load_odom_poses(con, graph.keys())
    wo = C.wall_metric(scans, odom)
    print("walls odometry    (python): ref {ref_median_mm:.1f} / local {local_median_mm:.1f} / "
          "global {global_median_mm:.1f} mm median   p90 {ref_p90_mm:.1f} / {local_p90_mm:.1f} / "
          "{global_p90_mm:.1f}   (n={n})".format(**wo))

    dt = time.time() - t0
    C.log_trial("validate_port.py", DB, "rtabmap_graph(Admin.opt_poses)",
                {"clean": C.asdict(C.CleanCfg()), "wall": C.asdict(C.WallCfg())},
                {"scans": len(scans), "kept_pts": n_pts, "graph": wt, "odom": wo}, dt,
                notes="port validation vs C++ metrics.txt")
    print(f"runtime {dt:.0f} s")


if __name__ == "__main__":
    main()
