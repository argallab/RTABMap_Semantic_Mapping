"""Analysis 1b: does the residual pose error come from motion (timestamp
latency) or from a fixed sensor offset?

Parts (select with --parts):
  bins   wall spread binned by the node's turn rate and speed, for all / near
         (<1.3 m) / far (>1.9 m) points.  Latency error ~ (turn rate x latency)
         x range; sensor noise does not depend on motion.
  dt     time-offset sweep: each node's pose is shifted along its local
         odometry twist by delta, T' = T * SE2(v*delta, w*delta), and the wall
         metric is recomputed.  delta < 0 means "the scan was captured before
         its stamp".
  rigid  joint grid over a fixed base_camera->base_link offset (dx, dy, dyaw)
         applied to every scan, and delta.

Poses are held fixed (no re-optimization): this is a first-order test of
whether the data prefer a non-zero latency / offset, not a corrected map.

Usage: python3 a1b_motion.py 09-29-2.db [--poses label=path] [--source graph]
"""
import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import common as C

TURN_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 10.0]     # |w| rad/s
SPEED_EDGES = [0.0, 0.02, 0.10, 0.20, 0.30, 0.50, 10.0]    # |v| m/s
CHEAP = C.WallCfg(target_pts=750_000, queries=5_000)       # for sweeps
DELTAS = [round(x, 2) for x in np.arange(-0.20, 0.2001, 0.02)]

G = {}  # globals shared with forked workers


def twists(con):
    """Body-frame twist (vx, vy, w) per node from odometry, central difference
    over the neighbouring nodes in stamp order (all nodes, incl. weight -9)."""
    nodes = C.load_nodes(con)
    ids = sorted(nodes, key=lambda i: nodes[i][0])
    tw = {}
    for k, i in enumerate(ids):
        a, b = ids[max(0, k - 1)], ids[min(len(ids) - 1, k + 1)]
        ta, tb = nodes[a][0], nodes[b][0]
        dt = tb - ta
        if dt <= 1e-3 or dt > 2.0:
            tw[i] = (0.0, 0.0, 0.0, dt)
            continue
        rel = np.linalg.inv(nodes[a][2]) @ nodes[b][2]
        yaw = np.arctan2(rel[1, 0], rel[0, 0])
        tw[i] = (rel[0, 3] / dt, rel[1, 3] / dt, yaw / dt, dt)
    return tw


def se2(x, y, th):
    T = np.eye(4)
    c, s = np.cos(th), np.sin(th)
    T[:2, :2] = [[c, -s], [s, c]]
    T[0, 3], T[1, 3] = x, y
    return T


def shifted(poses, tw, delta, mode="full"):
    """mode: full | rot (yaw part only) | trans (translation part only)."""
    if delta == 0.0:
        return poses
    kt = 0.0 if mode == "rot" else 1.0
    kr = 0.0 if mode == "trans" else 1.0
    out = {}
    for i, T in poses.items():
        vx, vy, w, _ = tw.get(i, (0.0, 0.0, 0.0, 0.0))
        out[i] = T @ se2(kt * vx * delta, kt * vy * delta, kr * w * delta)
    return out


def offset_scans(scans, dx, dy, dyaw):
    if dx == 0 and dy == 0 and dyaw == 0:
        return scans
    R = se2(dx, dy, dyaw)
    return {i: (p.astype(np.float64) @ R[:3, :3].T + R[:3, 3]).astype(np.float32) for i, p in scans.items()}


def near(nid, p):
    r = np.hypot(p[:, 0], p[:, 1])
    return np.where(r < 1.3, 0.5, np.nan)


def summarize(res):
    nb = res["near"][0]
    return dict(n=res["n"], ref=res["ref_median_mm"], local=res["local_median_mm"],
                glob=res["global_median_mm"], near_ref=nb["ref_median_mm"],
                near_glob=nb["global_median_mm"])


def shifted2(poses, tw, d_trans, d_rot):
    """Separate delays for the translation and yaw parts of the twist."""
    if d_trans == 0.0 and d_rot == 0.0:
        return poses
    out = {}
    for i, T in poses.items():
        vx, vy, w, _ = tw.get(i, (0.0, 0.0, 0.0, 0.0))
        out[i] = T @ se2(vx * d_trans, vy * d_trans, w * d_rot)
    return out


def eval_point(args):
    if isinstance(args, dict):   # {d_trans, d_rot, dx, dy, dyaw, full}
        a = args
        poses = shifted2(G["poses"], G["tw"], a["d_trans"], a["d_rot"])
        scans = offset_scans(G["scans"], a["dx"], a["dy"], a["dyaw"])
        res = C.wall_metric(scans, poses, C.WallCfg() if a["full"] else CHEAP,
                            bins={"near": (near, np.array([0.0, 1.0]))})
        return dict(d_trans=a["d_trans"], d_rot=a["d_rot"], dx=a["dx"], dy=a["dy"],
                    dyaw_deg=float(np.degrees(a["dyaw"])), full=a["full"], **summarize(res))
    delta, dx, dy, dyaw, full = args[:5]
    mode = args[5] if len(args) > 5 else "full"
    poses = shifted(G["poses"], G["tw"], delta, mode)
    scans = offset_scans(G["scans"], dx, dy, dyaw)
    res = C.wall_metric(scans, poses, C.WallCfg() if full else CHEAP,
                        bins={"near": (near, np.array([0.0, 1.0]))})
    return dict(delta=delta, dx=dx, dy=dy, dyaw_deg=np.degrees(dyaw), full=full, mode=mode, **summarize(res))


def run_pool(points, procs):
    with Pool(procs) as pool:
        return pool.map(eval_point, points, chunksize=1)


def part_bins(db, label, poses, scans, tw):
    def turn(nid, p, sel=None):
        v = np.full(len(p), abs(tw.get(nid, (0, 0, 0, 0))[2]))
        return v
    def speed(nid, p):
        vx, vy, _, _ = tw.get(nid, (0, 0, 0, 0))
        return np.full(len(p), np.hypot(vx, vy))
    def masked(fn, lo, hi):
        def f(nid, p):
            r = np.hypot(p[:, 0], p[:, 1])
            v = fn(nid, p).astype(float)
            v[(r < lo) | (r >= hi)] = np.nan
            return v
        return f
    bins = {}
    for name, fn, edges in (("turn", turn, TURN_EDGES), ("speed", speed, SPEED_EDGES)):
        for rname, lo, hi in (("all", 0, 9), ("near", 0, 1.3), ("far", 1.9, 9)):
            bins[f"{name}_{rname}"] = (masked(fn, lo, hi), np.array(edges))
    t0 = time.time()
    res = C.wall_metric(scans, poses, bins=bins)
    w = np.array([abs(tw[i][2]) for i in poses if i in tw])
    v = np.array([np.hypot(tw[i][0], tw[i][1]) for i in poses if i in tw])
    print(f"\n=== bins  {db}  poses={label} ===")
    print("node |w| rad/s percentiles 10/50/90/99: " + " ".join(f"{np.percentile(w, q):.3f}" for q in (10, 50, 90, 99)))
    print("node |v| m/s   percentiles 10/50/90/99: " + " ".join(f"{np.percentile(v, q):.3f}" for q in (10, 50, 90, 99)))
    for key in bins:
        print(f"  {key:11s}  bin            scan    ref  local global   (patches ref/global)")
        for r in res[key]:
            print("              {lo:5.2f}-{hi:5.2f}  {scan_median_mm:6.1f} {ref_median_mm:6.1f} {local_median_mm:6.1f} "
                  "{global_median_mm:6.1f}   ({ref_n}/{global_n})".format(**r))
    C.log_trial("a1b_motion.py:bins", db, label,
                {"turn_edges": TURN_EDGES, "speed_edges": SPEED_EDGES, "wall": C.asdict(C.WallCfg())},
                res, time.time() - t0, notes="analysis 1b: spread by turn rate / speed")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--source", default="graph", help="odom | graph | label of a --poses CSV")
    ap.add_argument("--poses", nargs="*", default=[])
    ap.add_argument("--parts", default="bins,dt,rigid")
    ap.add_argument("--procs", type=int, default=4)   # ~1 GB peak per worker; keep under the 8 GB cap
    ap.add_argument("--ctrl-delta", type=float, default=-0.16)
    a = ap.parse_args()

    con = C.connect(a.db)
    graph = C.load_rtabmap_opt_poses(con)
    srcs = {"graph": graph, "odom": C.load_odom_poses(con, graph.keys())}
    for spec in a.poses:
        k, p = spec.split("=", 1)
        srcs[k] = C.load_pose_csv(p)
    poses = srcs[a.source]
    scans = C.clean_scans(a.db, graph.keys())
    tw = twists(con)
    G.update(poses=poses, scans=scans, tw=tw)
    tag = f"{a.db.replace('.db', '')}_{a.source}"
    os.makedirs(f"{C.ROOT}/analysis/results", exist_ok=True)
    out = {}

    if "bins" in a.parts:
        out["bins"] = part_bins(a.db, a.source, poses, scans, tw)

    if "dt" in a.parts:
        t0 = time.time()
        pts = [(d, 0.0, 0.0, 0.0, False) for d in DELTAS]
        rows = sorted(run_pool(pts, a.procs), key=lambda r: r["delta"])
        best = min(rows, key=lambda r: r["glob"])
        bestn = min(rows, key=lambda r: r["near_glob"])
        full = run_pool([(0.0, 0, 0, 0, True), (best["delta"], 0, 0, 0, True),
                         (bestn["delta"], 0, 0, 0, True)], 3)
        print(f"\n=== time-offset sweep  {a.db}  poses={a.source}  (cheap cfg: 750k pts / 5k queries) ===")
        print("  delta(s)   global   near_global   ref   near_ref   n")
        for r in rows:
            print("  {delta:+6.2f}   {glob:6.1f}   {near_glob:8.1f}   {ref:6.1f}  {near_ref:6.1f}   {n}".format(**r))
        print("  full-cfg confirmation:")
        for r in full:
            print("  {delta:+6.2f}   {glob:6.1f}   {near_glob:8.1f}   {ref:6.1f}  {near_ref:6.1f}   {n}".format(**r))
        out["dt"] = {"rows": rows, "full": full}
        C.log_trial("a1b_motion.py:dt", a.db, a.source, {"deltas": DELTAS, "cheap": C.asdict(CHEAP)},
                    out["dt"], time.time() - t0, notes="analysis 1b: time-offset sweep along odom twist")

    if "ctrl" in a.parts:
        # Which part of the twist shift carries the gain: yaw or translation?
        t0 = time.time()
        d = a.ctrl_delta
        rows = run_pool([(0.0, 0, 0, 0, True), (d, 0, 0, 0, True, "full"),
                         (d, 0, 0, 0, True, "rot"), (d, 0, 0, 0, True, "trans")], min(a.procs, 4))
        print(f"\n=== twist-shift control  {a.db}  poses={a.source}  delta={d:+.2f} s (full cfg) ===")
        print("  mode    global  near_global   ref  near_ref    n")
        for r in rows:
            print("  {mode:6s}  {glob:6.1f}  {near_glob:8.1f}  {ref:6.1f}  {near_ref:6.1f}  {n}".format(
                **dict(r, mode=r["mode"] if r["delta"] else "zero")))
        out["ctrl"] = rows
        C.log_trial("a1b_motion.py:ctrl", a.db, a.source, {"delta": d}, rows, time.time() - t0,
                    notes="analysis 1b: yaw-only vs translation-only twist shift")

    def show(title, rows, keys):
        rows = sorted(rows, key=lambda r: r["glob"])
        zero = [r for r in rows if all(r[k] == 0 for k in keys)]
        print(f"\n=== {title}  {a.db}  poses={a.source}  ({len(rows)} pts, cheap cfg) ===")
        if zero:
            z = zero[0]
            print(f"  all-zero: global {z['glob']:.1f}  near_global {z['near_glob']:.1f}  ref {z['ref']:.1f}")
        print("  best 10 by global:  " + " ".join(f"{k:>8s}" for k in keys) + "   global near_glob   ref")
        for r in rows[:10]:
            print("                      " + " ".join(f"{r[k]:+8.3f}" for k in keys) +
                  "   {glob:6.1f}  {near_glob:6.1f}  {ref:5.1f}".format(**r))
        for k in keys:   # profile: best global at each value of k
            vals = sorted({r[k] for r in rows})
            print(f"  profile {k:9s}: " + "  ".join(
                f"{v:+.3f}:{min(r['glob'] for r in rows if r[k] == v):.1f}" for v in vals))
        return rows

    if "dt2" in a.parts:
        # Separate translation / yaw delays (wheel vs IMU paths in the EKF can lag differently).
        t0 = time.time()
        grid = [dict(d_trans=float(t), d_rot=float(r), dx=0.0, dy=0.0, dyaw=0.0, full=False)
                for t in np.round(np.arange(-0.24, 0.0001, 0.04), 2)
                for r in np.round(np.arange(-0.20, 0.1001, 0.05), 2)]
        rows = show("translation-delay x yaw-delay grid", run_pool(grid, a.procs), ["d_trans", "d_rot"])
        out["dt2"] = rows
        C.log_trial("a1b_motion.py:dt2", a.db, a.source, {"grid": "d_trans -0.24..0 /0.04, d_rot -0.2..0.1 /0.05",
                    "cheap": C.asdict(CHEAP)}, {"best": rows[:5]}, time.time() - t0,
                    notes="analysis 1b: separate translation/yaw delay")

    if "rigid" in a.parts:
        # Fixed base_camera->base_link offset, jointly with the translation delay.
        # A constant lever arm and a speed-proportional latency both shift points
        # along track; only the latency scales with speed, so they separate here.
        t0 = time.time()
        grid = [dict(d_trans=float(t), d_rot=0.0, dx=float(x), dy=float(y), dyaw=float(np.radians(th)), full=False)
                for t in (0.0, a.ctrl_delta)
                for x in (-0.10, -0.05, 0.0, 0.05, 0.10)
                for y in (-0.10, -0.05, 0.0, 0.05, 0.10)
                for th in (-1.0, 0.0, 1.0)]
        rows = show("rigid offset x translation-delay grid", run_pool(grid, a.procs),
                    ["d_trans", "dx", "dy", "dyaw_deg"])
        out["rigid"] = rows
        C.log_trial("a1b_motion.py:rigid", a.db, a.source,
                    {"d_trans": [0.0, a.ctrl_delta], "dx": [-0.1, -0.05, 0, 0.05, 0.1],
                     "dy": [-0.1, -0.05, 0, 0.05, 0.1], "dyaw_deg": [-1, 0, 1], "cheap": C.asdict(CHEAP)},
                    {"best": rows[:5]}, time.time() - t0, notes="analysis 1b: rigid offset x delay grid")

    with open(f"{C.ROOT}/analysis/results/a1b_{tag}.json", "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
