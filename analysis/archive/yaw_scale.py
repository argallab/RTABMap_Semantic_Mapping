"""Odometry turn / distance scale vs ceiling-camera ground truth.

For consecutive graph nodes, compares the odometry (Node.pose) heading change
and travelled distance with the GT ones at the same stamps.  The yaw ratio is
the least-squares slope dθ_odom = k · dθ_gt over 1 s windows that turn
>= 2° (whole-run cumulative ratio printed too); WC_ODOM_YAW_SCALE = 1 / k.
Also reports left/right turns separately, the distance ratio, and the GT
heading offset on straight driving (camera yaw calibration).

Usage: python3 yaw_scale.py 10-12-01.db tf_10-12-01 [more db bag pairs...]
"""
import os
import sys

import numpy as np

import common as C
import gt as G


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def analyse(db, bag):
    con = C.connect(db)
    nodes = C.load_nodes(con)
    ids = sorted(i for i in nodes if i > 0)
    t = np.array([nodes[i][0] for i in ids])
    odo = G.poses_to_xyyaw(C.load_odom_poses(con), ids)
    g = G.load_bag(os.path.join(C.DB_DIR, bag))["gt"]
    gq = G.interp_pose(g, t)
    ok = np.isfinite(gq).all(1)

    # 1 s windows (resampled by node stamps), both endpoints with GT
    win = []
    j = 0
    for i in range(len(ids)):
        while j < len(ids) and t[j] - t[i] < 1.0:
            j += 1
        if j >= len(ids):
            break
        if ok[i] and ok[j]:
            win.append((wrap(odo[j, 2] - odo[i, 2]), wrap(gq[j, 2] - gq[i, 2]),
                        np.hypot(*(odo[j, :2] - odo[i, :2])), np.hypot(*(gq[j, :2] - gq[i, :2]))))
    w = np.array(win)
    turn = np.abs(w[:, 1]) > np.radians(2)
    k = (w[turn, 0] @ w[turn, 1]) / (w[turn, 1] @ w[turn, 1])
    left, right = turn & (w[:, 1] > 0), turn & (w[:, 1] < 0)
    kl = (w[left, 0] @ w[left, 1]) / (w[left, 1] @ w[left, 1])
    kr = (w[right, 0] @ w[right, 1]) / (w[right, 1] @ w[right, 1])
    # bootstrap spread of k
    rng = np.random.default_rng(0)
    idx = np.nonzero(turn)[0]
    boots = []
    for _ in range(200):
        s = rng.choice(idx, len(idx))
        boots.append((w[s, 0] @ w[s, 1]) / (w[s, 1] @ w[s, 1]))
    # cumulative over consecutive GT-covered steps
    st = ok[:-1] & ok[1:]
    do = wrap(np.diff(odo[:, 2]))[st]
    dg = wrap(np.diff(gq[:, 2]))[st]
    cum = np.abs(do).sum() / np.abs(dg).sum()
    dist = (np.hypot(*np.diff(odo[:, :2], axis=0).T)[st].sum() /
            np.hypot(*np.diff(gq[:, :2], axis=0).T)[st].sum())
    # GT heading offset on straight driving: travel direction vs GT heading
    offs = []
    j = 0
    for i in range(len(ids)):
        while j < len(ids) and t[j] - t[i] < 1.0:
            j += 1
        if j >= len(ids):
            break
        if ok[i] and ok[j]:
            d = gq[j, :2] - gq[i, :2]
            if np.hypot(*d) > 0.15 and abs(wrap(gq[j, 2] - gq[i, 2])) < np.radians(0.5):
                offs.append(np.degrees(wrap(np.arctan2(d[1], d[0]) - gq[i, 2])))
    offs = np.array(offs)
    fwd = offs[np.abs(offs) < 90]
    print(f"\n=== {db}  ({len(ids)} nodes, GT on {ok.mean()*100:.0f}%)")
    print(f"  yaw ratio odom/GT, 1 s windows turning >= 2 deg: k = {k:.4f}  "
          f"(bootstrap 5-95%: {np.percentile(boots, 5):.4f} - {np.percentile(boots, 95):.4f}, n={turn.sum()})")
    print(f"     left turns {kl:.4f} (n={left.sum()})   right turns {kr:.4f} (n={right.sum()})")
    print(f"  cumulative |turn| ratio over the run: {cum:.4f}   (total GT turn {np.degrees(np.abs(dg).sum()):.0f} deg)")
    print(f"  distance ratio odom/GT: {dist:.4f}")
    print(f"  -> WC_ODOM_YAW_SCALE = 1/k = {1/k:.4f}   (current setting 0.9606)")
    if len(fwd):
        # travel direction - GT heading: add this to GT yaw (same sign convention as --heading-offset-deg)
        print(f"  GT heading offset on straight driving: {np.median(fwd):+.2f} deg  (n={len(fwd)}; rough, lever arm not removed)")
    return k


if __name__ == "__main__":
    a = sys.argv[1:]
    for db, bag in zip(a[::2], a[1::2]):
        analyse(db, bag)
