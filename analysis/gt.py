"""Ground-truth (ceiling camera) trajectory: loading, association, ATE / RPE.

Bag layout (e.g. databases/tf_10-07-01):
  /ground_truth/odom          nav_msgs/Odometry  ground_truth -> gt_base_link (~48 Hz)
  /ground_truth/initial_pose  PoseWithCovarianceStamped (start pose in ground_truth)
  /tf                         odom -> base_link (live EKF), map -> odom (live RTAB-Map)

Assumptions (stated, not hidden):
  * gt_base_link and base_link are the same body point and heading
    (gt_base_link was re-calibrated onto the drive axle on 2026-09-24).  An
    optional fitted body-frame lever arm tests this.
  * Evaluation is 2D (x, y, yaw); the pipeline is Force3DoF.
"""
from __future__ import annotations

import os

import numpy as np

import common as C


def _yaw(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def load_bag(bag_dir: str, use_cache=True) -> dict:
    """Returns {'gt': (N,4) [t,x,y,yaw], 'gt_cov': (N,), 'initial_pose': [(t,x,y,yaw),...],
    'odom_tf': (M,4), 'map_odom_tf': (K,4)}; header stamps (s)."""
    name = os.path.basename(os.path.normpath(bag_dir))
    cpath = os.path.join(C.CACHE_DIR, f"gt_{name}.npz")
    if use_cache and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        return {k: z[k] for k in z.files}
    from pathlib import Path
    from rosbags.highlevel import AnyReader
    gt, cov, init, odom, mo = [], [], [], [], []
    with AnyReader([Path(bag_dir)]) as r:
        want = {"/ground_truth/odom", "/ground_truth/initial_pose", "/tf"}
        conns = [c for c in r.connections if c.topic in want]
        for c, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            if c.topic == "/tf":
                for tr in m.transforms:
                    key = (tr.header.frame_id, tr.child_frame_id)
                    if key not in (("odom", "base_link"), ("map", "odom")):
                        continue
                    t = tr.header.stamp.sec + 1e-9 * tr.header.stamp.nanosec
                    row = (t, tr.transform.translation.x, tr.transform.translation.y, _yaw(tr.transform.rotation))
                    (odom if key[0] == "odom" else mo).append(row)
                continue
            t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
            p = m.pose.pose
            row = (t, p.position.x, p.position.y, _yaw(p.orientation))
            if c.topic == "/ground_truth/odom":
                gt.append(row)
                cov.append(m.pose.covariance[0])
            else:
                init.append(row)
    out = dict(gt=np.array(sorted(gt)), gt_cov=np.array(cov), initial_pose=np.array(sorted(init)),
               odom_tf=np.array(sorted(odom)), map_odom_tf=np.array(sorted(mo)))
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    np.savez(cpath, **out)
    return out


def interp_pose(series: np.ndarray, t: np.ndarray, max_gap=0.1):
    """Linear interpolation of (t,x,y,yaw) at times t; NaN where the bracketing
    samples are more than max_gap s apart or t is outside the series."""
    ts = series[:, 0]
    i = np.searchsorted(ts, t)
    ok = (i > 0) & (i < len(ts))
    i = np.clip(i, 1, len(ts) - 1)
    t0, t1 = ts[i - 1], ts[i]
    ok &= (t1 - t0) <= max_gap
    a = np.where(t1 > t0, (t - t0) / np.maximum(t1 - t0, 1e-9), 0.0)
    x = series[i - 1, 1] + a * (series[i, 1] - series[i - 1, 1])
    y = series[i - 1, 2] + a * (series[i, 2] - series[i - 1, 2])
    dyaw = np.arctan2(np.sin(series[i, 3] - series[i - 1, 3]), np.cos(series[i, 3] - series[i - 1, 3]))
    yaw = series[i - 1, 3] + a * dyaw
    out = np.stack([x, y, yaw], axis=1)
    out[~ok] = np.nan
    return out


def poses_to_xyyaw(poses: dict, ids):
    return np.array([[poses[i][0, 3], poses[i][1, 3], np.arctan2(poses[i][1, 0], poses[i][0, 0])] for i in ids])


def fit_se2(src: np.ndarray, dst: np.ndarray):
    """Least-squares rigid 2D transform (R, t) with dst ~ R @ src + t (Umeyama, no scale)."""
    ms, md = src.mean(0), dst.mean(0)
    H = (src - ms).T @ (dst - md)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, md - R @ ms


def apply_se2(R, t, xyyaw):
    xy = xyyaw[:, :2] @ R.T + t
    yaw = xyyaw[:, 2] + np.arctan2(R[1, 0], R[0, 0])
    return np.column_stack([xy, np.arctan2(np.sin(yaw), np.cos(yaw))])


def ate_rpe(est: np.ndarray, ref: np.ndarray, stamps: np.ndarray, path_deltas=(1.0, 5.0)):
    """est/ref: aligned (N,3) x,y,yaw at the same stamps.  ATE: position error
    stats (mm) and heading error (deg).  RPE: for each pair (i, j) with ~d m of
    reference path between them, compare the relative motion."""
    e = est[:, :2] - ref[:, :2]
    d = np.linalg.norm(e, axis=1) * 1000
    dyaw = np.degrees(np.abs(np.arctan2(np.sin(est[:, 2] - ref[:, 2]), np.cos(est[:, 2] - ref[:, 2]))))
    out = dict(n=int(len(d)), ate_rmse_mm=float(np.sqrt((d ** 2).mean())), ate_median_mm=float(np.median(d)),
               ate_p90_mm=float(np.percentile(d, 90)), ate_max_mm=float(d.max()),
               yaw_err_median_deg=float(np.median(dyaw)), yaw_err_p90_deg=float(np.percentile(dyaw, 90)))
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ref[:, :2], axis=0), axis=1))])

    def rel(p, i, j):
        c, s = np.cos(p[i, 2]), np.sin(p[i, 2])
        dx, dy = p[j, 0] - p[i, 0], p[j, 1] - p[i, 1]
        return np.array([c * dx + s * dy, -s * dx + c * dy]), np.arctan2(np.sin(p[j, 2] - p[i, 2]), np.cos(p[j, 2] - p[i, 2]))

    for D in path_deltas:
        j = np.searchsorted(cum, cum + D)
        sel = np.where(j < len(cum))[0]
        tr, ro = [], []
        for i in sel:
            a, ya = rel(est, i, j[i]); b, yb = rel(ref, i, j[i])
            tr.append(np.linalg.norm(a - b) * 1000)
            ro.append(abs(np.degrees(np.arctan2(np.sin(ya - yb), np.cos(ya - yb)))))
        if tr:
            out[f"rpe_{D:g}m_trans_median_mm"] = float(np.median(tr))
            out[f"rpe_{D:g}m_trans_p90_mm"] = float(np.percentile(tr, 90))
            out[f"rpe_{D:g}m_rot_median_deg"] = float(np.median(ro))
    return out


def evaluate(poses: dict, stamps: dict, gt: np.ndarray, time_offset=0.0, anchor=None):
    """ATE/RPE of `poses` (id -> 4x4) against ground truth.
    time_offset: GT is sampled at node_stamp + time_offset.
    anchor: optional (x, y, yaw) of the FIRST node in GT; if given, also report
    anchored ATE (no fit).  Returns {'fit': {...}, 'anchored': {...}, 'T_gt_map': ...}."""
    ids = [i for i in sorted(poses) if i in stamps]
    ts = np.array([stamps[i] for i in ids])
    ref = interp_pose(gt, ts + time_offset)
    est = poses_to_xyyaw(poses, ids)
    ok = np.isfinite(ref).all(axis=1)
    est, ref, ts = est[ok], ref[ok], ts[ok]
    R, t = fit_se2(est[:, :2], ref[:, :2])
    res = {"coverage": float(ok.mean()), "fit": ate_rpe(apply_se2(R, t, est), ref, ts),
           "T_gt_map": [float(t[0]), float(t[1]), float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))]}
    if anchor is not None:
        th = anchor[2] - est[0, 2]
        Ra = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        ta = np.array(anchor[:2]) - Ra @ est[0, :2]
        res["anchored"] = ate_rpe(apply_se2(Ra, ta, est), ref, ts)
    return res
