"""Pose graph rebuilt from the raw bag: /wheel_odom + every tag36h11 detection on /tf.

The RTAB-Map database keeps only keyframe detections and uses the EKF output
(/odometry/filtered) as odometry.  This rebuilds the graph from the recording
itself, on the database's node stamps (so the exporter's scans still apply):
  * odometry links between consecutive nodes from /wheel_odom (interpolated at the node stamps)
  * every tag36h11 detection (/tf child 'tag36h11:<id>', parent a RealSense
    optical frame) moved into base_link with /tf_static, attached to the nearest
    node in time with the wheel-odometry motion between the two stamps
then compares all tags / random halves / no tags against ground truth, with
both the whole-drive fit (ATE) and drift (aligned on the first 60 s only), and
rebuilds the map for each from the run's scans.

  python3 bag_graph.py R01_02.db ../output/<run> [--draws 10] [--exclude 44] [--odom-scale 16] [--cam-yaw 3]

Writes analysis/<db>_tag_ablation/ next to the run, i.e. output/<db>/analysis/ (or --out): maps.png, trajectories.png, summary.txt
"""
import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np

import common as C
import gt as G
import grid_sim as GS
import tag_ablation as TA
import thin_walls as TW

wrap = lambda a: np.arctan2(np.sin(a), np.cos(a))


def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tf_mat(tr):
    T = np.eye(4)
    T[:3, :3] = quat_mat((tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w))
    T[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
    return T


def read_bag(bag_dir, cache):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["odom"], z["det"], z["frames"].tolist()
    from rosbags.highlevel import AnyReader
    static, odom, det, frames, tf_odom = {}, [], [], [], []
    with AnyReader([Path(bag_dir)]) as r:
        conns = [c for c in r.connections if c.topic in ("/tf", "/tf_static", "/wheel_odom")]
        for c, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            if c.topic == "/wheel_odom":
                p = m.pose.pose
                yaw = np.arctan2(2 * (p.orientation.w * p.orientation.z + p.orientation.x * p.orientation.y),
                                 1 - 2 * (p.orientation.y ** 2 + p.orientation.z ** 2))
                odom.append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, p.position.x, p.position.y, yaw))
                continue
            for tr in m.transforms:
                if c.topic == "/tf_static":
                    static[tr.child_frame_id] = (tr.header.frame_id, tf_mat(tr.transform))
                elif (tr.header.frame_id, tr.child_frame_id) == ("odom", "base_link"):
                    q_ = tr.transform.rotation
                    tf_odom.append((tr.header.stamp.sec + 1e-9 * tr.header.stamp.nanosec, tr.transform.translation.x, tr.transform.translation.y,
                                    np.arctan2(2 * (q_.w * q_.z + q_.x * q_.y), 1 - 2 * (q_.y ** 2 + q_.z ** 2))))
                elif tr.child_frame_id.startswith("tag36h11:"):
                    if tr.header.frame_id not in frames:
                        frames.append(tr.header.frame_id)
                    T = tf_mat(tr.transform)
                    det.append((tr.header.stamp.sec + 1e-9 * tr.header.stamp.nanosec, int(tr.child_frame_id.split(":")[1]),
                                frames.index(tr.header.frame_id), *T[:3, 3]))
    # base_link <- each detection frame, from the static tree
    base_T = []
    for f in frames:
        T = np.eye(4); cur = f
        while cur != "base_link":
            parent, Tp = static[cur]
            T = Tp @ T; cur = parent
        base_T.append(T)
    if not odom:                                   # no /wheel_odom in this bag: /tf odom->base_link (filtered odometry)
        print(f"[bag] no /wheel_odom: using /tf odom->base_link ({len(tf_odom)} samples)")
        odom = tf_odom
    odom = np.array(sorted(odom)); det = np.array(sorted(det))
    # tag position in base_link
    pts = np.c_[det[:, 3:6], np.ones(len(det))]
    pb = np.array([base_T[int(f)] @ p for f, p in zip(det[:, 2], pts)])[:, :3]
    det = np.c_[det[:, :2], pb]                    # t, tag id, x, y, z in base_link
    np.savez(cache, odom=odom, det=det, frames=np.array(frames, dtype=object))
    return odom, det, frames


def graph_inputs(db, exclude=(44,), cam_yaw=3.0, max_dt=0.25):
    """Graph inputs rebuilt from the bag (see module doc).  Returns
    (frames, nodes, ids, stamps, wheel poses, odometry links, tag detections, tags, GT at nodes, GT ok mask)."""
    name = db.replace(".db", "")
    odom, det, frames = read_bag(os.path.join(C.DB_DIR, "tf_" + name), os.path.join(C.CACHE_DIR, f"bag_graph_{name}.npz"))

    con = C.connect(db); nodes = C.load_nodes(con)
    ids = sorted(C.load_rtabmap_opt_poses(con)); N = len(ids)
    stamps = np.array([nodes[i][0] for i in ids])
    wo = G.interp_pose(odom, stamps, max_gap=0.2)          # wheel odometry at the node stamps
    if not np.isfinite(wo).all():
        bad = ~np.isfinite(wo).all(1); wo[bad] = G.interp_pose(odom, stamps[bad], max_gap=5.0)[:]
    # odometry links between consecutive nodes, DB default information (sigma 3.16 cm / 1.81 deg)
    sx, sy, sth = 1 / 0.0316, 1 / 0.0316, 1 / np.radians(1.81)
    odl = []
    for k in range(N - 1):
        p, q_ = wo[k], wo[k + 1]
        c, s = np.cos(p[2]), np.sin(p[2]); dx, dy = q_[0] - p[0], q_[1] - p[1]
        odl.append((k, k + 1, (c * dx + s * dy, -s * dx + c * dy, wrap(q_[2] - p[2])), sx, sy, sth))
    # detections -> nearest node, compensating the wheel motion between the stamps
    cy, sy_ = np.cos(np.radians(cam_yaw)), np.sin(np.radians(cam_yaw))
    dw = G.interp_pose(odom, det[:, 0], max_gap=0.2)
    near = np.clip(np.searchsorted(stamps, det[:, 0]), 1, N - 1)
    near = np.where(np.abs(stamps[near - 1] - det[:, 0]) < np.abs(stamps[near] - det[:, 0]), near - 1, near)
    lml = []
    for d, n, w in zip(det, near, dw):
        if abs(stamps[n] - d[0]) > max_dt or not np.isfinite(w).all() or int(d[1]) in exclude:
            continue
        x, y = d[2], d[3]
        x, y = cy * x - sy_ * y, sy_ * x + cy * y                    # camera yaw correction
        # detection frame (base at time d) -> node frame (base at node stamp)
        p = wo[n]; c, s = np.cos(p[2]), np.sin(p[2])
        gx, gy = w[0] + np.cos(w[2]) * x - np.sin(w[2]) * y, w[1] + np.sin(w[2]) * x + np.cos(w[2]) * y
        lml.append((n, int(d[1]), c * (gx - p[0]) + s * (gy - p[1]), -s * (gx - p[0]) + c * (gy - p[1])))
    tags = sorted({l[1] for l in lml})

    q = G.interp_pose(G.load_bag(os.path.join(C.DB_DIR, "tf_" + name))["gt"], stamps); ok = np.isfinite(q).all(1)
    return frames, nodes, ids, stamps, wo, odl, lml, tags, q, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db"); ap.add_argument("run")
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--exclude", type=int, nargs="*", default=[44])
    ap.add_argument("--odom-scale", type=float, default=16.0)
    ap.add_argument("--cam-yaw", type=float, default=3.0)
    ap.add_argument("--max-dt", type=float, default=0.25, help="max time between a detection and its node (s)")
    ap.add_argument("--out", default=None, help="output folder (default: analysis/<db>_tag_ablation next to the run)")
    a = ap.parse_args()
    name = a.db.replace(".db", ""); ts = os.path.basename(os.path.normpath(a.run))
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(os.path.normpath(a.run))), "analysis", f"{name}_tag_ablation")
    os.makedirs(out, exist_ok=True)
    frames, nodes, ids, stamps, wo, odl, lml, tags, q, ok = graph_inputs(a.db, set(a.exclude), a.cam_yaw, a.max_dt)
    N = len(ids)
    tt = stamps - stamps[0]
    rm = lambda e: float(np.sqrt((e ** 2).mean()))

    # alignment window for drift: from the start until the chair has driven >= 5 m along a path that is
    # not a straight line (minor spread >= 0.25 m), so the rotation between the paths is determined.
    # (A fixed 60 s failed on SB-01-10-22: 1.8 m driven, 0.22 x 0.01 m spread.)
    k_align = np.zeros_like(ok)
    gi = np.where(ok)[0]; cum = 0.0
    for j in range(1, len(gi)):
        stp = np.hypot(*(q[gi[j], :2] - q[gi[j - 1], :2])); cum += stp if stp < 0.5 else 0.0
        if cum >= 5.0:
            Pw = q[gi[:j + 1], :2]
            if np.sqrt(np.linalg.eigvalsh(np.cov((Pw - Pw.mean(0)).T))[0]) >= 0.25:
                k_align[gi[:j + 1]] = True; break
    if not k_align.any():
        k_align = ok.copy()
    align_s = tt[k_align].max()

    def drift(P):
        k = k_align
        R, t = G.fit_se2(P[k, :2], q[k, :2])
        return np.hypot(*((R @ P[:, :2].T).T + t - q[:, :2]).T) * 1000

    db_odo = np.array([[nodes[i][2][0, 3], nodes[i][2][1, 3], np.arctan2(nodes[i][2][1, 0], nodes[i][2][0, 0])] for i in ids])
    lines = [f"{a.db} from the bag: {N} nodes, /wheel_odom, {len(lml)} tag36h11 detections on {len(tags)} tags "
             f"(DB had keyframe detections only); frames {frames}; excluded {a.exclude}; odometry x{a.odom_scale:g}; camera yaw {a.cam_yaw:+g} deg", "",
             f"  {'':34s} {'ATE rmse':>8s} {'p90':>6s} {'max':>6s} mm  (one whole-drive fit to ground truth)"]

    def row(lbl, P):
        e = TA.ate(P, q, ok)[0]; d = drift(P)[ok]
        lines.append(f"  {lbl:34s} {rm(e):8.1f} {np.percentile(e, 90):6.1f} {e.max():6.0f}")
        return rm(e), d.max()

    row("EKF odometry (DB, /odometry/filtered)", db_odo)
    row("odometry only (bag: /wheel_odom, else /tf odom)", wo)
    P_all = TA.solve(N, odl, lml, wo, a.odom_scale, set(tags)); row(f"all {len(tags)} tags", P_all)
    rng = np.random.default_rng(0); halves = []
    for dd in range(a.draws):
        keep = set(rng.choice(tags, size=len(tags) // 2, replace=False).tolist())
        P = TA.solve(N, odl, lml, wo, a.odom_scale, keep)
        e = TA.ate(P, q, ok)[0]; d = drift(P)[ok]
        halves.append((rm(e), d.max(), np.percentile(d, 90), keep, P))
        print(f"  half draw {dd}: ATE {rm(e):.1f}  drift max {d.max():.0f}", flush=True)
    halves.sort(key=lambda h: h[0]); med = halves[len(halves) // 2]
    np.savez(os.path.join(out, "poses.npz"), ids=np.array(ids), stamps=stamps, gt=q, wheel=wo, all=P_all, half=med[4],
             half_keep=np.array(sorted(med[3])))
    lines.append(f"  half the tags ({len(tags)//2}), {a.draws} random draws: ATE rmse best {halves[0][0]:.1f} / median {med[0]:.1f} / worst {halves[-1][0]:.1f} mm")
    # (drift aligned on a start window was dropped: a short window mis-sets the rotation by up to ~2 deg;
    #  the drift-over-distance table below needs no alignment)

    # drift over distance driven (relative pose error, no alignment), GT heading offset from the all-tags fit
    Rf, tf_ = G.fit_se2(P_all[ok, :2], q[ok, :2]); hoff = np.median(wrap(q[ok, 2] - (P_all[ok, 2] + np.arctan2(Rf[1, 0], Rf[0, 0]))))
    gi = np.where(ok)[0]; Gq = q[gi].copy(); Gq[:, 2] -= hoff
    stp = np.hypot(*np.diff(Gq[:, :2], axis=0).T); stp[stp > 0.5] = 0; cum = np.r_[0, np.cumsum(stp)]
    lines += ["", f"  path {cum[-1]:.0f} m.  Position error after driving D metres (relative, no alignment): median / p90 mm, heading p90 deg"]
    lines.append(f"    {'':22s}" + "".join(f"{'D = ' + str(D) + ' m':>24s}" for D in (5, 10, 20)))
    for lbl, P in (("no tags (odometry)", wo), ("half the tags (median)", med[4]), ("all tags", P_all)):
        row = []
        for D in (5, 10, 20):
            j = np.searchsorted(cum, cum + D); m = j < len(gi); i0, j0 = np.where(m)[0], j[m]
            def rel(X):
                c, s = np.cos(X[i0, 2]), np.sin(X[i0, 2]); dx, dy = X[j0, 0] - X[i0, 0], X[j0, 1] - X[i0, 1]
                return np.c_[c * dx + s * dy, -s * dx + c * dy], wrap(X[j0, 2] - X[i0, 2])
            (pe, he), (pg, hg) = rel(P[gi]), rel(Gq)
            e = np.hypot(*(pe - pg).T) * 1000; h = np.degrees(np.abs(wrap(he - hg)))
            row.append(f"{np.median(e):6.0f} /{np.percentile(e, 90):5.0f}  {np.percentile(h, 90):4.1f} deg")
        lines.append(f"    {lbl:22s}" + "".join(f"{r_:>24s}" for r_ in row))

    # maps from the run's scans
    scans, origins, st, align, _ = GS.load_run(a.db, a.run)
    Pf = None
    for stg in ("final", "manhattan", "tag_GTSAM", "rtabmap_graph"):
        f = os.path.join(a.run, f"poses_{stg}.csv")
        if os.path.exists(f):
            Pf = C.load_pose_csv(f); break
    sid = [i for i in ids if i in scans]; kid = {i: k for k, i in enumerate(ids)}
    # runs made without WC_CAMERA_YAW_DEG have unrotated scans: rotate them here to match the tag detections
    run_yaw = re.search(r"WC_CAMERA_YAW_DEG=([-0-9.]+)", open(os.path.join(a.run, "metrics.txt")).read()) if os.path.exists(os.path.join(a.run, "metrics.txt")) else None
    extra_yaw = np.radians(a.cam_yaw - (float(run_yaw.group(1)) if run_yaw else 0.0))
    Rz_extra = np.array([[np.cos(extra_yaw), -np.sin(extra_yaw), 0], [np.sin(extra_yaw), np.cos(extra_yaw), 0], [0, 0, 1]])
    # ground truth -> this run's map frame (gt_to_map from the run's map yaml)
    yml = open(os.path.splitext(C.grid_pgm(a.run))[0] + ".yaml").read()
    mt = re.search(r"gt_to_map:\s*\{x:\s*([-0-9.]+),\s*y:\s*([-0-9.]+),\s*yaw:\s*([-0-9.]+)\}", yml)
    gtx, gty, gth = map(float, mt.groups())
    Rg = np.array([[np.cos(gth), -np.sin(gth)], [np.sin(gth), np.cos(gth)]])
    gt_map = (Rg.T @ (q[:, :2] - [gtx, gty]).T).T          # GT positions in the map frame
    tiles = []
    for lbl, P, keep in (("all tags", P_all, set(tags)), ("half the tags (median draw)", med[4], med[3]), ("no tags (wheel odometry)", wo, set())):
        # place this trajectory with ONE rotation + translation fitted to ground truth over the whole
        # drive (as ATE is measured), then into the map frame: what remains is the graph's real error.
        # (Aligning on a short start window added 1.4-2.1 deg of artificial rotation on SB / 10-12-01.)
        k60 = ok
        R1, t1 = G.fit_se2(P[k60, :2], q[k60, :2])
        R = Rg.T @ R1; t = Rg.T @ (t1 - [gtx, gty])
        dth = np.arctan2(R[1, 0], R[0, 0]); ns, no = {}, {}
        for i in sid:
            Pinv = np.linalg.inv(Pf[i])
            base = (scans[i].astype(np.float64) @ Pinv[:3, :3].T + Pinv[:3, 3]) @ Rz_extra.T
            ob = Pinv @ np.r_[origins[i], Pf[i][2, 3], 1.0]
            x, y = R @ P[kid[i], :2] + t; th = P[kid[i], 2] + dth
            T = np.eye(4); c, s = np.cos(th), np.sin(th); T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = [x, y]
            ns[i] = (base @ T[:3, :3].T + T[:3, 3]).astype(np.float32); no[i] = (T @ ob)[:2]
        img, feat, aux = GS.build_grid(ns, no, st, 0.0, "pct", log=lambda *x: None, want_aux=True)
        g, _ = TW.thin(img, feat, ledge_mask=TW.detect_ledge(img, aux)[0])
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        allxy = np.concatenate([ns[i][:, :2] for i in sid] + [np.array([no[i] for i in sid])]); xmin, ymin = allxy.min(0) - 1.0
        px = lambda xy: np.c_[(xy[:, 0] - xmin) / 0.025, g.shape[0] - 1 - (xy[:, 1] - ymin) / 0.025].astype(np.int32)
        # trajectories: ground truth (blue), this graph (red); grey ticks join the same moment every 50 nodes
        Pm = (R @ P[:, :2].T).T + t
        for kk in range(0, N, 50):
            if ok[kk]: cv2.line(vis, tuple(px(Pm[kk:kk + 1])[0]), tuple(px(gt_map[kk:kk + 1])[0]), (130, 130, 130), 1)
        cv2.polylines(vis, [px(gt_map[ok])], False, (220, 120, 0), 1)
        cv2.polylines(vis, [px(Pm)], False, (40, 40, 220), 1)
        if keep and keep != set(tags):
            for tag in tags:
                pts = [R @ [P[n, 0] + np.cos(P[n, 2]) * mx - np.sin(P[n, 2]) * my, P[n, 1] + np.sin(P[n, 2]) * mx + np.cos(P[n, 2]) * my] + t
                       for (n, tg, mx, my) in lml if tg == tag]
                m = np.mean(pts, 0)
                cv2.circle(vis, (int((m[0] - xmin) / 0.025), g.shape[0] - 1 - int((m[1] - ymin) / 0.025)), 4, (0, 170, 0) if tag in keep else (0, 0, 230), -1)
        e = TA.ate(P, q, ok)[0]; d = drift(P)[ok]
        tiles.append(TA.labelled(vis, f"{lbl}: ATE {rm(e):.0f} mm, p90 {np.percentile(e, 90):.0f}, max {e.max():.0f} mm  (whole-drive fit)   blue = ground truth, red = this graph"))
        cv2.imwrite(os.path.join(out, f"map_{lbl.split()[0]}.pgm"), g)
        with open(os.path.join(out, f"map_{lbl.split()[0]}.yaml"), "w") as yf:   # world origin, same frame as the run's map
            yf.write(f"image: map_{lbl.split()[0]}.pgm\nresolution: 0.025\norigin: [{xmin:.4f}, {ymin:.4f}, 0]\n")
        np.save(os.path.join(out, f"traj_{lbl.split()[0]}.npy"), Pm)
    w = max(t_.shape[1] for t_ in tiles)
    cv2.imwrite(os.path.join(out, "maps.png"), np.vstack([cv2.copyMakeBorder(t_, 0, 8, 0, w - t_.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t_ in tiles]))
    # trajectories (one whole-drive fit to ground truth, as for ATE)
    H = 900; lo, hi = q[ok, :2].min(0) - 0.6, q[ok, :2].max(0) + 0.6
    sc = (H - 40) / (hi[1] - lo[1]); Wd = int((hi[0] - lo[0]) * sc) + 40
    to_px = lambda xy: np.c_[(xy[:, 0] - lo[0]) * sc + 20, H - 20 - (xy[:, 1] - lo[1]) * sc].astype(np.int32)
    pan = []
    for lbl, P in (("all tags", P_all), ("half tags", med[4]), ("wheel odometry", wo)):
        img = np.full((H, Wd, 3), 255, np.uint8)
        k = ok; R, t = G.fit_se2(P[k, :2], q[k, :2])
        cv2.polylines(img, [to_px(q[ok, :2])], False, (220, 120, 0), 2)
        cv2.polylines(img, [to_px((R @ P[:, :2].T).T + t)], False, (40, 40, 220), 2)
        cv2.putText(img, f"{lbl} (whole-drive fit)", (10, 24), 0, 0.6, (0, 0, 0), 2)
        pan.append(img)
    cv2.imwrite(os.path.join(out, "trajectories.png"), np.hstack(pan))
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"wrote {out}/maps.png, trajectories.png, summary.txt")


if __name__ == "__main__":
    main()
