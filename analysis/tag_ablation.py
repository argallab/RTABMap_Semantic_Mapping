"""What would the map look like with fewer AprilTags?

Re-solves a recording's 2D pose graph (odometry links + position-only tag
landmarks, the exporter's weighting) with
  * all tags, * a random half of the tag IDs (several draws), * no tags (odometry only),
scores each against ground truth, and rebuilds the occupancy grid for each from
the exporter run's own cleaned scans (grid_sim + wall thinning), so only the
poses differ.

Settings mirror a dev run: --exclude 44 --odom-scale 16 --cam-yaw 3.

  python3 tag_ablation.py R01_02.db ../output/<run> [--draws 10] [--exclude 44] [--odom-scale 16] [--cam-yaw 3]

Writes analysis/results/tag_ablation/<db>_<run>/: maps.png (all | half | none, with
kept / removed tags marked on the half map), trajectories.png, summary.txt.
"""
import argparse
import os

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import common as C
import gt as G
import grid_sim as GS
import thin_walls as TW

wrap = lambda a: np.arctan2(np.sin(a), np.cos(a))


def load_graph(db, exclude, cam_yaw_deg, yaw_scale=1.0):
    con = C.connect(db)
    nodes = C.load_nodes(con)
    ids = sorted(C.load_rtabmap_opt_poses(con))
    idx = {i: k for k, i in enumerate(ids)}
    xyz = lambda T: (T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0]))
    odl, lml = [], []
    c, s = np.cos(np.radians(cam_yaw_deg)), np.sin(np.radians(cam_yaw_deg))
    for frm, to, typ, info, tb in con.execute("select from_id, to_id, type, information_matrix, transform from Link where type in (0, 8)"):
        T = np.eye(4); T[:3, :] = np.frombuffer(tb, dtype=np.float32).reshape(3, 4)
        I = np.frombuffer(info, dtype=np.float64).reshape(6, 6)
        if typ == 0 and frm in idx and to in idx and frm < to:
            x, y, th = xyz(T)
            odl.append((idx[frm], idx[to], (x, y, th * yaw_scale), np.sqrt(I[0, 0]), np.sqrt(I[1, 1]), np.sqrt(I[5, 5])))
        elif typ == 8 and frm in idx and -to not in exclude:
            mx, my = T[0, 3], T[1, 3]
            lml.append((idx[frm], -to, c * mx - s * my, s * mx + c * my))
    odo = np.array([xyz(nodes[i][2]) for i in ids])
    return ids, nodes, odl, lml, odo


def solve(N, odl, lml, odo, odo_scale, keep_tags, var=0.2):
    L = [l for l in lml if l[1] in keep_tags]
    if not L:
        # odometry only: the chain of odometry links from the first pose
        return odo.copy()
    tags = sorted({l[1] for l in L}); tidx = {t: k for k, t in enumerate(tags)}
    x0 = np.r_[odo.ravel(), np.zeros(2 * len(tags))]
    for (n, t, mx, my) in L:
        k = 3 * N + 2 * tidx[t]
        if x0[k] == 0 and x0[k + 1] == 0:
            px, py, pt = odo[n]
            x0[k], x0[k + 1] = px + np.cos(pt) * mx - np.sin(pt) * my, py + np.sin(pt) * mx + np.cos(pt) * my
    a = np.array([l[0] for l in odl]); b = np.array([l[1] for l in odl])
    meas = np.array([l[2] for l in odl]); wo = np.array([l[3:] for l in odl]) * np.sqrt(odo_scale)
    ln = np.array([l[0] for l in L]); lt = np.array([3 * N + 2 * tidx[l[1]] for l in L]); lm = np.array([[l[2], l[3]] for l in L])
    sw = 1.0 / np.sqrt(var)

    def res(x):
        P = x[:3 * N].reshape(N, 3)
        pa, pb = P[a], P[b]
        c, s = np.cos(pa[:, 2]), np.sin(pa[:, 2])
        dx, dy = pb[:, 0] - pa[:, 0], pb[:, 1] - pa[:, 1]
        r_od = np.stack([(c * dx + s * dy - meas[:, 0]) * wo[:, 0], (-s * dx + c * dy - meas[:, 1]) * wo[:, 1],
                         wrap(pb[:, 2] - pa[:, 2] - meas[:, 2]) * wo[:, 2]], 1).ravel()
        pn = P[ln]; c, s = np.cos(pn[:, 2]), np.sin(pn[:, 2])
        dx, dy = x[lt] - pn[:, 0], x[lt + 1] - pn[:, 1]
        r_lm = np.stack([(c * dx + s * dy - lm[:, 0]) * sw, (-s * dx + c * dy - lm[:, 1]) * sw], 1).ravel()
        return np.r_[r_od, r_lm, (P[0] - odo[0]) * 1e4]

    from scipy.sparse import coo_matrix
    nres, nvar = 3 * len(odl) + 2 * len(L) + 3, len(x0)
    ko = np.arange(len(odl)); kl = np.arange(len(L))

    def jac(x):
        # exact sparse Jacobian (numerical differencing was ~3 min per solve with all bag detections)
        P = x[:3 * N].reshape(N, 3)
        pa, pb = P[a], P[b]
        c, s = np.cos(pa[:, 2]), np.sin(pa[:, 2]); dx, dy = pb[:, 0] - pa[:, 0], pb[:, 1] - pa[:, 1]
        w0, w1, w2 = wo[:, 0], wo[:, 1], wo[:, 2]
        r0, r1, r2 = 3 * ko, 3 * ko + 1, 3 * ko + 2
        ia, ib = 3 * a, 3 * b
        rows = [r0, r0, r0, r0, r0, r1, r1, r1, r1, r1, r2, r2]
        cols = [ia, ia + 1, ia + 2, ib, ib + 1, ia, ia + 1, ia + 2, ib, ib + 1, ia + 2, ib + 2]
        vals = [-c * w0, -s * w0, (-s * dx + c * dy) * w0, c * w0, s * w0,
                s * w1, -c * w1, (-c * dx - s * dy) * w1, -s * w1, c * w1, -w2, w2]
        pn = P[ln]; c, s = np.cos(pn[:, 2]), np.sin(pn[:, 2]); dx, dy = x[lt] - pn[:, 0], x[lt + 1] - pn[:, 1]
        o = 3 * len(odl); q0, q1 = o + 2 * kl, o + 2 * kl + 1; inn = 3 * ln
        rows += [q0, q0, q0, q0, q0, q1, q1, q1, q1, q1]
        cols += [inn, inn + 1, inn + 2, lt, lt + 1, inn, inn + 1, inn + 2, lt, lt + 1]
        vals += [-c * sw, -s * sw, (-s * dx + c * dy) * sw, c * sw, s * sw,
                 s * sw, -c * sw, (-c * dx - s * dy) * sw, -s * sw, c * sw]
        pr = o + 2 * len(L)
        rows += [np.array([pr, pr + 1, pr + 2])]; cols += [np.array([0, 1, 2])]; vals += [np.full(3, 1e4)]
        rr = np.concatenate([np.broadcast_to(r_, v_.shape) if np.ndim(r_) else np.full(np.shape(v_), r_) for r_, v_ in zip(rows, vals)])
        cc = np.concatenate([np.broadcast_to(c_, v_.shape) for c_, v_ in zip(cols, vals)])
        vv = np.concatenate([np.broadcast_to(v_, np.shape(v_)) if np.ndim(v_) else np.full(1, v_) for v_ in vals])
        return coo_matrix((vv, (rr, cc)), shape=(nres, nvar)).tocsr()

    sol = least_squares(res, x0, jac=jac, method="trf", x_scale="jac", max_nfev=60)
    return sol.x[:3 * N].reshape(N, 3)


def ate(P, q, ok):
    R, t = G.fit_se2(P[ok, :2], q[ok, :2])
    e = np.hypot(*((R @ P[ok, :2].T).T + t - q[ok, :2]).T) * 1000
    return e, R, t


def labelled(vis, title):
    v = cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    H, W = v.shape[:2]; M = 30
    o = cv2.copyMakeBorder(v, M, 26, M, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for k, x in enumerate(range(0, W, 100)):
        cv2.line(o, (x + M, M), (x + M, H + M), (200, 160, 60), 1); cv2.putText(o, chr(65 + k), (x + M + 44, 21), 0, 0.6, (200, 80, 0), 2)
    for k, y in enumerate(range(0, H, 100)):
        cv2.line(o, (M, y + M), (W + M, y + M), (200, 160, 60), 1); cv2.putText(o, str(k + 1), (8, y + M + 56), 0, 0.6, (200, 80, 0), 2)
    cv2.putText(o, title, (M, H + M + 19), 0, 0.55, (0, 0, 200), 2)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db"); ap.add_argument("run")
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--exclude", type=int, nargs="*", default=[44])
    ap.add_argument("--odom-scale", type=float, default=16.0)
    ap.add_argument("--cam-yaw", type=float, default=3.0)
    ap.add_argument("--yaw-scale", type=float, default=1.0, help="odometry yaw scale (0.9606 for recordings before the wheel-base fix)")
    a = ap.parse_args()
    ts = os.path.basename(os.path.normpath(a.run))
    out = os.path.join(C.ROOT, "analysis", "results", "tag_ablation", f"{a.db.replace('.db', '')}_{ts}")
    os.makedirs(out, exist_ok=True)

    ids, nodes, odl, lml, odo = load_graph(a.db, set(a.exclude), a.cam_yaw, a.yaw_scale)
    N = len(ids)
    stamps = np.array([nodes[i][0] for i in ids])
    q = G.interp_pose(G.load_bag(os.path.join(C.DB_DIR, "tf_" + a.db.replace(".db", "")))["gt"], stamps)
    ok = np.isfinite(q).all(1)
    tags = sorted({l[1] for l in lml})
    lines = [f"{a.db}: {N} nodes, {len(tags)} tags used ({len(lml)} detections); excluded {a.exclude}, odometry x{a.odom_scale:g}, camera yaw {a.cam_yaw:+g} deg"]

    variants = {}
    P_all = solve(N, odl, lml, odo, a.odom_scale, set(tags)); e_all = ate(P_all, q, ok)[0]
    variants["all"] = (P_all, set(tags))
    rng = np.random.default_rng(0)
    halves = []
    for d in range(a.draws):
        keep = set(rng.choice(tags, size=len(tags) // 2, replace=False).tolist())
        P = solve(N, odl, lml, odo, a.odom_scale, keep)
        e = ate(P, q, ok)[0]
        halves.append((np.sqrt((e ** 2).mean()), np.percentile(e, 90), keep, P))
        print(f"  half draw {d}: rmse {halves[-1][0]:.1f} mm", flush=True)
    halves.sort(key=lambda h: h[0])
    med = halves[len(halves) // 2]
    variants["half"] = (med[3], med[2])
    P_none = odo.copy(); e_none = ate(P_none, q, ok)[0]
    variants["none"] = (P_none, set())
    rm = lambda e: np.sqrt((e ** 2).mean())
    lines += [f"  all {len(tags)} tags      : ATE rmse {rm(e_all):5.1f}  p90 {np.percentile(e_all, 90):5.1f}  max {e_all.max():6.1f} mm",
              f"  half the tags ({len(tags)//2}), {a.draws} random draws: rmse best {halves[0][0]:.1f} / median {med[0]:.1f} / worst {halves[-1][0]:.1f} mm;"
              f" p90 median {np.median([h[1] for h in halves]):.1f} mm",
              f"  no tags (odometry)  : ATE rmse {rm(e_none):5.1f}  p90 {np.percentile(e_none, 90):5.1f}  max {e_none.max():6.1f} mm"]

    # maps from the run's own scans, placed with each variant (aligned to the run's frame)
    scans, origins, st, align, _ = GS.load_run(a.db, a.run)
    Pf = None
    for stg in ("final", "manhattan", "tag_GTSAM", "rtabmap_graph"):
        f = os.path.join(a.run, f"poses_{stg}.csv")
        if os.path.exists(f):
            Pf = C.load_pose_csv(f); break
    sid = [i for i in ids if i in scans]
    tiles, trajs = [], {}
    for name, (P, keep) in variants.items():
        Pd = {i: P[k] for k, i in enumerate(ids)}
        R, t = G.fit_se2(np.array([Pd[i][:2] for i in sid]), np.array([Pf[i][:2, 3] for i in sid]))
        dth = np.arctan2(R[1, 0], R[0, 0])
        new_s, new_o = {}, {}
        for i in sid:
            Pinv = np.linalg.inv(Pf[i])
            base = scans[i].astype(np.float64) @ Pinv[:3, :3].T + Pinv[:3, 3]
            ob = Pinv @ np.r_[origins[i], Pf[i][2, 3], 1.0]
            x, y = R @ Pd[i][:2] + t; th = Pd[i][2] + dth
            T = np.eye(4); c, s = np.cos(th), np.sin(th); T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = [x, y]
            new_s[i] = (base @ T[:3, :3].T + T[:3, 3]).astype(np.float32); new_o[i] = (T @ ob)[:2]
        img, feat, aux = GS.build_grid(new_s, new_o, st, 0.0, "pct", log=lambda *x: None, want_aux=True)
        g, _ = TW.thin(img, feat, ledge_mask=TW.detect_ledge(img, aux)[0])
        cv2.imwrite(os.path.join(out, f"map_{name}.pgm"), g)
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        trajs[name] = (R, t, Pd)
        if name == "half":
            # mark tags: kept = green, removed = red (tag position = mean of its detections placed with this variant)
            allxy = np.concatenate([new_s[i][:, :2] for i in sid] + [np.array([new_o[i] for i in sid])])
            xmin, ymin = allxy.min(0) - 1.0
            for tag in tags:
                pts = []
                for (n, tt, mx, my) in lml:
                    if tt != tag: continue
                    px, py, pt = P[n]; wx, wy = px + np.cos(pt) * mx - np.sin(pt) * my, py + np.sin(pt) * mx + np.cos(pt) * my
                    pts.append(R @ [wx, wy] + t)
                m = np.mean(pts, 0)
                cc, rr = int((m[0] - xmin) / 0.025), g.shape[0] - 1 - int((m[1] - ymin) / 0.025)
                cv2.circle(vis, (cc, rr), 4, (0, 170, 0) if tag in keep else (0, 0, 230), -1)
        e = ate(P, q, ok)[0]
        title = {"all": f"all {len(tags)} tags", "half": f"half the tags ({len(keep)}; green kept, red removed)", "none": "no tags (odometry only)"}[name]
        tiles.append(labelled(vis, f"{title}: ATE rmse {rm(e):.0f} mm, p90 {np.percentile(e, 90):.0f} mm"))
    w = max(t_.shape[1] for t_ in tiles)
    padw = lambda x: cv2.copyMakeBorder(x, 0, 8, 0, w - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.imwrite(os.path.join(out, "maps.png"), np.vstack([padw(t_) for t_ in tiles]))

    # trajectories vs ground truth, each in GT frame
    H = 900; allg = q[ok, :2]; lo, hi = allg.min(0) - 0.5, allg.max(0) + 0.5
    sc = (H - 40) / (hi[1] - lo[1]); Wd = int((hi[0] - lo[0]) * sc) + 40
    pan = []
    for name, (P, keep) in variants.items():
        img = np.full((H, Wd, 3), 255, np.uint8)
        e, R, t = ate(P, q, ok)
        to_px = lambda xy: np.c_[(xy[:, 0] - lo[0]) * sc + 20, H - 20 - (xy[:, 1] - lo[1]) * sc].astype(np.int32)
        cv2.polylines(img, [to_px(q[ok, :2])], False, (220, 120, 0), 2)
        cv2.polylines(img, [to_px((R @ P[:, :2].T).T + t)], False, (40, 40, 220), 2)
        cv2.putText(img, f"{name}: rmse {rm(e):.0f} mm  (blue = ground truth, red = graph)", (10, 24), 0, 0.6, (0, 0, 0), 2)
        pan.append(img)
    cv2.imwrite(os.path.join(out, "trajectories.png"), np.hstack(pan))
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"wrote {out}/maps.png, trajectories.png, summary.txt")


if __name__ == "__main__":
    main()
