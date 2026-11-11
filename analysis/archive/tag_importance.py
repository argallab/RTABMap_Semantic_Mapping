"""Which AprilTags matter most?  (pose graph rebuilt from the bag, scored against ground truth)

Two measures per tag:
  * leave-one-out: all tags vs all tags minus this one -> how much worse without it
    (tags in well-covered areas score ~0: their neighbours cover for them)
  * sparse importance: many solves with a random half of the tags; mean error when the
    tag is left out minus mean error when it is kept -> its value when coverage is thin
Errors: ATE rmse (whole-drive fit) and drift = p90 position error after driving 20 m
(relative, no alignment).  Also per tag: detections, median range, time span seen,
scatter of its detections placed with ground-truth poses (detection quality).

  python3 tag_importance.py R01_02.db ../output/<run> [--draws 40] [--exclude 44]

Writes analysis/results/tag_importance/<db>/: tags.csv, importance_map.png, summary.txt
"""
import argparse
import os

import cv2
import numpy as np

import bag_graph as B
import common as C
import gt as G
import tag_ablation as TA

wrap = lambda a: np.arctan2(np.sin(a), np.cos(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db"); ap.add_argument("run")
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--exclude", type=int, nargs="*", default=[44])
    ap.add_argument("--odom-scale", type=float, default=16.0)
    ap.add_argument("--cam-yaw", type=float, default=3.0)
    a = ap.parse_args()
    name = a.db.replace(".db", "")
    out = os.path.join(C.ROOT, "analysis", "results", "tag_importance", name)
    os.makedirs(out, exist_ok=True)
    frames, nodes, ids, stamps, wo, odl, lml, tags, q, ok = B.graph_inputs(a.db, set(a.exclude), a.cam_yaw)
    N = len(ids); idx = np.where(ok)[0]
    G_ = q[idx]
    stp = np.hypot(*np.diff(G_[:, :2], axis=0).T); stp[stp > 0.5] = 0; cum = np.r_[0, np.cumsum(stp)]
    j20 = np.searchsorted(cum, cum + 20.0); m20 = j20 < len(idx); i0, j0 = np.where(m20)[0], j20[m20]

    P_all = TA.solve(N, odl, lml, wo, a.odom_scale, set(tags))
    R, t = G.fit_se2(P_all[ok, :2], q[ok, :2]); rot = np.arctan2(R[1, 0], R[0, 0])
    hoff = np.median(wrap(q[ok, 2] - (P_all[ok, 2] + rot)))     # GT yaw - graph heading
    Gh = G_.copy(); Gh[:, 2] -= hoff

    def scores(P):
        e = TA.ate(P, q, ok)[0]
        X = P[idx]
        def rel(Y):
            c, s = np.cos(Y[i0, 2]), np.sin(Y[i0, 2]); dx, dy = Y[j0, 0] - Y[i0, 0], Y[j0, 1] - Y[i0, 1]
            return np.c_[c * dx + s * dy, -s * dx + c * dy]
        d = np.hypot(*(rel(X) - rel(Gh)).T) * 1000
        return float(np.sqrt((e ** 2).mean())), float(np.percentile(d, 90))

    base = scores(P_all); none = scores(wo)
    print(f"{a.db}: {len(tags)} tags, {len(lml)} detections; all tags ATE {base[0]:.1f} mm drift20 p90 {base[1]:.0f} mm; no tags {none[0]:.1f} / {none[1]:.0f}", flush=True)

    # per-tag stats
    stats = {}
    for tg in tags:
        L = [l for l in lml if l[1] == tg]
        w = []
        for (n, _, mx, my) in L:
            if not ok[n]: continue
            x, y, th = q[n, 0], q[n, 1], q[n, 2] - hoff
            w.append((x + np.cos(th) * mx - np.sin(th) * my, y + np.sin(th) * mx + np.cos(th) * my))
        w = np.array(w)
        scat = float(np.sqrt(((w - w.mean(0)) ** 2).sum(1).mean() / 2) * 1000) if len(w) > 2 else float("nan")
        ts = np.array([stamps[n] for (n, _, _, _) in L]) - stamps[0]
        visits = 1 + int(np.sum(np.diff(np.sort(ts)) > 10))
        stats[tg] = dict(n=len(L), rng=float(np.median([np.hypot(mx, my) for (_, _, mx, my) in L])), scatter=scat,
                         span=float(ts.max() - ts.min()), visits=visits)

    # leave-one-out
    loo = {}
    for k, tg in enumerate(tags):
        loo[tg] = scores(TA.solve(N, odl, lml, wo, a.odom_scale, set(tags) - {tg}))
        print(f"  leave out {tg:3d} ({k+1}/{len(tags)}): ATE {loo[tg][0]:.1f}  drift20 {loo[tg][1]:.0f}", flush=True)

    # sparse importance from random halves
    rng = np.random.default_rng(1)
    draws = []
    for d in range(a.draws):
        keep = set(rng.choice(tags, size=len(tags) // 2, replace=False).tolist())
        draws.append((keep, scores(TA.solve(N, odl, lml, wo, a.odom_scale, keep))))
        print(f"  half draw {d+1}/{a.draws}: ATE {draws[-1][1][0]:.1f}  drift20 {draws[-1][1][1]:.0f}", flush=True)
    sparse = {}
    for tg in tags:
        inn = [s for kp, s in draws if tg in kp]; outn = [s for kp, s in draws if tg not in kp]
        sparse[tg] = (np.mean([s[0] for s in outn]) - np.mean([s[0] for s in inn]),
                      np.mean([s[1] for s in outn]) - np.mean([s[1] for s in inn]))

    # table
    rows = sorted(tags, key=lambda tg: -(sparse[tg][0] / max(1e-6, np.std([s[0] for _, s in draws])) + sparse[tg][1] / max(1e-6, np.std([s[1] for _, s in draws]))))
    hdr = "tag,detections,median_range_m,scatter_mm,seen_over_s,visits,loo_dATE_mm,loo_ddrift20_mm,sparse_dATE_mm,sparse_ddrift20_mm"
    with open(os.path.join(out, "tags.csv"), "w") as f:
        f.write(hdr + "\n")
        for tg in rows:
            s_ = stats[tg]
            f.write(f"{tg},{s_['n']},{s_['rng']:.2f},{s_['scatter']:.0f},{s_['span']:.0f},{s_['visits']},"
                    f"{loo[tg][0]-base[0]:.2f},{loo[tg][1]-base[1]:.1f},{sparse[tg][0]:.2f},{sparse[tg][1]:.1f}\n")
    lines = [f"{a.db}: {len(tags)} tags ({len(lml)} detections from the bag), excluded {a.exclude}; odometry x{a.odom_scale:g}; camera yaw {a.cam_yaw:+g} deg",
             f"all tags: ATE {base[0]:.1f} mm, drift (p90 error after 20 m) {base[1]:.0f} mm;  no tags: {none[0]:.1f} / {none[1]:.0f} mm;  "
             f"{a.draws} random halves: ATE {np.mean([s[0] for _, s in draws]):.1f} (sd {np.std([s[0] for _, s in draws]):.1f}), drift {np.mean([s[1] for _, s in draws]):.0f} (sd {np.std([s[1] for _, s in draws]):.0f})",
             "", "ranked by importance when tags are sparse (positive = error goes UP without the tag):",
             f"  {'tag':>4s} {'det':>5s} {'range':>6s} {'scatter':>8s} {'seen':>6s} {'visits':>6s} | {'leave-one-out dATE / ddrift':>28s} | {'sparse dATE / ddrift':>22s}"]
    for tg in rows:
        s_ = stats[tg]
        lines.append(f"  {tg:4d} {s_['n']:5d} {s_['rng']:5.2f}m {s_['scatter']:6.0f}mm {s_['span']:5.0f}s {s_['visits']:6d} | "
                     f"{loo[tg][0]-base[0]:+12.1f} mm {loo[tg][1]-base[1]:+8.0f} mm | {sparse[tg][0]:+8.1f} mm {sparse[tg][1]:+8.0f} mm")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # map: tag positions on the run's grid, colour = sparse importance (ATE), size = detections
    import re
    pgm = C.grid_pgm(a.run); grid = cv2.imread(pgm, 0)
    o = [float(v) for v in re.search(r"origin:\s*\[([^\]]+)\]", open(pgm[:-4] + ".yaml").read()).group(1).split(",")[:2]]
    Pf = None
    for stg in ("final", "manhattan", "tag_GTSAM", "rtabmap_graph"):
        f = os.path.join(a.run, f"poses_{stg}.csv")
        if os.path.exists(f):
            Pf = C.load_pose_csv(f); break
    sid = [k for k, i in enumerate(ids) if i in Pf]
    Rm, tm = G.fit_se2(P_all[sid, :2], np.array([Pf[ids[k]][:2, 3] for k in sid]))
    vis = cv2.cvtColor(np.where(grid == 0, 140, np.where(grid == 254, 255, 225)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    imp = np.array([sparse[tg][0] for tg in tags]); vmax = max(1e-6, np.abs(imp).max())
    for tg in tags:
        pts = [Rm @ [P_all[n, 0] + np.cos(P_all[n, 2]) * mx - np.sin(P_all[n, 2]) * my,
                     P_all[n, 1] + np.sin(P_all[n, 2]) * mx + np.cos(P_all[n, 2]) * my] + tm for (n, t_, mx, my) in lml if t_ == tg]
        m = np.mean(pts, 0)
        cc, rr = int((m[0] - o[0]) / 0.025), grid.shape[0] - 1 - int((m[1] - o[1]) / 0.025)
        v = sparse[tg][0] / vmax                                   # -1..1
        col = (int(255 * (1 - max(v, 0))), int(255 * (1 - abs(v))), int(255 * (1 - max(-v, 0))))   # red = important, blue = harmful
        col = (col[0], col[1], 255) if v > 0 else col
        col = tuple(int(c_) for c_ in ((40, 40, 230) if v > 0.5 else (60, 160, 240) if v > 0.2 else (180, 180, 180) if v > -0.2 else (230, 140, 0)))
        rad = int(3 + 6 * np.sqrt(stats[tg]["n"] / max(s_["n"] for s_ in stats.values())))
        cv2.circle(vis, (cc, rr), rad, col, -1); cv2.circle(vis, (cc, rr), rad, (0, 0, 0), 1)
        cv2.putText(vis, str(tg), (cc + rad + 1, rr + 4), 0, 0.33, (0, 0, 0), 1)
    img = TA.labelled(vis, "dark red = most important, orange = useful, grey = little effect, blue = removing it helps;  size = detections")
    cv2.imwrite(os.path.join(out, "importance_map.png"), img)
    print(f"wrote {out}/tags.csv, summary.txt, importance_map.png")


if __name__ == "__main__":
    main()
