"""Analyses 2, 4, 5 from tools/icp_probe output.

  validate : probe counts vs the exporter's metrics.txt for the same round
  a2       : ICP repeatability — spread of converged answers from offset restarts,
             split into H's strong axis, weak axis and yaw; per run (baseline /
             range cap / latency shift / both)
  a4       : where the "limit out of bounds: rot" rejections are, whether they
             cluster, what rotation they want (wide run), and the turn rate there
  a5       : per accepted match: fitness (overlap), inlier RMSE, Hessian
             eigenvalues, degenerate-direction flags

Usage: python3 a245_probe.py 09-29-2.db label=probe/<prefix> [label=...] [--map poses.csv]
"""
import argparse
import json
import os
import time

import numpy as np

import common as C
from a1b_motion import twists


def load(prefix):
    import csv
    with open(prefix + "_queries.csv") as f:
        Q = list(csv.DictReader(f))
    with open(prefix + "_restarts.csv") as f:
        Rr = list(csv.DictReader(f))
    return Q, Rr


def fnum(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else np.nan


def pct(v, qs=(50, 90)):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return [float(np.percentile(v, q)) if len(v) else float("nan") for q in qs]


def summarize_run(label, Q, Rr, tw):
    out = {"label": label}
    st = {}
    for r in Q:
        k = r["status"] if r["status"] != "icp_rejected" else "icp_rejected: " + r["reason"]
        st[k] = st.get(k, 0) + 1
    out["status"] = st
    acc = [r for r in Q if r["status"] == "accepted"]
    out["n_queries"], out["n_accepted"] = len(Q), len(acc)

    # ── a2: repeatability ────────────────────────────────────────────────
    base = {int(r["qid"]): r for r in acc}
    d_strong, d_weak, d_yaw, n_conv, n_all = [], [], [], 0, 0
    per_q = {}
    for r in Rr:
        q = int(r["qid"])
        if q not in base:
            continue
        n_all += 1
        if r["ok"] != "1":
            continue
        n_conv += 1
        b = base[q]
        es = np.array([fnum(b, "es_x"), fnum(b, "es_y")])
        ew = np.array([-es[1], es[0]])
        dt = np.array([fnum(r, "tx") - fnum(b, "tx"), fnum(r, "ty") - fnum(b, "ty")])
        dy = np.degrees(np.arctan2(np.sin(fnum(r, "tyaw") - fnum(b, "tyaw")), np.cos(fnum(r, "tyaw") - fnum(b, "tyaw"))))
        d_strong.append(1000 * abs(dt @ es)); d_weak.append(1000 * abs(dt @ ew)); d_yaw.append(abs(dy))
        pq = per_q.setdefault(q, [])
        pq.append(1000 * abs(dt @ es))
    worst_strong = [max(v) for v in per_q.values()]
    out["a2"] = dict(restarts=n_all, converged=n_conv,
                     strong_mm_p50_p90=pct(d_strong), weak_mm_p50_p90=pct(d_weak), yaw_deg_p50_p90=pct(d_yaw),
                     per_query_worst_strong_mm_p50_p90=pct(worst_strong),
                     frac_queries_worst_strong_gt_10mm=float(np.mean(np.array(worst_strong) > 10)) if worst_strong else float("nan"),
                     corr_strong_mm_mean=float(np.nanmean([fnum(r, "corr_strong_mm") for r in acc])) if acc else float("nan"),
                     corr_weak_mm_mean=float(np.nanmean([fnum(r, "corr_weak_mm") for r in acc])) if acc else float("nan"))

    # ── a5: per-registration diagnostics ─────────────────────────────────
    ov = [fnum(r, "overlap") for r in acc]
    rm = [fnum(r, "inlier_rmse_mm") for r in acc]
    ratio = [np.sqrt(max(fnum(r, "l_weak"), 0) / fnum(r, "l_strong")) for r in acc if fnum(r, "l_strong") > 0]
    yaw_info = [fnum(r, "h3_min") / fnum(r, "h3_max") for r in acc if fnum(r, "h3_max") > 0]
    degen = np.array(ratio) < 0.2
    cw = np.array([fnum(r, "corr_weak_mm") for r in acc if fnum(r, "l_strong") > 0])
    out["a5"] = dict(overlap_p10_p50=pct(ov, (10, 50)), inlier_rmse_mm_p50_p90=pct(rm),
                     constraint_ratio_p10_p50_p90=pct(ratio, (10, 50, 90)),
                     h3_min_over_max_p10_p50=pct(yaw_info, (10, 50)),
                     frac_degenerate_ratio_lt_0p2=float(degen.mean()) if len(degen) else float("nan"),
                     corr_weak_mm_mean_degenerate=float(cw[degen].mean()) if degen.any() else float("nan"),
                     corr_weak_mm_mean_wellposed=float(cw[~degen].mean()) if (~degen).any() else float("nan"))

    # ── a4: rotation-limit rejections ────────────────────────────────────
    rot = [r for r in Q if r["status"] == "icp_rejected" and "rot" in r["reason"]]
    allq = [r for r in Q if r["status"] not in ("no_submap",)]
    XY = np.array([[fnum(r, "x"), fnum(r, "y")] for r in allq])
    RX = np.array([[fnum(r, "x"), fnum(r, "y")] for r in rot]) if rot else np.zeros((0, 2))
    a4 = dict(n=len(rot))
    if len(rot) >= 3:
        from scipy.spatial import cKDTree
        def nn_med(P):
            d, _ = cKDTree(P).query(P, k=2)
            return float(np.median(d[:, 1]))
        # Null: same number of queries drawn from all queries (fixed seed)
        rng = np.random.default_rng(0)
        null = [nn_med(XY[rng.choice(len(XY), len(rot), replace=False)]) for _ in range(200)]
        # clusters: connected components at 1.0 m
        t = cKDTree(RX)
        pairs = t.query_pairs(1.0)
        parent = list(range(len(RX)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for i, j in pairs:
            parent[find(i)] = find(j)
        comp = {}
        for i in range(len(RX)):
            comp.setdefault(find(i), []).append(i)
        sizes = sorted((len(v) for v in comp.values()), reverse=True)
        wide_yaw = [abs(np.degrees(fnum(r, "wide_tyaw"))) for r in rot if r["wide_status"] == "ok"]
        w_rot = [abs(tw.get(int(r["qid"]), (0, 0, 0, 0))[2]) for r in rot]
        w_all = [abs(tw.get(int(r["qid"]), (0, 0, 0, 0))[2]) for r in allq]
        acc_wide = [abs(np.degrees(fnum(r, "tyaw"))) for r in acc]
        a4.update(nn_median_m=nn_med(RX), nn_median_null_p5_p50=[float(np.percentile(null, 5)), float(np.percentile(null, 50))],
                  cluster_sizes_1m=sizes[:10], n_clusters=len(sizes),
                  wide_converged=len(wide_yaw), wide_yaw_deg_p10_p50_p90=pct(wide_yaw, (10, 50, 90)),
                  accepted_yaw_deg_p50_p90=pct(acc_wide),
                  turn_rate_rot_p50_p90=pct(w_rot), turn_rate_allq_p50_p90=pct(w_all))
    out["a4"] = a4
    return out, rot, acc


def plot_map(db, map_poses_path, runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    poses = C.load_pose_csv(map_poses_path)
    scans = C.clean_scans(db, poses.keys())
    pts = []
    for i in sorted(scans)[::3]:
        p = scans[i][::40]
        p = p[(p[:, 2] > 0.1) & (p[:, 2] < 1.0)]
        T = poses[i]
        pts.append(p @ T[:3, :3].T + T[:3, 3])
    pts = np.concatenate(pts)
    n = len(runs)
    fig, axs = plt.subplots(1, n, figsize=(7 * n, 7), squeeze=False)
    for ax, (label, Q) in zip(axs[0], runs):
        ax.scatter(pts[:, 0], pts[:, 1], s=0.05, c="0.75", rasterized=True)
        groups = {"accepted": ("tab:green", 6), "rot": ("tab:red", 22), "low complexity": ("tab:orange", 14),
                  "other reject": ("tab:purple", 14)}
        for g, (col, s) in groups.items():
            sel = []
            for r in Q:
                if r["status"] == "accepted" and g == "accepted": sel.append(r)
                elif r["status"] == "icp_rejected":
                    if "rot" in r["reason"] and g == "rot": sel.append(r)
                    elif "complexity" in r["reason"] and g == "low complexity": sel.append(r)
                    elif "rot" not in r["reason"] and "complexity" not in r["reason"] and g == "other reject": sel.append(r)
            if sel:
                ax.scatter([float(r["x"]) for r in sel], [float(r["y"]) for r in sel], s=s, c=col,
                           label=f"{g} ({len(sel)})", edgecolors="none")
        ax.set_aspect("equal"); ax.set_title(label); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"map -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("runs", nargs="+", help="label=prefix")
    ap.add_argument("--map", help="pose CSV to draw the wall map with")
    a = ap.parse_args()
    t0 = time.time()
    tw = twists(C.connect(a.db))
    res, plot_runs = {}, []
    for spec in a.runs:
        label, prefix = spec.split("=", 1)
        Q, Rr = load(prefix)
        s, rot, acc = summarize_run(label, Q, Rr, tw)
        res[label] = s
        plot_runs.append((label, Q))
        print(f"\n=== {label}  ({prefix}) ===")
        print(json.dumps(s, indent=1))
    os.makedirs(f"{C.ROOT}/analysis/results", exist_ok=True)
    tag = "_".join(r.split("=")[0] for r in a.runs)
    with open(f"{C.ROOT}/analysis/results/a245_{a.db.replace('.db','')}_{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    if a.map:
        plot_map(a.db, a.map, plot_runs, f"{C.ROOT}/analysis/results/a4_map_{a.db.replace('.db','')}_{tag}.png")
    C.log_trial("a245_probe.py", a.db, ",".join(a.runs), {"runs": a.runs}, res, time.time() - t0,
                notes="analyses 2/4/5 from icp_probe")


if __name__ == "__main__":
    main()
