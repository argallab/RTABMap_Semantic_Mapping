"""Analysis 3: independent check with held-out AprilTags.

For every run of tools/tag_holdout (one fold of tags removed from ALL
optimization), each held-out tag's observations are placed in the world with
the run's poses (node pose * landmark link transform) and their spread across
the run is measured.  Nothing in the pipeline saw these tags, so if ICP (or the
latency shift) really improves poses, the held-out tags must tighten.

Spread per tag: RMS of XY distances to the tag's mean (same definition as the
exporter's tag_std, XY only), and the number of visits (node-ID gaps > 150).
Folds cover every tag exactly once, so per-stage numbers are over all tags.

Also: validation that the no-holdout run reproduces the exporter's dumps, and
the wall metric per stage (global median) for the same runs.

Usage: python3 a3_tags.py 09-29-2.db [--exporter ../output/<ts>]
"""
import argparse
import glob
import json
import os
import time

import numpy as np

import common as C

STAGES = ["rtabmap_graph", "tag_GTSAM", "icp_round_1", "icp_round_2"]


def landmark_obs(con):
    obs = {}
    for frm, to, tb in con.execute("select from_id, to_id, transform from Link where type=8"):
        T = np.eye(4)
        T[:3, :] = np.frombuffer(tb, dtype=np.float32).reshape(3, 4)
        obs.setdefault(-to, []).append((frm, T))
    return obs


def tag_spread(obs_t, poses):
    pts, ids = [], []
    for frm, T in obs_t:
        P = poses.get(frm)
        if P is None:
            continue
        w = P @ T
        pts.append(w[:2, 3]); ids.append(frm)
    if len(pts) < 3:
        return None
    pts = np.array(pts)
    m = pts.mean(axis=0)
    rms = float(np.sqrt(((pts - m) ** 2).sum(axis=1).mean() / 2.0) * 1000.0)
    ids = np.sort(np.array(ids))
    visits = int(1 + np.sum(np.diff(ids) > 150))
    return rms, len(pts), visits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--exporter", help="exporter output dir with poses_*.csv (validation)")
    ap.add_argument("--no-walls", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    tag = a.db.replace(".db", "")
    hdir = f"{C.ROOT}/analysis/holdout/{tag}"
    con = C.connect(a.db)
    obs = landmark_obs(con)
    res = {}

    # ── validation: no-holdout run == exporter dumps ─────────────────────
    if a.exporter:
        for st in STAGES:
            p1, p2 = f"{hdir}/validate_{st}.csv", f"{a.exporter}/poses_{st}.csv"
            if not (os.path.exists(p1) and os.path.exists(p2)):
                continue
            A, B = C.load_pose_csv(p1), C.load_pose_csv(p2)
            d = [np.linalg.norm(A[i][:2, 3] - B[i][:2, 3]) for i in A if i in B]
            print(f"validate {st:13s}: {len(d)} nodes, max |dxy| = {1000 * max(d):.3f} mm, mean {1000 * np.mean(d):.4f} mm")
            res.setdefault("validate", {})[st] = {"n": len(d), "max_mm": 1000 * max(d)}

    scans = None if a.no_walls else C.clean_scans(a.db, C.load_rtabmap_opt_poses(con).keys())
    VARIANTS = {"s0": "tags 0.02/0.05, no shift", "s015": "tags 0.02/0.05, shift -0.15 s",
                "v2_s0": "tags 0.2/999, no shift", "v2_s015": "tags 0.2/999, shift -0.15 s"}
    for shift in VARIANTS:
        per_stage = {st: [] for st in STAGES}
        per_stage_in = {st: [] for st in STAGES}
        walls = {st: [] for st in STAGES}
        folds = sorted(glob.glob(f"{hdir}/fold[0-9]_{shift}_heldout.txt"))
        for hf in folds:
            prefix = hf[: -len("_heldout.txt")]
            held = {int(x) for x in open(hf).read().split()}
            for st in STAGES:
                path = f"{prefix}_{st}.csv"
                if not os.path.exists(path):
                    continue
                poses = C.load_pose_csv(path)
                for t, o in obs.items():
                    s = tag_spread(o, poses)
                    if s is None:
                        continue
                    (per_stage if t in held else per_stage_in)[st].append((t,) + s)
                if scans is not None:
                    w = C.wall_metric(scans, poses)
                    walls[st].append(w["global_median_mm"])
        if not folds:
            continue
        print(f"\n=== held-out tag spread  {a.db}  {VARIANTS[shift]}  ({len(folds)} folds) ===")
        print("  stage            held-out tags: mean / median XY RMS (mm), n tags | in-graph tags mean | wall global median (mean over folds)")
        out = {}
        for st in STAGES:
            v = np.array([x[1] for x in per_stage[st]])
            vm = np.array([x[1] for x in per_stage[st] if x[3] >= 2])
            vin = np.array([x[1] for x in per_stage_in[st]])
            if len(v) == 0:
                continue
            row = dict(heldout_mean=float(v.mean()), heldout_median=float(np.median(v)), n=len(v),
                       heldout_multivisit_mean=float(vm.mean()) if len(vm) else float("nan"), n_multivisit=len(vm),
                       ingraph_mean=float(vin.mean()) if len(vin) else float("nan"),
                       walls_global=float(np.mean(walls[st])) if walls[st] else float("nan"),
                       per_tag={int(x[0]): round(x[1], 1) for x in per_stage[st]})
            out[st] = row
            print(f"  {st:15s}  {row['heldout_mean']:6.1f} / {row['heldout_median']:6.1f}  (n={row['n']}; multi-visit "
                  f"{row['heldout_multivisit_mean']:.1f}, n={row['n_multivisit']}) | {row['ingraph_mean']:6.1f} | {row['walls_global']:.1f}")
        res[shift] = out
    os.makedirs(f"{C.ROOT}/analysis/results", exist_ok=True)
    with open(f"{C.ROOT}/analysis/results/a3_{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    C.log_trial("a3_tags.py", a.db, "tag_holdout folds", {"stages": STAGES, "folds": "every 4th sorted tag id",
                "shifts": [0, -0.15]}, {k: {s: {kk: vv for kk, vv in r.items() if kk != "per_tag"} for s, r in v.items()}
                if isinstance(v, dict) else v for k, v in res.items()}, time.time() - t0,
                notes="analysis 3: held-out AprilTag consistency")


if __name__ == "__main__":
    main()
