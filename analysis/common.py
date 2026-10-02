"""Shared helpers for the offline evaluation scripts.

Everything here reads RTAB-Map .db files read-only (sqlite URI mode=ro) and
replicates the parts of the C++ exporter that the metrics depend on:

  * clean_scans()      -> WCDatabaseExporter::clean_scans()
                          (crop z/range, 2D radius outlier removal)
  * wall_metric()      -> WCDatabaseExporter::wall_thickness()
                          (+ optional per-point attribute bins)

No randomness is used anywhere; results are deterministic for a given DB,
pose set and config.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import subprocess
import time
import zlib
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(ROOT, "databases")
OUT_DIR = os.path.join(ROOT, "output")
CACHE_DIR = os.path.join(ROOT, "analysis", "cache")
LEDGER = os.path.join(ROOT, "analysis", "ledger.csv")


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------
def connect(db_name: str) -> sqlite3.Connection:
    path = os.path.join(DB_DIR, db_name)
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _mat34(blob) -> np.ndarray:
    """RTAB-Map Transform blob (12 float32, row-major 3x4) -> 4x4 float64."""
    T = np.eye(4)
    T[:3, :] = np.frombuffer(blob, dtype=np.float32).reshape(3, 4)
    return T


def load_nodes(con) -> dict:
    """{id: (stamp, weight, odom 4x4)} for every node."""
    out = {}
    for nid, stamp, weight, pose in con.execute("select id, stamp, weight, pose from Node"):
        out[nid] = (stamp, weight, _mat34(pose))
    return out


def load_rtabmap_opt_poses(con) -> dict:
    """RTAB-Map's own optimized graph as saved in Admin (robot ids > 0 only)."""
    ids_b, poses_b = con.execute("select opt_ids, opt_poses from Admin").fetchone()
    ids = np.frombuffer(zlib.decompress(ids_b), dtype=np.int32)
    poses = np.frombuffer(zlib.decompress(poses_b), dtype=np.float32).reshape(-1, 3, 4)
    assert len(ids) == len(poses), (len(ids), len(poses))
    out = {}
    for i, P in zip(ids, poses):
        if i > 0:
            T = np.eye(4)
            T[:3, :] = P
            out[int(i)] = T
    return out


def load_odom_poses(con, ids=None) -> dict:
    nodes = load_nodes(con)
    return {i: v[2] for i, v in nodes.items() if ids is None or i in ids}


def load_pose_csv(path: str) -> dict:
    """Pose dump written by the exporter: id,stamp,x,y,z,qx,qy,qz,qw."""
    from scipy.spatial.transform import Rotation as R
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            T = np.eye(4)
            T[:3, :3] = R.from_quat([float(row[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
            T[:3, 3] = [float(row[k]) for k in ("x", "y", "z")]
            out[int(row["id"])] = T
    return out


def _scan_local_tf(scan_info) -> np.ndarray:
    """scan_info = [format, rangeMin, rangeMax, angleMin, angleMax, angleInc,
    maxPoints, 12 x localTransform]."""
    a = np.frombuffer(scan_info, dtype=np.float32)
    T = np.eye(4)
    T[:3, :] = a[7:19].reshape(3, 4)
    return T


# ---------------------------------------------------------------------------
# Scan cleaning (replica of WCDatabaseExporter::clean_scans)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CleanCfg:
    z_lo: float = 0.05
    z_hi: float = 2.00
    range_min: float = 0.30
    range_max: float = 2.50
    ror_radius: float = 0.05
    ror_min_neigh: int = 12      # neighbours excluding self (PCL semantics)
    min_scan_pts: int = 50


def clean_scans(db_name: str, node_ids, cfg: CleanCfg = CleanCfg(), use_cache=True) -> dict:
    """{id: (N,3) float32 cleaned points in base_link} for ids in node_ids
    that have a scan.  Cached to analysis/cache (keyed on DB + cfg)."""
    node_ids = sorted(int(i) for i in node_ids if i > 0)
    key = hashlib.sha1(json.dumps([db_name, asdict(cfg), node_ids]).encode()).hexdigest()[:16]
    cpath = os.path.join(CACHE_DIR, f"clean_{os.path.splitext(db_name)[0]}_{key}.npz")
    if use_cache and os.path.exists(cpath):
        z = np.load(cpath)
        ids, offs, pts = z["ids"], z["offs"], z["pts"]
        return {int(i): pts[offs[k]:offs[k + 1]] for k, i in enumerate(ids)}

    con = connect(db_name)
    want = set(node_ids)
    out = {}
    for nid, scan_b, info_b in con.execute(
            "select id, scan, scan_info from Data where scan is not null and length(scan) > 0"):
        if nid not in want:
            continue
        pts = np.frombuffer(zlib.decompress(scan_b), dtype=np.float32).reshape(-1, 3)
        T = _scan_local_tf(info_b)
        base = (pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        base = base[np.isfinite(base).all(axis=1)]
        r = np.hypot(base[:, 0], base[:, 1])
        m = (base[:, 2] >= cfg.z_lo) & (base[:, 2] <= cfg.z_hi) & \
            (r >= cfg.range_min) & (r <= cfg.range_max)
        crop = base[m]
        if len(crop) == 0:
            continue
        tree = cKDTree(crop[:, :2])
        cnt = tree.query_ball_point(crop[:, :2], cfg.ror_radius, return_length=True)
        kept = crop[cnt > cfg.ror_min_neigh]   # count includes self
        if len(kept) < cfg.min_scan_pts:
            continue
        out[nid] = kept
    con.close()

    os.makedirs(CACHE_DIR, exist_ok=True)
    ids = np.array(sorted(out), dtype=np.int64)
    offs = np.zeros(len(ids) + 1, dtype=np.int64)
    for k, i in enumerate(ids):
        offs[k + 1] = offs[k] + len(out[i])
    np.savez(cpath, ids=ids, offs=offs,
             pts=np.concatenate([out[i] for i in ids]) if len(ids) else np.zeros((0, 3), np.float32))
    return out


# ---------------------------------------------------------------------------
# Wall metric (replica of WCDatabaseExporter::wall_thickness, + bins)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WallCfg:
    z_lo: float = 0.10
    z_hi: float = 1.00
    target_pts: int = 1_500_000
    queries: int = 20_000
    radius: float = 0.20
    ref_window: int = 30
    local_window: int = 150
    min_ref_pts: int = 30
    min_height_span: float = 0.60
    min_length: float = 0.08
    max_line_ratio: float = 0.15
    min_bin_pts: int = 10        # bins only: min points in a bin to report a patch


def _robust_sigma_mm(d: np.ndarray) -> float:
    # Same as the C++ helper: upper median (element at n/2) for both steps.
    n = len(d)
    med = np.partition(d, n // 2)[n // 2]
    ad = np.abs(d - med)
    return 1.4826 * np.partition(ad, n // 2)[n // 2] * 1000.0


def pool_points(scans: dict, poses: dict, cfg: WallCfg, attrs: dict | None = None):
    """Strided world-frame XY pool, as in the C++.  attrs: {name: fn(nid, pts_base)
    -> per-point array} evaluated on the base-frame points of each scan."""
    total = sum(len(p) for p in scans.values())
    stride = max(1, total // cfg.target_pts)
    xy, pid, pz = [], [], []
    extra = {k: [] for k in (attrs or {})}
    k = 0
    for nid in sorted(scans):
        pts = scans[nid]
        n = len(pts)
        T = poses.get(nid)
        if T is None:
            k += n
            continue
        idx = np.arange(k, k + n)
        sel = (idx % stride) == 0
        k += n
        p = pts[sel]
        zm = (p[:, 2] >= cfg.z_lo) & (p[:, 2] <= cfg.z_hi)
        p = p[zm]
        w = p.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        xy.append(w[:, :2].astype(np.float32))
        pid.append(np.full(len(p), nid, dtype=np.int64))
        pz.append(p[:, 2])
        for name, fn in (attrs or {}).items():
            extra[name].append(np.asarray(fn(nid, p)))
    out = dict(xy=np.concatenate(xy), pid=np.concatenate(pid), pz=np.concatenate(pz))
    for name, v in extra.items():
        out[name] = np.concatenate(v)
    return out


def wall_metric(scans: dict, poses: dict, cfg: WallCfg = WallCfg(), bins: dict | None = None):
    """Returns dict with ref/local/global median+p90 (C++-equivalent) and, for
    each bin spec {name: (attr_fn, edges)}, per-bin per-window medians.

    Bin semantics: the patch (wall line, normal) is classified exactly as in
    the C++ using all points; the per-bin spread is the robust sigma of the
    offsets of only those neighbours whose attribute falls in the bin (needs
    >= min_bin_pts)."""
    attrs = {name: spec[0] for name, spec in (bins or {}).items()}
    P = pool_points(scans, poses, cfg, attrs)
    xy, pid, pz = P["xy"], P["pid"], P["pz"]
    if len(xy) < cfg.min_ref_pts:
        return None
    tree = cKDTree(xy)
    qstride = max(1, len(xy) // cfg.queries)
    qidx = np.arange(0, len(xy), qstride)
    nbrs = tree.query_ball_point(xy[qidx], cfg.radius)

    windows = {"ref": cfg.ref_window, "local": cfg.local_window, "global": None}
    th = {w: [] for w in windows}
    # Bins also get a "scan" window (same node only): pure single-scan sensor
    # spread, the floor no pose correction can go below.
    bin_windows = ["scan", "ref", "local", "global"]
    bin_th = {name: {w: [[] for _ in range(len(spec[1]) - 1)] for w in bin_windows}
              for name, spec in (bins or {}).items()}
    patches = []
    for qi, idx in zip(qidx, nbrs):
        if len(idx) < cfg.min_ref_pts:
            continue
        idx = np.asarray(idx)
        dt = np.abs(pid[idx] - pid[qi])
        ref = idx[dt <= cfg.ref_window]
        if len(ref) < cfg.min_ref_pts:
            continue
        if pz[ref].max() - pz[ref].min() < cfg.min_height_span:
            continue
        r = xy[ref].astype(np.float64)
        m = r.mean(axis=0)
        d = r - m
        a = (d[:, 0] ** 2).mean(); b = (d[:, 0] * d[:, 1]).mean(); c = (d[:, 1] ** 2).mean()
        h = np.sqrt(0.25 * (a - c) ** 2 + b * b)
        lmax = 0.5 * (a + c) + h
        lmin = max(0.0, 0.5 * (a + c) - h)
        if np.sqrt(lmax) < cfg.min_length or lmin > cfg.max_line_ratio * lmax:
            continue
        n = np.array([b, lmin - a]) if abs(b) > 1e-12 else (np.array([1.0, 0.0]) if a < c else np.array([0.0, 1.0]))
        n /= np.linalg.norm(n)
        off = (xy[idx].astype(np.float64) - m) @ n
        masks = {"ref": dt <= cfg.ref_window, "local": dt <= cfg.local_window, "global": np.ones(len(idx), bool)}
        for w, mk in masks.items():
            th[w].append(_robust_sigma_mm(off[mk]))
        bmasks = dict(masks, scan=dt == 0)
        for name, (_, edges) in (bins or {}).items():
            v = P[name][idx]
            bi = np.digitize(v, edges) - 1
            for w, mk in bmasks.items():
                for b_ in range(len(edges) - 1):
                    sel = mk & (bi == b_)
                    if sel.sum() >= cfg.min_bin_pts:
                        bin_th[name][w][b_].append(_robust_sigma_mm(off[sel]))
        patches.append(qi)

    def pct(v, q):
        v = np.asarray(v)
        at = int(q * (len(v) - 1))
        return float(np.partition(v, at)[at])

    res = {"n": len(th["ref"])}
    for w in windows:
        res[f"{w}_median_mm"] = pct(th[w], 0.5) if th[w] else float("nan")
        res[f"{w}_p90_mm"] = pct(th[w], 0.9) if th[w] else float("nan")
    for name, (_, edges) in (bins or {}).items():
        res[name] = []
        for b_ in range(len(edges) - 1):
            row = {"lo": float(edges[b_]), "hi": float(edges[b_ + 1])}
            for w in bin_windows:
                v = bin_th[name][w][b_]
                row[f"{w}_median_mm"] = pct(v, 0.5) if v else float("nan")
                row[f"{w}_n"] = len(v)
            res[name].append(row)
    return res


# ---------------------------------------------------------------------------
# Reference-trajectory hook (ATE / RPE)
# ---------------------------------------------------------------------------
def ate_rpe(poses: dict, reference_path: str | None):
    """>>> HOOK: ATE/RPE against a reference trajectory. <<<

    Intended use once a ceiling-camera ground-truth recording exists:
      1. load the reference (stamp, pose) series from `reference_path`,
      2. associate DB node stamps to reference stamps (and apply the
         gt_base_link -> base_link extrinsic),
      3. SE(2)/SE(3) align, then compute ATE (RMSE of aligned positions) and
         RPE over fixed path-length / time deltas.
    Not implemented: there is no reference trajectory yet.  Do not fill with
    placeholder numbers."""
    if reference_path is None:
        return None
    raise NotImplementedError("ATE/RPE hook: implement when a reference trajectory exists")


# ---------------------------------------------------------------------------
# Trial ledger
# ---------------------------------------------------------------------------
LEDGER_FIELDS = ["trial_id", "utc", "git_commit", "git_dirty", "script", "db",
                 "pose_source", "config", "metrics", "runtime_s", "notes"]


def git_state():
    try:
        h = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain",
                                              "--untracked-files=no"], text=True).strip())
        return h, dirty
    except Exception:
        return "unknown", True


def log_trial(script, db, pose_source, config, metrics, runtime_s, notes=""):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    new = not os.path.exists(LEDGER)
    h, dirty = git_state()
    tid = hashlib.sha1(f"{time.time()}{script}{db}{pose_source}".encode()).hexdigest()[:10]
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new:
            w.writeheader()
        w.writerow(dict(trial_id=tid, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        git_commit=h, git_dirty=dirty, script=script, db=db,
                        pose_source=pose_source, config=json.dumps(config, sort_keys=True),
                        metrics=json.dumps(metrics, sort_keys=True), runtime_s=round(runtime_s, 1),
                        notes=notes))
    return tid
