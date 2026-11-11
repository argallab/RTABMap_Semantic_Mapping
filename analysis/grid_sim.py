"""Python replica of the exporter's occupancy-grid step, for testing grid
changes on whole recordings before building.

Inputs come from an exporter run (output/<ts>): the exact posed, cleaned
points (cloud/<ts>.pcd, streamed scan by scan in node order), the final poses
(poses_manhattan.csv, else the last stage present) and the alignment angle
(metrics.txt).  Per-scan point counts come from a replica of clean_scans
(crop + top-edge filter + 2D ROR); the split is verified against the PCD.

Grid (mirrors finalize_and_return_result, kHeightAwareFree=false):
  pass 1: per-cell hit weight, hit z min/max, near/all hit + scan counts, first/last hit time
  ray pass: log-odds, hit +2*w(r) (wall band), free -0.15, clamp ±8, own-scan
            hit protection, height veto (w >= 2, ray outside [zmin-5cm, zmax+5cm])
  near keep, height-span keep (+10 s persistence), 3x3 close, CC filter < 75
  (feature cells exempt), chair path free, enclosed unknown holes <= 1 m² free.
Rays are traced with a symmetric DDA (round(k·d/n)); the exporter uses
Bresenham — they differ only on exact ties.

Variants (--veto): "minmax" (exporter) or "pct" (veto range = p10..p90 of the
cell's wall-band hit heights, from a 2 cm height histogram).

Usage: python3 grid_sim.py 10-12-01.db ../output/2026-10-12_15-37-25 [--veto pct] [--out name]
"""
import argparse
import os
import re
import zlib

import cv2
import numpy as np
from scipy.spatial import cKDTree

import common as C

RES, PAD = 0.025, 1.0
WALL_ZLO, WALL_ZHI = 0.10, 1.00
OCC_HIT, FREE_MISS, CLAMP, R_REF = 2.0, 0.15, 8.0, 1.0
SENSOR_Z = 0.12
VETO_MIN_W, VETO_MARGIN = 2.0, 0.05
NEAR_R, KEEP_MIN_HITS, KEEP_MIN_SCANS = 1.3, 8, 3
SPAN_MIN_HITS, SPAN_MIN_SCANS, SPAN_MAX_DZ, SPAN_MIN_S = 15, 5, 0.12, 10.0
MIN_CC, MAX_HOLE_M2 = 75, 1.0


# ---------------------------------------------------------------------------
# scan cleaning replica (clean_scans incl. top-edge filter) -> per-scan counts
# ---------------------------------------------------------------------------
def _raw_scans(con, ids):
    want = set(ids)
    out = {}
    for nid, scan_b, info_b in con.execute(
            "select id, scan, scan_info from Data where scan is not null and length(scan) > 0"):
        if nid not in want:
            continue
        pts = np.frombuffer(zlib.decompress(scan_b), dtype=np.float32).reshape(-1, 3)
        T = C._scan_local_tf(info_b)
        base = (pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        out[nid] = (base, T)
    return out


def clean_counts(con, ids, cam_yaw_deg=0.0):
    """Cleaned base-frame points per scan, exactly as clean_scans keeps them."""
    raw = _raw_scans(con, ids)
    # top edge: 99.5th-pct z per 10 cm range bin over every 4th pose id
    NR, NZ, Z0 = 27, 300, -0.5
    hist = np.zeros((NR, NZ), np.int64)
    for k, i in enumerate(ids):
        if k % 4 or i not in raw:
            continue
        b = raw[i][0]
        b = b[np.isfinite(b).all(1)]
        rb = (np.hypot(b[:, 0], b[:, 1]) / 0.10).astype(int)
        zb = np.floor((b[:, 2] - Z0) / 0.01).astype(int)
        m = (rb >= 0) & (rb < NR) & (zb >= 0) & (zb < NZ)
        np.add.at(hist, (rb[m], zb[m]), 1)
    top = np.full(NR, 1e9)
    for rb in range(NR):
        tot = hist[rb].sum()
        if tot < 100:
            continue
        top[rb] = Z0 + (np.argmax(np.cumsum(hist[rb]) >= 0.995 * tot) + 1) * 0.01
    out = {}
    for i in ids:
        if i not in raw:
            continue
        b = raw[i][0]
        if cam_yaw_deg:   # dev WC_CAMERA_YAW_DEG: rotate scans about base_link (float32 like the C++)
            c_, s_ = np.float32(np.cos(np.radians(cam_yaw_deg))), np.float32(np.sin(np.radians(cam_yaw_deg)))
            b = np.c_[c_ * b[:, 0] - s_ * b[:, 1], s_ * b[:, 0] + c_ * b[:, 1], b[:, 2]].astype(np.float32)
        b = b[np.isfinite(b).all(1)]
        r = np.hypot(b[:, 0], b[:, 1])
        m = (b[:, 2] >= 0.05) & (b[:, 2] <= 2.0) & (r >= 0.30) & (r <= 2.50)
        rb = np.minimum(NR - 1, (r / 0.10).astype(int))
        m &= ~((r > 1.8) & (top[rb] < 1e8) & (b[:, 2] > top[rb] - 0.10))
        c = b[m]
        if len(c) == 0:
            continue
        cnt = cKDTree(c[:, :2]).query_ball_point(c[:, :2], 0.05, return_length=True)
        k = c[cnt > 12]
        if len(k) < 50:
            continue
        out[i] = (k, raw[i][1])
    return out


def load_run(db, run):
    ts = os.path.basename(os.path.normpath(run))
    con = C.connect(db)
    nodes = C.load_nodes(con)
    for st in ("final", "manhattan", "icp_round_2", "icp_round_1", "tag_GTSAM", "rtabmap_graph"):
        f = os.path.join(run, f"poses_{st}.csv")
        if os.path.exists(f):
            P = C.load_pose_csv(f)
            break
    ids = sorted(i for i in P if i >= 1)
    menv = open(os.path.join(run, "metrics.txt")).read()
    m = re.search(r"WC_CAMERA_YAW_DEG=([-0-9.]+)", menv)
    cam_yaw = float(m.group(1)) if m else 0.0
    cache = os.path.join(C.CACHE_DIR, f"gridsim_clean_{db.replace('.db', '')}_{len(ids)}" + (f"_yaw{cam_yaw:g}" if cam_yaw else "") + ".npz")
    if os.path.exists(cache):
        z = np.load(cache)
        cids, offs, pts, tfs = z["ids"], z["offs"], z["pts"], z["tfs"]
        cl = {int(i): (pts[offs[k]:offs[k + 1]], tfs[k]) for k, i in enumerate(cids)}
    else:
        cl = clean_counts(con, ids, cam_yaw)
        cids = np.array(sorted(cl))
        offs = np.concatenate([[0], np.cumsum([len(cl[i][0]) for i in cids])])
        os.makedirs(C.CACHE_DIR, exist_ok=True)
        np.savez(cache, ids=cids, offs=offs, pts=np.concatenate([cl[i][0] for i in cids]),
                 tfs=np.array([cl[i][1] for i in cids]))
    # exact exporter points from the PCD, split by the replica's per-scan counts
    pcd = os.path.join(run, "cloud", f"{ts}.pcd")
    if not os.path.exists(pcd):          # folder renamed after the run (or files renamed): take the cloud that's there
        import glob
        found = sorted(glob.glob(os.path.join(run, "cloud", "*.pcd")))
        pcd = found[0] if found else pcd
    with open(pcd, "rb") as f:
        while not f.readline().startswith(b"DATA"):
            pass
        W = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 3)
    order = [i for i in sorted(cl) if i in P]
    n = np.array([len(cl[i][0]) for i in order])
    if n.sum() != len(W):
        raise SystemExit(f"replica kept {n.sum()} points, PCD has {len(W)} — cleaning replica out of sync")
    scans, origins, k = {}, {}, 0
    for i, m in zip(order, n):
        scans[i] = W[k:k + m]; k += m
        So = P[i] @ cl[i][1]
        origins[i] = So[:2, 3]
    # spot check: replica posed points vs PCD
    i0 = order[len(order) // 2]
    mine = cl[i0][0].astype(np.float64) @ P[i0][:3, :3].T + P[i0][:3, 3]
    err = np.abs(mine - scans[i0]).max()
    ang = float(re.search(r"grid alignment: rotated ([-0-9.]+)", open(os.path.join(run, "metrics.txt")).read()).group(1))
    stamps = {i: nodes[i][0] for i in order}
    return scans, origins, stamps, np.radians(ang), err


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
def build_grid(scans, origins, stamps, align, veto="minmax", log=print, want_aux=False):
    ca, sa = np.float32(np.cos(align)), np.float32(np.sin(align))
    rot = lambda xy: np.c_[ca * xy[:, 0] - sa * xy[:, 1], sa * xy[:, 0] + ca * xy[:, 1]]
    ids = sorted(scans)
    R = {i: rot(scans[i][:, :2].astype(np.float32)) for i in ids}
    O = {i: rot(origins[i][None].astype(np.float32))[0] for i in ids}
    allxy = np.concatenate([R[i] for i in ids] + [np.array([O[i] for i in ids])])
    xmin, ymin = allxy.min(0) - PAD
    xmax, ymax = allxy.max(0) + PAD
    gw = int(np.ceil((xmax - xmin) / RES)) + 1
    gh = int(np.ceil((ymax - ymin) / RES)) + 1
    N = gw * gh
    cell = lambda q: (((q[:, 0] - xmin) / RES).astype(np.int64), ((q[:, 1] - ymin) / RES).astype(np.int64))

    # pass 1
    hw = np.zeros(N); hzmin = np.full(N, 1e9); hzmax = np.full(N, -1e9)
    near_h = np.zeros(N, np.int64); near_s = np.zeros(N, np.int64)
    all_h = np.zeros(N, np.int64); all_s = np.zeros(N, np.int64)
    tf = np.full(N, 1e18); tl = np.full(N, -1e18)
    zh = np.zeros((N, 45), np.float32) if veto == "pct" else None   # 2 cm bins 0.10-1.00
    for i in ids:
        p = scans[i]; band = (p[:, 2] >= WALL_ZLO) & (p[:, 2] <= WALL_ZHI)
        cx, cy = cell(R[i][band]); ok = (cx >= 0) & (cx < gw) & (cy >= 0) & (cy < gh)
        idx = (cy * gw + cx)[ok]; z = p[band, 2][ok]
        r = np.hypot(*(R[i][band][ok] - O[i]).T)
        w = np.where(r > R_REF, (R_REF / np.maximum(r, 1e-6)) ** 2, 1.0)
        np.add.at(hw, idx, w); np.minimum.at(hzmin, idx, z); np.maximum.at(hzmax, idx, z)
        np.add.at(all_h, idx, 1); u = np.unique(idx); all_s[u] += 1
        np.minimum.at(tf, u, stamps[i]); np.maximum.at(tl, u, stamps[i])
        nm = r < NEAR_R
        np.add.at(near_h, idx[nm], 1); near_s[np.unique(idx[nm])] += 1
        if zh is not None:
            np.add.at(zh, (idx, np.minimum(44, ((z - WALL_ZLO) / 0.02).astype(int))), 1)
    if veto == "pct":
        cum = np.cumsum(zh, 1) / np.maximum(zh.sum(1), 1)[:, None]
        vlo = WALL_ZLO + 0.02 * np.argmax(cum >= 0.10, 1)
        vhi = WALL_ZLO + 0.02 * (np.argmax(cum >= 0.90, 1) + 1)
        del zh, cum
    else:
        vlo, vhi = hzmin, hzmax
    veto_ok = hw >= VETO_MIN_W

    # ray pass
    lo = np.zeros(N)
    n_free = n_prot = n_veto = 0
    for i in ids:
        p = scans[i]
        ex, ey = cell(R[i]); ocx, ocy = int((O[i][0] - xmin) / RES), int((O[i][1] - ymin) / RES)
        if not (0 <= ocx < gw and 0 <= ocy < gh):
            continue
        inb = (ex >= 0) & (ex < gw) & (ey >= 0) & (ey < gh)
        p, ex, ey = p[inb], ex[inb], ey[inb]
        band = (p[:, 2] >= WALL_ZLO) & (p[:, 2] <= WALL_ZHI)
        hit_idx = ey[band] * gw + ex[band]
        own = np.zeros(N, bool); own[hit_idx] = True
        r = np.hypot(*(R[i][inb] - O[i]).T)
        w = np.where(r > R_REF, (R_REF / np.maximum(r, 1e-6)) ** 2, 1.0)
        # hits (all positive within a scan: sequential clamp == clamp of the sum)
        add = np.bincount(hit_idx, weights=OCC_HIT * w[band], minlength=N)
        # free cells along rays, endpoint excluded
        dx, dy = ex - ocx, ey - ocy
        n = np.maximum(np.abs(dx), np.abs(dy))
        K = int(n.max()) if len(n) else 0
        if K > 0:
            k = np.arange(K)[None, :]
            valid = k < n[:, None]
            t = k / np.maximum(n, 1)[:, None]
            cx = ocx + np.rint(t * dx[:, None]).astype(np.int64)
            cy = ocy + np.rint(t * dy[:, None]).astype(np.int64)
            zr = SENSOR_Z + t * (p[:, 2][:, None] - SENSOR_Z)
            cx, cy, zr = cx[valid], cy[valid], zr[valid]
            idx = cy * gw + cx
            vt = veto_ok[idx] & ((zr > vhi[idx] + VETO_MARGIN) | (zr < vlo[idx] - VETO_MARGIN))
            pr = own[idx] & ~vt
            fr = ~vt & ~pr
            n_veto += vt.sum(); n_prot += pr.sum(); n_free += fr.sum()
            sub = np.bincount(idx[fr], minlength=N) * FREE_MISS
        else:
            sub = 0.0
        lo = np.minimum(lo + add, CLAMP)          # cells hit by this scan get no free update from it
        lo = np.maximum(lo - sub, -CLAMP)
    grid = np.full(N, -1, np.int8)
    grid[lo > 0] = 100
    grid[lo < 0] = 0
    tot = max(1, n_free + n_prot + n_veto)
    log(f"  ray cells: veto {100*n_veto/tot:.2f}%  own-scan protected {100*n_prot/max(1,n_prot+n_free):.2f}%")

    near = (grid != 100) & (near_h >= KEEP_MIN_HITS) & (near_s >= KEEP_MIN_SCANS) & (tl - tf >= SPAN_MIN_S)
    grid[near] = 100
    feat = (all_h >= SPAN_MIN_HITS) & (all_s >= SPAN_MIN_SCANS) & (hzmax - hzmin <= SPAN_MAX_DZ) & (tl - tf >= SPAN_MIN_S)
    span_restored = (feat & (grid != 100)).sum()
    grid[feat] = 100
    log(f"  near keep {near.sum()} cells, span keep {span_restored} restored / {feat.sum()} flagged")

    G = grid.reshape(gh, gw)
    occ = (G == 100).astype(np.uint8) * 255
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    G[(occ > 0)] = 100
    n_lab, lab, st, _ = cv2.connectedComponentsWithStats((G == 100).astype(np.uint8))
    small = st[:, cv2.CC_STAT_AREA] < MIN_CC; small[0] = False
    G[small[lab] & ~feat.reshape(gh, gw)] = -1
    for i in ids:
        ocx, ocy = int((O[i][0] - xmin) / RES), int((O[i][1] - ymin) / RES)
        if 0 <= ocx < gw and 0 <= ocy < gh and G[ocy, ocx] != 100:
            G[ocy, ocx] = 0
    n_lab, lab, st, _ = cv2.connectedComponentsWithStats((G == -1).astype(np.uint8), connectivity=4)
    x, y, w_, h_, a = (st[:, k] for k in range(5))
    fill = ~((x == 0) | (y == 0) | (x + w_ == gw) | (y + h_ == gh)) & (a <= int(MAX_HOLE_M2 / RES ** 2))
    fill[0] = False
    G[fill[lab]] = 0
    img = np.full((gh, gw), 205, np.uint8)
    img[G == 0] = 254
    img[G == 100] = 0
    aux = dict(hw=hw.reshape(gh, gw)[::-1], zlo=np.asarray(vlo).reshape(gh, gw)[::-1],
               zhi=np.asarray(vhi).reshape(gh, gw)[::-1], scans=all_s.reshape(gh, gw)[::-1],
               hits=all_h.reshape(gh, gw)[::-1], tspan=(tl - tf).reshape(gh, gw)[::-1])
    return (np.flipud(img), feat.reshape(gh, gw)[::-1], aux) if want_aux else (np.flipud(img), feat.reshape(gh, gw)[::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db"); ap.add_argument("run")
    ap.add_argument("--veto", default="minmax", choices=["minmax", "pct"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    scans, origins, stamps, align, err = load_run(a.db, a.run)
    print(f"loaded {len(scans)} scans; replica vs PCD max point error {err*1000:.3f} mm")
    img, feat = build_grid(scans, origins, stamps, align, a.veto)
    ts = os.path.basename(os.path.normpath(a.run))
    ref = cv2.imread(C.grid_pgm(a.run), cv2.IMREAD_GRAYSCALE)
    if ref is not None and ref.shape == img.shape:
        same = (ref == img).mean()
        print(f"  vs exporter grid: {100*same:.2f}% cells identical   occupied {(img==0).sum()*RES**2:.2f} m2 (exporter {(ref==0).sum()*RES**2:.2f})"
              f"   occ->other {((ref==0)&(img!=0)).sum()}  other->occ {((ref!=0)&(img==0)).sum()}")
    elif ref is not None:
        print(f"  shape differs from exporter grid: {img.shape} vs {ref.shape}")
    out = a.out or f"sim_{ts}_{a.veto}"
    d = os.path.join(C.ROOT, "analysis", "results", "gridsim")
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, f"{out}.pgm"), img)
    print(f"  wrote analysis/results/gridsim/{out}.pgm")


if __name__ == "__main__":
    main()
