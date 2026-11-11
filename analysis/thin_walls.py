"""Centre-line wall thinning on a finished grid (prototype for the exporter).

Wall-like occupied cells are reduced to a skeleton, spurs shorter than PRUNE
cells are trimmed, and the skeleton is redrawn WIDTH cells wide.  Untouched:
  * feature cells (height-span keep flags from grid_sim), dilated by 1 cell
  * solid blobs: cells >= BLOB_DT cells from free/unknown, in seed areas of
    >= MIN_SEED cells that are compact (not >= ELONG:1 and >= MIN_WALL_LEN
    long — that is a thick wall), grown back over their footprint
  * ledge walls: occupied cells within LEDGE_BAND cells of unknown space
    inside user-marked grid squares (50-cell squares, A.. columns, 1.. rows)
Cells removed from a wall become free if they are nearer free space than
unknown space, else unknown (the unseen back of an outer wall stays unknown).

Usage: python3 thin_walls.py 10-12-01.db ../output/2026-10-12_15-37-25 --veto pct \
          --ledge B2 C2 D2 F2 H2 I2 J1 K1 K2 K3 K4 E6 B6 C6
"""
import argparse
import os

import cv2
import numpy as np

import common as C
import grid_sim as GS

BLOB_DT, MIN_SEED, PRUNE, WIDTH, LEDGE_BAND = 3.5, 12, 5, 3, 8
ELONG, MIN_WALL_LEN = 4.0, 20   # thick core >= 4:1 and >= 50 cm long = wall, not furniture
LEDGE_ZMIN, LEDGE_NEAR_UNK, LEDGE_MIN_LEN, LEDGE_GROW = 0.50, 10, 20, 6   # auto ledge detection


def endpoints(s):
    n = cv2.filter2D(s.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - s
    return s & (n == 1)


def detect_ledge(img, aux):
    """Ledge seeds: occupied cells near unknown space (outer walls) whose
    wall-band hits all sit high (p10 >= LEDGE_ZMIN, hit weight >= 2) — the
    front of a ~0.75 m ledge.  Seed runs shorter than LEDGE_MIN_LEN cells
    (after bridging 2-cell gaps) are dropped; the protected band is every
    occupied cell within LEDGE_NEAR_UNK of unknown and LEDGE_GROW of a seed."""
    occ, unk = img == 0, img == 205
    d_unk = cv2.distanceTransform((~unk).astype(np.uint8), cv2.DIST_L2, 5)
    seed = (occ & (aux["hw"] >= 2) & (aux["zlo"] >= LEDGE_ZMIN) & (d_unk <= LEDGE_NEAR_UNK)).astype(np.uint8)
    joined = cv2.dilate(seed, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        ys, xs = np.nonzero(lab == k)
        (_, _), (w_, h_), _ = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        keep[k] = max(w_, h_) >= LEDGE_MIN_LEN
    seed = seed.astype(bool) & keep[lab]
    near = cv2.dilate(seed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * LEDGE_GROW + 1,) * 2)).astype(bool)
    return occ & (d_unk <= LEDGE_NEAR_UNK) & near, seed


def thin(img, feat, ledge_squares=(), sq=50, ledge_mask=None):
    occ = img == 0
    free = img == 254
    unk = img == 205
    H, W = img.shape
    # protected: features
    prot = cv2.dilate(feat.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & occ
    # protected: solid blobs (furniture)
    dt = cv2.distanceTransform(occ.astype(np.uint8), cv2.DIST_L2, 5)
    seed = (dt >= BLOB_DT).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < MIN_SEED:
            continue
        # a thick WALL's core is a long narrow ridge; furniture cores are compact
        ys, xs = np.nonzero(lab == k)
        (_, _), (w_, h_), _ = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        lng, sht = max(w_, h_) + 1, min(w_, h_) + 1
        keep[k] = not (lng >= ELONG * sht and lng >= MIN_WALL_LEN)
    blob = cv2.dilate(keep[lab].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool) & occ
    # protected: ledge walls (outer side borders unknown) in the marked squares
    d_unk = cv2.distanceTransform((~unk).astype(np.uint8), cv2.DIST_L2, 5)
    ledge = np.zeros_like(occ)
    for s in ledge_squares:
        c = ord(s[0].upper()) - ord("A"); r = int(s[1:]) - 1
        ledge[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = True
    # a ledge wall is the whole wall band next to unknown: grow from unknown-adjacent cells
    ledge &= occ & (d_unk <= LEDGE_BAND)
    if ledge_mask is not None:
        ledge |= ledge_mask
    protect = prot | blob | ledge
    wall = occ & ~protect
    sk = cv2.ximgproc.thinning(wall.astype(np.uint8) * 255) > 0
    s = sk.copy()
    for _ in range(PRUNE):
        s &= ~endpoints(s)
    grow = s.copy()
    for _ in range(PRUNE):
        grow |= cv2.dilate(endpoints(grow).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & sk
    new_wall = cv2.dilate(grow.astype(np.uint8), np.ones((WIDTH, WIDTH), np.uint8)).astype(bool) & (wall | free)
    out = img.copy()
    removed = wall & ~new_wall
    d_free = cv2.distanceTransform((~free).astype(np.uint8), cv2.DIST_L2, 5)
    out[removed & (d_free <= d_unk)] = 254
    out[removed & (d_free > d_unk)] = 205
    out[new_wall] = 0
    out[protect] = 0
    # seal: a removed cell that now separates nothing (free touches unknown
    # through it) is put back, until no thinning-made free/unknown contact is left
    def contact(o):
        fr, un = o == 254, o == 205
        nb_un = np.zeros_like(un); nb_un[1:] |= un[:-1]; nb_un[:-1] |= un[1:]; nb_un[:, 1:] |= un[:, :-1]; nb_un[:, :-1] |= un[:, 1:]
        nb_fr = np.zeros_like(fr); nb_fr[1:] |= fr[:-1]; nb_fr[:-1] |= fr[1:]; nb_fr[:, 1:] |= fr[:, :-1]; nb_fr[:, :-1] |= fr[:, 1:]
        return (fr & nb_un) | (un & nb_fr)
    base = contact(img)
    restored = np.zeros_like(occ)
    for _ in range(20):
        bad = contact(out) & ~base & removed & (out != 0)
        if not bad.any():
            break
        out[bad] = 0; restored |= bad
    return out, dict(feature=prot, blob=blob, ledge=ledge, removed=removed & ~restored, restored=restored)


def leaks(img):
    """free cells 4-adjacent to unknown (openings of free space into unknown)."""
    fr, un = img == 254, img == 205
    adj = np.zeros_like(un)
    adj[1:] |= un[:-1]; adj[:-1] |= un[1:]; adj[:, 1:] |= un[:, :-1]; adj[:, :-1] |= un[:, 1:]
    return int((fr & adj).sum())


def thickness(img):
    occ = (img == 0).astype(np.uint8)
    dt = cv2.distanceTransform(occ, cv2.DIST_L2, 5)
    r = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-3)
    return np.median((2 * dt[r] - 1) * 2.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db"); ap.add_argument("run")
    ap.add_argument("--veto", default="pct")
    ap.add_argument("--ledge", nargs="*", default=[])
    ap.add_argument("--auto-ledge", action="store_true")
    a = ap.parse_args()
    scans, origins, stamps, align, _ = GS.load_run(a.db, a.run)
    img, feat, aux = GS.build_grid(scans, origins, stamps, align, a.veto, log=lambda *x: None, want_aux=True)
    lm = detect_ledge(img, aux)[0] if a.auto_ledge else None
    out, parts = thin(img, feat, a.ledge, ledge_mask=lm)
    print(f"protected: features {parts['feature'].sum()}  blobs {parts['blob'].sum()}  ledge walls {parts['ledge'].sum()} cells   seal restored {parts['restored'].sum()}")
    for nm, g in (("before", img), ("after", out)):
        print(f"  {nm:6s}: occupied {(g==0).sum()*GS.RES**2:.2f} m2   wall thickness median {thickness(g):.1f} cm   free-unknown contact cells {leaks(g)}")
    ts = os.path.basename(os.path.normpath(a.run))
    d = os.path.join(C.ROOT, "analysis", "results", "gridsim")
    tag = "_auto" if a.auto_ledge else ""
    cv2.imwrite(os.path.join(d, f"thin_{ts}{tag}.pgm"), out)
    # side by side with grid labels; protected cells tinted
    def labelled(g, tint):
        v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if tint:
            v[parts["ledge"]] = (180, 90, 0); v[parts["blob"] & ~parts["ledge"]] = (0, 120, 0); v[parts["feature"]] = (0, 0, 200)
        v = cv2.resize(v, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        Hh, Ww = v.shape[:2]; M = 30
        o = cv2.copyMakeBorder(v, M, 0, M, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for k, x in enumerate(range(0, Ww, 100)):
            cv2.line(o, (x + M, M), (x + M, Hh + M), (200, 160, 60), 1); cv2.putText(o, chr(65 + k), (x + M + 44, 21), 0, 0.6, (200, 80, 0), 2)
        for k, y in enumerate(range(0, Hh, 100)):
            cv2.line(o, (M, y + M), (Ww + M, y + M), (200, 160, 60), 1); cv2.putText(o, str(k + 1), (8, y + M + 56), 0, 0.6, (200, 80, 0), 2)
        return o
    cv2.imwrite(os.path.join(d, f"thin_{ts}{tag}_compare.png"),
                np.vstack([labelled(img, False), np.full((10, labelled(img, False).shape[1], 3), 128, np.uint8), labelled(out, True)]))
    print(f"  wrote analysis/results/gridsim/thin_{ts}{tag}_compare.png (top: ledge-fix grid, bottom: thinned; blue = ledge walls, green = furniture, red = features)")


if __name__ == "__main__":
    main()
