"""Compare a DEFAULT-pipeline run with a DEV-pipeline run of the same recording.

  python3 compare_pipelines.py R01_02.db ../output/R01_02_frozen ../output/R01_02_dev_3 [--bag tf_R01_02]

Writes analysis/<db>_frozen_vs_dev_<N>/ next to the dev run, i.e. output/<db>/analysis/ (or --out):
  compare.png   default grid | dev grid | overlay (green = occupied only in dev,
                red = occupied only in default), with the A..K / 1..n squares
  summary.md    pipeline ids, grid numbers, ATE vs ground truth (if a bag is
                found), and the metrics.txt lines that differ — paste into the user's notes (e.g. output/R01_02/analysis/R01.md)
Both grids are placed by their YAML origins, so they line up even if the dev
grid has a different size.
"""
import argparse
import os
import re

import cv2
import numpy as np

import common as C
import gt as G
import pipeline_diff as PD

RES = 0.025


def read_yaml_origin(pgm):
    y = os.path.splitext(pgm)[0] + ".yaml"
    txt = open(y).read()
    m = re.search(r"origin:\s*\[\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)", txt)
    return float(m.group(1)), float(m.group(2))


def metrics(run):
    f = os.path.join(run, "metrics.txt")
    return open(f).read().splitlines() if os.path.exists(f) else []


def metric(lines, prefix):
    for ln in lines:
        if ln.startswith(prefix):
            return ln
    return ""


def env_settings(lines):
    """{name: value} from the 'env:' metrics line (WC_* settings of the run)."""
    ln = metric(lines, "env:")
    if not ln or "no WC_" in ln:
        return {} if ln else None
    return dict(kv.split("=", 1) for kv in ln[len("env:"):].split() if "=" in kv)


def source_sha(cpp, hpp):
    """Same fingerprint CMake stamps: sha256(sha256(cpp) + sha256(hpp)), first 12 hex."""
    import hashlib
    h = lambda f: hashlib.sha256(open(os.path.join(C.ROOT, f), "rb").read()).hexdigest()
    return hashlib.sha256((h(cpp) + h(hpp)).encode()).hexdigest()[:12]


def grid_numbers(img):
    occ = (img == 0).astype(np.uint8)
    dt = cv2.distanceTransform(occ, cv2.DIST_L2, 5)
    r = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-3)
    fr, un = img == 254, img == 205
    adj = np.zeros_like(un)
    adj[1:] |= un[:-1]; adj[:-1] |= un[1:]; adj[:, 1:] |= un[:, :-1]; adj[:, :-1] |= un[:, 1:]
    return dict(occupied_m2=occ.sum() * RES ** 2, free_m2=fr.sum() * RES ** 2,
                wall_cm=float(np.median((2 * dt[r] - 1) * 2.5)) if r.any() else float("nan"),
                free_unknown_contacts=int((fr & adj).sum()))


def labelled(img_bgr, scale=2, step=50):
    v = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    H, W = v.shape[:2]; M = 30; px = step * scale
    o = cv2.copyMakeBorder(v, M, 0, M, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for k, x in enumerate(range(0, W, px)):
        cv2.line(o, (x + M, M), (x + M, H + M), (200, 160, 60), 1)
        cv2.putText(o, chr(65 + k), (x + M + px // 2 - 6, 21), 0, 0.6, (200, 80, 0), 2)
    for k, y in enumerate(range(0, H, px)):
        cv2.line(o, (M, y + M), (W + M, y + M), (200, 160, 60), 1)
        cv2.putText(o, str(k + 1), (8, y + M + px // 2 + 6), 0, 0.6, (200, 80, 0), 2)
    return o


def heading_offset_deg(run):
    """gt_heading_offset_deg written to the map YAML by the exporter (GT yaw − map heading)."""
    try:
        m = re.search(r"gt_heading_offset_deg:\s*([-0-9.eE]+)", open(os.path.splitext(C.grid_pgm(run))[0] + ".yaml").read())
        return float(m.group(1)) if m else 0.0
    except OSError:
        return 0.0


def ate(db, bag, run):
    f = os.path.join(run, "poses_final.csv")
    if not os.path.exists(f):
        f = os.path.join(run, "poses_manhattan.csv")
    if not (bag and os.path.exists(os.path.join(C.DB_DIR, bag)) and os.path.exists(f)):
        return None
    con = C.connect(db)
    stamps = {i: v[0] for i, v in C.load_nodes(con).items()}
    g = G.load_bag(os.path.join(C.DB_DIR, bag))["gt"].copy()
    g[:, 3] -= np.radians(heading_offset_deg(run))     # remove the ceiling camera's heading bias
    return G.evaluate(C.load_pose_csv(f), stamps, g)["fit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db"); ap.add_argument("default_run"); ap.add_argument("dev_run")
    ap.add_argument("--bag", default=None, help="ground-truth bag in databases/ (default tf_<db name> if present)")
    ap.add_argument("--out", default=None, help="output folder (default: analysis/<db>_frozen_vs_dev_<N> next to the dev run)")
    a = ap.parse_args()
    bag = a.bag or "tf_" + a.db.replace(".db", "")
    names = [os.path.basename(os.path.normpath(r)) for r in (a.default_run, a.dev_run)]
    # <db>_frozen_vs_dev_<N>, next to the dev run (falls back to both folder names for older runs)
    stem = a.db.replace(".db", "")
    m = re.fullmatch(re.escape(stem) + r"_dev_(\d+)", names[1])
    cmp_name = f"{stem}_frozen_vs_dev_{m.group(1)}" if m else f"{stem}_{names[0]}_vs_{names[1]}"
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(os.path.normpath(a.dev_run))), "analysis", cmp_name)
    os.makedirs(out, exist_ok=True)

    pgms = [C.grid_pgm(r) for r in (a.default_run, a.dev_run)]
    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in pgms]
    origins = [read_yaml_origin(p) for p in pgms]
    # common canvas in world coordinates (image row 0 = max y)
    x0 = min(o[0] for o in origins); y0 = min(o[1] for o in origins)
    x1 = max(o[0] + im.shape[1] * RES for o, im in zip(origins, imgs))
    y1 = max(o[1] + im.shape[0] * RES for o, im in zip(origins, imgs))
    W, H = int(round((x1 - x0) / RES)), int(round((y1 - y0) / RES))
    canv = []
    for o, im in zip(origins, imgs):
        c = np.full((H, W), 205, np.uint8)
        cx = int(round((o[0] - x0) / RES)); cy = H - int(round((o[1] - y0) / RES)) - im.shape[0]
        c[cy:cy + im.shape[0], cx:cx + im.shape[1]] = im
        canv.append(c)
    d, v = canv
    ov = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    ov[(v == 0) & (d != 0)] = (0, 170, 0)
    ov[(d == 0) & (v != 0)] = (0, 0, 220)
    tiles = [labelled(cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)), labelled(cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)), labelled(ov)]
    for t, title in zip(tiles, ("DEFAULT " + names[0], "DEV " + names[1], "green = only dev, red = only default")):
        cv2.putText(t, title, (40, t.shape[0] - 8), 0, 0.55, (0, 0, 200), 2)
    sep = np.full((10, tiles[0].shape[1], 3), 128, np.uint8)
    cv2.imwrite(os.path.join(out, "compare.png"), np.vstack([tiles[0], sep, tiles[1], sep, tiles[2]]))

    md, mv = metrics(a.default_run), metrics(a.dev_run)
    nd, nv = grid_numbers(d), grid_numbers(v)
    lines = [f"### {a.db}: default `{names[0]}` vs dev `{names[1]}`", "",
             f"- default: {metric(md, 'pipeline:') or 'pipeline: (not stamped — older build)'}",
             f"- dev:     {metric(mv, 'pipeline:') or 'pipeline: (not stamped)'}", "",
             "| | default | dev |", "|---|---|---|",
             f"| occupied | {nd['occupied_m2']:.2f} m² | {nv['occupied_m2']:.2f} m² |",
             f"| free | {nd['free_m2']:.1f} m² | {nv['free_m2']:.1f} m² |",
             f"| median wall thickness | {nd['wall_cm']:.1f} cm | {nv['wall_cm']:.1f} cm |",
             f"| free cells touching unknown (openings) | {nd['free_unknown_contacts']} | {nv['free_unknown_contacts']} |",
             f"| cells occupied only in dev / only in default | {int(((v == 0) & (d != 0)).sum())} | {int(((d == 0) & (v != 0)).sum())} |"]
    ed, ev = ate(a.db, bag, a.default_run), ate(a.db, bag, a.dev_run)
    if ed and ev:
        lines += [f"| ATE vs ground truth (rmse / median / p90) | {ed['ate_rmse_mm']:.0f} / {ed['ate_median_mm']:.0f} / {ed['ate_p90_mm']:.0f} mm"
                  f" | {ev['ate_rmse_mm']:.0f} / {ev['ate_median_mm']:.0f} / {ev['ate_p90_mm']:.0f} mm |",
                  f"| heading error p90 (GT heading bias removed) | {ed['yaw_err_p90_deg']:.2f}° | {ev['yaw_err_p90_deg']:.2f}° |"]
    # what differs: settings (env) and code
    ed_, ev_ = env_settings(md), env_settings(mv)
    lines += ["", "## Settings (WC_* env) that differ", ""]
    if ed_ is None or ev_ is None:
        lines.append("- not recorded (run built before the 'env:' line was added)")
    else:
        keys = sorted(set(ed_) | set(ev_))
        diffs = [k for k in keys if ed_.get(k) != ev_.get(k)]
        lines += [f"- `{k}`: default {ed_.get(k, '(unset)')} → dev **{ev_.get(k, '(unset)')}**" for k in diffs] or ["- none"]
    code_md, _ = PD.report()
    lines += ["", code_md.strip()]
    m = re.search(r"source sha256 (\w+)", metric(mv, "pipeline:"))
    cur = source_sha("src/wc_database_exporter_dev.cpp", "include/wc_database_exporter_dev.hpp")
    if m and m.group(1) != cur:
        lines.append(f"- NOTE: the dev run was built from different dev source (sha {m.group(1)}) than the current file (sha {cur}); "
                     "the code changes above describe the current file.")
    strip = lambda ln: re.sub(r"output=\S+|total runtime \d+ s|grid name: \S+|source sha256 \w+", "", ln)
    sd = {strip(x) for x in md}
    changed = [x for x in mv if strip(x) not in sd and not x.startswith(("pipeline:", "output="))]
    lines += ["", f"metrics.txt lines that differ in dev ({len(changed)}):", ""] + [f"    {x}" for x in changed[:40]]
    lines += ["", f"overlay: `{os.path.relpath(os.path.join(out, 'compare.png'), C.ROOT)}`"]
    txt = "\n".join(lines) + "\n"
    open(os.path.join(out, "summary.md"), "w").write(txt)
    print(txt)
    print(f"wrote {out}/compare.png and summary.md")


if __name__ == "__main__":
    main()
