"""What the DEV pipeline changes in code, compared with the frozen DEFAULT pipeline.

  python3 pipeline_diff.py              # summary + full diff to the terminal
  python3 pipeline_diff.py --md out.md  # also write it as markdown (for the user's notes, e.g. output/R01_02/analysis/R01.md)

Compares src/wc_database_exporter_dev.cpp / include/wc_database_exporter_dev.hpp
with the default files after undoing the renames (class name, header name,
banner), so only real changes show.  Reports:
  * named constants whose value differs (constexpr / const ... kName = value), e.g.
    kMinComponentCells 75 -> 0
  * constants added or removed
  * which functions / pipeline steps the changed lines fall in
  * the unified diff
Runtime settings (WC_* env) are not code: each run writes them to metrics.txt
("env:" line), and compare_pipelines.py shows the ones that differ.
"""
import argparse
import difflib
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = [("src/wc_database_exporter.cpp", "src/wc_database_exporter_dev.cpp"),
         ("include/wc_database_exporter.hpp", "include/wc_database_exporter_dev.hpp")]
CONST = re.compile(r"\b(?:constexpr|const)\s+[\w:<>]+\s+(k\w+)\s*=\s*([^;]+);")


def normalized_dev(text):
    lines = text.splitlines()
    while lines and (lines[0].startswith("// DEV COPY") or lines[0].startswith("// Change THIS")
                     or lines[0].startswith("// (wc_database") or lines[0].startswith("// change and why")
                     or not lines[0].strip()):
        lines.pop(0)
    t = "\n".join(lines) + "\n"
    t = t.replace("WCDatabaseExporterDev", "WCDatabaseExporter")
    t = t.replace("wc_database_exporter_dev.hpp", "wc_database_exporter.hpp")
    t = t.replace("WC_DATABASE_EXPORTER_DEV_HPP", "WC_DATABASE_EXPORTER_HPP")
    return t


def constants(text):
    out = {}
    for ln in text.splitlines():
        for name, val in CONST.findall(ln):
            out.setdefault(name, re.sub(r"\s+", " ", val.strip()))
    return out


def section_of(lines, idx):
    """Nearest '// ── N. title' step header or function definition above line idx."""
    for j in range(idx, -1, -1):
        s = lines[j]
        m = re.match(r"\s*// ── (.+?) ─", s)
        if m:
            return m.group(1).strip()
        m = re.match(r"^[\w:<>,\s\*&]+?\b(WCDatabaseExporter::\w+)\s*\(", s)
        if m:
            return m.group(1)
    return "top of file"


def report():
    md, full = [], []
    any_change = False
    for dpath, vpath in PAIRS:
        d = open(os.path.join(ROOT, dpath)).read()
        v = normalized_dev(open(os.path.join(ROOT, vpath)).read())
        if d == v:
            md.append(f"- `{vpath}`: **no code changes** vs `{dpath}`")
            continue
        any_change = True
        cd, cv = constants(d), constants(v)
        changed = [(k, cd[k], cv[k]) for k in cd if k in cv and cd[k] != cv[k]]
        added = [(k, cv[k]) for k in cv if k not in cd]
        removed = [(k, cd[k]) for k in cd if k not in cv]
        dl, vl = d.splitlines(), v.splitlines()
        sm = difflib.SequenceMatcher(None, dl, vl, autojunk=False)
        hunks = [op for op in sm.get_opcodes() if op[0] != "equal"]
        sections = []
        for tag, i1, i2, j1, j2 in hunks:
            # headings inside the changed dev lines name it best; else the step it sits in
            heads = [re.match(r"\s*// ── (.+?) ─", ln).group(1).strip() for ln in vl[j1:j2] if re.match(r"\s*// ── (.+?) ─", ln)]
            sec = "; ".join(heads) if heads else section_of(vl, j1 if j1 < len(vl) else len(vl) - 1)
            n = max(i2 - i1, j2 - j1)
            sections.append((sec, tag, i2 - i1, j2 - j1))
        md.append(f"- `{vpath}` vs `{dpath}`: {len(hunks)} changed block(s)")
        if changed:
            md.append("  - constants changed: " + "; ".join(f"`{k}` {a} → **{b}**" for k, a, b in changed))
        if added:
            md.append("  - constants added: " + "; ".join(f"`{k}` = {b}" for k, b in added))
        if removed:
            md.append("  - constants removed: " + "; ".join(f"`{k}` (was {a})" for k, a in removed))
        for sec, tag, nd, nv in sections:
            md.append(f"  - in **{sec}**: {tag} ({nd} default line(s) → {nv} dev line(s))")
        full += list(difflib.unified_diff(dl, vl, fromfile=f"default/{dpath}", tofile=f"dev/{vpath} (renames undone)", lineterm="", n=3))
    head = ["## Code changes: dev vs default", ""] + (md if md else ["- none"])
    if not any_change:
        head.append("- The dev pipeline is identical to the default; differences between runs come from WC_* settings only.")
    return "\n".join(head) + "\n", "\n".join(full) + ("\n" if full else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", help="write summary + diff to this markdown file")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    summ, diff = report()
    print(summ)
    if diff and not a.summary_only:
        print(diff)
    if a.md:
        with open(a.md, "w") as f:
            f.write(summ + ("\n```diff\n" + diff + "```\n" if diff else ""))
        print(f"wrote {a.md}")


if __name__ == "__main__":
    main()
