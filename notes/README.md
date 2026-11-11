# Per-user pipeline notes

Two exporters are built from this repo:

| Executable | Source | Role |
|---|---|---|
| `database_exporter` | `src/wc_database_exporter.cpp` | **Default pipeline, frozen.** Run it unchanged for every user. |
| `database_exporter_dev` | `src/wc_database_exporter_dev.cpp` | **Dev pipeline.** Change this copy to try improvements. |

Every run writes `pipeline: default|dev (source sha256 <hash>)` to `metrics.txt`.
The default hash only changes if the frozen source changes; record it below so
any change to the default pipeline is visible. Everything for a recording goes in `output/<db>/`: frozen runs in `<db>_frozen` (reused on every run), dev runs in
`<db>_dev_<N>` (N counts up per database), and in `output/<db>/analysis/` the comparisons `<db>_frozen_vs_dev_<N>`, `<db>_tag_ablation` and the user's notes.
Each run folder has `run_info.txt` with the start time.

Pick the executable in Docker with `EXPORTER_BIN` (`entrypoint.sh`, `scripts/pipeline.sh`):

```
docker run ... <image> R01_02.db                                         # default
docker run -e EXPORTER_BIN=database_exporter_dev ... <image> R01_02.db   # dev
# or inside the container
/app/build/database_exporter R01_02.db
/app/build/database_exporter_dev R01_02.db
```

## Workflow per user

1. Run the **default** pipeline on the user's recording(s). Note the run folder.
2. Look at the map and list what's wrong (grid squares A..K / 1..n from
   `analysis/compare_pipelines.py` or the labelled grid images).
3. Change `src/wc_database_exporter_dev.cpp` (or only env settings), rebuild, run the **dev** pipeline.
4. Compare:
   ```
   cd analysis
   python3 compare_pipelines.py R01_02.db ../output/R01_02/R01_02_frozen ../output/R01_02/R01_02_dev_3
   ```
   Paste `summary.md` into the user's notes and keep `compare.png`. It lists what
   differs between the two runs:
   - **settings:** the `WC_*` env each run used (every run writes an `env:` line to `metrics.txt`),
     e.g. `WC_MANHATTAN: default (unset) → dev 0`;
   - **code:** what `src/wc_database_exporter_dev.cpp` changes vs the default, with changed
     constants by name (e.g. `kMinComponentCells 75 → 0`) and the pipeline step they're in.
     For the full diff on its own: `python3 pipeline_diff.py` (or `--md file.md`).
5. In the user's notes file, record **what was wrong, what you changed, and why**,
   so it can be decided later whether a change should go into the default pipeline
   (which then becomes a new frozen version for everyone).

Start each user's notes from `TEMPLATE.md` and keep them with the runs, e.g. `output/R01_02/analysis/R01.md`.

## Frozen default pipeline

| Frozen on | Default source sha256 (from metrics.txt) | Notes |
|---|---|---|
| 2026-10-29 | `8da21f5b8be3` (first run: output/R01_02/R01_02_frozen, R01_02) | percentile height veto, wall thinning + automatic ledge detection, first node = map origin, no grid rotation, whole-drive ground-truth fit in the map YAML |
| 2026-10-30 | `076f239050cf` *(expected; confirm from the first run)* | same behaviour as `8da21f5b8be3`; only the output folder changed to `output/<db>/<db>_frozen` |
