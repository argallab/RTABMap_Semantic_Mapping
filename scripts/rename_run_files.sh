#!/usr/bin/env bash
# Rename the files inside exporter run folders to match the folder's name, after the
# folder itself was renamed (e.g. output/R01_02/2026-10-29_10-37-35_dev -> R01_02_dev_1):
#   grid/<old>.pgm, .yaml (and its "image:" line), _tags.png  ->  grid/<folder>.*
#   cloud/<old>.pcd                                            ->  cloud/<folder>.pcd
# and write run_info.txt (start time from the old timestamp name) if it's missing.
# Folders without grid/ are skipped.  Run where you can write the folders
# (they are root-owned when the container made them):
#
#   docker exec gpu_map_8gb bash /app/scripts/rename_run_files.sh /app/output/R01_02/R01_02_*
set -euo pipefail

for run in "$@"; do
  [[ -d "$run/grid" ]] || { echo "skip $run (no grid/)"; continue; }
  new=$(basename "$run")
  pgm=$(ls "$run"/grid/*.pgm 2>/dev/null | head -1)
  if [[ -n "$pgm" ]]; then
    old=$(basename "$pgm" .pgm)
    if [[ "$old" != "$new" ]]; then
      mv "$run/grid/$old.pgm" "$run/grid/$new.pgm"
      [[ -f "$run/grid/$old.yaml" ]] && mv "$run/grid/$old.yaml" "$run/grid/$new.yaml" \
        && sed -i "s#^image: .*#image: $new.pgm#" "$run/grid/$new.yaml"
      [[ -f "$run/grid/${old}_tags.png" ]] && mv "$run/grid/${old}_tags.png" "$run/grid/${new}_tags.png"
      echo "$run: grid $old -> $new"
    fi
  fi
  pcd=$(ls "$run"/cloud/*.pcd 2>/dev/null | head -1)
  oldc=""
  if [[ -n "$pcd" ]]; then
    oldc=$(basename "$pcd" .pcd)
    if [[ "$oldc" != "$new" ]]; then
      mv "$pcd" "$run/cloud/$new.pcd"
      echo "$run: cloud $oldc -> $new"
    fi
  fi
  if [[ ! -f "$run/run_info.txt" ]]; then
    # the old names were YYYY-MM-DD_HH-MM-SS[_dev]; take the start time from whichever had it
    ts=$(printf '%s\n' "${old:-}" "$oldc" | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}' | head -1 || true)
    db=$(grep -oE '^db=[^ ]+' "$run/metrics.txt" 2>/dev/null | head -1 | cut -d= -f2 || true)
    printf 'started: %s\ndatabase: %s\noutput: %s\n' "${ts:-unknown}" "${db:-unknown}" "$new" > "$run/run_info.txt"
    echo "$run: wrote run_info.txt (started ${ts:-unknown})"
  fi
done
