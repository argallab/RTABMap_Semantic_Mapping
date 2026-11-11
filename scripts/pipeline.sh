#!/usr/bin/env bash
# Full mapping pipeline: exporter (GTSAM + raytrace) → slam_toolbox offline
#
# Usage:
#   ./scripts/pipeline.sh <db_name> [model_name]
#
# <db_name> is just the filename (no path) — the exporter looks in /app/databases/.
#
# Example (inside Docker):
#   ./scripts/pipeline.sh keep.db

set -eo pipefail

source /opt/ros/humble/setup.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARAMS="${SCRIPT_DIR}/../config/slam_toolbox_offline.yaml"
EXPORTER="/app/build/${EXPORTER_BIN:-database_exporter}"   # EXPORTER_BIN=database_exporter_dev for the dev pipeline

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <db_name> [model_name]   (db_name is just the filename, e.g. keep.db)"
  exit 1
fi

# ── Step 1: exporter ──────────────────────────────────────────────────────────
EXPORTER_LOG=$(mktemp)
"$EXPORTER" "$@" 2>&1 | tee "$EXPORTER_LOG"

BAG_DIR=$(grep -oP '\[slam_toolbox\] Exported.*→ \K.*' "$EXPORTER_LOG" | tail -1)
rm -f "$EXPORTER_LOG"

if [[ -z "$BAG_DIR" || ! -d "$BAG_DIR" ]]; then
  echo "[pipeline] No slam_toolbox bag found — exiting after exporter"
  exit 0
fi

GRID_DIR="$(dirname "$BAG_DIR")/grid"
MAP_OUT="$GRID_DIR/slam_map"

# ── Step 2: slam_toolbox ──────────────────────────────────────────────────────
echo ""
echo "=== slam_toolbox offline =============================================="
echo "  Bag : $BAG_DIR"
echo "  Map : $MAP_OUT.{pgm,yaml}"
echo "======================================================================"

ros2 run slam_toolbox sync_slam_toolbox_node \
    --ros-args --params-file "$PARAMS" &
SLAM_PID=$!

echo "[pipeline] Waiting for slam_toolbox..."
for i in $(seq 1 30); do
  ros2 service list 2>/dev/null | grep -q "/slam_toolbox/save_map" && break
  sleep 1
done

ros2 bag play "$BAG_DIR" \
    --topics /scan /tf \
    --rate 5.0 \
    --clock

sleep 3

echo "[pipeline] Saving map → $MAP_OUT"
ros2 service call /slam_toolbox/save_map \
    slam_toolbox/srv/SaveMap \
    "{name: {data: '$MAP_OUT'}}" || echo "[pipeline] WARNING: save_map failed"

sleep 2
kill "$SLAM_PID" 2>/dev/null || true

echo ""
echo "[pipeline] Done."
echo "  Raytrace : $GRID_DIR/$(basename "$(dirname "$BAG_DIR")").pgm"
echo "  slam_map : $MAP_OUT.pgm"
