#!/usr/bin/env bash
# Run slam_toolbox offline on the bag exported by wc_database_exporter.
#
# Usage:
#   ./run_slam_toolbox.sh <slam_toolbox_input_dir> [map_output_name]
#
# Example:
#   ./run_slam_toolbox.sh output/2026-09-04_15-37-53/slam_toolbox_input \
#                         output/2026-09-04_15-37-53/grid/slam_map
#
# The map is saved as <map_output_name>.pgm / <map_output_name>.yaml
# via the /slam_toolbox/save_map service.

set -e

BAG_DIR="${1}"
MAP_OUT="${2:-slam_map}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARAMS="${SCRIPT_DIR}/../config/slam_toolbox_offline.yaml"

if [[ -z "${BAG_DIR}" ]]; then
  echo "Usage: $0 <slam_toolbox_input_dir> [map_output_name]"
  exit 1
fi

if [[ ! -f "${BAG_DIR}/metadata.yaml" ]]; then
  echo "ERROR: No bag found at ${BAG_DIR} (missing metadata.yaml)"
  exit 1
fi

echo "=== slam_toolbox offline ==="
echo "  Bag:    ${BAG_DIR}"
echo "  Params: ${PARAMS}"
echo "  Output: ${MAP_OUT}.{pgm,yaml}"
echo ""
echo "Starting slam_toolbox node in the background..."

# Launch slam_toolbox in synchronous (offline-friendly) mode.
ros2 run slam_toolbox sync_slam_toolbox_node \
    --ros-args \
    --params-file "${PARAMS}" &
SLAM_PID=$!

echo "Waiting for slam_toolbox to be ready..."
for i in $(seq 1 30); do
  ros2 service list 2>/dev/null | grep -q "/slam_toolbox/save_map" && break
  sleep 1
done

echo "Playing bag (topics: /scan /tf)..."
ros2 bag play "${BAG_DIR}" \
    --topics /scan /tf \
    --rate 1.0 \
    --clock

echo ""
echo "Bag finished. Saving map..."

# Save the map via the slam_toolbox service.
ros2 service call /slam_toolbox/save_map \
    slam_toolbox/srv/SaveMap \
    "{name: {data: '${MAP_OUT}'}}" || true

sleep 2
echo "Saved → ${MAP_OUT}.pgm / ${MAP_OUT}.yaml"

kill ${SLAM_PID} 2>/dev/null || true
echo "Done."
