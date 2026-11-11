#!/usr/bin/env bash
# Build / create / compile the GPU dev container (replaces the hand-built gpu_map_8gb).
#
#   scripts/gpu_container.sh build     build the image from docker/Dockerfile
#   scripts/gpu_container.sh create    create the container (same settings as gpu_map_8gb) and enter it
#   scripts/gpu_container.sh shell     start (if stopped) and enter the existing container
#   scripts/gpu_container.sh compile   build database_exporter + database_exporter_dev into /app/build
#
# Override names with IMAGE=... NAME=... ; requires the NVIDIA Container Toolkit on the host.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-rtabmap_semantic_gpu}"
NAME="${NAME:-gpu_map_8gb}"

case "${1:-}" in
  build)
    docker build -t "$IMAGE" "$REPO/docker"
    ;;
  create)
    # Same run settings as the old container: all GPUs, host network, 8 GB /dev/shm,
    # repo bind-mounted at /app, X11 socket for GUI windows.
    # For GUI windows you may also need on the host:  xhost +local:
    docker run -it --name "$NAME" \
      --gpus all \
      --network host \
      --shm-size=8g \
      -e DISPLAY="${DISPLAY:-:0}" \
      -e QT_X11_NO_MITSHM=1 \
      -v "$REPO":/app \
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
      "$IMAGE" bash
    ;;
  shell)
    docker start "$NAME" >/dev/null
    docker exec -it "$NAME" bash
    ;;
  compile)
    docker start "$NAME" >/dev/null
    docker exec "$NAME" bash -c '
      source /opt/ros/humble/setup.bash &&
      cmake -S /app -B /app/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch &&
      cmake --build /app/build'
    ;;
  *)
    sed -n '2,9p' "$0"; exit 1
    ;;
esac
