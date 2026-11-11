#!/usr/bin/env bash
# Build / create / compile the GPU dev container (replaces the hand-built gpu_map_8gb).
#
#   scripts/gpu_container.sh build     build the image from docker/Dockerfile
#   scripts/gpu_container.sh create    create the container and enter it
#   scripts/gpu_container.sh shell     start (if stopped) and enter the existing container
#   scripts/gpu_container.sh compile   build database_exporter + database_exporter_dev into /app/build
#
# Settings (environment variables, read by `create`; mounts are fixed once the container exists,
# so to change them:  docker rm <name>  then  create  again; nothing is lost, the code and data
# live on the host):
#   IMAGE=rtabmap_semantic_gpu   image name
#   NAME=gpu_map_8gb             container name
#   DB_DIR=/path/to/databases    mount this host folder as /app/databases (default: the repo's databases/)
#   OUTPUT_DIR=/path/to/output   mount this host folder as /app/output    (default: the repo's output/)
#   SHM_SIZE=8g                  /dev/shm size; only PyTorch DataLoader workers (U-Net training) need it
#   DOCKER_ARGS="..."            any extra `docker run` flags, e.g. DOCKER_ARGS="-v /mnt/bags:/bags"
#
# GUI windows: an X auth cookie is generated for the container, so `xhost +local:` is not needed.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-rtabmap_semantic_gpu}"
NAME="${NAME:-gpu_map_8gb}"
SHM_SIZE="${SHM_SIZE:-8g}"
XAUTH_DIR="$HOME/.cache/${NAME}-xauth"   # bind-mounted as a folder so a refreshed cookie is seen

# Write an X cookie for the current $DISPLAY that is valid from any hostname ("ffff" family),
# so root inside the container is authorised without opening the display to all local users.
# Refreshed on every create/shell/compile because the desktop session makes a new one at each login.
refresh_xauth() {
  mkdir -p "$XAUTH_DIR"; chmod 700 "$XAUTH_DIR"
  local f="$XAUTH_DIR/Xauthority"
  rm -f "$f"; touch "$f"; chmod 600 "$f"
  if [[ -n "${DISPLAY:-}" ]] && command -v xauth >/dev/null; then
    xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -q -f "$f" nmerge - 2>/dev/null || true
  fi
}

case "${1:-}" in
  build)
    docker build -t "$IMAGE" "$REPO/docker"
    ;;
  create)
    refresh_xauth
    mounts=(-v "$REPO":/app)
    [[ -n "${DB_DIR:-}" ]]     && mounts+=(-v "$(realpath "$DB_DIR")":/app/databases)
    [[ -n "${OUTPUT_DIR:-}" ]] && mounts+=(-v "$(realpath "$OUTPUT_DIR")":/app/output)
    # shellcheck disable=SC2086  # DOCKER_ARGS is meant to word-split
    docker run -it --name "$NAME" \
      --gpus all \
      --network host \
      --shm-size="$SHM_SIZE" \
      -e DISPLAY="${DISPLAY:-:0}" \
      -e QT_X11_NO_MITSHM=1 \
      -e XAUTHORITY=/tmp/.xauth/Xauthority \
      -v "$XAUTH_DIR":/tmp/.xauth:ro \
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
      "${mounts[@]}" \
      ${DOCKER_ARGS:-} \
      "$IMAGE" bash
    ;;
  shell)
    refresh_xauth
    docker start "$NAME" >/dev/null
    # pass the current DISPLAY (it can change between logins, e.g. :0 vs :1)
    docker exec -it -e DISPLAY="${DISPLAY:-:0}" "$NAME" bash
    ;;
  compile)
    refresh_xauth
    docker start "$NAME" >/dev/null
    docker exec "$NAME" bash -c '
      source /opt/ros/humble/setup.bash &&
      cmake -S /app -B /app/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch &&
      cmake --build /app/build'
    ;;
  *)
    sed -n '2,20p' "$0"; exit 1
    ;;
esac
