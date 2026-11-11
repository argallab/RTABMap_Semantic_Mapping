# GPU dev image for RTABMap_Semantic_Mapping.
# Reproducible replacement for the hand-built `mapping_backup` image / `gpu_map_8gb`
# container (Ubuntu 22.04, CUDA 12.2, ROS 2 Humble, libtorch 2.5.1+cu121).
#
# The repo is NOT copied in: it is bind-mounted at /app (see scripts/gpu_container.sh),
# so the build context is just this folder.
#
#   scripts/gpu_container.sh build    # build the image
#   scripts/gpu_container.sh create   # create + enter the container
#   scripts/gpu_container.sh compile  # cmake/ninja build of the exporters into /app/build

# devel = CUDA 12.2 toolkit (nvcc + dev libs) that libtorch's CMake config needs.
# The old image apt-installed cuda-toolkit-12-1 AND 12-2 (+ Nsight, ~9 GB); only 12.2 is used.
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    QT_X11_NO_MITSHM=1

# --- locale + ROS 2 apt source (official ROS install steps) ---------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends locales software-properties-common curl ca-certificates gnupg2 \
 && locale-gen en_US en_US.UTF-8 \
 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
 && add-apt-repository -y universe \
 && ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | awk -F'"' '{print $4}') \
 && curl -fL -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" \
 && dpkg -i /tmp/ros2-apt-source.deb \
 && rm /tmp/ros2-apt-source.deb \
 && rm -rf /var/lib/apt/lists/*

# --- ROS 2 Humble + build deps ---------------------------------------------------------
# rtabmap-ros pulls RTABMap 0.23 + GTSAM + libpointmatcher from ROS apt; that is what the
# exporters link against (the old image's /usr/local RTABMap 0.22 source build was unused).
RUN apt-get update \
 && apt-get install -y \
      ros-humble-ros-base \
      ros-humble-rtabmap-ros \
      ros-humble-navigation2 \
      ros-humble-pcl-conversions \
      ros-humble-rosbag2-storage-mcap \
      ros-humble-slam-toolbox \
      ros-humble-apriltag-msgs \
      build-essential cmake ninja-build git g++ wget unzip \
      pybind11-dev python3-pip \
      libpcl-dev libboost-all-dev libcgal-dev \
      sqlite3 nano byobu sudo \
 && rm -rf /var/lib/apt/lists/*

# --- libtorch (C++), found by CMake via Torch_ROOT -------------------------------------
ARG LIBTORCH_URL=https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip
RUN wget -q -O /tmp/libtorch.zip "${LIBTORCH_URL}" \
 && unzip -q /tmp/libtorch.zip -d /opt \
 && rm /tmp/libtorch.zip
ENV Torch_ROOT=/opt/libtorch

# --- Python (U-Net IR-dot removal, YOLO via pybind11, analysis scripts) ----------------
# Versions pinned to what ran in gpu_map_8gb. --no-cache-dir: the old image carried 7.4 GB of pip cache.
RUN python3 -m pip install --no-cache-dir \
      torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
      --index-url https://download.pytorch.org/whl/cu121 \
 && python3 -m pip install --no-cache-dir \
      ultralytics==8.3.133 \
      numpy==2.2.6 \
      scipy==1.15.3 \
      scikit-learn==1.7.2 \
      pandas==2.3.3 \
      polars==1.37.1 \
      matplotlib==3.10.8 \
      seaborn==0.13.2 \
      opencv-python==4.13.0.90 \
      pytorch-msssim==1.0.0

RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc

WORKDIR /app
CMD ["bash"]
