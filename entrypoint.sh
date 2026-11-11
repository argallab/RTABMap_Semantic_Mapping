#!/bin/bash

# Source the ROS 2 environment
source /opt/ros/humble/setup.bash

# Execute the passed command (arguments are passed via "$@")
# EXPORTER_BIN=database_exporter_dev runs the dev pipeline (default: the frozen one)
exec /app/build/${EXPORTER_BIN:-database_exporter} "$@"
