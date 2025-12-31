#!/bin/bash
set -e

# Source the ROS2 environment
source /opt/ros/humble/setup.bash

# Go to your workspace directory
cd /ros2_ws

# Build user_packages if it exists
USER_PKG_DIR="/ros2_ws/src/user_packages/"
ALL_PACKAGES=$(colcon list)
USER_PACKAGE_PATHS=$(find "$USER_PKG_DIR" -mindepth 1 -maxdepth 3 -type f -name 'package.xml' -exec dirname {} \;)
PACKAGES_TO_BUILD=()

for pkg_path in $USER_PACKAGE_PATHS; do
  pkg_name=$(basename "$pkg_path")
  if echo "$ALL_PACKAGES" | grep -q "$pkg_name"; then
    PACKAGES_TO_BUILD+=("$pkg_name")
  fi
done

echo "Packages to build: ${PACKAGES_TO_BUILD[@]}"

if [ ${#PACKAGES_TO_BUILD[@]} -gt 0 ]; then
  colcon build --packages-select ${PACKAGES_TO_BUILD[@]}
else
  echo "No user packages found to build."
fi

# Source the workspace setup file if it exists
if [ -f "/ros2_ws/install/setup.bash" ]; then
  source /ros2_ws/install/setup.bash
fi

exec "$@"