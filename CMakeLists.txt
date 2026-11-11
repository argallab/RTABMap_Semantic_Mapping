cmake_minimum_required(VERSION 3.22)
project(rtabmap_database_export)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PYBIND11_FINDPYTHON ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build." FORCE)
endif()

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_definitions(-DPROJECT_PATH="${CMAKE_SOURCE_DIR}")

# set(CMAKE_PREFIX_PATH "/opt/ros/humble" ${CMAKE_PREFIX_PATH})

# find dependencies
find_package(PCL 1.12 REQUIRED)
find_package(RTABMap REQUIRED)
find_package(nav2_map_server REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(OpenCV 4.5.4 REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Development Interpreter REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(PkgConfig REQUIRED)
# set(CMAKE_PREFIX_PATH "/app/libtorch-cuda12.2")
find_package(Torch REQUIRED)

# Two executables from the same dependencies:
#   database_exporter      DEFAULT pipeline, frozen (src/wc_database_exporter.cpp) — same for every user
#   database_exporter_dev  DEV pipeline, the working copy (src/wc_database_exporter_dev.cpp)
# Each run writes "pipeline: <name> (source sha256 <hash>)" to metrics.txt; the
# hash covers that pipeline's exporter source + header, so a changed default
# pipeline shows up as a different hash.
set(EXPORTER_COMMON_SRC
    src/database_exporter.cpp
    src/semantic_mapping.cpp
)
add_executable(database_exporter     src/main.cpp     src/wc_database_exporter.cpp     ${EXPORTER_COMMON_SRC})
add_executable(database_exporter_dev src/main_dev.cpp src/wc_database_exporter_dev.cpp ${EXPORTER_COMMON_SRC})

file(SHA256 ${CMAKE_SOURCE_DIR}/src/wc_database_exporter.cpp      SHA_DEFAULT_SRC)
file(SHA256 ${CMAKE_SOURCE_DIR}/include/wc_database_exporter.hpp  SHA_DEFAULT_HDR)
file(SHA256 ${CMAKE_SOURCE_DIR}/src/wc_database_exporter_dev.cpp     SHA_DEV_SRC)
file(SHA256 ${CMAKE_SOURCE_DIR}/include/wc_database_exporter_dev.hpp SHA_DEV_HDR)
string(SHA256 SHA_DEFAULT "${SHA_DEFAULT_SRC}${SHA_DEFAULT_HDR}")
string(SHA256 SHA_DEV     "${SHA_DEV_SRC}${SHA_DEV_HDR}")
string(SUBSTRING ${SHA_DEFAULT} 0 12 SHA_DEFAULT)
string(SUBSTRING ${SHA_DEV}     0 12 SHA_DEV)
# re-run CMake (and re-stamp the hash) whenever either exporter source changes
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    src/wc_database_exporter.cpp include/wc_database_exporter.hpp
    src/wc_database_exporter_dev.cpp include/wc_database_exporter_dev.hpp)
target_compile_definitions(database_exporter     PRIVATE PIPELINE_NAME="default" PIPELINE_SRC_SHA256="${SHA_DEFAULT}")
target_compile_definitions(database_exporter_dev PRIVATE PIPELINE_NAME="dev"     PIPELINE_SRC_SHA256="${SHA_DEV}")

include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${pcl_conversions_INCLUDE_DIRS}
    ${RTABMap_INCLUDE_DIRS}
    ${nav2_map_server_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${rclcpp_INCLUDE_DIRS}
    ${rosbag2_cpp_INCLUDE_DIRS}
    ${rosbag2_storage_INCLUDE_DIRS}
    ${tf2_msgs_INCLUDE_DIRS}
    ${geometry_msgs_INCLUDE_DIRS}
    ${nav_msgs_INCLUDE_DIRS}
    ${sensor_msgs_INCLUDE_DIRS}
)

foreach(exe database_exporter database_exporter_dev)
  target_link_libraries(${exe} PUBLIC
      ${PCL_LIBRARIES}
      ${OpenCV_LIBRARIES}
      ${pcl_conversions_LIBRARIES}
      ${nav2_map_server_LIBRARIES}
      rtabmap::rtabmap
      torch
      ${rclcpp_LIBRARIES}
      ${rosbag2_cpp_LIBRARIES}
      ${rosbag2_storage_LIBRARIES}
      tf2_msgs::tf2_msgs__rosidl_typesupport_cpp
      geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
      nav_msgs::nav_msgs__rosidl_typesupport_cpp
      sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
      )
endforeach()

install(TARGETS database_exporter database_exporter_dev
    DESTINATION ${CMAKE_BINARY_DIR}/lib
)



