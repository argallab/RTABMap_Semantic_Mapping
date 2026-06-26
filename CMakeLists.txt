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
find_package(PkgConfig REQUIRED)
# set(CMAKE_PREFIX_PATH "/app/libtorch-cuda12.2")
find_package(Torch REQUIRED)
find_package(CGAL REQUIRED) 

pkg_check_modules(JSONCPP jsoncpp)

add_executable(database_exporter
    src/main.cpp
    src/lidar_database_exporter.cpp
    src/wc_database_exporter.cpp
    src/database_exporter.cpp
    src/semantic_mapping.cpp
    src/rtabmap_vectorize.cpp
)

# add_executable(vectorize_test
#     src/vectorize_test.cpp
#     src/rtabmap_vectorize.cpp
#     src/graph_vectorize.cpp
# )

include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${pcl_conversions_INCLUDE_DIRS}
    ${RTABMap_INCLUDE_DIRS}
    ${nav2_map_server_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(database_exporter PUBLIC
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${pcl_conversions_LIBRARIES}
    ${nav2_map_server_LIBRARIES}
    ${JSONCPP_LIBRARIES}
    rtabmap::rtabmap
    torch
    )

# target_link_libraries(vectorize_test PUBLIC
#     ${PCL_LIBRARIES}
#     ${OpenCV_LIBRARIES}
#     ${pcl_conversions_LIBRARIES}
#     CGAL::CGAL
#     rtabmap::rtabmap)

install(TARGETS database_exporter 
    DESTINATION ${CMAKE_BINARY_DIR}/lib
)



