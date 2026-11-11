# Standalone analysis tools — separate from the pipeline build.
#   cd /app/analysis/tools && cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build
cmake_minimum_required(VERSION 3.22)
project(analysis_tools CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL 1.12 REQUIRED)
find_package(RTABMap REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(icp_probe icp_probe.cpp)
target_include_directories(icp_probe PRIVATE ${PCL_INCLUDE_DIRS} ${RTABMap_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(icp_probe PRIVATE ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} rtabmap::rtabmap)

add_executable(tag_holdout tag_holdout.cpp)
target_include_directories(tag_holdout PRIVATE ${PCL_INCLUDE_DIRS} ${RTABMap_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tag_holdout PRIVATE ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} rtabmap::rtabmap)
