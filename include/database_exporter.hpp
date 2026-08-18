#ifndef DATABASE_EXPORTER_H
#define DATABASE_EXPORTER_H

#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/ProgressState.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/util2d.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/core/util3d_filtering.h>
#include <rtabmap/core/util3d_surface.h>
#include <rtabmap/utilite/UFile.h>
#include <rtabmap/utilite/UStl.h>
#include <rtabmap/utilite/UTimer.h>

#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nav2_map_server/map_io.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "pcl/io/pcd_io.h"
#include <pcl/cloud_iterator.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <unordered_set>

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>

namespace py = pybind11;

struct MouseData {
  cv::Mat image;
  cv::Point start_point;
  cv::Point end_point;
  cv::Rect bounding_box;
  bool drawing = false;
  bool finished = false;
};

struct Object {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  pcl::PointXYZ centroid;
  std::string label;
  float confidence;
};

struct Result {
  bool success = false;
  std::string timestamp = "";
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  std::vector<std::tuple<cv::Mat, cv::Mat, rtabmap::Transform,
                         std::map<std::pair<int, int>, int>>>
    mapping_data;
};

struct BoundingBox {
  int x1, y1, x2, y2;
  BoundingBox(int x1, int y1, int x2, int y2) : x1(x1), y1(y1), x2(x2), y2(y2)
  {
  }
};

class DatabaseExporter {
public:
  DatabaseExporter(std::string rtabmap_database_name, std::string model_name);
  virtual ~DatabaseExporter();

  // @brief Load the rtabmap database
  // @return The result of the operation
  Result load_rtabmap_db();

  virtual bool initialize_rtabmap_database() = 0;
  virtual void assembleSceneFromOptimizedPoses() = 0;
  virtual void projectAndColorizePointCloud() = 0;
  virtual void assemble_colored_point_cloud() = 0;
  virtual void finalize_and_return_result(Result &result) = 0;

  // void RANSAC();

  // @brief Generate a timestamp string
  // @return The timestamp string in the format %Y-%m-%d_%H-%M-%S
  std::string generate_timestamp_string();

protected:
  cv::dnn::Net net_;
  nav_msgs::msg::OccupancyGrid::SharedPtr rtabmap_occupancy_grid_;

  std::string rtabmap_database_path_;
  std::string model_path_;
  std::string timestamp_;
  std::vector<cv::Mat> images_;
  std::vector<std::vector<rtabmap::CameraModel>> camera_models_;
  std::vector<std::vector<rtabmap::StereoCameraModel>> stereo_models_;

  // rgb image, depth image, transform from camera to world, pixel to point map
  std::vector<std::tuple<cv::Mat, cv::Mat, rtabmap::Transform,
                         std::map<std::pair<int, int>, int>>>
    mapping_data_;

  bool export_images_;

  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_int_distribution<> dis_;

  std::map<int, rtabmap::Signature> nodes;
  std::map<int, rtabmap::Transform> optimizedPoses;
  std::multimap<int, rtabmap::Link> links;

  std::map<int, rtabmap::Transform> robotPoses;
  std::vector<std::map<int, rtabmap::Transform>> cameraPoses;
  std::map<int, rtabmap::Transform> scanPoses;

  std::map<int, double> cameraStamps;
  std::map<int, std::vector<rtabmap::CameraModel>> cameraModels;
  std::map<int, cv::Mat> cameraDepths;

  std::vector<int> rawViewpointIndices;
  std::map<int, rtabmap::Transform> rawViewpoints;
  std::map<int, cv::Mat> depth_images;
  std::map<int, cv::Mat> rgb_images;

  std::vector<std::pair<std::pair<int, int>, pcl::PointXY>> pointToPixel;
};

#endif // DATABASE_EXPORTER_H