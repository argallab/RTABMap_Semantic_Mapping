#pragma once

#include <json/json.h>
#include <map>
#include <opencv2/opencv.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/search/kdtree.h>
#include <vector>

#include <pcl/features/normal_3d.h>     // for pcl::NormalEstimation
#include <pcl/common/angles.h>          // for pcl::deg2rad        // for pcl::Normal
#include <pcl/segmentation/region_growing.h> // for pcl::RegionGrowing

// #include <geometry_msgs/Point.h>

// Vector primitive structures
// struct Line {
//   geometry_msgs::Point start;
//   geometry_msgs::Point end;

//   Line(geometry_msgs::Point s, geometry_msgs::Point e);
// };

struct Line2D {
  cv::Point2f start;
  cv::Point2f end;
  float confidence;
  int plane_id;

  Line2D(cv::Point2f s, cv::Point2f e, float conf = 1.0f, int pid = -1);
};

struct VectorMap {
  std::vector<Line2D> lines;
  cv::Rect2f bounds;
  std::string timestamp;

  VectorMap();
};

struct Plane3D {
  pcl::ModelCoefficients::Ptr coefficients;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr points;
  int id;

  Plane3D();
};

class RTABMapVectorizer {
public:
  RTABMapVectorizer();
  RTABMapVectorizer(float plane_distance_thresh, int plane_max_iter,
                    int plane_min_points, float line_distance_thresh,
                    int line_max_iter, int line_min_points,
                    float merge_distance_thresh, float merge_angle_thresh);

  VectorMap vectorize_point_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                  const std::string &timestamp = "");

private:
  // Parameters for plane segmentation
  float plane_distance_threshold_;
  int plane_max_iterations_;
  int plane_min_points_;

  // Parameters for line extraction
  float line_distance_threshold_;
  int line_max_iterations_;
  unsigned long int line_min_points_;

  // Parameters for line merging
  float merge_distance_threshold_;
  float merge_angle_threshold_;

  void export_to_svg(const VectorMap &map, const std::string &filename);

  cv::Rect2f calculate_bounds(const std::vector<Line2D> &lines);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
  filter_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

  std::vector<Line2D> merge_lines(const std::vector<Line2D> &lines);

  bool are_lines_similar(const Line2D &line1, const Line2D &line2);

  Line2D merge_line_group(const std::vector<Line2D> &lines);

  double calculate_line_length(const Line2D &line);

  std::vector<Line2D> filter_short_lines(const std::vector<Line2D> &lines,
                                         float min_length);
    std::vector<Plane3D> segment_planes(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
    std::vector<cv::Point2f> project_plane_to_2d(const Plane3D &plane);
    std::vector<Line2D> extract_lines_from_2d_points(
        const std::vector<cv::Point2f> &points, int plane_id);
};
