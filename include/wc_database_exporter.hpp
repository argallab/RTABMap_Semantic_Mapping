#ifndef WC_DATABASE_EXPORTER_HPP
#define WC_DATABASE_EXPORTER_HPP

#include <database_exporter.hpp>

#include <opencv2/opencv.hpp>
#include <memory>

#include "torch/script.h"
#include <rtabmap/core/CameraModel.h>

class WCDatabaseExporter : public DatabaseExporter {
public:
  using DatabaseExporter::DatabaseExporter;

  ~WCDatabaseExporter() override;

  void assembleSceneFromOptimizedPoses() override;
  void projectAndColorizePointCloud() override;
  void assemble_colored_point_cloud() override;
  void finalize_and_return_result(Result &result) override;
  bool initialize_rtabmap_database() override;

private:
  nav_msgs::msg::OccupancyGrid::SharedPtr
  point_cloud_to_occupancy_grid(
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
      const std::vector<std::pair<float,float>> &scan_origins);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
  filter_point_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

  std::pair<cv::Mat, std::map<std::pair<int, int>, int>>
  project_cloud_to_camera(const cv::Size &image_size,
                          const cv::Mat &camera_matrix,
                          const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                          const rtabmap::Transform &camera_transform);

  // Runs the U-Net on a grayscale image in original camera orientation
  // (rows=360, cols=640) and returns the cleaned grayscale image of the same size.
  cv::Mat run_unet(const cv::Mat &gray_img);

  pcl::PointCloud<pcl::PointXYZI>::Ptr assembledCloud{
    new pcl::PointCloud<pcl::PointXYZI>};

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudToExport{
    new pcl::PointCloud<pcl::PointXYZRGBNormal>};
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloudIToExport{
    new pcl::PointCloud<pcl::PointXYZINormal>};

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr rtabmap_cloud_{
    new pcl::PointCloud<pcl::PointXYZRGB>};

  // Scan-based flat cloud built from lidar data — used as a cleaner alternative
  // to the depth image cloud for 2D occupancy grid generation.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr scan_cloud_{
    new pcl::PointCloud<pcl::PointXYZRGB>};

  torch::jit::script::Module unet_model_;
  bool unet_loaded_ = false;

  // Cleaned combined (left|right) grayscale images keyed by node ID,
  // populated during re-SLAM so downstream stages don't re-run U-Net.
  std::map<int, cv::Mat> cleaned_combined_cache_;

  // Tag world positions and camera poses stored for the destructor overlay.
  std::map<int, std::vector<cv::Point3f>> tag_obs_;
  std::map<int, rtabmap::Transform> viz_poses_;

  // Per-node scan data for log-odds occupancy grid (populated in assembleScene).
  // per_scan_xyz_    : all scan endpoints (x,y,z) in world frame — z used to
  //                    gate occupied votes (wall band only); free rays cast for all
  // per_scan_origin_ : lidar sensor XY origin in world frame, keyed by node_id
  std::map<int, std::vector<cv::Point3f>> per_scan_xyz_;
  std::map<int, cv::Point2f>              per_scan_origin_;

  // Reads AprilTag detections from the bag file alongside the DB (if present)
  // and returns additional camera→tag observations keyed by DB node ID.
  // Each value is {physical_tag_id, camera_to_tag Transform}.
  std::multimap<int, std::pair<int, rtabmap::Transform>>
  load_bag_detections(const rtabmap::ParametersMap &);

  nav_msgs::msg::OccupancyGrid::SharedPtr
  per_scan_log_odds_grid(float resolution = 0.05f);

  void edge_detection();
  void build_map_from_edges();
  std::vector<cv::Mat> cleaned_imgs;
  std::vector<cv::Mat> edge_imgs;
  std::vector<cv::Mat> depth_edge_imgs_; // depth-only Canny, used for wall map
  // One entry per cleaned_imgs frame — camera model with intrinsics and
  // localTransform used for direct depth back-projection in build_map_from_edges().
  std::vector<rtabmap::CameraModel> frame_camera_models_;
};

#endif // WC_DATABASE_EXPORTER_HPP
