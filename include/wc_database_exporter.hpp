#ifndef WC_DATABASE_EXPORTER_HPP
#define WC_DATABASE_EXPORTER_HPP

#include <database_exporter.hpp>
#include <opencv2/opencv.hpp>

class WCDatabaseExporter : public DatabaseExporter {
public:
  using DatabaseExporter::DatabaseExporter;

  ~WCDatabaseExporter() override;

  bool initialize_rtabmap_database() override;
  void assembleSceneFromOptimizedPoses() override;
  void projectAndColorizePointCloud() override;
  void assemble_colored_point_cloud() override;
  void finalize_and_return_result(Result &result) override;

private:
  std::multimap<int, std::pair<int, rtabmap::Transform>>
  load_bag_detections(const rtabmap::ParametersMap &);

  // Per-node scan data: world-frame 3D endpoints + lidar sensor origin XY
  std::map<int, std::vector<cv::Point3f>> per_scan_xyz_;
  std::map<int, cv::Point2f>             per_scan_origin_;

  // Tag world observations and robot poses (post-GTSAM), used for overlay PNG
  std::map<int, std::vector<cv::Point3f>> tag_obs_;
  std::map<int, rtabmap::Transform>       viz_poses_;
};

#endif // WC_DATABASE_EXPORTER_HPP
