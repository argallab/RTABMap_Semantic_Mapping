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

  void export_slam_toolbox_bag(const std::string &path);

  // Crop + 2D outlier-filter every node's depth-projected scan into
  // per_scan_base_ (base_link frame).
  void clean_scans();

  // Scan-to-local-map ICP: registers each query node's cleaned scan against a
  // submap of spatially nearby (but path-distant) nodes, adds the accepted
  // registrations as kLocalSpaceClosure links, and re-runs the graph
  // optimization (robust) on top of the tag graph in `links`.
  // tag_poses: current negative-ID tag poses; updated in place.
  void refine_poses_with_local_map_icp(const rtabmap::ParametersMap &parameters,
                                       std::map<int, rtabmap::Transform> &tag_poses);

  // Wall crispness of the cleaned scans placed with `poses`.  Patches are
  // classified as wall (tall, long, straight) using only scans within ±ref
  // nodes in time, then the robust spread across that wall (mm) is measured
  // with scans from three windows: ref (≈ noise floor), local, and global
  // (all visits).  global − local is the between-visit misalignment.
  struct WallThickness {
    double ref_median_mm = 0, ref_p90_mm = 0;
    double local_median_mm = 0, local_p90_mm = 0;
    double global_median_mm = 0, global_p90_mm = 0;
    int n = 0;
  };
  WallThickness wall_thickness(const std::map<int, rtabmap::Transform> &poses) const;
  // Computes wall_thickness(poses), prints it and adds it as a metric line.
  void report_wall_thickness(const std::string &label,
                             const std::map<int, rtabmap::Transform> &poses);

  // printf-style: collects one line for the copy-paste METRICS block that is
  // printed (and saved to metrics.txt) at the end of the run.
  void add_metric(const char *fmt, ...);
  std::vector<std::string> metrics_;
  UTimer run_timer_;

  // RTABMap's own graph poses (before tag re-optimization), for the baseline
  // wall-thickness metric.
  std::map<int, rtabmap::Transform> rtabmap_graph_poses_;

  // Cleaned scan points in base_link frame, and each scan's localTransform.
  // Consumed (moved into per_scan_xyz_) by assembleSceneFromOptimizedPoses.
  std::map<int, std::vector<cv::Point3f>> per_scan_base_;
  std::map<int, rtabmap::Transform>       scan_local_tf_;

  // Per-node scan data: cleaned world-frame 3D endpoints (depth-projected
  // scan) + scan origin XY
  std::map<int, std::vector<cv::Point3f>> per_scan_xyz_;
  std::map<int, cv::Point2f>             per_scan_origin_;

  // Tag world observations and robot poses (post-GTSAM), used for overlay PNG
  std::map<int, std::vector<cv::Point3f>> tag_obs_;
  std::map<int, rtabmap::Transform>       viz_poses_;
};

#endif // WC_DATABASE_EXPORTER_HPP
