#ifndef WC_DATABASE_EXPORTER_HPP
#define WC_DATABASE_EXPORTER_HPP

#include <database_exporter.hpp>

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
  point_cloud_to_occupancy_grid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

  // @brief Filter the point cloud using statistical and radius outlier removal
  // @param cloud The point cloud
  // @return The filtered point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
  filter_point_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

  // @brief Project the point cloud onto the camera image, and keep track of
  // which points are in each image
  // @param image_size The size of the image
  // @param camera_matrix The camera matrix
  // @param cloud The point cloud
  // @param camera_transform The transform of the camera
  // @return The input image overlayed with the point cloud projection
  std::pair<cv::Mat, std::map<std::pair<int, int>, int>>
  project_cloud_to_camera(const cv::Size &image_size,
                          const cv::Mat &camera_matrix,
                          const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                          const rtabmap::Transform &camera_transform);

  // @brief Convert a point cloud to an occupancy grid
  // @param cloud The point cloud
  // @return The occupancy grid

  pcl::PointCloud<pcl::PointXYZI>::Ptr assembledCloud{
    new pcl::PointCloud<pcl::PointXYZI>};

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudToExport{
    new pcl::PointCloud<pcl::PointXYZRGBNormal>};
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloudIToExport{
    new pcl::PointCloud<pcl::PointXYZINormal>};

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr rtabmap_cloud_{
    new pcl::PointCloud<pcl::PointXYZRGB>};
};

#endif // WC_DATABASE_EXPORTER_HPP