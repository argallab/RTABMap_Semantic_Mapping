#ifndef SEMANTIC_MAPPING_HPP
#define SEMANTIC_MAPPING_HPP

#include <database_exporter.hpp>
#include <pybind11/iostream.h>

// @brief Convert a cv::Mat to a numpy array
// @param mat The cv::Mat
// @return The numpy array
py::array mat_to_numpy(const cv::Mat &mat);

// @brief: Calculate the centroid of a given point cloud
// @param cloud: The point cloud to calculate the centroid of
// @return: The centroid of the point cloud
pcl::PointXYZ calculate_centroid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

// @brief: Figure out which points in a given point cloud belong to the object
// within a given bounding box in an image frame using a map from points in the
// point cloud to pixel coordinates in the image frame
// @param bounding_box: The bounding box of the object in the image frame
// @param pixel_to_point_map: The map from pixel coordinates to points in the
// @param cloud: The point cloud
// @param pose: The pose of the camera in the world frame
pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud_from_bounding_box(
  std::tuple<std::string, float, BoundingBox> bounding_box,
  std::map<std::pair<int, int>, int> pixel_to_point_map,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, rtabmap::Transform pose);

// @brief: Perform semantic mapping on a given point cloud using a given neural
// network
// @param net: The neural network to use for semantic mapping
// @param exporter: The database exporter to use for converting images to numpy
// arrays
// @param mapping_data: The data to use for semantic mapping
// @param cloud: The point cloud to perform semantic mapping on
// @param timestamp: The timestamp of the data
// @return: A vector of objects containing the point cloud of the object, the
// closest point to the camera, the label of the object, and the confidence of
// the label
std::vector<Object> semantic_mapping(
  py::object &net,
  std::vector<std::tuple<cv::Mat, cv::Mat, rtabmap::Transform,
                         std::map<std::pair<int, int>, int>>> &mapping_data,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, std::string &timestamp);

#endif // SEMANTIC_MAPPING_HPP