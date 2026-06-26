// Created by Graham Clifford
//
// This file was created as a part of this project:
// https://graham-clifford.com/Localizing-and-Navigating-in-Semantic-Maps-Created-by-an-iPhone/
//
// This script extracts RGB images, depth images, a point cloud, and camera
// matrices from an RTABMap database file created by an iPhone without LIDAR
// enabled. The information extracted from the database is then used to create
// a semantic map of the environment.

// I used this RTABMap source code as a reference for this file
// https://github.com/introlab/rtabmap/blob/ff61266430017eb4924605b832cd688c8739af18/tools/Export/main.cpp#L1104-L1115

#include "database_exporter.hpp"

DatabaseExporter::DatabaseExporter(std::string rtabmap_database_name,
                                   std::string model_name)
  : timestamp_(generate_timestamp_string())
{
  // initialize variables, create output directories
  if (rtabmap_database_name.empty()) {
    std::cout << "RTABMap database name is empty" << std::endl;
    return;
  }

  rtabmap_database_path_ =
    std::string(PROJECT_PATH) + "/databases/" + rtabmap_database_name;

  rtabmap_occupancy_grid_ =
    nav_msgs::msg::OccupancyGrid::SharedPtr(new nav_msgs::msg::OccupancyGrid);

  if (model_name.empty()) {
    std::cout << "Model name is empty, not performing semantic mapping"
              << std::endl;
  } else {
    model_path_ = std::string(PROJECT_PATH) + "/models/" + model_name;
    std::cout << "Loading model: " << model_path_ << std::endl;

    net_ = cv::dnn::readNet(model_path_);
  }

  // base path
  std::string path = std::string(PROJECT_PATH) + "/output/" + timestamp_;
  if (!std::filesystem::create_directory(path)) {
    std::cout << "Failed to create output directory" << std::endl;
    return;
  }
  // rgb images from every pose in the pose graph in order
  if (!std::filesystem::create_directory(path + "/images")) {
    std::cout << "Failed to create images directory" << std::endl;
    return;
  }
  // depth images from every pose in the pose graph in order
  if (!std::filesystem::create_directory(path + "/depths")) {
    std::cout << "Failed to create depths directory" << std::endl;
    return;
  }
  // full point cloud (.pcl)
  if (!std::filesystem::create_directory(path + "/cloud")) {
    std::cout << "Failed to create cloud directory" << std::endl;
    return;
  }
  // occupancy grid files saved with nav2_map_server (.pgm and .yaml)
  if (!std::filesystem::create_directory(path + "/grid")) {
    std::cout << "Failed to create grid directory" << std::endl;
    return;
  }
  // camera matrices for each pose in the pose grpah in .yaml format
  if (!std::filesystem::create_directory(path + "/camera_models")) {
    std::cout << "Failed to create camera_models directory" << std::endl;
    return;
  }
  // individual .pcl files for objects detected in the environment
  if (!std::filesystem::create_directory(path + "/objects")) {
    std::cout << "Failed to create objects directory" << std::endl;
    return;
  }
  // a .yaml file listing the landmarks from semantic mapping (name and xy pos)
  if (!std::filesystem::create_directory(path + "/landmarks")) {
    std::cout << "Failed to create landmarks directory" << std::endl;
    return;
  }
  // RGB and depth images with YOLOv8 detections overlaid
  if (!std::filesystem::create_directory(path + "/detections")) {
    std::cout << "Failed to create detections directory" << std::endl;
    return;
  }

  if (!std::filesystem::create_directory(path + "/cleaned")) {
    std::cout << "Failed to create cleaned directory" << std::endl;
    return;
  }

  if (!std::filesystem::create_directory(path + "/edges")) {
    std::cout << "Failed to create edges directory" << std::endl;
    return;
  }
}

DatabaseExporter::~DatabaseExporter() {}

// @brief: Generate a string from the current time
// @return: The string representation of the current time in the format
// YYYY-MM-DD_HH-MM-SS
std::string DatabaseExporter::generate_timestamp_string()
{
  std::time_t now = std::time(nullptr);
  std::tm *ptm = std::localtime(&now);

  std::ostringstream oss;

  oss << std::put_time(ptm, "%Y-%m-%d_%H-%M-%S");

  return oss.str();
}

// void DatabaseExporter::RANSAC()
// {
//   pcl::ModelCoefficients::Ptr coefficients_0(new pcl::ModelCoefficients);
//   pcl::ModelCoefficients::Ptr coefficients_1(new pcl::ModelCoefficients);
//   pcl::PointIndices::Ptr inliers_0(new pcl::PointIndices);
//   pcl::PointIndices::Ptr inliers_1(new pcl::PointIndices);
//   // Create the segmentation object
//   pcl::SACSegmentation<pcl::PointXYZRGB> seg;
//   // Optional
//   seg.setOptimizeCoefficients(true);
//   // Mandatory
//   seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
//   seg.setMethodType(pcl::SAC_RANSAC);
//   seg.setDistanceThreshold(0.1);
//   seg.setAxis(Eigen::Vector3f(
//     1.0, 0.0,
//     0.0)); // Normal should be orthogonal to this (i.e., vertical plane)
//   seg.setEpsAngle(60.0 * M_PI / 180.0);

//   seg.setInputCloud(rtabmap_cloud_);
//   seg.segment(*inliers_0, *coefficients_0);

//   seg.setDistanceThreshold(0.1);
//   seg.setInputCloud(rtabmap_cloud_);
//   seg.segment(*inliers_1, *coefficients_1);

//   if (inliers_0->indices.size() == 0) {
//     PCL_ERROR("Could not estimate a planar model for the given dataset.\n");
//     return;
//   }

//   std::cerr << "Model coefficients: " << coefficients_0->values[0] << " "
//             << coefficients_0->values[1] << " " << coefficients_0->values[2]
//             << " " << coefficients_0->values[3] << std::endl;

//   std::cerr << "Model inliers: " << inliers_0->indices.size() << std::endl;
//   std::cerr << "Model inliers: " << inliers_1->indices.size() << std::endl;
//   // for (const auto& idx: inliers->indices)
//   //   std::cerr << idx << "    " << rtabmap_cloud_->points[idx].x << " "
//   //                              << rtabmap_cloud_->points[idx].y << " "
//   //                              << rtabmap_cloud_->points[idx].z << std::endl;

//   std::unordered_set<int> inlier0_set(inliers_0->indices.begin(),
//                                       inliers_0->indices.end());

//   std::vector<int> to_remove;
//   for (int idx : inliers_1->indices) {
//     if (inlier0_set.find(idx) == inlier0_set.end()) {
//       to_remove.push_back(
//         idx); // This point is in inliers_1 but not in inliers_0
//     }
//   }

//   pcl::PointIndices::Ptr removal_indices(new pcl::PointIndices);
//   removal_indices->indices = to_remove;

//   pcl::ExtractIndices<pcl::PointXYZRGB> extract;
//   extract.setInputCloud(rtabmap_cloud_);
//   // extract.setIndices(removal_indices);
//   extract.setIndices(inliers_0);
//   // extract.setNegative(true);
//   extract.setNegative(
//     false); // Keep everything except the near-plane non-plane points

//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cleaned_cloud(
//     new pcl::PointCloud<pcl::PointXYZRGB>);
//   extract.filter(*cleaned_cloud);

//   // (Optional) assign back if needed
//   // rtabmap_cloud_ = cleaned_cloud;
// }

Result DatabaseExporter::load_rtabmap_db()
{
  Result result;

  bool flag = initialize_rtabmap_database();
  if (!flag) {
    return result;
  }

  assembleSceneFromOptimizedPoses();
  projectAndColorizePointCloud();
  assemble_colored_point_cloud();
  finalize_and_return_result(result);

  return result;
}

// void DatabaseExporter::assembleSceneFromOptimizedPoses()
// {
//   UTimer timer;
//   for (std::map<int, rtabmap::Transform>::iterator iter =
//          optimizedPoses.lower_bound(1);
//        iter != optimizedPoses.end(); ++iter) {
//     rtabmap::Signature node = nodes.find(iter->first)->second;

//     // uncompress data
//     std::vector<rtabmap::CameraModel> models =
//     node.sensorData().cameraModels();

//     cv::Mat rgb;
//     cv::Mat depth;

//     // pcl::IndicesPtr indices(new std::vector<int>);
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
//     // pcl::PointCloud<pcl::PointXYZI>::Ptr cloudI;
//     if (node.getWeight() != -1) {
//       int decimation = 1;
//       int maxRange = 100.0;
//       int minRange = 0.0;
//       cv::Mat tmpDepth;
//       rtabmap::LaserScan scan;
//       node.sensorData().uncompressData(
//         &rgb,
//         !node.sensorData().depthOrRightCompressed().empty() ? &tmpDepth : 0,
//         &scan);

//       if (scan.empty()) {
//         std::cout << "Node " << iter->first
//                   << " doesn't have scan data, empty cloud is created."
//                   << std::endl;
//       }

//       scan =
//         rtabmap::util3d::commonFiltering(scan, decimation, minRange,
//         maxRange);
//       if (scan.hasRGB()) {
//         cloud = rtabmap::util3d::laserScanToPointCloudRGB(
//           scan, scan.localTransform());
//       } // else {
//       //   cloudI = rtabmap::util3d::laserScanToPointCloudI(scan,
//       //   scan.localTransform());
//       // }
//     }

//     node.sensorData().uncompressData(&rgb, &depth);

//     // store these for later
//     if (!rgb.empty()) {
//       rgb_images[iter->first] = rgb;
//       // save calibration per image
//       camera_models_.push_back(models);
//       // calibration can change over time, e.g. camera has auto focus
//     }

//     if (cloud.get() && !cloud->empty()) {
//       cloud = rtabmap::util3d::transformPointCloud(cloud, iter->second);
//     } // else if (cloudI.get() && !cloudI->empty())
//       // cloudI = rtabmap::util3d::transformPointCloud(cloudI, iter->second);

//     rtabmap::Transform lidarViewpoint =
//       iter->second * node.sensorData().laserScanRaw().localTransform();
//     rawViewpoints.insert(std::make_pair(iter->first, lidarViewpoint));

//     if (cloud.get() && !cloud->empty()) {
//       if (assembledCloud->empty()) {
//         *assembledCloud = *cloud;
//       } else {
//         *assembledCloud += *cloud;
//       }
//       rawViewpointIndices.resize(assembledCloud->size(), iter->first);
//     } // else if (cloudI.get() && !cloudI->empty()) {
//     //   if (assembledCloudI->empty()) {
//     //     *assembledCloudI = *cloudI;
//     //   } else {
//     //     *assembledCloudI += *cloudI;
//     //   }
//     //   rawViewpointIndices.resize(assembledCloudI->size(), iter->first);
//     // }

//     std::cout << "assembledCloud: " << assembledCloud->size() << std::endl;
//     // std::cout << "assembledCloudI: " << assembledCloudI->size() <<
//     std::endl;

//     robotPoses.insert(std::make_pair(iter->first, iter->second));
//     cameraStamps.insert(std::make_pair(iter->first, node.getStamp()));
//     if (models.empty() && node.getWeight() == -1 && !cameraModels.empty()) {
//       // For intermediate nodes, use latest models
//       models = cameraModels.rbegin()->second;
//     }
//     if (!models.empty()) {
//       if (!node.sensorData().imageCompressed().empty()) {
//         cameraModels.insert(std::make_pair(iter->first, models));
//       }
//       if (true) {
//         if (cameraPoses.empty()) {
//           cameraPoses.resize(models.size());
//         }
//         UASSERT_MSG(models.size() == cameraPoses.size(),
//                     "Not all nodes have same number of cameras to export "
//                     "camera poses.");
//         for (size_t i = 0; i < models.size(); ++i) {
//           cameraPoses[i].insert(std::make_pair(
//             iter->first, iter->second * models[i].localTransform()));
//         }
//       }
//     }
//     if (!depth.empty() &&
//         (depth.type() == CV_16UC1 || depth.type() == CV_32FC1)) {
//       cameraDepths.insert(std::make_pair(iter->first, depth));
//     }

//     if (!node.sensorData().laserScanCompressed().empty()) {
//       scanPoses.insert(std::make_pair(
//         iter->first,
//         iter->second *
//           node.sensorData().laserScanCompressed().localTransform()));
//     }
//     std::cout << "Create and assemble the clouds... done (" << timer.ticks()
//               << "s, "
//               << (!assembledCloud->empty() ? (int)assembledCloud->size()
//                                            : (int)assembledCloudI->size())
//               << " points)." << std::endl;
//   }
// }
