#include "wc_database_exporter.hpp"

WCDatabaseExporter::~WCDatabaseExporter()
{
  // base path
  std::string path = std::string(PROJECT_PATH) + "/output/" + timestamp_;

  // save the point cloud
  std::string cloud_path = path + "/cloud/" + timestamp_ + ".pcd";
  pcl::io::savePCDFileBinary(cloud_path, *rtabmap_cloud_);

  // save the occupancy grid
  std::string grid_path = path + "/grid/" + timestamp_;
  std::cout << "occupancy grid size: " << rtabmap_cloud_->points.size()
            << std::endl;
  rtabmap_occupancy_grid_ = point_cloud_to_occupancy_grid(rtabmap_cloud_);
  nav2_map_server::SaveParameters save_params;
  save_params.map_file_name = grid_path;
  save_params.image_format = "pgm";
  save_params.free_thresh = 0.196;
  save_params.occupied_thresh = 0.65;
  nav2_map_server::saveMapToFile(*rtabmap_occupancy_grid_, save_params);
  int rgbImagesExported = 0;
  int depthImagesExported = 0;
  for (const auto &data : mapping_data_) {
    // save rgb images
    std::string image_path =
      path + "/images/" + std::to_string(rgbImagesExported) + ".jpg";
    cv::imwrite(image_path, std::get<0>(data));
    ++rgbImagesExported;

    // save depth images
    cv::Mat depthExported = std::get<1>(data);
    std::string depth_path =
      path + "/depths/" + std::to_string(depthImagesExported) + ".jpg";

    cv::imwrite(depth_path, depthExported);
    ++depthImagesExported;
  }

  // save calibration per image (calibration can change over time, e.g.
  // camera has auto focus)
  for (size_t i = 0; i < camera_models_.size(); i++) {
    for (size_t j = 0; j < camera_models_.at(i).size(); j++) {
      rtabmap::CameraModel model = camera_models_.at(i).at(j);
      std::string modelName = std::to_string(i);
      std::string dir = path + "/camera_models/";

      if (camera_models_.at(i).size() > 1) {
        modelName += "_" + uNumber2Str((int)j);
      }
      model.setName(modelName);
      model.save(dir);
    }
  }
  std::cout << "RGB Images exported: " << rgbImagesExported << std::endl;
  std::cout << "Depth Images exported: " << depthImagesExported << std::endl;
}

nav_msgs::msg::OccupancyGrid::SharedPtr
WCDatabaseExporter::point_cloud_to_occupancy_grid(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
  //   // calculate the centroid
  Eigen::Matrix<float, 4, 1> centroid;
  pcl::ConstCloudIterator<pcl::PointXYZRGB> cloud_iterator(*cloud);
  pcl::compute3DCentroid(cloud_iterator, centroid);

  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();

  for (const auto &point : cloud->points) {
    if (point.x > max_x) {
      max_x = point.x;
    }
    if (point.y > max_y) {
      max_y = point.y;
    }
    if (point.x < min_x) {
      min_x = point.x;
    }
    if (point.y < min_y) {
      min_y = point.y;
    }
  }

  nav_msgs::msg::OccupancyGrid::SharedPtr occupancy_grid =
    std::make_shared<nav_msgs::msg::OccupancyGrid>();
  // necessary? can't remember why i did this but i remember there was a reason
  cloud->width = cloud->points.size();
  occupancy_grid->info.resolution = 0.05;
  occupancy_grid->info.width =
    std::abs(max_x - min_x) / occupancy_grid->info.resolution + 1;
  occupancy_grid->info.height =
    std::abs(max_y - min_y) / occupancy_grid->info.resolution + 1;
  occupancy_grid->info.origin.position.x = min_x;
  occupancy_grid->info.origin.position.y = min_y;
  occupancy_grid->info.origin.position.z = 0;
  occupancy_grid->info.origin.orientation.x = 0;
  occupancy_grid->info.origin.orientation.y = 0;
  occupancy_grid->info.origin.orientation.z = 0;
  occupancy_grid->info.origin.orientation.w = 1;
  occupancy_grid->data.resize(
    occupancy_grid->info.width * occupancy_grid->info.height, 0);
  for (const auto &point : cloud->points) {
    int x = (point.x - min_x) / occupancy_grid->info.resolution;
    int y = (point.y - min_y) / occupancy_grid->info.resolution;
    int index = y * occupancy_grid->info.width + x;
    occupancy_grid->data.at(index) = 100;
  }
  return occupancy_grid;
}

// @brief : This function takes in a point cloud and a camera transform and
// projects the point cloud to the camera frame. The sequence of
// filters was determined by trial and error
// @param cloud: The point cloud to filter
// @return The filtered point cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr WCDatabaseExporter::filter_point_cloud(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cleaned_cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);
  for (const auto &pt : cloud->points) {
    if (pcl::isFinite(pt) && std::abs(pt.x) <= 1e5 && std::abs(pt.y) <= 1e5 &&
        std::abs(pt.z) <= 1e5) {
      cleaned_cloud->points.push_back(pt);
    } else {
      // std::cout << "Bad point detected: x=" << pt.x << ", y=" << pt.y
      //           << ", z=" << pt.z << std::endl;
      if (!std::isnan(pt.x) && !std::isnan(pt.y) && !std::isnan(pt.z)) {
        std::cout << "Bad point detected: x=" << pt.x << ", y=" << pt.y
                  << ", z=" << pt.z << std::endl;
      }
    }
  }
  cleaned_cloud->width = cleaned_cloud->points.size();
  cleaned_cloud->height = 1;
  cleaned_cloud->is_dense = true;

  // statistical outlier removal
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sor_cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud(cleaned_cloud);
  sor.setMeanK(50); // increase for more permissive, decrease for less
  sor.setStddevMulThresh(
    1.0); // increase for more permissive, decrease for less
  sor.filter(*sor_cloud);

  // radius outlier removal
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr radius_cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> radius_outlier;
  radius_outlier.setInputCloud(sor_cloud);
  radius_outlier.setRadiusSearch(
    0.2); // adjust based on spacing in the point cloud
  radius_outlier.setMinNeighborsInRadius(
    5); // increase for more aggressive outlier removal
  radius_outlier.filter(*radius_cloud);
  radius_cloud->width = radius_cloud->points.size();

  // find the lowest point in the pointcloud
  auto min_point_iter =
    std::min_element(radius_cloud->points.begin(), radius_cloud->points.end(),
                     [](const pcl::PointXYZRGB &lhs,
                        const pcl::PointXYZRGB &rhs) { return lhs.z < rhs.z; });
  // passthrough filter
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pass_cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud(radius_cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(min_point_iter->z +
                         0.7,    // 0.7 meters, magic number sorry
                       FLT_MAX); // adjust based on the scene
  pass.filter(*pass_cloud);
  pass_cloud->width = pass_cloud->points.size();

  //   another radius outlier removal
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr radius2_cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> radius2_outlier;
  radius2_outlier.setInputCloud(pass_cloud);
  radius2_outlier.setRadiusSearch(
    0.2); // adjust based on spacing in the point cloud
  radius2_outlier.setMinNeighborsInRadius(
    3); // increase for more aggressive outlier removal
  radius2_outlier.filter(*radius2_cloud);
  radius2_cloud->width = radius2_cloud->points.size();

  return radius2_cloud;
}

bool WCDatabaseExporter::initialize_rtabmap_database()
{
  rtabmap::ParametersMap parameters;
  rtabmap::DBDriver *driver = rtabmap::DBDriver::create();

  if (driver->openConnection(rtabmap_database_path_)) {
    parameters = driver->getLastParameters();
    driver->closeConnection(false);
  } else {
    std::cout << "Failed to open database" << std::endl;
    return false;
  }
  delete driver;
  driver = 0;

  UTimer timer;

  std::cout << "Loading database: " << rtabmap_database_path_ << std::endl;
  rtabmap::Rtabmap rtabmap;
  rtabmap.init(parameters, rtabmap_database_path_);
  std::cout << "Loaded database in " << timer.ticks() << "s" << std::endl;

  std::cout << "Optimizing the map..." << std::endl;
  rtabmap.getGraph(optimizedPoses, links, true, true, &nodes, true, true, true,
                   true);
  std::cout << "Optimizing the map... done (" << timer.ticks()
            << "s, poses=" << optimizedPoses.size() << ")." << std::endl;

  if (optimizedPoses.size() == 0) {
    std::cout << "No optimized poses found" << std::endl;
    return false;
  }

  return true;
}

void WCDatabaseExporter::assembleSceneFromOptimizedPoses()
{
  UTimer timer;
  for (std::map<int, rtabmap::Transform>::iterator iter =
         optimizedPoses.lower_bound(1);
       iter != optimizedPoses.end(); ++iter) {

    rtabmap::Signature node = nodes.find(iter->first)->second;

    // Get camera model
    std::vector<rtabmap::CameraModel> models = node.sensorData().cameraModels();

    // Uncompress RGB + depth
    pcl::IndicesPtr indices(new std::vector<int>);
    cv::Mat rgb, depth;
    rtabmap::LaserScan scan;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;

    float noiseRadius = 0.0f;
    int noiseMinNeighbors = 5;

    if (node.getWeight() != -1) {
      node.sensorData().uncompressData(&rgb, &depth, &scan);
    }

    // Skip if no camera or no depth
    if (scan.empty()) {
      printf("Node %d doesn't have scan data, empty cloud is created.\n",
             iter->first);
    }
    // Store images and calibration
    rgb_images[iter->first] = rgb;
    camera_models_.push_back(models);

    // Create point cloud from
    cloud =
      rtabmap::util3d::laserScanToPointCloudI(scan, scan.localTransform());
    if (noiseRadius > 0.0f && noiseMinNeighbors > 0) {
      indices =
        rtabmap::util3d::radiusFiltering(cloud, noiseRadius, noiseMinNeighbors);
    }

    // Transform point cloud into global frame
    if (cloud && !cloud->empty()) {
      cloud = rtabmap::util3d::transformPointCloud(cloud, iter->second);

      if (assembledCloud->empty()) {
        *assembledCloud = *cloud;
      } else {
        *assembledCloud += *cloud;
      }

      rawViewpointIndices.resize(assembledCloud->size(), iter->first);
    }

    // Record viewpoint, robot pose, etc.

    rtabmap::Transform lidarViewpoint =
      iter->second * node.sensorData().laserScanRaw().localTransform();
    rawViewpoints.insert(std::make_pair(iter->first, lidarViewpoint));
    robotPoses.insert(std::make_pair(iter->first, iter->second));
    cameraStamps.insert(std::make_pair(iter->first, node.getStamp()));

    cameraModels.insert(std::make_pair(iter->first, models));
    if (cameraPoses.empty()) {
      cameraPoses.resize(models.size());
    }

    for (size_t i = 0; i < models.size(); ++i) {
      cameraPoses[i].insert(
        std::make_pair(iter->first, iter->second * models[i].localTransform()));
    }

    if (depth.type() == CV_16UC1 || depth.type() == CV_32FC1) {
      cameraDepths.insert(std::make_pair(iter->first, depth));
    }

    std::cout << "assembledCloud: " << assembledCloud->size() << std::endl;
  }

  std::cout << "Create and assemble the clouds... done (" << timer.ticks()
            << "s, " << assembledCloud->size() << " points)." << std::endl;
}

void WCDatabaseExporter::projectAndColorizePointCloud()
{
  UTimer timer;
  pcl::copyPointCloud(*assembledCloud, *rtabmap_cloud_);

  // extract the camera poses
  std::map<int, rtabmap::Transform> optimized_poses;

  for (std::map<int, rtabmap::Transform>::iterator iter =
         optimizedPoses.lower_bound(1);
       iter != optimizedPoses.end(); ++iter) {
    optimized_poses[iter->first] = iter->second;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudWithoutNormals(
    new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr rawAssembledCloud(
    new pcl::PointCloud<pcl::PointXYZ>);

  pcl::copyPointCloud(*rtabmap_cloud_, *cloudWithoutNormals);
  rawAssembledCloud = cloudWithoutNormals;

  pcl::PointCloud<pcl::Normal>::Ptr normals =
    rtabmap::util3d::computeNormals(cloudWithoutNormals, 20, 0);

  bool groundNormalsUp = true;
  if (!assembledCloud->empty()) {
    UASSERT(assembledCloud->size() == normals->size());
    pcl::concatenateFields(*assembledCloud, *normals, *cloudIToExport);
    std::cout << "Computing normals of the assembled cloud... done! ("
              << timer.ticks() << "s, " << (int)assembledCloud->size()
              << " points)" << std::endl;
    assembledCloud->clear();

    // adjust with point of views
    std::cout << "Adjust normals to viewpoints of the assembled cloud... ("
              << cloudIToExport->size() << " points)" << std::endl;
    rtabmap::util3d::adjustNormalsToViewPoints(rawViewpoints, rawAssembledCloud,
                                               rawViewpointIndices,
                                               cloudIToExport, groundNormalsUp);
    std::cout << "adjust normals to viewpoints of the assembled cloud... ("
              << timer.ticks() << "s, " << (int)cloudIToExport->size()
              << " points)" << std::endl;
  }
}

void WCDatabaseExporter::assemble_colored_point_cloud()
{
  float textureRange = 0.0f;
  float textureAngle = 0.0f;
  std::vector<float> textureRoiRatios;
  cv::Mat projMask;
  bool distanceToCamPolicy = false;
  const rtabmap::ProgressState progressState;
  pointToPixel = rtabmap::util3d::projectCloudToCameras(
    *cloudIToExport, robotPoses, cameraModels, textureRange, textureAngle,
    textureRoiRatios, projMask, distanceToCamPolicy, &progressState);

  std::vector<int> pointToCamId;
  std::vector<float> pointToCamIntensity;
  pointToCamId.resize(cloudIToExport->size());

  UASSERT(pointToPixel.empty() || pointToPixel.size() == pointToCamId.size());
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr assembledCloudValidPoints(
    new pcl::PointCloud<pcl::PointXYZRGBNormal>());
  assembledCloudValidPoints->resize(pointToCamId.size());

  // Figure out what color each point in the pointcloud should be based on
  // pixel
  // color
  int imagesDone = 1;
  for (std::map<int, rtabmap::Transform>::iterator iter = robotPoses.begin();
       iter != robotPoses.end(); ++iter) {
    int nodeID = iter->first;
    cv::Mat image;
    if (uContains(nodes, nodeID) &&
        !nodes.at(nodeID).sensorData().imageCompressed().empty()) {
      nodes.at(nodeID).sensorData().uncompressDataConst(&image, 0);
    }
    if (!image.empty()) {
      std::cout << "Have image" << std::endl;
      UASSERT(cameraModels.find(nodeID) != cameraModels.end());
      int modelsSize = cameraModels.at(nodeID).size();
      for (size_t i = 0; i < pointToPixel.size(); ++i) {
        int cameraIndex = pointToPixel[i].first.second;
        if (nodeID == pointToPixel[i].first.first && cameraIndex >= 0) {
          pcl::PointXYZRGBNormal pt;
          float intensity = 0;
          if (!cloudIToExport->empty()) {
            pt.x = cloudIToExport->at(i).x;
            pt.y = cloudIToExport->at(i).y;
            pt.z = cloudIToExport->at(i).z;
            pt.normal_x = cloudIToExport->at(i).normal_x;
            pt.normal_y = cloudIToExport->at(i).normal_y;
            pt.normal_z = cloudIToExport->at(i).normal_z;
            intensity = cloudIToExport->at(i).intensity;
          }

          int subImageWidth = image.cols / modelsSize;
          cv::Mat subImage = image(
            cv::Range::all(), cv::Range(cameraIndex * subImageWidth,
                                        (cameraIndex + 1) * subImageWidth));

          int x = pointToPixel[i].second.x * (float)subImage.cols;
          int y = pointToPixel[i].second.y * (float)subImage.rows;
          UASSERT(x >= 0 && x < subImage.cols);
          UASSERT(y >= 0 && y < subImage.rows);

          UASSERT(subImage.type() == CV_8UC1);
          pt.r = pt.g = pt.b = subImage.at<unsigned char>(
            pointToPixel[i].second.y * subImage.rows,
            pointToPixel[i].second.x * subImage.cols);

          int exportedId = nodeID;
          pointToCamId[i] = exportedId;
          if (!pointToCamIntensity.empty()) {
            pointToCamIntensity[i] = intensity;
          }
          assembledCloudValidPoints->at(i) = pt;
        }
      }
    }
    std::cout << "Processed " << imagesDone++ << "/"
              << static_cast<int>(robotPoses.size()) << " images\n";
  }

  pcl::IndicesPtr validIndices(new std::vector<int>(pointToPixel.size()));
  size_t oi = 0;
  for (size_t i = 0; i < pointToPixel.size(); ++i) {
    if (pointToPixel[i].first.first <= 0) {
      pcl::PointXYZRGBNormal pt;
      float intensity = 0;
      if (!cloudIToExport->empty()) {
        pt.x = cloudIToExport->at(i).x;
        pt.y = cloudIToExport->at(i).y;
        pt.z = cloudIToExport->at(i).z;
        pt.normal_x = cloudIToExport->at(i).normal_x;
        pt.normal_y = cloudIToExport->at(i).normal_y;
        pt.normal_z = cloudIToExport->at(i).normal_z;
        intensity = cloudIToExport->at(i).intensity;
      }

      pointToCamId[i] = 0; // invalid
      pt.b = 0;
      pt.g = 0;
      pt.r = 255;
      if (!pointToCamIntensity.empty()) {
        pointToCamIntensity[i] = intensity;
      }
      assembledCloudValidPoints->at(i) = pt; // red
      validIndices->at(oi++) = i;
    } else {
      validIndices->at(oi++) = i;
    }
  }

  if (oi != validIndices->size()) {
    validIndices->resize(oi);
    assembledCloudValidPoints = rtabmap::util3d::extractIndices(
      assembledCloudValidPoints, validIndices, false, false);
    std::vector<int> pointToCamIdTmp(validIndices->size());
    std::vector<float> pointToCamIntensityTmp(validIndices->size());
    for (size_t i = 0; i < validIndices->size(); ++i) {
      pointToCamIdTmp[i] = pointToCamId[validIndices->at(i)];
      pointToCamIntensityTmp[i] = pointToCamIntensity[validIndices->at(i)];
    }
    pointToCamId = pointToCamIdTmp;
    pointToCamIntensity = pointToCamIntensityTmp;
    pointToCamIdTmp.clear();
    pointToCamIntensityTmp.clear();
  }

  cloudToExport = assembledCloudValidPoints;
  std::cout << "Assembling colored point cloud... done!" << std::endl;
}

void WCDatabaseExporter::finalize_and_return_result(Result &result)
{
  pcl::copyPointCloud(*cloudToExport, *rtabmap_cloud_);
  rtabmap_cloud_ = filter_point_cloud(rtabmap_cloud_);

  // RANSAC();

  for (std::map<int, std::vector<rtabmap::CameraModel>>::iterator iter =
         cameraModels.begin();
       iter != cameraModels.end(); ++iter) {

    std::cout << "Processing node " << iter->first << std::endl;

    // Create an empty frame for a Mono8 image (grayscale)
    cv::Mat frame = cv::Mat::zeros(iter->second.front().imageHeight(),
                                   iter->second.front().imageWidth(), CV_8UC1);
    cv::Mat depth(iter->second.front().imageHeight(),
                  iter->second.front().imageWidth() * iter->second.size(),
                  CV_32FC1);
    cv::Mat combined_image =
      rgb_images[iter->first]; // Assuming mono_images map
    // stores the Mono8 images
    int width = combined_image.cols / 2;
    int height = combined_image.rows;
    cv::Mat left_image = combined_image(cv::Rect(0, 0, width, height)).clone();
    cv::Mat right_image =
      combined_image(cv::Rect(width, 0, width, height)).clone();
    std::pair<cv::Mat, std::map<std::pair<int, int>, int>> depth_map;

    // Iterate over each camera model in the node
    for (size_t i = 0; i < iter->second.size(); ++i) {
      cv::Mat mono_frame = (i == 0) ? left_image : right_image;

      cv::Mat image_rotate;
      cv::rotate(mono_frame, image_rotate, cv::ROTATE_90_COUNTERCLOCKWISE);

      cv::Mat color_image;
      cv::cvtColor(image_rotate, color_image, cv::COLOR_GRAY2BGR);

      cv::Mat lab;
      cv::cvtColor(color_image, lab, cv::COLOR_BGR2Lab);

      std::vector<cv::Mat> lab_planes(3);
      cv::split(lab, lab_planes);
      cv::Mat l_channel = lab_planes[0];
      cv::Mat a = lab_planes[1];
      cv::Mat b = lab_planes[2];

      // Apply CLAHE to the L-channel
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
      cv::Mat cl;
      clahe->apply(l_channel, cl);

      // Merge the CLAHE enhanced L-channel back with A and B channels
      cv::Mat limg;
      cv::merge(std::vector<cv::Mat>{cl, a, b}, limg);

      // Convert the enhanced LAB image back to BGR
      cv::Mat enhanced_img;
      cv::cvtColor(limg, enhanced_img, cv::COLOR_Lab2BGR);

      depth_map = project_cloud_to_camera(
        iter->second.at(i).imageSize(), iter->second.at(i).K(), rtabmap_cloud_,
        robotPoses.at(iter->first) * iter->second.at(i).localTransform());

      // Copy the depth values to the depth matrix for this camera
      depth_map.first.copyTo(
        depth(cv::Range::all(),
              cv::Range(i * iter->second.front().imageWidth(),
                        (i + 1) * iter->second.front().imageWidth())));

      // Iterate over all pixels and visualize the depth by drawing circles on
      // the grayscale image
      for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
          if (depth.at<float>(y, x) > 0.0f) { // Valid depth
            // In a grayscale image, use the intensity directly for
            // visualization
            uchar intensity =
              enhanced_img.at<uchar>(y, x); // Intensity from the Mono8 image
            // We use intensity as a grayscale color for the circle (white
            // on black background)
            cv::circle(frame, cv::Point(x, y), 1, cv::Scalar(intensity), -1);
          }
        }
      }
      // Store the mapping data (Mono8 image, frame with depth circles, pose,
      // and depth map)
      mapping_data_.push_back(
        {enhanced_img, frame, robotPoses.at(iter->first), depth_map.second});
    }
  }

  result.success = true;
  result.timestamp = timestamp_;
  result.cloud = rtabmap_cloud_;
  result.mapping_data = mapping_data_;

  std::cout << "Finished loading database" << std::endl;
  std::cout << "Number of images: " << mapping_data_.size() << std::endl;
  std::cout << "Number of points in cloud: " << rtabmap_cloud_->points.size()
            << std::endl;
  std::cout << "Timestamp: " << timestamp_ << std::endl;

  return;
}

// @brief Project a point cloud to a given image frame, and map the index of
// each point that is visible in the camera frame to the pixel coordinate in
// the image frame
// @param image_size: The size of the image frame
// @param camera_matrix: The camera matrix
// @param cloud: The point cloud to project to the camera frame
// @param camera_transform: The transformation matrix from the camera frame to
// the world frame
// @return A pair containing the depth image and a map from pixel

std::pair<cv::Mat, std::map<std::pair<int, int>, int>>
WCDatabaseExporter::project_cloud_to_camera(
  const cv::Size &image_size, const cv::Mat &camera_matrix,
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
  const rtabmap::Transform &camera_transform)
{
  assert(!camera_transform.isNull());
  assert(!cloud->empty());
  assert(camera_matrix.type() == CV_64FC1 && camera_matrix.cols == 3 &&
         camera_matrix.cols == 3);

  float fx = camera_matrix.at<double>(0, 0);
  float fy = camera_matrix.at<double>(1, 1);
  float cx = camera_matrix.at<double>(0, 2);
  float cy = camera_matrix.at<double>(1, 2);

  cv::Mat depth_image = cv::Mat::zeros(image_size, CV_32FC1);
  rtabmap::Transform t = camera_transform.inverse();

  // create a map from each pixel coordinate to the index of their point in the
  // pointcloud
  std::map<std::pair<int, int>, int> pixel_to_point_map;

  int count = 0;
  for (pcl::PointCloud<pcl::PointXYZRGB>::const_iterator it = cloud->begin();
       it != cloud->end(); ++it) {
    pcl::PointXYZRGB ptScan = *it;

    // transform point from world frame to camera frame
    ptScan = rtabmap::util3d::transformPoint(ptScan, t);

    // re-project in camera frame
    float z = ptScan.z;
    bool set = false;
    if (z > 0.0f) {
      float invZ = 1.0f / z;
      float dx = (fx * ptScan.x) * invZ + cx;
      float dy = (fy * ptScan.y) * invZ + cy;
      int dx_low = dx;
      int dy_low = dy;
      int dx_high = dx + 0.5f;
      int dy_high = dy + 0.5f;
      if (uIsInBounds(dx_low, 0, depth_image.cols) &&
          uIsInBounds(dy_low, 0, depth_image.rows)) {
        set = true;
        float &zReg = depth_image.at<float>(dy_low, dx_low);
        if (zReg == 0 || z < zReg) {
          zReg = z;
        }
      }
      if ((dx_low != dx_high || dy_low != dy_high) &&
          uIsInBounds(dx_high, 0, depth_image.cols) &&
          uIsInBounds(dy_high, 0, depth_image.rows)) {
        set = true;
        float &zReg = depth_image.at<float>(dy_high, dx_high);
        if (zReg == 0 || z < zReg) {
          zReg = z;
        }
      }
      if (set) {
        pixel_to_point_map[{dy_low, dx_low}] = count;
      }
    }
    count++;
  }

  return {depth_image, pixel_to_point_map};
}