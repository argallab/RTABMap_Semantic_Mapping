#include "semantic_mapping.hpp"

// @brief: This function takes in a cv::Mat object and converts it to a numpy
// array
// @param mat: The cv::Mat object to convert
// @return: The numpy array
py::array mat_to_numpy(const cv::Mat &mat)
{
  return py::array_t<uint8_t>({mat.rows, mat.cols, mat.channels()},
                              {mat.step[0], mat.step[1], sizeof(uint8_t)},
                              mat.data);
}

pcl::PointXYZ calculate_centroid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
  pcl::PointXYZ centroid;
  for (const auto &point : cloud->points) {
    centroid.x += point.x;
    centroid.y += point.y;
    centroid.z += point.z;
  }
  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;
  cloud->points.resize(cloud->width * cloud->height);

  centroid.x /= cloud->size();
  centroid.y /= cloud->size();
  centroid.z /= cloud->size();

  return centroid;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud_from_bounding_box(
  std::tuple<std::string, float, BoundingBox> bounding_box,
  std::map<std::pair<int, int>, int> pixel_to_point_map,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, rtabmap::Transform pose)
{
  std::random_device rd;
  std::mt19937 gen;
  std::uniform_int_distribution<> dis;
  gen.seed(rd());

  std::string label = std::get<0>(bounding_box);
  BoundingBox box = std::get<2>(bounding_box);
  int r = dis(gen);
  int g = dis(gen);
  int b = dis(gen);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointXYZRGB closest_point;
  float l2_closest = 0.0;
  // figure out what the closest point in the pointcloud is to the camera
  // and add all points to the object cloud
  for (int y = box.y1; y < box.y2; ++y) {
    for (int x = box.x1; x < box.x2; ++x) {
      if (pixel_to_point_map.find(std::make_pair(y, x)) !=
          pixel_to_point_map.end()) {
        int index = pixel_to_point_map.at(std::make_pair(y, x));
        pcl::PointXYZRGB point = cloud->points.at(index);

        if (object_cloud->empty()) {
          closest_point = point;
          l2_closest = (point.x - pose.x()) * (point.x - pose.x()) +
                       (point.y - pose.y()) * (point.y - pose.y()) +
                       (point.z - pose.z()) * (point.z - pose.z());
        }

        float l2 = (point.x - pose.x()) * (point.x - pose.x()) +
                   (point.y - pose.y()) * (point.y - pose.y()) +
                   (point.z - pose.z()) * (point.z - pose.z());

        if (l2 < l2_closest) {
          closest_point = point;
          l2_closest = l2;
        }

        point.r = r;
        point.g = g;
        point.b = b;
        object_cloud->push_back(point);
      }
    }
  }

  if (object_cloud->empty()) {
    return object_cloud;
  }

  // filter out the points that are too far away from the closest point to
  // the camera
  float threshold = 1.0;
  object_cloud->points.erase(
    std::remove_if(object_cloud->points.begin(), object_cloud->points.end(),
                   [&closest_point, threshold](const pcl::PointXYZRGB &point) {
                     float l2 =
                       std::sqrt(std::pow(point.x - closest_point.x, 2) +
                                 std::pow(point.y - closest_point.y, 2) +
                                 std::pow(point.z - closest_point.z, 2));
                     return l2 > threshold;
                   }),
    object_cloud->points.end());

  object_cloud->width = object_cloud->points.size();
  object_cloud->height = 1;

  return object_cloud;
}

std::vector<Object> semantic_mapping(
  py::object &net,
  std::vector<std::tuple<cv::Mat, cv::Mat, rtabmap::Transform,
                         std::map<std::pair<int, int>, int>>> &mapping_data,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, std::string &timestamp)
{
  // start by getting the detections, and creating some pretty images
  std::vector<Object> objects; // cloud, closest point, label, confidence
  std::unordered_map<std::string, int> object_counts;
  std::vector<cv::Mat> detection_frames;
  std::vector<cv::Mat> annotated_depth_frames;
  int iter = 0;
  for (const auto &frame : mapping_data) {
    cv::Mat rgb = std::get<0>(frame);
    cv::Mat depth = std::get<1>(frame);
    rtabmap::Transform pose = std::get<2>(frame);
    std::map<std::pair<int, int>, int> pixel_to_point_map = std::get<3>(frame);
    py::array np_array = mat_to_numpy(rgb);
    py::list detections = net.attr("predict")(np_array);

    std::vector<std::tuple<std::string, float, BoundingBox>> bounding_boxes;
    for (auto detection : detections) {
      py::object boxes = detection.attr("boxes");
      py::object names = detection.attr("names");
      py::object speed = detection.attr("speed");
      if (!boxes.is_none()) {

        auto box_list =
          boxes.attr("xyxy")
            .cast<py::list>(); // Example of accessing box coordinates

        for (size_t i = 0; i < py::len(box_list); ++i) {
          py::object box = box_list[i];
          py::object conf_tensor = boxes.attr("conf");
          if (conf_tensor[py::int_(i)].cast<float>() < 0.9) {
            continue;
          }

          std::cout << "Above 90 confidence" << std::endl;

          // Extract the box coordinates
          auto numpy_array =
            box_list[py::int_(i)].attr("cpu")().attr("numpy")();

          // Access individual elements using NumPy indexing.
          int x1 = numpy_array[py::int_(0)].cast<int>();
          int y1 = numpy_array[py::int_(1)].cast<int>();
          int x2 = numpy_array[py::int_(2)].cast<int>();
          int y2 = numpy_array[py::int_(3)].cast<int>();

          cv::Mat detection_frame = rgb.clone();
          cv::Mat depth_frame = depth.clone();

          // Draw the box on the image
          cv::rectangle(detection_frame, cv::Point(x1, y1), cv::Point(x2, y2),
                        cv::Scalar(0, 255, 0), 2);
          cv::rectangle(depth_frame, cv::Point(x1, y1), cv::Point(x2, y2),
                        cv::Scalar(0, 255, 0), 2);

          // Add the label to the image
          py::object names = detection.attr("names");
          py::object classes = boxes.attr("cls");
          std::string label =
            names[py::int_(classes[py::int_(i)])].cast<std::string>();
          cv::putText(detection_frame, label, cv::Point(x1, y1 - 10),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
          cv::putText(depth_frame, label, cv::Point(x1, y1 - 10),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

          // Add the confidence to the image
          float confidence = conf_tensor[py::int_(i)].cast<float>();
          cv::putText(detection_frame, std::to_string(confidence),
                      cv::Point(x1, y1 - 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                      cv::Scalar(0, 255, 0), 2);
          cv::putText(depth_frame, std::to_string(confidence),
                      cv::Point(x1, y1 - 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                      cv::Scalar(0, 255, 0), 2);

          // add the bounding box with the label to the list for later
          if (label != "refrigerator") {
            bounding_boxes.push_back(
              {label, confidence, BoundingBox(x1, y1, x2, y2)});
          } else {
            continue;
          }

          // Add the speed to the image
          py::object speed_py = detection.attr("speed");
          std::string speed = py::str(speed_py).cast<std::string>();
          cv::putText(detection_frame, speed, cv::Point(x1, y1 - 50),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
          cv::putText(depth_frame, speed, cv::Point(x1, y1 - 50),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

          // Add the timestamp to the image
          cv::putText(detection_frame, timestamp, cv::Point(x1, y1 - 70),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
          cv::putText(depth_frame, timestamp, cv::Point(x1, y1 - 70),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

          // Display the image
          // cv::imshow("Image", detection_frame);
          // cv::imshow("Depth", depth);
          // cv::waitKey(1);

          detection_frames.push_back(detection_frame);
          annotated_depth_frames.push_back(depth_frame);
        }
      }
    }
    // cv::destroyAllWindows();

    std::vector<Object> new_objects;
    std::vector<cv::Mat> images;
    for (const auto &elem : bounding_boxes) {
      std::string label = std::get<0>(elem);
      float conf = std::get<1>(elem);
      auto detection = detection_frames.at(iter);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud =
        object_cloud_from_bounding_box(elem, pixel_to_point_map, cloud, pose);

      // calculate the centroid of the object's pointcloud
      pcl::PointXYZ centroid = calculate_centroid(object_cloud);

      // figure out if we've already seen this object before, and if we have
      // then add its points to the pointcloud, recalculate the centroid, and
      // update the label and confidence with the highest confidence value.
      if (objects.empty()) {
        objects.push_back({object_cloud, centroid, label, conf});
      } else {
        bool found = false;
        for (auto &elem : objects) {
          float l2 = std::sqrt(std::pow((elem.centroid.x - centroid.x), 2) +
                               std::pow((elem.centroid.y - centroid.y), 2) +
                               std::pow((elem.centroid.z - centroid.z), 2));
          if (l2 < 0.5) {
            elem.label = conf > elem.confidence ? label : elem.label;
            elem.confidence = conf > elem.confidence ? conf : elem.confidence;
            found = true;
          }
        }
        if (!found) {
          objects.push_back({object_cloud, centroid, label, conf});
          images.push_back(detection);
          new_objects.push_back({object_cloud, centroid, label, conf});
          object_counts[label]++;
        }
      }
      iter++;
    }

    std::string file_path = std::string(PROJECT_PATH) + "/output/" + timestamp +
                            "/landmarks/" + timestamp + ".yaml";
    YAML::Node node;

    for (const auto &object : new_objects) {

      if (object.cloud->size() < 5) {
        std::cout << "Object " << object.label
                  << " has less than 5 points, skipping. Size: "
                  << object.cloud->size() << std::endl;
        std::cout << "centroid: " << object.centroid.x << ", "
                  << object.centroid.y << ", " << object.centroid.z
                  << std::endl;
        continue;
      }
      std::string path = std::string(PROJECT_PATH) + "/output/" + timestamp +
                         "/objects/" + object.label +
                         std::to_string(object_counts[object.label]) + ".pcd";
      pcl::io::savePCDFileBinary(path, *object.cloud);

      if (std::filesystem::exists(file_path)) {
        node = YAML::LoadFile(file_path);
      }

      // Save the new landmark's name and coordinates
      node[object.label + std::to_string(object_counts[object.label])]["x"] =
        object.centroid.x;
      node[object.label + std::to_string(object_counts[object.label])]["y"] =
        object.centroid.y;

      // Open the file for writing
      std::ofstream file(file_path);

      // Write to the file and close it
      file << node;
      file.close();
    }
  }

  iter = 0;
  std::string path =
    std::string(PROJECT_PATH) + "/output/" + timestamp + "/detections/";
  std::cout << "saving images to: " << path << std::endl;
  for (const auto &frame : detection_frames) {
    cv::Mat detection_frame = frame;

    // save the image
    if (std::filesystem::exists(path)) {
      cv::imwrite(path + "detection" + std::to_string(iter) + ".png",
                  detection_frame);
    }
    iter++;
  }
  iter = 0;
  for (const auto &frame : annotated_depth_frames) {
    // save the image
    if (std::filesystem::exists(path)) {
      cv::imwrite(path + "depth_detection" + std::to_string(iter) + ".png",
                  frame);
    }
    iter++;
  }
  std::cout << "Finished semantic mapping" << std::endl;
  return objects;
}

