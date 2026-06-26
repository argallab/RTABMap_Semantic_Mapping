#include "rtabmap_vectorize.hpp"
#include <algorithm>
#include <cfloat>
#include <climits>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>

#include <cmath>       // for fmod, abs
#include <sstream>     // for std::stringstream  
#include <iomanip>     // for std::hex, std::setfill, std::setw

Line2D::Line2D(cv::Point2f s, cv::Point2f e, float conf, int pid)
  : start(s), end(e), confidence(conf), plane_id(pid)
{
}

VectorMap::VectorMap() : bounds(0, 0, 0, 0), timestamp("") {}

Plane3D::Plane3D()
  : coefficients(new pcl::ModelCoefficients),
    points(new pcl::PointCloud<pcl::PointXYZRGB>), id(-1)
{
}

// RTABMapVectorizer implementation
RTABMapVectorizer::RTABMapVectorizer()
{
  plane_distance_threshold_ = 0.02f;
  plane_max_iterations_ = 1000;
  plane_min_points_ = 100;
  line_distance_threshold_ = 0.01f;
  line_max_iterations_ = 1000;
  line_min_points_ = 10;
  merge_distance_threshold_ = 0.1f;
  merge_angle_threshold_ = 5.0f;
}

RTABMapVectorizer::RTABMapVectorizer(
  float plane_distance_threshold, int plane_max_iterations,
  int plane_min_points, float line_distance_threshold, int line_max_iterations,
  int line_min_points, float merge_distance_threshold,
  float merge_angle_threshold)
{
  plane_distance_threshold_ = plane_distance_threshold;
  plane_max_iterations_ = plane_max_iterations;
  plane_min_points_ = plane_min_points;
  line_distance_threshold_ = line_distance_threshold;
  line_max_iterations_ = line_max_iterations;
  line_min_points_ = line_min_points;
  merge_distance_threshold_ = merge_distance_threshold;
  merge_angle_threshold_ = merge_angle_threshold;
}

// void RTABMapVectorizer::export_to_svg(const VectorMap &map,
//                                       const std::string &filename)
// {
//   std::ofstream file(filename);

//   float margin = 50;
//   float svg_width =
//     map.bounds.width * 100 + 2 * margin; // Scale up for better visibility
//   float svg_height = map.bounds.height * 100 + 2 * margin;

//   file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
//   file << "<svg width=\"" << svg_width << "\" height=\"" << svg_height
//        << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";

//   // Background
//   file << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";

//   // Lines
//   for (const auto &line : map.lines) {
//     float x1 = (line.start.x - map.bounds.x) * 100 + margin;
//     float y1 = (line.start.y - map.bounds.y) * 100 + margin;
//     float x2 = (line.end.x - map.bounds.x) * 100 + margin;
//     float y2 = (line.end.y - map.bounds.y) * 100 + margin;

//     // Generate maximally distinct color for large number of categories
//     std::string color = "black";
//     if (line.plane_id >= 0) {

//       std::cout << "Line " << line.plane_id << ": ";
//       // Multi-dimensional color space approach for maximum distinction
//       int id = line.plane_id;
      
//       // Use multiple prime numbers to spread colors across different dimensions
//       int hue_steps = 37;      // Prime number for hue distribution
//       int sat_steps = 5;       // Different saturation levels
//       int val_steps = 4;       // Different brightness levels
      
//       // Calculate HSV components with maximum spread
//       double hue = (id % hue_steps) * (360.0 / hue_steps);
//       double saturation = 0.4 + (((id / hue_steps) % sat_steps) * 0.6 / sat_steps);
//       double value = 0.5 + (((id / (hue_steps * sat_steps)) % val_steps) * 0.5 / val_steps);
      
//       // For every 740 IDs (37*5*4), shift the hue slightly to create new variants
//       int cycle = id / (hue_steps * sat_steps * val_steps);
//       hue = fmod(hue + (cycle * 17.3), 360.0); // 17.3 is chosen to avoid repeating patterns
      
//       // Convert HSV to RGB
//       double c = value * saturation;
//       double x = c * (1 - abs(fmod(hue / 60.0, 2) - 1));
//       double m = value - c;
      
//       double r_prime, g_prime, b_prime;
      
//       if (hue < 60) {
//         r_prime = c; g_prime = x; b_prime = 0;
//       } else if (hue < 120) {
//         r_prime = x; g_prime = c; b_prime = 0;
//       } else if (hue < 180) {
//         r_prime = 0; g_prime = c; b_prime = x;
//       } else if (hue < 240) {
//         r_prime = 0; g_prime = x; b_prime = c;
//       } else if (hue < 300) {
//         r_prime = x; g_prime = 0; b_prime = c;
//       } else {
//         r_prime = c; g_prime = 0; b_prime = x;
//       }
      
//       int r = static_cast<int>((r_prime + m) * 255);
//       int g = static_cast<int>((g_prime + m) * 255);
//       int b = static_cast<int>((b_prime + m) * 255);
      
//       // Ensure minimum contrast against white background
//       if (r + g + b < 300) { // If color is too light
//         r = static_cast<int>(r * 0.6);
//         g = static_cast<int>(g * 0.6);
//         b = static_cast<int>(b * 0.6);
//       }
      
//       // Convert to hex color string
//       std::stringstream color_stream;
//       color_stream << "#" << std::hex << std::setfill('0') 
//                    << std::setw(2) << r 
//                    << std::setw(2) << g 
//                    << std::setw(2) << b;
//       color = color_stream.str();
//     }

//     // Make lines thicker and add plane ID as text for debugging
//     file << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2
//          << "\" y2=\"" << y2 << "\" stroke=\"" << color
//          << "\" stroke-width=\"3\"/>\n";
    
//     // Add plane ID label at midpoint of line for verification
//     float mid_x = (x1 + x2) / 2;
//     float mid_y = (y1 + y2) / 2;
//     file << "<text x=\"" << mid_x << "\" y=\"" << mid_y 
//          << "\" font-family=\"Arial\" font-size=\"8\" fill=\"black\" "
//          << "text-anchor=\"middle\" dominant-baseline=\"middle\">"
//          << line.plane_id << "</text>\n";
//   }

//   file << "</svg>\n";
// }

void RTABMapVectorizer::export_to_svg(const VectorMap &map,
                                      const std::string &filename)
{
  std::ofstream file(filename);

  float margin = 50;
  float svg_width =
    map.bounds.width * 100 + 2 * margin; // Scale up for better visibility
  float svg_height = map.bounds.height * 100 + 2 * margin;

  file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  file << "<svg width=\"" << svg_width << "\" height=\"" << svg_height
       << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";

  // Background
  file << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";

  // Lines
  for (const auto &line : map.lines) {
    float x1 = (line.start.x - map.bounds.x) * 100 + margin;
    float y1 = (line.start.y - map.bounds.y) * 100 + margin;
    float x2 = (line.end.x - map.bounds.x) * 100 + margin;
    float y2 = (line.end.y - map.bounds.y) * 100 + margin;

    // Simple black lines
    file << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2
         << "\" y2=\"" << y2 << "\" stroke=\"black"
         << "\" stroke-width=\"3\"/>\n";
  }

  file << "</svg>\n";
}

cv::Rect2f RTABMapVectorizer::calculate_bounds(const std::vector<Line2D> &lines)
{

  if (lines.empty()) {
    return cv::Rect2f(0, 0, 1, 1);
  }

  float min_x = FLT_MAX, min_y = FLT_MAX;
  float max_x = -FLT_MAX, max_y = -FLT_MAX;

  for (const auto &line : lines) {
    min_x = std::min({min_x, line.start.x, line.end.x});
    min_y = std::min({min_y, line.start.y, line.end.y});
    max_x = std::max({max_x, line.start.x, line.end.x});
    max_y = std::max({max_y, line.start.y, line.end.y});
  }

  return cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
}

VectorMap RTABMapVectorizer::vectorize_point_cloud(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const std::string &timestamp)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud = filter_cloud(cloud);

  auto planes = segment_planes(filtered_cloud);
  std::cout << "Found " << planes.size() << " planes" << std::endl;
  std::vector<Line2D> all_lines;
  for (size_t i = 0; i < planes.size(); ++i) {
    // Project to 2D
    std::vector<cv::Point2f> points_2d = project_plane_to_2d(planes[i]);

    if (points_2d.size() < static_cast<size_t>(line_min_points_))
      continue;

    // Extract lines
    std::vector<Line2D> plane_lines =
      extract_lines_from_2d_points(points_2d, i);
    all_lines.insert(all_lines.end(), plane_lines.begin(), plane_lines.end());
  }

  //   // Merge similar lines
  std::cout << "Extracted " << all_lines.size() << " lines before merging."
            << std::endl;
  std::vector<Line2D> merged_lines = merge_lines(all_lines);

  std::cout << "Merged " << merged_lines.size() << " lines after merging."
            << std::endl;

  //   // Filter short lines
  //   std::vector<Line2D> filtered_lines =
  //     filter_short_lines(merged_lines, 0.5f); // min length 0.5m

  VectorMap result;
  result.timestamp = timestamp;
  result.lines = merged_lines;
  result.bounds = calculate_bounds(merged_lines);

  export_to_svg(result, "vector_map-region-grow-rgb.svg");

  return result;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
RTABMapVectorizer::filter_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
  voxel_filter.setInputCloud(cloud);
  voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
  voxel_filter.filter(*filtered_cloud);
  return filtered_cloud;
}


std::vector<Line2D>
RTABMapVectorizer::merge_lines(const std::vector<Line2D> &lines)
{
    std::cout << "Merging lines by selecting longest... Input: " << lines.size() << " lines" << std::endl;
    
    if (lines.empty()) return lines;
    
    std::vector<bool> processed(lines.size(), false);
    std::vector<Line2D> merged_lines;
    bool any_merged = false;

    for (size_t i = 0; i < lines.size(); i++) {
        if (processed[i]) continue;
        
        // Start a new group with the current line
        std::vector<size_t> group_indices;
        group_indices.push_back(i);
        processed[i] = true;
        
        // Find all lines similar to lines[i]
        for (size_t j = i + 1; j < lines.size(); j++) {
            if (!processed[j] && are_lines_similar(lines[i], lines[j])) {
                group_indices.push_back(j);
                processed[j] = true;
            }
        }
        
        // Select the longest line from the group
        if (group_indices.size() > 1) {
            size_t longest_index = group_indices[0];
            double longest_length = calculate_line_length(lines[longest_index]);
            
            for (size_t k = 1; k < group_indices.size(); k++) {
                double current_length = calculate_line_length(lines[group_indices[k]]);
                if (current_length > longest_length) {
                    longest_length = current_length;
                    longest_index = group_indices[k];
                }
            }
            
            std::cout << "Found group of " << group_indices.size() << " lines with plane ID " 
                      << lines[i].plane_id << ", selected longest (length: " << longest_length << ")" << std::endl;
            
            merged_lines.push_back(lines[longest_index]);
            any_merged = true;
        } else {
            // Keep the single line as-is
            merged_lines.push_back(lines[i]);
        }
    }

    std::cout << "Result: " << merged_lines.size() << " lines after merging." << std::endl;

    // Recursively merge if we made changes (in case there are still groups to merge)
    if (any_merged && merged_lines.size() > 1) {
        std::cout << "Recursing down..." << std::endl;
        merged_lines = merge_lines(merged_lines);
        std::cout << "Recursing up..." << std::endl;
    }

    return merged_lines;
}

// Helper function to calculate line length
double RTABMapVectorizer::calculate_line_length(const Line2D &line)
{
    double dx = line.end.x - line.start.x;
    double dy = line.end.y - line.start.y;
    return sqrt(dx * dx + dy * dy);
}

// Keep your existing similarity function
bool RTABMapVectorizer::are_lines_similar(const Line2D &line1, const Line2D &line2)
{
    if (line1.plane_id != line2.plane_id)
        return false;
    return true;
}
// std::vector<Line2D>
// RTABMapVectorizer::filter_short_lines(const std::vector<Line2D> &lines,
//                                       float min_length)
// {
//   std::vector<Line2D> filtered_lines;

//   for (const auto &line : lines) {
//     if (line.length() >= min_length) {
//       filtered_lines.push_back(line);
//     }
//   }

//   return filtered_lines;
// }

///////////////////////////////////////////////////////////////////////////////////////////////////////////

// TEST FUNCTIONS

///////////////////////////////////////////////////////////////////////////////////////////////////////////

// std::vector<Plane3D>
// RTABMapVectorizer::segment_planes(pcl::PointCloud<pcl::PointXYZRGB>::Ptr
// cloud)
// {
//     std::vector<Plane3D> planes;

//     // Step 1: Compute normals (required for RegionGrowing)
//     pcl::search::Search<pcl::PointXYZRGB>::Ptr tree(
//         new pcl::search::KdTree<pcl::PointXYZRGB>());

//     pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
//     ne.setSearchMethod(tree);
//     ne.setInputCloud(cloud);
//     ne.setKSearch(30); // number of neighbors for normal estimation

//     pcl::PointCloud<pcl::Normal>::Ptr normals(new
//     pcl::PointCloud<pcl::Normal>); ne.compute(*normals);

//     // Step 2: Region Growing Segmentation (no RGB)
//     pcl::RegionGrowing<pcl::PointXYZRGB, pcl::Normal> reg;
//     reg.setMinClusterSize(plane_min_points_);
//     reg.setMaxClusterSize(cloud->size());
//     reg.setSearchMethod(tree);
//     reg.setNumberOfNeighbours(30);
//     reg.setInputCloud(cloud);
//     reg.setInputNormals(normals);
//     reg.setSmoothnessThreshold(pcl::deg2rad(3.0f));  // angle threshold
//     between normals reg.setCurvatureThreshold(1.0f);                 //
//     curvature limit

//     std::vector<pcl::PointIndices> clusters;
//     reg.extract(clusters);

//     // Step 3: Optionally fit planes to each region for coefficients
//     int plane_id = 0;
//     for (const auto &indices : clusters)
//     {
//         if (indices.indices.size() < static_cast<size_t>(plane_min_points_))
//             continue;

//         // Extract cluster points
//         pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new
//         pcl::PointCloud<pcl::PointXYZRGB>); pcl::copyPointCloud(*cloud,
//         indices, *cluster_cloud);

//         // Fit a plane to the cluster
//         pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
//         pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//         pcl::SACSegmentation<pcl::PointXYZRGB> seg;
//         seg.setOptimizeCoefficients(true);
//         seg.setModelType(pcl::SACMODEL_PLANE);
//         seg.setMethodType(pcl::SAC_RANSAC);
//         seg.setDistanceThreshold(plane_distance_threshold_);
//         seg.setInputCloud(cluster_cloud);
//         seg.segment(*inliers, *coefficients);

//         if (inliers->indices.size() < static_cast<size_t>(plane_min_points_))
//             continue;

//         // Store plane
//         Plane3D plane;
//         plane.id = plane_id++;
//         plane.coefficients = coefficients;
//         plane.points = cluster_cloud;
//         planes.push_back(plane);
//     }

//     return planes;
// }

std::vector<Plane3D>
RTABMapVectorizer::segment_planes(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
  std::vector<Plane3D> planes;

  // Step 1: Build KD-Tree for search
  // pcl::search::Search<pcl::PointXYZRGB>::Ptr tree =
  //   boost::make_shared<pcl::search::KdTree<pcl::PointXYZRGB>>();
  // tree->setInputCloud(cloud);

  pcl::search::Search<pcl::PointXYZRGB>::Ptr tree(
    new pcl::search::KdTree<pcl::PointXYZRGB>());
  tree->setInputCloud(cloud);

  // Step 2: Region Growing RGB Segmentation
  pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
  reg.setInputCloud(cloud);
  reg.setSearchMethod(tree);
  reg.setDistanceThreshold(
    plane_distance_threshold_);   // spatial distance threshold
  reg.setPointColorThreshold(6);  // color difference threshold per channel
  reg.setRegionColorThreshold(5); // color difference threshold between regions
  reg.setMinClusterSize(plane_min_points_);

  std::vector<pcl::PointIndices> clusters;
  reg.extract(clusters);

  // Step 3: For each cluster, estimate plane coefficients and store in Plane3D
  int plane_id = 0;
  for (const auto &indices : clusters) {
    if (indices.indices.size() < static_cast<size_t>(plane_min_points_))
      continue;

    // Extract cluster points
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud, indices, *cluster_cloud);

    // Fit a plane to the cluster
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(plane_distance_threshold_);
    seg.setInputCloud(cluster_cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() < static_cast<size_t>(plane_min_points_))
      continue;

    // Store plane
    Plane3D plane;
    plane.id = plane_id++;
    plane.coefficients = coefficients;
    plane.points = cluster_cloud; // keep all cluster points
    planes.push_back(plane);
  }

  return planes;
}

// std::vector<Plane3D>
// RTABMapVectorizer::segment_planes(pcl::PointCloud<pcl::PointXYZRGB>::Ptr
// cloud)
// {

// pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_cloud(
//   new pcl::PointCloud<pcl::PointXYZRGB>(*cloud));

// pcl::SACSegmentation<pcl::PointXYZRGB> seg;
// pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
// pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
// pcl::ExtractIndices<pcl::PointXYZRGB> extract;

// seg.setOptimizeCoefficients(true);
// seg.setModelType(pcl::SACMODEL_PLANE);
// seg.setMethodType(pcl::SAC_RANSAC);
// seg.setMaxIterations(plane_max_iterations_);
// seg.setDistanceThreshold(plane_distance_threshold_);

// int plane_id = 0;
// while (remaining_cloud->points.size() >
//        static_cast<size_t>(plane_min_points_)) {
//   seg.setInputCloud(remaining_cloud);
//   seg.segment(*inliers, *coefficients);

//   if (inliers->indices.size() < static_cast<size_t>(plane_min_points_)) {
//     break;
//   }

//   // Extract plane points
//   Plane3D plane;
//   plane.id = plane_id++;
//   plane.coefficients =
//     pcl::ModelCoefficients::Ptr(new pcl::ModelCoefficients(*coefficients));

//   extract.setInputCloud(remaining_cloud);
//   extract.setIndices(inliers);
//   extract.setNegative(false);
//   extract.filter(*plane.points);

//   planes.push_back(plane);

//   // Remove plane points from remaining cloud
//   extract.setNegative(true);
//   extract.filter(*remaining_cloud);
// }

// return planes;
// }

std::vector<cv::Point2f>
RTABMapVectorizer::project_plane_to_2d(const Plane3D &plane)
{
  std::vector<cv::Point2f> points_2d;

  // Simple projection: drop Z coordinate for horizontal planes,
  // or project to best-fit 2D plane for vertical planes
  float c = plane.coefficients->values[2];

  // Check if plane is mostly vertical (normal vector has small Z component)
  if (std::abs(c) < 0.7f) {
    // Vertical plane - project to XY
    for (const auto &point : plane.points->points) {
      points_2d.emplace_back(point.x, point.y);
    }
  } else {
    // Horizontal plane - project to XY (same as above for now)
    for (const auto &point : plane.points->points) {
      points_2d.emplace_back(point.x, point.y);
    }
  }

  return points_2d;
}

std::vector<Line2D> RTABMapVectorizer::extract_lines_from_2d_points(
  const std::vector<cv::Point2f> &points, int plane_id)
{

  std::vector<Line2D> lines;

  if (points.size() < static_cast<size_t>(line_min_points_))
    return lines;

  // Convert to format suitable for OpenCV HoughLinesP
  cv::Mat img = cv::Mat::zeros(1000, 1000, CV_8UC1);

  // Find bounds and scale points to image
  float min_x = FLT_MAX, min_y = FLT_MAX, max_x = -FLT_MAX, max_y = -FLT_MAX;
  for (const auto &p : points) {
    min_x = std::min(min_x, p.x);
    min_y = std::min(min_y, p.y);
    max_x = std::max(max_x, p.x);
    max_y = std::max(max_y, p.y);
  }

  float scale_x = 900.0f / (max_x - min_x + 0.1f);
  float scale_y = 900.0f / (max_y - min_y + 0.1f);
  float scale = std::min(scale_x, scale_y);

  // Draw points on image
  for (const auto &p : points) {
    int x = (p.x - min_x) * scale + 50;
    int y = (p.y - min_y) * scale + 50;
    if (x >= 0 && x < img.cols && y >= 0 && y < img.rows) {
      cv::circle(img, cv::Point(x, y), 1, cv::Scalar(255), -1);
    }
  }

  // Apply HoughLinesP
  std::vector<cv::Vec4i> hough_lines;
  cv::HoughLinesP(img, hough_lines, 1, CV_PI / 180, 20, 30, 10);

  // Convert back to world coordinates
  for (const auto &line : hough_lines) {
    cv::Point2f start((line[0] - 50) / scale + min_x,
                      (line[1] - 50) / scale + min_y);
    cv::Point2f end((line[2] - 50) / scale + min_x,
                    (line[3] - 50) / scale + min_y);

    lines.emplace_back(start, end, 1.0f, plane_id);
  }

  return lines;
}