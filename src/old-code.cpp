#include "rtabmap_vectorize.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <climits>
#include <cfloat>

// Line2D implementation
Line2D::Line2D(cv::Point2f s, cv::Point2f e, float conf, int pid) 
    : start(s), end(e), confidence(conf), plane_id(pid) {}

float Line2D::length() const {
    return cv::norm(end - start);
}

cv::Point2f Line2D::direction() const {
    cv::Point2f dir = end - start;
    float len = length();
    return len > 0 ? dir / len : cv::Point2f(0, 0);
}

// Plane3D implementation
Plane3D::Plane3D() : coefficients(new pcl::ModelCoefficients), 
            points(new pcl::PointCloud<pcl::PointXYZRGB>), id(-1) {}

// VectorMap implementation
VectorMap::VectorMap(float res) : resolution(res) {}

// RTABMapVectorizer implementation
RTABMapVectorizer::RTABMapVectorizer() 
    : plane_distance_threshold_(0.02f)
    , plane_max_iterations_(1000)
    , plane_min_points_(100)
    , line_distance_threshold_(0.01f)
    , line_max_iterations_(1000)
    , line_min_points_(10)
    , merge_distance_threshold_(0.1f)
    , merge_angle_threshold_(5.0f)
    , douglas_peucker_epsilon_(0.05f) {}

VectorMap RTABMapVectorizer::vectorize_point_cloud(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const std::string& timestamp) {
    
    VectorMap result;
    result.timestamp = timestamp;
    
    std::cout << "Starting vectorization of " << cloud->points.size() << " points..." << std::endl;
    
    // Step 3: Segment planes
    std::vector<Plane3D> planes = segment_planes(cloud);
    std::cout << "Found " << planes.size() << " planes" << std::endl;
    
    // Step 4-5: Process each plane
    std::vector<Line2D> all_lines;
    for (size_t i = 0; i < planes.size(); ++i) {
        // Project to 2D
        std::vector<cv::Point2f> points_2d = project_plane_to_2d(planes[i]);
        
        if (points_2d.size() < static_cast<size_t>(line_min_points_)) continue;
        
        // Extract lines
        std::vector<Line2D> plane_lines = extract_lines_from_2d_points(points_2d, i);
        all_lines.insert(all_lines.end(), plane_lines.begin(), plane_lines.end());
    }
    
    std::cout << "Extracted " << all_lines.size() << " initial lines" << std::endl;
    
    // Step 6: Merge and simplify
    std::cout << "Merging similar lines..." << std::endl;
    std::vector<Line2D> merged_lines = merge_similar_lines(all_lines);
    std::cout << "Merged Finished: " << merged_lines.size() << " lines" << std::endl;
    std::vector<Line2D> simplified_lines = simplify_lines(merged_lines);
    
    std::cout << "Final result: " << simplified_lines.size() << " lines" << std::endl;
    
    result.lines = simplified_lines;
    result.bounds = calculate_bounds(simplified_lines);
    
    return result;
}

std::vector<Plane3D> RTABMapVectorizer::segment_planes(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    
    std::vector<Plane3D> planes;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>(*cloud));
    
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(plane_max_iterations_);
    seg.setDistanceThreshold(plane_distance_threshold_);
    
    int plane_id = 0;
    while (remaining_cloud->points.size() > static_cast<size_t>(plane_min_points_)) {
        seg.setInputCloud(remaining_cloud);
        seg.segment(*inliers, *coefficients);
        
        if (inliers->indices.size() < static_cast<size_t>(plane_min_points_)) {
            break;
        }
        
        // Extract plane points
        Plane3D plane;
        plane.id = plane_id++;
        plane.coefficients = pcl::ModelCoefficients::Ptr(new pcl::ModelCoefficients(*coefficients));
        
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane.points);
        
        planes.push_back(plane);
        
        // Remove plane points from remaining cloud
        extract.setNegative(true);
        extract.filter(*remaining_cloud);
    }
    
    return planes;
}

std::vector<cv::Point2f> RTABMapVectorizer::project_plane_to_2d(const Plane3D& plane) {
    std::vector<cv::Point2f> points_2d;
    
    // Simple projection: drop Z coordinate for horizontal planes,
    // or project to best-fit 2D plane for vertical planes
    float c = plane.coefficients->values[2];
    
    // Check if plane is mostly vertical (normal vector has small Z component)
    if (std::abs(c) < 0.7f) {
        // Vertical plane - project to XY
        for (const auto& point : plane.points->points) {
            points_2d.emplace_back(point.x, point.y);
        }
    } else {
        // Horizontal plane - project to XY (same as above for now)
        for (const auto& point : plane.points->points) {
            points_2d.emplace_back(point.x, point.y);
        }
    }
    
    return points_2d;
}

std::vector<Line2D> RTABMapVectorizer::extract_lines_from_2d_points(
    const std::vector<cv::Point2f>& points, int plane_id) {
    
    std::vector<Line2D> lines;
    
    if (points.size() < static_cast<size_t>(line_min_points_)) return lines;
    
    // Convert to format suitable for OpenCV HoughLinesP
    cv::Mat img = cv::Mat::zeros(1000, 1000, CV_8UC1);
    
    // Find bounds and scale points to image
    float min_x = FLT_MAX, min_y = FLT_MAX, max_x = -FLT_MAX, max_y = -FLT_MAX;
    for (const auto& p : points) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
    }
    
    float scale_x = 900.0f / (max_x - min_x + 0.1f);
    float scale_y = 900.0f / (max_y - min_y + 0.1f);
    float scale = std::min(scale_x, scale_y);
    
    // Draw points on image
    for (const auto& p : points) {
        int x = (p.x - min_x) * scale + 50;
        int y = (p.y - min_y) * scale + 50;
        if (x >= 0 && x < img.cols && y >= 0 && y < img.rows) {
            cv::circle(img, cv::Point(x, y), 1, cv::Scalar(255), -1);
        }
    }
    
    // Apply HoughLinesP
    std::vector<cv::Vec4i> hough_lines;
    cv::HoughLinesP(img, hough_lines, 1, CV_PI/180, 20, 30, 10);
    
    // Convert back to world coordinates
    for (const auto& line : hough_lines) {
        cv::Point2f start((line[0] - 50) / scale + min_x, (line[1] - 50) / scale + min_y);
        cv::Point2f end((line[2] - 50) / scale + min_x, (line[3] - 50) / scale + min_y);
        
        lines.emplace_back(start, end, 1.0f, plane_id);
    }
    
    return lines;
}

// std::vector<Line2D> RTABMapVectorizer::merge_similar_lines(const std::vector<Line2D>& lines) {
//     std::cout << "Merging similar lines function" << std::endl;
//     std::vector<Line2D> merged_lines;
//     std::vector<bool> used(lines.size(), false);
//     bool found_merge = false;
    
//     for (size_t i = 0; i < lines.size(); ++i) {
//         if (used[i]) continue;
        
//         Line2D current_line = lines[i];
//         used[i] = true;
        
//         // Find similar lines to merge
//         for (size_t j = i + 1; j < lines.size(); ++j) {
//             if (used[j]) continue;
            
//             if (are_lines_similar(current_line, lines[j])) {
//                 current_line = merge_two_lines(current_line, lines[j]);
//                 used[j] = true;
//                 found_merge = true;
//             }
//         }
        
//         merged_lines.push_back(current_line);
//     }
    
//     // If we found any merges, recursively call the function again
//     if (found_merge) {
//         return merge_similar_lines(merged_lines);
//     }
    
//     return merged_lines;
// }

std::vector<Line2D> RTABMapVectorizer::merge_similar_lines(const std::vector<Line2D>& lines) {
    std::vector<Line2D> merged_lines;
    std::vector<bool> used(lines.size(), false);

    for (size_t i = 0; i < lines.size(); ++i) {
        if (used[i]) continue;

        Line2D current_line = lines[i];
        used[i] = true;

        // Find similar lines to merge
        for (size_t j = i + 1; j < lines.size(); ++j) {
            if (used[j]) continue;

            if (are_lines_similar(current_line, lines[j])) {
                current_line = merge_two_lines(current_line, lines[j]);
                used[j] = true;
            }
        }

        merged_lines.push_back(current_line);
    }

    return merged_lines;
}

std::vector<Line2D> RTABMapVectorizer::simplify_lines(const std::vector<Line2D>& lines) {
    // For now, just return the input lines
    // You could implement Douglas-Peucker or other simplification algorithms here
    return lines;
}

void RTABMapVectorizer::export_to_json(const VectorMap& map, const std::string& filename) {
    Json::Value root;
    root["timestamp"] = map.timestamp;
    root["resolution"] = map.resolution;
    root["bounds"]["x"] = map.bounds.x;
    root["bounds"]["y"] = map.bounds.y;
    root["bounds"]["width"] = map.bounds.width;
    root["bounds"]["height"] = map.bounds.height;
    
    Json::Value lines_array(Json::arrayValue);
    for (const auto& line : map.lines) {
        Json::Value line_obj;
        line_obj["start"]["x"] = line.start.x;
        line_obj["start"]["y"] = line.start.y;
        line_obj["end"]["x"] = line.end.x;
        line_obj["end"]["y"] = line.end.y;
        line_obj["confidence"] = line.confidence;
        line_obj["plane_id"] = line.plane_id;
        line_obj["length"] = line.length();
        lines_array.append(line_obj);
    }
    root["lines"] = lines_array;
    
    std::ofstream file(filename);
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &file);
}

void RTABMapVectorizer::export_to_svg(const VectorMap& map, const std::string& filename) {
    std::ofstream file(filename);
    
    float margin = 50;
    float svg_width = map.bounds.width * 100 + 2 * margin;  // Scale up for better visibility
    float svg_height = map.bounds.height * 100 + 2 * margin;
    
    file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    file << "<svg width=\"" << svg_width << "\" height=\"" << svg_height 
         << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    
    // Background
    file << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    
    // Lines
    for (const auto& line : map.lines) {
        float x1 = (line.start.x - map.bounds.x) * 100 + margin;
        float y1 = (line.start.y - map.bounds.y) * 100 + margin;
        float x2 = (line.end.x - map.bounds.x) * 100 + margin;
        float y2 = (line.end.y - map.bounds.y) * 100 + margin;
        
        // Color based on plane ID
        std::string color = "black";
        if (line.plane_id >= 0) {
            std::vector<std::string> colors = {"red", "blue", "green", "orange", "purple", "brown"};
            color = colors[line.plane_id % colors.size()];
        }
        
        file << "<line x1=\"" << x1 << "\" y1=\"" << y1 
             << "\" x2=\"" << x2 << "\" y2=\"" << y2 
             << "\" stroke=\"" << color << "\" stroke-width=\"2\"/>\n";
    }
    
    file << "</svg>\n";
}

cv::Mat RTABMapVectorizer::visualize_vector_map(const VectorMap& map, cv::Size img_size) {
    cv::Mat img = cv::Mat::zeros(img_size, CV_8UC3);
    
    if (map.lines.empty()) return img;
    
    float scale_x = img_size.width / map.bounds.width;
    float scale_y = img_size.height / map.bounds.height;
    float scale = std::min(scale_x, scale_y) * 0.9f;  // Leave some margin
    
    cv::Point2f offset(
        (img_size.width - map.bounds.width * scale) / 2 - map.bounds.x * scale,
        (img_size.height - map.bounds.height * scale) / 2 - map.bounds.y * scale
    );
    
    // Define colors for different planes
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),    // red
        cv::Scalar(255, 0, 0),    // blue  
        cv::Scalar(0, 255, 0),    // green
        cv::Scalar(0, 165, 255),  // orange
        cv::Scalar(255, 0, 255),  // magenta
        cv::Scalar(42, 42, 165)   // brown
    };
    
    for (const auto& line : map.lines) {
        cv::Point start(
            line.start.x * scale + offset.x,
            line.start.y * scale + offset.y
        );
        cv::Point end(
            line.end.x * scale + offset.x,
            line.end.y * scale + offset.y
        );
        
        cv::Scalar color = cv::Scalar(255, 255, 255); // white default
        if (line.plane_id >= 0) {
            color = colors[line.plane_id % colors.size()];
        }
        
        cv::line(img, start, end, color, 2);
        
        // Draw endpoints
        cv::circle(img, start, 3, color, -1);
        cv::circle(img, end, 3, color, -1);
    }
    
    return img;
}

void RTABMapVectorizer::set_plane_parameters(float distance_thresh, int max_iter, int min_points) {
    plane_distance_threshold_ = distance_thresh;
    plane_max_iterations_ = max_iter;
    plane_min_points_ = min_points;
}

void RTABMapVectorizer::set_line_parameters(float distance_thresh, int max_iter, int min_points) {
    line_distance_threshold_ = distance_thresh;
    line_max_iterations_ = max_iter;
    line_min_points_ = min_points;
}

void RTABMapVectorizer::set_merge_parameters(float distance_thresh, float angle_thresh) {
    merge_distance_threshold_ = distance_thresh;
    merge_angle_threshold_ = angle_thresh;
}

// Helper function implementations
bool RTABMapVectorizer::are_lines_similar(const Line2D& line1, const Line2D& line2) {
    // Check angle similarity
    float angle_diff = angle_between_lines(line1, line2);
    if (angle_diff > merge_angle_threshold_) return false;
    
    // Check distance between lines
    float dist1 = point_to_line_distance(line1.start, line2);
    float dist2 = point_to_line_distance(line1.end, line2);
    float avg_dist_1 = (dist1 + dist2) / 2.0f;

    dist1 = point_to_line_distance(line2.start, line1);
    dist2 = point_to_line_distance(line2.end, line1);
    float avg_dist_2 = (dist1 + dist2) / 2.0f;

    float avg_dist = std::min(avg_dist_1, avg_dist_2);
    
    return avg_dist < merge_distance_threshold_;
}

Line2D RTABMapVectorizer::merge_two_lines(const Line2D& line1, const Line2D& line2) {
    // Simple merge: find the extreme points
    std::vector<cv::Point2f> points = {line1.start, line1.end, line2.start, line2.end};
    
    // Find the two points that are farthest apart
    float max_dist = 0;
    cv::Point2f new_start, new_end;
    
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            float dist = cv::norm(points[i] - points[j]);
            if (dist > max_dist) {
                max_dist = dist;
                new_start = points[i];
                new_end = points[j];
            }
        }
    }
    
    float combined_confidence = (line1.confidence + line2.confidence) / 2.0f;
    return Line2D(new_start, new_end, combined_confidence, line1.plane_id);
}

float RTABMapVectorizer::point_to_line_distance(const cv::Point2f& point, const Line2D& line) {
    cv::Point2f line_vec = line.end - line.start;
    cv::Point2f point_vec = point - line.start;
    
    float line_len_sq = line_vec.dot(line_vec);
    if (line_len_sq < 1e-6) return cv::norm(point_vec);
    
    float t = std::max(0.0f, std::min(1.0f, point_vec.dot(line_vec) / line_len_sq));
    cv::Point2f projection = line.start + t * line_vec;
    
    return cv::norm(point - projection);
}

float RTABMapVectorizer::angle_between_lines(const Line2D& line1, const Line2D& line2) {
    cv::Point2f dir1 = line1.direction();
    cv::Point2f dir2 = line2.direction();
    
    float dot = dir1.dot(dir2);
    dot = std::max(-1.0f, std::min(1.0f, dot)); // Clamp to avoid numerical errors
    
    float angle_rad = std::acos(std::abs(dot)); // abs to get acute angle
    return angle_rad * 180.0f / CV_PI;
}

std::vector<cv::Point2f> RTABMapVectorizer::douglas_peucker_simplify(
    const std::vector<cv::Point2f>& points, float epsilon) {
    // Simple implementation of Douglas-Peucker algorithm
    if (points.size() < 3) return points;
    
    // Find the point with maximum distance from line segment
    float max_dist = 0;
    size_t max_index = 0;
    
    cv::Point2f start = points.front();
    cv::Point2f end = points.back();
    
    for (size_t i = 1; i < points.size() - 1; ++i) {
        // Calculate distance from point to line segment
        cv::Point2f line_vec = end - start;
        cv::Point2f point_vec = points[i] - start;
        
        float line_len_sq = line_vec.dot(line_vec);
        if (line_len_sq < 1e-6) continue;
        
        float t = std::max(0.0f, std::min(1.0f, point_vec.dot(line_vec) / line_len_sq));
        cv::Point2f projection = start + t * line_vec;
        float dist = cv::norm(points[i] - projection);
        
        if (dist > max_dist) {
            max_dist = dist;
            max_index = i;
        }
    }
    
    // If max distance is greater than epsilon, recursively simplify
    if (max_dist > epsilon) {
        // Recursive call on first part
        std::vector<cv::Point2f> first_part(points.begin(), points.begin() + max_index + 1);
        std::vector<cv::Point2f> first_result = douglas_peucker_simplify(first_part, epsilon);
        
        // Recursive call on second part
        std::vector<cv::Point2f> second_part(points.begin() + max_index, points.end());
        std::vector<cv::Point2f> second_result = douglas_peucker_simplify(second_part, epsilon);
        
        // Combine results (remove duplicate point at junction)
        std::vector<cv::Point2f> result = first_result;
        result.insert(result.end(), second_result.begin() + 1, second_result.end());
        return result;
    } else {
        // If no point is far enough, return just endpoints
        return {start, end};
    }
}

cv::Rect2f RTABMapVectorizer::calculate_bounds(const std::vector<Line2D>& lines) {
    if (lines.empty()) return cv::Rect2f(0, 0, 1, 1);
    
    float min_x = FLT_MAX, min_y = FLT_MAX, max_x = -FLT_MAX, max_y = -FLT_MAX;
    
    for (const auto& line : lines) {
        min_x = std::min({min_x, line.start.x, line.end.x});
        min_y = std::min({min_y, line.start.y, line.end.y});
        max_x = std::max({max_x, line.start.x, line.end.x});
        max_y = std::max({max_y, line.start.y, line.end.y});
    }
    
    return cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
}