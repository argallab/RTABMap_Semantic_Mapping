#include "graph_vectorize.hpp"
#include <iostream>
#include <sstream>

pid_t open_svg_file(const std::string &filename)
{
  // Convert SVG to PNG first
  std::string png_file = filename;
  png_file.replace(png_file.find(".svg"), 4, ".png");
  
  std::string convert_cmd = "rsvg-convert " + filename + " -o " + png_file + " 2>/dev/null";
  int ret = system(convert_cmd.c_str());
  (void) ret;
  
  // Launch in background
  std::string view_cmd = "feh --geometry 1000x800 --auto-zoom " + png_file + " >/dev/null 2>&1 &";
  ret = system(view_cmd.c_str());
  (void) ret;
  
  return -1;  // Can't track PID easily
}

void close_svg_viewer([[maybe_unused]] pid_t pid)
{
  // Kill all feh processes showing cluster previews
  int ret = system("pkill -f 'feh.*cluster_preview' 2>/dev/null");
  (void) ret;
}

GraphVectorizer::GraphVectorizer() {}

GraphVectorizer::~GraphVectorizer() {}

// void GraphVectorizer::vectorizeGraph(
//   const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
//   const std::string &timestamp)
// {
//   std::cout << "Vectorizing graph for timestamp: " << timestamp
//             << " with point cloud size: " << cloud->points.size() << std::endl;

//   std::cout << "Filtering point cloud..." << std::endl;
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud = filter_cloud(cloud);

//   std::cout << "Assembling clusters from filtered cloud of size: "
//             << filtered_cloud->points.size() << std::endl;
//   std::vector<std::vector<Point2D>> clusters = assemble_clusters(filtered_cloud);

//   // Prompt user for cluster selection
//   std::vector<bool> selected_clusters = promptClusterSelection(clusters);

//   std::cout << "\nCreating graphs from selected clusters..." << std::endl;
//   std::vector<std::vector<Edge>> cluster_graphs;

//   for (size_t i = 0; i < clusters.size(); ++i) {
//     if (!selected_clusters[i]) {
//       std::cout << "Skipping cluster " << i << " (not selected)" << std::endl;
//       cluster_graphs.push_back(std::vector<Edge>());
//       continue;
//     }

//     std::cout << "Processing cluster " << i << " with " << clusters[i].size()
//               << " points" << std::endl;

//     std::vector<Edge> cluster_edges = create_graph(clusters[i]);
//     // simplify_graph(cluster_edges, clusters[i]);
//     cluster_graphs.push_back(cluster_edges);
//   }

//   // Export to SVG for visualization
//   export_graph_to_svg(clusters, cluster_graphs, "/app/output/graph.svg");
// }

void GraphVectorizer::vectorizeGraph(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
  const std::string &timestamp)
{

  std::cout << "Vectorizing graph for timestamp: " << timestamp
            << " with point cloud size: " << cloud->points.size() << std::endl;

  std::cout << "Filtering point cloud..." << std::endl;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud = filter_cloud(cloud);

  std::cout << "Assembling clusters from filtered cloud of size: "
            << filtered_cloud->points.size() << std::endl;
  std::vector<std::vector<Point2D>> clusters =
    assemble_clusters(filtered_cloud);

  std::cout << "Creating graphs from clusters..." << std::endl;
  std::vector<std::vector<Edge>> cluster_graphs;

  for (size_t i = 0; i < clusters.size(); ++i) {
  // for (size_t i = 0; i < 2; ++i) {
    std::cout << "Processing cluster " << i << " with " << clusters[i].size()
              << " points" << std::endl;

    if (clusters[i].size() < 75) {
      std::cout << "  Skipping cluster with < 75 points" << std::endl;
      cluster_graphs.push_back(std::vector<Edge>());
      continue;
    }

    std::vector<Edge> cluster_edges = create_graph(clusters[i]);
    // simplify_graph(cluster_edges, clusters[i]);
    cluster_graphs.push_back(cluster_edges);
  }

  // Export to SVG for visualization
  export_graph_to_svg(clusters, cluster_graphs, "/app/output/graph.svg");
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
GraphVectorizer::filter_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
  voxel_filter.setInputCloud(cloud);
  voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);
  voxel_filter.filter(*filtered_cloud);
  return filtered_cloud;
}

std::vector<std::vector<Point2D>> GraphVectorizer::assemble_clusters(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{

  std::vector<std::vector<double>> py_points;

  for (const auto &p : cloud->points) {
    py_points.push_back({static_cast<double>(p.x), static_cast<double>(p.y)});
  }

  // Import sklearn DBSCAN
  py::object sklearn = py::module::import("sklearn.cluster");
  py::object DBSCAN = sklearn.attr("DBSCAN");

  // Create DBSCAN instance
  py::object db = DBSCAN("eps"_a = 0.1, "min_samples"_a = 5);

  // Fit DBSCAN
  py::object labels = db.attr("fit_predict")(py_points);

  // Convert result back to C++
  std::vector<int> cluster_labels = labels.cast<std::vector<int>>();

  // Create clusters from the labels
  std::map<int, std::vector<Point2D>> cluster_map;

  for (size_t i = 0; i < cluster_labels.size(); ++i) {
    int label = cluster_labels[i];
    if (label != -1) {
      Point2D point(cloud->points[i].x, cloud->points[i].y, label);
      cluster_map[label].push_back(point);
    }
  }

  auto clusters = std::vector<std::vector<Point2D>>();

  for (const auto &pair : cluster_map) {
    clusters.push_back(pair.second);
  }

  return clusters;
}

std::vector<std::vector<Point2D>>
GraphVectorizer::assemble_clusters(const std::vector<Point2D> &cloud)
{

  std::vector<std::vector<double>> py_points;

  for (const auto &p : cloud) {
    py_points.push_back({static_cast<double>(p.x), static_cast<double>(p.y)});
  }

  // Import sklearn DBSCAN
  py::object sklearn = py::module::import("sklearn.cluster");
  py::object DBSCAN = sklearn.attr("DBSCAN");

  // Create DBSCAN instance
  py::object db = DBSCAN("eps"_a = 0.05, "min_samples"_a = 5);

  // Fit DBSCAN
  py::object labels = db.attr("fit_predict")(py_points);

  // Convert result back to C++
  std::vector<int> cluster_labels = labels.cast<std::vector<int>>();

  // Create clusters from the labels
  std::map<int, std::vector<Point2D>> cluster_map;

  for (size_t i = 0; i < cluster_labels.size(); ++i) {
    int label = cluster_labels[i];
    if (label != -1) {
      Point2D point(cloud[i].x, cloud[i].y, label);
      cluster_map[label].push_back(point);
    }
  }

  auto clusters = std::vector<std::vector<Point2D>>();

  for (const auto &pair : cluster_map) {
    clusters.push_back(pair.second);
  }

  return clusters;
}

std::vector<Edge> GraphVectorizer::create_graph(std::vector<Point2D> &cluster)
{
  // if (cluster.empty()) {
  //   return {};
  // }

  // int cluster_id = cluster[0].cluster_id;
  // std::vector<std::vector<Point2D>> p_clusters = assemble_clusters(cluster);
  // cluster.clear();
  // for (const auto &p_cluster : p_clusters) {
  //   auto centroid_x = 0.0;
  //   auto centroid_y = 0.0;
  //   for (const auto &p : p_cluster) {
  //     centroid_x += p.x;
  //     centroid_y += p.y;
  //   }
  //   centroid_x /= p_cluster.size();
  //   centroid_y /= p_cluster.size();

  //   cluster.push_back(Point2D(centroid_x, centroid_y, cluster_id));
  // }

  // std::vector<Edge> cluster_edges;

  // for (size_t i = 0; i < cluster.size(); ++i) {
  //   for (size_t j = i + 1; j < cluster.size(); ++j) {
  //     float distance = cluster[i].distanceTo(cluster[j]);
  //     cluster_edges.emplace_back(i, j, distance);
  //   }
  // }

  // std::cout << "    Created " << cluster_edges.size()
  //           << " edges between centroids" << std::endl;

  if (cluster.empty()) {
    return {};
  }

  // int cluster_id = cluster[0].cluster_id;
  // std::vector<std::vector<Point2D>> p_clusters = assemble_clusters(cluster);
  // cluster.clear();
  
  // for (const auto &p_cluster : p_clusters) {
  //   auto centroid_x = 0.0;
  //   auto centroid_y = 0.0;
  //   for (const auto &p : p_cluster) {
  //     centroid_x += p.x;
  //     centroid_y += p.y;
  //   }
  //   centroid_x /= p_cluster.size();
  //   centroid_y /= p_cluster.size();

  //   cluster.push_back(Point2D(centroid_x, centroid_y, cluster_id));
  // }

  // Create Delaunay triangulation
  Delaunay dt;
  std::vector<Delaunay::Vertex_handle> vertex_handles;
  
  for (const auto &point : cluster) {
    auto vh = dt.insert(CGALPoint(point.x, point.y));
    vertex_handles.push_back(vh);
  }

  // Extract edges from triangulation
  std::vector<Edge> cluster_edges;
  std::set<std::pair<size_t, size_t>> edge_set; // Avoid duplicates

  for (auto eit = dt.finite_edges_begin(); eit != dt.finite_edges_end(); ++eit) {
    auto vh1 = eit->first->vertex((eit->second + 1) % 3);
    auto vh2 = eit->first->vertex((eit->second + 2) % 3);
    
    // Find indices in original cluster vector
    auto it1 = std::find(vertex_handles.begin(), vertex_handles.end(), vh1);
    auto it2 = std::find(vertex_handles.begin(), vertex_handles.end(), vh2);
    
    if (it1 != vertex_handles.end() && it2 != vertex_handles.end()) {
      size_t i = std::distance(vertex_handles.begin(), it1);
      size_t j = std::distance(vertex_handles.begin(), it2);
      
      if (i > j) std::swap(i, j);
      
      if (edge_set.insert({i, j}).second) {
        float distance = cluster[i].distanceTo(cluster[j]);
        cluster_edges.emplace_back(i, j, distance);
      }
    }
  }

  std::cout << "    Created " << cluster_edges.size()
            << " edges using Delaunay triangulation" << std::endl;

  std::cout << "    Simplifying graph with " << cluster_edges.size()
          << " edges using Prim's MST algorithm with angle continuity..." << std::endl;

// Create adjacency list representation using existing Edge struct
std::map<int, std::vector<Edge>> adjacency_list;

// Build adjacency list from edges
for (const auto &edge : cluster_edges) {
  adjacency_list[edge.point1_idx].push_back(edge);
  // For undirected graph, add reverse edge
  adjacency_list[edge.point2_idx].emplace_back(edge.point2_idx,
                                               edge.point1_idx, edge.length);
}

// Get all unique vertices
std::set<int> all_vertices;
for (const auto &edge : cluster_edges) {
  all_vertices.insert(edge.point1_idx);
  all_vertices.insert(edge.point2_idx);
}

if (all_vertices.empty()) {
  return {};
}

// Helper function to calculate angle penalty
auto calculate_angle_penalty = [&](int prev_vertex, int current_vertex, int next_vertex) -> float {
  float dx1 = cluster[current_vertex].x - cluster[prev_vertex].x;
  float dy1 = cluster[current_vertex].y - cluster[prev_vertex].y;
  
  float dx2 = cluster[next_vertex].x - cluster[current_vertex].x;
  float dy2 = cluster[next_vertex].y - cluster[current_vertex].y;
  
  float len1 = sqrt(dx1*dx1 + dy1*dy1);
  float len2 = sqrt(dx2*dx2 + dy2*dy2);
  
  if (len1 < 0.001f || len2 < 0.001f) return 1.0f;
  
  dx1 /= len1; dy1 /= len1;
  dx2 /= len2; dy2 /= len2;
  
  float dot = dx1*dx2 + dy1*dy2;
  dot = std::max(-1.0f, std::min(1.0f, dot));
  
  // Calculate angle in degrees
  float angle_radians = acos(dot);
  float angle_degrees = angle_radians * 180.0f / M_PI;
  
  // Manhattan world: allow 0° and 90° (with tolerance)
  float TOLERANCE = 15.0f;
  float PENALTY = 10.0f;
  
  // Check if close to 0° (straight)
  if (angle_degrees < TOLERANCE) {
    return 1.0f;
  }
  
  // Check if close to 90° (right angle)
  if (abs(angle_degrees - 90.0f) < TOLERANCE) {
    return 2.0f;
  }
  
  // Check if close to 180° (U-turn)
  if (angle_degrees > (180.0f - TOLERANCE)) {
    return PENALTY * 2.0f;
  }
  
  // Anything else gets penalized
  return PENALTY;
};

// Find top-left most node (minimum x+y, prioritizing top-left corner)
int start_vertex = -1;
float min_score = FLT_MAX;

for (int v : all_vertices) {
  // Score favors left (small x) and top (small y in typical coords, or large y if y-up)
  // Assuming standard screen coords where y increases downward:
  float score = cluster[v].x + cluster[v].y;  // Top-left = minimum sum
  
  // If you have y-up coordinates (mathematical), use:
  // float score = cluster[v].x - cluster[v].y;  // Top-left = min x, max y
  
  if (score < min_score) {
    min_score = score;
    start_vertex = v;
  }
}

std::cout << "    Starting MST from top-left vertex: " << start_vertex 
          << " at (" << cluster[start_vertex].x << ", " 
          << cluster[start_vertex].y << ")" << std::endl;

// Store the parent/previous vertex for each vertex in MST
std::map<int, int> parent_vertex;

// Custom comparator for priority queue
auto edge_compare = [](const std::pair<Edge, float> &a, const std::pair<Edge, float> &b) {
  return a.second > b.second;
};

// Prim's algorithm implementation with angle continuity
std::vector<Edge> mst_edges;
std::set<int> in_mst;
std::priority_queue<std::pair<Edge, float>, 
                    std::vector<std::pair<Edge, float>>, 
                    decltype(edge_compare)> pq(edge_compare);

in_mst.insert(start_vertex);
parent_vertex[start_vertex] = -1;

// Add initial edges with no angle penalty
for (const auto &edge : adjacency_list[start_vertex]) {
  int other_vertex =
    (edge.point1_idx == start_vertex) ? edge.point2_idx : edge.point1_idx;
  if (in_mst.find(other_vertex) == in_mst.end()) {
    pq.push({edge, edge.length});
  }
}

while (!pq.empty() && in_mst.size() < all_vertices.size()) {
  auto [current_edge, weighted_cost] = pq.top();
  pq.pop();

  int new_vertex = -1;
  int existing_vertex = -1;
  
  if (in_mst.find(current_edge.point1_idx) != in_mst.end() &&
      in_mst.find(current_edge.point2_idx) == in_mst.end()) {
    new_vertex = current_edge.point2_idx;
    existing_vertex = current_edge.point1_idx;
  } else if (in_mst.find(current_edge.point2_idx) != in_mst.end() &&
             in_mst.find(current_edge.point1_idx) == in_mst.end()) {
    new_vertex = current_edge.point1_idx;
    existing_vertex = current_edge.point2_idx;
  } else {
    continue;
  }

  mst_edges.push_back(current_edge);
  in_mst.insert(new_vertex);
  parent_vertex[new_vertex] = existing_vertex;

  // Add all edges from new_vertex with angle penalty
  for (const auto &edge : adjacency_list[new_vertex]) {
    int other_vertex =
      (edge.point1_idx == new_vertex) ? edge.point2_idx : edge.point1_idx;
    
    if (in_mst.find(other_vertex) == in_mst.end()) {
      float base_length = edge.length;
      float final_weight = base_length;
      
      // Apply angle penalty if we have a previous direction
      int parent = parent_vertex[new_vertex];
      if (parent != -1) {
        float angle_penalty = calculate_angle_penalty(parent, new_vertex, other_vertex);
        final_weight = base_length * angle_penalty;
      }
      
      pq.push({edge, final_weight});
    }
  }
}

// Replace the original edges with MST edges
cluster_edges = mst_edges;

std::cout << "    MST created with " << mst_edges.size() << " edges"
          << std::endl;

  return cluster_edges;
}

void GraphVectorizer::simplify_graph(std::vector<Edge> &cluster_edges,
                                     const std::vector<Point2D> &cluster)
{

  std::map<int, int> node_degree;
  std::map<int, std::vector<Edge>> node_edges;

  // create node degree + adjacency list
  for (const auto &edge : cluster_edges) {
    node_degree[edge.point1_idx]++;
    node_degree[edge.point2_idx]++;

    node_edges[edge.point1_idx].push_back(edge);
    node_edges[edge.point2_idx].emplace_back(edge.point2_idx, edge.point1_idx,
                                             edge.length);
  }

  // adds the first end or junction node
  std::queue<Edge> path_starts;
  for (long unsigned int i = 0; i < node_degree.size(); i++) {
    if (node_degree[i] != 2) {
      for (int j = 0; j < node_degree[i]; j++) {
        path_starts.push(node_edges[i][j]);
      }
      break;
    }
  }

  // finds the ordered path for each start edge in the queue
  std::map<int, std::vector<std::vector<Edge>>> all_paths;

  while (!path_starts.empty()) {
    std::vector<Edge> path;
    path.push_back(path_starts.front());
    int next_node = path_starts.front().point2_idx;

    // adds edges next in the ordered path
    while (node_degree[next_node] == 2) {
      int current_node = next_node;
      next_node = -1;
      for (const auto &edge : node_edges[current_node]) {
        if (edge.point2_idx != path[path.size() - 1].point1_idx) {
          next_node = edge.point2_idx;
          path.push_back(edge);
          break;
        }
      }
    }
    all_paths[path_starts.front().point1_idx].push_back(path);
    all_paths[path[path.size() - 1].point2_idx].push_back(path);
    path_starts.pop();

    // adds new start edges for junctions
    if (node_degree[next_node] > 2) {
      for (const auto &edge : node_edges[next_node]) {
        if (edge.point2_idx != path[path.size() - 1].point1_idx) {
          path_starts.push(edge);
        }
      }
    }
  }

  // finds the longest and second longest path
  for (unsigned long int i = 0; i < node_degree.size(); i++) {
    if (node_degree[i] > 2) {

      std::vector<std::pair<int, float>> path_lengths; // (path_index, length)
      int path_idx = 0;

      for (const auto &path : all_paths[i]) {
        float len = 0.0f;
        for (const auto &edge : path) {
          len += edge.length;
        }

        path_lengths.emplace_back(path_idx, len);
        path_idx++;
      }

      // sort by length, descending
      std::sort(path_lengths.begin(), path_lengths.end(),
                [](auto &a, auto &b) { return a.second > b.second; });

      int longest_idx = path_lengths[0].first;
      int second_longest_idx = path_lengths[1].first;

      const auto &longest_path = all_paths[i][longest_idx];
      const auto &second_longest_path = all_paths[i][second_longest_idx];

      // removes the shorter paths from its other end
      for (long unsigned int j = 2; j < path_lengths.size(); j++) {
        int path_to_remove_idx = path_lengths[j].first;

        int node_1 = all_paths[i][path_to_remove_idx][0].point1_idx;
        int node_2 = all_paths[i][path_to_remove_idx].back().point2_idx;

        auto &paths = all_paths[node_2];
        paths.erase(std::remove_if(paths.begin(), paths.end(),
                                   [&](const std::vector<Edge> &path) {
                                     return path.front().point1_idx == node_1 &&
                                            path.back().point2_idx == node_2;
                                   }),
                    paths.end());
      }
      all_paths[i] = {longest_path, second_longest_path};
    }
  }

  // updates the edges with the simplified graph
  std::vector<bool> flags(all_paths.size(), false);
  cluster_edges.clear();

  for (long unsigned int i = 0; i < all_paths.size(); i++) {
    if (all_paths[i].empty()) {
      continue;
    }

    std::cout << "\nNode " << i << " has paths:" << std::endl;
    for (const auto &path : all_paths[i]) {
      if (!flags[path[0].point1_idx] || !flags[path.back().point2_idx]) {
        flags[path[0].point1_idx] = true;
        flags[path.back().point2_idx] = true;

        std::cout << path[0].point1_idx << " - " << path.back().point2_idx
                  << std::endl;

        for (const auto &edge : path) {
          cluster_edges.push_back(edge);
        }
      }
    }
  }
}

void GraphVectorizer::export_graph_to_svg(
  const std::vector<std::vector<Point2D>> &clusters,
  const std::vector<std::vector<Edge>> &cluster_graphs,
  const std::string &filename)
{
  if (clusters.empty()) {
    std::cout << "No clusters to export." << std::endl;
    return;
  }

  std::ofstream svg_file(filename);
  if (!svg_file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing."
              << std::endl;
    return;
  }

  int width = 800;
  int height = 600;

  // Find bounds of all points for scaling
  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float min_y = std::numeric_limits<float>::max();
  float max_y = std::numeric_limits<float>::lowest();

  // for (const auto &cluster : clusters) {
  for (size_t i = 1; i < clusters.size(); ++i) {
    const auto &cluster = clusters[i];
    for (const auto &point : cluster) {
      min_x = std::min(min_x, point.x);
      max_x = std::max(max_x, point.x);
      min_y = std::min(min_y, point.y);
      max_y = std::max(max_y, point.y);
    }
  }

  // Add small margin
  float margin = 20.0f;
  float range_x = max_x - min_x;
  float range_y = max_y - min_y;

  // Scale factors to fit within SVG canvas
  float scale_x = (width - 2 * margin) / range_x;
  float scale_y = (height - 2 * margin) / range_y;
  float scale = std::min(scale_x, scale_y);

  // Simple color palette
  std::vector<std::string> colors = {
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#800000", "#008000", "#000080", "#808000", "#800080", "#008080",
    "#FFA500", "#800080", "#008B8B", "#B22222", "#228B22", "#4B0082"};

  // SVG header
  svg_file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  svg_file << "<svg width=\"" << width << "\" height=\"" << height
           << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";

  // White background
  svg_file << "<rect width=\"" << width << "\" height=\"" << height
           << "\" fill=\"white\"/>\n";

  // Draw edges first (behind points)
  for (size_t cluster_idx = 0;
       cluster_idx < clusters.size() && cluster_idx < cluster_graphs.size();
       ++cluster_idx) {
    const auto &cluster = clusters[cluster_idx];
    const auto &edges = cluster_graphs[cluster_idx];
    std::string color = colors[cluster_idx % colors.size()];

    for (const auto &edge : edges) {
      const auto &p1 = cluster[edge.point1_idx];
      const auto &p2 = cluster[edge.point2_idx];

      float svg_x1 = margin + (p1.x - min_x) * scale;
      float svg_y1 = height - (margin + (p1.y - min_y) * scale);
      float svg_x2 = margin + (p2.x - min_x) * scale;
      float svg_y2 = height - (margin + (p2.y - min_y) * scale);

      // FIXED: Put the entire line tag on one line or use proper string
      // concatenation
      svg_file << "<line x1=\"" << svg_x1 << "\" y1=\"" << svg_y1 << "\" x2=\""
               << svg_x2 << "\" y2=\"" << svg_y2 << "\" stroke=\"" << color
               << "\" stroke-width=\"1\"/>\n";
    }
  }

  // Draw points
  // for (size_t cluster_idx = 0; cluster_idx < clusters.size(); ++cluster_idx)
  // {
  //   const auto &cluster = clusters[cluster_idx];
  //   std::string color = colors[cluster_idx % colors.size()];

  //   for (const auto &point : cluster) {
  //     float svg_x = margin + (point.x - min_x) * scale;
  //     float svg_y = height - (margin + (point.y - min_y) * scale);

  //     svg_file << "<circle cx=\"" << svg_x << "\" cy=\"" << svg_y
  //              << "\" r=\"2\" fill=\"" << color << "\"/>\n";
  //   }
  // }

  svg_file << "</svg>\n";
  svg_file.close();

  int total_edges = 0;
  for (const auto &edges : cluster_graphs) {
    total_edges += edges.size();
  }

  std::cout << "Exported " << clusters.size() << " clusters with "
            << total_edges << " total edges to " << filename << std::endl;
}

// Helper to export preview SVG of a single cluster
// void GraphVectorizer::export_cluster_preview(
//     const std::vector<Point2D> &cluster,
//     size_t cluster_idx,
//     const std::string &filename)
// {
//   std::ofstream svg_file(filename);
//   if (!svg_file.is_open()) {
//     std::cerr << "Error: Could not open file " << filename << std::endl;
//     return;
//   }

//   int width = 400;
//   int height = 300;

//   // Find bounds
//   float min_x = std::numeric_limits<float>::max();
//   float max_x = std::numeric_limits<float>::lowest();
//   float min_y = std::numeric_limits<float>::max();
//   float max_y = std::numeric_limits<float>::lowest();

//   for (const auto &point : cluster) {
//     min_x = std::min(min_x, point.x);
//     max_x = std::max(max_x, point.x);
//     min_y = std::min(min_y, point.y);
//     max_y = std::max(max_y, point.y);
//   }

//   float margin = 20.0f;
//   float range_x = max_x - min_x;
//   float range_y = max_y - min_y;
  
//   // Prevent division by zero
//   if (range_x < 0.001f) range_x = 1.0f;
//   if (range_y < 0.001f) range_y = 1.0f;

//   float scale_x = (width - 2 * margin) / range_x;
//   float scale_y = (height - 2 * margin) / range_y;
//   float scale = std::min(scale_x, scale_y);

//   // SVG header
//   svg_file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
//   svg_file << "<svg width=\"" << width << "\" height=\"" << height
//            << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
//   svg_file << "<rect width=\"" << width << "\" height=\"" << height
//            << "\" fill=\"white\"/>\n";

//   // Draw points
//   for (const auto &point : cluster) {
//     float svg_x = margin + (point.x - min_x) * scale;
//     float svg_y = height - (margin + (point.y - min_y) * scale);
//     svg_file << "<circle cx=\"" << svg_x << "\" cy=\"" << svg_y
//              << "\" r=\"2\" fill=\"#0000FF\"/>\n";
//   }

//   // Add cluster info text
//   svg_file << "<text x=\"10\" y=\"20\" font-family=\"Arial\" font-size=\"14\" fill=\"black\">";
//   svg_file << "Cluster " << cluster_idx << " - " << cluster.size() << " points";
//   svg_file << "</text>\n";

//   svg_file << "</svg>\n";
//   svg_file.close();
// }

void GraphVectorizer::export_cluster_preview(
    const std::vector<std::vector<Point2D>> &all_clusters,
    size_t highlighted_cluster_idx,
    const std::string &filename)
{
  std::ofstream svg_file(filename);
  if (!svg_file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return;
  }

  int width = 800;
  int height = 600;

  // Find bounds of ALL clusters
  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float min_y = std::numeric_limits<float>::max();
  float max_y = std::numeric_limits<float>::lowest();

  for (const auto &cluster : all_clusters) {
    for (const auto &point : cluster) {
      min_x = std::min(min_x, point.x);
      max_x = std::max(max_x, point.x);
      min_y = std::min(min_y, point.y);
      max_y = std::max(max_y, point.y);
    }
  }

  float margin = 40.0f;
  float range_x = max_x - min_x;
  float range_y = max_y - min_y;
  
  // Prevent division by zero
  if (range_x < 0.001f) range_x = 1.0f;
  if (range_y < 0.001f) range_y = 1.0f;

  float scale_x = (width - 2 * margin) / range_x;
  float scale_y = (height - 2 * margin) / range_y;
  float scale = std::min(scale_x, scale_y);

  // SVG header
  svg_file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  svg_file << "<svg width=\"" << width << "\" height=\"" << height
           << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
  svg_file << "<rect width=\"" << width << "\" height=\"" << height
           << "\" fill=\"white\"/>\n";

  // Draw all clusters
  for (size_t cluster_idx = 0; cluster_idx < all_clusters.size(); ++cluster_idx) {
    const auto &cluster = all_clusters[cluster_idx];
    
    // Determine color and size based on whether this is the highlighted cluster
    bool is_highlighted = (cluster_idx == highlighted_cluster_idx);
    std::string color = is_highlighted ? "#FF0000" : "#000000";  // Red for highlighted, black for others
    float radius = is_highlighted ? 3.0f : 1.5f;
    float opacity = is_highlighted ? 1.0f : 0.3f;
    
    for (const auto &point : cluster) {
      float svg_x = margin + (point.x - min_x) * scale;
      float svg_y = height - (margin + (point.y - min_y) * scale);
      svg_file << "<circle cx=\"" << svg_x << "\" cy=\"" << svg_y
               << "\" r=\"" << radius << "\" fill=\"" << color 
               << "\" opacity=\"" << opacity << "\"/>\n";
    }
  }

  // Add cluster info text with background
  svg_file << "<rect x=\"5\" y=\"5\" width=\"250\" height=\"30\" fill=\"white\" opacity=\"0.8\"/>\n";
  svg_file << "<text x=\"10\" y=\"25\" font-family=\"Arial\" font-size=\"16\" font-weight=\"bold\" fill=\"#FF0000\">";
  svg_file << "Cluster " << highlighted_cluster_idx << " - " 
           << all_clusters[highlighted_cluster_idx].size() << " points";
  svg_file << "</text>\n";

  svg_file << "</svg>\n";
  svg_file.close();
}

// Interactive yes/no selection for each cluster
std::vector<bool> GraphVectorizer::promptClusterSelection(
    const std::vector<std::vector<Point2D>> &clusters)
{
  std::vector<bool> selected(clusters.size(), false);
  
  std::cout << "\n=== Interactive Cluster Selection ===" << std::endl;
  std::cout << "Found " << clusters.size() << " clusters" << std::endl;
  std::cout << "Each cluster will be highlighted in red while others are shown in black.\n" << std::endl;

  pid_t viewer_process = 0;

  for (size_t i = 0; i < clusters.size(); ++i) {
    std::cout << "\n--- Cluster " << i << " ---" << std::endl;
    std::cout << "Points: " << clusters[i].size() << std::endl;
    
    // Calculate bounds for size info
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    
    for (const auto &point : clusters[i]) {
      min_x = std::min(min_x, point.x);
      max_x = std::max(max_x, point.x);
      min_y = std::min(min_y, point.y);
      max_y = std::max(max_y, point.y);
    }
    
    float width = max_x - min_x;
    float height = max_y - min_y;
    
    std::cout << "Dimensions: " << width << " x " << height << " units" << std::endl;
    
    // Export preview with this cluster highlighted
    std::string preview_file = "/app/output/cluster_preview_" + std::to_string(i) + ".svg";
    export_cluster_preview(clusters, i, preview_file);
    
    // Open the SVG file
    std::cout << "Opening preview..." << std::endl;
    viewer_process = open_svg_file(preview_file);
    
    // Give the viewer a moment to open
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Prompt for inclusion
    std::cout << "\nInclude this cluster? (y/n/q to quit): ";
    std::string response;
    std::getline(std::cin, response);
    
    // Close the viewer after response
    close_svg_viewer(viewer_process);
    
    // Convert to lowercase
    std::transform(response.begin(), response.end(), response.begin(), ::tolower);
    
    if (response == "q" || response == "quit") {
      std::cout << "\nSelection stopped. Processing " << i << " clusters." << std::endl;
      break;
    }
    else if (response == "y" || response == "yes") {
      selected[i] = true;
      std::cout << "✓ Cluster " << i << " included" << std::endl;
    }
    else {
      std::cout << "✗ Cluster " << i << " excluded" << std::endl;
    }
  }
  
  // Summary
  int selected_count = std::count(selected.begin(), selected.end(), true);
  std::cout << "\n=== Selection Complete ===" << std::endl;
  std::cout << "Total selected: " << selected_count << "/" << clusters.size() << " clusters" << std::endl;
  std::cout << "Selected clusters: ";
  for (size_t i = 0; i < selected.size(); ++i) {
    if (selected[i]) {
      std::cout << i << " ";
    }
  }
  std::cout << "\n" << std::endl;
  
  return selected;
}