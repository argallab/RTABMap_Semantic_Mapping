#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/voxel_grid.h>

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> Delaunay;
typedef K::Point_2 CGALPoint;

using namespace pybind11::literals;
namespace py = pybind11;

struct Point2D {
  float x;
  float y;
  int cluster_id;

  Point2D(float x_, float y_, int cluster_id_)
    : x(x_), y(y_), cluster_id(cluster_id_)
  {
  }

  // Useful for sorting and set operations
  bool operator<(const Point2D &other) const
  {
    if (x != other.x)
      return x < other.x;
    return y < other.y;
  }

  // Calculate distance to another point
  float distanceTo(const Point2D &other) const
  {
    return std::sqrt((x - other.x) * (x - other.x) +
                     (y - other.y) * (y - other.y));
  }
};

struct Edge {
  int point1_idx;
  int point2_idx;
  float length;

  Edge(int p1, int p2, float len) : point1_idx(p1), point2_idx(p2), length(len)
  {
  }
};

class GraphVectorizer {

public:
  GraphVectorizer();
  ~GraphVectorizer();

  void vectorizeGraph(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                      const std::string &timestamp);

private:
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
  filter_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

  std::vector<std::vector<Point2D>>
  assemble_clusters(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

  std::vector<std::vector<Point2D>>
  assemble_clusters(const std::vector<Point2D> &cloud);

  std::vector<Edge> create_graph(std::vector<Point2D> &cluster);
  void simplify_graph(std::vector<Edge> &cluster_edges,
                      const std::vector<Point2D> &cluster);

  // Utility methods
  void export_graph_to_svg(const std::vector<std::vector<Point2D>> &clusters,
                           const std::vector<std::vector<Edge>> &cluster_graphs,
                           const std::string &filename);

  // pid_t open_svg_file(const std::string &filename);
  // void close_svg_viewer(pid_t pid);

  // void export_cluster_preview(const std::vector<Point2D> &cluster,
  //                             size_t cluster_idx, const std::string &filename);

  void
  export_cluster_preview(const std::vector<std::vector<Point2D>> &all_clusters,
                         size_t highlighted_cluster_idx,
                         const  std::string &filename);

  std::vector<bool>
  promptClusterSelection(const std::vector<std::vector<Point2D>> &clusters);
};