
#include <rtabmap_vectorize.hpp>
#include <graph_vectorize.hpp>
#include <pcl/io/pcd_io.h>   
#include <pcl/point_types.h>   
#include <pcl/point_cloud.h>  

int main()
{
  py::scoped_interpreter guard{};
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);

  std::string filename = "/app/output/2025-12-04_17-22-45/cloud/2025-12-04_17-22-45.pcd";

  pcl::io::loadPCDFile<pcl::PointXYZRGB>(filename, *cloud);
  std::cout << "Loaded " << cloud->width * cloud->height << " data points from "
            << filename << std::endl;

  std::string timestamp = "2025-08-07T12:00:00Z"; // Example timestamp
  // auto vectorizer = std::make_unique<RTABMapVectorizer>();
  // VectorMap vector_map = vectorizer->vectorize_point_cloud(cloud, timestamp);

  auto vectorizer = std::make_unique<GraphVectorizer>();
  vectorizer->vectorizeGraph(cloud, timestamp);
}
