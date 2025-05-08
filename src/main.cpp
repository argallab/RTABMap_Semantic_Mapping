#include <database_exporter.hpp>
#include <lidar_database_exporter.hpp>
#include <wc_database_exporter.hpp>
#include <semantic_mapping.hpp>

int main(int argc, char *argv[])
{
  try {
    py::scoped_interpreter guard{};
    py::module yolov8 = py::module::import("ultralytics");
    py::object YOLO = yolov8.attr("YOLO");
    py::object net = YOLO("/app/models/yolov8m.pt");

    std::string rtabmap_database_name;
    std::string model_name;
    bool lidar_used;
    if (argc == 1) {
      return 1;
    } else if (argc == 2) {
      rtabmap_database_name = argv[1];
      lidar_used = true;
      model_name = "";
    } else if (argc == 3) {
      rtabmap_database_name = argv[1];
      std::string arg2 = argv[2];
      std::transform(arg2.begin(), arg2.end(), arg2.begin(), ::tolower);
      lidar_used = (arg2 == "1" || arg2 == "true");
      model_name = "";
    } else if (argc == 4) {
      rtabmap_database_name = argv[1];
      std::string arg2 = argv[2];
      std::transform(arg2.begin(), arg2.end(), arg2.begin(), ::tolower);
      lidar_used = (arg2 == "1" || arg2 == "true");
      model_name = argv[3];
    } else {
      return 1;
    }

    std::unique_ptr<DatabaseExporter> extractor;
    if (lidar_used) {
      extractor = std::make_unique<LidarDatabaseExporter>(rtabmap_database_name, model_name);
    } else {
      extractor = std::make_unique<WCDatabaseExporter>(rtabmap_database_name, model_name);
    }
  
    Result result = extractor->load_rtabmap_db();

    // std::vector<Object> objects = semantic_mapping(
    //   net, *extractor, result.mapping_data, result.cloud, result.timestamp);

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  return 0;
}
