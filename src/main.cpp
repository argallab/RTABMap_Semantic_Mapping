#include <database_exporter.hpp>
#include <lidar_database_exporter.hpp>
#include <rtabmap_vectorize.hpp>
#include <semantic_mapping.hpp>
#include <wc_database_exporter.hpp>

int main(int argc, char *argv[])
{
  try {
    std::cout << "Starting database exporter..." << std::endl;
    // try {
    //   py::scoped_interpreter guard{};

    //   auto sys = py::module::import("sys");
    //   std::cout << "Python executable: "
    //             << sys.attr("executable").cast<std::string>() << std::endl;
    //   std::cout << "Python version: " << sys.attr("version").cast<std::string>()
    //             << std::endl;

    //   auto torch = py::module::import("torch");
    //   std::cout << "Torch version: "
    //             << torch.attr("__version__").cast<std::string>() << std::endl;
    //   std::cout << "CUDA available: "
    //             << torch.attr("cuda").attr("is_available")().cast<bool>()
    //             << std::endl;

    //   // std::cout << "Importing YOLOv8 model..." << std::endl;
    //   // py::module yolov8 = py::module::import("ultralytics");
    //   std::cout << "Importing YOLOv8 module..." << std::endl;
    //   py::module yolov8 = py::module::import("ultralytics");
    //   std::cout << "Imported YOLOv8 module!" << std::endl;

    //   std::cout << "Loading YOLOv8 model..." << std::endl;
    //   py::object YOLO = yolov8.attr("YOLO");

    //   std::cout << "Creating YOLOv8 model instance..." << std::endl;
    //   py::object net = YOLO("/app/models/yolov8m.pt");
    // } catch (py::error_already_set &e) {
    //   std::cerr << "Python error: " << e.what() << std::endl;
    //   return 1;
    // }

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

    std::cout << "RTAB-Map database: " << rtabmap_database_name << std::endl;
    std::unique_ptr<DatabaseExporter> extractor;
    if (lidar_used) {
      extractor = std::make_unique<LidarDatabaseExporter>(rtabmap_database_name,
                                                          model_name);
      extractor->configure_vectorizer();
    } else {
      extractor =
        std::make_unique<WCDatabaseExporter>(rtabmap_database_name, model_name);
    }

    std::cout << "Loading RTAB-Map database..." << std::endl;
    Result result = extractor->load_rtabmap_db();
    // std::vector<Object> objects = semantic_mapping(
    //   net, result.mapping_data, result.cloud, result.timestamp);

    // auto vectorizer = std::make_unique<RTABMapVectorizer>();
    // VectorMap vector_map = vectorizer->vectorize_point_cloud(result.cloud,
    // result.timestamp);

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  return 0;
}
