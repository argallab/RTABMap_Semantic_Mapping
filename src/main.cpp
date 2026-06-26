#include <database_exporter.hpp>
#include <semantic_mapping.hpp>
#include <wc_database_exporter.hpp>

int main(int argc, char *argv[])
{
  try {
    std::cout << "Starting database exporter..." << std::endl;

    if (argc < 2 || argc > 3) {
      std::cerr << "Usage: database_exporter <db_name> [model_name]" << std::endl;
      return 1;
    }

    std::string rtabmap_database_name = argv[1];
    std::string model_name = (argc == 3) ? argv[2] : "";

    std::cout << "RTAB-Map database: " << rtabmap_database_name << std::endl;
    auto extractor = std::make_unique<WCDatabaseExporter>(rtabmap_database_name, model_name);

    std::cout << "Loading RTAB-Map database..." << std::endl;
    Result result = extractor->load_rtabmap_db();

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  return 0;
}
