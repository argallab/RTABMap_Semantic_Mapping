// Entry point of the DEV pipeline (database_exporter_dev); outputs go to output/<db>/<db>_dev_<N>.
#include <algorithm>
#include <filesystem>

#include <database_exporter.hpp>
#include <semantic_mapping.hpp>
#include <wc_database_exporter_dev.hpp>

int main(int argc, char *argv[])
{
  try {
    std::cout << "Starting database exporter (DEV pipeline)..." << std::endl;

    if (argc < 2 || argc > 3) {
      std::cerr << "Usage: database_exporter_dev <db_name> [model_name]" << std::endl;
      return 1;
    }

    std::string rtabmap_database_name = argv[1];
    std::string model_name = (argc == 3) ? argv[2] : "";

    std::cout << "RTAB-Map database: " << rtabmap_database_name << std::endl;
    // output/<db>/<db>_dev_<N>: N = 1 + the highest existing number for this database (never overwrites)
    std::string stem = std::filesystem::path(rtabmap_database_name).stem().string();
    const std::string prefix = stem + "_dev_";
    int next = 1;
    std::error_code ec;
    // looks in output/ and one level below (runs moved into folders like output/R01-cleaned-maps/ still count)
    auto consider = [&](const std::filesystem::path &p) {
      const std::string n = p.filename().string();
      if (n.rfind(prefix, 0) != 0) return;
      const std::string num = n.substr(prefix.size());
      if (!num.empty() && num.find_first_not_of("0123456789") == std::string::npos)
        next = std::max(next, std::stoi(num) + 1);
    };
    for (const auto &e : std::filesystem::directory_iterator(std::string(PROJECT_PATH) + "/output", ec)) {
      consider(e.path());
      if (e.is_directory(ec))
        for (const auto &e2 : std::filesystem::directory_iterator(e.path(), ec)) consider(e2.path());
    }
    auto extractor = std::make_unique<WCDatabaseExporterDev>(rtabmap_database_name, model_name,
                                                             prefix + std::to_string(next), stem);

    std::cout << "Loading RTAB-Map database..." << std::endl;
    Result result = extractor->load_rtabmap_db();

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  return 0;
}
