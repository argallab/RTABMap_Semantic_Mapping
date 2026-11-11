// DEV COPY of wc_database_exporter.cpp — the working pipeline (database_exporter_dev).
// Change THIS file when tuning for a user; the default pipeline
// (wc_database_exporter.cpp → database_exporter) stays frozen.  Record every
// change and why in notes/<user>.md.  Outputs go to output/<timestamp>_dev.

#include "wc_database_exporter_dev.hpp"

// Set per executable by CMakeLists.txt (default = frozen pipeline, dev = working copy).
#ifndef PIPELINE_NAME
#define PIPELINE_NAME "unknown"
#endif
#ifndef PIPELINE_SRC_SHA256
#define PIPELINE_SRC_SHA256 "unknown"
#endif

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <queue>
#include <set>
#include <sstream>
#include <unistd.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/RegistrationIcp.h>
#include <rtabmap/core/RegistrationInfo.h>

#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Eigenvalues>

// DEV: camera yaw correction (env WC_CAMERA_YAW_DEG, default 0).  The depth
// cameras' scans AND their AprilTag detections appear rotated ~3° (CCW,
// about base_link) relative to the chair: against ground truth, rotating the
// tag detections by +3° cuts ATE rmse (odometry x16) 37.8 → 22.5 mm on 10-12-01,
// 47.7 → 37.6 mm on 10-08-01, 47.9 → 45.0 mm on R01_02 (+2°: 43.5); rotating the
// scans by +3° sharpens R01_02's walls 22.2 → 16.0 mm (between passes) with the
// map's own poses and makes its double E3–H3 wall one line.  Left and right
// camera are not reliably different (best pairs vary per recording; +3/+3 is
// within ~3 mm of the best everywhere).  Likely a ~3° yaw error in the
// cameras' extrinsics on the chair — fix there too.
static double camera_yaw_rad()
{
  const char *e = std::getenv("WC_CAMERA_YAW_DEG");
  return e ? std::atof(e) * M_PI / 180.0 : 0.0;
}
// Rotate a landmark link's measurement by the camera yaw correction.
static void rotate_landmark_link(rtabmap::Link &lk, double yaw)
{
  const rtabmap::Transform Rz(0, 0, 0, 0, 0, (float)yaw);
  if (lk.from() > 0) lk.setTransform(Rz * lk.transform());            // node -> tag, in the node (base) frame
  else               lk.setTransform(lk.transform() * Rz.inverse());  // tag -> node
}

// ---------------------------------------------------------------------------
// add_metric / wall_thickness
// ---------------------------------------------------------------------------
void WCDatabaseExporterDev::add_metric(const char *fmt, ...)
{
  char buf[512];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  metrics_.emplace_back(buf);
}

// Pools wall-band points (z in [0.10, 1.00]) from all cleaned scans placed
// with `poses`, flattened to XY and point-strided to ~kTargetPts so every part
// of the run is represented.  Each pooled point keeps its node ID (≈ time)
// and height.  At ~kQueries sample points, the neighbours within kRadius are
// split by |node ID − sample's node ID|:
//   ref (≤ kRefWindow)  : decides whether the patch is a wall — spans
//                         ≥ kMinHeightSpan of height (rejects furniture,
//                         clutter), ≥ kMinLength along it, and is straight
//                         (λmin ≤ kMaxLineRatio·λmax; rejects corners).
//                         Classifying on a short time window means a wall
//                         doubled by misalignment between visits still counts.
//   local (≤ kLocalWindow), global (all) : robust spread (1.4826·MAD, mm) of
//                         their offsets along the ref wall normal.
// On 09-29-2.db odometry-only poses: ref 20.2 / local 22.5 / global 42.7 mm.
WCDatabaseExporterDev::WallThickness
WCDatabaseExporterDev::wall_thickness(const std::map<int, rtabmap::Transform> &poses) const
{
  constexpr float  kZlo = 0.10f, kZhi = 1.00f;
  constexpr size_t kTargetPts     = 1500000;
  constexpr size_t kQueries       = 20000;
  constexpr float  kRadius        = 0.20f;
  constexpr int    kRefWindow     = 30;     // nodes (~1 m of path)
  constexpr int    kLocalWindow   = 150;    // nodes (~6 m of path)
  constexpr int    kMinRefPts     = 30;
  constexpr float  kMinHeightSpan = 0.60f;  // m
  constexpr double kMinLength     = 0.08;   // m, sqrt(λmax) (≈ 28 cm of wall)
  constexpr double kMaxLineRatio  = 0.15;

  WallThickness out;
  size_t total = 0;
  for (const auto &[nid, pts] : per_scan_base_) total += pts.size();
  if (total == 0) return out;
  const size_t stride = std::max<size_t>(1, total / kTargetPts);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<int>   pid;  // node ID per pooled point
  std::vector<float> pz;   // height per pooled point
  size_t k = 0;
  for (const auto &[nid, pts] : per_scan_base_) {
    auto pit = poses.find(nid);
    if (pit == poses.end()) { k += pts.size(); continue; }
    Eigen::Affine3f T = pit->second.toEigen3f();
    for (const auto &p : pts) {
      if (k++ % stride) continue;
      if (p.z < kZlo || p.z > kZhi) continue;
      Eigen::Vector3f w = T * Eigen::Vector3f(p.x, p.y, p.z);
      cloud->push_back(pcl::PointXYZ(w.x(), w.y(), 0.0f));
      pid.push_back(nid);
      pz.push_back(p.z);
    }
  }
  if (cloud->size() < (size_t)kMinRefPts) return out;

  auto robust_sigma_mm = [](std::vector<double> &d) {
    std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
    const double med = d[d.size() / 2];
    for (auto &v : d) v = std::abs(v - med);
    std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
    return 1.4826 * d[d.size() / 2] * 1000.0;
  };

  pcl::KdTreeFLANN<pcl::PointXYZ> kd;
  kd.setInputCloud(cloud);
  const size_t qstride = std::max<size_t>(1, cloud->size() / kQueries);
  std::vector<double> th_ref, th_local, th_global;
  pcl::Indices idx;
  std::vector<float> d2;
  std::vector<int> ref;
  std::vector<double> off_ref, off_local, off_global;
  for (size_t i = 0; i < cloud->size(); i += qstride) {
    if (kd.radiusSearch(cloud->points[i], kRadius, idx, d2) < kMinRefPts) continue;
    const int qn = pid[i];

    ref.clear();
    float zlo = 1e9f, zhi = -1e9f;
    for (auto j : idx)
      if (std::abs(pid[j] - qn) <= kRefWindow) {
        ref.push_back(j);
        zlo = std::min(zlo, pz[j]);
        zhi = std::max(zhi, pz[j]);
      }
    if ((int)ref.size() < kMinRefPts || zhi - zlo < kMinHeightSpan) continue;

    double mx = 0, my = 0;
    for (auto j : ref) { mx += cloud->points[j].x; my += cloud->points[j].y; }
    mx /= ref.size(); my /= ref.size();
    double a = 0, b = 0, c = 0;
    for (auto j : ref) {
      double dx = cloud->points[j].x - mx, dy = cloud->points[j].y - my;
      a += dx * dx; b += dx * dy; c += dy * dy;
    }
    a /= ref.size(); b /= ref.size(); c /= ref.size();
    const double h = std::sqrt(0.25 * (a - c) * (a - c) + b * b);
    const double lmax = 0.5 * (a + c) + h, lmin = std::max(0.0, 0.5 * (a + c) - h);
    if (std::sqrt(lmax) < kMinLength || lmin > kMaxLineRatio * lmax) continue;
    Eigen::Vector2d n = std::abs(b) > 1e-12 ? Eigen::Vector2d(b, lmin - a)
                      : (a < c ? Eigen::Vector2d(1, 0) : Eigen::Vector2d(0, 1));
    n.normalize();

    off_ref.clear(); off_local.clear(); off_global.clear();
    for (auto j : idx) {
      const double d = n.x() * (cloud->points[j].x - mx) + n.y() * (cloud->points[j].y - my);
      const int dt = std::abs(pid[j] - qn);
      off_global.push_back(d);
      if (dt <= kLocalWindow) off_local.push_back(d);
      if (dt <= kRefWindow)   off_ref.push_back(d);
    }
    th_ref.push_back(robust_sigma_mm(off_ref));
    th_local.push_back(robust_sigma_mm(off_local));
    th_global.push_back(robust_sigma_mm(off_global));
  }
  if (th_ref.empty()) return out;

  auto pct = [](std::vector<double> &v, double q) {
    const size_t at = (size_t)(q * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + at, v.end());
    return v[at];
  };
  out.n = (int)th_ref.size();
  out.ref_median_mm    = pct(th_ref, 0.5);    out.ref_p90_mm    = pct(th_ref, 0.9);
  out.local_median_mm  = pct(th_local, 0.5);  out.local_p90_mm  = pct(th_local, 0.9);
  out.global_median_mm = pct(th_global, 0.5); out.global_p90_mm = pct(th_global, 0.9);
  return out;
}

void WCDatabaseExporterDev::report_wall_thickness(const std::string &label,
                                               const std::map<int, rtabmap::Transform> &poses)
{
  const WallThickness wt = wall_thickness(poses);

  // analysis/: per-stage pose dump for offline evaluation (no effect on the pipeline)
  {
    std::string tag = label;
    tag.erase(tag.find_last_not_of(' ') + 1);
    std::replace(tag.begin(), tag.end(), ' ', '_');
    std::ofstream pf(output_path_ + "/poses_" + tag + ".csv");
    pf << "id,stamp,x,y,z,qx,qy,qz,qw\n";
    for (const auto &[nid, T] : poses) {
      if (nid <= 0) continue;
      auto nit = nodes.find(nid);
      const Eigen::Quaternionf q = T.getQuaternionf();
      pf << nid << ',' << std::setprecision(17) << (nit != nodes.end() ? nit->second.getStamp() : 0.0)
         << std::setprecision(9) << ',' << T.x() << ',' << T.y() << ',' << T.z() << ','
         << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << '\n';
    }
  }

  const std::string line = "walls " + label + ": ref(±30) %.1f / local(±150) %.1f / global %.1f mm median"
                           "   p90 %.1f / %.1f / %.1f   (n=%d)";
  printf(("[walls] " + line + "\n").c_str(),
         wt.ref_median_mm, wt.local_median_mm, wt.global_median_mm,
         wt.ref_p90_mm, wt.local_p90_mm, wt.global_p90_mm, wt.n);
  add_metric(line.c_str(),
             wt.ref_median_mm, wt.local_median_mm, wt.global_median_mm,
             wt.ref_p90_mm, wt.local_p90_mm, wt.global_p90_mm, wt.n);
}

// ---------------------------------------------------------------------------
// load_bag_detections
// ---------------------------------------------------------------------------
// Reads AprilTag poses from the bag file alongside the database.
// Convention: db at "foo/jb-KEEP.db" → bag at "foo/jb-KEEP-detections/".
//
// Reads /tf messages. The apriltag_ros node publishes
// "camera_frame → tag_X" transforms on /tf whenever a tag is detected.
// We collect those, apply the camera's localTransform to get body→tag,
// then match each detection to the nearest DB node by timestamp.
//
// All observations are merged into raw_landmark_links so the tag stats and
// re-optimisation code sees the denser data.
// ---------------------------------------------------------------------------
std::multimap<int, std::pair<int, rtabmap::Transform>>
WCDatabaseExporterDev::load_bag_detections(const rtabmap::ParametersMap & /*parameters*/)
{
  std::multimap<int, std::pair<int, rtabmap::Transform>> result;

  // ── 1. Derive bag path from database path ──────────────────────────────
  std::string bag_path = rtabmap_database_path_;
  auto dot = bag_path.rfind(".db");
  if (dot == std::string::npos) {
    printf("[bag] Cannot derive detections bag path — DB path '%s' has no .db extension\n",
           bag_path.c_str());
    return result;
  }
  bag_path = bag_path.substr(0, dot) + "-detections";
  std::string meta_path = bag_path + "/metadata.yaml";
  if (access(meta_path.c_str(), F_OK) != 0) {
    printf("[bag] No detections bag found at '%s' — skipping bag augmentation\n",
           bag_path.c_str());
    return result;
  }
  printf("\n[bag] ── Loading detections bag ────────────────────────────────\n");
  printf("[bag] Path : %s\n", bag_path.c_str());

  // ── 2. Build timestamp → node_id lookup from loaded DB nodes ──────────
  std::map<double, int> stamp_to_node;
  for (const auto &[nid, sig] : nodes) {
    double s = sig.getStamp();
    if (s > 1e6)
      stamp_to_node[s] = nid;
  }
  printf("[bag] DB nodes with valid timestamps: %zu  (of %zu total)\n",
         stamp_to_node.size(), nodes.size());
  if (stamp_to_node.empty()) {
    printf("[bag] No usable node timestamps — cannot match bag messages\n");
    return result;
  }
  double t0 = stamp_to_node.begin()->first;
  double t1 = stamp_to_node.rbegin()->first;
  printf("[bag] DB timestamp range: %.3f → %.3f  (duration %.1f s)\n",
         t0, t1, t1 - t0);

  // ── 3. Build camera calibration table ─────────────────────────────────
  struct CamCal {
    rtabmap::Transform local_tf;
    std::string name;
  };
  std::vector<CamCal> cam_cals;
  for (const auto &[nid, sig] : nodes) {
    const auto &models = sig.sensorData().cameraModels();
    if (models.empty()) continue;
    printf("[bag] Camera models from node %d (%zu cameras):\n", nid, models.size());
    for (int ci = 0; ci < (int)models.size(); ++ci) {
      const auto &m = models[ci];
      CamCal cc;
      cc.name     = m.name().empty() ? ("cam" + std::to_string(ci)) : m.name();
      cc.local_tf = m.localTransform();
      cam_cals.push_back(cc);
      printf("[bag]   [%d] name='%s'  localTransform: tx=%.3f ty=%.3f tz=%.3f\n",
             ci, cc.name.c_str(), cc.local_tf.x(), cc.local_tf.y(), cc.local_tf.z());
    }
    break;
  }
  if (cam_cals.empty()) {
    printf("[bag] WARNING: No camera calibration in DB — using identity localTransform\n");
    CamCal cc;
    cc.name     = "identity";
    cc.local_tf = rtabmap::Transform::getIdentity();
    cam_cals.push_back(cc);
  }

  // ── 4. Helper lambdas ──────────────────────────────────────────────────
  auto is_tag_frame = [](const std::string &frame) -> bool {
    if (frame.find("tag") == std::string::npos &&
        frame.find("Tag") == std::string::npos) return false;
    for (char c : frame) if (std::isdigit(c)) return true;
    return false;
  };

  auto extract_last_int = [](const std::string &s) -> int {
    int end = -1;
    for (int i = (int)s.size() - 1; i >= 0; --i) {
      if (std::isdigit(s[i])) { end = i; break; }
    }
    if (end < 0) return -1;
    int start = end;
    while (start > 0 && std::isdigit(s[start - 1])) --start;
    return std::stoi(s.substr(start, end - start + 1));
  };

  auto pick_camera = [&](const std::string &parent_frame) -> const CamCal & {
    for (const auto &cc : cam_cals)
      if (parent_frame.find(cc.name) != std::string::npos ||
          cc.name.find(parent_frame) != std::string::npos)
        return cc;
    bool has_left  = parent_frame.find("left")  != std::string::npos;
    bool has_right = parent_frame.find("right") != std::string::npos;
    if ((has_left || has_right) && cam_cals.size() >= 2) {
      int idx_neg = 0, idx_pos = 0;
      for (int i = 1; i < (int)cam_cals.size(); ++i) {
        if (cam_cals[i].local_tf.x() < cam_cals[idx_neg].local_tf.x()) idx_neg = i;
        if (cam_cals[i].local_tf.x() > cam_cals[idx_pos].local_tf.x()) idx_pos = i;
      }
      return cam_cals[has_left ? idx_neg : idx_pos];
    }
    return cam_cals[0];
  };

  // ── 5. Open bag ────────────────────────────────────────────────────────
  rosbag2_cpp::Reader reader;
  rosbag2_storage::StorageOptions storage_opts;
  storage_opts.uri        = bag_path;
  storage_opts.storage_id = "mcap";
  try {
    reader.open(storage_opts);
  } catch (const std::exception &e) {
    printf("[bag] Failed to open bag: %s\n", e.what());
    return result;
  }

  // ── 6. Read /tf messages ───────────────────────────────────────────────
  constexpr double kMaxDt   = 0.5;
  constexpr double kMinDist = 0.05;
  constexpr double kMaxDist = 8.0;

  rclcpp::Serialization<tf2_msgs::msg::TFMessage> tf_serializer;
  int n_tf_msgs = 0, n_tag_tf = 0, n_no_match = 0;
  int n_large_dt = 0, n_bad_pose = 0, n_out_of_range = 0, n_ok = 0;
  double max_dt_seen = 0.0;
  std::map<int, int> tag_counts;
  std::set<std::string> seen_tag_frames, seen_parent_frames;

  while (reader.has_next()) {
    auto storage_msg = reader.read_next();
    if (storage_msg->topic_name != "/tf") continue;
    ++n_tf_msgs;

    rclcpp::SerializedMessage sm(*storage_msg->serialized_data);
    tf2_msgs::msg::TFMessage tf_msg;
    tf_serializer.deserialize_message(&sm, &tf_msg);

    for (const auto &ts : tf_msg.transforms) {
      if (!is_tag_frame(ts.child_frame_id)) continue;
      ++n_tag_tf;
      seen_tag_frames.insert(ts.child_frame_id);
      seen_parent_frames.insert(ts.header.frame_id);

      int tag_id = extract_last_int(ts.child_frame_id);
      if (tag_id < 0) { ++n_bad_pose; continue; }

      double stamp = ts.header.stamp.sec + ts.header.stamp.nanosec * 1e-9;

      auto it_hi = stamp_to_node.lower_bound(stamp);
      int    best_node = -1;
      double best_dt   = 1e9;
      if (it_hi != stamp_to_node.end()) {
        double dt = std::abs(it_hi->first - stamp);
        if (dt < best_dt) { best_dt = dt; best_node = it_hi->second; }
      }
      if (it_hi != stamp_to_node.begin()) {
        --it_hi;
        double dt = std::abs(it_hi->first - stamp);
        if (dt < best_dt) { best_dt = dt; best_node = it_hi->second; }
      }
      if (best_node < 0)    { ++n_no_match;  continue; }
      if (best_dt > kMaxDt) { ++n_large_dt;  continue; }
      max_dt_seen = std::max(max_dt_seen, best_dt);

      const auto &tr = ts.transform;
      double tx = tr.translation.x, ty = tr.translation.y, tz = tr.translation.z;
      double dist = std::sqrt(tx*tx + ty*ty + tz*tz);
      if (dist < kMinDist || dist > kMaxDist) { ++n_out_of_range; continue; }

      rtabmap::Transform cam_to_tag(
          (float)tx, (float)ty, (float)tz,
          (float)tr.rotation.x, (float)tr.rotation.y,
          (float)tr.rotation.z, (float)tr.rotation.w);
      if (cam_to_tag.isNull()) { ++n_bad_pose; continue; }

      const CamCal &cc = pick_camera(ts.header.frame_id);
      rtabmap::Transform body_to_tag = cc.local_tf * cam_to_tag;
      if (body_to_tag.isNull()) { ++n_bad_pose; continue; }

      ++n_ok;
      tag_counts[tag_id]++;
      result.insert({best_node, {tag_id, body_to_tag}});
    }
  }

  // ── 7. Summary ─────────────────────────────────────────────────────────
  printf("[bag] /tf messages read            : %d\n", n_tf_msgs);
  printf("[bag] Tag transforms found         : %d\n", n_tag_tf);
  printf("[bag] Tag frame IDs seen           :");
  for (const auto &f : seen_tag_frames) printf(" %s", f.c_str());
  printf("\n");
  printf("[bag] Camera frames seen → model mapping:\n");
  for (const auto &f : seen_parent_frames) {
    const CamCal &chosen = pick_camera(f);
    printf("[bag]   %-45s → '%s' (tx=%.3f)\n",
           f.c_str(), chosen.name.c_str(), chosen.local_tf.x());
  }
  printf("[bag] No DB node match             : %d\n", n_no_match);
  printf("[bag] Timestamp gap > %.1f s       : %d\n", kMaxDt, n_large_dt);
  printf("[bag] Max gap seen                 : %.1f ms\n", max_dt_seen * 1000.0);
  printf("[bag] Bad pose                     : %d\n", n_bad_pose);
  printf("[bag] Out of range                 : %d\n", n_out_of_range);
  printf("[bag] Accepted observations        : %d\n", n_ok);
  printf("[bag] Per-tag counts from bag:\n");
  for (const auto &[tid, cnt] : tag_counts)
    printf("[bag]   Tag %2d : %d observations\n", tid, cnt);
  printf("[bag] ─────────────────────────────────────────────────────────\n\n");

  return result;
}

// ---------------------------------------------------------------------------

WCDatabaseExporterDev::~WCDatabaseExporterDev() {}

// ---------------------------------------------------------------------------
// initialize_rtabmap_database
// ---------------------------------------------------------------------------
// Load DB, augment landmark links with bag detections, run 3-pass GTSAM
// re-optimization, then store tag_obs_ and viz_poses_ for the grid overlay.
// ---------------------------------------------------------------------------
bool WCDatabaseExporterDev::initialize_rtabmap_database()
{
  // Never write into the input DB.  Rtabmap::init() writes its search index
  // into the DB it opens, and closing with save (the destructor default)
  // rewrites Admin.opt_poses -- the live session's graph.  Work on a /tmp copy
  // that is deleted when this function returns; rtabmap_database_path_ stays
  // the original (bag lookup and metrics use its name).
  const std::string work_db = "/tmp/database_exporter_" + std::to_string(::getpid()) + ".db";
  std::filesystem::copy_file(rtabmap_database_path_, work_db,
                             std::filesystem::copy_options::overwrite_existing);
  struct RemoveOnExit {
    std::string p;
    ~RemoveOnExit() { std::error_code ec; std::filesystem::remove(p, ec); }
  } remove_work_db{work_db};
  printf("Working on a copy of the database: %s\n", work_db.c_str());

  rtabmap::ParametersMap parameters;
  rtabmap::DBDriver *driver = rtabmap::DBDriver::create();

  std::multimap<int, rtabmap::Link> raw_landmark_links;
  if (driver->openConnection(work_db)) {
    parameters = driver->getLastParameters();
    std::multimap<int, rtabmap::Link> all_raw_links;
    driver->getAllLinks(all_raw_links, true, true);
    for (const auto &kv : all_raw_links)
      if (kv.second.type() == rtabmap::Link::kLandmark)
        raw_landmark_links.insert(kv);
    driver->closeConnection(false);
  } else {
    std::cerr << "Failed to open database: " << rtabmap_database_path_ << "\n";
    return false;
  }
  delete driver;

  // Print Marker parameters so we can verify tag size matches physical tags.
  printf("\n--- Marker parameters from DB ---\n");
  for (const auto &[key, val] : parameters)
    if (key.find("Marker/") != std::string::npos)
      printf("  %s = %s\n", key.c_str(), val.c_str());
  printf("---------------------------------\n\n");

  // Load DB and get optimized pose graph.
  UTimer timer;
  printf("Loading database: %s\n", rtabmap_database_path_.c_str());
  rtabmap::Rtabmap rtabmap;
  rtabmap.init(parameters, work_db);
  printf("Loaded in %.2f s\n", timer.ticks());

  printf("Optimizing map...\n");
  rtabmap.getGraph(optimizedPoses, links, true, true, &nodes, true, true, true, true);
  printf("Optimized: %zu poses (%.2f s)\n", optimizedPoses.size(), timer.ticks());
  rtabmap.close(false);  // everything needed is in memory now; don't save into the copy either

  if (optimizedPoses.empty()) {
    std::cerr << "No optimized poses found\n";
    return false;
  }

  // ── Odometry turn-scale correction (env WC_ODOM_YAW_SCALE, default 1 = off) ─
  // Wheel odometry over-rotates by 4.1% on every GT recording (odom/GT turn
  // ratio 1.041 on 10-07-01, 10-07-03, 10-08-01; distance scale 1.00), i.e. the
  // configured track width is ~4% small.  Scaling each odometry link's yaw by
  // 1/1.041 = 0.9606 undoes it offline; the translation is rotated by the
  // half-step yaw change so the step keeps its mid-step heading.  The graph
  // re-solve below then starts from these corrected links.
  double odom_yaw_scale = 1.0;
  if (const char *e = std::getenv("WC_ODOM_YAW_SCALE")) odom_yaw_scale = std::atof(e);
  if (odom_yaw_scale != 1.0) {
    int n_scaled = 0;
    for (auto &kv : links) {
      rtabmap::Link &lk = kv.second;
      if (lk.type() != rtabmap::Link::kNeighbor && lk.type() != rtabmap::Link::kNeighborMerged) continue;
      float x, y, z, roll, pitch, yaw;
      lk.transform().getTranslationAndEulerAngles(x, y, z, roll, pitch, yaw);
      const float dyaw = (float)((odom_yaw_scale - 1.0) * yaw);
      const float c = std::cos(0.5f * dyaw), s = std::sin(0.5f * dyaw);
      lk.setTransform(rtabmap::Transform(c * x - s * y, s * x + c * y, z, roll, pitch, yaw + dyaw));
      ++n_scaled;
    }
    printf("[odom] scaled yaw of %d odometry links by %.4f\n", n_scaled, odom_yaw_scale);
    add_metric("odom yaw scale: %.4f applied to %d neighbour links (WC_ODOM_YAW_SCALE)", odom_yaw_scale, n_scaled);
  } else {
    add_metric("odom yaw scale: 1.0 (off)");
  }

  // ── DEV: camera yaw correction on tag detections (env WC_CAMERA_YAW_DEG) ─
  if (const double cy = camera_yaw_rad(); cy != 0.0) {
    int n = 0, n_raw = 0;
    for (auto &kv : links)
      if (kv.second.type() == rtabmap::Link::kLandmark) { rotate_landmark_link(kv.second, cy); ++n; }
    for (auto &kv : raw_landmark_links) { rotate_landmark_link(kv.second, cy); ++n_raw; }
    printf("[camera] yaw correction %.2f deg on %d graph / %d raw landmark links\n", cy * 180.0 / M_PI, n, n_raw);
    add_metric("camera yaw correction (WC_CAMERA_YAW_DEG): %.2f deg on scans and tag detections (%d graph / %d raw landmark links)",
               cy * 180.0 / M_PI, n, n_raw);
  }

  // ── DEV: drop tags (env WC_TAG_EXCLUDE="44" or "44,9") ──────────────────
  // R01_02: tag 44 is seen 162 times from ~2.25 m during t = 210–284 s; its
  // detections scatter 52 mm (GT poses), the worst of 40 tags on every GT
  // recording, and with that many of them the graph bends the trajectory to
  // fit them: graph error 70 mm vs odometry 21 mm in 180–300 s, and the
  // E3–H3 wall splits into extra lines.  Without tag 44 (Python replica of
  // this graph): ATE rmse 53.3 → 42.8 mm on R01_02; 10-12-01 / 10-08-01 within 1 mm.
  // Every link touching the tag and its landmark pose are removed.
  {
    std::set<int> excl;
    if (const char *e = std::getenv("WC_TAG_EXCLUDE")) {
      std::string s = e;
      for (char &ch : s) if (ch == ',' || ch == ';') ch = ' ';
      std::istringstream ss(s);
      int id;
      while (ss >> id) excl.insert(std::abs(id));
    }
    if (!excl.empty()) {
      auto touches = [&](const rtabmap::Link &lk) {
        return (lk.to() < 0 && excl.count(-lk.to())) || (lk.from() < 0 && excl.count(-lk.from()));
      };
      int n_links = 0, n_raw = 0;
      for (auto it = links.begin(); it != links.end();)
        if (touches(it->second)) { it = links.erase(it); ++n_links; } else ++it;
      for (auto it = raw_landmark_links.begin(); it != raw_landmark_links.end();)
        if (touches(it->second)) { it = raw_landmark_links.erase(it); ++n_raw; } else ++it;
      std::string list;
      for (int id : excl) { optimizedPoses.erase(-id); list += (list.empty() ? "" : ",") + std::to_string(id); }
      printf("[tags] excluded tag(s) %s: removed %d graph links, %d raw landmark links\n", list.c_str(), n_links, n_raw);
      add_metric("tags excluded (WC_TAG_EXCLUDE): %s — %d graph links, %d raw landmark links removed",
                 list.c_str(), n_links, n_raw);
    }
  }

  // ── DEV: odometry link weight (env WC_ODOM_INFO_SCALE, default 1) ───────
  // The DB gives every odometry link σ ≈ 3.2 cm / 1.8° per step, far looser
  // than the wheels are since the wheel-base fix, so tag noise moves the
  // trajectory easily.  Multiplying the odometry information by 16 (σ / 4):
  // 10-12-01 ATE rmse 44.8 → 37.8 mm, p90 74 → 56 mm; with tag 44 excluded
  // R01_02 42.8 → 44.1 mm but the E3–H3 wall becomes one line.  (An older
  // test on 09-29-2.db judged tighter odometry worse, but by the wall metric,
  // before ground truth and before the wheel-base fix.)
  {
    double s = 1.0;
    if (const char *e = std::getenv("WC_ODOM_INFO_SCALE")) s = std::atof(e);
    if (s > 0.0 && s != 1.0) {
      int n = 0;
      for (auto &kv : links) {
        rtabmap::Link &lk = kv.second;
        if (lk.type() != rtabmap::Link::kNeighbor && lk.type() != rtabmap::Link::kNeighborMerged) continue;
        lk.setInfMatrix(lk.infMatrix() * s);
        ++n;
      }
      printf("[odom] scaled the information of %d odometry links by %.2f\n", n, s);
      add_metric("odometry link weight: information x%.2f on %d neighbour links (WC_ODOM_INFO_SCALE)", s, n);
    }
  }

  // ── Tag-link weights, used by this graph stage AND the tag GTSAM passes ─
  // The live DB weights over-constrain the tags, and tag orientation is much
  // noisier than tag position, so: σ ≈ 45 cm per observation, orientation
  // effectively ignored (position-only landmarks).
  // Previous tag-GTSAM values: lin 0.02 / ang 0.05; before that 0.005.
  constexpr double landmark_var_linear  = 0.2;
  constexpr double landmark_var_angular = 999.0;

  // Re-solve RTAB-Map's graph with every stored landmark link reweighted,
  // starting from RTAB-Map's own solution (odometry links keep DB weights).
  {
    cv::Mat lm_inf = cv::Mat::zeros(6, 6, CV_64FC1);
    lm_inf.at<double>(0,0) = lm_inf.at<double>(1,1) = lm_inf.at<double>(2,2) = 1.0 / landmark_var_linear;
    lm_inf.at<double>(3,3) = lm_inf.at<double>(4,4) = lm_inf.at<double>(5,5) = 1.0 / landmark_var_angular;
    int n_lm = 0;
    for (auto &kv : links)
      if (kv.second.type() == rtabmap::Link::kLandmark) { kv.second.setInfMatrix(lm_inf); ++n_lm; }
    rtabmap::Optimizer *opt = rtabmap::Optimizer::create(parameters);
    cv::Mat cov;
    std::map<int, rtabmap::Transform> g =
        opt->optimize(optimizedPoses.lower_bound(1)->first, optimizedPoses, links, cov);
    delete opt;
    if (!g.empty()) optimizedPoses = g;
    else std::cerr << "[graph] landmark-reweighted re-optimization failed, keeping RTAB-Map's poses\n";
    add_metric("graph stage: %d landmark links reweighted lin=%.3f ang=%.1f, re-opt %s",
               n_lm, landmark_var_linear, landmark_var_angular, g.empty() ? "FAILED" : "ok");
  }
  rtabmap_graph_poses_ = optimizedPoses;

  // ── Tighten odometry (neighbour) links ────────────────────────────────
  // The DB stores a constant default on every neighbour link (σ ≈ 3.2 cm,
  // 1.8° per ~4 cm step on 09-29-2.db), far looser than real odometry over a
  // few cm.  That lets tag-detection noise jitter each node independently and
  // stops nodes between ICP query nodes from following their corrections.
  // Replace the x, y, yaw information with σ = base + fraction of the step.
  //
  // DISABLED: on 09-29-2.db this made everything worse (tag spread 14.9 →
  // 26.7 mm, tag-GTSAM global walls 26.9 → 34.6 mm, final ICP 24 → 29 mm).
  // The odometry genuinely drifts at the few-cm scale, and the loose DB
  // default is what lets the tags correct it.
  constexpr bool kTightenOdometry = false;
  if (kTightenOdometry) {
    constexpr double kOdomSigmaLinBase = 0.005;               // m
    constexpr double kOdomSigmaLinFrac = 0.02;                // × step length
    constexpr double kOdomSigmaYawBase = 0.3 * M_PI / 180.0;  // rad
    constexpr double kOdomSigmaYawFrac = 0.05;                // × |Δyaw|

    int n_tight = 0;
    std::vector<double> old_sig;
    for (auto &kv : links) {
      rtabmap::Link &lk = kv.second;
      if (lk.type() != rtabmap::Link::kNeighbor &&
          lk.type() != rtabmap::Link::kNeighborMerged) continue;
      cv::Mat inf = lk.infMatrix().clone();
      old_sig.push_back(std::sqrt(1.0 / inf.at<double>(0, 0)));
      const rtabmap::Transform &t = lk.transform();
      const double s_lin = kOdomSigmaLinBase + kOdomSigmaLinFrac * std::hypot(t.x(), t.y());
      const double s_yaw = kOdomSigmaYawBase + kOdomSigmaYawFrac * std::abs(t.theta());
      const int idx3[3] = {0, 1, 5};
      for (int a : idx3)
        for (int b : idx3) inf.at<double>(a, b) = 0.0;
      inf.at<double>(0, 0) = inf.at<double>(1, 1) = 1.0 / (s_lin * s_lin);
      inf.at<double>(5, 5) = 1.0 / (s_yaw * s_yaw);
      lk.setInfMatrix(inf);
      ++n_tight;
    }
    double old_med = 0.0;
    if (!old_sig.empty()) {
      std::nth_element(old_sig.begin(), old_sig.begin() + old_sig.size() / 2, old_sig.end());
      old_med = old_sig[old_sig.size() / 2];
    }
    printf("[odom] tightened %d neighbour links (DB σ_xy median %.1f mm)\n", n_tight, 1000.0 * old_med);
    add_metric("odom links: tightened %d  sigma_xy=%.1fmm+%.0f%%*step  sigma_yaw=%.2fdeg+%.0f%%*dyaw  (DB was %.1f mm)",
               n_tight, 1000.0 * kOdomSigmaLinBase, 100.0 * kOdomSigmaLinFrac,
               kOdomSigmaYawBase * 180.0 / M_PI, 100.0 * kOdomSigmaYawFrac, 1000.0 * old_med);
  } else {
    add_metric("odom links: DB default (tightening disabled)");
  }

  // Augment raw_landmark_links with bag detections.
  {
    auto bag_obs = load_bag_detections(parameters);
    if (!bag_obs.empty()) {
      int n_before = (int)raw_landmark_links.size();
      cv::Mat dummy_inf = cv::Mat::eye(6, 6, CV_64FC1);
      for (const auto &[node_id, id_tf] : bag_obs) {
        auto [tag_id, body_to_tag] = id_tf;
        rtabmap::Link lk(node_id, -tag_id,
                         rtabmap::Link::kLandmark, body_to_tag, dummy_inf);
        if (const double cy = camera_yaw_rad(); cy != 0.0) rotate_landmark_link(lk, cy);
        raw_landmark_links.insert({node_id, lk});
      }
      printf("[bag] raw_landmark_links: %d (DB) + %d (bag) = %d total\n",
             n_before, (int)bag_obs.size(), (int)raw_landmark_links.size());
    } else {
      printf("[bag] No bag observations — using DB detections only (%zu)\n",
             raw_landmark_links.size());
    }
  }

  // Print link type counts.
  auto count_links = [](const std::multimap<int, rtabmap::Link> &lks,
                        rtabmap::Link::Type t) {
    return std::count_if(lks.begin(), lks.end(),
      [t](const auto &kv){ return kv.second.type() == t; });
  };
  printf("DB link counts: neighbor=%ld  global_closure=%ld  local_closure=%ld  landmark=%ld\n",
         count_links(links, rtabmap::Link::kNeighbor),
         count_links(links, rtabmap::Link::kGlobalClosure),
         count_links(links, rtabmap::Link::kLocalSpaceClosure),
         count_links(links, rtabmap::Link::kLandmark));

  {
    std::set<int> tag_ids;
    for (const auto &kv : raw_landmark_links) tag_ids.insert(std::abs(kv.second.to()));
    auto f3 = parameters.find(rtabmap::Parameters::kRegForce3DoF());
    add_metric("db=%s  robot_nodes=%ld  force3DoF=%s",
               UFile::getName(rtabmap_database_path_).c_str(),
               std::count_if(optimizedPoses.begin(), optimizedPoses.end(),
                             [](const auto &kv) { return kv.first > 0; }),
               f3 != parameters.end() ? f3->second.c_str() : "unset");
    add_metric("links: neighbor=%ld global=%ld local=%ld landmark_obs=%zu (bag=%s) distinct_tags=%zu",
               count_links(links, rtabmap::Link::kNeighbor),
               count_links(links, rtabmap::Link::kGlobalClosure),
               count_links(links, rtabmap::Link::kLocalSpaceClosure),
               raw_landmark_links.size(),
               count_links(links, rtabmap::Link::kLandmark) < (long)raw_landmark_links.size() ? "yes" : "no",
               tag_ids.size());
  }

  // Print loop closure details.
  printf("\nLoop closures:\n");
  for (const auto &kv : links) {
    const rtabmap::Link &lk = kv.second;
    if (lk.type() == rtabmap::Link::kNeighbor  ||
        lk.type() == rtabmap::Link::kLandmark   ||
        lk.type() == rtabmap::Link::kPosePrior) continue;
    const rtabmap::Transform &t = lk.transform();
    Eigen::Matrix3f R = t.toEigen3f().linear();
    float yaw_deg = std::atan2(R(1,0), R(0,0)) * 180.f / M_PI;
    const char *type_str =
      lk.type() == rtabmap::Link::kGlobalClosure     ? "Global" :
      lk.type() == rtabmap::Link::kLocalSpaceClosure ? "LocalSpace" :
      lk.type() == rtabmap::Link::kUserClosure       ? "User" : "Other";
    printf("  [%s] %d → %d : Δtrans=%.3f m, Δyaw=%.1f°\n",
           type_str, lk.from(), lk.to(),
           std::sqrt(t.x()*t.x() + t.y()*t.y() + t.z()*t.z()), yaw_deg);
  }

  // Print initial per-tag position spread.
  {
    struct Obs { float x, y, z; };
    std::map<int, std::vector<Obs>> tag_obs;
    for (const auto &kv : raw_landmark_links) {
      const rtabmap::Link &lk = kv.second;
      auto it = optimizedPoses.find(lk.from());
      if (it == optimizedPoses.end()) continue;
      rtabmap::Transform wt = it->second * lk.transform();
      tag_obs[std::abs(lk.to())].push_back({wt.x(), wt.y(), wt.z()});
    }
    printf("\n--- AprilTag world positions (pre-GTSAM) ---\n");
    printf("  %4s  %5s  %8s  %8s  %8s  %8s\n", "Tag", "N", "MeanX", "MeanY", "MeanZ", "StdXYZ");
    for (auto &[tag_id, obs] : tag_obs) {
      int n = obs.size();
      float mx=0, my=0, mz=0;
      for (auto &o : obs) { mx+=o.x; my+=o.y; mz+=o.z; }
      mx/=n; my/=n; mz/=n;
      float sx=0, sy=0, sz=0;
      for (auto &o : obs) {
        sx+=(o.x-mx)*(o.x-mx); sy+=(o.y-my)*(o.y-my); sz+=(o.z-mz)*(o.z-mz);
      }
      printf("  %4d  %5d  %8.3f  %8.3f  %8.3f  %8.4f\n",
             tag_id, n, mx, my, mz,
             std::sqrt((sx+sy+sz)/(3.0f*n)));
    }
    printf("--------------------------------------------\n\n");
  }

  // ── GTSAM 3-pass re-optimization ─────────────────────────────────────
  // landmark_var_linear / landmark_var_angular: defined at the graph stage above.
  constexpr double prior_var_linear_base = 0.001;   // tighter: prior strength floored at ~3cm
  constexpr double prior_var_angular_base= 0.001;
  // 1 pass: at lin 0.2 / ang 999 the graph stage already solves this problem
  // (pass 1 moves robot nodes 0.0 mm on 09-29-2, 10-07-01, 10-07-03), and passes
  // 2-3 were no-ops even at 0.02.  Pass 1 is kept because it adds the per-tag
  // position priors and tag poses that the ICP rounds optimize against.
  constexpr int    n_reopt_passes        = 1;
  constexpr float  kMaxTagStd            = 0.45f;
  add_metric("gtsam config: landmark_var lin=%.4f ang=%.4f  prior_var base lin=%.4f ang=%.4f  max_tag_std=%.2f",
             landmark_var_linear, landmark_var_angular,
             prior_var_linear_base, prior_var_angular_base, kMaxTagStd);

  // Tag (negative-ID) poses from the last successful pass; seeds the ICP re-opt.
  std::map<int, rtabmap::Transform> tag_poses;

  {
    // base_links: odometry + loop closures only (no kLandmark).
    // We add kLandmark links per-pass so bag observations are included.
    std::multimap<int, rtabmap::Link> base_links;
    for (const auto &kv : links)
      if (kv.second.type() != rtabmap::Link::kLandmark)
        base_links.insert(kv);

    struct TagAccum {
      float tx=0,ty=0,tz=0, qx=0,qy=0,qz=0,qw=0; int n=0;
      std::vector<std::array<float,3>> pts;
    };

    for (int pass = 1; pass <= n_reopt_passes; ++pass) {
      printf("\n--- Re-optimization pass %d/%d ---\n", pass, n_reopt_passes);

      // Compute consensus tag world pose from current optimizedPoses.
      std::map<int, TagAccum> tag_accum;
      for (const auto &kv : raw_landmark_links) {
        const rtabmap::Link &lk = kv.second;
        auto it = optimizedPoses.find(lk.from());
        if (it == optimizedPoses.end()) continue;
        rtabmap::Transform wt = it->second * lk.transform();
        int tid = std::abs(lk.to());
        auto &a = tag_accum[tid];
        a.tx += wt.x(); a.ty += wt.y(); a.tz += wt.z();
        Eigen::Quaternionf q = wt.getQuaternionf();
        if (a.n > 0 && (q.x()*a.qx + q.y()*a.qy + q.z()*a.qz + q.w()*a.qw) < 0.f)
          q.coeffs() = -q.coeffs();
        a.qx += q.x(); a.qy += q.y(); a.qz += q.z(); a.qw += q.w();
        a.pts.push_back({wt.x(), wt.y(), wt.z()});
        a.n++;
      }

      std::map<int, rtabmap::Transform> tag_world_mean;
      std::map<int, float> tag_std;
      std::map<int, float> tag_std_xyz;
      for (auto &[tid, a] : tag_accum) {
        float inv  = 1.0f / a.n;
        float norm = std::sqrt(a.qx*a.qx + a.qy*a.qy + a.qz*a.qz + a.qw*a.qw);
        float mx = a.tx*inv, my = a.ty*inv, mz = a.tz*inv;
        tag_world_mean[tid] = rtabmap::Transform(
            mx, my, mz, a.qx/norm, a.qy/norm, a.qz/norm, a.qw/norm);
        float sx=0, sy=0, sz=0;
        for (auto &p : a.pts) {
          sx += (p[0]-mx)*(p[0]-mx);
          sy += (p[1]-my)*(p[1]-my);
          sz += (p[2]-mz)*(p[2]-mz);
        }
        tag_std[tid]     = std::sqrt((sx + sy) / (2.0f * a.n));
        tag_std_xyz[tid] = std::sqrt((sx + sy + sz) / (3.0f * a.n));
      }

      // Print per-tag std.
      printf("  Tag priors (std → action):\n  ");
      for (const auto &[tid, s] : tag_std) {
        if (s > kMaxTagStd)
          printf("%d(σ=%.3f→SKIP) ", tid, s);
        else
          printf("%d(σ=%.3f→v=%.4f) ", tid, s,
                 prior_var_linear_base + (double)(s*s));
      }
      printf("\n");

      // Build reopt_links = base + filtered kLandmark + kPosePrior per good tag.
      std::multimap<int, rtabmap::Link> reopt_links = base_links;

      cv::Mat lm_inf = cv::Mat::zeros(6, 6, CV_64FC1);
      lm_inf.at<double>(0,0) = lm_inf.at<double>(1,1) = lm_inf.at<double>(2,2) =
          1.0 / landmark_var_linear;
      lm_inf.at<double>(3,3) = lm_inf.at<double>(4,4) = lm_inf.at<double>(5,5) =
          1.0 / landmark_var_angular;

      int n_lm_added = 0, n_lm_skipped = 0;
      for (const auto &kv : raw_landmark_links) {
        const rtabmap::Link &lk = kv.second;
        int tid = std::abs(lk.to());
        float s = tag_std.count(tid) ? tag_std.at(tid) : 0.0f;
        if (s > kMaxTagStd) { ++n_lm_skipped; continue; }
        if (optimizedPoses.find(lk.from()) == optimizedPoses.end()) continue;
        reopt_links.insert({lk.from(),
            rtabmap::Link(lk.from(), lk.to(),
                          rtabmap::Link::kLandmark, lk.transform(), lm_inf)});
        ++n_lm_added;
      }

      int n_tag_priors = 0;
      for (const auto &[tid, mean] : tag_world_mean) {
        float s = tag_std.count(tid) ? tag_std.at(tid) : 0.0f;
        if (s > kMaxTagStd) continue;
        double var_lin = prior_var_linear_base  + (double)(s * s);
        double var_ang = prior_var_angular_base + (double)(s * s);
        cv::Mat tag_inf = cv::Mat::zeros(6, 6, CV_64FC1);
        tag_inf.at<double>(0,0) = tag_inf.at<double>(1,1) = tag_inf.at<double>(2,2) =
            1.0 / var_lin;
        tag_inf.at<double>(3,3) = tag_inf.at<double>(4,4) = tag_inf.at<double>(5,5) =
            1.0 / var_ang;
        reopt_links.insert({-tid,
            rtabmap::Link(-tid, -tid, rtabmap::Link::kPosePrior, mean, tag_inf)});
        ++n_tag_priors;
      }
      printf("  %d landmark links (%d skipped), %d tag priors\n",
             n_lm_added, n_lm_skipped, n_tag_priors);

      // Initial poses: robot nodes from optimizedPoses + accepted tag nodes.
      // getGraph(includeLandmarks=true) puts negative-ID tag poses into
      // optimizedPoses; we exclude them so GTSAM only sees nodes with edges.
      std::map<int, rtabmap::Transform> initial_poses;
      for (const auto &[nid, pose] : optimizedPoses)
        if (nid > 0) initial_poses[nid] = pose;
      for (const auto &[tid, mean] : tag_world_mean) {
        float s = tag_std.count(tid) ? tag_std.at(tid) : 0.0f;
        if (s <= kMaxTagStd) initial_poses[-tid] = mean;
      }

      int root_id = initial_poses.lower_bound(1)->first;
      rtabmap::ParametersMap opt_params = parameters;
      opt_params[rtabmap::Parameters::kOptimizerRobust()] = "true";
      rtabmap::Optimizer *opt = rtabmap::Optimizer::create(opt_params);
      cv::Mat out_cov;
      std::map<int, rtabmap::Transform> reoptPoses =
          opt->optimize(root_id, initial_poses, reopt_links, out_cov);
      delete opt;

      if (!reoptPoses.empty()) {
        // Measure change across robot nodes.
        double sum_trans = 0, sum_rot = 0; int n_moved = 0;
        std::vector<double> dt_vals, dr_vals;
        for (const auto &[nid, prev] : optimizedPoses) {
          if (nid < 0) continue;
          auto next_it = reoptPoses.find(nid);
          if (next_it == reoptPoses.end()) continue;
          rtabmap::Transform delta = prev.inverse() * next_it->second;
          float r, p, y; delta.getEulerAngles(r, p, y);
          double dt = delta.getNorm();
          double dr = std::sqrt((double)r*r + p*p + y*y) * 180.0 / M_PI;
          sum_trans += dt; sum_rot += dr; ++n_moved;
          dt_vals.push_back(dt); dr_vals.push_back(dr);
        }
        if (n_moved > 0) {
          double mean_t = sum_trans / n_moved, mean_r = sum_rot / n_moved;
          double var_t = 0, var_r = 0;
          for (int k = 0; k < n_moved; ++k) {
            var_t += (dt_vals[k]-mean_t)*(dt_vals[k]-mean_t);
            var_r += (dr_vals[k]-mean_r)*(dr_vals[k]-mean_r);
          }
          printf("  %d robot nodes: mean Δtrans=%.4f m (σ=%.4f), Δrot=%.2f° (σ=%.2f)\n",
                 n_moved, mean_t, std::sqrt(var_t/n_moved),
                 mean_r, std::sqrt(var_r/n_moved));
        }
        printf("  Pass %d done: %zu total poses (%zu robot)\n",
               pass, reoptPoses.size(),
               (size_t)std::count_if(reoptPoses.begin(), reoptPoses.end(),
                   [](const auto &kv){ return kv.first > 0; }));

        tag_poses.clear();
        for (const auto &[nid, pose] : reoptPoses) {
          if (nid > 0) optimizedPoses[nid] = pose;
          else         tag_poses[nid]      = pose;
        }
        links = reopt_links;

        // Recompute tag spread on updated poses.
        std::map<int, TagAccum> post_accum;
        for (const auto &kv : raw_landmark_links) {
          const rtabmap::Link &lk = kv.second;
          auto it2 = optimizedPoses.find(lk.from());
          if (it2 == optimizedPoses.end()) continue;
          rtabmap::Transform wt = it2->second * lk.transform();
          int tid = std::abs(lk.to());
          auto &a = post_accum[tid];
          a.tx += wt.x(); a.ty += wt.y(); a.tz += wt.z();
          a.pts.push_back({wt.x(), wt.y(), wt.z()});
          a.n++;
        }
        printf("  Tag StdXYZ (before → after):\n");
        printf("  %5s %5s  %8s  %8s  %8s\n", "Tag", "N", "Before", "After", "Delta");
        float sum_delta = 0.0f, sum_before = 0.0f, sum_after = 0.0f;
        float worst_after = 0.0f; int worst_tid = -1, n_shown = 0;
        for (auto &[tid, a] : post_accum) {
          float inv = 1.0f / a.n;
          float mx = a.tx*inv, my = a.ty*inv, mz = a.tz*inv;
          float sx=0, sy=0, sz=0;
          for (auto &p : a.pts) {
            sx += (p[0]-mx)*(p[0]-mx);
            sy += (p[1]-my)*(p[1]-my);
            sz += (p[2]-mz)*(p[2]-mz);
          }
          float after  = std::sqrt((sx+sy+sz)/(3.0f*a.n));
          float before = tag_std_xyz.count(tid) ? tag_std_xyz.at(tid) : after;
          printf("  %5d %5d  %8.4f  %8.4f  %+8.4f\n", tid, a.n, before, after, after-before);
          sum_delta += after - before;
          sum_before += before; sum_after += after;
          if (after > worst_after) { worst_after = after; worst_tid = tid; }
          ++n_shown;
        }
        add_metric("gtsam pass %d: tags_used=%d lm_links=%d (skipped %d)  tag_std_mean %.1f -> %.1f mm  worst tag %d %.1f mm  robot mean dmove=%.1f mm",
                   pass, n_tag_priors, n_lm_added, n_lm_skipped,
                   n_shown ? 1000.0 * sum_before / n_shown : 0.0,
                   n_shown ? 1000.0 * sum_after / n_shown : 0.0,
                   worst_tid, 1000.0 * worst_after,
                   n_moved ? 1000.0 * sum_trans / n_moved : 0.0);
        if (n_shown > 0)
          printf("  Mean StdXYZ change: %+.4f m (%s)\n",
                 sum_delta/n_shown,
                 sum_delta/n_shown < 0.0f ? "improved" : "worsened");
      } else {
        std::cerr << "  Pass " << pass << " failed, keeping previous poses\n";
        add_metric("gtsam pass %d: FAILED (kept previous poses)", pass);
        break;
      }
    }
  }

  // ── Scan-to-local-map ICP on top of the tag graph ─────────────────────
  clean_scans();
  refine_poses_with_local_map_icp(parameters, tag_poses);

  // ── Wall-direction (Manhattan) heading priors ─────────────────────────
  // A single scan measures its wall directions without any pose (opposite
  // walls parallel within ~0.5° on 10-08-01).  In a rectilinear building each
  // scan's walls should line up with the building axes once its heading is
  // right, so the mismatch e = wrap90(yaw + wall_dir_scan - axis_world) is a
  // per-node heading-error measurement.  On 10-08-01 it tracks the GT heading
  // error (r = 0.65 per node, 0.86 smoothed over ~3–6 m); correcting by it cut
  // the p90 heading error 1.90° -> ~1.4°.  Single-node e is noisy (~1°), so it
  // goes in as a weak heading-only prior (σ = kMhSigmaDeg) and the odometry
  // links average it.  This fixes slow heading drift, e.g. the 2.5° slanted
  // wall / tapering room on 10-08-01.  Disable with env WC_MANHATTAN=0.
  {
    bool enabled = true;
    if (const char *e = std::getenv("WC_MANHATTAN")) enabled = std::atoi(e) != 0;
    // Per-node e is noisy (~1°); with raw e as targets (σ 3°, first run on
    // 10-08-01) ATE improved 10% but RPE1m worsened 7% and the slanted E3–I3
    // wall only went -2.0° -> -1.8°.  So the targets are now a running median
    // of e over kMhSmooth consecutive measured nodes (~3 m; GT corr 0.86 when
    // smoothed vs 0.65 raw), which carries only the slow drift, and σ can be
    // tighter.  Env WC_MANHATTAN_SIGMA / WC_MANHATTAN_SMOOTH override.
    double kMhSigmaDeg = 2.0;              // per-node heading prior σ
    int    kMhSmooth   = 31;               // running-median window (measured nodes), 1 = raw
    if (const char *e = std::getenv("WC_MANHATTAN_SIGMA"))  kMhSigmaDeg = std::atof(e);
    if (const char *e = std::getenv("WC_MANHATTAN_SMOOTH")) kMhSmooth = std::max(1, std::atoi(e));
    constexpr double kMhMaxErrDeg = 8.0;   // larger mismatch → not a Manhattan wall view, skip
    constexpr double kMhMinCoh    = 0.5;   // coherence of the scan's wall directions
    constexpr int    kMhMinPts    = 500;
    constexpr float  kMhRes       = 0.025f;

    // Dominant wall direction (rad, mod 90°, in (-45°, 45°]) and coherence of
    // a 2D point set: gradient directions of a small hit image, circular mean
    // of 4·phi weighted by gradient magnitude.
    auto dominant_dir = [](const std::vector<cv::Point2f> &xy, float res, double &coh) -> double {
      coh = 0.0;
      if (xy.size() < 10) return 0.0;
      float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
      for (const auto &p : xy) { x0 = std::min(x0, p.x); y0 = std::min(y0, p.y); x1 = std::max(x1, p.x); y1 = std::max(y1, p.y); }
      const int W = (int)((x1 - x0) / res) + 5, H = (int)((y1 - y0) / res) + 5;
      if ((double)W * H > 4e6) return 0.0;
      cv::Mat img(H, W, CV_32F, cv::Scalar(0));
      for (const auto &p : xy) img.at<float>((int)((p.y - y0) / res) + 2, (int)((p.x - x0) / res) + 2) += 1.0f;
      cv::min(img, cv::Scalar(10.0), img);
      cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
      cv::Mat gx, gy;
      cv::Sobel(img, gx, CV_32F, 1, 0, 3);
      cv::Sobel(img, gy, CV_32F, 0, 1, 3);
      double s = 0, c = 0, wsum = 0;
      for (int r = 0; r < H; ++r)
        for (int k = 0; k < W; ++k) {
          const double a = gx.at<float>(r, k), b = gy.at<float>(r, k), m = std::hypot(a, b);
          if (m < 1e-3) continue;
          const double ph = std::atan2(b, a);
          s += m * std::sin(4 * ph); c += m * std::cos(4 * ph); wsum += m;
        }
      if (wsum <= 0) return 0.0;
      coh = std::hypot(s, c) / wsum;
      return 0.25 * std::atan2(s, c);
    };
    auto wrap90 = [](double a) { const double q = M_PI / 2; a = std::fmod(a + q / 2, q); if (a < 0) a += q; return a - q / 2; };

    if (!enabled) {
      add_metric("manhattan heading priors: off (WC_MANHATTAN=0)");
    } else {
      // Building axis from all wall-band points at the current poses.
      std::vector<cv::Point2f> world;
      for (const auto &[nid, pts] : per_scan_base_) {
        auto pit = optimizedPoses.find(nid);
        if (pit == optimizedPoses.end()) continue;
        const Eigen::Affine3f T = pit->second.toEigen3f();
        for (size_t k = 0; k < pts.size(); k += 20) {
          const auto &p = pts[k];
          if (p.z < 0.10f || p.z > 1.00f) continue;
          const Eigen::Vector3f w = T * Eigen::Vector3f(p.x, p.y, p.z);
          world.emplace_back(w.x(), w.y());
        }
      }
      double coh_w = 0.0;
      const double axis = dominant_dir(world, 0.05f, coh_w);

      // Per-node mismatch → heading prior.
      // Tag priors (kPosePrior on -tid) were never applied before (DB has
      // Optimizer/PriorsIgnored=true).  They can't simply be dropped: GTSAM
      // finds each landmark through a link keyed by its id, and that prior is
      // the only one.  So keep them but zero their position weight (no pull),
      // keeping their angular part so landmarks stay set up exactly as before
      // (tag orientation is decoupled anyway: landmark links have ang var 999).
      std::multimap<int, rtabmap::Link> mh_links;
      for (const auto &kv : links) {
        if (kv.second.type() == rtabmap::Link::kPosePrior && kv.second.from() < 0) {
          rtabmap::Link lk = kv.second;
          cv::Mat inf = lk.infMatrix().clone();
          inf.at<double>(0, 0) = inf.at<double>(1, 1) = inf.at<double>(2, 2) = 1e-6;
          lk.setInfMatrix(inf);
          mh_links.insert({kv.first, lk});
        } else {
          mh_links.insert(kv);
        }
      }
      std::vector<double> e_before;
      int n_prior = 0, n_lowcoh = 0, n_far = 0;
      std::vector<std::pair<int, double>> meas;   // (node id, e) of usable scans, node-ID order
      std::vector<cv::Point2f> xy;
      for (const auto &[nid, pts] : per_scan_base_) {
        auto pit = optimizedPoses.find(nid);
        if (pit == optimizedPoses.end()) continue;
        xy.clear();
        for (const auto &p : pts)
          if (p.z >= 0.10f && p.z <= 1.00f) xy.emplace_back(p.x, p.y);
        if ((int)xy.size() < kMhMinPts) continue;
        double coh = 0.0;
        const double dir = dominant_dir(xy, kMhRes, coh);
        if (coh < kMhMinCoh) { ++n_lowcoh; continue; }
        const rtabmap::Transform &P = pit->second;
        const double e = wrap90(P.theta() + dir - axis);
        if (std::abs(e) > kMhMaxErrDeg * M_PI / 180.0) { ++n_far; continue; }
        e_before.push_back(std::abs(e) * 180.0 / M_PI);
        meas.push_back({nid, e});
      }
      // Smoothed targets: running median of e over kMhSmooth measured nodes
      // (node-ID order ≈ path order), so each prior carries the slow drift
      // around it, not its own scan's noise.
      for (size_t k = 0; k < meas.size(); ++k) {
        const int h = kMhSmooth / 2;
        const size_t a = k > (size_t)h ? k - h : 0, b = std::min(meas.size(), k + h + 1);
        std::vector<double> win;
        for (size_t j = a; j < b; ++j) win.push_back(meas[j].second);
        std::nth_element(win.begin(), win.begin() + win.size() / 2, win.end());
        const double e_s = win[win.size() / 2];
        const rtabmap::Transform &P = optimizedPoses.at(meas[k].first);
        cv::Mat inf = cv::Mat::zeros(6, 6, CV_64FC1);
        inf.at<double>(0, 0) = inf.at<double>(1, 1) = 1.0 / 100.0;      // x, y: σ 10 m (gauge only)
        inf.at<double>(2, 2) = inf.at<double>(3, 3) = inf.at<double>(4, 4) = 1.0 / 100.0;
        const double s = kMhSigmaDeg * M_PI / 180.0;
        inf.at<double>(5, 5) = 1.0 / (s * s);
        mh_links.insert({meas[k].first, rtabmap::Link(meas[k].first, meas[k].first, rtabmap::Link::kPosePrior,
                                            rtabmap::Transform(P.x(), P.y(), 0.0f, 0.0f, 0.0f, (float)(P.theta() - e_s)), inf)});
        ++n_prior;
      }

      std::set<int> referenced;
      for (const auto &kv : mh_links) { referenced.insert(kv.second.from()); referenced.insert(kv.second.to()); }
      std::map<int, rtabmap::Transform> initial;
      for (const auto &[nid, pose] : optimizedPoses) if (nid > 0 || referenced.count(nid)) initial[nid] = pose;
      for (const auto &[nid, pose] : tag_poses) if (referenced.count(nid)) initial[nid] = pose;

      rtabmap::ParametersMap opt_params = parameters;
      opt_params[rtabmap::Parameters::kOptimizerRobust()] = "true";
      opt_params[rtabmap::Parameters::kOptimizerPriorsIgnored()] = "false";  // DB has true → priors would be dropped
      rtabmap::Optimizer *opt = rtabmap::Optimizer::create(opt_params);
      cv::Mat cov;
      std::map<int, rtabmap::Transform> np = n_prior > 0
          ? opt->optimize(initial.lower_bound(1)->first, initial, mh_links, cov)
          : std::map<int, rtabmap::Transform>();
      delete opt;

      if (np.empty()) {
        add_metric("manhattan heading priors: %s (kept previous poses)", n_prior ? "re-opt FAILED" : "no usable scans");
      } else {
        // Re-anchor: with heading priors on, the optimizer no longer holds the
        // root, and the whole solution drifted ~8 cm / 1° (10-08-01, 10-12-01).
        // Move it back rigidly so the first node keeps its pose from before
        // this stage (= the live map frame).  The ground-truth link is
        // initial_pose (GT pose at the first node) × inverse(first node pose),
        // which needs the first node where the live graph had it.
        {
          const int root = initial.lower_bound(1)->first;
          auto before = optimizedPoses.find(root);
          auto after = np.find(root);
          if (before != optimizedPoses.end() && after != np.end()) {
            const rtabmap::Transform A = before->second * after->second.inverse();
            add_metric("manhattan re-anchor: first node %d held at its previous pose (solution moved back %.1f mm / %.2f deg)",
                       root, 1000.0 * std::hypot(A.x(), A.y()), A.theta() * 180.0 / M_PI);
            for (auto &kv : np) kv.second = A * kv.second;
          }
        }
        double sum_t = 0, max_t = 0, sum_r = 0; int n_moved = 0;
        for (const auto &[nid, pose] : np) {
          if (nid > 0) {
            auto pit = optimizedPoses.find(nid);
            if (pit != optimizedPoses.end()) {
              const rtabmap::Transform d = pit->second.inverse() * pose;
              const double dt = std::hypot(d.x(), d.y());
              sum_t += dt; max_t = std::max(max_t, dt); sum_r += std::abs(d.theta()); ++n_moved;
            }
            optimizedPoses[nid] = pose;
          } else {
            tag_poses[nid] = pose;
          }
        }
        // Mismatch after, same nodes/criteria (diagnostic only).
        std::vector<double> e_after;
        for (const auto &[nid, pts] : per_scan_base_) {
          auto pit = optimizedPoses.find(nid);
          if (pit == optimizedPoses.end()) continue;
          xy.clear();
          for (const auto &p : pts) if (p.z >= 0.10f && p.z <= 1.00f) xy.emplace_back(p.x, p.y);
          if ((int)xy.size() < kMhMinPts) continue;
          double coh = 0.0;
          const double dir = dominant_dir(xy, kMhRes, coh);
          if (coh < kMhMinCoh) continue;
          const double e = wrap90(pit->second.theta() + dir - axis);
          if (std::abs(e) <= kMhMaxErrDeg * M_PI / 180.0) e_after.push_back(std::abs(e) * 180.0 / M_PI);
        }
        auto med = [](std::vector<double> v) { if (v.empty()) return 0.0; std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return v[v.size() / 2]; };
        printf("[manhattan] axis %.2f deg (coh %.2f): %d heading priors (σ %.1f deg), skipped %d low-coherence, %d >%.0f deg; |e| median %.2f -> %.2f deg; moved mean %.1f mm / %.2f deg, max %.1f mm\n",
               axis * 180 / M_PI, coh_w, n_prior, kMhSigmaDeg, n_lowcoh, n_far, kMhMaxErrDeg, med(e_before), med(e_after),
               n_moved ? 1000 * sum_t / n_moved : 0.0, n_moved ? sum_r / n_moved * 180 / M_PI : 0.0, 1000 * max_t);
        add_metric("manhattan heading priors: axis %.2f deg (coh %.2f)  %d priors σ=%.1f deg, smoothed over %d nodes (skipped %d low-coh, %d >%.0f deg)  wall mismatch |e| median %.2f -> %.2f deg  moved mean %.1f mm / %.2f deg, max %.1f mm",
                   axis * 180 / M_PI, coh_w, n_prior, kMhSigmaDeg, kMhSmooth, n_lowcoh, n_far, kMhMaxErrDeg, med(e_before), med(e_after),
                   n_moved ? 1000 * sum_t / n_moved : 0.0, n_moved ? sum_r / n_moved * 180 / M_PI : 0.0, 1000 * max_t);
        links = mh_links;
        report_wall_thickness("manhattan", optimizedPoses);   // also dumps poses_manhattan.csv
      }
    }
  }

  // Map origin (env WC_MAP_ORIGIN): first_node (default) re-expresses the
  // final poses so the first graph node is (0, 0) facing 0° — where
  // RTAB-Map was started and /ground_truth/initial_pose was saved, so
  // T_gt<-map = initial_pose (heading-corrected) with no extra term, and the
  // exported map, ground truth and live localization in this map compare
  // directly.  odom keeps RTAB-Map's map frame (origin = where odometry
  // started, before any driving prior to the first node).  Everything after
  // this point (scans, grid, tag overlay) uses the shifted poses; the final
  // poses are dumped to poses_final.csv in the grid's frame.  The per-stage
  // poses_*.csv above stay in RTAB-Map's frame.
  {
    const char *mo_env = std::getenv("WC_MAP_ORIGIN");
    const bool first_node_origin = !(mo_env && std::string(mo_env) == "odom");
    auto first = optimizedPoses.lower_bound(1);
    if (first_node_origin && first != optimizedPoses.end()) {
      const rtabmap::Transform F = first->second;   // copy: the loop below overwrites first->second
      const rtabmap::Transform A = rtabmap::Transform(F.x(), F.y(), F.theta()).inverse();
      for (auto &kv : optimizedPoses) kv.second = A * kv.second;
      for (auto &kv : tag_poses) kv.second = A * kv.second;
      add_metric("map origin: first node %d (was x %.3f y %.3f yaw %.2f deg in RTAB-Map's map frame; WC_MAP_ORIGIN=odom keeps that frame)",
                 first->first, F.x(), F.y(), F.theta() * 180.0 / M_PI);
    } else {
      add_metric("map origin: RTAB-Map map frame (odometry start; WC_MAP_ORIGIN=odom)");
    }
    std::ofstream pf(output_path_ + "/poses_final.csv");
    pf << "id,stamp,x,y,z,qx,qy,qz,qw\n";
    for (const auto &[nid, T] : optimizedPoses) {
      if (nid <= 0) continue;
      auto nit = nodes.find(nid);
      const Eigen::Quaternionf q = T.getQuaternionf();
      pf << nid << ',' << std::setprecision(17) << (nit != nodes.end() ? nit->second.getStamp() : 0.0)
         << std::setprecision(9) << ',' << T.x() << ',' << T.y() << ',' << T.z() << ','
         << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << '\n';
    }
  }

  // Snapshot final poses and tag observations for the grid overlay.
  viz_poses_ = optimizedPoses;

  tag_obs_.clear();
  for (const auto &kv : raw_landmark_links) {
    const rtabmap::Link &lk = kv.second;
    auto it = optimizedPoses.find(lk.from());
    if (it == optimizedPoses.end()) continue;
    rtabmap::Transform wt = it->second * lk.transform();
    tag_obs_[std::abs(lk.to())].push_back({wt.x(), wt.y(), wt.z()});
  }

  return true;
}

// ---------------------------------------------------------------------------
// clean_scans
// ---------------------------------------------------------------------------
// Cleans each node's depth-projected scan in base_link (x-forward, z up from
// floor; the scan's localTransform is rotation-only) into per_scan_base_.
//   1. Crop z to [kZlo, kZhi] and XY range to [kRangeMin, kRangeMax].
//      Stereo depth σ grows ~r² (≈0.4 cm @1 m, ≈1.7 cm @2.5 m on 09-29-2.db).
//   2. 2D radius outlier removal on the XY-collapsed cloud.  The grid only
//      uses XY, and real walls/obstacles stack many heights onto one line, so
//      they are dense in 2D even at grazing angles where a 3D radius filter
//      strips them.  Flying pixels at depth edges and stereo speckle stay
//      sparse in 2D and get removed (~1–8% of points per scan).
// ---------------------------------------------------------------------------
void WCDatabaseExporterDev::clean_scans()
{
  constexpr float kZlo          = 0.05f, kZhi = 2.00f;
  constexpr float kRangeMin     = 0.30f;  // wheelchair body / user's legs
  constexpr float kRangeMax     = 2.50f;  // depth noise ~r²; scans are capped at 2.5 m anyway
  constexpr float kRorRadius    = 0.05f;  // 2D neighbour search radius (m)
  constexpr int   kRorMinNeigh  = 12;     // neighbours required (excluding self)
  constexpr int   kMinScanPts   = 50;     // drop scans left nearly empty after cleaning

  per_scan_base_.clear();
  scan_local_tf_.clear();

  // Scan scale correction (env WC_SCAN_XSCALE / WC_SCAN_YSCALE, default 1 = off).
  // EXPERIMENTAL — NOT RECOMMENDED.  Single-scan corner angles on 10-08-01
  // vary as -1.8°·sin(2α) (α = corner orientation in the scan), which looked
  // like a ~3.3% sideways-vs-forward scale error, and y *= 1/1.033 removes that
  // pattern.  But it makes multi-scan wall consistency WORSE (global wall
  // spread with GT poses 30.9 -> 34.3 mm at y 0.968), so the scale hypothesis
  // is rejected; the cause of the corner pattern is still open.  Kept only for
  // experiments.  Applied to x/y in base_link before cropping.
  float scan_xscale = 1.0f, scan_yscale = 1.0f;
  if (const char *e = std::getenv("WC_SCAN_XSCALE")) scan_xscale = (float)std::atof(e);
  if (const char *e = std::getenv("WC_SCAN_YSCALE")) scan_yscale = (float)std::atof(e);
  if (scan_xscale != 1.0f || scan_yscale != 1.0f)
    printf("[scan] scale correction x %.4f  y %.4f\n", scan_xscale, scan_yscale);
  const double cam_yaw = camera_yaw_rad();   // DEV: rotate scans with the tag detections

  // Top-of-view edge filter.  Points along the top border of the depth image
  // are unreliable; seen from a similar spot on several passes they repeat and
  // survive every persistence test (10-08-01: a false object in I3, all 37 hits
  // within 9.3 cm of the top edge at 2.2 m).  The edge is measured from this
  // recording (99.5th-pct height per 10 cm of range, every 4th scan), and
  // points within kTopEdgeMargin of it beyond kTopEdgeMinR are dropped.
  // Cost on the user's marked features: 0–7% of their hits.
  constexpr bool  kTopEdgeFilter = true;
  constexpr float kTopEdgeMargin = 0.10f;  // m below the edge
  constexpr float kTopEdgeMinR   = 1.8f;   // m; nearer the edge is inside the band anyway
  constexpr float kEdgeRBin = 0.10f, kEdgeZBin = 0.01f, kEdgeZ0 = -0.5f;
  constexpr int   kEdgeNR = 27, kEdgeNZ = 300;
  std::vector<float> top_edge(kEdgeNR, 1e9f);   // z of the top edge per range bin (1e9 = unknown → no filtering)
  if (kTopEdgeFilter) {
    std::vector<std::vector<uint32_t>> hist(kEdgeNR, std::vector<uint32_t>(kEdgeNZ, 0));
    int k = 0;
    for (auto iter = optimizedPoses.lower_bound(1); iter != optimizedPoses.end(); ++iter) {
      if (k++ % 4) continue;
      auto node_it = nodes.find(iter->first);
      if (node_it == nodes.end()) continue;
      rtabmap::LaserScan scan;
      node_it->second.sensorData().uncompressData(nullptr, nullptr, &scan);
      if (scan.isEmpty()) continue;
      auto pts = rtabmap::util3d::laserScanToPointCloud(scan, scan.localTransform());
      for (const auto &pt : pts->points) {
        if (!pcl::isFinite(pt)) continue;
        const float r = std::hypot(pt.x * scan_xscale, pt.y * scan_yscale);
        const int rb = (int)(r / kEdgeRBin), zb = (int)((pt.z - kEdgeZ0) / kEdgeZBin);
        if (rb < 0 || rb >= kEdgeNR || zb < 0 || zb >= kEdgeNZ) continue;
        ++hist[rb][zb];
      }
    }
    for (int rb = 0; rb < kEdgeNR; ++rb) {
      uint64_t tot = 0;
      for (uint32_t c : hist[rb]) tot += c;
      if (tot < 100) continue;
      uint64_t acc = 0;
      for (int zb = 0; zb < kEdgeNZ; ++zb) {
        acc += hist[rb][zb];
        if (acc >= 0.995 * tot) { top_edge[rb] = kEdgeZ0 + (zb + 1) * kEdgeZBin; break; }
      }
    }
  }
  auto near_top_edge = [&](float r, float z) {
    if (!kTopEdgeFilter || r <= kTopEdgeMinR) return false;
    const int rb = std::min(kEdgeNR - 1, (int)(r / kEdgeRBin));
    return top_edge[rb] < 1e8f && z > top_edge[rb] - kTopEdgeMargin;
  };
  size_t n_top_edge = 0;

  int n_scans = 0, n_dropped = 0;
  size_t n_raw = 0, n_cropped = 0, n_kept = 0;
  for (auto iter = optimizedPoses.lower_bound(1); iter != optimizedPoses.end(); ++iter) {
    auto node_it = nodes.find(iter->first);
    if (node_it == nodes.end()) continue;

    rtabmap::LaserScan scan;
    node_it->second.sensorData().uncompressData(nullptr, nullptr, &scan);
    if (scan.isEmpty()) continue;

    auto base_pts = rtabmap::util3d::laserScanToPointCloud(scan, scan.localTransform());
    if (scan_xscale != 1.0f || scan_yscale != 1.0f)
      for (auto &pt : base_pts->points) { pt.x *= scan_xscale; pt.y *= scan_yscale; }
    if (cam_yaw != 0.0) {                       // DEV: camera yaw correction (about base_link)
      const float c = (float)std::cos(cam_yaw), sn = (float)std::sin(cam_yaw);
      for (auto &pt : base_pts->points) { const float x = pt.x; pt.x = c * x - sn * pt.y; pt.y = sn * x + c * pt.y; }
    }
    n_raw += base_pts->size();

    // 1. Crop.  flat is the same cloud with z=0 for the 2D outlier filter.
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropped(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr flat(new pcl::PointCloud<pcl::PointXYZ>);
    cropped->reserve(base_pts->size());
    flat->reserve(base_pts->size());
    for (const auto &pt : base_pts->points) {
      if (!pcl::isFinite(pt)) continue;
      if (pt.z < kZlo || pt.z > kZhi) continue;
      float r = std::hypot(pt.x, pt.y);
      if (r < kRangeMin || r > kRangeMax) continue;
      if (near_top_edge(r, pt.z)) { ++n_top_edge; continue; }
      cropped->push_back(pt);
      flat->push_back(pcl::PointXYZ(pt.x, pt.y, 0.0f));
    }
    n_cropped += cropped->size();

    // 2. 2D radius outlier removal.
    pcl::Indices kept;
    if (!flat->empty()) {
      pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
      ror.setInputCloud(flat);
      ror.setRadiusSearch(kRorRadius);
      ror.setMinNeighborsInRadius(kRorMinNeigh);
      ror.filter(kept);
    }
    if ((int)kept.size() < kMinScanPts) { ++n_dropped; continue; }
    n_kept += kept.size();

    scan_local_tf_[iter->first] = scan.localTransform();
    auto &pts = per_scan_base_[iter->first];
    pts.reserve(kept.size());
    for (auto idx : kept) {
      const auto &p = cropped->points[idx];
      pts.emplace_back(p.x, p.y, p.z);
    }
    ++n_scans;
  }
  printf("[scan] %d scans cleaned (%d dropped < %d pts)\n", n_scans, n_dropped, kMinScanPts);
  add_metric("scan clean: scans=%d dropped=%d  pts raw=%zu crop=%zu kept=%zu  ROR removed %.1f%% (r=%.2f k=%d)",
             n_scans, n_dropped, n_raw, n_cropped, n_kept,
             n_cropped ? 100.0 * (n_cropped - n_kept) / n_cropped : 0.0,
             kRorRadius, kRorMinNeigh);
  add_metric("scan scale: x %.4f  y %.4f  (WC_SCAN_XSCALE / WC_SCAN_YSCALE)", scan_xscale, scan_yscale);
  {
    const int rb18 = (int)(1.85f / kEdgeRBin), rb22 = (int)(2.25f / kEdgeRBin);
    add_metric("scan top-edge filter: %s  dropped %zu points (%.2f%% of cropped) within %.2f m of the top of view beyond %.1f m  (edge z @1.85 m %.2f, @2.25 m %.2f)",
               kTopEdgeFilter ? "on" : "off", n_top_edge,
               n_cropped + n_top_edge ? 100.0 * n_top_edge / (n_cropped + n_top_edge) : 0.0,
               kTopEdgeMargin, kTopEdgeMinR, top_edge[rb18] < 1e8f ? top_edge[rb18] : 0.0f, top_edge[rb22] < 1e8f ? top_edge[rb22] : 0.0f);
  }
  printf("[scan] points: raw=%zu  after crop=%zu  after 2D ROR(r=%.2f,k=%d)=%zu  (%.1f%% removed by ROR)\n",
         n_raw, n_cropped, kRorRadius, kRorMinNeigh, n_kept,
         n_cropped ? 100.0 * (n_cropped - n_kept) / n_cropped : 0.0);
}

// ---------------------------------------------------------------------------
// refine_poses_with_local_map_icp
// ---------------------------------------------------------------------------
// The tag graph fixes the global shape, but tag pose noise (a few cm / deg at
// 1–2 m) jitters neighbouring poses and smears walls.  This adds scan-matching
// constraints between spatially close, path-distant nodes:
//
//   1. Query a node every kQueryStep m / kQueryYaw rad of travel.
//   2. Submap = cleaned scans of nodes within kSubmapRadius (XY) whose path
//      distance to the query is ≥ kMinPathSep (so it is a different view, not
//      the same frames), in the query node's current frame.  Merging several
//      nodes widens the narrow (~87°, ≤2.5 m) depth FOV — scan-to-scan ICP is
//      degenerate on it (29% of scans on 09-29-2.db are ~one straight wall).
//   3. Point-to-plane ICP, identity guess, 3DoF when Reg/Force3DoF.  Low
//      structural complexity (single wall) → rejected, not half-constrained.
//   4. Accepted (overlap ≥ kMinOverlap) → kLocalSpaceClosure link from the
//      nearest submap node to the query.  Its (x, y, yaw) information comes
//      from the matched wall normals, so directions the scan cannot see
//      (along a single wall) carry ~no weight; RTABMap's ICP covariance is
//      isotropic and made those links fight the tags.
//   5. Re-optimize tag graph + ICP links with Optimizer/Robust (Vertigo
//      switchable constraints) so bad registrations get switched off instead
//      of dragging poses.
// Repeated kRounds times; each round rebuilds the ICP links from the updated
// poses (the tag graph in `links` is the fixed base).
// ---------------------------------------------------------------------------
void WCDatabaseExporterDev::refine_poses_with_local_map_icp(
    const rtabmap::ParametersMap &parameters,
    std::map<int, rtabmap::Transform> &tag_poses)
{
  constexpr int    kRounds         = 0;
  constexpr float  kQueryStep      = 0.25f;   // m of travel between query nodes
  constexpr float  kQueryYaw       = 0.26f;   // rad (~15°) of turn between query nodes
  constexpr int    kMinQueryPts    = 500;     // skip near-empty query scans
  constexpr float  kSubmapRadius   = 2.0f;    // m, XY distance to candidate nodes
  constexpr float  kMinPathSep     = 1.0f;    // m of path between query and submap nodes
  constexpr int    kMinSubmapNodes = 3;
  constexpr int    kMaxSubmapNodes = 30;
  constexpr float  kVoxel          = 0.04f;   // ICP voxel for both clouds (m); 0.02 gave the same walls at 3× the time
  constexpr int    kNormalK        = 10;      // ICP point-to-plane normal neighbours (use 20 at 2 cm voxel)
  constexpr float  kOverlapDist    = 0.10f;   // m, query→submap NN distance counted as overlap
  constexpr float  kMinOverlap     = 0.30f;   // fraction of voxelized query points overlapping
  constexpr float  kNormalRadius   = 0.10f;   // m, submap neighbourhood for 2D wall normals
  constexpr double kVarLinMin      = 0.0004;  // (2 cm)² variance of the best-constrained xy direction
  constexpr double kVarAngMin      = 0.0003;  // (~1°)² floor on yaw variance

  if (per_scan_base_.empty()) {
    printf("[lmicp] No cleaned scans — skipping local-map ICP\n");
    return;
  }

  auto f3 = parameters.find(rtabmap::Parameters::kRegForce3DoF());
  std::string force3dof = f3 != parameters.end() ? f3->second : "true";

  // Voxelization and the overlap check are done here rather than inside
  // RegistrationIcp: its correspondence count is backend-dependent (with
  // libpointmatcher it rejected 68% of queries on 09-29-2.db whose measured
  // overlap was fine), so we measure overlap ourselves after registration.
  rtabmap::ParametersMap icp_params;
  icp_params[rtabmap::Parameters::kRegForce3DoF()]                        = force3dof;
  icp_params[rtabmap::Parameters::kIcpVoxelSize()]                        = "0";
  icp_params[rtabmap::Parameters::kIcpPointToPlane()]                     = "true";
  icp_params[rtabmap::Parameters::kIcpPointToPlaneK()]                    = std::to_string(kNormalK);
  icp_params[rtabmap::Parameters::kIcpPointToPlaneRadius()]               = "0";
  icp_params[rtabmap::Parameters::kIcpPointToPlaneMinComplexity()]        = "0.02";
  icp_params[rtabmap::Parameters::kIcpPointToPlaneLowComplexityStrategy()] = "0";  // reject
  icp_params[rtabmap::Parameters::kIcpMaxCorrespondenceDistance()]        = "0.10";
  icp_params[rtabmap::Parameters::kIcpReciprocalCorrespondences()]        = "false"; // dense submap vs sparse scan
  icp_params[rtabmap::Parameters::kIcpCorrespondenceRatio()]              = "0";     // checked below
  icp_params[rtabmap::Parameters::kIcpIterations()]                       = "30";
  icp_params[rtabmap::Parameters::kIcpMaxTranslation()]                   = "0.20";
  icp_params[rtabmap::Parameters::kIcpMaxRotation()]                      = "0.10";
  rtabmap::RegistrationIcp icp(icp_params);

  auto voxelize = [&](const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(kVoxel, kVoxel, kVoxel);
    vg.filter(*out);
    return out;
  };
  auto to_sensor_data = [](const pcl::PointCloud<pcl::PointXYZ> &cloud) {
    cv::Mat m(1, (int)cloud.size(), CV_32FC3);
    float *p = m.ptr<float>();
    for (const auto &pt : cloud.points) { *p++ = pt.x; *p++ = pt.y; *p++ = pt.z; }
    rtabmap::SensorData d;
    d.setLaserScan(rtabmap::LaserScan(m, (int)cloud.size(), 0.0f, rtabmap::LaserScan::kXYZ));
    return d;
  };

  printf("\n[lmicp] ── Local-map ICP (%d rounds, force3DoF=%s) ──────────────\n",
         kRounds, force3dof.c_str());
  add_metric("lmicp config: rounds=%d step=%.2fm/%.2frad submap r=%.1fm sep=%.1fm nodes=%d-%d voxel=%.2f normalK=%d corr>=%.2f var_min=%.4f/%.4f",
             kRounds, kQueryStep, kQueryYaw, kSubmapRadius, kMinPathSep,
             kMinSubmapNodes, kMaxSubmapNodes, kVoxel, kNormalK, kMinOverlap, kVarLinMin, kVarAngMin);

  const std::multimap<int, rtabmap::Link> tag_graph = links;  // as left by the tag GTSAM passes

  report_wall_thickness("rtabmap graph", rtabmap_graph_poses_);
  report_wall_thickness("tag GTSAM   ", optimizedPoses);

  for (int round = 1; round <= kRounds; ++round) {
    UTimer timer;

    // Path distance along node-ID order (≈ time).
    std::map<int, float> cum;
    {
      float d = 0.0f;
      const rtabmap::Transform *prev = nullptr;
      for (auto it = optimizedPoses.lower_bound(1); it != optimizedPoses.end(); ++it) {
        if (prev) d += std::hypot(it->second.x() - prev->x(), it->second.y() - prev->y());
        cum[it->first] = d;
        prev = &it->second;
      }
    }

    // Submap candidates: every node with a cleaned scan.
    struct Cand { int id; float x, y; };
    std::vector<Cand> cands;
    for (const auto &[nid, pts] : per_scan_base_) {
      auto pit = optimizedPoses.find(nid);
      if (pit == optimizedPoses.end()) continue;
      cands.push_back({nid, pit->second.x(), pit->second.y()});
    }

    // Query nodes: one per kQueryStep of travel or kQueryYaw of turn.
    std::vector<int> queries;
    {
      const rtabmap::Transform *last = nullptr;
      for (const auto &c : cands) {
        if ((int)per_scan_base_.at(c.id).size() < kMinQueryPts) continue;
        const rtabmap::Transform &p = optimizedPoses.at(c.id);
        if (last) {
          rtabmap::Transform d = last->inverse() * p;
          if (std::hypot(d.x(), d.y()) < kQueryStep && std::abs(d.theta()) < kQueryYaw) continue;
        }
        queries.push_back(c.id);
        last = &p;
      }
    }
    printf("[lmicp] round %d: %zu query nodes, %zu candidate scans\n",
           round, queries.size(), cands.size());
    fflush(stdout);

    std::multimap<int, rtabmap::Link> icp_links;
    std::map<std::string, int> reject_reasons;
    int n_no_submap = 0, n_low_corr = 0, n_ok = 0;
    double sum_dt = 0.0, sum_dyaw = 0.0, sum_strong = 0.0, sum_weak = 0.0;
    std::vector<float> overlaps;  // every converged registration, accepted or not
    std::vector<double> constraint_ratios;  // sqrt(λweak/λstrong) of accepted links

    for (size_t qi = 0; qi < queries.size(); ++qi) {
      if (qi > 0 && qi % 100 == 0) {
        printf("[lmicp]   %zu / %zu  (ok=%d no_submap=%d rejected=%zu)\n",
               qi, queries.size(), n_ok, n_no_submap, qi - n_ok - n_no_submap);
        fflush(stdout);
      }
      const int qid = queries[qi];
      const rtabmap::Transform &Pq = optimizedPoses.at(qid);
      const float qc = cum.at(qid);

      std::vector<std::pair<float, int>> near;  // (XY distance, node id)
      for (const auto &c : cands) {
        if (std::abs(cum.at(c.id) - qc) < kMinPathSep) continue;
        float d = std::hypot(c.x - Pq.x(), c.y - Pq.y());
        if (d <= kSubmapRadius) near.push_back({d, c.id});
      }
      if ((int)near.size() < kMinSubmapNodes) { ++n_no_submap; continue; }
      const int ref_id = std::min_element(near.begin(), near.end())->second;

      if ((int)near.size() > kMaxSubmapNodes) {
        // Even spread over node ID so every pass through this area
        // contributes, rather than the nearest cluster from a single pass.
        std::sort(near.begin(), near.end(),
                  [](const auto &a, const auto &b) { return a.second < b.second; });
        std::vector<std::pair<float, int>> picked;
        for (int k = 0; k < kMaxSubmapNodes; ++k)
          picked.push_back(near[(size_t)k * near.size() / kMaxSubmapNodes]);
        near.swap(picked);
      }

      // Submap in the query node's current frame.
      pcl::PointCloud<pcl::PointXYZ>::Ptr sub(new pcl::PointCloud<pcl::PointXYZ>);
      const rtabmap::Transform Pq_inv = Pq.inverse();
      for (const auto &[d, nid] : near) {
        Eigen::Affine3f T = (Pq_inv * optimizedPoses.at(nid)).toEigen3f();
        for (const auto &p : per_scan_base_.at(nid)) {
          Eigen::Vector3f w = T * Eigen::Vector3f(p.x, p.y, p.z);
          sub->push_back(pcl::PointXYZ(w.x(), w.y(), w.z()));
        }
      }
      pcl::PointCloud<pcl::PointXYZ>::Ptr qcloud(new pcl::PointCloud<pcl::PointXYZ>);
      for (const auto &p : per_scan_base_.at(qid)) qcloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));

      auto sub_v = voxelize(sub);
      auto q_v   = voxelize(qcloud);
      const int n_q = (int)q_v->size();
      if (n_q == 0 || sub_v->empty()) { ++n_no_submap; continue; }

      // from = submap (query frame), to = query scan → T is the corrected pose
      // of the query in its own old frame: Pq' = Pq · T.
      rtabmap::RegistrationInfo info;
      rtabmap::Transform T = icp.computeTransformation(
          to_sensor_data(*sub_v), to_sensor_data(*q_v),
          rtabmap::Transform::getIdentity(), &info);
      if (T.isNull()) {
        std::string why = info.rejectedMsg.substr(0, info.rejectedMsg.find_first_of("0123456789("));
        reject_reasons[why.empty() ? "unknown" : why.substr(0, 70)]++;
        continue;
      }

      // After registration, for each query voxel:
      //  - overlap: nearest submap voxel within kOverlapDist;
      //  - constraint: the 2D normal n of the submap surface there (PCA of XY
      //    neighbours) gives one point-to-plane row J = [n_x, n_y, n × r],
      //    r = lever arm from the query origin.  H = Σ JᵀJ is the (x, y, yaw)
      //    information in the query frame — strong across walls, ~0 along a
      //    single wall, so ICP sliding along a corridor carries no weight.
      const Eigen::Affine3f Tq = T.toEigen3f();
      pcl::KdTreeFLANN<pcl::PointXYZ> kd;
      kd.setInputCloud(sub_v);
      pcl::Indices nn(1), nbr;
      std::vector<float> nd2(1), nbr_d2;
      int n_in = 0;
      Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
      for (const auto &p : q_v->points) {
        const Eigen::Vector3f w = Tq * p.getVector3fMap();
        const pcl::PointXYZ wp(w.x(), w.y(), w.z());
        if (kd.nearestKSearch(wp, 1, nn, nd2) <= 0 || nd2[0] > kOverlapDist * kOverlapDist)
          continue;
        ++n_in;

        if (kd.radiusSearch(sub_v->points[nn[0]], kNormalRadius, nbr, nbr_d2) < 5) continue;
        double mx = 0, my = 0;
        for (auto j : nbr) { mx += sub_v->points[j].x; my += sub_v->points[j].y; }
        mx /= nbr.size(); my /= nbr.size();
        double a = 0, b = 0, c = 0;
        for (auto j : nbr) {
          double dx = sub_v->points[j].x - mx, dy = sub_v->points[j].y - my;
          a += dx * dx; b += dx * dy; c += dy * dy;
        }
        const double h    = std::sqrt(0.25 * (a - c) * (a - c) + b * b);
        const double lmin = 0.5 * (a + c) - h, lmax = 0.5 * (a + c) + h;
        if (lmax <= 0.0 || lmin > 0.25 * lmax) continue;  // not line-like (corner / clutter)
        Eigen::Vector2d n = std::abs(b) > 1e-12 ? Eigen::Vector2d(b, lmin - a)
                          : (a < c ? Eigen::Vector2d(1, 0) : Eigen::Vector2d(0, 1));
        n.normalize();
        const double rx = w.x() - T.x(), ry = w.y() - T.y();
        const Eigen::Vector3d J(n.x(), n.y(), n.x() * -ry + n.y() * rx);
        H += J * J.transpose();
      }
      const float overlap = (float)n_in / n_q;
      overlaps.push_back(overlap);
      if (overlap < kMinOverlap) { ++n_low_corr; continue; }

      // Scale so the best-constrained xy direction gets variance kVarLinMin
      // (per-point rows are correlated, so raw H is wildly overconfident);
      // weaker directions keep their ratio.  Yaw is capped at 1/kVarAngMin via
      // a symmetric D·H·D scaling so the matrix stays PSD.
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(H.topLeftCorner<2, 2>());
      const double l_weak = std::max(0.0, es.eigenvalues()(0));
      const double l_strong = es.eigenvalues()(1);
      if (l_strong <= 0.0) { ++n_low_corr; continue; }
      Eigen::Matrix3d I3 = ((1.0 / kVarLinMin) / l_strong) * H;
      if (I3(2, 2) > 1.0 / kVarAngMin) {
        const double k = std::sqrt((1.0 / kVarAngMin) / I3(2, 2));
        I3.row(2) *= k;
        I3.col(2) *= k;
      }

      const Eigen::Vector2d t_corr(T.x(), T.y());
      sum_dt   += t_corr.norm();
      sum_dyaw += std::abs(T.theta());
      sum_strong += std::abs(t_corr.dot(es.eigenvectors().col(1)));
      sum_weak   += std::abs(t_corr.dot(es.eigenvectors().col(0)));
      constraint_ratios.push_back(std::sqrt(l_weak / l_strong));

      // Link ref → query, measured relative pose Pref⁻¹ · Pq · T.  GTSAM's
      // Pose2 between-factor error is in the "to" (query) frame, matching H.
      rtabmap::Transform rel = optimizedPoses.at(ref_id).inverse() * Pq * T;
      cv::Mat inf = cv::Mat::zeros(6, 6, CV_64FC1);
      const int idx3[3] = {0, 1, 5};  // x, y, yaw
      for (int r = 0; r < 3; ++r)
        for (int cc = 0; cc < 3; ++cc)
          inf.at<double>(idx3[r], idx3[cc]) = I3(r, cc);
      inf.at<double>(2, 2) = 1.0 / kVarLinMin;  // z, roll, pitch: unused in 2D
      inf.at<double>(3, 3) = inf.at<double>(4, 4) = 1.0 / kVarAngMin;
      icp_links.insert({ref_id,
          rtabmap::Link(ref_id, qid, rtabmap::Link::kLocalSpaceClosure, rel, inf)});
      ++n_ok;
    }

    int n_rejected = 0;
    for (const auto &[why, n] : reject_reasons) n_rejected += n;
    const double t_icp = timer.ticks();
    printf("[lmicp] round %d: %d links accepted / %zu queries  (no_submap=%d icp_rejected=%d low_overlap=%d)  %.1f s\n",
           round, n_ok, queries.size(), n_no_submap, n_rejected, n_low_corr, t_icp);
    for (const auto &[why, n] : reject_reasons)
      printf("[lmicp]   rejected %4d × %s\n", n, why.c_str());
    if (n_ok > 0)
      printf("[lmicp]   mean ICP correction: %.3f m, %.2f°\n",
             sum_dt / n_ok, sum_dyaw / n_ok * 180.0 / M_PI);
    add_metric("lmicp round %d: queries=%zu accepted=%d no_submap=%d icp_rejected=%d low_overlap=%d  mean corr %.1f mm / %.2f deg  (%.0f s)",
               round, queries.size(), n_ok, n_no_submap, n_rejected, n_low_corr,
               n_ok ? 1000.0 * sum_dt / n_ok : 0.0,
               n_ok ? sum_dyaw / n_ok * 180.0 / M_PI : 0.0, t_icp);
    if (!overlaps.empty()) {
      std::sort(overlaps.begin(), overlaps.end());
      auto pct = [&](double q) { return overlaps[(size_t)(q * (overlaps.size() - 1))]; };
      add_metric("  round %d overlap@%.2fm (converged n=%zu): p10 %.2f  p25 %.2f  median %.2f  p75 %.2f",
                 round, kOverlapDist, overlaps.size(), pct(0.10), pct(0.25), pct(0.50), pct(0.75));
    }
    if (n_ok > 0) {
      std::sort(constraint_ratios.begin(), constraint_ratios.end());
      auto cr = [&](double q) { return constraint_ratios[(size_t)(q * (constraint_ratios.size() - 1))]; };
      add_metric("  round %d correction split: strong-axis %.1f mm  weak-axis %.1f mm   constraint ratio sqrt(weak/strong): p10 %.2f  median %.2f  p90 %.2f",
                 round, 1000.0 * sum_strong / n_ok, 1000.0 * sum_weak / n_ok,
                 cr(0.10), cr(0.50), cr(0.90));
    }
    for (const auto &[why, n] : reject_reasons)
      add_metric("  round %d rejected %d x %s", round, n, why.c_str());
    if (n_ok == 0) break;

    // Re-optimize: tag graph + this round's ICP links.
    std::multimap<int, rtabmap::Link> round_links = tag_graph;
    round_links.insert(icp_links.begin(), icp_links.end());

    std::set<int> referenced;
    for (const auto &kv : round_links) {
      referenced.insert(kv.second.from());
      referenced.insert(kv.second.to());
    }
    std::map<int, rtabmap::Transform> initial;
    for (const auto &[nid, pose] : optimizedPoses)
      if (nid > 0 || referenced.count(nid)) initial[nid] = pose;
    for (const auto &[nid, pose] : tag_poses)
      if (referenced.count(nid)) initial[nid] = pose;

    rtabmap::ParametersMap opt_params = parameters;
    opt_params[rtabmap::Parameters::kOptimizerRobust()] = "true";
    rtabmap::Optimizer *opt = rtabmap::Optimizer::create(opt_params);
    cv::Mat out_cov;
    std::map<int, rtabmap::Transform> new_poses =
        opt->optimize(initial.lower_bound(1)->first, initial, round_links, out_cov);
    delete opt;

    if (new_poses.empty()) {
      std::cerr << "[lmicp] round " << round << " optimization failed, keeping previous poses\n";
      add_metric("lmicp round %d re-opt: FAILED (kept previous poses)", round);
      break;
    }

    double sum_t = 0.0, sum_r = 0.0, max_t = 0.0;
    int n_moved = 0;
    for (const auto &[nid, pose] : new_poses) {
      if (nid > 0) {
        auto pit = optimizedPoses.find(nid);
        if (pit != optimizedPoses.end()) {
          rtabmap::Transform d = pit->second.inverse() * pose;
          double dt = std::hypot(d.x(), d.y());
          sum_t += dt; sum_r += std::abs(d.theta()); max_t = std::max(max_t, dt);
          ++n_moved;
        }
        optimizedPoses[nid] = pose;
      } else {
        tag_poses[nid] = pose;
      }
    }
    links = round_links;
    if (n_moved > 0)
      printf("[lmicp] round %d re-opt: %d robot nodes, mean Δ=%.4f m / %.2f°, max Δ=%.3f m\n",
             round, n_moved, sum_t / n_moved, sum_r / n_moved * 180.0 / M_PI, max_t);

    add_metric("lmicp round %d re-opt: mean dmove %.1f mm / %.2f deg  max %.1f mm",
               round,
               n_moved ? 1000.0 * sum_t / n_moved : 0.0,
               n_moved ? sum_r / n_moved * 180.0 / M_PI : 0.0,
               1000.0 * max_t);
    report_wall_thickness("icp round " + std::to_string(round), optimizedPoses);
  }
  printf("[lmicp] ────────────────────────────────────────────────────────\n\n");
}

// ---------------------------------------------------------------------------
// assembleSceneFromOptimizedPoses
// ---------------------------------------------------------------------------
// Moves the cleaned base-frame scans into world frame with the final poses.
// per_scan_xyz_    : cleaned endpoints in world frame
// per_scan_origin_ : scan origin XY in world frame
// ---------------------------------------------------------------------------
void WCDatabaseExporterDev::assembleSceneFromOptimizedPoses()
{
  if (per_scan_base_.empty()) clean_scans();

  int n_scans = 0;
  size_t n_pts = 0;
  for (auto &[nid, base] : per_scan_base_) {
    auto pit = optimizedPoses.find(nid);
    if (pit == optimizedPoses.end()) continue;

    rtabmap::Transform sensor_world = pit->second * scan_local_tf_.at(nid);
    per_scan_origin_[nid] = {sensor_world.x(), sensor_world.y()};

    Eigen::Affine3f T = pit->second.toEigen3f();
    auto &xyz = per_scan_xyz_[nid];
    xyz.reserve(base.size());
    for (const auto &p : base) {
      Eigen::Vector3f w = T * Eigen::Vector3f(p.x, p.y, p.z);
      xyz.emplace_back(w.x(), w.y(), w.z());
    }
    n_pts += xyz.size();
    std::vector<cv::Point3f>().swap(base);  // free base-frame copy as we go
    ++n_scans;
  }
  per_scan_base_.clear();
  printf("[scan] %d scans assembled in world frame (%zu points)\n", n_scans, n_pts);
}

void WCDatabaseExporterDev::projectAndColorizePointCloud() {}
void WCDatabaseExporterDev::assemble_colored_point_cloud() {}

// ---------------------------------------------------------------------------
// finalize_and_return_result
// ---------------------------------------------------------------------------
// 1. Build log-odds raytrace occupancy grid (occupied / free / unknown)
// 2. Morphological close to bridge small gaps
// 3. Connected component filter to remove noise blobs
// 4. BFS flood-fill to mark traversable free space
// 5. Save grid PGM + YAML + tag overlay PNG
// ---------------------------------------------------------------------------
void WCDatabaseExporterDev::finalize_and_return_result(Result &result)
{
  if (per_scan_xyz_.empty()) {
    std::cerr << "[finalize] No scan data assembled — cannot build grid\n";
    return;
  }

  std::string path = output_path_;
  // Grid file name (env WC_GRID_NAME, default = the run timestamp): names
  // grid/<name>.pgm, .yaml and _tags.png; the run folder keeps its timestamp.
  std::string grid_name = timestamp_;
  if (const char *e = std::getenv("WC_GRID_NAME")) {
    std::string n = e;
    for (char &ch : n) if (ch == '/' || ch == ' ') ch = '_';
    if (!n.empty()) grid_name = n;
  }
  add_metric("grid name: %s (WC_GRID_NAME)", grid_name.c_str());

  constexpr float kRes = 0.025f;
  constexpr float kPad = 1.0f;

  // ── 1. Align walls with the grid axes BEFORE rasterizing ─────────────────
  // The old step 7 rotated the finished raster (PCA angle, INTER_NEAREST):
  // walls at an arbitrary angle were staircased once when rasterized and
  // resampled again by the rotation -> jagged, uneven walls.  Instead, find
  // the dominant wall direction from the wall-band points and rasterize once
  // in a frame where walls run along cell rows/columns.  Points stay in world
  // frame (PCD unchanged); only grid indexing and the overlay are rotated.
  // Env WC_GRID_ALIGN: 0 (default) = no rotation, the grid stays in the map
  // frame (the saved YAML origin has no yaw, so a rotated grid does not line
  // up with the live map frame); 1 = align to the dominant wall direction
  // before rasterizing; pca = old post-raster PCA rotation (step 7).
  int kGridAlign = 0;
  if (const char *e = std::getenv("WC_GRID_ALIGN"))
    kGridAlign = std::string(e) == "pca" ? 2 : (std::atoi(e) != 0 ? 1 : 0);
  const bool kAlignBeforeRaster = kGridAlign == 1;
  double align_rad = 0.0;
  if (kGridAlign == 0)
    add_metric("grid alignment: rotated 0.00 deg (off: grid in the map frame; WC_GRID_ALIGN=1 aligns to walls, WC_GRID_ALIGN=pca old PCA rotation)");
  if (kAlignBeforeRaster) {
    // Coarse 5 cm wall-band hit image; gradient directions of a wall image
    // cluster at the wall normals, which repeat every 90 deg in a rectilinear
    // building -> circular mean of 4*phi gives the dominant direction.
    float bx0 = 1e9f, bx1 = -1e9f, by0 = 1e9f, by1 = -1e9f;
    for (const auto &[nid, pts] : per_scan_xyz_)
      for (size_t k = 0; k < pts.size(); k += 4) {
        const auto &p = pts[k];
        if (p.z < 0.10f || p.z > 1.00f) continue;
        bx0 = std::min(bx0, p.x); bx1 = std::max(bx1, p.x);
        by0 = std::min(by0, p.y); by1 = std::max(by1, p.y);
      }
    constexpr float kA = 0.05f;
    const int aw = (int)((bx1 - bx0) / kA) + 3, ah = (int)((by1 - by0) / kA) + 3;
    cv::Mat cnt(ah, aw, CV_32F, cv::Scalar(0));
    for (const auto &[nid, pts] : per_scan_xyz_)
      for (size_t k = 0; k < pts.size(); k += 4) {
        const auto &p = pts[k];
        if (p.z < 0.10f || p.z > 1.00f) continue;
        cnt.at<float>((int)((p.y - by0) / kA) + 1, (int)((p.x - bx0) / kA) + 1) += 1.0f;
      }
    cv::Mat img, gx, gy;
    cv::min(cnt, cv::Scalar(20.0), img);  // cap so dense spots don't dominate
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0);
    cv::Sobel(img, gx, CV_32F, 1, 0, 3);
    cv::Sobel(img, gy, CV_32F, 0, 1, 3);
    double s4 = 0.0, c4 = 0.0;
    for (int r = 0; r < ah; ++r)
      for (int c = 0; c < aw; ++c) {
        const double a = gx.at<float>(r, c), b = gy.at<float>(r, c);
        const double m = std::hypot(a, b);
        if (m < 1e-3) continue;
        const double phi = std::atan2(b, a);
        s4 += m * std::sin(4.0 * phi);
        c4 += m * std::cos(4.0 * phi);
      }
    align_rad = -0.25 * std::atan2(s4, c4);  // rotate wall normals onto the axes
    // Landscape: if the aligned map is taller than wide, add a quarter turn
    // (exact for a coordinate rotation, unlike rotating a raster).
    {
      const double ca = std::cos(align_rad), sa = std::sin(align_rad);
      float ux0 = 1e9f, ux1 = -1e9f, uy0 = 1e9f, uy1 = -1e9f;
      for (const auto &[nid, pts] : per_scan_xyz_)
        for (size_t k = 0; k < pts.size(); k += 16) {
          const auto &p = pts[k];
          const float u = (float)(ca * p.x - sa * p.y), v = (float)(sa * p.x + ca * p.y);
          ux0 = std::min(ux0, u); ux1 = std::max(ux1, u); uy0 = std::min(uy0, v); uy1 = std::max(uy1, v);
        }
      if (uy1 - uy0 > ux1 - ux0) align_rad -= M_PI / 2.0;
    }
    printf("[grid] wall alignment: rotating %.2f deg before rasterizing\n", align_rad * 180.0 / M_PI);
    add_metric("grid alignment: rotated %.2f deg onto grid axes before rasterizing (dominant wall direction)",
               align_rad * 180.0 / M_PI);
  }
  const float rot_c = (float)std::cos(align_rad), rot_s = (float)std::sin(align_rad);
  auto rot = [&](float x, float y) { return cv::Point2f(rot_c * x - rot_s * y, rot_s * x + rot_c * y); };

  float xmin=1e9f, xmax=-1e9f, ymin=1e9f, ymax=-1e9f;
  for (const auto &[nid, pts] : per_scan_xyz_)
    for (const auto &p : pts) {
      const cv::Point2f q = rot(p.x, p.y);
      xmin=std::min(xmin,q.x); xmax=std::max(xmax,q.x);
      ymin=std::min(ymin,q.y); ymax=std::max(ymax,q.y);
    }
  for (const auto &[nid, o] : per_scan_origin_) {
    const cv::Point2f q = rot(o.x, o.y);
    xmin=std::min(xmin,q.x); xmax=std::max(xmax,q.x);
    ymin=std::min(ymin,q.y); ymax=std::max(ymax,q.y);
  }
  xmin -= kPad; ymin -= kPad; xmax += kPad; ymax += kPad;

  int gw = (int)std::ceil((xmax - xmin) / kRes) + 1;
  int gh = (int)std::ceil((ymax - ymin) / kRes) + 1;

  std::vector<int8_t> grid(gw * gh, -1);

  // World (x, y) -> cell, in the aligned frame.
  auto to_cell = [&](float x, float y, int &cx_, int &cy_) {
    const cv::Point2f q = rot(x, y);
    cx_ = (int)((q.x - xmin) / kRes);
    cy_ = (int)((q.y - ymin) / kRes);
  };

  // Grid mode: 1 = direct projection (every cleaned point marks its cell),
  //            0 = range-weighted log-odds raytrace + morph / CC / flood fill.
#define WC_GRID_DIRECT_PROJECTION 0
#if WC_GRID_DIRECT_PROJECTION
  // ── 2. Direct projection ("smash to ground") ──────────────────────────
  // Every cleaned scan point (already cropped to z in [0.05, 2.0] and 2D-ROR
  // filtered in clean_scans) marks its XY cell occupied.  No free space, no
  // morphology, no blob filter, no flood fill.
  {
    size_t n_proj = 0;
    for (const auto &[nid, pts] : per_scan_xyz_)
      for (const auto &p : pts) {
        int cx, cy;
        to_cell(p.x, p.y, cx, cy);
        if (cx < 0 || cx >= gw || cy < 0 || cy >= gh) continue;
        grid[cy * gw + cx] = 100;
        ++n_proj;
      }
    printf("[grid] %d × %d cells  (%.1f × %.1f m)  direct projection of %zu cleaned points\n",
           gw, gh, gw * kRes, gh * kRes, n_proj);
    add_metric("grid mode: direct projection (raytrace/morph/CC/flood-fill disabled)");
  }

#else  // ── range-weighted raytrace + post-filtering pipeline ──────────────
  // ── 2. Log-odds raytrace ──────────────────────────────────────────────
  // Each ray from sensor origin to scan endpoint updates a log-odds grid:
  //   free cells along the ray   → subtract kLogFreeMiss  (evidence of free space)
  //   wall-band endpoint hit     → add    kLogOccHit · w(r) (evidence of occupancy)
  // Cells must accumulate consistent evidence across many scans before being
  // classified — this suppresses noise and produces clean, 1-cell-wide walls.
  //
  // Range weighting: stereo depth noise grows steeply with range (09-29-2.db
  // wall spread 8.6 mm from < 1.3 m vs 24.4 mm from > 1.9 m), so a hit's vote
  // is w(r) = min(1, (kHitRangeRef / r)^2): full at ≤ 1 m, ~1/6 at 2.5 m.
  // Near views then set where a wall is; stray far hits need several agreeing
  // votes before they paint a cell.  Free-space updates are unweighted.
  constexpr float kWallZlo    = 0.10f, kWallZhi = 1.00f;
  constexpr float kLogOccHit  =  2.00f;  // occupied evidence per full-weight hit
  constexpr float kLogFreeMiss=  0.15f;  // 13:1 ratio — single-hit cells survive more free rays at 2.5 cm resolution
  constexpr float kLogClamp   =  8.0f;   // prevent saturation
  constexpr float kHitRangeRef=  1.0f;   // m, range at/below which a hit has full weight
  // Height-aware free space: occupancy is kept in kLayers height layers of
  // kLayerH over the wall band.  Each free cell along a ray is cleared only in
  // the layer the ray actually passes through there (height interpolated from
  // the camera at kSensorZ to the endpoint), and hits go to the layer of their
  // own height.  A ray over a tabletop or under a chair seat no longer erases
  // it; rays outside the band (e.g. to the floor) clear nothing.  A cell is
  // occupied if any layer is, free if some layer is free and none occupied.
  // kHeightAwareFree = false → one layer, rays at any height clear the whole
  // 2D path (previous behaviour, identical results).
  // DISABLED: on 10-07-03 / 10-08-01 it grew occupied area +77% — a cell is
  // occupied if ANY of 9 layers is, and each layer is cleared only by the few
  // rays at that height, so wall scatter survives.  Replaced by the height
  // veto below, which keeps the 2D grid (same wall trimming) and only stops
  // rays that pass clearly over/under a cell's solidly observed hits.
  constexpr bool  kHeightAwareFree = false;
  constexpr float kSensorZ = 0.12f;   // m in base_link; tf_static camera links z 0.107–0.122
  constexpr float kLayerH  = 0.10f;
  constexpr int   kLayers  = kHeightAwareFree ? (int)((kWallZhi - kWallZlo) / kLayerH + 0.5f) : 1;
  // Height veto (furniture protection): a free update is skipped when the
  // cell has >= kVetoMinW weighted hits and the ray passes more than
  // kVetoMargin above the highest or below the lowest hit in that cell — the
  // ray went over a tabletop / under a seat in 3D.  Walls (hits at all
  // heights) and sparse speckles (little evidence) are cleared as before.
  constexpr bool  kHeightVeto  = true;
  constexpr float kVetoMinW    = 2.0f;   // ≈ 2 full-weight hits (≈ 12 hits from 2.5 m)
  constexpr float kVetoMargin  = 0.05f;  // m
  // The veto's height range is the 10th–90th percentile of the cell's
  // wall-band hit heights (2 cm histogram), not min–max.  A ledge at ~0.75 m
  // jutting 12–15 cm out of most outer walls (10-12-01, user-confirmed) got a
  // few stray low points from the wall's foot in its cells; min–max then
  // spanned 0.05–0.80 m, rays under the ledge were never vetoed and the
  // ledge footprint came out as a hollow free strip.  Percentiles ignore those
  // strays (Python replica analysis/grid_sim.py, 10-12-01: strip under the C2–D2
  // ledge filled, +534/-81 occupied cells, rest of the map unchanged).
  constexpr bool  kVetoPercentile = true;
  constexpr float kVetoLoPct = 0.10f, kVetoHiPct = 0.90f;
  constexpr float kVetoZBin  = 0.02f;
  constexpr int   kVetoNZ    = (int)((kWallZhi - kWallZlo) / kVetoZBin + 0.5f);   // 45 bins
  // Close-range evidence wins (furniture keep): a cell hit >= kKeepMinHits
  // times from < kNearR by >= kKeepMinScans different scans stays occupied
  // whatever the free rays say.  Near points are precise (~4 mm at < 1.3 m),
  // so they sit on real surfaces; wall scatter comes mostly from far points
  // and is still trimmed.  The veto alone missed furniture seen mainly from
  // 1.5–2.5 m (low hit weight) and objects with gaps (chairs, table legs).
  // Keeps walls the chair drove along close to (e.g. the long E3–I3 wall on
  // 10-08-01: 4 cells with it, 1 cell with gaps without).  It does not catch
  // objects seen only from 1.2–2.2 m — the height-span keep below does.
  constexpr bool  kNearKeep     = true;
  constexpr float kNearR        = 1.3f;  // m
  constexpr int   kKeepMinHits  = 8;
  constexpr int   kKeepMinScans = 3;
  // Height-span keep (objects at one height): a cell with >= kSpanMinHits
  // wall-band hits from >= kSpanMinScans scans, all within kSpanMaxDz of
  // height, stays occupied — tabletops, shelves, low boxes are thin in z and
  // get cleared by rays skimming just over/under them.  Wall scatter has hits
  // spread over the whole band (median 0.48 m) and is still trimmed.
  // Chosen on 10-08-01 against user-marked features (median span 2–10 cm vs
  // 48 cm for unwanted fringe); these cells are also exempt from the CC filter.
  constexpr bool  kSpanKeep     = true;
  constexpr int   kSpanMinHits  = 15;
  constexpr int   kSpanMinScans = 5;
  constexpr float kSpanMaxDz    = 0.12f;  // m
  // ...and seen over >= kSpanMinSeconds between first and last hit: furniture
  // is seen again on later passes, a person walking by is not.  On 10-08-01 the
  // user's features span a median 231 s; the shin-height dashes in I5/J5 (a
  // passer-by) span 2 s.  10 s keeps 88% of feature cells, drops 95% of those.
  constexpr double kSpanMinSeconds = 10.0;
  std::vector<uint8_t> feature_keep((size_t)gw * gh, 0);
  std::vector<float> cell_zlo, cell_hw;   // per-cell hit-height p10 and hit weight (ledge detection, step 7)
  std::vector<float> min_hit_r(gw * gh, 1e9f);  // nearest range that hit each cell (for metrics)
  double sum_w = 0.0;
  size_t n_hits = 0, n_free_upd = 0, n_free_protected = 0, n_free_outside = 0, n_free_vetoed = 0;
  {
    const size_t N = (size_t)gw * gh;
    // Pass 1: per-cell hit weight and hit height range (veto), and close-range
    // hit / scan counts (near keep).
    std::vector<float> hz_min, hz_max, hw;
    std::vector<float> veto_lo, veto_hi;   // height range the veto protects (percentiles or min/max)
    std::vector<uint32_t> zhist;           // per-cell wall-band height histogram (veto percentiles)
    std::vector<uint16_t> near_hits, near_scans, all_hits, all_scans;
    std::vector<uint32_t> near_last, all_last;
    std::vector<double> t_first, t_last;   // first / last time each cell was hit (s)
    if (kHeightVeto || kNearKeep || kSpanKeep) {
      hz_min.assign(N, 1e9f); hz_max.assign(N, -1e9f); hw.assign(N, 0.0f);
      near_hits.assign(N, 0); near_scans.assign(N, 0); near_last.assign(N, 0);
      all_hits.assign(N, 0); all_scans.assign(N, 0); all_last.assign(N, 0);
      t_first.assign(N, 1e18); t_last.assign(N, -1e18);
      if (kHeightVeto && kVetoPercentile) zhist.assign(N * kVetoNZ, 0);
      uint32_t sno = 0;
      for (const auto &[nid, pts] : per_scan_xyz_) {
        auto oit = per_scan_origin_.find(nid);
        if (oit == per_scan_origin_.end()) continue;
        ++sno;
        auto nit = nodes.find(nid);
        const double stamp = nit != nodes.end() ? nit->second.getStamp() : 0.0;
        for (const auto &p : pts) {
          if (p.z < kWallZlo || p.z > kWallZhi) continue;
          int hcx, hcy;
          to_cell(p.x, p.y, hcx, hcy);
          if (hcx < 0 || hcx >= gw || hcy < 0 || hcy >= gh) continue;
          const float r = std::hypot(p.x - oit->second.x, p.y - oit->second.y);
          const size_t i = (size_t)hcy * gw + hcx;
          hw[i] += r > kHitRangeRef ? (kHitRangeRef / r) * (kHitRangeRef / r) : 1.0f;
          hz_min[i] = std::min(hz_min[i], p.z);
          hz_max[i] = std::max(hz_max[i], p.z);
          if (!zhist.empty())
            ++zhist[i * kVetoNZ + std::min(kVetoNZ - 1, (int)((p.z - kWallZlo) / kVetoZBin))];
          if (all_hits[i] < 65535) ++all_hits[i];
          t_first[i] = std::min(t_first[i], stamp);
          t_last[i]  = std::max(t_last[i], stamp);
          if (all_last[i] != sno) { all_last[i] = sno; if (all_scans[i] < 65535) ++all_scans[i]; }
          if (r < kNearR) {
            if (near_hits[i] < 65535) ++near_hits[i];
            if (near_last[i] != sno) { near_last[i] = sno; if (near_scans[i] < 65535) ++near_scans[i]; }
          }
        }
      }
    }
    if (kHeightVeto) {
      if (kVetoPercentile) {
        veto_lo.assign(N, 1e9f); veto_hi.assign(N, -1e9f);
        for (size_t i = 0; i < N; ++i) {
          const uint32_t *h = &zhist[i * kVetoNZ];
          uint64_t tot = 0;
          for (int b = 0; b < kVetoNZ; ++b) tot += h[b];
          if (tot == 0) continue;
          uint64_t acc = 0;
          bool have_lo = false;
          for (int b = 0; b < kVetoNZ; ++b) {
            acc += h[b];
            if (!have_lo && acc >= kVetoLoPct * tot) { veto_lo[i] = kWallZlo + b * kVetoZBin; have_lo = true; }
            if (acc >= kVetoHiPct * tot) { veto_hi[i] = kWallZlo + (b + 1) * kVetoZBin; break; }
          }
        }
        std::vector<uint32_t>().swap(zhist);
      } else {
        veto_lo = hz_min; veto_hi = hz_max;
      }
      cell_zlo = veto_lo;
    }
    cell_hw = hw;
    std::vector<float> lo(N * kLayers, 0.0f);  // log-odds per cell and layer, init = unknown
    // Own-scan hit protection (as in OctoMap): a scan's rays never clear a
    // cell (layer) the same scan hit.  Otherwise a scan's noisy points behind
    // a wall send rays through that wall's cells and erode it — worst for far,
    // weakly-weighted walls.  Rays from other scans can still clear it.
    std::vector<uint32_t> hit_stamp(N * kLayers, 0);
    uint32_t scan_no = 0;

    // Layer of a height; -1 = outside the wall band (height-aware mode only).
    auto layer_of = [&](float z) -> int {
      if (!kHeightAwareFree) return 0;
      if (z < kWallZlo || z >= kWallZhi) return -1;
      return std::min(kLayers - 1, (int)((z - kWallZlo) / kLayerH));
    };

    // f(x, y, step, n_steps): step 0 = ray origin, step n_steps = endpoint.
    auto bresenham = [&](int x0, int y0, int x1, int y1, auto &&f) {
      int dx = std::abs(x1-x0), dy = std::abs(y1-y0);
      int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
      int err = dx - dy;
      const int n = std::max(dx, dy);
      for (int i = 0;; ++i) {
        f(x0, y0, i, n);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 <  dx) { err += dx; y0 += sy; }
      }
    };

    for (const auto &[nid, pts] : per_scan_xyz_) {
      auto oit = per_scan_origin_.find(nid);
      if (oit == per_scan_origin_.end()) continue;
      int ocx, ocy;
      to_cell(oit->second.x, oit->second.y, ocx, ocy);
      if (ocx < 0 || ocx >= gw || ocy < 0 || ocy >= gh) continue;

      ++scan_no;
      for (const auto &p : pts) {
        if (p.z < kWallZlo || p.z > kWallZhi) continue;
        int hcx, hcy;
        to_cell(p.x, p.y, hcx, hcy);
        const int l = std::max(0, layer_of(std::min(p.z, kWallZhi - 1e-4f)));
        if (hcx >= 0 && hcx < gw && hcy >= 0 && hcy < gh) hit_stamp[l * N + hcy * gw + hcx] = scan_no;
      }

      for (const auto &p : pts) {
        int ecx, ecy;
        to_cell(p.x, p.y, ecx, ecy);
        if (ecx < 0 || ecx >= gw || ecy < 0 || ecy >= gh) continue;
        bool wall_hit = (p.z >= kWallZlo && p.z <= kWallZhi);
        const float r = std::hypot(p.x - oit->second.x, p.y - oit->second.y);
        const float w = r > kHitRangeRef ? (kHitRangeRef / r) * (kHitRangeRef / r) : 1.0f;

        bresenham(ocx, ocy, ecx, ecy, [&](int cx, int cy, int step, int n_steps) {
          if (cx < 0 || cx >= gw || cy < 0 || cy >= gh) return;
          const size_t idx = (size_t)cy * gw + cx;
          if (cx == ecx && cy == ecy) {
            if (wall_hit) {
              const int l = std::max(0, layer_of(std::min(p.z, kWallZhi - 1e-4f)));
              float &v = lo[l * N + idx];
              v = std::min(v + kLogOccHit * w, kLogClamp);
              min_hit_r[idx] = std::min(min_hit_r[idx], r);
              sum_w += w;
              ++n_hits;
            }
            return;
          }
          const float t = n_steps > 0 ? (float)step / n_steps : 1.0f;
          const float zr = kSensorZ + t * (p.z - kSensorZ);  // ray height at this cell
          if (kHeightVeto && hw[idx] >= kVetoMinW &&
              (zr > veto_hi[idx] + kVetoMargin || zr < veto_lo[idx] - kVetoMargin)) {
            ++n_free_vetoed;
            return;
          }
          const int l = layer_of(zr);
          if (l < 0) { ++n_free_outside; return; }
          if (hit_stamp[l * N + idx] == scan_no) {
            ++n_free_protected;
          } else {
            float &v = lo[l * N + idx];
            v = std::max(v - kLogFreeMiss, -kLogClamp);
            ++n_free_upd;
          }
        });
      }
    }

    for (size_t i = 0; i < N; ++i) {
      bool occ = false, fr = false;
      for (int l = 0; l < kLayers; ++l) {
        const float v = lo[l * N + i];
        if (v > 0.0f) occ = true;
        else if (v < 0.0f) fr = true;
      }
      if      (occ) grid[i] = 100;
      else if (fr)  grid[i] = 0;
      // no evidence in any layer → stays unknown (-1)
    }
    if (kNearKeep) {
      size_t n_kept = 0;
      for (size_t i = 0; i < N; ++i)
        if (grid[i] != 100 && near_hits[i] >= kKeepMinHits && near_scans[i] >= kKeepMinScans &&
            t_last[i] - t_first[i] >= kSpanMinSeconds) {   // same persistence test: people pass close too
          grid[i] = 100;
          ++n_kept;
        }
      printf("[grid] near-range keep: %zu cells restored as occupied\n", n_kept);
      add_metric("grid near-range keep: %zu cells (%.2f m2) kept occupied (>= %d hits from < %.1f m by >= %d scans)",
                 n_kept, n_kept * (double)kRes * kRes, kKeepMinHits, kNearR, kKeepMinScans);
    }
    if (kSpanKeep) {
      size_t n_kept = 0, n_flag = 0, n_transient = 0;
      for (size_t i = 0; i < N; ++i) {
        if (all_hits[i] < kSpanMinHits || all_scans[i] < kSpanMinScans) continue;
        if (hz_max[i] - hz_min[i] > kSpanMaxDz) continue;
        if (t_last[i] - t_first[i] < kSpanMinSeconds) { ++n_transient; continue; }
        feature_keep[i] = 1; ++n_flag;
        if (grid[i] != 100) { grid[i] = 100; ++n_kept; }
      }
      printf("[grid] height-span keep: %zu cells restored (%zu flagged, %zu transient skipped)\n", n_kept, n_flag, n_transient);
      add_metric("grid height-span keep: %zu cells (%.2f m2) restored, %zu flagged, %zu transient skipped (>= %d hits, >= %d scans, z span <= %.2f m, seen over >= %.0f s; exempt from CC filter)",
                 n_kept, n_kept * (double)kRes * kRes, n_flag, n_transient, kSpanMinHits, kSpanMinScans, kSpanMaxDz, kSpanMinSeconds);
    }
  }
  {
    const double tot = std::max<size_t>(1, n_free_outside + n_free_upd + n_free_protected + n_free_vetoed);
    add_metric("grid free space: %s%s  (sensor z=%.2f m)  ray cells skipped: outside band %.1f%%, height veto %.2f%% (min w %.1f, margin %.2f m, range %s)",
               kHeightAwareFree ? "height layers" : "2D", kHeightVeto ? " + height veto" : "", kSensorZ,
               100.0 * n_free_outside / tot, 100.0 * n_free_vetoed / tot, kVetoMinW, kVetoMargin,
               kVetoPercentile ? "hit z p10-p90" : "hit z min-max");
  }
  printf("[grid] %d × %d cells  (%.1f × %.1f m)  log-odds raytrace"
         "  wall=[%.2f,%.2f m]  occ+=%.2f·w(r)  free-=%.2f  w ref=%.1f m\n",
         gw, gh, gw*kRes, gh*kRes, kWallZlo, kWallZhi, kLogOccHit, kLogFreeMiss, kHitRangeRef);
  add_metric("grid mode: range-weighted raytrace  occ+=%.2f*w(r) w=min(1,(%.1fm/r)^2)  free-=%.2f  clamp=%.1f  wall z=[%.2f,%.2f]  mean hit w=%.2f",
             kLogOccHit, kHitRangeRef, kLogFreeMiss, kLogClamp, kWallZlo, kWallZhi,
             n_hits ? sum_w / n_hits : 0.0);
  add_metric("  own-scan hit protection: %.2f%% of free updates skipped (%zu of %zu)",
             100.0 * n_free_protected / std::max<size_t>(1, n_free_protected + n_free_upd),
             n_free_protected, n_free_protected + n_free_upd);

  // ── 3. Morph close then erode: merge double walls, then thin back down ───
  {
    cv::Mat occ_mask(gh, gw, CV_8U, cv::Scalar(0));
    for (int r = 0; r < gh; ++r)
      for (int c = 0; c < gw; ++c)
        if (grid[r*gw+c] == 100) occ_mask.at<uint8_t>(r,c) = 255;

    cv::Mat k3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::morphologyEx(occ_mask, occ_mask, cv::MORPH_CLOSE, k3);  // bridge small gaps

    for (int r = 0; r < gh; ++r)
      for (int c = 0; c < gw; ++c) {
        if (occ_mask.at<uint8_t>(r,c) > 0) grid[r*gw+c] = 100;
        else if (grid[r*gw+c] == 100)       grid[r*gw+c] = -1;  // erased by thinning → unknown
      }
  }

  // ── 4. Connected component filter — remove small isolated occupied blobs ─
  // Walls form large connected shapes; noise/furniture hits are small clusters.
  {
    cv::Mat occ_mask(gh, gw, CV_8U, cv::Scalar(0));
    for (int r = 0; r < gh; ++r)
      for (int c = 0; c < gw; ++c)
        if (grid[r*gw+c] == 100) occ_mask.at<uint8_t>(r,c) = 255;

    cv::Mat labels, stats, centroids;
    int n_labels = cv::connectedComponentsWithStats(occ_mask, labels, stats, centroids);

    constexpr int kMinComponentCells = 75;
    for (int r = 0; r < gh; ++r)
      for (int c = 0; c < gw; ++c) {
        int lbl = labels.at<int>(r, c);
        if (lbl == 0) continue;  // background
        if (stats.at<int>(lbl, cv::CC_STAT_AREA) < kMinComponentCells && !feature_keep[r * gw + c])
          grid[r * gw + c] = -1;  // remove small blob → unknown (height-span features exempt)
      }
    printf("[grid] CC filter: removed blobs < %d cells  (%d components total)\n",
           kMinComponentCells, n_labels - 1);
  }

  // ── 6. Free space: ray-observed + chair path + small enclosed holes ─────
  // The old BFS flooded from the sensor origins through every non-occupied
  // cell, unknown included, so one gap in an outer wall flooded the whole grid
  // rectangle.  Now free space is only:
  //   - cells the raytrace observed as free (already 0),
  //   - the cells the chair drove through,
  //   - unknown pockets that are enclosed (don't touch the grid border) and
  //     small (≤ kMaxHoleM2) — behind furniture, under tables.  Larger unseen
  //     areas (e.g. a closet never looked into) stay unknown.
  // Outside the building stays unknown regardless of wall gaps.
  {
    constexpr double kMaxHoleM2 = 1.0;

    for (const auto &[nid, o] : per_scan_origin_) {
      int ocx, ocy;
      to_cell(o.x, o.y, ocx, ocy);
      if (ocx < 0 || ocx >= gw || ocy < 0 || ocy >= gh) continue;
      int8_t &v = grid[ocy * gw + ocx];
      if (v != 100) v = 0;
    }

    cv::Mat unk(gh, gw, CV_8U, cv::Scalar(0));
    for (int i = 0; i < gw * gh; ++i)
      if (grid[i] == -1) unk.data[i] = 255;
    cv::Mat labels, stats, centroids;
    const int n_lab = cv::connectedComponentsWithStats(unk, labels, stats, centroids, 4);

    const int max_hole_cells = (int)(kMaxHoleM2 / ((double)kRes * kRes));
    std::vector<uint8_t> fill(n_lab, 0);
    int n_holes = 0, n_big_enclosed = 0;
    size_t n_hole_cells = 0;
    for (int l = 1; l < n_lab; ++l) {
      const int x = stats.at<int>(l, cv::CC_STAT_LEFT),  y = stats.at<int>(l, cv::CC_STAT_TOP);
      const int w = stats.at<int>(l, cv::CC_STAT_WIDTH), h = stats.at<int>(l, cv::CC_STAT_HEIGHT);
      const int area = stats.at<int>(l, cv::CC_STAT_AREA);
      const bool touches_border = x == 0 || y == 0 || x + w == gw || y + h == gh;
      if (touches_border) continue;
      if (area > max_hole_cells) { ++n_big_enclosed; continue; }
      fill[l] = 1;
      ++n_holes;
      n_hole_cells += area;
    }
    for (int r = 0; r < gh; ++r)
      for (int c = 0; c < gw; ++c)
        if (fill[labels.at<int>(r, c)]) grid[r * gw + c] = 0;

    printf("[grid] free space: filled %d enclosed unknown holes (%zu cells), kept %d larger enclosed areas unknown\n",
           n_holes, n_hole_cells, n_big_enclosed);
    add_metric("free space: ray-observed + path + %d enclosed holes filled (%.2f m2, each <= %.1f m2); %d larger enclosed unknown areas kept",
               n_holes, n_hole_cells * (double)kRes * kRes, kMaxHoleM2, n_big_enclosed);
  }
  // Where the final occupied cells' evidence came from: nearest hit range.
  {
    size_t n_near = 0, n_mid = 0, n_far = 0, n_nohit = 0;
    for (int i = 0; i < gw * gh; ++i) {
      if (grid[i] != 100) continue;
      const float r = min_hit_r[i];
      if      (r > 1e8f) ++n_nohit;  // added by morph close
      else if (r < 1.3f) ++n_near;
      else if (r < 1.9f) ++n_mid;
      else               ++n_far;
    }
    const double tot = std::max<size_t>(1, n_near + n_mid + n_far + n_nohit);
    add_metric("grid occupied by nearest hit range: <1.3m %.1f%%  1.3-1.9m %.1f%%  >1.9m %.1f%%  morph-only %.1f%%",
               100.0 * n_near / tot, 100.0 * n_mid / tot, 100.0 * n_far / tot, 100.0 * n_nohit / tot);
  }

  // ── 7. Wall thinning (env WC_THIN_WALLS=1) ──────────────────────────────
  // Wall-like occupied cells are reduced to a centre line (Zhang–Suen), spurs
  // shorter than kThinPrune cells are trimmed, and the line is redrawn
  // kThinWidth cells wide (7.5 cm).  Untouched:
  //   - height-span feature cells (feature_keep), dilated by 1 cell;
  //   - furniture: cells >= kThinBlobDt from the edge, in compact cores of
  //     >= kThinMinSeed cells (a core >= kThinElong:1 and >= kThinMinWallLen
  //     long is a thick wall and IS thinned), grown back over the footprint;
  //   - ledge walls, found automatically: most outer walls carry a ~0.75 m
  //     ledge 12–15 cm deep, and for the chair its front is the obstacle edge,
  //     so those walls stay thick.  Seeds = occupied cells within
  //     kLedgeNearUnk of unknown space (outer walls) whose hits all sit high
  //     (hit-height p10 >= kLedgeZmin, weight >= 2) — the ledge front.  Seed
  //     runs shorter than kLedgeMinLen (after bridging 2-cell gaps) are
  //     dropped; protected = occupied cells within kLedgeNearUnk of unknown
  //     and kLedgeGrow of a seed.  On 10-12-01 this finds all 14 squares the
  //     user marked except one short wall (F2), plus the ledge continuing
  //     along the bottom wall (D6, F6–H6) and in H1/J2.
  //     Optional extra squares: WC_LEDGE_SQUARES="B2 K3" (50-cell squares of
  //     the saved PGM, columns A.. left to right, rows 1.. top to bottom).
  // Removed cells become free if nearer free than unknown space, else unknown;
  // removed cells that would let free space touch unknown space are put back.
  // Validated in Python (analysis/thin_walls.py --auto-ledge) on 10-12-01:
  // walls 11.5 -> 7.5 cm median, no new openings, features and furniture kept.
  // On by default; WC_THIN_WALLS=0 disables.
  // Runs in PGM orientation (row 0 = max y) so it matches the Python exactly.
  {
    const char *tw_env = std::getenv("WC_THIN_WALLS");
    const bool kThinWalls = !(tw_env && std::atoi(tw_env) == 0);
    if (kThinWalls) {
      constexpr float kThinBlobDt     = 3.5f;   // cells (core >= ~17.5 cm thick)
      constexpr int   kThinMinSeed    = 12;     // cells
      constexpr float kThinElong      = 4.0f;
      constexpr float kThinMinWallLen = 20.0f;  // cells (50 cm)
      constexpr int   kThinPrune      = 5;      // cells (12.5 cm spurs)
      constexpr int   kThinWidth      = 3;      // cells (7.5 cm)
      constexpr float kThinLedgeBand  = 8.0f;   // cells (20 cm)
      constexpr int   kThinSquare     = 50;     // cells per labelled square
      constexpr float kLedgeZmin      = 0.50f;  // m, hit-height p10 of a ledge-front cell
      constexpr float kLedgeNearUnk   = 10.0f;  // cells (25 cm) from unknown space
      constexpr float kLedgeMinLen    = 20.0f;  // cells (50 cm) of seed run
      constexpr int   kLedgeGrow      = 6;      // cells (15 cm) from a seed
      const int H = gh, W = gw;
      auto at = [&](int ir, int c) -> int8_t & { return grid[(size_t)(H - 1 - ir) * W + c]; };
      cv::Mat occ(H, W, CV_8U, cv::Scalar(0)), fre(H, W, CV_8U, cv::Scalar(0)), unk(H, W, CV_8U, cv::Scalar(0));
      cv::Mat feat(H, W, CV_8U, cv::Scalar(0));
      for (int ir = 0; ir < H; ++ir)
        for (int c = 0; c < W; ++c) {
          const int8_t v = at(ir, c);
          if (v == 100) occ.at<uint8_t>(ir, c) = 1;
          else if (v == 0) fre.at<uint8_t>(ir, c) = 1;
          else unk.at<uint8_t>(ir, c) = 1;
          if (feature_keep[(size_t)(H - 1 - ir) * W + c]) feat.at<uint8_t>(ir, c) = 1;
        }
      const cv::Mat k3 = cv::Mat::ones(3, 3, CV_8U);
      // distance (cells) to the nearest cell of a class
      auto dist_to = [&](const cv::Mat &cls) {
        cv::Mat src = (cls == 0) / 255, d;   // 1 where not the class, 0 on the class
        cv::distanceTransform(src, d, cv::DIST_L2, 5);
        return d;
      };
      // protected: features
      cv::Mat prot;
      cv::dilate(feat, prot, k3);
      prot = prot & occ;
      // protected: compact thick cores (furniture)
      cv::Mat dt_occ;
      cv::distanceTransform(occ, dt_occ, cv::DIST_L2, 5);
      cv::Mat seed = (dt_occ >= kThinBlobDt) / 255;
      cv::Mat lab, st, cen;
      const int n_lab = cv::connectedComponentsWithStats(seed, lab, st, cen, 8);
      std::vector<std::vector<cv::Point2f>> comp_pts(n_lab);
      for (int ir = 0; ir < H; ++ir)
        for (int c = 0; c < W; ++c) {
          const int l = lab.at<int>(ir, c);
          if (l > 0) comp_pts[l].emplace_back((float)c, (float)ir);
        }
      cv::Mat keep_seed(H, W, CV_8U, cv::Scalar(0));
      int n_furn = 0, n_thick_wall = 0;
      for (int l = 1; l < n_lab; ++l) {
        if (st.at<int>(l, cv::CC_STAT_AREA) < kThinMinSeed) continue;
        const cv::RotatedRect rr = cv::minAreaRect(comp_pts[l]);
        const float lng = std::max(rr.size.width, rr.size.height) + 1.0f;
        const float sht = std::min(rr.size.width, rr.size.height) + 1.0f;
        if (lng >= kThinElong * sht && lng >= kThinMinWallLen) { ++n_thick_wall; continue; }
        ++n_furn;
        for (const auto &p : comp_pts[l]) keep_seed.at<uint8_t>((int)p.y, (int)p.x) = 1;
      }
      cv::Mat blob;
      cv::dilate(keep_seed, blob, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
      blob = blob & occ;
      // protected: ledge walls in the listed squares
      const cv::Mat d_unk = dist_to(unk), d_free = dist_to(fre);
      cv::Mat ledge(H, W, CV_8U, cv::Scalar(0));
      int n_ledge_seed = 0, n_ledge_runs = 0;
      if (!cell_zlo.empty() && !cell_hw.empty()) {
        cv::Mat lseed(H, W, CV_8U, cv::Scalar(0));
        for (int ir = 0; ir < H; ++ir)
          for (int c = 0; c < W; ++c) {
            const size_t gi = (size_t)(H - 1 - ir) * W + c;
            if (occ.at<uint8_t>(ir, c) && cell_hw[gi] >= 2.0f && cell_zlo[gi] >= kLedgeZmin &&
                d_unk.at<float>(ir, c) <= kLedgeNearUnk)
              lseed.at<uint8_t>(ir, c) = 1;
          }
        cv::Mat joined, llab, lst, lcen;
        cv::dilate(lseed, joined, cv::Mat::ones(5, 5, CV_8U));
        const int nl = cv::connectedComponentsWithStats(joined, llab, lst, lcen, 8);
        std::vector<std::vector<cv::Point2f>> lpts(nl);
        for (int ir = 0; ir < H; ++ir)
          for (int c = 0; c < W; ++c) {
            const int l = llab.at<int>(ir, c);
            if (l > 0) lpts[l].emplace_back((float)c, (float)ir);
          }
        std::vector<uint8_t> lkeep(nl, 0);
        for (int l = 1; l < nl; ++l) {
          const cv::RotatedRect rr = cv::minAreaRect(lpts[l]);
          if (std::max(rr.size.width, rr.size.height) >= kLedgeMinLen) { lkeep[l] = 1; ++n_ledge_runs; }
        }
        for (int ir = 0; ir < H; ++ir)
          for (int c = 0; c < W; ++c)
            if (lseed.at<uint8_t>(ir, c) && !lkeep[llab.at<int>(ir, c)]) lseed.at<uint8_t>(ir, c) = 0;
        n_ledge_seed = cv::countNonZero(lseed);
        cv::Mat near;
        cv::dilate(lseed, near, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * kLedgeGrow + 1, 2 * kLedgeGrow + 1)));
        for (int ir = 0; ir < H; ++ir)
          for (int c = 0; c < W; ++c)
            if (near.at<uint8_t>(ir, c) && occ.at<uint8_t>(ir, c) && d_unk.at<float>(ir, c) <= kLedgeNearUnk)
              ledge.at<uint8_t>(ir, c) = 1;
      }
      std::string ledge_list;
      if (const char *e = std::getenv("WC_LEDGE_SQUARES")) {
        std::istringstream ss(e);
        std::string sq;
        while (ss >> sq) {
          if (sq.size() < 2 || !std::isalpha((unsigned char)sq[0])) continue;
          const int cs = std::toupper((unsigned char)sq[0]) - 'A', rs = std::atoi(sq.c_str() + 1) - 1;
          if (rs < 0) continue;
          for (int ir = rs * kThinSquare; ir < std::min(H, (rs + 1) * kThinSquare); ++ir)
            for (int c = cs * kThinSquare; c < std::min(W, (cs + 1) * kThinSquare); ++c)
              if (occ.at<uint8_t>(ir, c) && d_unk.at<float>(ir, c) <= kThinLedgeBand) ledge.at<uint8_t>(ir, c) = 1;
          ledge_list += (ledge_list.empty() ? "" : " ") + sq;
        }
      }
      cv::Mat protect = prot | blob | ledge;
      cv::Mat wall = occ & (1 - protect);

      // Zhang–Suen thinning (as cv::ximgproc::thinning, THINNING_ZHANGSUEN)
      cv::Mat sk = wall.clone();
      for (bool changed = true; changed;) {
        changed = false;
        for (int iter = 0; iter < 2; ++iter) {
          cv::Mat marker(H, W, CV_8U, cv::Scalar(0));
          for (int i = 1; i < H - 1; ++i)
            for (int j = 1; j < W - 1; ++j) {
              if (!sk.at<uint8_t>(i, j)) continue;
              const int p2 = sk.at<uint8_t>(i - 1, j), p3 = sk.at<uint8_t>(i - 1, j + 1);
              const int p4 = sk.at<uint8_t>(i, j + 1), p5 = sk.at<uint8_t>(i + 1, j + 1);
              const int p6 = sk.at<uint8_t>(i + 1, j), p7 = sk.at<uint8_t>(i + 1, j - 1);
              const int p8 = sk.at<uint8_t>(i, j - 1), p9 = sk.at<uint8_t>(i - 1, j - 1);
              const int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                            (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) + (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
              const int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
              const int m1 = iter == 0 ? p2 * p4 * p6 : p2 * p4 * p8;
              const int m2 = iter == 0 ? p4 * p6 * p8 : p2 * p6 * p8;
              if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) marker.at<uint8_t>(i, j) = 1;
            }
          if (cv::countNonZero(marker)) { sk = sk & (1 - marker); changed = true; }
        }
      }
      // spur pruning: peel endpoints kThinPrune times, regrow along the skeleton
      auto endpoints = [&](const cv::Mat &s) {
        cv::Mat n;
        cv::filter2D(s, n, CV_32F, cv::Mat::ones(3, 3, CV_32F), cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        cv::Mat sf; s.convertTo(sf, CV_32F);
        return (s > 0) & ((n - sf) == 1.0f);     // 255 where endpoint
      };
      cv::Mat s = sk.clone();
      for (int k = 0; k < kThinPrune; ++k) s = s & (1 - endpoints(s) / 255);
      cv::Mat grow = s.clone();
      for (int k = 0; k < kThinPrune; ++k) {
        cv::Mat ep = endpoints(grow) / 255, epd;
        cv::dilate(ep, epd, k3);
        grow = grow | (epd & sk);
      }
      cv::Mat new_wall;
      cv::dilate(grow, new_wall, cv::Mat::ones(kThinWidth, kThinWidth, CV_8U));
      new_wall = new_wall & (wall | fre);
      cv::Mat removed = wall & (1 - new_wall);

      // write back: removed -> free / unknown, new wall and protected -> occupied
      for (int ir = 0; ir < H; ++ir)
        for (int c = 0; c < W; ++c) {
          if (removed.at<uint8_t>(ir, c))
            at(ir, c) = d_free.at<float>(ir, c) <= d_unk.at<float>(ir, c) ? 0 : -1;
          if (new_wall.at<uint8_t>(ir, c) || protect.at<uint8_t>(ir, c)) at(ir, c) = 100;
        }
      // seal: no free/unknown contact that the input grid did not have
      auto contact = [&](bool from_grid, const cv::Mat &f0, const cv::Mat &u0) {
        cv::Mat f(H, W, CV_8U), u(H, W, CV_8U);
        for (int ir = 0; ir < H; ++ir)
          for (int c = 0; c < W; ++c) {
            const int8_t v = from_grid ? at(ir, c) : (f0.at<uint8_t>(ir, c) ? 0 : (u0.at<uint8_t>(ir, c) ? -1 : 100));
            f.at<uint8_t>(ir, c) = v == 0; u.at<uint8_t>(ir, c) = v == -1;
          }
        cv::Mat out(H, W, CV_8U, cv::Scalar(0));
        for (int ir = 0; ir < H; ++ir)
          for (int c = 0; c < W; ++c) {
            const bool fr = f.at<uint8_t>(ir, c), un = u.at<uint8_t>(ir, c);
            if (!fr && !un) continue;
            const cv::Mat &other = fr ? u : f;
            if ((ir > 0 && other.at<uint8_t>(ir - 1, c)) || (ir < H - 1 && other.at<uint8_t>(ir + 1, c)) ||
                (c > 0 && other.at<uint8_t>(ir, c - 1)) || (c < W - 1 && other.at<uint8_t>(ir, c + 1)))
              out.at<uint8_t>(ir, c) = 1;
          }
        return out;
      };
      const cv::Mat base = contact(false, fre, unk);
      size_t n_restored = 0;
      for (int it = 0; it < 20; ++it) {
        const cv::Mat now = contact(true, fre, unk);
        size_t n_it = 0;
        for (int ir = 0; ir < H; ++ir)
          for (int c = 0; c < W; ++c)
            if (now.at<uint8_t>(ir, c) && !base.at<uint8_t>(ir, c) && removed.at<uint8_t>(ir, c) && at(ir, c) != 100) {
              at(ir, c) = 100; ++n_it;
            }
        n_restored += n_it;
        if (n_it == 0) break;
      }
      const size_t n_removed = (size_t)cv::countNonZero(removed) - n_restored;
      printf("[grid] wall thinning: %zu cells removed (%zu put back by the seal); protected: features %d, furniture %d, ledge %d cells\n",
             n_removed, n_restored, cv::countNonZero(prot), cv::countNonZero(blob), cv::countNonZero(ledge));
      add_metric("grid wall thinning (WC_THIN_WALLS=0 disables): %zu cells (%.2f m2) removed, %zu put back by the seal; walls redrawn %d cells wide; protected cells: features %d, furniture %d (%d compact cores, %d thick-wall cores thinned), ledge %d (auto: %d seed cells in %d runs; extra squares \"%s\")",
                 n_removed, n_removed * (double)kRes * kRes, n_restored, kThinWidth,
                 cv::countNonZero(prot), cv::countNonZero(blob), n_furn, n_thick_wall, cv::countNonZero(ledge),
                 n_ledge_seed, n_ledge_runs, ledge_list.c_str());
    }
  }
#endif  // WC_GRID_DIRECT_PROJECTION

  // ── 7. PCA rotation — orient map so widest extent is horizontal ──────────
  // Only with WC_GRID_ALIGN=pca; step 1 aligns the grid with WC_GRID_ALIGN=1.
  if (kGridAlign == 2) {
    // Encode grid into uint8 image: 255=occupied, 128=free, 0=unknown
    cv::Mat grid_mat(gh, gw, CV_8U, cv::Scalar(0));
    for (int r = 0; r < gh; ++r)
      for (int c = 0; c < gw; ++c) {
        int8_t v = grid[r * gw + c];
        grid_mat.at<uint8_t>(r, c) = (v == 100) ? 255u : (v == 0) ? 128u : 0u;
      }

    // Compute PCA on occupied wall cells — their distribution directly encodes
    // the building's structural orientation, more reliably than the trajectory.
    double mx = 0, my = 0;
    int n_occ = 0;
    for (int r = 0; r < gh; ++r)
      for (int c = 0; c < gw; ++c)
        if (grid[r * gw + c] == 100) { mx += c; my += r; ++n_occ; }

    if (n_occ > 1) {
      mx /= n_occ;
      my /= n_occ;
      double cxx = 0, cyy = 0, cxy = 0;
      for (int r = 0; r < gh; ++r)
        for (int c = 0; c < gw; ++c)
          if (grid[r * gw + c] == 100) {
            double dc = c - mx;
            double dr = r - my;
            cxx += dc * dc;
            cyy += dr * dr;
            cxy += dc * dr;
          }
      // Principal axis angle (radians, CCW from +col / +x)
      double angle_rad = 0.5 * std::atan2(2.0 * cxy, cxx - cyy);
      double angle_deg = angle_rad * 180.0 / M_PI;

      // Rotate grid image so the principal axis aligns with horizontal.
      // getRotationMatrix2D(+a) rotates content CW in standard y-up coords;
      // a pixel at angle θ from horizontal moves to horizontal when we pass +θ.
      cv::Point2f center((float)(gw * 0.5), (float)(gh * 0.5));
      cv::Mat M = cv::getRotationMatrix2D(center, angle_deg, 1.0);

      // New canvas large enough to contain the fully rotated image
      double abs_cos = std::abs(M.at<double>(0, 0));
      double abs_sin = std::abs(M.at<double>(0, 1));
      int new_w = (int)std::ceil(gh * abs_sin + gw * abs_cos);
      int new_h = (int)std::ceil(gh * abs_cos + gw * abs_sin);

      // Shift translation so rotated content is centered in the new canvas
      M.at<double>(0, 2) += (new_w - gw) * 0.5;
      M.at<double>(1, 2) += (new_h - gh) * 0.5;

      cv::Mat rotated;
      cv::warpAffine(grid_mat, rotated, M, cv::Size(new_w, new_h),
                     cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

      // If portrait, rotate 90° CW to make landscape
      if (rotated.rows > rotated.cols) {
        cv::Mat landscape;
        cv::rotate(rotated, landscape, cv::ROTATE_90_CLOCKWISE);
        rotated = landscape;
        angle_deg += 90.0;
      }

      // Keep the world-space center of the grid fixed, update origin
      double old_cx = xmin + gw * 0.5 * kRes;
      double old_cy = ymin + gh * 0.5 * kRes;
      gw = rotated.cols;
      gh = rotated.rows;
      xmin = (float)(old_cx - gw * 0.5 * kRes);
      ymin = (float)(old_cy - gh * 0.5 * kRes);

      // Write rotated image back into the flat grid vector
      grid.assign((size_t)gw * (size_t)gh, (int8_t)-1);
      for (int r = 0; r < gh; ++r)
        for (int c = 0; c < gw; ++c) {
          uint8_t pv = rotated.at<uint8_t>(r, c);
          grid[r * gw + c] = (pv == 255u) ? (int8_t)100
                            : (pv == 128u) ? (int8_t)0
                                           : (int8_t)-1;
        }

      printf("[grid] PCA rotation %.1f deg → new size %d x %d cells\n",
             angle_deg, gw, gh);
    }
  }

  // Build Nav2 OccupancyGrid message.
  // Row 0 = min-y (south), data[r*w+c] = world (xmin+c*res, ymin+r*res).
  rtabmap_occupancy_grid_ = std::make_shared<nav_msgs::msg::OccupancyGrid>();
  {
    auto &info = rtabmap_occupancy_grid_->info;
    info.resolution           = kRes;
    info.width                = (uint32_t)gw;
    info.height               = (uint32_t)gh;
    info.origin.position.x    = xmin;
    info.origin.position.y    = ymin;
    info.origin.position.z    = 0.0;
    info.origin.orientation.w = 1.0;
    rtabmap_occupancy_grid_->data = std::vector<int8_t>(grid.begin(), grid.end());
  }

  {
    size_t n_occ = 0, n_free = 0, n_unk = 0;
    for (int8_t v : grid) {
      if (v == 100) ++n_occ;
      else if (v == 0) ++n_free;
      else ++n_unk;
    }
    const double cell_m2 = (double)kRes * kRes;
    add_metric("grid: %d x %d @ %.3f m  occupied=%zu (%.2f m2)  free=%zu (%.1f m2)  unknown=%zu  occ/free=%.4f",
               gw, gh, kRes, n_occ, n_occ * cell_m2, n_free, n_free * cell_m2, n_unk,
               n_free ? (double)n_occ / n_free : 0.0);
  }

  // ── scan cloud → PCD (streamed scan-by-scan, no voxelization) ───────────
  {
    size_t total_pts = 0;
    for (const auto &[nid, pts] : per_scan_xyz_) total_pts += pts.size();

    std::string cloud_path = path + "/cloud/" + timestamp_ + ".pcd";
    std::ofstream ofs(cloud_path, std::ios::binary);
    ofs << "# .PCD v0.7 - Point Cloud Data file format\n"
        << "VERSION 0.7\n"
        << "FIELDS x y z\n"
        << "SIZE 4 4 4\n"
        << "TYPE F F F\n"
        << "COUNT 1 1 1\n"
        << "WIDTH "    << total_pts << "\n"
        << "HEIGHT 1\n"
        << "VIEWPOINT 0 0 0 1 0 0 0\n"
        << "POINTS "   << total_pts << "\n"
        << "DATA binary\n";

    for (const auto &[nid, pts] : per_scan_xyz_)
      for (const auto &p : pts) {
        ofs.write(reinterpret_cast<const char*>(&p.x), 4);
        ofs.write(reinterpret_cast<const char*>(&p.y), 4);
        ofs.write(reinterpret_cast<const char*>(&p.z), 4);
      }
    ofs.close();
    printf("[cloud] Saved %zu points → %s\n", total_pts, cloud_path.c_str());
  }

  // ── 5. Save grid PGM + YAML ───────────────────────────────────────────
  {
    std::string grid_path = path + "/grid/" + grid_name;
    nav2_map_server::SaveParameters sp;
    sp.map_file_name   = grid_path;
    sp.image_format    = "pgm";
    sp.free_thresh     = 0.196;
    sp.occupied_thresh = 0.65;
    nav2_map_server::saveMapToFile(*rtabmap_occupancy_grid_, sp);
    printf("[grid] Saved grid → %s.{pgm,yaml}\n", grid_path.c_str());

    // Ground truth -> map transform, appended to the map YAML (map_server
    // ignores extra keys).  Fitted over the whole drive: graph node positions
    // vs /ground_truth/odom interpolated at the node stamps (SE(2) least
    // squares).  On four recordings that is 33–38 mm median error; anchoring on
    // /ground_truth/initial_pose alone (GT pose at the first node) gave 76–173 mm,
    // because the map rotation then hangs on the first node's graph heading
    // (~2° heading error per node).  initial_pose is still written for reference.
    // gt_heading_offset_deg = median(GT yaw − map heading after the fit): the
    // ceiling camera reads +2.4..+2.7° high against the map's headings (four
    // recordings); subtract it from GT yaw before comparing headings.
    // Bag: WC_GT_BAG, else <db dir>/tf_<db name>.  Skipped without a bag.
    {
      std::string gt_bag;
      if (const char *e = std::getenv("WC_GT_BAG")) gt_bag = e;
      else {
        const std::string db = rtabmap_database_path_;
        const size_t sl = db.rfind('/');
        const std::string dir = sl == std::string::npos ? "." : db.substr(0, sl);
        std::string name = sl == std::string::npos ? db : db.substr(sl + 1);
        if (name.size() > 3 && name.compare(name.size() - 3, 3, ".db") == 0) name.resize(name.size() - 3);
        gt_bag = dir + "/tf_" + name;
      }
      auto first = viz_poses_.lower_bound(1);
      if (access((gt_bag + "/metadata.yaml").c_str(), F_OK) != 0) {
        add_metric("ground truth tf: no bag at %s (set WC_GT_BAG) — not added to the map yaml", gt_bag.c_str());
      } else if (first == viz_poses_.end()) {
        add_metric("ground truth tf: no graph nodes — not added to the map yaml");
      } else {
        auto nit0 = nodes.find(first->first);
        const double t_first = nit0 != nodes.end() ? nit0->second.getStamp() : 0.0;
        auto yaw_of = [](const geometry_msgs::msg::Quaternion &q) {
          return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        };
        auto wrap = [](double a) { return std::atan2(std::sin(a), std::cos(a)); };
        struct GtS { double t, x, y, yaw; };
        std::vector<GtS> gt;
        bool have_ip = false;
        double ix = 0, iy = 0, iyaw = 0, it = 0;
        std::string gt_frame;
        try {
          rosbag2_cpp::Reader reader;
          rosbag2_storage::StorageOptions so;
          so.uri = gt_bag;
          so.storage_id = "mcap";
          reader.open(so);
          rosbag2_storage::StorageFilter filt;
          filt.topics = {"/ground_truth/initial_pose", "/ground_truth/odom"};
          reader.set_filter(filt);
          rclcpp::Serialization<geometry_msgs::msg::PoseWithCovarianceStamped> ser_ip;
          rclcpp::Serialization<nav_msgs::msg::Odometry> ser_od;
          while (reader.has_next()) {
            auto m = reader.read_next();
            rclcpp::SerializedMessage sm(*m->serialized_data);
            if (m->topic_name == "/ground_truth/odom") {
              nav_msgs::msg::Odometry msg;
              ser_od.deserialize_message(&sm, &msg);
              gt.push_back({msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec,
                            msg.pose.pose.position.x, msg.pose.pose.position.y, yaw_of(msg.pose.pose.orientation)});
              if (gt_frame.empty() && !msg.header.frame_id.empty()) gt_frame = msg.header.frame_id;
            } else if (m->topic_name == "/ground_truth/initial_pose") {
              geometry_msgs::msg::PoseWithCovarianceStamped msg;
              ser_ip.deserialize_message(&sm, &msg);
              const double t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec;
              if (t > t_first + 1.0 || (have_ip && t < it)) continue;   // last one before the recording
              ix = msg.pose.pose.position.x; iy = msg.pose.pose.position.y;
              iyaw = yaw_of(msg.pose.pose.orientation); it = t; have_ip = true;
            }
          }
        } catch (const std::exception &e) {
          printf("[gt] could not read %s: %s\n", gt_bag.c_str(), e.what());
        }
        if (gt_frame.empty()) gt_frame = "ground_truth";
        std::sort(gt.begin(), gt.end(), [](const GtS &a, const GtS &b) { return a.t < b.t; });
        // node positions (map) and GT interpolated at the node stamps (gaps <= 0.1 s)
        std::vector<std::array<double, 3>> mp, gp;
        for (const auto &[nid, P] : viz_poses_) {
          if (nid < 1) continue;
          auto nit = nodes.find(nid);
          if (nit == nodes.end() || gt.size() < 2) continue;
          const double t = nit->second.getStamp();
          auto hi = std::lower_bound(gt.begin(), gt.end(), t, [](const GtS &g, double v) { return g.t < v; });
          if (hi == gt.begin() || hi == gt.end()) continue;
          auto lo = hi - 1;
          if (hi->t - lo->t > 0.1) continue;
          const double u = (t - lo->t) / std::max(1e-9, hi->t - lo->t);
          mp.push_back({P.x(), P.y(), P.theta()});
          gp.push_back({lo->x + u * (hi->x - lo->x), lo->y + u * (hi->y - lo->y), lo->yaw + u * wrap(hi->yaw - lo->yaw)});
        }
        if (mp.size() < 50) {
          add_metric("ground truth tf: only %zu nodes with ground truth in %s — not added", mp.size(), gt_bag.c_str());
        } else {
          // SE(2) least squares: g = R m + t
          double mx = 0, my = 0, gx = 0, gy = 0;
          for (size_t k = 0; k < mp.size(); ++k) { mx += mp[k][0]; my += mp[k][1]; gx += gp[k][0]; gy += gp[k][1]; }
          mx /= mp.size(); my /= mp.size(); gx /= gp.size(); gy /= gp.size();
          double sc = 0, ss = 0;
          for (size_t k = 0; k < mp.size(); ++k) {
            const double ax = mp[k][0] - mx, ay = mp[k][1] - my, bx = gp[k][0] - gx, by = gp[k][1] - gy;
            sc += ax * bx + ay * by; ss += ax * by - ay * bx;
          }
          const double th = std::atan2(ss, sc), c = std::cos(th), sn = std::sin(th);
          const double tx = gx - (c * mx - sn * my), ty = gy - (sn * mx + c * my);
          std::vector<double> err, dh;
          for (size_t k = 0; k < mp.size(); ++k) {
            const double px = c * mp[k][0] - sn * mp[k][1] + tx, py = sn * mp[k][0] + c * mp[k][1] + ty;
            err.push_back(1000.0 * std::hypot(px - gp[k][0], py - gp[k][1]));
            dh.push_back(wrap(gp[k][2] - (th + mp[k][2])));
          }
          auto pct = [](std::vector<double> v, double q) {
            std::nth_element(v.begin(), v.begin() + (size_t)(q * (v.size() - 1)), v.end());
            return v[(size_t)(q * (v.size() - 1))];
          };
          const double e50 = pct(err, 0.5), e90 = pct(err, 0.9), hoff = pct(dh, 0.5);
          std::ofstream yf(grid_path + ".yaml", std::ios::app);
          yf << std::fixed << std::setprecision(6)
             << "\n"   // nav2's saveMapToFile leaves no trailing newline
             << "# Ground truth -> map: pose of the map frame in the ground-truth frame, fitted over the whole drive\n"
             << "# (graph node positions vs /ground_truth/odom).  static TF: parent gt_frame, child map,\n"
             << "# translation (x, y, 0), rotation yaw (rad) about z.\n"
             << "gt_frame: " << gt_frame << "\n"
             << "gt_to_map: {x: " << tx << ", y: " << ty << ", yaw: " << th << "}\n"
             << std::setprecision(1)
             << "gt_fit_error_mm: {median: " << e50 << ", p90: " << e90 << ", nodes: " << mp.size() << "}\n"
             << std::setprecision(3)
             << "# GT yaw minus map heading (deg); subtract from GT yaw before comparing headings.\n"
             << "gt_heading_offset_deg: " << hoff * 180.0 / M_PI << "\n";
          if (have_ip) {
            // reference only: the single-pose transform from initial_pose
            const rtabmap::Transform IP((float)ix, (float)iy, (float)(iyaw - hoff));
            const rtabmap::Transform N0(first->second.x(), first->second.y(), first->second.theta());
            const rtabmap::Transform Ti = IP * N0.inverse();
            yf << std::setprecision(6)
               << "# Reference only (less accurate): from /ground_truth/initial_pose alone (GT pose at the first node).\n"
               << "gt_initial_pose: {x: " << ix << ", y: " << iy << ", yaw: " << iyaw
               << ", stamp: " << std::setprecision(3) << it << "}\n" << std::setprecision(6)
               << "gt_to_map_from_initial_pose: {x: " << Ti.x() << ", y: " << Ti.y() << ", yaw: " << Ti.theta() << "}\n";
          }
          add_metric("ground truth tf: %s -> map  x %.4f y %.4f yaw %.3f deg  fitted over %zu nodes, error median %.0f mm p90 %.0f mm; GT heading offset %+.2f deg (written to the map yaml)",
                     gt_frame.c_str(), tx, ty, th * 180.0 / M_PI, mp.size(), e50, e90, hoff * 180.0 / M_PI);
        }
      }
    }
  }

  // ── 6. Tag overlay PNG ────────────────────────────────────────────────
  if (!tag_obs_.empty()) {
    const int S = 4;  // pixels per grid cell
    int iw = gw * S, ih = gh * S;
    cv::Mat img(ih, iw, CV_8UC3);
    for (int r = 0; r < gh; ++r)
      for (int c = 0; c < gw; ++c) {
        int8_t v = grid[r*gw+c];
        cv::Vec3b col = (v < 0)   ? cv::Vec3b{128,128,128}
                      : (v == 0)  ? cv::Vec3b{240,240,240}
                                  : cv::Vec3b{30,30,30};
        img(cv::Rect(c*S, ih-1-r*S-S+1, S, S)).setTo(cv::Scalar(col[0],col[1],col[2]));
      }

    // Helper: world XY → image pixel (through the same alignment as the grid,
    // so path and tags land on the walls they belong to).
    auto wp = [&](float x, float y) -> cv::Point {
      const cv::Point2f q = rot(x, y);
      return { (int)((q.x-xmin)/kRes*S), ih-1-(int)((q.y-ymin)/kRes*S) };
    };

    // Draw robot path.
    cv::Point prev(-1,-1);
    for (const auto &[id, T] : viz_poses_) {
      if (id <= 0) continue;
      cv::Point pt = wp(T.x(), T.y());
      bool in = pt.x>=0 && pt.x<iw && pt.y>=0 && pt.y<ih;
      if (in && prev.x >= 0)
        cv::line(img, prev, pt, cv::Scalar(140,140,140), 1, cv::LINE_AA);
      if (in)
        cv::circle(img, pt, 1, cv::Scalar(100,100,100), -1, cv::LINE_AA);
      prev = in ? pt : cv::Point(-1,-1);
    }

    // Draw tags: individual detections (dim), mean position (bright), std circle.
    int num_tags = (int)tag_obs_.size(), tag_idx = 0;
    for (auto &[tag_id, obs] : tag_obs_) {
      float hue = 360.0f * tag_idx / std::max(num_tags, 1);
      cv::Mat hsv(1,1,CV_8UC3,cv::Scalar((uint8_t)(hue/2),210,230));
      cv::Mat bgrc; cv::cvtColor(hsv, bgrc, cv::COLOR_HSV2BGR);
      cv::Vec3b c = bgrc.at<cv::Vec3b>(0,0);
      cv::Scalar color(c[0],c[1],c[2]), dim(c[0]*.4,c[1]*.4,c[2]*.4);
      int n = (int)obs.size();
      float mx=0,my=0;
      for (auto &o : obs) { mx+=o.x; my+=o.y; }
      mx/=n; my/=n;
      float sx=0,sy=0;
      for (auto &o : obs) { sx+=(o.x-mx)*(o.x-mx); sy+=(o.y-my)*(o.y-my); }
      float std_2d = std::sqrt((sx/n+sy/n)/2.0f);
      for (auto &o : obs) {
        cv::Point pt = wp(o.x,o.y);
        if (pt.x>=0&&pt.x<iw&&pt.y>=0&&pt.y<ih)
          cv::circle(img, pt, 3, dim, -1, cv::LINE_AA);
      }
      cv::Point mp = wp(mx,my);
      int r_px = (int)(std_2d/kRes*S);
      if (r_px > 1) cv::circle(img, mp, r_px, color, 1, cv::LINE_AA);
      cv::circle(img, mp, 7, cv::Scalar(0,0,0), -1, cv::LINE_AA);
      cv::circle(img, mp, 5, color, -1, cv::LINE_AA);
      if (mp.x>=0&&mp.x<iw&&mp.y>=0&&mp.y<ih)
        cv::putText(img, std::to_string(tag_id), mp+cv::Point(9,4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
      ++tag_idx;
    }
    cv::putText(img, "dim=detection  bright=mean  circle=std  grey=robot path",
                {4, ih-6}, cv::FONT_HERSHEY_SIMPLEX, 0.38,
                cv::Scalar(100,100,100), 1, cv::LINE_AA);

    std::string tag_png = path + "/grid/" + grid_name + "_tags.png";
    cv::imwrite(tag_png, img);
    printf("[grid] Tag overlay → %s\n", tag_png.c_str());
  }

  // ── Copy-paste metrics block (also saved to <output>/metrics.txt) ─────
  {
    add_metric("pipeline: %s  (source sha256 %s)", PIPELINE_NAME, PIPELINE_SRC_SHA256);
    {
      // every WC_* setting this run used (env), sorted, so runs record their switches
      std::vector<std::string> env;
      for (char **e = environ; e && *e; ++e)
        if (std::strncmp(*e, "WC_", 3) == 0) env.emplace_back(*e);
      std::sort(env.begin(), env.end());
      std::string line;
      for (const auto &kv : env) line += (line.empty() ? "" : " ") + kv;
      add_metric("env: %s", line.empty() ? "(no WC_* settings — all defaults)" : line.c_str());
    }
    add_metric("output=%s  total runtime %.0f s", timestamp_.c_str(), run_timer_.elapsed());
    std::ofstream mf(path + "/metrics.txt");
    printf("\n===== METRICS (copy from here) =====\n");
    for (const auto &line : metrics_) {
      printf("%s\n", line.c_str());
      mf << line << "\n";
    }
    printf("===== END METRICS =====\n\n");
  }

  result.success   = true;
  result.timestamp = timestamp_;
  printf("[done] timestamp=%s\n", timestamp_.c_str());
}

// ---------------------------------------------------------------------------
// export_slam_toolbox_bag
// ---------------------------------------------------------------------------
// Writes a ROS2 mcap bag to <path>/slam_toolbox_input/ containing:
//   /tf_static  base_link → laser   (sensor localTransform, written once)
//   /tf         odom → base_link    (GTSAM-optimized pose per scan)
//   /scan       sensor_msgs/LaserScan in "laser" frame per scan
//
// Feed to slam_toolbox offline:
//   ros2 bag play <path>/slam_toolbox_input --topics /scan /tf /tf_static
//   ros2 launch slam_toolbox online_sync_launch.py   (params from config/)
// ---------------------------------------------------------------------------
void WCDatabaseExporterDev::export_slam_toolbox_bag(const std::string &path)
{
  // Sort nodes by timestamp so the bag is time-ordered.
  std::vector<std::pair<double, int>> time_nodes;
  for (const auto &[nid, pose] : optimizedPoses) {
    if (nid <= 0) continue;
    auto node_it = nodes.find(nid);
    if (node_it == nodes.end()) continue;
    double t = node_it->second.getStamp();
    if (t < 1e6) t = static_cast<double>(nid) * 0.1;
    time_nodes.push_back({t, nid});
  }
  std::sort(time_nodes.begin(), time_nodes.end());

  if (time_nodes.empty()) {
    printf("[slam_toolbox] No nodes to export\n");
    return;
  }

  std::string bag_dir = path + "/slam_toolbox_input";
  rosbag2_cpp::Writer writer;
  rosbag2_storage::StorageOptions storage_opts;
  storage_opts.uri        = bag_dir;
  storage_opts.storage_id = "mcap";
  try {
    writer.open(storage_opts);
  } catch (const std::exception &e) {
    printf("[slam_toolbox] Failed to open bag writer: %s\n", e.what());
    return;
  }

  // Helper: fill a TransformStamped from an RTABMap Transform.
  auto make_ts = [](const rtabmap::Transform &T,
                    int32_t sec, uint32_t nsec,
                    const std::string &parent,
                    const std::string &child) -> geometry_msgs::msg::TransformStamped {
    geometry_msgs::msg::TransformStamped ts;
    ts.header.stamp.sec     = sec;
    ts.header.stamp.nanosec = nsec;
    ts.header.frame_id      = parent;
    ts.child_frame_id       = child;
    Eigen::Quaternionf q    = T.getQuaternionf();
    ts.transform.translation.x = static_cast<double>(T.x());
    ts.transform.translation.y = static_cast<double>(T.y());
    ts.transform.translation.z = static_cast<double>(T.z());
    ts.transform.rotation.x    = static_cast<double>(q.x());
    ts.transform.rotation.y    = static_cast<double>(q.y());
    ts.transform.rotation.z    = static_cast<double>(q.z());
    ts.transform.rotation.w    = static_cast<double>(q.w());
    return ts;
  };

  // ── Pre-pass: determine canonical scan geometry ──────────────────────────
  // slam_toolbox requires every scan in the bag to have the same angle_min,
  // angle_max, and angle_increment.  Compute these once from the first scan
  // that carries valid RTABMap metadata; fall back to full-circle 720-bin
  // defaults if none do.
  float canon_a_min = -static_cast<float>(M_PI);
  float canon_a_max =  static_cast<float>(M_PI);
  float canon_a_inc = 2.0f * static_cast<float>(M_PI) / 720.0f;
  int   canon_n_bins = 720;

  for (const auto &[stamp, nid] : time_nodes) {
    auto node_it = nodes.find(nid);
    if (node_it == nodes.end()) continue;
    rtabmap::LaserScan sc;
    node_it->second.sensorData().uncompressData(nullptr, nullptr, &sc);
    if (sc.isEmpty()) continue;
    if (!std::isnan(sc.angleMin()) && !std::isnan(sc.angleIncrement()) && sc.angleIncrement() > 0.f) {
      canon_a_min  = sc.angleMin();
      canon_a_max  = sc.angleMax();
      canon_a_inc  = sc.angleIncrement();
      canon_n_bins = std::max(1, static_cast<int>(std::round((canon_a_max - canon_a_min) / canon_a_inc)) + 1);
      break;
    }
  }
  printf("[slam_toolbox] Canonical scan: %.3f→%.3f rad, inc=%.5f rad, %d bins\n",
         canon_a_min, canon_a_max, canon_a_inc, canon_n_bins);

  // Capture the sensor→base transform from the first valid scan so we can
  // bundle it into every /tf message.  Using /tf_static from a bag fails
  // because bag2 writes it with VOLATILE durability while tf2 subscribers
  // expect TRANSIENT_LOCAL — the static TF is never received and all scans
  // are dropped.  Including it on /tf each time sidesteps the QoS issue.
  rtabmap::Transform laser_to_base = rtabmap::Transform::getIdentity();
  bool laser_tf_captured = false;
  int  n_written = 0;

  for (const auto &[stamp, nid] : time_nodes) {
    auto node_it = nodes.find(nid);
    const rtabmap::Transform &pose = optimizedPoses.at(nid);

    rtabmap::LaserScan scan;
    node_it->second.sensorData().uncompressData(nullptr, nullptr, &scan);
    if (scan.isEmpty()) continue;

    if (!laser_tf_captured) {
      laser_to_base    = scan.localTransform();
      laser_tf_captured = true;
    }

    int64_t  stamp_ns = static_cast<int64_t>(stamp * 1e9);
    int32_t  sec      = static_cast<int32_t>(stamp_ns / 1000000000LL);
    uint32_t nsec     = static_cast<uint32_t>(stamp_ns % 1000000000LL);
    rclcpp::Time ros_t(sec, nsec, RCL_ROS_TIME);

    // Both transforms in one /tf message: odom→base_link + base_link→laser.
    // Bundling the sensor transform here avoids the /tf_static QoS problem.
    {
      tf2_msgs::msg::TFMessage msg;
      msg.transforms.push_back(make_ts(pose, sec, nsec, "gtsam_odom", "base_link"));
      msg.transforms.push_back(make_ts(laser_to_base, sec, nsec, "base_link", "laser"));
      writer.write(msg, "/tf", ros_t);
    }

    // LaserScan in "laser" frame: Cartesian (x,y) → polar (angle, range).
    // Always use the canonical geometry so every scan has the same bin count.
    {
      const cv::Mat &raw = scan.data();
      int n_pts = raw.cols;
      int ch    = raw.channels();
      if (n_pts == 0 || ch < 2) continue;

      sensor_msgs::msg::LaserScan ls;
      ls.header.stamp.sec     = sec;
      ls.header.stamp.nanosec = nsec;
      ls.header.frame_id      = "laser";
      ls.angle_min            = canon_a_min;
      ls.angle_max            = canon_a_max;
      ls.angle_increment      = canon_a_inc;
      ls.time_increment       = 0.f;
      ls.scan_time            = 0.1f;
      ls.range_min            = 0.05f;
      ls.range_max            = 25.f;
      ls.ranges.assign(canon_n_bins, std::numeric_limits<float>::infinity());

      for (int i = 0; i < n_pts; ++i) {
        const float *p = raw.ptr<float>(0) + i * ch;
        float x = p[0], y = p[1];
        float r = std::sqrt(x*x + y*y);
        if (!std::isfinite(r) || r < 0.02f || r > 25.f) continue;
        int idx = static_cast<int>(std::round((std::atan2(y, x) - canon_a_min) / canon_a_inc));
        if (idx >= 0 && idx < canon_n_bins)
          ls.ranges[idx] = std::min(ls.ranges[idx], r);
      }

      writer.write(ls, "/scan", rclcpp::Time(stamp_ns + 1000, RCL_ROS_TIME));
      ++n_written;
    }
  }

  printf("[slam_toolbox] Exported %d scans → %s\n", n_written, bag_dir.c_str());
  printf("[slam_toolbox] Run: scripts/run_slam_toolbox.sh %s\n", bag_dir.c_str());
}
