#include "wc_database_exporter.hpp"

#include <chrono>
#include <iomanip>
#include <set>
#include <unistd.h>
#include <rtabmap/core/Features2d.h>
#include <rtabmap/core/Optimizer.h>

#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

// ---------------------------------------------------------------------------
// load_bag_detections
// ---------------------------------------------------------------------------
// Reads AprilTag poses from the bag file alongside the database.
// Convention: db at "foo/jb-KEEP.db" → bag at "foo/jb-KEEP-detections/".
//
// Preferred bag format: record /tf (and optionally /tf_static).
//   The apriltag_ros node publishes "camera_frame → tag_X" transforms on /tf
//   whenever a tag is detected.  We collect those, apply the camera's
//   localTransform from the DB to get a body→tag transform, then match each
//   detection to the nearest DB node by timestamp.
//
// Fallback bag format: record /detections (apriltag_msgs).
//   Tag pose is recovered from the homography + camera K stored in the DB.
//
// All observations are merged into raw_landmark_links so the tag stats and
// re-optimisation code sees the denser data.
// ---------------------------------------------------------------------------
std::multimap<int, std::pair<int, rtabmap::Transform>>
WCDatabaseExporter::load_bag_detections(const rtabmap::ParametersMap & /*parameters*/)
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
    if (s > 1e6)  // sanity: expect Unix timestamps (> year 2001)
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

  // ── 3. Build camera calibration table (localTransform only) ───────────
  // localTransform() = transform from camera optical frame to body frame.
  // Needed to convert the TF camera→tag into body→tag.
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
             ci, cc.name.c_str(),
             cc.local_tf.x(), cc.local_tf.y(), cc.local_tf.z());
    }
    break;  // only need the first node that has calibration
  }

  if (cam_cals.empty()) {
    // No calibration in DB — fall back to identity (body ≡ camera).
    printf("[bag] WARNING: No camera calibration in DB — using identity localTransform\n");
    CamCal cc;
    cc.name     = "identity";
    cc.local_tf = rtabmap::Transform::getIdentity();
    cam_cals.push_back(cc);
  }

  // ── 4. Helper lambdas ──────────────────────────────────────────────────

  // Returns true if frame_id looks like an apriltag child frame.
  // Matches "tag_1", "tag36h11_5", "apriltag_23", etc.
  auto is_tag_frame = [](const std::string &frame) -> bool {
    if (frame.find("tag") == std::string::npos &&
        frame.find("Tag") == std::string::npos) return false;
    for (char c : frame) if (std::isdigit(c)) return true;
    return false;
  };

  // Extracts the last run of digits from a string ("tag36h11_5" → 5).
  // Returns -1 if no digits found.
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

  // Returns the camera model that best matches parent_frame.
  // Strategy 1: substring match on model name (works if DB stores frame names).
  // Strategy 2: if DB names are generic ("cam0"/"cam1"), use "left"/"right"
  //             keywords paired with the camera's tx sign:
  //               left  → most-negative  tx (body-frame left = negative Y or X)
  //               right → most-positive  tx
  // Falls back to cam_cals[0] if nothing matches.
  auto pick_camera = [&](const std::string &parent_frame) -> const CamCal & {
    // Strategy 1: direct name substring match.
    for (const auto &cc : cam_cals)
      if (parent_frame.find(cc.name) != std::string::npos ||
          cc.name.find(parent_frame) != std::string::npos)
        return cc;

    // Strategy 2: left/right keyword → sort cameras by tx, pick extremes.
    bool has_left  = parent_frame.find("left")  != std::string::npos;
    bool has_right = parent_frame.find("right") != std::string::npos;
    if ((has_left || has_right) && cam_cals.size() >= 2) {
      // Find the cam with most-negative tx and most-positive tx.
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

  // ── 6. Read /tf messages and collect tag observations ─────────────────
  // The apriltag_ros node publishes: parent=camera_optical_frame, child=tag_X.
  // The TransformStamped gives the tag's position+orientation in the camera
  // optical frame (i.e. camera→tag).  We apply the camera's localTransform
  // to get body→tag, then match by timestamp to the nearest DB node.
  //
  // kMaxDt = 2 s: RTABMap keyframe rate ~0.47 Hz → one node every ~2 s.
  // Any detection within 2 s of a node can be usefully attributed to it.
  constexpr double kMaxDt   = 0.5;   // max allowed timestamp gap (seconds)
  constexpr double kMinDist = 0.05;  // discard detections closer than 5 cm
  constexpr double kMaxDist = 8.0;   // discard detections farther than 8 m

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

      // Find nearest DB node by timestamp
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

      // TF TransformStamped: translation = tag origin in parent (camera) frame,
      // rotation = orientation of tag in camera frame → this IS camera→tag.
      const auto &tr = ts.transform;
      double tx = tr.translation.x, ty = tr.translation.y, tz = tr.translation.z;
      double dist = std::sqrt(tx*tx + ty*ty + tz*tz);
      if (dist < kMinDist || dist > kMaxDist) { ++n_out_of_range; continue; }

      rtabmap::Transform cam_to_tag(
          (float)tx, (float)ty, (float)tz,
          (float)tr.rotation.x, (float)tr.rotation.y,
          (float)tr.rotation.z, (float)tr.rotation.w);
      if (cam_to_tag.isNull()) { ++n_bad_pose; continue; }

      // body_to_tag = localTransform * cam_to_tag
      // RTABMap kLandmark link.transform() must be in body frame because
      // world_tag = optimizedPoses[node] * link.transform() during re-opt.
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

WCDatabaseExporter::~WCDatabaseExporter() {}

nav_msgs::msg::OccupancyGrid::SharedPtr
WCDatabaseExporter::per_scan_log_odds_grid(float resolution)
{
  // Log-odds parameters — same sign convention as RTABMap's OccupancyGrid.
  constexpr float l_occ       =  0.85f;  // ln(0.7/0.3) — RTABMap default
  constexpr float l_free      = -0.41f;  // ln(0.4/0.6) — RTABMap default
  constexpr float l_min       = -2.0f;
  constexpr float l_max       =  3.5f;
  constexpr float occ_thresh  =  0.5f;
  constexpr float free_thresh = -0.5f;
  constexpr float kMaxRange   =  8.0f;   // m
  // Wall-height band: rays in this z range cast an occupied vote at endpoint.
  // All rays (any z) still mark free along their path.
  constexpr float kWallMinZ   =  0.05f;
  constexpr float kWallMaxZ   =  0.30f;

  if (per_scan_xyz_.empty()) {
    printf("[log_odds_grid] no per-scan data — returning empty grid\n");
    return std::make_shared<nav_msgs::msg::OccupancyGrid>();
  }

  // Grid bounds from all endpoints.
  float xmin = 1e9f, xmax = -1e9f, ymin = 1e9f, ymax = -1e9f;
  for (const auto &[nid, pts] : per_scan_xyz_)
    for (const auto &p : pts) {
      xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
      ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
    }
  constexpr float kPad = 1.0f;
  xmin -= kPad; ymin -= kPad; xmax += kPad; ymax += kPad;

  const int w = (int)((xmax - xmin) / resolution) + 1;
  const int h = (int)((ymax - ymin) / resolution) + 1;
  const int total = w * h;
  std::vector<float> lo(total, 0.0f);

  auto to_grid = [&](float wx, float wy) -> std::pair<int,int> {
    return {(int)((wx - xmin) / resolution),
            (int)((wy - ymin) / resolution)};
  };

  for (const auto &[nid, pts] : per_scan_xyz_) {
    auto oit = per_scan_origin_.find(nid);
    if (oit == per_scan_origin_.end()) continue;
    auto [ox, oy] = to_grid(oit->second.x, oit->second.y);

    for (const auto &ep : pts) {
      float dx = ep.x - oit->second.x, dy = ep.y - oit->second.y;
      if (dx*dx + dy*dy > kMaxRange*kMaxRange) continue;

      auto [ex, ey] = to_grid(ep.x, ep.y);
      if (ex < 0 || ex >= w || ey < 0 || ey >= h) continue;

      // Whether this endpoint should cast an occupied vote (wall-height only).
      const bool is_wall = (ep.z >= kWallMinZ && ep.z <= kWallMaxZ);

      // Bresenham from sensor origin to endpoint.
      // All rays mark free along the path; wall-height endpoints also mark occupied.
      int x = ox, y = oy;
      int ddx = std::abs(ex-x), sx = x<ex?1:-1;
      int ddy = std::abs(ey-y), sy = y<ey?1:-1;
      int err = ddx - ddy;
      for (;;) {
        bool at_end = (x==ex && y==ey);
        if (x>=0 && x<w && y>=0 && y<h) {
          float &v = lo[y*w+x];
          if (at_end && is_wall) v = std::min(v + l_occ,  l_max);
          else if (!at_end)      v = std::max(v + l_free, l_min);
          // non-wall endpoint: ray stops here but no occupied vote
        }
        if (at_end) break;
        int e2 = 2*err;
        if (e2 > -ddy) { err -= ddy; x += sx; }
        if (e2 <  ddx) { err += ddx; y += sy; }
      }
    }
  }

  auto grid = std::make_shared<nav_msgs::msg::OccupancyGrid>();
  grid->info.resolution = resolution;
  grid->info.width  = (uint32_t)w;
  grid->info.height = (uint32_t)h;
  grid->info.origin.position.x = xmin;
  grid->info.origin.position.y = ymin;
  grid->info.origin.position.z = 0;
  grid->info.origin.orientation.w = 1;
  grid->data.resize(total);
  for (int i = 0; i < total; ++i) {
    if      (lo[i] >= occ_thresh)  grid->data[i] = 100;
    else if (lo[i] <= free_thresh) grid->data[i] = 0;
    else                           grid->data[i] = -1;
  }

  printf("[log_odds_grid] %d×%d cells (%.1f×%.1f m) from %zu scans\n",
         w, h, w*resolution, h*resolution, per_scan_xyz_.size());
  return grid;
}

nav_msgs::msg::OccupancyGrid::SharedPtr
WCDatabaseExporter::point_cloud_to_occupancy_grid(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
  const std::vector<std::pair<float,float>> &scan_origins)
{
  // Only keep points in the wall-contact height band.
  constexpr float min_z = 0.05f;
  constexpr float max_z = 0.30f;

  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();

  for (const auto &point : cloud->points) {
    if (point.z < min_z || point.z > max_z) continue;
    if (point.x > max_x) max_x = point.x;
    if (point.y > max_y) max_y = point.y;
    if (point.x < min_x) min_x = point.x;
    if (point.y < min_y) min_y = point.y;
  }

  if (min_x > max_x) {
    std::cout << "point_cloud_to_occupancy_grid: no points in z=["
              << min_z << ", " << max_z << "] — returning empty grid.\n";
    return std::make_shared<nav_msgs::msg::OccupancyGrid>();
  }

  // Pad grid bounds slightly so sensor origins near the edge fall inside.
  constexpr float kPad = 1.0f;  // 1 m margin
  min_x -= kPad; min_y -= kPad;
  max_x += kPad; max_y += kPad;

  nav_msgs::msg::OccupancyGrid::SharedPtr grid =
    std::make_shared<nav_msgs::msg::OccupancyGrid>();
  const float resolution = 0.05f;
  const int w = (int)((max_x - min_x) / resolution) + 1;
  const int h = (int)((max_y - min_y) / resolution) + 1;
  const int total_cells = w * h;

  grid->info.resolution = resolution;
  grid->info.width  = w;
  grid->info.height = h;
  grid->info.origin.position.x = min_x;
  grid->info.origin.position.y = min_y;
  grid->info.origin.position.z = 0;
  grid->info.origin.orientation.w = 1;

  // -1 = UNKNOWN, 0 = FREE, 100 = OCCUPIED
  grid->data.assign(total_cells, -1);

  // ── Step 1: mark occupied cells ───────────────────────────────────────
  constexpr int min_pts_per_cell = 8;
  std::vector<int> cell_counts(total_cells, 0);
  for (const auto &pt : cloud->points) {
    if (pt.z < min_z || pt.z > max_z) continue;
    int gx = (int)((pt.x - min_x) / resolution);
    int gy = (int)((pt.y - min_y) / resolution);
    if (gx < 0 || gx >= w || gy < 0 || gy >= h) continue;
    cell_counts[gy * w + gx]++;
  }
  for (int i = 0; i < total_cells; ++i)
    if (cell_counts[i] >= min_pts_per_cell) grid->data[i] = 100;

  // Morphological open: remove isolated noise cells before raytracing
  // so stray points don't prematurely stop rays.
  {
    cv::Mat mat(h, w, CV_8U);
    for (int i = 0; i < total_cells; ++i)
      mat.data[i] = (grid->data[i] == 100) ? 255 : 0;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mat, mat, cv::MORPH_OPEN, kernel);
    for (int i = 0; i < total_cells; ++i)
      grid->data[i] = (mat.data[i] > 0) ? 100 : -1;
  }

  // ── Step 2: raytrace from sensor origins to mark free space ──────────
  // Subsample origins (every kStride-th pose) and cast kNRays evenly-spaced
  // rays up to kMaxRange.  Cells before the first OCCUPIED hit are FREE (0);
  // cells never reached stay UNKNOWN (-1).
  // Complexity target: ~8M iterations (fast enough for the destructor).
  constexpr int   kStride   = 10;    // use every 10th robot pose as an origin
  constexpr int   kNRays    = 120;   // ray directions (3° spacing)
  constexpr float kMaxRange = 8.0f;  // m
  constexpr float kStepSize = 0.025f;// m — half a cell, guarantees no cell skipping
  const int kMaxSteps = (int)(kMaxRange / kStepSize) + 1;

  int n_origins_used = 0;
  for (int oi = 0; oi < (int)scan_origins.size(); oi += kStride) {
    const auto &[ox, oy] = scan_origins[oi];
    ++n_origins_used;
    for (int ri = 0; ri < kNRays; ++ri) {
      float angle = ri * (float)(2.0 * M_PI / kNRays);
      float dx = std::cos(angle) * kStepSize;
      float dy = std::sin(angle) * kStepSize;
      float cx = ox, cy = oy;
      for (int step = 0; step < kMaxSteps; ++step) {
        cx += dx; cy += dy;
        int gx = (int)((cx - min_x) / resolution);
        int gy = (int)((cy - min_y) / resolution);
        if (gx < 0 || gx >= w || gy < 0 || gy >= h) break;
        int idx = gy * w + gx;
        if (grid->data[idx] == 100) break;  // hit wall — stop
        grid->data[idx] = 0;                // confirmed free
      }
    }
  }

  printf("[grid] Raytracing: %d origins (stride %d), %d rays, %.0f m range → done\n",
         n_origins_used, kStride, kNRays, kMaxRange);

  printf("[grid] Grid: %d×%d cells (%.1f×%.1f m) at %.2f m/cell\n",
         w, h, w * resolution, h * resolution, resolution);

  return grid;
}

// @brief : This function takes in a point cloud and a camera transform and
// projects the point cloud to the camera frame. The sequence of
// filters was determined by trial and error
// @param cloud: The point cloud to filter
// @return The filtered point cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr WCDatabaseExporter::filter_point_cloud(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
//   auto start = std::chrono::high_resolution_clock::now();

//   // Step 1: Remove invalid points
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cleaned_cloud(
//     new pcl::PointCloud<pcl::PointXYZRGB>);

//   cleaned_cloud->points.reserve(cloud->points.size());

//   for (const auto &pt : cloud->points) {
//     if (pcl::isFinite(pt) &&
//         std::abs(pt.x) <= 1e5 &&
//         std::abs(pt.y) <= 1e5 &&
//         std::abs(pt.z) <= 1e5) {
//       cleaned_cloud->points.push_back(pt);
//     }
//   }

//   cleaned_cloud->width = cleaned_cloud->points.size();
//   cleaned_cloud->height = 1;
//   cleaned_cloud->is_dense = true;

//   std::cout << "After invalid removal: "
//             << cleaned_cloud->size()
//             << " points" << std::endl;


//   // Step 2: Downsample
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxel_cloud(
//     new pcl::PointCloud<pcl::PointXYZRGB>);

//   pcl::VoxelGrid<pcl::PointXYZRGB> voxel;
//   voxel.setInputCloud(cleaned_cloud);

//   // Smaller voxel to preserve more detail
//   voxel.setLeafSize(
//     0.01f,  // x
//     0.01f,  // y
//     0.01f   // z
//   );

//   voxel.filter(*voxel_cloud);

//   std::cout << "After voxel filtering: "
//             << voxel_cloud->size()
//             << " points" << std::endl;


//   // Step 3: Statistical outlier removal
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr sor_cloud(
//     new pcl::PointCloud<pcl::PointXYZRGB>);

//   pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
//   sor.setInputCloud(voxel_cloud);

//   sor.setMeanK(50);
//   sor.setStddevMulThresh(1.0);

//   sor.filter(*sor_cloud);

//   sor_cloud->width = sor_cloud->points.size();
//   sor_cloud->height = 1;
//   sor_cloud->is_dense = true;

//   std::cout << "After SOR: "
//             << sor_cloud->size()
//             << " points" << std::endl;


//   return sor_cloud;
// }
// pcl::PointCloud<pcl::PointXYZRGB>::Ptr WCDatabaseExporter::filter_point_cloud(
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
// {

  // 5 cm voxel — brings 170M+ depth clouds down to ~5M without SOR/radius.
  // SOR on 80M points (what 2 cm left) OOMs via KD-tree; voxel alone is
  // sufficient cleanup for the 3D PCD export since the grid uses scan_cloud_.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxel_cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);
  {
    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.05f, 0.05f, 0.05f);
    vg.filter(*voxel_cloud);
    std::cout << "After voxel (5 cm): " << cloud->size()
              << " → " << voxel_cloud->size() << " points\n";
  }

  return voxel_cloud;
}

bool WCDatabaseExporter::initialize_rtabmap_database()
{
  rtabmap::ParametersMap parameters;
  rtabmap::DBDriver *driver = rtabmap::DBDriver::create();

  std::map<int, rtabmap::Transform> odomPoses;
  std::multimap<int, rtabmap::Link> raw_landmark_links;  // physical marker IDs (to_id = -markerID)
  if (driver->openConnection(rtabmap_database_path_)) {
    parameters = driver->getLastParameters();
    driver->getAllOdomPoses(odomPoses);
    std::multimap<int, rtabmap::Link> all_raw_links;
    driver->getAllLinks(all_raw_links, true, true);  // withLandmarks=true
    for (const auto &kv : all_raw_links)
      if (kv.second.type() == rtabmap::Link::kLandmark)
        raw_landmark_links.insert(kv);
    driver->closeConnection(false);
  } else {
    std::cout << "Failed to open database" << std::endl;
    return false;
  }
  delete driver;
  driver = 0;

  // Print all Marker parameters so we can verify tag size matches physical tags.
  std::cout << "\n--- Marker parameters from DB ---\n";
  for (const auto &[key, val] : parameters) {
    if (key.find("Marker/") != std::string::npos) {
      std::cout << "  " << key << " = " << val << "\n";
    }
  }
  std::cout << "---------------------------------\n\n";

  UTimer timer;

  std::cout << "Loading database: " << rtabmap_database_path_ << std::endl;
  rtabmap::Rtabmap rtabmap;
  rtabmap.init(parameters, rtabmap_database_path_);
  std::cout << "Loaded database in " << timer.ticks() << "s" << std::endl;

  std::cout << "Optimizing the map..." << std::endl;
  rtabmap.getGraph(optimizedPoses, links, true, true, &nodes, true, true, true,
                   true);
  std::cout << "Optimizing the map... done (" << timer.ticks()
            << "s, poses=" << optimizedPoses.size() << ")." << std::endl;

  if (optimizedPoses.size() == 0) {
    std::cout << "No optimized poses found" << std::endl;
    return false;
  }

  // ── Augment raw_landmark_links with bag detections (if bag is present) ──
  // The bag contains every AprilTag detection the live detector saw,
  // including ones RTABMap rejected because they conflicted with the
  // (potentially drifted) map at that moment.  Denser observations give a
  // better consensus mean and std, which improves the adaptive prior weights.
  {
    auto bag_obs = load_bag_detections(parameters);
    if (!bag_obs.empty()) {
      // Count DB-only observations before merging so we can report the delta.
      int n_db_before = (int)raw_landmark_links.size();

      // Create a placeholder info-matrix (will be overridden in base_links).
      cv::Mat dummy_inf = cv::Mat::eye(6, 6, CV_64FC1);
      for (const auto &[node_id, id_tf] : bag_obs) {
        auto [tag_id, cam_to_tag] = id_tf;
        // kLandmark to_id convention: negative physical marker ID
        rtabmap::Link lk(node_id, -tag_id,
                         rtabmap::Link::kLandmark, cam_to_tag, dummy_inf);
        raw_landmark_links.insert({node_id, lk});
      }
      printf("[bag] raw_landmark_links: %d (DB) + %d (bag) = %d total\n",
             n_db_before, (int)bag_obs.size(), (int)raw_landmark_links.size());
    } else {
      printf("[bag] No bag observations loaded — using DB detections only (%zu)\n",
             raw_landmark_links.size());
    }
  }

  // When true, re-SLAM uses AprilTag-constrained optimized poses as its
  // starting point. When false, re-SLAM runs from raw odometry so AprilTag
  // distortions have no influence — loop closures found during re-SLAM
  // become the sole constraints in the final map.
  constexpr bool use_april_tags = true;

  auto count_links_by_type = [](const std::multimap<int, rtabmap::Link> &lks,
                                rtabmap::Link::Type type) -> int {
    return (int)std::count_if(lks.begin(), lks.end(),
      [type](const auto &kv) { return kv.second.type() == type; });
  };
  auto print_link_counts = [&](const std::string &label,
                                const std::multimap<int, rtabmap::Link> &lks) {
    std::cout << label << "\n"
              << "  Global closures (BoW):  "
              << count_links_by_type(lks, rtabmap::Link::kGlobalClosure) << "\n"
              << "  Local space closures:   "
              << count_links_by_type(lks, rtabmap::Link::kLocalSpaceClosure) << "\n"
              << "  Landmark (AprilTag):    "
              << count_links_by_type(lks, rtabmap::Link::kLandmark) << "\n"
              << "  User closures:          "
              << count_links_by_type(lks, rtabmap::Link::kUserClosure) << "\n";
  };
  print_link_counts("Original DB link counts:", links);

  // List every non-neighbor closure with its rotation angle so we can spot
  // any single closure that caused a large map rotation.
  {
    printf("\nLoop closure details (from→to, Δtrans m, Δrot °):\n");
    int n_printed = 0;
    for (const auto &kv : links) {
      const rtabmap::Link &lk = kv.second;
      if (lk.type() == rtabmap::Link::kNeighbor ||
          lk.type() == rtabmap::Link::kLandmark  ||
          lk.type() == rtabmap::Link::kPosePrior)
        continue;
      const rtabmap::Transform &t = lk.transform();
      float tx = t.x(), ty = t.y(), tz = t.z();
      float dist = std::sqrt(tx*tx + ty*ty + tz*tz);
      // Extract yaw angle from rotation matrix
      Eigen::Matrix3f R = t.toEigen3f().linear();
      float yaw_deg = std::atan2(R(1,0), R(0,0)) * 180.f / M_PI;
      const char *type_str =
        lk.type() == rtabmap::Link::kGlobalClosure     ? "Global" :
        lk.type() == rtabmap::Link::kLocalSpaceClosure ? "LocalSpace" :
        lk.type() == rtabmap::Link::kUserClosure       ? "User" : "Other";
      printf("  [%s] %d → %d : Δtrans=%.3f m, Δyaw=%.1f°\n",
             type_str, lk.from(), lk.to(), dist, yaw_deg);
      ++n_printed;
    }
    if (n_printed == 0) printf("  (none)\n");
    printf("\n");
  }

  // Print odometry link variance so we know what the priors are competing against.
  {
    double sum_trans_var = 0, sum_rot_var = 0; int n = 0;
    for (const auto &kv : links) {
      if (kv.second.type() != rtabmap::Link::kNeighbor) continue;
      const cv::Mat &inf = kv.second.infMatrix();
      if (inf.empty()) continue;
      sum_trans_var += 1.0 / inf.at<double>(0,0);
      sum_rot_var   += 1.0 / inf.at<double>(3,3);
      ++n;
    }
    if (n > 0)
      printf("Odometry links (%d): mean trans_var=%.6f (σ=%.4f m), "
             "mean rot_var=%.6f (σ=%.4f rad)\n\n",
             n, sum_trans_var/n, std::sqrt(sum_trans_var/n),
             sum_rot_var/n,   std::sqrt(sum_rot_var/n));
  }

  // Per-tag position statistics from the original optimized poses.
  {
    struct Obs { float x, y, z; };
    std::map<int, std::vector<Obs>> tag_obs;  // key = physical marker ID
    for (const auto &kv : raw_landmark_links) {
      const rtabmap::Link &lk = kv.second;
      auto it = optimizedPoses.find(lk.from());
      if (it == optimizedPoses.end()) continue;
      rtabmap::Transform world_tag = it->second * lk.transform();
      tag_obs[std::abs(lk.to())].push_back({world_tag.x(), world_tag.y(), world_tag.z()});
    }

    // Store for destructor overlay onto occupancy grid PNG.
    for (auto &[tid, obs] : tag_obs)
      for (auto &o : obs)
        tag_obs_[tid].push_back({o.x, o.y, o.z});

    printf("\n--- AprilTag world positions (from optimized poses) ---\n");
    printf("  %4s  %5s  %8s  %8s  %8s  %7s  %7s  %7s  %8s\n",
           "Tag", "N", "MeanX", "MeanY", "MeanZ", "StdX", "StdY", "StdZ", "StdXYZ");
    for (auto &[tag_id, obs] : tag_obs) {
      int n = obs.size();
      float mx = 0, my = 0, mz = 0;
      for (auto &o : obs) { mx += o.x; my += o.y; mz += o.z; }
      mx /= n; my /= n; mz /= n;
      float sx = 0, sy = 0, sz = 0;
      for (auto &o : obs) {
        sx += (o.x-mx)*(o.x-mx);
        sy += (o.y-my)*(o.y-my);
        sz += (o.z-mz)*(o.z-mz);
      }
      sx = std::sqrt(sx/n); sy = std::sqrt(sy/n); sz = std::sqrt(sz/n);
      float rms = std::sqrt((sx*sx + sy*sy + sz*sz) / 3.0f);
      printf("  %4d  %5d  %8.3f  %8.3f  %8.3f  %7.4f  %7.4f  %7.4f  %8.4f\n",
             tag_id, n, mx, my, mz, sx, sy, sz, rms);
    }
    printf("-------------------------------------------------------\n\n");
  }

  // Re-optimize the pose graph using raw landmark factors.
  //
  // Strategy: add all kLandmark links (DB + bag) directly to the factor graph
  // so GTSAM couples every pair of nodes that observed the same tag.  A tag
  // seen before AND after a rotation event forces the optimizer to reconcile
  // those two viewpoints — equivalent to a virtual loop closure through the tag.
  //
  // Each tag node (-tid) gets a kPosePrior at the consensus mean to prevent
  // GTSAM indeterminacy (tags with only one observer would be unconstrained).
  // The prior variance scales with the tag's observation std so noisy tags
  // anchor loosely; tight tags anchor firmly.
  constexpr double landmark_var_linear  = 0.005;  // m²  σ ≈ 7 cm
  constexpr double landmark_var_angular = 0.05;   // rad² looser — tag yaw is noisier
  constexpr double prior_var_linear_base  = 0.01; // m²  base tag-node prior (σ ≈ 10 cm)
  constexpr double prior_var_angular_base = 0.01; // rad²
  constexpr int    n_reopt_passes         = 3;    // iterate: better poses → better tag means
  {
    // base_links: odometry + loop closures only.
    // kLandmark links are added per-pass from raw_landmark_links so the bag
    // observations are included and weighting is consistent.
    std::multimap<int, rtabmap::Link> base_links;
    for (const auto &kv : links) {
      if (kv.second.type() != rtabmap::Link::kLandmark)
        base_links.insert(kv);
    }

    struct TagAccum {
      float tx=0, ty=0, tz=0, qx=0, qy=0, qz=0, qw=0; int n=0;
      // For std computation — accumulated after mean is known (two-pass).
      // Stored as observation list for simplicity.
      std::vector<std::array<float,3>> pts;
    };

    for (int pass = 1; pass <= n_reopt_passes; ++pass) {
      std::cout << "\n--- Re-optimization pass " << pass
                << "/" << n_reopt_passes << " ---\n";

      // Recompute consensus tag world pose from current optimizedPoses.
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
        if (a.n > 0 &&
            (q.x()*a.qx + q.y()*a.qy + q.z()*a.qz + q.w()*a.qw) < 0.f)
          q.coeffs() = -q.coeffs();
        a.qx += q.x(); a.qy += q.y(); a.qz += q.z(); a.qw += q.w();
        a.pts.push_back({wt.x(), wt.y(), wt.z()});
        a.n++;
      }

      // Build consensus transforms and compute per-tag std for filtering.
      std::map<int, rtabmap::Transform> tag_world_mean;
      std::map<int, float> tag_std;      // XY RMS std (threshold filter)
      std::map<int, float> tag_std_xyz;  // full XYZ RMS std (before/after comparison)
      for (auto &[tid, a] : tag_accum) {
        float inv  = 1.0f / a.n;
        float norm = std::sqrt(a.qx*a.qx + a.qy*a.qy + a.qz*a.qz + a.qw*a.qw);
        float mx = a.tx*inv, my = a.ty*inv, mz = a.tz*inv;
        tag_world_mean[tid] = rtabmap::Transform(
            mx, my, mz,
            a.qx/norm, a.qy/norm, a.qz/norm, a.qw/norm);
        float sx = 0, sy = 0, sz = 0;
        for (auto &p : a.pts) {
          sx += (p[0]-mx)*(p[0]-mx);
          sy += (p[1]-my)*(p[1]-my);
          sz += (p[2]-mz)*(p[2]-mz);
        }
        tag_std[tid]     = std::sqrt((sx + sy) / (2.0f * a.n));
        tag_std_xyz[tid] = std::sqrt((sx + sy + sz) / (3.0f * a.n));
      }

      // Tags with high position spread are unreliable anchors — exclude them
      // from generating priors entirely.  0.20 m is roughly 2× the parallax
      // std of a tag viewed from a moving robot; anything above that indicates
      // either a bad detection geometry or too few observations.
      constexpr float kMaxTagStd = 0.20f;  // m — tags above this are skipped

      // Print per-tag std and resulting tag-node prior variance.
      std::cout << "  Tag node priors (std → var_linear):\n  ";
      for (const auto &[tid, s] : tag_std) {
        if (s > kMaxTagStd) {
          std::cout << tid << "(σ=" << std::fixed << std::setprecision(3)
                    << s << "→SKIP) ";
          continue;
        }
        double v = prior_var_linear_base + (double)(s * s);
        std::cout << tid << "(σ=" << std::fixed << std::setprecision(3)
                  << s << "→v=" << v << ") ";
      }
      std::cout << "\n";

      // Build reopt_links: base (odometry+closures) + all raw landmark links
      // + one kPosePrior per tag node to prevent GTSAM indeterminacy.
      std::multimap<int, rtabmap::Link> reopt_links = base_links;

      // Shared information matrix for all landmark links.
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

      // Anchor each tag node with a prior at its consensus world position.
      // Variance = base + std²: tight tags anchor firmly, noisy tags loosely.
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
      printf("  %d landmark links (%d skipped high-std), %d tag node priors\n",
             n_lm_added, n_lm_skipped, n_tag_priors);

      // Initial poses for the optimizer include both robot nodes and tag nodes.
      std::map<int, rtabmap::Transform> initial_poses = optimizedPoses;
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
        // Measure change across all robot nodes (positive IDs only).
        double sum_trans = 0, sum_rot = 0; int n_moved = 0;
        std::vector<double> dt_vals, dr_vals;
        for (const auto &[nid, prev_pose] : optimizedPoses) {
          if (nid < 0) continue;
          auto next_it = reoptPoses.find(nid);
          if (next_it == reoptPoses.end()) continue;
          rtabmap::Transform delta = prev_pose.inverse() * next_it->second;
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
            var_t += (dt_vals[k] - mean_t) * (dt_vals[k] - mean_t);
            var_r += (dr_vals[k] - mean_r) * (dr_vals[k] - mean_r);
          }
          printf("  Landmark effect on %d robot nodes: "
                 "mean Δtrans=%.4f m (σ=%.4f), mean Δrot=%.2f deg (σ=%.2f)\n",
                 n_moved, mean_t, std::sqrt(var_t / n_moved),
                 mean_r, std::sqrt(var_r / n_moved));
        }
        printf("  Pass %d done: %zu total poses (%zu robot)\n",
               pass, reoptPoses.size(),
               (size_t)std::count_if(reoptPoses.begin(), reoptPoses.end(),
                   [](const auto &kv){ return kv.first > 0; }));

        // Update only robot node poses; tag positions are recomputed next pass.
        for (const auto &[nid, pose] : reoptPoses)
          if (nid > 0) optimizedPoses[nid] = pose;
        links = reopt_links;

        // Recompute tag spread on updated poses to measure landmark effect.
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
        printf("  Post-reopt tag StdXYZ (before → after):\n");
        printf("  %5s %5s  %8s  %8s  %8s\n", "Tag", "N", "Before", "After", "Delta");
        float sum_delta = 0.0f; int n_shown = 0;
        for (auto &[tid, a] : post_accum) {
          float inv = 1.0f / a.n;
          float mx = a.tx*inv, my = a.ty*inv, mz = a.tz*inv;
          float sx = 0, sy = 0, sz = 0;
          for (auto &p : a.pts) {
            sx += (p[0]-mx)*(p[0]-mx);
            sy += (p[1]-my)*(p[1]-my);
            sz += (p[2]-mz)*(p[2]-mz);
          }
          float after  = std::sqrt((sx + sy + sz) / (3.0f * a.n));
          float before = tag_std_xyz.count(tid) ? tag_std_xyz.at(tid) : after;
          printf("  %5d %5d  %8.4f  %8.4f  %+8.4f\n", tid, a.n, before, after, after - before);
          sum_delta += after - before;
          ++n_shown;
        }
        if (n_shown > 0)
          printf("  Mean StdXYZ change across %d tags: %+.4f m (%s)\n",
                 n_shown, sum_delta / n_shown,
                 sum_delta / n_shown < 0.0f ? "improved" : "worsened");
      } else {
        std::cerr << "  Pass " << pass << " failed, keeping previous poses\n";
        break;
      }
    }
  }

  // Snapshot post-reopt camera poses for the destructor overlay.
  viz_poses_ = optimizedPoses;

  // Recompute tag overlay positions from post-GTSAM poses so the tag markers
  // are consistent with the scan cloud (also built from post-GTSAM poses).
  tag_obs_.clear();
  for (const auto &kv : raw_landmark_links) {
    const rtabmap::Link &lk = kv.second;
    auto it = optimizedPoses.find(lk.from());
    if (it == optimizedPoses.end()) continue;
    rtabmap::Transform world_tag = it->second * lk.transform();
    tag_obs_[std::abs(lk.to())].push_back(
        {world_tag.x(), world_tag.y(), world_tag.z()});
  }

  std::string model_path = "/app/nn-model/24-core-unet/unet-24.pt";
  try {
    unet_model_ = torch::jit::load(model_path);
    unet_model_.to(torch::Device(torch::kCUDA));
    unet_model_.eval();
    unet_loaded_ = true;
    std::cout << "U-Net model loaded from " << model_path << std::endl;
  } catch (const c10::Error &e) {
    std::cerr << "Failed to load U-Net model: " << e.what() << std::endl;
  }
  
  // Toggle: set false to skip re-SLAM and run ICP on the original optimized poses.
  // Useful for diagnosing whether re-SLAM or ICP is the source of map corruption.
  constexpr bool enable_reslam = false;

  if (enable_reslam && unet_loaded_) {

    std::cout << "Re-running SLAM with U-Net cleaned images..." << std::endl;

    // Override memory parameters for offline re-SLAM.
    // Key insight: use optimized poses (not raw odom) so the proximity search
    // operates in corrected world-space — nodes that are physically nearby stay
    // nearby in pose-space even after odometry drift.
    rtabmap::ParametersMap reslam_params = parameters;
    reslam_params[rtabmap::Parameters::kRtabmapDetectionRate()]      = "0";  // unlimited WM
    reslam_params[rtabmap::Parameters::kRtabmapMaxRetrieved()]      = "500";  // unlimited WM
    reslam_params[rtabmap::Parameters::kKpMaxFeatures()]      = "4000";  // unlimited WM
    reslam_params[rtabmap::Parameters::kRtabmapMemoryThr()]      = "0";  // unlimited WM
    reslam_params[rtabmap::Parameters::kMemSTMSize()]            = "30";  // bypass STM
    reslam_params[rtabmap::Parameters::kRGBDLocalRadius()]       = "20"; // 20m radius
    // Proximity detection: how far (metres) to search for nearby-pose candidates
    reslam_params[rtabmap::Parameters::kRGBDProximityMaxGraphDepth()] = "0";   // unlimited graph depth
    reslam_params[rtabmap::Parameters::kRGBDProximityPathMaxNeighbors()] = "20"; // neighbours per path
    reslam_params[rtabmap::Parameters::kRGBDNeighborLinkRefining()]  = "true"; // refine sequential links
    reslam_params[rtabmap::Parameters::kVisMinInliers()]              = "30";
    reslam_params[rtabmap::Parameters::kVisMaxFeatures()]             = "4000";
    reslam_params[rtabmap::Parameters::kVisCorNNDR()]                 = "0.8";
    reslam_params[rtabmap::Parameters::kMemInitWMWithAllNodes()]      = "true";
    reslam_params[rtabmap::Parameters::kRGBDProximityPathRawPosesUsed()] = "false";
    reslam_params[rtabmap::Parameters::kRtabmapLoopThr()]             = "0.6";   // moderate: fires on strong visual matches with cleaned images
    reslam_params[rtabmap::Parameters::kRGBDProximityBySpace()]       = "false"; // local space closures caused map corruption — BoW only
    reslam_params[rtabmap::Parameters::kOptimizerRobust()]            = "true";
    reslam_params[rtabmap::Parameters::kRGBDOptimizeMaxError()]       = "2.0";
    // Looser than live (0.05) so tags guide the graph without over-constraining.
    reslam_params[rtabmap::Parameters::kMarkerVarianceLinear()]       = "0.5";
    reslam_params[rtabmap::Parameters::kMarkerVarianceAngular()]      = "2.0";
    

    rtabmap::Feature2D *detector = rtabmap::Feature2D::create(reslam_params);

    int total_raw = 0, total_clean = 0, compared = 0;
    double total_unet_ms = 0.0;
    std::cout << "\n" << std::setw(8)  << "Node"
              << std::setw(8)  << "Raw"
              << std::setw(10) << "Cleaned"
              << std::setw(8)  << "Delta"
              << std::setw(12) << "UNet(ms)" << "\n";

    rtabmap::Rtabmap rtabmap2;
    rtabmap2.init(reslam_params);

    // std::cout << "Re-SLAM parameters:\n";
    // for (const auto &kv : rtabmap2.getParameters()) {
    //   std::cout << "  " << kv.first << ": " << kv.second << "\n";
    // }

    for (auto &[id, node] : nodes) {
      if (node.getWeight() == -1) continue;

      // Use the optimized pose from the first SLAM pass (already constrained by
      // the original 6 loop closures) so that the re-SLAM proximity search
      // operates in corrected world-space rather than drifted odom-space.
      // Fall back to raw odometry only if the node wasn't in the optimized graph.
      auto opt_it  = optimizedPoses.find(id);
      auto odom_it = odomPoses.find(id);
      if (opt_it == optimizedPoses.end() && odom_it == odomPoses.end()) continue;
      const rtabmap::Transform &reslam_pose =
        (use_april_tags && opt_it != optimizedPoses.end()) ? opt_it->second
        : (odom_it != odomPoses.end())                     ? odom_it->second
                                                           : opt_it->second;

      cv::Mat rgb;
      cv::Mat depth;
      rtabmap::LaserScan scan;
      node.sensorData().uncompressData(&rgb, &depth, &scan);
      if (rgb.empty()) continue;

      int half_w = rgb.cols / 2;
      cv::Mat left  = rgb(cv::Rect(0,      0, half_w, rgb.rows)).clone();
      cv::Mat right = rgb(cv::Rect(half_w, 0, half_w, rgb.rows)).clone();

      auto t0 = std::chrono::steady_clock::now();
      cv::Mat cleaned_left  = run_unet(left);
      cv::Mat cleaned_right = run_unet(right);
      double unet_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
      total_unet_ms += unet_ms;

      // --- Feature count comparison on the left camera half ---
      std::vector<cv::KeyPoint> kpts_raw   = detector->generateKeypoints(left,         cv::Mat());
      std::vector<cv::KeyPoint> kpts_clean = detector->generateKeypoints(cleaned_left, cv::Mat());
      int n_raw   = (int)kpts_raw.size();
      int n_clean = (int)kpts_clean.size();
      total_raw   += n_raw;
      total_clean += n_clean;
      ++compared;

      std::cout << std::setw(8)  << id
                << std::setw(8)  << n_raw
                << std::setw(10) << n_clean
                << std::setw(8)  << (n_clean - n_raw)
                << std::setw(11) << std::fixed << std::setprecision(1) << unet_ms << "ms\n";

      // --- Re-SLAM with cleaned combined image ---
      cv::Mat cleaned_combined;
      cv::hconcat(cleaned_left, cleaned_right, cleaned_combined);
      cleaned_combined_cache_[id] = cleaned_combined;

      rtabmap::SensorData sensor_data(
        scan, cleaned_combined, depth,
        node.sensorData().cameraModels(), id, node.getStamp());

      rtabmap2.process(sensor_data, reslam_pose);
    }

    delete detector;

    if (compared > 0) {
      double avg_unet_ms   = total_unet_ms / compared;
      double avg_unet_per  = avg_unet_ms / 2.0; // two run_unet calls per frame
      std::cout << std::fixed << std::setprecision(1)
                << "\n--- Feature count summary (left camera) ---\n"
                << "  Frames compared: " << compared << "\n"
                << "  Avg raw:         " << (total_raw   / compared) << " kpts\n"
                << "  Avg cleaned:     " << (total_clean / compared) << " kpts\n"
                << "  Avg delta:       " << ((total_clean - total_raw) / compared) << " kpts\n"
                << "\n--- U-Net timing (both cameras per frame) ---\n"
                << "  Total time:      " << total_unet_ms     << " ms\n"
                << "  Avg per frame:   " << avg_unet_ms       << " ms  (left + right)\n"
                << "  Avg per image:   " << avg_unet_per      << " ms\n"
                << "  Throughput:      " << std::setprecision(1)
                                         << (1000.0 / avg_unet_per) << " images/s\n";
    }

    rtabmap2.getGraph(optimizedPoses, links, true, true, nullptr,
                      false, false, false, false);

    std::cout << "\n";
    print_link_counts("Re-SLAM (U-Net) link counts:", links);
    std::cout << "\n";

    std::cout << "Re-SLAM complete: " << optimizedPoses.size() << " poses\n";
  }


  return true;
}

cv::Mat WCDatabaseExporter::run_unet(const cv::Mat &gray_img)
{
  cv::Mat img_padded;
  cv::copyMakeBorder(gray_img, img_padded, 4, 4, 0, 0, cv::BORDER_REFLECT);

  img_padded.convertTo(img_padded, CV_32F, 1.0 / 255.0);

  // torch::Tensor tensor =
  //   torch::from_blob(img_padded.data, {img_padded.rows, img_padded.cols},
  //                    torch::kFloat32)
  //     .clone();
  // tensor = tensor.unsqueeze(0).unsqueeze(0);

  // torch::Tensor output;
  // {
  //   torch::NoGradGuard no_grad;
  //   std::vector<torch::jit::IValue> inputs;
  //   inputs.push_back(tensor);
  //   output = unet_model_.forward(inputs).toTensor();
  // }

  // output = output.squeeze(0).squeeze(0);
  // output = output.slice(0, 4, 364);

  torch::Tensor tensor =
      torch::from_blob(img_padded.data,
                      {img_padded.rows, img_padded.cols},
                      torch::kFloat32)
          .clone()
          .unsqueeze(0)
          .unsqueeze(0)
          .to(torch::kCUDA);

  torch::Tensor output;

  {
    torch::NoGradGuard no_grad;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);

    output = unet_model_.forward(inputs).toTensor();
  }

  output = output.squeeze(0).squeeze(0);
  output = output.slice(0, 4, 364);

  output = output.to(torch::kCPU);

  cv::Mat output_img(output.size(0), output.size(1), CV_32F,
                     output.data_ptr<float>());
  output_img = output_img.clone();
  output_img.convertTo(output_img, CV_8U, 255.0);
  return output_img;
}

void WCDatabaseExporter::assembleSceneFromOptimizedPoses()
{
  UTimer timer;
  for (std::map<int, rtabmap::Transform>::iterator iter =
         optimizedPoses.lower_bound(1);
       iter != optimizedPoses.end(); ++iter) {

    auto node_it = nodes.find(iter->first);
    if (node_it == nodes.end()) {
      std::cerr << "assembleSceneFromOptimizedPoses: node " << iter->first
                << " not found in nodes map — skipping\n";
      continue;
    }
    rtabmap::Signature node = node_it->second;

    // Get camera model
    std::vector<rtabmap::CameraModel> models = node.sensorData().cameraModels();

    // Uncompress RGB + depth
    cv::Mat rgb, depth;
    if (node.getWeight() != -1) {
      node.sensorData().uncompressData(&rgb, &depth);
    }

    // Store images and calibration
    depth_images[iter->first] = depth;
    rgb_images[iter->first] = rgb;
    camera_models_.push_back(models);

    // Build dense XYZRGB point cloud from RealSense depth image
    if (!depth.empty() && !rgb.empty()) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
        rtabmap::util3d::cloudRGBFromSensorData(
          node.sensorData(),
          4,      // decimation 4 = 16× fewer points; prevents OOM on 170M+ assemblies
          10.0f,  // max depth (m)
          0.1f);  // min depth (m)

      if (cloud && !cloud->empty()) {
        cloud = rtabmap::util3d::transformPointCloud(cloud, iter->second);
        *rtabmap_cloud_ += *cloud;
        rawViewpointIndices.resize(rtabmap_cloud_->size(), iter->first);
      }
    } else {
      printf("Node %d has no depth/RGB — skipping cloud contribution.\n",
             iter->first);
    }

    // Build scan-based cloud for the 2D occupancy grid.
    // Lidar points are cleaner than depth images for wall detection.
    {
      rtabmap::LaserScan scan;
      node.sensorData().uncompressData(nullptr, nullptr, &scan);
      if (!scan.isEmpty()) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr scan_pts =
          rtabmap::util3d::laserScanToPointCloud(scan, scan.localTransform());
        scan_pts = rtabmap::util3d::transformPointCloud(scan_pts, iter->second);

        // Sample z range from first node to diagnose frame alignment.
        static bool z_printed = false;
        if (!z_printed && !scan_pts->empty()) {
          float zmin = std::numeric_limits<float>::max();
          float zmax = -std::numeric_limits<float>::max();
          for (const auto &pt : scan_pts->points) {
            if (!pcl::isFinite(pt)) continue;
            zmin = std::min(zmin, pt.z);
            zmax = std::max(zmax, pt.z);
          }
          printf("Scan z range (world frame, first node): %.3f to %.3f m\n", zmin, zmax);
          z_printed = true;
        }

        // Lidar sensor origin in world frame (for log-odds raytracing).
        rtabmap::Transform sensor_world = iter->second * scan.localTransform();
        per_scan_origin_[iter->first] = {sensor_world.x(), sensor_world.y()};

        auto &scan_xyz = per_scan_xyz_[iter->first];
        for (const auto &pt : scan_pts->points) {
          if (!pcl::isFinite(pt)) continue;
          if (pt.z < 0.05f || pt.z > 1.5f) continue;
          // All points in [0.05, 1.5m]: used for free-space rays.
          // Occupied vote only cast for wall-height band in log_odds_grid.
          scan_xyz.emplace_back(pt.x, pt.y, pt.z);
          // scan_cloud_ used for PCA and 3D export (keep existing z filter).
          if (pt.z >= 0.1f) {
            pcl::PointXYZRGB p;
            p.x = pt.x; p.y = pt.y; p.z = 0.15f;
            p.r = 255; p.g = 255; p.b = 255;
            scan_cloud_->push_back(p);
          }
        }
      }
    }

    // Use camera viewpoint for normal orientation (not lidar)
    rtabmap::Transform cameraViewpoint =
      models.empty() ? iter->second : iter->second * models[0].localTransform();
    rawViewpoints.insert(std::make_pair(iter->first, cameraViewpoint));
    robotPoses.insert(std::make_pair(iter->first, iter->second));
    cameraStamps.insert(std::make_pair(iter->first, node.getStamp()));

    cameraModels.insert(std::make_pair(iter->first, models));
    if (cameraPoses.empty()) {
      cameraPoses.resize(models.size());
    }

    for (size_t i = 0; i < models.size(); ++i) {
      cameraPoses[i].insert(
        std::make_pair(iter->first, iter->second * models[i].localTransform()));
    }

    if (depth.type() == CV_16UC1 || depth.type() == CV_32FC1) {
      cameraDepths.insert(std::make_pair(iter->first, depth));
    }

    std::cout << "rtabmap_cloud_: " << rtabmap_cloud_->size() << " points\n";
  }

  std::cout << "Assembled depth cloud: " << rtabmap_cloud_->size()
            << " points (" << timer.ticks() << "s)\n";
}

void WCDatabaseExporter::projectAndColorizePointCloud()
{
  UTimer timer;

  if (rtabmap_cloud_->empty()) {
    std::cout << "projectAndColorizePointCloud: cloud is empty, skipping.\n";
    return;
  }

  // Extract XYZ for normal computation
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(*rtabmap_cloud_, *cloudXYZ);

  pcl::PointCloud<pcl::Normal>::Ptr normals =
    rtabmap::util3d::computeNormals(cloudXYZ, 20, 0);

  UASSERT(rtabmap_cloud_->size() == normals->size());
  std::cout << "Computing normals... done (" << timer.ticks() << "s, "
            << rtabmap_cloud_->size() << " points)\n";

  // Combine XYZRGB + normals into cloudToExport
  pcl::concatenateFields(*rtabmap_cloud_, *normals, *cloudToExport);

  // Orient normals toward their camera viewpoints
  rtabmap::util3d::adjustNormalsToViewPoints(
    rawViewpoints, cloudXYZ, rawViewpointIndices, cloudToExport, true);

  std::cout << "Normals adjusted (" << timer.ticks() << "s, "
            << cloudToExport->size() << " points)\n";
}

void WCDatabaseExporter::assemble_colored_point_cloud()
{
  // Cloud is already colored by cloudRGBFromSensorData using the aligned
  // RealSense color camera. cloudToExport (XYZRGB+Normal) is populated in
  // projectAndColorizePointCloud() — nothing to do here.
  std::cout << "assemble_colored_point_cloud: using depth-image colors, skipping re-colorization.\n";
  return;

  float textureRange = 0.0f;
  float textureAngle = 0.0f;
  float maxDepthError = 0.0f;
  std::vector<float> textureRoiRatios;
  cv::Mat projMask;
  bool distanceToCamPolicy = false;
  const rtabmap::ProgressState progressState;
  pointToPixel = rtabmap::util3d::projectCloudToCameras(
    *cloudIToExport, robotPoses, cameraModels, textureRange, textureAngle,
    maxDepthError, textureRoiRatios, projMask, distanceToCamPolicy,
    &progressState);

  std::vector<int> pointToCamId;
  std::vector<float> pointToCamIntensity;
  pointToCamId.resize(cloudIToExport->size());

  UASSERT(pointToPixel.empty() || pointToPixel.size() == pointToCamId.size());
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr assembledCloudValidPoints(
    new pcl::PointCloud<pcl::PointXYZRGBNormal>());
  assembledCloudValidPoints->resize(pointToCamId.size());

  // Figure out what color each point in the pointcloud should be based on
  // pixel
  // color
  int imagesDone = 1;
  for (std::map<int, rtabmap::Transform>::iterator iter = robotPoses.begin();
       iter != robotPoses.end(); ++iter) {
    int nodeID = iter->first;
    cv::Mat image;
    if (uContains(nodes, nodeID) &&
        !nodes.at(nodeID).sensorData().imageCompressed().empty()) {
      nodes.at(nodeID).sensorData().uncompressDataConst(&image, 0);
    }
    if (!image.empty()) {
      UASSERT(cameraModels.find(nodeID) != cameraModels.end());
      int modelsSize = cameraModels.at(nodeID).size();
      for (size_t i = 0; i < pointToPixel.size(); ++i) {
        int cameraIndex = pointToPixel[i].first.second;
        if (nodeID == pointToPixel[i].first.first && cameraIndex >= 0) {
          pcl::PointXYZRGBNormal pt;
          float intensity = 0;
          if (!cloudIToExport->empty()) {
            pt.x = cloudIToExport->at(i).x;
            pt.y = cloudIToExport->at(i).y;
            pt.z = cloudIToExport->at(i).z;
            pt.normal_x = cloudIToExport->at(i).normal_x;
            pt.normal_y = cloudIToExport->at(i).normal_y;
            pt.normal_z = cloudIToExport->at(i).normal_z;
            intensity = cloudIToExport->at(i).intensity;
          }

          int subImageWidth = image.cols / modelsSize;
          cv::Mat subImage = image(
            cv::Range::all(), cv::Range(cameraIndex * subImageWidth,
                                        (cameraIndex + 1) * subImageWidth));

          int x = pointToPixel[i].second.x * (float)subImage.cols;
          int y = pointToPixel[i].second.y * (float)subImage.rows;
          UASSERT(x >= 0 && x < subImage.cols);
          UASSERT(y >= 0 && y < subImage.rows);

          UASSERT(subImage.type() == CV_8UC1);
          pt.r = pt.g = pt.b = subImage.at<unsigned char>(
            pointToPixel[i].second.y * subImage.rows,
            pointToPixel[i].second.x * subImage.cols);

          int exportedId = nodeID;
          pointToCamId[i] = exportedId;
          if (!pointToCamIntensity.empty()) {
            pointToCamIntensity[i] = intensity;
          }
          assembledCloudValidPoints->at(i) = pt;
        }
      }
    }
    std::cout << "Processed " << imagesDone++ << "/"
              << static_cast<int>(robotPoses.size()) << " images\n";
  }

  pcl::IndicesPtr validIndices(new std::vector<int>(pointToPixel.size()));
  size_t oi = 0;
  for (size_t i = 0; i < pointToPixel.size(); ++i) {
    if (pointToPixel[i].first.first <= 0) {
      pcl::PointXYZRGBNormal pt;
      float intensity = 0;
      if (!cloudIToExport->empty()) {
        pt.x = cloudIToExport->at(i).x;
        pt.y = cloudIToExport->at(i).y;
        pt.z = cloudIToExport->at(i).z;
        pt.normal_x = cloudIToExport->at(i).normal_x;
        pt.normal_y = cloudIToExport->at(i).normal_y;
        pt.normal_z = cloudIToExport->at(i).normal_z;
        intensity = cloudIToExport->at(i).intensity;
      }

      pointToCamId[i] = 0; // invalid
      pt.b = 0;
      pt.g = 0;
      pt.r = 255;
      if (!pointToCamIntensity.empty()) {
        pointToCamIntensity[i] = intensity;
      }
      assembledCloudValidPoints->at(i) = pt; // red
      validIndices->at(oi++) = i;
    } else {
      validIndices->at(oi++) = i;
    }
  }

  if (oi != validIndices->size()) {
    validIndices->resize(oi);
    assembledCloudValidPoints = rtabmap::util3d::extractIndices(
      assembledCloudValidPoints, validIndices, false, false);
    std::vector<int> pointToCamIdTmp(validIndices->size());
    std::vector<float> pointToCamIntensityTmp(validIndices->size());
    for (size_t i = 0; i < validIndices->size(); ++i) {
      pointToCamIdTmp[i] = pointToCamId[validIndices->at(i)];
      pointToCamIntensityTmp[i] = pointToCamIntensity[validIndices->at(i)];
    }
    pointToCamId = pointToCamIdTmp;
    pointToCamIntensity = pointToCamIntensityTmp;
    pointToCamIdTmp.clear();
    pointToCamIntensityTmp.clear();
  }

  cloudToExport = assembledCloudValidPoints;
  std::cout << "Assembling colored point cloud... done!" << std::endl;
}

void WCDatabaseExporter::finalize_and_return_result(Result &result)
{
  std::cout << "Starting copyPointCloud..." << std::endl;
  pcl::copyPointCloud(*cloudToExport, *rtabmap_cloud_);
  std::cout << "Starting filter_point_cloud..." << std::endl;
  rtabmap_cloud_ = filter_point_cloud(rtabmap_cloud_);
  std::cout << "Finished filtering point cloud." << std::endl;

  // RANSAC();

  for (std::map<int, std::vector<rtabmap::CameraModel>>::iterator iter =
         cameraModels.begin();
       iter != cameraModels.end(); ++iter) {

    std::cout << "Processing node " << iter->first << std::endl;

    // Create an empty frame for a Mono8 image (grayscale)

    // std::cout << "Number of images for this node: "
    //           << rgb_images[iter->first].cols << "x"
    //           << rgb_images[iter->first].rows << std::endl;
    cv::Mat frame = cv::Mat::zeros(iter->second.front().imageHeight(),
                                   iter->second.front().imageWidth(), CV_8UC1);
    // std::cout << "Created empty frame of size: " << frame.cols << "x"
    //           << frame.rows << std::endl;
    cv::Mat depth(iter->second.front().imageHeight(),
                  iter->second.front().imageWidth(), CV_32FC1);
    // std::cout << "Created empty depth of size: " << depth.cols << "x"
    //           << depth.rows << std::endl;
    cv::Mat combined_image =
      rgb_images[iter->first]; // Assuming mono_images map
    // stores the Mono8 images

    // std::cout << "Combined image size: " << combined_image.cols << "x"
    //           << combined_image.rows << std::endl;
    int width = combined_image.cols / 2;
    int height = combined_image.rows;

    // std::cout << "Width: " << width << ", Height: " << height << std::endl;
    cv::Mat left_image = combined_image(cv::Rect(0, 0, width, height)).clone();
    cv::Mat right_image = combined_image(cv::Rect(width, 0, width, height)).clone();
    std::pair<cv::Mat, std::map<std::pair<int, int>, int>> depth_map;

    cv::Mat combined_depth = depth_images[iter->first];
    cv::Mat left_depth = combined_depth(cv::Rect(0, 0, width, height)).clone();
    cv::Mat right_depth = combined_depth(cv::Rect(width, 0, width, height)).clone();
    // Iterate over each camera model in the node
    // std::cout << "Number of camera models: " << iter->second.size()
    //           << std::endl;
    for (size_t i = 0; i < iter->second.size(); ++i) {
      cv::Mat mono_frame = (i == 0) ? left_image : right_image;
      cv::Mat mono_depth = (i == 0) ? left_depth : right_depth;

      // Use the cleaned image already produced during re-SLAM; fall back to
      // running U-Net now only if the cache is missing (e.g. U-Net wasn't
      // loaded during re-SLAM).
      auto cache_it = cleaned_combined_cache_.find(iter->first);
      if (cache_it != cleaned_combined_cache_.end()) {
        int hw = cache_it->second.cols / 2;
        mono_frame = (i == 0)
          ? cache_it->second(cv::Rect(0, 0, hw, cache_it->second.rows)).clone()
          : cache_it->second(cv::Rect(hw, 0, hw, cache_it->second.rows)).clone();
      } else if (unet_loaded_) {
        mono_frame = run_unet(mono_frame);
      }

      // Store the cleaned grayscale in original camera orientation so
      // edge_detection() can use it directly without re-running U-Net.
      cleaned_imgs.push_back(mono_frame.clone());
      // Parallel entry so build_map_from_edges() has intrinsics + localTransform.
      frame_camera_models_.push_back(iter->second.at(i));

      cv::Mat image_rotate;
      cv::rotate(mono_frame, image_rotate, cv::ROTATE_90_COUNTERCLOCKWISE);

      cv::Mat color_image;
      cv::cvtColor(image_rotate, color_image, cv::COLOR_GRAY2BGR);

      cv::Mat lab;
      cv::cvtColor(color_image, lab, cv::COLOR_BGR2Lab);

      std::vector<cv::Mat> lab_planes(3);
      cv::split(lab, lab_planes);
      cv::Mat l_channel = lab_planes[0];
      cv::Mat a = lab_planes[1];
      cv::Mat b = lab_planes[2];

      // Apply CLAHE to the L-channel
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
      cv::Mat cl;
      clahe->apply(l_channel, cl);

      // Merge the CLAHE enhanced L-channel back with A and B channels
      cv::Mat limg;
      cv::merge(std::vector<cv::Mat>{cl, a, b}, limg);

      // Convert the enhanced LAB image back to BGR
      cv::Mat enhanced_img;
      cv::cvtColor(limg, enhanced_img, cv::COLOR_Lab2BGR);

      // std::cout << "Size of Image is : " << iter->second.at(i).imageSize()
      //           << std::endl;
      std::pair<cv::Mat, std::map<std::pair<int, int>, int>> depth_map =
        project_cloud_to_camera(
          cv::Size(640, 360), iter->second.at(i).K(), rtabmap_cloud_,
          robotPoses.at(iter->first) * iter->second.at(i).localTransform());

      cv::Mat rotated_depth;
      cv::rotate(depth_map.first, rotated_depth, 0);

      // Create a frame for visualization with the same dimensions as rotated
      // images
      cv::Mat frame =
        cv::Mat::zeros(rotated_depth.rows, rotated_depth.cols, CV_8UC1);

      // Iterate over all pixels and visualize the depth by drawing circles
      for (int y = 0; y < rotated_depth.rows; ++y) {
        for (int x = 0; x < rotated_depth.cols; ++x) {
          if (rotated_depth.at<float>(y, x) > 0.0f) { // Valid depth
            // Use intensity from the enhanced image for visualization
            uchar intensity =
              enhanced_img.at<cv::Vec3b>(y, x)[0]; // Use first channel
            cv::circle(frame, cv::Point(x, y), 1, cv::Scalar(intensity), -1);
          }
        }
      }
      // Store the mapping data (Mono8 image, frame with depth circles, pose,
      // and depth map)
      // mapping_data_.push_back(
      //   {enhanced_img, frame, robotPoses.at(iter->first), depth_map.second});
      mapping_data_.push_back(
        {enhanced_img, mono_depth, robotPoses.at(iter->first), depth_map.second});
    }
  }

  edge_detection();
  build_map_from_edges();

  // ── Occupancy grid ────────────────────────────────────────────────────────
  std::string path = std::string(PROJECT_PATH) + "/output/" + timestamp_;

  // Save point cloud.
  std::string cloud_path = path + "/cloud/" + timestamp_ + ".pcd";
  if (!rtabmap_cloud_->empty())
    pcl::io::savePCDFileBinary(cloud_path, *rtabmap_cloud_);

  // PCA rotation: rotate the scan cloud and all overlay data so the room's
  // long axis aligns with the grid X axis before rasterisation.
  {
    constexpr float kMinZ = 0.05f, kMaxZ = 0.30f;
    auto grid_cloud = (!scan_cloud_->empty()) ? scan_cloud_ : rtabmap_cloud_;

    float pca_angle = 0.0f, cx = 0.0f, cy = 0.0f;
    int n_pts = 0;
    for (const auto &pt : grid_cloud->points) {
      if (pt.z < kMinZ || pt.z > kMaxZ) continue;
      cx += pt.x; cy += pt.y; ++n_pts;
    }
    if (n_pts >= 20) {
      cx /= n_pts; cy /= n_pts;
      cv::Mat pd(n_pts, 2, CV_32F);
      int i = 0;
      for (const auto &pt : grid_cloud->points) {
        if (pt.z < kMinZ || pt.z > kMaxZ) continue;
        pd.at<float>(i, 0) = pt.x; pd.at<float>(i, 1) = pt.y; ++i;
      }
      cv::PCA pca(pd, cv::Mat(), cv::PCA::DATA_AS_ROW);
      float dx = pca.eigenvectors.at<float>(0, 0);
      float dy = pca.eigenvectors.at<float>(0, 1);
      pca_angle = std::atan2(dy, dx) * 180.0f / (float)M_PI;
      if (pca_angle >  90.0f) pca_angle -= 180.0f;
      if (pca_angle <= -90.0f) pca_angle += 180.0f;
    }

    float total_angle = -pca_angle;
    {
      float ra = total_angle * (float)M_PI / 180.0f;
      float ca = std::cos(ra), sa = std::sin(ra);
      float xmin = 1e9f, xmax = -1e9f, ymin = 1e9f, ymax = -1e9f;
      for (const auto &pt : grid_cloud->points) {
        if (pt.z < kMinZ || pt.z > kMaxZ) continue;
        float rx = ca*(pt.x-cx) - sa*(pt.y-cy);
        float ry = sa*(pt.x-cx) + ca*(pt.y-cy);
        xmin = std::min(xmin, rx); xmax = std::max(xmax, rx);
        ymin = std::min(ymin, ry); ymax = std::max(ymax, ry);
      }
      if ((ymax - ymin) > (xmax - xmin)) total_angle += 90.0f;
    }

    const float rad = total_angle * (float)M_PI / 180.0f;
    const float ca  = std::cos(rad), sa = std::sin(rad);
    auto rot2 = [&](float x, float y) -> std::pair<float,float> {
      float dx = x - cx, dy = y - cy;
      return {cx + ca*dx - sa*dy, cy + sa*dx + ca*dy};
    };
    for (auto &pt : grid_cloud->points) {
      auto [rx, ry] = rot2(pt.x, pt.y); pt.x = rx; pt.y = ry;
    }
    for (auto &[tid, obs] : tag_obs_)
      for (auto &o : obs) {
        auto [rx, ry] = rot2(o.x, o.y); o.x = rx; o.y = ry;
      }
    for (auto &[nid, T] : viz_poses_) {
      auto [rx, ry] = rot2(T.x(), T.y());
      T = rtabmap::Transform(rx, ry, T.z(), 0.0f, 0.0f, 0.0f);
    }
    for (auto &[nid, pts] : per_scan_xyz_)
      for (auto &p : pts) {
        auto [rx, ry] = rot2(p.x, p.y); p.x = rx; p.y = ry; // z unchanged
      }
    for (auto &[nid, p] : per_scan_origin_) {
      auto [rx, ry] = rot2(p.x, p.y); p.x = rx; p.y = ry;
    }
    printf("[occ_grid] PCA pre-rotation: pca=%.1f deg, total=%.1f deg\n",
           pca_angle, total_angle);
  }

  // Build and save the occupancy grid.
  rtabmap_occupancy_grid_ = per_scan_log_odds_grid();
  {
    std::string grid_path = path + "/grid/" + timestamp_;
    nav2_map_server::SaveParameters sp;
    sp.map_file_name    = grid_path;
    sp.image_format     = "pgm";
    sp.free_thresh      = 0.196;
    sp.occupied_thresh  = 0.65;
    nav2_map_server::saveMapToFile(*rtabmap_occupancy_grid_, sp);
  }

  // Tag overlay PNG.
  if (rtabmap_occupancy_grid_ && !tag_obs_.empty()) {
    const auto &info = rtabmap_occupancy_grid_->info;
    int gw = (int)info.width, gh = (int)info.height;
    float res = info.resolution;
    float ox  = (float)info.origin.position.x;
    float oy  = (float)info.origin.position.y;
    const int S = 4;
    int iw = gw * S, ih = gh * S;
    cv::Mat img(ih, iw, CV_8UC3);
    for (int r = 0; r < gh; ++r)
      for (int c = 0; c < gw; ++c) {
        int8_t v = (int8_t)rtabmap_occupancy_grid_->data[(gh-1-r)*gw+c];
        cv::Vec3b col = (v < 0) ? cv::Vec3b{128,128,128}
                      : (v==0) ? cv::Vec3b{240,240,240}
                               : cv::Vec3b{30,30,30};
        img(cv::Rect(c*S, r*S, S, S)).setTo(cv::Scalar(col[0],col[1],col[2]));
      }
    auto wp = [&](float x, float y) -> cv::Point {
      return {(int)((x-ox)/res*S), ih-1-(int)((y-oy)/res*S)};
    };
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
    int num_tags = (int)tag_obs_.size(), tag_idx = 0;
    for (auto &[tag_id, obs] : tag_obs_) {
      float hue = 360.0f * tag_idx / std::max(num_tags, 1);
      cv::Mat hsv(1,1,CV_8UC3,cv::Scalar((uint8_t)(hue/2),210,230));
      cv::Mat bgrc; cv::cvtColor(hsv, bgrc, cv::COLOR_HSV2BGR);
      cv::Vec3b c = bgrc.at<cv::Vec3b>(0,0);
      cv::Scalar color(c[0],c[1],c[2]), dim(c[0]*.4,c[1]*.4,c[2]*.4);
      int n = (int)obs.size();
      float mx=0, my=0;
      for (auto &o : obs) { mx+=o.x; my+=o.y; }
      mx/=n; my/=n;
      float sx=0, sy=0;
      for (auto &o : obs) { sx+=(o.x-mx)*(o.x-mx); sy+=(o.y-my)*(o.y-my); }
      float std_2d = std::sqrt((sx/n+sy/n)/2.0f);
      for (auto &o : obs) {
        cv::Point pt = wp(o.x, o.y);
        if (pt.x>=0&&pt.x<iw&&pt.y>=0&&pt.y<ih)
          cv::circle(img, pt, 3, dim, -1, cv::LINE_AA);
      }
      cv::Point mp = wp(mx, my);
      int r_px = (int)(std_2d/res*S);
      if (r_px > 1) cv::circle(img, mp, r_px, color, 1, cv::LINE_AA);
      cv::circle(img, mp, 7, cv::Scalar(0,0,0), -1, cv::LINE_AA);
      cv::circle(img, mp, 5, color, -1, cv::LINE_AA);
      if (mp.x>=0&&mp.x<iw&&mp.y>=0&&mp.y<ih)
        cv::putText(img, std::to_string(tag_id), mp+cv::Point(9,4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
      ++tag_idx;
    }
    cv::putText(img, "dim=detection  bright=mean  circle=XY std  grey=camera path",
                {4,ih-6}, cv::FONT_HERSHEY_SIMPLEX, 0.38,
                cv::Scalar(100,100,100), 1, cv::LINE_AA);
    std::string tag_png = path + "/grid/" + timestamp_ + "_tags.png";
    cv::imwrite(tag_png, img);
    std::cout << "Tag overlay saved to " << tag_png << "\n";
  }

  // Save images, cleaned frames, edge frames, and camera models.
  int rgbImagesExported = 0, depthImagesExported = 0;
  for (const auto &data : mapping_data_) {
    cv::imwrite(path+"/images/"+std::to_string(rgbImagesExported)+".jpg",
                std::get<0>(data));
    ++rgbImagesExported;
    cv::Mat depth_vis;
    cv::normalize(std::get<1>(data), depth_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite(path+"/depths/"+std::to_string(depthImagesExported)+".jpg",
                depth_vis);
    ++depthImagesExported;
  }
  int cleanedImagesExported = 0;
  for (const auto &img : cleaned_imgs)
    cv::imwrite(path+"/cleaned/"+std::to_string(cleanedImagesExported++)+".jpg", img);
  int edgeImagesExported = 0;
  for (const auto &img : edge_imgs)
    cv::imwrite(path+"/edges/"+std::to_string(edgeImagesExported++)+".jpg", img);
  for (size_t i = 0; i < camera_models_.size(); i++)
    for (size_t j = 0; j < camera_models_.at(i).size(); j++) {
      rtabmap::CameraModel model = camera_models_.at(i).at(j);
      std::string name = std::to_string(i);
      if (camera_models_.at(i).size() > 1) name += "_" + uNumber2Str((int)j);
      model.setName(name);
      model.save(path + "/camera_models/");
    }
  std::cout << "RGB Images exported: " << rgbImagesExported << "\n"
            << "Depth Images exported: " << depthImagesExported << "\n"
            << "Cleaned Images exported: " << cleanedImagesExported << "\n"
            << "Edge Images exported: " << edgeImagesExported << "\n";

  result.success = true;
  result.timestamp = timestamp_;
  result.cloud = rtabmap_cloud_;
  result.mapping_data = std::move(mapping_data_);  // move avoids deep-copying ~1384 cv::Mat pairs

  std::cout << "Finished loading database" << std::endl;
  std::cout << "Number of images: " << result.mapping_data.size() << std::endl;
  std::cout << "Number of points in cloud: " << rtabmap_cloud_->points.size()
            << std::endl;
  std::cout << "Timestamp: " << timestamp_ << std::endl;

  return;
}

void WCDatabaseExporter::edge_detection()
{
  for (size_t i = 0; i < cleaned_imgs.size(); i++) {
    const auto &gray = cleaned_imgs[i];

    // Appearance edges from U-Net cleaned grayscale
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);
    cv::Mat gray_edges;
    cv::Canny(blurred, gray_edges, 50.0, 150.0);

    cv::Mat combined = gray_edges.clone();
    cv::Mat depth_edges_out; // depth-only, kept separate for wall map

    // Depth edges — finds geometric boundaries (floor-wall seam, object
    // silhouettes) that may be missed by appearance alone.
    if (i < mapping_data_.size()) {
      const cv::Mat &depth_raw = std::get<1>(mapping_data_[i]);
      if (!depth_raw.empty()) {
        cv::Mat depth_f;
        if (depth_raw.type() == CV_16UC1)
          depth_raw.convertTo(depth_f, CV_32F, 0.001f); // mm → m
        else
          depth_raw.convertTo(depth_f, CV_32F);

        if (depth_f.size() != gray.size())
          cv::resize(depth_f, depth_f, gray.size(), 0, 0, cv::INTER_NEAREST);

        // Clip at 5 m
        cv::threshold(depth_f, depth_f, 5.0f, 5.0f, cv::THRESH_TRUNC);

        // Fill invalid (zero) depth holes with a blurred estimate from valid
        // neighbours so Canny doesn't fire on zero→valid boundaries (IR dots).
        cv::Mat invalid_mask = (depth_f < 0.05f);
        cv::Mat depth_filled = depth_f.clone();
        cv::Mat depth_hole_fill;
        cv::GaussianBlur(depth_f, depth_hole_fill, cv::Size(21, 21), 7.0);
        depth_hole_fill.copyTo(depth_filled, invalid_mask);

        cv::Mat depth_u8;
        cv::normalize(depth_filled, depth_u8, 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::Mat db;
        cv::GaussianBlur(depth_u8, db, cv::Size(3, 3), 0);
        cv::Mat depth_edges;
        cv::Canny(db, depth_edges, 300.0, 600.0);

        // Remove edges that fall inside originally-invalid regions (IR dot holes):
        // erode the valid mask so edge pixels near holes are also suppressed.
        cv::Mat valid_mask_eroded;
        cv::erode(~invalid_mask, valid_mask_eroded,
                  cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
        cv::bitwise_and(depth_edges, valid_mask_eroded, depth_edges);

        cv::bitwise_or(gray_edges, depth_edges, combined);
        depth_edges_out = depth_edges;
      }
    }

    edge_imgs.push_back(combined);
    depth_edge_imgs_.push_back(depth_edges_out); // empty if no depth available
  }
}

void WCDatabaseExporter::build_map_from_edges()
{
  // Resolution and height band.
  // Camera is at z=0.135 m above base_link (floor).
  // Wall-ground contact edges are at world z ≈ 0; furniture (table faces,
  // chair seats) sits at 0.4 m+.  Keeping only [0.05, 0.20] m isolates the
  // wall floor-contact line and ignores mid-height furniture edges.
  // Chair legs may still appear but are small enough for the area filter later.
  const float resolution     = 0.05f;  // m/pixel
  const float min_obstacle_h = -0.05f; // m — wall-floor seam is at world z≈0; allow small negative for floor tilt/drift
  const float max_obstacle_h = 0.12f;  // m — tight band reduces back-projected spread from camera tilt
  const float max_depth_m    = 8.00f;  // m — discard unreliable depth readings

  std::vector<std::pair<float, float>> footprint; // (world_x, world_y)

  // --- Transform sanity check (remove once verified) ---
  if (!frame_camera_models_.empty() && !mapping_data_.empty()) {
    for (size_t cam = 0; cam < std::min(frame_camera_models_.size(), (size_t)2); cam++) {
      rtabmap::Transform cam_in_world =
        std::get<2>(mapping_data_[cam]) * frame_camera_models_[cam].localTransform();
      Eigen::Matrix4f T = cam_in_world.toEigen4f();
      printf("[cam %zu] position in world:      %.3f  %.3f  %.3f  (expect ~±0.265, 0.365, 0.135)\n",
             cam, T(0,3), T(1,3), T(2,3));
      printf("[cam %zu] optical Z axis in world: %.3f  %.3f  %.3f  (expect mostly negative Z = downward)\n",
             cam, T(0,2), T(1,2), T(2,2));
    }
  }
  // ------------------------------------------------------

  for (size_t i = 0; i < edge_imgs.size(); i++) {
    if (i >= frame_camera_models_.size() || i >= mapping_data_.size()) break;

    // Use depth-only edges: depth Canny fires at geometric discontinuities
    // (wall-floor seam, object silhouettes) but NOT on floor texture, which
    // is the main source of noise when cameras look mostly downward.
    if (i >= depth_edge_imgs_.size() || depth_edge_imgs_[i].empty()) continue;
    const cv::Mat &edges           = depth_edge_imgs_[i];
    const cv::Mat &depth_raw       = std::get<1>(mapping_data_[i]);
    const rtabmap::Transform &pose = std::get<2>(mapping_data_[i]);
    const rtabmap::CameraModel &model = frame_camera_models_[i];

    if (depth_raw.empty() || model.fx() == 0.0) continue;

    // Convert depth to float metres
    cv::Mat depth_m;
    if (depth_raw.type() == CV_16UC1)
      depth_raw.convertTo(depth_m, CV_32F, 0.001f); // mm → m
    else if (depth_raw.type() == CV_32FC1)
      depth_m = depth_raw;
    else
      continue;

    // Align depth to edge image size (nearest-neighbour to avoid interpolating depth)
    if (depth_m.size() != edges.size())
      cv::resize(depth_m, depth_m, edges.size(), 0, 0, cv::INTER_NEAREST);

    const double fx = model.fx(), fy = model.fy();
    const double cx = model.cx(), cy = model.cy();

    // world_T_camera: maps a point expressed in the camera optical frame to world
    const Eigen::Matrix4f T = (pose * model.localTransform()).toEigen4f();

    for (int v = 0; v < edges.rows; v++) {
      for (int u = 0; u < edges.cols; u++) {
        if (edges.at<uchar>(v, u) == 0) continue;

        const float d = depth_m.at<float>(v, u);
        if (d < 0.05f || d > max_depth_m) continue;

        // Back-project pixel (u, v, d) → camera optical frame
        const float X = static_cast<float>((u - cx) / fx) * d;
        const float Y = static_cast<float>((v - cy) / fy) * d;

        // Transform to world frame
        const Eigen::Vector4f pw = T * Eigen::Vector4f(X, Y, d, 1.0f);
        const float wz = pw.z();

        // Keep only obstacle-height band; XY footprint is cast to floor
        if (wz < min_obstacle_h || wz > max_obstacle_h) continue;

        footprint.emplace_back(pw.x(), pw.y());
      }
    }
  }

  if (footprint.empty()) {
    std::cout << "build_map_from_edges: no footprint points — skipping map.\n";
    return;
  }

  float mn_x =  std::numeric_limits<float>::max();
  float mn_y =  std::numeric_limits<float>::max();
  float mx_x = -std::numeric_limits<float>::max();
  float mx_y = -std::numeric_limits<float>::max();
  for (const auto &[x, y] : footprint) {
    mn_x = std::min(mn_x, x); mn_y = std::min(mn_y, y);
    mx_x = std::max(mx_x, x); mx_y = std::max(mx_y, y);
  }
  mn_x -= 1.0f; mn_y -= 1.0f;
  mx_x += 1.0f; mx_y += 1.0f;

  const int grid_w = static_cast<int>((mx_x - mn_x) / resolution) + 1;
  const int grid_h = static_cast<int>((mx_y - mn_y) / resolution) + 1;

  cv::Mat grid(grid_h, grid_w, CV_8U, cv::Scalar(205)); // 205 = unknown

  for (const auto &[x, y] : footprint) {
    int gx = static_cast<int>((x - mn_x) / resolution);
    int gy = static_cast<int>((y - mn_y) / resolution);
    if (gx >= 0 && gx < grid_w && gy >= 0 && gy < grid_h)
      grid.at<uchar>(gy, gx) = 0; // occupied
  }

  // Post-process: thin the wall lines before saving.
  // 1. Open (5x5) — removes blobs up to 2 cells wide; clears most noise.
  // 2. Square erode (3x3) — shrinks every wall edge by 1 cell on all 8 sides.
  // 3. CC filter — removes stray fragments shorter than ~1 m of wall (40 cells).
  {
    cv::Mat occ = (grid == 0); // 255 where occupied
    cv::Mat k5  = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat k3  = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(occ, occ, cv::MORPH_OPEN, k5);
    cv::erode(occ, occ, k3);
    cv::Mat labels, stats, centroids;
    int ncomp = cv::connectedComponentsWithStats(occ, labels, stats, centroids);
    for (int lbl = 1; lbl < ncomp; ++lbl)
      if (stats.at<int>(lbl, cv::CC_STAT_AREA) < 40)
        occ.setTo(0, labels == lbl);
    grid.setTo(cv::Scalar(205));
    grid.setTo(cv::Scalar(0), occ);
  }

  // Standardize orientation: rotate so the room's long axis is horizontal.
  // PCA on occupied cells gives the principal direction; clamping the angle
  // to (-90°, 90°] resolves the 180° eigenvector ambiguity canonically.
  // A 90° correction is applied afterwards if the result is portrait.
  {
    std::vector<cv::Point2f> pts;
    for (int r = 0; r < grid.rows; r++)
      for (int c = 0; c < grid.cols; c++)
        if (grid.at<uchar>(r, c) == 0)
          pts.emplace_back((float)c, (float)r);

    if (pts.size() >= 20) {
      cv::Mat data((int)pts.size(), 2, CV_32F);
      for (int i = 0; i < (int)pts.size(); i++) {
        data.at<float>(i, 0) = pts[i].x;
        data.at<float>(i, 1) = pts[i].y;
      }
      cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
      float dx = pca.eigenvectors.at<float>(0, 0);
      float dy = pca.eigenvectors.at<float>(0, 1);
      float angle = std::atan2(dy, dx) * 180.0f / (float)M_PI;
      // Clamp to (-90, 90] so the canonical direction is always rightward.
      if (angle > 90.0f)  angle -= 180.0f;
      if (angle <= -90.0f) angle += 180.0f;

      // Expand canvas to diagonal so corners don't clip during rotation.
      int diag = (int)std::ceil(std::hypot(grid.cols, grid.rows)) + 4;
      cv::Mat canvas(diag, diag, CV_8U, cv::Scalar(205));
      int ox = (diag - grid.cols) / 2, oy = (diag - grid.rows) / 2;
      grid.copyTo(canvas(cv::Rect(ox, oy, grid.cols, grid.rows)));

      cv::Mat M = cv::getRotationMatrix2D({diag / 2.0f, diag / 2.0f}, angle, 1.0);
      cv::Mat rotated;
      cv::warpAffine(canvas, rotated, M, {diag, diag},
                     cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(205));

      // Crop to tight bounding box of occupied cells + small padding.
      int rmin = rotated.rows, rmax = -1, cmin = rotated.cols, cmax = -1;
      for (int r = 0; r < rotated.rows; r++)
        for (int c = 0; c < rotated.cols; c++)
          if (rotated.at<uchar>(r, c) == 0) {
            rmin = std::min(rmin, r); rmax = std::max(rmax, r);
            cmin = std::min(cmin, c); cmax = std::max(cmax, c);
          }
      if (rmax >= rmin && cmax >= cmin) {
        const int pad = 5;
        int r0 = std::max(0, rmin - pad), r1 = std::min(rotated.rows - 1, rmax + pad);
        int c0 = std::max(0, cmin - pad), c1 = std::min(rotated.cols - 1, cmax + pad);
        grid = rotated(cv::Rect(c0, r0, c1 - c0 + 1, r1 - r0 + 1)).clone();

        // Ensure landscape: if portrait, rotate 90° clockwise.
        if (grid.rows > grid.cols)
          cv::rotate(grid, grid, cv::ROTATE_90_CLOCKWISE);

        printf("[edge_map] PCA angle=%.1f deg → rotated+cropped to %dx%d px\n",
               angle, grid.cols, grid.rows);
      }
    }
  }

  std::string grid_dir = std::string(PROJECT_PATH) + "/output/" + timestamp_ + "/grid/";
  std::string pgm_path  = grid_dir + "edge_map.pgm";
  std::string yaml_path = grid_dir + "edge_map.yaml";

  cv::imwrite(pgm_path, grid);

  std::ofstream yaml(yaml_path);
  yaml << "image: edge_map.pgm\n"
       << "resolution: " << resolution << "\n"
       << "origin: [" << mn_x << ", " << mn_y << ", 0.0]\n"
       << "negate: 0\n"
       << "occupied_thresh: 0.65\n"
       << "free_thresh: 0.196\n";
  yaml.close();

  std::cout << "Map saved: " << pgm_path << " (" << grid_w << "x" << grid_h
            << " px at " << resolution << " m/px, "
            << footprint.size() << " footprint points)\n";
}

// @brief Project a point cloud to a given image frame, and map the index of
// each point that is visible in the camera frame to the pixel coordinate in
// the image frame
// @param image_size: The size of the image frame
// @param camera_matrix: The camera matrix
// @param cloud: The point cloud to project to the camera frame
// @param camera_transform: The transformation matrix from the camera frame to
// the world frame
// @return A pair containing the depth image and a map from pixel

std::pair<cv::Mat, std::map<std::pair<int, int>, int>>
WCDatabaseExporter::project_cloud_to_camera(
  const cv::Size &image_size, const cv::Mat &camera_matrix,
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
  const rtabmap::Transform &camera_transform)
{
  assert(!camera_transform.isNull());
  assert(!cloud->empty());
  assert(camera_matrix.type() == CV_64FC1 && camera_matrix.cols == 3 &&
         camera_matrix.cols == 3);

  float fx = camera_matrix.at<double>(0, 0);
  float fy = camera_matrix.at<double>(1, 1);
  float cx = camera_matrix.at<double>(0, 2);
  float cy = camera_matrix.at<double>(1, 2);

  cv::Mat depth_image = cv::Mat::zeros(image_size, CV_32FC1);
  rtabmap::Transform t = camera_transform.inverse();

  // create a map from each pixel coordinate to the index of their point in the
  // pointcloud
  std::map<std::pair<int, int>, int> pixel_to_point_map;

  int count = 0;
  for (pcl::PointCloud<pcl::PointXYZRGB>::const_iterator it = cloud->begin();
       it != cloud->end(); ++it) {
    pcl::PointXYZRGB ptScan = *it;

    // transform point from world frame to camera frame
    ptScan = rtabmap::util3d::transformPoint(ptScan, t);

    // re-project in camera frame
    float z = ptScan.z;
    bool set = false;
    if (z > 0.0f) {
      float invZ = 1.0f / z;
      float dx = (fx * ptScan.x) * invZ + cx;
      float dy = (fy * ptScan.y) * invZ + cy;
      int dx_low = dx;
      int dy_low = dy;
      int dx_high = dx + 0.5f;
      int dy_high = dy + 0.5f;
      if (uIsInBounds(dx_low, 0, depth_image.cols) &&
          uIsInBounds(dy_low, 0, depth_image.rows)) {
        set = true;
        float &zReg = depth_image.at<float>(dy_low, dx_low);
        if (zReg == 0 || z < zReg) {
          zReg = z;
        }
      }
      if ((dx_low != dx_high || dy_low != dy_high) &&
          uIsInBounds(dx_high, 0, depth_image.cols) &&
          uIsInBounds(dy_high, 0, depth_image.rows)) {
        set = true;
        float &zReg = depth_image.at<float>(dy_high, dx_high);
        if (zReg == 0 || z < zReg) {
          zReg = z;
        }
      }
      if (set) {
        pixel_to_point_map[{dy_low, dx_low}] = count;
      }
    }
    count++;
  }

  return {depth_image, pixel_to_point_map};
}