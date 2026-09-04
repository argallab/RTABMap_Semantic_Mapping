#include "wc_database_exporter.hpp"

#include <iomanip>
#include <queue>
#include <set>
#include <unistd.h>
#include <rtabmap/core/Optimizer.h>

#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

#include <pcl/filters/voxel_grid.h>

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

WCDatabaseExporter::~WCDatabaseExporter() {}

// ---------------------------------------------------------------------------
// initialize_rtabmap_database
// ---------------------------------------------------------------------------
// Load DB, augment landmark links with bag detections, run 3-pass GTSAM
// re-optimization, then store tag_obs_ and viz_poses_ for the grid overlay.
// ---------------------------------------------------------------------------
bool WCDatabaseExporter::initialize_rtabmap_database()
{
  rtabmap::ParametersMap parameters;
  rtabmap::DBDriver *driver = rtabmap::DBDriver::create();

  std::multimap<int, rtabmap::Link> raw_landmark_links;
  if (driver->openConnection(rtabmap_database_path_)) {
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
  rtabmap.init(parameters, rtabmap_database_path_);
  printf("Loaded in %.2f s\n", timer.ticks());

  printf("Optimizing map...\n");
  rtabmap.getGraph(optimizedPoses, links, true, true, &nodes, true, true, true, true);
  printf("Optimized: %zu poses (%.2f s)\n", optimizedPoses.size(), timer.ticks());

  if (optimizedPoses.empty()) {
    std::cerr << "No optimized poses found\n";
    return false;
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
  constexpr double landmark_var_linear   = 0.005;
  constexpr double landmark_var_angular  = 0.05;
  constexpr double prior_var_linear_base = 0.01;
  constexpr double prior_var_angular_base= 0.01;
  constexpr int    n_reopt_passes        = 3;
  constexpr float  kMaxTagStd            = 0.20f;

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

        for (const auto &[nid, pose] : reoptPoses)
          if (nid > 0) optimizedPoses[nid] = pose;
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
        float sum_delta = 0.0f; int n_shown = 0;
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
          ++n_shown;
        }
        if (n_shown > 0)
          printf("  Mean StdXYZ change: %+.4f m (%s)\n",
                 sum_delta/n_shown,
                 sum_delta/n_shown < 0.0f ? "improved" : "worsened");
      } else {
        std::cerr << "  Pass " << pass << " failed, keeping previous poses\n";
        break;
      }
    }
  }

  // Snapshot post-GTSAM poses and tag observations for the grid overlay.
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
// assembleSceneFromOptimizedPoses
// ---------------------------------------------------------------------------
// For each robot node collect all lidar scan points in world frame.
// per_scan_xyz_    : all endpoints with z in [0.05, 1.5 m]
// per_scan_origin_ : lidar sensor origin XY in world frame
// ---------------------------------------------------------------------------
void WCDatabaseExporter::assembleSceneFromOptimizedPoses()
{
  int n_scans = 0;
  for (auto iter = optimizedPoses.lower_bound(1); iter != optimizedPoses.end(); ++iter) {
    auto node_it = nodes.find(iter->first);
    if (node_it == nodes.end()) continue;

    rtabmap::LaserScan scan;
    node_it->second.sensorData().uncompressData(nullptr, nullptr, &scan);
    if (scan.isEmpty()) continue;

    auto scan_pts = rtabmap::util3d::laserScanToPointCloud(scan, scan.localTransform());
    scan_pts = rtabmap::util3d::transformPointCloud(scan_pts, iter->second);

    rtabmap::Transform sensor_world = iter->second * scan.localTransform();
    per_scan_origin_[iter->first] = {sensor_world.x(), sensor_world.y()};

    auto &xyz = per_scan_xyz_[iter->first];
    for (const auto &pt : scan_pts->points) {
      if (!pcl::isFinite(pt)) continue;
      if (pt.z < 0.05f || pt.z > 1.5f) continue;
      xyz.emplace_back(pt.x, pt.y, pt.z);
    }
    ++n_scans;
  }
  printf("[scan] %d scans assembled, %zu nodes with origins\n",
         n_scans, per_scan_origin_.size());
}

void WCDatabaseExporter::projectAndColorizePointCloud() {}
void WCDatabaseExporter::assemble_colored_point_cloud() {}

// ---------------------------------------------------------------------------
// finalize_and_return_result
// ---------------------------------------------------------------------------
// 1. PCA-rotate per_scan_xyz_ and per_scan_origin_ to align room axes
// 2. Build raytrace occupancy grid (occupied / free / unknown)
// 3. Morphological open to remove noise
// 4. Build voxelized scan cloud and save as PCD
// 5. Save grid PGM + YAML
// 6. Save tag overlay PNG
// ---------------------------------------------------------------------------
void WCDatabaseExporter::finalize_and_return_result(Result &result)
{
  if (per_scan_xyz_.empty()) {
    std::cerr << "[finalize] No scan data assembled — cannot build grid\n";
    return;
  }

  std::string path = std::string(PROJECT_PATH) + "/output/" + timestamp_;

  constexpr float kRes = 0.05f;
  constexpr float kPad = 1.0f;

  float xmin=1e9f, xmax=-1e9f, ymin=1e9f, ymax=-1e9f;
  for (const auto &[nid, pts] : per_scan_xyz_)
    for (const auto &p : pts) {
      xmin=std::min(xmin,p.x); xmax=std::max(xmax,p.x);
      ymin=std::min(ymin,p.y); ymax=std::max(ymax,p.y);
    }
  for (const auto &[nid, o] : per_scan_origin_) {
    xmin=std::min(xmin,o.x); xmax=std::max(xmax,o.x);
    ymin=std::min(ymin,o.y); ymax=std::max(ymax,o.y);
  }
  xmin -= kPad; ymin -= kPad; xmax += kPad; ymax += kPad;

  int gw = (int)std::ceil((xmax - xmin) / kRes) + 1;
  int gh = (int)std::ceil((ymax - ymin) / kRes) + 1;

  std::vector<int8_t> grid(gw * gh, -1);

  auto to_cell = [&](float x, float y, int &cx_, int &cy_) {
    cx_ = (int)((x - xmin) / kRes);
    cy_ = (int)((y - ymin) / kRes);
  };

  // ── 2. Density-based occupied marking ────────────────────────────────
  constexpr float kWallZlo = 0.05f, kWallZhi = 0.75f;
  constexpr int   kMinHits = 8;
  {
    std::vector<int> hits(gw * gh, 0);
    for (const auto &[nid, pts] : per_scan_xyz_)
      for (const auto &p : pts) {
        if (p.z < kWallZlo || p.z > kWallZhi) continue;
        int cx, cy;
        to_cell(p.x, p.y, cx, cy);
        if (cx < 0 || cx >= gw || cy < 0 || cy >= gh) continue;
        ++hits[cy * gw + cx];
      }
    for (int i = 0; i < gw * gh; ++i)
      if (hits[i] >= kMinHits) grid[i] = 100;
  }
  printf("[grid] %d × %d cells  (%.1f × %.1f m)  wall band [%.2f, %.2f m]  min_hits=%d\n",
         gw, gh, gw*kRes, gh*kRes, kWallZlo, kWallZhi, kMinHits);

  // ── 3. Morphological close on occupied cells ──────────────────────────
  // CLOSE (dilate then erode) fills small gaps in thin walls without destroying them.
  {
    cv::Mat occ_mask(gh, gw, CV_8U, cv::Scalar(0));
    for (int r = 0; r < gh; ++r)
      for (int c = 0; c < gw; ++c)
        if (grid[r*gw+c] == 100) occ_mask.at<uint8_t>(r,c) = 255;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::morphologyEx(occ_mask, occ_mask, cv::MORPH_CLOSE, kernel);

    for (int r = 0; r < gh; ++r)
      for (int c = 0; c < gw; ++c)
        if (occ_mask.at<uint8_t>(r,c) > 0) grid[r*gw+c] = 100;
  }

  // ── 4. Flood-fill free space from sensor origins ──────────────────────
  // BFS from every sensor origin through non-occupied cells → marks interior free.
  {
    std::vector<bool> queued(gw * gh, false);
    std::queue<int> bfs;
    for (const auto &[nid, o] : per_scan_origin_) {
      int ocx, ocy;
      to_cell(o.x, o.y, ocx, ocy);
      if (ocx < 0 || ocx >= gw || ocy < 0 || ocy >= gh) continue;
      int idx = ocy * gw + ocx;
      if (queued[idx]) continue;
      queued[idx] = true;
      grid[idx] = 0;
      bfs.push(idx);
    }
    const int dirs[4][2] = {{0,1},{0,-1},{1,0},{-1,0}};
    while (!bfs.empty()) {
      int idx = bfs.front(); bfs.pop();
      int r = idx / gw, c = idx % gw;
      for (auto &d : dirs) {
        int nr = r + d[0], nc = c + d[1];
        if (nr < 0 || nr >= gh || nc < 0 || nc >= gw) continue;
        int nidx = nr * gw + nc;
        if (queued[nidx] || grid[nidx] == 100) continue;
        queued[nidx] = true;
        grid[nidx] = 0;
        bfs.push(nidx);
      }
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

  // ── voxelized scan cloud → PCD ────────────────────────────────────────
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scan_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto &[nid, pts] : per_scan_xyz_)
      for (const auto &p : pts) {
        pcl::PointXYZRGB pt;
        pt.x=p.x; pt.y=p.y; pt.z=p.z;
        pt.r=200; pt.g=200; pt.b=200;
        scan_cloud->push_back(pt);
      }

    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(scan_cloud);
    vg.setLeafSize(kRes, kRes, kRes);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxelized(new pcl::PointCloud<pcl::PointXYZRGB>);
    vg.filter(*voxelized);

    std::string cloud_path = path + "/cloud/" + timestamp_ + ".pcd";
    pcl::io::savePCDFileBinary(cloud_path, *voxelized);
    printf("[cloud] Saved %zu points → %s\n", voxelized->size(), cloud_path.c_str());

    result.cloud = voxelized;
  }

  // ── 5. Save grid PGM + YAML ───────────────────────────────────────────
  {
    std::string grid_path = path + "/grid/" + timestamp_;
    nav2_map_server::SaveParameters sp;
    sp.map_file_name   = grid_path;
    sp.image_format    = "pgm";
    sp.free_thresh     = 0.196;
    sp.occupied_thresh = 0.65;
    nav2_map_server::saveMapToFile(*rtabmap_occupancy_grid_, sp);
    printf("[grid] Saved grid → %s.{pgm,yaml}\n", grid_path.c_str());
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

    // Helper: world XY → image pixel.
    auto wp = [&](float x, float y) -> cv::Point {
      return { (int)((x-xmin)/kRes*S), ih-1-(int)((y-ymin)/kRes*S) };
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

    std::string tag_png = path + "/grid/" + timestamp_ + "_tags.png";
    cv::imwrite(tag_png, img);
    printf("[grid] Tag overlay → %s\n", tag_png.c_str());
  }

  result.success   = true;
  result.timestamp = timestamp_;
  printf("[done] timestamp=%s\n", timestamp_.c_str());
}
