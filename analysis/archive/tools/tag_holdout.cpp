// tag_holdout: re-runs the exporter's pose pipeline (RTAB-Map graph -> tag GTSAM
// 3 passes -> local-map ICP 2 rounds) with a set of AprilTags HELD OUT of every
// optimization, and dumps the poses after each stage.  The held-out tags' spread
// across visits (computed in analysis/a3_tags.py from the dumped poses and the DB's
// landmark links) is then an independent check of pose quality.
//
// Optional --shift DELTA re-times the odometry: every node's odometry pose is moved
// along its local twist by DELTA seconds, neighbour links are recomputed from the
// shifted odometry, and the graph is re-optimized (the latency hypothesis, applied
// inside the graph rather than to fixed poses).
//
// Mirrors src/wc_database_exporter.cpp @ ea2695e: initialize_rtabmap_database()
// (GTSAM passes), clean_scans(), refine_poses_with_local_map_icp().  With no
// holdout and no shift its dumps must equal the exporter's poses_*.csv.
//
// Never touches the input DB: everything runs on a /tmp copy that is deleted at
// exit (Rtabmap::init() writes into the DB it opens, even with close(false); the
// exporter's destructor-close additionally rewrites Admin.opt_poses).
//
// Usage: tag_holdout <db_path> <out_prefix> [--holdout 3,7,12] [--shift -0.15]
//                    [--lm-var-lin 0.02] [--lm-var-ang 0.05]

#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/Link.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/RegistrationIcp.h>
#include <rtabmap/core/RegistrationInfo.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/core/util3d.h>

#include <pcl/common/point_tests.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using Poses = std::map<int, rtabmap::Transform>;
using Links = std::multimap<int, rtabmap::Link>;

namespace {

std::map<int, rtabmap::Signature> g_nodes;
double g_lm_var_lin = 0.02, g_lm_var_ang = 0.05;   // tag GTSAM landmark variances (--lm-var-lin/--lm-var-ang)
bool   g_lm_override = false;  // set by either option: ALSO reweights the graph stage (exporter after ea2695e)

void dump(const std::string &path, const Poses &poses)
{
  std::ofstream pf(path);
  pf << "id,stamp,x,y,z,qx,qy,qz,qw\n";
  for (const auto &[nid, T] : poses) {
    if (nid <= 0) continue;
    auto nit = g_nodes.find(nid);
    const Eigen::Quaternionf q = T.getQuaternionf();
    pf << nid << ',' << std::setprecision(17) << (nit != g_nodes.end() ? nit->second.getStamp() : 0.0)
       << std::setprecision(9) << ',' << T.x() << ',' << T.y() << ',' << T.z() << ','
       << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << '\n';
  }
  fprintf(stderr, "[holdout] wrote %s\n", path.c_str());
}

rtabmap::Transform se2(double x, double y, double th)
{
  return rtabmap::Transform((float)x, (float)y, 0.0f, 0.0f, 0.0f, (float)th);
}

// ── tag GTSAM passes: initialize_rtabmap_database() lines 608-834 ─────────
void tag_gtsam(const rtabmap::ParametersMap &parameters, const Links &raw_landmark_links,
               Poses &optimizedPoses, Links &links, Poses &tag_poses)
{
  const double landmark_var_linear  = g_lm_var_lin;
  const double landmark_var_angular = g_lm_var_ang;
  constexpr double prior_var_linear_base = 0.001;
  constexpr double prior_var_angular_base= 0.001;
  constexpr int    n_reopt_passes        = 3;
  constexpr float  kMaxTagStd            = 0.45f;

  Links base_links;
  for (const auto &kv : links)
    if (kv.second.type() != rtabmap::Link::kLandmark) base_links.insert(kv);

  struct TagAccum { float tx=0,ty=0,tz=0, qx=0,qy=0,qz=0,qw=0; int n=0; std::vector<std::array<float,3>> pts; };

  for (int pass = 1; pass <= n_reopt_passes; ++pass) {
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
      if (a.n > 0 && (q.x()*a.qx + q.y()*a.qy + q.z()*a.qz + q.w()*a.qw) < 0.f) q.coeffs() = -q.coeffs();
      a.qx += q.x(); a.qy += q.y(); a.qz += q.z(); a.qw += q.w();
      a.pts.push_back({wt.x(), wt.y(), wt.z()});
      a.n++;
    }
    std::map<int, rtabmap::Transform> tag_world_mean;
    std::map<int, float> tag_std;
    for (auto &[tid, a] : tag_accum) {
      float inv = 1.0f / a.n;
      float norm = std::sqrt(a.qx*a.qx + a.qy*a.qy + a.qz*a.qz + a.qw*a.qw);
      float mx = a.tx*inv, my = a.ty*inv, mz = a.tz*inv;
      tag_world_mean[tid] = rtabmap::Transform(mx, my, mz, a.qx/norm, a.qy/norm, a.qz/norm, a.qw/norm);
      float sx=0, sy=0;
      for (auto &p : a.pts) { sx += (p[0]-mx)*(p[0]-mx); sy += (p[1]-my)*(p[1]-my); }
      tag_std[tid] = std::sqrt((sx + sy) / (2.0f * a.n));
    }

    Links reopt_links = base_links;
    cv::Mat lm_inf = cv::Mat::zeros(6, 6, CV_64FC1);
    lm_inf.at<double>(0,0) = lm_inf.at<double>(1,1) = lm_inf.at<double>(2,2) = 1.0 / landmark_var_linear;
    lm_inf.at<double>(3,3) = lm_inf.at<double>(4,4) = lm_inf.at<double>(5,5) = 1.0 / landmark_var_angular;
    for (const auto &kv : raw_landmark_links) {
      const rtabmap::Link &lk = kv.second;
      int tid = std::abs(lk.to());
      float s = tag_std.count(tid) ? tag_std.at(tid) : 0.0f;
      if (s > kMaxTagStd) continue;
      if (optimizedPoses.find(lk.from()) == optimizedPoses.end()) continue;
      reopt_links.insert({lk.from(), rtabmap::Link(lk.from(), lk.to(), rtabmap::Link::kLandmark, lk.transform(), lm_inf)});
    }
    for (const auto &[tid, mean] : tag_world_mean) {
      float s = tag_std.count(tid) ? tag_std.at(tid) : 0.0f;
      if (s > kMaxTagStd) continue;
      double var_lin = prior_var_linear_base + (double)(s * s);
      double var_ang = prior_var_angular_base + (double)(s * s);
      cv::Mat tag_inf = cv::Mat::zeros(6, 6, CV_64FC1);
      tag_inf.at<double>(0,0) = tag_inf.at<double>(1,1) = tag_inf.at<double>(2,2) = 1.0 / var_lin;
      tag_inf.at<double>(3,3) = tag_inf.at<double>(4,4) = tag_inf.at<double>(5,5) = 1.0 / var_ang;
      reopt_links.insert({-tid, rtabmap::Link(-tid, -tid, rtabmap::Link::kPosePrior, mean, tag_inf)});
    }
    Poses initial_poses;
    for (const auto &[nid, pose] : optimizedPoses) if (nid > 0) initial_poses[nid] = pose;
    for (const auto &[tid, mean] : tag_world_mean) {
      float s = tag_std.count(tid) ? tag_std.at(tid) : 0.0f;
      if (s <= kMaxTagStd) initial_poses[-tid] = mean;
    }
    int root_id = initial_poses.lower_bound(1)->first;
    rtabmap::ParametersMap opt_params = parameters;
    opt_params[rtabmap::Parameters::kOptimizerRobust()] = "true";
    rtabmap::Optimizer *opt = rtabmap::Optimizer::create(opt_params);
    cv::Mat out_cov;
    Poses reoptPoses = opt->optimize(root_id, initial_poses, reopt_links, out_cov);
    delete opt;
    if (reoptPoses.empty()) { fprintf(stderr, "[holdout] tag GTSAM pass %d FAILED\n", pass); break; }
    tag_poses.clear();
    for (const auto &[nid, pose] : reoptPoses) {
      if (nid > 0) optimizedPoses[nid] = pose;
      else tag_poses[nid] = pose;
    }
    links = reopt_links;
    fprintf(stderr, "[holdout] tag GTSAM pass %d ok (%zu poses)\n", pass, reoptPoses.size());
  }
}

// ── clean_scans() ──────────────────────────────────────────────────────────
std::map<int, std::vector<Eigen::Vector3f>> clean_scans(const Poses &optimizedPoses)
{
  constexpr float kZlo = 0.05f, kZhi = 2.00f, kRangeMin = 0.30f, kRangeMax = 2.50f, kRorRadius = 0.05f;
  constexpr int kRorMinNeigh = 12, kMinScanPts = 50;
  std::map<int, std::vector<Eigen::Vector3f>> out;
  for (auto iter = optimizedPoses.lower_bound(1); iter != optimizedPoses.end(); ++iter) {
    auto node_it = g_nodes.find(iter->first);
    if (node_it == g_nodes.end()) continue;
    rtabmap::LaserScan scan;
    node_it->second.sensorData().uncompressData(nullptr, nullptr, &scan);
    if (scan.isEmpty()) continue;
    auto base_pts = rtabmap::util3d::laserScanToPointCloud(scan, scan.localTransform());
    Cloud::Ptr cropped(new Cloud), flat(new Cloud);
    for (const auto &pt : base_pts->points) {
      if (!pcl::isFinite(pt)) continue;
      if (pt.z < kZlo || pt.z > kZhi) continue;
      float r = std::hypot(pt.x, pt.y);
      if (r < kRangeMin || r > kRangeMax) continue;
      cropped->push_back(pt);
      flat->push_back(pcl::PointXYZ(pt.x, pt.y, 0.0f));
    }
    pcl::Indices kept;
    if (!flat->empty()) {
      pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
      ror.setInputCloud(flat);
      ror.setRadiusSearch(kRorRadius);
      ror.setMinNeighborsInRadius(kRorMinNeigh);
      ror.filter(kept);
    }
    if ((int)kept.size() < kMinScanPts) continue;
    auto &v = out[iter->first];
    v.reserve(kept.size());
    for (auto idx : kept) v.push_back(cropped->points[idx].getVector3fMap());
  }
  return out;
}

// ── refine_poses_with_local_map_icp() (metrics/prints removed) ─────────────
void local_map_icp(const rtabmap::ParametersMap &parameters, const std::map<int, std::vector<Eigen::Vector3f>> &scans,
                   Poses &optimizedPoses, Links &links, Poses &tag_poses, const std::string &out_prefix)
{
  constexpr int    kRounds = 2;
  constexpr float  kQueryStep = 0.25f, kQueryYaw = 0.26f;
  constexpr int    kMinQueryPts = 500;
  constexpr float  kSubmapRadius = 2.0f, kMinPathSep = 1.0f;
  constexpr int    kMinSubmapNodes = 3, kMaxSubmapNodes = 30;
  constexpr float  kVoxel = 0.04f;
  constexpr int    kNormalK = 10;
  constexpr float  kOverlapDist = 0.10f, kMinOverlap = 0.30f, kNormalRadius = 0.10f;
  constexpr double kVarLinMin = 0.0004, kVarAngMin = 0.0003;

  auto f3 = parameters.find(rtabmap::Parameters::kRegForce3DoF());
  std::string force3dof = f3 != parameters.end() ? f3->second : "true";
  rtabmap::ParametersMap icp_params;
  icp_params[rtabmap::Parameters::kRegForce3DoF()]                         = force3dof;
  icp_params[rtabmap::Parameters::kIcpVoxelSize()]                         = "0";
  icp_params[rtabmap::Parameters::kIcpPointToPlane()]                      = "true";
  icp_params[rtabmap::Parameters::kIcpPointToPlaneK()]                     = std::to_string(kNormalK);
  icp_params[rtabmap::Parameters::kIcpPointToPlaneRadius()]                = "0";
  icp_params[rtabmap::Parameters::kIcpPointToPlaneMinComplexity()]         = "0.02";
  icp_params[rtabmap::Parameters::kIcpPointToPlaneLowComplexityStrategy()] = "0";
  icp_params[rtabmap::Parameters::kIcpMaxCorrespondenceDistance()]         = "0.10";
  icp_params[rtabmap::Parameters::kIcpReciprocalCorrespondences()]         = "false";
  icp_params[rtabmap::Parameters::kIcpCorrespondenceRatio()]               = "0";
  icp_params[rtabmap::Parameters::kIcpIterations()]                        = "30";
  icp_params[rtabmap::Parameters::kIcpMaxTranslation()]                    = "0.20";
  icp_params[rtabmap::Parameters::kIcpMaxRotation()]                       = "0.10";
  rtabmap::RegistrationIcp icp(icp_params);

  auto voxelize = [&](const Cloud::Ptr &cloud) {
    Cloud::Ptr out(new Cloud);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(kVoxel, kVoxel, kVoxel);
    vg.filter(*out);
    return out;
  };
  auto to_sensor_data = [](const Cloud &cloud) {
    cv::Mat m(1, (int)cloud.size(), CV_32FC3);
    float *p = m.ptr<float>();
    for (const auto &pt : cloud.points) { *p++ = pt.x; *p++ = pt.y; *p++ = pt.z; }
    rtabmap::SensorData d;
    d.setLaserScan(rtabmap::LaserScan(m, (int)cloud.size(), 0.0f, rtabmap::LaserScan::kXYZ));
    return d;
  };

  const Links tag_graph = links;
  for (int round = 1; round <= kRounds; ++round) {
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
    struct Cand { int id; float x, y; };
    std::vector<Cand> cands;
    for (const auto &[nid, pts] : scans) {
      auto pit = optimizedPoses.find(nid);
      if (pit == optimizedPoses.end()) continue;
      cands.push_back({nid, pit->second.x(), pit->second.y()});
    }
    std::vector<int> queries;
    {
      const rtabmap::Transform *last = nullptr;
      for (const auto &c : cands) {
        if ((int)scans.at(c.id).size() < kMinQueryPts) continue;
        const rtabmap::Transform &p = optimizedPoses.at(c.id);
        if (last) {
          rtabmap::Transform d = last->inverse() * p;
          if (std::hypot(d.x(), d.y()) < kQueryStep && std::abs(d.theta()) < kQueryYaw) continue;
        }
        queries.push_back(c.id);
        last = &p;
      }
    }

    Links icp_links;
    int n_ok = 0;
    for (size_t qi = 0; qi < queries.size(); ++qi) {
      const int qid = queries[qi];
      const rtabmap::Transform &Pq = optimizedPoses.at(qid);
      const float qc = cum.at(qid);
      std::vector<std::pair<float, int>> near;
      for (const auto &c : cands) {
        if (std::abs(cum.at(c.id) - qc) < kMinPathSep) continue;
        float d = std::hypot(c.x - Pq.x(), c.y - Pq.y());
        if (d <= kSubmapRadius) near.push_back({d, c.id});
      }
      if ((int)near.size() < kMinSubmapNodes) continue;
      const int ref_id = std::min_element(near.begin(), near.end())->second;
      if ((int)near.size() > kMaxSubmapNodes) {
        std::sort(near.begin(), near.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
        std::vector<std::pair<float, int>> picked;
        for (int k = 0; k < kMaxSubmapNodes; ++k) picked.push_back(near[(size_t)k * near.size() / kMaxSubmapNodes]);
        near.swap(picked);
      }
      Cloud::Ptr sub(new Cloud), qcloud(new Cloud);
      const rtabmap::Transform Pq_inv = Pq.inverse();
      for (const auto &[d, nid] : near) {
        Eigen::Affine3f T = (Pq_inv * optimizedPoses.at(nid)).toEigen3f();
        for (const auto &p : scans.at(nid)) {
          Eigen::Vector3f w = T * p;
          sub->push_back(pcl::PointXYZ(w.x(), w.y(), w.z()));
        }
      }
      for (const auto &p : scans.at(qid)) qcloud->push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
      auto sub_v = voxelize(sub);
      auto q_v = voxelize(qcloud);
      const int n_q = (int)q_v->size();
      if (n_q == 0 || sub_v->empty()) continue;

      rtabmap::RegistrationInfo info;
      rtabmap::Transform T = icp.computeTransformation(to_sensor_data(*sub_v), to_sensor_data(*q_v),
                                                       rtabmap::Transform::getIdentity(), &info);
      if (T.isNull()) continue;

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
        if (kd.nearestKSearch(wp, 1, nn, nd2) <= 0 || nd2[0] > kOverlapDist * kOverlapDist) continue;
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
        const double h = std::sqrt(0.25 * (a - c) * (a - c) + b * b);
        const double lmin = 0.5 * (a + c) - h, lmax = 0.5 * (a + c) + h;
        if (lmax <= 0.0 || lmin > 0.25 * lmax) continue;
        Eigen::Vector2d n = std::abs(b) > 1e-12 ? Eigen::Vector2d(b, lmin - a)
                          : (a < c ? Eigen::Vector2d(1, 0) : Eigen::Vector2d(0, 1));
        n.normalize();
        const double rx = w.x() - T.x(), ry = w.y() - T.y();
        const Eigen::Vector3d J(n.x(), n.y(), n.x() * -ry + n.y() * rx);
        H += J * J.transpose();
      }
      const float overlap = (float)n_in / n_q;
      if (overlap < kMinOverlap) continue;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(H.topLeftCorner<2, 2>());
      const double l_strong = es.eigenvalues()(1);
      if (l_strong <= 0.0) continue;
      Eigen::Matrix3d I3 = ((1.0 / kVarLinMin) / l_strong) * H;
      if (I3(2, 2) > 1.0 / kVarAngMin) {
        const double k = std::sqrt((1.0 / kVarAngMin) / I3(2, 2));
        I3.row(2) *= k;
        I3.col(2) *= k;
      }
      rtabmap::Transform rel = optimizedPoses.at(ref_id).inverse() * Pq * T;
      cv::Mat inf = cv::Mat::zeros(6, 6, CV_64FC1);
      const int idx3[3] = {0, 1, 5};
      for (int r = 0; r < 3; ++r)
        for (int cc = 0; cc < 3; ++cc) inf.at<double>(idx3[r], idx3[cc]) = I3(r, cc);
      inf.at<double>(2, 2) = 1.0 / kVarLinMin;
      inf.at<double>(3, 3) = inf.at<double>(4, 4) = 1.0 / kVarAngMin;
      icp_links.insert({ref_id, rtabmap::Link(ref_id, qid, rtabmap::Link::kLocalSpaceClosure, rel, inf)});
      ++n_ok;
    }
    fprintf(stderr, "[holdout] icp round %d: %d links / %zu queries\n", round, n_ok, queries.size());
    if (n_ok == 0) break;

    Links round_links = tag_graph;
    round_links.insert(icp_links.begin(), icp_links.end());
    std::set<int> referenced;
    for (const auto &kv : round_links) { referenced.insert(kv.second.from()); referenced.insert(kv.second.to()); }
    Poses initial;
    for (const auto &[nid, pose] : optimizedPoses) if (nid > 0 || referenced.count(nid)) initial[nid] = pose;
    for (const auto &[nid, pose] : tag_poses) if (referenced.count(nid)) initial[nid] = pose;
    rtabmap::ParametersMap opt_params = parameters;
    opt_params[rtabmap::Parameters::kOptimizerRobust()] = "true";
    rtabmap::Optimizer *opt = rtabmap::Optimizer::create(opt_params);
    cv::Mat out_cov;
    Poses new_poses = opt->optimize(initial.lower_bound(1)->first, initial, round_links, out_cov);
    delete opt;
    if (new_poses.empty()) { fprintf(stderr, "[holdout] icp round %d re-opt FAILED\n", round); break; }
    for (const auto &[nid, pose] : new_poses) {
      if (nid > 0) optimizedPoses[nid] = pose;
      else tag_poses[nid] = pose;
    }
    links = round_links;
    dump(out_prefix + "_icp_round_" + std::to_string(round) + ".csv", optimizedPoses);
  }
}

}  // namespace

int run(int argc, char **argv)
{
  if (argc < 3) {
    std::cerr << "usage: tag_holdout <db_path> <out_prefix> [--holdout 3,7,12] [--shift -0.15]\n";
    return 1;
  }
  const std::string src_db = argv[1], out_prefix = argv[2];
  // Work on a private copy: Rtabmap::init() writes into the DB it opens (search
  // index, Admin.time_enter) even when closed with databaseSaved=false, and
  // parallel runs on one file then fail with "database is locked".
  const std::string db_path = "/tmp/tag_holdout_" + std::to_string(::getpid()) + ".db";
  std::filesystem::copy_file(src_db, db_path, std::filesystem::copy_options::overwrite_existing);
  struct Cleanup { std::string p; ~Cleanup() { std::error_code ec; std::filesystem::remove(p, ec); } } cleanup{db_path};
  fprintf(stderr, "[holdout] working on copy %s of %s\n", db_path.c_str(), src_db.c_str());
  std::set<int> holdout;
  double shift = 0.0;
  for (int i = 3; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "--holdout") {
      std::stringstream ss(argv[i + 1]);
      std::string tok;
      while (std::getline(ss, tok, ',')) if (!tok.empty()) holdout.insert(std::stoi(tok));
    } else if (k == "--shift") {
      shift = std::stod(argv[i + 1]);
    } else if (k == "--lm-var-lin") {
      g_lm_var_lin = std::stod(argv[i + 1]);
      g_lm_override = true;
    } else if (k == "--lm-var-ang") {
      g_lm_var_ang = std::stod(argv[i + 1]);
      g_lm_override = true;
    }
  }
  fprintf(stderr, "[holdout] tag GTSAM landmark var lin=%.4f ang=%.4f\n", g_lm_var_lin, g_lm_var_ang);

  // ── DB: parameters, raw landmark links, all-node odometry (for twists) ──
  rtabmap::ParametersMap parameters;
  Links raw_landmark_links;
  std::map<int, std::pair<double, rtabmap::Transform>> odom;  // id -> (stamp, odom pose), ALL nodes
  {
    rtabmap::DBDriver *driver = rtabmap::DBDriver::create();
    if (!driver->openConnection(db_path, false, true)) { std::cerr << "cannot open " << db_path << "\n"; return 1; }
    parameters = driver->getLastParameters();
    Links all_raw_links;
    driver->getAllLinks(all_raw_links, true, true);
    for (const auto &kv : all_raw_links)
      if (kv.second.type() == rtabmap::Link::kLandmark && !holdout.count(std::abs(kv.second.to())))
        raw_landmark_links.insert(kv);
    std::set<int> ids;
    driver->getAllNodeIds(ids);
    for (int id : ids) {
      rtabmap::Transform pose, gt;
      int map_id = 0, weight = 0;
      std::string label;
      double stamp = 0.0;
      std::vector<float> vel;
      rtabmap::GPS gps;
      rtabmap::EnvSensors sensors;
      if (driver->getNodeInfo(id, pose, map_id, weight, label, stamp, gt, vel, gps, sensors))
        odom[id] = {stamp, pose};
    }
    driver->closeConnection(false);
    delete driver;
  }
  fprintf(stderr, "[holdout] held out %zu tags; %zu landmark obs kept; shift %+.3f s\n",
          holdout.size(), raw_landmark_links.size(), shift);
  {
    std::ofstream hf(out_prefix + "_heldout.txt");
    for (int t : holdout) hf << t << "\n";
  }

  // ── RTAB-Map graph, exactly as the exporter loads it ───────────────────
  Poses optimizedPoses;
  Links links;
  rtabmap::Rtabmap rtabmap;
  rtabmap.init(parameters, db_path);
  rtabmap.getGraph(optimizedPoses, links, true, true, &g_nodes, true, true, true, true);
  rtabmap.close(false);  // do NOT save back into the input DB
  fprintf(stderr, "[holdout] graph: %zu poses, %zu links, %zu nodes\n", optimizedPoses.size(), links.size(), g_nodes.size());

  // ── Optional re-timing of odometry along the local twist ───────────────
  if (shift != 0.0) {
    std::vector<std::pair<double, int>> by_t;
    for (const auto &[id, so] : odom) by_t.push_back({so.first, id});
    std::sort(by_t.begin(), by_t.end());
    std::map<int, rtabmap::Transform> odom_shifted, delta_tf;
    for (size_t k = 0; k < by_t.size(); ++k) {
      const int id = by_t[k].second;
      const int a = by_t[k == 0 ? 0 : k - 1].second, b = by_t[std::min(by_t.size() - 1, k + 1)].second;
      const double dt = odom.at(b).first - odom.at(a).first;
      rtabmap::Transform d = rtabmap::Transform::getIdentity();
      if (dt > 1e-3 && dt <= 2.0) {
        const rtabmap::Transform rel = odom.at(a).second.inverse() * odom.at(b).second;
        d = se2(rel.x() / dt * shift, rel.y() / dt * shift, rel.theta() / dt * shift);
      }
      delta_tf[id] = d;
      odom_shifted[id] = odom.at(id).second * d;
    }
    for (auto &[nid, P] : optimizedPoses)
      if (nid > 0 && delta_tf.count(nid)) P = P * delta_tf.at(nid);
    int n_relinked = 0;
    for (auto &kv : links) {
      rtabmap::Link &lk = kv.second;
      if (lk.type() != rtabmap::Link::kNeighbor && lk.type() != rtabmap::Link::kNeighborMerged) continue;
      if (!odom_shifted.count(lk.from()) || !odom_shifted.count(lk.to())) continue;
      lk.setTransform(odom_shifted.at(lk.from()).inverse() * odom_shifted.at(lk.to()));
      ++n_relinked;
    }
    fprintf(stderr, "[holdout] shifted %zu node poses, recomputed %d neighbour links\n", delta_tf.size(), n_relinked);
  }

  // ── Remove held-out tags from the graph and re-optimize (RTAB-Map's own
  //    weights).  Skipped when nothing changed, so the no-op run equals the exporter.
  if (!holdout.empty() || shift != 0.0 || g_lm_override) {
    cv::Mat lm_inf = cv::Mat::zeros(6, 6, CV_64FC1);
    lm_inf.at<double>(0,0) = lm_inf.at<double>(1,1) = lm_inf.at<double>(2,2) = 1.0 / g_lm_var_lin;
    lm_inf.at<double>(3,3) = lm_inf.at<double>(4,4) = lm_inf.at<double>(5,5) = 1.0 / g_lm_var_ang;
    // A link touching a held-out tag can have the tag on either end
    // (landmark IDs are negative), so test both.
    auto held = [&](int id) { return id < 0 && holdout.count(-id); };
    Poses initial;
    for (const auto &[nid, P] : optimizedPoses)
      if (!held(nid)) initial[nid] = P;
    Links g_links;
    int n_drop_held = 0, n_drop_missing = 0;
    for (const auto &kv : links) {
      const rtabmap::Link &lk = kv.second;
      if (held(lk.from()) || held(lk.to())) { ++n_drop_held; continue; }
      if (!initial.count(lk.from()) || !initial.count(lk.to())) { ++n_drop_missing; continue; }
      auto it = g_links.insert(kv);
      if (g_lm_override && it->second.type() == rtabmap::Link::kLandmark) it->second.setInfMatrix(lm_inf);
    }
    fprintf(stderr, "[holdout] graph re-opt: %zu links kept, %d dropped (held-out tag), %d dropped (endpoint without pose)\n",
            g_links.size(), n_drop_held, n_drop_missing);
    Poses g;
    try {
      rtabmap::Optimizer *opt = rtabmap::Optimizer::create(parameters);
      cv::Mat cov;
      g = opt->optimize(initial.lower_bound(1)->first, initial, g_links, cov);
      delete opt;
    } catch (const std::exception &e) {
      std::cerr << "[holdout] graph re-opt FAILED (exception): " << e.what() << "\n";
      return 1;
    }
    if (g.empty()) { std::cerr << "[holdout] graph re-opt FAILED\n"; return 1; }
    optimizedPoses = g;
    links = g_links;
  }
  dump(out_prefix + "_rtabmap_graph.csv", optimizedPoses);

  Poses tag_poses;
  tag_gtsam(parameters, raw_landmark_links, optimizedPoses, links, tag_poses);
  dump(out_prefix + "_tag_GTSAM.csv", optimizedPoses);

  const auto scans = clean_scans(optimizedPoses);
  fprintf(stderr, "[holdout] %zu cleaned scans\n", scans.size());
  local_map_icp(parameters, scans, optimizedPoses, links, tag_poses, out_prefix);
  fprintf(stderr, "[holdout] done\n");
  return 0;
}

int main(int argc, char **argv)
{
  try {
    return run(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "[holdout] FAILED (exception): " << e.what() << "\n";
    return 2;
  }
}
