// icp_probe: replays ONE round of WCDatabaseExporter::refine_poses_with_local_map_icp()
// from a dumped pose CSV and logs per-query diagnostics.  Read-only on the DB;
// does not re-optimize anything.
//
// Replicated exactly from src/wc_database_exporter.cpp (ea2695e):
//   clean_scans()  : crop z [0.05,2.0], r [0.30,2.50], 2D ROR r=0.05 k=12, >=50 pts
//   query/submap   : step 0.25 m / 0.26 rad, >=500 pts, submap r 2.0 m, path sep 1.0 m,
//                    3..30 nodes spread over node ID, voxel 0.04, ICP params below,
//                    overlap@0.10 >= 0.30, 2D normal Hessian H (radius 0.10, line test 0.25)
//
// Extras (analysis only):
//   --range-cap R  : keep only points with base-frame range <= R in query AND submap
//                    clouds fed to ICP (query selection still uses uncapped sizes, so
//                    the query set is identical across runs)
//   restarts       : for every converged query, ICP again from 6 offset guesses:
//                    +-off along H's strong axis, +-off along the weak axis, +-yaw_off
//   wide           : ICP again with MaxRotation/MaxTranslation 0.5 (what do the
//                    "limit out of bounds" queries actually want?)
//   fitness        : overlap (fraction of query voxels with a submap voxel <= 0.10 m)
//   inlier RMSE    : RMS of 2D point-to-line distances over inliers with a valid normal
//
// Usage:
//   icp_probe <db_path> <poses.csv> <out_prefix> [--range-cap R] [--off 0.03] [--yaw-off-deg 1.0]
// Writes <out_prefix>_queries.csv and <out_prefix>_restarts.csv.

#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/RegistrationIcp.h>
#include <rtabmap/core/RegistrationInfo.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/util3d.h>

#include <pcl/common/point_tests.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using Cloud = pcl::PointCloud<pcl::PointXYZ>;

namespace {

// Same constants as the exporter.
constexpr float  kZlo = 0.05f, kZhi = 2.00f, kRangeMin = 0.30f, kRangeMax = 2.50f;
constexpr float  kRorRadius = 0.05f;
constexpr int    kRorMinNeigh = 12, kMinScanPts = 50;
constexpr float  kQueryStep = 0.25f, kQueryYaw = 0.26f;
constexpr int    kMinQueryPts = 500;
constexpr float  kSubmapRadius = 2.0f, kMinPathSep = 1.0f;
constexpr int    kMinSubmapNodes = 3, kMaxSubmapNodes = 30;
constexpr float  kVoxel = 0.04f;
constexpr int    kNormalK = 10;
constexpr float  kOverlapDist = 0.10f, kMinOverlap = 0.30f, kNormalRadius = 0.10f;

std::map<int, rtabmap::Transform> read_poses(const std::string &path)
{
  std::map<int, rtabmap::Transform> out;
  std::ifstream f(path);
  std::string line;
  std::getline(f, line);  // header
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    std::string tok;
    std::vector<double> v;
    while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
    if (v.size() < 9 || v[0] <= 0) continue;
    out[(int)v[0]] = rtabmap::Transform((float)v[2], (float)v[3], (float)v[4],
                                        (float)v[5], (float)v[6], (float)v[7], (float)v[8]);
  }
  return out;
}

rtabmap::ParametersMap icp_params(const std::string &force3dof, float max_rot, float max_trans)
{
  rtabmap::ParametersMap p;
  p[rtabmap::Parameters::kRegForce3DoF()]                         = force3dof;
  p[rtabmap::Parameters::kIcpVoxelSize()]                         = "0";
  p[rtabmap::Parameters::kIcpPointToPlane()]                      = "true";
  p[rtabmap::Parameters::kIcpPointToPlaneK()]                     = std::to_string(kNormalK);
  p[rtabmap::Parameters::kIcpPointToPlaneRadius()]                = "0";
  p[rtabmap::Parameters::kIcpPointToPlaneMinComplexity()]         = "0.02";
  p[rtabmap::Parameters::kIcpPointToPlaneLowComplexityStrategy()] = "0";
  p[rtabmap::Parameters::kIcpMaxCorrespondenceDistance()]         = "0.10";
  p[rtabmap::Parameters::kIcpReciprocalCorrespondences()]         = "false";
  p[rtabmap::Parameters::kIcpCorrespondenceRatio()]               = "0";
  p[rtabmap::Parameters::kIcpIterations()]                        = "30";
  p[rtabmap::Parameters::kIcpMaxTranslation()]                    = std::to_string(max_trans);
  p[rtabmap::Parameters::kIcpMaxRotation()]                       = std::to_string(max_rot);
  return p;
}

Cloud::Ptr voxelize(const Cloud::Ptr &cloud)
{
  Cloud::Ptr out(new Cloud);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(cloud);
  vg.setLeafSize(kVoxel, kVoxel, kVoxel);
  vg.filter(*out);
  return out;
}

rtabmap::SensorData to_sensor_data(const Cloud &cloud)
{
  cv::Mat m(1, (int)cloud.size(), CV_32FC3);
  float *p = m.ptr<float>();
  for (const auto &pt : cloud.points) { *p++ = pt.x; *p++ = pt.y; *p++ = pt.z; }
  rtabmap::SensorData d;
  d.setLaserScan(rtabmap::LaserScan(m, (int)cloud.size(), 0.0f, rtabmap::LaserScan::kXYZ));
  return d;
}

std::string short_reason(const std::string &msg)
{
  std::string why = msg.substr(0, msg.find_first_of("0123456789("));
  if (why.empty()) why = "unknown";
  why = why.substr(0, 70);
  std::replace(why.begin(), why.end(), ',', ';');
  while (!why.empty() && why.back() == ' ') why.pop_back();
  return why;
}

struct PostFit {
  float overlap = 0.f;
  int n_in = 0, n_rows = 0;
  double rmse_mm = 0.0;
  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
};

// Overlap, Hessian and inlier RMSE after registration — the H part is the exporter's code.
PostFit post_fit(const Cloud::Ptr &sub_v, const Cloud::Ptr &q_v, const rtabmap::Transform &T)
{
  PostFit r;
  const Eigen::Affine3f Tq = T.toEigen3f();
  pcl::KdTreeFLANN<pcl::PointXYZ> kd;
  kd.setInputCloud(sub_v);
  pcl::Indices nn(1), nbr;
  std::vector<float> nd2(1), nbr_d2;
  double se = 0.0;
  for (const auto &p : q_v->points) {
    const Eigen::Vector3f w = Tq * p.getVector3fMap();
    const pcl::PointXYZ wp(w.x(), w.y(), w.z());
    if (kd.nearestKSearch(wp, 1, nn, nd2) <= 0 || nd2[0] > kOverlapDist * kOverlapDist) continue;
    ++r.n_in;
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
    r.H += J * J.transpose();
    // point-to-line distance to the local wall line through the neighbourhood mean
    const double d = n.x() * (w.x() - mx) + n.y() * (w.y() - my);
    se += d * d;
    ++r.n_rows;
  }
  r.overlap = q_v->empty() ? 0.f : (float)r.n_in / q_v->size();
  r.rmse_mm = r.n_rows ? 1000.0 * std::sqrt(se / r.n_rows) : 0.0;
  return r;
}

}  // namespace

int main(int argc, char **argv)
{
  if (argc < 4) {
    std::cerr << "usage: icp_probe <db_path> <poses.csv> <out_prefix> [--range-cap R] [--off 0.03] [--yaw-off-deg 1.0]\n";
    return 1;
  }
  const std::string db_path = argv[1], poses_path = argv[2], out_prefix = argv[3];
  float range_cap = 0.0f, off = 0.03f, yaw_off = 1.0f * (float)M_PI / 180.0f;
  for (int i = 4; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "--range-cap") range_cap = std::stof(argv[i + 1]);
    else if (k == "--off") off = std::stof(argv[i + 1]);
    else if (k == "--yaw-off-deg") yaw_off = std::stof(argv[i + 1]) * (float)M_PI / 180.0f;
  }

  std::map<int, rtabmap::Transform> poses = read_poses(poses_path);
  fprintf(stderr, "[probe] %zu poses from %s\n", poses.size(), poses_path.c_str());

  // ── Load + clean scans (exactly clean_scans()) ─────────────────────────
  rtabmap::DBDriver *driver = rtabmap::DBDriver::create();
  if (!driver->openConnection(db_path, false, true)) {
    std::cerr << "cannot open " << db_path << "\n";
    return 1;
  }
  const rtabmap::ParametersMap db_params = driver->getLastParameters();
  auto f3 = db_params.find(rtabmap::Parameters::kRegForce3DoF());
  const std::string force3dof = f3 != db_params.end() ? f3->second : "true";

  std::map<int, std::vector<Eigen::Vector3f>> scans;   // cleaned, base frame
  for (const auto &[nid, P] : poses) {
    rtabmap::SensorData data;
    driver->getNodeData(nid, data, false, true, false, false);
    rtabmap::LaserScan scan;
    data.uncompressData(nullptr, nullptr, &scan);
    if (scan.isEmpty()) continue;
    auto base_pts = rtabmap::util3d::laserScanToPointCloud(scan, scan.localTransform());
    Cloud::Ptr cropped(new Cloud), flat(new Cloud);
    for (const auto &pt : base_pts->points) {
      if (!pcl::isFinite(pt)) continue;
      if (pt.z < kZlo || pt.z > kZhi) continue;
      const float r = std::hypot(pt.x, pt.y);
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
    auto &v = scans[nid];
    v.reserve(kept.size());
    for (auto idx : kept) v.push_back(cropped->points[idx].getVector3fMap());
  }
  driver->closeConnection(false);
  delete driver;
  size_t npts = 0;
  for (const auto &[nid, v] : scans) npts += v.size();
  fprintf(stderr, "[probe] %zu cleaned scans, %zu points; force3DoF=%s range_cap=%.2f\n",
          scans.size(), npts, force3dof.c_str(), range_cap);

  rtabmap::RegistrationIcp icp(icp_params(force3dof, 0.10f, 0.20f));
  rtabmap::RegistrationIcp icp_wide(icp_params(force3dof, 0.50f, 0.50f));

  // ── Query / candidate selection (exactly the exporter's round loop) ────
  std::map<int, float> cum;
  {
    float d = 0.0f;
    const rtabmap::Transform *prev = nullptr;
    for (auto it = poses.lower_bound(1); it != poses.end(); ++it) {
      if (prev) d += std::hypot(it->second.x() - prev->x(), it->second.y() - prev->y());
      cum[it->first] = d;
      prev = &it->second;
    }
  }
  struct Cand { int id; float x, y; };
  std::vector<Cand> cands;
  for (const auto &[nid, pts] : scans) {
    auto pit = poses.find(nid);
    if (pit == poses.end()) continue;
    cands.push_back({nid, pit->second.x(), pit->second.y()});
  }
  std::vector<int> queries;
  {
    const rtabmap::Transform *last = nullptr;
    for (const auto &c : cands) {
      if ((int)scans.at(c.id).size() < kMinQueryPts) continue;
      const rtabmap::Transform &p = poses.at(c.id);
      if (last) {
        rtabmap::Transform d = last->inverse() * p;
        if (std::hypot(d.x(), d.y()) < kQueryStep && std::abs(d.theta()) < kQueryYaw) continue;
      }
      queries.push_back(c.id);
      last = &p;
    }
  }
  fprintf(stderr, "[probe] %zu queries, %zu candidates\n", queries.size(), cands.size());

  auto keep = [&](const Eigen::Vector3f &p) {
    return range_cap <= 0.0f || std::hypot(p.x(), p.y()) <= range_cap;
  };

  std::ofstream qf(out_prefix + "_queries.csv"), rf(out_prefix + "_restarts.csv");
  qf << std::setprecision(9);
  rf << std::setprecision(9);
  qf << "qid,x,y,yaw,n_sub_nodes,n_q,n_sub,status,reason,tx,ty,tyaw,overlap,inlier_rmse_mm,n_rows,"
        "pm_inliers_ratio,pm_rms,pm_complexity,pm_correspondences,"
        "l_weak,l_strong,h3_min,h3_mid,h3_max,es_x,es_y,corr_strong_mm,corr_weak_mm,"
        "wide_status,wide_reason,wide_tx,wide_ty,wide_tyaw\n";
  rf << "qid,restart,gx,gy,gyaw,ok,reason,tx,ty,tyaw\n";

  int n_ok = 0;
  for (size_t qi = 0; qi < queries.size(); ++qi) {
    if (qi % 50 == 0) fprintf(stderr, "[probe] %zu / %zu (accepted %d)\n", qi, queries.size(), n_ok);
    const int qid = queries[qi];
    const rtabmap::Transform &Pq = poses.at(qid);
    const float qc = cum.at(qid);

    std::vector<std::pair<float, int>> near;
    for (const auto &c : cands) {
      if (std::abs(cum.at(c.id) - qc) < kMinPathSep) continue;
      float d = std::hypot(c.x - Pq.x(), c.y - Pq.y());
      if (d <= kSubmapRadius) near.push_back({d, c.id});
    }
    auto row_prefix = [&](int nsub, int nq, int nsubpts) {
      std::ostringstream s;
      s << std::setprecision(9) << qid << ',' << Pq.x() << ',' << Pq.y() << ',' << Pq.theta() << ','
        << nsub << ',' << nq << ',' << nsubpts;
      return s.str();
    };
    const std::string empty_tail(24, ',');   // 24 empty fields: tx .. wide_tyaw
    if ((int)near.size() < kMinSubmapNodes) {
      qf << row_prefix((int)near.size(), 0, 0) << ",no_submap," << empty_tail << "\n";
      continue;
    }
    if ((int)near.size() > kMaxSubmapNodes) {
      std::sort(near.begin(), near.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
      std::vector<std::pair<float, int>> picked;
      for (int k = 0; k < kMaxSubmapNodes; ++k) picked.push_back(near[(size_t)k * near.size() / kMaxSubmapNodes]);
      near.swap(picked);
    }

    Cloud::Ptr sub(new Cloud), qcloud(new Cloud);
    const rtabmap::Transform Pq_inv = Pq.inverse();
    for (const auto &[d, nid] : near) {
      Eigen::Affine3f T = (Pq_inv * poses.at(nid)).toEigen3f();
      for (const auto &p : scans.at(nid)) {
        if (!keep(p)) continue;
        const Eigen::Vector3f w = T * p;
        sub->push_back(pcl::PointXYZ(w.x(), w.y(), w.z()));
      }
    }
    for (const auto &p : scans.at(qid))
      if (keep(p)) qcloud->push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
    if (sub->empty() || qcloud->empty()) {
      qf << row_prefix((int)near.size(), 0, 0) << ",no_submap," << empty_tail << "\n";
      continue;
    }
    auto sub_v = voxelize(sub);
    auto q_v = voxelize(qcloud);
    const auto sd_sub = to_sensor_data(*sub_v);
    const auto sd_q = to_sensor_data(*q_v);

    rtabmap::RegistrationInfo info;
    rtabmap::Transform T = icp.computeTransformation(sd_sub, sd_q, rtabmap::Transform::getIdentity(), &info);

    rtabmap::RegistrationInfo winfo;
    rtabmap::Transform Tw = icp_wide.computeTransformation(sd_sub, sd_q, rtabmap::Transform::getIdentity(), &winfo);
    std::ostringstream wide;
    wide << std::setprecision(9);
    if (Tw.isNull()) wide << "rejected," << short_reason(winfo.rejectedMsg) << ",,,";
    else             wide << "ok,," << Tw.x() << ',' << Tw.y() << ',' << Tw.theta();

    const std::string pre = row_prefix((int)near.size(), (int)q_v->size(), (int)sub_v->size());
    if (T.isNull()) {
      qf << pre << ",icp_rejected," << short_reason(info.rejectedMsg) << ",,,,,,,"
         << info.icpInliersRatio << ',' << info.icpRMS << ',' << info.icpStructuralComplexity << ','
         << info.icpCorrespondences << ",,,,,,,,,," << wide.str() << "\n";
      continue;
    }

    const PostFit pf = post_fit(sub_v, q_v, T);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(pf.H.topLeftCorner<2, 2>());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es3(pf.H);
    const double l_weak = std::max(0.0, es.eigenvalues()(0)), l_strong = es.eigenvalues()(1);
    const Eigen::Vector2d e_strong = es.eigenvectors().col(1), e_weak = es.eigenvectors().col(0);
    const Eigen::Vector2d t_corr(T.x(), T.y());
    std::string status = "accepted";
    if (pf.overlap < kMinOverlap) status = "low_overlap";
    else if (l_strong <= 0.0) status = "low_strong";
    else ++n_ok;

    qf << pre << ',' << status << ",," << T.x() << ',' << T.y() << ',' << T.theta() << ','
       << pf.overlap << ',' << pf.rmse_mm << ',' << pf.n_rows << ','
       << info.icpInliersRatio << ',' << info.icpRMS << ',' << info.icpStructuralComplexity << ','
       << info.icpCorrespondences << ','
       << l_weak << ',' << l_strong << ',' << es3.eigenvalues()(0) << ',' << es3.eigenvalues()(1) << ','
       << es3.eigenvalues()(2) << ',' << e_strong.x() << ',' << e_strong.y() << ','
       << 1000.0 * std::abs(t_corr.dot(e_strong)) << ',' << 1000.0 * std::abs(t_corr.dot(e_weak)) << ','
       << wide.str() << "\n";

    // ── Repeatability restarts (converged queries only) ─────────────────
    struct G { const char *name; double x, y, yaw; };
    const G guesses[6] = {
      {"strong+", off * e_strong.x(), off * e_strong.y(), 0.0},
      {"strong-", -off * e_strong.x(), -off * e_strong.y(), 0.0},
      {"weak+", off * e_weak.x(), off * e_weak.y(), 0.0},
      {"weak-", -off * e_weak.x(), -off * e_weak.y(), 0.0},
      {"yaw+", 0.0, 0.0, (double)yaw_off},
      {"yaw-", 0.0, 0.0, -(double)yaw_off},
    };
    for (const auto &g : guesses) {
      rtabmap::RegistrationInfo ri;
      const rtabmap::Transform guess((float)g.x, (float)g.y, 0.0f, 0.0f, 0.0f, (float)g.yaw);
      rtabmap::Transform Tr = icp.computeTransformation(sd_sub, sd_q, guess, &ri);
      rf << qid << ',' << g.name << ',' << g.x << ',' << g.y << ',' << g.yaw << ',';
      if (Tr.isNull()) rf << "0," << short_reason(ri.rejectedMsg) << ",,,\n";
      else             rf << "1,," << Tr.x() << ',' << Tr.y() << ',' << Tr.theta() << "\n";
    }
  }
  fprintf(stderr, "[probe] done: %d accepted of %zu queries\n", n_ok, queries.size());
  return 0;
}
