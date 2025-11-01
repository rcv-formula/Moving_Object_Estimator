// obstacle_detector_node.cpp  (ExactTime 3-way; no timer; MapManager keeps scan+pose+obstacles triplets)
// Inputs : /processed_scan (sensor_msgs/PointCloud2)
//          /detected_obstacles (object_detector/MarkerArrayStamped)
//          /processed_odom (nav_msgs::Odometry)
// Outputs: /static_obstacle (geometry_msgs::PointStamped)
//          /dynamic_obstacle (nav_msgs::Odometry)   // KF-tracked dynamic object
//          /dbscan_clusters (visualization_msgs::MarkerArray)
//          /aligned_obj_history (visualization_msgs::MarkerArray)
//          /current_scan_pcl (sensor_msgs::PointCloud2)
//          /icp_aligned_hist_cloud (sensor_msgs::PointCloud2)
//          /icp_frames_markers (visualization_msgs::MarkerArray)

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <limits>
#include <cmath>
#include <memory>
#include <algorithm>
#include <string>
#include <tuple>
#include <optional>
#include <chrono>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

#include "object_detector/msg/marker_array_stamped.hpp"
using object_detector::msg::MarkerArrayStamped;

#include "map_manager_pair.hpp"
#include "dynamic_obstacle_detector.hpp"
#include "icp_point_to_point.hpp"

// ===== SE(3) 평균 유틸: 파일 전역(클래스 밖) =====
namespace se3_avg {

inline Eigen::Quaterniond averageQuaternions(const std::vector<Eigen::Quaterniond>& qs,
                                             const std::vector<double>& w) {
  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  for (size_t i=0;i<qs.size();++i) {
    Eigen::Vector4d q(qs[i].w(), qs[i].x(), qs[i].y(), qs[i].z());
    A += w[i] * (q * q.transpose());
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(A);
  Eigen::Vector4d qv = es.eigenvectors().col(3);
  Eigen::Quaterniond q(qv[0], qv[1], qv[2], qv[3]);
  if (q.w() < 0) q.coeffs() *= -1.0;
  q.normalize();
  return q;
}

inline Eigen::Matrix4d averageSE3(const std::vector<Eigen::Matrix4d>& Ts,
                                  const std::vector<double>& w) {
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  double wsum = 0.0;
  std::vector<Eigen::Quaterniond> qs; qs.reserve(Ts.size());
  for (size_t i=0;i<Ts.size();++i) {
    t    += w[i] * Ts[i].block<3,1>(0,3);
    wsum += w[i];
    qs.emplace_back(Eigen::Quaterniond(Ts[i].block<3,3>(0,0)));
  }
  if (wsum <= 1e-12) return Eigen::Matrix4d::Identity();
  t /= wsum;
  Eigen::Quaterniond qavg = averageQuaternions(qs, w);
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3,3>(0,0) = qavg.toRotationMatrix();
  T.block<3,1>(0,3) = t;
  return T;
}

} // namespace se3_avg

class ObstacleDetector : public rclcpp::Node {
public:
  ObstacleDetector()
  : Node("obstacle_detector"),
    map_manager_(/*max_deque_size=*/10)
  {
    // ===== Parameters (DBSCAN) =====
    this->declare_parameter<double>("dbscan_eps", 0.3);
    this->declare_parameter<int>("dbscan_min_points", 1);
    this->declare_parameter<bool>("use_weighted_median", false);
    this->declare_parameter<int>("min_candidates_to_process", 1);

    // ===== Parameters (topics) =====
    this->declare_parameter<std::string>("processed_scan_topic", processed_scan_topic_);
    this->declare_parameter<std::string>("detected_markers_topic", detected_markers_topic_);
    this->declare_parameter<std::string>("processed_odom_topic", processed_odom_topic_);
    this->declare_parameter<std::string>("dynamic_odom_topic", dynamic_odom_topic_); // "/dynamic_obstacle"

    // ===== Parameters (ICP) =====
    this->declare_parameter<bool>("icp.enable", true);
    this->declare_parameter<bool>("icp.viz.enable", true);
    this->declare_parameter<int>("icp.max_history_for_icp", 9);
    this->declare_parameter<int>("icp.max_iterations", 5);
    this->declare_parameter<double>("icp.max_corr_dist", 0.2);
    this->declare_parameter<double>("icp.trans_eps", 1e-4);
    this->declare_parameter<double>("icp.fit_eps", 1e-3);
    this->declare_parameter<double>("icp.voxel_leaf", 0.10);
    this->declare_parameter<bool>("icp.use_downsample", true);
    this->declare_parameter<double>("icp.gate.fitness", 0.05);
    this->declare_parameter<double>("icp.gate.dtrans", 0.5);
    this->declare_parameter<double>("icp.gate.drot", 0.35);

    this->declare_parameter<bool>("icp.precheck.enable", true);
    this->declare_parameter<double>("icp.frame_fitness_thresh", 0.2);

    // ===== Parameters (KF for dynamic object) =====
    this->declare_parameter<bool>("use_kalman_filter", true);
    this->declare_parameter<double>("kalman_process_noise", 0.1);
    this->declare_parameter<double>("kalman_measurement_noise", 0.05);
    this->declare_parameter<double>("kf_gate_dist", 0.4);          // [m] gating
    this->declare_parameter<double>("kf_reset_timeout_sec", 0.20);  // [s] ≈ 2 frames at 20Hz

    // ===== Load Parameters =====
    this->get_parameter("dbscan_eps", dbscan_eps_);
    this->get_parameter("dbscan_min_points", dbscan_min_points_);
    this->get_parameter("use_weighted_median", use_weighted_median_);
    this->get_parameter("min_candidates_to_process", min_candidates_to_process_);

    this->get_parameter("processed_scan_topic", processed_scan_topic_);
    this->get_parameter("detected_markers_topic", detected_markers_topic_);
    this->get_parameter("processed_odom_topic", processed_odom_topic_);
    this->get_parameter("dynamic_odom_topic", dynamic_odom_topic_);

    this->get_parameter("icp.enable", icp_enable_);
    this->get_parameter("icp.viz.enable", icp_viz_enable_);
    this->get_parameter("icp.max_history_for_icp", icp_max_history_);
    this->get_parameter("icp.max_iterations", icp_max_iterations_);
    this->get_parameter("icp.max_corr_dist", icp_max_corr_dist_);
    this->get_parameter("icp.trans_eps", icp_trans_eps_);
    this->get_parameter("icp.fit_eps", icp_fit_eps_);
    this->get_parameter("icp.voxel_leaf", icp_voxel_leaf_);
    this->get_parameter("icp.use_downsample", icp_use_downsample_);
    this->get_parameter("icp.gate.fitness", icp_gate_fitness_);
    this->get_parameter("icp.gate.dtrans", icp_gate_dtrans_);
    this->get_parameter("icp.gate.drot", icp_gate_drot_);

    this->get_parameter("icp.precheck.enable", icp_precheck_enable_);
    this->get_parameter("icp.frame_fitness_thresh", icp_frame_fitness_thresh_);

    this->get_parameter("use_kalman_filter", use_kalman_filter_);
    this->get_parameter("kalman_process_noise", kalman_process_noise_);
    this->get_parameter("kalman_measurement_noise", kalman_measurement_noise_);
    this->get_parameter("kf_gate_dist", kf_gate_dist_);
    this->get_parameter("kf_reset_timeout_sec", kf_reset_timeout_sec_);

    // ===== ICP Refiner init =====
    {
      icp_comparator::IcpParams ip;
      ip.max_iterations = icp_max_iterations_;
      ip.max_correspondence_distance = static_cast<float>(icp_max_corr_dist_);
      ip.transformation_epsilon = icp_trans_eps_;
      ip.euclidean_fitness_epsilon = icp_fit_eps_;
      ip.voxel_leaf_size = static_cast<float>(icp_voxel_leaf_);
      ip.use_downsample = icp_use_downsample_;
      ip.reject_far_points = true;
      ip.reject_radius = 5.0f;
      ip.min_points = 50;
      icp_refiner_ = std::make_shared<icp_comparator::IcpPointToPoint>(ip);
    }

    // ===== Publishers =====
    static_pub_     = this->create_publisher<geometry_msgs::msg::PointStamped>("/static_obstacle", 10);
    dynamic_pub_    = this->create_publisher<nav_msgs::msg::Odometry>(dynamic_odom_topic_, 20);
    dbscan_vis_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/dbscan_clusters", 10);
    aligned_history_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/aligned_obj_history", 10);
    current_scan_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/current_scan_pcl", 10);
    icp_aligned_hist_pub_   = this->create_publisher<sensor_msgs::msg::PointCloud2>("/icp_aligned_hist_cloud", 5);
    icp_frames_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/icp_frames_markers", 5);

    // ===== ExactTime 3-way sync =====
    proc_scan_sub_.subscribe(this, processed_scan_topic_.c_str(), rmw_qos_profile_sensor_data);
    det_sub_.subscribe(this, detected_markers_topic_.c_str(), rmw_qos_profile_sensor_data);
    proc_odom_sub_.subscribe(this, processed_odom_topic_.c_str(), rmw_qos_profile_sensor_data);

    using Approx3 = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::PointCloud2, MarkerArrayStamped, nav_msgs::msg::Odometry>;
    sync_proc_det_ = std::make_shared<message_filters::Synchronizer<Approx3>>(Approx3(50), proc_scan_sub_, det_sub_, proc_odom_sub_);
    sync_proc_det_->registerCallback(
      std::bind(&ObstacleDetector::procDetExactCallback, this,
                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
      
  }

private:
  // ===== Types & ICP helpers =====
  using PointT   = pcl::PointXYZI;
  using Cloud    = pcl::PointCloud<PointT>;
  using CloudPtr = Cloud::Ptr;

  static CloudPtr toCloud(const sensor_msgs::msg::PointCloud2& pc2) {
    CloudPtr c(new Cloud);
    pcl::fromROSMsg(pc2, *c);
    return c;
  }

  static Cloud downsampleIf(const Cloud& in, float leaf) {
    if (leaf <= 1e-6f) return in;
    pcl::VoxelGrid<PointT> vg;
    vg.setLeafSize(leaf, leaf, leaf);
    Cloud out;
    vg.setInputCloud(in.makeShared());
    vg.filter(out);
    return out;
  }

  static std::pair<double,double> deltaRT(const Eigen::Matrix4f& T) {
    const Eigen::Vector3f t = T.block<3,1>(0,3);
    const double dtrans = t.head<2>().norm();
    const Eigen::Matrix3f R = T.block<3,3>(0,0);
    const Eigen::AngleAxisf aa(R);
    return {dtrans, std::abs(aa.angle())};
  }

  static Eigen::Matrix4d poseToT(const geometry_msgs::msg::Pose &pose)
  {
    Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    q.normalize();
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = q.toRotationMatrix();
    T(0,3) = pose.position.x; T(1,3) = pose.position.y; T(2,3) = pose.position.z;
    return T;
  }

  inline geometry_msgs::msg::PointStamped
  transformLocalWithPose(const geometry_msgs::msg::PointStamped& p_local,
                         const geometry_msgs::msg::Pose& pose_world) const
  {
    // === base_link → laser 고정 변환 ===
    Eigen::Matrix4d T_base_laser = Eigen::Matrix4d::Identity();
    T_base_laser(0,3) = 0.27;   // x offset
    T_base_laser(1,3) = 0.0;
    T_base_laser(2,3) = 0.11;   // z offset

    // === map → base_link ===
    const Eigen::Matrix4d T_world_base = poseToT(pose_world);

    // === map ← base_link ← laser ===
    const Eigen::Matrix4d T_world_laser = T_world_base * T_base_laser;

    // === 변환 ===
    const Eigen::Vector4d pl(p_local.point.x, p_local.point.y, p_local.point.z, 1.0);
    const Eigen::Vector4d pw = T_world_laser * pl;

    geometry_msgs::msg::PointStamped out;
    out.header = p_local.header;
    out.header.frame_id = "map";  // 전역 좌표계로 변경
    out.point.x = pw.x();
    out.point.y = pw.y();
    out.point.z = pw.z();
    return out;
  }

  static inline void hsvToRgb(double h, double s, double v,
                              float &r, float &g, float &b)
  {
    const double i = std::floor(h * 6.0);
    const double f = h * 6.0 - i;
    const double p = v * (1.0 - s);
    const double q = v * (1.0 - f * s);
    const double t = v * (1.0 - (1.0 - f) * s);
    switch (static_cast<int>(i) % 6) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      case 5: r = v; g = p; b = q; break;
    }
  }

  // ===== DBSCAN =====
  std::vector<std::vector<size_t>>
  performDBSCAN(const std::vector<geometry_msgs::msg::PointStamped>& pts) const
  {
    const size_t N = pts.size();
    if (N == 0) return {};
    std::vector<int> cluster_ids(N, -1);

    const int effective_min_pts = std::max(1, dbscan_min_points_);
    auto distance = [&](size_t i, size_t j) -> double {
      const auto &a = pts[i].point, &b = pts[j].point;
      const double dx = a.x - b.x, dy = a.y - b.y;
      return std::sqrt(dx*dx + dy*dy);
    };
    auto regionQuery = [&](size_t i) -> std::vector<size_t> {
      std::vector<size_t> nbs;
      for (size_t j = 0; j < N; ++j) if (distance(i, j) <= dbscan_eps_) nbs.push_back(j);
      return nbs;
    };

    int cluster_id = 0;
    for (size_t i = 0; i < N; ++i) {
      if (cluster_ids[i] != -1) continue;
      auto neighbors = regionQuery(i);
      if (neighbors.size() < static_cast<size_t>(effective_min_pts)) { cluster_ids[i] = -2; continue; }
      cluster_ids[i] = cluster_id;
      std::vector<size_t> seed = std::move(neighbors);
      for (size_t k = 0; k < seed.size(); ++k) {
        size_t j = seed[k];
        if (cluster_ids[j] == -2) cluster_ids[j] = cluster_id;
        if (cluster_ids[j] != -1) continue;
        cluster_ids[j] = cluster_id;
        auto nbj = regionQuery(j);
        if (nbj.size() >= static_cast<size_t>(effective_min_pts))
          seed.insert(seed.end(), nbj.begin(), nbj.end());
      }
      ++cluster_id;
    }

    std::vector<std::vector<size_t>> clusters(cluster_id);
    for (size_t i = 0; i < N; ++i) if (cluster_ids[i] >= 0) clusters[cluster_ids[i]].push_back(i);
    return clusters;
  }

  std::pair<double, double>
  computeRepresentativePoint(const std::vector<size_t>& cluster,
                             const std::vector<geometry_msgs::msg::PointStamped>& pts) const
  {
    double sum_x = 0.0, sum_y = 0.0;
    for (auto idx : cluster) { sum_x += pts[idx].point.x; sum_y += pts[idx].point.y; }
    const double cx = sum_x / cluster.size();
    const double cy = sum_y / cluster.size();

    const double eps = 1e-3;
    if (!use_weighted_median_) {
      double wx = 0.0, wy = 0.0, tw = 0.0;
      for (auto idx : cluster) {
        const double dx = pts[idx].point.x - cx;
        const double dy = pts[idx].point.y - cy;
        const double w = 1.0 / (std::sqrt(dx*dx + dy*dy) + eps);
        wx += pts[idx].point.x * w;
        wy += pts[idx].point.y * w;
        tw += w;
      }
      return { wx / tw, wy / tw };
    }

    struct Wv { double v; double w; };
    std::vector<Wv> xs, ys; xs.reserve(cluster.size()); ys.reserve(cluster.size());
    double tw = 0.0;
    for (auto idx : cluster) {
      const double dx = pts[idx].point.x - cx;
      const double dy = pts[idx].point.y - cy;
      const double w = 1.0 / (std::sqrt(dx*dx + dy*dy) + eps);
      xs.push_back({pts[idx].point.x, w});
      ys.push_back({pts[idx].point.y, w});
      tw += w;
    }
    auto cmp = [](const Wv&a, const Wv&b){ return a.v < b.v; };
    std::sort(xs.begin(), xs.end(), cmp);
    std::sort(ys.begin(), ys.end(), cmp);
    double c = 0.0; double mx = xs.front().v;
    for (const auto &e : xs) { c += e.w; if (c >= tw/2.0) { mx = e.v; break; } }
    c = 0.0; double my = ys.front().v;
    for (const auto &e : ys) { c += e.w; if (c >= tw/2.0) { my = e.v; break; } }
    return {mx, my};
  }

  // ===== [추가] 프레임 축 마커(Arrow) 생성 =====
  visualization_msgs::msg::Marker makeAxisLine(const std::string& ns, int id,
                                               const Eigen::Matrix4d& T,
                                               const std::string& frame_id,
                                               const rclcpp::Time& stamp,
                                               float r, float g, float b) {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame_id;
    m.header.stamp = stamp;
    m.ns = ns;
    m.id = id;
    m.type = visualization_msgs::msg::Marker::ARROW;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = 0.02; // shaft diameter
    m.scale.y = 0.04; // head diameter
    m.scale.z = 0.04; // head length
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.9f;
    geometry_msgs::msg::Point p0, p1;
    p0.x = T(0,3); p0.y = T(1,3); p0.z = T(2,3);
    const Eigen::Vector3d ex = T.block<3,3>(0,0) * Eigen::Vector3d::UnitX();
    p1.x = p0.x + 0.3*ex.x();
    p1.y = p0.y + 0.3*ex.y();
    p1.z = p0.z + 0.3*ex.z();
    m.points.push_back(p0);
    m.points.push_back(p1);
    m.lifetime = rclcpp::Duration::from_seconds(0.2);
    return m;
  }

  // ===== ExactTime 3-way callback: Predict → (meas) Update → Publish =====
  void procDetExactCallback(
      const sensor_msgs::msg::PointCloud2::ConstSharedPtr& processed_scan,
      const MarkerArrayStamped::ConstSharedPtr& det_msg,
      const nav_msgs::msg::Odometry::ConstSharedPtr& odom_proc)
  {
    // (0) visualize current scan
    current_scan_pub_->publish(*processed_scan);
    

    // (1) markers → points
    std::vector<geometry_msgs::msg::PointStamped> frame_points;
    frame_points.reserve(det_msg->markers.size());
    for (const auto &m : det_msg->markers) {
      if (m.action != visualization_msgs::msg::Marker::ADD) continue;
      geometry_msgs::msg::PointStamped pt;
      pt.header = det_msg->header;
      pt.point  = m.pose.position;
      frame_points.push_back(pt);
    }
    
    auto pose_used = odom_proc->pose.pose;

    // if (frame_points.size() < static_cast<size_t>(std::max(1, min_candidates_to_process_))) {
    //   visualization_msgs::msg::MarkerArray arr;
    //   visualization_msgs::msg::Marker del;
    //   del.header = processed_scan->header;
    //   del.action = visualization_msgs::msg::Marker::DELETEALL;
    //   arr.markers.push_back(del);
    //   dbscan_vis_pub_->publish(arr);

    //   auto scan_copy_empty = std::make_shared<sensor_msgs::msg::PointCloud2>(*processed_scan);
    //   map_manager_.addScanObjWithPose(scan_copy_empty, {}, pose_used);

    //   return;
    // }

    // (2) DBSCAN → centers
    auto clusters = performDBSCAN(frame_points);
    std::vector<geometry_msgs::msg::PointStamped> centers;
    centers.reserve(std::max<size_t>(1, clusters.size()));
    if (!clusters.empty()) {
      for (const auto &c : clusters) {
        auto [cx, cy] = computeRepresentativePoint(c, frame_points);
        geometry_msgs::msg::PointStamped p;
        p.header = frame_points[c.front()].header;
        p.point.x = cx; p.point.y = cy; p.point.z = 0.0;
        centers.push_back(p);
      }
    } else {
      for (const auto &pt : frame_points) centers.push_back(pt);
    }

    // (3) align history to current
    const auto triplets = map_manager_.snapshot_triplets();
    std::vector<std::vector<geometry_msgs::msg::Point>> aligned_frames;
    aligned_frames.reserve(triplets.size());

    const Eigen::Matrix4d T_world_curr = poseToT(pose_used);
    const Eigen::Matrix4d T_curr_world = T_world_curr.inverse();

    CloudPtr curr = toCloud(*processed_scan);
    Cloud concat_aligned_curr;

    for (const auto &tr : triplets) {
      const auto &scan_hist      = std::get<0>(tr);
      const auto &pose_hist      = std::get<2>(tr);
      const auto &obs_local_hist = std::get<1>(tr);

      const Eigen::Matrix4d T_world_hist = poseToT(pose_hist);
      const Eigen::Matrix4d T_curr_hist  = T_curr_world * T_world_hist;
      
      //icp 보정 수행 -> return fitness
      double fitness = std::numeric_limits<double>::infinity();
      Eigen::Matrix4f icp_pose =
          icp_refiner_->refine(/*target=*/scan_hist, /*source=*/curr,
                               T_curr_hist.cast<float>(), &fitness);

      if(fitness > icp_gate_fitness_) continue;

      Cloud aligned_in_curr;
      pcl::transformPointCloud(*scan_hist, aligned_in_curr, icp_pose);
      concat_aligned_curr += aligned_in_curr;

      if (icp_aligned_hist_pub_) {
        sensor_msgs::msg::PointCloud2 out;
        pcl::toROSMsg(concat_aligned_curr, out);
        out.header = processed_scan->header; 
        icp_aligned_hist_pub_->publish(out);
      }

      std::vector<geometry_msgs::msg::Point> pts_curr;
      pts_curr.reserve(obs_local_hist.size());
      for (const auto &ps : obs_local_hist) {
        Eigen::Vector4d pl(ps.point.x, ps.point.y, ps.point.z, 1.0);
        Eigen::Vector4d pc = icp_pose.cast<double>() * pl;
        geometry_msgs::msg::Point q; q.x = pc.x(); q.y = pc.y(); q.z = pc.z();
        pts_curr.push_back(q);
      }

      aligned_frames.push_back(std::move(pts_curr));
    }

    detector_.publishAlignedFramesMarkers(
      aligned_frames,
      processed_scan->header.frame_id,
      processed_scan->header.stamp,
      aligned_history_markers_pub_,
      0.06, 0.1
    );
    
    // (4) update triplet
    {
      auto scan_copy = std::make_shared<sensor_msgs::msg::PointCloud2>(*processed_scan);
      map_manager_.addScanObjWithPose(scan_copy, centers, pose_used);
    }

    // (5) footprint classification
    enum Label { UNKNOWN=0, STATIC=1, DYNAMIC=2 };
    std::vector<Label> labels(centers.size(), UNKNOWN);
    for (size_t i = 0; i < centers.size(); ++i) {
      std::vector<geometry_msgs::msg::Point> footprint;
      double span = 0.0;
      const int dyn = detector_.classifyDynamicByFootprint(
        centers[i].point, aligned_frames,
        /*eps=*/0.2, /*minPts=*/2, /*search_radius=*/0.4,
        /*exclude_current=*/false, /*motion_thresh=*/0.2,
        &footprint, &span);

      detector_.visualizeFootprint(
        footprint, dyn,
        processed_scan->header.frame_id, static_cast<int>(i),
        processed_scan->header.stamp, aligned_history_markers_pub_);

      labels[i] = static_cast<Label>(dyn);
    }

    int num_static = 0, num_dynamic = 0, num_unknown = 0;
    for (auto lb : labels) {
      if (lb == STATIC)      ++num_static;
      else if (lb == DYNAMIC)++num_dynamic;
      else                   ++num_unknown;
    }
    RCLCPP_INFO(this->get_logger(),
                "[CLF] centers=%zu  static=%d  dynamic=%d  unknown=%d",
                centers.size(), num_static, num_dynamic, num_unknown);

    // (6) DBSCAN viz
    {
      visualization_msgs::msg::MarkerArray marr;
      visualization_msgs::msg::Marker del;
      del.header = processed_scan->header;
      del.action = visualization_msgs::msg::Marker::DELETEALL;
      marr.markers.push_back(del);

      const std::string frame_id = processed_scan->header.frame_id;
      const auto stamp = processed_scan->header.stamp;

      for (size_t i=0; i<clusters.size(); ++i) {
        visualization_msgs::msg::Marker pts;
        pts.header.frame_id = frame_id; pts.header.stamp = stamp;
        pts.ns = "dbscan_points"; pts.id = static_cast<int>(i);
        pts.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        pts.action = visualization_msgs::msg::Marker::ADD;
        pts.scale.x = 0.06; pts.scale.y = 0.06; pts.scale.z = 0.06;

        float r=1,g=1,b=1; hsvToRgb((i % 12) / 12.0, 0.9, 0.95, r, g, b);
        pts.color.r = r; pts.color.g = g; pts.color.b = b; pts.color.a = 0.9f;

        for (auto idx : clusters[i]) {
          geometry_msgs::msg::Point p; p.x = frame_points[idx].point.x; p.y = frame_points[idx].point.y; p.z = 0.0;
          pts.points.push_back(p);
        }
        pts.lifetime = rclcpp::Duration::from_seconds(0.2);
        marr.markers.push_back(pts);
      }

      for (size_t i=0; i<centers.size(); ++i) {
        visualization_msgs::msg::Marker c;
        c.header.frame_id = frame_id; c.header.stamp = stamp;
        c.ns = "dbscan_centers"; c.id = 1000 + static_cast<int>(i);
        c.type = visualization_msgs::msg::Marker::SPHERE;
        c.action = visualization_msgs::msg::Marker::ADD;
        c.pose.position = centers[i].point;
        c.scale.x = 0.15; c.scale.y = 0.15; c.scale.z = 0.15;

        if (labels[i] == DYNAMIC)       { c.color.r = 1.0f; c.color.g = 0.1f; c.color.b = 0.1f; c.color.a = 0.95f; }
        else if (labels[i] == STATIC) { c.color.r = 0.1f; c.color.g = 0.4f; c.color.b = 1.0f; c.color.a = 0.95f; }
        else { c.color.r = 0.4f; c.color.g = 0.4f; c.color.b = 0.4f; c.color.a = 0.95f; }
        c.lifetime = rclcpp::Duration::from_seconds(0.2);
        marr.markers.push_back(c);
      }
      dbscan_vis_pub_->publish(marr);
    }

    // (7) publish nearest static
    if (!centers.empty()) {
      const geometry_msgs::msg::PointStamped* best_static = nullptr;
      double best_static_d2 = std::numeric_limits<double>::infinity();
      for (size_t i=0; i<centers.size(); ++i) {
        if (labels[i] != STATIC) continue;
        const double d2 = centers[i].point.x*centers[i].point.x + centers[i].point.y*centers[i].point.y;
        if (d2 < best_static_d2) { best_static_d2 = d2; best_static = &centers[i]; }
      }
      if (best_static) {
        auto best_map = transformLocalWithPose(*best_static, pose_used);
        static_pub_->publish(best_map);
      }
    }

    // (8) Predict → Association by KF-pred → Update/Init-or-Reinit → Publish
    if (use_kalman_filter_) {
      const rclcpp::Time stamp(processed_scan->header.stamp);

      // --- Predict with dt from last_kf_time_ ---
      kfPredict(stamp);

      // --- 현재 프레임의 DYNAMIC center들을 map 좌표로 변환해 후보 수집 ---
      struct DynMeas { double x, y; double d2; size_t i; };
      std::vector<DynMeas> dyn_candidates;
      dyn_candidates.reserve(centers.size());

      for (size_t i=0; i<centers.size(); ++i) {
        if (labels[i] != DYNAMIC) continue;
        auto p_map = transformLocalWithPose(centers[i], pose_used);
        DynMeas dm; dm.x = p_map.point.x; dm.y = p_map.point.y; dm.i = i;
        if (kalman_initialized_) {
          const double dx = dm.x - kf_state_[0];
          const double dy = dm.y - kf_state_[1];
          dm.d2 = dx*dx + dy*dy;            // KF 예측 위치 기준 연관 거리
        } else {
          dm.d2 = dm.x*dm.x + dm.y*dm.y;    // 초기화 전: 원점 기준(또는 차량 기준)
        }
        dyn_candidates.push_back(dm);
      }

      bool have_meas = false;
      double meas_x = 0.0, meas_y = 0.0;

      if (!dyn_candidates.empty()) {
        auto best = std::min_element(dyn_candidates.begin(), dyn_candidates.end(),
                                     [](const DynMeas& a, const DynMeas& b){ return a.d2 < b.d2; });
        const double best_dist = std::sqrt(best->d2);

        if (kalman_initialized_) {
          // 기존 트랙이 있고, 게이트 내면 → 연속 업데이트
          if (best_dist <= kf_gate_dist_) {
            meas_x = best->x; meas_y = best->y; have_meas = true;
          } else {
            // 게이트 밖: 기존 트랙은 유지/예측만 하고, “새 동적”으로 간주해 새로 시작
            kfInit(best->x, best->y, stamp);
            publishDynamicOdom(stamp); // init 직후 1회 발행
            have_meas = false;         // 이번 프레임은 init만 수행
          }
        } else {
          // 트랙이 없으면 가장 가까운 후보로 init
          kfInit(best->x, best->y, stamp);
          publishDynamicOdom(stamp);   // init 직후 1회 발행
          have_meas = false;           // 안정성을 위해 이번 프레임은 update 생략(원하면 true로 바꾸어 즉시 update 가능)
        }
      }

      // --- Update (있을 때만) ---
      if (have_meas) {
        const bool was_init = kalman_initialized_;
        kfUpdatePosition(meas_x, meas_y, stamp);  // 내부 gate 재확인 포함
        last_kf_update_time_ = stamp;
        kf_miss_count_ = 0;

        if (!was_init && kalman_initialized_) {
          publishDynamicOdom(stamp);
          RCLCPP_INFO(this->get_logger(), "KF initialized at (%.2f, %.2f).", kf_state_[0], kf_state_[1]);
        }
      } else {
        // 이번 프레임에서 측정 업데이트가 없었음 → 미검출 처리(타임아웃 시 reset)
        if (kalman_initialized_) {
          ++kf_miss_count_;
          const double dt_since_update = (stamp - last_kf_update_time_).seconds();
          if (dt_since_update >= kf_reset_timeout_sec_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
              "KF reset by timeout: no update for %.0f ms (>= %.0f ms).",
              1000.0*dt_since_update, 1000.0*kf_reset_timeout_sec_);
            kfReset();
          }
        }
      }

      // --- Publish at each callback if initialized ---
      if (kalman_initialized_) publishDynamicOdom(stamp);
    }
  }

  // ====== KF utilities ======
  void kfInit(double meas_x, double meas_y, const rclcpp::Time& stamp) {
    kf_state_[0] = meas_x;   // px
    kf_state_[1] = meas_y;   // py
    kf_state_[2] = 0.0;      // vx
    kf_state_[3] = 0.0;      // vy
    for (int r=0;r<4;++r) for (int c=0;c<4;++c) kf_P_[r][c] = 0.0;
    kf_P_[0][0] = 1.0;  kf_P_[1][1] = 1.0;
    kf_P_[2][2] = 10.0; kf_P_[3][3] = 10.0;
    last_kf_time_ = stamp;
    last_kf_update_time_ = stamp;
    kalman_initialized_ = true;

    prev_x_ = meas_x; prev_y_ = meas_y;
    prev_heading_ = 0.0;
    has_prev_position_ = false;
    kf_miss_count_ = 0;
  }

  void kfPredict(const rclcpp::Time& now) {
    if (!kalman_initialized_) return;
    const double dt = (now - last_kf_time_).seconds();
    if (dt <= 0.0) return;

    // CV model
    kf_state_[0] += kf_state_[2] * dt;
    kf_state_[1] += kf_state_[3] * dt;

    const double q = kalman_process_noise_;
    kf_P_[0][0] += q; kf_P_[1][1] += q; kf_P_[2][2] += q; kf_P_[3][3] += q;

    last_kf_time_ = now;
  }

  void kfPredictPublishOnly(const builtin_interfaces::msg::Time& stamp_msg) {
    if (!kalman_initialized_ || !use_kalman_filter_) return;
    rclcpp::Time stamp(stamp_msg);
    kfPredict(stamp);
    publishDynamicOdom(stamp);
    // timeout reset (no measurement this frame)
    const double dt_since_update = (stamp - last_kf_update_time_).seconds();
    if (dt_since_update >= kf_reset_timeout_sec_) kfReset();
  }

  bool passGate(double meas_x, double meas_y) const {
    if (!kalman_initialized_) return true;
    const double dx = meas_x - kf_state_[0];
    const double dy = meas_y - kf_state_[1];
    return std::hypot(dx,dy) <= kf_gate_dist_;
  }

  void kfUpdatePosition(double meas_x, double meas_y, const rclcpp::Time& stamp) {
    if (!kalman_initialized_) {
      kfInit(meas_x, meas_y, stamp);
      return;
    }
    if (!passGate(meas_x, meas_y)) {
      // gate fail → keep prediction only
      return;
    }

    const double yx = meas_x - kf_state_[0];
    const double yy = meas_y - kf_state_[1];

    const double Rm = kalman_measurement_noise_;
    const double Sx = kf_P_[0][0] + Rm;
    const double Sy = kf_P_[1][1] + Rm;

    const double Kx = kf_P_[0][0] / Sx;
    const double Ky = kf_P_[1][1] / Sy;

    kf_state_[0] += Kx * yx;
    kf_state_[1] += Ky * yy;

    // nudge velocity
    const double vel_gain = 0.1;
    kf_state_[2] += vel_gain * (yx);
    kf_state_[3] += vel_gain * (yy);

    kf_P_[0][0] *= (1.0 - Kx);
    kf_P_[1][1] *= (1.0 - Ky);

    last_kf_time_ = stamp;
  }

  void kfReset() {
    kalman_initialized_ = false;
    for (int r=0;r<4;++r) for (int c=0;c<4;++c) kf_P_[r][c] = 0.0;
    kf_state_[0]=kf_state_[1]=kf_state_[2]=kf_state_[3]=0.0;
    has_prev_position_ = false;
    prev_heading_ = 0.0;
    kf_miss_count_ = 0;
  }

  static double smoothHeading(double prev_heading, double heading_pos, double heading_vel, double alpha=0.5) {
    double raw = 0.5*(heading_pos + heading_vel);
    double delta = raw - prev_heading;
    while (delta >  M_PI) delta -= 2.0*M_PI;
    while (delta < -M_PI) delta += 2.0*M_PI;
    return prev_heading + alpha * delta;
  }

  void publishDynamicOdom(const rclcpp::Time& stamp) {
    double heading_pos = prev_heading_;
    if (has_prev_position_) {
      const double dx = kf_state_[0] - prev_x_;
      const double dy = kf_state_[1] - prev_y_;
      if (std::hypot(dx,dy) > 1e-3) heading_pos = std::atan2(dy,dx);
    }
    const double spd = std::hypot(kf_state_[2], kf_state_[3]);
    const double heading_vel = (spd>1e-3) ? std::atan2(kf_state_[3], kf_state_[2]) : heading_pos;

    const double yaw = smoothHeading(prev_heading_, heading_pos, heading_vel, 0.5);
    prev_heading_ = yaw;
    prev_x_ = kf_state_[0];
    prev_y_ = kf_state_[1];
    has_prev_position_ = true;

    const double sz = std::sin(0.5*yaw);
    const double cz = std::cos(0.5*yaw);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";
    odom.child_frame_id  = "dynamic_obj";
    odom.pose.pose.position.x = kf_state_[0];
    odom.pose.pose.position.y = kf_state_[1];
    odom.pose.pose.position.z = 0.0;
    odom.pose.pose.orientation.x = 0.0;
    odom.pose.pose.orientation.y = 0.0;
    odom.pose.pose.orientation.z = sz;
    odom.pose.pose.orientation.w = cz;
    odom.twist.twist.linear.x = kf_state_[2];
    odom.twist.twist.linear.y = kf_state_[3];
    odom.twist.twist.linear.z = 0.0;
    dynamic_pub_->publish(odom);
  }

  // ===== members =====
  // pubs
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr   static_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr            dynamic_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr dbscan_vis_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr aligned_history_markers_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr      current_scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr      icp_aligned_hist_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr icp_frames_markers_pub_;

  // subs (ExactTime 3-way)
  std::string processed_scan_topic_{"/processed_scan"};
  std::string detected_markers_topic_{"/detected_obstacles"};
  std::string processed_odom_topic_{"/processed_odom"};
  std::string dynamic_odom_topic_{"/dynamic_obstacle"};
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2>  proc_scan_sub_;
  message_filters::Subscriber<MarkerArrayStamped>             det_sub_;
  message_filters::Subscriber<nav_msgs::msg::Odometry>        proc_odom_sub_;
  std::shared_ptr< message_filters::Synchronizer<
  message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::PointCloud2, MarkerArrayStamped, nav_msgs::msg::Odometry>>> sync_proc_det_;

  // params (DBSCAN)
  double dbscan_eps_{0.3};
  int    dbscan_min_points_{1};
  int    min_candidates_to_process_{1};
  bool   use_weighted_median_{false};

  // SINGLE manager
  MapManager map_manager_;

  // helpers
  DynamicObjectDetector detector_;

  // ICP
  bool   icp_enable_{true};
  bool   icp_viz_enable_{true};
  int    icp_max_history_{10};
  int    icp_max_iterations_{2};
  double icp_max_corr_dist_{0.1};
  double icp_trans_eps_{1e-5};
  double icp_fit_eps_{1e-4};
  double icp_voxel_leaf_{0.10};
  bool   icp_use_downsample_{true};
  double icp_gate_fitness_{0.02};
  double icp_gate_dtrans_{0.5};
  double icp_gate_drot_{0.35};
  std::shared_ptr<icp_comparator::IcpPointToPoint> icp_refiner_;

  // precheck
  bool   icp_precheck_enable_{true};
  double icp_frame_fitness_thresh_{0.2};

  // ICP viz state
  bool last_icp_ok_{false};
  Eigen::Matrix4f last_Ticp_{Eigen::Matrix4f::Identity()};
  std::vector<Eigen::Matrix4d> last_T_curr_hist_;
  std_msgs::msg::Header last_curr_header_;
  int last_icp_used_hist_count_{0};

  // ===== KF state =====
  bool   use_kalman_filter_{true};
  bool   kalman_initialized_{false};
  double kf_state_[4]{0,0,0,0};     // [px,py,vx,vy]
  double kf_P_[4][4]{{0}};          // covariance
  double kalman_process_noise_{0.1};
  double kalman_measurement_noise_{0.05};
  double kf_gate_dist_{0.4};
  double kf_reset_timeout_sec_{0.20};
  rclcpp::Time last_kf_time_{0,0,RCL_ROS_TIME};
  rclcpp::Time last_kf_update_time_{0,0,RCL_ROS_TIME};
  int    kf_miss_count_{0};

  // heading smoothing cache
  double prev_x_{0.0}, prev_y_{0.0};
  double prev_heading_{0.0};
  bool   has_prev_position_{false};
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObstacleDetector>());
  rclcpp::shutdown();
  return 0;
}
