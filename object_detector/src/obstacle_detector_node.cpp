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
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/color_rgba.hpp>
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
#include <cstdint>
#include <unordered_map>

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
#include "wall_map/wall.hpp"

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

    const std::string default_track_csv_path =
      ament_index_cpp::get_package_share_directory("object_detector") + "/track/0120_track.csv";
    this->declare_parameter<std::string>("track_csv_path", default_track_csv_path);
    this->declare_parameter<bool>("track_csv_has_header", true);
    this->declare_parameter<bool>("only_static", true);

    // ===== Parameters (static wall-map filtering) =====
    this->declare_parameter<bool>("wall_map.enable", true);
    this->declare_parameter<bool>("wall_map.use_config", true);
    this->declare_parameter<std::string>("wall_map.config_path", "config.yaml");
    this->declare_parameter<std::string>("wall_map.yaml_path", "0120.yaml");
    this->declare_parameter<bool>("wall_map.unknown_occupied", true);
    this->declare_parameter<int>("wall_map.occupied_threshold", 50);
    this->declare_parameter<bool>("wall_map.reject_out_of_map", false);

    // ===== Parameters (dynamic/static classification) =====
    this->declare_parameter<double>("dynamic_classification.match_gate", 0.6);
    this->declare_parameter<int>("dynamic_classification.min_history_frames", 2);
    this->declare_parameter<double>("dynamic_classification.static_thresh", 0.10);
    this->declare_parameter<double>("dynamic_classification.dynamic_thresh", 0.30);
    this->declare_parameter<double>("dynamic_classification.unknown_hold_sec", 1.2);
    this->declare_parameter<double>("dynamic_classification.unknown_hold_gate", 0.8);
    this->declare_parameter<bool>("dynamic_classification.smoothing_enable", true);
    this->declare_parameter<double>("dynamic_classification.smoothing_gate", 0.7);
    this->declare_parameter<double>("dynamic_classification.smoothing_max_age_sec", 1.0);
    this->declare_parameter<double>("dynamic_classification.static_memory_max_age_sec", 60.0);
    this->declare_parameter<int>("dynamic_classification.dynamic_confirm_frames", 2);
    this->declare_parameter<int>("dynamic_classification.static_confirm_frames", 2);
    this->declare_parameter<bool>("dynamic_classification.use_icp_aligned_history", false);
    this->declare_parameter<int>("dynamic_classification.static_to_dynamic_confirm_frames", 8);
    this->declare_parameter<double>("dynamic_classification.static_lock_break_dist", 0.35);

    // ===== Parameters (planner bridge) =====
    this->declare_parameter<int>("planner_bridge.static_confirm_frames", 2);
    this->declare_parameter<int>("planner_bridge.dynamic_confirm_frames", 2);

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

    this->get_parameter("track_csv_path", track_csv_path_);
    this->get_parameter("track_csv_has_header", track_csv_has_header_);
    this->get_parameter("only_static", only_static_);

    this->get_parameter("wall_map.enable", wall_map_enabled_);
    this->get_parameter("wall_map.use_config", wall_map_use_config_);
    this->get_parameter("wall_map.config_path", wall_map_config_path_);
    this->get_parameter("wall_map.yaml_path", wall_map_yaml_path_);
    this->get_parameter("wall_map.unknown_occupied", wall_map_unknown_occupied_);
    this->get_parameter("wall_map.occupied_threshold", wall_map_occupied_threshold_);
    this->get_parameter("wall_map.reject_out_of_map", wall_map_reject_out_of_map_);

    this->get_parameter("dynamic_classification.match_gate", dyn_match_gate_);
    this->get_parameter("dynamic_classification.min_history_frames", dyn_min_history_frames_);
    this->get_parameter("dynamic_classification.static_thresh", dyn_static_thresh_);
    this->get_parameter("dynamic_classification.dynamic_thresh", dyn_dynamic_thresh_);
    this->get_parameter("dynamic_classification.unknown_hold_sec", dyn_unknown_hold_sec_);
    this->get_parameter("dynamic_classification.unknown_hold_gate", dyn_unknown_hold_gate_);
    this->get_parameter("dynamic_classification.smoothing_enable", dyn_smoothing_enable_);
    this->get_parameter("dynamic_classification.smoothing_gate", dyn_smoothing_gate_);
    this->get_parameter("dynamic_classification.smoothing_max_age_sec", dyn_smoothing_max_age_sec_);
    this->get_parameter("dynamic_classification.static_memory_max_age_sec", dyn_static_memory_max_age_sec_);
    this->get_parameter("dynamic_classification.dynamic_confirm_frames", dyn_dynamic_confirm_frames_);
    this->get_parameter("dynamic_classification.static_confirm_frames", dyn_static_confirm_frames_);
    this->get_parameter("dynamic_classification.use_icp_aligned_history", dyn_use_icp_aligned_history_);
    this->get_parameter("dynamic_classification.static_to_dynamic_confirm_frames", dyn_static_to_dynamic_confirm_frames_);
    this->get_parameter("dynamic_classification.static_lock_break_dist", dyn_static_lock_break_dist_);
    this->get_parameter("planner_bridge.static_confirm_frames", obj_flag_static_confirm_frames_);
    this->get_parameter("planner_bridge.dynamic_confirm_frames", obj_flag_dynamic_confirm_frames_);

    map_manager_.set_max_deque_size(
      std::max<std::size_t>(10, static_cast<std::size_t>(std::max(1, icp_max_history_))));

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

    detector_.loadTrackCsvPCL(track_csv_path_, track_csv_has_header_);
    initializeWallMapFilter();

    // ===== Publishers =====
    static_pub_     = this->create_publisher<geometry_msgs::msg::PointStamped>("/static_obstacle", 10);
    dynamic_pub_    = this->create_publisher<nav_msgs::msg::Odometry>(dynamic_odom_topic_, 20);
    obj_flag_pub_   = this->create_publisher<geometry_msgs::msg::PointStamped>("/obj_flag", 10);
    dbscan_vis_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/dbscan_clusters", 10);
    aligned_history_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/aligned_obj_history", 10);
    current_scan_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/current_scan_pcl", 10);
    icp_aligned_hist_pub_   = this->create_publisher<sensor_msgs::msg::PointCloud2>("/icp_aligned_hist_cloud", 5);
    icp_frames_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/icp_frames_markers", 5);
    wall_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/track_wall_marker", 1);

    // ===== Detection node check =====
    RCLCPP_DEBUG(this->get_logger(), "ObstacleDetector node mode:: %s", only_static_ ? "ONLY STATIC":"DYNAMIC+STATIC");

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

  std::string track_csv_path_;
  bool track_csv_has_header_{true};

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
    const double eps_sq = dbscan_eps_ * dbscan_eps_;
    const double cell_size = std::max(dbscan_eps_, 1e-9);

    auto cellCoord = [cell_size](double v) -> int {
      return static_cast<int>(std::floor(v / cell_size));
    };

    auto cellKey = [](int cx, int cy) -> std::uint64_t {
      return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cx)) << 32) |
             static_cast<std::uint32_t>(cy);
    };

    std::unordered_map<std::uint64_t, std::vector<size_t>> grid;
    grid.reserve(N * 2);
    for (size_t i = 0; i < N; ++i) {
      const int cx = cellCoord(pts[i].point.x);
      const int cy = cellCoord(pts[i].point.y);
      grid[cellKey(cx, cy)].push_back(i);
    }

    auto distanceSquared = [&](size_t i, size_t j) -> double {
      const auto &a = pts[i].point, &b = pts[j].point;
      const double dx = a.x - b.x, dy = a.y - b.y;
      return dx*dx + dy*dy;
    };
    auto regionQuery = [&](size_t i) -> std::vector<size_t> {
      std::vector<size_t> nbs;
      const int cx = cellCoord(pts[i].point.x);
      const int cy = cellCoord(pts[i].point.y);

      for (int ox = -1; ox <= 1; ++ox) {
        for (int oy = -1; oy <= 1; ++oy) {
          const auto it = grid.find(cellKey(cx + ox, cy + oy));
          if (it == grid.end()) continue;

          for (size_t j : it->second) {
            if (distanceSquared(i, j) <= eps_sq) nbs.push_back(j);
          }
        }
      }
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
    
    if(wall_pub_){
      auto marker = detector_.makeWallLineMarker();
      wall_pub_->publish(marker);
    }

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
    std::vector<std::vector<size_t>> center_clusters;
    center_clusters.reserve(std::max<size_t>(1, clusters.size()));

    if (!clusters.empty()) {
      for (const auto &c : clusters) {
        auto [cx, cy] = computeRepresentativePoint(c, frame_points);
        geometry_msgs::msg::PointStamped p;
        p.header = frame_points[c.front()].header;
        p.point.x = cx; p.point.y = cy; p.point.z = 0.0;

        const auto p_world = transformLocalWithPose(p, pose_used);
        geometry_msgs::msg::Point obs = p_world.point;
        if (isValidObstacleInMap(obs)) {
          // 유효(벽 안쪽) → centers에 push_back
          centers.push_back(p);
          center_clusters.push_back(c);
        }
      }
    } else {
      for (size_t idx = 0; idx < frame_points.size(); ++idx) {
        const auto &pt = frame_points[idx];
        const auto p_world = transformLocalWithPose(pt, pose_used);
        geometry_msgs::msg::Point obs = p_world.point;
        if (isValidObstacleInMap(obs)) {
          centers.push_back(pt);
          center_clusters.push_back({idx});
        }
      }
    }

    std::vector<std::vector<geometry_msgs::msg::Point>> aligned_frames;

    // (3) align history to current only when dynamic classification needs it
    if (!only_static_) {
      const auto triplets = map_manager_.snapshot_triplets();
      const size_t max_history = static_cast<size_t>(std::max(0, icp_max_history_));
      const size_t start_idx =
        (max_history > 0 && triplets.size() > max_history) ? (triplets.size() - max_history) : 0;
      aligned_frames.reserve(triplets.size() - start_idx);

      const Eigen::Matrix4d T_world_curr = poseToT(pose_used);
      const Eigen::Matrix4d T_curr_world = T_world_curr.inverse();

      CloudPtr curr = toCloud(*processed_scan);
      Cloud concat_aligned_curr;

      for (size_t ti = start_idx; ti < triplets.size(); ++ti) {
        const auto &tr = triplets[ti];
        const auto &scan_hist      = std::get<0>(tr);
        const auto &pose_hist      = std::get<2>(tr);
        const auto &obs_local_hist = std::get<1>(tr);

        const Eigen::Matrix4d T_world_hist = poseToT(pose_hist);
        const Eigen::Matrix4d T_curr_hist  = T_curr_world * T_world_hist;

        const Eigen::Matrix4f odom_pose = T_curr_hist.cast<float>();
        Eigen::Matrix4f icp_pose = odom_pose;
        Eigen::Matrix4f obstacle_history_pose = odom_pose;
        double fitness = 0.0;
        const bool need_icp = icp_enable_ && (icp_viz_enable_ || dyn_use_icp_aligned_history_);
        if (need_icp) {
          fitness = std::numeric_limits<double>::infinity();
          const Eigen::Matrix4f refined_pose = icp_refiner_->refine(
              /*target=*/scan_hist, /*source=*/curr,
              odom_pose, &fitness);

          const Eigen::Matrix4f correction = refined_pose * odom_pose.inverse();
          const auto [dtrans, drot] = deltaRT(correction);
          const bool accept_icp =
            std::isfinite(fitness) &&
            fitness <= icp_gate_fitness_ &&
            dtrans <= icp_gate_dtrans_ &&
            drot <= icp_gate_drot_;

          if (accept_icp) {
            icp_pose = refined_pose;
            if (dyn_use_icp_aligned_history_) {
              obstacle_history_pose = refined_pose;
            }
          } else {
            icp_pose = odom_pose;
          }
        }

        if (icp_viz_enable_) {
          Cloud aligned_in_curr;
          pcl::transformPointCloud(*scan_hist, aligned_in_curr, icp_pose);
          concat_aligned_curr += aligned_in_curr;
        }

        std::vector<geometry_msgs::msg::Point> pts_curr;
        pts_curr.reserve(obs_local_hist.size());
        for (const auto &ps : obs_local_hist) {
          Eigen::Vector4d pl(ps.point.x, ps.point.y, ps.point.z, 1.0);
          Eigen::Vector4d pc = obstacle_history_pose.cast<double>() * pl;
          geometry_msgs::msg::Point q; q.x = pc.x(); q.y = pc.y(); q.z = pc.z();
          pts_curr.push_back(q);
        }

        aligned_frames.push_back(std::move(pts_curr));
      }

      if (icp_viz_enable_ && icp_aligned_hist_pub_ && !concat_aligned_curr.empty()) {
        sensor_msgs::msg::PointCloud2 out;
        pcl::toROSMsg(concat_aligned_curr, out);
        out.header = processed_scan->header;
        icp_aligned_hist_pub_->publish(out);
      }

      if (icp_viz_enable_) {
        detector_.publishAlignedFramesMarkers(
          aligned_frames,
          processed_scan->header.frame_id,
          processed_scan->header.stamp,
          aligned_history_markers_pub_,
          0.06, 0.1
        );
      }
    }
    
    // (4) update triplet
    {
      auto scan_copy = std::make_shared<sensor_msgs::msg::PointCloud2>(*processed_scan);
      map_manager_.addScanObjWithPose(scan_copy, centers, pose_used);
    }

    // (5) footprint classification
    enum Label { UNKNOWN=0, STATIC=1, DYNAMIC=2 };
    std::vector<Label> labels(centers.size(), UNKNOWN);
    for (size_t i = 0; i < centers.size(); ++i) {
      
      if(only_static_) {
        labels[i] = STATIC;
        continue;
      }

      std::vector<geometry_msgs::msg::Point> footprint;
      double span = 0.0;
      const int dyn = detector_.classifyDynamicByFootprint(
        centers[i].point, aligned_frames,
        dyn_static_thresh_, dyn_min_history_frames_, dyn_match_gate_,
        /*exclude_current=*/false, dyn_dynamic_thresh_,
        &footprint, &span);

      detector_.visualizeFootprint(
        footprint, dyn,
        processed_scan->header.frame_id, static_cast<int>(i),
        processed_scan->header.stamp, aligned_history_markers_pub_);

      labels[i] = static_cast<Label>(dyn);
    }

    if (!only_static_) {
      const rclcpp::Time label_stamp(processed_scan->header.stamp);
      const double hold_gate_sq = dyn_unknown_hold_gate_ * dyn_unknown_hold_gate_;
      bool time_jumped_back = false;

      dynamic_label_memory_.erase(
        std::remove_if(
          dynamic_label_memory_.begin(), dynamic_label_memory_.end(),
          [&](const DynamicLabelMemory & memory) {
            const double age = (label_stamp - memory.stamp).seconds();
            if (age < -0.001) {
              time_jumped_back = true;
              return true;
            }
            return age > dyn_unknown_hold_sec_;
          }),
        dynamic_label_memory_.end());

      if (time_jumped_back) {
        dynamic_label_memory_.clear();
      }

      if (dyn_unknown_hold_sec_ > 0.0 && dyn_unknown_hold_gate_ > 0.0) {
        for (size_t i = 0; i < centers.size(); ++i) {
          if (labels[i] != UNKNOWN) continue;

          const auto p_map = transformLocalWithPose(centers[i], pose_used);
          for (const auto & memory : dynamic_label_memory_) {
            const double dx = p_map.point.x - memory.point.x;
            const double dy = p_map.point.y - memory.point.y;
            if (dx * dx + dy * dy <= hold_gate_sq) {
              labels[i] = DYNAMIC;
              break;
            }
          }
        }
      }

      if (dyn_smoothing_enable_ && dyn_smoothing_gate_ > 0.0) {
        const double smooth_gate_sq = dyn_smoothing_gate_ * dyn_smoothing_gate_;

        label_smoothing_memory_.erase(
          std::remove_if(
            label_smoothing_memory_.begin(), label_smoothing_memory_.end(),
            [&](const LabelSmoothingMemory & memory) {
              const double age = (label_stamp - memory.stamp).seconds();
              const bool is_static = memory.label == static_cast<int>(STATIC);
              const double max_age = is_static ?
                std::max(dyn_smoothing_max_age_sec_, dyn_static_memory_max_age_sec_) :
                dyn_smoothing_max_age_sec_;
              return age < -0.001 || age > max_age;
            }),
          label_smoothing_memory_.end());

        const auto find_static_anchor_idx =
          [&](const geometry_msgs::msg::Point & point) -> size_t {
            const double break_dist = std::max(0.0, dyn_static_lock_break_dist_);
            if (break_dist <= 0.0) {
              return label_smoothing_memory_.size();
            }

            const double break_dist_sq = break_dist * break_dist;
            size_t best_idx = label_smoothing_memory_.size();
            double best_d2 = break_dist_sq;
            for (size_t j = 0; j < label_smoothing_memory_.size(); ++j) {
              const auto & memory = label_smoothing_memory_[j];
              if (memory.label != static_cast<int>(STATIC) || !memory.has_static_anchor) {
                continue;
              }

              const double dx = point.x - memory.static_anchor.x;
              const double dy = point.y - memory.static_anchor.y;
              const double d2 = dx * dx + dy * dy;
              if (d2 <= best_d2) {
                best_d2 = d2;
                best_idx = j;
              }
            }
            return best_idx;
          };

        std::vector<bool> matched(label_smoothing_memory_.size(), false);
        for (size_t i = 0; i < centers.size(); ++i) {
          const auto p_map = transformLocalWithPose(centers[i], pose_used);
          Label raw_label = labels[i];
          const size_t locked_static_idx =
            (raw_label == DYNAMIC) ? find_static_anchor_idx(p_map.point) : label_smoothing_memory_.size();
          const bool locked_by_static_anchor =
            locked_static_idx != label_smoothing_memory_.size();
          if (locked_by_static_anchor) {
            raw_label = STATIC;
          }

          size_t best_idx = label_smoothing_memory_.size();
          double best_d2 = smooth_gate_sq;
          for (size_t j = 0; j < label_smoothing_memory_.size(); ++j) {
            if (matched[j]) continue;
            const double dx = p_map.point.x - label_smoothing_memory_[j].point.x;
            const double dy = p_map.point.y - label_smoothing_memory_[j].point.y;
            double d2 = dx * dx + dy * dy;
            const auto & memory = label_smoothing_memory_[j];
            if (memory.label == static_cast<int>(STATIC) && memory.has_static_anchor) {
              const double adx = p_map.point.x - memory.static_anchor.x;
              const double ady = p_map.point.y - memory.static_anchor.y;
              d2 = std::min(d2, adx * adx + ady * ady);
            }
            if (d2 <= best_d2) {
              best_d2 = d2;
              best_idx = j;
            }
          }

          if (best_idx == label_smoothing_memory_.size()) {
            LabelSmoothingMemory memory;
            memory.point = p_map.point;
            memory.stamp = label_stamp;
            memory.label = (raw_label == DYNAMIC) ? static_cast<int>(DYNAMIC) :
              (locked_by_static_anchor ? static_cast<int>(STATIC) : static_cast<int>(UNKNOWN));
            memory.pending_label = (raw_label == UNKNOWN) ? static_cast<int>(UNKNOWN) : static_cast<int>(raw_label);
            memory.pending_count = (raw_label == UNKNOWN) ? 0 : 1;
            if (locked_by_static_anchor) {
              memory.static_anchor = label_smoothing_memory_[locked_static_idx].static_anchor;
              memory.has_static_anchor = true;
            } else if (memory.label == static_cast<int>(STATIC)) {
              memory.static_anchor = p_map.point;
              memory.has_static_anchor = true;
            }
            label_smoothing_memory_.push_back(memory);
            matched.push_back(true);
            labels[i] = static_cast<Label>(memory.label);
            continue;
          }

          auto & memory = label_smoothing_memory_[best_idx];
          matched[best_idx] = true;

          if (locked_by_static_anchor) {
            memory.label = static_cast<int>(STATIC);
            memory.static_anchor = label_smoothing_memory_[locked_static_idx].static_anchor;
            memory.has_static_anchor = true;
            memory.pending_label = static_cast<int>(UNKNOWN);
            memory.pending_count = 0;
            labels[i] = STATIC;
            memory.point = p_map.point;
            memory.stamp = label_stamp;
            continue;
          }

          if (memory.label == static_cast<int>(STATIC) && raw_label == DYNAMIC) {
            if (!memory.has_static_anchor) {
              memory.static_anchor = memory.point;
              memory.has_static_anchor = true;
            }
            const double dx = p_map.point.x - memory.static_anchor.x;
            const double dy = p_map.point.y - memory.static_anchor.y;
            const double break_dist = std::max(0.0, dyn_static_lock_break_dist_);
            if (dx * dx + dy * dy <= break_dist * break_dist) {
              labels[i] = STATIC;
              memory.point = p_map.point;
              memory.stamp = label_stamp;
              memory.pending_label = static_cast<int>(UNKNOWN);
              memory.pending_count = 0;
              continue;
              }
          }

          if (raw_label != UNKNOWN && raw_label != static_cast<Label>(memory.label)) {
            if (memory.pending_label == static_cast<int>(raw_label)) {
              ++memory.pending_count;
            } else {
              memory.pending_label = static_cast<int>(raw_label);
              memory.pending_count = 1;
            }

            int confirm_frames = (raw_label == DYNAMIC) ?
              std::max(1, dyn_dynamic_confirm_frames_) :
              std::max(1, dyn_static_confirm_frames_);
            if (memory.label == static_cast<int>(STATIC) && raw_label == DYNAMIC) {
              confirm_frames = std::max(confirm_frames, dyn_static_to_dynamic_confirm_frames_);
            }
            if (memory.pending_count >= confirm_frames) {
              memory.label = static_cast<int>(raw_label);
              memory.pending_label = static_cast<int>(UNKNOWN);
              memory.pending_count = 0;
              if (raw_label == STATIC) {
                memory.static_anchor = p_map.point;
                memory.has_static_anchor = true;
              } else if (raw_label == DYNAMIC) {
                memory.has_static_anchor = false;
              }
            }
          } else if (raw_label != UNKNOWN) {
            memory.label = static_cast<int>(raw_label);
            memory.pending_label = static_cast<int>(UNKNOWN);
            memory.pending_count = 0;
            if (raw_label == STATIC) {
              if (!memory.has_static_anchor) {
                memory.static_anchor = p_map.point;
                memory.has_static_anchor = true;
              }
            } else if (raw_label == DYNAMIC) {
              memory.has_static_anchor = false;
            }
          }

          labels[i] = static_cast<Label>(memory.label);
          memory.point = p_map.point;
          memory.stamp = label_stamp;
        }

        constexpr size_t kMaxLabelSmoothingMemory = 200;
        if (label_smoothing_memory_.size() > kMaxLabelSmoothingMemory) {
          label_smoothing_memory_.erase(
            label_smoothing_memory_.begin(),
            label_smoothing_memory_.begin() +
              static_cast<std::ptrdiff_t>(label_smoothing_memory_.size() - kMaxLabelSmoothingMemory));
        }
      }

      for (size_t i = 0; i < centers.size(); ++i) {
        if (labels[i] != DYNAMIC) continue;

        const auto p_map = transformLocalWithPose(centers[i], pose_used);
        DynamicLabelMemory memory;
        memory.point = p_map.point;
        memory.stamp = label_stamp;
        dynamic_label_memory_.push_back(memory);
      }

      constexpr size_t kMaxDynamicLabelMemory = 200;
      if (dynamic_label_memory_.size() > kMaxDynamicLabelMemory) {
        dynamic_label_memory_.erase(
          dynamic_label_memory_.begin(),
          dynamic_label_memory_.begin() +
            static_cast<std::ptrdiff_t>(dynamic_label_memory_.size() - kMaxDynamicLabelMemory));
      }
    }

    int num_static = 0, num_dynamic = 0, num_unknown = 0;
    for (auto lb : labels) {
      if (lb == STATIC)      ++num_static;
      else if (lb == DYNAMIC)++num_dynamic;
      else                   ++num_unknown;
    }
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "[OBSTACLE_STATUS] centers=%zu  static=%d  dynamic=%d  unknown=%d",
                centers.size(), num_static, num_dynamic, num_unknown);

    const int static_confirm_frames = std::max(1, obj_flag_static_confirm_frames_);
    const int dynamic_confirm_frames = std::max(1, obj_flag_dynamic_confirm_frames_);
    obj_flag_static_count_ = (num_static > 0) ?
      std::min(obj_flag_static_count_ + 1, static_confirm_frames) : 0;
    obj_flag_dynamic_count_ = (num_dynamic > 0) ?
      std::min(obj_flag_dynamic_count_ + 1, dynamic_confirm_frames) : 0;

    const bool confirmed_static = obj_flag_static_count_ >= static_confirm_frames;
    const bool confirmed_dynamic = obj_flag_dynamic_count_ >= dynamic_confirm_frames;

    std::vector<Label> output_labels = labels;
    for (auto & label : output_labels) {
      if (label == STATIC && !confirmed_static) {
        label = UNKNOWN;
      } else if (label == DYNAMIC && !confirmed_dynamic) {
        label = UNKNOWN;
      }
    }

    {
      geometry_msgs::msg::PointStamped flag_msg;
      flag_msg.header.stamp = processed_scan->header.stamp;
      flag_msg.header.frame_id = "map";
      flag_msg.point.x = confirmed_dynamic ? 1.0 : 0.0;
      flag_msg.point.y = confirmed_static ? 1.0 : 0.0;
      flag_msg.point.z = confirmed_static ? 2.0 : (confirmed_dynamic ? 1.0 : 0.0);
      obj_flag_pub_->publish(flag_msg);
    }

    // (6) DBSCAN viz
    {
      visualization_msgs::msg::MarkerArray marr;
      visualization_msgs::msg::Marker del;
      del.header = processed_scan->header;
      del.action = visualization_msgs::msg::Marker::DELETEALL;
      marr.markers.push_back(del);

      const std::string frame_id = processed_scan->header.frame_id;
      const auto stamp = processed_scan->header.stamp;

      auto set_label_color = [](std_msgs::msg::ColorRGBA & color, Label label, float alpha) {
        if (label == DYNAMIC) {
          color.r = 1.0f; color.g = 0.0f; color.b = 0.0f; color.a = alpha;
        } else if (label == STATIC) {
          color.r = 0.1f; color.g = 0.4f; color.b = 1.0f; color.a = alpha;
        } else {
          color.r = 0.4f; color.g = 0.4f; color.b = 0.4f; color.a = alpha;
        }
      };

      for (size_t i=0; i<center_clusters.size() && i<labels.size(); ++i) {
        visualization_msgs::msg::Marker pts;
        pts.header.frame_id = frame_id; pts.header.stamp = stamp;
        pts.ns = "dbscan_points"; pts.id = static_cast<int>(i);
        pts.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        pts.action = visualization_msgs::msg::Marker::ADD;
        pts.scale.x = 0.06; pts.scale.y = 0.06; pts.scale.z = 0.06;
        set_label_color(pts.color, output_labels[i], 0.9f);

        for (auto idx : center_clusters[i]) {
          if (idx >= frame_points.size()) continue;
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

        set_label_color(c.color, output_labels[i], 0.95f);
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
          RCLCPP_DEBUG(this->get_logger(), "KF initialized at (%.2f, %.2f).", kf_state_[0], kf_state_[1]);
        }
      } else {
        // 이번 프레임에서 측정 업데이트가 없었음 → 미검출 처리(타임아웃 시 reset)
        if (kalman_initialized_) {
          ++kf_miss_count_;
          const double dt_since_update = (stamp - last_kf_update_time_).seconds();
          if (dt_since_update >= kf_reset_timeout_sec_) {
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
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

  void initializeWallMapFilter()
  {
    if (!wall_map_enabled_) {
      return;
    }

    static_wall_map_ =
      wall_map::StaticWallMap(wall_map_unknown_occupied_, wall_map_occupied_threshold_);

    const bool loaded = wall_map_use_config_ ?
      static_wall_map_.load_from_config(wall_map_config_path_) :
      static_wall_map_.load_from_yaml(wall_map_yaml_path_);

    if (!loaded) {
      wall_map_enabled_ = false;
      RCLCPP_DEBUG(
        this->get_logger(),
        "wall_map filter disabled: failed to load %s '%s'",
        wall_map_use_config_ ? "config" : "yaml",
        wall_map_use_config_ ? wall_map_config_path_.c_str() : wall_map_yaml_path_.c_str());
      return;
    }

    const auto & map = static_wall_map_.map();
    RCLCPP_DEBUG(
      this->get_logger(),
      "wall_map filter enabled: %ux%u resolution=%.3f source=%s",
      map.info.width,
      map.info.height,
      map.info.resolution,
      wall_map_use_config_ ? wall_map_config_path_.c_str() : wall_map_yaml_path_.c_str());
  }

  bool isValidObstacleInMap(const geometry_msgs::msg::Point & obstacle_in_map) const
  {
    if (!detector_.isObstacleWithinWallPCL(obstacle_in_map)) {
      return false;
    }

    if (!wall_map_enabled_ || !static_wall_map_.has_map()) {
      return true;
    }

    const auto is_wall = static_wall_map_.is_wall_at(obstacle_in_map.x, obstacle_in_map.y);
    if (!is_wall.has_value()) {
      return !wall_map_reject_out_of_map_;
    }

    return !is_wall.value();
  }

  // ===== members =====
  // pubs
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr   static_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr            dynamic_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr   obj_flag_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr dbscan_vis_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr aligned_history_markers_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr      current_scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr      icp_aligned_hist_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr icp_frames_markers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr wall_pub_;

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
  bool   only_static_{false};

  // params (static wall-map filtering)
  bool wall_map_enabled_{true};
  bool wall_map_use_config_{true};
  std::string wall_map_config_path_{"config.yaml"};
  std::string wall_map_yaml_path_{"0120.yaml"};
  bool wall_map_unknown_occupied_{true};
  int wall_map_occupied_threshold_{50};
  bool wall_map_reject_out_of_map_{false};

  // params (dynamic/static classification)
  double dyn_match_gate_{0.6};
  int    dyn_min_history_frames_{2};
  double dyn_static_thresh_{0.10};
  double dyn_dynamic_thresh_{0.30};
  double dyn_unknown_hold_sec_{1.2};
  double dyn_unknown_hold_gate_{0.8};
  bool dyn_smoothing_enable_{true};
  double dyn_smoothing_gate_{0.7};
  double dyn_smoothing_max_age_sec_{1.0};
  double dyn_static_memory_max_age_sec_{60.0};
  int dyn_dynamic_confirm_frames_{2};
  int dyn_static_confirm_frames_{2};
  bool dyn_use_icp_aligned_history_{false};
  int dyn_static_to_dynamic_confirm_frames_{8};
  double dyn_static_lock_break_dist_{0.35};
  int obj_flag_static_confirm_frames_{2};
  int obj_flag_dynamic_confirm_frames_{2};
  int obj_flag_static_count_{0};
  int obj_flag_dynamic_count_{0};

  struct DynamicLabelMemory {
    geometry_msgs::msg::Point point;
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
  };
  std::vector<DynamicLabelMemory> dynamic_label_memory_;

  struct LabelSmoothingMemory {
    geometry_msgs::msg::Point point;
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    int label{0};
    int pending_label{0};
    int pending_count{0};
    geometry_msgs::msg::Point static_anchor;
    bool has_static_anchor{false};
  };
  std::vector<LabelSmoothingMemory> label_smoothing_memory_;

  // SINGLE manager
  MapManager map_manager_;

  // helpers
  DynamicObjectDetector detector_;
  wall_map::StaticWallMap static_wall_map_;

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
