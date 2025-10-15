#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <sstream>
#include <deque>

#include <Eigen/Dense>  // ★ 추가

using namespace std::chrono_literals;

#include "map_manager_pair.hpp"
#include "dynamic_obstacle_detector.hpp"

class ObstacleDetector : public rclcpp::Node {
public:
  ObstacleDetector()
  : Node("obstacle_detector"),
    kalman_initialized_(false),
    has_prev_position_(false),
    prev_heading_(0.0),
    scan_map_manager_(40),
    obj_manager_(10)
  {
    this->declare_parameter<double>("dbscan_eps", 0.3);
    this->declare_parameter<int>("dbscan_min_points", 3);
    this->declare_parameter<bool>("use_weighted_median", false);
    this->declare_parameter<double>("kalman_process_noise", 0.1);
    this->declare_parameter<double>("kalman_measurement_noise", 0.1);
    this->declare_parameter<bool>("use_kalman_filter", true);
    this->declare_parameter<double>("obstacle_timeout", 1.0);
    this->declare_parameter<int>("min_candidates_to_process", 3);
    this->declare_parameter<double>("window_seconds", 0.25);
    this->declare_parameter<std::string>("wall_topic", wall_topic_);
    this->declare_parameter<int>("wall_deque_size", wall_deque_size_);
    this->declare_parameter<double>("voxel_leaf", voxel_leaf_);
    this->declare_parameter<int>("knn_k", knn_k_);
    this->declare_parameter<double>("wall_dist_thresh", wall_dist_thresh_);
    this->declare_parameter<double>("association_gate", gate_dyn_);
    this->declare_parameter<std::string>("scan_topic", scan_topic_);
    this->declare_parameter<std::string>("odom_topic", odom_topic_);

    this->get_parameter("dbscan_eps", dbscan_eps_);
    this->get_parameter("dbscan_min_points", dbscan_min_points_);
    this->get_parameter("use_weighted_median", use_weighted_median_);
    this->get_parameter("kalman_process_noise", kalman_process_noise_);
    this->get_parameter("kalman_measurement_noise", kalman_measurement_noise_);
    this->get_parameter("use_kalman_filter", use_kalman_filter_);
    this->get_parameter("obstacle_timeout", obstacle_timeout_);
    this->get_parameter("min_candidates_to_process", min_candidates_to_process_);
    this->get_parameter("window_seconds", window_seconds_);
    this->get_parameter("wall_topic", wall_topic_);
    this->get_parameter("wall_deque_size", wall_deque_size_);
    this->get_parameter("voxel_leaf", voxel_leaf_);
    this->get_parameter("knn_k", knn_k_);
    this->get_parameter("wall_dist_thresh", wall_dist_thresh_);
    this->get_parameter("association_gate", gate_dyn_);
    this->get_parameter("scan_topic", scan_topic_);
    this->get_parameter("odom_topic", odom_topic_);

    static_pub_      = this->create_publisher<geometry_msgs::msg::PointStamped>("/static_obstacle", 10);
    dynamic_pub_     = this->create_publisher<nav_msgs::msg::Odometry>("/dynamic_obstacle", 20);
    dbscan_vis_pub_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("/dbscan_clusters", 10);
    wall_accum_pub_  = this->create_publisher<sensor_msgs::msg::PointCloud2>("/wall_points_accum", 10);

    current_scan_pub_   = this->create_publisher<sensor_msgs::msg::PointCloud2>("/current_scan_pcl", 10);
    aligned_history_pub_= this->create_publisher<sensor_msgs::msg::PointCloud2>("/aligned_history_scans", 10);
    aligned_history_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/aligned_obj_history", 10);
    
    wall_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      wall_topic_, 50,
      std::bind(&ObstacleDetector::wallCallback, this, std::placeholders::_1));

    marker_sub_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
    "/detected_obstacles", rclcpp::SensorDataQoS(),
    std::bind(&ObstacleDetector::markerCallbackSolo, this, std::placeholders::_1));
    
    odom_only_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(odom_topic_, rclcpp::SensorDataQoS(),
                                      [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { last_odom_ = std::move(msg); });

    scan_sub_.subscribe(this, scan_topic_.c_str(), rmw_qos_profile_sensor_data);
    odom_sub_.subscribe(this, odom_topic_.c_str(), rmw_qos_profile_sensor_data);
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(1000), scan_sub_, odom_sub_);
    sync_->registerCallback(std::bind(&ObstacleDetector::scanSyncCallback, this,
                                      std::placeholders::_1, std::placeholders::_2));

    last_measurement_time_ = rclcpp::Time(0, 0, this->get_clock()->get_clock_type());

    RCLCPP_INFO(this->get_logger(), "ObstacleDetector with Scan-Alignment pipeline started.");
    RCLCPP_INFO(this->get_logger(), "  wall_topic: %s | scan_topic: %s | odom_topic: %s",
                wall_topic_.c_str(), scan_topic_.c_str(), odom_topic_.c_str());
  }

private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::LaserScan, nav_msgs::msg::Odometry>;
  
  void scanSyncCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg,
                        const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg)
  {
    auto local_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    local_cloud->reserve(scan_msg->ranges.size());

    for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
      const double r = scan_msg->ranges[i];
      if (!std::isfinite(r)) continue;
      const double ang = scan_msg->angle_min + i * scan_msg->angle_increment;

      pcl::PointXYZI pt;
      pt.x = static_cast<float>(r * std::cos(ang));
      pt.y = static_cast<float>(r * std::sin(ang));
      pt.z = 0.0f;
      pt.intensity = 255.0f;
      local_cloud->push_back(pt);
    }

    sensor_msgs::msg::PointCloud2 current_cloud_msg;
    pcl::toROSMsg(*local_cloud, current_cloud_msg);
    current_cloud_msg.header = scan_msg->header;
    current_scan_pub_->publish(current_cloud_msg);

    scan_map_manager_.addCloudWithPose(
      std::make_shared<sensor_msgs::msg::PointCloud2>(current_cloud_msg),
      odom_msg->pose.pose
    );

    auto pairs = scan_map_manager_.snapshot_pairs();
    if (pairs.size() < 2) return;

    const auto& current_pose = pairs.back().second;
    Eigen::Matrix4d T_map_curr = MapManager::poseToMatrix(current_pose);
    Eigen::Matrix4d T_curr_map = T_map_curr.inverse();

    pcl::PointCloud<pcl::PointXYZI> all_aligned_scans;
    const size_t frames = pairs.size() - 1;

    for (size_t i = 0; i < frames; ++i) {
      const auto& hist_scan_pcl = pairs[i].first;
      const auto& hist_pose     = pairs[i].second;

      Eigen::Matrix4d T_map_hist  = MapManager::poseToMatrix(hist_pose);
      Eigen::Matrix4d T_curr_hist = T_curr_map * T_map_hist;

      pcl::PointCloud<pcl::PointXYZI> aligned_scan_i;
      pcl::transformPointCloud(*hist_scan_pcl, aligned_scan_i, T_curr_hist.cast<float>());

      float intensity_value = static_cast<float>((frames - 1) - i);
      for (auto& pt : aligned_scan_i.points) pt.intensity = intensity_value;

      all_aligned_scans += aligned_scan_i;
    }

    sensor_msgs::msg::PointCloud2 aligned_history_msg;
    pcl::toROSMsg(all_aligned_scans, aligned_history_msg);
    aligned_history_msg.header.stamp = scan_msg->header.stamp;
    aligned_history_msg.header.frame_id = scan_msg->header.frame_id;
    aligned_history_pub_->publish(aligned_history_msg);
  }

  void hsvToRgb(double h, double s, double v, float &r, float &g, float &b) {
    double i = std::floor(h * 6.0);
    double f = h * 6.0 - i;
    double p = v * (1.0 - s);
    double q = v * (1.0 - f * s);
    double t = v * (1.0 - (1.0 - f) * s);
    switch (static_cast<int>(i) % 6) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      case 5: r = v; g = p; b = q; break;
    }
  }

  void wallCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // 벽 파이프라인은 기존 API 유지 (map_manager_는 기존 addCloud/snapshot 사용)
    // map_manager_.addCloud(msg);

    // auto snap = map_manager_.snapshot_cloud();
    // detector_.rebuild(*snap);

    // auto snap_obj = map_manager_.snapshot_obj();
    // detector_.rebuild_obj(*snap_obj);
  }

  void markerCallbackSolo(const visualization_msgs::msg::MarkerArray::ConstSharedPtr& msg)
  {
    if (msg->markers.empty()) return;

    for (const auto &m : msg->markers) {
      if (m.action != visualization_msgs::msg::Marker::ADD) continue;
      geometry_msgs::msg::PointStamped pt;
      pt.header = m.header;
      pt.point  = m.pose.position;
      candidate_points_.push_back(pt);
    }

    auto latest = to_node_time(msg->markers.front().header.stamp);
    pruneWindow(latest);

    if (candidate_points_.size() < static_cast<size_t>(min_candidates_to_process_)) {
      visualization_msgs::msg::MarkerArray arr;
      visualization_msgs::msg::Marker del;
      del.header.frame_id = "map";
      del.header.stamp = latest;
      del.action = visualization_msgs::msg::Marker::DELETEALL;
      arr.markers.push_back(del);
      dbscan_vis_pub_->publish(arr);
      return;
    }

    auto clusters = performDBSCAN();
    if (clusters.empty()) {
      visualization_msgs::msg::MarkerArray arr;
      visualization_msgs::msg::Marker del;
      del.header.frame_id = "map";
      del.header.stamp = this->now();
      del.action = visualization_msgs::msg::Marker::DELETEALL;
      arr.markers.push_back(del);
      dbscan_vis_pub_->publish(arr);
      return;
    }

    std::vector<geometry_msgs::msg::PointStamped> centers;
    centers.reserve(clusters.size());
    for (const auto &c : clusters) {
      auto [cx, cy] = computeRepresentativePoint(c);
      geometry_msgs::msg::PointStamped p;
      geometry_msgs::msg::Point p2;
      p.header = candidate_points_[c.front()].header;
      p.point.x = cx; p.point.y = cy; p.point.z = 0.0;
      p2.x = cx; p2.y = cy; p2.z = 0.0;
      centers.push_back(p);
    }

    enum Label { UNKNOWN=0, STATIC=1, DYNAMIC=2 };
    std::vector<Label> labels(centers.size(), UNKNOWN);
    std::vector<geometry_msgs::msg::PointStamped> static_list, dynamic_list;
    static_list.reserve(centers.size());
    dynamic_list.reserve(centers.size());

    const auto pairs = obj_manager_.snapshot_obj_pairs();  // 프레임 점벡터 + 당시 pose
    auto aligned_frames = detector_.transformObjHistoryToCurrentFrames(pairs, last_odom_->pose.pose);

    detector_.publishAlignedFramesMarkers(
    aligned_frames,
    "laser",           
    this->now(),          
    aligned_history_markers_pub_,  
    0.06,                  // 점 크기
    0.1                   // lifetime(sec)
    );

    for (size_t i = 0; i < centers.size(); ++i) {
      const auto &pt = centers[i];
      std::vector<geometry_msgs::msg::Point> footprint;
      double span = 0.0;

      int dyn = detector_.classifyDynamicByFootprint(
        pt.point, aligned_frames,
        /*eps=*/0.2, /*minPts=*/2, /*search_radius=*/0.6, /*exclude_current=*/false,
        /*motion_thresh=*/0.2,
        &footprint, &span
      );

      if (dyn == 0) {
        labels[i] = STATIC;
        static_list.push_back(pt);
      } else if (dyn == 1){
        labels[i] = DYNAMIC;
        dynamic_list.push_back(pt);
      }

    }
    // obj들을 추가 -> obj_mapmanager
    obj_manager_.addObjWithPose(centers, last_odom_->pose.pose);

    RCLCPP_INFO(this->get_logger(),"Cluster :: %ld",centers.size());
    RCLCPP_INFO(this->get_logger(),"Static :: %ld",static_list.size());
    RCLCPP_INFO(this->get_logger(),"Dynamic :: %ld",dynamic_list.size());
    RCLCPP_INFO(this->get_logger(),
            "Obj frames: %zu (last frame points: %zu)",
            pairs.size(),
            pairs.empty() ? 0u : pairs.back().first.size());

    {
      visualization_msgs::msg::MarkerArray marr;
      visualization_msgs::msg::Marker del;
      del.header.frame_id = centers.front().header.frame_id.empty() ? "map" : centers.front().header.frame_id;
      del.header.stamp = this->now();
      del.action = visualization_msgs::msg::Marker::DELETEALL;
      marr.markers.push_back(del);

      const std::string frame_id = del.header.frame_id;
      rclcpp::Time stamp = this->now();
      int id_base = 0;

      for (size_t i=0; i<clusters.size(); ++i) {
        visualization_msgs::msg::Marker pts;
        pts.header.frame_id = frame_id;
        pts.header.stamp = stamp;
        pts.ns = "dbscan_points";
        pts.id = id_base + static_cast<int>(i);
        pts.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        pts.action = visualization_msgs::msg::Marker::ADD;
        pts.scale.x = 0.06; pts.scale.y = 0.06; pts.scale.z = 0.06;

        float r=1,g=1,b=1;
        hsvToRgb((i % 12) / 12.0, 0.9, 0.95, r, g, b);
        pts.color.r = r; pts.color.g = g; pts.color.b = b; pts.color.a = 0.9f;

        for (auto idx : clusters[i]) {
          geometry_msgs::msg::Point p;
          p.x = candidate_points_[idx].point.x;
          p.y = candidate_points_[idx].point.y;
          p.z = 0.0;
          pts.points.push_back(p);
        }
        pts.lifetime = rclcpp::Duration::from_seconds(0.2);
        marr.markers.push_back(pts);
      }

      for (size_t i=0; i<centers.size(); ++i) {
        visualization_msgs::msg::Marker c;
        c.header.frame_id = frame_id; c.header.stamp = stamp;
        c.ns = "dbscan_centers"; c.id = 1000 + static_cast<int>(i);
        c.type = visualization_msgs::msg::Marker::SPHERE; c.action = visualization_msgs::msg::Marker::ADD;
        c.pose.position = centers[i].point;
        c.scale.x = 0.15; c.scale.y = 0.15; c.scale.z = 0.15;
        switch (labels[i]) {
          case STATIC:  c.color.r = 0.0f; c.color.g = 0.4f; c.color.b = 1.0f; c.color.a = 0.95f; break;
          case DYNAMIC: c.color.r = 1.0f; c.color.g = 0.1f; c.color.b = 0.1f; c.color.a = 0.95f; break;
          case UNKNOWN: c.color.r = 0.6f; c.color.g = 0.6f; c.color.b = 0.6f; c.color.a = 0.8f;  break;
        }
        c.lifetime = rclcpp::Duration::from_seconds(0.2);
        marr.markers.push_back(c);

        visualization_msgs::msg::Marker t;
        t.header.frame_id = frame_id; t.header.stamp = stamp;
        t.ns = "dbscan_text"; t.id = 2000 + static_cast<int>(i);
        t.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING; t.action = visualization_msgs::msg::Marker::ADD;
        t.pose.position = centers[i].point; t.pose.position.z += 0.18;
        t.scale.z = 0.18;
        t.color.r = 1.0f; t.color.g = 1.0f; t.color.b = 1.0f; t.color.a = 0.95f;
        char buf[64];
        const char *lbl = (labels[i]==STATIC? "S" : (labels[i]==DYNAMIC? "D" : "U"));
        std::snprintf(buf, sizeof(buf), "#%zu %s (N=%zu)", i, lbl, clusters[i].size());
        t.text = buf;
        t.lifetime = rclcpp::Duration::from_seconds(0.2);
        marr.markers.push_back(t);
      }
      dbscan_vis_pub_->publish(marr);
    }

    if (!centers.empty()) {
      const geometry_msgs::msg::PointStamped* best = nullptr;
      double best_d2 = std::numeric_limits<double>::infinity();
      for (size_t i=0;i<centers.size();++i) {
        if (labels[i] != STATIC) continue;
        double dx = centers[i].point.x;
        double dy = centers[i].point.y;
        double d2 = dx*dx + dy*dy;
        if (d2 < best_d2) { best_d2 = d2; best = &centers[i]; }
      }
      if (best) {
        geometry_msgs::msg::PointStamped ps;
        ps.header.frame_id = centers.front().header.frame_id.empty() ? "map" : centers.front().header.frame_id;
        ps.header.stamp    = best->header.stamp;
        ps.point = best->point;
        static_pub_->publish(ps);
      }
    }

    if (!centers.empty()) {
      std::vector<geometry_msgs::msg::PointStamped> dyns;
      for (size_t i=0;i<centers.size();++i) if (labels[i]==DYNAMIC) dyns.push_back(centers[i]);
      if (!dyns.empty()) {
        const geometry_msgs::msg::PointStamped* meas = chooseDynamicMeasurement(dyns);
        if (meas) {
          processAndUpdateMeasurementWithPoint(meas->point.x, meas->point.y, meas->header.stamp);
        } else {
          if (kalman_initialized_) {
            const rclcpp::Time current_stamp = centers.front().header.stamp;
            const double since_last_meas = (current_stamp - last_measurement_time_).seconds();
            if (since_last_meas > obstacle_timeout_) {
              RCLCPP_WARN(this->get_logger(),
                          "Dynamic obstacle track lost (timeout: %.2fs). Resetting KF.",
                          obstacle_timeout_);
              kalman_initialized_ = false;
            }
          }
        }
      }
    }
  }

  void pruneWindow(const rclcpp::Time& now) {
    const double T = window_seconds_;
    while (!candidate_points_.empty()) {
      const auto &st = candidate_points_.front().header.stamp;
      rclcpp::Time ts = (st.sec == 0 && st.nanosec == 0) ? now : rclcpp::Time(st);
      if ((now - ts).seconds() <= T) break;
      candidate_points_.pop_front();
    }
  }

  const geometry_msgs::msg::PointStamped* chooseDynamicMeasurement(
      const std::vector<geometry_msgs::msg::PointStamped>& list) const
  {
    if (list.empty()) return nullptr;

    const geometry_msgs::msg::PointStamped* best = nullptr;
    double best_score = std::numeric_limits<double>::infinity();

    if (!kalman_initialized_) {
      for (const auto &m : list) {
        double d2 = m.point.x*m.point.x + m.point.y*m.point.y;
        if (d2 < best_score) { best_score = d2; best = &m; }
      }
      return best;
    }

    const double px = kf_state_[0], py = kf_state_[1];
    for (const auto &m : list) {
      double dx = m.point.x - px, dy = m.point.y - py;
      double d  = std::hypot(dx, dy);
      if (d < best_score) { best_score = d; best = &m; }
    }
    if (best && best_score > gate_dyn_) return nullptr;
    return best;
  }

  std::vector<std::vector<size_t>> performDBSCAN() {
    size_t N = candidate_points_.size();
    std::vector<int> cluster_ids(N, -1);
    auto distance = [this](size_t i, size_t j) -> double {
      double dx = candidate_points_[i].point.x - candidate_points_[j].point.x;
      double dy = candidate_points_[i].point.y - candidate_points_[j].point.y;
      return std::sqrt(dx * dx + dy * dy);
    };
    auto regionQuery = [this, N, &distance](size_t i) -> std::vector<size_t> {
      std::vector<size_t> neighbors;
      for (size_t j = 0; j < N; j++) if (distance(i, j) <= dbscan_eps_) neighbors.push_back(j);
      return neighbors;
    };
    int cluster_id = 0;
    for (size_t i = 0; i < N; i++) {
      if (cluster_ids[i] != -1) continue;
      auto neighbors = regionQuery(i);
      if (neighbors.size() < static_cast<size_t>(dbscan_min_points_)) { cluster_ids[i] = -2; continue; }
      cluster_ids[i] = cluster_id;
      std::vector<size_t> seed_set = std::move(neighbors);
      for (size_t idx = 0; idx < seed_set.size(); idx++) {
        size_t j = seed_set[idx];
        if (cluster_ids[j] == -2) cluster_ids[j] = cluster_id;
        if (cluster_ids[j] != -1) continue;
        cluster_ids[j] = cluster_id;
        auto neighbors_j = regionQuery(j);
        if (neighbors_j.size() >= static_cast<size_t>(dbscan_min_points_)) {
          seed_set.insert(seed_set.end(), neighbors_j.begin(), neighbors_j.end());
        }
      }
      cluster_id++;
    }
    std::vector<std::vector<size_t>> clusters(cluster_id);
    for (size_t i = 0; i < N; i++) if (cluster_ids[i] >= 0) clusters[cluster_ids[i]].push_back(i);
    return clusters;
  }

  std::pair<double, double> computeRepresentativePoint(const std::vector<size_t>& cluster) {
    double sum_x = 0.0, sum_y = 0.0;
    for (size_t idx : cluster) { sum_x += candidate_points_[idx].point.x; sum_y += candidate_points_[idx].point.y; }
    double center_x = sum_x / cluster.size();
    double center_y = sum_y / cluster.size();
    const double epsilon = 1e-3;
    double rep_x = 0.0, rep_y = 0.0;

    if (!use_weighted_median_) {
      double weighted_sum_x = 0.0, weighted_sum_y = 0.0, total_weight = 0.0;
      for (size_t idx : cluster) {
        double dx = candidate_points_[idx].point.x - center_x;
        double dy = candidate_points_[idx].point.y - center_y;
        double d = std::sqrt(dx * dx + dy * dy);
        double w = 1.0 / (d + epsilon);
        weighted_sum_x += candidate_points_[idx].point.x * w;
        weighted_sum_y += candidate_points_[idx].point.y * w;
        total_weight += w;
      }
      rep_x = weighted_sum_x / total_weight;
      rep_y = weighted_sum_y / total_weight;
    } else {
      struct WeightedVal { double val; double weight; };
      std::vector<WeightedVal> wx, wy;
      double total_weight = 0.0;
      for (size_t idx : cluster) {
        double dx = candidate_points_[idx].point.x - center_x;
        double dy = candidate_points_[idx].point.y - center_y;
        double d = std::sqrt(dx * dx + dy * dy);
        double w = 1.0 / (d + epsilon);
        wx.push_back({candidate_points_[idx].point.x, w});
        wy.push_back({candidate_points_[idx].point.y, w});
        total_weight += w;
      }
      auto cmp = [](const WeightedVal &a, const WeightedVal &b) { return a.val < b.val; };
      std::sort(wx.begin(), wx.end(), cmp);
      std::sort(wy.begin(), wy.end(), cmp);
      double cum = 0.0, median_x = wx.front().val;
      for (const auto &w : wx) { cum += w.weight; if (cum >= total_weight / 2.0) { median_x = w.val; break; } }
      cum = 0.0; double median_y = wy.front().val;
      for (const auto &w : wy) { cum += w.weight; if (cum >= total_weight / 2.0) { median_y = w.val; break; } }
      rep_x = median_x; rep_y = median_y;
    }
    return {rep_x, rep_y};
  }

  void processAndUpdateMeasurementWithPoint(double meas_x, double meas_y, const rclcpp::Time& stamp) {
    if (!use_kalman_filter_) {
      kf_state_[0] = meas_x; kf_state_[1] = meas_y; kf_state_[2] = 0.0; kf_state_[3] = 0.0;
      kalman_initialized_ = true;
      publishOdomWithKalmanState(stamp);
      last_kf_time_ = stamp; last_measurement_time_ = stamp;
      return;
    }

    if (!kalman_initialized_) {
      kf_state_[0] = meas_x; kf_state_[1] = meas_y; kf_state_[2] = 0.0; kf_state_[3] = 0.0;
      kf_P_[0][0] = 1.0; kf_P_[0][1] = 0.0; kf_P_[0][2] = 0.0; kf_P_[0][3] = 0.0;
      kf_P_[1][0] = 0.0; kf_P_[1][1] = 1.0; kf_P_[1][2] = 0.0; kf_P_[1][3] = 0.0;
      kf_P_[2][0] = 0.0; kf_P_[2][1] = 0.0; kf_P_[2][2] = 1.0; kf_P_[2][3] = 0.0;
      kf_P_[3][0] = 0.0; kf_P_[3][1] = 0.0; kf_P_[3][2] = 0.0; kf_P_[3][3] = 1.0;
      kalman_initialized_ = true;
      last_kf_time_ = stamp; last_measurement_time_ = stamp;
      publishOdomWithKalmanState(stamp);
      return;
    }

    const double since_meas = (stamp - last_measurement_time_).seconds();
    if (since_meas > obstacle_timeout_) {
      RCLCPP_WARN(this->get_logger(),
                  "No measurement for %.2fs > timeout(%.2fs). Stop publishing & reset tracker.",
                  since_meas, obstacle_timeout_);
      kalman_initialized_ = false;
      return;
    }

    double dt = (stamp - last_kf_time_).seconds();
    if (dt < 0) { RCLCPP_WARN(this->get_logger(), "Negative dt detected, skip predict. dt=%.4f", dt); dt = 0; }
    kf_state_[0] += kf_state_[2] * dt;
    kf_state_[1] += kf_state_[3] * dt;
    kf_P_[0][0] += kalman_process_noise_;
    kf_P_[1][1] += kalman_process_noise_;
    kf_P_[2][2] += kalman_process_noise_;
    kf_P_[3][3] += kalman_process_noise_;

    double P00 = kf_P_[0][0], P11 = kf_P_[1][1];
    double S0 = P00 + kalman_measurement_noise_;
    double S1 = P11 + kalman_measurement_noise_;
    double K0 = P00 / S0, K1 = P11 / S1;
    double y0 = meas_x - kf_state_[0], y1 = meas_y - kf_state_[1];
    kf_state_[0] += K0 * y0; kf_state_[1] += K1 * y1;
    kf_P_[0][0] = (1 - K0) * P00; kf_P_[1][1] = (1 - K1) * P11;

    publishOdomWithKalmanState(stamp);
    last_kf_time_ = stamp; last_measurement_time_ = stamp;
  }

  void publishOdomWithKalmanState(const rclcpp::Time & stamp) {
    double heading_pos = prev_heading_;
    if (has_prev_position_) {
      double dx = kf_state_[0] - prev_x_, dy = kf_state_[1] - prev_y_;
      if (std::sqrt(dx*dx + dy*dy) > 1e-3) heading_pos = std::atan2(dy, dx);
    }
    double speed = std::sqrt(kf_state_[2]*kf_state_[2] + kf_state_[3]*kf_state_[3]);
    double heading_vel = (speed > 1e-3) ? std::atan2(kf_state_[3], kf_state_[2]) : heading_pos;
    double raw_heading = (heading_pos + heading_vel) / 2.0;

    double alpha = 0.5;
    double delta = raw_heading - prev_heading_;
    while (delta > M_PI)  delta -= 2.0 * M_PI;
    while (delta < -M_PI) delta += 2.0 * M_PI;
    double smoothed_heading = prev_heading_ + alpha * delta;

    prev_heading_ = smoothed_heading;
    prev_x_ = kf_state_[0]; prev_y_ = kf_state_[1];
    has_prev_position_ = true;

    double sin_yaw = std::sin(smoothed_heading * 0.5);
    double cos_yaw = std::cos(smoothed_heading * 0.5);

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = stamp;
    odom_msg.header.frame_id = "map";
    odom_msg.pose.pose.position.x = kf_state_[0];
    odom_msg.pose.pose.position.y = kf_state_[1];
    odom_msg.pose.pose.position.z = 0.0;
    odom_msg.twist.twist.linear.x = kf_state_[2];
    odom_msg.twist.twist.linear.y = kf_state_[3];
    odom_msg.twist.twist.linear.z = 0.0;
    odom_msg.pose.pose.orientation.x = 0.0;
    odom_msg.pose.pose.orientation.y = 0.0;
    odom_msg.pose.pose.orientation.z = sin_yaw;
    odom_msg.pose.pose.orientation.w = cos_yaw;
    odom_msg.twist.twist.angular.x = 0.0;
    odom_msg.twist.twist.angular.y = 0.0;
    odom_msg.twist.twist.angular.z = 0.0;

    dynamic_pub_->publish(odom_msg);
  }

private:

  inline rclcpp::Time to_node_time(const builtin_interfaces::msg::Time& t) const {
    return rclcpp::Time(t, this->get_clock()->get_clock_type());
  }

  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr marker_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_only_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr        wall_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr        static_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr                 dynamic_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr    dbscan_vis_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr           wall_accum_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr current_scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_history_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr aligned_history_markers_pub_;

  std::string scan_topic_{"/scan"};
  std::string odom_topic_{"/odom_airio"};

  message_filters::Subscriber<sensor_msgs::msg::LaserScan> scan_sub_;
  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;


  std::deque<geometry_msgs::msg::PointStamped> candidate_points_;
  double window_seconds_{0.25};

  nav_msgs::msg::Odometry::ConstSharedPtr last_odom_;  // 최신 odom 저장

  double dbscan_eps_;
  int dbscan_min_points_;
  bool use_weighted_median_;
  int min_candidates_to_process_;
  bool use_kalman_filter_;
  double kalman_process_noise_;
  double kalman_measurement_noise_;
  double obstacle_timeout_;
  double gate_dyn_{1.2};

  bool kalman_initialized_;
  double kf_state_[4]{0,0,0,0};
  double kf_P_[4][4]{{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
  rclcpp::Time last_kf_time_;
  rclcpp::Time last_measurement_time_;
  double prev_x_{0.0}, prev_y_{0.0}, prev_heading_{0.0};
  bool has_prev_position_;

  std::string wall_topic_{"/wall_points"};
  int    wall_deque_size_{10};
  double voxel_leaf_{0.03};
  int    knn_k_{5};
  double wall_dist_thresh_{0.06};


  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  MapManager scan_map_manager_;
  MapManager obj_manager_;

  DynamicObjectDetector detector_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ObstacleDetector>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
