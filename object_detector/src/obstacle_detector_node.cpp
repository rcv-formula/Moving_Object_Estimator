/**
 * @file obstacle_detector_node.cpp
 * @brief ROS2 node for detecting and tracking static/dynamic obstacles
 * 
 * This node performs:
 * - Obstacle candidate clustering using DBSCAN
 * - Static/dynamic classification using footprint analysis
 * - Kalman filter-based tracking of dynamic obstacles
 * - Visualization of detection results
 */

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

#include <Eigen/Dense>

#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <deque>
#include <array>
#include <optional>

#include "map_manager_pair.hpp"
#include "dynamic_obstacle_detector.hpp"

using namespace std::chrono_literals;

namespace {

/// @brief Classification labels for obstacles
enum class ObstacleLabel {
  kUnknown = 0,
  kStatic = 1,
  kDynamic = 2
};

/// @brief Color definitions for visualization
struct VisualizationColor {
  float r, g, b, a;
  
  static constexpr VisualizationColor Static() {
    return {0.0f, 0.4f, 1.0f, 0.95f};
  }
  
  static constexpr VisualizationColor Dynamic() {
    return {1.0f, 0.1f, 0.1f, 0.95f};
  }
  
  static constexpr VisualizationColor Unknown() {
    return {0.6f, 0.6f, 0.6f, 0.8f};
  }
  
  static constexpr VisualizationColor White() {
    return {1.0f, 1.0f, 1.0f, 0.95f};
  }
};

/// @brief Constants for visualization
constexpr double kClusterPointScale = 0.06;
constexpr double kCenterSphereScale = 0.15;
constexpr double kTextScale = 0.18;
constexpr double kTextOffsetZ = 0.18;
constexpr double kMarkerLifetime = 0.2; // seconds

/// @brief Default configuration values
constexpr size_t kDefaultScanBufferSize = 40;
constexpr size_t kDefaultObjectBufferSize = 10;
constexpr double kDefaultFootprintEpsilon = 0.2;
constexpr int kDefaultFootprintMinPoints = 2;
constexpr double kDefaultFootprintSearchRadius = 0.6;
constexpr double kDefaultMotionThreshold = 0.2;

}  // namespace

/**
 * @brief Main obstacle detection and tracking node
 */
class ObstacleDetectorNode : public rclcpp::Node {
public:
  /**
   * @brief Configuration parameters
   */
  struct Parameters {
    // DBSCAN clustering
    double dbscan_epsilon{0.3};
    int dbscan_min_points{3};
    bool use_weighted_median{false};
    int min_candidates_to_process{3};
    
    // Kalman filter
    bool use_kalman_filter{true};
    double kalman_process_noise{0.1};
    double kalman_measurement_noise{0.1};
    double obstacle_timeout{1.0};
    
    // Data association
    double association_gate{1.2};
    
    // Sliding window
    double window_seconds{0.25};
    
    // Topics
    std::string wall_topic{"/wall_points"};
    std::string scan_topic{"/scan"};
    std::string odom_topic{"/odom"};
    
    // Wall processing (for future use)
    int wall_deque_size{10};
    double voxel_leaf{0.03};
    int knn_k{5};
    double wall_distance_threshold{0.06};
  };

  ObstacleDetectorNode()
      : Node("obstacle_detector"),
        scan_buffer_(kDefaultScanBufferSize),
        object_buffer_(kDefaultObjectBufferSize) {
    
    InitializeParameters();
    InitializeTfListener();
    InitializePublishers();
    InitializeSubscribers();
    
    ResetKalmanFilter();
    
    RCLCPP_INFO(this->get_logger(), 
                "ObstacleDetectorNode initialized");
    RCLCPP_INFO(this->get_logger(), 
                "  wall_topic: %s | scan_topic: %s | odom_topic: %s",
                params_.wall_topic.c_str(),
                params_.scan_topic.c_str(),
                params_.odom_topic.c_str());
  }

private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::LaserScan, nav_msgs::msg::Odometry>;

  // ========== Initialization ==========
  
  void InitializeParameters() {
    DeclareParameters();
    LoadParameters();
  }

  void DeclareParameters() {
    this->declare_parameter<double>("dbscan_eps", params_.dbscan_epsilon);
    this->declare_parameter<int>("dbscan_min_points", params_.dbscan_min_points);
    this->declare_parameter<bool>("use_weighted_median", params_.use_weighted_median);
    this->declare_parameter<double>("kalman_process_noise", params_.kalman_process_noise);
    this->declare_parameter<double>("kalman_measurement_noise", params_.kalman_measurement_noise);
    this->declare_parameter<bool>("use_kalman_filter", params_.use_kalman_filter);
    this->declare_parameter<double>("obstacle_timeout", params_.obstacle_timeout);
    this->declare_parameter<int>("min_candidates_to_process", params_.min_candidates_to_process);
    this->declare_parameter<double>("window_seconds", params_.window_seconds);
    this->declare_parameter<std::string>("wall_topic", params_.wall_topic);
    this->declare_parameter<std::string>("scan_topic", params_.scan_topic);
    this->declare_parameter<std::string>("odom_topic", params_.odom_topic);
    this->declare_parameter<int>("wall_deque_size", params_.wall_deque_size);
    this->declare_parameter<double>("voxel_leaf", params_.voxel_leaf);
    this->declare_parameter<int>("knn_k", params_.knn_k);
    this->declare_parameter<double>("wall_dist_thresh", params_.wall_distance_threshold);
    this->declare_parameter<double>("association_gate", params_.association_gate);
  }

  void LoadParameters() {
    this->get_parameter("dbscan_eps", params_.dbscan_epsilon);
    this->get_parameter("dbscan_min_points", params_.dbscan_min_points);
    this->get_parameter("use_weighted_median", params_.use_weighted_median);
    this->get_parameter("kalman_process_noise", params_.kalman_process_noise);
    this->get_parameter("kalman_measurement_noise", params_.kalman_measurement_noise);
    this->get_parameter("use_kalman_filter", params_.use_kalman_filter);
    this->get_parameter("obstacle_timeout", params_.obstacle_timeout);
    this->get_parameter("min_candidates_to_process", params_.min_candidates_to_process);
    this->get_parameter("window_seconds", params_.window_seconds);
    this->get_parameter("wall_topic", params_.wall_topic);
    this->get_parameter("scan_topic", params_.scan_topic);
    this->get_parameter("odom_topic", params_.odom_topic);
    this->get_parameter("wall_deque_size", params_.wall_deque_size);
    this->get_parameter("voxel_leaf", params_.voxel_leaf);
    this->get_parameter("knn_k", params_.knn_k);
    this->get_parameter("wall_dist_thresh", params_.wall_distance_threshold);
    this->get_parameter("association_gate", params_.association_gate);
  }

  void InitializeTfListener() {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  }

  void InitializePublishers() {
    static_obstacle_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
        "/static_obstacle", 10);
    
    dynamic_obstacle_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
        "/dynamic_obstacle", 20);
    
    dbscan_visualization_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/dbscan_clusters", 10);
    
    wall_accumulation_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/wall_points_accum", 10);
    
    footprint_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "footprint_markers", 10);
    
    current_scan_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/current_scan_pcl", 10);
    
    aligned_history_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/aligned_history_scans", 10);
    
    aligned_history_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/aligned_obj_history", 10);
  }

  void InitializeSubscribers() {
    // Wall points subscriber
    wall_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        params_.wall_topic,
        50,
        std::bind(&ObstacleDetectorNode::WallCallback, this, std::placeholders::_1));
    
    // Detected obstacles marker subscriber
    marker_sub_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
        "/detected_obstacles",
        rclcpp::SensorDataQoS(),
        std::bind(&ObstacleDetectorNode::MarkerCallback, this, std::placeholders::_1));
    
    // Odometry only subscriber (for latest pose)
    odom_only_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        params_.odom_topic,
        rclcpp::SensorDataQoS(),
        [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) {
          latest_odom_ = std::move(msg);
        });
    
    // Synchronized scan and odometry subscribers
    scan_sub_.subscribe(this, params_.scan_topic.c_str(), rmw_qos_profile_sensor_data);
    odom_sub_.subscribe(this, params_.odom_topic.c_str(), rmw_qos_profile_sensor_data);
    
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(1000), scan_sub_, odom_sub_);
    sync_->registerCallback(
        std::bind(&ObstacleDetectorNode::ScanOdomSyncCallback,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2));
  }

  // ========== Callbacks ==========

  void WallCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Wall processing pipeline (placeholder for future implementation)
    // Currently not used but can be extended for wall-based filtering
  }

  void ScanOdomSyncCallback(
      const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg,
      const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg) {
    
    // Convert scan to point cloud
    auto local_cloud = ConvertScanToPointCloud(scan_msg);
    
    // Publish current scan
    sensor_msgs::msg::PointCloud2 current_cloud_msg;
    pcl::toROSMsg(*local_cloud, current_cloud_msg);
    current_cloud_msg.header = scan_msg->header;
    current_scan_pub_->publish(current_cloud_msg);
    
    // Add to scan buffer with pose
    scan_buffer_.AddCloudWithPose(
        std::make_shared<sensor_msgs::msg::PointCloud2>(current_cloud_msg),
        odom_msg->pose.pose);
    
    // Get scan history
    auto scan_pairs = scan_buffer_.GetCloudPoseSnapshot();
    if (scan_pairs.size() < 2) {
      return;
    }
    
    // Transform historical scans to current frame
    const auto& current_pose = scan_pairs.back().second;
    auto aligned_scans = TransformScansToCurrentFrame(scan_pairs, current_pose);
    
    // Publish aligned historical scans
    PublishAlignedScans(aligned_scans, scan_msg);
  }

  void MarkerCallback(const visualization_msgs::msg::MarkerArray::ConstSharedPtr& msg) {
    if (msg->markers.empty()) {
      return;
    }
    
    if (!latest_odom_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Odometry not received yet; skipping Marker processing.");
      return;
    }

    // Extract obstacle candidates from markers
    for (const auto& marker : msg->markers) {
      if (marker.action != visualization_msgs::msg::Marker::ADD) {
        continue;
      }
      
      geometry_msgs::msg::PointStamped candidate;
      candidate.header = marker.header;
      candidate.point = marker.pose.position;
      
      last_transform_stamp_ = marker.header.stamp;
      candidate_points_.push_back(candidate);
    }
    
    // Prune old points outside time window
    const auto latest_time = ToNodeTime(msg->markers.front().header.stamp);
    PruneTimeWindow(latest_time);
    
    // Check if we have enough candidates
    if (candidate_points_.size() < static_cast<size_t>(params_.min_candidates_to_process)) {
      PublishEmptyMarkers(latest_time);
      return;
    }
    
    // Perform DBSCAN clustering
    auto clusters = PerformDbscan();
    if (clusters.empty()) {
      PublishEmptyMarkers(this->now());
      return;
    }
    
    // Compute cluster centers
    auto centers = ComputeClusterCenters(clusters);
    
    // Classify obstacles as static or dynamic
    auto [labels, static_list, dynamic_list] = ClassifyObstacles(centers);
    
    // Add detected objects to history
    object_buffer_.AddObjectsWithPose(centers, latest_odom_->pose.pose);
    
    // Log detection results
    LogDetectionResults(centers, static_list, dynamic_list);
    
    // Visualize results
    VisualizeDbscanClusters(clusters, centers, labels);
    
    // Publish static obstacle (nearest one)
    PublishStaticObstacle(centers, labels);
    
    // Process and track dynamic obstacles
    ProcessDynamicObstacles(centers, labels);
  }

  // ========== Point Cloud Processing ==========

  [[nodiscard]] pcl::PointCloud<pcl::PointXYZI>::Ptr ConvertScanToPointCloud(
      const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg) const {
    
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    cloud->reserve(scan_msg->ranges.size());
    
    for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
      const double range = scan_msg->ranges[i];
      if (!std::isfinite(range)) {
        continue;
      }
      
      const double angle = scan_msg->angle_min + i * scan_msg->angle_increment;
      
      pcl::PointXYZI point;
      point.x = static_cast<float>(range * std::cos(angle));
      point.y = static_cast<float>(range * std::sin(angle));
      point.z = 0.0f;
      point.intensity = 255.0f;
      
      cloud->push_back(point);
    }
    
    return cloud;
  }

  [[nodiscard]] pcl::PointCloud<pcl::PointXYZI> TransformScansToCurrentFrame(
      const std::vector<MapManager::CloudPosePair>& scan_pairs,
      const geometry_msgs::msg::Pose& current_pose) const {
    
    const Eigen::Matrix4d transform_map_to_current = 
        MapManager::PoseToMatrix(current_pose);
    const Eigen::Matrix4d transform_current_to_map = 
        transform_map_to_current.inverse();
    
    pcl::PointCloud<pcl::PointXYZI> all_aligned_scans;
    const size_t num_frames = scan_pairs.size() - 1;
    
    for (size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
      const auto& historical_scan = scan_pairs[frame_idx].first;
      const auto& historical_pose = scan_pairs[frame_idx].second;
      
      // Compute transformation: historical -> current
      const Eigen::Matrix4d transform_map_to_historical = 
          MapManager::PoseToMatrix(historical_pose);
      const Eigen::Matrix4d transform_current_to_historical = 
          transform_current_to_map * transform_map_to_historical;
      
      // Transform point cloud
      pcl::PointCloud<pcl::PointXYZI> aligned_scan;
      pcl::transformPointCloud(
          *historical_scan,
          aligned_scan,
          transform_current_to_historical.cast<float>());
      
      // Set intensity based on age (for visualization)
      const float intensity_value = static_cast<float>((num_frames - 1) - frame_idx);
      for (auto& point : aligned_scan.points) {
        point.intensity = intensity_value;
      }
      
      all_aligned_scans += aligned_scan;
    }
    
    return all_aligned_scans;
  }

  void PublishAlignedScans(
      const pcl::PointCloud<pcl::PointXYZI>& aligned_scans,
      const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg) {
    
    sensor_msgs::msg::PointCloud2 aligned_msg;
    pcl::toROSMsg(aligned_scans, aligned_msg);
    aligned_msg.header.stamp = scan_msg->header.stamp;
    aligned_msg.header.frame_id = scan_msg->header.frame_id;
    
    aligned_history_pub_->publish(aligned_msg);
  }

  // ========== DBSCAN Clustering ==========

  [[nodiscard]] std::vector<std::vector<size_t>> PerformDbscan() const {
    const size_t num_points = candidate_points_.size();
    std::vector<int> cluster_ids(num_points, -1);
    
    auto compute_distance = [this](size_t i, size_t j) {
      const double dx = candidate_points_[i].point.x - candidate_points_[j].point.x;
      const double dy = candidate_points_[i].point.y - candidate_points_[j].point.y;
      return std::hypot(dx, dy);
    };
    
    auto region_query = [&](size_t point_idx) {
      std::vector<size_t> neighbors;
      for (size_t j = 0; j < num_points; ++j) {
        if (compute_distance(point_idx, j) <= params_.dbscan_epsilon) {
          neighbors.push_back(j);
        }
      }
      return neighbors;
    };
    
    int current_cluster_id = 0;
    
    for (size_t i = 0; i < num_points; ++i) {
      if (cluster_ids[i] != -1) {
        continue;
      }
      
      auto neighbors = region_query(i);
      
      if (neighbors.size() < static_cast<size_t>(params_.dbscan_min_points)) {
        cluster_ids[i] = -2; // Noise
        continue;
      }
      
      cluster_ids[i] = current_cluster_id;
      std::vector<size_t> seed_set = std::move(neighbors);
      
      for (size_t idx = 0; idx < seed_set.size(); ++idx) {
        const size_t current_idx = seed_set[idx];
        
        if (cluster_ids[current_idx] == -2) {
          cluster_ids[current_idx] = current_cluster_id;
        }
        
        if (cluster_ids[current_idx] != -1) {
          continue;
        }
        
        cluster_ids[current_idx] = current_cluster_id;
        
        auto current_neighbors = region_query(current_idx);
        if (current_neighbors.size() >= static_cast<size_t>(params_.dbscan_min_points)) {
          seed_set.insert(seed_set.end(),
                         current_neighbors.begin(),
                         current_neighbors.end());
        }
      }
      
      current_cluster_id++;
    }
    
    // Extract clusters
    std::vector<std::vector<size_t>> clusters(current_cluster_id);
    for (size_t i = 0; i < num_points; ++i) {
      if (cluster_ids[i] >= 0) {
        clusters[cluster_ids[i]].push_back(i);
      }
    }
    
    return clusters;
  }

  [[nodiscard]] std::vector<geometry_msgs::msg::PointStamped> ComputeClusterCenters(
      const std::vector<std::vector<size_t>>& clusters) const {
    
    std::vector<geometry_msgs::msg::PointStamped> centers;
    centers.reserve(clusters.size());
    
    for (const auto& cluster : clusters) {
      const auto [center_x, center_y] = ComputeRepresentativePoint(cluster);
      
      geometry_msgs::msg::PointStamped center;
      center.header = candidate_points_[cluster.front()].header;
      center.point.x = center_x;
      center.point.y = center_y;
      center.point.z = 0.0;
      
      centers.push_back(center);
    }
    
    return centers;
  }

  [[nodiscard]] std::pair<double, double> ComputeRepresentativePoint(
      const std::vector<size_t>& cluster) const {
    
    // Compute centroid
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (size_t idx : cluster) {
      sum_x += candidate_points_[idx].point.x;
      sum_y += candidate_points_[idx].point.y;
    }
    const double center_x = sum_x / cluster.size();
    const double center_y = sum_y / cluster.size();
    
    constexpr double kEpsilon = 1e-3;
    
    if (!params_.use_weighted_median) {
      // Inverse distance weighted average
      double weighted_sum_x = 0.0;
      double weighted_sum_y = 0.0;
      double total_weight = 0.0;
      
      for (size_t idx : cluster) {
        const double dx = candidate_points_[idx].point.x - center_x;
        const double dy = candidate_points_[idx].point.y - center_y;
        const double distance = std::hypot(dx, dy);
        const double weight = 1.0 / (distance + kEpsilon);
        
        weighted_sum_x += candidate_points_[idx].point.x * weight;
        weighted_sum_y += candidate_points_[idx].point.y * weight;
        total_weight += weight;
      }
      
      return {weighted_sum_x / total_weight, weighted_sum_y / total_weight};
    } else {
      // Weighted median
      struct WeightedValue {
        double value;
        double weight;
      };
      
      std::vector<WeightedValue> weighted_x;
      std::vector<WeightedValue> weighted_y;
      double total_weight = 0.0;
      
      for (size_t idx : cluster) {
        const double dx = candidate_points_[idx].point.x - center_x;
        const double dy = candidate_points_[idx].point.y - center_y;
        const double distance = std::hypot(dx, dy);
        const double weight = 1.0 / (distance + kEpsilon);
        
        weighted_x.push_back({candidate_points_[idx].point.x, weight});
        weighted_y.push_back({candidate_points_[idx].point.y, weight});
        total_weight += weight;
      }
      
      auto comparator = [](const WeightedValue& a, const WeightedValue& b) {
        return a.value < b.value;
      };
      
      std::sort(weighted_x.begin(), weighted_x.end(), comparator);
      std::sort(weighted_y.begin(), weighted_y.end(), comparator);
      
      // Find weighted median
      double cumulative_weight = 0.0;
      double median_x = weighted_x.front().value;
      for (const auto& wv : weighted_x) {
        cumulative_weight += wv.weight;
        if (cumulative_weight >= total_weight / 2.0) {
          median_x = wv.value;
          break;
        }
      }
      
      cumulative_weight = 0.0;
      double median_y = weighted_y.front().value;
      for (const auto& wv : weighted_y) {
        cumulative_weight += wv.weight;
        if (cumulative_weight >= total_weight / 2.0) {
          median_y = wv.value;
          break;
        }
      }
      
      return {median_x, median_y};
    }
  }

  // ========== Obstacle Classification ==========

  [[nodiscard]] std::tuple<
      std::vector<ObstacleLabel>,
      std::vector<geometry_msgs::msg::PointStamped>,
      std::vector<geometry_msgs::msg::PointStamped>>
  ClassifyObstacles(const std::vector<geometry_msgs::msg::PointStamped>& centers) {
    
    std::vector<ObstacleLabel> labels(centers.size(), ObstacleLabel::kUnknown);
    std::vector<geometry_msgs::msg::PointStamped> static_list;
    std::vector<geometry_msgs::msg::PointStamped> dynamic_list;
    
    static_list.reserve(centers.size());
    dynamic_list.reserve(centers.size());
    
    // Get historical object detections
    const auto object_pairs = object_buffer_.GetObjectPoseSnapshot();
    
    // Transform to current frame
    auto aligned_frames = detector_.TransformHistoricalObjects(
        object_pairs,
        latest_odom_->pose.pose);
    
    // Visualize aligned historical detections
    detector_.PublishAlignedFramesMarkers(
        aligned_frames,
        "laser",
        this->now(),
        aligned_history_markers_pub_,
        0.06,   // point scale
        0.1);   // lifetime
    
    // Classify each center
    for (size_t i = 0; i < centers.size(); ++i) {
      const auto& center = centers[i];
      
      std::vector<geometry_msgs::msg::Point> footprint;
      double span = 0.0;
      
      const ObstacleType classification = detector_.ClassifyDynamicByFootprint(
          center.point,
          aligned_frames,
          false,  // exclude_current
          &footprint,
          &span);
      
      // Visualize footprint
      detector_.VisualizeFootprint(
          footprint,
          classification,
          "laser",
          static_cast<int>(i),
          this->now(),
          footprint_pub_);
      
      // Store classification
      if (classification == ObstacleType::kStatic) {
        labels[i] = ObstacleLabel::kStatic;
        static_list.push_back(center);
      } else if (classification == ObstacleType::kDynamic) {
        labels[i] = ObstacleLabel::kDynamic;
        dynamic_list.push_back(center);
      } else {
        labels[i] = ObstacleLabel::kUnknown;
      }
    }
    
    return {labels, static_list, dynamic_list};
  }

  // ========== Kalman Filter ==========

  void ResetKalmanFilter() noexcept {
    kalman_initialized_ = false;
    
    std::fill(std::begin(kf_state_), std::end(kf_state_), 0.0);
    
    for (auto& row : kf_covariance_) {
      std::fill(std::begin(row), std::end(row), 0.0);
    }
    
    for (size_t i = 0; i < 4; ++i) {
      kf_covariance_[i][i] = 1.0;
    }
    
    previous_position_ = {0.0, 0.0};
    previous_heading_ = 0.0;
    has_previous_position_ = false;
    
    last_kf_time_ = rclcpp::Time(0, 0, this->get_clock()->get_clock_type());
    last_measurement_time_ = rclcpp::Time(0, 0, this->get_clock()->get_clock_type());
  }

  void ProcessDynamicObstacles(
      const std::vector<geometry_msgs::msg::PointStamped>& centers,
      const std::vector<ObstacleLabel>& labels) {
    
    if (centers.empty()) {
      return;
    }
    
    // Extract dynamic obstacles
    std::vector<geometry_msgs::msg::PointStamped> dynamic_obstacles;
    for (size_t i = 0; i < centers.size(); ++i) {
      if (labels[i] == ObstacleLabel::kDynamic) {
        dynamic_obstacles.push_back(centers[i]);
      }
    }
    
    if (dynamic_obstacles.empty()) {
      CheckForTimeout(centers.front().header.stamp);
      return;
    }
    
    // Choose best measurement
    const auto* measurement = ChooseDynamicMeasurement(dynamic_obstacles);
    
    if (measurement) {
      auto measurement_in_map = TransformLocalWithPose(*measurement, latest_odom_);
      ProcessMeasurementUpdate(
          measurement_in_map.point.x,
          measurement_in_map.point.y,
          measurement_in_map.header.stamp);
    } else {
      CheckForTimeout(centers.front().header.stamp);
    }
  }

  [[nodiscard]] const geometry_msgs::msg::PointStamped* ChooseDynamicMeasurement(
      const std::vector<geometry_msgs::msg::PointStamped>& dynamic_obstacles) const {
    
    if (dynamic_obstacles.empty()) {
      return nullptr;
    }
    
    const geometry_msgs::msg::PointStamped* best = nullptr;
    double best_score = std::numeric_limits<double>::infinity();
    
    if (!kalman_initialized_) {
      // Choose nearest to origin
      for (const auto& obstacle : dynamic_obstacles) {
        const double distance_sq = obstacle.point.x * obstacle.point.x +
                                   obstacle.point.y * obstacle.point.y;
        if (distance_sq < best_score) {
          best_score = distance_sq;
          best = &obstacle;
        }
      }
      return best;
    }
    
    // Choose nearest to predicted position
    const double predicted_x = kf_state_[0];
    const double predicted_y = kf_state_[1];
    
    for (const auto& obstacle : dynamic_obstacles) {
      const double dx = obstacle.point.x - predicted_x;
      const double dy = obstacle.point.y - predicted_y;
      const double distance = std::hypot(dx, dy);
      
      if (distance < best_score) {
        best_score = distance;
        best = &obstacle;
      }
    }
    
    // Apply gating
    if (best && best_score > params_.association_gate) {
      return nullptr;
    }
    
    return best;
  }

  void ProcessMeasurementUpdate(
      double measurement_x,
      double measurement_y,
      const rclcpp::Time& timestamp) {
    
    if (!params_.use_kalman_filter) {
      // Direct assignment without filtering
      kf_state_[0] = measurement_x;
      kf_state_[1] = measurement_y;
      kf_state_[2] = 0.0;
      kf_state_[3] = 0.0;
      kalman_initialized_ = true;
      
      PublishDynamicObstacleOdometry(timestamp);
      last_kf_time_ = timestamp;
      last_measurement_time_ = timestamp;
      return;
    }
    
    if (!kalman_initialized_) {
      InitializeKalmanFilter(measurement_x, measurement_y, timestamp);
      return;
    }
    
    // Check for timeout
    const double time_since_measurement = 
        (timestamp - last_measurement_time_).seconds();
    if (time_since_measurement > params_.obstacle_timeout) {
      RCLCPP_WARN(this->get_logger(),
                  "No measurement for %.2fs > timeout(%.2fs). "
                  "Stopping tracking and resetting.",
                  time_since_measurement,
                  params_.obstacle_timeout);
      ResetKalmanFilter();
      return;
    }
    
    // Prediction step
    const double dt = (timestamp - last_kf_time_).seconds();
    if (dt < 0.0) {
      RCLCPP_WARN(this->get_logger(),
                  "Negative time delta detected: %.4f. Skipping prediction.",
                  dt);
    } else {
      PredictKalmanFilter(dt);
    }
    
    // Update step
    UpdateKalmanFilter(measurement_x, measurement_y);
    
    // Publish result
    PublishDynamicObstacleOdometry(timestamp);
    
    last_kf_time_ = timestamp;
    last_measurement_time_ = timestamp;
  }

  void InitializeKalmanFilter(
      double initial_x,
      double initial_y,
      const rclcpp::Time& timestamp) {
    
    kf_state_[0] = initial_x;
    kf_state_[1] = initial_y;
    kf_state_[2] = 0.0;
    kf_state_[3] = 0.0;
    
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        kf_covariance_[i][j] = (i == j) ? 1.0 : 0.0;
      }
    }
    
    kalman_initialized_ = true;
    last_kf_time_ = timestamp;
    last_measurement_time_ = timestamp;
    
    PublishDynamicObstacleOdometry(timestamp);
  }

  void PredictKalmanFilter(double dt) noexcept {
    // Constant velocity model: x = x + vx*dt, y = y + vy*dt
    kf_state_[0] += kf_state_[2] * dt;
    kf_state_[1] += kf_state_[3] * dt;
    
    // Add process noise to covariance
    kf_covariance_[0][0] += params_.kalman_process_noise;
    kf_covariance_[1][1] += params_.kalman_process_noise;
    kf_covariance_[2][2] += params_.kalman_process_noise;
    kf_covariance_[3][3] += params_.kalman_process_noise;
  }

  void UpdateKalmanFilter(double measurement_x, double measurement_y) noexcept {
    // Simplified Kalman update for position only
    const double p_xx = kf_covariance_[0][0];
    const double p_yy = kf_covariance_[1][1];
    
    const double innovation_cov_x = p_xx + params_.kalman_measurement_noise;
    const double innovation_cov_y = p_yy + params_.kalman_measurement_noise;
    
    const double kalman_gain_x = p_xx / innovation_cov_x;
    const double kalman_gain_y = p_yy / innovation_cov_y;
    
    const double innovation_x = measurement_x - kf_state_[0];
    const double innovation_y = measurement_y - kf_state_[1];
    
    kf_state_[0] += kalman_gain_x * innovation_x;
    kf_state_[1] += kalman_gain_y * innovation_y;
    
    kf_covariance_[0][0] = (1.0 - kalman_gain_x) * p_xx;
    kf_covariance_[1][1] = (1.0 - kalman_gain_y) * p_yy;
  }

  void CheckForTimeout(const builtin_interfaces::msg::Time& current_stamp) {
    if (!kalman_initialized_) {
      return;
    }
    
    const rclcpp::Time current_time = ToNodeTime(current_stamp);
    const double time_since_measurement = 
        (current_time - last_measurement_time_).seconds();
    
    if (time_since_measurement > params_.obstacle_timeout) {
      RCLCPP_WARN(this->get_logger(),
                  "Dynamic obstacle track lost (timeout: %.2fs). Resetting KF.",
                  params_.obstacle_timeout);
      ResetKalmanFilter();
    }
  }

  void PublishDynamicObstacleOdometry(const rclcpp::Time& timestamp) {
    // Compute heading from position history
    double heading_from_position = previous_heading_;
    if (has_previous_position_) {
      const double dx = kf_state_[0] - previous_position_.first;
      const double dy = kf_state_[1] - previous_position_.second;
      if (std::hypot(dx, dy) > 1e-3) {
        heading_from_position = std::atan2(dy, dx);
      }
    }
    
    // Compute heading from velocity
    const double speed = std::hypot(kf_state_[2], kf_state_[3]);
    const double heading_from_velocity = (speed > 1e-3) 
        ? std::atan2(kf_state_[3], kf_state_[2])
        : heading_from_position;
    
    // Average the two headings
    double raw_heading = (heading_from_position + heading_from_velocity) / 2.0;
    
    // Smooth heading transition
    constexpr double kAlpha = 0.5;
    double delta = raw_heading - previous_heading_;
    while (delta > M_PI) delta -= 2.0 * M_PI;
    while (delta < -M_PI) delta += 2.0 * M_PI;
    const double smoothed_heading = previous_heading_ + kAlpha * delta;
    
    // Update state
    previous_heading_ = smoothed_heading;
    previous_position_ = {kf_state_[0], kf_state_[1]};
    has_previous_position_ = true;
    
    // Create quaternion from yaw
    const double half_yaw = smoothed_heading * 0.5;
    const double sin_half_yaw = std::sin(half_yaw);
    const double cos_half_yaw = std::cos(half_yaw);
    
    // Publish odometry message
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = timestamp;
    odom_msg.header.frame_id = "map";
    
    odom_msg.pose.pose.position.x = kf_state_[0];
    odom_msg.pose.pose.position.y = kf_state_[1];
    odom_msg.pose.pose.position.z = 0.0;
    
    odom_msg.pose.pose.orientation.x = 0.0;
    odom_msg.pose.pose.orientation.y = 0.0;
    odom_msg.pose.pose.orientation.z = sin_half_yaw;
    odom_msg.pose.pose.orientation.w = cos_half_yaw;
    
    odom_msg.twist.twist.linear.x = kf_state_[2];
    odom_msg.twist.twist.linear.y = kf_state_[3];
    odom_msg.twist.twist.linear.z = 0.0;
    
    odom_msg.twist.twist.angular.x = 0.0;
    odom_msg.twist.twist.angular.y = 0.0;
    odom_msg.twist.twist.angular.z = 0.0;
    
    dynamic_obstacle_pub_->publish(odom_msg);
  }

  // ========== Visualization ==========

  void VisualizeDbscanClusters(
      const std::vector<std::vector<size_t>>& clusters,
      const std::vector<geometry_msgs::msg::PointStamped>& centers,
      const std::vector<ObstacleLabel>& labels) {
    
    visualization_msgs::msg::MarkerArray marker_array;
    
    // Delete all previous markers
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header.frame_id = centers.front().header.frame_id.empty() 
        ? "map" 
        : centers.front().header.frame_id;
    delete_marker.header.stamp = this->now();
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_marker);
    
    const std::string frame_id = delete_marker.header.frame_id;
    const rclcpp::Time stamp = this->now();
    
    // Visualize cluster points
    for (size_t cluster_idx = 0; cluster_idx < clusters.size(); ++cluster_idx) {
      visualization_msgs::msg::Marker points_marker;
      points_marker.header.frame_id = frame_id;
      points_marker.header.stamp = stamp;
      points_marker.ns = "dbscan_points";
      points_marker.id = static_cast<int>(cluster_idx);
      points_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
      points_marker.action = visualization_msgs::msg::Marker::ADD;
      points_marker.scale.x = points_marker.scale.y = points_marker.scale.z = 
          kClusterPointScale;
      
      // Color based on cluster index
      float r, g, b;
      HsvToRgb((cluster_idx % 12) / 12.0, 0.9, 0.95, r, g, b);
      points_marker.color.r = r;
      points_marker.color.g = g;
      points_marker.color.b = b;
      points_marker.color.a = 0.9f;
      
      for (size_t point_idx : clusters[cluster_idx]) {
        geometry_msgs::msg::Point point;
        point.x = candidate_points_[point_idx].point.x;
        point.y = candidate_points_[point_idx].point.y;
        point.z = 0.0;
        points_marker.points.push_back(point);
      }
      
      points_marker.lifetime = rclcpp::Duration::from_seconds(kMarkerLifetime);
      marker_array.markers.push_back(points_marker);
    }
    
    // Visualize cluster centers
    for (size_t i = 0; i < centers.size(); ++i) {
      // Center sphere
      visualization_msgs::msg::Marker center_marker;
      center_marker.header.frame_id = frame_id;
      center_marker.header.stamp = stamp;
      center_marker.ns = "dbscan_centers";
      center_marker.id = 1000 + static_cast<int>(i);
      center_marker.type = visualization_msgs::msg::Marker::SPHERE;
      center_marker.action = visualization_msgs::msg::Marker::ADD;
      center_marker.pose.position = centers[i].point;
      center_marker.pose.orientation.w = 1.0;
      center_marker.scale.x = center_marker.scale.y = center_marker.scale.z = 
          kCenterSphereScale;
      
      const auto color = GetLabelColor(labels[i]);
      center_marker.color.r = color.r;
      center_marker.color.g = color.g;
      center_marker.color.b = color.b;
      center_marker.color.a = color.a;
      
      center_marker.lifetime = rclcpp::Duration::from_seconds(kMarkerLifetime);
      marker_array.markers.push_back(center_marker);
      
      // Text label
      visualization_msgs::msg::Marker text_marker;
      text_marker.header.frame_id = frame_id;
      text_marker.header.stamp = stamp;
      text_marker.ns = "dbscan_text";
      text_marker.id = 2000 + static_cast<int>(i);
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;
      text_marker.pose.position = centers[i].point;
      text_marker.pose.position.z += kTextOffsetZ;
      text_marker.pose.orientation.w = 1.0;
      text_marker.scale.z = kTextScale;
      
      const auto text_color = VisualizationColor::White();
      text_marker.color.r = text_color.r;
      text_marker.color.g = text_color.g;
      text_marker.color.b = text_color.b;
      text_marker.color.a = text_color.a;
      
      const char* label_str = GetLabelString(labels[i]);
      text_marker.text = "#" + std::to_string(i) + " " + label_str + 
                        " (N=" + std::to_string(clusters[i].size()) + ")";
      
      text_marker.lifetime = rclcpp::Duration::from_seconds(kMarkerLifetime);
      marker_array.markers.push_back(text_marker);
    }
    
    dbscan_visualization_pub_->publish(marker_array);
  }

  void PublishStaticObstacle(
      const std::vector<geometry_msgs::msg::PointStamped>& centers,
      const std::vector<ObstacleLabel>& labels) {
    
    if (centers.empty()) {
      return;
    }
    
    // Find nearest static obstacle
    const geometry_msgs::msg::PointStamped* nearest_static = nullptr;
    double min_distance_sq = std::numeric_limits<double>::infinity();
    
    for (size_t i = 0; i < centers.size(); ++i) {
      if (labels[i] != ObstacleLabel::kStatic) {
        continue;
      }
      
      const double distance_sq = centers[i].point.x * centers[i].point.x +
                                 centers[i].point.y * centers[i].point.y;
      
      if (distance_sq < min_distance_sq) {
        min_distance_sq = distance_sq;
        nearest_static = &centers[i];
      }
    }
    
    if (nearest_static) {
      auto static_in_map = TransformLocalWithPose(*nearest_static, latest_odom_);
      static_obstacle_pub_->publish(static_in_map);
    }
  }

  void PublishEmptyMarkers(const rclcpp::Time& timestamp) {
    visualization_msgs::msg::MarkerArray marker_array;
    
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header.frame_id = "map";
    delete_marker.header.stamp = timestamp;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    
    marker_array.markers.push_back(delete_marker);
    dbscan_visualization_pub_->publish(marker_array);
  }

  // ========== Utility Functions ==========

  void PruneTimeWindow(const rclcpp::Time& current_time) {
    while (!candidate_points_.empty()) {
      const auto& stamp = candidate_points_.front().header.stamp;
      const rclcpp::Time point_time = 
          (stamp.sec == 0 && stamp.nanosec == 0) 
              ? current_time 
              : rclcpp::Time(stamp);
      
      if ((current_time - point_time).seconds() <= params_.window_seconds) {
        break;
      }
      
      candidate_points_.pop_front();
    }
  }

  [[nodiscard]] rclcpp::Time ToNodeTime(
      const builtin_interfaces::msg::Time& stamp) const {
    return rclcpp::Time(stamp, this->get_clock()->get_clock_type());
  }

  [[nodiscard]] geometry_msgs::msg::PointStamped TransformLocalWithPose(
      const geometry_msgs::msg::PointStamped& local_point,
      const nav_msgs::msg::Odometry::ConstSharedPtr& odom) const {
    
    if (!odom) {
      throw std::runtime_error("Odometry message is null");
    }
    
    const Eigen::Matrix4d transform = PoseToTransformMatrix(odom->pose.pose);
    const Eigen::Vector4d point_local(
        local_point.point.x,
        local_point.point.y,
        local_point.point.z,
        1.0);
    
    const Eigen::Vector4d point_world = transform * point_local;
    
    geometry_msgs::msg::PointStamped output;
    output.header.stamp = (local_point.header.stamp.sec || local_point.header.stamp.nanosec)
        ? local_point.header.stamp
        : odom->header.stamp;
    output.header.frame_id = odom->header.frame_id;
    output.point.x = point_world.x();
    output.point.y = point_world.y();
    output.point.z = point_world.z();
    
    return output;
  }

  [[nodiscard]] static Eigen::Matrix4d PoseToTransformMatrix(
      const geometry_msgs::msg::Pose& pose) noexcept {
    
    Eigen::Quaterniond quaternion(
        pose.orientation.w,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z);
    quaternion.normalize();
    
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = quaternion.toRotationMatrix();
    transform(0, 3) = pose.position.x;
    transform(1, 3) = pose.position.y;
    transform(2, 3) = pose.position.z;
    
    return transform;
  }

  static void HsvToRgb(
      double hue,
      double saturation,
      double value,
      float& r,
      float& g,
      float& b) noexcept {
    
    const double i = std::floor(hue * 6.0);
    const double f = hue * 6.0 - i;
    const double p = value * (1.0 - saturation);
    const double q = value * (1.0 - f * saturation);
    const double t = value * (1.0 - (1.0 - f) * saturation);
    
    switch (static_cast<int>(i) % 6) {
      case 0: r = value; g = t; b = p; break;
      case 1: r = q; g = value; b = p; break;
      case 2: r = p; g = value; b = t; break;
      case 3: r = p; g = q; b = value; break;
      case 4: r = t; g = p; b = value; break;
      case 5: r = value; g = p; b = q; break;
    }
  }

  [[nodiscard]] static VisualizationColor GetLabelColor(
      ObstacleLabel label) noexcept {
    switch (label) {
      case ObstacleLabel::kStatic:
        return VisualizationColor::Static();
      case ObstacleLabel::kDynamic:
        return VisualizationColor::Dynamic();
      case ObstacleLabel::kUnknown:
      default:
        return VisualizationColor::Unknown();
    }
  }

  [[nodiscard]] static const char* GetLabelString(ObstacleLabel label) noexcept {
    switch (label) {
      case ObstacleLabel::kStatic: return "S";
      case ObstacleLabel::kDynamic: return "D";
      case ObstacleLabel::kUnknown:
      default: return "U";
    }
  }

  void LogDetectionResults(
      const std::vector<geometry_msgs::msg::PointStamped>& centers,
      const std::vector<geometry_msgs::msg::PointStamped>& static_list,
      const std::vector<geometry_msgs::msg::PointStamped>& dynamic_list) {
    
    const auto object_pairs = object_buffer_.GetObjectPoseSnapshot();
    
    RCLCPP_INFO(this->get_logger(), "Clusters: %zu", centers.size());
    RCLCPP_INFO(this->get_logger(), "Static: %zu", static_list.size());
    RCLCPP_INFO(this->get_logger(), "Dynamic: %zu", dynamic_list.size());
    RCLCPP_INFO(this->get_logger(),
                "Object frames: %zu (last frame points: %zu)",
                object_pairs.size(),
                object_pairs.empty() ? 0u : object_pairs.back().first.size());
  }

  // ========== Member Variables ==========

  Parameters params_;
  
  // TF
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  
  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr static_obstacle_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr dynamic_obstacle_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr dbscan_visualization_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr wall_accumulation_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr current_scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_history_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr aligned_history_markers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr footprint_pub_;
  
  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr wall_sub_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr marker_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_only_sub_;
  
  message_filters::Subscriber<sensor_msgs::msg::LaserScan> scan_sub_;
  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  
  // Data buffers
  MapManager scan_buffer_;
  MapManager object_buffer_;
  std::deque<geometry_msgs::msg::PointStamped> candidate_points_;
  
  // State
  nav_msgs::msg::Odometry::ConstSharedPtr latest_odom_;
  rclcpp::Time last_transform_stamp_{0, 0, RCL_ROS_TIME};
  
  // Kalman filter state
  bool kalman_initialized_{false};
  std::array<double, 4> kf_state_{0.0, 0.0, 0.0, 0.0};  // [x, y, vx, vy]
  std::array<std::array<double, 4>, 4> kf_covariance_{};
  
  rclcpp::Time last_kf_time_;
  rclcpp::Time last_measurement_time_;
  
  // Tracking state
  std::pair<double, double> previous_position_{0.0, 0.0};
  double previous_heading_{0.0};
  bool has_previous_position_{false};
  
  // Detector
  DynamicObjectDetector detector_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ObstacleDetectorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}