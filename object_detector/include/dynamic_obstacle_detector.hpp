#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <limits>
#include <optional>
#include <array>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <rclcpp/rclcpp.hpp>

#include "map_manager_pair.hpp"

/**
 * @brief Classification result for dynamic obstacle detection
 */
enum class ObstacleType {
  kUnknown = -1,  ///< Could not determine type
  kStatic = 0,    ///< Static obstacle
  kDynamic = 1    ///< Dynamic/moving obstacle
};

/**
 * @brief Color representation in RGB
 */
struct RgbColor {
  float r{1.0f};
  float g{1.0f};
  float b{1.0f};
  float a{1.0f};
};

/**
 * @brief Detects and classifies dynamic objects using footprint analysis
 * 
 * This class analyzes historical point cloud data to determine if detected
 * objects are static or dynamic based on their motion footprint over time.
 */
class DynamicObjectDetector {
public:
  using PointVector = std::vector<geometry_msgs::msg::PointStamped>;
  using ObjectPosePair = std::pair<PointVector, geometry_msgs::msg::Pose>;
  using TransformedFrames = std::vector<PointVector>;
  using MarkerPublisher = rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr;

  /**
   * @brief Configuration parameters for the detector
   */
  struct Config {
    double proximity_threshold{0.05};      ///< [m] Very close distance threshold
    double boundary_threshold{0.1};        ///< [m] Boundary zone upper limit
    int target_frame_index{5};             ///< Target historical frame index
    double dbscan_epsilon{0.12};           ///< [m] DBSCAN clustering radius
    int dbscan_min_points{3};              ///< Minimum points for DBSCAN core
    double search_radius{0.40};            ///< [m] Candidate search radius
    double motion_threshold{0.10};         ///< [m] Motion detection threshold (10cm)
    int min_distinct_frames{5};            ///< Minimum number of distinct frames required
  };

  DynamicObjectDetector() = default;
  explicit DynamicObjectDetector(const Config& config) noexcept : config_(config) {}

  // Disable copying, allow moving
  DynamicObjectDetector(const DynamicObjectDetector&) = delete;
  DynamicObjectDetector& operator=(const DynamicObjectDetector&) = delete;
  DynamicObjectDetector(DynamicObjectDetector&&) noexcept = default;
  DynamicObjectDetector& operator=(DynamicObjectDetector&&) noexcept = default;
  
  ~DynamicObjectDetector() = default;

  /**
   * @brief Set configuration parameters
   */
  void SetConfig(const Config& config) noexcept { config_ = config; }
  [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }

  /**
   * @brief Set proximity threshold (clamped to non-negative)
   */
  void SetProximityThreshold(double value) noexcept {
    config_.proximity_threshold = std::max(0.0, value);
  }

  /**
   * @brief Set boundary threshold (clamped to non-negative)
   */
  void SetBoundaryThreshold(double value) noexcept {
    config_.boundary_threshold = std::max(0.0, value);
  }

  /**
   * @brief Set target frame index for analysis
   */
  void SetTargetFrameIndex(int index) noexcept {
    config_.target_frame_index = index;
  }

  [[nodiscard]] double GetProximityThreshold() const noexcept {
    return config_.proximity_threshold;
  }
  
  [[nodiscard]] double GetBoundaryThreshold() const noexcept {
    return config_.boundary_threshold;
  }
  
  [[nodiscard]] int GetTargetFrameIndex() const noexcept {
    return config_.target_frame_index;
  }

  /**
   * @brief Transform historical object detections to current frame
   * 
   * @param historical_objects Object-pose pairs from historical frames
   * @param current_pose Current robot pose in map frame
   * @return Transformed point collections, each representing one historical frame
   */
  [[nodiscard]] TransformedFrames TransformHistoricalObjects(
      const std::vector<ObjectPosePair>& historical_objects,
      const geometry_msgs::msg::Pose& current_pose) const {
    
    TransformedFrames output;
    output.reserve(historical_objects.size());

    const Eigen::Matrix4d transform_map_to_current = 
        MapManager::PoseToMatrix(current_pose);
    const Eigen::Matrix4d transform_current_to_map = 
        transform_map_to_current.inverse();

    for (const auto& [frame_points, historical_pose] : historical_objects) {
      // Transform: historical_local -> map -> current_local
      const Eigen::Matrix4d transform_map_to_historical = 
          MapManager::PoseToMatrix(historical_pose);
      const Eigen::Matrix4d transform_current_to_historical = 
          transform_current_to_map * transform_map_to_historical;

      PointVector transformed_frame;
      transformed_frame.reserve(frame_points.size());

      for (const auto& point_stamped : frame_points) {
        const Eigen::Vector4d point_historical(
            point_stamped.point.x,
            point_stamped.point.y,
            point_stamped.point.z,
            1.0
        );
        
        const Eigen::Vector4d point_current = 
            transform_current_to_historical * point_historical;

        geometry_msgs::msg::PointStamped output_point;
        output_point.header.stamp = point_stamped.header.stamp;
        output_point.point.x = point_current.x();
        output_point.point.y = point_current.y();
        output_point.point.z = point_current.z();

        transformed_frame.emplace_back(std::move(output_point));
      }
      
      output.emplace_back(std::move(transformed_frame));
    }
    
    return output;
  }

  /**
   * @brief Visualize aligned historical frames as markers
   * 
   * @param aligned_frames Transformed historical frames in current coordinate system
   * @param frame_id ROS frame ID for markers
   * @param stamp Timestamp for markers
   * @param publisher Marker array publisher
   * @param point_scale Size of point markers
   * @param lifetime_sec Marker lifetime in seconds
   */
  void PublishAlignedFramesMarkers(
      const TransformedFrames& aligned_frames,
      const std::string& frame_id,
      const rclcpp::Time& stamp,
      const MarkerPublisher& publisher,
      double point_scale = 0.06,
      double lifetime_sec = 0.25) const {
    
    if (!publisher || aligned_frames.empty()) {
      return;
    }

    visualization_msgs::msg::MarkerArray marker_array;

    // Delete previous markers
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header.frame_id = frame_id;
    delete_marker.header.stamp = stamp;
    delete_marker.ns = "aligned_frames";
    delete_marker.id = 0;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_marker);

    const int total_frames = static_cast<int>(aligned_frames.size());
    int marker_id = 1;

    for (int frame_idx = 0; frame_idx < total_frames; ++frame_idx) {
      const auto& frame_points = aligned_frames[frame_idx];
      if (frame_points.empty()) {
        continue;
      }

      // Create sphere list marker for points
      visualization_msgs::msg::Marker sphere_marker;
      sphere_marker.header.frame_id = frame_id;
      sphere_marker.header.stamp = stamp;
      sphere_marker.ns = "aligned_frames";
      sphere_marker.id = marker_id++;
      sphere_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
      sphere_marker.action = visualization_msgs::msg::Marker::ADD;
      sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = point_scale;

      // Color: older frames darker, newer frames brighter
      const double hue = (total_frames > 1) 
          ? static_cast<double>(frame_idx) / (total_frames - 1) 
          : 0.0;
      const double value = 0.95 * (0.35 + 0.65 * (1.0 - static_cast<double>(frame_idx) / 
          std::max(1, total_frames - 1)));
      
      const RgbColor color = HsvToRgb(hue, 0.9, value);
      sphere_marker.color.r = color.r;
      sphere_marker.color.g = color.g;
      sphere_marker.color.b = color.b;
      sphere_marker.color.a = 0.95f;

      // Add points to marker
      sphere_marker.points.reserve(frame_points.size());
      for (const auto& point_stamped : frame_points) {
        geometry_msgs::msg::Point point;
        point.x = point_stamped.point.x;
        point.y = point_stamped.point.y;
        point.z = point_stamped.point.z;
        sphere_marker.points.push_back(point);
      }

      sphere_marker.lifetime = rclcpp::Duration::from_seconds(lifetime_sec);
      marker_array.markers.push_back(sphere_marker);

      // Add text marker for frame index
      visualization_msgs::msg::Marker text_marker;
      text_marker.header = sphere_marker.header;
      text_marker.ns = "aligned_frames_text";
      text_marker.id = marker_id++;
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;
      text_marker.scale.z = point_scale * 3.0;
      text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0f;
      text_marker.color.a = 0.9f;

      // Calculate centroid for text position
      geometry_msgs::msg::Point centroid{};
      for (const auto& ps : frame_points) {
        centroid.x += ps.point.x;
        centroid.y += ps.point.y;
        centroid.z += ps.point.z;
      }
      centroid.x /= frame_points.size();
      centroid.y /= frame_points.size();
      centroid.z /= frame_points.size();
      centroid.z += point_scale * 2.0; // Slightly above points

      text_marker.pose.position = centroid;
      text_marker.text = "frame " + std::to_string(frame_idx) + "/" + 
                         std::to_string(total_frames - 1);
      text_marker.lifetime = rclcpp::Duration::from_seconds(lifetime_sec);
      marker_array.markers.push_back(text_marker);
    }

    publisher->publish(marker_array);
  }

  /**
   * @brief Visualize obstacle footprint with dynamic/static classification
   * 
   * @param footprint Footprint points to visualize
   * @param obstacle_type Classification result
   * @param frame_id ROS frame ID
   * @param marker_id Unique marker ID
   * @param stamp Timestamp
   * @param publisher Marker publisher
   */
  void VisualizeFootprint(
      const std::vector<geometry_msgs::msg::Point>& footprint,
      ObstacleType obstacle_type,
      const std::string& frame_id,
      int marker_id,
      const rclcpp::Time& stamp,
      const MarkerPublisher& publisher) const {
    
    if (!publisher || footprint.empty()) {
      return;
    }

    visualization_msgs::msg::MarkerArray marker_array;

    // Line strip marker
    visualization_msgs::msg::Marker line_marker;
    line_marker.header.frame_id = frame_id;
    line_marker.header.stamp = stamp;
    line_marker.ns = "footprint_line";
    line_marker.id = marker_id;
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::msg::Marker::ADD;
    line_marker.pose.orientation.w = 1.0;
    line_marker.scale.x = 0.03; // Line width
    line_marker.lifetime = rclcpp::Duration::from_seconds(0.5);

    // Set color based on classification
    if (obstacle_type == ObstacleType::kStatic) {
      line_marker.color.r = 0.0f;
      line_marker.color.g = 1.0f;
      line_marker.color.b = 0.0f;
      line_marker.color.a = 0.8f;
    } else {
      line_marker.color.r = 1.0f;
      line_marker.color.g = 0.0f;
      line_marker.color.b = 0.0f;
      line_marker.color.a = 0.9f;
    }

    // Add footprint points
    for (const auto& point : footprint) {
      line_marker.points.push_back(point);
    }

    // Close the loop if we have enough points
    if (footprint.size() > 2) {
      line_marker.points.push_back(footprint.front());
    }

    // Sphere list marker
    visualization_msgs::msg::Marker sphere_marker = line_marker;
    sphere_marker.ns = "footprint_points";
    sphere_marker.id = marker_id + 10000;
    sphere_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = 0.06;
    sphere_marker.points = footprint;

    marker_array.markers.push_back(line_marker);
    marker_array.markers.push_back(sphere_marker);
    
    publisher->publish(marker_array);
  }

  /**
   * @brief Extract nearest cluster around an object using DBSCAN
   * 
   * @param object_position Target object position
   * @param aligned_frames Historical frames aligned to current coordinate system
   * @param exclude_current Whether to exclude the most recent frame
   * @return Cluster points if found, empty vector otherwise
   */
  [[nodiscard]] std::vector<geometry_msgs::msg::Point> ExtractNearestCluster(
      const geometry_msgs::msg::Point& object_position,
      const TransformedFrames& aligned_frames,
      bool exclude_current = true) const {
    
    struct Point2DWithFrame {
      double x;
      double y;
      int frame_index;
    };

    std::vector<Point2DWithFrame> candidate_pool;
    candidate_pool.reserve(1024);

    if (aligned_frames.empty()) {
      return {};
    }

    // Collect candidate points within search radius
    const double search_radius_sq = config_.search_radius * config_.search_radius;
    const int last_frame_idx = static_cast<int>(aligned_frames.size()) - 1;
    const int end_idx = exclude_current ? last_frame_idx : (last_frame_idx + 1);

    for (int frame_idx = 0; frame_idx < end_idx; ++frame_idx) {
      for (const auto& point_stamped : aligned_frames[frame_idx]) {
        const double dx = point_stamped.point.x - object_position.x;
        const double dy = point_stamped.point.y - object_position.y;
        
        if (!std::isfinite(dx) || !std::isfinite(dy)) {
          continue;
        }
        
        if (dx * dx + dy * dy <= search_radius_sq) {
          candidate_pool.push_back({
            point_stamped.point.x,
            point_stamped.point.y,
            frame_idx
          });
        }
      }
    }

    if (candidate_pool.size() < static_cast<size_t>(config_.dbscan_min_points)) {
      return {};
    }

    // Check minimum distinct frames requirement
    if (!HasSufficientDistinctFrames(candidate_pool, end_idx)) {
      return {};
    }

    // Run DBSCAN clustering
    const auto cluster_labels = RunDbscan2D(candidate_pool);
    
    if (cluster_labels.empty()) {
      return {};
    }

    // Find cluster nearest to object
    const int nearest_cluster_id = FindNearestCluster(
        candidate_pool,
        cluster_labels,
        object_position);

    if (nearest_cluster_id < 0) {
      return {};
    }

    // Extract points from the nearest cluster
    return ExtractClusterPoints(candidate_pool, cluster_labels, nearest_cluster_id);
  }

  /**
   * @brief Classify obstacle as static or dynamic based on footprint analysis
   * 
   * @param object_position Object to classify
   * @param aligned_frames Historical frames in current coordinate system
   * @param exclude_current Whether to exclude current frame from analysis
   * @param[out] output_cluster Optional output for extracted cluster
   * @param[out] output_span Optional output for computed footprint span
   * @return Classification result
   */
  [[nodiscard]] ObstacleType ClassifyDynamicByFootprint(
      const geometry_msgs::msg::Point& object_position,
      const TransformedFrames& aligned_frames,
      bool exclude_current = true,
      std::vector<geometry_msgs::msg::Point>* output_cluster = nullptr,
      double* output_span = nullptr) const {
    
    // Extract nearest cluster
    auto cluster = ExtractNearestCluster(
        object_position,
        aligned_frames,
        exclude_current);

    if (output_cluster) {
      *output_cluster = cluster;
    }

    if (cluster.size() < static_cast<size_t>(config_.dbscan_min_points)) {
      return ObstacleType::kUnknown;
    }

    // Compute footprint span
    const double span = ComputeFootprintSpan(cluster);
    
    if (output_span) {
      *output_span = span;
    }

    if (!std::isfinite(span)) {
      return ObstacleType::kUnknown;
    }

    // Classify based on motion threshold
    return (span > config_.motion_threshold) 
        ? ObstacleType::kDynamic 
        : ObstacleType::kStatic;
  }

  /**
   * @brief Compute span (diameter) of point cluster
   * 
   * Uses a two-pass algorithm to approximate the diameter efficiently.
   * 
   * @param cluster Points forming the cluster
   * @return Maximum distance between any two points, or NaN if < 2 points
   */
  [[nodiscard]] static double ComputeFootprintSpan(
      const std::vector<geometry_msgs::msg::Point>& cluster) noexcept {
    
    if (cluster.size() < 2) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    auto compute_distance = [](const geometry_msgs::msg::Point& a,
                               const geometry_msgs::msg::Point& b) {
      const double dx = a.x - b.x;
      const double dy = a.y - b.y;
      const double dz = a.z - b.z;
      return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    // Two-pass diameter approximation
    const size_t seed_idx = 0;
    
    // First pass: find farthest point from seed
    size_t farthest_idx = seed_idx;
    double max_distance = -1.0;
    
    for (size_t i = 0; i < cluster.size(); ++i) {
      const double distance = compute_distance(cluster[i], cluster[seed_idx]);
      if (distance > max_distance) {
        max_distance = distance;
        farthest_idx = i;
      }
    }

    // Second pass: find farthest point from the first farthest point
    double span = 0.0;
    for (size_t i = 0; i < cluster.size(); ++i) {
      const double distance = compute_distance(cluster[i], cluster[farthest_idx]);
      span = std::max(span, distance);
    }

    return span;
  }

private:
  /**
   * @brief Convert HSV color to RGB
   */
  [[nodiscard]] static RgbColor HsvToRgb(
      double hue,
      double saturation,
      double value) noexcept {
    
    const double i = std::floor(hue * 6.0);
    const double f = hue * 6.0 - i;
    const double p = value * (1.0 - saturation);
    const double q = value * (1.0 - f * saturation);
    const double t = value * (1.0 - (1.0 - f) * saturation);

    RgbColor color;
    
    switch (static_cast<int>(i) % 6) {
      case 0: color.r = value; color.g = t; color.b = p; break;
      case 1: color.r = q; color.g = value; color.b = p; break;
      case 2: color.r = p; color.g = value; color.b = t; break;
      case 3: color.r = p; color.g = q; color.b = value; break;
      case 4: color.r = t; color.g = p; color.b = value; break;
      case 5: color.r = value; color.g = p; color.b = q; break;
    }
    
    return color;
  }

  /**
   * @brief Check if candidate pool has sufficient distinct frames
   */
  template<typename PointType>
  [[nodiscard]] bool HasSufficientDistinctFrames(
      const std::vector<PointType>& points,
      int max_frames) const noexcept {
    
    std::vector<bool> frame_seen(std::max(1, max_frames), false);
    int distinct_count = 0;

    for (const auto& point : points) {
      if (point.frame_index >= 0 && 
          point.frame_index < max_frames && 
          !frame_seen[point.frame_index]) {
        frame_seen[point.frame_index] = true;
        distinct_count++;
        
        if (distinct_count >= config_.min_distinct_frames) {
          return true;
        }
      }
    }

    return distinct_count >= config_.min_distinct_frames;
  }

  /**
   * @brief Run DBSCAN clustering on 2D points
   */
  template<typename PointType>
  [[nodiscard]] std::vector<int> RunDbscan2D(
      const std::vector<PointType>& points) const {
    
    const int num_points = static_cast<int>(points.size());
    std::vector<int> labels(num_points, -1); // -1: unvisited, -2: noise, >=0: cluster ID
    
    const double eps_sq = config_.dbscan_epsilon * config_.dbscan_epsilon;

    auto region_query = [&](int point_idx) {
      std::vector<int> neighbors;
      neighbors.reserve(32);
      
      for (int j = 0; j < num_points; ++j) {
        if (j == point_idx) continue;
        
        const double dx = points[point_idx].x - points[j].x;
        const double dy = points[point_idx].y - points[j].y;
        
        if (dx * dx + dy * dy <= eps_sq) {
          neighbors.push_back(j);
        }
      }
      
      return neighbors;
    };

    int cluster_id = 0;
    
    for (int i = 0; i < num_points; ++i) {
      if (labels[i] != -1) {
        continue; // Already processed
      }
      
      auto neighbors = region_query(i);
      
      if (static_cast<int>(neighbors.size()) + 1 < config_.dbscan_min_points) {
        labels[i] = -2; // Mark as noise
        continue;
      }
      
      // Start new cluster
      labels[i] = cluster_id;
      std::vector<int> seed_set = neighbors;
      
      for (size_t k = 0; k < seed_set.size(); ++k) {
        const int current_idx = seed_set[k];
        
        if (labels[current_idx] == -2) {
          labels[current_idx] = cluster_id; // Change noise to border point
        }
        
        if (labels[current_idx] != -1) {
          continue; // Already processed
        }
        
        labels[current_idx] = cluster_id;
        
        auto current_neighbors = region_query(current_idx);
        if (static_cast<int>(current_neighbors.size()) + 1 >= config_.dbscan_min_points) {
          seed_set.insert(seed_set.end(), 
                         current_neighbors.begin(), 
                         current_neighbors.end());
        }
      }
      
      cluster_id++;
    }

    return labels;
  }

  /**
   * @brief Find cluster nearest to target position
   */
  template<typename PointType>
  [[nodiscard]] int FindNearestCluster(
      const std::vector<PointType>& points,
      const std::vector<int>& labels,
      const geometry_msgs::msg::Point& target) const noexcept {
    
    const int num_points = static_cast<int>(points.size());
    const int max_cluster_id = labels.empty() ? 
        0 : *std::max_element(labels.begin(), labels.end());
    
    if (max_cluster_id < 0) {
      return -1;
    }

    int best_cluster_id = -1;
    double best_distance = std::numeric_limits<double>::infinity();

    for (int cluster_id = 0; cluster_id <= max_cluster_id; ++cluster_id) {
      double sum_x = 0.0;
      double sum_y = 0.0;
      int count = 0;

      // Compute centroid
      for (int i = 0; i < num_points; ++i) {
        if (labels[i] == cluster_id) {
          sum_x += points[i].x;
          sum_y += points[i].y;
          count++;
        }
      }

      if (count == 0) {
        continue;
      }

      const double centroid_x = sum_x / count;
      const double centroid_y = sum_y / count;
      const double dx = centroid_x - target.x;
      const double dy = centroid_y - target.y;
      const double distance = std::sqrt(dx * dx + dy * dy);

      if (distance < best_distance) {
        best_distance = distance;
        best_cluster_id = cluster_id;
      }
    }

    return best_cluster_id;
  }

  /**
   * @brief Extract all points belonging to a specific cluster
   */
  template<typename PointType>
  [[nodiscard]] static std::vector<geometry_msgs::msg::Point> ExtractClusterPoints(
      const std::vector<PointType>& points,
      const std::vector<int>& labels,
      int cluster_id) {
    
    std::vector<geometry_msgs::msg::Point> cluster_points;
    
    for (size_t i = 0; i < points.size(); ++i) {
      if (labels[i] == cluster_id) {
        geometry_msgs::msg::Point point;
        point.x = points[i].x;
        point.y = points[i].y;
        point.z = 0.0;
        cluster_points.push_back(point);
      }
    }
    
    return cluster_points;
  }

  Config config_;
};