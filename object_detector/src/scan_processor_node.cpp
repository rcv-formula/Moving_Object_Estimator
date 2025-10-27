/**
 * @file scan_processor_node.cpp
 * @brief ROS2 node for processing laser scan data and detecting obstacles
 * 
 * This node performs the following operations:
 * - Filters laser scan data by range and angle
 * - Removes statistical outliers using PCL
 * - Detects and removes wall points
 * - Clusters remaining points using DBSCAN
 * - Publishes obstacle candidates
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <Eigen/Dense>

#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
#include <optional>
#include <array>

namespace {

/// @brief Configuration constants
constexpr double kDefaultPointScale = 0.05;
constexpr double kDefaultMarkerLifetime = 0.25;  // seconds
constexpr float kDefaultMarkerAlpha = 1.0f;

/// @brief 2D point with intensity
struct Point2D {
  double x{0.0};
  double y{0.0};
  float intensity{0.0f};
  
  [[nodiscard]] double DistanceTo(const Point2D& other) const noexcept {
    const double dx = x - other.x;
    const double dy = y - other.y;
    return std::hypot(dx, dy);
  }
};

/// @brief Color definition
struct Color {
  float r{1.0f};
  float g{1.0f};
  float b{1.0f};
  float a{1.0f};
  
  static constexpr Color Green() { return {0.0f, 1.0f, 0.0f, 1.0f}; }
  static constexpr Color Yellow() { return {1.0f, 0.85f, 0.2f, 1.0f}; }
};

}  // namespace

/**
 * @brief ROS2 node for laser scan processing and obstacle detection
 */
class ScanProcessorNode : public rclcpp::Node {
public:
  /**
   * @brief Configuration parameters
   */
  struct Parameters {
    // Scan filtering
    double scan_range_min{0.0};
    double scan_range_max{10.0};
    double scan_angle_min{-M_PI / 3.0};
    double scan_angle_max{M_PI / 3.0};
    
    // DBSCAN clustering
    int min_cluster_points{5};
    double dbscan_epsilon{0.5};
    int dbscan_max_points{50};
    
    // Wall filtering
    double wall_distance_threshold{0.1};
    double wall_line_max_error{0.05};
    int min_wall_cluster_points{5};
    double wall_length_threshold{0.35};
    
    // Far obstacle filtering
    double far_obstacle_distance_threshold{5.0};
    int far_obstacle_min_points{3};
    double dynamic_wall_gap_factor{1.5};
    
    // Statistical Outlier Removal (SOR)
    int sor_mean_k{10};
    double sor_stddev_multiplier{2.0};
  };

  ScanProcessorNode() : Node("scan_processor") {
    DeclareParameters();
    LoadParameters();
    InitializePublishersAndSubscribers();
    
    RCLCPP_INFO(this->get_logger(), 
                "ScanProcessorNode initialized successfully");
  }

private:
  void DeclareParameters() {
    // Scan filtering parameters
    this->declare_parameter<double>("scan_range_min", params_.scan_range_min);
    this->declare_parameter<double>("scan_range_max", params_.scan_range_max);
    this->declare_parameter<double>("scan_angle_min", params_.scan_angle_min);
    this->declare_parameter<double>("scan_angle_max", params_.scan_angle_max);
    
    // DBSCAN parameters
    this->declare_parameter<int>("min_cluster_points", params_.min_cluster_points);
    this->declare_parameter<double>("dbscan_epsilon", params_.dbscan_epsilon);
    this->declare_parameter<int>("dbscan_max_points", params_.dbscan_max_points);
    
    // Wall filtering parameters
    this->declare_parameter<double>("wall_distance_threshold", 
                                     params_.wall_distance_threshold);
    this->declare_parameter<double>("wall_line_max_error", 
                                     params_.wall_line_max_error);
    this->declare_parameter<int>("min_wall_cluster_points", 
                                 params_.min_wall_cluster_points);
    this->declare_parameter<double>("wall_length_threshold", 
                                     params_.wall_length_threshold);
    
    // Far obstacle parameters
    this->declare_parameter<double>("far_obstacle_distance_threshold", 
                                     params_.far_obstacle_distance_threshold);
    this->declare_parameter<int>("far_obstacle_min_points", 
                                 params_.far_obstacle_min_points);
    this->declare_parameter<double>("dynamic_wall_gap_factor", 
                                     params_.dynamic_wall_gap_factor);
    
    // SOR parameters
    this->declare_parameter<int>("sor_mean_k", params_.sor_mean_k);
    this->declare_parameter<double>("sor_stddev_mul", params_.sor_stddev_multiplier);
  }

  void LoadParameters() {
    this->get_parameter("scan_range_min", params_.scan_range_min);
    this->get_parameter("scan_range_max", params_.scan_range_max);
    this->get_parameter("scan_angle_min", params_.scan_angle_min);
    this->get_parameter("scan_angle_max", params_.scan_angle_max);
    
    this->get_parameter("min_cluster_points", params_.min_cluster_points);
    this->get_parameter("dbscan_epsilon", params_.dbscan_epsilon);
    this->get_parameter("dbscan_max_points", params_.dbscan_max_points);
    
    this->get_parameter("wall_distance_threshold", params_.wall_distance_threshold);
    this->get_parameter("wall_line_max_error", params_.wall_line_max_error);
    this->get_parameter("min_wall_cluster_points", params_.min_wall_cluster_points);
    this->get_parameter("wall_length_threshold", params_.wall_length_threshold);
    
    this->get_parameter("far_obstacle_distance_threshold", 
                        params_.far_obstacle_distance_threshold);
    this->get_parameter("far_obstacle_min_points", params_.far_obstacle_min_points);
    this->get_parameter("dynamic_wall_gap_factor", params_.dynamic_wall_gap_factor);
    
    this->get_parameter("sor_mean_k", params_.sor_mean_k);
    this->get_parameter("sor_stddev_mul", params_.sor_stddev_multiplier);
  }

  void InitializePublishersAndSubscribers() {
    // Subscribers
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan",
        rclcpp::SensorDataQoS(),
        std::bind(&ScanProcessorNode::ScanCallback, this, std::placeholders::_1));
    
    // Publishers
    candidate_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
        "/obstacle_candidates", 20);
    
    filtered_scan_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "filtered_scan_points", 20);
    
    wall_scan_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "wall_points", 20);
    
    marker_array_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/detected_obstacles", 10);
    
    sor_scan_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/sor_filtered_points", 10);
  }

  void ScanCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg) {
    // 1. Convert laser scan to local points
    std::vector<Point2D> local_points = ConvertScanToPoints(scan_msg);
    
    // 2. Apply Statistical Outlier Removal
    local_points = ApplyStatisticalOutlierRemoval(local_points, scan_msg);
    
    // 3. Filter wall points
    const Point2D sensor_origin{0.0, 0.0, 0.0f};
    auto [filtered_points, wall_points] = FilterWallPoints(
        local_points,
        sensor_origin,
        scan_msg->angle_increment);
    
    // 4. Publish wall points
    PublishWallPoints(wall_points, scan_msg);
    
    // 5. Publish filtered scan visualization
    PublishFilteredScan(filtered_points, scan_msg);
    
    // 6. Cluster obstacles using DBSCAN
    auto clusters = ClusterPointsDbscan(filtered_points);
    
    // 7. Publish obstacle candidates and markers
    PublishObstacleCandidates(clusters, sensor_origin, scan_msg);
  }

  [[nodiscard]] std::vector<Point2D> ConvertScanToPoints(
      const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg) const {
    
    std::vector<Point2D> points;
    points.reserve(scan_msg->ranges.size());
    
    for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
      const double range = scan_msg->ranges[i];
      const double angle = scan_msg->angle_min + i * scan_msg->angle_increment;
      
      if (!IsValidScanPoint(range, angle)) {
        continue;
      }
      
      points.push_back({
        range * std::cos(angle),
        range * std::sin(angle),
        0.0f
      });
    }
    
    return points;
  }

  [[nodiscard]] bool IsValidScanPoint(double range, double angle) const noexcept {
    return std::isfinite(range) &&
           range >= params_.scan_range_min &&
           range <= params_.scan_range_max &&
           angle >= params_.scan_angle_min &&
           angle <= params_.scan_angle_max;
  }

  [[nodiscard]] std::vector<Point2D> ApplyStatisticalOutlierRemoval(
      const std::vector<Point2D>& input_points,
      const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg) {
    
    // Convert to PCL cloud
    auto pcl_cloud = ConvertToPclCloud(input_points);
    
    // Apply SOR filter
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor_filter;
    sor_filter.setInputCloud(pcl_cloud);
    sor_filter.setMeanK(params_.sor_mean_k);
    sor_filter.setStddevMulThresh(params_.sor_stddev_multiplier);
    sor_filter.setNegative(false);
    sor_filter.filter(*filtered_cloud);
    
    // Publish SOR result
    PublishPclCloud(filtered_cloud, scan_msg, sor_scan_pub_);
    
    // Convert back to Point2D
    return ConvertFromPclCloud(filtered_cloud);
  }

  [[nodiscard]] pcl::PointCloud<pcl::PointXYZI>::Ptr ConvertToPclCloud(
      const std::vector<Point2D>& points) const {
    
    auto cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>);
    cloud->reserve(points.size());
    
    for (const auto& pt : points) {
      pcl::PointXYZI pcl_point;
      pcl_point.x = pt.x;
      pcl_point.y = pt.y;
      pcl_point.z = 0.0f;
      pcl_point.intensity = pt.intensity;
      cloud->push_back(pcl_point);
    }
    
    return cloud;
  }

  [[nodiscard]] std::vector<Point2D> ConvertFromPclCloud(
      const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) const {
    
    std::vector<Point2D> points;
    points.reserve(cloud->size());
    
    for (const auto& pcl_point : cloud->points) {
      points.push_back({pcl_point.x, pcl_point.y, pcl_point.intensity});
    }
    
    return points;
  }

  void PublishPclCloud(
      const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud,
      const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg,
      const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher) {
    
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = scan_msg->header.frame_id;
    cloud_msg.header.stamp = scan_msg->header.stamp;
    publisher->publish(cloud_msg);
  }

  [[nodiscard]] std::pair<std::vector<Point2D>, std::vector<Point2D>> FilterWallPoints(
      const std::vector<Point2D>& points,
      const Point2D& sensor_origin,
      double angle_increment) const {
    
    if (points.empty()) {
      return {{}, {}};
    }
    
    std::vector<Point2D> filtered_points;
    std::vector<Point2D> wall_points;
    
    // Calculate distances from sensor
    std::vector<double> sensor_distances;
    sensor_distances.reserve(points.size());
    for (const auto& pt : points) {
      sensor_distances.push_back(pt.DistanceTo(sensor_origin));
    }
    
    // Group consecutive points
    std::vector<Point2D> current_cluster;
    current_cluster.push_back(points[0]);
    
    const double gap_factor = angle_increment * params_.dynamic_wall_gap_factor;
    int cluster_id = 1;
    
    for (size_t i = 1; i < points.size(); ++i) {
      double actual_gap_sq = points[i].DistanceTo(points[i - 1]);
      actual_gap_sq *= actual_gap_sq;
      
      const double avg_range = 0.5 * (sensor_distances[i - 1] + sensor_distances[i]);
      const double expected_gap = avg_range * gap_factor;
      const double dynamic_threshold = params_.wall_distance_threshold + expected_gap;
      const double dynamic_threshold_sq = dynamic_threshold * dynamic_threshold;
      
      if (actual_gap_sq < dynamic_threshold_sq) {
        auto point_with_id = points[i];
        point_with_id.intensity = static_cast<float>(cluster_id);
        current_cluster.push_back(point_with_id);
      } else {
        // Process current cluster
        if (IsWallCluster(current_cluster)) {
          cluster_id++;
          wall_points.insert(wall_points.end(), 
                           current_cluster.begin(), 
                           current_cluster.end());
        } else {
          filtered_points.insert(filtered_points.end(),
                               current_cluster.begin(),
                               current_cluster.end());
        }
        
        current_cluster.clear();
        current_cluster.push_back(points[i]);
      }
    }
    
    // Process final cluster
    if (IsWallCluster(current_cluster)) {
      wall_points.insert(wall_points.end(), 
                        current_cluster.begin(), 
                        current_cluster.end());
    } else {
      filtered_points.insert(filtered_points.end(),
                           current_cluster.begin(),
                           current_cluster.end());
    }
    
    return {filtered_points, wall_points};
  }

  [[nodiscard]] bool IsWallCluster(const std::vector<Point2D>& cluster) const noexcept {
    if (cluster.size() < static_cast<size_t>(params_.min_wall_cluster_points)) {
      return false;
    }
    
    const Point2D& p1 = cluster.front();
    const Point2D& p2 = cluster.back();
    const double dx = p2.x - p1.x;
    const double dy = p2.y - p1.y;
    const double length = std::hypot(dx, dy);
    
    // Check if segment is long enough to be a wall
    if (length > params_.wall_length_threshold) {
      return true;
    }
    
    if (length < 1e-6) {
      return false;
    }
    
    // Check if points fit a line well
    double max_error = 0.0;
    for (const auto& pt : cluster) {
      const double error = std::abs(dy * pt.x - dx * pt.y + p2.x * p1.y - p2.y * p1.x) / length;
      max_error = std::max(max_error, error);
    }
    
    return max_error < params_.wall_line_max_error;
  }

  void PublishWallPoints(
      const std::vector<Point2D>& wall_points,
      const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg) {
    
    auto wall_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>);
    wall_cloud->reserve(wall_points.size());
    
    for (const auto& pt : wall_points) {
      pcl::PointXYZI pcl_point;
      pcl_point.x = pt.x;
      pcl_point.y = pt.y;
      pcl_point.z = 0.0f;
      pcl_point.intensity = pt.intensity;
      wall_cloud->push_back(pcl_point);
    }
    
    PublishPclCloud(wall_cloud, scan_msg, wall_scan_pub_);
  }

  void PublishFilteredScan(
      const std::vector<Point2D>& filtered_points,
      const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg) {
    
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = scan_msg->header.frame_id;
    marker.header.stamp = scan_msg->header.stamp;
    marker.ns = "filtered_scan";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = kDefaultPointScale;
    marker.scale.y = kDefaultPointScale;
    
    const auto color = Color::Green();
    marker.color.r = color.r;
    marker.color.g = color.g;
    marker.color.b = color.b;
    marker.color.a = color.a;
    
    marker.points.reserve(filtered_points.size());
    for (const auto& pt : filtered_points) {
      geometry_msgs::msg::Point point;
      point.x = pt.x;
      point.y = pt.y;
      point.z = 0.0;
      marker.points.push_back(point);
    }
    
    filtered_scan_pub_->publish(marker);
  }

  [[nodiscard]] std::vector<std::vector<Point2D>> ClusterPointsDbscan(
      const std::vector<Point2D>& points) const {
    
    const int num_points = static_cast<int>(points.size());
    if (num_points == 0) {
      return {};
    }
    
    std::vector<bool> visited(num_points, false);
    std::vector<int> cluster_ids(num_points, -1);
    int current_cluster_id = 0;
    
    const double eps_sq = params_.dbscan_epsilon * params_.dbscan_epsilon;
    
    auto region_query = [&](int index) {
      std::vector<int> neighbors;
      neighbors.reserve(32);
      
      for (int j = 0; j < num_points; ++j) {
        if (j == index) continue;
        
        const double dist_sq = points[index].DistanceTo(points[j]);
        if (dist_sq * dist_sq <= eps_sq) {
          neighbors.push_back(j);
        }
      }
      
      return neighbors;
    };
    
    // DBSCAN algorithm
    for (int i = 0; i < num_points; ++i) {
      if (visited[i]) {
        continue;
      }
      
      visited[i] = true;
      auto neighbors = region_query(i);
      
      if (neighbors.size() < static_cast<size_t>(params_.min_cluster_points)) {
        continue; // Noise point
      }
      
      cluster_ids[i] = current_cluster_id;
      std::vector<int> seed_set = neighbors;
      std::vector<bool> in_seed_set(num_points, false);
      for (int idx : seed_set) {
        in_seed_set[idx] = true;
      }
      
      for (size_t k = 0; k < seed_set.size(); ++k) {
        const int current_idx = seed_set[k];
        
        if (!visited[current_idx]) {
          visited[current_idx] = true;
          auto current_neighbors = region_query(current_idx);
          
          if (current_neighbors.size() >= static_cast<size_t>(params_.min_cluster_points)) {
            for (int neighbor_idx : current_neighbors) {
              if (!in_seed_set[neighbor_idx]) {
                seed_set.push_back(neighbor_idx);
                in_seed_set[neighbor_idx] = true;
              }
            }
          }
        }
        
        if (cluster_ids[current_idx] == -1) {
          cluster_ids[current_idx] = current_cluster_id;
        }
      }
      
      current_cluster_id++;
    }
    
    // Extract clusters and limit size
    std::vector<std::vector<Point2D>> clusters(current_cluster_id);
    for (int i = 0; i < num_points; ++i) {
      if (cluster_ids[i] != -1) {
        clusters[cluster_ids[i]].push_back(points[i]);
      }
    }
    
    // Limit cluster size
    for (auto& cluster : clusters) {
      if (cluster.size() > static_cast<size_t>(params_.dbscan_max_points)) {
        LimitClusterSize(cluster, params_.dbscan_max_points);
      }
    }
    
    return clusters;
  }

  void LimitClusterSize(std::vector<Point2D>& cluster, int max_size) const {
    // Calculate centroid
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (const auto& pt : cluster) {
      sum_x += pt.x;
      sum_y += pt.y;
    }
    const Point2D centroid{sum_x / cluster.size(), sum_y / cluster.size(), 0.0f};
    
    // Sort by distance to centroid
    std::sort(cluster.begin(), cluster.end(),
              [&centroid](const Point2D& a, const Point2D& b) {
                return a.DistanceTo(centroid) < b.DistanceTo(centroid);
              });
    
    cluster.resize(max_size);
  }

  void PublishObstacleCandidates(
      const std::vector<std::vector<Point2D>>& clusters,
      const Point2D& sensor_origin,
      const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg) {
    
    visualization_msgs::msg::MarkerArray marker_array;
    
    // Delete all previous markers
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header.frame_id = scan_msg->header.frame_id;
    delete_marker.header.stamp = scan_msg->header.stamp;
    delete_marker.ns = "detected_obstacles";
    delete_marker.id = 0;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_marker);
    
    int marker_id = 0;
    
    for (const auto& cluster : clusters) {
      if (cluster.empty()) {
        continue;
      }
      
      // Calculate centroid
      double sum_x = 0.0;
      double sum_y = 0.0;
      for (const auto& pt : cluster) {
        sum_x += pt.x;
        sum_y += pt.y;
      }
      const double centroid_x = sum_x / cluster.size();
      const double centroid_y = sum_y / cluster.size();
      
      // Check far obstacle filter
      const double distance = std::hypot(centroid_x - sensor_origin.x, 
                                        centroid_y - sensor_origin.y);
      if (distance >= params_.far_obstacle_distance_threshold &&
          cluster.size() < static_cast<size_t>(params_.far_obstacle_min_points)) {
        continue;
      }
      
      // Publish candidate point
      geometry_msgs::msg::PointStamped candidate;
      candidate.header.frame_id = scan_msg->header.frame_id;
      candidate.header.stamp = scan_msg->header.stamp;
      candidate.point.x = centroid_x;
      candidate.point.y = centroid_y;
      candidate.point.z = 0.0;
      candidate_pub_->publish(candidate);
      
      // Create visualization marker
      visualization_msgs::msg::Marker marker;
      marker.header = candidate.header;
      marker.ns = "detected_obstacles";
      marker.id = marker_id++;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position = candidate.point;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = marker.scale.y = marker.scale.z = 0.18;
      
      const auto color = Color::Yellow();
      marker.color.r = color.r;
      marker.color.g = color.g;
      marker.color.b = color.b;
      marker.color.a = color.a;
      
      marker.lifetime = rclcpp::Duration::from_seconds(kDefaultMarkerLifetime);
      marker_array.markers.push_back(marker);
    }
    
    marker_array_pub_->publish(marker_array);
  }

  // Member variables
  Parameters params_;
  
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr candidate_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr filtered_scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr wall_scan_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr sor_scan_pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ScanProcessorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}