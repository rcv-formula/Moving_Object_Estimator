// scan_processor_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>  // ★ 추가
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <vector>
#include <limits>
#include <deque>
#include <algorithm>

//PCL include
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>

//Eigen
#include <Eigen/Dense>

// 2차원 점 구조체 정의
struct Point2D
{
  double x, y;
  float i;
};

class ScanProcessor : public rclcpp::Node
{
public:
  ScanProcessor() : Node("scan_processor"),
                    tf_buffer_(this->get_clock()),
                    tf_listener_(tf_buffer_)
  {
    // 스캔 데이터 필터링 파라미터 [m, radian]
    this->declare_parameter<double>("scan_range_min", 0.0);
    this->declare_parameter<double>("scan_range_max", 10.0);
    this->declare_parameter<double>("scan_angle_min", -M_PI / 3);
    this->declare_parameter<double>("scan_angle_max", M_PI / 3);

    // DB clustering 파라미터
    this->declare_parameter<int>("min_cluster_points", 5);
    this->declare_parameter<double>("dbscan_epsilon", 0.5);
    this->declare_parameter<int>("dbscan_max_points", 50);

    // 벽 필터링 관련 파라미터
    this->declare_parameter<double>("wall_distance_threshold", 0.1);
    this->declare_parameter<double>("wall_line_max_error", 0.05);
    this->declare_parameter<int>("min_wall_cluster_points", 5);
    this->declare_parameter<double>("wall_length_threshold", 0.35);

    // 멀리 있는 장애물 필터링 관련 파라미터
    this->declare_parameter<double>("far_obstacle_distance_threshold", 5.0);
    this->declare_parameter<int>("far_obstacle_min_points", 3);
    this->declare_parameter<double>("dynamic_wall_gap_factor", 1.5);
    this->declare_parameter<int>("sor_mean_k", 10);
    this->declare_parameter<double>("sor_stddev_mul", 2.0);

    // 파라미터 값 로드
    this->get_parameter("scan_range_min", scan_range_min_);
    this->get_parameter("scan_range_max", scan_range_max_);
    this->get_parameter("scan_angle_min", scan_angle_min_);
    this->get_parameter("scan_angle_max", scan_angle_max_);
    this->get_parameter("min_cluster_points", min_cluster_points_);
    this->get_parameter("dbscan_epsilon", dbscan_epsilon_);
    this->get_parameter("dbscan_max_points", dbscan_max_points_);
    this->get_parameter("wall_distance_threshold", wall_distance_threshold_);
    this->get_parameter("wall_line_max_error", wall_line_max_error_);
    this->get_parameter("min_wall_cluster_points", min_wall_cluster_points_);
    this->get_parameter("wall_length_threshold", wall_length_threshold_);
    this->get_parameter("far_obstacle_distance_threshold", far_obstacle_distance_threshold_);
    this->get_parameter("far_obstacle_min_points", far_obstacle_min_points_);
    this->get_parameter("dynamic_wall_gap_factor", dynamic_wall_gap_factor_);
    this->get_parameter("sor_mean_k", sor_mean_k_);
    this->get_parameter("sor_stddev_mul", sor_stddev_mul_);

    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 20, std::bind(&ScanProcessor::scanCallback, this, std::placeholders::_1));

    candidate_pub_     = this->create_publisher<geometry_msgs::msg::PointStamped>("/obstacle_candidates", 20);
    filtered_scan_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("filtered_scan_points", 20);
    wall_scan_pub_     = this->create_publisher<sensor_msgs::msg::PointCloud2>("wall_points", 20);
    marker_array_pub_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("/detected_obstacles", 10);
    sor_scan_pub_     = this->create_publisher<sensor_msgs::msg::PointCloud2>("/sor_filtered_points", 10);
  }

private:
  // DBSCAN 함수
  std::vector<std::vector<Point2D>> dbscanClustering(const std::vector<Point2D> &points, double eps, int minPts)
  {
    std::vector<std::vector<Point2D>> clusters;
    const int n = points.size();
    if (n == 0) return clusters;

    std::vector<bool> visited(n, false);
    std::vector<int> cluster_ids(n, -1);
    int cluster_id = 0;

    double eps_sq = eps * eps;

    auto regionQuery = [&](int index) -> std::vector<int>
    {
      std::vector<int> ret;
      for (int i = 0; i < n; i++)
      {
        if (i == index) continue;
        double dx = points[index].x - points[i].x;
        double dy = points[index].y - points[i].y;
        double dist_sq = dx * dx + dy * dy;
        if (dist_sq <= eps_sq) ret.push_back(i);
      }
      return ret;
    };

    for (int i = 0; i < n; i++)
    {
      if (visited[i]) continue;

      visited[i] = true;
      std::vector<int> neighbors = regionQuery(i);
      if (neighbors.size() < static_cast<size_t>(minPts))
      {
        continue; // 노이즈
      }

      cluster_ids[i] = cluster_id;
      std::vector<int> seeds = neighbors;
      std::vector<bool> inSeeds(n, false);
      for (int idx : seeds) inSeeds[idx] = true;

      for (size_t j = 0; j < seeds.size(); j++)
      {
        int curr = seeds[j];
        if (!visited[curr])
        {
          visited[curr] = true;
          std::vector<int> curr_neighbors = regionQuery(curr);
          if (curr_neighbors.size() >= static_cast<size_t>(minPts))
          {
            for (int neighbor : curr_neighbors)
            {
              if (!inSeeds[neighbor])
              {
                seeds.push_back(neighbor);
                inSeeds[neighbor] = true;
              }
            }
          }
        }
        if (cluster_ids[curr] == -1)
        {
          cluster_ids[curr] = cluster_id;
        }
      }
      cluster_id++;
    }

    clusters.resize(cluster_id);
    for (int i = 0; i < n; i++)
    {
      int id = cluster_ids[i];
      if (id != -1) clusters[id].push_back(points[i]);
    }

    for (auto &cluster : clusters)
    {
      if (cluster.size() > static_cast<size_t>(dbscan_max_points_))
      {
        double sumX = 0, sumY = 0;
        for (const auto &pt : cluster) { sumX += pt.x; sumY += pt.y; }
        double centerX = sumX / cluster.size();
        double centerY = sumY / cluster.size();

        std::sort(cluster.begin(), cluster.end(), [=](const Point2D &a, const Point2D &b)
        {
          double dxA = a.x - centerX, dyA = a.y - centerY;
          double dxB = b.x - centerX, dyB = b.y - centerY;
          return (dxA * dxA + dyA * dyA) < (dxB * dxB + dyB * dyB);
        });
        cluster.resize(dbscan_max_points_);
      }
    }

    return clusters;
  }

  bool isWallCluster(const std::vector<Point2D> &cluster)
  {
    if (cluster.size() < static_cast<size_t>(min_wall_cluster_points_)) return false;

    const Point2D &p1 = cluster.front();
    const Point2D &p2 = cluster.back();
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double cluster_length = std::sqrt(dx * dx + dy * dy);

    if (cluster_length > wall_length_threshold_) return true;

    double norm = cluster_length;
    if (norm < 1e-6) return false;

    double max_error = 0.0;
    for (const auto &pt : cluster)
    {
      double error = std::fabs(dy * pt.x - dx * pt.y + p2.x * p1.y - p2.y * p1.x) / norm;
      if (error > max_error) max_error = error;
    }
    return max_error < wall_line_max_error_;
  }

  std::pair<std::vector<Point2D>,std::vector<Point2D>> filterWallPoints(
      std::vector<Point2D> &points, double sensor_x, double sensor_y, double angle_increment)
  {
    std::vector<Point2D> filtered_points;
    std::vector<Point2D> wall_points;
    if (points.empty()) return {filtered_points,wall_points};

    std::vector<double> sensor_dists;
    sensor_dists.reserve(points.size());
    for (const auto &pt : points)
    {
      double dx = pt.x - sensor_x;
      double dy = pt.y - sensor_y;
      sensor_dists.push_back(std::sqrt(dx * dx + dy * dy));
    }

    std::vector<Point2D> current_cluster;
    current_cluster.push_back(points[0]);

    double factor = angle_increment * dynamic_wall_gap_factor_;
    int cluster_cnt = 1;

    for (size_t i = 1; i < points.size(); ++i)
    {
      double dx = points[i].x - points[i - 1].x;
      double dy = points[i].y - points[i - 1].y;
      double actual_gap_sq = dx * dx + dy * dy;

      double avg_range = (sensor_dists[i - 1] + sensor_dists[i]) * 0.5;
      double expected_gap = avg_range * factor;
      double dynamic_threshold = wall_distance_threshold_ + expected_gap;
      double dynamic_threshold_sq = dynamic_threshold * dynamic_threshold;

      if (actual_gap_sq < dynamic_threshold_sq)
      {
        points[i].i = static_cast<float>(cluster_cnt);
        current_cluster.push_back(points[i]);
      }
      else
      {
        if (!isWallCluster(current_cluster))
        {
          filtered_points.insert(filtered_points.end(),
                                 current_cluster.begin(), current_cluster.end());
        } else {
          cluster_cnt++;
          wall_points.insert(wall_points.end(),
                             current_cluster.begin(), current_cluster.end());
        }
        current_cluster.clear();
        current_cluster.push_back(points[i]);
      }
    }

    if (!isWallCluster(current_cluster))
    {
      filtered_points.insert(filtered_points.end(),
                             current_cluster.begin(), current_cluster.end());
    } else {
      wall_points.insert(wall_points.end(),
                         current_cluster.begin(), current_cluster.end());
    }
    return {filtered_points, wall_points};
  }

  std::vector<geometry_msgs::msg::PointStamped> interpolateCandidates(
      const geometry_msgs::msg::PointStamped &p1,
      const geometry_msgs::msg::PointStamped &p2,
      int num_points)
  {
    std::vector<geometry_msgs::msg::PointStamped> interp;
    for (int i = 1; i <= num_points; i++)
    {
      double fraction = static_cast<double>(i) / (num_points + 1);
      geometry_msgs::msg::PointStamped new_point;
      new_point.header.frame_id = p1.header.frame_id;
      new_point.header.stamp = p1.header.stamp;
      new_point.point.x = p1.point.x + fraction * (p2.point.x - p1.point.x);
      new_point.point.y = p1.point.y + fraction * (p2.point.y - p1.point.y);
      new_point.point.z = 0.0;
      interp.push_back(new_point);
    }
    return interp;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr invert_to_PointXYZI(const std::vector<Point2D>& points) {
    auto cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    cloud->reserve(points.size());
    for (const auto& pt : points) {
      pcl::PointXYZI p;
      p.x = static_cast<float>(pt.x);
      p.y = static_cast<float>(pt.y);
      p.z = 0.0f;
      p.intensity = 0.0f;
      cloud->push_back(p);
    }
    return cloud;
  }

  std::vector<Point2D> invert_to_Point2D(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) {
    std::vector<Point2D> points;
    points.reserve(cloud->size());
    for (const auto& p : cloud->points) {
      points.push_back(Point2D{p.x, p.y});
    }
    return points;
  }

  std::vector<Point2D> do_sor(const std::vector<Point2D>& points, int mean_k = 10, double stdmul = 4.0) {
    auto cloud = invert_to_PointXYZI(points);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(mean_k);
    sor.setStddevMulThresh(stdmul);
    sor.setNegative(false);
    sor.filter(*cloud_filtered);

    return invert_to_Point2D(cloud_filtered);
  }

  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg)
  {
    std::vector<Point2D> points;
    points.reserve(scan_msg->ranges.size());
    for (size_t i = 0; i < scan_msg->ranges.size(); ++i)
    {
      double range = scan_msg->ranges[i];
      double angle = scan_msg->angle_min + i * scan_msg->angle_increment;
      if (range < scan_range_min_ || range > scan_range_max_ ||
          angle < scan_angle_min_ || angle > scan_angle_max_)
        continue;
      points.push_back({range * std::cos(angle), range * std::sin(angle)});
    }

    std::vector<Point2D> map_points;
    map_points.reserve(points.size());
    if (!tf_buffer_.canTransform("map", scan_msg->header.frame_id, scan_msg->header.stamp, tf2::durationFromSec(0.1)))
    {
      RCLCPP_WARN(this->get_logger(), "Transform not available");
      return;
    }
    geometry_msgs::msg::TransformStamped transformStamped;
    try
    {
      transformStamped = tf_buffer_.lookupTransform("map", scan_msg->header.frame_id, scan_msg->header.stamp, tf2::durationFromSec(0.1));
    }
    catch (tf2::TransformException &ex)
    {
      RCLCPP_WARN(this->get_logger(), "Lookup transform failed: %s", ex.what());
      return;
    }

    for (const auto &pt : points)
    {
      geometry_msgs::msg::PointStamped laser_pt, map_pt;
      laser_pt.header = scan_msg->header;
      laser_pt.point.x = pt.x;
      laser_pt.point.y = pt.y;
      laser_pt.point.z = 0.0;

      try
      {
        tf2::doTransform(laser_pt, map_pt, transformStamped);
        map_points.push_back({map_pt.point.x, map_pt.point.y});
      }
      catch (tf2::TransformException &ex)
      {
        RCLCPP_WARN(this->get_logger(), "doTransform failed: %s", ex.what());
      }
    }

    geometry_msgs::msg::TransformStamped sensor_transform;
    try
    {
      sensor_transform = tf_buffer_.lookupTransform("map", scan_msg->header.frame_id, scan_msg->header.stamp, tf2::durationFromSec(0.1));
    }
    catch (tf2::TransformException &ex)
    {
      RCLCPP_WARN(this->get_logger(), "Transform(Laser Sensor location in map frame) lookup failed: %s", ex.what());
      sensor_transform.transform.translation.x = 0.0;
      sensor_transform.transform.translation.y = 0.0;
    }
    double sensor_x = sensor_transform.transform.translation.x;
    double sensor_y = sensor_transform.transform.translation.y;
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in = invert_to_PointXYZI(map_points);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_sor(new pcl::PointCloud<pcl::PointXYZI>);
    {
      pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
      sor.setInputCloud(cloud_in);
      sor.setMeanK(sor_mean_k_);
      sor.setStddevMulThresh(sor_stddev_mul_);
      sor.setNegative(false); 
      sor.filter(*cloud_sor);
    }

    sensor_msgs::msg::PointCloud2 sor_msg;
    pcl::toROSMsg(*cloud_sor, sor_msg);
    sor_msg.header.frame_id = "map";
    sor_msg.header.stamp    = scan_msg->header.stamp;
    sor_scan_pub_->publish(sor_msg);
    
    map_points = invert_to_Point2D(cloud_sor);

    auto fw = filterWallPoints(map_points, sensor_x, sensor_y, scan_msg->angle_increment);
    std::vector<Point2D> filtered_map_points = std::move(fw.first);
    std::vector<Point2D> wall_map_points     = std::move(fw.second);

    // 벽 포인트 RViz 확인용
    size_t wall_cloud_size = wall_map_points.size();
    pcl::PointCloud<pcl::PointXYZI>::Ptr wall_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    wall_cloud->reserve(wall_cloud_size);
    for(size_t i = 0; i < wall_cloud_size; ++i){
      pcl::PointXYZI pt;
      pt.x = wall_map_points[i].x;
      pt.y = wall_map_points[i].y;
      pt.intensity = wall_map_points[i].i;
      pt.z = 0.0;
      wall_cloud->push_back(pt);
    }
    wall_cloud->width = wall_cloud->size();
    wall_cloud->height = 1;
    wall_cloud->is_dense = false;

    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*wall_cloud, cloud_msg);
    cloud_msg.header.frame_id = "map";
    cloud_msg.header.stamp = scan_msg->header.stamp;
    wall_scan_pub_->publish(cloud_msg);

    // 필터링된 점들(=벽 제외)을 RViz에 점 군집으로 표시
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = scan_msg->header.stamp;
    marker.ns = "filtered_scan";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;
    for (const auto &pt : filtered_map_points)
    {
      geometry_msgs::msg::Point p;
      p.x = pt.x; p.y = pt.y; p.z = 0.0;
      marker.points.push_back(p);
    }
    filtered_scan_pub_->publish(marker);

    // === DBSCAN으로 장애물 클러스터링 (벽 제외된 점들 대상으로) ===
    auto clusters = dbscanClustering(filtered_map_points, dbscan_epsilon_, min_cluster_points_);

    // === (추가) 이번 프레임에서 검출된 모든 클러스터를 MarkerArray로 한 번에 퍼블리시 ===
    visualization_msgs::msg::MarkerArray marr;

    // 먼저 이전 프레임 마커 지우기(잔상 방지)
    {
      visualization_msgs::msg::Marker del;
      del.header.frame_id = "map";
      del.header.stamp    = scan_msg->header.stamp;
      del.ns   = "detected_obstacles";
      del.id   = 0;
      del.action = visualization_msgs::msg::Marker::DELETEALL;
      marr.markers.push_back(del);
    }

    int mid = 0;

    // 각 클러스터에 대해 후보 중심 퍼블리시 + 마커 추가
    for (const auto &cluster : clusters)
    {
      if (cluster.empty()) continue;

      // 클러스터 중심 계산
      double sum_x = 0, sum_y = 0;
      for (const auto &pt : cluster) { sum_x += pt.x; sum_y += pt.y; }
      const double cx = sum_x / cluster.size();
      const double cy = sum_y / cluster.size();

      // (기존 유지) 후보 후보점 퍼블리시
      geometry_msgs::msg::PointStamped candidate;
      candidate.header.frame_id = "map";
      candidate.header.stamp = scan_msg->header.stamp;
      candidate.point.x = cx;
      candidate.point.y = cy;
      candidate.point.z = 0.0;

      // 멀리 있는 후보의 최소 점 개수 조건 유지
      double dx = cx - sensor_x;
      double dy = cy - sensor_y;
      double candidate_distance = std::sqrt(dx * dx + dy * dy);
      if (candidate_distance >= far_obstacle_distance_threshold_ &&
          cluster.size() < static_cast<size_t>(far_obstacle_min_points_))
      {
        continue; // 후보/마커 모두 스킵
      }
      candidate_pub_->publish(candidate);

      // (추가) 마커 생성(클러스터 대표점 표시)
      visualization_msgs::msg::Marker m;
      m.header.frame_id = "map";
      m.header.stamp    = scan_msg->header.stamp;
      m.ns   = "detected_obstacles";
      m.id   = mid++;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = cx;
      m.pose.position.y = cy;
      m.pose.position.z = 0.0;
      m.pose.orientation.w = 1.0;

      m.scale.x = 0.18;
      m.scale.y = 0.18;
      m.scale.z = 0.18;

      // 벽은 이미 제외된 점들로 군집을 만들었으므로 모두 같은 색(노란색 계열)로 표기
      m.color.r = 1.0f; m.color.g = 0.85f; m.color.b = 0.2f; m.color.a = 1.0f;

      m.lifetime = rclcpp::Duration::from_seconds(0.25);
      marr.markers.push_back(m);
    }

    // 한 번만 퍼블리시
    marker_array_pub_->publish(marr);
  }

  // --- ROS I/O ---
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr candidate_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr filtered_scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr wall_scan_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_pub_; // ★ 추가
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr sor_scan_pub_;

  // --- TF ---
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // --- Params ---
  double scan_range_min_, scan_range_max_, scan_angle_min_, scan_angle_max_;
  int min_cluster_points_, dbscan_max_points_;
  double dbscan_epsilon_;
  double wall_distance_threshold_;
  double wall_line_max_error_;
  int min_wall_cluster_points_;
  double wall_length_threshold_;

  double far_obstacle_distance_threshold_;
  int far_obstacle_min_points_;
  double dynamic_wall_gap_factor_;

  int    sor_mean_k_{8};
  double sor_stddev_mul_{2.0};
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ScanProcessor>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
