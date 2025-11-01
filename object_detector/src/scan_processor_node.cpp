// scan_processor_node.cpp (ApproximateTime: /scan + /odom 동기화 + /processed_odom 재발행)
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
#include "object_detector/msg/marker_array_stamped.hpp"

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>

// Eigen (필요시 남겨둠)
#include <Eigen/Dense>

// message_filters (ApproximateTime)
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

struct Point2D { double x, y; float i; };

class ScanProcessor : public rclcpp::Node
{
public:
  using Laser = sensor_msgs::msg::LaserScan;
  using Odom  = nav_msgs::msg::Odometry;
  using ApproxPolicy = message_filters::sync_policies::ApproximateTime<Laser, Odom>;

  ScanProcessor() : Node("scan_processor")
  {
    // 스캔 필터링 파라미터
    this->declare_parameter<double>("scan_range_min", 0.0);
    this->declare_parameter<double>("scan_range_max", 10.0);
    this->declare_parameter<double>("scan_angle_min", -M_PI / 3);
    this->declare_parameter<double>("scan_angle_max",  M_PI / 3);

    // DBSCAN/벽/SOR 파라미터
    this->declare_parameter<int>("min_cluster_points", 5);
    this->declare_parameter<double>("dbscan_epsilon", 0.5);
    this->declare_parameter<int>("dbscan_max_points", 50);

    this->declare_parameter<double>("wall_distance_threshold", 0.1);
    this->declare_parameter<double>("wall_line_max_error", 0.05);
    this->declare_parameter<int>("min_wall_cluster_points", 5);
    this->declare_parameter<double>("wall_length_threshold", 0.35);

    this->declare_parameter<double>("far_obstacle_distance_threshold", 5.0);
    this->declare_parameter<int>("far_obstacle_min_points", 3);
    this->declare_parameter<double>("dynamic_wall_gap_factor", 1.5);
    this->declare_parameter<int>("sor_mean_k", 10);
    this->declare_parameter<double>("sor_stddev_mul", 2.0);

    // 파라미터 로드
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

    // 퍼블리셔
    candidate_pub_     = this->create_publisher<geometry_msgs::msg::PointStamped>("/obstacle_candidates", 20);
    filtered_scan_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("filtered_scan_points", 20);
    wall_scan_pub_     = this->create_publisher<sensor_msgs::msg::PointCloud2>("wall_points", 20);
    marker_array_pub_  = this->create_publisher<object_detector::msg::MarkerArrayStamped>("/detected_obstacles", 10);
    scan_pub_          = this->create_publisher<sensor_msgs::msg::PointCloud2>("/processed_scan", 10);
    processed_odom_pub_= this->create_publisher<nav_msgs::msg::Odometry>("/processed_odom", 10);

    // === message_filters 구독자 + ApproximateTime 동기화 ===
    // 센서 데이터 QoS (best_effort, volatile)
    rclcpp::SensorDataQoS sensor_qos;
    auto rmw_qos = sensor_qos.get_rmw_qos_profile();

    scan_mf_sub_.subscribe(this, "/scan", rmw_qos);
    odom_mf_sub_.subscribe(this, "/odom", rmw_qos);

    sync_ = std::make_shared<message_filters::Synchronizer<ApproxPolicy>>(ApproxPolicy(1000), scan_mf_sub_, odom_mf_sub_);
    sync_->registerCallback(std::bind(&ScanProcessor::syncCallback, this,
                                      std::placeholders::_1, std::placeholders::_2));
  }

private:
  // 동기화 콜백: LaserScan + Odom
  void syncCallback(const Laser::ConstSharedPtr& scan_msg,
                    const Odom::ConstSharedPtr& odom_msg)
  {
    // --- /processed_odom: scan과 같은 timestamp로 발행 ---
    {
      nav_msgs::msg::Odometry odom_out = *odom_msg;          // 원본 내용 복사
      odom_out.header.stamp = scan_msg->header.stamp;        // 스캔과 동일한 stamp로 맞춤
      // odom_out.header.frame_id 는 원본 유지 (필요 시 scan_msg->header.frame_id로 바꿔도 됨)
      processed_odom_pub_->publish(odom_out);
    }
    

    // 1) LaserScan → LiDAR 로컬 포인트
    std::vector<Point2D> local_points;
    local_points.reserve(scan_msg->ranges.size());
    for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
      const double r = scan_msg->ranges[i];
      const double a = scan_msg->angle_min + i * scan_msg->angle_increment;
      if (r < scan_range_min_ || r > scan_range_max_ ||
          a < scan_angle_min_ || a > scan_angle_max_ ||
          !std::isfinite(r)) continue;
      local_points.push_back({r * std::cos(a), r * std::sin(a), 0.0f});
    }

    // 센서 원점(로컬)
    constexpr double sensor_x = 0.0, sensor_y = 0.0;

    // 3) 벽 필터링
    auto [filtered_local_points, wall_local_points] =
      filterWallPoints(local_points, sensor_x, sensor_y, scan_msg->angle_increment);

    // 4) 벽 포인트 퍼블리시 (로컬)
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr wall_cloud(new pcl::PointCloud<pcl::PointXYZI>);
      wall_cloud->reserve(wall_local_points.size());
      for (const auto& w : wall_local_points) {
        pcl::PointXYZI p; p.x = w.x; p.y = w.y; p.z = 0.0f; p.intensity = w.i; wall_cloud->push_back(p);
      }
      sensor_msgs::msg::PointCloud2 wall_msg;
      pcl::toROSMsg(*wall_cloud, wall_msg);
      wall_msg.header.frame_id = scan_msg->header.frame_id;
      wall_msg.header.stamp    = scan_msg->header.stamp;
      wall_scan_pub_->publish(wall_msg);
    }

    // 5) 필터링 점 마커 (로컬)
    {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = scan_msg->header.frame_id;
      marker.header.stamp    = scan_msg->header.stamp;
      marker.ns = "filtered_scan";
      marker.id = 0;
      marker.type = visualization_msgs::msg::Marker::POINTS;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.scale.x = 0.05; marker.scale.y = 0.05;
      marker.color.r = 0.0;  marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 1.0;
      for (const auto &pt : filtered_local_points) {
        geometry_msgs::msg::Point p; p.x = pt.x; p.y = pt.y; p.z = 0.0;
        marker.points.push_back(p);
      }
      filtered_scan_pub_->publish(marker);
    }

    // 6) DBSCAN (로컬)
    auto clusters = dbscanClustering(filtered_local_points, dbscan_epsilon_, min_cluster_points_);

    // 7) 검출 마커/후보 (로컬)
    object_detector::msg::MarkerArrayStamped marr;
    {
      marr.header = scan_msg->header;
      visualization_msgs::msg::Marker del;
      del.header.frame_id = scan_msg->header.frame_id;
      del.header.stamp    = scan_msg->header.stamp;
      del.ns = "detected_obstacles";
      del.id = 0;
      del.action = visualization_msgs::msg::Marker::DELETEALL;
      marr.markers.push_back(del);
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in = toPointCloud(local_points);

    sensor_msgs::msg::PointCloud2 sor_msg;
    pcl::toROSMsg(*cloud_in, sor_msg);
    sor_msg.header.frame_id = scan_msg->header.frame_id;
    sor_msg.header.stamp    = scan_msg->header.stamp;

    int mid = 0;
    for (const auto &cluster : clusters) {
      if (cluster.empty()) continue;

      double sx = 0, sy = 0;
      for (const auto &pt : cluster) { sx += pt.x; sy += pt.y; }
      const double cx = sx / cluster.size();
      const double cy = sy / cluster.size();

      // 후보 퍼블리시 (로컬)
      geometry_msgs::msg::PointStamped candidate;
      candidate.header.frame_id = scan_msg->header.frame_id;
      candidate.header.stamp    = scan_msg->header.stamp;
      candidate.point.x = cx; candidate.point.y = cy; candidate.point.z = 0.0;

      const double dx = cx - sensor_x, dy = cy - sensor_y;
      const double dist = std::sqrt(dx*dx + dy*dy);
      if (dist >= far_obstacle_distance_threshold_ &&
          cluster.size() < static_cast<size_t>(far_obstacle_min_points_)) {
        continue;
      }
      candidate_pub_->publish(candidate);

      visualization_msgs::msg::Marker m;
      m.header.frame_id = scan_msg->header.frame_id;
      m.header.stamp    = scan_msg->header.stamp;
      m.ns = "detected_obstacles";
      m.id = mid++;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = cx; m.pose.position.y = cy; m.pose.position.z = 0.0;
      m.pose.orientation.w = 1.0;
      m.scale.x = 0.18; m.scale.y = 0.18; m.scale.z = 0.18;
      m.color.r = 1.0f; m.color.g = 0.85f; m.color.b = 0.2f; m.color.a = 1.0f;
      m.lifetime = rclcpp::Duration::from_seconds(0.25);
      marr.markers.push_back(m);
    }

    marker_array_pub_->publish(marr);
    scan_pub_->publish(sor_msg);
  }

  // ====== 헬퍼 ======
  pcl::PointCloud<pcl::PointXYZI>::Ptr toPointCloud(const std::vector<Point2D>& points) const
  {
    auto cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    cloud->reserve(points.size());
    for (const auto& pt : points) {
      pcl::PointXYZI p; p.x = pt.x; p.y = pt.y; p.z = 0.0f; p.intensity = 0.0f; cloud->push_back(p);
    }
    return cloud;
  }

  std::vector<Point2D> toPoint2D(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) const
  {
    std::vector<Point2D> points; points.reserve(cloud->size());
    for (const auto& p : cloud->points) points.push_back(Point2D{p.x, p.y, 0.0f});
    return points;
  }

  std::vector<std::vector<Point2D>> dbscanClustering(const std::vector<Point2D> &points, double eps, int minPts) const
  {
    std::vector<std::vector<Point2D>> clusters;
    const int n = static_cast<int>(points.size());
    if (n == 0) return clusters;

    std::vector<bool> visited(n, false);
    std::vector<int> cluster_ids(n, -1);
    int cluster_id = 0;
    const double eps_sq = eps * eps;

    auto regionQuery = [&](int idx) {
      std::vector<int> ret; ret.reserve(32);
      for (int j = 0; j < n; ++j) {
        if (j == idx) continue;
        const double dx = points[idx].x - points[j].x;
        const double dy = points[idx].y - points[j].y;
        if (dx*dx + dy*dy <= eps_sq) ret.push_back(j);
      }
      return ret;
    };

    for (int i = 0; i < n; ++i) {
      if (visited[i]) continue;
      visited[i] = true;
      auto neighbors = regionQuery(i);
      if (neighbors.size() < static_cast<size_t>(minPts)) continue;

      cluster_ids[i] = cluster_id;
      std::vector<int> seeds = neighbors;
      std::vector<bool> inSeeds(n, false);
      for (int idx : seeds) inSeeds[idx] = true;

      for (size_t k = 0; k < seeds.size(); ++k) {
        int cur = seeds[k];
        if (!visited[cur]) {
          visited[cur] = true;
          auto nbs = regionQuery(cur);
          if (nbs.size() >= static_cast<size_t>(minPts)) {
            for (int nb : nbs) if (!inSeeds[nb]) { seeds.push_back(nb); inSeeds[nb] = true; }
          }
        }
        if (cluster_ids[cur] == -1) cluster_ids[cur] = cluster_id;
      }
      cluster_id++;
    }

    clusters.resize(cluster_id);
    for (int i = 0; i < n; ++i) if (cluster_ids[i] != -1) clusters[cluster_ids[i]].push_back(points[i]);

    // 최대 포인트 제한
    for (auto &cluster : clusters) {
      if (cluster.size() > static_cast<size_t>(dbscan_max_points_)) {
        double sx = 0, sy = 0; for (auto &p: cluster){ sx+=p.x; sy+=p.y; }
        const double cx = sx/cluster.size(), cy = sy/cluster.size();
        std::sort(cluster.begin(), cluster.end(), [=](const Point2D&a, const Point2D&b){
          const double dax=a.x-cx, day=a.y-cy, dbx=b.x-cx, dby=b.y-cy;
          return (dax*dax+day*day) < (dbx*dbx+dby*dby);
        });
        cluster.resize(dbscan_max_points_);
      }
    }
    return clusters;
  }

  bool isWallCluster(const std::vector<Point2D> &cluster) const
  {
    if (cluster.size() < static_cast<size_t>(min_wall_cluster_points_)) return false;

    const Point2D &p1 = cluster.front();
    const Point2D &p2 = cluster.back();
    const double dx = p2.x - p1.x, dy = p2.y - p1.y;
    const double len = std::sqrt(dx*dx + dy*dy);
    if (len > wall_length_threshold_) return true;
    if (len < 1e-6) return false;

    double max_error = 0.0;
    for (const auto &pt : cluster) {
      const double err = std::fabs(dy*pt.x - dx*pt.y + p2.x*p1.y - p2.y*p1.x) / len;
      if (err > max_error) max_error = err;
    }
    return max_error < wall_line_max_error_;
  }

  std::pair<std::vector<Point2D>,std::vector<Point2D>>
  filterWallPoints(std::vector<Point2D> &points, double sensor_x, double sensor_y, double angle_increment) const
  {
    std::vector<Point2D> filtered_points, wall_points;
    if (points.empty()) return {filtered_points, wall_points};

    std::vector<double> sensor_dists; sensor_dists.reserve(points.size());
    for (const auto &pt : points) {
      const double dx = pt.x - sensor_x, dy = pt.y - sensor_y;
      sensor_dists.push_back(std::sqrt(dx*dx + dy*dy));
    }

    std::vector<Point2D> current_cluster; current_cluster.push_back(points[0]);
    const double factor = angle_increment * dynamic_wall_gap_factor_;
    int cluster_cnt = 1;

    for (size_t i = 1; i < points.size(); ++i)
    {
      const double dx = points[i].x - points[i-1].x;
      const double dy = points[i].y - points[i-1].y;
      const double actual_gap_sq = dx*dx + dy*dy;

      const double avg_range = 0.5*(sensor_dists[i-1] + sensor_dists[i]);
      const double expected_gap = avg_range * factor;
      const double dynamic_threshold = wall_distance_threshold_ + expected_gap;
      const double dynamic_threshold_sq = dynamic_threshold * dynamic_threshold;

      if (actual_gap_sq < dynamic_threshold_sq) {
        points[i].i = static_cast<float>(cluster_cnt);
        current_cluster.push_back(points[i]);
      } else {
        if (!isWallCluster(current_cluster))
          filtered_points.insert(filtered_points.end(), current_cluster.begin(), current_cluster.end());
        else {
          cluster_cnt++;
          wall_points.insert(wall_points.end(), current_cluster.begin(), current_cluster.end());
        }
        current_cluster.clear();
        current_cluster.push_back(points[i]);
      }
    }

    if (!isWallCluster(current_cluster))
      filtered_points.insert(filtered_points.end(), current_cluster.begin(), current_cluster.end());
    else
      wall_points.insert(wall_points.end(), current_cluster.begin(), current_cluster.end());

    return {filtered_points, wall_points};
  }

  // === message_filters 구독자/동기화자 ===
  message_filters::Subscriber<Laser> scan_mf_sub_;
  message_filters::Subscriber<Odom>  odom_mf_sub_;
  std::shared_ptr<message_filters::Synchronizer<ApproxPolicy>> sync_;

  // 퍼블리셔
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr      candidate_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr       filtered_scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr         wall_scan_pub_;
  rclcpp::Publisher<object_detector::msg::MarkerArrayStamped>::SharedPtr  marker_array_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr         scan_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr               processed_odom_pub_;

  // 파라미터
  double scan_range_min_, scan_range_max_, scan_angle_min_, scan_angle_max_;
  int    min_cluster_points_, dbscan_max_points_;
  double dbscan_epsilon_;
  double wall_distance_threshold_, wall_line_max_error_;
  int    min_wall_cluster_points_;
  double wall_length_threshold_;
  double far_obstacle_distance_threshold_; int far_obstacle_min_points_;
  double dynamic_wall_gap_factor_;
  int    sor_mean_k_; double sor_stddev_mul_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ScanProcessor>());
  rclcpp::shutdown();
  return 0;
}
