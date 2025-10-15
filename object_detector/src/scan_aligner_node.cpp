// scan_aligner_node.cpp (pair 기반으로 수정)
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

#include <Eigen/Dense>

#include "map_manager_pair.hpp"

class ScanAligner : public rclcpp::Node
{
public:
  ScanAligner() : Node("scan_aligner_node"), map_manager_(100)
  {
    scan_sub_.subscribe(this, "/scan", rmw_qos_profile_sensor_data);
    odom_sub_.subscribe(this, "/odom", rmw_qos_profile_sensor_data);

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(1000), scan_sub_, odom_sub_);
    sync_->registerCallback(std::bind(&ScanAligner::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

    current_scan_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/current_scan_pcl", 10);
    aligned_history_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/aligned_history_scans", 10);

    RCLCPP_INFO(this->get_logger(), "Scan Aligner Node has been started.");
  }

private:
  void syncCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg,
                    const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg)
  {
    auto local_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    local_cloud->reserve(scan_msg->ranges.size());

    for (size_t i = 0; i < scan_msg->ranges.size(); ++i)
    {
      const double range = scan_msg->ranges[i];
      if (!std::isfinite(range)) continue;

      const double angle = scan_msg->angle_min + i * scan_msg->angle_increment;

      pcl::PointXYZI pt;
      pt.x = static_cast<float>(range * std::cos(angle));
      pt.y = static_cast<float>(range * std::sin(angle));
      pt.z = 0.0f;
      pt.intensity = 255.0f;
      local_cloud->push_back(pt);
    }

    sensor_msgs::msg::PointCloud2 current_cloud_msg;
    pcl::toROSMsg(*local_cloud, current_cloud_msg);
    current_cloud_msg.header = scan_msg->header;
    current_scan_pub_->publish(current_cloud_msg);

    // 핵심: 스캔과 해당 시점의 pose를 페어로 저장
    map_manager_.addCloudWithPose(
      std::make_shared<sensor_msgs::msg::PointCloud2>(current_cloud_msg),
      odom_msg->pose.pose
    );

    // 페어 스냅샷으로 바로 정렬
    auto pairs = map_manager_.snapshot_pairs();
    if (pairs.size() < 2) return;

    const auto& current_pose = pairs.back().second;
    Eigen::Matrix4d T_map_curr = MapManager::poseToMatrix(current_pose);
    Eigen::Matrix4d T_curr_map = T_map_curr.inverse();

    pcl::PointCloud<pcl::PointXYZI> all_aligned_scans;

    const size_t frames = pairs.size() - 1; // 현재 프레임 제외한 과거 프레임 수
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

  message_filters::Subscriber<sensor_msgs::msg::LaserScan> scan_sub_;
  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;

  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::LaserScan, nav_msgs::msg::Odometry>;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr current_scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_history_pub_;

  MapManager map_manager_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ScanAligner>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
