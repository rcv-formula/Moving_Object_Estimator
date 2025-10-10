// scan_aligner_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>

// 메시지 동기화를 위한 헤더
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// PCL 및 변환 관련 헤더
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

// 제공해주셨던 MapManager 헤더를 그대로 사용합니다.
#include "map_manager.hpp" // ★ 기존 map_manager.hpp 파일을 그대로 사용

class ScanAligner : public rclcpp::Node
{
public:
  ScanAligner() : Node("scan_aligner_node"), map_manager_(10) // 최대 10개 프레임 저장
  {
    // Message Filter Subscriber 초기화
    scan_sub_.subscribe(this, "/scan", rmw_qos_profile_sensor_data);
    odom_sub_.subscribe(this, "/odom_airio", rmw_qos_profile_sensor_data);

    // Approximate Time Synchronizer 설정
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), scan_sub_, odom_sub_);
    sync_->registerCallback(std::bind(&ScanAligner::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

    // 정렬된 과거 스캔을 시각화하기 위한 Publisher 생성
    aligned_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/aligned_prev_scan", 10);

    RCLCPP_INFO(this->get_logger(), "Scan Aligner Node has been started.");
  }

private:
  // 동기화 콜백 함수
  void syncCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg,
                  const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg)
  {
    // 1. LaserScan 데이터를 라이다 'Local' 좌표계의 PCL PointCloud로 변환
    auto local_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    local_cloud->reserve(scan_msg->ranges.size());

    for (size_t i = 0; i < scan_msg->ranges.size(); ++i)
    {
      const double range = scan_msg->ranges[i];
      if (!std::isfinite(range)) continue;
      
      const double angle = scan_msg->angle_min + i * scan_msg->angle_increment;
      
      pcl::PointXYZ pt;
      pt.x = static_cast<float>(range * std::cos(angle));
      pt.y = static_cast<float>(range * std::sin(angle));
      pt.z = 0.0f;
      local_cloud->push_back(pt);
    }
    
    // 2. 변환되지 않은 Local Scan과 그 당시의 Pose를 MapManager에 저장
    // ★ addScan 함수를 사용하기 위해 map_manager.hpp를 수정해야 합니다. 
    //    기존 addCloud를 PCL cloud를 받도록 수정하거나, 여기서 ROS Msg -> PCL 변환을 직접 수행합니다.
    //    여기서는 기존 MapManager의 addCloud 함수를 활용하기 위해 PointCloud2 메시지로 다시 변환합니다.
    sensor_msgs::msg::PointCloud2 temp_cloud_msg;
    pcl::toROSMsg(*local_cloud, temp_cloud_msg);
    auto shared_temp_cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>(temp_cloud_msg);
    
    // 이 부분은 map_manager.hpp의 구현에 따라 달라집니다.
    // 만약 map_manager.hpp가 (local_scan, pose)를 저장하도록 수정되었다면 그 코드를 사용합니다.
    // 여기서는 제공된 map_manager.hpp를 그대로 쓴다고 가정하고, 
    // alignment 로직을 위해 pose를 별도로 저장합니다.
    
    map_manager_.addCloud(shared_temp_cloud_msg);
    if(pose_deque_.size() >= map_manager_.max_deque_size()) {
        pose_deque_.pop_front();
    }
    pose_deque_.push_back(odom_msg->pose.pose);


    // 3. Alignment 확인: 저장된 스캔이 2개 이상일 때 정렬 및 시각화
    if (map_manager_.size() >= 2) {
      auto scan_history = map_manager_.snapshot();
      
      const auto& current_scan_pcl  = scan_history.back(); // 가장 최신 스캔
      const auto& previous_scan_pcl = scan_history[scan_history.size() - 2]; // 바로 직전 스캔
      
      const auto& current_pose = pose_deque_.back();
      const auto& previous_pose = pose_deque_[pose_deque_.size() - 2];

      // SE(3) 변환 행렬 계산
      Eigen::Matrix4d T_map_curr = MapManager::poseToMatrix(current_pose);
      Eigen::Matrix4d T_map_prev = MapManager::poseToMatrix(previous_pose);
      Eigen::Matrix4d T_curr_map = T_map_curr.inverse();
      Eigen::Matrix4d T_curr_prev = T_curr_map * T_map_prev;
      
      // 이전 스캔을 T_curr_prev를 이용해 현재 센서 좌표계로 변환
      pcl::PointCloud<pcl::PointXYZ> aligned_prev_cloud;
      pcl::transformPointCloud(*(previous_scan_pcl), aligned_prev_cloud, T_curr_prev.cast<float>());
      
      // 변환된(정렬된) 이전 스캔을 PointCloud2 메시지로 만들어 퍼블리시
      sensor_msgs::msg::PointCloud2 aligned_msg;
      pcl::toROSMsg(aligned_prev_cloud, aligned_msg);
      aligned_msg.header.stamp = scan_msg->header.stamp;
      // frame_id는 현재 스캔의 좌표계(예: 'base_link')가 되어야 함.
      // odom 메시지의 child_frame_id를 사용하면 일반적임.
      aligned_msg.header.frame_id = odom_msg->child_frame_id;
      aligned_cloud_pub_->publish(aligned_msg);
    }
  }

  // --- ROS I/O ---
  message_filters::Subscriber<sensor_msgs::msg::LaserScan> scan_sub_;
  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
  
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::LaserScan, nav_msgs::msg::Odometry>;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_cloud_pub_;

  // --- Map Manager ---
  MapManager map_manager_;
  std::deque<geometry_msgs::msg::Pose> pose_deque_; // Pose를 별도로 저장
};


int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ScanAligner>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}