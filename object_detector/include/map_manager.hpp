#pragma once

#include <deque>
#include <vector>
#include <memory>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp> // ★ 추가

// ★ SE(3) 행렬을 위한 Eigen 헤더 추가
#include <Eigen/Dense>

class MapManager {
public:
  using PointXYZ = pcl::PointXYZ;
  using CloudT = pcl::PointCloud<pcl::PointXYZ>;
  using CloudPtr = CloudT::Ptr;
  using CloudConstPtr = CloudT::ConstPtr;

  explicit MapManager(size_t max_deque_size = 5)
    : max_deque_size_(max_deque_size) {}
    
  // ★ Odometry Pose 메시지를 SE(3) 행렬로 변환하는 static 유틸리티 함수 추가
  static Eigen::Matrix4d poseToMatrix(const geometry_msgs::msg::Pose& pose) {
    Eigen::Vector3d translation(pose.position.x, pose.position.y, pose.position.z);
    Eigen::Quaterniond rotation(pose.orientation.w, 
                                pose.orientation.x, 
                                pose.orientation.y, 
                                pose.orientation.z);
    
    Eigen::Matrix4d se3_matrix = Eigen::Matrix4d::Identity();
    se3_matrix.block<3,3>(0,0) = rotation.toRotationMatrix();
    se3_matrix.block<3,1>(0,3) = translation;

    return se3_matrix;
  }

  void set_max_deque_size(size_t n) { max_deque_size_ = (n == 0 ? 1 : n); }
  size_t max_deque_size() const { return max_deque_size_; }

  size_t size() const { return deque_.size(); }
  bool empty() const { return deque_.empty(); }

  void addCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &ros_cloud) {
    CloudPtr cloud(new CloudT());
    pcl::fromROSMsg(*ros_cloud, *cloud);

    if (deque_.size() >= max_deque_size_) deque_.pop_front();
    deque_.push_back(cloud);
  }

  void addObj(const geometry_msgs::msg::Point &point) {
    pcl::PointXYZ pt;
    pt.x = static_cast<float>(point.x);
    pt.y = static_cast<float>(point.y);
    pt.z = 0.0;

    if (obj_deque_.size() >= max_deque_size_) obj_deque_.pop_front();
    obj_deque_.push_back(pt);
  }

  std::vector<CloudConstPtr> snapshot() const {
    std::vector<CloudConstPtr> out;
    out.reserve(deque_.size());
    for (const auto &c : deque_) out.push_back(c);
    return out;
  }

  std::vector<PointXYZ> snapshot_obj() const {
    std::vector<PointXYZ> out;
    out.reserve(obj_deque_.size());
    for (const auto &c : obj_deque_) out.push_back(c);
    return out;
  }

private:
  size_t max_deque_size_;
  std::deque<CloudPtr> deque_;
  std::deque<PointXYZ> obj_deque_;
};