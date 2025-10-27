#pragma once

#include <deque>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>

class MapManager {
public:
  using Cloud = pcl::PointCloud<pcl::PointXYZI>;
  using CloudPtr = typename Cloud::Ptr;

  struct PointXYZ {
    float x{0.f}, y{0.f}, z{0.f};
  };

  explicit MapManager(size_t max_deque = 100)
  : max_deque_size_(max_deque) {}

  void set_max_deque_size(size_t n) { max_deque_size_ = n; trim_(); }
  size_t max_deque_size() const { return max_deque_size_; }

  static Eigen::Matrix4d poseToMatrix(const geometry_msgs::msg::Pose& p) {
    Eigen::Quaterniond q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = q.normalized().toRotationMatrix();
    T(0,3) = p.position.x;
    T(1,3) = p.position.y;
    T(2,3) = p.position.z;
    return T;
  }

  void addCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) {
    CloudPtr pcl_cloud(new Cloud());
    pcl::fromROSMsg(*msg, *pcl_cloud);
    deque_.push_back(pcl_cloud);
    trim_();
  }

  std::vector<CloudPtr> snapshot_cloud() const {
    return std::vector<CloudPtr>(deque_.begin(), deque_.end());
  }

  void addCloudWithPose(const sensor_msgs::msg::PointCloud2::SharedPtr& msg,
                        const geometry_msgs::msg::Pose& pose) {
    CloudPtr pcl_cloud(new Cloud());
    pcl::fromROSMsg(*msg, *pcl_cloud);
    deque_pair_.emplace_back(pcl_cloud, pose);
    trim_();
  }

  std::vector<std::pair<CloudPtr, geometry_msgs::msg::Pose>> snapshot_pairs() const {
    return std::vector<std::pair<CloudPtr, geometry_msgs::msg::Pose>>(
      deque_pair_.begin(), deque_pair_.end());
  }

  void addObj(const std::vector<geometry_msgs::msg::PointStamped>& p) {
    obj_deque_.push_back(p);
    trim_();
  }

  std::vector<std::vector<geometry_msgs::msg::PointStamped>> snapshot_obj() const {
    std::vector<std::vector<geometry_msgs::msg::PointStamped>> out;
    out.reserve(obj_deque_.size());
    for (const auto &c : obj_deque_) out.push_back(c);
    return out;
  }

  void addObjWithPose(const std::vector<geometry_msgs::msg::PointStamped>& p, const geometry_msgs::msg::Pose& pose) {
    obj_pair_deque_.emplace_back(p, pose);
    trim_();
  }

  std::vector<std::pair<std::vector<geometry_msgs::msg::PointStamped>, geometry_msgs::msg::Pose>> snapshot_obj_pairs() const {
    return std::vector<std::pair<std::vector<geometry_msgs::msg::PointStamped>, geometry_msgs::msg::Pose>>(
      obj_pair_deque_.begin(), obj_pair_deque_.end());
  }

  size_t size() const {
    size_t a = deque_.size();
    size_t b = deque_pair_.size();
    size_t c = obj_deque_.size();
    size_t d = obj_pair_deque_.size();
    return std::max(std::max(a,b), std::max(c,d));
  }

private:
  void trim_() {
    while (deque_.size() > max_deque_size_) deque_.pop_front();
    while (deque_pair_.size() > max_deque_size_) deque_pair_.pop_front();
    while (obj_deque_.size() > max_deque_size_) obj_deque_.pop_front();
    while (obj_pair_deque_.size() > max_deque_size_) obj_pair_deque_.pop_front();
  }

  size_t max_deque_size_{40};
  std::deque<CloudPtr> deque_;
  std::deque<std::vector<geometry_msgs::msg::PointStamped>> obj_deque_;
  std::deque<std::pair<CloudPtr, geometry_msgs::msg::Pose>> deque_pair_;
  std::deque<std::pair<std::vector<geometry_msgs::msg::PointStamped>, geometry_msgs::msg::Pose>> obj_pair_deque_;
};