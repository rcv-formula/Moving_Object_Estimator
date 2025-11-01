#pragma once
/**
 * @file map_manager_pair.hpp
 * @brief 스캔/오브젝트/포즈 히스토리를 보관하는 가벼운 매니저 (헤더 온리)
 *
 * 제공 기능:
 *  - addCloudWithPose(PointCloud2, Pose) / snapshot_pairs()
 *  - addObjWithPose(vector<PointStamped>, Pose) / snapshot_obj_pairs()
 *  - addScanObjWithPose(PointCloud2, vector<PointStamped>, Pose) / snapshot_triplets()
 *  - poseToMatrix(Pose) 유틸 (static)
 *
 * 주의: 단일 스레드 사용을 전제로 함(멀티스레드는 별도 mutex 필요).
 */

#include <deque>
#include <tuple>
#include <vector>
#include <memory>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class MapManager {
public:
  using Cloud      = pcl::PointCloud<pcl::PointXYZI>;
  using CloudPtr   = std::shared_ptr<Cloud>;

  // (scan, pose)
  using CloudPosePair = std::pair<CloudPtr, geometry_msgs::msg::Pose>;
  // (obstacles, pose)
  using ObjPosePair   = std::pair<std::vector<geometry_msgs::msg::PointStamped>, geometry_msgs::msg::Pose>;
  // (scan, obstacles, pose)
  using ScanObjPoseTriplet =
    std::tuple<CloudPtr, std::vector<geometry_msgs::msg::PointStamped>, geometry_msgs::msg::Pose>;

  explicit MapManager(std::size_t max_deque_size = 10)
  : max_deque_size_(max_deque_size) {}

  // === API: scan + pose 저장 ===
  void addCloudWithPose(const sensor_msgs::msg::PointCloud2::SharedPtr& scan_msg,
                        const geometry_msgs::msg::Pose& pose)
  {
    CloudPtr pcl_cloud(new Cloud());
    pcl::fromROSMsg(*scan_msg, *pcl_cloud);
    deque_pair_.emplace_back(std::move(pcl_cloud), pose);
    trim_();
  }

  // 복사본 반환
  std::vector<CloudPosePair> snapshot_pairs() const {
    return std::vector<CloudPosePair>(deque_pair_.begin(), deque_pair_.end());
  }

  // === API: obstacles + pose 저장 ===
  void addObjWithPose(const std::vector<geometry_msgs::msg::PointStamped>& obstacles,
                      const geometry_msgs::msg::Pose& pose)
  {
    obj_pair_deque_.emplace_back(obstacles, pose);
    trim_();
  }

  // 복사본 반환
  std::vector<ObjPosePair> snapshot_obj_pairs() const {
    return std::vector<ObjPosePair>(obj_pair_deque_.begin(), obj_pair_deque_.end());
  }

  // === API: scan + obstacles + pose (3종 세트) 저장 ===
  void addScanObjWithPose(const sensor_msgs::msg::PointCloud2::SharedPtr& scan_msg,
                          const std::vector<geometry_msgs::msg::PointStamped>& obstacles,
                          const geometry_msgs::msg::Pose& pose)
  {
    CloudPtr pcl_cloud(new Cloud());
    pcl::fromROSMsg(*scan_msg, *pcl_cloud);
    triplet_deque_.emplace_back(std::move(pcl_cloud), obstacles, pose);
    trim_();
  }

  // 복사본 반환
  std::vector<ScanObjPoseTriplet> snapshot_triplets() const {
    return std::vector<ScanObjPoseTriplet>(triplet_deque_.begin(), triplet_deque_.end());
  }

  // === 유틸: Pose → 4x4 변환행렬 ===
  static Eigen::Matrix4d poseToMatrix(const geometry_msgs::msg::Pose& pose)
  {
    Eigen::Quaterniond q(pose.orientation.w,
                         pose.orientation.x,
                         pose.orientation.y,
                         pose.orientation.z);
    q.normalize();

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = q.toRotationMatrix();
    T(0,3) = pose.position.x;
    T(1,3) = pose.position.y;
    T(2,3) = pose.position.z;
    return T;
  }

  void set_max_deque_size(std::size_t n) { max_deque_size_ = n; trim_(); }

private:
  void trim_() {
    while (deque_pair_.size()     > max_deque_size_) deque_pair_.pop_front();
    while (obj_pair_deque_.size() > max_deque_size_) obj_pair_deque_.pop_front();
    while (triplet_deque_.size()  > max_deque_size_) triplet_deque_.pop_front();
  }

  std::size_t max_deque_size_;

  // 내부 버퍼들
  std::deque<CloudPosePair>        deque_pair_;        // scan + pose
  std::deque<ObjPosePair>          obj_pair_deque_;    // obstacles + pose
  std::deque<ScanObjPoseTriplet>   triplet_deque_;     // scan + obstacles + pose
};
