#pragma once

#include <deque>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <optional>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>

/**
 * @brief Manages temporal collections of point clouds and detected objects with pose information
 * 
 * This class maintains sliding window buffers for:
 * - Point clouds (with and without pose)
 * - Detected objects (with and without pose)
 */
class MapManager {
public:
  using Cloud = pcl::PointCloud<pcl::PointXYZI>;
  using CloudPtr = typename Cloud::Ptr;
  using CloudConstPtr = typename Cloud::ConstPtr;
  
  using PointVector = std::vector<geometry_msgs::msg::PointStamped>;
  using CloudPosePair = std::pair<CloudPtr, geometry_msgs::msg::Pose>;
  using ObjectPosePair = std::pair<PointVector, geometry_msgs::msg::Pose>;

  /**
   * @brief Simple 3D point structure
   */
  struct PointXYZ {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
  };

  /**
   * @brief Construct a new Map Manager
   * @param max_deque Maximum number of elements to store in each buffer
   */
  explicit MapManager(size_t max_deque = 100) noexcept
    : max_deque_size_(max_deque) {}

  // Prevent copying, allow moving
  MapManager(const MapManager&) = delete;
  MapManager& operator=(const MapManager&) = delete;
  MapManager(MapManager&&) noexcept = default;
  MapManager& operator=(MapManager&&) noexcept = default;
  
  ~MapManager() = default;

  /**
   * @brief Set maximum buffer size and trim if necessary
   */
  void SetMaxDequeSize(size_t size) noexcept {
    max_deque_size_ = size;
    TrimBuffers();
  }
  
  [[nodiscard]] size_t GetMaxDequeSize() const noexcept { return max_deque_size_; }

  /**
   * @brief Convert ROS Pose message to 4x4 transformation matrix
   */
  [[nodiscard]] static Eigen::Matrix4d PoseToMatrix(const geometry_msgs::msg::Pose& pose) noexcept {
    const Eigen::Quaterniond q(
      pose.orientation.w,
      pose.orientation.x,
      pose.orientation.y,
      pose.orientation.z
    );
    
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    transform(0, 3) = pose.position.x;
    transform(1, 3) = pose.position.y;
    transform(2, 3) = pose.position.z;
    
    return transform;
  }

  /**
   * @brief Add point cloud without pose information
   */
  void AddCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) {
    if (!msg) return;
    
    CloudPtr pcl_cloud = std::make_shared<Cloud>();
    pcl::fromROSMsg(*msg, *pcl_cloud);
    cloud_deque_.push_back(std::move(pcl_cloud));
    
    TrimBuffers();
  }

  /**
   * @brief Get snapshot of all stored clouds
   */
  [[nodiscard]] std::vector<CloudPtr> GetCloudSnapshot() const {
    return {cloud_deque_.begin(), cloud_deque_.end()};
  }

  /**
   * @brief Add point cloud with associated pose
   */
  void AddCloudWithPose(
      const sensor_msgs::msg::PointCloud2::SharedPtr& msg,
      const geometry_msgs::msg::Pose& pose) {
    if (!msg) return;
    
    CloudPtr pcl_cloud = std::make_shared<Cloud>();
    pcl::fromROSMsg(*msg, *pcl_cloud);
    cloud_pose_deque_.emplace_back(std::move(pcl_cloud), pose);
    
    TrimBuffers();
  }

  /**
   * @brief Get snapshot of all cloud-pose pairs
   */
  [[nodiscard]] std::vector<CloudPosePair> GetCloudPoseSnapshot() const {
    return {cloud_pose_deque_.begin(), cloud_pose_deque_.end()};
  }

  /**
   * @brief Add detected objects without pose
   */
  void AddObjects(const PointVector& points) {
    object_deque_.push_back(points);
    TrimBuffers();
  }

  /**
   * @brief Get snapshot of all stored objects
   */
  [[nodiscard]] std::vector<PointVector> GetObjectSnapshot() const {
    return {object_deque_.begin(), object_deque_.end()};
  }

  /**
   * @brief Add detected objects with associated pose
   */
  void AddObjectsWithPose(
      const PointVector& points,
      const geometry_msgs::msg::Pose& pose) {
    object_pose_deque_.emplace_back(points, pose);
    TrimBuffers();
  }

  /**
   * @brief Get snapshot of all object-pose pairs
   */
  [[nodiscard]] std::vector<ObjectPosePair> GetObjectPoseSnapshot() const {
    return {object_pose_deque_.begin(), object_pose_deque_.end()};
  }

  /**
   * @brief Get maximum size across all buffers
   */
  [[nodiscard]] size_t Size() const noexcept {
    return std::max({
      cloud_deque_.size(),
      cloud_pose_deque_.size(),
      object_deque_.size(),
      object_pose_deque_.size()
    });
  }

  /**
   * @brief Check if all buffers are empty
   */
  [[nodiscard]] bool Empty() const noexcept {
    return cloud_deque_.empty() && 
           cloud_pose_deque_.empty() && 
           object_deque_.empty() && 
           object_pose_deque_.empty();
  }

  /**
   * @brief Clear all buffers
   */
  void Clear() noexcept {
    cloud_deque_.clear();
    cloud_pose_deque_.clear();
    object_deque_.clear();
    object_pose_deque_.clear();
  }

private:
  /**
   * @brief Trim all buffers to maximum size
   */
  void TrimBuffers() noexcept {
    while (cloud_deque_.size() > max_deque_size_) {
      cloud_deque_.pop_front();
    }
    while (cloud_pose_deque_.size() > max_deque_size_) {
      cloud_pose_deque_.pop_front();
    }
    while (object_deque_.size() > max_deque_size_) {
      object_deque_.pop_front();
    }
    while (object_pose_deque_.size() > max_deque_size_) {
      object_pose_deque_.pop_front();
    }
  }

  size_t max_deque_size_{40};
  
  std::deque<CloudPtr> cloud_deque_;
  std::deque<PointVector> object_deque_;
  std::deque<CloudPosePair> cloud_pose_deque_;
  std::deque<ObjectPosePair> object_pose_deque_;
};