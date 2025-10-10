#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "map_manager.hpp"

class DynamicObjectDetector {
public:
  using CloudT = pcl::PointCloud<pcl::PointXYZ>;
  using CloudPtr = CloudT::Ptr;
  using CloudConstPtr = CloudT::ConstPtr;

  DynamicObjectDetector(int knn_k = 5, double wall_dist_thresh = 0.06, double voxel_leaf = 0.03)
  : knn_k_(knn_k), wall_dist_thresh_(wall_dist_thresh), voxel_leaf_(voxel_leaf),
    accum_(new CloudT()), static_obj_(new CloudT()), tree_ready_(false), tree_ready_obj_(false) {}

  void set_knn_k(int k) { knn_k_ = std::max(1, k); }
  void set_wall_dist_thresh(double d) { wall_dist_thresh_ = d; }
  void set_voxel_leaf(double v) { voxel_leaf_ = std::max(1e-3, v); }

  int knn_k() const { return knn_k_; }
  double wall_dist_thresh() const { return wall_dist_thresh_; }
  double voxel_leaf() const { return voxel_leaf_; }

  // MapManager의 deque 스냅샷을 받아 누적→다운샘플→KDTree 재빌드
  void rebuild(const std::vector<CloudConstPtr> &deque_snapshot) {
    CloudPtr merged(new CloudT());
    size_t total = 0;
    for (auto &c : deque_snapshot) total += c->size();
    merged->reserve(total);
    for (auto &c : deque_snapshot) *merged += *c;

    CloudPtr ds(new CloudT());
    if (!merged->empty()) {
      pcl::VoxelGrid<pcl::PointXYZ> vg;
      vg.setInputCloud(merged);
      vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
      vg.filter(*ds);
    }

    accum_.swap(ds);
    if (accum_ && !accum_->empty()) {
      kdtree_.setInputCloud(accum_);
      tree_ready_ = true;
    } else {
      tree_ready_ = false;
    }
  }

  void rebuild_obj(const std::vector<pcl::PointXYZ> &deque_snapshot) {
    CloudPtr merged(new CloudT());
    size_t total = deque_snapshot.size();

    merged->reserve(total);
    for (auto &c : deque_snapshot) merged->push_back(c);

    CloudPtr ds(new CloudT());
    if (!merged->empty()) {
      pcl::VoxelGrid<pcl::PointXYZ> vg;
      vg.setInputCloud(merged);
      vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
      vg.filter(*ds);
    }

    static_obj_.swap(ds);
    if (static_obj_ && !static_obj_->empty()) {
      kdtree_obj_.setInputCloud(static_obj_);
      tree_ready_obj_ = true;
    } else {
      tree_ready_obj_ = false;
    }
  }

  bool ready() const { return tree_ready_ && accum_ && !accum_->empty(); }
  bool ready_obj() const { return tree_ready_obj_ && static_obj_ && !static_obj_->empty(); }

  // 입력 포인트가 벽에 속하는가?
  // true  -> wall (드롭)
  // false -> dynamic(통과)
  bool isWall(const geometry_msgs::msg::Point &p) const {
    if (!ready()) return false;

    pcl::PointXYZ q;
    q.x = static_cast<float>(p.x);
    q.y = static_cast<float>(p.y);
    q.z = static_cast<float>(p.z);

    std::vector<int> knn_idx(knn_k_);
    std::vector<float> knn_sq(knn_k_);
    int found_knn = kdtree_.nearestKSearch(q, knn_k_, knn_idx, knn_sq);

    bool knn_check = false;
    if (found_knn > 0) {
      float min_sqr = *std::min_element(knn_sq.begin(), knn_sq.begin() + found_knn);
      double dmin = std::sqrt(min_sqr);
      knn_check = (dmin < wall_dist_thresh_);
    }

    std::vector<int> rad_idx;
    std::vector<float> rad_sq;
    int found_rad = kdtree_.radiusSearch(q, wall_dist_thresh_, rad_idx, rad_sq);

    bool radius_check = (found_rad > 0);

    return knn_check || radius_check;
  }

  bool isStatic(const geometry_msgs::msg::Point &p) const {
    if (!ready_obj()) return false;

    pcl::PointXYZ q;
    q.x = static_cast<float>(p.x);
    q.y = static_cast<float>(p.y);
    q.z = static_cast<float>(p.z);

    std::vector<int> knn_idx(knn_k_);
    std::vector<float> knn_sq(knn_k_);
    int found_knn = kdtree_obj_.nearestKSearch(q, knn_k_, knn_idx, knn_sq);

    bool knn_check = false;
    if (found_knn > 0) {
      float min_sqr = *std::min_element(knn_sq.begin(), knn_sq.begin() + found_knn);
      double dmin = std::sqrt(min_sqr);
      knn_check = (dmin < wall_dist_thresh_);
    }

    std::vector<int> rad_idx;
    std::vector<float> rad_sq;
    int found_rad = kdtree_obj_.radiusSearch(q, wall_dist_thresh_, rad_idx, rad_sq);

    bool radius_check = (found_rad > 0);

    return knn_check || radius_check;
  }

  void addStatic(const geometry_msgs::msg::Point &p) {
    pcl::PointXYZ pt;
    pt.x = static_cast<float>(p.x);
    pt.y = static_cast<float>(p.y);
    pt.z = 0.0;

    static_vec_.push_back(pt);
  }

  std::vector<pcl::PointXYZ> get_static_vec() const {
    return static_vec_;
  }

  size_t get_static_vec_size() const {
    return static_vec_.size();
  }

private:
  int knn_k_;
  double wall_dist_thresh_;
  double voxel_leaf_;

  CloudPtr accum_;
  CloudPtr static_obj_;
  std::vector<pcl::PointXYZ> static_vec_;

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_;
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_obj_;
  bool tree_ready_;
  bool tree_ready_obj_;
};
