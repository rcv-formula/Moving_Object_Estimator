#pragma once
/**
 * @file dynamic_obstacle_detector.hpp
 * @brief 로컬-맵 정렬된 과거 프레임을 관리/시각화하고, footprint 기반 동적/정적 판정을 수행하는 보조 클래스
 *
 * 제공 기능:
 *  - transformObjHistoryToCurrentFrames(obj_pairs, current_pose):
 *      과거 오브젝트 점들을 현재 프레임(=현재 포즈 기준 로컬 좌표계)로 변환하여 프레임별 점 리스트 반환
 *  - publishAlignedFramesMarkers(aligned_frames, frame_id, stamp, pub, point_scale, alpha):
 *      정렬된 과거 오브젝트 프레임들을 MarkerArray로 시각화
 *  - classifyDynamicByFootprint(target, aligned_frames, static_thresh, min_history_frames, match_gate, exclude_current, dynamic_thresh, out_footprint, out_span):
 *      현재 타깃과 과거 프레임별 nearest center의 이동량 분포로 동적/정적을 분류.
 *      유효 매칭 프레임이 부족하거나 판단 경계에 있으면 UNKNOWN.
 *  - visualizeFootprint(footprint, label, frame_id, id, stamp, pub):
 *      footprint를 MarkerArray로 시각화
 */

#include <functional>
#include <vector>
#include <cmath>
#include <limits>
#include <memory>
#include <algorithm>
#include <unordered_set>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>

class DynamicObjectDetector {
public:
  // 라벨 정의: UNKNOWN=0, STATIC=1, DYNAMIC=2
  enum Label { UNKNOWN=0, STATIC=1, DYNAMIC=2 };

  DynamicObjectDetector() = default;

    inline bool loadTrackCsvPCL(const std::string& csv_path, bool has_header=true)
    {
      track_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
      kdtree_pcl_.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());

      std::ifstream ifs(csv_path);
      if (!ifs.is_open()) {
        RCLCPP_DEBUG(rclcpp::get_logger("DynamicObjectDetector"),
                    "Failed to open CSV (PCL): %s", csv_path.c_str());
        track_cloud_.reset();
        kdtree_pcl_.reset();
        return false;
      }

      std::string line;
      if (has_header) std::getline(ifs, line); // 헤더 스킵 (열 순서 고정)

      size_t line_no = has_header ? 2 : 1;
      while (std::getline(ifs, line)) {
        if (line.empty()) { ++line_no; continue; }
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cols;
        while (std::getline(ss, cell, ',')) cols.push_back(cell);

        auto to_double = [](const std::string& s, double& out)->bool{
          try { out = std::stod(s); return true; }
          catch(...) { out = std::numeric_limits<double>::quiet_NaN(); return false; }
        };

        double x=0.0, y=0.0, w=std::numeric_limits<double>::quiet_NaN();
        bool ok = true;
        ok &= to_double(cols[0], x);
        ok &= to_double(cols[1], y);
        ok &= to_double(cols[2], w);

        if (!ok || std::isnan(x) || std::isnan(y) || std::isnan(w)) {
          RCLCPP_DEBUG(rclcpp::get_logger("DynamicObjectDetector"),
                      "CSV line %zu parse failed (x/y/w). Skip.", line_no);
          ++line_no; continue;
        }

        pcl::PointXYZI pt;
        pt.x = static_cast<float>(x);
        pt.y = static_cast<float>(y);
        pt.z = 0.0f;                           // 2D 트랙 가정
        pt.intensity = static_cast<float>(w);   // intensity에 '벽까지 거리' 저장
        track_cloud_->push_back(pt);

        ++line_no;
      }

      kdtree_pcl_->setInputCloud(track_cloud_);
      RCLCPP_DEBUG(rclcpp::get_logger("DynamicObjectDetector"),
                  "Loaded %zu center points (PCL) and built KD-Tree.",
                  track_cloud_->size());
      return true;
    }
  
  inline visualization_msgs::msg::Marker makeWallLineMarker() const
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map";
    m.header.stamp = rclcpp::Clock().now();
    m.ns = "wall_line";
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::LINE_STRIP;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = 0.05;
  
    m.color.r = 1.0;
    m.color.g = 0.6;
    m.color.b = 0.0;
    m.color.a = 1.0;
  
    if (!track_cloud_ || track_cloud_->empty())
      return m;
    std::vector<geometry_msgs::msg::Point> tmp_right;

    for (size_t i = 0; i < track_cloud_->size(); ++i) {
      const auto& pt = (*track_cloud_)[i];
    
      // 이전 점과 다음 점으로부터 진행방향 추정
      Eigen::Vector2d dir(1.0, 0.0);
      if (i + 1 < track_cloud_->size()) {
        const auto& next = (*track_cloud_)[i + 1];
        dir << next.x - pt.x, next.y - pt.y;
      } else if (i > 0) {
        const auto& prev = (*track_cloud_)[i - 1];
        dir << pt.x - prev.x, pt.y - prev.y;
      }
      if (dir.norm() > 1e-6)
        dir.normalize();
    
      // 법선 방향 (좌우)
      Eigen::Vector2d normal(-dir.y(), dir.x());
    
      // 벽 거리 (intensity)
      double w = static_cast<double>(pt.intensity) - 0.2;
    
      // 좌우 점 생성
      geometry_msgs::msg::Point left, right;
      left.x  = pt.x + normal.x() * w;
      left.y  = pt.y + normal.y() * w;
      left.z  = pt.z;
    
      right.x = pt.x - normal.x() * w;
      right.y = pt.y - normal.y() * w;
      right.z = pt.z;
    
      // 왼쪽-오른쪽 순으로 추가 (시각적으로 띠 형태)
      m.points.push_back(left);
      tmp_right.emplace_back(right);
    }
    for(auto r : tmp_right){
      m.points.push_back(r);
    }
    
    return m;
  }

  inline bool isObstacleWithinWallPCL(const geometry_msgs::msg::Point& obstacle_in_map) const
  {
    if (!track_cloud_ || !kdtree_pcl_ || track_cloud_->empty()) {
      RCLCPP_DEBUG(rclcpp::get_logger("DynamicObjectDetector"),
                  "Track CSV (PCL) not loaded. Treat obstacle as valid.");
      return true; // 인덱스가 없으면 무효 판정 불가 → 유효 처리
    }

    pcl::PointXYZI query;
    query.x = static_cast<float>(obstacle_in_map.x);
    query.y = static_cast<float>(obstacle_in_map.y);
    query.z = 0.0f;

    std::vector<int> knn_idx(1);
    std::vector<float> knn_d2(1);

    const int found = kdtree_pcl_->nearestKSearch(query, 1, knn_idx, knn_d2);
    if (found <= 0 || knn_idx[0] < 0 || static_cast<size_t>(knn_idx[0]) >= track_cloud_->size()) {
      RCLCPP_DEBUG(rclcpp::get_logger("DynamicObjectDetector"),
                  "PCL KD-Tree query failed. Treat obstacle as valid.");
      return true;
    }

    const pcl::PointXYZI& nn = (*track_cloud_)[knn_idx[0]];
    const double d_center_obs = std::sqrt(static_cast<double>(knn_d2[0]));
    const double wall_dist    = static_cast<double>(nn.intensity);

    const bool valid = (d_center_obs <= wall_dist - 0.5); // 여유 10cm

    RCLCPP_DEBUG(rclcpp::get_logger("DynamicObjectDetector"),
                 "[PCL ObstacleCheck] nn=(%.3f,%.3f) obs=(%.3f,%.3f) d=%.3f wall=%.3f -> %s",
                 nn.x, nn.y, query.x, query.y, d_center_obs, wall_dist,
                 valid ? "VALID" : "INVALID");
    return valid;
  }

  // === 과거 obj (map 좌표) → 현재 프레임(현재 포즈 기준 로컬 좌표)로 정렬 ===
  std::vector<std::vector<geometry_msgs::msg::Point>>
  transformObjHistoryToCurrentFrames(
      const std::vector<std::pair<std::vector<geometry_msgs::msg::PointStamped>, geometry_msgs::msg::Pose>>& pairs_obj,
      const geometry_msgs::msg::Pose& current_pose_map) const
  {
    std::vector<std::vector<geometry_msgs::msg::Point>> out;
    out.reserve(pairs_obj.size());

    const Eigen::Matrix4d T_map_curr = poseToT(current_pose_map); // map->curr
    const Eigen::Matrix4d T_curr_map = T_map_curr.inverse();      // curr->map

    for (const auto& pair : pairs_obj) {
      const auto& pts  = pair.first;  // map 좌표에 있는 과거 점들

      std::vector<geometry_msgs::msg::Point> transformed;
      transformed.reserve(pts.size());

      for (const auto& ps : pts) {
        Eigen::Vector4d pw(ps.point.x, ps.point.y, ps.point.z, 1.0); // map
        Eigen::Vector4d pc = T_curr_map * pw;                        // curr(frame)
        geometry_msgs::msg::Point p; p.x = pc.x(); p.y = pc.y(); p.z = pc.z();
        transformed.push_back(p);
      }
      out.push_back(std::move(transformed));
    }
    return out;
  }

  // === 정렬된 과거 프레임 점들을 RViz Marker로 시각화 ===
  void publishAlignedFramesMarkers(
      const std::vector<std::vector<geometry_msgs::msg::Point>>& aligned_frames,
      const std::string& frame_id,
      const builtin_interfaces::msg::Time& stamp,
      const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& pub,
      double point_scale = 0.06,
      double alpha = 0.15) const
  {
    if (!pub) return;

    visualization_msgs::msg::MarkerArray arr;

    // DELETEALL
    {
      visualization_msgs::msg::Marker del;
      del.header.frame_id = frame_id;
      del.header.stamp    = stamp;
      del.ns   = "aligned_obj_history";
      del.id   = 0;
      del.action = visualization_msgs::msg::Marker::DELETEALL;
      arr.markers.push_back(del);
    }

    // SPHERE_LIST per frame
    for (size_t i = 0; i < aligned_frames.size(); ++i) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = frame_id;
      m.header.stamp    = stamp;
      m.ns   = "aligned_obj_history";
      m.id   = static_cast<int>(i + 1);
      m.type = visualization_msgs::msg::Marker::SPHERE_LIST;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.scale.x = point_scale; m.scale.y = point_scale; m.scale.z = point_scale;

      float r, g, b;
      hsvToRgb(static_cast<float>((i % 12) / 12.0), 0.8f, 0.9f, r, g, b);
      m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = static_cast<float>(alpha);

      for (const auto& p : aligned_frames[i]) m.points.push_back(p);
      arr.markers.push_back(m);
    }

    pub->publish(arr);
  }

  /**
   * @brief 현재 center와 과거 프레임별 nearest center를 비교해 정적/동적을 분류.
   *
   * aligned_frames는 현재 프레임을 포함하지 않는 과거 프레임 목록이다.
   * 각 과거 프레임에서 target과 가장 가까운 center 1개만 사용하고, match_gate 밖이면 버린다.
   * 유효 매칭 수가 부족하면 UNKNOWN, median 이동량이 작으면 STATIC,
   * 여러 프레임에서 충분히 멀리 움직였으면 DYNAMIC이다.
   */
  int classifyDynamicByFootprint(
      const geometry_msgs::msg::Point& target_in_current,
      const std::vector<std::vector<geometry_msgs::msg::Point>>& aligned_frames,
      double static_thresh,
      int min_history_frames,
      double match_gate,
      bool   /*exclude_current*/,
      double dynamic_thresh,
      std::vector<geometry_msgs::msg::Point>* out_footprint,
      double* out_span) const
  {
    if (out_footprint) out_footprint->clear();
    if (out_span) *out_span = 0.0;

    if (aligned_frames.empty()) return UNKNOWN;

    std::vector<geometry_msgs::msg::Point> nearest_points;
    std::vector<double> dist_vec;
    nearest_points.reserve(aligned_frames.size());
    dist_vec.reserve(aligned_frames.size());

    const double gate2 = match_gate * match_gate;
    for (const auto& frame : aligned_frames) {
      bool found = false;
      geometry_msgs::msg::Point nearest;
      double best_d2 = std::numeric_limits<double>::infinity();

      for (const auto& p : frame) {
        const double dx = p.x - target_in_current.x;
        const double dy = p.y - target_in_current.y;
        const double dz = p.z - target_in_current.z;
        const double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best_d2) {
          best_d2 = d2;
          nearest = p;
          found = true;
        }
      }

      if (found && best_d2 <= gate2) {
        nearest_points.push_back(nearest);
        dist_vec.push_back(std::sqrt(best_d2));
      }
    }

    if (out_span) {
      if (nearest_points.empty()) *out_span = 0.0;
      else {
        double xmin=+1e9, xmax=-1e9;
        for (const auto& p : nearest_points) {
          xmin = std::min(xmin, p.x);
          xmax = std::max(xmax, p.x);
        }
        *out_span = (xmax - xmin);
      }
    }

    if (out_footprint) {
      out_footprint->reserve(nearest_points.size());
      for (const auto& p : nearest_points) out_footprint->push_back(p);
    }

    if (static_cast<int>(dist_vec.size()) < std::max(1, min_history_frames)) {
      return UNKNOWN;
    }

    std::sort(dist_vec.begin(), dist_vec.end());
    const double median = dist_vec[dist_vec.size() / 2];
    const int dynamic_hits = static_cast<int>(std::count_if(
      dist_vec.begin(), dist_vec.end(),
      [dynamic_thresh](double d) { return d >= dynamic_thresh; }));

    if (dynamic_hits >= std::max(2, min_history_frames)) return DYNAMIC;
    if (median >= dynamic_thresh) return DYNAMIC;
    if (median <= static_thresh && dynamic_hits == 0) return STATIC;

    return UNKNOWN;
  }

  // === footprint 시각화 ===
  void visualizeFootprint(
      const std::vector<geometry_msgs::msg::Point>& footprint,
      int dyn_label,                         // 0=unknown, 1=static, 2=dynamic
      const std::string& frame_id,
      int id_base,
      const builtin_interfaces::msg::Time& stamp,
      const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& pub) const
  {
    if (!pub) return;

    visualization_msgs::msg::MarkerArray arr;

    // 포인트 구름
    {
      visualization_msgs::msg::Marker pts;
      pts.header.frame_id = frame_id;
      pts.header.stamp    = stamp;
      pts.ns   = "footprint_points";
      pts.id   = id_base;
      pts.type = visualization_msgs::msg::Marker::SPHERE_LIST;
      pts.action = visualization_msgs::msg::Marker::ADD;
      pts.scale.x = 0.06; pts.scale.y = 0.06; pts.scale.z = 0.06;

      if (dyn_label == DYNAMIC) {         // 빨강
        pts.color.r = 1.0f; pts.color.g = 0.0f; pts.color.b = 0.0f; pts.color.a = 0.95f;
      } else if (dyn_label == STATIC) {   // 파랑
        pts.color.r = 0.1f; pts.color.g = 0.4f; pts.color.b = 1.0f; pts.color.a = 0.95f;
      } else {                            // UNKNOWN = 노랑
        pts.color.r = 1.0f; pts.color.g = 0.9f; pts.color.b = 0.1f; pts.color.a = 0.95f;
      }

      for (const auto& p : footprint) pts.points.push_back(p);
      pts.lifetime = rclcpp::Duration::from_seconds(0.25);
      arr.markers.push_back(pts);
    }

    // 외곽선(Convex Hull 유사 라이트 버전: x 정렬 후 polyline)
    if (footprint.size() >= 3) {
      std::vector<geometry_msgs::msg::Point> hull = polylineByX(footprint);

      visualization_msgs::msg::Marker line;
      line.header.frame_id = frame_id;
      line.header.stamp    = stamp;
      line.ns   = "footprint_outline";
      line.id   = id_base + 10000;
      line.type = visualization_msgs::msg::Marker::LINE_STRIP;
      line.action = visualization_msgs::msg::Marker::ADD;
      line.scale.x = 0.025;

      // UNKNOWN은 회백색, 나머지는 흰색
      if (dyn_label == UNKNOWN) { line.color.r = 0.8f; line.color.g = 0.8f; line.color.b = 0.8f; line.color.a = 0.8f; }
      else                      { line.color.r = 0.95f; line.color.g = 0.95f; line.color.b = 0.95f; line.color.a = 0.8f; }

      for (const auto& p : hull) line.points.push_back(p);
      line.points.push_back(hull.front()); // 닫기
      line.lifetime = rclcpp::Duration::from_seconds(0.25);
      arr.markers.push_back(line);
    }

    pub->publish(arr);
  }

private:
  
pcl::PointCloud<pcl::PointXYZI>::Ptr track_cloud_{nullptr};
  std::unique_ptr<pcl::KdTreeFLANN<pcl::PointXYZI>> kdtree_pcl_{nullptr};

  static Eigen::Matrix4d poseToT(const geometry_msgs::msg::Pose &pose)
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

  static void hsvToRgb(float h, float s, float v, float& r, float& g, float& b)
  {
    float i = std::floor(h * 6.0f);
    float f = h * 6.0f - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);
    switch (static_cast<int>(i) % 6) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      case 5: r = v; g = p; b = q; break;
    }
  }

  static double hypot3(const geometry_msgs::msg::Point& a, const geometry_msgs::msg::Point& b)
  {
    const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
  }

  static std::vector<geometry_msgs::msg::Point>
  polylineByX(const std::vector<geometry_msgs::msg::Point>& pts)
  {
    std::vector<geometry_msgs::msg::Point> v = pts;
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b){ return a.x < b.x; });
    if (v.size() < 3) return v;

    // 상단 껍질
    std::vector<geometry_msgs::msg::Point> up;
    for (const auto& p : v) {
      while (up.size() >= 2 && cross(up[up.size()-2], up.back(), p) <= 0.0) up.pop_back();
      up.push_back(p);
    }
    // 하단 껍질
    std::vector<geometry_msgs::msg::Point> lo;
    for (int i = static_cast<int>(v.size()) - 1; i >= 0; --i) {
      const auto& p = v[i];
      while (lo.size() >= 2 && cross(lo[lo.size()-2], lo.back(), p) <= 0.0) lo.pop_back();
      lo.push_back(p);
    }
    // 병합(마지막 중복 제거)
    up.pop_back();
    lo.pop_back();
    up.insert(up.end(), lo.begin(), lo.end());
    return up;
  }

  static double cross(const geometry_msgs::msg::Point& a,
                      const geometry_msgs::msg::Point& b,
                      const geometry_msgs::msg::Point& c)
  {
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
  }
};
